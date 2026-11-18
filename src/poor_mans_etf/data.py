"""Loading the stock universe: constituents, prices and share counts, with an on-disk cache.

The cache in ``data/`` is committed to the repo so results are reproducible without
hitting Yahoo Finance. ``fetch`` refreshes it.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

CONSTITUENTS_FILE = "sp500_constituents.csv"
ADJ_CLOSE_FILE = "adj_close.csv.gz"
CLOSE_FILE = "close.csv.gz"
SHARES_FILE = "shares.csv"

BENCHMARK = "SPY"
START_DATE = dt.date(2023, 6, 29)
END_DATE = dt.date(2026, 6, 29)


def to_yahoo_ticker(symbol: str) -> str:
    """Yahoo spells share classes with a dash: ``BRK.B`` -> ``BRK-B``."""
    return symbol.replace(".", "-")


def load_constituents(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """S&P 500 constituents indexed by Yahoo ticker, one share class per company.

    Dual-class companies (GOOGL/GOOG, FOXA/FOX, NWSA/NWS) share a CIK. Yahoo reports the
    whole company's market cap for each class, so keeping both would double count them.
    """
    df = pd.read_csv(data_dir / CONSTITUENTS_FILE)
    df["ticker"] = df["symbol"].map(to_yahoo_ticker)
    dupes = df[df.duplicated("cik")]
    if not dupes.empty:
        log.info("Dropping secondary share classes: %s", ", ".join(dupes["ticker"]))
    df = df.drop_duplicates("cik", keep="first")
    return df.set_index("ticker")[["name", "sector"]]


# ---------------------------------------------------------------- downloading


def download_prices(
    tickers: list[str], start: dt.date, end: dt.date, retries: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily (adjusted close, close) for ``tickers``. Retries tickers that come back empty."""
    import yfinance as yf

    adj_close, close = pd.DataFrame(), pd.DataFrame()
    missing = list(tickers)
    for attempt in range(1, retries + 1):
        raw = yf.download(missing, start=start, end=end, auto_adjust=False, progress=False)
        got = raw["Adj Close"].dropna(axis=1, how="all").columns
        adj_close = pd.concat([adj_close, raw["Adj Close"][got]], axis=1)
        close = pd.concat([close, raw["Close"][got]], axis=1)
        missing = [t for t in missing if t not in got]
        if not missing:
            break
        log.warning("Attempt %d: no data for %d tickers, retrying", attempt, len(missing))
    if missing:
        log.warning("Giving up on: %s", ", ".join(missing))
    order = [t for t in tickers if t in adj_close.columns]
    return adj_close[order].sort_index(), close[order].sort_index()


def download_shares(tickers: list[str], retries: int = 3) -> pd.Series:
    """Shares outstanding implied by Yahoo's market cap (``marketCap / price``).

    Storing shares rather than market cap lets us value each company on any historical
    date as ``shares * close``, instead of using today's market cap for a past portfolio.
    """
    import yfinance as yf

    shares = {}
    for attempt in range(1, retries + 1):
        for ticker in [t for t in tickers if t not in shares]:
            try:
                info = yf.Ticker(ticker).info
                price = info.get("currentPrice") or info.get("regularMarketPrice")
                shares[ticker] = info["marketCap"] / price
            except Exception as exc:  # yfinance raises a zoo of exception types
                log.warning("Attempt %d: no market cap for %s: %s", attempt, ticker, exc)
    return pd.Series(shares, name="shares")


def fetch(
    start: dt.date = START_DATE,
    end: dt.date = END_DATE,
    benchmark: str = BENCHMARK,
    data_dir: Path = DATA_DIR,
) -> None:
    """Download prices and share counts for the universe and benchmark into ``data_dir``."""
    tickers = list(load_constituents(data_dir).index)
    adj_close, close = download_prices([*tickers, benchmark], start, end)
    shares = download_shares(tickers)
    adj_close.to_csv(data_dir / ADJ_CLOSE_FILE, float_format="%.6g")
    close.to_csv(data_dir / CLOSE_FILE, float_format="%.6g")
    shares.to_csv(data_dir / SHARES_FILE, index_label="ticker", float_format="%.6g")
    log.info("Saved %d tickers x %d days to %s", adj_close.shape[1], len(adj_close), data_dir)


# ------------------------------------------------------------------- universe


@dataclass
class Universe:
    """Everything the optimiser needs about the investable stocks and the benchmark."""

    prices: pd.DataFrame  # adjusted close, one column per stock
    close: pd.DataFrame  # unadjusted close, for market caps
    benchmark: pd.Series  # adjusted close of the benchmark
    shares: pd.Series  # shares outstanding
    sectors: pd.Series
    names: pd.Series

    @property
    def tickers(self) -> list[str]:
        return list(self.prices.columns)

    def __len__(self) -> int:
        return self.prices.shape[1]

    def between(self, start: dt.date | None = None, end: dt.date | None = None) -> Universe:
        """The same universe restricted to dates in ``[start, end]``."""
        dates = slice(pd.Timestamp(start) if start else None, pd.Timestamp(end) if end else None)
        return Universe(
            prices=self.prices.loc[dates],
            close=self.close.loc[dates],
            benchmark=self.benchmark.loc[dates],
            shares=self.shares,
            sectors=self.sectors,
            names=self.names,
        )

    def market_caps(self) -> pd.Series:
        """Market caps on the last date of the window (no peeking past it)."""
        return self.shares * self.close.iloc[-1]

    def returns(self) -> tuple[pd.DataFrame, pd.Series]:
        """Daily simple returns of the stocks and the benchmark."""
        return (
            self.prices.pct_change(fill_method=None).iloc[1:],
            self.benchmark.pct_change(fill_method=None).iloc[1:],
        )


def load_universe(data_dir: Path = DATA_DIR, benchmark: str = BENCHMARK) -> Universe:
    """Load the cached data, keeping only stocks with a complete price history.

    Dropping incomplete *stocks* (late IPOs and spin-offs) rather than incomplete *days*
    keeps the full history: a single stock listed last month would otherwise truncate the
    covariance estimate for everyone.
    """
    constituents = load_constituents(data_dir)
    adj_close = pd.read_csv(data_dir / ADJ_CLOSE_FILE, index_col=0, parse_dates=True)
    close = pd.read_csv(data_dir / CLOSE_FILE, index_col=0, parse_dates=True)
    shares = pd.read_csv(data_dir / SHARES_FILE, index_col=0)["shares"]

    adj_close = adj_close[adj_close[benchmark].notna()]
    bench = adj_close.pop(benchmark)
    close = close.loc[adj_close.index].drop(columns=benchmark)

    complete = adj_close.columns[adj_close.notna().all() & close.notna().all()]
    keep = [t for t in constituents.index if t in complete and t in shares.index]
    dropped = sorted(set(constituents.index) - set(keep))
    if dropped:
        log.info("Dropping %d tickers without full history: %s", len(dropped), ", ".join(dropped))

    return Universe(
        prices=adj_close[keep],
        close=close[keep],
        benchmark=bench,
        shares=shares[keep],
        sectors=constituents.loc[keep, "sector"],
        names=constituents.loc[keep, "name"],
    )
