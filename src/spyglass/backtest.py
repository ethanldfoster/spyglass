"""Out-of-sample evaluation: build portfolios on a training window, then buy them at the
split date and hold them through a test window the optimiser never saw."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .annealing import AnnealResult, anneal
from .data import Universe
from .tracking import (
    TRADING_DAYS,
    annualise,
    augmented_covariance,
    optimal_weights,
    top_k_cap_weights,
    tracking_variance,
)

SPLIT_DATE = dt.date(2025, 6, 30)

NAIVE = "Top-K by market cap"
ANNEALED = "Simulated annealing"


def portfolio_returns(
    weights: pd.Series, stock_returns: pd.DataFrame, rebalance: bool = False
) -> pd.Series:
    """Daily returns of a portfolio that starts at ``weights``.

    ``rebalance=False`` buys once and holds, so weights drift with prices (like the index
    itself does). ``rebalance=True`` resets to ``weights`` every day, which is what the
    optimiser's objective assumes.
    """
    returns = stock_returns[weights.index]
    if rebalance:
        return returns @ weights
    value = (1 + returns).cumprod() @ weights
    return value.pct_change().fillna(value.iloc[0] - 1)


def tracking_error(port_returns: pd.Series, bench_returns: pd.Series) -> float:
    """Annualised standard deviation of daily return differences."""
    return float(np.sqrt(TRADING_DAYS) * (port_returns - bench_returns).std())


class Backtest:
    """A train/test split of a universe, plus the portfolios we compare on it."""

    def __init__(self, universe: Universe, split: dt.date = SPLIT_DATE):
        self.universe = universe
        self.split = split
        self.train = universe.between(None, split)
        self.test = universe.between(split, None)
        self.sigma = augmented_covariance(*self.train.returns())
        self._test_returns = self.test.returns()

    def _to_series(self, indices: np.ndarray, weights: np.ndarray) -> pd.Series:
        tickers = np.array(self.universe.tickers)[indices]
        return pd.Series(weights, index=tickers).sort_values(ascending=False)

    # ------------------------------------------------------------ portfolios

    def naive(self, k: int) -> pd.Series:
        """Top ``k`` stocks by market cap on the split date, cap-weighted."""
        weights = top_k_cap_weights(self.train.market_caps().to_numpy(), k)
        held = np.flatnonzero(weights)
        return self._to_series(held, weights[held])

    def anneal(self, k: int | None, **kwargs) -> tuple[pd.Series, AnnealResult]:
        """The basket simulated annealing picks using the training window only."""
        result = anneal(self.sigma, k=k, **kwargs)
        return self._to_series(result.selection, result.weights), result

    def full_universe(self) -> pd.Series:
        """Every stock, with weights fitted on the training window. In-sample this tracks
        almost perfectly, so it shows how much of any fit is overfitting."""
        _, weights = optimal_weights(self.sigma)
        held = np.flatnonzero(weights > 1e-6)
        return self._to_series(held, weights[held] / weights[held].sum())

    # ------------------------------------------------------------ evaluation

    def fit_tracking_error(self, weights: pd.Series) -> float:
        """In-sample tracking error on the training window (the optimiser's objective)."""
        full = pd.Series(0.0, index=self.universe.tickers)
        full[weights.index] = weights
        return annualise(tracking_variance(full.to_numpy(), self.sigma))

    def test_tracking_error(self, weights: pd.Series) -> float:
        """Out-of-sample tracking error, buying at the split date and holding."""
        stocks, bench = self._test_returns
        return tracking_error(portfolio_returns(weights, stocks), bench)

    def growth(self, weights: pd.Series) -> pd.Series:
        """Value of $1 invested at the split date, through the test window."""
        stocks, _ = self._test_returns
        return self._grow(portfolio_returns(weights, stocks))

    def benchmark_growth(self) -> pd.Series:
        return self._grow(self._test_returns[1])

    def _grow(self, returns: pd.Series) -> pd.Series:
        start = pd.Series([1.0], index=[self.test.prices.index[0]])
        return pd.concat([start, (1 + returns).cumprod()])

    def frontier(
        self, ks: list[int], seeds: tuple[int, ...] = (0, 1, 2, 3, 4), **anneal_kwargs
    ) -> pd.DataFrame:
        """Train and test tracking error against basket size: one row for the naive
        portfolio at each K, and one per seed for simulated annealing (runs vary, so
        report the spread rather than a lucky seed)."""
        rows = []
        for k in ks:
            portfolios = [(NAIVE, None, self.naive(k))]
            portfolios += [(ANNEALED, s, self.anneal(k, seed=s, **anneal_kwargs)[0]) for s in seeds]
            for method, seed, weights in portfolios:
                rows.append(
                    {
                        "k": k,
                        "method": method,
                        "seed": seed,
                        "train": self.fit_tracking_error(weights),
                        "test": self.test_tracking_error(weights),
                    }
                )
        return pd.DataFrame(rows)
