"""Command line interface: ``spyglass {fetch,run,frontier,animate,figures}``."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from pathlib import Path

import matplotlib
import pandas as pd

from . import data
from .backtest import ANNEALED, NAIVE, SPLIT_DATE, Backtest

ASSETS = data.DATA_DIR.parent / "assets"
FRONTIER_KS = [5, 10, 15, 20, 30, 40, 60, 80, 100]


def _date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def _k(s: str) -> int | None:
    return None if s == "auto" else int(s)


def _backtest(args) -> Backtest:
    return Backtest(data.load_universe(), split=args.split)


def _anneal_kwargs(args) -> dict:
    return {"n_steps": args.steps, "alpha": args.alpha, "holding_cost": args.holding_cost}


# ------------------------------------------------------------------ commands


def cmd_fetch(args) -> None:
    data.fetch(start=args.start, end=args.end)


def cmd_run(args) -> None:
    bt = _backtest(args)
    started = time.perf_counter()
    annealed, result = bt.anneal(args.k, seed=args.seed, **_anneal_kwargs(args))
    elapsed = time.perf_counter() - started
    k = len(annealed)
    naive = bt.naive(k)

    train, test = bt.train.prices.index, bt.test.prices.index
    print(f"Universe:  {len(bt.universe)} S&P 500 stocks, benchmark SPY")
    print(f"Train:     {train[0]:%Y-%m-%d} to {train[-1]:%Y-%m-%d} ({len(train)} days)")
    print(f"Test:      {test[0]:%Y-%m-%d} to {test[-1]:%Y-%m-%d} ({len(test)} days)")
    print(f"Annealing: {args.steps:,} steps in {elapsed:.1f}s, seed {args.seed}\n")

    rows = [(f"{ANNEALED} (K={k})", annealed), (f"{NAIVE} (K={k})", naive)]
    print(f"{'Tracking error vs SPY (annualised)':<40}{'Train':>8}{'Test':>8}")
    for label, weights in rows:
        print(
            f"{label:<40}{bt.fit_tracking_error(weights):>8.2%}"
            f"{bt.test_tracking_error(weights):>8.2%}"
        )

    print(f"\nHoldings ({k} stocks):")
    for ticker, weight in annealed.items():
        print(
            f"  {ticker:<6} {weight:>6.2%}  {bt.universe.names[ticker]} "
            f"({bt.universe.sectors[ticker]})"
        )


def cmd_frontier(args) -> pd.DataFrame:
    from .plots import THEMES, plot_frontier

    bt = _backtest(args)
    frontier = bt.frontier(args.ks, seeds=tuple(range(args.seeds)), **_anneal_kwargs(args))
    summary = frontier.groupby(["k", "method"])[["train", "test"]].mean().unstack("method")
    print(summary.map("{:.2%}".format).to_string())
    out = args.out / "frontier.png"
    frontier.to_csv(args.out / "frontier.csv", index=False, float_format="%.6f")
    for theme in THEMES:
        plot_frontier(frontier, out, theme)
    print(f"Wrote {out} (+ dark variant) and frontier.csv")
    return frontier


def cmd_animate(args) -> None:
    from .animate import animate
    from .plots import THEMES

    bt = _backtest(args)
    _, result = bt.anneal(args.k, seed=args.seed, snapshot_every=args.every, **_anneal_kwargs(args))
    naive_te = bt.fit_tracking_error(bt.naive(len(result.selection)))
    for theme in THEMES:
        path = animate(
            result,
            bt.universe.sectors,
            bt.train.market_caps(),
            naive_te,
            args.out / "annealing.gif",
            theme,
        )
        print(f"Wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


def cmd_figures(args) -> None:
    """Every README asset, from one seed so they all show the same portfolio."""
    from .plots import THEMES, plot_backtest, plot_holdings

    bt = _backtest(args)
    annealed, _ = bt.anneal(args.k, seed=args.seed, **_anneal_kwargs(args))
    naive = bt.naive(len(annealed))
    growth = {
        f"{ANNEALED} ({len(annealed)} stocks)": bt.growth(annealed),
        f"{NAIVE} ({len(naive)} stocks)": bt.growth(naive),
    }
    for theme in THEMES:
        plot_backtest(growth, bt.benchmark_growth(), args.out / "backtest.png", theme)
        plot_holdings(annealed, bt.universe.sectors, args.out / "holdings.png", theme)
    print(f"Wrote backtest.png and holdings.png to {args.out}")
    cmd_animate(args)
    cmd_frontier(args)


# ------------------------------------------------------------------ parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spyglass",
        description="Track SPY with a handful of S&P 500 stocks, picked by simulated annealing.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="show data-cleaning logs")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download prices and share counts into data/")
    fetch.add_argument("--start", type=_date, default=data.START_DATE)
    fetch.add_argument("--end", type=_date, default=data.END_DATE)
    fetch.set_defaults(func=cmd_fetch)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--split",
        type=_date,
        default=SPLIT_DATE,
        help="train on data up to this date, test after it (default %(default)s)",
    )
    common.add_argument("--steps", type=int, default=4000, help="annealing steps")
    common.add_argument("--alpha", type=float, default=0.999, help="cooling rate per step")
    common.add_argument(
        "--holding-cost",
        type=float,
        default=0.0,
        help="penalty per stock held, in annualised tracking error; use with --k auto",
    )
    common.add_argument("--out", type=Path, default=ASSETS, help="where to write figures")

    single = argparse.ArgumentParser(add_help=False)
    single.add_argument(
        "--k", type=_k, default=20, help="stocks to hold, or 'auto' to let --holding-cost decide"
    )
    single.add_argument("--seed", type=int, default=0)

    run = sub.add_parser(
        "run",
        parents=[common, single],
        help="optimise one portfolio and compare it with the naive one",
    )
    run.set_defaults(func=cmd_run)

    animate = sub.add_parser("animate", parents=[common, single], help="render the GIF")
    animate.add_argument("--every", type=int, default=25, help="steps per frame")
    animate.set_defaults(func=cmd_animate)

    frontier = sub.add_parser(
        "frontier", parents=[common], help="tracking error against basket size"
    )
    frontier.add_argument("--ks", type=int, nargs="+", default=FRONTIER_KS)
    frontier.add_argument("--seeds", type=int, default=5, help="annealing runs per K")
    frontier.set_defaults(func=cmd_frontier)

    figures = sub.add_parser(
        "figures", parents=[common, single], help="regenerate every image in assets/"
    )
    figures.add_argument("--every", type=int, default=25)
    figures.add_argument("--ks", type=int, nargs="+", default=FRONTIER_KS)
    figures.add_argument("--seeds", type=int, default=5)
    figures.set_defaults(func=cmd_figures)
    return parser


def main(argv: list[str] | None = None) -> None:
    matplotlib.use("Agg")  # files only, never windows
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )
    args.func(args)


if __name__ == "__main__":
    main()
