"""Static figures for the README: tracking error vs basket size, the out-of-sample
backtest, and the chosen holdings. Each comes in a light and a dark theme."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

from .backtest import ANNEALED, NAIVE


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    ink_secondary: str
    muted: str
    grid: str
    baseline: str
    annealed: str  # categorical slot 1 (blue)
    naive: str  # slot 2 (orange)
    added: str  # slot 3 (aqua)
    weight_ramp: tuple[str, str]  # sequential blue, light -> dark


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    baseline="#c3c2b7",
    annealed="#2a78d6",
    naive="#eb6834",
    added="#1baf7a",
    weight_ramp=("#9ec5f4", "#104281"),
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    ink="#ffffff",
    ink_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    baseline="#383835",
    annealed="#3987e5",
    naive="#d95926",
    added="#199e70",
    weight_ramp=("#184f95", "#9ec5f4"),
)

THEMES = (LIGHT, DARK)
METHOD_COLOURS = {NAIVE: "naive", ANNEALED: "annealed"}


def apply_theme(theme: Theme) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": theme.surface,
            "axes.facecolor": theme.surface,
            "savefig.facecolor": theme.surface,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "text.color": theme.ink,
            "axes.labelcolor": theme.ink_secondary,
            "axes.titlecolor": theme.ink,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.edgecolor": theme.baseline,
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": theme.grid,
            "grid.linewidth": 0.8,
            "xtick.color": theme.muted,
            "ytick.color": theme.muted,
            "xtick.labelcolor": theme.ink_secondary,
            "ytick.labelcolor": theme.ink_secondary,
            "legend.frameon": False,
            "legend.labelcolor": theme.ink_secondary,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def themed_path(path: Path, theme: Theme) -> Path:
    """``frontier.png`` -> ``frontier.png`` (light) / ``frontier-dark.png`` (dark)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path if theme is LIGHT else path.with_name(f"{path.stem}-dark{path.suffix}")


def _end_label(ax, x, y, text, theme: Theme, dy: float = 0) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=(6, dy),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=theme.ink_secondary,
    )


# ----------------------------------------------------------------- frontier


def plot_frontier(frontier: pd.DataFrame, out: Path, theme: Theme = LIGHT) -> None:
    """Tracking error vs K, training window beside test window, on a shared y-axis."""
    apply_theme(theme)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    titles = {"train": "Training window (in-sample fit)", "test": "Test window (out of sample)"}

    for ax, window in zip(axes, ("train", "test"), strict=True):
        ends = []
        for method in (NAIVE, ANNEALED):
            colour = getattr(theme, METHOD_COLOURS[method])
            stats = (
                frontier[frontier.method == method].groupby("k")[window].agg(["mean", "min", "max"])
            )
            if method == ANNEALED:
                ax.fill_between(
                    stats.index, stats["min"], stats["max"], color=colour, alpha=0.12, lw=0
                )
            ax.plot(stats.index, stats["mean"], color=colour, label=method)
            ax.plot(stats.index, stats["mean"], "o", color=colour, ms=5, mec=theme.surface, mew=1.5)
            ends.append((stats.index[-1], stats["mean"].iloc[-1]))
        ax.set_title(titles[window])
        ax.set_xlabel("Stocks held (K)")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_ylim(bottom=0)
        ax.margins(x=0.08)
        # End labels only where they won't collide; the legend and axis cover the rest.
        if abs(ends[0][1] - ends[1][1]) > 0.06 * ax.get_ylim()[1]:
            for x, y in ends:
                _end_label(ax, x, y, f"{y:.1%}", theme)
    axes[0].set_ylabel("Tracking error vs SPY (annualised)")
    axes[1].legend(loc="upper right")
    fig.text(
        0.01,
        0.01,
        "Simulated annealing: mean over seeds, band = best to worst seed.",
        fontsize=8,
        color=theme.muted,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(themed_path(out, theme), dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------- backtest


def plot_backtest(
    growth: dict[str, pd.Series], benchmark: pd.Series, out: Path, theme: Theme = LIGHT
) -> None:
    """Top: value of $1 through the test window. Bottom: cumulative difference vs SPY."""
    apply_theme(theme)
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )

    top.plot(benchmark.index, benchmark, color=theme.ink, lw=1.5, label="SPY")
    _end_label(top, benchmark.index[-1], benchmark.iloc[-1], "SPY", theme)
    for label, series in growth.items():
        colour = theme.annealed if label.startswith("Simulated") else theme.naive
        top.plot(series.index, series, color=colour, label=label)
        diff = series - benchmark
        bottom.plot(diff.index, diff, color=colour, label=label)
        _end_label(bottom, diff.index[-1], diff.iloc[-1], f"{diff.iloc[-1]:+.1%}", theme)

    top.set_title("Growth of $1, bought at the split date and held")
    top.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:.2f}"))
    top.legend(loc="upper left")
    bottom.axhline(0, color=theme.baseline, lw=1)
    bottom.set_title("Cumulative return minus SPY")
    bottom.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    bottom.margins(x=0.06)
    fig.tight_layout()
    fig.savefig(themed_path(out, theme), dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------- holdings


def plot_holdings(weights: pd.Series, sectors: pd.Series, out: Path, theme: Theme = LIGHT) -> None:
    """Horizontal bars of the annealed portfolio's weights, largest first."""
    apply_theme(theme)
    weights = weights.sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(weights) + 1.0))
    ax.barh(range(len(weights)), weights, height=0.6, color=theme.annealed)
    ax.set_yticks(
        range(len(weights)),
        [f"{t}  ·  {sectors[t]}" for t in weights.index],
        fontsize=9,
    )
    ax.tick_params(axis="y", length=0)
    for i, w in enumerate(weights):
        ax.annotate(
            f"{w:.1%}",
            (w, i),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=theme.ink_secondary,
        )
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_title(f"The {len(weights)} stocks simulated annealing picked")
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(themed_path(out, theme), dpi=160)
    plt.close(fig)
