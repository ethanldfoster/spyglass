"""Render an annealing run as an animated GIF.

Left: every stock in the universe, grouped by sector and ordered by market cap, lit up
while it is held. Right: the tracking error, temperature and acceptance rate so far.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
from PIL import Image

from .annealing import AnnealResult, Snapshot
from .plots import LIGHT, Theme, apply_theme, themed_path

GRID_COLUMNS = 40
CELL = 0.82  # side of a stock's square, leaving a gap to its neighbours


def _grid_layout(sectors: pd.Series, market_caps: pd.Series) -> tuple[np.ndarray, list]:
    """(x, y) cell for each stock (in universe order) and (y, name) sector labels.

    Sectors are stacked largest first; within a sector, stocks run by market cap.
    """
    positions = np.zeros((len(sectors), 2))
    order = {t: i for i, t in enumerate(sectors.index)}
    labels, row = [], 0.0
    for sector in sectors.value_counts().index:
        members = market_caps[sectors.index[sectors == sector]].sort_values(ascending=False)
        labels.append((-row, sector))
        row += 1.1  # room for the label
        for j, ticker in enumerate(members.index):
            positions[order[ticker]] = (j % GRID_COLUMNS, -(row + j // GRID_COLUMNS))
        row += math.ceil(len(members) / GRID_COLUMNS) + 0.2
    return positions, labels


def animate(
    result: AnnealResult,
    sectors: pd.Series,
    market_caps: pd.Series,
    naive_te: float,
    out: Path,
    theme: Theme = LIGHT,
    frame_ms: int = 70,
    hold_ms: int = 3000,
    dpi: int = 90,
) -> Path:
    """Write ``result`` (which needs ``snapshots``) as a GIF and return its path."""
    if not result.snapshots:
        raise ValueError("anneal(..., snapshot_every=N) is needed to animate a run")
    apply_theme(theme)
    history = result.history
    n_steps = int(history.index[-1])
    k = len(result.selection)
    tickers = np.array(sectors.index)

    fig = plt.figure(figsize=(13, 6.6), dpi=dpi)
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.35, 1],
        height_ratios=[2, 1, 1],
        left=0.03,
        right=0.97,
        top=0.86,
        bottom=0.08,
        wspace=0.12,
        hspace=0.55,
    )
    ax_u = fig.add_subplot(grid[:, 0])
    ax_te = fig.add_subplot(grid[0, 1])
    ax_t = fig.add_subplot(grid[1, 1], sharex=ax_te)
    ax_acc = fig.add_subplot(grid[2, 1], sharex=ax_te)

    fig.text(
        0.03,
        0.95,
        f"Simulated annealing: choosing {k} of {len(tickers)} S&P 500 stocks to track SPY",
        fontsize=14,
        weight="bold",
    )
    status = fig.text(0.03, 0.905, "", fontsize=10.5, color=theme.ink_secondary)

    # ---- the universe
    ramp = LinearSegmentedColormap.from_list("weight", theme.weight_ramp)
    positions, labels = _grid_layout(sectors, market_caps)
    ax_u.set_axis_off()
    ax_u.set_aspect("equal", anchor="NW")
    ax_u.set_xlim(-0.5, GRID_COLUMNS - 0.5)
    ax_u.set_ylim(positions[:, 1].min() - 0.5, 0.5)
    for y, name in labels:
        ax_u.text(-0.45, y - 0.15, name, fontsize=8, color=theme.muted, va="center")
    cells = PatchCollection(
        [Rectangle((x - CELL / 2, y - CELL / 2), CELL, CELL) for x, y in positions],
        linewidths=0,
    )
    ax_u.add_collection(cells)
    ax_u.legend(
        handles=[
            Line2D(
                [],
                [],
                ls="",
                marker="s",
                ms=8,
                color=ramp(0.6),
                label="Held (stronger = larger weight)",
            ),
            Line2D([], [], ls="", marker="s", ms=8, color=theme.added, label="Just swapped in"),
            Line2D([], [], ls="", marker="s", ms=8, color=theme.grid, label="Not held"),
        ],
        loc="upper left",
        bbox_to_anchor=(0, 0.0),
        ncol=3,
        fontsize=8.5,
        handletextpad=0.2,
        columnspacing=1.2,
    )
    holdings = ax_u.text(
        0, -0.075, "", transform=ax_u.transAxes, fontsize=8.5, color=theme.ink_secondary
    )

    idle, added = to_rgba(theme.grid), to_rgba(theme.added)
    max_weight = max(s.weights.max() for s in result.snapshots)

    # ---- traces
    steps = history.index.to_numpy()
    te, best_te = history["tracking_error"].to_numpy(), history["best_tracking_error"].to_numpy()
    temperature = history["temperature"].to_numpy()
    acceptance = history["accepted"].rolling(100, min_periods=1).mean().to_numpy()

    ax_te.set_title("Tracking error (annualised, training window)")
    ax_te.axhline(naive_te, color=theme.naive, lw=1.5)
    ax_te.annotate(
        f"Top-{k} by market cap: {naive_te:.1%}",
        (n_steps, naive_te),
        xytext=(0, 5),
        textcoords="offset points",
        ha="right",
        fontsize=8.5,
        color=theme.ink_secondary,
    )
    (te_line,) = ax_te.plot([], [], color=theme.annealed, lw=1.2, label="Current basket")
    (best_line,) = ax_te.plot([], [], color=theme.ink, lw=1.8, label="Best so far")
    ax_te.set_ylim(0, max(te.max(), naive_te) * 1.12)
    ax_te.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_te.legend(loc="upper right", bbox_to_anchor=(1, 0.88), fontsize=8.5)

    ax_t.set_title("Temperature")
    ax_t.set_yscale("log")
    ax_t.set_ylim(temperature.min() / 1.5, temperature.max() * 1.5)
    (t_line,) = ax_t.plot([], [], color=theme.ink_secondary, lw=1.5)

    ax_acc.set_title("Share of proposals accepted (last 100 steps)")
    ax_acc.set_ylim(0, 1)
    ax_acc.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    (acc_line,) = ax_acc.plot([], [], color=theme.ink_secondary, lw=1.5)
    ax_acc.set_xlim(0, n_steps)
    ax_acc.set_xlabel("Step")
    for ax in (ax_te, ax_t):
        ax.tick_params(labelbottom=False)

    def draw(snap: Snapshot, previous: Snapshot | None, final: bool) -> Image.Image:
        colours = np.tile(idle, (len(tickers), 1))
        colours[snap.selection] = ramp(0.15 + 0.85 * snap.weights / max_weight)
        if previous is not None and not final:
            new = np.setdiff1d(snap.selection, previous.selection)
            colours[new] = added
        cells.set_facecolors(colours)

        upto = steps <= max(snap.step, 1)
        for line, ys in (
            (te_line, te),
            (best_line, best_te),
            (t_line, temperature),
            (acc_line, acceptance),
        ):
            line.set_data(steps[upto], ys[upto])

        top = np.argsort(snap.weights)[::-1][:6]
        holdings.set_text(
            "Largest holdings:   "
            + "   ".join(f"{tickers[snap.selection[i]]} {snap.weights[i]:.1%}" for i in top)
        )
        if final:
            status.set_text(
                f"Done after {n_steps:,} steps. Best basket tracks SPY with "
                f"{snap.tracking_error:.2%} tracking error in the training window, "
                f"vs {naive_te:.2%} "
                f"for the {k} largest stocks."
            )
        else:
            row = history.loc[max(snap.step, 1)]
            status.set_text(
                f"Step {snap.step:,} of {n_steps:,}   ·   temperature "
                f"{row.temperature:.1e}   ·   tracking error "
                f"{snap.tracking_error:.2%}   (best {row.best_tracking_error:.2%})"
            )
        fig.canvas.draw()
        return Image.frombuffer(
            "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
        ).convert("RGB")

    frames = [
        draw(s, p, False)
        for p, s in zip([None, *result.snapshots[:-1]], result.snapshots, strict=True)
    ]
    best = Snapshot(n_steps, result.selection, result.weights, result.tracking_error)
    frames.append(draw(best, None, True))
    plt.close(fig)

    # One shared palette (taken from the first, middle and last frames) keeps colours
    # steady across frames and lets the GIF encoder store only what changed.
    sample = [frames[0], frames[len(frames) // 2], frames[-1]]
    strip = Image.new("RGB", (sample[0].width, sample[0].height * 3))
    for i, im in enumerate(sample):
        strip.paste(im, (0, i * im.height))
    palette = strip.quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    gif = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in frames]

    path = themed_path(out, theme)
    durations = [frame_ms] * (len(gif) - 1) + [hold_ms]
    gif[0].save(
        path,
        save_all=True,
        append_images=gif[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )
    return path
