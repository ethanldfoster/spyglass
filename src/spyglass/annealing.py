"""Simulated annealing over *which* stocks to hold.

Choosing K of N stocks is a combinatorial problem (C(500, 20) ~ 10^35 baskets), but for
any one basket the best weights are a cheap convex QP (``tracking.optimal_weights``). So
the annealer walks over baskets, scoring each by the tracking error its optimal weights
achieve.

Each step proposes a neighbouring basket (swap one stock, or add/remove one when K is
free) and accepts it with the Metropolis rule: always if it is better, otherwise with
probability ``exp(-delta / T)``. The temperature ``T`` decays geometrically, so early on
the search roams freely and later it settles into a good basin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .tracking import annualise, optimal_weights, restrict

MOVES = ("add", "swap", "remove")


class SelectionScorer:
    """Scores a basket of stock indices: annualised tracking error of its optimal weights,
    plus ``holding_cost`` per stock held. Results are cached, since late in a run the
    annealer revisits the same baskets often."""

    def __init__(self, sigma_aug: np.ndarray, holding_cost: float = 0.0):
        self.sigma_aug = sigma_aug
        self.holding_cost = holding_cost
        self._cache: dict[frozenset, tuple[float, np.ndarray]] = {}

    def __call__(self, selection: frozenset) -> tuple[float, np.ndarray]:
        """``(score, weights)`` with weights ordered like ``sorted(selection)``."""
        if selection not in self._cache:
            variance, weights = optimal_weights(restrict(self.sigma_aug, selection))
            score = annualise(variance) + self.holding_cost * len(selection)
            self._cache[selection] = (score, weights)
        return self._cache[selection]

    def tracking_error(self, score: float, size: int) -> float:
        return score - self.holding_cost * size


def initial_state(n: int, k: int, rng: np.random.Generator) -> frozenset:
    return frozenset(rng.choice(n, size=k, replace=False).tolist())


def propose_move(
    state: frozenset,
    n: int,
    rng: np.random.Generator,
    probs: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
) -> tuple[frozenset, str]:
    """A neighbouring basket and the name of the move that produced it.

    ``probs`` are the (add, swap, remove) probabilities. Moves that are impossible for
    the current basket size are dropped and the rest renormalised.
    """
    p = np.array(probs, dtype=float)
    if len(state) == n:
        p[:2] = 0.0  # nothing left to add or swap in
    if len(state) == 1:
        p[2] = 0.0  # never empty the basket
    move = MOVES[rng.choice(3, p=p / p.sum())]

    held = sorted(state)
    outside = np.setdiff1d(np.arange(n), held)
    if move == "add":
        return state | {int(rng.choice(outside))}, move
    if move == "remove":
        return state - {int(rng.choice(held))}, move
    return (state - {int(rng.choice(held))}) | {int(rng.choice(outside))}, move


def calibrate_temperature(
    scorer: SelectionScorer,
    n: int,
    k: int,
    probs: tuple[float, float, float],
    rng: np.random.Generator,
    n_probes: int = 200,
    acceptance: float = 0.8,
) -> float:
    """A starting temperature at which a typical uphill move is accepted with probability
    ``acceptance``, measured on a random walk using the same moves as the real run."""
    state = initial_state(n, k, rng)
    score, _ = scorer(state)
    uphill = []
    for _ in range(n_probes):
        new_state, _ = propose_move(state, n, rng, probs)
        new_score, _ = scorer(new_state)
        if new_score > score:
            uphill.append(new_score - score)
        state, score = new_state, new_score
    return -float(np.mean(uphill)) / math.log(acceptance)


@dataclass
class Snapshot:
    """The current basket at some step, for animating the run."""

    step: int
    selection: np.ndarray
    weights: np.ndarray
    tracking_error: float


@dataclass
class AnnealResult:
    selection: np.ndarray  # sorted stock indices of the best basket found
    weights: np.ndarray  # its optimal weights, aligned with ``selection``
    tracking_error: float  # annualised, in-sample
    score: float  # tracking error + holding cost
    history: pd.DataFrame  # one row per step
    snapshots: list[Snapshot] = field(default_factory=list)


def anneal(
    sigma_aug: np.ndarray,
    k: int | None = 20,
    n_steps: int = 4000,
    t_init: float | None = None,
    alpha: float = 0.999,
    holding_cost: float = 0.0,
    move_probs: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    snapshot_every: int = 0,
    seed: int | None = None,
) -> AnnealResult:
    """Search for the basket of stocks that best tracks the benchmark.

    Args:
        sigma_aug: covariance of stock and benchmark returns, benchmark last.
        k: basket size. ``None`` lets the size vary, trading tracking error against
            ``holding_cost`` per stock (starting from N/3 stocks).
        n_steps: number of proposals.
        t_init: starting temperature, in units of annualised tracking error. ``None``
            calibrates it so ~80% of uphill moves are accepted at the start.
        alpha: geometric cooling rate, ``T <- alpha * T`` each step.
        holding_cost: penalty per stock held, in units of annualised tracking error.
        move_probs: (add, swap, remove) probabilities when ``k`` is ``None``.
        snapshot_every: record the current basket every this many steps (0 = never).
        seed: for reproducible runs.
    """
    rng = np.random.default_rng(seed)
    n = sigma_aug.shape[0] - 1
    probs = (0.0, 1.0, 0.0) if k is not None else move_probs
    scorer = SelectionScorer(sigma_aug, holding_cost)

    temperature = t_init
    if temperature is None:
        temperature = calibrate_temperature(scorer, n, k or n // 3, probs, rng)

    state = initial_state(n, k or n // 3, rng)
    score, weights = scorer(state)
    best_state, best_score, best_weights = state, score, weights

    def snapshot(step: int) -> Snapshot:
        te = scorer.tracking_error(score, len(state))
        return Snapshot(step, np.array(sorted(state)), weights, te)

    snapshots = [snapshot(0)] if snapshot_every else []
    rows = []
    for step in range(1, n_steps + 1):
        new_state, move = propose_move(state, n, rng, probs)
        new_score, new_weights = scorer(new_state)

        delta = new_score - score
        accepted = delta <= 0 or rng.random() < math.exp(-delta / temperature)
        if accepted:
            state, score, weights = new_state, new_score, new_weights
            if score < best_score:
                best_state, best_score, best_weights = state, score, weights

        rows.append(
            (
                step,
                temperature,
                move,
                accepted,
                len(state),
                score,
                best_score,
                scorer.tracking_error(score, len(state)),
                scorer.tracking_error(best_score, len(best_state)),
            )
        )
        if snapshot_every and step % snapshot_every == 0:
            snapshots.append(snapshot(step))
        temperature *= alpha

    history = pd.DataFrame(
        rows,
        columns=[
            "step",
            "temperature",
            "move",
            "accepted",
            "size",
            "score",
            "best_score",
            "tracking_error",
            "best_tracking_error",
        ],
    ).set_index("step")

    return AnnealResult(
        selection=np.array(sorted(best_state)),
        weights=best_weights,
        tracking_error=scorer.tracking_error(best_score, len(best_state)),
        score=best_score,
        history=history,
        snapshots=snapshots,
    )
