from itertools import combinations

import numpy as np
import pytest

from poor_mans_etf.annealing import SelectionScorer, anneal, propose_move
from poor_mans_etf.tracking import augmented_covariance


@pytest.fixture
def sigma(market):
    return augmented_covariance(*market[:2])


def test_swap_keeps_size_fixed():
    rng = np.random.default_rng(0)
    state = frozenset({0, 1, 2})
    for _ in range(200):
        new, move = propose_move(state, 10, rng, probs=(0, 1, 0))
        assert move == "swap"
        assert len(new) == 3 and len(new - state) == 1
        state = new


def test_moves_stay_within_bounds():
    rng = np.random.default_rng(0)
    n, state = 5, frozenset({0})
    for _ in range(500):
        new, move = propose_move(state, n, rng)
        assert 1 <= len(new) <= n
        assert len(new) - len(state) == {"add": 1, "swap": 0, "remove": -1}[move]
        assert all(0 <= i < n for i in new)
        state = new


def test_full_basket_can_only_shrink():
    rng = np.random.default_rng(0)
    full = frozenset(range(4))
    assert all(propose_move(full, 4, rng)[1] == "remove" for _ in range(50))


def test_finds_brute_force_optimum(sigma):
    scorer = SelectionScorer(sigma)
    n = sigma.shape[0] - 1
    best = min(scorer(frozenset(c))[0] for c in combinations(range(n), 3))

    result = anneal(sigma, k=3, n_steps=1500, seed=0)
    assert np.isclose(result.score, best)


def test_best_weights_defined_without_improvement(sigma):
    # Regression: the original code only set best_weights once a better state was found.
    result = anneal(sigma, k=3, n_steps=0, t_init=1.0, seed=0)
    assert len(result.weights) == 3
    assert np.isclose(result.weights.sum(), 1)


def test_free_size_trades_off_holding_cost(sigma):
    cheap = anneal(sigma, k=None, n_steps=800, holding_cost=0.0, seed=0)
    pricey = anneal(sigma, k=None, n_steps=800, holding_cost=0.05, seed=0)
    assert len(pricey.selection) < len(cheap.selection)


def test_history_and_snapshots(sigma):
    result = anneal(sigma, k=4, n_steps=100, snapshot_every=10, seed=1)
    assert len(result.history) == 100
    assert result.history["best_score"].is_monotonic_decreasing
    assert [s.step for s in result.snapshots] == list(range(0, 101, 10))


def test_seed_makes_runs_reproducible(sigma):
    a = anneal(sigma, k=4, n_steps=200, seed=7)
    b = anneal(sigma, k=4, n_steps=200, seed=7)
    assert np.array_equal(a.selection, b.selection)
    assert a.score == b.score
