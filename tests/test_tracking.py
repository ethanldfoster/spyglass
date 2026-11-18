import numpy as np
from scipy.optimize import minimize

from spyglass.tracking import (
    annualise,
    augmented_covariance,
    optimal_weights,
    restrict,
    top_k_cap_weights,
    tracking_variance,
)


def test_tracking_variance_matches_direct_calculation(market):
    returns, bench, _ = market
    sigma = augmented_covariance(returns, bench)
    w = np.random.default_rng(1).dirichlet(np.ones(returns.shape[1]))

    direct = np.var(returns.to_numpy() @ w - bench.to_numpy(), ddof=1)
    assert np.isclose(tracking_variance(w, sigma), direct)


def test_matches_general_qp_solver(market):
    sigma = augmented_covariance(*market[:2])
    n = sigma.shape[0] - 1
    reference = minimize(
        lambda w: tracking_variance(w, sigma) / sigma[-1, -1],
        np.full(n, 1 / n),
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    variance, w = optimal_weights(sigma)
    assert variance <= tracking_variance(reference.x, sigma) * (1 + 1e-6)
    assert np.allclose(w, reference.x, atol=1e-4)


def test_handles_singular_covariance(market):
    returns, bench, _ = market
    sigma = augmented_covariance(returns.iloc[:8], bench.iloc[:8])  # 12 stocks, 8 days
    variance, w = optimal_weights(sigma)
    assert np.isfinite(variance) and np.isclose(w.sum(), 1) and np.all(w >= 0)


def test_recovers_exact_replicating_weights(market):
    returns, _, _ = market
    true_w = np.array([0.5, 0.3, 0.2])
    bench = returns.iloc[:, :3] @ true_w
    sigma = augmented_covariance(returns.iloc[:, :5], bench)

    variance, w = optimal_weights(sigma)
    assert variance < 1e-12
    assert np.allclose(w, [0.5, 0.3, 0.2, 0, 0], atol=1e-4)


def test_optimal_weights_are_long_only_and_fully_invested(market):
    sigma = augmented_covariance(*market[:2])
    _, w = optimal_weights(restrict(sigma, {0, 3, 5, 7}))
    assert len(w) == 4
    assert np.all(w >= 0)
    assert np.isclose(w.sum(), 1)


def test_optimal_weights_beat_naive_cap_weights(market):
    returns, bench, caps = market
    sigma = augmented_covariance(returns, bench)
    naive = top_k_cap_weights(caps, 4)
    selection = np.flatnonzero(naive)

    best, _ = optimal_weights(restrict(sigma, selection))
    assert best <= tracking_variance(naive, sigma)


def test_top_k_cap_weights():
    w = top_k_cap_weights(np.array([1.0, 5.0, 3.0, 2.0]), 2)
    assert np.allclose(w, [0, 5 / 8, 3 / 8, 0])


def test_annualise():
    assert np.isclose(annualise(1e-4 / 252), 0.01)
