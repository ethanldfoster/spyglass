"""Tracking-error maths and the inner problem: the best weights for a fixed set of stocks.

Notation: ``r`` is the vector of daily stock returns, ``r_b`` the benchmark's, and
``sigma_aug`` the covariance matrix of ``[r, r_b]`` (the benchmark is the last row and
column). For weights ``w`` and ``a = [w, -1]``::

    Var(w @ r - r_b) = a @ sigma_aug @ a

which is the (daily) tracking variance we minimise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import nnls

TRADING_DAYS = 252


def augmented_covariance(stock_returns: pd.DataFrame, bench_returns: pd.Series) -> np.ndarray:
    """Covariance of daily returns with the benchmark appended as the last row and column."""
    joined = pd.concat([stock_returns, bench_returns], axis=1, join="inner").dropna()
    return joined.cov().to_numpy()


def tracking_variance(weights: np.ndarray, sigma_aug: np.ndarray) -> float:
    """Daily variance of portfolio return minus benchmark return."""
    a = np.append(weights, -1.0)
    return float(a @ sigma_aug @ a)


def annualise(variance: float) -> float:
    """Daily tracking variance -> annualised tracking error (a standard deviation)."""
    return float(np.sqrt(TRADING_DAYS * max(variance, 0.0)))


def restrict(sigma_aug: np.ndarray, selection) -> np.ndarray:
    """The block of ``sigma_aug`` for the selected stocks (plus the benchmark)."""
    idx = [*sorted(selection), sigma_aug.shape[0] - 1]
    return sigma_aug[np.ix_(idx, idx)]


def optimal_weights(sigma_aug: np.ndarray, penalty: float = 1e3) -> tuple[float, np.ndarray]:
    """Long-only, fully invested weights over *all* stocks in ``sigma_aug`` minimising
    tracking variance. Returns ``(tracking_variance, weights)``.

    The problem is the QP ``min w'Sw - 2s'w  s.t.  w >= 0, sum(w) = 1``, where ``S`` is
    the stock covariance block and ``s`` the stocks' covariance with the benchmark. With
    the Cholesky factor ``S = U'U`` and ``U'y = s``, the objective is ``|Uw - y|^2`` up to
    a constant: a non-negative least-squares problem. The budget constraint goes in as a
    heavily weighted extra row, and the result is renormalised. This gives the same optimum
    as a general QP solver (SLSQP) about 40x faster.
    """
    # Work relative to the benchmark's variance so the numbers are O(1).
    scaled = sigma_aug / sigma_aug[-1, -1]
    cov, cov_bench = scaled[:-1, :-1], scaled[:-1, -1]
    n = len(cov_bench)
    try:
        upper = cholesky(cov)
    except np.linalg.LinAlgError:  # singular, e.g. more stocks than observations
        upper = cholesky(cov + 1e-10 * np.trace(cov) / n * np.eye(n))
    y = solve_triangular(upper, cov_bench, trans="T")

    weights, _ = nnls(np.vstack([upper, np.full(n, penalty)]), np.append(y, penalty))
    weights /= weights.sum()
    return tracking_variance(weights, sigma_aug), weights


def top_k_cap_weights(market_caps: np.ndarray, k: int) -> np.ndarray:
    """The naive approach: hold the ``k`` largest companies, weighted by market cap."""
    weights = np.zeros(len(market_caps))
    top = np.argsort(-market_caps)[:k]
    weights[top] = market_caps[top]
    return weights / weights.sum()
