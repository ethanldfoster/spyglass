import numpy as np
import pandas as pd
import pytest


def make_market(n_stocks: int, n_days: int = 500, seed: int = 0):
    """Synthetic one-factor market: stock returns plus a cap-weighted benchmark."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0004, 0.01, n_days)
    beta = rng.uniform(0.6, 1.4, n_stocks)
    idio = rng.normal(0.0, 0.015, (n_days, n_stocks))
    returns = pd.DataFrame(
        market[:, None] * beta + idio, columns=[f"S{i}" for i in range(n_stocks)]
    )
    caps = rng.lognormal(0.0, 1.0, n_stocks)
    bench = pd.Series(returns.to_numpy() @ (caps / caps.sum()), name="BENCH")
    return returns, bench, caps


@pytest.fixture
def market():
    return make_market(12)
