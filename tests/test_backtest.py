import numpy as np
import pandas as pd
import pytest

from spyglass.backtest import Backtest, portfolio_returns, tracking_error
from spyglass.data import load_universe, to_yahoo_ticker


@pytest.fixture(scope="module")
def backtest():
    return Backtest(load_universe())


def test_buy_and_hold_lets_weights_drift():
    returns = pd.DataFrame({"A": [0.10, 0.0], "B": [-0.10, 0.10]})
    weights = pd.Series({"A": 0.5, "B": 0.5})

    held = portfolio_returns(weights, returns)
    # Day 1: 0.5*1.1 + 0.5*0.9 = 1.0. Day 2: only B (now worth 0.45) moves, +10%.
    assert np.allclose(held, [0.0, 0.045])
    assert np.allclose(portfolio_returns(weights, returns, rebalance=True), [0.0, 0.05])


def test_tracking_error_of_benchmark_is_zero():
    r = pd.Series([0.01, -0.02, 0.005])
    assert tracking_error(r, r) == 0


def test_yahoo_ticker_mapping():
    assert to_yahoo_ticker("BRK.B") == "BRK-B"
    assert to_yahoo_ticker("AAPL") == "AAPL"


def test_universe_is_clean(backtest):
    universe = backtest.universe
    assert len(universe) > 450
    assert not universe.prices.isna().any().any()
    assert "BRK-B" in universe.tickers  # share-class tickers are mapped, not dropped
    assert "GOOG" not in universe.tickers  # one share class per company
    assert universe.benchmark.index.equals(universe.prices.index)


def test_split_does_not_leak(backtest):
    assert backtest.train.prices.index[-1] <= pd.Timestamp(backtest.split)
    assert backtest.test.prices.index[0] >= pd.Timestamp(backtest.split)
    # Naive weights use market caps on the split date, not later.
    caps = backtest.train.market_caps()
    assert np.allclose(caps, backtest.universe.shares * backtest.train.close.iloc[-1])


def test_annealing_beats_naive_in_sample(backtest):
    annealed, _ = backtest.anneal(10, n_steps=300, seed=0)
    naive = backtest.naive(10)
    assert len(annealed) == len(naive) == 10
    assert np.isclose(annealed.sum(), 1)
    assert backtest.fit_tracking_error(annealed) < backtest.fit_tracking_error(naive)


def test_growth_starts_at_one(backtest):
    growth = backtest.growth(backtest.naive(5))
    assert growth.iloc[0] == 1.0
    assert growth.index.equals(backtest.benchmark_growth().index)
