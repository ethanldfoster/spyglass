import numpy as np
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
import pandas as pd
from scipy.optimize import minimize
import random

# ---------------- Data Functions ----------------
def get_historical_data(indexes, start_date: datetime.date, end_date: datetime.date):
    df = yf.download(
        indexes, 
        start=start_date,
        end=end_date,
        auto_adjust=False,
    )
    return df['Adj Close'][indexes]


def get_market_caps(indexes: list) -> np.ndarray:
    txs = yf.Tickers(" ".join(indexes)).tickers
    return np.array([txs[tx].info["marketCap"] for tx in indexes])


def get_daily_returns(historical_data: pd.DataFrame) -> pd.DataFrame:
    price_data = historical_data
    return price_data.pct_change().dropna()


def get_covariance_matrix(historical_data: pd.DataFrame) -> np.ndarray:
    daily_returns = get_daily_returns(historical_data)
    return daily_returns.cov().to_numpy()



def get_benchmark_weights(market_caps: np.ndarray) -> np.ndarray:
    return np.array([n / sum(market_caps) for n in market_caps])


# ---------------- Analysis Functions ----------------
def AbsTrackingError(weights: np.ndarray, benchmark_weights: np.ndarray, sigma: np.ndarray):
    diff_vector = weights - benchmark_weights

    return diff_vector @ sigma @ diff_vector


def RelativeTrackingError(weights: np.ndarray, sigma_aug: np.ndarray):
    a = np.append(weights, -1)
    return a @ sigma_aug @ a


def get_naive_k_weightings(market_caps, K):
    weights = np.zeros(len(market_caps))
    cap_sorted_idx = np.argsort(market_caps, descending=True)[:K]
    weights[cap_sorted_idx] = market_caps[cap_sorted_idx]
    weights /= sum(weights)
    return weights


def get_sigma_aug(historical_data, benchmark_data):
    daily_returns = get_daily_returns(historical_data)
    benchmark_returns = get_daily_returns(benchmark_data)
    return pd.concat([daily_returns, benchmark_returns], axis=1).dropna().cov().to_numpy()


def calc_floor_performance(market_caps: np.ndarray, sigma_aug: np.ndarray, K: int):
    w = get_naive_k_weightings(market_caps, K)
    return RelativeTrackingError(w, sigma_aug)


# Best possible tracking of fund (computable for small N)
def calc_ceiling_performance(sigma_aug: np.ndarray, N: int):
    def ScaledTE(weights: np.ndarray):
        return 1e6 * RelativeTrackingError(weights, sigma_aug)
    
    w0 = np.ones(N) / N

    res = minimize(
        ScaledTE,
        w0,
        method='SLSQP',
        bounds=[(0, None)] * N,
        constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    )

    return (res.fun / 1e6), res.x


# Simulated Annealing Functions
def get_initial_state(N: int, K: int | None = None):
    K = N // 3 if K is None else K
    return set(random.sample(range(N), K))


def propose_move(
    selected: set,
    N: int,
    p_add: float = 0.33,
    p_swap: float = 0.34,
    p_remove: float = 0.33
) -> set:
    chance = random.random()
    new_set = selected.copy()

    # guards
    if len(selected) == 1:
        p_remove = 0
    if len(selected) == N:
        p_add = 0

    if chance < p_add:
        candidates = tuple(set(range(N)) - selected)
        new_set.add(random.choice(candidates))
    elif chance < p_add + p_remove:
        new_set.remove(random.choice(tuple(selected)))
    else:
        candidates = tuple(set(range(N)) - selected)
        if not candidates:  # size == N, can't swap, fall back to remove
            new_set.remove(random.choice(tuple(selected)))
        else:
            to_remove = random.choice(tuple(selected))
            new_set.remove(to_remove)
            new_set.add(random.choice(candidates))

    return new_set

    

def ScoreSelection(selected_indices: set, sigma_aug: np.ndarray, cost_factor: float = 1e-6, w0 = None) -> tuple[float, np.ndarray]:
    # Make Cov matrix smaller for speedier calcs
    bench_idx = sigma_aug.shape[0] - 1
    sub_idx = sorted(selected_indices) + [bench_idx]
    submatrix = sigma_aug[np.ix_(sub_idx, sub_idx)]

    def ScaledTE(weights: np.ndarray):
        return 1e6 * RelativeTrackingError(weights, submatrix)
    
    K = len(selected_indices)
    
    w0 = np.ones(K) / K if w0 is None else w0

    res = minimize(
        ScaledTE,
        w0,
        method='SLSQP',
        bounds=[(0, None)] * K,
        constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    )

    if res.success:
        return (res.fun / 1e6) + (K * cost_factor), res.x
    else:
        print(f"Failed to resolve! {selected_indices=}")
        return float('inf'), np.zeros(K)
    

def adjust_weights(weights: np.ndarray, state: set, new_state: set):
    # Map stock index → its current weight
    weight_map = {stock: w for stock, w in zip(sorted(state), weights)}
    
    # Drop removed stocks, add new ones with weight 0
    for stock in (state - new_state):
        del weight_map[stock]
    for stock in (new_state - state):
        weight_map[stock] = 0.0
    
    # Return in sorted order, renormalised
    weights = np.array([weight_map[s] for s in sorted(new_state)])
    return weights / weights.sum() if weights.sum() > 0 else np.ones(len(new_state)) / len(new_state)


def RunSimulatedAnnealing(
    N: int,  # Size of universe
    sigma_aug: np.ndarray,
    n_steps: int = 4_000,  # Loops till termination
    T_init: float = 8.6e-6,  # Initial "temp"
    alpha: float = 0.999,  # Decay rate of temp
    K: int | None = None,  # Desired holding size
    cost_factor: float = 1e-6,  # Scale factor for cost of additional holding
    p_add: float = 0.33,  # Probability next move adds an index
    p_swap: float = 0.34,  # Probability next move swaps an index
    p_remove: float = 0.33,  # Probability next move removes an index
) -> tuple[set, float, dict]:
    # If set K change probabilities
    if K:
        p_add = 0
        p_remove = 0
        p_swap = 1
    
    assert abs(p_add + p_swap + p_remove - 1) < 1e-9, "Move probabilities must sum to one"

    # Setup initial conditions
    state = get_initial_state(N, K)
    score, current_weights = ScoreSelection(state, sigma_aug, cost_factor)
    T = T_init
    best_state, best_score = state, score

    # Setup analytics
    score_history = [best_score]
    acceptance_history = []
    accepted = 0
    total = 0

    for step in range(n_steps):
        new_state = propose_move(state, N, p_add, p_swap, p_remove)
        w0 = adjust_weights(current_weights, state, new_state)
        new_score, new_weights = ScoreSelection(new_state, sigma_aug, cost_factor, w0=w0)

        delta = new_score - score
        
        # If new score is better (or randomly)
        if delta < 0 or random.random() < np.exp(-delta / T):
            state, score = new_state, new_score
            current_weights = new_weights
            accepted += 1
            score_history.append(score)
            # Update best score if it is
            if score < best_score:
                best_state, best_score = state, score
                best_weights = current_weights

        total += 1
        
        if total == 50:
            acceptance_history.append(accepted / total)
            accepted = 0
            total = 0
        T *= alpha

    return best_state, best_weights, best_score, {
        "scoreHistory": score_history,
        "acceptanceHistory": acceptance_history,
    }


def probe_temperature(N, sigma_aug, n_probes=100, cost_factor=1e-6):
    state = get_initial_state(N)
    deltas = []
    for _ in range(n_probes):
        new_state = propose_move(state, N)
        s1, _ = ScoreSelection(state, sigma_aug, cost_factor)
        s2, _ = ScoreSelection(new_state, sigma_aug, cost_factor)
        deltas.append(abs(s2 - s1))
        state = new_state  # keep walking
    return np.mean(deltas) / 0.29  # Approx 75% initial acceptance