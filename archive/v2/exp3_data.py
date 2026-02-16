#!/usr/bin/env python3

import numpy as np
import pandas as pd
import random
import os
from tqdm import tqdm
from joblib import Parallel, delayed
from datetime import datetime

# --- Constants ---
N_BID_BINS = 11
N_VAL_BINS = 11
ACTION_SPACE = np.linspace(0, 1, N_BID_BINS)

DEBUG = False
def dprint(*args):
    if DEBUG:
        print(*args)

# --------------------------------------------------------------
# 1) Valuation with eta (normalized)
# --------------------------------------------------------------
def get_valuation(eta, own_signal, others_signals):
    if not isinstance(others_signals, np.ndarray) or others_signals.size == 0:
        return own_signal
    return (1.0 - 0.5 * eta) * own_signal + (0.5 * eta) * np.mean(others_signals)

# --------------------------------------------------------------
# 2) Payoffs
# --------------------------------------------------------------
def get_payoffs(bids, valuations, auction_type, r):
    n_bidders = len(bids)
    valid_bids_mask = bids >= r
    if not np.any(valid_bids_mask):
        return np.zeros(n_bidders), None, 0.0, 0.0

    valid_bids = bids[valid_bids_mask]
    original_indices = np.where(valid_bids_mask)[0]
    
    sorted_valid_indices = np.argsort(valid_bids)[::-1]
    highest_bid = valid_bids[sorted_valid_indices[0]]
    winner_indices = original_indices[valid_bids == highest_bid]
    winner = np.random.choice(winner_indices)

    if auction_type == 'first':
        sale_price = highest_bid
    else: # second-price
        sale_price = valid_bids[sorted_valid_indices[1]] if len(valid_bids) > 1 else r
            
    rewards = np.zeros(n_bidders)
    rewards[winner] = valuations[winner] - sale_price
    return rewards, winner, sale_price, bids[winner]

# --------------------------------------------------------------
# 3) Bandit helpers: UCB and Linear (Contextual)
# --------------------------------------------------------------
class UCBBandit:
    def __init__(self, n_actions, c):
        self.n_actions = n_actions
        self.c = c
        self.counts = np.zeros(n_actions)
        self.sums = np.zeros(n_actions)
        self.total_pulls = 0

    def select_action(self):
        untried = np.where(self.counts == 0)[0]
        if len(untried) > 0:
            return np.random.choice(untried)
        avg = self.sums / self.counts
        ucb = avg + self.c * np.sqrt(np.log(self.total_pulls) / self.counts)
        return np.argmax(ucb)

    def update(self, action, reward):
        self.counts[action] += 1
        self.sums[action] += reward
        self.total_pulls += 1

class LinearContextualBandit:
    def __init__(self, n_actions, context_dim, c, reg=1.0):
        self.n_actions = n_actions
        self.c = c
        self.A = [reg * np.eye(context_dim) for _ in range(n_actions)]
        self.b = [np.zeros(context_dim) for _ in range(n_actions)]

    def select_action(self, context):
        ucb_scores = []
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            mean_est = theta @ context
            bonus = self.c * np.sqrt(context.T @ A_inv @ context)
            ucb_scores.append(mean_est + bonus)
        return np.argmax(ucb_scores)

    def update(self, action, context, reward):
        self.A[action] += np.outer(context, context)
        self.b[action] += context * reward

# --------------------------------------------------------------
# 4) Run bandit experiment (replaces Q-learning)
# --------------------------------------------------------------
def run_bandit_experiment(
    eta, auction_type, bandit_type, c, n_bidders, r, reg,
    seed=0, max_rounds=50000
):
    np.random.seed(seed)
    random.seed(seed)

    context_dim = 3 # own_signal, last_median_bid, last_winning_bid
    if bandit_type == "ucb":
        bandits = [UCBBandit(N_BID_BINS, c) for _ in range(n_bidders)]
    else: # 'contextual'
        bandits = [LinearContextualBandit(N_BID_BINS, context_dim, c, reg) for _ in range(n_bidders)]

    revenues = []
    past_bids = np.zeros(n_bidders)
    past_winner_bid = 0.0

    for _ in range(max_rounds):
        signals = np.random.rand(n_bidders)
        valuations = np.array([get_valuation(eta, signals[i], np.delete(signals, i)) for i in range(n_bidders)])
        
        contexts = [np.array([signals[i], np.median(np.delete(past_bids, i)) if n_bidders > 1 else 0, past_winner_bid]) for i in range(n_bidders)]

        actions_taken = [b.select_action() if bandit_type == "ucb" else b.select_action(c) for i, (b, c) in enumerate(zip(bandits, contexts))]
        bids = np.array([ACTION_SPACE[a] for a in actions_taken])
        
        rewards, winner, sale_price, winner_bid = get_payoffs(bids, valuations, auction_type, r)
        
        if winner is not None:
            for i, (a, ctx) in enumerate(zip(actions_taken, contexts)):
                bandit_reward = rewards[i]
                bandits[i].update(a, bandit_reward) if bandit_type == "ucb" else bandits[i].update(a, ctx, bandit_reward)

        revenues.append(sale_price)
        past_bids = bids
        past_winner_bid = winner_bid if winner is not None else 0.0

    window_size = 1000
    avg_rev = np.mean(revenues[-window_size:])
    avg_regret = np.mean([1.0 - rev for rev in revenues])

    time_to_converge = max_rounds
    if len(revenues) > window_size:
        rev_series = pd.Series(revenues)
        roll_avg = rev_series.rolling(window_size, min_periods=1).mean()
        final_rev = avg_rev
        lower_band, upper_band = 0.95 * final_rev, 1.05 * final_rev
        in_band = (roll_avg >= lower_band) & (roll_avg <= upper_band)
        last_out_of_band = in_band.where(~in_band).last_valid_index()
        if last_out_of_band is None:
            time_to_converge = window_size
        else:
            first_stable_index = last_out_of_band + 1
            if first_stable_index < len(in_band):
                time_to_converge = first_stable_index

    return avg_rev, time_to_converge / max_rounds, avg_regret

# --------------------------------------------------------------
# 5) Main experiment loop
# --------------------------------------------------------------
def main_experiment(K=50):
    results = []
    auction_type_options = ["first", "second"]
    bandit_type_options = ["ucb", "contextual"]

    for seed in trange(K, desc="Generating experiments"):
        eta = random.uniform(0.0, 1.0)
        c = random.uniform(0.01, 2.0)  # exploration parameter
        n_bidders = random.choice([2, 4, 6])
        bandit_type = random.choice(bandit_type_options)
        auction_type = random.choice(auction_type_options)

        avg_rev, time_to_converge, avg_regret = run_bandit_experiment(
            eta=eta,
            auction_type=auction_type,
            bandit_type=bandit_type,
            c=c,
            n_bidders=n_bidders,
            seed=seed
        )

        results.append({
            "eta": eta,
            "c": c,
            "auction_type": auction_type,
            "bandit_type": bandit_type,
            "n_bidders": n_bidders,
            "avg_rev": avg_rev,
            "time_to_converge": time_to_converge,
            "avg_regret_seller": avg_regret
        })

    df = pd.DataFrame(results)
    return df

# --------------------------------------------------------------
# 6) Run and save data
# --------------------------------------------------------------
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(script_dir, f"experiment3_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(script_dir, "LATEST_RUN.txt"), "w") as f:
        f.write(output_dir)

    param_space = {
        "eta": [0.0, 0.25, 0.5, 0.75, 1.0],
        "c": [0.1, 0.5, 1.0, 2.0],
        "reg": [0.1, 1.0, 10.0],
        "auction_type": ["first", "second"],
        "bandit_type": ["ucb", "contextual"],
        "n_bidders": [2, 4, 6],
        "r": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    }

    K = 100 # For testing
    all_params = []
    for k in range(K):
        params = {key: random.choice(values) for key, values in param_space.items()}
        params["seed"] = k
        all_params.append(params)
    
    print(f"Running {K} experiments for Experiment 3 in parallel...")
    results = Parallel(n_jobs=-1)(
        delayed(run_bandit_experiment)(**p) for p in tqdm(all_params)
    )

    final_results = []
    for params, outcome in zip(all_params, results):
        res = params.copy()
        res["avg_rev"], res["time_to_converge"], res["avg_regret_seller"] = outcome
        final_results.append(res)
        
    df = pd.DataFrame(final_results)
    csv_path = os.path.join(output_dir, "data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Data generation complete. Saved to '{csv_path}'.")
