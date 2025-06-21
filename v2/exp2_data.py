#!/usr/bin/env python3

import numpy as np
import pandas as pd
import random
import os
from tqdm import tqdm
from joblib import Parallel, delayed
from datetime import datetime

# --- Constants ---
N_BID_BINS = 11  # For discretizing bids and state space for bids
N_VAL_BINS = 11  # For discretizing signals and state space for signals
ACTION_SPACE = np.linspace(0, 1, N_BID_BINS)

def get_valuation(eta, own_signal, others_signals):
    # When there's only one bidder, there are no "other" signals.
    if not isinstance(others_signals, (list, np.ndarray)) or len(others_signals) == 0:
        return own_signal
    alpha = 1.0 - 0.5 * eta
    beta = 0.5 * eta
    return alpha * own_signal + beta * np.mean(others_signals)

def get_payoffs(bids, valuations, auction_type, r=0.0):
    n_bidders = len(bids)
    valid_bids_mask = bids >= r
    if not np.any(valid_bids_mask):
        # No-sale event
        return np.zeros(n_bidders), None, 0.0, 0.0

    valid_bids = bids[valid_bids_mask]
    original_indices = np.where(valid_bids_mask)[0]
    
    sorted_valid_indices = np.argsort(valid_bids)[::-1]
    highest_bid = valid_bids[sorted_valid_indices[0]]

    winner_indices = original_indices[valid_bids == highest_bid]
    winner = np.random.choice(winner_indices)

    if auction_type == "first":
        sale_price = highest_bid
    else:  # second-price
        if len(valid_bids) > 1:
            sale_price = valid_bids[sorted_valid_indices[1]]
        else:
            sale_price = r
            
    rewards = np.zeros(n_bidders)
    rewards[winner] = valuations[winner] - sale_price
    
    # The winning bid itself (for state tracking) is different from sale price in 2nd price auction
    winner_bid = bids[winner]
    return rewards, winner, sale_price, winner_bid

def build_state(own_signal, median_opp_bid, past_winning_bid, median_flag, winner_flag):
    signal_idx = int(own_signal * (N_VAL_BINS - 1))
    
    median_idx = 0
    if median_flag:
        median_idx = int(median_opp_bid * (N_BID_BINS - 1))

    winner_idx = 0
    if winner_flag:
        winner_idx = int(past_winning_bid * (N_BID_BINS - 1))

    # Calculate unique state index
    m_size = N_BID_BINS if median_flag else 1
    
    idx = signal_idx
    if median_flag:
        idx = idx * m_size + median_idx
    if winner_flag:
        w_size = N_BID_BINS if winner_flag else 1
        idx = idx * w_size + winner_idx
        
    return idx

def run_experiment(eta, auction_type, alpha, gamma, episodes,
                   init, exploration, asynchronous, n_bidders,
                   median_opp_past_bid_index, winner_bid_index_state, r,
                   boltzmann_temp_start, seed=0):
    np.random.seed(seed)
    random.seed(seed)

    m_size = N_BID_BINS if median_opp_past_bid_index else 1
    w_size = N_BID_BINS if winner_bid_index_state else 1
    n_states = N_VAL_BINS * m_size * w_size

    if init == "random":
        Q = np.random.rand(n_bidders, n_states, N_BID_BINS)
    else:
        Q = np.zeros((n_bidders, n_states, N_BID_BINS))

    eps_start, eps_end = 1.0, 0.0
    temp_start, temp_end = boltzmann_temp_start, 0.01
    decay_end = int(0.9 * episodes)

    revenues = []
    past_bids = np.zeros(n_bidders)
    past_winner_bid = 0.0
    window_size = 1000

    for ep in range(episodes):
        eps = eps_start - (ep / decay_end) * (eps_start - eps_end) if ep < decay_end else eps_end
        temp = temp_start - (ep / decay_end) * (temp_start - temp_end) if ep < decay_end else temp_end

        signals = np.random.randint(0, N_VAL_BINS, size=n_bidders) / (N_VAL_BINS - 1)
        valuations = np.array([get_valuation(eta, signals[i], np.delete(signals, i)) for i in range(n_bidders)])

        current_states = np.zeros(n_bidders, dtype=int)
        for i in range(n_bidders):
            median_opp_bid = np.median(np.delete(past_bids, i)) if n_bidders > 1 else 0
            current_states[i] = build_state(signals[i], median_opp_bid, past_winner_bid, median_opp_past_bid_index, winner_bid_index_state)
        s = current_states

        chosen_actions = []
        for i in range(n_bidders):
            if np.random.rand() > eps:
                q_values = Q[i, s[i]]
                max_q = np.max(q_values)
                best_actions = np.where(q_values == max_q)[0]
                a_i = np.random.choice(best_actions)
            else:
                a_i = np.random.randint(N_BID_BINS)
            chosen_actions.append(a_i)
        
        bids = np.array([ACTION_SPACE[a] for a in chosen_actions])
        rewards, winner, sale_price, winner_bid = get_payoffs(bids, valuations, auction_type, r)

        next_states = np.zeros(n_bidders, dtype=int)
        for i in range(n_bidders):
            # The next state is determined by the actions taken in this round
            # and a *new* draw of signals for the next round.
            next_signals = np.random.randint(0, N_VAL_BINS, size=n_bidders) / (N_VAL_BINS - 1)
            
            # Robustly calculate median of opponent bids from this round
            opp_bids = np.delete(bids, i)
            median_opp_bid_next = np.median(opp_bids) if opp_bids.size > 0 else 0.0

            winner_bid_next = winner_bid if winner is not None else 0.0
            next_states[i] = build_state(next_signals[i], median_opp_bid_next, winner_bid_next, median_opp_past_bid_index, winner_bid_index_state)
        s_prime = next_states
        
        if asynchronous == 1:
            for i in range(n_bidders):
                old_q = Q[i, s[i], chosen_actions[i]]
                max_next_q = np.max(Q[i, s_prime[i]])
                td_target = rewards[i] + gamma * max_next_q
                Q[i, s[i], chosen_actions[i]] = old_q + alpha * (td_target - old_q)
        else: # Synchronous updates
            for i in range(n_bidders):
                cf_rewards = np.zeros(N_BID_BINS)
                for a_alt in range(N_BID_BINS):
                    cf_bids = bids.copy()
                    cf_bids[i] = ACTION_SPACE[a_alt]
                    cf_r, _, _, _ = get_payoffs(cf_bids, valuations, auction_type, r)
                    cf_rewards[a_alt] = cf_r[i]
                
                max_next_q = np.max(Q[i, s_prime[i]])
                Q[i, s[i], :] = (1 - alpha) * Q[i, s[i], :] + alpha * (cf_rewards + gamma * max_next_q)

        revenues.append(sale_price)
        past_bids = bids
        past_winner_bid = winner_bid if winner is not None else 0.0

    avg_rev_last_1000 = np.mean(revenues[-window_size:])
    avg_regret_of_seller = np.mean([1.0 - r for r in revenues])

    time_to_converge = episodes
    if len(revenues) > window_size:
        rev_series = pd.Series(revenues)
        roll_avg = rev_series.rolling(window_size, min_periods=1).mean()
        final_rev = avg_rev_last_1000
        lower_band, upper_band = 0.95 * final_rev, 1.05 * final_rev
        in_band = (roll_avg >= lower_band) & (roll_avg <= upper_band)
        last_out_of_band = in_band.where(~in_band).last_valid_index()
        if last_out_of_band is None:
            time_to_converge = window_size
        else:
            first_stable_index = last_out_of_band + 1
            if first_stable_index < len(in_band):
                time_to_converge = first_stable_index
    
    return {
        "avg_rev_last_1000": avg_rev_last_1000,
        "time_to_converge": time_to_converge / episodes,
        "avg_regret_of_seller": avg_regret_of_seller
    }

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(script_dir, f"experiment2_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(script_dir, "LATEST_RUN.txt"), "w") as f:
        f.write(output_dir)

    param_space = {
        "eta": [0.0, 0.25, 0.5, 0.75, 1.0],
        "alpha": [0.001, 0.005, 0.01, 0.05, 0.1],
        "gamma": [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
        "auction_type": ["first", "second"],
        "init": ["random", "zeros"],
        "exploration": ["egreedy"], # Simplified for now
        "asynchronous": [1], # Simplified for now
        "n_bidders": [2, 4, 6],
        "median_opp_past_bid_index": [False, True],
        "winner_bid_index_state": [False, True],
        "r": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "boltzmann_temp_start": [1.0] # Not used with egreedy
    }

    K = 10 # Smaller run for testing
    all_params = []
    for k in range(K):
        params = {key: random.choice(values) for key, values in param_space.items()}
        params["episodes"] = int(random.uniform(10_000, 50_000))
        params["seed"] = k
        all_params.append(params)

    print(f"Running {K} experiments for Experiment 2 in parallel...")
    results = Parallel(n_jobs=-1)(
        delayed(run_experiment)(**p) for p in tqdm(all_params)
    )

    final_results = []
    for params, outcome in zip(all_params, results):
        res = params.copy()
        res.update(outcome)
        final_results.append(res)

    df = pd.DataFrame(final_results)
    csv_path = os.path.join(output_dir, "data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Data generation complete. Saved to '{csv_path}'.")