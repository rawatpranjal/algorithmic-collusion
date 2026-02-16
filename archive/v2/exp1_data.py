# generate_data.py
import numpy as np
import random
import pandas as pd
import os
from tqdm import tqdm
from joblib import Parallel, delayed
from datetime import datetime

# --- Constants ---
N_ACTIONS = 11
ACTION_SPACE = np.linspace(0, 1, N_ACTIONS)

def significance_stars(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    else:
        return ""

def get_rewards(bids, auction_type="first", r=0.0):
    """
    Generalized for multiple bidders, including reserve price and correct sale price.
    bids: array of shape (n_bidders,).
    auction_type: "first" or "second".
    r: reserve price.

    Returns:
      (rewards, winner, sale_price)
        - rewards: array of shape (n_bidders,) with payoff for each bidder
        - winner: integer index of the winning bidder, or None for no-sale
        - sale_price: the actual price paid by the winner
    """
    n_bidders = len(bids)
    valuations = np.ones(n_bidders)

    valid_bids_mask = bids >= r
    if not np.any(valid_bids_mask):
        return np.zeros(n_bidders), None, 0.0

    valid_bids = bids[valid_bids_mask]
    original_indices = np.where(valid_bids_mask)[0]

    sorted_valid_indices = np.argsort(valid_bids)[::-1]
    highest_bid = valid_bids[sorted_valid_indices[0]]

    # Find all bidders who tied for the highest valid bid
    winner_indices = original_indices[valid_bids == highest_bid]
    winner = np.random.choice(winner_indices)

    # Determine sale price
    if auction_type == "first":
        sale_price = highest_bid
    else:  # second-price
        if len(valid_bids) > 1:
            sale_price = valid_bids[sorted_valid_indices[1]]
        else: # Only one valid bid, winner pays reserve price
            sale_price = r
            
    rewards = np.zeros(n_bidders)
    rewards[winner] = valuations[winner] - sale_price

    return rewards, winner, sale_price


def build_state_space(median_opp_past_bid_index, winner_bid_index_state, n_actions):
    """
    Decide how many total states we have, based on:
      - median_opp_past_bid_index (bool)
      - winner_bid_index_state (bool)
      - n_actions = 6

    If both are False -> 1 state
    If exactly one is True -> n_actions states
    If both True -> n_actions * n_actions states

    We'll define a helper function to map (median_idx, winner_idx) -> single integer in [0..n_states-1].
    """
    if not median_opp_past_bid_index and not winner_bid_index_state:
        n_states = 1
    elif median_opp_past_bid_index and not winner_bid_index_state:
        n_states = n_actions
    elif not median_opp_past_bid_index and winner_bid_index_state:
        n_states = n_actions
    else:
        # both True
        n_states = n_actions * n_actions

    return n_states

def state_to_index(median_idx, winner_idx, median_flag, winner_flag, n_actions):
    """
    Convert the pair (median_idx, winner_idx) into a single integer.

    Cases:
     1) If both flags are False -> always return 0 (only 1 state).
     2) If exactly one flag is True -> return median_idx OR winner_idx as the state ID.
     3) If both are True -> we do a 2D -> 1D mapping: state_id = median_idx * n_actions + winner_idx
    """
    if not median_flag and not winner_flag:
        return 0
    elif median_flag and not winner_flag:
        return median_idx
    elif not median_flag and winner_flag:
        return winner_idx
    else:
        # both True
        return median_idx * n_actions + winner_idx

def run_experiment(alpha, gamma, episodes, auction_type, init, exploration,
                   asynchronous, n_bidders, median_opp_past_bid_index,
                   winner_bid_index_state, r, boltzmann_temp_start,
                   seed=0):
    """
    Runs a single experiment with corrected logic and full parameter space.
    """
    np.random.seed(seed)
    random.seed(seed)

    # State space setup
    n_states = build_state_space(median_opp_past_bid_index, winner_bid_index_state, N_ACTIONS)

    # Q-table initialization
    if init == "random":
        Q = np.random.rand(n_bidders, n_states, N_ACTIONS)
    else:
        Q = np.zeros((n_bidders, n_states, N_ACTIONS))

    # Trackers for state features
    prev_bids = np.zeros(n_bidders)
    prev_winner_bid = 0.0

    def bid_to_state_index(bid):
        return np.argmin(np.abs(ACTION_SPACE - bid))

    revenues = []
    window_size = 1000

    # Exploration decay schedules
    eps_start, eps_end = 1.0, 0.0
    temp_start, temp_end = boltzmann_temp_start, 0.01
    decay_end = int(0.9 * episodes)

    for ep in range(episodes):
        # --- 1. Determine Current State(s) ---
        current_states = np.zeros(n_bidders, dtype=int)
        winner_idx = bid_to_state_index(prev_winner_bid) if winner_bid_index_state else 0

        for i in range(n_bidders):
            median_idx = 0
            if median_opp_past_bid_index:
                # Median of *other* bidders' previous bids
                opp_bids = np.delete(prev_bids, i)
                median_val = np.median(opp_bids)
                median_idx = bid_to_state_index(median_val)
            current_states[i] = state_to_index(median_idx, winner_idx, median_opp_past_bid_index, winner_bid_index_state, N_ACTIONS)

        s = current_states # s is now a vector of states, one for each agent

        # --- 2. Choose Actions ---
        eps = eps_start - (ep / decay_end) * (eps_start - eps_end) if ep < decay_end else eps_end
        temp = temp_start - (ep / decay_end) * (temp_start - temp_end) if ep < decay_end else temp_end

        chosen_actions = []
        for i in range(n_bidders):
            if exploration == "egreedy":
                if np.random.rand() > eps:
                    q_values = Q[i, s[i]]
                    max_q = np.max(q_values)
                    best_actions = np.where(q_values == max_q)[0]
                    a_i = np.random.choice(best_actions)
                else:
                    a_i = np.random.randint(N_ACTIONS)
            elif exploration == "boltzmann":
                if temp <= 0: # Avoid division by zero
                    q_values = Q[i, s[i]]
                    max_q = np.max(q_values)
                    best_actions = np.where(q_values == max_q)[0]
                    a_i = np.random.choice(best_actions)
                else:
                    qvals = Q[i, s[i]] / temp
                    exp_q = np.exp(qvals - np.max(qvals))
                    probs = exp_q / np.sum(exp_q)
                    a_i = np.random.choice(range(N_ACTIONS), p=probs)
            else: # Default to egreedy
                if np.random.rand() > eps:
                    q_values = Q[i, s[i]]
                    max_q = np.max(q_values)
                    best_actions = np.where(q_values == max_q)[0]
                    a_i = np.random.choice(best_actions)
                else:
                    a_i = np.random.randint(N_ACTIONS)
            chosen_actions.append(a_i)

        bids = np.array([ACTION_SPACE[a_i] for a_i in chosen_actions])

        # --- 3. Get Rewards and Actual Sale Price ---
        rewards, winner, sale_price = get_rewards(bids, auction_type, r)

        # --- 4. Determine Next State(s) ---
        # Note: Winning bid for state is the actual bid, not the sale price
        winner_bid = bids[winner] if winner is not None else 0.0
        
        next_states = np.zeros(n_bidders, dtype=int)
        next_winner_idx = bid_to_state_index(winner_bid) if winner_bid_index_state else 0

        for i in range(n_bidders):
            next_median_idx = 0
            if median_opp_past_bid_index:
                opp_bids = np.delete(bids, i)
                next_median_val = np.median(opp_bids)
                next_median_idx = bid_to_state_index(next_median_val)
            next_states[i] = state_to_index(next_median_idx, next_winner_idx, median_opp_past_bid_index, winner_bid_index_state, N_ACTIONS)
        
        s_prime = next_states

        # --- 5. Q-Update ---
        if asynchronous == 1:
            for i in range(n_bidders):
                old_q = Q[i, s[i], chosen_actions[i]]
                max_next_q = np.max(Q[i, s_prime[i]])
                td_target = rewards[i] + gamma * max_next_q
                Q[i, s[i], chosen_actions[i]] = old_q + alpha * (td_target - old_q)
        else: # Synchronous
            for i in range(n_bidders):
                cf_rewards = np.zeros(N_ACTIONS)
                for a_alt in range(N_ACTIONS):
                    cf_bids = bids.copy()
                    cf_bids[i] = ACTION_SPACE[a_alt]
                    # Note: Using reserve price r here is critical for counterfactuals
                    cf_r, _, _ = get_rewards(cf_bids, auction_type, r)
                    cf_rewards[a_alt] = cf_r[i]
                
                # The next state s_prime[i] is the same for all counterfactuals
                # because it depends on the actual actions taken in the round.
                max_next_q = np.max(Q[i, s_prime[i]])
                Q[i, s[i], :] = (1 - alpha) * Q[i, s[i], :] + alpha * (cf_rewards + gamma * max_next_q)

        # --- 6. Log and Update State Trackers ---
        revenues.append(sale_price)
        prev_bids = bids
        prev_winner_bid = winner_bid

    # --- Final Stats ---
    if not revenues: # Handle case of no sales ever occurring
        return {
            "avg_rev_last_1000": 0,
            "time_to_converge": episodes,
            "avg_regret_of_seller": 1.0
        }

    avg_rev_last_1000 = np.mean(revenues[-window_size:]) if len(revenues) >= window_size else np.mean(revenues)

    # Efficiently calculate time_to_converge
    time_to_converge = episodes
    if len(revenues) > window_size:
        rev_series = pd.Series(revenues)
        roll_avg = rev_series.rolling(window_size, min_periods=1).mean()
        final_rev = avg_rev_last_1000
        lower_band = 0.95 * final_rev
        upper_band = 1.05 * final_rev
        
        # Find first time the rolling average enters the band and stays there
        in_band = (roll_avg >= lower_band) & (roll_avg <= upper_band)
        # We need the first index of a sequence of `True`
        first_stable_index = next(i for i, x in enumerate(in_band) if x)
        if first_stable_index < len(in_band):
            time_to_converge = first_stable_index
    
    avg_regret_of_seller = np.mean([1.0 - r for r in revenues])

    return {
        "avg_rev_last_1000": avg_rev_last_1000,
        "time_to_converge": time_to_converge / episodes, # Normalize here
        "avg_regret_of_seller": avg_regret_of_seller
    }


if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create a unique, timestamped output directory inside the script's directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(script_dir, f"experiment1_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save the path to this run for the analysis script to use
    with open(os.path.join(script_dir, "LATEST_RUN.txt"), "w") as f:
        f.write(output_dir)

    param_space = {
        "alpha": [0.001, 0.005, 0.01, 0.05, 0.1],
        "gamma": [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
        "auction_type": ["first", "second"],
        "init": ["random", "zeros"],
        "exploration": ["egreedy", "boltzmann"],
        "asynchronous": [0, 1],
        "n_bidders": [2, 4, 6],
        "median_opp_past_bid_index": [False, True],
        "winner_bid_index_state": [False, True],
        "r": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "boltzmann_temp_start": [0.1, 0.5, 1.0, 2.0]
    }

    K = 10
    all_params = []
    for k in range(K):
        params = {key: random.choice(values) for key, values in param_space.items()}
        params["episodes"] = int(random.uniform(10_000, 100_000))
        params["seed"] = k
        all_params.append(params)

    print(f"Running {K} experiments in parallel...")
    results = Parallel(n_jobs=-1)(
        delayed(run_experiment)(**p) for p in tqdm(all_params)
    )

    # Combine results with parameters for the final dataframe
    final_results = []
    for params, outcome in zip(all_params, results):
        res = params.copy()
        res.update(outcome)
        final_results.append(res)

    df = pd.DataFrame(final_results)
    csv_path = os.path.join(output_dir, "data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Data generation complete. Saved to '{csv_path}'")