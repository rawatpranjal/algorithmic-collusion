#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import json
import argparse
import matplotlib.pyplot as plt
from tqdm import trange

# ---------------------------------------------------------------------
# 1) Parameter Mappings
# ---------------------------------------------------------------------
param_mappings = {
    "auction_type": {"first": 1, "second": 0},
    "init": {"random": 0, "zeros": 1},
    "exploration": {"egreedy": 0, "boltzmann": 1},
    "asynchronous": {0: 0, 1: 1},
    "median_opp_past_bid_index": {False: 0, True: 1},
    "winner_bid_index_state": {False: 0, True: 1}
}

# ---------------------------------------------------------------------
# 2) Payoff with Reserve Price
# ---------------------------------------------------------------------
def get_rewards(bids, valuations, auction_type="first", reserve_price=0.0):
    """
    Returns:
      rewards: array of shape (n_bidders,)
      winner_global: index of the winner or -1 if no valid (>= reserve_price) bids
      winner_bid: the actual winning bid (0.0 if no sale)
    """
    n_bidders = len(bids)
    rewards = np.zeros(n_bidders)

    valid_indices = np.where(bids >= reserve_price)[0]
    if len(valid_indices) == 0:
        # No sale if nobody meets reserve
        return rewards, -1, 0.0

    valid_bids = bids[valid_indices]
    sorted_indices = np.argsort(valid_bids)[::-1]
    highest_idx_local = [sorted_indices[0]]
    highest_bid = valid_bids[sorted_indices[0]]

    # Tie-break among top
    for idx_l in sorted_indices[1:]:
        if np.isclose(valid_bids[idx_l], highest_bid):
            highest_idx_local.append(idx_l)
        else:
            break

    # Resolve tie randomly
    if len(highest_idx_local) > 1:
        winner_local = np.random.choice(highest_idx_local)
    else:
        winner_local = highest_idx_local[0]
    winner_global = valid_indices[winner_local]
    winner_bid = bids[winner_global]

    # Second-highest among valid
    if len(valid_indices) == len(highest_idx_local):
        second_highest_bid = highest_bid
    else:
        second_idx_local = None
        for idx_l in sorted_indices:
            if idx_l not in highest_idx_local:
                second_idx_local = idx_l
                break
        second_highest_bid = highest_bid if second_idx_local is None else valid_bids[second_idx_local]

    # Auction payoff
    if auction_type == "first":
        rewards[winner_global] = valuations[winner_global] - winner_bid
    else:
        rewards[winner_global] = valuations[winner_global] - second_highest_bid

    return rewards, winner_global, winner_bid

# ---------------------------------------------------------------------
# 3) Valuation with eta
# ---------------------------------------------------------------------
def get_valuation(eta, own_signal, others_signals):
    """
    Linear affiliation: 
      valuation = alpha * own_signal + beta * mean(others_signals),
    where alpha = 1 - 0.5*eta, beta = 0.5*eta.
    """
    alpha = 1.0 - 0.5 * eta
    beta = 0.5 * eta
    return alpha * own_signal + beta * np.mean(others_signals)

# ---------------------------------------------------------------------
# 4) Theoretical Revenue (Linear-Affiliation)
# ---------------------------------------------------------------------
def simulate_linear_affiliation_revenue(N, eta, auction_type, M=50_000):
    """
    Approximates BNE-based average revenue for 'first' or 'second' auctions
    in a linear-affiliation model, via Monte Carlo.
    """
    alpha = 1.0 - 0.5 * eta
    beta = 0.5 * eta / max(N - 1, 1)
    if auction_type == "first":
        # factor = (N-1)/N * [alpha + (N/2)*beta]
        factor = ((N - 1) / float(N)) * (alpha + (N / 2.0) * beta)
    else:  # second
        # factor = alpha + (N/2)*beta
        factor = alpha + (N / 2.0) * beta

    rev_sum = 0.0
    for _ in range(M):
        t = np.random.rand(N)  # signals in [0..1]
        bids = factor * t
        max_bid = np.max(bids)
        top_idx = np.where(np.isclose(bids, max_bid))[0]
        winner = np.random.choice(top_idx)

        if auction_type == "first":
            price = bids[winner]
        else:
            if len(top_idx) == 1:
                tmp = np.delete(bids, winner)
                price = np.max(tmp) if len(tmp) else 0.0
            else:
                price = max_bid
        rev_sum += price

    return rev_sum / M

# ---------------------------------------------------------------------
# 5) Q-Learning: State-Space
# ---------------------------------------------------------------------
def build_state_space(median_opp_past_bid_index, winner_bid_index_state, n_actions):
    """
    If both flags= False -> 1
    If exactly one= True -> n_actions
    If both True -> n_actions * n_actions
    """
    if not median_opp_past_bid_index and not winner_bid_index_state:
        return 1
    elif median_opp_past_bid_index and not winner_bid_index_state:
        return n_actions
    elif not median_opp_past_bid_index and winner_bid_index_state:
        return n_actions
    else:
        return n_actions * n_actions

# ---------------------------------------------------------------------
# 6) Single Q-Learning Experiment
# ---------------------------------------------------------------------
def run_experiment(
    eta,
    auction_type,
    alpha, gamma, episodes,
    init, exploration, asynchronous, n_bidders,
    median_opp_past_bid_index, winner_bid_index_state,
    reserve_price, seed=0,
    store_qtables=False, qtable_folder=None,
    progress_callback=None
):
    """
    Q-learning experiment with:
      - signals in [0..1]
      - valuations ~ get_valuation(...)
      - n_bid_bins=6 for discrete bids
      - unify with Exp.1 logic:
        => track winners/no-sale
        => time_to_converge requires remain in ±5% band
        => store Q table snapshots every 1000 episodes
        => final summary includes:
           avg_rev_last_1000, time_to_converge,
           avg_regret_of_seller, no_sale_rate,
           price_volatility, winner_entropy
    """
    np.random.seed(seed)

    n_val_bins = 6
    n_bid_bins = 6
    action_space = np.linspace(0, 1, n_bid_bins)
    n_states = build_state_space(median_opp_past_bid_index, winner_bid_index_state, n_bid_bins)

    # Initialize Q
    if init == "random":
        Q = np.random.rand(n_bidders, n_states, n_bid_bins)
    else:
        Q = np.zeros((n_bidders, n_states, n_bid_bins))

    # Stats
    revenues = []
    winning_bids_list = []
    round_history = []
    no_sale_count = 0
    winners_list = []

    eps_start, eps_end = 1.0, 0.0
    decay_end = int(0.9 * episodes)

    past_bids = np.zeros(n_bidders)
    past_winner_bid = 0.0

    save_interval = 1000
    if store_qtables and qtable_folder is not None:
        os.makedirs(qtable_folder, exist_ok=True)

    # Function to convert (median_val, winner_val) -> state index
    def state_index(median_val, winner_val):
        if median_opp_past_bid_index:
            med_idx = int(median_val * (n_bid_bins - 1) + 0.4999999)
        else:
            med_idx = 0

        if winner_bid_index_state:
            win_idx = int(winner_val * (n_bid_bins - 1) + 0.4999999)
        else:
            win_idx = 0

        if (not median_opp_past_bid_index) and (not winner_bid_index_state):
            return 0
        elif median_opp_past_bid_index and (not winner_bid_index_state):
            return med_idx
        elif (not median_opp_past_bid_index) and winner_bid_index_state:
            return win_idx
        else:
            return med_idx * n_bid_bins + win_idx

    for ep in range(episodes):
        # Progress callback every 1000 episodes
        if progress_callback and ep % 1000 == 0:
            progress_callback(current=ep, total=episodes)

        # Epsilon decay
        if ep < decay_end:
            eps = eps_start - (ep / decay_end) * (eps_start - eps_end)
        else:
            eps = eps_end

        # signals
        signals = np.random.randint(n_val_bins, size=n_bidders) / (n_val_bins - 1)
        valuations = np.zeros(n_bidders)
        for i in range(n_bidders):
            others = np.delete(signals, i)
            valuations[i] = get_valuation(eta, signals[i], others)

        # current state
        median_val = np.median(past_bids) if median_opp_past_bid_index else 0.0
        s = state_index(median_val, past_winner_bid)

        # pick actions
        chosen_actions = []
        for i in range(n_bidders):
            qvals = Q[i, s]
            if exploration == "boltzmann":
                # Boltzmann
                shifted = qvals - np.max(qvals)
                ex = np.exp(shifted)
                probs = ex / np.sum(ex)
                a_i = np.random.choice(n_bid_bins, p=probs)
            else:
                # E-greedy: if rand() > eps => exploit
                if np.random.rand() > eps:
                    a_i = np.argmax(qvals)
                else:
                    a_i = np.random.randint(n_bid_bins)
            chosen_actions.append(a_i)

        # Convert actions to bids
        bids = np.array([action_space[a] for a in chosen_actions])

        # Auction payoff
        rew, winner, winner_bid_val = get_rewards(bids, valuations, auction_type, reserve_price)

        # Seller revenue
        valid_bids = bids[bids >= reserve_price]
        revenue_t = np.max(valid_bids) if len(valid_bids) > 0 else 0.0
        revenues.append(revenue_t)
        winning_bids_list.append(winner_bid_val)

        if winner == -1:
            no_sale_count += 1
        else:
            winners_list.append(winner)

        # Next state
        next_winner_val = winner_bid_val if winner != -1 else 0.0
        next_median_val = np.median(bids) if median_opp_past_bid_index else 0.0
        s_next = state_index(next_median_val, next_winner_val)

        # Q-update
        if asynchronous == 1:
            # Asynchronous
            for i in range(n_bidders):
                old_q = Q[i, s, chosen_actions[i]]
                td_target = rew[i] + gamma * np.max(Q[i, s_next])
                Q[i, s, chosen_actions[i]] = old_q + alpha * (td_target - old_q)
        else:
            # Synchronous
            for i in range(n_bidders):
                cf_rewards = np.zeros(n_bid_bins)
                for alt_a in range(n_bid_bins):
                    alt_bids = bids.copy()
                    alt_bids[i] = action_space[alt_a]
                    alt_r, _, _ = get_rewards(alt_bids, valuations, auction_type, reserve_price)
                    cf_rewards[alt_a] = alt_r[i]
                best_future = np.max(Q[i, s_next])
                Q[i, s, :] = (1 - alpha)*Q[i, s, :] + alpha*(cf_rewards + gamma * best_future)

        # Log each episode
        for i in range(n_bidders):
            round_history.append({
                "episode": ep,
                "bidder_id": i,
                "bid": bids[i],
                "reward": rew[i],
                "is_winner": (i == winner),
                "valuation": valuations[i]
            })

        # Advance
        past_bids = bids
        past_winner_bid = winner_bid_val if winner != -1 else 0.0

        # Save Q snapshot every 1000 episodes
        if store_qtables and (ep % save_interval == 0) and (qtable_folder is not None):
            snap_path = os.path.join(qtable_folder, f"q_after_{ep}.npy")
            np.save(snap_path, Q)

    # Final progress update
    if progress_callback:
        progress_callback(current=episodes, total=episodes)

    # final Q
    if store_qtables and (qtable_folder is not None):
        final_path = os.path.join(qtable_folder, "q_after_final.npy")
        np.save(final_path, Q)

    # Summaries: replicate Exp.1 approach
    window_size = 1000
    import pandas as pd
    if len(revenues) >= window_size:
        avg_rev_last_1000 = np.mean(revenues[-window_size:])
    else:
        avg_rev_last_1000 = np.mean(revenues)

    rev_series = pd.Series(revenues)
    roll_avg = rev_series.rolling(window=window_size).mean()
    final_rev = avg_rev_last_1000
    lower_band = 0.95 * final_rev
    upper_band = 1.05 * final_rev
    time_to_converge = episodes
    for t in range(len(revenues) - window_size):
        window_val = roll_avg.iloc[t + window_size - 1]
        if lower_band <= window_val <= upper_band:
            stay_in_band = True
            for j in range(t + window_size, len(revenues) - window_size):
                v_j = roll_avg.iloc[j + window_size - 1]
                if not (lower_band <= v_j <= upper_band):
                    stay_in_band = False
                    break
            if stay_in_band:
                time_to_converge = t + window_size
                break

    regrets = [1.0 - r for r in revenues]
    avg_regret_of_seller = np.mean(regrets)
    no_sale_rate = no_sale_count / episodes
    price_volatility = np.std(winning_bids_list)
    if len(winners_list) == 0:
        winner_entropy = 0.0
    else:
        unique_winners, counts = np.unique(winners_list, return_counts=True)
        p = counts / counts.sum()
        winner_entropy = -np.sum(p * np.log(p + 1e-12))

    summary = {
        "avg_rev_last_1000": avg_rev_last_1000,
        "time_to_converge": time_to_converge,
        "avg_regret_of_seller": avg_regret_of_seller,
        "no_sale_rate": no_sale_rate,
        "price_volatility": price_volatility,
        "winner_entropy": winner_entropy
    }
    return summary, revenues, round_history, Q

# ---------------------------------------------------------------------
# 7) Full Orchestrator: run_full_experiment
# ---------------------------------------------------------------------
def run_full_experiment(
    experiment_id=2,
    K=300,
    eta_values=[0.0, 0.25, 0.5, 0.75, 1.0],
    alpha_values=[0.001, 0.005, 0.01, 0.05, 0.1],
    gamma_values=[0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
    episodes_values=[100_000],
    reserve_price_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    init_values=["random", "zeros"],
    exploration_values=["egreedy", "boltzmann"],
    async_values=[0, 1],
    n_bidders_values=[2, 4, 6],
    median_flags=[False, True],
    winner_flags=[False, True],
    seed=42,
    output_dir=None
):
    """
    Orchestrates Experiment 2, sampling discrete sets for
    (eta, alpha, gamma, episodes, reserve_price, etc.), then running
    'first' & 'second' auctions. Also calls the theoretical revenue
    function. Stores final data in output_dir.
    """
    folder_name = output_dir if output_dir else f"results/exp{experiment_id}"
    os.makedirs(folder_name, exist_ok=True)

    # Save param mappings
    with open(os.path.join(folder_name, "param_mappings.json"), "w") as f:
        json.dump(param_mappings, f, indent=2)

    trial_folder = os.path.join(folder_name, "trials")
    os.makedirs(trial_folder, exist_ok=True)

    q_tables_folder = os.path.join(folder_name, "q_tables")
    os.makedirs(q_tables_folder, exist_ok=True)

    # Cache for theoretical revenue
    theory_cache = {}

    results = []
    rng = np.random.default_rng(seed)

    # Outer loop: run K experiments
    for run_id in trange(K, desc="Overall Runs"):
        eta = rng.choice(eta_values)
        alpha = rng.choice(alpha_values)
        gamma = rng.choice(gamma_values)
        episodes = rng.choice(episodes_values)
        reserve_price = rng.choice(reserve_price_values)
        init_str = rng.choice(init_values)
        exploration_str = rng.choice(exploration_values)
        async_val = rng.choice(async_values)
        n_bidders_val = rng.choice(n_bidders_values)
        median_flag = rng.choice(median_flags)
        winner_flag = rng.choice(winner_flags)

        # We'll do both first- & second-price auctions
        for auction_type_str in ["first", "second"]:
            # Check if theoretical revenue is cached
            cache_key = (n_bidders_val, eta, auction_type_str)
            if cache_key not in theory_cache:
                rev_theory = simulate_linear_affiliation_revenue(n_bidders_val, eta, auction_type_str)
                theory_cache[cache_key] = rev_theory
            else:
                rev_theory = theory_cache[cache_key]

            # Folder to store Q-tables
            q_run_folder = os.path.join(q_tables_folder, f"trial_{run_id}_{auction_type_str}")

            # Run Q-learning
            summary_out, rev_list, round_hist, Q_table = run_experiment(
                eta=eta,
                auction_type=auction_type_str,
                alpha=alpha,
                gamma=gamma,
                episodes=episodes,
                init=init_str,
                exploration=exploration_str,
                asynchronous=async_val,
                n_bidders=n_bidders_val,
                median_opp_past_bid_index=median_flag,
                winner_bid_index_state=winner_flag,
                reserve_price=reserve_price,
                seed=run_id,
                store_qtables=True,
                qtable_folder=q_run_folder
            )

            # Build round-level logs
            import pandas as pd
            for row in round_hist:
                row["run_id"] = run_id
                row["eta"] = eta
                row["alpha"] = alpha
                row["gamma"] = gamma
                row["episodes"] = episodes
                row["reserve_price"] = reserve_price
                row["auction_type_code"] = param_mappings["auction_type"][auction_type_str]
                row["init_code"] = param_mappings["init"][init_str]
                row["exploration_code"] = param_mappings["exploration"][exploration_str]
                row["asynchronous_code"] = param_mappings["asynchronous"][async_val]
                row["n_bidders"] = n_bidders_val
                row["median_opp_past_bid_index_code"] = param_mappings["median_opp_past_bid_index"][median_flag]
                row["winner_bid_index_state_code"] = param_mappings["winner_bid_index_state"][winner_flag]
                row["theoretical_revenue"] = rev_theory

            df_hist = pd.DataFrame(round_hist)
            hist_filename = f"history_run_{run_id}_{auction_type_str}.csv"
            df_hist.to_csv(os.path.join(trial_folder, hist_filename), index=False)

            # Summarize
            outcome = dict(summary_out)
            outcome["run_id"] = run_id
            outcome["eta"] = eta
            outcome["alpha"] = alpha
            outcome["gamma"] = gamma
            outcome["episodes"] = episodes
            outcome["reserve_price"] = reserve_price
            outcome["auction_type_code"] = param_mappings["auction_type"][auction_type_str]
            outcome["init_code"] = param_mappings["init"][init_str]
            outcome["exploration_code"] = param_mappings["exploration"][exploration_str]
            outcome["asynchronous_code"] = param_mappings["asynchronous"][async_val]
            outcome["n_bidders"] = n_bidders_val
            outcome["median_opp_past_bid_index_code"] = param_mappings["median_opp_past_bid_index"][median_flag]
            outcome["winner_bid_index_state_code"] = param_mappings["winner_bid_index_state"][winner_flag]
            outcome["theoretical_revenue"] = rev_theory

            # ratio to theoretical
            if rev_theory > 1e-8:
                ratio = outcome["avg_rev_last_1000"] / rev_theory
            else:
                ratio = None
            outcome["ratio_to_theory"] = ratio

            results.append(outcome)

    # Final summary DF
    df_final = pd.DataFrame(results)
    csv_path = os.path.join(folder_name, "data.csv")
    df_final.to_csv(csv_path, index=False)
    print(f"Data generation complete. Final summary => '{csv_path}'.")


# ------------------------------------
#  Main
# ------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment 2: Affiliated Values with Reserve Prices')
    parser.add_argument('--quick', action='store_true', help='Run quick test with reduced parameters')
    args = parser.parse_args()

    if args.quick:
        # Quick test mode: reduced parameters for fast validation
        print("=" * 50)
        print("QUICK TEST MODE - Reduced parameters for fast validation")
        print("=" * 50)
        run_full_experiment(
            experiment_id=2,
            K=5,                                    # Reduced runs
            eta_values=[0.0, 0.5],                  # Fewer eta values
            alpha_values=[0.01, 0.1],               # Fewer alpha values
            gamma_values=[0.9],                     # Single gamma
            episodes_values=[1000],                 # Reduced episodes
            reserve_price_values=[0.0],             # Single reserve price
            init_values=["random"],                 # Single init
            exploration_values=["egreedy"],         # Single exploration
            async_values=[0],                       # Single async value
            n_bidders_values=[2],                   # Single bidder count
            median_flags=[False],                   # Single flag
            winner_flags=[False],                   # Single flag
            seed=42,
            output_dir="results/exp2/quick_test"
        )
    else:
        # Full experiment mode
        run_full_experiment(
            experiment_id=2,
            K=250,
            eta_values=[0.0, 0.25, 0.5, 0.75, 1.0],
            alpha_values=[0.001, 0.005, 0.01, 0.05, 0.1],
            gamma_values=[0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
            episodes_values=[100_000],
            reserve_price_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            init_values=["random", "zeros"],
            exploration_values=["egreedy", "boltzmann"],
            async_values=[0, 1],
            n_bidders_values=[2, 4, 6],
            median_flags=[False, True],
            winner_flags=[False, True],
            seed=42,
            output_dir="results/exp2"
        )
