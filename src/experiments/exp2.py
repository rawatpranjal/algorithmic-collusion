#!/usr/bin/env python3
"""
Experiment 2: Affiliated-Values Q-Learning with BNE-Benchmarked Outcomes.

Focuses on 4 structural factors (auction_type, eta, n_bidders, state_info)
with Q-learning hyperparameters fixed at levels identified in Experiment 1.

Design: 3 × 2^3 = 24 cells (eta has 3 levels; others have 2).
"""

import numpy as np
import pandas as pd
import os
import json
import argparse

# ---------------------------------------------------------------------
# 1) Parameter Mappings
# ---------------------------------------------------------------------
param_mappings = {
    "auction_type": {"first": 1, "second": 0},
    "state_info": {"signal_only": 0, "signal_winner": 1},
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
        return rewards, -1, 0.0

    valid_bids = bids[valid_indices]
    sorted_indices = np.argsort(valid_bids)[::-1]
    highest_idx_local = [sorted_indices[0]]
    highest_bid = valid_bids[sorted_indices[0]]

    for idx_l in sorted_indices[1:]:
        if np.isclose(valid_bids[idx_l], highest_bid):
            highest_idx_local.append(idx_l)
        else:
            break

    if len(highest_idx_local) > 1:
        winner_local = np.random.choice(highest_idx_local)
    else:
        winner_local = highest_idx_local[0]
    winner_global = valid_indices[winner_local]
    winner_bid = bids[winner_global]

    if len(valid_indices) == len(highest_idx_local):
        # All valid bidders are tied at highest; in SPA use reserve price
        if auction_type != "first" and len(valid_indices) == 1:
            second_highest_bid = reserve_price
        else:
            second_highest_bid = highest_bid
    else:
        second_idx_local = None
        for idx_l in sorted_indices:
            if idx_l not in highest_idx_local:
                second_idx_local = idx_l
                break
        second_highest_bid = highest_bid if second_idx_local is None else valid_bids[second_idx_local]

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
# 4) Efficient Benchmark E[v_(1)] (Linear-Affiliation)
# ---------------------------------------------------------------------
def efficient_benchmark_ev1(N, eta):
    """
    Closed-form efficient benchmark: E[v_(1)] where v_i = alpha*s_i + beta_sum*sum_{j!=i} s_j.
    Here alpha = 1 - 0.5*eta, beta_sum = 0.5*eta/(N-1).
    E[v_(1)] = (alpha - beta_sum)*N/(N+1) + (N*beta_sum)/2.
    """
    if N < 2:
        alpha = 1.0 - 0.5 * eta
        return alpha * (N / (N + 1.0))
    alpha = 1.0 - 0.5 * eta
    beta_sum = 0.5 * eta / (N - 1)
    return (alpha - beta_sum) * (N / (N + 1.0)) + (N * beta_sum) / 2.0

# ---------------------------------------------------------------------
# 5) Q-Learning Experiment (Redesigned)
# ---------------------------------------------------------------------
def run_experiment(
    eta, auction_type, n_bidders, state_info,
    alpha=0.1, gamma=0.95, episodes=100_000,
    n_bid_actions=101, n_signal_bins=11,
    reserve_price=0.0, seed=0,
    store_qtables=False, qtable_folder=None,
    progress_callback=None,
):
    """
    Q-learning experiment with affiliated valuations and BNE-benchmarked outcomes.

    Fixed hyperparameters: epsilon-greedy with linear decay, zero init, async TD.
    4 design factors: eta, auction_type, n_bidders, state_info.
    """
    np.random.seed(seed)

    action_space = np.linspace(0, 1, n_bid_actions)

    # State space size
    if state_info == "signal_winner":
        n_states = n_signal_bins * n_bid_actions
    else:
        n_states = n_signal_bins

    # Initialize Q-tables (zeros)
    Q = np.zeros((n_bidders, n_states, n_bid_actions))

    # Tracking
    revenues = []
    winning_bids_list = []
    no_sale_count = 0
    winners_list = []

    # Epsilon schedule: linear decay from 1.0 to 0.0 over 90% of episodes
    eps_start, eps_end = 1.0, 0.0
    decay_end = int(0.9 * episodes)

    # For signal_winner state: track last winning bid bin
    last_winner_bid_bin = 0

    # Storage for final-window detailed data
    window_size = min(1000, episodes)
    final_window_bids = []
    final_window_vals = []
    final_window_winners = []
    final_window_payments = []
    final_window_signals = []

    save_interval = 1000
    if store_qtables and qtable_folder is not None:
        os.makedirs(qtable_folder, exist_ok=True)

    for ep in range(episodes):
        if progress_callback and ep % 1000 == 0:
            progress_callback(current=ep, total=episodes)

        # Epsilon decay
        if ep < decay_end:
            eps = eps_start - (ep / decay_end) * (eps_start - eps_end)
        else:
            eps = eps_end

        # Continuous signals, discretized for Q-table
        signals = np.random.uniform(0, 1, size=n_bidders)
        signal_bins = np.round(signals * (n_signal_bins - 1)).astype(int)

        # Valuations from continuous signals
        valuations = np.zeros(n_bidders)
        for i in range(n_bidders):
            others = np.delete(signals, i)
            valuations[i] = get_valuation(eta, signals[i], others)

        # Construct states
        if state_info == "signal_winner":
            states = signal_bins * n_bid_actions + last_winner_bid_bin
        else:
            states = signal_bins

        # Pick actions (epsilon-greedy)
        chosen_actions = np.zeros(n_bidders, dtype=int)
        for i in range(n_bidders):
            if np.random.rand() > eps:
                chosen_actions[i] = np.argmax(Q[i, states[i]])
            else:
                chosen_actions[i] = np.random.randint(n_bid_actions)

        bids = action_space[chosen_actions]

        # Auction
        rew, winner, winner_bid_val = get_rewards(bids, valuations, auction_type, reserve_price)

        # Seller revenue
        valid_bids = bids[bids >= reserve_price]
        if auction_type == "first":
            revenue_t = float(np.max(valid_bids)) if len(valid_bids) > 0 else 0.0
        else:
            if len(valid_bids) >= 2:
                sorted_valid = np.sort(valid_bids)
                revenue_t = float(sorted_valid[-2])
            elif len(valid_bids) == 1:
                revenue_t = float(valid_bids[0])
            else:
                revenue_t = 0.0
        revenues.append(revenue_t)
        winning_bids_list.append(winner_bid_val)

        if winner == -1:
            no_sale_count += 1
        else:
            winners_list.append(winner)

        # Track final-window detailed data
        in_final_window = ep >= (episodes - window_size)
        if in_final_window:
            final_window_bids.append(bids.copy())
            final_window_vals.append(valuations.copy())
            final_window_winners.append(winner)
            final_window_signals.append(signals.copy())
            # Payment
            if winner >= 0:
                if auction_type == "first":
                    final_window_payments.append(bids[winner])
                else:
                    # second-price payment
                    if len(valid_bids) >= 2:
                        sorted_valid = np.sort(valid_bids)
                        final_window_payments.append(sorted_valid[-2])
                    else:
                        final_window_payments.append(bids[winner])
            else:
                final_window_payments.append(0.0)

        # Next state
        if winner >= 0:
            new_winner_bid_bin = int(np.round(winner_bid_val * (n_bid_actions - 1)))
            new_winner_bid_bin = min(max(new_winner_bid_bin, 0), n_bid_actions - 1)
        else:
            new_winner_bid_bin = 0

        if state_info == "signal_winner":
            next_states = signal_bins * n_bid_actions + new_winner_bid_bin
        else:
            next_states = signal_bins

        # Q-update (asynchronous TD)
        for i in range(n_bidders):
            old_q = Q[i, states[i], chosen_actions[i]]
            td_target = rew[i] + gamma * np.max(Q[i, next_states[i]])
            Q[i, states[i], chosen_actions[i]] = old_q + alpha * (td_target - old_q)

        last_winner_bid_bin = new_winner_bid_bin

        # Save Q snapshot
        if store_qtables and (ep % save_interval == 0) and (qtable_folder is not None):
            snap_path = os.path.join(qtable_folder, f"q_after_{ep}.npy")
            np.save(snap_path, Q)

    # Final progress
    if progress_callback:
        progress_callback(current=episodes, total=episodes)

    if store_qtables and (qtable_folder is not None):
        final_path = os.path.join(qtable_folder, "q_after_final.npy")
        np.save(final_path, Q)

    # ---------------------------------------------------------------
    # Compute summary metrics
    # ---------------------------------------------------------------

    # --- Carried-forward metrics ---
    avg_rev_last_1000 = float(np.mean(revenues[-window_size:]))

    # Time to converge
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

    no_sale_rate = no_sale_count / episodes

    price_volatility = float(np.std(winning_bids_list))

    if len(winners_list) == 0:
        winner_entropy = 0.0
    else:
        unique_winners, counts = np.unique(winners_list, return_counts=True)
        p = counts / counts.sum()
        winner_entropy = float(-np.sum(p * np.log(p + 1e-12)))

    # --- BNE benchmarks (import from verification) ---
    try:
        from verification.bne_verify import (
            analytical_revenue,
            compute_bne_bid_coefficient,
            grid_adjusted_bne_revenue,
            bne_btv_benchmark,
            bne_winners_curse_benchmark,
            bne_bid_dispersion_benchmark,
        )
    except ImportError:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from verification.bne_verify import (
            analytical_revenue,
            compute_bne_bid_coefficient,
            grid_adjusted_bne_revenue,
            bne_btv_benchmark,
            bne_winners_curse_benchmark,
            bne_bid_dispersion_benchmark,
        )

    R_bne = analytical_revenue(eta, n_bidders)
    ev1 = efficient_benchmark_ev1(n_bidders, eta)
    phi_bne = compute_bne_bid_coefficient(eta, n_bidders, auction_type)
    # Grid-adjusted benchmark (captures discretization effects)
    R_grid, R_grid_se, tie_top_rate = grid_adjusted_bne_revenue(
        eta, n_bidders, auction_type, n_bid_actions=n_bid_actions, M=100_000, seed=seed + 1337
    )

    # --- Regret metrics ---
    raw_regret = 1.0 - avg_rev_last_1000
    efficient_regret = 1.0 - (avg_rev_last_1000 / ev1) if ev1 > 1e-12 else 0.0
    excess_regret = 1.0 - (avg_rev_last_1000 / R_bne) if R_bne > 1e-12 else 0.0

    # Revenue decomposition: structural + shading + excess = 1 - R_obs
    structural_gap = 1.0 - ev1  # unavoidable gap
    shading_gap = (ev1 - R_bne) if ev1 > 1e-12 else 0.0  # BNE shading
    excess_gap = (R_bne - avg_rev_last_1000) if R_bne > 1e-12 else 0.0  # learning loss

    # --- Distributional metrics from final window ---
    fw_bids = np.array(final_window_bids)      # (W, N)
    fw_vals = np.array(final_window_vals)       # (W, N)
    fw_winners = np.array(final_window_winners) # (W,)
    fw_payments = np.array(final_window_payments) # (W,)
    fw_signals = np.array(final_window_signals) # (W, N)

    # Bid-to-value ratio (for winners with positive valuations)
    valid_rounds = fw_winners >= 0
    if valid_rounds.sum() > 0:
        w_idx = fw_winners[valid_rounds].astype(int)
        rows = np.arange(valid_rounds.sum())
        w_vals = fw_vals[valid_rounds][rows, w_idx]
        w_payments = fw_payments[valid_rounds]
        btv = np.where(w_vals > 1e-12, w_payments / w_vals, 0.0)
        btv_median = float(np.median(btv))
        btv_iqr = float(np.percentile(btv, 75) - np.percentile(btv, 25))
    else:
        btv_median = 0.0
        btv_iqr = 0.0

    # Winner's curse: payment > winner's valuation
    if valid_rounds.sum() > 0:
        w_idx = fw_winners[valid_rounds].astype(int)
        rows = np.arange(valid_rounds.sum())
        w_vals = fw_vals[valid_rounds][rows, w_idx]
        w_payments = fw_payments[valid_rounds]
        winners_curse_freq = float((w_payments > w_vals).mean())
    else:
        winners_curse_freq = 0.0

    # Bid dispersion: mean within-round bid SD
    if len(fw_bids) > 0:
        round_sds = fw_bids.std(axis=1)
        bid_dispersion = float(round_sds.mean())
    else:
        bid_dispersion = 0.0

    # Signal responsiveness: OLS slope of bid on signal (across all bidders in final window)
    if len(fw_signals) > 0 and len(fw_bids) > 0:
        all_signals_flat = fw_signals.flatten()
        all_bids_flat = fw_bids.flatten()
        # OLS: slope = cov(x,y)/var(x)
        cov_sb = np.cov(all_signals_flat, all_bids_flat)[0, 1]
        var_s = np.var(all_signals_flat)
        observed_slope = cov_sb / var_s if var_s > 1e-12 else 0.0
        signal_slope_ratio = observed_slope / phi_bne if abs(phi_bne) > 1e-12 else 0.0
    else:
        signal_slope_ratio = 0.0

    # Tie-at-top rate in final window (useful for SPA with coarse grids)
    if len(fw_bids) > 0:
        sorted_b = np.sort(fw_bids, axis=1)
        if sorted_b.shape[1] >= 2:
            tie_top_rate_fw = float((sorted_b[:, -1] == sorted_b[:, -2]).mean())
        else:
            tie_top_rate_fw = 0.0
    else:
        tie_top_rate_fw = 0.0

    summary = {
        # Carried forward
        "avg_rev_last_1000": avg_rev_last_1000,
        "time_to_converge": time_to_converge,
        "no_sale_rate": no_sale_rate,
        "price_volatility": price_volatility,
        "winner_entropy": winner_entropy,
        # Regret
        "raw_regret": raw_regret,
        "efficient_regret": efficient_regret,
        "excess_regret": excess_regret,
        # Revenue decomposition
        "structural_gap": structural_gap,
        "shading_gap": shading_gap,
        "excess_gap": excess_gap,
        # Distributional
        "btv_median": btv_median,
        "btv_iqr": btv_iqr,
        "winners_curse_freq": winners_curse_freq,
        "bid_dispersion": bid_dispersion,
        "signal_slope_ratio": signal_slope_ratio,
        # BNE references
        "theoretical_revenue": R_bne,
        "efficient_benchmark": ev1,
        "bne_bid_coefficient": phi_bne,
        # Grid-adjusted benchmark and tie-at-top rate
        "theoretical_revenue_grid": R_grid,
        "theoretical_revenue_grid_se": R_grid_se,
        "tie_top_rate": tie_top_rate_fw,
    }
    return summary, revenues, [], Q


# ------------------------------------
#  Main (standalone test)
# ------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment 2: Affiliated Values Q-Learning (Redesigned)')
    parser.add_argument('--quick', action='store_true', help='Run a quick single-cell test')
    args = parser.parse_args()

    if args.quick:
        print("Quick single-cell test...")
        summary, revs, _, Q = run_experiment(
            eta=0.5, auction_type="second", n_bidders=2,
            state_info="signal_only", episodes=2000, seed=42,
        )
        for k, v in summary.items():
            print(f"  {k}: {v}")
    else:
        print("Use scripts/run_experiment.py --exp 2 for factorial runs.")
