#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import json
import pickle
import argparse
import matplotlib.pyplot as plt
from tqdm import trange

# ---------------------------------------------------------------------
# 1) Numeric Parameter Mappings
# ---------------------------------------------------------------------
param_mappings = {
    "auction_type": {"first": 1, "second": 0},
    "use_median_of_others": {False: 0, True: 1},
    "use_past_winner_bid": {False: 0, True: 1}
}

# ---------------------------------------------------------------------
# 2) Valuation with eta (Affiliation)
# ---------------------------------------------------------------------
def get_valuation(eta, own_signal, others_signals):
    """
    Linear affiliation:
      valuation = alpha * own_signal + beta * mean(others_signals),
    alpha = 1.0 - 0.5 * eta, beta = 0.5 * eta.
    """
    alpha = 1.0 - 0.5 * eta
    beta = 0.5 * eta
    return alpha * own_signal + beta * np.mean(others_signals)

# ---------------------------------------------------------------------
# 3) Payoff with Reserve Price
# ---------------------------------------------------------------------
def get_rewards(bids, valuations, auction_type="first", reserve_price=0.0):
    """
    Returns:
      rewards: shape (n_bidders,) => payoff for each bidder
      winner_global: int => index of winner or -1 if no sale
      winner_bid: float => winning bid or 0.0 if no sale
    """
    n_bidders = len(bids)
    rewards = np.zeros(n_bidders)

    valid_indices = np.where(bids >= reserve_price)[0]
    if len(valid_indices) == 0:
        return rewards, -1, 0.0  # no sale

    valid_bids = bids[valid_indices]
    sorted_idx = np.argsort(valid_bids)[::-1]
    highest_idx_local = [sorted_idx[0]]
    highest_bid = valid_bids[sorted_idx[0]]

    # tie among top bids
    for idx_l in sorted_idx[1:]:
        if np.isclose(valid_bids[idx_l], highest_bid):
            highest_idx_local.append(idx_l)
        else:
            break

    # pick winner randomly if tie
    if len(highest_idx_local) > 1:
        winner_local = np.random.choice(highest_idx_local)
    else:
        winner_local = highest_idx_local[0]
    winner_global = valid_indices[winner_local]
    winner_bid = bids[winner_global]

    # second-highest
    if len(valid_indices) == len(highest_idx_local):
        second_highest_bid = highest_bid
    else:
        second_idx_local = None
        for idx_l in sorted_idx:
            if idx_l not in highest_idx_local:
                second_idx_local = idx_l
                break
        if second_idx_local is None:
            second_highest_bid = highest_bid
        else:
            second_highest_bid = valid_bids[second_idx_local]

    # payoff
    if auction_type == "first":
        rewards[winner_global] = valuations[winner_global] - winner_bid
    else:  # second-price
        rewards[winner_global] = valuations[winner_global] - second_highest_bid

    return rewards, winner_global, winner_bid

# ---------------------------------------------------------------------
# 4) LinUCB Bandit
# ---------------------------------------------------------------------
class LinUCB:
    """
    For each action a:
      A[a] = XᵀX + lambda*I
      b[a] = Xᵀy
    p(a) = theta_hat[a]ᵀ context + c * sqrt(contextᵀ A[a]^(-1) context)
    """
    def __init__(self, n_actions, context_dim, c, lam):
        self.n_actions = n_actions
        self.context_dim = context_dim
        self.c = c
        self.lam = lam

        self.A = [lam * np.eye(context_dim) for _ in range(n_actions)]
        self.b = [np.zeros(context_dim) for _ in range(n_actions)]

    def select_action(self, context_vec):
        p_vals = []
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta_hat = A_inv @ self.b[a]
            mean_est = theta_hat @ context_vec
            var_est = context_vec.T @ A_inv @ context_vec
            bonus = self.c * np.sqrt(var_est)
            p_vals.append(mean_est + bonus)
        return np.argmax(p_vals)

    def update(self, action, reward, context_vec):
        self.A[action] += np.outer(context_vec, context_vec)
        self.b[action] += reward * context_vec

# ---------------------------------------------------------------------
# 5) Theoretical BNE Revenue (Linear Affiliation)
# ---------------------------------------------------------------------
def simulate_linear_affiliation_revenue(N, eta, auction_type, M=50_000):
    """
    For each bidder i: t_i ~ U[0..1].
    BNE strategy:
      FPA => factor = ((N-1)/N)*[alpha + (N/2)*beta]
      SPA => factor = alpha + (N/2)*beta
    alpha=1-0.5*eta, beta=(0.5*eta)/(N-1).
    Return average revenue (Monte Carlo).
    """
    if N < 1:
        return 0.0

    alpha = 1.0 - 0.5 * eta
    beta = 0.5 * eta / max(N - 1, 1.0)

    if auction_type == "first":
        factor = ((N - 1) / float(N)) * (alpha + (N / 2.0) * beta)
    else:
        factor = alpha + (N / 2.0) * beta

    rev_sum = 0.0
    for _ in range(M):
        t = np.random.rand(N)
        bids = factor * t
        max_bid = np.max(bids)
        top_idx = np.where(np.isclose(bids, max_bid))[0]
        winner = np.random.choice(top_idx)

        if auction_type == "first":
            price = max_bid
        else:
            if len(top_idx) == 1:
                other_bids = np.delete(bids, winner)
                second = np.max(other_bids) if len(other_bids) else 0.0
                price = second
            else:
                price = max_bid
        rev_sum += price

    return rev_sum / M

# ---------------------------------------------------------------------
# 6) Single Bandit Experiment
# ---------------------------------------------------------------------
def run_bandit_experiment(
    eta, auction_type, c, lam,
    n_bidders, reserve_price, max_rounds,
    use_median_of_others, use_past_winner_bid,
    seed=0,
    progress_callback=None
):
    """
    Key changes:
     - We'll store 'revenues' each round to compute time_to_converge
       with the ±5% band approach from Exp.1/2.
     - We'll also track no_sale_count, the highest bid each round,
       and the identity of the winner => can compute winner_entropy & price_volatility.
    """
    np.random.seed(seed)

    n_val_bins = 6
    n_bid_bins = 6
    actions = np.linspace(0, 1, n_bid_bins)

    # Build context dimension
    # We'll treat signals[i] as mandatory 1-dim + optional median + optional past_winner
    context_dim = 1
    if use_median_of_others:
        context_dim += 1
    if use_past_winner_bid:
        context_dim += 1

    bandits = [LinUCB(n_actions=n_bid_bins, context_dim=context_dim, c=c, lam=lam)
               for _ in range(n_bidders)]

    revenues = []
    winners_list = []
    winning_bids_list = []
    round_history = []
    no_sale_count = 0

    past_bids = np.zeros(n_bidders)
    past_winner_bid = 0.0

    def build_context(bidder_i, signals):
        ctx_list = [signals[bidder_i]]  # always include own signal
        if use_median_of_others:
            # median-of-others in last round
            med_bid = np.median(np.delete(past_bids, bidder_i)) if n_bidders > 1 else 0.0
            ctx_list.append(med_bid)
        if use_past_winner_bid:
            ctx_list.append(past_winner_bid)
        return np.array(ctx_list)

    for ep in range(max_rounds):
        # Progress callback every 1000 rounds
        if progress_callback and ep % 1000 == 0:
            progress_callback(current=ep, total=max_rounds)

        signals = np.random.randint(n_val_bins, size=n_bidders) / (n_val_bins - 1)
        valuations = np.zeros(n_bidders)
        for i in range(n_bidders):
            others_signals = np.delete(signals, i)
            valuations[i] = get_valuation(eta, signals[i], others_signals)

        # Choose bids
        chosen_actions = []
        chosen_bids = []
        for i in range(n_bidders):
            ctx_vec = build_context(i, signals)
            a_i = bandits[i].select_action(ctx_vec)
            chosen_actions.append((i, a_i, ctx_vec))
            chosen_bids.append(actions[a_i])

        chosen_bids = np.array(chosen_bids)
        rew, winner, highest_bid = get_rewards(chosen_bids, valuations, auction_type, reserve_price)

        # Update bandits
        for (i, a_idx, ctx_vec) in chosen_actions:
            bandits[i].update(a_idx, rew[i], ctx_vec)

        # Revenue
        valid_bids = chosen_bids[chosen_bids >= reserve_price]
        revenue_t = np.max(valid_bids) if len(valid_bids) > 0 else 0.0
        revenues.append(revenue_t)
        winning_bids_list.append(highest_bid)

        if winner == -1:
            no_sale_count += 1
        else:
            winners_list.append(winner)

        # log round
        for i in range(n_bidders):
            round_history.append({
                "episode": ep,
                "bidder_id": i,
                "signal": signals[i],
                "chosen_bid": chosen_bids[i],
                "reward": rew[i],
                "is_winner": (i == winner)
            })

        # Update memory
        past_bids = chosen_bids
        past_winner_bid = highest_bid if winner != -1 else 0.0

    # Final progress update
    if progress_callback:
        progress_callback(current=max_rounds, total=max_rounds)

    # After max_rounds
    # time_to_converge: unify with Exp.1/2 => rolling window of 1000, stay in ±5% band
    window_size = 1000
    import pandas as pd
    rev_series = pd.Series(revenues)

    if len(revenues) >= window_size:
        avg_rev_last_1000 = np.mean(revenues[-window_size:])
    else:
        avg_rev_last_1000 = np.mean(revenues)

    roll_avg = rev_series.rolling(window=window_size).mean()
    final_rev = avg_rev_last_1000
    lower_band = 0.95 * final_rev
    upper_band = 1.05 * final_rev
    time_to_converge = max_rounds
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

    # average regret
    regrets = [1.0 - r for r in revenues]
    avg_regret_seller = np.mean(regrets)

    # no_sale_rate
    no_sale_rate = no_sale_count / max_rounds

    # price_volatility
    price_volatility = np.std(winning_bids_list) if len(winning_bids_list) else 0.0

    # winner_entropy
    if len(winners_list) == 0:
        winner_entropy = 0.0
    else:
        unique_winners, counts = np.unique(winners_list, return_counts=True)
        p = counts / counts.sum()
        winner_entropy = -np.sum(p * np.log(p + 1e-12))

    summary = {
        "avg_rev_last_1000": avg_rev_last_1000,
        "time_to_converge": time_to_converge,
        "avg_regret_seller": avg_regret_seller,
        "no_sale_rate": no_sale_rate,
        "price_volatility": price_volatility,
        "winner_entropy": winner_entropy
    }
    return summary, revenues, round_history, bandits  # bandits hold final models

# ---------------------------------------------------------------------
# 7) Main Orchestrator: run_full_experiment
# ---------------------------------------------------------------------
def run_full_experiment(
    experiment_id=3,
    K=300,
    eta_values=[0.0, 0.25, 0.5, 0.75, 1.0],
    c_values=[0.01, 0.1, 0.5, 1.0, 2.0],
    lam_values=[0.1, 1.0, 5.0],
    n_bidders_values=[2, 4, 6],
    reserve_price_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    seed=1234,
    max_rounds_values=[10000],
    use_median_flags=[False, True],
    use_winner_flags=[False, True],
    output_dir=None
):
    """
    Unify with Exp.2 style:
     - param_mappings => store in param_mappings.json
     - final data => data.csv
     - store round_history in 'trials'
     - store final LinUCB model in 'linucb_models'
     - track the additional outcome metrics
     - each run => do 'first' & 'second'

    Now with rolling-average plots for each run (first vs second).
    """
    folder_name = output_dir if output_dir else f"results/exp{experiment_id}"
    os.makedirs(folder_name, exist_ok=True)

    # Save param mappings
    with open(os.path.join(folder_name, "param_mappings.json"), "w") as f:
        json.dump(param_mappings, f, indent=2)

    trial_folder = os.path.join(folder_name, "trials")
    os.makedirs(trial_folder, exist_ok=True)
    models_folder = os.path.join(folder_name, "linucb_models")
    os.makedirs(models_folder, exist_ok=True)

    # (NEW) Subfolder for rolling-average plots
    plots_folder = os.path.join(folder_name, "plots")
    os.makedirs(plots_folder, exist_ok=True)

    results = []
    theory_cache = {}

    rng = np.random.default_rng(seed)

    for run_id in trange(K, desc="Overall Runs"):
        eta = rng.choice(eta_values)
        c = rng.choice(c_values)
        lam = rng.choice(lam_values)
        n_bidders = rng.choice(n_bidders_values)
        r_price = rng.choice(reserve_price_values)
        use_median = rng.choice(use_median_flags)
        use_winner = rng.choice(use_winner_flags)
        max_rounds = rng.choice(max_rounds_values)

        # We'll track revenues from first & second auctions for rolling-average plots
        revenues_dict = {}

        for auc_str in ["first", "second"]:
            # Theoretical revenue
            key = (n_bidders, eta, auc_str)
            if key not in theory_cache:
                rev_theory = simulate_linear_affiliation_revenue(n_bidders, eta, auc_str)
                theory_cache[key] = rev_theory
            else:
                rev_theory = theory_cache[key]

            summary_out, revenues_list, round_hist, final_bandits = run_bandit_experiment(
                eta=eta,
                auction_type=auc_str,
                c=c,
                lam=lam,
                n_bidders=n_bidders,
                reserve_price=r_price,
                max_rounds=max_rounds,
                use_median_of_others=use_median,
                use_past_winner_bid=use_winner,
                seed=run_id
            )

            # Keep revenues_list for plotting
            revenues_dict[auc_str] = revenues_list

            # Build round logs
            df_hist = pd.DataFrame(round_hist)
            df_hist["run_id"] = run_id
            df_hist["eta"] = eta
            df_hist["c"] = c
            df_hist["lam"] = lam
            df_hist["n_bidders"] = n_bidders
            df_hist["auction_type_code"] = param_mappings["auction_type"][auc_str]
            df_hist["reserve_price"] = r_price
            df_hist["max_rounds"] = max_rounds
            df_hist["use_median_of_others_code"] = param_mappings["use_median_of_others"][use_median]
            df_hist["use_past_winner_bid_code"] = param_mappings["use_past_winner_bid"][use_winner]
            df_hist["theoretical_revenue"] = rev_theory

            # Save round-level logs
            hist_filename = f"history_run_{run_id}_{auc_str}.csv"
            df_hist.to_csv(os.path.join(trial_folder, hist_filename), index=False)

            # Save final LinUCB model
            for b_i, bandit in enumerate(final_bandits):
                model_dict = {
                    "A_list": [bandit.A[a] for a in range(len(bandit.A))],
                    "b_list": [bandit.b[a] for a in range(len(bandit.b))]
                }
                model_fname = f"LinUCB_run_{run_id}_{auc_str}_bidder_{b_i}.pkl"
                with open(os.path.join(models_folder, model_fname), "wb") as fmod:
                    pickle.dump(model_dict, fmod)

            # Final outcome
            outcome = dict(summary_out)
            outcome["run_id"] = run_id
            outcome["eta"] = eta
            outcome["c"] = c
            outcome["lam"] = lam
            outcome["n_bidders"] = n_bidders
            outcome["auction_type_code"] = param_mappings["auction_type"][auc_str]
            outcome["reserve_price"] = r_price
            outcome["max_rounds"] = max_rounds
            outcome["use_median_of_others_code"] = param_mappings["use_median_of_others"][use_median]
            outcome["use_past_winner_bid_code"] = param_mappings["use_past_winner_bid"][use_winner]
            outcome["theoretical_revenue"] = rev_theory

            # ratio
            if rev_theory > 1e-8:
                ratio = outcome["avg_rev_last_1000"] / rev_theory
            else:
                ratio = None
            outcome["ratio_to_theory"] = ratio

            results.append(outcome)

        # -----------------------------------------------------------------
        # PLOTTING: rolling-average of revenues for first vs second
        # -----------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 4))
        window_size = 1000
        for auc_type in ["first", "second"]:
            rev_series = pd.Series(revenues_dict[auc_type])
            roll_avg = rev_series.rolling(window=window_size, min_periods=1).mean()
            label_str = "First-Price" if auc_type == "first" else "Second-Price"
            ax.plot(roll_avg, label=label_str)

        title_line_1 = f"Run {run_id}"
        title_line_2 = (
            f"eta={eta}, c={c}, lam={lam}, nb={n_bidders}, reserve={r_price}, max_rounds={max_rounds}"
        )
        ax.set_title(f"{title_line_1}\n{title_line_2}")
        ax.set_xlabel("Round")
        ax.set_ylabel("Rolling Avg Revenue (window=1000)")
        ax.legend()
        fig.tight_layout()

        plot_filename = f"plot_run_{run_id}.png"
        fig.savefig(os.path.join(plots_folder, plot_filename), bbox_inches='tight')
        plt.close(fig)
        # -----------------------------------------------------------------

    # Summaries across all runs
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(folder_name, "data.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nAll done. Final summary => {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment 3: LinUCB Bandits with Affiliated Values')
    parser.add_argument('--quick', action='store_true', help='Run quick test with reduced parameters')
    args = parser.parse_args()

    if args.quick:
        # Quick test mode: reduced parameters for fast validation
        print("=" * 50)
        print("QUICK TEST MODE - Reduced parameters for fast validation")
        print("=" * 50)
        run_full_experiment(
            experiment_id=3,
            K=5,                                    # Reduced runs
            eta_values=[0.0, 0.5],                  # Fewer eta values
            c_values=[0.1, 1.0],                    # Fewer c values
            lam_values=[1.0],                       # Single lambda
            n_bidders_values=[2],                   # Single bidder count
            reserve_price_values=[0.0],             # Single reserve price
            seed=1234,
            max_rounds_values=[1000],               # Reduced rounds
            use_median_flags=[False],               # Single flag
            use_winner_flags=[False],               # Single flag
            output_dir="results/exp3/quick_test"
        )
    else:
        # Full experiment mode
        run_full_experiment(
            experiment_id=3,
            K=250,
            eta_values=[0.0, 0.25, 0.5, 0.75, 1.0],
            c_values=[0.01, 0.1, 0.5, 1.0, 2.0],
            lam_values=[0.1, 1.0, 5.0],
            n_bidders_values=[2, 4, 6],
            reserve_price_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            seed=1234,
            max_rounds_values=[100_000],
            use_median_flags=[False, True],
            use_winner_flags=[False, True],
            output_dir="results/exp3"
        )
