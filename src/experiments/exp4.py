#!/usr/bin/env python3
"""
Experiment 4: Budget-Constrained Pacing Algorithms in Auctions.

Compares PID controller vs multiplicative (Lagrangian dual) pacing agents
bidding in repeated first-price and second-price auctions with affiliated
private values. First factorial study combining budget constraints with
algorithmic collusion analysis.

Design: 2^(8-1) Resolution VIII half-fraction (H = ABCDEFG)
Factors: algorithm, auction_type, n_bidders, budget_tightness, eta,
         aggressiveness, update_frequency, initial_multiplier
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# 1) Numeric Parameter Mappings
# ---------------------------------------------------------------------
param_mappings = {
    "auction_type": {"first": 1, "second": 0},
    "algorithm": {"multiplicative": 0, "pid": 1},
}


# ---------------------------------------------------------------------
# 2) Valuation with eta (Affiliation) — same as exp2/exp3
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
# 3) Payoff with Reserve Price — same as exp2/exp3
# ---------------------------------------------------------------------
def get_rewards(bids, valuations, auction_type="first", reserve_price=0.0):
    """
    Returns:
      rewards: shape (n_bidders,) => payoff for each bidder
      winner_global: int => index of winner or -1 if no sale
      winner_bid: float => winning bid or 0.0 if no sale
      second_price: float => payment in SPA (equals winner_bid in FPA)
    """
    n_bidders = len(bids)
    rewards = np.zeros(n_bidders)

    valid_indices = np.where(bids >= reserve_price)[0]
    if len(valid_indices) == 0:
        return rewards, -1, 0.0, 0.0  # no sale

    valid_bids = bids[valid_indices]
    sorted_idx = np.argsort(valid_bids)[::-1]
    highest_idx_local = [sorted_idx[0]]
    highest_bid = valid_bids[sorted_idx[0]]

    # tie among top bids (use exact equality to avoid false ties at very small bid values)
    for idx_l in sorted_idx[1:]:
        if valid_bids[idx_l] == highest_bid:
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

    # second-highest bid for SPA payment
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
        payment = winner_bid
    else:  # second-price
        rewards[winner_global] = valuations[winner_global] - second_highest_bid
        payment = second_highest_bid

    return rewards, winner_global, winner_bid, payment


# ---------------------------------------------------------------------
# 4) Theoretical BNE Revenue — same as exp3
# ---------------------------------------------------------------------
def simulate_linear_affiliation_revenue(N, eta, auction_type, M=50_000):
    """
    Monte Carlo estimate of BNE revenue under linear affiliation.
    Used as theoretical benchmark for cross-experiment comparison.
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
# 5) Pacing Agents
# ---------------------------------------------------------------------

class PacingAgent:
    """Base class for budget-constrained pacing agents."""

    def __init__(self, budget, T, aggressiveness, initial_multiplier):
        self.budget = float(budget)
        self.T = int(T)
        self.aggressiveness = float(aggressiveness)
        self.multiplier = float(initial_multiplier)
        self.cumulative_spend = 0.0
        self.t = 0  # rounds completed
        self.multiplier_history = []  # sampled every 10 rounds

    def get_bid(self, valuation):
        """Compute bid with hard budget constraint: bid <= min(v*mu, remaining)."""
        bid = valuation * self.multiplier
        remaining = max(0.0, self.budget - self.cumulative_spend)
        return float(np.clip(min(bid, remaining), 0.0, float(valuation)))

    def record_outcome(self, cost):
        """Record payment and advance round counter."""
        self.cumulative_spend += cost
        self.t += 1

    def update_multiplier(self, avg_round_cost):
        """Subclasses implement specific pacing control law."""
        raise NotImplementedError


class PIDPacer(PacingAgent):
    """
    Proportional-Integral controller for bid pacing.
    K_D = 0 (literature consensus: derivative unhelpful in stochastic auctions).
    Error signal: target_spend - actual_spend (underspend = positive error = raise bids).
    """

    def __init__(self, budget, T, aggressiveness, initial_multiplier):
        super().__init__(budget, T, aggressiveness, initial_multiplier)
        self.K_P = 0.30 * aggressiveness
        self.K_I = 0.05 * aggressiveness
        self.integral_error = 0.0

    def update_multiplier(self, avg_round_cost):
        """PI control law using cumulative spend vs target spend."""
        # target_spend at current time: (t/T) * budget
        target_spend = (self.t / self.T) * self.budget
        error = target_spend - self.cumulative_spend
        self.integral_error += error
        delta = self.K_P * error + self.K_I * self.integral_error
        self.multiplier = float(np.clip(self.multiplier + delta, 0.01, 1.5))


class MultiplicativePacer(PacingAgent):
    """
    Multiplicative pacing via Lagrangian dual ascent.
    mu_{t+1} = max(0, mu_t + eta_step * (cost - budget/T))
    multiplier = 1 / (1 + mu)
    """

    def __init__(self, budget, T, aggressiveness, initial_multiplier):
        super().__init__(budget, T, aggressiveness, initial_multiplier)
        self.eta_step = (1.0 / np.sqrt(T)) * aggressiveness
        self.mu = 0.0  # Lagrangian dual variable (lambda in some papers)

    def update_multiplier(self, avg_round_cost):
        """Dual ascent step using average per-round cost."""
        target_cost = self.budget / self.T
        self.mu = max(0.0, self.mu + self.eta_step * (avg_round_cost - target_cost))
        self.multiplier = float(np.clip(1.0 / (1.0 + self.mu), 0.01, 1.5))


# ---------------------------------------------------------------------
# 6) Main Simulation: run_experiment
# ---------------------------------------------------------------------

def run_experiment(
    algorithm,
    auction_type,
    n_bidders,
    budget_tightness,
    eta,
    aggressiveness,
    update_frequency,
    initial_multiplier,
    max_rounds=10_000,
    reserve_price=0.0,
    seed=0,
    progress_callback=None,
):
    """
    Run one cell of the budget-constrained pacing experiment.

    Returns:
      summary: dict with all response variables
      revenues: list of per-round revenues (for convergence analysis)
      round_history: list of per-round dicts (thin for memory efficiency)
      agents: list of final PacingAgent objects
    """
    np.random.seed(seed)

    T = int(max_rounds)
    n_bidders = int(n_bidders)
    update_frequency = int(update_frequency)

    # Budget per agent: budget_tightness * E[v] * T
    # E[v] = 0.5 for U[0,1] marginals (both eta=0 and eta=1)
    E_v = 0.5
    budget_per_agent = float(budget_tightness) * E_v * T

    # Instantiate agents
    agents = []
    for _ in range(n_bidders):
        if algorithm == "pid":
            agent = PIDPacer(budget_per_agent, T, aggressiveness, initial_multiplier)
        else:
            agent = MultiplicativePacer(budget_per_agent, T, aggressiveness, initial_multiplier)
        agents.append(agent)

    # Pre-allocate tracking arrays
    revenues = []
    winners_list = []
    winning_bids_list = []
    no_sale_count = 0

    # Per-agent tracking for budget metrics
    per_agent_bids = [[] for _ in range(n_bidders)]
    per_agent_valuations = [[] for _ in range(n_bidders)]
    per_agent_payments = [[] for _ in range(n_bidders)]

    # Batch cost tracking for update_frequency > 1
    batch_costs = [[] for _ in range(n_bidders)]

    # Budget violation counter (sanity check: should stay 0)
    violation_count = 0

    for ep in range(T):
        if progress_callback and ep % 1000 == 0:
            progress_callback(current=ep, total=T)

        # Draw private signals (iid U[0,1])
        signals = np.random.uniform(0.0, 1.0, n_bidders)
        valuations = np.zeros(n_bidders)
        for i in range(n_bidders):
            others = np.delete(signals, i)
            valuations[i] = get_valuation(eta, signals[i], others)

        # Compute bids (hard constraint enforced in get_bid)
        bids = np.array([agents[i].get_bid(valuations[i]) for i in range(n_bidders)])

        # Run auction; get_rewards extended to return actual payment
        rew, winner, winner_bid, payment_amount = get_rewards(
            bids, valuations, auction_type, reserve_price
        )

        # Determine per-agent costs: only winner pays
        costs = np.zeros(n_bidders)
        if winner >= 0:
            costs[winner] = payment_amount

        # Record outcomes
        for i in range(n_bidders):
            agents[i].record_outcome(costs[i])
            batch_costs[i].append(costs[i])
            per_agent_bids[i].append(bids[i])
            per_agent_valuations[i].append(valuations[i])
            per_agent_payments[i].append(costs[i])

            # Budget violation check (should never trigger due to hard constraint)
            if agents[i].cumulative_spend > agents[i].budget + 1e-9:
                violation_count += 1

        # Batch multiplier update
        if (ep + 1) % update_frequency == 0 or ep == T - 1:
            for i in range(n_bidders):
                if batch_costs[i]:
                    avg_cost = float(np.mean(batch_costs[i]))
                    agents[i].update_multiplier(avg_cost)
                    batch_costs[i] = []

        # Sample multiplier every 10 rounds for convergence diagnostics
        if ep % 10 == 0:
            for i in range(n_bidders):
                agents[i].multiplier_history.append(agents[i].multiplier)

        # Revenue tracking (use winner_bid = highest bid, consistent with exp1-3)
        if winner >= 0:
            revenue_t = winner_bid
            winners_list.append(winner)
        else:
            revenue_t = 0.0
            no_sale_count += 1

        revenues.append(revenue_t)
        winning_bids_list.append(winner_bid)

    if progress_callback:
        progress_callback(current=T, total=T)

    # ------------------------------------------------------------------
    # Compute summary statistics
    # ------------------------------------------------------------------
    window_size = min(1000, T)
    rev_series = pd.Series(revenues)

    avg_rev_last_1000 = float(np.mean(revenues[-window_size:]))

    # Time to converge: first episode where rolling-mean stays in ±5% band
    roll_avg = rev_series.rolling(window=window_size).mean()
    final_rev = avg_rev_last_1000
    lower_band = 0.95 * final_rev
    upper_band = 1.05 * final_rev
    time_to_converge = T
    for t_idx in range(len(revenues) - window_size):
        window_val = roll_avg.iloc[t_idx + window_size - 1]
        if lower_band <= window_val <= upper_band:
            stay_in_band = True
            for j in range(t_idx + window_size, len(revenues) - window_size):
                v_j = roll_avg.iloc[j + window_size - 1]
                if not (lower_band <= v_j <= upper_band):
                    stay_in_band = False
                    break
            if stay_in_band:
                time_to_converge = t_idx + window_size
                break

    # Standard metrics (matching Exp 1-2 naming)
    avg_regret_of_seller = float(np.mean([1.0 - r for r in revenues]))
    no_sale_rate = no_sale_count / T
    price_volatility = float(np.std(winning_bids_list)) if winning_bids_list else 0.0

    if len(winners_list) == 0:
        winner_entropy = 0.0
    else:
        unique_w, counts = np.unique(winners_list, return_counts=True)
        p = counts / counts.sum()
        winner_entropy = float(-np.sum(p * np.log(p + 1e-12)))

    # Budget-specific metrics
    budget_utilization = float(np.mean([
        agents[i].cumulative_spend / agents[i].budget
        for i in range(n_bidders)
    ]))

    # Spend volatility: CV of per-period payments in last window_size rounds
    last_payments = []
    for i in range(n_bidders):
        last_payments.extend(per_agent_payments[i][-window_size:])
    spend_mean = float(np.mean(last_payments)) if last_payments else 0.0
    spend_std = float(np.std(last_payments)) if last_payments else 0.0
    spend_volatility = spend_std / (spend_mean + 1e-10)

    # Budget violation rate (sanity: should = 0 due to hard constraint)
    budget_violation_rate = violation_count / (T * n_bidders)

    # Effective bid shading: mean(1 - bid/valuation) across all bids
    shading_vals = []
    for i in range(n_bidders):
        for b, v in zip(per_agent_bids[i], per_agent_valuations[i]):
            if v > 1e-9:
                shading_vals.append(1.0 - b / v)
    effective_bid_shading = float(np.mean(shading_vals)) if shading_vals else 0.0

    # ------------------------------------------------------------------
    # Multiplier convergence diagnostics (averaged across agents)
    # ------------------------------------------------------------------
    # multiplier_convergence_time: first sample index where CV of trailing
    # 50 samples (500 rounds) stays below 0.02 for the rest of the run.
    # Reported in units of rounds (sample_idx * 10).
    multiplier_convergence_times = []
    multiplier_final_means = []
    multiplier_final_stds = []

    n_samples = len(agents[0].multiplier_history)
    window_samples = 50  # 50 samples = 500 rounds

    for i in range(n_bidders):
        hist = agents[i].multiplier_history
        conv_time = T  # default: never converged

        if n_samples >= window_samples:
            for s in range(n_samples - window_samples):
                window = hist[s:s + window_samples]
                mean_w = float(np.mean(window))
                std_w = float(np.std(window))
                cv = std_w / (abs(mean_w) + 1e-10)
                if cv < 0.02:
                    # Check it stays stable for the rest of the history
                    stay_stable = True
                    for s2 in range(s + window_samples, n_samples - window_samples):
                        w2 = hist[s2:s2 + window_samples]
                        m2 = float(np.mean(w2))
                        cv2 = float(np.std(w2)) / (abs(m2) + 1e-10)
                        if cv2 >= 0.02:
                            stay_stable = False
                            break
                    if stay_stable:
                        conv_time = (s + window_samples) * 10  # convert to rounds
                        break

        multiplier_convergence_times.append(conv_time)

        last_samples = hist[-window_samples:] if len(hist) >= window_samples else hist
        multiplier_final_means.append(float(np.mean(last_samples)))
        multiplier_final_stds.append(float(np.std(last_samples)))

    multiplier_convergence_time = float(np.mean(multiplier_convergence_times))
    multiplier_final_mean = float(np.mean(multiplier_final_means))
    multiplier_final_std = float(np.mean(multiplier_final_stds))

    summary = {
        "avg_rev_last_1000": avg_rev_last_1000,
        "time_to_converge": time_to_converge,
        "avg_regret_of_seller": avg_regret_of_seller,
        "no_sale_rate": no_sale_rate,
        "price_volatility": price_volatility,
        "winner_entropy": winner_entropy,
        "budget_utilization": budget_utilization,
        "spend_volatility": spend_volatility,
        "budget_violation_rate": budget_violation_rate,
        "effective_bid_shading": effective_bid_shading,
        "multiplier_convergence_time": multiplier_convergence_time,
        "multiplier_final_mean": multiplier_final_mean,
        "multiplier_final_std": multiplier_final_std,
    }

    # Thin round history (episode-level summary only, to keep memory low)
    round_history = [
        {"episode": ep, "revenue": revenues[ep]}
        for ep in range(T)
    ]

    return summary, revenues, round_history, agents


# ---------------------------------------------------------------------
# 7) Legacy Orchestrator (deprecated; kept for CLI compatibility)
# ---------------------------------------------------------------------

def run_full_experiment(
    experiment_id=4,
    K=50,
    algorithm_values=("pid", "multiplicative"),
    auction_type_values=("first", "second"),
    n_bidders_values=(2, 4),
    budget_tightness_values=(0.25, 0.75),
    eta_values=(0.0, 1.0),
    aggressiveness_values=(0.5, 2.0),
    update_frequency_values=(1, 100),
    initial_multiplier_values=(0.5, 1.0),
    max_rounds=10_000,
    seed=1234,
    output_dir=None,
):
    """
    Deprecated random-sampling orchestrator (retained for direct CLI use).
    For factorial design, use scripts/run_experiment.py --exp 4.
    """
    folder = output_dir or f"results/exp{experiment_id}"
    os.makedirs(folder, exist_ok=True)

    with open(os.path.join(folder, "param_mappings.json"), "w") as f:
        json.dump(param_mappings, f, indent=2)

    rng = np.random.default_rng(seed)
    results = []

    for run_id in range(K):
        kwargs = {
            "algorithm": rng.choice(list(algorithm_values)),
            "auction_type": rng.choice(list(auction_type_values)),
            "n_bidders": int(rng.choice(list(n_bidders_values))),
            "budget_tightness": float(rng.choice(list(budget_tightness_values))),
            "eta": float(rng.choice(list(eta_values))),
            "aggressiveness": float(rng.choice(list(aggressiveness_values))),
            "update_frequency": int(rng.choice(list(update_frequency_values))),
            "initial_multiplier": float(rng.choice(list(initial_multiplier_values))),
            "max_rounds": max_rounds,
            "seed": run_id,
        }

        summary, _, _, _ = run_experiment(**kwargs)
        row = dict(summary)
        row["run_id"] = run_id
        for k, v in kwargs.items():
            row[k] = v
        results.append(row)
        print(f"  Run {run_id + 1}/{K}: avg_rev={row['avg_rev_last_1000']:.4f}, "
              f"budget_util={row['budget_utilization']:.4f}")

    df = pd.DataFrame(results)
    csv_path = os.path.join(folder, "data.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nDone. {len(results)} runs => '{csv_path}'")


# ---------------------------------------------------------------------
# 8) CLI Entry Point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 4: Budget-Constrained Pacing Algorithms"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test: 10 random runs, 1000 rounds each"
    )
    args = parser.parse_args()

    if args.quick:
        print("=" * 50)
        print("QUICK TEST MODE - Budget-Constrained Pacing")
        print("=" * 50)
        run_full_experiment(
            experiment_id=4,
            K=5,
            algorithm_values=("pid", "multiplicative"),
            auction_type_values=("first", "second"),
            n_bidders_values=(2,),
            budget_tightness_values=(0.25, 0.75),
            eta_values=(0.0,),
            aggressiveness_values=(1.0,),
            update_frequency_values=(1,),
            initial_multiplier_values=(1.0,),
            max_rounds=1000,
            output_dir="results/exp4/quick_test",
        )
    else:
        run_full_experiment(
            experiment_id=4,
            K=100,
            max_rounds=10_000,
            output_dir="results/exp4",
        )
