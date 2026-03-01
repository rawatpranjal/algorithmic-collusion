#!/usr/bin/env python3
"""
Experiment 4a: Autobidding with Regenerating Budgets (Multiplicative Dual Pacing).

Studies how auction format, bidder objective, and market thickness jointly
determine welfare, revenue, and collusion when dual-based pacing agents
learn with regenerating budgets over multiple episodes.

Motivated by the PoA literature (Balseiro & Gur 2019, Aggarwal et al. 2019,
Deng et al. 2021): FPA achieves PoA=1 for value-maximizers while SPA
achieves PoA=2.

Design: 2^6 = 64 full factorial (auction_type x objective x n_bidders
x budget_multiplier x reserve_price x sigma).
Each cell is replicated across 8 independent seeds.
Each run: D=100 episodes x T=1,000 rounds with budget regeneration.
"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------
# 1) Numeric Parameter Mappings
# ---------------------------------------------------------------------
param_mappings = {
    "auction_type": {"first": 1, "second": 0},
    "objective": {"value_max": 0, "utility_max": 1},
}

MU_RANGE = (0.5, 1.5)     # Bidder-specific mean range
MU_CLIP = (1e-4, 100.0)   # Dual variable bounds


# ---------------------------------------------------------------------
# 2) Auction Mechanism
# ---------------------------------------------------------------------
def run_auction(bids, valuations, auction_type, reserve_price=0.0):
    """
    Run a single-item auction with optional reserve price.

    Returns:
        winner: int index of winner, or -1 if no valid bids
        payment: float, the price paid
        rewards: ndarray of shape (n_bidders,), payoff for each bidder
    """
    n_bidders = len(bids)
    rewards = np.zeros(n_bidders)

    valid = np.where(bids >= reserve_price)[0] if reserve_price > 0 else np.where(bids > 0)[0]
    if len(valid) == 0:
        return -1, 0.0, rewards

    valid_bids = bids[valid]
    sorted_idx = np.argsort(valid_bids)[::-1]
    highest_bid = valid_bids[sorted_idx[0]]

    # Tie-breaking
    tied = [sorted_idx[0]]
    for idx in sorted_idx[1:]:
        if valid_bids[idx] == highest_bid:
            tied.append(idx)
        else:
            break

    winner_local = np.random.choice(tied) if len(tied) > 1 else tied[0]
    winner = valid[winner_local]

    if auction_type == "first":
        payment = bids[winner]
    else:
        # Second-price: payment = max(second-highest bid, reserve)
        if len(valid) == 1:
            # Single bidder above reserve: pays the reserve price
            payment = reserve_price
        elif len(valid) <= len(tied):
            # All valid bidders tied: pay own bid (no lower competing bid)
            payment = highest_bid
        else:
            if len(tied) >= 2:
                # 2+ tied at top: second-price = tied price
                payment = highest_bid
            else:
                # Single winner: find next-highest bid
                second_highest = reserve_price
                for idx in sorted_idx:
                    if idx not in tied:
                        second_highest = valid_bids[idx]
                        break
                payment = max(second_highest, reserve_price)

    rewards[winner] = valuations[winner] - payment
    return winner, payment, rewards


# ---------------------------------------------------------------------
# 3) Dual Pacing Agent
# ---------------------------------------------------------------------
class DualPacingAgent:
    """
    Multiplicative dual pacing agent for autobidding.

    Bid formulas:
        value_max:   b = v / mu
        utility_max: b = v / (1 + mu)

    Dual update: mu = clip(mu * exp(eta * (payment - rho)), MU_CLIP)
    where eta = 1/sqrt(T) and rho = budget/T (target spend rate).
    """

    def __init__(self, budget, T, objective, mu_init=1.0):
        self.budget = float(budget)
        self.T = int(T)
        self.objective = objective
        self.mu = float(mu_init)
        self.eta = 1.0 / math.sqrt(T)
        self.rho = budget / T  # target per-round spend

        self.cumulative_spend = 0.0
        self.t = 0

        # Per-episode tracking: circular buffer for mu (only last 200 needed)
        self._mu_tail_size = 200
        self._mu_tail = np.empty(self._mu_tail_size)
        self._mu_tail_idx = 0
        self._mu_tail_count = 0

    def get_bid(self, valuation):
        """Compute bid with hard budget constraint."""
        if self.objective == "value_max":
            raw_bid = valuation / max(self.mu, 1e-8)
        else:  # utility_max
            raw_bid = valuation / (1.0 + self.mu)

        remaining = max(0.0, self.budget - self.cumulative_spend)
        bid = min(raw_bid, remaining)
        bid = max(bid, 0.0)
        return float(bid)

    def update(self, payment):
        """Record payment and update dual variable."""
        self.cumulative_spend += payment
        self.t += 1

        # Multiplicative dual update (math.exp for scalar, max/min for scalar clip)
        gradient = payment - self.rho
        self.mu = self.mu * math.exp(self.eta * gradient)
        self.mu = max(MU_CLIP[0], min(self.mu, MU_CLIP[1]))
        # Circular buffer for mu history
        self._mu_tail[self._mu_tail_idx % self._mu_tail_size] = self.mu
        self._mu_tail_idx += 1
        self._mu_tail_count = min(self._mu_tail_count + 1, self._mu_tail_size)

    @property
    def mu_history(self):
        """Return the tail of mu values as a list (for backward compatibility)."""
        if self._mu_tail_count == 0:
            return []
        if self._mu_tail_count < self._mu_tail_size:
            return self._mu_tail[:self._mu_tail_count].tolist()
        # Buffer is full; return in chronological order
        start = self._mu_tail_idx % self._mu_tail_size
        return np.roll(self._mu_tail, -start).tolist()

    def reset_episode(self, new_budget):
        """Reset budget/spend for new episode, keep mu (warm-start)."""
        self.budget = float(new_budget)
        self.rho = new_budget / self.T
        self.cumulative_spend = 0.0
        self.t = 0
        self._mu_tail_idx = 0
        self._mu_tail_count = 0


# ---------------------------------------------------------------------
# 4) Offline Optimum (for PoA denominator)
# ---------------------------------------------------------------------
def compute_offline_optimum(valuations_matrix, budgets, T):
    """
    Greedy offline optimum: each round, assign item to highest-value
    bidder with remaining budget, payment = value (liquid welfare).

    Args:
        valuations_matrix: shape (T, N) matrix of valuations
        budgets: shape (N,) array of budgets
        T: number of rounds

    Returns:
        liquid_welfare: total value captured, capped at budgets
    """
    N = len(budgets)
    remaining = budgets.copy().astype(float)
    total_welfare = 0.0

    for t in range(T):
        vals = valuations_matrix[t]
        # Sort bidders by valuation, descending
        order = np.argsort(vals)[::-1]
        for i in order:
            if remaining[i] >= vals[i]:
                total_welfare += vals[i]
                remaining[i] -= vals[i]
                break
            elif remaining[i] > 0:
                # Partial: can only spend remaining budget
                total_welfare += remaining[i]
                remaining[i] = 0.0
                break

    return total_welfare


def compute_lp_offline_optimum(valuations_matrix, budgets, T):
    """
    LP-based offline optimum: solve the fractional allocation LP.

    Maximize sum_t sum_i v_{ti} * x_{ti}
    Subject to:
        sum_i x_{ti} <= 1       for all t  (one item per round)
        sum_t v_{ti} * x_{ti} <= B_i  for all i  (budget constraint)
        0 <= x_{ti} <= 1        for all t, i

    This provides an upper bound on achievable liquid welfare.

    Args:
        valuations_matrix: shape (T, N) matrix of valuations
        budgets: shape (N,) array of budgets
        T: number of rounds

    Returns:
        lp_welfare: optimal fractional liquid welfare
    """
    from scipy.optimize import linprog

    N = len(budgets)
    n_vars = T * N  # x_{ti} for each (t, i)

    # Objective: maximize sum v_{ti} * x_{ti} => minimize -v_{ti} * x_{ti}
    c = -valuations_matrix.ravel()  # shape (T*N,)

    # Constraint 1: sum_i x_{ti} <= 1 for each round t
    # Each row has N ones in the block for round t
    A_rounds = np.zeros((T, n_vars))
    for t in range(T):
        A_rounds[t, t * N:(t + 1) * N] = 1.0
    b_rounds = np.ones(T)

    # Constraint 2: sum_t v_{ti} * x_{ti} <= B_i for each bidder i
    A_budget = np.zeros((N, n_vars))
    for i in range(N):
        for t in range(T):
            A_budget[i, t * N + i] = valuations_matrix[t, i]
    b_budget = budgets.astype(float)

    # Combine constraints
    A_ub = np.vstack([A_rounds, A_budget])
    b_ub = np.concatenate([b_rounds, b_budget])

    # Bounds: 0 <= x_{ti} <= 1
    bounds = [(0.0, 1.0)] * n_vars

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if not result.success:
        raise RuntimeError(
            f"LP solver failed (status={result.status}): {result.message}. "
            f"T={T}, N={N}, budget range=[{budgets.min():.1f}, {budgets.max():.1f}]"
        )
    return -result.fun  # negate because we minimized -objective


# ---------------------------------------------------------------------
# 5) Episode Runner
# ---------------------------------------------------------------------
def run_episode(agents, N, T, auction_type, valuations_matrix, reserve_price=0.0):
    """
    Run T rounds of auction with given agents and valuations.

    Returns dict with per-episode metrics.
    """
    winners = np.empty(T, dtype=int)
    payments_arr = np.empty(T)
    bidder_bids = np.empty((N, T))
    bidder_vals = np.empty((N, T))

    for t in range(T):
        valuations = valuations_matrix[t]
        bids = np.array([agents[i].get_bid(valuations[i]) for i in range(N)])

        for i in range(N):
            bidder_bids[i, t] = bids[i]
            bidder_vals[i, t] = valuations[i]

        winner, payment, rewards = run_auction(bids, valuations, auction_type, reserve_price)

        # Update agents: only winner pays
        for i in range(N):
            cost = payment if i == winner else 0.0
            agents[i].update(cost)

        winners[t] = winner
        payments_arr[t] = payment

    # --- Compute episode metrics ---
    total_revenue = float(payments_arr.sum())

    # Liquid welfare: sum_i min(B_i, total_value_i) per Gaitonde et al. (2023)
    # Each bidder's contribution is capped at their budget
    valid_mask = winners >= 0
    bidder_value = np.zeros(N)
    for t in range(T):
        if valid_mask[t]:
            bidder_value[winners[t]] += valuations_matrix[t, winners[t]]
    liquid_welfare = sum(
        min(agents[i].budget, bidder_value[i]) for i in range(N)
    )

    # Allocative efficiency: fraction of rounds highest-value bidder wins
    efficient_count = 0
    for t in range(T):
        if valid_mask[t] and winners[t] == np.argmax(valuations_matrix[t]):
            efficient_count += 1
    allocative_efficiency = efficient_count / T

    # Budget utilization
    budget_util = np.mean([
        agents[i].cumulative_spend / max(agents[i].budget, 1e-10)
        for i in range(N)
    ])

    # Bid-to-value ratio (vectorized)
    all_bids_flat = bidder_bids.ravel()
    all_vals_flat = bidder_vals.ravel()
    valid_btv = all_vals_flat > 1e-9
    if valid_btv.any():
        bid_to_value = float(np.mean(all_bids_flat[valid_btv] / all_vals_flat[valid_btv]))
    else:
        bid_to_value = 0.0

    # Dual variable stats (last 200 rounds from circular buffer)
    dual_finals = []
    dual_cvs = []
    for i in range(N):
        count = agents[i]._mu_tail_count
        if count > 0:
            if count < agents[i]._mu_tail_size:
                tail_mu = agents[i]._mu_tail[:count]
            else:
                tail_mu = agents[i]._mu_tail
            dual_finals.append(float(tail_mu[(agents[i]._mu_tail_idx - 1) % agents[i]._mu_tail_size]))
            mean_mu = np.mean(tail_mu)
            std_mu = np.std(tail_mu)
            dual_cvs.append(float(std_mu / (abs(mean_mu) + 1e-10)))

    dual_final_mean = float(np.mean(dual_finals)) if dual_finals else 1.0
    dual_cv = float(np.mean(dual_cvs)) if dual_cvs else 0.0

    # No sale rate
    no_sale_count = int(np.sum(winners < 0))
    no_sale_rate = no_sale_count / T

    # Winner entropy
    valid_winners = winners[valid_mask]
    if len(valid_winners) > 0:
        unique_w, counts = np.unique(valid_winners, return_counts=True)
        p = counts / counts.sum()
        winner_entropy = float(-np.sum(p * np.log(p + 1e-12)))
    else:
        winner_entropy = 0.0

    return {
        "liquid_welfare": liquid_welfare,
        "platform_revenue": total_revenue,
        "allocative_efficiency": allocative_efficiency,
        "budget_utilization": budget_util,
        "bid_to_value": bid_to_value,
        "dual_final_mean": dual_final_mean,
        "dual_cv": dual_cv,
        "no_sale_rate": no_sale_rate,
        "winner_entropy": winner_entropy,
    }


# ---------------------------------------------------------------------
# 6) Main Experiment
# ---------------------------------------------------------------------
def run_experiment(
    auction_type,
    objective,
    n_bidders,
    n_episodes=100,
    T=1000,
    sigma=0.3,
    seed=0,
    progress_callback=None,
    budget_multiplier=0.5,
    reserve_price=0.0,
):
    """
    Run one cell of the autobidding pacing experiment.

    Episodic structure: D episodes x T rounds, budgets regenerate,
    dual variables warm-start across episodes.

    Returns:
        summary: dict with run-level aggregate metrics
        episode_data: list of per-episode metric dicts
        agents: list of final DualPacingAgent objects
    """
    rng = np.random.default_rng(seed)
    # Also set global numpy seed for auction tie-breaking
    np.random.seed(seed % (2**31))

    N = int(n_bidders)
    D = int(n_episodes)
    T = int(T)

    # Draw bidder-specific means once per seed (creates asymmetry)
    bidder_means = rng.uniform(MU_RANGE[0], MU_RANGE[1], size=N)

    # Compute budgets: budget_multiplier * E[v_i] * T where E[v] = exp(mu_i + sigma^2/2)
    expected_values = np.exp(bidder_means + sigma**2 / 2.0)
    budgets = budget_multiplier * expected_values * T

    # Create agents
    agents = [
        DualPacingAgent(budgets[i], T, objective, mu_init=1.0)
        for i in range(N)
    ]

    episode_data = []
    all_revenues = []
    all_btvs = []

    for d in range(D):
        if progress_callback and d % 10 == 0:
            progress_callback(current=d, total=D)

        # Draw valuations for this episode: LogNormal(bidder_means[i], sigma)
        # shape: (T, N)
        valuations_matrix = np.zeros((T, N))
        for i in range(N):
            valuations_matrix[:, i] = rng.lognormal(
                mean=bidder_means[i], sigma=sigma, size=T
            )

        # Compute offline optima for this episode
        offline_opt = compute_offline_optimum(valuations_matrix, budgets, T)
        lp_opt = compute_lp_offline_optimum(valuations_matrix, budgets, T)

        # Reset agent budgets (warm-start mu)
        for i in range(N):
            agents[i].reset_episode(budgets[i])

        # Run episode
        ep_metrics = run_episode(agents, N, T, auction_type, valuations_matrix, reserve_price)

        # Add PoA and offline optimum (greedy)
        if offline_opt > 1e-10:
            ep_metrics["effective_poa"] = offline_opt / max(ep_metrics["liquid_welfare"], 1e-10)
        else:
            ep_metrics["effective_poa"] = 1.0
        ep_metrics["offline_optimum"] = offline_opt

        # Add LP-based offline optimum
        ep_metrics["lp_offline_welfare"] = lp_opt
        if lp_opt > 1e-10:
            ep_metrics["effective_poa_lp"] = lp_opt / max(ep_metrics["liquid_welfare"], 1e-10)
        else:
            ep_metrics["effective_poa_lp"] = 1.0
        ep_metrics["episode"] = d

        episode_data.append(ep_metrics)
        all_revenues.append(ep_metrics["platform_revenue"])
        all_btvs.append(ep_metrics["bid_to_value"])

    if progress_callback:
        progress_callback(current=D, total=D)

    # ------------------------------------------------------------------
    # Run-level summary (aggregated from episodes d >= 10, i.e. 90 eps)
    # ------------------------------------------------------------------
    burn_in = min(10, D)
    post_burn = episode_data[burn_in:]

    if not post_burn:
        post_burn = episode_data  # fallback for very short runs

    def mean_field(field):
        return float(np.mean([ep[field] for ep in post_burn]))

    summary = {
        "mean_platform_revenue": mean_field("platform_revenue"),
        "mean_liquid_welfare": mean_field("liquid_welfare"),
        "mean_effective_poa": mean_field("effective_poa"),
        "mean_budget_utilization": mean_field("budget_utilization"),
        "mean_bid_to_value": mean_field("bid_to_value"),
        "mean_allocative_efficiency": mean_field("allocative_efficiency"),
        "mean_dual_cv": mean_field("dual_cv"),
        "mean_no_sale_rate": mean_field("no_sale_rate"),
        "mean_winner_entropy": mean_field("winner_entropy"),
        "mean_lp_offline_welfare": mean_field("lp_offline_welfare"),
        "mean_effective_poa_lp": mean_field("effective_poa_lp"),
        "mean_rev_all": float(np.mean(all_revenues)),
    }

    # Learning metrics
    if D >= 2:
        summary["warm_start_benefit"] = (
            episode_data[1]["platform_revenue"] - episode_data[0]["platform_revenue"]
        )
    else:
        summary["warm_start_benefit"] = 0.0

    post_burn_revs = [ep["platform_revenue"] for ep in post_burn]
    rev_mean = np.mean(post_burn_revs)
    rev_std = np.std(post_burn_revs)
    summary["inter_episode_volatility"] = float(rev_std / (rev_mean + 1e-10))

    # Collusion indicators
    # Competitive bid-to-value: FPA competitive = (N-1)/N, SPA competitive = 1.0
    if auction_type == "first":
        competitive_btv = (N - 1) / N
    else:
        competitive_btv = 1.0
    mean_btv = mean_field("bid_to_value")
    summary["bid_suppression_ratio"] = mean_btv / (competitive_btv + 1e-10)

    # Cross-episode drift: slope of btv across post-burn-in episodes
    if len(post_burn) >= 2:
        x = np.arange(len(post_burn))
        y = np.array([ep["bid_to_value"] for ep in post_burn])
        slope, _, _, _, _ = scipy_stats.linregress(x, y)
        summary["cross_episode_drift"] = float(slope)
    else:
        summary["cross_episode_drift"] = 0.0

    return summary, episode_data, agents


# ---------------------------------------------------------------------
# 7) CLI Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 4: Autobidding with Regenerating Budgets"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test: small number of episodes and rounds"
    )
    args = parser.parse_args()

    configs = [
        {"auction_type": "first", "objective": "value_max", "n_bidders": 2},
        {"auction_type": "second", "objective": "utility_max", "n_bidders": 4},
    ]

    for cfg in configs:
        print(f"\n{'='*50}")
        print(f"Config: {cfg}")
        print(f"{'='*50}")
        summary, episodes, agents = run_experiment(
            **cfg,
            n_episodes=10 if args.quick else 100,
            T=100 if args.quick else 1000,
            seed=42,
        )
        for k, v in sorted(summary.items()):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
