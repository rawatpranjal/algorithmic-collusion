#!/usr/bin/env python3
"""
Generate single-run trace visualizations for the paper.

Produces one illustrative figure per experiment showing learning dynamics
from a representative configuration. Outputs to paper/figures/e{1-4}_trace.png.
"""

import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

FIGURE_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

DPI = 300
STYLE_KW = dict(linewidth=0.8)


def rolling_mean(data, window):
    return pd.Series(data).rolling(window=window, min_periods=1).mean().values


# =====================================================================
# Experiment 1: Q-Learning with Constant Valuations
# =====================================================================
def generate_exp1_trace():
    from experiments.exp1 import run_experiment, theoretical_revenue_constant_valuation

    print("Exp1: Running single trial...")
    summary, revenues, round_history, Q = run_experiment(
        auction_type="first",
        alpha=0.15,
        gamma=0.95,
        episodes=10_000,
        init="zeros",
        exploration="boltzmann",
        asynchronous=0,
        n_bidders=2,
        n_actions=21,
        info_feedback="minimal",
        reserve_price=0.0,
        decay_type="linear",
        seed=42,
    )

    bne_rev = theoretical_revenue_constant_valuation(2, "first")

    # Extract per-bidder bids from round_history
    df = pd.DataFrame(round_history)
    episodes = np.arange(len(revenues))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    # Panel 1: Revenue rolling mean
    ax = axes[0]
    rm = rolling_mean(revenues, 200)
    ax.plot(episodes, rm, color="black", **STYLE_KW)
    ax.axhline(bne_rev, color="gray", linestyle="--", linewidth=0.7, label=f"BNE = {bne_rev:.2f}")
    ax.set_ylabel("Revenue (rolling mean)")
    ax.set_xlabel("Episode")
    ax.legend(frameon=False)

    # Panel 2: Per-bidder mean bid evolution
    ax = axes[1]
    styles = ["-", "--"]
    for bidder_id in sorted(df["bidder_id"].unique()):
        bids = df[df["bidder_id"] == bidder_id]["bid"].values
        rm_bids = rolling_mean(bids, 200)
        ax.plot(np.arange(len(rm_bids)), rm_bids,
                linestyle=styles[bidder_id % len(styles)],
                color="black", label=f"Bidder {bidder_id}", **STYLE_KW)
    ax.axhline(bne_rev, color="gray", linestyle="--", linewidth=0.7, label="BNE bid")
    ax.set_ylabel("Mean bid (rolling mean)")
    ax.set_xlabel("Episode")
    ax.legend(frameon=False)

    # Panel 3: Epsilon decay curve
    ax = axes[2]
    decay_end = int(0.9 * len(revenues))
    eps_vals = []
    for ep in range(len(revenues)):
        if ep < decay_end:
            eps_vals.append(1.0 - (ep / decay_end))
        else:
            eps_vals.append(0.0)
    ax.plot(episodes, eps_vals, color="black", **STYLE_KW)
    ax.set_ylabel("Exploration temperature")
    ax.set_xlabel("Episode")

    fig.tight_layout()
    path = os.path.join(FIGURE_DIR, "e1_trace.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    print(f"  Summary: avg_rev={summary['avg_rev_last_1000']:.4f}, BNE={bne_rev:.4f}")


# =====================================================================
# Experiment 2: Q-Learning with Affiliated Valuations
# =====================================================================
def generate_exp2_trace():
    from experiments.exp2 import run_experiment
    from verification.bne_verify import analytical_revenue

    print("Exp2: Running single trial...")
    summary, revenues, _, Q = run_experiment(
        eta=0.5,
        auction_type="first",
        n_bidders=2,
        state_info="signal_only",
        episodes=10_000,
        seed=42,
    )

    bne_rev = analytical_revenue(0.5, 2)
    episodes = np.arange(len(revenues))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    # Panel 1: Revenue rolling mean with BNE line
    ax = axes[0]
    rm = rolling_mean(revenues, 200)
    ax.plot(episodes, rm, color="black", **STYLE_KW)
    ax.axhline(bne_rev, color="gray", linestyle="--", linewidth=0.7,
               label=f"BNE = {bne_rev:.3f}")
    ax.set_ylabel("Revenue (rolling mean)")
    ax.set_xlabel("Episode")
    ax.legend(frameon=False)

    # Panel 2: Raw revenue scatter in final 2000 episodes
    ax = axes[1]
    tail = min(2000, len(revenues))
    tail_eps = episodes[-tail:]
    tail_revs = revenues[-tail:]
    ax.scatter(tail_eps, tail_revs, s=1, color="black", alpha=0.3, rasterized=True)
    ax.axhline(bne_rev, color="gray", linestyle="--", linewidth=0.7,
               label=f"BNE = {bne_rev:.3f}")
    ax.set_ylabel("Revenue")
    ax.set_xlabel("Episode")
    ax.legend(frameon=False)

    fig.tight_layout()
    path = os.path.join(FIGURE_DIR, "e2_trace.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    print(f"  Summary: avg_rev={summary['avg_rev_last_1000']:.4f}, BNE={bne_rev:.4f}")


# =====================================================================
# Experiment 3: Contextual Bandits (LinUCB)
# =====================================================================
def generate_exp3_trace():
    from experiments.exp3 import run_bandit_experiment, simulate_linear_affiliation_revenue

    eta, n_bidders, auction_type = 0.5, 2, "first"
    max_rounds = 5000

    print("Exp3: Running single trial...")
    summary, revenues, round_history, bandits = run_bandit_experiment(
        eta=eta,
        auction_type=auction_type,
        lam=1.0,
        n_bidders=n_bidders,
        reserve_price=0.0,
        max_rounds=max_rounds,
        algorithm="linucb",
        exploration_intensity="low",
        context_richness="minimal",
        seed=42,
    )

    bne_rev = simulate_linear_affiliation_revenue(n_bidders, eta, auction_type, M=50_000)

    # BNE bid function: b(s) = phi * s
    alpha_val = 1.0 - 0.5 * eta
    beta_val = 0.5 * eta / max(n_bidders - 1, 1)
    phi_fpa = ((n_bidders - 1) / n_bidders) * (alpha_val + n_bidders * beta_val / 2.0)

    df = pd.DataFrame(round_history)
    rounds = np.arange(len(revenues))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    # Panel 1: Revenue rolling mean with BNE line
    ax = axes[0]
    rm = rolling_mean(revenues, 100)
    ax.plot(rounds, rm, color="black", **STYLE_KW)
    ax.axhline(bne_rev, color="gray", linestyle="--", linewidth=0.7,
               label=f"BNE = {bne_rev:.3f}")
    ax.set_ylabel("Revenue (rolling mean)")
    ax.set_xlabel("Round")
    ax.legend(frameon=False)

    # Panel 2: Bid vs signal scatter (last 1000 rounds) with BNE overlay
    ax = axes[1]
    tail_df = df[df["episode"] >= max_rounds - 1000]
    ax.scatter(tail_df["signal"], tail_df["chosen_bid"], s=2, color="black",
               alpha=0.3, rasterized=True, label="Observed bids")
    s_line = np.linspace(0, 1, 100)
    ax.plot(s_line, phi_fpa * s_line, color="gray", linestyle="--",
            linewidth=0.7, label=f"BNE: $b = {phi_fpa:.2f} s$")
    ax.set_xlabel("Signal")
    ax.set_ylabel("Bid")
    ax.legend(frameon=False)

    # Panel 3: Cumulative average reward per bidder
    ax = axes[2]
    styles = ["-", "--"]
    for bidder_id in sorted(df["bidder_id"].unique()):
        rewards = df[df["bidder_id"] == bidder_id]["reward"].values
        cum_avg = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
        ax.plot(np.arange(len(cum_avg)), cum_avg,
                linestyle=styles[bidder_id % len(styles)],
                color="black", label=f"Bidder {bidder_id}", **STYLE_KW)
    ax.set_ylabel("Cumulative avg. reward")
    ax.set_xlabel("Round")
    ax.legend(frameon=False)

    fig.tight_layout()
    path = os.path.join(FIGURE_DIR, "e3_trace.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    print(f"  Summary: avg_rev={summary['avg_rev_last_1000']:.4f}, BNE={bne_rev:.4f}")


# =====================================================================
# Experiment 4: Autobidding with Regenerating Budgets
# =====================================================================
def generate_exp4_trace():
    from experiments.exp4 import run_experiment

    print("Exp4: Running single trial...")
    summary, episode_data, agents = run_experiment(
        auction_type="first",
        objective="value_max",
        n_bidders=2,
        n_episodes=100,
        T=100,
        seed=42,
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    ep_nums = [ep["episode"] for ep in episode_data]
    ep_revs = [ep["platform_revenue"] for ep in episode_data]
    ep_btvs = [ep["bid_to_value"] for ep in episode_data]

    # Panel 1: Platform revenue per episode
    ax = axes[0]
    ax.plot(ep_nums, ep_revs, color="black", **STYLE_KW)
    ax.set_ylabel("Platform revenue")
    ax.set_xlabel("Episode")

    # Panel 2: Dual variable (mu) trajectories in last episode
    ax = axes[1]
    styles = ["-", "--", ":", "-."]
    for i, agent in enumerate(agents):
        if agent.mu_history:
            ax.plot(agent.mu_history,
                    linestyle=styles[i % len(styles)],
                    color="black", label=f"Bidder {i}", **STYLE_KW)
    ax.set_ylabel("Dual variable ($\\mu$)")
    ax.set_xlabel("Round (last episode)")
    ax.legend(frameon=False)

    # Panel 3: Mean bid-to-value ratio per episode
    ax = axes[2]
    ax.plot(ep_nums, ep_btvs, color="black", **STYLE_KW)
    competitive_btv = (2 - 1) / 2  # FPA with 2 bidders: 0.5
    ax.axhline(competitive_btv, color="gray", linestyle="--", linewidth=0.7,
               label=f"Competitive $b/v = {competitive_btv:.2f}$")
    ax.set_ylabel("Mean bid-to-value ratio")
    ax.set_xlabel("Episode")
    ax.legend(frameon=False)

    fig.tight_layout()
    path = os.path.join(FIGURE_DIR, "e4_trace.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    print(f"  Summary: mean_rev={summary['mean_platform_revenue']:.4f}, "
          f"mean_btv={summary['mean_bid_to_value']:.4f}")


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    print("Generating trace plots for paper...\n")
    generate_exp1_trace()
    print()
    generate_exp2_trace()
    print()
    generate_exp3_trace()
    print()
    generate_exp4_trace()
    print("\nAll trace plots generated.")
