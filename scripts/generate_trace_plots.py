#!/usr/bin/env python3
"""
Generate single-run trace visualizations for the paper.

Produces one illustrative figure per experiment showing learning dynamics
from a representative configuration. Outputs to paper/figures/e{1a,1b,2a,2b,3a,3b}_trace.png.
"""

import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

FIGURE_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

DPI = 300
STYLE_KW = dict(linewidth=0.8)

# Map code experiment IDs to paper figure prefixes
_TRACE_FILE_NAMES = {
    "exp1": "e1a", "exp2": "e1b",
    "exp3a": "e2a", "exp3b": "e2b",
    "exp4a": "e3a", "exp4b": "e3b",
}


def rolling_mean(data, window):
    return pd.Series(data).rolling(window=window, min_periods=1).mean().values


# =====================================================================
# Experiment 1: Q-Learning with Constant Valuations
# =====================================================================
def generate_exp1_trace():
    from experiments.exp1 import run_experiment

    print("Exp1: Running single trial...")
    summary, revenues, round_history, Q = run_experiment(
        auction_type="first",
        alpha=0.15,
        gamma=0.95,
        episodes=100_000,
        init="zeros",
        exploration="boltzmann",
        asynchronous=0,
        n_bidders=2,
        n_actions=11,
        info_feedback="minimal",
        reserve_price=0.0,
        decay_type="linear",
        seed=42,
        collect_history=True,
    )

    # Extract per-bidder bids from round_history
    df = pd.DataFrame(round_history)
    episodes = np.arange(len(revenues))

    fig, ax = plt.subplots(figsize=(10, 3.5))

    # Per-bidder mean bid evolution
    styles = ["-", "--"]
    for bidder_id in sorted(df["bidder_id"].unique()):
        bids = df[df["bidder_id"] == bidder_id]["bid"].values
        rm_bids = rolling_mean(bids, 2000)
        ax.plot(np.arange(len(rm_bids)), rm_bids,
                linestyle=styles[bidder_id % len(styles)],
                color="black", label=f"Bidder {bidder_id}", **STYLE_KW)
    # BNE bid for FPA with 2 bidders, constant valuations v=1: b = (n-1)/n = 0.5
    bne_bid = 0.5
    ax.axhline(bne_bid, color="gray", linestyle="--", linewidth=0.7,
               label=f"BNE bid = {bne_bid:.2f}")
    ax.set_ylabel("Mean bid (rolling mean)")
    ax.set_xlabel("Episode")
    ax.legend(frameon=False)

    fig.tight_layout()
    path = os.path.join(FIGURE_DIR, f"{_TRACE_FILE_NAMES['exp1']}_trace.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    print(f"  Summary: avg_rev={summary['avg_rev_last_1000']:.4f}")


# =====================================================================
# Experiment 2: Q-Learning with Affiliated Valuations
# =====================================================================
def generate_exp2_trace():
    from experiments.exp2 import run_experiment

    print("Exp2: Running single trial...")
    summary, revenues, _, Q = run_experiment(
        eta=0.5,
        auction_type="first",
        n_bidders=2,
        state_info="signal_only",
        episodes=100_000,
        seed=42,
    )

    episodes = np.arange(len(revenues))

    fig, ax = plt.subplots(figsize=(10, 3.5))

    # Revenue rolling mean
    rm = rolling_mean(revenues, 2000)
    ax.plot(episodes, rm, color="black", **STYLE_KW)

    # BNE revenue for FPA, 2 bidders, affiliated valuations with eta=0.5
    # alpha = 1 - eta/2 = 0.75, beta = eta/2 = 0.25, phi = 1.0
    # bne_rev = (n-1)/(n+1) * phi = 1/3
    bne_rev = (2 - 1) / (2 + 1) * 1.0
    ax.axhline(bne_rev, color="gray", linestyle="--", linewidth=0.7,
               label=f"BNE = {bne_rev:.3f}")
    ax.set_ylabel("Revenue (rolling mean)")
    ax.set_xlabel("Episode")
    ax.legend(frameon=False)

    fig.tight_layout()
    path = os.path.join(FIGURE_DIR, f"{_TRACE_FILE_NAMES['exp2']}_trace.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    print(f"  Summary: avg_rev={summary['avg_rev_last_1000']:.4f}")


# =====================================================================
# Experiment 3a: LinUCB Contextual Bandits
# =====================================================================
def generate_exp3a_trace():
    from experiments.exp3 import run_bandit_experiment

    eta, n_bidders, auction_type = 0.5, 2, "first"
    max_rounds = 100_000

    print("Exp3a (LinUCB): Running single trial...")
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
        memory_decay=1.0,
        seed=42,
        collect_history=True,
    )

    df = pd.DataFrame(round_history)

    fig, ax = plt.subplots(figsize=(10, 3.5))

    # Bid vs signal scatter (last 1000 rounds)
    tail_df = df[df["episode"] >= max_rounds - 1000]
    ax.scatter(tail_df["signal"], tail_df["chosen_bid"], s=2, color="black",
               alpha=0.3, rasterized=True, label="Observed bids")

    # BNE bid function: b(s) = phi_fpa * s where phi_fpa = ((n-1)/n) * (alpha + n*beta/2)
    # eta=0.5, n=2: alpha=0.75, beta=0.25 => phi_fpa = 0.5 * (0.75 + 2*0.25/2) = 0.5
    phi_fpa = 0.5
    s_line = np.linspace(0, 1, 100)
    ax.plot(s_line, phi_fpa * s_line, color="gray", linestyle="--", linewidth=0.7,
            label=f"BNE: $b = {phi_fpa:.2f}s$")
    ax.set_xlabel("Signal")
    ax.set_ylabel("Bid")
    ax.legend(frameon=False)

    fig.tight_layout()
    path = os.path.join(FIGURE_DIR, f"{_TRACE_FILE_NAMES['exp3a']}_trace.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    print(f"  Summary: avg_rev={summary['avg_rev_last_1000']:.4f}")


# =====================================================================
# Experiment 3b: Thompson Sampling Contextual Bandits
# =====================================================================
def generate_exp3b_trace():
    from experiments.exp3 import run_bandit_experiment

    eta, n_bidders, auction_type = 0.5, 2, "first"
    max_rounds = 100_000

    print("Exp3b (Thompson): Running single trial...")
    summary, revenues, round_history, bandits = run_bandit_experiment(
        eta=eta,
        auction_type=auction_type,
        lam=1.0,
        n_bidders=n_bidders,
        reserve_price=0.0,
        max_rounds=max_rounds,
        algorithm="thompson",
        exploration_intensity="low",
        context_richness="minimal",
        memory_decay=1.0,
        seed=42,
        collect_history=True,
    )

    df = pd.DataFrame(round_history)

    fig, ax = plt.subplots(figsize=(10, 3.5))

    # Bid vs signal scatter (last 1000 rounds)
    tail_df = df[df["episode"] >= max_rounds - 1000]
    ax.scatter(tail_df["signal"], tail_df["chosen_bid"], s=2, color="black",
               alpha=0.3, rasterized=True, label="Observed bids")

    # BNE bid function: same as exp3a (eta=0.5, n=2 => phi_fpa = 0.5)
    phi_fpa = 0.5
    s_line = np.linspace(0, 1, 100)
    ax.plot(s_line, phi_fpa * s_line, color="gray", linestyle="--", linewidth=0.7,
            label=f"BNE: $b = {phi_fpa:.2f}s$")
    ax.set_xlabel("Signal")
    ax.set_ylabel("Bid")
    ax.legend(frameon=False)

    fig.tight_layout()
    path = os.path.join(FIGURE_DIR, f"{_TRACE_FILE_NAMES['exp3b']}_trace.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    print(f"  Summary: avg_rev={summary['avg_rev_last_1000']:.4f}")


# =====================================================================
# Experiment 4a: Autobidding with Dual Pacing
# =====================================================================
def generate_exp4a_trace():
    from experiments.exp4a import run_experiment

    print("Exp4a: Running single trial...")
    summary, episode_data, agents = run_experiment(
        auction_type="first",
        objective="value_max",
        n_bidders=2,
        n_episodes=100,
        T=1000,
        budget_multiplier=0.5,
        reserve_price=0.0,
        sigma=0.3,
        seed=42,
    )

    ep_nums = [ep["episode"] for ep in episode_data]
    ep_btvs = [ep["bid_to_value"] for ep in episode_data]

    fig, ax = plt.subplots(figsize=(10, 3.5))

    # Mean bid-to-value ratio per episode
    ax.plot(ep_nums, ep_btvs, color="black", **STYLE_KW)
    competitive_btv = (2 - 1) / 2  # FPA with 2 bidders: 0.5
    ax.axhline(competitive_btv, color="gray", linestyle="--", linewidth=0.7,
               label=f"Competitive $b/v = {competitive_btv:.2f}$")
    ax.set_ylabel("Mean bid-to-value ratio")
    ax.set_xlabel("Episode")
    ax.legend(frameon=False)

    fig.tight_layout()
    path = os.path.join(FIGURE_DIR, f"{_TRACE_FILE_NAMES['exp4a']}_trace.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    print(f"  Summary: mean_rev={summary['mean_platform_revenue']:.4f}, "
          f"mean_btv={summary['mean_bid_to_value']:.4f}")


# =====================================================================
# Experiment 4b: Autobidding with PI Controller Pacing
# =====================================================================
def generate_exp4b_trace():
    from experiments.exp4b import run_experiment

    print("Exp4b: Running single trial...")
    summary, episode_data, agents = run_experiment(
        auction_type="first",
        aggressiveness=1.0,
        n_bidders=2,
        n_episodes=100,
        T=1000,
        budget_multiplier=0.5,
        reserve_price=0.0,
        sigma=0.3,
        seed=42,
    )

    ep_nums = [ep["episode"] for ep in episode_data]
    ep_btvs = [ep["bid_to_value"] for ep in episode_data]

    fig, ax = plt.subplots(figsize=(10, 3.5))

    # Mean bid-to-value ratio per episode
    ax.plot(ep_nums, ep_btvs, color="black", **STYLE_KW)
    # For PI, competitive btv = 1.0 (lambda=1 means bid full value)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.7,
               label="Competitive $b/v = 1.00$")
    ax.set_ylabel("Mean bid-to-value ratio")
    ax.set_xlabel("Episode")
    ax.legend(frameon=False)

    fig.tight_layout()
    path = os.path.join(FIGURE_DIR, f"{_TRACE_FILE_NAMES['exp4b']}_trace.png")
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
    generate_exp3a_trace()
    print()
    generate_exp3b_trace()
    print()
    generate_exp4a_trace()
    print()
    generate_exp4b_trace()
    print("\nAll trace plots generated.")
