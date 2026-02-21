#!/usr/bin/env python3
"""
Unified single-run deep dive analysis tool.

Runs one experiment with specified parameters, saves all outputs (config,
summary, revenues, round history, final state), generates trace plots,
and optionally prints verbose post-hoc diagnostics.

Usage:
    PYTHONPATH=src python3 scripts/deep_dive.py --exp 1
    PYTHONPATH=src python3 scripts/deep_dive.py --exp 2 --param eta=1.0 --verbose
    PYTHONPATH=src python3 scripts/deep_dive.py --exp 3 --list-params
    PYTHONPATH=src python3 scripts/deep_dive.py --exp 4 --no-plots --no-save
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default parameter registries (matching generate_trace_plots.py configs)
# ---------------------------------------------------------------------------

EXP1_DEFAULTS = {
    "auction_type": "first",
    "alpha": 0.15,
    "gamma": 0.95,
    "episodes": 10_000,
    "init": "zeros",
    "exploration": "boltzmann",
    "asynchronous": 0,
    "n_bidders": 2,
    "n_actions": 21,
    "info_feedback": "minimal",
    "reserve_price": 0.0,
    "decay_type": "linear",
}

EXP2_DEFAULTS = {
    "eta": 0.5,
    "auction_type": "first",
    "n_bidders": 2,
    "state_info": "signal_only",
    "episodes": 10_000,
}

EXP3_DEFAULTS = {
    "eta": 0.5,
    "auction_type": "first",
    "lam": 1.0,
    "n_bidders": 2,
    "reserve_price": 0.0,
    "max_rounds": 5000,
    "algorithm": "linucb",
    "exploration_intensity": "low",
    "context_richness": "minimal",
}

EXP4_DEFAULTS = {
    "auction_type": "first",
    "objective": "value_max",
    "n_bidders": 2,
    "n_episodes": 100,
    "T": 100,
}

DEFAULTS = {1: EXP1_DEFAULTS, 2: EXP2_DEFAULTS, 3: EXP3_DEFAULTS, 4: EXP4_DEFAULTS}

# Type coercion for --param overrides
PARAM_TYPES = {
    "alpha": float, "gamma": float, "episodes": int, "asynchronous": int,
    "n_bidders": int, "n_actions": int, "reserve_price": float,
    "eta": float, "lam": float, "max_rounds": int,
    "n_episodes": int, "T": int,
}

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DeepDiveResult:
    exp_num: int
    params: dict
    summary: dict
    revenues: list
    round_history: list
    final_state: Any
    bne_revenue: float | None = None
    elapsed_sec: float = 0.0

# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_exp1(params, seed):
    from experiments.exp1 import run_experiment, theoretical_revenue_constant_valuation

    summary, revenues, round_history, Q = run_experiment(**params, seed=seed)
    bne_rev = theoretical_revenue_constant_valuation(
        params["n_bidders"], params["auction_type"]
    )
    return DeepDiveResult(
        exp_num=1, params=params, summary=summary,
        revenues=revenues, round_history=round_history,
        final_state=Q, bne_revenue=bne_rev,
    )


def run_exp2(params, seed):
    from experiments.exp2 import run_experiment
    try:
        from verification.bne_verify import analytical_revenue
    except ImportError:
        analytical_revenue = None

    summary, revenues, _, Q = run_experiment(**params, seed=seed)
    bne_rev = None
    if analytical_revenue is not None:
        try:
            bne_rev = analytical_revenue(params["eta"], params["n_bidders"])
        except Exception:
            pass
    if bne_rev is None:
        bne_rev = summary.get("theoretical_revenue")
    return DeepDiveResult(
        exp_num=2, params=params, summary=summary,
        revenues=revenues, round_history=[],
        final_state=Q, bne_revenue=bne_rev,
    )


def run_exp3(params, seed):
    from experiments.exp3 import run_bandit_experiment, simulate_linear_affiliation_revenue

    summary, revenues, round_history, bandits = run_bandit_experiment(**params, seed=seed)
    bne_rev = simulate_linear_affiliation_revenue(
        params["n_bidders"], params["eta"], params["auction_type"], M=50_000,
    )
    return DeepDiveResult(
        exp_num=3, params=params, summary=summary,
        revenues=revenues, round_history=round_history,
        final_state=bandits, bne_revenue=bne_rev,
    )


def run_exp4(params, seed):
    from experiments.exp4 import run_experiment

    summary, episode_data, agents = run_experiment(**params, seed=seed)
    revenues = [ep["platform_revenue"] for ep in episode_data]
    return DeepDiveResult(
        exp_num=4, params=params, summary=summary,
        revenues=revenues, round_history=episode_data,
        final_state=agents, bne_revenue=None,
    )

RUNNERS = {1: run_exp1, 2: run_exp2, 3: run_exp3, 4: run_exp4}

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(result: DeepDiveResult, output_dir: str, seed: int):
    os.makedirs(output_dir, exist_ok=True)

    # config.json
    config = {"exp": result.exp_num, "seed": seed, "params": result.params}
    if result.bne_revenue is not None:
        config["bne_revenue"] = result.bne_revenue
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    # summary.json
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(result.summary, f, indent=2, default=str)

    # revenues.csv
    rev_df = pd.DataFrame({"episode": range(len(result.revenues)), "revenue": result.revenues})
    rev_df.to_csv(os.path.join(output_dir, "revenues.csv"), index=False)

    # round_history.csv
    if result.round_history:
        rh_df = pd.DataFrame(result.round_history)
        rh_df.to_csv(os.path.join(output_dir, "round_history.csv"), index=False)
    else:
        with open(os.path.join(output_dir, "round_history.csv"), "w") as f:
            f.write("# Empty: this experiment does not store per-round data.\n")

    # final_state/
    state_dir = os.path.join(output_dir, "final_state")
    os.makedirs(state_dir, exist_ok=True)
    _save_final_state(result, state_dir)

    print(f"Results saved to {output_dir}/")


def _save_final_state(result: DeepDiveResult, state_dir: str):
    if result.exp_num in (1, 2):
        # Q-table: numpy array (n_bidders, n_states, n_actions)
        Q = result.final_state
        if Q is not None:
            np.save(os.path.join(state_dir, "q_table.npy"), Q)
    elif result.exp_num == 3:
        # Bandits: list of LinUCB or CTS objects
        bandits = result.final_state
        if bandits:
            for i, b in enumerate(bandits):
                data = {}
                if hasattr(b, "A"):
                    data["A"] = np.array([a.tolist() if hasattr(a, "tolist") else a for a in b.A], dtype=object)
                if hasattr(b, "b"):
                    data["b"] = np.array([bi.tolist() if hasattr(bi, "tolist") else bi for bi in b.b], dtype=object)
                np.savez(os.path.join(state_dir, f"bandit_{i}.npz"), **{k: v for k, v in data.items()})
    elif result.exp_num == 4:
        # Agents: list of DualPacingAgent
        agents = result.final_state
        if agents:
            for i, agent in enumerate(agents):
                agent_data = {
                    "mu": agent.mu,
                    "budget": agent.budget,
                    "mu_history": agent.mu_history,
                    "bid_history": agent.bid_history,
                    "payment_history": getattr(agent, "payment_history", []),
                }
                with open(os.path.join(state_dir, f"agent_{i}.json"), "w") as f:
                    json.dump(agent_data, f, indent=2, default=float)

# ---------------------------------------------------------------------------
# Trace plots
# ---------------------------------------------------------------------------

DPI = 200
STYLE_KW = dict(linewidth=0.8)


def rolling_mean(data, window):
    return pd.Series(data).rolling(window=window, min_periods=1).mean().values


def plot_trace(result: DeepDiveResult, output_dir: str):
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, "trace.png")

    plotters = {1: _plot_exp1, 2: _plot_exp2, 3: _plot_exp3, 4: _plot_exp4}
    plotters[result.exp_num](result, path)
    print(f"Trace plot saved to {path}")


def _plot_exp1(result, path):
    from experiments.exp1 import theoretical_revenue_constant_valuation

    revenues = result.revenues
    bne_rev = result.bne_revenue
    episodes = np.arange(len(revenues))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    # Revenue rolling mean
    ax = axes[0]
    ax.plot(episodes, rolling_mean(revenues, 200), color="black", **STYLE_KW)
    ax.axhline(bne_rev, color="gray", linestyle="--", linewidth=0.7, label=f"BNE = {bne_rev:.2f}")
    ax.set_ylabel("Revenue (rolling mean)")
    ax.set_xlabel("Episode")
    ax.legend(frameon=False)

    # Per-bidder bid evolution
    ax = axes[1]
    if result.round_history:
        df = pd.DataFrame(result.round_history)
        styles = ["-", "--"]
        for bidder_id in sorted(df["bidder_id"].unique()):
            bids = df[df["bidder_id"] == bidder_id]["bid"].values
            ax.plot(np.arange(len(bids)), rolling_mean(bids, 200),
                    linestyle=styles[bidder_id % len(styles)],
                    color="black", label=f"Bidder {bidder_id}", **STYLE_KW)
        ax.axhline(bne_rev, color="gray", linestyle="--", linewidth=0.7, label="BNE bid")
    ax.set_ylabel("Mean bid (rolling mean)")
    ax.set_xlabel("Episode")
    ax.legend(frameon=False)

    # Epsilon decay
    ax = axes[2]
    decay_end = int(0.9 * len(revenues))
    eps_vals = [max(0.0, 1.0 - ep / decay_end) if decay_end > 0 else 0.0 for ep in range(len(revenues))]
    ax.plot(episodes, eps_vals, color="black", **STYLE_KW)
    ax.set_ylabel("Exploration temperature")
    ax.set_xlabel("Episode")

    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_exp2(result, path):
    revenues = result.revenues
    bne_rev = result.bne_revenue
    episodes = np.arange(len(revenues))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    # Revenue rolling mean
    ax = axes[0]
    ax.plot(episodes, rolling_mean(revenues, 200), color="black", **STYLE_KW)
    if bne_rev is not None:
        ax.axhline(bne_rev, color="gray", linestyle="--", linewidth=0.7,
                   label=f"BNE = {bne_rev:.3f}")
    ax.set_ylabel("Revenue (rolling mean)")
    ax.set_xlabel("Episode")
    ax.legend(frameon=False)

    # Raw revenue scatter (last 2000)
    ax = axes[1]
    tail = min(2000, len(revenues))
    tail_eps = episodes[-tail:]
    tail_revs = revenues[-tail:]
    ax.scatter(tail_eps, tail_revs, s=1, color="black", alpha=0.3, rasterized=True)
    if bne_rev is not None:
        ax.axhline(bne_rev, color="gray", linestyle="--", linewidth=0.7,
                   label=f"BNE = {bne_rev:.3f}")
    ax.set_ylabel("Revenue")
    ax.set_xlabel("Episode")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_exp3(result, path):
    revenues = result.revenues
    bne_rev = result.bne_revenue
    params = result.params
    n_bidders = params["n_bidders"]
    eta = params["eta"]
    max_rounds = len(revenues)
    rounds = np.arange(max_rounds)

    # BNE bid coefficient
    alpha_val = 1.0 - 0.5 * eta
    beta_val = 0.5 * eta / max(n_bidders - 1, 1)
    phi_fpa = ((n_bidders - 1) / n_bidders) * (alpha_val + n_bidders * beta_val / 2.0)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    # Revenue rolling mean
    ax = axes[0]
    ax.plot(rounds, rolling_mean(revenues, 100), color="black", **STYLE_KW)
    if bne_rev is not None:
        ax.axhline(bne_rev, color="gray", linestyle="--", linewidth=0.7,
                   label=f"BNE = {bne_rev:.3f}")
    ax.set_ylabel("Revenue (rolling mean)")
    ax.set_xlabel("Round")
    ax.legend(frameon=False)

    # Bid vs signal scatter
    ax = axes[1]
    if result.round_history:
        df = pd.DataFrame(result.round_history)
        tail_df = df[df["episode"] >= max_rounds - 1000]
        ax.scatter(tail_df["signal"], tail_df["chosen_bid"], s=2, color="black",
                   alpha=0.3, rasterized=True, label="Observed bids")
    s_line = np.linspace(0, 1, 100)
    ax.plot(s_line, phi_fpa * s_line, color="gray", linestyle="--",
            linewidth=0.7, label=f"BNE: $b = {phi_fpa:.2f} s$")
    ax.set_xlabel("Signal")
    ax.set_ylabel("Bid")
    ax.legend(frameon=False)

    # Cumulative average reward per bidder
    ax = axes[2]
    if result.round_history:
        df = pd.DataFrame(result.round_history)
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
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_exp4(result, path):
    episode_data = result.round_history  # episode_data stored as round_history
    agents = result.final_state
    params = result.params
    n_bidders = params["n_bidders"]
    auction_type = params["auction_type"]

    ep_nums = [ep["episode"] for ep in episode_data]
    ep_revs = [ep["platform_revenue"] for ep in episode_data]
    ep_btvs = [ep["bid_to_value"] for ep in episode_data]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    # Platform revenue per episode
    ax = axes[0]
    ax.plot(ep_nums, ep_revs, color="black", **STYLE_KW)
    ax.set_ylabel("Platform revenue")
    ax.set_xlabel("Episode")

    # Dual variable trajectories (last episode)
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

    # Bid-to-value ratio
    ax = axes[2]
    ax.plot(ep_nums, ep_btvs, color="black", **STYLE_KW)
    competitive_btv = 1.0 if auction_type == "second" else (n_bidders - 1) / n_bidders
    ax.axhline(competitive_btv, color="gray", linestyle="--", linewidth=0.7,
               label=f"Competitive $b/v = {competitive_btv:.2f}$")
    ax.set_ylabel("Mean bid-to-value ratio")
    ax.set_xlabel("Episode")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Verbose diagnostics
# ---------------------------------------------------------------------------

def print_verbose(result: DeepDiveResult):
    print("\n" + "=" * 60)
    print(f"VERBOSE DIAGNOSTICS: Experiment {result.exp_num}")
    print("=" * 60)

    verbose_fns = {1: _verbose_exp1, 2: _verbose_exp2, 3: _verbose_exp3, 4: _verbose_exp4}
    verbose_fns[result.exp_num](result)


def _verbose_exp1(result):
    Q = result.final_state  # (n_bidders, n_states, n_actions)
    params = result.params
    n_bidders = params["n_bidders"]
    n_actions = params["n_actions"]
    bids = np.linspace(0, 1, n_actions)

    print(f"\nQ-table shape: {Q.shape}")
    print(f"Q-table range: [{Q.min():.4f}, {Q.max():.4f}]")
    print(f"Q-table mean: {Q.mean():.4f}, std: {Q.std():.4f}")
    nonzero = np.count_nonzero(Q)
    total = Q.size
    print(f"Sparsity: {nonzero}/{total} non-zero ({100 * nonzero / total:.1f}%)")

    print("\nGreedy policy (state -> best bid):")
    for bidder in range(min(n_bidders, 4)):
        print(f"  Bidder {bidder}:")
        for state in range(Q.shape[1]):
            best_action = np.argmax(Q[bidder, state])
            best_bid = bids[best_action]
            q_val = Q[bidder, state, best_action]
            print(f"    State {state}: bid={best_bid:.3f} (Q={q_val:.4f})")

    # Revenue convergence
    revs = result.revenues
    last_1k = revs[-1000:]
    print(f"\nFinal 1000 episodes: mean={np.mean(last_1k):.4f}, std={np.std(last_1k):.4f}")
    if result.bne_revenue:
        ratio = np.mean(last_1k) / result.bne_revenue
        print(f"Ratio to BNE: {ratio:.4f}")


def _verbose_exp2(result):
    Q = result.final_state  # (n_bidders, n_signal_bins, n_bid_actions)
    params = result.params
    n_bidders = params["n_bidders"]

    print(f"\nQ-table shape: {Q.shape}")
    print(f"Q-table range: [{Q.min():.4f}, {Q.max():.4f}]")
    print(f"Q-table mean: {Q.mean():.4f}, std: {Q.std():.4f}")
    nonzero = np.count_nonzero(Q)
    total = Q.size
    print(f"Sparsity: {nonzero}/{total} non-zero ({100 * nonzero / total:.1f}%)")

    n_signal_bins = Q.shape[1]
    n_bid_actions = Q.shape[2]
    bids = np.linspace(0, 1, n_bid_actions)
    signals = np.linspace(0, 1, n_signal_bins)

    print("\nConverged policy table (signal bin -> best bid):")
    for bidder in range(min(n_bidders, 2)):
        print(f"  Bidder {bidder}:")
        print(f"    {'Signal':>8s}  {'Bid':>8s}  {'Q-value':>10s}")
        for s_idx in range(n_signal_bins):
            best_a = np.argmax(Q[bidder, s_idx])
            best_bid = bids[best_a]
            q_val = Q[bidder, s_idx, best_a]
            print(f"    {signals[s_idx]:8.2f}  {best_bid:8.3f}  {q_val:10.4f}")

    # BNE comparison
    if result.bne_revenue is not None:
        print(f"\nBNE revenue: {result.bne_revenue:.4f}")
        print(f"Achieved revenue (last 1000): {result.summary.get('avg_rev_last_1000', 'N/A')}")

    # BNE bid coefficient for reference
    eta = params["eta"]
    alpha_val = 1.0 - 0.5 * eta
    beta_val = 0.5 * eta / max(n_bidders - 1, 1)
    phi = ((n_bidders - 1) / n_bidders) * (alpha_val + n_bidders * beta_val / 2.0)
    print(f"\nBNE bid coefficient (phi): {phi:.4f}")
    print(f"BNE bid function: b(s) = {phi:.3f} * s")

    # Check if learned bids match BNE slope
    print("\nBNE slope comparison:")
    for bidder in range(min(n_bidders, 2)):
        learned_bids = [bids[np.argmax(Q[bidder, s])] for s in range(n_signal_bins)]
        if n_signal_bins > 1 and signals[-1] > signals[0]:
            slope = (learned_bids[-1] - learned_bids[0]) / (signals[-1] - signals[0])
            print(f"  Bidder {bidder}: learned slope={slope:.3f}, BNE slope={phi:.3f}")


def _verbose_exp3(result):
    bandits = result.final_state
    params = result.params

    print(f"\nAlgorithm: {params['algorithm']}")
    print(f"Number of bandits: {len(bandits)}")

    for i, b in enumerate(bandits):
        print(f"\n  Bandit {i}:")
        n_actions = len(b.A) if hasattr(b, "A") else 0
        print(f"    Actions: {n_actions}")

        if hasattr(b, "A") and hasattr(b, "b"):
            print(f"    Learned theta per action:")
            print(f"    {'Action':>8s}  {'theta':>30s}")
            for a_idx in range(min(n_actions, 10)):
                A_inv = np.linalg.solve(b.A[a_idx], np.eye(b.A[a_idx].shape[0]))
                theta = A_inv @ b.b[a_idx]
                theta_str = np.array2string(theta.flatten(), precision=3, separator=", ")
                print(f"    {a_idx:8d}  {theta_str}")

    # Action distribution from round history
    if result.round_history:
        df = pd.DataFrame(result.round_history)
        print("\n  Action distributions (last 1000 rounds):")
        tail = df[df["episode"] >= len(result.revenues) - 1000]
        for bidder_id in sorted(tail["bidder_id"].unique()):
            bid_vals = tail[tail["bidder_id"] == bidder_id]["chosen_bid"]
            print(f"    Bidder {bidder_id}: mean={bid_vals.mean():.3f}, "
                  f"std={bid_vals.std():.3f}, min={bid_vals.min():.3f}, max={bid_vals.max():.3f}")

    # BNE comparison
    if result.bne_revenue is not None:
        print(f"\nBNE revenue: {result.bne_revenue:.4f}")
        print(f"Achieved revenue (last 1000): {result.summary.get('avg_rev_last_1000', 'N/A')}")


def _verbose_exp4(result):
    agents = result.final_state
    episode_data = result.round_history
    params = result.params

    print(f"\nAgents: {len(agents)}")
    for i, agent in enumerate(agents):
        print(f"\n  Agent {i}:")
        print(f"    Final mu: {agent.mu:.4f}")
        print(f"    Budget: {agent.budget:.4f}")
        if agent.mu_history:
            mu_arr = np.array(agent.mu_history)
            print(f"    Mu history (last episode): mean={mu_arr.mean():.4f}, "
                  f"std={mu_arr.std():.4f}, final={mu_arr[-1]:.4f}")
        if agent.bid_history:
            bid_arr = np.array(agent.bid_history)
            print(f"    Bid history (last episode): mean={bid_arr.mean():.4f}, "
                  f"std={bid_arr.std():.4f}")
        if hasattr(agent, "payment_history") and agent.payment_history:
            pay_arr = np.array(agent.payment_history)
            total_pay = pay_arr.sum()
            print(f"    Total payments (last episode): {total_pay:.4f}")
            print(f"    Budget utilization: {total_pay / agent.budget:.2%}" if agent.budget > 0 else "")

    # Episode-level stats
    if episode_data:
        revs = [ep["platform_revenue"] for ep in episode_data]
        btvs = [ep["bid_to_value"] for ep in episode_data]
        print(f"\nEpisode statistics ({len(episode_data)} episodes):")
        print(f"  Revenue: mean={np.mean(revs):.4f}, std={np.std(revs):.4f}")
        print(f"  Bid-to-value: mean={np.mean(btvs):.4f}, std={np.std(btvs):.4f}")

        # Burn-in comparison
        burn_in = min(10, len(episode_data))
        if len(episode_data) > burn_in:
            early = np.mean(revs[:burn_in])
            late = np.mean(revs[burn_in:])
            print(f"  Warm-start: early mean={early:.4f}, post-burn mean={late:.4f}")

# ---------------------------------------------------------------------------
# Stdout tee
# ---------------------------------------------------------------------------

class TeeOutput:
    """Tees stdout to both console and a log file."""

    def __init__(self, filepath):
        self.file = open(filepath, "w")
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.stdout
        self.file.close()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified single-run deep dive analysis tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3, 4],
                        help="Experiment number")
    parser.add_argument("--param", action="append", default=[],
                        help="Parameter override as key=value (repeatable)")
    parser.add_argument("--list-params", action="store_true",
                        help="List available parameters and defaults, then exit")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/deep_dive/expN_TIMESTAMP)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed post-hoc diagnostics")
    parser.add_argument("--no-plots", action="store_true", help="Skip trace plot generation")
    parser.add_argument("--no-save", action="store_true", help="Console-only, skip file output")
    return parser.parse_args()


def build_params(exp_num, param_overrides):
    """Build parameter dict from defaults + CLI overrides."""
    params = dict(DEFAULTS[exp_num])
    for override in param_overrides:
        if "=" not in override:
            print(f"Error: --param must be key=value, got: {override}")
            sys.exit(1)
        key, val = override.split("=", 1)
        if key not in params:
            print(f"Error: unknown parameter '{key}' for experiment {exp_num}")
            print(f"Available: {', '.join(sorted(params.keys()))}")
            sys.exit(1)
        # Type coercion
        if key in PARAM_TYPES:
            val = PARAM_TYPES[key](val)
        elif isinstance(params[key], int):
            val = int(val)
        elif isinstance(params[key], float):
            val = float(val)
        params[key] = val
    return params

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.list_params:
        defaults = DEFAULTS[args.exp]
        print(f"\nExperiment {args.exp} parameters:\n")
        print(f"  {'Parameter':<25s} {'Default':<20s} {'Type'}")
        print(f"  {'-'*25} {'-'*20} {'-'*10}")
        for k, v in sorted(defaults.items()):
            print(f"  {k:<25s} {str(v):<20s} {type(v).__name__}")
        print(f"\nUsage: --param {list(defaults.keys())[0]}={list(defaults.values())[0]}")
        return

    params = build_params(args.exp, args.param)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join("results", "deep_dive", f"exp{args.exp}_{timestamp}")

    # Set up tee if saving
    tee = None
    if not args.no_save:
        os.makedirs(output_dir, exist_ok=True)
        tee = TeeOutput(os.path.join(output_dir, "console.log"))

    print(f"Deep dive: Experiment {args.exp}")
    print(f"Seed: {args.seed}")
    print(f"Parameters: {json.dumps(params, indent=2)}")
    print()

    # Run experiment
    t0 = time.time()
    result = RUNNERS[args.exp](params, args.seed)
    result.elapsed_sec = time.time() - t0
    print(f"\nCompleted in {result.elapsed_sec:.1f}s")

    # Print summary
    print(f"\n--- Summary ---")
    for k, v in sorted(result.summary.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    if result.bne_revenue is not None:
        print(f"  bne_revenue: {result.bne_revenue:.6f}")

    # Verbose diagnostics
    if args.verbose:
        print_verbose(result)

    # Save
    if not args.no_save:
        print()
        save_results(result, output_dir, args.seed)

    # Plot
    if not args.no_plots and not args.no_save:
        plot_trace(result, output_dir)

    if tee:
        tee.close()


if __name__ == "__main__":
    main()
