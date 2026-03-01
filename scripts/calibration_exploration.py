#!/usr/bin/env python3
"""
Exploration Calibration: Boltzmann vs Epsilon-Greedy

Measures effective exploration intensity (Shannon entropy) of both strategies
under identical parameter decay schedules, then computes a calibrated Boltzmann
temperature schedule that matches epsilon-greedy entropy at every episode.

Modes:
    single  -- Original per-episode calibration (default)
    multi   -- Constant-multiplier fitting across {FPA,SPA} x {2,4} x {linear,exp}
    exp3    -- Measure exploration entropy for LinUCB and Thompson Sampling

Usage:
    python scripts/calibration_exploration.py [--mode single] [--seeds 10]
    python scripts/calibration_exploration.py --mode multi [--seeds 5]
    python scripts/calibration_exploration.py --mode exp3 [--seeds 5]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar

# ---------------------------------------------------------------------------
# Auction payoff (minimal extraction from exp1.py)
# ---------------------------------------------------------------------------

def get_rewards(bids, valuations, auction_type="first", reserve_price=0.0):
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

    if auction_type == "first":
        rewards[winner_global] = valuations[winner_global] - winner_bid
    else:
        if len(valid_indices) == len(highest_idx_local):
            if len(valid_indices) == 1:
                second_highest_bid = reserve_price
            else:
                second_highest_bid = highest_bid
        else:
            second_idx_local = None
            for idx_l in sorted_indices:
                if idx_l not in highest_idx_local:
                    second_idx_local = idx_l
                    break
            second_highest_bid = (
                highest_bid if second_idx_local is None
                else valid_bids[second_idx_local]
            )
        rewards[winner_global] = valuations[winner_global] - second_highest_bid

    return rewards, winner_global, winner_bid


# ---------------------------------------------------------------------------
# Instrumented Q-learning with per-episode entropy tracking
# ---------------------------------------------------------------------------

def run_instrumented(exploration, episodes, seed,
                     n_bidders=2, n_actions=11, alpha=0.1, gamma=0.95,
                     auction_type="first", reserve_price=0.0,
                     decay_type="linear"):
    """
    Run Q-learning with entropy instrumentation.

    Returns:
        entropies: array of shape (episodes,) with per-episode Shannon entropy
        Q: final Q-table
    """
    rng = np.random.RandomState(seed)
    action_space = np.linspace(0, 1, n_actions)
    Q = np.zeros((n_bidders, 1, n_actions))  # minimal state (1 state)

    eps_start, eps_end = 1.0, 0.01
    temp_start, temp_end = 1.0, 0.01
    decay_end = int(0.9 * episodes)

    entropies = np.empty(episodes)
    max_entropy = np.log(n_actions)
    past_winner_bid = 0.0

    for ep in range(episodes):
        # Decay schedule
        if ep < decay_end:
            if decay_type == "linear":
                eps = eps_start - (ep / decay_end) * (eps_start - eps_end)
                temp = temp_start - (ep / decay_end) * (temp_start - temp_end)
            else:
                eps = eps_start * (eps_end / eps_start) ** (ep / decay_end)
                temp = temp_start * (temp_end / temp_start) ** (ep / decay_end)
        else:
            eps = 0.0
            temp = temp_end

        valuations = np.ones(n_bidders)
        s = 0  # minimal state

        # Track entropy across bidders this episode
        ep_entropy = 0.0
        chosen_actions = []
        for i in range(n_bidders):
            qvals = Q[i, s]
            if exploration == "boltzmann":
                shifted = (qvals - np.max(qvals)) / max(temp, 1e-10)
                ex = np.exp(shifted)
                probs = ex / np.sum(ex)
                a_i = rng.choice(n_actions, p=probs)
            else:  # egreedy
                if rng.rand() > eps:
                    a_i = np.argmax(qvals)
                    # Probability distribution: 1 on argmax, 0 elsewhere
                    probs = np.zeros(n_actions)
                    probs[a_i] = 1.0
                    # For entropy: greedy part contributes 0 entropy
                    # Random part (eps) is uniform
                    # Actual distribution: (1-eps)*one_hot + eps/n_actions
                    probs = np.full(n_actions, eps / n_actions)
                    probs[np.argmax(qvals)] += (1.0 - eps)
                else:
                    a_i = rng.randint(n_actions)
                    probs = np.full(n_actions, eps / n_actions)
                    probs[np.argmax(qvals)] += (1.0 - eps)

            # Shannon entropy of action-selection distribution
            h = -np.sum(probs * np.log(probs + 1e-30))
            ep_entropy += h
            chosen_actions.append(a_i)

        entropies[ep] = ep_entropy / n_bidders  # average across bidders

        # Execute auction
        bids = np.array([action_space[a] for a in chosen_actions])
        rew, winner, winner_bid_val = get_rewards(
            bids, valuations, auction_type, reserve_price
        )

        # Q-update (asynchronous, minimal state)
        s_next = 0
        for i in range(n_bidders):
            old_q = Q[i, s, chosen_actions[i]]
            td_target = rew[i] + gamma * np.max(Q[i, s_next])
            Q[i, s, chosen_actions[i]] = old_q + alpha * (td_target - old_q)

        past_winner_bid = winner_bid_val if winner != -1 else 0.0

    return entropies, Q


# ---------------------------------------------------------------------------
# Compute Boltzmann entropy for a given Q-table and temperature
# ---------------------------------------------------------------------------

def boltzmann_entropy(Q, temp, n_bidders, n_actions):
    """Average Shannon entropy of Boltzmann policy across bidders."""
    total_h = 0.0
    for i in range(n_bidders):
        qvals = Q[i, 0]
        shifted = (qvals - np.max(qvals)) / max(temp, 1e-10)
        ex = np.exp(shifted)
        probs = ex / np.sum(ex)
        total_h += -np.sum(probs * np.log(probs + 1e-30))
    return total_h / n_bidders


# ---------------------------------------------------------------------------
# Calibration: find temperature that matches target entropy
# ---------------------------------------------------------------------------

def calibrate_temperature(Q, target_entropy, n_bidders, n_actions,
                          temp_lo=1e-6, temp_hi=100.0):
    """
    Use Brent's method to find temperature tau* such that
    boltzmann_entropy(Q, tau*) = target_entropy.
    """
    # Check if target is achievable
    h_lo = boltzmann_entropy(Q, temp_lo, n_bidders, n_actions)
    h_hi = boltzmann_entropy(Q, temp_hi, n_bidders, n_actions)

    if target_entropy <= h_lo:
        return temp_lo
    if target_entropy >= h_hi:
        return temp_hi

    def objective(tau):
        return boltzmann_entropy(Q, tau, n_bidders, n_actions) - target_entropy

    return brentq(objective, temp_lo, temp_hi, xtol=1e-8, maxiter=200)


# ---------------------------------------------------------------------------
# Main calibration pipeline
# ---------------------------------------------------------------------------

def run_calibration(n_seeds=10, episodes=100_000, output_dir="results/calibration",
                    no_plots=False, n_bidders=2, n_actions=11, alpha=0.1,
                    gamma=0.95, decay_type="linear"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Running calibration: {n_seeds} seeds x {episodes} episodes")
    print(f"Settings: n_bidders={n_bidders}, n_actions={n_actions}, "
          f"alpha={alpha}, gamma={gamma}, decay={decay_type}")
    max_entropy = np.log(n_actions)

    # ------------------------------------------------------------------
    # Pass 1: Run epsilon-greedy simulations
    # ------------------------------------------------------------------
    print("\n--- Pass 1: Epsilon-greedy simulations ---")
    eg_entropies = np.empty((n_seeds, episodes))
    for s in range(n_seeds):
        print(f"  Seed {s+1}/{n_seeds}", end="\r")
        eg_entropies[s], _ = run_instrumented(
            "egreedy", episodes, seed=s * 1000,
            n_bidders=n_bidders, n_actions=n_actions,
            alpha=alpha, gamma=gamma, decay_type=decay_type,
        )
    eg_mean = eg_entropies.mean(axis=0)
    eg_std = eg_entropies.std(axis=0)
    print(f"  Epsilon-greedy done. Mean final entropy: {eg_mean[-1]:.6f}")

    # ------------------------------------------------------------------
    # Pass 2: Run Boltzmann simulations (default schedule)
    # ------------------------------------------------------------------
    print("\n--- Pass 2: Boltzmann simulations (default schedule) ---")
    bz_entropies = np.empty((n_seeds, episodes))
    bz_Q_tables = []
    for s in range(n_seeds):
        print(f"  Seed {s+1}/{n_seeds}", end="\r")
        ent, Q = run_instrumented(
            "boltzmann", episodes, seed=s * 1000,
            n_bidders=n_bidders, n_actions=n_actions,
            alpha=alpha, gamma=gamma, decay_type=decay_type,
        )
        bz_entropies[s] = ent
        bz_Q_tables.append(Q)
    bz_mean = bz_entropies.mean(axis=0)
    bz_std = bz_entropies.std(axis=0)
    print(f"  Boltzmann done. Mean final entropy: {bz_mean[-1]:.6f}")

    # ------------------------------------------------------------------
    # Pass 3: Calibrate Boltzmann temperature schedule
    # ------------------------------------------------------------------
    print("\n--- Pass 3: Calibrating temperature schedule ---")
    # Use first seed's Q-table trajectory for calibration.
    # Re-run with snapshots to get Q at each checkpoint.
    n_checkpoints = min(500, episodes)
    checkpoint_indices = np.linspace(0, episodes - 1, n_checkpoints, dtype=int)

    # Re-run Boltzmann with Q-table snapshots at checkpoints
    print("  Re-running Boltzmann seed 0 with Q-table snapshots...")
    Q_snapshots = _run_with_snapshots(
        episodes, seed=0, checkpoint_indices=checkpoint_indices,
        n_bidders=n_bidders, n_actions=n_actions,
        alpha=alpha, gamma=gamma, decay_type=decay_type,
    )

    # Original temperature schedule at checkpoints
    decay_end = int(0.9 * episodes)
    original_temps = np.empty(n_checkpoints)
    for ci, ep in enumerate(checkpoint_indices):
        if ep < decay_end:
            if decay_type == "linear":
                original_temps[ci] = 1.0 - (ep / decay_end) * (1.0 - 0.01)
            else:
                original_temps[ci] = 1.0 * (0.01 / 1.0) ** (ep / decay_end)
        else:
            original_temps[ci] = 0.01

    # Target entropy at checkpoints (from epsilon-greedy mean)
    target_entropies = eg_mean[checkpoint_indices]

    # Calibrate
    calibrated_temps = np.empty(n_checkpoints)
    for ci in range(n_checkpoints):
        Q_snap = Q_snapshots[ci]
        calibrated_temps[ci] = calibrate_temperature(
            Q_snap, target_entropies[ci], n_bidders, n_actions,
        )
        if (ci + 1) % 100 == 0:
            print(f"  Calibrated {ci+1}/{n_checkpoints} checkpoints", end="\r")
    print(f"  Calibration done. Temp multiplier range: "
          f"{(calibrated_temps / (original_temps + 1e-30)).min():.2f}x "
          f"to {(calibrated_temps / (original_temps + 1e-30)).max():.2f}x")

    # ------------------------------------------------------------------
    # Pass 4: Verify calibrated schedule
    # ------------------------------------------------------------------
    print("\n--- Pass 4: Verification run with calibrated schedule ---")
    # Interpolate calibrated temps to all episodes
    calibrated_full = np.interp(
        np.arange(episodes),
        checkpoint_indices,
        calibrated_temps,
    )
    cal_entropies = np.empty((n_seeds, episodes))
    for s in range(n_seeds):
        print(f"  Seed {s+1}/{n_seeds}", end="\r")
        cal_entropies[s], _ = _run_calibrated(
            episodes, seed=s * 1000, temp_schedule=calibrated_full,
            n_bidders=n_bidders, n_actions=n_actions,
            alpha=alpha, gamma=gamma,
        )
    cal_mean = cal_entropies.mean(axis=0)
    cal_std = cal_entropies.std(axis=0)

    rmse = np.sqrt(np.mean((cal_mean - eg_mean) ** 2))
    print(f"  Verification done. RMSE(calibrated vs egreedy): {rmse:.6f}")

    # ------------------------------------------------------------------
    # Output: CSVs
    # ------------------------------------------------------------------
    print("\n--- Writing outputs ---")

    # Entropy trajectories (subsample for tractable CSV size)
    subsample = max(1, episodes // 2000)
    ep_idx = np.arange(0, episodes, subsample)
    rows = []
    for s in range(n_seeds):
        for ei in ep_idx:
            rows.append({
                "episode": int(ei),
                "seed": s,
                "strategy": "egreedy",
                "entropy": float(eg_entropies[s, ei]),
            })
            rows.append({
                "episode": int(ei),
                "seed": s,
                "strategy": "boltzmann",
                "entropy": float(bz_entropies[s, ei]),
            })
            rows.append({
                "episode": int(ei),
                "seed": s,
                "strategy": "boltzmann_calibrated",
                "entropy": float(cal_entropies[s, ei]),
            })
    df_traj = pd.DataFrame(rows)
    traj_path = os.path.join(output_dir, "entropy_trajectories.csv")
    df_traj.to_csv(traj_path, index=False)
    print(f"  {traj_path}")

    # Calibrated schedule
    df_cal = pd.DataFrame({
        "episode": checkpoint_indices,
        "original_tau": original_temps,
        "calibrated_tau": calibrated_temps,
        "target_entropy": target_entropies,
        "multiplier": calibrated_temps / (original_temps + 1e-30),
    })
    cal_path = os.path.join(output_dir, "calibrated_schedule.csv")
    df_cal.to_csv(cal_path, index=False)
    print(f"  {cal_path}")

    # Summary text
    entropy_ratio_early = bz_mean[int(0.1 * episodes)] / (eg_mean[int(0.1 * episodes)] + 1e-30)
    entropy_ratio_mid = bz_mean[int(0.5 * episodes)] / (eg_mean[int(0.5 * episodes)] + 1e-30)
    entropy_ratio_late = bz_mean[int(0.85 * episodes)] / (eg_mean[int(0.85 * episodes)] + 1e-30)
    max_gap_ep = int(np.argmax(np.abs(bz_mean - eg_mean)))
    max_gap_val = float(np.abs(bz_mean[max_gap_ep] - eg_mean[max_gap_ep]))
    multiplier = calibrated_temps / (original_temps + 1e-30)

    summary_lines = [
        "Exploration Calibration Summary",
        "=" * 40,
        f"Seeds: {n_seeds}",
        f"Episodes: {episodes}",
        f"Settings: n_bidders={n_bidders}, n_actions={n_actions}, "
        f"alpha={alpha}, gamma={gamma}, decay={decay_type}",
        f"Max entropy (uniform over {n_actions} actions): {max_entropy:.4f}",
        "",
        "Entropy Ratio (Boltzmann / Epsilon-Greedy):",
        f"  At 10% of training: {entropy_ratio_early:.4f}",
        f"  At 50% of training: {entropy_ratio_mid:.4f}",
        f"  At 85% of training: {entropy_ratio_late:.4f}",
        "",
        f"Max divergence at episode {max_gap_ep}: {max_gap_val:.4f}",
        "",
        "Calibration Multiplier (calibrated_tau / original_tau):",
        f"  Min: {multiplier.min():.4f}",
        f"  Max: {multiplier.max():.4f}",
        f"  Mean: {multiplier.mean():.4f}",
        f"  Median: {np.median(multiplier):.4f}",
        "",
        f"Verification RMSE (calibrated Boltzmann vs egreedy): {rmse:.6f}",
    ]
    summary_text = "\n".join(summary_lines) + "\n"
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"  {summary_path}")
    print()
    print(summary_text)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if not no_plots:
        _plot_entropy_comparison(
            eg_mean, eg_std, bz_mean, bz_std, cal_mean, cal_std,
            episodes, max_entropy, output_dir,
        )
        _plot_param_vs_entropy(
            eg_mean, bz_mean, episodes, max_entropy, decay_type, output_dir,
        )
        _plot_calibrated_schedule(
            checkpoint_indices, original_temps, calibrated_temps,
            eg_mean, bz_mean, cal_mean, episodes, output_dir,
        )
        print("  Plots saved.")


# ---------------------------------------------------------------------------
# Re-run Boltzmann with Q-table snapshots
# ---------------------------------------------------------------------------

def _run_with_snapshots(episodes, seed, checkpoint_indices,
                        n_bidders=2, n_actions=11, alpha=0.1, gamma=0.95,
                        decay_type="linear"):
    """Run Boltzmann Q-learning and snapshot Q at specified episodes."""
    rng = np.random.RandomState(seed)
    action_space = np.linspace(0, 1, n_actions)
    Q = np.zeros((n_bidders, 1, n_actions))

    decay_end = int(0.9 * episodes)
    checkpoint_set = set(checkpoint_indices)
    snapshots = {}

    for ep in range(episodes):
        if ep in checkpoint_set:
            snapshots[ep] = Q.copy()

        if ep < decay_end:
            if decay_type == "linear":
                temp = 1.0 - (ep / decay_end) * (1.0 - 0.01)
            else:
                temp = 1.0 * (0.01 / 1.0) ** (ep / decay_end)
        else:
            temp = 0.01

        valuations = np.ones(n_bidders)
        chosen_actions = []
        for i in range(n_bidders):
            qvals = Q[i, 0]
            shifted = (qvals - np.max(qvals)) / max(temp, 1e-10)
            ex = np.exp(shifted)
            probs = ex / np.sum(ex)
            a_i = rng.choice(n_actions, p=probs)
            chosen_actions.append(a_i)

        bids = np.array([action_space[a] for a in chosen_actions])
        rew, winner, winner_bid_val = get_rewards(
            bids, valuations, "first", 0.0
        )

        for i in range(n_bidders):
            old_q = Q[i, 0, chosen_actions[i]]
            td_target = rew[i] + gamma * np.max(Q[i, 0])
            Q[i, 0, chosen_actions[i]] = old_q + alpha * (td_target - old_q)

    # Return ordered list
    return [snapshots[ep] for ep in checkpoint_indices]


# ---------------------------------------------------------------------------
# Run Boltzmann with externally supplied temperature schedule
# ---------------------------------------------------------------------------

def _run_calibrated(episodes, seed, temp_schedule,
                    n_bidders=2, n_actions=11, alpha=0.1, gamma=0.95):
    """Run Boltzmann Q-learning with a pre-computed temperature schedule."""
    rng = np.random.RandomState(seed)
    action_space = np.linspace(0, 1, n_actions)
    Q = np.zeros((n_bidders, 1, n_actions))

    entropies = np.empty(episodes)

    for ep in range(episodes):
        temp = temp_schedule[ep]
        valuations = np.ones(n_bidders)

        ep_entropy = 0.0
        chosen_actions = []
        for i in range(n_bidders):
            qvals = Q[i, 0]
            shifted = (qvals - np.max(qvals)) / max(temp, 1e-10)
            ex = np.exp(shifted)
            probs = ex / np.sum(ex)
            a_i = rng.choice(n_actions, p=probs)

            h = -np.sum(probs * np.log(probs + 1e-30))
            ep_entropy += h
            chosen_actions.append(a_i)

        entropies[ep] = ep_entropy / n_bidders

        bids = np.array([action_space[a] for a in chosen_actions])
        rew, winner, winner_bid_val = get_rewards(
            bids, valuations, "first", 0.0
        )

        for i in range(n_bidders):
            old_q = Q[i, 0, chosen_actions[i]]
            td_target = rew[i] + gamma * np.max(Q[i, 0])
            Q[i, 0, chosen_actions[i]] = old_q + alpha * (td_target - old_q)

    return entropies, Q


# ---------------------------------------------------------------------------
# Run Boltzmann with constant temperature multiplier
# ---------------------------------------------------------------------------

def _run_with_multiplier(episodes, seed, multiplier, decay_type,
                         n_bidders=2, n_actions=11, alpha=0.1, gamma=0.95,
                         auction_type="first", reserve_price=0.0):
    """Run Boltzmann Q-learning with temperature = raw_schedule * multiplier."""
    rng = np.random.RandomState(seed)
    action_space = np.linspace(0, 1, n_actions)
    Q = np.zeros((n_bidders, 1, n_actions))

    decay_end = int(0.9 * episodes)
    temp_start, temp_end = 1.0, 0.01
    entropies = np.empty(episodes)

    for ep in range(episodes):
        if ep < decay_end:
            if decay_type == "linear":
                raw_temp = temp_start - (ep / decay_end) * (temp_start - temp_end)
            else:
                raw_temp = temp_start * (temp_end / temp_start) ** (ep / decay_end)
        else:
            raw_temp = temp_end
        temp = raw_temp * multiplier

        valuations = np.ones(n_bidders)
        ep_entropy = 0.0
        chosen_actions = []
        for i in range(n_bidders):
            qvals = Q[i, 0]
            shifted = (qvals - np.max(qvals)) / max(temp, 1e-10)
            ex = np.exp(shifted)
            probs = ex / np.sum(ex)
            a_i = rng.choice(n_actions, p=probs)
            h = -np.sum(probs * np.log(probs + 1e-30))
            ep_entropy += h
            chosen_actions.append(a_i)

        entropies[ep] = ep_entropy / n_bidders

        bids = np.array([action_space[a] for a in chosen_actions])
        rew, winner, winner_bid_val = get_rewards(
            bids, valuations, auction_type, reserve_price
        )

        for i in range(n_bidders):
            old_q = Q[i, 0, chosen_actions[i]]
            td_target = rew[i] + gamma * np.max(Q[i, 0])
            Q[i, 0, chosen_actions[i]] = old_q + alpha * (td_target - old_q)

    return entropies, Q


# ---------------------------------------------------------------------------
# Constant-multiplier fitting
# ---------------------------------------------------------------------------

def fit_constant_multiplier(auction_type="first", n_bidders=2, decay_type="linear",
                            n_seeds=5, episodes=100_000, n_actions=11,
                            alpha=0.1, gamma=0.95, reserve_price=0.0):
    """Find constant M minimizing RMSE(Boltzmann(temp*M) entropy, egreedy entropy).

    Uses Q-table snapshots from a single Boltzmann run plus analytical entropy
    computation, so the optimization loop is fast (no re-simulation per M).
    """
    # Step 1: epsilon-greedy target entropy (averaged over seeds)
    eg_entropies = np.empty((n_seeds, episodes))
    for s in range(n_seeds):
        eg_entropies[s], _ = run_instrumented(
            "egreedy", episodes, seed=s * 1000,
            n_bidders=n_bidders, n_actions=n_actions,
            alpha=alpha, gamma=gamma,
            auction_type=auction_type, decay_type=decay_type,
        )
    eg_mean = eg_entropies.mean(axis=0)

    # Step 2: Boltzmann Q-table snapshots (single seed, default schedule)
    n_checkpoints = min(200, episodes)
    checkpoint_indices = np.linspace(0, episodes - 1, n_checkpoints, dtype=int)
    Q_snapshots = _run_with_snapshots(
        episodes, seed=0, checkpoint_indices=checkpoint_indices,
        n_bidders=n_bidders, n_actions=n_actions,
        alpha=alpha, gamma=gamma, decay_type=decay_type,
    )

    # Raw temperature at checkpoints
    decay_end = int(0.9 * episodes)
    raw_temps = np.empty(n_checkpoints)
    for ci, ep in enumerate(checkpoint_indices):
        if ep < decay_end:
            if decay_type == "linear":
                raw_temps[ci] = 1.0 - (ep / decay_end) * (1.0 - 0.01)
            else:
                raw_temps[ci] = 1.0 * (0.01 / 1.0) ** (ep / decay_end)
        else:
            raw_temps[ci] = 0.01

    target_at_checkpoints = eg_mean[checkpoint_indices]

    # Step 3: Optimize M analytically
    def objective(M):
        pred = np.array([
            boltzmann_entropy(Q_snapshots[ci], raw_temps[ci] * M, n_bidders, n_actions)
            for ci in range(n_checkpoints)
        ])
        return np.sqrt(np.mean((pred - target_at_checkpoints) ** 2))

    result = minimize_scalar(objective, bounds=(0.01, 2.0), method="bounded",
                             options={"xatol": 0.005, "maxiter": 50})
    return result.x, result.fun, eg_mean


# ---------------------------------------------------------------------------
# Multi-setting calibration grid
# ---------------------------------------------------------------------------

def run_multi_setting_calibration(n_seeds=5, episodes=100_000, output_dir="results/calibration",
                                  no_plots=False):
    """Run multiplier fitting across 16 settings: {FPA,SPA} x {2,4} x {linear,exp} x {gamma=0,0.95}.

    Outputs:
        multiplier_grid.csv  -- per-setting M and RMSE
        summary.txt          -- recommended M per gamma level (median across settings)
        entropy_comparison.png -- verification plot for recommended M
    """
    os.makedirs(output_dir, exist_ok=True)

    settings = [
        (at, nb, dt, g)
        for at in ["first", "second"]
        for nb in [2, 4]
        for dt in ["linear", "exponential"]
        for g in [0.0, 0.95]
    ]

    rows = []
    for auction_type, n_bidders, decay_type, g in settings:
        print(f"\nFitting M for {auction_type}, n={n_bidders}, {decay_type}, gamma={g}...")
        M, rmse, _ = fit_constant_multiplier(
            auction_type=auction_type, n_bidders=n_bidders,
            decay_type=decay_type, n_seeds=n_seeds, episodes=episodes,
            gamma=g,
        )
        print(f"  M = {M:.4f}, RMSE = {rmse:.6f}")
        rows.append({
            "auction_type": auction_type,
            "n_bidders": n_bidders,
            "decay_type": decay_type,
            "gamma": g,
            "multiplier": M,
            "rmse": rmse,
        })

    df = pd.DataFrame(rows)
    grid_path = os.path.join(output_dir, "multiplier_grid.csv")
    df.to_csv(grid_path, index=False)
    print(f"\n  {grid_path}")

    # Summary — report per-gamma medians
    median_M = float(df["multiplier"].median())
    by_gamma = df.groupby("gamma")["multiplier"].median()
    by_decay = df.groupby("decay_type")["multiplier"].median()

    summary_lines = [
        "Constant-Multiplier Calibration Summary",
        "=" * 50,
        f"Settings tested: {len(settings)}",
        f"Seeds per setting: {n_seeds}",
        f"Episodes: {episodes}",
        "",
        "Per-setting results:",
    ]
    for _, row in df.iterrows():
        summary_lines.append(
            f"  {row['auction_type']:6s} n={row['n_bidders']:.0f} "
            f"{row['decay_type']:11s} gamma={row['gamma']:.2f}: "
            f"M={row['multiplier']:.4f}, RMSE={row['rmse']:.6f}"
        )
    summary_lines.extend([
        "",
        f"Overall median M: {median_M:.4f}",
        "",
        "Recommended M per gamma level:",
    ])
    for g_val in sorted(by_gamma.index):
        summary_lines.append(f"  gamma={g_val:.2f}: M={by_gamma[g_val]:.4f}")
    summary_lines.extend([
        "",
        f"  Linear decay median:      {by_decay.get('linear', 0):.4f}",
        f"  Exponential decay median:  {by_decay.get('exponential', 0):.4f}",
    ])

    summary_text = "\n".join(summary_lines) + "\n"
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"  {summary_path}")
    print()
    print(summary_text)

    # Verification plot with recommended M (use gamma=0.95 for backward compat)
    if not no_plots:
        _plot_multiplier_verification(
            by_gamma.get(0.95, median_M), output_dir, episodes, n_seeds,
        )

    return dict(by_gamma)


# ---------------------------------------------------------------------------
# Exp3 exploration entropy measurement
# ---------------------------------------------------------------------------

def measure_exp3_entropy(episodes=100_000, n_seeds=5,
                         output_dir="results/calibration"):
    """Measure empirical action-selection entropy for LinUCB and Thompson Sampling.

    LinUCB is deterministic given context, so entropy is measured from empirical
    action frequency over 1000-round windows. Thompson Sampling has inherent
    randomness from posterior sampling.

    Outputs:
        exp3_entropy.csv -- [algorithm, intensity, seed, epoch_pct, mean_entropy]
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from experiments.exp3 import (
        LinUCB, ContextualThompsonSampling, EXPLORATION_MAP,
        get_rewards as exp3_get_rewards,
    )

    os.makedirs(output_dir, exist_ok=True)

    n_actions = 11
    n_bidders = 2
    eta = 0.5
    lam = 1.0
    window = 1000
    action_space = np.linspace(0, 1, n_actions)

    rows = []
    for algorithm in ["linucb", "thompson"]:
        for intensity in ["low", "high"]:
            explore_val = EXPLORATION_MAP[(algorithm, intensity)]
            print(f"\nMeasuring {algorithm} {intensity} (param={explore_val})...")

            for seed_idx in range(n_seeds):
                seed = seed_idx * 1000
                np.random.seed(seed)
                rng = np.random.default_rng(seed)

                if algorithm == "linucb":
                    bandits = [LinUCB(n_actions, 1, explore_val, lam)
                               for _ in range(n_bidders)]
                else:
                    bandits = [ContextualThompsonSampling(
                                   n_actions, 1, explore_val, lam,
                                   rng=np.random.default_rng(seed + i))
                               for i in range(n_bidders)]

                action_counts = np.zeros((n_bidders, n_actions))
                val_alpha = 1.0 - 0.5 * eta
                val_beta = 0.5 * eta / max(n_bidders - 1, 1)

                for ep in range(episodes):
                    signals = np.random.randint(11, size=n_bidders) / 10.0
                    valuations = val_alpha * signals + val_beta * (signals.sum() - signals)

                    bids = np.empty(n_bidders)
                    actions_chosen = []
                    for i in range(n_bidders):
                        ctx = np.array([signals[i]])
                        a = bandits[i].select_action(ctx)
                        actions_chosen.append(a)
                        bids[i] = action_space[a]
                        action_counts[i, a] += 1

                    rew, winner, _ = exp3_get_rewards(bids, valuations, "first", 0.0)
                    for i in range(n_bidders):
                        bandits[i].update(actions_chosen[i], rew[i],
                                          np.array([signals[i]]))

                    # Compute entropy from empirical action frequency each window
                    if (ep + 1) % window == 0:
                        epoch_pct = (ep + 1) / episodes
                        mean_h = 0.0
                        for i in range(n_bidders):
                            total = action_counts[i].sum()
                            if total > 0:
                                p = action_counts[i] / total
                                p = p[p > 0]
                                mean_h += -np.sum(p * np.log(p))
                        mean_h /= n_bidders
                        rows.append({
                            "algorithm": algorithm,
                            "intensity": intensity,
                            "seed": seed_idx,
                            "epoch_pct": round(epoch_pct, 3),
                            "mean_entropy": mean_h,
                        })
                        action_counts[:] = 0  # reset for next window

    df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "exp3_entropy.csv")
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    # Print summary
    summary = df.groupby(["algorithm", "intensity"])["mean_entropy"].agg(["mean", "std"])
    print("\nExp3 Exploration Entropy Summary:")
    print(summary.to_string())


# ---------------------------------------------------------------------------
# Plot 1: Entropy trajectories
# ---------------------------------------------------------------------------

def _plot_entropy_comparison(eg_mean, eg_std, bz_mean, bz_std,
                             cal_mean, cal_std, episodes, max_entropy,
                             output_dir):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(episodes)
    # Subsample for plotting
    step = max(1, episodes // 2000)
    xs = x[::step]

    ax.fill_between(xs, (eg_mean - eg_std)[::step], (eg_mean + eg_std)[::step],
                     alpha=0.15, color="0.3")
    ax.plot(xs, eg_mean[::step], color="0.3", linewidth=1.5,
            label="Epsilon-greedy")

    ax.fill_between(xs, (bz_mean - bz_std)[::step], (bz_mean + bz_std)[::step],
                     alpha=0.15, color="0.6")
    ax.plot(xs, bz_mean[::step], color="0.6", linewidth=1.5, linestyle="--",
            label="Boltzmann (default)")

    ax.fill_between(xs, (cal_mean - cal_std)[::step], (cal_mean + cal_std)[::step],
                     alpha=0.15, color="0.1")
    ax.plot(xs, cal_mean[::step], color="0.1", linewidth=1.0, linestyle=":",
            label="Boltzmann (calibrated)")

    ax.axhline(max_entropy, color="0.8", linestyle="-", linewidth=0.8,
               label=f"Uniform (ln {int(np.exp(max_entropy)+0.5)})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Shannon Entropy (nats)")
    ax.set_title("Effective Exploration Intensity")
    ax.legend(fontsize=8)
    ax.set_xlim(0, episodes)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    path = os.path.join(output_dir, "entropy_comparison.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Parameter value vs effective entropy
# ---------------------------------------------------------------------------

def _plot_param_vs_entropy(eg_mean, bz_mean, episodes, max_entropy,
                           decay_type, output_dir):
    decay_end = int(0.9 * episodes)
    x = np.arange(episodes)

    # Compute parameter schedules
    eps_schedule = np.empty(episodes)
    tau_schedule = np.empty(episodes)
    for ep in range(episodes):
        if ep < decay_end:
            if decay_type == "linear":
                eps_schedule[ep] = 1.0 - (ep / decay_end) * (1.0 - 0.01)
                tau_schedule[ep] = 1.0 - (ep / decay_end) * (1.0 - 0.01)
            else:
                eps_schedule[ep] = 1.0 * (0.01 / 1.0) ** (ep / decay_end)
                tau_schedule[ep] = 1.0 * (0.01 / 1.0) ** (ep / decay_end)
        else:
            eps_schedule[ep] = 0.0
            tau_schedule[ep] = 0.01

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    step = max(1, episodes // 2000)
    xs = x[::step]

    # Left: epsilon-greedy
    ax1_twin = ax1.twinx()
    ax1.plot(xs, eps_schedule[::step], color="0.7", linewidth=1.5,
             label=r"$\varepsilon$ (parameter)")
    ax1_twin.plot(xs, eg_mean[::step], color="0.2", linewidth=1.5,
                  label="Entropy")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel(r"$\varepsilon$", color="0.7")
    ax1_twin.set_ylabel("Shannon Entropy", color="0.2")
    ax1.set_title("Epsilon-Greedy")
    lines1 = ax1.get_lines() + ax1_twin.get_lines()
    ax1.legend(lines1, [l.get_label() for l in lines1], fontsize=8, loc="center right")

    # Right: Boltzmann
    ax2_twin = ax2.twinx()
    ax2.plot(xs, tau_schedule[::step], color="0.7", linewidth=1.5,
             label=r"$\tau$ (parameter)")
    ax2_twin.plot(xs, bz_mean[::step], color="0.2", linewidth=1.5,
                  label="Entropy")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel(r"$\tau$", color="0.7")
    ax2_twin.set_ylabel("Shannon Entropy", color="0.2")
    ax2.set_title("Boltzmann")
    lines2 = ax2.get_lines() + ax2_twin.get_lines()
    ax2.legend(lines2, [l.get_label() for l in lines2], fontsize=8, loc="center right")

    fig.tight_layout()
    path = os.path.join(output_dir, "param_vs_entropy.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Calibrated schedule
# ---------------------------------------------------------------------------

def _plot_calibrated_schedule(checkpoint_indices, original_temps,
                              calibrated_temps, eg_mean, bz_mean, cal_mean,
                              episodes, output_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # Top: temperature schedules
    ax1.plot(checkpoint_indices, original_temps, color="0.6", linewidth=1.5,
             linestyle="--", label=r"Original $\tau$")
    ax1.plot(checkpoint_indices, calibrated_temps, color="0.1", linewidth=1.5,
             label=r"Calibrated $\tau^*$")
    ax1.set_ylabel("Temperature")
    ax1.set_title("Boltzmann Temperature Schedules")
    ax1.legend(fontsize=8)
    ax1.set_ylim(bottom=0)

    # Bottom: resulting entropy
    step = max(1, episodes // 2000)
    xs = np.arange(episodes)[::step]
    ax2.plot(xs, eg_mean[::step], color="0.3", linewidth=1.5,
             label="Epsilon-greedy (target)")
    ax2.plot(xs, bz_mean[::step], color="0.6", linewidth=1.5, linestyle="--",
             label="Boltzmann (default)")
    ax2.plot(xs, cal_mean[::step], color="0.1", linewidth=1.0, linestyle=":",
             label="Boltzmann (calibrated)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Shannon Entropy (nats)")
    ax2.set_title("Entropy Verification")
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, episodes)
    ax2.set_ylim(bottom=0)

    fig.tight_layout()
    path = os.path.join(output_dir, "calibrated_schedule.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: Multiplier verification (for multi mode)
# ---------------------------------------------------------------------------

def _plot_multiplier_verification(M, output_dir, episodes=100_000, n_seeds=5):
    """Plot entropy comparison using the recommended constant multiplier M."""
    n_actions = 11
    n_bidders = 2
    max_entropy = np.log(n_actions)

    # Run epsilon-greedy
    eg_entropies = np.empty((n_seeds, episodes))
    for s in range(n_seeds):
        eg_entropies[s], _ = run_instrumented(
            "egreedy", episodes, seed=s * 1000,
            n_bidders=n_bidders, n_actions=n_actions,
        )
    eg_mean = eg_entropies.mean(axis=0)
    eg_std = eg_entropies.std(axis=0)

    # Run Boltzmann with default schedule
    bz_entropies = np.empty((n_seeds, episodes))
    for s in range(n_seeds):
        bz_entropies[s], _ = run_instrumented(
            "boltzmann", episodes, seed=s * 1000,
            n_bidders=n_bidders, n_actions=n_actions,
        )
    bz_mean = bz_entropies.mean(axis=0)
    bz_std = bz_entropies.std(axis=0)

    # Run Boltzmann with multiplier M
    cal_entropies = np.empty((n_seeds, episodes))
    for s in range(n_seeds):
        cal_entropies[s], _ = _run_with_multiplier(
            episodes, seed=s * 1000, multiplier=M, decay_type="linear",
            n_bidders=n_bidders, n_actions=n_actions,
        )
    cal_mean = cal_entropies.mean(axis=0)
    cal_std = cal_entropies.std(axis=0)

    _plot_entropy_comparison(
        eg_mean, eg_std, bz_mean, bz_std, cal_mean, cal_std,
        episodes, max_entropy, output_dir,
    )
    print(f"  Verification plot saved to {output_dir}/entropy_comparison.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exploration calibration: Boltzmann vs epsilon-greedy"
    )
    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "multi", "exp3"],
                        help="Calibration mode (default: single)")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of random seeds (default: 10)")
    parser.add_argument("--episodes", type=int, default=100_000,
                        help="Episodes per run (default: 100000)")
    parser.add_argument("--output-dir", type=str, default="results/calibration",
                        help="Output directory (default: results/calibration)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--auction-type", type=str, default="first",
                        choices=["first", "second"],
                        help="Auction type for single mode (default: first)")
    parser.add_argument("--n-bidders", type=int, default=2,
                        help="Number of bidders for single mode (default: 2)")
    parser.add_argument("--decay-type", type=str, default="linear",
                        choices=["linear", "exponential"],
                        help="Decay schedule for single mode (default: linear)")
    args = parser.parse_args()

    if args.mode == "multi":
        run_multi_setting_calibration(
            n_seeds=args.seeds,
            episodes=args.episodes,
            output_dir=args.output_dir,
            no_plots=args.no_plots,
        )
    elif args.mode == "exp3":
        measure_exp3_entropy(
            episodes=args.episodes,
            n_seeds=args.seeds,
            output_dir=args.output_dir,
        )
    else:
        run_calibration(
            n_seeds=args.seeds,
            episodes=args.episodes,
            output_dir=args.output_dir,
            no_plots=args.no_plots,
        )


if __name__ == "__main__":
    main()
