#!/usr/bin/env python3
"""
Calibration script for Experiment 4b aggressiveness levels.

Sweeps aggressiveness over candidate values to find two levels that produce
meaningfully different behavior without degeneracy (budget exhaustion or
near-zero spending).

Usage:
    PYTHONPATH=src python3 scripts/calibrate_exp4b.py
    PYTHONPATH=src python3 scripts/calibrate_exp4b.py --seeds 5
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from experiments.exp4b import run_experiment


AGGRESSIVENESS_LEVELS = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]

DEFAULT_ENV = {
    "auction_type": "first",
    "n_bidders": 2,
    "budget_multiplier": 0.5,
    "reserve_price": 0.0,
    "sigma": 0.3,
    "n_episodes": 50,
    "T": 500,
}


def run_calibration(seeds=3, output_dir="results/calibration/exp4b"):
    os.makedirs(output_dir, exist_ok=True)

    results = {level: [] for level in AGGRESSIVENESS_LEVELS}

    total = len(AGGRESSIVENESS_LEVELS) * seeds
    done = 0

    for level in AGGRESSIVENESS_LEVELS:
        for s in range(seeds):
            done += 1
            print(f"  [{done}/{total}] aggressiveness={level}, seed={s}")
            summary, _, _ = run_experiment(
                aggressiveness=level,
                seed=42 + s,
                **DEFAULT_ENV,
            )
            results[level].append(summary)

    # Aggregate
    metrics = ["mean_budget_utilization", "mean_bid_to_value", "mean_platform_revenue",
               "mean_effective_poa", "bid_suppression_ratio", "mean_dual_cv"]

    print("\n" + "=" * 80)
    print(f"{'Aggr':>6s}", end="")
    for m in metrics:
        short = m.replace("mean_", "").replace("_", " ")[:15]
        print(f"  {short:>15s}", end="")
    print()
    print("-" * 80)

    agg_data = {}
    for level in AGGRESSIVENESS_LEVELS:
        row = {}
        for m in metrics:
            vals = [r[m] for r in results[level]]
            row[m] = np.mean(vals)
        agg_data[level] = row
        print(f"{level:6.1f}", end="")
        for m in metrics:
            print(f"  {row[m]:15.4f}", end="")
        print()

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for i, m in enumerate(metrics):
        ax = axes[i]
        x = AGGRESSIVENESS_LEVELS
        y = [agg_data[lev][m] for lev in x]
        ax.plot(x, y, "ko-", markersize=5, linewidth=1)
        short = m.replace("mean_", "").replace("_", " ").title()
        ax.set_title(short)
        ax.set_xlabel("Aggressiveness")
        ax.set_ylabel(short)

    fig.tight_layout()
    path = os.path.join(output_dir, "calibration_sweep.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {path}")

    # Recommendation
    print("\nRecommendation:")
    print("  Select two levels with clear separation in budget_utilization and btv.")
    print("  Avoid levels where budget_utilization > 0.95 (budget exhaustion)")
    print("  or budget_utilization < 0.1 (underspending).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate exp4b aggressiveness levels")
    parser.add_argument("--seeds", type=int, default=3, help="Seeds per level (default: 3)")
    parser.add_argument("--output-dir", type=str, default="results/calibration/exp4b")
    args = parser.parse_args()

    print("Experiment 4b Aggressiveness Calibration")
    print("=" * 40)
    run_calibration(seeds=args.seeds, output_dir=args.output_dir)
