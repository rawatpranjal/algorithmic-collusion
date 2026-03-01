#!/usr/bin/env python3
"""
Budget Robustness Check for Experiment 4a.

Tests whether Exp4a factorial effects (auction_type x objective x n_bidders)
are robust to different budget levels. The default budget in Exp4a is
0.5 * E[v_i] * T; this script varies the 0.5 multiplier.

Usage:
    PYTHONPATH=src python3 scripts/budget_robustness.py
    PYTHONPATH=src python3 scripts/budget_robustness.py --budget-multipliers 0.1,0.25,0.5,1.0,2.0
    PYTHONPATH=src python3 scripts/budget_robustness.py --seed 123
"""

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.exp4a import run_experiment


# =====================================================================
# Factorial cells: 2^3 = 8 (auction_type x objective x n_bidders)
# =====================================================================
CELLS = list(itertools.product(
    ["first", "second"],          # auction_type
    ["value_max", "utility_max"], # objective
    [2, 4],                       # n_bidders
))

METRICS = [
    "mean_platform_revenue",
    "mean_effective_poa",
    "mean_bid_to_value",
]

METRIC_LABELS = {
    "mean_platform_revenue": "Platform Revenue",
    "mean_effective_poa": "Effective PoA",
    "mean_bid_to_value": "Bid-to-Value Ratio",
}

FACTOR_NAMES = ["auction_type", "objective", "n_bidders"]


# =====================================================================
# Coded values for factorial analysis
# =====================================================================
def code_factor(name, value):
    """Map raw factor values to coded -1/+1."""
    coding = {
        "auction_type": {"second": -1, "first": 1},
        "objective": {"value_max": -1, "utility_max": 1},
        "n_bidders": {2: -1, 4: 1},
    }
    return coding[name][value]


def compute_effect_rankings(df, metrics):
    """
    Compute main effect sizes for each metric using coded contrasts.

    Returns a dict: {metric: [(factor, effect_size), ...]} sorted by |effect|.
    """
    rankings = {}
    for metric in metrics:
        effects = []
        for factor in FACTOR_NAMES:
            coded_col = f"{factor}_coded"
            high = df[df[coded_col] == 1][metric].mean()
            low = df[df[coded_col] == -1][metric].mean()
            effect = high - low
            effects.append((factor, effect))
        # Sort by absolute effect size, descending
        effects.sort(key=lambda x: abs(x[1]), reverse=True)
        rankings[metric] = effects
    return rankings


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Budget robustness check for Experiment 4a"
    )
    parser.add_argument(
        "--budget-multipliers",
        type=str,
        default="0.25,0.5,1.0",
        help="Comma-separated budget multipliers (default: 0.25,0.5,1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    args = parser.parse_args()

    multipliers = [float(x.strip()) for x in args.budget_multipliers.split(",")]
    base_seed = args.seed
    n_seeds = 5
    n_episodes = 50
    T = 500

    output_dir = Path("results/exp4a/robust/budget")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Budget Robustness Check for Experiment 4a")
    print(f"  Multipliers: {multipliers}")
    print(f"  Cells: {len(CELLS)} (2^3 factorial)")
    print(f"  Seeds per cell: {n_seeds}")
    print(f"  Episodes: {n_episodes}, Rounds/episode: {T}")
    print(f"  Total runs: {len(multipliers) * len(CELLS) * n_seeds}")
    print(f"  Output: {output_dir}")
    print()

    # -----------------------------------------------------------------
    # Run all combinations
    # -----------------------------------------------------------------
    all_rows = []
    total_runs = len(multipliers) * len(CELLS) * n_seeds
    run_count = 0
    t_start = time.time()

    for mult in multipliers:
        print(f"--- Budget multiplier = {mult} ---")
        for cell_idx, (auction_type, objective, n_bidders) in enumerate(CELLS):
            for s in range(n_seeds):
                run_count += 1
                seed = (base_seed + s) * 10000 + cell_idx

                summary, _, _ = run_experiment(
                    auction_type=auction_type,
                    objective=objective,
                    n_bidders=n_bidders,
                    n_episodes=n_episodes,
                    T=T,
                    seed=seed,
                    budget_multiplier=mult,
                )

                row = {
                    "budget_multiplier": mult,
                    "auction_type": auction_type,
                    "objective": objective,
                    "n_bidders": n_bidders,
                    "auction_type_coded": code_factor("auction_type", auction_type),
                    "objective_coded": code_factor("objective", objective),
                    "n_bidders_coded": code_factor("n_bidders", n_bidders),
                    "seed": seed,
                }
                for metric in METRICS:
                    row[metric] = summary[metric]
                all_rows.append(row)

                if run_count % 10 == 0 or run_count == total_runs:
                    elapsed = time.time() - t_start
                    rate = run_count / elapsed if elapsed > 0 else 0
                    remaining = (total_runs - run_count) / rate if rate > 0 else 0
                    print(
                        f"  [{run_count}/{total_runs}] "
                        f"mult={mult}, {auction_type}/{objective}/n={n_bidders} "
                        f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)"
                    )

    elapsed_total = time.time() - t_start
    print(f"\nAll runs complete in {elapsed_total:.1f}s")

    # -----------------------------------------------------------------
    # Build DataFrame and save comparison.csv
    # -----------------------------------------------------------------
    df = pd.DataFrame(all_rows)

    # Aggregate: mean across seeds for each (multiplier, cell)
    agg_cols = ["budget_multiplier", "auction_type", "objective", "n_bidders",
                "auction_type_coded", "objective_coded", "n_bidders_coded"]
    df_agg = df.groupby(agg_cols, as_index=False)[METRICS].mean()

    # Also save a long-form comparison.csv
    long_rows = []
    for _, row in df_agg.iterrows():
        for metric in METRICS:
            long_rows.append({
                "budget_multiplier": row["budget_multiplier"],
                "auction_type": row["auction_type"],
                "objective": row["objective"],
                "n_bidders": int(row["n_bidders"]),
                "metric_name": metric,
                "metric_value": row[metric],
            })
    df_long = pd.DataFrame(long_rows)
    csv_path = output_dir / "comparison.csv"
    df_long.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    # -----------------------------------------------------------------
    # Compute effect rankings per budget level
    # -----------------------------------------------------------------
    rankings_by_mult = {}
    for mult in multipliers:
        subset = df_agg[df_agg["budget_multiplier"] == mult]
        rankings_by_mult[mult] = compute_effect_rankings(subset, METRICS)

    # -----------------------------------------------------------------
    # Write summary.txt
    # -----------------------------------------------------------------
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("Budget Robustness Check: Effect Rankings by Budget Level")
    summary_lines.append("=" * 70)
    summary_lines.append(f"Multipliers tested: {multipliers}")
    summary_lines.append(f"Seeds per cell: {n_seeds}")
    summary_lines.append(f"Design: {n_episodes} episodes x {T} rounds")
    summary_lines.append("")

    for metric in METRICS:
        summary_lines.append(f"--- {METRIC_LABELS[metric]} ({metric}) ---")
        summary_lines.append("")

        # Table header
        header = f"  {'Multiplier':<12}"
        for rank in range(len(FACTOR_NAMES)):
            header += f"  {'Rank ' + str(rank + 1):<30}"
        summary_lines.append(header)
        summary_lines.append("  " + "-" * (12 + 30 * len(FACTOR_NAMES) + 2 * len(FACTOR_NAMES)))

        rank_orders = []
        for mult in multipliers:
            effects = rankings_by_mult[mult][metric]
            rank_order = [f[0] for f in effects]
            rank_orders.append(rank_order)

            line = f"  {mult:<12.2f}"
            for factor, effect in effects:
                line += f"  {factor} ({effect:+.4f})"
                # Pad to 30 chars
                padding = 30 - len(f"{factor} ({effect:+.4f})")
                line += " " * max(0, padding)
            summary_lines.append(line)

        summary_lines.append("")

        # Check if rankings are stable
        if len(rank_orders) > 1:
            all_same = all(r == rank_orders[0] for r in rank_orders[1:])
            if all_same:
                summary_lines.append(
                    f"  STABLE: Effect ranking is identical across all budget levels."
                )
            else:
                summary_lines.append(
                    f"  CHANGED: Effect ranking varies across budget levels."
                )
                # Show which pairs differ
                for i in range(len(multipliers)):
                    for j in range(i + 1, len(multipliers)):
                        if rank_orders[i] != rank_orders[j]:
                            summary_lines.append(
                                f"    Differs between mult={multipliers[i]} "
                                f"and mult={multipliers[j]}: "
                                f"{rank_orders[i]} vs {rank_orders[j]}"
                            )
        summary_lines.append("")

    # Overall stability assessment
    summary_lines.append("=" * 70)
    summary_lines.append("Overall Assessment")
    summary_lines.append("=" * 70)

    n_stable = 0
    for metric in METRICS:
        rank_orders = []
        for mult in multipliers:
            effects = rankings_by_mult[mult][metric]
            rank_orders.append([f[0] for f in effects])
        if len(rank_orders) > 1 and all(r == rank_orders[0] for r in rank_orders[1:]):
            n_stable += 1

    summary_lines.append(
        f"{n_stable}/{len(METRICS)} metrics have stable effect rankings across "
        f"budget levels {multipliers}."
    )
    if n_stable == len(METRICS):
        summary_lines.append("Conclusion: Results are ROBUST to budget variation.")
    elif n_stable >= len(METRICS) // 2 + 1:
        summary_lines.append("Conclusion: Results are MOSTLY robust to budget variation.")
    else:
        summary_lines.append(
            "Conclusion: Results are SENSITIVE to budget variation. "
            "Interpret Exp4a findings with caution."
        )
    summary_lines.append("")

    summary_text = "\n".join(summary_lines)
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"Saved {summary_path}")
    print()
    print(summary_text)

    # -----------------------------------------------------------------
    # Generate budget_comparison.png
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 5))
    if len(METRICS) == 1:
        axes = [axes]

    for ax, metric in zip(axes, METRICS):
        # For each cell, plot metric vs budget multiplier
        for cell_idx, (auction_type, objective, n_bidders) in enumerate(CELLS):
            mask = (
                (df_agg["auction_type"] == auction_type)
                & (df_agg["objective"] == objective)
                & (df_agg["n_bidders"] == n_bidders)
            )
            subset = df_agg[mask].sort_values("budget_multiplier")

            label = f"{auction_type[0].upper()}PA/{objective[:3]}/n={n_bidders}"
            ax.plot(
                subset["budget_multiplier"],
                subset[metric],
                marker="o",
                markersize=4,
                linewidth=1.2,
                label=label,
            )

        ax.set_xlabel("Budget Multiplier")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(METRIC_LABELS[metric])
        ax.legend(fontsize=6, loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Exp4a Budget Robustness", fontsize=13, y=1.02)
    fig.tight_layout()
    fig_path = output_dir / "budget_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
