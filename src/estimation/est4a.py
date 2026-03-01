#!/usr/bin/env python3
"""
Experiment 4a Analysis: Factorial ANOVA.
"""

import os

import pandas as pd
from estimation.factorial_analysis import run_factorial_analysis

# Factorial ANOVA coded columns and responses
CODED_COLS = [
    "auction_type_coded",
    "objective_coded",
    "n_bidders_coded",
    "budget_multiplier_coded",
    "reserve_price_coded",
    "sigma_coded",
]

RESPONSE_COLS = [
    "mean_platform_revenue",
    "mean_liquid_welfare",
    "mean_effective_poa",
    "mean_budget_utilization",
    "mean_bid_to_value",
    "mean_allocative_efficiency",
    "mean_dual_cv",
    "mean_no_sale_rate",
    "mean_winner_entropy",
    "warm_start_benefit",
    "inter_episode_volatility",
    "bid_suppression_ratio",
    "cross_episode_drift",
    "mean_lp_offline_welfare",
    "mean_effective_poa_lp",
    "mean_rev_all",
]

if __name__ == "__main__":
    output_dir = "results/exp4a"

    print("=" * 60)
    print("Factorial ANOVA (run-level aggregates)")
    print("=" * 60)

    df = pd.read_csv(os.path.join(output_dir, "data.csv"))
    print(f"  Loaded {len(df)} runs")

    run_factorial_analysis(
        df,
        coded_cols=CODED_COLS,
        response_cols=[r for r in RESPONSE_COLS if r in df.columns],
        output_dir=output_dir,
        experiment_id="4a",
    )

    # Robustness analysis on run-level data
    from estimation.robust_analysis import run_robust_analysis
    run_robust_analysis(df, coded_cols=CODED_COLS,
                        response_cols=[r for r in RESPONSE_COLS if r in df.columns],
                        output_dir=output_dir, experiment_id="4a")

