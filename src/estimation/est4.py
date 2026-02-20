#!/usr/bin/env python3
"""
Experiment 4 Analysis: Factorial ANOVA for budget-constrained pacing.

Thin wrapper around factorial_analysis.run_factorial_analysis().
"""

import pandas as pd
from estimation.factorial_analysis import run_factorial_analysis

CODED_COLS = [
    "algorithm_coded",
    "auction_type_coded",
    "n_bidders_coded",
    "budget_tightness_coded",
    "eta_coded",
    "aggressiveness_coded",
    "update_frequency_coded",
    "initial_multiplier_coded",
    "reserve_price_coded",
]

RESPONSE_COLS = [
    "avg_rev_last_1000",
    "time_to_converge",
    "avg_regret_of_seller",
    "no_sale_rate",
    "price_volatility",
    "winner_entropy",
    "budget_utilization",
    "spend_volatility",
    "budget_violation_rate",
    "effective_bid_shading",
    "multiplier_convergence_time",
    "multiplier_final_mean",
    "multiplier_final_std",
]

if __name__ == "__main__":
    df = pd.read_csv("results/exp4/data.csv")

    if "time_to_converge" in df.columns and "max_rounds" in df.columns:
        df["time_to_converge"] = df["time_to_converge"] / df["max_rounds"]

    run_factorial_analysis(
        df,
        coded_cols=CODED_COLS,
        response_cols=RESPONSE_COLS,
        output_dir="results/exp4",
        experiment_id=4,
    )
