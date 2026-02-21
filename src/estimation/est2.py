#!/usr/bin/env python3
"""
Experiment 2 Analysis: Factorial ANOVA for affiliated values Q-learning.

Thin wrapper around factorial_analysis.run_factorial_analysis().
"""

import pandas as pd
from estimation.factorial_analysis import run_factorial_analysis

CODED_COLS = [
    "auction_type_coded",
    "eta_linear_coded",
    "eta_quadratic_coded",
    "n_bidders_coded",
    "state_info_coded",
]

RESPONSE_COLS = [
    "avg_rev_last_1000",
    "time_to_converge",
    "no_sale_rate",
    "price_volatility",
    "winner_entropy",
    "excess_regret",
    "efficient_regret",
    "btv_median",
    "winners_curse_freq",
    "bid_dispersion",
    "signal_slope_ratio",
]

if __name__ == "__main__":
    df = pd.read_csv("results/exp2/data.csv")

    if "time_to_converge" in df.columns and "episodes" in df.columns:
        df["time_to_converge"] = df["time_to_converge"] / df["episodes"]

    run_factorial_analysis(
        df,
        coded_cols=CODED_COLS,
        response_cols=RESPONSE_COLS,
        output_dir="results/exp2",
        experiment_id=2,
    )

    from estimation.robust_analysis import run_robust_analysis
    run_robust_analysis(df, coded_cols=CODED_COLS, response_cols=RESPONSE_COLS,
                        output_dir="results/exp2", experiment_id=2)
