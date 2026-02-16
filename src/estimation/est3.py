#!/usr/bin/env python3
"""
Experiment 3 Analysis: Factorial ANOVA for LinUCB bandits.

Thin wrapper around factorial_analysis.run_factorial_analysis().
"""

import pandas as pd
from estimation.factorial_analysis import run_factorial_analysis

CODED_COLS = [
    "auction_type_coded",
    "eta_coded",
    "c_coded",
    "lam_coded",
    "n_bidders_coded",
    "reserve_price_coded",
    "use_median_of_others_coded",
    "use_past_winner_bid_coded",
]

RESPONSE_COLS = [
    "avg_rev_last_1000",
    "time_to_converge",
    "avg_regret_seller",
    "no_sale_rate",
    "price_volatility",
    "winner_entropy",
]

if __name__ == "__main__":
    df = pd.read_csv("results/exp3/data.csv")

    if "time_to_converge" in df.columns and "max_rounds" in df.columns:
        df["time_to_converge"] = df["time_to_converge"] / df["max_rounds"]

    run_factorial_analysis(
        df,
        coded_cols=CODED_COLS,
        response_cols=RESPONSE_COLS,
        output_dir="results/exp3",
        experiment_id=3,
    )
