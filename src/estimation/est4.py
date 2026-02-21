#!/usr/bin/env python3
"""
Experiment 4 Analysis: Factorial ANOVA + Panel Regression.

Two-stage analysis:
  Stage 1: Factorial ANOVA on run-level aggregates (uses shared engine)
  Stage 2: Panel regression on episode-level data with seed fixed effects
"""

import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from estimation.factorial_analysis import run_factorial_analysis

# Stage 1: Factorial ANOVA coded columns and responses
CODED_COLS = [
    "auction_type_coded",
    "objective_coded",
    "n_bidders_coded",
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
]

# Stage 2: Panel regression responses
PANEL_RESPONSES = [
    "platform_revenue",
    "liquid_welfare",
    "effective_poa",
    "budget_utilization",
    "bid_to_value",
    "allocative_efficiency",
]


def run_panel_regression(df_episodes, output_dir):
    """
    Panel regression with seed fixed effects.

    Y_{scd} = alpha + beta1*FPA + beta2*UtilityMax + beta3*N4
              + all 2-way interactions + gamma_s + epsilon

    Uses OLS with seed dummies, SEs clustered at seed level.
    Saves panel_results.json.
    """
    results = {}

    # Create treatment indicators
    df = df_episodes.copy()
    df["FPA"] = (df["auction_type_coded"] == 1).astype(float)
    df["UtilityMax"] = (df["objective_coded"] == 1).astype(float)
    df["N4"] = (df["n_bidders_coded"] == 1).astype(float)

    # Interactions
    df["FPA_x_UtilityMax"] = df["FPA"] * df["UtilityMax"]
    df["FPA_x_N4"] = df["FPA"] * df["N4"]
    df["UtilityMax_x_N4"] = df["UtilityMax"] * df["N4"]
    df["FPA_x_UtilityMax_x_N4"] = df["FPA"] * df["UtilityMax"] * df["N4"]

    treatment_vars = [
        "FPA", "UtilityMax", "N4",
        "FPA_x_UtilityMax", "FPA_x_N4", "UtilityMax_x_N4",
        "FPA_x_UtilityMax_x_N4",
    ]

    # Seed fixed effects via dummies
    seed_dummies = pd.get_dummies(df["seed"], prefix="seed", drop_first=True, dtype=float)

    for response in PANEL_RESPONSES:
        if response not in df.columns:
            continue

        y = df[response].values
        X = pd.concat([df[treatment_vars], seed_dummies], axis=1)
        X = sm.add_constant(X)

        try:
            model = sm.OLS(y, X).fit(
                cov_type="cluster",
                cov_kwds={"groups": df["seed"].values},
            )

            coef_dict = {}
            for var in ["const"] + treatment_vars:
                if var in model.params.index:
                    coef_dict[var] = {
                        "estimate": float(model.params[var]),
                        "std_err": float(model.bse[var]),
                        "t_value": float(model.tvalues[var]),
                        "p_value": float(model.pvalues[var]),
                    }

            results[response] = {
                "coefficients": coef_dict,
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "n_obs": int(model.nobs),
                "n_seeds": int(df["seed"].nunique()),
            }
        except Exception as e:
            print(f"  Panel regression failed for {response}: {e}")
            results[response] = {"error": str(e)}

    # Save results
    panel_path = os.path.join(output_dir, "panel_results.json")
    with open(panel_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Panel results => '{panel_path}'")

    return results


if __name__ == "__main__":
    output_dir = "results/exp4"

    # Stage 1: Factorial ANOVA on run-level data
    print("=" * 60)
    print("Stage 1: Factorial ANOVA (run-level aggregates)")
    print("=" * 60)

    df = pd.read_csv(os.path.join(output_dir, "data.csv"))
    print(f"  Loaded {len(df)} runs")

    run_factorial_analysis(
        df,
        coded_cols=CODED_COLS,
        response_cols=[r for r in RESPONSE_COLS if r in df.columns],
        output_dir=output_dir,
        experiment_id=4,
    )

    # Robustness analysis on run-level data
    from estimation.robust_analysis import run_robust_analysis
    run_robust_analysis(df, coded_cols=CODED_COLS,
                        response_cols=[r for r in RESPONSE_COLS if r in df.columns],
                        output_dir=output_dir, experiment_id=4)

    # Stage 2: Panel regression on episode-level data
    ep_path = os.path.join(output_dir, "data_episodes.csv")
    if os.path.exists(ep_path):
        print()
        print("=" * 60)
        print("Stage 2: Panel Regression (episode-level data)")
        print("=" * 60)

        df_ep = pd.read_csv(ep_path)
        print(f"  Loaded {len(df_ep)} episodes")

        run_panel_regression(df_ep, output_dir)
    else:
        print(f"\n  Skipping panel regression: {ep_path} not found")
