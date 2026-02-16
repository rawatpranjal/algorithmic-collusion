#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd
import doubleml as dml
import matplotlib.pyplot as plt
import patsy
from doubleml import DoubleMLData, DoubleMLIRM
from lightgbm import LGBMRegressor, LGBMClassifier
from scipy.stats import norm, pearsonr
from tabulate import tabulate
import math
import json
# Globally set display settings (no truncation):
pd.set_option('display.max_rows', None)         # Show all rows
pd.set_option('display.max_columns', None)      # Show all columns
pd.set_option('display.width', 0)               # Auto-detect console width
pd.set_option('display.max_colwidth', None)     # No column width truncation

def main():
    # ---------------------------------------------------------------------
    # 0) Redirect stdout to a log file (like in Exp.1)
    # ---------------------------------------------------------------------
    exp_folder = "results/exp2"  # Path to your Exp.2 folder
    os.makedirs(exp_folder, exist_ok=True)
    analysis_log_path = os.path.join(exp_folder, "analysis_stdout.txt")
    orig_stdout = sys.stdout

    with open(analysis_log_path, "w") as f:
        sys.stdout = f  # Redirect prints to the log file

        # ---------------------------------------------------------------------
        # 1) Variable Definitions (Experiment 2)
        # ---------------------------------------------------------------------
        var_definitions = {
            "eta": "Affiliation parameter in [0..1].",
            "alpha": "Learning rate for Q-updates.",
            "gamma": "Discount factor for future rewards (0.0..0.99).",
            "episodes": "Total number of training episodes.",
            "reserve_price": "Reserve price in [0..0.5].",
            "auction_type_code": "Treatment: 1=first-price, 0=second-price.",
            "init_code": "Initialization: 0='random', 1='zeros'.",
            "exploration_code": "Exploration: 0='egreedy', 1='boltzmann'.",
            "asynchronous_code": "Update mode: 0=synchronous, 1=asynchronous.",
            "n_bidders": "Number of bidding agents (2,4,6).",
            "median_opp_past_bid_index_code": "Median-of-opponents in state? 1=Yes, 0=No.",
            "winner_bid_index_state_code": "Track winner-bid index in state? 1=Yes, 0=No.",
            "avg_rev_last_1000": "Mean seller revenue in final 1000 episodes.",
            "time_to_converge": "Fraction of episodes until ±5% convergence.",
            "avg_regret_of_seller": "Average regret for the seller (1 - revenue).",
            "no_sale_rate": "Fraction of episodes with no valid bid.",
            "price_volatility": "Std dev of winning bid across episodes.",
            "winner_entropy": "Shannon entropy of the winner distribution."
        }
        df_var_defs = pd.DataFrame(
            [{"Parameter": k, "Definition": v} for k, v in var_definitions.items()]
        )

        print("\n=== Variable Definitions ===")
        print(tabulate(df_var_defs, headers="keys", tablefmt="github"))

        # ---------------------------------------------------------------------
        # 2) Read in the experiment data
        # ---------------------------------------------------------------------
        data_path = os.path.join(exp_folder, "data.csv")
        df = pd.read_csv(data_path)

        # If time_to_converge is present, normalize by episodes
        if "time_to_converge" in df.columns:
            df["time_to_converge"] = df["time_to_converge"] / df["episodes"]

        # ---------------------------------------------------------------------
        # 3) Define treatment, covariates, outcomes
        # ---------------------------------------------------------------------
        treatment_col = "auction_type_code"
        covariates_list = [
            "eta", "alpha", "gamma", "episodes",
            "reserve_price", "init_code", "exploration_code",
            "asynchronous_code", "n_bidders",
            "median_opp_past_bid_index_code", "winner_bid_index_state_code"
        ]

        # Including the additional outcomes from Exp.1 style:
        outcomes_list = [
            "avg_rev_last_1000",
            "time_to_converge",
            "avg_regret_of_seller",
            "no_sale_rate",
            "price_volatility",
            "winner_entropy"
        ]

        # ---------------------------------------------------------------------
        # 4) Summary Statistics
        # ---------------------------------------------------------------------
        cols_for_summary = covariates_list + [treatment_col] + outcomes_list
        df_summary = df[cols_for_summary].describe().T
        print("\n=== Summary Statistics ===")
        print(tabulate(df_summary, headers='keys', tablefmt='github'))

        # ---------------------------------------------------------------------
        # 5) Correlation with Treatment
        # ---------------------------------------------------------------------
        corr_rows = []
        for col in covariates_list + outcomes_list:
            r, p = pearsonr(df[col], df[treatment_col])
            corr_rows.append({'Variable': col, 'Correlation': r, 'p-value': p})

        print("\n=== Correlations with 'auction_type_code' ===")
        print(tabulate(corr_rows, headers='keys', tablefmt='github'))

        # ---------------------------------------------------------------------
        # 6) Create Folders for GATE & CATE plots
        # ---------------------------------------------------------------------
        gate_folder = os.path.join(exp_folder, "gate_plots")
        cate_folder = os.path.join(exp_folder, "cate_plots")
        os.makedirs(gate_folder, exist_ok=True)
        os.makedirs(cate_folder, exist_ok=True)

        # ---------------------------------------------------------------------
        # 7) DoubleML IRM for each outcome: ATE, GATE, BLP CATE, partial-dep
        # ---------------------------------------------------------------------
        ml_g = LGBMRegressor(random_state=123, verbose=-1)
        ml_m = LGBMClassifier(random_state=123, verbose=-1)

        ate_results = []
        all_gate_results = {}

        for outcome in outcomes_list:
            print(f"\n\n=== OUTCOME: {outcome} ===")

            # 7A) ATE via IRM
            df["Y"] = df[outcome]
            dml_data = dml.DoubleMLData(
                df,
                y_col="Y",
                d_cols=treatment_col,
                x_cols=covariates_list
            )

            dml_irm = dml.DoubleMLIRM(
                dml_data,
                ml_g=ml_g,
                ml_m=ml_m,
                n_folds=5,
                score="ATE"
            )
            dml_irm.fit()

            ate_res = {
                "Outcome": outcome,
                "ATE": dml_irm.coef[0],
                "StdErr": dml_irm.se[0],
                "p-value": dml_irm.pval[0]
            }
            ate_results.append(ate_res)

            print("\n[ATE for outcome]")
            print(tabulate([ate_res], headers='keys', floatfmt='.4f', tablefmt='github'))

            # 7B) GATE: for all binary covariates
            print(f"\n[GATE: {outcome}] => plotting for all binary covariates...")
            bin_covs = [c for c in covariates_list if df[c].nunique() == 2]
            n_bin = len(bin_covs)
            gate_diff_tests = []

            if n_bin > 0:
                nrows_gate = math.ceil(n_bin / 3)
                ncols_gate = min(n_bin, 3)
                fig_gate, axes_gate = plt.subplots(nrows=nrows_gate, ncols=ncols_gate,
                                                   figsize=(5*ncols_gate, 4*nrows_gate))
                if n_bin == 1:
                    axes_gate = np.array([axes_gate]).flatten()

                for i, bin_col in enumerate(bin_covs):
                    groups_df = df[[bin_col]].astype("category")
                    gate_obj = dml_irm.gate(groups=groups_df)
                    ci = gate_obj.confint(level=0.95)
                    eff = ci["effect"]
                    lo = ci["2.5 %"]
                    hi = ci["97.5 %"]

                    ax = axes_gate.flatten()[i] if n_bin > 1 else axes_gate[i]
                    xvals = np.arange(len(eff))
                    ax.errorbar(xvals, eff, yerr=[eff - lo, hi - eff],
                                fmt="o", capsize=5)
                    ax.set_title(f"GATE: {bin_col} ({outcome})")
                    ax.set_xticks(xvals)
                    ax.set_xticklabels([f"{bin_col}={lvl}" for lvl in range(len(eff))])
                    ax.set_ylabel("Estimated GATE")

                    # If exactly 2 levels => difference test
                    if len(eff) == 2:
                        eff0, eff1 = eff.iloc[0], eff.iloc[1]
                        se0, se1 = gate_obj.summary["std err"]
                        diff = eff1 - eff0
                        se_diff = math.sqrt(se0**2 + se1**2)
                        tval = diff / se_diff
                        pval = 2.0*(1.0 - norm.cdf(abs(tval)))
                        gate_diff_tests.append({
                            "Outcome": outcome,
                            "BinaryCov": bin_col,
                            "Group0": eff0,
                            "Group1": eff1,
                            "Diff": diff,
                            "t": tval,
                            "p": pval
                        })

                fig_gate.tight_layout()
                gate_plot_path = os.path.join(gate_folder, f"gate_plots_{outcome}.png")
                fig_gate.savefig(gate_plot_path, bbox_inches="tight")
                plt.close(fig_gate)
                print(f"  -> GATE plots saved to {gate_plot_path}")
            else:
                print("  -> No binary covariates found, skipping GATE plots.")

            if gate_diff_tests:
                print(f"\n[GATE T-tests for outcome={outcome}]")
                print(tabulate(gate_diff_tests, headers='keys', floatfmt='.4f', tablefmt='github'))
                all_gate_results[outcome] = gate_diff_tests

            # 7C) BLP CATE (all covariates + squares)
            print(f"\n[BLP CATE for {outcome}] => Printing the regression summary.")
            big_basis = df[covariates_list].copy()
            for col in covariates_list:
                if big_basis[col].nunique() > 2:
                    big_basis[col + "_sq"] = big_basis[col]**2

            cate_obj_blp = dml_irm.cate(basis=big_basis)
            print("\n=== BLP Summary ===")
            print(cate_obj_blp.summary)

            # 7D) Partial-dep CATE: numeric covariates
            cont_covs = [c for c in covariates_list if df[c].nunique() > 2]
            print(f"\n[Partial-dep CATE: {outcome}] => plotting each numeric covariate in a grid...")
            if len(cont_covs) > 0:
                n_cont = len(cont_covs)
                nrows_cate = math.ceil(n_cont / 3)
                ncols_cate = min(n_cont, 3)
                fig_cate, axes_cate = plt.subplots(nrows=nrows_cate, ncols=ncols_cate,
                                                   figsize=(5*ncols_cate, 4*nrows_cate))
                if n_cont == 1:
                    axes_cate = np.array([axes_cate]).flatten()

                for i, cvar in enumerate(cont_covs):
                    design_matrix = patsy.dmatrix(f"bs({cvar}, df=5, degree=2)", df)
                    spline_basis = pd.DataFrame(design_matrix)
                    cate_obj_spline = dml_irm.cate(basis=spline_basis)
                    ci_95_cate = cate_obj_spline.confint(basis=spline_basis, level=0.95)
                    eff_cate = ci_95_cate["effect"].values
                    lo_cate = ci_95_cate["2.5 %"].values
                    hi_cate = ci_95_cate["97.5 %"].values

                    xvals = df[cvar].values
                    idx_sort = np.argsort(xvals)
                    x_sort = xvals[idx_sort]
                    eff_sort = eff_cate[idx_sort]
                    lo_sort = lo_cate[idx_sort]
                    hi_sort = hi_cate[idx_sort]

                    ax = axes_cate.flatten()[i] if n_cont > 1 else axes_cate[i]
                    ax.plot(x_sort, eff_sort, label="CATE", color="blue")
                    ax.fill_between(x_sort, lo_sort, hi_sort,
                                    alpha=0.2, label="95% CI", color="blue")
                    ax.axhline(0, color="black", linestyle="--")
                    ax.set_title(f"{cvar} ({outcome})")
                    ax.set_xlabel(cvar)
                    ax.set_ylabel("Estimated TE")
                    ax.legend()

                    # Save individual CATE figure
                    fig_single, ax_single = plt.subplots(figsize=(6, 4))
                    ax_single.plot(x_sort, eff_sort, label="CATE", color="blue")
                    ax_single.fill_between(x_sort, lo_sort, hi_sort,
                                           alpha=0.2, label="95% CI", color="blue")
                    ax_single.axhline(0, color="black", linestyle="--")
                    ax_single.set_title(f"{cvar} ({outcome})")
                    ax_single.set_xlabel(cvar)
                    ax_single.set_ylabel("Estimated TE")
                    ax_single.legend()
                    fig_single.tight_layout()
                    fig_single.savefig(os.path.join(cate_folder, f"cate_{outcome}_{cvar}.png"),
                                       bbox_inches="tight", dpi=150)
                    plt.close(fig_single)

                fig_cate.tight_layout()
                cate_plot_path = os.path.join(cate_folder, f"cate_plots_{outcome}.png")
                fig_cate.savefig(cate_plot_path, bbox_inches="tight")
                plt.close(fig_cate)
                print(f"  -> Partial-dep CATE plots saved to {cate_plot_path}")
            else:
                print("  -> No numeric covariates to do partial dependence, skipping.")

        # ---------------------------------------------------------------------
        # 8) Summarize ATE across all outcomes
        # ---------------------------------------------------------------------
        print("\n=== Final ATE Estimates (DoubleMLIRM) for All Outcomes ===")
        print(tabulate(ate_results, headers='keys', floatfmt='.4f', tablefmt='github'))

        print("\n=== Analysis Complete ===")

        # Save structured results as JSON
        structured = {
            "experiment": 2,
            "ate_results": ate_results,
            "gate_results": all_gate_results,
        }
        with open(os.path.join(exp_folder, "estimation_results.json"), "w") as jf:
            json.dump(structured, jf, indent=2, default=str)
        print("Structured results saved to results/exp2/estimation_results.json")

        # Restore original stdout
        sys.stdout = orig_stdout

    # Once outside the with-block, printing goes to console again
    print(f"Analysis complete. All logs saved to '{analysis_log_path}'.")

if __name__ == "__main__":
    main()
