#!/usr/bin/env python3

import numpy as np
import pandas as pd
import doubleml as dml
import matplotlib.pyplot as plt
import os
import sys
from lightgbm import LGBMRegressor, LGBMClassifier
import patsy
from scipy.stats import norm, pearsonr
from tabulate import tabulate
import math
# Globally set display settings (no truncation):
pd.set_option('display.max_rows', None)         # Show all rows
pd.set_option('display.max_columns', None)      # Show all columns
pd.set_option('display.width', 0)               # Auto-detect console width
pd.set_option('display.max_colwidth', None)     # No column width truncation

# ---------------------------------------------------------------------
# 0) Redirect stdout to a log file in new_experiment3
# ---------------------------------------------------------------------
exp_folder = "results/exp3"
os.makedirs(exp_folder, exist_ok=True)

analysis_log_path = os.path.join(exp_folder, "analysis_stdout.txt")
orig_stdout = sys.stdout
f = open(analysis_log_path, "w")
sys.stdout = f  # All prints go to the file

def main():
    # ---------------------------------------------------------------------
    # 1) Variable Definitions
    # ---------------------------------------------------------------------
    var_definitions = {
        "eta": "Affiliation parameter in [0,1].",
        "c": "Exploration parameter for LinUCB (bonus factor).",
        "lam": "Regularization parameter for LinUCB (lambda).",
        "auction_type_code": "Treatment: 1=first-price, 0=second-price.",
        "n_bidders": "Number of bidders (2,4,6).",
        "reserve_price": "Reserve price in [0..0.5].",
        "max_rounds": "Total number of bandit rounds (LinUCB).",
        "avg_rev_last_1000": "Rolling final avg revenue (Exp.1 style).",
        "time_to_converge": "Fraction of rounds until ±5% stable.",
        "avg_regret_seller": "Mean(1 - revenue) across rounds.",
        "no_sale_rate": "Fraction of episodes with no valid bid >= reserve.",
        "price_volatility": "Std dev of winning bid across episodes.",
        "winner_entropy": "Shannon entropy of winner distribution.",
        "use_median_of_others_code": "Median-of-others in context? (0=No,1=Yes).",
        "use_past_winner_bid_code": "Past-winner-bid in context? (0=No,1=Yes).",
        "theoretical_revenue": "Simulated BNE revenue for that config."
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

    # If "time_to_converge" and "max_rounds" exist, normalize
    if "time_to_converge" in df.columns and "max_rounds" in df.columns:
        df["time_to_converge"] = df["time_to_converge"] / df["max_rounds"]

    # ---------------------------------------------------------------------
    # 3) Define Treatment, Covariates, Outcomes
    # ---------------------------------------------------------------------
    treatment_col = "auction_type_code"
    covariates_list = [
        "eta", "c", "lam",
        "n_bidders", "reserve_price", "max_rounds",
        "use_median_of_others_code", "use_past_winner_bid_code"
    ]

    outcomes_list = [
        "avg_rev_last_1000",
        "time_to_converge",
        "avg_regret_seller",
        "no_sale_rate",
        "price_volatility",
        "winner_entropy"
    ]

    # ---------------------------------------------------------------------
    # 4) Summary Statistics
    # ---------------------------------------------------------------------
    cols_for_summary = covariates_list + [treatment_col] + outcomes_list
    summary_stats = df[cols_for_summary].describe().T
    print("\n=== Summary Statistics ===")
    print(tabulate(summary_stats, headers='keys', tablefmt='github'))

    # ---------------------------------------------------------------------
    # 5) Correlation vs. Treatment
    # ---------------------------------------------------------------------
    corr_results = []
    for col in covariates_list + outcomes_list:
        r_val, p_val = pearsonr(df[col], df[treatment_col])
        corr_results.append({
            "Variable": col,
            "Correlation": r_val,
            "p-value": p_val
        })

    print("\n=== Correlations with 'auction_type_code' ===")
    print(tabulate(corr_results, headers='keys', tablefmt='github'))

    # ---------------------------------------------------------------------
    # 6) DoubleML IRM for each outcome
    # ---------------------------------------------------------------------
    # Make subfolders for gate/cate plots
    gate_folder = os.path.join(exp_folder, "gate_plots")
    cate_folder = os.path.join(exp_folder, "cate_plots")
    os.makedirs(gate_folder, exist_ok=True)
    os.makedirs(cate_folder, exist_ok=True)

    ml_g = LGBMRegressor(random_state=123, verbose=-1)
    ml_m = LGBMClassifier(random_state=123, verbose=-1)

    ate_results = []

    for outcome in outcomes_list:
        print(f"\n\n=== OUTCOME: {outcome} ===")
        df["Y"] = df[outcome]

        # Create DoubleMLData
        dml_data = dml.DoubleMLData(
            df,
            y_col="Y",
            d_cols=treatment_col,
            x_cols=covariates_list
        )

        # IRM
        dml_irm = dml.DoubleMLIRM(
            dml_data,
            ml_g=ml_g,
            ml_m=ml_m,
            n_folds=5,
            score="ATE"
        )
        dml_irm.fit()

        print(f"\n[ATE for {outcome}]")
        print(dml_irm.summary)
        ate_dict = {
            "Outcome": outcome,
            "ATE": dml_irm.coef[0],
            "StdErr": dml_irm.se[0],
            "p-value": dml_irm.pval[0]
        }
        ate_results.append(ate_dict)

        # ----------------------------------------------------------
        # 6A) GATE for binary covariates
        # ----------------------------------------------------------
        print(f"\n[GATE: {outcome}] => checking binary covariates...")
        bin_covs = [c for c in covariates_list if df[c].nunique() == 2]
        gate_diff_tests = []

        if len(bin_covs) > 0:
            n_bin = len(bin_covs)
            nrows_gate = math.ceil(n_bin / 3)
            ncols_gate = min(n_bin, 3)
            fig_gate, axes_gate = plt.subplots(
                nrows=nrows_gate, ncols=ncols_gate,
                figsize=(5 * ncols_gate, 4 * nrows_gate)
            )
            if n_bin == 1:
                axes_gate = np.array([axes_gate]).flatten()

            for i, bin_col in enumerate(bin_covs):
                # GATE
                groups_df = df[[bin_col]].astype("category")
                gate_obj = dml_irm.gate(groups=groups_df)
                ci_95 = gate_obj.confint(level=0.95)
                eff = ci_95["effect"]
                lo = ci_95["2.5 %"]
                hi = ci_95["97.5 %"]
                gate_sum = gate_obj.summary
                errs = gate_sum["std err"]

                ax = axes_gate.flatten()[i] if n_bin > 1 else axes_gate[i]
                xvals = np.arange(len(eff))
                ax.errorbar(
                    xvals, eff,
                    yerr=[eff - lo, hi - eff],
                    fmt="o", capsize=5
                )
                ax.set_title(f"GATE: {bin_col} ({outcome})")
                ax.set_xticks(xvals)
                ax.set_xticklabels([f"{bin_col}={lvl}" for lvl in range(len(eff))])
                ax.set_ylabel("Estimated GATE")

                if len(eff) == 2:
                    dval = eff.iloc[1] - eff.iloc[0]
                    dvar = errs.iloc[1]**2 + errs.iloc[0]**2
                    dse = math.sqrt(dvar)
                    tval = dval / dse
                    pval = 2.0 * (1.0 - norm.cdf(abs(tval)))
                    gate_diff_tests.append({
                        "BinaryCov": bin_col,
                        "Group0_Effect": f"{eff.iloc[0]:.4f}",
                        "Group1_Effect": f"{eff.iloc[1]:.4f}",
                        "Diff": f"{dval:.4f}",
                        "StdErr(Diff)": f"{dse:.4f}",
                        "t": f"{tval:.4f}",
                        "p": f"{pval:.4f}"
                    })

            fig_gate.tight_layout()
            gate_path = os.path.join(gate_folder, f"gate_plots_{outcome}.png")
            fig_gate.savefig(gate_path, bbox_inches="tight")
            plt.close(fig_gate)
            print(f"  -> GATE plots saved to {gate_path}")

            if gate_diff_tests:
                print(f"\n[GATE T-tests for {outcome}]")
                print(tabulate(gate_diff_tests, headers="keys", tablefmt="github"))
        else:
            print("  -> No binary covariates to do GATE analysis.")

        # ----------------------------------------------------------
        # 6B) CATE for continuous covariates
        # ----------------------------------------------------------
        cont_covs = [c for c in covariates_list if df[c].nunique() > 2]
        if len(cont_covs) > 0:
            n_cont = len(cont_covs)
            nrows_cate = math.ceil(n_cont / 3)
            ncols_cate = min(n_cont, 3)
            fig_cate, axes_cate = plt.subplots(
                nrows=nrows_cate, ncols=ncols_cate,
                figsize=(5 * ncols_cate, 4 * nrows_cate)
            )
            if n_cont == 1:
                axes_cate = np.array([axes_cate]).flatten()

            for i, cc in enumerate(cont_covs):
                design_matrix = patsy.dmatrix(f"bs({cc}, df=5, degree=2)", df)
                spline_basis = pd.DataFrame(design_matrix)
                cate_obj = dml_irm.cate(basis=spline_basis)
                ci_95_cate = cate_obj.confint(basis=spline_basis, level=0.95)

                eff_cate = ci_95_cate["effect"].values
                lo_cate = ci_95_cate["2.5 %"].values
                hi_cate = ci_95_cate["97.5 %"].values

                xvals = df[cc].values
                idx_sort = np.argsort(xvals)
                x_sort = xvals[idx_sort]
                eff_sort = eff_cate[idx_sort]
                lo_sort = lo_cate[idx_sort]
                hi_sort = hi_cate[idx_sort]

                axc = axes_cate.flatten()[i] if n_cont > 1 else axes_cate[i]
                axc.plot(x_sort, eff_sort, label="CATE", color="blue")
                axc.fill_between(
                    x_sort, lo_sort, hi_sort,
                    color="blue", alpha=0.2,
                    label="95% CI"
                )
                axc.axhline(0, color="black", linestyle="--")
                axc.set_title(f"CATE: {cc} ({outcome})")
                axc.set_xlabel(cc)
                axc.set_ylabel("Estimated TE")
                axc.legend()

            fig_cate.tight_layout()
            cate_path = os.path.join(cate_folder, f"cate_plots_{outcome}.png")
            fig_cate.savefig(cate_path, bbox_inches="tight")
            plt.close(fig_cate)
            print(f"  -> CATE plots saved to {cate_path}")
        else:
            print("  -> No continuous covariates to do partial dependence CATE.")

    # Summarize ATE across outcomes
    print("\n=== Final ATE Results for All Outcomes ===")
    print(tabulate(ate_results, headers="keys", floatfmt=".4f", tablefmt="github"))

    print("\n=== Analysis Complete for Experiment 3 ===")

if __name__ == "__main__":
    main()

    # Restore stdout
    sys.stdout = orig_stdout
    f.close()
