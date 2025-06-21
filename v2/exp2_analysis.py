#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# DoubleML
import doubleml as dml
from doubleml import DoubleMLData, DoubleMLIRM

# LightGBM
from lightgbm import LGBMRegressor, LGBMClassifier

# For spline expansions (CATE)
import patsy


def main():
    # ------------------------------------------------------------------------------
    # 1) Setup: Create output directory
    # ------------------------------------------------------------------------------
    os.makedirs("experiment2", exist_ok=True)

    # ------------------------------------------------------------------------------
    # 2) Load Experiment 2 Data
    # ------------------------------------------------------------------------------
    df = pd.read_csv("experiment2/data.csv")

    # ------------------------------------------------------------------------------
    # 3) Treatment and Covariates
    # ------------------------------------------------------------------------------
    # Define treatment
    df["D"] = (df["auction_type"] == "first").astype(int)

    # Continuous covariates
    cont_cols = ["eta", "alpha", "gamma", "episodes"]

    # Binary covariates
    df["init"] = (df["init"] == "random").astype(int)
    df["exploration"] = (df["exploration"] == "egreedy").astype(int)
    df["median_opp_past_bid_index"] = df["median_opp_past_bid_index"].astype(int)
    df["winner_bid_index_state"] = df["winner_bid_index_state"].astype(int)
    df["asynchronous"] = df["asynchronous"].astype(int)

    binary_cols = ["init", "exploration", "median_opp_past_bid_index", "winner_bid_index_state", "asynchronous"]

    # All covariates
    X_cols = cont_cols + binary_cols

    # Define outcomes
    outcomes = ["avg_rev_last_1000", "time_to_converge", "avg_regret_of_seller"]

    # ------------------------------------------------------------------------------
    # 4) Loop Over Outcomes
    # ------------------------------------------------------------------------------
    for outcome in outcomes:
        print(f"\n========== Inference for Outcome: {outcome} ==========")

        # Define current outcome
        df["Y"] = df[outcome]

        # ------------------------------------------------------------------------------
        # 5) Prepare DoubleML Data and Model
        # ------------------------------------------------------------------------------
        dml_data = DoubleMLData(
            df,
            y_col="Y",
            d_cols="D",
            x_cols=X_cols
        )

        # Define learners
        ml_g = LGBMRegressor(verbose=-1, random_state=123)
        ml_m = LGBMClassifier(verbose=-1, random_state=123)

        # Initialize DoubleML IRM Model
        dml_irm = DoubleMLIRM(
            dml_data,
            ml_g=ml_g,
            ml_m=ml_m,
            n_folds=3,
            score="ATE"
        )

        # Fit the model
        dml_irm.fit()

        # ------------------------------------------------------------------------------
        # 6) Print and Save ATE Results
        # ------------------------------------------------------------------------------
        print(f"========== ATE Results for {outcome} ==========")
        print(dml_irm.summary)

        # ------------------------------------------------------------------------------
        # 7) GATEs for Binary Covariates
        # ------------------------------------------------------------------------------
        print(f"Generating GATE plots for {outcome}...")
        n_bin = len(binary_cols)
        nrows_gate = int(np.ceil(n_bin / 3))  # up to 3 columns
        ncols_gate = min(n_bin, 3)

        fig_gate, axes_gate = plt.subplots(nrows=nrows_gate, ncols=ncols_gate,
                                           figsize=(5 * ncols_gate, 4 * nrows_gate))

        if n_bin == 1:
            axes_gate = np.array([axes_gate])

        for i, bin_col in enumerate(binary_cols):
            groups_df = df[[bin_col]].astype("category")
            gate_obj = dml_irm.gate(groups=groups_df)
            ci_95_gate = gate_obj.confint(level=0.95)

            effects = ci_95_gate["effect"]
            lower_95 = ci_95_gate["2.5 %"]
            upper_95 = ci_95_gate["97.5 %"]

            ax = axes_gate.flatten()[i] if n_bin > 1 else axes_gate[0]
            x_positions = [0, 1]
            ax.errorbar(
                x_positions, effects,
                yerr=[effects - lower_95, upper_95 - effects],
                fmt="o", capsize=5
            )
            ax.set_title(f"GATE: {bin_col}")
            ax.set_xticks(x_positions)
            ax.set_xticklabels([f"{bin_col}=0", f"{bin_col}=1"])
            ax.set_ylabel("Estimated GATE")

        fig_gate.tight_layout()
        gate_plot_path = os.path.join("experiment2", f"gate_plots_{outcome}.png")
        fig_gate.savefig(gate_plot_path, bbox_inches="tight")
        plt.close(fig_gate)
        print(f"GATE plots for {outcome} saved to: {gate_plot_path}")

        # ------------------------------------------------------------------------------
        # 8) CATEs for Continuous Covariates
        # ------------------------------------------------------------------------------
        print(f"Generating CATE plots for {outcome}...")
        n_cont = len(cont_cols)
        nrows_cate = int(np.ceil(n_cont / 3))
        ncols_cate = min(n_cont, 3)

        fig_cate, axes_cate = plt.subplots(nrows=nrows_cate, ncols=ncols_cate,
                                           figsize=(5 * ncols_cate, 4 * nrows_cate))

        if n_cont == 1:
            axes_cate = np.array([axes_cate])

        for i, cont_col in enumerate(cont_cols):
            design_matrix = patsy.dmatrix(f"bs({cont_col}, df=5, degree=2)", df)
            spline_basis = pd.DataFrame(design_matrix)

            cate_obj = dml_irm.cate(basis=spline_basis)
            ci_95_cate = cate_obj.confint(basis=spline_basis, level=0.95)

            effects_cate = ci_95_cate["effect"].values
            lower_95_cate = ci_95_cate["2.5 %"].values
            upper_95_cate = ci_95_cate["97.5 %"].values

            x_values = df[cont_col].values
            idx_sort = np.argsort(x_values)

            x_sorted = x_values[idx_sort]
            eff_sorted = effects_cate[idx_sort]
            low_sorted = lower_95_cate[idx_sort]
            up_sorted = upper_95_cate[idx_sort]

            ax_cate = axes_cate.flatten()[i] if n_cont > 1 else axes_cate[0]
            ax_cate.plot(x_sorted, eff_sorted, label="CATE")
            ax_cate.fill_between(x_sorted, low_sorted, up_sorted,
                                 alpha=0.2, label="95% CI")
            ax_cate.set_title(f"CATE: {cont_col}")
            ax_cate.set_xlabel(cont_col)
            ax_cate.set_ylabel("Estimated Treatment Effect")
            ax_cate.legend()

        fig_cate.tight_layout()
        cate_plot_path = os.path.join("experiment2", f"cate_plots_{outcome}.png")
        fig_cate.savefig(cate_plot_path, bbox_inches="tight")
        plt.close(fig_cate)
        print(f"CATE plots for {outcome} saved to: {cate_plot_path}")

    print("\nInference complete. All results saved in 'experiment2/' folder.")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3

import numpy as np
import pandas as pd
import doubleml
import matplotlib.pyplot as plt
import os
import doubleml as dml
from doubleml import DoubleMLData, DoubleMLIRM
from lightgbm import LGBMRegressor, LGBMClassifier
import patsy
from scipy.stats import norm, pearsonr
from tabulate import tabulate
import math

if __name__ == "__main__":
    # ------------------------------------------------------------------------------
    # 1) Variable definitions and reading data
    # ------------------------------------------------------------------------------
    var_definitions = {
        "eta": "Affiliation parameter (0.0 to 1.0).",
        "alpha": "Learning rate for Q-updates (0.01 to 0.1).",
        "gamma": "Discount factor for future rewards (0.0 to 0.99).",
        "episodes": "Total number of training episodes (10k to 100k).",
        "auction_type": "Treatment: 0 = first-price, 1 = second-price.",
        "init": "Q-initialization: 'random' or 'zeros'.",
        "exploration": "Exploration strategy: 'egreedy' or 'boltzmann'.",
        "asynchronous": "Update mode: 0 = synchronous, 1 = asynchronous.",
        "n_bidders": "Number of bidders (2, 4, or 6).",
        "median_opp_past_bid_index": "Use median of opponents' bids in state?",
        "winner_bid_index_state": "Track winning bid index in state?",
        "avg_rev_last_1000": "Mean seller revenue in the final 1000 episodes.",
        "time_to_converge": "Fraction of episodes until ±5% convergence.",
        "avg_regret_of_seller": "Average regret for the seller (max valuation = 1)."
    }

    df_var_defs = pd.DataFrame([{"Parameter": k, "Definition": v} for k, v in var_definitions.items()])
    print("\n=== Variable Definitions ===")
    print(tabulate(df_var_defs, headers="keys", tablefmt="github"))

    # Load data
    df = pd.read_csv("experiment2/data.csv")

    # Recode auction_type: 'second' -> 1, 'first' -> 0
    df["auction_type"] = (df["auction_type"] == "second").astype(int)

    # Convert time_to_converge to fraction of total episodes
    df["time_to_converge"] = df["time_to_converge"] / df["episodes"]

    # ------------------------------------------------------------------------------
    # 2) Set up columns
    # ------------------------------------------------------------------------------
    treatment_col = "auction_type"
    covariates_list = [
        "eta", "alpha", "gamma", "episodes", "init",
        "exploration", "asynchronous", "n_bidders",
        "median_opp_past_bid_index", "winner_bid_index_state"
    ]
    outcomes_list = ["avg_rev_last_1000", "time_to_converge", "avg_regret_of_seller"]

    # Ensure binary covariates are int
    for b in ["init", "exploration", "asynchronous", "median_opp_past_bid_index", "winner_bid_index_state"]:
        if df[b].dtype == bool:
            df[b] = df[b].astype(int)
        if df[b].dtype not in [np.float64, np.int64, float, int]:
            df[b], _ = pd.factorize(df[b])

    # ------------------------------------------------------------------------------
    # 3) Summary statistics
    # ------------------------------------------------------------------------------
    cols_for_summary = covariates_list + [treatment_col] + outcomes_list
    summary_stats = df[cols_for_summary].describe().T.drop("count", axis=1, errors="ignore")
    print("\n=== Summary Statistics ===")
    print(tabulate(summary_stats, headers="keys", tablefmt="github"))

    # ------------------------------------------------------------------------------
    # 4) Correlations with treatment
    # ------------------------------------------------------------------------------
    corr_results = []
    for col in cols_for_summary:
        if col != treatment_col:
            r, p = pearsonr(df[col], df[treatment_col])
            corr_results.append({"Variable": col, "Correlation": r, "p-value": p})

    print("\n=== Correlations with auction_type ===")
    print(tabulate(corr_results, headers="keys", tablefmt="github"))

    # ------------------------------------------------------------------------------
    # 5) DoubleML Analysis: ATE, GATE, CATE
    # ------------------------------------------------------------------------------
    os.makedirs("experiment2", exist_ok=True)

    for outcome in outcomes_list:
        # Prepare for DoubleML
        df["Y"] = df[outcome]
        dml_data = doubleml.DoubleMLData(
            df, y_col="Y", d_cols=treatment_col, x_cols=covariates_list
        )
        ml_g = LGBMRegressor(random_state=123, verbose=-1)
        ml_m = LGBMClassifier(random_state=123, verbose=-1)
        dml_irm = doubleml.DoubleMLIRM(
            dml_data, ml_g=ml_g, ml_m=ml_m, n_folds=10, n_rep=1, score="ATE"
        )
        dml_irm.fit()

        print(f"\n========== {outcome.upper()} | ATE Results ==========")
        print(dml_irm.summary)

        # --------------------------------------
        # GATE for binary covariates
        # --------------------------------------
        binary_covs = [c for c in covariates_list if df[c].nunique() == 2]
        n_bin = len(binary_covs)
        nrows_gate = math.ceil(n_bin / 3)
        ncols_gate = min(n_bin, 3)
        fig_gate, axes_gate = plt.subplots(nrows=nrows_gate, ncols=ncols_gate,
                                           figsize=(5 * ncols_gate, 4 * nrows_gate))
        if n_bin == 1:
            axes_gate = np.array([axes_gate])
        gate_results = []

        for i, bin_col in enumerate(binary_covs):
            groups_df = df[[bin_col]].astype("category")
            gate_obj = dml_irm.gate(groups=groups_df)
            ci_95 = gate_obj.confint(level=0.95)
            eff = ci_95["effect"]
            lo = ci_95["2.5 %"]
            hi = ci_95["97.5 %"]
            gate_sum = gate_obj.summary
            errs = gate_sum["std err"]

            ax = axes_gate.flatten()[i] if n_bin > 1 else axes_gate[0]
            x_pos = np.arange(len(eff))
            ax.errorbar(x_pos, eff, yerr=[eff - lo, hi - eff], fmt="o", capsize=5)
            ax.set_title(f"GATE: {bin_col} ({outcome})")
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"{bin_col}={lvl}" for lvl in range(len(eff))])
            ax.set_ylabel("Estimated GATE")

            # Two-group difference t-test
            if len(eff) == 2:
                dval = eff.iloc[1] - eff.iloc[0]
                dvar = errs.iloc[1]**2 + errs.iloc[0]**2
                dse = math.sqrt(dvar)
                tval = dval / dse
                pval = 2.0 * (1.0 - norm.cdf(abs(tval)))
                gate_results.append({
                    "Variable": bin_col,
                    "Group0_Effect": f"{eff.iloc[0]:.4f}",
                    "Group1_Effect": f"{eff.iloc[1]:.4f}",
                    "Diff(Group1-Group0)": f"{dval:.4f}",
                    "StdErr(Diff)": f"{dse:.4f}",
                    "t-value": f"{tval:.4f}",
                    "p-value": f"{pval:.4f}"
                })

        fig_gate.tight_layout()
        gate_path = os.path.join("experiment2", f"gate_plots_{outcome}.png")
        fig_gate.savefig(gate_path, bbox_inches="tight")
        plt.close(fig_gate)

        if gate_results:
            print(f"\nT-tests for GATE ({outcome}):")
            print(tabulate(gate_results, headers="keys", tablefmt="github"))

        # --------------------------------------
        # CATE for continuous covariates
        # --------------------------------------
        cont_covs = [c for c in covariates_list if df[c].nunique() > 2]
        n_cont = len(cont_covs)
        nrows_cate = math.ceil(n_cont / 3)
        ncols_cate = min(n_cont, 3)
        fig_cate, axes_cate = plt.subplots(nrows=nrows_cate, ncols=ncols_cate,
                                           figsize=(5 * ncols_cate, 4 * nrows_cate))
        if n_cont == 1:
            axes_cate = np.array([axes_cate])

        for i, cont_col in enumerate(cont_covs):
            design_matrix = patsy.dmatrix(f"bs({cont_col}, df=5, degree=2)", df)
            spline_basis = pd.DataFrame(design_matrix)
            cate_obj = dml_irm.cate(basis=spline_basis)
            ci_95_cate = cate_obj.confint(basis=spline_basis, level=0.95)
            eff_cate = ci_95_cate["effect"].values
            lo_cate = ci_95_cate["2.5 %"].values
            hi_cate = ci_95_cate["97.5 %"].values

            xvals = df[cont_col].values
            idx_sort = np.argsort(xvals)
            x_sort = xvals[idx_sort]
            eff_sort = eff_cate[idx_sort]
            lo_sort = lo_cate[idx_sort]
            hi_sort = hi_cate[idx_sort]

            axc = axes_cate.flatten()[i] if n_cont > 1 else axes_cate[0]
            axc.plot(x_sort, eff_sort, label="CATE")
            axc.fill_between(x_sort, lo_sort, hi_sort, alpha=0.2, label="95% CI")
            axc.set_title(f"CATE: {cont_col} ({outcome})")
            axc.set_xlabel(cont_col)
            axc.set_ylabel("Estimated TE")
            axc.legend()

        fig_cate.tight_layout()
        cate_path = os.path.join("experiment2", f"cate_plots_{outcome}.png")
        fig_cate.savefig(cate_path, bbox_inches="tight")
        plt.close(fig_cate)

    print("\nAll analysis complete. Results saved in 'experiment2/' directory.")
