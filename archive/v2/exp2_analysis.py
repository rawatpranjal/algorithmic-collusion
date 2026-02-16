#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import doubleml as dml
from doubleml import DoubleMLData, DoubleMLIRM
from lightgbm import LGBMRegressor, LGBMClassifier
import patsy
from tabulate import tabulate
import math
import warnings

# Suppress known, non-critical warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Read the path of the latest experiment run from the script's directory
with open(os.path.join(script_dir, "LATEST_RUN.txt"), "r") as f:
    output_dir = f.read().strip()

df = pd.read_csv(os.path.join(output_dir, "data.csv"))

# --- Preprocessing ---
# Create explicit dummy variables for categorical features
df = pd.get_dummies(df, columns=['init', 'exploration'], drop_first=True, dtype=int)
# Convert treatment to binary 0/1
treatment_col = 'auction_type'
df[treatment_col] = (df[treatment_col] == 'first').astype(int)
# Convert boolean flags to integers for analysis
for col in ['median_opp_past_bid_index', 'winner_bid_index_state', 'asynchronous']:
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)

# --- Define variable lists ---
outcomes_list = ['avg_rev_last_1000', 'time_to_converge', 'avg_regret_of_seller']
# Programmatically define covariates
non_feature_cols = outcomes_list + [treatment_col, 'seed']
covariates_list = [col for col in df.columns if col not in non_feature_cols]

# --- Analysis ---
with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
    # Summary Statistics
    summary_stats = df[covariates_list + [treatment_col] + outcomes_list].describe().T
    f.write("=== Summary Statistics ===\n")
    f.write(tabulate(summary_stats, headers='keys', tablefmt="github"))
    f.write("\n\n")
    
    for outcome in outcomes_list:
        f.write(f"\n=======================================================\n")
        f.write(f"           ANALYSIS FOR OUTCOME: {outcome.upper()}         \n")
        f.write(f"=======================================================\n\n")

        dml_data = DoubleMLData(df, y_col=outcome, d_cols=treatment_col, x_cols=covariates_list)
        ml_g = LGBMRegressor(random_state=123, verbose=-1)
        ml_m = LGBMClassifier(random_state=123, verbose=-1)
        
        n_folds = 2 if len(df) <= 10 else (3 if len(df) < 50 else 10)
        dml_irm = DoubleMLIRM(dml_data, ml_g=ml_g, ml_m=ml_m, n_folds=n_folds, n_rep=1, score='ATE')
        dml_irm.fit()

        f.write(f"========== {outcome.upper()} | ATE Results ==========\n")
        f.write(str(dml_irm.summary))
        f.write("\n\n")

        # GATEs
        f.write(f"=== Group Average Treatment Effects (GATEs) for {outcome} ===\n")
        binary_covs = [c for c in covariates_list if df[c].nunique() == 2]
        
        n_bin = len(binary_covs)
        if n_bin > 0:
            nrows_gate = math.ceil(n_bin / 3)
            ncols_gate = min(n_bin, 3)
            fig_gate, axes_gate = plt.subplots(nrows=nrows_gate, ncols=ncols_gate, figsize=(5 * ncols_gate, 4 * nrows_gate))
            if n_bin == 1: axes_gate = np.array([axes_gate])

            for i, bin_col in enumerate(binary_covs):
                groups_df = df[[bin_col]].astype('category')
                gate_obj = dml_irm.gate(groups=groups_df)
                f.write(f"\n--- GATE for {bin_col} ---\n")
                f.write(str(gate_obj.summary))
                f.write("\n")
                
                # Plotting logic for GATE
                ci_95 = gate_obj.confint(level=0.95)
                eff = ci_95['effect']
                lo = ci_95['2.5 %']
                hi = ci_95['97.5 %']
                ax = axes_gate.flatten()[i]
                x_pos = np.arange(len(eff))
                ax.errorbar(x_pos, eff, yerr=[eff - lo, hi - eff], fmt='o', capsize=5)
                ax.set_title(f"GATE: {bin_col}")
                ax.set_xticks(x_pos)
                ax.set_xticklabels([f"{bin_col}={lvl}" for lvl in eff.index])
                ax.set_ylabel("Estimated GATE")

            fig_gate.tight_layout()
            gate_path = os.path.join(output_dir, f"gate_plots_{outcome}.png")
            fig_gate.savefig(gate_path, bbox_inches='tight')
            plt.close(fig_gate)
        
        # CATE Summary Table (Best Linear Predictor)
        f.write(f"\n=== CATE Drivers for {outcome} (Best Linear Predictor) ===\n")
        blp_basis_df = df[covariates_list]
        blp_obj = dml_irm.cate(basis=blp_basis_df)
        f.write("Coefficients for the Best Linear Predictor of the CATE on all covariates:\n")
        f.write(str(blp_obj.summary))
        f.write("\n\n")
        
        # CATE Plots
        cont_covs = [c for c in covariates_list if df[c].nunique() > 2]
        if cont_covs:
            n_cont = len(cont_covs)
            nrows_cate = math.ceil(n_cont / 3)
            ncols_cate = min(n_cont, 3)
            fig_cate, axes_cate = plt.subplots(nrows=nrows_cate, ncols=ncols_cate, figsize=(5 * ncols_cate, 4 * nrows_cate))
            if n_cont == 1: axes_cate = np.array([axes_cate])

            for i, cont_col in enumerate(cont_covs):
                design_matrix = patsy.dmatrix(f"cr({cont_col}, df=5)", df, return_type='dataframe')
                spline_basis = pd.DataFrame(design_matrix)
                cate_obj = dml_irm.cate(basis=spline_basis)
                ci_95_cate = cate_obj.confint(basis=spline_basis, level=0.95)
                
                axc = axes_cate.flatten()[i] if n_cont > 1 else axes_cate[0]
                idx = np.argsort(df[cont_col].values)
                axc.plot(df[cont_col].values[idx], ci_95_cate['effect'].values[idx], label='CATE')
                axc.fill_between(df[cont_col].values[idx], ci_95_cate['2.5 %'].values[idx], ci_95_cate['97.5 %'].values[idx], alpha=0.2, label='95% CI')
                axc.set_title(f"CATE: {cont_col}")
                axc.set_xlabel(cont_col)
                axc.set_ylabel("Estimated TE")
                axc.legend()

            fig_cate.tight_layout()
            cate_path = os.path.join(output_dir, f"cate_plots_{outcome}.png")
            fig_cate.savefig(cate_path, bbox_inches='tight')
            plt.close(fig_cate)

print(f"Analysis complete. All results saved in {output_dir}")
