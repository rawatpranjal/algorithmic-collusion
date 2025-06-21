import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import doubleml as dml
from doubleml import DoubleMLData, DoubleMLIRM, DoubleMLBLP
from lightgbm import LGBMRegressor, LGBMClassifier
import patsy
from scipy.stats import norm
from tabulate import tabulate
import math
np.random.seed(12234334)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# 1) Dictionary of Variable Definitions
var_definitions = {
    "alpha": "Learning rate for Q-updates (0.01 to 0.1).",
    "gamma": "Discount factor for future rewards (0.0 to 0.99).",
    "episodes": "Total number of training episodes (10k to 100k).",
    "auction_type": "'first' or 'second' price auction (treatment).",
    "init": "Q-initialization: 'random' or 'zeros'.",
    "exploration": "Exploration strategy: 'egreedy' or 'boltzmann'.",
    "asynchronous": "Update mode: 0=synchronous, 1=asynchronous.",
    "n_bidders": "Number of bidding agents (2, 4, 6).",
    "median_opp_past_bid_index": "Use median of opponents' past bids in state?",
    "winner_bid_index_state": "Track winning bid index in state?",
    "avg_rev_last_1000": "Mean seller revenue in the final 1000 episodes.",
    "time_to_converge": "Fraction of episodes until ±5% convergence.",
    "avg_regret_of_seller": "Average regret for the seller (valuations=1).",
    "r": "Reserve price for the auction (0.0 to 0.5).",
    "boltzmann_temp_start": "Starting temperature for Boltzmann exploration."
}

df_var_defs = pd.DataFrame([{"Parameter": k, "Definition": v} for k, v in var_definitions.items()])

print("\n=== Variable Definitions ===")
print(tabulate(df_var_defs, headers="keys", tablefmt="github"))

# 3) Load and Preprocess Data
# Read the path of the latest experiment run from the script's directory
with open(os.path.join(script_dir, "LATEST_RUN.txt"), "r") as f:
    output_dir = f.read().strip()

df = pd.read_csv(os.path.join(output_dir, "data.csv"))

# Create explicit dummy variables for categorical features for interpretability
df = pd.get_dummies(df, columns=['init', 'exploration'], drop_first=True, dtype=int)

# Convert treatment to binary 0/1
treatment_col = 'auction_type'
df[treatment_col] = (df[treatment_col] == 'first').astype(int)

# Convert boolean flags to integers for analysis
for col in ['median_opp_past_bid_index', 'winner_bid_index_state']:
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)

# Programmatically define covariates
outcomes_list = ['avg_rev_last_1000', 'time_to_converge', 'avg_regret_of_seller']
# Exclude outcomes, treatment, and the seed from the feature set
non_feature_cols = outcomes_list + [treatment_col, 'seed']
covariates_list = [col for col in df.columns if col not in non_feature_cols]

# Open a results file to capture all textual output
with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
    f.write("=== Variable Definitions ===\n")
    f.write(tabulate(df_var_defs, headers="keys", tablefmt="github"))
    f.write("\n\n")

    # 4) Print Summary Statistics
    cols_for_summary = covariates_list + [treatment_col] + outcomes_list
    summary_stats = df[cols_for_summary].describe().T.drop('count', axis=1, errors='ignore')
    f.write("=== Summary Statistics ===\n")
    f.write(tabulate(summary_stats, headers='keys', tablefmt='github'))
    f.write("\n\n")

    # 5) DoubleML IRM with GATE and CATE
    for outcome in outcomes_list:
        df['Y'] = df[outcome]
        dml_data = dml.DoubleMLData(df, y_col='Y', d_cols=treatment_col, x_cols=covariates_list)
        ml_g = LGBMRegressor(random_state=123, verbose=-1)
        ml_m = LGBMClassifier(random_state=123, verbose=-1)
        # GATE/CATE methods require n_rep=1
        dml_irm = DoubleMLIRM(dml_data, ml_g=ml_g, ml_m=ml_m, n_folds=10, n_rep=1, score='ATE')
        dml_irm.fit()

        f.write(f"========== {outcome.upper()} | ATE Results ==========\n")
        f.write(str(dml_irm.summary))
        f.write("\n\n")

        # 5a) GATE
        f.write(f"=== Group Average Treatment Effects (GATEs) for {outcome} ===\n")
        binary_covs = [c for c in covariates_list if df[c].nunique() == 2]
        n_bin = len(binary_covs)
        nrows_gate = math.ceil(n_bin / 3)
        ncols_gate = min(n_bin, 3)
        fig_gate, axes_gate = plt.subplots(nrows=nrows_gate, ncols=ncols_gate, figsize=(5*ncols_gate, 4*nrows_gate))
        if n_bin == 1:
            axes_gate = np.array([axes_gate])

        for i, bin_col in enumerate(binary_covs):
            groups_df = df[[bin_col]].astype('category')
            gate_obj = dml_irm.gate(groups=groups_df)
            
            # Write summary table to results file
            f.write(f"\n--- GATE for {bin_col} ---\n")
            f.write(str(gate_obj.summary))
            f.write("\n")

            ci_95 = gate_obj.confint(level=0.95)
            eff = ci_95['effect']
            lo = ci_95['2.5 %']
            hi = ci_95['97.5 %']
            ax = axes_gate.flatten()[i] if n_bin > 1 else axes_gate[0]
            x_pos = np.arange(len(eff))
            ax.errorbar(x_pos, eff, yerr=[eff - lo, hi - eff], fmt='o', capsize=5)
            ax.set_title(f"GATE: {bin_col} ({outcome})")
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"{bin_col}={lvl}" for lvl in range(len(eff))])
            ax.set_ylabel("Estimated GATE")

        fig_gate.tight_layout()
        gate_path = os.path.join(output_dir, f"gate_plots_{outcome}.png")
        fig_gate.savefig(gate_path, bbox_inches='tight')
        plt.close(fig_gate)

        # 5b) CATE Plots
        cont_covs = [c for c in covariates_list if df[c].nunique() > 2]
        if cont_covs:
            n_cont = len(cont_covs)
            nrows_cate = math.ceil(n_cont / 3)
            ncols_cate = min(n_cont, 3)
            fig_cate, axes_cate = plt.subplots(nrows=nrows_cate, ncols=ncols_cate, figsize=(5*ncols_cate, 4*nrows_cate))
            if n_cont == 1:
                axes_cate = np.array([axes_cate])

            for i, cont_col in enumerate(cont_covs):
                # Use natural splines (cr) for more stable CATE plots
                design_matrix = patsy.dmatrix(f"cr({cont_col}, df=5)", df, return_type='dataframe')
                spline_basis = pd.DataFrame(design_matrix)
                cate_obj = dml_irm.cate(basis=spline_basis)
                ci_95_cate = cate_obj.confint(basis=spline_basis, level=0.95)
                eff_cate = ci_95_cate['effect'].values
                lo_cate = ci_95_cate['2.5 %'].values
                hi_cate = ci_95_cate['97.5 %'].values
                xvals = df[cont_col].values
                idx = np.argsort(xvals)
                x_sort = xvals[idx]
                eff_sort = eff_cate[idx]
                lo_sort = lo_cate[idx]
                hi_sort = hi_cate[idx]
                axc = axes_cate.flatten()[i] if n_cont > 1 else axes_cate[0]
                axc.plot(x_sort, eff_sort, label='CATE')
                axc.fill_between(x_sort, lo_sort, hi_sort, alpha=0.2, label='95% CI')
                axc.set_title(f"CATE: {cont_col} ({outcome})")
                axc.set_xlabel(cont_col)
                axc.set_ylabel("Estimated TE")
                axc.legend()

            fig_cate.tight_layout()
            cate_path = os.path.join(output_dir, f"cate_plots_{outcome}.png")
            fig_cate.savefig(cate_path, bbox_inches='tight')
            plt.close(fig_cate)

        # 5c) CATE Summary Table (Best Linear Predictor)
        f.write(f"\n=== CATE Drivers for {outcome} (Best Linear Predictor) ===\n")
        
        # Project the CATE onto all covariates to find drivers of heterogeneity
        blp_basis_df = df[covariates_list]
        blp_obj = dml_irm.cate(basis=blp_basis_df)

        # The result is a BLP object, and its summary gives the coefficients.
        f.write("Coefficients for the Best Linear Predictor of the CATE on all covariates:\n")
        f.write(str(blp_obj.summary))
        f.write("\n\n")

print(f"Analysis complete. All results saved in {output_dir}")
