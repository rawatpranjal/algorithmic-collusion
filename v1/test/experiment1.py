import numpy as np
import random
import pandas as pd
from scipy.stats import ttest_ind
from tabulate import tabulate
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# econml
from econml.dml import CausalForestDML, LinearDML
from econml.orf import DMLOrthoForest
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import matplotlib.pyplot as plt

def significance_stars(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    else:
        return ""

def get_rewards(b1, b2, auction_type="first"):
    if b1 > b2:
        winner = "agent"
    elif b2 > b1:
        winner = "rival"
    else:
        winner = "agent" if np.random.rand() < 0.5 else "rival"

    if winner == "agent":
        if auction_type == "first":
            return 1 - b1, 0
        else:
            return 1 - b2, 0
    else:
        if auction_type == "first":
            return 0, 1 - b2
        else:
            return 0, 1 - b1

def run_experiment(alpha, gamma, episodes, auction_type, init, 
                   algo, exploration, winning_info, opp_past_bid_index,
                   seed=0):
    np.random.seed(seed)
    random.seed(seed)

    n_actions = 6
    actions = np.linspace(0, 1, n_actions)

    # Q-table init
    if init == "random":
        Q1 = np.random.rand(n_actions)
        Q2 = np.random.rand(n_actions)
    else:
        Q1 = np.zeros(n_actions)
        Q2 = np.zeros(n_actions)

    # Epsilon decay
    start_eps, end_eps = 0.1, 0.0
    decay_end = int(0.9 * episodes)

    revenues = []
    last_rival_action = None

    for ep in range(episodes):
        if ep < decay_end:
            eps = start_eps - (ep / decay_end) * (start_eps - end_eps)
        else:
            eps = end_eps

        a1 = np.argmax(Q1) if np.random.rand() > eps else np.random.randint(n_actions)
        a2 = np.argmax(Q2) if np.random.rand() > eps else np.random.randint(n_actions)

        b1, b2 = actions[a1], actions[a2]
        r1, r2 = get_rewards(b1, b2, auction_type)

        # Q-update
        Q1[a1] += alpha * (r1 + gamma * np.max(Q1) - Q1[a1])
        Q2[a2] += alpha * (r2 + gamma * np.max(Q2) - Q2[a2])

        last_rival_action = a2
        revenues.append(max(b1, b2))

    avg_rev_last_1000 = np.mean(revenues[-1000:]) if len(revenues) >= 1000 else np.mean(revenues)
    final_rev = avg_rev_last_1000
    lower_bound = 0.95 * final_rev
    window = 200
    time_to_converge = episodes
    for t in range(len(revenues) - window):
        if np.mean(revenues[t : t + window]) >= lower_bound:
            time_to_converge = t
            break

    frac_lost = [1 - r for r in revenues]
    avg_frac_rev_loss = np.mean(frac_lost)

    return {
        "avg_rev_last_1000": avg_rev_last_1000,
        "time_to_converge": time_to_converge,
        "avg_frac_rev_loss": avg_frac_rev_loss
    }

# ------------------------------------------------------------------------
# 1) Parameter definitions
# ------------------------------------------------------------------------
param_space = {
    "alpha":              [0.01, 0.05],
    "gamma":              [0.9, 0.99],
    "episodes":           [50_000, 100_000],
    "auction_type":       ["first", "second"],
    "init":               ["random", "zeros"],
    "algo":               ["qlearning", "sarsa"],
    "exploration":        ["egreedy", "boltzmann"],
    "winning_info":       [False, True],
    "opp_past_bid_index": [False, True]
}
outcomes = ["avg_rev_last_1000", "time_to_converge", "avg_frac_rev_loss"]

# ------------------------------------------------------------------------
# 2) Generate data from random parameter draws
# ------------------------------------------------------------------------
K = 100
results = []
for k in range(K):
    alpha = random.choice(param_space["alpha"])
    gamma = random.choice(param_space["gamma"])
    episodes = random.choice(param_space["episodes"])
    auction_type = random.choice(param_space["auction_type"])
    init = random.choice(param_space["init"])
    algo = random.choice(param_space["algo"])
    exploration = random.choice(param_space["exploration"])
    winning_info = random.choice(param_space["winning_info"])
    opp_past_bid_index = random.choice(param_space["opp_past_bid_index"])

    outcome = run_experiment(alpha, gamma, episodes, auction_type, init,
                             algo, exploration, winning_info, opp_past_bid_index,
                             seed=k)
    outcome["alpha"] = alpha
    outcome["gamma"] = gamma
    outcome["episodes"] = episodes
    outcome["auction_type"] = auction_type
    outcome["init"] = init
    outcome["algo"] = algo
    outcome["exploration"] = exploration
    outcome["winning_info"] = winning_info
    outcome["opp_past_bid_index"] = opp_past_bid_index
    results.append(outcome)

df = pd.DataFrame(results)

# ------------------------------------------------------------------------
# (Optional) T-tests, OLS, and Causal Forest sections
# ------------------------------------------------------------------------

# ----------------------------
# 1) FDR-Corrected T-Tests
# ----------------------------
param_pairs = {
    "alpha":              [0.01, 0.05],
    "gamma":              [0.9, 0.99],
    "episodes":           [50_000, 100_000],
    "auction_type":       ["first", "second"],
    "init":               ["random", "zeros"],
    "algo":               ["qlearning", "sarsa"],
    "exploration":        ["egreedy", "boltzmann"],
    "winning_info":       [False, True],
    "opp_past_bid_index": [False, True]
}

raw_pvals = []
ttest_results = []
for param, (v1, v2) in param_pairs.items():
    for out in outcomes:
        data_v1 = df.loc[df[param] == v1, out]
        data_v2 = df.loc[df[param] == v2, out]
        if len(data_v1) > 1 and len(data_v2) > 1:
            t_stat, p_val = ttest_ind(data_v1, data_v2, equal_var=False)
            raw_pvals.append(p_val)
            ttest_results.append((param, v1, v2, out, t_stat, p_val))
        else:
            ttest_results.append((param, v1, v2, out, np.nan, np.nan))

raw_pvals_array = np.array([res[5] for res in ttest_results if not np.isnan(res[5])])
reject, pvals_corrected, _, _ = multipletests(raw_pvals_array, alpha=0.05, method='fdr_bh')
print("Applied FDR correction to T-test p-values.\n")

pc_idx = 0
corrected_data = []
for (param, v1, v2, out, t_stat, old_p) in ttest_results:
    if not np.isnan(t_stat):
        new_p = pvals_corrected[pc_idx]
        pc_idx += 1
    else:
        new_p = np.nan
    corrected_data.append((param, v1, v2, out, t_stat, new_p))

grouped = {}
for (param, v1, v2, out, t_stat, new_p) in corrected_data:
    grouped.setdefault((param, v1, v2), {})[out] = (t_stat, new_p)

rows = []
for (param, v1, v2), outcome_dict in grouped.items():
    row = [param, f"{v1} vs {v2}"]
    for out in outcomes:
        t_stat, adj_p = outcome_dict.get(out, (np.nan, np.nan))
        if not np.isnan(t_stat):
            row.append(f"{t_stat:.3f}{significance_stars(adj_p)}")
            row.append(f"{adj_p:.3g}")
        else:
            row.append("nan")
            row.append("nan")
    rows.append(row)

ttest_headers = [
    "Param", "Values",
    "T-stat(Rev)", "p(FDR)_Rev",
    "T-stat(Time)", "p(FDR)_Time",
    "T-stat(Frac)", "p(FDR)_Frac",
]
rows_sorted = sorted(rows, key=lambda x: x[0])
print("T-TEST RESULTS with FDR-Corrected P-Values:\n")
print(tabulate(rows_sorted, headers=ttest_headers, tablefmt="pretty"))
print("\n")

# ----------------------------
# 2) OLS Regressions
# ----------------------------
df["alpha_001"]        = (df["alpha"] == 0.01).astype(int)
df["gamma_09"]         = (df["gamma"] == 0.9).astype(int)
df["episodes_50k"]     = (df["episodes"] == 50_000).astype(int)
df["auction_first"]    = (df["auction_type"] == "first").astype(int)
df["init_random"]      = (df["init"] == "random").astype(int)
df["algo_qlearning"]   = (df["algo"] == "qlearning").astype(int)
df["explore_egreedy"]  = (df["exploration"] == "egreedy").astype(int)
df["wininfo_true"]     = df["winning_info"].astype(int)
df["opp_bid_idx_true"] = df["opp_past_bid_index"].astype(int)

regressors = [
    "alpha_001", "gamma_09", "episodes_50k", "auction_first",
    "init_random", "algo_qlearning", "explore_egreedy",
    "wininfo_true", "opp_bid_idx_true"
]
formula_base = " ~ " + " + ".join(regressors)
ols_results = {}

for out in outcomes:
    formula = out + formula_base
    model = sm.OLS.from_formula(formula, data=df)
    try:
        fit = model.fit()
        ols_results[out] = fit
        print(f"  Successfully fitted OLS for outcome: {out}")
    except Exception as e:
        print(f"  Error fitting OLS for outcome {out}: {e}")

var_list = ["Intercept"] + regressors
combined_table = []
for var in var_list:
    row = [var]
    for out in outcomes:
        coef = ols_results[out].params.get(var, np.nan)
        pval = ols_results[out].pvalues.get(var, np.nan)
        star = significance_stars(pval)
        if np.isnan(coef):
            row.append("nan")
        else:
            row.append(f"{coef:.3g}{star}")
    combined_table.append(row)

stat_row = ["N / R2 / F-pval"]
for out in outcomes:
    if out in ols_results:
        n_obs = ols_results[out].nobs
        r2 = ols_results[out].rsquared
        f_pval = ols_results[out].f_pvalue
        stat_row.append(f"N={int(n_obs)}, R2={r2:.2f}, p={f_pval:.3g}")
    else:
        stat_row.append("nan")
combined_table.append(stat_row)

combined_headers = ["Variable", "Rev", "Time", "Frac"]
print("OLS REGRESSION RESULTS (all outcomes in one table):\n")
print(tabulate(combined_table, headers=combined_headers, tablefmt="pretty"))
print("\n")

# ----------------------------
# 3) Causal Forest for Auction Type
# ----------------------------
print("Starting Causal Forest Analysis for Auction Type's Impact on Outcomes...\n")
df["T"] = df["auction_first"].astype(int)
unique_T = df["T"].unique()
counts_T = df["T"].value_counts()
print(f"Treatment Variable 'T' Distribution: {counts_T.to_dict()}")
if len(unique_T) < 2:
    print("Error: Not enough variation in 'T'.")
else:
    X_cols = [
        "alpha_001", "gamma_09", "episodes_50k",
        "init_random", "algo_qlearning",
        "explore_egreedy", "wininfo_true", "opp_bid_idx_true"
    ]
    print(f"Feature columns for Causal Forest: {X_cols}\n")

    cate_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    for i, out in enumerate(outcomes):
        print(f"Processing Causal Forest for outcome: {out}")
        Y = df[out].values
        T = df["T"].values
        X = df[X_cols].values

        unique_Y = np.unique(Y)
        if len(unique_Y) < 2:
            print(f"  Warning: Outcome '{out}' has insufficient variation.")
            cate_rows.append([out, np.nan])
            axes[i].set_title(f"{out} (Insufficient Y variation)")
            continue

        cf_model = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100, max_depth=5),
            model_t=RandomForestRegressor(n_estimators=100, max_depth=5),
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=10,
            max_samples=0.5,
            random_state=42,
            inference=True
        )
        try:
            print("  Fitting CausalForestDML...")
            cf_model.fit(Y=Y, T=T, X=X)
            print("  CausalForestDML fitted successfully.")
            tau_hat = cf_model.effect(X)
            avg_te = np.mean(tau_hat)
            cate_rows.append([out, avg_te])
            print(f"  Average Treatment Effect (ATE) for '{out}': {avg_te:.4f}")

            axes[i].scatter(X[:, 0], tau_hat, alpha=0.5)
            axes[i].set_title(f"{out} (Avg TE={avg_te:.3f})")
            axes[i].set_xlabel("alpha_001")
            axes[i].set_ylabel("Estimated Treatment Effect")

        except Exception as e:
            print(f"  Error fitting CausalForestDML for outcome '{out}': {e}")
            cate_rows.append([out, np.nan])
            axes[i].set_title(f"{out} (Error)")

    plt.tight_layout()
    plt.show()

    try:
        print("Computing Feature Importances from the last fitted CF model...")
        feat_importances = cf_model.feature_importances()
        print("Feature Importances computed successfully.")

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(X_cols)), feat_importances)
        plt.xticks(range(len(X_cols)), X_cols, rotation=45, ha='right')
        plt.title("Causal Forest Feature Importances")
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error computing feature importances: {e}")

    print("\nEstimated CATE (Auction Type = 'first' vs 'second'):")
    print(tabulate(cate_rows, headers=["Outcome", "Avg Treatment Effect"], tablefmt="pretty"))

# ------------------------------------------------------------------------
# 4) FIXED DML CODE SECTION
#    LinearDML for each outcome with "auction_type" as treatment
#    Prints summary for each fitted model.
# ------------------------------------------------------------------------
df_ml = df.dropna(subset=outcomes)
if len(df_ml) < 10:
    print("Insufficient data for DML.")
else:
    df_ml["alpha_001"]       = (df_ml["alpha"] == 0.01).astype(int)
    df_ml["gamma_09"]        = (df_ml["gamma"] == 0.9).astype(int)
    df_ml["episodes_50k"]    = (df_ml["episodes"] == 50_000).astype(int)
    df_ml["init_random"]     = (df_ml["init"] == "random").astype(int)
    df_ml["algo_qlearning"]  = (df_ml["algo"] == "qlearning").astype(int)
    df_ml["explore_egreedy"] = (df_ml["exploration"] == "egreedy").astype(int)
    df_ml["wininfo_true"]    = df_ml["winning_info"].astype(int)
    df_ml["opp_bid_idx_true"] = df_ml["opp_past_bid_index"].astype(int)

    X_cols = [
        "alpha_001", "gamma_09", "episodes_50k", "init_random",
        "algo_qlearning", "explore_egreedy", "wininfo_true", "opp_bid_idx_true"
    ]
    X_data = df_ml[X_cols]
    T_data = (df_ml["auction_type"] == "first").astype(int).values

    for outc in outcomes:
        y_data = df_ml[outc].values

        dml_est = LinearDML(
            model_y=RandomForestRegressor(),
            model_t=RandomForestClassifier(),
            fit_cate_intercept=True,
            discrete_treatment=True,
            random_state=123
        )
        dml_est.fit(Y=y_data, T=T_data, X=X_data, inference="statsmodels")
        print(f"\n[LinearDML Summary for Outcome: {outc}]")
        print(dml_est.summary())
