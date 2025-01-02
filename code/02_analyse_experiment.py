import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# ============ LOAD DATA ============
df_path = os.path.join("code", "data", "data_experiment_main.pkl")
df = pd.read_pickle(df_path)

# (Optional) rename columns if needed:
df.columns = [
    'bid2val','episodes','bid2val_std','bid2val_min','bid2val_max',
    'N','alpha','gamma','egreedy','asynchronous','design','feedback',
    'num_actions','decay'
]

print(df.head())
print(df.describe())

# ============ CREATE FIGURES FOLDER ============
if not os.path.exists("code/figures"):
    os.makedirs("code/figures")

# ============ PLOTS =============
sns.boxplot(data=df, x="design", y="bid2val")
plt.savefig("code/figures/boxplot_bid2val.png")
plt.close()

sns.boxplot(data=df, x="design", y="bid2val_std")
plt.savefig("code/figures/boxplot_vol.png")
plt.close()

sns.boxplot(data=df, x="design", y="episodes")
plt.savefig("code/figures/boxplot_episodes.png")
plt.close()

# ============ REGRESSION ==========
est1 = smf.ols('bid2val ~ design', data=df).fit()
est2 = smf.ols('bid2val ~ design + N + alpha + gamma + egreedy + asynchronous + feedback + num_actions + decay', data=df).fit()

print("\n==== Regression 1: Simple OLS ====")
print(est1.summary())

print("\n==== Regression 2: OLS with Covariates ====")
print(est2.summary())
