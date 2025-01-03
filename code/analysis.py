#!/usr/bin/env python3

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import researchpy as rp
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from stargazer.stargazer import Stargazer
from econml.dml import LinearDML
from econml.orf import DMLOrthoForest
from econml.sklearn_extensions.linear_model import WeightedLasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
import eli5
from eli5.sklearn import PermutationImportance

if __name__ == "__main__":

    # Read data
    df = pd.read_csv("data/data.csv")
    print(df.head())
    print(df.describe())
    
    # --------------------------
    # FIGURE 1 & FIGURE 2 EXAMPLES
    # (You would create them separately as you like, e.g. first-price, second-price, etc.)
    # --------------------------
    # Example:
    # sns.lineplot(...)  # replicate your "first-price-visual.png"
    # plt.savefig("figures/first-price-visual.png")

    # --------------------------
    # FIGURE 3: stacked boxplots for bid2val, bid2val_std, episodes
    # --------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    sns.boxplot(ax=axes[0], data=df, x="design", y="bid2val")
    axes[0].set_title("Bid2Val")
    sns.boxplot(ax=axes[1], data=df, x="design", y="bid2val_std")
    axes[1].set_title("Volatility")
    sns.boxplot(ax=axes[2], data=df, x="design", y="episodes")
    axes[2].set_title("Episodes")
    plt.tight_layout()
    plt.savefig("figures/boxplot_stacked.png")
    plt.close()

    # --------------------------
    # SIMPLE T-TEST
    # --------------------------
    df0 = df[df['design'] == 0]['bid2val']
    df1 = df[df['design'] == 1]['bid2val']
    if len(df0) > 1 and len(df1) > 1:
        summary, results = rp.ttest(group1=df0, group1_name="1st Price",
                                    group2=df1, group2_name="2nd Price")
        print("[T-test Results]\n", summary)
        print(stats.ttest_ind(df0, df1))

    # --------------------------
    # OLS FOR BIDS-TO-VALUE (TABLE 4)
    # --------------------------
    est1 = smf.ols("bid2val ~ design", data=df).fit()
    est2 = smf.ols("bid2val ~ design + N + alpha + gamma + egreedy + asynchronous + feedback + num_actions + explore_frac", data=df).fit()
    print("\n[OLS for Bids-to-Value]\n")
    print(est1.summary())
    print(est2.summary())

    # --------------------------
    # OLS FOR bid2val_stdATILITY (TABLE 5)
    # --------------------------
    est_bid2val_std1 = smf.ols("bid2val_std ~ design", data=df).fit()
    est_bid2val_std2 = smf.ols("bid2val_std ~ design + N + alpha + gamma + egreedy + asynchronous + feedback + num_actions + explore_frac", data=df).fit()
    print("\n[OLS for bid2val_stdatility]\n")
    print(est_bid2val_std1.summary())
    print(est_bid2val_std2.summary())

    # --------------------------
    # OLS FOR EPISODES (TABLE 6)
    # --------------------------
    est_ep1 = smf.ols("episodes ~ design", data=df).fit()
    est_ep2 = smf.ols("episodes ~ design + N + alpha + gamma + egreedy + asynchronous + feedback + num_actions + explore_frac", data=df).fit()
    print("\n[OLS for Episodes]\n")
    print(est_ep1.summary())
    print(est_ep2.summary())

    # Save Stargazer outputs if desired
    sg_bid = Stargazer([est1, est2])
    sg_bid2val_std = Stargazer([est_bid2val_std1, est_bid2val_std2])
    sg_ep  = Stargazer([est_ep1, est_ep2])
    with open("figures/ols_bid2val.tex", "w") as f:
        f.write(sg_bid.render_latex())
    with open("figures/ols_bid2val_std.tex", "w") as f:
        f.write(sg_bid2val_std.render_latex())
    with open("figures/ols_episodes.tex", "w") as f:
        f.write(sg_ep.render_latex())

    # --------------------------
    # DOUBLE ML (TABLE 7)
    # --------------------------
    # Drop missing
    df_ml = df.dropna()
    if len(df_ml) > 1:
        y = df_ml["bid2val"]
        T = df_ml["design"]
        # Exclude columns not used in X
        X = df_ml.drop(["bid2val", "episodes", "bid2val_std", "bid2val_min",
                        "bid2val_max", "design", "explore_frac"], axis=1)
        dml_est = LinearDML()
        dml_est.fit(y, T, X=X)
        print("\n[LinearDML for Bids-to-Value]\n", dml_est.summary())

        # OrthoForest
        orf = DMLOrthoForest(n_trees=10, max_depth=4,
                             model_Y=WeightedLasso(alpha=0.01),
                             model_T=WeightedLasso(alpha=0.01),
                             random_state=123)
        orf.fit(y, T, X=X)
        te = orf.effect(X)
        print("\n[OrthoForest Avg TE]:", te.mean())
        # Example scatter plot
        if X.shape[1] > 0:
            plt.scatter(X.iloc[:, 0], te, alpha=0.5)
            plt.title("DMLOrthoForest Estimated Treatment Effects")
            plt.xlabel(X.columns[0])
            plt.ylabel("Treatment Effect")
            plt.savefig("figures/orf_treatment_effects.png")
            plt.close()

    # --------------------------
    # RANDOM FOREST (EXTRA EXAMPLE)
    # --------------------------
    y_cv = df["episodes"]
    X_cv = df.drop(["bid2val","episodes","bid2val_std","bid2val_min","bid2val_max"], axis=1)
    if len(X_cv) > 0 and len(y_cv) > 0:
        rf = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=123)
        kf = KFold(n_splits=3, shuffle=True, random_state=123)
        scores = cross_val_score(rf, X_cv, y_cv, cv=kf, scoring="r2")
        print("\n[RF R^2 scores]:", scores)
        print("Mean R^2:", scores.mean())

        rf.fit(X_cv, y_cv)
        imps = rf.feature_importances_
        inds = imps.argsort()[::-1]
        print("\n[Feature Importances]:")
        for i, idx in enumerate(inds):
            print(f"{i+1}. {X_cv.columns[idx]}: {imps[idx]:.4f}")
        perm_imp = PermutationImportance(rf).fit(X_cv, y_cv)
        print("\n[Permutation Importances]:\n", 
              eli5.format_as_text(eli5.explain_weights(perm_imp)))

    print("Analysis complete.")
