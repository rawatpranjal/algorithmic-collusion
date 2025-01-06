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

# EconML imports (Double ML, Causal Forest)
from econml.dml import LinearDML
from econml.dml import CausalForestDML  # if you want direct CF from econml
from econml.orf import DMLOrthoForest
from econml.sklearn_extensions.linear_model import WeightedLasso
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

import eli5
from eli5.sklearn import PermutationImportance

def main():
    #----------------------------------------------------------------
    # 1) LOAD DATA
    #----------------------------------------------------------------
    df = pd.read_csv("data/new_data.csv")  # updated CSV with new columns
    print(df.head())
    print(df.describe())

    #----------------------------------------------------------------
    # 2) QUICK EDA
    #    - Distribution plots (kde/hist) for each numeric variable
    #    - Categorical distribution for discrete ones
    #----------------------------------------------------------------
    numeric_cols = ["revenue", "time_to_converge", "volatility", 
                    "alpha", "gamma", "num_actions", "explore_frac"]
    cat_cols = ["N", "egreedy", "asynchronous", "design", 
                "observation", "algorithm", "init_method"]

    # KDE/Histogram for numeric
    for col in numeric_cols:
        plt.figure()
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(f"figures/dist_{col}.png")
        plt.close()

    # Countplot for categorical
    for col in cat_cols:
        plt.figure()
        sns.countplot(data=df, x=col)
        plt.title(f"Count of {col}")
        plt.tight_layout()
        plt.savefig(f"figures/count_{col}.png")
        plt.close()

    # Distribution of Y across each X (e.g. boxplot or violin)
    # Example with `revenue` across `design`
    plt.figure()
    sns.boxplot(data=df, x="design", y="revenue")
    plt.title("Revenue by Auction Design")
    plt.tight_layout()
    plt.savefig("figures/revenue_by_design.png")
    plt.close()

    #----------------------------------------------------------------
    # 3) SIMPLE T-TEST: design vs. revenue
    #    Suppose design=0 => second-price, design=1 => first-price
    #----------------------------------------------------------------
    df0 = df[df['design'] == 0]['revenue']
    df1 = df[df['design'] == 1]['revenue']
    if len(df0) >= 2 and len(df1) >= 2:
        summary, results = rp.ttest(group1=df0, group1_name="Design=0",
                                    group2=df1, group2_name="Design=1")
        print("\n[T-test Summary]\n", summary)
        print("\n[SciPy ttest_ind]\n", stats.ttest_ind(df0, df1))

    #----------------------------------------------------------------
    # 4) REGRESSION ANALYSIS (OLS)
    #    Example with revenue, volatility, time_to_converge
    #----------------------------------------------------------------
    # -- OLS for REVENUE
    formula_rev1 = "revenue ~ design"
    formula_rev2 = (
        "revenue ~ design + N + alpha + gamma + egreedy + asynchronous "
        "+ num_actions + explore_frac + observation + algorithm + init_method"
    )
    est_rev1 = smf.ols(formula_rev1, data=df).fit()
    est_rev2 = smf.ols(formula_rev2, data=df).fit()
    print("\n[OLS for Revenue]\n")
    print(est_rev1.summary())
    print(est_rev2.summary())

    # -- OLS for VOLATILITY
    formula_vol1 = "volatility ~ design"
    formula_vol2 = (
        "volatility ~ design + N + alpha + gamma + egreedy + asynchronous "
        "+ num_actions + explore_frac + observation + algorithm + init_method"
    )
    est_vol1 = smf.ols(formula_vol1, data=df).fit()
    est_vol2 = smf.ols(formula_vol2, data=df).fit()
    print("\n[OLS for Volatility]\n")
    print(est_vol1.summary())
    print(est_vol2.summary())

    # -- OLS for TIME_TO_CONVERGE
    formula_ttc1 = "time_to_converge ~ design"
    formula_ttc2 = (
        "time_to_converge ~ design + N + alpha + gamma + egreedy + asynchronous "
        "+ num_actions + explore_frac + observation + algorithm + init_method"
    )
    est_ttc1 = smf.ols(formula_ttc1, data=df).fit()
    est_ttc2 = smf.ols(formula_ttc2, data=df).fit()
    print("\n[OLS for Time to Converge]\n")
    print(est_ttc1.summary())
    print(est_ttc2.summary())

    # Save Stargazer outputs if desired
    sg_rev = Stargazer([est_rev1, est_rev2])
    sg_vol = Stargazer([est_vol1, est_vol2])
    sg_ttc = Stargazer([est_ttc1, est_ttc2])
    with open("figures/ols_revenue.tex", "w") as f:
        f.write(sg_rev.render_latex())
    with open("figures/ols_volatility.tex", "w") as f:
        f.write(sg_vol.render_latex())
    with open("figures/ols_time_to_converge.tex", "w") as f:
        f.write(sg_ttc.render_latex())

    #----------------------------------------------------------------
    # 5) DOUBLE ML (LinearDML + OrthoForest)
    #----------------------------------------------------------------
    # Make sure df has no missing
    df_ml = df.dropna()
    if len(df_ml) > 50:  # some arbitrary threshold
        # Suppose T = design, Y = revenue
        # X => everything else (drop Y, T, other outcome columns)
        Y = df_ml["revenue"]
        T = df_ml["design"]
        X = df_ml.drop(["revenue", "time_to_converge", "volatility", "design"], axis=1)

        # -- LinearDML
        dml_est = LinearDML(random_state=42)
        dml_est.fit(Y, T, X=X)
        print("\n[LinearDML for Revenue vs. Design]\n", dml_est.summary())

        # -- OrthoForest
        orf = DMLOrthoForest(
            n_trees=50, max_depth=5,
            model_Y=WeightedLasso(alpha=0.01),
            model_T=WeightedLasso(alpha=0.01),
            random_state=42
        )
        orf.fit(Y, T, X=X)
        te_ = orf.effect(X)
        print("\n[OrthoForest Average Treatment Effect]:", np.mean(te_))

        # Example visualization
        if X.shape[1] > 0:
            plt.figure()
            plt.scatter(X.iloc[:, 0], te_, alpha=0.5)
            plt.xlabel(X.columns[0])
            plt.ylabel("Estimated Treatment Effect")
            plt.title("OrthoForest T.E. vs. First X Feature")
            plt.savefig("figures/orf_te_scatter.png")
            plt.close()

    #----------------------------------------------------------------
    # 6) CAUSAL FOREST (CausalForestDML) - direct demonstration
    #----------------------------------------------------------------
    # Example: treat T=design, outcome Y=revenue
    # (Below is minimal – you can adapt partial dependence, feature importances, etc.)
    if len(df_ml) > 50:
        Y_cf = df_ml["revenue"]
        T_cf = df_ml["design"]
        X_cf = df_ml.drop(["revenue", "time_to_converge", "volatility", "design"], axis=1)

        # Split for train/test
        X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
            X_cf, T_cf, Y_cf, test_size=0.2, random_state=42
        )

        cf_model = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100, max_depth=5),
            model_t=RandomForestRegressor(n_estimators=100, max_depth=5),
            n_estimators=100, max_depth=5, min_samples_leaf=10, max_samples=0.7,
            random_state=42
        )
        cf_model.fit(Y=Y_train, T=T_train, X=X_train)

        # Effects
        te_pred = cf_model.effect(X_test)
        print("\n[CausalForestDML] Mean T.E.:", np.mean(te_pred))

        # Optional feature importances
        fi = cf_model.feature_importances()
        if len(fi) == X_train.shape[1]:
            plt.figure()
            plt.bar(range(len(fi)), fi)
            plt.title("CausalForest Feature Importances")
            plt.xlabel("Feature Index")
            plt.ylabel("Importance")
            plt.savefig("figures/cf_feature_importance.png")
            plt.close()

    #----------------------------------------------------------------
    # 7) RANDOM FOREST FOR AN OUTCOME (EXAMPLE)
    #----------------------------------------------------------------
    # e.g. Predict time_to_converge from the rest
    y_cv = df["time_to_converge"]
    X_cv = df.drop(["revenue", "time_to_converge", "volatility", "design"], axis=1)
    X_cv = X_cv.dropna()
    y_cv = y_cv[X_cv.index]

    if len(X_cv) > 10:
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

if __name__ == "__main__":
    main()
