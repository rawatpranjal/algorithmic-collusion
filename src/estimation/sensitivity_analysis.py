#!/usr/bin/env python3
"""
Global Sensitivity Analysis (shared module for all experiments).

Computes variance-based sensitivity indices (Sobol') and multi-method
importance rankings for factorial experimental designs.

Sections:
  A. Analytical Sobol' indices from ANOVA sum of squares
  B. Random Forest permutation importance
  C. SHAP values via TreeSHAP (LightGBM)
  D. Morris method / elementary effects (via surrogate)
  E. Monte Carlo Sobol' via surrogate (3 surrogates: NN, LGBM, Kriging)
  E2. GP-based Sobol' with posterior UQ (Marrel et al. 2009)
  F. FAST - Fourier Amplitude Sensitivity Test (via surrogate)
  G. Borgonovo delta moment-independent measure
  H. Cross-method concordance
  I. Summary

Public API:
    run_sensitivity_analysis(df, coded_cols, response_cols, output_dir,
                              experiment_id=None, factor_groups=None)

Standalone usage:
    PYTHONPATH=src python3 src/estimation/sensitivity_analysis.py --exp 1
"""

import argparse
import itertools
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
})

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from tabulate import tabulate

# Optional: SALib (sections D-G)
try:
    import SALib
    from SALib.sample import morris as morris_sample
    from SALib.sample import saltelli as saltelli_sample
    from SALib.sample import fast_sampler
    from SALib.analyze import morris as morris_analyze
    from SALib.analyze import sobol as sobol_analyze
    from SALib.analyze import fast as fast_analyze
    from SALib.analyze import delta as delta_analyze
    from SALib.sample import sobol as sobol_sample_mod
    _HAS_SALIB = True
except ImportError:
    _HAS_SALIB = False

# Optional: shap
try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _clean(name):
    """Remove _coded suffix for display."""
    return name.replace("_coded", "")


def _json_safe(obj):
    """Make numpy types JSON-serializable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


def _build_formula(response, coded_cols):
    """Y ~ X1 + X2 + ... + X1:X2 + X1:X3 + ..."""
    main = coded_cols
    interactions = [f"{a}:{b}" for a, b in itertools.combinations(coded_cols, 2)]
    return f"{response} ~ {' + '.join(main + interactions)}"


def _infer_factor_groups(coded_cols):
    """Auto-detect factor groups from coded column names.

    Groups columns like eta_linear_coded + eta_quadratic_coded into a single
    factor 'eta'. Single columns map to themselves.

    Returns dict: factor_name -> list of coded column names.
    """
    groups = {}
    used = set()

    # Detect paired contrasts (_linear_coded / _quadratic_coded)
    for col in coded_cols:
        if col.endswith("_linear_coded"):
            base = col.replace("_linear_coded", "")
            quad = base + "_quadratic_coded"
            if quad in coded_cols:
                groups[base] = [col, quad]
                used.add(col)
                used.add(quad)

    # Remaining columns are single-factor groups
    for col in coded_cols:
        if col not in used:
            base = col.replace("_coded", "")
            groups[base] = [col]

    return groups


def _build_salib_problem(factor_groups):
    """Build SALib problem dict from factor groups.

    Each coded column becomes a separate variable with bounds [-1, 1].
    """
    names = []
    bounds = []
    for factor, cols in factor_groups.items():
        for col in cols:
            names.append(col)
            bounds.append([-1.0, 1.0])
    return {"num_vars": len(names), "names": names, "bounds": bounds}


def _aggregate_to_factors(values_per_col, factor_groups):
    """Sum per-column values into per-factor values for grouped factors."""
    result = {}
    for factor, cols in factor_groups.items():
        result[factor] = sum(values_per_col.get(col, 0.0) for col in cols)
    return result


def _fit_surrogates(X, y):
    """Fit three surrogate models: NN, LightGBM, Kriging.

    Returns dict of {name: (model, r2_cv)} with cross-validated R².
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.neural_network import MLPRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel
    import lightgbm as lgb

    surrogates = {}
    n_samples = X.shape[0]

    # 1. LightGBM
    lgbm = lgb.LGBMRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        num_leaves=31, min_child_samples=max(5, n_samples // 50),
        verbose=-1, n_jobs=-1,
    )
    try:
        cv_r2 = cross_val_score(lgbm, X, y, cv=min(5, n_samples // 10),
                                scoring="r2")
        lgbm.fit(X, y)
        surrogates["LGBM"] = (lgbm, float(np.mean(cv_r2)))
    except Exception as e:
        print(f"    LGBM surrogate failed: {e}")

    # 2. Neural Network (MLP)
    nn = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=1000,
            early_stopping=True, validation_fraction=0.15,
            random_state=42, learning_rate="adaptive",
        )),
    ])
    try:
        cv_r2 = cross_val_score(nn, X, y, cv=min(5, n_samples // 10),
                                scoring="r2")
        nn.fit(X, y)
        surrogates["NN"] = (nn, float(np.mean(cv_r2)))
    except Exception as e:
        print(f"    NN surrogate failed: {e}")

    # 3. Kriging (Gaussian Process) — subsample if large dataset
    max_gp_samples = 500
    if n_samples > max_gp_samples:
        idx = np.random.RandomState(42).choice(n_samples, max_gp_samples,
                                                replace=False)
        X_gp, y_gp = X[idx], y[idx]
    else:
        X_gp, y_gp = X, y

    kernel = ConstantKernel(1.0) * Matern(nu=2.5)
    gp = Pipeline([
        ("scaler", StandardScaler()),
        ("gp", GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=3, random_state=42,
            normalize_y=True,
        )),
    ])
    try:
        cv_r2 = cross_val_score(gp, X_gp, y_gp,
                                cv=min(5, max(2, len(X_gp) // 10)),
                                scoring="r2")
        gp.fit(X_gp, y_gp)
        surrogates["Kriging"] = (gp, float(np.mean(cv_r2)))
    except Exception as e:
        print(f"    Kriging surrogate failed: {e}")

    return surrogates


def _fit_gp_for_sobol(X, y, max_samples=500):
    """Fit a dedicated GP for Sobol posterior UQ.

    Returns (scaler, gp, r2_cv) or None if fitting fails.
    Keeps scaler and GP separate for direct predict(return_cov=True) access.

    For factorial designs with binary inputs, averages replicates at each
    unique design point and constrains length scales to enable smooth
    interpolation for continuous Saltelli evaluation.
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

    # Average replicates at unique design points for cleaner GP training
    X_unique, inverse = np.unique(X, axis=0, return_inverse=True)
    if len(X_unique) < X.shape[0]:
        y_means = np.zeros(len(X_unique))
        y_counts = np.zeros(len(X_unique))
        for i, idx in enumerate(inverse):
            y_means[idx] += y[i]
            y_counts[idx] += 1
        y_means /= y_counts
        X_train, y_train = X_unique, y_means
    else:
        X_train, y_train = X, y

    n = X_train.shape[0]
    if n > max_samples:
        idx = np.random.RandomState(42).choice(n, max_samples, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Constrain length scales: minimum 0.5 prevents the GP from collapsing
    # to a lookup table on binary factorial inputs, enabling smooth
    # interpolation for continuous Saltelli evaluation points.
    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-2, 1e3))
        * Matern(nu=2.5, length_scale=1.0, length_scale_bounds=(0.5, 5.0))
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1.0))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=10, normalize_y=True,
        random_state=42,
    )
    try:
        cv_folds = min(5, max(2, len(X_scaled) // 10))
        cv_r2 = cross_val_score(gp, X_scaled, y_train, cv=cv_folds,
                                scoring="r2")
        gp.fit(X_scaled, y_train)
        return scaler, gp, float(np.mean(cv_r2))
    except Exception as e:
        print(f"    GP fitting failed: {e}")
        return None


# ===================================================================
# A. Analytical Sobol' Indices from ANOVA SS
# ===================================================================

def section_analytical_sobol(df, coded_cols, response_cols, factor_groups,
                              output_dir):
    """Compute exact Sobol' indices from Type III ANOVA sum of squares.

    For orthogonal factorial designs with effects coding (-1/+1), the ANOVA
    decomposition IS the Sobol/Hoeffding decomposition. This gives exact
    first-order (S1), second-order (S2), and total-order (ST) indices.
    """
    print("\n" + "=" * 60)
    print("A. ANALYTICAL SOBOL' INDICES (ANOVA-based)")
    print("=" * 60)

    results = {}

    for resp in response_cols:
        if resp not in df.columns:
            continue
        df_clean = df.dropna(subset=[resp])

        formula = _build_formula(resp, coded_cols)
        model = smf.ols(formula, data=df_clean).fit()
        anova = sm.stats.anova_lm(model, typ=3)

        # Extract sum of squares
        ss = {}
        for term in anova.index:
            if term in ("Intercept", "Residual"):
                continue
            ss[term] = float(anova.loc[term, "sum_sq"])

        ss_residual = float(anova.loc["Residual", "sum_sq"])
        ss_total = sum(ss.values()) + ss_residual

        if ss_total <= 0:
            print(f"\n  {_clean(resp)}: SS_total <= 0, skipping")
            continue

        # First-order: S1 per factor (sum SS of all coded cols in the group)
        s1_per_col = {}
        for term, val in ss.items():
            if ":" not in term:
                s1_per_col[term] = val / ss_total
        s1 = _aggregate_to_factors(s1_per_col, factor_groups)

        # Second-order: S2 per factor pair
        s2_raw = {}
        for term, val in ss.items():
            if ":" in term:
                s2_raw[term] = val / ss_total

        # Aggregate interactions to factor pairs
        s2 = {}
        for term, val in s2_raw.items():
            parts = term.split(":")
            # Find which factor each coded col belongs to
            factor_a = factor_b = None
            for factor, cols in factor_groups.items():
                if parts[0] in cols:
                    factor_a = factor
                if parts[1] in cols:
                    factor_b = factor
            if factor_a and factor_b and factor_a != factor_b:
                pair = tuple(sorted([factor_a, factor_b]))
                s2[pair] = s2.get(pair, 0.0) + val
            elif factor_a == factor_b and factor_a:
                # Intra-factor interaction (e.g., eta_linear:eta_quadratic)
                # Add to first-order for the factor
                s1[factor_a] = s1.get(factor_a, 0.0) + val

        # Total-order: ST = S1 + sum of S2 involving this factor
        st = {}
        for factor in factor_groups:
            st[factor] = s1.get(factor, 0.0)
            for pair, val in s2.items():
                if factor in pair:
                    st[factor] = st[factor] + val

        # Residual fraction
        residual_frac = ss_residual / ss_total

        # Validation: sum should equal 1.0
        check_sum = sum(s1.values()) + sum(s2.values()) + residual_frac
        valid = abs(check_sum - 1.0) < 1e-6

        print(f"\n  {_clean(resp)}:")
        print(f"    R² = {model.rsquared:.4f}")
        print(f"    Residual fraction = {residual_frac:.4f}")
        print(f"    Validation sum = {check_sum:.6f} ({'PASS' if valid else 'FAIL'})")
        print(f"    {'Factor':25s} {'S1':>8s} {'ST':>8s} {'ST-S1':>8s}")
        print(f"    {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
        for factor in sorted(st, key=lambda f: st[f], reverse=True):
            gap = st[factor] - s1.get(factor, 0.0)
            print(f"    {factor:25s} {s1.get(factor, 0.0):8.4f} "
                  f"{st[factor]:8.4f} {gap:8.4f}")

        results[resp] = {
            "s1": s1,
            "s2": {f"{p[0]}:{p[1]}": v for p, v in s2.items()},
            "st": st,
            "residual_fraction": residual_frac,
            "ss_total": ss_total,
            "r_squared": float(model.rsquared),
            "validation_sum": check_sum,
            "valid": valid,
        }

    # Plots: stacked bar + pie for each response
    for resp, res in results.items():
        _plot_sobol_stacked(res, resp, output_dir)
        _plot_sobol_pie(res, resp, output_dir)

    return results


def _plot_sobol_stacked(res, resp, output_dir):
    """Stacked bar: S1 (blue) + interaction contribution (orange)."""
    factors = sorted(res["st"], key=lambda f: res["st"][f], reverse=True)
    s1_vals = [res["s1"].get(f, 0.0) for f in factors]
    int_vals = [res["st"][f] - res["s1"].get(f, 0.0) for f in factors]

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(factors) * 0.4)))
    y = range(len(factors))
    ax.barh(y, s1_vals, color="#3498db", label="First-order (S1)")
    ax.barh(y, int_vals, left=s1_vals, color="#e67e22",
            label="Interaction (ST - S1)")
    ax.set_yticks(y)
    ax.set_yticklabels(factors)
    ax.set_xlabel("Fraction of total variance")
    ax.set_title(f"Sobol' Indices: {_clean(resp)}")
    ax.legend(loc="lower right")
    ax.set_xlim(0, max(res["st"].values()) * 1.15 if res["st"] else 1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"sobol_stacked_{resp}.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_sobol_pie(res, resp, output_dir):
    """Pie chart of variance decomposition."""
    factors = sorted(res["st"], key=lambda f: res["st"][f], reverse=True)
    sizes = [res["st"][f] for f in factors]
    residual = res["residual_fraction"]
    labels = factors + ["Residual"]
    sizes.append(residual)

    # Merge tiny slices (<1%) into "Other"
    threshold = 0.01
    other = sum(s for s in sizes if s < threshold)
    if other > 0:
        filtered = [(l, s) for l, s in zip(labels, sizes) if s >= threshold]
        labels = [l for l, _ in filtered] + ["Other"]
        sizes = [s for _, s in filtered] + [other]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title(f"Variance Decomposition: {_clean(resp)}")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"sobol_pie_{resp}.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# B. Random Forest Permutation Importance
# ===================================================================

def section_rf_importance(df, coded_cols, response_cols, factor_groups,
                          output_dir):
    """Permutation importance from Random Forest."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance

    print("\n" + "=" * 60)
    print("B. RANDOM FOREST PERMUTATION IMPORTANCE")
    print("=" * 60)

    results = {}

    for resp in response_cols:
        if resp not in df.columns:
            continue
        df_clean = df.dropna(subset=[resp])
        X = df_clean[coded_cols].values
        y = df_clean[resp].values

        rf = RandomForestRegressor(n_estimators=200, max_depth=6,
                                    random_state=42, n_jobs=-1)
        rf.fit(X, y)
        oob_r2 = float(rf.score(X, y))

        perm = permutation_importance(rf, X, y, n_repeats=10,
                                       random_state=42, n_jobs=-1)
        imp_per_col = dict(zip(coded_cols, perm.importances_mean))
        imp_std_per_col = dict(zip(coded_cols, perm.importances_std))

        # Aggregate to factor level
        imp = _aggregate_to_factors(imp_per_col, factor_groups)
        imp_std = _aggregate_to_factors(imp_std_per_col, factor_groups)

        # Normalize
        total = sum(max(0, v) for v in imp.values())
        if total > 0:
            imp_norm = {f: max(0, v) / total for f, v in imp.items()}
        else:
            imp_norm = {f: 0.0 for f in imp}

        print(f"\n  {_clean(resp)} (R² = {oob_r2:.4f}):")
        for factor in sorted(imp_norm, key=lambda f: imp_norm[f], reverse=True):
            print(f"    {factor:25s} {imp_norm[factor]:8.4f}")

        results[resp] = {
            "r2": oob_r2,
            "importance": imp_norm,
            "importance_raw": imp,
            "importance_std": imp_std,
        }

    # Plot for primary response
    for resp, res in results.items():
        factors = sorted(res["importance"], key=lambda f: res["importance"][f])
        vals = [res["importance"][f] for f in factors]

        fig, ax = plt.subplots(figsize=(8, max(3.5, len(factors) * 0.4)))
        ax.barh(range(len(factors)), vals, color="#27ae60")
        ax.set_yticks(range(len(factors)))
        ax.set_yticklabels(factors)
        ax.set_xlabel("Normalized permutation importance")
        ax.set_title(f"RF Importance: {_clean(resp)}")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"rf_importance_{resp}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

    return results


# ===================================================================
# C. SHAP Values (TreeSHAP via LightGBM)
# ===================================================================

def section_shap(df, coded_cols, response_cols, factor_groups, output_dir):
    """SHAP values via TreeSHAP on LightGBM model."""
    if not _HAS_SHAP:
        print("\n  SHAP not installed, skipping section C.")
        return {}

    import lightgbm as lgb

    print("\n" + "=" * 60)
    print("C. SHAP VALUES (TreeSHAP)")
    print("=" * 60)

    results = {}

    for resp in response_cols:
        if resp not in df.columns:
            continue
        df_clean = df.dropna(subset=[resp])
        X = df_clean[coded_cols].values
        y = df_clean[resp].values
        X_df = df_clean[coded_cols]

        model = lgb.LGBMRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            num_leaves=31, min_child_samples=max(5, len(df_clean) // 50),
            verbose=-1, n_jobs=-1,
        )
        model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Mean absolute SHAP per column
        mean_abs_shap_col = dict(zip(coded_cols,
                                      np.abs(shap_values).mean(axis=0)))
        # Aggregate to factors
        mean_abs_shap = _aggregate_to_factors(mean_abs_shap_col, factor_groups)

        # Normalize
        total = sum(mean_abs_shap.values())
        if total > 0:
            shap_norm = {f: v / total for f, v in mean_abs_shap.items()}
        else:
            shap_norm = {f: 0.0 for f in mean_abs_shap}

        print(f"\n  {_clean(resp)}:")
        for factor in sorted(shap_norm, key=lambda f: shap_norm[f],
                              reverse=True):
            print(f"    {factor:25s} {shap_norm[factor]:8.4f}")

        results[resp] = {
            "importance": shap_norm,
            "importance_raw": mean_abs_shap,
        }

        # SHAP bar plot
        fig, ax = plt.subplots(figsize=(8, max(3.5, len(shap_norm) * 0.4)))
        factors = sorted(shap_norm, key=lambda f: shap_norm[f])
        vals = [shap_norm[f] for f in factors]
        ax.barh(range(len(factors)), vals, color="#8e44ad")
        ax.set_yticks(range(len(factors)))
        ax.set_yticklabels(factors)
        ax.set_xlabel("Normalized mean |SHAP|")
        ax.set_title(f"SHAP Importance: {_clean(resp)}")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"shap_bar_{resp}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

        # SHAP beeswarm
        try:
            fig = plt.figure(figsize=(10, max(4, len(coded_cols) * 0.4)))
            shap.summary_plot(shap_values, X_df, show=False, max_display=20)
            plt.title(f"SHAP Beeswarm: {_clean(resp)}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"shap_beeswarm_{resp}.png"),
                        dpi=200, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"    Beeswarm plot failed: {e}")

    return results


# ===================================================================
# D. Morris Method / Elementary Effects
# ===================================================================

def section_morris(df, coded_cols, response_cols, factor_groups, output_dir):
    """Morris method using surrogates (NN, LGBM, Kriging)."""
    if not _HAS_SALIB:
        print("\n  SALib not installed, skipping section D.")
        return {}

    print("\n" + "=" * 60)
    print("D. MORRIS METHOD / ELEMENTARY EFFECTS")
    print("=" * 60)

    problem = _build_salib_problem(factor_groups)
    results = {}

    for resp in response_cols:
        if resp not in df.columns:
            continue
        df_clean = df.dropna(subset=[resp])
        X = df_clean[coded_cols].values
        y = df_clean[resp].values

        # Fit surrogates
        surrogates = _fit_surrogates(X, y)
        if not surrogates:
            print(f"\n  {_clean(resp)}: all surrogates failed, skipping")
            continue

        # Generate Morris samples
        try:
            X_morris = morris_sample.sample(problem, N=1000, num_levels=4)
        except Exception as e:
            print(f"\n  Morris sampling failed: {e}")
            continue

        # Evaluate each surrogate
        resp_results = {}
        for name, (model, r2_cv) in surrogates.items():
            try:
                Y_pred = model.predict(X_morris)
                analysis = morris_analyze.analyze(
                    problem, X_morris, Y_pred,
                    print_to_console=False,
                )
                mu_star = dict(zip(problem["names"], analysis["mu_star"]))
                sigma = dict(zip(problem["names"], analysis["sigma"]))

                # Aggregate to factors
                mu_star_f = _aggregate_to_factors(mu_star, factor_groups)
                sigma_f = _aggregate_to_factors(sigma, factor_groups)

                resp_results[name] = {
                    "mu_star": mu_star_f,
                    "sigma": sigma_f,
                    "r2_cv": r2_cv,
                }
            except Exception as e:
                print(f"    Morris ({name}) failed for {_clean(resp)}: {e}")

        if resp_results:
            results[resp] = resp_results
            print(f"\n  {_clean(resp)}:")
            for sname, sres in resp_results.items():
                print(f"    Surrogate: {sname} (CV R² = {sres['r2_cv']:.4f})")
                for f in sorted(sres["mu_star"],
                                key=lambda x: sres["mu_star"][x], reverse=True):
                    print(f"      {f:25s} μ*={sres['mu_star'][f]:10.4f}  "
                          f"σ={sres['sigma'][f]:10.4f}")

    # Plot: μ* vs σ scatter for best surrogate
    for resp, surrogate_results in results.items():
        best = max(surrogate_results, key=lambda s: surrogate_results[s]["r2_cv"])
        sres = surrogate_results[best]
        factors = list(sres["mu_star"].keys())
        mu_vals = [sres["mu_star"][f] for f in factors]
        sig_vals = [sres["sigma"][f] for f in factors]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(mu_vals, sig_vals, s=80, zorder=3, color="#e74c3c")
        for i, f in enumerate(factors):
            ax.annotate(f, (mu_vals[i], sig_vals[i]), fontsize=9,
                        ha="left", va="bottom",
                        xytext=(5, 5), textcoords="offset points")
        ax.set_xlabel("μ* (mean absolute elementary effect)")
        ax.set_ylabel("σ (std of elementary effects)")
        ax.set_title(f"Morris Plot: {_clean(resp)} ({best} surrogate)")
        # Classification lines
        if mu_vals:
            max_mu = max(mu_vals)
            ax.axvline(0.1 * max_mu, color="gray", linestyle="--", alpha=0.5,
                       label="Negligible threshold")
            ax.plot([0, max_mu * 1.1], [0, max_mu * 1.1], "k--", alpha=0.3,
                    label="σ = μ* (nonlinear/interactive)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"morris_{resp}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

    return results


# ===================================================================
# E. Monte Carlo Sobol' via Surrogates (NN, LGBM, Kriging)
# ===================================================================

def section_mc_sobol(df, coded_cols, response_cols, factor_groups, output_dir):
    """Monte Carlo Sobol' indices using three surrogate models."""
    if not _HAS_SALIB:
        print("\n  SALib not installed, skipping section E.")
        return {}

    print("\n" + "=" * 60)
    print("E. MONTE CARLO SOBOL' INDICES (3 Surrogates)")
    print("=" * 60)

    problem = _build_salib_problem(factor_groups)
    results = {}

    # Generate Saltelli samples once
    try:
        X_saltelli = saltelli_sample.sample(problem, N=4096,
                                             calc_second_order=True)
    except Exception as e:
        print(f"  Saltelli sampling failed: {e}")
        return {}

    for resp in response_cols:
        if resp not in df.columns:
            continue
        df_clean = df.dropna(subset=[resp])
        X = df_clean[coded_cols].values
        y = df_clean[resp].values

        surrogates = _fit_surrogates(X, y)
        if not surrogates:
            print(f"\n  {_clean(resp)}: all surrogates failed, skipping")
            continue

        resp_results = {}
        for name, (model, r2_cv) in surrogates.items():
            try:
                Y_pred = model.predict(X_saltelli)
                analysis = sobol_analyze.analyze(
                    problem, Y_pred, calc_second_order=True,
                    print_to_console=False,
                )
                s1_col = dict(zip(problem["names"], analysis["S1"]))
                st_col = dict(zip(problem["names"], analysis["ST"]))
                s1_conf_col = dict(zip(problem["names"], analysis["S1_conf"]))
                st_conf_col = dict(zip(problem["names"], analysis["ST_conf"]))

                # Aggregate to factors
                s1 = _aggregate_to_factors(s1_col, factor_groups)
                st = _aggregate_to_factors(st_col, factor_groups)
                s1_conf = _aggregate_to_factors(s1_conf_col, factor_groups)
                st_conf = _aggregate_to_factors(st_conf_col, factor_groups)

                resp_results[name] = {
                    "s1": s1, "st": st,
                    "s1_conf": s1_conf, "st_conf": st_conf,
                    "r2_cv": r2_cv,
                }
            except Exception as e:
                print(f"    MC Sobol ({name}) failed for {_clean(resp)}: {e}")

        if resp_results:
            results[resp] = resp_results
            print(f"\n  {_clean(resp)}:")
            for sname, sres in resp_results.items():
                print(f"    Surrogate: {sname} (CV R² = {sres['r2_cv']:.4f})")
                print(f"    {'Factor':25s} {'S1':>8s} {'ST':>8s} {'ST-S1':>8s}")
                for f in sorted(sres["st"],
                                key=lambda x: sres["st"][x], reverse=True):
                    gap = sres["st"][f] - sres["s1"].get(f, 0.0)
                    print(f"      {f:25s} {sres['s1'].get(f, 0.0):8.4f} "
                          f"{sres['st'][f]:8.4f} {gap:8.4f}")

    # Plot: analytical vs MC comparison for best surrogate
    return results


# ===================================================================
# E2. GP-Based Sobol' with Posterior Uncertainty (Marrel et al. 2009)
# ===================================================================

def section_gp_sobol(df, coded_cols, response_cols, factor_groups, output_dir,
                     n_saltelli=512, n_realizations=200, ci_level=0.90):
    """GP-based Sobol' indices with posterior uncertainty quantification.

    Uses the GP posterior covariance (Marrel et al. 2009, Approach 2) to
    produce distributional Sobol' indices with confidence intervals.
    Each S_i and S_T becomes a random variable whose distribution reflects
    metamodel uncertainty.
    """
    if not _HAS_SALIB:
        print("\n  SALib not installed, skipping section E2.")
        return {}

    print("\n" + "=" * 60)
    print("E2. GP-BASED SOBOL' WITH POSTERIOR UQ (Marrel et al. 2009)")
    print("=" * 60)

    problem = _build_salib_problem(factor_groups)
    d = problem["num_vars"]
    results = {}
    alpha = (1.0 - ci_level) / 2.0

    for resp in response_cols:
        if resp not in df.columns:
            continue
        df_clean = df.dropna(subset=[resp])
        X = df_clean[coded_cols].values
        y = df_clean[resp].values

        print(f"\n  {_clean(resp)}:")

        # 1. Fit dedicated GP
        gp_result = _fit_gp_for_sobol(X, y)
        if gp_result is None:
            print("    Skipping: GP fitting failed")
            continue
        scaler, gp, r2_cv = gp_result
        print(f"    GP CV R² = {r2_cv:.4f}  (training points: {len(gp.X_train_)})")

        # 2. Adapt N based on dimensionality
        N_eff = min(n_saltelli, 256) if d >= 9 else n_saltelli
        M = N_eff * (d + 2)

        # 3. Generate Saltelli samples
        try:
            X_eval = sobol_sample_mod.sample(problem, N=N_eff,
                                             calc_second_order=False)
        except Exception as e:
            print(f"    Saltelli sampling failed: {e}")
            continue

        # 4. GP predictions with full covariance
        X_eval_scaled = scaler.transform(X_eval)
        try:
            mu, cov = gp.predict(X_eval_scaled, return_cov=True)
        except Exception as e:
            print(f"    GP predict(return_cov=True) failed: {e}")
            continue

        # 5. Predictor-only (Approach 1)
        try:
            pred_only = sobol_analyze.analyze(
                problem, mu, calc_second_order=False,
                print_to_console=False,
            )
            pred_s1_col = dict(zip(problem["names"], pred_only["S1"]))
            pred_st_col = dict(zip(problem["names"], pred_only["ST"]))
            pred_s1 = _aggregate_to_factors(pred_s1_col, factor_groups)
            pred_st = _aggregate_to_factors(pred_st_col, factor_groups)
        except Exception as e:
            print(f"    Predictor-only Sobol failed: {e}")
            continue

        # 6. Posterior sampling (Approach 2)
        # Cholesky with progressive jitter
        L = None
        jitter_used = 0.0
        for jitter_exp in range(-8, 2):  # 1e-8 to 1e+1
            jitter = 10.0 ** jitter_exp
            try:
                L = np.linalg.cholesky(cov + jitter * np.eye(M))
                jitter_used = jitter
                break
            except np.linalg.LinAlgError:
                continue

        use_diagonal = False
        if L is None:
            print("    WARNING: Cholesky failed after 10 jitter attempts, "
                  "using diagonal (independent) sampling")
            diag_std = np.sqrt(np.maximum(np.diag(cov), 0.0))
            use_diagonal = True
            jitter_used = -1.0

        # Free cov matrix
        del cov

        # Draw realizations and compute Sobol indices per draw
        rng = np.random.RandomState(42)
        s1_draws = {col: [] for col in problem["names"]}
        st_draws = {col: [] for col in problem["names"]}
        n_successful = 0

        for k in range(n_realizations):
            z = rng.randn(M)
            if use_diagonal:
                y_k = mu + diag_std * z
            else:
                y_k = mu + L @ z

            try:
                sa_k = sobol_analyze.analyze(
                    problem, y_k, calc_second_order=False,
                    print_to_console=False,
                )
                for i, col in enumerate(problem["names"]):
                    s1_draws[col].append(sa_k["S1"][i])
                    st_draws[col].append(sa_k["ST"][i])
                n_successful += 1
            except Exception:
                continue

        if L is not None:
            del L

        if n_successful < 50:
            print(f"    WARNING: Only {n_successful}/{n_realizations} "
                  "realizations succeeded")
            if n_successful == 0:
                continue

        # 7. Aggregate draws to factor level, compute stats
        factors = list(factor_groups.keys())
        s1_mean, s1_std, s1_lo, s1_hi = {}, {}, {}, {}
        st_mean, st_std, st_lo, st_hi = {}, {}, {}, {}

        for factor, cols in factor_groups.items():
            # Sum contrast columns per draw
            factor_s1 = np.zeros(n_successful)
            factor_st = np.zeros(n_successful)
            for col in cols:
                factor_s1 += np.array(s1_draws[col][:n_successful])
                factor_st += np.array(st_draws[col][:n_successful])

            s1_mean[factor] = float(np.mean(factor_s1))
            s1_std[factor] = float(np.std(factor_s1))
            s1_lo[factor] = float(np.percentile(factor_s1, 100 * alpha))
            s1_hi[factor] = float(np.percentile(factor_s1, 100 * (1 - alpha)))

            st_mean[factor] = float(np.mean(factor_st))
            st_std[factor] = float(np.std(factor_st))
            st_lo[factor] = float(np.percentile(factor_st, 100 * alpha))
            st_hi[factor] = float(np.percentile(factor_st, 100 * (1 - alpha)))

        # Print results
        print(f"    Eval points: {M}, Realizations: {n_successful}/{n_realizations}")
        ci_pct = int(ci_level * 100)
        print(f"    {'Factor':25s} {'ST_mean':>8s} {'ST_std':>8s} "
              f"{'CI_lo':>8s} {'CI_hi':>8s}")
        print(f"    {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for f in sorted(st_mean, key=lambda x: st_mean[x], reverse=True):
            print(f"    {f:25s} {st_mean[f]:8.4f} {st_std[f]:8.4f} "
                  f"{st_lo[f]:8.4f} {st_hi[f]:8.4f}")

        resp_result = {
            "predictor_only": {"s1": pred_s1, "st": pred_st},
            "posterior": {
                "s1_mean": s1_mean, "s1_std": s1_std,
                "s1_ci_lo": s1_lo, "s1_ci_hi": s1_hi,
                "st_mean": st_mean, "st_std": st_std,
                "st_ci_lo": st_lo, "st_ci_hi": st_hi,
            },
            "gp_r2_cv": r2_cv,
            "gp_training_points": int(len(gp.X_train_)),
            "n_eval_points": M,
            "n_realizations": n_realizations,
            "n_successful": n_successful,
            "cholesky_jitter": jitter_used,
        }
        if r2_cv < 0.10:
            resp_result["warning"] = "GP_R2_LOW"
            print(f"    WARNING: GP R² = {r2_cv:.4f} < 0.10, results unreliable")

        results[resp] = resp_result

        # Plots
        _plot_gp_sobol_posterior(resp_result, resp, output_dir, ci_pct)

    return results


def _plot_gp_sobol_posterior(result, resp, output_dir, ci_pct):
    """Horizontal bar chart of posterior mean ST with CI error bars."""
    post = result["posterior"]
    pred = result["predictor_only"]
    factors = sorted(post["st_mean"], key=lambda f: post["st_mean"][f],
                     reverse=True)

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(factors) * 0.45)))
    y_pos = np.arange(len(factors))

    means = [post["st_mean"][f] for f in factors]
    lo = [post["st_mean"][f] - post["st_ci_lo"][f] for f in factors]
    hi = [post["st_ci_hi"][f] - post["st_mean"][f] for f in factors]
    pred_vals = [pred["st"].get(f, 0.0) for f in factors]

    ax.barh(y_pos, means, xerr=[lo, hi], color="#3498db", alpha=0.7,
            capsize=4, label=f"Posterior mean ({ci_pct}% CI)")
    ax.scatter(pred_vals, y_pos, marker="D", color="black", s=40, zorder=5,
               label="Predictor-only")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(factors)
    ax.set_xlabel("Total-order Sobol' index (ST)")
    ax.set_title(f"GP Posterior Sobol': {_clean(resp)}")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"gp_sobol_posterior_{resp}.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_gp_sobol_vs_analytical(all_results, response_cols, factor_groups,
                                  output_dir):
    """Grouped bar: Analytical ST vs GP predictor-only ST vs GP posterior mean ST."""
    if "analytical_sobol" not in all_results or "gp_sobol" not in all_results:
        return

    factors = list(factor_groups.keys())

    for resp in response_cols:
        if resp not in all_results["analytical_sobol"]:
            continue
        if resp not in all_results["gp_sobol"]:
            continue

        analytical = all_results["analytical_sobol"][resp]
        gp_data = all_results["gp_sobol"][resp]
        post = gp_data["posterior"]
        pred = gp_data["predictor_only"]

        a_st = [analytical["st"].get(f, 0.0) for f in factors]
        p_st = [pred["st"].get(f, 0.0) for f in factors]
        g_st = [post["st_mean"].get(f, 0.0) for f in factors]
        g_lo = [post["st_mean"][f] - post["st_ci_lo"][f] for f in factors]
        g_hi = [post["st_ci_hi"][f] - post["st_mean"][f] for f in factors]

        fig, ax = plt.subplots(figsize=(max(8, len(factors) * 1.2), 5))
        x = np.arange(len(factors))
        w = 0.25

        ax.bar(x - w, a_st, w, label="Analytical", color="#2c3e50", alpha=0.9)
        ax.bar(x, p_st, w, label="GP predictor-only", color="#e67e22", alpha=0.7)
        ax.bar(x + w, g_st, w, yerr=[g_lo, g_hi], capsize=3,
               label="GP posterior mean (90% CI)", color="#3498db", alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(factors, rotation=45, ha="right")
        ax.set_ylabel("Total-order Sobol' index (ST)")
        ax.set_title(f"Analytical vs GP Sobol': {_clean(resp)}")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir,
                                  f"gp_sobol_comparison_{resp}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)


# ===================================================================
# F. FAST (Fourier Amplitude Sensitivity Test)
# ===================================================================

def section_fast(df, coded_cols, response_cols, factor_groups, output_dir):
    """FAST via surrogate evaluation."""
    if not _HAS_SALIB:
        print("\n  SALib not installed, skipping section F.")
        return {}

    print("\n" + "=" * 60)
    print("F. FAST (Fourier Amplitude Sensitivity Test)")
    print("=" * 60)

    problem = _build_salib_problem(factor_groups)
    results = {}

    for resp in response_cols:
        if resp not in df.columns:
            continue
        df_clean = df.dropna(subset=[resp])
        X = df_clean[coded_cols].values
        y = df_clean[resp].values

        surrogates = _fit_surrogates(X, y)
        if not surrogates:
            continue

        # Use best surrogate
        best_name = max(surrogates, key=lambda s: surrogates[s][1])
        model, r2_cv = surrogates[best_name]

        try:
            X_fast = fast_sampler.sample(problem, N=2048)
            Y_pred = model.predict(X_fast)
            analysis = fast_analyze.analyze(problem, Y_pred,
                                             print_to_console=False)
            s1_col = dict(zip(problem["names"], analysis["S1"]))
            st_col = dict(zip(problem["names"], analysis["ST"]))

            s1 = _aggregate_to_factors(s1_col, factor_groups)
            st = _aggregate_to_factors(st_col, factor_groups)

            results[resp] = {
                "s1": s1, "st": st,
                "surrogate": best_name,
                "r2_cv": r2_cv,
            }

            print(f"\n  {_clean(resp)} ({best_name}, CV R² = {r2_cv:.4f}):")
            for f in sorted(st, key=lambda x: st[x], reverse=True):
                print(f"    {f:25s} S1={s1.get(f, 0.0):8.4f}  "
                      f"ST={st[f]:8.4f}")
        except Exception as e:
            print(f"    FAST failed for {_clean(resp)}: {e}")

    return results


# ===================================================================
# G. Borgonovo Delta Moment-Independent Measure
# ===================================================================

def section_delta(df, coded_cols, response_cols, factor_groups, output_dir):
    """Borgonovo delta moment-independent measure (works directly on data)."""
    if not _HAS_SALIB:
        print("\n  SALib not installed, skipping section G.")
        return {}

    print("\n" + "=" * 60)
    print("G. BORGONOVO DELTA MOMENT-INDEPENDENT MEASURE")
    print("=" * 60)

    problem = _build_salib_problem(factor_groups)
    results = {}

    for resp in response_cols:
        if resp not in df.columns:
            continue
        df_clean = df.dropna(subset=[resp])
        X = df_clean[coded_cols].values
        y = df_clean[resp].values

        try:
            analysis = delta_analyze.analyze(
                problem, X, y, num_resamples=100,
                print_to_console=False,
            )
            delta_col = dict(zip(problem["names"], analysis["delta"]))
            delta_conf_col = dict(zip(problem["names"],
                                       analysis["delta_conf"]))

            delta = _aggregate_to_factors(delta_col, factor_groups)
            delta_conf = _aggregate_to_factors(delta_conf_col, factor_groups)

            # Normalize
            total = sum(max(0, v) for v in delta.values())
            if total > 0:
                delta_norm = {f: max(0, v) / total for f, v in delta.items()}
            else:
                delta_norm = {f: 0.0 for f in delta}

            results[resp] = {
                "delta": delta,
                "delta_conf": delta_conf,
                "delta_norm": delta_norm,
            }

            print(f"\n  {_clean(resp)}:")
            for f in sorted(delta_norm,
                            key=lambda x: delta_norm[x], reverse=True):
                print(f"    {f:25s} δ={delta[f]:8.4f}  "
                      f"(norm={delta_norm[f]:8.4f})")
        except Exception as e:
            print(f"    Delta failed for {_clean(resp)}: {e}")

    return results


# ===================================================================
# H. Cross-Method Concordance
# ===================================================================

def section_concordance(all_results, response_cols, factor_groups, output_dir):
    """Compare factor importance rankings across all methods."""
    print("\n" + "=" * 60)
    print("H. CROSS-METHOD CONCORDANCE")
    print("=" * 60)

    factors = list(factor_groups.keys())
    results = {}

    for resp in response_cols:
        # Collect rankings from each method
        method_rankings = {}

        # A. Analytical Sobol ST
        if "analytical_sobol" in all_results and resp in all_results["analytical_sobol"]:
            st = all_results["analytical_sobol"][resp]["st"]
            method_rankings["Sobol_ST"] = st

        # B. RF importance
        if "rf_importance" in all_results and resp in all_results["rf_importance"]:
            method_rankings["RF"] = all_results["rf_importance"][resp]["importance"]

        # C. SHAP
        if "shap" in all_results and resp in all_results["shap"]:
            method_rankings["SHAP"] = all_results["shap"][resp]["importance"]

        # D. Morris (best surrogate)
        if "morris" in all_results and resp in all_results["morris"]:
            morris_data = all_results["morris"][resp]
            best = max(morris_data, key=lambda s: morris_data[s]["r2_cv"])
            mu_star = morris_data[best]["mu_star"]
            total = sum(max(0, v) for v in mu_star.values())
            if total > 0:
                method_rankings[f"Morris_{best}"] = {
                    f: max(0, v) / total for f, v in mu_star.items()
                }

        # E. MC Sobol (best surrogate)
        if "mc_sobol" in all_results and resp in all_results["mc_sobol"]:
            mc_data = all_results["mc_sobol"][resp]
            best = max(mc_data, key=lambda s: mc_data[s]["r2_cv"])
            st = mc_data[best]["st"]
            total = sum(max(0, v) for v in st.values())
            if total > 0:
                method_rankings[f"MC_Sobol_{best}"] = {
                    f: max(0, v) / total for f, v in st.items()
                }

        # F. FAST
        if "fast" in all_results and resp in all_results["fast"]:
            st = all_results["fast"][resp]["st"]
            total = sum(max(0, v) for v in st.values())
            if total > 0:
                method_rankings["FAST"] = {
                    f: max(0, v) / total for f, v in st.items()
                }

        # G. Borgonovo delta
        if "delta" in all_results and resp in all_results["delta"]:
            method_rankings["Delta"] = all_results["delta"][resp]["delta_norm"]

        # E2. GP Sobol (posterior mean ST)
        if "gp_sobol" in all_results and resp in all_results["gp_sobol"]:
            gp_data = all_results["gp_sobol"][resp]
            st = gp_data["posterior"]["st_mean"]
            total = sum(max(0, v) for v in st.values())
            if total > 0:
                method_rankings["GP_Sobol"] = {
                    f: max(0, v) / total for f, v in st.items()
                }

        if len(method_rankings) < 2:
            print(f"\n  {_clean(resp)}: insufficient methods for comparison")
            continue

        # Build importance matrix (methods × factors)
        methods = list(method_rankings.keys())
        imp_matrix = np.zeros((len(methods), len(factors)))
        for i, m in enumerate(methods):
            for j, f in enumerate(factors):
                imp_matrix[i, j] = method_rankings[m].get(f, 0.0)

        # Compute rank matrix
        rank_matrix = np.zeros_like(imp_matrix)
        for i in range(len(methods)):
            rank_matrix[i] = stats.rankdata(-imp_matrix[i])

        # Spearman rank correlation between methods
        n_methods = len(methods)
        spearman = np.ones((n_methods, n_methods))
        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                r, _ = stats.spearmanr(rank_matrix[i], rank_matrix[j])
                spearman[i, j] = r
                spearman[j, i] = r

        # Top-3 agreement
        top3_per_method = {}
        for i, m in enumerate(methods):
            ranked_factors = [factors[j] for j in np.argsort(-imp_matrix[i])]
            top3_per_method[m] = ranked_factors[:3]

        # Check if top factor is consistent
        top1_factors = [top3_per_method[m][0] for m in methods]
        top1_agreement = len(set(top1_factors)) == 1

        print(f"\n  {_clean(resp)}:")
        print(f"    Methods: {methods}")
        print(f"    Top-1 agreement: {'YES' if top1_agreement else 'NO'}")
        print(f"    Top-1 per method: {dict(zip(methods, top1_factors))}")

        mean_spearman = np.mean(spearman[np.triu_indices(n_methods, k=1)])
        print(f"    Mean Spearman correlation: {mean_spearman:.3f}")

        results[resp] = {
            "methods": methods,
            "importance_matrix": imp_matrix.tolist(),
            "rank_matrix": rank_matrix.tolist(),
            "spearman_matrix": spearman.tolist(),
            "mean_spearman": float(mean_spearman),
            "top1_agreement": top1_agreement,
            "top3_per_method": top3_per_method,
        }

        # Plot: concordance heatmap
        fig, ax = plt.subplots(figsize=(max(6, n_methods * 0.8 + 2),
                                         max(4, n_methods * 0.6 + 1)))
        im = ax.imshow(spearman, cmap="RdYlGn", vmin=-1, vmax=1)
        ax.set_xticks(range(n_methods))
        ax.set_yticks(range(n_methods))
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(methods, fontsize=9)
        for i in range(n_methods):
            for j in range(n_methods):
                ax.text(j, i, f"{spearman[i, j]:.2f}", ha="center",
                        va="center", fontsize=9,
                        color="white" if abs(spearman[i, j]) > 0.5 else "black")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"Method Concordance: {_clean(resp)}")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir,
                                  f"concordance_heatmap_{resp}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Plot: grouped bar chart (factors × methods)
        fig, ax = plt.subplots(figsize=(max(8, len(factors) * 1.2), 5))
        x = np.arange(len(factors))
        width = 0.8 / len(methods)
        colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
        for i, m in enumerate(methods):
            vals = [method_rankings[m].get(f, 0.0) for f in factors]
            ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                   label=m, color=colors[i])
        ax.set_xticks(x)
        ax.set_xticklabels(factors, rotation=45, ha="right")
        ax.set_ylabel("Normalized importance")
        ax.set_title(f"Factor Importance Comparison: {_clean(resp)}")
        ax.legend(fontsize=8, ncol=min(3, len(methods)))
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir,
                                  f"factor_comparison_{resp}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

    return results


# ===================================================================
# I. Summary
# ===================================================================

def section_summary(all_results, response_cols, factor_groups, output_dir,
                    experiment_id=None):
    """Generate summary of sensitivity analysis findings."""
    print("\n" + "=" * 60)
    print("I. SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 60)

    factors = list(factor_groups.keys())
    summary_lines = []
    exp_label = f"Experiment {experiment_id}" if experiment_id else "All"
    summary_lines.append("=" * 70)
    summary_lines.append(f"SENSITIVITY ANALYSIS SUMMARY -- {exp_label}")
    summary_lines.append("=" * 70)
    summary_lines.append("")

    results = {}

    for resp in response_cols:
        resp_summary = {}

        # Dominant factor (highest ST from analytical Sobol)
        if "analytical_sobol" in all_results and resp in all_results["analytical_sobol"]:
            st = all_results["analytical_sobol"][resp]["st"]
            s1 = all_results["analytical_sobol"][resp]["s1"]
            dominant = max(st, key=lambda f: st[f])
            resp_summary["dominant_factor"] = dominant
            resp_summary["dominant_st"] = st[dominant]
            resp_summary["dominant_s1"] = s1.get(dominant, 0.0)

            # Most interactive (largest ST - S1 gap)
            gaps = {f: st[f] - s1.get(f, 0.0) for f in st}
            most_interactive = max(gaps, key=lambda f: gaps[f])
            resp_summary["most_interactive"] = most_interactive
            resp_summary["interaction_gap"] = gaps[most_interactive]

            # Negligible factors (ST < 0.01)
            negligible = [f for f in st if st[f] < 0.01]
            resp_summary["negligible_factors"] = negligible

            summary_lines.append(f"{_clean(resp)}:")
            summary_lines.append(f"  Dominant factor: {dominant} "
                                 f"(ST={st[dominant]:.4f}, "
                                 f"S1={s1.get(dominant, 0.0):.4f})")
            summary_lines.append(f"  Most interactive: {most_interactive} "
                                 f"(gap={gaps[most_interactive]:.4f})")
            if negligible:
                summary_lines.append(
                    f"  Negligible (ST<0.01): {', '.join(negligible)}")
            summary_lines.append("")

        # Cross-method agreement
        if "concordance" in all_results and resp in all_results["concordance"]:
            conc = all_results["concordance"][resp]
            resp_summary["mean_spearman"] = conc["mean_spearman"]
            resp_summary["top1_agreement"] = conc["top1_agreement"]
            summary_lines.append(
                f"  Method agreement: mean Spearman = "
                f"{conc['mean_spearman']:.3f}, "
                f"top-1 {'consistent' if conc['top1_agreement'] else 'INCONSISTENT'}")
            summary_lines.append("")

        # Surrogate comparison
        if "mc_sobol" in all_results and resp in all_results["mc_sobol"]:
            mc_data = all_results["mc_sobol"][resp]
            summary_lines.append("  Surrogate CV R²:")
            for sname, sres in mc_data.items():
                summary_lines.append(f"    {sname:10s}: {sres['r2_cv']:.4f}")
            summary_lines.append("")

        results[resp] = resp_summary

    summary_text = "\n".join(summary_lines)
    summary_path = os.path.join(output_dir, "sensitivity_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"\n  Summary written to {summary_path}")
    print(summary_text)

    return results


# ===================================================================
# E (continued). Analytical vs MC Sobol Comparison Plot
# ===================================================================

def _plot_mc_comparison(all_results, response_cols, factor_groups, output_dir):
    """Plot analytical Sobol vs MC Sobol comparison with error bars."""
    if "analytical_sobol" not in all_results or "mc_sobol" not in all_results:
        return

    for resp in response_cols:
        if resp not in all_results["analytical_sobol"]:
            continue
        if resp not in all_results["mc_sobol"]:
            continue

        analytical = all_results["analytical_sobol"][resp]
        mc_data = all_results["mc_sobol"][resp]

        factors = list(factor_groups.keys())

        # Get analytical ST
        a_st = [analytical["st"].get(f, 0.0) for f in factors]

        # Get MC ST for each surrogate
        fig, ax = plt.subplots(figsize=(max(8, len(factors) * 1.2), 5))
        x = np.arange(len(factors))

        # Analytical
        ax.bar(x - 0.3, a_st, 0.2, label="Analytical", color="#2c3e50",
               alpha=0.9)

        # MC surrogates
        colors = {"LGBM": "#e74c3c", "NN": "#3498db", "Kriging": "#27ae60"}
        for i, (sname, sres) in enumerate(mc_data.items()):
            st_vals = [sres["st"].get(f, 0.0) for f in factors]
            conf_vals = [sres["st_conf"].get(f, 0.0) for f in factors]
            offset = -0.1 + i * 0.2
            ax.bar(x + offset, st_vals, 0.18,
                   label=f"MC ({sname}, R²={sres['r2_cv']:.3f})",
                   color=colors.get(sname, f"C{i+1}"), alpha=0.7,
                   yerr=conf_vals, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(factors, rotation=45, ha="right")
        ax.set_ylabel("Total-order Sobol' index (ST)")
        ax.set_title(f"Analytical vs MC Sobol': {_clean(resp)}")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir,
                                  f"sobol_mc_comparison_{resp}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)


# ===================================================================
# Public API
# ===================================================================

def run_sensitivity_analysis(df, coded_cols, response_cols, output_dir,
                              experiment_id=None, factor_groups=None):
    """
    Run the full multi-method sensitivity analysis.

    Parameters
    ----------
    df : DataFrame
        Experiment data with coded columns and response columns.
    coded_cols : list[str]
        Coded factor column names (e.g. ["auction_type_coded", ...]).
    response_cols : list[str]
        Response variable column names.
    output_dir : str
        Base output directory (sensitivity/ subdirectory is created).
    experiment_id : str or int, optional
        Experiment identifier for labeling.
    factor_groups : dict, optional
        Maps factor names to lists of coded columns. Auto-detected if None.

    Returns
    -------
    dict
        All sensitivity results keyed by section name.
    """
    exp_label = f"Experiment {experiment_id}" if experiment_id else "Sensitivity Analysis"
    print("=" * 70)
    print(f"GLOBAL SENSITIVITY ANALYSIS -- {exp_label}")
    print("=" * 70)

    sens_dir = os.path.join(output_dir, "sensitivity")
    os.makedirs(sens_dir, exist_ok=True)

    # Verify coded columns exist
    missing = [c for c in coded_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing coded columns: {missing}")
        return {}

    # Auto-detect factor groups if not provided
    if factor_groups is None:
        factor_groups = _infer_factor_groups(coded_cols)
    print(f"\nFactor groups ({len(factor_groups)}):")
    for name, cols in factor_groups.items():
        print(f"  {name}: {cols}")

    # Filter to available, non-constant responses
    active_responses = []
    for r in response_cols:
        if r not in df.columns:
            print(f"  Skipping {r}: not in data")
            continue
        if df[r].nunique() < 2:
            print(f"  Skipping {r}: constant")
            continue
        active_responses.append(r)

    print(f"\nActive responses: {active_responses}")

    all_results = {}

    # A. Analytical Sobol (always works)
    all_results["analytical_sobol"] = section_analytical_sobol(
        df, coded_cols, active_responses, factor_groups, sens_dir
    )

    # B. RF importance
    all_results["rf_importance"] = section_rf_importance(
        df, coded_cols, active_responses, factor_groups, sens_dir
    )

    # C. SHAP
    all_results["shap"] = section_shap(
        df, coded_cols, active_responses, factor_groups, sens_dir
    )

    # D. Morris (requires SALib)
    all_results["morris"] = section_morris(
        df, coded_cols, active_responses, factor_groups, sens_dir
    )

    # E. MC Sobol (requires SALib)
    all_results["mc_sobol"] = section_mc_sobol(
        df, coded_cols, active_responses, factor_groups, sens_dir
    )

    # E2. GP-based Sobol with posterior UQ (Marrel et al. 2009)
    all_results["gp_sobol"] = section_gp_sobol(
        df, coded_cols, active_responses, factor_groups, sens_dir
    )

    # F. FAST (requires SALib)
    all_results["fast"] = section_fast(
        df, coded_cols, active_responses, factor_groups, sens_dir
    )

    # G. Borgonovo delta (requires SALib)
    all_results["delta"] = section_delta(
        df, coded_cols, active_responses, factor_groups, sens_dir
    )

    # H. Concordance
    all_results["concordance"] = section_concordance(
        all_results, active_responses, factor_groups, sens_dir
    )

    # Plot: analytical vs MC comparison
    _plot_mc_comparison(all_results, active_responses, factor_groups, sens_dir)

    # Plot: GP vs analytical comparison
    _plot_gp_sobol_vs_analytical(all_results, active_responses, factor_groups,
                                  sens_dir)

    # I. Summary
    all_results["summary"] = section_summary(
        all_results, active_responses, factor_groups, sens_dir,
        experiment_id=experiment_id,
    )

    # Save JSON
    json_path = os.path.join(sens_dir, "sensitivity_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_json_safe)
    print(f"\n{'=' * 70}")
    print(f"All results saved to {json_path}")
    print(f"Plots saved to {sens_dir}/")
    print(f"{'=' * 70}")

    return all_results


# ===================================================================
# Standalone CLI
# ===================================================================

_EXP_DEFAULTS = {
    "1": {
        "coded_cols": [
            "auction_type_coded", "alpha_coded", "gamma_coded",
            "reserve_price_coded", "init_coded", "exploration_coded",
            "asynchronous_coded", "n_bidders_coded",
            "info_feedback_coded", "decay_type_coded",
        ],
        "response_cols": [
            "avg_rev_last_1000", "time_to_converge",
            "no_sale_rate", "price_volatility", "winner_entropy",
        ],
        "data_path": "results/exp1/data.csv",
        "output_dir": "results/exp1",
        "time_norm_col": "episodes",
    },
    "2": {
        "coded_cols": [
            "auction_type_coded", "eta_linear_coded", "eta_quadratic_coded",
            "n_bidders_coded", "state_info_coded",
        ],
        "response_cols": [
            "avg_rev_last_1000", "time_to_converge", "no_sale_rate",
            "price_volatility", "winner_entropy",
            "btv_median", "winners_curse_freq",
            "bid_dispersion", "signal_slope_ratio",
        ],
        "data_path": "results/exp2/data.csv",
        "output_dir": "results/exp2",
        "time_norm_col": "episodes",
    },
    "3a": {
        "coded_cols": [
            "auction_type_coded", "n_bidders_coded",
            "reserve_price_coded", "eta_linear_coded", "eta_quadratic_coded",
            "exploration_intensity_coded", "context_richness_coded",
            "lam_coded", "memory_decay_coded",
        ],
        "response_cols": [
            "avg_rev_last_1000", "time_to_converge",
            "no_sale_rate", "price_volatility", "winner_entropy",
        ],
        "data_path": "results/exp3a/data.csv",
        "output_dir": "results/exp3a",
        "time_norm_col": "max_rounds",
    },
    "3b": {
        "coded_cols": [
            "auction_type_coded", "n_bidders_coded",
            "reserve_price_coded", "eta_linear_coded", "eta_quadratic_coded",
            "exploration_intensity_coded", "context_richness_coded",
        ],
        "response_cols": [
            "avg_rev_last_1000", "time_to_converge",
            "no_sale_rate", "price_volatility", "winner_entropy",
        ],
        "data_path": "results/exp3b/data.csv",
        "output_dir": "results/exp3b",
        "time_norm_col": "max_rounds",
    },
    "4a": {
        "coded_cols": [
            "auction_type_coded", "objective_coded", "n_bidders_coded",
            "budget_multiplier_coded", "reserve_price_coded", "sigma_coded",
        ],
        "response_cols": [
            "mean_platform_revenue", "mean_liquid_welfare", "mean_effective_poa",
            "mean_budget_utilization", "mean_bid_to_value",
            "mean_allocative_efficiency", "mean_dual_cv", "mean_no_sale_rate",
            "mean_winner_entropy", "warm_start_benefit",
            "inter_episode_volatility", "bid_suppression_ratio",
            "cross_episode_drift",
            "mean_lp_offline_welfare", "mean_effective_poa_lp",
            "mean_rev_all",
        ],
        "data_path": "results/exp4a/data.csv",
        "output_dir": "results/exp4a",
        "time_norm_col": None,
    },
    "4b": {
        "coded_cols": [
            "auction_type_coded", "aggressiveness_coded", "n_bidders_coded",
            "budget_multiplier_coded", "reserve_price_coded", "sigma_coded",
        ],
        "response_cols": [
            "mean_platform_revenue", "mean_liquid_welfare", "mean_effective_poa",
            "mean_budget_utilization", "mean_bid_to_value",
            "mean_allocative_efficiency", "mean_dual_cv", "mean_no_sale_rate",
            "mean_winner_entropy", "warm_start_benefit",
            "inter_episode_volatility", "bid_suppression_ratio",
            "cross_episode_drift",
            "mean_lp_offline_welfare", "mean_effective_poa_lp",
            "mean_rev_all",
        ],
        "data_path": "results/exp4b/data.csv",
        "output_dir": "results/exp4b",
        "time_norm_col": None,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Global sensitivity analysis (Sobol' + multi-method GSA)")
    parser.add_argument("--exp", type=str, required=True,
                        choices=["1", "2", "3a", "3b", "4a", "4b"],
                        help="Experiment identifier")
    args = parser.parse_args()

    cfg = _EXP_DEFAULTS[args.exp]
    df = pd.read_csv(cfg["data_path"])
    print(f"\nLoaded {len(df)} rows from {cfg['data_path']}")

    # Normalize time_to_converge if applicable
    norm_col = cfg["time_norm_col"]
    if norm_col and "time_to_converge" in df.columns and norm_col in df.columns:
        df["time_to_converge"] = df["time_to_converge"] / df[norm_col]

    run_sensitivity_analysis(
        df,
        coded_cols=cfg["coded_cols"],
        response_cols=cfg["response_cols"],
        output_dir=cfg["output_dir"],
        experiment_id=args.exp,
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
    warnings.filterwarnings("ignore", category=UserWarning, module="shap")
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    main()
