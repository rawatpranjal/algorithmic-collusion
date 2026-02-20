#!/usr/bin/env python3
"""
Robust Statistical Analysis for Experiment 4.

Extends the baseline factorial OLS with:
  A. HC3 robust standard errors
  B. Multiple testing corrections (Holm-Bonferroni, Benjamini-Hochberg)
  C. Lack-of-fit test
  D. PRESS / predicted R²
  E. Box-Cox transformation
  F. Mixed-effects model (replicate as random intercept)
  G. Quantile regression (τ = 0.10 … 0.90)
  H. LASSO with cross-validation
  I. LightGBM nonparametric R² upper bound
  J. Wild bootstrap p-values
  K. Power analysis
  L. Response dependency analysis
  M. Summary comparison table

Usage:
    PYTHONPATH=src python3 src/estimation/robust_analysis4.py

Reads:  results/exp4/data.csv, results/exp4/design_info.json
Writes: results/exp4/robust/robust_results.json
        results/exp4/robust/*.png
        results/exp4/robust/robust_summary.txt
"""

import itertools
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tabulate import tabulate

# ---------------------------------------------------------------------------
# Constants (mirrored from est1.py)
# ---------------------------------------------------------------------------

CODED_COLS = [
    "algorithm_coded",
    "auction_type_coded",
    "n_bidders_coded",
    "budget_tightness_coded",
    "eta_coded",
    "aggressiveness_coded",
    "update_frequency_coded",
    "initial_multiplier_coded",
    "reserve_price_coded",
]

RESPONSE_COLS = [
    "avg_rev_last_1000",
    "time_to_converge",
    "avg_regret_of_seller",
    "no_sale_rate",
    "price_volatility",
    "winner_entropy",
    "budget_utilization",
    "spend_volatility",
    "budget_violation_rate",
    "effective_bid_shading",
    "multiplier_convergence_time",
    "multiplier_final_mean",
    "multiplier_final_std",
]

DATA_PATH = "results/exp4/data.csv"
DESIGN_INFO_PATH = "results/exp4/design_info.json"
OUTPUT_DIR = "results/exp4/robust"


def _clean(name):
    """Remove _coded suffix for display."""
    return name.replace("_coded", "")


def _build_formula(response, coded_cols):
    """Y ~ X1 + X2 + … + X1:X2 + X1:X3 + …"""
    main = coded_cols
    interactions = [f"{a}:{b}" for a, b in itertools.combinations(coded_cols, 2)]
    return f"{response} ~ {' + '.join(main + interactions)}"


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


# ===================================================================
# A. HC3 Robust Standard Errors
# ===================================================================

def section_hc3(df, coded_cols, response_cols):
    """Compare OLS vs HC3 robust standard errors for every term."""
    print("\n" + "=" * 60)
    print("A. HC3 ROBUST STANDARD ERRORS")
    print("=" * 60)

    results = {}
    for resp in response_cols:
        formula = _build_formula(resp, coded_cols)
        ols = smf.ols(formula, data=df).fit()
        hc3 = smf.ols(formula, data=df).fit(cov_type="HC3")

        terms = {}
        flipped = []
        for name in ols.params.index:
            if name == "Intercept":
                continue
            label = _clean(name)
            ols_sig = ols.pvalues[name] < 0.05
            hc3_sig = hc3.pvalues[name] < 0.05
            changed = ols_sig != hc3_sig
            terms[label] = {
                "ols_se": float(ols.bse[name]),
                "hc3_se": float(hc3.bse[name]),
                "ols_p": float(ols.pvalues[name]),
                "hc3_p": float(hc3.pvalues[name]),
                "changed_significance": changed,
            }
            if changed:
                flipped.append(label)

        results[resp] = {"terms": terms, "n_flipped": len(flipped), "flipped": flipped}

        if flipped:
            print(f"\n  {resp}: {len(flipped)} terms flipped significance:")
            for f in flipped:
                t = terms[f]
                print(f"    {f:40s}  OLS p={t['ols_p']:.4f}  HC3 p={t['hc3_p']:.4f}")
        else:
            print(f"\n  {resp}: No terms changed significance under HC3.")

    return results


# ===================================================================
# B. Multiple Testing Corrections
# ===================================================================

def section_multiplicity(df, coded_cols, response_cols):
    """Apply Holm-Bonferroni (FWER) and Benjamini-Hochberg (FDR) corrections."""
    print("\n" + "=" * 60)
    print("B. MULTIPLE TESTING CORRECTIONS")
    print("=" * 60)

    # Collect all p-values across responses
    all_pvals = []
    all_labels = []
    per_response = {}

    for resp in response_cols:
        formula = _build_formula(resp, coded_cols)
        model = smf.ols(formula, data=df).fit()
        resp_pvals = []
        resp_labels = []
        for name in model.pvalues.index:
            if name == "Intercept":
                continue
            all_pvals.append(float(model.pvalues[name]))
            all_labels.append(f"{resp}|{_clean(name)}")
            resp_pvals.append(float(model.pvalues[name]))
            resp_labels.append(_clean(name))
        per_response[resp] = {"pvals": resp_pvals, "labels": resp_labels}

    pvals = np.array(all_pvals)

    # Holm-Bonferroni (FWER control)
    holm_reject, holm_p, _, _ = multipletests(pvals, alpha=0.05, method="holm")
    # Benjamini-Hochberg (FDR control)
    bh_reject, bh_p, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

    n_raw = int(np.sum(pvals < 0.05))
    n_holm = int(np.sum(holm_reject))
    n_bh = int(np.sum(bh_reject))

    print(f"\n  Total tests: {len(pvals)}")
    print(f"  Significant at raw α=0.05:        {n_raw}")
    print(f"  Significant after Holm-Bonferroni: {n_holm}")
    print(f"  Significant after Benjamini-Hochberg: {n_bh}")

    # Build per-test results
    tests = []
    for i, label in enumerate(all_labels):
        tests.append({
            "label": label,
            "raw_p": float(pvals[i]),
            "holm_p": float(holm_p[i]),
            "bh_p": float(bh_p[i]),
            "raw_sig": bool(pvals[i] < 0.05),
            "holm_sig": bool(holm_reject[i]),
            "bh_sig": bool(bh_reject[i]),
        })

    # Print survivors
    print("\n  Effects surviving Holm-Bonferroni:")
    for t in sorted(tests, key=lambda x: x["raw_p"]):
        if t["holm_sig"]:
            print(f"    {t['label']:55s}  raw={t['raw_p']:.4e}  holm={t['holm_p']:.4e}")

    return {
        "n_tests": len(pvals),
        "n_raw_sig": n_raw,
        "n_holm_sig": n_holm,
        "n_bh_sig": n_bh,
        "tests": tests,
    }


# ===================================================================
# C. Lack-of-Fit Test
# ===================================================================

def section_lack_of_fit(df, coded_cols, response_cols):
    """Decompose residual SS into pure error and lack-of-fit components."""
    print("\n" + "=" * 60)
    print("C. LACK-OF-FIT TEST")
    print("=" * 60)

    results = {}
    for resp in response_cols:
        formula = _build_formula(resp, coded_cols)
        model = smf.ols(formula, data=df).fit()

        # Build cell identifier from coded columns
        cell_key = df[coded_cols].astype(str).agg("|".join, axis=1)
        df_work = pd.DataFrame({"y": df[resp].values, "fitted": model.fittedvalues.values,
                                "resid": model.resid.values, "cell": cell_key.values})

        # Pure error: within-cell variation
        ss_pe = 0.0
        df_pe = 0
        ss_lof = 0.0
        for _, grp in df_work.groupby("cell"):
            n_i = len(grp)
            if n_i < 2:
                continue
            cell_mean = grp["y"].mean()
            ss_pe += float(((grp["y"] - cell_mean) ** 2).sum())
            df_pe += n_i - 1
            # LOF: cell mean vs model prediction (constant within cell for balanced design)
            fitted_mean = grp["fitted"].mean()
            ss_lof += n_i * (cell_mean - fitted_mean) ** 2

        ss_resid = float((model.resid ** 2).sum())
        df_model_terms = model.df_model  # number of model parameters (excl intercept)
        n_cells = cell_key.nunique()
        df_lof = n_cells - int(df_model_terms) - 1  # cells - model params

        if df_pe > 0 and df_lof > 0:
            ms_lof = ss_lof / df_lof
            ms_pe = ss_pe / df_pe
            f_lof = ms_lof / ms_pe if ms_pe > 0 else np.nan
            p_lof = 1 - stats.f.cdf(f_lof, df_lof, df_pe) if np.isfinite(f_lof) else np.nan
        else:
            f_lof = np.nan
            p_lof = np.nan

        results[resp] = {
            "ss_residual": ss_resid,
            "ss_lack_of_fit": float(ss_lof),
            "ss_pure_error": float(ss_pe),
            "df_lack_of_fit": int(df_lof),
            "df_pure_error": int(df_pe),
            "f_lack_of_fit": float(f_lof) if np.isfinite(f_lof) else None,
            "p_lack_of_fit": float(p_lof) if np.isfinite(p_lof) else None,
        }

        sig = ""
        if np.isfinite(p_lof) and p_lof < 0.05:
            sig = " *** SIGNIFICANT — linear model may be inadequate"
        print(f"\n  {resp}:")
        print(f"    SS_LOF={ss_lof:.6f}  SS_PE={ss_pe:.6f}  "
              f"F({df_lof},{df_pe})={f_lof:.3f}  p={p_lof:.4f}{sig}")

    return results


# ===================================================================
# D. PRESS / Predicted R²
# ===================================================================

def section_press(df, coded_cols, response_cols):
    """Compute PRESS statistic and predicted R² via hat matrix."""
    print("\n" + "=" * 60)
    print("D. PRESS / PREDICTED R²")
    print("=" * 60)

    results = {}
    for resp in response_cols:
        formula = _build_formula(resp, coded_cols)
        model = smf.ols(formula, data=df).fit()

        influence = model.get_influence()
        h = influence.hat_matrix_diag
        resid = model.resid.values

        # PRESS residuals: e_i / (1 - h_ii)
        press_resid = resid / (1 - h)
        press = float(np.sum(press_resid ** 2))
        ss_total = float(np.sum((df[resp] - df[resp].mean()) ** 2))
        pred_r2 = 1 - press / ss_total if ss_total > 0 else np.nan
        gap = model.rsquared - pred_r2

        flag = ""
        if gap > 0.10:
            flag = " ⚠ GAP > 0.10 (possible overfitting)"

        results[resp] = {
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "press": press,
            "predicted_r_squared": float(pred_r2),
            "gap_r2_pred_r2": float(gap),
        }

        print(f"\n  {resp}:")
        print(f"    R²={model.rsquared:.4f}  Adj-R²={model.rsquared_adj:.4f}  "
              f"Pred-R²={pred_r2:.4f}  Gap={gap:.4f}{flag}")

    return results


# ===================================================================
# E. Box-Cox Transformation
# ===================================================================

def section_boxcox(df, coded_cols, response_cols):
    """Test Box-Cox transformation for positive-valued responses."""
    print("\n" + "=" * 60)
    print("E. BOX-COX TRANSFORMATION")
    print("=" * 60)

    results = {}
    for resp in response_cols:
        y = df[resp].values
        if np.any(y <= 0):
            print(f"\n  {resp}: Contains non-positive values, skipping Box-Cox.")
            results[resp] = {"skipped": True, "reason": "non-positive values"}
            continue

        formula = _build_formula(resp, coded_cols)
        ols_orig = smf.ols(formula, data=df).fit()

        # Estimate optimal lambda
        y_transformed, lam = stats.boxcox(y)

        # Fit model on transformed response
        df_bc = df.copy()
        df_bc[f"{resp}_bc"] = y_transformed
        ols_bc = smf.ols(_build_formula(f"{resp}_bc", coded_cols), data=df_bc).fit()

        # Count significant effects
        n_sig_orig = int(np.sum(
            [ols_orig.pvalues[n] < 0.05 for n in ols_orig.pvalues.index if n != "Intercept"]
        ))
        n_sig_bc = int(np.sum(
            [ols_bc.pvalues[n] < 0.05 for n in ols_bc.pvalues.index if n != "Intercept"]
        ))

        results[resp] = {
            "skipped": False,
            "optimal_lambda": float(lam),
            "r2_original": float(ols_orig.rsquared),
            "r2_boxcox": float(ols_bc.rsquared),
            "n_sig_original": n_sig_orig,
            "n_sig_boxcox": n_sig_bc,
        }

        lam_interp = "log" if abs(lam) < 0.05 else f"λ={lam:.3f}"
        print(f"\n  {resp}: optimal λ={lam:.3f} ({lam_interp})")
        print(f"    Original R²={ols_orig.rsquared:.4f} ({n_sig_orig} sig effects)  →  "
              f"Box-Cox R²={ols_bc.rsquared:.4f} ({n_sig_bc} sig effects)")

    return results


# ===================================================================
# F. Mixed-Effects Model
# ===================================================================

def section_mixed_effects(df, coded_cols, response_cols):
    """Fit mixed-effects model with replicate as random intercept."""
    print("\n" + "=" * 60)
    print("F. MIXED-EFFECTS MODEL")
    print("=" * 60)

    # Build cell identifier
    cell_key = df[coded_cols].astype(str).agg("|".join, axis=1)
    df_work = df.copy()
    df_work["cell_id_str"] = cell_key

    results = {}
    for resp in response_cols:
        formula = _build_formula(resp, coded_cols)
        ols_model = smf.ols(formula, data=df_work).fit()

        # Build explicit design matrix for MixedLM
        main = coded_cols
        interactions = [f"{a}:{b}" for a, b in itertools.combinations(coded_cols, 2)]

        X = df_work[coded_cols].copy()
        for a, b in itertools.combinations(coded_cols, 2):
            X[f"{a}:{b}"] = df_work[a] * df_work[b]
        X = sm.add_constant(X)

        try:
            mixed = sm.MixedLM(
                df_work[resp], X, groups=df_work["cell_id_str"]
            ).fit(reml=True)

            sigma_u = float(np.sqrt(mixed.cov_re.iloc[0, 0])) if hasattr(mixed.cov_re, "iloc") else 0.0
            sigma_e = float(np.sqrt(mixed.scale))
            icc = sigma_u ** 2 / (sigma_u ** 2 + sigma_e ** 2) if (sigma_u ** 2 + sigma_e ** 2) > 0 else 0.0

            results[resp] = {
                "sigma_cell": sigma_u,
                "sigma_residual": sigma_e,
                "icc": float(icc),
                "converged": True,
            }

            print(f"\n  {resp}:")
            print(f"    σ_cell={sigma_u:.6f}  σ_resid={sigma_e:.6f}  ICC={icc:.4f}")
            if icc < 0.05:
                print(f"    → Random effects negligible (ICC < 0.05)")
            else:
                print(f"    → Non-negligible cell-level variation")
        except Exception as e:
            results[resp] = {"converged": False, "error": str(e)}
            print(f"\n  {resp}: MixedLM failed — {e}")

    return results


# ===================================================================
# G. Quantile Regression
# ===================================================================

def section_quantile_regression(df, coded_cols, response_cols, output_dir):
    """Fit quantile regressions at multiple quantiles; plot coefficient paths."""
    print("\n" + "=" * 60)
    print("G. QUANTILE REGRESSION")
    print("=" * 60)

    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    # Focus on revenue and regret
    qr_responses = [r for r in ["avg_rev_last_1000", "avg_regret_of_seller"]
                    if r in response_cols]

    results = {}
    for resp in qr_responses:
        formula = _build_formula(resp, coded_cols)

        coefs_by_q = {}
        for q in quantiles:
            qr = smf.quantreg(formula, data=df).fit(q=q)
            coefs_by_q[q] = {}
            for name in qr.params.index:
                if name == "Intercept":
                    continue
                coefs_by_q[q][_clean(name)] = {
                    "coef": float(qr.params[name]),
                    "p_value": float(qr.pvalues[name]),
                }

        results[resp] = {"quantiles": {str(q): coefs_by_q[q] for q in quantiles}}

        # Select top 5 effects by |OLS coefficient|
        ols = smf.ols(formula, data=df).fit()
        abs_coefs = ols.params.drop("Intercept", errors="ignore").abs().sort_values(ascending=False)
        top5 = [_clean(n) for n in abs_coefs.head(5).index]

        # Plot coefficient paths
        fig, axes = plt.subplots(1, len(top5), figsize=(4 * len(top5), 3.5), sharey=False)
        if len(top5) == 1:
            axes = [axes]

        for ax, term in zip(axes, top5):
            coef_vals = [coefs_by_q[q][term]["coef"] for q in quantiles]
            pvals = [coefs_by_q[q][term]["p_value"] for q in quantiles]
            colors = ["#e74c3c" if p < 0.05 else "#95a5a6" for p in pvals]
            ax.plot(quantiles, coef_vals, "o-", color="#2980b9", markersize=6)
            for qi, (q, c, col) in enumerate(zip(quantiles, coef_vals, colors)):
                ax.scatter([q], [c], color=col, s=40, zorder=5)
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Quantile (τ)")
            ax.set_ylabel("Coefficient")
            ax.set_title(term, fontsize=9)
            ax.set_xticks(quantiles)

        fig.suptitle(f"Quantile Regression Coefficients: {_clean(resp)}", fontsize=11)
        fig.tight_layout()
        path = os.path.join(output_dir, f"quantile_coefs_{resp}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  {resp}: Plot saved to {path}")

        # Print key finding: auction_type:exploration at tails
        key = "auction_type:exploration"
        if key in coefs_by_q.get(0.10, {}):
            c10 = coefs_by_q[0.10][key]["coef"]
            c90 = coefs_by_q[0.90][key]["coef"]
            print(f"    auction_type×exploration: τ=0.10 coef={c10:.4f}, τ=0.90 coef={c90:.4f}")

    return results


# ===================================================================
# H. LASSO with Cross-Validation
# ===================================================================

def section_lasso(df, coded_cols, response_cols, output_dir):
    """LASSO with 5-fold CV; report surviving terms."""
    print("\n" + "=" * 60)
    print("H. LASSO WITH CROSS-VALIDATION")
    print("=" * 60)

    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    # Build design matrix
    X = df[coded_cols].copy()
    for a, b in itertools.combinations(coded_cols, 2):
        X[f"{_clean(a)}:{_clean(b)}"] = df[a] * df[b]
    feature_names = list(X.columns)

    results = {}
    for resp in response_cols:
        y = df[resp].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
        lasso.fit(X_scaled, y)

        coefs = lasso.coef_
        surviving = [(feature_names[i], float(coefs[i]))
                     for i in range(len(coefs)) if abs(coefs[i]) > 1e-8]
        dropped = [feature_names[i] for i in range(len(coefs)) if abs(coefs[i]) <= 1e-8]

        # Check heredity: flag interactions where a parent was dropped
        heredity_violations = []
        for name, coef in surviving:
            if ":" in name:
                parents = name.split(":")
                for p in parents:
                    # Match against coded col names
                    p_coded = p + "_coded" if p + "_coded" in coded_cols else p
                    if p_coded in dropped or p in dropped:
                        heredity_violations.append(name)
                        break

        # OLS R² for comparison
        formula = _build_formula(resp, coded_cols)
        ols = smf.ols(formula, data=df).fit()

        results[resp] = {
            "alpha_cv": float(lasso.alpha_),
            "n_surviving": len(surviving),
            "n_dropped": len(dropped),
            "surviving_terms": {name: coef for name, coef in surviving},
            "dropped_terms": dropped,
            "heredity_violations": heredity_violations,
            "lasso_r2": float(lasso.score(X_scaled, y)),
            "ols_r2": float(ols.rsquared),
        }

        print(f"\n  {resp}: α_CV={lasso.alpha_:.6f}")
        print(f"    Surviving: {len(surviving)}/{len(feature_names)}  "
              f"LASSO R²={lasso.score(X_scaled, y):.4f}  OLS R²={ols.rsquared:.4f}")
        if heredity_violations:
            print(f"    Heredity violations: {heredity_violations}")

    return results


# ===================================================================
# I. LightGBM Nonparametric R² Upper Bound
# ===================================================================

def section_lightgbm(df, coded_cols, response_cols):
    """LightGBM 5-fold CV R² as upper bound for linear model adequacy."""
    print("\n" + "=" * 60)
    print("I. LIGHTGBM NONPARAMETRIC R² UPPER BOUND")
    print("=" * 60)

    import lightgbm as lgb
    from sklearn.model_selection import cross_val_score

    X = df[coded_cols]

    results = {}
    for resp in response_cols:
        y = df[resp].values

        # OLS R²
        formula = _build_formula(resp, coded_cols)
        ols = smf.ols(formula, data=df).fit()

        # LightGBM CV
        model = lgb.LGBMRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            num_leaves=16, random_state=42, verbose=-1,
        )
        cv_scores = cross_val_score(model, X.values, y, cv=5, scoring="r2")
        lgbm_r2 = float(np.mean(cv_scores))

        gap = lgbm_r2 - ols.rsquared
        flag = ""
        if gap > 0.05:
            flag = " ⚠ Nonlinear signal detected"

        results[resp] = {
            "ols_r2": float(ols.rsquared),
            "lgbm_cv_r2": lgbm_r2,
            "lgbm_cv_std": float(np.std(cv_scores)),
            "gap": float(gap),
        }

        print(f"\n  {resp}:")
        print(f"    OLS R²={ols.rsquared:.4f}  LightGBM CV R²={lgbm_r2:.4f} "
              f"(±{np.std(cv_scores):.4f})  Gap={gap:.4f}{flag}")

    # Plot comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(response_cols))
    w = 0.35
    ols_vals = [results[r]["ols_r2"] for r in response_cols]
    lgbm_vals = [results[r]["lgbm_cv_r2"] for r in response_cols]
    ax.bar(x - w / 2, ols_vals, w, label="OLS (with interactions)", color="#2980b9")
    ax.bar(x + w / 2, lgbm_vals, w, label="LightGBM 5-fold CV", color="#27ae60")
    ax.set_xticks(x)
    ax.set_xticklabels([_clean(r) for r in response_cols], fontsize=8, rotation=15)
    ax.set_ylabel("R²")
    ax.set_title("Linear vs Nonparametric Model Adequacy")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "lgbm_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved to {path}")

    return results


# ===================================================================
# J. Wild Bootstrap P-values
# ===================================================================

def section_wild_bootstrap(df, coded_cols, response_cols, n_boot=1000):
    """Rademacher wild bootstrap for HC3-based inference on top effects."""
    print("\n" + "=" * 60)
    print("J. WILD BOOTSTRAP P-VALUES")
    print("=" * 60)

    rng = np.random.RandomState(42)
    results = {}

    for resp in response_cols:
        formula = _build_formula(resp, coded_cols)
        model = smf.ols(formula, data=df).fit(cov_type="HC3")

        # Get top 10 effects by |t|
        tvals = model.tvalues.drop("Intercept", errors="ignore")
        top10_names = tvals.abs().sort_values(ascending=False).head(10).index.tolist()

        # Restricted residuals (under H0: beta_j = 0 for each term tested)
        X = model.model.exog
        y = model.model.endog
        n = len(y)

        boot_results = {}
        for term in top10_names:
            observed_t = float(model.tvalues[term])
            # Full model t-stat; bootstrap under full model
            fitted = model.fittedvalues.values
            resid = model.resid.values

            boot_ts = np.zeros(n_boot)
            col_idx = list(model.params.index).index(term)
            for b in range(n_boot):
                # Rademacher weights
                w = rng.choice([-1, 1], size=n)
                y_boot = fitted + resid * w
                try:
                    boot_model = sm.OLS(y_boot, X).fit(cov_type="HC3")
                    boot_ts[b] = boot_model.tvalues[col_idx]
                except Exception:
                    boot_ts[b] = observed_t  # conservative: don't count as extreme

            # Recenter: unrestricted bootstrap centers around observed_t
            # Under H0, test statistic ~ 0, so use |t*_boot - t_obs| >= |t_obs|
            boot_p = float(np.mean(np.abs(boot_ts - observed_t) >= np.abs(observed_t)))

            boot_results[_clean(term)] = {
                "observed_t": observed_t,
                "asymptotic_p": float(model.pvalues[term]),
                "bootstrap_p": boot_p,
                "boot_ci_025": float(np.percentile(boot_ts, 2.5)),
                "boot_ci_975": float(np.percentile(boot_ts, 97.5)),
            }

        results[resp] = boot_results
        print(f"\n  {resp}: Top 10 effects (asymptotic p vs bootstrap p):")
        for term, info in sorted(boot_results.items(), key=lambda x: abs(x[1]["observed_t"]),
                                  reverse=True):
            print(f"    {term:40s}  t={info['observed_t']:+.3f}  "
                  f"p_asym={info['asymptotic_p']:.4f}  p_boot={info['bootstrap_p']:.4f}")

    return results


# ===================================================================
# K. Power Analysis
# ===================================================================

def section_power(df, coded_cols, response_cols):
    """Compute minimum detectable effect sizes at 80% power."""
    print("\n" + "=" * 60)
    print("K. POWER ANALYSIS")
    print("=" * 60)

    results = {}
    n = len(df)

    for resp in response_cols:
        formula = _build_formula(resp, coded_cols)
        model = smf.ols(formula, data=df).fit()

        sigma = float(np.sqrt(model.mse_resid))
        df_resid = int(model.df_resid)

        # For coded ±1 design, SE of a main effect = sigma / sqrt(n)
        # For interaction, same if balanced
        se_main = sigma / np.sqrt(n)

        # t critical for two-sided alpha=0.05
        t_crit = stats.t.ppf(0.975, df_resid)

        # Minimum detectable effect at 80% power (approx: need |beta| > (t_crit + t_80%) * SE)
        # t_80% = z_{0.80} ≈ 0.842
        t_power = stats.norm.ppf(0.80)
        mde_main = (t_crit + t_power) * se_main
        mde_interaction = mde_main  # same SE for balanced 2^k

        # As fraction of response mean and std
        resp_mean = float(df[resp].mean())
        resp_std = float(df[resp].std())

        results[resp] = {
            "sigma_error": sigma,
            "se_main_effect": float(se_main),
            "mde_main_effect": float(mde_main),
            "mde_interaction": float(mde_interaction),
            "mde_as_pct_mean": float(mde_main / abs(resp_mean) * 100) if resp_mean != 0 else None,
            "mde_as_pct_std": float(mde_main / resp_std * 100) if resp_std > 0 else None,
            "n": n,
            "df_resid": df_resid,
        }

        print(f"\n  {resp}:")
        print(f"    σ_error={sigma:.6f}  SE_main={se_main:.6f}")
        print(f"    MDE (80% power): main={mde_main:.6f}  "
              f"interaction={mde_interaction:.6f}")
        if resp_mean != 0:
            print(f"    MDE as % of mean response: {mde_main / abs(resp_mean) * 100:.1f}%")

    # Plot detectable effect sizes
    fig, ax = plt.subplots(figsize=(8, 4))
    resp_labels = [_clean(r) for r in response_cols]
    mdes = [results[r]["mde_main_effect"] for r in response_cols]
    ax.barh(resp_labels, mdes, color="#e67e22")
    ax.set_xlabel("Minimum Detectable Effect (80% power, α=0.05)")
    ax.set_title("Power Analysis: Detectable Effect Sizes")
    for i, v in enumerate(mdes):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "power_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved to {path}")

    return results


# ===================================================================
# L. Response Dependency Analysis
# ===================================================================

def section_response_deps(df, response_cols, output_dir):
    """Correlation matrix and near-algebraic dependency check."""
    print("\n" + "=" * 60)
    print("L. RESPONSE DEPENDENCY ANALYSIS")
    print("=" * 60)

    avail = [r for r in response_cols if r in df.columns]
    corr = df[avail].corr()

    print("\n  Correlation matrix:")
    print(tabulate(corr, headers="keys", tablefmt="github", floatfmt=".3f"))

    # Flag high correlations
    pairs = []
    for i, r1 in enumerate(avail):
        for j, r2 in enumerate(avail):
            if i < j:
                r = corr.loc[r1, r2]
                if abs(r) > 0.80:
                    pairs.append((r1, r2, float(r)))
                    print(f"\n  ⚠ High correlation: {_clean(r1)} ↔ {_clean(r2)}: r={r:.3f}")

    # Check winner_entropy vs n_bidders (known deterministic mapping)
    if "winner_entropy" in df.columns and "n_bidders_coded" in df.columns:
        r2_ent = df[["winner_entropy", "n_bidders_coded"]].corr().iloc[0, 1] ** 2
        print(f"\n  winner_entropy ~ n_bidders: R²={r2_ent:.4f}")
        if r2_ent > 0.95:
            print("    → Near-deterministic mapping; winner_entropy largely redundant with n_bidders")

    # Exp4-specific sanity checks: budget constraint satisfaction
    if "budget_violation_rate" in df.columns:
        max_viol = df["budget_violation_rate"].max()
        print(f"\n  Sanity check — budget_violation_rate max={max_viol:.6f} "
              f"({'PASS' if max_viol < 1e-6 else 'FAIL — violations detected!'})")
    if "budget_utilization" in df.columns:
        util_range = (df["budget_utilization"].min(), df["budget_utilization"].max())
        ok = (0.0 <= util_range[0]) and (util_range[1] <= 1.0 + 1e-6)
        print(f"  Sanity check — budget_utilization ∈ [{util_range[0]:.3f}, {util_range[1]:.3f}] "
              f"({'PASS' if ok else 'FAIL — out of [0,1]!'})")
    if "effective_bid_shading" in df.columns:
        shade_range = (df["effective_bid_shading"].min(), df["effective_bid_shading"].max())
        ok = (shade_range[0] >= -0.01) and (shade_range[1] <= 1.01)
        print(f"  Sanity check — effective_bid_shading ∈ [{shade_range[0]:.3f}, {shade_range[1]:.3f}] "
              f"({'PASS' if ok else 'FAIL — out of [0,1]!'})")

    # Heatmap
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(avail)))
    ax.set_yticks(range(len(avail)))
    ax.set_xticklabels([_clean(r) for r in avail], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([_clean(r) for r in avail], fontsize=8)
    for i in range(len(avail)):
        for j in range(len(avail)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Response Correlation Matrix")
    fig.tight_layout()
    path = os.path.join(output_dir, "response_correlations.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved to {path}")

    return {
        "correlation_matrix": {r1: {r2: float(corr.loc[r1, r2]) for r2 in avail} for r1 in avail},
        "high_correlation_pairs": [{"r1": p[0], "r2": p[1], "r": p[2]} for p in pairs],
    }


# ===================================================================
# M. Summary & Comparison Table
# ===================================================================

def section_summary(all_results, df, coded_cols, response_cols, output_dir):
    """Master comparison table across all methods."""
    print("\n" + "=" * 60)
    print("M. SUMMARY & COMPARISON TABLE")
    print("=" * 60)

    rows = []
    for resp in response_cols:
        row = {"response": _clean(resp)}

        # R² metrics
        if "press" in all_results and resp in all_results["press"]:
            pr = all_results["press"][resp]
            row["R²"] = f"{pr['r_squared']:.4f}"
            row["Pred-R²"] = f"{pr['predicted_r_squared']:.4f}"
        if "lightgbm" in all_results and resp in all_results["lightgbm"]:
            row["LGBM R²"] = f"{all_results['lightgbm'][resp]['lgbm_cv_r2']:.4f}"
        if "lack_of_fit" in all_results and resp in all_results["lack_of_fit"]:
            lof = all_results["lack_of_fit"][resp]
            p = lof.get("p_lack_of_fit")
            row["LOF p"] = f"{p:.4f}" if p is not None else "N/A"
        if "boxcox" in all_results and resp in all_results["boxcox"]:
            bc = all_results["boxcox"][resp]
            if not bc.get("skipped"):
                row["Box-Cox λ"] = f"{bc['optimal_lambda']:.3f}"
            else:
                row["Box-Cox λ"] = "N/A"

        # Count significant effects by method
        if "hc3" in all_results and resp in all_results["hc3"]:
            hc3_terms = all_results["hc3"][resp]["terms"]
            n_ols_sig = sum(1 for t in hc3_terms.values() if t["ols_p"] < 0.05)
            n_hc3_sig = sum(1 for t in hc3_terms.values() if t["hc3_p"] < 0.05)
            row["# Sig OLS"] = str(n_ols_sig)
            row["# Sig HC3"] = str(n_hc3_sig)

        if "multiplicity" in all_results:
            tests = all_results["multiplicity"]["tests"]
            resp_tests = [t for t in tests if t["label"].startswith(resp + "|")]
            row["# Sig Holm"] = str(sum(1 for t in resp_tests if t["holm_sig"]))
            row["# Sig BH"] = str(sum(1 for t in resp_tests if t["bh_sig"]))

        if "lasso" in all_results and resp in all_results["lasso"]:
            row["# LASSO"] = str(all_results["lasso"][resp]["n_surviving"])

        rows.append(row)

    print("\n" + tabulate(rows, headers="keys", tablefmt="github"))

    # Write human-readable summary
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("ROBUST ANALYSIS SUMMARY — Experiment 4")
    summary_lines.append("=" * 70)
    summary_lines.append("")

    # Key finding: algorithm × budget_tightness (Exp4's primary interaction)
    summary_lines.append("KEY FINDING: algorithm × budget_tightness interaction")
    summary_lines.append("-" * 50)
    if "multiplicity" in all_results:
        tests = all_results["multiplicity"]["tests"]
        for t in tests:
            if "algorithm:budget_tightness" in t["label"]:
                status = "SURVIVES" if t["holm_sig"] else "does NOT survive"
                summary_lines.append(
                    f"  {t['label']}: raw p={t['raw_p']:.4e}, holm p={t['holm_p']:.4e} "
                    f"→ {status} Holm-Bonferroni"
                )
    summary_lines.append("")

    # Model adequacy
    summary_lines.append("MODEL ADEQUACY")
    summary_lines.append("-" * 50)
    if "press" in all_results:
        for resp in response_cols:
            pr = all_results["press"][resp]
            summary_lines.append(
                f"  {_clean(resp):30s}  R²={pr['r_squared']:.4f}  "
                f"Pred-R²={pr['predicted_r_squared']:.4f}  Gap={pr['gap_r2_pred_r2']:.4f}"
            )
    summary_lines.append("")

    if "lightgbm" in all_results:
        summary_lines.append("LINEAR vs NONPARAMETRIC")
        summary_lines.append("-" * 50)
        for resp in response_cols:
            lg = all_results["lightgbm"][resp]
            summary_lines.append(
                f"  {_clean(resp):30s}  OLS={lg['ols_r2']:.4f}  "
                f"LGBM={lg['lgbm_cv_r2']:.4f}  Gap={lg['gap']:.4f}"
            )
        summary_lines.append("")

    if "power" in all_results:
        summary_lines.append("POWER ANALYSIS (80% power, α=0.05)")
        summary_lines.append("-" * 50)
        for resp in response_cols:
            pw = all_results["power"][resp]
            pct = pw.get("mde_as_pct_mean")
            pct_str = f"({pct:.1f}% of mean)" if pct is not None else ""
            summary_lines.append(
                f"  {_clean(resp):30s}  MDE={pw['mde_main_effect']:.6f} {pct_str}"
            )

    summary_text = "\n".join(summary_lines)
    summary_path = os.path.join(output_dir, "robust_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"\n  Summary written to {summary_path}")

    return {"table": rows}


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 70)
    print("ROBUST STATISTICAL ANALYSIS — Experiment 4")
    print("=" * 70)

    # Load data
    df = pd.read_csv(DATA_PATH)
    print(f"\nLoaded {len(df)} rows from {DATA_PATH}")

    # Normalize time_to_converge
    if "time_to_converge" in df.columns and "max_rounds" in df.columns:
        df["time_to_converge"] = df["time_to_converge"] / df["max_rounds"]

    # Verify coded columns exist
    missing = [c for c in CODED_COLS if c not in df.columns]
    if missing:
        print(f"ERROR: Missing coded columns: {missing}")
        sys.exit(1)

    # Filter to available responses (skip no_sale_rate if all zeros)
    active_responses = []
    for r in RESPONSE_COLS:
        if r not in df.columns:
            print(f"  Skipping {r}: not in data")
            continue
        if df[r].nunique() < 2:
            print(f"  Skipping {r}: constant (all {df[r].iloc[0]})")
            continue
        active_responses.append(r)

    print(f"\nActive responses: {active_responses}")
    print(f"Coded factors: {CODED_COLS}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {}

    # Run all sections
    all_results["hc3"] = section_hc3(df, CODED_COLS, active_responses)
    all_results["multiplicity"] = section_multiplicity(df, CODED_COLS, active_responses)
    all_results["lack_of_fit"] = section_lack_of_fit(df, CODED_COLS, active_responses)
    all_results["press"] = section_press(df, CODED_COLS, active_responses)
    all_results["boxcox"] = section_boxcox(df, CODED_COLS, active_responses)
    all_results["mixed_effects"] = section_mixed_effects(df, CODED_COLS, active_responses)
    all_results["quantile_regression"] = section_quantile_regression(
        df, CODED_COLS, active_responses, OUTPUT_DIR
    )
    all_results["lasso"] = section_lasso(df, CODED_COLS, active_responses, OUTPUT_DIR)
    all_results["lightgbm"] = section_lightgbm(df, CODED_COLS, active_responses)
    all_results["wild_bootstrap"] = section_wild_bootstrap(
        df, CODED_COLS, active_responses, n_boot=1000
    )
    all_results["power"] = section_power(df, CODED_COLS, active_responses)
    all_results["response_deps"] = section_response_deps(df, active_responses, OUTPUT_DIR)
    all_results["summary"] = section_summary(
        all_results, df, CODED_COLS, active_responses, OUTPUT_DIR
    )

    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, "robust_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_json_safe)
    print(f"\n{'=' * 70}")
    print(f"All results saved to {json_path}")
    print(f"Plots saved to {OUTPUT_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
    main()
