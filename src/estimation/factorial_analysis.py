#!/usr/bin/env python3
"""
Factorial Analysis Engine for 2^k Experimental Designs.

For each response variable:
1. OLS regression with main effects + all 2-way interactions (coded -1/+1)
2. Type III ANOVA table (F-statistic, p-value)
3. Pareto chart of |t-statistic| sorted descending
4. Half-normal probability plot to identify active effects
5. Main effects plot (mean response at -1 vs +1)
6. Interaction plots for top significant 2-way interactions
7. Residual diagnostics (QQ plot + residuals vs fitted)
8. JSON export (coefficients, p-values, R², adj-R²)
"""

import hashlib
import itertools
import json
import os
import sys

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


def lenth_pse(model):
    """
    Compute Lenth's pseudo-standard error for effect screening.

    Returns (PSE, ME, SME) where:
      PSE = pseudo-standard error (robust scale estimate)
      ME  = margin of error (individual significance threshold)
      SME = simultaneous margin of error (Bonferroni-corrected threshold)
    """
    effects = model.params.drop("Intercept", errors="ignore").values
    abs_effects = np.abs(effects)
    k = len(abs_effects)

    # Initial scale estimate
    s0 = 1.5 * np.median(abs_effects)

    # Remove effects exceeding 2.5 * s0 and recompute
    trimmed = abs_effects[abs_effects < 2.5 * s0]
    if len(trimmed) == 0:
        trimmed = abs_effects  # fallback: use all
    pse = 1.5 * np.median(trimmed)

    # Degrees of freedom for t-distribution
    d = max(1, k // 3)

    # Margin of error (individual)
    me = stats.t.ppf(0.975, d) * pse

    # Simultaneous margin of error (Bonferroni-corrected)
    sme = stats.t.ppf(1 - 0.025 / k, d) * pse

    return pse, me, sme


def _build_formula(response, coded_cols):
    """Build OLS formula: Y ~ X1 + X2 + ... + X1:X2 + X1:X3 + ..."""
    main_terms = coded_cols
    interaction_terms = [
        f"{a}:{b}" for a, b in itertools.combinations(coded_cols, 2)
    ]
    rhs = " + ".join(main_terms + interaction_terms)
    return f"{response} ~ {rhs}"


def _run_ols(df, response, coded_cols):
    """Fit OLS model with main effects + 2-way interactions."""
    formula = _build_formula(response, coded_cols)
    model = smf.ols(formula, data=df).fit()
    return model


def _anova_table(model):
    """Compute Type III ANOVA table."""
    try:
        anova = sm.stats.anova_lm(model, typ=3)
    except Exception:
        anova = sm.stats.anova_lm(model, typ=2)
    return anova


def plot_pareto_chart(model, response, output_dir):
    """Bar chart of |t-statistic| sorted descending with significance line."""
    params = model.params.drop("Intercept", errors="ignore")
    tvalues = model.tvalues.drop("Intercept", errors="ignore")

    abs_t = tvalues.abs().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(abs_t) * 0.3)))
    colors = ["#e74c3c" if model.pvalues[name] < 0.05 else "#95a5a6"
              for name in abs_t.index]
    ax.barh(range(len(abs_t)), abs_t.values, color=colors)
    ax.set_yticks(range(len(abs_t)))
    ax.set_yticklabels([_clean_label(n) for n in abs_t.index], fontsize=10)
    ax.set_xlabel("|t-statistic|")
    ax.set_title(f"Pareto Chart: {_clean_label(response)}")

    # Significance line at t critical value (alpha=0.05, two-sided)
    t_crit = stats.t.ppf(0.975, model.df_resid)
    ax.axvline(t_crit, color="red", linestyle="--", alpha=0.7,
               label=f"t_crit = {t_crit:.2f}")
    ax.legend()

    fig.tight_layout()
    os.makedirs(os.path.join(output_dir, "pareto_charts"), exist_ok=True)
    fig.savefig(os.path.join(output_dir, "pareto_charts",
                             f"pareto_{response}.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_half_normal(model, response, output_dir):
    """Half-normal probability plot to identify active effects."""
    effects = model.params.drop("Intercept", errors="ignore")
    abs_effects = effects.abs().sort_values()

    n = len(abs_effects)
    expected = [stats.halfnorm.ppf((i + 0.5) / n) for i in range(n)]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(expected, abs_effects.values, s=30, zorder=3)

    for i, name in enumerate(abs_effects.index):
        if model.pvalues[name] < 0.05:
            ax.annotate(_clean_label(name),
                       (expected[i], abs_effects.iloc[i]),
                       fontsize=9, ha="left", va="bottom")

    # Lenth's PSE reference lines
    pse, me, sme = lenth_pse(model)
    ax.axhline(me, color="blue", linestyle="--", alpha=0.6,
               label=f"ME = {me:.4f}")
    ax.axhline(sme, color="red", linestyle="--", alpha=0.6,
               label=f"SME = {sme:.4f}")
    ax.legend(fontsize=9)

    ax.set_xlabel("Half-Normal Quantiles")
    ax.set_ylabel("|Effect|")
    ax.set_title(f"Half-Normal Plot: {_clean_label(response)}")

    fig.tight_layout()
    os.makedirs(os.path.join(output_dir, "normal_prob_plots"), exist_ok=True)
    fig.savefig(os.path.join(output_dir, "normal_prob_plots",
                             f"halfnormal_{response}.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_main_effects(df, coded_cols, response, output_dir, model=None):
    """Main effects plot: mean response at -1 vs +1 for significant factors."""
    # Filter to significant main effects only
    if model is not None:
        sig_cols = [c for c in coded_cols if c in model.pvalues and model.pvalues[c] < 0.05]
        if not sig_cols:
            # Fallback: show top 3 by |t|
            ranked = sorted(coded_cols, key=lambda c: abs(model.tvalues.get(c, 0)), reverse=True)
            sig_cols = ranked[:3]
        coded_cols = sig_cols

    n = len(coded_cols)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.atleast_2d(axes)

    for i, col in enumerate(coded_cols):
        ax = axes.flat[i]
        means = df.groupby(col)[response].mean()
        ax.plot([-1, 1], [means.get(-1, np.nan), means.get(1, np.nan)],
                "o-", color="#2980b9", markersize=8)
        ax.set_xticks([-1, 1])
        ax.set_xticklabels(["Low (-1)", "High (+1)"])
        ax.set_xlabel(_clean_label(col))
        ax.set_ylabel(f"Mean {_clean_label(response)}")
        ax.set_title(_clean_label(col))

    # Hide unused axes
    for j in range(n, nrows * ncols):
        axes.flat[j].set_visible(False)

    fig.suptitle(f"Main Effects: {_clean_label(response)}", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.join(output_dir, "main_effects"), exist_ok=True)
    fig.savefig(os.path.join(output_dir, "main_effects",
                             f"main_effects_{response}.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_interactions(df, model, coded_cols, response, output_dir, top_n=6):
    """Interaction plots for top significant 2-way interactions."""
    interactions = []
    for a, b in itertools.combinations(coded_cols, 2):
        term = f"{a}:{b}"
        if term in model.pvalues:
            interactions.append((term, a, b, abs(model.tvalues[term]),
                                model.pvalues[term]))

    interactions.sort(key=lambda x: x[3], reverse=True)
    top = interactions[:top_n]

    if not top:
        return

    ncols = min(3, len(top))
    nrows = int(np.ceil(len(top) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if len(top) == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    for i, (term, a, b, tval, pval) in enumerate(top):
        ax = axes.flat[i]
        for level_b in [-1, 1]:
            subset = df[df[b] == level_b]
            means = subset.groupby(a)[response].mean()
            label = f"{_clean_label(b)}={'High' if level_b == 1 else 'Low'}"
            ax.plot([-1, 1],
                    [means.get(-1, np.nan), means.get(1, np.nan)],
                    "o-", label=label, markersize=6)
        ax.set_xticks([-1, 1])
        ax.set_xticklabels(["Low", "High"])
        ax.set_xlabel(_clean_label(a))
        ax.set_ylabel(f"Mean {_clean_label(response)}")
        sig = "*" if pval < 0.05 else ""
        ax.set_title(f"{_clean_label(a)} x {_clean_label(b)}{sig}", fontsize=12)
        ax.legend(fontsize=10)

    for j in range(len(top), nrows * ncols):
        axes.flat[j].set_visible(False)

    fig.suptitle(f"Interaction Plots: {_clean_label(response)}", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.join(output_dir, "interaction_plots"), exist_ok=True)
    fig.savefig(os.path.join(output_dir, "interaction_plots",
                             f"interactions_{response}.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(model, response, output_dir):
    """QQ plot + residuals vs fitted values."""
    resid = model.resid
    fitted = model.fittedvalues

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Residuals vs fitted
    ax1.scatter(fitted, resid, s=10, alpha=0.5)
    ax1.axhline(0, color="red", linestyle="--")
    ax1.set_xlabel("Fitted Values")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs Fitted")

    # QQ plot
    sm.qqplot(resid, line="45", ax=ax2, markersize=3, alpha=0.5)
    ax2.set_title("Normal Q-Q Plot")

    fig.suptitle(f"Residual Diagnostics: {_clean_label(response)}", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.join(output_dir, "residuals"), exist_ok=True)
    fig.savefig(os.path.join(output_dir, "residuals",
                             f"residuals_{response}.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def _clean_label(name):
    """Remove _coded suffix for display."""
    return name.replace("_coded", "")


def _file_sha256(filepath):
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def check_data_freshness(output_dir, experiment_id=None):
    """Check whether data.csv is fresh by comparing source checksums.

    Loads data_manifest.json, recomputes checksums of source files, and
    prints a WARNING if any file has changed since data generation.

    Returns True if data is fresh (or cannot be verified), False if stale.
    """
    from pathlib import Path
    repo_root = Path(__file__).parent.parent.parent  # src/estimation -> repo root

    manifest_path = os.path.join(output_dir, "data_manifest.json")
    if not os.path.exists(manifest_path):
        print(f"  WARNING: No data manifest found in {output_dir} "
              f"— cannot verify data freshness")
        return True  # cannot verify, don't block

    with open(manifest_path) as f:
        manifest = json.load(f)

    stale_files = []
    for rel_path, recorded_checksum in manifest.get("source_checksums", {}).items():
        abs_path = repo_root / rel_path
        if not abs_path.exists():
            stale_files.append((rel_path, "file missing"))
            continue
        current = f"sha256:{_file_sha256(abs_path)}"
        if current != recorded_checksum:
            stale_files.append((rel_path, "checksum mismatch"))

    if stale_files:
        exp_label = manifest.get("experiment", experiment_id or "?")
        gen_time = manifest.get("generated_at", "unknown")
        print(f"\n  *** WARNING: data.csv for Experiment {exp_label} may be STALE ***")
        print(f"  Data generated at: {gen_time}")
        for path, reason in stale_files:
            print(f"    Changed: {path} ({reason})")
        print(f"  Re-run the experiment to regenerate data.\n")
        return False

    return True


def check_all_freshness():
    """Check data freshness for all experiments. Callable from CLI."""
    from pathlib import Path
    repo_root = Path(__file__).parent.parent.parent

    experiments = ["1", "2", "3a", "3b", "4a", "4b"]
    all_fresh = True
    for exp_id in experiments:
        output_dir = repo_root / "results" / f"exp{exp_id}"
        if not output_dir.exists():
            print(f"Experiment {exp_id}: no results directory")
            continue
        print(f"Experiment {exp_id}:", end="")
        fresh = check_data_freshness(str(output_dir), exp_id)
        if fresh:
            manifest_path = output_dir / "data_manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    m = json.load(f)
                print(f"  OK (generated {m.get('generated_at', '?')})")
            else:
                print(f"  no manifest")
        all_fresh = all_fresh and fresh

    if all_fresh:
        print("\nAll experiments: data appears fresh.")
    else:
        print("\nSome experiments have potentially stale data. "
              "Re-run affected experiments.")


def run_factorial_analysis(df, coded_cols, response_cols, output_dir,
                           experiment_id=None):
    """
    Run full factorial analysis for all response variables.

    Parameters
    ----------
    df : DataFrame
        Must contain coded columns (values -1/+1) and response columns.
    coded_cols : list of str
        Column names for coded factors (e.g. 'auction_type_coded').
    response_cols : list of str
        Column names for response variables.
    output_dir : str
        Directory to write plots and results.
    experiment_id : str or int, optional
        Experiment identifier for labeling (e.g. 1, "4a", "4b").

    Returns
    -------
    dict : Structured results with coefficients, p-values, R², etc.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check data provenance before analysis
    check_data_freshness(output_dir, experiment_id)

    analysis_log = os.path.join(output_dir, "analysis_stdout.txt")
    orig_stdout = sys.stdout

    results = {
        "experiment": experiment_id,
        "n_observations": len(df),
        "n_factors": len(coded_cols),
        "factors": coded_cols,
        "responses": {},
    }

    with open(analysis_log, "w") as log_f:
        sys.stdout = log_f

        print(f"=== Factorial Analysis: Experiment {experiment_id} ===")
        print(f"Observations: {len(df)}")
        print(f"Factors ({len(coded_cols)}): {coded_cols}")
        print(f"Responses: {response_cols}")
        print()

        # Summary statistics
        all_cols = coded_cols + response_cols
        available = [c for c in all_cols if c in df.columns]
        summary = df[available].describe().T
        print("=== Summary Statistics ===")
        print(tabulate(summary, headers="keys", tablefmt="github",
                       floatfmt=".4f"))
        print()

        for response in response_cols:
            if response not in df.columns:
                print(f"\nWARNING: {response} not in data, skipping.")
                continue

            # Drop rows with NaN in response
            df_clean = df.dropna(subset=[response])
            if len(df_clean) < len(coded_cols) + 2:
                print(f"\nWARNING: Too few observations for {response}, skipping.")
                continue

            print(f"\n{'='*60}")
            print(f"RESPONSE: {response}")
            print(f"{'='*60}")

            # Fit OLS
            model = _run_ols(df_clean, response, coded_cols)

            print("\n--- OLS Regression Summary ---")
            print(model.summary())
            print(f"\nR² = {model.rsquared:.4f}")
            print(f"Adj R² = {model.rsquared_adj:.4f}")

            # ANOVA
            print("\n--- ANOVA Table (Type III) ---")
            try:
                anova = _anova_table(model)
                print(tabulate(anova, headers="keys", tablefmt="github",
                               floatfmt=".4f"))
            except Exception as e:
                print(f"  ANOVA failed: {e}")
                anova = None

            # Significant effects
            sig = model.pvalues[model.pvalues < 0.05].drop("Intercept",
                                                            errors="ignore")
            print(f"\n--- Significant Effects (p < 0.05): {len(sig)} ---")
            for name, pval in sig.sort_values().items():
                coef = model.params[name]
                print(f"  {_clean_label(name):40s} coef={coef:+.4f}  "
                      f"p={pval:.4f}")

            # Store results
            res = {
                "r_squared": model.rsquared,
                "adj_r_squared": model.rsquared_adj,
                "f_statistic": model.fvalue,
                "f_pvalue": model.f_pvalue,
                "n_obs": int(model.nobs),
                "coefficients": {},
            }
            for name in model.params.index:
                if name == "Intercept":
                    continue
                res["coefficients"][_clean_label(name)] = {
                    "estimate": model.params[name],
                    "std_err": model.bse[name],
                    "t_value": model.tvalues[name],
                    "p_value": model.pvalues[name],
                }
            results["responses"][response] = res

            # Generate plots
            plot_pareto_chart(model, response, output_dir)
            plot_half_normal(model, response, output_dir)
            plot_main_effects(df_clean, coded_cols, response, output_dir,
                             model=model)
            plot_interactions(df_clean, model, coded_cols, response,
                            output_dir)
            plot_residuals(model, response, output_dir)

            print(f"\n  Plots saved to {output_dir}/")

        print("\n=== Analysis Complete ===")

        # Save structured results
        json_path = os.path.join(output_dir, "estimation_results.json")
        with open(json_path, "w") as jf:
            json.dump(results, jf, indent=2, default=_json_serializer)
        print(f"Structured results saved to {json_path}")

    sys.stdout = orig_stdout
    print(f"Analysis complete. Logs saved to '{analysis_log}'.")

    return results


def _json_serializer(obj):
    """Handle numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
