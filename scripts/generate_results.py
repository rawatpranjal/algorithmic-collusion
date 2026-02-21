#!/usr/bin/env python3
"""
Generate a standalone results.pdf with ALL analysis outputs.

Reads factorial ANOVA estimation results, design info, and data summaries
from results/exp{1,2,3}/ and produces a comprehensive LaTeX document
compiled to paper/results.pdf.

Usage:
    python scripts/generate_results.py

Output:
    paper/results.tex
    paper/results.pdf
"""

import json
import math
import os
import subprocess
import sys

import pandas as pd

RESULTS_DIR = "results"
PAPER_DIR = "paper"

EXP_TITLES = {
    1: "Q-Learning with Constant Valuations",
    2: "Q-Learning with Affiliated Values",
    3: "LinUCB Bandits with Affiliated Values",
    4: "Autobidding Pacing: Auction Format, Objective, and Market Thickness",
}

EXP_DESIGNS = {
    1: {"design": "$2^{11-1}$ half-fraction (Res~V)", "k": 11},
    2: {"design": "$3 \\times 2^3 = 24$ mixed-level factorial", "k": 4},
    3: {"design": "$2^{8}$ full factorial", "k": 8},
    4: {"design": "$2 \\times 2 \\times 2 = 8$ cells $\\times$ 50 seeds", "k": 3},
}

# Response variables expected per experiment
EXP_RESPONSES = {
    1: [
        "avg_rev_last_1000",
        "time_to_converge",
        "avg_regret_of_seller",
        "no_sale_rate",
        "price_volatility",
        "winner_entropy",
    ],
    2: [
        "avg_rev_last_1000",
        "time_to_converge",
        "no_sale_rate",
        "price_volatility",
        "winner_entropy",
        "excess_regret",
        "efficient_regret",
        "btv_median",
        "winners_curse_freq",
        "bid_dispersion",
        "signal_slope_ratio",
    ],
    3: [
        "avg_rev_last_1000",
        "time_to_converge",
        "avg_regret_seller",
        "no_sale_rate",
        "price_volatility",
        "winner_entropy",
    ],
    4: [
        "mean_platform_revenue",
        "mean_liquid_welfare",
        "mean_effective_poa",
        "mean_budget_utilization",
        "mean_bid_to_value",
        "mean_allocative_efficiency",
        "mean_dual_cv",
        "mean_no_sale_rate",
        "mean_winner_entropy",
        "warm_start_benefit",
        "inter_episode_volatility",
        "bid_suppression_ratio",
        "cross_episode_drift",
    ],
}

# Plot subdirectories and filename patterns
PLOT_TYPES = [
    ("pareto_charts", "pareto_{response}", "Pareto Chart", "Pareto Charts"),
    ("normal_prob_plots", "halfnormal_{response}", "Half-Normal Probability Plot", "Half-Normal Probability Plots"),
    ("main_effects", "main_effects_{response}", "Main Effects Plot", "Main Effects Plots"),
    ("interaction_plots", "interactions_{response}", "Interaction Plot", "Interaction Plots"),
    ("residuals", "residuals_{response}", "Residual Diagnostics", "Residual Diagnostics"),
]


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------


def escape_latex(s):
    """Escape underscores and other special chars for LaTeX."""
    s = str(s)
    s = s.replace("_", r"\_")
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    s = s.replace("#", r"\#")
    return s


def format_number(x, decimals=4):
    """Format a number, handling inf/nan."""
    if x is None:
        return "---"
    if isinstance(x, str):
        return x
    if math.isnan(x) or math.isinf(x):
        return "---"
    return f"{x:.{decimals}f}"


def format_pval(p):
    """Format p-value with scientific notation for very small values."""
    if p is None or (isinstance(p, float) and (math.isnan(p) or math.isinf(p))):
        return "---"
    p = float(p)
    if p < 0.0001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def significance_stars(p):
    """Return significance stars for a p-value."""
    if p is None or (isinstance(p, float) and (math.isnan(p) or math.isinf(p))):
        return ""
    p = float(p)
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.1:
        return "."
    return ""


def safe_float(val):
    """Safely convert to float, returning NaN for bad values."""
    try:
        v = float(val)
        return v
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_design_info(exp_num):
    """Load design_info.json, checking main dir then quick_test fallback."""
    primary = os.path.join(RESULTS_DIR, f"exp{exp_num}", "design_info.json")
    fallback = os.path.join(RESULTS_DIR, f"exp{exp_num}", "quick_test", "design_info.json")

    for path in [primary, fallback]:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None


def load_estimation_results(exp_num):
    """Load estimation_results.json if it exists."""
    path = os.path.join(RESULTS_DIR, f"exp{exp_num}", "estimation_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_data_csv(exp_num):
    """Load data.csv if it exists."""
    path = os.path.join(RESULTS_DIR, f"exp{exp_num}", "data.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def find_plot(exp_num, subdir, filename_pattern, response):
    """Find a plot file, return relative path from PAPER_DIR or None."""
    filename = filename_pattern.format(response=response) + ".png"
    abs_path = os.path.join(RESULTS_DIR, f"exp{exp_num}", subdir, filename)
    if os.path.exists(abs_path):
        return os.path.relpath(abs_path, PAPER_DIR)
    return None


def load_robust_results(exp_num):
    """Load robust/robust_results.json if it exists."""
    path = os.path.join(RESULTS_DIR, f"exp{exp_num}", "robust", "robust_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------


def gen_design_summary(exp_num, design_info):
    """Generate the Design Summary subsection."""
    lines = []
    lines.append(r"\subsection{Design Summary}")

    if design_info is None:
        lines.append(r"\textit{No design info available for this experiment.}")
        return "\n".join(lines)

    factors = design_info.get("factors", {})
    n_factors = design_info.get("n_factors", len(factors))
    design_desc = EXP_DESIGNS.get(exp_num, {}).get("design", f"$2^{{{n_factors}}}$")
    k = EXP_DESIGNS.get(exp_num, {}).get("k", n_factors)
    if exp_num == 2:
        n_cells = 24  # 3 × 2^3 mixed-level
    elif exp_num == 4:
        n_cells = 8  # 2^3 full factorial
    elif exp_num in (1,):
        n_cells = 2 ** (k - 1)  # half-fraction
    else:
        n_cells = 2 ** k if k <= 11 else 2 ** (k - 1)

    lines.append(f"Design: {design_desc} with {n_factors} factors.")
    lines.append("")

    # Factor definitions table
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        f"\\caption{{Factor definitions for Experiment {exp_num}.}}"
    )
    lines.append(f"\\label{{tab:exp{exp_num}_factors}}")
    lines.append(r"\begin{tabular}{llll}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Factor} & \textbf{Low ($-1$)} & \textbf{High ($+1$)} & \textbf{Coded Column} \\"
    )
    lines.append(r"\midrule")

    for fname, fdef in factors.items():
        low_val = fdef.get("low", "---")
        high_val = fdef.get("high", "---")
        # Format booleans nicely
        if isinstance(low_val, bool):
            low_val = str(low_val)
        if isinstance(high_val, bool):
            high_val = str(high_val)
        coded_col = escape_latex(f"{fname}_coded")
        lines.append(
            f"  {escape_latex(fname)} & {escape_latex(str(low_val))} "
            f"& {escape_latex(str(high_val))} & \\texttt{{{coded_col}}} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def gen_data_summary(exp_num, df):
    """Generate descriptive statistics table from data.csv."""
    lines = []
    lines.append(r"\subsection{Data Summary}")

    if df is None:
        lines.append(r"\textit{No data.csv found for this experiment.}")
        return "\n".join(lines)

    n_obs = len(df)
    responses = EXP_RESPONSES.get(exp_num, [])
    # Filter to columns that actually exist
    available = [r for r in responses if r in df.columns]

    if not available:
        lines.append(
            r"\textit{No response variables found in data.csv.}"
        )
        return "\n".join(lines)

    lines.append(f"Total observations: {n_obs}.")
    lines.append("")

    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        f"\\caption{{Descriptive statistics for Experiment {exp_num} response variables.}}"
    )
    lines.append(f"\\label{{tab:exp{exp_num}_desc}}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Response} & \textbf{Mean} & \textbf{Std Dev} "
        r"& \textbf{Min} & \textbf{Max} \\"
    )
    lines.append(r"\midrule")

    for resp in available:
        col = df[resp].dropna()
        if len(col) == 0:
            lines.append(f"  {escape_latex(resp)} & --- & --- & --- & --- \\\\")
            continue
        mean_val = col.mean()
        std_val = col.std()
        min_val = col.min()
        max_val = col.max()
        lines.append(
            f"  {escape_latex(resp)} & {format_number(mean_val)} "
            f"& {format_number(std_val)} & {format_number(min_val)} "
            f"& {format_number(max_val)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def gen_coefficients_table(exp_num, est_data, response_key, response_info):
    """Generate a longtable of OLS coefficients for one response variable."""
    lines = []
    coefficients = response_info.get("coefficients", {})
    if not coefficients:
        return ""

    resp_display = escape_latex(response_key)
    lines.append(r"\begin{longtable}{lrrrrr}")
    lines.append(
        f"\\caption{{OLS coefficients for \\texttt{{{resp_display}}} "
        f"(Experiment {exp_num}).}} "
        f"\\label{{tab:exp{exp_num}_coeff_{response_key.replace('_', '')}}} \\\\"
    )
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Effect} & \textbf{Estimate} & \textbf{Std Err} "
        r"& \textbf{t-value} & \textbf{p-value} & \textbf{Sig.} \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    # Continuation header
    lines.append(
        f"\\multicolumn{{6}}{{l}}{{\\textit{{(continued) "
        f"Coefficients for \\texttt{{{resp_display}}}}}}} \\\\"
    )
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Effect} & \textbf{Estimate} & \textbf{Std Err} "
        r"& \textbf{t-value} & \textbf{p-value} & \textbf{Sig.} \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{r}{\textit{Continued on next page}} \\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    row_idx = 0
    for effect_name, coeff_info in coefficients.items():
        estimate = safe_float(coeff_info.get("estimate"))
        std_err = safe_float(coeff_info.get("std_err"))
        t_value = safe_float(coeff_info.get("t_value"))
        p_value = safe_float(coeff_info.get("p_value"))
        stars = significance_stars(p_value)

        # Alternate row shading
        if row_idx % 2 == 0:
            lines.append(r"\rowcolor{gray!8}")

        lines.append(
            f"  {escape_latex(effect_name)} & {format_number(estimate)} "
            f"& {format_number(std_err)} & {format_number(t_value)} "
            f"& {format_pval(p_value)} & {stars} \\\\"
        )
        row_idx += 1

    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def gen_ols_coefficients(exp_num, est_data):
    """Generate all OLS coefficient tables for an experiment."""
    lines = []
    lines.append(r"\subsection{OLS Coefficients}")

    if est_data is None:
        lines.append(
            r"\textit{No estimation results available for this experiment.}"
        )
        return "\n".join(lines)

    responses = est_data.get("responses", {})
    if not responses:
        lines.append(r"\textit{No response models found.}")
        return "\n".join(lines)

    lines.append(
        "Significance codes: "
        r"\texttt{***}~$p<0.001$, \texttt{**}~$p<0.01$, "
        r"\texttt{*}~$p<0.05$, \texttt{.}~$p<0.1$."
    )
    lines.append("")

    for resp_key, resp_info in responses.items():
        table = gen_coefficients_table(exp_num, est_data, resp_key, resp_info)
        if table:
            lines.append(table)
            lines.append("")

    return "\n".join(lines)


def gen_model_fit_summary(exp_num, est_data):
    """Generate compact model fit summary table."""
    lines = []
    lines.append(r"\subsection{Model Fit Summary}")

    if est_data is None:
        lines.append(
            r"\textit{No estimation results available for this experiment.}"
        )
        return "\n".join(lines)

    responses = est_data.get("responses", {})
    if not responses:
        lines.append(r"\textit{No response models found.}")
        return "\n".join(lines)

    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        f"\\caption{{Model fit summary for Experiment {exp_num}.}}"
    )
    lines.append(f"\\label{{tab:exp{exp_num}_fit}}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Response} & \textbf{$R^2$} & \textbf{Adj.\ $R^2$} "
        r"& \textbf{F-statistic} & \textbf{F p-value} \\"
    )
    lines.append(r"\midrule")

    for resp_key, resp_info in responses.items():
        r2 = safe_float(resp_info.get("r_squared"))
        adj_r2 = safe_float(resp_info.get("adj_r_squared"))
        f_stat = safe_float(resp_info.get("f_statistic"))
        f_pval = safe_float(resp_info.get("f_pvalue"))

        lines.append(
            f"  {escape_latex(resp_key)} & {format_number(r2)} "
            f"& {format_number(adj_r2)} & {format_number(f_stat)} "
            f"& {format_pval(f_pval)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def gen_significant_effects(exp_num, est_data):
    """Generate cross-response table of significant effects (p < 0.05)."""
    lines = []
    lines.append(r"\subsection{Significant Effects Summary}")

    if est_data is None:
        lines.append(
            r"\textit{No estimation results available for this experiment.}"
        )
        return "\n".join(lines)

    responses = est_data.get("responses", {})
    sig_rows = []

    for resp_key, resp_info in responses.items():
        coefficients = resp_info.get("coefficients", {})
        for effect_name, coeff_info in coefficients.items():
            p_value = safe_float(coeff_info.get("p_value"))
            if not math.isnan(p_value) and not math.isinf(p_value) and p_value < 0.05:
                sig_rows.append(
                    {
                        "response": resp_key,
                        "effect": effect_name,
                        "estimate": safe_float(coeff_info.get("estimate")),
                        "p_value": p_value,
                    }
                )

    if not sig_rows:
        lines.append(
            r"\textit{No effects reached significance at $p < 0.05$.}"
        )
        return "\n".join(lines)

    lines.append(
        f"Effects significant at $p < 0.05$ across all response variables "
        f"({len(sig_rows)} total)."
    )
    lines.append("")

    lines.append(r"\begin{longtable}{llrr}")
    lines.append(
        f"\\caption{{Significant effects ($p < 0.05$) for Experiment {exp_num}.}} "
        f"\\label{{tab:exp{exp_num}_sig}} \\\\"
    )
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Response} & \textbf{Effect} "
        r"& \textbf{Estimate} & \textbf{p-value} \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(
        r"\multicolumn{4}{l}{\textit{(continued) Significant effects}} \\"
    )
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Response} & \textbf{Effect} "
        r"& \textbf{Estimate} & \textbf{p-value} \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    row_idx = 0
    for row in sig_rows:
        if row_idx % 2 == 0:
            lines.append(r"\rowcolor{gray!8}")
        lines.append(
            f"  {escape_latex(row['response'])} & {escape_latex(row['effect'])} "
            f"& {format_number(row['estimate'])} & {format_pval(row['p_value'])} \\\\"
        )
        row_idx += 1

    lines.append(r"\end{longtable}")

    return "\n".join(lines)


def gen_plot_section(exp_num, est_data, plot_type_label, subdir, filename_pattern, section_title=None):
    """Generate a subsection with plots for all response variables."""
    lines = []
    heading = section_title or f"{plot_type_label}s"
    lines.append(f"\\subsection{{{heading}}}")

    # Determine which responses to look for
    responses = []
    if est_data is not None:
        responses = list(est_data.get("responses", {}).keys())
    if not responses:
        # Fallback to the expected responses list
        responses = EXP_RESPONSES.get(exp_num, [])

    found_any = False
    for resp in responses:
        rel_path = find_plot(exp_num, subdir, filename_pattern, resp)
        if rel_path is None:
            continue
        found_any = True
        resp_display = escape_latex(resp)
        lines.append(r"\begin{figure}[H]")
        lines.append(r"\centering")
        lines.append(f"\\includegraphics[width=0.85\\textwidth]{{{rel_path}}}")
        lines.append(
            f"\\caption{{{plot_type_label}: \\texttt{{{resp_display}}} "
            f"(Experiment {exp_num}).}}"
        )
        lines.append(
            f"\\label{{fig:exp{exp_num}_{subdir}_{resp.replace('_', '')}}}"
        )
        lines.append(r"\end{figure}")
        lines.append("")

    if not found_any:
        lines.append(
            f"\\textit{{No plots found for this experiment in \\texttt{{{escape_latex(subdir)}}}/.}}"
        )

    return "\n".join(lines)


def gen_robustness_summary(exp_num, robust_data):
    """Generate a robustness analysis subsection from robust_results.json."""
    lines = []
    lines.append(r"\subsection{Robustness Analysis}")

    if robust_data is None:
        lines.append(
            r"\textit{No robustness results available. "
            r"Run \texttt{est" + str(exp_num) + r".py} to generate.}"
        )
        return "\n".join(lines)

    # --- Model adequacy table ---
    press_data = robust_data.get("press", {})
    lgbm_data = robust_data.get("lightgbm", {})
    lof_data = robust_data.get("lack_of_fit", {})
    responses = list(press_data.keys()) or list(lgbm_data.keys())

    if responses:
        lines.append(r"\subsubsection{Model Adequacy}")
        lines.append(r"\begin{table}[H]")
        lines.append(r"\centering")
        lines.append(
            f"\\caption{{Model adequacy diagnostics for Experiment {exp_num}.}}"
        )
        lines.append(f"\\label{{tab:exp{exp_num}_robust_adequacy}}")
        lines.append(r"\begin{tabular}{lrrrr}")
        lines.append(r"\toprule")
        lines.append(
            r"\textbf{Response} & \textbf{$R^2$} & \textbf{Pred-$R^2$} "
            r"& \textbf{LGBM $R^2$} & \textbf{LOF $p$} \\"
        )
        lines.append(r"\midrule")

        for resp in responses:
            r2 = safe_float(press_data.get(resp, {}).get("r_squared"))
            pred_r2 = safe_float(press_data.get(resp, {}).get("predicted_r_squared"))
            lgbm_r2 = safe_float(lgbm_data.get(resp, {}).get("lgbm_cv_r2"))
            lof_p = lof_data.get(resp, {}).get("p_lack_of_fit")
            lof_str = format_pval(lof_p) if lof_p is not None else "---"

            lines.append(
                f"  {escape_latex(resp)} & {format_number(r2)} "
                f"& {format_number(pred_r2)} & {format_number(lgbm_r2)} "
                f"& {lof_str} \\\\"
            )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

    # --- Inference robustness table ---
    hc3_data = robust_data.get("hc3", {})
    mult_data = robust_data.get("multiplicity", {})

    if hc3_data and mult_data:
        lines.append(r"\subsubsection{Inference Robustness}")
        lines.append(r"\begin{table}[H]")
        lines.append(r"\centering")
        lines.append(
            f"\\caption{{Inference robustness for Experiment {exp_num}: "
            f"effect counts by correction method.}}"
        )
        lines.append(f"\\label{{tab:exp{exp_num}_robust_inference}}")
        lines.append(r"\begin{tabular}{lrrrr}")
        lines.append(r"\toprule")
        lines.append(
            r"\textbf{Response} & \textbf{\# HC3 Flipped} "
            r"& \textbf{\# Sig OLS} & \textbf{\# Sig Holm} "
            r"& \textbf{\# Sig BH} \\"
        )
        lines.append(r"\midrule")

        tests = mult_data.get("tests", [])
        for resp in hc3_data:
            n_flipped = hc3_data[resp].get("n_flipped", 0)
            hc3_terms = hc3_data[resp].get("terms", {})
            n_ols_sig = sum(1 for t in hc3_terms.values() if t.get("ols_p", 1) < 0.05)
            resp_tests = [t for t in tests if t["label"].startswith(resp + "|")]
            n_holm = sum(1 for t in resp_tests if t.get("holm_sig", False))
            n_bh = sum(1 for t in resp_tests if t.get("bh_sig", False))

            lines.append(
                f"  {escape_latex(resp)} & {n_flipped} & {n_ols_sig} "
                f"& {n_holm} & {n_bh} \\\\"
            )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

        # Global summary
        n_total = mult_data.get("n_tests", 0)
        n_raw = mult_data.get("n_raw_sig", 0)
        n_holm_total = mult_data.get("n_holm_sig", 0)
        n_bh_total = mult_data.get("n_bh_sig", 0)
        lines.append(
            f"Across {n_total} total tests: {n_raw} significant at raw "
            f"$\\alpha=0.05$, {n_holm_total} survive Holm-Bonferroni, "
            f"{n_bh_total} survive Benjamini-Hochberg."
        )
        lines.append("")

    # --- Robustness plots ---
    robust_plots = [
        ("lgbm_comparison.png", "Linear vs.~nonparametric model comparison"),
        ("response_correlations.png", "Response variable correlation matrix"),
        ("power_analysis.png", "Minimum detectable effect sizes (80\\% power)"),
    ]

    found_plots = False
    for filename, caption_text in robust_plots:
        abs_path = os.path.join(RESULTS_DIR, f"exp{exp_num}", "robust", filename)
        if os.path.exists(abs_path):
            if not found_plots:
                lines.append(r"\subsubsection{Robustness Diagnostics}")
                found_plots = True
            rel_path = os.path.relpath(abs_path, PAPER_DIR)
            label_stem = filename.replace(".png", "").replace("_", "")
            lines.append(r"\begin{figure}[H]")
            lines.append(r"\centering")
            lines.append(f"\\includegraphics[width=0.85\\textwidth]{{{rel_path}}}")
            lines.append(
                f"\\caption{{{caption_text} (Experiment {exp_num}).}}"
            )
            lines.append(f"\\label{{fig:exp{exp_num}_robust_{label_stem}}}")
            lines.append(r"\end{figure}")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------


def generate_experiment_section(exp_num):
    """Generate the complete LaTeX section for one experiment."""
    title = EXP_TITLES.get(exp_num, f"Experiment {exp_num}")
    lines = []
    lines.append(f"\\section{{Experiment {exp_num}: {title}}}")

    # Add trace figure if it exists
    trace_path = os.path.join(PAPER_DIR, "figures", f"e{exp_num}_trace.png")
    if os.path.exists(trace_path):
        rel_path = os.path.relpath(trace_path, PAPER_DIR)
        lines.append(r"\begin{figure}[H]")
        lines.append(r"\centering")
        lines.append(f"\\includegraphics[width=0.95\\textwidth]{{{rel_path}}}")
        lines.append(f"\\caption{{Representative learning trajectory for Experiment {exp_num}.}}")
        lines.append(f"\\label{{fig:exp{exp_num}_trace}}")
        lines.append(r"\end{figure}")
        lines.append("")

    # Load all data sources
    design_info = load_design_info(exp_num)
    est_data = load_estimation_results(exp_num)
    df = load_data_csv(exp_num)
    robust_data = load_robust_results(exp_num)

    # 1. Design Summary
    lines.append(gen_design_summary(exp_num, design_info))
    lines.append("")

    # 2. Data Summary
    lines.append(gen_data_summary(exp_num, df))
    lines.append("")

    # 3. OLS Coefficients Tables
    lines.append(gen_ols_coefficients(exp_num, est_data))
    lines.append("")

    # 4. Model Fit Summary
    lines.append(gen_model_fit_summary(exp_num, est_data))
    lines.append("")

    # 5. Significant Effects Summary
    lines.append(gen_significant_effects(exp_num, est_data))
    lines.append("")

    # 6-10. Plot sections
    for subdir, filename_pattern, plot_label, section_title in PLOT_TYPES:
        lines.append(
            gen_plot_section(
                exp_num, est_data, plot_label, subdir, filename_pattern,
                section_title=section_title,
            )
        )
        lines.append("")

    # 11. Robustness Analysis
    lines.append(gen_robustness_summary(exp_num, robust_data))
    lines.append("")

    return "\n".join(lines)


def gen_verification_section():
    """Generate BNE verification section from JSON results."""
    lines = []
    lines.append(r"\section{BNE Verification}")

    json_path = os.path.join(RESULTS_DIR, "verification", "bne_verification_results.json")
    if not os.path.exists(json_path):
        lines.append(r"\textit{No verification results found. Run src/verification/bne\_verify.py first.}")
        return "\n".join(lines)

    with open(json_path) as f:
        results = json.load(f)

    params = results.get("parameters", {})
    lines.append(
        f"Verification ran with {params.get('M_deviation', '?')} MC draws for deviation checks "
        f"and {params.get('M_revenue', '?')} draws for revenue validation."
    )
    lines.append("")

    # Deviation table
    lines.append(r"\subsection{Deviation Checks}")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Maximum payoff gain from unilateral deviation from BNE.}")
    lines.append(r"\label{tab:results_bne_deviation}")
    lines.append(r"\begin{tabular}{cccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{$\eta$} & \textbf{N} & \textbf{Auction} & \textbf{$\phi$} & \textbf{Max Gain} & \textbf{Status} \\")
    lines.append(r"\midrule")

    for check in results.get("deviation_checks", []):
        eta = check["eta"]
        N = check["N"]
        atype = check["auction_type"].upper()
        phi = check["phi"]
        gain = check["max_gain"]
        status = "Pass" if gain < 0.005 else "Fail"
        lines.append(f"  {eta} & {N} & {atype} & {phi:.4f} & {gain:.6f} & {status} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # Revenue table
    lines.append(r"\subsection{Revenue Formula Validation}")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Analytical vs.\ Monte Carlo revenue with revenue equivalence check.}")
    lines.append(r"\label{tab:results_bne_revenue}")
    lines.append(r"\begin{tabular}{cccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{$\eta$} & \textbf{N} & \textbf{Analytical} & \textbf{FPA MC} & \textbf{SPA MC} & \textbf{$|$FPA$-$SPA$|$} \\")
    lines.append(r"\midrule")

    for entry in results.get("revenue_checks", []):
        eta = entry["eta"]
        N = entry["N"]
        ana = entry["analytical"]
        fpa = entry["fpa_mc"]
        spa = entry["spa_mc"]
        gap = entry["fpa_spa_gap"]
        lines.append(f"  {eta} & {N} & {ana:.4f} & {fpa:.4f} & {spa:.4f} & {gap:.4f} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_results_tex():
    """Generate the complete results.tex document."""
    parts = []

    # Preamble
    parts.append(
        r"""\documentclass[11pt]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{float}
\usepackage{hyperref}
\usepackage{longtable}
\usepackage{caption}
\usepackage[table]{xcolor}

\hypersetup{
    colorlinks=true,
    linkcolor=blue!60!black,
    urlcolor=blue!60!black,
}

\title{Algorithmic Collusion in Auctions\\[0.5em]
\large Comprehensive Factorial Analysis Results}
\author{Auto-Generated Report}
\date{\today}

\begin{document}
\maketitle
\tableofcontents
\newpage
"""
    )

    # Experiment sections
    for exp_num in [1, 2, 3, 4]:
        print(f"  Processing Experiment {exp_num}...")
        parts.append(generate_experiment_section(exp_num))
        parts.append(r"\newpage")
        parts.append("")

    # BNE Verification section
    print("  Processing BNE Verification...")
    parts.append(gen_verification_section())
    parts.append(r"\newpage")
    parts.append("")

    # End document
    parts.append(r"\end{document}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    os.makedirs(PAPER_DIR, exist_ok=True)

    print("Generating results.tex...")
    tex_content = generate_results_tex()
    tex_path = os.path.join(PAPER_DIR, "results.tex")

    with open(tex_path, "w") as f:
        f.write(tex_content)
    print(f"  Wrote {tex_path}")

    # Compile to PDF (run twice for TOC)
    print("Compiling PDF (pass 1 of 2)...")
    result1 = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "results.tex"],
        cwd=PAPER_DIR,
        capture_output=True,
        text=True,
    )

    print("Compiling PDF (pass 2 of 2)...")
    result2 = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "results.tex"],
        cwd=PAPER_DIR,
        capture_output=True,
        text=True,
    )

    pdf_path = os.path.join(PAPER_DIR, "results.pdf")
    if os.path.exists(pdf_path):
        print(f"  Success: {pdf_path}")
    else:
        print("PDF compilation failed. LaTeX output:")
        output = result2.stdout or result1.stdout or ""
        # Print last 3000 chars which usually contain the error
        print(output[-3000:])
        sys.exit(1)


if __name__ == "__main__":
    main()
