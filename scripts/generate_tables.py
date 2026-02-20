#!/usr/bin/env python3
"""
Generate LaTeX table snippets and copy publication figures from factorial
analysis results.

Reads results/expN/estimation_results.json (OLS coefficient tables from
factorial ANOVA) and writes:
  - paper/tables/expN_coefficients.tex  (main-effect coefficient tables per response)
  - paper/tables/expN_model_fit.tex     (model fit summary across responses)
  - paper/tables/expN_significant.tex   (cross-response significant effects, p<0.05)

Also copies key publication figures (Pareto charts and main-effects plots)
to paper/figures/ with short names for LaTeX inclusion.

Usage:
    python scripts/generate_tables.py
"""

import json
import os
import shutil

RESULTS_DIR = "results"
TABLES_DIR = "paper/tables"
FIGURES_DIR = "paper/figures"

# Response variables per experiment
RESPONSES = {
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
        "avg_regret_of_seller",
        "no_sale_rate",
        "price_volatility",
        "winner_entropy",
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
    ],
}

# The regret variable name differs in Exp3
REGRET_VAR = {
    1: "avg_regret_of_seller",
    2: "avg_regret_of_seller",
    3: "avg_regret_seller",
    4: "avg_regret_of_seller",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def escape_latex(s):
    """Escape underscores and other special chars for LaTeX text."""
    return s.replace("_", r"\_")


def format_pval(p):
    """Format a p-value for display: '< 0.0001' when very small, else 4 dp."""
    p = float(p)
    if p < 0.0001:
        return "< 0.0001"
    return f"{p:.4f}"


def significance_stars(p):
    """Return significance stars: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1."""
    p = float(p)
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return ""


def is_main_effect(name):
    """Return True if the coefficient key is a main effect (no colon)."""
    return ":" not in name


def pretty_response_name(response):
    """Make a response variable name nicer for captions."""
    return response.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

def generate_coefficients_tables(exp_num, responses_data):
    """Generate one coefficient table per response variable.

    Each table shows main effects only (no interactions) with:
    Effect | Estimate | Std Err | t-value | p-value | Sig
    """
    tables = []

    for response, rdata in responses_data.items():
        coefficients = rdata.get("coefficients", {})
        # Filter to main effects only
        main_effects = {
            k: v for k, v in coefficients.items() if is_main_effect(k)
        }
        if not main_effects:
            continue

        lines = []
        lines.append(r"\begin{table}[H]")
        lines.append(r"\centering")
        lines.append(
            r"\caption{Experiment %d: Main-effect coefficients for %s.}"
            % (exp_num, escape_latex(response))
        )
        lines.append(r"\label{tab:exp%d_coef_%s}" % (exp_num, response))
        lines.append(r"\begin{tabular}{lrrrrl}")
        lines.append(r"\toprule")
        lines.append(
            r"\textbf{Effect} & \textbf{Estimate} & \textbf{Std Err} "
            r"& \textbf{t-value} & \textbf{p-value} & \\")
        lines.append(r"\midrule")

        for effect, vals in main_effects.items():
            est = vals["estimate"]
            se = vals["std_err"]
            tval = vals["t_value"]
            pval = vals["p_value"]
            stars = significance_stars(pval)
            lines.append(
                f"{escape_latex(effect)} & {est:.4f} & {se:.4f} "
                f"& {tval:.3f} & {format_pval(pval)} & {stars} \\\\"
            )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(
            r"\par\smallskip\footnotesize "
            r"Significance: {*}{*}{*}\,$p<0.001$, "
            r"{*}{*}\,$p<0.01$, {*}\,$p<0.05$, .\,$p<0.1$."
        )
        lines.append(r"\end{table}")
        tables.append("\n".join(lines))

    return "\n\n".join(tables)


def generate_model_fit_table(exp_num, responses_data):
    """Generate a model-fit summary table across all response variables.

    Columns: Response | R^2 | Adj R^2 | F-stat | F p-value
    """
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Experiment %d: OLS model fit summary across response variables.}"
        % exp_num
    )
    lines.append(r"\label{tab:exp%d_fit}" % exp_num)
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Response} & \textbf{$R^2$} & \textbf{Adj.\,$R^2$} "
        r"& \textbf{F-stat} & \textbf{F p-value} \\"
    )
    lines.append(r"\midrule")

    for response, rdata in responses_data.items():
        r2 = rdata.get("r_squared", float("nan"))
        adj_r2 = rdata.get("adj_r_squared", float("nan"))
        fstat = rdata.get("f_statistic", float("nan"))
        fpval = rdata.get("f_pvalue", float("nan"))
        lines.append(
            f"{escape_latex(response)} & {r2:.4f} & {adj_r2:.4f} "
            f"& {fstat:.3f} & {format_pval(fpval)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_significant_effects_table(exp_num, responses_data):
    """Generate a cross-response table of significant effects (p < 0.05).

    Includes both main effects and interactions.
    Columns: Response | Effect | Estimate | p-value
    """
    sig_rows = []

    for response, rdata in responses_data.items():
        coefficients = rdata.get("coefficients", {})
        for effect, vals in coefficients.items():
            pval = vals["p_value"]
            if pval < 0.05:
                sig_rows.append({
                    "response": response,
                    "effect": effect,
                    "estimate": vals["estimate"],
                    "p_value": pval,
                })

    if not sig_rows:
        return (
            "%% No significant effects at p<0.05 for Experiment %d" % exp_num
        )

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Experiment %d: All significant effects ($p < 0.05$) "
        r"across response variables.}" % exp_num
    )
    lines.append(r"\label{tab:exp%d_sig}" % exp_num)
    lines.append(r"\begin{tabular}{llrr}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Response} & \textbf{Effect} "
        r"& \textbf{Estimate} & \textbf{p-value} \\"
    )
    lines.append(r"\midrule")

    for row in sig_rows:
        lines.append(
            f"{escape_latex(row['response'])} & {escape_latex(row['effect'])} "
            f"& {row['estimate']:.4f} & {format_pval(row['p_value'])} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure copying
# ---------------------------------------------------------------------------

def copy_figures(exp_num):
    """Copy the most important publication figures for an experiment.

    For each experiment, copy:
      - pareto chart for avg_rev_last_1000      -> eN_pareto_rev.png
      - main effects for avg_rev_last_1000       -> eN_main_rev.png
      - pareto chart for regret variable          -> eN_pareto_reg.png
      - main effects for regret variable          -> eN_main_reg.png
    """
    regret_var = REGRET_VAR[exp_num]
    exp_dir = os.path.join(RESULTS_DIR, f"exp{exp_num}")
    n = exp_num  # shorthand for naming

    copy_pairs = [
        (
            os.path.join(exp_dir, "pareto_charts", "pareto_avg_rev_last_1000.png"),
            os.path.join(FIGURES_DIR, f"e{n}_pareto_rev.png"),
        ),
        (
            os.path.join(exp_dir, "main_effects", "main_effects_avg_rev_last_1000.png"),
            os.path.join(FIGURES_DIR, f"e{n}_main_rev.png"),
        ),
        (
            os.path.join(exp_dir, "pareto_charts", f"pareto_{regret_var}.png"),
            os.path.join(FIGURES_DIR, f"e{n}_pareto_reg.png"),
        ),
        (
            os.path.join(exp_dir, "main_effects", f"main_effects_{regret_var}.png"),
            os.path.join(FIGURES_DIR, f"e{n}_main_reg.png"),
        ),
    ]

    for src, dst in copy_pairs:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied {src} -> {dst}")
        else:
            print(f"  WARNING: Source not found: {src}")


# ---------------------------------------------------------------------------
# Per-experiment processing
# ---------------------------------------------------------------------------

def process_experiment(exp_num):
    """Process a single experiment: generate tables and copy figures."""
    json_path = os.path.join(RESULTS_DIR, f"exp{exp_num}", "estimation_results.json")

    if not os.path.exists(json_path):
        print(f"Skipping exp{exp_num}: {json_path} not found")
        return

    with open(json_path) as f:
        data = json.load(f)

    responses_data = data.get("responses", {})
    if not responses_data:
        print(f"Skipping exp{exp_num}: no response data in JSON")
        return

    # 1. Coefficient tables (one per response, combined into one file)
    coef_tex = generate_coefficients_tables(exp_num, responses_data)
    coef_path = os.path.join(TABLES_DIR, f"exp{exp_num}_coefficients.tex")
    with open(coef_path, "w") as f:
        f.write(coef_tex + "\n")
    print(f"  Wrote {coef_path}")

    # 2. Model fit summary table
    fit_tex = generate_model_fit_table(exp_num, responses_data)
    fit_path = os.path.join(TABLES_DIR, f"exp{exp_num}_model_fit.tex")
    with open(fit_path, "w") as f:
        f.write(fit_tex + "\n")
    print(f"  Wrote {fit_path}")

    # 3. Significant effects table
    sig_tex = generate_significant_effects_table(exp_num, responses_data)
    sig_path = os.path.join(TABLES_DIR, f"exp{exp_num}_significant.tex")
    with open(sig_path, "w") as f:
        f.write(sig_tex + "\n")
    print(f"  Wrote {sig_path}")

    # 4. Copy publication figures
    copy_figures(exp_num)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    for exp_num in [1, 2, 3, 4]:
        print(f"\nProcessing Experiment {exp_num}...")
        process_experiment(exp_num)

    print("\nDone. Tables written to paper/tables/, figures copied to paper/figures/")


if __name__ == "__main__":
    main()
