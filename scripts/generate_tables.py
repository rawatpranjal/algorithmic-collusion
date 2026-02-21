#!/usr/bin/env python3
"""
Generate LaTeX table snippets and copy publication figures from factorial
analysis results.

Reads results/expN/estimation_results.json (OLS coefficient tables from
factorial ANOVA) and writes:
  - paper/tables/expN_coefficients.tex  (main-effect coefficient tables per response)
  - paper/tables/expN_model_fit.tex     (model fit summary across responses)
  - paper/tables/expN_significant.tex   (cross-response significant effects, p<0.05)
  - paper/tables/expN_ranked_rev.tex    (ranked significant effects for revenue)
  - paper/tables/expN_ranked_reg.tex    (ranked significant effects for regret)
  - paper/tables/expN_ranked_vol.tex    (ranked significant effects for volatility)

Also copies key publication figures (main-effects and interaction plots)
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

# The regret variable name differs in Exp3
REGRET_VAR = {
    1: "avg_regret_of_seller",
    2: "avg_regret_of_seller",
    3: "avg_regret_seller",
    4: "mean_effective_poa",
}

# Key response variables for ranked-effects tables (rev, reg, vol)
KEY_RESPONSES = {
    1: {
        "rev": "avg_rev_last_1000",
        "reg": "avg_regret_of_seller",
        "vol": "price_volatility",
    },
    2: {
        "rev": "avg_rev_last_1000",
        "reg": "avg_regret_of_seller",
        "vol": "price_volatility",
    },
    3: {
        "rev": "avg_rev_last_1000",
        "reg": "avg_regret_seller",
        "vol": "price_volatility",
    },
    4: {
        "rev": "mean_platform_revenue",
        "reg": "mean_effective_poa",
        "vol": "mean_bid_to_value",
    },
}

# Roman numeral experiment labels
EXP_ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV"}

# Human-readable factor names
READABLE_NAMES = {
    "auction_type_coded": "Auction format",
    "auction_type": "Auction format",
    "alpha_coded": "Learning rate ($\\alpha$)",
    "alpha": "Learning rate ($\\alpha$)",
    "gamma_coded": "Discount factor ($\\gamma$)",
    "gamma": "Discount factor ($\\gamma$)",
    "exploration_coded": "Exploration strategy",
    "exploration": "Exploration strategy",
    "exploration_egreedy": "Exploration strategy",
    "asynchronous_coded": "Update mode",
    "asynchronous": "Update mode",
    "n_bidders_coded": "Number of bidders",
    "n_bidders": "Number of bidders",
    "reserve_price_coded": "Reserve price",
    "reserve_price": "Reserve price",
    "r_coded": "Reserve price",
    "r": "Reserve price",
    "init_coded": "Initialisation",
    "init": "Initialisation",
    "init_zeros": "Initialisation",
    "median_opp_coded": "Median opponent bid (state)",
    "median_opp_past_bid_index_coded": "Median opponent bid (state)",
    "median_opp_past_bid_index": "Median opponent bid (state)",
    "winner_bid_coded": "Winner bid (state)",
    "winner_bid_index_state_coded": "Winner bid (state)",
    "winner_bid_index_state": "Winner bid (state)",
    "eta_coded": "Affiliation ($\\eta$)",
    "eta": "Affiliation ($\\eta$)",
    "c_coded": "Exploration bonus ($c$)",
    "c": "Exploration bonus ($c$)",
    "lam_coded": "Regularisation ($\\lambda$)",
    "lam": "Regularisation ($\\lambda$)",
    "reg_coded": "Regularisation ($\\lambda$)",
    "reg": "Regularisation ($\\lambda$)",
    "use_median_coded": "Use median context",
    "use_median": "Use median context",
    "use_winner_coded": "Use winner context",
    "use_winner": "Use winner context",
    "budget_tightness_coded": "Budget tightness",
    "budget_tightness": "Budget tightness",
    "algorithm_coded": "Algorithm",
    "algorithm": "Algorithm",
    "aggressiveness_coded": "Aggressiveness",
    "aggressiveness": "Aggressiveness",
    "update_frequency_coded": "Update frequency",
    "update_frequency": "Update frequency",
    "initial_multiplier_coded": "Initial multiplier",
    "initial_multiplier": "Initial multiplier",
    "bandit_type_ucb_coded": "Bandit type",
    "bandit_type_ucb": "Bandit type",
    "episodes_coded": "Training episodes",
    "episodes": "Training episodes",
    "boltzmann_temp_start_coded": "Boltzmann temperature",
    "boltzmann_temp_start": "Boltzmann temperature",
    "n_actions_coded": "Action granularity",
    "n_actions": "Action granularity",
    "info_feedback_coded": "Information feedback",
    "info_feedback": "Information feedback",
    "exploration_intensity_coded": "Exploration intensity",
    "exploration_intensity": "Exploration intensity",
    "context_richness_coded": "Context richness",
    "context_richness": "Context richness",
    "eta_linear_coded": "Affiliation (linear)",
    "eta_quadratic_coded": "Affiliation (quadratic)",
    "decay_type_coded": "Decay type",
    "decay_type": "Decay type",
    "objective_coded": "Bidder objective",
    "objective": "Bidder objective",
}

READABLE_RESPONSES = {
    "avg_rev_last_1000": "Average Revenue",
    "time_to_converge": "Convergence Time",
    "avg_regret_of_seller": "Seller Regret",
    "avg_regret_seller": "Seller Regret",
    "no_sale_rate": "No-Sale Rate",
    "price_volatility": "Price Volatility",
    "winner_entropy": "Winner Entropy",
    "budget_utilization": "Budget Utilisation",
    "spend_volatility": "Spend Volatility",
    "budget_violation_rate": "Budget Violation Rate",
    "effective_bid_shading": "Effective Bid Shading",
    "multiplier_convergence_time": "Multiplier Convergence Time",
    "multiplier_final_mean": "Final Multiplier Mean",
    "multiplier_final_std": "Final Multiplier Std Dev",
    "mean_platform_revenue": "Platform Revenue",
    "mean_liquid_welfare": "Liquid Welfare",
    "mean_effective_poa": "Effective PoA",
    "mean_budget_utilization": "Budget Utilisation",
    "mean_bid_to_value": "Bid-to-Value Ratio",
    "mean_allocative_efficiency": "Allocative Efficiency",
    "mean_dual_cv": "Dual Variable CV",
    "mean_no_sale_rate": "No-Sale Rate",
    "mean_winner_entropy": "Winner Entropy",
    "warm_start_benefit": "Warm-Start Benefit",
    "inter_episode_volatility": "Inter-Episode Volatility",
    "bid_suppression_ratio": "Bid Suppression Ratio",
    "cross_episode_drift": "Cross-Episode Drift",
}

RANKED_RESPONSE_LABELS = {
    "rev": "average revenue",
    "reg": "seller regret",
    "vol": "price volatility",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def escape_latex(s):
    """Escape underscores and other special chars for LaTeX text."""
    return s.replace("_", r"\_")


def readable_effect_name(effect_key):
    """Convert a coded factor name or interaction to a readable name."""
    parts = effect_key.split(":")
    readable_parts = [READABLE_NAMES.get(p.strip(), escape_latex(p.strip())) for p in parts]
    return " $\\times$ ".join(readable_parts)


def readable_response(response_key):
    """Convert a response variable name to a readable name."""
    return READABLE_RESPONSES.get(response_key, response_key.replace("_", " ").title())


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


def effect_direction(effect_key, estimate):
    """Generate a short direction description for a significant effect."""
    name = readable_effect_name(effect_key)
    if estimate > 0:
        return f"{name} increases"
    return f"{name} decreases"


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

def generate_coefficients_tables(exp_num, responses_data):
    """Generate one coefficient table per response variable.

    Each table shows main effects only (no interactions) with:
    Effect | Estimate | Std Err | t-value | p-value | Sig
    Uses readable names.
    """
    tables = []
    roman = EXP_ROMAN[exp_num]

    for response, rdata in responses_data.items():
        coefficients = rdata.get("coefficients", {})
        main_effects = {
            k: v for k, v in coefficients.items() if is_main_effect(k)
        }
        if not main_effects:
            continue

        resp_name = readable_response(response)
        lines = []
        lines.append(r"\begin{table}[H]")
        lines.append(r"\centering")
        lines.append(
            r"\caption{Experiment %s: Main-effect coefficients for %s.}"
            % (roman, resp_name.lower())
        )
        lines.append(r"\label{tab:exp%d_coef_%s}" % (exp_num, response))
        lines.append(r"\begin{tabular}{lrrrrl}")
        lines.append(r"\toprule")
        lines.append(
            r"\textbf{Effect} & \textbf{Estimate} & \textbf{Std Err} "
            r"& \textbf{$t$-value} & \textbf{$p$-value} & \\")
        lines.append(r"\midrule")

        for effect, vals in main_effects.items():
            est = vals["estimate"]
            se = vals["std_err"]
            tval = vals["t_value"]
            pval = vals["p_value"]
            stars = significance_stars(pval)
            name = readable_effect_name(effect)
            lines.append(
                f"{name} & {est:.4f} & {se:.4f} "
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
    Uses readable response names.
    """
    roman = EXP_ROMAN[exp_num]
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Experiment %s: OLS model fit summary across response variables.}"
        % roman
    )
    lines.append(r"\label{tab:exp%d_fit}" % exp_num)
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Response} & \textbf{$R^2$} & \textbf{Adj.\,$R^2$} "
        r"& \textbf{F-stat} & \textbf{F $p$-value} \\"
    )
    lines.append(r"\midrule")

    for response, rdata in responses_data.items():
        r2 = rdata.get("r_squared", float("nan"))
        adj_r2 = rdata.get("adj_r_squared", float("nan"))
        fstat = rdata.get("f_statistic", float("nan"))
        fpval = rdata.get("f_pvalue", float("nan"))
        resp_name = readable_response(response)
        lines.append(
            f"{resp_name} & {r2:.4f} & {adj_r2:.4f} "
            f"& {fstat:.3f} & {format_pval(fpval)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_significant_effects_table(exp_num, responses_data):
    """Generate a cross-response table of significant effects (p < 0.05).

    Includes both main effects and interactions.
    Columns: Response | Effect | Estimate | |t| | p-value
    Uses readable names, sorted by |t| within each response.
    """
    roman = EXP_ROMAN[exp_num]
    sig_rows = []

    for response, rdata in responses_data.items():
        coefficients = rdata.get("coefficients", {})
        resp_rows = []
        for effect, vals in coefficients.items():
            pval = vals["p_value"]
            if pval < 0.05:
                resp_rows.append({
                    "response": response,
                    "effect": effect,
                    "estimate": vals["estimate"],
                    "t_abs": abs(vals["t_value"]),
                    "p_value": pval,
                })
        resp_rows.sort(key=lambda r: r["t_abs"], reverse=True)
        sig_rows.extend(resp_rows)

    if not sig_rows:
        return (
            "%% No significant effects at p<0.05 for Experiment %d" % exp_num
        )

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Experiment %s: All significant effects ($p < 0.05$) "
        r"across response variables, ranked by $|t|$.}" % roman
    )
    lines.append(r"\label{tab:exp%d_sig}" % exp_num)
    lines.append(r"\begin{tabular}{llrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Response} & \textbf{Effect} "
        r"& \textbf{Coeff.} & \textbf{$|t|$} & \textbf{$p$-value} \\"
    )
    lines.append(r"\midrule")

    for row in sig_rows:
        resp_name = readable_response(row["response"])
        effect_name = readable_effect_name(row["effect"])
        lines.append(
            f"{resp_name} & {effect_name} "
            f"& {row['estimate']:.4f} & {row['t_abs']:.2f} & {format_pval(row['p_value'])} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_ranked_effects_table(exp_num, response_key, short_key, rdata):
    """Generate a compact ranked-effects table for a single response variable.

    Shows significant effects (p < 0.05) sorted by |t|, with readable names
    and a direction column. This replaces Pareto chart figures.
    """
    roman = EXP_ROMAN[exp_num]
    resp_label = RANKED_RESPONSE_LABELS.get(short_key, short_key)

    coefficients = rdata.get("coefficients", {})
    sig_rows = []
    for effect, vals in coefficients.items():
        pval = vals["p_value"]
        if pval < 0.05:
            sig_rows.append({
                "effect": effect,
                "estimate": vals["estimate"],
                "t_abs": abs(vals["t_value"]),
                "p_value": pval,
            })

    sig_rows.sort(key=lambda r: r["t_abs"], reverse=True)

    # Cap at top 15 effects to keep table compact
    sig_rows = sig_rows[:15]

    if not sig_rows:
        return (
            "%% No significant effects at p<0.05 for Experiment %d %s"
            % (exp_num, response_key)
        )

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Experiment %s: Significant effects for %s ($p < 0.05$), ranked by $|t|$.}"
        % (roman, resp_label)
    )
    lines.append(r"\label{tab:exp%d_ranked_%s}" % (exp_num, short_key))
    lines.append(r"\begin{tabular}{lrrl}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Effect} & \textbf{Coeff.} & \textbf{$|t|$} & \textbf{Direction} \\"
    )
    lines.append(r"\midrule")

    for row in sig_rows:
        effect_name = readable_effect_name(row["effect"])
        direction = "+" if row["estimate"] > 0 else "$-$"
        lines.append(
            f"{effect_name} & {row['estimate']:.4f} & {row['t_abs']:.2f} & {direction} \\\\"
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
      - main effects for avg_rev_last_1000       -> eN_main_rev.png
      - main effects for regret variable          -> eN_main_reg.png
      - main effects for price_volatility         -> eN_main_vol.png
      - interaction plots for revenue, regret, volatility
      - pareto charts (kept for reference)
    """
    regret_var = REGRET_VAR[exp_num]
    exp_dir = os.path.join(RESULTS_DIR, f"exp{exp_num}")
    n = exp_num

    copy_pairs = [
        # Main effects plots
        (
            os.path.join(exp_dir, "main_effects", "main_effects_avg_rev_last_1000.png"),
            os.path.join(FIGURES_DIR, f"e{n}_main_rev.png"),
        ),
        (
            os.path.join(exp_dir, "main_effects", f"main_effects_{regret_var}.png"),
            os.path.join(FIGURES_DIR, f"e{n}_main_reg.png"),
        ),
        (
            os.path.join(exp_dir, "main_effects", "main_effects_price_volatility.png"),
            os.path.join(FIGURES_DIR, f"e{n}_main_vol.png"),
        ),
        # Interaction plots
        (
            os.path.join(exp_dir, "interaction_plots", "interactions_avg_rev_last_1000.png"),
            os.path.join(FIGURES_DIR, f"e{n}_int_rev.png"),
        ),
        (
            os.path.join(exp_dir, "interaction_plots", f"interactions_{regret_var}.png"),
            os.path.join(FIGURES_DIR, f"e{n}_int_reg.png"),
        ),
        (
            os.path.join(exp_dir, "interaction_plots", "interactions_price_volatility.png"),
            os.path.join(FIGURES_DIR, f"e{n}_int_vol.png"),
        ),
        # Pareto charts (still copied for reference, not shown in paper)
        (
            os.path.join(exp_dir, "pareto_charts", "pareto_avg_rev_last_1000.png"),
            os.path.join(FIGURES_DIR, f"e{n}_pareto_rev.png"),
        ),
        (
            os.path.join(exp_dir, "pareto_charts", f"pareto_{regret_var}.png"),
            os.path.join(FIGURES_DIR, f"e{n}_pareto_reg.png"),
        ),
        (
            os.path.join(exp_dir, "pareto_charts", "pareto_price_volatility.png"),
            os.path.join(FIGURES_DIR, f"e{n}_pareto_vol.png"),
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

    # 3. Significant effects table (cross-response)
    sig_tex = generate_significant_effects_table(exp_num, responses_data)
    sig_path = os.path.join(TABLES_DIR, f"exp{exp_num}_significant.tex")
    with open(sig_path, "w") as f:
        f.write(sig_tex + "\n")
    print(f"  Wrote {sig_path}")

    # 4. Ranked effects tables (one per key response, replacing Pareto charts)
    key_resp = KEY_RESPONSES.get(exp_num, {})
    for short_key, response_key in key_resp.items():
        if response_key in responses_data:
            ranked_tex = generate_ranked_effects_table(
                exp_num, response_key, short_key, responses_data[response_key]
            )
            ranked_path = os.path.join(TABLES_DIR, f"exp{exp_num}_ranked_{short_key}.tex")
            with open(ranked_path, "w") as f:
                f.write(ranked_tex + "\n")
            print(f"  Wrote {ranked_path}")

    # 5. Copy publication figures
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
