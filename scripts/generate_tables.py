#!/usr/bin/env python3
"""
Generate LaTeX table snippets and copy publication figures from factorial
analysis results.

Reads results/expN/estimation_results.json (OLS coefficient tables from
factorial ANOVA) and writes (using paper-facing experiment numbers):
  - paper/tables/exp{1a,1b,2a,2b,3a,3b}_coefficients.tex  (main-effect coefficient tables)
  - paper/tables/exp{...}_model_fit.tex     (model fit summary across responses)
  - paper/tables/exp{...}_significant.tex   (cross-response significant effects, p<0.05)
  - paper/tables/exp{...}_ranked_rev.tex    (ranked significant effects for revenue)
  - paper/tables/exp{...}_ranked_poa.tex    (ranked significant effects for PoA, exp3a/3b only)
  - paper/tables/exp{...}_ranked_vol.tex    (ranked significant effects for volatility)

Also copies key publication figures (main-effects and interaction plots)
to paper/figures/ with short names for LaTeX inclusion.

Usage:
    python scripts/generate_tables.py
"""

import json
import os
import shutil
import sys

RESULTS_DIR = "results"
TABLES_DIR = "paper/tables"
FIGURES_DIR = "paper/figures"

# Key response variables for ranked-effects tables (rev, vol; poa for exp4a/4b)
KEY_RESPONSES = {
    "1":  {"rev": "avg_rev_last_1000",     "vol": "price_volatility",  "bne": "ratio_to_theory"},
    "2":  {"rev": "avg_rev_last_1000",     "vol": "price_volatility",  "bne": "ratio_to_theory"},
    "3a": {"rev": "avg_rev_last_1000",     "vol": "price_volatility",  "bne": "ratio_to_theory"},
    "3b": {"rev": "avg_rev_last_1000",     "vol": "price_volatility",  "bne": "ratio_to_theory"},
    "4a": {"rev": "mean_platform_revenue", "poa": "mean_effective_poa_lp", "vol": "mean_bid_to_value"},
    "4b": {"rev": "mean_platform_revenue", "poa": "mean_effective_poa_lp", "vol": "mean_bid_to_value"},
}

# Experiment labels for captions (Arabic numerals, grouped by algorithm family)
EXP_ROMAN = {"1": "1a", "2": "1b", "3a": "2a", "3b": "2b", "4a": "3a", "4b": "3b"}

# Mapping from internal experiment ID to paper file naming
EXP_FILE_NUM = {"1": "1a", "2": "1b", "3a": "2a", "3b": "2b", "4a": "3a", "4b": "3b"}

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
    "eta_linear": "Affiliation (linear)",
    "eta_quadratic_coded": "Affiliation (quadratic)",
    "eta_quadratic": "Affiliation (quadratic)",
    "state_info_coded": "State information",
    "state_info": "State information",
    "decay_type_coded": "Decay type",
    "decay_type": "Decay type",
    "objective_coded": "Bidder objective",
    "objective": "Bidder objective",
    "budget_multiplier_coded": "Budget multiplier",
    "budget_multiplier": "Budget multiplier",
    "sigma_coded": "Value dispersion ($\\sigma$)",
    "sigma": "Value dispersion ($\\sigma$)",
    "memory_decay_coded": "Memory decay ($\\gamma_m$)",
    "memory_decay": "Memory decay ($\\gamma_m$)",
}

READABLE_RESPONSES = {
    "avg_rev_last_1000": "Average Revenue",
    "avg_rev_all": "Lifetime Revenue",
    "mean_rev_all": "Lifetime Revenue",
    "time_to_converge": "Convergence Time",
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
    "mean_effective_poa": "Effective PoA (Greedy)",
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
    "mean_lp_offline_welfare": "LP Offline Welfare",
    "mean_effective_poa_lp": "Effective PoA",
    "ratio_to_theory": "Revenue / BNE",
}

RANKED_RESPONSE_LABELS = {
    "rev": "average revenue",
    "poa": "effective Price of Anarchy",
    "vol": "price volatility",
}

# Revenue variable name for quantile figure copying
REVENUE_VAR = {
    "1": "avg_rev_last_1000",
    "2": "avg_rev_last_1000",
    "3a": "avg_rev_last_1000",
    "3b": "avg_rev_last_1000",
    "4a": "mean_platform_revenue",
    "4b": "mean_platform_revenue",
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
    if p is None:
        return "N/A"
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
        lines.append(r"\label{tab:exp%s_coef_%s}" % (EXP_FILE_NUM[exp_num], response))
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
    lines.append(r"\label{tab:exp%s_fit}" % EXP_FILE_NUM[exp_num])
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
            "%% No significant effects at p<0.05 for Experiment %s" % exp_num
        )

    label = "tab:exp%s_sig" % EXP_FILE_NUM[exp_num]
    lines = []
    lines.append(r"\begin{longtable}{llrrr}")
    lines.append(
        r"\caption{Experiment %s: All significant effects ($p < 0.05$) "
        r"across response variables, ranked by $|t|$.}" % roman
    )
    lines.append(r"\label{%s}\\" % label)
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Response} & \textbf{Effect} "
        r"& \textbf{Coeff.} & \textbf{$|t|$} & \textbf{$p$-value} \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\multicolumn{5}{l}{\small\emph{Table~\ref{%s} continued}} \\" % label)
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Response} & \textbf{Effect} "
        r"& \textbf{Coeff.} & \textbf{$|t|$} & \textbf{$p$-value} \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{r}{\small\emph{Continued on next page}} \\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    for row in sig_rows:
        resp_name = readable_response(row["response"])
        effect_name = readable_effect_name(row["effect"])
        lines.append(
            f"{resp_name} & {effect_name} "
            f"& {row['estimate']:.4f} & {row['t_abs']:.2f} & {format_pval(row['p_value'])} \\\\"
        )

    lines.append(r"\end{longtable}")

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

    # Cap at top 10 effects to keep table compact
    sig_rows = sig_rows[:10]

    if not sig_rows:
        return (
            "%% No significant effects at p<0.05 for Experiment %s %s"
            % (exp_num, response_key)
        )

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Experiment %s: Significant effects for %s ($p < 0.05$), ranked by $|t|$.}"
        % (roman, resp_label)
    )
    lines.append(r"\label{tab:exp%s_ranked_%s}" % (EXP_FILE_NUM[exp_num], short_key))
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
# Robustness table generators
# ---------------------------------------------------------------------------

def load_robust_data(exp_num):
    """Load robust_results.json for an experiment."""
    path = os.path.join(RESULTS_DIR, f"exp{exp_num}", "robust", "robust_results.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"exp{exp_num}: {path} not found")
    with open(path) as f:
        return json.load(f)


def generate_adequacy_table(exp_num, robust_data):
    """Generate model adequacy table from robustness diagnostics.

    Columns: Response | R² | Pred-R² | Gap | LGBM R² | LOF p
    Uses KEY_RESPONSES for the 3 primary outcomes per experiment.
    """
    roman = EXP_ROMAN[exp_num]
    key_resp = KEY_RESPONSES.get(exp_num, {})

    # Build lookup from summary table list
    summary_lookup = {}
    for item in robust_data.get("summary", {}).get("table", []):
        summary_lookup[item["response"]] = item

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Experiment %s: Model adequacy diagnostics.}" % roman
    )
    lines.append(r"\label{tab:exp%s_adequacy}" % EXP_FILE_NUM[exp_num])
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Response} & \textbf{$R^2$} & \textbf{Pred-$R^2$} "
        r"& \textbf{Gap} & \textbf{LGBM $R^2$} & \textbf{LOF $p$} \\"
    )
    lines.append(r"\midrule")

    for _short_key, response_key in key_resp.items():
        resp_name = readable_response(response_key)
        press = robust_data.get("press", {}).get(response_key, {})
        lgbm = robust_data.get("lightgbm", {}).get(response_key, {})
        lof = robust_data.get("lack_of_fit", {}).get(response_key, {})

        r2 = press.get("r_squared", float("nan"))
        pred_r2 = press.get("predicted_r_squared", float("nan"))
        gap = press.get("gap_r2_pred_r2", float("nan"))
        lgbm_r2 = lgbm.get("lgbm_cv_r2", float("nan"))
        lof_p = lof.get("p_lack_of_fit", float("nan"))

        lines.append(
            f"{resp_name} & {r2:.4f} & {pred_r2:.4f} "
            f"& {gap:.4f} & {lgbm_r2:.4f} & {format_pval(lof_p)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\par\smallskip\footnotesize "
        r"Gap $= R^2 - \text{Pred-}R^2$. "
        r"LGBM $R^2$: five-fold cross-validated LightGBM. "
        r"LOF $p$: lack-of-fit $F$-test."
    )
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_inference_robust_table(exp_num, robust_data):
    """Generate inference robustness table from multiplicity and HC3 data.

    Columns: Response | OLS Sig | HC3 Flipped | Holm Sig | BH Sig
    Parses multiplicity.tests to compute per-response counts.
    """
    roman = EXP_ROMAN[exp_num]
    key_resp = KEY_RESPONSES.get(exp_num, {})
    key_response_names = set(key_resp.values())

    # Count per-response multiplicity from tests array
    from collections import defaultdict
    resp_counts = defaultdict(lambda: {"ols": 0, "holm": 0, "bh": 0, "total": 0})
    for t in robust_data.get("multiplicity", {}).get("tests", []):
        resp = t["label"].split("|")[0]
        resp_counts[resp]["total"] += 1
        if t.get("raw_sig"):
            resp_counts[resp]["ols"] += 1
        if t.get("holm_sig"):
            resp_counts[resp]["holm"] += 1
        if t.get("bh_sig"):
            resp_counts[resp]["bh"] += 1

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Experiment %s: Inference robustness under heteroskedasticity and multiple testing corrections.}"
        % roman
    )
    lines.append(r"\label{tab:exp%s_inference}" % EXP_FILE_NUM[exp_num])
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Response} & \textbf{OLS Sig} & \textbf{HC3 Flipped} "
        r"& \textbf{Holm Sig} & \textbf{BH Sig} \\"
    )
    lines.append(r"\midrule")

    for _short_key, response_key in key_resp.items():
        resp_name = readable_response(response_key)
        hc3_resp = robust_data.get("hc3", {}).get(response_key, {})
        n_flipped = hc3_resp.get("n_flipped", 0)
        n_terms = len(hc3_resp.get("terms", {}))
        rc = resp_counts.get(response_key, {"ols": 0, "holm": 0, "bh": 0, "total": 0})

        lines.append(
            f"{resp_name} & {rc['ols']}/{rc['total']} "
            f"& {n_flipped}/{n_terms} "
            f"& {rc['holm']}/{rc['total']} & {rc['bh']}/{rc['total']} \\\\"
        )

    # Add totals row
    m = robust_data.get("multiplicity", {})
    total_flipped = sum(
        robust_data.get("hc3", {}).get(r, {}).get("n_flipped", 0)
        for r in robust_data.get("hc3", {})
    )
    total_terms = sum(
        len(robust_data.get("hc3", {}).get(r, {}).get("terms", {}))
        for r in robust_data.get("hc3", {})
    )
    lines.append(r"\midrule")
    lines.append(
        r"\textit{All responses} & %d/%d & %d/%d & %d/%d & %d/%d \\"
        % (
            m.get("n_raw_sig", 0), m.get("n_tests", 0),
            total_flipped, total_terms,
            m.get("n_holm_sig", 0), m.get("n_tests", 0),
            m.get("n_bh_sig", 0), m.get("n_tests", 0),
        )
    )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\par\smallskip\footnotesize "
        r"OLS Sig: $p < 0.05$ under OLS standard errors. "
        r"HC3 Flipped: effects changing significance under HC3 robust standard errors. "
        r"Holm/BH Sig: effects surviving Holm--Bonferroni/Benjamini--Hochberg correction."
    )
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure copying
# ---------------------------------------------------------------------------

def copy_figures(exp_num):
    """Copy the most important publication figures for an experiment.

    For each experiment, copy main effects, interaction, and pareto figures
    for each key response variable (rev, vol; poa for exp4a/4b).
    """
    key_resp = KEY_RESPONSES.get(exp_num, {})
    exp_dir = os.path.join(RESULTS_DIR, f"exp{exp_num}")
    fn = EXP_FILE_NUM[exp_num]  # paper-facing file prefix

    copy_pairs = []
    for short_key, var_name in key_resp.items():
        copy_pairs.extend([
            (
                os.path.join(exp_dir, "main_effects", f"main_effects_{var_name}.png"),
                os.path.join(FIGURES_DIR, f"e{fn}_main_{short_key}.png"),
            ),
            (
                os.path.join(exp_dir, "interaction_plots", f"interactions_{var_name}.png"),
                os.path.join(FIGURES_DIR, f"e{fn}_int_{short_key}.png"),
            ),
            (
                os.path.join(exp_dir, "pareto_charts", f"pareto_{var_name}.png"),
                os.path.join(FIGURES_DIR, f"e{fn}_pareto_{short_key}.png"),
            ),
        ])

    # Quantile regression coefficient plots
    rev_var = REVENUE_VAR.get(exp_num, "avg_rev_last_1000")
    copy_pairs.append((
        os.path.join(exp_dir, "robust", f"quantile_coefs_{rev_var}.png"),
        os.path.join(FIGURES_DIR, f"e{fn}_quantile_rev.png"),
    ))

    for src, dst in copy_pairs:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied {src} -> {dst}")
        else:
            raise FileNotFoundError(f"Missing figure source: {src}")


# ---------------------------------------------------------------------------
# Per-experiment processing
# ---------------------------------------------------------------------------

def process_experiment(exp_num):
    """Process a single experiment: generate tables and copy figures."""
    json_path = os.path.join(RESULTS_DIR, f"exp{exp_num}", "estimation_results.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"exp{exp_num}: {json_path} not found")

    with open(json_path) as f:
        data = json.load(f)

    responses_data = data.get("responses", {})
    if not responses_data:
        raise ValueError(f"exp{exp_num}: no response data in {json_path}")

    fn = EXP_FILE_NUM[exp_num]  # paper-facing file prefix

    # 1. Coefficient tables (one per response, combined into one file)
    coef_tex = generate_coefficients_tables(exp_num, responses_data)
    coef_path = os.path.join(TABLES_DIR, f"exp{fn}_coefficients.tex")
    with open(coef_path, "w") as f:
        f.write(coef_tex + "\n")
    print(f"  Wrote {coef_path}")

    # 2. Model fit summary table
    fit_tex = generate_model_fit_table(exp_num, responses_data)
    fit_path = os.path.join(TABLES_DIR, f"exp{fn}_model_fit.tex")
    with open(fit_path, "w") as f:
        f.write(fit_tex + "\n")
    print(f"  Wrote {fit_path}")

    # 3. Significant effects table (cross-response)
    sig_tex = generate_significant_effects_table(exp_num, responses_data)
    sig_path = os.path.join(TABLES_DIR, f"exp{fn}_significant.tex")
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
            ranked_path = os.path.join(TABLES_DIR, f"exp{fn}_ranked_{short_key}.tex")
            with open(ranked_path, "w") as f:
                f.write(ranked_tex + "\n")
            print(f"  Wrote {ranked_path}")

    # 5. Robustness tables (adequacy + inference)
    robust_data = load_robust_data(exp_num)
    adequacy_tex = generate_adequacy_table(exp_num, robust_data)
    adequacy_path = os.path.join(TABLES_DIR, f"exp{fn}_adequacy.tex")
    with open(adequacy_path, "w") as f:
        f.write(adequacy_tex + "\n")
    print(f"  Wrote {adequacy_path}")

    inference_tex = generate_inference_robust_table(exp_num, robust_data)
    inference_path = os.path.join(TABLES_DIR, f"exp{fn}_inference_robust.tex")
    with open(inference_path, "w") as f:
        f.write(inference_tex + "\n")
    print(f"  Wrote {inference_path}")

    # 6. Copy publication figures
    copy_figures(exp_num)


# ---------------------------------------------------------------------------
# Paper numbers auto-generation
# ---------------------------------------------------------------------------

# Roman numeral for LaTeX command names
_CMD_ROMAN = {"1": "OneA", "2": "OneB", "3a": "TwoA", "3b": "TwoB", "4a": "ThreeA", "4b": "ThreeB"}

# Short keys for response variable command names
_CMD_RESPONSE_SHORT = {
    "avg_rev_last_1000": "Rev",
    "time_to_converge": "Conv",
    "no_sale_rate": "NoSale",
    "price_volatility": "Vol",
    "winner_entropy": "Ent",
    "mean_platform_revenue": "Rev",
    "mean_liquid_welfare": "Welfare",
    "mean_effective_poa": "PoAGreedy",
    "mean_effective_poa_lp": "PoA",
    "mean_budget_utilization": "BudUtil",
    "mean_bid_to_value": "BTV",
    "mean_allocative_efficiency": "AllocEff",
    "mean_dual_cv": "DualCV",
    "mean_no_sale_rate": "NoSale",
    "mean_winner_entropy": "Ent",
}


def _format_p_latex(p):
    """Format p-value for LaTeX inline use.

    Returns a string suitable for $p \\ExpThreeAXxxPFmt$ usage:
    - '= 0.004' for moderate p
    - '< 0.001' for small p
    - '< 10^{-15}' for very small p
    """
    import math
    p = float(p)
    if p != p or p >= 1:  # NaN guard
        return "= \\text{NA}"
    if p >= 0.001:
        return f"= {p:.3f}"
    if p < 1e-4:
        exp = int(math.ceil(math.log10(max(p, 1e-300))))
        return f"< 10^{{{exp}}}"
    return "< 0.001"


# Exp1 coefficient macros: (response_key, coded_factor) → macro suffix
_EXP1_COEF_MACROS = {
    # Revenue: interaction t-stats referenced in res1.tex
    ("avg_rev_last_1000", "auction_type:n_bidders"): "RevAuctionxNbid",
    ("avg_rev_last_1000", "auction_type:gamma"): "RevAuctionxGamma",
    # Note: auction_type main effect T/AbsT/PFmt generated in auction format section below
}

# Exp2 coefficient macros: (response_key, coded_factor) → macro suffix
_EXP2_COEF_MACROS = {
    # End-state revenue: n_bidders and auction_type individual t-stats
    ("avg_rev_last_1000", "n_bidders"): "RevNbid",
    # Lifetime revenue: individual factor t-stats and interaction
    ("avg_rev_all", "auction_type"): "AllRevAuction",
    ("avg_rev_all", "n_bidders"): "AllRevNbid",
    ("avg_rev_all", "state_info"): "AllRevState",
    ("avg_rev_all", "auction_type:n_bidders"): "AllRevAuctionxNbid",
}

# Exp4a coefficient macros: (response_key, coded_factor) → macro suffix
_EXP4A_COEF_MACROS = {
    # Allocative efficiency: objective × n_bidders interaction
    ("mean_allocative_efficiency", "objective:n_bidders"): "AllocEffObjxNbid",
    # Platform revenue: individual factor t-stats
    ("mean_platform_revenue", "n_bidders"): "RevNbid",
    ("mean_platform_revenue", "objective"): "RevObj",
    ("mean_platform_revenue", "objective:n_bidders"): "RevObjxNbid",
    # Cross-episode drift
    ("cross_episode_drift", "auction_type"): "DriftAuction",
    ("cross_episode_drift", "n_bidders"): "DriftNbid",
    # Warm-start benefit
    ("warm_start_benefit", "objective"): "WarmObj",
    ("warm_start_benefit", "auction_type:objective"): "WarmObjxAuction",
    # Inter-episode volatility
    ("inter_episode_volatility", "objective"): "IEVolObj",
}

# Exp4b coefficient macros: (response_key, coded_factor) → macro suffix
_EXP4B_COEF_MACROS = {
    # Platform revenue: individual factor t-stats
    ("mean_platform_revenue", "n_bidders"): "RevNbid",
    ("mean_platform_revenue", "aggressiveness"): "RevAggr",
    # Cross-episode drift
    ("cross_episode_drift", "auction_type"): "DriftAuction",
    ("cross_episode_drift", "n_bidders"): "DriftNbid",
    # Warm-start benefit
    ("warm_start_benefit", "aggressiveness"): "WarmAggr",
    # Inter-episode volatility
    ("inter_episode_volatility", "aggressiveness"): "IEVolAggr",
}


def _generate_generic_coefficient_macros(responses_data, lines, macros_dict, prefix, label):
    """Generate \\newcommand macros for specific coefficients in any experiment.

    For each entry in macros_dict, writes:
      \\{prefix}{suffix}T       - signed t-value
      \\{prefix}{suffix}AbsT    - |t| value
      \\{prefix}{suffix}PFmt    - formatted p-value
    """
    lines.append(f"% {label} per-coefficient macros")
    for (resp_key, factor_key), suffix in macros_dict.items():
        rdata = responses_data.get(resp_key, {})
        coeffs = rdata.get("coefficients", {})
        coef = coeffs.get(factor_key, {})
        t_val = coef.get("t_value", 0)
        p_val = coef.get("p_value", 1)
        if t_val != t_val:  # NaN
            t_val = 0
        if p_val != p_val:  # NaN
            p_val = 1
        lines.append(f"\\newcommand{{\\{prefix}{suffix}T}}{{{t_val:.1f}}}")
        lines.append(f"\\newcommand{{\\{prefix}{suffix}AbsT}}{{{abs(t_val):.1f}}}")
        lines.append(f"\\newcommand{{\\{prefix}{suffix}PFmt}}{{{_format_p_latex(p_val)}}}")



def _generate_exp2_lifetime_macros(responses_data, lines, exp_num):
    """Generate Exp2 lifetime revenue macros: per-format means, premiums, per-cell means."""
    import pandas as pd
    prefix = f"Exp{_CMD_ROMAN['2']}"
    lines.append("% Exp2 lifetime revenue macros")

    # R² for avg_rev_all model
    all_rev = responses_data.get("avg_rev_all", {})
    r2 = all_rev.get("r_squared", 0)
    lines.append(f"\\newcommand{{\\{prefix}AllRevRsq}}{{{r2:.3f}}}")

    csv_path = os.path.join(RESULTS_DIR, f"exp{exp_num}", "data.csv")
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)

    # Per-format means for both lifetime and end-state
    for rev_col, tag in [("avg_rev_all", "All"), ("avg_rev_last_1000", "End")]:
        if rev_col not in df.columns:
            continue
        if "auction_type" in df.columns:
            fpa_mean = df[df["auction_type"] == "first"][rev_col].mean()
            spa_mean = df[df["auction_type"] == "second"][rev_col].mean()
            lines.append(f"\\newcommand{{\\{prefix}{tag}MeanFPA}}{{{fpa_mean:.3f}}}")
            lines.append(f"\\newcommand{{\\{prefix}{tag}MeanSPA}}{{{spa_mean:.3f}}}")

    # Learning premiums (lifetime - end-state)
    if all(c in df.columns for c in ["avg_rev_all", "avg_rev_last_1000", "auction_type"]):
        for fmt, label in [("first", "FPA"), ("second", "SPA")]:
            sub = df[df["auction_type"] == fmt]
            premium = sub["avg_rev_all"].mean() - sub["avg_rev_last_1000"].mean()
            lines.append(f"\\newcommand{{\\{prefix}{label}Premium}}{{{premium:.3f}}}")

    # Per-cell means (auction_type × n_bidders) for lifetime revenue
    if all(c in df.columns for c in ["avg_rev_all", "auction_type", "n_bidders"]):
        for fmt, fmt_label in [("first", "FPA"), ("second", "SPA")]:
            for nb, nb_label in [(2, "TwoBid"), (4, "FourBid")]:
                sub = df[(df["auction_type"] == fmt) & (df["n_bidders"] == nb)]
                if len(sub) > 0:
                    mean_val = sub["avg_rev_all"].mean()
                    lines.append(
                        f"\\newcommand{{\\{prefix}All{fmt_label}{nb_label}}}"
                        f"{{{mean_val:.3f}}}"
                    )


def _generate_exp4a_detail_macros(responses_data, lines, exp_num):
    """Generate Exp4a detail macros: efficiency ranges, drift diagnostics, R² for welfare/drift."""
    import pandas as pd
    prefix = f"Exp{_CMD_ROMAN['4a']}"
    lines.append("% Exp4a detail macros (efficiency, drift, welfare)")

    # R² for liquid welfare and cross-episode drift
    welfare_data = responses_data.get("mean_liquid_welfare", {})
    drift_data = responses_data.get("cross_episode_drift", {})
    welfare_r2 = welfare_data.get("r_squared", 0)
    drift_r2 = drift_data.get("r_squared", 0)
    drift_f_p = drift_data.get("f_pvalue", 1)
    lines.append(f"\\newcommand{{\\{prefix}WelfareRsq}}{{{welfare_r2:.2f}}}")
    lines.append(f"\\newcommand{{\\{prefix}WelfareRsqPct}}{{{welfare_r2*100:.0f}}}")
    lines.append(f"\\newcommand{{\\{prefix}DriftRsq}}{{{drift_r2:.2f}}}")
    lines.append(f"\\newcommand{{\\{prefix}DriftRsqPct}}{{{drift_r2*100:.0f}}}")
    lines.append(f"\\newcommand{{\\{prefix}DriftFP}}{{{drift_f_p:.2f}}}")
    lines.append(f"\\newcommand{{\\{prefix}DriftFPFmt}}{{{_format_p_latex(drift_f_p)}}}")

    csv_path = os.path.join(RESULTS_DIR, f"exp{exp_num}", "data.csv")
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)

    # Efficiency ranges by objective × n_bidders (cell means, not individual runs)
    if all(c in df.columns for c in ["mean_allocative_efficiency", "objective", "n_bidders"]):
        factors = [c for c in df.columns if c.endswith("_coded")]
        raw_factors = [c.replace("_coded", "") for c in factors]
        # Use raw factor columns that exist
        grp_cols = [c for c in raw_factors if c in df.columns]
        if not grp_cols:
            grp_cols = ["auction_type", "objective", "n_bidders", "budget_multiplier", "reserve_price", "sigma"]
            grp_cols = [c for c in grp_cols if c in df.columns]
        util_cells = df[df["objective"] == "utility_max"].groupby(grp_cols)["mean_allocative_efficiency"].mean()
        val4_cells = df[(df["objective"] == "value_max") & (df["n_bidders"] == 4)].groupby(
            [c for c in grp_cols if c not in ["objective", "n_bidders"]]
        )["mean_allocative_efficiency"].mean()
        if len(util_cells) > 0:
            lines.append(f"\\newcommand{{\\{prefix}UtilEffLow}}{{{util_cells.min():.2f}}}")
            lines.append(f"\\newcommand{{\\{prefix}UtilEffHigh}}{{{util_cells.max():.2f}}}")
        if len(val4_cells) > 0:
            lines.append(f"\\newcommand{{\\{prefix}ValFourEffLow}}{{{val4_cells.min():.2f}}}")
            lines.append(f"\\newcommand{{\\{prefix}ValFourEffHigh}}{{{val4_cells.max():.2f}}}")


def generate_paper_numbers():
    """Generate paper/numbers.tex with \\newcommand definitions for all statistics.

    Extracts from estimation_results.json and robust/robust_results.json:
    - R², Adj-R², F-stat, n_obs per response
    - PRESS gaps, LGBM R², LOF p-values
    - Multiplicity counts (Holm, BH survivors)
    - Top-ranked |t|-statistics for key responses
    """
    lines = ["% Auto-generated by generate_tables.py — do not edit manually"]
    lines.append("% Regenerate with: make tables")
    lines.append("")

    for exp_num in ["1", "2", "3a", "3b", "4a", "4b"]:
        roman = _CMD_ROMAN[exp_num]

        # Load estimation results
        est_path = os.path.join(RESULTS_DIR, f"exp{exp_num}", "estimation_results.json")
        if not os.path.exists(est_path):
            print(f"  Skipping exp{exp_num} numbers: {est_path} not found")
            continue

        with open(est_path) as f:
            est_data = json.load(f)

        responses_data = est_data.get("responses", {})
        lines.append(f"% === Experiment {exp_num} ===")

        # n_obs (same across all responses)
        first_resp = next(iter(responses_data.values()), {})
        n_obs = first_resp.get("n_obs", 0)
        lines.append(f"\\newcommand{{\\Exp{roman}Nobs}}{{{n_obs}}}")

        # Per-response R², top |t|
        key_resp = KEY_RESPONSES.get(exp_num, {})
        for short_key, response_key in key_resp.items():
            rdata = responses_data.get(response_key, {})
            cmd_suffix = short_key.capitalize()  # Rev, Reg, Vol

            r2 = rdata.get("r_squared", 0)
            adj_r2 = rdata.get("adj_r_squared", 0)
            lines.append(f"\\newcommand{{\\Exp{roman}{cmd_suffix}Rsq}}{{{r2:.3f}}}")
            lines.append(f"\\newcommand{{\\Exp{roman}{cmd_suffix}AdjRsq}}{{{adj_r2:.3f}}}")

            # Top effect |t|
            coefficients = rdata.get("coefficients", {})
            if coefficients:
                top_effects = sorted(
                    coefficients.items(),
                    key=lambda x: abs(x[1]["t_value"]),
                    reverse=True
                )
                if top_effects:
                    top_name, top_vals = top_effects[0]
                    top_t = abs(top_vals["t_value"])
                    top_readable = readable_effect_name(top_name).replace("$", "").replace("\\", "")
                    lines.append(f"\\newcommand{{\\Exp{roman}{cmd_suffix}TopT}}{{{top_t:.1f}}}")

                # Top-3 factor names and |t|-values (for cross-experiment tables)
                if short_key == "rev":
                    for rank in range(min(3, len(top_effects))):
                        eff_name, eff_vals = top_effects[rank]
                        eff_t = abs(eff_vals["t_value"])
                        eff_readable = readable_effect_name(eff_name)
                        ordinal = ["One", "Two", "Three"][rank]
                        lines.append(
                            f"\\newcommand{{\\Exp{roman}RevFactor{ordinal}}}"
                            f"{{{eff_readable}}}"
                        )
                        lines.append(
                            f"\\newcommand{{\\Exp{roman}RevFactorT{ordinal}}}"
                            f"{{{eff_t:.1f}}}"
                        )

                    # Percentage effect of top revenue factor
                    top_coeff = top_effects[0][1]["estimate"]
                    effect_size = 2 * abs(top_coeff)
                    rev_var = REVENUE_VAR.get(exp_num, "avg_rev_last_1000")
                    csv_path = os.path.join(RESULTS_DIR, f"exp{exp_num}", "data.csv")
                    if os.path.exists(csv_path):
                        import pandas as pd
                        df_tmp = pd.read_csv(csv_path)
                        grand_mean = df_tmp[rev_var].mean()
                        if grand_mean != 0:
                            pct_effect = effect_size / grand_mean * 100
                            lines.append(
                                f"\\newcommand{{\\Exp{roman}RevTopPctEffect}}"
                                f"{{{pct_effect:.1f}}}"
                            )
                        # Ratio of top-1 effect to auction format effect
                        auction_coeff = coefficients.get(
                            "auction_type_coded", coefficients.get("auction_type", {})
                        ).get("estimate", None)
                        if auction_coeff is not None and abs(auction_coeff) > 0:
                            ratio = abs(top_coeff) / abs(auction_coeff)
                            lines.append(
                                f"\\newcommand{{\\Exp{roman}RevTopVsAuctionRatio}}"
                                f"{{{ratio:.1f}}}"
                            )

        # R² range across all responses
        all_r2 = [rd.get("r_squared", 0) for rd in responses_data.values()]
        if all_r2:
            lines.append(f"\\newcommand{{\\Exp{roman}RsqMin}}{{{min(all_r2):.2f}}}")
            lines.append(f"\\newcommand{{\\Exp{roman}RsqMax}}{{{max(all_r2):.2f}}}")

        # Load robustness results
        try:
            robust_data = load_robust_data(exp_num)
        except FileNotFoundError:
            robust_data = None
        if robust_data:
            # LGBM R² for key responses
            lgbm = robust_data.get("lightgbm", {})
            for short_key, response_key in key_resp.items():
                cmd_suffix = short_key.capitalize()
                lgbm_r2 = lgbm.get(response_key, {}).get("lgbm_cv_r2", 0)
                lines.append(f"\\newcommand{{\\Exp{roman}{cmd_suffix}LGBMRsq}}{{{lgbm_r2:.3f}}}")

            # PRESS gaps for key responses
            press = robust_data.get("press", {})
            for short_key, response_key in key_resp.items():
                cmd_suffix = short_key.capitalize()
                gap = press.get(response_key, {}).get("gap_r2_pred_r2", 0)
                pred_r2 = press.get(response_key, {}).get("predicted_r_squared", 0)
                lines.append(f"\\newcommand{{\\Exp{roman}{cmd_suffix}PredRsq}}{{{pred_r2:.3f}}}")
                lines.append(f"\\newcommand{{\\Exp{roman}{cmd_suffix}PRESSGap}}{{{gap:.3f}}}")

            # PRESS gap range
            all_gaps = [v.get("gap_r2_pred_r2", 0) for v in press.values()]
            if all_gaps:
                lines.append(f"\\newcommand{{\\Exp{roman}PRESSGapMin}}{{{min(all_gaps):.3f}}}")
                lines.append(f"\\newcommand{{\\Exp{roman}PRESSGapMax}}{{{max(all_gaps):.3f}}}")

            # Multiplicity
            mult = robust_data.get("multiplicity", {})
            n_tests = mult.get("n_tests", 0)
            n_raw = mult.get("n_raw_sig", 0)
            n_holm = mult.get("n_holm_sig", 0)
            n_bh = mult.get("n_bh_sig", 0)
            lines.append(f"\\newcommand{{\\Exp{roman}NTests}}{{{n_tests}}}")
            lines.append(f"\\newcommand{{\\Exp{roman}NRawSig}}{{{n_raw}}}")
            lines.append(f"\\newcommand{{\\Exp{roman}NHolmSig}}{{{n_holm}}}")
            lines.append(f"\\newcommand{{\\Exp{roman}NBHSig}}{{{n_bh}}}")
            if n_raw > 0:
                bh_pct = 100 * n_bh / n_raw
                lines.append(f"\\newcommand{{\\Exp{roman}BHPct}}{{{bh_pct:.0f}}}")

            # LGBM R² range
            all_lgbm = [v.get("lgbm_cv_r2", 0) for v in lgbm.values()]
            if all_lgbm:
                lines.append(f"\\newcommand{{\\Exp{roman}LGBMRsqMin}}{{{min(all_lgbm):.2f}}}")
                lines.append(f"\\newcommand{{\\Exp{roman}LGBMRsqMax}}{{{max(all_lgbm):.2f}}}")

        # --- Auction format t-stat and mean revenue by format ---
        import pandas as pd
        csv_path = os.path.join(RESULTS_DIR, f"exp{exp_num}", "data.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            rev_var = REVENUE_VAR.get(exp_num, "avg_rev_last_1000")

            # Auction type t-stat for revenue
            rev_data = responses_data.get(rev_var, {})
            coeffs = rev_data.get("coefficients", {})
            auction_coef = coeffs.get("auction_type_coded", coeffs.get("auction_type", {}))
            if auction_coef:
                auction_t_raw = auction_coef.get("t_value", 0)
                auction_t = abs(auction_t_raw)
                auction_p = auction_coef.get("p_value", 1)
                lines.append(f"\\newcommand{{\\Exp{roman}RevAuctionT}}{{{auction_t:.1f}}}")
                lines.append(f"\\newcommand{{\\Exp{roman}RevAuctionAbsT}}{{{auction_t:.1f}}}")
                lines.append(f"\\newcommand{{\\Exp{roman}RevAuctionPFmt}}{{{_format_p_latex(auction_p)}}}")

            # Mean revenue by auction format
            if "auction_type" in df.columns and rev_var in df.columns:
                fpa = df[df["auction_type"] == "first"][rev_var].mean()
                spa = df[df["auction_type"] == "second"][rev_var].mean()
                if spa != 0:
                    gap_pct = (fpa - spa) / spa * 100
                else:
                    gap_pct = 0.0
                lines.append(f"\\newcommand{{\\Exp{roman}RevMeanFPA}}{{{fpa:.3f}}}")
                lines.append(f"\\newcommand{{\\Exp{roman}RevMeanSPA}}{{{spa:.3f}}}")
                lines.append(f"\\newcommand{{\\Exp{roman}RevGapPct}}{{{gap_pct:.1f}}}")

            # Convergence summary macros
            conv_col = "time_to_converge"
            if conv_col in df.columns:
                med = df[conv_col].median()
                mean_val = df[conv_col].mean()
                lines.append(f"\\newcommand{{\\Exp{roman}ConvMedian}}{{{med:,.0f}}}")
                lines.append(f"\\newcommand{{\\Exp{roman}ConvMean}}{{{mean_val:,.0f}}}")
            else:
                lines.append(f"\\newcommand{{\\Exp{roman}ConvMedian}}{{N/A}}")
                lines.append(f"\\newcommand{{\\Exp{roman}ConvMean}}{{N/A}}")

        # --- Per-coefficient macros for inline t-stats ---
        prefix = f"Exp{roman}"
        if exp_num == "1":
            _generate_generic_coefficient_macros(
                responses_data, lines, _EXP1_COEF_MACROS, prefix, "Exp1")
        elif exp_num == "2":
            _generate_generic_coefficient_macros(
                responses_data, lines, _EXP2_COEF_MACROS, prefix, "Exp2")
            # Exp2 lifetime revenue macros
            _generate_exp2_lifetime_macros(responses_data, lines, exp_num)
        elif exp_num == "4a":
            _generate_generic_coefficient_macros(
                responses_data, lines, _EXP4A_COEF_MACROS, prefix, "Exp4a")
            _generate_exp4a_detail_macros(responses_data, lines, exp_num)
        elif exp_num == "4b":
            _generate_generic_coefficient_macros(
                responses_data, lines, _EXP4B_COEF_MACROS, prefix, "Exp4b")

        # --- Cross-experiment margin ratio (top-1 / top-2 |t|) for revenue ---
        rev_var = REVENUE_VAR.get(exp_num, "avg_rev_last_1000")
        rev_rdata = responses_data.get(rev_var, {})
        rev_coeffs = rev_rdata.get("coefficients", {})
        if rev_coeffs:
            sorted_by_t = sorted(
                rev_coeffs.items(),
                key=lambda x: abs(x[1]["t_value"]),
                reverse=True
            )
            if len(sorted_by_t) >= 2:
                t1 = abs(sorted_by_t[0][1]["t_value"])
                t2 = abs(sorted_by_t[1][1]["t_value"])
                if t2 > 0:
                    margin = t1 / t2
                    lines.append(
                        f"\\newcommand{{\\Exp{roman}RevTopMarginRatio}}"
                        f"{{{margin:.1f}}}"
                    )

        lines.append("")

    numbers_path = os.path.join("paper", "numbers.tex")
    with open(numbers_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {numbers_path} ({len(lines)} lines)")

    # --- Validation checks ---
    import pandas as pd
    errors = []
    for exp_num in ["1", "2", "3a", "3b", "4a", "4b"]:
        roman = _CMD_ROMAN[exp_num]
        csv_path = os.path.join(RESULTS_DIR, f"exp{exp_num}", "data.csv")
        est_path = os.path.join(RESULTS_DIR, f"exp{exp_num}", "estimation_results.json")
        if not os.path.exists(csv_path) or not os.path.exists(est_path):
            continue
        df = pd.read_csv(csv_path)
        with open(est_path) as f:
            est_data = json.load(f)
        responses_data = est_data.get("responses", {})
        first_resp = next(iter(responses_data.values()), {})
        json_nobs = first_resp.get("n_obs", 0)
        csv_nobs = len(df)
        if json_nobs != csv_nobs:
            errors.append(f"Exp{exp_num}: n_obs mismatch: JSON={json_nobs}, CSV={csv_nobs}")

        # Check LGBM vs OLS gap (diagnostic, not blocking)
        try:
            robust_data = load_robust_data(exp_num)
        except FileNotFoundError:
            continue
        lgbm = robust_data.get("lightgbm", {})
        for resp_key, rdata in responses_data.items():
            ols_r2 = rdata.get("r_squared", 0)
            lgbm_r2 = lgbm.get(resp_key, {}).get("lgbm_cv_r2", 0)
            gap = lgbm_r2 - ols_r2
            if gap > 0.05:
                print(f"  NOTE Exp{exp_num}: LGBM R² exceeds OLS R² by {gap:.3f} for {resp_key}",
                      file=sys.stderr)

    if errors:
        for e in errors:
            print(f"  ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    else:
        print("\n  All validation checks passed")

    return numbers_path


# ---------------------------------------------------------------------------
# Summary statistics table generator
# ---------------------------------------------------------------------------

# Factor columns per experiment (raw value columns, not coded)
_EXP_FACTOR_COLS = {
    "1": ["auction_type", "alpha", "gamma", "reserve_price", "init",
        "exploration", "asynchronous", "n_bidders", "info_feedback",
        "decay_type"],
    "2": ["auction_type", "n_bidders", "state_info", "eta"],
    "3a": ["auction_type", "n_bidders", "reserve_price",
        "eta", "exploration_intensity", "context_richness", "lam", "memory_decay"],
    "3b": ["auction_type", "n_bidders", "reserve_price",
        "eta", "exploration_intensity", "context_richness"],
    "4a": ["auction_type", "objective", "n_bidders"],
    "4b": ["auction_type", "aggressiveness", "n_bidders"],
}

# Response columns per experiment
_EXP_RESPONSE_COLS = {
    "1": ["avg_rev_last_1000", "time_to_converge",
        "no_sale_rate", "price_volatility", "winner_entropy"],
    "2": ["avg_rev_last_1000", "time_to_converge",
        "no_sale_rate", "price_volatility", "winner_entropy"],
    "3a": ["avg_rev_last_1000", "time_to_converge",
        "no_sale_rate", "price_volatility", "winner_entropy"],
    "3b": ["avg_rev_last_1000", "time_to_converge",
        "no_sale_rate", "price_volatility", "winner_entropy"],
    "4a": ["mean_platform_revenue", "mean_liquid_welfare", "mean_effective_poa_lp",
        "mean_budget_utilization", "mean_bid_to_value",
        "mean_allocative_efficiency", "mean_winner_entropy",
        "warm_start_benefit", "inter_episode_volatility",
        "bid_suppression_ratio", "cross_episode_drift"],
    "4b": ["mean_platform_revenue", "mean_liquid_welfare", "mean_effective_poa_lp",
        "mean_budget_utilization", "mean_bid_to_value",
        "mean_allocative_efficiency", "mean_winner_entropy",
        "warm_start_benefit", "inter_episode_volatility",
        "bid_suppression_ratio", "cross_episode_drift"],
}


def generate_summary_statistics(exp_num):
    """Generate a summary statistics table for an experiment.

    Reads data.csv and produces paper/tables/expN_summary.tex with
    factor parameters above the midrule and response variables below.
    """
    import pandas as pd

    csv_path = os.path.join(RESULTS_DIR, f"exp{exp_num}", "data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"exp{exp_num}: {csv_path} not found")

    df = pd.read_csv(csv_path)
    roman = EXP_ROMAN[exp_num]

    factor_cols = [c for c in _EXP_FACTOR_COLS.get(exp_num, []) if c in df.columns]
    response_cols = [c for c in _EXP_RESPONSE_COLS.get(exp_num, []) if c in df.columns]

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Summary Statistics for Experiment %s (%d observations).}"
        % (roman, len(df))
    )
    lines.append(r"\label{tab:summary_statistics_exp%s}" % EXP_FILE_NUM[exp_num])
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Variable} & \textbf{Mean} & \textbf{Std} "
        r"& \textbf{Min} & \textbf{Max} \\"
    )
    lines.append(r"\midrule")

    def _add_row(col_name, display_name, series):
        """Add a summary row, handling both numeric and categorical columns."""
        if series.dtype == object:
            # For categorical: show as fraction of most common value
            return  # Skip categorical in numeric summary
        mean_v = series.mean()
        std_v = series.std()
        min_v = series.min()
        max_v = series.max()
        # Format large numbers with commas
        if abs(mean_v) >= 1000:
            lines.append(
                f"{display_name} & {mean_v:,.0f} & {std_v:,.0f} "
                f"& {min_v:,.0f} & {max_v:,.0f} \\\\"
            )
        else:
            lines.append(
                f"{display_name} & {mean_v:.3f} & {std_v:.3f} "
                f"& {min_v:.3f} & {max_v:.3f} \\\\"
            )

    # Factor rows
    for col in factor_cols:
        display = READABLE_NAMES.get(col, col.replace("_", " ").title())
        if df[col].dtype == object:
            continue  # Skip categorical factors in numeric summary
        _add_row(col, display, df[col])

    lines.append(r"\midrule")

    # Response rows
    for col in response_cols:
        display = READABLE_RESPONSES.get(col, col.replace("_", " ").title())
        if col in df.columns:
            _add_row(col, display, df[col])

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\par\smallskip\footnotesize "
        r"Factor parameters (above the line) describe the experimental design; "
        r"response variables (below) are measured outcomes."
    )
    lines.append(r"\end{table}")

    out_path = os.path.join(TABLES_DIR, f"exp{EXP_FILE_NUM[exp_num]}_summary.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    for exp_num in ["1", "2", "3a", "3b", "4a", "4b"]:
        print(f"\nProcessing Experiment {exp_num}...")
        try:
            process_experiment(exp_num)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")

    # Copy calibration entropy figure if available
    cal_fig_src = os.path.join(RESULTS_DIR, "calibration", "entropy_comparison.png")
    cal_fig_dst = os.path.join(FIGURES_DIR, "calibration_entropy.png")
    if os.path.exists(cal_fig_src):
        shutil.copy2(cal_fig_src, cal_fig_dst)
        print(f"\n  Copied {cal_fig_src} -> {cal_fig_dst}")
    else:
        print(f"\n  NOTE: {cal_fig_src} not found (run make calibrate-multi to generate)")


    # Generate summary statistics tables
    for exp_num in ["1", "2", "3a", "3b", "4a", "4b"]:
        print(f"\nGenerating summary statistics for Experiment {exp_num}...")
        try:
            generate_summary_statistics(exp_num)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")

    # Generate paper numbers
    generate_paper_numbers()

    print("\nDone. Tables written to paper/tables/, figures copied to paper/figures/")


if __name__ == "__main__":
    main()
