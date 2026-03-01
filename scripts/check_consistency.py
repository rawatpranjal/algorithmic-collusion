#!/usr/bin/env python3
"""
Verify that paper numbers in paper/numbers.tex match the actual data.

Parses all \\newcommand{\\ExpN...}{value} macros from numbers.tex and cross-
references them against:
  - results/expN/data.csv              (row counts)
  - results/expN/estimation_results.json (R², adj-R², top |t|, n_obs)
  - results/expN/robust/robust_results.json (LGBM R², PRESS gaps, multiplicity)

Prints WARNING for any mismatch (tolerance: 0.001 for floats).
Also warns if LGBM R² exceeds OLS R² by more than 0.05 for any response.
Exits with code 0 if all checks pass, code 1 if any warnings.

Usage:
    python scripts/check_consistency.py
"""

import argparse
import csv
import json
import math
import os
import re
import sys

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUMBERS_TEX = os.path.join(BASE_DIR, "paper", "numbers.tex")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

FLOAT_TOL = 0.001
LGBM_EXCESS_THRESHOLD = 0.05

# Roman numeral mapping: command name prefix -> experiment number
ROMAN_TO_NUM = {"One": "1", "Two": "2", "Three": "3", "FourA": "4a", "FourB": "4b"}
NUM_TO_ROMAN = {"1": "One", "2": "Two", "3": "Three", "4a": "FourA", "4b": "FourB"}

# KEY_RESPONSES: short key -> response variable name per experiment
KEY_RESPONSES = {
    "1": {
        "rev": "avg_rev_last_1000",
        "reg": "avg_regret_of_seller",
        "vol": "price_volatility",
    },
    "2": {
        "rev": "avg_rev_last_1000",
        "reg": "excess_regret",
        "vol": "price_volatility",
    },
    "3": {
        "rev": "avg_rev_last_1000",
        "reg": "avg_regret_seller",
        "vol": "price_volatility",
    },
    "4a": {
        "rev": "mean_platform_revenue",
        "reg": "mean_effective_poa_lp",
        "vol": "mean_bid_to_value",
    },
    "4b": {
        "rev": "mean_platform_revenue",
        "reg": "mean_effective_poa_lp",
        "vol": "mean_bid_to_value",
    },
}

# Short key capitalized -> command suffix mapping
# Rev, Reg, Vol map to the response variable via KEY_RESPONSES
CMD_SUFFIX_TO_SHORT = {"Rev": "rev", "Reg": "reg", "Vol": "vol"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_numbers_tex(path):
    """Parse all \\newcommand definitions from numbers.tex.

    Returns a dict mapping command name (without backslash) to string value.
    E.g. {"ExpOneNobs": "2048", "ExpOneRevRsq": "0.809"}
    """
    macros = {}
    pattern = re.compile(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}")
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                macros[m.group(1)] = m.group(2)
    return macros


def parse_value(s):
    """Parse a macro value as int or float. Returns the parsed number."""
    s = s.strip()
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def values_match(expected, actual, tol=FLOAT_TOL):
    """Check whether two values match within tolerance for floats."""
    if isinstance(expected, str) or isinstance(actual, str):
        return str(expected) == str(actual)
    if isinstance(expected, int) and isinstance(actual, int):
        return expected == actual
    # Float comparison
    if math.isnan(expected) and math.isnan(actual):
        return True
    return abs(expected - actual) <= tol


def load_json(path):
    """Load a JSON file or return None if missing."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def count_csv_rows(path):
    """Count data rows in a CSV file (excluding header)."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return 0
        return sum(1 for _ in reader)


def fmt(val):
    """Format a value for display."""
    if isinstance(val, float):
        return f"{val:.6f}"
    return str(val)


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def check_nobs(exp_num, macros, est_data, csv_rows, warnings):
    """Check n_observations macro matches data.csv row count and estimation JSON."""
    roman = NUM_TO_ROMAN[exp_num]
    cmd = f"Exp{roman}Nobs"

    if cmd not in macros:
        warnings.append(f"[Exp{exp_num}] Missing macro \\{cmd}")
        return 0

    macro_val = parse_value(macros[cmd])

    checks = 0

    # Check against CSV row count
    if csv_rows is not None:
        checks += 1
        if not values_match(macro_val, csv_rows):
            warnings.append(
                f"[Exp{exp_num}] \\{cmd} = {macro_val} but data.csv has {csv_rows} rows"
            )

    # Check against estimation JSON n_obs
    if est_data:
        responses = est_data.get("responses", {})
        first_resp = next(iter(responses.values()), {})
        json_nobs = first_resp.get("n_obs")
        if json_nobs is not None:
            checks += 1
            if not values_match(macro_val, json_nobs):
                warnings.append(
                    f"[Exp{exp_num}] \\{cmd} = {macro_val} but "
                    f"estimation_results.json n_obs = {json_nobs}"
                )

    return checks


def check_r_squared(exp_num, macros, est_data, warnings):
    """Check R² and adj-R² macros against estimation_results.json."""
    roman = NUM_TO_ROMAN[exp_num]
    key_resp = KEY_RESPONSES.get(exp_num, {})
    responses = est_data.get("responses", {}) if est_data else {}

    checks = 0

    for short_key, response_key in key_resp.items():
        cmd_suffix = short_key.capitalize()  # Rev, Reg, Vol
        rdata = responses.get(response_key, {})

        # R²
        cmd_r2 = f"Exp{roman}{cmd_suffix}Rsq"
        if cmd_r2 in macros:
            macro_val = parse_value(macros[cmd_r2])
            json_val = round(rdata.get("r_squared", float("nan")), 3)
            checks += 1
            if not values_match(macro_val, json_val):
                warnings.append(
                    f"[Exp{exp_num}] \\{cmd_r2} = {fmt(macro_val)} but "
                    f"JSON r_squared = {fmt(json_val)}"
                )

        # Adj R²
        cmd_adj = f"Exp{roman}{cmd_suffix}AdjRsq"
        if cmd_adj in macros:
            macro_val = parse_value(macros[cmd_adj])
            json_val = round(rdata.get("adj_r_squared", float("nan")), 3)
            checks += 1
            if not values_match(macro_val, json_val):
                warnings.append(
                    f"[Exp{exp_num}] \\{cmd_adj} = {fmt(macro_val)} but "
                    f"JSON adj_r_squared = {fmt(json_val)}"
                )

        # Top |t|
        cmd_top_t = f"Exp{roman}{cmd_suffix}TopT"
        if cmd_top_t in macros:
            macro_val = parse_value(macros[cmd_top_t])
            coefficients = rdata.get("coefficients", {})
            if coefficients:
                top_t = max(abs(v["t_value"]) for v in coefficients.values())
                json_val = round(top_t, 1)
                checks += 1
                if not values_match(macro_val, json_val):
                    warnings.append(
                        f"[Exp{exp_num}] \\{cmd_top_t} = {fmt(macro_val)} but "
                        f"JSON top |t| = {fmt(json_val)}"
                    )

    # R² range across all responses
    all_r2 = [rd.get("r_squared", 0) for rd in responses.values()]
    if all_r2:
        cmd_min = f"Exp{roman}RsqMin"
        if cmd_min in macros:
            macro_val = parse_value(macros[cmd_min])
            json_val = round(min(all_r2), 2)
            checks += 1
            if not values_match(macro_val, json_val):
                warnings.append(
                    f"[Exp{exp_num}] \\{cmd_min} = {fmt(macro_val)} but "
                    f"JSON min R² = {fmt(json_val)}"
                )

        cmd_max = f"Exp{roman}RsqMax"
        if cmd_max in macros:
            macro_val = parse_value(macros[cmd_max])
            json_val = round(max(all_r2), 2)
            checks += 1
            if not values_match(macro_val, json_val):
                warnings.append(
                    f"[Exp{exp_num}] \\{cmd_max} = {fmt(macro_val)} but "
                    f"JSON max R² = {fmt(json_val)}"
                )

    return checks


def check_lgbm(exp_num, macros, robust_data, warnings):
    """Check LGBM R² macros against robust_results.json."""
    roman = NUM_TO_ROMAN[exp_num]
    key_resp = KEY_RESPONSES.get(exp_num, {})
    lgbm = robust_data.get("lightgbm", {}) if robust_data else {}

    checks = 0

    for short_key, response_key in key_resp.items():
        cmd_suffix = short_key.capitalize()
        cmd = f"Exp{roman}{cmd_suffix}LGBMRsq"

        if cmd not in macros:
            continue

        macro_val = parse_value(macros[cmd])
        json_val = round(lgbm.get(response_key, {}).get("lgbm_cv_r2", float("nan")), 3)
        checks += 1
        if not values_match(macro_val, json_val):
            warnings.append(
                f"[Exp{exp_num}] \\{cmd} = {fmt(macro_val)} but "
                f"JSON lgbm_cv_r2 = {fmt(json_val)}"
            )

    # LGBM R² range
    all_lgbm = [v.get("lgbm_cv_r2", 0) for v in lgbm.values()]
    if all_lgbm:
        cmd_min = f"Exp{roman}LGBMRsqMin"
        if cmd_min in macros:
            macro_val = parse_value(macros[cmd_min])
            json_val = round(min(all_lgbm), 2)
            checks += 1
            if not values_match(macro_val, json_val):
                warnings.append(
                    f"[Exp{exp_num}] \\{cmd_min} = {fmt(macro_val)} but "
                    f"JSON min LGBM R² = {fmt(json_val)}"
                )

        cmd_max = f"Exp{roman}LGBMRsqMax"
        if cmd_max in macros:
            macro_val = parse_value(macros[cmd_max])
            json_val = round(max(all_lgbm), 2)
            checks += 1
            if not values_match(macro_val, json_val):
                warnings.append(
                    f"[Exp{exp_num}] \\{cmd_max} = {fmt(macro_val)} but "
                    f"JSON max LGBM R² = {fmt(json_val)}"
                )

    return checks


def check_press(exp_num, macros, robust_data, warnings):
    """Check PRESS gap macros against robust_results.json."""
    roman = NUM_TO_ROMAN[exp_num]
    key_resp = KEY_RESPONSES.get(exp_num, {})
    press = robust_data.get("press", {}) if robust_data else {}

    checks = 0

    for short_key, response_key in key_resp.items():
        cmd_suffix = short_key.capitalize()

        # Predicted R²
        cmd_pred = f"Exp{roman}{cmd_suffix}PredRsq"
        if cmd_pred in macros:
            macro_val = parse_value(macros[cmd_pred])
            json_val = round(
                press.get(response_key, {}).get("predicted_r_squared", float("nan")), 3
            )
            checks += 1
            if not values_match(macro_val, json_val):
                warnings.append(
                    f"[Exp{exp_num}] \\{cmd_pred} = {fmt(macro_val)} but "
                    f"JSON predicted_r_squared = {fmt(json_val)}"
                )

        # PRESS gap
        cmd_gap = f"Exp{roman}{cmd_suffix}PRESSGap"
        if cmd_gap in macros:
            macro_val = parse_value(macros[cmd_gap])
            json_val = round(
                press.get(response_key, {}).get("gap_r2_pred_r2", float("nan")), 3
            )
            checks += 1
            if not values_match(macro_val, json_val):
                warnings.append(
                    f"[Exp{exp_num}] \\{cmd_gap} = {fmt(macro_val)} but "
                    f"JSON gap_r2_pred_r2 = {fmt(json_val)}"
                )

    # PRESS gap range across all responses
    all_gaps = [v.get("gap_r2_pred_r2", 0) for v in press.values()]
    if all_gaps:
        cmd_min = f"Exp{roman}PRESSGapMin"
        if cmd_min in macros:
            macro_val = parse_value(macros[cmd_min])
            json_val = round(min(all_gaps), 3)
            checks += 1
            if not values_match(macro_val, json_val):
                warnings.append(
                    f"[Exp{exp_num}] \\{cmd_min} = {fmt(macro_val)} but "
                    f"JSON min PRESS gap = {fmt(json_val)}"
                )

        cmd_max = f"Exp{roman}PRESSGapMax"
        if cmd_max in macros:
            macro_val = parse_value(macros[cmd_max])
            json_val = round(max(all_gaps), 3)
            checks += 1
            if not values_match(macro_val, json_val):
                warnings.append(
                    f"[Exp{exp_num}] \\{cmd_max} = {fmt(macro_val)} but "
                    f"JSON max PRESS gap = {fmt(json_val)}"
                )

    return checks


def check_multiplicity(exp_num, macros, robust_data, warnings):
    """Check multiplicity count macros against robust_results.json."""
    roman = NUM_TO_ROMAN[exp_num]
    mult = robust_data.get("multiplicity", {}) if robust_data else {}

    checks = 0

    field_map = {
        f"Exp{roman}NTests": "n_tests",
        f"Exp{roman}NRawSig": "n_raw_sig",
        f"Exp{roman}NHolmSig": "n_holm_sig",
        f"Exp{roman}NBHSig": "n_bh_sig",
    }

    for cmd, field in field_map.items():
        if cmd not in macros:
            continue
        macro_val = parse_value(macros[cmd])
        json_val = mult.get(field)
        if json_val is not None:
            checks += 1
            if not values_match(macro_val, json_val):
                warnings.append(
                    f"[Exp{exp_num}] \\{cmd} = {macro_val} but "
                    f"JSON {field} = {json_val}"
                )

    # BH percentage
    cmd_pct = f"Exp{roman}BHPct"
    if cmd_pct in macros:
        macro_val = parse_value(macros[cmd_pct])
        n_raw = mult.get("n_raw_sig", 0)
        n_bh = mult.get("n_bh_sig", 0)
        if n_raw > 0:
            json_val = round(100 * n_bh / n_raw)
            checks += 1
            if not values_match(macro_val, json_val):
                warnings.append(
                    f"[Exp{exp_num}] \\{cmd_pct} = {macro_val} but "
                    f"computed BH% = {json_val}"
                )

    return checks


def check_lgbm_excess(est_data, robust_data, warnings):
    """Warn if LGBM R² exceeds OLS R² by more than 0.05 for any response.

    This check runs across all experiments.
    """
    checks = 0
    if not est_data or not robust_data:
        return checks

    responses = est_data.get("responses", {})
    lgbm = robust_data.get("lightgbm", {})
    exp_num = est_data.get("experiment", "?")

    for response_key in responses:
        ols_r2 = responses[response_key].get("r_squared")
        lgbm_r2 = lgbm.get(response_key, {}).get("lgbm_cv_r2")

        if ols_r2 is None or lgbm_r2 is None:
            continue

        checks += 1
        excess = lgbm_r2 - ols_r2
        if excess > LGBM_EXCESS_THRESHOLD:
            warnings.append(
                f"[Exp{exp_num}] LGBM R² ({lgbm_r2:.3f}) exceeds OLS R² "
                f"({ols_r2:.3f}) by {excess:.3f} for {response_key}"
            )

    return checks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Verify paper numbers in numbers.tex match actual data."
    )
    parser.parse_args()

    if not os.path.exists(NUMBERS_TEX):
        print(f"ERROR: {NUMBERS_TEX} not found")
        sys.exit(1)

    macros = parse_numbers_tex(NUMBERS_TEX)
    print(f"Parsed {len(macros)} macros from {NUMBERS_TEX}")

    warnings = []
    total_checks = 0

    for exp_num in ["1", "2", "3", "4a", "4b"]:
        print(f"\n--- Experiment {exp_num} ---")

        # Load data sources
        csv_path = os.path.join(RESULTS_DIR, f"exp{exp_num}", "data.csv")
        est_path = os.path.join(RESULTS_DIR, f"exp{exp_num}", "estimation_results.json")
        robust_path = os.path.join(
            RESULTS_DIR, f"exp{exp_num}", "robust", "robust_results.json"
        )

        csv_rows = count_csv_rows(csv_path)
        est_data = load_json(est_path)
        robust_data = load_json(robust_path)

        if csv_rows is None:
            print(f"  SKIP: {csv_path} not found")
        else:
            print(f"  data.csv: {csv_rows} rows")

        if est_data is None:
            print(f"  SKIP: {est_path} not found")
        else:
            n_responses = len(est_data.get("responses", {}))
            print(f"  estimation_results.json: {n_responses} responses")

        if robust_data is None:
            print(f"  SKIP: {robust_path} not found")
        else:
            print(f"  robust_results.json: loaded")

        # Run checks
        n = check_nobs(exp_num, macros, est_data, csv_rows, warnings)
        total_checks += n
        print(f"  n_obs checks: {n}")

        if est_data:
            n = check_r_squared(exp_num, macros, est_data, warnings)
            total_checks += n
            print(f"  R-squared checks: {n}")

        if robust_data:
            n = check_lgbm(exp_num, macros, robust_data, warnings)
            total_checks += n
            print(f"  LGBM R-squared checks: {n}")

            n = check_press(exp_num, macros, robust_data, warnings)
            total_checks += n
            print(f"  PRESS gap checks: {n}")

            n = check_multiplicity(exp_num, macros, robust_data, warnings)
            total_checks += n
            print(f"  Multiplicity checks: {n}")

        # LGBM excess check
        if est_data and robust_data:
            n = check_lgbm_excess(est_data, robust_data, warnings)
            total_checks += n
            print(f"  LGBM excess checks: {n}")

    # Summary
    n_passed = total_checks - len(warnings)
    print("\n" + "=" * 60)

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  WARNING: {w}")

    print(f"\n{n_passed} checks passed, {len(warnings)} warnings")

    if warnings:
        sys.exit(1)
    else:
        print("All checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
