# Code Audit: Stale Variable Names in Pipeline Scripts

**Date:** 2026-02-26
**Trigger:** Post-refactoring audit after Exp2 BNE metrics, Exp4 LP-based PoA, and convergence module extraction.

## Summary

Five bugs found across four files. All involve stale variable names in downstream pipeline/table-generation scripts that reference old column names from before refactoring. None cause hard crashes (all have graceful fallbacks), but they produce incomplete output: missing tables, missing figure copies, and skipped validations.

**Fixed:** 4 of 5 bugs (Bug 4 was already fixed in a prior session).

---

## Bug 1 (MEDIUM) -- FIXED

**File:** `scripts/generate_tables.py` lines 78, 92
**Issue:** `REGRET_VAR[2]` and `KEY_RESPONSES[2]["reg"]` referenced `"avg_regret_of_seller"` instead of `"excess_regret"`. Exp2 was refactored to use `excess_regret` as its primary regret metric.
**Impact:** Exp2 regret ranked-effects table silently skipped; regret figure copies printed WARNING.
**Fix:** Changed both to `"excess_regret"`.

## Bug 2 (MEDIUM) -- FIXED

**File:** `scripts/check_consistency.py` line 51
**Issue:** `KEY_RESPONSES[2]["reg"]` referenced `"avg_regret_of_seller"` instead of `"excess_regret"`.
**Impact:** Exp2 regret macro validation silently skipped during consistency checks.
**Fix:** Changed to `"excess_regret"`.

## Bug 3 (MEDIUM) -- FIXED

**File:** `scripts/check_consistency.py` line 61
**Issue:** `KEY_RESPONSES[4]["reg"]` referenced `"mean_effective_poa"` instead of `"mean_effective_poa_lp"`. After the LP-based PoA refactor, the primary metric is `mean_effective_poa_lp`; the greedy version still exists but is secondary.
**Impact:** Validated wrong metric for Exp4 regret consistency checks.
**Fix:** Changed to `"mean_effective_poa_lp"`.

## Bug 4 (LOW) -- ALREADY FIXED

**File:** `scripts/generate_results.py` line 58
**Issue:** Plan identified `"avg_regret_of_seller"` in Exp2 response list, but file already contains `"excess_regret"`. Fixed in a prior session.
**No action taken.**

## Bug 5 (LOW) -- FIXED

**File:** `src/estimation/robust_analysis.py` lines 1076-1134 (standalone CLI defaults)
**Issue:** `_EXP_DEFAULTS` dict had stale defaults for standalone `python robust_analysis.py --exp N` mode:
- Exp1: missing 4 coded columns (`reserve_price_coded`, `init_coded`, `info_feedback_coded`, `decay_type_coded`) and 1 response (`no_sale_rate`)
- Exp4: missing 2 new responses (`mean_lp_offline_welfare`, `mean_effective_poa_lp`)

**Impact:** Only affects standalone invocation; `make analyze` is unaffected (passes correct cols from est*.py wrappers).
**Fix:** Added missing columns and responses to match est1.py and est4.py.

---

## Verification

1. All three modified Python files pass syntax check (`ast.parse`).
2. `grep -rn "avg_regret_of_seller" scripts/` confirms remaining references are Exp1-context only (correct) or unused `RESPONSES` dict entries (dead code, harmless).
3. `grep -rn 'mean_effective_poa[^_]' scripts/` confirms remaining references are for the greedy metric which still exists as a secondary metric.

## Note: Dead Code

The `RESPONSES` dict at `scripts/generate_tables.py` lines 31-73 is defined but never referenced in the code. It contains stale entries (e.g., Exp2 still lists `"avg_regret_of_seller"` on line 43). This is harmless dead code. Not fixed to minimize diff, but could be removed in a future cleanup.
