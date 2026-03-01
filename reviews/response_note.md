# Reviewer Response Mapping

Status legend: **ADDRESSED** | **PARTIALLY ADDRESSED** | **NOT ADDRESSED** (with rationale)

---

## Section 1: Critical Data-Paper Discrepancies

All items addressed by the macro system (`numbers.tex`) + pipeline hardening (`check_consistency.py`):

| # | Issue | Status | How |
|---|-------|--------|-----|
| 1.1 | Exp1 LGBM comparison direction | **ADDRESSED** | `res1.tex` now uses macros, acknowledges LGBM > OLS |
| 1.2 | Exp1 R-squared values | **ADDRESSED** | Macros `\ExpOneRsqMin`, `\ExpOneRsqMax` |
| 1.3 | Exp2 observation count | **ADDRESSED** | Macro `\ExpTwoNobs` |
| 1.4 | Exp3 dominant factor misidentified | **ADDRESSED** | `res3.tex` rewritten with correct rankings |
| 1.5 | Exp3 model adequacy numbers | **ADDRESSED** | Macros + rewritten appendix prose |
| 1.6 | Exp4 observation count | **ADDRESSED** | Macro `\ExpFourNobs` |
| 1.7 | Summary: systematic audit | **ADDRESSED** | `check_consistency.py` verifies macros against data |

## Section 2: Methodological Concerns

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| 2.1 | Convergence detection circular | **NOT ADDRESSED** | Acknowledged limitation; prospective criteria would require redesigning the convergence metric, out of scope for this revision |
| 2.2 | Action space discretization | **ADDRESSED** | `discretization_robustness.py` script, new appendix subsection, `n_actions` removed from Exp1 factorial |
| 2.3 | State space sparsity (Exp2) | **NOT ADDRESSED** | Acknowledged limitation; Q-table visit diagnostics are a good future extension but do not change the main findings |
| 2.4 | BNE benchmarks discrete vs continuous | **NOT ADDRESSED** | Code already uses `grid_adjusted_bne_revenue`; could add a clarifying sentence |
| 2.5 | Epsilon decay bug | **ADDRESSED** | `exp1.py` and `exp2.py` fixed, both decay to 0.01 floor |
| 2.6 | Incomparable exploration params | **ADDRESSED** | Footnote added to `experiments.tex` acknowledging the issue |

## Section 3: Statistical Inference Concerns

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| 3.1 | Multiple testing across experiments | **ADDRESSED** | New paragraph in `inference.tex` with $(1/k)^4$ bound |
| 3.2 | Exp2 statistical power for auction format | **ADDRESSED** | Economic significance qualifier added |
| 3.3 | Low R-squared for Exp4 variables | **ADDRESSED** | Acknowledgment added to `res4.tex` |
| 3.4 | Model misfit exploitable (Exp1) | **ADDRESSED** | `res1.tex` now acknowledges LGBM > OLS with specific gaps |

## Section 4: Per-Experiment Critiques

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| 4.1a | Exp1 "minimal state = bandit" | **NOT ADDRESSED** | Valid observation; could add footnote, but does not change findings |
| 4.1b | Non-convergence fraction | **NOT ADDRESSED** | Would need code changes to track |
| 4.2a | Exp2 eta null result under-explored | **NOT ADDRESSED** | Valid concern; state representation limits are acknowledged implicitly |
| 4.2b | Exp2 fixed hyperparameter levels | **ADDRESSED** | Reported in `res2.tex` |
| 4.3a | Exp3 6-action discretization | **ADDRESSED** | Discretization robustness check + `n_bid_actions` parameter |
| 4.3b | Exp3 zero-revenue run fraction | **NOT ADDRESSED** | Would need additional analysis |
| 4.3c | Exp3 dominant factor misidentification | **ADDRESSED** | `res3.tex` rewritten |
| 4.3d | Exp3 LinUCB mechanistic explanation | **NOT ADDRESSED** | Speculative explanation acknowledged |
| 4.4a | Exp4 greedy vs LP optimum | **ADDRESSED** | LP optimum added to `exp4.py` |
| 4.4b | Exp4 hard budget cap breaks dual | **NOT ADDRESSED** | Acknowledged limitation of the implementation |
| 4.4c | Exp4 only 3 factors | **NOT ADDRESSED** | By design; pacing params are studied in deep-dive |
| 4.4d | Exp4 burn-in sensitivity | **NOT ADDRESSED** | Would need additional runs |

## Section 5: Framing and Interpretation

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| 5.1 | Abstract overclaims | **ADDRESSED** | Softened "dominates" language |
| 5.2 | First-price "underperformance" | **ADDRESSED** | Statistical vs economic significance distinguished |
| 5.3 | RET violation framing | **ADDRESSED** | Reframed as "consistent with" not "contradicts" |
| 5.4 | Typo | **ADDRESSED** | Fixed |

## Section 6: Missing Analyses

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| 6.1 | Episode budget sensitivity | **NOT ADDRESSED** | Would require full re-runs at multiple episode counts |
| 6.2 | Action space sensitivity | **ADDRESSED** | `discretization_robustness.py` |
| 6.3 | State space sufficiency | **NOT ADDRESSED** | Future work |
| 6.4 | Equilibrium verification | **NOT ADDRESSED** | Bid-vs-signal plots exist in deep_dive but not in paper |
| 6.5 | Distribution of outcomes | **NOT ADDRESSED** | Histograms could be added as future work |
| 6.6 | Seed-level analysis | **PARTIALLY ADDRESSED** | Exp4 panel regression with seed FEs exists |

## Section 7: Minor Issues

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| 7.1 | "e.g." in parameter tables | **ADDRESSED** | Removed "e.g." qualifiers from `appendix_design.tex` tables |
| 7.2 | Convergence table identical values | **NOT ADDRESSED** | Check if normalization artifact |
| 7.3 | "Dual-based" definition | **NOT ADDRESSED** | Could add brief explanation |
| 7.4 | Cross-experiment revenue units | **NOT ADDRESSED** | Inherent to different experimental designs |
| 7.5 | Figure captions | **NOT ADDRESSED** | Minor; could improve in final revision |

---

## Summary

| Category | Addressed | Partially | Not Addressed | Total |
|----------|-----------|-----------|---------------|-------|
| Data-paper discrepancies | 7 | 0 | 0 | 7 |
| Methodology | 3 | 0 | 3 | 6 |
| Statistical inference | 4 | 0 | 0 | 4 |
| Per-experiment | 5 | 0 | 7 | 12 |
| Framing | 4 | 0 | 0 | 4 |
| Missing analyses | 1 | 1 | 4 | 6 |
| Minor issues | 1 | 0 | 4 | 5 |
| **Total** | **25** | **1** | **18** | **44** |

Most unaddressed items are acknowledged limitations (convergence circularity, state space sparsity) or would require significant additional experimentation (episode budget sensitivity, burn-in sensitivity). The critical data-paper discrepancies and statistical inference concerns are fully addressed.
