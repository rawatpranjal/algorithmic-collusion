# CLAUDE.md - Algorithmic Collusion Experiments

## Communication Style

When asked for a TLDR or summary of results, provide **intuitive, interpretive, human-level explanations** of what happened and why — not just numbers or statistical jargon. Explain findings as you would to a smart colleague over coffee: what's the story, what's surprising, what's the takeaway.

## Quick Start (Makefile)

```bash
make help             # Show all targets and options
make smoke            # Quick-test all experiments (16 cells, 1 rep)
make experiments      # Run all experiments (override REPS per exp)
make analyze          # Factorial ANOVA for all experiments
make robust           # Robustness checks for all experiments
make traces           # Single-run trace plots for paper figures
make tables           # Generate LaTeX tables + copy figures
make pdf              # Standalone results PDF
make paper            # Compile main paper (pdflatex + bibtex + 2x pdflatex)
make check-freshness  # Verify data.csv files match current code
make all              # Full pipeline: experiments -> analyze -> robust -> traces -> tables -> pdf -> paper
```

### Individual Experiments
```bash
make smoke1           # Quick-test experiment 1 only
make exp1             # Run experiment 1
make analyze1         # Factorial ANOVA for experiment 1
make robust1          # Robustness checks for experiment 1
```

### Deep Dive (Single Run)
```bash
make dive1            # Single-run deep dive for experiment 1
# Or with custom params:
PYTHONPATH=src python3 scripts/deep_dive.py --exp 2 --param eta=1.0 --verbose
```

### Options
```bash
make exp1 REPS=5      # Custom replicates per cell (default: 2)
make exp1 SEED=123    # Custom random seed (default: 42)
make exp1 WORKERS=8   # Custom parallel workers (default: cpu_count/2)
```

**Production REPS per experiment:** `make exp1 REPS=2`, `make exp2 REPS=8`, `make exp3a REPS=2`, `make exp3b REPS=2`, `make exp4a REPS=8`, `make exp4b REPS=8`

**Why REPS vary:** Experiments with many cells (exp1: 512, exp3a: 384) need few reps because the large number of cells provides sufficient degrees of freedom. Exp4a/4b have only 64 cells (2^6 full factorial) but 8 reps per cell (512 rows) provides 448 pure error DF — more than sufficient for all robustness checks given the large effect sizes observed.

## CLI Reference

| Flag | Description |
|------|-------------|
| `--exp N` | Experiment number (1, 2, 3a, 3b, 4a, or 4b) |
| `--quick` | Quick test: 2^4=16 cells, 1 replicate, 1000 episodes |
| `--parallel` | Use multiprocessing |
| `--workers N` | Number of parallel workers (default: cpu_count // 2) |
| `--replicates N` | Replicates per cell (default: 2) |
| `--output-dir PATH` | Custom output directory |
| `--seed N` | Random seed (default: 42) |
| `--cloud` | Run on Google Cloud VM |
| `--detached` | Fire-and-forget mode (VM self-deletes after completion) |

### Deep Dive CLI (`scripts/deep_dive.py`)

| Flag | Description |
|------|-------------|
| `--exp N` | Experiment number (required) |
| `--param key=value` | Parameter override (repeatable) |
| `--list-params` | List available parameters and defaults, then exit |
| `--output-dir PATH` | Output directory (default: `results/deep_dive/expN_TIMESTAMP`) |
| `--seed N` | Random seed (default: 42) |
| `--verbose` | Print detailed post-hoc diagnostics |
| `--no-plots` | Skip trace plot generation |
| `--no-save` | Console-only mode, skip file output |

## Factorial Design

Experiments use 2^k factorial designs (coded -1/+1) instead of random sampling.

| Exp | Design | Factors | Cells | Reps | Total Runs | Est. Time |
|-----|--------|---------|-------|------|------------|-----------|
| 1 | 2^(10-1) half-fraction (Res V) | 10 | 512 | 2 | 1024 | ~2-3 hrs |
| 2 | 3 × 2^3 mixed-level | 4 (3 binary + 1 three-level) | 24 | 8 | 192 | ~2 hrs |
| 3a | 3 × 2^7 mixed-level (LinUCB) | 8 (7 binary + 1 three-level) | 384 | 2 | 768 | ~2-4 hrs |
| 3b | 3 × 2^5 mixed-level (Thompson) | 6 (5 binary + 1 three-level) | 96 | 2 | 192 | ~1 hr |
| 4a | 2^6 full factorial | 6 | 64 | 8 | 512 | ~30 min |
| 4b | 2^6 full factorial | 6 | 64 | 8 | 512 | ~30 min |

Quick mode: 2^4 = 16 cells × 1 replicate for smoke testing (Exp 1-3b); 64 cells × 2 seeds for Exp 4a-4b.

## Pipeline: From Simulation to Paper

### Full pipeline (after experiments have been run)
```bash
make analyze          # 1. Factorial ANOVA (JSON + plots)
make robust           # 2. Robustness checks
make traces           # 3. Single-run trace plots
make tables           # 4. LaTeX tables + copy figures
make pdf              # 5. Standalone results PDF
make paper            # 6. Compile paper
```

Or in one command: `make analyze robust traces tables pdf paper`

<details>
<summary>Manual commands (without Make)</summary>

```bash
PYTHONPATH=src python3 src/estimation/est1.py
PYTHONPATH=src python3 src/estimation/est2.py
PYTHONPATH=src python3 src/estimation/est3a.py
PYTHONPATH=src python3 src/estimation/est3b.py
PYTHONPATH=src python3 src/estimation/est4a.py
PYTHONPATH=src python3 src/estimation/est4b.py
PYTHONPATH=src python3 scripts/generate_trace_plots.py
python3 scripts/generate_tables.py
python3 scripts/generate_results.py
cd paper && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex && cd ..
```
</details>

### What each step produces
- **est*.py** → `results/expN/estimation_results.json`, `pareto_charts/*.png`, `main_effects/*.png`, `interaction_plots/*.png`, `normal_prob_plots/*.png`, `residuals/*.png`, `robust/robust_results.json`, `robust/*.png`, `robust/robust_summary.txt`. Auto-checks `data_manifest.json` and warns if source code changed since data generation.
- **generate_trace_plots.py** → `paper/figures/e{1,2,3a,3b,4a,4b}_trace.png` (single-run learning trajectory visualizations)
- **generate_tables.py** → `paper/tables/expN_coefficients.tex`, `paper/tables/expN_model_fit.tex`, `paper/tables/expN_significant.tex`, copies key figures to `paper/figures/`
- **generate_results.py** → `paper/results.pdf` (comprehensive factorial analysis report with all tables, plots, and robustness diagnostics)

## Cloud Execution

### Running on GCP (fire-and-forget)
```bash
# Test with small N first
python3 scripts/run_experiment.py --exp 1 --cloud --detached --quick

# Full experiments (can close terminal after launch)
python3 scripts/run_experiment.py --exp 1 --cloud --detached
python3 scripts/run_experiment.py --exp 2 --cloud --detached
python3 scripts/run_experiment.py --exp 3a --cloud --detached
python3 scripts/run_experiment.py --exp 3b --cloud --detached
python3 scripts/run_experiment.py --exp 4a --cloud --detached
python3 scripts/run_experiment.py --exp 4b --cloud --detached
```

### Monitoring Cloud Experiments

**Find GCP project (bucket name contains project ID):**
```bash
gcloud projects list
# Look for project like: collusion-exp-XXXXXXXXXX
```

**Check if VM is still running:**
```bash
gcloud compute instances list --project=collusion-exp-XXXXXXXXXX
```

**View live experiment logs on VM:**
```bash
gcloud compute ssh VM_NAME --project=collusion-exp-XXXXXXXXXX --zone=us-central1-a \
  --command="tail -50 /var/log/experiment.log"
```

**Check GCS for results (appears after completion):**
```bash
gsutil ls gs://collusion-exp-XXXXXXXXXX-collusion-experiments/results/
gsutil -m cp -r gs://collusion-exp-XXXXXXXXXX-collusion-experiments/results/ ./cloud_results/
```

### Running Analytics After Cloud Completion

```bash
PROJECT=collusion-exp-XXXXXXXXXX
TIMESTAMP=YYYYMMDD-HHMMSS  # from gsutil ls

# 1. Download results
gsutil cp gs://${PROJECT}-collusion-experiments/results/exp1/$TIMESTAMP/exp1/data.csv results/exp1/data.csv
gsutil cp gs://${PROJECT}-collusion-experiments/results/exp2/$TIMESTAMP/exp2/data.csv results/exp2/data.csv
gsutil cp gs://${PROJECT}-collusion-experiments/results/exp3a/$TIMESTAMP/exp3a/data.csv results/exp3a/data.csv
gsutil cp gs://${PROJECT}-collusion-experiments/results/exp3b/$TIMESTAMP/exp3b/data.csv results/exp3b/data.csv
gsutil cp gs://${PROJECT}-collusion-experiments/results/exp4a/$TIMESTAMP/exp4a/data.csv results/exp4a/data.csv
gsutil cp gs://${PROJECT}-collusion-experiments/results/exp4b/$TIMESTAMP/exp4b/data.csv results/exp4b/data.csv

# 2. Run full pipeline
make analyze tables pdf paper
```

## Experiments Overview

| Exp | Algorithm | Valuations | Design | Factors |
|-----|-----------|------------|--------|---------|
| 1 | Q-learning | Constant (v=1.0) | 2^(10-1) Res V | auction_type, alpha, gamma, reserve_price, init, exploration, asynchronous, n_bidders, info_feedback, decay_type |
| 2 | Q-learning | Affiliated (eta) | 3 × 2^3 mixed | auction_type, n_bidders, state_info, eta |
| 3a | LinUCB | Affiliated (eta) | 3 × 2^7 mixed | auction_type, n_bidders, reserve_price, eta, exploration_intensity, context_richness, lam, memory_decay |
| 3b | Thompson | Affiliated (eta) | 3 × 2^5 mixed | auction_type, n_bidders, reserve_price, eta, exploration_intensity, context_richness |
| 4a | Dual Pacing | LogNormal (asymmetric) | 2^6 full | auction_type, objective, n_bidders, budget_multiplier, reserve_price, sigma |
| 4b | PI Pacing | LogNormal (asymmetric) | 2^6 full | auction_type, aggressiveness, n_bidders, budget_multiplier, reserve_price, sigma |

## Output Structure

```
results/
├── exp1/
│   ├── data.csv                  # Factorial design data with coded columns
│   ├── data_manifest.json        # Code provenance (git hash + source checksums)
│   ├── design_info.json          # Factor definitions
│   ├── param_mappings.json
│   ├── estimation_results.json   # OLS/ANOVA results
│   ├── analysis_stdout.txt       # Full analysis log
│   ├── pareto_charts/*.png       # |t-statistic| bar charts
│   ├── normal_prob_plots/*.png   # Half-normal effect identification
│   ├── main_effects/*.png        # Mean response at -1 vs +1
│   ├── interaction_plots/*.png   # Top 2-way interactions
│   ├── residuals/*.png           # QQ + residuals vs fitted
│   └── robust/                   # Robustness analysis (auto-generated by est*.py)
│       ├── robust_results.json   # All 13 robustness checks
│       ├── robust_summary.txt    # Human-readable summary
│       ├── lgbm_comparison.png   # Linear vs nonparametric R²
│       ├── response_correlations.png
│       └── power_analysis.png
├── exp2/ (same structure)
├── exp3a/ (same structure, LinUCB)
├── exp3b/ (same structure, Thompson)
├── exp4a/
│   ├── data.csv                  # 400 rows (run-level aggregates)
│   └── (same plot + robust/ structure as above)
└── exp4b/ (same structure as exp4a)
```

### Key Output Columns
- `*_coded`: Factor coded as -1 (low) or +1 (high) — used for analysis
- `avg_rev_last_1000`: Average revenue in final 1000 episodes
- `avg_rev_all`: Average revenue across all episodes (lifetime); `mean_rev_all` for Exp4a/4b
- `time_to_converge`: First episode where revenue stays in ±5% band
- `no_sale_rate`: Fraction of rounds with no valid bids
- `price_volatility`: Std dev of winning bids
- `winner_entropy`: Entropy of winner distribution
- `theoretical_revenue`: BNE prediction
- `ratio_to_theory`: Actual / theoretical revenue

## Repo Structure

```
src/
├── experiments/
│   ├── exp1.py    # Constant valuations Q-learning
│   ├── exp2.py    # Affiliated values Q-learning
│   ├── exp3.py    # LinUCB + Thompson bandits (shared simulation engine for 3a/3b)
│   ├── exp4a.py   # Autobidding dual pacing
│   └── exp4b.py   # Autobidding PI controller pacing
├── estimation/
│   ├── factorial_analysis.py  # Shared OLS + ANOVA engine
│   ├── robust_analysis.py     # Shared robustness checks (13 tests)
│   ├── est1.py                # Thin wrapper for Exp1 (ANOVA + robustness)
│   ├── est2.py                # Thin wrapper for Exp2 (ANOVA + robustness)
│   ├── est3a.py               # Thin wrapper for Exp3a LinUCB (ANOVA + robustness)
│   ├── est3b.py               # Thin wrapper for Exp3b Thompson (ANOVA + robustness)
│   ├── est4a.py               # Factorial ANOVA + robustness
│   └── est4b.py               # Factorial ANOVA + robustness (PI)
└── cloud/         # Distributed execution support

scripts/
├── run_experiment.py          # Unified CLI (factorial design)
├── deep_dive.py               # Unified single-run analysis (trace + verbose + save)
├── generate_trace_plots.py    # Single-run trace visualizations
├── generate_tables.py         # JSON → LaTeX tables + figure copy
├── generate_results.py        # Standalone results.pdf generator
├── budget_robustness.py       # Budget sensitivity analysis
├── calibrate_exp4b.py         # PI controller calibration
├── calibration_exploration.py # Parameter space exploration
├── check_consistency.py       # Cross-experiment consistency checks
├── discretization_robustness.py # Action-space discretization sensitivity
├── debug_exp2.py              # Q-learning diagnostics with BNE comparison
├── debug_exp3.py              # Algorithm comparison diagnostics
└── verification/              # Mathematical claim verification suite
    ├── run_all.py             # Runner for all verification tests
    ├── helpers.py             # Pure-math helpers (LP, BNE, welfare)
    ├── test_equilibria.py     # BNE and valuation model tests
    ├── test_model.py          # Simulation model correctness tests
    ├── test_novel_claims.py   # LP benchmark and parametrization verification
    ├── test_pacing.py         # Pacing algorithm tests
    ├── test_payment_rules_exp4.py # Payment rule verification for Exp4
    └── test_welfare.py        # LP optimality and PoA bounds tests

results/           # Experiment output data (see Output Structure section)

reviews/           # Audit reports (simulation_audit.md, code_audit.md, etc.)

paper/             # LaTeX paper source (single unified document, no separate appendix)
├── main.tex       # Unified document: main body + \appendix + 8 appendix sections
├── references.bib # Canonical bibliography (natbib/plainnat)
├── numbers.tex    # Auto-generated statistics macros
├── tables/        # Auto-generated .tex table snippets
├── figures/       # Publication figures
└── sections/      # Paper section .tex files (13 main body + 8 appendix)

archive/
├── paper/         # Superseded paper files (appendix.tex, references.tex, appendix_welfare.tex, dead section files)
├── scripts/       # Superseded utility scripts
├── iteration1/    # First iteration code
├── iteration3/    # Third iteration code
├── cate/          # Old CATE analysis
├── old_results/   # Stale DoubleML/CATE/grid results
├── doubleml_estimation/   # Old DoubleML est*.py files
├── random_sampling_results/  # Old random-sampling data.csv files
└── v2/            # Second iteration code

docs/
└── plans/         # Development roadmaps
```

## Architecture & Design Patterns

### Pipeline Data Flow
```
deep_dive.py      → results/deep_dive/expN_TIMESTAMP/{config,summary,revenues,round_history,final_state,figures}
run_experiment.py → results/expN/data.csv
                  → results/expN/design_info.json, param_mappings.json, data_manifest.json
est*.py           → results/expN/estimation_results.json + plots + robust/
generate_tables.py→ paper/tables/*.tex + paper/figures/*.png
generate_results.py → paper/results.pdf
pdflatex + bibtex → paper/main.pdf (single unified document with natbib bibliography)
```

### Experiment Module Contract
Every `src/experiments/expN.py` must expose:
```python
def run_experiment(**factors, seed=0, progress_callback=None)
    -> (summary_dict, revenue_list, round_history, final_state)
```
- `summary_dict`: flat dict of metric names to floats (no nesting)
- Keys use `snake_case` (e.g., `avg_rev_last_1000`, `no_sale_rate`)
- Exp4a/4b prefix run-level aggregates with `mean_` (e.g., `mean_platform_revenue`)

### Orchestration Contract (run_experiment.py)
For each experiment N, three things must exist:
1. `EXPN_FACTORS` dict of `{name: (low, high)}` tuples
2. `get_expN_tasks(quick, output_dir, seed, replicates) -> List[task_dict]`
3. `aggregate_expN_results(results, tasks, output_dir) -> None`

### Estimation Wrapper Contract (est*.py)
Each `src/estimation/estN.py` defines:
- `CODED_COLS`: list of `"factor_coded"` column names used as OLS predictors
- `RESPONSE_COLS`: list of outcome variable names to analyze

Then calls `run_factorial_analysis()` and `run_robust_analysis()` with those lists.

### Factor Coding Rules
- Binary factors: coded `-1` (low) / `+1` (high), column name `factor_coded`
- Three-level factors (eta): TWO orthogonal contrasts, `eta_linear_coded` and `eta_quadratic_coded`
- Both raw value (`auction_type="first"`) AND coded value (`auction_type_coded=1`) stored in data.csv
- OLS formulas must use ONLY `_coded` columns (orthogonality requirement)

### Seed Formula
```
seed = (base_seed + replicate) * 10000 + cell_id
```
Assumes `cell_id < 10000`. Max current design: Exp1 = 512 cells.

### Table/Figure Pipeline Conventions
- `generate_tables.py` maps response vars to short keys: `rev`, `vol` (exp1-3b); `rev`, `poa`, `vol` (exp4a/4b)
- Figures copied as `paper/figures/eN_{main|int|pareto}_{rev|vol}.png` (exp1-3b) or `_{rev|poa|vol}.png` (exp4a/4b)
- Tables written as `paper/tables/expN_{coefficients|model_fit|significant|ranked_rev|ranked_vol}.tex` (exp1-3b), plus `ranked_poa` for exp4a/4b
- `READABLE_NAMES` dict maps coded column names to LaTeX-safe display names

## Critical Invariants (Do Not Break)

| Invariant | Why |
|-----------|-----|
| `_coded` columns must be numeric -1/+1 | OLS orthogonality breaks otherwise |
| OLS uses only `_coded` cols, never raw values | Confounds main effects with interactions |
| `matplotlib.use("Agg")` before `import plt` | Crashes on headless servers |
| `PYTHONPATH=src` when running estimation | Enables `from estimation import ...` |
| Seed formula assumes cell_id < 10000 | Seed collisions corrupt reproducibility |
| `time_to_converge` normalized by episodes in est*.py | Raw values non-comparable across designs |
| No-sale: winner=-1, revenue=0 | Downstream metrics assume this encoding |
| data.csv has BOTH raw + coded columns | Aggregation and analysis use different ones |
| `run_experiment()` returns (summary, revenues, history, state) | Orchestrator unpacks this tuple |
| Three-level factors need TWO contrast columns | Single column loses quadratic information |
| Parameter defaults in `deep_dive.py` and `generate_trace_plots.py` must match production values in `run_experiment.py` | Validation non-representative otherwise |
| `data_manifest.json` written alongside `data.csv` | Enables staleness detection in `make analyze` |
| All citations use `\citet{}` or `\citep{}` natbib commands | Never use inline text citations; `references.bib` is the canonical source |
| `make paper` runs pdflatex + bibtex + 2x pdflatex | Bibliography and cross-references require multiple passes |

## Common Modification Recipes

**Add a new response variable:**
1. Add metric computation to `expN.py` `run_experiment()` summary dict
2. Add column name to `estN.py` `RESPONSE_COLS`
3. Add entry to `generate_tables.py` `RESPONSES` dict

**Add a new factor:**
1. Add to `EXPN_FACTORS` in `run_experiment.py`
2. Add coded column creation in `aggregate_expN_results()`
3. Add to `CODED_COLS` in `estN.py`
4. Add to `READABLE_NAMES` in `generate_tables.py`

**Change figure selection for paper:**
Update `KEY_RESPONSES` in `generate_tables.py`.

## Development

### Running Tests
```bash
make smoke            # Quick validation of all experiments
```

### Adding New Experiments
1. Create `src/experiments/expN.py` following exp2.py pattern
2. Define `EXPN_FACTORS` dict in `scripts/run_experiment.py`
3. Add `get_expN_tasks()` and `aggregate_expN_results()`
4. Create `src/estimation/estN.py` wrapper with coded cols and response cols
5. Update CLI to handle new experiment number

## Audit Methodology

### Deep Dive Simulation Audit Checklist

Use `scripts/deep_dive.py --exp N --verbose` with specific parameter overrides to verify simulation mechanics at the individual-run level. The automated pipeline (13 robustness tests, 15 verification tests) validates statistics; deep dives validate mechanics.

**Payment Rule Verification**: Run FPA and SPA configs with 2 and 4 bidders. Check that FPA revenue = winner's bid, SPA revenue = 2nd highest bid, winner reward = valuation - payment, loser reward = 0.

**Reserve Price Mechanics**: Run with `reserve_price=0.5`. Verify `no_sale_rate > 0` and all winners have bid >= reserve. In SPA, a single valid bidder pays exactly the reserve price.

**Exploration Sufficiency**: Check Q-table sparsity (non-zero entries / total). Flag if < 50% coverage. For full state space (`info_feedback=full, n_bidders=4`), Q-table is (4, 11, 11) = 484 entries; verify all are explored.

**Convergence Detection**: Manually recompute from `revenues.csv` using 1000-episode non-overlapping windows. Verify that after reported `time_to_converge`, all subsequent windows stay within ±5% of the convergence-point mean.

**Seed Reproducibility**: Run the same config twice with identical seed. Diff `summary.json` and `revenues.csv`; must show zero differences.

**Cross-Validation**: Pick a cell from `results/expN/data.csv`, reconstruct its parameters and seed (`seed = (base_seed + replicate) * 10000 + cell_id`), run deep dive, and verify `avg_rev_last_1000` matches.

### Audit Results

See `reviews/simulation_audit.md` for detailed audit results by experiment (Exp1, Exp2, Exp3a, Exp3b, Exp4a, Exp4b).

## Literature References

Before searching the web for papers/references, check `docs/transcriptions/` first. The repo contains PDF transcriptions of all key literature.

## Paper Writing Guide

### Voice and Register

- Voice of a senior statistician and economist: precise, measured, data-driven.
- Simple, direct sentences. No jargon unless defined. No unusual or rare words.
- Primary audience: economists. Secondary: CS and statistics researchers.
- CS/stats concepts (Q-learning, contextual bandits, factorial ANOVA) need brief plain-language explanation; auction theory and equilibrium concepts can assume familiarity.
- Explain intuition before formalism. Use concrete examples early; do not front-load notation.
- Formal academic register. No conversational tone, no enthusiasm ("exciting", "groundbreaking", "novel").
- Active voice preferred. Short declarative sentences.
- No long convoluted sentences. If a sentence has more than one subordinate clause, split it.
- Focus on what was done and what was found, not on what was "argued" or "proposed."
- Weigh competing interpretations, but take a position. The reader wants guidance, not a menu of possibilities.

### Reporting vs. Interpretation

- Maintain a clear division between reporting facts and interpreting them.
- Report results first: direction, magnitude, statistical significance, table/figure reference.
- Interpret second: what the result means economically, why it matters, how it connects to prior work.
- State what the data shows with confidence. Do not hedge statistically significant findings.
- Qualify clearly when extrapolating beyond experimental scope ("this suggests", "the results are consistent with").
- Distinguish between: (1) robust empirical findings (state directly), (2) plausible interpretations (signal uncertainty), (3) speculation (mark explicitly or omit).
- Never overstate. "X is the dominant factor" requires X to have the largest effect size. "X contributes to" if it is one of several factors.

### Paragraph Discipline

- Before writing a paragraph, plan in bullets what it will say. Write prose only after the structure is clear.
- Each paragraph focuses on one main point. Topic sentence states the claim. Evidence follows. A concluding sentence synthesizes or transitions.
- Each sentence carries one idea. Do not stack multiple complex concepts into a single sentence.
- Aim for 4-8 sentences per paragraph. Paragraphs within a section should be roughly equal in length, creating a visually uniform page. No stub paragraphs of 1-2 sentences; merge orphans into adjacent paragraphs.

### Cross-referencing and Non-repetition

- One canonical location per idea. Do not duplicate content across sections.
- Cross-reference liberally: "as shown in Table~\ref{...}", "Section~\ref{...} discusses", "recall the definition from Equation~\ref{...}".
- Cross-references over restatement. Never re-explain a concept introduced in an earlier section; use a `\ref{}` pointer.
- Section openings do not re-motivate. Do not restate the introduction; start at the technical content.
- When two sections touch the same topic (e.g., auction format effects in results and discussion), the results section reports facts and the discussion section interprets and synthesizes. Neither repeats the other.

### Footnotes

- Relegate technical details to footnotes: grid sizes, convergence thresholds, hyperparameter values, step counts, implementation specifics, historical side notes.
- Main text carries the argument. Footnotes carry the evidence trail and interesting tangents.
- Use footnotes for tedious but necessary information that would break the flow of a paragraph.

### Figures and Tables

- Prefer 1x2 or 1x3 panel figures (subfigures side by side) to show more information per float.
- If a graph can convey everything a table does, use the graph. But tables excel at precise coefficient values.
- Do not duplicate content between a table and a figure for the same result. Pick one as primary; the other, if included, must show something different (e.g., table for coefficients, figure for interaction pattern).
- Main body: figures preferred for visual patterns (main effects, interactions); tables for ranked coefficients and model fit.
- Appendixes: tables preferred for detailed results.
- Figure captions label what is shown (axes, panels, legend). Interpretation belongs in the citing prose paragraph.
- Table captions identify rows, columns, and units. The conclusion belongs in the citing prose paragraph.
- Do not restate numbers in prose that are already visible in a table or figure. Reference the float and state the interpretation.

### Formatting Rules (hard constraints)

- No em dashes (`---`) in any LaTeX file. Use commas, semicolons, or restructure.
- No colons in running prose. Use commas, semicolons, or separate sentences.
- No bullet lists (`\begin{itemize}`, `\begin{enumerate}`) in the paper. Use flowing prose.
- No `\paragraph{Bold.}` starters. Use `\subsection{}` or `\subsubsection{}`.
- No `\textbf{}` in running prose (only in table headers/captions).
- No data variable names in prose. Use full English names ("average revenue" not `avg_rev_last_1000`).
- No DoubleML terminology (CATE, ATE, GATE, BLP, partial-dependence).
- `\section`, `\subsection`, `\subsubsection` only. Keep hierarchy broad; avoid over-granular subsectioning.

### Citation Style

- All citations use natbib commands via `references.bib`. Never use inline text citations like "Author (Year)".
- `\citet{Key}` for textual citations where the author is part of the sentence: "As \citet{Calvano2020} show..."
- `\citep{Key}` for parenthetical citations: "...under budget constraints \citep{Balseiro2019Learning}."
- Multiple parenthetical: `\citep{Key1, Key2}` renders as "(Author1, Year1; Author2, Year2)".
- When adding a new reference, add the bib entry to `paper/references.bib` first, then use `\citet` or `\citep` in the text.

### Notation and Definitions

- Define all notation before first use. Every symbol in an equation must be explained in the surrounding prose.
- After each equation or definition, add 1-2 sentences of intuition explaining what it means economically or computationally.
- Move lengthy derivations to the appendix. The main text presents the result and its intuition.
- Move technical implementation details to footnotes or appendix tables.

### Section-Specific Guidelines

**Introduction.** Write last, after all other sections are complete. This is the high-level coverage of the entire paper. State the most important results upfront, since some readers only read the introduction. Structure: motivation and gap, what this paper does, key findings, paper outline.

**Literature Review.** Reference map first, then the gap, then how this paper fills it. Important references get dedicated space; minor ones get a parenthetical citation. Connect each stream of literature to our experimental setup (e.g., "we test this prediction in Experiment 2"). Do not just survey; show where the field stands and where it falls short.

**Theory (Auctions, Algorithms).** Very formal: full notation, definitions, key properties. Include a narrative explanation of how each auction format or algorithm works and what behaviour to expect. Relegate technical details (proof steps, implementation constants) to footnotes.

**Experimental Design.** Clean recipe: state what was done clearly and concisely. Factor definitions, design table, number of cells and replicates. Justifications for specific choices go in footnotes or the design appendix, not the main text.

**Statistical Inference.** Main body, concise (2-3 pages). Model specification, why factorial ANOVA with effects coding, how robustness is assessed. Full details of the 13 robustness checks belong in the appendix.

**Results (one section per experiment, standalone).** Structured walk through the factorial analysis: dominant effect first, then main effects, then interactions, covering all significant effects but briefly. Each section stands alone. Cross-experiment comparison happens only in the Discussion. Follow the results template below.

**Robustness.** Appendix only. Main body results sections note "robustness checks confirm these findings (Appendix~\ref{...})" in the summary paragraph.

**Discussion.** Equal weight to: (1) cross-experiment synthesis connecting patterns into unified takeaways, and (2) policy and practical implications for auction designers and regulators. Close with limitations, open questions, and future directions.

**Appendix sections.** Light orientation: one sentence at the top of each appendix section stating what it contains and which main-body section it supports. Then straight to the content.

### Results Sections Template

Each experiment's results section follows this structure:
1. One-paragraph setup: what this experiment tests, the design (factors, cells, replicates), and the key question.
2. Primary finding: state the dominant effect, reference the ranked-effects table or figure.
3. Main effects: 1-2 figures showing the largest effects, with interpretation.
4. Interactions: reference interaction plots for notable non-additive effects.
5. Summary paragraph: what this experiment establishes, with forward reference to discussion. Note that robustness checks confirm findings (Appendix~\ref{...}).

Results prose reports effect direction and magnitude, references the table/figure, and interprets economically. Do not restate numbers already visible in a float.

### Writing Workflow

- Work one subsection at a time. Outline each paragraph in bullets first. Write prose only after the user approves the outline.
- Painfully slow and deliberate. Keep asking for feedback. When in doubt, ask precise, careful questions.
- Every section follows a logical progression: setup, method, result, interpretation.

### Audit Reports

- When producing an audit document, put TLDR bullets at the top summarizing: number of issues by severity, actions required, and key findings.
