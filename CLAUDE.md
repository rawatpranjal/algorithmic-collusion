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
make paper            # Compile main paper
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
cd paper && pdflatex main.tex && cd ..
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
├── run_experiment.py      # Unified CLI (factorial design)
├── deep_dive.py           # Unified single-run analysis (trace + verbose + save)
├── generate_trace_plots.py # Single-run trace visualizations
├── generate_tables.py     # JSON → LaTeX tables + figure copy
├── generate_results.py    # Standalone results.pdf generator
├── debug_exp2.py          # Q-learning diagnostics with BNE comparison
└── debug_exp3.py          # Algorithm comparison diagnostics

paper/             # LaTeX paper source
├── tables/        # Auto-generated .tex table snippets
├── figures/       # Publication figures
└── sections/      # Paper section .tex files

archive/
├── scripts/       # Superseded utility scripts
├── estimation/    # Old per-experiment robust_analysis*.py
├── old_results/   # Stale DoubleML/CATE/grid results
├── doubleml_estimation/   # Old DoubleML est*.py files
├── random_sampling_results/  # Old random-sampling data.csv files
├── v1/            # First iteration code
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
pdflatex main.tex → paper/main.pdf
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

### Formatting Rules (hard constraints)

- No em dashes (`---`) in any LaTeX file. Use commas, semicolons, or restructure.
- No bullet lists (`\begin{itemize}`, `\begin{enumerate}`) in the paper. Use flowing prose.
- No `\paragraph{Bold.}` starters. Use `\subsection{}` or `\subsubsection{}`.
- No `\textbf{}` in running prose (only in table headers/captions).
- No data variable names in prose. Use full English names ("average revenue" not `avg_rev_last_1000`).
- No DoubleML terminology (CATE, ATE, GATE, BLP, partial-dependence).
- `\section`, `\subsection`, `\subsubsection` only. Keep hierarchy broad; avoid over-granular subsectioning.
- No stub paragraphs. When editing cuts leave 1–2 sentence orphans, merge them into the adjacent paragraph so every paragraph is at least 3–4 sentences. The paper should read as flowing prose, not a bulleted outline.

### Audience & Tone

- Primary audience: economists. Secondary: CS and statistics researchers.
- CS/stats concepts (Q-learning, contextual bandits, factorial ANOVA) need brief plain-language explanation; auction theory and equilibrium concepts can assume familiarity.
- Formal academic register. No conversational tone, no LinkedIn-style enthusiasm ("exciting", "groundbreaking", "novel").
- Balanced, measured tone. Short declarative sentences. Active voice preferred.
- No long convoluted sentences. If a sentence has more than one subordinate clause, split it.

### Claims & Confidence

- State what the data shows with confidence. Do not hedge empirical findings that are statistically significant.
- Qualify clearly when extrapolating beyond experimental scope ("the results are likely to hold", "this suggests").
- Distinguish between: (1) robust empirical findings (state directly), (2) plausible interpretations (signal uncertainty), (3) speculation (mark explicitly or omit).
- Never overstate. "X is the dominant factor" requires X to have the largest effect size. "X contributes to" if it is one of several factors.

### Structure & Flow

- Every section follows a logical progression: setup, method, result, interpretation.
- Each paragraph opens with a topic sentence stating the claim. Supporting evidence (table/figure references, numbers) follows. A concluding sentence synthesizes or transitions.
- Cross-reference liberally: "as shown in Table~\ref{...}" or "Section~\ref{...} discusses". Never duplicate content across sections; refer readers to the canonical location.
- Tables are the primary evidence format. Figures supplement tables for visual patterns (main effects, interactions). Use figures sparingly.

### Notation & Definitions

- Define all notation before first use. Every symbol in an equation must be explained in the surrounding prose.
- After each equation or definition, add 1-2 sentences of intuition explaining what it means economically or computationally.
- Move lengthy derivations to the appendix. The main text should present the result and its intuition.
- Move technical implementation details (grid sizes, convergence thresholds, hyperparameter values) to footnotes or appendix tables.

### Results Sections Pattern

Each experiment's results section follows this template:
1. One-paragraph setup: what this experiment tests, the design (factors, cells, replicates), and the key question.
2. Primary finding: state the dominant effect, reference the ranked-effects table.
3. Main effects: 1-2 figures showing the largest effects, with interpretation.
4. Interactions: reference interaction plots for notable non-additive effects.
5. Summary paragraph: what this experiment establishes, with forward reference to discussion.

Results prose should report effect direction and magnitude, reference the table, and interpret economically. Do not restate numbers that are already in a table.

### Audit Reports

- When producing an audit document, always put TLDR bullets at the top summarizing: number of issues by severity, actions required, and key findings.
