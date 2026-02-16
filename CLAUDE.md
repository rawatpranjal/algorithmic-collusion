# CLAUDE.md - Algorithmic Collusion Experiments

## One-Button Execution

### Quick Test (all experiments, validates setup)
```bash
for exp in 1 2 3; do python3 scripts/run_experiment.py --exp $exp --quick --parallel; done
```

### Full Run (parallel, 2 replicates per cell)
```bash
python3 scripts/run_experiment.py --exp 1 --parallel
python3 scripts/run_experiment.py --exp 2 --parallel
python3 scripts/run_experiment.py --exp 3 --parallel
```

### Custom Replicates
```bash
python3 scripts/run_experiment.py --exp 1 --parallel --replicates 5
```

## CLI Reference

| Flag | Description |
|------|-------------|
| `--exp N` | Experiment number (1, 2, or 3) |
| `--quick` | Quick test: 2^4=16 cells, 1 replicate, 1000 episodes |
| `--parallel` | Use multiprocessing |
| `--workers N` | Number of parallel workers (default: cpu_count // 2) |
| `--replicates N` | Replicates per cell (default: 2) |
| `--output-dir PATH` | Custom output directory |
| `--seed N` | Random seed (default: 42) |
| `--cloud` | Run on Google Cloud VM |
| `--detached` | Fire-and-forget mode (VM self-deletes after completion) |

## Factorial Design

Experiments use 2^k factorial designs (coded -1/+1) instead of random sampling.

| Exp | Design | Factors | Cells | × 2 reps | Total |
|-----|--------|---------|-------|----------|-------|
| 1 | 2^10 full | 10 | 1024 | 2048 | ~2-4 hrs |
| 2 | 2^(11-1) half-fraction (Res V) | 11 | 1024 | 2048 | ~3-5 hrs |
| 3 | 2^8 full | 8 | 256 | 512 | ~1-2 hrs |

Quick mode: 2^4 = 16 cells × 1 replicate for smoke testing.

## Pipeline: From Simulation to Paper

### Full pipeline (after experiments have been run)
```bash
# 1. Run factorial ANOVA analysis (produces JSON + plots)
PYTHONPATH=src python3 src/estimation/est1.py
PYTHONPATH=src python3 src/estimation/est2.py
PYTHONPATH=src python3 src/estimation/est3.py

# 2. Generate LaTeX tables and copy publication figures
python3 scripts/generate_tables.py

# 3. Generate standalone results PDF
python3 scripts/generate_results.py

# 4. Compile paper
cd paper && pdflatex main.tex && cd ..
```

### What each step produces
- **est*.py** → `results/expN/estimation_results.json`, `pareto_charts/*.png`, `main_effects/*.png`, `interaction_plots/*.png`, `normal_prob_plots/*.png`, `residuals/*.png`
- **generate_tables.py** → `paper/tables/expN_coefficients.tex`, `paper/tables/expN_model_fit.tex`, `paper/tables/expN_significant.tex`, copies key figures to `paper/figures/`
- **generate_results.py** → `paper/results.pdf` (comprehensive factorial analysis report with all tables and plots)

## Cloud Execution

### Running on GCP (fire-and-forget)
```bash
# Test with small N first
python3 scripts/run_experiment.py --exp 1 --cloud --detached --quick

# Full experiments (can close terminal after launch)
python3 scripts/run_experiment.py --exp 1 --cloud --detached
python3 scripts/run_experiment.py --exp 2 --cloud --detached
python3 scripts/run_experiment.py --exp 3 --cloud --detached
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
gsutil cp gs://${PROJECT}-collusion-experiments/results/exp3/$TIMESTAMP/exp3/data.csv results/exp3/data.csv

# 2. Run full pipeline
PYTHONPATH=src python3 src/estimation/est1.py
PYTHONPATH=src python3 src/estimation/est2.py
PYTHONPATH=src python3 src/estimation/est3.py
python3 scripts/generate_tables.py
python3 scripts/generate_results.py
cd paper && pdflatex main.tex && cd ..
```

## Experiments Overview

| Exp | Algorithm | Valuations | Design | Factors |
|-----|-----------|------------|--------|---------|
| 1 | Q-learning | Constant (v=1.0) | 2^10 full | auction_type, alpha, gamma, reserve_price, init, exploration, asynchronous, n_bidders, median_opp, winner_bid |
| 2 | Q-learning | Affiliated (eta) | 2^(11-1) Res V | All Exp1 factors + eta |
| 3 | LinUCB | Affiliated (eta) | 2^8 full | auction_type, eta, c, lam, n_bidders, reserve_price, use_median, use_winner |

## Output Structure

```
results/
├── exp1/
│   ├── data.csv                  # Factorial design data with coded columns
│   ├── design_info.json          # Factor definitions
│   ├── param_mappings.json
│   ├── estimation_results.json   # OLS/ANOVA results
│   ├── analysis_stdout.txt       # Full analysis log
│   ├── pareto_charts/*.png       # |t-statistic| bar charts
│   ├── normal_prob_plots/*.png   # Half-normal effect identification
│   ├── main_effects/*.png        # Mean response at -1 vs +1
│   ├── interaction_plots/*.png   # Top 2-way interactions
│   └── residuals/*.png           # QQ + residuals vs fitted
├── exp2/ (same structure)
└── exp3/ (same structure)
```

### Key Output Columns
- `*_coded`: Factor coded as -1 (low) or +1 (high) — used for analysis
- `avg_rev_last_1000`: Average revenue in final 1000 episodes
- `time_to_converge`: First episode where revenue stays in ±5% band
- `avg_regret_of_seller`: Mean (1 - revenue) across all episodes
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
│   └── exp3.py    # LinUCB bandits
├── estimation/
│   ├── factorial_analysis.py  # Shared OLS + ANOVA engine
│   ├── est1.py                # Thin wrapper for Exp1
│   ├── est2.py                # Thin wrapper for Exp2
│   └── est3.py                # Thin wrapper for Exp3
└── cloud/         # Distributed execution support

scripts/
├── run_experiment.py    # Unified CLI (factorial design)
├── generate_tables.py   # JSON → LaTeX tables + figure copy
└── generate_results.py  # Standalone results.pdf generator

paper/             # LaTeX paper source
├── tables/        # Auto-generated .tex table snippets
├── figures/       # Publication figures
└── sections/      # Paper section .tex files

archive/
├── doubleml_estimation/   # Old DoubleML est*.py files
├── random_sampling_results/  # Old random-sampling data.csv files
├── v1/            # First iteration code
└── v2/            # Second iteration code

docs/
├── plans/         # Development roadmaps
├── literature/    # Reference papers
└── transcriptions/
```

## Development

### Running Tests
```bash
# Quick validation of all experiments
for exp in 1 2 3; do python3 scripts/run_experiment.py --exp $exp --quick --parallel; done
```

### Adding New Experiments
1. Create `src/experiments/expN.py` following exp2.py pattern
2. Define `EXPN_FACTORS` dict in `scripts/run_experiment.py`
3. Add `get_expN_tasks()` and `aggregate_expN_results()`
4. Create `src/estimation/estN.py` wrapper with coded cols and response cols
5. Update CLI to handle new experiment number
