# CLAUDE.md - Algorithmic Collusion Experiments

## One-Button Execution

### Quick Test (all experiments, validates setup)
```bash
for exp in 1 2 3; do python scripts/run_experiment.py --exp $exp --quick; done
```

### Full Run (parallel, uses all CPUs)
```bash
python scripts/run_experiment.py --exp 1 --parallel
python scripts/run_experiment.py --exp 2 --parallel
python scripts/run_experiment.py --exp 3 --parallel
```

### Sequential Run (single-threaded)
```bash
python scripts/run_experiment.py --exp 1
python scripts/run_experiment.py --exp 2
python scripts/run_experiment.py --exp 3
```

## CLI Reference

| Flag | Description |
|------|-------------|
| `--exp N` | Experiment number (1, 2, or 3) |
| `--quick` | Reduced parameters for fast testing |
| `--parallel` | Use multiprocessing |
| `--workers N` | Number of parallel workers (default: cpu_count - 1) |
| `--output-dir PATH` | Custom output directory |
| `--seed N` | Random seed (default: 42) |

## Experiments Overview

| Exp | Algorithm | Valuations | Key Parameters |
|-----|-----------|------------|----------------|
| 1 | Q-learning | Constant (v=1.0) | alpha, gamma, exploration |
| 2 | Q-learning | Affiliated (eta) | eta, alpha, gamma |
| 3 | LinUCB | Affiliated (eta) | c, lambda, eta |

### Experiment 1: Constant Valuations
- All bidders have v_i = 1.0
- Tests collusion emergence with simple valuations
- Parameters: n_bidders {2,4,6}, alpha, gamma, reserve_price, exploration type

### Experiment 2: Affiliated Values with Q-Learning
- Valuations depend on signals: v_i = (1 - 0.5*eta)*s_i + 0.5*eta*mean(others)
- Tests how information structure affects collusion
- Parameters: eta {0-1}, n_bidders, alpha, gamma, reserve_price

### Experiment 3: LinUCB Bandits
- Upper Confidence Bound algorithm with linear features
- Tests alternative learning algorithm
- Parameters: c (exploration), lambda (regularization), eta

## Output Structure

```
results/
├── exp1/
│   ├── data.csv           # Summary statistics (250 runs x 2 auctions)
│   ├── param_mappings.json
│   ├── trials/            # Per-run history CSVs
│   └── q_tables/          # Q-table snapshots
├── exp2/
│   ├── data.csv
│   ├── param_mappings.json
│   ├── trials/
│   └── q_tables/
└── exp3/
    ├── data.csv
    └── param_mappings.json
```

### Key Output Columns
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
├── estimation/    # DoubleML causal analysis
└── cloud/         # Distributed execution support

scripts/
└── run_experiment.py  # Unified CLI entry point

paper/             # LaTeX paper source
archive/           # Archived code (case studies)
```

## Development

### Running Tests
```bash
# Quick validation of all experiments
for exp in 1 2 3; do python scripts/run_experiment.py --exp $exp --quick; done
```

### Adding New Experiments
1. Create `src/experiments/expN.py` following exp2.py pattern
2. Add `get_expN_tasks()` and `aggregate_expN_results()` in run_experiment.py
3. Update CLI to handle new experiment number
