# Algorithmic Collusion

This repository accompanies **"Designing Auctions when Algorithms Learn to Bid: The Critical Role of Payment Rules"**
[arXiv link](https://arxiv.org/abs/2306.09437)

## Abstract

Algorithms are increasingly being used to automate participation in online markets. Banchio and Skrzypacz (2022) demonstrate how exploration under identical valuation in first-price auctions may lead to spontaneous coupling into sub-competitive bidding. However, it is an open question if these findings extend to affiliated values, optimal exploration, and specifically which algorithmic details play a role in facilitating algorithmic collusion. This paper contributes to the literature by generating robust stylized facts to cover these gaps. I conduct a set of fully randomized experiments in a controlled laboratory setup and apply double machine learning to estimate granular conditional treatment effects of auction design on seller revenues. I find that first-price auctions lead to lower seller revenues, slower convergence, and higher seller regret under identical values, affiliated values, and also under optimal exploration. There is more possibility of such tacit collusion under fewer bidders, lower learning rates, and higher discount factors. This evidence suggests that programmatic auctions, e.g. the Google Ad Exchange, which depend on first-price auctions, might be susceptible to coordinated bid suppression and significant revenue losses.

## Repository Structure

```
algorithmic-collusion/
├── paper/                  # LaTeX paper and presentation
│   ├── main.tex
│   ├── presentation.tex
│   ├── sections/
│   └── planning/
├── src/
│   ├── experiments/        # Experiment simulation code
│   │   ├── exp1.py         # Q-learning, constant valuations
│   │   ├── exp2.py         # Q-learning, stochastic valuations with affiliation
│   │   └── exp3.py         # LinUCB bandits, stochastic valuations
│   └── estimation/         # DoubleML analysis scripts
│       ├── est1.py
│       ├── est2.py
│       └── est3.py
├── scripts/
│   └── run_experiment.py   # Unified CLI entry point
├── results/                # Experiment output data
│   ├── exp1/
│   ├── exp2/
│   └── exp3/
├── figures/                # Generated figures for paper
├── archive/                # Old iteration code
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Running Experiments

All experiments are run through the unified CLI at `scripts/run_experiment.py`.

### Quick Tests

```bash
python scripts/run_experiment.py --exp 1 --quick
python scripts/run_experiment.py --exp 2 --quick
python scripts/run_experiment.py --exp 3 --quick
```

### Full Experiments (Parallel)

```bash
python scripts/run_experiment.py --exp 1 --parallel
python scripts/run_experiment.py --exp 2 --parallel
python scripts/run_experiment.py --exp 3 --parallel
```

### Full Experiments (Sequential)

```bash
python scripts/run_experiment.py --exp 1
python scripts/run_experiment.py --exp 2
python scripts/run_experiment.py --exp 3
```

Output is saved to `results/exp{N}/`.

### CLI Reference

| Flag | Description |
|------|-------------|
| `--exp N` | Experiment number (1, 2, or 3) |
| `--quick` | Reduced parameters for fast testing |
| `--parallel` | Use multiprocessing |
| `--workers N` | Number of parallel workers |

### Running Estimation/Analysis

After running experiments, analyze results with:

```bash
python src/estimation/est1.py
python src/estimation/est2.py
python src/estimation/est3.py
```

## Experiments Overview

- **Experiment 1**: Q-learning with constant valuations (vi=1.0), varying information regimes
- **Experiment 2**: Q-learning with affiliated values, varying eta (affiliation parameter)
- **Experiment 3**: LinUCB contextual bandits with affiliated values

## Paper Compilation

```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Citation

If you find this work useful, please cite the arXiv preprint above.

Rawat, P. (2023). Designing Auctions when Algorithms Learn to Bid: The critical role of Payment Rules. arXiv preprint arXiv:2306.09437.
