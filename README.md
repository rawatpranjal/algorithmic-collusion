# Algorithmic Collusion

This repository accompanies **"Designing Auctions when Algorithms Learn to Bid: An Experimental Approach"**
[arXiv link](https://arxiv.org/abs/2306.09437)

## Quick Start

```bash
pip install -r requirements.txt
make smoke       # Quick-test all experiments (~5 min)
make all         # Full pipeline: experiments -> analysis -> paper
```

## Experiments

| Exp | Algorithm | Valuations | Design | Key Question |
|-----|-----------|------------|--------|--------------|
| 1 | Q-learning | Constant (v=1) | 2^(11-1) Res V | Which learning parameters drive collusion? |
| 2 | Q-learning | Affiliated (eta) | 3 x 2^3 mixed | Does valuation structure alter collusion? |
| 3 | LinUCB + CTS | Affiliated (eta) | 3 x 2^7 mixed | Does sophisticated exploration help sellers? |
| 4 | Dual Pacing | LogNormal | 2^3 full | Do budget constraints mitigate or amplify collusion? |

## Pipeline

```bash
make experiments  # Run all experiments (2 reps/cell)
make analyze      # Factorial ANOVA (JSON + plots)
make robust       # Robustness checks (HC3, multiplicity, PRESS, LightGBM)
make traces       # Single-run trace plots for paper figures
make tables       # Generate LaTeX tables + copy figures
make paper        # Compile paper
```

Individual experiments: `make exp1`, `make analyze1`, `make robust1`, etc.

Options: `make exp1 REPS=5 SEED=123 WORKERS=8`

### Deep Dive (Single Run)

```bash
make dive1                                          # Default params
PYTHONPATH=src python3 scripts/deep_dive.py --exp 2 --param eta=1.0 --verbose  # Custom
```

## Repository Structure

```
src/
  experiments/
    exp1.py          # Q-learning, constant valuations
    exp2.py          # Q-learning, affiliated valuations
    exp3.py          # LinUCB/CTS contextual bandits
    exp4.py          # Dual pacing autobidding
  estimation/
    factorial_analysis.py   # Shared OLS + ANOVA engine
    robust_analysis.py      # 13 robustness checks
    est1.py .. est4.py      # Per-experiment wrappers

scripts/
  run_experiment.py         # Unified CLI (factorial design)
  deep_dive.py              # Single-run trace analysis
  generate_trace_plots.py   # Learning trajectory figures
  generate_tables.py        # JSON -> LaTeX tables + figure copy
  generate_results.py       # Standalone results PDF

paper/                      # LaTeX source
  sections/                 # Paper section .tex files
  tables/                   # Auto-generated .tex tables
  figures/                  # Publication figures

results/
  exp{1,2,3,4}/
    data.csv                # Factorial design data
    estimation_results.json # OLS/ANOVA results
    robust/                 # Robustness diagnostics
    pareto_charts/          # Ranked effect bar charts
    main_effects/           # Factor main effect plots
    interaction_plots/      # Two-way interaction plots
```

## Key Findings

1. **First-price auctions systematically underperform** across all algorithm classes, valuation structures, and budget regimes.
2. **Exploration mechanisms matter more than rates**: Boltzmann exploration mitigates collusion; UCB-based exploration exacerbates it.
3. **Budget constraints impose discipline** but introduce a distinct form of bid suppression through pacing dynamics.
4. **Valuation structure (affiliation) has negligible impact**, contradicting classical auction theory predictions about winner's curse effects.
5. **Collusion is symmetric**: first-price auctions produce higher winner entropy (equitable rotation at lower prices) rather than asymmetric predation.

## Citation

Rawat, P. (2023). Designing Auctions when Algorithms Learn to Bid: An Experimental Approach. arXiv preprint arXiv:2306.09437.
