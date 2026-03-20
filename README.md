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
| 1a | Q-learning | Constant (v=1) | 2^(10-1) Res V | Which structural and learning parameters drive bid suppression? |
| 1b | Q-learning | Affiliated (eta) | 3 x 2^3 mixed | Does valuation affiliation alter revenue outcomes? |
| 2a | LinUCB | Affiliated (eta) | 3 x 2^7 mixed | How does contextual exploration interact with auction design? |
| 2b | Thompson | Affiliated (eta) | 3 x 2^5 mixed | Does the exploration mechanism change bid suppression patterns? |
| 3a | Dual Pacing | LogNormal | 2^6 full | How do budget constraints reshape auction format effects? |
| 3b | PI Pacing | LogNormal | 2^6 full | Does the pacing algorithm alter revenue and welfare outcomes? |

## Pipeline

```bash
make experiments  # Run all experiments (2 reps/cell)
make analyze      # Factorial ANOVA (JSON + plots)
make robust       # Robustness checks (HC3, multiplicity, PRESS, LightGBM)
make traces       # Single-run trace plots for paper figures
make tables       # Generate LaTeX tables + copy figures
make paper        # Compile paper
make arxiv        # Package paper for arXiv submission
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
    exp3.py          # Shared simulation engine for 3a/3b
    exp4a.py         # Dual pacing autobidding
    exp4b.py         # PI controller pacing
  estimation/
    factorial_analysis.py   # Shared OLS + ANOVA engine
    robust_analysis.py      # 13 robustness checks
    est1.py .. est4b.py     # Per-experiment wrappers

scripts/
  run_experiment.py         # Unified CLI (factorial design)
  deep_dive.py              # Single-run trace analysis
  generate_trace_plots.py   # Learning trajectory figures
  generate_tables.py        # JSON -> LaTeX tables + figure copy
  generate_results.py       # Standalone results PDF
  make_arxiv.py             # Package paper for arXiv submission

paper/                      # LaTeX source
  sections/                 # Paper section .tex files
  tables/                   # Auto-generated .tex tables
  figures/                  # Publication figures

results/
  exp{1,2,3a,3b,4a,4b}/
    data.csv                # Factorial design data
    estimation_results.json # OLS/ANOVA results
    robust/                 # Robustness diagnostics
    pareto_charts/          # Ranked effect bar charts
    main_effects/           # Factor main effect plots
    interaction_plots/      # Two-way interaction plots
```

## Key Findings

1. **Structural parameters dominate algorithmic design choices** across all algorithm classes. The number of bidders and auction format consistently explain more variance than learning rates, discount factors, or exploration parameters.
2. **Competition is the strongest revenue predictor** in unconstrained settings (Experiments 1a, 1b, 2a, 2b). Budget tightness takes over as the dominant factor under pacing constraints (Experiments 3a, 3b).
3. **Auction format effects are context-dependent.** First-price auctions suppress revenue under Q-learning and contextual bandits, but the effect reverses under budget-constrained pacing, where first-price formats yield higher revenue.
4. **No single auction format is universally superior.** The optimal format depends on the bidding technology deployed, undermining blanket policy recommendations.
5. **Valuation affiliation has negligible impact** on algorithmic bidding outcomes, contrary to classical auction theory predictions about the winner's curse.

## Citation

Rawat, P. (2023). Designing Auctions when Algorithms Learn to Bid: An Experimental Approach. arXiv preprint arXiv:2306.09437.
