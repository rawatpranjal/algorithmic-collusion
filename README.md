# Algo-Collude

This repository contains data and analysis for the paper:  
**"Designing Auctions when Algorithms Learn to Bid: The Critical Role of Payment Rules"**  
[arXiv link](https://arxiv.org/abs/2306.09437)

### Abstract  
We run a fully randomized experiment with Q-learning bidders participating in repeated auctions. The first price auction sees coordinated bid suppression, with winning bids averaging about 20% below true values, while the second price auction aligns bids to values, reduces learning volatility, and speeds convergence. Regression and machine learning methods confirm that payment rules are critical for efficiency—especially with fewer bidders, high discount factors, asynchronous learning, and coarse bid spaces.

---

## How to Use
1. **Run** `code/01_run_experiment.py` to generate data in `code/data/`.  
2. **Run** `code/02_analyse_experiment.py` to produce figures in `code/figures/` and regression outputs.

---

## Repository Structure
```
docs/
code/
  ├─ data/              # Stores output data
  ├─ figures/           # Stores output plots/figures
  ├─ 01_run_experiment.py
  └─ 02_analyse_experiment.py
README.md
```
