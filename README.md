# Algo-Collude

This repository contains data and analysis for the paper:  
**"Designing Auctions when Algorithms Learn to Bid: The Critical Role of Payment Rules"**  
[arXiv link](https://arxiv.org/abs/2306.09437)

## Abstract  
I run a fully randomized experiment with Q-learning bidders participating in repeated auctions.  
- **First-price auction**: susceptible to coordinated bid suppression, with winning bids ~20% below true values.  
- **Second-price auction**: aligns bids to values, reduces learning volatility, speeds convergence.  
Regression and machine learning methods confirm the **critical** role of payment rules—especially with fewer bidders, high discount factors, asynchronous learning, and coarse bid spaces.

---

## How to Use
1. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Main Script**  
   ```bash
   python code/main.py
   ```
   - Generates sample plots for first-price vs. second-price auctions in `code/figures/`:
     - `first-price-visual.png`
     - `second-price-visual.png`
   - Randomly samples parameter sets (K=10 demo) and saves data to `code/data/data.csv`.
   - Produces boxplots, runs t-tests, regressions, and advanced ML analyses, with output in `code/figures/`.

3. **Inspect Outputs**  
   - **Figures**: `code/figures/` folder for:
     - Boxplots (`boxplot_bid2val.png`, `boxplot_vol.png`, `boxplot_episodes.png`)
     - Regression tables in `.tex` format (`regression_1.tex`, `regression_2.tex`)
     - Machine learning effect plots (`orf_treatment_effects.png`)
     - Example convergence figures (`first-price-visual.png`, `second-price-visual.png`)
   - **Data**: `code/data/data.csv` for aggregated experimental results.

---