# Referee Report: "Designing Auctions when Algorithms Learn to Bid"

**Recommendation: Major Revision**

This paper uses factorial experimental designs to study how reinforcement-learning agents behave in repeated auctions. The scope is ambitious: four experiments spanning Q-learning, contextual bandits, and budget-constrained pacing. The factorial approach is a genuine contribution over the one-factor-at-a-time designs common in this literature. However, I have identified serious discrepancies between the reported statistics and the underlying data, methodological concerns about the simulation design, and statistical inference issues that must be resolved before the paper can be considered for publication.

---

## 1. Critical Data-Paper Discrepancies

These are the most serious issues. Multiple numerical claims in the text do not match the estimation output files that the analysis pipeline actually produces. This pattern suggests that the prose was written against an earlier dataset and was not updated when the experiments were re-run.

### 1.1 Experiment 1: LightGBM Comparison is Backwards

The paper states (Section 5.3, res1.tex):

> "LightGBM cross-validated R² is lower than OLS R² for every response, confirming that the detectable misfit is too small to exploit with a flexible nonparametric learner."

The actual robust analysis output (`results/exp1/robust/robust_summary.txt`) shows the opposite:

| Response | OLS R² | LGBM R² | Gap |
|----------|--------|---------|-----|
| avg_rev_last_1000 | 0.809 | 0.888 | +0.078 |
| time_to_converge | 0.677 | 0.746 | +0.070 |
| avg_regret_of_seller | 0.925 | 0.970 | +0.044 |
| no_sale_rate | 0.933 | 0.990 | +0.057 |
| price_volatility | 0.906 | 0.960 | +0.054 |

LightGBM R² is **higher** than OLS R² for every response variable. The gaps range from 0.044 to 0.078, exceeding the paper's own 0.05 threshold for concluding that the linear model is sufficient (inference.tex, Section 4.4). This directly undermines the paper's claim that "the linear model with two-way interactions captures most of the signal." In fact, the data suggest meaningful nonlinear or higher-order structure that the OLS model misses, particularly for revenue (7.8 percentage points) and no-sale rate (5.7 percentage points). The authors should either (a) add higher-order terms or nonlinear transformations, or (b) acknowledge this limitation and temper their claims about model adequacy.

### 1.2 Experiment 1: R² Values Misreported

The paper states (res1.tex, Section 5.3):

> "OLS R² ranges from 0.79 (average revenue) to 0.98 (price volatility)"

The actual R² for price volatility is **0.906**, not 0.98. No response variable in Experiment 1 has R² = 0.98. The highest is winner_entropy at 0.999, and price volatility is 0.906. If the paper intends to report the range across the three "primary" responses (revenue, regret, volatility), the correct statement is "from 0.81 to 0.93."

### 1.3 Experiment 2: Observation Count Mismatch (48 vs. 192)

The inference section states "24 cells, replicated twice for 48 observations." The results section (res2.tex) discusses statistical power "with only 48 observations." However, the estimation results file (`results/exp2/estimation_results.json`) reports **n_observations = 192**, implying 8 replicates per cell, not 2. This 4x discrepancy propagates throughout the Experiment 2 discussion:

- The claim that "PRESS gaps are substantial, reflecting the limited replication" is undermined: actual PRESS gaps are 0.0007 (revenue), 0.062 (convergence), and 0.0004 (volatility), not the reported 0.60, 0.69, and 0.54.
- The claim that "Holm-Bonferroni retains none" of 31 nominally significant effects is contradicted by the robust summary, which shows multiple effects surviving Holm-Bonferroni (e.g., eta_linear:n_bidders at p < 10^-100).
- The claim that "LightGBM cross-validated R² is negative for all responses" is contradicted: LGBM R² values are comparable to OLS (e.g., 0.995 vs. 0.996 for revenue).
- The narrative about "low power inherent in a 48-observation design" is simply wrong if the actual sample is 192.

The R² for Experiment 2 revenue is 0.996, not the moderate values the text implies. The entire model adequacy discussion for Experiment 2 appears to describe a different dataset.

### 1.4 Experiment 3: Results Section Contradicts Its Own Data

The results section (res3.tex) states:

> "Auction type remains the dominant factor, with the largest absolute t-statistic (|t| ≈ 12) for both revenue and regret."

The actual estimation results show:

| Factor | |t| for revenue |
|--------|----------------|
| n_bidders | **67.0** |
| auction_type | 27.7 |
| context_richness | 13.7 |
| eta_linear | 13.5 |

Number of bidders is the dominant factor with |t| = 67, not auction type. Auction type has |t| = 27.7, not 12. The introduction correctly reports these as |t| = 67 vs. 28, but the results section tells a different story. This inconsistency between the introduction and the results section of the same paper is unacceptable.

### 1.5 Experiment 3: Model Adequacy Numbers are Stale

The paper reports (res3.tex):

> "PRESS gaps are substantially larger... 0.18 for revenue, 0.18 for regret, and 0.14 for volatility"
>
> "LightGBM cross-validated R² falls below OLS R² for revenue (0.57 vs. 0.77) and regret (0.56 vs. 0.77)"

Actual values from the robust summary:

| Response | OLS R² | PRESS Gap | LGBM R² |
|----------|--------|-----------|---------|
| Revenue | 0.900 | 0.013 | 0.923 |
| Regret | 0.915 | 0.011 | 0.939 |
| Volatility | 0.906 | 0.012 | 0.921 |

The actual R² values are ~0.90, not 0.77. PRESS gaps are ~0.01, not 0.18. LGBM R² is again **higher** than OLS R², not lower. The claim that "Experiment 3 exhibits the weakest model adequacy among all four experiments" is incorrect: Experiment 1 has lower R² for revenue (0.81 vs. 0.90).

### 1.6 Experiment 4: Observation Count Mismatch (120 vs. 400)

The paper states "replicated across 15 seeds per cell (120 runs total)." The data file contains 400 observations (50 seeds per cell). The inference section confirms this with "50 independent seeds for 400 observations," but the results section contradicts it. This inconsistency needs to be resolved.

### 1.7 Summary of Discrepancies

| Location | Paper Claims | Actual Data |
|----------|-------------|-------------|
| Exp1 R² range | 0.79-0.98 | 0.81-0.91 (primary) |
| Exp1 LGBM vs OLS | LGBM lower everywhere | LGBM higher everywhere |
| Exp2 n_obs | 48 | 192 |
| Exp2 PRESS gap (rev) | 0.60 | 0.0007 |
| Exp2 Holm survivors | None | Many (p < 10^-100) |
| Exp3 dominant factor | Auction type (|t|≈12) | n_bidders (|t|=67) |
| Exp3 OLS R² (revenue) | 0.77 | 0.90 |
| Exp3 LGBM R² (revenue) | 0.57 | 0.92 |
| Exp3 PRESS gap (rev) | 0.18 | 0.013 |
| Exp4 n_obs | 120 | 400 |

The sheer number of stale statistics raises concerns about the editorial process. I urge the authors to run a systematic audit comparing every numerical claim in the text to the current pipeline output.

---

## 2. Methodological Concerns

### 2.1 Convergence Detection is Circular

All four experiments define convergence time as the last episode at which a rolling average leaves a ±5% band around the **final** mean revenue (`exp1.py`, lines 309-324). This is circular: the convergence target is defined by the endpoint, so the measure is retrospective rather than prospective. A series that oscillates wildly before settling will be recorded as "converged early" if its oscillations happen to stay within 5% of the terminal value. Worse, if the terminal value itself is not an equilibrium (e.g., the algorithm is still drifting slowly), the measure will still report convergence.

Standard convergence diagnostics in the RL literature use prospective criteria: e.g., the Geweke test (comparing early vs. late means), or a CUSUM-based changepoint detector. The 5% threshold is arbitrary and not justified by learning theory. A sensitivity analysis varying this threshold (e.g., 1%, 2%, 10%) is needed to demonstrate that convergence time rankings are robust.

### 2.2 Action Space Discretization

Bidders choose from a coarse grid of 11 discrete bid levels in Experiment 1 (or 21 in the high condition), 101 in Experiment 2, and only **6** in Experiment 3. The paper acknowledges discretization (auctions.tex: "I choose a moderate granularity... that is fine enough to allow distinct bidding behaviors") and claims "increasing or decreasing the grid size does not materially change the results." However, no supporting evidence is provided for this claim. With 11 actions, the bid increment is 0.1, meaning agents cannot express bid differences smaller than 10% of the value range. Auction theory predicts that equilibrium bids in discretized first-price auctions differ from continuous-bid equilibria by an amount proportional to the grid spacing (Chwe, 1989). At 10% granularity, this artifact could be substantial.

Experiment 3 is particularly concerning: with only 6 bid levels (0.0, 0.2, 0.4, 0.6, 0.8, 1.0), the action space is extremely coarse. A contextual bandit choosing among 6 options is a qualitatively different learning problem than one choosing among 101 options. The finding that "contextual bandits amplify variability" may simply reflect the coarser action space, not a property of the bandit algorithm. The authors should run a sensitivity analysis varying the action space size while holding other factors constant.

### 2.3 State Space Sparsity in Experiment 2

In Experiment 2, the "signal_winner" state representation creates a state space of n_signal_bins × n_bid_actions = 11 × 101 = 1,111 states. With 101 actions per state, each bidder's Q-table has 112,211 entries. With 4 bidders and ~30,000 training episodes (the mean for Exp2), each Q-entry receives fewer than 2 visits on average. Standard Q-learning convergence guarantees require every state-action pair to be visited infinitely often (Watkins & Dayan, 1992); with fewer than 2 visits per entry, the Q-table is severely under-explored.

The paper does not discuss this sparsity issue or verify that Q-values have converged in the large-state conditions. A simple diagnostic (e.g., fraction of Q-entries visited more than 10 times) would help assess whether the Q-learning results in Experiment 2 reflect converged policies or random noise from sparse tables.

### 2.4 BNE Benchmarks Assume Continuous Signals, but Experiments Use Discrete Bins

Experiments 2 and 3 benchmark against Bayesian Nash Equilibrium bid functions derived for continuous signal distributions on [0, 1]. However, the actual experiments discretize signals into bins (`exp2.py`, line 179: `signal_bins = np.round(signals * (n_signal_bins - 1)).astype(int)`). With n_signal_bins = 11, there are only 11 possible signal values. The BNE for an 11-point discrete signal distribution differs from the continuous BNE, potentially by a non-trivial amount. The paper never acknowledges this mismatch or quantifies the discretization error in the BNE benchmark. A grid-adjusted BNE is computed in the code (`grid_adjusted_bne_revenue`), but it is unclear whether this correction is actually used in the reported metrics.

### 2.5 Exploration Implementation Bug in Experiment 1

The linear epsilon decay formula in `exp1.py` (line ~191) decays epsilon from 1.0 to 0.0, ignoring the `eps_end = 0.01` floor that is used by the exponential decay schedule:

```python
# Linear decay: eps = eps_start - (ep / decay_end) * eps_start  → decays to 0
# Exponential: eps = eps_start * (eps_end / eps_start) ** (ep / decay_end)  → decays to 0.01
```

This means the two decay schedules are not comparable: linear decay reaches zero exploration 10% before training ends (at `decay_end = 0.9 * episodes`), while exponential decay maintains a 1% exploration floor until `decay_end`. The "decay_type" factor is thus confounded with the presence or absence of residual exploration. After `decay_end`, both schedules hard-cut to `eps = 0.0`, leaving the final 10% of training with zero exploration. If the algorithm has not converged by then, there is no recovery mechanism. This implementation detail should be documented and its impact on the decay_type effect assessed.

### 2.6 Incomparable Exploration Parameters Across Algorithms in Experiment 3

Experiment 3 compares LinUCB (exploration parameter c) and Contextual Thompson Sampling (exploration parameter σ²). The "low" and "high" levels are hardcoded as c ∈ {0.5, 2.0} and σ² ∈ {0.1, 1.0}. These parameters have fundamentally different semantics: c controls the width of the UCB confidence interval, while σ² controls the posterior sampling variance. There is no theoretical or empirical justification that "low c = 0.5" is comparable in exploration intensity to "low σ² = 0.1." The factorial analysis treats these as exchangeable levels of a single "exploration_intensity" factor, but this coding conflates algorithm-specific exploration calibration with the exploration intensity treatment.

---

## 3. Statistical Inference Concerns

### 3.1 Multiple Testing Across Four Experiments

The paper tests hundreds of effects across four experiments and multiple response variables. The inference section (Section 4) describes Holm-Bonferroni and Benjamini-Hochberg corrections within each experiment, but no correction is applied across the four experiments. The cross-experiment claims in the abstract and discussion ("the number of bidders is the strongest predictor of seller outcomes in **every** experiment") constitute a joint hypothesis that should be evaluated as such. At minimum, the probability of observing n_bidders as the dominant factor in all four experiments by chance should be computed, even if informally.

### 3.2 Experiment 2 Statistical Power for Auction Format

The auction_type effect in Experiment 2 has |t| = 5.2, which the paper correctly identifies as "economically modest." However, the paper then makes the strong claim that "whether valuations are purely private or highly correlated, learning agents display similar bidding patterns, and the auction mechanism overshadows valuation interdependence." An effect size of 0.0036 (the auction_type coefficient for revenue) is tiny: it implies switching from second-price to first-price changes mean revenue by 0.72% (= 2 × 0.0036 / mean). Calling this "overshadowing" stretches the interpretation. The null result on eta should also be qualified: with the actual observation count of 192, the MDE for excess_regret is 11.7% of the mean, meaning the design lacks power to detect economically meaningful effects on this response.

### 3.3 Low R² for Key Response Variables

Several response variables central to the paper's narrative have low R²:

- Experiment 1, time_to_converge: R² = 0.68
- Experiment 4, liquid_welfare: R² = 0.22
- Experiment 4, cross_episode_drift: R² = 0.24

The factorial model explains less than a quarter of the variance in liquid welfare and cross-episode drift for Experiment 4. The discussion of PoA and progressive collusion dynamics relies on these poorly-explained variables. The paper should acknowledge that the factorial design captures only a fraction of the outcome variation for these responses, likely because omitted factors (seed-specific valuation draws, random initialization) dominate.

### 3.4 Model Misfit is Detectable and Exploitable (Experiment 1)

The paper dismisses the significant lack-of-fit test in Experiment 1 by arguing that LightGBM cannot exploit the misfit. As documented in Section 1.1 above, LightGBM **does** exploit it: LGBM R² exceeds OLS R² by 5-8 percentage points for every response. This means the linear model with two-way interactions is a biased approximation, and the reported coefficients may be attenuated or inflated by omitted nonlinear terms. At minimum, the authors should report LGBM feature importance rankings and check whether the dominant factors change under the nonparametric model.

---

## 4. Per-Experiment Critiques

### 4.1 Experiment 1

**Strengths:** The 2^(11-1) Resolution V design is well-chosen for screening 11 factors. The sample size (2,048) provides excellent power.

**Weaknesses:**
- The "minimal" state representation (1-state MDP) is not meaningful learning: agents in this condition have no ability to condition on history and are simply exploring action frequencies. Labelling this as "information feedback = minimal" obscures the fact that these agents are effectively playing a bandit problem, not an MDP. The information_feedback factor is thus confounded with the learning paradigm (bandit vs. MDP).
- The paper does not report the fraction of runs where convergence was not reached within the episode budget (100K episodes). If a substantial fraction did not converge, the revenue and regret metrics reflect transient rather than equilibrium behavior.

### 4.2 Experiment 2

**Strengths:** The affiliated valuation model is economically motivated and enables BNE benchmarking.

**Weaknesses:**
- As documented above, the entire model adequacy section appears to describe a different dataset (48 obs vs. 192 actual).
- The null result on eta is the most striking finding, but it is under-explored. Does the Q-learning agent's state representation even encode information about affiliation? If agents observe only their own signal and the previous winner bid, they have no way to learn about the common-value component. The null result may simply reflect that the state space is too impoverished to detect affiliation, not that affiliation does not matter.
- The mixed-level design (3 × 2³) with only 4 factors is a dramatic reduction from Experiment 1's 11 factors. The paper does not explain why most of Experiment 1's factors were dropped (learning rate, discount factor, exploration type, update mode, etc.) rather than fixed at identified "good" levels. If these factors were fixed, at what levels? The paper says "Q-learning hyperparameters fixed at levels identified in Experiment 1" but does not report these levels.

### 4.3 Experiment 3

**Strengths:** Testing contextual bandits alongside Q-learning is valuable for external validity.

**Weaknesses:**
- The 6-action discretization is far coarser than Experiments 1-2 (11-101 actions). A bandit with 6 arms is a qualitatively simpler learning problem. The finding that "algorithmic sophistication does not improve seller outcomes" may be an artifact of this design choice rather than a property of LinUCB.
- The paper claims "certain runs converge to near-zero revenue or universal no-sales." What fraction of runs exhibit this pathological behavior? If it is substantial, the OLS model (which estimates conditional means) may be poorly suited to a bimodal response distribution. A mixture model or zero-inflated regression would be more appropriate.
- As noted in Section 1.4, the results section incorrectly identifies auction type as the dominant factor when the data clearly show n_bidders dominates.
- The claim that "more bidders worsen first-price performance under LinUCB" (a reversal from Experiments 1-2) is a key finding but is reported without a mechanistic explanation grounded in the algorithm. The proposed explanation ("optimism-based exploration may reinforce risk-averse bidding") is speculative and not supported by analysis of the agents' actual bid distributions or uncertainty estimates.

### 4.4 Experiment 4

**Strengths:** Budget-constrained pacing is highly relevant to real ad exchanges. The connection to PoA literature is well-drawn.

**Weaknesses:**
- The offline optimum used for the PoA calculation is a greedy allocation, not a true offline optimum. A greedy algorithm that assigns items to the highest-value bidder with remaining budget is suboptimal when budget constraints bind: it may exhaust a high-value bidder's budget on low-value items early, leaving no budget for high-value items later. The PoA values reported may therefore be biased toward 1 (appearing more efficient than they are) because the benchmark itself is suboptimal.
- The hard budget constraint in the code (`bid = min(raw_bid, remaining_budget)`) breaks the theoretical correspondence between dual variables and bids. When the remaining budget is less than the raw bid, the agent bids its remaining budget regardless of the dual variable. This means the Lagrangian dual ascent is not controlling the agent's behavior in budget-constrained rounds, undermining the claim that "dual-based pacing agents" are being studied.
- With only 3 binary factors, the full 2³ factorial has 8 design cells. Even with 50 seeds per cell, the design cannot test any factors beyond auction type, objective, and n_bidders. This means the experiment cannot identify the effects of pacing algorithm parameters (step size, update frequency, initial multiplier) that are listed in Table 5 but not varied.
- The burn-in of 10 episodes (out of 100) is 10%. The paper does not verify that dual variables have converged after 10 episodes. A sensitivity analysis varying the burn-in (5, 10, 20, 50 episodes) would strengthen the results.

---

## 5. Framing and Interpretation Issues

### 5.1 Abstract Overclaims Universality

The abstract states: "market thickness dominates all other design levers." This is a strong universality claim. While n_bidders is the largest main effect in every experiment, in Experiment 1 the exploration strategy has |t| = 44.8 versus n_bidders at |t| = 57.5, a ratio of only 1.3x. Calling n_bidders "dominant" when it is only 30% larger than exploration strategy overstates the finding. In Experiments 3 and 4, n_bidders is more clearly dominant, but the paper should acknowledge the heterogeneity rather than presenting a uniform narrative.

### 5.2 The First-Price "Underperformance" Framing

The paper frames first-price auctions as systematically underperforming. But in Experiment 2, the gap is 0.5% (Table 7 in the discussion: 0.546 vs. 0.549). This is economically negligible. And in Experiment 4, auction format is statistically insignificant for the PoA (the efficiency measure). The discussion should more carefully distinguish between "statistically significant" and "economically meaningful" underperformance. The 42.3% gap in Experiment 3 and the 14.4% gap in Experiment 1 are meaningful; the 0.5% gap in Experiment 2 is not.

### 5.3 Revenue Equivalence Violation is Expected, Not Surprising

The discussion states that the first-price shortfall "directly contradicts the revenue equivalence theorem (Myerson, 1981)." Revenue equivalence requires risk-neutral bidders with independent private values who play Bayesian Nash Equilibria. Q-learning agents do not play BNE. The violation is therefore not a "contradiction" of the theorem but simply a confirmation that the theorem's assumptions do not hold. This is a well-known result in the algorithmic game theory literature (e.g., Banchio & Skrzypacz, 2022). The paper should frame this as "consistent with prior findings" rather than as a contradiction.

### 5.4 Typo in Introduction

The introduction (line 5) contains a typo: "informtion" should be "information."

---

## 6. Missing Analyses

1. **Episode budget sensitivity.** All experiments use a single episode budget (e.g., 100K for Exp1). An ablation at 10K, 50K, 200K, and 500K episodes would reveal whether results reflect converged equilibria or transient dynamics.

2. **Action space sensitivity.** As discussed in Section 2.2, no systematic sensitivity analysis is reported despite the claim that "increasing or decreasing the grid size does not materially change the results."

3. **State space sufficiency.** No analysis verifies that the chosen state representations (e.g., previous winner bid) are sufficient statistics for the agents' decision problems. A comparison of state representations (none, own bid, winner bid, full bid profile) would be informative.

4. **Equilibrium verification.** The code computes BNE bid functions but never compares the agents' learned policies to these benchmarks. A plot of learned bid-vs-signal functions overlaid on the BNE prediction would immediately reveal whether agents are approximating equilibrium play or converging to qualitatively different strategies.

5. **Distribution of outcomes.** The paper analyzes conditional means (OLS) but several experiments appear to have multimodal outcome distributions (Experiment 3's zero-revenue runs). Histograms or kernel density plots of the response variables would help assess whether mean-based inference is appropriate.

6. **Seed-level analysis.** With 50 seeds per cell in Experiment 4 and 2-8 replicates in Experiments 1-3, the paper could report within-cell variance decomposition to distinguish between design-driven variation (interesting) and seed-driven noise (uninteresting).

---

## 7. Minor Issues

- The parameter tables in the experiments section (Tables 1-4) list ranges like "e.g. {0.001, 0.005, 0.01, 0.05, 0.1}" with "e.g." qualifiers. The actual values used should be reported precisely.
- The convergence time comparison table in the discussion (Table 8) reports median = 1,000 and mean = 1,000 for Experiments 1 and 3. If these are normalized values, the normalization should be stated. If they are literal episode counts, it is suspicious that two different experiments produce identical convergence statistics.
- The paper uses "Experiment 4 shifts from tabular learners to dual-based autobidding agents" but never defines "dual-based" for the economics audience. A brief explanation of Lagrangian duality in the context of constrained bidding would help readability.
- The cross-experiment revenue table (Table 7) reports revenues in different units: Experiments 1-3 use per-round revenue in [0, 1], while Experiment 4 uses absolute revenue in the hundreds. This makes cross-experiment comparison unintuitive.
- Figure captions could be more informative. For example, Figure 1's caption should state what the BNE prediction is for the specific parameter combination shown.

---

## 8. Overall Assessment

The paper tackles an important question with a systematic factorial approach that improves on the ad hoc parameter sweeps common in this literature. The core finding that competitive pressure dominates auction format is policy-relevant and appears robust across the experimental conditions, even after accounting for the data discrepancies documented above. However, the current manuscript has a serious data integrity problem: numerous statistics in the text do not match the underlying data files. Whether this reflects stale prose from an earlier data generation or a pipeline error, it must be resolved comprehensively before publication. The methodological concerns (convergence detection, discretization, state space sparsity) are addressable with additional robustness checks and sensitivity analyses. I encourage the authors to undertake a complete audit of every numerical claim against the current pipeline output, add the missing sensitivity analyses, and resubmit.
