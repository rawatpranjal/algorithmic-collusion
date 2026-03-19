# Summary of Findings

## Executive Summary

Six experiments tested how three families of bidding algorithms (Q-learning, contextual bandits, budget-pacing controllers) interact with auction format choice across varying market structures. The central finding is that **structural parameters dominate auction format effects across all algorithm families**, but the direction and magnitude of format effects depend critically on which algorithm bidders use. The number of bidders is the universal dominant factor in unconstrained settings (Experiments 1--3b), while budget constraints take over as the primary determinant under autobidding (Experiments 4a--4b). Auction format is statistically negligible under Q-learning with constant valuations, modestly favors second-price under affiliated values and bandits, and reverses to favor first-price under budget-pacing. This algorithm-dependence means policy conclusions about auction design are not portable across bidding technologies.

---

## Per-Experiment Findings

### Experiment 1: Q-learning with Constant Valuations

**Design:** 2^(10-1) Resolution V half-fraction, 10 factors, 512 cells, 2 reps/cell (1,024 runs)

**Primary revenue response (avg_rev_last_1000):** R² = 0.423, Adj-R² = 0.390. Grand mean = 0.816.

**What R² = 0.42 means:** The 10 factors and their pairwise interactions explain 42% of the variation in converged revenue. The remaining 58% is unexplained variance. This is the lowest R² across all experiments, consistent with Q-learning being the least deterministic algorithm tested.

**MDE:** 1.69% of grand mean (0.014 revenue units). The large sample (n=1024) provides excellent power to detect small effects.

#### Effects Table

| Rank | Factor | Coefficient | t-stat | p-value | Effect (% of mean) | Survives Holm? | Practical Significance |
|------|--------|-------------|------|---------|---------------------|----------------|------------------------|
| 1 | n_bidders | +0.087 | 17.8 | 3.0e-61 | 21.3% | Yes | **High** |
| 2 | gamma (discount) | -0.045 | 9.2 | 1.7e-19 | 11.1% | Yes | Moderate |
| 3 | asynchronous | +0.037 | 7.5 | 2.0e-13 | 9.0% | Yes | Moderate |
| 4 | reserve_price | +0.025 | 5.1 | 4.1e-07 | 6.1% | -- | Moderate |
| 5 | info_feedback | +0.018 | 3.7 | 2.2e-04 | 4.5% | -- | Small |
| -- | **auction_type** | **+0.003** | **0.6** | **0.529** | **0.8%** | **No** | **None** |

*Effect (% of mean) = 2 * |coefficient| / grand_mean * 100, representing the full swing from factor level -1 to +1.*

#### Main Effects

**n_bidders (+0.087, 21.3%):** More bidders increase revenue by 21.3% of grand mean, the dominant factor. Direction consistent with standard competitive-pressure predictions from auction theory.

**gamma (-0.045, 11.1%):** Higher discount factors *reduce* revenue. Patient agents (high gamma) produce lower converged revenue than impatient ones (low gamma). Direction consistent with the theoretical prediction that patient learners can sustain lower-bid equilibria.

**asynchronous (+0.037, 9.0%):** Asynchronous updating increases revenue by 9.0% of grand mean. Simultaneous updating produces lower revenue, consistent with the coordination literature's prediction that synchronization facilitates tacit collusion.

**reserve_price (+0.025, 6.1%):** Higher reserve prices increase revenue. With constant valuations (v=1.0), the reserve sets a floor on winning bids.

**auction_type (+0.003, 0.8%):** The FPA-SPA revenue difference is economically negligible (0.8% of grand mean, well below the MDE of 1.69%). Consistent with the revenue equivalence theorem for symmetric constant-value settings.

#### Top Interactions

| Interaction | Coefficient | t-stat | p-value |
|-------------|-------------|------|---------|
| gamma x n_bidders | +0.030 | 6.1 | 1.7e-09 |
| auction_type x n_bidders | -0.028 | 5.6 | 2.4e-08 |
| reserve_price x info_feedback | -0.022 | 4.5 | 8.8e-06 |

**gamma x n_bidders conditional means:**

|  | 2 bidders | 4 bidders |
|--|-----------|-----------|
| Low gamma (impatient) | 0.804 | 0.918 |
| High gamma (patient) | 0.654 | 0.887 |

*What it means:* The gamma effect depends on market size. With 4 bidders the gap is 3.4%; with 2 bidders it widens to 18.4%. The discount factor matters primarily in thin markets.

**auction_type x n_bidders conditional means:**

|  | 2 bidders | 4 bidders |
|--|-----------|-----------|
| SPA | 0.698 | 0.927 |
| FPA | 0.759 | 0.878 |

*What it means:* The FPA-SPA comparison reverses with market size. In thin 2-bidder markets, FPA outperforms SPA (0.759 vs 0.698, +8.7%). In competitive 4-bidder markets, SPA pulls ahead (0.927 vs 0.878, +5.6%). Even though auction format's *average* effect is null, it has meaningful conditional effects.

**reserve_price x info_feedback conditional means:**

|  | No info | Full info |
|--|---------|-----------|
| Low reserve | 0.751 | 0.831 |
| High reserve | 0.844 | 0.837 |

*What it means:* Information feedback and reserve price are substitutes. Without reserves, full info increases revenue (0.831 vs 0.751). With a high reserve, additional information has no benefit (0.837 vs 0.844, negligible difference).

**Other responses:** no_sale_rate (R² = 0.424), price_volatility (R² = 0.447), winner_entropy (R² = 0.459) all show similar factor hierarchies with n_bidders dominant.

**Robustness:** Core main effects (n_bidders, gamma, asynchronous) survive Holm-Bonferroni. Pred-R² = 0.354 (gap = 0.069). LGBM R² = 0.455 (gap = +0.032, mild nonlinearity).

---

### Experiment 2: Q-learning with Affiliated Valuations

**Design:** 3 x 2³ mixed-level, 5 factors (incl. eta with linear+quadratic contrasts), 24 cells, 8 reps/cell (192 runs)

**Primary revenue response (avg_rev_last_1000):** R² = 0.404, Adj-R² = 0.357. Grand mean = 0.459.

**What R² = 0.40 means:** Similar explanatory power to Experiment 1, consistent with Q-learning's inherent stochasticity. The 60% residual is comparable to Experiment 1 despite the richer valuation environment.

**MDE:** 6.35% of grand mean (0.029 revenue units). The small sample (n=192) limits power, particularly for detecting moderate effects.

#### Effects Table

| Rank | Factor | Coefficient | t-stat | p-value | Effect (% of mean) | Survives Holm? | Practical Significance |
|------|--------|-------------|------|---------|---------------------|----------------|------------------------|
| 1 | n_bidders | +0.055 | 6.5 | 6.8e-10 | 24.0% | Yes | **High** |
| 2 | state_info | -0.049 | 5.9 | 2.3e-08 | 21.6% | Yes | **High** |
| 3 | auction_type | -0.027 | 3.2 | 1.8e-03 | 11.7% | No (Holm p=0.098) | Moderate but fragile |
| 4 | eta_linear | -0.010 | 1.9 | 0.056 | 4.3% | No | None (not sig.) |
| 5 | eta_quadratic | -0.001 | 0.2 | 0.824 | 0.6% | No | None |

#### Main Effects

**n_bidders (+0.055, 24.0%):** Same direction as Experiment 1. Moving from 2 to 4 bidders increases converged revenue by about 24% of the grand mean (from ~0.404 to ~0.514). Proportionally larger than in Exp 1.

**state_info (-0.049, 21.6%):** When agents receive full information about opponents' past bids, converged revenue *decreases* by 21.6% of the grand mean. This is consistent with the theoretical prediction that transparency can facilitate tacit collusion among adaptive agents.

**auction_type (-0.027, 11.7%):** SPA produces modestly higher converged revenue than FPA. The magnitude (11.7% of grand mean) would be practically relevant, but does not survive Holm-Bonferroni correction (adjusted p = 0.098). The multiple-testing burden across 15 coefficients is too heavy for this moderate signal.

**eta_linear (-0.010, 4.3%):** The affiliation parameter does not significantly affect aggregate revenue (p = 0.056) despite being the defining feature of the valuation model. However, eta *does* significantly affect bidding strategy (signal_slope_ratio |t| = 8.0) and winner's curse frequency, confirming that affiliation affects behavior without changing aggregate outcomes under Q-learning.

#### Top Interactions

| Interaction | Coefficient | t-stat | p-value |
|-------------|-------------|------|---------|
| auction_type x n_bidders | -0.029 | 3.4 | 9.0e-04 |
| n_bidders x state_info | -0.026 | 3.1 | 2.2e-03 |

**auction_type x n_bidders conditional means:**

|  | 2 bidders | 4 bidders |
|--|-----------|-----------|
| SPA | 0.402 | 0.569 |
| FPA | 0.406 | 0.459 |

*What it means:* The SPA advantage emerges only in competitive markets. With 2 bidders, FPA and SPA generate nearly identical revenue (0.406 vs 0.402). With 4 bidders, SPA substantially outperforms FPA (0.569 vs 0.459, a 24% gap). Direction consistent with the linkage principle from auction theory.

**n_bidders x state_info conditional means:**

|  | No info | Full info |
|--|---------|-----------|
| 2 bidders | 0.427 | 0.381 |
| 4 bidders | 0.590 | 0.438 |

*What it means:* Information feedback is most damaging in competitive markets. With 2 bidders, full info reduces revenue modestly (0.381 vs 0.427, -10.8%). With 4 bidders, the damage is severe (0.438 vs 0.590, -25.8%). Competition (which should raise bids) and information (which lowers them) interact, and information dominates.

**Lifetime revenue (avg_rev_all):** Different hierarchy. State_info (|t| = 4.5) overtakes auction_type (|t| = 1.4, p = 0.152, not significant). FPA shows a learning-phase premium (mean 0.114 vs SPA's 0.035) that vanishes at convergence.

**Specialized metrics:** Winners' curse frequency shows strong auction_type x n_bidders interaction (|t| = 8.3). Signal_slope_ratio reveals eta_linear as dominant (|t| = 8.0), confirming that affiliation affects bidding strategies even when it does not affect aggregate revenue.

**Robustness:** Auction format main effect for revenue does not survive multiple-testing correction. Pred-R² = 0.299 (gap = 0.105). LGBM R² = 0.336 (OLS outperforms; small sample).

---

### Experiment 3a: LinUCB Contextual Bandits

**Design:** 3 x 2⁷ mixed-level, 9 factors, 384 cells, 2 reps/cell (768 runs)

**Primary revenue response (avg_rev_last_1000):** R² = 0.694, Adj-R² = 0.676. Grand mean = 0.459.

**What R² = 0.69 means:** A substantial jump from Q-learning's 0.40-0.42. LinUCB produces more predictable outcomes than Q-learning, so the experimental factors explain a larger share of the variation.

**MDE:** 2.89% of grand mean (0.013 revenue units). Good power with n=768.

#### Effects Table

| Rank | Factor | Coefficient | t-stat | p-value | Effect (% of mean) | Survives Holm? | Practical Significance |
|------|--------|-------------|------|---------|---------------------|----------------|------------------------|
| 1 | n_bidders | +0.131 | 33.9 | 1.7e-151 | 57.0% | Yes | **Very high** |
| 2 | auction_type | -0.048 | 12.5 | 1.5e-32 | 21.0% | Yes | **High** |
| 3 | eta_linear | -0.013 | 5.7 | 1.7e-08 | 5.9% | Yes | Moderate |
| 4 | exploration_intensity | -0.017 | 4.4 | 1.4e-05 | 7.3% | -- | Moderate |
| 5 | context_richness | -0.013 | 3.5 | 5.6e-04 | 5.8% | -- | Small |

#### Main Effects

**n_bidders (+0.131, 57.0%):** The competitive-pressure effect is dramatically amplified under LinUCB. Moving from 2 to 4 bidders increases revenue by 57% of the grand mean, nearly triple the effect under Q-learning (21-24%). Direction consistent with standard auction theory.

**auction_type (-0.048, 21.0%):** The strongest auction format effect among all unconstrained experiments. SPA outperforms FPA by about 21% of the grand mean (SPA mean = 0.507, FPA mean = 0.411). Direction consistent with the linkage principle for affiliated values. Survives all robustness checks including Holm-Bonferroni (adjusted p = 2.7e-06).

**eta_linear (-0.013, 5.9%):** Unlike in Experiment 2, the affiliation parameter is now significant. Higher affiliation modestly reduces revenue. The contrast with Exp 2's null result suggests that Q-learning's coarser approach masks affiliation effects that LinUCB detects.

**exploration_intensity (-0.017, 7.3%):** Higher exploration reduces revenue by 7.3% of grand mean.

#### Top Interactions

| Interaction | Coefficient | t-stat | p-value |
|-------------|-------------|------|---------|
| n_bidders x reserve_price | -0.040 | 10.4 | 1.5e-23 |
| auction_type x n_bidders | -0.024 | 6.3 | 3.9e-10 |
| reserve_price x context_richness | -0.022 | 5.8 | 1.1e-08 |

**n_bidders x reserve_price conditional means:**

|  | No reserve | With reserve |
|--|------------|--------------|
| 2 bidders | 0.282 | 0.373 |
| 4 bidders | 0.624 | 0.555 |

*What it means:* Reserve prices help thin markets and hurt thick markets. With 2 bidders, adding a reserve increases revenue substantially (0.373 vs 0.282, +32%). With 4 bidders, reserves reduce revenue (0.555 vs 0.624, -11%). A strong finding (|t| = 10.4) with direct policy implications: reserves are beneficial when competition is limited but counterproductive when competition is already strong.

**auction_type x n_bidders conditional means:**

|  | 2 bidders | 4 bidders |
|--|-----------|-----------|
| SPA | 0.352 | 0.662 |
| FPA | 0.304 | 0.517 |

*What it means:* SPA's advantage is larger in thick markets. With 2 bidders, SPA outperforms FPA by 0.048 (15.8%). With 4 bidders, the gap widens to 0.145 (28.0%). Same direction as the auction_type x n_bidders interaction in Experiments 1 and 2.

**reserve_price x context_richness conditional means:**

|  | Low context | High context |
|--|-------------|--------------|
| No reserve | 0.444 | 0.462 |
| With reserve | 0.500 | 0.429 |

*What it means:* Reserve price and context richness interact negatively. With low context, reserves increase revenue (0.500 vs 0.444). With high context, reserves decrease revenue (0.429 vs 0.462).

**Other responses:** no_sale_rate (R² = 0.806), winner_entropy (R² = 0.809) show very high explanatory power. Reserve_price and n_bidders jointly dominate no-sale outcomes.

**Robustness:** Revenue auction_type effect survives Holm-Bonferroni (Holm p = 2.7e-06). Pred-R² = 0.655 (gap = 0.039, small). LGBM R² = 0.714 (gap = +0.019, negligible nonlinearity). Linear model is adequate.

---

### Experiment 3b: Thompson Sampling Contextual Bandits

**Design:** 3 x 2⁵ mixed-level, 7 factors, 96 cells, 2 reps/cell (192 runs)

**Primary revenue response (avg_rev_last_1000):** R² = 0.609, Adj-R² = 0.544. Grand mean = 0.509.

**What R² = 0.61 means:** Lower than LinUCB's 0.69, reflecting Thompson Sampling's greater inherent stochasticity. The gap between Exp 3a (0.69) and 3b (0.61) quantifies the additional unpredictability of Thompson Sampling relative to LinUCB.

**MDE:** 4.09% of grand mean (0.021 revenue units). Moderate power with n=192.

#### Effects Table

| Rank | Factor | Coefficient | t-stat | p-value | Effect (% of mean) | Survives Holm? | Practical Significance |
|------|--------|-------------|------|---------|---------------------|----------------|------------------------|
| 1 | n_bidders | +0.053 | 8.7 | 3.6e-15 | 20.6% | Yes | **High** |
| 2 | reserve_price | -0.048 | 7.9 | 2.9e-13 | 18.9% | Yes | **High** |
| 3 | auction_type | -0.023 | 3.8 | 1.8e-04 | 9.1% | No | Moderate but fragile |
| 4 | exploration_intensity | -0.020 | 3.4 | 9.0e-04 | 8.0% | No | Moderate but fragile |
| 5 | eta_linear | -0.012 | 3.2 | 1.8e-03 | 4.6% | No | Small |

#### Main Effects

**n_bidders (+0.053, 20.6%):** Same direction as all other experiments, but much smaller than LinUCB's 57%.

**reserve_price (-0.048, 18.9%):** This is the **sign reversal** compared to Experiment 1 (where reserve_price was positive, +6.1%). Under Thompson Sampling, higher reserve prices *reduce* revenue. This contrasts with both Exp 1 and Exp 3a, where reserves had positive or context-dependent effects.

**auction_type (-0.023, 9.1%):** SPA is favored, consistent with the bandit-family pattern. The effect is less than half of LinUCB's (9.1% vs 21.0%).

#### Top Interactions

| Interaction | Coefficient | t-stat | p-value |
|-------------|-------------|------|---------|
| n_bidders x context_richness | +0.028 | 4.7 | 5.3e-06 |
| auction_type x n_bidders | -0.025 | 4.1 | 7.4e-05 |

**n_bidders x context_richness conditional means:**

|  | Low context | High context |
|--|-------------|--------------|
| 2 bidders | 0.502 | 0.411 |
| 4 bidders | 0.550 | 0.572 |

*What it means:* Context richness helps in thick markets but hurts in thin markets. With 2 bidders, more context dimensions reduce revenue (0.411 vs 0.502, -18%). With 4 bidders, context slightly helps (0.572 vs 0.550, +4%).

**auction_type x n_bidders conditional means:**

|  | 2 bidders | 4 bidders |
|--|-----------|-----------|
| SPA | 0.455 | 0.609 |
| FPA | 0.458 | 0.514 |

*What it means:* Same pattern as LinUCB and Exp 1: the SPA advantage grows with competition. With 2 bidders, FPA and SPA are nearly identical. With 4 bidders, SPA outperforms FPA by 18.5%. This recurring interaction pattern across all unconstrained experiments is one of the most robust findings in the study.

**Robustness:** Revenue effects do not survive Holm-Bonferroni (small sample). Pred-R² = 0.464 (gap = 0.145, largest gap across all experiments). LGBM R² = 0.557 (OLS outperforms; small sample).

**Limitation:** Pred-R² gap of 0.145 is the largest across all experiments, suggesting some overfitting risk with 7 factors and only 192 observations. Core findings (n_bidders dominance, reserve price effect) are confirmed by the more powerful Exp 3a.

---

### Experiment 4a: Multiplicative Dual Pacing

**Design:** 2⁶ full factorial, 6 factors, 64 cells, 8 reps/cell (512 runs)

**Primary revenue response (mean_platform_revenue):** R² = 0.889, Adj-R² = 0.885. Grand mean = $4,257.

**What R² = 0.89 means:** The six factors and their interactions explain 89% of revenue variation, the highest across all experiments. Dual pacing produces the most predictable outcomes of any algorithm family tested.

**MDE:** 3.35% of grand mean ($143 in revenue units). Uniform across all factors due to balanced 2^6 design.

#### Effects Table

| Rank | Factor | Coefficient | t-stat | p-value | Effect (% of mean) | Survives Holm? | Practical Significance |
|------|--------|-------------|------|---------|---------------------|----------------|------------------------|
| 1 | budget_multiplier | +$2,007 | 39.4 | 3.8e-154 | 94.3% | Yes | **Very high** |
| 2 | objective | -$1,428 | 28.1 | 4.8e-104 | 67.1% | Yes | **Very high** |
| 3 | n_bidders | +$1,158 | 22.8 | 8.5e-79 | 54.4% | Yes | **Very high** |
| 4 | sigma | +$357 | 7.0 | 7.8e-12 | 16.8% | Yes | Moderate |
| 5 | auction_type | +$184 | 3.6 | 3.3e-04 | 8.6% | -- | Moderate |

#### Main Effects

**budget_multiplier (+$2,007, 94.3%):** The largest single-factor effect in any experiment. Moving from tight budgets (0.5x) to loose budgets (2.0x) nearly doubles mean revenue.

**objective (-$1,428, 67.1%):** Value-maximizing bidders produce far less revenue than ROI-maximizing bidders. The gap is 67% of grand mean.

**n_bidders (+$1,158, 54.4%):** Same direction as unconstrained experiments, but now the third-ranked factor (behind budget and objective) rather than the dominant one. Budget constraints reshape the entire hierarchy.

**auction_type (+$184, 8.6%):** FPA produces higher revenue than SPA under dual pacing, the **reversal** of the pattern in Experiments 2-3b.

#### Top Interactions

| Interaction | Coefficient | t-stat | p-value |
|-------------|-------------|------|---------|
| objective x budget_multiplier | -$1,387 | 27.3 | 2.7e-100 |
| objective x n_bidders | -$630 | 12.4 | 8.4e-31 |
| n_bidders x budget_multiplier | +$449 | 8.8 | 2.0e-17 |

**objective x budget_multiplier conditional means:**

|  | Tight budget | Loose budget |
|--|--------------|--------------|
| ROI-maximizer | $2,290 | $9,079 |
| Value-maximizer | $2,209 | $3,448 |

*What it means:* The largest interaction in the entire study (|t| = 27.3). Under tight budgets, the two objectives produce similar revenue ($2,290 vs $2,209, +3.7%). Under loose budgets, the divergence is extreme ($9,079 vs $3,448, +163%). Budget constraints equalize the two objective types; loose budgets let them diverge.

**objective x n_bidders conditional means:**

|  | 2 bidders | 4 bidders |
|--|-----------|-----------|
| ROI-maximizer | $3,897 | $7,473 |
| Value-maximizer | $2,300 | $3,357 |

*What it means:* Competition amplifies the objective-type difference. With 4 bidders, ROI-maximizers generate $7,473 vs value-maximizers' $3,357, more than double.

**n_bidders x budget_multiplier conditional means:**

|  | Tight budget | Loose budget |
|--|--------------|--------------|
| 2 bidders | $1,541 | $4,656 |
| 4 bidders | $2,959 | $7,870 |

*What it means:* Competition and budget looseness are complements. The revenue boost from adding bidders is $3,214 under loose budgets but only $1,418 under tight budgets. When budgets are tight, additional competition is partially neutralized because all bidders are budget-constrained.

**Price of Anarchy:** R² = 0.640. Auction format has no significant effect on PoA (|t| = 0.7, p = 0.493).

**Cross-episode drift:** R² = 0.125 (near zero). No evidence of progressive bid suppression across campaign cycles.

**Robustness:** Revenue effects overwhelmingly robust. Pred-R² = 0.879 (gap = 0.010, smallest). LGBM R² = 0.607 (OLS dramatically outperforms; linear model is strongly adequate).

---

### Experiment 4b: PI Controller Pacing

**Design:** 2⁶ full factorial, 6 factors, 64 cells, 8 reps/cell (512 runs)

**Primary revenue response (mean_platform_revenue):** R² = 0.768, Adj-R² = 0.758. Grand mean = $3,651.

**What R² = 0.77 means:** Lower than Exp 4a's 0.89, but still high. The PI controller produces slightly more variable outcomes than the simpler multiplicative update, but remains far more deterministic than Q-learning or bandits.

**MDE:** 3.19% of grand mean ($117 in revenue units). Uniform across all factors.

#### Effects Table

| Rank | Factor | Coefficient | t-stat | p-value | Effect (% of mean) | Survives Holm? | Practical Significance |
|------|--------|-------------|------|---------|---------------------|----------------|------------------------|
| 1 | budget_multiplier | +$1,316 | 31.7 | 1.2e-120 | 72.1% | Yes | **Very high** |
| 2 | n_bidders | +$668 | 16.1 | 5.4e-47 | 36.6% | Yes | **Very high** |
| 3 | auction_type | +$458 | 11.0 | 2.0e-25 | 25.1% | Yes | **High** |
| 4 | sigma | +$361 | 8.7 | 5.9e-17 | 19.8% | Yes | Moderate |
| 5 | aggressiveness | +$15 | 0.4 | 0.713 | 0.8% | No | None |

#### Main Effects

**budget_multiplier (+$1,316, 72.1%):** Same direction as Exp 4a, but 34% smaller in magnitude.

**n_bidders (+$668, 36.6%):** Competition rises to second place (from fourth in Exp 4a). Without the objective factor, n_bidders reclaims its natural importance.

**auction_type (+$458, 25.1%):** The FPA advantage is **much larger** under PI pacing than under dual pacing (25.1% vs 8.6%, |t| = 11.0 vs 3.6).

**sigma (+$361, 19.8%):** Higher value dispersion increases revenue by 19.8% of grand mean.

**aggressiveness (+$15, 0.8%):** The PI controller's aggressiveness parameter has no significant effect on revenue (p = 0.713).

#### Top Interactions

| Interaction | Coefficient | t-stat | p-value |
|-------------|-------------|------|---------|
| auction_type x budget_multiplier | +$394 | 9.5 | 1.1e-19 |
| auction_type x sigma | +$234 | 5.6 | 3.2e-08 |
| budget_multiplier x sigma | +$175 | 4.2 | 3.0e-05 |

**auction_type x budget_multiplier conditional means:**

|  | Tight budget | Loose budget |
|--|--------------|--------------|
| SPA | $2,270 | $4,115 |
| FPA | $2,400 | $5,819 |

*What it means:* FPA's advantage explodes under loose budgets. With tight budgets, FPA and SPA are similar ($2,400 vs $2,270, +5.7%). With loose budgets, FPA generates 41.4% more revenue than SPA ($5,819 vs $4,115). Budget constraints equalize the two formats; loose budgets let them diverge.

**auction_type x sigma conditional means:**

|  | Low sigma | High sigma |
|--|-----------|------------|
| SPA | $3,066 | $3,320 |
| FPA | $3,515 | $4,704 |

*What it means:* FPA benefits more from value dispersion than SPA. When values are concentrated (low sigma), the FPA-SPA gap is modest ($3,515 vs $3,066, +14.6%). When values are dispersed (high sigma), FPA pulls further ahead ($4,704 vs $3,320, +41.7%).

**Price of Anarchy:** R² = 0.391. Unlike Exp 4a, auction format is now the **dominant** PoA factor (|t| = 10.7). FPA produces higher PoA (more welfare loss) than SPA under PI pacing.

**Robustness:** Revenue effects robust. Pred-R² = 0.746 (gap = 0.021). LGBM R² = 0.709 (OLS outperforms).

---

## Cross-Experiment Patterns

### Auction Format Effect Across Algorithm Families

| Exp | Algorithm | Coefficient | t-stat | Effect (% of mean) | Direction | Survives Holm? | Practical Significance |
|-----|-----------|-------------|------|---------------------|-----------|----------------|------------------------|
| 1 | Q-learning (constant) | +0.003 | 0.6 | 0.8% | Negligible | No | None |
| 2 | Q-learning (affiliated) | -0.027 | 3.2 | 11.7% | SPA favored | No (p=0.098) | Moderate but fragile |
| 3a | LinUCB | -0.048 | 12.5 | 21.0% | SPA favored | **Yes** | **High** |
| 3b | Thompson Sampling | -0.023 | 3.8 | 9.1% | SPA favored | No | Moderate but fragile |
| 4a | Dual Pacing | +$184 | 3.6 | 8.6% | FPA favored | -- | Moderate |
| 4b | PI Pacing | +$458 | 11.0 | 25.1% | FPA favored | **Yes** | **High** |

The pattern is clear: unconstrained algorithms favor SPA (or show no preference), while budget-pacing algorithms favor FPA. The two strongest effects (LinUCB at 21.0% and PI Pacing at 25.1%) both survive Holm-Bonferroni and represent practically significant revenue differences for an auction designer.

### Number of Bidders Effect Across Experiments

| Exp | Algorithm | Coefficient | t-stat | Effect (% of mean) | Rank |
|-----|-----------|-------------|------|---------------------|------|
| 1 | Q-learning (constant) | +0.087 | 17.8 | 21.3% | 1st |
| 2 | Q-learning (affiliated) | +0.055 | 6.5 | 24.0% | 1st |
| 3a | LinUCB | +0.131 | 33.9 | 57.0% | 1st |
| 3b | Thompson Sampling | +0.053 | 8.7 | 20.6% | 1st |
| 4a | Dual Pacing | +$1,158 | 22.8 | 54.4% | 4th |
| 4b | PI Pacing | +$668 | 16.1 | 36.6% | 2nd |

N_bidders ranks first in every unconstrained experiment. Its effect ranges from 21-57% of grand mean. Under budget constraints, it drops in rank but remains a top factor. This is the single most consistent and practically significant finding.

### Consistent Findings

1. **Structural parameters dominate auction format.** In every experiment, market structure variables (n_bidders, budget_multiplier) have effect sizes 1.5x–28x larger than auction format. Auction format is never the top-ranked factor.

2. **N_bidders is the universal dominant factor in unconstrained settings.** Across Experiments 1-3b, n_bidders ranks first with |t| values of 17.8, 6.5, 33.9, and 8.7. This is the single most robust finding.

3. **Budget constraints reshape the entire factor hierarchy.** In Experiments 4a-4b, budget_multiplier takes over as the dominant factor (|t| = 39.4 and 31.7), with n_bidders dropping to fourth and second place respectively.

4. **Auction format effects are algorithm-dependent.** The direction reverses across algorithm families (see table above).

5. **Model fit increases with algorithm sophistication.** R² progresses from ~0.4 (Q-learning) to ~0.6-0.7 (bandits) to ~0.8-0.9 (pacing), reflecting that more deterministic algorithms produce more predictable outcomes.

6. **Interaction effects matter.** In every experiment, at least one two-way interaction ranks in the top 5 effects. The objective x budget_multiplier interaction in Exp 4a (|t| = 27.3) is the largest interaction effect in the entire study.

7. **auction_type x n_bidders is a recurring interaction.** It appears in the top 3 interactions for Experiments 1, 2, 3a, and 3b, always with a negative coefficient. The pattern is consistent: auction format differences become larger with more bidders. In thin markets, FPA and SPA converge; in thick markets, they diverge.

### Inconsistent or Surprising Findings

1. **Reserve price sign reversal.** Reserve price has a positive effect on revenue in Exp 1 (+6.1%) but a negative effect in Exp 3b (-18.9%). The direction depends on both the algorithm family and valuation structure.

2. **Eta (affiliation) null result.** Despite being the defining parameter of the affiliated values model, eta has no significant effect on primary revenue in Exp 2 (p = 0.056). It becomes significant in Exp 3a (|t| = 5.7) and 3b (|t| = 3.2). The affiliation effect on revenue appears only with more sophisticated algorithms.

3. **PoA sensitivity differs between pacing algorithms.** In Exp 4a, auction format has no effect on PoA (p = 0.493). In Exp 4b, auction format is the dominant PoA factor (|t| = 10.7). This means efficiency implications of format choice depend on pacing technology, not just revenue implications.

4. **Cross-episode drift null.** Both Exp 4a and 4b show near-zero R² for cross-episode drift (0.125 and 0.054). Pacing algorithms do not progressively suppress bids over campaign cycles. A meaningful null result.

5. **Factor hierarchy differences between pacing algorithms.** Exp 4a's hierarchy (budget > objective > objective:budget > n_bidders > auction_type) differs substantially from Exp 4b's (budget > n_bidders > auction_type > auction_type:budget > sigma). Even shared factors (n_bidders, auction_type) change rank.

### Statistical vs Practical Significance Summary

**Practically significant AND statistically robust (actionable for auction designers):**
- n_bidders everywhere (21-57% of grand mean, survives Holm in all experiments)
- budget_multiplier in Exp 4a/4b (72-94% of grand mean, survives Holm)
- auction_type in Exp 3a (21% of grand mean, Holm p = 2.7e-06)
- auction_type in Exp 4b (25% of grand mean, survives Holm)
- objective x budget_multiplier in Exp 4a (largest interaction, Holm p = 8.6e-98)

**Statistically significant but practically small:**
- auction_type in Exp 1 (0.8% of grand mean, not even statistically significant)
- eta across all experiments (4-6% of grand mean, borderline significance)
- aggressiveness in Exp 4b (0.8% of grand mean, not significant)

**Practically large but statistically fragile (interpret with caution):**
- auction_type in Exp 2 (11.7% of grand mean, does not survive Holm)
- auction_type in Exp 3b (9.1% of grand mean, does not survive Holm in small sample)
- state_info in Exp 2 (21.6% of grand mean, survives Holm; robust but single experiment)

---

## Robustness Assessment

### Which findings survive multiple-testing corrections?

**Strong (survive Holm-Bonferroni for primary revenue):**
- n_bidders dominance: All experiments
- Budget_multiplier dominance in Exp 4a/4b: Yes
- Auction format in Exp 3a: Yes (Holm p = 2.7e-06)
- Auction format in Exp 4b: Yes
- Objective x budget_multiplier in Exp 4a: Yes (Holm p = 8.6e-98)

**Weak (do not survive Holm-Bonferroni for revenue):**
- Auction format in Exp 2: Does not survive (Holm p = 0.098)
- Revenue effects in Exp 3b: Do not survive (small sample)
- Reserve_price x info_feedback in Exp 1: Does not survive for revenue

### HC3 Robust Standard Errors

HC3 results are consistent with OLS across all experiments. The number of significant effects under HC3 matches or nearly matches OLS in every case, indicating that heteroscedasticity does not materially affect inference.

### LASSO Variable Selection

LASSO consistently selects more variables than OLS significance testing, indicating that many small but real effects exist beyond the significant ones. This is expected in factorial designs where many effects are estimable but small.

---

## Model Adequacy

| Experiment | R² | Pred-R² | Gap | LGBM R² | Linear Adequate? |
|------------|-----|---------|-----|---------|-----------------|
| 1 | 0.423 | 0.354 | 0.069 | 0.455 | Yes (mild nonlinearity) |
| 2 | 0.404 | 0.299 | 0.105 | 0.336 | Yes (OLS > LGBM) |
| 3a | 0.694 | 0.655 | 0.039 | 0.714 | Yes (negligible gap) |
| 3b | 0.609 | 0.464 | 0.145 | 0.557 | Marginal (large Pred-R² gap) |
| 4a | 0.889 | 0.879 | 0.010 | 0.607 | Yes (OLS >> LGBM) |
| 4b | 0.768 | 0.746 | 0.021 | 0.709 | Yes (OLS > LGBM) |

**Why R² increases across algorithm families:** The R² progression (Q-learning ~0.4, bandits ~0.6-0.7, pacing ~0.8-0.9) tracks the algorithms' inherent stochasticity, from most random to most deterministic.

The linear factorial model is adequate for all experiments. LGBM underperforms OLS in 4/6 experiments (the small-sample ones), confirming that the linear specification captures the essential structure. Experiment 3b's Pred-R² gap of 0.145 is the largest and warrants a caveat about overfitting risk, but its core findings (n_bidders dominance, reserve price effect) are confirmed by the more powerful Exp 3a.

---

## Sensitivity Analysis (Sobol' Variance Decomposition)

Seven global sensitivity methods were applied to each experiment's response surfaces: analytical Sobol' indices, Random Forest importance, SHAP, Morris screening, Monte Carlo Sobol', FAST, and Borgonovo delta indices. For balanced 2^k factorial designs, the analytical Sobol' decomposition is exact (the variance partition follows directly from the orthogonal design). The surrogate-based methods (RF, SHAP, neural network, Kriging) provide independent cross-validation. Cross-method concordance was assessed via pairwise Spearman rank correlations across all methods, averaged into a single agreement score per outcome.

### ANOVA vs. Sobol' Comparison (Primary Revenue Response)

| Exp | ANOVA R² | ANOVA Top Factor | \|t\| | Sobol' Top Factor | ST | S1 | ST−S1 | Method ρ |
|-----|----------|------------------|-------|-------------------|------|------|-------|----------|
| 1 | 0.42 | n_bidders | 17.8 | n_bidders | 0.24 | 0.19 | 0.05 | 0.87 |
| 2 | 0.40 | n_bidders | 6.5 | n_bidders | 0.22 | 0.14 | 0.08 | 0.49 |
| 3a | 0.69 | n_bidders | 33.9 | n_bidders | 0.55 | 0.48 | 0.07 | 0.58 |
| 3b | 0.61 | n_bidders | 8.7 | n_bidders | 0.31 | 0.18 | 0.13 | −0.05 |
| 4a | 0.89 | budget_multiplier | 39.4 | budget_multiplier | 0.54 | 0.35 | 0.19 | 0.95 |
| 4b | 0.77 | budget_multiplier | 31.7 | budget_multiplier | 0.53 | 0.48 | 0.05 | 0.91 |

*ST = total-order Sobol' index (includes all interactions). S1 = first-order index (main effect only). ST−S1 = variance attributable to interactions. Method ρ = mean pairwise Spearman rank correlation across 7 sensitivity methods.*

### Key Findings

1. **Dominant factor confirmed in all 6 experiments.** The Sobol' analysis identifies the same top factor as the ANOVA for every experiment's primary revenue response: n_bidders in Experiments 1–3b, budget_multiplier in Experiments 4a–4b.

2. **Method concordance is highest for the pacing experiments and Experiment 1.** The budget-pacing experiments show near-perfect cross-method agreement (ρ = 0.91–0.95), and Experiment 1 is close behind (ρ = 0.87). Concordance is weaker for the small-sample experiments: ρ = 0.49 for Exp 2 and −0.05 for Exp 3b. Experiment 3a falls in between (ρ = 0.58). Both ANOVA R² and sample size appear to contribute to method agreement.

3. **Negligible factors confirmed.** Factors with ST < 0.01 in the Sobol' analysis match those with insignificant |t|-statistics in the ANOVA. Examples: alpha, init, and decay_type in Experiment 1; lam and memory_decay in Experiment 3a; auction_type and reserve_price for Experiment 4a's revenue response.

4. **Interaction importance confirmed independently.** The ST−S1 gap quantifies each factor's contribution through interactions. The largest gaps (budget_multiplier in Exp 4a at 0.19; n_bidders in Exp 3b at 0.13) correspond to experiments where ANOVA interaction terms are statistically significant and substantively large.

5. **Experiment 3b flagged as weakest.** The negative cross-method Spearman correlation (ρ = −0.05) means the seven sensitivity methods produce essentially random factor rankings for Experiment 3b's revenue response. This is consistent with Exp 3b's known limitations: smallest effective sample (192 runs), largest Pred-R² gap (0.145), and several effects that do not survive multiple-testing correction. Fine-grained factor rankings for Exp 3b should be interpreted with caution; only the dominant role of n_bidders is reliably established.

**Bottom line:** The global variance-based sensitivity analysis independently validates the linear model's factor hierarchy across all six experiments. The dominant factor, negligible factors, and interaction structure identified by ANOVA are confirmed by Sobol' decomposition. No contradictions were found for any experiment's top-ranked factor.

---

## Value of the Laboratory Approach

### What the multi-technology comparison reveals

The six experiments test six bidding technologies: Q-learning with constant valuations (Exp 1), Q-learning with affiliated values (Exp 2), LinUCB bandits (Exp 3a), Thompson Sampling bandits (Exp 3b), dual pacing (Exp 4a), and PI controller pacing (Exp 4b). Several findings emerge only from comparing across these technologies.

**Auction format direction reversal.** SPA favors revenue under LinUCB bandits (Exp 3a: −21.0%) while FPA favors revenue under PI pacing (Exp 4b: +25.1%). Both effects survive Holm-Bonferroni correction and represent the two largest auction format effects in the study. A study examining any single bidding technology would report one direction as "the" effect of auction format on revenue. The reversal itself is the finding.

**Factor hierarchy shift.** n_bidders is the dominant factor in Experiments 1–3b; budget_multiplier takes over in Experiments 4a–4b. The structural determinants of auction outcomes depend on which algorithm bidders use. Competition intensity matters most when bidders learn incrementally; budget constraints matter most when bidders pace expenditure over campaigns.

**Reserve price sign reversal.** Reserve prices have a positive effect on revenue in Exp 1 (+6.1%) but a negative effect in Exp 3b (−18.9%). The policy implication for an auction designer flips depending on the bidding technology in the market.

**Affiliation parameter from null to significant.** The eta parameter has no significant effect on revenue under Q-learning (Exp 2, p = 0.056) but is highly significant under LinUCB (Exp 3a, p = 1.7e-08). Whether the valuation structure matters for revenue depends on the learning algorithm.

### What the factorial design reveals

A one-factor-at-a-time (OFAT) design would vary each factor while holding all others fixed. The factorial design's orthogonality estimates all main effects and two-way interactions simultaneously, revealing conditional effects that OFAT would miss.

**objective × budget_multiplier** (Exp 4a, |t| = 27.3) is the largest interaction effect in the entire study. Under tight budgets, the two objective types produce similar revenue (+3.7% difference); under loose budgets, they diverge by +163%. OFAT would detect neither the interaction nor its magnitude.

**n_bidders × reserve_price** (Exp 3a, |t| = 10.4) shows that reserve prices help thin markets (+32% with 2 bidders) and hurt thick markets (−11% with 4 bidders). An OFAT study would report one direction depending on which market size was held fixed.

**auction_type × n_bidders** appears in the top three interactions for 4 of 6 experiments (Exp 1, 2, 3a, 3b), always with the same sign: auction format differences widen as competition increases. The format comparison is conditional on the competition level.

An OFAT design would have required many more simulation runs to detect these interactions, if it detected them at all, and would have missed the conditional nature of most policy-relevant effects.

### What the sensitivity analysis extension reveals

A global variance-based sensitivity analysis (seven methods including analytical Sobol decomposition, SHAP, Random Forest permutation importance, and Morris screening) was run on all six experiments to independently validate the linear model's factor hierarchy.

**Confirmation.** The top factor identified by ANOVA matches the Sobol-dominant factor in all six experiments. Factors that the linear model finds negligible (e.g., aggressiveness in Exp 4b, decay_type in Exp 1) are confirmed negligible by all seven methods. The interaction structure also aligns: experiments where ANOVA detects large interactions show correspondingly large gaps between first-order and total-order Sobol indices.

**Deviation from expectation.** Cross-method concordance varies dramatically. For the pacing experiments (4a, 4b), Spearman rank correlations among the seven methods reach ρ = 0.91–0.95, indicating near-unanimous agreement on factor rankings. For the bandit experiments, agreement drops to ρ = 0.40–0.60 (Exp 3a) and collapses to ρ = −0.05 (Exp 3b), meaning different methods produce essentially unrelated rankings. If ANOVA identifies clear significant effects, one would expect different sensitivity methods to agree on factor rankings. They often do not, particularly for experiments with smaller samples or lower R².

**Implication.** Top-factor identification is robust across methods, but fine-grained factor rankings are method-sensitive. The choice of sensitivity method matters more than expected, which is itself a finding worth reporting. Results sections should cite the sensitivity analysis as corroborating top-factor conclusions without overstating the precision of lower-ranked factor orderings.

---

## Limitations and Caveats

1. **Stylized environment.** All experiments use identical-item auctions with symmetric bidders (within valuation classes). Multi-item, combinatorial, or asymmetric settings may produce different patterns.

2. **Specific algorithm families.** Three algorithm families (Q-learning, contextual bandits, multiplicative/PI pacing) represent common approaches but do not exhaust the space. Deep RL, mean-field game approaches, or hybrid strategies are untested.

3. **Small samples in Exp 2 and 3b.** With only 192 observations and relatively many parameters, these experiments have lower statistical power and larger overfitting risk. Their findings should be interpreted alongside the larger Exp 1 and 3a.

4. **No multi-agent learning dynamics.** Experiments use independent learners. Strategic interaction between adaptive agents (where one agent's learning affects another's environment) is not modeled.

5. **Fixed episode structure.** Pacing experiments use fixed campaign lengths with budget regeneration. Real-world ad exchanges have variable campaign durations, budget adjustments, and competitive entry/exit.

6. **Lack-of-fit.** All experiments show significant lack-of-fit (p < 0.001 in most cases), meaning the linear model with 2-way interactions does not capture all systematic variation. However, the practical significance is limited given the high Pred-R² values in experiments with sufficient data.

---

## Discrepancy Log

| # | Location | Claim | Data | Severity | Status |
|---|----------|-------|------|----------|--------|
| 1 | res4b.tex, res4a.tex | Budget multiplier t-stat = 31.2 for Exp4b | Actual t-stat = 31.7 (macro correct at 31.7) | Minor | **Fixed** (replaced hardcoded values with macro) |

No other discrepancies found. ~80 factual claims cross-checked against estimation_results.json across all 6 experiments and the discussion section. All factor rankings, effect directions, significance levels, R² values, and cross-experiment narrative claims are accurate.

---

## Auction Format Effect Across All Responses

This section shows how auction format (FPA vs SPA) affects *every* measured outcome, not just revenue. Coding: `auction_type_coded = +1` is FPA, `-1` is SPA. The coefficient is the half-effect; the full FPA-SPA difference is 2x the coefficient.

### Experiment 1: Q-learning, Constant Valuations (6 responses)

| Response | Coefficient | t-stat | p-value | Holm sig? | Direction |
|----------|-------------|-------|---------|-----------|-----------|
| Converged revenue | +0.003 | 0.6 | 0.529 | No | -- |
| Lifetime revenue | +0.023 | 6.8 | 2.2e-11 | **Yes** | FPA higher |
| Convergence time | -8,243 | 10.1 | 6.5e-23 | **Yes** | FPA faster |
| No-sale rate | +0.001 | 1.0 | 0.301 | No | -- |
| Price volatility | +0.003 | 2.3 | 0.019 | No | -- |
| Winner entropy | +0.012 | 1.5 | 0.139 | No | -- |

**Interpretation:** Auction format barely matters for converged revenue (the null result consistent with revenue equivalence). But FPA converges ~16,500 episodes faster and earns more *during the learning phase*, creating a meaningful transient advantage. A seller who cares about lifetime revenue (including the learning period) would weakly prefer FPA. A seller who only cares about the steady state would be indifferent.

### Experiment 2: Q-learning, Affiliated Values (10 responses)

| Response | Coefficient | t-stat | p-value | Holm sig? | Direction |
|----------|-------------|-------|---------|-----------|-----------|
| Converged revenue | -0.027 | 3.2 | 0.002 | No | SPA higher |
| Lifetime revenue | +0.013 | 1.4 | 0.152 | No | -- |
| Convergence time | +109 | 0.1 | 0.939 | No | -- |
| No-sale rate | 0.000 | -- | -- | N/A | Constant zero |
| Price volatility | +0.009 | 3.8 | 2.2e-04 | **Yes** | FPA more volatile |
| Winner entropy | -0.022 | 1.0 | 0.324 | No | -- |
| Bid-to-value ratio | -0.051 | 4.6 | 7.7e-06 | **Yes** | SPA higher (closer to truthful) |
| Winner's curse freq | -0.084 | 7.6 | 1.8e-12 | **Yes** | SPA more cursed |
| Bid dispersion | -0.015 | 4.1 | 6.2e-05 | **Yes** | SPA higher (wider bid spread) |
| Signal slope ratio | +0.006 | 0.9 | 0.393 | No | -- |

**Interpretation:** Under affiliated values, SPA earns higher converged revenue (but this does not survive Holm correction). The robust findings are about *bidding behavior*, not revenue: SPA produces more winner's curse events (26.5% vs 9.8% under FPA), while FPA produces more price volatility and lower bid-to-value ratios (more shading). SPA bidders bid closer to their values and display more varied strategies (higher dispersion), but this produces nearly three times the curse rate of FPA.

### Experiment 3a: LinUCB Contextual Bandits (6 responses)

| Response | Coefficient | t-stat | p-value | Holm sig? | Direction |
|----------|-------------|-------|---------|-----------|-----------|
| Converged revenue | -0.048 | 12.5 | 1.5e-32 | **Yes** | SPA higher |
| Lifetime revenue | -0.046 | 13.5 | 3.6e-37 | **Yes** | SPA higher |
| Convergence time | -12,445 | 10.9 | 6.2e-26 | **Yes** | FPA faster |
| No-sale rate | -0.002 | 2.8 | 0.005 | No | -- |
| Price volatility | -0.019 | 13.4 | 1.2e-36 | **Yes** | SPA more volatile |
| Winner entropy | +0.026 | 4.2 | 2.7e-05 | **Yes** | FPA more equal |

**Interpretation:** The strongest and most consistent auction format effect among unconstrained experiments. SPA dominates revenue (both converged and lifetime), but FPA converges faster and produces more equal winner distributions. The volatility finding is notable: under LinUCB, *SPA* is more volatile, reversing the FPA-volatile pattern from Q-learning.

### Experiment 3b: Thompson Sampling Bandits (6 responses)

| Response | Coefficient | t-stat | p-value | Holm sig? | Direction |
|----------|-------------|-------|---------|-----------|-----------|
| Converged revenue | -0.023 | 3.8 | 1.8e-04 | **Yes** | SPA higher |
| Lifetime revenue | -0.020 | 3.5 | 5.8e-04 | No | SPA higher |
| Convergence time | -2,458 | 1.6 | 0.116 | No | -- |
| No-sale rate | +0.001 | 0.7 | 0.460 | No | -- |
| Price volatility | -0.012 | 4.2 | 3.8e-05 | **Yes** | SPA more volatile |
| Winner entropy | +0.006 | 0.3 | 0.734 | No | -- |

**Interpretation:** Same directional pattern as LinUCB (SPA higher revenue, SPA more volatile) but attenuated. Only converged revenue and price volatility survive Holm correction. Thompson Sampling's stochastic posterior sampling adds noise that obscures format differences, consistent with its lower R².

### Experiment 4a: Multiplicative Dual Pacing (16 responses)

| Response | Coefficient | t-stat | p-value | Holm sig? | Direction |
|----------|-------------|-------|---------|-----------|-----------|
| Revenue | +$184 | 3.6 | 3.3e-04 | No | FPA higher |
| Lifetime revenue | +$184 | 3.6 | 3.4e-04 | No | FPA higher |
| Liquid welfare | +$7 | 0.3 | 0.782 | No | -- |
| LP offline welfare | +$3 | 0.1 | 0.905 | No | -- |
| Effective PoA | -0.001 | 0.7 | 0.493 | No | -- |
| LP-based PoA | -0.001 | 0.7 | 0.494 | No | -- |
| Budget utilization | +0.022 | 8.4 | 4.5e-16 | **Yes** | FPA higher |
| Bid-to-value | -0.841 | 2.1 | 0.034 | No | -- |
| Allocative efficiency | -0.006 | 2.1 | 0.037 | No | -- |
| Dual variable CV | -0.010 | 3.0 | 0.003 | No | -- |
| No-sale rate | +0.001 | 5.6 | 3.4e-08 | **Yes** | FPA higher |
| Winner entropy | +0.003 | 0.5 | 0.622 | No | -- |
| Warm-start benefit | -$5.7 | 2.6 | 0.009 | No | -- |
| Inter-episode vol. | -0.001 | 3.1 | 0.002 | No | -- |
| Bid suppression | -0.501 | 1.3 | 0.205 | No | -- |
| Cross-episode drift | -0.026 | 2.1 | 0.034 | No | -- |

**Interpretation:** Under dual pacing, only 2 of 16 responses survive Holm correction for auction format: budget utilization (FPA higher) and no-sale rate (FPA higher). Revenue is raw-significant but does not survive multiplicity correction. Welfare and efficiency metrics are all unaffected by auction format.

### Experiment 4b: PI Controller Pacing (16 responses)

| Response | Coefficient | t-stat | p-value | Holm sig? | Direction |
|----------|-------------|-------|---------|-----------|-----------|
| Revenue | +$458 | 11.0 | 2.0e-25 | **Yes** | FPA higher |
| Lifetime revenue | +$458 | 11.0 | 2.0e-25 | **Yes** | FPA higher |
| Liquid welfare | -$55 | 1.8 | 0.080 | No | -- |
| LP offline welfare | -$4 | 0.1 | 0.903 | No | -- |
| Effective PoA | +0.014 | 10.7 | 3.5e-24 | **Yes** | FPA higher (worse) |
| LP-based PoA | +0.014 | 10.7 | 3.9e-24 | **Yes** | FPA higher (worse) |
| Budget utilization | +0.061 | 12.1 | 8.0e-30 | **Yes** | FPA higher |
| Bid-to-value | -0.159 | 22.3 | 2.4e-76 | **Yes** | FPA lower (more shading) |
| Allocative efficiency | -0.040 | 7.7 | 6.8e-14 | **Yes** | FPA lower (worse) |
| Dual variable CV | +0.193 | 22.1 | 1.2e-75 | **Yes** | FPA higher (less stable) |
| No-sale rate | +0.027 | 12.7 | 4.4e-32 | **Yes** | FPA higher |
| Winner entropy | +0.047 | 6.1 | 1.8e-09 | **Yes** | FPA higher (more equal) |
| Warm-start benefit | -$0.8 | 0.3 | 0.751 | No | -- |
| Inter-episode vol. | -0.005 | 11.9 | 6.1e-29 | **Yes** | FPA lower (more stable) |
| Bid suppression | -0.159 | 22.3 | 2.4e-76 | **Yes** | FPA lower |
| Cross-episode drift | -0.000 | 0.4 | 0.709 | No | -- |

**Interpretation:** Auction format is a dominant force under PI pacing, affecting 12 of 16 metrics. FPA generates 25% more revenue but at a cost: higher Price of Anarchy, lower allocative efficiency, more no-sale events, and much greater pacing instability (dual variable CV doubles). FPA also produces more equal winner distributions and less cross-episode revenue variance. Format differences are far larger under PI pacing than under dual pacing.

---

## The Seller's Verdict: FPA vs SPA (Multi-Dimensional)

A platform choosing between FPA and SPA cares about many outcomes beyond revenue. This section weighs all the evidence across every dimension we measured, organized by what a seller would care about.

### Dimension 1: Revenue

| Setting | Winner | Gap | Robust? | Strength |
|---------|--------|-----|---------|----------|
| Q-learning, constant (Exp 1) | Tie | 0.8% | Not sig | **Strong null** (n=1024) |
| Q-learning, affiliated (Exp 2) | SPA | 11.7% | Not Holm | Weak |
| LinUCB (Exp 3a) | **SPA** | **21.0%** | **Holm-sig** | **Strong** |
| Thompson (Exp 3b) | SPA | 9.1% | Holm-sig | Moderate |
| Dual pacing (Exp 4a) | FPA | 8.6% | Not Holm | Weak |
| PI pacing (Exp 4b) | **FPA** | **25.1%** | **Holm-sig** | **Strong** |

**Verdict on revenue:** Split decision. SPA wins 2 experiments with Holm-significant effects (strongly in 3a, moderately in 3b), FPA wins 1 (strongly in 4b), 3 show no robust effect. The two strongest robust results are nearly symmetric in magnitude (21% vs 25%). Revenue alone does not settle the question.

*Footnote on Exp 1 lifetime revenue:* While converged revenue shows no format effect, FPA earns significantly more during the learning phase (lifetime revenue |t|=6.8, Holm p=6.9e-9). A seller who cares about total revenue including the transient learning period would weakly prefer FPA in constant-valuation Q-learning settings. This is not counted as a revenue "win" to avoid double-counting with the converged-revenue metric used for all other experiments.

### Dimension 2: Price Volatility (Bidder Experience)

Price volatility measures how unpredictable winning prices are. High volatility makes it harder for bidders to budget, plan, and trust the marketplace. Sellers care because volatile markets drive away sophisticated bidders and increase support costs.

| Setting | More volatile format | t-stat | Robust? |
|---------|---------------------|-------|---------|
| Q-learning, constant (Exp 1) | FPA | 2.3 | No |
| Q-learning, affiliated (Exp 2) | **FPA** | **3.8** | **Holm-sig** |
| LinUCB (Exp 3a) | **SPA** | **13.4** | **Holm-sig** |
| Thompson (Exp 3b) | **SPA** | **4.2** | **Holm-sig** |
| Dual pacing (Exp 4a) | -- | -- | Not measured directly |
| PI pacing (Exp 4b) | -- | -- | Not measured directly |

**Verdict on volatility:** Genuinely mixed. The *direction flips by algorithm family*. Under Q-learning, FPA is more volatile. Under bandits, SPA is more volatile. Neither format is universally "smoother." Which format is less volatile depends on which algorithm bidders run.

### Dimension 3: No-Sale Rate (Wasted Inventory)

A no-sale means no bids cleared the reserve, wasting the seller's inventory slot. Even without reserves, no-sale can occur if all bids are zero.

| Setting | Higher no-sale format | t-stat | Robust? | Magnitudes |
|---------|----------------------|-------|---------|------------|
| Q-learning, constant (Exp 1) | -- | 1.0 | No | Both ~1.1% |
| Q-learning, affiliated (Exp 2) | -- | -- | N/A | Both 0.0% |
| LinUCB (Exp 3a) | -- | 2.8 | No | Both ~1.8% |
| Thompson (Exp 3b) | -- | 0.7 | No | Both ~2.0% |
| Dual pacing (Exp 4a) | **FPA** | **5.6** | **Holm-sig** | FPA 0.73%, SPA 0.44% |
| PI pacing (Exp 4b) | **FPA** | **12.7** | **Holm-sig** | FPA 5.5%, SPA 0.16% |

**Verdict on no-sale:** In unconstrained settings, auction format does not affect no-sale rates. Under pacing, FPA consistently produces more no-sales, and the gap is dramatic under PI pacing (5.5% vs 0.16%, a 34x ratio).

For a seller running paced auctions, FPA's higher no-sale rate means 5.3 percentage points of inventory waste under PI pacing. That is real money left on the table, partially offsetting FPA's 25% revenue *conditional on sale*.

### Dimension 4: Allocative Efficiency (Right Item to Right Bidder)

Allocative efficiency measures whether the highest-value bidder actually wins. Low efficiency means items go to the wrong bidders, destroying value and potentially driving high-value bidders away.

| Setting | More efficient format | t-stat | Robust? | Gap |
|---------|----------------------|-------|---------|-----|
| Exp 1-3b | Not measured (implicit in welfare) | -- | -- | -- |
| Dual pacing (Exp 4a) | -- | 2.1 | No | Negligible (75.0% vs 76.2%) |
| PI pacing (Exp 4b) | **SPA** | **7.7** | **Holm-sig** | SPA 72.6%, FPA 64.7% |

**Verdict on efficiency:** Under dual pacing, format barely matters for efficiency. Under PI pacing, SPA allocates 8 percentage points more efficiently (72.6% vs 64.7%). This is substantial: nearly 1 in 12 items under FPA goes to the wrong bidder compared to SPA.

### Dimension 5: Price of Anarchy (Welfare Loss)

Price of Anarchy measures how much total welfare is lost due to strategic behavior compared to an omniscient allocation. A PoA of 1.0 means no loss; higher is worse. Regulators and policy-minded platforms care about this.

| Setting | Higher PoA (worse) | t-stat | Robust? | Values |
|---------|-------------------|-------|---------|--------|
| Dual pacing (Exp 4a) | -- | 0.7 | No | Both ~1.036 |
| PI pacing (Exp 4b) | **FPA** | **10.7** | **Holm-sig** | FPA 1.036, SPA 1.008 |

**Verdict on PoA:** Under dual pacing, format doesn't affect welfare. Under PI pacing, FPA produces 3.6% welfare loss vs SPA's 0.8%. SPA is nearly efficient; FPA is not. This is important for platforms operating under regulatory scrutiny or in markets where welfare arguments matter (e.g., spectrum auctions, public procurement).

### Dimension 6: Budget Utilization (Are Bidders Spending?)

Higher budget utilization means bidders are deploying their budgets, which correlates with bidder satisfaction (they got what they came for) and platform revenue (more money flowing through).

| Setting | Higher utilization | t-stat | Robust? | Values |
|---------|-------------------|-------|---------|--------|
| Dual pacing (Exp 4a) | **FPA** | **8.4** | **Holm-sig** | FPA 85.0%, SPA 80.5% |
| PI pacing (Exp 4b) | **FPA** | **12.1** | **Holm-sig** | FPA 82.0%, SPA 69.9% |

**Verdict on budget utilization:** FPA consistently achieves higher budget utilization in both pacing experiments, and the effect is robust. The 12-point gap under PI pacing (82.0% vs 69.9%) is especially large. Higher utilization means more money flowing through the platform from the seller's perspective.

### Dimension 7: Convergence Speed (Time to Stable Market)

How quickly does the market settle into a stable equilibrium? Matters for new product launches, new advertisers entering, or any scenario where the learning phase is a cost.

| Setting | Faster format | t-stat | Robust? | Gap (episodes) |
|---------|--------------|-------|---------|----------------|
| Q-learning, constant (Exp 1) | **FPA** | **10.1** | **Holm-sig** | FPA 45,055 vs SPA 61,541 |
| Q-learning, affiliated (Exp 2) | -- | 0.1 | No | Same (~82,000) |
| LinUCB (Exp 3a) | **FPA** | **10.9** | **Holm-sig** | FPA 26,523 vs SPA 51,414 |
| Thompson (Exp 3b) | -- | 1.6 | No | FPA ~18,700 vs SPA ~23,600 |

**Verdict on convergence:** FPA converges faster in 2 of 4 unconstrained experiments (Exp 1 and 3a), both with strong evidence. In Exp 1, FPA converges 27% faster; in Exp 3a, 48% faster.

In Exp 3a, FPA converges faster to a *worse* equilibrium (lower revenue). SPA takes longer but arrives at a better outcome. Fast convergence is only valuable if the destination is good.

### Dimension 8: Winner Fairness (Who Gets to Win?)

Winner entropy measures how equally wins are distributed across bidders. High entropy = many bidders win sometimes. Low entropy = one bidder dominates.

| Setting | More equal (higher entropy) | t-stat | Robust? |
|---------|----------------------------|-------|---------|
| Q-learning, constant (Exp 1) | -- | 1.5 | No |
| Q-learning, affiliated (Exp 2) | -- | 1.0 | No |
| LinUCB (Exp 3a) | **FPA** | **4.2** | **Holm-sig** |
| Thompson (Exp 3b) | -- | 0.3 | No |
| Dual pacing (Exp 4a) | -- | 0.5 | No |
| PI pacing (Exp 4b) | **FPA** | **6.1** | **Holm-sig** |

**Verdict on winner fairness:** When format matters (Exp 3a and 4b), FPA distributes wins more equally across bidders. This effect appears in only 2 of 6 experiments.

### Dimension 9: Pacing Stability (Bidder Control Experience)

Dual variable CV measures how much the pacing multiplier fluctuates within a campaign. High CV = the controller is constantly overcorrecting, leading to erratic bids and an unpredictable experience for the bidder.

| Setting | More stable (lower CV) | t-stat | Robust? | Values |
|---------|----------------------|-------|---------|--------|
| Dual pacing (Exp 4a) | -- | 3.0 | No | FPA 0.138, SPA 0.157 |
| PI pacing (Exp 4b) | **SPA** | **22.1** | **Holm-sig** | FPA 0.734, SPA 0.349 |

**Verdict on pacing stability:** Under PI pacing, SPA produces dramatically more stable pacing (CV of 0.35 vs 0.73, a 2.1x ratio). This is one of the *strongest* format effects in the entire study (|t|=22.1). Under dual pacing, pacing stability is comparable across formats.

For a platform selling an "autobidding" product, pacing stability directly affects advertiser trust. SPA's stability advantage under PI-style pacing is a significant non-revenue argument in its favor.

### Dimension 10: Revenue Consistency (Predictable Platform Income)

Inter-episode revenue volatility measures how much platform revenue varies across campaign cycles. Lower volatility means more predictable income for the seller, easier financial planning, and a more stable marketplace signal to bidders.

| Setting | More consistent (lower vol.) | t-stat | Robust? | Values |
|---------|------------------------------|-------|---------|--------|
| Exp 1-3b | Not applicable (single-episode framework) | -- | -- | -- |
| Dual pacing (Exp 4a) | -- | 3.1 | No | FPA lower, but not Holm-sig |
| PI pacing (Exp 4b) | **FPA** | **11.9** | **Holm-sig** | FPA substantially lower inter-episode volatility |

**Verdict on revenue consistency:** Under PI pacing, FPA produces significantly more consistent revenue across campaign cycles (|t|=11.9, Holm p=1.86e-26). Under dual pacing, the effect trends the same direction but is not robust after correction.

### Dimension 11: Winner's Curse (Bidder Regret)

The winner's curse measures how often winning bidders overpay relative to their true value. High curse frequency erodes bidder trust and long-term participation. This is a direct outcome that sellers care about because cursed bidders leave the platform.

Note on bid-to-value ratios (BTV): BTV measures a bidding *input* (how agents bid relative to value), not an *outcome* a seller independently cares about. BTV's effects are already captured in the outcome dimensions (revenue, efficiency, no-sale rate). Including it as a separate scorecard dimension would double-count. We therefore score this dimension on winner's curse alone.

| Setting | Fewer curse events | t-stat | Robust? | Values |
|---------|-------------------|-------|---------|--------|
| Q-learning, affiliated (Exp 2) | **FPA** | **7.6** | **Holm-sig** | FPA 9.8%, SPA 26.5% |
| Other experiments | Not measured | -- | -- | -- |

**Verdict on winner's curse:** In the one experiment that measures curse frequency, FPA produces dramatically fewer regret events (9.8% vs 26.5%, a 2.7x ratio favoring FPA). This finding is limited to a single experiment with affiliated-value Q-learning and may not generalize to other algorithm families.

### Dimension 12: Cross-Episode Learning and Collusion Risk

Cross-episode drift measures whether bidders progressively learn to suppress bids over repeated campaign cycles. A positive drift toward lower bids would indicate algorithmic collusion emerging over time.

| Setting | Evidence of progressive collusion? | t-stat | Robust? |
|---------|-----------------------------------|-------|---------|
| Dual pacing (Exp 4a) | No | 2.1 | No |
| PI pacing (Exp 4b) | No | 0.4 | No |

**Verdict on collusion risk:** Neither format shows evidence of progressive bid suppression under pacing. A meaningful null result, equally reassuring for both FPA and SPA.

---

### Multi-Dimensional Scorecard

Summarizing all robust (Holm-Bonferroni significant, α=0.05) format effects across all experiments and all outcomes. Each dimension is an outcome a seller independently cares about; behavioral inputs (e.g., bid-to-value ratio) whose effects are already captured by outcome dimensions are excluded to avoid double-counting.

| Dimension | FPA Wins | SPA Wins | NS | Net |
|-----------|----------|----------|-----|-----|
| Revenue | 1 (Exp 4b) | 2 (Exp 3a, 3b) | 3 | **SPA +1** |
| Price stability | 2 (Exp 3a, 3b) | 1 (Exp 2) | 1 | **FPA +1** |
| Revenue consistency | 1 (Exp 4b) | 0 | 1 | **FPA +1** |
| No-sale rate | 0 | 2 (Exp 4a, 4b) | 3 | **SPA +2** |
| Allocative efficiency | 0 | 1 (Exp 4b) | 1 | **SPA +1** |
| Welfare (PoA) | 0 | 1 (Exp 4b) | 1 | **SPA +1** |
| Budget utilization | 2 (Exp 4a, 4b) | 0 | 0 | **FPA +2** |
| Convergence speed | 2 (Exp 1, 3a) | 0 | 2 | **FPA +2** |
| Winner fairness | 2 (Exp 3a, 4b) | 0 | 4 | **FPA +2** |
| Pacing stability | 0 | 1 (Exp 4b) | 1 | **SPA +1** |
| Winner's curse | 1 (Exp 2) | 0 | 0 | **FPA +1** |
| Collusion risk | 0 | 0 | 2 | **Tied** |
| **Total robust wins** | **11** | **8** | | **FPA +3** |

*NS columns exclude experiments where the metric was not measured (N/A). Exp 1 lifetime revenue (FPA Holm-sig advantage during the learning phase) is noted in Dimension 1 but not scored separately to avoid double-counting with converged revenue.*

### What the Scorecard Means

The simple count (FPA 11, SPA 8) favors FPA, but the picture is nuanced and the scorecard should not be read as declaring an overall winner.

**SPA's wins are concentrated in Experiment 4b (PI pacing).** Five of SPA's eight wins come from a single experiment. Under PI pacing, SPA dominates on efficiency, welfare, pacing stability, and no-sale rate. Remove Exp 4b and SPA retains only three wins across the remaining five experiments (revenue in 3a and 3b, no-sale rate in 4a).

**FPA's wins are spread across experiments.** FPA wins in five of six experiments across multiple dimensions: convergence speed (Exp 1, 3a), price stability (Exp 3a, 3b), budget utilization (Exp 4a, 4b), revenue consistency (Exp 4b), winner fairness (Exp 3a, 4b), revenue (Exp 4b), and winner's curse protection (Exp 2).

**The winner depends on what you weight.** A regulator who weights allocative efficiency, welfare, and pacing stability heavily would favor SPA. A seller who weights revenue, convergence speed, and budget deployment would favor FPA. Neither format dominates on all dimensions that matter.

**The volatility direction reversal is itself a finding.** Under Q-learning, FPA produces more volatile prices. Under contextual bandits, SPA produces more volatile prices. The "less volatile" format depends on which algorithms bidders run.

### The Honest Bottom Line

1. **Format is never the biggest lever.** Across all dimensions, market structure (bidders, budgets) dominates format choice by 1.5x–28x. Getting one more bidder into each auction does more for revenue, efficiency, and every other metric than switching formats.

2. **On pure revenue, it's close.** Two robust experiments favor SPA (3a at 21%, 3b at 9%), one robust experiment favors FPA (4b at 25%), three show no robust effect. The magnitudes of the two strongest results are nearly symmetric. Anyone claiming one format is universally better for revenue is overstating the evidence.

3. **Neither format dominates on market health.** SPA accumulates wins on no-sale rate, allocative efficiency, welfare, and pacing stability, but these are concentrated in Exp 4b. FPA accumulates wins on price stability, convergence speed, winner fairness, revenue consistency, and winner's curse protection, spread across multiple experiments. The "healthier marketplace" conclusion depends on which health metrics you prioritize and which algorithm family your bidders use.

4. **FPA's revenue advantage under pacing comes with heavy side effects.** The 25% FPA revenue edge under PI pacing (Exp 4b) is accompanied by 5.5% no-sale rate (vs 0.2%), 8-point efficiency loss, doubled pacing instability, and 3.6% welfare loss (vs 0.8%). Under dual pacing (Exp 4a), the FPA revenue edge is borderline and the side effects are minimal. The pacing algorithm's sophistication determines how costly FPA's revenue premium is.

5. **SPA's revenue advantage under bandits is clean.** The 21% SPA revenue edge under LinUCB (Exp 3a) comes with *no* adverse side effects on efficiency or welfare. SPA is more volatile under bandits, but that reflects prices tracking underlying values, not market dysfunction.

6. **The interaction with competition is universal.** In every unconstrained experiment, the FPA-SPA gap (in any direction) widens with more bidders. Format choice matters more in competitive, thick markets and barely matters in thin ones.

7. **Five of SPA's eight wins come from one experiment.** Exp 4b (PI pacing) is SPA's strongest showing. If a platform uses a more sophisticated pacing algorithm (like Exp 4a's dual pacing), most of SPA's market-health advantages vanish. The choice between FPA and SPA is therefore entangled with the choice of pacing technology.

8. **Industry context:** Major platforms (Google, Meta) switched from SPA to FPA for pacing-autobidding environments, optimizing for revenue extraction. Our data is consistent with that choice on the revenue dimension. But the data also shows this comes at a real cost to market quality that those platforms may be absorbing or passing to advertisers.

---

## Code Audit Summary

**Simulation code (Phase 1):** CLEAN. All auction mechanics (FPA/SPA payment rules, reserve price filtering, no-sale encoding, tie-breaking), valuation models, and algorithm implementations verified against theory sections. 15/15 mathematical verification tests pass.

**Estimation code (Phase 2):** CLEAN. OLS formula construction, Type III ANOVA, 13 robustness checks, and per-experiment wrappers all verified. All 6 estimation + robustness pipelines re-run successfully. Table/figure generation pipeline validated.
