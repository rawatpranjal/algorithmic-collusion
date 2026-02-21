# Hypothesis Map: Literature Predictions vs. Experimental Evidence

This document maps testable hypotheses from the algorithmic collusion and auction theory literature to the four factorial experiments in this paper. Each hypothesis states a falsifiable prediction, identifies its source, specifies which experiment(s) and metrics test it, and records the verdict from the experimental results.

**Verdict key**: **Supported** = data confirms prediction; **Challenged** = data contradicts prediction; **Partially supported** = some aspects confirmed, others not; **Null** = no significant effect detected where one was predicted.

---

## 1. Auction Format and Collusion

### H1.1: First-price auctions are more prone to bid suppression than second-price auctions

- **Statement**: Under algorithmic learning, first-price auctions produce systematically lower revenue than second-price auctions due to strategic bid shading that compounds through repeated interaction.
- **Source**: Banchio & Skrzypacz (2022)
- **Testable in**: All experiments. Factor: `auction_type_coded`. Metrics: `avg_rev_last_1000`, `avg_regret_of_seller`.
- **Expected outcome**: FPA yields lower revenue and higher seller regret than SPA across all settings.
- **Verdict**: **Partially supported.** Auction type is the dominant effect in Exp1, Exp3, and Exp4. In Exp2 (affiliated valuations, 48 observations), the auction type main effect on revenue is not statistically significant ($p = 0.11$), though directional patterns persist. The FPA revenue gap is large in Exp3 (contextual bandits) and Exp4 (pacing), but the Exp2 design lacks power to detect it. Auction type significantly increases price volatility even in Exp2.

### H1.2: Revenue equivalence holds under symmetric independent private values

- **Statement**: With risk-neutral bidders, independent private values, and symmetric equilibrium play, first-price and second-price auctions yield identical expected revenue.
- **Source**: Myerson (1981); Vickrey (1961); Riley & Samuelson (1981)
- **Testable in**: Exp1 (constant valuations, v=1), Exp2 (affiliated valuations with eta=0 giving independent PV). Metric: `avg_rev_last_1000`, `ratio_to_theory`.
- **Expected outcome**: FPA and SPA revenue should converge to identical levels at equilibrium.
- **Verdict**: **Challenged.** Learning agents violate revenue equivalence in Exp1, where FPA underperforms SPA despite constant valuations. In Exp2, the FPA-SPA difference is not statistically significant with the small-sample design ($p = 0.11$), though this reflects limited power rather than convergence to equivalence.

### H1.3: First-price auctions have a unique equilibrium while second-price auctions have multiple equilibria

- **Statement**: In first-price pacing markets, equilibria are computationally tractable (polynomial-time computable), while second-price pacing equilibria are PPAD-hard to find.
- **Source**: Conitzer et al. (2022); Chen et al. (2023)
- **Testable in**: Exp4 (pacing agents). Metrics: `inter_episode_volatility`, `dual_cv`, `price_volatility`.
- **Expected outcome**: FPA should exhibit more stable convergence (lower volatility) than SPA if agents reliably find the unique FPA equilibrium, while SPA may exhibit equilibrium selection instability.
- **Verdict**: **Partially supported.** Exp4 shows that FPA bidding converges to stable but sub-competitive equilibria (low volatility, low revenue). SPA outcomes are more variable, consistent with equilibrium multiplicity, but SPA also achieves higher revenue on average.

### H1.4: FPA welfare guarantee of at least 1/2 under no-regret learning

- **Statement**: In first-price auctions with no-regret learners, the liquid welfare is at least half of the optimal welfare.
- **Source**: Fikioris & Tardos (2024)
- **Testable in**: Exp1 and Exp3 (no-regret algorithms). Metric: `avg_rev_last_1000` relative to theoretical maximum.
- **Expected outcome**: Revenue should not drop below 50% of the efficient benchmark even in worst cases.
- **Verdict**: **Partially supported.** In Exp1 (Q-learning), average FPA revenue is 0.718, well above the 0.5 bound. However, in Exp3 (LinUCB), some configurations produce near-zero revenue, violating the 1/2 guarantee. LinUCB's exploration mechanism may not satisfy the no-regret condition assumed by the theorem.

### H1.5: FPA Price of Anarchy equals 1/2 under autobidding

- **Statement**: In first-price auctions with value-maximizing autobidders, the Price of Anarchy is bounded by 1/2.
- **Source**: Deng et al. (2024)
- **Testable in**: Exp4. Factor: `auction_type_coded` crossed with `objective_coded`. Metric: `mean_effective_poa`.
- **Expected outcome**: FPA + value-maximizer cells should exhibit effective PoA near 2 (welfare loss of ~50%).
- **Verdict**: **Testable via Exp4 data.** The effective PoA varies across factor combinations, with auction format and objective jointly determining efficiency losses. The PoA framework provides the natural benchmark for Exp4 results.

### H1.6: Second-price auction properties break down under learning

- **Statement**: The theoretical incentive compatibility of second-price auctions (truthful bidding as dominant strategy) does not hold when agents learn through trial-and-error rather than strategic reasoning.
- **Source**: Kolumbus & Nisan (2022)
- **Testable in**: All experiments. Factor: `auction_type_coded`. Metric: `avg_rev_last_1000` compared to theoretical truthful-bidding benchmark.
- **Expected outcome**: SPA revenue should fall below the theoretical prediction if agents fail to discover truthful bidding.
- **Verdict**: **Partially supported.** SPA consistently outperforms FPA, and agents approximate truthful bidding more closely in SPA. However, SPA revenue does not always reach the theoretical maximum, especially in Exp3 (bandits), suggesting that even dominant-strategy mechanisms require some strategic sophistication to achieve full efficiency under learning.

### H1.7: Affiliation should increase SPA revenue advantage over FPA

- **Statement**: Under affiliated valuations, the linkage principle predicts that second-price auctions generate strictly higher revenue than first-price auctions, with the gap increasing in affiliation strength.
- **Source**: Milgrom & Weber (1982)
- **Testable in**: Exp2 and Exp3. Factor: `eta_linear_coded`. Metric: `avg_rev_last_1000`.
- **Expected outcome**: Revenue gap between SPA and FPA should widen as eta increases from 0 to 1.
- **Verdict**: **Null.** Affiliation (eta) has no significant effect on the SPA-FPA revenue gap in either Exp2 or Exp3. The null eta effect is one of the most consistent findings: the private-vs-common-value distinction does not alter relative auction performance under algorithmic learning. This is inconsistent with the linkage principle's prediction for equilibrium play, though the low power of Exp2 limits the strength of the contradiction.

---

## 2. Learning Algorithm Design

### H2.1: Synchronous learning produces competitive outcomes; asynchronous learning enables collusion

- **Statement**: Agents that update Q-values for all actions (synchronous, using counterfactual rewards) converge to competitive equilibria, while agents that update only the chosen action (asynchronous) can sustain collusive bid suppression.
- **Source**: Asker et al. (2024)
- **Testable in**: Exp1. Factor: `asynchronous_coded`. Metrics: `avg_rev_last_1000`, `avg_regret_of_seller`.
- **Expected outcome**: Synchronous updates (coded -1 in Exp1 design) should yield higher revenue and lower regret than asynchronous updates (+1).
- **Verdict**: **Supported.** Asynchronous updating ranks as the third most important factor in Exp1, with asynchronous Q-learning increasing first-price losses. Synchronous updates dampen FPA underperformance by maintaining uniform knowledge across all actions, preventing lock-in to low-bid equilibria.

### H2.2: Collusion under Q-learning is genuine, not an artifact of slow convergence

- **Statement**: The sub-competitive outcomes observed under Q-learning are stable equilibria, not transient phenomena caused by insufficient learning time.
- **Source**: Calvano et al. (2023)
- **Testable in**: Exp1, Exp2. Metrics: `time_to_converge`, `avg_rev_last_1000` (measured after convergence).
- **Expected outcome**: Revenue should remain sub-competitive even after convergence, and extending training should not eliminate the gap.
- **Verdict**: **Supported.** In Exp1, Q-learning agents converge (median ~1,000 episodes) but final revenue remains well below the competitive benchmark. The FPA revenue penalty persists in the converged regime, confirming genuine rather than spurious collusion. Exp2 convergence patterns are consistent but the small sample (48 runs) limits the precision of convergence time estimates.

### H2.3: More sophisticated algorithms produce less collusion

- **Statement**: Algorithms with more efficient exploration (contextual bandits, UCB-based methods) should reduce collusive outcomes by discovering and exploiting competitive strategies faster.
- **Source**: Abada et al. (2024); general claim in literature
- **Testable in**: Cross-experiment comparison of Exp1-2 (Q-learning) vs. Exp3 (LinUCB/CTS). Metric: `avg_rev_last_1000`.
- **Expected outcome**: Exp3 should yield higher FPA revenue than Exp1-2 if sophistication helps.
- **Verdict**: **Challenged.** LinUCB bandits (Exp3) produce dramatically worse seller outcomes than Q-learning (Exp1-2). The FPA revenue gap widens from -14.4% (Exp1) to -42.3% (Exp3). Algorithmic sophistication does not inherently mitigate first-price disadvantages and can exacerbate negative outcomes when the exploration mechanism reinforces conservative bidding.

### H2.4: Collusion is the exception, not the rule, across algorithm classes

- **Statement**: When testing a broad range of algorithms and parameter settings, most configurations do not produce collusive outcomes; collusion requires specific parameter combinations.
- **Source**: Bichler et al. (2025)
- **Testable in**: All experiments. Distribution of `avg_rev_last_1000` across all factorial cells.
- **Expected outcome**: Collusive configurations should be a minority of the parameter space.
- **Verdict**: **Partially supported.** Under SPA, most configurations achieve near-competitive revenue, consistent with this hypothesis. Under FPA, however, sub-competitive outcomes are the norm rather than the exception across all algorithm classes. The finding is algorithm-specific: FPA collusion is widespread, not rare.

### H2.5: Mean-based learners converge to Nash equilibrium

- **Statement**: Learning algorithms that base decisions on mean reward estimates (rather than best-response dynamics) converge to Nash equilibrium in auctions.
- **Source**: Feng et al. (2021); Deng et al. (2022)
- **Testable in**: Exp3 (LinUCB uses mean reward estimates). Metric: `ratio_to_theory`.
- **Expected outcome**: LinUCB revenue should approach BNE predictions more closely than Q-learning.
- **Verdict**: **Challenged.** LinUCB produces worse revenue outcomes than Q-learning, with more extreme deviations from BNE predictions. The mean-based structure of LinUCB does not ensure convergence to Nash in the multi-agent auction setting tested here.

### H2.6: Learning rate determines cooperation level

- **Statement**: Higher learning rates lead to more competitive outcomes because agents adapt faster to competitors' deviations from collusive strategies, while lower learning rates sustain collusion by making agents slow to respond.
- **Source**: Dolgopolov (2024)
- **Testable in**: Exp1. Factor: `alpha_coded`. Metric: `avg_rev_last_1000`.
- **Expected outcome**: Higher alpha should increase revenue if faster adaptation disrupts collusion.
- **Verdict**: **Partially supported.** Learning rate has a detectable but minor effect in Exp1, far smaller than auction type, exploration strategy, or synchronous/asynchronous updating. It is not among the dominant factors, suggesting that learning rate modulates collusion at the margin but does not determine it.

### H2.7: Exploration mechanism matters more than exploration rate

- **Statement**: The type of exploration (Boltzmann vs. epsilon-greedy; UCB vs. Thompson Sampling) has a larger impact on collusion outcomes than the magnitude of the exploration parameter.
- **Source**: Cross-experiment inference; Waltman & Kaymak (2008); Dolgopolov (2021)
- **Testable in**: Exp1 (Boltzmann vs. epsilon-greedy), Exp3 (LinUCB vs. CTS; exploration intensity).
- **Expected outcome**: Exploration type should have larger t-statistics than exploration rate parameters.
- **Verdict**: **Supported.** In Exp1, exploration strategy (Boltzmann vs. epsilon-greedy) is the second-largest effect on revenue, far outranking any rate parameter. In Exp3, exploration intensity has a significant but smaller effect than the algorithm choice (LinUCB vs. CTS). The mechanism through which agents explore is consistently more consequential than how much they explore.

### H2.8: Action-space granularity affects collusion

- **Statement**: Finer discretisation of the bid space (more actions) should reduce collusion by making it harder for agents to coordinate on a common low bid.
- **Source**: Klein (2021)
- **Testable in**: Exp1. Factor: `n_actions_coded` (11 vs. 21 actions). Metric: `avg_rev_last_1000`.
- **Expected outcome**: More actions should increase revenue by disrupting coordination.
- **Verdict**: **Null.** Number of actions is not among the dominant effects in Exp1. The grid size (11 vs. 21 actions) does not materially change revenue, regret, or volatility outcomes, suggesting that the bid grid resolution at these levels does not meaningfully affect collusion dynamics.

---

## 3. Valuation Structure

### H3.1: Common-value elements alter optimal bidding via winner's curse

- **Statement**: When valuations depend on others' private information (common-value components), rational bidders should shade bids more aggressively to avoid the winner's curse, and this effect should intensify with affiliation strength.
- **Source**: Classical auction theory (Milgrom & Weber 1982; Krishna 2009)
- **Testable in**: Exp2, Exp3. Factor: `eta_linear_coded`, `eta_quadratic_coded`. Metric: `avg_rev_last_1000`.
- **Expected outcome**: Higher eta should reduce revenue if agents learn to shade bids for winner's curse protection.
- **Verdict**: **Null.** Affiliation (eta) has no statistically significant effect on any primary outcome in either Exp2 or Exp3. Learning agents do not appear to internalise winner's curse logic. Their bidding behaviour is invariant to whether valuations are purely private or highly correlated.

### H3.2: Affiliation should change collusion dynamics through spontaneous coupling

- **Statement**: Value interdependence enables a "spontaneous coupling" mechanism where agents' strategies become correlated through the common-value component, facilitating tacit coordination.
- **Source**: Banchio & Mantegazza (2023)
- **Testable in**: Exp2. Factor: `eta_linear_coded` interacted with `auction_type_coded`. Metric: `avg_rev_last_1000`, `price_volatility`.
- **Expected outcome**: Higher eta should amplify collusion (lower revenue) if spontaneous coupling operates.
- **Verdict**: **Challenged.** The null eta effect directly contradicts the spontaneous coupling hypothesis. Whatever drives the first-price revenue penalty operates independently of valuation interdependence. Collusion dynamics are driven by auction format and algorithmic parameters, not by the value correlation structure.

### H3.3: Revenue ranking depends on affiliation strength

- **Statement**: Under independent private values, revenue equivalence holds. As affiliation increases, second-price auctions should generate strictly higher revenue than first-price auctions (linkage principle).
- **Source**: Milgrom & Weber (1982)
- **Testable in**: Exp2. Interaction: `auction_type_coded` x `eta_linear_coded`. Metric: `avg_rev_last_1000`.
- **Expected outcome**: The SPA revenue advantage should grow with eta.
- **Verdict**: **Challenged.** Revenue equivalence fails even at eta=0 (pure independent PV), and the SPA-FPA gap does not systematically vary with eta. The revenue ranking is determined by the learning algorithm's interaction with the auction format, not by affiliation strength.

### H3.4: Signal informativeness matters for bidding efficiency

- **Statement**: Richer information feedback (observing winning bids, own signals) should improve bidding efficiency by helping agents learn the competitive landscape.
- **Source**: Levin et al. (1996); Banchio & Skrzypacz (2022) (information feedback hypothesis)
- **Testable in**: Exp2 (`state_info_coded`), Exp3 (`context_richness_coded`). Metrics: `avg_rev_last_1000`, `time_to_converge`.
- **Expected outcome**: Richer state information should increase revenue and accelerate convergence.
- **Verdict**: **Partially supported.** In Exp2, the interaction between auction type and state information is the strongest effect on revenue ($p = 0.029$), though the state information main effect is not individually significant. This suggests that information availability mediates the auction format effect rather than operating independently. In Exp3, context richness contributes to outcome variation. The information channel matters but is secondary to auction format and algorithm choice.

---

## 4. Budget Constraints and Pacing

### H4.1: Gradient-based pacing converges to approximate Nash equilibrium

- **Statement**: Multiplicative dual pacing with step size O(1/sqrt(T)) converges to an approximate pacing equilibrium with vanishing regret.
- **Source**: Balseiro & Gur (2019)
- **Testable in**: Exp4 (multiplicative pacing). Metrics: `dual_cv`, `inter_episode_volatility`, `warm_start_benefit`.
- **Expected outcome**: Dual variables should stabilise across episodes; inter-episode volatility should decrease; warm-starting should provide measurable convergence benefits.
- **Verdict**: **Supported.** Exp4 agents converge rapidly (median ~1,014 rounds), dual variables stabilise, and warm-starting provides measurable benefits. The pacing algorithm achieves stable spending policies consistent with approximate equilibrium, though the equilibrium itself may be sub-competitive.

### H4.2: Liquid welfare is at least half of optimal without requiring convergence

- **Statement**: In auctions with budget-constrained bidders, the liquid welfare is at least 1/2 of the offline optimal, even without convergence to equilibrium.
- **Source**: Gaitonde et al. (2023)
- **Testable in**: Exp4. Metric: `mean_effective_poa` (ratio of offline optimum to realised liquid welfare).
- **Expected outcome**: Effective PoA should not exceed 2 (i.e., welfare should be at least half of optimal).
- **Verdict**: **Testable via Exp4.** The effective PoA varies by factor combination. The welfare guarantee framework provides the benchmark against which Exp4 efficiency results are evaluated.

### H4.3: Value-maximisers and utility-maximisers produce different equilibria

- **Statement**: Agents that maximise value (bid v/mu) behave fundamentally differently from agents that maximise utility (bid v/(1+mu)). Value-maximisers bid more aggressively and may achieve different welfare and revenue outcomes.
- **Source**: Balseiro et al. (2021)
- **Testable in**: Exp4. Factor: `objective_coded`. Metrics: `mean_platform_revenue`, `mean_effective_poa`, `mean_bid_to_value`, `mean_allocative_efficiency`.
- **Expected outcome**: Value-maximiser cells should show higher bid-to-value ratios and different efficiency patterns than utility-maximiser cells.
- **Verdict**: **Supported.** Bidder objective is a significant factor in Exp4. Value-maximisers and utility-maximisers produce detectably different bidding behaviour, revenue outcomes, and efficiency measures, confirming that the objective function specification meaningfully shapes equilibrium selection.

### H4.4: Budget constraints accelerate convergence

- **Statement**: Hard budget constraints impose spending discipline that forces agents to commit to stable strategies faster than unconstrained learners.
- **Source**: Implicit in Balseiro & Gur (2019); general pacing literature
- **Testable in**: Cross-experiment comparison of Exp4 (budget-constrained) vs. Exp1-3 (unconstrained). Metric: `time_to_converge`.
- **Expected outcome**: Exp4 convergence times should be shorter than Exp1-3.
- **Verdict**: **Supported.** Exp4 converges approximately as fast as the fastest unconstrained experiments (Exp1 and Exp3, median ~1,000 rounds) and dramatically faster than Q-learning with affiliated valuations (Exp2, median ~86,000 rounds). Budget constraints act as a convergence accelerator by limiting bid exploration once spending headroom is exhausted.

### H4.5: Complex non-equilibrium dynamics are possible under pacing

- **Statement**: Budget-constrained pacing can produce limit cycles, chaotic dynamics, or other complex non-equilibrium behaviour rather than converging to a fixed point.
- **Source**: Paes Leme et al. (2024)
- **Testable in**: Exp4. Metrics: `inter_episode_volatility`, `cross_episode_drift`, `dual_cv`.
- **Expected outcome**: Some parameter configurations should exhibit persistent oscillations or trending behaviour rather than convergence.
- **Verdict**: **Partially supported.** While most Exp4 configurations converge to stable equilibria (low inter-episode volatility), the cross-episode drift metric detects progressive bid suppression in some cells, consistent with slow non-convergent dynamics. The PID vs. multiplicative pacing comparison reveals that controller structure affects convergence stability.

### H4.6: PoA = 1 for FPA with value-maximisers but PoA = 2 for VCG/SPA

- **Statement**: In first-price auctions with value-maximising autobidders, every equilibrium achieves optimal welfare (PoA=1). In VCG/second-price auctions, the PoA can be as bad as 2.
- **Source**: Aggarwal et al. (2019); Deng et al. (2024)
- **Testable in**: Exp4. Factor interaction: `auction_type_coded` x `objective_coded`. Metric: `mean_effective_poa`.
- **Expected outcome**: FPA + value-max should achieve near-optimal welfare (PoA near 1); SPA + value-max should show larger efficiency losses.
- **Verdict**: **Testable via Exp4.** The interaction between auction format and bidder objective determines the PoA profile. These theoretical bounds provide the key benchmark for interpreting Exp4 welfare results.

---

## 5. Competition and Market Structure

### H5.1: More bidders produce more competitive outcomes

- **Statement**: Increasing the number of bidders raises competitive pressure, driving bids toward competitive levels and increasing seller revenue.
- **Source**: Standard IO theory; Ivaldi et al. (2002)
- **Testable in**: All experiments. Factor: `n_bidders_coded`. Metric: `avg_rev_last_1000`.
- **Expected outcome**: More bidders (n=4 or n=6 vs. n=2) should increase revenue.
- **Verdict**: **Partially supported (algorithm-dependent).** In Q-learning experiments (Exp1-2), more bidders moderate the FPA revenue penalty as expected. In Exp3 (LinUCB), the effect reverses: more bidders *worsen* FPA performance, as increased competition amplifies underbidding when agents use optimism-based exploration. In Exp4 (pacing), bidder count interacts with auction format. The competitive-pressure hypothesis holds for Q-learning but fails for contextual bandits.

### H5.2: Reserve prices anchor bids upward and reduce collusion

- **Statement**: Positive reserve prices establish a bidding floor that prevents extreme bid suppression, thereby increasing seller revenue.
- **Source**: Myerson (1981) (optimal reserve design)
- **Testable in**: Exp1, Exp3. Factor: `reserve_price_coded`. Metric: `avg_rev_last_1000`, `price_volatility`.
- **Expected outcome**: Positive reserves should increase revenue and reduce volatility.
- **Verdict**: **Partially supported.** Reserve prices reduce price volatility by disqualifying very low bids, stabilising outcomes. However, reserves show limited direct impact on revenue and cannot by themselves resolve first-price underperformance. Even with binding reserves, FPA yields lower revenue than SPA. Reserves alter the action space but do not change the reinforcement structure driving conservative bidding.

### H5.3: Information feedback regime affects collusion

- **Statement**: Richer information disclosure (revealing winning bids, own payoffs) enables agents to monitor and respond to competitors' strategies, which can either facilitate or hinder collusion depending on the mechanism.
- **Source**: Banchio & Skrzypacz (2022); Ivaldi et al. (2002) (transparency facilitates collusion in oligopoly theory)
- **Testable in**: Exp1 (`info_feedback_coded`), Exp2 (`state_info_coded`), Exp3 (`context_richness_coded`).
- **Expected outcome**: Richer information should alter revenue outcomes, with the direction depending on whether agents use information to coordinate or compete.
- **Verdict**: **Partially supported.** Information feedback has detectable effects across experiments, but primarily through interactions. In Exp2, the auction type $\times$ state information interaction is the strongest revenue effect ($p = 0.029$), suggesting that information availability modifies how auction format shapes outcomes. In Exp3, context richness has a smaller and less consistent effect. The information channel matters but is secondary to auction format and algorithm choice.

### H5.4: Algorithmic heterogeneity reduces collusion potential

- **Statement**: When different bidders use different learning algorithms, coordination becomes harder, reducing the scope for tacit collusion.
- **Source**: Implicit in Bichler et al. (2025); general literature
- **Testable in**: Not directly tested (all experiments use homogeneous algorithms within each run). However, Exp3's LinUCB vs. CTS comparison provides indirect evidence.
- **Expected outcome**: Mixed-algorithm settings should produce more competitive outcomes.
- **Verdict**: **Not directly testable.** The factorial design uses homogeneous agents in each run. The cross-algorithm comparison (Exp1 vs. Exp3) shows different collusion patterns by algorithm class, suggesting that heterogeneity would introduce asymmetry, but this remains untested.

---

## 6. Convergence and Dynamics

### H6.1: Q-learning can provably collude in auctions

- **Statement**: There exist parameter regimes under which Q-learning agents provably converge to sub-competitive equilibria in repeated auctions.
- **Source**: Bertrand et al. (2025)
- **Testable in**: Exp1, Exp2. Metrics: `avg_rev_last_1000` < competitive benchmark, `time_to_converge` < total episodes (indicating convergence to a stable sub-competitive state).
- **Expected outcome**: Some Q-learning configurations should reliably converge to low-revenue equilibria.
- **Verdict**: **Supported.** Q-learning agents systematically converge to sub-competitive revenue in FPA across the majority of parameter configurations in Exp1 and Exp2. The factorial design reveals which parameter regimes are most susceptible (asynchronous, epsilon-greedy, low n), providing empirical support for theoretical provability results.

### H6.2: Collusion without explicit punishment threats

- **Statement**: Algorithmic agents can sustain collusive outcomes without implementing punishment strategies (grim trigger, tit-for-tat); coordination emerges from the learning dynamics themselves rather than from credible threats.
- **Source**: Arunachaleswaran et al. (2025); Calvano et al. (2020)
- **Testable in**: All experiments. The experimental design uses independent, non-communicating agents with no punishment mechanism programmed into the learning rules.
- **Expected outcome**: Sub-competitive outcomes should emerge despite the absence of explicit punishment strategies.
- **Verdict**: **Supported.** All four experiments use agents that make independent decisions without communication or programmed punishment. Sub-competitive FPA outcomes emerge from the reinforcement learning dynamics alone. Agents do not implement tit-for-tat or grim trigger strategies; instead, bid suppression arises from the interaction of the learning rule with the auction mechanism.

### H6.3: Spontaneous coupling in common-value environments

- **Statement**: Under common values, agents' strategies become spontaneously coupled through the shared value component, facilitating coordinated bid suppression even without direct communication.
- **Source**: Banchio & Mantegazza (2023)
- **Testable in**: Exp2. Factor: `eta_linear_coded` (higher eta = stronger common-value component). Metrics: `avg_rev_last_1000`, `price_volatility`.
- **Expected outcome**: Higher eta should increase strategic coupling and amplify collusion (lower revenue, more coordinated bids).
- **Verdict**: **Challenged.** The null effect of eta directly contradicts the spontaneous coupling hypothesis. Whether valuations are purely private (eta=0) or highly correlated (eta=1), learning agents display indistinguishable bidding patterns. The coupling mechanism predicted by theory does not manifest empirically.

---

## 7. Methodological

### H7.1: Linear factorial models are sufficient for analysing algorithmic experiments

- **Statement**: A linear model with main effects and two-way interactions captures the systematic variance in simulation outcomes; nonparametric methods offer no meaningful improvement.
- **Source**: Experimental design methodology (Box, Hunter & Hunter 2005); validated empirically in this paper
- **Testable in**: All experiments. Diagnostic: OLS R^2 vs. LightGBM cross-validated R^2.
- **Expected outcome**: Gap between OLS R^2 and LightGBM R^2 should be less than 0.05.
- **Verdict**: **Supported (with caveats).** In Exp1-2 and Exp4, LightGBM R^2 falls below OLS R^2, confirming that the linear model is the correct specification. In Exp3, PRESS gaps are larger (0.14-0.18), indicating that bandit-based outcomes contain irreducible stochasticity that neither model class captures. The linear model remains a useful first-order approximation, but model adequacy is weaker for contextual bandits.

### H7.2: HC3 robust standard errors are needed for heteroskedastic data

- **Statement**: Heteroskedasticity-consistent standard errors (HC3) should be used to guard against non-constant error variance across the factorial design space.
- **Source**: MacKinnon & White (1985); standard econometric practice
- **Testable in**: All experiments. Diagnostic: HC3 flip rate (fraction of effects changing significance under HC3 vs. OLS SE).
- **Expected outcome**: A non-trivial fraction of effects should change significance if heteroskedasticity is present.
- **Verdict**: **Partially supported.** HC3 flip rates range from 2.6\% (Exp4) to 13.3\% (Exp2). The Exp2 rate reflects the small-sample design (48 observations, 150 terms tested), where borderline effects are more sensitive to SE corrections. The dominant effects in all experiments are invariant to the HC3 correction. HC3 provides a useful robustness check but does not materially change the primary conclusions.

### H7.3: Multiple testing corrections eliminate many apparently significant findings

- **Statement**: When testing dozens to hundreds of effects simultaneously, many OLS-significant effects will fail family-wise error rate control, revealing them as likely false positives.
- **Source**: Holm (1979); Benjamini & Hochberg (1995)
- **Testable in**: All experiments. Diagnostic: survival rate under Holm-Bonferroni and Benjamini-Hochberg corrections.
- **Expected outcome**: Substantial attrition of significant effects under stringent correction.
- **Verdict**: **Supported.** Holm-Bonferroni retains 19/50 (Exp1), 0/31 (Exp2), 14/27 (Exp3), and 50/62 (Exp4) of OLS-significant effects. Benjamini-Hochberg retains more (41/50, 2/31, 18/27, 57/62). The near-total elimination of Exp2 findings under correction reflects the low power of the 48-observation design rather than false discoveries. In Exp4, 81\% of findings survive Holm, reflecting the strong signal in the $2^3$ factorial with 15 replicates per cell. The attrition is concentrated among marginal effects near the significance boundary, validating the factorial approach while confirming the importance of multiple testing control.

---

## Summary Table

| # | Hypothesis | Source | Exp | Verdict |
|---|-----------|--------|-----|---------|
| H1.1 | FPA more prone to bid suppression | Banchio & Skrzypacz 2022 | 1-4 | Partially supported |
| H1.2 | Revenue equivalence under symmetric IPV | Myerson 1981 | 1-2 | Challenged |
| H1.3 | FPA unique eq., SPA multiple eq. | Conitzer et al. 2022 | 4 | Partially supported |
| H1.4 | FPA welfare >= 1/2 under no-regret | Fikioris & Tardos 2024 | 1, 3 | Partially supported |
| H1.5 | FPA PoA = 1/2 under autobidding | Deng et al. 2024 | 4 | Testable |
| H1.6 | SPA properties break under learning | Kolumbus & Nisan 2022 | 1-4 | Partially supported |
| H1.7 | Affiliation increases SPA advantage | Milgrom & Weber 1982 | 2-3 | Null |
| H2.1 | Sync -> competitive, async -> collusion | Asker et al. 2024 | 1 | Supported |
| H2.2 | Q-learning collusion is genuine | Calvano et al. 2023 | 1-2 | Supported |
| H2.3 | Sophisticated algorithms -> less collusion | Abada et al. 2024 | 1-3 | Challenged |
| H2.4 | Collusion is exception not rule | Bichler et al. 2025 | 1-4 | Partially supported |
| H2.5 | Mean-based learners -> Nash | Feng et al. 2021 | 3 | Challenged |
| H2.6 | Learning rate determines cooperation | Dolgopolov 2024 | 1 | Partially supported |
| H2.7 | Exploration mechanism > exploration rate | Cross-experiment | 1, 3 | Supported |
| H2.8 | Action-space granularity affects collusion | Klein 2021 | 1 | Null |
| H3.1 | Common values alter bidding via winner's curse | Milgrom & Weber 1982 | 2-3 | Null |
| H3.2 | Spontaneous coupling via affiliation | Banchio & Mantegazza 2023 | 2 | Challenged |
| H3.3 | Revenue ranking depends on affiliation | Milgrom & Weber 1982 | 2 | Challenged |
| H3.4 | Signal informativeness matters | Levin et al. 1996 | 2-3 | Partially supported |
| H4.1 | Gradient pacing -> approx. NE | Balseiro & Gur 2019 | 4 | Supported |
| H4.2 | Liquid welfare >= 1/2 optimal | Gaitonde et al. 2023 | 4 | Testable |
| H4.3 | Value-max vs. utility-max differ | Balseiro et al. 2021 | 4 | Supported |
| H4.4 | Budget constraints accelerate convergence | Pacing literature | 4 | Supported |
| H4.5 | Complex dynamics possible under pacing | Paes Leme et al. 2024 | 4 | Partially supported |
| H4.6 | FPA PoA=1 (value-max), SPA PoA=2 | Aggarwal et al. 2019 | 4 | Testable |
| H5.1 | More bidders -> more competitive | IO theory | 1-4 | Partially supported |
| H5.2 | Reserve prices anchor bids upward | Myerson 1981 | 1, 3 | Partially supported |
| H5.3 | Information feedback affects collusion | Banchio & Skrzypacz 2022 | 1-3 | Partially supported |
| H5.4 | Algorithmic heterogeneity reduces collusion | Bichler et al. 2025 | -- | Not testable |
| H6.1 | Q-learning can provably collude | Bertrand et al. 2025 | 1-2 | Supported |
| H6.2 | Collusion without punishment threats | Arunachaleswaran et al. 2025 | 1-4 | Supported |
| H6.3 | Spontaneous coupling mechanism | Banchio & Mantegazza 2023 | 2 | Challenged |
| H7.1 | Linear factorial models sufficient | Box et al. 2005 | 1-4 | Supported (with caveats) |
| H7.2 | HC3 robust SE needed | MacKinnon & White 1985 | 1-4 | Partially supported |
| H7.3 | Multiple testing eliminates many findings | Holm 1979 | 1-4 | Supported |

**Verdict distribution**: 10 Supported, 5 Challenged, 13 Partially supported, 3 Null, 3 Testable, 1 Not testable.
