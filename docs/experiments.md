# Experimental Overview

This project studies whether algorithmic bidding agents learn to collude in repeated sealed-bid auctions. Four experiments progressively increase environmental complexity, from constant valuations with Q-learning to budget-constrained autobidding with dual pacing. All experiments use factorial experimental designs with effects-coded factors (-1/+1) to enable orthogonal estimation of main effects and interactions via OLS.

## Quick Reference

| Exp | Algorithm | Valuations | Design | Factors | Cells | Key Question |
|-----|-----------|------------|--------|---------|-------|--------------|
| 1 | Q-learning | Constant (v=1) | 2^(11-1) Res V | 11 | 1024 | How do Q-learners bid under first- vs second-price with constant values? |
| 2 | Q-learning | Affiliated (eta) | 3 x 2^3 mixed | 4 | 24 | Does valuation interdependence alter collusion patterns? |
| 3 | LinUCB / CTS | Affiliated (eta) | 3 x 2^7 mixed | 8 | 384 | Do bandit algorithms converge faster or to different equilibria than Q-learning? |
| 4 | Dual Pacing | LogNormal (asymmetric) | 2^3 full | 3 | 8 x 50 seeds | How do budget constraints and bidder objectives interact with auction format? |

## Experiment 1: Identical Valuations

**Research question.** How do Q-learning agents learn to bid when valuations are constant, and how does the payment rule interact with algorithm hyperparameters?

**Valuation model.** All bidders have v_i = 1.0 every round.

**Algorithm.** Tabular Q-learning with epsilon-greedy or Boltzmann exploration. Agents choose bids from a discretized [0, 1] grid.

**Design.** 2^(11-1) Resolution V half-fraction (1,024 cells x 2 replicates = 2,048 runs).

| Factor | Low (-1) | High (+1) |
|--------|----------|-----------|
| Auction type | Second-price | First-price |
| Learning rate (alpha) | 0.01 | 0.30 |
| Discount factor (gamma) | 0.0 | 0.95 |
| Reserve price (r) | 0.0 | 0.5 |
| Initialisation | Zeros | Random |
| Exploration | Epsilon-greedy | Boltzmann |
| Synchronous | Asynchronous | Synchronous |
| Number of bidders (n) | 2 | 4 |
| Number of actions | 11 | 21 |
| Information feedback | Winner only | Winner + own payoff |
| Decay type | Linear | Exponential |

**Outcome metrics.** Average revenue (last 1000 rounds), time to converge, seller regret, no-sale rate, price volatility, winner entropy.

## Experiment 2: Affiliated Valuations

**Research question.** Does signal-based valuation interdependence change how Q-learners bid relative to the constant-valuation baseline?

**Valuation model.** Each bidder draws a private signal s_i in [0, 1] each round. Valuations are:

    v_i = (1 - 0.5 * eta) * s_i + 0.5 * eta * m_{-i}

where m_{-i} is the mean signal of opponents and eta in {0, 0.5, 1} controls affiliation strength. At eta=0, valuations are private; at eta=1, valuations weight own and others' signals equally.

**Algorithm.** Tabular Q-learning (same as Experiment 1).

**Design.** 3 x 2^3 mixed-level (24 cells x 2 replicates = 48 runs).

| Factor | Low (-1) | High (+1) |
|--------|----------|-----------|
| Auction type | Second-price | First-price |
| Number of bidders (n) | 2 | 6 |
| State information | None | Signal + previous winner |

| Factor | Levels |
|--------|--------|
| Affiliation (eta) | 0, 0.5, 1 (linear + quadratic contrasts) |

**Outcome metrics.** Average revenue, time to converge, no-sale rate, price volatility, winner entropy, excess regret, efficient regret, bid-to-value median, winner's curse frequency, bid dispersion.

**BNE benchmark.** Theoretical Bayesian Nash Equilibrium revenue computed analytically for each (auction type, n, eta) combination.

## Experiment 3: Bandit Approaches

**Research question.** Do contextual bandit algorithms, which use structured exploration, converge faster or reach different equilibria than Q-learning under the same affiliated valuations?

**Valuation model.** Same affiliated model as Experiment 2.

**Algorithm.** LinUCB (linear upper confidence bound) or CTS (Contextual Thompson Sampling). Each bidder treats each feasible bid as an arm and uses contextual features (own signal, optionally previous winning bid) to estimate payoffs.

**Design.** 3 x 2^7 mixed-level (384 cells x 2 replicates = 768 runs).

| Factor | Low (-1) | High (+1) |
|--------|----------|-----------|
| Algorithm | LinUCB | CTS |
| Auction type | Second-price | First-price |
| Number of bidders (n) | 2 | 6 |
| Reserve price (r) | 0.0 | 0.3 |
| Exploration intensity | Low | High |
| Context richness | Signal only | Signal + previous winner |
| Regularisation (lambda) | 0.1 | 10.0 |

| Factor | Levels |
|--------|--------|
| Affiliation (eta) | 0, 0.5, 1 (linear + quadratic contrasts) |

**Outcome metrics.** Average revenue, time to converge, seller regret, no-sale rate, price volatility, winner entropy.

## Experiment 4: Budget-Constrained Autobidding

**Research question.** How do budget constraints and bidder objectives interact with auction format in a pacing equilibrium setting?

**Valuation model.** Bidder-specific means drawn from [0.5, 1.5]; per-round valuations drawn from LogNormal(mean_i, sigma=0.3). Asymmetric across bidders.

**Algorithm.** Multiplicative dual pacing. Each bidder maintains a dual variable (pacing multiplier) updated episode-by-episode to satisfy budget constraints. Bids are shaded by the pacing multiplier: b_i = v_i / (1 + mu_i) for value-maximizers.

**Design.** 2^3 full factorial (8 cells x 50 seeds = 400 runs).

| Factor | Low (-1) | High (+1) |
|--------|----------|-----------|
| Auction type | Second-price | First-price |
| Bidder objective | Value-maximizer | Utility-maximizer |
| Number of bidders (n) | 2 | 4 |

**Outcome metrics.** Platform revenue, liquid welfare, effective price of anarchy, budget utilization, bid-to-value ratio, allocative efficiency, dual variable CV, no-sale rate, winner entropy, warm-start benefit.

## Common Outcome Metrics

These metrics are collected across experiments (with naming variations noted in CLAUDE.md):

| Metric | Definition |
|--------|------------|
| Average revenue | Mean auction revenue over the final 1,000 rounds |
| Time to converge | First round after which revenue stays within +/-5% of its final average |
| Seller regret | 1 - observed revenue, measuring shortfall from ideal |
| No-sale rate | Fraction of rounds where no bid meets the reserve price |
| Price volatility | Standard deviation of winning bids |
| Winner entropy | Entropy of the winner distribution (measures allocation concentration) |

Experiment 4 adds budget-specific metrics: budget utilization, liquid welfare, effective price of anarchy, and allocative efficiency.

## Cross-Experiment Progression

The four experiments form a deliberate progression in environmental complexity.

**Experiment 1** establishes the baseline: constant valuations eliminate strategic uncertainty about values, isolating the effect of algorithm hyperparameters and auction rules on learned bidding behaviour. With 11 factors, this is the broadest screening experiment.

**Experiment 2** introduces stochastic, interdependent valuations via the affiliation parameter eta. This tests whether Q-learners can still learn effective strategies when values are signal-based and opponents' signals matter. The design narrows to 4 factors, focusing on the new valuation structure.

**Experiment 3** holds the valuation model fixed but replaces Q-learning with contextual bandit algorithms (LinUCB, CTS). These algorithms use structured exploration via confidence bounds rather than epsilon-greedy, testing whether the exploration method affects equilibrium selection and convergence speed.

**Experiment 4** shifts the setting entirely: from repeated single-round auctions to budget-constrained autobidding over multi-round episodes. Agents use multiplicative dual pacing rather than Q-learning or bandits, introducing a budget constraint that creates inter-temporal strategic considerations absent from Experiments 1-3.
