# Extended Summary of Literature

This document contains detailed paragraph summaries of all papers in the project's transcription library, organized thematically. Each entry identifies the source file and provides a 5-8 sentence summary of the paper's contribution, methods, key findings, and relevance to auction design, algorithmic collusion, and autobidding.

---

## 1. Auction Theory and Equilibrium

### Myerson (1981) — "Optimal Auction Design"
**File:** `myerson_1981.md`

Myerson's seminal paper solves the problem of designing revenue-maximizing auctions for a seller who does not know buyers' valuations. Using the revelation principle, Myerson shows that attention can be restricted to direct revelation mechanisms where bidders report their value estimates, reducing an otherwise intractable design problem to a manageable optimization over allocation and payment functions subject to probability, individual-rationality, and incentive-compatibility constraints. The paper derives the optimal auction for a wide class of problems under a regularity condition on the value distributions, showing that the seller should allocate to the bidder with the highest "virtual valuation" (a distribution-dependent transformation of the reported value) provided it exceeds the seller's reserve, and should not sell at all otherwise. The optimal auction may involve reserve prices that exclude otherwise willing buyers and may not always sell to the highest bidder, both of which reduce efficiency but increase expected revenue. The framework accommodates both preference uncertainty (bidders differ in tastes) and quality uncertainty (bidders have private information about the object's value), through revision effect functions that capture how one bidder's information would cause others to revise their valuations. This paper provides the theoretical foundation for revenue-optimal auction design and the concept of virtual valuations that underpins much of the subsequent literature on mechanism design, reserve price setting, and the efficiency-revenue tradeoffs that are central to autobidding and platform auction design.

### Roughgarden, Syrgkanis, and Tardos (2017) — "The Price of Anarchy in Auctions"
**File:** `roughgarden_2017.md`

This survey develops a general and modular theoretical framework for proving worst-case efficiency guarantees (price of anarchy bounds) for equilibria of auctions in complex settings, complementing traditional economic analysis that focuses on exact optimal solutions in stylized environments. The framework rests on three user-friendly analytical tools: smoothness-type inequalities that yield approximation guarantees for many auction formats under complete information and deterministic strategies; extension theorems that generalize these guarantees to randomized strategies, no-regret learning outcomes, and incomplete-information settings; and composition theorems that extend single-item guarantees to multi-item auctions. A central result establishes that every Bayes-Nash equilibrium of a first-price single-item auction achieves social welfare at least 1 - 1/e (approximately 63%) of the maximum possible, regardless of the number of bidders, valuation distributions, or degree of asymmetry, and this bound holds even with correlated valuations. The paper demonstrates that non-trivial efficiency guarantees can be obtained without solving for or characterizing equilibria, relying only on the best-response property. This work is foundational for the algorithmic collusion literature because it provides the theoretical benchmark against which revenue and welfare losses from strategic algorithmic behavior can be measured, and because its no-regret learning extension directly addresses settings where bidders use adaptive algorithms.

### Kawasaki, Ariu, Abe, and Fujimoto (2024) — "Time-Varyingness in Auction Breaks Revenue Equivalence"
**File:** `kawasaki_2024.md`

This paper demonstrates that revenue equivalence between first-price and second-price auctions breaks down when bidders' value distributions vary over time, even when values remain symmetric, independent, and private within each period. The key mechanism is an asymmetry in how bidding behavior adapts to non-stationarity. In second-price auctions, truthful bidding is a dominant strategy regardless of distributional parameters, so bidders automatically track the equilibrium. In first-price auctions, the equilibrium bidding function depends on the value distribution's parameters, which bidders must learn from data; this learning introduces tracking error when the environment changes. The authors formalize this using gradient descent-ascent dynamics for parameter estimation and derive theoretical conditions under which revenue equivalence breaks for uniform value distributions: when the distribution's basis value and interval width correlate positively over time, bidders earn more in first-price auctions (and sellers earn more in second-price), and vice versa. Experiments with both uniform and log-normal distributions verify these theoretical predictions. This paper is directly relevant to auction design in algorithmic settings because it shows that the choice between first-price and second-price formats has revenue implications in non-stationary environments, which is precisely the setting in which adaptive bidding algorithms operate.

### Braverman, Mao, Schneider, and Weinberg (2018) — "Selling to a No-Regret Buyer"
**File:** `braverman_2018.md`

This paper studies optimal auction design when the buyer employs a no-regret learning algorithm rather than reasoning strategically about how current bids affect future seller behavior. The authors provide a nearly complete characterization of the seller's optimal revenue across three scenarios. First, if the buyer uses a "mean-based" algorithm such as EXP3 or Multiplicative Weights Update, the seller can extract revenue arbitrarily close to the full expected welfare (not just Myerson revenue) by designing an unnatural auction that lures the buyer into high bids early via free items, then charges high prices later. Second, the authors construct a specific no-regret algorithm against which the seller's best strategy is simply to post the Myerson reserve every round, demonstrating that the exploitability result is algorithm-dependent. Third, when the buyer uses a mean-based algorithm but the seller is restricted to "critical" auctions where overbidding is dominated (such as first-price or GSP formats), the optimal seller strategy involves a pay-your-bid format with decreasing reserves over time, and the achievable revenue is characterized exactly by a linear program. This revenue lies strictly between Myerson revenue and full welfare, and the gap can be unbounded in both directions. The paper is foundational for understanding the interaction between automated bidding and mechanism design, showing that the choice of learning algorithm has first-order effects on market outcomes and that standard auction theory benchmarks may not apply when buyers learn adaptively.

---

## 2. Mechanism Design and Optimal Auctions

### Akbarpour and Li (2020) — "Credible Auctions: A Trilemma"
**File:** `akbarpour_li_2020.md`

Published in Econometrica, this paper studies auction design when the auctioneer herself may deviate from the rules, provided no single bidder can detect the deviation through their private communication channel. A protocol is defined as "credible" if it is incentive-compatible for the auctioneer to follow the rules against all safe deviations. Under symmetric independent private values with regular distributions, the paper establishes a fundamental trilemma: an optimal auction can be any two of static, strategy-proof, or credible, but not all three simultaneously, and each pair uniquely characterizes a canonical auction format. The first-price auction with reserve is the unique credible static optimal mechanism (since the winner's payment is determined by their own bid, leaving no room for auctioneer manipulation), while the ascending auction with optimal reserve is the unique credible strategy-proof optimal mechanism (where chandelier bids are deterred by the risk of bidder exit). The second-price auction, despite being static and strategy-proof, fails the credibility test because the auctioneer can profitably exaggerate the second-highest bid. This work provides formal justification for the recent industry shift from second-price to first-price auctions in online advertising, grounding the transition in the auctioneer's incentive compatibility rather than just bidder-side considerations.

### Dutting, Feng, Narasimhan, Parkes, and Ravindranath (2024) — "Optimal Auctions through Deep Learning"
**File:** `dutting_2024.md`

This paper pioneers the use of deep learning for automated optimal auction design, framing the problem of finding revenue-maximizing incentive-compatible auctions as a constrained learning problem solvable with standard machine learning pipelines. Two neural network architectures are introduced: RochetNet, which leverages characterization results for DSIC mechanisms to constrain the search space via menu-based representations for single-bidder settings, and RegretNet, which replaces incentive compatibility constraints with a differentiable expected ex post regret penalty for multi-bidder multi-item settings, finding approximately DSIC mechanisms. The paper demonstrates that these architectures recover essentially all known optimal auction designs from 40+ years of theoretical work (including Myerson's auction), while also discovering novel mechanisms for settings where analytical solutions remain unknown. The approach is vastly more scalable than the traditional LP-based automated mechanism design formulation, which scales exponentially in bidders and items. Generalization bounds provide confidence intervals on expected revenue and regret based on network complexity and training sample size. The paper launched the research program of "differentiable economics," spawning extensive follow-up work on budget-constrained bidders, fairness-revenue tradeoffs, two-sided matching, and taxation policy design using neural networks.

### Bei, Lu, Wang, Xiao, and Yan (2025) — "Optimal Auction Design for Mixed Bidders"
**File:** `bei_2025.md`

This paper investigates revenue-maximizing auction design when the auctioneer faces a mixture of utility maximizers (UMs), who maximize quasi-linear surplus, and value maximizers (VMs), who maximize total allocated value subject to a return-on-spend constraint, and the bidder type is private information unknown to the auctioneer. The presence of both types creates a multi-parameter mechanism design problem where bidders can misreport both their value and their UM/VM type, substantially complicating the design of incentive-compatible mechanisms compared to standard single-parameter settings. For a single bidder, the authors characterize the optimal auction structure and show that it smoothly interpolates between a first-price auction (optimal when the bidder is certainly a VM) and a Myerson auction (optimal when the bidder is certainly a UM) as the probability of the bidder being a UM varies. The characterization reveals that the optimal payment rule for VMs is always first-price (the winner pays their full bid), while UMs receive Myerson's payment, and the VM allocation rule is tightly linked to the UM allocation through a connecting function. For multiple bidders, the authors provide an algorithm for computing the optimal lookahead auction. This work is relevant to practical auction platform design because real-world advertising exchanges observe heterogeneous bidder types, and designing a single mechanism that handles both without type information is essential.

### Golrezaei, Lobel, and Paes Leme (2021) — "Auction Design for ROI-Constrained Buyers"
**File:** `golrezaei_2021.md`

This paper combines theory and empirics to demonstrate that many buyers in online advertising markets are financially constrained and behave as if they have minimum return-on-investment (ROI) requirements, then designs optimal auctions accounting for these constraints. Using data from a field experiment on Google's advertising exchange (AdX) where reserve prices were randomized, the authors find that a significant fraction of buyers lower their bids when reserve prices increase, contradicting the predictions of standard auction theory for quasilinear bidders. This behavior is consistent with ROI-constrained buyers who shade their bids in second-price auctions to maintain a minimum ROI target, winning fewer auctions but earning greater surplus on those they win. The paper then derives optimal auction mechanisms for ROI-constrained buyers: for symmetric buyers, the optimal auction is either a second-price auction with reduced reserve prices or a subsidized second-price auction; for asymmetric buyers, it involves a modification of Myerson's virtual values. Returning to the AdX data, the authors show that using ROI-aware optimal auctions can yield large revenue gains for the platform and large welfare gains for buyers. This work is highly relevant to autobidding systems where budget and ROI constraints fundamentally alter bidder behavior and optimal mechanism design.

### Mehta (2022) — "Auction Design in an Auto-bidding Setting: Randomization Improves Efficiency Beyond VCG"
**File:** `mehta_2022.md`

Mehta studies auction design for the auto-bidding setting where advertisers specify target cost-per-acquisition constraints and autobidding agents optimize bids on their behalf. The paper's central contribution is proving that randomization can improve welfare beyond what VCG achieves, even in a prior-free setting without additional value information. Specifically, the Rand(alpha, p) auction, which allocates deterministically to the higher bidder only when bids differ by more than a factor of alpha and otherwise randomizes with probability p favoring the higher bidder, achieves a PoA of approximately 1.896 for two bidders. This strictly improves upon the PoA of 2 that is tight for VCG. The result is surprising because in the standard quasi-linear utility setting, VCG is welfare-optimal, and improving over VCG revenue typically requires Bayesian or prior-independent assumptions. The paper also proves a sharp impossibility result: as the number of bidders per query grows, no randomized anonymous truthful auction can achieve a PoA strictly better than 2, closing the problem for prior-free auctions.

### Deng, Mao, Mirrokni, and Zuo (2021) — "Towards Efficient Auctions in an Auto-bidding World"
**File:** `deng_2021.md`

This paper proposes a family of auctions with additive boosts to improve welfare efficiency in auto-bidding environments where advertisers specify high-level objectives (target ROAS or CPA constraints) and delegate bidding to automated agents using uniform bidding strategies. The central innovation is the concept of "c-value-competitive boosts," where the platform adds bidder-specific boosts to bids that scale with the difference in bidders' values, effectively steering the allocation toward higher-value bidders. The authors prove that VCG auctions with c-value-competitive boosts achieve a (c+1)/(c+2) approximation to optimal welfare, approaching full efficiency as c grows, though with a revenue tradeoff since boosts are deducted from payments. Importantly, the welfare guarantees hold without requiring bidders to be in Nash equilibrium, only requiring feasible and undominated strategies. The paper extends these results to settings with both ROAS and budget constraints through "benchmark-competitive boosts" and shows applicability to GSP auctions under uniform bidding. Empirical results validate that properly selected boost weights can simultaneously improve both welfare and revenue, providing a practical mechanism design lever for platforms transitioning to auto-bidding.

### Deng, Mao, Mirrokni, Zhang, and Zuo (2022) — "Efficiency of the First-Price Auction in the Autobidding World"
**File:** `deng_2022_fpa.md`

This paper precisely characterizes the price of anarchy (PoA) of first-price auctions across three bidding environments: full autobidding (all value maximizers), mixed autobidding (both value maximizers and utility maximizers), and no autobidding (all utility maximizers). In the full autobidding world, the PoA is exactly 1/2, matching the second-price auction result; but in the mixed world with both bidder types, the PoA degrades to approximately 0.457, which is strictly worse than second-price auctions' 1/2 bound. The analysis is technically challenging because first-price auctions are not truthful, meaning value maximizers may use non-uniform and even randomized bidding strategies, unlike in second-price settings. The proof introduces a novel "local analysis" technique that characterizes the tradeoff between the value rightful winners receive and the payments other bidders make in equilibrium, overcoming the difficulty that utility maximizers may sacrifice value for better utility. The authors further show that machine-learned advice about bidder values can be used to set reserves that smoothly improve the PoA from 0.457 toward 1 as advice accuracy increases, paralleling a similar result for second-price auctions.

### Liaw, Mehta, and Perlroth (2022) — "Efficiency of Non-Truthful Auctions under Auto-bidding"
**File:** `liaw_2022.md`

Liaw, Mehta, and Perlroth study the price of anarchy (PoA) of non-truthful auctions in the prior-free auto-bidding setting with return-on-spend constraints. Their first result is that non-truthfulness provides no benefit for deterministic auctions: any deterministic mechanism has PoA at least 2, even for two bidders, matching what truthful mechanisms (like SPA) achieve. They prove the first-price auction has PoA of exactly 2. Their second, more surprising result is constructive: a randomized non-truthful auction (randomized FPA, or rFPA) achieves a PoA of 1.8 for two bidders, the best known bound for this problem, strictly improving over the previous best of 1.9 achieved by truthful mechanisms. The rFPA auction works by randomizing the allocation when the two bids are close (within a factor alpha), and deterministically allocating to the higher bidder otherwise, with the winner always paying their bid. The paper also proves an impossibility result: no auction (randomized or non-truthful) can improve upon a PoA of 2 as the number of bidders grows to infinity.

### Liaw, Mehta, and Zhu (2024) — "Efficiency of Non-Truthful Auctions in Auto-bidding with Budget Constraints"
**File:** `liaw_2024.md`

Liaw, Mehta, and Zhu extend the analysis of non-truthful auctions to auto-bidders with both return-on-spend and budget constraints. The paper's first main result is that the first-price auction is optimal among deterministic mechanisms in this setting, but the PoA degrades from 2 (with only ROS constraints) to n (the number of bidders) due to the large gap between optimal randomized and deterministic allocations. Under the mild assumption that no bidder's value for any query exceeds their budget, the PoA recovers to 2. Two randomized mechanisms bypass the deterministic lower bounds: rFPA achieves PoA of 1.8 for two bidders without any assumptions, and a "quasi-proportional" FPA (allocating proportional to a power of the bids) achieves PoA of 2 for any number of bidders. A counterintuitive finding is that uniform bidding, which is optimal for FPA without budgets, becomes detrimental under budget constraints, making the integral PoA as bad as n. In contrast, uniform bidding improves the efficiency of rFPA, reducing PoA to 1.5 for two bidders.

### Colini-Baldeschi et al. (2025) — "Optimal Type-Dependent Liquid Welfare Guarantees for Autobidding Agents with Budgets"
**File:** `colini_baldeschi_2025.md`

This paper analyzes the efficiency of simultaneous first-price auctions (FPA) when autobidding agents are heterogeneous, operating anywhere along a spectrum between value maximizers (who maximize outcome value subject to ROI and budget constraints) and utility maximizers (who maximize value minus payment). The authors derive optimal type-dependent liquid welfare price of anarchy (PoA) bounds for FPA, where "liquid welfare" discounts an agent's value by their budget when the budget is binding, providing a more realistic efficiency benchmark than standard welfare for budget-constrained settings. The main technical contribution is showing that the PoA depends on the composition of agent types in the market and establishing tight bounds that vary continuously along the value-maximizer to utility-maximizer spectrum. For pure value maximizers, the PoA can be as bad as 1/2, while for pure utility maximizers the bound is tighter, and mixed populations interpolate between these extremes. The results provide practical guidance for auction platform designers by identifying which combinations of autobidder types lead to the greatest efficiency losses.

---

## 3. Algorithmic Collusion — Simulation Studies

### Calvano, Calzolari, Denicolo, and Pastorello (2020) — "Artificial Intelligence, Algorithmic Pricing, and Collusion"
**File:** `calvano_2020.md`

This paper is the seminal study of algorithmic collusion in pricing games, demonstrating that Q-learning algorithms can autonomously learn to sustain supra-competitive prices in a repeated Bertrand oligopoly without being explicitly programmed to collude. The authors construct Q-learning agents that interact in an infinitely repeated pricing game, conditioning their actions on the history of own and rival past prices, and run 1,000 sessions per parameter configuration until convergence. The key finding is that Q-learning algorithms consistently converge to prices substantially above the static Nash equilibrium (though somewhat below the monopoly level), and these prices are sustained by punishment-and-reward strategies: when one agent deviates to a lower price, the rival retaliates with a finite phase of punishment followed by a gradual return to cooperative pricing. This punishment structure resembles the collusive equilibria studied in the theoretical literature on repeated games, yet it emerges purely from the learning dynamics without any design intent. The results are robust across variations in the number of firms, product differentiation, and discount factors. The paper has been highly influential in regulatory discussions (OECD, FTC, European competition authorities) and has shaped the debate about whether antitrust enforcement should extend to algorithmic pricing. However, subsequent work has questioned the generality of these findings, noting that the results are sensitive to the specific learning algorithm used and may not extend to no-regret algorithms or incomplete-information settings.

### Banchio and Skrzypacz (2022) — "Artificial Intelligence and Auction Design"
**File:** `banchio_skrzypacz_2022.md`

This paper studies how auction format affects outcomes when bidders use Q-learning algorithms rather than playing Nash equilibrium strategies, motivated by the practical shift in online advertising from second-price to first-price auctions. In a simple setting with two bidders of known constant values competing over a discrete bid grid, the authors find a stark dichotomy: Q-learning agents in second-price auctions converge to the competitive Nash equilibrium prediction, while agents in first-price auctions converge to tacitly collusive bids far below their values (average bid of 0.24 versus a value of 1.0). Through systematic experimentation, the authors isolate the driving mechanism: in first-price auctions, a winning bidder has incentives to win by just one bid increment, which facilitates re-coordination on low bids after phases of experimentation. The paper's second key finding is that providing additional feedback, specifically revealing the minimum winning bid (as Google implemented when switching to first-price auctions), enables "synchronous updating" of Q-values for counterfactual bids, which eliminates the collusive dynamics and restores competitive outcomes. The paper also examines the effects of reserve prices and competitive fringes, finding that while more competition reduces the severity of collusion, it does not eliminate it entirely.

### Banchio and Mantegazza (2022) — "Artificial Intelligence and Spontaneous Collusion"
**File:** `banchio_mantegazza_2022.md`

This paper develops a tractable analytical framework for understanding when and why learning algorithms collude, identifying a novel mechanism the authors call "spontaneous coupling." Using fluid approximations based on Kurtz's theorem, the authors convert the stochastic discrete-time dynamics of interacting reinforcement learning algorithms into deterministic continuous-time dynamical systems that can be analyzed with standard ODE tools. The central insight is that algorithms with non-uniform learning rates develop endogenous statistical correlations in their payoff estimates: because actions played infrequently retain stale estimates, agents tend to synchronize their play, periodically cycling through cooperative and competitive phases rather than converging to static Nash equilibria. The paper proves that this spontaneous coupling can sustain collusion in both pricing (Bertrand competition) and market division (auction keyword splitting) settings without requiring intentional collusion, monitoring, or punishment strategies. Crucially, the authors characterize a class of algorithms immune to spontaneous coupling: those with uniform learning rates, which update all actions at the same speed, learn to play only undominated strategies. The paper also applies these results to mechanism design, showing that platforms can eliminate spontaneous coupling by providing sufficient feedback (such as counterfactual information) to assist algorithmic learning.

### Bertrand, Duque, Calvano, and Gidel (2025) — "Self-Play Q-Learners Can Provably Collude in the Iterated Prisoner's Dilemma"
**File:** `bertrand_2025.md`

This paper provides the first rigorous theoretical proof that self-play Q-learning agents converge to cooperative (collusive) behavior in the iterated prisoner's dilemma, offering formal foundations for the computational findings in the algorithmic collusion literature. The authors characterize the dynamics of epsilon-greedy Q-learning with one-step memory, where each player's action is conditioned on the previous joint action, and show that multiple fixed-point policies exist for the self-play multi-agent Bellman equation: always defect, Pavlov (win-stay, lose-shift), and grim trigger. The main result proves that with "optimistic enough" initializations (Q-values set higher than optimal, a standard practice in reinforcement learning), the learning dynamics follow a specific trajectory: starting from the always-defect policy, transitioning through a lose-shift policy, and ultimately converging to the cooperative Pavlov policy. The convergence to Pavlov rather than always-defect depends critically on the discount factor being sufficiently large and the learning rate being sufficiently small, providing precise conditions under which cooperation (and by extension, collusion in economic settings) emerges. The theoretical results are validated experimentally and shown to be robust across a broader class of deep learning algorithms.

### Deng, Schiffer, and Bichler (2024) — "Algorithmic Collusion in Dynamic Pricing with Deep Reinforcement Learning"
**File:** `deng_schiffer_bichler_2024.md`

This paper provides a comprehensive numerical study of algorithmic collusion across multiple reinforcement learning algorithms and Bertrand oligopoly variants, going beyond the narrow focus on Tabular Q-learning (TQL) that dominates the existing literature. The authors compare TQL, Deep Q-Networks (DQN), Proximal Policy Optimization (PPO), and Soft Actor-Critic (SAC) across standard Bertrand competition, Bertrand-Edgeworth competition with capacity constraints, and Bertrand competition with logit demand. A central finding is that collusive outcomes depend strongly on both the algorithm and the market environment: TQL consistently exhibits higher collusion and price dispersion artifacts across all market models, while DRL algorithms show more varied behavior. PPO in particular appears less susceptible to collusive outcomes, which the authors attribute to its on-policy nature providing implicit exploration during learning. The paper identifies "dispersion effects," where two algorithms converge at different prices, as artifacts of poor exploration in off-policy methods rather than genuine market outcomes. These results challenge the generalizability of prior findings based solely on TQL.

### Conjeaud (2024) — "Algorithmic Collusion under Competitive Design"
**File:** `conjeaud_2024.md`

This paper studies algorithmic collusion when the algorithms themselves are designed strategically, formalizing a "designing game" in which two players simultaneously choose the exploration parameter of their epsilon-greedy Q-learning algorithms before deploying them in a repeated prisoner's dilemma. Building on the "spontaneous coupling" framework of Banchio and Mantegazza (2023), the key theoretical contribution is proving that every Nash equilibrium of this designing game must feature some cooperative (collusive) behavior with positive probability. The paper analytically characterizes extreme cases: when one algorithm always explores (epsilon=1), spontaneous coupling disappears and defection dominates, while when both use epsilon=0 (fully greedy), the profile constitutes a Nash equilibrium where spontaneous coupling remains possible. Extensive numerical simulations reveal that equilibria of the designing game are generally symmetric and located on a bell-shaped curve with respect to a parameter controlling the value of cooperation, and that moderate exploration levels sustain collusion through alternating cooperative and non-cooperative phases.

### Kasberger, Martin, Normann, and Werner (2024) — "Algorithmic Cooperation"
**File:** `kasberger_2024.md`

Kasberger et al. systematically investigate how Q-learning algorithms play the infinitely repeated Prisoner's Dilemma, comparing their behavior to humans using the same experimental frameworks and analytical tools from the behavioral economics literature. The study varies three main treatment dimensions: the reward from mutual cooperation, the discount factor (continuation probability), and the algorithm's memory length (one, two, or three periods). A key methodological contribution is applying the Strategy Frequency Estimation Method (SFEM), developed for analyzing human experimental data, to classify algorithms' strategies into interpretable types (tit-for-tat, grim trigger, win-stay-lose-shift, etc.). The authors find that the same factors increasing human cooperation (higher discount factors, higher cooperation rewards, grim trigger being a subgame perfect equilibrium) also drive algorithmic cooperation. However, significant differences emerge: algorithms favor win-stay-lose-shift over grim trigger (which humans prefer), and algorithms fail to cooperate in environments where cooperation is very risky or not incentive compatible. An extension using ChatGPT shows that the determinants of cooperation for LLMs differ from those for Q-learning agents.

### Fish, Gonczarowski, and Shorrer (2024) — "Algorithmic Collusion by Large Language Models"
**File:** `fish_2024.md`

This paper provides experimental evidence that Large Language Model (LLM) based pricing agents autonomously reach supra-competitive prices and profits in oligopoly settings, without any explicit instructions to collude. Using a repeated Bertrand oligopoly environment from the Calvano et al. (2020) framework with logit demand, the authors deploy GPT-4-based agents instructed only to maximize long-term profit, finding that they quickly and consistently arrive at prices significantly above the Nash equilibrium, to the detriment of consumers. A particularly striking finding is that seemingly innocuous variations in prompt wording systematically influence the degree of supra-competitive pricing; instructions that reiterate long-term profit focus lead to near-monopoly profits, while instructions mentioning the possibility of gaining market share through lower prices reduce (but do not eliminate) collusion. Using a novel causal text analysis technique, the authors uncover that LLM agents avoid price reductions due to "fear" of triggering price wars, and on-path behavior is consistent with reward-punishment schemes where the intensity decays over several periods. These results extend to first-price auction settings and persist across multiple frontier LLMs.

### Han (2021) — "Algorithmic Pricing with Independent Learners and Relative Experience Replay"
**File:** `han_2021.md`

This paper investigates algorithmic collusion in a repeated Bertrand oligopoly by integrating relative performance (RP) evaluation with experience replay (ER) techniques in a multi-agent reinforcement learning framework. Building on the finding by Calvano et al. (2020) that independent tabular Q-learning produces supra-competitive prices, the author addresses two limitations of that work: the overfitting problem where agents trained in separate instances fail to sustain cooperation when paired together, and the neglect of agents' heterogeneous attitudes toward relative performance. The paper introduces a "relative ER" framework in which agents sample experience tuples from a replay buffer using a softmax distribution whose inverse temperature parameter (the RP coefficient) controls how much agents care about outperforming competitors. Results show that agents with positive RP coefficients (tolerant of underperformance) converge to supra-competitive prices, while agents with negative coefficients (averse to underperformance) converge to the Bertrand-Nash equilibrium.

### Zhang (2023) — "Pricing via Artificial Intelligence: The Impact of Neural Network Architecture on Algorithmic Collusion"
**File:** `zhang_weipeng_2023.md`

This paper by Weipeng Zhang (Georgetown University) investigates how the architecture of deep reinforcement learning algorithms affects the propensity for algorithmic collusion in repeated Bertrand pricing games. Contrary to the view that more complex algorithms would amplify collusion risk, Zhang shows that deep RL algorithms (Deep Q-Networks, Deep Recurrent Q-Networks with LSTM) may actually foster more competition than naive tabular Q-learning. The key mechanism is that experience replay and other training techniques required by deep networks break the temporal correlation between consecutive states and actions, which is precisely what allows simple Q-learning to maintain collusive reward-punishment strategies. Without that temporal correlation, the more realistic algorithms cannot sustain collusion, as the non-stationarity of multi-agent learning processes prevents convergence to cooperative equilibria. The paper adapts several widely used neural network architectures to a model-free RL framework and experiments across the standard Calvano et al. oligopoly model. The findings suggest that algorithmic collusion may not be an immediate concern for antitrust authorities, since the training processes required by practical deep learning algorithms inherently destabilize the collusive dynamics that emerge under simpler tabular methods.

### Zhang (2025) — "Too Noisy to Collude? Algorithmic Collusion Under Laplacian Noise"
**File:** `zhang_2025.md`

This paper proposes controlling information quality as an ex ante regulatory lever against algorithmic collusion, demonstrating that injecting calibrated Laplacian noise into the pooled market data that pricing algorithms consume can disrupt cartel formation before supracompetitive prices emerge. The model considers a duopoly with linear demand where firms set prices based on estimates of their rival's baseline demand, with pricing algorithms acting as intermediaries that aggregate data across sellers to produce these estimates. The analysis proceeds in three parts: first, quantifying how imperfect information distorts equilibrium prices, profits, and consumer welfare; second, modeling cartel dynamics as a DeGroot social learning process where forceful agents push consensus toward higher prices; and third, deriving a feasibility region for noise injection that disrupts collusion without undermining legitimate firms' pricing or consumer welfare. Key results include formal bounds on how much noise is needed to slow cartel convergence, a robustness guarantee that noise reliably destabilizes coordination, and the surprising finding that under collusion, the leader firm's own customers may actually benefit when rival prices rise.

### Arunachaleswaran, Collina, Kannan, Roth, and Ziani (2025) — "Algorithmic Collusion Without Threats"
**File:** `arunachaleswaran_2025.md`

This paper challenges the prevailing economic view that supra-competitive prices require either explicit threats or a failure to optimize. The authors study sequential Bertrand pricing games where a "learner" first commits to a no-regret algorithm and an "optimizer" subsequently deploys any pricing strategy, showing that near-monopoly prices robustly emerge even when neither party encodes threats and both are optimizing. Specifically, the optimizer can extract a constant fraction of monopoly revenue against any no-regret learner by using a non-responsive (fixed distribution) pricing strategy that cannot by construction encode threats, and simultaneously the learner also earns a constant fraction of monopoly revenue through their no-regret guarantee. The paper further demonstrates that within the class of no-swap-regret algorithms for the learner and non-responsive strategies for the optimizer, there exists a Nash equilibrium in algorithm space that supports near-monopoly prices with neither party having incentive to deviate. The finding that the mere act of committing to an algorithm creates the conditions for collusion substantially broadens the definition of algorithmic collusion and suggests that regulatory approaches focused solely on detecting threats may be insufficient.

### Lambin (2024) — "Less Than Meets the Eye: Simultaneous Experiments as a Source of Algorithmic Seeming Collusion"
**File:** `lambin_2024.md`

Lambin offers a skeptical reassessment of the algorithmic collusion literature, arguing that what appears to be collusion in prior studies may instead be an artifact of simultaneous experimentation during the learning phase. The paper replicates the seminal setup of Calvano et al. (AER 2020) and shows that the supracompetitive pricing outcomes rest on critical assumptions that are unlikely to hold in practice: extensive simultaneous exploration in early phases that creates upward-biased beliefs, and insufficient experimentation during the operational phase that prevents agents from discovering profitable deviations. The paper demonstrates that simple extensions of the original settings can make the collusive appearance disappear, and provides methodological critiques of how claims about punishment-reward strategies and tit-for-tat mechanisms have been interpreted as evidence of collusion. This work is part of a broader debate in the literature involving Asker et al. (2023), den Boer et al. (2023), and Abada and Lambin (2023), which collectively question whether the convergence of Q-learning algorithms to supracompetitive prices constitutes genuine collusion or is better understood as an artifact of the learning process.

### Lamba and Zhuk (2022) — "Pricing with Algorithms"
**File:** `lamba_zhuk_2022.md`

Lamba and Zhuk study Markov perfect equilibria in a repeated duopoly model where sellers choose algorithms, defined as mappings from the competitor's price to own price. The model captures two key features of algorithmic pricing: fast response times and temporary commitment to a pricing rule. Their main result is that collusion is not merely possible but inevitable. In the simple two-price case with standard demand specifications, monopoly pricing is the unique equilibrium outcome. In the general multi-price setting, all Markov perfect equilibria yield payoffs above the competitive outcome for both sellers, and at least one seller's payoff is bounded close to monopoly profit. The intuition is that the competitive price cannot survive the perfectness criterion: a seller facing a competitor playing the competitive strategy can profitably switch to tit-for-tat, inducing the competitor to eventually do the same, converging to monopoly pricing. The findings raise significant antitrust concerns because the collusion requires no direct communication between firms.

### Kolumbus and Nisan (2022) — "Auctions between Regret-Minimizing Agents"
**File:** `kolumbus_nisan_2022.md`

This paper analyzes repeated auctions in which software agents, implemented as regret-minimizing algorithms, bid on behalf of their users. The authors study both first-price and second-price auctions, as well as their generalized versions used in ad auctions. Their central finding is a striking asymmetry between auction formats with respect to incentive compatibility at the user-agent interface. In second-price auctions, users have incentives to misreport their true valuations to their own learning agents, whereas in first-price auctions it is a dominant strategy for all players to report truthfully. This result has direct practical implications for online advertising platforms, where bidding agents are commonplace and the choice of auction format determines whether advertisers can profitably manipulate their own autobidders. The paper combines theoretical analysis with simulations, modeling agents that use algorithms such as multiplicative weights, online gradient descent, and follow-the-perturbed-leader.

---

## 4. Algorithmic Collusion — Surveys and Theory

### Bichler and Durmann (2025) — "Algorithmic Pricing and Algorithmic Collusion"
**File:** `bichler_survey_2025.md`

This survey article provides a comprehensive overview of the algorithmic collusion literature, situating it at the intersection of online learning in computer science and equilibrium learning in game theory. The authors begin by noting that algorithmic pricing is now widespread in online retail, with algorithms setting prices for roughly a third of top Amazon products by 2015, and they examine whether autonomous pricing agents can learn to sustain supra-competitive prices without explicit communication. The paper reviews the foundational experimental findings, particularly the work of Calvano et al. (2020) showing that Q-learning agents can learn collusive pricing in Bertrand oligopoly, while also presenting the theoretical counterarguments, including results showing that mean-based no-regret algorithms converge to Nash equilibrium in many settings. Bichler and Durmann highlight a critical distinction between results in complete-information models (where collusion findings are algorithm-specific and fragile) and incomplete-information Bayesian models (where convergence to equilibrium appears more robust). The survey identifies specific research opportunities and is particularly useful as a roadmap of the field, connecting disparate strands of literature and clarifying the conditions under which algorithmic collusion is a genuine concern versus an artifact of specific experimental setups.

### Aggarwal et al. (2024) — "Auto-bidding and Auctions in Online Advertising: A Survey"
**File:** `aggarwal_survey_2024.md`

This comprehensive survey by a large team at Google Research covers the rapidly growing literature on automated bidding in online advertising auctions, organized around three pillars: bidding algorithms, equilibrium analysis and efficiency of auction formats, and optimal auction design. The paper formalizes the autobidding agent's problem as maximizing a hybrid objective subject to budget and return-on-spend (RoS) constraints, and presents LP-based optimal bidding formulas for truthful auctions that reduce to uniform pacing strategies. For online learning, the survey covers dual-gradient-descent algorithms achieving O(sqrt(T)) regret in truthful auctions and the substantially harder problem of non-truthful (first-price) auctions where optimal bidding is a nonlinear function of values. On the equilibrium side, the paper discusses existence and uniqueness results for pacing equilibria, price-of-anarchy bounds, and convergence of learning dynamics. The survey also covers optimal mechanism design in the autobidding world, where the classical Myerson results break down because bidders are intermediaries with constraints rather than direct utility maximizers.

### Marty and Warin (2025) — "Algorithmic Collusion and Bandit Algorithms"
**File:** `marty_warin_2025.md`

Marty and Warin examine algorithmic collusion from both legal and economic perspectives, focusing specifically on bandit algorithms (as distinct from Q-learning and deep reinforcement learning). The paper argues that bandit algorithms can converge faster to supracompetitive price equilibria than Q-learning algorithms, partially addressing the criticism that algorithmic collusion via self-reinforcing learning is too slow to be practically relevant. Beyond speed, the authors highlight a second mechanism: bandit algorithms' exploration-exploitation patterns can function as implicit price signals, making a firm's pricing strategy more decipherable to competitors without explicit communication. This framing connects algorithmic collusion to the antitrust concept of facilitating practices and unilateral collusive contract offers. The paper compares bandit, reinforcement learning, and deep learning models along dimensions of convergence speed, susceptibility to collusion, and regulatory detectability, noting that bandit algorithms are paradoxically both more prone to tacit collusion and more transparent to regulators than deep learning approaches.

### Brown and MacKay (2021) — "Competition in Pricing Algorithms"
**File:** `brown_mackay_2021.md`

This paper documents new empirical facts about algorithmic pricing using high-frequency data from major online retailers and develops a theoretical model explaining how pricing algorithms can raise prices through competitive (non-collusive) mechanisms. Using hourly price data for over-the-counter allergy medications across five large U.S. retailers, the authors establish three stylized facts: firms differ in pricing frequency (some update hourly, others weekly), faster firms rapidly respond to price changes by slower rivals consistent with automated strategies, and asymmetric pricing technology is associated with persistent price differences of 10-30 percent across retailers. The key insight is that even simple linear algorithms (price as a function of rivals' prices) can support supra-competitive prices in Markov perfect equilibrium without any collusive punishment schemes, because the faster firm's credible commitment to best-respond softens competition. Calibrated counterfactual simulations suggest algorithmic competition increases average prices by 5 percent and profits by 10 percent relative to simultaneous Bertrand, with mergers generating larger price increases under algorithmic competition.

---

## 5. Autobidding and Pacing

### Balseiro and Gur (2019) — "Learning in Repeated Auctions with Budgets: Regret Minimization and Equilibrium"
**File:** `balseiro_gur_2019.md`

This paper introduces adaptive pacing strategies for budget-constrained advertisers competing in repeated second-price auctions under incomplete information, where bidders know neither their own value distributions nor their competitors' budgets or strategies. The strategy adjusts a single dual multiplier based on observed expenditures, shading bids as b = v/(1+mu) to pace budget depletion, and updates this multiplier via a primal-dual scheme. The authors establish three key performance results: under stationary competition (i.i.d. competing bids), the strategy converges to the best performance achievable in hindsight; under arbitrary (adversarial) competition, the strategy is asymptotically optimal, achieving the best possible competitive ratio of v_bar/rho; and when all bidders adopt adaptive pacing simultaneously, the resulting dynamics converge and constitute an approximate Nash equilibrium in large markets with many auctions. The paper also provides a fundamental impossibility result showing that no strategy can achieve "no-regret" relative to dynamic benchmarks when budgets bind, making the competitive ratio bound tight.

### Balseiro, Deng, Mao, Mirrokni, and Zuo (2021) — "Robust Auction Design in the Auto-bidding World"
**File:** `balseiro_2021_robust.md`

This paper addresses auction design when bidders are value maximizers with return-on-spend constraints, demonstrating that reserve prices, which in classical auction theory improve revenue but not welfare, can simultaneously improve both revenue and social welfare when bidders are value maximizers. The key insight is that value-maximizing bidders react strategically to reserve prices to satisfy their ROS constraints, and appropriately chosen reserves eliminate inefficient bidding strategies. The authors develop a general technical lemma that translates the accuracy of value signals into approximation guarantees for both welfare and revenue across VCG, GSP, and first-price auctions, and they show how to combine reserve prices with additive boosts for further welfare improvements. Importantly, their results are robust along three dimensions: they tolerate inaccurate value signals, they apply to mixed environments where value maximizers and utility maximizers coexist, and the guarantees hold for undominated bidding strategies rather than only at equilibrium.

### Balseiro, Bhawalkar, Mirrokni, and Sivan (2024) — "A Field Guide for Pacing Budget and ROS Constraints"
**File:** `balseiro_field_guide_2024.md`

This paper addresses a practical architectural challenge in advertising platforms: how to coordinate the separate systems that enforce budget constraints and return-on-spend (ROS) constraints for autobidding advertisers. The authors compare three pacing architectures with increasing degrees of coordination. Sequential pacing, where a ROS pacing service feeds an intermediate bid to a separate budget pacing service, is the simplest but can lead to unstable dynamics and either linear ROS constraint violations or linear regret. The joint dual-optimal pacing algorithm, which maintains centralized dual variables for both constraints updated via online mirror descent, achieves O(sqrt(T)) bounds on budget depletion timing, ROS constraint violation, and regret. The paper's main theoretical contribution shows that "min pacing," a minimally coupled architecture that takes the minimum of the bids from two independent pacing services, also achieves the same O(sqrt(T)) guarantees as the centralized approach, despite the absence of a natural Lyapunov function.

### Balseiro, Kumar, and Kroer (2023) — "Contextual Standard Auctions with Budgets: Revenue Equivalence and Efficiency Guarantees"
**File:** `balseiro_kroer_2023.md`

This paper introduces a contextual valuation framework for studying equilibrium bidding in budget-constrained auctions, where bidder values are determined by the inner product of advertiser weight vectors and item feature vectors, creating realistic correlations across bidder valuations. The authors prove the existence of value-pacing-based Bayes-Nash equilibria for all standard auctions (including first-price and second-price), and their most striking result is a revenue equivalence theorem showing that all standard auctions generate the same expected revenue in the presence of in-expectation budget constraints, despite the well-known result that revenue equivalence breaks under strict budget constraints. This provides the first principled theoretical justification for the empirical observation that publishers' revenues returned to pre-switch levels after ad exchanges moved from second-price to first-price auctions. The existence proof requires a novel fixed-point argument in the infinite-dimensional space of pacing multipliers using the topology of functions of bounded variation.

### Balseiro, Lu, and Mirrokni (2023) — "The Best of Many Worlds: Dual Mirror Descent for Online Allocation Problems"
**File:** `balseiro_mirror_2023.md`

This paper develops a unified algorithmic framework for online allocation problems with resource constraints that performs well across multiple input models without knowing which model it faces. The core algorithm maintains dual multipliers for resource constraints updated via online mirror descent, choosing actions that maximize opportunity-cost-adjusted rewards given current dual estimates. The key theoretical contribution is proving that the same algorithm achieves O(sqrt(T)) regret under i.i.d. stochastic inputs, an asymptotically optimal fixed competitive ratio under adversarial inputs, and O(sqrt(T)) regret under various non-stationary stochastic models including ergodic and periodic inputs. A natural self-correcting property ensures resources are never depleted too early on any sample path. The framework is notable for its minimal requirements: unlike prior methods, it handles non-concave reward functions, non-linear consumption functions, and integral action spaces without needing to solve large convex programs.

### Conitzer, Kroer, Sodomka, and Stier-Moses (2021) — "Multiplicative Pacing Equilibria in Auction Markets"
**File:** `conitzer_2021_sppe.md`

This paper introduces and formally analyzes pacing equilibria in sequential second-price auction markets where buyers face budget constraints. The authors define a second price pacing equilibrium (SPPE) in which a platform applies a pacing multiplier between 0 and 1 to uniformly scale each buyer's bids. They prove that such pacing equilibria are guaranteed to exist and show that multiple equilibria can arise, with potentially large variation in social welfare and revenue. A key theoretical contribution is establishing that pacing equilibria refine competitive equilibria, while also demonstrating revenue non-monotonicity (adding bidders can reduce revenue). Computing welfare-maximizing or revenue-maximizing pacing equilibria is shown to be NP-hard, but the authors present a mixed-integer programming formulation. Empirical analysis using instances generated from real-world auction data shows that equilibrium multiplicity is rare in practice.

### Conitzer et al. (2022) — "Pacing Equilibrium in First Price Auction Markets"
**File:** `conitzer_2022.md`

This paper extends the pacing equilibrium framework to first-price auctions, motivated by the industry-wide shift from second-price to first-price auctions in display advertising. Their central finding is that first-price auctions offer surprising theoretical endorsement as the underlying mechanism for budget-paced auction markets, guaranteeing uniqueness of the steady-state pacing equilibrium, monotonicity of outcomes in bids and budgets, and efficient computation through the Eisenberg-Gale convex program. Contrary to concerns about strategic manipulation in first-price settings, they show that incentive issues are not a practical barrier, with bidders exhibiting small ex-post regret and little incentive to misreport. The paper bridges a gap between theory and practice by showing that budget-constrained bidders tend to have even smaller regrets, and that budgets, pacing multipliers, and regrets are positively associated.

### Chen and Kroer (2024) — "The Complexity of Pacing for Second-Price Auctions"
**File:** `chen_kroer_2024.md`

This paper resolves the open question posed by Conitzer et al. (2021) by proving that computing an approximate pacing equilibrium in second-price auctions is PPAD-complete, establishing that budget management via multiplicative pacing is fundamentally intractable in multi-buyer settings. The second-price rule plays a crucial role in the hardness reduction, as the payment made by a winning buyer is determined by the competing buyer's bid, allowing the construction of gadgets that encode Nash equilibria of arbitrary bimatrix games. A major practical implication is the disproof of Borgs et al.'s (2007) conjecture that tatonnement-style budget-management dynamics converge efficiently for second-price auctions. This contrasts sharply with first-price auctions, where pacing equilibria are efficiently computable due to a direct connection to market equilibria.

### Chen and Li (2025) — "Constant Inapproximability of Pacing Equilibria in Second-Price Auctions"
**File:** `chen_li_2025.md`

This paper strengthens the hardness results of Chen, Kroer, and Kumar (2024) by proving that even computing a constant-factor approximation of a pacing equilibrium in second-price auctions is PPAD-hard, ruling out a PTAS unless PPAD equals FP. The proof uses a reduction from the Pure-Circuit problem, which has emerged as a powerful technique for establishing strong constant inapproximability results in PPAD. The construction encodes Pure-Circuit variables as pacing multipliers of buyers, with NOT, NOR, and NPURIFY gates simulated through carefully calibrated goods and budget constraints. This result closes the gap in the complexity landscape of budget-constrained auction equilibria.

### Gaitonde, Li, Light, Lucier, and Slivkins (2023) — "Budget Pacing in Repeated Auctions: Regret and Efficiency without Convergence"
**File:** `gaitonde_2023.md`

This paper proves that when bidding agents in repeated auctions with budgets simultaneously apply gradient-based pacing algorithms, the aggregate liquid welfare achieved over the learning dynamics is at least half of the optimal expected liquid welfare, without requiring the algorithms to converge to equilibrium. This result sidesteps known PPAD-hardness obstacles and applies to a broad class of "core auctions" that includes first-price, second-price, and GSP auctions. For individual regret guarantees, the paper establishes O(T^{3/4}) regret for any single agent relative to the best fixed pacing multiplier, under a monotone bang-per-buck property satisfied by common auction formats. Semi-synthetic numerical simulations based on Microsoft Bing Advertising data complement the theoretical findings. This is a foundational result for autobidding systems, demonstrating that simple, practical learning algorithms can simultaneously guarantee near-optimal market efficiency and good individual performance.

### Fikioris and Tardos (2023) — "Liquid Welfare Guarantees for No-Regret Learning in Sequential Budgeted Auctions"
**File:** `fikioris_tardos_2023.md`

This paper establishes liquid welfare guarantees for sequential first-price auctions with budget-constrained buyers who use no-regret learning algorithms based on bid shading multipliers. The main result shows that the total liquid welfare is within a factor of approximately 2.41 of the optimal when learners achieve no regret. A key contrast with second-price auctions is highlighted: even with no-regret learners, liquid welfare in sequential second-price auctions can be arbitrarily bad, making first-price auctions structurally superior for welfare in budgeted sequential settings. The authors also provide a learning algorithm that achieves the required competitive ratio in adversarial settings. Their behavioral assumption is notably weaker than prior work, requiring only competitive ratio relative to the best fixed shading multiplier rather than convergence to equilibrium.

### Lucier, Pattathil, Slivkins, and Zhang (2024) — "Autobidders with Budget and ROI Constraints: Efficiency, Regret, and Pacing Dynamics"
**File:** `lucier_2024.md`

Lucier et al. study a game between autobidding algorithms competing in a repeated auction platform, where each autobidder maximizes its advertiser's total value subject to budget, ROI, and maximum bid constraints. They propose a gradient-based learning algorithm that satisfies all constraints ex post with probability 1 (not merely in expectation) and achieves vanishing individual regret using only bandit feedback. The paper's main result is an aggregate welfare guarantee: when all autobidders deploy this algorithm, the expected liquid welfare is at least half of the optimal, regardless of whether the dynamics converges to an equilibrium. The approach uses a non-standard variant of stochastic gradient descent where the autobidder myopically uses the smaller of two constraint-pacing bids each round. The algorithm generalizes to first-price, second-price, and any "intermediate" auction format.

### Feng, Lucier, and Slivkins (2024) — "Strategic Budget Selection in a Competitive Autobidding World"
**File:** `feng_2024.md`

This paper studies a metagame played by advertisers who strategically choose budget constraints and maximum bids to submit to autobidding algorithms on a platform running first-price auctions. The authors prove that at any pure Nash equilibrium of the metagame, the resulting allocation achieves at least half of the optimal liquid welfare, and this bound is tight. For mixed Nash and Bayes-Nash equilibria, the approximation factor degrades to 4. A striking negative result shows that if advertisers can only specify a value-per-click or ROI target without a budget, the welfare loss at equilibrium can be as bad as linear in the number of advertisers. These findings underscore the importance of budget constraints as a strategic instrument.

### Paes Leme, Piliouras, Schneider, Spendlove, and Zuo (2024) — "Complex Dynamics in Autobidding Systems"
**File:** `paes_leme_2024.md`

This paper from Google Research investigates the dynamical behavior of automated bidding agents that optimize bids to satisfy return-over-spend constraints, demonstrating that even simple autobidding systems can exhibit surprisingly complex non-equilibrium dynamics. While prior work on auction dynamics for quasi-linear bidders showed reliable convergence, the authors prove that the autobidding model produces bi-stability, periodic orbits, and quasi-periodicity. For two bidders the system always converges, but for three or more bidders complex oscillatory behavior emerges. Two theoretical results establish the general expressive power of these systems: autobidding dynamics can simulate arbitrary linear dynamical systems and can implement logical boolean gates, implying that the complexity of their behavior is essentially unbounded.

### Bergemann, Bonatti, and Wu (2025a) — "How Do Digital Advertising Auctions Impact Product Prices?"
**File:** `bergemann_bonatti_2025a.md`

This paper examines how digital advertising platforms, through their auction mechanisms and data-driven bidding, affect the product prices that advertisers charge consumers both on and off the platform. The model considers a monopolistic digital platform selling access to consumers, where advertisers can use platform data to offer personalized prices to on-platform shoppers while offering uniform prices to off-platform loyal customers. Access to platform data broadens the market by enabling advertisers to reach consumers who would otherwise be priced out, improving matching efficiency. However, advertisers face a "showrooming" constraint: on-platform offers must be at least as attractive as off-platform prices. This work connects the internal mechanics of advertising auctions to consumer-facing outcomes rather than treating the auction as an isolated mechanism.

### Bergemann, Bonatti, and Wu (2025b) — "Bidding with Budgets: Algorithmic and Data-Driven Bids in Digital Advertising"
**File:** `bergemann_bonatti_2025b.md`

This paper develops an equilibrium model of auto-bidding with budget constraints on digital advertising platforms, explicitly connecting each advertiser's budget choice to a formal bidding mechanism where per-impression bids are generated by the platform's algorithm. The framework models a platform that operates both a managed campaign (enforcing budget constraints internally) and a visible shadow auction (where effective bids depend on chosen budgets). The authors determine the optimal bidding algorithm that maximizes advertiser value subject to budget constraints, then characterize the "bid equilibrium" where each advertiser optimally chooses their budget. This departs from prior auto-bidding literature by adopting a steady-state rather than dynamic perspective and by enlarging the equilibrium notion to require that advertisers choose both their budget constraints and their bids optimally.

### Alimohammadi, Mehta, and Perlroth (2023) — "Incentive Compatibility in the Auto-bidding World"
**File:** `alimohammadi_2023.md`

This paper examines whether canonical auction formats remain incentive-compatible when advertisers interact with auctions through autobidding intermediaries, introducing the concept of auto-bidding incentive compatibility (AIC). The main result establishes that both first-price auctions and second-price auctions fail to be AIC for value-maximizing advertisers with either budget or tCPA constraints, meaning advertisers can strategically misreport to gain advantage through the autobidder's optimization. This contrasts with the finding that FPA is AIC when autobidders are restricted to the suboptimal uniform bidding policy, highlighting how the sophistication of the bidding algorithm itself affects strategic incentives. The paper further generalizes the negative result, showing that any truthful, scale-invariant, and symmetric auction cannot be AIC.

### Aggarwal and Fikioris (2025) — "No-Regret Algorithms in non-Truthful Auctions with Budget and ROI Constraints"
**File:** `aggarwal_fikioris_2025.md`

This paper addresses the problem of designing online autobidding algorithms for a buyer who must optimize value subject to both budget and ROI constraints in non-truthful auctions. The main result is a full-information algorithm achieving near-optimal O(sqrt(T)) regret against the benchmark of the best Lipschitz continuous function mapping values to bids, a substantially richer class than the uniform pacing multipliers used in prior work. For the bandit setting where the bidder observes only win/loss outcomes, the authors prove a fundamental Omega(T^{2/3}) lower bound on regret, demonstrating a large gap between full-information and bandit feedback. The paper also provides a black-box reduction that converts algorithms with approximate ROI satisfaction into ones with strict constraint enforcement.

### Liao, Kroer et al. (2025) — "Interference Among First-Price Pacing Equilibria: A Bias and Variance Analysis"
**File:** `liao_2025.md`

Liao, Kroer, and co-authors from Meta address the problem of A/B testing in advertising markets where budget-constrained buyers create interference between experimental arms. They propose a parallel budget-controlled A/B testing design that segments the market into submarkets and runs independent experiments within each segment. Their main theoretical contribution is a debiased surrogate that eliminates the first-order bias inherent in FPPE-based measurements, along with a plug-in estimator whose asymptotic normality they establish. The practical contribution is validated through 99 real paired experiments at Meta, showing that the parallel design agrees with the gold-standard budget-split design in 75-79% of cases.

---

## 6. Learning in Auctions

### Nedelec, El Karoui, and Perchet (2022) — "Learning in Repeated Auctions"
**File:** `nedelec_2022.md`

This survey covers the intersection of statistical learning theory and auction design, tracing the evolution from classical Bayesian mechanism design through modern data-driven approaches to auction optimization. A central contribution is the treatment of the setting where both sellers and bidders are adaptive and strategic over repeated interactions, a scenario that arises naturally in online advertising platforms where demand-side platforms participate in billions of auctions daily. The survey demonstrates how strategic buyers can manipulate a seller's learning process by shading their bids to influence the inferred value distribution, thereby obtaining more favorable reserve prices over time. This strategic manipulation reverses the classical information asymmetry between patient sellers and myopic buyers, creating rich game-theoretic dynamics.

### Bichler, Oberlechner, Lunowa, Pieroth, and Wohlmuth (2024) — "On the Convergence of Learning Algorithms in Bayesian Auction Games"
**File:** `bichler_convergence_2024.md`

This paper provides a rigorous mathematical analysis of why gradient-based learning algorithms converge to Bayes-Nash equilibria in auction games, despite the absence of standard sufficient conditions. The authors prove that neither first-price nor second-price auctions satisfy monotonicity, pseudo-monotonicity, or quasi-monotonicity in the relevant function spaces. However, they establish that the BNE is the unique solution to the variational inequality within the class of uniformly increasing bid functions, which guarantees that any gradient-based algorithm that converges must converge to the equilibrium. The paper introduces a novel proof technique based on the Gateaux derivative of the ex-ante utility function. These results explain why practical equilibrium learning algorithms succeed in auctions despite the theoretical non-monotonicity.

### Bichler, Fichtl, Heidekruger, Kohring, and Sutterer (2021) — "Learning Equilibria in Symmetric Auction Games using Artificial Neural Networks"
**File:** `bichler_nature_2021.md`

This paper introduces Neural Pseudogradient Ascent (NPGA), a method for computing Bayes-Nash equilibria in symmetric auction games by representing bid functions as neural networks and learning through self-play. The central challenge is that ex-post utility functions in auctions are discontinuous at the bid value determining whether a bidder wins. NPGA circumvents this by using evolutionary strategy (ES) optimization to compute pseudo-gradients that effectively smooth the objective. The authors prove that in symmetric auction games, shared-weight NPGA reduces to gradient ascent in a potential game, guaranteeing convergence to local Nash equilibria, and empirically verify that these coincide with global BNE. The method handles interdependent valuations, non-quasilinear utility functions, and multi-object settings.

### Bichler, Fichtl, and Oberlechner (2025) — "Computing Bayes Nash Equilibrium Strategies in Auction Games via Simultaneous Online Dual Averaging"
**File:** `bichler_oda_2025.md`

This paper introduces SODA (Simultaneous Online Dual Averaging), an algorithmic framework for computing Bayes-Nash equilibria in auction games by discretizing type and action spaces and learning distributional strategies via online convex optimization. A key advantage is that it makes no assumptions about the shape of the bid function, and the expected utility is linear in the strategies. The authors establish that equilibria of the discretized game approximate equilibria in the continuous game, providing a theoretical bridge between the computational method and the underlying economic model. SODA approximates known analytical BNE closely in seconds for symmetric models, positioning it as a broadly applicable equilibrium solver.

### Deng, Hu, Lin, and Zheng (2022) — "Nash Convergence of Mean-Based Learning Algorithms in First-Price Auctions"
**File:** `deng_nash_2022.md`

This paper completely characterizes the convergence properties of mean-based learning algorithms (including Multiplicative Weights Update, Follow the Perturbed Leader, and epsilon-Greedy) in repeated first-price auctions with fixed bidder values. The results depend critically on how many bidders share the highest value: when three or more bidders tie at the top, the dynamics converge to Nash equilibrium in both time-average and last-iterate senses; when exactly two tie, convergence occurs in time-average but not necessarily last-iterate; and when only one bidder has the highest value, convergence may fail entirely. The core technical contribution is formally proving that mean-based algorithms can iteratively eliminate dominated strategies across multiple phases. The equivalence between first-price auctions and Bertrand competition under fixed values means these results also characterize price convergence when firms use mean-based learning algorithms.

### Aggarwal, Gupta, Perlroth, and Velegkas (2024) — "Randomized Truthful Auctions with Learning Agents"
**File:** `aggarwal_2024_neurips.md`

This NeurIPS paper studies auction design when bidders use no-regret learning algorithms in repeated single-item auctions with persistent valuations. The key constructive result demonstrates that randomized strictly-IC auctions guarantee convergence to truthful bidding regardless of learning rate ratios, and a black-box transformation can convert any IC auction into a nearby strictly-IC auction with negligible differences. This yields the striking finding that randomized auctions achieve strictly higher revenue than Myerson-optimal second-price auctions with reserves when facing learning bidders. In the non-asymptotic regime, the paper establishes tight auctioneer-regret bounds of Theta(T^{3/4}) for constant auction policies and Theta(sqrt(T)) for two-phase schedules.

### Kumar, Schneider, and Sivan (2024) — "Strategically-Robust Learning Algorithms for Bidding in First-Price Auctions"
**File:** `kumar_2024.md`

Kumar et al. propose a novel concave formulation for pure-strategy bidding in first-price auctions and use it to analyze Online Gradient Ascent as a bidding algorithm. The paper demonstrates that Gradient Ascent simultaneously achieves optimal O(sqrt(T)) regret against adversarial inputs and is strategically robust, meaning a seller cannot extract more revenue than the Myerson-optimal mechanism. This is the first algorithm to achieve both properties. Under stochastic competition, regret improves to O(log T), exponentially better than previous bounds. A surprising technical insight is that "agile" projections (used by Gradient Ascent) versus "lazy" projections (used by FTRL/Hedge) is the key driver of robustness.

### Badanidiyuru, Feng, and Guruganesh (2023) — "Learning to Bid in Contextual First Price Auctions"
**File:** `badanidiyuru_2023.md`

This paper studies a single bidder who repeatedly participates in contextual first-price auctions where competitors' maximum bids follow a structured linear model with unknown parameters and log-concave noise. The authors design no-regret learning algorithms under binary feedback (win/loss only) achieving O(sqrt(log(d)*T)) regret, and under full information with unknown noise achieving O(sqrt(d*T)) regret with a matching lower bound. They also extend results to partially known noise distributions. This work is directly relevant to automated bidding in display advertising, where the shift to first-price auctions forces bidders to learn about competitors' bid distributions.

### Han, Zhou, Flores, Ordentlich, and Weissman (2020) — "Learning to Bid Optimally and Efficiently in Adversarial First-price Auctions"
**File:** `han_2020.md`

This paper develops the first minimax optimal online bidding algorithm for repeated first-price auctions where both valuations and competing bids can be arbitrary, achieving O-tilde(sqrt(T)) regret relative to all Lipschitz bidding policies. The core algorithmic innovation is a hierarchical expert-chaining structure called Chained Exponential Weighting (ChEW), and a computationally efficient version (SEW) with O(T) space and O(T^{3/2}) time complexity. The paper proves an impossibility result showing sublinear regret against all monotone bidding policies is unachievable. Experiments on three real-world first-price auction datasets demonstrate robust practical performance.

### Banchio, Skrzypacz / Bichler, Gupta (2023/2024) — "Revenue in First- and Second-Price Display Advertising Auctions"
**File:** `banchio_skrzypacz_2023.md` / `bichler_2024.md`

This pair of papers challenges the algorithmic collusion narrative for first-price auctions by examining a broader range of learning algorithms and utility models. The authors argue that collusive outcomes found by Banchio and Skrzypacz (2022) are specific to symmetric Q-learning agents with identical hyperparameters in complete-information settings, and they demonstrate that collusion is not robust even among Q-learning agents when hyperparameters differ. When combining Q-learning with Exp3 or Thompson Sampling, or in Bayesian incomplete-information models, no systematic deviations from equilibrium emerge. The second major contribution addresses ROI/ROS-maximizing bidders, finding that second-price auctions yield higher expected revenue than first-price for these objective types, suggesting revenue losses after the industry shift may stem from non-quasi-linear bidder objectives rather than collusion.

### Khezr and Taylor (2025) — "Artificial Intelligence for Multi-Unit Auction Design"
**File:** `khezr_taylor_2025.md`

Khezr and Taylor apply reinforcement learning to simulate bidding behavior in three sealed-bid multi-unit auction formats: Discriminatory Price, Generalized Second-Price, and Uniform-Price auctions. The paper compares six RL algorithms spanning three families: tabular Q-learning and Deep Q-Networks (value-based), vanilla and deep policy gradient (policy-based), and A2C and PPO (actor-critic). PPO dominates in both payoff and learning stability across all formats. The uniform-price auction achieves highest allocative efficiency while the discriminatory price auction generally dominates in revenue except when units are very scarce. This extends the RL-in-auctions paradigm beyond Q-learning to modern policy gradient methods.

### d'Eon, Newman, and Leyton-Brown (2024) — "Understanding Iterative Combinatorial Auction Designs via Multi-Agent Reinforcement Learning"
**File:** `deon_2024.md`

This paper develops a general methodology for applying multi-agent reinforcement learning (MARL) to analyze iterative combinatorial auctions, used prominently for spectrum privatization. The methodology recommends abstracting action spaces, controlling bidder count, limiting auction length, and crucially maintaining imperfect information and bidder asymmetry, as perfect-information models produce unrealistic equilibria. The case study compares bid-processing rules using both Monte-Carlo CFR and PPO, finding that a designer would reach the opposite conclusion about a rule change's effects if they modeled bidders as following heuristics instead of learned strategies. The paper provides detailed guidance on avoiding brittle equilibria, validating convergence via NashConv, and interpreting multiple equilibria.

---

## 7. Contextual Bandits

### Thompson (1933) — "On the Likelihood that One Unknown Probability Exceeds Another"
**File:** `thompson_1933.md`

This foundational paper by William R. Thompson introduces what is now known as Thompson sampling, the earliest known algorithm for adaptively allocating experimental effort based on posterior probability estimates, motivated by clinical trial design where the rate of data accumulation is slow or the subjects treated are valuable. Thompson's key insight is that rather than committing to one treatment based on current evidence, one should allocate treatments in proportion to the posterior probability that each is superior, thereby reducing the expected number of individuals harmed while still gathering evidence. The mathematical core derives exact algebraic expressions for the probability that one unknown binomial success rate exceeds another. Although the paper's notation reflects 1930s mathematical statistics, the underlying idea of probability matching for sequential allocation has proved extraordinarily influential, forming the basis for bandit algorithms now used at scale in internet advertising, recommendation systems, and adaptive bidding agents.

### Li, Chu, Langford, and Schapire (2010) — "A Contextual-Bandit Approach to Personalized News Article Recommendation"
**File:** `li_2010.md`

Li et al. introduce LinUCB, a contextual bandit algorithm for personalized content recommendation. The algorithm models the expected payoff of each arm (article) as a linear function of contextual features and selects arms by computing an upper confidence bound on the predicted payoff using ridge regression. The paper contributes the LinUCB algorithm with O(sqrt(KdT)) regret, an unbiased offline evaluation methodology, and a large-scale empirical validation on over 33 million events showing a 12.5% click-lift over context-free bandits. The algorithm comes in disjoint and hybrid versions, enabling knowledge transfer across arms. This paper is foundational for the contextual bandit literature and directly underlies the LinUCB framework used in Experiment 3a of this project.

### Abbasi-Yadkori, Pal, and Szepesvari (2011) — "Improved Algorithms for Linear Stochastic Bandits"
**File:** `abbasi_yadkori_2011.md`

This paper presents improved algorithms for the linear stochastic multi-armed bandit problem using a novel self-normalized tail inequality for vector-valued martingales, which enables substantially tighter confidence ellipsoids. Using these within the OFUL algorithm, the authors achieve O(d*sqrt(n)*log(n)) regret, improving over previous results. For the simpler d-armed bandit, their modified UCB achieves constant high-probability regret. The paper also introduces a rarely-switching variant recomputing estimates only O(log n) times with comparable regret bounds. This work provides foundational tools for contextual bandit algorithms, including the LinUCB family used in this project's experiments.

### Agrawal and Goyal (2013) — "Thompson Sampling for Contextual Bandits with Linear Payoffs"
**File:** `agrawal_goyal_2013.md`

This paper provides the first theoretical regret guarantees for Thompson Sampling in the stochastic contextual multi-armed bandit problem with linear payoff functions, proving a high-probability bound of O(d^{3/2}*sqrt(T)) matching the best bound achieved by any computationally efficient algorithm. The key technical innovation is a martingale-based analysis that divides arms into "saturated" and "unsaturated" groups. Critically, the analysis is frequentist despite the Bayesian algorithm design: the regret bounds hold regardless of whether actual reward distributions match the assumed Gaussian model. This paper establishes the theoretical foundation for Thompson Sampling in contextual settings, directly underpinning the algorithms used in Experiment 3b.

### Russo, Van Roy, Kazerouni, Osband, and Wen (2018) — "A Tutorial on Thompson Sampling"
**File:** `russo_2018.md`

This comprehensive tutorial presents Thompson sampling through progressively complex examples including Bernoulli bandits, online shortest path problems, assortment optimization, active learning with neural networks, and reinforcement learning in MDPs. The authors contrast Thompson sampling with greedy algorithms, showing how the latter can lock onto suboptimal actions with high confidence while Thompson sampling's stochastic exploration prevents such failures. The tutorial emphasizes problems with complex information structures where observations from one action inform beliefs about others. This is directly relevant to auction collusion research because Thompson sampling is one of the principal bandit algorithms used by adaptive bidders in contextual auction settings.

### Flajolet and Jaillet (2017) — "Real-Time Bidding with Side Information"
**File:** `flajolet_jaillet_2017.md`

This paper addresses repeated bidding in online advertising second-price auctions when contextual side information is available as a d-dimensional vector. The problem is modeled as a contextual bandit with a knapsack constraint and continuous action space. The authors develop UCB-type algorithms combining ellipsoidal confidence sets with probabilistic bisection search for the optimal bid shading multiplier, establishing O-tilde(d*sqrt(T)) regret bounds. A key insight is that overbidding serves a dual purpose: incentivizing exploration and discovering the optimal bidding strategy in the partially censored information setting of real-time bidding. The paper contributes to the intersection of contextual bandits and bandits-with-knapsacks.

---

## 8. SPA-to-FPA Transition

### Despotakis, Ravi, and Sayedi (2021) — "First-price Auctions in Online Display Advertising"
**File:** `despotakis_2021.md`

This paper provides a clean game-theoretic explanation for why display advertising exchanges shifted from second-price to first-price auctions, attributing the change directly to publishers' prior adoption of header bidding. The model shows that under waterfalling, revenue equivalence ensures no incentive to change format. But once publishers move to header bidding, a unique equilibrium emerges where both exchanges adopt first-price auctions. The combination of header bidding and first-price auctions commoditizes exchanges' offerings and drives buyer-side fees to zero, consistent with observed industry trends. The new regime allows publishers to achieve Myerson-optimal revenue without direct access to advertisers and greatly simplifies reserve price optimization.

### Alcobendas and Zeithammer (2025) — "Slim Shading in Ad Auctions"
**File:** `alcobendas_2025.md`

This empirical paper documents how bidders on Yahoo Ad Exchange responded to the switch from second-price to first-price auction rules, using detailed transaction-level data. A nonparametric estimator provides lower bounds on valuations from observed first-price bids. The key finding is that while large bidders reduced their bids after the format switch as theory predicts, the reduction was insufficient: post-switch bids remained too high, implying incomplete bid-shading that persisted for more than three months. This provides direct empirical evidence that theoretical predictions of immediate equilibrium adjustment are overly optimistic in practice.

### Goke, Weintraub, Mastromonaco, and Seljan (2022) — "Bidders' Responses to Auction Format Change"
**File:** `goke_2022.md`

This paper provides one of the first large-scale field studies on format-change responses, using data from the Xandr ad exchange covering hundreds of millions of daily auctions. Exploiting staggered adoption across four batches, the authors show that immediately after the format change, average revenue per impression jumped 35-75%, but this dissipated over 30-60 days, consistent with initially insufficient bid shading followed by learning. The convergence of first-price and second-price price levels is consistent with the revenue equivalence theorem. The authors also document heterogeneity in bidder sophistication: advertisers using the exchange's own bidding algorithm adapted faster than those using third-party algorithms.

### Gligorijevic et al. (2020) — "Bid Shading in The Brave New World of First-Price Auctions"
**File:** `gligorijevic_2020.md`

This paper proposes a machine learning approach for optimal bid shading in open first-price auctions, where every bidder observes the minimum bid needed to win. The authors deploy Factorization Machine models to learn the shading ratio from historical auction data, achieving approximately 18.6% offline and 20.5% online surplus improvements on a major DSP (Verizon Media). The paper distinguishes open from censored first-price settings and represents one of the first published descriptions of a working bid shading algorithm deployed at industrial scale.

### Jauvion et al. (2018) — "Optimization of a SSP's Header Bidding Strategy using Thompson Sampling"
**File:** `jauvion_2018.md`

Jauvion et al. address the optimization problem facing a Supply-Side Platform competing in header bidding first-price auctions. The SSP's decision is formalized as a contextual bandit: given its internal second-price closing price, choose a bid balancing winning against margin preservation. The authors develop Thompson Sampling with particle filter inference to estimate competing bid distributions. Experiments on two real RTB datasets show Thompson Sampling outperforms UCB and epsilon-greedy approaches. This provides a concrete example of Thompson Sampling applied to auction bidding, the same approach studied in Experiment 3b.

---

## 9. Empirical Evidence

### Calder-Wang and Kim (2024) — "Coordinated vs Efficient Prices: The Impact of Algorithmic Pricing on Multifamily Rental Markets"
**File:** `calder_wang_2024.md`

This paper empirically evaluates algorithmic pricing (specifically RealPage's Yieldstar and Rainmaker LRO) on U.S. multifamily rental housing, finding that at least 25% of buildings (34% of units) used pricing algorithms by 2019. At the building level, algorithmic pricing makes prices more responsive to market conditions. At the market level, higher penetration is associated with 3% higher rents and lower occupancy. To distinguish responsive pricing from coordination, the authors estimate a structural model and conduct a formal conduct test. The conduct test finds own-profit-maximization generally favored over full coordination. The paper's central insight is that reduced-form evidence alone cannot distinguish coordination from responsive pricing, making structural modeling essential.

### Council of Economic Advisers (2024) — "The Cost of Anticompetitive Pricing Algorithms in Rental Housing"
**File:** `cea_2024.md`

This White House policy analysis quantifies the impact of RealPage's algorithmic pricing on rental housing, estimating renters pay $70/month more on average, totaling $3.8 billion in excess costs in 2023. The analysis draws on Calder-Wang and Kim (2024) and notes the DOJ lawsuit filed against RealPage in August 2024. The CEA argues the software weakens competition because pooled competitor data feeds recommendations accepted up to 90% of the time. The $70/month estimate is framed as a lower bound. This is notable as an official U.S. government assessment treating algorithmic pricing coordination as a quantifiable harm.

### Ezrachi and Stucke (2024) — "The Role of Secondary Algorithmic Tacit Collusion in Achieving Market Alignment"
**File:** `ezrachi_stucke_2024.md`

This article introduces "secondary algorithmic tacit collusion" (STC), where anticompetitive price alignment persists even when competitors use different algorithmic pricing hubs rather than a single shared provider. The key contribution is showing that competition among pricing hubs does not prevent collusion on the primary market; conscious parallelism among hubs can sustain anticompetitive outcomes across both levels. STC can occur in markets with many rivals and differentiated products, conditions traditionally thought to preclude tacit collusion. The authors propose several legal avenues including targeting data exchange and improving merger review.

---

## 10. Legal and Policy

### Harrington (2018) — "Developing Competition Law for Collusion by Autonomous Artificial Agents"
**File:** `harrington_2018.md`

Harrington addresses the fundamental legal challenge posed by autonomous agents that learn to collude on supracompetitive prices without human coordination. A central argument is that under current US competition law (Section 1 of the Sherman Act), collusion by learning algorithms is not unlawful because liability requires evidence of an "agreement" achieved through overt communication. The paper proposes shifting legal focus from the process (communication) to properties of the pricing algorithm itself, suggesting regulators could audit algorithms to determine whether they embody reward-punishment schemes. This contribution is foundational to the algorithmic collusion literature, framing the legal gap that subsequent regulatory proposals attempt to fill.

### Harrington (2025a) — "The Challenges of Third-Party Pricing Algorithms for Competition Law"
**File:** `harrington_challenges_2025.md`

This paper examines antitrust challenges when competing firms delegate pricing to a common third-party data analytics company. Harrington identifies three distinct sources of anticompetitive harm: explicit price-fixing agreements, information exchange facilitating coordination, and unilateral third-party conduct producing supracompetitive prices without firms' knowledge. He reviews the efficiencies that third-party algorithms deliver, including better demand estimation through pooled data, and evaluates recently proposed remedies finding them insufficient. The paper argues existing competition law is inadequate, particularly for the scenario where the third party acts unilaterally.

### Harrington (2025b) — "A Critique of Recent Remedies for Third-Party Pricing Algorithms"
**File:** `harrington_critique_2025.md`

Harrington critically examines remedies prohibiting use of nonpublic competitor data, as embodied in the Preventing Algorithmic Collusion Act of 2024 and local laws in San Francisco and Philadelphia. He makes two arguments: first, the remedy creates significant inefficiencies by reducing available data; second, and more critically, a third party can still achieve supracompetitive prices without using nonpublic competitor data. The core insight is that the source of harm is not shared data but shared objective: the third party optimizes a collective profit function. This shows regulatory focus on data flows rather than algorithmic objectives is fundamentally misguided.

### Harrington (2025c) — "Hub-and-Spoke Collusion with a Third-Party Pricing Algorithm"
**File:** `harrington_hub_2025.md`

This paper develops a formal model of hub-and-spoke collusion where a data analytics company coordinates subscribing competitors' prices. The novel feature is that the hub simultaneously delivers genuine efficiency: demand-responsive pricing. Several notable findings emerge: stable collusion is feasible even with many small firms (because non-participation means forgoing the efficiency), the supracompetitive markup is increasing in the efficiency delivered, and the pricing algorithm's slope (demand responsiveness) is identical under competition and collusion, with collusion operating purely through an upward level shift.

### Harrington (2025d) — "An Economic Test for an Unlawful Agreement to Adopt a Third-Party's Pricing Algorithm"
**File:** `harrington_test_2025.md`

Harrington develops an empirical framework for detecting whether a third party's pricing algorithm embodies a collusive agreement. Under coordinated adoption, the algorithm produces a supracompetitive markup; under independent adoption, no markup above competitive levels. The key test compares observed average prices against the theoretical competitive benchmark. The paper also proposes an "algorithm audit" test where regulators with access to the algorithm and cost data can directly compute whether the intercept exceeds the competitive level. This bridges theory and enforcement by providing concrete tools antitrust authorities could use.

### Hartline, Long, and Zhang (2024) — "Regulation of Algorithmic Collusion"
**File:** `hartline_2024.md`

Hartline et al. propose "plausible algorithmic non-collusion" and a statistical audit procedure based on calibrated regret. The core idea is that a non-collusive algorithm should exhibit low calibrated regret, meaning it approximately best-responds to market conditions. Good algorithms can be augmented to pass the audit, while colluding algorithms cannot. The framework provides a per se regulatory rule: algorithms failing the audit are deemed non-compliant regardless of intent. The authors analyze the statistical complexity, determining how much transaction data suffices. This is one of the first rigorous, implementable regulatory proposals for algorithmic collusion.

### Hartline, Wang, and Zhang (2025) — "Regulation of Algorithmic Collusion, Refined"
**File:** `hartline_2025.md`

This paper refines the Hartline et al. (2024) framework in three directions: reducing false positives via "pessimistic calibrated regret," strengthening the case for calibrated (over best-in-hindsight) regret by constructing a manipulation example, and showing that algorithms can pass the audit by pretending to have higher costs. The paper situates its contribution within a comprehensive categorization of proposed regulations and interprets the auditing approach as a per se rule within US antitrust doctrine.

### Gal (2019) — "Algorithms as Illegal Agreements"
**File:** `gal_2019.md`

Gal systematically analyzes whether existing antitrust laws can address coordination facilitated by pricing algorithms, concluding they are insufficient. The paper charts how algorithms affect Stigler's three conditions for collusion sustainability: reaching understanding, detecting deviations, and creating credible retaliation. The author argues algorithms challenge fundamental assumptions of antitrust law, including meeting of minds, intent, and communication, because algorithms can effectively "read minds" of competing algorithms by observing their structure or outputs. The paper refutes the FTC's position that existing laws suffice and calls for widening the regulatory net.

### Mehra (2016) — "Antitrust and the Robo-Seller"
**File:** `mehra_2016.md`

Mehra provides one of the earliest comprehensive legal analyses of how algorithmic pricing challenges antitrust law. The author identifies three mechanisms by which "robo-sellers" facilitate collusion: eliminating time lags between defection and detection, reducing noise and errors, and removing hyperbolic discounting. Using the Amazon textbook pricing spiral ($23.7 million for a fruit-fly biology textbook) as a concrete example, Mehra identifies a fundamental gap in the Sherman Act's reliance on anthropomorphic concepts of intent and agreement. The paper ultimately favors expanding FTC authority to address algorithmic pricing concerns.

### Petit (2017) — "Antitrust and Artificial Intelligence: A Research Agenda"
**File:** `petit_2017.md`

Petit offers a measured critique of the emerging literature, arguing that scenarios described by Ezrachi and Stucke rest on assumptions needing more testing. He identifies five areas requiring investigation: destabilizing effects of algorithms on collusion, countervailing buyer strategies, robustness under algorithmic heterogeneity, the need for more empirical evidence, and whether antitrust goals should extend to algorithmic wealth transfers. The essay advocates a cautious, evidence-based approach, aligning with EU Competition Commissioner Vestager's position of monitoring without premature intervention.

### Crane (2024) — "Antitrust After the Coming Wave"
**File:** `crane_2024.md`

Crane argues that AI will fundamentally undermine the four pillars of antitrust: information discovery, incentive alignment, competitive market structure, and conduct enforcement. AI systems will surpass competitive markets in detecting preferences and determining efficiency without price signals. A key inversion: human managers have opaque intentions but observable actions, while AI has transparent objectives but black-box processing. Near-infinite AI scalability will likely drive extreme market concentration. Crane predicts a future of direct regulation of AI objective functions rather than mandating inter-firm competition.

### OECD (2017) — "Algorithms and Collusion: Competition Policy in the Digital Age"
**File:** `oecd_2017.md`

This comprehensive policy report examines how pricing algorithms and big data are changing the competitive landscape. A key contribution is the taxonomic framework distinguishing among algorithms used as explicit cartel tools, hub-and-spoke arrangements through shared software, predictable-agent behavior enabling tacit coordination, and fully autonomous self-learning algorithms. The report discusses how algorithms affect each pillar of collusion sustainability and addresses enforcement challenges. It remains a foundational policy reference for researchers studying algorithmic pricing.

### OECD (2023) — "Algorithmic Competition"
**File:** `oecd_2023.md`

This updated background note broadens scope to encompass unilateral algorithmic conduct including self-preferencing, predatory pricing, and exploitative abuses. It surveys growing evidence on algorithm prevalence, drawing on competition authority surveys showing monitoring and dynamic pricing algorithms are widely used while personalized pricing remains less common. A significant addition is the treatment of algorithmic auditing techniques. The report identifies hub-and-spoke arrangements through shared pricing software as the most immediate practical threat, while acknowledging the magnitude of autonomous tacit collusion remains disputed.

---

## 11. Methodology

### Athey and Imbens (2016) — "The Econometrics of Randomized Experiments"
**File:** `athey_imbens_2016.md`

This chapter presents statistical methods for randomized experiments emphasizing randomization-based inference over sampling-based approaches. The paper covers completely randomized, stratified, paired, and clustered designs, strongly recommending stratification with at least two units per stratum per arm. For non-compliance, it contrasts ITT with IV/LATE approaches. A substantial contribution addresses heterogeneous treatment effects with many covariates. The chapter also covers network experiments with interference. This reference is relevant to the factorial experimental designs used throughout this project.

### Kleijnen (2005) — "An Overview of the Design and Analysis of Simulation Experiments for Sensitivity Analysis"
**File:** `kleijnen_2005.md`

Kleijnen reviews DOE methodology as applied to computer simulations treated as black boxes, covering metamodel specification, experimental design selection (full and fractional factorials, Plackett-Burman, Latin hypercube, central composite), and validation via cross-validation. A central theme is the distinction between screening and response surface designs. The paper discusses practical advantages of 2^k designs including orthogonality for independent effect estimation, and addresses variance heterogeneity common in simulation output. This is directly foundational to the project's experimental design, which employs 2^k factorial and fractional factorial designs.

### Sanchez and Lucas (2002) — "Exploring the World of Agent-Based Simulations"
**File:** `sanchez_lucas_2002.md`

This paper from the Naval Postgraduate School provides a methodological overview of agent-based simulations, emphasizing that while individual agents follow simple rules, aggregate behavior can be highly nonlinear. A key insight is that for many ABS applications the goal is not prediction but searching for insights, identifying important factors, and finding behavioral thresholds. The paper advocates factorial experimental designs for systematic parameter exploration. This is directly relevant because learning bidders in repeated auctions are precisely the kind of adaptive agents whose emergent behavior requires systematic factorial analysis.

### Lee et al. (2015) — "The Complexities of Agent-Based Modeling Output Analysis"
**File:** `lee_2015.md`

Lee et al. provide a comprehensive overview of challenges in analyzing ABM output, addressing minimum simulation runs, sensitivity analysis, spatio-temporal analysis, and visualization. The paper highlights that standard parametric assumptions may not hold for ABM outputs and advocates distribution-free approaches to variance stability. For sensitivity analysis, it discusses factorial designs, Latin hypercube sampling, and screening methods. This is relevant to the project as best-practice guidance on replication counts, hypothesis testing under non-standard distributions, and the risks of under- or over-powered studies.

### Lin (2013) — "Agnostic Notes on Regression Adjustments to Experimental Data"
**File:** `lin_2013.md`

Lin addresses Freedman's critique of OLS regression adjustment in randomized experiments, proving that OLS adjustment with treatment-by-covariate interactions cannot hurt asymptotic precision and is guaranteed to improve it unless covariates are uncorrelated with the outcome. The Huber-White sandwich estimator provides asymptotically valid confidence intervals regardless of model misspecification. The paper draws an analogy between regression adjustment in experiments and regression estimators in survey sampling. This provides rigorous justification for covariate adjustment in simulation-based factorial ANOVA.

### Wu and Lange (2008) — "Coordinate Descent Algorithms for Lasso Penalized Regression"
**File:** `freedman_2008.md`

*Note: This file is mis-transcribed and does not contain the intended Freedman paper.* The actual content covers coordinate descent algorithms for lasso-penalized regression in L1 and L2 loss settings, with focus on high-dimensional problems. The algorithms' simplicity and speed (skipping most coordinate updates in underdetermined problems) are highlighted, along with extensions to grouped penalties. While not directly about auctions, the paper provides foundational statistical methodology for model selection applicable to factorial analysis.

---

## Appendix: Transcription Errors

The following files contain incorrect transcriptions (wrong PDF source documents) and could not be summarized from their file contents. Notes on the intended papers are provided where available from citations in other transcriptions.

### Abada, Lambin, and Tchuente (2022) — "Artificial Intelligence and Collusion"
**File:** `abada_2022.md` — **Contains:** quantum physics paper on cross phase modulation in cold atom gradient echo memory (Leung et al.)

Based on citations elsewhere, the intended paper studies whether reinforcement learning algorithms can learn to collude in repeated pricing games, finding that collusion can arise from suboptimal exploration rather than deliberate coordination.

### Asker, Fershtman, and Pakes (2022) — "Artificial Intelligence, Algorithmic Pricing, and Collusion"
**File:** `asker_2022.md` — **Contains:** Chen, Hansen, and Hansen, "Robust Identification of Investor Beliefs" (NBER WP 27257, 2020)

The intended paper studies how AI-based pricing algorithms interact in oligopolistic markets, providing both theoretical analysis and experimental evidence on conditions under which Q-learning converges to collusive outcomes.

### Dolgopolov (2021/2024) — "Stochastic Stability of Q-Learning"
**File:** `dopoloov_2021.md` — **Contains:** MSMatch semi-supervised multispectral scene classification paper (Gomez and Meoni)

The intended paper characterizes stochastically stable states for Q-learning in the prisoner's dilemma, proving that under epsilon-greedy policies the only stable outcome is mutual defection, while logit exploration can sustain cooperation.

### Hansen, Misra, and Pai (2021) — "Algorithmic Collusion: Supra-Competitive Prices via Independent Algorithms"
**File:** `hansen_2021.md` — **Contains:** power systems frequency regulation paper

The intended Marketing Science "Frontiers" article provides evidence that a classic reinforcement learning algorithm other than Q-learning can produce collusive outcomes, complementing Calvano et al. (2020).

### Hettich (2021) — [Algorithmic Collusion Study]
**File:** `hettich_2021.md` — **Contains:** "Tukey's Depth for Object Data" (Dai and Lopez-Pintado)

Transcription error. The actual paper likely concerns algorithmic pricing or collusion but cannot be summarized from available file contents.

### Klein (2021) — "Autonomous Algorithmic Collusion"
**File:** `klein_2021.md` — **Contains:** "Reformulation of the No-Free-Lunch Theorem for Entangled Data Sets" (Sharma et al., Los Alamos)

The intended paper is a well-known study showing Q-learning algorithms learn collusive pricing in oligopoly settings.

### Balseiro et al. (2021) — "The Landscape of Auto-bidding Auctions"
**File:** `balseiro_landscape_2021.md` — **Known wrong content** (skipped per plan)

### Leme et al. (2024) — [Autobidding Dilemma]
**File:** `leme_dilemma_2024.md` — **Known wrong content** (physics paper, skipped per plan)
