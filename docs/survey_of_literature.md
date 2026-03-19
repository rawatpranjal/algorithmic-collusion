# Literature Survey: Mapping Citations to Experimental Program

## 1. Citation Inventory

- **Total unique citation keys used across all .tex files:** 89
- **Total bib entries defined in `references.bib`:** 105
- **Bib entries defined but never cited:** 16
- **Citation keys used but missing from bib:** 0

### Section 2.1: Auction Theory and Equilibrium Predictions

| Citation Key | Claim Attributed | Experiment Connection | Treatment |
|---|---|---|---|
| `Vickrey1961` | Revenue equivalence under IPV | Exp 1 (null hypothesis for format effect) | Substantive |
| `MilgromWeber1982` | Linkage principle; SPA weakly dominates FPA under affiliation | Exp 2, 3a, 3b (revenue ranking benchmark) | Substantive |
| `BulowKlemperer1996` | Additional bidder > optimal reserve | All experiments (market thickness factor) | Substantive |
| `Krishna2009` | Textbook synthesis for equilibrium benchmarks | All experiments | Parenthetical |
| `Ivaldi2003` | Determinants of tacit collusion in repeated games | Exp 1 (discount factor, bidders, info) | Substantive |

### Section 2.2: Algorithmic Collusion

| Citation Key | Claim Attributed | Experiment Connection | Treatment |
|---|---|---|---|
| `Calvano2020` | Q-learning develops reward-punishment strategies in Bertrand | Exp 1, 2 (Q-learning baseline) | Substantive |
| `Klein2021` | Action space granularity matters for collusion | Exp 1 (discretisation sensitivity) | Substantive |
| `Hansen2021` | Collusion from misspecified demand, no strategic intent | Exp 1-3 (general motivation) | Substantive |
| `Calvano2023` | Genuine vs spurious collusion distinction | Exp 1, 3a, 3b (exploration factor) | Substantive |
| `Douglas2024` | Deterministic bandits always converge to collusion; stochastic exploration prevents it | Exp 3a, 3b (memory decay factor) | Substantive |
| `BanchioSkrzypacz2022` | FPA more prone to bid suppression under Q-learning | Central foil; Exp 1-3 vs 4a/4b format reversal | Substantive |
| `WaltmanKaymak2008` | RL agents attain mixed-strategy equilibria in Cournot/auction | General context | Parenthetical |
| `Bandyopadhyay2008` | Similar conclusions for reverse auctions | General context | Parenthetical |
| `Tellidou2007` | Tacit collusion sustained in simulated electricity markets | General context | Parenthetical |
| `Abada2022` | Exploration strategy choice affects collusive convergence | Exp 1 (exploration factor) | Parenthetical |
| `Dolgopolov2021` | Exploration strategy choice affects collusion | General context | Parenthetical |
| `Asker2022` | Sync vs async updating affects collusion | Exp 1 (update mode factor) | Parenthetical |
| `Hettich2021` | Deep RL extends collusion analysis | General context | Parenthetical |
| `Han2022` | Deep RL architectures for collusion | General context | Parenthetical |
| `Zhang2021` | Deep RL comparison | General context | Parenthetical |
| `KuhnTadelis2017` | Algorithms studied bear little resemblance to practice | Motivates Exp 4a/4b | Parenthetical |
| `Schwalbe2018` | Conditions for collusion more restrictive than claimed | Sceptical counterpoint | Parenthetical |
| `BanchioMantegazza2022` | "Spontaneous coupling" synchronises Q-learners | Exp 1 (mechanism context) | Substantive |
| `Bertrand2025` | Q-learners converge to cooperative Pavlov policy | Theoretical context for Exp 1-2 | Substantive |
| `Calzolari2021` | Market observability interacts with learning | Exp 1 (info feedback), Exp 2 (state info) | Substantive |
| `Zhang2025Noise` | Calibrated noise injection disrupts coordination | Policy implications | Substantive |
| `Fish2024` | LLMs reach supra-competitive prices | Extends collusion beyond RL | Substantive |
| `Arunachaleswaran2025` | Supra-competitive prices without threat capacity | Theoretical motivation | Substantive |
| `Heo2025` | RL collusion in construction procurement | Extends beyond advertising | Parenthetical |

### Section 2.2: Empirical Evidence

| Citation Key | Claim Attributed | Experiment Connection | Treatment |
|---|---|---|---|
| `Chen2016` | 1/3 of Amazon best-sellers use algorithmic pricing | General motivation | Parenthetical |
| `Assad2020` | 9-28% margin increase after algorithmic pricing | General motivation | Parenthetical |
| `BrownMacKay2023` | 10% price increase among large online retailers | General motivation | Parenthetical |

### Section 2.2: Experimental Literature (Human Bidders)

| Citation Key | Claim Attributed | Experiment Connection | Treatment |
|---|---|---|---|
| `KagelLevin2016` | Human bids generally above Nash in FPA and SPA | Benchmark for all experiments | Substantive |
| `Levin1996` | Even experienced bidders suffer winner's curse | Exp 2 (winner's curse metric) | Parenthetical |
| `Keser1996` | Declining prices in sequential auctions | General context | Parenthetical |
| `Neugebauer2007` | Declining prices in sequential FPA | General context | Parenthetical |

### Section 2.3: Autobidding, Pacing, and Platform Welfare

| Citation Key | Claim Attributed | Experiment Connection | Treatment |
|---|---|---|---|
| `Aggarwal2019` | Uniform bid-scaling optimal; PoA=2 for SPA+budgets | Exp 4a, 4b (PoA benchmark) | Substantive |
| `Conitzer2022FPPE` | FPA pacing equilibria unique and computable | Exp 4a (theoretical basis) | Substantive |
| `Deng2021TowardsEfficient` | PoA=1 under RoS constraints (not budget) | Exp 4a, 4b (RoS vs budget distinction) | Substantive |
| `Aggarwal2024Survey` | Survey of autobidding field | General context for Exp 4a/4b | Parenthetical |
| `Balseiro2019Learning` | Dual-based pacing with O(sqrt(T)) regret | Exp 4a (implements this framework) | Substantive |
| `Gaitonde2023` | Half-optimal welfare without convergence | Exp 4a, 4b (welfare benchmark) | Substantive |
| `PaesLeme2024` | Autobidding can exhibit bi-stability/periodic orbits | Exp 4a (drift test; near-zero drift falsifies prediction) | Substantive |
| `FikiorisTardos2023` | FPA welfare within 2.41x optimum; SPA can be arbitrarily bad | Exp 4a, 4b (format comparison) | Substantive |
| `Lucier2023PacingDynamics` | Gradient autobidding achieves half-optimal welfare | Exp 4a, 4b (welfare guarantees) | Substantive |

### Section 2.4: Legal and Regulatory Context

| Citation Key | Claim Attributed | Experiment Connection | Treatment |
|---|---|---|---|
| `EzrachiStucke2017` | Four modes of algorithmic participation in collusion | Motivates the autonomous learning case | Substantive |
| `Mehra2016` | "Agreement gap" in antitrust doctrine | General legal context | Substantive |
| `OECD2017` | Algorithms amplify tacit coordination conditions | General motivation | Parenthetical |
| `OECD2023` | No jurisdiction has sanctioned autonomous collusion; legislative landscape | Policy implications | Parenthetical |
| `Harrington2018` | Per se prohibition on collusion-facilitating algorithms | Policy lever | Substantive |
| `Gal2019` | Algorithmic transparency changes mode of coordination; expands scope of "agreement" | Policy implications | Substantive |
| `Petit2017` | Algorithmic heterogeneity may limit convergence; Eturas ruling | Sceptical counterpoint | Substantive |
| `Schwalbe2018` | Conditions for collusion more restrictive than claimed | Sceptical counterpoint | Substantive |
| `EzrachiStucke2024` | Hub-and-spoke collusion via shared pricing software (RealPage) | Enforcement boundary | Substantive |
| `Hartline2024` | Calibrated regret auditing framework | Policy lever | Substantive |
| `Hartline2025` | Refined regret auditing; per se rule interpretation | Policy lever | Substantive |
| `Zhang2025Noise` | Noise injection as market-design remedy | Policy lever | Substantive |
| `BanchioSkrzypacz2022` | Format choice as market-design lever | Policy lever | Parenthetical |
| `Crane2024` | AI threatens four pillars of antitrust order | Broader policy context | Substantive |
| `BichlerGuptaOberlechner2024` | Collusion fragile across hyperparameters and algorithm pairings | Evidence problem | Substantive |
| `Calvano2023` | Genuine vs. spurious collusion distinction | Evidence problem | Substantive |
| `Douglas2024` | Deterministic bandits always collude; stochastic exploration prevents it | Evidence problem | Substantive |
| `KuhnTadelis2017` | Algorithms studied don't resemble deployed ones | Evidence problem | Parenthetical |
| `Heo2025` | RL collusion in procurement auctions | Evidence problem | Parenthetical |
| `BulowKlemperer1996` | Additional bidder > optimal reserve (confirmed in algorithmic setting) | Our contribution | Parenthetical |
| `Johnson2023` | Platform design when sellers use pricing algorithms | Outcome-based monitoring | Parenthetical |

#### Review Notes (2026-03-18)

**Issue 1 (Important) — Gal 2019 mischaracterized.** Our text says Gal argues algorithms' inspectability enables *regulatory auditing*. But Gal's actual argument (lines 325-387 of transcription) is about *inter-firm transparency* facilitating coordination, and that this should expand the legal concept of "agreement" to cover algorithmic interactions. The auditing-as-opportunity framing is more Harrington's point (lines 270-284). **Fix when rewriting:** revise to say Gal argues algorithmic transparency changes the mode of communication and should expand the scope of antitrust liability, not that it enables regulatory auditing.

**Issue 2 (Suggestion) — Legislative citation imprecise.** The sentence bundling the Preventing Algorithmic Collusion Act, EU Horizontal Cooperation Guidelines, Eturas ruling, and UK CMA screening tools cites only `\citep{OECD2023}`. The OECD 2023 transcription does not mention the Preventing Algorithmic Collusion Act by name. Eturas is confirmed via Petit 2017 (lines 45-57), not OECD 2023. **Fix when rewriting:** either add a footnote with specific legislative references or split the citation so each item points to its proper source (e.g., Eturas to Petit 2017).

**Issue 3 (Important) — Content overlap with Algorithmic Collusion subsection.** Paragraph 3 of the legal subsection (line 46) repeats substantially the same technology-specificity argument made in the final paragraph of the Algorithmic Collusion subsection (line 20). Both state that nearly all findings come from Q-learning, cite Kuhn & Tadelis / Schwalbe, argue budget pacing is the real-world technology, and conclude no prior work tests pacing. CLAUDE.md says "one canonical location per idea." **Fix when rewriting:** in the legal subsection, replace the detailed restatement with a cross-reference to the earlier subsection, keeping only the legal-specific framing (the evidence base is shaky, so policy proposals built on it may be misleading).

**Issue 4 (Suggestion) — Paragraph 5 sentence structure.** Paragraph 5 (line 50) has only ~3 extremely long sentences. The style guide requires 4-8 sentences per paragraph, each carrying one idea. **Fix when rewriting:** break into 5-6 shorter sentences.

**Issue 5 — RESOLVED.** Crane 2024 bib title is now correct: "Antitrust After the Coming Wave" (verified at line 830 of references.bib, 2026-03-18).

**Issue 6 — RESOLVED.** Kuhn & Tadelis bib title is now correct: "The Economics of Algorithmic Pricing: Is Collusion Really Inevitable?" (verified at line 533 of references.bib, 2026-03-18). Minor: bib key says "2017" but year field is "2018" (cosmetic key/year mismatch, not a bug).

### The Argument: Algorithmic Collusion and the Technology-Specificity Trap

*Draft argument for eventual rewrite of legal subsection. Provided by user 2026-03-18.*

The legal challenge of algorithmic collusion centers on the "agreement gap," where autonomous agents reach supra-competitive outcomes without explicit human communication \citep{Mehra2016, Harrington2018}. Because Section 1 of the Sherman Act requires evidence of a "meeting of minds," regulators struggle to prosecute convergence driven by black-box optimization \citep{Gal2019}. In response, scholars have proposed moving away from intent-based enforcement toward outcome-focused remedies. These include "plausible non-collusion" audits based on calibrated regret \citep{Hartline2024} and market design interventions, such as mandating first-price auction formats to disrupt the signaling mechanisms used by learning algorithms \citep{BanchioSkrzypacz2022}.

However, the empirical evidence underpinning these regulatory proposals is often narrow and technology-specific. Most simulation studies focus on reinforcement learning agents, specifically tabular Q-learning, in complete-information environments \citep{Calvano2020}. Recent research suggests that these collusive findings are fragile and may disappear when algorithms use different hyperparameters or encounter stochastic market noise \citep{BichlerGuptaOberlechner2024, Zhang2025}. This creates a significant evidentiary risk for regulators because an intervention designed to curb one class of bidding technology may have unintended consequences when applied to another.

Our experimental results demonstrate that auction format effects are not universal but are fundamentally technology-specific. In Experiments 1 through 3, we find that learning-driven algorithms (Q-learning and contextual bandits) generate higher revenue in second-price auctions, which is consistent with the literature suggesting that first-price formats are more susceptible to algorithmic coordination. Yet, in Experiments 4a and 4b, this effect reverses. For budget-constrained pacing algorithms, which represent the dominant bidding technology in modern advertising exchanges, the first-price auction generates significantly higher revenue and efficiency. A regulator mandating a specific format based on evidence from learning algorithms would inadvertently harm market performance if the participants primarily utilize pacing controllers.

Furthermore, our framework highlights the "agreement gap" from a different perspective by showing that revenue shortfalls can be "accidental." In our pacing experiments, price suppression occurs not through strategic retaliation but as a mechanical byproduct of budget rationing under fixed constraints. This suggests that outcome-based audits may struggle to distinguish between genuine collusive intent and non-strategic optimization under scarcity. If the same "collusive" outcome can be produced by two entirely different computational mechanisms, then a technology-agnostic regulatory rule is likely to be miscalibrated.

Finally, we find that across all algorithm families and auction formats, structural market parameters overwhelmingly dominate behavioral or design choices. The number of bidders in the auction is a more powerful determinant of revenue and welfare than the choice of auction format or the specific learning algorithm used. This result provides a strong empirical defense for structural antitrust remedies, such as lowering entry barriers and increasing bidder participation, over behavioral mandates. By demonstrating that the impact of competition is 3 to 28 times larger than the impact of format choice, our results suggest that the most robust regulatory lever remains the preservation of competitive market structures \citep{BulowKlemperer1996}.

**NOTE**: The "3 to 28 times" claim is overstated. Actual coefficient ratios (|n_bidders| / |auction_type|) range from 1.5x (Exp 4b) to 28.1x (Exp 1). Use qualitative language when rewriting for the paper.

#### Summary Bullets (Connection to Legal Literature)

* **The Agreement Gap:** Pacing results (Exp 4) show that "collusive" outcomes (low revenue) can arise from simple budget constraints, not just strategic coordination \citep{Mehra2016}.
* **Audit Fragility:** Sign reversals between Exp 1-3 and Exp 4 prove that "plausible non-collusion" tests \citep{Hartline2024} must be validated against multiple technology classes.
* **Mandate Risk:** Proves that mandating First-Price auctions based on Q-learning studies \citep{BanchioSkrzypacz2022} would backfire in pacing-dominated markets.
* **Structural Primacy:** Validates \citet{BulowKlemperer1996} by showing that $n$ (competition) is the only universal dominant factor across all bidding technologies.

---

### Citations Appearing Only Outside literature.tex

**Methodology (experiments.tex, inference.tex):** `Box2005`, `Montgomery2017`, `BoxWilson1951`, `BoxBehnken1960`, `JonesNachtsheim2011`, `McKayBeckmanConover1979`, `SantnerWilliamsNotz2018`, `Tesfatsion2006`, `MacKinnonWhite1985`, `Lenth1989`

**Statistical appendices:** `Holm1979`, `BenjaminiHochberg1995`, `DavidsonFlachaire2008`, `KoenkerBassett1978`, `Daniel1959`, `Tibshirani1996`

**Sensitivity appendix:** `Sobol2001`, `Saltelli2008`, `Morris1991`, `Borgonovo2007`

**Algorithms:** `Russac2019` (discounted linear bandits; memory decay), `Balseiro2019Learning` (also in algorithms.tex)

---

## 2. Gap Analysis

*Updated 2026-03-18. Previous version listed bandit citations and Myerson as missing; all have since been added to references.bib and cited in literature.tex.*

### Resolved Gaps (for record-keeping)

The following gaps identified in the original survey have been addressed:

- **Contextual bandit citations:** `Li2010LinUCB`, `Thompson1933`, `AgrawalGoyal2013`, `AbbasiYadkori2011`, `Russo2018Thompson` — all now present in references.bib and cited in literature.tex line 18.
- **Myerson (1981):** Now present in references.bib (line 21) and cited in literature.tex line 8.
- **Autobidding citations:** `Deng2022FPAEfficiency`, `Liaw2022EfficiencyNonTruthful`, `Liaw2024EfficiencyBudget`, `Mehta2022AuctionDesign` — all now cited in literature.tex.

### Remaining Gaps

**res3.tex has zero citations.** The bandit results section (Experiment 3a/3b) contains no `\cite` commands. A reader would expect at minimum a citation to the source algorithms (Li et al. 2010 for LinUCB, Thompson 1933 or Russo et al. 2018 for Thompson Sampling) in the results discussion.

**Unused high-relevance entries (2 remaining):**

| Key | Relevance to Paper |
|---|---|
| `Balseiro2021RobustAuction` | Robust auction design under autobidding |
| `Conitzer2021SPPE` | Multiplicative pacing equilibria (complements Conitzer2022FPPE) |

---

## 3. Experiment-to-Literature Mapping Table

**Framing rule:** "benchmarked against" or "consistent/inconsistent with the prediction of X."

| Experiment | Benchmarked Against | Inconsistent With | Not Addressed By |
|---|---|---|---|
| **Exp 1** (Q-learning, constant) | Vickrey1961 (revenue equivalence); BulowKlemperer1996 (n_bidders message holds); BanchioSkrzypacz2022 (format vulnerability) | Calvano2020 (collusion not universal; depends on factorial structure) | Systematic factorial ranking of 10 simultaneous factors |
| **Exp 2** (Q-learning, affiliated) | MilgromWeber1982 (affiliation prediction); Calvano2023 (genuine vs spurious) | MilgromWeber1982 (affiliation has no significant effect under Q-learning) | Whether tabular Q-learning can exploit continuous signal correlation |
| **Exp 3a** (LinUCB) | MilgromWeber1982 (SPA advantage emerges here); Douglas2024; Li2010LinUCB | Welfare not improved by richer algorithms (lower revenue than Q-learning despite faster convergence) | res3.tex has zero citations (bandit source algorithms now in bib but not cited in results section) |
| **Exp 3b** (Thompson) | Same as 3a; Thompson1933; AgrawalGoyal2013 | Reserve price effect reverses sign vs Exp 1 | res3.tex has zero citations |
| **Exp 4a** (Dual pacing) | Balseiro2019Learning (regret); Aggarwal2019/Gaitonde2023 (PoA bound); PaesLeme2024 (drift prediction); FikiorisTardos2023 (FPA welfare) | PaesLeme2024 (near-zero drift falsifies progressive suppression prediction) | Seller revenue under budget-mediated suppression; false-positive enforcement risk |
| **Exp 4b** (PI pacing) | Same theoretical benchmarks as 4a | Consistent with 4a; FPA dominates revenue | PI controller pacing in auctions not previously studied |

---

## 4. Recommendations for literature.tex Edits

### Completed (2026-03-18)

1. ~~**Add foundational bandit citations**~~ — DONE. `Li2010LinUCB`, `Thompson1933`, `AgrawalGoyal2013`, `AbbasiYadkori2011`, `Russo2018Thompson` added to references.bib and cited in literature.tex.

2. ~~**Cite Myerson (1981) in Sec 2.1**~~ — DONE. Now cited in literature.tex line 8.

3. ~~**Incorporate unused autobidding references in Sec 2.3**~~ — DONE. `Deng2022FPAEfficiency`, `Liaw2022EfficiencyNonTruthful`, `Liaw2024EfficiencyBudget`, `Mehta2022AuctionDesign` now cited in literature.tex.

### Remaining

4. **Add a citation to res3.tex** (currently zero citations in the bandit results section) — STILL OPEN

5. **Clean up bib file:** remove `DengVarious` and `Rawat2025AlgorithmicCollusion` placeholders — check if still present

### Low Priority

6. Consider citing `AtheyImbens2016` in experimental design section
7. Consider citing `BenekeMackenrodt2021` in legal section

---

## 5. Algorithmic Collusion (Q-Learning) Literature Mapping (2026-03-18)

*Framing notes for Sec 2.2 rewrite. Positions our Exp 1-3 findings within the Q-learning collusion literature.*

### The Q-Learning Canon

The literature divides into "foundational collusion" papers and a "skeptical/robustness" corrective.

**Foundational:**

- **Calvano et al. (2020):** Seminal work establishing that autonomous Q-learners sustain supra-competitive prices in pricing games via learned punishment strategies (grim trigger).
- **Banchio & Skrzypacz (2022):** Most direct predecessor to our work. Argues Q-learning agents are significantly more prone to "tacitly collusive" bid suppression in FPA than SPA.
- **Banchio & Mantegazza (2022):** Provides the "spontaneous coupling" mechanism, explaining that collusion can arise from correlated estimation errors in infrequent exploration rather than deliberate punishment schemes.

**Skeptical/Robustness:**

- **Deng, Schiffer, & Bichler (2024) / Zhang (2023):** Show collusive findings are often artifacts of tabular Q-learning and disappear under more sophisticated Deep RL architectures or heterogeneous hyperparameters.

### How We Speak to Each Paper

We approach this literature not by looking for "collusion" (which we purposefully avoid defining as "intent"), but as "learning-driven bid suppression."

- **To Banchio & Skrzypacz (2022):** We provide a direct qualification. While they find a strong "FPA penalty," our Exp 1 (Q-learning, constant values) shows this effect is negligible compared to structural factors. The "FPA penalty" they observed is not a universal property of the format but is secondary to market thickness.
- **To Calvano et al. (2020):** We shift focus from "punishment schemes" to a hierarchy of factors. Our |t|-statistic analysis shows that even if agents "collude" in the Calvano sense, the most effective remedy for the seller is not an algorithmic audit but simply adding one more bidder (the Bulow-Klemperer effect).
- **To the skeptical school (Bichler/Zhang/Lambin):** We support their skepticism regarding the robustness of the Q-learning "collusion" narrative. Our results show that the format effect is fragile and algorithm-dependent: neutral for Q-learning, negative for bandits, and positive for pacing.

### Our School of Thought: "Factorial/Structural"

**Who we support:**

- The Bulow & Klemperer (1996) tradition. Our findings empirically prove that market thickness (number of bidders) is the dominant lever for revenue, effectively "out-muscling" any algorithmic collusion effects.
- The welfare/pacing school (Fikioris & Tardos 2023) by showing that their preference for FPA holds true in budgeted environments.

**Who we qualify/challenge:**

- The "algorithm-specific mandate" school. We argue against blanket policy recommendations (like "always use SPA to prevent collusion") because such mandates derived from Q-learning experiments fail when applied to the pacing agents (autobidders) that actually dominate modern markets.

### Best Framing: Two Species of Bid Suppression

1. **Learning-driven suppression (the Q-learning focus):** Revenue shortfalls arise from agents failing to converge to competitive equilibria. In this regime, SPA is weakly superior, and the literature's concerns are somewhat valid but overstated relative to market thickness.
2. **Constraint-driven suppression (the autobidding reality):** Shortfalls arise from budget rationing. In this regime, the format effect reverses, and FPA becomes the superior revenue-generator.

### The Diagnostic Frame

We should frame our paper as a diagnostic tool for regulators. If a platform observes bid suppression, it cannot determine the remedy from outcome data alone. Our factorial framework provides the diagnostic: because the format effect reverses sign between these two species, regulators must first identify the "species" of suppression (the bidding technology) before prescribing a format change. Misdiagnosis (treating pacing agents with Q-learning remedies) is the primary policy risk we highlight.

---

## 6. Autobidding Literature Mapping (2026-03-18)

*Working notes mapping autobidding/pacing literature to Exp 4a/4b findings. For eventual rewrite of Sec 2.3.*

### Format Comparison: FPA vs SPA Under Pacing

| Paper | Key Prediction | Connection to Our Findings |
|-------|---------------|---------------------------|
| Deng et al. (2021, 2022) | FPA PoA = 1 under pure ROI constraints (2021); degrades to ~0.457 in mixed autobidding populations (2022) | Our Exp 4a/4b show format *still* matters for revenue, suggesting non-equilibrium dynamics amplify format asymmetries |
| Fikioris & Tardos (2023) | FPA welfare bounded at 2.41x optimum; SPA can be arbitrarily bad under budget constraints | Consistent with our FPA dominance in pacing; SPA's unbounded worst case may explain weaker SPA performance |
| Conitzer et al. (2022) | FPA pacing equilibria are unique and computable (Eisenberg-Gale) | High R² (0.77-0.89) in Exp 4a/4b indicates tight constraint binding and reduced unexplained variance, consistent with uniqueness prediction |
| Chen & Kroer (2024) | SPA pacing equilibria are PPAD-complete (intractable) | Computational hardness may explain why SPA pacing is less efficient in practice |

### Budget Constraints and Structural Dominance

| Paper | Key Claim | Connection |
|-------|-----------|------------|
| Aggarwal et al. (2019) | Uniform bid-scaling optimal; PoA=2 for SPA+budgets | Exp 4a/4b benchmark; agents operate near constraint-binding regime |
| Aggarwal et al. (2024 Survey) | Budget constraints are "oldest and most studied" autobidding product | Our budget_multiplier accounts for 72-94% of total effect magnitude (not R²; full-model R² is 0.77-0.89), validating theoretical emphasis |
| Balseiro & Gur (2019) | Dual pacing achieves O(sqrt(T)) regret; converges to approximate Nash in large markets | Exp 4a implements this framework; near-zero drift confirms convergence |

### Dynamics and Stability

| Paper | Key Prediction | Connection |
|-------|---------------|------------|
| Paes Leme et al. (2024) | Autobidding exhibits bi-stability and periodic orbits; pure formats converge but mixed formats (>85% SPA) show chaos | Our near-zero drift in pure FPA/SPA falsifies progressive suppression; supports format-purity stability |
| Gaitonde et al. (2023) | Gradient pacing achieves half-optimal welfare WITHOUT convergence | R² of 0.77-0.89 in Exp 4a/4b consistent with tight constraint binding even in non-converged dynamics |
| Lucier et al. (2024) | Autobidders with budget+ROI achieve half-optimal welfare for any core auction format | Format-agnostic welfare guarantee; our format difference is in revenue distribution, not welfare floor |

### Learning vs Pacing: Why the Format Reverses

| Aspect | Learning (Exp 1-3) | Pacing (Exp 4a/4b) |
|--------|-------------------|-------------------|
| Optimal format | SPA (higher revenue) | FPA (higher revenue) |
| Mechanism | Truthful auction simplifies learning | Non-truthful; gradient multipliers bind tightly |
| R² | 0.4 (Q-learning) to 0.7 (bandits) | 0.77-0.89 |
| Dominant factor | Algorithm + exploration | Budget constraint (72-94% of total effect magnitude) |
| Convergence | Non-convergent learning | Near-zero drift |
| Theory basis | Banchio & Skrzypacz (2022): FPA suppression | Fikioris-Tardos (2023): FPA welfare bounds |

### Key Insight: Algorithm Structure Overrides Value Distribution

Milgrom-Weber's linkage principle predicts SPA dominance under affiliated values. Our Exp 2-3 confirm this for learning algorithms. But Exp 4a/4b reverse the format ordering under pacing despite similar value structures. This demonstrates that algorithm choice (learning vs. pacing) is the primary driver of format effects, not value distribution properties.

### Model Fit Progression (Novel Finding)

R² increases from ~0.4 (Q-learning) to ~0.6-0.7 (bandits) to 0.77-0.89 (pacing). Not explicitly predicted by any single paper, but consistent with increasing algorithmic determinism and constraint tightness. Pacing algorithms follow gradient trajectories with tight budget/RoS binding; learning algorithms exhibit emergent exploratory dynamics with high unexplained variance.

### Papers to Cite More Prominently in Sec 2.3 Rewrite

Currently cited but underutilized:
- `Deng2022FPAEfficiency` — FPA efficiency under mixed autobidding populations
- `Liaw2022EfficiencyNonTruthful` / `Liaw2024EfficiencyBudget` — tighter welfare bounds
- `Mehta2022AuctionDesign` — randomization improves beyond VCG
- `ChenKroer2024` / `ChenLi2025` — PPAD-hardness of SPA pacing (computational rationale for FPA preference)
- `Conitzer2021SPPE` — SPA pacing equilibria: existence but non-uniqueness (contrast with FPA uniqueness)
- `ColiniBaldeschi2025` — type-dependent welfare bounds
- `Alimohammadi2023` — incentive compatibility fails under autobidding
- `BalseiroBhawalkar2024` — practical min-pacing architecture

### NOTE: Quote Verification Status

Quotes labeled "Direct Quote" in the full mapping document (/tmp/literature_mapping.md) were extracted from transcription .md files or the Aggarwal survey transcription. Several are paraphrased from the extended_summary_of_literature.md rather than verbatim. Verify against actual transcriptions before using in paper prose.

---

## 7. Three Schools of Autobidding and Our Bridge Role (2026-03-18)

*Framing notes for the autobidding subsection rewrite. Our Exp 4a/4b serve as an empirical bridge testing the theoretical claims of three autobidding schools against each other.*

### A. The "Equilibrium and Computation" School

**Core idea:** Existence, uniqueness, and complexity of pacing equilibria.

**Key papers:** Conitzer et al. (2021, 2022), Chen & Kroer (2024).

**Their argument:** FPA is "theoretically superior" for autobidding markets because it guarantees a unique, efficiently computable equilibrium (via the Eisenberg-Gale program), whereas SPA is PPAD-hard to solve and can have multiple, unstable equilibria.

**Connection to us:** Exp 4a/4b provide empirical validation for the "FPA is superior" claim. While they argue from computational complexity, we show that in a dynamic laboratory, FPA actually delivers higher revenue and efficiency when bidders are budget-constrained pacing controllers.

### B. The "Price of Anarchy" School

**Core idea:** Worst-case welfare guarantees when agents are value maximizers (autobidders) rather than utility maximizers.

**Key papers:** Deng et al. (2022), Liaw et al. (2024), Lucier et al. (2024).

**Their argument:** Autobidding fundamentally changes the efficiency of auction formats. Liaw et al. (2024) show that under strict budget constraints, FPA is optimal among deterministic mechanisms, but the PoA can degrade significantly if budgets are tight.

**Connection to us:** We bridge their "worst-case" theory to "average-case" simulation. We confirm their intuition that budget tightness is a first-order determinant of outcomes, but our factorial analysis adds a ranking they lack: the number of bidders mitigates PoA concerns more effectively than changing the auction format itself.

### C. The "Learning and Regret" School

**Core idea:** How agents adapt pacing multipliers over time using online learning (Dual Mirror Descent).

**Key papers:** Balseiro & Gur (2019), Gaitonde et al. (2023), Balseiro et al. (2024 Field Guide).

**Their argument:** Simple gradient-based pacing algorithms can achieve near-optimal liquid welfare even without reaching equilibrium. Focus is on individual agent performance.

**Connection to us:** We move from the individual to the system. We use their algorithms (Balseiro-style dual multipliers) but test them in a factorial game environment. Our contribution is showing that while these algorithms are individually robust, their interaction with the auction format creates the sign reversal.

### The Bridge: Resolving the Collusion vs. Autobidding Conflict

| Setting | Collusion School Prediction | Autobidding School Prediction | Our Finding |
|---------|----------------------------|------------------------------|-------------|
| Format choice | SPA is better (FPA facilitates collusion) | FPA is better (SPA is unstable/complex) | Both are right, but technology-specific |
| Bidding tech | Focuses on Q-learning | Focuses on pacing controllers | Sign reversal: Exp 1-3 supports Collusion School; Exp 4a/4b supports Autobidding School |

### The "Revenue Equivalence" Connection

Standard theory (Myerson 1981) predicts FPA and SPA are equal. Recent autobidding theory (Balseiro, Kumar, Kroer 2023) argues that in large markets, revenue equivalence is restored even with budgets. **Our challenge:** In finite, small-scale auctions (typical of specific ad-slots or niches), equivalence fails spectacularly. Format choice matters because the technology (pacing vs. learning) interacts with the payment rule differently.

### The "Structural Dominance" Connection

We provide an empirical answer to the Bulow-Klemperer question in the age of algorithms.

- **The literature says:** One extra bidder is better than an optimal reserve price (Bulow & Klemperer 1996).
- **We say:** One extra bidder is 1.5 to 28 times more impactful than choosing between FPA and SPA, regardless of whether the bidders are Q-learners, bandits, or pacing controllers. This reunites the fragmented autobidding and collusion schools under a single structural truth: market structure beats mechanism design.

**NOTE**: The "3 to 28 times" language from the user's draft is overstated (actual range 1.5x-28x). Use qualitative language ("consistently dominates," "substantially larger") when rewriting for the paper.

### Priority Citations for Sec 2.3 Rewrite

1. **Conitzer et al. (2022):** Support why FPA performs well in Exp 4a/4b (uniqueness, computability).
2. **Balseiro & Gur (2019):** Ground the choice of pacing algorithm.
3. **Paes Leme et al. (2024):** Complex dynamics / instability in SPA pacing.
4. **Liaw et al. (2024):** Budget constraints degrade first-price efficiency (theoretical basis for our budget_multiplier dominance).

---

## 8. Positioning Against All Literatures (2026-03-18)

*Unified positioning strategy for the paper. Shows how our findings speak to all four literatures simultaneously.*

### Core Thesis: Two Species of Bid Suppression

The central organizing principle of the paper is that bid suppression in algorithmic auctions arises from two fundamentally different mechanisms, and the optimal auction format depends on which mechanism is operative.

1. **Learning-driven suppression (Experiments 1-3).** Revenue shortfalls arise from agents failing to converge to competitive equilibria. Algorithms explore too little, coordinate on low bids through shared learning dynamics, or suppress bids through strategic retaliation. In this regime, SPA weakly dominates FPA for revenue, consistent with the collusion literature's predictions.

2. **Constraint-driven suppression (Experiments 4a-4b).** Revenue shortfalls arise from budget rationing. Pacing controllers shade bids to satisfy hard budget or return-on-spend constraints, not because of strategic coordination. In this regime, FPA dominates SPA for revenue, consistent with the autobidding literature's predictions.

The sign reversal of the auction format coefficient between these two regimes is the paper's central empirical finding.

### Main Arguments

**A. Structural dominance.** Across all six experiments and both species of suppression, the number of bidders (market thickness) and budget tightness consistently dominate auction format choice in magnitude. This validates Bulow & Klemperer (1996) in the algorithmic setting and suggests that structural market parameters are more powerful levers than mechanism design choices.

**B. Fragile collusion narrative.** The Q-learning collusion findings that motivate regulatory concern are not robust. Auction format is statistically insignificant in Exp 1 (the canonical Q-learning setup). The "FPA penalty" documented by Banchio & Skrzypacz (2022) emerges only with more sophisticated algorithms (bandits in Exp 3a) and reverses entirely for pacing (Exp 4a-4b). Recent work by Bichler et al. (2024) and Zhang (2023) confirms that collusion findings are fragile across hyperparameters and algorithm pairings.

**C. Autobidding reversal.** For the pacing algorithms that dominate real-world advertising auctions, FPA produces higher revenue and efficiency than SPA. This is consistent with the computational complexity argument (Conitzer et al. 2022; Chen & Kroer 2024) and worst-case welfare bounds (Fikioris & Tardos 2023), but provides the first factorial experimental evidence in a repeated-game setting.

**D. False positive risk.** Both species produce below-equilibrium revenue, making them observationally indistinguishable in outcome data alone. However, they require opposite format remedies. A regulator diagnosing "collusion" from outcome data and mandating SPA (the learning-driven remedy) would harm a market dominated by pacing agents where FPA is superior. Our framework provides the diagnostic by showing that the "species" of suppression must be identified before prescribing a format change.

### Positioning vs. Auction Theory (Sec 2.1)

| Classical Prediction | Our Finding | Status |
|---|---|---|
| Revenue equivalence under IPV (Vickrey 1961) | Exp 1: FPA mean 0.819 vs SPA mean 0.813, gap < 1%, not significant | Consistent |
| SPA weakly dominates under affiliation (Milgrom & Weber 1982) | Exp 2: not significant for Q-learning; Exp 3a: SPA premium of 19% for LinUCB | Qualified — holds only for sufficiently sophisticated algorithms |
| Additional bidder > optimal reserve (Bulow & Klemperer 1996) | n_bidders dominates auction_type by 1.5x to 28x across all experiments | Strongly confirmed; extended to algorithmic setting |
| Optimal mechanism design (Myerson 1981) | Format choice matters less than structural parameters across all experiments | Consistent with structural view |

### Positioning vs. Algorithmic Collusion (Sec 2.2)

| Literature Claim | Our Finding | Status |
|---|---|---|
| Q-learning sustains collusion via punishment (Calvano et al. 2020) | Not contradicted, but collusion effect dwarfed by structural factors (n_bidders 28x larger) | Contextualized |
| FPA more prone to bid suppression (Banchio & Skrzypacz 2022) | Exp 1: not significant; Exp 3a: confirmed (SPA premium 19%); Exp 4a/4b: reversed | Technology-specific, not universal |
| Collusion fragile across algorithms (Bichler et al. 2024; Zhang 2023) | Supported: format effect changes sign, magnitude, and significance across algorithm classes | Confirmed |
| Stochastic exploration disrupts collusion (Douglas 2024) | Consistent with Exp 3a/3b bandit results showing exploration-dependent outcomes | Consistent |
| Noise injection as remedy (Zhang 2025) | Our finding that structural factors dominate supports this class of market-design interventions | Complementary |

### Positioning vs. Autobidding Literature (Sec 2.3)

| Literature Claim | Our Finding | Status |
|---|---|---|
| FPA pacing equilibria unique and computable (Conitzer et al. 2022) | FPA higher revenue in Exp 4a/4b; high R² (0.77-0.89) indicates tight constraint binding and reduced unexplained variance | Empirically supported |
| SPA pacing is PPAD-hard (Chen & Kroer 2024) | SPA produces lower revenue and efficiency in Exp 4a/4b | Consistent with computational hardness producing worse outcomes |
| PoA degrades with budget tightness (Liaw et al. 2024) | budget_multiplier accounts for 72-94% of total effect magnitude in Exp 4a/4b (full-model R² = 0.77-0.89) | Confirmed; budget tightness is the first-order effect |
| Gradient pacing achieves half-optimal welfare (Gaitonde et al. 2023) | Exp 4a/4b welfare R² = 0.80; high explanatory power from constraints alone | Consistent |
| Autobidding can exhibit bi-stability (Paes Leme et al. 2024) | Exp 4a drift R² = 0.12 with near-zero cell estimates; falsifies progressive suppression | Contradicted |
| FPA welfare within 2.41x optimum (Fikioris & Tardos 2023) | FPA outperforms SPA in revenue and efficiency under pacing | Confirmed in finite-game setting |

### Positioning vs. Legal/Policy Literature (Sec 2.4)

| Legal Claim | Our Finding | Implication |
|---|---|---|
| Agreement gap prevents prosecution (Mehra 2016) | Pacing produces "collusive" outcomes without coordination | Widens the gap: non-strategic mechanisms also produce below-equilibrium revenue |
| Per se rules for collusion-facilitating algorithms (Harrington 2018) | Format effect is technology-specific; no single rule works | Per se rules must specify which technology class they target |
| Calibrated regret audits (Hartline 2024, 2025) | Sign reversal means audit must be validated against multiple technology classes | Audit fragility: same outcome, opposite remedies |
| Format mandates to prevent collusion (Banchio & Skrzypacz 2022) | Mandating SPA based on Q-learning evidence backfires for pacing agents | Policy risk from technology-specific mandates |
| Structural remedies (competition) most robust (Bulow & Klemperer 1996) | n_bidders dominates format choice 1.5x-28x across all experiments | Strongest support: competition is the universal lever |

### Summary Comparison Table

| Dimension | Learning-Driven (Exp 1-3) | Constraint-Driven (Exp 4a-4b) |
|---|---|---|
| Revenue-superior format | SPA (premium 9-19%; 8.7% in Exp 3b, 11% in Exp 2, 19% in Exp 3a) | FPA (premium 9-29%) |
| Dominant factor | n_bidders | budget_multiplier |
| R² range | 0.40-0.69 | 0.77-0.89 |
| Format |t| | 0.6-12.5 | 3.6-11.0 |
| Mechanism | Exploration failure / coordination | Budget rationing |
| Convergence | Slow (55k-89k episodes) | N/A (constraint-binding) |
| Theory alignment | Collusion school (Banchio & Skrzypacz) | Autobidding school (Conitzer, Fikioris-Tardos) |
| Policy implication | SPA may help; competition helps more | FPA is better; competition helps most |

---

## 9. Ten Key Claims With Supporting Evidence (2026-03-18)

*Each claim references specific LaTeX macros from `paper/numbers.tex` for traceability to source data. Macro names are given in backticks.*

### Claim 1: Market thickness is the universal dominant factor in unconstrained settings

n_bidders is the highest-ranked main effect in every learning-driven experiment.

| Experiment | n_bidders |t| | Macro | Rank |
|---|---|---|---|
| Exp 1 (Q-learning) | 17.8 | `\ExpOneRevTopT` | #1 of 55 effects |
| Exp 2 (Q-learning, affiliated) | 6.5 | `\ExpTwoRevFactorTOne` | #1 of 15 effects |
| Exp 3a (LinUCB) | 33.9 | `\ExpThreeARevTopT` | #1 of 36 effects |
| Exp 3b (Thompson) | 8.7 | `\ExpThreeBRevTopT` | #1 of 21 effects |

Moving from 2 to 4 bidders increases revenue by 21.3% of grand mean in Exp 1 (`\ExpOneRevTopPctEffect`) and 57.0% in Exp 3a (`\ExpThreeARevTopPctEffect`).

### Claim 2: Budget tightness is the universal dominant factor in constrained settings

budget_multiplier is the highest-ranked main effect in both pacing experiments.

| Experiment | budget_multiplier |t| | Macro | % of total effect magnitude |
|---|---|---|---|
| Exp 4a (Dual Pacing) | 39.4 | `\ExpFourARevTopT` | 94.3% (`\ExpFourARevTopPctEffect`) |
| Exp 4b (PI Pacing) | 31.7 | `\ExpFourBRevTopT` | 72.1% (`\ExpFourBRevTopPctEffect`) |

**Note:** These percentages represent the top factor's share of the summed absolute effect magnitudes, not R² or partial variance. Full-model R² values are 0.889 (Exp 4a) and 0.768 (Exp 4b).

### Claim 3: Auction format effect is negligible for basic Q-learning

In Exp 1, auction_type has |t| = 0.6 (p = 0.529), making it statistically indistinguishable from zero.

| Metric | Value | Macro |
|---|---|---|
| auction_type |t| | 0.6 | `\ExpOneRevAuctionAbsT` |
| p-value | 0.529 | `\ExpOneRevAuctionPFmt` |
| FPA mean revenue | 0.819 | `\ExpOneRevMeanFPA` |
| SPA mean revenue | 0.813 | `\ExpOneRevMeanSPA` |
| Revenue gap | 0.8% | `\ExpOneRevGapPct` |

This directly qualifies Banchio & Skrzypacz (2022), who find a strong FPA penalty. In a 10-factor factorial design, format choice ranks near the bottom.

### Claim 4: Auction format effect emerges for sophisticated learners (bandits)

In Exp 3a (LinUCB), auction_type becomes the second-ranked effect with |t| = 12.5 (p < 10^-31). SPA produces a 19% revenue premium over FPA.

| Metric | Value | Macro |
|---|---|---|
| auction_type |t| | 12.5 | `\ExpThreeARevAuctionAbsT` |
| p-value | < 10^-31 | `\ExpThreeARevAuctionPFmt` |
| FPA mean revenue | 0.411 | `\ExpThreeARevMeanFPA` |
| SPA mean revenue | 0.507 | `\ExpThreeARevMeanSPA` |
| SPA premium | 19.0% | `\ExpThreeARevGapPct` |

In Exp 3b (Thompson), the effect is weaker but still significant: |t| = 3.8 (`\ExpThreeBRevAuctionAbsT`), SPA premium 8.7% (`\ExpThreeBRevGapPct`).

### Claim 5: Auction format effect REVERSES for pacing algorithms

The sign of the auction format coefficient flips in Experiments 4a and 4b. FPA produces *higher* revenue than SPA, the opposite of the learning-driven experiments.

| Experiment | auction_type |t| | FPA premium | Macros |
|---|---|---|---|
| Exp 4a (Dual Pacing) | 3.6 | 9.0% | `\ExpFourARevAuctionAbsT`, `\ExpFourARevGapPct` |
| Exp 4b (PI Pacing) | 11.0 | 28.7% | `\ExpFourBRevAuctionAbsT`, `\ExpFourBRevGapPct` |

FPA mean revenue in Exp 4b: 4109 (`\ExpFourBRevMeanFPA`) vs SPA mean: 3193 (`\ExpFourBRevMeanSPA`).

This contradicts the collusion literature's recommendation of FPA avoidance and supports the autobidding literature's preference for FPA.

### Claim 6: n_bidders dominates auction format by 1.5x-28x across all experiments

The ratio of |coefficient(n_bidders)| to |coefficient(auction_type)| is consistently greater than 1 across all six experiments.

| Experiment | n_bidders |t| | auction_type |t| | Ratio | Auction ratio macro |
|---|---|---|---|---|
| Exp 1 | 17.8 | 0.6 | 28.2x | `\ExpOneRevTopVsAuctionRatio` |
| Exp 2 | 6.5 | 3.2 | 2.1x | `\ExpTwoRevTopVsAuctionRatio` |
| Exp 3a | 33.9 | 12.5 | 2.7x | `\ExpThreeARevTopVsAuctionRatio` |
| Exp 3b | 8.7 | 3.8 | 2.3x | `\ExpThreeBRevTopVsAuctionRatio` |
| Exp 4a | 22.8 | 3.6 | 6.3x | computed from `\ExpFourARevNbidAbsT` / `\ExpFourARevAuctionAbsT` |
| Exp 4b | 16.1 | 11.0 | 1.5x | computed from `\ExpFourBRevNbidAbsT` / `\ExpFourBRevAuctionAbsT` |

**NOTE:** The `TopVsAuctionRatio` macros in numbers.tex compare the *top-ranked factor* (not necessarily n_bidders) to auction_type. In Exp 4a, the top factor is budget_multiplier (10.9x), not n_bidders (6.3x). The ratios above use n_bidders specifically, computed from |t|-statistic ratios. Previous drafts overstated this as "3-28 times"; the correct range is 1.5x-28x.

**CAVEAT:** In Exp 4a, the second-ranked factor is `objective` (|t| = 28.1, `\ExpFourARevObjAbsT`), which ranks above n_bidders (|t| = 22.8). The "Two Species" framework and the positioning sections do not discuss the `objective` factor, which represents whether bidders maximize value or utility. This is a gap in the narrative: the framework presents n_bidders as the universal dominant structural factor, but in Exp 4a the bidder objective is a larger effect than market thickness.

### Claim 7: Model fit increases with algorithmic determinism

R² for the revenue model increases monotonically as algorithms become more constrained and deterministic.

| Algorithm Class | Experiment | Revenue R² | Macro |
|---|---|---|---|
| Q-learning (tabular, stochastic) | Exp 1 | 0.42 | `\ExpOneRevRsq` |
| Q-learning (affiliated) | Exp 2 | 0.40 | `\ExpTwoRevRsq` |
| LinUCB (contextual bandit) | Exp 3a | 0.69 | `\ExpThreeARevRsq` |
| Thompson Sampling | Exp 3b | 0.61 | `\ExpThreeBRevRsq` |
| Dual Pacing (gradient) | Exp 4a | 0.89 | `\ExpFourARevRsq` |
| PI Pacing (controller) | Exp 4b | 0.77 | `\ExpFourBRevRsq` |

The progression from 0.42 to 0.89 reflects increasing constraint tightness. Pacing algorithms follow deterministic gradient trajectories with hard budget binding; Q-learning exhibits emergent exploratory dynamics with high unexplained variance.

### Claim 8: Near-zero drift falsifies progressive suppression

Paes Leme et al. (2024) predict that autobidding systems can exhibit bi-stability and progressive price suppression over time. Our Exp 4a data contradicts this.

| Metric | Value | Macro |
|---|---|---|
| Drift model R² | 0.12 | `\ExpFourADriftRsq` |
| Drift model F-test p-value | < 10^-5 | `\ExpFourADriftFPFmt` |

While technically significant (large N), the drift R² of 0.12 means the model explains almost none of the variation in revenue trajectories. Cell-level drift estimates are near zero (`\ExpFourADriftFP`). Revenue levels are determined by the initial constraint configuration, not by progressive learning dynamics.

In Exp 4b, the drift effect is even weaker: auction_type drift |t| = 0.4 (p = 0.709, `\ExpFourBDriftAuctionPFmt`); n_bidders drift |t| = 0.5 (p = 0.607, `\ExpFourBDriftNbidPFmt`).

### Claim 9: Affiliation prediction holds for bandits but fails for Q-learning

Milgrom & Weber (1982) predict that SPA weakly dominates FPA when bidder valuations are affiliated (correlated).

**Exp 2 (Q-learning, affiliated values):** The overall auction_type effect is marginally significant (|t| = 3.2, `\ExpTwoRevAuctionAbsT`), but the SPA premium is modest (11%, `\ExpTwoRevGapPct`). The affiliation parameter (eta) does not appear as a top-ranked main effect; n_bidders (|t| = 6.5) and state_info (|t| = 5.9) both dominate. Q-learning's tabular discretization cannot exploit continuous signal correlation.

**Exp 3a (LinUCB, affiliated values):** The auction format effect is now the second-ranked factor (|t| = 12.5), with a 19% SPA premium. The contextual bandit architecture can exploit value correlation through its linear feature model. The linkage principle prediction is confirmed, but only with sufficiently sophisticated algorithms.

**Conclusion:** Milgrom-Weber holds conditionally. The theoretical prediction requires an algorithm sophisticated enough to exploit the information structure.

### Claim 10: Pacing revenue shortfalls are indistinguishable from collusion in outcome data

Both learning-driven and constraint-driven experiments produce revenue below the competitive equilibrium benchmark. However, they have opposite format prescriptions.

| Setting | Below-equilibrium revenue? | Optimal format | Source of shortfall |
|---|---|---|---|
| Learning (Exp 1-3) | Yes | SPA | Exploration failure, strategic coordination |
| Pacing (Exp 4a-4b) | Yes | FPA | Budget rationing under fixed constraints |

A regulator observing below-equilibrium revenue cannot determine from outcome data alone whether the shortfall is caused by algorithmic coordination (requiring SPA mandate or audit) or budget rationing (requiring FPA or structural remedies). The sign reversal between Exp 1-3 and Exp 4a-4b proves that the same "symptom" (low revenue) can arise from two different "diseases" that require opposite "treatments."

This creates a false positive risk for outcome-based enforcement: if regulators mandate SPA based on the Q-learning collusion literature (Banchio & Skrzypacz 2022; Calvano et al. 2020), they would reduce revenue in pacing-dominated markets where FPA is superior. Conversely, mandating FPA based on the autobidding literature would harm markets where learning-driven agents dominate.

The only factor that unambiguously improves revenue across both species is increasing the number of bidders (Claim 6). This supports structural competition policy over format mandates.

---

## 10. Top 10 Essential Papers

A curated reading list for researchers entering this project's four core areas: algorithmic collusion, auction format design under learning agents, autobidding and pacing, and legal/regulatory responses. Each entry explains why the paper is essential and identifies the key insight most relevant to our experimental program.

### 1. Calvano, Calzolari, Denicolo, and Pastorello (2020) — "Artificial Intelligence, Algorithmic Pricing, and Collusion"

- **Why essential:** The existence proof that launched the algorithmic collusion literature. Demonstrates that Q-learning agents autonomously converge to supra-competitive prices in repeated Bertrand games, sustained by emergent punishment-and-reward strategies resembling the Folk Theorem equilibria studied in classical repeated-game theory. Every subsequent paper in the field positions itself relative to this result.
- **Key insight:** Collusion emerges from the learning dynamics themselves, not from explicit programming or communication. The punishment structure (temporary price war followed by gradual return to cooperation) arises endogenously from Q-value updates.
- **Connection to our experiments:** Experiments 1 and 2 use Q-learning agents in auction settings, directly extending Calvano et al.'s pricing-game framework to first-price and second-price auctions with factorial variation across algorithmic and market parameters.

### 2. Banchio and Skrzypacz (2022) — "Artificial Intelligence and Auction Design"

- **Why essential:** The first paper to study how auction format affects algorithmic collusion, finding a stark dichotomy: Q-learning agents collude in first-price auctions but converge to competitive outcomes in second-price auctions. Also shows that providing counterfactual feedback (minimum winning bid) eliminates collusion in FPA.
- **Key insight:** In first-price auctions, a winning bidder's incentive to win by just one bid increment facilitates re-coordination on low bids. Second-price auctions break this dynamic because truthful bidding is a dominant strategy regardless of the rival's behavior.
- **Connection to our experiments:** Our Experiments 1-3 test and extend this finding across multiple algorithms (Q-learning, LinUCB, Thompson Sampling) and market structures. The SPA advantage documented in Experiments 1-3 is consistent with Banchio and Skrzypacz's prediction, but the reversal in Experiments 4a-4b (where FPA dominates under pacing) shows the result is technology-specific.

### 3. Balseiro and Gur (2019) — "Learning in Repeated Auctions with Budgets: Regret Minimization and Equilibrium"

- **Why essential:** The foundational paper for budget-constrained autobidding. Introduces adaptive pacing via dual multiplier updates (b = v/(1+mu)) and proves convergence to approximate Nash equilibrium in large markets. Establishes the impossibility of no-regret learning relative to dynamic benchmarks when budgets bind.
- **Key insight:** A single scalar multiplier, updated based on observed expenditures, is sufficient to achieve near-optimal budget pacing. When all bidders adopt this strategy simultaneously, the resulting dynamics converge, providing theoretical grounding for the pacing algorithms deployed on real advertising platforms.
- **Connection to our experiments:** Experiment 4a implements dual pacing directly based on this framework. The budget multiplier emerges as the dominant factor (|t| = 39.4), confirming that the tightness of budget constraints overwhelms all other design choices.

### 4. Akbarpour and Li (2020) — "Credible Auctions: A Trilemma"

- **Why essential:** Provides the formal justification for the industry shift from second-price to first-price auctions in online advertising. Proves that an optimal auction can be any two of {static, strategy-proof, credible}, but not all three. The second-price auction fails credibility because the auctioneer can profitably inflate the second-highest bid.
- **Key insight:** The first-price auction with reserve is the unique credible static optimal mechanism, because the winner's payment is determined entirely by their own bid, leaving no room for auctioneer manipulation. This resolves the puzzle of why platforms abandoned the theoretically elegant second-price format.
- **Connection to our experiments:** The credibility argument motivates why first-price auctions are the relevant institutional setting for Experiments 4a-4b. Our finding that FPA dominates under pacing provides additional support for the industry transition, beyond the credibility rationale.

### 5. Roughgarden, Syrgkanis, and Tardos (2017) — "The Price of Anarchy in Auctions"

- **Why essential:** Develops the modular framework for bounding worst-case efficiency loss in auctions under incomplete information, no-regret learning, and complex multi-item settings. Proves that every Bayes-Nash equilibrium of a first-price single-item auction achieves welfare at least 1 - 1/e of the maximum, regardless of the number of bidders or degree of asymmetry.
- **Key insight:** Non-trivial efficiency guarantees can be obtained without solving for or characterizing equilibria, relying only on the best-response property. The smoothness-based approach extends from static games to no-regret learning dynamics, making it applicable to algorithmic bidding environments.
- **Connection to our experiments:** The Price of Anarchy framework provides the theoretical welfare benchmarks against which our Experiments 4a-4b measure efficiency losses. Our ratio_to_theory metric compares realized revenue to BNE predictions grounded in this tradition.

### 6. Deng, Mao, Mirrokni, and Zuo (2021) — "Towards Efficient Auctions in an Auto-bidding World"

- **Why essential:** Proposes auctions with additive boosts to improve welfare in autobidding environments, proving that VCG with value-competitive boosts approaches full efficiency. Demonstrates that welfare guarantees hold without requiring bidders to be in Nash equilibrium, only requiring feasible and undominated strategies.
- **Key insight:** Platform-side interventions (boosts that steer allocation toward higher-value bidders) can recover welfare losses caused by uniform bidding strategies under ROI and budget constraints, creating a practical mechanism design lever for ad platforms.
- **Connection to our experiments:** Our Experiments 4a-4b operate in the autobidding world this paper addresses. The welfare losses we document under pacing correspond to the efficiency gaps that boost-based mechanisms aim to close. Our finding that structural parameters (budget tightness, bidder count) dominate auction format complements their theoretical result that format alone is insufficient.

### 7. Dutting, Feng, Narasimhan, Parkes, and Ravindranath (2024) — "Optimal Auctions through Deep Learning"

- **Why essential:** Pioneers neural network-based automated mechanism design, showing that deep learning can recover all known optimal auctions and discover novel mechanisms for unsolved settings. Introduces RegretNet, which replaces hard incentive-compatibility constraints with a differentiable regret penalty. Launched the "differentiable economics" research program.
- **Key insight:** Framing mechanism design as constrained optimization over neural network parameters makes previously intractable multi-bidder multi-item auction design computationally feasible, bypassing the exponential scaling of LP-based approaches.
- **Connection to our experiments:** Represents the frontier of the mechanism design literature that our work complements from the empirical side. While Dutting et al. optimize auction rules given rational bidders, our experiments ask the converse question: given fixed auction rules, how do different learning algorithms shape market outcomes?

### 8. Harrington (2018) — "Developing Competition Law for Collusion by Autonomous Artificial Agents"

- **Why essential:** The foundational legal paper on algorithmic collusion. Identifies the "agreement gap" in competition law: under Section 1 of the Sherman Act, liability requires evidence of an agreement achieved through overt communication, but learning algorithms collude without any communication. Proposes shifting legal focus from the process (communication) to properties of the pricing algorithm itself.
- **Key insight:** Regulators could audit algorithms to determine whether they embody reward-punishment schemes, rather than searching for evidence of human coordination. This per se approach based on algorithmic properties represents a fundamental shift in competition law doctrine.
- **Connection to our experiments:** Our Claim 10 (pacing revenue shortfalls are indistinguishable from collusion in outcome data) directly challenges outcome-based regulatory approaches. If a regulator cannot distinguish budget-driven shortfalls from collusion-driven shortfalls without experimental variation, algorithm auditing as Harrington proposes becomes even more critical.

### 9. Zhang (2025) — "Too Noisy to Collude? Algorithmic Collusion Under Laplacian Noise"

- **Why essential:** Proposes a concrete, implementable regulatory remedy: injecting calibrated Laplacian noise into pooled market data disrupts cartel formation before supra-competitive prices emerge. Models cartel dynamics as a DeGroot belief-averaging process and derives the feasibility region for noise injection that disrupts collusion without undermining legitimate pricing.
- **Key insight:** Information quality is a regulatory lever. Calibrated noise slows cartel convergence because it destabilizes the belief synchronization that coordination requires, while leaving individual firms' pricing accuracy largely intact. This is an ex ante remedy that prevents collusion rather than detecting it after the fact.
- **Connection to our experiments:** Our Experiments 1-3 show that exploration noise and information feedback are among the top-ranked factors affecting collusion outcomes. Zhang's noise injection proposal is conceptually aligned with our finding that information structure mediates algorithmic coordination.

### 10. Brown and MacKay (2021) — "Competition in Pricing Algorithms"

- **Why essential:** Provides the counter-narrative to the collusion literature: pricing algorithms can raise prices through purely competitive (non-collusive) mechanisms. Using high-frequency data from major U.S. online retailers, documents that asymmetric pricing technology (some firms update hourly, others weekly) produces persistent price differences of 10-30%, and algorithmic competition increases average prices by approximately 5% through competitive best-response dynamics.
- **Key insight:** Even simple linear pricing algorithms support supra-competitive prices in Markov perfect equilibrium without any punishment schemes, because the faster firm's credible commitment to best-respond softens competition. This means supra-competitive prices are not sufficient evidence of collusion.
- **Connection to our experiments:** Reinforces our Claim 10 that below-equilibrium revenue has multiple possible causes requiring different remedies. Brown and MacKay show that supra-competitive *prices* can arise without collusion; our experiments show that sub-competitive *revenue* can arise from budget constraints rather than coordination. Together, these results caution against inferring collusion from outcome data alone.
