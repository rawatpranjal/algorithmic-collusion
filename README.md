# Algorithmic Collusion

This repository accompanies **"Designing Auctions when Algorithms Learn to Bid: The Critical Role of Payment Rules"**  
[arXiv link](https://arxiv.org/abs/2306.09437)

## Abstract

Algorithms are increasingly being used to automate participation in online markets. Banchio and Skrzypacz (2022) demonstrate how exploration under identical valuation in first-price auctions may lead to spontaneous coupling into sub-competitive bidding. However, it is an open question if these findings extend to affiliated values, optimal exploration, and specifically which algorithmic details play a role in facilitating algorithmic collusion. This paper contributes to the literature by generating robust stylized facts to cover these gaps. I conduct a set of fully randomized experiments in a controlled laboratory setup and apply double machine learning to estimate granular conditional treatment effects of auction design on seller revenues. I find that first-price auctions lead to lower seller revenues, slower convergence, and higher seller regret under identical values, affiliated values, and also under optimal exploration. There is more possibility of such tacit collusion under fewer bidders, lower learning rates, and higher discount factors. This evidence suggests that programmatic auctions, e.g. the Google Ad Exchange, which depend on first-price auctions, might be susceptible to coordinated bid suppression and significant revenue losses.

## Contents
- **experiment1**: Q-learning, constant valuations  
- **experiment2**: Q-learning, stochastic valuations with affiliation  
- **experiment3**: UCB and linear contextual bandits, stochastic valuations  
- **analysis**: Illustrative scripts for statistical/ML analyses (ATE, GATE, CATE)

## Citation
If you find this work useful, please cite the arXiv preprint above.  

Rawat, P. (2023). Designing Auctions when Algorithms Learn to Bid: The critical role of Payment Rules. arXiv preprint arXiv:2306.09437.
