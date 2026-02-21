# Collusion Analysis - none Information Regime

## 1. Summary

| Auction Type   |   Collusion Score | Interpretation                                     |
|:---------------|------------------:|:---------------------------------------------------|
| FPA            |                 1 | Weak evidence of collusion, likely just 'coupling' |
| SPA            |                 1 | Weak evidence of collusion, likely just 'coupling' |

## 2. Key Findings

| Auction Type   | Key Findings                                                                  |
|:---------------|:------------------------------------------------------------------------------|
| FPA            | High best-response consistency (1.00): Agents have optimized their strategies |
| SPA            | High best-response consistency (1.00): Agents have optimized their strategies |

## 3. Retaliation Ratio

| Agent       |   Retaliation Count |   Higher Count |   Ratio |
|:------------|--------------------:|---------------:|--------:|
| FPA_bidder0 |                   0 |              0 |       0 |
| FPA_bidder1 |                   0 |              0 |       0 |
| SPA_bidder0 |                   0 |              0 |       0 |
| SPA_bidder1 |                   0 |              0 |       0 |

## 4. Best Response Consistency

| Agent       |   Valid States |   BR Violations |   Consistency |
|:------------|---------------:|----------------:|--------------:|
| FPA_bidder0 |              4 |               0 |             1 |
| FPA_bidder1 |              4 |               0 |             1 |
| SPA_bidder0 |              4 |               0 |             1 |
| SPA_bidder1 |              4 |               0 |             1 |

## 5. Conditional Correlation

| Auction Type   |   Bidder0 next vs Bidder1 |   Bidder1 next vs Bidder0 |   Observations |
|:---------------|--------------------------:|--------------------------:|---------------:|
| FPA            |                    0.0562 |                    0.0363 |            999 |
| SPA            |                    0.1364 |                    0.0555 |            999 |

## 6. Deviation Response

| Test Case            | Initial State   | Next State   | Best Action   | Is Retaliatory   |
|:---------------------|:----------------|:-------------|:--------------|:-----------------|
| FPA_bidder0_state0.2 | (0.2,)          | N/A          | None          | No               |
| FPA_bidder1_state0.2 | (0.2,)          | N/A          | None          | No               |
| SPA_bidder0_state0.2 | (0.2,)          | N/A          | None          | No               |
| SPA_bidder1_state0.2 | (0.2,)          | N/A          | None          | No               |
| FPA_bidder0_state0.4 | (0.4,)          | N/A          | None          | No               |
| FPA_bidder1_state0.4 | (0.4,)          | N/A          | None          | No               |
| SPA_bidder0_state0.4 | (0.4,)          | N/A          | None          | No               |
| SPA_bidder1_state0.4 | (0.4,)          | N/A          | None          | No               |
| FPA_bidder0_state0.6 | (0.6,)          | N/A          | None          | No               |
| FPA_bidder1_state0.6 | (0.6,)          | N/A          | None          | No               |
| SPA_bidder0_state0.6 | (0.6,)          | N/A          | None          | No               |
| SPA_bidder1_state0.6 | (0.6,)          | N/A          | None          | No               |
| FPA_bidder0_state0.8 | (0.8,)          | N/A          | None          | No               |
| FPA_bidder1_state0.8 | (0.8,)          | N/A          | None          | No               |
| SPA_bidder0_state0.8 | (0.8,)          | N/A          | None          | No               |
| SPA_bidder1_state0.8 | (0.8,)          | N/A          | None          | No               |

