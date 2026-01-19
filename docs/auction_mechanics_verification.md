# Auction Mechanics Verification Report

## Goal
Verify that auction payment/reward mechanics match the expected rules across all experiment files.

## Expected Rules

| Rule | Description |
|------|-------------|
| **FPA Winner** | Highest bidder wins, pays their bid, profit = value - bid |
| **SPA Winner** | Highest bidder wins, pays 2nd highest bid, profit = value - 2nd_bid |
| **Losers** | All non-winners receive 0 reward |
| **Negative Rewards** | Allowed when bid > value |
| **Ties** | Broken randomly |

## Verification Results

### EXP1: `immediate_reward_and_next_state_any()` (lines 104-124)

**File:** `src/experiments/exp1.py`

**Key code:**
```python
if auction_type == "FPA":
    price = max_bid
else:  # SPA
    price = bids[loser]  # second highest

r0 = 1.0 - price if winner == 0 else 0.0
r1 = 1.0 - price if winner == 1 else 0.0
```

- Fixed valuation = 1.0 (hardcoded)
- FPA: Winner pays `max_bid`, profit = `1.0 - max_bid`
- SPA: Winner pays `loser_bid`, profit = `1.0 - loser_bid`
- Losers: 0.0
- Ties: `np.random.choice([0, 1])`
- Negative rewards: Possible if bid > 1.0

**Status: CORRECT**

---

### EXP2: `get_rewards()` (lines 31-83)

**File:** `src/experiments/exp2.py`

**Key code:**
```python
rewards = np.zeros(n_bidders)  # Losers get 0

if auction_type == "first":
    rewards[winner_global] = valuations[winner_global] - winner_bid
else:  # second-price
    rewards[winner_global] = valuations[winner_global] - second_highest_bid
```

- FPA: `valuation - winner_bid`
- SPA: `valuation - second_highest_bid`
- Losers: 0 (zeros array, only winner updated)
- Ties: `np.random.choice(highest_idx_local)`
- Negative rewards: Possible if bid > valuation
- Reserve price: Properly enforced (no-sale returns all zeros)

**Status: CORRECT**

---

### EXP3: `get_rewards()` (lines 42-96)

**File:** `src/experiments/exp3.py`

**Key code:**
```python
rewards = np.zeros(n_bidders)  # Losers get 0

if auction_type == "first":
    rewards[winner_global] = valuations[winner_global] - winner_bid
else:  # second-price
    rewards[winner_global] = valuations[winner_global] - second_highest_bid
```

Nearly identical to EXP2 with same mechanics:

- FPA: `valuation - winner_bid`
- SPA: `valuation - second_highest_bid`
- Losers: 0 (zeros array, only winner updated)
- Ties: `np.random.choice(highest_idx_local)`
- Negative rewards: Possible if bid > valuation
- Reserve price: Properly enforced (no-sale returns all zeros)

**Status: CORRECT**

---

## Summary

| Rule | EXP1 | EXP2 | EXP3 |
|------|:----:|:----:|:----:|
| FPA: Winner = highest bidder | ✓ | ✓ | ✓ |
| FPA: Winner pays their bid | ✓ | ✓ | ✓ |
| FPA: Profit = value - bid | ✓ | ✓ | ✓ |
| SPA: Winner = highest bidder | ✓ | ✓ | ✓ |
| SPA: Winner pays 2nd highest | ✓ | ✓ | ✓ |
| SPA: Profit = value - 2nd bid | ✓ | ✓ | ✓ |
| Losers get 0 reward | ✓ | ✓ | ✓ |
| Negative rewards possible | ✓ | ✓ | ✓ |
| Ties broken randomly | ✓ | ✓ | ✓ |

## Conclusion

**All auction mechanics are correctly implemented across all three experiments.**

Key differences between experiments:
- **Exp1**: 2-bidder only, fixed valuation (1.0), no reserve price
- **Exp2/Exp3**: N-bidder, variable valuations, reserve price support, affiliated values
