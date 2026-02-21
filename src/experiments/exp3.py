#!/usr/bin/env python3

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# 1) Numeric Parameter Mappings
# ---------------------------------------------------------------------
param_mappings = {
    "auction_type": {"first": 1, "second": 0},
    "algorithm": {"linucb": 0, "thompson": 1},
    "context_richness": {"minimal": 0, "rich": 1},
}

# Algorithm-specific exploration parameter mapping
# exploration_intensity "low"/"high" maps to different values per algorithm
EXPLORATION_MAP = {
    ("linucb", "low"):     0.5,   # c parameter
    ("linucb", "high"):    2.0,
    ("thompson", "low"):   0.1,   # sigma2 parameter
    ("thompson", "high"):  1.0,
}

# ---------------------------------------------------------------------
# 2) Valuation with eta (Affiliation)
# ---------------------------------------------------------------------
def get_valuation(eta, own_signal, others_signals):
    """
    Linear affiliation:
      valuation = alpha * own_signal + beta * mean(others_signals),
    alpha = 1.0 - 0.5 * eta, beta = 0.5 * eta.
    """
    alpha = 1.0 - 0.5 * eta
    beta = 0.5 * eta
    return alpha * own_signal + beta * np.mean(others_signals)

# ---------------------------------------------------------------------
# 3) Payoff with Reserve Price
# ---------------------------------------------------------------------
def get_rewards(bids, valuations, auction_type="first", reserve_price=0.0):
    """
    Returns:
      rewards: shape (n_bidders,) => payoff for each bidder
      winner_global: int => index of winner or -1 if no sale
      winner_bid: float => winning bid or 0.0 if no sale
    """
    n_bidders = len(bids)
    rewards = np.zeros(n_bidders)

    valid_indices = np.where(bids >= reserve_price)[0]
    if len(valid_indices) == 0:
        return rewards, -1, 0.0  # no sale

    valid_bids = bids[valid_indices]
    sorted_idx = np.argsort(valid_bids)[::-1]
    highest_idx_local = [sorted_idx[0]]
    highest_bid = valid_bids[sorted_idx[0]]

    # tie among top bids
    for idx_l in sorted_idx[1:]:
        if np.isclose(valid_bids[idx_l], highest_bid):
            highest_idx_local.append(idx_l)
        else:
            break

    # pick winner randomly if tie
    if len(highest_idx_local) > 1:
        winner_local = np.random.choice(highest_idx_local)
    else:
        winner_local = highest_idx_local[0]
    winner_global = valid_indices[winner_local]
    winner_bid = bids[winner_global]

    # second-highest
    if len(valid_indices) == len(highest_idx_local):
        # All valid bidders are tied at highest; in SPA use reserve price
        if auction_type != "first" and len(valid_indices) == 1:
            second_highest_bid = reserve_price
        else:
            second_highest_bid = highest_bid
    else:
        second_idx_local = None
        for idx_l in sorted_idx:
            if idx_l not in highest_idx_local:
                second_idx_local = idx_l
                break
        if second_idx_local is None:
            second_highest_bid = highest_bid
        else:
            second_highest_bid = valid_bids[second_idx_local]

    # payoff
    if auction_type == "first":
        rewards[winner_global] = valuations[winner_global] - winner_bid
    else:  # second-price
        rewards[winner_global] = valuations[winner_global] - second_highest_bid

    return rewards, winner_global, winner_bid

# ---------------------------------------------------------------------
# 4) LinUCB Bandit
# ---------------------------------------------------------------------
class LinUCB:
    """
    For each action a:
      A[a] = XᵀX + lambda*I
      b[a] = Xᵀy
    p(a) = theta_hat[a]ᵀ context + c * sqrt(contextᵀ A[a]^(-1) context)
    """
    def __init__(self, n_actions, context_dim, c, lam):
        self.n_actions = n_actions
        self.context_dim = context_dim
        self.c = c
        self.lam = lam

        self.A = [lam * np.eye(context_dim) for _ in range(n_actions)]
        self.b = [np.zeros(context_dim) for _ in range(n_actions)]

    def select_action(self, context_vec):
        p_vals = []
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta_hat = A_inv @ self.b[a]
            mean_est = theta_hat @ context_vec
            var_est = context_vec.T @ A_inv @ context_vec
            bonus = self.c * np.sqrt(var_est)
            p_vals.append(mean_est + bonus)
        return np.argmax(p_vals)

    def update(self, action, reward, context_vec):
        self.A[action] += np.outer(context_vec, context_vec)
        self.b[action] += reward * context_vec


# ---------------------------------------------------------------------
# 4b) Contextual Thompson Sampling
# ---------------------------------------------------------------------
class ContextualThompsonSampling:
    """
    Bayesian linear regression per action.
    A[a] = lambda*I + sum(x x^T), b[a] = sum(r x)
    Sample theta ~ N(A^{-1} b, sigma2 * A^{-1}), pick argmax theta^T x.
    """
    def __init__(self, n_actions, context_dim, sigma2, lam, rng=None):
        self.n_actions = n_actions
        self.context_dim = context_dim
        self.sigma2 = sigma2
        self.lam = lam
        self.A = [lam * np.eye(context_dim) for _ in range(n_actions)]
        self.b = [np.zeros(context_dim) for _ in range(n_actions)]
        self.rng = rng or np.random.default_rng()

    def select_action(self, context_vec):
        sampled_vals = []
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta_hat = A_inv @ self.b[a]
            cov = self.sigma2 * A_inv
            theta_sample = self.rng.multivariate_normal(theta_hat, cov)
            sampled_vals.append(theta_sample @ context_vec)
        return np.argmax(sampled_vals)

    def update(self, action, reward, context_vec):
        self.A[action] += np.outer(context_vec, context_vec)
        self.b[action] += reward * context_vec


# ---------------------------------------------------------------------
# 5) Theoretical BNE Revenue (Linear Affiliation)
# ---------------------------------------------------------------------
def simulate_linear_affiliation_revenue(N, eta, auction_type, M=50_000):
    """
    For each bidder i: t_i ~ U[0..1].
    BNE strategy:
      FPA => factor = ((N-1)/N)*[alpha + (N/2)*beta]
      SPA => factor = alpha + (N/2)*beta
    alpha=1-0.5*eta, beta=(0.5*eta)/(N-1).
    Return average revenue (Monte Carlo).
    """
    if N < 1:
        return 0.0

    alpha = 1.0 - 0.5 * eta
    beta = 0.5 * eta / max(N - 1, 1.0)

    if auction_type == "first":
        factor = ((N - 1) / float(N)) * (alpha + (N / 2.0) * beta)
    else:
        factor = alpha + (N / 2.0) * beta

    rev_sum = 0.0
    for _ in range(M):
        t = np.random.rand(N)
        bids = factor * t
        max_bid = np.max(bids)
        top_idx = np.where(np.isclose(bids, max_bid))[0]
        winner = np.random.choice(top_idx)

        if auction_type == "first":
            price = max_bid
        else:
            if len(top_idx) == 1:
                other_bids = np.delete(bids, winner)
                second = np.max(other_bids) if len(other_bids) else 0.0
                price = second
            else:
                price = max_bid
        rev_sum += price

    return rev_sum / M

# ---------------------------------------------------------------------
# 6) Single Bandit Experiment
# ---------------------------------------------------------------------
def run_bandit_experiment(
    eta, auction_type, lam,
    n_bidders, reserve_price, max_rounds,
    algorithm="linucb",
    exploration_intensity="low",
    context_richness="minimal",
    seed=0,
    progress_callback=None
):
    """
    Run a single contextual bandit auction experiment.

    Args:
        algorithm: "linucb" or "thompson"
        exploration_intensity: "low" or "high" (maps to algorithm-specific values)
        context_richness: "minimal" (own signal) or "rich" (signal + median + winner + win_rate)
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    n_val_bins = 6
    n_bid_bins = 6
    actions = np.linspace(0, 1, n_bid_bins)

    # Build context dimension based on context_richness
    if context_richness == "rich":
        context_dim = 4  # own_signal, median_others_bid, past_winner_bid, win_rate
    else:
        context_dim = 1  # own_signal only

    # Resolve exploration parameter from map
    explore_val = EXPLORATION_MAP[(algorithm, exploration_intensity)]

    if algorithm == "linucb":
        bandits = [LinUCB(n_actions=n_bid_bins, context_dim=context_dim,
                          c=explore_val, lam=lam)
                   for _ in range(n_bidders)]
    elif algorithm == "thompson":
        bandits = [ContextualThompsonSampling(n_actions=n_bid_bins, context_dim=context_dim,
                                              sigma2=explore_val, lam=lam,
                                              rng=np.random.default_rng(seed + i))
                   for i in range(n_bidders)]

    revenues = []
    winners_list = []
    winning_bids_list = []
    round_history = []
    no_sale_count = 0

    past_bids = np.zeros(n_bidders)
    past_winner_bid = 0.0
    win_counts = np.zeros(n_bidders)

    def build_context(bidder_i, signals, ep):
        ctx_list = [signals[bidder_i]]  # always include own signal
        if context_richness == "rich":
            med_bid = np.median(np.delete(past_bids, bidder_i)) if n_bidders > 1 else 0.0
            ctx_list.append(med_bid)
            ctx_list.append(past_winner_bid)
            win_rate = win_counts[bidder_i] / max(1, ep)
            ctx_list.append(win_rate)
        return np.array(ctx_list)

    for ep in range(max_rounds):
        # Progress callback every 1000 rounds
        if progress_callback and ep % 1000 == 0:
            progress_callback(current=ep, total=max_rounds)

        signals = np.random.randint(n_val_bins, size=n_bidders) / (n_val_bins - 1)
        valuations = np.zeros(n_bidders)
        for i in range(n_bidders):
            others_signals = np.delete(signals, i)
            valuations[i] = get_valuation(eta, signals[i], others_signals)

        # Choose bids
        chosen_actions = []
        chosen_bids = []
        for i in range(n_bidders):
            ctx_vec = build_context(i, signals, ep)
            a_i = bandits[i].select_action(ctx_vec)
            chosen_actions.append((i, a_i, ctx_vec))
            chosen_bids.append(actions[a_i])

        chosen_bids = np.array(chosen_bids)
        rew, winner, highest_bid = get_rewards(chosen_bids, valuations, auction_type, reserve_price)

        # Update bandits
        for (i, a_idx, ctx_vec) in chosen_actions:
            bandits[i].update(a_idx, rew[i], ctx_vec)

        # Revenue (auction-type aware)
        valid_bids = chosen_bids[chosen_bids >= reserve_price]
        if len(valid_bids) == 0:
            revenue_t = 0.0
        elif auction_type == "first":
            revenue_t = float(np.max(valid_bids))
        else:  # second-price: seller gets second-highest bid
            if len(valid_bids) >= 2:
                revenue_t = float(np.sort(valid_bids)[-2])
            else:
                revenue_t = float(valid_bids[0])
        revenues.append(revenue_t)
        winning_bids_list.append(highest_bid)

        if winner == -1:
            no_sale_count += 1
        else:
            winners_list.append(winner)
            win_counts[winner] += 1

        # log round
        for i in range(n_bidders):
            round_history.append({
                "episode": ep,
                "bidder_id": i,
                "signal": signals[i],
                "chosen_bid": chosen_bids[i],
                "reward": rew[i],
                "is_winner": (i == winner)
            })

        # Update memory
        past_bids = chosen_bids
        past_winner_bid = highest_bid if winner != -1 else 0.0

    # Final progress update
    if progress_callback:
        progress_callback(current=max_rounds, total=max_rounds)

    # After max_rounds
    # time_to_converge: unify with Exp.1/2 => rolling window of 1000, stay in ±5% band
    window_size = 1000
    import pandas as pd
    rev_series = pd.Series(revenues)

    if len(revenues) >= window_size:
        avg_rev_last_1000 = np.mean(revenues[-window_size:])
    else:
        avg_rev_last_1000 = np.mean(revenues)

    roll_avg = rev_series.rolling(window=window_size).mean()
    final_rev = avg_rev_last_1000
    lower_band = 0.95 * final_rev
    upper_band = 1.05 * final_rev
    time_to_converge = max_rounds
    for t in range(len(revenues) - window_size):
        window_val = roll_avg.iloc[t + window_size - 1]
        if lower_band <= window_val <= upper_band:
            stay_in_band = True
            for j in range(t + window_size, len(revenues) - window_size):
                v_j = roll_avg.iloc[j + window_size - 1]
                if not (lower_band <= v_j <= upper_band):
                    stay_in_band = False
                    break
            if stay_in_band:
                time_to_converge = t + window_size
                break

    # average regret
    regrets = [1.0 - r for r in revenues]
    avg_regret_seller = np.mean(regrets)

    # no_sale_rate
    no_sale_rate = no_sale_count / max_rounds

    # price_volatility
    price_volatility = np.std(winning_bids_list) if len(winning_bids_list) else 0.0

    # winner_entropy
    if len(winners_list) == 0:
        winner_entropy = 0.0
    else:
        unique_winners, counts = np.unique(winners_list, return_counts=True)
        p = counts / counts.sum()
        winner_entropy = -np.sum(p * np.log(p + 1e-12))

    summary = {
        "avg_rev_last_1000": avg_rev_last_1000,
        "time_to_converge": time_to_converge,
        "avg_regret_seller": avg_regret_seller,
        "no_sale_rate": no_sale_rate,
        "price_volatility": price_volatility,
        "winner_entropy": winner_entropy
    }
    return summary, revenues, round_history, bandits  # bandits hold final models
