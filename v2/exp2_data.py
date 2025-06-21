#!/usr/bin/env python3

import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
from tqdm import trange

DEBUG = False
def dprint(*args):
    if DEBUG:
        print(*args)

# --------------------------------------------------------------
# 1) Valuation with eta (normalized)
# --------------------------------------------------------------
def get_valuation(eta, own_signal, others_signals):
    """
    Valuation with affiliation parameter eta on normalized signals in [0, 1].
    alpha(eta) = 1 - 0.5*eta
    beta(eta)  = 0.5*eta
    
    v_i = alpha(eta)*own_signal + beta(eta)*mean(others_signals)
    Since own_signal and others_signals are in [0,1], v_i remains in [0,1].
    """
    alpha = 1.0 - 0.5 * eta
    beta = 0.5 * eta
    return alpha * own_signal + beta * np.mean(others_signals)

# --------------------------------------------------------------
# 2) Payoffs
# --------------------------------------------------------------
def get_payoffs(bids, valuations, auction_type):
    """
    Calculate payoffs for multiple bidders.
    First-price: winner pays own bid.
    Second-price: winner pays second-highest bid.
    Ties are broken randomly.
    """
    n_bidders = len(bids)
    rewards = np.zeros(n_bidders)
    sorted_indices = np.argsort(bids)[::-1]
    winner = sorted_indices[0]
    highest_bid = bids[winner]

    # Handle ties for the highest bid
    tied_indices = [i for i in sorted_indices if bids[i] == highest_bid]
    if len(tied_indices) > 1:
        winner = random.choice(tied_indices)

    second_highest_bid = bids[sorted_indices[1]] if len(bids) > 1 else highest_bid

    if auction_type == "first":
        rewards[winner] = valuations[winner] - highest_bid
    else:  # second-price
        rewards[winner] = valuations[winner] - second_highest_bid

    return rewards, winner, highest_bid

# --------------------------------------------------------------
# 3) Q-learning update
# --------------------------------------------------------------
def qlearning_update(Q, s, a, r, s_next, alpha, gamma):
    """
    Q-learning update for single state-action pair.
    """
    old_q = Q[s, a]
    best_future = np.max(Q[s_next])
    Q[s, a] = old_q + alpha * (r + gamma * best_future - old_q)

# --------------------------------------------------------------
# 4) Build state index (with normalization)
# --------------------------------------------------------------
def build_state(own_signal, median_opp_bid, past_winning_bid,
                median_opp_past_bid_index, winner_bid_index_state,
                n_val_bins, n_bid_bins):
    """
    Build a state index for Q-table lookups:
    - own_signal in [0, 1], discretized into [0, n_val_bins - 1].
    - median_opp_bid in [0, 1], discretized into [0, n_bid_bins - 1] if included.
    - past_winning_bid in [0, 1], discretized into [0, n_bid_bins - 1] if included.
    
    The final idx is computed based on these discrete values.
    """
    # Discretize own_signal to [0, n_val_bins-1]
    own_signal_idx = int(own_signal * (n_val_bins - 1))

    # Discretize median_opp_bid to [0, n_bid_bins-1] (only if used)
    median_idx = int(median_opp_bid * (n_bid_bins - 1)) if median_opp_past_bid_index else 0

    # Discretize past_winning_bid to [0, n_bid_bins-1] (only if used)
    winner_idx = int(past_winning_bid * (n_bid_bins - 1)) if winner_bid_index_state else 0

    m_size = n_val_bins if median_opp_past_bid_index else 1
    w_size = n_val_bins if winner_bid_index_state else 1

    idx = own_signal_idx
    idx += n_val_bins * median_idx
    idx += (n_val_bins * m_size) * winner_idx

    dprint(f"State index: {idx} (own_signal={own_signal}, median_opp_bid={median_opp_bid}, past_winning_bid={past_winning_bid})")
    return idx

# --------------------------------------------------------------
# 5) Run single experiment
# --------------------------------------------------------------
def run_experiment(eta, auction_type, alpha, gamma, episodes,
                   init, exploration, asynchronous, n_bidders,
                   median_opp_past_bid_index, winner_bid_index_state, seed=0):
    """
    Run Q-learning for one experiment with optional state features,
    using normalized signals, valuations, and states.
    """
    np.random.seed(seed)
    random.seed(seed)

    # Number of bins for signals (value states) and bids
    n_val_bins = 6
    n_bid_bins = 6
    m_size = n_bid_bins if median_opp_past_bid_index else 1
    w_size = n_bid_bins if winner_bid_index_state else 1

    # Total number of states = n_val_bins * m_size * w_size
    n_states = n_val_bins * m_size * w_size

    # Initialize Q-tables
    if init == "random":
        Q = np.random.rand(n_bidders, n_states, n_bid_bins)
    else:  # init == "zeros"
        Q = np.zeros((n_bidders, n_states, n_bid_bins))

    # Actions in [0, 1] range, discretized into n_bid_bins
    actions = np.linspace(0, 1, n_bid_bins)

    # Epsilon schedule
    eps_start, eps_end = 1.0, 0.01
    decay_end = int(0.9 * episodes)

    revenues = []
    past_bids = np.zeros(n_bidders)  # Will store normalized past bids
    past_winner_bid = 0.0            # Will store normalized past winning bid

    def choose_action(Q_row, eps):
        if exploration == "boltzmann":
            ex = np.exp(Q_row)
            probs = ex / np.sum(ex)
            return np.random.choice(len(Q_row), p=probs)
        else:  # e-greedy
            if np.random.rand() < eps:
                return np.random.randint(n_bid_bins)
            return np.argmax(Q_row)

    for ep in range(episodes):
        # Update epsilon
        eps = eps_start - (ep / decay_end) * (eps_start - eps_end) if ep < decay_end else eps_end

        # ----------------------------------------------------
        # 1) Generate normalized signals
        # ----------------------------------------------------
        # Signals in [0, 1] by dividing by (n_val_bins - 1).
        raw_signals = np.random.randint(n_val_bins, size=n_bidders)
        signals = raw_signals / (n_val_bins - 1)  # Now in [0,1]

        # ----------------------------------------------------
        # 2) Calculate valuations (already in [0,1])
        # ----------------------------------------------------
        valuations = [get_valuation(eta, signals[i], np.delete(signals, i)) 
                      for i in range(n_bidders)]

        # ----------------------------------------------------
        # 3) Build states
        # ----------------------------------------------------
        states = [
            build_state(
                own_signal=signals[i],
                median_opp_bid=np.median(np.delete(past_bids, i)),  # median of last round's bids in [0,1]
                past_winning_bid=past_winner_bid,                  # winning bid of last round in [0,1]
                median_opp_past_bid_index=median_opp_past_bid_index,
                winner_bid_index_state=winner_bid_index_state,
                n_val_bins=n_val_bins,
                n_bid_bins=n_bid_bins
            )
            for i in range(n_bidders)
        ]

        # ----------------------------------------------------
        # 4) Choose actions and get bids
        # ----------------------------------------------------
        actions_taken = [choose_action(Q[i, states[i]], eps) for i in range(n_bidders)]
        bids = [actions[a] for a in actions_taken]  # Bids in [0,1]

        # ----------------------------------------------------
        # 5) Calculate payoffs
        # ----------------------------------------------------
        rewards, winner, highest_bid = get_payoffs(bids, valuations, auction_type)

        # ----------------------------------------------------
        # 6) Update Q-tables
        # ----------------------------------------------------
        if asynchronous:
            # Asynchronous update: each bidder's next state is built from a new signal draw
            for i in range(n_bidders):
                # Generate new signals for the "next" state
                raw_next_signals = np.random.randint(n_val_bins, size=n_bidders)
                next_signals = raw_next_signals / (n_val_bins - 1)

                next_state = build_state(
                    own_signal=next_signals[i],
                    median_opp_bid=np.median(np.delete(bids, i)),  # median of current round's bids
                    past_winning_bid=highest_bid,                  # current round's winning bid
                    median_opp_past_bid_index=median_opp_past_bid_index,
                    winner_bid_index_state=winner_bid_index_state,
                    n_val_bins=n_val_bins,
                    n_bid_bins=n_bid_bins
                )
                qlearning_update(Q[i], states[i], actions_taken[i], rewards[i], next_state, alpha, gamma)
        else:
            # Synchronous (counterfactual) updates
            for i in range(n_bidders):
                state = states[i]
                for alt_action in range(n_bid_bins):
                    # Counterfactual: what if bidder i took alt_action instead
                    counterfactual_bids = bids.copy()
                    counterfactual_bids[i] = actions[alt_action]
                    counterfactual_rewards, _, _ = get_payoffs(counterfactual_bids, valuations, auction_type)
                    max_next_q = np.max(Q[i, state])  # Next state is the same in synchronous setting
                    Q[i, state, alt_action] = (1 - alpha) * Q[i, state, alt_action] + \
                                              alpha * (counterfactual_rewards[i] + gamma * max_next_q)

        # ----------------------------------------------------
        # 7) Track results and update memory
        # ----------------------------------------------------
        revenues.append(max(bids))
        past_bids = np.array(bids)      # Store normalized bids
        past_winner_bid = highest_bid   # Store normalized winning bid

    # --------------------------------------------------------
    # 8) Compute final statistics
    # --------------------------------------------------------
    avg_rev_last_1000 = np.mean(revenues[-1000:]) if len(revenues) >= 1000 else np.mean(revenues)
    regrets = [1.0 - r for r in revenues]  # Example: if max possible is 1, then regret = 1 - revenue
    avg_regret_of_seller = np.mean(regrets)

    # Approximate time to converge: first time rolling avg is within ±5% of the final average
    time_to_converge = episodes
    rolling_avg = pd.Series(revenues).rolling(window=1000, min_periods=1).mean()
    lower_bound, upper_bound = 0.95 * avg_rev_last_1000, 1.05 * avg_rev_last_1000
    for t in range(len(rolling_avg)):
        if lower_bound <= rolling_avg.iloc[t] <= upper_bound:
            time_to_converge = t
            break

    return avg_rev_last_1000, time_to_converge, avg_regret_of_seller

# --------------------------------------------------------------
# 6) Main experiment loop
# --------------------------------------------------------------
def main_experiment(K=50):
    """
    Generate data for K experiments with random parameter draws.
    Now uses normalized signals, valuations, and states.
    """
    results = []
    auction_type_options = ["first", "second"]
    init_options = ["random", "zeros"]
    exploration_options = ["egreedy", "boltzmann"]

    for seed in trange(K, desc="Generating experiments"):
        eta = random.uniform(0.0, 1.0)
        alpha = random.uniform(0.01, 0.1)
        gamma = random.uniform(0.0, 0.99)
        episodes = int(random.uniform(10_000, 100_000))
        auction_type = random.choice(auction_type_options)
        init = random.choice(init_options)
        exploration = random.choice(exploration_options)
        asynchronous = random.choice([0, 1])
        n_bidders = random.choice([2, 4, 6])
        median_opp_past_bid_index = random.choice([False, True])
        winner_bid_index_state = random.choice([False, True])

        avg_rev_last_1000, time_to_converge, avg_regret_of_seller = run_experiment(
            eta=eta,
            auction_type=auction_type,
            alpha=alpha,
            gamma=gamma,
            episodes=episodes,
            init=init,
            exploration=exploration,
            asynchronous=asynchronous,
            n_bidders=n_bidders,
            median_opp_past_bid_index=median_opp_past_bid_index,
            winner_bid_index_state=winner_bid_index_state,
            seed=seed
        )

        results.append({
            "eta": eta,
            "alpha": alpha,
            "gamma": gamma,
            "episodes": episodes,
            "auction_type": auction_type,
            "init": init,
            "exploration": exploration,
            "asynchronous": asynchronous,
            "n_bidders": n_bidders,
            "median_opp_past_bid_index": median_opp_past_bid_index,
            "winner_bid_index_state": winner_bid_index_state,
            "avg_rev_last_1000": avg_rev_last_1000,
            "time_to_converge": time_to_converge,
            "avg_regret_of_seller": avg_regret_of_seller
        })
    print(results)
    return pd.DataFrame(results)

# --------------------------------------------------------------
# 7) Run and save data
# --------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs("experiment2", exist_ok=True)

    df = main_experiment(K=1000)
    csv_path = "experiment2/data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Data generation complete. Saved to '{csv_path}'.")