#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
from collections import defaultdict

# Create necessary directories
os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("logs", exist_ok=True)  # Add logs directory
np.random.seed(23423)

# Main hyperparameters - easily tune these
HYPERPARAMS = {
    "ALPHA_START": 0.1,
    "ALPHA_END": 0.1,                     # Learning rate
    "GAMMA": 0.95,                        # Discount factor
    "EPSILON_START": 1.0,                 # Exploration rate
    "EPSILON_END": 0.0,
    "DECAY_FACTOR": 2.0,                  # Controls exponential decay rate
    "BID_SPACE_GRANULARITY": 6,           # Number of bid options
    "MAX_LEARNING_ROUNDS": 1_000_000      # Training iterations
}

# Derive bid actions from granularity
HYPERPARAMS["BID_ACTIONS"] = np.linspace(0.0, 1.0, HYPERPARAMS["BID_SPACE_GRANULARITY"]).tolist()

def exponential_decay(value_start, value_end, t, t_max, decay_factor):
    """Exponential decay from value_start to value_end over t_max steps"""
    progress_ratio = t / float(t_max)
    decay = np.exp(-decay_factor * progress_ratio)
    decayed_val = value_end + (value_start - value_end) * decay
    return max(value_end, decayed_val)

def safe_get_q(qtable, state, action):
    """Safely get Q-value, return random initial value if not present"""
    if state not in qtable or action not in qtable[state]:
        return np.random.uniform(0.0, 1.0)
    return qtable[state][action]

def safe_set_q(qtable, state, action, value):
    """Safely set Q-value, creating state entry if needed"""
    if state not in qtable:
        qtable[state] = {}
    qtable[state][action] = value

def q_update(qtable, s, a, r, s_next, alpha, gamma, bid_actions):
    """Update Q-value using Q-learning update rule"""
    old_q = safe_get_q(qtable, s, a)
    nxt_qs = [safe_get_q(qtable, s_next, act) for act in bid_actions]
    td_target = r + gamma * max(nxt_qs)
    new_q = old_q + alpha * (td_target - old_q)
    safe_set_q(qtable, s, a, new_q)
    return abs(new_q - old_q)

def get_state(last_bids, bidder_idx, mode="none"):
    """Get state representation based on information disclosure mode"""
    if last_bids[0] is None or last_bids[1] is None:
        return ("START",)
    
    # CHANGE: For "none" mode, use a single state constant rather than previous bid
    if mode == "none":
        return ("SINGLE_STATE",)
    
    my_bid = last_bids[bidder_idx]
    opp_bid = last_bids[1 - bidder_idx]
    winning_bid = max(last_bids)
    
    if mode == "winning":
        return (round(my_bid, 2), round(winning_bid, 2))
    else:  # full disclosure
        return (round(my_bid, 2), round(opp_bid, 2))

def choose_action(qtable, state, epsilon, bid_actions):
    """Choose action using epsilon-greedy policy"""
    if np.random.rand() < epsilon:
        return np.random.choice(bid_actions)
    vals = [safe_get_q(qtable, state, a) for a in bid_actions]
    m = max(vals)
    cands = [a for (a, v) in zip(bid_actions, vals) if v == m]
    return np.random.choice(cands)

def coin_flip_winner(indices):
    """Randomly choose a winner from tied bidders"""
    return np.random.choice(indices)

def immediate_reward_and_next_state_any(state, a0, a1, auction_type="FPA", gamma=0.95):
    """Calculate immediate rewards and next state for both agents"""
    bids = [a0, a1]
    max_bid = max(bids)
    
    if a0 == a1:
        winner = np.random.choice([0, 1])
    else:
        winner = 0 if a0 > a1 else 1

    if auction_type == "FPA":
        price = max_bid
    else:
        if a0 == a1:
            price = max_bid
        else:
            loser = 1 - winner
            price = bids[loser]

    r0 = 1.0 - price if winner == 0 else 0.0
    r1 = 1.0 - price if winner == 1 else 0.0
    return r0, r1, (a0, a1)

def analyze_q_tables(q0, q1, auction_type="FPA", gamma=0.95):
    """Analyze Q-tables to compute Bellman errors and best response consistency"""
    all_states = set(q0.keys()) | set(q1.keys())
    be0, be1 = [], []
    brv0, brv1, valid_count = 0, 0, 0

    def Q(qt, s, a): 
        return qt.get(s, {}).get(a, None)
    
    def best_act(qt, s):
        if s not in qt or not qt[s]:
            return None
        return max(qt[s], key=lambda x: qt[s][x])

    def greedy(qt, s):
        if s not in qt or not qt[s]:
            return None
        max_q = max(qt[s].values())
        for (act, val) in qt[s].items():
            if val == max_q:
                return act
        return None

    for s in all_states:
        a0s = greedy(q0, s)
        a1s = greedy(q1, s)
        if a0s is None or a1s is None:
            continue
        valid_count += 1
        
        r0, r1, s_next = immediate_reward_and_next_state_any(s, a0s, a1s, auction_type, gamma)

        a0n = best_act(q0, s_next)
        a1n = best_act(q1, s_next)

        Q0sa = Q(q0, s, a0s)
        Q1sa = Q(q1, s, a1s)

        if Q0sa is not None:
            nxt_val = Q(q0, s_next, a0n) if a0n is not None else 0
            t0 = r0 + gamma * nxt_val
            be0.append(abs(Q0sa - t0))
        if Q1sa is not None:
            nxt_val = Q(q1, s_next, a1n) if a1n is not None else 0
            t1 = r1 + gamma * nxt_val
            be1.append(abs(Q1sa - t1))

        possible_0 = list(q0[s].keys())
        best_val_0 = -1e9
        chosen_val_0 = r0 + gamma*(Q(q0, s_next, a0n) if a0n else 0)
        for alt0 in possible_0:
            rr0, _, sn = immediate_reward_and_next_state_any(s, alt0, a1s, auction_type, gamma)
            alt0n = best_act(q0, sn)
            alt_val = rr0 + gamma*(Q(q0, sn, alt0n) if alt0n else 0)
            best_val_0 = max(best_val_0, alt_val)
        if chosen_val_0 + 1e-7 < best_val_0:
            brv0 += 1

        possible_1 = list(q1[s].keys())
        best_val_1 = -1e9
        chosen_val_1 = r1 + gamma*(Q(q1, s_next, a1n) if a1n else 0)
        for alt1 in possible_1:
            _, rr1, sn = immediate_reward_and_next_state_any(s, a0s, alt1, auction_type, gamma)
            alt1n = best_act(q1, sn)
            alt_val = rr1 + gamma*(Q(q1, sn, alt1n) if alt1n else 0)
            best_val_1 = max(best_val_1, alt_val)
        if chosen_val_1 + 1e-7 < best_val_1:
            brv1 += 1

    if not valid_count:
        return {
            "mean_bellman_error_agent0": None,
            "mean_bellman_error_agent1": None,
            "best_response_consistency_agent0": None,
            "best_response_consistency_agent1": None,
            "num_states_checked": 0
        }
    return {
        "mean_bellman_error_agent0": np.mean(be0) if be0 else 0,
        "mean_bellman_error_agent1": np.mean(be1) if be1 else 0,
        "best_response_consistency_agent0": 1.0 - (brv0 / valid_count),
        "best_response_consistency_agent1": 1.0 - (brv1 / valid_count),
        "num_states_checked": valid_count
    }

def run_case_study(auction_type, state_mode="none", seed=0, alpha_start=None, alpha_end=None, gamma=None, epsilon_start=None, 
                  epsilon_end=None, bid_actions=None, max_learning_rounds=None, decay_factor=None):
    """Run a single auction experiment with specified parameters"""
    # Use provided hyperparameters or fall back to defaults from HYPERPARAMS
    alpha_start = alpha_start if alpha_start is not None else HYPERPARAMS["ALPHA_START"]
    alpha_end = alpha_end if alpha_end is not None else HYPERPARAMS["ALPHA_END"]
    gamma = gamma if gamma is not None else HYPERPARAMS["GAMMA"]
    epsilon_start = epsilon_start if epsilon_start is not None else HYPERPARAMS["EPSILON_START"]
    epsilon_end = epsilon_end if epsilon_end is not None else HYPERPARAMS["EPSILON_END"]
    decay_factor = decay_factor if decay_factor is not None else HYPERPARAMS["DECAY_FACTOR"]
    bid_actions = bid_actions if bid_actions is not None else HYPERPARAMS["BID_ACTIONS"]
    max_learning_rounds = max_learning_rounds if max_learning_rounds is not None else HYPERPARAMS["MAX_LEARNING_ROUNDS"]
    
    info_regime_name = {
        "none": "No Disclosure",
        "winning": "Winning Bid Only",
        "full": "Full Disclosure"
    }[state_mode]
    
    np.random.seed(seed)
    q_tables = [{}, {}]
    last_bids = [None, None]
    actions_log, q_evol, rev_log, bell_errors = [], [], [], []

    for t in range(max_learning_rounds):
        alpha_t = exponential_decay(alpha_start, alpha_end, t, max_learning_rounds, decay_factor)
        eps_t = exponential_decay(epsilon_start, epsilon_end, t, max_learning_rounds, decay_factor)
        
        if t % 100000 == 0:
            print(f"Progress: {t:,}/{max_learning_rounds:,} iterations complete ({t/max_learning_rounds*100:.2f}%) - {auction_type} {info_regime_name}")
        
        s0 = get_state(last_bids, 0, state_mode)
        s1 = get_state(last_bids, 1, state_mode)

        a0 = choose_action(q_tables[0], s0, eps_t, bid_actions)
        a1 = choose_action(q_tables[1], s1, eps_t, bid_actions)

        bids = [a0, a1]
        max_bid = max(bids)
        winners = [i for i, b in enumerate(bids) if b == max_bid]
        
        if len(winners) > 1:
            winner = coin_flip_winner(winners)
        else:
            winner = winners[0]

        if auction_type == "FPA":
            price = max_bid
        else:
            if len(winners) > 1:
                price = max_bid
            else:
                loser = 1 - winner
                price = bids[loser]

        if winner == 0:
            r0 = 1.0 - price
            r1 = 0.0
        else:
            r0 = 0.0
            r1 = 1.0 - price

        rev_log.append({
            "t": t,
            "auction_type": auction_type,
            "winner": winner,
            "price": price
        })

        next_bids = [a0, a1]
        s0_next = get_state(next_bids, 0, state_mode)
        s1_next = get_state(next_bids, 1, state_mode)

        td0 = q_update(q_tables[0], s0, a0, r0, s0_next, alpha_t, gamma, bid_actions)
        td1 = q_update(q_tables[1], s1, a1, r1, s1_next, alpha_t, gamma, bid_actions)

        bell_errors.append({
            "t": t, "auction_type": auction_type, 
            "agent": 0, "abs_td_error": td0
        })
        bell_errors.append({
            "t": t, "auction_type": auction_type, 
            "agent": 1, "abs_td_error": td1
        })

        actions_log.append({
            "t": t, "auction_type": auction_type, 
            "bidder": 0, "chosen_bid": a0, 
            "reward": r0
        })
        actions_log.append({
            "t": t, "auction_type": auction_type, 
            "bidder": 1, "chosen_bid": a1, 
            "reward": r1
        })

        if t % 1000 == 0:
            for b in [0, 1]:
                for st, adict in q_tables[b].items():
                    for act, val in adict.items():
                        q_evol.append({
                            "t": t, "auction_type": auction_type, 
                            "bidder": b, "state": st, 
                            "action": act, "Q_value": val
                        })

        last_bids = next_bids
    
    print(f"Completed {max_learning_rounds:,} iterations for {auction_type} with {info_regime_name} information regime")
    return q_evol, actions_log, q_tables, rev_log, bell_errors

def run_experiment_for_info_regime(state_mode, alpha_start=None, alpha_end=None, gamma=None, epsilon_start=None, 
                                  epsilon_end=None, bid_actions=None, max_learning_rounds=None, seed=42, decay_factor=None):
    """Run experiments for a specific information regime and save the data"""
    # Use provided hyperparameters or fall back to defaults from HYPERPARAMS
    alpha_start = alpha_start if alpha_start is not None else HYPERPARAMS["ALPHA_START"]
    alpha_end = alpha_end if alpha_end is not None else HYPERPARAMS["ALPHA_END"]
    gamma = gamma if gamma is not None else HYPERPARAMS["GAMMA"]
    epsilon_start = epsilon_start if epsilon_start is not None else HYPERPARAMS["EPSILON_START"]
    epsilon_end = epsilon_end if epsilon_end is not None else HYPERPARAMS["EPSILON_END"]
    decay_factor = decay_factor if decay_factor is not None else HYPERPARAMS["DECAY_FACTOR"]
    bid_actions = bid_actions if bid_actions is not None else HYPERPARAMS["BID_ACTIONS"]
    max_learning_rounds = max_learning_rounds if max_learning_rounds is not None else HYPERPARAMS["MAX_LEARNING_ROUNDS"]
    
    info_regime_name = {
        "none": "No Disclosure",
        "winning": "Winning Bid Only",
        "full": "Full Disclosure"
    }[state_mode]
    
    print(f"\nRunning experiments for {info_regime_name} information regime...")
    
    # Run FPA experiment
    print(f"Running First-Price Auction (FPA) experiment...")
    fpa_q, fpa_act, fpa_qt, fpa_rev, fpa_be = run_case_study(
        "FPA", state_mode=state_mode, seed=seed, alpha_start=alpha_start, alpha_end=alpha_end, 
        gamma=gamma, epsilon_start=epsilon_start, epsilon_end=epsilon_end, bid_actions=bid_actions, 
        max_learning_rounds=max_learning_rounds, decay_factor=decay_factor
    )
    
    # Run SPA experiment
    print(f"Running Second-Price Auction (SPA) experiment...")
    spa_q, spa_act, spa_qt, spa_rev, spa_be = run_case_study(
        "SPA", state_mode=state_mode, seed=seed, alpha_start=alpha_start, alpha_end=alpha_end, 
        gamma=gamma, epsilon_start=epsilon_start, epsilon_end=epsilon_end, bid_actions=bid_actions, 
        max_learning_rounds=max_learning_rounds, decay_factor=decay_factor
    )

    # Analyze Q-tables
    fpa_analysis = analyze_q_tables(fpa_qt[0], fpa_qt[1], "FPA", gamma=gamma)
    spa_analysis = analyze_q_tables(spa_qt[0], spa_qt[1], "SPA", gamma=gamma)
    
    # Save all the data
    print(f"Saving data for {info_regime_name}...")
    
    # Convert to DataFrames for easier analysis later
    df_act = pd.DataFrame(fpa_act + spa_act)
    df_be = pd.DataFrame(fpa_be + spa_be)
    df_rev = pd.DataFrame(fpa_rev + spa_rev)
    
    # Save the DataFrames
    df_act.to_csv(f"data/actions_{state_mode}.csv", index=False)
    df_be.to_csv(f"data/bellman_errors_{state_mode}.csv", index=False)
    df_rev.to_csv(f"data/revenue_{state_mode}.csv", index=False)
    
    # Save Q-tables and other data in compressed format
    save_path = f"data/{state_mode}_data.npz"
    np.savez_compressed(
        save_path,
        fpa_qtable0=fpa_qt[0],
        fpa_qtable1=fpa_qt[1],
        spa_qtable0=spa_qt[0],
        spa_qtable1=spa_qt[1],
        fpa_revenue=[entry["price"] for entry in fpa_rev],
        spa_revenue=[entry["price"] for entry in spa_rev],
        fpa_actions=[(entry["bidder"], entry["chosen_bid"]) for entry in fpa_act],
        spa_actions=[(entry["bidder"], entry["chosen_bid"]) for entry in spa_act],
        fpa_analysis=fpa_analysis,
        spa_analysis=spa_analysis
    )
    
    # Save hyperparameters used for this run
    settings = {
        "state_mode": state_mode,
        "alpha_start": alpha_start,
        "alpha_end": alpha_end,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "decay_factor": decay_factor,
        "bid_actions": bid_actions,
        "max_learning_rounds": max_learning_rounds
    }
    
    # Save settings in text format
    with open(f"data/settings_{state_mode}.txt", 'w') as f:
        for key, value in settings.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.2f}\n")
            elif isinstance(value, list) and all(isinstance(x, float) for x in value):
                f.write(f"{key}: {[round(b, 2) for b in value]}\n")
            else:
                f.write(f"{key}: {value}\n")

def main():
    """Main function to run all experiments"""
    # Save general experiment settings
    with open("data/experiment_settings.txt", "w") as f:
        f.write("Experiment Settings:\n")
        f.write(f"Learning Rate (α): {HYPERPARAMS['ALPHA_START']:.2f} to {HYPERPARAMS['ALPHA_END']:.2f} (exponential decay)\n")
        f.write(f"Decay Factor: {HYPERPARAMS['DECAY_FACTOR']:.2f}\n")
        f.write(f"Discount Factor (γ): {HYPERPARAMS['GAMMA']:.2f}\n")
        f.write(f"Exploration Rate (ε): {HYPERPARAMS['EPSILON_START']:.2f} to {HYPERPARAMS['EPSILON_END']:.2f} (exponential decay)\n")
        f.write(f"Bid Space: {[round(b, 2) for b in HYPERPARAMS['BID_ACTIONS']]}\n")
        f.write(f"Training Rounds: {HYPERPARAMS['MAX_LEARNING_ROUNDS']:,}\n")
    
    # Also save hyperparameters as Python pickle for other modules to use
    import pickle
    with open("data/hyperparams.pkl", "wb") as f:
        pickle.dump(HYPERPARAMS, f)
    
    # Run experiments for each information regime
    run_experiment_for_info_regime("none")
    run_experiment_for_info_regime("winning")
    run_experiment_for_info_regime("full")
    
    print("Experiment completed. Data saved to data/ directory.")

if __name__ == "__main__":
    main()