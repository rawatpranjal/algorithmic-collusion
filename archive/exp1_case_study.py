#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
import scipy.stats as stats
from collections import defaultdict
import os
import sys
from datetime import datetime
import argparse
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend

# Default output directory (set by main() when running as script)
OUTPUT_DIR = "results/exp1"
log_file = None

def _init_output(output_dir: str):
    """Initialize output directory and log file."""
    global OUTPUT_DIR, log_file
    OUTPUT_DIR = output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_filename = f"{OUTPUT_DIR}/experiment1_log.txt"
    log_file = open(log_filename, "w")

np.random.seed(23423)

def log_print(message):
    print(message)
    if log_file is not None:
        log_file.write(message + "\n")
        log_file.flush()

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20,
    'axes.grid': True,
    'grid.alpha': 0.6,
    'figure.figsize': (10, 6)
})

COLOR_PALETTE = sns.color_palette("colorblind")
sns.set_palette(COLOR_PALETTE)

def linear_decay(value_start, value_end, t, t_max):
    frac = t / float(t_max)
    decayed_val = value_start * (1 - frac) + value_end * frac
    return max(value_end, decayed_val)

def safe_get_q(qtable, state, action):
    if state not in qtable or action not in qtable[state]:
        return np.random.uniform(0.0, 1.0)
    return qtable[state][action]

def safe_set_q(qtable, state, action, value):
    if state not in qtable:
        qtable[state] = {}
    qtable[state][action] = value

def q_update(qtable, s, a, r, s_next, alpha, gamma, bid_actions):
    old_q = safe_get_q(qtable, s, a)
    nxt_qs = [safe_get_q(qtable, s_next, act) for act in bid_actions]
    td_target = r + gamma * max(nxt_qs)
    new_q = old_q + alpha * (td_target - old_q)
    safe_set_q(qtable, s, a, new_q)
    return abs(new_q - old_q)

def get_state(last_bids, bidder_idx, mode="none"):
    if last_bids[0] is None or last_bids[1] is None:
        return ("START",)
    
    my_bid = last_bids[bidder_idx]
    opp_bid = last_bids[1 - bidder_idx]
    winning_bid = max(last_bids)
    
    if mode == "none":
        return (round(my_bid, 2),)
    elif mode == "winning":
        return (round(my_bid, 2), round(winning_bid, 2))
    else:
        return (round(my_bid, 2), round(opp_bid, 2))

def choose_action(qtable, state, epsilon, bid_actions):
    if np.random.rand() < epsilon:
        return np.random.choice(bid_actions)
    vals = [safe_get_q(qtable, state, a) for a in bid_actions]
    m = max(vals)
    cands = [a for (a, v) in zip(bid_actions, vals) if v == m]
    return np.random.choice(cands)

def coin_flip_winner(indices):
    return np.random.choice(indices)

def immediate_reward_and_next_state_any(state, a0, a1, auction_type="FPA", gamma=0.95):
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

def analyze_retaliation(qtable, mode):
    if mode == "none":
        return {"retaliation_count": 0, "total_possible": 0, "ratio": 0.0}
    
    retaliation_count = 0
    total_possible = 0
    
    for state, actions in qtable.items():
        if len(state) < 2:
            continue
            
        my_bid = state[0]
        
        if mode == "winning":
            winning_bid = state[1]
            if winning_bid > my_bid:
                total_possible += 1
                best_action = max(actions.items(), key=lambda x: x[1])[0]
                if best_action > my_bid:
                    retaliation_count += 1
                    
        elif mode == "full":
            opp_bid = state[1]
            if opp_bid > my_bid:
                total_possible += 1
                best_action = max(actions.items(), key=lambda x: x[1])[0]
                if best_action > my_bid:
                    retaliation_count += 1
    
    ratio = retaliation_count / total_possible if total_possible > 0 else 0.0
    
    return {
        "retaliation_count": retaliation_count,
        "total_possible": total_possible,
        "ratio": ratio
    }

def run_case_study(auction_type, state_mode="none", seed=0, alpha_start=0.1, alpha_end=0.0, gamma=0.9, epsilon_start=0.1,
                  epsilon_end=0.0, bid_actions=None, max_learning_rounds=1_000_000, rolling_window=10000, bellman_window=50000,
                  progress_callback=None):
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
        # Progress callback every 1000 iterations
        if progress_callback and t % 1000 == 0:
            progress_callback(current=t, total=max_learning_rounds)
        elif t % 100000 == 0:
            print(f"Progress: {t:,}/{max_learning_rounds:,} iterations complete ({t/max_learning_rounds*100:.0f}%) - {auction_type} {info_regime_name}")

        alpha_t = linear_decay(alpha_start, alpha_end, t, max_learning_rounds)
        eps_t = linear_decay(epsilon_start, epsilon_end, t, max_learning_rounds)
        
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

    # Final progress update
    if progress_callback:
        progress_callback(current=max_learning_rounds, total=max_learning_rounds)

    print(f"Completed {max_learning_rounds:,} iterations for {auction_type} with {info_regime_name} information regime")
    return q_evol, actions_log, q_tables, rev_log, bell_errors

def plot_combined_results(df_act, df_be, df_rev, state_mode, rolling_window):
    info_regime_name = {
        "none": "No Disclosure",
        "winning": "Winning Bid Only",
        "full": "Full Disclosure"
    }[state_mode]
    
    df_act = df_act.copy()
    df_act["rolling_bid"] = df_act.groupby(["auction_type", "bidder"])["chosen_bid"] \
                                 .transform(lambda x: x.rolling(rolling_window, min_periods=1).mean())
    df_act["rolling_reward"] = df_act.groupby(["auction_type", "bidder"])["reward"] \
                                    .transform(lambda x: x.rolling(rolling_window, min_periods=1).mean())
    
    df_be = df_be.copy()
    df_be["auction_agent"] = df_be["auction_type"] + " Agent " + df_be["agent"].astype(str)
    
    df_be_grp = df_be.groupby(["t", "auction_agent"])["abs_td_error"].mean().reset_index()
    df_be_grp["rolling_error"] = df_be_grp.groupby("auction_agent")["abs_td_error"] \
                                      .transform(lambda x: x.rolling(rolling_window*5, min_periods=1).mean())
    
    df_rev = df_rev.copy()
    df_rev["rolling_price"] = df_rev.groupby("auction_type")["price"] \
                                   .transform(lambda x: x.rolling(rolling_window, min_periods=1).mean())
    
    tmp0 = df_act[df_act.bidder == 0]
    tmp1 = df_act[df_act.bidder == 1]
    
    plt.style.use('grayscale')
    
    line_styles = ['-', '--', '-.', ':']
    
    fig1, axes1 = plt.subplots(1, 2, figsize=(20, 8), sharex=True, facecolor='white')
    plt.suptitle(f"Convergence and Revenue - Information Regime: {info_regime_name}", fontsize=18)
    
    for i, (name, group) in enumerate(df_be_grp.groupby("auction_agent")):
        style_idx = i % len(line_styles)
        axes1[0].plot(group["t"], group["rolling_error"], 
                     linestyle=line_styles[style_idx], 
                     linewidth=2,
                     label=name)
    
    axes1[0].set_title(f"Mean Bellman Error (Window={rolling_window*5})")
    axes1[0].legend()
    axes1[0].set_ylim(bottom=0)
    axes1[0].set_xlabel('Training Steps')
    axes1[0].set_ylabel('Mean Bellman Error')
    axes1[0].grid(True, linestyle='--', alpha=0.7)
    
    for i, (name, group) in enumerate(df_rev.groupby("auction_type")):
        style_idx = i % len(line_styles)
        axes1[1].plot(group["t"], group["rolling_price"], 
                     linestyle=line_styles[style_idx], 
                     linewidth=2,
                     label=name)
    
    axes1[1].set_title(f"Rolling Seller Revenue (Window={rolling_window})")
    axes1[1].legend()
    axes1[1].set_ylim(0, 1)
    axes1[1].set_xlabel('Training Steps')
    axes1[1].set_ylabel('Revenue')
    axes1[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    filename1 = f"{OUTPUT_DIR}/bellman_revenue_{state_mode}.png"
    fig1.savefig(filename1, dpi=300, bbox_inches='tight')
    plt.close(fig1)  # Close figure after saving
    log_print(f"Saved figure: {filename1}")
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(20, 8), sharex=True, facecolor='white')
    plt.suptitle(f"Bidder Rewards - Information Regime: {info_regime_name}", fontsize=18)
    
    for i, (name, group) in enumerate(tmp0.groupby("auction_type")):
        style_idx = i % len(line_styles)
        axes2[0].plot(group["t"], group["rolling_reward"], 
                     linestyle=line_styles[style_idx], 
                     linewidth=2,
                     label=name)
    
    axes2[0].set_title(f"Rolling Rewards - Bidder 0 (Window={rolling_window})")
    axes2[0].legend()
    axes2[0].set_ylim(0, 1)
    axes2[0].set_xlabel('Training Steps')
    axes2[0].set_ylabel('Reward')
    axes2[0].grid(True, linestyle='--', alpha=0.7)
    
    for i, (name, group) in enumerate(tmp1.groupby("auction_type")):
        style_idx = i % len(line_styles)
        axes2[1].plot(group["t"], group["rolling_reward"], 
                     linestyle=line_styles[style_idx], 
                     linewidth=2,
                     label=name)
    
    axes2[1].set_title(f"Rolling Rewards - Bidder 1 (Window={rolling_window})")
    axes2[1].legend()
    axes2[1].set_ylim(0, 1)
    axes2[1].set_xlabel('Training Steps')
    axes2[1].set_ylabel('Reward')
    axes2[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    filename2 = f"{OUTPUT_DIR}/rewards_{state_mode}.png"
    fig2.savefig(filename2, dpi=300, bbox_inches='tight')
    plt.close(fig2)  # Close figure after saving
    log_print(f"Saved figure: {filename2}")
    
    fig3, axes3 = plt.subplots(1, 2, figsize=(20, 8), sharex=True, facecolor='white')
    plt.suptitle(f"Bidding Behavior - Information Regime: {info_regime_name}", fontsize=18)
    
    for i, (name, group) in enumerate(tmp0.groupby("auction_type")):
        style_idx = i % len(line_styles)
        axes3[0].plot(group["t"], group["rolling_bid"], 
                     linestyle=line_styles[style_idx], 
                     linewidth=2,
                     label=name)
    
    axes3[0].set_title(f"Rolling Bids - Bidder 0 (Window={rolling_window})")
    axes3[0].legend()
    axes3[0].set_ylim(0, 1)
    axes3[0].set_xlabel('Training Steps')
    axes3[0].set_ylabel('Bid')
    axes3[0].grid(True, linestyle='--', alpha=0.7)
    
    for i, (name, group) in enumerate(tmp1.groupby("auction_type")):
        style_idx = i % len(line_styles)
        axes3[1].plot(group["t"], group["rolling_bid"], 
                     linestyle=line_styles[style_idx], 
                     linewidth=2,
                     label=name)
    
    axes3[1].set_title(f"Rolling Bids - Bidder 1 (Window={rolling_window})")
    axes3[1].legend()
    axes3[1].set_ylim(0, 1)
    axes3[1].set_xlabel('Training Steps')
    axes3[1].set_ylabel('Bid')
    axes3[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    filename3 = f"{OUTPUT_DIR}/bids_{state_mode}.png"
    fig3.savefig(filename3, dpi=300, bbox_inches='tight')
    plt.close(fig3)  # Close figure after saving
    log_print(f"Saved figure: {filename3}")

def create_distribution_table(fpa_rev, spa_rev, fpa_act, spa_act, state_mode, window_size=10000, bid_actions=None):
    info_regime_name = {
        "none": "No Disclosure",
        "winning": "Winning Bid Only",
        "full": "Full Disclosure"
    }[state_mode]
    
    fpa_prices = [entry["price"] for entry in fpa_rev[-window_size:]]
    spa_prices = [entry["price"] for entry in spa_rev[-window_size:]]
    
    fpa_slice = fpa_act[-2*window_size:]
    spa_slice = spa_act[-2*window_size:]
    
    df_fpa = pd.DataFrame(fpa_slice)
    df_spa = pd.DataFrame(spa_slice)
    
    fpa0 = df_fpa[df_fpa.bidder == 0]['chosen_bid'].values[-window_size//2:]
    fpa1 = df_fpa[df_fpa.bidder == 1]['chosen_bid'].values[-window_size//2:]
    spa0 = df_spa[df_spa.bidder == 0]['chosen_bid'].values[-window_size//2:]
    spa1 = df_spa[df_spa.bidder == 1]['chosen_bid'].values[-window_size//2:]
    
    def count_bid(bid_list, target_bid):
        return len([b for b in bid_list if abs(b - target_bid) < 0.001])
    
    rows = []
    headers = ["Bid", "FPA Price (%)", "SPA Price (%)", "FPA Bid0 (%)", "FPA Bid1 (%)", "SPA Bid0 (%)", "SPA Bid1 (%)"]
    
    for bid in bid_actions:
        fpa_price_count = count_bid(fpa_prices, bid)
        spa_price_count = count_bid(spa_prices, bid)
        
        fpa0_count = count_bid(fpa0, bid)
        fpa1_count = count_bid(fpa1, bid)
        spa0_count = count_bid(spa0, bid)
        spa1_count = count_bid(spa1, bid)
        
        fpa_price_pct = (fpa_price_count / window_size) * 100
        spa_price_pct = (spa_price_count / window_size) * 100
        fpa0_pct = (fpa0_count / (window_size//2)) * 100
        fpa1_pct = (fpa1_count / (window_size//2)) * 100
        spa0_pct = (spa0_count / (window_size//2)) * 100
        spa1_pct = (spa1_count / (window_size//2)) * 100
        
        rows.append([
            f"{bid:.2f}", 
            f"{fpa_price_count} ({fpa_price_pct:.1f}%)", 
            f"{spa_price_count} ({spa_price_pct:.1f}%)",
            f"{fpa0_count} ({fpa0_pct:.1f}%)",
            f"{fpa1_count} ({fpa1_pct:.1f}%)",
            f"{spa0_count} ({spa0_pct:.1f}%)",
            f"{spa1_count} ({spa1_pct:.1f}%)"
        ])
    
    table_title = f"\n--- Distribution Table (Last {window_size} Rounds) - {info_regime_name} ---\n"
    table_str = table_title + tabulate(rows, headers=headers, tablefmt="pipe")
    
    filename = f"{OUTPUT_DIR}/distribution_table_{state_mode}.txt"
    with open(filename, 'w') as f:
        f.write(table_str)
    log_print(f"Saved distribution table to: {filename}")
    
    latex_filename = f"{OUTPUT_DIR}/distribution_table_{state_mode}.tex"
    with open(latex_filename, 'w') as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Distribution of Bids and Prices - " + info_regime_name + "}\n")
        f.write("\\label{tab:distribution-" + state_mode + "}\n")
        f.write("\\small\n")
        f.write(tabulate(rows, headers=headers, tablefmt="latex"))
        f.write("\n\\end{table}")
    log_print(f"Saved LaTeX distribution table to: {latex_filename}")
    
    return table_str

def plot_learned_strategies(qtable_fpa, qtable_spa, bidder, state_mode, bid_actions=None):
    info_regime_name = {
        "none": "No Disclosure",
        "winning": "Winning Bid Only",
        "full": "Full Disclosure"
    }[state_mode]
    
    plt.style.use('grayscale')
    table_str = ""  # Initialize table_str
    
    if state_mode == "none":
        # Extract states and best actions from Q-tables
        def get_best_action(qtable, state):
            if state not in qtable or not qtable[state]:
                return None
            return max(qtable[state], key=lambda a: qtable[state][a])

        fpa_states = []
        fpa_data = []
        for state, actions in qtable_fpa.items():
            if state != ("START",) and actions:
                prev_bid = state[0]
                best_action = get_best_action(qtable_fpa, state)
                if best_action is not None:
                    fpa_states.append(prev_bid)
                    fpa_data.append((prev_bid, best_action))

        spa_states = []
        spa_data = []
        for state, actions in qtable_spa.items():
            if state != ("START",) and actions:
                prev_bid = state[0]
                best_action = get_best_action(qtable_spa, state)
                if best_action is not None:
                    spa_states.append(prev_bid)
                    spa_data.append((prev_bid, best_action))

        # Create table string
        rows = []
        headers = ["Previous Bid", "FPA Best Action", "SPA Best Action"]

        all_prev_bids = sorted(set(fpa_states + spa_states))

        fpa_map = dict(fpa_data)
        spa_map = dict(spa_data)
        
        for prev_bid in all_prev_bids:
            fpa_best = fpa_map.get(prev_bid, "N/A")
            spa_best = spa_map.get(prev_bid, "N/A")
            
            rows.append([f"{prev_bid:.2f}", f"{fpa_best:.2f}" if fpa_best != "N/A" else "N/A", 
                         f"{spa_best:.2f}" if spa_best != "N/A" else "N/A"])
        
        table_title = f"\n--- Learned Strategy Table - Bidder {bidder} - {info_regime_name} ---\n"
        table_str = table_title + tabulate(rows, headers=headers, tablefmt="pipe")
        
        # Save table files
        filename = f"{OUTPUT_DIR}/strategy_table_{state_mode}_bidder{bidder}.txt"
        with open(filename, 'w') as f:
            f.write(table_str)
        log_print(f"Saved strategy table to: {filename}")
        
        latex_filename = f"{OUTPUT_DIR}/strategy_table_{state_mode}_bidder{bidder}.tex"
        with open(latex_filename, 'w') as f:
            f.write("\\begin{table}[ht]\n")
            f.write("\\centering\n")
            f.write("\\caption{Learned Strategy - Bidder " + str(bidder) + " - " + info_regime_name + "}\n")
            f.write("\\label{tab:strategy-" + state_mode + "-bidder" + str(bidder) + "}\n")
            f.write(tabulate(rows, headers=headers, tablefmt="latex"))
            f.write("\n\\end{table}")
        log_print(f"Saved LaTeX strategy table to: {latex_filename}")
        
        # Plot and save figures
        # ... rest of none mode code ...

    elif state_mode == "winning":
        # ... existing winning mode code ...
        pass

    else:  # state_mode == "full"
        # ... existing full mode code ...
        pass

    return table_str

def plot_bid_distribution(fpa_rev, spa_rev, fpa_act, spa_act, state_mode, window_size=10000, bid_actions=None):
    table_str = create_distribution_table(fpa_rev, spa_rev, fpa_act, spa_act, state_mode, window_size, bid_actions)
    log_print(table_str)

def tabulate_analysis_results(fpa_analysis, spa_analysis):
    table_data = [
        ["FPA", 
         round(fpa_analysis["mean_bellman_error_agent0"], 2), 
         round(fpa_analysis["mean_bellman_error_agent1"], 2),
         round(fpa_analysis["best_response_consistency_agent0"], 2),
         round(fpa_analysis["best_response_consistency_agent1"], 2),
         fpa_analysis["num_states_checked"]],
        ["SPA", 
         round(spa_analysis["mean_bellman_error_agent0"], 2), 
         round(spa_analysis["mean_bellman_error_agent1"], 2),
         round(spa_analysis["best_response_consistency_agent0"], 2),
         round(spa_analysis["best_response_consistency_agent1"], 2),
         spa_analysis["num_states_checked"]]
    ]
    headers = ["Auction Type", 
               "Mean Bellman Error (Agent 0)", 
               "Mean Bellman Error (Agent 1)",
               "Best Response Consistency (Agent 0)", 
               "Best Response Consistency (Agent 1)", 
               "Number of States Checked"]
    
    table_str = "\n--- Analysis Results ---\n"
    table_str += tabulate(table_data, headers=headers)
    log_print(table_str)

def tabulate_combined_strategies(fpa_qt0, fpa_qt1, spa_qt0, spa_qt1):
    rows = []
    
    for st, acts in fpa_qt0.items():
        if st == ('START',):
            continue
        max_q = max(acts.values())
        for a, qv in sorted(acts.items()):
            star = "*" if abs(qv - max_q) < 1e-12 else ""
            rows.append(["FPA", 0, f"{st}", f"{a:.2f}", f"{qv:.2f}{star}"])
    
    for st, acts in fpa_qt1.items():
        if st == ('START',):
            continue
        max_q = max(acts.values())
        for a, qv in sorted(acts.items()):
            star = "*" if abs(qv - max_q) < 1e-12 else ""
            rows.append(["FPA", 1, f"{st}", f"{a:.2f}", f"{qv:.2f}{star}"])
    
    for st, acts in spa_qt0.items():
        if st == ('START',):
            continue
        max_q = max(acts.values())
        for a, qv in sorted(acts.items()):
            star = "*" if abs(qv - max_q) < 1e-12 else ""
            rows.append(["SPA", 0, f"{st}", f"{a:.2f}", f"{qv:.2f}{star}"])
    
    for st, acts in spa_qt1.items():
        if st == ('START',):
            continue
        max_q = max(acts.values())
        for a, qv in sorted(acts.items()):
            star = "*" if abs(qv - max_q) < 1e-12 else ""
            rows.append(["SPA", 1, f"{st}", f"{a:.2f}", f"{qv:.2f}{star}"])
    
    rows.sort(key=lambda x: (x[0], x[1], str(x[2])))
    
    table_str = "\n--- Learned Strategies (Sample) ---\n"
    sample_rows = rows[:20]
    table_str += tabulate(sample_rows, headers=["Auction Type", "Agent ID", "State", "Action", "Q-Value"])
    log_print(table_str)
    
    strategies_file = f"{OUTPUT_DIR}/learned_strategies.txt"
    with open(strategies_file, 'w') as f:
        f.write(tabulate(rows, headers=["Auction Type", "Agent ID", "State", "Action", "Q-Value"]))
    log_print(f"Saved complete strategy table to: {strategies_file}")
    
    return rows

def tabulate_retaliation_analysis(fpa_ret0, fpa_ret1, spa_ret0, spa_ret1):
    table_data = [
        ["FPA", 0, 
         fpa_ret0["retaliation_count"], 
         fpa_ret0["total_possible"],
         round(fpa_ret0["ratio"] * 100, 2)],
        ["FPA", 1, 
         fpa_ret1["retaliation_count"], 
         fpa_ret1["total_possible"],
         round(fpa_ret1["ratio"] * 100, 2)],
        ["SPA", 0, 
         spa_ret0["retaliation_count"], 
         spa_ret0["total_possible"],
         round(spa_ret0["ratio"] * 100, 2)],
        ["SPA", 1,
         spa_ret1["retaliation_count"], 
         spa_ret1["total_possible"],
         round(spa_ret1["ratio"] * 100, 2)]
    ]
    headers = ["Auction Type", "Agent ID", "Retaliation Count", "Total Possible States", "Retaliation Ratio (%)"]
    
    table_str = "\n--- Retaliation Analysis ---\n"
    table_str += tabulate(table_data, headers=headers)
    log_print(table_str)

# -----------------------------------------------------------
# Collusion Analysis Functions
# -----------------------------------------------------------

def compute_retaliation_ratio(qtable, state_mode="winning"):
    """
    Computes the retaliation ratio - the fraction of states where an agent
    increases its bid in response to being outbid.
    
    Parameters:
    -----------
    qtable : dict
        Q-table for the agent being analyzed
    state_mode : str
        "winning" for (my_bid, winning_bid) state representation
        "none" for (my_bid,) state representation (no disclosure case)
        
    Returns:
    --------
    dict
        Dictionary with retaliation count, total possible states, and ratio
    """
    if state_mode == "none":
        # In no disclosure, agent doesn't know if it was outbid, so no retaliation is possible
        return {
            "retaliation_count": 0, 
            "higher_count": 0, 
            "ratio": 0.0,
            "message": "Retaliation not possible in 'none' state mode as agent cannot observe opponent's bid"
        }
    
    retaliation_count = 0
    higher_count = 0
    
    for state, action_dict in qtable.items():
        if len(state) < 2 or state == ("START",):
            continue
        
        my_last_bid = state[0]
        winning_bid = state[1]  # In "winning" mode, this is the winning bid

        # If winning bid > my last bid, I lost the auction
        if winning_bid > my_last_bid:
            higher_count += 1
            # Get best action (bid) for this state
            best_action = max(action_dict, key=action_dict.get)
            if best_action > my_last_bid:  # I increase my bid -> 'retaliation'
                retaliation_count += 1
    
    ratio = (retaliation_count / higher_count) if higher_count > 0 else 0.0
    
    return {
        "retaliation_count": retaliation_count,
        "higher_count": higher_count,
        "ratio": ratio
    }

def compute_best_response_consistency(qtable_self, qtable_opp, state_mode="winning"):
    """
    Fraction of states where the chosen action in qtable_self is truly a best-response
    against qtable_opp's best action.
    
    Parameters:
    -----------
    qtable_self : dict
        Q-table for the agent being analyzed
    qtable_opp : dict
        Q-table for the opponent
    state_mode : str
        "winning" for (my_bid, winning_bid) state representation
        "none" for (my_bid,) state representation (no disclosure case)
        
    Returns:
    --------
    float
        Value in [0,1] representing the fraction of states where the agent's action
        is truly optimal
    """
    def Q(qt, s, a):
        return qt.get(s, {}).get(a, None)
    
    def best_action(qt, s):
        if s not in qt or not qt[s]:
            return None
        return max(qt[s], key=qt[s].get)
    
    valid_count = 0
    br_violations = 0

    for state, my_actions_dict in qtable_self.items():
        if state == ("START",):
            continue
            
        # Check appropriate state format based on state_mode
        if state_mode == "none" and len(state) != 1:
            continue
        if state_mode == "winning" and len(state) != 2:
            continue
        
        # My "greedy" action in this state
        my_chosen_act = best_action(qtable_self, state)
        if my_chosen_act is None:
            continue
        
        # Check if there's a strictly better alternative than my_chosen_act
        my_chosen_Q = Q(qtable_self, state, my_chosen_act)
        if my_chosen_Q is None:
            continue
        
        valid_count += 1

        # If any alternative has a strictly higher Q-value, that is a violation
        for alt_a in my_actions_dict.keys():
            alt_Q = Q(qtable_self, state, alt_a)
            if alt_Q and alt_Q > my_chosen_Q + 1e-9:
                br_violations += 1
                break
    
    if valid_count == 0:
        return 1.0  # no states => trivially consistent
    
    consistency = 1.0 - (br_violations / valid_count)
    
    return {
        "valid_count": valid_count,
        "br_violations": br_violations,
        "consistency": consistency
    }

def compute_conditional_correlation(actions_df, window_size=10000):
    """
    Compute correlation between one agent's bid and the other agent's next bid.
    
    Parameters:
    -----------
    actions_df : pandas.DataFrame
        DataFrame with columns ['t', 'auction_type', 'bidder', 'chosen_bid']
    window_size : int, optional
        Number of most recent actions to consider, default 10000
        
    Returns:
    --------
    dict
        Dictionary with correlation values for each auction type
    """
    import pandas as pd
    
    results = {}
    
    # Process for each auction type separately
    for auction_type in actions_df['auction_type'].unique():
        # Filter for this auction type and take last window_size*2 entries
        # (multiplied by 2 because each round has 2 entries, one per agent)
        df_filtered = actions_df[actions_df['auction_type'] == auction_type].tail(window_size*2)
        
        # Pivot to get separate columns for each bidder's bids
        df_pivoted = df_filtered.pivot_table(index='t', columns='bidder', values='chosen_bid')
        df_pivoted.columns = ['bid0', 'bid1']
        
        # Create lagged columns
        df_pivoted['bid0_next'] = df_pivoted['bid0'].shift(-1)
        df_pivoted['bid1_next'] = df_pivoted['bid1'].shift(-1)
        
        # Drop rows with NaN (due to shifting)
        df_pivoted.dropna(inplace=True)
        
        # Calculate correlations
        corr_0next_1 = df_pivoted['bid0_next'].corr(df_pivoted['bid1'])
        corr_1next_0 = df_pivoted['bid1_next'].corr(df_pivoted['bid0'])
        
        results[auction_type] = {
            'corr_bidder0_next_vs_bidder1': corr_0next_1,
            'corr_bidder1_next_vs_bidder0': corr_1next_0,
            'num_observations': len(df_pivoted)
        }
    
    return results

def test_deviation_response(qtable, state, higher_bid, state_mode="winning"):
    """
    Test how an agent would respond if it observes a higher bid.
    For 'winning' mode, we set the winning bid to higher_bid.
    For 'none' mode, the agent doesn't observe the higher bid, so we keep the state as is.
    
    Parameters:
    -----------
    qtable : dict
        Q-table for the agent being analyzed
    state : tuple
        Current state
    higher_bid : float
        Bid value higher than the current bid
    state_mode : str
        "winning" for (my_bid, winning_bid) state representation
        "none" for (my_bid,) state representation (no disclosure case)
        
    Returns:
    --------
    dict
        Dictionary with information about the response
    """
    if state_mode == "none":
        # In no disclosure, agent doesn't see opponent's bid
        # so we just check if it would naturally increase its bid
        my_bid = state[0]
        
        # If state not in qtable, can't evaluate
        if state not in qtable:
            return {
                "state": state,
                "is_retaliatory": False,
                "best_action": None,
                "message": "State not in Q-table"
            }
        
        # Get best action
        best_action = max(qtable[state], key=qtable[state].get)
        
        # Check if best action is higher than current bid
        is_retaliatory = (best_action > my_bid)
        
        return {
            "state": state,
            "is_retaliatory": is_retaliatory,
            "best_action": best_action,
            "message": "Agent doesn't observe opponent's bid in 'none' mode"
        }
    
    else:  # state_mode == "winning"
        my_bid = state[0]
        
        # Next state would have winning bid = higher_bid (since higher_bid > my_bid)
        next_state = (my_bid, higher_bid)
        
        # If next_state not in qtable, can't evaluate
        if next_state not in qtable:
            return {
                "state": state,
                "next_state": next_state,
                "is_retaliatory": False,
                "best_action": None,
                "message": "Next state not in Q-table"
            }
        
        # Get best action for the next state
        best_action = max(qtable[next_state], key=qtable[next_state].get)
        
        # Check if best action is higher than current bid
        is_retaliatory = (best_action > my_bid)
        
        return {
            "state": state,
            "next_state": next_state,
            "is_retaliatory": is_retaliatory,
            "best_action": best_action
        }

def analyze_collusion(fpa_qtable0, fpa_qtable1, spa_qtable0, spa_qtable1, fpa_actions, spa_actions, state_mode="winning"):
    """
    Run a comprehensive collusion analysis using all four tests.
    
    Parameters:
    -----------
    fpa_qtable0, fpa_qtable1 : dict
        Q-tables for FPA auction
    spa_qtable0, spa_qtable1 : dict
        Q-tables for SPA auction
    fpa_actions, spa_actions : list
        Action logs for FPA and SPA
    state_mode : str
        "winning" for (my_bid, winning_bid) state representation
        "none" for (my_bid,) state representation (no disclosure case)
        
    Returns:
    --------
    dict
        Dictionary with all analysis results
    """
    import pandas as pd
    import numpy as np
    
    results = {
        "state_mode": state_mode,
        "retaliation_ratio": {},
        "best_response_consistency": {},
        "conditional_correlation": {},
        "deviation_response": {}
    }
    
    # 1. Retaliation Ratio
    results["retaliation_ratio"]["FPA_bidder0"] = compute_retaliation_ratio(fpa_qtable0, state_mode)
    results["retaliation_ratio"]["FPA_bidder1"] = compute_retaliation_ratio(fpa_qtable1, state_mode)
    results["retaliation_ratio"]["SPA_bidder0"] = compute_retaliation_ratio(spa_qtable0, state_mode)
    results["retaliation_ratio"]["SPA_bidder1"] = compute_retaliation_ratio(spa_qtable1, state_mode)
    
    # 2. Best Response Consistency
    results["best_response_consistency"]["FPA_bidder0"] = compute_best_response_consistency(fpa_qtable0, fpa_qtable1, state_mode)
    results["best_response_consistency"]["FPA_bidder1"] = compute_best_response_consistency(fpa_qtable1, fpa_qtable0, state_mode)
    results["best_response_consistency"]["SPA_bidder0"] = compute_best_response_consistency(spa_qtable0, spa_qtable1, state_mode)
    results["best_response_consistency"]["SPA_bidder1"] = compute_best_response_consistency(spa_qtable1, spa_qtable0, state_mode)
    
    # 3. Conditional Correlation
    # Convert actions lists to DataFrame
    fpa_df = pd.DataFrame(fpa_actions)
    spa_df = pd.DataFrame(spa_actions)
    all_actions_df = pd.concat([fpa_df, spa_df])
    
    results["conditional_correlation"] = compute_conditional_correlation(all_actions_df)
    
    # 4. Deviation Response Tests
    # We'll test a few representative states for each agent/auction
    
    # For 'none' mode, states are just (bid,)
    if state_mode == "none":
        test_bids = [0.2, 0.4, 0.6, 0.8]
        higher_bids = [0.4, 0.6, 0.8, 1.0]  # higher bids to test deviation response
        
        for bid, higher_bid in zip(test_bids, higher_bids):
            state = (bid,)
            
            # For each agent and auction type
            results["deviation_response"][f"FPA_bidder0_state{bid}"] = test_deviation_response(
                fpa_qtable0, state, higher_bid, state_mode)
            results["deviation_response"][f"FPA_bidder1_state{bid}"] = test_deviation_response(
                fpa_qtable1, state, higher_bid, state_mode)
            results["deviation_response"][f"SPA_bidder0_state{bid}"] = test_deviation_response(
                spa_qtable0, state, higher_bid, state_mode)
            results["deviation_response"][f"SPA_bidder1_state{bid}"] = test_deviation_response(
                spa_qtable1, state, higher_bid, state_mode)
    
    # For 'winning' mode, states are (my_bid, winning_bid)
    else:
        test_states = [(0.2, 0.2), (0.4, 0.4), (0.6, 0.6), (0.8, 0.8)]  # starting with equal bids
        higher_bids = [0.4, 0.6, 0.8, 1.0]  # higher bids to test deviation response
        
        for state, higher_bid in zip(test_states, higher_bids):
            # For each agent and auction type
            results["deviation_response"][f"FPA_bidder0_state{state}"] = test_deviation_response(
                fpa_qtable0, state, higher_bid, state_mode)
            results["deviation_response"][f"FPA_bidder1_state{state}"] = test_deviation_response(
                fpa_qtable1, state, higher_bid, state_mode)
            results["deviation_response"][f"SPA_bidder0_state{state}"] = test_deviation_response(
                spa_qtable0, state, higher_bid, state_mode)
            results["deviation_response"][f"SPA_bidder1_state{state}"] = test_deviation_response(
                spa_qtable1, state, higher_bid, state_mode)
    
    # 5. Add a summary analysis interpreting the results
    results["summary"] = analyze_collusion_summary(results)
    
    return results

def analyze_collusion_summary(results):
    """
    Provide a summary analysis of collusion test results.
    
    Parameters:
    -----------
    results : dict
        Results from analyze_collusion
        
    Returns:
    --------
    dict
        Summary and interpretation of results
    """
    summary = {
        "FPA": {
            "collusion_score": 0,
            "interpretation": "",
            "key_findings": []
        },
        "SPA": {
            "collusion_score": 0,
            "interpretation": "",
            "key_findings": []
        }
    }
    
    state_mode = results["state_mode"]
    
    # Process FPA
    if state_mode == "winning":
        # For winning bid mode, check retaliation ratio
        fpa_ret0 = results["retaliation_ratio"]["FPA_bidder0"]["ratio"]
        fpa_ret1 = results["retaliation_ratio"]["FPA_bidder1"]["ratio"]
        
        fpa_retaliation_avg = (fpa_ret0 + fpa_ret1) / 2
        
        # Add to collusion score based on retaliation
        if fpa_retaliation_avg > 0.8:
            summary["FPA"]["collusion_score"] += 2
            summary["FPA"]["key_findings"].append(f"High retaliation ratio ({fpa_retaliation_avg:.2f}): Strong evidence of punishment")
        elif fpa_retaliation_avg > 0.5:
            summary["FPA"]["collusion_score"] += 1
            summary["FPA"]["key_findings"].append(f"Moderate retaliation ratio ({fpa_retaliation_avg:.2f}): Some evidence of punishment")
        else:
            summary["FPA"]["key_findings"].append(f"Low retaliation ratio ({fpa_retaliation_avg:.2f}): Little evidence of punishment")
    
    # Check best response consistency for FPA
    fpa_br0 = results["best_response_consistency"]["FPA_bidder0"]["consistency"]
    fpa_br1 = results["best_response_consistency"]["FPA_bidder1"]["consistency"]
    
    fpa_br_avg = (fpa_br0 + fpa_br1) / 2
    
    if fpa_br_avg > 0.9:
        summary["FPA"]["collusion_score"] += 1
        summary["FPA"]["key_findings"].append(f"High best-response consistency ({fpa_br_avg:.2f}): Agents have optimized their strategies")
    
    # Check conditional correlation for FPA
    if "FPA" in results["conditional_correlation"]:
        fpa_corr0 = results["conditional_correlation"]["FPA"]["corr_bidder0_next_vs_bidder1"]
        fpa_corr1 = results["conditional_correlation"]["FPA"]["corr_bidder1_next_vs_bidder0"]
        
        fpa_corr_avg = (fpa_corr0 + fpa_corr1) / 2
        
        if fpa_corr_avg < -0.5:
            summary["FPA"]["collusion_score"] += 2
            summary["FPA"]["key_findings"].append(f"Strong negative correlation ({fpa_corr_avg:.2f}): Indicates retaliatory patterns")
        elif fpa_corr_avg < -0.2:
            summary["FPA"]["collusion_score"] += 1
            summary["FPA"]["key_findings"].append(f"Moderate negative correlation ({fpa_corr_avg:.2f}): Some retaliatory patterns")
        elif fpa_corr_avg > 0.5:
            summary["FPA"]["key_findings"].append(f"Strong positive correlation ({fpa_corr_avg:.2f}): Indicates coupling rather than collusion")
    
    # Check deviation responses for FPA
    fpa_dev_count = 0
    fpa_ret_count = 0
    
    for key, value in results["deviation_response"].items():
        if key.startswith("FPA_") and "is_retaliatory" in value:
            fpa_dev_count += 1
            if value["is_retaliatory"]:
                fpa_ret_count += 1
    
    if fpa_dev_count > 0:
        fpa_dev_ratio = fpa_ret_count / fpa_dev_count
        
        if fpa_dev_ratio > 0.8:
            summary["FPA"]["collusion_score"] += 2
            summary["FPA"]["key_findings"].append(f"Strong deviation response ({fpa_dev_ratio:.2f}): Agents quickly punish deviations")
        elif fpa_dev_ratio > 0.5:
            summary["FPA"]["collusion_score"] += 1
            summary["FPA"]["key_findings"].append(f"Moderate deviation response ({fpa_dev_ratio:.2f}): Some punishment for deviations")
    
    # Process SPA similarly
    if state_mode == "winning":
        spa_ret0 = results["retaliation_ratio"]["SPA_bidder0"]["ratio"]
        spa_ret1 = results["retaliation_ratio"]["SPA_bidder1"]["ratio"]
        
        spa_retaliation_avg = (spa_ret0 + spa_ret1) / 2
        
        if spa_retaliation_avg > 0.8:
            summary["SPA"]["collusion_score"] += 2
            summary["SPA"]["key_findings"].append(f"High retaliation ratio ({spa_retaliation_avg:.2f}): Strong evidence of punishment")
        elif spa_retaliation_avg > 0.5:
            summary["SPA"]["collusion_score"] += 1
            summary["SPA"]["key_findings"].append(f"Moderate retaliation ratio ({spa_retaliation_avg:.2f}): Some evidence of punishment")
        else:
            summary["SPA"]["key_findings"].append(f"Low retaliation ratio ({spa_retaliation_avg:.2f}): Little evidence of punishment")
    
    # Best response consistency for SPA
    spa_br0 = results["best_response_consistency"]["SPA_bidder0"]["consistency"]
    spa_br1 = results["best_response_consistency"]["SPA_bidder1"]["consistency"]
    
    spa_br_avg = (spa_br0 + spa_br1) / 2
    
    if spa_br_avg > 0.9:
        summary["SPA"]["collusion_score"] += 1
        summary["SPA"]["key_findings"].append(f"High best-response consistency ({spa_br_avg:.2f}): Agents have optimized their strategies")
    
    # Conditional correlation for SPA
    if "SPA" in results["conditional_correlation"]:
        spa_corr0 = results["conditional_correlation"]["SPA"]["corr_bidder0_next_vs_bidder1"]
        spa_corr1 = results["conditional_correlation"]["SPA"]["corr_bidder1_next_vs_bidder0"]
        
        spa_corr_avg = (spa_corr0 + spa_corr1) / 2
        
        if spa_corr_avg < -0.5:
            summary["SPA"]["collusion_score"] += 2
            summary["SPA"]["key_findings"].append(f"Strong negative correlation ({spa_corr_avg:.2f}): Indicates retaliatory patterns")
        elif spa_corr_avg < -0.2:
            summary["SPA"]["collusion_score"] += 1
            summary["SPA"]["key_findings"].append(f"Moderate negative correlation ({spa_corr_avg:.2f}): Some retaliatory patterns")
        elif spa_corr_avg > 0.5:
            summary["SPA"]["key_findings"].append(f"Strong positive correlation ({spa_corr_avg:.2f}): Indicates coupling rather than collusion")
    
    # Deviation responses for SPA
    spa_dev_count = 0
    spa_ret_count = 0
    
    for key, value in results["deviation_response"].items():
        if key.startswith("SPA_") and "is_retaliatory" in value:
            spa_dev_count += 1
            if value["is_retaliatory"]:
                spa_ret_count += 1
    
    if spa_dev_count > 0:
        spa_dev_ratio = spa_ret_count / spa_dev_count
        
        if spa_dev_ratio > 0.8:
            summary["SPA"]["collusion_score"] += 2
            summary["SPA"]["key_findings"].append(f"Strong deviation response ({spa_dev_ratio:.2f}): Agents quickly punish deviations")
        elif spa_dev_ratio > 0.5:
            summary["SPA"]["collusion_score"] += 1
            summary["SPA"]["key_findings"].append(f"Moderate deviation response ({spa_dev_ratio:.2f}): Some punishment for deviations")
    
    # Final interpretation
    for auction_type in ["FPA", "SPA"]:
        if summary[auction_type]["collusion_score"] >= 5:
            summary[auction_type]["interpretation"] = "Strong evidence of collusion with punishment strategies"
        elif summary[auction_type]["collusion_score"] >= 3:
            summary[auction_type]["interpretation"] = "Moderate evidence of collusion"
        elif summary[auction_type]["collusion_score"] >= 1:
            summary[auction_type]["interpretation"] = "Weak evidence of collusion, likely just 'coupling'"
        else:
            summary[auction_type]["interpretation"] = "No evidence of collusion, agents appear to act independently"
    
    return summary

def np_to_python(obj):
    """
    Convert NumPy types to native Python types for JSON serialization
    """
    import numpy as np
    
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                       np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, dict):
        # Convert dictionary keys to strings to ensure they're hashable
        return {str(np_to_python(key)): np_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [np_to_python(item) for item in obj]
    else:
        return obj


def run_perturbation_analysis(fpa_qtable0, fpa_qtable1, spa_qtable0, spa_qtable1, bid_actions, state_mode="winning"):
    """
    Perform perturbation checks to analyze strategic responses.
    
    Parameters:
    -----------
    fpa_qtable0, fpa_qtable1 : dict
        Q-tables for FPA auction
    spa_qtable0, spa_qtable1 : dict
        Q-tables for SPA auction
    bid_actions : list
        The list of possible bid values
    state_mode : str
        Information regime mode
        
    Returns:
    --------
    dict
        Collusion analysis results
    """
    # Step 1: Run a simulation to determine state visitation frequencies
    log_print("Computing state visitation frequencies...")
    fpa_visitation = simulate_visitation_frequency(fpa_qtable0, fpa_qtable1, bid_actions, auction_type="FPA", 
                                              state_mode=state_mode, num_steps=10000)
    spa_visitation = simulate_visitation_frequency(spa_qtable0, spa_qtable1, bid_actions, auction_type="SPA", 
                                              state_mode=state_mode, num_steps=10000)
    
    # Step 2: Perform perturbation tests on each visited state
    log_print("Testing perturbation responses...")
    fpa_perturbation_results = test_perturbations(fpa_qtable0, fpa_qtable1, fpa_visitation, bid_actions, state_mode)
    spa_perturbation_results = test_perturbations(spa_qtable0, spa_qtable1, spa_visitation, bid_actions, state_mode)
    
    # Step 3 & 4: Calculate collusion indices
    log_print("Calculating collusion indices...")
    fpa_collusion = calculate_collusion_index(fpa_perturbation_results, fpa_visitation)
    spa_collusion = calculate_collusion_index(spa_perturbation_results, spa_visitation)
    
    results = {
        "FPA": {
            "visitation_frequency": fpa_visitation,
            "perturbation_results": fpa_perturbation_results,
            "collusion_index": fpa_collusion
        },
        "SPA": {
            "visitation_frequency": spa_visitation,
            "perturbation_results": spa_perturbation_results,
            "collusion_index": spa_collusion
        }
    }
    
    return results


def simulate_visitation_frequency(qtable0, qtable1, bid_actions, auction_type, state_mode, num_steps=10000):
    """
    Simulate auction using deterministic Q-tables and record state visitation frequencies.
    """
    from collections import Counter
    import numpy as np
    
    visitation = Counter()
    last_bids = [None, None]  # Initial bids
    
    # Helper function to get best action from Q-table
    def best_action(qtable, state):
        if state not in qtable or not qtable[state]:
            # If state not in Q-table or has no actions, return random bid
            return np.random.choice(bid_actions)
        return max(qtable[state].items(), key=lambda x: x[1])[0]
    
    for t in range(num_steps):
        # Get states
        s0 = get_state(last_bids, 0, state_mode)
        s1 = get_state(last_bids, 1, state_mode)
        
        # Record state visitations
        visitation[s0] += 1
        visitation[s1] += 1
        
        # Choose best actions according to Q-tables
        a0 = best_action(qtable0, s0)
        a1 = best_action(qtable1, s1)
        
        # Determine auction outcome
        bids = [a0, a1]
        max_bid = max(bids)
        winners = [i for i, b in enumerate(bids) if b == max_bid]
        
        if len(winners) > 1:
            winner = coin_flip_winner(winners)  # Reuse original function
        else:
            winner = winners[0]
            
        # Update bids for next iteration
        last_bids = [a0, a1]
    
    # Normalize to get frequencies
    total_visits = sum(visitation.values())
    frequencies = {state: count / total_visits for state, count in visitation.items()}
    
    return frequencies


def test_perturbations(qtable0, qtable1, visitation, bid_actions, state_mode):
    """
    Test perturbation responses for all visited states.
    """
    results = {}
    
    # Helper function to perturb bid (up or down)
    def perturb_bid(bid, direction, bid_actions):
        idx = bid_actions.index(bid)
        if direction == "up" and idx < len(bid_actions) - 1:
            return bid_actions[idx + 1]
        elif direction == "down" and idx > 0:
            return bid_actions[idx - 1]
        return bid  # No change if at boundary
    
    # Helper to get best action
    def best_action(qtable, state):
        if state not in qtable or not qtable[state]:
            return None
        return max(qtable[state].items(), key=lambda x: x[1])[0]
    
    # Process each visited state
    for state in visitation:
        if state == ("START",) or visitation[state] < 0.001:  # Skip rarely visited states
            continue
        
        results[state] = {
            "agent0": {"retaliation": False, "accommodation": False},
            "agent1": {"retaliation": False, "accommodation": False}
        }
        
        # Process based on state mode
        if state_mode == "none":
            # In 'none' mode, we need to test reactions indirectly
            
            # For agent0 (state is my_bid)
            if len(state) == 1 and state in qtable0:
                my_bid = state[0]
                best_act0 = best_action(qtable0, state)
                
                # Test if agent is retaliatory by checking if it naturally increases bid
                # (this is a simplification since in 'none' mode agents don't observe opponent bids)
                if best_act0 is not None and best_act0 > my_bid:
                    results[state]["agent0"]["retaliation"] = True
                
                # Test if agent accommodates by checking if it maintains or lowers bid
                if best_act0 is not None and best_act0 <= my_bid:
                    results[state]["agent0"]["accommodation"] = True
            
            # For agent1 (state is my_bid)
            if len(state) == 1 and state in qtable1:
                my_bid = state[0]
                best_act1 = best_action(qtable1, state)
                
                # Similar logic for agent1
                if best_act1 is not None and best_act1 > my_bid:
                    results[state]["agent1"]["retaliation"] = True
                
                if best_act1 is not None and best_act1 <= my_bid:
                    results[state]["agent1"]["accommodation"] = True
                    
        elif state_mode == "winning":
            # In 'winning' mode, state is (my_bid, winning_bid)
            
            # For agent0
            if len(state) == 2 and state in qtable0:
                my_bid = state[0]
                winning_bid = state[1]
                
                # If I lost, test retaliation by checking response to higher opponent bid
                if winning_bid > my_bid:
                    # Create a "higher winning bid" perturbed state
                    higher_win = perturb_bid(winning_bid, "up", bid_actions)
                    next_state_up = (my_bid, higher_win)
                    
                    # Check agent's best response to this perturbation
                    if next_state_up in qtable0:
                        best_act_up = best_action(qtable0, next_state_up)
                        if best_act_up is not None and best_act_up > my_bid:
                            results[state]["agent0"]["retaliation"] = True
                
                    # Create a "lower winning bid" perturbed state
                    lower_win = perturb_bid(winning_bid, "down", bid_actions)
                    next_state_down = (my_bid, lower_win)
                    
                    # Check agent's best response to this perturbation
                    if next_state_down in qtable0:
                        best_act_down = best_action(qtable0, next_state_down)
                        if best_act_down is not None and best_act_down <= my_bid:
                            results[state]["agent0"]["accommodation"] = True
            
            # For agent1 - apply same logic
            if len(state) == 2 and state in qtable1:
                my_bid = state[0]
                winning_bid = state[1]
                
                if winning_bid > my_bid:
                    # Test retaliation
                    higher_win = perturb_bid(winning_bid, "up", bid_actions)
                    next_state_up = (my_bid, higher_win)
                    
                    if next_state_up in qtable1:
                        best_act_up = best_action(qtable1, next_state_up)
                        if best_act_up is not None and best_act_up > my_bid:
                            results[state]["agent1"]["retaliation"] = True
                
                    # Test accommodation
                    lower_win = perturb_bid(winning_bid, "down", bid_actions)
                    next_state_down = (my_bid, lower_win)
                    
                    if next_state_down in qtable1:
                        best_act_down = best_action(qtable1, next_state_down)
                        if best_act_down is not None and best_act_down <= my_bid:
                            results[state]["agent1"]["accommodation"] = True
                            
        elif state_mode == "full":
            # In 'full' mode, state is (my_bid, opponent_bid)
            
            # For agent0
            if len(state) == 2 and state in qtable0:
                my_bid = state[0]
                opp_bid = state[1]
                
                # Test retaliation - if opponent increases bid
                higher_opp = perturb_bid(opp_bid, "up", bid_actions)
                next_state_up = (my_bid, higher_opp)
                
                if next_state_up in qtable0:
                    best_act_up = best_action(qtable0, next_state_up)
                    if best_act_up is not None and best_act_up > my_bid:
                        results[state]["agent0"]["retaliation"] = True
                
                # Test accommodation - if opponent decreases bid
                lower_opp = perturb_bid(opp_bid, "down", bid_actions)
                next_state_down = (my_bid, lower_opp)
                
                if next_state_down in qtable0:
                    best_act_down = best_action(qtable0, next_state_down)
                    if best_act_down is not None and best_act_down <= my_bid:
                        results[state]["agent0"]["accommodation"] = True
            
            # For agent1 - apply same logic with swapped indices
            if len(state) == 2 and state in qtable1:
                my_bid = state[0]
                opp_bid = state[1]
                
                # Test retaliation
                higher_opp = perturb_bid(opp_bid, "up", bid_actions)
                next_state_up = (my_bid, higher_opp)
                
                if next_state_up in qtable1:
                    best_act_up = best_action(qtable1, next_state_up)
                    if best_act_up is not None and best_act_up > my_bid:
                        results[state]["agent1"]["retaliation"] = True
                
                # Test accommodation
                lower_opp = perturb_bid(opp_bid, "down", bid_actions)
                next_state_down = (my_bid, lower_opp)
                
                if next_state_down in qtable1:
                    best_act_down = best_action(qtable1, next_state_down)
                    if best_act_down is not None and best_act_down <= my_bid:
                        results[state]["agent1"]["accommodation"] = True
    
    return results


def calculate_collusion_index(perturbation_results, visitation):
    """
    Calculate collusion index by weighting perturbation results by visitation frequency.
    """
    weighted_retaliation0 = 0
    weighted_accommodation0 = 0
    weighted_retaliation1 = 0
    weighted_accommodation1 = 0
    total_weight = 0
    
    for state, results in perturbation_results.items():
        if state in visitation:
            weight = visitation[state]
            total_weight += weight
            
            # Agent 0
            if results["agent0"]["retaliation"]:
                weighted_retaliation0 += weight
            if results["agent0"]["accommodation"]:
                weighted_accommodation0 += weight
            
            # Agent 1
            if results["agent1"]["retaliation"]:
                weighted_retaliation1 += weight
            if results["agent1"]["accommodation"]:
                weighted_accommodation1 += weight
    
    if total_weight > 0:
        weighted_retaliation0 /= total_weight
        weighted_accommodation0 /= total_weight
        weighted_retaliation1 /= total_weight
        weighted_accommodation1 /= total_weight
    
    # Calculate collusion indices
    collusion_index0 = (weighted_retaliation0 + weighted_accommodation0) / 2
    collusion_index1 = (weighted_retaliation1 + weighted_accommodation1) / 2
    overall_index = (collusion_index0 + collusion_index1) / 2
    
    return {
        "agent0": {
            "retaliation": weighted_retaliation0,
            "accommodation": weighted_accommodation0,
            "collusion_index": collusion_index0
        },
        "agent1": {
            "retaliation": weighted_retaliation1,
            "accommodation": weighted_accommodation1,
            "collusion_index": collusion_index1
        },
        "overall_collusion_index": overall_index,
        "interpretation": interpret_collusion_index(overall_index)
    }


def interpret_collusion_index(index):
    """
    Provide an interpretation of the collusion index.
    """
    if index > 0.8:
        return "Strong evidence of collusion with effective punishment strategies"
    elif index > 0.6:
        return "Moderate evidence of collusion with some punishment mechanisms"
    elif index > 0.4:
        return "Some evidence of strategic coordination, weak collusion"
    elif index > 0.2:
        return "Mostly independent behavior with minimal coordination"
    else:
        return "No evidence of collusion, agents act independently"


def save_perturbation_analysis(results, state_mode, output_dir=f"{OUTPUT_DIR}"):
    """
    Save perturbation analysis results to files.
    
    Parameters:
    -----------
    results : dict
        Results from run_perturbation_analysis
    state_mode : str
        Information regime mode
    output_dir : str
        Directory to save results
    """
    import os
    import json
    from tabulate import tabulate
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert NumPy types to Python native types
    python_results = np_to_python(results)
    
    # Save full results as JSON
    with open(f"{output_dir}/perturbation_analysis_{state_mode}.json", "w") as f:
        json.dump(python_results, f, indent=2)
    
    # Create summary tables
    
    # 1. Collusion Index Table
    index_rows = []
    index_headers = ["Auction Type", "Agent", "Retaliation", "Accommodation", "Collusion Index"]
    
    for auction_type in ["FPA", "SPA"]:
        for agent in ["agent0", "agent1"]:
            collusion_data = results[auction_type]["collusion_index"][agent]
            index_rows.append([
                auction_type,
                agent,
                f"{collusion_data['retaliation']:.4f}",
                f"{collusion_data['accommodation']:.4f}",
                f"{collusion_data['collusion_index']:.4f}"
            ])
        
        # Add overall row
        overall = results[auction_type]["collusion_index"]["overall_collusion_index"]
        index_rows.append([
            auction_type,
            "Overall",
            "",
            "",
            f"{overall:.4f}"
        ])
    
    index_table = tabulate(index_rows, headers=index_headers, tablefmt="pipe")
    
    # 2. Interpretation Table
    interp_rows = []
    interp_headers = ["Auction Type", "Collusion Index", "Interpretation"]
    
    for auction_type in ["FPA", "SPA"]:
        interp_rows.append([
            auction_type,
            f"{results[auction_type]['collusion_index']['overall_collusion_index']:.4f}",
            results[auction_type]["collusion_index"]["interpretation"]
        ])
    
    interp_table = tabulate(interp_rows, headers=interp_headers, tablefmt="pipe")
    
    # Write tables to file
    with open(f"{output_dir}/perturbation_tables_{state_mode}.md", "w") as f:
        f.write(f"# Perturbation Analysis - {state_mode} Information Regime\n\n")
        
        f.write("## 1. Collusion Index Summary\n\n")
        f.write(index_table)
        f.write("\n\n")
        
        f.write("## 2. Interpretation\n\n")
        f.write(interp_table)
        f.write("\n\n")
    
    # Create LaTeX versions of tables
    with open(f"{output_dir}/perturbation_tables_{state_mode}.tex", "w") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{Perturbation Analysis Summary - {state_mode}}}\n")
        f.write(f"\\label{{tab:perturbation-{state_mode}}}\n")
        f.write(tabulate(index_rows, headers=index_headers, tablefmt="latex"))
        f.write("\n\\end{table}\n\n")
        
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{Perturbation Analysis Interpretation - {state_mode}}}\n")
        f.write(f"\\label{{tab:perturbation-interp-{state_mode}}}\n")
        f.write(tabulate(interp_rows, headers=interp_headers, tablefmt="latex"))
        f.write("\n\\end{table}\n\n")
    
    log_print(f"Saved perturbation analysis for {state_mode} to {output_dir}")
    
    return True

# Then modify the save_collusion_analysis function to use this helper function:
def save_collusion_analysis(results, state_mode, output_dir=f"{OUTPUT_DIR}"):
    """
    Save collusion analysis results to files.
    """
    import os
    import json
    from tabulate import tabulate
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert NumPy types to Python native types before saving to JSON
    python_results = np_to_python(results)
    
    # Save full results as JSON
    with open(f"{output_dir}/collusion_analysis_{state_mode}.json", "w") as f:
        json.dump(python_results, f, indent=2)
    
    # Rest of the function remains the same...
    # Create summary tables
    
    # 1. Retaliation Ratio
    ret_rows = []
    ret_headers = ["Agent", "Retaliation Count", "Higher Count", "Ratio"]
    
    for key, value in results["retaliation_ratio"].items():
        if "ratio" in value:  # Skip if not applicable (e.g., "none" mode)
            ret_rows.append([
                key,
                value["retaliation_count"],
                value["higher_count"],
                f"{value['ratio']:.4f}"
            ])
    
    ret_table = tabulate(ret_rows, headers=ret_headers, tablefmt="pipe")
    
    # 2. Best Response Consistency
    br_rows = []
    br_headers = ["Agent", "Valid States", "BR Violations", "Consistency"]
    
    for key, value in results["best_response_consistency"].items():
        br_rows.append([
            key,
            value["valid_count"],
            value["br_violations"],
            f"{value['consistency']:.4f}"
        ])
    
    br_table = tabulate(br_rows, headers=br_headers, tablefmt="pipe")
    
    # 3. Conditional Correlation
    corr_rows = []
    corr_headers = ["Auction Type", "Bidder0 next vs Bidder1", "Bidder1 next vs Bidder0", "Observations"]
    
    for key, value in results["conditional_correlation"].items():
        corr_rows.append([
            key,
            f"{value['corr_bidder0_next_vs_bidder1']:.4f}",
            f"{value['corr_bidder1_next_vs_bidder0']:.4f}",
            value['num_observations']
        ])
    
    corr_table = tabulate(corr_rows, headers=corr_headers, tablefmt="pipe")
    
    # 4. Deviation Response Summary
    dev_rows = []
    dev_headers = ["Test Case", "Initial State", "Next State", "Best Action", "Is Retaliatory"]
    
    for key, value in results["deviation_response"].items():
        if "is_retaliatory" in value:
            dev_rows.append([
                key,
                str(value["state"]),
                str(value.get("next_state", "N/A")),
                f"{value.get('best_action', 'N/A')}",
                "Yes" if value["is_retaliatory"] else "No"
            ])
    
    dev_table = tabulate(dev_rows, headers=dev_headers, tablefmt="pipe")
    
    # 5. Summary
    summary_rows = []
    summary_headers = ["Auction Type", "Collusion Score", "Interpretation"]
    
    for key, value in results["summary"].items():
        summary_rows.append([
            key,
            value["collusion_score"],
            value["interpretation"]
        ])
    
    summary_table = tabulate(summary_rows, headers=summary_headers, tablefmt="pipe")
    
    # Detailed key findings
    findings_rows = []
    findings_headers = ["Auction Type", "Key Findings"]
    
    for key, value in results["summary"].items():
        for finding in value["key_findings"]:
            findings_rows.append([key, finding])
    
    findings_table = tabulate(findings_rows, headers=findings_headers, tablefmt="pipe")
    
    # Write tables to file
    with open(f"{output_dir}/collusion_tables_{state_mode}.md", "w") as f:
        f.write(f"# Collusion Analysis - {state_mode} Information Regime\n\n")
        
        f.write("## 1. Summary\n\n")
        f.write(summary_table)
        f.write("\n\n")
        
        f.write("## 2. Key Findings\n\n")
        f.write(findings_table)
        f.write("\n\n")
        
        if len(ret_rows) > 0:
            f.write("## 3. Retaliation Ratio\n\n")
            f.write(ret_table)
            f.write("\n\n")
        
        f.write("## 4. Best Response Consistency\n\n")
        f.write(br_table)
        f.write("\n\n")
        
        f.write("## 5. Conditional Correlation\n\n")
        f.write(corr_table)
        f.write("\n\n")
        
        f.write("## 6. Deviation Response\n\n")
        f.write(dev_table)
        f.write("\n\n")
    
    # Create LaTeX versions of tables
    with open(f"{output_dir}/collusion_tables_{state_mode}.tex", "w") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{Collusion Analysis Summary - {state_mode}}}\n")
        f.write(f"\\label{{tab:collusion-summary-{state_mode}}}\n")
        f.write(tabulate(summary_rows, headers=summary_headers, tablefmt="latex"))
        f.write("\n\\end{table}\n\n")
        
        if len(ret_rows) > 0:
            f.write("\\begin{table}[ht]\n")
            f.write("\\centering\n")
            f.write(f"\\caption{{Retaliation Ratio - {state_mode}}}\n")
            f.write(f"\\label{{tab:retaliation-{state_mode}}}\n")
            f.write(tabulate(ret_rows, headers=ret_headers, tablefmt="latex"))
            f.write("\n\\end{table}\n\n")
        
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{Best Response Consistency - {state_mode}}}\n")
        f.write(f"\\label{{tab:best-response-{state_mode}}}\n")
        f.write(tabulate(br_rows, headers=br_headers, tablefmt="latex"))
        f.write("\n\\end{table}\n\n")
        
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{Conditional Correlation - {state_mode}}}\n")
        f.write(f"\\label{{tab:correlation-{state_mode}}}\n")
        f.write(tabulate(corr_rows, headers=corr_headers, tablefmt="latex"))
        f.write("\n\\end{table}\n\n")
    
    log_print(f"Saved collusion analysis for {state_mode} to {output_dir}")
    
    return True



# Modify the run_experiment_for_info_regime function to add perturbation analysis
def run_experiment_for_info_regime(state_mode, alpha_start, alpha_end, gamma, epsilon_start, epsilon_end, bid_actions, 
                                  max_learning_rounds, rolling_window, bellman_window, seed=42):
    info_regime_name = {
        "none": "No Disclosure",
        "winning": "Winning Bid Only",
        "full": "Full Disclosure"
    }[state_mode]
    
    info_header = f"\n\nINFORMATION REGIME: {info_regime_name}\n"
    info_header += "-"*50
    log_print(info_header)
    
    fpa_q, fpa_act, fpa_qt, fpa_rev, fpa_be = run_case_study(
        "FPA", state_mode=state_mode, seed=seed, alpha_start=alpha_start, alpha_end=alpha_end, 
        gamma=gamma, epsilon_start=epsilon_start, epsilon_end=epsilon_end, bid_actions=bid_actions, 
        max_learning_rounds=max_learning_rounds, rolling_window=rolling_window, bellman_window=bellman_window
    )
    spa_q, spa_act, spa_qt, spa_rev, spa_be = run_case_study(
        "SPA", state_mode=state_mode, seed=seed, alpha_start=alpha_start, alpha_end=alpha_end, 
        gamma=gamma, epsilon_start=epsilon_start, epsilon_end=epsilon_end, bid_actions=bid_actions, 
        max_learning_rounds=max_learning_rounds, rolling_window=rolling_window, bellman_window=bellman_window
    )

    fpa_analysis = analyze_q_tables(fpa_qt[0], fpa_qt[1], "FPA", gamma=gamma)
    spa_analysis = analyze_q_tables(spa_qt[0], spa_qt[1], "SPA", gamma=gamma)
    
    if state_mode != "none":
        fpa_ret0 = analyze_retaliation(fpa_qt[0], state_mode)
        fpa_ret1 = analyze_retaliation(fpa_qt[1], state_mode)
        spa_ret0 = analyze_retaliation(spa_qt[0], state_mode)
        spa_ret1 = analyze_retaliation(spa_qt[1], state_mode)
        tabulate_retaliation_analysis(fpa_ret0, fpa_ret1, spa_ret0, spa_ret1)

    df_act = pd.DataFrame(fpa_act + spa_act)
    df_be = pd.DataFrame(fpa_be + spa_be)
    df_rev = pd.DataFrame(fpa_rev + spa_rev)

    log_print("\nGenerating training dynamics plots (3x2 grid)")
    plot_combined_results(df_act, df_be, df_rev, state_mode, rolling_window)
    
    log_print("\nGenerating bid and price distribution plots (3x2 grid)")
    plot_bid_distribution(fpa_rev, spa_rev, fpa_act, spa_act, state_mode, window_size=rolling_window, bid_actions=bid_actions)

    log_print("\nGenerating learned strategy plots")
    for bidder in [0, 1]:
        plot_learned_strategies(fpa_qt[bidder], spa_qt[bidder], bidder, state_mode, bid_actions)
    
    tabulate_analysis_results(fpa_analysis, spa_analysis)
    strategy_rows = tabulate_combined_strategies(fpa_qt[0], fpa_qt[1], spa_qt[0], spa_qt[1])

    # Add this after your existing analysis but before plotting
    log_print("\nAnalyzing collusion strategies...")
    collusion_results = analyze_collusion(
        fpa_qt[0], fpa_qt[1], spa_qt[0], spa_qt[1], 
        fpa_act, spa_act, 
        state_mode=state_mode
    )
    save_collusion_analysis(collusion_results, state_mode)

    # Add the new perturbation analysis
    log_print("\nPerforming perturbation analysis...")
    perturbation_results = run_perturbation_analysis(
        fpa_qt[0], fpa_qt[1], spa_qt[0], spa_qt[1], 
        bid_actions, 
        state_mode=state_mode
    )
    save_perturbation_analysis(perturbation_results, state_mode)
    
    save_path = f"{OUTPUT_DIR}/{state_mode}_data.npz"
    np.savez_compressed(
        save_path,
        fpa_qtable0=fpa_qt[0],
        fpa_qtable1=fpa_qt[1],
        spa_qtable0=spa_qt[0],
        spa_qtable1=spa_qt[1],
        fpa_revenue=[entry["price"] for entry in fpa_rev],
        spa_revenue=[entry["price"] for entry in spa_rev],
        fpa_actions=[(entry["bidder"], entry["chosen_bid"]) for entry in fpa_act],
        spa_actions=[(entry["bidder"], entry["chosen_bid"]) for entry in spa_act]
    )
    log_print(f"Saved experiment data to: {save_path}")

def print_experiment_settings(alpha_start, alpha_end, gamma, epsilon_start, epsilon_end, bid_actions, max_learning_rounds, rolling_window, bellman_window):
    settings_str = "\n" + "="*50 + "\n"
    settings_str += "EXPERIMENT 1: IDENTICAL VALUES (N=2, vi=1.0)\n"
    settings_str += "="*50 + "\n"
    settings_str += f"Q-Learning Parameters:\n"
    settings_str += f"  Learning Rate (α): {alpha_start} to {alpha_end} (linear decay)\n"
    settings_str += f"  Discount Factor (γ): {gamma}\n"
    settings_str += f"  Exploration Rate (ε): {epsilon_start} to {epsilon_end} (linear decay)\n"
    settings_str += f"  Initial Q-Values: Random in [0, 1]\n"
    settings_str += f"  Training Rounds: {max_learning_rounds:,}\n"
    settings_str += f"Bid Space: {[round(b, 2) for b in bid_actions]}\n"
    settings_str += f"Rolling Window Size: {rolling_window:,}\n"
    settings_str += f"Bellman Error Window Size: {bellman_window:,}\n"
    settings_str += "Information Regimes:\n"
    settings_str += f"  - No Disclosure ('none')\n"
    settings_str += f"  - Winning Bid Only ('winning')\n"
    settings_str += f"  - Full Disclosure ('full')\n"  
    settings_str += "="*50
    
    log_print(settings_str)
    
    with open(f"{OUTPUT_DIR}/experiment_settings.txt", "w") as f:
        f.write(settings_str)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Experiment 1: Identical Values (N=2, vi=1.0)')
    parser.add_argument('--quick', action='store_true', help='Run quick test with reduced parameters')
    args = parser.parse_args()

    # Initialize output directory
    if args.quick:
        _init_output("results/exp1/quick_test")
    else:
        _init_output("results/exp1")

    # Key hyperparameters - easily tune these
    ALPHA_START, ALPHA_END = 0.1, 0.0       # Learning rate
    GAMMA = 0.9                             # Discount factor
    EPSILON_START, EPSILON_END = 1.0, 0.0   # Exploration rate

    if args.quick:
        # Quick test mode: reduced parameters for fast validation
        log_print("=" * 50)
        log_print("QUICK TEST MODE - Reduced parameters for fast validation")
        log_print("=" * 50)
        BID_SPACE_GRANULARITY = 4               # Fewer bid options
        MAX_LEARNING_ROUNDS = 1000              # Reduced training iterations
        ROLLING_WINDOW = 100                    # Smaller window for rolling averages
        BELLMAN_WINDOW = 500                    # Smaller window for Bellman error
        INFO_REGIMES = ["none"]                 # Single info regime only
    else:
        # Full experiment mode
        BID_SPACE_GRANULARITY = 6               # Number of bid options
        MAX_LEARNING_ROUNDS = 1_000_00          # Training iterations
        ROLLING_WINDOW = 10000                  # Window for rolling averages
        BELLMAN_WINDOW = 50000                  # Window for Bellman error
        INFO_REGIMES = ["none", "winning", "full"]

    BID_ACTIONS = np.linspace(0.0, 1.0, BID_SPACE_GRANULARITY).tolist()

    print_experiment_settings(ALPHA_START, ALPHA_END, GAMMA, EPSILON_START, EPSILON_END, BID_ACTIONS,
                            MAX_LEARNING_ROUNDS, ROLLING_WINDOW, BELLMAN_WINDOW)

    # Run experiments for each information regime
    for regime in INFO_REGIMES:
        run_experiment_for_info_regime(regime, ALPHA_START, ALPHA_END, GAMMA, EPSILON_START, EPSILON_END,
                                     BID_ACTIONS, MAX_LEARNING_ROUNDS, ROLLING_WINDOW, BELLMAN_WINDOW)

    log_print(f"Experiment completed. Log files and figures saved to {OUTPUT_DIR}/ directory.")
    log_file.close()


if __name__ == "__main__":
    main()