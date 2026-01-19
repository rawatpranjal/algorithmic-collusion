#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
import os
import sys
import pickle
from datetime import datetime

# Create necessary directories if they don't exist
os.makedirs("figures", exist_ok=True)
os.makedirs("logs", exist_ok=True)  # Add logs directory

# Set up logging with path to logs directory
log_filename = "logs/plot_log.txt"
log_file = open(log_filename, "w")

# Try to load hyperparameters from data module
try:
    with open("data/hyperparams.pkl", "rb") as f:
        HYPERPARAMS = pickle.load(f)
        BID_ACTIONS = HYPERPARAMS["BID_ACTIONS"]
except:
    # Fallback if file doesn't exist
    BID_ACTIONS = np.linspace(0.0, 1.0, 3).tolist()

def log_print(message):
    """Print to console and log file simultaneously"""
    print(message)
    log_file.write(message + "\n")
    log_file.flush()

# Set matplotlib parameters for consistent visualizations
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

def load_experiment_data(state_mode):
    """Load experiment data for a specific information regime"""
    log_print(f"Loading data for {state_mode} information regime...")
    
    # Load DataFrames
    df_act = pd.read_csv(f"data/actions_{state_mode}.csv")
    df_be = pd.read_csv(f"data/bellman_errors_{state_mode}.csv")
    df_rev = pd.read_csv(f"data/revenue_{state_mode}.csv")
    
    # Load NPZ data
    npz_data = np.load(f"data/{state_mode}_data.npz", allow_pickle=True)
    
    # Extract data from NPZ
    fpa_qtable0 = npz_data['fpa_qtable0'].item()
    fpa_qtable1 = npz_data['fpa_qtable1'].item()
    spa_qtable0 = npz_data['spa_qtable0'].item()
    spa_qtable1 = npz_data['spa_qtable1'].item()
    
    # Convert lists to proper data structures
    fpa_revenue = list(npz_data['fpa_revenue'])
    spa_revenue = list(npz_data['spa_revenue'])
    
    # Convert to appropriate format for plotting functions
    fpa_rev = [{"t": i, "auction_type": "FPA", "price": p} for i, p in enumerate(fpa_revenue)]
    spa_rev = [{"t": i, "auction_type": "SPA", "price": p} for i, p in enumerate(spa_revenue)]
    
    # Get analysis results if available
    fpa_analysis = npz_data['fpa_analysis'].item() if 'fpa_analysis' in npz_data else None
    spa_analysis = npz_data['spa_analysis'].item() if 'spa_analysis' in npz_data else None
    
    log_print(f"Successfully loaded data for {state_mode}")
    
    return {
        'df_act': df_act,
        'df_be': df_be,
        'df_rev': df_rev,
        'fpa_qtable0': fpa_qtable0,
        'fpa_qtable1': fpa_qtable1,
        'spa_qtable0': spa_qtable0,
        'spa_qtable1': spa_qtable1,
        'fpa_rev': fpa_rev,
        'spa_rev': spa_rev,
        'fpa_analysis': fpa_analysis,
        'spa_analysis': spa_analysis
    }

def extract_bid_actions(qtables):
    """Extract bid actions from Q-tables"""
    all_actions = set()
    for qtable in qtables:
        for state in qtable:
            for action in qtable[state]:
                all_actions.add(action)
    
    return sorted(list(all_actions))

def plot_combined_results(df_act, df_be, df_rev, state_mode, rolling_window):
    """Plot combined training dynamics"""
    info_regime_name = {
        "none": "No Disclosure",
        "winning": "Winning Bid Only",
        "full": "Full Disclosure"
    }[state_mode]
    
    log_print(f"Generating combined results plots for {info_regime_name}...")
    
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
    
    # Plot 1: Bellman error and revenue
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
    
    filename1 = f"figures/bellman_revenue_{state_mode}.png"
    fig1.savefig(filename1, dpi=300, bbox_inches='tight')
    log_print(f"Saved figure: {filename1}")
    
    # Plot 2: Bidder rewards
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
    
    filename2 = f"figures/rewards_{state_mode}.png"
    fig2.savefig(filename2, dpi=300, bbox_inches='tight')
    log_print(f"Saved figure: {filename2}")
    
    # Plot 3: Bidding behavior
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
    
    filename3 = f"figures/bids_{state_mode}.png"
    fig3.savefig(filename3, dpi=300, bbox_inches='tight')
    log_print(f"Saved figure: {filename3}")
    
    plt.close('all')

def create_distribution_table(fpa_rev, spa_rev, df_act, state_mode, window_size=10000, bid_actions=None):
    """Create distribution table of bids and prices"""
    info_regime_name = {
        "none": "No Disclosure",
        "winning": "Winning Bid Only",
        "full": "Full Disclosure"
    }[state_mode]
    
    log_print(f"Creating bid distribution table for {info_regime_name}...")
    
    # Get most recent data
    fpa_prices = [entry["price"] for entry in fpa_rev[-window_size:]]
    spa_prices = [entry["price"] for entry in spa_rev[-window_size:]]
    
    # Filter DataFrame to get most recent actions
    df_fpa = df_act[df_act['auction_type'] == 'FPA'].tail(2*window_size)
    df_spa = df_act[df_act['auction_type'] == 'SPA'].tail(2*window_size)
    
    fpa0 = df_fpa[df_fpa.bidder == 0]['chosen_bid'].values[-window_size//2:]
    fpa1 = df_fpa[df_fpa.bidder == 1]['chosen_bid'].values[-window_size//2:]
    spa0 = df_spa[df_spa.bidder == 0]['chosen_bid'].values[-window_size//2:]
    spa1 = df_spa[df_spa.bidder == 1]['chosen_bid'].values[-window_size//2:]
    
    def count_bid(bid_list, target_bid):
        return len([b for b in bid_list if abs(b - target_bid) < 0.001])
    
    rows = []
    headers = ["Bid", "FPA Price (%)", "SPA Price (%)", "FPA Bid0 (%)", "FPA Bid1 (%)", "SPA Bid0 (%)", "SPA Bid1 (%)"]
    
    # Use provided bid_actions or get from hyperparameters or extract from Q-tables
    if bid_actions is None:
        bid_actions = BID_ACTIONS
    
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
            f"{fpa_price_count} ({fpa_price_pct:.2f}%)", 
            f"{spa_price_count} ({spa_price_pct:.2f}%)",
            f"{fpa0_count} ({fpa0_pct:.2f}%)",
            f"{fpa1_count} ({fpa1_pct:.2f}%)",
            f"{spa0_count} ({spa0_pct:.2f}%)",
            f"{spa1_count} ({spa1_pct:.2f}%)"
        ])
    
    table_title = f"\n--- Distribution Table (Last {window_size} Rounds) - {info_regime_name} ---\n"
    table_str = table_title + tabulate(rows, headers=headers, tablefmt="pipe")
    
    log_print(table_str)
    
    return table_str

def plot_learned_strategies(qtable_fpa, qtable_spa, bidder, state_mode, bid_actions=None):
    """Plot learned bidding strategies"""
    info_regime_name = {
        "none": "No Disclosure",
        "winning": "Winning Bid Only",
        "full": "Full Disclosure"
    }[state_mode]
    
    log_print(f"Plotting learned strategies for Bidder {bidder} in {info_regime_name}...")
    
    plt.style.use('grayscale')
    
    # Use provided bid_actions or get from hyperparameters
    if bid_actions is None:
        bid_actions = BID_ACTIONS
    
    if state_mode == "none":
        # For 'none' mode, just show the best actions for the single state
        # Since there's now only one state in the 'none' mode, we can create a simple bar chart
        # showing the Q-values for each action for each auction type
        
        fpa_state = ('SINGLE_STATE',)
        spa_state = ('SINGLE_STATE',)
        
        # Extract Q-values for FPA
        fpa_actions = []
        fpa_qvalues = []
        if fpa_state in qtable_fpa:
            for action, qvalue in sorted(qtable_fpa[fpa_state].items()):
                fpa_actions.append(action)
                fpa_qvalues.append(qvalue)
        
        # Extract Q-values for SPA
        spa_actions = []
        spa_qvalues = []
        if spa_state in qtable_spa:
            for action, qvalue in sorted(qtable_spa[spa_state].items()):
                spa_actions.append(action)
                spa_qvalues.append(qvalue)
        
        # Create table of Q-values
        rows = []
        headers = ["Action", "FPA Q-Value", "SPA Q-Value"]
        
        # Use all possible actions
        all_actions = sorted(set(fpa_actions + spa_actions))
        
        for action in all_actions:
            fpa_q = qtable_fpa.get(fpa_state, {}).get(action, "N/A")
            spa_q = qtable_spa.get(spa_state, {}).get(action, "N/A")
            
            fpa_q_str = f"{fpa_q:.4f}" if fpa_q != "N/A" else "N/A"
            spa_q_str = f"{spa_q:.4f}" if spa_q != "N/A" else "N/A"
            
            # Add marker for best action
            if fpa_q != "N/A" and fpa_q == max(qtable_fpa[fpa_state].values()):
                fpa_q_str += " *"
            if spa_q != "N/A" and spa_q == max(qtable_spa[spa_state].values()):
                spa_q_str += " *"
            
            rows.append([f"{action:.2f}", fpa_q_str, spa_q_str])
        
        table_title = f"\n--- Q-Values for Single State - Bidder {bidder} - {info_regime_name} ---\n"
        table_str = table_title + tabulate(rows, headers=headers, tablefmt="pipe")
        
        log_print(table_str)
        
        # Create bar chart of Q-values
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='white')
        plt.suptitle(f"Q-Values for Single State - Bidder {bidder} - {info_regime_name}", fontsize=18)
        
        bar_width = 0.6
        
        if fpa_actions:
            axes[0].bar([str(round(a, 2)) for a in fpa_actions], fpa_qvalues, bar_width, color='black')
            axes[0].set_title('First-Price Auction (FPA)')
            axes[0].set_xlabel('Action (Bid)')
            axes[0].set_ylabel('Q-Value')
            best_idx = np.argmax(fpa_qvalues)
            axes[0].text(best_idx, fpa_qvalues[best_idx], "Best", 
                         ha='center', va='bottom', weight='bold')
            axes[0].grid(True, linestyle='--', alpha=0.7)
        else:
            axes[0].text(0.5, 0.5, "No data available", 
                         ha='center', va='center', transform=axes[0].transAxes)
        
        if spa_actions:
            axes[1].bar([str(round(a, 2)) for a in spa_actions], spa_qvalues, bar_width, color='black')
            axes[1].set_title('Second-Price Auction (SPA)')
            axes[1].set_xlabel('Action (Bid)')
            axes[1].set_ylabel('Q-Value')
            best_idx = np.argmax(spa_qvalues)
            axes[1].text(best_idx, spa_qvalues[best_idx], "Best", 
                         ha='center', va='bottom', weight='bold')
            axes[1].grid(True, linestyle='--', alpha=0.7)
        else:
            axes[1].text(0.5, 0.5, "No data available", 
                         ha='center', va='center', transform=axes[1].transAxes)
        
        plt.tight_layout()
        
        figure_filename = f"figures/strategy_{state_mode}_bidder{bidder}.png"
        fig.savefig(figure_filename, dpi=300, bbox_inches='tight')
        log_print(f"Saved figure: {figure_filename}")
        plt.close('all')
        
        return table_str
    
    elif state_mode == "winning":
        # For 'winning' mode, create heatmaps
        def create_strategy_matrix(qtable):
            matrix = np.zeros((len(bid_actions), len(bid_actions)))
            matrix.fill(np.nan)
            
            for state, actions in qtable.items():
                if len(state) == 2 and state != ('START',):
                    my_bid_idx = np.abs(np.array(bid_actions) - state[0]).argmin()
                    winning_bid_idx = np.abs(np.array(bid_actions) - state[1]).argmin()
                    
                    if bid_actions[winning_bid_idx] >= bid_actions[my_bid_idx]:
                        best_action = max(actions.items(), key=lambda x: x[1])[0]
                        matrix[my_bid_idx, winning_bid_idx] = best_action
            
            return matrix
        
        def create_qvalue_matrix(qtable):
            matrix = np.zeros((len(bid_actions), len(bid_actions)))
            matrix.fill(np.nan)
            
            for state, actions in qtable.items():
                if len(state) == 2 and state != ('START',):
                    my_bid_idx = np.abs(np.array(bid_actions) - state[0]).argmin()
                    winning_bid_idx = np.abs(np.array(bid_actions) - state[1]).argmin()
                    
                    if bid_actions[winning_bid_idx] >= bid_actions[my_bid_idx]:
                        max_q = max(actions.values())
                        matrix[my_bid_idx, winning_bid_idx] = max_q
            
            return matrix
        
        fpa_strategy = create_strategy_matrix(qtable_fpa)
        spa_strategy = create_strategy_matrix(qtable_spa)
        fpa_qvalues = create_qvalue_matrix(qtable_fpa)
        spa_qvalues = create_qvalue_matrix(qtable_spa)
        
        masked_fpa_strat = np.ma.masked_invalid(fpa_strategy)
        masked_spa_strat = np.ma.masked_invalid(spa_strategy)
        masked_fpa_q = np.ma.masked_invalid(fpa_qvalues)
        masked_spa_q = np.ma.masked_invalid(spa_qvalues)
        
        fig1, axes1 = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')
        plt.suptitle(f"Best Bid Strategy - Bidder {bidder} - {info_regime_name}", fontsize=18)
        
        im00 = axes1[0].imshow(masked_fpa_strat, cmap='gray', origin='lower', vmin=0, vmax=1)
        axes1[0].set_title('FPA Best Bid')
        axes1[0].set_xlabel('Winning Bid')
        axes1[0].set_ylabel('My Previous Bid')
        axes1[0].set_xticks(range(len(bid_actions)))
        axes1[0].set_yticks(range(len(bid_actions)))
        axes1[0].set_xticklabels([f"{b:.2f}" for b in bid_actions])
        axes1[0].set_yticklabels([f"{b:.2f}" for b in bid_actions])
        
        for i in range(len(bid_actions)):
            for j in range(len(bid_actions)):
                if j < i:
                    axes1[0].add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color='lightgray', hatch='//'))
        
        fig1.colorbar(im00, ax=axes1[0], label='Best Bid')
        
        im01 = axes1[1].imshow(masked_spa_strat, cmap='gray', origin='lower', vmin=0, vmax=1)
        axes1[1].set_title('SPA Best Bid')
        axes1[1].set_xlabel('Winning Bid')
        axes1[1].set_ylabel('My Previous Bid')
        axes1[1].set_xticks(range(len(bid_actions)))
        axes1[1].set_yticks(range(len(bid_actions)))
        axes1[1].set_xticklabels([f"{b:.2f}" for b in bid_actions])
        axes1[1].set_yticklabels([f"{b:.2f}" for b in bid_actions])
        
        for i in range(len(bid_actions)):
            for j in range(len(bid_actions)):
                if j < i:
                    axes1[1].add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color='lightgray', hatch='\\\\'))
        
        fig1.colorbar(im01, ax=axes1[1], label='Best Bid')
        
        plt.tight_layout()
        
        filename1 = f"figures/strategy_best_bid_{state_mode}_bidder{bidder}.png"
        fig1.savefig(filename1, dpi=300, bbox_inches='tight')
        log_print(f"Saved figure: {filename1}")
        
        fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')
        plt.suptitle(f"Q-Values - Bidder {bidder} - {info_regime_name}", fontsize=18)
        
        im10 = axes2[0].imshow(masked_fpa_q, cmap='gray', origin='lower')
        axes2[0].set_title('FPA Max Q-Value')
        axes2[0].set_xlabel('Winning Bid')
        axes2[0].set_ylabel('My Previous Bid')
        axes2[0].set_xticks(range(len(bid_actions)))
        axes2[0].set_yticks(range(len(bid_actions)))
        axes2[0].set_xticklabels([f"{b:.2f}" for b in bid_actions])
        axes2[0].set_yticklabels([f"{b:.2f}" for b in bid_actions])
        
        for i in range(len(bid_actions)):
            for j in range(len(bid_actions)):
                if j < i:
                    axes2[0].add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color='lightgray', hatch='//'))
        
        fig2.colorbar(im10, ax=axes2[0], label='Q-Value')
        
        im11 = axes2[1].imshow(masked_spa_q, cmap='gray', origin='lower')
        axes2[1].set_title('SPA Max Q-Value')
        axes2[1].set_xlabel('Winning Bid')
        axes2[1].set_ylabel('My Previous Bid')
        axes2[1].set_xticks(range(len(bid_actions)))
        axes2[1].set_yticks(range(len(bid_actions)))
        axes2[1].set_xticklabels([f"{b:.2f}" for b in bid_actions])
        axes2[1].set_yticklabels([f"{b:.2f}" for b in bid_actions])
        
        for i in range(len(bid_actions)):
            for j in range(len(bid_actions)):
                if j < i:
                    axes2[1].add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color='lightgray', hatch='\\\\'))
        
        fig2.colorbar(im11, ax=axes2[1], label='Q-Value')
        
        plt.tight_layout()
        
        filename2 = f"figures/strategy_qvalues_{state_mode}_bidder{bidder}.png"
        fig2.savefig(filename2, dpi=300, bbox_inches='tight')
        log_print(f"Saved figure: {filename2}")
        
        plt.close('all')
    
    else:  # full information case
        def create_strategy_matrix(qtable):
            matrix = np.zeros((len(bid_actions), len(bid_actions)))
            for state, actions in qtable.items():
                if len(state) == 2 and state != ('START',):
                    my_bid_idx = np.abs(np.array(bid_actions) - state[0]).argmin()
                    opp_bid_idx = np.abs(np.array(bid_actions) - state[1]).argmin()
                    
                    best_action = max(actions.items(), key=lambda x: x[1])[0]
                    matrix[my_bid_idx, opp_bid_idx] = best_action
            
            return matrix
        
        def create_qvalue_matrix(qtable):
            matrix = np.zeros((len(bid_actions), len(bid_actions)))
            for state, actions in qtable.items():
                if len(state) == 2 and state != ('START',):
                    my_bid_idx = np.abs(np.array(bid_actions) - state[0]).argmin()
                    opp_bid_idx = np.abs(np.array(bid_actions) - state[1]).argmin()
                    
                    max_q = max(actions.values())
                    matrix[my_bid_idx, opp_bid_idx] = max_q
            
            return matrix
        
        fpa_strategy = create_strategy_matrix(qtable_fpa)
        spa_strategy = create_strategy_matrix(qtable_spa)
        fpa_qvalues = create_qvalue_matrix(qtable_fpa)
        spa_qvalues = create_qvalue_matrix(qtable_spa)
        
        fig1, axes1 = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')
        plt.suptitle(f"Best Bid Strategy - Bidder {bidder} - {info_regime_name}", fontsize=18)
        
        im00 = axes1[0].imshow(fpa_strategy, cmap='gray', origin='lower', vmin=0, vmax=1)
        axes1[0].set_title('FPA Best Bid')
        axes1[0].set_xlabel('Opponent Previous Bid')
        axes1[0].set_ylabel('My Previous Bid')
        axes1[0].set_xticks(range(len(bid_actions)))
        axes1[0].set_yticks(range(len(bid_actions)))
        axes1[0].set_xticklabels([f"{b:.2f}" for b in bid_actions])
        axes1[0].set_yticklabels([f"{b:.2f}" for b in bid_actions])
        fig1.colorbar(im00, ax=axes1[0], label='Best Bid')
        
        im01 = axes1[1].imshow(spa_strategy, cmap='gray', origin='lower', vmin=0, vmax=1)
        axes1[1].set_title('SPA Best Bid')
        axes1[1].set_xlabel('Opponent Previous Bid')
        axes1[1].set_ylabel('My Previous Bid')
        axes1[1].set_xticks(range(len(bid_actions)))
        axes1[1].set_yticks(range(len(bid_actions)))
        axes1[1].set_xticklabels([f"{b:.2f}" for b in bid_actions])
        axes1[1].set_yticklabels([f"{b:.2f}" for b in bid_actions])
        fig1.colorbar(im01, ax=axes1[1], label='Best Bid')
        
        plt.tight_layout()
        
        filename1 = f"figures/strategy_best_bid_{state_mode}_bidder{bidder}.png"
        fig1.savefig(filename1, dpi=300, bbox_inches='tight')
        log_print(f"Saved figure: {filename1}")
        
        fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')
        plt.suptitle(f"Q-Values - Bidder {bidder} - {info_regime_name}", fontsize=18)
        
        im10 = axes2[0].imshow(fpa_qvalues, cmap='gray', origin='lower')
        axes2[0].set_title('FPA Max Q-Value')
        axes2[0].set_xlabel('Opponent Previous Bid')
        axes2[0].set_ylabel('My Previous Bid')
        axes2[0].set_xticks(range(len(bid_actions)))
        axes2[0].set_yticks(range(len(bid_actions)))
        axes2[0].set_xticklabels([f"{b:.2f}" for b in bid_actions])
        axes2[0].set_yticklabels([f"{b:.2f}" for b in bid_actions])
        fig2.colorbar(im10, ax=axes2[0], label='Q-Value')
        
        im11 = axes2[1].imshow(spa_qvalues, cmap='gray', origin='lower')
        axes2[1].set_title('SPA Max Q-Value')
        axes2[1].set_xlabel('Opponent Previous Bid')
        axes2[1].set_ylabel('My Previous Bid')
        axes2[1].set_xticks(range(len(bid_actions)))
        axes2[1].set_yticks(range(len(bid_actions)))
        axes2[1].set_xticklabels([f"{b:.2f}" for b in bid_actions])
        axes2[1].set_yticklabels([f"{b:.2f}" for b in bid_actions])
        fig2.colorbar(im11, ax=axes2[1], label='Q-Value')
        
        plt.tight_layout()
        
        filename2 = f"figures/strategy_qvalues_{state_mode}_bidder{bidder}.png"
        fig2.savefig(filename2, dpi=300, bbox_inches='tight')
        log_print(f"Saved figure: {filename2}")
        
        plt.close('all')

def plot_bid_distribution(fpa_rev, spa_rev, df_act, state_mode, window_size=10000, bid_actions=None):
    """Generate bid distribution table and save as file"""
    table_str = create_distribution_table(fpa_rev, spa_rev, df_act, state_mode, window_size, bid_actions)
    return table_str

def tabulate_analysis_results(fpa_analysis, spa_analysis, state_mode):
    """Create table of Q-learning analysis results"""
    info_regime_name = {
        "none": "No Disclosure",
        "winning": "Winning Bid Only",
        "full": "Full Disclosure"
    }[state_mode]
    
    log_print(f"Creating analysis results table for {info_regime_name}...")
    
    table_data = [
        ["FPA", 
         round(fpa_analysis["mean_bellman_error_agent0"], 4), 
         round(fpa_analysis["mean_bellman_error_agent1"], 4),
         round(fpa_analysis["best_response_consistency_agent0"], 4),
         round(fpa_analysis["best_response_consistency_agent1"], 4),
         fpa_analysis["num_states_checked"]],
        ["SPA", 
         round(spa_analysis["mean_bellman_error_agent0"], 4), 
         round(spa_analysis["mean_bellman_error_agent1"], 4),
         round(spa_analysis["best_response_consistency_agent0"], 4),
         round(spa_analysis["best_response_consistency_agent1"], 4),
         spa_analysis["num_states_checked"]]
    ]
    headers = ["Auction Type", 
               "Mean Bellman Error (Agent 0)", 
               "Mean Bellman Error (Agent 1)",
               "Best Response Consistency (Agent 0)", 
               "Best Response Consistency (Agent 1)", 
               "States Checked"]
    
    table_str = f"\n--- Analysis Results for {info_regime_name} Information Regime ---\n"
    table_str += tabulate(table_data, headers=headers, tablefmt="pipe")
    
    log_print(table_str)
    
    return table_str

def tabulate_combined_strategies(fpa_qt0, fpa_qt1, spa_qt0, spa_qt1, state_mode):
    """Create table of learned strategies"""
    info_regime_name = {
        "none": "No Disclosure",
        "winning": "Winning Bid Only",
        "full": "Full Disclosure"
    }[state_mode]
    
    log_print(f"Creating strategy table for {info_regime_name}...")
    
    rows = []
    
    for st, acts in fpa_qt0.items():
        if st == ('START',):
            continue
        max_q = max(acts.values())
        for a, qv in sorted(acts.items()):
            star = "*" if abs(qv - max_q) < 1e-12 else ""
            rows.append(["FPA", 0, f"{st}", f"{a:.2f}", f"{qv:.4f}{star}"])
    
    for st, acts in fpa_qt1.items():
        if st == ('START',):
            continue
        max_q = max(acts.values())
        for a, qv in sorted(acts.items()):
            star = "*" if abs(qv - max_q) < 1e-12 else ""
            rows.append(["FPA", 1, f"{st}", f"{a:.2f}", f"{qv:.4f}{star}"])
    
    for st, acts in spa_qt0.items():
        if st == ('START',):
            continue
        max_q = max(acts.values())
        for a, qv in sorted(acts.items()):
            star = "*" if abs(qv - max_q) < 1e-12 else ""
            rows.append(["SPA", 0, f"{st}", f"{a:.2f}", f"{qv:.4f}{star}"])
    
    for st, acts in spa_qt1.items():
        if st == ('START',):
            continue
        max_q = max(acts.values())
        for a, qv in sorted(acts.items()):
            star = "*" if abs(qv - max_q) < 1e-12 else ""
            rows.append(["SPA", 1, f"{st}", f"{a:.2f}", f"{qv:.4f}{star}"])
    
    rows.sort(key=lambda x: (x[0], x[1], str(x[2])))
    
    headers = ["Auction Type", "Agent ID", "State", "Action", "Q-Value"]
    
    # Save sample to log
    table_str = f"\n--- Learned Strategies (Sample) for {info_regime_name} Information Regime ---\n"
    sample_rows = rows  # Show only first 20 rows in log
    table_str += tabulate(sample_rows, headers=headers, tablefmt="pipe")
    log_print(table_str)
    
    return rows

def analyze_all_regimes(rolling_window=10000):
    """Load and analyze data for all information regimes"""
    info_regimes = ["none", "winning", "full"]
    
    # Load experiment settings
    try:
        with open("data/hyperparams.pkl", "rb") as f:
            hyperparams = pickle.load(f)
            bid_actions = hyperparams.get("BID_ACTIONS", np.linspace(0.0, 1.0, 3).tolist())
            
            log_print("Experiment Settings:")
            for key, value in hyperparams.items():
                if isinstance(value, float):
                    log_print(f"{key}: {value:.2f}")
                elif isinstance(value, list) and all(isinstance(x, float) for x in value):
                    log_print(f"{key}: {[round(b, 2) for b in value]}")
                else:
                    log_print(f"{key}: {value}")
    except:
        # Fallback if file doesn't exist
        log_print("Warning: Could not find hyperparameters file. Using default values.")
        bid_actions = np.linspace(0.0, 1.0, 3).tolist()
    
    # Process each information regime
    for regime in info_regimes:
        log_print(f"\n\n{'-'*50}\nAnalyzing {regime} information regime\n{'-'*50}")
        
        # Load data
        data = load_experiment_data(regime)
        
        # Plot results
        log_print(f"Generating plots for {regime}...")
        plot_combined_results(data['df_act'], data['df_be'], data['df_rev'], regime, rolling_window)
        
        # Generate distribution tables
        log_print(f"Generating bid distribution tables for {regime}...")
        plot_bid_distribution(data['fpa_rev'], data['spa_rev'], data['df_act'], regime, window_size=rolling_window, bid_actions=bid_actions)
        
        # Generate strategy plots
        log_print(f"Generating strategy plots for {regime}...")
        for bidder in [0, 1]:
            plot_learned_strategies(data['fpa_qtable0'] if bidder == 0 else data['fpa_qtable1'], 
                               data['spa_qtable0'] if bidder == 0 else data['spa_qtable1'], 
                               bidder, regime, bid_actions)
        
        # Generate analysis tables
        log_print(f"Generating analysis tables for {regime}...")
        tabulate_analysis_results(data['fpa_analysis'], data['spa_analysis'], regime)
        tabulate_combined_strategies(data['fpa_qtable0'], data['fpa_qtable1'], data['spa_qtable0'], data['spa_qtable1'], regime)

def main():
    """Main function to analyze all regimes"""
    log_print("Starting analysis and plotting...")
    log_print("Results will be saved to the figures/ folder and this log file.")
    analyze_all_regimes(rolling_window=10000)
    log_print("Analysis complete. All figures have been saved to figures/ folder.")
    log_file.close()

if __name__ == "__main__":
    main()