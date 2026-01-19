#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import os
import pickle

# Create necessary directory for figures and logs
os.makedirs("figures", exist_ok=True)
os.makedirs("logs", exist_ok=True)  # Add logs directory

# Set up logging to logs directory
log_filename = "logs/differential_impulse_log.txt"
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

# Set matplotlib parameters for visualizations
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

# Helper functions from original code
def get_state(last_bids, bidder_idx, mode="none"):
    """Get state representation based on information disclosure mode"""
    if last_bids[0] is None or last_bids[1] is None:
        return ("START",)
    
    # For "none" mode, use a single state constant rather than previous bid
    if mode == "none":
        return ("SINGLE_STATE",)
    
    my_bid = last_bids[bidder_idx]
    opp_bid = last_bids[1 - bidder_idx]
    winning_bid = max(last_bids)
    
    if mode == "winning":
        return (round(my_bid, 2), round(winning_bid, 2))
    else:  # full disclosure
        return (round(my_bid, 2), round(opp_bid, 2))

def coin_flip_winner(indices):
    """Randomly choose a winner from tied bidders"""
    return np.random.choice(indices)

def load_experiment_data(state_mode):
    """Load experiment data for a specific information regime"""
    log_print(f"Loading Q-tables for {state_mode} information regime...")
    
    # Load data from NPZ file
    npz_data = np.load(f"data/{state_mode}_data.npz", allow_pickle=True)
    
    # Extract Q-tables
    fpa_qtable0 = npz_data['fpa_qtable0'].item()
    fpa_qtable1 = npz_data['fpa_qtable1'].item()
    spa_qtable0 = npz_data['spa_qtable0'].item()
    spa_qtable1 = npz_data['spa_qtable1'].item()
    
    return {
        'fpa_qtable0': fpa_qtable0,
        'fpa_qtable1': fpa_qtable1,
        'spa_qtable0': spa_qtable0,
        'spa_qtable1': spa_qtable1
    }

def get_next_higher_bid(current_bid, bid_actions):
    """Get the next higher bid from the available bid actions"""
    higher_bids = [b for b in bid_actions if b > current_bid]
    if not higher_bids:
        return current_bid  # Already at max bid
    return min(higher_bids)  # Get the next higher bid (minimum of all higher bids)

def run_differential_impulse_analysis(fpa_qtable0, fpa_qtable1, spa_qtable0, spa_qtable1, bid_actions, state_mode):
    """
    Run differential impulse response analysis by comparing baseline (no shock) 
    with perturbed simulations (shock at period 10).
    
    Includes:
    - Single impulse (up/down)
    - Permanent defection (bidder raises bid by 1 unit and maintains it)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if state_mode == "none":
        log_print("Impulse response analysis not applicable for No Disclosure mode (single state).")
        return
    
    log_print(f"\n{'-'*50}\nRunning differential impulse analysis for {state_mode} information regime\n{'-'*50}")
    
    # Parameters
    total_rounds = 30    # Extend total simulation length for permanent defection
    impulse_round = 10   # When the shock happens
    
    # Storage for simulation results
    auction_types = ["FPA", "SPA"]
    impulse_types = ["up", "down", "permanent_up"]  # Add permanent_up for permanent defection
    
    results = {}
    
    for auction_type in auction_types:
        results[auction_type] = {}
        
        # Create baseline simulation entry
        results[auction_type]["baseline"] = {
            'bidder0_bids': [],
            'bidder1_bids': [],
            'prices': []
        }
        
        for impulse_type in impulse_types:
            results[auction_type][impulse_type] = {}
            for shock_bidder in [0, 1]:  # Run analysis with shock to each bidder
                results[auction_type][impulse_type][shock_bidder] = {
                    'bidder0_bids': [],
                    'bidder1_bids': [],
                    'prices': [],
                    'impulse_round': impulse_round,
                    'difference_bidder0': [],  # Will store difference from baseline
                    'difference_bidder1': []   # Will store difference from baseline
                }
    
    # Helper function to choose best action based on Q-values
    def best_action(qtable, state):
        if state not in qtable or not qtable[state]:
            return np.random.choice(bid_actions)
        return max(qtable[state].items(), key=lambda x: x[1])[0]
    
    # Run simulations
    for auction_type in auction_types:
        qtable0 = fpa_qtable0 if auction_type == "FPA" else spa_qtable0
        qtable1 = fpa_qtable1 if auction_type == "FPA" else spa_qtable1
        log_print(f"Running differential impulse simulations for {auction_type}...")
        
        # First run baseline simulation (no shock)
        baseline_data = results[auction_type]["baseline"]
        last_bids = [None, None]
        
        for t in range(total_rounds):
            # Get states
            s0 = get_state(last_bids, 0, state_mode)
            s1 = get_state(last_bids, 1, state_mode)
            
            # Choose actions based on Q-tables
            a0 = best_action(qtable0, s0)
            a1 = best_action(qtable1, s1)
            
            # Determine auction outcome
            bids = [a0, a1]
            max_bid = max(bids)
            winners = [i for i, b in enumerate(bids) if b == max_bid]
            
            if len(winners) > 1:
                winner = coin_flip_winner(winners)
            else:
                winner = winners[0]
            
            # Calculate price
            if auction_type == "FPA":
                price = max_bid
            else:  # SPA
                if len(winners) > 1:
                    price = max_bid
                else:
                    loser = 1 - winner
                    price = bids[loser]
            
            # Store data
            baseline_data['bidder0_bids'].append(a0)
            baseline_data['bidder1_bids'].append(a1)
            baseline_data['prices'].append(price)
            
            # Update bids for next iteration
            last_bids = [a0, a1]
        
        # Now run shock simulations
        for impulse_type in impulse_types:
            for shock_bidder in [0, 1]:
                # Initialize bidding history
                last_bids = [None, None]
                data = results[auction_type][impulse_type][shock_bidder]
                
                # For permanent defection, track the increased bid
                if impulse_type == "permanent_up":
                    # Start with None, will be set at impulse round
                    permanent_higher_bid = None
                
                # Run simulation
                for t in range(total_rounds):
                    # Get states
                    s0 = get_state(last_bids, 0, state_mode)
                    s1 = get_state(last_bids, 1, state_mode)
                    
                    # Normal actions first
                    normal_a0 = best_action(qtable0, s0)
                    normal_a1 = best_action(qtable1, s1)
                    
                    # Choose actions - apply impulse at impulse_round
                    if impulse_type == "permanent_up" and t >= impulse_round and shock_bidder == 0:
                        # Apply permanent defection for bidder 0
                        if t == impulse_round:
                            # For first shock round, find the next higher bid
                            permanent_higher_bid = get_next_higher_bid(normal_a0, bid_actions)
                        a0 = permanent_higher_bid  # Use the permanent higher bid
                        a1 = normal_a1
                        
                    elif impulse_type == "permanent_up" and t >= impulse_round and shock_bidder == 1:
                        # Apply permanent defection for bidder 1 
                        a0 = normal_a0
                        if t == impulse_round:
                            # For first shock round, find the next higher bid
                            permanent_higher_bid = get_next_higher_bid(normal_a1, bid_actions)
                        a1 = permanent_higher_bid  # Use the permanent higher bid
                        
                    elif t == impulse_round and shock_bidder == 0 and impulse_type == "up":
                        # Apply one-time shock to bidder 0 - force highest bid
                        a0 = max(bid_actions)
                        a1 = normal_a1
                        
                    elif t == impulse_round and shock_bidder == 0 and impulse_type == "down":
                        # Apply one-time shock to bidder 0 - force lowest bid
                        a0 = min(bid_actions)
                        a1 = normal_a1
                        
                    elif t == impulse_round and shock_bidder == 1 and impulse_type == "up":
                        # Apply one-time shock to bidder 1 - force highest bid
                        a0 = normal_a0
                        a1 = max(bid_actions)
                        
                    elif t == impulse_round and shock_bidder == 1 and impulse_type == "down":
                        # Apply one-time shock to bidder 1 - force lowest bid
                        a0 = normal_a0 
                        a1 = min(bid_actions)
                        
                    else:
                        # Normal bidding when not applying shock
                        a0 = normal_a0
                        a1 = normal_a1
                    
                    # Determine auction outcome
                    bids = [a0, a1]
                    max_bid = max(bids)
                    winners = [i for i, b in enumerate(bids) if b == max_bid]
                    
                    if len(winners) > 1:
                        winner = coin_flip_winner(winners)
                    else:
                        winner = winners[0]
                    
                    # Calculate price
                    if auction_type == "FPA":
                        price = max_bid
                    else:  # SPA
                        if len(winners) > 1:
                            price = max_bid
                        else:
                            loser = 1 - winner
                            price = bids[loser]
                    
                    # Store data
                    data['bidder0_bids'].append(a0)
                    data['bidder1_bids'].append(a1)
                    data['prices'].append(price)
                    
                    # Update bids for next iteration
                    last_bids = [a0, a1]
                
                # Calculate differences from baseline
                for t in range(total_rounds):
                    data['difference_bidder0'].append(data['bidder0_bids'][t] - baseline_data['bidder0_bids'][t])
                    data['difference_bidder1'].append(data['bidder1_bids'][t] - baseline_data['bidder1_bids'][t])
    
    # Plot the results
    for auction_type in auction_types:
        for impulse_type in impulse_types:
            # Create plot with 1 row for each shocked bidder
            fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
            
            # Create title based on impulse type
            if impulse_type == "permanent_up":
                plot_title = f"Permanent Defection Analysis - {auction_type} - {state_mode.capitalize()}"
            else:
                plot_title = f"Differential Impulse Analysis - {auction_type} - {impulse_type.capitalize()} Shock - {state_mode.capitalize()}"
                
            plt.suptitle(plot_title, fontsize=16)
            
            for row, shock_bidder in enumerate([0, 1]):
                baseline_data = results[auction_type]["baseline"]
                shock_data = results[auction_type][impulse_type][shock_bidder]
                t_values = np.arange(total_rounds)
                
                # Plot 1: Raw bids with baseline comparison
                axes[row, 0].plot(t_values, baseline_data['bidder0_bids'], 'b--', alpha=0.5, label='Bidder 0 (Baseline)')
                axes[row, 0].plot(t_values, baseline_data['bidder1_bids'], 'r--', alpha=0.5, label='Bidder 1 (Baseline)')
                axes[row, 0].plot(t_values, shock_data['bidder0_bids'], 'b-', label='Bidder 0 (Shock)')
                axes[row, 0].plot(t_values, shock_data['bidder1_bids'], 'r-', label='Bidder 1 (Shock)')
                
                axes[row, 0].axvline(x=impulse_round, color='k', linestyle='--')
                
                # Adjust title based on impulse type
                if impulse_type == "permanent_up":
                    axes[row, 0].set_title(f"Permanent Defection by Bidder {shock_bidder}")
                else:
                    axes[row, 0].set_title(f"Bids with {impulse_type} shock to Bidder {shock_bidder}")
                
                axes[row, 0].set_ylabel("Bid")
                axes[row, 0].set_ylim(-0.1, 1.1)
                axes[row, 0].grid(True, alpha=0.3)
                axes[row, 0].legend()
                
                # Add shock annotation
                shock_y = shock_data['bidder0_bids'][impulse_round] if shock_bidder == 0 else shock_data['bidder1_bids'][impulse_round]
                
                if impulse_type == "permanent_up":
                    annotation_text = "Permanent Defection Starts"
                else:
                    annotation_text = "Shock"
                
                axes[row, 0].annotate(annotation_text,
                            xy=(impulse_round, shock_y),
                            xytext=(impulse_round-3, shock_y + 0.2),
                            arrowprops=dict(facecolor='black', shrink=0.05))
                
                # Plot 2: Difference from baseline
                axes[row, 1].plot(t_values, shock_data['difference_bidder0'], 'b-', label='Bidder 0 Difference')
                axes[row, 1].plot(t_values, shock_data['difference_bidder1'], 'r-', label='Bidder 1 Difference')
                axes[row, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
                axes[row, 1].axvline(x=impulse_round, color='k', linestyle='--')
                
                if impulse_type == "permanent_up":
                    axes[row, 1].set_title(f"Bid Differences from Baseline (Defection by Bidder {shock_bidder})")
                else:
                    axes[row, 1].set_title(f"Bid Differences from Baseline (Shock to Bidder {shock_bidder})")
                
                axes[row, 1].set_ylabel("Bid Difference")
                axes[row, 1].grid(True, alpha=0.3)
                axes[row, 1].legend()
                
                # Compute a centered moving average to show trends
                window = 3  # Smaller window for shorter simulation
                if total_rounds >= window:
                    # Compute centered moving average for difference data
                    diff0_padded = np.pad(shock_data['difference_bidder0'], (window//2, window//2), mode='edge')
                    diff1_padded = np.pad(shock_data['difference_bidder1'], (window//2, window//2), mode='edge')
                    
                    diff0_ma = np.convolve(diff0_padded, np.ones(window)/window, mode='valid')
                    diff1_ma = np.convolve(diff1_padded, np.ones(window)/window, mode='valid')
                    
                    axes[row, 1].plot(t_values, diff0_ma, 'b--', alpha=0.7, linewidth=1)
                    axes[row, 1].plot(t_values, diff1_ma, 'r--', alpha=0.7, linewidth=1)
            
            # Set common x-axis label
            for ax in axes[-1]:
                ax.set_xlabel("Round")
                ax.set_xticks(np.arange(0, total_rounds, 5))
            
            # Adjust y-limits for difference plots to be symmetric around zero
            for row in range(2):
                max_diff = max(
                    np.max(np.abs(results[auction_type][impulse_type][0]['difference_bidder0'])),
                    np.max(np.abs(results[auction_type][impulse_type][0]['difference_bidder1'])),
                    np.max(np.abs(results[auction_type][impulse_type][1]['difference_bidder0'])),
                    np.max(np.abs(results[auction_type][impulse_type][1]['difference_bidder1']))
                )
                # Add a bit of margin
                max_diff = max_diff * 1.2
                axes[row, 1].set_ylim(-max_diff, max_diff)
            
            # Add text interpretation
            if impulse_type == "permanent_up":
                interpretation_text = (
                    f"Interpretation: This plot shows the response when Bidder {shock_bidder} permanently increases their bid at round {impulse_round}.\n" +
                    "The right panel shows bid differences from baseline, which reveal strategic responses to permanent defection."
                )
            else:
                interpretation_text = (
                    f"Interpretation: This plot shows the deviation from baseline when Bidder {shock_bidder} makes a {impulse_type} impulse at round {impulse_round}.\n" +
                    "The right panel shows bid differences from baseline, which reveal strategic responses to deviations."
                )
            
            plt.figtext(0.5, 0.01, 
                        interpretation_text,
                        ha="center", fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save figure
            if impulse_type == "permanent_up":
                filename = f"figures/permanent_defection_{auction_type}_{state_mode}.png"
            else:
                filename = f"figures/diff_impulse_{auction_type}_{impulse_type}_{state_mode}.png" 
            
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            log_print(f"Saved figure: {filename}")
            
            plt.close()
    
    log_print("Differential impulse and permanent defection analysis completed.")
    return results

def analyze_differential_responses(response_results, state_mode):
    """
    Analyze the differential impulse results to quantify strategic response patterns
    based on differences from baseline behavior.
    """
    if not response_results or state_mode == "none":
        return
    
    log_print(f"\nAnalyzing differential responses for {state_mode} information regime...")
    
    response_metrics = {}
    
    for auction_type, impulse_data in response_results.items():
        response_metrics[auction_type] = {}
        
        for impulse_type, shock_data in impulse_data.items():
            if impulse_type == "baseline":
                continue
            
            response_metrics[auction_type][impulse_type] = {}
            
            for shock_bidder, data in shock_data.items():
                # Get the responding bidder index
                responding_bidder = 1 - shock_bidder
                
                # Calculate pre-shock baseline difference (should be near zero)
                pre_shock = np.mean(data[f'difference_bidder{responding_bidder}'][:data['impulse_round']])
                
                # Calculate immediate response difference (rounds impulse_round to impulse_round+3)
                post_immediate = data['impulse_round'] + 3
                immediate_response = np.mean(data[f'difference_bidder{responding_bidder}'][data['impulse_round']:post_immediate])
                
                # Calculate later response difference (rounds impulse_round+3 to end)
                later_response = np.mean(data[f'difference_bidder{responding_bidder}'][post_immediate:])
                
                # Calculate long-term response for permanent defection (last 5 rounds)
                long_term_response = np.mean(data[f'difference_bidder{responding_bidder}'][-5:])
                
                # Determine if there was a punishment response
                # (non-shocked bidder increases bid in response to higher bid, or decreases in response to lower bid)
                punishment_response = False
                punishment_threshold = 0.1
                
                if impulse_type in ["up", "permanent_up"] and immediate_response > punishment_threshold:
                    punishment_response = True
                elif impulse_type == "down" and immediate_response < -punishment_threshold:
                    punishment_response = True
                
                # Determine if there was a return to cooperation
                return_to_cooperation = False
                cooperation_threshold = 0.05
                
                # For permanent defection, check if there's eventual reciprocity (also raising bid)
                if impulse_type == "permanent_up":
                    reciprocity = long_term_response > 0.1  # Long-term bid increase
                    return_to_cooperation = False  # There should be no return to cooperation
                else:
                    reciprocity = False
                    if abs(later_response - pre_shock) < cooperation_threshold:
                        return_to_cooperation = True
                
                response_metrics[auction_type][impulse_type][shock_bidder] = {
                    "pre_shock_difference": pre_shock,
                    "immediate_response_difference": immediate_response,
                    "later_response_difference": later_response,
                    "long_term_response": long_term_response,
                    "punishment_response": punishment_response,
                    "return_to_cooperation": return_to_cooperation,
                    "reciprocity": reciprocity if impulse_type == "permanent_up" else None
                }
    
    # Create table of results for regular impulses
    rows = []
    headers = ["Auction", "Impulse", "Shock To", "Response From", "Immediate Diff", "Later Diff", 
               "Punishment?", "Return to Cooperation?"]
    
    log_print("\n=== SINGLE IMPULSE ANALYSIS ===")
    
    for auction_type, impulse_data in response_metrics.items():
        for impulse_type, bidder_data in impulse_data.items():
            if impulse_type in ["up", "down"]:  # Only include regular impulses here
                for shock_bidder, metrics in bidder_data.items():
                    rows.append([
                        auction_type,
                        impulse_type.capitalize(),
                        f"Bidder {shock_bidder}",
                        f"Bidder {1-shock_bidder}",
                        f"{metrics['immediate_response_difference']:.2f}",
                        f"{metrics['later_response_difference']:.2f}",
                        "Yes" if metrics["punishment_response"] else "No",
                        "Yes" if metrics["return_to_cooperation"] else "No"
                    ])
    
    table_str = f"\n--- Differential Impulse Analysis for {state_mode.capitalize()} Information Regime ---\n"
    table_str += tabulate(rows, headers=headers, tablefmt="pipe")
    log_print(table_str)
    
    # Create separate table for permanent defection
    perm_rows = []
    perm_headers = ["Auction", "Defector", "Responder", "Immediate Response", "Long-term Response", 
                   "Initial Punishment?", "Eventual Reciprocity?"]
    
    log_print("\n=== PERMANENT DEFECTION ANALYSIS ===")
    
    for auction_type, impulse_data in response_metrics.items():
        if "permanent_up" in impulse_data:
            for shock_bidder, metrics in impulse_data["permanent_up"].items():
                perm_rows.append([
                    auction_type,
                    f"Bidder {shock_bidder}",
                    f"Bidder {1-shock_bidder}",
                    f"{metrics['immediate_response_difference']:.2f}",
                    f"{metrics['long_term_response']:.2f}",
                    "Yes" if metrics["punishment_response"] else "No",
                    "Yes" if metrics["reciprocity"] else "No"
                ])
    
    perm_table_str = f"\n--- Permanent Defection Analysis for {state_mode.capitalize()} Information Regime ---\n"
    perm_table_str += tabulate(perm_rows, perm_headers, tablefmt="pipe")
    log_print(perm_table_str)
    
    # Summarize collusion evidence from regular impulses
    punishment_count = sum(1 for row in rows if row[6] == "Yes")
    return_count = sum(1 for row in rows if row[7] == "Yes")
    total_scenarios = len(rows)
    
    # Summarize permanent defection responses
    perm_punishment_count = sum(1 for row in perm_rows if row[5] == "Yes")
    perm_reciprocity_count = sum(1 for row in perm_rows if row[6] == "Yes")
    perm_total_scenarios = len(perm_rows)
    
    log_print("\n=== SUMMARY OF FINDINGS ===")
    
    if total_scenarios > 0:
        collusion_evidence = (punishment_count / total_scenarios) * 0.5 + (return_count / total_scenarios) * 0.5
        
        interpretation = "No evidence of collusion"
        if collusion_evidence > 0.8:
            interpretation = "Strong evidence of collusion"
        elif collusion_evidence > 0.6:
            interpretation = "Moderate evidence of collusion"
        elif collusion_evidence > 0.4:
            interpretation = "Some evidence of collusion"
        elif collusion_evidence > 0.2:
            interpretation = "Weak evidence of collusion"
        
        summary = f"\nSingle Impulse Collusion Evidence Score: {collusion_evidence:.2f} - {interpretation}\n"
        summary += f"Punishment Responses: {punishment_count}/{total_scenarios} ({punishment_count/total_scenarios*100:.0f}%)\n"
        summary += f"Return to Cooperation: {return_count}/{total_scenarios} ({return_count/total_scenarios*100:.0f}%)\n"
        log_print(summary)
    else:
        log_print("No single impulse scenarios to analyze for collusion evidence.")
    
    if perm_total_scenarios > 0:
        perm_evidence = (perm_punishment_count / perm_total_scenarios) * 0.6 + (perm_reciprocity_count / perm_total_scenarios) * 0.4
        
        perm_interpretation = "No evidence of strategic response to defection"
        if perm_evidence > 0.8:
            perm_interpretation = "Strong evidence of strategic response to defection"
        elif perm_evidence > 0.6:
            perm_interpretation = "Moderate evidence of strategic response to defection"
        elif perm_evidence > 0.4:
            perm_interpretation = "Some evidence of strategic response to defection"
        elif perm_evidence > 0.2:
            perm_interpretation = "Weak evidence of strategic response to defection"
        
        perm_summary = f"\nPermanent Defection Response Score: {perm_evidence:.2f} - {perm_interpretation}\n"
        perm_summary += f"Initial Punishment Responses: {perm_punishment_count}/{perm_total_scenarios} ({perm_punishment_count/perm_total_scenarios*100:.0f}%)\n"
        perm_summary += f"Eventual Reciprocity: {perm_reciprocity_count}/{perm_total_scenarios} ({perm_reciprocity_count/perm_total_scenarios*100:.0f}%)\n"
        log_print(perm_summary)
    else:
        log_print("No permanent defection scenarios to analyze.")
    
    return response_metrics

def main():
    """Main function to run differential impulse analysis for all regimes"""
    log_print("Starting differential impulse analysis with permanent defection...")
    
    # Information regimes to analyze
    info_regimes = ["winning", "full"]  # Skip "none" since it's not applicable
    
    # Store all differential impulse results
    all_impulse_results = {}
    
    # For each information regime
    for regime in info_regimes:
        try:
            # Load Q-tables
            data = load_experiment_data(regime)
            
            # Extract bid actions from Q-tables or use from hyperparameters
            bid_actions = BID_ACTIONS
            log_print(f"Using bid actions: {[round(b, 2) for b in bid_actions]}")
            
            # Run differential impulse analysis with permanent defection
            impulse_results = run_differential_impulse_analysis(
                data['fpa_qtable0'], 
                data['fpa_qtable1'], 
                data['spa_qtable0'], 
                data['spa_qtable1'],
                bid_actions,
                regime
            )
            
            # Analyze differential responses including permanent defection
            analyze_differential_responses(impulse_results, regime)
            
            # Store results
            all_impulse_results[regime] = impulse_results
            
        except Exception as e:
            log_print(f"Error processing {regime} regime: {str(e)}")
            import traceback
            log_print(traceback.format_exc())
    
    log_print("\nDifferential impulse analysis with permanent defection complete. Results saved to logs/differential_impulse_log.txt and figures/")
    log_file.close()

if __name__ == "__main__":
    main()