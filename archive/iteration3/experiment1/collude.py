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
log_filename = "logs/collusion_log.txt"
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

def extract_bid_actions(qtables):
    """Extract bid actions from Q-tables"""
    all_actions = set()
    for qtable in qtables:
        for state in qtable:
            for action in qtable[state]:
                all_actions.add(action)
    
    return sorted(list(all_actions))

def simulate_visitation_frequency(qtable0, qtable1, bid_actions, auction_type, state_mode, num_steps=10000):
    """
    Simulate auction using deterministic Q-tables and record state visitation frequencies.
    """
    from collections import Counter
    
    # Skip for "none" mode since there's only one state
    if state_mode == "none":
        # In "none" mode, there's only one state
        return {("SINGLE_STATE",): 1.0}
    
    log_print(f"Computing state visitation frequencies for {auction_type} with {state_mode} information regime...")
    
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
            winner = coin_flip_winner(winners)
        else:
            winner = winners[0]
            
        # Update bids for next iteration
        last_bids = [a0, a1]
    
    # Normalize to get frequencies
    total_visits = sum(visitation.values())
    frequencies = {state: count / total_visits for state, count in visitation.items()}
    
    log_print(f"Recorded visitation frequencies for {len(frequencies)} states")
    
    return frequencies

def perturb_bid(bid, direction, bid_actions):
    """
    Find the next higher or lower bid in the bid_actions list.
    Handles floating-point precision issues.
    """
    # Find the closest index using numpy
    idx = np.abs(np.array(bid_actions) - bid).argmin()
    
    if direction == "up" and idx < len(bid_actions) - 1:
        return bid_actions[idx + 1]
    elif direction == "down" and idx > 0:
        return bid_actions[idx - 1]
    return bid  # No change if at boundary

def test_perturbations(qtable0, qtable1, visitation, bid_actions, state_mode):
    """
    Test perturbation responses for all visited states.
    """
    # Skip for "none" mode since collusion analysis is not applicable
    if state_mode == "none":
        log_print("Skipping perturbation analysis for No Disclosure mode (not applicable).")
        return {}
    
    log_print(f"Testing perturbation responses for {state_mode} information regime...")
    
    results = {}
    
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
        if state_mode == "winning":
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
    
    log_print(f"Completed perturbation testing for {len(results)} states")
    
    return results

def calculate_collusion_index(perturbation_results, visitation):
    """
    Calculate collusion index by weighting perturbation results by visitation frequency.
    """
    # If no perturbation results (e.g., for "none" mode), return default values
    if not perturbation_results:
        return {
            "agent0": {
                "retaliation": 0.0,
                "accommodation": 0.0,
                "collusion_index": 0.0
            },
            "agent1": {
                "retaliation": 0.0,
                "accommodation": 0.0,
                "collusion_index": 0.0
            },
            "overall_collusion_index": 0.0,
            "interpretation": "Not applicable for No Disclosure mode"
        }
    
    log_print("Calculating collusion indices...")
    
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

def run_perturbation_analysis(fpa_qtable0, fpa_qtable1, spa_qtable0, spa_qtable1, bid_actions, state_mode):
    """
    Perform perturbation checks to analyze strategic responses.
    """
    log_print(f"\n{'-'*50}\nRunning perturbation analysis for {state_mode} information regime\n{'-'*50}")
    
    # Check if analysis is applicable for this information regime
    if state_mode == "none":
        log_print("Collusion analysis is not applicable for No Disclosure mode (single state).")
        
        # Return placeholder results
        return {
            "FPA": {
                "visitation_frequency": {("SINGLE_STATE",): 1.0},
                "perturbation_results": {},
                "collusion_index": {
                    "agent0": {"retaliation": 0.0, "accommodation": 0.0, "collusion_index": 0.0},
                    "agent1": {"retaliation": 0.0, "accommodation": 0.0, "collusion_index": 0.0},
                    "overall_collusion_index": 0.0,
                    "interpretation": "Not applicable for No Disclosure mode"
                }
            },
            "SPA": {
                "visitation_frequency": {("SINGLE_STATE",): 1.0},
                "perturbation_results": {},
                "collusion_index": {
                    "agent0": {"retaliation": 0.0, "accommodation": 0.0, "collusion_index": 0.0},
                    "agent1": {"retaliation": 0.0, "accommodation": 0.0, "collusion_index": 0.0},
                    "overall_collusion_index": 0.0,
                    "interpretation": "Not applicable for No Disclosure mode"
                }
            }
        }
    
    # Step 1: Run a simulation to determine state visitation frequencies
    fpa_visitation = simulate_visitation_frequency(fpa_qtable0, fpa_qtable1, bid_actions, auction_type="FPA", 
                                              state_mode=state_mode, num_steps=10000)
    spa_visitation = simulate_visitation_frequency(spa_qtable0, spa_qtable1, bid_actions, auction_type="SPA", 
                                              state_mode=state_mode, num_steps=10000)
    
    # Step 2: Perform perturbation tests on each visited state
    fpa_perturbation_results = test_perturbations(fpa_qtable0, fpa_qtable1, fpa_visitation, bid_actions, state_mode)
    spa_perturbation_results = test_perturbations(spa_qtable0, spa_qtable1, spa_visitation, bid_actions, state_mode)
    
    # Step 3 & 4: Calculate collusion indices
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

def plot_collusion_indices(results_by_regime):
    """
    Plot collusion indices for different information regimes.
    """
    log_print("Generating collusion index plot...")
    
    # Extract data for plotting - skip "none" regime
    regimes = [r for r in results_by_regime.keys() if r != "none"]
    fpa_indices = [results_by_regime[regime]["FPA"]["collusion_index"]["overall_collusion_index"] for regime in regimes]
    spa_indices = [results_by_regime[regime]["SPA"]["collusion_index"]["overall_collusion_index"] for regime in regimes]
    
    # If only "none" regime is available or no regimes, skip plotting
    if not regimes:
        log_print("No applicable information regimes for collusion analysis. Skipping plot.")
        return
    
    # Map regime codes to readable names
    regime_names = {
        "winning": "Winning Bid Only",
        "full": "Full Disclosure"
    }
    
    readable_regimes = [regime_names[regime] for regime in regimes]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(regimes))
    width = 0.35
    
    plt.bar(x - width/2, fpa_indices, width, label='First-Price Auction', color='darkgray')
    plt.bar(x + width/2, spa_indices, width, label='Second-Price Auction', color='lightgray')
    
    plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='Strong Collusion')
    plt.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='Moderate Collusion')
    plt.axhline(y=0.4, color='g', linestyle='--', alpha=0.7, label='Weak Collusion')
    plt.axhline(y=0.2, color='b', linestyle='--', alpha=0.7, label='Minimal Coordination')
    
    plt.xlabel('Information Regime')
    plt.ylabel('Collusion Index')
    plt.title('Collusion Index by Auction Type and Information Regime')
    plt.xticks(x, readable_regimes)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    filename = "figures/collusion_indices.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    log_print(f"Saved collusion index plot to {filename}")
    
    plt.close()

def plot_component_indices(results_by_regime):
    """
    Plot retaliation and accommodation components by information regime.
    """
    log_print("Generating component indices plot...")
    
    # Skip "none" regime
    regimes = [r for r in results_by_regime.keys() if r != "none"]
    
    # If only "none" regime is available or no regimes, skip plotting
    if not regimes:
        log_print("No applicable information regimes for collusion analysis. Skipping plot.")
        return
    
    # Set up figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Map regime codes to readable names
    regime_names = {
        "winning": "Winning Bid Only",
        "full": "Full Disclosure"
    }
    readable_regimes = [regime_names[regime] for regime in regimes]
    
    # Prepare data for FPA
    fpa_retaliation = [results_by_regime[regime]["FPA"]["collusion_index"]["agent0"]["retaliation"] + 
                       results_by_regime[regime]["FPA"]["collusion_index"]["agent1"]["retaliation"] / 2 
                       for regime in regimes]
    
    fpa_accommodation = [results_by_regime[regime]["FPA"]["collusion_index"]["agent0"]["accommodation"] + 
                         results_by_regime[regime]["FPA"]["collusion_index"]["agent1"]["accommodation"] / 2 
                         for regime in regimes]
    
    # Prepare data for SPA
    spa_retaliation = [results_by_regime[regime]["SPA"]["collusion_index"]["agent0"]["retaliation"] + 
                       results_by_regime[regime]["SPA"]["collusion_index"]["agent1"]["retaliation"] / 2 
                       for regime in regimes]
    
    spa_accommodation = [results_by_regime[regime]["SPA"]["collusion_index"]["agent0"]["accommodation"] + 
                         results_by_regime[regime]["SPA"]["collusion_index"]["agent1"]["accommodation"] / 2 
                         for regime in regimes]
    
    x = np.arange(len(regimes))
    width = 0.35
    
    # Plot FPA components
    axes[0].bar(x - width/2, fpa_retaliation, width, label='Retaliation', color='#ff9999')
    axes[0].bar(x + width/2, fpa_accommodation, width, label='Accommodation', color='#99ccff')
    axes[0].set_title('First-Price Auction Components')
    axes[0].set_xlabel('Information Regime')
    axes[0].set_ylabel('Component Strength')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(readable_regimes)
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot SPA components
    axes[1].bar(x - width/2, spa_retaliation, width, label='Retaliation', color='#ff9999')
    axes[1].bar(x + width/2, spa_accommodation, width, label='Accommodation', color='#99ccff')
    axes[1].set_title('Second-Price Auction Components')
    axes[1].set_xlabel('Information Regime')
    axes[1].set_ylabel('Component Strength')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(readable_regimes)
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    filename = "figures/component_indices.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    log_print(f"Saved component indices plot to {filename}")
    
    plt.close()

def print_collusion_table(results_by_regime):
    """
    Create and print a summary table of collusion analysis results.
    """
    log_print("\nCollusion Analysis Summary Table:")
    
    rows = []
    headers = ["Info Regime", "Auction Type", "Agent", "Retaliation", "Accommodation", "Collusion Index", "Interpretation"]
    
    for regime, results in results_by_regime.items():
        regime_name = {
            "none": "No Disclosure",
            "winning": "Winning Bid Only", 
            "full": "Full Disclosure"
        }[regime]
        
        # For "none" mode, add a single row with "N/A" values
        if regime == "none":
            rows.append([
                regime_name,
                "All",
                "All",
                "N/A",
                "N/A",
                "N/A",
                "Not applicable for No Disclosure mode"
            ])
            continue
        
        for auction_type in ["FPA", "SPA"]:
            auction_results = results[auction_type]["collusion_index"]
            
            # Add agent 0 row
            rows.append([
                regime_name,
                auction_type,
                "Agent 0",
                f"{auction_results['agent0']['retaliation']:.2f}",
                f"{auction_results['agent0']['accommodation']:.2f}",
                f"{auction_results['agent0']['collusion_index']:.2f}",
                ""
            ])
            
            # Add agent 1 row
            rows.append([
                regime_name,
                auction_type,
                "Agent 1",
                f"{auction_results['agent1']['retaliation']:.2f}",
                f"{auction_results['agent1']['accommodation']:.2f}",
                f"{auction_results['agent1']['collusion_index']:.2f}",
                ""
            ])
            
            # Add overall row
            rows.append([
                regime_name,
                auction_type,
                "Overall",
                "",
                "",
                f"{auction_results['overall_collusion_index']:.2f}",
                auction_results['interpretation']
            ])
    
    table_str = tabulate(rows, headers=headers, tablefmt="pipe")
    log_print(table_str)

def main():
    """Main function to run perturbation analysis for all regimes"""
    log_print("Starting perturbation analysis for collusion detection...")
    
    # Information regimes to analyze
    info_regimes = ["none", "winning", "full"]
    
    # Store results for all regimes
    all_results = {}
    
    # For each information regime
    for regime in info_regimes:
        try:
            # Load Q-tables
            data = load_experiment_data(regime)
            
            # Extract bid actions from Q-tables or use from hyperparameters
            bid_actions = BID_ACTIONS
            log_print(f"Using bid actions: {[round(b, 2) for b in bid_actions]}")
            
            # Run perturbation analysis
            results = run_perturbation_analysis(
                data['fpa_qtable0'], 
                data['fpa_qtable1'], 
                data['spa_qtable0'], 
                data['spa_qtable1'],
                bid_actions,
                regime
            )
            
            # Store results
            all_results[regime] = results
            
            # Print detailed results for this regime (skip for "none" mode)
            if regime != "none":
                log_print(f"\n--- Perturbation Analysis Results for {regime} ---")
                
                for auction_type in ["FPA", "SPA"]:
                    collusion_data = results[auction_type]["collusion_index"]
                    log_print(f"\n{auction_type} Collusion Results:")
                    log_print(f"Agent 0: Retaliation = {collusion_data['agent0']['retaliation']:.2f}, " +
                            f"Accommodation = {collusion_data['agent0']['accommodation']:.2f}, " +
                            f"Index = {collusion_data['agent0']['collusion_index']:.2f}")
                    log_print(f"Agent 1: Retaliation = {collusion_data['agent1']['retaliation']:.2f}, " +
                            f"Accommodation = {collusion_data['agent1']['accommodation']:.2f}, " +
                            f"Index = {collusion_data['agent1']['collusion_index']:.2f}")
                    log_print(f"Overall Collusion Index: {collusion_data['overall_collusion_index']:.2f}")
                    log_print(f"Interpretation: {collusion_data['interpretation']}")
            else:
                log_print(f"\n--- No Perturbation Analysis for {regime} (Single State) ---")
        except Exception as e:
            log_print(f"Error processing {regime} regime: {str(e)}")
            import traceback
            log_print(traceback.format_exc())
    
    # Only generate summary plots and tables if we have results
    if all_results:
        # Generate summary plots and tables
        plot_collusion_indices(all_results)
        plot_component_indices(all_results)
        print_collusion_table(all_results)
    
    log_print("\nPerturbation analysis complete. Results saved to logs/collusion_log.txt and figures/")
    log_file.close()

if __name__ == "__main__":
    main()