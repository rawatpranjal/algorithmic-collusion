#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import os
import time
from datetime import datetime
from scipy.stats import entropy
import multiprocessing as mp
from functools import partial

# Import from existing modules
from data import run_case_study, get_state, exponential_decay
from collude import (
    perturb_bid, 
    test_perturbations, 
    calculate_collusion_index
)

# Create necessary directories
os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)  # Add logs directory

# Set up logging to logs directory
log_filename = "logs/experiment_log.txt"
log_file = open(log_filename, "w")

def log_print(message):
    """Print to console and log file simultaneously"""
    print(message)
    log_file.write(message + "\n")
    log_file.flush()

# Set matplotlib parameters
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

def compute_visitation_frequency(actions_log, state_mode, bid_actions, num_rows=1000):
    """
    Compute state visitation frequencies from action logs
    """
    from collections import Counter
    
    visitation = Counter()
    
    # Get the last num_rows of actions (paired for each bidder)
    last_actions = actions_log[-num_rows*2:]
    
    # Group by timestep 't'
    df = pd.DataFrame(last_actions)
    # Sort by 't' to ensure proper ordering
    df = df.sort_values('t')
    by_t = df.groupby('t')
    
    for t, group in by_t:
        if len(group) != 2:
            continue  # Skip if we don't have exactly 2 entries
            
        # Get bids from both agents
        bids = group['chosen_bid'].values
        if len(bids) != 2:
            continue
            
        # Compute states for both agents
        bidder0_state = get_state(bids, 0, state_mode)
        bidder1_state = get_state(bids, 1, state_mode)
        
        visitation[bidder0_state] += 1
        visitation[bidder1_state] += 1
    
    # Normalize to get frequencies
    total_visits = sum(visitation.values())
    if total_visits == 0:
        return {}
        
    frequencies = {state: count / total_visits for state, count in visitation.items()}
    
    return frequencies

def calculate_winner_entropy(actions_log, num_rounds=1000):
    """
    Calculate entropy of winner distribution in the last N rounds.
    Higher entropy means more balanced winning between bidders.
    """
    df = pd.DataFrame(actions_log[-num_rounds*2:])
    
    # Get rewards for last N rounds, non-zero rewards indicate winning
    winners = []
    for t, group in df.groupby('t'):
        if len(group) != 2:
            continue
        
        # Winner is the one with non-zero reward
        winner_idx = group[group['reward'] > 0]['bidder'].values
        if len(winner_idx) == 1:
            winners.append(winner_idx[0])
    
    if not winners:
        return 0.0
    
    # Count occurrences of each bidder winning
    counts = np.bincount(winners)
    probabilities = counts / len(winners)
    
    # Calculate entropy (base 2 for bits)
    return entropy(probabilities, base=2)

def run_single_experiment(seed, auction_type, state_mode, alpha_start=0.1, alpha_end=0.0, 
                         gamma=0.9, epsilon_start=0.1, epsilon_end=0.0, decay_factor=5.0,
                         max_learning_rounds=1_000_000):
    """
    Run a single experiment with given parameters and return metrics
    """
    try:
        # Set fixed bid actions
        bid_actions = np.linspace(0.0, 1.0, 3).tolist()
        
        # Run the experiment with exponential decay
        q_evol, actions_log, q_tables, rev_log, bell_errors = run_case_study(
            auction_type, state_mode=state_mode, seed=seed, 
            alpha_start=alpha_start, alpha_end=alpha_end, 
            gamma=gamma, epsilon_start=epsilon_start, epsilon_end=epsilon_end, 
            bid_actions=bid_actions, max_learning_rounds=max_learning_rounds,
            decay_factor=decay_factor
        )
        
        # Calculate average revenue (prices) from the last 1000 rounds
        last_prices = [entry["price"] for entry in rev_log[-1000:]]
        avg_revenue = np.mean(last_prices)
        
        # Calculate winner entropy
        win_entropy = calculate_winner_entropy(actions_log, 1000)
        
        # Calculate state visitation frequencies
        visitation = compute_visitation_frequency(actions_log, state_mode, bid_actions, 1000)
        
        # Run perturbation tests
        perturbation_results = test_perturbations(
            q_tables[0], q_tables[1], visitation, bid_actions, state_mode
        )
        
        # Calculate collusion index
        collusion_data = calculate_collusion_index(perturbation_results, visitation)
        collusion_index = collusion_data["overall_collusion_index"]
        
        return {
            "seed": seed,
            "auction_type": auction_type,
            "info_regime": state_mode,
            "avg_revenue": avg_revenue,
            "winner_entropy": win_entropy,
            "collusion_index": collusion_index
        }
    except Exception as e:
        print(f"Error in experiment {auction_type}/{state_mode}/seed={seed}: {str(e)}")
        # Return default values for failed experiments
        return {
            "seed": seed,
            "auction_type": auction_type,
            "info_regime": state_mode,
            "avg_revenue": np.nan,
            "winner_entropy": np.nan,
            "collusion_index": np.nan
        }

def run_experiment_set(auction_types, info_regimes, num_runs=20, decay_factor=5.0):
    """
    Run experiments for all combinations of auction_types and info_regimes
    for num_runs times each
    """
    results = []
    total_jobs = len(auction_types) * len(info_regimes) * num_runs
    completed = 0
    
    log_print(f"Starting {total_jobs} experiments...")
    start_time = time.time()
    
    # Run in parallel using multiprocessing
    pool = mp.Pool(processes=mp.cpu_count())
    
    for auction_type in auction_types:
        for info_regime in info_regimes:
            # Create partial function with fixed auction_type and info_regime
            run_func = partial(
                run_single_experiment, 
                auction_type=auction_type, 
                state_mode=info_regime,
                decay_factor=decay_factor
            )
            
            # Map seeds to the run function
            seed_results = pool.map(run_func, range(num_runs))
            results.extend(seed_results)
            
            completed += num_runs
            elapsed = time.time() - start_time
            remaining = (elapsed / completed) * (total_jobs - completed)
            
            log_print(f"Completed {completed}/{total_jobs} runs. " +
                     f"Elapsed: {elapsed:.1f}s, Estimated remaining: {remaining:.1f}s")
    
    pool.close()
    pool.join()
    
    log_print(f"All experiments completed in {time.time() - start_time:.1f} seconds")
    
    # Convert to DataFrame for analysis
    df_results = pd.DataFrame(results)
    return df_results

def analyze_results(df_results):
    """
    Analyze and summarize results of multiple experiment runs
    """
    log_print("Analyzing results...")
    
    # Group by auction type and info regime
    grouped = df_results.groupby(['auction_type', 'info_regime'])
    
    # Calculate mean and std for each metric
    summary = grouped.agg({
        'avg_revenue': ['mean', 'std'],
        'winner_entropy': ['mean', 'std'],
        'collusion_index': ['mean', 'std']
    })
    
    # Format for display
    table_rows = []
    headers = ["Auction", "Info Regime", "Avg Revenue", "Winner Entropy", "Collusion Index"]
    
    # Iterate through the MultiIndex
    for (auction, regime), row in summary.iterrows():
        revenue_mean = row[('avg_revenue', 'mean')]
        revenue_std = row[('avg_revenue', 'std')]
        entropy_mean = row[('winner_entropy', 'mean')]
        entropy_std = row[('winner_entropy', 'std')]
        collusion_mean = row[('collusion_index', 'mean')]
        collusion_std = row[('collusion_index', 'std')]
        
        table_rows.append([
            auction,
            regime,
            f"{revenue_mean:.4f} ± {revenue_std:.4f}",
            f"{entropy_mean:.4f} ± {entropy_std:.4f}",
            f"{collusion_mean:.4f} ± {collusion_std:.4f}"
        ])
    
    # Print and save table
    table_str = tabulate(table_rows, headers=headers, tablefmt="pipe")
    log_print("\nSummary Results:\n" + table_str)
    
    with open("results/summary_stats.txt", "w") as f:
        f.write(table_str)
    
    # Save LaTeX version
    with open("results/summary_stats.tex", "w") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Summary Statistics (Mean $\\pm$ Std across seeds)}\n")
        f.write("\\begin{tabular}{llccc}\n")
        f.write("\\toprule\n")
        f.write("Auction & Info Regime & Avg Revenue & Winner Entropy & Collusion Index \\\\\n")
        f.write("\\midrule\n")
        
        current_auction = None
        for row in table_rows:
            auction, regime, revenue, entropy, collusion = row
            
            # Add midrule between auction types
            if current_auction is not None and current_auction != auction:
                f.write("\\midrule\n")
            current_auction = auction
            
            f.write(f"{auction} & {regime} & {revenue} & {entropy} & {collusion} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}")
    
    # Create plots
    create_summary_plots(df_results)
    
    return summary

def create_summary_plots(df_results):
    """
    Create summary plots from experiment results
    """
    log_print("Creating summary plots...")
    
    # Map regime codes to readable names
    regime_names = {
        "none": "No Disclosure",
        "winning": "Winning Bid Only", 
        "full": "Full Disclosure"
    }
    
    # Set plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Plot 1: Revenue by auction type and regime
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='info_regime', y='avg_revenue', hue='auction_type', data=df_results)
    plt.title('Revenue by Auction Type and Information Regime')
    plt.xlabel('Information Regime')
    plt.ylabel('Average Revenue')
    plt.xticks(ticks=[0, 1, 2], labels=[regime_names[r] for r in ['full', 'none', 'winning']])
    plt.legend(title='Auction Type')
    plt.tight_layout()
    plt.savefig('figures/revenue_comparison.png', dpi=300, bbox_inches='tight')
    
    # Plot 2: Winner entropy by auction type and regime
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='info_regime', y='winner_entropy', hue='auction_type', data=df_results)
    plt.title('Winner Entropy by Auction Type and Information Regime')
    plt.xlabel('Information Regime')
    plt.ylabel('Winner Entropy')
    plt.xticks(ticks=[0, 1, 2], labels=[regime_names[r] for r in ['full', 'none', 'winning']])
    plt.legend(title='Auction Type')
    plt.tight_layout()
    plt.savefig('figures/entropy_comparison.png', dpi=300, bbox_inches='tight')
    
    # Plot 3: Collusion index by auction type and regime
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='info_regime', y='collusion_index', hue='auction_type', data=df_results)
    plt.title('Collusion Index by Auction Type and Information Regime')
    plt.xlabel('Information Regime')
    plt.ylabel('Collusion Index')
    plt.xticks(ticks=[0, 1, 2], labels=[regime_names[r] for r in ['full', 'none', 'winning']])
    plt.legend(title='Auction Type')
    plt.tight_layout()
    plt.savefig('figures/collusion_comparison.png', dpi=300, bbox_inches='tight')
    
    # Close all plots
    plt.close('all')

def main():
    """
    Main function to run experiments
    """
    log_print("Starting experiment suite...")
    log_print(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    auction_types = ["FPA", "SPA"]
    info_regimes = ["none", "winning", "full"]
    num_runs = 100  # Run each configuration 100 times
    decay_factor = 5.0  # Set exponential decay factor
    
    # Create output directories
    os.makedirs("results", exist_ok=True)
    
    # Run experiments with exponential decay
    results_df = run_experiment_set(auction_types, info_regimes, num_runs, decay_factor)
    
    # Drop rows with NaN values (from failed experiments)
    results_df = results_df.dropna()
    
    # Save raw results
    results_df.to_csv("results/all_results.csv", index=False)
    
    # Analyze and plot results
    summary = analyze_results(results_df)
    
    log_print("Experiment completed successfully.")
    log_file.close()

if __name__ == "__main__":
    main()