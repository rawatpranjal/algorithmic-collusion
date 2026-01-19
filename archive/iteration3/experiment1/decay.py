#!/usr/bin/env python
# coding: utf-8

# Fix font issues before importing matplotlib.pyplot
import matplotlib
# Set font family and enable fallback
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Bitstream Vera Sans', 'Lucida Grande', 'Verdana', 'Geneva', 'sans-serif']
# Enable fallback to prevent the error
matplotlib.rcParams['mathtext.fallback'] = 'cm'
matplotlib.rcParams['mathtext.fontset'] = 'stix'

import numpy as np
import matplotlib.pyplot as plt
import os

# Create necessary directories
os.makedirs("figures", exist_ok=True)

def linear_decay(value_start, value_end, t, t_max):
    """Linear decay from value_start to value_end over t_max steps"""
    frac = t / float(t_max)
    decayed_val = value_start * (1 - frac) + value_end * frac
    return max(value_end, decayed_val)

def exponential_decay(value_start, value_end, t, t_max, decay_factor):
    """Exponential decay from value_start to value_end over t_max steps"""
    progress_ratio = t / float(t_max)
    decay = np.exp(-decay_factor * progress_ratio)
    decayed_val = value_end + (value_start - value_end) * decay
    return max(value_end, decayed_val)

def plot_decay_rates():
    """Plot comparison of decay rates with different factors"""
    # Parameters
    alpha_start = 0.1
    alpha_end = 0.0
    t_max = 1_000_000  # 1 million iterations
    
    # Sample points - use logarithmic scale to better show early behavior
    sample_points = np.concatenate([
        np.linspace(0, 100, 10),                # First 100 iterations
        np.linspace(100, 1000, 10),             # Next 900 iterations
        np.linspace(1000, 10000, 10),           # Next 9,000 iterations
        np.linspace(10000, 100000, 10),         # Next 90,000 iterations
        np.linspace(100000, 1000000, 10)        # Final 900,000 iterations
    ])
    sample_points = np.unique(np.round(sample_points).astype(int))
    
    # Generate decay values for different methods
    linear_values = [linear_decay(alpha_start, alpha_end, t, t_max) for t in sample_points]
    exp_values_factor1 = [exponential_decay(alpha_start, alpha_end, t, t_max, 1.0) for t in sample_points]
    exp_values_factor5 = [exponential_decay(alpha_start, alpha_end, t, t_max, 5.0) for t in sample_points]
    exp_values_factor10 = [exponential_decay(alpha_start, alpha_end, t, t_max, 10.0) for t in sample_points]

    # Plot the decay rates - ensure figure creation
    plt.figure(figsize=(12, 8))
    
    plt.plot(sample_points, linear_values, 'b-', linewidth=2, label='Linear Decay')
    plt.plot(sample_points, exp_values_factor1, 'g--', linewidth=2, label='Exponential Decay (factor=1.0)')
    plt.plot(sample_points, exp_values_factor5, 'r-.', linewidth=2, label='Exponential Decay (factor=5.0)')
    plt.plot(sample_points, exp_values_factor10, 'm:', linewidth=2, label='Exponential Decay (factor=10.0)')
    
    plt.title('Comparison of Learning Rate Decay Methods', fontsize=18)
    plt.xlabel('Training Iteration', fontsize=14)
    plt.ylabel('Learning Rate (α)', fontsize=14)
    plt.xscale('log')  # Use log scale to better show early behavior
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=12)
    
    # Add annotations to explain key behaviors - use simpler annotation
    plt.annotate('Exponential decay reduces\nthe rate much faster\nearly in training', 
                xy=(1000, 0.05), 
                xytext=(5000, 0.07),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=12, ha='center')
    
    plt.annotate('Higher decay factors\ncause faster initial drop', 
                xy=(100, exponential_decay(alpha_start, alpha_end, 100, t_max, 10.0)), 
                xytext=(100, 0.02),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=12, ha='center')
    
    # Adjust layout with a fixed padding to avoid font issues
    plt.tight_layout(pad=2.0)
    plt.savefig('figures/decay_rates_comparison.png', dpi=300, bbox_inches='tight')
    print("Created figures/decay_rates_comparison.png")
    
    # Create a zoomed-in version for the early iterations
    plt.figure(figsize=(12, 8))
    
    # Only use first 10% of training for this plot
    early_sample_points = np.linspace(0, t_max * 0.1, 100).astype(int)
    
    # Generate decay values for early iterations
    early_linear_values = [linear_decay(alpha_start, alpha_end, t, t_max) for t in early_sample_points]
    early_exp_values_factor1 = [exponential_decay(alpha_start, alpha_end, t, t_max, 1.0) for t in early_sample_points]
    early_exp_values_factor5 = [exponential_decay(alpha_start, alpha_end, t, t_max, 5.0) for t in early_sample_points]
    early_exp_values_factor10 = [exponential_decay(alpha_start, alpha_end, t, t_max, 10.0) for t in early_sample_points]
    
    plt.plot(early_sample_points, early_linear_values, 'b-', linewidth=2, label='Linear Decay')
    plt.plot(early_sample_points, early_exp_values_factor1, 'g--', linewidth=2, label='Exponential Decay (factor=1.0)')
    plt.plot(early_sample_points, early_exp_values_factor5, 'r-.', linewidth=2, label='Exponential Decay (factor=5.0)')
    plt.plot(early_sample_points, early_exp_values_factor10, 'm:', linewidth=2, label='Exponential Decay (factor=10.0)')
    
    plt.title('Early Training Behavior of Decay Methods (First 10% of Training)', fontsize=18)
    plt.xlabel('Training Iteration', fontsize=14)
    plt.ylabel('Learning Rate (α)', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=12)
    
    # Adjust layout with a fixed padding
    plt.tight_layout(pad=2.0)
    plt.savefig('figures/decay_rates_early_training.png', dpi=300, bbox_inches='tight')
    print("Created figures/decay_rates_early_training.png")
    
    # Print summary of decay speeds at different points
    print("\nDecay rate values at different points in training:")
    print(f"{'Method':<25} {'1%':<10} {'5%':<10} {'10%':<10} {'25%':<10} {'50%':<10}")
    print('-' * 75)
    
    points = [int(t_max * p) for p in [0.01, 0.05, 0.1, 0.25, 0.5]]
    
    print(f"{'Linear Decay':<25}", end='')
    for p in points:
        print(f"{linear_decay(alpha_start, alpha_end, p, t_max):<10.5f}", end='')
    print()
    
    print(f"{'Exponential (factor=1.0)':<25}", end='')
    for p in points:
        print(f"{exponential_decay(alpha_start, alpha_end, p, t_max, 1.0):<10.5f}", end='')
    print()
    
    print(f"{'Exponential (factor=5.0)':<25}", end='')
    for p in points:
        print(f"{exponential_decay(alpha_start, alpha_end, p, t_max, 5.0):<10.5f}", end='')
    print()
    
    print(f"{'Exponential (factor=10.0)':<25}", end='')
    for p in points:
        print(f"{exponential_decay(alpha_start, alpha_end, p, t_max, 10.0):<10.5f}", end='')
    print()

if __name__ == "__main__":
    plot_decay_rates()
    print("\nDecay rate plots created in figures/ directory.")