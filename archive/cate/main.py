"""Main entry point for CATE influence function simulation."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import dgp, simulation
from .simulation import SCENARIOS


def create_plots(results_n1000: pd.DataFrame, results_n5000: pd.DataFrame,
                 output_dir: str):
    """Create visualization plots."""

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Coverage by X value (bar chart, 95% line)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (df, n) in zip(axes, [(results_n1000, 1000), (results_n5000, 5000)]):
        x_pos = np.arange(len(df))
        bars = ax.bar(x_pos, df['if_coverage'] * 100, color='steelblue', alpha=0.7)
        ax.axhline(y=95, color='red', linestyle='--', linewidth=2, label='95% target')
        ax.set_xlabel('X value')
        ax.set_ylabel('Coverage (%)')
        ax.set_title(f'IF-based Coverage (n={n})')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{x:.1f}' for x in df['x']])
        ax.set_ylim(80, 100)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'coverage.png'), dpi=150)
    plt.close()

    # 2. SE comparison (IF analytical vs Oracle MC, 45° line)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (df, n) in zip(axes, [(results_n1000, 1000), (results_n5000, 5000)]):
        ax.scatter(df['oracle_mc_se'], df['if_analytical_se'],
                   s=80, alpha=0.7, c='steelblue', edgecolors='black')

        # Add 45-degree line
        max_val = max(df['oracle_mc_se'].max(), df['if_analytical_se'].max()) * 1.1
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='45° line')

        # Label points with X values
        for i, row in df.iterrows():
            ax.annotate(f'{row["x"]:.1f}',
                        (row['oracle_mc_se'], row['if_analytical_se']),
                        textcoords='offset points', xytext=(5, 5), fontsize=8)

        ax.set_xlabel('Oracle Monte Carlo SE')
        ax.set_ylabel('IF Analytical SE')
        ax.set_title(f'SE Comparison (n={n})')
        ax.legend()
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'se_comparison.png'), dpi=150)
    plt.close()

    # 3. CATE curve (true τ(X) with IF ± 2SE bands)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (df, n) in zip(axes, [(results_n1000, 1000), (results_n5000, 5000)]):
        x = df['x'].values
        true_tau = df['true_tau'].values

        # Compute mean IF estimate from bias: E[τ̂] = τ + bias
        if_mean = true_tau + df['if_bias'].values
        if_se = df['if_analytical_se'].values

        # Plot true CATE
        ax.plot(x, true_tau, 'k-', linewidth=2, label='True τ(X)')

        # Plot IF estimates with error bands
        ax.plot(x, if_mean, 'b-', linewidth=1.5, label='IF estimate')
        ax.fill_between(x, if_mean - 2*if_se, if_mean + 2*if_se,
                        alpha=0.3, color='blue', label='±2 SE')

        ax.set_xlabel('X')
        ax.set_ylabel('CATE τ(X)')
        ax.set_title(f'CATE Estimation (n={n})')
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cate_curve.png'), dpi=150)
    plt.close()

    # 4. Bias-variance decomposition (stacked bar)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (df, n) in zip(axes, [(results_n1000, 1000), (results_n5000, 5000)]):
        x_pos = np.arange(len(df))
        width = 0.35

        bias_sq = df['if_bias']**2
        variance = df['if_variance']

        ax.bar(x_pos - width/2, bias_sq, width, label='Bias²', color='coral')
        ax.bar(x_pos - width/2, variance, width, bottom=bias_sq,
               label='Variance', color='steelblue')

        # Add RMSE line
        ax.plot(x_pos - width/2, df['if_rmse']**2, 'ko-', markersize=6,
                label='MSE (Bias² + Var)')

        ax.set_xlabel('X value')
        ax.set_ylabel('MSE Components')
        ax.set_title(f'Bias-Variance Decomposition (n={n})')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{x:.1f}' for x in df['x']])
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bias_variance.png'), dpi=150)
    plt.close()

    print(f"Plots saved to {output_dir}")


def main(R: int = 500, quick: bool = False):
    """
    Run the full CATE simulation study.

    Parameters
    ----------
    R : int
        Number of Monte Carlo replications
    quick : bool
        If True, use reduced parameters for quick testing
    """
    if quick:
        R = 50
        sample_sizes = [500]
        print("Running in quick mode: R=50, n=[500]")
    else:
        sample_sizes = [1000, 5000]
        print(f"Running full simulation: R={R}, n={sample_sizes}")

    # Get X values from DGP
    x_values = dgp.get_x_values()
    print(f"X values: {x_values}")
    print(f"True CATE: {dgp.true_tau(x_values)}")

    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)

    all_nuisance_metrics = []
    results_by_n = {}

    for n in sample_sizes:
        print(f"\n{'='*60}")
        print(f"Running simulation for n={n}")
        print('='*60)

        # Run Monte Carlo
        mc_results = simulation.run_monte_carlo(n, R, x_values)

        # Compute summary metrics
        summary = simulation.compute_summary_metrics(mc_results)
        results_by_n[n] = summary

        # Save results
        summary.to_csv(os.path.join(output_dir, f'results_n{n}.csv'), index=False)
        print(f"\nResults saved to results_n{n}.csv")

        # Collect nuisance metrics
        all_nuisance_metrics.append(mc_results['nuisance_metrics'])

        # Print summary
        print("\nSummary Statistics:")
        print(f"  Mean IF Coverage: {summary['if_coverage'].mean():.3f}")
        print(f"  Mean SE Ratio (IF analytical / Oracle MC): {summary['se_ratio'].mean():.3f}")
        print(f"  Mean SE Calibration (IF analytical / IF MC): {summary['se_calibration'].mean():.3f}")
        print(f"  Mean IF Bias: {summary['if_bias'].mean():.4f}")
        print(f"  Mean IF RMSE: {summary['if_rmse'].mean():.4f}")

    # Combine and save nuisance metrics
    nuisance_df = pd.concat(all_nuisance_metrics, ignore_index=True)
    nuisance_df.to_csv(os.path.join(output_dir, 'nuisance_fit.csv'), index=False)
    print(f"\nNuisance fit metrics saved to nuisance_fit.csv")

    # Print nuisance summary
    print("\nNuisance Fit Summary (mean across all simulations):")
    print(f"  μ₀ RMSE: {nuisance_df['mu0_rmse'].mean():.4f}")
    print(f"  μ₀ R²: {nuisance_df['mu0_r2'].mean():.4f}")
    print(f"  μ₁ RMSE: {nuisance_df['mu1_rmse'].mean():.4f}")
    print(f"  μ₁ R²: {nuisance_df['mu1_r2'].mean():.4f}")
    print(f"  π RMSE: {nuisance_df['pi_rmse'].mean():.4f}")
    print(f"  π R²: {nuisance_df['pi_r2'].mean():.4f}")
    print(f"  π AUC: {nuisance_df['pi_auc'].mean():.4f}")

    # Create plots
    if len(results_by_n) == 2:
        create_plots(results_by_n[1000], results_by_n[5000], output_dir)
    elif len(results_by_n) == 1:
        # Quick mode - create single-panel plots
        n = list(results_by_n.keys())[0]
        df = results_by_n[n]

        fig, ax = plt.subplots(figsize=(8, 5))
        x_pos = np.arange(len(df))
        ax.bar(x_pos, df['if_coverage'] * 100, color='steelblue', alpha=0.7)
        ax.axhline(y=95, color='red', linestyle='--', linewidth=2, label='95% target')
        ax.set_xlabel('X value')
        ax.set_ylabel('Coverage (%)')
        ax.set_title(f'IF-based Coverage (n={n})')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{x:.1f}' for x in df['x']])
        ax.set_ylim(80, 100)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'coverage.png'), dpi=150)
        plt.close()

        print(f"\nPlot saved to {output_dir}/coverage.png")

    print("\n" + "="*60)
    print("Simulation complete!")
    print("="*60)


def create_comparison_plots(results_df: pd.DataFrame, output_dir: str):
    """Create comparison plots for DR vs Plugin under misspecification."""

    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Bias comparison by scenario (grouped bar)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, scenario in zip(axes.flat, SCENARIOS.keys()):
        df = results_df[results_df['scenario'] == scenario]
        x_pos = np.arange(len(df))
        width = 0.35

        ax.bar(x_pos - width/2, np.abs(df['dr_bias']), width,
               label='DR', color='steelblue', alpha=0.8)
        ax.bar(x_pos + width/2, np.abs(df['plugin_bias']), width,
               label='Plugin', color='coral', alpha=0.8)

        ax.set_xlabel('X value')
        ax.set_ylabel('|Bias|')
        ax.set_title(f'{SCENARIOS[scenario]["description"]}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{x:.1f}' for x in df['x']])
        ax.legend()

    plt.suptitle('Absolute Bias: DR vs Plugin', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_bias.png'), dpi=150)
    plt.close()

    # 2. Coverage comparison by scenario
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, scenario in zip(axes.flat, SCENARIOS.keys()):
        df = results_df[results_df['scenario'] == scenario]
        x_pos = np.arange(len(df))
        width = 0.35

        ax.bar(x_pos - width/2, df['dr_coverage'] * 100, width,
               label='DR', color='steelblue', alpha=0.8)
        ax.bar(x_pos + width/2, df['plugin_coverage'] * 100, width,
               label='Plugin', color='coral', alpha=0.8)
        ax.axhline(y=95, color='red', linestyle='--', linewidth=2, label='95% target')

        ax.set_xlabel('X value')
        ax.set_ylabel('Coverage (%)')
        ax.set_title(f'{SCENARIOS[scenario]["description"]}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{x:.1f}' for x in df['x']])
        ax.set_ylim(0, 100)
        ax.legend()

    plt.suptitle('Coverage: DR vs Plugin', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_coverage.png'), dpi=150)
    plt.close()

    # 3. RMSE comparison by scenario
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, scenario in zip(axes.flat, SCENARIOS.keys()):
        df = results_df[results_df['scenario'] == scenario]
        x_pos = np.arange(len(df))
        width = 0.35

        ax.bar(x_pos - width/2, df['dr_rmse'], width,
               label='DR', color='steelblue', alpha=0.8)
        ax.bar(x_pos + width/2, df['plugin_rmse'], width,
               label='Plugin', color='coral', alpha=0.8)

        ax.set_xlabel('X value')
        ax.set_ylabel('RMSE')
        ax.set_title(f'{SCENARIOS[scenario]["description"]}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{x:.1f}' for x in df['x']])
        ax.legend()

    plt.suptitle('RMSE: DR vs Plugin', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_rmse.png'), dpi=150)
    plt.close()

    # 4. Summary table plot (mean metrics by scenario)
    summary_data = results_df.groupby('scenario').agg({
        'dr_bias': lambda x: np.abs(x).mean(),
        'dr_rmse': 'mean',
        'dr_coverage': 'mean',
        'plugin_bias': lambda x: np.abs(x).mean(),
        'plugin_rmse': 'mean',
        'plugin_coverage': 'mean'
    }).reset_index()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    # Reorder scenarios for display
    scenario_order = ['both_correct', 'mu_wrong', 'pi_wrong', 'both_wrong']
    summary_data = summary_data.set_index('scenario').loc[scenario_order].reset_index()

    cell_text = []
    for _, row in summary_data.iterrows():
        cell_text.append([
            SCENARIOS[row['scenario']]['description'],
            f"{row['dr_bias']:.4f}",
            f"{row['dr_rmse']:.4f}",
            f"{row['dr_coverage']*100:.1f}%",
            f"{row['plugin_bias']:.4f}",
            f"{row['plugin_rmse']:.4f}",
            f"{row['plugin_coverage']*100:.1f}%"
        ])

    table = ax.table(
        cellText=cell_text,
        colLabels=['Scenario', 'DR |Bias|', 'DR RMSE', 'DR Cov',
                   'Plugin |Bias|', 'Plugin RMSE', 'Plugin Cov'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    plt.title('Summary: DR vs Plugin by Misspecification Scenario', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Comparison plots saved to {output_dir}")


def run_comparison(R: int = 500, quick: bool = False):
    """
    Run DR vs Plugin comparison under misspecification scenarios.

    Parameters
    ----------
    R : int
        Number of Monte Carlo replications
    quick : bool
        If True, use reduced parameters for quick testing
    """
    if quick:
        R = 50
        n = 500
        print("Running comparison in quick mode: R=50, n=500")
    else:
        n = 1000
        print(f"Running comparison: R={R}, n={n}")

    # Get X values from DGP
    x_values = dgp.get_x_values()
    print(f"X values: {x_values}")
    print(f"True CATE: {dgp.true_tau(x_values)}")

    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("Running DR vs Plugin comparison across scenarios")
    print("="*60)

    # Run all scenarios
    all_results = simulation.run_all_comparison_scenarios(n, R, x_values)

    # Save results
    all_results.to_csv(os.path.join(output_dir, 'comparison_all.csv'), index=False)
    print(f"\nResults saved to comparison_all.csv")

    # Print summary by scenario
    print("\n" + "="*60)
    print("Summary by Scenario (mean across X values)")
    print("="*60)

    for scenario in SCENARIOS:
        df = all_results[all_results['scenario'] == scenario]
        print(f"\n{SCENARIOS[scenario]['description']}:")
        print(f"  DR  - Mean |Bias|: {np.abs(df['dr_bias']).mean():.4f}, "
              f"RMSE: {df['dr_rmse'].mean():.4f}, Coverage: {df['dr_coverage'].mean():.3f}")
        print(f"  Plugin - Mean |Bias|: {np.abs(df['plugin_bias']).mean():.4f}, "
              f"RMSE: {df['plugin_rmse'].mean():.4f}, Coverage: {df['plugin_coverage'].mean():.3f}")

    # Create comparison plots
    create_comparison_plots(all_results, output_dir)

    # Key findings
    print("\n" + "="*60)
    print("Key Findings")
    print("="*60)

    # Check double robustness
    both_correct = all_results[all_results['scenario'] == 'both_correct']
    mu_wrong = all_results[all_results['scenario'] == 'mu_wrong']
    pi_wrong = all_results[all_results['scenario'] == 'pi_wrong']
    both_wrong = all_results[all_results['scenario'] == 'both_wrong']

    dr_robust_mu = mu_wrong['dr_coverage'].mean() >= 0.90
    dr_robust_pi = pi_wrong['dr_coverage'].mean() >= 0.90
    plugin_biased = mu_wrong['plugin_coverage'].mean() < 0.85 or both_wrong['plugin_coverage'].mean() < 0.85

    print(f"\n1. Double Robustness Test:")
    print(f"   - DR robust when μ wrong, π correct: {'PASS' if dr_robust_mu else 'FAIL'} (coverage={mu_wrong['dr_coverage'].mean():.3f})")
    print(f"   - DR robust when π wrong, μ correct: {'PASS' if dr_robust_pi else 'FAIL'} (coverage={pi_wrong['dr_coverage'].mean():.3f})")

    print(f"\n2. Plugin Bias Check:")
    print(f"   - Plugin biased when μ wrong: {'YES' if mu_wrong['plugin_coverage'].mean() < 0.90 else 'NO'} (coverage={mu_wrong['plugin_coverage'].mean():.3f})")
    print(f"   - Plugin biased when both wrong: {'YES' if both_wrong['plugin_coverage'].mean() < 0.90 else 'NO'} (coverage={both_wrong['plugin_coverage'].mean():.3f})")

    print("\n" + "="*60)
    print("Comparison complete!")
    print("="*60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CATE Influence Function Simulation')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with reduced parameters')
    parser.add_argument('--R', type=int, default=500,
                        help='Number of Monte Carlo replications')
    parser.add_argument('--compare', action='store_true',
                        help='Run DR vs Plugin comparison under misspecification')

    args = parser.parse_args()

    if args.compare:
        run_comparison(R=args.R, quick=args.quick)
    else:
        main(R=args.R, quick=args.quick)
