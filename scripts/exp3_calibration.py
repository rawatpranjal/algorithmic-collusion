#!/usr/bin/env python3
"""
Exp3 Memory Decay Calibration

Sweeps memory_decay × lambda × algorithm to find meaningful levels for the
factorial design. Produces CSV + heatmaps showing revenue, regret, and
convergence across the grid.

Usage:
    PYTHONPATH=src python3 scripts/exp3_calibration.py --parallel --seeds 3
    PYTHONPATH=src python3 scripts/exp3_calibration.py --seeds 1 --no-plots
"""

import argparse
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_single(params):
    """Run one cell of the calibration grid. Importable for multiprocessing."""
    import sys, os
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from experiments.exp3 import run_bandit_experiment

    summary, revenues, _, _ = run_bandit_experiment(
        eta=params["eta"],
        auction_type=params["auction_type"],
        lam=params["lam"],
        n_bidders=params["n_bidders"],
        reserve_price=params["reserve_price"],
        max_rounds=params["max_rounds"],
        algorithm=params["algorithm"],
        exploration_intensity=params["exploration_intensity"],
        context_richness=params["context_richness"],
        memory_decay=params["memory_decay"],
        seed=params["seed"],
    )
    row = {
        "memory_decay": params["memory_decay"],
        "lam": params["lam"],
        "algorithm": params["algorithm"],
        "seed": params["seed"],
    }
    row.update(summary)
    return row


def build_grid(seeds, max_rounds=100_000):
    """Build the full parameter grid."""
    memory_decay_levels = [1.0, 0.9999, 0.999, 0.99, 0.9]
    lam_levels = [0.01, 0.1, 1.0, 5.0, 50.0]
    algorithms = ["linucb", "thompson"]

    # Fixed parameters
    fixed = dict(
        eta=0.5,
        auction_type="first",
        n_bidders=2,
        reserve_price=0.0,
        exploration_intensity="low",
        context_richness="minimal",
        max_rounds=max_rounds,
    )

    tasks = []
    for md, lam, algo in itertools.product(memory_decay_levels, lam_levels, algorithms):
        for s in range(seeds):
            params = dict(fixed)
            params["memory_decay"] = md
            params["lam"] = lam
            params["algorithm"] = algo
            params["seed"] = 42 + s
            tasks.append(params)
    return tasks


def make_heatmaps(df, output_dir):
    """Generate heatmaps of key metrics: memory_decay × lambda, per algorithm."""
    metrics = [
        ("avg_rev_last_1000", "Avg Revenue (Last 1k)", "viridis"),
        ("excess_regret", "Excess Regret", "RdYlGn_r"),
        ("price_volatility", "Price Volatility", "magma"),
    ]

    for algo in ["linucb", "thompson"]:
        sub = df[df["algorithm"] == algo]
        if sub.empty:
            continue

        for metric, label, cmap in metrics:
            pivot = sub.pivot_table(
                index="memory_decay", columns="lam", values=metric, aggfunc="mean"
            )
            # Sort index descending so 1.0 is at top
            pivot = pivot.sort_index(ascending=False)

            fig, ax = plt.subplots(figsize=(7, 5))
            im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{v:.2g}" for v in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{v}" for v in pivot.index])
            ax.set_xlabel("Lambda")
            ax.set_ylabel("Memory Decay")
            ax.set_title(f"{algo.upper()}: {label}")

            # Annotate cells
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=8, color="white" if val < pivot.values.mean() else "black")

            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fname = f"{algo}_{metric}.png"
            fig.savefig(os.path.join(output_dir, fname), dpi=150)
            plt.close(fig)
            print(f"  Saved {fname}")


def main():
    parser = argparse.ArgumentParser(description="Exp3 memory decay calibration")
    parser.add_argument("--seeds", type=int, default=3, help="Seeds per cell")
    parser.add_argument("--parallel", action="store_true", help="Use multiprocessing")
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers")
    parser.add_argument("--output-dir", type=str, default="results/calibration/exp3_memory_decay")
    parser.add_argument("--max-rounds", type=int, default=100_000, help="Rounds per run")
    parser.add_argument("--no-plots", action="store_true", help="Skip heatmap generation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tasks = build_grid(args.seeds, max_rounds=args.max_rounds)
    n_tasks = len(tasks)
    print(f"Calibration grid: {n_tasks} runs ({n_tasks // args.seeds} cells × {args.seeds} seeds)")

    results = []
    t0 = time.time()

    if args.parallel:
        workers = args.workers or max(1, os.cpu_count() // 2)
        print(f"Running with {workers} workers...")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_single, t): i for i, t in enumerate(tasks)}
            for done_count, future in enumerate(as_completed(futures), 1):
                row = future.result()
                results.append(row)
                if done_count % 10 == 0 or done_count == n_tasks:
                    elapsed = time.time() - t0
                    rate = done_count / elapsed
                    eta_min = (n_tasks - done_count) / rate / 60
                    print(f"  [{done_count}/{n_tasks}] {elapsed:.0f}s elapsed, ~{eta_min:.1f}min remaining")
    else:
        for i, t in enumerate(tasks):
            row = run_single(t)
            results.append(row)
            if (i + 1) % 5 == 0 or i == n_tasks - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta_min = (n_tasks - i - 1) / rate / 60
                print(f"  [{i+1}/{n_tasks}] {elapsed:.0f}s elapsed, ~{eta_min:.1f}min remaining")

    elapsed = time.time() - t0
    print(f"Completed {n_tasks} runs in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "exp3_factor_sweep.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    # Summary table: mean across seeds
    summary_cols = ["memory_decay", "lam", "algorithm"]
    metric_cols = ["avg_rev_last_1000", "excess_regret", "price_volatility",
                   "winner_entropy", "time_to_converge"]
    available_metrics = [c for c in metric_cols if c in df.columns]
    agg = df.groupby(summary_cols)[available_metrics].mean().round(4)
    print("\n=== Summary (mean across seeds) ===")
    print(agg.to_string())

    if not args.no_plots:
        print("\nGenerating heatmaps...")
        make_heatmaps(df, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
