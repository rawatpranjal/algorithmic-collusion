#!/usr/bin/env python3
"""
Discretization robustness check.

Tests whether experiment results are sensitive to the bid grid size
(number of discrete actions). For each experiment, runs a representative
subset of cells at multiple grid sizes and compares mean response values
and effect rankings across discretizations.

Usage:
    PYTHONPATH=src python3 scripts/discretization_robustness.py --exp 1
    PYTHONPATH=src python3 scripts/discretization_robustness.py --exp 2 --grid-sizes 6,11,21,41
    PYTHONPATH=src python3 scripts/discretization_robustness.py --exp 3 --seed 99

Output:
    results/expN/robust/discretization/comparison.csv
    results/expN/robust/discretization/summary.txt
    results/expN/robust/discretization/grid_comparison.png
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import csv
import itertools
import os
import sys
import textwrap
import time

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Path setup (mirrors PYTHONPATH=src convention)
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ---------------------------------------------------------------------------
# Response metrics extracted from each experiment's summary dict
# ---------------------------------------------------------------------------
RESPONSE_KEYS = [
    "avg_rev_last_1000",
    "price_volatility",
    "winner_entropy",
]


# ---------------------------------------------------------------------------
# Representative cells for each experiment
# ---------------------------------------------------------------------------
def get_exp1_cells():
    """
    4 corner cells for Exp1: 2 auction types x 2 n_bidders levels.
    Other factors held at fixed mid/default values.
    """
    base = dict(
        alpha=0.1,
        gamma=0.95,
        episodes=10_000,
        init="zeros",
        exploration="egreedy",
        asynchronous=0,
        info_feedback="minimal",
        reserve_price=0.0,
        decay_type="linear",
    )
    cells = []
    for auction_type in ["first", "second"]:
        for n_bidders in [2, 5]:
            cell = dict(base)
            cell["auction_type"] = auction_type
            cell["n_bidders"] = n_bidders
            cells.append(cell)
    return cells


def get_exp2_cells():
    """
    4 corner cells for Exp2: 2 auction types x 2 n_bidders levels.
    eta fixed at 0.5, state_info at signal_only.
    """
    base = dict(
        eta=0.5,
        state_info="signal_only",
        alpha=0.1,
        gamma=0.95,
        episodes=10_000,
    )
    cells = []
    for auction_type in ["first", "second"]:
        for n_bidders in [2, 5]:
            cell = dict(base)
            cell["auction_type"] = auction_type
            cell["n_bidders"] = n_bidders
            cells.append(cell)
    return cells


def get_exp3_cells():
    """
    4 corner cells for Exp3: 2 auction types x 2 n_bidders levels.
    Other factors at defaults.
    """
    base = dict(
        eta=0.5,
        algorithm="linucb",
        exploration_intensity="low",
        context_richness="minimal",
        lam=1.0,
        reserve_price=0.0,
        max_rounds=10_000,
    )
    cells = []
    for auction_type in ["first", "second"]:
        for n_bidders in [2, 5]:
            cell = dict(base)
            cell["auction_type"] = auction_type
            cell["n_bidders"] = n_bidders
            cells.append(cell)
    return cells


# ---------------------------------------------------------------------------
# Cell description (human-readable label)
# ---------------------------------------------------------------------------
def cell_label(cell):
    """Short string identifying the cell configuration."""
    parts = []
    if "auction_type" in cell:
        parts.append(cell["auction_type"])
    if "n_bidders" in cell:
        parts.append(f"N={cell['n_bidders']}")
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Runners: call the experiment function with the grid-size override
# ---------------------------------------------------------------------------
def run_exp1_cell(cell, grid_size, seed):
    from experiments.exp1 import run_experiment
    summary, _, _, _ = run_experiment(
        auction_type=cell["auction_type"],
        alpha=cell["alpha"],
        gamma=cell["gamma"],
        episodes=cell["episodes"],
        init=cell["init"],
        exploration=cell["exploration"],
        asynchronous=cell["asynchronous"],
        n_bidders=cell["n_bidders"],
        n_actions=grid_size,
        info_feedback=cell["info_feedback"],
        reserve_price=cell["reserve_price"],
        decay_type=cell["decay_type"],
        seed=seed,
    )
    return summary


def run_exp2_cell(cell, grid_size, seed):
    from experiments.exp2 import run_experiment
    summary, _, _, _ = run_experiment(
        eta=cell["eta"],
        auction_type=cell["auction_type"],
        n_bidders=cell["n_bidders"],
        state_info=cell["state_info"],
        alpha=cell["alpha"],
        gamma=cell["gamma"],
        episodes=cell["episodes"],
        n_bid_actions=grid_size,
        seed=seed,
    )
    return summary


def run_exp3_cell(cell, grid_size, seed):
    from experiments.exp3 import run_bandit_experiment
    summary, _, _, _ = run_bandit_experiment(
        eta=cell["eta"],
        auction_type=cell["auction_type"],
        lam=cell["lam"],
        n_bidders=cell["n_bidders"],
        reserve_price=cell["reserve_price"],
        max_rounds=cell["max_rounds"],
        algorithm=cell["algorithm"],
        exploration_intensity=cell["exploration_intensity"],
        context_richness=cell["context_richness"],
        n_bid_actions=grid_size,
        seed=seed,
    )
    return summary


# ---------------------------------------------------------------------------
# Mapping from experiment number to helpers
# ---------------------------------------------------------------------------
CELL_GETTERS = {1: get_exp1_cells, 2: get_exp2_cells, 3: get_exp3_cells}
CELL_RUNNERS = {1: run_exp1_cell, 2: run_exp2_cell, 3: run_exp3_cell}


# ---------------------------------------------------------------------------
# Effect ranking: for each response, rank cells by mean value at each grid
# ---------------------------------------------------------------------------
def compute_effect_rankings(rows, grid_sizes):
    """
    For each response key and grid size, rank the cells by their response
    value (highest = rank 1). Returns a dict:
        {response_key: {grid_size: [rank_for_cell_0, rank_for_cell_1, ...]}}
    """
    # Group rows by grid size
    by_grid = {}
    for r in rows:
        gs = r["grid_size"]
        by_grid.setdefault(gs, []).append(r)

    # Get unique cell labels in order
    cell_labels = []
    seen = set()
    for r in rows:
        lbl = r["cell"]
        if lbl not in seen:
            cell_labels.append(lbl)
            seen.add(lbl)

    rankings = {}
    for key in RESPONSE_KEYS:
        rankings[key] = {}
        for gs in grid_sizes:
            grid_rows = by_grid.get(gs, [])
            # Map cell label to value
            val_by_cell = {}
            for r in grid_rows:
                val_by_cell[r["cell"]] = r.get(key, float("nan"))
            # Sort cells by value (descending) and assign ranks
            sorted_cells = sorted(cell_labels,
                                  key=lambda c: val_by_cell.get(c, float("-inf")),
                                  reverse=True)
            rank_map = {c: i + 1 for i, c in enumerate(sorted_cells)}
            rankings[key][gs] = [rank_map.get(c, -1) for c in cell_labels]
    return rankings, cell_labels


def ranking_concordance(rankings, grid_sizes):
    """
    For each response, check whether the cell ranking is identical across
    all grid sizes. Returns a dict {response_key: bool}.
    """
    concordance = {}
    for key, gs_ranks in rankings.items():
        rank_lists = [tuple(gs_ranks[gs]) for gs in grid_sizes]
        concordance[key] = len(set(rank_lists)) == 1
    return concordance


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_grid_comparison(rows, grid_sizes, cell_labels, output_path):
    """
    One subplot per response metric. X-axis: grid sizes. Lines: cells.
    """
    n_metrics = len(RESPONSE_KEYS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4),
                             squeeze=False)
    axes = axes[0]

    # Organise data: {metric: {cell: {grid_size: value}}}
    data = {k: {cl: {} for cl in cell_labels} for k in RESPONSE_KEYS}
    for r in rows:
        for k in RESPONSE_KEYS:
            data[k][r["cell"]][r["grid_size"]] = r.get(k, float("nan"))

    for ax, key in zip(axes, RESPONSE_KEYS):
        for cl in cell_labels:
            vals = [data[key][cl].get(gs, float("nan")) for gs in grid_sizes]
            ax.plot(grid_sizes, vals, marker="o", label=cl)
        ax.set_xlabel("Grid size (n_actions)")
        ax.set_ylabel(key)
        ax.set_title(key.replace("_", " ").title())
        ax.set_xticks(grid_sizes)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Discretization robustness check: tests whether results "
                    "are sensitive to the bid grid size.")
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3, 4],
                        help="Experiment number (1-4)")
    parser.add_argument("--grid-sizes", type=str, default="6,11,21",
                        help="Comma-separated grid sizes to test (default: 6,11,21)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/expN/robust/discretization/)")
    args = parser.parse_args()

    # ---- Exp4: not applicable ----
    if args.exp == 4:
        print("Experiment 4 (dual pacing) does not use bid discretization.")
        print("This robustness check is not applicable. Exiting.")
        sys.exit(0)

    grid_sizes = [int(s.strip()) for s in args.grid_sizes.split(",")]
    grid_sizes.sort()

    # Output directory
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(ROOT_DIR, "results",
                               f"exp{args.exp}", "robust", "discretization")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Discretization robustness check for Experiment {args.exp}")
    print(f"  Grid sizes : {grid_sizes}")
    print(f"  Seed       : {args.seed}")
    print(f"  Output dir : {out_dir}")
    print()

    # ---- Get cells and runner ----
    cells = CELL_GETTERS[args.exp]()
    runner = CELL_RUNNERS[args.exp]
    labels = [cell_label(c) for c in cells]

    print(f"Representative cells ({len(cells)}):")
    for lbl in labels:
        print(f"  - {lbl}")
    print()

    # ---- Run experiments ----
    rows = []
    total = len(cells) * len(grid_sizes)
    done = 0

    for gs in grid_sizes:
        for cell, lbl in zip(cells, labels):
            done += 1
            print(f"[{done}/{total}] grid_size={gs:>3d}  cell={lbl:<20s} ...",
                  end="", flush=True)
            t0 = time.time()
            summary = runner(cell, gs, args.seed)
            elapsed = time.time() - t0
            print(f" {elapsed:.1f}s")

            row = {
                "grid_size": gs,
                "cell": lbl,
            }
            for key in RESPONSE_KEYS:
                row[key] = summary.get(key, float("nan"))
            rows.append(row)

    print()

    # ---- Save CSV ----
    csv_path = os.path.join(out_dir, "comparison.csv")
    fieldnames = ["grid_size", "cell"] + RESPONSE_KEYS
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved comparison table: {csv_path}")

    # ---- Compute rankings and concordance ----
    rankings, ranked_labels = compute_effect_rankings(rows, grid_sizes)
    concordance = ranking_concordance(rankings, grid_sizes)

    # ---- Plot ----
    fig_path = os.path.join(out_dir, "grid_comparison.png")
    plot_grid_comparison(rows, grid_sizes, ranked_labels, fig_path)
    print(f"Saved figure: {fig_path}")

    # ---- Text summary ----
    lines = []
    lines.append(f"Discretization Robustness Summary -- Experiment {args.exp}")
    lines.append("=" * 60)
    lines.append(f"Grid sizes tested: {grid_sizes}")
    lines.append(f"Cells: {ranked_labels}")
    lines.append(f"Seed: {args.seed}")
    lines.append("")

    # Per-metric table
    lines.append("--- Mean response values by grid size ---")
    lines.append("")
    for key in RESPONSE_KEYS:
        lines.append(f"  {key}:")
        header = f"    {'Cell':<22s}" + "".join(f"  grid={gs:<5d}" for gs in grid_sizes)
        lines.append(header)
        for lbl in ranked_labels:
            vals = []
            for gs in grid_sizes:
                for r in rows:
                    if r["grid_size"] == gs and r["cell"] == lbl:
                        vals.append(r.get(key, float("nan")))
                        break
            val_str = "".join(f"  {v:<10.4f}" for v in vals)
            lines.append(f"    {lbl:<22s}{val_str}")
        lines.append("")

    # Rankings
    lines.append("--- Cell rankings by grid size (1 = highest) ---")
    lines.append("")
    for key in RESPONSE_KEYS:
        lines.append(f"  {key}:")
        header = f"    {'Cell':<22s}" + "".join(f"  grid={gs:<5d}" for gs in grid_sizes)
        lines.append(header)
        for i, lbl in enumerate(ranked_labels):
            rank_vals = [str(rankings[key][gs][i]) for gs in grid_sizes]
            rank_str = "".join(f"  {rv:<10s}" for rv in rank_vals)
            lines.append(f"    {lbl:<22s}{rank_str}")
        same = concordance[key]
        lines.append(f"    Ranking stable across grids: {'YES' if same else 'NO'}")
        lines.append("")

    # Overall verdict
    all_stable = all(concordance.values())
    lines.append("--- Overall verdict ---")
    if all_stable:
        lines.append("All effect rankings are stable across grid sizes.")
        lines.append("Results appear robust to discretization.")
    else:
        unstable = [k for k, v in concordance.items() if not v]
        lines.append("Some rankings changed across grid sizes:")
        for k in unstable:
            lines.append(f"  - {k}")
        lines.append("Consider using a finer grid or investigating sensitivity.")

    # Max relative change
    lines.append("")
    lines.append("--- Maximum relative change (vs. finest grid) ---")
    finest = max(grid_sizes)
    for key in RESPONSE_KEYS:
        max_rel = 0.0
        for lbl in ranked_labels:
            finest_val = None
            for r in rows:
                if r["grid_size"] == finest and r["cell"] == lbl:
                    finest_val = r.get(key, float("nan"))
                    break
            if finest_val is None or finest_val == 0 or np.isnan(finest_val):
                continue
            for gs in grid_sizes:
                if gs == finest:
                    continue
                for r in rows:
                    if r["grid_size"] == gs and r["cell"] == lbl:
                        other_val = r.get(key, float("nan"))
                        if not np.isnan(other_val):
                            rel = abs(other_val - finest_val) / abs(finest_val)
                            max_rel = max(max_rel, rel)
                        break
        lines.append(f"  {key}: {max_rel:.1%}")

    summary_text = "\n".join(lines) + "\n"

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"Saved summary: {summary_path}")
    print()
    print(summary_text)


if __name__ == "__main__":
    main()
