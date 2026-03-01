#!/usr/bin/env python3
"""
Unified CLI for running algorithmic collusion experiments.

Uses 2^k factorial design for structured parameter exploration.

Supports:
- Sequential execution (default)
- Local parallel execution (--parallel)
- Cloud execution (--cloud)

Examples:
    # Quick test (16-cell fractional factorial, 1 replicate)
    python scripts/run_experiment.py --exp 1 --quick

    # Full factorial (2 replicates per cell)
    python scripts/run_experiment.py --exp 1 --parallel

    # Custom replicates
    python scripts/run_experiment.py --exp 1 --parallel --replicates 5
"""

import argparse
import datetime
import hashlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cloud.config import LocalConfig, ExperimentConfig
from cloud.runner import DistributedRunner, run_sequential, TaskResult
from cloud.gcs import GCSBucket
from cloud.startup import generate_startup_script


# =====================================================================
# Factor Definitions (2 levels each, coded -1/+1)
# =====================================================================

EXP1_FACTORS = {
    # factor_name: (low_value, high_value)
    "auction_type":   ("second", "first"),
    "alpha":          (0.01, 0.1),
    "gamma":          (0.0, 0.95),
    "reserve_price":  (0.0, 0.5),
    "init":           ("zeros", "optimistic"),
    "exploration":    ("egreedy", "boltzmann"),
    "asynchronous":   (0, 1),
    "n_bidders":      (2, 4),
    "info_feedback":  ("minimal", "full"),
    "decay_type":     ("linear", "exponential"),
}

# Fixed n_actions for Exp1 (removed from factorial; tested in discretization robustness)
EXP1_N_ACTIONS = 11

# Exp2: Mixed-level design (3 × 2^3 = 24 cells)
# Binary factors (coded ±1)
EXP2_BINARY_FACTORS = {
    "auction_type": ("second", "first"),
    "n_bidders":    (2, 4),
    "state_info":   ("signal_only", "signal_winner"),
}
# 3-level factor
EXP2_ETA_LEVELS = [0.0, 0.5, 1.0]
# Legacy compat: full factor dict for design_info reporting
EXP2_FACTORS = {
    "auction_type": ("second", "first"),
    "n_bidders":    (2, 4),
    "state_info":   ("signal_only", "signal_winner"),
    "eta":          (0.0, 1.0),
}

# Exp3a: LinUCB — 3 × 2^7 = 384 cells
EXP3A_BINARY_FACTORS = {
    "auction_type":          ("second", "first"),
    "n_bidders":             (2, 4),
    "reserve_price":         (0.0, 0.3),
    "exploration_intensity": ("low", "high"),
    "context_richness":      ("minimal", "rich"),
    "lam":                   (0.1, 5.0),
    "memory_decay":          (1.0, 0.999),
}
EXP3A_ETA_LEVELS = [0.0, 0.5, 1.0]
# Legacy compat for design_info reporting
EXP3A_FACTORS = {
    "auction_type":          ("second", "first"),
    "n_bidders":             (2, 4),
    "reserve_price":         (0.0, 0.3),
    "eta":                   (0.0, 0.5, 1.0),
    "exploration_intensity": ("low", "high"),
    "context_richness":      ("minimal", "rich"),
    "lam":                   (0.1, 5.0),
    "memory_decay":          (1.0, 0.999),
}

# Exp3b: Thompson — 3 × 2^5 = 96 cells
EXP3B_BINARY_FACTORS = {
    "auction_type":          ("second", "first"),
    "n_bidders":             (2, 4),
    "reserve_price":         (0.0, 0.3),
    "exploration_intensity": ("low", "high"),
    "context_richness":      ("minimal", "rich"),
}
EXP3B_ETA_LEVELS = [0.0, 0.5, 1.0]
# Legacy compat for design_info reporting
EXP3B_FACTORS = {
    "auction_type":          ("second", "first"),
    "n_bidders":             (2, 4),
    "reserve_price":         (0.0, 0.3),
    "eta":                   (0.0, 0.5, 1.0),
    "exploration_intensity": ("low", "high"),
    "context_richness":      ("minimal", "rich"),
}

EXP4A_FACTORS = {
    "auction_type":      ("second", "first"),
    "objective":         ("value_max", "utility_max"),
    "n_bidders":         (2, 4),
    "budget_multiplier": (0.25, 1.0),
    "reserve_price":     (0.0, 0.3),
    "sigma":             (0.1, 0.5),
}

EXP4B_FACTORS = {
    "auction_type":      ("second", "first"),
    "aggressiveness":    (0.3, 3.0),
    "n_bidders":         (2, 4),
    "budget_multiplier": (0.25, 1.0),
    "reserve_price":     (0.0, 0.3),
    "sigma":             (0.1, 0.5),
}


# =====================================================================
# Factorial Design Builder
# =====================================================================

def build_factorial_design(factor_defs, replicates, seed, half_fraction=False):
    """
    Generate 2^k full (or 2^(k-1) half-fraction) factorial design.

    Returns list of dicts, each containing:
      - actual parameter values
      - coded values (factor_name + '_coded' = -1 or +1)
      - replicate number and seed

    For half-fraction (Resolution V): last factor's sign = product of
    all other factor signs. This aliases the highest-order interaction
    with the last main effect, keeping all main effects and 2-way
    interactions estimable.
    """
    factor_names = list(factor_defs.keys())
    k = len(factor_names)

    if half_fraction:
        # Generate full factorial on first k-1 factors
        base_factors = factor_names[:-1]
        last_factor = factor_names[-1]
        coded_combos = list(itertools.product([-1, 1], repeat=k - 1))

        # For Resolution V half-fraction: last factor = product of all others
        # This gives 2^(k-1) runs with Resolution V
        full_coded = []
        for combo in coded_combos:
            last_code = 1
            for c in combo:
                last_code *= c
            full_coded.append(list(combo) + [last_code])
    else:
        coded_combos = list(itertools.product([-1, 1], repeat=k))
        full_coded = [list(c) for c in coded_combos]

    design = []
    for rep in range(replicates):
        rep_seed = seed + rep
        for cell_id, coded in enumerate(full_coded):
            row = {
                "cell_id": cell_id,
                "replicate": rep,
                "seed": rep_seed * 10000 + cell_id,
            }
            for i, fname in enumerate(factor_names):
                low, high = factor_defs[fname]
                code = coded[i]
                row[fname + "_coded"] = code
                row[fname] = low if code == -1 else high
            design.append(row)

    return design


def build_quick_design(factor_defs, seed):
    """
    Generate a small 2^4 fractional factorial for quick testing.
    Picks the first 4 factors, 1 replicate, 16 cells.
    Remaining factors are set to their low (-1) level.
    """
    factor_names = list(factor_defs.keys())
    quick_factors = factor_names[:4]
    fixed_factors = factor_names[4:]

    coded_combos = list(itertools.product([-1, 1], repeat=4))

    design = []
    for cell_id, coded in enumerate(coded_combos):
        row = {
            "cell_id": cell_id,
            "replicate": 0,
            "seed": seed * 10000 + cell_id,
        }
        # Quick factors get varied
        for i, fname in enumerate(quick_factors):
            low, high = factor_defs[fname]
            code = coded[i]
            row[fname + "_coded"] = code
            row[fname] = low if code == -1 else high

        # Fixed factors stay at low level
        for fname in fixed_factors:
            low, high = factor_defs[fname]
            row[fname + "_coded"] = -1
            row[fname] = low

        design.append(row)

    return design


def build_exp2_design(replicates, seed):
    """
    Build 3 × 2^3 = 24 cell mixed-level factorial for Experiment 2.

    Binary factors (auction_type, n_bidders, state_info) are fully crossed
    with 3-level eta factor. Eta is coded with two orthogonal contrasts:
      - eta_linear_coded:    {-1, 0, +1}
      - eta_quadratic_coded: {+1, -2, +1}
    """
    binary_factors = EXP2_BINARY_FACTORS
    eta_levels = EXP2_ETA_LEVELS

    # Linear and quadratic contrast codes for eta
    eta_codes = {
        0.0: {"eta_linear_coded": -1, "eta_quadratic_coded": 1},
        0.5: {"eta_linear_coded": 0,  "eta_quadratic_coded": -2},
        1.0: {"eta_linear_coded": 1,  "eta_quadratic_coded": 1},
    }

    factor_names = list(binary_factors.keys())
    coded_combos = list(itertools.product([-1, 1], repeat=len(factor_names)))

    # Build 24 base cells (3 eta × 8 binary combos)
    base_cells = []
    for eta in eta_levels:
        for coded in coded_combos:
            base_cells.append((eta, coded))

    design = []
    for rep in range(replicates):
        rep_seed = seed + rep
        for cell_id, (eta, coded) in enumerate(base_cells):
            row = {
                "cell_id": cell_id,
                "replicate": rep,
                "seed": rep_seed * 10000 + cell_id,
            }
            # Binary factors
            for i, fname in enumerate(factor_names):
                low, high = binary_factors[fname]
                code = coded[i]
                row[fname + "_coded"] = code
                row[fname] = low if code == -1 else high
            # Eta
            row["eta"] = eta
            row["eta_linear_coded"] = eta_codes[eta]["eta_linear_coded"]
            row["eta_quadratic_coded"] = eta_codes[eta]["eta_quadratic_coded"]
            design.append(row)

    return design


def build_exp2_quick_design(seed):
    """
    Quick test for Exp2: all 24 cells × 1 replicate.
    """
    return build_exp2_design(replicates=1, seed=seed)


def build_exp3a_design(replicates, seed):
    """
    Build 3 x 2^7 = 384 cell mixed-level factorial for Experiment 3a (LinUCB).

    Binary factors (auction_type, n_bidders, reserve_price,
    exploration_intensity, context_richness, lam, memory_decay) fully crossed
    with 3-level eta factor. Eta coded with orthogonal contrasts:
      - eta_linear_coded:    {-1, 0, +1}
      - eta_quadratic_coded: {+1, -2, +1}
    """
    binary_factors = EXP3A_BINARY_FACTORS
    eta_levels = EXP3A_ETA_LEVELS

    eta_codes = {
        0.0: {"eta_linear_coded": -1, "eta_quadratic_coded": 1},
        0.5: {"eta_linear_coded": 0,  "eta_quadratic_coded": -2},
        1.0: {"eta_linear_coded": 1,  "eta_quadratic_coded": 1},
    }

    factor_names = list(binary_factors.keys())
    coded_combos = list(itertools.product([-1, 1], repeat=len(factor_names)))

    design = []
    for rep in range(replicates):
        rep_seed = seed + rep
        cell_id = 0
        for eta in eta_levels:
            for coded in coded_combos:
                row = {
                    "cell_id": cell_id,
                    "replicate": rep,
                    "seed": rep_seed * 10000 + cell_id,
                }
                for i, fname in enumerate(factor_names):
                    low, high = binary_factors[fname]
                    code = coded[i]
                    row[fname + "_coded"] = code
                    row[fname] = low if code == -1 else high
                row["eta"] = eta
                row["eta_linear_coded"] = eta_codes[eta]["eta_linear_coded"]
                row["eta_quadratic_coded"] = eta_codes[eta]["eta_quadratic_coded"]
                design.append(row)
                cell_id += 1

    return design


def build_exp3a_quick_design(seed):
    """Quick test for Exp3a: 2^4=16 subset of binary factors, eta fixed at 0.0."""
    design = build_quick_design(EXP3A_BINARY_FACTORS, seed)
    # Inject eta=0.0 with contrast codes into each row
    for row in design:
        row["eta"] = 0.0
        row["eta_linear_coded"] = -1
        row["eta_quadratic_coded"] = 1
    return design


def build_exp3b_design(replicates, seed):
    """
    Build 3 x 2^5 = 96 cell mixed-level factorial for Experiment 3b (Thompson).

    Binary factors (auction_type, n_bidders, reserve_price,
    exploration_intensity, context_richness) fully crossed with 3-level eta
    factor. Eta coded with orthogonal contrasts:
      - eta_linear_coded:    {-1, 0, +1}
      - eta_quadratic_coded: {+1, -2, +1}
    """
    binary_factors = EXP3B_BINARY_FACTORS
    eta_levels = EXP3B_ETA_LEVELS

    eta_codes = {
        0.0: {"eta_linear_coded": -1, "eta_quadratic_coded": 1},
        0.5: {"eta_linear_coded": 0,  "eta_quadratic_coded": -2},
        1.0: {"eta_linear_coded": 1,  "eta_quadratic_coded": 1},
    }

    factor_names = list(binary_factors.keys())
    coded_combos = list(itertools.product([-1, 1], repeat=len(factor_names)))

    design = []
    for rep in range(replicates):
        rep_seed = seed + rep
        cell_id = 0
        for eta in eta_levels:
            for coded in coded_combos:
                row = {
                    "cell_id": cell_id,
                    "replicate": rep,
                    "seed": rep_seed * 10000 + cell_id,
                }
                for i, fname in enumerate(factor_names):
                    low, high = binary_factors[fname]
                    code = coded[i]
                    row[fname + "_coded"] = code
                    row[fname] = low if code == -1 else high
                row["eta"] = eta
                row["eta_linear_coded"] = eta_codes[eta]["eta_linear_coded"]
                row["eta_quadratic_coded"] = eta_codes[eta]["eta_quadratic_coded"]
                design.append(row)
                cell_id += 1

    return design


def build_exp3b_quick_design(seed):
    """Quick test for Exp3b: 2^4=16 subset of binary factors, eta fixed at 0.0."""
    design = build_quick_design(EXP3B_BINARY_FACTORS, seed)
    # Inject eta=0.0 with contrast codes into each row
    for row in design:
        row["eta"] = 0.0
        row["eta_linear_coded"] = -1
        row["eta_quadratic_coded"] = 1
    return design


# =====================================================================
# Summary-only wrappers (reduce memory in parallel mode)
# =====================================================================

def _run_exp1_summary_only(**kwargs):
    from experiments import exp1
    summary, _, _, _ = exp1.run_experiment(**kwargs)
    return summary


def _run_exp2_summary_only(**kwargs):
    from experiments import exp2
    summary, _, _, _ = exp2.run_experiment(**kwargs)
    return summary


def _run_exp3_summary_only(**kwargs):
    from experiments import exp3
    summary, _, _, _ = exp3.run_bandit_experiment(**kwargs)
    return summary


def _run_exp4a_wrapper(**kwargs):
    from experiments import exp4a
    summary, episode_data, _ = exp4a.run_experiment(**kwargs)
    return {"summary": summary, "episode_data": episode_data}


def _run_exp4b_wrapper(**kwargs):
    from experiments import exp4b
    summary, episode_data, _ = exp4b.run_experiment(**kwargs)
    return {"summary": summary, "episode_data": episode_data}


# =====================================================================
# Task Generators (factorial design)
# =====================================================================

def get_exp1_tasks(quick, output_dir, seed=42, replicates=2):
    """Generate task list for Experiment 1 using half-fraction factorial."""
    if quick:
        design = build_quick_design(EXP1_FACTORS, seed)
    else:
        # 2^(10-1) = 512 cells, Resolution V half-fraction
        design = build_factorial_design(EXP1_FACTORS, replicates, seed,
                                        half_fraction=True)

    tasks = []
    for row in design:
        task_id = f"exp1_cell{row['cell_id']}_rep{row['replicate']}"
        tasks.append({
            "task_id": task_id,
            "func": _run_exp1_summary_only,
            "kwargs": {
                "auction_type": row["auction_type"],
                "alpha": float(row["alpha"]),
                "gamma": float(row["gamma"]),
                "episodes": 1000 if quick else 100_000,
                "init": row["init"],
                "exploration": row["exploration"],
                "asynchronous": int(row["asynchronous"]),
                "n_bidders": int(row["n_bidders"]),
                "n_actions": EXP1_N_ACTIONS,
                "info_feedback": row["info_feedback"],
                "reserve_price": float(row["reserve_price"]),
                "decay_type": row["decay_type"],
                "seed": row["seed"],
                "store_qtables": False,
                "qtable_folder": None,
            },
            "run_id": row["cell_id"],
            "design_row": row,
        })

    return tasks


def get_exp2_tasks(quick, output_dir, seed=42, replicates=8, n_bid_actions=None):
    """Generate task list for Experiment 2 using 3 × 2^3 mixed-level design."""
    if quick:
        design = build_exp2_quick_design(seed)
    else:
        design = build_exp2_design(replicates, seed)

    tasks = []
    for row in design:
        task_id = f"exp2_cell{row['cell_id']}_rep{row['replicate']}"
        kwargs = {
            "eta": float(row["eta"]),
            "auction_type": row["auction_type"],
            "n_bidders": int(row["n_bidders"]),
            "state_info": row["state_info"],
            "episodes": 1000 if quick else 100_000,
            "seed": row["seed"],
            "store_qtables": False,
            "qtable_folder": None,
        }
        if n_bid_actions is not None:
            kwargs["n_bid_actions"] = int(n_bid_actions)

        tasks.append({
            "task_id": task_id,
            "func": _run_exp2_summary_only,
            "kwargs": kwargs,
            "run_id": row["cell_id"],
            "design_row": row,
        })

    return tasks


def get_exp3a_tasks(quick, output_dir, seed=42, replicates=2):
    """Generate task list for Experiment 3a (LinUCB) using 3 x 2^7 mixed-level design."""
    if quick:
        design = build_exp3a_quick_design(seed)
    else:
        design = build_exp3a_design(replicates, seed)

    tasks = []
    for row in design:
        task_id = f"exp3a_cell{row['cell_id']}_rep{row['replicate']}"
        tasks.append({
            "task_id": task_id,
            "func": _run_exp3_summary_only,
            "kwargs": {
                "algorithm": "linucb",
                "eta": float(row["eta"]),
                "auction_type": row["auction_type"],
                "exploration_intensity": row["exploration_intensity"],
                "context_richness": row["context_richness"],
                "lam": float(row["lam"]),
                "memory_decay": float(row["memory_decay"]),
                "n_bidders": int(row["n_bidders"]),
                "reserve_price": float(row["reserve_price"]),
                "max_rounds": 1000 if quick else 100_000,
                "seed": row["seed"],
            },
            "run_id": row["cell_id"],
            "design_row": row,
        })

    return tasks


def get_exp3b_tasks(quick, output_dir, seed=42, replicates=2):
    """Generate task list for Experiment 3b (Thompson) using 3 x 2^5 mixed-level design."""
    if quick:
        design = build_exp3b_quick_design(seed)
    else:
        design = build_exp3b_design(replicates, seed)

    tasks = []
    for row in design:
        task_id = f"exp3b_cell{row['cell_id']}_rep{row['replicate']}"
        tasks.append({
            "task_id": task_id,
            "func": _run_exp3_summary_only,
            "kwargs": {
                "algorithm": "thompson",
                "eta": float(row["eta"]),
                "auction_type": row["auction_type"],
                "exploration_intensity": row["exploration_intensity"],
                "context_richness": row["context_richness"],
                "lam": 1.0,
                "memory_decay": 1.0,
                "n_bidders": int(row["n_bidders"]),
                "reserve_price": float(row["reserve_price"]),
                "max_rounds": 1000 if quick else 100_000,
                "seed": row["seed"],
            },
            "run_id": row["cell_id"],
            "design_row": row,
        })

    return tasks


def build_exp4a_design(seeds_per_cell, base_seed):
    """
    Build 2^k full factorial for Experiment 4a.

    Each cell gets seeds_per_cell independent seeds.
    Returns list of design rows with actual values + coded +-1 columns.
    """
    factors = EXP4A_FACTORS
    factor_names = list(factors.keys())
    coded_combos = list(itertools.product([-1, 1], repeat=len(factor_names)))

    design = []
    for cell_id, coded in enumerate(coded_combos):
        for seed_idx in range(seeds_per_cell):
            row = {
                "cell_id": cell_id,
                "replicate": seed_idx,
                "seed": base_seed * 10000 + cell_id * 1000 + seed_idx,
            }
            for i, fname in enumerate(factor_names):
                low, high = factors[fname]
                code = coded[i]
                row[fname + "_coded"] = code
                row[fname] = low if code == -1 else high
            design.append(row)

    return design


def get_exp4a_tasks(quick, output_dir, seed=42, replicates=50):
    """Generate task list for Experiment 4a: 2^k cells x seeds_per_cell."""
    if quick:
        design = build_exp4a_design(seeds_per_cell=2, base_seed=seed)
        n_episodes = 10
        T = 100
    else:
        design = build_exp4a_design(seeds_per_cell=replicates, base_seed=seed)
        n_episodes = 100
        T = 1000

    tasks = []
    for row in design:
        task_id = f"exp4a_cell{row['cell_id']}_seed{row['replicate']}"
        tasks.append({
            "task_id": task_id,
            "func": _run_exp4a_wrapper,
            "kwargs": {
                "auction_type": row["auction_type"],
                "objective": row["objective"],
                "n_bidders": int(row["n_bidders"]),
                "budget_multiplier": float(row["budget_multiplier"]),
                "reserve_price": float(row["reserve_price"]),
                "sigma": float(row["sigma"]),
                "n_episodes": n_episodes,
                "T": T,
                "seed": row["seed"],
            },
            "run_id": row["cell_id"],
            "design_row": row,
        })

    return tasks


def build_exp4b_design(seeds_per_cell, base_seed):
    """
    Build 2^k full factorial for Experiment 4b.

    Each cell gets seeds_per_cell independent seeds.
    Returns list of design rows with actual values + coded +-1 columns.
    """
    factors = EXP4B_FACTORS
    factor_names = list(factors.keys())
    coded_combos = list(itertools.product([-1, 1], repeat=len(factor_names)))

    design = []
    for cell_id, coded in enumerate(coded_combos):
        for seed_idx in range(seeds_per_cell):
            row = {
                "cell_id": cell_id,
                "replicate": seed_idx,
                "seed": base_seed * 10000 + cell_id * 1000 + seed_idx,
            }
            for i, fname in enumerate(factor_names):
                low, high = factors[fname]
                code = coded[i]
                row[fname + "_coded"] = code
                row[fname] = low if code == -1 else high
            design.append(row)

    return design


def get_exp4b_tasks(quick, output_dir, seed=42, replicates=50):
    """Generate task list for Experiment 4b: 2^k cells x seeds_per_cell."""
    if quick:
        design = build_exp4b_design(seeds_per_cell=2, base_seed=seed)
        n_episodes = 10
        T = 100
    else:
        design = build_exp4b_design(seeds_per_cell=replicates, base_seed=seed)
        n_episodes = 100
        T = 1000

    tasks = []
    for row in design:
        task_id = f"exp4b_cell{row['cell_id']}_seed{row['replicate']}"
        tasks.append({
            "task_id": task_id,
            "func": _run_exp4b_wrapper,
            "kwargs": {
                "auction_type": row["auction_type"],
                "aggressiveness": float(row["aggressiveness"]),
                "n_bidders": int(row["n_bidders"]),
                "budget_multiplier": float(row["budget_multiplier"]),
                "reserve_price": float(row["reserve_price"]),
                "sigma": float(row["sigma"]),
                "n_episodes": n_episodes,
                "T": T,
                "seed": row["seed"],
            },
            "run_id": row["cell_id"],
            "design_row": row,
        })

    return tasks


# =====================================================================
# Aggregation Functions
# =====================================================================

# =====================================================================
# Data Provenance Manifest
# =====================================================================

# Source files to checksum per experiment
_EXPERIMENT_SOURCE_FILES = {
    "1":  ["src/experiments/exp1.py", "scripts/run_experiment.py"],
    "2":  ["src/experiments/exp2.py", "scripts/run_experiment.py"],
    "3a": ["src/experiments/exp3.py", "scripts/run_experiment.py"],
    "3b": ["src/experiments/exp3.py", "scripts/run_experiment.py"],
    "4a": ["src/experiments/exp4a.py", "scripts/run_experiment.py"],
    "4b": ["src/experiments/exp4b.py", "scripts/run_experiment.py"],
}

def _file_sha256(filepath):
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_data_manifest(experiment_id, output_dir, n_rows, n_factors,
                        replicates=None, seed=None):
    """Save data_manifest.json with code fingerprints for staleness detection."""
    repo_root = Path(__file__).parent.parent

    # Git commit (best-effort)
    git_commit = None
    git_dirty = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=repo_root, timeout=5,
        )
        if result.returncode == 0:
            git_commit = result.stdout.strip()
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=repo_root, timeout=5,
        )
        if result.returncode == 0:
            git_dirty = len(result.stdout.strip()) > 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass  # git not available

    # Source file checksums
    source_checksums = {}
    source_files = _EXPERIMENT_SOURCE_FILES.get(str(experiment_id), [])
    for rel_path in source_files:
        abs_path = repo_root / rel_path
        if abs_path.exists():
            source_checksums[rel_path] = f"sha256:{_file_sha256(abs_path)}"

    manifest = {
        "experiment": str(experiment_id),
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "source_checksums": source_checksums,
        "n_rows": n_rows,
        "n_factors": n_factors,
    }
    if replicates is not None:
        manifest["replicates"] = replicates
    if seed is not None:
        manifest["seed"] = seed

    manifest_path = os.path.join(output_dir, "data_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Data manifest saved to '{manifest_path}'")


def _save_design_info(factor_defs, output_dir, experiment_id):
    """Save factor definitions as design_info.json."""
    info = {
        "experiment": experiment_id,
        "n_factors": len(factor_defs),
        "factors": {
            name: {"low": low, "high": high}
            for name, (low, high) in factor_defs.items()
        },
    }
    # Convert non-serializable types
    for name in info["factors"]:
        for level in ["low", "high"]:
            v = info["factors"][name][level]
            if isinstance(v, bool):
                info["factors"][name][level] = v
            elif hasattr(v, 'item'):
                info["factors"][name][level] = v.item()
    with open(os.path.join(output_dir, "design_info.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)


def aggregate_exp1_results(results, tasks, output_dir):
    """Aggregate parallel exp1 results into data.csv with coded columns."""
    import pandas as pd
    from experiments.exp1 import param_mappings, theoretical_revenue_constant_valuation

    os.makedirs(output_dir, exist_ok=True)
    _save_design_info(EXP1_FACTORS, output_dir, 1)

    with open(os.path.join(output_dir, "param_mappings.json"), "w") as f:
        json.dump(param_mappings, f, indent=2)

    rows = []
    theory_cache = {}

    for result, task in zip(results, tasks):
        if not result.success:
            print(f"Skipping failed task {result.task_id}: {result.error}")
            continue

        summary = result.result
        design = task["design_row"]

        cache_key = (int(design["n_bidders"]), design["auction_type"])
        if cache_key not in theory_cache:
            theory_cache[cache_key] = theoretical_revenue_constant_valuation(
                *cache_key)

        outcome = dict(summary)
        # Add actual values
        for fname in EXP1_FACTORS:
            outcome[fname] = design[fname]
        # Add coded values
        for fname in EXP1_FACTORS:
            outcome[fname + "_coded"] = design[fname + "_coded"]
        # Add design metadata
        outcome["cell_id"] = design["cell_id"]
        outcome["replicate"] = design["replicate"]
        outcome["seed"] = design["seed"]
        # Legacy columns for compatibility
        outcome["auction_type_code"] = param_mappings["auction_type"][design["auction_type"]]
        outcome["n_bidders"] = int(design["n_bidders"])
        outcome["theoretical_revenue"] = theory_cache[cache_key]
        if theory_cache[cache_key] > 1e-8:
            outcome["ratio_to_theory"] = outcome["avg_rev_last_1000"] / theory_cache[cache_key]
        else:
            outcome["ratio_to_theory"] = None

        rows.append(outcome)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Factorial design complete. {len(rows)} runs => '{csv_path}'")
    print(f"  Design: 2^({len(EXP1_FACTORS)}-1) = {2**(len(EXP1_FACTORS)-1)} cells (Resolution V half-fraction)")
    _save_data_manifest("1", output_dir, len(rows), len(EXP1_FACTORS))


def _save_exp2_design_info(output_dir):
    """Save Exp2 mixed-level design info."""
    info = {
        "experiment": 2,
        "design": "3 x 2^3 = 24 cells (mixed-level factorial)",
        "n_factors": 4,
        "factors": {
            "auction_type": {"levels": ["second", "first"], "coding": "binary +-1"},
            "n_bidders": {"levels": [2, 4], "coding": "binary +-1"},
            "state_info": {"levels": ["signal_only", "signal_winner"], "coding": "binary +-1"},
            "eta": {
                "levels": [0.0, 0.5, 1.0],
                "coding": "orthogonal contrasts: eta_linear (-1,0,+1), eta_quadratic (+1,-2,+1)",
            },
        },
    }
    with open(os.path.join(output_dir, "design_info.json"), "w") as f:
        json.dump(info, f, indent=2)


def aggregate_exp2_results(results, tasks, output_dir):
    """Aggregate parallel exp2 results into data.csv with coded columns."""
    import pandas as pd
    from experiments.exp2 import param_mappings

    os.makedirs(output_dir, exist_ok=True)
    _save_exp2_design_info(output_dir)

    with open(os.path.join(output_dir, "param_mappings.json"), "w") as f:
        json.dump(param_mappings, f, indent=2)

    rows = []
    for result, task in zip(results, tasks):
        if not result.success:
            print(f"Skipping failed task {result.task_id}: {result.error}")
            continue

        summary = result.result
        design = task["design_row"]

        outcome = dict(summary)
        # Actual values
        outcome["auction_type"] = design["auction_type"]
        outcome["n_bidders"] = int(design["n_bidders"])
        outcome["state_info"] = design["state_info"]
        outcome["eta"] = float(design["eta"])
        # Coded columns
        outcome["auction_type_coded"] = design["auction_type_coded"]
        outcome["n_bidders_coded"] = design["n_bidders_coded"]
        outcome["state_info_coded"] = design["state_info_coded"]
        outcome["eta_linear_coded"] = design["eta_linear_coded"]
        outcome["eta_quadratic_coded"] = design["eta_quadratic_coded"]
        # Design metadata
        outcome["cell_id"] = design["cell_id"]
        outcome["replicate"] = design["replicate"]
        outcome["seed"] = design["seed"]
        # Legacy compat
        outcome["auction_type_code"] = param_mappings["auction_type"][design["auction_type"]]
        # ratio_to_theory from summary's theoretical_revenue
        R_bne = outcome.get("theoretical_revenue", 0)
        if R_bne and R_bne > 1e-8:
            outcome["ratio_to_theory"] = outcome["avg_rev_last_1000"] / R_bne
        else:
            outcome["ratio_to_theory"] = None

        rows.append(outcome)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Factorial design complete. {len(rows)} runs => '{csv_path}'")
    print(f"  Design: 3 x 2^3 = 24 cells (mixed-level factorial)")
    _save_data_manifest("2", output_dir, len(rows), 4)


def _save_exp3a_design_info(output_dir):
    """Save Exp3a (LinUCB) mixed-level design info."""
    info = {
        "experiment": "3a",
        "design": "3 x 2^7 = 384 cells (mixed-level factorial)",
        "n_factors": 8,
        "factors": {
            "auction_type": {"levels": ["second", "first"], "coding": "binary +-1"},
            "n_bidders": {"levels": [2, 4], "coding": "binary +-1"},
            "reserve_price": {"levels": [0.0, 0.3], "coding": "binary +-1"},
            "eta": {
                "levels": [0.0, 0.5, 1.0],
                "coding": "orthogonal contrasts: eta_linear (-1,0,+1), eta_quadratic (+1,-2,+1)",
            },
            "exploration_intensity": {"levels": ["low", "high"], "coding": "binary +-1"},
            "context_richness": {"levels": ["minimal", "rich"], "coding": "binary +-1"},
            "lam": {"levels": [0.1, 5.0], "coding": "binary +-1"},
            "memory_decay": {"levels": [1.0, 0.999], "coding": "binary +-1"},
        },
    }
    with open(os.path.join(output_dir, "design_info.json"), "w") as f:
        json.dump(info, f, indent=2)


def _save_exp3b_design_info(output_dir):
    """Save Exp3b (Thompson) mixed-level design info."""
    info = {
        "experiment": "3b",
        "design": "3 x 2^5 = 96 cells (mixed-level factorial)",
        "n_factors": 6,
        "factors": {
            "auction_type": {"levels": ["second", "first"], "coding": "binary +-1"},
            "n_bidders": {"levels": [2, 4], "coding": "binary +-1"},
            "reserve_price": {"levels": [0.0, 0.3], "coding": "binary +-1"},
            "eta": {
                "levels": [0.0, 0.5, 1.0],
                "coding": "orthogonal contrasts: eta_linear (-1,0,+1), eta_quadratic (+1,-2,+1)",
            },
            "exploration_intensity": {"levels": ["low", "high"], "coding": "binary +-1"},
            "context_richness": {"levels": ["minimal", "rich"], "coding": "binary +-1"},
        },
    }
    with open(os.path.join(output_dir, "design_info.json"), "w") as f:
        json.dump(info, f, indent=2)


def aggregate_exp3a_results(results, tasks, output_dir):
    """Aggregate parallel exp3a (LinUCB) results into data.csv with coded columns."""
    import pandas as pd
    from experiments.exp3 import param_mappings, simulate_linear_affiliation_revenue

    os.makedirs(output_dir, exist_ok=True)
    _save_exp3a_design_info(output_dir)

    with open(os.path.join(output_dir, "param_mappings.json"), "w") as f:
        json.dump(param_mappings, f, indent=2)

    rows = []
    theory_cache = {}

    for result, task in zip(results, tasks):
        if not result.success:
            print(f"Skipping failed task {result.task_id}: {result.error}")
            continue

        summary = result.result
        design = task["design_row"]

        cache_key = (int(design["n_bidders"]), float(design["eta"]),
                     design["auction_type"])
        if cache_key not in theory_cache:
            theory_cache[cache_key] = simulate_linear_affiliation_revenue(*cache_key)

        outcome = dict(summary)
        # Binary factor actual + coded values
        for fname in EXP3A_BINARY_FACTORS:
            outcome[fname] = design[fname]
            outcome[fname + "_coded"] = design[fname + "_coded"]
        # Eta actual + contrast codes
        outcome["eta"] = float(design["eta"])
        outcome["eta_linear_coded"] = design["eta_linear_coded"]
        outcome["eta_quadratic_coded"] = design["eta_quadratic_coded"]
        # Design metadata
        outcome["cell_id"] = design["cell_id"]
        outcome["replicate"] = design["replicate"]
        outcome["seed"] = design["seed"]
        # Legacy compat
        outcome["auction_type_code"] = param_mappings["auction_type"][design["auction_type"]]
        outcome["n_bidders"] = int(design["n_bidders"])
        outcome["theoretical_revenue"] = theory_cache[cache_key]
        if theory_cache[cache_key] > 1e-8:
            outcome["ratio_to_theory"] = outcome["avg_rev_last_1000"] / theory_cache[cache_key]
        else:
            outcome["ratio_to_theory"] = None

        rows.append(outcome)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Factorial design complete. {len(rows)} runs => '{csv_path}'")
    print(f"  Design: 3 x 2^7 = 384 cells (mixed-level factorial)")
    _save_data_manifest("3a", output_dir, len(rows), 8)


def aggregate_exp3b_results(results, tasks, output_dir):
    """Aggregate parallel exp3b (Thompson) results into data.csv with coded columns."""
    import pandas as pd
    from experiments.exp3 import param_mappings, simulate_linear_affiliation_revenue

    os.makedirs(output_dir, exist_ok=True)
    _save_exp3b_design_info(output_dir)

    with open(os.path.join(output_dir, "param_mappings.json"), "w") as f:
        json.dump(param_mappings, f, indent=2)

    rows = []
    theory_cache = {}

    for result, task in zip(results, tasks):
        if not result.success:
            print(f"Skipping failed task {result.task_id}: {result.error}")
            continue

        summary = result.result
        design = task["design_row"]

        cache_key = (int(design["n_bidders"]), float(design["eta"]),
                     design["auction_type"])
        if cache_key not in theory_cache:
            theory_cache[cache_key] = simulate_linear_affiliation_revenue(*cache_key)

        outcome = dict(summary)
        # Binary factor actual + coded values
        for fname in EXP3B_BINARY_FACTORS:
            outcome[fname] = design[fname]
            outcome[fname + "_coded"] = design[fname + "_coded"]
        # Eta actual + contrast codes
        outcome["eta"] = float(design["eta"])
        outcome["eta_linear_coded"] = design["eta_linear_coded"]
        outcome["eta_quadratic_coded"] = design["eta_quadratic_coded"]
        # Design metadata
        outcome["cell_id"] = design["cell_id"]
        outcome["replicate"] = design["replicate"]
        outcome["seed"] = design["seed"]
        # Legacy compat
        outcome["auction_type_code"] = param_mappings["auction_type"][design["auction_type"]]
        outcome["n_bidders"] = int(design["n_bidders"])
        outcome["theoretical_revenue"] = theory_cache[cache_key]
        if theory_cache[cache_key] > 1e-8:
            outcome["ratio_to_theory"] = outcome["avg_rev_last_1000"] / theory_cache[cache_key]
        else:
            outcome["ratio_to_theory"] = None

        rows.append(outcome)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Factorial design complete. {len(rows)} runs => '{csv_path}'")
    print(f"  Design: 3 x 2^5 = 96 cells (mixed-level factorial)")
    _save_data_manifest("3b", output_dir, len(rows), 6)


def _save_exp4a_design_info(output_dir):
    """Save Exp4a design info."""
    info = {
        "experiment": "4a",
        "design": "2^6 = 64 cells (full factorial) x seeds",
        "n_factors": 6,
        "factors": {
            name: {"low": low, "high": high}
            for name, (low, high) in EXP4A_FACTORS.items()
        },
    }
    # Convert non-serializable types
    for name in info["factors"]:
        for level in ["low", "high"]:
            v = info["factors"][name][level]
            if hasattr(v, 'item'):
                info["factors"][name][level] = v.item()
    with open(os.path.join(output_dir, "design_info.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)


def aggregate_exp4a_results(results, tasks, output_dir):
    """Aggregate parallel exp4a results into data.csv."""
    import pandas as pd
    from experiments.exp4a import param_mappings

    os.makedirs(output_dir, exist_ok=True)
    _save_exp4a_design_info(output_dir)

    with open(os.path.join(output_dir, "param_mappings.json"), "w") as f:
        json.dump(param_mappings, f, indent=2)

    run_rows = []

    for result, task in zip(results, tasks):
        if not result.success:
            print(f"Skipping failed task {result.task_id}: {result.error}")
            continue

        payload = result.result
        summary = payload["summary"]
        design = task["design_row"]

        # Run-level row
        outcome = dict(summary)
        for fname in EXP4A_FACTORS:
            outcome[fname] = design[fname]
            outcome[fname + "_coded"] = design[fname + "_coded"]
        outcome["cell_id"] = design["cell_id"]
        outcome["replicate"] = design["replicate"]
        outcome["seed"] = design["seed"]
        outcome["auction_type_code"] = param_mappings["auction_type"][design["auction_type"]]
        outcome["objective_code"] = param_mappings["objective"][design["objective"]]
        outcome["n_bidders"] = int(design["n_bidders"])
        run_rows.append(outcome)

    # Save run-level data
    df = pd.DataFrame(run_rows)
    csv_path = os.path.join(output_dir, "data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Factorial design complete. {len(run_rows)} runs => '{csv_path}'")
    print(f"  Design: 2^6 = 64 cells (full factorial)")
    _save_data_manifest("4a", output_dir, len(run_rows), len(EXP4A_FACTORS))


def _save_exp4b_design_info(output_dir):
    """Save Exp4b design info."""
    info = {
        "experiment": "4b",
        "design": "2^6 = 64 cells (full factorial) x seeds",
        "n_factors": 6,
        "factors": {
            name: {"low": low, "high": high}
            for name, (low, high) in EXP4B_FACTORS.items()
        },
    }
    for name in info["factors"]:
        for level in ["low", "high"]:
            v = info["factors"][name][level]
            if hasattr(v, 'item'):
                info["factors"][name][level] = v.item()
    with open(os.path.join(output_dir, "design_info.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)


def aggregate_exp4b_results(results, tasks, output_dir):
    """Aggregate parallel exp4b results into data.csv."""
    import pandas as pd
    from experiments.exp4b import param_mappings

    os.makedirs(output_dir, exist_ok=True)
    _save_exp4b_design_info(output_dir)

    with open(os.path.join(output_dir, "param_mappings.json"), "w") as f:
        json.dump(param_mappings, f, indent=2)

    run_rows = []

    for result, task in zip(results, tasks):
        if not result.success:
            print(f"Skipping failed task {result.task_id}: {result.error}")
            continue

        payload = result.result
        summary = payload["summary"]
        design = task["design_row"]

        # Run-level row
        outcome = dict(summary)
        for fname in EXP4B_FACTORS:
            outcome[fname] = design[fname]
            outcome[fname + "_coded"] = design[fname + "_coded"]
        outcome["cell_id"] = design["cell_id"]
        outcome["replicate"] = design["replicate"]
        outcome["seed"] = design["seed"]
        outcome["auction_type_code"] = param_mappings["auction_type"][design["auction_type"]]
        outcome["n_bidders"] = int(design["n_bidders"])
        run_rows.append(outcome)

    # Save run-level data
    df = pd.DataFrame(run_rows)
    csv_path = os.path.join(output_dir, "data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Factorial design complete. {len(run_rows)} runs => '{csv_path}'")
    print(f"  Design: 2^6 = 64 cells (full factorial)")
    _save_data_manifest("4b", output_dir, len(run_rows), len(EXP4B_FACTORS))


# =====================================================================
# Sequential Mode
# =====================================================================

def run_exp_sequential(exp, quick, output_dir):
    """Run experiment sequentially using subprocess."""
    import subprocess
    if exp == "1":
        cmd = [sys.executable, "src/experiments/exp1.py"]
    elif exp == "2":
        cmd = [sys.executable, "src/experiments/exp2.py"]
    elif exp in ("3a", "3b"):
        print("Error: Experiments 3a and 3b do not support sequential mode. Use --parallel.")
        sys.exit(1)
    else:
        print(f"Error: Experiment {exp} does not support sequential mode. Use --parallel.")
        sys.exit(1)
    if quick:
        cmd.append("--quick")
    subprocess.run(cmd, check=True)


# =====================================================================
# Cloud Mode (fire-and-forget)
# =====================================================================

def run_detached(args):
    """Launch experiment on cloud VM in fire-and-forget mode."""
    import subprocess
    from cloud.vm import CloudVM, VMConfig

    if args.project:
        project_id = args.project
    else:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True, text=True
        )
        project_id = result.stdout.strip()
        if not project_id:
            print("No GCP project specified. Use --project or run 'gcloud config set project PROJECT_ID'")
            sys.exit(1)

    print(f"Using GCP project: {project_id}")

    bucket_name = f"{project_id}-collusion-experiments"
    bucket = GCSBucket(bucket_name, project_id)
    bucket.create()

    print("Uploading code to GCS...")
    bucket.upload_code()

    startup_script = generate_startup_script(
        bucket_name=bucket_name,
        exp=args.exp,
        quick=args.quick,
        parallel=True,
        runs=args.replicates
    )

    vm_config = VMConfig(
        name=f"collusion-exp{args.exp}",
        machine_type="n2-standard-8",
        startup_script=startup_script
    )
    vm = CloudVM(project_id, vm_config)

    if vm.exists():
        print(f"VM {vm_config.name} already exists. Delete it first or use a different name.")
        sys.exit(1)

    vm.create()

    print(f"\n{'='*60}")
    print(f"Experiment {args.exp} launched in detached mode!")
    print(f"{'='*60}")
    print(f"\nVM: {vm_config.name}")
    print(f"Bucket: gs://{bucket_name}/")
    print(f"\nYou can close this terminal now.")


def run_cloud(args):
    """Run experiment on Google Cloud."""
    from cloud.vm import CloudVM, VMConfig

    if args.project:
        project_id = args.project
    else:
        import subprocess
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True, text=True
        )
        project_id = result.stdout.strip()
        if not project_id:
            print("No GCP project specified. Use --project or run 'gcloud config set project PROJECT_ID'")
            sys.exit(1)

    print(f"Using GCP project: {project_id}")
    workers = args.workers or 4

    vm_config = VMConfig(
        name=f"collusion-exp{args.exp}",
        machine_type="n2-standard-8",
    )
    vm = CloudVM(project_id, vm_config)

    try:
        if not vm.exists():
            vm.create()
            vm.setup_environment()
            vm.upload_code()
            vm.install_requirements()
        else:
            print(f"VM {vm_config.name} already exists, reusing...")
            vm.upload_code()

        print(f"\n{'='*50}")
        print(f"Running Experiment {args.exp} on cloud VM...")
        print(f"{'='*50}\n")

        remote_results = vm.run_experiment(
            exp=args.exp,
            quick=args.quick,
            parallel=True,
            workers=workers,
            runs=args.replicates
        )

        suffix = "/quick_test" if args.quick else ""
        local_output = args.output_dir or f"results/exp{args.exp}{suffix}_cloud"
        vm.download_results(remote_results, local_output)

        print(f"\nResults downloaded to: {local_output}")

        response = input("\nDelete cloud VM? [y/N]: ").strip().lower()
        if response == 'y':
            vm.delete()

    except KeyboardInterrupt:
        print("\nInterrupted. VM is still running.")
    except Exception as e:
        print(f"\nError: {e}")
        raise


# =====================================================================
# Main CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified CLI for algorithmic collusion experiments (factorial design)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--exp", type=str, required=True,
        choices=["1", "2", "3", "3a", "3b", "4", "4a", "5", "4b"],
        help="Experiment number (1, 2, 3a, 3b, 4/4a, or 5/4b)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick test: 2^4=16 cells, 1 replicate, 1000 episodes"
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Run in parallel using multiprocessing"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: cpu_count // 2)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: results/expN)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--replicates", type=int, default=2,
        help="Number of replicates per cell (default: 2)"
    )
    parser.add_argument(
        "--exp2-bid-actions", type=int, default=11,
        help="Exp2 bid grid size (default: 11)"
    )

    # Cloud options
    parser.add_argument("--cloud", action="store_true")
    parser.add_argument("--detached", action="store_true")
    parser.add_argument("--project", type=str)
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--download", action="store_true")

    args = parser.parse_args()

    # Handle legacy --exp 3 (now split into 3a and 3b)
    if args.exp == "3":
        print("Error: Experiment 3 has been split into 3a (LinUCB) and 3b (Thompson).")
        print("Use --exp 3a or --exp 3b.")
        sys.exit(1)

    # Legacy aliases: --exp 4 -> 4a, --exp 5 -> 4b
    if args.exp == "4":
        args.exp = "4a"
    elif args.exp == "5":
        args.exp = "4b"

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        suffix = "/quick_test" if args.quick else ""
        output_dir = f"results/exp{args.exp}{suffix}"

    # Cloud mode
    if args.cloud:
        if args.detached:
            run_detached(args)
        else:
            run_cloud(args)
        return

    if args.monitor or args.download:
        print("Use --cloud to manage cloud VMs")
        sys.exit(1)

    # Print design info
    if args.exp == "2":
        # Exp2 uses mixed-level design
        if args.quick:
            n_cells = 24
            n_reps = 1
            total = 24
            print(f"Experiment 2: Quick test (3 x 2^3 = 24 cells, 1 replicate)")
            print(f"  Total runs: {total}")
        else:
            n_cells = 24
            n_reps = args.replicates
            total = n_cells * n_reps
            design_desc = "3 x 2^3 = 24 (mixed-level factorial)"
            print(f"Experiment 2: {design_desc}")
            print(f"  Replicates: {n_reps}")
            print(f"  Total runs: {total}")
    elif args.exp == "3a":
        # Exp3a: LinUCB, 3 x 2^7 = 384 cells
        if args.quick:
            n_cells = 16
            n_reps = 1
            total = 16
            print(f"Experiment 3a (LinUCB): Quick test (2^4 = 16 cells, 1 replicate, eta=0.0)")
            print(f"  Total runs: {total}")
        else:
            n_cells = 384
            n_reps = args.replicates
            total = n_cells * n_reps
            design_desc = "3 x 2^7 = 384 (mixed-level factorial)"
            print(f"Experiment 3a (LinUCB): {design_desc}")
            print(f"  Replicates: {n_reps}")
            print(f"  Total runs: {total}")
    elif args.exp == "3b":
        # Exp3b: Thompson, 3 x 2^5 = 96 cells
        if args.quick:
            n_cells = 16
            n_reps = 1
            total = 16
            print(f"Experiment 3b (Thompson): Quick test (2^4 = 16 cells, 1 replicate, eta=0.0)")
            print(f"  Total runs: {total}")
        else:
            n_cells = 96
            n_reps = args.replicates
            total = n_cells * n_reps
            design_desc = "3 x 2^5 = 96 (mixed-level factorial)"
            print(f"Experiment 3b (Thompson): {design_desc}")
            print(f"  Replicates: {n_reps}")
            print(f"  Total runs: {total}")
    elif args.exp in ("4a", "4b"):
        factors = EXP4A_FACTORS if args.exp == "4a" else EXP4B_FACTORS
        n_cells = 2 ** len(factors)
        if args.quick:
            seeds = 2
            total = n_cells * seeds
            print(f"Experiment {args.exp}: Quick test (2^{len(factors)} = {n_cells} cells, {seeds} seeds/cell)")
            print(f"  Total runs: {total}")
        else:
            seeds = args.replicates
            total = n_cells * seeds
            print(f"Experiment {args.exp}: 2^{len(factors)} = {n_cells} cells, {seeds} seeds/cell")
            print(f"  Total runs: {total}")
    else:
        factor_defs = EXP1_FACTORS
        k = len(factor_defs)
        if args.quick:
            n_cells = 16
            n_reps = 1
            total = 16
        else:
            n_cells = 2 ** (k - 1)
            design_desc = f"2^({k}-1) = {n_cells} (Resolution V half-fraction)"
            n_reps = args.replicates
            total = n_cells * n_reps

        if not args.quick:
            print(f"Experiment {args.exp}: {design_desc}")
            print(f"  Replicates: {n_reps}")
            print(f"  Total runs: {total}")
        else:
            print(f"Experiment {args.exp}: Quick test (2^4 = 16 cells, 1 replicate)")
            print(f"  Total runs: {total}")

    # Sequential mode
    if not args.parallel:
        print(f"\nRunning Experiment {args.exp} sequentially...")
        run_exp_sequential(args.exp, args.quick, output_dir)
        return

    # Parallel mode
    print(f"\nRunning Experiment {args.exp} in parallel...")

    if args.exp == "1":
        tasks = get_exp1_tasks(args.quick, output_dir, args.seed, args.replicates)
        desc = "Experiment 1: Constant Values (Factorial)"
    elif args.exp == "2":
        tasks = get_exp2_tasks(args.quick, output_dir, args.seed, args.replicates,
                               n_bid_actions=args.exp2_bid_actions)
        desc = "Experiment 2: Affiliated Values (Factorial)"
    elif args.exp == "3a":
        tasks = get_exp3a_tasks(args.quick, output_dir, args.seed, args.replicates)
        desc = "Experiment 3a: LinUCB Bandits (Mixed-Level Factorial)"
    elif args.exp == "3b":
        tasks = get_exp3b_tasks(args.quick, output_dir, args.seed, args.replicates)
        desc = "Experiment 3b: Thompson Sampling (Mixed-Level Factorial)"
    elif args.exp == "4a":
        tasks = get_exp4a_tasks(args.quick, output_dir, args.seed, args.replicates)
        desc = "Experiment 4a: Autobidding Dual Pacing (Full Factorial)"
    else:
        tasks = get_exp4b_tasks(args.quick, output_dir, args.seed, args.replicates)
        desc = "Experiment 4b: PI Controller Pacing (Full Factorial)"

    print(f"Generated {len(tasks)} tasks")

    config = LocalConfig(max_workers=args.workers)
    runner = DistributedRunner(config)
    results = runner.run(tasks, desc=desc)

    os.makedirs(output_dir, exist_ok=True)
    if args.exp == "1":
        aggregate_exp1_results(results, tasks, output_dir)
    elif args.exp == "2":
        aggregate_exp2_results(results, tasks, output_dir)
    elif args.exp == "3a":
        aggregate_exp3a_results(results, tasks, output_dir)
    elif args.exp == "3b":
        aggregate_exp3b_results(results, tasks, output_dir)
    elif args.exp == "4a":
        aggregate_exp4a_results(results, tasks, output_dir)
    else:
        aggregate_exp4b_results(results, tasks, output_dir)

    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
