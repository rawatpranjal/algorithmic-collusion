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
import itertools
import json
import os
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
    "auction_type":              ("second", "first"),
    "alpha":                     (0.01, 0.1),
    "gamma":                     (0.0, 0.95),
    "reserve_price":             (0.0, 0.5),
    "init":                      ("zeros", "random"),
    "exploration":               ("egreedy", "boltzmann"),
    "asynchronous":              (0, 1),
    "n_bidders":                 (2, 4),
    "median_opp_past_bid_index": (False, True),
    "winner_bid_index_state":    (False, True),
}

EXP2_FACTORS = {
    # Same as Exp1 + eta
    "auction_type":              ("second", "first"),
    "alpha":                     (0.01, 0.1),
    "gamma":                     (0.0, 0.95),
    "reserve_price":             (0.0, 0.5),
    "init":                      ("zeros", "random"),
    "exploration":               ("egreedy", "boltzmann"),
    "asynchronous":              (0, 1),
    "n_bidders":                 (2, 4),
    "median_opp_past_bid_index": (False, True),
    "winner_bid_index_state":    (False, True),
    "eta":                       (0.0, 1.0),
}

EXP3_FACTORS = {
    "auction_type":       ("second", "first"),
    "eta":                (0.0, 1.0),
    "c":                  (0.1, 2.0),
    "lam":                (0.1, 5.0),
    "n_bidders":          (2, 4),
    "reserve_price":      (0.0, 0.5),
    "use_median_of_others": (False, True),
    "use_past_winner_bid":  (False, True),
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


# =====================================================================
# Task Generators (factorial design)
# =====================================================================

def get_exp1_tasks(quick, output_dir, seed=42, replicates=2):
    """Generate task list for Experiment 1 using factorial design."""
    if quick:
        design = build_quick_design(EXP1_FACTORS, seed)
    else:
        design = build_factorial_design(EXP1_FACTORS, replicates, seed)

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
                "median_opp_past_bid_index": bool(row["median_opp_past_bid_index"]),
                "winner_bid_index_state": bool(row["winner_bid_index_state"]),
                "reserve_price": float(row["reserve_price"]),
                "seed": row["seed"],
                "store_qtables": False,
                "qtable_folder": None,
            },
            "run_id": row["cell_id"],
            "design_row": row,
        })

    return tasks


def get_exp2_tasks(quick, output_dir, seed=42, replicates=2):
    """Generate task list for Experiment 2 using half-fraction factorial."""
    if quick:
        design = build_quick_design(EXP2_FACTORS, seed)
    else:
        # 2^(11-1) = 1024 cells, Resolution V half-fraction
        design = build_factorial_design(EXP2_FACTORS, replicates, seed,
                                        half_fraction=True)

    tasks = []
    for row in design:
        task_id = f"exp2_cell{row['cell_id']}_rep{row['replicate']}"
        tasks.append({
            "task_id": task_id,
            "func": _run_exp2_summary_only,
            "kwargs": {
                "eta": float(row["eta"]),
                "auction_type": row["auction_type"],
                "alpha": float(row["alpha"]),
                "gamma": float(row["gamma"]),
                "episodes": 1000 if quick else 100_000,
                "init": row["init"],
                "exploration": row["exploration"],
                "asynchronous": int(row["asynchronous"]),
                "n_bidders": int(row["n_bidders"]),
                "median_opp_past_bid_index": bool(row["median_opp_past_bid_index"]),
                "winner_bid_index_state": bool(row["winner_bid_index_state"]),
                "reserve_price": float(row["reserve_price"]),
                "seed": row["seed"],
                "store_qtables": False,
                "qtable_folder": None,
            },
            "run_id": row["cell_id"],
            "design_row": row,
        })

    return tasks


def get_exp3_tasks(quick, output_dir, seed=42, replicates=2):
    """Generate task list for Experiment 3 using full factorial."""
    if quick:
        design = build_quick_design(EXP3_FACTORS, seed)
    else:
        design = build_factorial_design(EXP3_FACTORS, replicates, seed)

    tasks = []
    for row in design:
        task_id = f"exp3_cell{row['cell_id']}_rep{row['replicate']}"
        tasks.append({
            "task_id": task_id,
            "func": _run_exp3_summary_only,
            "kwargs": {
                "eta": float(row["eta"]),
                "auction_type": row["auction_type"],
                "c": float(row["c"]),
                "lam": float(row["lam"]),
                "n_bidders": int(row["n_bidders"]),
                "reserve_price": float(row["reserve_price"]),
                "max_rounds": 1000 if quick else 100_000,
                "use_median_of_others": bool(row["use_median_of_others"]),
                "use_past_winner_bid": bool(row["use_past_winner_bid"]),
                "seed": row["seed"],
            },
            "run_id": row["cell_id"],
            "design_row": row,
        })

    return tasks


# =====================================================================
# Aggregation Functions
# =====================================================================

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

        cache_key = (int(design["n_bidders"]), design["auction_type"],
                     float(design["reserve_price"]))
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
    print(f"  Design: 2^{len(EXP1_FACTORS)} = {2**len(EXP1_FACTORS)} cells")


def aggregate_exp2_results(results, tasks, output_dir):
    """Aggregate parallel exp2 results into data.csv with coded columns."""
    import pandas as pd
    from experiments.exp2 import param_mappings, simulate_linear_affiliation_revenue

    os.makedirs(output_dir, exist_ok=True)
    _save_design_info(EXP2_FACTORS, output_dir, 2)

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
        for fname in EXP2_FACTORS:
            outcome[fname] = design[fname]
        for fname in EXP2_FACTORS:
            outcome[fname + "_coded"] = design[fname + "_coded"]
        outcome["cell_id"] = design["cell_id"]
        outcome["replicate"] = design["replicate"]
        outcome["seed"] = design["seed"]
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
    print(f"  Design: 2^(11-1) = 1024 cells (Resolution V half-fraction)")


def aggregate_exp3_results(results, tasks, output_dir):
    """Aggregate parallel exp3 results into data.csv with coded columns."""
    import pandas as pd
    from experiments.exp3 import param_mappings, simulate_linear_affiliation_revenue

    os.makedirs(output_dir, exist_ok=True)
    _save_design_info(EXP3_FACTORS, output_dir, 3)

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
        for fname in EXP3_FACTORS:
            outcome[fname] = design[fname]
        for fname in EXP3_FACTORS:
            outcome[fname + "_coded"] = design[fname + "_coded"]
        outcome["cell_id"] = design["cell_id"]
        outcome["replicate"] = design["replicate"]
        outcome["seed"] = design["seed"]
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
    print(f"  Design: 2^{len(EXP3_FACTORS)} = {2**len(EXP3_FACTORS)} cells")


# =====================================================================
# Sequential Mode
# =====================================================================

def run_exp_sequential(exp, quick, output_dir):
    """Run experiment sequentially using subprocess."""
    import subprocess
    if exp == 1:
        cmd = [sys.executable, "src/experiments/exp1.py"]
    elif exp == 2:
        cmd = [sys.executable, "src/experiments/exp2.py"]
    else:
        cmd = [sys.executable, "src/experiments/exp3.py"]
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
        "--exp", type=int, required=True, choices=[1, 2, 3],
        help="Experiment number (1, 2, or 3)"
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

    # Cloud options
    parser.add_argument("--cloud", action="store_true")
    parser.add_argument("--detached", action="store_true")
    parser.add_argument("--project", type=str)
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--download", action="store_true")

    args = parser.parse_args()

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
    factor_defs = {1: EXP1_FACTORS, 2: EXP2_FACTORS, 3: EXP3_FACTORS}[args.exp]
    k = len(factor_defs)
    if args.quick:
        n_cells = 16
        n_reps = 1
        total = 16
    else:
        if args.exp == 2:
            n_cells = 2 ** (k - 1)
            design_desc = f"2^({k}-1) = {n_cells} (Resolution V half-fraction)"
        else:
            n_cells = 2 ** k
            design_desc = f"2^{k} = {n_cells} (full factorial)"
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

    if args.exp == 1:
        tasks = get_exp1_tasks(args.quick, output_dir, args.seed, args.replicates)
        desc = "Experiment 1: Constant Values (Factorial)"
    elif args.exp == 2:
        tasks = get_exp2_tasks(args.quick, output_dir, args.seed, args.replicates)
        desc = "Experiment 2: Affiliated Values (Factorial)"
    else:
        tasks = get_exp3_tasks(args.quick, output_dir, args.seed, args.replicates)
        desc = "Experiment 3: LinUCB Bandits (Factorial)"

    print(f"Generated {len(tasks)} tasks")

    config = LocalConfig(max_workers=args.workers)
    runner = DistributedRunner(config)
    results = runner.run(tasks, desc=desc)

    os.makedirs(output_dir, exist_ok=True)
    if args.exp == 1:
        aggregate_exp1_results(results, tasks, output_dir)
    elif args.exp == 2:
        aggregate_exp2_results(results, tasks, output_dir)
    else:
        aggregate_exp3_results(results, tasks, output_dir)

    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
