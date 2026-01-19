#!/usr/bin/env python3
"""
Unified CLI for running algorithmic collusion experiments.

Supports:
- Sequential execution (default, same as original scripts)
- Local parallel execution (--parallel)
- Cloud execution (--cloud, Phase 2)

Examples:
    # Sequential (default)
    python scripts/run_experiment.py --exp 2 --quick

    # Local parallel
    python scripts/run_experiment.py --exp 2 --parallel --workers 8

    # Local parallel (auto-detect workers)
    python scripts/run_experiment.py --exp 2 --parallel

    # Cloud execution (Phase 2)
    python scripts/run_experiment.py --exp 2 --cloud --project my-project
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cloud.config import LocalConfig, ExperimentConfig
from cloud.runner import DistributedRunner, run_sequential, TaskResult
from cloud.gcs import GCSBucket
from cloud.startup import generate_startup_script
from typing import List, Dict, Any


def _run_exp1_summary_only(**kwargs):
    """Wrapper that returns only summary to reduce memory in parallel mode."""
    from experiments import exp1
    summary, _, _, _ = exp1.run_experiment(**kwargs)
    return summary


def _run_exp2_summary_only(**kwargs):
    """Wrapper that returns only summary to reduce memory in parallel mode."""
    from experiments import exp2
    summary, _, _, _ = exp2.run_experiment(**kwargs)
    return summary


def _run_exp3_summary_only(**kwargs):
    """Wrapper that returns only summary to reduce memory in parallel mode."""
    from experiments import exp3
    summary, _, _, _ = exp3.run_bandit_experiment(**kwargs)
    return summary


def get_exp1_tasks(quick: bool, output_dir: str, seed: int = 42, runs: int = 250) -> List[Dict[str, Any]]:
    """Generate task list for Experiment 1 (Constant Valuations)."""
    import numpy as np

    from experiments import exp1

    if quick:
        K = 5
        alpha_values = [0.01, 0.1]
        gamma_values = [0.9]
        episodes_values = [1000]
        reserve_price_values = [0.0]
        init_values = ["random"]
        exploration_values = ["egreedy"]
        async_values = [0]
        n_bidders_values = [2]
        median_flags = [False]
        winner_flags = [False]
    else:
        K = runs
        alpha_values = [0.001, 0.005, 0.01, 0.05, 0.1]
        gamma_values = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        episodes_values = [100_000]
        reserve_price_values = [0.0, 0.1, 0.2, 0.3, 0.5]
        init_values = ["random", "zeros"]
        exploration_values = ["egreedy", "boltzmann"]
        async_values = [0, 1]
        n_bidders_values = [2, 3, 4]
        median_flags = [False, True]
        winner_flags = [False, True]

    rng = np.random.default_rng(seed)

    tasks = []
    for run_id in range(K):
        alpha = rng.choice(alpha_values)
        gamma = rng.choice(gamma_values)
        episodes = rng.choice(episodes_values)
        reserve_price = rng.choice(reserve_price_values)
        init_str = rng.choice(init_values)
        exploration_str = rng.choice(exploration_values)
        async_val = rng.choice(async_values)
        n_bidders_val = rng.choice(n_bidders_values)
        median_flag = rng.choice(median_flags)
        winner_flag = rng.choice(winner_flags)

        for auction_type in ["first", "second"]:
            task_id = f"exp1_run{run_id}_{auction_type}"
            tasks.append({
                "task_id": task_id,
                "func": _run_exp1_summary_only,
                "kwargs": {
                    "auction_type": auction_type,
                    "alpha": float(alpha),
                    "gamma": float(gamma),
                    "episodes": int(episodes),
                    "init": init_str,
                    "exploration": exploration_str,
                    "asynchronous": int(async_val),
                    "n_bidders": int(n_bidders_val),
                    "median_opp_past_bid_index": bool(median_flag),
                    "winner_bid_index_state": bool(winner_flag),
                    "reserve_price": float(reserve_price),
                    "seed": run_id,
                    "store_qtables": False,
                    "qtable_folder": None,
                },
                "run_id": run_id,
                "config": {
                    "alpha": float(alpha),
                    "gamma": float(gamma),
                    "episodes": int(episodes),
                    "reserve_price": float(reserve_price),
                    "init": init_str,
                    "exploration": exploration_str,
                    "asynchronous": int(async_val),
                    "n_bidders": int(n_bidders_val),
                    "median_flag": bool(median_flag),
                    "winner_flag": bool(winner_flag),
                    "auction_type": auction_type,
                }
            })

    return tasks


def get_exp2_tasks(quick: bool, output_dir: str, seed: int = 42, runs: int = 250) -> List[Dict[str, Any]]:
    """Generate task list for Experiment 2."""
    import numpy as np

    from experiments import exp2

    if quick:
        K = 5
        eta_values = [0.0, 0.5]
        alpha_values = [0.01, 0.1]
        gamma_values = [0.9]
        episodes_values = [1000]
        reserve_price_values = [0.0]
        init_values = ["random"]
        exploration_values = ["egreedy"]
        async_values = [0]
        n_bidders_values = [2]
        median_flags = [False]
        winner_flags = [False]
    else:
        K = runs
        eta_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        alpha_values = [0.001, 0.005, 0.01, 0.05, 0.1]
        gamma_values = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        episodes_values = [100_000]
        reserve_price_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        init_values = ["random", "zeros"]
        exploration_values = ["egreedy", "boltzmann"]
        async_values = [0, 1]
        n_bidders_values = [2, 3, 4]
        median_flags = [False, True]
        winner_flags = [False, True]

    rng = np.random.default_rng(seed)

    tasks = []
    for run_id in range(K):
        eta = rng.choice(eta_values)
        alpha = rng.choice(alpha_values)
        gamma = rng.choice(gamma_values)
        episodes = rng.choice(episodes_values)
        reserve_price = rng.choice(reserve_price_values)
        init_str = rng.choice(init_values)
        exploration_str = rng.choice(exploration_values)
        async_val = rng.choice(async_values)
        n_bidders_val = rng.choice(n_bidders_values)
        median_flag = rng.choice(median_flags)
        winner_flag = rng.choice(winner_flags)

        for auction_type in ["first", "second"]:
            task_id = f"exp2_run{run_id}_{auction_type}"
            tasks.append({
                "task_id": task_id,
                "func": _run_exp2_summary_only,
                "kwargs": {
                    "eta": float(eta),
                    "auction_type": auction_type,
                    "alpha": float(alpha),
                    "gamma": float(gamma),
                    "episodes": int(episodes),
                    "init": init_str,
                    "exploration": exploration_str,
                    "asynchronous": int(async_val),
                    "n_bidders": int(n_bidders_val),
                    "median_opp_past_bid_index": bool(median_flag),
                    "winner_bid_index_state": bool(winner_flag),
                    "reserve_price": float(reserve_price),
                    "seed": run_id,
                    "store_qtables": False,  # Disable for parallel runs to save memory
                    "qtable_folder": None,
                },
                "run_id": run_id,
                "config": {
                    "eta": float(eta),
                    "alpha": float(alpha),
                    "gamma": float(gamma),
                    "episodes": int(episodes),
                    "reserve_price": float(reserve_price),
                    "init": init_str,
                    "exploration": exploration_str,
                    "asynchronous": int(async_val),
                    "n_bidders": int(n_bidders_val),
                    "median_flag": bool(median_flag),
                    "winner_flag": bool(winner_flag),
                    "auction_type": auction_type,
                }
            })

    return tasks


def get_exp3_tasks(quick: bool, output_dir: str, seed: int = 42, runs: int = 250) -> List[Dict[str, Any]]:
    """Generate task list for Experiment 3."""
    import numpy as np

    from experiments import exp3

    if quick:
        K = 5
        eta_values = [0.0, 0.5]
        c_values = [0.1, 1.0]
        lam_values = [1.0]
        n_bidders_values = [2]
        reserve_price_values = [0.0]
        max_rounds_values = [1000]
        use_median_flags = [False]
        use_winner_flags = [False]
    else:
        K = runs
        eta_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        c_values = [0.01, 0.1, 0.5, 1.0, 2.0]
        lam_values = [0.1, 1.0, 5.0]
        n_bidders_values = [2, 3, 4]
        reserve_price_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        max_rounds_values = [100_000]
        use_median_flags = [False, True]
        use_winner_flags = [False, True]

    rng = np.random.default_rng(seed)

    tasks = []
    for run_id in range(K):
        eta = rng.choice(eta_values)
        c = rng.choice(c_values)
        lam = rng.choice(lam_values)
        n_bidders = rng.choice(n_bidders_values)
        r_price = rng.choice(reserve_price_values)
        use_median = rng.choice(use_median_flags)
        use_winner = rng.choice(use_winner_flags)
        max_rounds = rng.choice(max_rounds_values)

        for auction_type in ["first", "second"]:
            task_id = f"exp3_run{run_id}_{auction_type}"
            tasks.append({
                "task_id": task_id,
                "func": _run_exp3_summary_only,
                "kwargs": {
                    "eta": float(eta),
                    "auction_type": auction_type,
                    "c": float(c),
                    "lam": float(lam),
                    "n_bidders": int(n_bidders),
                    "reserve_price": float(r_price),
                    "max_rounds": int(max_rounds),
                    "use_median_of_others": bool(use_median),
                    "use_past_winner_bid": bool(use_winner),
                    "seed": run_id,
                },
                "run_id": run_id,
                "config": {
                    "eta": float(eta),
                    "c": float(c),
                    "lam": float(lam),
                    "n_bidders": int(n_bidders),
                    "reserve_price": float(r_price),
                    "max_rounds": int(max_rounds),
                    "use_median": bool(use_median),
                    "use_winner": bool(use_winner),
                    "auction_type": auction_type,
                }
            })

    return tasks


def aggregate_exp1_results(results: List[TaskResult], tasks: List[Dict], output_dir: str):
    """Aggregate parallel exp1 results into a single data.csv."""
    import pandas as pd
    import json
    from experiments.exp1 import param_mappings, theoretical_revenue_constant_valuation

    os.makedirs(output_dir, exist_ok=True)

    # Save param mappings
    with open(os.path.join(output_dir, "param_mappings.json"), "w") as f:
        json.dump(param_mappings, f, indent=2)

    rows = []
    theory_cache = {}

    for result, task in zip(results, tasks):
        if not result.success:
            print(f"Skipping failed task {result.task_id}: {result.error}")
            continue

        summary = result.result  # wrapper returns summary directly
        config = task["config"]

        # Get theoretical revenue
        cache_key = (config["n_bidders"], config["auction_type"], config["reserve_price"])
        if cache_key not in theory_cache:
            rev_theory = theoretical_revenue_constant_valuation(
                config["n_bidders"], config["auction_type"], config["reserve_price"]
            )
            theory_cache[cache_key] = rev_theory
        else:
            rev_theory = theory_cache[cache_key]

        outcome = dict(summary)
        outcome["run_id"] = task["run_id"]
        outcome["alpha"] = config["alpha"]
        outcome["gamma"] = config["gamma"]
        outcome["episodes"] = config["episodes"]
        outcome["reserve_price"] = config["reserve_price"]
        outcome["auction_type_code"] = param_mappings["auction_type"][config["auction_type"]]
        outcome["init_code"] = param_mappings["init"][config["init"]]
        outcome["exploration_code"] = param_mappings["exploration"][config["exploration"]]
        outcome["asynchronous_code"] = param_mappings["asynchronous"][config["asynchronous"]]
        outcome["n_bidders"] = config["n_bidders"]
        outcome["median_opp_past_bid_index_code"] = param_mappings["median_opp_past_bid_index"][config["median_flag"]]
        outcome["winner_bid_index_state_code"] = param_mappings["winner_bid_index_state"][config["winner_flag"]]
        outcome["theoretical_revenue"] = rev_theory

        if rev_theory > 1e-8:
            outcome["ratio_to_theory"] = outcome["avg_rev_last_1000"] / rev_theory
        else:
            outcome["ratio_to_theory"] = None

        rows.append(outcome)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Data generation complete. Final summary => '{csv_path}'")


def aggregate_exp2_results(results: List[TaskResult], tasks: List[Dict], output_dir: str):
    """Aggregate parallel exp2 results into a single data.csv."""
    import pandas as pd
    import json
    import numpy as np
    from experiments.exp2 import param_mappings, simulate_linear_affiliation_revenue

    os.makedirs(output_dir, exist_ok=True)

    # Save param mappings
    with open(os.path.join(output_dir, "param_mappings.json"), "w") as f:
        json.dump(param_mappings, f, indent=2)

    # Build results dataframe
    rows = []
    theory_cache = {}

    for result, task in zip(results, tasks):
        if not result.success:
            print(f"Skipping failed task {result.task_id}: {result.error}")
            continue

        summary = result.result  # wrapper returns summary directly
        config = task["config"]

        # Get theoretical revenue
        cache_key = (config["n_bidders"], config["eta"], config["auction_type"])
        if cache_key not in theory_cache:
            rev_theory = simulate_linear_affiliation_revenue(
                config["n_bidders"], config["eta"], config["auction_type"]
            )
            theory_cache[cache_key] = rev_theory
        else:
            rev_theory = theory_cache[cache_key]

        outcome = dict(summary)
        outcome["run_id"] = task["run_id"]
        outcome["eta"] = config["eta"]
        outcome["alpha"] = config["alpha"]
        outcome["gamma"] = config["gamma"]
        outcome["episodes"] = config["episodes"]
        outcome["reserve_price"] = config["reserve_price"]
        outcome["auction_type_code"] = param_mappings["auction_type"][config["auction_type"]]
        outcome["init_code"] = param_mappings["init"][config["init"]]
        outcome["exploration_code"] = param_mappings["exploration"][config["exploration"]]
        outcome["asynchronous_code"] = param_mappings["asynchronous"][config["asynchronous"]]
        outcome["n_bidders"] = config["n_bidders"]
        outcome["median_opp_past_bid_index_code"] = param_mappings["median_opp_past_bid_index"][config["median_flag"]]
        outcome["winner_bid_index_state_code"] = param_mappings["winner_bid_index_state"][config["winner_flag"]]
        outcome["theoretical_revenue"] = rev_theory

        if rev_theory > 1e-8:
            outcome["ratio_to_theory"] = outcome["avg_rev_last_1000"] / rev_theory
        else:
            outcome["ratio_to_theory"] = None

        rows.append(outcome)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Data generation complete. Final summary => '{csv_path}'")


def aggregate_exp3_results(results: List[TaskResult], tasks: List[Dict], output_dir: str):
    """Aggregate parallel exp3 results into a single data.csv."""
    import pandas as pd
    import json
    from experiments.exp3 import param_mappings, simulate_linear_affiliation_revenue

    os.makedirs(output_dir, exist_ok=True)

    # Save param mappings
    with open(os.path.join(output_dir, "param_mappings.json"), "w") as f:
        json.dump(param_mappings, f, indent=2)

    rows = []
    theory_cache = {}

    for result, task in zip(results, tasks):
        if not result.success:
            print(f"Skipping failed task {result.task_id}: {result.error}")
            continue

        summary = result.result  # wrapper returns summary directly
        config = task["config"]

        # Get theoretical revenue
        cache_key = (config["n_bidders"], config["eta"], config["auction_type"])
        if cache_key not in theory_cache:
            rev_theory = simulate_linear_affiliation_revenue(
                config["n_bidders"], config["eta"], config["auction_type"]
            )
            theory_cache[cache_key] = rev_theory
        else:
            rev_theory = theory_cache[cache_key]

        outcome = dict(summary)
        outcome["run_id"] = task["run_id"]
        outcome["eta"] = config["eta"]
        outcome["c"] = config["c"]
        outcome["lam"] = config["lam"]
        outcome["n_bidders"] = config["n_bidders"]
        outcome["auction_type_code"] = param_mappings["auction_type"][config["auction_type"]]
        outcome["reserve_price"] = config["reserve_price"]
        outcome["max_rounds"] = config["max_rounds"]
        outcome["use_median_of_others_code"] = param_mappings["use_median_of_others"][config["use_median"]]
        outcome["use_past_winner_bid_code"] = param_mappings["use_past_winner_bid"][config["use_winner"]]
        outcome["theoretical_revenue"] = rev_theory

        if rev_theory > 1e-8:
            outcome["ratio_to_theory"] = outcome["avg_rev_last_1000"] / rev_theory
        else:
            outcome["ratio_to_theory"] = None

        rows.append(outcome)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Data generation complete. Final summary => '{csv_path}'")


def run_exp1_sequential(quick: bool, output_dir: str):
    """Run exp1 sequentially."""
    import subprocess
    cmd = [sys.executable, "src/experiments/exp1.py"]
    if quick:
        cmd.append("--quick")
    subprocess.run(cmd, check=True)


def run_detached(args):
    """Launch experiment on cloud VM in fire-and-forget mode."""
    import subprocess
    from cloud.vm import CloudVM, VMConfig
    from cloud.gcs import GCSBucket
    from cloud.startup import generate_startup_script

    # Get project ID
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

    # Setup GCS bucket
    bucket_name = f"{project_id}-collusion-experiments"
    bucket = GCSBucket(bucket_name, project_id)
    bucket.create()

    # Upload code
    print("Uploading code to GCS...")
    bucket.upload_code()

    # Generate startup script
    startup_script = generate_startup_script(
        bucket_name=bucket_name,
        exp=args.exp,
        quick=args.quick,
        parallel=True
    )

    # Create VM with startup script
    vm_config = VMConfig(
        name=f"collusion-exp{args.exp}",
        machine_type="n2-standard-4",
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
    print(f"\nThe VM will:")
    print(f"  1. Install dependencies")
    print(f"  2. Run experiment {args.exp}")
    print(f"  3. Upload results to gs://{bucket_name}/results/exp{args.exp}/")
    print(f"  4. Upload logs to gs://{bucket_name}/logs/")
    print(f"  5. Self-delete")
    print(f"\nYou can close this terminal now.")
    print(f"\nTo check progress:")
    print(f"  gcloud compute instances list --filter='name~collusion-exp'")
    print(f"  gsutil ls gs://{bucket_name}/results/")
    print(f"  gsutil ls gs://{bucket_name}/logs/")


def run_cloud(args):
    """Run experiment on Google Cloud."""
    from cloud.vm import CloudVM, VMConfig

    # Get project ID
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

    # Determine workers based on machine type
    workers = args.workers or 4

    # Create VM manager
    vm_config = VMConfig(
        name=f"collusion-exp{args.exp}",
        machine_type="n2-standard-4",  # 4 vCPUs
    )
    vm = CloudVM(project_id, vm_config)

    try:
        # Create VM if needed
        if not vm.exists():
            vm.create()
            vm.setup_environment()
            vm.upload_code()
            vm.install_requirements()
        else:
            print(f"VM {vm_config.name} already exists, reusing...")
            # Still upload latest code
            vm.upload_code()

        # Run experiment
        print(f"\n{'='*50}")
        print(f"Running Experiment {args.exp} on cloud VM...")
        print(f"{'='*50}\n")

        remote_results = vm.run_experiment(
            exp=args.exp,
            quick=args.quick,
            parallel=True,
            workers=workers,
            runs=args.runs
        )

        # Download results
        suffix = "/quick_test" if args.quick else ""
        local_output = args.output_dir or f"results/exp{args.exp}{suffix}_cloud"
        vm.download_results(remote_results, local_output)

        print(f"\n{'='*50}")
        print(f"Results downloaded to: {local_output}")
        print(f"{'='*50}")

        # Ask about VM cleanup
        response = input("\nDelete cloud VM? [y/N]: ").strip().lower()
        if response == 'y':
            vm.delete()

    except KeyboardInterrupt:
        print("\nInterrupted. VM is still running.")
        print(f"To delete: gcloud compute instances delete {vm_config.name} --zone={vm_config.zone} --project={project_id}")
    except Exception as e:
        print(f"\nError: {e}")
        print(f"VM may still be running: {vm_config.name}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Unified CLI for algorithmic collusion experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--exp", type=int, required=True, choices=[1, 2, 3],
        help="Experiment number (1, 2, or 3)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick test with reduced parameters"
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Run in parallel using multiprocessing"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: cpu_count - 1)"
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
        "--runs", type=int, default=250,
        help="Number of runs (default: 250, ignored with --quick which uses 5)"
    )

    # Cloud options (Phase 2)
    parser.add_argument(
        "--cloud", action="store_true",
        help="Run on Google Cloud (not yet implemented)"
    )
    parser.add_argument(
        "--detached", action="store_true",
        help="Fire-and-forget cloud mode: VM auto-runs, uploads to GCS, self-deletes"
    )
    parser.add_argument(
        "--project", type=str,
        help="GCP project ID (for --cloud mode)"
    )
    parser.add_argument(
        "--monitor", action="store_true",
        help="Monitor running cloud job (not yet implemented)"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download results from cloud (not yet implemented)"
    )

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

    # Sequential mode (default, maintains backward compatibility)
    if not args.parallel:
        print(f"Running Experiment {args.exp} sequentially...")
        if args.exp == 1:
            run_exp1_sequential(args.quick, output_dir)
        elif args.exp == 2:
            import subprocess
            cmd = [sys.executable, "src/experiments/exp2.py"]
            if args.quick:
                cmd.append("--quick")
            subprocess.run(cmd, check=True)
        else:  # exp == 3
            import subprocess
            cmd = [sys.executable, "src/experiments/exp3.py"]
            if args.quick:
                cmd.append("--quick")
            subprocess.run(cmd, check=True)
        return

    # Parallel mode
    print(f"Running Experiment {args.exp} in parallel...")

    # Get tasks based on experiment
    if args.exp == 1:
        tasks = get_exp1_tasks(args.quick, output_dir, args.seed, args.runs)
        desc = "Experiment 1: Identical Values"
    elif args.exp == 2:
        tasks = get_exp2_tasks(args.quick, output_dir, args.seed, args.runs)
        desc = "Experiment 2: Affiliated Values"
    else:  # exp == 3
        tasks = get_exp3_tasks(args.quick, output_dir, args.seed, args.runs)
        desc = "Experiment 3: LinUCB Bandits"

    print(f"Generated {len(tasks)} tasks")

    # Create runner
    config = LocalConfig(max_workers=args.workers)
    runner = DistributedRunner(config)

    # Run tasks
    results = runner.run(tasks, desc=desc)

    # Aggregate results
    os.makedirs(output_dir, exist_ok=True)

    if args.exp == 1:
        aggregate_exp1_results(results, tasks, output_dir)
    elif args.exp == 2:
        aggregate_exp2_results(results, tasks, output_dir)
    else:  # exp == 3
        aggregate_exp3_results(results, tasks, output_dir)

    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
