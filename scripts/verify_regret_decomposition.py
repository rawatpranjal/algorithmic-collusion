#!/usr/bin/env Rscript
"""
Verification of the Regret Decomposition Identity.

This script runs a single simulation from Experiment 2 and verifies that
the components of the seller's regret decomposition sum correctly.

The decomposition is:
  Total Regret = Valuation Shortfall + Equilibrium Gap + Learning Gap
  1 - R        = (1 - E[v₁])         + (E[v₁] - R_BNE) + (R_BNE - R)

The script executes a Q-learning simulation in an affiliated-values auction
environment, retrieves the calculated values for each term, and confirms
the identity holds.

To run:
  PYTHONPATH=src python3 scripts/verify_regret_decomposition.py
"""

import sys
import os
import numpy as np

# This script assumes that the 'src' directory is in the Python path.
# You can set it by running: export PYTHONPATH=$PYTHONPATH:$(pwd)/src
# from the project root, or by running the script with `PYTHONPATH=src ...`

try:
    from experiments.exp2 import run_experiment
except ImportError as e:
    print(f"Error: {e}")
    print("Could not import 'run_experiment' from 'experiments.exp2'.")
    print("Please run this script from the project root with `PYTHONPATH=src`.")
    print("Example: PYTHONPATH=src python3 scripts/verify_regret_decomposition.py")
    sys.exit(1)


def main():
    """
    Main function to run the verification.
    """
    print("=" * 70)
    print("Verifying Regret Decomposition Identity")
    print("=" * 70)

    # --- 1. Define Simulation Parameters ---
    # Using a moderately complex setting for a meaningful test.
    params = {
        "eta": 0.5,
        "auction_type": "first",
        "n_bidders": 4,
        "state_info": "signal_winner",
        "episodes": 20_000,
        "seed": 42,
    }
    print("Running simulation with the following parameters:")
    for key, value in params.items():
        print(f"  - {key}: {value}")
    print("-" * 70)

    # --- 2. Run the Experiment ---
    print("Starting simulation...")
    summary, _, _, _ = run_experiment(**params)
    print("Simulation complete. Analyzing results...")
    print("-" * 70)

    # --- 3. Extract Decomposition Components ---
    total_regret = summary.get("raw_regret")
    valuation_shortfall = summary.get("structural_gap")
    equilibrium_gap = summary.get("shading_gap")
    learning_gap = summary.get("excess_gap")

    if any(v is None for v in [total_regret, valuation_shortfall, equilibrium_gap, learning_gap]):
        print("Error: One or more required metrics not found in simulation summary.")
        print("Summary dictionary contents:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return

    # --- 4. Verify the Identity ---
    print("Extracted Components of Regret:")
    print(f"  - Total Observed Regret (1 - R)          : {total_regret:.6f}")
    print(f"  - Valuation Shortfall (1 - E[v₁])        : {valuation_shortfall:.6f}")
    print(f"  - Equilibrium Gap (E[v₁] - R_BNE)        : {equilibrium_gap:.6f}")
    print(f"  - Learning Gap (R_BNE - R)               : {learning_gap:.6f}")
    print("-" * 70)

    sum_of_gaps = valuation_shortfall + equilibrium_gap + learning_gap
    discrepancy = abs(sum_of_gaps - total_regret)

    print("Verification Check:")
    print(f"  - Sum of Gaps   : {sum_of_gaps:.6f}")
    print(f"  - Total Regret  : {total_regret:.6f}")
    print(f"  - Discrepancy   : {discrepancy:.6e}")
    print("-" * 70)

    # --- 5. Conclusion ---
    if discrepancy < 1e-9:
        print("SUCCESS: The regret decomposition identity holds.")
        print("         Total Regret = Valuation Shortfall + Equilibrium Gap + Learning Gap")
    else:
        print("FAILURE: The regret decomposition identity does NOT hold.")
        print(f"          Discrepancy of {discrepancy:.6e} is larger than tolerance (1e-9).")

    print("=" * 70)


if __name__ == "__main__":
    main()
