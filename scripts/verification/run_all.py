#!/usr/bin/env python3
"""
Independent Mathematical Verification Suite

Runs 15 tests verifying every mathematical claim in the paper:
- Tests 1-6:  Nash/BNE equilibria (test_equilibria.py)
- Tests 7-9:  LP optimality, PoA bounds (test_welfare.py)
- Tests 10-12: Bid optimality, budget feasibility (test_pacing.py)
- Tests 13-15: Valuation model properties (test_model.py)

Zero imports from project source code. All formulas re-derived from paper LaTeX.

Usage:
    python3 scripts/verification/run_all.py           # full suite (~3-5 min)
    python3 scripts/verification/run_all.py --quick    # reduced MC (~30s)
    python3 scripts/verification/run_all.py -v         # verbose output
"""

import argparse
import json
import os
import sys
import time

# Add this directory to path so test modules can import helpers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_equilibria import run_all as run_equilibria
from test_welfare import run_all as run_welfare
from test_pacing import run_all as run_pacing
from test_model import run_all as run_model


def main():
    parser = argparse.ArgumentParser(
        description="Independent Mathematical Verification Suite"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Reduce MC samples (algebraic tests unchanged)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output per test")
    parser.add_argument("--json", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    print("=" * 70)
    print("INDEPENDENT MATHEMATICAL VERIFICATION SUITE")
    print("=" * 70)
    print(f"Mode: {'quick' if args.quick else 'full'}")
    print()

    t_start = time.time()
    all_results = []

    # Module 1: Equilibria (Tests 1-6)
    print("-" * 70)
    print("MODULE: Equilibria (Tests 1-6)")
    print("-" * 70)
    equilibria_results = run_equilibria(verbose=args.verbose, quick=args.quick)
    all_results.extend(equilibria_results)
    print()

    # Module 2: Welfare (Tests 7-9)
    print("-" * 70)
    print("MODULE: Welfare & PoA (Tests 7-9)")
    print("-" * 70)
    welfare_results = run_welfare(verbose=args.verbose, quick=args.quick)
    all_results.extend(welfare_results)
    print()

    # Module 3: Pacing (Tests 10-12)
    print("-" * 70)
    print("MODULE: Pacing (Tests 10-12)")
    print("-" * 70)
    pacing_results = run_pacing(verbose=args.verbose, quick=args.quick)
    all_results.extend(pacing_results)
    print()

    # Module 4: Model (Tests 13-15)
    print("-" * 70)
    print("MODULE: Model Properties (Tests 13-15)")
    print("-" * 70)
    model_results = run_model(verbose=args.verbose, quick=args.quick)
    all_results.extend(model_results)
    print()

    # Summary
    elapsed = time.time() - t_start
    n_pass = sum(1 for r in all_results if r["passed"])
    n_total = len(all_results)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    for r in all_results:
        status = "PASS" if r["passed"] else "FAIL"
        method = r.get("method", "")
        print(f"  [{status}] Test {r['test']:2d}: {r['name']}")
        print(f"           Method: {method}")

    print()
    print(f"  {n_pass}/{n_total} tests passed in {elapsed:.1f}s")
    print()

    if n_pass == n_total:
        print("  ALL TESTS PASSED")
    else:
        failed = [r for r in all_results if not r["passed"]]
        print("  FAILURES:")
        for r in failed:
            print(f"    Test {r['test']}: {r['name']}")

    # Save JSON if requested
    if args.json:
        # Make results JSON-serializable
        def make_serializable(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        import numpy as np

        json_results = json.loads(
            json.dumps(all_results, default=make_serializable)
        )
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({
                "mode": "quick" if args.quick else "full",
                "elapsed_seconds": elapsed,
                "n_pass": n_pass,
                "n_total": n_total,
                "all_passed": n_pass == n_total,
                "results": json_results,
            }, f, indent=2)
        print(f"  Results saved to {args.json}")

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
