#!/usr/bin/env python3
"""
Sequential runner for Experiment 2 with a custom bid grid size.

Generates the 3 x 2^3 = 24-cell mixed-level design and runs each cell
sequentially using experiments.exp2.run_experiment with the specified
number of bid actions. Writes a single CSV of summaries including the
grid-adjusted benchmark and tie-at-top rate.

Usage examples:
  python3 scripts/run_exp2_grid.py --n-bid-actions 101 --episodes 100000
  python3 scripts/run_exp2_grid.py --n-bid-actions 101 --episodes 1000   # quick
"""

import argparse
import os
import json
import pandas as pd

import sys
sys.path.insert(0, os.path.abspath(""))
from scripts.run_experiment import build_exp2_design, _save_exp2_design_info
from experiments.exp2 import run_experiment, param_mappings


def main():
    ap = argparse.ArgumentParser(description="Sequential Exp2 runner with custom bid grid")
    ap.add_argument("--episodes", type=int, default=100_000)
    ap.add_argument("--n-bid-actions", type=int, default=101)
    ap.add_argument("--replicates", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()

    design = build_exp2_design(replicates=args.replicates, seed=args.seed)

    out_dir = args.output_dir or f"results/exp2/grid{args.n_bid_actions}_{args.episodes}"
    os.makedirs(out_dir, exist_ok=True)
    _save_exp2_design_info(out_dir)
    with open(os.path.join(out_dir, "param_mappings.json"), "w") as f:
        json.dump(param_mappings, f, indent=2)

    rows = []
    for row in design:
        summary, _, _, _ = run_experiment(
            eta=float(row["eta"]),
            auction_type=row["auction_type"],
            n_bidders=int(row["n_bidders"]),
            state_info=row["state_info"],
            episodes=args.episodes,
            seed=int(row["seed"]),
            n_bid_actions=args.n_bid_actions,
            store_qtables=False,
            qtable_folder=None,
        )

        out = dict(summary)
        out["auction_type"] = row["auction_type"]
        out["n_bidders"] = int(row["n_bidders"])
        out["state_info"] = row["state_info"]
        out["eta"] = float(row["eta"])
        out["auction_type_coded"] = row["auction_type_coded"]
        out["n_bidders_coded"] = row["n_bidders_coded"]
        out["state_info_coded"] = row["state_info_coded"]
        out["eta_linear_coded"] = row["eta_linear_coded"]
        out["eta_quadratic_coded"] = row["eta_quadratic_coded"]
        out["cell_id"] = row["cell_id"]
        out["replicate"] = row["replicate"]
        out["seed"] = int(row["seed"])
        out["auction_type_code"] = param_mappings["auction_type"][row["auction_type"]]

        R_bne = out.get("theoretical_revenue", 0)
        out["ratio_to_theory"] = (out["avg_rev_last_1000"] / R_bne) if R_bne else None

        rows.append(out)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} rows to {csv_path}")


if __name__ == "__main__":
    main()
