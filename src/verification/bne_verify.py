#!/usr/bin/env python3
"""
BNE Verification for the Affiliated-Values Auction Model.

Verifies:
1. No profitable unilateral deviation from BNE bid functions (101-point grid)
2. Analytical revenue formulas match Monte Carlo simulation
3. Revenue equivalence: FPA = SPA revenue under iid signals

Usage:
    PYTHONPATH=src python3 src/verification/bne_verify.py          # full
    PYTHONPATH=src python3 src/verification/bne_verify.py --quick   # quick test
"""

import argparse
import json
import os
import time

import numpy as np


# ---------------------------------------------------------------------------
# BNE bid coefficients
# ---------------------------------------------------------------------------

def compute_alpha_beta(eta, N):
    """Return (alpha, beta_per_opponent) for the linear affiliation model."""
    alpha = 1.0 - 0.5 * eta
    beta = 0.5 * eta / max(N - 1, 1)
    return alpha, beta


def compute_bne_bid_coefficient(eta, N, auction_type):
    """
    Return phi where b(s) = phi * s is the symmetric BNE.

    SPA: phi = alpha + N*beta/2
    FPA: phi = (N-1)/N * (alpha + N*beta/2)
    """
    alpha, beta = compute_alpha_beta(eta, N)
    base = alpha + N * beta / 2.0
    if auction_type == "first":
        return (N - 1) / N * base
    else:
        return base


# ---------------------------------------------------------------------------
# Analytical revenue
# ---------------------------------------------------------------------------

def analytical_revenue(eta, N):
    """
    Closed-form expected revenue per round under the BNE.

    Both FPA and SPA yield the same revenue (revenue equivalence with iid signals):
        R = (N-1)/(N+1) * (alpha + N*beta/2)
    """
    alpha, beta = compute_alpha_beta(eta, N)
    return (N - 1) / (N + 1) * (alpha + N * beta / 2.0)


# ---------------------------------------------------------------------------
# Deviation check (vectorized Monte Carlo)
# ---------------------------------------------------------------------------

def check_no_profitable_deviation(eta, N, auction_type, n_grid=101, M=200_000,
                                  signal_values=None, seed=42):
    """
    For each test signal s_i, compute expected payoff for every bid on a fine grid.
    Opponents play BNE: b_j = phi * s_j, with s_j ~ U[0,1] iid.

    Returns dict with:
        - max_gain: largest payoff gain from deviating across all signals
        - details: list of per-signal results
    """
    rng = np.random.default_rng(seed)
    phi = compute_bne_bid_coefficient(eta, N, auction_type)
    alpha, beta = compute_alpha_beta(eta, N)

    if signal_values is None:
        signal_values = np.linspace(0.1, 0.9, 9)

    bid_grid = np.linspace(0, 1, n_grid)  # shape (G,)

    # Draw opponent signals: (M, N-1)
    opp_signals = rng.uniform(0, 1, size=(M, N - 1))
    # Opponent bids under BNE
    opp_bids = phi * opp_signals  # (M, N-1)
    # Max opponent bid per draw
    opp_max = opp_bids.max(axis=1)  # (M,)

    details = []
    overall_max_gain = -np.inf

    for s_i in signal_values:
        # Valuation of bidder i given signal s_i and opponent signals
        # v_i = alpha * s_i + beta * sum(opp_signals per draw)
        opp_sum = opp_signals.sum(axis=1)  # (M,)
        v_i = alpha * s_i + beta * opp_sum  # (M,)

        # BNE bid for this signal
        bne_bid = phi * s_i

        # Expected payoff for each bid on the grid
        # win_matrix[g, m] = bid_grid[g] > opp_max[m] (strict) or tie-break
        # For continuous distributions, ties have measure zero; use >=
        win_matrix = bid_grid[:, None] >= opp_max[None, :]  # (G, M)

        if auction_type == "first":
            # payoff = (v_i - bid) * I(win)
            payoff_if_win = v_i[None, :] - bid_grid[:, None]  # (G, M)
        else:
            # payoff = (v_i - second_price) * I(win)
            # second_price = opp_max when bidder i wins
            payoff_if_win = v_i[None, :] - opp_max[None, :]  # (G, M)

        # Expected payoff per bid
        expected_payoff = (payoff_if_win * win_matrix).mean(axis=1)  # (G,)

        # BNE payoff
        bne_idx = np.argmin(np.abs(bid_grid - bne_bid))
        bne_payoff = expected_payoff[bne_idx]

        # Best alternative
        best_idx = np.argmax(expected_payoff)
        best_payoff = expected_payoff[best_idx]
        best_bid = bid_grid[best_idx]

        gain = best_payoff - bne_payoff

        details.append({
            "signal": float(s_i),
            "bne_bid": float(bne_bid),
            "bne_payoff": float(bne_payoff),
            "best_bid": float(best_bid),
            "best_payoff": float(best_payoff),
            "gain": float(gain),
        })

        if gain > overall_max_gain:
            overall_max_gain = gain

    return {
        "eta": eta,
        "N": N,
        "auction_type": auction_type,
        "phi": float(phi),
        "max_gain": float(overall_max_gain),
        "details": details,
    }


# ---------------------------------------------------------------------------
# Monte Carlo revenue
# ---------------------------------------------------------------------------

def mc_revenue(eta, N, auction_type, M=500_000, seed=123):
    """
    Simulate N BNE bidders, measure average seller revenue.
    """
    rng = np.random.default_rng(seed)
    phi = compute_bne_bid_coefficient(eta, N, auction_type)

    signals = rng.uniform(0, 1, size=(M, N))  # (M, N)
    bids = phi * signals  # (M, N)

    if auction_type == "first":
        # Revenue = max bid
        revenue = bids.max(axis=1)  # (M,)
    else:
        # Revenue = second-highest bid
        sorted_bids = np.sort(bids, axis=1)  # ascending
        revenue = sorted_bids[:, -2]  # second-highest

    return float(revenue.mean()), float(revenue.std() / np.sqrt(M))


# ---------------------------------------------------------------------------
# MC Benchmarks for Distributional Metrics under BNE
# ---------------------------------------------------------------------------
def grid_adjusted_bne_revenue(eta, N, auction_type, n_bid_actions=21, M=200_000, seed=7):
    """
    Monte Carlo benchmark for expected revenue under BNE when bids are
    discretized to a fixed grid of n_bid_actions in [0,1]. Returns
    (rev_mean, rev_se, tie_top_rate) where tie_top_rate is the fraction
    of rounds in which the highest and second-highest snapped bids are equal.
    """
    rng = np.random.default_rng(seed)
    phi = compute_bne_bid_coefficient(eta, N, auction_type)

    # Signals and continuous BNE bids
    signals = rng.uniform(0, 1, size=(M, N))
    cont_bids = phi * signals

    # Snap to nearest grid point in [0,1]
    k = max(n_bid_actions - 1, 1)
    snapped = np.round(cont_bids * k) / k

    # Revenue and tie-at-top indicator
    if auction_type == "first":
        revenue = snapped.max(axis=1)
        # For FPA, top tie does not change price vs max, but report anyway
        sorted_snapped = np.sort(snapped, axis=1)
        tie_top = (sorted_snapped[:, -1] == sorted_snapped[:, -2])
    else:
        sorted_snapped = np.sort(snapped, axis=1)
        revenue = sorted_snapped[:, -2]
        tie_top = (sorted_snapped[:, -1] == sorted_snapped[:, -2])

    rev_mean = float(revenue.mean())
    rev_se = float(revenue.std() / np.sqrt(M))
    tie_top_rate = float(tie_top.mean())
    return rev_mean, rev_se, tie_top_rate

def bne_btv_benchmark(eta, N, auction_type, M=100_000, seed=42):
    """MC benchmark for bid-to-value ratio under BNE. Returns (median, iqr)."""
    rng = np.random.default_rng(seed)
    phi = compute_bne_bid_coefficient(eta, N, auction_type)
    alpha, beta = compute_alpha_beta(eta, N)

    signals = rng.uniform(0, 1, size=(M, N))
    bids = phi * signals

    # Winner is highest bidder each round
    winner_idx = bids.argmax(axis=1)
    rows = np.arange(M)

    # Winner's valuation: v_i = alpha*s_i + beta*mean(others)
    winner_signals = signals[rows, winner_idx]
    # Sum of all signals minus winner's
    all_signal_sums = signals.sum(axis=1)
    other_sums = all_signal_sums - winner_signals
    winner_vals = alpha * winner_signals + (0.5 * eta / max(N - 1, 1)) * other_sums

    winner_bids = bids[rows, winner_idx]

    # For SPA, payment is second-highest bid, not winner's bid
    if auction_type == "first":
        payment = winner_bids
    else:
        sorted_bids = np.sort(bids, axis=1)
        payment = sorted_bids[:, -2]

    btv = np.where(winner_vals > 1e-12, payment / winner_vals, 0.0)
    median = float(np.median(btv))
    iqr = float(np.percentile(btv, 75) - np.percentile(btv, 25))
    return median, iqr


def bne_winners_curse_benchmark(eta, N, auction_type, M=100_000, seed=42):
    """MC benchmark for winner's curse frequency under BNE. Returns float."""
    rng = np.random.default_rng(seed)
    phi = compute_bne_bid_coefficient(eta, N, auction_type)
    alpha, beta = compute_alpha_beta(eta, N)

    signals = rng.uniform(0, 1, size=(M, N))
    bids = phi * signals

    winner_idx = bids.argmax(axis=1)
    rows = np.arange(M)

    winner_signals = signals[rows, winner_idx]
    all_signal_sums = signals.sum(axis=1)
    other_sums = all_signal_sums - winner_signals
    winner_vals = alpha * winner_signals + (0.5 * eta / max(N - 1, 1)) * other_sums

    if auction_type == "first":
        payment = bids[rows, winner_idx]
    else:
        sorted_bids = np.sort(bids, axis=1)
        payment = sorted_bids[:, -2]

    curse_freq = float((payment > winner_vals).mean())
    return curse_freq


def bne_bid_dispersion_benchmark(eta, N, auction_type, M=100_000, seed=42):
    """MC benchmark for within-round bid SD under BNE. Returns float."""
    rng = np.random.default_rng(seed)
    phi = compute_bne_bid_coefficient(eta, N, auction_type)

    signals = rng.uniform(0, 1, size=(M, N))
    bids = phi * signals

    round_sds = bids.std(axis=1)
    return float(round_sds.mean())


# ---------------------------------------------------------------------------
# Full verification
# ---------------------------------------------------------------------------

def run_full_verification(quick=False):
    """
    Run all verification checks across parameter grid.
    """
    eta_values = [0.0, 0.5, 1.0]
    N_values = [2, 3, 6]
    auction_types = ["first", "second"]

    if quick:
        M_dev = 50_000
        M_rev = 100_000
        n_grid = 51
    else:
        M_dev = 200_000
        M_rev = 500_000
        n_grid = 101

    results = {
        "deviation_checks": [],
        "revenue_checks": [],
        "parameters": {
            "eta_values": eta_values,
            "N_values": N_values,
            "M_deviation": M_dev,
            "M_revenue": M_rev,
            "n_grid": n_grid,
        },
    }

    # 1. Deviation checks: 18 configurations
    print("=" * 60)
    print("BNE DEVIATION CHECKS")
    print("=" * 60)
    for eta in eta_values:
        for N in N_values:
            for atype in auction_types:
                label = f"eta={eta}, N={N}, {atype.upper()}"
                print(f"  Checking {label}...", end=" ", flush=True)
                t0 = time.time()
                result = check_no_profitable_deviation(
                    eta, N, atype, n_grid=n_grid, M=M_dev
                )
                elapsed = time.time() - t0
                status = "PASS" if result["max_gain"] < 0.005 else "FAIL"
                print(f"max_gain={result['max_gain']:.6f}  [{status}]  ({elapsed:.1f}s)")
                results["deviation_checks"].append(result)

    # 2. Revenue checks: 9 configurations (compare FPA vs SPA and vs analytical)
    print()
    print("=" * 60)
    print("REVENUE FORMULA VALIDATION")
    print("=" * 60)
    for eta in eta_values:
        for N in N_values:
            ana_rev = analytical_revenue(eta, N)
            fpa_rev, fpa_se = mc_revenue(eta, N, "first", M=M_rev)
            spa_rev, spa_se = mc_revenue(eta, N, "second", M=M_rev)
            fpa_gap = abs(fpa_rev - ana_rev)
            spa_gap = abs(spa_rev - ana_rev)
            equiv_gap = abs(fpa_rev - spa_rev)

            entry = {
                "eta": eta,
                "N": N,
                "analytical": ana_rev,
                "fpa_mc": fpa_rev,
                "fpa_se": fpa_se,
                "spa_mc": spa_rev,
                "spa_se": spa_se,
                "fpa_gap": fpa_gap,
                "spa_gap": spa_gap,
                "fpa_spa_gap": equiv_gap,
            }
            results["revenue_checks"].append(entry)

            fpa_ok = fpa_gap < 2 * fpa_se
            spa_ok = spa_gap < 2 * spa_se
            equiv_ok = equiv_gap < 0.001
            status = "PASS" if (fpa_ok and spa_ok and equiv_ok) else "WARN"
            print(
                f"  eta={eta}, N={N}: "
                f"analytical={ana_rev:.6f}  "
                f"FPA_MC={fpa_rev:.6f}(±{fpa_se:.6f})  "
                f"SPA_MC={spa_rev:.6f}(±{spa_se:.6f})  "
                f"FPA-SPA={equiv_gap:.6f}  [{status}]"
            )

    return results


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def generate_deviation_table(results):
    """Generate LaTeX table summarizing deviation checks."""
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  \caption{Maximum payoff gain from unilateral deviation. "
                 r"Values near zero confirm BNE optimality (200K MC draws per configuration).}")
    lines.append(r"  \label{tab:bne_deviation}")
    lines.append(r"  \begin{tabular}{ccccc}")
    lines.append(r"    \toprule")
    lines.append(r"    $\eta$ & $N$ & Auction & Bid slope & Max Gain \\")
    lines.append(r"    \midrule")

    for check in results["deviation_checks"]:
        eta = check["eta"]
        N = check["N"]
        atype = check["auction_type"].upper()
        phi = check["phi"]
        gain = check["max_gain"]
        lines.append("    {} & {} & {} & {:.4f} & {:.6f} \\\\".format(eta, N, atype, phi, gain))

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_revenue_table(results):
    """Generate LaTeX table comparing analytical and MC revenue."""
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  \caption{Revenue formula validation: analytical vs.\ Monte Carlo (500K draws). Revenue equivalence holds under iid signals.}")
    lines.append(r"  \label{tab:bne_revenue_match}")
    lines.append(r"  \begin{tabular}{cccccc}")
    lines.append(r"    \toprule")
    lines.append(r"    $\eta$ & $N$ & $R^{\text{analytical}}$ "
                 r"& $R^{\text{FPA}}_{\text{MC}}$ & $R^{\text{SPA}}_{\text{MC}}$ "
                 r"& $|R^{\text{FPA}} - R^{\text{SPA}}|$ \\")
    lines.append(r"    \midrule")

    for entry in results["revenue_checks"]:
        eta = entry["eta"]
        N = entry["N"]
        ana = entry["analytical"]
        fpa = entry["fpa_mc"]
        spa = entry["spa_mc"]
        gap = entry["fpa_spa_gap"]
        lines.append("    {} & {} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\".format(eta, N, ana, fpa, spa, gap))

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BNE Verification")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer MC draws")
    args = parser.parse_args()

    print(f"Running BNE verification ({'quick' if args.quick else 'full'} mode)...\n")
    t0 = time.time()
    results = run_full_verification(quick=args.quick)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    # Save JSON
    out_dir = os.path.join("results", "verification")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "bne_verification_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Generate LaTeX tables
    tables_dir = os.path.join("paper", "tables")
    os.makedirs(tables_dir, exist_ok=True)

    dev_table = generate_deviation_table(results)
    dev_path = os.path.join(tables_dir, "bne_deviation.tex")
    with open(dev_path, "w") as f:
        f.write(dev_table)
    print(f"Deviation table: {dev_path}")

    rev_table = generate_revenue_table(results)
    rev_path = os.path.join(tables_dir, "bne_revenue_match.tex")
    with open(rev_path, "w") as f:
        f.write(rev_table)
    print(f"Revenue table:   {rev_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    max_gains = [c["max_gain"] for c in results["deviation_checks"]]
    print(f"  Deviation checks: {len(max_gains)} configs, max gain = {max(max_gains):.6f}")
    all_pass = all(g < 0.005 for g in max_gains)
    print(f"  All deviations < 0.005: {'YES' if all_pass else 'NO'}")

    rev_gaps = [e["fpa_spa_gap"] for e in results["revenue_checks"]]
    print(f"  Revenue checks: {len(rev_gaps)} configs, max FPA-SPA gap = {max(rev_gaps):.6f}")
    equiv_pass = all(g < 0.001 for g in rev_gaps)
    print(f"  Revenue equivalence (gap < 0.001): {'YES' if equiv_pass else 'NO'}")


if __name__ == "__main__":
    main()
