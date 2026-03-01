#!/usr/bin/env python3
"""
Tests 13-15: Valuation model properties.

All formulas from paper/sections/equilibria.tex.
Zero imports from project source code.
"""

import sys
import time
from fractions import Fraction

import numpy as np

from helpers import (
    compute_alpha_beta,
    compute_alpha_beta_exact,
    analytical_revenue,
    analytical_revenue_exact,
    bne_bid_slope,
    bne_bid_slope_exact,
)


# ---------------------------------------------------------------------------
# Test 13: alpha - beta >= 0 for eta in [0,1], n >= 2
# ---------------------------------------------------------------------------

def test_13_alpha_minus_beta(verbose=False):
    """
    Claim (equilibria.tex L28):
    (alpha - beta) >= 0 for eta in [0,1] and n >= 2.
    Equality only at n=2, eta=1.

    Algebraic proof:
    alpha - beta = (1 - eta/2) - eta/(2(n-1))
                = 1 - eta*(1/2 + 1/(2(n-1)))
                = 1 - eta*n/(2(n-1))

    For eta <= 1 and n >= 2:
      eta*n/(2(n-1)) <= n/(2(n-1)) = 1/(2*(1-1/n)) <= 1
    Since n >= 2: n/(2(n-1)) = 1 + 1/(2(n-1)) ... wait let me redo.
      n/(2(n-1)) for n=2: 2/2 = 1, so alpha-beta = 1-eta*1 = 1-eta >= 0 for eta<=1
      n/(2(n-1)) for n=3: 3/4, so alpha-beta = 1-3eta/4 >= 1-3/4 = 1/4 > 0
      In general n/(2(n-1)) is decreasing in n and equals 1 at n=2.
      So alpha-beta = 1 - eta*n/(2(n-1)) >= 1 - 1 = 0, with equality iff eta=1 and n=2.

    Verify with exact Fraction for all (eta, n) combos.
    """
    results = []

    # Symbolic proof using Fraction
    # alpha - beta = 1 - eta/2 - eta/(2(n-1)) = 1 - eta*n/(2(n-1))
    for eta_num in range(0, 11):  # eta = 0/10, 1/10, ..., 10/10
        eta = Fraction(eta_num, 10)
        for n in range(2, 21):
            n_f = Fraction(n)
            alpha, beta = compute_alpha_beta_exact(eta, n)
            diff = alpha - beta

            # Verify the algebraic identity
            diff_formula = 1 - eta * n_f / (2 * (n_f - 1))
            identity_holds = (diff == diff_formula)

            # Check non-negativity
            non_negative = (diff >= 0)

            # Check strict positivity except at boundary
            if eta == 1 and n == 2:
                expected_zero = (diff == 0)
                passed = identity_holds and non_negative and expected_zero
            else:
                expected_positive = (diff > 0)
                passed = identity_holds and non_negative and expected_positive

            if not passed:
                results.append({
                    "eta": float(eta), "n": n,
                    "diff": str(diff), "diff_float": float(diff),
                    "identity_holds": identity_holds,
                    "non_negative": non_negative,
                })

    n_checked = 11 * 19  # 11 eta values * 19 n values
    all_passed = len(results) == 0

    if verbose:
        print(f"  Checked {n_checked} (eta, n) pairs using exact Fraction arithmetic")
        print(f"  Boundary case n=2, eta=1: alpha-beta = "
              f"{float(compute_alpha_beta_exact(1, 2)[0] - compute_alpha_beta_exact(1, 2)[1])}")
        if results:
            for r in results[:3]:
                print(f"  FAIL: eta={r['eta']}, n={r['n']}, diff={r['diff_float']}")

    return {
        "test": 13,
        "name": "alpha - beta >= 0",
        "method": "Exact Fraction arithmetic, exhaustive sweep",
        "passed": all_passed,
        "n_checked": n_checked,
        "n_failures": len(results),
        "failures": results[:5],
    }


# ---------------------------------------------------------------------------
# Test 14: Highest signal => highest valuation
# ---------------------------------------------------------------------------

def test_14_highest_signal_highest_value(verbose=False, quick=False):
    """
    Claim (equilibria.tex L28):
    The highest signal bidder has the highest valuation (efficient allocation).

    Proof: v_i - v_j = (alpha - beta)*(s_i - s_j).
    Since alpha - beta > 0 (strict for interior params), sign(v_i - v_j) = sign(s_i - s_j).

    Verify with exact arithmetic for all parameter combos,
    then MC at 1M samples as secondary confirmation.
    """
    # Part 1: Algebraic proof using Fraction
    algebraic_results = []
    for eta_num in range(0, 11):
        eta = Fraction(eta_num, 10)
        for n in range(2, 11):
            alpha, beta = compute_alpha_beta_exact(eta, n)
            diff = alpha - beta

            if eta == 1 and n == 2:
                # Boundary: diff = 0, ties possible
                algebraic_results.append({
                    "eta": float(eta), "n": n,
                    "alpha_minus_beta": float(diff),
                    "strict_positive": False,
                    "boundary": True,
                })
            else:
                strict_pos = (diff > 0)
                if not strict_pos:
                    algebraic_results.append({
                        "eta": float(eta), "n": n,
                        "alpha_minus_beta": float(diff),
                        "strict_positive": strict_pos,
                        "FAIL": True,
                    })

    algebraic_ok = not any(r.get("FAIL") for r in algebraic_results)

    # Part 2: MC confirmation
    mc_samples = 1_000_000 if not quick else 100_000
    rng = np.random.default_rng(42)
    mc_results = []

    for eta in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for n in [2, 3, 6]:
            if eta == 1.0 and n == 2:
                continue  # boundary case, ties expected

            alpha, beta = compute_alpha_beta(eta, n)
            signals = rng.uniform(0, 1, size=(mc_samples, n))
            signal_sums = signals.sum(axis=1, keepdims=True)

            # v_i = (alpha - beta)*s_i + beta*S
            valuations = (alpha - beta) * signals + beta * signal_sums

            # Check: argmax(signal) == argmax(valuation)
            max_signal_idx = signals.argmax(axis=1)
            max_value_idx = valuations.argmax(axis=1)

            # With continuous signals, ties have measure zero
            match_rate = (max_signal_idx == max_value_idx).mean()
            mc_results.append({
                "eta": eta, "n": n,
                "match_rate": float(match_rate),
            })

            if verbose:
                print(f"  eta={eta}, n={n}: signal-value winner match = {match_rate:.6f}")

    mc_ok = all(r["match_rate"] > 0.999 for r in mc_results)

    passed = algebraic_ok and mc_ok
    return {
        "test": 14,
        "name": "Highest signal => highest valuation",
        "method": "Exact Fraction proof + 1M MC confirmation",
        "passed": passed,
        "algebraic_ok": algebraic_ok,
        "mc_ok": mc_ok,
        "mc_results": mc_results,
    }


# ---------------------------------------------------------------------------
# Test 15: Full end-to-end BNE auction simulation
# ---------------------------------------------------------------------------

def test_15_bne_auction_e2e(verbose=False, quick=False):
    """
    Claim (equilibria.tex):
    Under BNE play with affiliated values, revenue equals the closed-form
    R_BNE = (n-1)/(n+1) * phi for BOTH auction formats.

    Unlike Test 3 (which verifies the algebraic identity phi_FPA * E[s_(n:n)]
    = phi_SPA * E[s_(n-1:n)]) and Test 5 (which verifies regret via BNE simulation),
    this test runs a COMPLETE end-to-end auction pipeline:

    1. Draw iid signals s_i ~ U[0,1]
    2. Compute affiliated values v_i = alpha*s_i + beta*sum_{j!=i} s_j
    3. Compute BNE bids: b_i = phi * s_i
    4. Determine winner (highest bid, random tie-break)
    5. Compute payment (FPA: winner's bid; SPA: second-highest bid)
    6. Accumulate revenue

    This tests the ENTIRE pipeline from model primitives to revenue, catching
    any inconsistency between the valuation model, bid functions, and payment
    rules.

    Method: 1M simulated auctions per (eta, n, auction_type) configuration.
    Verify mean revenue matches R_BNE within statistical tolerance.
    Also verify winner is always the highest-signal bidder (efficiency).
    """
    mc_samples = 1_000_000 if not quick else 100_000
    rng = np.random.default_rng(42)
    results = []

    for eta_frac in [0, Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), 1]:
        for n in [2, 3, 6]:
            eta_f = float(eta_frac)
            alpha, beta = compute_alpha_beta(eta_f, n)

            # Closed-form revenue
            R_formula = analytical_revenue(eta_f, n)

            for auction_type in ["first", "second"]:
                phi = bne_bid_slope(eta_f, n, auction_type)

                # 1. Draw signals
                signals = rng.uniform(0, 1, size=(mc_samples, n))

                # 2. Compute affiliated values
                signal_sums = signals.sum(axis=1, keepdims=True)
                valuations = (alpha - beta) * signals + beta * signal_sums

                # 3. Compute BNE bids
                bids = phi * signals

                # 4. Determine winner (highest bid = highest signal since phi > 0)
                winner_idx = np.argmax(bids, axis=1)
                rows = np.arange(mc_samples)

                # 5. Compute payment
                if auction_type == "first":
                    revenue_per_auction = bids[rows, winner_idx]
                else:
                    sorted_bids = np.sort(bids, axis=1)
                    revenue_per_auction = sorted_bids[:, -2]

                # 6. Accumulate revenue
                mc_revenue = revenue_per_auction.mean()
                se_rev = revenue_per_auction.std() / np.sqrt(mc_samples)

                # Verify revenue matches formula
                rev_ok = abs(mc_revenue - R_formula) < 4 * se_rev + 1e-4

                # Verify efficiency: winner should be highest-signal bidder
                max_signal_idx = np.argmax(signals, axis=1)
                efficiency_rate = (winner_idx == max_signal_idx).mean()
                # With continuous signals, ties have measure zero
                eff_ok = efficiency_rate > 0.9999

                # Verify winner's value is the highest value
                winner_vals = valuations[rows, winner_idx]
                max_vals = valuations.max(axis=1)
                # For eta=1, n=2 (boundary), alpha=beta so all values equal S/2
                if eta_f == 1.0 and n == 2:
                    val_match_rate = 1.0  # all values identical in this case
                else:
                    val_match_rate = (np.abs(winner_vals - max_vals) < 1e-10).mean()
                val_ok = val_match_rate > 0.9999

                passed = rev_ok and eff_ok and val_ok

                results.append({
                    "eta": eta_f, "n": n, "auction_type": auction_type,
                    "R_formula": R_formula, "mc_revenue": float(mc_revenue),
                    "se": float(se_rev),
                    "efficiency_rate": float(efficiency_rate),
                    "val_match_rate": float(val_match_rate),
                    "rev_ok": rev_ok, "eff_ok": eff_ok, "val_ok": val_ok,
                    "passed": passed,
                })

                if verbose:
                    status = "PASS" if passed else "FAIL"
                    print(f"  eta={eta_f}, n={n}, {auction_type}: "
                          f"R={mc_revenue:.5f} (formula={R_formula:.5f}), "
                          f"eff={efficiency_rate:.6f} [{status}]")

    all_passed = all(r["passed"] for r in results)
    return {
        "test": 15,
        "name": "Full end-to-end BNE auction simulation",
        "method": "Complete auction pipeline (signals -> values -> bids -> winner -> payment -> revenue) at 1M samples",
        "passed": all_passed,
        "n_configs": len(results),
        "n_failures": sum(1 for r in results if not r["passed"]),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all(verbose=False, quick=False):
    """Run all model tests."""
    tests = [
        ("Test 13: alpha-beta >= 0", lambda: test_13_alpha_minus_beta(verbose)),
        ("Test 14: Highest signal => highest value", lambda: test_14_highest_signal_highest_value(verbose, quick)),
        ("Test 15: BNE auction e2e", lambda: test_15_bne_auction_e2e(verbose, quick)),
    ]

    results = []
    for name, test_fn in tests:
        t0 = time.time()
        result = test_fn()
        elapsed = time.time() - t0
        result["elapsed"] = elapsed
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {name} ({elapsed:.1f}s)")
        results.append(result)

    return results


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    results = run_all(verbose=verbose, quick=quick)
    n_pass = sum(1 for r in results if r["passed"])
    n_total = len(results)
    print(f"\n{n_pass}/{n_total} tests passed")
    sys.exit(0 if n_pass == n_total else 1)
