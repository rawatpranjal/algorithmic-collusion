#!/usr/bin/env python3
"""
Tests 1-6: Nash/BNE equilibria verification.

All formulas re-derived from paper/sections/equilibria.tex.
Zero imports from project source code.
"""

import sys
import time
from fractions import Fraction

import numpy as np

from helpers import (
    compute_alpha_beta,
    compute_alpha_beta_exact,
    bne_bid_slope,
    bne_bid_slope_exact,
    analytical_revenue,
    analytical_revenue_exact,
    efficient_benchmark,
    efficient_benchmark_exact,
    bne_regret,
    bne_regret_exact,
)


# ---------------------------------------------------------------------------
# Test 1: Constant-value discrete NE threshold
# ---------------------------------------------------------------------------

def test_1_discrete_ne(verbose=False):
    """
    Claim (equilibria.tex L6-12):
    In FPA with v=1 and bids in {0, 0.1, ..., 1.0}, symmetric profile (b,...,b)
    is NE iff b >= (0.9n - 1)/(n - 1).

    Method: EXHAUSTIVE ENUMERATION. For each b on the grid and each n,
    check ALL possible deviations. Pure algebra, zero randomness.
    """
    results = []
    bid_grid = [round(x * 0.1, 1) for x in range(11)]  # {0, 0.1, ..., 1.0}

    for n in [2, 3, 4, 5, 6, 10]:
        threshold_exact = Fraction(9 * n - 10, 10 * (n - 1))
        threshold_float = float(threshold_exact)

        # Find minimum bid on grid at or above threshold
        min_ne_bid = None
        for b in bid_grid:
            if Fraction(b) >= threshold_exact:
                min_ne_bid = b
                break

        for b in bid_grid:
            is_ne = True
            # Payoff at symmetric profile: win with prob 1/n (tie-break)
            payoff_sym = (1.0 - b) / n

            # Check all deviations
            for b_dev in bid_grid:
                if b_dev == b:
                    continue

                if b_dev > b:
                    # Win with certainty
                    payoff_dev = 1.0 - b_dev
                elif b_dev < b:
                    # Lose with certainty (all others bid b > b_dev)
                    payoff_dev = 0.0
                else:
                    payoff_dev = payoff_sym  # same bid

                if payoff_dev > payoff_sym + 1e-15:
                    is_ne = False
                    break

            # Check against threshold
            expected_ne = (Fraction(b) >= threshold_exact)
            if is_ne != expected_ne:
                results.append({
                    "n": n, "b": b, "is_ne": is_ne,
                    "expected_ne": expected_ne,
                    "threshold": threshold_float,
                    "status": "FAIL"
                })
                if verbose:
                    print(f"  FAIL: n={n}, b={b}, is_ne={is_ne}, expected={expected_ne}")
            elif verbose and b == min_ne_bid:
                print(f"  n={n}: threshold={(9*n-10)}/{10*(n-1)}={threshold_float:.4f}, "
                      f"min NE bid={min_ne_bid}")

    n_checked = sum(1 for n in [2, 3, 4, 5, 6, 10] for _ in bid_grid)
    n_fail = len(results)
    passed = n_fail == 0

    return {
        "test": 1,
        "name": "Constant-value discrete NE threshold",
        "method": "Exhaustive enumeration",
        "passed": passed,
        "n_checked": n_checked,
        "n_failures": n_fail,
        "failures": results[:5],  # first 5 only
    }


# ---------------------------------------------------------------------------
# Test 2: Affiliated BNE best-response verification
# ---------------------------------------------------------------------------

def test_2_bne_best_response(verbose=False, quick=False):
    """
    Claim (equilibria.tex L31-39):
    b^SPA(s) = phi*s with phi = alpha + n*beta/2
    b^FPA(s) = (n-1)/n * phi * s

    Method: For each signal s, evaluate expected payoff over a fine bid grid
    using ANALYTICAL integration (not MC). Opponents play BNE with iid U[0,1].

    For FPA with n bidders:
      E[payoff | bid b, signal s] = integral over opponent signals
      Win iff b >= phi * max(s_{-i}) => max(s_{-i}) <= b/phi
      P(win) = (b/phi)^{n-1} for b in [0, phi]
      E[v_i | win, s] = alpha*s + beta*(n-1)*E[s_j | s_j <= b/phi]/1
                       = alpha*s + beta*(n-1)*(b/phi)/2 ... but actually:
      E[v_i] depends on opponent signals given winning.

    Actually for FPA: payoff = (v_i - b) * P(win)
    Since opponents are iid with bids phi*s_j, P(b >= all opp bids) = P(s_j <= b/phi for all j)
    = (min(b/phi, 1))^{n-1}

    E[payoff] needs to integrate over opponent signals. Let's do it properly:
    P(win) = (min(b/phi, 1))^{n-1}
    E[v_i | s_i, win] = alpha*s_i + beta*sum_j E[s_j | s_j <= b/phi]
                       = alpha*s_i + beta*(n-1)*(min(b/phi, 1))/2

    So E[payoff_FPA | bid b, signal s_i]
       = [(alpha*s_i + beta*(n-1)*min(b/phi,1)/2) - b] * (min(b/phi,1))^{n-1}
    """
    n_grid = 10000 if not quick else 2000
    configs = []
    for eta in [0.0, 0.5, 1.0]:
        for n in [2, 3, 6]:
            for auction_type in ["first", "second"]:
                configs.append((eta, n, auction_type))

    max_gain_overall = -np.inf
    all_details = []

    for eta, n, auction_type in configs:
        alpha, beta = compute_alpha_beta(eta, n)
        phi = bne_bid_slope(eta, n, auction_type)
        phi_spa = bne_bid_slope(eta, n, "second")  # for SPA payment

        signal_values = np.linspace(0.05, 0.95, 19)
        bid_grid = np.linspace(0, max(phi * 1.1, 1.0), n_grid)

        max_gain_config = -np.inf

        for s_i in signal_values:
            bne_bid = phi * s_i

            if auction_type == "first":
                # P(win | bid b) = (min(b/phi_spa, 1))^{n-1} where phi_spa is SPA slope
                # Wait - opponents bid phi_FPA * s_j in FPA, phi_SPA * s_j in SPA
                # In FPA: opponent bids = phi_FPA * s_j
                phi_opp = bne_bid_slope(eta, n, auction_type)
                # P(win) = P(phi_opp * s_j < b for all j) = (min(b/phi_opp, 1))^{n-1}
                p_win = np.clip(bid_grid / phi_opp, 0, 1) ** (n - 1)

                # E[s_j | s_j <= b/phi_opp] = min(b/phi_opp, 1) / 2
                cond_mean_sj = np.clip(bid_grid / phi_opp, 0, 1) / 2.0

                # E[v_i | s_i, win at bid b]
                e_vi_given_win = alpha * s_i + beta * (n - 1) * cond_mean_sj

                # Payoff = (E[v_i | win] - b) * P(win)
                expected_payoff = (e_vi_given_win - bid_grid) * p_win

            else:
                # SPA: payment = second-highest bid = phi * s_(n-1:n-1)
                # opponents bid phi * s_j
                phi_opp = bne_bid_slope(eta, n, auction_type)
                # P(win) = P(all opp signals < b/phi_opp)
                p_win = np.clip(bid_grid / phi_opp, 0, 1) ** (n - 1)

                # E[payment | win at bid b]
                # Payment = max opponent bid = phi_opp * max(s_{-i})
                # E[max(s_{-i}) | all s_{-i} < b/phi_opp]
                # = E[max of n-1 uniforms on [0, b/phi_opp]]
                # = (n-1)/n * min(b/phi_opp, 1)
                cap = np.clip(bid_grid / phi_opp, 0, 1)
                e_payment = phi_opp * (n - 1) / n * cap

                # E[v_i | win]
                cond_mean_sj = cap / 2.0
                e_vi_given_win = alpha * s_i + beta * (n - 1) * cond_mean_sj

                # Payoff = (E[v_i | win] - E[payment | win]) * P(win)
                expected_payoff = (e_vi_given_win - e_payment) * p_win

            # Find best bid on grid
            best_idx = np.argmax(expected_payoff)
            best_payoff = expected_payoff[best_idx]
            best_bid = bid_grid[best_idx]

            # BNE payoff
            bne_idx = np.argmin(np.abs(bid_grid - bne_bid))
            bne_payoff = expected_payoff[bne_idx]

            gain = best_payoff - bne_payoff
            max_gain_config = max(max_gain_config, gain)

        max_gain_overall = max(max_gain_overall, max_gain_config)
        passed_config = max_gain_config < 1e-4
        all_details.append({
            "eta": eta, "n": n, "auction_type": auction_type,
            "max_gain": float(max_gain_config),
            "passed": passed_config,
        })

        if verbose:
            status = "PASS" if passed_config else "FAIL"
            print(f"  eta={eta}, n={n}, {auction_type}: max_gain={max_gain_config:.8f} [{status}]")

    passed = all(d["passed"] for d in all_details)
    return {
        "test": 2,
        "name": "Affiliated BNE best-response",
        "method": "Analytical expected payoff on 10K bid grid",
        "passed": passed,
        "max_gain": float(max_gain_overall),
        "details": all_details,
    }


# ---------------------------------------------------------------------------
# Test 3: Revenue equivalence analytical verification
# ---------------------------------------------------------------------------

def test_3_revenue_equivalence(verbose=False, quick=False):
    """
    Claim (equilibria.tex L42-48):
    R^FPA = R^SPA = (n-1)/(n+1) * (alpha + n*beta/2)

    Method: EXACT INTEGRATION.
    FPA: R = phi_FPA * E[s_(n:n)] = (n-1)/n * phi * n/(n+1) = (n-1)/(n+1) * phi
    SPA: R = phi_SPA * E[s_(n-1:n)] = phi * (n-1)/(n+1)
    Both equal (n-1)/(n+1) * phi. QED.

    Verify algebraically with exact Fraction arithmetic, then confirm with MC.
    """
    results = []
    mc_samples = 1_000_000 if not quick else 100_000
    rng = np.random.default_rng(42)

    for eta in [0, Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), 1]:
        for n in [2, 3, 4, 6, 10]:
            eta_f = Fraction(eta)
            n_f = Fraction(n)
            alpha, beta = compute_alpha_beta_exact(eta_f, n_f)
            phi = alpha + n_f * beta / 2

            # Exact revenue
            R_exact = (n_f - 1) / (n_f + 1) * phi

            # Verify FPA path: phi_FPA * E[s_(n:n)]
            phi_fpa = (n_f - 1) / n_f * phi
            E_smax = n_f / (n_f + 1)
            R_fpa_exact = phi_fpa * E_smax
            fpa_match = (R_fpa_exact == R_exact)

            # Verify SPA path: phi_SPA * E[s_(n-1:n)]
            phi_spa = phi
            E_s2nd = (n_f - 1) / (n_f + 1)
            R_spa_exact = phi_spa * E_s2nd
            spa_match = (R_spa_exact == R_exact)

            # MC confirmation
            signals = rng.uniform(0, 1, size=(mc_samples, int(n)))
            sorted_signals = np.sort(signals, axis=1)

            mc_fpa_rev = float(phi_fpa) * sorted_signals[:, -1].mean()
            mc_spa_rev = float(phi_spa) * sorted_signals[:, -2].mean()
            mc_equiv_gap = abs(mc_fpa_rev - mc_spa_rev)

            passed = fpa_match and spa_match and mc_equiv_gap < 0.002

            results.append({
                "eta": float(eta), "n": int(n),
                "R_exact": str(R_exact), "R_float": float(R_exact),
                "fpa_algebraic_match": fpa_match,
                "spa_algebraic_match": spa_match,
                "mc_fpa": mc_fpa_rev, "mc_spa": mc_spa_rev,
                "mc_equiv_gap": mc_equiv_gap,
                "passed": passed,
            })

            if verbose:
                status = "PASS" if passed else "FAIL"
                print(f"  eta={float(eta)}, n={int(n)}: R={float(R_exact):.6f}, "
                      f"FPA_alg={'OK' if fpa_match else 'FAIL'}, "
                      f"SPA_alg={'OK' if spa_match else 'FAIL'}, "
                      f"MC gap={mc_equiv_gap:.6f} [{status}]")

    all_passed = all(r["passed"] for r in results)
    return {
        "test": 3,
        "name": "Revenue equivalence R^FPA = R^SPA",
        "method": "Exact Fraction algebra + 1M MC confirmation",
        "passed": all_passed,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Test 4: E[v_(1)] efficient benchmark
# ---------------------------------------------------------------------------

def test_4_efficient_benchmark(verbose=False, quick=False):
    """
    Claim (equilibria.tex L51-55):
    E[v_(1)] = (alpha-beta)*n/(n+1) + beta*n/2

    Derivation:
    v_i = (alpha-beta)*s_i + beta*S  where S = sum of all signals
    max_i v_i occurs at max s_i since alpha-beta >= 0
    E[max v_i] = (alpha-beta)*E[s_(n:n)] + beta*E[S]
               = (alpha-beta)*n/(n+1) + beta*n/2

    Method: Verify symbolically with Fraction, then confirm with MC.
    """
    results = []
    mc_samples = 1_000_000 if not quick else 100_000
    rng = np.random.default_rng(42)

    for eta in [0, Fraction(1, 2), 1]:
        for n in [2, 3, 4, 6, 10]:
            # Exact computation
            E_v1_exact = efficient_benchmark_exact(eta, n)
            E_v1_float = float(E_v1_exact)

            # Verify derivation step by step
            alpha, beta = compute_alpha_beta_exact(eta, n)
            n_f = Fraction(n)
            E_smax = n_f / (n_f + 1)
            E_S = n_f / 2
            E_v1_derived = (alpha - beta) * E_smax + beta * E_S
            derivation_match = (E_v1_derived == E_v1_exact)

            # MC confirmation
            signals = rng.uniform(0, 1, size=(mc_samples, int(n)))
            alpha_f, beta_f = compute_alpha_beta(float(eta), int(n))

            # Compute all valuations
            signal_sums = signals.sum(axis=1, keepdims=True)  # (M, 1)
            # v_i = alpha*s_i + beta*(S - s_i) = (alpha-beta)*s_i + beta*S
            valuations = (alpha_f - beta_f) * signals + beta_f * signal_sums
            max_vals = valuations.max(axis=1)
            mc_mean = max_vals.mean()
            mc_se = max_vals.std() / np.sqrt(mc_samples)

            mc_match = abs(mc_mean - E_v1_float) < 3 * mc_se + 1e-6

            passed = derivation_match and mc_match
            results.append({
                "eta": float(eta), "n": int(n),
                "E_v1_exact": str(E_v1_exact),
                "E_v1_float": E_v1_float,
                "derivation_match": derivation_match,
                "mc_mean": float(mc_mean), "mc_se": float(mc_se),
                "mc_match": mc_match,
                "passed": passed,
            })

            if verbose:
                status = "PASS" if passed else "FAIL"
                print(f"  eta={float(eta)}, n={int(n)}: E[v1]={E_v1_float:.6f}, "
                      f"MC={mc_mean:.6f}+/-{mc_se:.6f} [{status}]")

    all_passed = all(r["passed"] for r in results)
    return {
        "test": 4,
        "name": "E[v_(1)] efficient benchmark formula",
        "method": "Exact Fraction derivation + 1M MC confirmation",
        "passed": all_passed,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Test 5: BNE regret formula via MC simulation
# ---------------------------------------------------------------------------

def test_5_bne_regret_mc(verbose=False, quick=False):
    """
    Claim (equilibria.tex L56-60):
    Regret* = 1 - R_BNE / E[v_(1)]

    where R_BNE and E[v_(1)] are the closed-form expressions from the paper.

    The old test verified the telescoping identity 1-R = (1-E[v1]) + (E[v1]-R_BNE) + (R_BNE-R),
    which is a tautology (a = a). This replacement actually verifies the BNE regret VALUE
    by simulating BNE play: draw signals, compute BNE bids, run the auction, measure
    revenue, and check that the empirical regret matches the closed-form Regret*.

    Method: For each (eta, n, auction_type), simulate 1M auctions under BNE play.
    Compute empirical revenue, empirical E[v_(1)], and empirical regret.
    Verify all three match their closed-form values within statistical tolerance.
    """
    mc_samples = 1_000_000 if not quick else 100_000
    rng = np.random.default_rng(42)
    results = []

    for eta_frac in [0, Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), 1]:
        for n in [2, 3, 6]:
            eta_f = float(eta_frac)
            alpha, beta = compute_alpha_beta(eta_f, n)

            # Closed-form values
            E_v1_formula = efficient_benchmark(eta_f, n)
            R_bne_formula = analytical_revenue(eta_f, n)
            regret_formula = bne_regret(eta_f, n)

            for auction_type in ["first", "second"]:
                phi = bne_bid_slope(eta_f, n, auction_type)

                # Draw iid signals
                signals = rng.uniform(0, 1, size=(mc_samples, n))
                signal_sums = signals.sum(axis=1, keepdims=True)

                # Compute affiliated valuations: v_i = alpha*s_i + beta*sum_{j!=i} s_j
                # = (alpha - beta)*s_i + beta*S
                valuations = (alpha - beta) * signals + beta * signal_sums

                # BNE bids: b_i = phi * s_i
                bids = phi * signals

                # Determine winner and payment
                winner_idx = np.argmax(bids, axis=1)  # highest bid wins
                rows = np.arange(mc_samples)

                # Winner's valuation
                winner_vals = valuations[rows, winner_idx]

                # Highest valuation (efficient benchmark)
                max_vals = valuations.max(axis=1)

                if auction_type == "first":
                    # Payment = winner's bid
                    revenue = bids[rows, winner_idx]
                else:
                    # Payment = second-highest bid
                    sorted_bids = np.sort(bids, axis=1)
                    revenue = sorted_bids[:, -2]

                # Empirical means
                mc_revenue = revenue.mean()
                mc_E_v1 = max_vals.mean()
                mc_regret = 1.0 - mc_revenue / mc_E_v1 if mc_E_v1 > 0 else 0.0

                # Standard errors
                se_rev = revenue.std() / np.sqrt(mc_samples)
                se_v1 = max_vals.std() / np.sqrt(mc_samples)

                # Check: revenue matches R_BNE (revenue equivalence holds for both formats)
                rev_ok = abs(mc_revenue - R_bne_formula) < 4 * se_rev + 1e-4
                # Check: E[v1] matches formula
                v1_ok = abs(mc_E_v1 - E_v1_formula) < 4 * se_v1 + 1e-4
                # Check: regret matches formula (use wider tolerance since it's a ratio)
                regret_ok = abs(mc_regret - regret_formula) < 0.005

                passed = rev_ok and v1_ok and regret_ok

                results.append({
                    "eta": eta_f, "n": n, "auction_type": auction_type,
                    "R_bne_formula": R_bne_formula, "mc_revenue": float(mc_revenue),
                    "E_v1_formula": E_v1_formula, "mc_E_v1": float(mc_E_v1),
                    "regret_formula": regret_formula, "mc_regret": float(mc_regret),
                    "rev_ok": rev_ok, "v1_ok": v1_ok, "regret_ok": regret_ok,
                    "passed": passed,
                })

                if verbose:
                    status = "PASS" if passed else "FAIL"
                    print(f"  eta={eta_f}, n={n}, {auction_type}: "
                          f"R={mc_revenue:.5f} (formula={R_bne_formula:.5f}), "
                          f"Regret={mc_regret:.4f} (formula={regret_formula:.4f}) [{status}]")

    all_passed = all(r["passed"] for r in results)
    return {
        "test": 5,
        "name": "BNE regret formula (MC simulation of BNE play)",
        "method": "Simulate 1M BNE auctions, verify revenue and regret match closed-form",
        "passed": all_passed,
        "n_configs": len(results),
        "n_failures": sum(1 for r in results if not r["passed"]),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Test 6: Paper table values verification
# ---------------------------------------------------------------------------

def test_6_paper_table_values(verbose=False):
    """
    Verify the table in equilibria.tex L69-82 using exact rational arithmetic.

    Table claims:
    eta=0, n=2: E[v1]=0.667, R_BNE=0.333, Regret*=50.0%, Raw=66.7%
    eta=0, n=6: E[v1]=0.857, R_BNE=0.714, Regret*=16.7%, Raw=28.6%
    eta=1, n=2: E[v1]=0.500, R_BNE=0.333, Regret*=33.3%, Raw=66.7%
    eta=1, n=6: E[v1]=0.643, R_BNE=0.571, Regret*=11.1%, Raw=42.9%
    """
    table_claims = [
        (0, 2, "0.667", "0.333", "50.0", "66.7"),
        (0, 6, "0.857", "0.714", "16.7", "28.6"),
        (1, 2, "0.500", "0.333", "33.3", "66.7"),
        (1, 6, "0.643", "0.571", "11.1", "42.9"),
    ]

    results = []

    for eta, n, ev1_str, rbne_str, regret_str, raw_str in table_claims:
        # Exact computation
        E_v1 = efficient_benchmark_exact(eta, n)
        R_bne = analytical_revenue_exact(eta, n)
        regret_star = bne_regret_exact(eta, n)
        raw_shortfall = 1 - R_bne

        # Check rounding to 3 decimal places
        ev1_rounded = f"{float(E_v1):.3f}"
        rbne_rounded = f"{float(R_bne):.3f}"
        regret_pct = f"{float(regret_star)*100:.1f}"
        raw_pct = f"{float(raw_shortfall)*100:.1f}"

        ev1_match = ev1_rounded == ev1_str
        rbne_match = rbne_rounded == rbne_str
        regret_match = regret_pct == regret_str
        raw_match = raw_pct == raw_str

        passed = ev1_match and rbne_match and regret_match and raw_match

        detail = {
            "eta": eta, "n": n,
            "E_v1_exact": str(E_v1), "E_v1_rounded": ev1_rounded,
            "E_v1_claim": ev1_str, "E_v1_match": ev1_match,
            "R_bne_exact": str(R_bne), "R_bne_rounded": rbne_rounded,
            "R_bne_claim": rbne_str, "R_bne_match": rbne_match,
            "regret_exact": str(regret_star), "regret_pct": regret_pct,
            "regret_claim": regret_str, "regret_match": regret_match,
            "raw_pct": raw_pct, "raw_claim": raw_str, "raw_match": raw_match,
            "passed": passed,
        }
        results.append(detail)

        if verbose:
            status = "PASS" if passed else "FAIL"
            mismatches = []
            if not ev1_match:
                mismatches.append(f"E[v1]: got {ev1_rounded} expected {ev1_str}")
            if not rbne_match:
                mismatches.append(f"R_BNE: got {rbne_rounded} expected {rbne_str}")
            if not regret_match:
                mismatches.append(f"Regret: got {regret_pct}% expected {regret_str}%")
            if not raw_match:
                mismatches.append(f"Raw: got {raw_pct}% expected {raw_str}%")
            extra = f"  ({'; '.join(mismatches)})" if mismatches else ""
            print(f"  eta={eta}, n={n}: [{status}]{extra}")

    all_passed = all(r["passed"] for r in results)
    return {
        "test": 6,
        "name": "Paper table values (exact rational arithmetic)",
        "method": "Fraction arithmetic, verify rounding",
        "passed": all_passed,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all(verbose=False, quick=False):
    """Run all equilibria tests. Returns list of result dicts."""
    tests = [
        ("Test 1: Discrete NE threshold", lambda: test_1_discrete_ne(verbose)),
        ("Test 2: BNE best-response", lambda: test_2_bne_best_response(verbose, quick)),
        ("Test 3: Revenue equivalence", lambda: test_3_revenue_equivalence(verbose, quick)),
        ("Test 4: E[v_(1)] benchmark", lambda: test_4_efficient_benchmark(verbose, quick)),
        ("Test 5: BNE regret (MC)", lambda: test_5_bne_regret_mc(verbose, quick)),
        ("Test 6: Paper table values", lambda: test_6_paper_table_values(verbose)),
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
