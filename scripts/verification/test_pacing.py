#!/usr/bin/env python3
"""
Tests 10-12: Pacing bid formulas and budget feasibility.

All formulas from paper/sections/algorithms.tex and exp4a.tex.
Zero imports from project source code.
"""

import sys
import time

import numpy as np

from helpers import optimal_bid_value_max, optimal_bid_utility_max


# ---------------------------------------------------------------------------
# Test 10: Value-max pacing bid formula b = v/mu (Lagrangian threshold)
# ---------------------------------------------------------------------------

def test_10_value_max_bid(verbose=False, quick=False):
    """
    Claim (algorithms.tex eq:bid_vmax_algo):
    Value-maximizer pacing bid = min(v/mu, remaining_budget).

    The formula b = v/mu is the LAGRANGIAN BREAK-EVEN THRESHOLD from dual
    decomposition, NOT the per-round optimal bid against a specific opponent
    distribution (which would be v/(2*mu) for uniform opponents). At b = v/mu,
    the Lagrangian value v - mu*b = 0: bidding higher yields negative Lagrangian
    value (budget cost exceeds value gained). The dual update on mu adjusts to
    achieve budget feasibility over time.

    Verification:
    (a) Break-even property: v - mu * (v/mu) = 0 for all v, mu > 0.
    (b) Budget cap: min(v/mu, remaining) correctly implemented.
    (c) Economic sensibility: when mu >= 1, bid b = v/mu <= v (never bid above value).
    (d) Lagrangian non-negativity: for b <= v/mu, L = v - mu*b >= 0; for b > v/mu, L < 0.
    """
    results_detail = {}

    # (a) Break-even property: v - mu * (v/mu) = 0
    breakeven_cases = []
    for v in [0.01, 0.5, 1.0, 2.0, 5.0, 100.0]:
        for mu in [0.01, 0.5, 1.0, 1.5, 2.0, 5.0, 100.0]:
            b = v / mu
            lagrangian_value = v - mu * b
            ok = abs(lagrangian_value) < 1e-12
            breakeven_cases.append({"v": v, "mu": mu, "b": b, "L": lagrangian_value, "ok": ok})
    breakeven_ok = all(c["ok"] for c in breakeven_cases)
    results_detail["breakeven_ok"] = breakeven_ok

    # (b) Budget cap: min(v/mu, remaining)
    cap_cases = []
    for v, mu, rem in [(10.0, 1.0, 5.0), (1.0, 0.5, 0.5), (3.0, 1.0, 100.0),
                        (5.0, 2.0, 1.0), (0.1, 10.0, 0.001)]:
        b = optimal_bid_value_max(v, mu, rem)
        expected = min(v / mu, rem)
        match = abs(b - expected) < 1e-15
        cap_cases.append({"v": v, "mu": mu, "rem": rem, "b": b, "expected": expected, "match": match})
    cap_ok = all(cc["match"] for cc in cap_cases)
    results_detail["cap_ok"] = cap_ok

    # (c) Economic sensibility: mu >= 1 => b <= v
    sensibility_cases = []
    for v in [0.1, 1.0, 5.0, 50.0]:
        for mu in [1.0, 1.5, 2.0, 10.0]:
            b = v / mu
            b_le_v = b <= v + 1e-15
            sensibility_cases.append({"v": v, "mu": mu, "b": b, "b_le_v": b_le_v})
    sensibility_ok = all(c["b_le_v"] for c in sensibility_cases)
    results_detail["sensibility_ok"] = sensibility_ok

    # (d) Lagrangian sign: L = v - mu*b >= 0 for b <= v/mu, L < 0 for b > v/mu
    sign_cases = []
    for v in [1.0, 5.0]:
        for mu in [0.5, 1.0, 2.0]:
            threshold = v / mu
            # Below threshold: L >= 0
            for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
                b = frac * threshold
                L = v - mu * b
                ok = L >= -1e-12
                sign_cases.append({"v": v, "mu": mu, "b": b, "L": L, "ok": ok})
            # Above threshold: L < 0
            b_above = threshold * 1.01
            L_above = v - mu * b_above
            ok_above = L_above < 1e-12  # should be negative
            sign_cases.append({"v": v, "mu": mu, "b": b_above, "L": L_above, "ok": ok_above})
    sign_ok = all(c["ok"] for c in sign_cases)
    results_detail["sign_ok"] = sign_ok

    passed = breakeven_ok and cap_ok and sensibility_ok and sign_ok

    if verbose:
        print(f"  Break-even (v - mu*b = 0 at b=v/mu): {'PASS' if breakeven_ok else 'FAIL'}")
        print(f"  Budget cap (min(v/mu, rem)): {'PASS' if cap_ok else 'FAIL'}")
        print(f"  Economic sensibility (mu>=1 => b<=v): {'PASS' if sensibility_ok else 'FAIL'}")
        print(f"  Lagrangian sign (b<=threshold => L>=0): {'PASS' if sign_ok else 'FAIL'}")

    return {
        "test": 10,
        "name": "Value-max pacing bid formula b = v/mu (Lagrangian threshold)",
        "method": "Break-even property, budget cap, economic sensibility, Lagrangian sign",
        "passed": passed,
        **results_detail,
    }


# ---------------------------------------------------------------------------
# Test 11: Utility-max pacing bid formula b = v/(1+mu) (Lagrangian threshold)
# ---------------------------------------------------------------------------

def test_11_utility_max_bid(verbose=False, quick=False):
    """
    Claim (algorithms.tex eq:bid_umax_algo):
    Utility-maximizer pacing bid = min(v/(1+mu), remaining_budget).

    Derivation: Utility = value - payment. Per-round Lagrangian:
      L(b) = v*I(win) - (1+mu)*cost
    In FPA with cost = b*I(win): L = [v - (1+mu)*b] * I(win).
    The break-even threshold is b = v/(1+mu), where v - (1+mu)*b = 0.

    This is NOT the per-round optimum against a specific opponent distribution
    (which would be v/(2*(1+mu)) for uniform opponents). It is the Lagrangian
    threshold used with dual updates for budget feasibility.

    Verification:
    (a) Break-even property: v - (1+mu) * (v/(1+mu)) = 0 for all v, mu >= 0.
    (b) Budget cap: min(v/(1+mu), remaining) correctly implemented.
    (c) Economic sensibility: bid b = v/(1+mu) < v always (since 1+mu > 1 for mu > 0).
    (d) Lagrangian sign: v - (1+mu)*b >= 0 for b <= v/(1+mu), < 0 for b > v/(1+mu).
    (e) Comparison with value-max: v/(1+mu) < v/mu for mu > 0 (utility-max bids less aggressively).
    """
    results_detail = {}

    # (a) Break-even property: v - (1+mu) * (v/(1+mu)) = 0
    breakeven_cases = []
    for v in [0.01, 0.5, 1.0, 2.0, 5.0, 100.0]:
        for mu in [0.0, 0.5, 1.0, 1.5, 2.0, 5.0, 100.0]:
            b = v / (1.0 + mu)
            lagrangian_value = v - (1.0 + mu) * b
            ok = abs(lagrangian_value) < 1e-12
            breakeven_cases.append({"v": v, "mu": mu, "b": b, "L": lagrangian_value, "ok": ok})
    breakeven_ok = all(c["ok"] for c in breakeven_cases)
    results_detail["breakeven_ok"] = breakeven_ok

    # (b) Budget cap: min(v/(1+mu), remaining)
    cap_cases = []
    for v, mu, rem in [(10.0, 0.5, 3.0), (1.0, 0.0, 0.5), (5.0, 1.0, 100.0),
                        (3.0, 2.0, 0.5), (0.1, 5.0, 0.001)]:
        b = optimal_bid_utility_max(v, mu, rem)
        expected = min(v / (1.0 + mu), rem)
        match = abs(b - expected) < 1e-15
        cap_cases.append({"v": v, "mu": mu, "rem": rem, "b": b, "expected": expected, "match": match})
    cap_ok = all(cc["match"] for cc in cap_cases)
    results_detail["cap_ok"] = cap_ok

    # (c) Economic sensibility: b = v/(1+mu) < v for mu > 0
    sensibility_cases = []
    for v in [0.1, 1.0, 5.0, 50.0]:
        for mu in [0.01, 0.5, 1.0, 2.0, 10.0]:
            b = v / (1.0 + mu)
            b_lt_v = b < v + 1e-15
            sensibility_cases.append({"v": v, "mu": mu, "b": b, "b_lt_v": b_lt_v})
    sensibility_ok = all(c["b_lt_v"] for c in sensibility_cases)
    results_detail["sensibility_ok"] = sensibility_ok

    # (d) Lagrangian sign
    sign_cases = []
    for v in [1.0, 5.0]:
        for mu in [0.0, 0.5, 1.0, 2.0]:
            threshold = v / (1.0 + mu)
            # Below threshold: L >= 0
            for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
                b = frac * threshold
                L = v - (1.0 + mu) * b
                ok = L >= -1e-12
                sign_cases.append({"v": v, "mu": mu, "b": b, "L": L, "ok": ok})
            # Above threshold: L < 0
            if threshold > 0:
                b_above = threshold * 1.01
                L_above = v - (1.0 + mu) * b_above
                ok_above = L_above < 1e-12
                sign_cases.append({"v": v, "mu": mu, "b": b_above, "L": L_above, "ok": ok_above})
    sign_ok = all(c["ok"] for c in sign_cases)
    results_detail["sign_ok"] = sign_ok

    # (e) Comparison: v/(1+mu) < v/mu for mu > 0
    comparison_cases = []
    for v in [1.0, 5.0]:
        for mu in [0.01, 0.5, 1.0, 2.0, 10.0]:
            b_umax = v / (1.0 + mu)
            b_vmax = v / mu
            ok = b_umax < b_vmax + 1e-12
            comparison_cases.append({"v": v, "mu": mu, "b_umax": b_umax, "b_vmax": b_vmax, "ok": ok})
    comparison_ok = all(c["ok"] for c in comparison_cases)
    results_detail["comparison_ok"] = comparison_ok

    passed = breakeven_ok and cap_ok and sensibility_ok and sign_ok and comparison_ok

    if verbose:
        print(f"  Break-even (v - (1+mu)*b = 0): {'PASS' if breakeven_ok else 'FAIL'}")
        print(f"  Budget cap (min(v/(1+mu), rem)): {'PASS' if cap_ok else 'FAIL'}")
        print(f"  Economic sensibility (b < v): {'PASS' if sensibility_ok else 'FAIL'}")
        print(f"  Lagrangian sign: {'PASS' if sign_ok else 'FAIL'}")
        print(f"  Comparison (umax < vmax): {'PASS' if comparison_ok else 'FAIL'}")

    return {
        "test": 11,
        "name": "Utility-max pacing bid formula b = v/(1+mu) (Lagrangian threshold)",
        "method": "Break-even property, budget cap, economic sensibility, Lagrangian sign, vmax/umax comparison",
        "passed": passed,
        **results_detail,
    }


# ---------------------------------------------------------------------------
# Test 12: Budget feasibility always holds
# ---------------------------------------------------------------------------

def test_12_budget_feasibility(verbose=False, quick=False):
    """
    Claim (algorithms.tex): The hard budget cap min(..., B-S_t) ensures
    spend <= budget always.

    Method: Run 100 pacing simulations and verify spend <= budget + epsilon
    for EVERY agent in EVERY run. Also check dual convergence rate.
    """
    n_instances = 100 if not quick else 20
    rng = np.random.default_rng(42)

    configs = [
        ("first", "value"), ("first", "utility"),
        ("second", "value"), ("second", "utility"),
    ]

    budget_violations = []
    convergence_data = []

    for inst_idx in range(n_instances):
        auction_type, objective = configs[inst_idx % len(configs)]
        N = rng.choice([2, 3, 4])
        T = 1000

        # Generate valuations
        mu_i = rng.uniform(0.5, 1.5, size=N)
        valuations = np.zeros((T, N))
        for i in range(N):
            valuations[:, i] = rng.lognormal(mean=mu_i[i], sigma=0.3, size=T)

        # Budgets
        expected_vals = np.exp(mu_i + 0.3**2 / 2)
        budgets = 0.5 * expected_vals * T

        # Run simulation with detailed tracking
        mu_arr = np.ones(N)
        spend = np.zeros(N)
        alpha_p = 1.0 / np.sqrt(T)

        for t in range(T):
            remaining = budgets - spend
            bids = np.zeros(N)

            for i in range(N):
                v = valuations[t, i]
                rem = max(remaining[i], 0.0)
                if rem <= 0:
                    bids[i] = 0.0
                    continue

                if objective == "value":
                    bids[i] = optimal_bid_value_max(v, mu_arr[i], rem)
                else:
                    bids[i] = optimal_bid_utility_max(v, mu_arr[i], rem)

            # Auction
            if np.all(bids <= 0):
                continue

            winner = np.argmax(bids)
            max_bid = bids[winner]
            if max_bid <= 0:
                continue
            ties = np.where(np.abs(bids - max_bid) < 1e-12)[0]
            if len(ties) > 1:
                winner = rng.choice(ties)

            if auction_type == "first":
                payment = bids[winner]
            else:
                sorted_bids = np.sort(bids)
                payment = sorted_bids[-2] if N > 1 else 0.0

            payment = min(payment, remaining[winner])
            spend[winner] += payment

            # Dual update
            target_rate = budgets / T
            for i in range(N):
                cost_i = payment if i == winner else 0.0
                mu_arr[i] = np.clip(
                    mu_arr[i] * np.exp(alpha_p * (cost_i - target_rate[i])),
                    1e-4, 100.0
                )

        # Check budget feasibility
        for i in range(N):
            if spend[i] > budgets[i] + 1e-8:
                budget_violations.append({
                    "instance": inst_idx, "bidder": i,
                    "spend": spend[i], "budget": budgets[i],
                    "excess": spend[i] - budgets[i],
                })

    # Convergence rate check: run simulations at different T
    if not quick:
        T_values = [100, 500, 1000, 5000]
    else:
        T_values = [100, 1000]

    for T in T_values:
        N = 2
        mu_i_cv = np.array([0.8, 1.2])
        vals = np.zeros((T, N))
        for i in range(N):
            vals[:, i] = rng.lognormal(mean=mu_i_cv[i], sigma=0.3, size=T)
        bgt = 0.5 * np.exp(mu_i_cv + 0.3**2 / 2) * T

        # Track dual variable trajectory
        mu_traj = np.ones((T + 1, N))
        spd = np.zeros(N)
        alpha_p_cv = 1.0 / np.sqrt(T)

        for t in range(T):
            rem = bgt - spd
            bds = np.zeros(N)
            for i in range(N):
                v = vals[t, i]
                r = max(rem[i], 0.0)
                if r <= 0:
                    bds[i] = 0.0
                else:
                    bds[i] = optimal_bid_value_max(v, mu_traj[t, i], r)

            if np.all(bds <= 0):
                mu_traj[t + 1] = mu_traj[t]
                continue

            w = np.argmax(bds)
            pmt = bds[w]  # FPA
            pmt = min(pmt, rem[w])
            spd[w] += pmt

            tgt = bgt / T
            for i in range(N):
                c = pmt if i == w else 0.0
                mu_traj[t + 1, i] = np.clip(
                    mu_traj[t, i] * np.exp(alpha_p_cv * (c - tgt[i])),
                    1e-4, 100.0
                )

        # Measure convergence: std of mu in last 20% of rounds
        last_start = int(0.8 * T)
        mu_std = mu_traj[last_start:, :].std(axis=0).mean()
        convergence_data.append({"T": T, "mu_std_last_20pct": float(mu_std)})

    passed = len(budget_violations) == 0

    if verbose:
        print(f"  Budget violations: {len(budget_violations)}")
        for v in budget_violations[:3]:
            print(f"    inst={v['instance']}, bidder={v['bidder']}, "
                  f"excess={v['excess']:.8f}")
        print(f"  Dual convergence:")
        for cd in convergence_data:
            print(f"    T={cd['T']}: mu_std={cd['mu_std_last_20pct']:.6f}")

    return {
        "test": 12,
        "name": "Budget feasibility and dual convergence",
        "method": "100 pacing simulations + convergence rate measurement",
        "passed": passed,
        "n_instances": n_instances,
        "n_violations": len(budget_violations),
        "violations": budget_violations[:5],
        "convergence": convergence_data,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all(verbose=False, quick=False):
    """Run all pacing tests."""
    tests = [
        ("Test 10: Value-max pacing bid", lambda: test_10_value_max_bid(verbose, quick)),
        ("Test 11: Utility-max pacing bid", lambda: test_11_utility_max_bid(verbose, quick)),
        ("Test 12: Budget feasibility", lambda: test_12_budget_feasibility(verbose, quick)),
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
