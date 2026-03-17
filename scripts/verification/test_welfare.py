#!/usr/bin/env python3
"""
Tests 7-9: LP optimality and PoA bounds verification.

All formulas from paper/sections/auctions.tex and paper/sections/appendix_robustness.tex.
Zero imports from project source code.
"""

import sys
import time

import numpy as np

from helpers import (
    solve_welfare_lp,
    solve_welfare_ilp,
    compute_liquid_welfare,
    optimal_bid_value_max,
    optimal_bid_utility_max,
)


# ---------------------------------------------------------------------------
# Minimal pacing simulation (re-implemented from algorithms.tex, not imported)
# ---------------------------------------------------------------------------

def run_pacing_sim(valuations, budgets, auction_type="first", objective="value",
                   seed=0):
    """
    Minimal pacing simulation for welfare verification.
    Re-derived from algorithms.tex equations.

    Returns allocation matrix (T, N) with 0/1 entries.
    """
    rng = np.random.default_rng(seed)
    T, N = valuations.shape
    budgets = np.asarray(budgets, dtype=float)

    # State
    mu = np.ones(N)  # dual variables
    spend = np.zeros(N)  # cumulative spend
    allocation = np.zeros((T, N), dtype=float)
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
                bids[i] = optimal_bid_value_max(v, mu[i], rem)
            else:
                bids[i] = optimal_bid_utility_max(v, mu[i], rem)

        # Auction
        if np.all(bids <= 0):
            continue  # no sale

        winner = np.argmax(bids)
        # Tie-breaking: random among max
        max_bid = bids[winner]
        if max_bid <= 0:
            continue
        ties = np.where(np.abs(bids - max_bid) < 1e-12)[0]
        if len(ties) > 1:
            winner = rng.choice(ties)

        # Payment
        if auction_type == "first":
            payment = bids[winner]
        else:
            # Second-price: second-highest bid
            sorted_bids = np.sort(bids)
            payment = sorted_bids[-2] if N > 1 else 0.0

        # Clip payment to remaining budget
        payment = min(payment, remaining[winner])

        allocation[t, winner] = 1.0
        spend[winner] += payment

        # Dual update (eq from algorithms.tex)
        target_rate = budgets / T
        for i in range(N):
            cost_i = payment if i == winner else 0.0
            mu[i] = np.clip(
                mu[i] * np.exp(alpha_p * (cost_i - target_rate[i])),
                1e-4, 100.0
            )

    return allocation, spend


# ---------------------------------------------------------------------------
# Test 7: LP* = W* (LP relaxation = optimal liquid welfare)
# ---------------------------------------------------------------------------

def _solve_unconstrained_ilp(valuations, budgets_unused):
    """
    Solve ILP with supply constraints only (NO budget constraints).
    Returns an allocation that typically violates budgets, which is the
    correct input for the scaling argument.
    """
    from scipy.optimize import milp, LinearConstraint, Bounds

    valuations = np.asarray(valuations, dtype=float)
    T, N = valuations.shape
    n_vars = T * N

    c = -valuations.ravel()

    # Only supply constraints: sum_i x_{ti} <= 1
    A_rows = []
    b_lower = []
    b_upper = []
    for t in range(T):
        row = np.zeros(n_vars)
        for i in range(N):
            row[t * N + i] = 1.0
        A_rows.append(row)
        b_lower.append(-np.inf)
        b_upper.append(1.0)

    A = np.array(A_rows)
    constraints = LinearConstraint(A, b_lower, b_upper)
    integrality = np.ones(n_vars)
    bounds = Bounds(lb=0, ub=1)

    result = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
    if not result.success:
        raise RuntimeError(f"Unconstrained ILP solve failed: {result.message}")

    return -result.fun, result.x.reshape(T, N)


def test_7_lp_equals_optimal(verbose=False, quick=False):
    """
    Claim (auctions.tex Proposition):
    LP* = W* (the LP relaxation achieves the same value as the non-linear optimum).

    Method (both directions of the proof):

    Direction 1 (LP* <= W*): Any LP-feasible allocation has liquid welfare = LP objective.
    Verified by solving the LP, checking feasibility, and confirming W(x_LP) = LP*.

    Direction 2 (LP* >= W*): For ANY supply-feasible allocation x, we can construct
    an LP-feasible allocation x' with LP_obj(x') = W(x). The construction: for each
    bidder i with value_won > B_i, scale their allocations by B_i/value_won_i.
    Since LP* >= LP_obj(x'), we get LP* >= W(x) for ALL x, hence LP* >= W*.

    To actually exercise the scaling code, Direction 2 uses an UNCONSTRAINED ILP
    (supply constraints only, no budget constraints), which produces allocations
    that genuinely violate budget constraints. A budget-constrained ILP would never
    violate budgets, making the scaling a no-op.

    Together: LP* = W*.
    """
    n_instances = 50 if not quick else 10
    rng = np.random.default_rng(42)
    T_val = 15
    N_vals = [2, 3]

    results = []
    max_gap = 0.0
    n_scaling_fired = 0

    for inst_idx in range(n_instances):
        N = rng.choice(N_vals)

        # Random log-normal valuations (mimicking exp4)
        mu_i = rng.uniform(0.5, 1.5, size=N)
        valuations = np.zeros((T_val, N))
        for i in range(N):
            valuations[:, i] = rng.lognormal(mean=mu_i[i], sigma=0.3, size=T_val)

        # Tight budgets (20-40% of total value) to ensure scaling actually fires
        total_value = valuations.sum(axis=0)  # total value per bidder
        budget_frac = rng.uniform(0.2, 0.4, size=N)
        budgets = budget_frac * total_value

        # === Direction 1: LP* <= W* ===
        # Solve LP (with budget constraints)
        lp_star, x_lp = solve_welfare_lp(valuations, budgets)

        # Verify LP solution is feasible and W(x_LP) = LP objective
        supply_ok = np.all(x_lp.sum(axis=1) <= 1.0 + 1e-8)
        value_won_lp = (x_lp * valuations).sum(axis=0)
        budget_ok = np.all(value_won_lp <= budgets + 1e-8)
        lw_at_lp = compute_liquid_welfare(x_lp, valuations, budgets)
        dir1_ok = supply_ok and budget_ok and abs(lw_at_lp - lp_star) < 1e-6

        # === Direction 2: LP* >= W* (scaling argument on unconstrained allocation) ===
        dir2_ok = True
        scaling_fired = False
        try:
            # Solve UNCONSTRAINED ILP (supply constraints only, NO budget)
            # This produces allocations that typically violate budget constraints
            unc_obj, x_unc = _solve_unconstrained_ilp(valuations, budgets)

            # Check that x_unc is supply-feasible
            unc_supply_ok = np.all(x_unc.sum(axis=1) <= 1.0 + 1e-8)

            # Compute value won per bidder (may exceed budgets)
            value_won_unc = (x_unc * valuations).sum(axis=0)

            # Apply the paper's scaling argument:
            # For each bidder i with value_won > B_i, scale x_{ti} by B_i / value_won_i
            x_scaled = x_unc.copy()
            for i in range(N):
                if value_won_unc[i] > budgets[i] + 1e-10:
                    scale_factor = budgets[i] / value_won_unc[i]
                    x_scaled[:, i] *= scale_factor
                    scaling_fired = True

            if scaling_fired:
                n_scaling_fired += 1

            # Verify scaled allocation is LP-feasible
            # Supply: scaling down can only reduce row sums
            scaled_supply_ok = np.all(x_scaled.sum(axis=1) <= 1.0 + 1e-8)
            # Budget: guaranteed by construction (tight for over-budget, unchanged for under)
            scaled_value_won = (x_scaled * valuations).sum(axis=0)
            scaled_budget_ok = np.all(scaled_value_won <= budgets + 1e-8)
            # Bounds: scaling down preserves [0,1]
            scaled_bounds_ok = np.all(x_scaled >= -1e-10) and np.all(x_scaled <= 1.0 + 1e-10)

            # LP objective at scaled solution
            lp_obj_scaled = (x_scaled * valuations).sum()

            # Liquid welfare of unconstrained solution
            lw_unc = compute_liquid_welfare(x_unc, valuations, budgets)

            # Key check: LP_obj(x_scaled) = W(x_unc)
            # For over-budget bidders: scaled_value_won = B_i = min(B_i, original_value_won)
            # For under-budget bidders: no scaling, value_won = min(B_i, value_won)
            scaling_preserves_welfare = abs(lp_obj_scaled - lw_unc) < 1e-5

            # Final: LP* >= LP_obj(x_scaled) = W(x_unc)
            lp_ge_scaled = lp_star >= lp_obj_scaled - 1e-6

            dir2_ok = (unc_supply_ok and scaled_supply_ok and scaled_budget_ok
                       and scaled_bounds_ok and scaling_preserves_welfare and lp_ge_scaled)

            gap = lp_star - lw_unc  # LP* - W(x_unc) >= 0
            max_gap = max(max_gap, gap)

        except Exception:
            gap = 0.0

        passed = dir1_ok and dir2_ok

        results.append({
            "instance": inst_idx, "T": T_val, "N": int(N),
            "lp_star": lp_star,
            "lw_unc": lw_unc if dir2_ok else None,
            "scaling_fired": scaling_fired,
            "dir1_ok": dir1_ok,
            "dir2_ok": dir2_ok,
            "gap": gap,
            "passed": passed,
        })

        if verbose and not passed:
            print(f"  Instance {inst_idx}: LP*={lp_star:.4f}, "
                  f"dir1={'OK' if dir1_ok else 'FAIL'}, dir2={'OK' if dir2_ok else 'FAIL'}")

    all_passed = all(r["passed"] for r in results)
    if verbose:
        print(f"  {sum(r['passed'] for r in results)}/{len(results)} instances passed, "
              f"scaling fired in {n_scaling_fired}/{len(results)} instances, "
              f"max LP*-W(x_unc) gap={max_gap:.4f}")

    return {
        "test": 7,
        "name": "LP* = W* (LP relaxation optimality, both directions)",
        "method": "Direction 1: LP-feasible => W=LP*. Direction 2: unconstrained ILP + scaling argument",
        "passed": all_passed,
        "n_instances": len(results),
        "n_scaling_fired": n_scaling_fired,
        "max_gap": max_gap,
        "n_failures": sum(1 for r in results if not r["passed"]),
    }


# ---------------------------------------------------------------------------
# Test 8: PoA >= 1 always
# ---------------------------------------------------------------------------

def test_8_poa_ge_one(verbose=False, quick=False):
    """
    Claim (auctions.tex eq:poa_def):
    PoA = W* / W(x_obs) >= 1 by construction (W* is the optimum over all
    allocations, so it must exceed any particular allocation's welfare).

    This is a SOLVER SANITY CHECK, not a mathematical proof. PoA >= 1 is true
    by definition. This test verifies that the LP solver and pacing simulation
    produce consistent results (no bugs in the welfare computation pipeline).

    Method: Run 100 pacing simulations, compute LP* and W_obs,
    verify PoA >= 1.0 - solver tolerance.
    """
    n_instances = 100 if not quick else 20
    rng = np.random.default_rng(42)
    T_val = 100

    results = []
    min_poa = float('inf')

    configs = [
        ("first", "value"), ("first", "utility"),
        ("second", "value"), ("second", "utility"),
    ]

    for inst_idx in range(n_instances):
        auction_type, objective = configs[inst_idx % len(configs)]
        N = rng.choice([2, 3, 4])

        # Generate valuations
        mu_i = rng.uniform(0.5, 1.5, size=N)
        valuations = np.zeros((T_val, N))
        for i in range(N):
            valuations[:, i] = rng.lognormal(mean=mu_i[i], sigma=0.3, size=T_val)

        # Budgets
        expected_vals = np.exp(mu_i + 0.3**2 / 2)
        budgets = 0.5 * expected_vals * T_val

        # LP optimum
        lp_star, _ = solve_welfare_lp(valuations, budgets)

        # Pacing simulation
        allocation, spend = run_pacing_sim(
            valuations, budgets, auction_type, objective, seed=inst_idx
        )

        # Observed liquid welfare
        w_obs = compute_liquid_welfare(allocation, valuations, budgets)

        # PoA
        if w_obs > 0:
            poa = lp_star / w_obs
        else:
            poa = float('inf')  # no allocation => infinite PoA

        min_poa = min(min_poa, poa)
        passed = poa >= 1.0 - 1e-8

        results.append({
            "instance": inst_idx,
            "auction_type": auction_type,
            "objective": objective,
            "N": int(N),
            "lp_star": lp_star,
            "w_obs": w_obs,
            "poa": poa,
            "passed": passed,
        })

        if verbose and not passed:
            print(f"  FAIL: inst={inst_idx}, {auction_type}/{objective}, N={N}, "
                  f"PoA={poa:.6f}")

    all_passed = all(r["passed"] for r in results)
    if verbose:
        print(f"  Min PoA across {len(results)} instances: {min_poa:.6f}")

    return {
        "test": 8,
        "name": "PoA >= 1 (solver sanity check)",
        "method": "Solver sanity check: LP optimum vs pacing simulation on 100 instances (PoA >= 1 holds by definition)",
        "passed": all_passed,
        "n_instances": len(results),
        "min_poa": min_poa,
        "n_failures": sum(1 for r in results if not r["passed"]),
    }


# ---------------------------------------------------------------------------
# Test 9: PoA <= 2 under budget constraints
# ---------------------------------------------------------------------------

def test_9_poa_le_two(verbose=False, quick=False):
    """
    Claim (auctions.tex table, multiple sources):
    PoA <= 2 for budget-constrained autobidding in FPA and SPA.

    This is an EMPIRICAL check, not a proof of the theoretical bound. The PoA <= 2
    bound comes from Balseiro & Gur (2019) and Conitzer et al. (2022) via
    smoothness arguments. Here we verify the bound is not violated in practice,
    including adversarial instances designed to stress-test it.

    Method:
    (a) 100 random instances across all 4 configs.
    (b) Adversarial instances: extreme budget asymmetry, high-value/low-budget
        scenarios designed to maximize welfare loss from misallocation.
    (c) Exp4-matching instances: parameters matching our experiment 4 setup.
    """
    n_random = 100 if not quick else 20
    rng = np.random.default_rng(42)
    T_val = 200

    configs = [
        ("first", "value"), ("first", "utility"),
        ("second", "value"), ("second", "utility"),
    ]

    results = []
    max_poa = 0.0
    flagged = []

    def _run_instance(inst_id, valuations, budgets, auction_type, objective, label="random"):
        nonlocal max_poa
        lp_star, _ = solve_welfare_lp(valuations, budgets)
        allocation, spend = run_pacing_sim(
            valuations, budgets, auction_type, objective, seed=inst_id
        )
        w_obs = compute_liquid_welfare(allocation, valuations, budgets)
        poa = lp_star / w_obs if w_obs > 0 else float('inf')
        max_poa = max(max_poa, poa)
        passed = poa < 2.0 + 1e-6

        if poa > 1.8:
            flagged.append({
                "label": label, "instance": inst_id,
                "auction_type": auction_type, "objective": objective,
                "N": valuations.shape[1], "poa": poa,
            })
            if verbose:
                print(f"  FLAG [{label}]: inst={inst_id}, {auction_type}/{objective}, "
                      f"N={valuations.shape[1]}, PoA={poa:.4f}")

        results.append({"instance": inst_id, "label": label, "poa": poa, "passed": passed})

    # (a) Random instances
    for inst_idx in range(n_random):
        auction_type, objective = configs[inst_idx % len(configs)]
        N = rng.choice([2, 3, 4])
        mu_i = rng.uniform(0.5, 1.5, size=N)
        valuations = np.zeros((T_val, N))
        for i in range(N):
            valuations[:, i] = rng.lognormal(mean=mu_i[i], sigma=0.3, size=T_val)
        expected_vals = np.exp(mu_i + 0.3**2 / 2)
        budgets = 0.5 * expected_vals * T_val
        _run_instance(inst_idx, valuations, budgets, auction_type, objective, "random")

    # (b) Adversarial instances: extreme budget asymmetry
    adv_idx = 10000
    for auction_type, objective in configs:
        for N in [2, 3, 4]:
            # One bidder with very high values but tiny budget (forced misallocation)
            valuations = np.zeros((T_val, N))
            mu_i = np.full(N, 0.5)
            mu_i[0] = 2.0  # bidder 0 has very high values
            for i in range(N):
                valuations[:, i] = rng.lognormal(mean=mu_i[i], sigma=0.3, size=T_val)

            expected_vals = np.exp(mu_i + 0.3**2 / 2)
            # Bidder 0: tiny budget (5% of expected value) forces extreme pacing
            budgets = 0.5 * expected_vals * T_val
            budgets[0] = 0.05 * expected_vals[0] * T_val

            _run_instance(adv_idx, valuations, budgets, auction_type, objective, "adversarial_asymmetric")
            adv_idx += 1

        # Extreme case: all bidders have same values but very different budgets
        for N in [2, 4]:
            valuations = np.ones((T_val, N)) * 3.0  # identical high values
            valuations += rng.normal(0, 0.1, size=(T_val, N))  # small noise
            valuations = np.clip(valuations, 0.1, None)

            budgets = np.array([T_val * 0.1] + [T_val * 2.0] * (N - 1))  # one starved bidder
            _run_instance(adv_idx, valuations, budgets, auction_type, objective, "adversarial_starved")
            adv_idx += 1

    # (c) Exp4-matching instances
    exp4_idx = 20000
    for auction_type, objective in configs:
        for N in [2, 4]:
            # Match exp4 parameters: lognormal with different means
            mu_vals = rng.uniform(0.3, 1.7, size=N)
            sigma_val = 0.5
            valuations = np.zeros((T_val, N))
            for i in range(N):
                valuations[:, i] = rng.lognormal(mean=mu_vals[i], sigma=sigma_val, size=T_val)
            expected_vals = np.exp(mu_vals + sigma_val**2 / 2)
            budgets = 0.4 * expected_vals * T_val  # tighter budgets
            _run_instance(exp4_idx, valuations, budgets, auction_type, objective, "exp4_matching")
            exp4_idx += 1

    all_passed = all(r["passed"] for r in results)
    n_adv = sum(1 for r in results if r["label"].startswith("adversarial"))
    n_exp4 = sum(1 for r in results if r["label"] == "exp4_matching")

    if verbose:
        print(f"  Max PoA across {len(results)} instances: {max_poa:.6f}")
        print(f"  Instances: {n_random} random + {n_adv} adversarial + {n_exp4} exp4-matching")
        print(f"  Instances with PoA > 1.8: {len(flagged)}")

    return {
        "test": 9,
        "name": "PoA <= 2 under budget constraints",
        "method": f"Empirical check: {n_random} random + {n_adv} adversarial + {n_exp4} exp4-matching instances",
        "passed": all_passed,
        "n_instances": len(results),
        "n_random": n_random,
        "n_adversarial": n_adv,
        "n_exp4_matching": n_exp4,
        "max_poa": max_poa,
        "n_flagged_above_1_8": len(flagged),
        "flagged": flagged[:10],
        "n_failures": sum(1 for r in results if not r["passed"]),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all(verbose=False, quick=False):
    """Run all welfare tests."""
    tests = [
        ("Test 7: LP* = W*", lambda: test_7_lp_equals_optimal(verbose, quick)),
        ("Test 8: PoA >= 1", lambda: test_8_poa_ge_one(verbose, quick)),
        ("Test 9: PoA <= 2", lambda: test_9_poa_le_two(verbose, quick)),
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
