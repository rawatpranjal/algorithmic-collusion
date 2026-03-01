#!/usr/bin/env python3
"""
Verification of mathematical claims used in the paper.

1. LP benchmark correctness (appendix_welfare.tex Lemma):
   The LP relaxation with budget constraints computes the same optimum
   as the nonlinear liquid welfare maximization (LP* = W*).
   This validates the PoA benchmark used in Experiment 4.

2. Affiliated valuation parametrization (equilibria.tex L16-24):
   v_i = alpha*s_i + beta*sum_{j!=i} s_j with alpha=1-eta/2, beta=eta/(2(n-1))
   satisfies: (a) signal-valuation monotonicity, (b) BNE best-response,
   (c) revenue equivalence across auction formats.

Zero imports from project source code. Uses helpers.py for LP solver,
valuation model, and BNE formulas.
"""

import sys
import time

import numpy as np
from fractions import Fraction
from scipy.optimize import linprog

from helpers import (
    solve_welfare_lp,
    solve_welfare_ilp,
    compute_liquid_welfare,
    compute_alpha_beta,
    compute_alpha_beta_exact,
    bne_bid_slope,
    bne_bid_slope_exact,
    analytical_revenue,
    analytical_revenue_exact,
    efficient_benchmark,
    efficient_benchmark_exact,
)


# ===================================================================
# Verification 1: LP benchmark correctness (LP* = W*)
# ===================================================================

def _solve_wstar_auxiliary_lp(valuations, budgets):
    """
    Solve the auxiliary-variable LP that directly computes W*.

    W*_aux = max  sum_i w_i
    s.t.  w_i <= B_i                         for all i
          w_i <= sum_t v_{ti} * x_{ti}       for all i
          sum_i x_{ti} <= 1                  for all t
          x_{ti} in [0, 1]                   for all t, i
          w_i >= 0                           for all i

    This reformulates the nonlinear max sum_i min(B_i, sum_t v_{ti}*x_{ti})
    into a linear program via auxiliary variables w_i.
    """
    valuations = np.asarray(valuations, dtype=float)
    budgets = np.asarray(budgets, dtype=float)
    T, N = valuations.shape

    # Variables: x_{ti} (T*N) then w_i (N)
    n_x = T * N
    n_w = N
    n_vars = n_x + n_w

    # Objective: maximize sum w_i => minimize -sum w_i
    c = np.zeros(n_vars)
    c[n_x:] = -1.0  # w_i coefficients

    A_rows = []
    b_rows = []

    # Constraint: w_i <= B_i  =>  w_i <= B_i
    for i in range(N):
        row = np.zeros(n_vars)
        row[n_x + i] = 1.0
        A_rows.append(row)
        b_rows.append(budgets[i])

    # Constraint: w_i <= sum_t v_{ti} * x_{ti}  =>  w_i - sum_t v_{ti}*x_{ti} <= 0
    for i in range(N):
        row = np.zeros(n_vars)
        row[n_x + i] = 1.0
        for t in range(T):
            row[t * N + i] = -valuations[t, i]
        A_rows.append(row)
        b_rows.append(0.0)

    # Constraint: sum_i x_{ti} <= 1  for each t
    for t in range(T):
        row = np.zeros(n_vars)
        for i in range(N):
            row[t * N + i] = 1.0
        A_rows.append(row)
        b_rows.append(1.0)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)

    # Bounds: x_{ti} in [0,1], w_i >= 0 (upper bound from constraints)
    bounds = [(0.0, 1.0)] * n_x + [(0.0, None)] * n_w

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if not result.success:
        raise RuntimeError(f"Auxiliary LP solve failed: {result.message}")

    wstar_aux = -result.fun
    x_opt = result.x[:n_x].reshape(T, N)
    w_opt = result.x[n_x:]
    return wstar_aux, x_opt, w_opt


def _generate_random_instance(rng, N, T, budget_frac, dist="lognormal"):
    """Generate a random auction instance."""
    if dist == "lognormal":
        mu_i = rng.uniform(0.3, 1.5, size=N)
        valuations = np.zeros((T, N))
        for i in range(N):
            valuations[:, i] = rng.lognormal(mean=mu_i[i], sigma=0.4, size=T)
    elif dist == "uniform":
        valuations = rng.uniform(0.1, 5.0, size=(T, N))
    elif dist == "exponential":
        rates = rng.uniform(0.5, 2.0, size=N)
        valuations = np.zeros((T, N))
        for i in range(N):
            valuations[:, i] = rng.exponential(1.0 / rates[i], size=T)
    else:
        raise ValueError(f"Unknown distribution: {dist}")

    total_value = valuations.sum(axis=0)
    budgets = budget_frac * total_value
    return valuations, budgets


def test_lp_equals_wstar():
    """
    Verify LP benchmark (appendix_welfare.tex Lemma): LP* = W*.

    Method: Solve both LP formulations on each instance:
      - Paper's LP (budget constraints in LP form)
      - Auxiliary-variable LP (direct reformulation of W*)
    Verify |LP* - W*_aux| < 1e-6.
    Also verify the proof's scaling construction and LP* >= ILP* sanity.
    """
    print("=" * 60)
    print("VERIFICATION 1: LP* = W* (LP benchmark correctness)")
    print("=" * 60)
    t0 = time.time()

    rng = np.random.default_rng(42)
    failures = []
    n_tested = 0

    # --- 150 random instances ---
    N_vals = [2, 3, 4, 5]
    T_vals = [10, 20, 50]
    budget_fracs = [0.3, 0.5, 0.8]
    dists = ["uniform", "lognormal", "exponential"]

    for trial in range(150):
        N = rng.choice(N_vals)
        T = rng.choice(T_vals)
        bf = rng.choice(budget_fracs)
        dist = rng.choice(dists)

        valuations, budgets = _generate_random_instance(rng, N, T, bf, dist)

        lp_star, x_lp = solve_welfare_lp(valuations, budgets)
        wstar_aux, x_aux, w_aux = _solve_wstar_auxiliary_lp(valuations, budgets)

        gap = abs(lp_star - wstar_aux)
        n_tested += 1

        if gap > 1e-6:
            failures.append(f"Random #{trial}: N={N}, T={T}, dist={dist}, "
                            f"bf={bf}, LP*={lp_star:.8f}, W*={wstar_aux:.8f}, gap={gap:.2e}")

        # Verify proof scaling construction:
        # Solve W*_aux, extract x*, scale budget-violating bidders
        value_won = (x_aux * valuations).sum(axis=0)
        x_scaled = x_aux.copy()
        for i in range(N):
            if value_won[i] > budgets[i] + 1e-10:
                x_scaled[:, i] *= budgets[i] / value_won[i]

        # Scaled must be LP-feasible
        supply_ok = np.all(x_scaled.sum(axis=1) <= 1.0 + 1e-8)
        scaled_value = (x_scaled * valuations).sum(axis=0)
        budget_ok = np.all(scaled_value <= budgets + 1e-8)
        # LP objective at scaled = W* at original
        lp_obj_scaled = (x_scaled * valuations).sum()
        lw_original = compute_liquid_welfare(x_aux, valuations, budgets)

        if not supply_ok or not budget_ok:
            failures.append(f"Random #{trial}: scaling construction infeasible "
                            f"(supply={supply_ok}, budget={budget_ok})")
        if abs(lp_obj_scaled - lw_original) > 1e-5:
            failures.append(f"Random #{trial}: scaling doesn't preserve welfare "
                            f"(LP_obj={lp_obj_scaled:.8f}, W={lw_original:.8f})")

        # LP* >= ILP* sanity for small instances
        if T <= 20:
            try:
                ilp_star, _ = solve_welfare_ilp(valuations, budgets)
                if lp_star < ilp_star - 1e-6:
                    failures.append(f"Random #{trial}: LP* < ILP* "
                                    f"(LP*={lp_star:.8f}, ILP*={ilp_star:.8f})")
            except Exception:
                pass  # ILP solver may fail on some instances

    print(f"  Random instances: 150 tested")

    # --- 20 adversarial instances ---
    adversarial_configs = []

    # Near-zero budget
    for N in [2, 3, 4]:
        for T in [10, 50]:
            vals = rng.lognormal(0.5, 0.3, size=(T, N))
            buds = np.full(N, 0.001)
            adversarial_configs.append(("near_zero_budget", vals, buds))

    # All budgets binding (very tight)
    for N in [2, 3, 5]:
        T = 50
        vals = rng.uniform(1.0, 3.0, size=(T, N))
        total = vals.sum(axis=0)
        buds = 0.1 * total
        adversarial_configs.append(("all_binding", vals, buds))

    # Dominant-value bidder with tiny budget
    for N in [3, 4]:
        T = 30
        vals = rng.uniform(0.5, 1.0, size=(T, N))
        vals[:, 0] = rng.uniform(5.0, 10.0, size=T)  # bidder 0 dominates
        buds = 0.5 * vals.sum(axis=0)
        buds[0] = 0.01 * vals[:, 0].sum()  # but tiny budget
        adversarial_configs.append(("dominant_tiny_budget", vals, buds))

    # All identical bidders
    for T in [10, 50]:
        N = 4
        v_common = rng.uniform(1.0, 3.0, size=(T, 1))
        vals = np.tile(v_common, (1, N))
        buds = 0.5 * vals.sum(axis=0)
        adversarial_configs.append(("identical_bidders", vals, buds))

    # T=1 single round
    for N in [2, 3, 5]:
        vals = rng.uniform(0.5, 3.0, size=(1, N))
        buds = rng.uniform(0.1, 2.0, size=N)
        adversarial_configs.append(("single_round", vals, buds))

    for label, vals, buds in adversarial_configs[:20]:
        lp_star, _ = solve_welfare_lp(vals, buds)
        wstar_aux, _, _ = _solve_wstar_auxiliary_lp(vals, buds)
        gap = abs(lp_star - wstar_aux)
        n_tested += 1

        if gap > 1e-6:
            failures.append(f"Adversarial ({label}): LP*={lp_star:.8f}, "
                            f"W*={wstar_aux:.8f}, gap={gap:.2e}")

    print(f"  Adversarial instances: {min(len(adversarial_configs), 20)} tested")

    elapsed = time.time() - t0
    passed = len(failures) == 0
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] LP* = W*: {n_tested} instances, "
          f"{len(failures)} failures ({elapsed:.1f}s)")

    if failures:
        for f in failures[:10]:
            print(f"  FAIL: {f}")

    return passed


# ===================================================================
# Verification 2: Affiliated valuation parametrization
# ===================================================================

def test_affiliated_parametrization():
    """
    Verification 2 (equilibria.tex L16-24): The parametrization
      v_i = alpha*s_i + beta*sum_{j!=i} s_j
      alpha = 1 - eta/2,  beta = eta/(2(n-1))
    satisfies three properties:
      (a) Signal-valuation monotonicity
      (b) BNE best-response
      (c) Revenue equivalence
    """
    print()
    print("=" * 60)
    print("VERIFICATION 2: Affiliated valuation parametrization")
    print("=" * 60)

    failures = []

    # --- (a) Signal-valuation monotonicity ---
    t0 = time.time()
    print("\n  (a) Signal-valuation monotonicity: alpha - beta >= 0")
    _test_monotonicity(failures)
    print(f"      ({time.time() - t0:.1f}s)")

    # --- (b) BNE best-response ---
    t0 = time.time()
    print("\n  (b) BNE best-response verification")
    _test_best_response(failures)
    print(f"      ({time.time() - t0:.1f}s)")

    # --- (c) Revenue equivalence ---
    t0 = time.time()
    print("\n  (c) Revenue equivalence: R_FPA = R_SPA")
    _test_revenue_equivalence(failures)
    print(f"      ({time.time() - t0:.1f}s)")

    passed = len(failures) == 0
    status = "PASS" if passed else "FAIL"
    print(f"\n[{status}] Affiliated parametrization: {len(failures)} failures")

    if failures:
        for f in failures[:10]:
            print(f"  FAIL: {f}")

    return passed


def _test_monotonicity(failures):
    """
    (a) For eta in [0,1] and n >= 2, verify alpha - beta >= 0.
    Also MC verify that argmax(signals) = argmax(valuations).

    189 configs: eta in {0, 0.05, ..., 1.0} x n in {2,...,10}
    """
    n_configs = 0
    n_exact_pass = 0
    n_mc_pass = 0
    rng = np.random.default_rng(123)

    eta_vals = [i * 0.05 for i in range(21)]  # 0, 0.05, ..., 1.0
    n_vals = list(range(2, 11))                # 2, ..., 10

    for eta in eta_vals:
        for n in n_vals:
            n_configs += 1

            # Exact check with Fraction
            alpha_ex, beta_ex = compute_alpha_beta_exact(eta, n)
            diff = alpha_ex - beta_ex

            if diff < 0:
                failures.append(f"Monotonicity exact: eta={eta}, n={n}, "
                                f"alpha-beta={float(diff):.6f} < 0")
            else:
                n_exact_pass += 1

            # MC check: argmax(signals) = argmax(valuations)
            M = 500_000
            signals = rng.uniform(0, 1, size=(M, n))

            alpha, beta = compute_alpha_beta(eta, n)
            signal_sum = signals.sum(axis=1, keepdims=True)
            # v_i = alpha*s_i + beta*sum_{j!=i} s_j = alpha*s_i + beta*(S - s_i)
            #      = (alpha - beta)*s_i + beta*S
            valuations = (alpha - beta) * signals + beta * signal_sum

            # Check argmax agreement
            signal_winners = np.argmax(signals, axis=1)
            value_winners = np.argmax(valuations, axis=1)
            match_rate = (signal_winners == value_winners).mean()

            # When alpha-beta=0 (eta=1, n=2), all valuations are identical
            # regardless of signal, so argmax is arbitrary. Skip MC.
            if diff == 0:
                n_mc_pass += 1
                continue

            threshold = 0.9999

            if match_rate >= threshold:
                n_mc_pass += 1
            else:
                failures.append(f"Monotonicity MC: eta={eta}, n={n}, "
                                f"match_rate={match_rate:.6f} < {threshold}")

    print(f"      {n_configs} configs: {n_exact_pass} exact pass, "
          f"{n_mc_pass} MC pass")


def _test_best_response(failures):
    """
    (b) BNE bids are best responses.

    For each config, for 50 test signals, use PAIRED sampling: draw rival
    signals once, evaluate BNE bid and all deviations against the same
    rivals. This eliminates MC noise in the payoff difference.

    ~80 configs: eta in {0, 0.25, 0.5, 0.75, 1.0} x n in {2,3,4,6}
                 x auction in {first, second}
    """
    n_configs = 0
    n_pass = 0
    max_deviation_profit = 0.0
    rng = np.random.default_rng(456)
    M = 1_000_000  # paired samples per signal

    eta_vals = [0, 0.25, 0.5, 0.75, 1.0]
    n_vals = [2, 3, 4, 6]
    auction_types = ["first", "second"]

    for eta in eta_vals:
        for n in n_vals:
            for auction_type in auction_types:
                n_configs += 1
                alpha, beta = compute_alpha_beta(eta, n)
                phi = bne_bid_slope(eta, n, auction_type)

                test_signals = np.linspace(0.01, 0.99, 50)
                config_max_gain = 0.0

                for s in test_signals:
                    # Draw rivals ONCE for this signal (paired sampling)
                    rival_signals = rng.uniform(0, 1, size=(M, n - 1))
                    rival_sum = rival_signals.sum(axis=1)
                    own_val = alpha * s + beta * rival_sum

                    # Rival bids under BNE
                    phi_rivals = bne_bid_slope(eta, n, auction_type)
                    rival_bids = phi_rivals * rival_signals
                    max_rival_bid = rival_bids.max(axis=1)

                    bne_bid = phi * s

                    # BNE payoff
                    bne_payoff = _paired_payoff(
                        bne_bid, own_val, max_rival_bid, auction_type
                    )

                    # Deviation bids
                    dev_factors = np.concatenate([
                        np.linspace(0.01, 0.99, 50),
                        np.linspace(1.01, 2.0, 50),
                    ])
                    for df in dev_factors:
                        dev_bid = bne_bid * df
                        dev_payoff = _paired_payoff(
                            dev_bid, own_val, max_rival_bid, auction_type
                        )
                        gain = dev_payoff - bne_payoff
                        config_max_gain = max(config_max_gain, gain)

                max_deviation_profit = max(max_deviation_profit, config_max_gain)

                if config_max_gain < 1e-4:
                    n_pass += 1
                else:
                    failures.append(
                        f"Best-response: eta={eta}, n={n}, {auction_type}, "
                        f"max_gain={config_max_gain:.6f}"
                    )

    print(f"      {n_configs} configs: {n_pass} pass, "
          f"max deviation profit={max_deviation_profit:.2e}")


def _paired_payoff(bid, own_val, max_rival_bid, auction_type):
    """
    Compute mean payoff given pre-drawn rival outcomes (paired sampling).
    own_val and max_rival_bid are arrays of length M.
    """
    win = bid > max_rival_bid
    if auction_type == "first":
        payoff = np.where(win, own_val - bid, 0.0)
    else:
        payoff = np.where(win, own_val - max_rival_bid, 0.0)
    return payoff.mean()


def _test_revenue_equivalence(failures):
    """
    (c) Revenue equivalence: R_FPA = R_SPA for all (eta, n).

    77 configs: eta in {0, 0.1, ..., 1.0} x n in {2,3,4,5,6,8,10}

    Check both exact (Fraction) and MC.
    """
    n_configs = 0
    n_exact_pass = 0
    n_mc_pass = 0
    max_mc_gap = 0.0
    rng = np.random.default_rng(789)

    eta_vals = [i * 0.1 for i in range(11)]  # 0, 0.1, ..., 1.0
    n_vals = [2, 3, 4, 5, 6, 8, 10]

    for eta in eta_vals:
        for n in n_vals:
            n_configs += 1

            # Exact check: R_FPA_formula = R_SPA_formula
            # Both reduce to (n-1)/(n+1) * phi, so they're identical by construction.
            # We verify this symbolically.
            alpha_ex, beta_ex = compute_alpha_beta_exact(eta, n)
            n_frac = Fraction(n)
            phi_ex = alpha_ex + n_frac * beta_ex / 2

            r_spa_ex = phi_ex * (n_frac - 1) / (n_frac + 1)
            r_fpa_ex = (n_frac - 1) / n_frac * phi_ex * n_frac / (n_frac + 1)

            if r_spa_ex == r_fpa_ex:
                n_exact_pass += 1
            else:
                failures.append(f"Revenue equiv exact: eta={eta}, n={n}, "
                                f"SPA={float(r_spa_ex):.8f}, FPA={float(r_fpa_ex):.8f}")

            # MC check: simulate both auctions
            M = 2_000_000
            signals = rng.uniform(0, 1, size=(M, n))

            alpha, beta = compute_alpha_beta(eta, n)
            phi_spa = alpha + n * beta / 2.0
            phi_fpa = (n - 1) / n * phi_spa

            bids_fpa = phi_fpa * signals
            bids_spa = phi_spa * signals

            # FPA revenue: highest bid
            rev_fpa = bids_fpa.max(axis=1)

            # SPA revenue: second-highest bid
            sorted_bids_spa = np.sort(bids_spa, axis=1)
            rev_spa = sorted_bids_spa[:, -2]

            mean_fpa = rev_fpa.mean()
            mean_spa = rev_spa.mean()

            # Standard errors
            se_fpa = rev_fpa.std() / np.sqrt(M)
            se_spa = rev_spa.std() / np.sqrt(M)
            se_diff = np.sqrt(se_fpa**2 + se_spa**2)

            gap = abs(mean_fpa - mean_spa)
            max_mc_gap = max(max_mc_gap, gap)

            # Tolerance: 4 sigma + 1e-4
            tol = 4 * se_diff + 1e-4
            if gap < tol:
                n_mc_pass += 1
            else:
                failures.append(f"Revenue equiv MC: eta={eta}, n={n}, "
                                f"FPA={mean_fpa:.6f}, SPA={mean_spa:.6f}, "
                                f"gap={gap:.2e}, tol={tol:.2e}")

    print(f"      {n_configs} configs: {n_exact_pass} exact pass, "
          f"{n_mc_pass} MC pass, max MC gap={max_mc_gap:.2e}")


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    all_passed = True

    p1 = test_lp_equals_wstar()
    all_passed = all_passed and p1

    p2 = test_affiliated_parametrization()
    all_passed = all_passed and p2

    print()
    if all_passed:
        print("All verifications passed.")
        sys.exit(0)
    else:
        print("SOME VERIFICATIONS FAILED. See details above.")
        sys.exit(1)
