#!/usr/bin/env python3
"""
Numerical checks for Exp2 BNE in the affiliated linear model.

Verifies two things for given (n, eta):
 1) Best-response optimality of the analytical SPA/FPA bidding rules
    via Monte Carlo utility comparisons around the equilibrium bid.
 2) Expected revenue under BNE equals the closed-form benchmark.

Model:
  s_i ~ U[0,1] iid, v_i = alpha * s_i + beta_sum * sum_{j!=i} s_j,
  with alpha = 1 - 0.5*eta, beta_sum = 0.5*eta/(n-1).

SPA equilibrium bid: b_spa(s) = phi * s,    phi = alpha + (n*beta_sum)/2.
FPA equilibrium bid: b_fpa(s) = theta * s,  theta = ((n-1)/n) * phi.

Revenue benchmark (both formats): R_BNE = ((n-1)/(n+1)) * phi.
"""
import argparse
import numpy as np


def params(n: int, eta: float):
    alpha = 1.0 - 0.5 * eta
    beta_sum = 0.5 * eta / max(n - 1, 1)
    phi = alpha + (n * beta_sum) / 2.0
    theta = ((n - 1) / n) * phi
    return alpha, beta_sum, phi, theta


def simulate_revenue(n: int, eta: float, auction: str, rounds: int = 200_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    alpha, beta_sum, phi, theta = params(n, eta)

    if auction == "spa":
        factor = phi
    elif auction == "fpa":
        factor = theta
    else:
        raise ValueError("auction must be 'spa' or 'fpa'")

    # Signals and bids
    s = rng.random((rounds, n))
    bids = factor * s
    # Winning price
    sort_idx = np.argsort(bids, axis=1)
    # max bid in last column
    max_bid = bids[np.arange(rounds), sort_idx[:, -1]]

    if auction == "fpa":
        price = max_bid
    else:  # spa
        # second-highest (or max in case of tie)
        second_bid = bids[np.arange(rounds), sort_idx[:, -2]]
        # if there is a strict unique max, price is second-highest; on ties it's equal to max
        # Our sort handles both; the standard definition uses second order statistic.
        price = second_bid

    return price.mean()


def check_bne_best_response_spa(n: int, eta: float, num_s: int = 25, inner_rounds: int = 50_000, seed: int = 1):
    """Pointwise best-response check around b_eq(s) for SPA.

    For a grid of s in [0,1], compare expected utility at b' = k * b_eq(s)
    for k in a small set of multiplicative deviations. Returns fraction of s
    points where k=1.0 attains the maximum (up to 1e-5 tolerance).
    """
    rng = np.random.default_rng(seed)
    alpha, beta_sum, phi, _ = params(n, eta)

    ks = np.array([0.7, 0.85, 1.0, 1.15, 1.3])
    s_grid = np.linspace(0.02, 0.98, num_s)
    wins = 0

    for s_i in s_grid:
        # Draw others' signals
        others = rng.random((inner_rounds, n - 1))
        y1 = others.max(axis=1)  # max rival signal

        # Expected utility for each k
        util = []
        for k in ks:
            bdev = k * (phi * s_i)
            # Win if bdev >= phi * y1
            win_mask = (bdev >= phi * y1)
            # Value conditional on y1 with other (n-2) signals iid U[0,y1] in expectation:
            # We approximate by using the actual draws to capture finite-sample behavior.
            # v_i = alpha * s_i + beta_sum * sum(others)
            v_i = alpha * s_i + beta_sum * others.sum(axis=1)
            # SPA price is phi * y1 (second-highest bid), independent of own bid when win
            payoff = np.where(win_mask, v_i - phi * y1, 0.0)
            util.append(payoff.mean())
        util = np.array(util)

        # Count if k=1.0 is (near) maximal
        if np.max(util) - util[2] <= 1e-5:
            wins += 1

    return wins / len(s_grid)


def check_bne_best_response_fpa(n: int, eta: float, num_s: int = 25, inner_rounds: int = 50_000, seed: int = 2):
    """Pointwise best-response check around b_eq(s) for FPA.

    Same structure as SPA, but payment equals own bid conditional on winning.
    """
    rng = np.random.default_rng(seed)
    alpha, beta_sum, phi, theta = params(n, eta)

    ks = np.array([0.7, 0.85, 1.0, 1.15, 1.3])
    s_grid = np.linspace(0.02, 0.98, num_s)
    wins = 0

    for s_i in s_grid:
        others = rng.random((inner_rounds, n - 1))
        y1 = others.max(axis=1)

        util = []
        for k in ks:
            bdev = k * (theta * s_i)
            win_mask = (bdev >= theta * y1)
            v_i = alpha * s_i + beta_sum * others.sum(axis=1)
            payoff = np.where(win_mask, v_i - bdev, 0.0)
            util.append(payoff.mean())
        util = np.array(util)

        if np.max(util) - util[2] <= 1e-5:
            wins += 1

    return wins / len(s_grid)


def closed_form_revenue(n: int, eta: float):
    alpha, beta_sum, phi, _ = params(n, eta)
    return ((n - 1) / (n + 1)) * phi


def parse_args():
    ap = argparse.ArgumentParser(description="Check BNE optimality and revenue for Exp2 model")
    ap.add_argument("--n", nargs="*", type=int, default=[2, 4, 5, 6], help="List of bidder counts")
    ap.add_argument("--eta", nargs="*", type=float, default=[0.0, 0.5, 0.75, 1.0], help="List of eta values")
    ap.add_argument("--rounds", type=int, default=400_000, help="Monte Carlo rounds for revenue")
    ap.add_argument("--inner-rounds", type=int, default=100_000, help="Inner draws for BR check")
    ap.add_argument("--num-s", type=int, default=41, help="Number of s-grid points for BR check")
    ap.add_argument("--seeds", type=int, default=5, help="Number of seeds for CI aggregation")
    return ap.parse_args()


def agg_mean_ci(samples):
    arr = np.array(samples, dtype=float)
    m = arr.mean()
    se = arr.std(ddof=1) / max(len(arr) ** 0.5, 1.0)
    return m, (m - 1.96 * se, m + 1.96 * se)


def main():
    args = parse_args()
    configs = [(n, e) for n in args.n for e in args.eta]

    header = (
        "n  eta   BR_SPA  BR_FPA  "
        "EmpRev_SPA  [95% CI]       EmpRev_FPA  [95% CI]       ClosedForm"
    )
    print(header)
    for n, eta in configs:
        # BR checks: average fraction across seeds
        br_spa_vals = [
            check_bne_best_response_spa(n, eta, num_s=args.num_s, inner_rounds=args.inner_rounds, seed=100 + s)
            for s in range(args.seeds)
        ]
        br_fpa_vals = [
            check_bne_best_response_fpa(n, eta, num_s=args.num_s, inner_rounds=args.inner_rounds, seed=200 + s)
            for s in range(args.seeds)
        ]
        br_spa_mean, _ = agg_mean_ci(br_spa_vals)
        br_fpa_mean, _ = agg_mean_ci(br_fpa_vals)

        # Revenue CIs via multiple seeds
        emp_spa_vals = [simulate_revenue(n, eta, "spa", rounds=args.rounds, seed=10 + s) for s in range(args.seeds)]
        emp_fpa_vals = [simulate_revenue(n, eta, "fpa", rounds=args.rounds, seed=20 + s) for s in range(args.seeds)]
        m_spa, ci_spa = agg_mean_ci(emp_spa_vals)
        m_fpa, ci_fpa = agg_mean_ci(emp_fpa_vals)
        cf = closed_form_revenue(n, eta)

        print(
            f"{n:<2d} {eta:<4.2f}  {br_spa_mean:>6.2f}  {br_fpa_mean:>6.2f}   "
            f"{m_spa:>10.4f}  [{ci_spa[0]:.4f}, {ci_spa[1]:.4f}]   "
            f"{m_fpa:>10.4f}  [{ci_fpa[0]:.4f}, {ci_fpa[1]:.4f}]   "
            f"{cf:>10.4f}"
        )


if __name__ == "__main__":
    main()
