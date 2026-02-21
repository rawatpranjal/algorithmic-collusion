#!/usr/bin/env python3
"""
Debug script for Experiment 3 (LinUCB + CTS) algorithms.

Runs both algorithms side-by-side on identical signals/valuations and prints
a detailed per-round trace for verification. Does NOT modify exp3.py.

Usage:
    PYTHONPATH=src python3 scripts/debug_exp3.py
    PYTHONPATH=src python3 scripts/debug_exp3.py --rounds 200 --verbose-rounds 5
    PYTHONPATH=src python3 scripts/debug_exp3.py --reserve 0.3 --auction first
"""

import argparse
import numpy as np
from experiments.exp3 import (
    get_valuation, get_rewards, LinUCB, ContextualThompsonSampling,
    EXPLORATION_MAP,
)


def build_context(bidder_i, signals, past_bids, past_winner_bid, win_counts,
                  ep, n_bidders, context_richness):
    ctx = [signals[bidder_i]]
    if context_richness == "rich":
        med_bid = np.median(np.delete(past_bids, bidder_i)) if n_bidders > 1 else 0.0
        ctx.append(med_bid)
        ctx.append(past_winner_bid)
        ctx.append(win_counts[bidder_i] / max(1, ep))
    return np.array(ctx)


def fmt_arr(arr, decimals=3):
    return "[" + ", ".join(f"{x:.{decimals}f}" for x in arr) + "]"


def run_debug(args):
    np.random.seed(args.seed)

    n_bid_bins = 6
    n_val_bins = 6
    actions = np.linspace(0, 1, n_bid_bins)
    eta = args.eta
    alpha_val = 1.0 - 0.5 * eta
    beta_val = 0.5 * eta

    context_dim = 4 if args.context == "rich" else 1

    # Create bandits for both algorithms
    linucb_c = EXPLORATION_MAP[("linucb", args.exploration)]
    cts_sigma2 = EXPLORATION_MAP[("thompson", args.exploration)]

    linucb_bandits = [
        LinUCB(n_actions=n_bid_bins, context_dim=context_dim, c=linucb_c, lam=args.lam)
        for _ in range(args.n_bidders)
    ]
    cts_bandits = [
        ContextualThompsonSampling(
            n_actions=n_bid_bins, context_dim=context_dim,
            sigma2=cts_sigma2, lam=args.lam,
            rng=np.random.default_rng(args.seed + i)
        )
        for i in range(args.n_bidders)
    ]

    # State for context building (separate per algorithm)
    lu_past_bids = np.zeros(args.n_bidders)
    lu_past_winner_bid = 0.0
    lu_win_counts = np.zeros(args.n_bidders)

    ct_past_bids = np.zeros(args.n_bidders)
    ct_past_winner_bid = 0.0
    ct_win_counts = np.zeros(args.n_bidders)

    # Tracking
    lu_revenues, ct_revenues = [], []
    lu_overbids, ct_overbids = 0, 0
    lu_no_sale, ct_no_sale = 0, 0
    lu_action_counts = np.zeros(n_bid_bins, dtype=int)
    ct_action_counts = np.zeros(n_bid_bins, dtype=int)

    print(f"=== Exp3 Debug: {args.rounds} rounds ===")
    print(f"  eta={eta} (alpha={alpha_val:.2f}, beta={beta_val:.2f})")
    print(f"  auction={args.auction}, reserve={args.reserve}, n_bidders={args.n_bidders}")
    print(f"  context={args.context} (dim={context_dim}), exploration={args.exploration}")
    print(f"  LinUCB c={linucb_c}, CTS sigma2={cts_sigma2}, lam={args.lam}")
    print(f"  actions={fmt_arr(actions)}")
    print(f"  seed={args.seed}")
    print()

    # Pre-generate all signals with a separate RNG so that tie-breaking
    # in get_rewards() (which uses global np.random) cannot desync signals
    # between the LinUCB and CTS paths.
    signal_rng = np.random.RandomState(args.seed)
    all_signals = []
    for _ in range(args.rounds):
        signals_pre = signal_rng.randint(n_val_bins, size=args.n_bidders) / (n_val_bins - 1)
        all_signals.append(signals_pre)

    for ep in range(args.rounds):
        signals = all_signals[ep]
        valuations = np.array([
            get_valuation(eta, signals[i], np.delete(signals, i))
            for i in range(args.n_bidders)
        ])

        verbose = ep < args.verbose_rounds

        if verbose:
            print(f"=== Round {ep} ===")
            print(f"  Signals:    {fmt_arr(signals)}")
            print(f"  Valuations: {fmt_arr(valuations)}  (eta={eta}, alpha={alpha_val:.2f}, beta={beta_val:.2f})")

        # --- LinUCB ---
        lu_chosen_actions = []
        lu_bids = []
        for i in range(args.n_bidders):
            ctx = build_context(i, signals, lu_past_bids, lu_past_winner_bid,
                                lu_win_counts, ep, args.n_bidders, args.context)
            # Get internal state for verbose output
            if verbose:
                p_vals = []
                for a in range(n_bid_bins):
                    A_inv = np.linalg.inv(linucb_bandits[i].A[a])
                    theta_hat = A_inv @ linucb_bandits[i].b[a]
                    mean_est = theta_hat @ ctx
                    bonus = linucb_bandits[i].c * np.sqrt(ctx.T @ A_inv @ ctx)
                    p_vals.append(mean_est + bonus)
                a_i = np.argmax(p_vals)
                print(f"  LinUCB bidder {i}: ctx={fmt_arr(ctx)} UCB={fmt_arr(p_vals)} "
                      f"-> act={a_i} (bid={actions[a_i]:.2f})")
            else:
                a_i = linucb_bandits[i].select_action(ctx)

            lu_chosen_actions.append((i, a_i, ctx))
            lu_bids.append(actions[a_i])
            lu_action_counts[a_i] += 1

        lu_bids_arr = np.array(lu_bids)
        lu_rew, lu_winner, lu_win_bid = get_rewards(
            lu_bids_arr, valuations, args.auction, args.reserve)

        for (i, a_i, ctx) in lu_chosen_actions:
            linucb_bandits[i].update(a_i, lu_rew[i], ctx)

        # Revenue for seller
        if args.auction == "first":
            lu_rev = lu_win_bid if lu_winner != -1 else 0.0
        else:
            # Second-price: revenue = second-highest valid bid (or reserve)
            valid = lu_bids_arr[lu_bids_arr >= args.reserve]
            if len(valid) >= 2:
                lu_rev = np.sort(valid)[-2]
            elif len(valid) == 1:
                lu_rev = args.reserve
            else:
                lu_rev = 0.0
        lu_revenues.append(lu_rev)

        if lu_winner == -1:
            lu_no_sale += 1
        for i in range(args.n_bidders):
            if lu_bids_arr[i] > valuations[i] + 1e-9:
                lu_overbids += 1

        lu_past_bids = lu_bids_arr
        lu_past_winner_bid = lu_win_bid if lu_winner != -1 else 0.0
        if lu_winner != -1:
            lu_win_counts[lu_winner] += 1

        # --- CTS ---
        ct_chosen_actions = []
        ct_bids = []
        for i in range(args.n_bidders):
            ctx = build_context(i, signals, ct_past_bids, ct_past_winner_bid,
                                ct_win_counts, ep, args.n_bidders, args.context)
            if verbose:
                sampled_vals = []
                for a in range(n_bid_bins):
                    A_inv = np.linalg.inv(cts_bandits[i].A[a])
                    theta_hat = A_inv @ cts_bandits[i].b[a]
                    cov = cts_bandits[i].sigma2 * A_inv
                    theta_sample = cts_bandits[i].rng.multivariate_normal(theta_hat, cov)
                    sampled_vals.append(theta_sample @ ctx)
                a_i = np.argmax(sampled_vals)
                print(f"  CTS    bidder {i}: ctx={fmt_arr(ctx)} sampled={fmt_arr(sampled_vals)} "
                      f"-> act={a_i} (bid={actions[a_i]:.2f})")
            else:
                a_i = cts_bandits[i].select_action(ctx)

            ct_chosen_actions.append((i, a_i, ctx))
            ct_bids.append(actions[a_i])
            ct_action_counts[a_i] += 1

        ct_bids_arr = np.array(ct_bids)
        ct_rew, ct_winner, ct_win_bid = get_rewards(
            ct_bids_arr, valuations, args.auction, args.reserve)

        for (i, a_i, ctx) in ct_chosen_actions:
            cts_bandits[i].update(a_i, ct_rew[i], ctx)

        if args.auction == "first":
            ct_rev = ct_win_bid if ct_winner != -1 else 0.0
        else:
            valid = ct_bids_arr[ct_bids_arr >= args.reserve]
            if len(valid) >= 2:
                ct_rev = np.sort(valid)[-2]
            elif len(valid) == 1:
                ct_rev = args.reserve
            else:
                ct_rev = 0.0
        ct_revenues.append(ct_rev)

        if ct_winner == -1:
            ct_no_sale += 1
        for i in range(args.n_bidders):
            if ct_bids_arr[i] > valuations[i] + 1e-9:
                ct_overbids += 1

        ct_past_bids = ct_bids_arr
        ct_past_winner_bid = ct_win_bid if ct_winner != -1 else 0.0
        if ct_winner != -1:
            ct_win_counts[ct_winner] += 1

        # Verbose auction outcome
        if verbose:
            print(f"\n  Auction ({args.auction}-price, reserve={args.reserve}):")

            if lu_winner != -1:
                lu_reward_str = ", ".join(
                    f"b{i}:{lu_rew[i]:+.3f}" + (" W" if i == lu_winner else "")
                    for i in range(args.n_bidders))
                print(f"    LinUCB: winner={lu_winner}, price={lu_rev:.3f}, "
                      f"rewards=[{lu_reward_str}]")
            else:
                print(f"    LinUCB: NO SALE (all bids below reserve)")

            if ct_winner != -1:
                ct_reward_str = ", ".join(
                    f"b{i}:{ct_rew[i]:+.3f}" + (" W" if i == ct_winner else "")
                    for i in range(args.n_bidders))
                print(f"    CTS:    winner={ct_winner}, price={ct_rev:.3f}, "
                      f"rewards=[{ct_reward_str}]")
            else:
                print(f"    CTS:    NO SALE (all bids below reserve)")

            # Sanity warnings
            warnings = []
            for i in range(args.n_bidders):
                if lu_bids_arr[i] > valuations[i] + 1e-9:
                    warnings.append(f"LinUCB bidder {i} overbid "
                                    f"(bid={lu_bids_arr[i]:.2f} > val={valuations[i]:.3f})")
                if ct_bids_arr[i] > valuations[i] + 1e-9:
                    warnings.append(f"CTS bidder {i} overbid "
                                    f"(bid={ct_bids_arr[i]:.2f} > val={valuations[i]:.3f})")
            if lu_winner != -1 and lu_rew[lu_winner] < -1e-9:
                warnings.append(f"LinUCB winner {lu_winner} got negative reward {lu_rew[lu_winner]:.3f}")
            if ct_winner != -1 and ct_rew[ct_winner] < -1e-9:
                warnings.append(f"CTS winner {ct_winner} got negative reward {ct_rew[ct_winner]:.3f}")

            for w in warnings:
                print(f"  SANITY: {w}")
            print()

    # --- Summary ---
    total_bid_events = args.rounds * args.n_bidders
    print(f"=== Summary ({args.rounds} rounds) ===")
    print(f"  LinUCB: avg_rev={np.mean(lu_revenues):.4f}, "
          f"no_sale={lu_no_sale/args.rounds*100:.1f}%, "
          f"overbid_rate={lu_overbids/total_bid_events*100:.1f}%, "
          f"action_dist={lu_action_counts.tolist()}")
    print(f"  CTS:    avg_rev={np.mean(ct_revenues):.4f}, "
          f"no_sale={ct_no_sale/args.rounds*100:.1f}%, "
          f"overbid_rate={ct_overbids/total_bid_events*100:.1f}%, "
          f"action_dist={ct_action_counts.tolist()}")

    # BNE reference
    from experiments.exp3 import simulate_linear_affiliation_revenue
    bne_rev = simulate_linear_affiliation_revenue(args.n_bidders, eta, args.auction, M=10_000)
    print(f"\n  BNE revenue (Monte Carlo, 10k): {bne_rev:.4f}")

    # Theta convergence
    print(f"\n  Theta estimates (LinUCB bidder 0):")
    for a in range(n_bid_bins):
        A_inv = np.linalg.inv(linucb_bandits[0].A[a])
        theta = A_inv @ linucb_bandits[0].b[a]
        print(f"    Action {a} (bid={actions[a]:.2f}): theta={fmt_arr(theta)}")

    print(f"\n  Theta estimates (CTS bidder 0):")
    for a in range(n_bid_bins):
        A_inv = np.linalg.inv(cts_bandits[0].A[a])
        theta = A_inv @ cts_bandits[0].b[a]
        print(f"    Action {a} (bid={actions[a]:.2f}): theta={fmt_arr(theta)}")

    # Sanity check summary
    print(f"\n  --- Sanity Checks ---")

    # Context range
    if args.context == "rich":
        print(f"  LinUCB win_rates: {fmt_arr(lu_win_counts / max(1, args.rounds))}")
        print(f"  CTS    win_rates: {fmt_arr(ct_win_counts / max(1, args.rounds))}")

    # Theta magnitude
    max_theta_lu = max(
        np.max(np.abs(np.linalg.inv(linucb_bandits[0].A[a]) @ linucb_bandits[0].b[a]))
        for a in range(n_bid_bins))
    max_theta_ct = max(
        np.max(np.abs(np.linalg.inv(cts_bandits[0].A[a]) @ cts_bandits[0].b[a]))
        for a in range(n_bid_bins))
    print(f"  Max |theta| - LinUCB: {max_theta_lu:.4f}, CTS: {max_theta_ct:.4f}")
    if max_theta_lu > 10:
        print(f"  WARNING: LinUCB theta magnitude seems large (>{max_theta_lu:.1f})")
    if max_theta_ct > 10:
        print(f"  WARNING: CTS theta magnitude seems large (>{max_theta_ct:.1f})")

    # Action diversity
    lu_entropy = -np.sum((lu_action_counts / total_bid_events) *
                         np.log(lu_action_counts / total_bid_events + 1e-12))
    ct_entropy = -np.sum((ct_action_counts / total_bid_events) *
                         np.log(ct_action_counts / total_bid_events + 1e-12))
    max_entropy = np.log(n_bid_bins)
    print(f"  Action entropy - LinUCB: {lu_entropy:.3f}/{max_entropy:.3f}, "
          f"CTS: {ct_entropy:.3f}/{max_entropy:.3f}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Debug Exp3 LinUCB + CTS algorithms")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--verbose-rounds", type=int, default=20)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--auction", choices=["first", "second"], default="second")
    parser.add_argument("--n-bidders", type=int, default=2)
    parser.add_argument("--reserve", type=float, default=0.0)
    parser.add_argument("--context", choices=["minimal", "rich"], default="rich")
    parser.add_argument("--exploration", choices=["low", "high"], default="high")
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_debug(args)


if __name__ == "__main__":
    main()
