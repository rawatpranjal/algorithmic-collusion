#!/usr/bin/env python3
"""
Verbose diagnostic for Exp2 Q-learning.

Traces episode mechanics and converged policy to understand why Q-learners
produce above-BNE revenue in highly-affiliated auctions.

Replicates the run_experiment() loop from exp2.py line-by-line (same seed,
same logic) but with print statements at key points.

Usage:
    python3 scripts/debug_exp2.py                          # worst cell defaults
    python3 scripts/debug_exp2.py --eta 1.0 --auction second --n-bidders 2
    python3 scripts/debug_exp2.py --episodes 5000 --trace-episodes 3
"""

import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from experiments.exp2 import get_rewards, get_valuation
from verification.bne_verify import analytical_revenue, compute_bne_bid_coefficient


def parse_args():
    p = argparse.ArgumentParser(description="Exp2 Q-learning debug trace")
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--auction", type=str, default="second", choices=["first", "second"])
    p.add_argument("--n-bidders", type=int, default=6)
    p.add_argument("--state-info", type=str, default="signal_only",
                   choices=["signal_only", "signal_winner"])
    p.add_argument("--episodes", type=int, default=10_000)
    p.add_argument("--trace-episodes", type=int, default=5,
                   help="Number of initial episodes to trace in full detail")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.1, help="Q-learning rate")
    p.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    p.add_argument("--n-bid-actions", type=int, default=21)
    p.add_argument("--n-signal-bins", type=int, default=11)
    return p.parse_args()


def run_debug(args):
    eta = args.eta
    auction_type = args.auction
    n_bidders = args.n_bidders
    state_info = args.state_info
    episodes = args.episodes
    n_trace_init = args.trace_episodes
    seed = args.seed
    alpha_lr = args.alpha
    gamma = args.gamma
    n_bid_actions = args.n_bid_actions
    n_signal_bins = args.n_signal_bins

    # BNE references
    R_bne = analytical_revenue(eta, n_bidders)
    phi_bne = compute_bne_bid_coefficient(eta, n_bidders, auction_type)

    # ---------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------
    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(f"  eta             = {eta}")
    print(f"  auction_type    = {auction_type}")
    print(f"  n_bidders       = {n_bidders}")
    print(f"  state_info      = {state_info}")
    print(f"  episodes        = {episodes}")
    print(f"  seed            = {seed}")
    print(f"  alpha (lr)      = {alpha_lr}")
    print(f"  gamma           = {gamma}")
    print(f"  n_bid_actions   = {n_bid_actions}")
    print(f"  n_signal_bins   = {n_signal_bins}")
    print(f"  BNE revenue     = {R_bne:.6f}")
    print(f"  BNE phi ({auction_type:>6}) = {phi_bne:.6f}")
    print(f"  BNE bid fn      = b(s) = {phi_bne:.4f} * s")
    print()

    # ---------------------------------------------------------------
    # Determine which episodes to trace
    # ---------------------------------------------------------------
    trace_set = set(range(n_trace_init))
    sample_fracs = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    for frac in sample_fracs:
        trace_set.add(int(frac * episodes))
    # Also trace last episode
    trace_set.add(episodes - 1)

    # ---------------------------------------------------------------
    # Replicate run_experiment() setup — must match exp2.py exactly
    # ---------------------------------------------------------------
    np.random.seed(seed)

    action_space = np.linspace(0, 1, n_bid_actions)

    if state_info == "signal_winner":
        n_states = n_signal_bins * n_bid_actions
    else:
        n_states = n_signal_bins

    Q = np.zeros((n_bidders, n_states, n_bid_actions))

    revenues = []
    winning_bids_list = []
    no_sale_count = 0
    winners_list = []

    eps_start, eps_end = 1.0, 0.0
    decay_end = int(0.9 * episodes)

    last_winner_bid_bin = 0

    # Final-window tracking
    window_size = min(1000, episodes)
    final_window_vals = []
    final_window_payments = []
    final_window_winners = []

    # ---------------------------------------------------------------
    # Episode loop
    # ---------------------------------------------------------------
    for ep in range(episodes):
        # Epsilon decay
        if ep < decay_end:
            eps = eps_start - (ep / decay_end) * (eps_start - eps_end)
        else:
            eps = eps_end

        tracing = ep in trace_set

        # Signals
        signals = np.random.uniform(0, 1, size=n_bidders)
        signal_bins = np.round(signals * (n_signal_bins - 1)).astype(int)

        # Valuations
        valuations = np.zeros(n_bidders)
        for i in range(n_bidders):
            others = np.delete(signals, i)
            valuations[i] = get_valuation(eta, signals[i], others)

        # States
        if state_info == "signal_winner":
            states = signal_bins * n_bid_actions + last_winner_bid_bin
        else:
            states = signal_bins

        # Action selection
        chosen_actions = np.zeros(n_bidders, dtype=int)
        explore_flags = []
        for i in range(n_bidders):
            if np.random.rand() > eps:
                chosen_actions[i] = np.argmax(Q[i, states[i]])
                explore_flags.append(False)
            else:
                chosen_actions[i] = np.random.randint(n_bid_actions)
                explore_flags.append(True)

        bids = action_space[chosen_actions]

        # Auction
        rew, winner, winner_bid_val = get_rewards(bids, valuations, auction_type, 0.0)

        # Revenue calculation (matches exp2.py exactly)
        valid_bids = bids[bids >= 0.0]
        if auction_type == "first":
            revenue_t = float(np.max(valid_bids)) if len(valid_bids) > 0 else 0.0
        else:
            if len(valid_bids) >= 2:
                sorted_valid = np.sort(valid_bids)
                revenue_t = float(sorted_valid[-2])
            elif len(valid_bids) == 1:
                revenue_t = float(valid_bids[0])
            else:
                revenue_t = 0.0

        revenues.append(revenue_t)
        winning_bids_list.append(winner_bid_val)
        if winner == -1:
            no_sale_count += 1
        else:
            winners_list.append(winner)

        # Final-window data
        in_final_window = ep >= (episodes - window_size)
        if in_final_window:
            final_window_vals.append(valuations.copy())
            final_window_winners.append(winner)
            if winner >= 0:
                if auction_type == "first":
                    final_window_payments.append(bids[winner])
                else:
                    if len(valid_bids) >= 2:
                        sorted_valid = np.sort(valid_bids)
                        final_window_payments.append(sorted_valid[-2])
                    else:
                        final_window_payments.append(bids[winner])
            else:
                final_window_payments.append(0.0)

        # Next state
        if winner >= 0:
            new_winner_bid_bin = int(np.round(winner_bid_val * (n_bid_actions - 1)))
            new_winner_bid_bin = min(max(new_winner_bid_bin, 0), n_bid_actions - 1)
        else:
            new_winner_bid_bin = 0

        if state_info == "signal_winner":
            next_states = signal_bins * n_bid_actions + new_winner_bid_bin
        else:
            next_states = signal_bins

        # ---------------------------------------------------------------
        # TRACE: print episode details
        # ---------------------------------------------------------------
        if tracing:
            print("=" * 70)
            print(f"EPISODE {ep}  (epsilon={eps:.4f})")
            print("=" * 70)

            print(f"  Signals (continuous): {np.array2string(signals, precision=4)}")
            print(f"  Signal bins:          {signal_bins}")
            print(f"  Valuations:           {np.array2string(valuations, precision=4)}")
            if state_info == "signal_winner":
                print(f"  Last winner bid bin:  {last_winner_bid_bin}")
            print(f"  States:               {states}")
            print()

            # Per-bidder details
            for i in range(n_bidders):
                q_row = Q[i, states[i]]
                top3_idx = np.argsort(q_row)[::-1][:3]
                top3_str = ", ".join(
                    f"a{idx}(b={action_space[idx]:.2f})={q_row[idx]:.6f}"
                    for idx in top3_idx
                )
                bne_bid_i = phi_bne * signals[i]
                flag = "EXPLORE" if explore_flags[i] else "EXPLOIT"
                print(f"  Bidder {i}: state={states[i]}  "
                      f"action={chosen_actions[i]}  bid={bids[i]:.4f}  "
                      f"BNE_bid={bne_bid_i:.4f}  [{flag}]")
                print(f"    Top-3 Q: {top3_str}")

            print()
            print(f"  All bids:   {np.array2string(bids, precision=4)}")
            print(f"  Winner:     {winner}  "
                  f"(winning bid={winner_bid_val:.4f})")

            if winner >= 0:
                if auction_type == "second":
                    print(f"  Payment (2nd price): {revenue_t:.4f}")
                else:
                    print(f"  Payment (1st price): {bids[winner]:.4f}")
                print(f"  Winner valuation:    {valuations[winner]:.4f}")
                print(f"  Winner payoff:       {rew[winner]:.4f}")
                if auction_type == "second" and revenue_t > valuations[winner]:
                    print(f"  ** WINNER'S CURSE: payment {revenue_t:.4f} > "
                          f"valuation {valuations[winner]:.4f} **")
            else:
                print(f"  No sale!")

            print(f"  Revenue:    {revenue_t:.4f}  (BNE={R_bne:.4f})")
            print(f"  Rewards:    {np.array2string(rew, precision=6)}")
            print()

        # Q-update (must match exp2.py exactly)
        for i in range(n_bidders):
            old_q = Q[i, states[i], chosen_actions[i]]
            td_target = rew[i] + gamma * np.max(Q[i, next_states[i]])
            new_q = old_q + alpha_lr * (td_target - old_q)

            if tracing:
                print(f"  Q-update bidder {i}: "
                      f"old={old_q:.6f}  td_target={td_target:.6f}  "
                      f"new={new_q:.6f}  (delta={new_q - old_q:.6f})")

            Q[i, states[i], chosen_actions[i]] = new_q

        last_winner_bid_bin = new_winner_bid_bin

        if tracing:
            print()

        # Progress indicator every 10% for non-traced episodes
        if ep > 0 and ep % (episodes // 10) == 0 and not tracing:
            avg_recent = np.mean(revenues[max(0, ep - 1000):ep])
            print(f"  [Progress] Episode {ep}/{episodes}  "
                  f"avg_rev(last 1000)={avg_recent:.4f}  "
                  f"eps={eps:.4f}")

    # ---------------------------------------------------------------
    # CONVERGED POLICY
    # ---------------------------------------------------------------
    print()
    print("=" * 70)
    print("CONVERGED POLICY")
    print("=" * 70)
    print()

    if state_info == "signal_only":
        print(f"{'Bin':>4} {'Signal':>7} {'BNE_bid':>8}", end="")
        for i in range(n_bidders):
            print(f" {'B' + str(i) + '_act':>7} {'B' + str(i) + '_bid':>8}", end="")
        print(f" {'AvgBid':>8} {'Overbid':>8}")
        print("-" * (30 + n_bidders * 16))

        overbid_count = 0
        total_bins = 0
        for b in range(n_signal_bins):
            signal_val = b / (n_signal_bins - 1)
            bne_bid = phi_bne * signal_val

            greedy_actions = []
            greedy_bids = []
            for i in range(n_bidders):
                a = np.argmax(Q[i, b])
                greedy_actions.append(a)
                greedy_bids.append(action_space[a])

            avg_bid = np.mean(greedy_bids)
            overbid = avg_bid > bne_bid + 0.01

            print(f"{b:4d} {signal_val:7.3f} {bne_bid:8.4f}", end="")
            for i in range(n_bidders):
                print(f" {greedy_actions[i]:7d} {greedy_bids[i]:8.4f}", end="")
            flag = " <<< OVERBID" if overbid else ""
            print(f" {avg_bid:8.4f} {flag}")

            if overbid:
                overbid_count += 1
            total_bins += 1

        print()
        print(f"Overbidding bins: {overbid_count}/{total_bins}")
    else:
        # signal_winner: show policy for last_winner_bid_bin = 0 (representative)
        print("(Showing policy for last_winner_bid_bin = 0)")
        print(f"{'Bin':>4} {'Signal':>7} {'BNE_bid':>8}", end="")
        for i in range(min(n_bidders, 4)):
            print(f" {'B' + str(i) + '_act':>7} {'B' + str(i) + '_bid':>8}", end="")
        print(f" {'AvgBid':>8}")
        print("-" * (30 + min(n_bidders, 4) * 16 + 9))

        for b in range(n_signal_bins):
            signal_val = b / (n_signal_bins - 1)
            bne_bid = phi_bne * signal_val
            state_idx = b * n_bid_actions + 0  # last_winner_bid_bin = 0

            greedy_bids = []
            print(f"{b:4d} {signal_val:7.3f} {bne_bid:8.4f}", end="")
            for i in range(min(n_bidders, 4)):
                a = np.argmax(Q[i, state_idx])
                bid = action_space[a]
                greedy_bids.append(bid)
                print(f" {a:7d} {bid:8.4f}", end="")
            print(f" {np.mean(greedy_bids):8.4f}")

    # ---------------------------------------------------------------
    # Q-TABLE STATISTICS
    # ---------------------------------------------------------------
    print()
    print("=" * 70)
    print("Q-TABLE STATISTICS")
    print("=" * 70)
    for i in range(n_bidders):
        nonzero = np.count_nonzero(Q[i])
        total = Q[i].size
        print(f"  Bidder {i}: nonzero={nonzero}/{total} ({100*nonzero/total:.1f}%)  "
              f"min={Q[i].min():.6f}  max={Q[i].max():.6f}  "
              f"mean(nonzero)={Q[i][Q[i] != 0].mean():.6f}" if nonzero > 0
              else f"  Bidder {i}: all zeros")

    # ---------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    avg_rev = np.mean(revenues[-window_size:])
    print(f"  Avg revenue (last {window_size}):  {avg_rev:.6f}")
    print(f"  BNE revenue:                 {R_bne:.6f}")
    print(f"  Ratio to BNE:                {avg_rev / R_bne:.4f}")
    print(f"  Excess regret:               {1.0 - avg_rev / R_bne:.4f}")
    print(f"  No-sale rate:                {no_sale_count / episodes:.4f}")

    # Winner's curse in final window
    fw_vals = np.array(final_window_vals)
    fw_winners = np.array(final_window_winners)
    fw_payments = np.array(final_window_payments)
    valid = fw_winners >= 0
    if valid.sum() > 0:
        w_idx = fw_winners[valid].astype(int)
        rows = np.arange(valid.sum())
        w_vals = fw_vals[valid][rows, w_idx]
        w_pay = fw_payments[valid]
        curse_freq = (w_pay > w_vals).mean()
        btv = np.where(w_vals > 1e-12, w_pay / w_vals, 0.0)
        print(f"  Winner's curse freq:         {curse_freq:.4f}")
        print(f"  Bid-to-value median:         {np.median(btv):.4f}")
        print(f"  Bid-to-value IQR:            "
              f"{np.percentile(btv, 75) - np.percentile(btv, 25):.4f}")
    else:
        print(f"  No valid winners in final window!")

    # Revenue trajectory milestones
    print()
    print("  Revenue trajectory:")
    milestones = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00]
    for frac in milestones:
        idx = min(int(frac * episodes) - 1, episodes - 1)
        idx = max(idx, 0)
        start = max(0, idx - 999)
        avg = np.mean(revenues[start:idx + 1])
        print(f"    ep {idx:>6d} ({frac*100:5.1f}%): avg_rev={avg:.4f}")


if __name__ == "__main__":
    args = parse_args()
    run_debug(args)
