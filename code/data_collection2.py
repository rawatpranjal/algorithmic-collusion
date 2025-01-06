#!/usr/bin/env python3

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate

def get_valuations(alpha, beta, signals):
    """
    v_i = alpha * t_i + beta * sum_{j != i} t_j
    """
    N = len(signals)
    ssum = np.sum(signals)
    return np.array([alpha*signals[i] + beta*(ssum - signals[i])
                     for i in range(N)])

def pick_action(Q, bidder_idx, eps):
    """
    Epsilon-greedy for one bidder row Q[bidder_idx].
    """
    if np.random.rand() < eps:
        return np.random.randint(Q.shape[1])
    return np.argmax(Q[bidder_idx])

def run_qlearning(value_type, alpha, beta, design,
                  N=2, T=20000, K=5, num_bids=6,
                  lr=0.1, gamma_disc=0.95,
                  eps_start=1.0, eps_end=0.01, eps_decay_portion=0.8):
    """
    value_type: 'Private', 'Common', 'Affiliated', 'Identical'
      - 'Identical' => signals are always np.ones(N).
      - otherwise => signals are np.random.rand(N).

    design in {0,1} => second-price or first-price.

    Returns:
      rev_history: shape (K, T) = revenue each step for each of K runs
      rev_theory: float if we can compute it, else None
    """
    bid_grid = np.linspace(0, 1, num_bids)
    rev_history = np.zeros((K, T))

    # Epsilon decay
    decay_span = int(eps_decay_portion * T)
    if decay_span < 1:
        decay_span = T
    decay_factor = (eps_end / eps_start) ** (1.0 / decay_span)

    for k in range(K):
        Q = np.zeros((N, num_bids))
        eps = eps_start
        for t in range(T):
            # signals
            if value_type == "Identical":
                signals = np.ones(N)
            else:
                signals = np.random.rand(N)

            vals = get_valuations(alpha, beta, signals)

            acts = [pick_action(Q, i, eps) for i in range(N)]
            bids = bid_grid[acts]

            maxb = np.max(bids)
            wset = np.where(bids == maxb)[0]
            winner = np.random.choice(wset) if len(wset) > 1 else wset[0]
            if len(wset) == 1:
                second = np.max(np.delete(bids, winner))
            else:
                second = maxb

            # design=0 => second-price, design=1 => first-price
            price = design*maxb + (1 - design)*second
            rev_history[k, t] = price

            payoff = np.zeros(N)
            payoff[winner] = vals[winner] - price

            # Update Q
            for i in range(N):
                old_q = Q[i, acts[i]]
                r_i = payoff[i]
                best_future = np.max(Q[i])
                Q[i, acts[i]] = old_q + lr*(r_i + gamma_disc*best_future - old_q)

            # Epsilon decay
            if t < decay_span:
                eps *= decay_factor
            else:
                eps = eps_end

    # --- Theoretical revenue ---
    rev_theory = None
    # If "Identical" and alpha=1,beta=0 => then the theoretical revenue is always 1
    # because each bidder's signal=1 => each valuation=1 => eq. bid=1 => price=1.
    if (value_type == "Identical") and (abs(alpha-1.0)<1e-9) and (abs(beta)<1e-9):
        rev_theory = 1.0
    else:
        if abs(design - 0.0) < 1e-9:
            # second-price eq => factor_sp
            factor_sp = alpha + (N/2)*beta
            M = 20000
            sum_price = 0.0
            for _ in range(M):
                if value_type == "Identical":
                    eq_bids = np.full(N, factor_sp)
                else:
                    s = np.random.rand(N)
                    eq_bids = factor_sp*s
                mb = np.max(eq_bids)
                wset = np.where(eq_bids == mb)[0]
                w2 = np.random.choice(wset) if len(wset)>1 else wset[0]
                if len(wset)==1:
                    secondb = np.max(np.delete(eq_bids, w2))
                else:
                    secondb = mb
                sum_price += secondb
            rev_theory = sum_price / M

        elif abs(design - 1.0) < 1e-9:
            # first-price eq => factor_fp
            factor_fp = (N-1)/N * (alpha + (N/2)*beta)
            M = 20000
            sum_price = 0.0
            for _ in range(M):
                if value_type == "Identical":
                    eq_bids = np.full(N, factor_fp)
                else:
                    s = np.random.rand(N)
                    eq_bids = factor_fp*s
                mb = np.max(eq_bids)
                wset = np.where(eq_bids == mb)[0]
                w2 = np.random.choice(wset) if len(wset)>1 else wset[0]
                sum_price += eq_bids[w2]
            rev_theory = sum_price / M

    return rev_history, rev_theory

def main():
    N = 2
    T = 20000
    K = 5

    combos = [
        ("Private",    1.0,      0.0),
        ("Common",     1.0/N,    1.0/N),
        ("Affiliated", 0.5,      0.5/(N-1)),
        ("Identical",  1.0,      0.0),  # signals=ones, alpha=1,beta=0 => valuations=1
    ]
    design_list = [0.0, 1.0]  # second-price or first-price

    records = []
    for (val_type, a, b) in combos:
        for dsgn in design_list:
            rev_runs, rev_eq = run_qlearning(
                value_type=val_type,
                alpha=a,
                beta=b,
                design=dsgn,
                N=N,
                T=T,
                K=K,
                lr=0.1,
                gamma_disc=0.95,
                eps_start=1.0,
                eps_end=0.01,
                eps_decay_portion=0.8
            )

            # final 1000-step average for each run => shape (K,)
            final_avg_each_run = [
                np.mean(rev_runs[k_idx, -1000:])
                for k_idx in range(K)
            ]
            final_avg_each_run = np.array(final_avg_each_run)
            learned_mean = np.mean(final_avg_each_run)
            learned_std = np.std(final_avg_each_run, ddof=1)

            ratio_mean = np.nan
            ratio_stderr = np.nan
            if rev_eq is not None and rev_eq > 1e-9:
                ratios = final_avg_each_run / rev_eq
                ratio_mean = np.mean(ratios)
                ratio_std = np.std(ratios, ddof=1)
                ratio_stderr = ratio_std / np.sqrt(K)

            records.append({
                'value_type': val_type,
                'alpha': a,
                'beta': b,
                'design': dsgn,
                'K': K,
                'learned_revenue_mean': learned_mean,
                'learned_revenue_std': learned_std,
                'theoretical_revenue': rev_eq,
                'ratio_mean': ratio_mean,
                'ratio_stderr': ratio_stderr
            })

    df = pd.DataFrame(records)
    df.to_csv("results_identical.csv", index=False)

    # Print as a table
    # We choose which columns to show
    show_cols = [
        'value_type','design','learned_revenue_mean','theoretical_revenue',
        'ratio_mean','ratio_stderr'
    ]
    # Make table data
    table_data = df[show_cols].values.tolist()
    headers = show_cols

    from tabulate import tabulate
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

    print("\nSaved: results_identical.csv")

if __name__ == "__main__":
    main()
