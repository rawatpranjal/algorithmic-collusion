#!/usr/bin/env python3

import numpy as np

def simulate_auction(N, alpha, beta, auction_type, M=10_000):
    """
    N: number of bidders
    alpha, beta: valuation parameters
    auction_type: 'FPA' or 'SPA'
    M: number of Monte Carlo runs
    Returns average payoff per bidder and average revenue.
    """
    payoffs = np.zeros(N)
    revenues = 0.0

    for _ in range(M):
        # Draw signals
        t = np.random.rand(N)
        # Compute valuations
        v = alpha * t + beta * (np.sum(t) - t)

        # Compute bids
        if auction_type == 'FPA':
            factor = (N - 1) / N * (alpha + (N / 2) * beta)
        else:
            factor = alpha + (N / 2) * beta
        b = factor * t

        # Determine winner (tie-break randomly)
        max_bid = np.max(b)
        winners = np.where(np.isclose(b, max_bid))[0]
        winner = np.random.choice(winners) if len(winners) > 1 else winners[0]

        # Compute price and payoffs
        if auction_type == 'FPA':
            price = b[winner]
        else:
            if len(winners) == 1:
                temp_b = np.delete(b, winner)
                price = np.max(temp_b)
            else:
                price = max_bid
        payoffs[winner] += max(v[winner] - price, 0)
        revenues += price

    return payoffs / M, revenues / M

def run_simulations(N_values, M=10_000):
    # Valuation cases
    # 1) Private:    alpha=1,       beta=0
    # 2) Common:     alpha=1/N,     beta=1/N
    # 3) Affiliated: alpha=1/2,     beta=1/[2(N-1)]
    # Auction types
    cases = [('Private', lambda N: (1.0, 0.0)),
             ('Common',  lambda N: (1.0/N, 1.0/N)),
             ('Affiliated', lambda N: (0.5, 0.5/(N-1)))]
    auctions = ['FPA', 'SPA']

    for N in N_values:
        print(f"=== Results for N={N} ===\n")
        print("Case       | Auction | AvgPayoffs (per bidder)          | AvgRevenue")
        print("-----------------------------------------------------------------------")
        for case_name, param_func in cases:
            alpha, beta = param_func(N)
            for a_type in auctions:
                avg_payoffs, avg_revenue = simulate_auction(N, alpha, beta, a_type, M)
                avg_payoffs_str = ", ".join(f"{p:.3f}" for p in avg_payoffs)
                print(f"{case_name:10} | {a_type:6} | [{avg_payoffs_str}] | {avg_revenue:.3f}")
        print()

def main():
    # Adjust as needed
    N_values = [3, 4, 5, 10]
    M = 5_000
    run_simulations(N_values, M)

if __name__ == "__main__":
    main()
