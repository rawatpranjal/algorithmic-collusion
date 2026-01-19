#!/usr/bin/env python3
import numpy as np
import pandas as pd
import statsmodels.api as sm
import random
from tabulate import tabulate

# Set DEBUG=False to suppress logs, True to show analysis logs.
DEBUG = False
def dprint(*args):
    if DEBUG:
        print(*args)

# --------------------------------------------------------------
# 1) Valuation with eta
# --------------------------------------------------------------
def get_valuation(eta, s_i, s_j):
    """
    N=2 bidders. Define:
      alpha(eta) = 1 - 0.5*eta
      beta(eta)  = 0.5*eta
    Then
      v_i = alpha(eta)*s_i + beta(eta)*s_j
    s_i, s_j in {0..5} => actual signals = 0.2*s_i, 0.2*s_j.

    Special cases:
      eta=0  => alpha=1, beta=0   => pure private
      eta=1  => alpha=0.5, beta=0.5 => pure common
      0<eta<1 => affiliated
    """
    # Convert discrete index to actual [0,1] signals
    si = 0.2 * s_i
    sj = 0.2 * s_j
    alpha = 1.0 - 0.5*eta
    beta  = 0.5*eta
    return alpha*si + beta*sj

# --------------------------------------------------------------
# 2) Payoffs
# --------------------------------------------------------------
def get_payoffs(b1, b2, v1, v2, auction_type):
    """
    Tie => random winner.
    First-price => winner pays own bid,
    Second-price => winner pays the other top bid.
    """
    if b1 > b2:
        if auction_type == "first":   return (v1 - b1, 0.0)
        else:                         return (v1 - b2, 0.0)
    elif b2 > b1:
        if auction_type == "first":   return (0.0, v2 - b2)
        else:                         return (0.0, v2 - b1)
    else:
        # Tie
        w = 0 if np.random.rand()<0.5 else 1
        if auction_type == "first":
            return ((v1 - b1, 0.0) if w==0 else (0.0, v2 - b2))
        else:
            return ((v1 - b2, 0.0) if w==0 else (0.0, v2 - b1))

# --------------------------------------------------------------
# 3) Q-learning update
# --------------------------------------------------------------
def qlearning_update(Q, s, a, r, s_next, alpha, gamma):
    old_q = Q[s, a]
    best_future = np.max(Q[s_next])
    Q[s, a] = old_q + alpha * (r + gamma * best_future - old_q)

# --------------------------------------------------------------
# 4) Build state index
# --------------------------------------------------------------
def build_state(
    own_signal,     
    last_win_bid,   
    last_opp_bid,   
    winning_info,
    opp_info,
    n_val_bins,
    n_bid_bins
):
    """
    Flatten: idx = own_signal
               + n_val_bins*(lw)
               + (n_val_bins*w_size)*(lo)

    lw = last_win_bid if winning_info else 0
    lo = last_opp_bid if opp_info else 0
    w_size = n_bid_bins if winning_info else 1
    """
    lw = last_win_bid if winning_info else 0
    lo = last_opp_bid if opp_info else 0
    w_size = n_bid_bins if winning_info else 1

    idx = own_signal
    idx += n_val_bins * lw
    idx += (n_val_bins * w_size) * lo
    return idx

# --------------------------------------------------------------
# 5) Run single experiment
# --------------------------------------------------------------
def run_experiment(
    eta,             # in {0, 0.5, 1.0}
    auction_type,    # "first","second"
    alpha_lr,        # learning rate
    gamma_disc,      # discount factor
    episodes,        
    init,            # "random" or "zeros"
    exploration,     # "egreedy" or "boltzmann"
    winning_info,    
    opp_info,
    seed=0
):
    np.random.seed(seed)
    random.seed(seed)

    n_val_bins = 6  # signals => {0..5} => actual 0..1 step 0.2
    n_bid_bins = 6  # bids    => {0..5} => actual 0..1 step 0.2

    w_size = n_bid_bins if winning_info else 1
    o_size = n_bid_bins if opp_info else 1
    n_states = n_val_bins * w_size * o_size

    # Q tables
    if init == "random":
        Q1 = np.random.rand(n_states, n_bid_bins)
        Q2 = np.random.rand(n_states, n_bid_bins)
    else:
        Q1 = np.zeros((n_states, n_bid_bins))
        Q2 = np.zeros((n_states, n_bid_bins))

    # Action set
    actions = np.linspace(0, 1, n_bid_bins)  # e.g. 0.0, 0.2, 0.4, ...

    # Epsilon schedule => decay ends at 90% of episodes
    eps_start, eps_end = 1.0, 0.01
    decay_end = int(0.9 * episodes)

    def choose_action(Qrow, eps):
        if exploration == "boltzmann":
            ex = np.exp(Qrow)
            probs = ex / np.sum(ex)
            return np.random.choice(len(Qrow), p=probs)
        else:  # e-greedy
            if np.random.rand() < eps:
                return np.random.randint(n_bid_bins)
            return np.argmax(Qrow)

    revenues = []
    # last winning-bid index & opp-bid index
    lw1 = lw2 = 0
    lo1 = lo2 = 0

    for ep in range(episodes):
        # Epsilon decay
        if ep < decay_end:
            eps = eps_start - (ep / decay_end)*(eps_start - eps_end)
        else:
            eps = eps_end

        # Random signals in {0..5}
        s1 = np.random.randint(n_val_bins)
        s2 = np.random.randint(n_val_bins)

        v1 = get_valuation(eta, s1, s2)
        v2 = get_valuation(eta, s2, s1)

        st1 = build_state(s1, lw1, lo1, winning_info, opp_info, n_val_bins, n_bid_bins)
        st2 = build_state(s2, lw2, lo2, winning_info, opp_info, n_val_bins, n_bid_bins)

        a1 = choose_action(Q1[st1], eps)
        a2 = choose_action(Q2[st2], eps)

        b1, b2 = actions[a1], actions[a2]
        r1, r2 = get_payoffs(b1, b2, v1, v2, auction_type)
        revenues.append(max(b1, b2))

        # Update last-win & opp-bid
        if b1 > b2:
            lw1, lw2 = a1, 0
            lo1, lo2 = a2, a1
        elif b2 > b1:
            lw1, lw2 = 0, a2
            lo1, lo2 = a2, a1
        else:
            w = 0 if np.random.rand()<0.5 else 1
            lw1 = a1 if w==0 else 0
            lw2 = a2 if w==1 else 0
            lo1, lo2 = a2, a1

        # Next-state for Q-update
        s1p = np.random.randint(n_val_bins)
        s2p = np.random.randint(n_val_bins)
        nxt_st1 = build_state(s1p, lw1 if winning_info else 0,
                              lo1 if opp_info else 0,
                              winning_info, opp_info,
                              n_val_bins, n_bid_bins)
        nxt_st2 = build_state(s2p, lw2 if winning_info else 0,
                              lo2 if opp_info else 0,
                              winning_info, opp_info,
                              n_val_bins, n_bid_bins)

        # Q-learning updates
        qlearning_update(Q1, st1, a1, r1, nxt_st1, alpha_lr, gamma_disc)
        qlearning_update(Q2, st2, a2, r2, nxt_st2, alpha_lr, gamma_disc)

    # Return final avg revenue over last 1000 steps
    return np.mean(revenues[-1000:])

# --------------------------------------------------------------
# 6) Main experiment => gather data => OLS
# --------------------------------------------------------------
def main_experiment(K=50):
    """
    We'll draw random combos from param_space, including eta in {0,0.5,1.0},
    run the experiment, and store final average revenue.
    """
    param_space = {
        "eta":            [0.0, 0.5, 1.0],
        "auction_type":   ["first","second"],
        "alpha_lr":       [0.01, 0.05],
        "gamma_disc":     [0.9, 0.99],
        "episodes":       [50_000, 100_000],
        "init":           ["random","zeros"],
        "exploration":    ["egreedy","boltzmann"],
        "winning_info":   [False, True],
        "opp_info":       [False, True]
    }

    results = []
    for seed in range(K):
        e     = random.choice(param_space["eta"])
        auc   = random.choice(param_space["auction_type"])
        aLR   = random.choice(param_space["alpha_lr"])
        gd    = random.choice(param_space["gamma_disc"])
        epsn  = random.choice(param_space["episodes"])
        ini   = random.choice(param_space["init"])
        exp   = random.choice(param_space["exploration"])
        wI    = random.choice(param_space["winning_info"])
        oI    = random.choice(param_space["opp_info"])

        final_rev = run_experiment(
            eta=e,
            auction_type=auc,
            alpha_lr=aLR,
            gamma_disc=gd,
            episodes=epsn,
            init=ini,
            exploration=exp,
            winning_info=wI,
            opp_info=oI,
            seed=seed
        )

        results.append({
            "eta":            e,
            "auction_type":   auc,
            "alpha_lr":       aLR,
            "gamma_disc":     gd,
            "episodes":       epsn,
            "init":           ini,
            "exploration":    exp,
            "winning_info":   wI,
            "opp_info":       oI,
            "final_avg_rev":  round(final_rev, 3)
        })

    return pd.DataFrame(results)

# --------------------------------------------------------------
# 7) Run & OLS with tabulate
# --------------------------------------------------------------
if __name__ == "__main__":
    # Activate debug prints in analysis stage only
    DEBUG = True

    dprint("Starting main experiment...")
    df = main_experiment(K=50)
    dprint("Experiment complete.\n")

    # Show a short table of results
    print("\nSample Data (head):")
    print(tabulate(df.head(10), headers=df.columns, floatfmt=".3f", tablefmt="pretty"))

    # Baseline: eta=0 => alpha=1,beta=0. Then define dummies for other etas
    df["eta_05"] = (df["eta"]==0.5).astype(int)
    df["eta_1"]  = (df["eta"]==1.0).astype(int)
    df["auc_first"]  = (df["auction_type"]=="first").astype(int)
    df["alpha_001"]  = (df["alpha_lr"]==0.01).astype(int)
    df["gamma_09"]   = (df["gamma_disc"]==0.9).astype(int)
    df["ep_50k"]     = (df["episodes"]==50_000).astype(int)
    df["init_rand"]  = (df["init"]=="random").astype(int)
    df["expl_boltz"] = (df["exploration"]=="boltzmann").astype(int)
    df["win_info"]   = df["winning_info"].astype(int)
    df["opp_info_"]  = df["opp_info"].astype(int)

    formula = (
        "final_avg_rev ~ "
        "eta_05 + eta_1 + auc_first + alpha_001 "
        "+ gamma_09 + ep_50k + init_rand + expl_boltz "
        "+ win_info + opp_info_"
    )
    model = sm.OLS.from_formula(formula, data=df)
    res = model.fit()

    print("\nOLS SUMMARY:")
    print(res.summary())

    # Final table of entire data
    print("\nFinal Data (tail):")
    print(tabulate(df.tail(10), headers=df.columns, floatfmt=".3f", tablefmt="pretty"))
