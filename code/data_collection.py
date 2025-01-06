#!/usr/bin/env python3

import os
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============ Initialization Methods ============

def initialize_qtable(N, num_actions, observation, init_method=0):
    """
    init_method:
      0 -> Random Uniform
      1 -> Zeros
      2 -> Optimistic (all 100)
    """
    # Determine Q-table shape based on 'observation'
    # observation=0 => Q shape (N, num_actions)
    # observation=1 => Q shape (N, num_actions, num_actions)
    # observation=2 => Q shape (N, num_actions^(N-1), num_actions)
    if observation == 0:
        shape = (N, num_actions)
    elif observation == 1:
        shape = (N, num_actions, num_actions)
    else:
        shape = (N, num_actions**(N-1), num_actions)

    # Initialize based on init_method
    if init_method == 0:
        return np.random.uniform(0, 1, shape)
    elif init_method == 1:
        return np.zeros(shape)
    else:
        return np.full(shape, 100.0)

# ============ Observation / State Indexing ============

def get_state_index(observation, last_actions, agent_idx, winning_action, num_actions):
    """
    observation:
      0 => no state => return None
      1 => 'winning_action' as state
      2 => encode all opponents' last actions into a single integer
    """
    if observation == 0:
        return None
    elif observation == 1:
        return winning_action
    else:
        # observation=2 => flatten opponents' actions
        # e.g. if N=3, each opponent can be in [0..(num_actions-1)],
        # so we get an index in [0..(num_actions^(N-1)-1)]
        opp = np.delete(last_actions, agent_idx)
        idx = 0
        for a in opp:
            idx = idx * num_actions + a
        return idx

# ============ Action Selection (Epsilon-greedy or Boltzmann) ============

def choose_action(Q, agent_idx, state_idx, egreedy, eps, beta, num_actions):
    """
    egreedy=1 => Epsilon-greedy
    egreedy=0 => Boltzmann
    """
    if state_idx is None:
        vals = Q[agent_idx]  # shape (num_actions,)
    else:
        vals = Q[agent_idx, state_idx]  # shape (num_actions,)

    if egreedy == 1:
        # Epsilon-greedy
        if np.random.rand() > eps:
            return np.argmax(vals)
        else:
            return np.random.choice(range(num_actions))
    else:
        # Boltzmann
        mx = np.max(vals)
        logits = (vals - mx) / beta
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return np.random.choice(range(num_actions), p=probs)

def exploratory_strategy(Q, N, egreedy, num_actions, observation,
                         last_actions, winning_action, eps, beta):
    """
    Decide each agent's action given Q, using the appropriate
    state index (depending on observation mode).
    """
    actions = np.zeros(N, dtype=int)
    for i in range(N):
        s_idx = get_state_index(observation, last_actions, i, winning_action, num_actions)
        actions[i] = choose_action(Q, i, s_idx, egreedy, eps, beta, num_actions)
    return actions

# ============ Reward Computation ============

def actions2rewards(actions, valuations, design, action2bid, method=0):
    """
    If method=0 => single random winner among top-bidding agents
    If method=1 => shared payoff among top-bidding agents
    design in [0..1], 0 => 'second-price', 1 => 'first-price'
    or anything in between (0.25, 0.5, etc.)
    """
    bids = action2bid[actions]
    uniq_sort = np.sort(np.unique(bids))[::-1]
    highest = uniq_sort[0]
    second = highest if len(uniq_sort) == 1 else uniq_sort[1]
    pay = design*highest + (1-design)*second
    winners_idx = np.where(bids == highest)[0]

    if method == 0:
        # Single random winner
        w = np.random.choice(winners_idx)
        r = np.zeros_like(bids)
        r[w] = valuations[w] - pay
    else:
        # Shared among the top
        top_mask = (bids == highest).astype(int)
        w = np.sum(top_mask)
        r = (1.0/w)*(valuations - pay)*top_mask
    return r

def counterfactual_reward(agent_idx, actions, valuations, design, num_actions, action2bid):
    """
    For synchronous updates. Compute reward vector for each possible action
    for 'agent_idx', while keeping other agents' actions fixed.
    We use method=1 => shared payoff, as in original code.
    """
    action_copy = actions.copy()
    cf_rewards = np.zeros(num_actions)
    for a in range(num_actions):
        action_copy[agent_idx] = a
        cf_rewards[a] = actions2rewards(
            action_copy, valuations, design, action2bid, method=1
        )[agent_idx]
    return cf_rewards

# ============ Q-value Update (Q-learning or SARSA) ============

def update_qvalues(Q, actions, rewards, alpha, gamma, asynchronous,
                   winning_action, observation, algorithm,
                   last_actions, next_actions,
                   design, valuations, num_actions, action2bid):
    """
    - algorithm=0 => Q-learning
    - algorithm=1 => SARSA
    - asynchronous=1 => 'one-step' update
    - asynchronous=0 => synchronous/counterfactual
    """
    N = Q.shape[0]
    for i in range(N):
        act_i = actions[i]
        r_i = rewards[i]
        s_idx = get_state_index(observation, last_actions, i, winning_action, num_actions)

        if asynchronous == 1:
            # --- Asynchronous update (original style) ---
            if observation == 1:
                # Q shape => Q[i, winning_action, act_i]
                if algorithm == 0:
                    # Q-learning => target uses max Q
                    Q[i, winning_action, act_i] = \
                        (1 - alpha)*Q[i, winning_action, act_i] + \
                        alpha*(r_i + gamma*np.max(Q[i, winning_action]))
                else:
                    # SARSA => target uses Q[i, winning_action, next_actions[i]]
                    Q[i, winning_action, act_i] = \
                        (1 - alpha)*Q[i, winning_action, act_i] + \
                        alpha*(r_i + gamma*Q[i, winning_action, next_actions[i]] - Q[i, winning_action, act_i])
            else:
                # Q shape => depends on s_idx
                if s_idx is None:
                    # observation=0 => Q[i, act_i] is a single dimension
                    if algorithm == 0:
                        Q[i, act_i] = (1-alpha)*Q[i, act_i] + \
                            alpha*(r_i + gamma*np.max(Q[i]))
                    else:
                        Q[i, act_i] = (1-alpha)*Q[i, act_i] + \
                            alpha*(r_i + gamma*Q[i, next_actions[i]] - Q[i, act_i])
                else:
                    # observation=2 => Q[i, s_idx, act_i]
                    if algorithm == 0:
                        Q[i, s_idx, act_i] = (1-alpha)*Q[i, s_idx, act_i] + \
                            alpha*(r_i + gamma*np.max(Q[i, s_idx]))
                    else:
                        na = next_actions[i]
                        Q[i, s_idx, act_i] = (1-alpha)*Q[i, s_idx, act_i] + \
                            alpha*(r_i + gamma*Q[i, s_idx, na] - Q[i, s_idx, act_i])

        else:
            # --- Synchronous update => counterfactual style ---
            r_vec = counterfactual_reward(i, actions, valuations, design, num_actions, action2bid)
            # 'old_vals' => row in Q we want to update
            if s_idx is None:
                old_vals = Q[i]  # shape (num_actions,)
            else:
                old_vals = Q[i, s_idx]  # shape (num_actions,)

            if algorithm == 0:
                # Q-learning => best action from old_vals
                target = r_vec + gamma*np.max(old_vals)
            else:
                # SARSA => next_actions used
                na = next_actions[i]
                target = r_vec + gamma*old_vals[na]

            new_vals = (1 - alpha)*old_vals + alpha*target
            if s_idx is None:
                Q[i] = new_vals
            else:
                Q[i, s_idx] = new_vals

    return Q

# ============ Main Experiment ============

def experiment(N, alpha, gamma, egreedy, asynchronous, design,
               num_actions, explore_frac=0.5, verbose=0,
               observation=1,   # 0=no state,1=only winning action,2=all opponents
               algorithm=0,     # 0=Q-learning,1=SARSA
               init_method=0    # 0=Random,1=Zeros,2=Optimistic(100)
               ):
    """
    Returns:
      revenue, time_to_converge, volatility,
      N, alpha, gamma, egreedy, asynchronous, design,
      num_actions, explore_frac, observation, algorithm, init_method,
      action_history, winning_bid_history
    """
    # 1) Initialize Q
    Q = initialize_qtable(N, num_actions, observation, init_method)
    valuations = np.ones(N)
    action2bid = np.linspace(0, 1, num_actions)

    # 2) Exploration parameters
    eps, beta = 1.0, 1.0
    min_eps, min_beta = 0.01, 0.01
    total_episodes = 100_000
    T_expl = int(explore_frac * total_episodes)
    eps_decay = (min_eps / eps) ** (1.0 / T_expl)
    beta_decay = (min_beta / beta) ** (1.0 / T_expl)

    # 3) Logs
    wbid_history = []
    act_history = []
    win_action = 0
    last_actions = np.zeros(N, dtype=int)  # needed if observation=2

    # 4) Main loop
    for ep in range(total_episodes):
        # (a) pick actions
        acts = exploratory_strategy(Q, N, egreedy, num_actions, observation,
                                    last_actions, win_action, eps, beta)
        # (b) if SARSA => predict next actions
        if algorithm == 1:
            nxt_acts = exploratory_strategy(Q, N, egreedy, num_actions, observation,
                                            acts, np.max(acts), eps, beta)
        else:
            nxt_acts = acts.copy()

        # (c) compute rewards
        rwds = actions2rewards(acts, valuations, design, action2bid)

        # (d) Q-update
        Q = update_qvalues(Q, acts, rwds, alpha, gamma, asynchronous,
                           win_action, observation, algorithm,
                           last_actions, nxt_acts,
                           design, valuations, num_actions, action2bid)

        # (e) decay
        if eps > min_eps:
            eps = max(min_eps, eps * eps_decay)
        if beta > min_beta:
            beta = max(min_beta, beta * beta_decay)

        # (f) record
        bids = acts / (num_actions - 1)
        wbid = np.max(bids)
        win_action = np.max(acts)
        wbid_history.append(wbid)
        act_history.append(action2bid[acts])
        last_actions = acts

        # (g) early stopping
        if ep > 1000 and np.std(wbid_history[-1000:]) < 1e-3 and (eps == min_eps or beta == min_beta):
            break

    # 5) Final metrics
    final_vals = wbid_history[-1000:] if len(wbid_history) > 1000 else wbid_history
    revenue = np.mean(final_vals)
    volatility = np.std(final_vals)
    time_to_converge = len(wbid_history)

    return (
        revenue,
        time_to_converge,
        volatility,
        N, alpha, gamma, egreedy, asynchronous, design,
        num_actions, explore_frac, observation, algorithm, init_method,
        act_history, wbid_history
    )

# ============ Demo / Data Collection ============

if __name__ == "__main__":
    print("[Generating example runs with 2nd price and 1st price]")

    # Example parameters
    example_params = {
        'N': 4,
        'alpha': 0.01,
        'gamma': 0.99,
        'egreedy': 0,         # Boltzmann
        'asynchronous': 0,    # synchronous
        'design': 0.0,        # second-price
        'num_actions': 6,
        'explore_frac': 0.9,
        'verbose': 0,
        'observation': 1,     # track only winning action
        'algorithm': 0,       # Q-learning
        'init_method': 0      # random init
    }

    def moving_average(x, w=1000):
        return np.convolve(x, np.ones(w), 'valid') / w

    # Run 2nd price example
    out = experiment(**example_params)
    act_hist, bid_hist = np.array(out[-2]), np.array(out[-1])
    limit = min(len(act_hist), 100_000)
    plt.figure()
    for i in range(act_hist.shape[1]):
        plt.plot(moving_average(act_hist[:limit, i]), label=f'Bidder {i+1}')
    plt.plot(moving_average(bid_hist[:limit]), label='Winning Bid', c='black')
    plt.title("Second Price Auction Example")
    plt.legend()
    if not os.path.exists("figures"):
        os.makedirs("figures")
    plt.savefig("figures/second-price-visual.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Switch to 1st price
    example_params['design'] = 1.0
    out = experiment(**example_params)
    act_hist, bid_hist = np.array(out[-2]), np.array(out[-1])
    limit = min(len(act_hist), 100_000)
    plt.figure()
    for i in range(act_hist.shape[1]):
        plt.plot(moving_average(act_hist[:limit, i]), label=f'Bidder {i+1}')
    plt.plot(moving_average(bid_hist[:limit]), label='Winning Bid', c='black')
    plt.title("First Price Auction Example")
    plt.legend()
    plt.savefig("figures/first-price-visual.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("[Sampling random sets of parameters and saving new_data.csv]")
    # Example random sampling
    parameters = {
        'N':            [2,4],
        'alpha':        [0.001,0.01],
        'gamma':        [0.9,0.99],
        'egreedy':      [0,1],
        'design':       [0.0,1.0],
        'asynchronous': [0,1],
        'num_actions':  [6],
        'explore_frac': [0.5,0.9],
        'observation':  [0,1,2],
        'algorithm':    [0,1],
        'init_method':  [0,1,2]
    }

    K = 50
    data = []
    for _ in tqdm(range(K), desc="Simulations"):
        local_params = {}
        for key, vals in parameters.items():
            local_params[key] = random.choice(vals)
        out = experiment(**local_params)
        # out => (revenue, time_to_converge, volatility, N, alpha, ..., init_method, act_hist, wbid_hist)
        # We'll store the first 13 columns, ignoring action histories
        data.append(out[:14])

    if not os.path.exists("data"):
        os.makedirs("data")

    cols = [
        'revenue', 'time_to_converge', 'volatility',
        'N','alpha','gamma','egreedy','asynchronous','design',
        'num_actions','explore_frac','observation','algorithm','init_method'
    ]
    df_main = pd.DataFrame(data, columns=cols)
    df_main.to_csv("data/new_data.csv", index=False)
    print("Data collection complete.")
