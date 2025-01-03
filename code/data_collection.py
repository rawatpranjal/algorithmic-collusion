#!/usr/bin/env python3

import os
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============ Q-LEARNING FUNCTIONS ============

def initialize_qtable(N, num_actions, feedback):
    """Initialize Q-table."""
    if feedback == 1:
        return np.random.uniform(0,1,(N, num_actions,num_actions))
    else:
        return np.random.uniform(0,1,(N, num_actions))

def exploratory_strategy(Q, N, egreedy, num_actions, past_win, winning_bid, eps, beta):
    """Decide actions for all agents, given Q-values."""
    actions = np.ones(N, dtype=int)
    for agent_idx in range(N):
        if egreedy == 1:  # Epsilon-greedy
            if past_win == 1:
                if np.random.rand() > eps:
                    actions[agent_idx] = np.argmax(Q[agent_idx, winning_bid])
                else:
                    actions[agent_idx] = np.random.choice(range(num_actions))
            else:
                if np.random.rand() > eps:
                    actions[agent_idx] = np.argmax(Q[agent_idx])
                else:
                    actions[agent_idx] = np.random.choice(range(num_actions))
        else:  # Boltzmann
            if past_win == 1:
                max_Q = np.max(Q[agent_idx, winning_bid])
                logits = (Q[agent_idx, winning_bid] - max_Q) / beta
                probs = np.exp(logits) / np.sum(np.exp(logits))
                actions[agent_idx] = np.random.choice(range(num_actions), p=probs)
            else:
                max_Q = np.max(Q[agent_idx])
                logits = (Q[agent_idx] - max_Q) / beta
                probs = np.exp(logits) / np.sum(np.exp(logits))
                actions[agent_idx] = np.random.choice(range(num_actions), p=probs)
    return actions

def actions2rewards(actions, valuations, design, action2bid, method=0):
    """Compute rewards for each agent."""
    bids = action2bid[actions]
    unique_values_sorted = np.sort(np.unique(bids))[::-1]
    first_highest_value = unique_values_sorted[0]
    second_highest_value = first_highest_value if len(unique_values_sorted) == 1 else unique_values_sorted[1]
    winners_payment = design * first_highest_value + (1 - design)*second_highest_value
    winners_idx = np.where(bids == first_highest_value)[0]

    if method == 0:
        # Single random winner among top bidders
        winner_idx = np.random.choice(winners_idx)
        rewards = np.zeros_like(bids)
        rewards[winner_idx] = valuations[winner_idx] - winners_payment
    else:
        # Shared reward among top bidders
        winning_bid_idx = np.where(bids == first_highest_value, 1, 0)
        no_of_winners = np.sum(winning_bid_idx)
        rewards = (1/no_of_winners) * (valuations - winners_payment) * winning_bid_idx

    return rewards

def counterfactual_reward(agent_idx, actions, valuations, design, num_actions, action2bid):
    """Compute reward vector for each possible action of a single agent."""
    action_copy = actions.copy()
    cf_rewards = np.zeros(num_actions)
    for agent_action in range(num_actions):
        action_copy[agent_idx] = agent_action
        cf_rewards[agent_action] = actions2rewards(
            action_copy, valuations, design, action2bid, method=1
        )[agent_idx]
    return cf_rewards

def update_qvalues(rewards, actions, Q, feedback, asynchronous, winning_bid,
                   alpha, gamma, valuations, design, action2bid):
    """Update Q-table."""
    N = Q.shape[0]
    for agent_idx in range(N):
        action = actions[agent_idx]
        reward = rewards[agent_idx]
        if asynchronous == 1:
            if feedback == 1:
                Q[agent_idx, winning_bid, action] = \
                    (1 - alpha)*Q[agent_idx, winning_bid, action] + \
                    alpha*(reward + gamma*np.max(Q[agent_idx, winning_bid]))
            else:
                Q[agent_idx, action] = \
                    (1 - alpha)*Q[agent_idx, action] + \
                    alpha*(reward + gamma*np.max(Q[agent_idx]))
        else:
            # Synchronous update
            reward_vec = counterfactual_reward(
                agent_idx, actions, valuations, design,
                Q.shape[1], action2bid
            )
            if feedback == 1:
                Q[agent_idx, winning_bid, :] = \
                    (1 - alpha)*Q[agent_idx, winning_bid, :] + \
                    alpha*(reward_vec + gamma*np.max(Q[agent_idx, winning_bid]))
            else:
                Q[agent_idx] = \
                    (1 - alpha)*Q[agent_idx] + \
                    alpha*(reward_vec + gamma*np.max(Q[agent_idx]))
    return Q

def experiment(N, alpha, gamma, egreedy, asynchronous, design, feedback,
               num_actions, explore_frac=0.5, verbose=0):
    """Run a single Q-learning experiment with explore_frac determining decay."""
    Q = initialize_qtable(N, num_actions, feedback)
    common_valuation = 1
    valuations = np.ones(N)*common_valuation
    action2bid = np.linspace(0, common_valuation, num_actions)

    # Exploration parameters
    eps = 1.0
    beta = 1.0
    min_eps = 0.01
    min_beta = 0.01

    num_episodes = 100_000
    T_expl = int(explore_frac * num_episodes)

    # Reverse-engineer decay from explore_frac
    eps_decay = (min_eps / eps) ** (1.0 / T_expl)
    beta_decay = (min_beta / beta) ** (1.0 / T_expl)

    winning_bid_history = []
    action_history = []
    winning_action = 0

    for i in range(num_episodes):
        actions = exploratory_strategy(Q, N, egreedy, num_actions, feedback, winning_action, eps, beta)
        rewards = actions2rewards(actions, valuations, design, action2bid)
        Q = update_qvalues(rewards, actions, Q, feedback, asynchronous, winning_action,
                           alpha, gamma, valuations, design, action2bid)

        # Decay eps, beta until hitting min values
        if eps > min_eps:
            eps = max(min_eps, eps * eps_decay)
        if beta > min_beta:
            beta = max(min_beta, beta * beta_decay)

        bids = actions * common_valuation / (num_actions - 1)
        winning_bid = np.max(bids)
        winning_action = np.max(actions)
        winning_bid_history.append(winning_bid)
        action_history.append(action2bid[actions])

        # Early stop if stable
        if (eps == min_eps or beta == min_beta) and len(winning_bid_history) > 1000:
            if np.std(winning_bid_history[-1000:]) < 0.001:
                break

    # Final stats
    if len(winning_bid_history) >= 1000:
        Y = np.mean(winning_bid_history[-1000:])
        Y_std = np.std(winning_bid_history[-1000:])
        Y_min = np.min(winning_bid_history[-1000:])
        Y_max = np.max(winning_bid_history[-1000:])
    else:
        Y = np.mean(winning_bid_history)
        Y_std = np.std(winning_bid_history)
        Y_min = np.min(winning_bid_history)
        Y_max = np.max(winning_bid_history)

    return (
        Y, len(winning_bid_history), Y_std, Y_min, Y_max,
        N, alpha, gamma, egreedy, asynchronous, design,
        feedback, num_actions, explore_frac, action_history, winning_bid_history
    )

if __name__ == "__main__":
    print("[Generating plots for design=0 (2nd price) and design=1 (1st price)]")

    # Example parameters
    example_params = {
        'N': 4, 'alpha': 0.01, 'gamma': 0.99,
        'egreedy': 0, 'asynchronous': 0, 'feedback': 1,
        'num_actions': 6, 'explore_frac': 0.9, 'verbose': 0
    }

    def moving_average(x, w=1000):
        return np.convolve(x, np.ones(w), 'valid') / w

    # 2nd price
    example_params['design'] = 0
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, act_hist, bid_hist = experiment(**example_params)
    act_hist, bid_hist = np.array(act_hist), np.array(bid_hist)
    limit = min(len(act_hist), 100_000)
    plt.figure()
    for i in range(act_hist.shape[1]):
        plt.plot(moving_average(act_hist[:limit,i]), label=f'Bidder {i+1}')
    plt.plot(moving_average(bid_hist[:limit]), label='Winning Bid', c='black')
    plt.title("Second Price Auction")
    plt.legend()
    if not os.path.exists("figures"):
        os.makedirs("figures")
    plt.savefig("figures/second-price-visual.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 1st price
    example_params['design'] = 1
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, act_hist, bid_hist = experiment(**example_params)
    act_hist, bid_hist = np.array(act_hist), np.array(bid_hist)
    limit = min(len(act_hist), 100_000)
    plt.figure()
    for i in range(act_hist.shape[1]):
        plt.plot(moving_average(act_hist[:limit,i]), label=f'Bidder {i+1}')
    plt.plot(moving_average(bid_hist[:limit]), label='Winning Bid', c='black')
    plt.title("First Price Auction")
    plt.legend()
    plt.savefig("figures/first-price-visual.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("[Sampling random sets of parameters and saving data.csv]")
    parameters = {
        'N':        [2,4,6,8],
        'alpha':    [0.001,0.01, 0.1],
        'gamma':    [0.0,0.9,0.99],
        'egreedy':  [0,1],
        'design':   [0.0,0.25,0.5,0.75,1.0],
        'asynchronous':[0,1],
        'feedback': [0,1],
        'num_actions':[6,11,21],
        'explore_frac': [0.5,0.7,0.9]  # example fractions
    }

    K = 500
    data = []
    for _ in tqdm(range(K), desc="Simulations"):
        sampled_params = {}
        for key, vals in parameters.items():
            sampled_params[key] = random.choice(vals)
        out = experiment(**sampled_params)
        data.append(out[:14])  # drop action histories

    if not os.path.exists("data"):
        os.makedirs("data")

    cols = [
        'bid2val','episodes','bid2val_std','bid2val_min','bid2val_max',
        'N','alpha','gamma','egreedy','asynchronous','design','feedback',
        'num_actions','explore_frac'
    ]
    df_main = pd.DataFrame(data, columns=cols)
    df_main.to_csv("data/data.csv", index=False)

    print("Data collection complete.")
