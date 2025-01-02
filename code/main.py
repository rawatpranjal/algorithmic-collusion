#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script for:
1) Generating two example plots for first-price vs second-price auctions.
2) Running random sampling of parameter sets with Q-learning auctions and saving data.
3) Producing basic analysis, t-tests, regressions, advanced ML, etc.
"""

import os
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from stargazer.stargazer import Stargazer
import researchpy as rp
import scipy.stats as stats
from econml.dml import LinearDML
from econml.orf import DMLOrthoForest
from econml.sklearn_extensions.linear_model import WeightedLasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
import eli5
from eli5.sklearn import PermutationImportance

from tqdm import tqdm

# For pretty-printing arrays
np.set_printoptions(precision=4, suppress=True)


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
        if egreedy == 1:
            # Epsilon-greedy
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
        else:
            # Boltzmann
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
    """Compute rewards for each agent based on actions and auction type."""
    bids = action2bid[actions]
    unique_values_sorted = np.sort(np.unique(bids))[::-1]
    first_highest_value = unique_values_sorted[0]
    if len(unique_values_sorted) > 1:
        second_highest_value = unique_values_sorted[1]
    else:
        second_highest_value = first_highest_value

    winners_payment = design * first_highest_value + (1-design)*second_highest_value
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
    """Update Q-table according to Q-learning rules."""
    N = Q.shape[0]
    num_actions = Q.shape[1]
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
                agent_idx, actions, valuations, design, num_actions, action2bid
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
               num_actions, decay, verbose=0):
    """Run a single experiment with Q-learning bidders in repeated auctions."""
    Q = initialize_qtable(N, num_actions, feedback)
    common_valuation = 1
    valuations = np.ones(N)*common_valuation
    action2bid = np.linspace(0, common_valuation, num_actions)

    eps = 1.0
    beta = 1.0
    min_eps = 0.01
    min_beta = 0.01
    eps_decay = decay
    beta_decay = decay

    winning_bid_history = []
    action_history = []
    num_episodes = 250000
    winning_action = 0

    for _ in range(num_episodes):
        # decide actions
        actions = exploratory_strategy(Q, N, egreedy, num_actions, feedback, winning_action, eps, beta)
        # rewards
        rewards = actions2rewards(actions, valuations, design, action2bid)
        # update Q
        Q = update_qvalues(rewards, actions, Q, feedback, asynchronous, winning_action,
                           alpha, gamma, valuations, design, action2bid)
        # decay
        eps = max(min_eps, eps * eps_decay)
        beta = max(min_beta, beta * beta_decay)

        # track
        bids = actions*common_valuation/(num_actions - 1)
        winning_bid = np.max(bids)
        winning_action = np.max(actions)
        winning_bid_history.append(winning_bid)
        action_history.append(action2bid[actions])

        # early stopping if stable
        if (eps == min_eps or beta == min_beta) and len(winning_bid_history) > 1000:
            if np.std(winning_bid_history[-1000:]) < 0.001:
                break

    # final stats
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

    return (Y, len(winning_bid_history), Y_std, Y_min, Y_max,
            N, alpha, gamma, egreedy, asynchronous, design,
            feedback, num_actions, decay, action_history, winning_bid_history)


# ============ MAIN RUN ============

if __name__ == "__main__":

    print("[Step 1] Running sample experiments for design=0 and design=1, to create example plots.")

    # Example parameters for first-price vs second-price:
    example_params = {
        'N': 4,
        'alpha': 0.1,
        'gamma': 0.99,
        'egreedy': 0,
        'asynchronous': 1,
        'feedback': 1,
        'num_actions': 6,
        'decay': 0.9999,
        'verbose': 0
    }

    # 1) Second Price => design=0
    example_params['design'] = 0
    (Y, ep_count, Y_std, Y_min, Y_max, N, alpha, gamma, egreedy, async_, design_, fb,
     acts, decay_, action_hist, bid_hist) = experiment(**example_params)

    # Save second-price plot
    print(" Plotting second-price example convergence (design=0)...")
    action_hist = np.array(action_hist)
    bid_hist = np.array(bid_hist)

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    # We'll only plot first 130000 for clarity
    limit = min(len(action_hist), 130000)
    plt.figure(figsize=(10,5))
    plt.plot(moving_average(action_hist[:limit,0], 1000), label='Bidder 1', c='blue')
    plt.plot(moving_average(action_hist[:limit,1], 1000), label='Bidder 2', c='green')
    plt.plot(moving_average(action_hist[:limit,2], 1000), label='Bidder 3', c='orange')
    plt.plot(moving_average(action_hist[:limit,3], 1000), label='Bidder 4', c='red')
    plt.plot(moving_average(bid_hist[:limit], 1000), label='Winning Bid', c='black')
    plt.title("Avg Bids in the Second Price Auction")
    plt.legend()
    if not os.path.exists("figures"):
        os.makedirs("figures")
    plt.savefig("figures/second-price-visual.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2) First Price => design=1
    example_params['design'] = 1
    (Y, ep_count, Y_std, Y_min, Y_max, N, alpha, gamma, egreedy, async_, design_, fb,
     acts, decay_, action_hist, bid_hist) = experiment(**example_params)

    print(" Plotting first-price example convergence (design=1)...")
    action_hist = np.array(action_hist)
    bid_hist = np.array(bid_hist)
    limit = min(len(action_hist), 130000)
    plt.figure(figsize=(10,5))
    plt.plot(moving_average(action_hist[:limit,0], 1000), label='Bidder 1', c='blue')
    plt.plot(moving_average(action_hist[:limit,1], 1000), label='Bidder 2', c='green')
    plt.plot(moving_average(action_hist[:limit,2], 1000), label='Bidder 3', c='orange')
    plt.plot(moving_average(action_hist[:limit,3], 1000), label='Bidder 4', c='red')
    plt.plot(moving_average(bid_hist[:limit], 1000), label='Winning Bid', c='black')
    plt.title("Avg Bids in the First Price Auction")
    plt.legend()
    plt.savefig("figures/first-price-visual.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("[Step 2] Running random sampling of K=10 for demonstration and saving data.csv...")

    from tqdm import tqdm
    parameters = {
        'N':        [2,4],
        'alpha':    [0.01,0.05,0.1,0.5],
        'gamma':    [0.0,0.25,0.5,0.75,0.99],
        'egreedy':  [0,1],
        'design':   [0.0, 0.25, 0.5, 0.75,1.0],
        'asynchronous':[0,1],
        'feedback': [0,1],
        'num_actions':[6,21,31,51],
        'decay':    [0.999,0.9995,0.9999,0.99995,0.99999]
    }
    K = 10
    data = []
    for _ in tqdm(range(K), desc="Simulations"):
        sampled_params = {}
        for key, vals in parameters.items():
            sampled_params[key] = random.choice(vals)
        out = experiment(**sampled_params, verbose=0)
        # drop the last two entries (action_hist, bid_hist) for data
        data.append(out[:14])  # first 14 fields

    if not os.path.exists("data"):
        os.makedirs("data")

    df_cols = [
        'bid2val','episodes','bid2val_std','bid2val_min','bid2val_max',
        'N','alpha','gamma','egreedy','asynchronous','design','feedback',
        'num_actions','decay'
    ]
    df_main = pd.DataFrame(data, columns=df_cols)
    df_main_path = os.path.join("data", "data.csv")
    df_main.to_csv(df_main_path, index=False)

    print("[Step 3] Basic analysis of saved data...")

    df = pd.read_csv(df_main_path)
    print(df.head())
    print(df.describe())

    # Quick boxplots
    sns.boxplot(data=df, x="design", y="bid2val")
    plt.savefig("figures/boxplot_bid2val.png")
    plt.close()

    sns.boxplot(data=df, x="design", y="bid2val_std")
    plt.savefig("figures/boxplot_vol.png")
    plt.close()

    sns.boxplot(data=df, x="design", y="episodes")
    plt.savefig("figures/boxplot_episodes.png")
    plt.close()

    # T-tests
    df0 = df[df['design']==0]['bid2val']
    df1 = df[df['design']==1]['bid2val']
    if len(df0)>1 and len(df1)>1:
        summary, _ = rp.ttest(group1=df0, group1_name="Second Price",
                              group2=df1, group2_name="First Price")
        print("T-test summary:\n", summary)
        print(stats.ttest_ind(df0, df1))

    # OLS
    est1 = smf.ols('bid2val ~ design', data=df).fit()
    est2 = smf.ols('bid2val ~ design + N + alpha + gamma + egreedy + asynchronous + feedback + num_actions + decay', data=df).fit()
    print("\nOLS 1:\n", est1.summary())
    print("\nOLS 2:\n", est2.summary())
    sg1 = Stargazer([est1])
    sg1.title("Regression 1: Simple OLS")
    sg2 = Stargazer([est2])
    sg2.title("Regression 2: OLS w/ Covariates")
    with open(os.path.join("figures","regression_1.tex"),"w") as f:
        f.write(sg1.render_latex())
    with open(os.path.join("figures","regression_2.tex"),"w") as f:
        f.write(sg2.render_latex())

    # Double ML
    df_ml = df.dropna()
    if len(df_ml)>1:
        y = df_ml['bid2val']
        T = df_ml['design']
        X = df_ml.drop(['bid2val','episodes','bid2val_std','bid2val_min','bid2val_max','design'], axis=1)
        dml_est = LinearDML()
        dml_est.fit(y, T, X=X)
        print("\nLinearDML:\n", dml_est.summary().as_latex())

        # DMLOrthoForest
        np.random.seed(123)
        orf = DMLOrthoForest(n_trees=10, max_depth=4,
                             model_Y=WeightedLasso(alpha=0.01),
                             model_T=WeightedLasso(alpha=0.01),
                             random_state=123)
        orf.fit(y, T, X=X)
        te = orf.effect(X)
        if X.shape[1]>0:
            plt.figure()
            plt.scatter(X.iloc[:,0], te, alpha=0.5)
            plt.title("DMLOrthoForest Estimated Treatment Effects")
            plt.xlabel(X.columns[0])
            plt.ylabel("Treatment Effect")
            plt.savefig("figures/orf_treatment_effects.png")
            plt.close()

    # Random Forest example
    y_cv = df['episodes']
    X_cv = df.drop(['bid2val','episodes','bid2val_std','bid2val_min','bid2val_max'], axis=1)
    if len(X_cv)>0 and len(y_cv)>0:
        rf = RandomForestRegressor(n_estimators=50,max_depth=3,random_state=123)
        kf = KFold(n_splits=3, shuffle=True, random_state=123)
        scores = cross_val_score(rf, X_cv, y_cv, cv=kf, scoring='r2')
        print("RF R^2 scores:", scores)
        print("RF mean R^2:", scores.mean())
        rf.fit(X_cv,y_cv)
        imps = rf.feature_importances_
        inds = np.argsort(imps)[::-1]
        print("\nFeature importances (RandomForestRegressor):")
        for i, idx in enumerate(inds):
            print(f"{i+1}. {X_cv.columns[idx]}: {imps[idx]:.4f}")
        perm_imp = PermutationImportance(rf).fit(X_cv, y_cv)
        print("\nPermutation Importances:\n", eli5.format_as_text(eli5.explain_weights(perm_imp)))

    print("\nAll steps complete.")
