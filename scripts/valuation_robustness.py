#!/usr/bin/env python3
"""
Valuation Model Robustness Check: Isolate FPA vs SPA Revenue Reversal.

2x2x2 factorial: auction_type x valuation_model x budget_structure.
Held constant: dual pacing, n_bidders=3, budget_mult=0.5, reserve=0, objective=value_max.
50 seeds per cell -> 400 runs.

Output: results/robustness/valuation_model/{data.csv, interaction_plot.png, anova.txt}
"""

import math
import os
import sys
import time
from itertools import product
from multiprocessing import Pool, cpu_count

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from experiments.exp4a import DualPacingAgent, run_auction, run_episode, MU_CLIP

# ── Constants ──────────────────────────────────────────────────────────
N_BIDDERS = 3
BUDGET_MULT = 0.5
RESERVE = 0.0
OBJECTIVE = "value_max"
SIGMA = 0.3
ETA = 0.5
N_SEEDS = 50

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "results", "robustness", "valuation_model"
)


# ── Valuation generators ──────────────────────────────────────────────
def generate_lognormal(rng, bidder_means, T):
    """LogNormal valuations: bidder i ~ LogNormal(mu_i, sigma)."""
    vals = np.column_stack([
        rng.lognormal(bidder_means[i], SIGMA, T) for i in range(len(bidder_means))
    ])
    expected = np.exp(bidder_means + SIGMA**2 / 2.0)
    return vals, expected


def generate_affiliated(rng, n_bidders, T):
    """Affiliated valuations: v_i = (1 - eta/2)*s_i + (eta/2)*mean(others)."""
    signals = rng.uniform(0, 1, (T, n_bidders))
    mean_others = (signals.sum(axis=1, keepdims=True) - signals) / max(n_bidders - 1, 1)
    vals = (1 - 0.5 * ETA) * signals + 0.5 * ETA * mean_others
    expected = np.full(n_bidders, 0.5)
    return vals, expected


# ── Single run ─────────────────────────────────────────────────────────
def run_single(args):
    """Run one configuration. Returns dict of results."""
    auction_type, val_model, budget_structure, seed = args
    rng = np.random.default_rng(seed)
    # Global seed for tie-breaking in run_auction
    np.random.seed(seed % (2**31))

    if budget_structure == "episodic":
        n_episodes, T = 100, 1000
        burn_in = 10
    else:
        n_episodes, T = 1, 100_000
        burn_in = 0

    # Draw bidder means once (used by lognormal; ignored by affiliated)
    bidder_means = rng.uniform(0.5, 1.5, N_BIDDERS)

    # Compute budgets normalized to expected value
    if val_model == "lognormal":
        expected = np.exp(bidder_means + SIGMA**2 / 2.0)
    else:
        expected = np.full(N_BIDDERS, 0.5)

    budgets = BUDGET_MULT * expected * T
    agents = [DualPacingAgent(budgets[i], T, OBJECTIVE) for i in range(N_BIDDERS)]

    total_revenue = 0.0
    counted = 0

    for ep in range(n_episodes):
        # Generate valuations
        if val_model == "lognormal":
            vals, _ = generate_lognormal(rng, bidder_means, T)
        else:
            vals, _ = generate_affiliated(rng, N_BIDDERS, T)

        # Reset agents (warm-start dual variable)
        for i in range(N_BIDDERS):
            agents[i].reset_episode(budgets[i])

        metrics = run_episode(agents, N_BIDDERS, T, auction_type, vals, RESERVE)

        if ep >= burn_in:
            total_revenue += metrics["platform_revenue"]
            counted += 1

    total_rounds = counted * T
    mean_rev = total_revenue / total_rounds if total_rounds > 0 else 0.0

    return {
        "seed": seed,
        "auction_type": auction_type,
        "valuation_model": val_model,
        "budget_structure": budget_structure,
        "mean_revenue_per_round": mean_rev,
    }


# ── ANOVA ──────────────────────────────────────────────────────────────
def run_anova(df):
    """Compute 2x2x2 factorial ANOVA with all interactions."""
    # Code factors as -1/+1
    df["A"] = np.where(df["auction_type"] == "first", 1, -1)
    df["V"] = np.where(df["valuation_model"] == "lognormal", 1, -1)
    df["B"] = np.where(df["budget_structure"] == "episodic", 1, -1)

    y = df["mean_revenue_per_round"].values
    n = len(y)
    grand_mean = y.mean()
    ss_total = np.sum((y - grand_mean) ** 2)

    # Effects via OLS: y = b0 + b1*A + b2*V + b3*B + b4*AV + b5*AB + b6*VB + b7*AVB
    X = np.column_stack([
        np.ones(n),
        df["A"], df["V"], df["B"],
        df["A"] * df["V"],
        df["A"] * df["B"],
        df["V"] * df["B"],
        df["A"] * df["V"] * df["B"],
    ])
    betas = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ betas
    ss_res = np.sum((y - y_hat) ** 2)
    ss_model = ss_total - ss_res

    names = ["Intercept", "Auction(FPA)", "Valuation(LogN)", "Budget(Episodic)",
             "Auction*Valuation", "Auction*Budget", "Valuation*Budget",
             "Auction*Val*Budget"]

    # Each effect has 1 df; SS_effect = n * beta^2 (balanced design)
    results = []
    df_res = n - 8
    ms_res = ss_res / df_res
    for i in range(1, 8):
        ss_eff = n * betas[i] ** 2
        f_val = ss_eff / ms_res
        p_val = 1 - scipy_stats.f.cdf(f_val, 1, df_res)
        results.append({
            "Effect": names[i],
            "Coefficient": betas[i],
            "SS": ss_eff,
            "F": f_val,
            "p-value": p_val,
            "Significant": "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "",
        })

    anova_df = pd.DataFrame(results)
    anova_df["R2_contribution"] = anova_df["SS"] / ss_total * 100

    summary = f"ANOVA Summary (N={n}, R²={ss_model/ss_total:.4f}, MSE={ms_res:.6f})\n"
    summary += "=" * 90 + "\n"
    summary += f"{'Effect':<25} {'Coef':>10} {'SS':>12} {'F':>10} {'p-value':>12} {'%R²':>8} {'Sig':>5}\n"
    summary += "-" * 90 + "\n"
    for _, row in anova_df.iterrows():
        summary += (
            f"{row['Effect']:<25} {row['Coefficient']:>10.6f} {row['SS']:>12.4f} "
            f"{row['F']:>10.2f} {row['p-value']:>12.2e} {row['R2_contribution']:>7.1f}% {row['Significant']:>5}\n"
        )
    return summary, anova_df


# ── Plotting ───────────────────────────────────────────────────────────
def plot_interaction(df, output_path):
    """2x2x2 grouped bar chart: FPA-SPA gap across valuation x budget combos."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax_idx, budget in enumerate(["episodic", "single_pool"]):
        sub = df[df["budget_structure"] == budget]
        ax = axes[ax_idx]

        x = np.arange(2)  # lognormal, affiliated
        width = 0.35

        for a_idx, (auction, color) in enumerate([("first", "#2196F3"), ("second", "#FF9800")]):
            means, cis = [], []
            for val_model in ["lognormal", "affiliated"]:
                cell = sub[(sub["auction_type"] == auction) & (sub["valuation_model"] == val_model)]
                m = cell["mean_revenue_per_round"].mean()
                se = cell["mean_revenue_per_round"].std() / np.sqrt(len(cell))
                means.append(m)
                cis.append(1.96 * se)

            offset = (a_idx - 0.5) * width
            bars = ax.bar(x + offset, means, width, yerr=cis, label=auction.upper(),
                          color=color, alpha=0.85, capsize=4, edgecolor="white", linewidth=0.5)

            # Add value labels
            for bar, m in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cis)*0.3,
                        f"{m:.4f}", ha="center", va="bottom", fontsize=8)

        budget_label = "Episodic (100 ep x 1K)" if budget == "episodic" else "Single Pool (1 ep x 100K)"
        ax.set_title(budget_label, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["LogNormal", "Affiliated"], fontsize=10)
        ax.set_xlabel("Valuation Model", fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel("Mean Revenue per Round", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("FPA vs SPA Revenue: Valuation Model x Budget Structure", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_fpa_spa_gap(df, output_path):
    """Bar chart showing FPA-SPA revenue gap for each val x budget combo."""
    fig, ax = plt.subplots(figsize=(8, 5))

    combos = list(product(["lognormal", "affiliated"], ["episodic", "single_pool"]))
    labels = []
    gaps = []
    cis = []

    for val_model, budget in combos:
        fpa = df[(df["auction_type"] == "first") & (df["valuation_model"] == val_model) &
                 (df["budget_structure"] == budget)]["mean_revenue_per_round"]
        spa = df[(df["auction_type"] == "second") & (df["valuation_model"] == val_model) &
                 (df["budget_structure"] == budget)]["mean_revenue_per_round"]

        gap = fpa.mean() - spa.mean()
        # SE of difference (independent samples)
        se = np.sqrt(fpa.var() / len(fpa) + spa.var() / len(spa))
        t_stat = gap / se if se > 0 else 0
        p_val = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), len(fpa) + len(spa) - 2))

        budget_short = "Episodic" if budget == "episodic" else "Single"
        val_short = "LogN" if val_model == "lognormal" else "Affil"
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        labels.append(f"{val_short}\n{budget_short}\n({sig})")
        gaps.append(gap)
        cis.append(1.96 * se)

    colors = ["#4CAF50" if g > 0 else "#F44336" for g in gaps]
    bars = ax.bar(range(len(gaps)), gaps, yerr=cis, color=colors, alpha=0.85,
                  capsize=5, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(gaps)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("FPA - SPA Revenue Gap", fontsize=11)
    ax.set_title("FPA-SPA Revenue Gap by Condition\n(Green = FPA better, Red = SPA better)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for bar, g in zip(bars, gaps):
        y = bar.get_height()
        sign = "+" if g > 0 else ""
        ax.text(bar.get_x() + bar.get_width()/2, y + (0.0005 if g > 0 else -0.0008),
                f"{sign}{g:.5f}", ha="center", va="bottom" if g > 0 else "top", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    configs = list(product(
        ["first", "second"],
        ["lognormal", "affiliated"],
        ["episodic", "single_pool"],
        range(N_SEEDS),
    ))

    n_cells = 8
    n_total = len(configs)
    print(f"Valuation Model Robustness Check")
    print(f"  {n_cells} cells x {N_SEEDS} seeds = {n_total} runs")
    print(f"  Factors: auction_type x valuation_model x budget_structure")
    print(f"  Held constant: dual pacing, N={N_BIDDERS}, budget_mult={BUDGET_MULT}, "
          f"reserve={RESERVE}, objective={OBJECTIVE}")
    print()

    workers = max(1, cpu_count() // 2)
    print(f"Running with {workers} workers...")
    t0 = time.time()

    with Pool(workers) as pool:
        results = pool.map(run_single, configs)

    elapsed = time.time() - t0
    print(f"Completed {n_total} runs in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Build DataFrame
    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "data.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Cell means
    print("\n" + "=" * 70)
    print("Cell Means (mean revenue per round)")
    print("=" * 70)
    cell_means = df.groupby(["auction_type", "valuation_model", "budget_structure"])[
        "mean_revenue_per_round"
    ].agg(["mean", "std", "count"])
    print(cell_means.to_string())

    # FPA-SPA gaps
    print("\n" + "=" * 70)
    print("FPA - SPA Revenue Gaps")
    print("=" * 70)
    for val_model in ["lognormal", "affiliated"]:
        for budget in ["episodic", "single_pool"]:
            fpa = df[(df["auction_type"] == "first") & (df["valuation_model"] == val_model) &
                     (df["budget_structure"] == budget)]["mean_revenue_per_round"]
            spa = df[(df["auction_type"] == "second") & (df["valuation_model"] == val_model) &
                     (df["budget_structure"] == budget)]["mean_revenue_per_round"]
            gap = fpa.mean() - spa.mean()
            se = np.sqrt(fpa.var()/len(fpa) + spa.var()/len(spa))
            t_stat = gap / se if se > 0 else 0
            p_val = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), len(fpa) + len(spa) - 2))
            direction = "FPA > SPA" if gap > 0 else "SPA > FPA"
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"  {val_model:>10} + {budget:>12}: gap = {gap:+.6f}  "
                  f"(t={t_stat:.2f}, p={p_val:.4f}) {sig}  [{direction}]")

    # ANOVA
    print("\n" + "=" * 70)
    anova_text, anova_df = run_anova(df)
    print(anova_text)

    anova_path = os.path.join(OUTPUT_DIR, "anova.txt")
    with open(anova_path, "w") as f:
        f.write(anova_text)
    print(f"Saved: {anova_path}")

    # Plots
    plot_interaction(df, os.path.join(OUTPUT_DIR, "interaction_plot.png"))
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'interaction_plot.png')}")

    plot_fpa_spa_gap(df, os.path.join(OUTPUT_DIR, "fpa_spa_gap.png"))
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'fpa_spa_gap.png')}")

    # Key finding
    print("\n" + "=" * 70)
    print("KEY FINDING")
    print("=" * 70)
    av_row = anova_df[anova_df["Effect"] == "Auction*Valuation"].iloc[0]
    ab_row = anova_df[anova_df["Effect"] == "Auction*Budget"].iloc[0]
    avb_row = anova_df[anova_df["Effect"] == "Auction*Val*Budget"].iloc[0]

    if av_row["p-value"] < 0.05 and ab_row["p-value"] >= 0.05:
        print("  -> VALUATION MODEL drives the FPA-SPA reversal")
        print(f"     Auction*Valuation: F={av_row['F']:.1f}, p={av_row['p-value']:.2e}, "
              f"{av_row['R2_contribution']:.1f}% of variance")
    elif ab_row["p-value"] < 0.05 and av_row["p-value"] >= 0.05:
        print("  -> BUDGET STRUCTURE drives the FPA-SPA reversal")
        print(f"     Auction*Budget: F={ab_row['F']:.1f}, p={ab_row['p-value']:.2e}, "
              f"{ab_row['R2_contribution']:.1f}% of variance")
    elif av_row["p-value"] < 0.05 and ab_row["p-value"] < 0.05:
        print("  -> BOTH valuation model AND budget structure contribute")
        print(f"     Auction*Valuation: F={av_row['F']:.1f}, p={av_row['p-value']:.2e}, "
              f"{av_row['R2_contribution']:.1f}%")
        print(f"     Auction*Budget:    F={ab_row['F']:.1f}, p={ab_row['p-value']:.2e}, "
              f"{ab_row['R2_contribution']:.1f}%")
    elif avb_row["p-value"] < 0.05:
        print("  -> THREE-WAY interaction: neither factor alone suffices")
        print(f"     Auction*Val*Budget: F={avb_row['F']:.1f}, p={avb_row['p-value']:.2e}, "
              f"{avb_row['R2_contribution']:.1f}%")
    else:
        print("  -> No significant interaction detected (unexpected)")

    print()


if __name__ == "__main__":
    main()
