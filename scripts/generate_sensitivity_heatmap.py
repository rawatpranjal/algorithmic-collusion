#!/usr/bin/env python3
"""
Generate a sensitivity heatmap showing total-order Sobol indices (S_T)
for platform revenue across all six sub-experiments.

Each cell shows the share of revenue variance attributable to that factor
(including all interaction effects involving it).  Gray cells indicate
factors absent from the experimental design.

Outputs:
    paper/figures/sensitivity_heatmap.png
    paper/figures/sensitivity_heatmap.pdf

Usage:
    python scripts/generate_sensitivity_heatmap.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURE_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(FIGURE_DIR, exist_ok=True)

DPI = 300

# -----------------------------------------------------------------
# Experiment definitions: (key, response_var, short_label)
# -----------------------------------------------------------------
EXPERIMENTS = [
    ("1",  "avg_rev_last_1000",     "1a"),
    ("2",  "avg_rev_last_1000",     "1b"),
    ("3a", "avg_rev_last_1000",     "2a"),
    ("3b", "avg_rev_last_1000",     "2b"),
    ("4a", "mean_platform_revenue", "3a"),
    ("4b", "mean_platform_revenue", "3b"),
]

ALGORITHM_LABELS = [
    "Q-learn", "Q-learn", "LinUCB", "Thompson", "Pacing", "PI Pacing",
]

# -----------------------------------------------------------------
# Factor groups (rows, top to bottom)
# Includes only factors with S_T >= 0.02 in at least one experiment.
# -----------------------------------------------------------------
FACTOR_GROUPS = [
    ("Market Structure", [
        "n_bidders",
        "auction_type",
        "reserve_price",
    ]),
    ("Budget / Objective", [
        "budget_multiplier",
        "objective",
    ]),
    ("Value Environment", [
        "sigma",
        "eta",
        "state_info",
        "context_richness",
    ]),
    ("Algorithm Tuning", [
        "gamma",
        "asynchronous",
        "info_feedback",
        "exploration_intensity",
    ]),
]

# -----------------------------------------------------------------
# Readable names (plain text for matplotlib)
# -----------------------------------------------------------------
READABLE_NAMES = {
    "n_bidders": "Number of bidders",
    "auction_type": "Auction format",
    "reserve_price": "Reserve price",
    "budget_multiplier": "Budget multiplier",
    "objective": "Bidder objective",
    "sigma": "Value dispersion (\u03c3)",
    "eta": "Affiliation (\u03b7)",
    "state_info": "State information",
    "context_richness": "Context richness",
    "gamma": "Discount factor (\u03b3)",
    "asynchronous": "Update mode",
    "info_feedback": "Information feedback",
    "exploration_intensity": "Exploration intensity",
}


def load_sobol_data():
    """Load analytical Sobol S_T values and R-squared for each experiment."""
    data = {}
    for exp_key, response_key, _ in EXPERIMENTS:
        json_path = os.path.join(
            RESULTS_DIR, f"exp{exp_key}", "sensitivity", "sensitivity_results.json"
        )
        with open(json_path) as f:
            raw = json.load(f)
        sobol = raw["analytical_sobol"][response_key]
        data[exp_key] = {
            "st": sobol["st"],
            "r_squared": sobol["r_squared"],
        }
    return data


def make_heatmap():
    """Build and save the sensitivity heatmap."""
    data = load_sobol_data()

    # Flatten factor list and record group boundaries
    all_factors = []
    group_starts = []
    for _, factors in FACTOR_GROUPS:
        group_starts.append(len(all_factors))
        all_factors.extend(factors)

    n_rows = len(all_factors)
    n_cols = len(EXPERIMENTS)

    # Build data matrix (NaN where factor is absent)
    matrix = np.full((n_rows, n_cols), np.nan)
    for j, (exp_key, _, _) in enumerate(EXPERIMENTS):
        st = data[exp_key]["st"]
        for i, factor in enumerate(all_factors):
            if factor in st:
                matrix[i, j] = st[factor]

    r_squared = [data[ek]["r_squared"] for ek, _, _ in EXPERIMENTS]

    # ── Figure ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5.5))

    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="#f0f0f0")

    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(
        masked, cmap=cmap, vmin=0, vmax=0.6,
        aspect="auto", interpolation="nearest",
    )

    # ── Cell text (show value if S_T >= 0.03) ──────────────────
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if np.isnan(val) or val < 0.03:
                continue
            color = "white" if val > 0.35 else "black"
            weight = "bold" if val >= 0.15 else "normal"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=8, color=color, fontweight=weight,
            )

    # ── Column labels (top) ────────────────────────────────────
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [lab for _, _, lab in EXPERIMENTS], fontsize=10, fontweight="bold",
    )
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # Algorithm sub-labels below experiment names
    for j, alg in enumerate(ALGORITHM_LABELS):
        ax.text(
            j, -0.85, alg,
            ha="center", va="bottom",
            fontsize=7.5, color="#666666", fontstyle="italic",
        )

    # ── Row labels (left) ──────────────────────────────────────
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(
        [READABLE_NAMES[f] for f in all_factors], fontsize=9,
    )

    # ── Group separators (white lines between groups) ──────────
    for gs in group_starts[1:]:
        ax.axhline(gs - 0.5, color="white", linewidth=2.5)

    # ── Learning | Pacing vertical divider ─────────────────────
    ax.axvline(3.5, color="#888888", linewidth=1.2, linestyle="--")

    # ── R-squared annotation row below the heatmap ─────────────
    for j, r2 in enumerate(r_squared):
        ax.text(
            j, n_rows, f"$R^2$={r2:.2f}",
            ha="center", va="top",
            fontsize=7.5, color="#333333",
        )

    # ── Limits (extra room for sub-labels top, R² bottom) ─────
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows + 0.7, -1.2)

    # ── Colorbar ───────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label("$S_T$ (total-order Sobol index)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # ── Clean up ───────────────────────────────────────────────
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    fig.tight_layout()

    # ── Save ───────────────────────────────────────────────────
    png_path = os.path.join(FIGURE_DIR, "sensitivity_heatmap.png")
    pdf_path = os.path.join(FIGURE_DIR, "sensitivity_heatmap.pdf")
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    make_heatmap()
