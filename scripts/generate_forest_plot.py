#!/usr/bin/env python3
"""
Generate a master forest plot showing standardized marginal effects of
structural interventions on platform revenue across all 6 experiments.

Effects are normalized as percentage of each experiment's grand mean,
making them comparable across experiments with different revenue scales.
Point color encodes the total-order Sobol index (S_T), indicating the
share of revenue variance attributable to each factor.

Outputs:
    paper/figures/master_forest.png    (combined, all groups)
    paper/figures/master_forest.pdf
    paper/figures/master_forest_1.png  (top half of groups)
    paper/figures/master_forest_1.pdf
    paper/figures/master_forest_2.png  (bottom half of groups)
    paper/figures/master_forest_2.pdf

Usage:
    python scripts/generate_forest_plot.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np

FIGURE_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

DPI = 300

# -----------------------------------------------------------------
# Grand means (for normalization to % of grand mean)
# -----------------------------------------------------------------
GRAND_MEANS = {
    "1a":  0.815736,
    "1b":  0.458949,
    "2a":  0.458698,
    "2b":  0.508970,
    "3a":  4256.536538,
    "3b":  3651.092477,
}

# -----------------------------------------------------------------
# Raw coefficients, standard errors, and Sobol S_T indices
# from estimation_results.json and sensitivity_results.json
# -----------------------------------------------------------------
# Each entry: (experiment_key, coefficient, std_error, S_T)

AUCTION_FORMAT = [
    ("1a",  0.003086,    0.004904,  0.04459),
    ("1b",  -0.026795,   0.008449,  0.07848),
    ("2a", -0.048182,   0.003858,  0.09209),
    ("2b", -0.023111,   0.006037,  0.07496),
    ("3a", 184.033593,  50.880581, 0.00848),
    ("3b", 458.330056,  41.549529, 0.11583),
]

MARKET_THICKNESS = [
    ("1a",  0.087059,    0.004904,  0.23763),
    ("1b",  0.055151,    0.008449,  0.22268),
    ("2a", 0.130764,    0.003858,  0.55059),
    ("2b", 0.052454,    0.006037,  0.30579),
    ("3a", 1158.265477, 50.880581, 0.17097),
    ("3b", 667.777870,  41.549529, 0.13114),
]

DISCOUNT_FACTOR = [
    ("1a",  -0.045255,   0.004904,  0.09650),   # gamma
]

BUDGET_MULTIPLIER = [
    ("3a", 2006.601858, 50.880581, 0.53988),
    ("3b", 1316.182071, 41.549529, 0.52907),
]

BIDDER_OBJECTIVE = [
    ("3a", -1428.01,    50.880581, 0.3837),
]

VALUE_DISPERSION = [
    ("3a", 356.78,      50.880581, 0.0138),
    ("3b", 360.61,      41.549529, 0.0652),
]

RESERVE_PRICE = [
    ("2a", 0.005539,    0.003858,  0.06711),
    ("2b", -0.047984,   0.006037,  0.18638),
]

STATE_INFORMATION = [
    ("1b",  -0.0495,     0.008449,  0.1561),
]

ASYNCHRONOUS = [
    ("1a",  0.0366,      0.004904,  0.0566),
]

# -----------------------------------------------------------------
# Row labels and marker shapes
# -----------------------------------------------------------------
ROW_LABELS = {
    "1a":  "1a (Q-learn)",
    "1b":  "1b (Q-learn)",
    "2a": "2a (LinUCB)",
    "2b": "2b (Thompson)",
    "3a": "3a (Pacing)",
    "3b": "3b (PI Pacing)",
}

# Circle for learning experiments, diamond for pacing experiments
MARKERS = {
    "1a": "o", "1b": "o", "2a": "o", "2b": "o",
    "3a": "D", "3b": "D",
}

# Colormap for S_T encoding
CMAP = plt.cm.YlOrRd
NORM = mcolors.Normalize(vmin=0, vmax=0.6)

SUBTITLES = {
    "Auction format": "FPA vs SPA",
    "No. of bidders": "4 vs 2",
    "Budget multiplier": "Ample vs Tight",
    "Bidder objective": "Utility vs Value",
    "Value spread (\u03c3)": "High vs Low",
    "Reserve price": "With vs Without",
    "Discount factor (\u03b3)": "0.95 vs 0",
    "State information": "Full vs Minimal",
    "Async. updates": "On vs Off",
}

# Spacing constants
INTER_GROUP_GAP = 0.8
HEADER_TO_ENTRY_GAP = 0.9
INTER_ENTRY_GAP = 1.1


def normalize(exp_key, coef, se):
    """Convert raw coefficient and SE to % of grand mean, with 95% CI."""
    gm = GRAND_MEANS[exp_key]
    effect = (coef / gm) * 100
    ci_low = ((coef - 1.96 * se) / gm) * 100
    ci_high = ((coef + 1.96 * se) / gm) * 100
    return effect, ci_low, ci_high


def sort_groups(groups):
    """Sort groups by max |normalised effect| descending.
    Within each group, sort entries by |effect| descending."""
    scored = []
    for group_label, entries in groups:
        normed = []
        for exp_key, coef, se, st in entries:
            eff, _, _ = normalize(exp_key, coef, se)
            normed.append((exp_key, coef, se, st, abs(eff)))
        max_abs = max(n[4] for n in normed)
        # Sort entries within group by |effect| descending
        normed.sort(key=lambda x: x[4], reverse=True)
        sorted_entries = [(n[0], n[1], n[2], n[3]) for n in normed]
        scored.append((group_label, sorted_entries, max_abs))
    scored.sort(key=lambda x: x[2], reverse=True)
    return [(label, entries) for label, entries, _ in scored]


def build_rows(groups):
    """Build the row structure for a set of groups.
    Returns list of (label, effect, ci_low, ci_high, exp_key, st, is_header)."""
    rows = []
    for group_label, entries in groups:
        rows.append((group_label, None, None, None, None, None, True))
        for exp_key, coef, se, st in entries:
            effect, ci_low, ci_high = normalize(exp_key, coef, se)
            rows.append((
                ROW_LABELS[exp_key], effect, ci_low, ci_high,
                exp_key, st, False,
            ))
    # Reverse so first group appears at top
    return rows[::-1]


def compute_y_positions(rows):
    """Compute y positions for each row with increased spacing."""
    y_positions = []
    y = 0
    prev_was_header = False
    for row in rows:
        is_header = row[6]
        if is_header:
            if y > 0:
                y += INTER_GROUP_GAP
            y_positions.append(y)
            prev_was_header = True
        else:
            if prev_was_header:
                y += HEADER_TO_ENTRY_GAP
                prev_was_header = False
            y_positions.append(y)
        y += INTER_ENTRY_GAP
    return y_positions


def draw_forest(ax, rows, y_positions):
    """Draw data points and whiskers on the given axes."""
    for i, row in enumerate(rows):
        label, effect, ci_low, ci_high, exp_key, st, is_header = row
        if is_header:
            continue
        yp = y_positions[i]
        color = CMAP(NORM(st))
        marker = MARKERS[exp_key]
        ax.plot(
            [ci_low, ci_high], [yp, yp],
            color=color, linewidth=1.5, solid_capstyle="round",
        )
        ax.plot(
            effect, yp, marker,
            color=color, markersize=6,
            markeredgecolor="#333333", markeredgewidth=0.6,
            zorder=5,
        )


def style_axes(ax, rows, y_positions):
    """Apply labels, gridlines, spines, and subtitles."""
    # Zero line
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--", zorder=0)

    # Vertical gridlines
    ax.xaxis.grid(True, alpha=0.3, linestyle=":", color="grey", zorder=0)
    ax.yaxis.grid(False)

    # Y-axis labels
    yticks = []
    ytick_labels = []
    for i, row in enumerate(rows):
        label = row[0]
        is_header = row[6]
        yticks.append(y_positions[i])
        if is_header:
            ytick_labels.append(label)
        else:
            ytick_labels.append(f"  {label}")

    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=10)

    # Bold group headers
    for i, row in enumerate(rows):
        if row[6]:
            ax.get_yticklabels()[i].set_fontweight("bold")
            ax.get_yticklabels()[i].set_fontsize(11)

    # Subtitles below each group header
    trans = ax.get_yaxis_transform()
    for i, row in enumerate(rows):
        if row[6]:
            subtitle = SUBTITLES.get(row[0], "")
            if subtitle:
                ax.text(
                    -0.01, y_positions[i] - 0.55,
                    subtitle,
                    transform=trans,
                    fontsize=8, fontstyle="italic", color="#666666",
                    ha="right", va="center",
                    clip_on=False,
                )

    # X-axis
    ax.set_xlabel("Effect on revenue (% of grand mean)", fontsize=12)

    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_colorbar(fig, ax):
    """Add the S_T colorbar."""
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("$S_T$ (variance explained)", fontsize=11)
    return cbar


def add_legend(ax):
    """Add marker-shape legend."""
    legend_elements = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="gray",
               markeredgecolor="#333333", markersize=6, label="Learning (1a\u20132b)"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="gray",
               markeredgecolor="#333333", markersize=6, label="Pacing (3a\u20133b)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
              framealpha=0.9, edgecolor="lightgray")


def figure_height(y_positions, max_height=9.5):
    """Compute figure height from y range, capped at max_height inches."""
    y_range = max(y_positions) - min(y_positions)
    return min(max_height, max(4, y_range * 0.35 + 1.5))


def save_figure(fig, stem):
    """Save as PNG and PDF."""
    png_path = os.path.join(FIGURE_DIR, f"{stem}.png")
    pdf_path = os.path.join(FIGURE_DIR, f"{stem}.pdf")
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def render_forest(groups, stem, figwidth=7):
    """Render a single forest plot for the given groups."""
    rows = build_rows(groups)
    y_positions = compute_y_positions(rows)
    height = figure_height(y_positions)

    fig, ax = plt.subplots(figsize=(figwidth, height))
    draw_forest(ax, rows, y_positions)
    style_axes(ax, rows, y_positions)
    add_colorbar(fig, ax)
    add_legend(ax)
    fig.tight_layout()
    save_figure(fig, stem)


def split_groups(groups):
    """Split groups into two halves balanced by row count.
    Each group contributes 1 (header) + len(entries) rows."""
    row_counts = [1 + len(entries) for _, entries in groups]
    total = sum(row_counts)
    target = total / 2

    cumsum = 0
    split_idx = 0
    for i, c in enumerate(row_counts):
        cumsum += c
        if cumsum >= target:
            # Pick whichever split point is closer to half
            if i > 0 and abs(cumsum - c - target) < abs(cumsum - target):
                split_idx = i
            else:
                split_idx = i + 1
            break

    if split_idx == 0:
        split_idx = 1
    if split_idx >= len(groups):
        split_idx = len(groups) - 1

    return groups[:split_idx], groups[split_idx:]


def make_forest_plot():
    """Build and save the master forest plot (combined + split)."""

    groups = [
        ("Auction format", AUCTION_FORMAT),
        ("No. of bidders", MARKET_THICKNESS),
        ("Budget multiplier", BUDGET_MULTIPLIER),
        ("Bidder objective", BIDDER_OBJECTIVE),
        ("Value spread (\u03c3)", VALUE_DISPERSION),
        ("Reserve price", RESERVE_PRICE),
        ("Discount factor (\u03b3)", DISCOUNT_FACTOR),
        ("State information", STATE_INFORMATION),
        ("Async. updates", ASYNCHRONOUS),
    ]

    # Sort by importance
    groups = sort_groups(groups)

    # Combined plot
    render_forest(groups, "master_forest")

    # Split into two halves
    half1, half2 = split_groups(groups)
    render_forest(half1, "master_forest_1")
    render_forest(half2, "master_forest_2")


# Experiment order for the intro plot (by experiment number)
INTRO_EXP_ORDER = ["1a", "1b", "2a", "2b", "3a", "3b"]


def order_by_experiment(entries):
    """Return entries sorted by experiment number (1a, 1b, 2a, 2b, 3a, 3b)."""
    order = {k: i for i, k in enumerate(INTRO_EXP_ORDER)}
    return sorted(entries, key=lambda e: order[e[0]])


def make_intro_plot():
    """Build a focused forest plot with only No. of bidders and Auction format.

    Used in the introduction to highlight the two cross-cutting factors
    measured across all six experiments. Entries ordered by experiment number.
    Outputs: paper/figures/intro_forest.{png,pdf}
    """
    groups = [
        ("No. of bidders", order_by_experiment(MARKET_THICKNESS)),
        ("Auction format", order_by_experiment(AUCTION_FORMAT)),
    ]
    render_forest(groups, "intro_forest")


if __name__ == "__main__":
    make_forest_plot()
    make_intro_plot()
