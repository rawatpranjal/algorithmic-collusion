"""
Pure-math helpers re-derived from paper LaTeX (equilibria.tex, auctions.tex, appendix_robustness.tex).
Zero imports from project source code.

All formulas use exact rational arithmetic (fractions.Fraction) where possible,
falling back to float only for scipy optimization / MC.
"""

from fractions import Fraction
import numpy as np
from scipy.optimize import linprog


# ---------------------------------------------------------------------------
# Valuation model (equilibria.tex L16-23)
# ---------------------------------------------------------------------------

def compute_alpha_beta(eta, n):
    """
    alpha = 1 - eta/2
    beta  = eta / (2*(n-1))

    v_i = alpha * s_i + beta * sum_{j != i} s_j
    """
    alpha = 1.0 - 0.5 * eta
    beta = 0.5 * eta / max(n - 1, 1)
    return alpha, beta


def compute_alpha_beta_exact(eta, n):
    """Same as above but returns Fraction for exact arithmetic."""
    eta = Fraction(eta)
    n = Fraction(n)
    alpha = 1 - eta / 2
    beta = eta / (2 * (n - 1)) if n > 1 else Fraction(0)
    return alpha, beta


# ---------------------------------------------------------------------------
# BNE bid functions (equilibria.tex L31-39)
# ---------------------------------------------------------------------------

def bne_bid_slope(eta, n, auction_type):
    """
    SPA: phi = alpha + n*beta/2
    FPA: phi = (n-1)/n * (alpha + n*beta/2)

    b(s) = phi * s
    """
    alpha, beta = compute_alpha_beta(eta, n)
    base = alpha + n * beta / 2.0
    if auction_type == "first":
        return (n - 1) / n * base
    return base


def bne_bid_slope_exact(eta, n, auction_type):
    """Same but exact Fraction."""
    alpha, beta = compute_alpha_beta_exact(eta, n)
    n = Fraction(n)
    base = alpha + n * beta / 2
    if auction_type == "first":
        return (n - 1) / n * base
    return base


# ---------------------------------------------------------------------------
# Analytical revenue (equilibria.tex L42-48)
# ---------------------------------------------------------------------------

def analytical_revenue(eta, n):
    """
    R_BNE = (n-1)/(n+1) * phi  where phi = alpha + n*beta/2
    Revenue equivalence: FPA = SPA under iid signals.
    """
    alpha, beta = compute_alpha_beta(eta, n)
    phi = alpha + n * beta / 2.0
    return (n - 1) / (n + 1) * phi


def analytical_revenue_exact(eta, n):
    """Exact rational arithmetic version."""
    alpha, beta = compute_alpha_beta_exact(eta, n)
    n = Fraction(n)
    phi = alpha + n * beta / 2
    return (n - 1) / (n + 1) * phi


# ---------------------------------------------------------------------------
# Efficient benchmark (equilibria.tex L51-55)
# ---------------------------------------------------------------------------

def efficient_benchmark(eta, n):
    """
    E[v_(1)] = (alpha - beta) * n/(n+1) + beta * n/2

    The expected highest valuation. Derivation:
    v_i = (alpha-beta)*s_i + beta*S where S = sum of all signals
    max_i v_i = (alpha-beta)*s_(n:n) + beta*S
    E[s_(n:n)] = n/(n+1), E[S] = n/2
    """
    alpha, beta = compute_alpha_beta(eta, n)
    return (alpha - beta) * n / (n + 1) + beta * n / 2.0


def efficient_benchmark_exact(eta, n):
    """Exact version."""
    alpha, beta = compute_alpha_beta_exact(eta, n)
    n_frac = Fraction(n)
    return (alpha - beta) * n_frac / (n_frac + 1) + beta * n_frac / 2


# ---------------------------------------------------------------------------
# Regret under BNE (equilibria.tex L56-60)
# ---------------------------------------------------------------------------

def bne_regret(eta, n):
    """
    Regret* = 1 - R_BNE / E[v_(1)]
    """
    R = analytical_revenue(eta, n)
    E_v1 = efficient_benchmark(eta, n)
    if E_v1 == 0:
        return 0.0
    return 1.0 - R / E_v1


def bne_regret_exact(eta, n):
    """Exact version."""
    R = analytical_revenue_exact(eta, n)
    E_v1 = efficient_benchmark_exact(eta, n)
    if E_v1 == 0:
        return Fraction(0)
    return 1 - R / E_v1


# ---------------------------------------------------------------------------
# LP for optimal liquid welfare (auctions.tex)
# ---------------------------------------------------------------------------

def solve_welfare_lp(valuations, budgets):
    """
    Solve the LP relaxation for optimal liquid welfare.

    LP* = max sum_{t,i} v_{ti} * x_{ti}
    s.t.  sum_i x_{ti} <= 1  for all t   (one item per round)
          sum_t v_{ti} * x_{ti} <= B_i  for all i  (budget)
          0 <= x_{ti} <= 1

    Args:
        valuations: array (T, N) of valuations v_{ti}
        budgets: array (N,) of budgets B_i

    Returns:
        (lp_star, x_opt): optimal value and allocation matrix
    """
    valuations = np.asarray(valuations, dtype=float)
    budgets = np.asarray(budgets, dtype=float)
    T, N = valuations.shape

    # Decision variables: x_{ti} flattened as x[t*N + i]
    n_vars = T * N

    # Objective: maximize sum v_{ti} * x_{ti} => minimize -sum v_{ti} * x_{ti}
    c = -valuations.ravel()

    # Inequality constraints: A_ub @ x <= b_ub
    # 1. Supply: sum_i x_{ti} <= 1 for each t  (T constraints)
    # 2. Budget: sum_t v_{ti} * x_{ti} <= B_i for each i  (N constraints)
    A_rows = []
    b_rows = []

    # Supply constraints
    for t in range(T):
        row = np.zeros(n_vars)
        for i in range(N):
            row[t * N + i] = 1.0
        A_rows.append(row)
        b_rows.append(1.0)

    # Budget constraints
    for i in range(N):
        row = np.zeros(n_vars)
        for t in range(T):
            row[t * N + i] = valuations[t, i]
        A_rows.append(row)
        b_rows.append(budgets[i])

    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)

    # Bounds: 0 <= x_{ti} <= 1
    bounds = [(0.0, 1.0)] * n_vars

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if not result.success:
        raise RuntimeError(f"LP solve failed: {result.message}")

    lp_star = -result.fun
    x_opt = result.x.reshape(T, N)
    return lp_star, x_opt


def solve_welfare_ilp(valuations, budgets):
    """
    Solve the integer LP for optimal liquid welfare (exact combinatorial optimum).

    Same as LP but x_{ti} in {0,1} and sum_i x_{ti} <= 1 (at most one winner).
    Uses scipy.optimize.milp (available scipy >= 1.7).

    Only feasible for small instances (T*N < ~500).
    """
    from scipy.optimize import milp, LinearConstraint, Bounds

    valuations = np.asarray(valuations, dtype=float)
    budgets = np.asarray(budgets, dtype=float)
    T, N = valuations.shape
    n_vars = T * N

    # Objective: minimize -sum v_{ti} * x_{ti}
    c = -valuations.ravel()

    # Constraints
    A_rows = []
    b_lower = []
    b_upper = []

    # Supply: sum_i x_{ti} <= 1
    for t in range(T):
        row = np.zeros(n_vars)
        for i in range(N):
            row[t * N + i] = 1.0
        A_rows.append(row)
        b_lower.append(-np.inf)
        b_upper.append(1.0)

    # Budget: sum_t v_{ti} * x_{ti} <= B_i
    for i in range(N):
        row = np.zeros(n_vars)
        for t in range(T):
            row[t * N + i] = valuations[t, i]
        A_rows.append(row)
        b_lower.append(-np.inf)
        b_upper.append(budgets[i])

    A = np.array(A_rows)
    constraints = LinearConstraint(A, b_lower, b_upper)
    integrality = np.ones(n_vars)  # all integer
    bounds = Bounds(lb=0, ub=1)

    result = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)

    if not result.success:
        raise RuntimeError(f"ILP solve failed: {result.message}")

    ilp_star = -result.fun
    x_opt = result.x.reshape(T, N)
    return ilp_star, x_opt


# ---------------------------------------------------------------------------
# Liquid welfare computation (auctions.tex)
# ---------------------------------------------------------------------------

def compute_liquid_welfare(allocation, valuations, budgets):
    """
    W(x) = sum_i min(B_i, sum_t x_{ti} * v_{ti})

    Args:
        allocation: (T, N) array, x_{ti} in [0,1] or {0,1}
        valuations: (T, N) array
        budgets: (N,) array
    """
    allocation = np.asarray(allocation, dtype=float)
    valuations = np.asarray(valuations, dtype=float)
    budgets = np.asarray(budgets, dtype=float)

    # Total value won by each bidder
    value_won = (allocation * valuations).sum(axis=0)  # (N,)
    # Cap at budget
    welfare_i = np.minimum(budgets, value_won)
    return float(welfare_i.sum())


# ---------------------------------------------------------------------------
# Pacing bid formulas (algorithms.tex L46-49)
# ---------------------------------------------------------------------------

def optimal_bid_value_max(v, mu, remaining_budget):
    """
    Value-maximizer bid: min(v/mu, remaining_budget)
    """
    if mu <= 0:
        return remaining_budget
    return min(v / mu, remaining_budget)


def optimal_bid_utility_max(v, mu, remaining_budget):
    """
    Utility-maximizer bid: min(v/(1+mu), remaining_budget)
    """
    return min(v / (1.0 + mu), remaining_budget)
