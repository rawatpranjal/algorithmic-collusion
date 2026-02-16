"""Data Generating Process for CATE simulation."""

import numpy as np


def true_tau(x: np.ndarray) -> np.ndarray:
    """True CATE function: τ(X) = X³ - 2X² + sin(2πX)"""
    return x**3 - 2 * x**2 + np.sin(2 * np.pi * x)


def true_mu0(x: np.ndarray) -> np.ndarray:
    """True baseline outcome: μ₀(X) = 1 + 2X"""
    return 1 + 2 * x


def true_mu1(x: np.ndarray) -> np.ndarray:
    """True treated outcome: μ₁(X) = μ₀(X) + τ(X)"""
    return true_mu0(x) + true_tau(x)


def true_propensity(x: np.ndarray) -> np.ndarray:
    """True propensity score: π(X) = 0.3 + 0.4X"""
    return 0.3 + 0.4 * x


def generate_data(n: int, seed: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data from the DGP.

    Parameters
    ----------
    n : int
        Sample size
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    X : ndarray of shape (n,)
        Covariates from {0.0, 0.1, ..., 0.9}
    T : ndarray of shape (n,)
        Binary treatment
    Y : ndarray of shape (n,)
        Outcome
    """
    if seed is not None:
        np.random.seed(seed)

    # X ∈ {0.0, 0.1, 0.2, ..., 0.9} uniformly sampled
    x_values = np.arange(0, 1, 0.1)
    X = np.random.choice(x_values, size=n)

    # T ~ Bernoulli(π(X))
    pi_x = true_propensity(X)
    T = np.random.binomial(1, pi_x)

    # Y = μ₀(X) + τ(X)·T + ε, ε ~ N(0, 1)
    epsilon = np.random.normal(0, 1, n)
    Y = true_mu0(X) + true_tau(X) * T + epsilon

    return X, T, Y


def get_x_values() -> np.ndarray:
    """Return the discrete X values used in the DGP."""
    return np.arange(0, 1, 0.1)
