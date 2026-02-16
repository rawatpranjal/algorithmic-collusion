"""Metrics for CATE simulation: coverage, bias, variance, RMSE."""

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_coverage(estimates_matrix: np.ndarray, se_matrix: np.ndarray,
                     true_tau: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Compute coverage of confidence intervals.

    Parameters
    ----------
    estimates_matrix : ndarray of shape (R, k)
        CATE estimates across R simulations for k X values
    se_matrix : ndarray of shape (R, k)
        Standard error estimates
    true_tau : ndarray of shape (k,)
        True CATE values
    alpha : float
        Significance level (default 0.05 for 95% CI)

    Returns
    -------
    coverage : ndarray of shape (k,)
        Coverage rate at each X value
    """
    from scipy import stats

    z = stats.norm.ppf(1 - alpha / 2)

    R, k = estimates_matrix.shape
    coverage = np.zeros(k)

    for j in range(k):
        tau_j = true_tau[j]
        estimates_j = estimates_matrix[:, j]
        se_j = se_matrix[:, j]

        # Count how often true value falls within CI
        lower = estimates_j - z * se_j
        upper = estimates_j + z * se_j

        # Exclude NaN values
        valid = ~(np.isnan(estimates_j) | np.isnan(se_j))
        if valid.sum() > 0:
            covers = (lower[valid] <= tau_j) & (tau_j <= upper[valid])
            coverage[j] = covers.mean()
        else:
            coverage[j] = np.nan

    return coverage


def compute_bias_variance(estimates_matrix: np.ndarray,
                          true_tau: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute bias, variance, and RMSE for CATE estimates.

    Parameters
    ----------
    estimates_matrix : ndarray of shape (R, k)
        CATE estimates across R simulations for k X values
    true_tau : ndarray of shape (k,)
        True CATE values

    Returns
    -------
    bias : ndarray of shape (k,)
        Bias at each X value: E[τ̂(xⱼ)] - τ(xⱼ)
    variance : ndarray of shape (k,)
        Variance at each X value: Var(τ̂(xⱼ))
    rmse : ndarray of shape (k,)
        RMSE at each X value: √(Bias² + Variance)
    """
    R, k = estimates_matrix.shape
    bias = np.zeros(k)
    variance = np.zeros(k)
    rmse = np.zeros(k)

    for j in range(k):
        estimates_j = estimates_matrix[:, j]
        valid = ~np.isnan(estimates_j)

        if valid.sum() > 0:
            mean_est = np.mean(estimates_j[valid])
            bias[j] = mean_est - true_tau[j]
            variance[j] = np.var(estimates_j[valid], ddof=1)
            rmse[j] = np.sqrt(bias[j]**2 + variance[j])
        else:
            bias[j] = np.nan
            variance[j] = np.nan
            rmse[j] = np.nan

    return bias, variance, rmse


def compute_monte_carlo_se(estimates_matrix: np.ndarray) -> np.ndarray:
    """
    Compute Monte Carlo standard error (empirical SD across simulations).

    Parameters
    ----------
    estimates_matrix : ndarray of shape (R, k)
        CATE estimates across R simulations for k X values

    Returns
    -------
    mc_se : ndarray of shape (k,)
        Monte Carlo SE at each X value
    """
    R, k = estimates_matrix.shape
    mc_se = np.zeros(k)

    for j in range(k):
        estimates_j = estimates_matrix[:, j]
        valid = ~np.isnan(estimates_j)

        if valid.sum() > 1:
            mc_se[j] = np.std(estimates_j[valid], ddof=1)
        else:
            mc_se[j] = np.nan

    return mc_se


def compute_mean_analytical_se(se_matrix: np.ndarray) -> np.ndarray:
    """
    Compute mean analytical SE across simulations.

    Parameters
    ----------
    se_matrix : ndarray of shape (R, k)
        Standard error estimates across R simulations

    Returns
    -------
    mean_se : ndarray of shape (k,)
        Mean analytical SE at each X value
    """
    k = se_matrix.shape[1]
    mean_se = np.zeros(k)

    for j in range(k):
        se_j = se_matrix[:, j]
        valid = ~np.isnan(se_j)

        if valid.sum() > 0:
            mean_se[j] = np.mean(se_j[valid])
        else:
            mean_se[j] = np.nan

    return mean_se


def nuisance_fit_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """
    Compute fit metrics for nuisance functions.

    Parameters
    ----------
    y_pred : ndarray
        Predicted values
    y_true : ndarray
        True values

    Returns
    -------
    metrics : dict
        Dictionary with 'rmse' and 'r2'
    """
    valid = ~(np.isnan(y_pred) | np.isnan(y_true))
    y_pred = y_pred[valid]
    y_true = y_true[valid]

    if len(y_pred) == 0:
        return {'rmse': np.nan, 'r2': np.nan}

    rmse = np.sqrt(np.mean((y_pred - y_true)**2))

    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {'rmse': rmse, 'r2': r2}


def propensity_fit_metrics(pi_pred: np.ndarray, pi_true: np.ndarray,
                           T: np.ndarray) -> dict:
    """
    Compute fit metrics for propensity score.

    Parameters
    ----------
    pi_pred : ndarray
        Predicted propensity scores
    pi_true : ndarray
        True propensity scores
    T : ndarray
        Actual treatment assignments

    Returns
    -------
    metrics : dict
        Dictionary with 'rmse', 'r2', and 'auc'
    """
    valid = ~(np.isnan(pi_pred) | np.isnan(pi_true))
    pi_pred = pi_pred[valid]
    pi_true = pi_true[valid]
    T = T[valid]

    if len(pi_pred) == 0:
        return {'rmse': np.nan, 'r2': np.nan, 'auc': np.nan}

    rmse = np.sqrt(np.mean((pi_pred - pi_true)**2))

    ss_res = np.sum((pi_true - pi_pred)**2)
    ss_tot = np.sum((pi_true - np.mean(pi_true))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # AUC for predicting actual treatment
    try:
        auc = roc_auc_score(T, pi_pred)
    except ValueError:
        auc = np.nan

    return {'rmse': rmse, 'r2': r2, 'auc': auc}
