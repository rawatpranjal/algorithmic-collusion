"""Monte Carlo simulation engine for CATE estimation."""

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import dgp, estimators, metrics


def run_single_sim(n: int, seed: int, x_values: np.ndarray,
                   n_folds: int = 5) -> dict:
    """
    Run a single simulation iteration.

    Parameters
    ----------
    n : int
        Sample size
    seed : int
        Random seed
    x_values : ndarray of shape (k,)
        X values at which to estimate CATE
    n_folds : int
        Number of folds for cross-fitting

    Returns
    -------
    results : dict
        Dictionary with:
        - 'oracle_tau': Oracle OLS CATE estimates
        - 'if_tau': IF-based CATE estimates
        - 'if_se': IF-based analytical SEs
        - 'nuisance_metrics': Dict with nuisance fit metrics
    """
    # Generate data
    X, T, Y = dgp.generate_data(n, seed)

    # Oracle OLS
    oracle_tau = estimators.oracle_ols(X, T, Y, x_values)

    # IF-based doubly robust
    if_tau, if_se = estimators.if_cate(X, T, Y, x_values, n_folds)

    # Nuisance fit metrics
    nuisance_preds = estimators.get_nuisance_predictions(X, T, Y, n_folds)

    mu0_metrics = metrics.nuisance_fit_metrics(
        nuisance_preds['mu0_hat'],
        nuisance_preds['mu0_true']
    )
    mu1_metrics = metrics.nuisance_fit_metrics(
        nuisance_preds['mu1_hat'],
        nuisance_preds['mu1_true']
    )
    pi_metrics = metrics.propensity_fit_metrics(
        nuisance_preds['pi_hat'],
        nuisance_preds['pi_true'],
        nuisance_preds['T']
    )

    return {
        'oracle_tau': oracle_tau,
        'if_tau': if_tau,
        'if_se': if_se,
        'nuisance_metrics': {
            'mu0_rmse': mu0_metrics['rmse'],
            'mu0_r2': mu0_metrics['r2'],
            'mu1_rmse': mu1_metrics['rmse'],
            'mu1_r2': mu1_metrics['r2'],
            'pi_rmse': pi_metrics['rmse'],
            'pi_r2': pi_metrics['r2'],
            'pi_auc': pi_metrics['auc']
        }
    }


def run_monte_carlo(n: int, R: int, x_values: np.ndarray,
                    n_folds: int = 5, base_seed: int = 42) -> dict:
    """
    Run Monte Carlo simulation.

    Parameters
    ----------
    n : int
        Sample size
    R : int
        Number of Monte Carlo replications
    x_values : ndarray of shape (k,)
        X values at which to estimate CATE
    n_folds : int
        Number of folds for cross-fitting
    base_seed : int
        Base random seed

    Returns
    -------
    results : dict
        Dictionary with:
        - 'oracle_estimates': (R, k) array of Oracle CATE estimates
        - 'if_estimates': (R, k) array of IF-based CATE estimates
        - 'if_se': (R, k) array of IF-based analytical SEs
        - 'nuisance_metrics': DataFrame with per-sim nuisance metrics
        - 'x_values': X values
        - 'true_tau': True CATE values
    """
    k = len(x_values)
    oracle_estimates = np.zeros((R, k))
    if_estimates = np.zeros((R, k))
    if_se = np.zeros((R, k))

    nuisance_records = []

    print(f"Running Monte Carlo simulation: n={n}, R={R}")

    for r in tqdm(range(R), desc=f"n={n}"):
        seed = base_seed + r
        sim_result = run_single_sim(n, seed, x_values, n_folds)

        oracle_estimates[r] = sim_result['oracle_tau']
        if_estimates[r] = sim_result['if_tau']
        if_se[r] = sim_result['if_se']

        nuisance_record = {'sim': r, 'n': n, 'seed': seed}
        nuisance_record.update(sim_result['nuisance_metrics'])
        nuisance_records.append(nuisance_record)

    true_tau = dgp.true_tau(x_values)

    return {
        'oracle_estimates': oracle_estimates,
        'if_estimates': if_estimates,
        'if_se': if_se,
        'nuisance_metrics': pd.DataFrame(nuisance_records),
        'x_values': x_values,
        'true_tau': true_tau
    }


def compute_summary_metrics(mc_results: dict) -> pd.DataFrame:
    """
    Compute summary metrics from Monte Carlo results.

    Parameters
    ----------
    mc_results : dict
        Results from run_monte_carlo

    Returns
    -------
    summary : DataFrame
        Summary statistics per X value
    """
    x_values = mc_results['x_values']
    true_tau = mc_results['true_tau']

    # Oracle metrics
    oracle_bias, oracle_var, oracle_rmse = metrics.compute_bias_variance(
        mc_results['oracle_estimates'], true_tau
    )
    oracle_mc_se = metrics.compute_monte_carlo_se(mc_results['oracle_estimates'])

    # IF-based metrics
    if_bias, if_var, if_rmse = metrics.compute_bias_variance(
        mc_results['if_estimates'], true_tau
    )
    if_mc_se = metrics.compute_monte_carlo_se(mc_results['if_estimates'])
    if_analytical_se = metrics.compute_mean_analytical_se(mc_results['if_se'])

    # Coverage
    if_coverage = metrics.compute_coverage(
        mc_results['if_estimates'],
        mc_results['if_se'],
        true_tau
    )

    # SE ratio: IF analytical / Oracle MC (as specified in plan)
    se_ratio = if_analytical_se / oracle_mc_se

    # SE calibration: IF analytical / IF MC (validates IF SE estimation)
    se_calibration = if_analytical_se / if_mc_se

    # Build summary DataFrame
    summary = pd.DataFrame({
        'x': x_values,
        'true_tau': true_tau,
        'oracle_bias': oracle_bias,
        'oracle_variance': oracle_var,
        'oracle_rmse': oracle_rmse,
        'oracle_mc_se': oracle_mc_se,
        'if_bias': if_bias,
        'if_variance': if_var,
        'if_rmse': if_rmse,
        'if_mc_se': if_mc_se,
        'if_analytical_se': if_analytical_se,
        'if_coverage': if_coverage,
        'se_ratio': se_ratio,
        'se_calibration': se_calibration
    })

    return summary


# =============================================================================
# Comparison simulation: DR vs Plugin under misspecification
# =============================================================================

SCENARIOS = {
    'both_correct': {'mu_degree': 3, 'pi_degree': 1, 'description': 'Both μ and π correct'},
    'mu_wrong': {'mu_degree': 2, 'pi_degree': 1, 'description': 'μ misspecified, π correct'},
    'pi_wrong': {'mu_degree': 3, 'pi_degree': 0, 'description': 'μ correct, π misspecified'},
    'both_wrong': {'mu_degree': 2, 'pi_degree': 0, 'description': 'Both μ and π misspecified'},
}


def run_comparison_sim(n: int, seed: int, x_values: np.ndarray,
                       mu_degree: int = 3, pi_degree: int = 1,
                       n_folds: int = 5) -> dict:
    """
    Run a single comparison simulation: DR vs Plugin.

    Parameters
    ----------
    n : int
        Sample size
    seed : int
        Random seed
    x_values : ndarray of shape (k,)
        X values at which to estimate CATE
    mu_degree : int
        Polynomial degree for outcome models. 3=correct, 2=misspecified.
    pi_degree : int
        Polynomial degree for propensity model. 1=correct, 0=misspecified.
    n_folds : int
        Number of folds for cross-fitting

    Returns
    -------
    results : dict
        Dictionary with:
        - 'dr_tau': DR CATE estimates
        - 'dr_se': DR analytical SEs
        - 'plugin_tau': Plugin CATE estimates
        - 'plugin_se': Plugin SEs
    """
    # Generate data
    X, T, Y = dgp.generate_data(n, seed)

    # DR estimator (doubly robust)
    dr_tau, dr_se = estimators.if_cate(X, T, Y, x_values, n_folds, mu_degree, pi_degree)

    # Plugin estimator (no IPW correction)
    plugin_tau, plugin_se = estimators.plugin_cate(X, T, Y, x_values, n_folds, mu_degree)

    return {
        'dr_tau': dr_tau,
        'dr_se': dr_se,
        'plugin_tau': plugin_tau,
        'plugin_se': plugin_se
    }


def run_comparison_monte_carlo(n: int, R: int, x_values: np.ndarray,
                               scenario: str, n_folds: int = 5,
                               base_seed: int = 42) -> dict:
    """
    Run Monte Carlo simulation for DR vs Plugin comparison.

    Parameters
    ----------
    n : int
        Sample size
    R : int
        Number of Monte Carlo replications
    x_values : ndarray of shape (k,)
        X values at which to estimate CATE
    scenario : str
        One of: 'both_correct', 'mu_wrong', 'pi_wrong', 'both_wrong'
    n_folds : int
        Number of folds for cross-fitting
    base_seed : int
        Base random seed

    Returns
    -------
    results : dict
        Dictionary with:
        - 'dr_estimates': (R, k) array of DR CATE estimates
        - 'dr_se': (R, k) array of DR analytical SEs
        - 'plugin_estimates': (R, k) array of Plugin CATE estimates
        - 'plugin_se': (R, k) array of Plugin SEs
        - 'x_values': X values
        - 'true_tau': True CATE values
        - 'scenario': Scenario name
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}. Must be one of {list(SCENARIOS.keys())}")

    mu_degree = SCENARIOS[scenario]['mu_degree']
    pi_degree = SCENARIOS[scenario]['pi_degree']

    k = len(x_values)
    dr_estimates = np.zeros((R, k))
    dr_se = np.zeros((R, k))
    plugin_estimates = np.zeros((R, k))
    plugin_se = np.zeros((R, k))

    print(f"Running comparison MC: n={n}, R={R}, scenario={scenario}")
    print(f"  {SCENARIOS[scenario]['description']}")

    for r in tqdm(range(R), desc=f"{scenario}"):
        seed = base_seed + r
        sim_result = run_comparison_sim(n, seed, x_values, mu_degree, pi_degree, n_folds)

        dr_estimates[r] = sim_result['dr_tau']
        dr_se[r] = sim_result['dr_se']
        plugin_estimates[r] = sim_result['plugin_tau']
        plugin_se[r] = sim_result['plugin_se']

    true_tau = dgp.true_tau(x_values)

    return {
        'dr_estimates': dr_estimates,
        'dr_se': dr_se,
        'plugin_estimates': plugin_estimates,
        'plugin_se': plugin_se,
        'x_values': x_values,
        'true_tau': true_tau,
        'scenario': scenario
    }


def compute_comparison_metrics(mc_results: dict) -> pd.DataFrame:
    """
    Compute comparison metrics for DR vs Plugin.

    Parameters
    ----------
    mc_results : dict
        Results from run_comparison_monte_carlo

    Returns
    -------
    summary : DataFrame
        Summary statistics per X value
    """
    x_values = mc_results['x_values']
    true_tau = mc_results['true_tau']
    scenario = mc_results['scenario']

    # DR metrics
    dr_bias, dr_var, dr_rmse = metrics.compute_bias_variance(
        mc_results['dr_estimates'], true_tau
    )
    dr_mc_se = metrics.compute_monte_carlo_se(mc_results['dr_estimates'])
    dr_analytical_se = metrics.compute_mean_analytical_se(mc_results['dr_se'])
    dr_coverage = metrics.compute_coverage(
        mc_results['dr_estimates'],
        mc_results['dr_se'],
        true_tau
    )

    # Plugin metrics
    plugin_bias, plugin_var, plugin_rmse = metrics.compute_bias_variance(
        mc_results['plugin_estimates'], true_tau
    )
    plugin_mc_se = metrics.compute_monte_carlo_se(mc_results['plugin_estimates'])
    plugin_analytical_se = metrics.compute_mean_analytical_se(mc_results['plugin_se'])
    plugin_coverage = metrics.compute_coverage(
        mc_results['plugin_estimates'],
        mc_results['plugin_se'],
        true_tau
    )

    # Build summary DataFrame
    summary = pd.DataFrame({
        'x': x_values,
        'true_tau': true_tau,
        'scenario': scenario,
        'dr_bias': dr_bias,
        'dr_variance': dr_var,
        'dr_rmse': dr_rmse,
        'dr_mc_se': dr_mc_se,
        'dr_analytical_se': dr_analytical_se,
        'dr_coverage': dr_coverage,
        'plugin_bias': plugin_bias,
        'plugin_variance': plugin_var,
        'plugin_rmse': plugin_rmse,
        'plugin_mc_se': plugin_mc_se,
        'plugin_analytical_se': plugin_analytical_se,
        'plugin_coverage': plugin_coverage
    })

    return summary


def run_all_comparison_scenarios(n: int, R: int, x_values: np.ndarray,
                                  n_folds: int = 5, base_seed: int = 42) -> pd.DataFrame:
    """
    Run comparison Monte Carlo for all misspecification scenarios.

    Parameters
    ----------
    n : int
        Sample size
    R : int
        Number of Monte Carlo replications
    x_values : ndarray of shape (k,)
        X values at which to estimate CATE
    n_folds : int
        Number of folds for cross-fitting
    base_seed : int
        Base random seed

    Returns
    -------
    all_results : DataFrame
        Combined results for all scenarios
    """
    all_summaries = []

    for scenario in SCENARIOS:
        mc_results = run_comparison_monte_carlo(
            n, R, x_values, scenario, n_folds, base_seed
        )
        summary = compute_comparison_metrics(mc_results)
        all_summaries.append(summary)

    return pd.concat(all_summaries, ignore_index=True)
