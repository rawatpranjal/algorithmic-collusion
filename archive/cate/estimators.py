"""CATE Estimators: Oracle OLS and IF-based Doubly Robust."""

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold


def oracle_ols(X: np.ndarray, T: np.ndarray, Y: np.ndarray,
               x_values: np.ndarray) -> np.ndarray:
    """
    Oracle OLS estimator with correctly specified CATE model.

    Regression: Y ~ 1 + X + T + T:X + T:X² + T:X³ + T:sin(2πX)

    Parameters
    ----------
    X : ndarray of shape (n,)
        Covariates
    T : ndarray of shape (n,)
        Treatment indicator
    Y : ndarray of shape (n,)
        Outcome
    x_values : ndarray of shape (k,)
        X values at which to estimate CATE

    Returns
    -------
    tau_hat : ndarray of shape (k,)
        CATE estimates at each x value
    """
    n = len(X)

    # Build design matrix: [1, X, T, T*X, T*X², T*X³, T*sin(2πX)]
    design = np.column_stack([
        np.ones(n),
        X,
        T,
        T * X,
        T * X**2,
        T * X**3,
        T * np.sin(2 * np.pi * X)
    ])

    # OLS: β = (X'X)^(-1) X'Y
    beta = np.linalg.lstsq(design, Y, rcond=None)[0]

    # τ(x) = β_T + β_{T:X}*x + β_{T:X²}*x² + β_{T:X³}*x³ + β_{T:sin}*sin(2πx)
    # Coefficients: β[2]=T, β[3]=T:X, β[4]=T:X², β[5]=T:X³, β[6]=T:sin
    tau_hat = (beta[2] +
               beta[3] * x_values +
               beta[4] * x_values**2 +
               beta[5] * x_values**3 +
               beta[6] * np.sin(2 * np.pi * x_values))

    return tau_hat


def _make_mu_features(x: np.ndarray, degree: int = 3) -> np.ndarray:
    """
    Build feature matrix for outcome models.

    Parameters
    ----------
    x : ndarray
        Covariate values
    degree : int
        Polynomial degree. degree=3 includes sin term (correctly specified).
        degree=2 omits sin term (misspecified).

    Returns
    -------
    features : ndarray
        Feature matrix
    """
    x = x.reshape(-1, 1) if x.ndim == 1 else x

    if degree >= 3:
        # Correctly specified: includes sin term
        return np.column_stack([
            x,
            x**2,
            x**3,
            np.sin(2 * np.pi * x)
        ])
    elif degree == 2:
        # Misspecified: degree-2 polynomial only
        return np.column_stack([x, x**2])
    else:
        # degree=1
        return x


def _make_pi_features(x: np.ndarray, degree: int = 1) -> np.ndarray:
    """
    Build feature matrix for propensity model.

    Parameters
    ----------
    x : ndarray
        Covariate values
    degree : int
        Polynomial degree. degree=1 is correctly specified (linear).
        degree=0 returns intercept-only (misspecified - constant propensity).

    Returns
    -------
    features : ndarray
        Feature matrix
    """
    x = x.reshape(-1, 1) if x.ndim == 1 else x

    if degree >= 1:
        # Correctly specified: linear in X
        return x
    else:
        # degree=0: constant propensity (misspecified)
        return np.ones((len(x), 1))


def fit_nuisances(X: np.ndarray, T: np.ndarray, Y: np.ndarray,
                  n_folds: int = 5,
                  mu_degree: int = 3,
                  pi_degree: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cross-fit nuisance functions: μ̂₀, μ̂₁, π̂.

    Parameters
    ----------
    X : ndarray of shape (n,)
        Covariates
    T : ndarray of shape (n,)
        Treatment indicator
    Y : ndarray of shape (n,)
        Outcome
    n_folds : int
        Number of folds for cross-fitting
    mu_degree : int
        Polynomial degree for outcome models. 3=correct (includes sin), 2=misspecified.
    pi_degree : int
        Polynomial degree for propensity model. 1=correct (linear), 0=misspecified (constant).

    Returns
    -------
    mu0_hat : ndarray of shape (n,)
        Out-of-fold predictions of E[Y|X, T=0]
    mu1_hat : ndarray of shape (n,)
        Out-of-fold predictions of E[Y|X, T=1]
    pi_hat : ndarray of shape (n,)
        Out-of-fold predictions of P(T=1|X)
    """
    n = len(X)
    mu0_hat = np.zeros(n)
    mu1_hat = np.zeros(n)
    pi_hat = np.zeros(n)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    X_mu_features = _make_mu_features(X, mu_degree)
    X_pi_features = _make_pi_features(X, pi_degree)

    for train_idx, test_idx in kf.split(X):
        X_mu_train, X_mu_test = X_mu_features[train_idx], X_mu_features[test_idx]
        X_pi_train, X_pi_test = X_pi_features[train_idx], X_pi_features[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        # Fit μ₀: E[Y|X, T=0] using control observations
        control_mask = T_train == 0
        if control_mask.sum() > 0:
            model_mu0 = Ridge(alpha=0.01)
            model_mu0.fit(X_mu_train[control_mask], Y_train[control_mask])
            mu0_hat[test_idx] = model_mu0.predict(X_mu_test)

        # Fit μ₁: E[Y|X, T=1] using treated observations
        treated_mask = T_train == 1
        if treated_mask.sum() > 0:
            model_mu1 = Ridge(alpha=0.01)
            model_mu1.fit(X_mu_train[treated_mask], Y_train[treated_mask])
            mu1_hat[test_idx] = model_mu1.predict(X_mu_test)

        # Fit π: P(T=1|X) using all observations
        model_pi = LogisticRegression(max_iter=1000, solver='lbfgs')
        model_pi.fit(X_pi_train, T_train)
        pi_hat[test_idx] = model_pi.predict_proba(X_pi_test)[:, 1]

    # Clip propensity scores to avoid extreme weights
    pi_hat = np.clip(pi_hat, 0.05, 0.95)

    return mu0_hat, mu1_hat, pi_hat


def compute_pseudo_outcomes(X: np.ndarray, T: np.ndarray, Y: np.ndarray,
                            mu0_hat: np.ndarray, mu1_hat: np.ndarray,
                            pi_hat: np.ndarray) -> np.ndarray:
    """
    Compute doubly robust pseudo-outcomes.

    Ŝᵢ = [μ̂₁(Xᵢ) - μ̂₀(Xᵢ)] + (Tᵢ - π̂(Xᵢ))·(Yᵢ - μ̂_Tᵢ(Xᵢ)) / [π̂(Xᵢ)(1-π̂(Xᵢ))]

    Parameters
    ----------
    X : ndarray of shape (n,)
        Covariates
    T : ndarray of shape (n,)
        Treatment indicator
    Y : ndarray of shape (n,)
        Outcome
    mu0_hat : ndarray of shape (n,)
        Estimated E[Y|X, T=0]
    mu1_hat : ndarray of shape (n,)
        Estimated E[Y|X, T=1]
    pi_hat : ndarray of shape (n,)
        Estimated P(T=1|X)

    Returns
    -------
    S_hat : ndarray of shape (n,)
        Pseudo-outcomes
    """
    # μ̂_T(X) - the relevant outcome model prediction
    mu_T_hat = np.where(T == 1, mu1_hat, mu0_hat)

    # Doubly robust pseudo-outcome
    regression_part = mu1_hat - mu0_hat
    ipw_correction = (T - pi_hat) * (Y - mu_T_hat) / (pi_hat * (1 - pi_hat))

    S_hat = regression_part + ipw_correction

    return S_hat


def if_cate(X: np.ndarray, T: np.ndarray, Y: np.ndarray,
            x_values: np.ndarray, n_folds: int = 5,
            mu_degree: int = 3, pi_degree: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    IF-based doubly robust CATE estimator.

    Parameters
    ----------
    X : ndarray of shape (n,)
        Covariates
    T : ndarray of shape (n,)
        Treatment indicator
    Y : ndarray of shape (n,)
        Outcome
    x_values : ndarray of shape (k,)
        X values at which to estimate CATE
    n_folds : int
        Number of folds for cross-fitting
    mu_degree : int
        Polynomial degree for outcome models. 3=correct, 2=misspecified.
    pi_degree : int
        Polynomial degree for propensity model. 1=correct, 0=misspecified.

    Returns
    -------
    tau_hat : ndarray of shape (k,)
        CATE estimates at each x value
    se_hat : ndarray of shape (k,)
        Analytical standard errors at each x value
    """
    # Step 1: Cross-fit nuisance functions
    mu0_hat, mu1_hat, pi_hat = fit_nuisances(X, T, Y, n_folds, mu_degree, pi_degree)

    # Step 2: Compute pseudo-outcomes
    S_hat = compute_pseudo_outcomes(X, T, Y, mu0_hat, mu1_hat, pi_hat)

    # Step 3: CATE at each x_j = mean(Ŝᵢ | Xᵢ = xⱼ)
    # Step 4: Analytical SE = SD(Ŝᵢ | Xᵢ = xⱼ) / √nⱼ
    k = len(x_values)
    tau_hat = np.zeros(k)
    se_hat = np.zeros(k)

    for j, x_j in enumerate(x_values):
        mask = np.isclose(X, x_j, atol=1e-9)
        n_j = mask.sum()

        if n_j > 0:
            S_j = S_hat[mask]
            tau_hat[j] = np.mean(S_j)
            if n_j > 1:
                se_hat[j] = np.std(S_j, ddof=1) / np.sqrt(n_j)
            else:
                se_hat[j] = np.nan
        else:
            tau_hat[j] = np.nan
            se_hat[j] = np.nan

    return tau_hat, se_hat


def plugin_cate(X: np.ndarray, T: np.ndarray, Y: np.ndarray,
                x_values: np.ndarray, n_folds: int = 5,
                mu_degree: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Plugin CATE estimator: τ̂(x) = μ̂₁(x) - μ̂₀(x).

    No IPW correction - only uses outcome model predictions.
    SE computed via bootstrap-like variance of predictions across folds.

    Parameters
    ----------
    X : ndarray of shape (n,)
        Covariates
    T : ndarray of shape (n,)
        Treatment indicator
    Y : ndarray of shape (n,)
        Outcome
    x_values : ndarray of shape (k,)
        X values at which to estimate CATE
    n_folds : int
        Number of folds for cross-fitting
    mu_degree : int
        Polynomial degree for outcome models. 3=correct, 2=misspecified.

    Returns
    -------
    tau_hat : ndarray of shape (k,)
        CATE estimates at each x value
    se_hat : ndarray of shape (k,)
        Standard errors at each x value (from sample variance of τ̂ at each x)
    """
    # Cross-fit outcome models only (propensity not needed for plugin)
    mu0_hat, mu1_hat, _ = fit_nuisances(X, T, Y, n_folds, mu_degree, pi_degree=1)

    # Plugin pseudo-outcome: just the difference in predicted outcomes
    plugin_pseudo = mu1_hat - mu0_hat

    # Aggregate at each x value
    k = len(x_values)
    tau_hat = np.zeros(k)
    se_hat = np.zeros(k)

    for j, x_j in enumerate(x_values):
        mask = np.isclose(X, x_j, atol=1e-9)
        n_j = mask.sum()

        if n_j > 0:
            pseudo_j = plugin_pseudo[mask]
            tau_hat[j] = np.mean(pseudo_j)
            if n_j > 1:
                se_hat[j] = np.std(pseudo_j, ddof=1) / np.sqrt(n_j)
            else:
                se_hat[j] = np.nan
        else:
            tau_hat[j] = np.nan
            se_hat[j] = np.nan

    return tau_hat, se_hat


def get_nuisance_predictions(X: np.ndarray, T: np.ndarray, Y: np.ndarray,
                             n_folds: int = 5) -> dict:
    """
    Get nuisance function predictions for diagnostics.

    Returns cross-fitted predictions along with true values for computing
    fit metrics.
    """
    from . import dgp

    mu0_hat, mu1_hat, pi_hat = fit_nuisances(X, T, Y, n_folds)

    return {
        'mu0_hat': mu0_hat,
        'mu1_hat': mu1_hat,
        'pi_hat': pi_hat,
        'mu0_true': dgp.true_mu0(X),
        'mu1_true': dgp.true_mu1(X),
        'pi_true': dgp.true_propensity(X),
        'T': T,
        'Y': Y
    }
