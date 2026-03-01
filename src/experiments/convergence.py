"""Shared prospective convergence detection for all experiments."""

import numpy as np


def compute_convergence_time(revenues, window_size=1000):
    """Prospective convergence: first window after which all subsequent
    windows have means within +/-5% of that window's mean.

    Unlike the retrospective method (which uses the final mean as reference,
    guaranteeing convergence by construction), this sets the reference at
    the candidate convergence point, making it a genuine forward-looking test.

    Args:
        revenues: array of per-episode revenues
        window_size: size of non-overlapping windows

    Returns:
        int: episode number at which convergence is detected, or len(revenues)
    """
    n = len(revenues)
    if n < 2 * window_size:
        return n  # not enough data

    # Compute non-overlapping window means
    n_windows = n // window_size
    window_means = np.array([
        np.mean(revenues[i * window_size:(i + 1) * window_size])
        for i in range(n_windows)
    ])

    # Find first window where all subsequent windows are within +/-5%
    for i in range(n_windows - 1):
        ref = window_means[i]
        if ref < 1e-10:  # near-zero revenue, skip
            continue
        lower, upper = 0.95 * ref, 1.05 * ref
        if np.all((window_means[i + 1:] >= lower) & (window_means[i + 1:] <= upper)):
            return (i + 1) * window_size
    return n  # never converged
