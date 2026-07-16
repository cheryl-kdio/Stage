
"""
ekf_poisson_parameter_estimation.py

Extended Kalman Filter (EKF) for a latent Ornstein–Uhlenbeck intensity model

    dX_t = kappa (mu - X_t) dt + sigma dW_t

    Y_k | X_k ~ Poisson(A_k exp(X_k))

where A_k is the integrated deterministic baseline over each observation bin.

This module also provides maximum-likelihood estimation of
(kappa, mu, sigma) using the approximate Gaussian innovation
log-likelihood produced by the EKF.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def integrate_baseline(lambda0, left, right, n_grid=30):
    grid = np.linspace(left, right, n_grid)
    values = np.asarray(lambda0(grid), dtype=float)
    return float(np.trapz(values, grid))


def compute_baseline_integrals(lambda0, bin_edges, n_grid=30):
    edges = np.asarray(bin_edges, dtype=float)
    A = np.empty(len(edges) - 1)
    for k in range(len(A)):
        A[k] = integrate_baseline(lambda0, edges[k], edges[k + 1], n_grid)
    return A


def extended_kalman_filter_counts(
    counts,
    bin_edges,
    baseline_integrals,
    *,
    kappa,
    long_run_mean,
    sigma,
    initial_mean,
    initial_variance,
):
    counts = np.asarray(counts, dtype=float)
    edges = np.asarray(bin_edges, dtype=float)
    A = np.asarray(baseline_integrals, dtype=float)

    n = len(counts)

    m_pred = np.zeros(n)
    P_pred = np.zeros(n)
    m_filt = np.zeros(n)
    P_filt = np.zeros(n)

    loglik = 0.0

    m_prev = initial_mean
    P_prev = initial_variance

    for k in range(n):
        dt = edges[k + 1] - edges[k]

        phi = np.exp(-kappa * dt)
        q = sigma ** 2 / (2.0 * kappa) * (1.0 - np.exp(-2.0 * kappa * dt))

        m_minus = long_run_mean + phi * (m_prev - long_run_mean)
        P_minus = phi ** 2 * P_prev + q

        yhat = A[k] * np.exp(m_minus)
        H = yhat
        R = max(yhat, 1e-8)

        S = H * H * P_minus + R
        innovation = counts[k] - yhat

        K = P_minus * H / S

        m = m_minus + K * innovation
        P = max((1.0 - K * H) * P_minus, 1e-12)

        loglik += -0.5 * (
            np.log(2.0 * np.pi)
            + np.log(S)
            + innovation ** 2 / S
        )

        m_pred[k] = m_minus
        P_pred[k] = P_minus
        m_filt[k] = m
        P_filt[k] = P

        m_prev = m
        P_prev = P

    return {
        "predicted_mean": m_pred,
        "predicted_variance": P_pred,
        "filtered_mean": m_filt,
        "filtered_variance": P_filt,
        "approximate_log_likelihood": float(loglik),
    }


def _decode(theta):
    log_half_life, log_stationary_var, mu = theta
    half_life = np.exp(log_half_life)
    stationary_var = np.exp(log_stationary_var)

    kappa = np.log(2.0) / half_life
    sigma = np.sqrt(2.0 * kappa * stationary_var)

    return kappa, mu, sigma, stationary_var


def negative_loglik(
    theta,
    counts,
    bin_edges,
    baseline_integrals,
):
    kappa, mu, sigma, stationary_var = _decode(theta)

    result = extended_kalman_filter_counts(
        counts=counts,
        bin_edges=bin_edges,
        baseline_integrals=baseline_integrals,
        kappa=kappa,
        long_run_mean=mu,
        sigma=sigma,
        initial_mean=mu,
        initial_variance=stationary_var,
    )

    return -result["approximate_log_likelihood"]


def estimate_parameters(
    counts,
    bin_edges,
    baseline_integrals,
):
    starts = [
        np.array([np.log(300.0), np.log(0.05), 0.0]),
        np.array([np.log(1800.0), np.log(0.20), 0.0]),
        np.array([np.log(7200.0), np.log(0.50), 0.0]),
    ]

    best = None

    for start in starts:
        res = minimize(
            negative_loglik,
            start,
            args=(counts, bin_edges, baseline_integrals),
            method="L-BFGS-B",
            bounds=[
                (np.log(30.0), np.log(6 * 3600.0)),
                (np.log(1e-4), np.log(5.0)),
                (-3.0, 3.0),
            ],
        )

        if best is None or res.fun < best.fun:
            best = res

    kappa, mu, sigma, stationary_var = _decode(best.x)

    return {
        "kappa": kappa,
        "mu": mu,
        "sigma": sigma,
        "stationary_variance": stationary_var,
        "optimization": best,
    }


if __name__ == "__main__":

    def lambda0(t):
        return np.ones_like(t)

    bin_edges = np.arange(0.0, 1001.0, 10.0)
    counts = np.random.poisson(2.0, len(bin_edges) - 1)

    A = compute_baseline_integrals(lambda0, bin_edges)

    result = estimate_parameters(counts, bin_edges, A)

    print(result)
