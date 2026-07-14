"""Filtres pour lambda(t)=lambda0(t)*exp(X_t), X_t OU.

Contient :
- filtre particulaire sur comptages ;
- filtre particulaire sur temps exacts ;
- EKF sur comptages ;
- exemples d'appel dans le bloc principal.
"""
from __future__ import annotations
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaln, logsumexp

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
SECONDS_PER_HOUR = 3600.0
SECONDS_PER_MINUTE = 60.0


def ou_exact_step(x_previous: FloatArray, dt: float, kappa: float,
                  long_run_mean: float, sigma: float,
                  rng: np.random.Generator) -> FloatArray:
    """Transition exacte d'un OU sur un pas dt."""
    if dt < 0 or kappa <= 0 or sigma < 0:
        raise ValueError("Paramètres OU invalides.")
    phi = np.exp(-kappa * dt)
    mean = long_run_mean + phi * (x_previous - long_run_mean)
    var = sigma**2 * (1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa)
    return mean + np.sqrt(max(var, 0.0)) * rng.normal(size=x_previous.shape)


def normalize_log_weights(log_weights: FloatArray) -> tuple[FloatArray, float]:
    """Normalise des poids calculés en logarithmes."""
    c = float(logsumexp(log_weights))
    return np.exp(log_weights - c), c


def effective_sample_size(weights: FloatArray) -> float:
    """ESS = 1 / sum(w_i^2)."""
    return float(1.0 / np.sum(weights**2))


def systematic_resampling_indices(weights: FloatArray,
                                  rng: np.random.Generator) -> IntArray:
    """Indices d'un rééchantillonnage systématique."""
    m = len(weights)
    positions = rng.uniform(0.0, 1.0 / m) + np.arange(m) / m
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right").astype(np.int64)


def weighted_mean_variance(particles: FloatArray,
                           weights: FloatArray) -> tuple[float, float]:
    mean = float(np.sum(weights * particles))
    var = float(np.sum(weights * (particles - mean) ** 2))
    return mean, var


def lambda0_double_peak(t: ArrayLike, base: float = 0.02,
                        amplitude_9h: float = 0.25,
                        amplitude_16h: float = 0.30,
                        sigma_9h: float = 45.0 * SECONDS_PER_MINUTE,
                        sigma_16h: float = 60.0 * SECONDS_PER_MINUTE) -> FloatArray:
    """Baseline à deux cloches gaussiennes centrées à 9 h et 16 h."""
    t = np.asarray(t, dtype=float)
    m1, m2 = 9.0 * SECONDS_PER_HOUR, 16.0 * SECONDS_PER_HOUR
    return (base
            + amplitude_9h * np.exp(-0.5 * ((t - m1) / sigma_9h) ** 2)
            + amplitude_16h * np.exp(-0.5 * ((t - m2) / sigma_16h) ** 2))


def integrate_baseline(lambda0: Callable[[ArrayLike], ArrayLike],
                       left: float, right: float, n_grid: int = 30) -> float:
    """Intègre lambda0 sur [left,right] par trapèzes."""
    grid = np.linspace(left, right, n_grid)
    vals = np.asarray(lambda0(grid), dtype=float)
    if np.any(vals < 0):
        raise ValueError("lambda0 doit être positive.")
    return float(np.trapezoid(vals, grid))


def compute_baseline_integrals(lambda0: Callable[[ArrayLike], ArrayLike],
                               bin_edges: ArrayLike,
                               n_grid: int = 30) -> FloatArray:
    edges = np.asarray(bin_edges, dtype=float)
    return np.array([
        integrate_baseline(lambda0, edges[k], edges[k + 1], n_grid)
        for k in range(len(edges) - 1)
    ])


def build_counts_from_events(event_times: ArrayLike,
                             observation_start: float,
                             observation_end: float,
                             bin_width: float) -> tuple[IntArray, FloatArray]:
    """Agrège les événements en comptages réguliers."""
    edges = np.arange(observation_start, observation_end + bin_width, bin_width)
    if edges[-1] != observation_end:
        edges[-1] = observation_end
    counts, _ = np.histogram(np.asarray(event_times, dtype=float), bins=edges)
    return counts.astype(np.int64), edges


def bootstrap_particle_filter_counts(
    counts: ArrayLike,
    bin_edges: ArrayLike,
    lambda0: Callable[[ArrayLike], ArrayLike],
    *, n_particles: int = 2000, kappa: float,
    long_run_mean: float, sigma: float,
    prior_mean: float, prior_std: float,
    ess_threshold: float = 0.5,
    integration_grid_size: int = 30,
    seed: int | None = None,
) -> dict[str, FloatArray | float]:
    """Bootstrap PF pour Y_k|X_k ~ Poisson(A_k exp(X_k))."""
    counts = np.asarray(counts, dtype=int)
    edges = np.asarray(bin_edges, dtype=float)
    if len(edges) != len(counts) + 1:
        raise ValueError("len(bin_edges) doit valoir len(counts)+1.")

    rng = np.random.default_rng(seed)
    n = len(counts)
    A = compute_baseline_integrals(lambda0, edges, integration_grid_size)

    particles_hist = np.empty((n, n_particles))
    weights_hist = np.empty((n, n_particles))
    x_hat = np.empty(n)
    x_var = np.empty(n)
    lambda_hat = np.empty(n)
    predicted_count = np.empty(n)
    ess = np.empty(n)
    resampled = np.zeros(n, dtype=bool)

    particles = rng.normal(prior_mean, prior_std, n_particles)
    weights = np.full(n_particles, 1.0 / n_particles)
    loglik = 0.0

    for k in range(n):
        dt = edges[k + 1] - edges[k]
        particles = ou_exact_step(particles, dt, kappa, long_run_mean, sigma, rng)

        means = np.maximum(A[k] * np.exp(particles), np.finfo(float).tiny)
        predicted_count[k] = float(np.sum(weights * means))

        y = counts[k]
        log_obs = y * np.log(means) - means - gammaln(y + 1.0)
        weights, inc = normalize_log_weights(np.log(weights) + log_obs)
        loglik += inc

        x_hat[k], x_var[k] = weighted_mean_variance(particles, weights)
        lambda_hat[k] = float(lambda0(edges[k + 1])) * np.sum(weights * np.exp(particles))
        ess[k] = effective_sample_size(weights)
        particles_hist[k], weights_hist[k] = particles, weights

        if ess[k] < ess_threshold * n_particles:
            idx = systematic_resampling_indices(weights, rng)
            particles = particles[idx]
            weights = np.full(n_particles, 1.0 / n_particles)
            resampled[k] = True

    return {
        "times": edges[1:], "filtered_mean_x": x_hat,
        "filtered_var_x": x_var, "filtered_mean_lambda": lambda_hat,
        "predictive_count_mean": predicted_count, "ess": ess,
        "resampled": resampled, "particles": particles_hist,
        "weights": weights_hist, "log_likelihood": float(loglik),
    }


def bootstrap_particle_filter_event_times(
    event_times: ArrayLike,
    lambda0: Callable[[ArrayLike], ArrayLike],
    observation_start: float,
    observation_end: float,
    *, n_particles: int = 2000, kappa: float,
    long_run_mean: float, sigma: float,
    prior_mean: float, prior_std: float,
    max_substep: float = 5.0,
    ess_threshold: float = 0.5,
    seed: int | None = None,
) -> dict[str, FloatArray | float]:
    """Bootstrap PF utilisant les temps exacts des événements."""
    events = np.asarray(event_times, dtype=float)
    rng = np.random.default_rng(seed)
    m = len(events)

    particles_hist = np.empty((m, n_particles))
    weights_hist = np.empty((m, n_particles))
    x_hat = np.empty(m)
    x_var = np.empty(m)
    lambda_hat = np.empty(m)
    ess = np.empty(m)
    resampled = np.zeros(m, dtype=bool)

    particles = rng.normal(prior_mean, prior_std, n_particles)
    weights = np.full(n_particles, 1.0 / n_particles)
    previous_time = observation_start
    loglik = 0.0

    for j, event_time in enumerate(events):
        length = event_time - previous_time
        n_steps = max(1, int(np.ceil(length / max_substep)))
        grid = np.linspace(previous_time, event_time, n_steps + 1)
        compensator = np.zeros(n_particles)
        lambda_left = float(lambda0(grid[0])) * np.exp(particles)

        for s in range(n_steps):
            dt = grid[s + 1] - grid[s]
            next_particles = ou_exact_step(
                particles, dt, kappa, long_run_mean, sigma, rng
            )
            lambda_right = float(lambda0(grid[s + 1])) * np.exp(next_particles)
            compensator += 0.5 * (lambda_left + lambda_right) * dt
            particles, lambda_left = next_particles, lambda_right

        event_intensity = np.maximum(lambda_left, np.finfo(float).tiny)
        log_obs = np.log(event_intensity) - compensator
        weights, inc = normalize_log_weights(np.log(weights) + log_obs)
        loglik += inc

        x_hat[j], x_var[j] = weighted_mean_variance(particles, weights)
        lambda_hat[j] = float(np.sum(weights * event_intensity))
        ess[j] = effective_sample_size(weights)
        particles_hist[j], weights_hist[j] = particles, weights

        if ess[j] < ess_threshold * n_particles:
            idx = systematic_resampling_indices(weights, rng)
            particles = particles[idx]
            weights = np.full(n_particles, 1.0 / n_particles)
            resampled[j] = True

        previous_time = event_time

    # Survie après le dernier événement.
    if observation_end > previous_time:
        length = observation_end - previous_time
        n_steps = max(1, int(np.ceil(length / max_substep)))
        grid = np.linspace(previous_time, observation_end, n_steps + 1)
        comp = np.zeros(n_particles)
        lambda_left = float(lambda0(grid[0])) * np.exp(particles)
        for s in range(n_steps):
            dt = grid[s + 1] - grid[s]
            next_particles = ou_exact_step(
                particles, dt, kappa, long_run_mean, sigma, rng
            )
            lambda_right = float(lambda0(grid[s + 1])) * np.exp(next_particles)
            comp += 0.5 * (lambda_left + lambda_right) * dt
            particles, lambda_left = next_particles, lambda_right
        weights, inc = normalize_log_weights(np.log(weights) - comp)
        loglik += inc

    return {
        "times": events, "filtered_mean_x": x_hat,
        "filtered_var_x": x_var, "filtered_mean_lambda": lambda_hat,
        "ess": ess, "resampled": resampled,
        "particles": particles_hist, "weights": weights_hist,
        "final_particles": particles, "final_weights": weights,
        "log_likelihood": float(loglik),
    }


def extended_kalman_filter_counts(
    counts: ArrayLike,
    bin_edges: ArrayLike,
    lambda0: Callable[[ArrayLike], ArrayLike],
    *, kappa: float, long_run_mean: float, sigma: float,
    initial_mean: float, initial_variance: float,
    integration_grid_size: int = 30,
    minimum_variance: float = 1e-10,
) -> dict[str, FloatArray | float]:
    """EKF pour Y_k|X_k ~ Poisson(A_k exp(X_k))."""
    counts = np.asarray(counts, dtype=int)
    edges = np.asarray(bin_edges, dtype=float)
    A = compute_baseline_integrals(lambda0, edges, integration_grid_size)
    n = len(counts)

    pred_m, pred_v = np.empty(n), np.empty(n)
    filt_m, filt_v = np.empty(n), np.empty(n)
    pred_count = np.empty(n)
    innov, innov_v, gain = np.empty(n), np.empty(n), np.empty(n)
    lambda_hat = np.empty(n)

    m, P = float(initial_mean), float(initial_variance)
    approx_loglik = 0.0

    for k in range(n):
        dt = edges[k + 1] - edges[k]
        phi = np.exp(-kappa * dt)
        Q = sigma**2 * (1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa)

        m_prior = long_run_mean + phi * (m - long_run_mean)
        P_prior = phi**2 * P + Q
        pred_m[k], pred_v[k] = m_prior, P_prior

        h = A[k] * np.exp(m_prior)
        H = h
        R = max(h, minimum_variance)
        S = max(H**2 * P_prior + R, minimum_variance)
        nu = counts[k] - h
        K = P_prior * H / S

        m = m_prior + K * nu
        P = (1.0 - K * H) ** 2 * P_prior + K**2 * R
        P = max(P, minimum_variance)

        filt_m[k], filt_v[k] = m, P
        pred_count[k], innov[k], innov_v[k], gain[k] = h, nu, S, K
        lambda_hat[k] = float(lambda0(edges[k + 1])) * np.exp(m + 0.5 * P)
        approx_loglik += -0.5 * (np.log(2.0 * np.pi) + np.log(S) + nu**2 / S)

    return {
        "times": edges[1:], "predicted_mean_x": pred_m,
        "predicted_var_x": pred_v, "filtered_mean_x": filt_m,
        "filtered_var_x": filt_v, "predicted_count_mean": pred_count,
        "innovation": innov, "innovation_variance": innov_v,
        "kalman_gain": gain, "filtered_mean_lambda": lambda_hat,
        "approximate_log_likelihood": float(approx_loglik),
    }


if __name__ == "__main__":
    # Remplacer ce jeu fictif par les temps obtenus par thinning.
    observation_start = 8.0 * SECONDS_PER_HOUR
    observation_end = 18.0 * SECONDS_PER_HOUR
    rng = np.random.default_rng(123)
    event_times = np.sort(rng.uniform(observation_start, observation_end, 500))

    kappa = 1.0 / (30.0 * SECONDS_PER_MINUTE)
    mu = 0.0
    sigma = 0.015
    prior_std = sigma / np.sqrt(2.0 * kappa)

    counts, bin_edges = build_counts_from_events(
        event_times, observation_start, observation_end, bin_width=60.0
    )

    pf_counts = bootstrap_particle_filter_counts(
        counts, bin_edges, lambda0_double_peak,
        n_particles=2000, kappa=kappa, long_run_mean=mu,
        sigma=sigma, prior_mean=mu, prior_std=prior_std,
        ess_threshold=0.5, seed=123,
    )

    pf_events = bootstrap_particle_filter_event_times(
        event_times, lambda0_double_peak,
        observation_start, observation_end,
        n_particles=2000, kappa=kappa, long_run_mean=mu,
        sigma=sigma, prior_mean=mu, prior_std=prior_std,
        max_substep=5.0, ess_threshold=0.5, seed=123,
    )

    ekf = extended_kalman_filter_counts(
        counts, bin_edges, lambda0_double_peak,
        kappa=kappa, long_run_mean=mu, sigma=sigma,
        initial_mean=mu, initial_variance=prior_std**2,
    )

    print("Log-vraisemblance PF comptages :", pf_counts["log_likelihood"])
    print("Log-vraisemblance PF temps exacts :", pf_events["log_likelihood"])
    print("Log-vraisemblance EKF approchée :", ekf["approximate_log_likelihood"])

    hours = pf_counts["times"] / SECONDS_PER_HOUR
    plt.figure(figsize=(11, 5))
    plt.plot(hours, pf_counts["filtered_mean_x"], label="PF comptages")
    plt.plot(hours, ekf["filtered_mean_x"], label="EKF comptages")
    plt.xlabel("Heure")
    plt.ylabel("État latent filtré")
    plt.legend()
    plt.tight_layout()
    plt.show()
