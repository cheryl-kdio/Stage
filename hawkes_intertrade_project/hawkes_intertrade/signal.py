"""Signal utilities for fitted exponential Hawkes models."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def current_hawkes_state(events: Sequence[Sequence[float]], decays: np.ndarray, t: float) -> np.ndarray:
    """Compute r[i, j](t) = beta[i, j] sum_{s in node j, s < t} exp(-beta[i, j](t-s))."""
    beta = np.asarray(decays, dtype=float)
    d = beta.shape[0]
    r = np.zeros((d, d), dtype=float)
    for j, arr in enumerate(events):
        x = np.asarray(arr, dtype=float).ravel()
        x = x[x < t]
        if x.size == 0:
            continue
        lag = t - x
        r[:, j] = np.sum(beta[:, [j]] * np.exp(-beta[:, [j]] * lag[None, :]), axis=1)
    return r


def integrated_intensity_horizon(model, events: Sequence[Sequence[float]], t: float, horizon: float) -> np.ndarray:
    """Approximate expected event counts per node over [t, t+horizon].

    For an exponential Hawkes conditional on no new jumps inside the horizon:

        int_t^{t+h} lambda_i(s) ds
        = mu_i h + sum_j alpha_ij r_ij(t) (1-exp(-beta_ij h))/beta_ij

    This quantity is a useful local signal. It is not a full forecast including
    recursively generated future offspring inside the horizon.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    mu = np.asarray(model.baseline_, dtype=float)
    alpha = np.asarray(model.adjacency_, dtype=float)
    beta = np.asarray(model.decays_, dtype=float)
    r = current_hawkes_state(events, beta, float(t))
    kernel_int = r * (1.0 - np.exp(-beta * horizon)) / beta
    return mu * horizon + np.sum(alpha * kernel_int, axis=1)


def activity_direction_score(
    model,
    events: Sequence[Sequence[float]],
    names: Sequence[str],
    t: float,
    horizon: float,
    up_name: str = "P_UP",
    down_name: str = "P_DOWN",
    eps: float = 1e-12,
) -> dict:
    """Compute activity and directional scores from P_UP/P_DOWN intensities."""
    names = list(names)
    if up_name not in names or down_name not in names:
        raise ValueError(f"names must contain {up_name} and {down_name}.")
    expected = integrated_intensity_horizon(model, events, t, horizon)
    up = float(expected[names.index(up_name)])
    down = float(expected[names.index(down_name)])
    activity = up + down
    direction = (up - down) / (activity + eps)
    return {
        "expected_counts": expected,
        "expected_up": up,
        "expected_down": down,
        "activity": activity,
        "direction": direction,
        "signal": activity * direction,
    }
