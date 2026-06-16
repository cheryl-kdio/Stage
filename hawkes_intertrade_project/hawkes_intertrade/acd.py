"""Autoregressive conditional duration benchmark with volume.

The model implemented here is a log-ACD with exponential innovations:

    duration_t = psi_t * epsilon_t
    epsilon_t ~ Exp(1)
    log psi_t = omega + a log(duration_{t-1}) + b log(psi_{t-1})
                + gamma mark_{t-1}

where mark is usually a standardized log-volume.

This is intentionally compact and robust enough for benchmarking the added
predictive value of volume before fitting a Hawkes model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize


@dataclass
class LogACDVolumeMLE:
    """Maximum-likelihood estimator for a Log-ACD model with one covariate.

    Parameters
    ----------
    max_iter:
        Maximum number of L-BFGS-B iterations.
    tol:
        Optimization tolerance.
    eps:
        Numerical floor for durations.
    """

    max_iter: int = 2000
    tol: float = 1e-9
    eps: float = 1e-12

    def _compute_log_psi(self, theta: np.ndarray, durations: np.ndarray, marks: np.ndarray) -> np.ndarray:
        omega, a, b, gamma = theta
        n = durations.size
        log_d = np.log(np.maximum(durations, self.eps))
        log_psi = np.empty(n, dtype=float)
        log_psi[0] = np.mean(log_d)
        for t in range(1, n):
            log_psi[t] = omega + a * log_d[t - 1] + b * log_psi[t - 1] + gamma * marks[t - 1]
        return log_psi

    def _neg_loglik(self, theta: np.ndarray, durations: np.ndarray, marks: np.ndarray) -> float:
        log_psi = self._compute_log_psi(theta, durations, marks)
        psi = np.exp(np.clip(log_psi, -50.0, 50.0))
        nll = np.sum(log_psi + durations / psi)
        if not np.isfinite(nll):
            return np.inf
        return float(nll)

    def fit(
        self,
        durations: np.ndarray,
        marks: Optional[np.ndarray] = None,
        x0: Optional[np.ndarray] = None,
    ) -> "LogACDVolumeMLE":
        """Fit the model.

        Parameters
        ----------
        durations:
            Positive intertrade durations.
        marks:
            Optional covariate. If None, zeros are used. For volume studies,
            use standardized log-volume lagged by one event.
        x0:
            Optional initial vector [omega, a, b, gamma].
        """

        durations = np.asarray(durations, dtype=float).ravel()
        if durations.size < 3:
            raise ValueError("At least three durations are required.")
        if np.any(~np.isfinite(durations)) or np.any(durations <= 0):
            raise ValueError("durations must be finite and strictly positive.")

        if marks is None:
            marks_arr = np.zeros_like(durations)
        else:
            marks_arr = np.asarray(marks, dtype=float).ravel()
            if marks_arr.shape != durations.shape:
                raise ValueError("marks and durations must have the same shape.")
            if np.any(~np.isfinite(marks_arr)):
                raise ValueError("marks must be finite.")

        if x0 is None:
            log_d = np.log(np.maximum(durations, self.eps))
            x0 = np.array([np.mean(log_d) * 0.1, 0.1, 0.8, 0.0], dtype=float)
        else:
            x0 = np.asarray(x0, dtype=float).ravel()
            if x0.size != 4:
                raise ValueError("x0 must have length 4: [omega, a, b, gamma].")

        # Bounds are deliberately mild. The stationarity-like condition is not
        # enforced here because this class is primarily a predictive benchmark.
        bounds = [(None, None), (-2.0, 2.0), (-0.99, 0.99), (-5.0, 5.0)]

        result = minimize(
            lambda th: self._neg_loglik(th, durations, marks_arr),
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        self.result_ = result
        self.success_ = bool(result.success)
        self.message_ = result.message
        self.params_ = result.x.copy()
        self.omega_, self.a_, self.b_, self.gamma_ = self.params_
        self.log_likelihood_ = -float(result.fun)
        self.fitted_log_psi_ = self._compute_log_psi(self.params_, durations, marks_arr)
        self.fitted_psi_ = np.exp(np.clip(self.fitted_log_psi_, -50.0, 50.0))
        self.durations_ = durations.copy()
        self.marks_ = marks_arr.copy()
        return self

    def predict_conditional_duration(self, durations: np.ndarray, marks: Optional[np.ndarray] = None) -> np.ndarray:
        """Return fitted conditional expected durations psi_t."""
        if not hasattr(self, "params_"):
            raise RuntimeError("The model must be fitted first.")
        durations = np.asarray(durations, dtype=float).ravel()
        if marks is None:
            marks_arr = np.zeros_like(durations)
        else:
            marks_arr = np.asarray(marks, dtype=float).ravel()
        log_psi = self._compute_log_psi(self.params_, durations, marks_arr)
        return np.exp(np.clip(log_psi, -50.0, 50.0))

    def summary(self) -> dict:
        """Return a small dictionary of fitted parameters."""
        if not hasattr(self, "params_"):
            raise RuntimeError("The model must be fitted first.")
        return {
            "omega": float(self.omega_),
            "a": float(self.a_),
            "b": float(self.b_),
            "gamma_volume": float(self.gamma_),
            "log_likelihood": float(self.log_likelihood_),
            "success": self.success_,
        }
