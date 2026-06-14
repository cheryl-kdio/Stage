"""
hawkes_estimators.py

Educational estimators for univariate and multivariate Hawkes processes.

Implemented methods
-------------------
PARAMETRIC
1. UnivariateHawkesExpMLE
   Maximum likelihood estimation for an exponential Hawkes process.

2. UnivariateHawkesExpEM
   EM estimation for an exponential Hawkes process.

3. MultivariateHawkesExpMLE
   Maximum likelihood estimation for a multivariate exponential Hawkes process
   with fixed decay matrix.

4. MultivariateHawkesExpEM
   EM estimation for a multivariate exponential Hawkes process with fixed decay
   matrix.

NON-PARAMETRIC
5. UnivariateHawkesNonparamEM
   Non-parametric EM estimation with piecewise-constant histogram kernels.

6. MultivariateHawkesNonparamEM
   Multivariate non-parametric EM with piecewise-constant histogram kernels.

7. HawkesL2ContrastEstimator
   Discretized L2-contrast estimator. Works for univariate and multivariate data.

8. UnivariateWienerHopfEstimator
   Non-parametric Wiener-Hopf estimator from second-order statistics.

9. MultivariateWienerHopfEstimator
   Multivariate Wiener-Hopf estimator from second-order statistics.

Notes
-----
These classes are written for clarity and experimentation. They are not intended
as a drop-in replacement for specialized optimized libraries. Several estimators
below use O(n^2) loops, which is acceptable for small and medium samples but not
for very large high-frequency datasets without further optimization.

Dependencies
------------
Only numpy and scipy are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize, minimize_scalar
from scipy.linalg import solve


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _as_sorted_1d_times(times: ArrayLike) -> np.ndarray:
    """Return sorted event times as a 1D float array."""
    x = np.asarray(times, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("times must contain at least one event.")
    if np.any(~np.isfinite(x)):
        raise ValueError("times contains non-finite values.")
    if np.any(x < 0):
        raise ValueError("event times must be non-negative.")
    x = np.sort(x)
    if np.any(np.diff(x) <= 0):
        # Ties break the simple point-process likelihood with unit jumps.
        raise ValueError("event times must be strictly increasing; ties were found.")
    return x


def _as_multivariate_events(times: ArrayLike, marks: ArrayLike, n_dims: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """Validate and sort multivariate marked event data.

    Parameters
    ----------
    times : array-like, shape (n_events,)
        Event times.
    marks : array-like, shape (n_events,)
        Integer marks in {0, ..., D-1}.
    n_dims : int or None
        Number of event types. If None, it is inferred from max(mark)+1.

    Returns
    -------
    times_sorted, marks_sorted, D
    """
    t = np.asarray(times, dtype=float).reshape(-1)
    k = np.asarray(marks, dtype=int).reshape(-1)
    if t.size == 0:
        raise ValueError("times must contain at least one event.")
    if t.shape != k.shape:
        raise ValueError("times and marks must have the same length.")
    if np.any(~np.isfinite(t)):
        raise ValueError("times contains non-finite values.")
    if np.any(t < 0):
        raise ValueError("event times must be non-negative.")
    if np.any(k < 0):
        raise ValueError("marks must be non-negative integer labels.")

    order = np.argsort(t)
    t = t[order]
    k = k[order]
    if np.any(np.diff(t) <= 0):
        raise ValueError("event times must be strictly increasing; ties were found.")

    D = int(k.max()) + 1 if n_dims is None else int(n_dims)
    if D <= 0:
        raise ValueError("n_dims must be positive.")
    if np.any(k >= D):
        raise ValueError("marks contain a value >= n_dims.")
    return t, k, D


def _spectral_radius(matrix: np.ndarray) -> float:
    """Return the spectral radius of a square matrix."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square.")
    if matrix.size == 1:
        return float(abs(matrix.item()))
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def _scale_to_stable(matrix: np.ndarray, target_radius: float = 0.999) -> np.ndarray:
    """Scale a non-negative branching matrix if its spectral radius exceeds target."""
    out = np.asarray(matrix, dtype=float).copy()
    rho = _spectral_radius(out)
    if rho > target_radius and rho > 0:
        out *= target_radius / rho
    return out


def _safe_log(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically safe log; returns -inf for non-positive inputs."""
    return np.log(x) if np.all(np.asarray(x) > 0) else -np.inf


# ---------------------------------------------------------------------------
# Parametric univariate MLE
# ---------------------------------------------------------------------------


@dataclass
class UnivariateHawkesExpMLE:
    """MLE for a univariate exponential Hawkes process.

    Model convention
    ----------------
    lambda(t) = mu + sum_{tj < t} alpha * exp(-beta * (t - tj))

    Stability condition
    -------------------
    integral_0^infty alpha exp(-beta t) dt = alpha / beta < 1.

    Parameters estimated
    --------------------
    mu > 0, alpha >= 0, beta > 0 with alpha < beta.

    Implementation details
    ----------------------
    The exponential kernel permits an O(n) likelihood recursion:
        A_i = sum_{j<i} exp(-beta * (t_i - t_j))
        A_i = exp(-beta * (t_i - t_{i-1})) * (1 + A_{i-1}).
    """

    T: Optional[float] = None
    enforce_stationarity: bool = True
    epsilon: float = 1e-9
    result_: Optional[object] = None
    params_: Optional[Dict[str, float]] = None
    loglik_: Optional[float] = None

    @staticmethod
    def _recursive_A(times: np.ndarray, beta: float) -> np.ndarray:
        """Compute A_i = sum_{j<i} exp(-beta (t_i - t_j)) in O(n)."""
        n = len(times)
        A = np.zeros(n, dtype=float)
        for i in range(1, n):
            A[i] = np.exp(-beta * (times[i] - times[i - 1])) * (1.0 + A[i - 1])
        return A

    @classmethod
    def loglikelihood(cls, times: ArrayLike, T: float, mu: float, alpha: float, beta: float) -> float:
        """Evaluate the log-likelihood for given parameters."""
        times = _as_sorted_1d_times(times)
        if T < times[-1]:
            raise ValueError("T must be >= last event time.")
        if mu <= 0 or alpha < 0 or beta <= 0 or alpha >= beta:
            return -np.inf

        A = cls._recursive_A(times, beta)
        intensities = mu + alpha * A
        if np.any(intensities <= 0):
            return -np.inf

        # Integral of lambda over [0, T].
        compensator = mu * T + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - times)))
        return float(np.sum(np.log(intensities)) - compensator)

    def fit(self, times: ArrayLike, initial: Optional[Tuple[float, float, float]] = None) -> "UnivariateHawkesExpMLE":
        """Fit the model by numerical minimization of the negative log-likelihood."""
        times = _as_sorted_1d_times(times)
        T = float(times[-1] if self.T is None else self.T)
        if T < times[-1]:
            raise ValueError("T must be >= last event time.")
        n = len(times)

        if initial is None:
            # Conservative initialization: part exogenous, part endogenous.
            empirical_rate = n / T
            mu0 = max(0.5 * empirical_rate, self.epsilon)
            beta0 = 1.0 / max(np.median(np.diff(np.r_[0.0, times])), self.epsilon)
            alpha0 = 0.5 * beta0  # alpha/beta = 0.5
            initial = (mu0, alpha0, beta0)

        mu0, alpha0, beta0 = map(float, initial)
        if mu0 <= 0 or alpha0 < 0 or beta0 <= 0:
            raise ValueError("initial must satisfy mu>0, alpha>=0, beta>0.")
        if self.enforce_stationarity and alpha0 >= beta0:
            alpha0 = 0.5 * beta0

        # Reparameterization:
        # mu = exp(x0), beta = exp(x1), alpha = beta * sigmoid(x2)
        # This enforces mu>0, beta>0, alpha/beta in (0,1).
        def logit(p: float) -> float:
            p = np.clip(p, 1e-6, 1.0 - 1e-6)
            return np.log(p / (1.0 - p))

        x0 = np.array([
            np.log(mu0),
            np.log(beta0),
            logit(alpha0 / beta0 if beta0 > 0 else 0.5),
        ])

        def unpack(x: np.ndarray) -> Tuple[float, float, float]:
            mu = float(np.exp(x[0]))
            beta = float(np.exp(x[1]))
            if self.enforce_stationarity:
                eta = 1.0 / (1.0 + np.exp(-x[2]))
                alpha = beta * eta
            else:
                alpha = float(np.exp(x[2]))
            return mu, alpha, beta

        def objective(x: np.ndarray) -> float:
            mu, alpha, beta = unpack(x)
            ll = self.loglikelihood(times, T, mu, alpha, beta)
            return -ll if np.isfinite(ll) else 1e100

        res = minimize(objective, x0, method="L-BFGS-B")
        mu, alpha, beta = unpack(res.x)
        self.result_ = res
        self.params_ = {
            "mu": mu,
            "alpha": alpha,
            "beta": beta,
            "branching_ratio": alpha / beta,
        }
        self.loglik_ = self.loglikelihood(times, T, mu, alpha, beta)
        return self

    def intensity_at_events(self, times: ArrayLike) -> np.ndarray:
        """Return fitted intensities evaluated at observed event times."""
        if self.params_ is None:
            raise RuntimeError("fit must be called before intensity_at_events.")
        times = _as_sorted_1d_times(times)
        beta = self.params_["beta"]
        A = self._recursive_A(times, beta)
        return self.params_["mu"] + self.params_["alpha"] * A


# ---------------------------------------------------------------------------
# Parametric univariate EM
# ---------------------------------------------------------------------------


@dataclass
class UnivariateHawkesExpEM:
    """EM estimator for a univariate exponential Hawkes process.

    Model convention
    ----------------
    lambda(t) = mu + sum_{tj < t} eta * beta * exp(-beta * (t - tj))

    Here eta is the branching ratio directly, because
        integral eta * beta exp(-beta t) dt = eta.

    If beta is fixed, the M-step has closed form for mu and eta.
    If beta is not fixed, beta is optimized by one-dimensional profiling.
    """

    T: Optional[float] = None
    max_iter: int = 200
    tol: float = 1e-6
    beta_fixed: Optional[float] = None
    beta_bounds: Tuple[float, float] = (1e-4, 1e3)
    verbose: bool = False
    params_: Optional[Dict[str, float]] = None
    history_: Optional[List[Dict[str, float]]] = None

    @staticmethod
    def loglikelihood(times: ArrayLike, T: float, mu: float, eta: float, beta: float) -> float:
        """Observed log-likelihood under the eta*beta convention."""
        times = _as_sorted_1d_times(times)
        if T < times[-1]:
            raise ValueError("T must be >= last event time.")
        if mu <= 0 or eta < 0 or eta >= 1 or beta <= 0:
            return -np.inf

        ll_events = 0.0
        for i, ti in enumerate(times):
            if i == 0:
                lam = mu
            else:
                dt = ti - times[:i]
                lam = mu + np.sum(eta * beta * np.exp(-beta * dt))
            if lam <= 0:
                return -np.inf
            ll_events += np.log(lam)

        compensator = mu * T + eta * np.sum(1.0 - np.exp(-beta * (T - times)))
        return float(ll_events - compensator)

    @staticmethod
    def _e_step(times: np.ndarray, mu: float, eta: float, beta: float) -> Tuple[float, float, float]:
        """Return sufficient statistics E[N0], E[N1], E[sum delays]."""
        n0 = 0.0  # expected number of immigrants
        n1 = 0.0  # expected number of offspring
        S = 0.0   # expected sum of parent-child delays

        for i, ti in enumerate(times):
            if i == 0:
                n0 += 1.0
                continue
            dt = ti - times[:i]
            weights = eta * beta * np.exp(-beta * dt)
            lam = mu + np.sum(weights)
            p0 = mu / lam
            pij = weights / lam
            n0 += p0
            n1 += float(np.sum(pij))
            S += float(np.sum(pij * dt))
        return n0, n1, S

    @staticmethod
    def _A(times: np.ndarray, T: float, beta: float) -> float:
        """A(beta) = sum_j (1 - exp(-beta * (T - t_j)))."""
        return float(np.sum(1.0 - np.exp(-beta * (T - times))))

    def fit(self, times: ArrayLike, initial: Optional[Tuple[float, float, float]] = None) -> "UnivariateHawkesExpEM":
        """Fit by EM."""
        times = _as_sorted_1d_times(times)
        T = float(times[-1] if self.T is None else self.T)
        if T < times[-1]:
            raise ValueError("T must be >= last event time.")
        n = len(times)

        if initial is None:
            empirical_rate = n / T
            mu = max(0.5 * empirical_rate, 1e-9)
            eta = 0.5
            beta = self.beta_fixed if self.beta_fixed is not None else 1.0 / max(np.median(np.diff(np.r_[0.0, times])), 1e-9)
        else:
            mu, eta, beta = map(float, initial)
        if self.beta_fixed is not None:
            beta = float(self.beta_fixed)

        if mu <= 0 or eta < 0 or eta >= 1 or beta <= 0:
            raise ValueError("initial must satisfy mu>0, eta in [0,1), beta>0.")

        history: List[Dict[str, float]] = []

        for it in range(self.max_iter):
            old = np.array([mu, eta, beta], dtype=float)
            n0, n1, S = self._e_step(times, mu, eta, beta)

            mu_new = max(n0 / T, 1e-12)

            if n1 <= 1e-12:
                eta_new = 0.0
                beta_new = beta
            elif self.beta_fixed is not None:
                beta_new = beta
                denom = self._A(times, T, beta_new)
                eta_new = min(max(n1 / max(denom, 1e-12), 0.0), 1.0 - 1e-10)
            else:
                # Profile Q(beta). Constants are omitted.
                def neg_profile_Q(b: float) -> float:
                    if b <= 0:
                        return np.inf
                    A = self._A(times, T, b)
                    if A <= 0:
                        return np.inf
                    # eta(b)=n1/A(b). Q = n1 log eta + n1 log beta - beta S - eta A.
                    # Since eta A = n1 and n1 log n1 is constant:
                    # Q_profile = n1 log beta - beta S - n1 log A + constant.
                    return -(n1 * np.log(b) - b * S - n1 * np.log(A))

                res = minimize_scalar(neg_profile_Q, bounds=self.beta_bounds, method="bounded")
                if not res.success:
                    raise RuntimeError("beta optimization failed in M-step.")
                beta_new = float(res.x)
                eta_new = n1 / max(self._A(times, T, beta_new), 1e-12)
                eta_new = min(max(eta_new, 0.0), 1.0 - 1e-10)

            mu, eta, beta = mu_new, eta_new, beta_new
            ll = self.loglikelihood(times, T, mu, eta, beta)
            history.append({"iter": it, "mu": mu, "eta": eta, "beta": beta, "loglik": ll})

            if self.verbose:
                print(f"iter={it:03d} mu={mu:.6g} eta={eta:.6g} beta={beta:.6g} loglik={ll:.6g}")

            if np.linalg.norm(np.array([mu, eta, beta]) - old) < self.tol:
                break

        self.params_ = {
            "mu": mu,
            "eta": eta,
            "beta": beta,
            "alpha_equivalent": eta * beta,
            "branching_ratio": eta,
        }
        self.history_ = history
        return self


# ---------------------------------------------------------------------------
# Parametric multivariate MLE with fixed decays
# ---------------------------------------------------------------------------


@dataclass
class MultivariateHawkesExpMLE:
    """MLE for multivariate exponential Hawkes with fixed decay matrix.

    Model convention
    ----------------
    lambda_i(t) = mu_i + sum_{events l: t_l < t} eta[i, k_l] * beta[i, k_l]
                  * exp(-beta[i, k_l] * (t - t_l))

    Here eta[i, j] is the integrated kernel mass from source j to target i.
    The stability condition is spectral_radius(eta) < 1.

    Notes
    -----
    beta is fixed. This keeps the optimization smaller and closer to what is done
    in many high-dimensional Hawkes applications.
    """

    beta: ArrayLike | float = 1.0
    n_dims: Optional[int] = None
    T: Optional[float] = None
    ridge: float = 0.0
    enforce_stationarity: bool = True
    max_iter: int = 500
    result_: Optional[object] = None
    params_: Optional[Dict[str, np.ndarray]] = None
    loglik_: Optional[float] = None

    def _beta_matrix(self, D: int) -> np.ndarray:
        beta = np.asarray(self.beta, dtype=float)
        if beta.ndim == 0:
            beta = np.full((D, D), float(beta))
        if beta.shape != (D, D):
            raise ValueError("beta must be scalar or shape (D, D).")
        if np.any(beta <= 0):
            raise ValueError("all beta entries must be > 0.")
        return beta

    @staticmethod
    def loglikelihood(times: ArrayLike, marks: ArrayLike, T: float, mu: np.ndarray, eta: np.ndarray, beta: np.ndarray) -> float:
        """Observed multivariate log-likelihood."""
        times, marks, D = _as_multivariate_events(times, marks, n_dims=len(mu))
        mu = np.asarray(mu, dtype=float)
        eta = np.asarray(eta, dtype=float)
        beta = np.asarray(beta, dtype=float)
        if mu.shape != (D,) or eta.shape != (D, D) or beta.shape != (D, D):
            raise ValueError("mu, eta, beta shapes are inconsistent.")
        if T < times[-1]:
            raise ValueError("T must be >= last event time.")
        if np.any(mu <= 0) or np.any(eta < 0) or np.any(beta <= 0):
            return -np.inf
        if _spectral_radius(eta) >= 1.0:
            return -np.inf

        ll_events = 0.0
        for m, tm in enumerate(times):
            target = marks[m]
            lam = mu[target]
            if m > 0:
                past_dt = tm - times[:m]
                past_marks = marks[:m]
                lam += np.sum(eta[target, past_marks] * beta[target, past_marks] * np.exp(-beta[target, past_marks] * past_dt))
            if lam <= 0:
                return -np.inf
            ll_events += np.log(lam)

        compensator = float(np.sum(mu) * T)
        for l, tl in enumerate(times):
            source = marks[l]
            # Contribution to all target dimensions.
            compensator += float(np.sum(eta[:, source] * (1.0 - np.exp(-beta[:, source] * (T - tl)))))
        return float(ll_events - compensator)

    def fit(self, times: ArrayLike, marks: ArrayLike, initial_mu: Optional[ArrayLike] = None, initial_eta: Optional[ArrayLike] = None) -> "MultivariateHawkesExpMLE":
        """Fit mu and eta by numerical likelihood maximization with fixed beta."""
        times, marks, D = _as_multivariate_events(times, marks, self.n_dims)
        T = float(times[-1] if self.T is None else self.T)
        if T < times[-1]:
            raise ValueError("T must be >= last event time.")
        beta = self._beta_matrix(D)

        counts = np.bincount(marks, minlength=D).astype(float)
        if initial_mu is None:
            mu0 = np.maximum(0.5 * counts / T, 1e-9)
        else:
            mu0 = np.asarray(initial_mu, dtype=float)
        if initial_eta is None:
            eta0 = np.full((D, D), 0.1 / max(D, 1))
        else:
            eta0 = np.asarray(initial_eta, dtype=float)
        if mu0.shape != (D,) or eta0.shape != (D, D):
            raise ValueError("initial shapes are inconsistent.")
        eta0 = np.maximum(eta0, 0.0)
        if self.enforce_stationarity:
            eta0 = _scale_to_stable(eta0, 0.8)

        # Parameterization: mu=exp(x[:D]), eta=softplus for non-negativity.
        # Stationarity is enforced by a penalty and optional scaling after fit.
        def softplus(x: np.ndarray) -> np.ndarray:
            return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

        def inv_softplus(y: np.ndarray) -> np.ndarray:
            y = np.maximum(y, 1e-12)
            return np.log(np.expm1(y))

        x0 = np.r_[np.log(mu0), inv_softplus(eta0).ravel()]

        def unpack(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            mu = np.exp(x[:D])
            eta = softplus(x[D:]).reshape(D, D)
            return mu, eta

        def objective(x: np.ndarray) -> float:
            mu, eta = unpack(x)
            rho = _spectral_radius(eta)
            ll = self.loglikelihood(times, marks, T, mu, eta, beta) if rho < 1.0 else -np.inf
            if not np.isfinite(ll):
                # Smooth-ish penalty outside stability region.
                return 1e50 + 1e6 * max(rho - 0.999, 0.0) ** 2
            val = -ll
            if self.ridge > 0:
                val += self.ridge * float(np.sum(eta ** 2))
            return val

        res = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": self.max_iter})
        mu, eta = unpack(res.x)
        if self.enforce_stationarity:
            eta = _scale_to_stable(eta, 0.999)
        self.result_ = res
        self.params_ = {"mu": mu, "eta": eta, "beta": beta, "branching_matrix": eta}
        self.loglik_ = self.loglikelihood(times, marks, T, mu, eta, beta)
        return self


# ---------------------------------------------------------------------------
# Parametric multivariate EM with fixed decays
# ---------------------------------------------------------------------------


@dataclass
class MultivariateHawkesExpEM:
    """EM estimator for multivariate exponential Hawkes with fixed decay matrix.

    Model convention
    ----------------
    phi_{ij}(u) = eta[i, j] * beta[i, j] * exp(-beta[i, j] u), u > 0.

    eta[i, j] is the integrated kernel mass from source j to target i.
    beta is fixed.
    """

    beta: ArrayLike | float = 1.0
    n_dims: Optional[int] = None
    T: Optional[float] = None
    max_iter: int = 100
    tol: float = 1e-6
    enforce_stationarity: bool = True
    verbose: bool = False
    params_: Optional[Dict[str, np.ndarray]] = None
    history_: Optional[List[Dict[str, float]]] = None

    def _beta_matrix(self, D: int) -> np.ndarray:
        beta = np.asarray(self.beta, dtype=float)
        if beta.ndim == 0:
            beta = np.full((D, D), float(beta))
        if beta.shape != (D, D):
            raise ValueError("beta must be scalar or shape (D, D).")
        if np.any(beta <= 0):
            raise ValueError("all beta entries must be > 0.")
        return beta

    def fit(self, times: ArrayLike, marks: ArrayLike, initial_mu: Optional[ArrayLike] = None, initial_eta: Optional[ArrayLike] = None) -> "MultivariateHawkesExpEM":
        """Fit by EM with fixed beta."""
        times, marks, D = _as_multivariate_events(times, marks, self.n_dims)
        T = float(times[-1] if self.T is None else self.T)
        if T < times[-1]:
            raise ValueError("T must be >= last event time.")
        beta = self._beta_matrix(D)
        n = len(times)

        counts = np.bincount(marks, minlength=D).astype(float)
        mu = np.maximum(0.5 * counts / T, 1e-9) if initial_mu is None else np.asarray(initial_mu, dtype=float)
        eta = np.full((D, D), 0.1 / max(D, 1)) if initial_eta is None else np.asarray(initial_eta, dtype=float)
        eta = np.maximum(eta, 0.0)
        if self.enforce_stationarity:
            eta = _scale_to_stable(eta, 0.8)

        history: List[Dict[str, float]] = []

        for it in range(self.max_iter):
            old_mu = mu.copy()
            old_eta = eta.copy()

            immigrant_counts = np.zeros(D, dtype=float)
            offspring_counts = np.zeros((D, D), dtype=float)  # target i, source j

            # E-step: compute parent probabilities and accumulate sufficient stats.
            for m in range(n):
                target = marks[m]
                tm = times[m]
                lam = mu[target]
                if m > 0:
                    past_dt = tm - times[:m]
                    past_marks = marks[:m]
                    weights = eta[target, past_marks] * beta[target, past_marks] * np.exp(-beta[target, past_marks] * past_dt)
                    weight_sum = float(np.sum(weights))
                    lam += weight_sum
                    if lam <= 0:
                        raise RuntimeError("non-positive intensity encountered.")
                    p_parent = weights / lam
                    for source in range(D):
                        mask = past_marks == source
                        if np.any(mask):
                            offspring_counts[target, source] += float(np.sum(p_parent[mask]))
                    immigrant_counts[target] += mu[target] / lam
                else:
                    immigrant_counts[target] += 1.0

            # M-step for mu.
            mu = np.maximum(immigrant_counts / T, 1e-12)

            # M-step for eta[i,j]. Denominator is exposure for source j to target i.
            denom = np.zeros((D, D), dtype=float)
            for l, tl in enumerate(times):
                source = marks[l]
                denom[:, source] += 1.0 - np.exp(-beta[:, source] * (T - tl))
            eta = offspring_counts / np.maximum(denom, 1e-12)
            eta = np.maximum(eta, 0.0)
            if self.enforce_stationarity:
                eta = _scale_to_stable(eta, 0.999)

            rho = _spectral_radius(eta)
            diff = float(np.linalg.norm(mu - old_mu) + np.linalg.norm(eta - old_eta))
            history.append({"iter": it, "rho": rho, "diff": diff})
            if self.verbose:
                print(f"iter={it:03d} rho={rho:.6g} diff={diff:.3e}")
            if diff < self.tol:
                break

        self.params_ = {"mu": mu, "eta": eta, "beta": beta, "branching_matrix": eta}
        self.history_ = history
        return self


# ---------------------------------------------------------------------------
# Non-parametric EM, univariate histogram
# ---------------------------------------------------------------------------


@dataclass
class UnivariateHawkesNonparamEM:
    """Non-parametric EM for a univariate Hawkes process with histogram kernel.

    Kernel approximation
    --------------------
    phi(u) = sum_m c[m] * 1_{bin_edges[m] <= u < bin_edges[m+1]}.

    E-step
    ------
    p_ij = phi(t_i - t_j) / lambda(t_i).

    M-step
    ------
    c[m] = expected number of parent-child pairs with delay in bin m
           divided by exposure of bin m.
    """

    bin_edges: ArrayLike
    T: Optional[float] = None
    max_iter: int = 100
    tol: float = 1e-6
    smooth_penalty: float = 0.0
    enforce_stationarity: bool = True
    verbose: bool = False
    params_: Optional[Dict[str, np.ndarray | float]] = None
    history_: Optional[List[Dict[str, float]]] = None

    def _bin_index(self, delays: np.ndarray) -> np.ndarray:
        edges = np.asarray(self.bin_edges, dtype=float)
        return np.searchsorted(edges, delays, side="right") - 1

    def _apply_smoothing(self, c: np.ndarray) -> np.ndarray:
        """Simple post-update smoothing; not an exact penalized M-step.

        This is deliberately simple and transparent. For production use, solve the
        penalized M-step explicitly with scipy.optimize.
        """
        if self.smooth_penalty <= 0 or len(c) < 3:
            return c
        out = c.copy()
        w = np.clip(self.smooth_penalty, 0.0, 1.0)
        for m in range(1, len(c) - 1):
            out[m] = (1 - w) * c[m] + w * 0.5 * (c[m - 1] + c[m + 1])
        return np.maximum(out, 0.0)

    def fit(self, times: ArrayLike, initial_mu: Optional[float] = None, initial_c: Optional[ArrayLike] = None) -> "UnivariateHawkesNonparamEM":
        times = _as_sorted_1d_times(times)
        edges = np.asarray(self.bin_edges, dtype=float)
        if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0) or edges[0] < 0:
            raise ValueError("bin_edges must be increasing and start at >= 0.")
        M = len(edges) - 1
        T = float(times[-1] if self.T is None else self.T)
        if T < times[-1]:
            raise ValueError("T must be >= last event time.")

        n = len(times)
        mu = float(0.5 * n / T if initial_mu is None else initial_mu)
        if mu <= 0:
            raise ValueError("initial_mu must be > 0.")
        if initial_c is None:
            c = np.full(M, 0.1 / max(edges[-1], 1e-12))
        else:
            c = np.asarray(initial_c, dtype=float).copy()
        if c.shape != (M,) or np.any(c < 0):
            raise ValueError("initial_c must have shape (n_bins,) and be non-negative.")

        widths = np.diff(edges)
        history: List[Dict[str, float]] = []

        for it in range(self.max_iter):
            old_mu = mu
            old_c = c.copy()

            immigrant_count = 0.0
            pair_counts = np.zeros(M, dtype=float)

            # E-step.
            for i, ti in enumerate(times):
                if i == 0:
                    immigrant_count += 1.0
                    continue
                delays = ti - times[:i]
                b = self._bin_index(delays)
                valid = (b >= 0) & (b < M)
                weights = np.zeros_like(delays)
                weights[valid] = c[b[valid]]
                lam = mu + float(np.sum(weights))
                if lam <= 0:
                    raise RuntimeError("non-positive intensity encountered.")
                immigrant_count += mu / lam
                p = weights / lam
                for m in range(M):
                    pair_counts[m] += float(np.sum(p[b == m]))

            # M-step for mu.
            mu = max(immigrant_count / T, 1e-12)

            # Exposure for each lag bin.
            exposure = np.zeros(M, dtype=float)
            for tj in times:
                remaining = T - tj
                for m in range(M):
                    start, end = edges[m], edges[m + 1]
                    exposure[m] += max(0.0, min(end, remaining) - start)

            c = pair_counts / np.maximum(exposure, 1e-12)
            c = self._apply_smoothing(c)

            if self.enforce_stationarity:
                mass = float(np.sum(c * widths))
                if mass >= 0.999:
                    c *= 0.999 / mass

            mass = float(np.sum(c * widths))
            diff = float(abs(mu - old_mu) + np.linalg.norm(c - old_c))
            history.append({"iter": it, "mu": mu, "kernel_mass": mass, "diff": diff})
            if self.verbose:
                print(f"iter={it:03d} mu={mu:.6g} mass={mass:.6g} diff={diff:.3e}")
            if diff < self.tol:
                break

        self.params_ = {"mu": mu, "c": c, "bin_edges": edges, "kernel_mass": float(np.sum(c * widths))}
        self.history_ = history
        return self

    def kernel_values(self, u: ArrayLike) -> np.ndarray:
        """Evaluate fitted histogram kernel at lags u."""
        if self.params_ is None:
            raise RuntimeError("fit must be called before kernel_values.")
        u = np.asarray(u, dtype=float)
        c = np.asarray(self.params_["c"], dtype=float)
        edges = np.asarray(self.params_["bin_edges"], dtype=float)
        b = np.searchsorted(edges, u, side="right") - 1
        out = np.zeros_like(u, dtype=float)
        valid = (b >= 0) & (b < len(c))
        out[valid] = c[b[valid]]
        return out


# ---------------------------------------------------------------------------
# Non-parametric EM, multivariate histogram
# ---------------------------------------------------------------------------


@dataclass
class MultivariateHawkesNonparamEM:
    """Multivariate non-parametric EM with histogram kernels.

    Kernel approximation
    --------------------
    phi_{ij}(u) = sum_m c[i, j, m] * 1_{bin_m}(u)

    where i is target dimension and j is source dimension.
    """

    bin_edges: ArrayLike
    n_dims: Optional[int] = None
    T: Optional[float] = None
    max_iter: int = 50
    tol: float = 1e-6
    enforce_stationarity: bool = True
    verbose: bool = False
    params_: Optional[Dict[str, np.ndarray]] = None
    history_: Optional[List[Dict[str, float]]] = None

    def fit(self, times: ArrayLike, marks: ArrayLike, initial_mu: Optional[ArrayLike] = None, initial_c: Optional[ArrayLike] = None) -> "MultivariateHawkesNonparamEM":
        times, marks, D = _as_multivariate_events(times, marks, self.n_dims)
        edges = np.asarray(self.bin_edges, dtype=float)
        if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0) or edges[0] < 0:
            raise ValueError("bin_edges must be increasing and start at >= 0.")
        M = len(edges) - 1
        widths = np.diff(edges)
        T = float(times[-1] if self.T is None else self.T)
        if T < times[-1]:
            raise ValueError("T must be >= last event time.")
        n = len(times)

        counts = np.bincount(marks, minlength=D).astype(float)
        mu = np.maximum(0.5 * counts / T, 1e-12) if initial_mu is None else np.asarray(initial_mu, dtype=float)
        if mu.shape != (D,) or np.any(mu <= 0):
            raise ValueError("initial_mu must have shape (D,) and positive entries.")

        if initial_c is None:
            c = np.full((D, D, M), 0.05 / max(edges[-1], 1e-12) / max(D, 1))
        else:
            c = np.asarray(initial_c, dtype=float).copy()
        if c.shape != (D, D, M) or np.any(c < 0):
            raise ValueError("initial_c must have shape (D,D,n_bins) and be non-negative.")

        history: List[Dict[str, float]] = []

        for it in range(self.max_iter):
            old_mu = mu.copy()
            old_c = c.copy()

            immigrant_counts = np.zeros(D, dtype=float)
            pair_counts = np.zeros((D, D, M), dtype=float)

            # E-step.
            for m in range(n):
                target = marks[m]
                tm = times[m]
                lam = mu[target]
                if m > 0:
                    delays = tm - times[:m]
                    sources = marks[:m]
                    bin_ids = np.searchsorted(edges, delays, side="right") - 1
                    weights = np.zeros(m, dtype=float)
                    valid = (bin_ids >= 0) & (bin_ids < M)
                    if np.any(valid):
                        weights[valid] = c[target, sources[valid], bin_ids[valid]]
                    lam += float(np.sum(weights))
                    if lam <= 0:
                        raise RuntimeError("non-positive intensity encountered.")
                    p_parent = weights / lam
                    for source in range(D):
                        source_mask = sources == source
                        for b in range(M):
                            mask = source_mask & (bin_ids == b)
                            if np.any(mask):
                                pair_counts[target, source, b] += float(np.sum(p_parent[mask]))
                    immigrant_counts[target] += mu[target] / lam
                else:
                    immigrant_counts[target] += 1.0

            # M-step.
            mu = np.maximum(immigrant_counts / T, 1e-12)

            exposure = np.zeros((D, D, M), dtype=float)
            # Exposure depends on source events and target dimension. For each source event,
            # all target dimensions have the same time exposure per bin.
            for l, tl in enumerate(times):
                source = marks[l]
                remaining = T - tl
                for b in range(M):
                    start, end = edges[b], edges[b + 1]
                    exp_b = max(0.0, min(end, remaining) - start)
                    exposure[:, source, b] += exp_b

            c = pair_counts / np.maximum(exposure, 1e-12)
            c = np.maximum(c, 0.0)

            # Stability is based on integrated kernel matrix G[i,j].
            G = np.sum(c * widths[None, None, :], axis=2)
            if self.enforce_stationarity:
                rho = _spectral_radius(G)
                if rho >= 0.999 and rho > 0:
                    scale = 0.999 / rho
                    c *= scale
                    G *= scale
            rho = _spectral_radius(G)
            diff = float(np.linalg.norm(mu - old_mu) + np.linalg.norm(c - old_c))
            history.append({"iter": it, "rho": rho, "diff": diff})
            if self.verbose:
                print(f"iter={it:03d} rho={rho:.6g} diff={diff:.3e}")
            if diff < self.tol:
                break

        self.params_ = {"mu": mu, "c": c, "bin_edges": edges, "branching_matrix": np.sum(c * widths[None, None, :], axis=2)}
        self.history_ = history
        return self


# ---------------------------------------------------------------------------
# L2 contrast estimator: univariate and multivariate with grid approximation
# ---------------------------------------------------------------------------


@dataclass
class HawkesL2ContrastEstimator:
    """Discretized non-parametric L2-contrast estimator.

    This class works for both univariate and multivariate data.

    Model
    -----
    phi_{ij}(u) = sum_m c[i, j, m] * 1_{bin_m}(u)

    For each target i, the intensity is linear in parameters:
        lambda_i(t) = mu_i + sum_{j,m} c[i,j,m] X_{j,m}(t),
    where X_{j,m}(t) counts source-j events whose lag to t lies in bin m.

    Approximate contrast
    --------------------
    C_i(theta_i) = integral lambda_i(t)^2 dt - 2 sum_{events of type i} lambda_i(t_event)

    The integral is approximated on a uniform grid. This is not as exact as an
    analytic Gram-matrix implementation, but it is simple and transparent.
    """

    bin_edges: ArrayLike
    n_dims: Optional[int] = None
    T: Optional[float] = None
    grid_size: int = 1000
    ridge: float = 1e-6
    nonnegative: bool = True
    enforce_stationarity: bool = True
    params_: Optional[Dict[str, np.ndarray]] = None

    @staticmethod
    def _features_at(points: np.ndarray, event_times: np.ndarray, event_marks: np.ndarray, D: int, edges: np.ndarray) -> np.ndarray:
        """Build feature tensor at points.

        Returns
        -------
        X : ndarray, shape (len(points), D, M)
            X[p, j, m] is number of source-j events before points[p] whose lag
            belongs to bin m.
        """
        M = len(edges) - 1
        X = np.zeros((len(points), D, M), dtype=float)
        for p, t in enumerate(points):
            mask = event_times < t
            if not np.any(mask):
                continue
            delays = t - event_times[mask]
            sources = event_marks[mask]
            bin_ids = np.searchsorted(edges, delays, side="right") - 1
            valid = (bin_ids >= 0) & (bin_ids < M)
            for source, b in zip(sources[valid], bin_ids[valid]):
                X[p, source, b] += 1.0
        return X

    def fit(self, times: ArrayLike, marks: Optional[ArrayLike] = None) -> "HawkesL2ContrastEstimator":
        """Fit the L2-contrast estimator.

        Parameters
        ----------
        times : array-like
            Event times.
        marks : array-like or None
            If None, data are treated as univariate with all marks equal to 0.
        """
        if marks is None:
            t = _as_sorted_1d_times(times)
            k = np.zeros_like(t, dtype=int)
            D = 1 if self.n_dims is None else int(self.n_dims)
            if D != 1:
                raise ValueError("marks=None implies univariate data; n_dims must be None or 1.")
        else:
            t, k, D = _as_multivariate_events(times, marks, self.n_dims)

        edges = np.asarray(self.bin_edges, dtype=float)
        if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0) or edges[0] < 0:
            raise ValueError("bin_edges must be increasing and start at >= 0.")
        M = len(edges) - 1
        widths = np.diff(edges)
        T = float(t[-1] if self.T is None else self.T)
        if T < t[-1]:
            raise ValueError("T must be >= last event time.")

        # Grid for integral approximation. Avoid including t=0 exactly only for convenience.
        grid = np.linspace(0.0, T, self.grid_size + 1)[1:]
        dt_grid = T / self.grid_size
        X_grid = self._features_at(grid, t, k, D, edges)

        mu = np.zeros(D, dtype=float)
        c = np.zeros((D, D, M), dtype=float)

        # Fit each target dimension separately.
        for target in range(D):
            event_points = t[k == target]
            X_events = self._features_at(event_points, t, k, D, edges)

            # Flatten features: [intercept, X_{0,0}, ..., X_{D-1,M-1}].
            Phi_grid = np.c_[np.ones(len(grid)), X_grid.reshape(len(grid), D * M)]
            Phi_events = np.c_[np.ones(len(event_points)), X_events.reshape(len(event_points), D * M)]

            # Quadratic contrast C = theta^T G theta - 2 b^T theta + ridge.
            G = dt_grid * (Phi_grid.T @ Phi_grid)
            b = np.sum(Phi_events, axis=0)
            if self.ridge > 0:
                # Do not heavily penalize the intercept.
                R = np.eye(G.shape[0])
                R[0, 0] = 0.0
                G = G + self.ridge * R

            if not self.nonnegative:
                theta = solve(G, b, assume_a="sym")
            else:
                # Bound-constrained quadratic minimization. Bounds enforce mu>=0 and c>=0.
                def obj(theta: np.ndarray) -> float:
                    return float(theta @ G @ theta - 2.0 * b @ theta)

                def grad(theta: np.ndarray) -> np.ndarray:
                    return 2.0 * (G @ theta - b)

                x0 = np.maximum(solve(G + 1e-8 * np.eye(G.shape[0]), b, assume_a="sym"), 1e-12)
                res = minimize(obj, x0, jac=grad, method="L-BFGS-B", bounds=[(0.0, None)] * len(x0))
                if not res.success:
                    warnings.warn(f"L2 optimization did not fully converge for target {target}: {res.message}")
                theta = np.maximum(res.x, 0.0)

            mu[target] = theta[0]
            c[target] = theta[1:].reshape(D, M)

        if self.enforce_stationarity:
            Gbranch = np.sum(c * widths[None, None, :], axis=2)
            rho = _spectral_radius(Gbranch)
            if rho >= 0.999 and rho > 0:
                scale = 0.999 / rho
                c *= scale

        self.params_ = {
            "mu": mu,
            "c": c,
            "bin_edges": edges,
            "branching_matrix": np.sum(c * widths[None, None, :], axis=2),
        }
        return self


# ---------------------------------------------------------------------------
# Wiener-Hopf estimators
# ---------------------------------------------------------------------------


@dataclass
class UnivariateWienerHopfEstimator:
    """Univariate Wiener-Hopf estimator from second-order statistics.

    The theoretical equation is
        g(t) = phi(t) + integral_0^t phi(s) g(t-s) ds.

    We estimate g(t) by histogramming positive inter-event delays, then solve the
    Volterra equation recursively on a grid.

    Important
    ---------
    This is a simple educational discretization. In serious applications, the
    estimation of g(t) should include careful edge correction and smoothing.
    """

    max_lag: float
    n_bins: int
    T: Optional[float] = None
    clip_negative: bool = False
    enforce_stationarity: bool = True
    params_: Optional[Dict[str, np.ndarray | float]] = None

    def fit(self, times: ArrayLike) -> "UnivariateWienerHopfEstimator":
        times = _as_sorted_1d_times(times)
        T = float(times[-1] if self.T is None else self.T)
        if T < times[-1]:
            raise ValueError("T must be >= last event time.")
        n = len(times)
        Lambda = n / T
        edges = np.linspace(0.0, self.max_lag, self.n_bins + 1)
        dt = edges[1] - edges[0]
        centers = 0.5 * (edges[:-1] + edges[1:])

        # Estimate conditional excess intensity g(t).
        # For each parent event, count future events within max_lag.
        counts = np.zeros(self.n_bins, dtype=float)
        for j, tj in enumerate(times):
            future = times[j + 1:]
            delays = future - tj
            delays = delays[delays < self.max_lag]
            if delays.size:
                counts += np.histogram(delays, bins=edges)[0]

        # Conditional rate after an event minus baseline Lambda.
        # This ignores edge corrections near T; adequate for didactic use.
        g = counts / max(n * dt, 1e-12) - Lambda

        # Solve g_m = phi_m + dt * sum_{l=0}^{m-1} phi_l * g_{m-l}
        phi = np.zeros_like(g)
        for m in range(self.n_bins):
            conv = 0.0
            for l in range(m):
                conv += phi[l] * g[m - l]
            phi[m] = g[m] - dt * conv
            if self.clip_negative:
                phi[m] = max(phi[m], 0.0)

        mass = float(np.sum(phi) * dt)
        if self.enforce_stationarity and mass >= 0.999 and mass > 0:
            phi *= 0.999 / mass
            mass = float(np.sum(phi) * dt)

        mu = Lambda * (1.0 - mass)
        self.params_ = {
            "mu": float(mu),
            "phi": phi,
            "g": g,
            "lag_centers": centers,
            "bin_edges": edges,
            "kernel_mass": mass,
            "Lambda": float(Lambda),
        }
        return self


@dataclass
class MultivariateWienerHopfEstimator:
    """Multivariate Wiener-Hopf estimator from second-order statistics.

    The matrix equation is
        g(t) = Phi(t) + integral_0^t Phi(s) @ g(t-s) ds.

    The recursive discretization is
        Phi_m = g_m - dt * sum_{l=0}^{m-1} Phi_l @ g_{m-l}.
    """

    max_lag: float
    n_bins: int
    n_dims: Optional[int] = None
    T: Optional[float] = None
    clip_negative: bool = False
    enforce_stationarity: bool = True
    params_: Optional[Dict[str, np.ndarray]] = None

    def fit(self, times: ArrayLike, marks: ArrayLike) -> "MultivariateWienerHopfEstimator":
        times, marks, D = _as_multivariate_events(times, marks, self.n_dims)
        T = float(times[-1] if self.T is None else self.T)
        if T < times[-1]:
            raise ValueError("T must be >= last event time.")
        counts_by_dim = np.bincount(marks, minlength=D).astype(float)
        Lambda = counts_by_dim / T
        edges = np.linspace(0.0, self.max_lag, self.n_bins + 1)
        dt = edges[1] - edges[0]
        centers = 0.5 * (edges[:-1] + edges[1:])

        # g[m, i, j] estimates excess intensity of target i at lag m after source j.
        g = np.zeros((self.n_bins, D, D), dtype=float)
        for source in range(D):
            source_times = times[marks == source]
            n_source = len(source_times)
            if n_source == 0:
                continue
            for target in range(D):
                target_times = times[marks == target]
                hist_counts = np.zeros(self.n_bins, dtype=float)
                for tj in source_times:
                    delays = target_times[target_times > tj] - tj
                    delays = delays[delays < self.max_lag]
                    if delays.size:
                        hist_counts += np.histogram(delays, bins=edges)[0]
                g[:, target, source] = hist_counts / max(n_source * dt, 1e-12) - Lambda[target]

        Phi = np.zeros_like(g)
        for m in range(self.n_bins):
            conv = np.zeros((D, D), dtype=float)
            for l in range(m):
                conv += Phi[l] @ g[m - l]
            Phi[m] = g[m] - dt * conv
            if self.clip_negative:
                Phi[m] = np.maximum(Phi[m], 0.0)

        Gbranch = np.sum(Phi, axis=0) * dt
        if self.enforce_stationarity:
            rho = _spectral_radius(Gbranch)
            if rho >= 0.999 and rho > 0:
                scale = 0.999 / rho
                Phi *= scale
                Gbranch *= scale

        mu = (np.eye(D) - Gbranch) @ Lambda
        self.params_ = {
            "mu": mu,
            "Phi": Phi,
            "g": g,
            "lag_centers": centers,
            "bin_edges": edges,
            "branching_matrix": Gbranch,
            "Lambda": Lambda,
        }
        return self


# ---------------------------------------------------------------------------
# Simple Ogata simulators for testing the estimators
# ---------------------------------------------------------------------------


def simulate_univariate_exp_hawkes(mu: float, eta: float, beta: float, T: float, seed: Optional[int] = None) -> np.ndarray:
    """Simulate a univariate exponential Hawkes process by Ogata thinning.

    Model: lambda(t) = mu + sum eta*beta*exp(-beta*(t-tj)).
    """
    if mu <= 0 or eta < 0 or eta >= 1 or beta <= 0 or T <= 0:
        raise ValueError("require mu>0, eta in [0,1), beta>0, T>0.")
    rng = np.random.default_rng(seed)
    t = 0.0
    events: List[float] = []
    excitation = 0.0  # current sum eta*beta*exp(-beta lag)
    while t < T:
        lambda_bar = mu + excitation
        if lambda_bar <= 0:
            break
        w = rng.exponential(1.0 / lambda_bar)
        t_candidate = t + w
        if t_candidate > T:
            break
        # Decay excitation from t to t_candidate.
        excitation *= np.exp(-beta * (t_candidate - t))
        lambda_candidate = mu + excitation
        if rng.uniform() <= lambda_candidate / lambda_bar:
            events.append(t_candidate)
            excitation += eta * beta
        t = t_candidate
    return np.asarray(events, dtype=float)


def simulate_multivariate_exp_hawkes(mu: ArrayLike, eta: ArrayLike, beta: ArrayLike | float, T: float, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate a multivariate exponential Hawkes process by Ogata thinning.

    Parameters
    ----------
    mu : shape (D,)
    eta : shape (D,D), integrated kernel masses target x source
    beta : scalar or shape (D,D)
    T : horizon

    Returns
    -------
    times, marks
    """
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, dtype=float)
    eta = np.asarray(eta, dtype=float)
    D = len(mu)
    beta = np.asarray(beta, dtype=float)
    if beta.ndim == 0:
        beta = np.full((D, D), float(beta))
    if mu.shape != (D,) or eta.shape != (D, D) or beta.shape != (D, D):
        raise ValueError("inconsistent shapes.")
    if np.any(mu <= 0) or np.any(eta < 0) or np.any(beta <= 0) or T <= 0:
        raise ValueError("invalid parameters.")
    if _spectral_radius(eta) >= 1:
        raise ValueError("eta must have spectral radius < 1 for stable simulation.")

    t = 0.0
    times: List[float] = []
    marks: List[int] = []

    while t < T:
        # Compute current intensity by summing all past events. This is O(nD).
        lam = mu.copy()
        for tl, kl in zip(times, marks):
            dt = t - tl
            lam += eta[:, kl] * beta[:, kl] * np.exp(-beta[:, kl] * dt)
        lambda_bar = float(np.sum(lam))
        if lambda_bar <= 0:
            break
        w = rng.exponential(1.0 / lambda_bar)
        tc = t + w
        if tc > T:
            break
        lam_c = mu.copy()
        for tl, kl in zip(times, marks):
            dt = tc - tl
            lam_c += eta[:, kl] * beta[:, kl] * np.exp(-beta[:, kl] * dt)
        lambda_c = float(np.sum(lam_c))
        if rng.uniform() <= lambda_c / lambda_bar:
            probs = lam_c / lambda_c
            mark = int(rng.choice(D, p=probs))
            times.append(tc)
            marks.append(mark)
        t = tc

    return np.asarray(times, dtype=float), np.asarray(marks, dtype=int)


# ---------------------------------------------------------------------------
# End of module
# ---------------------------------------------------------------------------
