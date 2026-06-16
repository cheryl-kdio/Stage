"""Multivariate exponential Hawkes estimators.

Two estimators are provided:

- HawkesExpFixedDecayMLE: estimates baseline and adjacency with fixed decays.
- HawkesExpFreeDecayMLE: estimates baseline, adjacency and decays.

Both accept one realization:

    events = [array_dim_0, array_dim_1, ...]

or several independent realizations:

    events = [
        [array_dim_0, array_dim_1, ...],
        [array_dim_0, array_dim_1, ...],
    ]

The convention is alpha[i, j] = excitation from source j to target i.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import warnings
import numpy as np
from scipy.optimize import minimize


ArrayList = List[np.ndarray]
Realizations = List[ArrayList]


def _as_sorted_array(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    if np.any(~np.isfinite(arr)):
        raise ValueError("Event timestamps must be finite.")
    return np.sort(arr)


def _looks_like_single_realization(events) -> bool:
    if isinstance(events, np.ndarray):
        return True
    if not isinstance(events, (list, tuple)) or len(events) == 0:
        return True
    first = events[0]
    return not isinstance(first, (list, tuple))


def _prepare_realizations(events, end_times=None) -> Tuple[Realizations, np.ndarray, int]:
    if isinstance(events, np.ndarray):
        realizations = [[_as_sorted_array(events)]]
    elif isinstance(events, (list, tuple)) and len(events) > 0 and _looks_like_single_realization(events):
        if all(np.ndim(x) == 0 for x in events):
            realizations = [[_as_sorted_array(events)]]
        else:
            realizations = [[_as_sorted_array(x) for x in events]]
    elif isinstance(events, (list, tuple)):
        realizations = []
        for realization in events:
            if isinstance(realization, np.ndarray):
                realizations.append([_as_sorted_array(realization)])
            else:
                realizations.append([_as_sorted_array(x) for x in realization])
    else:
        realizations = [[_as_sorted_array(events)]]

    if not realizations:
        raise ValueError("At least one realization is required.")

    d = len(realizations[0])
    if d < 1:
        raise ValueError("Each realization must contain at least one dimension.")

    for r_idx, realization in enumerate(realizations):
        if len(realization) != d:
            raise ValueError(f"All realizations must have {d} dimensions; realization {r_idx} differs.")

    if end_times is None:
        Ts = []
        for realization in realizations:
            max_t = max((arr[-1] for arr in realization if arr.size > 0), default=None)
            if max_t is None:
                raise ValueError("end_times is required when a realization has no events.")
            Ts.append(float(max_t))
        warnings.warn(
            "end_times was not provided; using the last observed event time. "
            "For likelihood estimation, provide the true observation horizon.",
            RuntimeWarning,
        )
    elif np.ndim(end_times) == 0:
        Ts = [float(end_times)] * len(realizations)
    else:
        Ts = [float(x) for x in np.asarray(end_times, dtype=float).ravel()]
        if len(Ts) != len(realizations):
            raise ValueError("end_times must be a scalar or have length n_realizations.")

    Ts_arr = np.asarray(Ts, dtype=float)
    for r_idx, (realization, T) in enumerate(zip(realizations, Ts_arr)):
        if not np.isfinite(T) or T <= 0:
            raise ValueError("Every end_time must be finite and strictly positive.")
        for j, arr in enumerate(realization):
            if np.any(arr < 0):
                raise ValueError(f"Negative timestamp in realization {r_idx}, node {j}.")
            if np.any(arr > T):
                raise ValueError(f"Timestamp exceeds end_time in realization {r_idx}, node {j}.")
    return realizations, Ts_arr, d


def _normalize_matrix(x, d: int, name: str, positive: bool = True) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        out = np.full((d, d), float(arr))
    elif arr.shape == (d, d):
        out = arr.copy()
    else:
        raise ValueError(f"{name} must be a scalar or a matrix with shape {(d, d)}.")
    if positive and np.any(out <= 0):
        raise ValueError(f"{name} values must be strictly positive.")
    return out


def _stack_events(events_list: ArrayList) -> Tuple[np.ndarray, np.ndarray]:
    times = []
    types = []
    for j, arr in enumerate(events_list):
        times.extend(arr)
        types.extend([j] * len(arr))
    if len(times) == 0:
        return np.array([], dtype=float), np.array([], dtype=int)
    times_arr = np.asarray(times, dtype=float)
    types_arr = np.asarray(types, dtype=int)
    order = np.lexsort((types_arr, times_arr))
    return times_arr[order], types_arr[order]


def _kernel_integrals(events_list: ArrayList, T: float, beta: np.ndarray) -> np.ndarray:
    d = beta.shape[0]
    I = np.zeros((d, d), dtype=float)
    for j, arr in enumerate(events_list):
        if arr.size == 0:
            continue
        u = T - arr
        I[:, j] = np.sum(1.0 - np.exp(-beta[:, [j]] * u[None, :]), axis=1)
    return I


def _kernel_integrals_and_beta_grad(events_list: ArrayList, T: float, beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    d = beta.shape[0]
    I = np.zeros((d, d), dtype=float)
    J = np.zeros((d, d), dtype=float)
    for j, arr in enumerate(events_list):
        if arr.size == 0:
            continue
        u = T - arr
        exp_term = np.exp(-beta[:, [j]] * u[None, :])
        I[:, j] = np.sum(1.0 - exp_term, axis=1)
        J[:, j] = np.sum(u[None, :] * exp_term, axis=1)
    return I, J


@dataclass
class HawkesExpFixedDecayMLE:
    """MLE for multivariate exponential Hawkes with fixed decays."""

    decays: float | np.ndarray
    max_iter: int = 2000
    tol: float = 1e-8
    min_baseline: float = 1e-12
    alpha_upper: Optional[float] = None
    alpha_l2: float = 0.0

    @staticmethod
    def _unpack(theta: np.ndarray, d: int) -> Tuple[np.ndarray, np.ndarray]:
        mu = theta[:d]
        alpha = theta[d:].reshape(d, d)
        return mu, alpha

    def _nll_grad_one(self, theta: np.ndarray, events_list: ArrayList, T: float, beta: np.ndarray) -> Tuple[float, np.ndarray]:
        d = beta.shape[0]
        mu, alpha = self._unpack(theta, d)
        if np.any(mu <= 0) or np.any(alpha < 0):
            return np.inf, np.zeros_like(theta)

        times, types = _stack_events(events_list)
        I = _kernel_integrals(events_list, T, beta)

        ll = 0.0
        grad_mu = np.zeros(d, dtype=float)
        grad_alpha = np.zeros((d, d), dtype=float)
        r = np.zeros((d, d), dtype=float)
        last_t = 0.0
        k = 0
        n = len(times)
        while k < n:
            t = times[k]
            dt = t - last_t
            if dt > 0:
                r *= np.exp(-beta * dt)
            k2 = k + 1
            while k2 < n and times[k2] == t:
                k2 += 1
            m_vec = np.bincount(types[k:k2], minlength=d).astype(float)
            lambda_vec = mu + np.sum(alpha * r, axis=1)
            active = np.where(m_vec > 0)[0]
            if np.any(lambda_vec[active] <= 0) or np.any(~np.isfinite(lambda_vec[active])):
                return np.inf, np.zeros_like(theta)
            for i in active:
                inv_lam = 1.0 / lambda_vec[i]
                ll += m_vec[i] * np.log(lambda_vec[i])
                grad_mu[i] += m_vec[i] * inv_lam
                grad_alpha[i, :] += m_vec[i] * r[i, :] * inv_lam
            for j in np.where(m_vec > 0)[0]:
                r[:, j] += m_vec[j] * beta[:, j]
            last_t = t
            k = k2

        ll -= T * np.sum(mu) + np.sum(alpha * I)
        grad_mu -= T
        grad_alpha -= I

        nll = -ll
        grad_mu = -grad_mu
        grad_alpha = -grad_alpha
        if self.alpha_l2 > 0:
            nll += 0.5 * self.alpha_l2 * np.sum(alpha * alpha)
            grad_alpha += self.alpha_l2 * alpha
        grad = np.concatenate([grad_mu, grad_alpha.ravel()])
        return float(nll), grad

    def _nll_grad_all(self, theta: np.ndarray, realizations: Realizations, Ts: np.ndarray, beta: np.ndarray) -> Tuple[float, np.ndarray]:
        nll_total = 0.0
        grad_total = np.zeros_like(theta)
        for events_list, T in zip(realizations, Ts):
            nll, grad = self._nll_grad_one(theta, events_list, float(T), beta)
            if not np.isfinite(nll):
                return np.inf, np.zeros_like(theta)
            nll_total += nll
            grad_total += grad
        return float(nll_total), grad_total

    def fit(self, events, end_times=None, x0: Optional[np.ndarray] = None) -> "HawkesExpFixedDecayMLE":
        realizations, Ts, d = _prepare_realizations(events, end_times)
        beta = _normalize_matrix(self.decays, d, "decays", positive=True)

        if x0 is None:
            total_T = np.sum(Ts)
            counts = np.zeros(d)
            for realization in realizations:
                counts += np.array([len(arr) for arr in realization], dtype=float)
            mu0 = np.maximum(0.6 * counts / max(total_T, 1e-12), self.min_baseline * 10)
            alpha0 = np.full((d, d), 0.03 / max(d, 1), dtype=float)
            theta0 = np.concatenate([mu0, alpha0.ravel()])
        else:
            theta0 = np.asarray(x0, dtype=float).ravel()
            expected = d + d * d
            if theta0.size != expected:
                raise ValueError(f"x0 must have length {expected}.")

        bounds = [(self.min_baseline, None)] * d + [(0.0, self.alpha_upper)] * (d * d)
        result = minimize(
            lambda th: self._nll_grad_all(th, realizations, Ts, beta),
            x0=theta0,
            jac=True,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )
        self.result_ = result
        self.success_ = bool(result.success)
        self.message_ = result.message
        self.n_iter_ = result.nit
        self.n_nodes_ = d
        self.n_realizations_ = len(realizations)
        self.end_times_ = Ts.copy()
        self.events_ = realizations
        self.decays_ = beta
        self.baseline_, self.adjacency_ = self._unpack(result.x, d)
        self.log_likelihood_ = -float(result.fun)
        eigvals = np.linalg.eigvals(self.adjacency_)
        self.spectral_radius_ = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
        self.is_stable_ = bool(self.spectral_radius_ < 1.0)
        return self

    def score(self, events=None, end_times=None) -> float:
        if not hasattr(self, "baseline_"):
            raise RuntimeError("The model must be fitted first.")
        if events is None:
            realizations, Ts = self.events_, self.end_times_
        else:
            realizations, Ts, d = _prepare_realizations(events, end_times)
            if d != self.n_nodes_:
                raise ValueError(f"Expected {self.n_nodes_} nodes, got {d}.")
        theta = np.concatenate([self.baseline_, self.adjacency_.ravel()])
        nll, _ = self._nll_grad_all(theta, realizations, Ts, self.decays_)
        return -float(nll)

    def get_params(self) -> dict:
        if not hasattr(self, "baseline_"):
            raise RuntimeError("The model must be fitted first.")
        return {
            "baseline": self.baseline_.copy(),
            "adjacency": self.adjacency_.copy(),
            "decays": self.decays_.copy(),
            "log_likelihood": float(self.log_likelihood_),
            "spectral_radius": float(self.spectral_radius_),
            "is_stable": bool(self.is_stable_),
            "success": bool(self.success_),
        }

    @staticmethod
    def simulate(baseline, adjacency, decays, end_time: float, seed=None, max_events: int = 100_000) -> ArrayList:
        rng = np.random.default_rng(seed)
        mu = np.asarray(baseline, dtype=float).ravel()
        alpha = np.asarray(adjacency, dtype=float)
        d = len(mu)
        if alpha.shape != (d, d):
            raise ValueError(f"adjacency must have shape {(d, d)}.")
        beta = _normalize_matrix(decays, d, "decays", positive=True)
        if np.any(mu < 0) or np.any(alpha < 0):
            raise ValueError("baseline and adjacency must be non-negative.")
        T = float(end_time)
        events = [[] for _ in range(d)]
        r = np.zeros((d, d), dtype=float)
        t = 0.0
        while t < T and sum(len(x) for x in events) < max_events:
            lam = mu + np.sum(alpha * r, axis=1)
            lam_sum = float(np.sum(lam))
            if lam_sum <= 0:
                break
            t_candidate = t + rng.exponential(1.0 / lam_sum)
            if t_candidate > T:
                break
            dt = t_candidate - t
            r *= np.exp(-beta * dt)
            t = t_candidate
            lam_new = mu + np.sum(alpha * r, axis=1)
            lam_new_sum = float(np.sum(lam_new))
            if lam_new_sum > 0 and rng.uniform() <= lam_new_sum / lam_sum:
                node = rng.choice(d, p=lam_new / lam_new_sum)
                events[node].append(t)
                r[:, node] += beta[:, node]
        return [np.asarray(x, dtype=float) for x in events]


@dataclass
class HawkesExpFreeDecayMLE:
    """MLE for multivariate exponential Hawkes estimating baseline, adjacency and decays."""

    decays_init: float | np.ndarray = 1.0
    max_iter: int = 3000
    tol: float = 1e-8
    min_baseline: float = 1e-12
    min_decay: float = 1e-8
    alpha_upper: Optional[float] = None
    decay_upper: Optional[float] = None
    alpha_l2: float = 0.0
    n_starts: int = 3
    random_state: Optional[int] = None

    @staticmethod
    def _unpack(theta: np.ndarray, d: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu = theta[:d]
        alpha_start = d
        alpha_end = d + d * d
        beta_end = alpha_end + d * d
        alpha = theta[alpha_start:alpha_end].reshape(d, d)
        beta = theta[alpha_end:beta_end].reshape(d, d)
        return mu, alpha, beta

    def _nll_grad_one(self, theta: np.ndarray, events_list: ArrayList, T: float) -> Tuple[float, np.ndarray]:
        d = len(events_list)
        mu, alpha, beta = self._unpack(theta, d)
        if np.any(mu <= 0) or np.any(alpha < 0) or np.any(beta <= 0):
            return np.inf, np.zeros_like(theta)

        times, types = _stack_events(events_list)
        ll = 0.0
        grad_mu = np.zeros(d, dtype=float)
        grad_alpha = np.zeros((d, d), dtype=float)
        grad_beta = np.zeros((d, d), dtype=float)
        r = np.zeros((d, d), dtype=float)
        q = np.zeros((d, d), dtype=float)  # derivative of r wrt beta
        last_t = 0.0
        k = 0
        n = len(times)
        while k < n:
            t = times[k]
            dt = t - last_t
            if dt > 0:
                E = np.exp(-beta * dt)
                q = E * (q - dt * r)
                r = E * r
            k2 = k + 1
            while k2 < n and times[k2] == t:
                k2 += 1
            m_vec = np.bincount(types[k:k2], minlength=d).astype(float)
            lambda_vec = mu + np.sum(alpha * r, axis=1)
            active = np.where(m_vec > 0)[0]
            if np.any(lambda_vec[active] <= 0) or np.any(~np.isfinite(lambda_vec[active])):
                return np.inf, np.zeros_like(theta)
            for i in active:
                inv_lam = 1.0 / lambda_vec[i]
                ll += m_vec[i] * np.log(lambda_vec[i])
                grad_mu[i] += m_vec[i] * inv_lam
                grad_alpha[i, :] += m_vec[i] * r[i, :] * inv_lam
                grad_beta[i, :] += m_vec[i] * alpha[i, :] * q[i, :] * inv_lam
            for j in np.where(m_vec > 0)[0]:
                r[:, j] += m_vec[j] * beta[:, j]
                q[:, j] += m_vec[j]
            last_t = t
            k = k2

        I, J = _kernel_integrals_and_beta_grad(events_list, T, beta)
        ll -= T * np.sum(mu) + np.sum(alpha * I)
        grad_mu -= T
        grad_alpha -= I
        grad_beta -= alpha * J

        nll = -ll
        grad_mu = -grad_mu
        grad_alpha = -grad_alpha
        grad_beta = -grad_beta
        if self.alpha_l2 > 0:
            nll += 0.5 * self.alpha_l2 * np.sum(alpha * alpha)
            grad_alpha += self.alpha_l2 * alpha
        grad = np.concatenate([grad_mu, grad_alpha.ravel(), grad_beta.ravel()])
        return float(nll), grad

    def _nll_grad_all(self, theta: np.ndarray, realizations: Realizations, Ts: np.ndarray) -> Tuple[float, np.ndarray]:
        nll_total = 0.0
        grad_total = np.zeros_like(theta)
        for events_list, T in zip(realizations, Ts):
            nll, grad = self._nll_grad_one(theta, events_list, float(T))
            if not np.isfinite(nll):
                return np.inf, np.zeros_like(theta)
            nll_total += nll
            grad_total += grad
        return float(nll_total), grad_total

    def _initial_theta(self, realizations: Realizations, Ts: np.ndarray, d: int) -> np.ndarray:
        total_T = np.sum(Ts)
        counts = np.zeros(d)
        for realization in realizations:
            counts += np.array([len(arr) for arr in realization], dtype=float)
        mu0 = np.maximum(0.6 * counts / max(total_T, 1e-12), self.min_baseline * 10)
        alpha0 = np.full((d, d), 0.03 / max(d, 1), dtype=float)
        beta0 = _normalize_matrix(self.decays_init, d, "decays_init", positive=True)
        beta0 = np.maximum(beta0, self.min_decay * 10)
        if self.decay_upper is not None:
            beta0 = np.minimum(beta0, self.decay_upper * 0.8)
        return np.concatenate([mu0, alpha0.ravel(), beta0.ravel()])

    def fit(self, events, end_times=None, x0: Optional[np.ndarray] = None) -> "HawkesExpFreeDecayMLE":
        realizations, Ts, d = _prepare_realizations(events, end_times)
        expected = d + 2 * d * d
        if x0 is None:
            theta_base = self._initial_theta(realizations, Ts, d)
        else:
            theta_base = np.asarray(x0, dtype=float).ravel()
            if theta_base.size != expected:
                raise ValueError(f"x0 must have length {expected}.")

        bounds = [(self.min_baseline, None)] * d
        bounds += [(0.0, self.alpha_upper)] * (d * d)
        bounds += [(self.min_decay, self.decay_upper)] * (d * d)
        rng = np.random.default_rng(self.random_state)

        best_result = None
        best_fun = np.inf
        for start in range(max(1, int(self.n_starts))):
            if start == 0:
                theta0 = theta_base.copy()
            else:
                mu0, alpha0, beta0 = self._unpack(theta_base, d)
                theta0 = np.concatenate([
                    mu0 * rng.lognormal(0.0, 0.4, size=d),
                    alpha0.ravel() * rng.lognormal(0.0, 0.7, size=d * d),
                    beta0.ravel() * rng.lognormal(0.0, 0.7, size=d * d),
                ])
            for idx, (lo, hi) in enumerate(bounds):
                if lo is not None and theta0[idx] < lo:
                    theta0[idx] = lo * 10 if lo > 0 else lo
                if hi is not None and theta0[idx] > hi:
                    theta0[idx] = hi * 0.9

            result = minimize(
                lambda th: self._nll_grad_all(th, realizations, Ts),
                x0=theta0,
                jac=True,
                bounds=bounds,
                method="L-BFGS-B",
                options={"maxiter": self.max_iter, "ftol": self.tol},
            )
            if result.fun < best_fun:
                best_fun = float(result.fun)
                best_result = result

        self.result_ = best_result
        self.success_ = bool(best_result.success)
        self.message_ = best_result.message
        self.n_iter_ = best_result.nit
        self.n_nodes_ = d
        self.n_realizations_ = len(realizations)
        self.end_times_ = Ts.copy()
        self.events_ = realizations
        self.baseline_, self.adjacency_, self.decays_ = self._unpack(best_result.x, d)
        self.log_likelihood_ = -float(best_result.fun)
        eigvals = np.linalg.eigvals(self.adjacency_)
        self.spectral_radius_ = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
        self.is_stable_ = bool(self.spectral_radius_ < 1.0)
        return self

    def score(self, events=None, end_times=None) -> float:
        if not hasattr(self, "baseline_"):
            raise RuntimeError("The model must be fitted first.")
        if events is None:
            realizations, Ts = self.events_, self.end_times_
        else:
            realizations, Ts, d = _prepare_realizations(events, end_times)
            if d != self.n_nodes_:
                raise ValueError(f"Expected {self.n_nodes_} nodes, got {d}.")
        theta = np.concatenate([self.baseline_, self.adjacency_.ravel(), self.decays_.ravel()])
        nll, _ = self._nll_grad_all(theta, realizations, Ts)
        return -float(nll)

    def get_params(self) -> dict:
        if not hasattr(self, "baseline_"):
            raise RuntimeError("The model must be fitted first.")
        return {
            "baseline": self.baseline_.copy(),
            "adjacency": self.adjacency_.copy(),
            "decays": self.decays_.copy(),
            "log_likelihood": float(self.log_likelihood_),
            "spectral_radius": float(self.spectral_radius_),
            "is_stable": bool(self.is_stable_),
            "success": bool(self.success_),
        }

    @staticmethod
    def simulate(baseline, adjacency, decays, end_time: float, seed=None, max_events: int = 100_000) -> ArrayList:
        return HawkesExpFixedDecayMLE.simulate(baseline, adjacency, decays, end_time, seed, max_events)


import numpy as np
from scipy.optimize import minimize


class HawkesExpMarkedFixedDecayMLE:
    """
    Hawkes exponentiel multivarié marqué, avec decays fixés.

    Modèle :

        lambda_i(t) = mu_i
                      + sum_j sum_{t_k^j < t}
                        alpha_ij * exp(eta_j * z_k^j)
                        * beta_ij * exp(-beta_ij * (t - t_k^j))

    où z_k^j = standardisation de log(1 + mark_k^j) par dimension j.

    Convention :
        alpha_ij = excitation de la dimension j vers la dimension i.
        eta_j > 0 signifie que les gros marks de la dimension j
        augmentent l'excitation future.
    """

    def __init__(
        self,
        decays,
        estimate_mark_impact=True,
        max_iter=3000,
        tol=1e-8,
        min_baseline=1e-12,
        alpha_upper=None,
        eta_bounds=(-5.0, 5.0),
        alpha_l2=0.0,
        eta_l2=0.0,
        n_starts=1,
        random_state=None,
    ):
        self.decays_input = decays
        self.estimate_mark_impact = bool(estimate_mark_impact)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.min_baseline = float(min_baseline)
        self.alpha_upper = alpha_upper
        self.eta_bounds = eta_bounds
        self.alpha_l2 = float(alpha_l2)
        self.eta_l2 = float(eta_l2)
        self.n_starts = int(n_starts)
        self.random_state = random_state

    @staticmethod
    def _normalize_decays(decays, d):
        decays = np.asarray(decays, dtype=float)

        if decays.ndim == 0:
            beta = np.full((d, d), float(decays))
        elif decays.shape == (d, d):
            beta = decays.copy()
        else:
            raise ValueError(f"decays doit être un scalaire ou une matrice {(d, d)}.")

        if np.any(beta <= 0):
            raise ValueError("Toutes les valeurs de decays doivent être strictement positives.")

        return beta

    @staticmethod
    def _prepare_one_dim(times, marks):
        times = np.asarray(times, dtype=float).ravel()

        if marks is None:
            marks = np.ones_like(times)
        else:
            marks = np.asarray(marks, dtype=float).ravel()

        if times.shape != marks.shape:
            raise ValueError("Chaque tableau de marks doit avoir la même taille que les timestamps.")

        if np.any(~np.isfinite(times)) or np.any(~np.isfinite(marks)):
            raise ValueError("timestamps et marks doivent être finis.")

        if np.any(marks < 0):
            raise ValueError("Les marks doivent être positifs ou nuls.")

        order = np.argsort(times)
        return times[order], marks[order]

    def _prepare_events_marks(self, events, marks=None):
        """
        Format attendu :
            events = [array_dim_0, array_dim_1, ..., array_dim_{d-1}]
            marks  = [marks_dim_0, marks_dim_1, ..., marks_dim_{d-1}]

        Pour univarié :
            events = np.array([...])
            marks  = np.array([...])
        """
        if isinstance(events, np.ndarray):
            t, m = self._prepare_one_dim(events, marks)
            return [t], [m]

        if not isinstance(events, (list, tuple)):
            t, m = self._prepare_one_dim(events, marks)
            return [t], [m]

        if len(events) == 0:
            raise ValueError("events ne peut pas être vide.")

        if all(np.ndim(x) == 0 for x in events):
            t, m = self._prepare_one_dim(events, marks)
            return [t], [m]

        if marks is None:
            marks_iter = [None] * len(events)
        else:
            if len(marks) != len(events):
                raise ValueError("marks doit avoir la même structure que events.")
            marks_iter = marks

        out_events = []
        out_marks = []

        for t_raw, m_raw in zip(events, marks_iter):
            t, m = self._prepare_one_dim(t_raw, m_raw)
            out_events.append(t)
            out_marks.append(m)

        return out_events, out_marks

    @staticmethod
    def _standardize_marks(mark_list):
        """
        z = standardize(log(1 + mark)) par dimension.
        """
        z_list = []

        for marks in mark_list:
            x = np.log1p(marks)

            if len(x) == 0:
                z_list.append(x.astype(float))
                continue

            mean = np.mean(x)
            std = np.std(x)

            if std <= 1e-12:
                std = 1.0

            z_list.append((x - mean) / std)

        return z_list

    @staticmethod
    def _stack(events, z_list):
        times = []
        types = []
        z_values = []

        for j, (arr, z) in enumerate(zip(events, z_list)):
            if len(arr) != len(z):
                raise ValueError("Les marks transformés doivent avoir la même taille que les timestamps.")

            times.extend(arr)
            types.extend([j] * len(arr))
            z_values.extend(z)

        if len(times) == 0:
            return np.array([]), np.array([], dtype=int), np.array([])

        times = np.asarray(times, dtype=float)
        types = np.asarray(types, dtype=int)
        z_values = np.asarray(z_values, dtype=float)

        order = np.lexsort((types, times))
        return times[order], types[order], z_values[order]

    @staticmethod
    def _unpack(theta, d, estimate_mark_impact):
        mu = theta[:d]
        alpha = theta[d:d + d * d].reshape(d, d)

        if estimate_mark_impact:
            eta = theta[d + d * d:d + d * d + d]
        else:
            eta = np.zeros(d, dtype=float)

        return mu, alpha, eta

    def _kernel_integrals(self, events, z_list, T, beta, eta):
        """
        S_ij = sum_{events source j} w_k * (1 - exp(-beta_ij * (T - t_k)))

        K_ij = dS_ij / d eta_j
             = sum w_k * z_k * (1 - exp(-beta_ij * (T - t_k)))
        """
        d = len(events)
        S = np.zeros((d, d), dtype=float)
        K = np.zeros((d, d), dtype=float)

        for j, arr in enumerate(events):
            if len(arr) == 0:
                continue

            z = z_list[j]
            w = np.exp(eta[j] * z)
            remaining = T - arr

            exp_term = np.exp(-beta[:, [j]] * remaining[None, :])

            S[:, j] = np.sum(w[None, :] * (1.0 - exp_term), axis=1)
            K[:, j] = np.sum(w[None, :] * z[None, :] * (1.0 - exp_term), axis=1)

        return S, K

    def _neg_loglik_grad(self, theta, events, z_list, T, beta):
        d = len(events)
        mu, alpha, eta = self._unpack(theta, d, self.estimate_mark_impact)

        if np.any(mu <= 0) or np.any(alpha < 0):
            return np.inf, np.zeros_like(theta)

        times, types, z_values = self._stack(events, z_list)

        ll = 0.0
        grad_mu = np.zeros(d, dtype=float)
        grad_alpha = np.zeros((d, d), dtype=float)
        grad_eta = np.zeros(d, dtype=float)

        # r_ij(t) = beta_ij * sum_{events source j before t}
        #           exp(eta_j z_k) exp(-beta_ij (t - t_k))
        r = np.zeros((d, d), dtype=float)

        # u_ij(t) = d r_ij(t) / d eta_j
        u = np.zeros((d, d), dtype=float)

        last_t = 0.0
        n = len(times)
        k = 0

        while k < n:
            t = times[k]
            dt = t - last_t

            if dt < -1e-12:
                raise RuntimeError("Les timestamps doivent être triés.")

            if dt > 0:
                decay_factor = np.exp(-beta * dt)
                r *= decay_factor
                u *= decay_factor

            # Groupe de timestamps égaux :
            # les événements simultanés ne s'excitent pas entre eux.
            k2 = k + 1
            while k2 < n and times[k2] == t:
                k2 += 1

            group_types = types[k:k2]
            group_z = z_values[k:k2]

            counts = np.bincount(group_types, minlength=d).astype(float)

            lambda_vec = mu + np.sum(alpha * r, axis=1)

            active = np.where(counts > 0)[0]

            if np.any(lambda_vec[active] <= 0) or np.any(~np.isfinite(lambda_vec[active])):
                return np.inf, np.zeros_like(theta)

            for i in active:
                c = counts[i]
                inv_lam = 1.0 / lambda_vec[i]

                ll += c * np.log(lambda_vec[i])
                grad_mu[i] += c * inv_lam
                grad_alpha[i, :] += c * r[i, :] * inv_lam

                if self.estimate_mark_impact:
                    grad_eta += c * alpha[i, :] * u[i, :] * inv_lam

            # Sauts après évaluation de lambda(t)
            weights = np.exp(eta[group_types] * group_z)

            for j in range(d):
                mask = group_types == j

                if not np.any(mask):
                    continue

                w_sum = np.sum(weights[mask])
                wz_sum = np.sum(weights[mask] * group_z[mask])

                r[:, j] += beta[:, j] * w_sum
                u[:, j] += beta[:, j] * wz_sum

            last_t = t
            k = k2

        S, K = self._kernel_integrals(events, z_list, T, beta, eta)

        # Compensateur
        ll -= T * np.sum(mu) + np.sum(alpha * S)

        grad_mu -= T
        grad_alpha -= S

        if self.estimate_mark_impact:
            grad_eta -= np.sum(alpha * K, axis=0)

        nll = -ll
        grad_mu = -grad_mu
        grad_alpha = -grad_alpha
        grad_eta = -grad_eta

        if self.alpha_l2 > 0:
            nll += 0.5 * self.alpha_l2 * np.sum(alpha * alpha)
            grad_alpha += self.alpha_l2 * alpha

        if self.estimate_mark_impact and self.eta_l2 > 0:
            nll += 0.5 * self.eta_l2 * np.sum(eta * eta)
            grad_eta += self.eta_l2 * eta

        if self.estimate_mark_impact:
            grad = np.concatenate([grad_mu, grad_alpha.ravel(), grad_eta])
        else:
            grad = np.concatenate([grad_mu, grad_alpha.ravel()])

        return nll, grad

    def fit(self, events, marks=None, end_time=None, x0=None):
        events, marks = self._prepare_events_marks(events, marks)
        d = len(events)

        if end_time is None:
            max_t = max((x[-1] for x in events if len(x) > 0), default=None)

            if max_t is None:
                raise ValueError("end_time est requis lorsqu'il n'y a aucun événement.")

            T = float(max_t)
        else:
            T = float(end_time)

        if T <= 0:
            raise ValueError("end_time doit être strictement positif.")

        for j, arr in enumerate(events):
            if np.any(arr < 0):
                raise ValueError(f"events[{j}] contient des timestamps négatifs.")

            if np.any(arr > T):
                raise ValueError(f"events[{j}] contient des timestamps supérieurs à end_time.")

        beta = self._normalize_decays(self.decays_input, d)
        z_list = self._standardize_marks(marks)

        counts = np.array([len(x) for x in events], dtype=float)

        if x0 is None:
            mu0 = np.maximum(0.5 * counts / T, self.min_baseline * 10)
            alpha0 = np.full((d, d), 0.05 / max(d, 1), dtype=float)

            if self.estimate_mark_impact:
                eta0 = np.zeros(d, dtype=float)
                theta0 = np.concatenate([mu0, alpha0.ravel(), eta0])
            else:
                theta0 = np.concatenate([mu0, alpha0.ravel()])
        else:
            theta0 = np.asarray(x0, dtype=float).ravel()
            expected = d + d * d + (d if self.estimate_mark_impact else 0)

            if theta0.size != expected:
                raise ValueError(f"x0 doit avoir une longueur {expected}.")

        bounds = [(self.min_baseline, None)] * d
        bounds += [(0.0, self.alpha_upper)] * (d * d)

        if self.estimate_mark_impact:
            eta_low, eta_high = self.eta_bounds
            bounds += [(eta_low, eta_high)] * d

        rng = np.random.default_rng(self.random_state)

        best_result = None
        best_fun = np.inf

        for start in range(max(1, self.n_starts)):
            if start == 0:
                start_theta = theta0.copy()
            else:
                mu, alpha, eta = self._unpack(theta0, d, self.estimate_mark_impact)

                mu_s = mu * rng.lognormal(0.0, 0.4, size=d)
                alpha_s = alpha * rng.lognormal(0.0, 0.7, size=(d, d))

                if self.estimate_mark_impact:
                    eta_s = eta + rng.normal(0.0, 0.3, size=d)
                    start_theta = np.concatenate([mu_s, alpha_s.ravel(), eta_s])
                else:
                    start_theta = np.concatenate([mu_s, alpha_s.ravel()])

            for idx, (lo, hi) in enumerate(bounds):
                if lo is not None and start_theta[idx] < lo:
                    start_theta[idx] = lo * 10 if lo > 0 else lo

                if hi is not None and start_theta[idx] > hi:
                    start_theta[idx] = hi * 0.9 if hi > 0 else hi

            result = minimize(
                fun=lambda th: self._neg_loglik_grad(th, events, z_list, T, beta),
                x0=start_theta,
                jac=True,
                bounds=bounds,
                method="L-BFGS-B",
                options={
                    "maxiter": self.max_iter,
                    "ftol": self.tol,
                },
            )

            if result.fun < best_fun:
                best_result = result
                best_fun = float(result.fun)

        self.result_ = best_result
        self.success_ = bool(best_result.success)
        self.message_ = best_result.message
        self.n_iter_ = best_result.nit

        self.n_nodes_ = d
        self.end_time_ = T
        self.events_ = events
        self.marks_ = marks
        self.z_marks_ = z_list
        self.decays_ = beta

        self.baseline_, self.adjacency_, self.mark_eta_ = self._unpack(
            best_result.x,
            d,
            self.estimate_mark_impact,
        )

        self.log_likelihood_ = -float(best_result.fun)

        eigvals = np.linalg.eigvals(self.adjacency_)
        self.spectral_radius_ = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
        self.is_stable_ = bool(self.spectral_radius_ < 1.0)

        return self

    def score(self, events=None, marks=None, end_time=None):
        if not hasattr(self, "baseline_"):
            raise RuntimeError("Le modèle doit être fitté avant score().")

        if events is None:
            events = self.events_
            z_list = self.z_marks_
            T = self.end_time_
        else:
            events, marks = self._prepare_events_marks(events, marks)
            z_list = self._standardize_marks(marks)
            T = self.end_time_ if end_time is None else float(end_time)

        if self.estimate_mark_impact:
            theta = np.concatenate(
                [
                    self.baseline_,
                    self.adjacency_.ravel(),
                    self.mark_eta_,
                ]
            )
        else:
            theta = np.concatenate(
                [
                    self.baseline_,
                    self.adjacency_.ravel(),
                ]
            )

        nll, _ = self._neg_loglik_grad(
            theta,
            events,
            z_list,
            T,
            self.decays_,
        )

        return -float(nll)

    def get_params(self):
        if not hasattr(self, "baseline_"):
            raise RuntimeError("Le modèle doit être fitté avant get_params().")

        return {
            "baseline": self.baseline_.copy(),
            "adjacency": self.adjacency_.copy(),
            "decays": self.decays_.copy(),
            "mark_eta": self.mark_eta_.copy(),
            "log_likelihood": self.log_likelihood_,
            "spectral_radius": self.spectral_radius_,
            "is_stable": self.is_stable_,
        }
    


import numpy as np
from scipy.optimize import minimize


class HawkesExpMarkedFreeDecayMLE:
    """
    Hawkes exponentiel multivarié marqué avec decays libres.

    Modèle :

        lambda_i(t) = mu_i
                      + sum_j sum_{t_k^j < t}
                        alpha_ij
                        * exp(eta_j * z_k^j)
                        * beta_ij
                        * exp(-beta_ij * (t - t_k^j))

    où :

        z_k^j = standardisation de log(1 + mark_k^j)
                pour la dimension source j.

    Paramètres estimés :

        mu_i     >= 0
        alpha_ij >= 0
        beta_ij  > 0
        eta_j    borné

    Convention :

        alpha_ij = excitation de la dimension j vers la dimension i.
        beta_ij  = decay de l'effet j -> i.
        eta_j    = effet du mark de la dimension source j.

    Interprétation :

        eta_j > 0 :
            les gros marks de la dimension j augmentent l'excitation future.

        eta_j < 0 :
            les gros marks de la dimension j réduisent l'excitation future.

        eta_j proche de 0 :
            le mark de la dimension j n'a pas d'effet clair.
    """

    def __init__(
        self,
        decays_init=1.0,
        estimate_mark_impact=True,
        max_iter=3000,
        tol=1e-8,
        min_baseline=1e-12,
        min_decay=1e-8,
        alpha_upper=None,
        decay_upper=None,
        eta_bounds=(-5.0, 5.0),
        alpha_l2=0.0,
        beta_l2=0.0,
        eta_l2=0.0,
        n_starts=1,
        random_state=None,
    ):
        self.decays_init = decays_init
        self.estimate_mark_impact = bool(estimate_mark_impact)

        self.max_iter = int(max_iter)
        self.tol = float(tol)

        self.min_baseline = float(min_baseline)
        self.min_decay = float(min_decay)

        self.alpha_upper = alpha_upper
        self.decay_upper = decay_upper
        self.eta_bounds = eta_bounds

        self.alpha_l2 = float(alpha_l2)
        self.beta_l2 = float(beta_l2)
        self.eta_l2 = float(eta_l2)

        self.n_starts = int(n_starts)
        self.random_state = random_state

    @staticmethod
    def _normalize_matrix(x, d, name):
        x = np.asarray(x, dtype=float)

        if x.ndim == 0:
            out = np.full((d, d), float(x))
        elif x.shape == (d, d):
            out = x.copy()
        else:
            raise ValueError(
                f"{name} doit être un scalaire ou une matrice de forme {(d, d)}."
            )

        return out

    @staticmethod
    def _prepare_one_dim(times, marks):
        times = np.asarray(times, dtype=float).ravel()

        if marks is None:
            marks = np.ones_like(times)
        else:
            marks = np.asarray(marks, dtype=float).ravel()

        if times.shape != marks.shape:
            raise ValueError(
                "Chaque tableau de marks doit avoir la même taille que les timestamps."
            )

        if np.any(~np.isfinite(times)) or np.any(~np.isfinite(marks)):
            raise ValueError("timestamps et marks doivent être finis.")

        if np.any(marks < 0):
            raise ValueError("Les marks doivent être positifs ou nuls.")

        order = np.argsort(times)

        return times[order], marks[order]

    def _prepare_events_marks(self, events, marks=None):
        """
        Formats acceptés.

        Univarié :

            events = np.array([t1, t2, ...])
            marks  = np.array([v1, v2, ...])

        Multivarié :

            events = [
                np.array([...]),  # dimension 0
                np.array([...]),  # dimension 1
            ]

            marks = [
                np.array([...]),  # marks dimension 0
                np.array([...]),  # marks dimension 1
            ]

        Si marks=None, tous les marks sont mis à 1.
        """

        if isinstance(events, np.ndarray):
            t, m = self._prepare_one_dim(events, marks)
            return [t], [m]

        if not isinstance(events, (list, tuple)):
            t, m = self._prepare_one_dim(events, marks)
            return [t], [m]

        if len(events) == 0:
            raise ValueError("events ne peut pas être vide.")

        # Cas liste de scalaires : univarié.
        if all(np.ndim(x) == 0 for x in events):
            t, m = self._prepare_one_dim(events, marks)
            return [t], [m]

        if marks is None:
            marks_iter = [None] * len(events)
        else:
            if len(marks) != len(events):
                raise ValueError("marks doit avoir la même structure que events.")
            marks_iter = marks

        out_events = []
        out_marks = []

        for t_raw, m_raw in zip(events, marks_iter):
            t, m = self._prepare_one_dim(t_raw, m_raw)
            out_events.append(t)
            out_marks.append(m)

        return out_events, out_marks

    @staticmethod
    def _fit_mark_standardization(mark_list):
        """
        Fit de la standardisation :

            x = log(1 + mark)
            z = (x - mean) / std

        par dimension source.
        """
        z_list = []
        stats = []

        for marks in mark_list:
            x = np.log1p(marks)

            if len(x) == 0:
                mean = 0.0
                std = 1.0
                z = x.astype(float)
            else:
                mean = float(np.mean(x))
                std = float(np.std(x))

                if std <= 1e-12:
                    std = 1.0

                z = (x - mean) / std

            z_list.append(z)
            stats.append((mean, std))

        return z_list, stats

    @staticmethod
    def _transform_marks(mark_list, stats):
        z_list = []

        if len(mark_list) != len(stats):
            raise ValueError("Nombre de dimensions incompatible avec les stats de marks.")

        for marks, (mean, std) in zip(mark_list, stats):
            x = np.log1p(marks)
            z = (x - mean) / std
            z_list.append(z)

        return z_list

    @staticmethod
    def _stack(events, z_list):
        times = []
        types = []
        z_values = []

        for j, (arr, z) in enumerate(zip(events, z_list)):
            if len(arr) != len(z):
                raise ValueError(
                    "Les marks transformés doivent avoir la même taille que les timestamps."
                )

            times.extend(arr)
            types.extend([j] * len(arr))
            z_values.extend(z)

        if len(times) == 0:
            return (
                np.array([], dtype=float),
                np.array([], dtype=int),
                np.array([], dtype=float),
            )

        times = np.asarray(times, dtype=float)
        types = np.asarray(types, dtype=int)
        z_values = np.asarray(z_values, dtype=float)

        order = np.lexsort((types, times))

        return times[order], types[order], z_values[order]

    @staticmethod
    def _unpack(theta, d, estimate_mark_impact=True):
        mu = theta[:d]

        alpha_start = d
        alpha_end = d + d * d

        beta_start = alpha_end
        beta_end = beta_start + d * d

        alpha = theta[alpha_start:alpha_end].reshape(d, d)
        beta = theta[beta_start:beta_end].reshape(d, d)

        if estimate_mark_impact:
            eta = theta[beta_end:beta_end + d]
        else:
            eta = np.zeros(d, dtype=float)

        return mu, alpha, beta, eta

    def _kernel_integrals_and_grads(self, events, z_list, T, beta, eta):
        """
        Compensateur :

            I_ij = sum_{t_k^j <= T}
                   w_k^j * (1 - exp(-beta_ij * (T - t_k^j)))

        avec :

            w_k^j = exp(eta_j * z_k^j)

        Dérivées :

            dI_ij / d beta_ij
            =
            sum w_k^j * (T - t_k^j) * exp(-beta_ij * (T - t_k^j))

            dI_ij / d eta_j
            =
            sum w_k^j * z_k^j * (1 - exp(-beta_ij * (T - t_k^j)))
        """
        d = len(events)

        I = np.zeros((d, d), dtype=float)
        J_beta = np.zeros((d, d), dtype=float)
        K_eta = np.zeros((d, d), dtype=float)

        for j, arr in enumerate(events):
            if len(arr) == 0:
                continue

            z = z_list[j]
            w = np.exp(eta[j] * z)

            remaining = T - arr

            exp_term = np.exp(-beta[:, [j]] * remaining[None, :])

            I[:, j] = np.sum(
                w[None, :] * (1.0 - exp_term),
                axis=1,
            )

            J_beta[:, j] = np.sum(
                w[None, :] * remaining[None, :] * exp_term,
                axis=1,
            )

            K_eta[:, j] = np.sum(
                w[None, :] * z[None, :] * (1.0 - exp_term),
                axis=1,
            )

        return I, J_beta, K_eta

    def _neg_loglik_grad(self, theta, events, z_list, T):
        d = len(events)

        mu, alpha, beta, eta = self._unpack(
            theta,
            d,
            self.estimate_mark_impact,
        )

        if np.any(mu <= 0) or np.any(alpha < 0) or np.any(beta <= 0):
            return np.inf, np.zeros_like(theta)

        times, types, z_values = self._stack(events, z_list)

        ll = 0.0

        grad_mu = np.zeros(d, dtype=float)
        grad_alpha = np.zeros((d, d), dtype=float)
        grad_beta = np.zeros((d, d), dtype=float)
        grad_eta = np.zeros(d, dtype=float)

        # r_ij(t) = beta_ij * sum_{events source j before t}
        #           exp(eta_j z_k)
        #           exp(-beta_ij (t - t_k))
        #
        # q_ij(t) = d r_ij(t) / d beta_ij
        #
        # u_ij(t) = d r_ij(t) / d eta_j
        r = np.zeros((d, d), dtype=float)
        q = np.zeros((d, d), dtype=float)
        u = np.zeros((d, d), dtype=float)

        last_t = 0.0
        n = len(times)
        k = 0

        while k < n:
            t = times[k]
            dt = t - last_t

            if dt < -1e-12:
                raise RuntimeError("Les timestamps doivent être triés.")

            if dt > 0:
                decay_factor = np.exp(-beta * dt)

                # Attention :
                # q doit utiliser l'ancien r.
                q = decay_factor * (q - dt * r)
                r = decay_factor * r
                u = decay_factor * u

            # Groupe de timestamps égaux.
            # Les événements simultanés ne s'excitent pas entre eux.
            k2 = k + 1
            while k2 < n and times[k2] == t:
                k2 += 1

            group_types = types[k:k2]
            group_z = z_values[k:k2]

            counts = np.bincount(group_types, minlength=d).astype(float)

            lambda_vec = mu + np.sum(alpha * r, axis=1)

            active = np.where(counts > 0)[0]

            if np.any(lambda_vec[active] <= 0) or np.any(~np.isfinite(lambda_vec[active])):
                return np.inf, np.zeros_like(theta)

            for i in active:
                c = counts[i]
                inv_lam = 1.0 / lambda_vec[i]

                ll += c * np.log(lambda_vec[i])

                grad_mu[i] += c * inv_lam
                grad_alpha[i, :] += c * r[i, :] * inv_lam
                grad_beta[i, :] += c * alpha[i, :] * q[i, :] * inv_lam

                if self.estimate_mark_impact:
                    grad_eta += c * alpha[i, :] * u[i, :] * inv_lam

            # Sauts après évaluation de lambda(t).
            weights = np.exp(eta[group_types] * group_z)

            for j in range(d):
                mask = group_types == j

                if not np.any(mask):
                    continue

                w_sum = np.sum(weights[mask])
                wz_sum = np.sum(weights[mask] * group_z[mask])

                r[:, j] += beta[:, j] * w_sum

                # Dérivée du saut beta * w par rapport à beta.
                q[:, j] += w_sum

                # Dérivée du saut beta * w par rapport à eta_j.
                u[:, j] += beta[:, j] * wz_sum

            last_t = t
            k = k2

        I, J_beta, K_eta = self._kernel_integrals_and_grads(
            events,
            z_list,
            T,
            beta,
            eta,
        )

        # Compensateur :
        #
        # int_0^T lambda_i(s) ds
        # =
        # mu_i T + sum_j alpha_ij I_ij
        ll -= T * np.sum(mu) + np.sum(alpha * I)

        grad_mu -= T
        grad_alpha -= I
        grad_beta -= alpha * J_beta

        if self.estimate_mark_impact:
            grad_eta -= np.sum(alpha * K_eta, axis=0)

        nll = -ll

        grad_mu = -grad_mu
        grad_alpha = -grad_alpha
        grad_beta = -grad_beta
        grad_eta = -grad_eta

        if self.alpha_l2 > 0:
            nll += 0.5 * self.alpha_l2 * np.sum(alpha * alpha)
            grad_alpha += self.alpha_l2 * alpha

        if self.beta_l2 > 0:
            nll += 0.5 * self.beta_l2 * np.sum(beta * beta)
            grad_beta += self.beta_l2 * beta

        if self.estimate_mark_impact and self.eta_l2 > 0:
            nll += 0.5 * self.eta_l2 * np.sum(eta * eta)
            grad_eta += self.eta_l2 * eta

        if self.estimate_mark_impact:
            grad = np.concatenate(
                [
                    grad_mu,
                    grad_alpha.ravel(),
                    grad_beta.ravel(),
                    grad_eta,
                ]
            )
        else:
            grad = np.concatenate(
                [
                    grad_mu,
                    grad_alpha.ravel(),
                    grad_beta.ravel(),
                ]
            )

        return nll, grad

    def _initial_theta(self, events, T, d):
        counts = np.array([len(x) for x in events], dtype=float)

        mu0 = np.maximum(
            0.5 * counts / max(T, 1e-12),
            self.min_baseline * 10,
        )

        alpha0 = np.full((d, d), 0.05 / max(d, 1), dtype=float)

        beta0 = self._normalize_matrix(
            self.decays_init,
            d,
            "decays_init",
        )

        beta0 = np.maximum(beta0, self.min_decay * 10)

        if self.decay_upper is not None:
            beta0 = np.minimum(beta0, self.decay_upper * 0.9)

        if self.estimate_mark_impact:
            eta0 = np.zeros(d, dtype=float)

            theta0 = np.concatenate(
                [
                    mu0,
                    alpha0.ravel(),
                    beta0.ravel(),
                    eta0,
                ]
            )
        else:
            theta0 = np.concatenate(
                [
                    mu0,
                    alpha0.ravel(),
                    beta0.ravel(),
                ]
            )

        return theta0

    def fit(self, events, marks=None, end_time=None, x0=None):
        events, marks = self._prepare_events_marks(events, marks)
        d = len(events)

        if end_time is None:
            max_t = max((x[-1] for x in events if len(x) > 0), default=None)

            if max_t is None:
                raise ValueError("end_time est requis lorsqu'il n'y a aucun événement.")

            T = float(max_t)
        else:
            T = float(end_time)

        if T <= 0 or not np.isfinite(T):
            raise ValueError("end_time doit être strictement positif et fini.")

        for j, arr in enumerate(events):
            if np.any(arr < 0):
                raise ValueError(f"events[{j}] contient des timestamps négatifs.")

            if np.any(arr > T):
                raise ValueError(
                    f"events[{j}] contient des timestamps supérieurs à end_time."
                )

        z_list, mark_stats = self._fit_mark_standardization(marks)

        if x0 is None:
            theta0 = self._initial_theta(events, T, d)
        else:
            theta0 = np.asarray(x0, dtype=float).ravel()

            expected = d + d * d + d * d + (d if self.estimate_mark_impact else 0)

            if theta0.size != expected:
                raise ValueError(f"x0 doit avoir une longueur {expected}.")

        bounds = [(self.min_baseline, None)] * d
        bounds += [(0.0, self.alpha_upper)] * (d * d)
        bounds += [(self.min_decay, self.decay_upper)] * (d * d)

        if self.estimate_mark_impact:
            eta_low, eta_high = self.eta_bounds
            bounds += [(eta_low, eta_high)] * d

        rng = np.random.default_rng(self.random_state)

        best_result = None
        best_fun = np.inf

        for start in range(max(1, self.n_starts)):
            if start == 0:
                start_theta = theta0.copy()
            else:
                mu0, alpha0, beta0, eta0 = self._unpack(
                    theta0,
                    d,
                    self.estimate_mark_impact,
                )

                mu_s = mu0 * rng.lognormal(
                    mean=0.0,
                    sigma=0.4,
                    size=d,
                )

                alpha_s = alpha0 * rng.lognormal(
                    mean=0.0,
                    sigma=0.7,
                    size=(d, d),
                )

                beta_s = beta0 * rng.lognormal(
                    mean=0.0,
                    sigma=0.7,
                    size=(d, d),
                )

                if self.estimate_mark_impact:
                    eta_s = eta0 + rng.normal(
                        loc=0.0,
                        scale=0.4,
                        size=d,
                    )

                    start_theta = np.concatenate(
                        [
                            mu_s,
                            alpha_s.ravel(),
                            beta_s.ravel(),
                            eta_s,
                        ]
                    )
                else:
                    start_theta = np.concatenate(
                        [
                            mu_s,
                            alpha_s.ravel(),
                            beta_s.ravel(),
                        ]
                    )

            # Projection simple de l'initialisation dans les bornes.
            for idx, (lo, hi) in enumerate(bounds):
                if lo is not None and start_theta[idx] < lo:
                    if lo > 0:
                        start_theta[idx] = lo * 10.0
                    else:
                        start_theta[idx] = lo

                if hi is not None and start_theta[idx] > hi:
                    if hi > 0:
                        start_theta[idx] = hi * 0.9
                    else:
                        start_theta[idx] = hi

            result = minimize(
                fun=lambda th: self._neg_loglik_grad(th, events, z_list, T),
                x0=start_theta,
                jac=True,
                bounds=bounds,
                method="L-BFGS-B",
                options={
                    "maxiter": self.max_iter,
                    "ftol": self.tol,
                },
            )

            if result.fun < best_fun:
                best_fun = float(result.fun)
                best_result = result

        self.result_ = best_result
        self.success_ = bool(best_result.success)
        self.message_ = best_result.message
        self.n_iter_ = best_result.nit

        self.n_nodes_ = d
        self.end_time_ = T

        self.events_ = events
        self.marks_ = marks
        self.z_marks_ = z_list
        self.mark_stats_ = mark_stats

        self.baseline_, self.adjacency_, self.decays_, self.mark_eta_ = self._unpack(
            best_result.x,
            d,
            self.estimate_mark_impact,
        )

        self.log_likelihood_ = -float(best_result.fun)

        # Pour un Hawkes marqué, la matrice de branchement effective empirique
        # est alpha_ij * E[exp(eta_j z_j)] pour chaque source j.
        mark_mean_weights = np.ones(d, dtype=float)

        for j, z in enumerate(self.z_marks_):
            if len(z) > 0:
                mark_mean_weights[j] = float(np.mean(np.exp(self.mark_eta_[j] * z)))

        self.mark_mean_weights_ = mark_mean_weights
        self.branching_matrix_ = self.adjacency_ * mark_mean_weights[None, :]

        eigvals = np.linalg.eigvals(self.branching_matrix_)
        self.spectral_radius_ = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
        self.is_stable_ = bool(self.spectral_radius_ < 1.0)

        return self

    def score(self, events=None, marks=None, end_time=None):
        if not hasattr(self, "baseline_"):
            raise RuntimeError("Le modèle doit être fitté avant score().")

        if events is None:
            events = self.events_
            z_list = self.z_marks_
            T = self.end_time_
        else:
            events, marks = self._prepare_events_marks(events, marks)

            if len(events) != self.n_nodes_:
                raise ValueError(
                    f"Le modèle attend {self.n_nodes_} dimensions, reçu {len(events)}."
                )

            z_list = self._transform_marks(marks, self.mark_stats_)
            T = self.end_time_ if end_time is None else float(end_time)

        if self.estimate_mark_impact:
            theta = np.concatenate(
                [
                    self.baseline_,
                    self.adjacency_.ravel(),
                    self.decays_.ravel(),
                    self.mark_eta_,
                ]
            )
        else:
            theta = np.concatenate(
                [
                    self.baseline_,
                    self.adjacency_.ravel(),
                    self.decays_.ravel(),
                ]
            )

        nll, _ = self._neg_loglik_grad(theta, events, z_list, T)

        return -float(nll)

    def get_params(self):
        if not hasattr(self, "baseline_"):
            raise RuntimeError("Le modèle doit être fitté avant get_params().")

        return {
            "baseline": self.baseline_.copy(),
            "adjacency": self.adjacency_.copy(),
            "decays": self.decays_.copy(),
            "mark_eta": self.mark_eta_.copy(),
            "mark_mean_weights": self.mark_mean_weights_.copy(),
            "branching_matrix": self.branching_matrix_.copy(),
            "log_likelihood": self.log_likelihood_,
            "spectral_radius": self.spectral_radius_,
            "is_stable": self.is_stable_,
        }