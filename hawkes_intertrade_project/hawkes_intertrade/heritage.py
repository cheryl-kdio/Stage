"""
# Utilisations
model = UnivariateHawkesMLE()
model.fit(events, end_times=T)

fig, ax = model.plot_N_vs_Lambda()
fig, ax = model.plot_martingale()
fig, ax = model.plot_residuals_qq()
fig, ax = model.plot_uniform_residuals()

tests = model.residual_ks_tests()
print(tests)

# =================================
model = UnivariateMarkedAmplitudeHawkesMLE()
model.fit(events, marks=marks, end_times=T)

fig, ax = model.plot_N_vs_Lambda()
fig, ax = model.plot_martingale()
fig, ax = model.plot_residuals_qq()

print(model.residual_ks_tests())

# =================================
model = UnivariateZDecayHawkesMLE()
model.fit(events, z=z, end_times=T)

fig, ax = model.plot_N_vs_Lambda()
fig, ax = model.plot_martingale()
fig, ax = model.plot_residuals_qq()

print(model.residual_ks_tests())
"""

import numpy as np
import warnings
from scipy.optimize import minimize
from scipy.stats import kstest

class BaseUnivariateHawkesMLE:
    param_names = ("mu", "alpha", "beta")

    def __init__(
        self,
        beta_init=1.0,
        max_iter=3000,
        tol=1e-8,
        min_baseline=1e-12,
        min_decay=1e-8,
        alpha_upper=None,
        beta_upper=None,
        alpha_l2=0.0,
        beta_l2=0.0,
        eta_l2=0.0,
        n_starts=1,
        random_state=None,
    ):
        self.beta_init = float(beta_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.min_baseline = float(min_baseline)
        self.min_decay = float(min_decay)
        self.alpha_upper = alpha_upper
        self.beta_upper = beta_upper
        self.alpha_l2 = float(alpha_l2)
        self.beta_l2 = float(beta_l2)
        self.eta_l2 = float(eta_l2)
        self.n_starts = int(n_starts)
        self.random_state = random_state

        if self.beta_init <= 0:
            raise ValueError("beta_init doit être strictement positif.")

    @staticmethod
    def _as_1d_float(x, name):
        arr = np.asarray(x, dtype=float).ravel()
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"{name} doit contenir des valeurs finies.")
        return arr

    def _prepare_one_realization(
        self,
        times,
        extra=None,
        extra_name=None,
        extra_default=1.0,
        positive_extra=False,
    ):
        times = self._as_1d_float(times, "times")

        if extra_name is None:
            order = np.argsort(times)
            return times[order]

        if extra is None:
            extra_arr = np.full_like(times, float(extra_default), dtype=float)
        else:
            extra_arr = self._as_1d_float(extra, extra_name)

        if times.shape != extra_arr.shape:
            raise ValueError(f"times et {extra_name} doivent avoir la même longueur.")

        if positive_extra and np.any(extra_arr <= 0):
            raise ValueError(f"Tous les {extra_name} doivent être strictement positifs.")

        order = np.argsort(times)
        return times[order], extra_arr[order]

    def _prepare_realizations(
        self,
        events,
        end_times=None,
        extra=None,
        extra_name=None,
        extra_default=1.0,
        positive_extra=False,
    ):
        has_extra = extra_name is not None

        realizations = []
        extra_realizations = [] if has_extra else None

        if isinstance(events, np.ndarray):
            if has_extra:
                t, x = self._prepare_one_realization(
                    events,
                    extra,
                    extra_name,
                    extra_default,
                    positive_extra,
                )
                realizations, extra_realizations = [t], [x]
            else:
                realizations = [self._prepare_one_realization(events)]

        elif isinstance(events, (list, tuple)):
            if len(events) == 0:
                raise ValueError("events ne peut pas être vide.")

            if all(np.ndim(x) == 0 for x in events):
                if has_extra:
                    t, x = self._prepare_one_realization(
                        events,
                        extra,
                        extra_name,
                        extra_default,
                        positive_extra,
                    )
                    realizations, extra_realizations = [t], [x]
                else:
                    realizations = [self._prepare_one_realization(events)]

            else:
                if has_extra:
                    if extra is None:
                        extra_iter = [None] * len(events)
                    else:
                        if len(extra) != len(events):
                            raise ValueError(
                                f"Pour plusieurs réalisations, {extra_name} doit avoir "
                                "la même longueur que events."
                            )
                        extra_iter = extra

                    for ev, ex in zip(events, extra_iter):
                        t, x = self._prepare_one_realization(
                            ev,
                            ex,
                            extra_name,
                            extra_default,
                            positive_extra,
                        )
                        realizations.append(t)
                        extra_realizations.append(x)

                else:
                    realizations = [
                        self._prepare_one_realization(ev)
                        for ev in events
                    ]

        else:
            if has_extra:
                t, x = self._prepare_one_realization(
                    events,
                    extra,
                    extra_name,
                    extra_default,
                    positive_extra,
                )
                realizations, extra_realizations = [t], [x]
            else:
                realizations = [self._prepare_one_realization(events)]

        end_times = self._prepare_end_times(realizations, end_times)

        if has_extra:
            return realizations, extra_realizations, end_times

        return realizations, end_times

    @staticmethod
    def _prepare_end_times(realizations, end_times):
        if end_times is None:
            Ts = []

            for t in realizations:
                if len(t) == 0:
                    raise ValueError(
                        "end_times est requis si une réalisation est vide."
                    )

                Ts.append(float(t[-1]))

            warnings.warn(
                "end_times non fourni : utilisation du dernier timestamp. "
                "Pour une MLE correcte, fournissez l'horizon réel d'observation.",
                RuntimeWarning,
            )

        elif np.ndim(end_times) == 0:
            Ts = [float(end_times)] * len(realizations)

        else:
            Ts = [
                float(x)
                for x in np.asarray(end_times, dtype=float).ravel()
            ]

            if len(Ts) != len(realizations):
                raise ValueError(
                    "end_times doit être scalaire ou de longueur n_realizations."
                )

        for idx, (t, T) in enumerate(zip(realizations, Ts)):
            if T <= 0 or not np.isfinite(T):
                raise ValueError(
                    "Chaque end_time doit être strictement positif et fini."
                )

            if np.any(t < 0):
                raise ValueError(f"Réalisation {idx}: timestamps négatifs.")

            if np.any(t > T):
                raise ValueError(
                    f"Réalisation {idx}: timestamps au-delà de end_time."
                )

        return np.asarray(Ts, dtype=float)

    def _prepare_fit_data(self, events, end_times=None, **kwargs):
        realizations, end_times = self._prepare_realizations(
            events,
            end_times=end_times,
        )
        return realizations, None, end_times

    def _prepare_score_data(self, events, end_times=None, **kwargs):
        return self._prepare_fit_data(
            events,
            end_times=end_times,
            **kwargs,
        )

    def _payload_at(self, payload, idx):
        return None if payload is None else payload[idx]

    def _unpack(self, theta):
        return {
            name: float(value)
            for name, value in zip(self.param_names, theta)
        }

    def _bounds(self):
        return [
            (self.min_baseline, None),
            (0.0, self.alpha_upper),
            (self.min_decay, self.beta_upper),
        ]

    def _initial_theta(self, realizations, end_times, payload=None):
        total_events = sum(len(x) for x in realizations)
        total_T = float(np.sum(end_times))

        mu0 = max(
            0.5 * total_events / max(total_T, 1e-12),
            self.min_baseline * 10,
        )

        alpha0 = 0.05
        beta0 = max(self.beta_init, self.min_decay * 10)

        if self.beta_upper is not None:
            beta0 = min(beta0, self.beta_upper * 0.9)

        return np.array([mu0, alpha0, beta0], dtype=float)

    def _random_start(self, theta0, rng):
        theta = theta0.copy()

        for i, name in enumerate(self.param_names):
            if name == "mu":
                theta[i] *= rng.lognormal(0.0, 0.4)

            elif name in ("alpha", "beta"):
                theta[i] *= rng.lognormal(0.0, 0.7)

            elif name == "eta":
                theta[i] += rng.normal(0.0, 0.4)

        return theta

    @staticmethod
    def _project_into_bounds(theta, bounds):
        theta = theta.copy()

        for idx, (lo, hi) in enumerate(bounds):
            if lo is not None and theta[idx] < lo:
                theta[idx] = lo * 10.0 if lo > 0 else lo

            if hi is not None and theta[idx] > hi:
                theta[idx] = hi * 0.9 if hi > 0 else hi

        return theta

    def _nll_grad_one(self, theta, times, payload, T):
        raise NotImplementedError

    def _nll_grad_all(self, theta, realizations, payload, end_times):
        nll_total = 0.0
        grad_total = np.zeros_like(theta, dtype=float)

        for idx, (times, T) in enumerate(zip(realizations, end_times)):
            nll, grad = self._nll_grad_one(
                theta,
                times,
                self._payload_at(payload, idx),
                float(T),
            )

            if not np.isfinite(nll):
                return np.inf, np.zeros_like(theta)

            nll_total += nll
            grad_total += grad

        return float(nll_total), grad_total

    def _add_l2(self, nll, grad, params):
        if self.alpha_l2 > 0:
            nll += 0.5 * self.alpha_l2 * params["alpha"] ** 2
            grad[self.param_names.index("alpha")] += self.alpha_l2 * params["alpha"]

        if self.beta_l2 > 0:
            nll += 0.5 * self.beta_l2 * params["beta"] ** 2
            grad[self.param_names.index("beta")] += self.beta_l2 * params["beta"]

        if "eta" in params and self.eta_l2 > 0:
            nll += 0.5 * self.eta_l2 * params["eta"] ** 2
            grad[self.param_names.index("eta")] += self.eta_l2 * params["eta"]

        return nll, grad

    def fit(self, events, end_times=None, x0=None, **kwargs):
        realizations, payload, end_times = self._prepare_fit_data(
            events,
            end_times=end_times,
            **kwargs,
        )

        if x0 is None:
            theta0 = self._initial_theta(realizations, end_times, payload)
        else:
            theta0 = np.asarray(x0, dtype=float).ravel()

            if theta0.size != len(self.param_names):
                raise ValueError(
                    f"x0 doit avoir une longueur {len(self.param_names)} : "
                    f"{list(self.param_names)}."
                )

        bounds = self._bounds()
        rng = np.random.default_rng(self.random_state)

        best_result = None
        best_fun = np.inf

        for start in range(max(1, self.n_starts)):
            if start == 0:
                start_theta = theta0.copy()
            else:
                start_theta = self._random_start(theta0, rng)

            start_theta = self._project_into_bounds(start_theta, bounds)

            result = minimize(
                fun=lambda th: self._nll_grad_all(
                    th,
                    realizations,
                    payload,
                    end_times,
                ),
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

        self.events_ = realizations
        self.end_times_ = end_times
        self._payload_ = payload

        params = self._unpack(best_result.x)

        self.baseline_ = params["mu"]

        for name, value in params.items():
            setattr(self, f"{name}_", value)

        self.log_likelihood_ = -float(best_result.fun)

        self._post_fit(payload)

        return self

    def _post_fit(self, payload):
        self.branching_ratio_ = float(self.alpha_)
        self.is_stable_ = bool(self.branching_ratio_ < 1.0)

    def score(self, events=None, end_times=None, **kwargs):
        if not hasattr(self, "baseline_"):
            raise RuntimeError("Le modèle doit être fitté avant score().")

        if events is None:
            realizations = self.events_
            payload = self._payload_
            end_times = self.end_times_
        else:
            realizations, payload, end_times = self._prepare_score_data(
                events,
                end_times=end_times,
                **kwargs,
            )

        theta = np.array(
            [
                getattr(self, f"{name}_")
                for name in self.param_names
            ],
            dtype=float,
        )

        nll, _ = self._nll_grad_all(
            theta,
            realizations,
            payload,
            end_times,
        )

        return -float(nll)

    def get_params(self):
        if not hasattr(self, "baseline_"):
            raise RuntimeError("Le modèle doit être fitté avant get_params().")

        out = {
            "baseline": self.baseline_,
        }

        for name in self.param_names:
            if name != "mu":
                out[name] = getattr(self, f"{name}_")

        out.update(
            {
                "log_likelihood": self.log_likelihood_,
                "success": self.success_,
                "message": self.message_,
                "n_iter": self.n_iter_,
            }
        )

        if hasattr(self, "branching_ratio_"):
            out["branching_ratio"] = self.branching_ratio_
            out["is_stable"] = self.is_stable_

        if hasattr(self, "branching_ratio_empirical_"):
            out["mean_z"] = self.mean_z_
            out["branching_ratio_empirical"] = self.branching_ratio_empirical_
            out["is_stable_empirical"] = self.is_stable_empirical_

        if hasattr(self, "mark_stats_"):
            out["mark_stats"] = dict(self.mark_stats_)
            out["mean_mark_weight"] = self.mean_mark_weight_

        return out
    
    def _check_fitted(self):
        if not hasattr(self, "baseline_"):
            raise RuntimeError("Le modèle doit être fitté avant validation.")

    def _kernel_cumulative_at(self, t, times, payload):
        """
        À redéfinir dans chaque classe fille.

        Doit retourner la partie noyau du compensateur :

            sum_k contribution_k(0,t)

        de sorte que :

            Lambda(t)=mu*t+alpha*_kernel_cumulative_at(t,...)
        """
        raise NotImplementedError

    def cumulative_intensity_at(self, times_eval=None, realization_idx=0):
        """
        Calcule Lambda(t)=int_0^t lambda(u)du.

        Paramètres
        ----------
        times_eval : None, float ou array-like
            Temps auxquels évaluer le compensateur.
            Si None, on l'évalue aux timestamps observés.
        realization_idx : int
            Indice de la réalisation.

        Retour
        ------
        Lambda : float ou np.ndarray
            Intensité cumulée estimée.
        """
        self._check_fitted()

        times = self.events_[realization_idx]
        payload = self._payload_at(self._payload_, realization_idx)

        scalar_input = np.ndim(times_eval) == 0 and times_eval is not None

        if times_eval is None:
            grid = times.copy()
        elif scalar_input:
            grid = np.array([float(times_eval)], dtype=float)
        else:
            grid = np.asarray(times_eval, dtype=float).ravel()

        if np.any(~np.isfinite(grid)):
            raise ValueError("times_eval doit contenir des valeurs finies.")
        if np.any(grid < 0):
            raise ValueError("times_eval doit être positif.")

        Lambda = np.empty(len(grid), dtype=float)

        for i, t in enumerate(grid):
            Lambda[i] = (
                self.baseline_ * t
                + self.alpha_ * self._kernel_cumulative_at(t, times, payload)
            )

        return float(Lambda[0]) if scalar_input else Lambda

    def counting_process_at(self, times_eval=None, realization_idx=0):
        """
        Calcule N(t), le nombre d'événements observés jusqu'à t inclus.
        """
        self._check_fitted()

        times = self.events_[realization_idx]

        scalar_input = np.ndim(times_eval) == 0 and times_eval is not None

        if times_eval is None:
            grid = times.copy()
        elif scalar_input:
            grid = np.array([float(times_eval)], dtype=float)
        else:
            grid = np.asarray(times_eval, dtype=float).ravel()

        if np.any(~np.isfinite(grid)):
            raise ValueError("times_eval doit contenir des valeurs finies.")
        if np.any(grid < 0):
            raise ValueError("times_eval doit être positif.")

        N = np.searchsorted(times, grid, side="right").astype(float)

        return float(N[0]) if scalar_input else N

    def validation_processes(self, realization_idx=0, grid=None, n_grid=1000):
        """
        Retourne les processus nécessaires aux diagnostics :

            N(t)
            Lambda(t)
            M(t)=N(t)-Lambda(t)

        Utile pour tracer N_t versus intensité cumulée.
        """
        self._check_fitted()

        times = self.events_[realization_idx]
        T = self.end_times_[realization_idx]

        if grid is None:
            grid = np.unique(
                np.r_[
                    np.linspace(0.0, T, int(n_grid)),
                    times,
                ]
            )
        else:
            grid = np.asarray(grid, dtype=float).ravel()

        N = self.counting_process_at(grid, realization_idx=realization_idx)
        Lambda = self.cumulative_intensity_at(grid, realization_idx=realization_idx)
        M = N - Lambda

        return {
            "time": grid,
            "N": N,
            "Lambda": Lambda,
            "M": M,
        }

    def martingale_at_events(self, realization_idx=0):
        """
        Calcule la martingale compensée aux dates d'événements :

            M(t_i)=N(t_i)-Lambda(t_i)=i-Lambda(t_i)

        Retourne aussi les incréments compensés :

            1 - (Lambda(t_i)-Lambda(t_{i-1}))
        """
        self._check_fitted()

        times = self.events_[realization_idx]

        if len(times) == 0:
            return {
                "event_times": times,
                "Lambda_events": np.array([]),
                "M_events": np.array([]),
                "tau": np.array([]),
                "martingale_increments": np.array([]),
            }

        if np.any(np.diff(times) == 0):
            warnings.warn(
                "Présence de timestamps égaux. Les résidus de time-rescaling "
                "sont théoriquement valides pour un processus ponctuel simple "
                "sans événements simultanés.",
                RuntimeWarning,
            )

        Lambda_events = self.cumulative_intensity_at(
            times,
            realization_idx=realization_idx,
        )

        N_events = np.arange(1, len(times) + 1, dtype=float)
        M_events = N_events - Lambda_events

        tau = np.diff(np.r_[0.0, Lambda_events])
        martingale_increments = np.ones_like(tau) - tau

        return {
            "event_times": times,
            "Lambda_events": Lambda_events,
            "M_events": M_events,
            "tau": tau,
            "martingale_increments": martingale_increments,
        }

    def time_rescaling_residuals(self, realization_idx=None):
        """
        Résidus de time-rescaling.

        Si le modèle est bien spécifié :

            tau_i = Lambda(t_i)-Lambda(t_{i-1}) ~ Exp(1)

        et

            U_i = 1-exp(-tau_i) ~ Uniform(0,1)

        Si realization_idx=None, concatène toutes les réalisations.
        """
        self._check_fitted()

        if realization_idx is None:
            indices = range(len(self.events_))
        else:
            indices = [int(realization_idx)]

        all_tau = []
        all_uniform = []
        all_realization = []
        all_event_index = []

        for idx in indices:
            out = self.martingale_at_events(realization_idx=idx)
            tau = out["tau"]

            if len(tau) == 0:
                continue

            uniform = 1.0 - np.exp(-tau)

            all_tau.append(tau)
            all_uniform.append(uniform)
            all_realization.append(np.full(len(tau), idx, dtype=int))
            all_event_index.append(np.arange(1, len(tau) + 1, dtype=int))

        if len(all_tau) == 0:
            return {
                "tau": np.array([]),
                "uniform": np.array([]),
                "realization": np.array([]),
                "event_index": np.array([]),
            }

        return {
            "tau": np.concatenate(all_tau),
            "uniform": np.concatenate(all_uniform),
            "realization": np.concatenate(all_realization),
            "event_index": np.concatenate(all_event_index),
        }

    def residual_ks_tests(self, realization_idx=None):
        """
        Tests de Kolmogorov-Smirnov sur les résidus.

        Test 1 :
            tau_i contre Exp(1)

        Test 2 :
            U_i=1-exp(-tau_i) contre Uniform(0,1)
        """
        residuals = self.time_rescaling_residuals(
            realization_idx=realization_idx,
        )

        tau = residuals["tau"]
        u = residuals["uniform"]

        mask = np.isfinite(tau) & np.isfinite(u) & (tau >= 0.0) & (u >= 0.0) & (u <= 1.0)

        tau = tau[mask]
        u = u[mask]

        if len(tau) == 0:
            raise ValueError("Aucun résidu valide pour effectuer les tests.")

        ks_exp = kstest(tau, "expon")
        ks_uniform = kstest(u, "uniform")

        return {
            "n_residuals": int(len(tau)),
            "exp_ks_stat": float(ks_exp.statistic),
            "exp_pvalue": float(ks_exp.pvalue),
            "uniform_ks_stat": float(ks_uniform.statistic),
            "uniform_pvalue": float(ks_uniform.pvalue),
        }

    def plot_N_vs_Lambda(self, realization_idx=0, grid=None, n_grid=1000):
        """
        Trace N(t) et Lambda(t) sur le même graphique.
        """
        import matplotlib.pyplot as plt

        d = self.validation_processes(
            realization_idx=realization_idx,
            grid=grid,
            n_grid=n_grid,
        )

        fig, ax = plt.subplots()

        ax.step(
            d["time"],
            d["N"],
            where="post",
            label="N(t)",
        )

        ax.plot(
            d["time"],
            d["Lambda"],
            label=r"$\hat{\Lambda}(t)$",
        )

        ax.set_xlabel("t")
        ax.set_ylabel("Valeur cumulée")
        ax.set_title(r"Processus de comptage $N(t)$ vs compensateur $\hat{\Lambda}(t)$")
        ax.legend()
        ax.grid(True)

        return fig, ax

    def plot_martingale(self, realization_idx=0, grid=None, n_grid=1000):
        """
        Trace M(t)=N(t)-Lambda(t).

        Sous bonne spécification, M(t) doit fluctuer autour de 0,
        sans tendance systématique.
        """
        import matplotlib.pyplot as plt

        d = self.validation_processes(
            realization_idx=realization_idx,
            grid=grid,
            n_grid=n_grid,
        )

        fig, ax = plt.subplots()

        ax.step(
            d["time"],
            d["M"],
            where="post",
            label=r"$M(t)=N(t)-\hat{\Lambda}(t)$",
        )

        ax.axhline(0.0, linestyle="--")
        ax.set_xlabel("t")
        ax.set_ylabel("Martingale compensée")
        ax.set_title(r"Diagnostic martingale $M(t)$")
        ax.legend()
        ax.grid(True)

        return fig, ax

    def plot_residuals_qq(self, realization_idx=None):
        """
        QQ-plot des résidus tau_i contre une loi Exp(1).
        """
        import matplotlib.pyplot as plt

        residuals = self.time_rescaling_residuals(
            realization_idx=realization_idx,
        )

        tau = residuals["tau"]
        tau = tau[np.isfinite(tau) & (tau >= 0.0)]

        if len(tau) == 0:
            raise ValueError("Aucun résidu valide à tracer.")

        tau_sorted = np.sort(tau)
        n = len(tau_sorted)

        probs = (np.arange(1, n + 1) - 0.5) / n
        exp_quantiles = -np.log(1.0 - probs)

        fig, ax = plt.subplots()

        ax.scatter(exp_quantiles, tau_sorted)

        lim_min = min(float(np.min(exp_quantiles)), float(np.min(tau_sorted)))
        lim_max = max(float(np.max(exp_quantiles)), float(np.max(tau_sorted)))

        ax.plot(
            [lim_min, lim_max],
            [lim_min, lim_max],
            linestyle="--",
        )

        ax.set_xlabel("Quantiles théoriques Exp(1)")
        ax.set_ylabel("Résidus observés")
        ax.set_title("QQ-plot des résidus de time-rescaling")
        ax.grid(True)

        return fig, ax

    def plot_uniform_residuals(self, realization_idx=None):
        """
        Trace les résidus uniformisés :

            U_i = 1 - exp(-tau_i)

        Sous bonne spécification, ils doivent ressembler à des Uniform(0,1).
        """
        import matplotlib.pyplot as plt

        residuals = self.time_rescaling_residuals(
            realization_idx=realization_idx,
        )

        u = residuals["uniform"]
        u = u[np.isfinite(u) & (u >= 0.0) & (u <= 1.0)]

        if len(u) == 0:
            raise ValueError("Aucun résidu uniformisé valide à tracer.")

        u_sorted = np.sort(u)
        n = len(u_sorted)
        empirical = np.arange(1, n + 1) / n

        fig, ax = plt.subplots()

        ax.step(
            u_sorted,
            empirical,
            where="post",
            label="CDF empirique",
        )

        ax.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            linestyle="--",
            label="CDF Uniform(0,1)",
        )

        ax.set_xlabel(r"$U_i=1-\exp(-\tau_i)$")
        ax.set_ylabel("CDF")
        ax.set_title("Résidus uniformisés")
        ax.legend()
        ax.grid(True)

        return fig, ax

class UnivariateHawkesMLE(BaseUnivariateHawkesMLE):
    """
    Hawkes exponentiel classique.

    lambda(t)=mu+sum_{t_k<t} alpha*beta*exp(-beta*(t-t_k))

    Paramètres estimés :
        mu
        alpha
        beta
    """

    @staticmethod
    def _kernel_integral_and_grad(times, T, beta):
        if len(times) == 0:
            return 0.0, 0.0

        rem = T - times
        e = np.exp(-beta * rem)

        S = np.sum(1.0 - e)
        S_beta = np.sum(rem * e)

        return float(S), float(S_beta)

    def _nll_grad_one(self, theta, times, payload, T):
        p = self._unpack(theta)

        mu = p["mu"]
        alpha = p["alpha"]
        beta = p["beta"]

        if mu <= 0 or alpha < 0 or beta <= 0:
            return np.inf, np.zeros_like(theta)

        ll = 0.0

        grad_mu = 0.0
        grad_alpha = 0.0
        grad_beta = 0.0

        r = 0.0
        q = 0.0

        last_t = 0.0
        n = len(times)
        k = 0

        while k < n:
            t = times[k]
            dt = t - last_t

            if dt < -1e-12:
                raise RuntimeError("Les timestamps doivent être triés.")

            if dt > 0:
                e = np.exp(-beta * dt)

                q = e * (q - dt * r)
                r = e * r

            k2 = k + 1

            while k2 < n and times[k2] == t:
                k2 += 1

            count = k2 - k

            lam = mu + alpha * r

            if lam <= 0 or not np.isfinite(lam):
                return np.inf, np.zeros_like(theta)

            inv_lam = 1.0 / lam

            ll += count * np.log(lam)

            grad_mu += count * inv_lam
            grad_alpha += count * r * inv_lam
            grad_beta += count * alpha * q * inv_lam

            r += beta * count
            q += count

            last_t = t
            k = k2

        S, S_beta = self._kernel_integral_and_grad(times, T, beta)

        ll -= mu * T + alpha * S

        grad_mu -= T
        grad_alpha -= S
        grad_beta -= alpha * S_beta

        nll = -ll

        grad = np.array(
            [
                -grad_mu,
                -grad_alpha,
                -grad_beta,
            ],
            dtype=float,
        )

        return self._add_l2(nll, grad, p)

    def intensity_at_events(self, events=None):
        if not hasattr(self, "baseline_"):
            raise RuntimeError(
                "Le modèle doit être fitté avant intensity_at_events()."
            )

        if events is None:
            times = self.events_[0]
        else:
            times = self._prepare_one_realization(events)

        intensities = np.zeros(len(times), dtype=float)

        r = 0.0
        last_t = 0.0

        n = len(times)
        k = 0

        while k < n:
            t = times[k]
            dt = t - last_t

            if dt > 0:
                r *= np.exp(-self.beta_ * dt)

            k2 = k + 1

            while k2 < n and times[k2] == t:
                k2 += 1

            intensities[k:k2] = self.baseline_ + self.alpha_ * r

            r += self.beta_ * (k2 - k)

            last_t = t
            k = k2

        return intensities
    
    def _kernel_cumulative_at(self, t, times, payload):
        if len(times) == 0 or t <= 0:
            return 0.0

        past = times[times < t]

        if len(past) == 0:
            return 0.0

        return float(
            np.sum(
                1.0 - np.exp(-self.beta_ * (t - past))
            )
        )

class UnivariateMarkedAmplitudeHawkesMLE(BaseUnivariateHawkesMLE):
    """
    Hawkes avec mark sur l'amplitude.

    lambda(t)=mu+sum_{t_k<t} alpha*exp(eta*z_k)*beta*exp(-beta*(t-t_k))

    Les marks fournis sont standardisés :

        z_k = (mark_k - mean) / std

    Paramètres estimés :
        mu
        alpha
        beta
        eta
    """

    param_names = ("mu", "alpha", "beta", "eta")

    def __init__(
        self,
        beta_init=1.0,
        max_iter=3000,
        tol=1e-8,
        min_baseline=1e-12,
        min_decay=1e-8,
        alpha_upper=None,
        beta_upper=None,
        eta_bounds=(-5.0, 5.0),
        alpha_l2=0.0,
        beta_l2=0.0,
        eta_l2=0.0,
        n_starts=1,
        random_state=None,
    ):
        super().__init__(
            beta_init=beta_init,
            max_iter=max_iter,
            tol=tol,
            min_baseline=min_baseline,
            min_decay=min_decay,
            alpha_upper=alpha_upper,
            beta_upper=beta_upper,
            alpha_l2=alpha_l2,
            beta_l2=beta_l2,
            eta_l2=eta_l2,
            n_starts=n_starts,
            random_state=random_state,
        )

        self.eta_bounds = eta_bounds

    def _bounds(self):
        return super()._bounds() + [self.eta_bounds]

    def _initial_theta(self, realizations, end_times, payload=None):
        base = super()._initial_theta(realizations, end_times, payload)
        return np.r_[base, 0.0]

    def _prepare_fit_data(self, events, end_times=None, marks=None, **kwargs):
        realizations, marks_realizations, end_times = self._prepare_realizations(
            events,
            end_times=end_times,
            extra=marks,
            extra_name="marks",
            extra_default=1.0,
        )

        z_realizations, stats = self._fit_mark_standardization(
            marks_realizations
        )

        self._marks_tmp_ = marks_realizations
        self._mark_stats_tmp_ = stats

        return realizations, z_realizations, end_times

    def _prepare_score_data(self, events, end_times=None, marks=None, **kwargs):
        realizations, marks_realizations, end_times = self._prepare_realizations(
            events,
            end_times=end_times,
            extra=marks,
            extra_name="marks",
            extra_default=1.0,
        )

        z_realizations = self._transform_marks_with_stats(
            marks_realizations,
            self.mark_stats_,
        )

        return realizations, z_realizations, end_times

    @staticmethod
    def _fit_mark_standardization(marks_realizations):
        raw_all = [
            np.asarray(m, dtype=float).ravel()
            for m in marks_realizations
        ]

        concatenated = (
            np.concatenate(raw_all)
            if sum(len(x) for x in raw_all) > 0
            else np.array([])
        )

        if len(concatenated) == 0:
            mean = 0.0
            std = 1.0

        else:
            mean = float(np.mean(concatenated))
            std = float(np.std(concatenated))

            if std <= 1e-12:
                std = 1.0

        z_realizations = [
            (raw - mean) / std
            for raw in raw_all
        ]

        return z_realizations, {
            "mean": mean,
            "std": std,
        }

    @staticmethod
    def _transform_marks_with_stats(marks_realizations, stats):
        return [
            (np.asarray(m, dtype=float).ravel() - stats["mean"]) / stats["std"]
            for m in marks_realizations
        ]

    @staticmethod
    def _kernel_integral_and_grads(times, z, T, beta, eta):
        if len(times) == 0:
            return 0.0, 0.0, 0.0

        rem = T - times

        w = np.exp(eta * z)
        e = np.exp(-beta * rem)

        S = np.sum(w * (1.0 - e))
        S_eta = np.sum(w * z * (1.0 - e))
        S_beta = np.sum(w * rem * e)

        return float(S), float(S_eta), float(S_beta)

    def _nll_grad_one(self, theta, times, z, T):
        p = self._unpack(theta)

        mu = p["mu"]
        alpha = p["alpha"]
        beta = p["beta"]
        eta = p["eta"]

        if mu <= 0 or alpha < 0 or beta <= 0:
            return np.inf, np.zeros_like(theta)

        ll = 0.0

        grad_mu = 0.0
        grad_alpha = 0.0
        grad_beta = 0.0
        grad_eta = 0.0

        r = 0.0
        q = 0.0
        u = 0.0

        last_t = 0.0
        n = len(times)
        k = 0

        while k < n:
            t = times[k]
            dt = t - last_t

            if dt < -1e-12:
                raise RuntimeError("Les timestamps doivent être triés.")

            if dt > 0:
                e = np.exp(-beta * dt)

                q = e * (q - dt * r)
                r = e * r
                u = e * u

            k2 = k + 1

            while k2 < n and times[k2] == t:
                k2 += 1

            z_group = z[k:k2]
            count = k2 - k

            lam = mu + alpha * r

            if lam <= 0 or not np.isfinite(lam):
                return np.inf, np.zeros_like(theta)

            inv_lam = 1.0 / lam

            ll += count * np.log(lam)

            grad_mu += count * inv_lam
            grad_alpha += count * r * inv_lam
            grad_beta += count * alpha * q * inv_lam
            grad_eta += count * alpha * u * inv_lam

            w_group = np.exp(eta * z_group)

            w_sum = np.sum(w_group)
            wz_sum = np.sum(w_group * z_group)

            r += beta * w_sum
            q += w_sum
            u += beta * wz_sum

            last_t = t
            k = k2

        S, S_eta, S_beta = self._kernel_integral_and_grads(
            times,
            z,
            T,
            beta,
            eta,
        )

        ll -= mu * T + alpha * S

        grad_mu -= T
        grad_alpha -= S
        grad_beta -= alpha * S_beta
        grad_eta -= alpha * S_eta

        nll = -ll

        grad = np.array(
            [
                -grad_mu,
                -grad_alpha,
                -grad_beta,
                -grad_eta,
            ],
            dtype=float,
        )

        return self._add_l2(nll, grad, p)

    def _post_fit(self, payload):
        self.marks_ = self._marks_tmp_
        self.z_marks_ = payload
        self.mark_stats_ = self._mark_stats_tmp_

        all_z = (
            np.concatenate(payload)
            if sum(len(z) for z in payload) > 0
            else np.array([])
        )

        if len(all_z) > 0:
            self.mean_mark_weight_ = float(
                np.mean(np.exp(self.eta_ * all_z))
            )
        else:
            self.mean_mark_weight_ = 1.0

        self.branching_ratio_ = float(
            self.alpha_ * self.mean_mark_weight_
        )

        self.is_stable_ = bool(self.branching_ratio_ < 1.0)

    def intensity_at_events(self, events=None, marks=None):
        if not hasattr(self, "baseline_"):
            raise RuntimeError(
                "Le modèle doit être fitté avant intensity_at_events()."
            )

        if events is None:
            times = self.events_[0]
            z = self.z_marks_[0]

        else:
            end_time = float(np.max(events)) if len(events) else 1.0

            realizations, z_realizations, _ = self._prepare_score_data(
                events,
                end_times=end_time,
                marks=marks,
            )

            if len(realizations) != 1:
                raise ValueError(
                    "intensity_at_events attend une seule réalisation."
                )

            times = realizations[0]
            z = z_realizations[0]

        intensities = np.zeros(len(times), dtype=float)

        r = 0.0
        last_t = 0.0

        n = len(times)
        k = 0

        while k < n:
            t = times[k]
            dt = t - last_t

            if dt > 0:
                r *= np.exp(-self.beta_ * dt)

            k2 = k + 1

            while k2 < n and times[k2] == t:
                k2 += 1

            intensities[k:k2] = self.baseline_ + self.alpha_ * r

            r += self.beta_ * np.sum(
                np.exp(self.eta_ * z[k:k2])
            )

            last_t = t
            k = k2

        return intensities
    
    def _kernel_cumulative_at(self, t, times, payload):
        if len(times) == 0 or t <= 0:
            return 0.0

        z = payload
        mask = times < t

        if not np.any(mask):
            return 0.0

        past = times[mask]
        past_z = z[mask]

        w = np.exp(self.eta_ * past_z)

        return float(
            np.sum(
                w * (1.0 - np.exp(-self.beta_ * (t - past)))
            )
        )

class UnivariateZDecayHawkesMLE(BaseUnivariateHawkesMLE):
    """
    Hawkes avec decay modulé par z_k.

    lambda(t)=mu+sum_{t_k<t} alpha*beta*exp(-beta*(t-t_k)/z_k)

    Les z_k sont observés et strictement positifs.

    Paramètres estimés :
        mu
        alpha
        beta

    Attention :
        le calcul exact est en O(n^2) si les z_k sont arbitraires.
    """

    def _prepare_fit_data(self, events, end_times=None, z=None, **kwargs):
        realizations, z_realizations, end_times = self._prepare_realizations(
            events,
            end_times=end_times,
            extra=z,
            extra_name="z",
            extra_default=1.0,
            positive_extra=True,
        )

        return realizations, z_realizations, end_times

    def _prepare_score_data(self, events, end_times=None, z=None, **kwargs):
        return self._prepare_fit_data(
            events,
            end_times=end_times,
            z=z,
        )

    @staticmethod
    def _kernel_integral_and_grad(times, z, T, beta):
        if len(times) == 0:
            return 0.0, 0.0

        rem = T - times
        e = np.exp(-beta * rem / z)

        S = np.sum(z * (1.0 - e))
        S_beta = np.sum(rem * e)

        return float(S), float(S_beta)

    def _nll_grad_one(self, theta, times, z, T):
        p = self._unpack(theta)

        mu = p["mu"]
        alpha = p["alpha"]
        beta = p["beta"]

        if mu <= 0 or alpha < 0 or beta <= 0:
            return np.inf, np.zeros_like(theta)

        ll = 0.0

        grad_mu = 0.0
        grad_alpha = 0.0
        grad_beta = 0.0

        n = len(times)
        k = 0

        while k < n:
            t = times[k]

            k2 = k + 1

            while k2 < n and times[k2] == t:
                k2 += 1

            count = k2 - k

            if k == 0:
                A = 0.0
                B = 0.0

            else:
                dt = t - times[:k]
                past_z = z[:k]

                e = np.exp(-beta * dt / past_z)

                A = np.sum(e)
                B = np.sum((dt / past_z) * e)

            lam = mu + alpha * beta * A

            if lam <= 0 or not np.isfinite(lam):
                return np.inf, np.zeros_like(theta)

            inv_lam = 1.0 / lam

            ll += count * np.log(lam)

            grad_mu += count * inv_lam
            grad_alpha += count * beta * A * inv_lam
            grad_beta += count * alpha * (A - beta * B) * inv_lam

            k = k2

        S, S_beta = self._kernel_integral_and_grad(
            times,
            z,
            T,
            beta,
        )

        ll -= mu * T + alpha * S

        grad_mu -= T
        grad_alpha -= S
        grad_beta -= alpha * S_beta

        nll = -ll

        grad = np.array(
            [
                -grad_mu,
                -grad_alpha,
                -grad_beta,
            ],
            dtype=float,
        )

        return self._add_l2(nll, grad, p)

    def _post_fit(self, payload):
        self.z_ = payload

        all_z = (
            np.concatenate(payload)
            if sum(len(x) for x in payload) > 0
            else np.array([])
        )

        if len(all_z) > 0:
            self.mean_z_ = float(np.mean(all_z))
        else:
            self.mean_z_ = 1.0

        self.branching_ratio_empirical_ = float(
            self.alpha_ * self.mean_z_
        )

        self.is_stable_empirical_ = bool(
            self.branching_ratio_empirical_ < 1.0
        )

    def intensity_at_events(self, events=None, z=None):
        if not hasattr(self, "baseline_"):
            raise RuntimeError(
                "Le modèle doit être fitté avant intensity_at_events()."
            )

        if events is None:
            times = self.events_[0]
            z = self.z_[0]

        else:
            times, z = self._prepare_one_realization(
                events,
                z,
                "z",
                1.0,
                True,
            )

        intensities = np.zeros(len(times), dtype=float)

        n = len(times)
        k = 0

        while k < n:
            t = times[k]

            k2 = k + 1

            while k2 < n and times[k2] == t:
                k2 += 1

            if k == 0:
                A = 0.0

            else:
                A = np.sum(
                    np.exp(
                        -self.beta_ * (t - times[:k]) / z[:k]
                    )
                )

            intensities[k:k2] = (
                self.baseline_
                + self.alpha_ * self.beta_ * A
            )

            k = k2

        return intensities
    
    def _kernel_cumulative_at(self, t, times, payload):
        if len(times) == 0 or t <= 0:
            return 0.0

        z = payload
        mask = times < t

        if not np.any(mask):
            return 0.0

        past = times[mask]
        past_z = z[mask]

        return float(
            np.sum(
                past_z * (
                    1.0 - np.exp(-self.beta_ * (t - past) / past_z)
                )
            )
        )