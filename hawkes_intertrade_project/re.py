import numpy as np
from scipy.optimize import minimize


class SimpleUnivariateHawkesMLE:
    """
    Hawkes univarié exponentiel, une seule réalisation.

    Sans marks :
        lambda(t) = mu + alpha * sum_{t_k < t} beta * exp(-beta * (t - t_k))

    Avec marks :
        lambda(t) = mu + alpha * sum_{t_k < t} mark_k * beta * exp(-beta * (t - t_k))

    Les marks sont des poids observés. On n'estime pas de paramètre de mark.
    """

    def __init__(
        self,
        beta_init=1.0,
        min_mu=1e-10,
        min_beta=1e-8,
        alpha_upper=None,
        beta_upper=None,
        max_iter=3000,
        tol=1e-8,
        allow_negative_marks=False,
    ):
        self.beta_init = float(beta_init)
        self.min_mu = float(min_mu)
        self.min_beta = float(min_beta)
        self.alpha_upper = alpha_upper
        self.beta_upper = beta_upper
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.allow_negative_marks = bool(allow_negative_marks)

        if self.beta_init <= 0:
            raise ValueError("beta_init doit être strictement positif.")

    # ------------------------------------------------------------------
    # Données
    # ------------------------------------------------------------------

    def _prepare_data(self, times, T, marks=None):
        times = np.asarray(times, dtype=float).ravel()

        if np.any(~np.isfinite(times)):
            raise ValueError("times contient des valeurs non finies.")

        T = float(T)

        if T <= 0 or not np.isfinite(T):
            raise ValueError("T doit être strictement positif et fini.")

        if np.any(times < 0):
            raise ValueError("Les timestamps doivent être positifs.")

        if np.any(times > T):
            raise ValueError("Tous les timestamps doivent être inférieurs ou égaux à T.")

        if marks is None:
            marks = np.ones_like(times, dtype=float)
        else:
            marks = np.asarray(marks, dtype=float).ravel()

            if len(marks) != len(times):
                raise ValueError("marks doit avoir la même longueur que times.")

            if np.any(~np.isfinite(marks)):
                raise ValueError("marks contient des valeurs non finies.")

            if not self.allow_negative_marks and np.any(marks < 0):
                raise ValueError("Les marks doivent être positifs ou nuls.")

        order = np.argsort(times)
        times = times[order]
        marks = marks[order]

        if len(times) > 1 and np.any(np.diff(times) <= 0):
            raise ValueError("Les timestamps doivent être strictement croissants.")

        return times, marks, T

    # ------------------------------------------------------------------
    # Paramètres
    # ------------------------------------------------------------------

    @staticmethod
    def _pack(mu, alpha, beta):
        return np.array([mu, alpha, beta], dtype=float)

    @staticmethod
    def _unpack(theta):
        theta = np.asarray(theta, dtype=float).ravel()

        if len(theta) != 3:
            raise ValueError("theta doit être de longueur 3 : [mu, alpha, beta].")

        return float(theta[0]), float(theta[1]), float(theta[2])

    def _bounds(self):
        return [
            (self.min_mu, None),
            (0.0, self.alpha_upper),
            (self.min_beta, self.beta_upper),
        ]

    def _initial_theta(self, times, T):
        n_events = len(times)

        mu0 = max(
            0.5 * n_events / max(T, 1e-12),
            10.0 * self.min_mu,
        )

        alpha0 = 0.05
        if self.alpha_upper is not None:
            alpha0 = min(alpha0, 0.9 * self.alpha_upper)

        beta0 = max(self.beta_init, 10.0 * self.min_beta)
        if self.beta_upper is not None:
            beta0 = min(beta0, 0.9 * self.beta_upper)

        return self._pack(mu0, alpha0, beta0)

    # ------------------------------------------------------------------
    # Négative log-vraisemblance + gradient
    # ------------------------------------------------------------------

    def neg_loglikelihood_with_grad(self, theta, times, T, marks=None):
        """
        Retourne :
            nll  : negative log-likelihood
            grad : gradient de nll par rapport à [mu, alpha, beta]

        Ici times et marks sont supposés déjà triés et valides.
        """
        mu, alpha, beta = self._unpack(theta)

        times = np.asarray(times, dtype=float).ravel()
        T = float(T)

        if marks is None:
            marks = np.ones_like(times, dtype=float)
        else:
            marks = np.asarray(marks, dtype=float).ravel()

        if mu <= 0 or alpha < 0 or beta <= 0:
            return np.inf, np.zeros_like(theta, dtype=float)

        # Mémoire du noyau :
        # r(t_k) = sum_{t_l < t_k} mark_l * beta * exp(-beta * (t_k - t_l))
        r = 0.0

        # Dérivée de r par rapport à beta
        q = 0.0

        log_term = 0.0

        grad_mu_log = 0.0
        grad_alpha_log = 0.0
        grad_beta_log = 0.0

        for k, t in enumerate(times):
            if k > 0:
                dt = t - times[k - 1]
                prev_mark = marks[k - 1]

                decay = np.exp(-beta * dt)

                # Ajout de l'événement précédent, puis décroissance
                r_start = r + prev_mark * beta
                q_start = q + prev_mark

                r = decay * r_start
                q = decay * (q_start - dt * r_start)

            lam = mu + alpha * r

            if lam <= 0 or not np.isfinite(lam):
                return np.inf, np.zeros_like(theta, dtype=float)

            inv_lam = 1.0 / lam

            log_term += np.log(lam)

            grad_mu_log += inv_lam
            grad_alpha_log += r * inv_lam
            grad_beta_log += alpha * q * inv_lam

        # Compensateur global :
        # Lambda(T) = mu*T + alpha * sum_k mark_k * (1 - exp(-beta*(T-t_k)))
        rem = T - times

        if np.any(rem < 0):
            return np.inf, np.zeros_like(theta, dtype=float)

        exp_tail = np.exp(-beta * rem)

        S_alpha = np.sum(marks * (1.0 - exp_tail))
        S_beta = np.sum(marks * rem * exp_tail)

        compensator = mu * T + alpha * S_alpha

        grad_mu_comp = T
        grad_alpha_comp = S_alpha
        grad_beta_comp = alpha * S_beta

        loglik = log_term - compensator
        nll = -float(loglik)

        # Gradient de nll = grad_comp - grad_log
        grad_mu = grad_mu_comp - grad_mu_log
        grad_alpha = grad_alpha_comp - grad_alpha_log
        grad_beta = grad_beta_comp - grad_beta_log

        grad = self._pack(grad_mu, grad_alpha, grad_beta)

        return nll, grad

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def fit(self, times, T, marks=None, x0=None):
        times, marks, T = self._prepare_data(times=times, T=T, marks=marks)

        if x0 is None:
            x0 = self._initial_theta(times, T)
        else:
            x0 = np.asarray(x0, dtype=float).ravel()

        result = minimize(
            fun=lambda th: self.neg_loglikelihood_with_grad(th, times, T, marks),
            x0=x0,
            jac=True,
            bounds=self._bounds(),
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        self.result_ = result
        self.success_ = bool(result.success)
        self.message_ = result.message
        self.n_iter_ = int(result.nit)

        self.times_ = times
        self.marks_ = marks
        self.T_ = T

        self.mu_, self.alpha_, self.beta_ = self._unpack(result.x)
        self.log_likelihood_ = -float(result.fun)

        mean_mark = float(np.mean(marks)) if len(marks) > 0 else 1.0
        self.branching_ratio_ = float(self.alpha_ * mean_mark)
        self.is_stable_ = bool(self.branching_ratio_ < 1.0)

        return self

    # ------------------------------------------------------------------
    # Intensité, compensateur, comptage
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not hasattr(self, "mu_"):
            raise RuntimeError("Le modèle doit être fitté avant utilisation.")

    def _as_grid(self, grid):
        scalar_input = np.ndim(grid) == 0
        grid = np.array([float(grid)]) if scalar_input else np.asarray(grid, dtype=float).ravel()

        if np.any(~np.isfinite(grid)):
            raise ValueError("grid contient des valeurs non finies.")
        if np.any(grid < 0):
            raise ValueError("grid doit contenir des temps positifs.")

        return grid, scalar_input

    def intensity_at(self, grid):
        """
        Retourne lambda(t).

        Si grid est scalaire : retourne un float.
        Sinon : retourne un array de shape (len(grid),).
        """
        self._check_fitted()
        grid, scalar_input = self._as_grid(grid)

        out = np.empty(len(grid), dtype=float)

        for idx, t in enumerate(grid):
            mask = self.times_ < t
            past = self.times_[mask]
            past_marks = self.marks_[mask]

            if len(past) == 0:
                kernel = 0.0
            else:
                rem = t - past
                kernel = np.sum(
                    past_marks * self.beta_ * np.exp(-self.beta_ * rem)
                )

            out[idx] = self.mu_ + self.alpha_ * kernel

        return float(out[0]) if scalar_input else out

    def cumulative_intensity_at(self, grid):
        """
        Retourne Lambda(t).

        Si grid est scalaire : retourne un float.
        Sinon : retourne un array de shape (len(grid),).
        """
        self._check_fitted()
        grid, scalar_input = self._as_grid(grid)

        out = np.empty(len(grid), dtype=float)

        for idx, t in enumerate(grid):
            mask = self.times_ < t
            past = self.times_[mask]
            past_marks = self.marks_[mask]

            if len(past) == 0:
                kernel_cum = 0.0
            else:
                rem = t - past
                kernel_cum = np.sum(
                    past_marks * (1.0 - np.exp(-self.beta_ * rem))
                )

            out[idx] = self.mu_ * t + self.alpha_ * kernel_cum

        return float(out[0]) if scalar_input else out

    def counting_process_at(self, grid):
        """
        Retourne N(t).
        """
        self._check_fitted()
        grid, scalar_input = self._as_grid(grid)

        out = np.searchsorted(self.times_, grid, side="right").astype(float)

        return float(out[0]) if scalar_input else out

    def validation_processes(self, grid=None, n_grid=1000):
        """
        Données utiles pour les diagnostics et les graphiques.
        """
        self._check_fitted()

        if grid is None:
            grid = np.unique(np.r_[np.linspace(0.0, self.T_, int(n_grid)), self.times_])
        else:
            grid = np.asarray(grid, dtype=float).ravel()

        lam = self.intensity_at(grid)
        Lambda = self.cumulative_intensity_at(grid)
        N = self.counting_process_at(grid)
        M = N - Lambda

        return {
            "time": grid,
            "lambda": lam,
            "Lambda": Lambda,
            "N": N,
            "M": M,
        }

    def intensity_at_events(self):
        self._check_fitted()
        return self.intensity_at(self.times_)

    def cumulative_intensity_at_events(self):
        self._check_fitted()
        return self.cumulative_intensity_at(self.times_)

    # ------------------------------------------------------------------
    # Résidus
    # ------------------------------------------------------------------

    def time_rescaling_residuals(self):
        self._check_fitted()

        if len(self.times_) == 0:
            return {"tau": np.array([]), "uniform": np.array([]), "event_times": np.array([])}

        Lambda_events = self.cumulative_intensity_at_events()
        tau = np.diff(np.r_[0.0, Lambda_events])
        uniform = 1.0 - np.exp(-tau)

        return {
            "tau": tau,
            "uniform": uniform,
            "event_times": self.times_,
            "Lambda_events": Lambda_events,
        }

    def residual_ks_tests(self):
        from scipy.stats import kstest

        residuals = self.time_rescaling_residuals()

        tau = residuals["tau"]
        u = residuals["uniform"]

        mask = (
            np.isfinite(tau)
            & np.isfinite(u)
            & (tau >= 0.0)
            & (u >= 0.0)
            & (u <= 1.0)
        )

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

    # ------------------------------------------------------------------
    # Graphiques
    # ------------------------------------------------------------------

    def plot_intensity(self, grid=None, n_grid=1000):
        import matplotlib.pyplot as plt

        data = self.validation_processes(grid=grid, n_grid=n_grid)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data["time"], data["lambda"], label=r"$\hat{\lambda}(t)$")
        ax.set_xlabel("t")
        ax.set_ylabel("Intensité")
        ax.set_title("Intensité estimée")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        return fig, ax

    def plot_cumulative_intensity(self, grid=None, n_grid=1000, show_counting=True):
        import matplotlib.pyplot as plt

        data = self.validation_processes(grid=grid, n_grid=n_grid)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data["time"], data["Lambda"], label=r"$\hat{\Lambda}(t)$")

        if show_counting:
            ax.step(data["time"], data["N"], where="post", label=r"$N(t)$")

        ax.set_xlabel("t")
        ax.set_ylabel("Valeur cumulée")
        ax.set_title("Compensateur cumulé")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        return fig, ax

    def plot_N_vs_Lambda(self, grid=None, n_grid=1000):
        import matplotlib.pyplot as plt

        data = self.validation_processes(grid=grid, n_grid=n_grid)

        fig, axes = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(10, 7),
            gridspec_kw={"height_ratios": [2, 1]},
        )

        ax0, ax1 = axes

        ax0.step(data["time"], data["N"], where="post", label=r"$N(t)$")
        ax0.plot(data["time"], data["Lambda"], label=r"$\hat{\Lambda}(t)$")
        ax0.set_ylabel("Valeur cumulée")
        ax0.set_title(r"$N(t)$ vs $\hat{\Lambda}(t)$")
        ax0.legend()
        ax0.grid(True)

        ax1.plot(data["time"], data["lambda"], label=r"$\hat{\lambda}(t)$")
        ax1.set_xlabel("t")
        ax1.set_ylabel("Intensité")
        ax1.set_title("Intensité estimée")
        ax1.legend()
        ax1.grid(True)

        fig.tight_layout()
        return fig, axes

    def plot_martingale(self, grid=None, n_grid=1000):
        import matplotlib.pyplot as plt

        data = self.validation_processes(grid=grid, n_grid=n_grid)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.step(data["time"], data["M"], where="post", label=r"$M(t)=N(t)-\hat{\Lambda}(t)$")
        ax.axhline(0.0, linestyle="--")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$N(t)-\hat{\Lambda}(t)$")
        ax.set_title("Martingale compensée")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        return fig, ax

    def plot_residuals_qq(self):
        import matplotlib.pyplot as plt

        residuals = self.time_rescaling_residuals()
        tau = residuals["tau"]
        tau = tau[np.isfinite(tau) & (tau >= 0.0)]

        if len(tau) == 0:
            raise ValueError("Aucun résidu valide à tracer.")

        tau_sorted = np.sort(tau)
        n = len(tau_sorted)
        probs = (np.arange(1, n + 1) - 0.5) / n
        exp_quantiles = -np.log(1.0 - probs)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(exp_quantiles, tau_sorted)

        lim_min = min(float(np.min(exp_quantiles)), float(np.min(tau_sorted)))
        lim_max = max(float(np.max(exp_quantiles)), float(np.max(tau_sorted)))
        ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--")

        ax.set_xlabel("Quantiles théoriques Exp(1)")
        ax.set_ylabel("Résidus observés")
        ax.set_title("QQ-plot des résidus")
        ax.grid(True)
        fig.tight_layout()

        return fig, ax

    def plot_uniform_residuals(self):
        import matplotlib.pyplot as plt

        residuals = self.time_rescaling_residuals()
        u = residuals["uniform"]
        u = u[np.isfinite(u) & (u >= 0.0) & (u <= 1.0)]

        if len(u) == 0:
            raise ValueError("Aucun résidu uniformisé valide à tracer.")

        u_sorted = np.sort(u)
        n = len(u_sorted)
        empirical = np.arange(1, n + 1) / n

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.step(u_sorted, empirical, where="post", label="CDF empirique")
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", label="CDF Uniform(0,1)")

        ax.set_xlabel(r"$U_i=1-\exp(-\tau_i)$")
        ax.set_ylabel("CDF")
        ax.set_title("Résidus uniformisés")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        return fig, ax

    # ------------------------------------------------------------------
    # Résumé
    # ------------------------------------------------------------------

    def get_params(self):
        self._check_fitted()

        return {
            "mu": self.mu_,
            "alpha": self.alpha_,
            "beta": self.beta_,
            "log_likelihood": self.log_likelihood_,
            "success": self.success_,
            "message": self.message_,
            "n_iter": self.n_iter_,
            "branching_ratio": self.branching_ratio_,
            "is_stable": self.is_stable_,
        }


class SimpleMultivariateHawkesMLE:
    """
    Hawkes multivarié exponentiel, une seule réalisation.

    Sans marks :
        lambda_i(t) = mu_i
                      + sum_j sum_{t_k^j < t}
                        alpha_ij * beta_ij * exp(-beta_ij * (t - t_k^j))

    Avec marks :
        lambda_i(t) = mu_i
                      + sum_j sum_{t_k^j < t}
                        alpha_ij * mark_k * beta_ij * exp(-beta_ij * (t - t_k^j))

    Les marks sont des poids observés. On n'estime pas de paramètre de mark.

    Convention :
        alpha[i, j] = effet d'un événement de type j sur l'intensité i.
        beta[i, j]  = vitesse de décroissance de l'effet j -> i.
    """

    def __init__(
        self,
        n_types,
        beta_init=1.0,
        min_mu=1e-10,
        min_beta=1e-8,
        alpha_upper=None,
        beta_upper=None,
        max_iter=3000,
        tol=1e-8,
        allow_negative_marks=False,
    ):
        self.n_types = int(n_types)
        self.beta_init = float(beta_init)
        self.min_mu = float(min_mu)
        self.min_beta = float(min_beta)
        self.alpha_upper = alpha_upper
        self.beta_upper = beta_upper
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.allow_negative_marks = bool(allow_negative_marks)

        if self.n_types <= 0:
            raise ValueError("n_types doit être strictement positif.")
        if self.beta_init <= 0:
            raise ValueError("beta_init doit être strictement positif.")

    # ------------------------------------------------------------------
    # Données
    # ------------------------------------------------------------------

    def _prepare_data(self, times, types, T, marks=None, types_start_at=0):
        times = np.asarray(times, dtype=float).ravel()
        types = np.asarray(types).ravel()

        if len(times) != len(types):
            raise ValueError("times et types doivent avoir la même longueur.")

        if np.any(~np.isfinite(times)):
            raise ValueError("times contient des valeurs non finies.")

        if np.any(types != np.floor(types)):
            raise ValueError("types doit contenir des entiers.")

        types = types.astype(int) - int(types_start_at)

        if np.any((types < 0) | (types >= self.n_types)):
            raise ValueError(
                f"types doit être compris entre {types_start_at} "
                f"et {types_start_at + self.n_types - 1}."
            )

        T = float(T)

        if T <= 0 or not np.isfinite(T):
            raise ValueError("T doit être strictement positif et fini.")

        if np.any(times < 0):
            raise ValueError("Les timestamps doivent être positifs.")

        if np.any(times > T):
            raise ValueError("Tous les timestamps doivent être inférieurs ou égaux à T.")

        if marks is None:
            marks = np.ones_like(times, dtype=float)
        else:
            marks = np.asarray(marks, dtype=float).ravel()

            if len(marks) != len(times):
                raise ValueError("marks doit avoir la même longueur que times.")

            if np.any(~np.isfinite(marks)):
                raise ValueError("marks contient des valeurs non finies.")

            if not self.allow_negative_marks and np.any(marks < 0):
                raise ValueError("Les marks doivent être positifs ou nuls.")

        order = np.argsort(times)
        times = times[order]
        types = types[order]
        marks = marks[order]

        if len(times) > 1 and np.any(np.diff(times) <= 0):
            raise ValueError("Les timestamps doivent être strictement croissants.")

        return times, types, marks, T

    # ------------------------------------------------------------------
    # Paramètres
    # ------------------------------------------------------------------

    def _pack(self, mu, alpha, beta):
        return np.r_[mu.ravel(), alpha.ravel(), beta.ravel()]

    def _unpack(self, theta):
        theta = np.asarray(theta, dtype=float).ravel()

        d = self.n_types
        expected = d + 2 * d * d

        if len(theta) != expected:
            raise ValueError(f"theta doit être de longueur {expected}.")

        idx = 0

        mu = theta[idx:idx + d]
        idx += d

        alpha = theta[idx:idx + d * d].reshape(d, d)
        idx += d * d

        beta = theta[idx:idx + d * d].reshape(d, d)

        return mu, alpha, beta

    def _bounds(self):
        d = self.n_types

        return (
            [(self.min_mu, None)] * d
            + [(0.0, self.alpha_upper)] * (d * d)
            + [(self.min_beta, self.beta_upper)] * (d * d)
        )

    def _initial_theta(self, times, types, T):
        d = self.n_types

        counts = np.bincount(types, minlength=d).astype(float)

        mu0 = np.maximum(
            0.5 * counts / max(T, 1e-12),
            10.0 * self.min_mu,
        )

        alpha0 = np.full((d, d), 0.05 / max(d, 1), dtype=float)

        if self.alpha_upper is not None:
            alpha0 = np.minimum(alpha0, 0.9 * self.alpha_upper)

        beta0_scalar = max(self.beta_init, 10.0 * self.min_beta)

        if self.beta_upper is not None:
            beta0_scalar = min(beta0_scalar, 0.9 * self.beta_upper)

        beta0 = np.full((d, d), beta0_scalar, dtype=float)

        return self._pack(mu0, alpha0, beta0)

    # ------------------------------------------------------------------
    # Négative log-vraisemblance + gradient
    # ------------------------------------------------------------------

    def neg_loglikelihood_with_grad(self, theta, times, types, T, marks=None):
        """
        Retourne :
            nll  : negative log-likelihood
            grad : gradient de nll par rapport à [mu, alpha.ravel(), beta.ravel()]

        Ici times, types et marks sont supposés déjà triés et valides.
        Les types doivent être codés entre 0 et n_types-1.
        """
        mu, alpha, beta = self._unpack(theta)

        times = np.asarray(times, dtype=float).ravel()
        types = np.asarray(types, dtype=int).ravel()
        T = float(T)

        if marks is None:
            marks = np.ones_like(times, dtype=float)
        else:
            marks = np.asarray(marks, dtype=float).ravel()

        if np.any(mu <= 0) or np.any(alpha < 0) or np.any(beta <= 0):
            return np.inf, np.zeros_like(theta, dtype=float)

        d = self.n_types
        n = len(times)

        # R[i, j] = sum_{events passés de type j}
        #           mark_k * beta[i,j] * exp(-beta[i,j] * age)
        R = np.zeros((d, d), dtype=float)

        # Dérivée de R[i,j] par rapport à beta[i,j]
        R_beta = np.zeros((d, d), dtype=float)

        log_term = 0.0

        grad_mu_log = np.zeros(d, dtype=float)
        grad_alpha_log = np.zeros((d, d), dtype=float)
        grad_beta_log = np.zeros((d, d), dtype=float)

        for k in range(n):
            if k > 0:
                dt = times[k] - times[k - 1]
                prev_type = types[k - 1]
                prev_mark = marks[k - 1]

                decay = np.exp(-beta * dt)

                # Ajout de l'événement précédent dans la colonne prev_type
                R[:, prev_type] += prev_mark * beta[:, prev_type]
                R_beta[:, prev_type] += prev_mark

                R_start = R.copy()
                R_beta_start = R_beta.copy()

                # Décroissance jusqu'au timestamp courant
                R = decay * R_start
                R_beta = decay * (R_beta_start - dt * R_start)

            current_type = types[k]

            lam = mu + np.sum(alpha * R, axis=1)
            lam_m = lam[current_type]

            if lam_m <= 0 or not np.isfinite(lam_m):
                return np.inf, np.zeros_like(theta, dtype=float)

            inv_lam_m = 1.0 / lam_m

            log_term += np.log(lam_m)

            # Seule la ligne current_type intervient dans log(lambda_current_type)
            grad_mu_log[current_type] += inv_lam_m
            grad_alpha_log[current_type, :] += R[current_type, :] * inv_lam_m
            grad_beta_log[current_type, :] += (
                alpha[current_type, :] * R_beta[current_type, :] * inv_lam_m
            )

        # Compensateur global :
        # sum_i Lambda_i(T)
        compensator = T * np.sum(mu)

        grad_mu_comp = np.full(d, T, dtype=float)
        grad_alpha_comp = np.zeros((d, d), dtype=float)
        grad_beta_comp = np.zeros((d, d), dtype=float)

        for j in range(d):
            mask = types == j
            tj = times[mask]
            wj = marks[mask]

            if len(tj) == 0:
                continue

            rem = T - tj

            if np.any(rem < 0):
                return np.inf, np.zeros_like(theta, dtype=float)

            decay_to_T = np.exp(-beta[:, [j]] * rem[None, :])

            # Pour chaque i :
            # S_alpha[i] = sum_{events de type j}
            #              mark_k * (1 - exp(-beta[i,j]*(T-t_k)))
            S_alpha = np.sum(wj[None, :] * (1.0 - decay_to_T), axis=1)

            # Dérivée de S_alpha par rapport à beta[i,j]
            S_beta = np.sum(wj[None, :] * rem[None, :] * decay_to_T, axis=1)

            compensator += np.sum(alpha[:, j] * S_alpha)

            grad_alpha_comp[:, j] += S_alpha
            grad_beta_comp[:, j] += alpha[:, j] * S_beta

        loglik = log_term - compensator
        nll = -float(loglik)

        # Gradient de nll = grad_comp - grad_log
        grad_mu = grad_mu_comp - grad_mu_log
        grad_alpha = grad_alpha_comp - grad_alpha_log
        grad_beta = grad_beta_comp - grad_beta_log

        grad = self._pack(grad_mu, grad_alpha, grad_beta)

        return nll, grad

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def fit(self, times, types, T, marks=None, x0=None, types_start_at=0):
        times, types, marks, T = self._prepare_data(
            times=times,
            types=types,
            T=T,
            marks=marks,
            types_start_at=types_start_at,
        )

        if x0 is None:
            x0 = self._initial_theta(times, types, T)
        else:
            x0 = np.asarray(x0, dtype=float).ravel()

        result = minimize(
            fun=lambda th: self.neg_loglikelihood_with_grad(th, times, types, T, marks),
            x0=x0,
            jac=True,
            bounds=self._bounds(),
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        self.result_ = result
        self.success_ = bool(result.success)
        self.message_ = result.message
        self.n_iter_ = int(result.nit)

        self.times_ = times
        self.types_ = types
        self.marks_ = marks
        self.T_ = T

        self.mu_, self.alpha_, self.beta_ = self._unpack(result.x)
        self.log_likelihood_ = -float(result.fun)

        self.branching_matrix_ = self.alpha_.copy()
        self.branching_ratio_ = float(
            np.max(np.abs(np.linalg.eigvals(self.branching_matrix_)))
        )
        self.is_stable_ = bool(self.branching_ratio_ < 1.0)

        return self

    # ------------------------------------------------------------------
    # Intensité, compensateur, comptage
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not hasattr(self, "mu_"):
            raise RuntimeError("Le modèle doit être fitté avant utilisation.")

    def _as_grid(self, grid):
        scalar_input = np.ndim(grid) == 0
        grid = np.array([float(grid)]) if scalar_input else np.asarray(grid, dtype=float).ravel()

        if np.any(~np.isfinite(grid)):
            raise ValueError("grid contient des valeurs non finies.")
        if np.any(grid < 0):
            raise ValueError("grid doit contenir des temps positifs.")

        return grid, scalar_input

    def intensity_at(self, grid):
        """
        Retourne lambda(t).

        Si grid est scalaire :
            array shape (d,)

        Sinon :
            array shape (len(grid), d)
        """
        self._check_fitted()
        grid, scalar_input = self._as_grid(grid)

        d = self.n_types
        out = np.zeros((len(grid), d), dtype=float)

        for idx, t in enumerate(grid):
            K = np.zeros((d, d), dtype=float)

            for j in range(d):
                mask = (self.types_ == j) & (self.times_ < t)
                past = self.times_[mask]
                past_marks = self.marks_[mask]

                if len(past) == 0:
                    continue

                rem = t - past

                K[:, j] = np.sum(
                    past_marks[None, :]
                    * self.beta_[:, [j]]
                    * np.exp(-self.beta_[:, [j]] * rem[None, :]),
                    axis=1,
                )

            out[idx, :] = self.mu_ + np.sum(self.alpha_ * K, axis=1)

        return out[0, :] if scalar_input else out

    def cumulative_intensity_at(self, grid):
        """
        Retourne Lambda(t).

        Si grid est scalaire :
            array shape (d,)

        Sinon :
            array shape (len(grid), d)
        """
        self._check_fitted()
        grid, scalar_input = self._as_grid(grid)

        d = self.n_types
        out = np.zeros((len(grid), d), dtype=float)

        for idx, t in enumerate(grid):
            Kcum = np.zeros((d, d), dtype=float)

            for j in range(d):
                mask = (self.types_ == j) & (self.times_ < t)
                past = self.times_[mask]
                past_marks = self.marks_[mask]

                if len(past) == 0:
                    continue

                rem = t - past

                Kcum[:, j] = np.sum(
                    past_marks[None, :]
                    * (1.0 - np.exp(-self.beta_[:, [j]] * rem[None, :])),
                    axis=1,
                )

            out[idx, :] = self.mu_ * t + np.sum(self.alpha_ * Kcum, axis=1)

        return out[0, :] if scalar_input else out

    def counting_process_at(self, grid):
        """
        Retourne N_i(t) pour chaque dimension i.

        Si grid est scalaire :
            array shape (d,)

        Sinon :
            array shape (len(grid), d)
        """
        self._check_fitted()
        grid, scalar_input = self._as_grid(grid)

        d = self.n_types
        out = np.zeros((len(grid), d), dtype=float)

        for j in range(d):
            tj = self.times_[self.types_ == j]
            out[:, j] = np.searchsorted(tj, grid, side="right").astype(float)

        return out[0, :] if scalar_input else out

    def validation_processes(self, grid=None, n_grid=1000):
        """
        Données utiles pour les diagnostics et les graphiques.
        """
        self._check_fitted()

        if grid is None:
            grid = np.unique(np.r_[np.linspace(0.0, self.T_, int(n_grid)), self.times_])
        else:
            grid = np.asarray(grid, dtype=float).ravel()

        lam = self.intensity_at(grid)
        Lambda = self.cumulative_intensity_at(grid)
        N = self.counting_process_at(grid)
        M = N - Lambda

        return {
            "time": grid,
            "lambda": lam,
            "Lambda": Lambda,
            "N": N,
            "M": M,
            "lambda_total": np.sum(lam, axis=1),
            "Lambda_total": np.sum(Lambda, axis=1),
            "N_total": np.sum(N, axis=1),
            "M_total": np.sum(M, axis=1),
        }

    def intensity_at_events(self, observed_only=True):
        self._check_fitted()

        lam = self.intensity_at(self.times_)

        if observed_only:
            return lam[np.arange(len(self.times_)), self.types_]

        return lam

    def cumulative_intensity_at_events(self, observed_only=True):
        self._check_fitted()

        Lambda = self.cumulative_intensity_at(self.times_)

        if observed_only:
            return Lambda[np.arange(len(self.times_)), self.types_]

        return Lambda

    # ------------------------------------------------------------------
    # Résidus
    # ------------------------------------------------------------------

    def time_rescaling_residuals(self, dimension=None, total=False):
        """
        Résidus de time-rescaling.

        total=True :
            résidus du processus agrégé, avec intensité totale.

        dimension=j :
            résidus de la dimension j uniquement.

        dimension=None et total=False :
            retourne les résidus concaténés de toutes les dimensions.
        """
        self._check_fitted()

        if total:
            Lambda_total_events = np.sum(self.cumulative_intensity_at(self.times_), axis=1)
            tau = np.diff(np.r_[0.0, Lambda_total_events])
            uniform = 1.0 - np.exp(-tau)

            return {
                "tau": tau,
                "uniform": uniform,
                "dimension": np.full(len(tau), -1, dtype=int),
            }

        dims = range(self.n_types) if dimension is None else [int(dimension)]

        all_tau = []
        all_uniform = []
        all_dim = []

        for j in dims:
            if not (0 <= j < self.n_types):
                raise ValueError("dimension doit être comprise entre 0 et n_types-1.")

            mask = self.types_ == j
            tj = self.times_[mask]

            if len(tj) == 0:
                continue

            Lambda_j = self.cumulative_intensity_at(tj)[:, j]
            tau_j = np.diff(np.r_[0.0, Lambda_j])
            uniform_j = 1.0 - np.exp(-tau_j)

            all_tau.append(tau_j)
            all_uniform.append(uniform_j)
            all_dim.append(np.full(len(tau_j), j, dtype=int))

        if len(all_tau) == 0:
            return {
                "tau": np.array([]),
                "uniform": np.array([]),
                "dimension": np.array([], dtype=int),
            }

        return {
            "tau": np.concatenate(all_tau),
            "uniform": np.concatenate(all_uniform),
            "dimension": np.concatenate(all_dim),
        }

    def residual_ks_tests(self, dimension=None, total=False):
        from scipy.stats import kstest

        residuals = self.time_rescaling_residuals(dimension=dimension, total=total)

        tau = residuals["tau"]
        u = residuals["uniform"]

        mask = (
            np.isfinite(tau)
            & np.isfinite(u)
            & (tau >= 0.0)
            & (u >= 0.0)
            & (u <= 1.0)
        )

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

    # ------------------------------------------------------------------
    # Graphiques
    # ------------------------------------------------------------------

    def _dims_to_plot(self, dims):
        if dims is None:
            return list(range(self.n_types))

        if np.ndim(dims) == 0:
            dims = [int(dims)]
        else:
            dims = [int(x) for x in dims]

        for j in dims:
            if not (0 <= j < self.n_types):
                raise ValueError("Toutes les dimensions doivent être dans 0,...,n_types-1.")

        return dims

    def plot_intensity(self, grid=None, n_grid=1000, dims=None, total=False):
        import matplotlib.pyplot as plt

        data = self.validation_processes(grid=grid, n_grid=n_grid)

        fig, ax = plt.subplots(figsize=(10, 4))

        if total:
            ax.plot(data["time"], data["lambda_total"], label=r"$\hat{\lambda}_{total}(t)$")
        else:
            for j in self._dims_to_plot(dims):
                ax.plot(data["time"], data["lambda"][:, j], label=fr"$\hat{{\lambda}}_{j}(t)$")

        ax.set_xlabel("t")
        ax.set_ylabel("Intensité")
        ax.set_title("Intensités estimées")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        return fig, ax

    def plot_cumulative_intensity(
        self,
        grid=None,
        n_grid=1000,
        dims=None,
        total=False,
        show_counting=True,
    ):
        import matplotlib.pyplot as plt

        data = self.validation_processes(grid=grid, n_grid=n_grid)

        fig, ax = plt.subplots(figsize=(10, 4))

        if total:
            ax.plot(data["time"], data["Lambda_total"], label=r"$\hat{\Lambda}_{total}(t)$")
            if show_counting:
                ax.step(data["time"], data["N_total"], where="post", label=r"$N_{total}(t)$")
        else:
            for j in self._dims_to_plot(dims):
                ax.plot(data["time"], data["Lambda"][:, j], label=fr"$\hat{{\Lambda}}_{j}(t)$")
                if show_counting:
                    ax.step(data["time"], data["N"][:, j], where="post", label=fr"$N_{j}(t)$")

        ax.set_xlabel("t")
        ax.set_ylabel("Valeur cumulée")
        ax.set_title("Compensateurs cumulés")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        return fig, ax

    def plot_N_vs_Lambda(self, grid=None, n_grid=1000, dims=None, total=False):
        import matplotlib.pyplot as plt

        data = self.validation_processes(grid=grid, n_grid=n_grid)

        fig, axes = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(10, 7),
            gridspec_kw={"height_ratios": [2, 1]},
        )

        ax0, ax1 = axes

        if total:
            ax0.step(data["time"], data["N_total"], where="post", label=r"$N_{total}(t)$")
            ax0.plot(data["time"], data["Lambda_total"], label=r"$\hat{\Lambda}_{total}(t)$")
            ax1.plot(data["time"], data["lambda_total"], label=r"$\hat{\lambda}_{total}(t)$")
        else:
            for j in self._dims_to_plot(dims):
                ax0.step(data["time"], data["N"][:, j], where="post", label=fr"$N_{j}(t)$")
                ax0.plot(data["time"], data["Lambda"][:, j], label=fr"$\hat{{\Lambda}}_{j}(t)$")
                ax1.plot(data["time"], data["lambda"][:, j], label=fr"$\hat{{\lambda}}_{j}(t)$")

        ax0.set_ylabel("Valeur cumulée")
        ax0.set_title(r"$N(t)$ vs $\hat{\Lambda}(t)$")
        ax0.legend()
        ax0.grid(True)

        ax1.set_xlabel("t")
        ax1.set_ylabel("Intensité")
        ax1.set_title("Intensités estimées")
        ax1.legend()
        ax1.grid(True)

        fig.tight_layout()
        return fig, axes

    def plot_martingale(self, grid=None, n_grid=1000, dims=None, total=False):
        import matplotlib.pyplot as plt

        data = self.validation_processes(grid=grid, n_grid=n_grid)

        fig, ax = plt.subplots(figsize=(10, 4))

        if total:
            ax.step(data["time"], data["M_total"], where="post", label=r"$M_{total}(t)$")
        else:
            for j in self._dims_to_plot(dims):
                ax.step(data["time"], data["M"][:, j], where="post", label=fr"$M_{j}(t)$")

        ax.axhline(0.0, linestyle="--")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$N(t)-\hat{\Lambda}(t)$")
        ax.set_title("Martingales compensées")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        return fig, ax

    def plot_residuals_qq(self, dimension=None, total=False):
        import matplotlib.pyplot as plt

        residuals = self.time_rescaling_residuals(dimension=dimension, total=total)

        tau = residuals["tau"]
        tau = tau[np.isfinite(tau) & (tau >= 0.0)]

        if len(tau) == 0:
            raise ValueError("Aucun résidu valide à tracer.")

        tau_sorted = np.sort(tau)
        n = len(tau_sorted)
        probs = (np.arange(1, n + 1) - 0.5) / n
        exp_quantiles = -np.log(1.0 - probs)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(exp_quantiles, tau_sorted)

        lim_min = min(float(np.min(exp_quantiles)), float(np.min(tau_sorted)))
        lim_max = max(float(np.max(exp_quantiles)), float(np.max(tau_sorted)))
        ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--")

        ax.set_xlabel("Quantiles théoriques Exp(1)")
        ax.set_ylabel("Résidus observés")
        ax.set_title("QQ-plot des résidus")
        ax.grid(True)
        fig.tight_layout()

        return fig, ax

    def plot_uniform_residuals(self, dimension=None, total=False):
        import matplotlib.pyplot as plt

        residuals = self.time_rescaling_residuals(dimension=dimension, total=total)

        u = residuals["uniform"]
        u = u[np.isfinite(u) & (u >= 0.0) & (u <= 1.0)]

        if len(u) == 0:
            raise ValueError("Aucun résidu uniformisé valide à tracer.")

        u_sorted = np.sort(u)
        n = len(u_sorted)
        empirical = np.arange(1, n + 1) / n

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.step(u_sorted, empirical, where="post", label="CDF empirique")
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", label="CDF Uniform(0,1)")

        ax.set_xlabel(r"$U_i=1-\exp(-\tau_i)$")
        ax.set_ylabel("CDF")
        ax.set_title("Résidus uniformisés")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        return fig, ax

    # ------------------------------------------------------------------
    # Résumé
    # ------------------------------------------------------------------

    def get_params(self):
        self._check_fitted()

        return {
            "mu": self.mu_,
            "alpha": self.alpha_,
            "beta": self.beta_,
            "log_likelihood": self.log_likelihood_,
            "success": self.success_,
            "message": self.message_,
            "n_iter": self.n_iter_,
            "branching_matrix": self.branching_matrix_,
            "branching_ratio": self.branching_ratio_,
            "is_stable": self.is_stable_,
        }
