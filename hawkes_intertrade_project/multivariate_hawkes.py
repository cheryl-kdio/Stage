import numpy as np
from scipy.optimize import minimize

def multivariate_loglikelihood_with_grad(theta, tList, dim=None, dimensional=False):
    """

    Parameters
    ----------
    theta : tuple of array
        Tuple containing 3 arrays. First corresponds to vector of baseline intensities mu. Second is a square matrix
        corresponding to interaction matrix alpha. Last is vector of recovery rates beta.

    tList : list of tuple
        List containing tuples (t, m) where t is the time of event and m is the mark (dimension). The marks must go from
        1 to nb_of_dimensions.
        Important to note that this algorithm expects the first and last time to mark the beginning and
        the horizon of the observed process. As such, first and last marks must be equal to 0, signifying that they are
        not real event times.
        The algorithm checks by itself if this condition is respected, otherwise it sets the beginning at 0 and the end
        equal to the last time event.
    dim : int
        Num_ker of processes only necessary if providing 1-dimensional theta. Default is None
    dimensional : bool
        Whether to return the sum of loglikelihood or decomposed in each dimension. Default is False.
Returns
    -------
    likelihood : array of float
        Value of likelihood at each process.
        The value returned is the opposite of the mathematical likelihood in order to use minimization packages.
    """

    if isinstance(theta, np.ndarray):
        if dim is None:
            raise ValueError("Must provide dimension to unpack correctly")
        else:
            mu = np.array(theta[:dim]).reshape((dim, 1))
            alpha = np.array(theta[dim:dim * (dim + 1)]).reshape((dim, dim))
            beta = np.array(theta[dim * (dim + 1):]).reshape((dim, 1))
    else:
        mu, alpha, beta = (i.copy() for i in theta)
    
    timestamps = tList.copy()

    # We first check if we have the correct beginning and ending.
    if timestamps[0][1] > 0:
        timestamps = [(0, 0)] + timestamps
    if timestamps[-1][1] > 0:
        timestamps += [(timestamps[-1][0], 0)]

    # Initialise values
    t_k, m_k = timestamps[1]
    m_k = m_k - 1
    # Compensator between beginning and first event time
    compensator = mu*t_k
    # Intensity before first jump
    log_i = np.zeros((dim,1))
    log_i[m_k-1] = np.log(mu[m_k]) 

    ############# Gradient

    # Initialize grad
    grad_mu = np.zeros((dim, 1))
    grad_alpha = np.zeros((dim, dim))
    grad_beta = np.zeros((dim, dim))

    # For first interval/jump
    grad_mu += t_k
    grad_mu[m_k] -= 1 / mu[m_k]

    # Memory process
    R = np.zeros((dim, dim))
    R_beta = np.zeros((dim, dim))
    
    # # For first event
    # R[:,[m_k]] += beta
    # R_beta[:, [m_k]] += 1.0

    last_t = t_k
    last_m = m_k
    # j=1

    for t, m in timestamps[2:]:
        dt = t - last_t
        m = m - 1

        # Decay
        decay = np.exp(-beta * dt)

        # Ajout de la dernière contribution
        R[:, last_m] += beta[:, last_m]
        R_beta[:, last_m] += 1.0

        # Etat utilisé pour le compensateur
        R_start = R.copy()
        R_beta_start = R_beta.copy()

        # Terme d'intégration
        A = (1.0 - decay) / beta

        # Compensateur sur l'intervalle
        compensator += (
            mu * dt
            + np.sum(alpha * R_start * A, axis=1, keepdims=True)
        )

        # Décroissance jusqu'au temps courant
        R = decay * R_start
        R_beta = -dt * R + decay * R_beta_start

        # Intensité au temps courant
        lambda_t = mu + np.sum(alpha * R, axis=1, keepdims=True)

        # Terme log
        log_i[m, 0] += np.log(lambda_t[m, 0])

        ############ Gradient de la log-vraisemblance

        # Dérivée de A = (1 - exp(-beta * dt)) / beta
        A_beta = (beta * dt * decay - (1.0 - decay)) / (beta ** 2)

        #### Gradient du compensateur

        # grad C wrt mu
        grad_mu -= dt

        # grad C wrt alpha
        grad_alpha -= R_start * A

        # grad C wrt beta
        grad_beta -= alpha * (R_beta_start * A + R_start * A_beta)

        #### Gradient du terme log

        # grad log(lambda_m) wrt mu_m
        grad_mu[m, 0] += 1.0 / lambda_t[m, 0]

        # grad log(lambda_m) wrt alpha_mj
        grad_alpha[m, :] += R[m, :] / lambda_t[m, 0]

        # grad log(lambda_m) wrt beta_mj
        grad_beta[m, :] += (
            alpha[m, :] * R_beta[m, :]
            / lambda_t[m, 0]
        )

        last_t = t
        last_m = m

    likelihood = log_i - compensator
    grad_comp = np.concatenate((grad_mu, np.ravel(grad_alpha).reshape((dim * dim, 1)), grad_beta))
    if not(dimensional):
        likelihood = np.sum(likelihood)
    return -likelihood, grad_comp.squeeze()


class multivariate_estimator_bfgs_grad(object):
    """
    Estimator class for Exponential Hawkes process obtained through minimizaton of a loss using the L-BFGS-B algorithm.

    Attributes
    ----------
    res : OptimizeResult
        Result from minimization.
    """

    def __init__(self, loss=multivariate_loglikelihood_simplified, grad=True, dimension=None, initial_guess="random",
                 options=None, penalty=False, C=1, eps=1e-6):
        """
        Parameters
        ----------
        loss : {loglikelihood, likelihood_approximated} or callable.
            Function to minimize. Default is loglikelihood.
        dimension : int
            Dimension of problem to optimize. Default is None.
        initial_guess : str or ndarray.
            Initial guess for estimated vector. Either random initialization, or given vector of dimension (2*dimension + dimension**2,). Default is "random".
        options : dict
            Options to pass to the minimization method. Default is {'disp': False}.

        Attributes
        ----------
        bounds :
        """
        if dimension is None:
            raise ValueError("Dimension is necessary for initialization.")
        self.dim = dimension
        self.penalty = penalty
        if penalty:
            self.loss = lambda x, y, z: loss(x, y, z) + C * np.linalg.norm(x[-dimension:])
    
        else:
            if isinstance(grad, bool) and grad:
                self.loss = multivariate_loglikelihood_with_grad
            else:
                self.loss = loss
            self.grad = grad

        self.bounds = [(1e-12, None) for i in range(self.dim)] + [(None, None) for i in range(self.dim * self.dim)] + [
            (1e-12, None) for i in range(self.dim)]
        if isinstance(initial_guess, str) and initial_guess == "random":
            self.initial_guess = np.concatenate(
                (np.concatenate((np.ones(self.dim), np.ones(self.dim * self.dim))), np.ones(self.dim)))
        if options is None:
            self.options = {'disp': False}
        else:
            self.options = options

    def fit(self, timestamps, limit=1000):
        """
        Parameters
        ----------
        timestamps : list of tuple.
            Ordered list containing event times and marks.
        """

        self.res = minimize(self.loss, self.initial_guess, method="L-BFGS-B", jac=self.grad,
                            args=(timestamps, self.dim), bounds=self.bounds,
                            options=self.options)

        self.mu_estim = np.array(self.res.x[0: self.dim])
        self.alpha_estim = np.array(self.res.x[self.dim: self.dim + self.dim ** 2]).reshape((self.dim, self.dim))
        self.beta_estim = np.array(self.res.x[-self.dim:])

        return self.mu_estim, self.alpha_estim, self.beta_estim
    

# OR
import numpy as np
from scipy.optimize import minimize


class SimpleMultivariateHawkesMLE:
    """
    Hawkes multivarié exponentiel, une seule réalisation.

    Sans marks :
        lambda_i(t) = mu_i + sum_j sum_{t_k^j < t}
                      alpha_ij * beta_ij * exp(-beta_ij * (t - t_k^j))

    Avec marks :
        lambda_i(t) = mu_i + sum_j sum_{t_k^j < t}
                      alpha_ij * mark_k * beta_ij * exp(-beta_ij * (t - t_k^j))

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
    ):
        self.n_types = int(n_types)
        self.beta_init = float(beta_init)
        self.min_mu = float(min_mu)
        self.min_beta = float(min_beta)
        self.alpha_upper = alpha_upper
        self.beta_upper = beta_upper
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        if self.n_types <= 0:
            raise ValueError("n_types doit être strictement positif.")
        if self.beta_init <= 0:
            raise ValueError("beta_init doit être strictement positif.")

    # ------------------------------------------------------------------
    # Préparation des données
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
            raise ValueError("Les temps doivent être positifs.")
        if np.any(times > T):
            raise ValueError("Tous les temps doivent être inférieurs ou égaux à T.")

        if marks is None:
            marks = np.ones_like(times, dtype=float)
        else:
            marks = np.asarray(marks, dtype=float).ravel()
            if len(marks) != len(times):
                raise ValueError("marks doit avoir la même longueur que times.")
            if np.any(~np.isfinite(marks)):
                raise ValueError("marks contient des valeurs non finies.")

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
    # Negative log-vraisemblance + gradient
    # ------------------------------------------------------------------

    def neg_loglikelihood_with_grad(self, theta, times, types, T, marks=None):
        mu, alpha, beta = self._unpack(theta)
        d = self.n_types

        times = np.asarray(times, dtype=float).ravel()
        types = np.asarray(types, dtype=int).ravel()

        if marks is None:
            marks = np.ones_like(times, dtype=float)
        else:
            marks = np.asarray(marks, dtype=float).ravel()

        if np.any(mu <= 0) or np.any(alpha < 0) or np.any(beta <= 0):
            return np.inf, np.zeros_like(theta)

        n = len(times)

        # R[i, j] = sum_{events passés de type j} mark_k * beta_ij * exp(-beta_ij * age)
        R = np.zeros((d, d), dtype=float)

        # dérivée de R[i, j] par rapport à beta[i, j]
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

                # Ajout de la contribution du dernier événement
                R[:, prev_type] += prev_mark * beta[:, prev_type]
                R_beta[:, prev_type] += prev_mark

                # Sauvegarde avant décroissance
                R_start = R.copy()
                R_beta_start = R_beta.copy()

                # Décroissance jusqu'au temps courant
                R = decay * R_start
                R_beta = decay * (R_beta_start - dt * R_start)

            current_type = types[k]

            lam = mu + np.sum(alpha * R, axis=1)
            lam_m = lam[current_type]

            if lam_m <= 0 or not np.isfinite(lam_m):
                return np.inf, np.zeros_like(theta)

            inv_lam_m = 1.0 / lam_m
            log_term += np.log(lam_m)

            # Gradient du terme log
            grad_mu_log[current_type] += inv_lam_m
            grad_alpha_log[current_type, :] += R[current_type, :] * inv_lam_m
            grad_beta_log[current_type, :] += (
                alpha[current_type, :] * R_beta[current_type, :] * inv_lam_m
            )

        # Compensateur global
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

            decay_to_T = np.exp(-beta[:, [j]] * rem[None, :])

            # S_alpha[i] = sum_{events type j} mark_k * (1 - exp(-beta_ij * (T - t_k)))
            S_alpha = np.sum(wj[None, :] * (1.0 - decay_to_T), axis=1)

            # dérivée wrt beta_ij
            S_beta = np.sum(wj[None, :] * rem[None, :] * decay_to_T, axis=1)

            compensator += np.sum(alpha[:, j] * S_alpha)

            grad_alpha_comp[:, j] += S_alpha
            grad_beta_comp[:, j] += alpha[:, j] * S_beta

        loglik = log_term - compensator
        nll = -float(loglik)

        # Gradient de la négative log-vraisemblance
        grad_mu = grad_mu_comp - grad_mu_log
        grad_alpha = grad_alpha_comp - grad_alpha_log
        grad_beta = grad_beta_comp - grad_beta_log

        grad = self._pack(grad_mu, grad_alpha, grad_beta)

        return nll, grad

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, times, types, T, marks=None, x0=None, types_start_at=0):
        times, types, marks, T = self._prepare_data(
            times=times,
            types=types,
            marks=marks,
            T=T,
            types_start_at=types_start_at,
        )

        if x0 is None:
            x0 = self._initial_theta(times, types, T)
        else:
            x0 = np.asarray(x0, dtype=float).ravel()

        result = minimize(
            fun=lambda th: self.neg_loglikelihood_with_grad(
                th, times, types, T, marks
            ),
            x0=x0,
            jac=True,
            bounds=self._bounds(),
            method="L-BFGS-B",
            options={
                "maxiter": self.max_iter,
                "ftol": self.tol,
            },
        )

        self.result_ = result
        self.success_ = bool(result.success)
        self.message_ = result.message
        self.n_iter_ = result.nit

        self.times_ = times
        self.types_ = types
        self.marks_ = marks
        self.T_ = T

        self.mu_, self.alpha_, self.beta_ = self._unpack(result.x)
        self.baseline_ = self.mu_

        self.log_likelihood_ = -float(result.fun)

        self.branching_matrix_ = self.alpha_.copy()
        self.branching_ratio_ = float(
            np.max(np.abs(np.linalg.eigvals(self.branching_matrix_)))
        )
        self.is_stable_ = bool(self.branching_ratio_ < 1.0)

        return self

    # ------------------------------------------------------------------
    # Intensité et compensateur
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not hasattr(self, "mu_"):
            raise RuntimeError("Le modèle doit être fitté avant utilisation.")

    def intensity_at(self, grid):
        """
        Retourne lambda(t) pour chaque t de grid.

        Output :
            array shape (len(grid), d)
        """
        self._check_fitted()

        grid = np.asarray(grid, dtype=float).ravel()
        d = self.n_types

        out = np.zeros((len(grid), d), dtype=float)

        for idx, t in enumerate(grid):
            K = np.zeros((d, d), dtype=float)

            for j in range(d):
                mask = (self.types_ == j) & (self.times_ < t)
                past = self.times_[mask]
                marks = self.marks_[mask]

                if len(past) == 0:
                    continue

                rem = t - past

                K[:, j] = np.sum(
                    marks[None, :]
                    * self.beta_[:, [j]]
                    * np.exp(-self.beta_[:, [j]] * rem[None, :]),
                    axis=1,
                )

            out[idx, :] = self.mu_ + np.sum(self.alpha_ * K, axis=1)

        return out

    def cumulative_intensity_at(self, grid):
        """
        Retourne Lambda(t) pour chaque t de grid.

        Output :
            array shape (len(grid), d)
        """
        self._check_fitted()

        grid = np.asarray(grid, dtype=float).ravel()
        d = self.n_types

        out = np.zeros((len(grid), d), dtype=float)

        for idx, t in enumerate(grid):
            Kcum = np.zeros((d, d), dtype=float)

            for j in range(d):
                mask = (self.types_ == j) & (self.times_ < t)
                past = self.times_[mask]
                marks = self.marks_[mask]

                if len(past) == 0:
                    continue

                rem = t - past

                Kcum[:, j] = np.sum(
                    marks[None, :]
                    * (1.0 - np.exp(-self.beta_[:, [j]] * rem[None, :])),
                    axis=1,
                )

            out[idx, :] = self.mu_ * t + np.sum(self.alpha_ * Kcum, axis=1)

        return out

    def counting_process_at(self, grid):
        """
        Retourne N_i(t) pour chaque dimension i.

        Output :
            array shape (len(grid), d)
        """
        self._check_fitted()

        grid = np.asarray(grid, dtype=float).ravel()
        d = self.n_types

        out = np.zeros((len(grid), d), dtype=float)

        for j in range(d):
            tj = self.times_[self.types_ == j]
            out[:, j] = np.searchsorted(tj, grid, side="right")

        return out

    def validation_processes(self, n_grid=1000, grid=None):
        self._check_fitted()

        if grid is None:
            grid = np.unique(
                np.r_[np.linspace(0.0, self.T_, int(n_grid)), self.times_]
            )
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

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _dims(self, dims):
        if dims is None:
            return list(range(self.n_types))
        if np.ndim(dims) == 0:
            return [int(dims)]
        return [int(x) for x in dims]

    def plot_intensity(self, n_grid=1000, grid=None, dims=None, total=False):
        import matplotlib.pyplot as plt

        data = self.validation_processes(n_grid=n_grid, grid=grid)

        fig, ax = plt.subplots(figsize=(10, 4))

        if total:
            ax.plot(data["time"], data["lambda_total"], label=r"$\hat{\lambda}_{total}(t)$")
        else:
            for j in self._dims(dims):
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
        n_grid=1000,
        grid=None,
        dims=None,
        total=False,
        show_counting=True,
    ):
        import matplotlib.pyplot as plt

        data = self.validation_processes(n_grid=n_grid, grid=grid)

        fig, ax = plt.subplots(figsize=(10, 4))

        if total:
            ax.plot(data["time"], data["Lambda_total"], label=r"$\hat{\Lambda}_{total}(t)$")
            if show_counting:
                ax.step(data["time"], data["N_total"], where="post", label=r"$N_{total}(t)$")
        else:
            for j in self._dims(dims):
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

    def plot_N_vs_Lambda(self, n_grid=1000, grid=None, dims=None, total=False):
        import matplotlib.pyplot as plt

        data = self.validation_processes(n_grid=n_grid, grid=grid)

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
            for j in self._dims(dims):
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

    def plot_martingale(self, n_grid=1000, grid=None, dims=None, total=False):
        import matplotlib.pyplot as plt

        data = self.validation_processes(n_grid=n_grid, grid=grid)

        fig, ax = plt.subplots(figsize=(10, 4))

        if total:
            ax.step(data["time"], data["M_total"], where="post", label=r"$M_{total}(t)$")
        else:
            for j in self._dims(dims):
                ax.step(data["time"], data["M"][:, j], where="post", label=fr"$M_{j}(t)$")

        ax.axhline(0.0, linestyle="--")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$N(t)-\hat{\Lambda}(t)$")
        ax.set_title("Martingales compensées")
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


# UNIVARIATE
import numpy as np
from scipy.optimize import minimize


class SimpleUnivariateHawkesMLE:
    """
    Hawkes univarié exponentiel, une seule réalisation.

    Sans marks :
        lambda(t) = mu + alpha * sum_{t_k < t} beta * exp(-beta * (t - t_k))

    Avec marks :
        lambda(t) = mu + alpha * sum_{t_k < t} mark_k * beta * exp(-beta * (t - t_k))

    Paramètres estimés :
        mu    : baseline, strictement positif
        alpha : amplitude d'auto-excitation, positif
        beta  : vitesse de décroissance, strictement positif
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
    # Préparation des données
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

    def _pack(self, mu, alpha, beta):
        return np.array([mu, alpha, beta], dtype=float)

    def _unpack(self, theta):
        theta = np.asarray(theta, dtype=float).ravel()

        if len(theta) != 3:
            raise ValueError("theta doit être de longueur 3 : [mu, alpha, beta].")

        mu = float(theta[0])
        alpha = float(theta[1])
        beta = float(theta[2])

        return mu, alpha, beta

    def _bounds(self):
        return [
            (self.min_mu, None),
            (0.0, self.alpha_upper),
            (self.min_beta, self.beta_upper),
        ]

    def _initial_theta(self, times, T):
        n = len(times)

        mu0 = max(
            0.5 * n / max(T, 1e-12),
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
    # Negative log-vraisemblance + gradient
    # ------------------------------------------------------------------

    def neg_loglikelihood_with_grad(self, theta, times, T, marks=None):
        mu, alpha, beta = self._unpack(theta)

        times = np.asarray(times, dtype=float).ravel()

        if marks is None:
            marks = np.ones_like(times, dtype=float)
        else:
            marks = np.asarray(marks, dtype=float).ravel()

        if mu <= 0 or alpha < 0 or beta <= 0:
            return np.inf, np.zeros_like(theta)

        n = len(times)

        # r(t_k) = sum_{t_l < t_k} mark_l * beta * exp(-beta * (t_k - t_l))
        r = 0.0

        # q(t_k) = dérivée de r(t_k) par rapport à beta
        q = 0.0

        log_term = 0.0

        grad_mu_log = 0.0
        grad_alpha_log = 0.0
        grad_beta_log = 0.0

        for k in range(n):
            if k > 0:
                dt = times[k] - times[k - 1]
                prev_mark = marks[k - 1]

                decay = np.exp(-beta * dt)

                # Ajout de la contribution du dernier événement
                r_start = r + prev_mark * beta
                q_start = q + prev_mark

                # Décroissance jusqu'au timestamp courant
                r = decay * r_start
                q = decay * (q_start - dt * r_start)

            lam = mu + alpha * r

            if lam <= 0 or not np.isfinite(lam):
                return np.inf, np.zeros_like(theta)

            inv_lam = 1.0 / lam
            log_term += np.log(lam)

            # Gradient du terme log
            grad_mu_log += inv_lam
            grad_alpha_log += r * inv_lam
            grad_beta_log += alpha * q * inv_lam

        # Compensateur global
        rem = T - times

        if np.any(rem < 0):
            return np.inf, np.zeros_like(theta)

        exp_tail = np.exp(-beta * rem)

        S_alpha = np.sum(marks * (1.0 - exp_tail))
        S_beta = np.sum(marks * rem * exp_tail)

        compensator = mu * T + alpha * S_alpha

        grad_mu_comp = T
        grad_alpha_comp = S_alpha
        grad_beta_comp = alpha * S_beta

        loglik = log_term - compensator
        nll = -float(loglik)

        # Gradient de la négative log-vraisemblance
        grad_mu = grad_mu_comp - grad_mu_log
        grad_alpha = grad_alpha_comp - grad_alpha_log
        grad_beta = grad_beta_comp - grad_beta_log

        grad = self._pack(grad_mu, grad_alpha, grad_beta)

        return nll, grad

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, times, T, marks=None, x0=None):
        times, marks, T = self._prepare_data(times=times, marks=marks, T=T)

        if x0 is None:
            x0 = self._initial_theta(times, T)
        else:
            x0 = np.asarray(x0, dtype=float).ravel()

        result = minimize(
            fun=lambda th: self.neg_loglikelihood_with_grad(
                th,
                times=times,
                T=T,
                marks=marks,
            ),
            x0=x0,
            jac=True,
            bounds=self._bounds(),
            method="L-BFGS-B",
            options={
                "maxiter": self.max_iter,
                "ftol": self.tol,
            },
        )

        self.result_ = result
        self.success_ = bool(result.success)
        self.message_ = result.message
        self.n_iter_ = result.nit

        self.times_ = times
        self.marks_ = marks
        self.T_ = T

        self.mu_, self.alpha_, self.beta_ = self._unpack(result.x)
        self.baseline_ = self.mu_

        self.log_likelihood_ = -float(result.fun)

        mean_mark = float(np.mean(marks)) if len(marks) > 0 else 1.0
        self.branching_ratio_ = float(self.alpha_ * mean_mark)
        self.is_stable_ = bool(self.branching_ratio_ < 1.0)

        return self

    # ------------------------------------------------------------------
    # Intensité, compensateur, processus de comptage
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not hasattr(self, "mu_"):
            raise RuntimeError("Le modèle doit être fitté avant utilisation.")

    def intensity_at(self, grid):
        """
        Retourne lambda(t) pour chaque t dans grid.

        Output :
            array shape (len(grid),)
        """
        self._check_fitted()

        scalar_input = np.ndim(grid) == 0
        grid = np.array([float(grid)]) if scalar_input else np.asarray(grid, dtype=float).ravel()

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
                    past_marks
                    * self.beta_
                    * np.exp(-self.beta_ * rem)
                )

            out[idx] = self.mu_ + self.alpha_ * kernel

        return float(out[0]) if scalar_input else out

    def cumulative_intensity_at(self, grid):
        """
        Retourne Lambda(t) pour chaque t dans grid.

        Output :
            array shape (len(grid),)
        """
        self._check_fitted()

        scalar_input = np.ndim(grid) == 0
        grid = np.array([float(grid)]) if scalar_input else np.asarray(grid, dtype=float).ravel()

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
                    past_marks
                    * (1.0 - np.exp(-self.beta_ * rem))
                )

            out[idx] = self.mu_ * t + self.alpha_ * kernel_cum

        return float(out[0]) if scalar_input else out

    def counting_process_at(self, grid):
        """
        Retourne N(t) pour chaque t dans grid.

        Output :
            array shape (len(grid),)
        """
        self._check_fitted()

        scalar_input = np.ndim(grid) == 0
        grid = np.array([float(grid)]) if scalar_input else np.asarray(grid, dtype=float).ravel()

        N = np.searchsorted(self.times_, grid, side="right").astype(float)

        return float(N[0]) if scalar_input else N

    def validation_processes(self, n_grid=1000, grid=None):
        self._check_fitted()

        if grid is None:
            grid = np.unique(
                np.r_[np.linspace(0.0, self.T_, int(n_grid)), self.times_]
            )
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
    # Résidus de time-rescaling
    # ------------------------------------------------------------------

    def time_rescaling_residuals(self):
        self._check_fitted()

        if len(self.times_) == 0:
            return {
                "tau": np.array([]),
                "uniform": np.array([]),
                "event_times": np.array([]),
            }

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
    # Plots
    # ------------------------------------------------------------------

    def plot_intensity(self, n_grid=1000, grid=None):
        import matplotlib.pyplot as plt

        data = self.validation_processes(n_grid=n_grid, grid=grid)

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(data["time"], data["lambda"], label=r"$\hat{\lambda}(t)$")

        ax.set_xlabel("t")
        ax.set_ylabel("Intensité")
        ax.set_title("Intensité estimée")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()
        return fig, ax

    def plot_cumulative_intensity(self, n_grid=1000, grid=None, show_counting=True):
        import matplotlib.pyplot as plt

        data = self.validation_processes(n_grid=n_grid, grid=grid)

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

    def plot_N_vs_Lambda(self, n_grid=1000, grid=None):
        import matplotlib.pyplot as plt

        data = self.validation_processes(n_grid=n_grid, grid=grid)

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

    def plot_martingale(self, n_grid=1000, grid=None):
        import matplotlib.pyplot as plt

        data = self.validation_processes(n_grid=n_grid, grid=grid)

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