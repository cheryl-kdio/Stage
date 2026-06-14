"""
Usage examples for hawkes_estimators.py.

Run with:
    python hawkes_usage_examples.py
"""

import numpy as np
from hawkes_estimators import (
    simulate_univariate_exp_hawkes,
    simulate_multivariate_exp_hawkes,
    UnivariateHawkesExpMLE,
    UnivariateHawkesExpEM,
    UnivariateHawkesNonparamEM,
    HawkesL2ContrastEstimator,
    UnivariateWienerHopfEstimator,
    MultivariateHawkesExpMLE,
    MultivariateHawkesExpEM,
    MultivariateHawkesNonparamEM,
    MultivariateWienerHopfEstimator,
)


def univariate_demo():
    print("\n=== Univariate demo ===")
    T = 100.0
    times = simulate_univariate_exp_hawkes(mu=0.8, eta=0.35, beta=2.0, T=T, seed=123)
    print(f"Number of events: {len(times)}")

    # 1) Parametric MLE with convention phi(t)=alpha exp(-beta t).
    mle = UnivariateHawkesExpMLE(T=T).fit(times)
    print("Parametric MLE:", mle.params_)

    # 2) Parametric EM with convention phi(t)=eta*beta exp(-beta t).
    em = UnivariateHawkesExpEM(T=T, beta_fixed=2.0, max_iter=50).fit(times)
    print("Parametric EM with fixed beta:", em.params_)

    # 3) Non-parametric EM with histogram kernel.
    bins = np.linspace(0.0, 5.0, 21)
    npm = UnivariateHawkesNonparamEM(bin_edges=bins, T=T, max_iter=30, smooth_penalty=0.2).fit(times)
    print("Non-parametric EM: mu=", npm.params_["mu"], "kernel mass=", npm.params_["kernel_mass"])

    # 4) L2 contrast estimator. Works with univariate data when marks=None.
    l2 = HawkesL2ContrastEstimator(bin_edges=bins, T=T, grid_size=500, ridge=1e-4).fit(times)
    print("L2 contrast: mu=", l2.params_["mu"], "branching=", l2.params_["branching_matrix"])

    # 5) Wiener-Hopf estimator.
    wh = UnivariateWienerHopfEstimator(max_lag=5.0, n_bins=20, T=T, clip_negative=True).fit(times)
    print("Wiener-Hopf: mu=", wh.params_["mu"], "kernel mass=", wh.params_["kernel_mass"])


def multivariate_demo():
    print("\n=== Multivariate demo ===")
    T = 100.0
    mu = np.array([0.5, 0.4])
    eta = np.array([[0.25, 0.08],
                    [0.05, 0.20]])
    beta = 2.0
    times, marks = simulate_multivariate_exp_hawkes(mu=mu, eta=eta, beta=beta, T=T, seed=456)
    print(f"Number of events: {len(times)}")
    print("Counts by dimension:", np.bincount(marks, minlength=2))

    # 1) Parametric multivariate MLE with fixed beta.
    mle = MultivariateHawkesExpMLE(beta=beta, n_dims=2, T=T, ridge=1e-5).fit(times, marks)
    print("Multivariate MLE mu:\n", mle.params_["mu"])
    print("Multivariate MLE eta:\n", mle.params_["eta"])

    # 2) Parametric multivariate EM with fixed beta.
    em = MultivariateHawkesExpEM(beta=beta, n_dims=2, T=T, max_iter=30).fit(times, marks)
    print("Multivariate EM mu:\n", em.params_["mu"])
    print("Multivariate EM eta:\n", em.params_["eta"])

    # 3) Non-parametric multivariate EM.
    bins = np.linspace(0.0, 5.0, 21)
    npm = MultivariateHawkesNonparamEM(bin_edges=bins, n_dims=2, T=T, max_iter=20).fit(times, marks)
    print("Multivariate non-param EM mu:\n", npm.params_["mu"])
    print("Multivariate non-param EM branching matrix:\n", npm.params_["branching_matrix"])

    # 4) L2 contrast estimator.
    l2 = HawkesL2ContrastEstimator(bin_edges=bins, n_dims=2, T=T, grid_size=500, ridge=1e-4).fit(times, marks)
    print("Multivariate L2 mu:\n", l2.params_["mu"])
    print("Multivariate L2 branching matrix:\n", l2.params_["branching_matrix"])

    # 5) Wiener-Hopf estimator.
    wh = MultivariateWienerHopfEstimator(max_lag=5.0, n_bins=20, n_dims=2, T=T, clip_negative=True).fit(times, marks)
    print("Multivariate Wiener-Hopf mu:\n", wh.params_["mu"])
    print("Multivariate Wiener-Hopf branching matrix:\n", wh.params_["branching_matrix"])


if __name__ == "__main__":
    univariate_demo()
    multivariate_demo()
