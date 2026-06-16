"""Synthetic Hawkes fit example."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from hawkes_intertrade import HawkesExpFixedDecayMLE, HawkesExpFreeDecayMLE


if __name__ == "__main__":
    mu_true = np.array([0.20, 0.15])
    alpha_true = np.array([
        [0.25, 0.10],
        [0.08, 0.22],
    ])
    beta_true = np.array([
        [1.50, 0.80],
        [1.20, 1.00],
    ])
    T = 200.0

    events = HawkesExpFixedDecayMLE.simulate(mu_true, alpha_true, beta_true, T, seed=123)
    print("Counts per node:", [len(x) for x in events])

    fixed = HawkesExpFixedDecayMLE(decays=beta_true, alpha_upper=0.99, max_iter=1000)
    fixed.fit(events, end_times=T)
    print("\nFixed decay fit")
    print("success:", fixed.success_)
    print("baseline:\n", fixed.baseline_)
    print("adjacency:\n", fixed.adjacency_)
    print("loglik:", fixed.log_likelihood_)

    free = HawkesExpFreeDecayMLE(
        decays_init=1.0,
        alpha_upper=0.99,
        decay_upper=5.0,
        n_starts=2,
        max_iter=1000,
        random_state=42,
    )
    free.fit(events, end_times=T)
    print("\nFree decay fit")
    print("success:", free.success_)
    print("baseline:\n", free.baseline_)
    print("adjacency:\n", free.adjacency_)
    print("decays:\n", free.decays_)
    print("loglik:", free.log_likelihood_)
