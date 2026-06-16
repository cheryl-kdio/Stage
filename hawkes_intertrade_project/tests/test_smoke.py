import numpy as np
import pandas as pd

from hawkes_intertrade import (
    LogACDVolumeMLE,
    HawkesExpFixedDecayMLE,
    add_volume_buckets,
    build_side_volume_price_streams,
    prepare_trade_dataframe,
    HawkesExpMarkedFreeDecayMLE
)


def test_hawkes_fixed_smoke():
    mu = np.array([0.2, 0.15])
    alpha = np.array([[0.1, 0.05], [0.04, 0.1]])
    beta = np.ones((2, 2))
    events = HawkesExpFixedDecayMLE.simulate(mu, alpha, beta, 30.0, seed=1)
    model = HawkesExpFixedDecayMLE(beta, max_iter=200).fit(events, end_times=30.0)
    assert model.baseline_.shape == (2,)
    assert model.adjacency_.shape == (2, 2)
    assert np.isfinite(model.log_likelihood_)


def test_data_and_acd_smoke():
    df = pd.DataFrame({
        "timestamp": np.arange(20, dtype=float),
        "price": 100 + np.cumsum(np.random.default_rng(1).normal(0, 0.01, 20)),
        "volume": np.arange(1, 21, dtype=float),
    })
    out = prepare_trade_dataframe(df)
    out = add_volume_buckets(out, n_buckets=3)
    events, names = build_side_volume_price_streams(out)
    assert len(events) == len(names)
    valid = out["duration"].notna()
    acd = LogACDVolumeMLE(max_iter=100).fit(out.loc[valid, "duration"], out.loc[valid, "log_volume_z"])
    assert np.isfinite(acd.log_likelihood_)


buy_times = np.array([0.10, 0.50, 1.40, 2.10, 2.80])
buy_volumes = np.array([100, 500, 200, 900, 300])

sell_times = np.array([0.20, 0.80, 1.70, 2.50, 2.90])
sell_volumes = np.array([150, 300, 1000, 250, 600])

events = [
    buy_times,
    sell_times,
]

marks = [
    buy_volumes,
    sell_volumes,
]

model = HawkesExpMarkedFreeDecayMLE(
    decays_init=np.array([
        [1.0, 1.0],
        [1.0, 1.0],
    ]),
    decay_upper=10.0,
    alpha_upper=0.99,
    eta_bounds=(-3.0, 3.0),
    alpha_l2=1e-6,
    beta_l2=1e-6,
    eta_l2=1e-4,
    n_starts=10,
    random_state=123,
)

model.fit(
    events=events,
    marks=marks,
    end_time=3.0,
)

params = model.get_params()

print("mu:")
print(params["baseline"])

print("alpha:")
print(params["adjacency"])

print("beta:")
print(params["decays"])

print("eta volume:")
print(params["mark_eta"])

print("matrice de branchement effective:")
print(params["branching_matrix"])

print("rayon spectral:")
print(params["spectral_radius"])