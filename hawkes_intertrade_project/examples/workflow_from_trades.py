"""End-to-end workflow from a synthetic trade table."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from hawkes_intertrade import (
    LogACDVolumeMLE,
    HawkesExpFixedDecayMLE,
    add_volume_buckets,
    build_side_volume_price_streams,
    prepare_trade_dataframe,
    activity_direction_score,
)
from hawkes_intertrade.backtest import threshold_direction_backtest


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    n = 800

    # Synthetic trades: irregular timestamps, random-walk-like prices and lognormal volumes.
    durations = rng.exponential(scale=0.8, size=n)
    t = np.cumsum(durations)
    volume = rng.lognormal(mean=2.5, sigma=0.7, size=n)
    signs = rng.choice([-1, 1], size=n, p=[0.49, 0.51])
    price = 100.0 + np.cumsum(0.01 * signs + rng.normal(0, 0.02, size=n))

    raw = pd.DataFrame({"timestamp": t, "price": price, "volume": volume})
    df = prepare_trade_dataframe(raw)
    df = add_volume_buckets(df, n_buckets=3)

    # ACD benchmark: use valid durations and standardized log-volume.
    valid = df["duration"].notna()
    acd = LogACDVolumeMLE().fit(
        durations=df.loc[valid, "duration"].to_numpy(),
        marks=df.loc[valid, "log_volume_z"].to_numpy(),
    )
    print("ACD summary:", acd.summary())

    # Hawkes streams: buy/sell by volume bucket + price up/down events.
    events, names = build_side_volume_price_streams(df)
    T = float(df["t"].iloc[-1] + 1.0)
    d = len(events)

    # Keep the example fast: fixed homogeneous decay.
    decays = np.full((d, d), 1.0)
    model = HawkesExpFixedDecayMLE(decays=decays, alpha_upper=0.99, max_iter=500)
    model.fit(events, end_times=T)

    print("\nHawkes nodes:", names)
    print("success:", model.success_)
    print("baseline shape:", model.baseline_.shape)
    print("adjacency shape:", model.adjacency_.shape)
    print("spectral radius:", model.spectral_radius_)

    # Score at an arbitrary time.
    t0 = float(df["t"].iloc[500])
    score = activity_direction_score(model, events, names, t=t0, horizon=2.0)
    print("\nSignal at t0:", {k: v for k, v in score.items() if k != "expected_counts"})

    bt = threshold_direction_backtest(
        df=df,
        model=model,
        events=events,
        names=names,
        horizon=2.0,
        direction_threshold=0.02,
        activity_threshold=0.01,
        fee_per_trade=0.0,
    )
    print("\nToy backtest summary")
    print("n trades:", int((bt["position"] != 0).sum()))
    print("cum pnl:", float(bt["pnl"].sum()))
