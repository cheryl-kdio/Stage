"""Data preparation utilities for trade-level Hawkes experiments."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


def _to_seconds_from_start(series: pd.Series) -> np.ndarray:
    """Convert timestamps or numeric times to seconds from first observation."""
    if np.issubdtype(series.dtype, np.datetime64):
        ts = pd.to_datetime(series)
        return (ts - ts.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    return values - values[0]


def infer_trade_side_tick_rule(price: Sequence[float]) -> np.ndarray:
    """Infer trade side with the tick rule.

    Returns an array in {-1, +1}. A positive sign means buy-initiated and a
    negative sign means sell-initiated. Zero price changes inherit the previous
    non-zero sign.
    """
    p = np.asarray(price, dtype=float).ravel()
    if p.size == 0:
        return np.array([], dtype=int)
    if np.any(~np.isfinite(p)):
        raise ValueError("price contains non-finite values.")

    dp = np.diff(p, prepend=p[0])
    side = np.zeros(p.size, dtype=int)
    last = 1
    for i, x in enumerate(dp):
        if x > 0:
            last = 1
        elif x < 0:
            last = -1
        side[i] = last
    return side


def prepare_trade_dataframe(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    price_col: str = "price",
    volume_col: str = "volume",
    side_col: str | None = None,
) -> pd.DataFrame:
    """Create canonical features from a raw trade DataFrame.

    Output columns include:
        - t: seconds from first trade
        - duration: intertrade duration
        - next_duration: duration after the current trade
        - log_volume
        - log_volume_z
        - side_sign in {-1, +1}
        - price_delta
        - price_up_event, price_down_event
    """
    required = {timestamp_col, price_col, volume_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out = out.sort_values(timestamp_col).reset_index(drop=True)
    out["t"] = _to_seconds_from_start(out[timestamp_col])
    out["price"] = pd.to_numeric(out[price_col], errors="raise").astype(float)
    out["volume"] = pd.to_numeric(out[volume_col], errors="raise").astype(float)

    if np.any(out["volume"].to_numpy() < 0):
        raise ValueError("volume must be non-negative.")

    out["duration"] = out["t"].diff()
    out.loc[out["duration"] <= 0, "duration"] = np.nan
    out["next_duration"] = out["duration"].shift(-1)

    out["log_volume"] = np.log1p(out["volume"])
    vol_std = out["log_volume"].std(ddof=0)
    if vol_std == 0 or not np.isfinite(vol_std):
        out["log_volume_z"] = 0.0
    else:
        out["log_volume_z"] = (out["log_volume"] - out["log_volume"].mean()) / vol_std

    if side_col is not None and side_col in out.columns:
        raw_side = out[side_col]
        if raw_side.dtype == object:
            mapped = raw_side.astype(str).str.lower().map(
                {"buy": 1, "b": 1, "+": 1, "1": 1, "sell": -1, "s": -1, "-": -1, "-1": -1}
            )
            if mapped.isna().any():
                raise ValueError("side_col contains values that cannot be mapped to buy/sell.")
            out["side_sign"] = mapped.astype(int)
        else:
            side = pd.to_numeric(raw_side, errors="raise").to_numpy(dtype=float)
            out["side_sign"] = np.where(side >= 0, 1, -1)
    else:
        out["side_sign"] = infer_trade_side_tick_rule(out["price"].to_numpy())

    out["price_delta"] = out["price"].diff().fillna(0.0)
    out["price_up_event"] = out["price_delta"] > 0
    out["price_down_event"] = out["price_delta"] < 0

    return out


def add_volume_buckets(
    df: pd.DataFrame,
    volume_col: str = "volume",
    n_buckets: int = 4,
    labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Add a volume_bucket column based on empirical quantiles."""
    if n_buckets < 2:
        raise ValueError("n_buckets must be at least 2.")
    out = df.copy()
    if labels is None:
        labels = [f"q{i}" for i in range(n_buckets)]
    if len(labels) != n_buckets:
        raise ValueError("labels must have length n_buckets.")

    # duplicates='drop' handles constant or very discrete volumes.
    bucket = pd.qcut(out[volume_col], q=n_buckets, labels=False, duplicates="drop")
    n_actual = int(pd.Series(bucket).dropna().max() + 1) if pd.Series(bucket).notna().any() else 1
    actual_labels = list(labels[:n_actual])
    out["volume_bucket_id"] = bucket.fillna(0).astype(int)
    out["volume_bucket"] = out["volume_bucket_id"].map(lambda x: actual_labels[min(int(x), n_actual - 1)])
    return out


def build_volume_bucket_streams(
    df: pd.DataFrame,
    time_col: str = "t",
    bucket_col: str = "volume_bucket",
) -> Tuple[List[np.ndarray], List[str]]:
    """Build one event stream per volume bucket."""
    if bucket_col not in df.columns:
        raise ValueError(f"{bucket_col} is missing. Call add_volume_buckets first.")
    names = list(pd.Index(df[bucket_col].astype(str).unique()).sort_values())
    streams = []
    for name in names:
        streams.append(df.loc[df[bucket_col].astype(str) == name, time_col].to_numpy(dtype=float))
    return streams, names


def build_side_volume_price_streams(
    df: pd.DataFrame,
    time_col: str = "t",
    side_col: str = "side_sign",
    bucket_col: str = "volume_bucket",
    include_price: bool = True,
) -> Tuple[List[np.ndarray], List[str]]:
    """Build multivariate streams: buy/sell by volume bucket plus price events.

    Names are of the form B_q0, S_q0, ..., P_UP, P_DOWN.
    """
    if bucket_col not in df.columns:
        raise ValueError(f"{bucket_col} is missing. Call add_volume_buckets first.")
    if side_col not in df.columns:
        raise ValueError(f"{side_col} is missing. Call prepare_trade_dataframe first.")

    streams: List[np.ndarray] = []
    names: List[str] = []
    bucket_names = list(pd.Index(df[bucket_col].astype(str).unique()).sort_values())

    for prefix, side_value in [("B", 1), ("S", -1)]:
        for bucket in bucket_names:
            mask = (df[side_col].to_numpy(dtype=int) == side_value) & (df[bucket_col].astype(str).to_numpy() == bucket)
            streams.append(df.loc[mask, time_col].to_numpy(dtype=float))
            names.append(f"{prefix}_{bucket}")

    if include_price:
        streams.append(df.loc[df["price_up_event"], time_col].to_numpy(dtype=float))
        names.append("P_UP")
        streams.append(df.loc[df["price_down_event"], time_col].to_numpy(dtype=float))
        names.append("P_DOWN")

    return streams, names
