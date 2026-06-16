"""Minimal exploratory backtest helpers.

These functions are intentionally simple. They are useful for prototypes, not as
production-grade high-frequency simulation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .signal import activity_direction_score


def threshold_direction_backtest(
    df: pd.DataFrame,
    model,
    events: Sequence[Sequence[float]],
    names: Sequence[str],
    horizon: float,
    direction_threshold: float = 0.25,
    activity_threshold: float = 0.05,
    t_col: str = "t",
    price_col: str = "price",
    fee_per_trade: float = 0.0,
) -> pd.DataFrame:
    """Naive event-time strategy using Hawkes direction/activity scores.

    At each trade time t, compute a signal using only events before t, then hold
    until the first observation at or after t+horizon. PnL is signed price change
    minus a flat fee. This ignores queue, spread and latency unless encoded by
    the user in fee_per_trade or prices.
    """
    rows = []
    times = df[t_col].to_numpy(dtype=float)
    prices = df[price_col].to_numpy(dtype=float)

    for idx in range(len(df) - 1):
        t = times[idx]
        exit_idx = int(np.searchsorted(times, t + horizon, side="left"))
        if exit_idx >= len(df):
            break
        sc = activity_direction_score(model, events, names, t, horizon)
        direction = sc["direction"]
        activity = sc["activity"]
        position = 0
        if activity >= activity_threshold and direction > direction_threshold:
            position = 1
        elif activity >= activity_threshold and direction < -direction_threshold:
            position = -1
        pnl = position * (prices[exit_idx] - prices[idx])
        if position != 0:
            pnl -= fee_per_trade
        rows.append({
            "t": t,
            "exit_t": times[exit_idx],
            "price": prices[idx],
            "exit_price": prices[exit_idx],
            "direction": direction,
            "activity": activity,
            "position": position,
            "pnl": pnl,
        })
    return pd.DataFrame(rows)
