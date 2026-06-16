"""
hawkes_multiday_formatting.py
================================

Utilities to prepare multi-day trade data for Hawkes-process estimation.

The central idea is:

    Fit one Hawkes model on several days by treating each day as an
    independent realization of the same process.

Mathematically, if theta denotes the common model parameters, the training
objective is:

    loglik_train(theta) = sum_day loglik_day(theta)

This is not the same thing as estimating one independent Hawkes model per day.
It is also not the same thing as naively concatenating days with overnight gaps.

Expected Hawkes input format
----------------------------

For a univariate Hawkes on several sessions:

    events = [
        [np.array([...])],  # day 1, dimension 0
        [np.array([...])],  # day 2, dimension 0
        [np.array([...])],  # day 3, dimension 0
    ]

For a multivariate Hawkes with d dimensions:

    events[day][node] = np.ndarray of event timestamps relative to day start

For example, with BUY and SELL streams:

    events = [
        [buy_times_day_1, sell_times_day_1],
        [buy_times_day_2, sell_times_day_2],
        ...
    ]

For a marked Hawkes, marks have exactly the same nested structure:

    marks[day][node] = np.ndarray of marks aligned with events[day][node]

Example:

    events, end_times, node_order, sessions, marks = build_hawkes_realizations_from_trades(
        train_df,
        timestamp_col="timestamp",
        node_col="side",
        mark_col="volume",
        node_order=["BUY", "SELL"],
    )

    # Unmarked model supporting multiple realizations:
    model.fit(events, end_times=end_times)

    # Marked model supporting multiple realizations:
    model.fit(events, marks=marks, end_times=end_times)

Important convention
--------------------

Timestamps are reset to zero at the beginning of each session. This is usually
better for markets with a daily close because it avoids treating the overnight
period as a long period of no events.

If your market is 24/7, such as crypto, a continuous timestamp representation
can be more natural. In that case, use these helpers with a different session
scheme or write a continuous split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, Sequence[float]]


@dataclass(frozen=True)
class HawkesRealizationBundle:
    """
    Container returned by build_hawkes_realizations_from_trades.

    Attributes
    ----------
    events:
        Nested list such that events[session_index][node_index] is a 1D array
        of timestamps relative to the session start.

    end_times:
        Observation horizon for each session, in seconds by default.

    node_order:
        Ordered list of Hawkes dimensions. Example: ["BUY", "SELL"].

    sessions:
        Ordered list of sessions included in the bundle.

    marks:
        Nested list with the same shape as events when mark_col is provided.
        Otherwise None.
    """

    events: List[List[np.ndarray]]
    end_times: np.ndarray
    node_order: List[Any]
    sessions: List[Any]
    marks: Optional[List[List[np.ndarray]]] = None


@dataclass(frozen=True)
class SplitBundle:
    """
    Container for a train/validation/test split.

    The DataFrames are sorted chronologically. The session arrays tell you which
    sessions are inside each split.
    """

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    train_sessions: pd.Index
    valid_sessions: pd.Index
    test_sessions: pd.Index


@dataclass(frozen=True)
class WalkForwardSplit:
    """
    One walk-forward split.

    warmup is optional context before the test window. It is useful when scoring
    a Hawkes model because the intensity at the beginning of the test depends on
    recent past events. If you reset the process at each session, warmup may be
    unnecessary. If you want continuity between adjacent sessions, use warmup.
    """

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    warmup: pd.DataFrame
    train_sessions: pd.Index
    valid_sessions: pd.Index
    test_sessions: pd.Index
    warmup_sessions: pd.Index


def _ensure_datetime(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """Return a copy with timestamp_col converted to pandas datetime."""
    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col])
    return out


def _add_session_column(
    df: pd.DataFrame,
    timestamp_col: str,
    session_col: Optional[str],
) -> Tuple[pd.DataFrame, str, bool]:
    """
    Ensure the DataFrame has a session column.

    If session_col is None, a temporary column named "_hawkes_session" is built
    from the calendar date of timestamp_col.
    """
    out = df.copy()

    if session_col is None:
        out["_hawkes_session"] = out[timestamp_col].dt.date
        return out, "_hawkes_session", True

    if session_col not in out.columns:
        raise KeyError(f"session_col={session_col!r} is not in df.columns")

    return out, session_col, False


def _sorted_sessions(df: pd.DataFrame, session_col: str) -> pd.Index:
    """Return sorted unique non-null sessions."""
    return pd.Index(df[session_col].dropna().unique()).sort_values()


def _default_session_bounds(
    g_day: pd.DataFrame,
    timestamp_col: str,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Default session bounds: first and last timestamp in the group.

    This is convenient, but for likelihood estimation it is better to pass
    official market open and close times through session_start_times and
    session_end_times.
    """
    return g_day[timestamp_col].min(), g_day[timestamp_col].max()


def build_hawkes_realizations_from_trades(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    node_col: str = "side",
    mark_col: Optional[str] = None,
    session_col: Optional[str] = None,
    node_order: Optional[Sequence[Any]] = None,
    session_start_times: Optional[Dict[Any, Any]] = None,
    session_end_times: Optional[Dict[Any, Any]] = None,
    drop_empty_sessions: bool = True,
) -> HawkesRealizationBundle:
    """
    Convert a trade table into multi-realization Hawkes arrays.

    Parameters
    ----------
    df:
        Trade table. It must contain at least timestamp_col and node_col. If
        mark_col is not None, it must also contain mark_col.

    timestamp_col:
        Datetime column containing trade timestamps.

    node_col:
        Column defining the Hawkes dimension. Examples: "side", "event_type",
        "stream". For a univariate Hawkes, create a constant column such as:

            df["event_type"] = "TRADE"

    mark_col:
        Optional mark column, typically "volume". The returned marks have the
        same nested shape as the events.

    session_col:
        Optional session column. If None, the calendar date extracted from
        timestamp_col is used.

    node_order:
        Optional fixed order of dimensions. Example: ["BUY", "SELL"]. If None,
        node labels are sorted automatically.

    session_start_times, session_end_times:
        Optional dictionaries mapping each session id to official start/end
        timestamps. If omitted, the first and last observed timestamps of each
        session are used.

        For a regular equity session, it is better to pass official bounds so
        that end_time is the true observation horizon, not the time of the last
        observed trade.

    drop_empty_sessions:
        If True, sessions with no valid events after filtering are skipped.

    Returns
    -------
    HawkesRealizationBundle
        Contains events, end_times, node_order, sessions and marks.

    Notes
    -----
    The returned timestamps are in seconds relative to session start.

    General output shape:

        events[day_index][node_index]
        marks[day_index][node_index]
        end_times[day_index]

    This is the format needed to fit one Hawkes model on multiple days by
    summing the daily log-likelihoods.
    """
    required = {timestamp_col, node_col}
    if mark_col is not None:
        required.add(mark_col)

    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    work = _ensure_datetime(df, timestamp_col)
    work = work.sort_values(timestamp_col).reset_index(drop=True)
    work, session_col_used, is_temp_session = _add_session_column(
        work, timestamp_col, session_col
    )

    if node_order is None:
        node_order_list = sorted(work[node_col].dropna().unique().tolist())
    else:
        node_order_list = list(node_order)

    if len(node_order_list) == 0:
        raise ValueError("node_order is empty; no Hawkes dimension was found.")

    sessions_all = _sorted_sessions(work, session_col_used)

    events: List[List[np.ndarray]] = []
    marks: Optional[List[List[np.ndarray]]] = [] if mark_col is not None else None
    end_times: List[float] = []
    used_sessions: List[Any] = []

    for sess in sessions_all:
        g_day = work[work[session_col_used] == sess].copy()

        if len(g_day) == 0:
            continue

        if session_start_times is not None:
            if sess not in session_start_times:
                raise KeyError(f"Missing session_start_times for session {sess!r}")
            t0 = pd.Timestamp(session_start_times[sess])
        else:
            t0, _ = _default_session_bounds(g_day, timestamp_col)

        if session_end_times is not None:
            if sess not in session_end_times:
                raise KeyError(f"Missing session_end_times for session {sess!r}")
            t1 = pd.Timestamp(session_end_times[sess])
        else:
            _, t1 = _default_session_bounds(g_day, timestamp_col)

        T = float((t1 - t0).total_seconds())
        if not np.isfinite(T) or T <= 0:
            if drop_empty_sessions:
                continue
            raise ValueError(f"Non-positive session horizon for session {sess!r}")

        day_events: List[np.ndarray] = []
        day_marks: Optional[List[np.ndarray]] = [] if mark_col is not None else None
        total_events = 0

        for node in node_order_list:
            g_node = g_day[g_day[node_col] == node].copy()
            g_node = g_node.sort_values(timestamp_col)

            t_rel_all = (g_node[timestamp_col] - t0).dt.total_seconds().to_numpy(
                dtype=float
            )
            valid_mask = (t_rel_all >= 0.0) & (t_rel_all <= T)
            t_rel = t_rel_all[valid_mask]

            # Guarantee sorted floating arrays.
            order = np.argsort(t_rel)
            t_rel = t_rel[order]

            day_events.append(t_rel)
            total_events += len(t_rel)

            if mark_col is not None:
                raw_marks = g_node[mark_col].to_numpy(dtype=float)[valid_mask]
                raw_marks = raw_marks[order]
                if np.any(~np.isfinite(raw_marks)):
                    raise ValueError(
                        f"Non-finite marks in session {sess!r}, node {node!r}"
                    )
                if day_marks is None:
                    raise RuntimeError("Internal error: day_marks is None")
                day_marks.append(raw_marks)

        if total_events == 0 and drop_empty_sessions:
            continue

        events.append(day_events)
        if mark_col is not None:
            if marks is None or day_marks is None:
                raise RuntimeError("Internal error: marks/day_marks is None")
            marks.append(day_marks)
        end_times.append(T)
        used_sessions.append(sess)

    if len(events) == 0:
        raise ValueError("No valid sessions/events were produced.")

    if is_temp_session:
        work = work.drop(columns=[session_col_used])

    return HawkesRealizationBundle(
        events=events,
        end_times=np.asarray(end_times, dtype=float),
        node_order=node_order_list,
        sessions=used_sessions,
        marks=marks,
    )


def chronological_train_valid_test_split(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    train_frac: float = 0.60,
    valid_frac: float = 0.20,
    session_col: Optional[str] = None,
) -> SplitBundle:
    """
    Split a trade table chronologically by sessions.

    This is a simple fixed split:

        earliest sessions -> train
        next sessions     -> validation
        latest sessions   -> test

    Do not shuffle high-frequency trades for Hawkes estimation or backtesting.
    Shuffling creates look-ahead and destroys temporal dependence.
    """
    if not (0 < train_frac < 1):
        raise ValueError("train_frac must be between 0 and 1")
    if not (0 < valid_frac < 1):
        raise ValueError("valid_frac must be between 0 and 1")
    if train_frac + valid_frac >= 1:
        raise ValueError("train_frac + valid_frac must be below 1")

    work = _ensure_datetime(df, timestamp_col)
    work = work.sort_values(timestamp_col).reset_index(drop=True)
    work, session_col_used, is_temp_session = _add_session_column(
        work, timestamp_col, session_col
    )

    sessions = _sorted_sessions(work, session_col_used)
    n_sessions = len(sessions)
    if n_sessions < 3:
        raise ValueError("At least 3 sessions are needed for train/valid/test.")

    n_train = max(1, int(train_frac * n_sessions))
    n_valid = max(1, int(valid_frac * n_sessions))

    if n_train + n_valid >= n_sessions:
        raise ValueError("The split leaves no test session.")

    train_sessions = sessions[:n_train]
    valid_sessions = sessions[n_train : n_train + n_valid]
    test_sessions = sessions[n_train + n_valid :]

    train = work[work[session_col_used].isin(train_sessions)].copy()
    valid = work[work[session_col_used].isin(valid_sessions)].copy()
    test = work[work[session_col_used].isin(test_sessions)].copy()

    if is_temp_session:
        for x in (train, valid, test):
            x.drop(columns=[session_col_used], inplace=True)

    return SplitBundle(
        train=train,
        valid=valid,
        test=test,
        train_sessions=train_sessions,
        valid_sessions=valid_sessions,
        test_sessions=test_sessions,
    )


def walk_forward_session_splits(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    session_col: Optional[str] = None,
    train_sessions: int = 20,
    valid_sessions: int = 5,
    test_sessions: int = 1,
    step_sessions: int = 1,
    warmup_sessions: int = 1,
) -> List[WalkForwardSplit]:
    """
    Build walk-forward splits by sessions.

    Recommended first setup for a liquid instrument:

        train_sessions = 20 to 60
        valid_sessions = 5
        test_sessions  = 1
        step_sessions  = 1

    Each split uses one common Hawkes model on all training sessions. The test
    set is kept chronologically after train and validation.

    warmup contains sessions immediately before the test window. It can be used
    to initialize intensities before scoring or trading the test set.
    """
    for name, value in {
        "train_sessions": train_sessions,
        "valid_sessions": valid_sessions,
        "test_sessions": test_sessions,
        "step_sessions": step_sessions,
        "warmup_sessions": warmup_sessions,
    }.items():
        if int(value) < 0:
            raise ValueError(f"{name} must be non-negative")

    if train_sessions <= 0 or test_sessions <= 0 or step_sessions <= 0:
        raise ValueError("train_sessions, test_sessions and step_sessions must be positive")

    work = _ensure_datetime(df, timestamp_col)
    work = work.sort_values(timestamp_col).reset_index(drop=True)
    work, session_col_used, is_temp_session = _add_session_column(
        work, timestamp_col, session_col
    )

    sessions = _sorted_sessions(work, session_col_used)
    total_window = train_sessions + valid_sessions + test_sessions
    if len(sessions) < total_window:
        raise ValueError(
            f"Not enough sessions: got {len(sessions)}, need at least {total_window}."
        )

    splits: List[WalkForwardSplit] = []
    start = 0

    while start + total_window <= len(sessions):
        train_s = sessions[start : start + train_sessions]

        valid_start = start + train_sessions
        valid_end = valid_start + valid_sessions
        valid_s = sessions[valid_start:valid_end]

        test_start = valid_end
        test_end = test_start + test_sessions
        test_s = sessions[test_start:test_end]

        warmup_start = max(0, test_start - warmup_sessions)
        warmup_s = sessions[warmup_start:test_start]

        train = work[work[session_col_used].isin(train_s)].copy()
        valid = work[work[session_col_used].isin(valid_s)].copy()
        test = work[work[session_col_used].isin(test_s)].copy()
        warmup = work[work[session_col_used].isin(warmup_s)].copy()

        if is_temp_session:
            for x in (train, valid, test, warmup):
                x.drop(columns=[session_col_used], inplace=True)

        splits.append(
            WalkForwardSplit(
                train=train,
                valid=valid,
                test=test,
                warmup=warmup,
                train_sessions=train_s,
                valid_sessions=valid_s,
                test_sessions=test_s,
                warmup_sessions=warmup_s,
            )
        )

        start += step_sessions

    return splits


def describe_hawkes_realizations(bundle: HawkesRealizationBundle) -> pd.DataFrame:
    """
    Return a compact table describing events per session and per node.

    This is useful to check that the multi-day formatting is correct before
    fitting a Hawkes model.
    """
    rows: List[Dict[str, Any]] = []

    for day_idx, (sess, day_events, T) in enumerate(
        zip(bundle.sessions, bundle.events, bundle.end_times)
    ):
        for node_idx, node_name in enumerate(bundle.node_order):
            times = day_events[node_idx]
            n = len(times)
            rows.append(
                {
                    "day_index": day_idx,
                    "session": sess,
                    "node_index": node_idx,
                    "node": node_name,
                    "n_events": n,
                    "end_time": T,
                    "first_time": float(times[0]) if n > 0 else np.nan,
                    "last_time": float(times[-1]) if n > 0 else np.nan,
                    "event_rate": n / T if T > 0 else np.nan,
                }
            )

    return pd.DataFrame(rows)


def make_univariate_trade_column(
    df: pd.DataFrame,
    col_name: str = "event_type",
    value: str = "TRADE",
) -> pd.DataFrame:
    """
    Convenience helper to fit a univariate Hawkes on all trades.

    Example
    -------
    df_uni = make_univariate_trade_column(df)
    bundle = build_hawkes_realizations_from_trades(
        df_uni,
        timestamp_col="timestamp",
        node_col="event_type",
        node_order=["TRADE"],
    )
    """
    out = df.copy()
    out[col_name] = value
    return out


def build_regular_session_bounds(
    sessions: Iterable[Any],
    open_time: str = "09:30:00",
    close_time: str = "16:00:00",
) -> Tuple[Dict[Any, pd.Timestamp], Dict[Any, pd.Timestamp]]:
    """
    Build dictionaries of regular session open/close timestamps.

    This helper assumes session identifiers are date-like, e.g. datetime.date,
    pandas Timestamp, or strings parseable as dates.

    Example
    -------
    starts, ends = build_regular_session_bounds(
        sessions,
        open_time="09:30:00",
        close_time="16:00:00",
    )

    bundle = build_hawkes_realizations_from_trades(
        df,
        timestamp_col="timestamp",
        node_col="side",
        session_start_times=starts,
        session_end_times=ends,
    )
    """
    starts: Dict[Any, pd.Timestamp] = {}
    ends: Dict[Any, pd.Timestamp] = {}

    for sess in sessions:
        date = pd.Timestamp(sess).date()
        starts[sess] = pd.Timestamp(f"{date} {open_time}")
        ends[sess] = pd.Timestamp(f"{date} {close_time}")

    return starts, ends


def example_usage_text() -> str:
    """
    Return a short usage example as a string.

    This is intentionally plain text so it can be printed in notebooks or logs.
    """
    return """
Example usage
-------------

# Fixed split
split = chronological_train_valid_test_split(
    df,
    timestamp_col="timestamp",
    train_frac=0.60,
    valid_frac=0.20,
)

# Build one Hawkes training dataset from all train days
bundle = build_hawkes_realizations_from_trades(
    split.train,
    timestamp_col="timestamp",
    node_col="side",
    mark_col="volume",
    node_order=["BUY", "SELL"],
)

print(describe_hawkes_realizations(bundle))

# For an unmarked multi-realization model:
# model.fit(bundle.events, end_times=bundle.end_times)

# For a marked multi-realization model:
# model.fit(bundle.events, marks=bundle.marks, end_times=bundle.end_times)

# Walk-forward splits
splits = walk_forward_session_splits(
    df,
    timestamp_col="timestamp",
    train_sessions=20,
    valid_sessions=5,
    test_sessions=1,
    step_sessions=1,
    warmup_sessions=1,
)

for wf in splits:
    train_bundle = build_hawkes_realizations_from_trades(
        wf.train,
        timestamp_col="timestamp",
        node_col="side",
        mark_col="volume",
        node_order=["BUY", "SELL"],
    )
    # Fit one model on train_bundle.events.
    # Select hyperparameters on wf.valid.
    # Evaluate on wf.test.
""".strip()


def _make_synthetic_trade_data(
    n_days: int = 8,
    trades_per_day: int = 100,
    seed: int = 123,
) -> pd.DataFrame:
    """Small synthetic dataset for the smoke test."""
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    base_date = pd.Timestamp("2026-01-05")

    for d in range(n_days):
        day = base_date + pd.Timedelta(days=d)
        open_ts = pd.Timestamp(f"{day.date()} 09:30:00")
        close_seconds = 6.5 * 3600

        # Uniform event times in the session for a simple smoke test.
        seconds = np.sort(rng.uniform(0, close_seconds, size=trades_per_day))
        sides = rng.choice(["BUY", "SELL"], size=trades_per_day)
        volumes = rng.lognormal(mean=5.0, sigma=0.8, size=trades_per_day)
        price = 100.0 + np.cumsum(rng.normal(0.0, 0.01, size=trades_per_day))

        for s, side, volume, p in zip(seconds, sides, volumes, price):
            rows.append(
                {
                    "timestamp": open_ts + pd.Timedelta(seconds=float(s)),
                    "side": side,
                    "volume": float(volume),
                    "price": float(p),
                }
            )

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def _smoke_test() -> None:
    """Run a minimal validation of all helpers."""
    df = _make_synthetic_trade_data()

    split = chronological_train_valid_test_split(
        df,
        timestamp_col="timestamp",
        train_frac=0.5,
        valid_frac=0.25,
    )
    assert len(split.train) > 0
    assert len(split.valid) > 0
    assert len(split.test) > 0

    bundle = build_hawkes_realizations_from_trades(
        split.train,
        timestamp_col="timestamp",
        node_col="side",
        mark_col="volume",
        node_order=["BUY", "SELL"],
    )

    assert len(bundle.events) == len(bundle.end_times)
    assert bundle.marks is not None
    assert len(bundle.events) == len(bundle.marks)
    assert bundle.node_order == ["BUY", "SELL"]

    for day_events, day_marks in zip(bundle.events, bundle.marks):
        assert len(day_events) == 2
        assert len(day_marks) == 2
        for ev, mk in zip(day_events, day_marks):
            assert len(ev) == len(mk)
            assert np.all(np.diff(ev) >= 0)

    desc = describe_hawkes_realizations(bundle)
    assert not desc.empty
    assert set(["session", "node", "n_events", "end_time"]).issubset(desc.columns)

    wf_splits = walk_forward_session_splits(
        df,
        timestamp_col="timestamp",
        train_sessions=3,
        valid_sessions=1,
        test_sessions=1,
        step_sessions=1,
        warmup_sessions=1,
    )
    assert len(wf_splits) > 0

    df_uni = make_univariate_trade_column(df)
    uni_bundle = build_hawkes_realizations_from_trades(
        df_uni,
        timestamp_col="timestamp",
        node_col="event_type",
        node_order=["TRADE"],
    )
    assert uni_bundle.node_order == ["TRADE"]
    assert all(len(day) == 1 for day in uni_bundle.events)


if __name__ == "__main__":
    _smoke_test()
    print("Smoke test OK")
    print()
    print(example_usage_text())
