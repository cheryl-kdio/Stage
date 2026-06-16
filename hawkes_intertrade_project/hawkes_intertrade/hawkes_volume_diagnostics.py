"""
hawkes_volume_diagnostics.py

Diagnostics descriptifs pour tester si le volume agit plutôt sur :

1. l'amplitude du noyau Hawkes ;
2. le temps effectif / la persistance ;
3. les deux ;
4. aucun effet robuste.

Ce fichier est conçu pour être ajouté directement au projet Hawkes intertrade.
Il ne fitte pas un modèle Hawkes complet. Il fournit des tests descriptifs et
semi-paramétriques à utiliser avant la modélisation MLE.

Données attendues
-----------------
Un DataFrame pandas avec au minimum :

    timestamp : temps numérique croissant ou convertible en float
    volume    : volume du trade

Optionnel :

    timestamp_datetime : datetime pandas pour corriger la saisonnalité intraday
    side               : signe du trade, par exemple +1/-1, BUY/SELL, etc.

Idée statistique
----------------
On classe les trades par buckets de volume, puis on regarde la réponse
post-trade : nombre moyen de trades observés après un trade de bucket q,
dans différents horizons temporels.

Signature amplitude-only :
    Les courbes de réponse post-trade sont plus hautes pour gros volumes,
    mais leurs pentes log-linéaires sont similaires.

Signature temps-effectif / persistance :
    Les courbes de réponse post-trade ont des pentes différentes ; les gros
    volumes décroissent plus lentement.

Signature amplitude + persistance :
    Les gros volumes donnent des courbes plus hautes et plus lentes.

Modèles conceptuels
-------------------
Amplitude-only :

    phi(u, v) = alpha * w_eta(v) * beta * exp(-beta * u)

Persistance-only :

    phi(u, v) = alpha * beta / m_delta(v) * exp(- beta * u / m_delta(v))

Amplitude + persistance :

    phi(u, v) = alpha * w_eta(v) * beta / m_delta(v)
                * exp(- beta * u / m_delta(v))

avec :

    z(v) = (log(1 + v) - mean) / std
    w_eta(v) = exp(eta * z(v))
    m_delta(v) = exp(delta * z(v))

Dépendances
-----------
    numpy
    pandas
    matplotlib optionnel pour les plots

Auteur : généré pour intégration dans le projet Hawkes intertrade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


AggMethod = Literal["mean", "median"]


# -----------------------------------------------------------------------------
# Structures de sortie
# -----------------------------------------------------------------------------


@dataclass
class DescriptiveDecision:
    """Résumé interprétable du diagnostic amplitude vs persistance."""

    verdict: str
    amplitude_score: float
    persistence_score: float
    comments: List[str]
    slope_table: pd.DataFrame
    next_duration_table: pd.DataFrame


# -----------------------------------------------------------------------------
# Utilitaires internes
# -----------------------------------------------------------------------------


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans df : {missing}")


def _to_float_array(x: pd.Series | np.ndarray | Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    if np.any(~np.isfinite(arr)):
        raise ValueError("Les valeurs doivent être finies.")
    return arr


def _safe_qcut(values: pd.Series, q: int) -> pd.Series:
    """
    qcut robuste aux valeurs constantes / duplications.

    Retourne une Series float contenant des buckets 0, 1, ... ou NaN si la
    découpe est impossible.
    """
    values = pd.Series(values)

    if values.notna().sum() == 0:
        return pd.Series(np.nan, index=values.index)

    if values.nunique(dropna=True) < 2:
        return pd.Series(0.0, index=values.index)

    out = pd.qcut(values, q=q, labels=False, duplicates="drop")
    return out.astype(float)


def _weighted_polyfit_slope(
    x: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """
    Régression y = intercept + slope * x.

    Retourne intercept, slope, r2 pondéré approximatif.
    """
    x = _to_float_array(x)
    y = _to_float_array(y)

    if weights is None:
        weights = np.ones_like(x)
    else:
        weights = _to_float_array(weights)

    if len(x) < 2:
        return np.nan, np.nan, np.nan

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    x = x[mask]
    y = y[mask]
    weights = weights[mask]

    if len(x) < 2:
        return np.nan, np.nan, np.nan

    x_bar = np.average(x, weights=weights)
    y_bar = np.average(y, weights=weights)

    var_x = np.average((x - x_bar) ** 2, weights=weights)
    if var_x <= 0:
        return np.nan, np.nan, np.nan

    cov_xy = np.average((x - x_bar) * (y - y_bar), weights=weights)
    slope = cov_xy / var_x
    intercept = y_bar - slope * x_bar

    y_hat = intercept + slope * x
    ss_res = np.sum(weights * (y - y_hat) ** 2)
    ss_tot = np.sum(weights * (y - y_bar) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return float(intercept), float(slope), float(r2)


def _statistic(values: np.ndarray, method: AggMethod) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    if method == "mean":
        return float(np.mean(values))
    if method == "median":
        return float(np.median(values))
    raise ValueError("method doit être 'mean' ou 'median'.")


# -----------------------------------------------------------------------------
# Préparation des données
# -----------------------------------------------------------------------------


def prepare_trade_dataframe(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    volume_col: str = "volume",
    datetime_col: Optional[str] = None,
    sort: bool = True,
) -> pd.DataFrame:
    """
    Nettoie et enrichit un DataFrame de trades.

    Ajoute :
        log_volume
        duration_prev
        duration_next
        trade_index

    Si datetime_col est fourni, ajoute :
        minute_of_day
        intraday_bucket_30m

    Parameters
    ----------
    df : pd.DataFrame
        Données de trades.
    timestamp_col : str
        Colonne de timestamps numériques. L'unité peut être seconde, minute, etc.
    volume_col : str
        Colonne de volume.
    datetime_col : str, optional
        Colonne datetime pour saisonnalité intraday.
    sort : bool
        Si True, trie par timestamp.

    Returns
    -------
    pd.DataFrame
    """
    _require_columns(df, [timestamp_col, volume_col])

    out = df.copy()
    out[timestamp_col] = pd.to_numeric(out[timestamp_col], errors="coerce")
    out[volume_col] = pd.to_numeric(out[volume_col], errors="coerce")

    out = out.dropna(subset=[timestamp_col, volume_col])
    out = out[out[volume_col] >= 0]

    if sort:
        out = out.sort_values(timestamp_col)

    out = out.reset_index(drop=True)
    out["trade_index"] = np.arange(len(out))
    out["log_volume"] = np.log1p(out[volume_col].astype(float))
    out["duration_prev"] = out[timestamp_col].diff()
    out["duration_next"] = out[timestamp_col].shift(-1) - out[timestamp_col]

    if datetime_col is not None:
        _require_columns(out, [datetime_col])
        dt = pd.to_datetime(out[datetime_col], errors="coerce")
        out["minute_of_day"] = dt.dt.hour * 60 + dt.dt.minute
        out["intraday_bucket_30m"] = (out["minute_of_day"] // 30).astype("Int64")

    return out


def add_volume_buckets(
    df: pd.DataFrame,
    n_buckets: int = 5,
    log_volume_col: str = "log_volume",
    intraday_bucket_col: Optional[str] = None,
    output_col: str = "volume_bucket",
) -> pd.DataFrame:
    """
    Ajoute des buckets de volume.

    Si intraday_bucket_col est fourni, les quantiles de volume sont calculés à
    l'intérieur de chaque bucket intraday. C'est recommandé pour éviter de
    confondre l'effet volume avec la saisonnalité intraday.

    Parameters
    ----------
    df : pd.DataFrame
    n_buckets : int
    log_volume_col : str
        Colonne de volume transformé, typiquement log_volume.
    intraday_bucket_col : str, optional
        Colonne de bucket intraday.
    output_col : str
        Nom de la colonne créée.

    Returns
    -------
    pd.DataFrame
    """
    _require_columns(df, [log_volume_col])
    out = df.copy()

    if intraday_bucket_col is None:
        out[output_col] = _safe_qcut(out[log_volume_col], q=n_buckets)
    else:
        _require_columns(out, [intraday_bucket_col])
        out[output_col] = (
            out.groupby(intraday_bucket_col, group_keys=False)[log_volume_col]
            .apply(lambda s: _safe_qcut(s, q=n_buckets))
        )

    return out


def add_forward_durations(
    df: pd.DataFrame,
    max_horizon: int = 20,
    timestamp_col: str = "timestamp",
    prefix: str = "duration_fwd_",
) -> pd.DataFrame:
    """
    Ajoute les durées futures :

        duration_fwd_h = timestamp[k+h] - timestamp[k+h-1]

    pour h = 1, ..., max_horizon.
    """
    _require_columns(df, [timestamp_col])
    out = df.copy().sort_values(timestamp_col).reset_index(drop=True)

    for h in range(1, max_horizon + 1):
        out[f"{prefix}{h}"] = (
            out[timestamp_col].shift(-h) - out[timestamp_col].shift(-(h - 1))
        )

    return out


# -----------------------------------------------------------------------------
# Diagnostics descriptifs : durées
# -----------------------------------------------------------------------------


def describe_next_duration_by_volume(
    df: pd.DataFrame,
    bucket_col: str = "volume_bucket",
    duration_col: str = "duration_next",
) -> pd.DataFrame:
    """
    Résume la prochaine durée intertrade par bucket de volume.

    Signature utile :
        median(duration_next | gros volume) < median(duration_next | petit volume)
        indique une activité plus rapide après gros volume.
    """
    _require_columns(df, [bucket_col, duration_col])

    tmp = df[[bucket_col, duration_col]].dropna()
    tmp = tmp[tmp[duration_col] >= 0]

    return (
        tmp.groupby(bucket_col)[duration_col]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
        .sort_values(bucket_col)
    )


def describe_forward_durations_by_volume(
    df: pd.DataFrame,
    bucket_col: str = "volume_bucket",
    max_horizon: int = 20,
    agg: AggMethod = "median",
    prefix: str = "duration_fwd_",
) -> pd.DataFrame:
    """
    Résume les durées futures par bucket de volume et horizon événementiel.

    Si l'effet volume disparaît après quelques horizons, il est plutôt court terme.
    S'il persiste sur h=5,10,20, c'est une signature de persistance.
    """
    cols = [f"{prefix}{h}" for h in range(1, max_horizon + 1)]
    _require_columns(df, [bucket_col] + cols)

    tmp = df[[bucket_col] + cols].copy()
    for c in cols:
        tmp = tmp[(tmp[c].isna()) | (tmp[c] >= 0)]

    if agg == "median":
        out = tmp.groupby(bucket_col)[cols].median()
    elif agg == "mean":
        out = tmp.groupby(bucket_col)[cols].mean()
    else:
        raise ValueError("agg doit être 'mean' ou 'median'.")

    return out.reset_index().sort_values(bucket_col)


# -----------------------------------------------------------------------------
# Réponse empirique post-trade
# -----------------------------------------------------------------------------


def empirical_post_trade_activity(
    df: pd.DataFrame,
    bins: Sequence[float],
    bucket_col: str = "volume_bucket",
    timestamp_col: str = "timestamp",
    exclude_same_timestamp: bool = True,
) -> pd.DataFrame:
    """
    Estime l'activité empirique post-trade par bucket de volume.

    Pour chaque trade ancre t0 dans un bucket q, on compte le nombre moyen de
    trades dans :

        [t0 + bins[b], t0 + bins[b+1])

    puis on divise par la largeur du bin pour obtenir un taux empirique.

    Parameters
    ----------
    df : pd.DataFrame
    bins : sequence of float
        Bins en unités de temps des timestamps. Exemple :
        [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    bucket_col : str
    timestamp_col : str
    exclude_same_timestamp : bool
        Si True, les événements exactement simultanés au trade ancre sont exclus.

    Returns
    -------
    pd.DataFrame avec colonnes :
        volume_bucket, bin_left, bin_right, bin_center, width,
        count, n_anchors, rate
    """
    _require_columns(df, [bucket_col, timestamp_col])

    bins = _to_float_array(bins)
    if len(bins) < 2 or np.any(np.diff(bins) <= 0):
        raise ValueError("bins doit être strictement croissant et de taille >= 2.")

    tmp = df[[bucket_col, timestamp_col]].dropna().copy()
    tmp = tmp.sort_values(timestamp_col).reset_index(drop=True)

    times = tmp[timestamp_col].to_numpy(dtype=float)
    buckets = sorted(tmp[bucket_col].dropna().unique())

    rows: List[Dict[str, float]] = []

    for q in buckets:
        anchor_idx = tmp.index[tmp[bucket_col] == q].to_numpy()
        anchor_times = times[anchor_idx]

        counts = np.zeros(len(bins) - 1, dtype=float)

        for t0 in anchor_times:
            # On veut souvent exclure le trade ancre. Pour bins[0] = 0,
            # side='right' exclut les événements exactement à t0.
            side_left = "right" if exclude_same_timestamp else "left"
            left = np.searchsorted(times, t0 + bins[:-1], side=side_left)
            right = np.searchsorted(times, t0 + bins[1:], side="left")
            counts += np.maximum(right - left, 0)

        n_anchors = len(anchor_times)
        widths = np.diff(bins)
        rates = counts / np.maximum(n_anchors * widths, 1e-12)

        for b, width in enumerate(widths):
            rows.append(
                {
                    "volume_bucket": float(q),
                    "bin_left": float(bins[b]),
                    "bin_right": float(bins[b + 1]),
                    "bin_center": float(0.5 * (bins[b] + bins[b + 1])),
                    "width": float(width),
                    "count": float(counts[b]),
                    "n_anchors": int(n_anchors),
                    "rate": float(rates[b]),
                }
            )

    return pd.DataFrame(rows)


def estimate_log_response_slopes(
    response: pd.DataFrame,
    bucket_col: str = "volume_bucket",
    x_col: str = "bin_center",
    rate_col: str = "rate",
    count_col: str = "count",
    min_rate: float = 1e-12,
    min_points: int = 3,
    use_weights: bool = True,
) -> pd.DataFrame:
    """
    Estime par bucket :

        log(rate_q(u)) = intercept_q + slope_q * u

    Lecture :
        intercept qui augmente avec le volume -> effet amplitude.
        slope moins négative pour gros volume -> effet persistance.

    Returns
    -------
    pd.DataFrame avec :
        volume_bucket, intercept, slope, decay_proxy, r2, n_points
    """
    _require_columns(response, [bucket_col, x_col, rate_col])

    rows: List[Dict[str, float]] = []

    for q, g in response.groupby(bucket_col):
        g = g.copy()
        g = g[np.isfinite(g[rate_col]) & (g[rate_col] > min_rate)]

        if len(g) < min_points:
            continue

        x = g[x_col].to_numpy(dtype=float)
        y = np.log(g[rate_col].to_numpy(dtype=float))

        if use_weights and count_col in g.columns:
            weights = np.maximum(g[count_col].to_numpy(dtype=float), 1.0)
        else:
            weights = None

        intercept, slope, r2 = _weighted_polyfit_slope(x, y, weights)

        rows.append(
            {
                "volume_bucket": float(q),
                "intercept": intercept,
                "slope": slope,
                "decay_proxy": -slope if np.isfinite(slope) else np.nan,
                "r2": r2,
                "n_points": int(len(g)),
            }
        )

    return pd.DataFrame(rows).sort_values("volume_bucket").reset_index(drop=True)


# -----------------------------------------------------------------------------
# Survie empirique des durées
# -----------------------------------------------------------------------------


def survival_by_volume_bucket(
    df: pd.DataFrame,
    grid: Sequence[float],
    bucket_col: str = "volume_bucket",
    duration_col: str = "duration_next",
) -> pd.DataFrame:
    """
    Estime :

        S_q(u) = P(duration_next > u | volume_bucket = q)

    Une séparation durable des courbes suggère un effet persistant.
    """
    _require_columns(df, [bucket_col, duration_col])

    grid = _to_float_array(grid)
    tmp = df[[bucket_col, duration_col]].dropna()
    tmp = tmp[tmp[duration_col] >= 0]

    rows: List[Dict[str, float]] = []

    for q, g in tmp.groupby(bucket_col):
        durations = g[duration_col].to_numpy(dtype=float)
        for u in grid:
            rows.append(
                {
                    "volume_bucket": float(q),
                    "u": float(u),
                    "survival": float(np.mean(durations > u)) if len(durations) else np.nan,
                    "n": int(len(durations)),
                }
            )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Tests descriptifs par permutation / bootstrap
# -----------------------------------------------------------------------------


def permutation_test_high_low_duration(
    df: pd.DataFrame,
    bucket_col: str = "volume_bucket",
    duration_col: str = "duration_next",
    method: AggMethod = "median",
    n_perm: int = 2000,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """
    Test par permutation : différence de durée suivante entre gros et petit volume.

    Statistique :

        stat = agg(duration | high bucket) - agg(duration | low bucket)

    Si stat < 0 : les gros volumes sont suivis de durées plus courtes.

    La p-value est bilatérale par permutation des labels low/high.
    """
    _require_columns(df, [bucket_col, duration_col])

    tmp = df[[bucket_col, duration_col]].dropna().copy()
    tmp = tmp[tmp[duration_col] >= 0]

    if tmp.empty:
        raise ValueError("Aucune observation valide pour le test.")

    low_bucket = tmp[bucket_col].min()
    high_bucket = tmp[bucket_col].max()

    sample = tmp[tmp[bucket_col].isin([low_bucket, high_bucket])].copy()
    labels = (sample[bucket_col].to_numpy() == high_bucket).astype(int)
    values = sample[duration_col].to_numpy(dtype=float)

    n_high = int(labels.sum())
    n_low = int(len(labels) - n_high)

    if n_high < 2 or n_low < 2:
        raise ValueError("Pas assez d'observations dans low/high buckets.")

    obs_high = _statistic(values[labels == 1], method)
    obs_low = _statistic(values[labels == 0], method)
    obs_stat = obs_high - obs_low

    rng = np.random.default_rng(random_state)
    perm_stats = np.empty(n_perm, dtype=float)

    for b in range(n_perm):
        perm = rng.permutation(labels)
        stat_high = _statistic(values[perm == 1], method)
        stat_low = _statistic(values[perm == 0], method)
        perm_stats[b] = stat_high - stat_low

    p_value = float(np.mean(np.abs(perm_stats) >= abs(obs_stat)))

    return {
        "low_bucket": float(low_bucket),
        "high_bucket": float(high_bucket),
        "n_low": float(n_low),
        "n_high": float(n_high),
        "method": method,
        "observed_low": float(obs_low),
        "observed_high": float(obs_high),
        "observed_diff_high_minus_low": float(obs_stat),
        "p_value_two_sided": p_value,
    }


def bootstrap_high_low_duration_ci(
    df: pd.DataFrame,
    bucket_col: str = "volume_bucket",
    duration_col: str = "duration_next",
    method: AggMethod = "median",
    n_boot: int = 2000,
    ci: float = 0.95,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """
    Intervalle de confiance bootstrap pour :

        agg(duration | high bucket) - agg(duration | low bucket)

    Utile pour quantifier l'effet descriptif brut du volume sur la durée suivante.
    """
    _require_columns(df, [bucket_col, duration_col])

    tmp = df[[bucket_col, duration_col]].dropna().copy()
    tmp = tmp[tmp[duration_col] >= 0]

    low_bucket = tmp[bucket_col].min()
    high_bucket = tmp[bucket_col].max()

    low = tmp.loc[tmp[bucket_col] == low_bucket, duration_col].to_numpy(dtype=float)
    high = tmp.loc[tmp[bucket_col] == high_bucket, duration_col].to_numpy(dtype=float)

    if len(low) < 2 or len(high) < 2:
        raise ValueError("Pas assez d'observations dans low/high buckets.")

    rng = np.random.default_rng(random_state)
    boot_stats = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        low_b = rng.choice(low, size=len(low), replace=True)
        high_b = rng.choice(high, size=len(high), replace=True)
        boot_stats[b] = _statistic(high_b, method) - _statistic(low_b, method)

    alpha = 1.0 - ci
    lo = np.quantile(boot_stats, alpha / 2.0)
    hi = np.quantile(boot_stats, 1.0 - alpha / 2.0)

    obs = _statistic(high, method) - _statistic(low, method)

    return {
        "low_bucket": float(low_bucket),
        "high_bucket": float(high_bucket),
        "method": method,
        "observed_diff_high_minus_low": float(obs),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_level": float(ci),
    }


def slope_trend_permutation_test(
    slope_table: pd.DataFrame,
    bucket_col: str = "volume_bucket",
    slope_col: str = "slope",
    n_perm: int = 10000,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """
    Test descriptif de tendance de pente selon le volume.

    On régresse :

        slope_q = a + b * bucket_q

    Si b > 0, les slopes deviennent moins négatives avec le volume,
    ce qui suggère une persistance plus forte après gros volumes.

    La p-value est obtenue par permutation des pentes entre buckets.
    """
    _require_columns(slope_table, [bucket_col, slope_col])

    tmp = slope_table[[bucket_col, slope_col]].dropna().sort_values(bucket_col)

    if len(tmp) < 3:
        raise ValueError("Il faut au moins 3 buckets pour tester une tendance.")

    x = tmp[bucket_col].to_numpy(dtype=float)
    y = tmp[slope_col].to_numpy(dtype=float)

    _, obs_trend, _ = _weighted_polyfit_slope(x, y)

    rng = np.random.default_rng(random_state)
    perm_trends = np.empty(n_perm, dtype=float)

    for b in range(n_perm):
        y_perm = rng.permutation(y)
        _, trend, _ = _weighted_polyfit_slope(x, y_perm)
        perm_trends[b] = trend

    p_two = float(np.mean(np.abs(perm_trends) >= abs(obs_trend)))
    p_positive = float(np.mean(perm_trends >= obs_trend))

    return {
        "observed_slope_trend": float(obs_trend),
        "p_value_two_sided": p_two,
        "p_value_positive_trend": p_positive,
        "interpretation": (
            "trend > 0 signifie que les pentes log-réponse deviennent moins négatives "
            "avec le volume, signature de persistance accrue."
        ),
    }


def intercept_trend_permutation_test(
    slope_table: pd.DataFrame,
    bucket_col: str = "volume_bucket",
    intercept_col: str = "intercept",
    n_perm: int = 10000,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """
    Test descriptif de tendance d'intercept selon le volume.

    On régresse :

        intercept_q = a + b * bucket_q

    Si b > 0, la réponse post-trade est plus haute pour gros volume,
    signature d'un effet amplitude.
    """
    _require_columns(slope_table, [bucket_col, intercept_col])

    tmp = slope_table[[bucket_col, intercept_col]].dropna().sort_values(bucket_col)

    if len(tmp) < 3:
        raise ValueError("Il faut au moins 3 buckets pour tester une tendance.")

    x = tmp[bucket_col].to_numpy(dtype=float)
    y = tmp[intercept_col].to_numpy(dtype=float)

    _, obs_trend, _ = _weighted_polyfit_slope(x, y)

    rng = np.random.default_rng(random_state)
    perm_trends = np.empty(n_perm, dtype=float)

    for b in range(n_perm):
        y_perm = rng.permutation(y)
        _, trend, _ = _weighted_polyfit_slope(x, y_perm)
        perm_trends[b] = trend

    p_two = float(np.mean(np.abs(perm_trends) >= abs(obs_trend)))
    p_positive = float(np.mean(perm_trends >= obs_trend))

    return {
        "observed_intercept_trend": float(obs_trend),
        "p_value_two_sided": p_two,
        "p_value_positive_trend": p_positive,
        "interpretation": (
            "trend > 0 signifie que la réponse log-post-trade est plus haute "
            "avec le volume, signature d'un effet amplitude."
        ),
    }


# -----------------------------------------------------------------------------
# Décision descriptive amplitude vs persistance
# -----------------------------------------------------------------------------


def diagnose_volume_effect(
    df: pd.DataFrame,
    bins: Sequence[float],
    timestamp_col: str = "timestamp",
    volume_col: str = "volume",
    datetime_col: Optional[str] = None,
    n_buckets: int = 5,
    use_intraday_buckets: bool = False,
    max_horizon: int = 20,
    n_perm: int = 2000,
    random_state: Optional[int] = None,
) -> DescriptiveDecision:
    """
    Pipeline complet de diagnostic descriptif.

    Étapes :
        1. préparation du DataFrame ;
        2. buckets de volume, éventuellement corrigés intraday ;
        3. résumé de duration_next ;
        4. réponse post-trade empirique ;
        5. pentes log-réponse ;
        6. tests de tendance intercept et slope ;
        7. verdict heuristique.

    Returns
    -------
    DescriptiveDecision
    """
    prepared = prepare_trade_dataframe(
        df,
        timestamp_col=timestamp_col,
        volume_col=volume_col,
        datetime_col=datetime_col,
    )

    intraday_col = None
    if use_intraday_buckets:
        if datetime_col is None and "intraday_bucket_30m" not in prepared.columns:
            raise ValueError(
                "use_intraday_buckets=True nécessite datetime_col ou intraday_bucket_30m."
            )
        intraday_col = "intraday_bucket_30m"

    prepared = add_volume_buckets(
        prepared,
        n_buckets=n_buckets,
        log_volume_col="log_volume",
        intraday_bucket_col=intraday_col,
        output_col="volume_bucket",
    )

    prepared = add_forward_durations(
        prepared,
        max_horizon=max_horizon,
        timestamp_col=timestamp_col,
    )

    next_table = describe_next_duration_by_volume(prepared)

    response = empirical_post_trade_activity(
        prepared,
        bins=bins,
        bucket_col="volume_bucket",
        timestamp_col=timestamp_col,
    )

    slopes = estimate_log_response_slopes(response)

    comments: List[str] = []

    amplitude_score = np.nan
    persistence_score = np.nan

    if len(slopes) >= 3:
        amp_test = intercept_trend_permutation_test(
            slopes,
            n_perm=n_perm,
            random_state=random_state,
        )
        per_test = slope_trend_permutation_test(
            slopes,
            n_perm=n_perm,
            random_state=random_state,
        )

        amplitude_score = amp_test["observed_intercept_trend"]
        persistence_score = per_test["observed_slope_trend"]

        if amplitude_score > 0:
            comments.append("L'intercept log-réponse augmente avec le volume : signature amplitude.")
        else:
            comments.append("Pas de tendance positive claire de l'intercept : amplitude peu visible.")

        if persistence_score > 0:
            comments.append(
                "La pente log-réponse devient moins négative avec le volume : signature persistance."
            )
        else:
            comments.append("Pas de tendance positive claire de pente : persistance peu visible.")

        comments.append(
            f"p-value amplitude approx. = {amp_test['p_value_two_sided']:.4f} ; "
            f"p-value persistance approx. = {per_test['p_value_two_sided']:.4f}."
        )

        amp_significant = amplitude_score > 0 and amp_test["p_value_two_sided"] < 0.10
        per_significant = persistence_score > 0 and per_test["p_value_two_sided"] < 0.10

        if amp_significant and per_significant:
            verdict = "amplitude_et_persistance"
        elif amp_significant:
            verdict = "amplitude_only"
        elif per_significant:
            verdict = "persistance_only"
        else:
            verdict = "pas_de_signature_forte"
    else:
        verdict = "insuffisant"
        comments.append("Pas assez de buckets valides pour conclure.")

    return DescriptiveDecision(
        verdict=verdict,
        amplitude_score=float(amplitude_score) if np.isfinite(amplitude_score) else np.nan,
        persistence_score=float(persistence_score) if np.isfinite(persistence_score) else np.nan,
        comments=comments,
        slope_table=slopes,
        next_duration_table=next_table,
    )


# -----------------------------------------------------------------------------
# Plots optionnels
# -----------------------------------------------------------------------------


def plot_post_trade_response(
    response: pd.DataFrame,
    bucket_col: str = "volume_bucket",
    x_col: str = "bin_center",
    rate_col: str = "rate",
    log_x: bool = True,
    log_y: bool = True,
):
    """
    Trace la réponse post-trade empirique par bucket de volume.

    Retourne fig, ax.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))

    for q, g in response.groupby(bucket_col):
        g = g.sort_values(x_col)
        ax.plot(g[x_col], g[rate_col], marker="o", label=f"bucket {q:g}")

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel("Temps après le trade")
    ax.set_ylabel("Taux empirique post-trade")
    ax.set_title("Réponse empirique post-trade par bucket de volume")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    return fig, ax


def plot_forward_durations(
    forward_table: pd.DataFrame,
    bucket_col: str = "volume_bucket",
    prefix: str = "duration_fwd_",
):
    """
    Trace les durées futures agrégées par bucket de volume.

    Retourne fig, ax.
    """
    import matplotlib.pyplot as plt

    duration_cols = [c for c in forward_table.columns if c.startswith(prefix)]
    horizons = np.array([int(c.replace(prefix, "")) for c in duration_cols])

    fig, ax = plt.subplots(figsize=(9, 5))

    for _, row in forward_table.sort_values(bucket_col).iterrows():
        y = row[duration_cols].to_numpy(dtype=float)
        ax.plot(horizons, y, marker="o", label=f"bucket {row[bucket_col]:g}")

    ax.set_xlabel("Horizon événementiel h")
    ax.set_ylabel("Durée future agrégée")
    ax.set_title("Durées futures par bucket de volume")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_survival_by_bucket(
    survival: pd.DataFrame,
    bucket_col: str = "volume_bucket",
    u_col: str = "u",
    survival_col: str = "survival",
):
    """
    Trace la survie empirique des durées par bucket de volume.

    Retourne fig, ax.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))

    for q, g in survival.groupby(bucket_col):
        g = g.sort_values(u_col)
        ax.plot(g[u_col], g[survival_col], label=f"bucket {q:g}")

    ax.set_xlabel("u")
    ax.set_ylabel("P(duration_next > u)")
    ax.set_title("Survie empirique des durées par bucket de volume")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


# -----------------------------------------------------------------------------
# Exemple synthétique et smoke tests utilisables avec pytest
# -----------------------------------------------------------------------------


def make_synthetic_trades(
    n: int = 3000,
    seed: int = 123,
    mode: Literal["amplitude", "persistence", "mixed", "none"] = "mixed",
) -> pd.DataFrame:
    """
    Génère un jeu synthétique de trades pour tester les diagnostics.

    Ce générateur n'est pas un Hawkes exact. Il sert seulement à tester le
    pipeline descriptif.
    """
    rng = np.random.default_rng(seed)

    volume = rng.lognormal(mean=4.5, sigma=1.0, size=n)
    z = (np.log1p(volume) - np.mean(np.log1p(volume))) / np.std(np.log1p(volume))

    base_rate = 3.0

    if mode == "none":
        effective_rate = base_rate * np.ones(n)
    elif mode == "amplitude":
        effective_rate = base_rate * np.exp(0.45 * z)
    elif mode == "persistence":
        # Ici, on crée une dépendance durable approximative en lissant z.
        persistent_signal = pd.Series(z).rolling(20, min_periods=1).mean().to_numpy()
        effective_rate = base_rate * np.exp(0.55 * persistent_signal)
    elif mode == "mixed":
        persistent_signal = pd.Series(z).rolling(20, min_periods=1).mean().to_numpy()
        effective_rate = base_rate * np.exp(0.30 * z + 0.35 * persistent_signal)
    else:
        raise ValueError("mode invalide.")

    effective_rate = np.clip(effective_rate, 0.05, 100.0)
    durations = rng.exponential(1.0 / effective_rate)
    timestamps = np.cumsum(durations)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "volume": volume,
            "price": 100.0 + np.cumsum(rng.normal(0.0, 0.01, size=n)),
        }
    )


def _smoke_test() -> None:
    """Test rapide exécutable avec : python hawkes_volume_diagnostics.py"""
    df = make_synthetic_trades(n=1000, seed=42, mode="mixed")
    bins = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0])

    prepared = prepare_trade_dataframe(df)
    prepared = add_volume_buckets(prepared, n_buckets=5)
    prepared = add_forward_durations(prepared, max_horizon=5)

    next_summary = describe_next_duration_by_volume(prepared)
    assert not next_summary.empty

    forward = describe_forward_durations_by_volume(prepared, max_horizon=5)
    assert not forward.empty

    response = empirical_post_trade_activity(prepared, bins=bins)
    assert not response.empty
    assert {"volume_bucket", "bin_center", "rate"}.issubset(response.columns)

    slopes = estimate_log_response_slopes(response)
    assert not slopes.empty

    surv = survival_by_volume_bucket(prepared, grid=np.linspace(0.0, 1.0, 10))
    assert not surv.empty

    duration_test = permutation_test_high_low_duration(
        prepared,
        n_perm=100,
        random_state=123,
    )
    assert "p_value_two_sided" in duration_test

    decision = diagnose_volume_effect(
        df,
        bins=bins,
        n_perm=100,
        random_state=123,
    )
    assert decision.verdict in {
        "amplitude_et_persistance",
        "amplitude_only",
        "persistance_only",
        "pas_de_signature_forte",
        "insuffisant",
    }


if __name__ == "__main__":
    _smoke_test()
    print("Smoke test OK.")
