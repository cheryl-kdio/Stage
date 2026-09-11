"""
Analyse conjointe de 6 futures dividendes et d'un future d'indice.

Pipeline:
1) Calcul des rendements minute.
2) Estimation de 6 betas de rendement sur une fenetre glissante.
3) Transformation vers une sensibilite compatible avec un modele en niveau:
       D_t(T) = alpha_t(T) + b(T) S_t
   avec b(T) = 1 - exp(-lambda * tau)
   et, localement:
       beta_ret(T) ~= b(T) * S_t / D_t(T)
   donc:
       b_tilde(T) = beta_ret(T) * D_t(T) / S_t.
4) Ajustement transversal, a chaque date t, de:
       b_tilde_t(tau_j) = 1 - exp(-lambda_t * tau_j)
   ou, en diagnostic:
       b_tilde_t(tau_j) = A_t * (1 - exp(-lambda_t * tau_j)).

Dependances:
    numpy
    pandas
    scipy
    statsmodels   # uniquement pour l'inference HAC optionnelle

Remarque:
- Le beta OLS avec constante est exactement Cov(r_D, r_S) / Var(r_S).
- Pour une estimation a chaque minute sur des donnees regulieres, le calcul par
  moments glissants est beaucoup plus rapide qu'un appel OLS a chaque minute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


@dataclass
class TermStructureFit:
    lambda_: float
    amplitude: float
    rmse: float
    n_obs: int
    success: bool


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Rendements logarithmiques: r_t = log(P_t) - log(P_{t-1}).

    Les prix doivent etre strictement positifs.
    """
    if (prices <= 0).any().any():
        raise ValueError("Tous les prix doivent etre strictement positifs.")
    return np.log(prices).diff()


def rolling_beta_fixed_bars(
    returns: pd.DataFrame,
    market_col: str,
    asset_cols: Iterable[str],
    window_bars: int,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    Beta OLS glissant, avec constante implicite:
        beta = Cov(r_asset, r_market) / Var(r_market)

    Cette implementation est adaptee si les observations sont des barres minute
    regulieres. Choisir window_bars = nombre de barres par seance * 20 pour une
    fenetre d'environ 20 seances.

    Si les horaires changent fortement ou s'il y a beaucoup de barres manquantes,
    il vaut mieux construire explicitement les fenetres par calendrier de trading.
    """
    asset_cols = list(asset_cols)
    if min_periods is None:
        min_periods = window_bars

    x = returns[market_col]
    var_x = x.rolling(window_bars, min_periods=min_periods).var(ddof=1)

    out = {}
    for col in asset_cols:
        cov_xy = returns[col].rolling(
            window_bars, min_periods=min_periods
        ).cov(x)
        out[col] = cov_xy / var_x

    return pd.DataFrame(out, index=returns.index)


def transform_return_beta_to_level_sensitivity(
    beta_ret: pd.DataFrame,
    prices: pd.DataFrame,
    market_col: str,
    asset_cols: Iterable[str],
) -> pd.DataFrame:
    """
    Transformation locale d'un beta de rendement vers la sensibilite en niveau.

    Si:
        D_t(T) = alpha_t(T) + b(T) S_t
    et si, sur un pas minute, Delta alpha est traite comme un residu:
        Delta D ~= b(T) Delta S

    alors:
        r_D ~= b(T) * S/D * r_S

    d'ou:
        b_tilde ~= beta_ret * D/S.

    ATTENTION:
    Cette egalite est une approximation locale. Si Delta alpha est correle a
    Delta S, le beta observe absorbe aussi cette composante.
    """
    asset_cols = list(asset_cols)
    s = prices[market_col]

    out = pd.DataFrame(index=beta_ret.index, columns=asset_cols, dtype=float)
    for col in asset_cols:
        out[col] = beta_ret[col] * prices[col] / s

    return out


def time_to_maturity_years(
    index: pd.DatetimeIndex,
    expiries: Dict[str, pd.Timestamp],
    day_count: float = 365.25,
) -> pd.DataFrame:
    """
    Calcule tau_t,j = temps restant jusqu'a l'echeance, en annees.

    expiries: dictionnaire {nom_colonne_future: date_echeance}

    Les echeances doivent etre compatibles avec le fuseau horaire de l'index.
    """
    idx = pd.DatetimeIndex(index)
    out = {}

    for col, expiry in expiries.items():
        exp = pd.Timestamp(expiry)

        if idx.tz is not None and exp.tzinfo is None:
            exp = exp.tz_localize(idx.tz)
        elif idx.tz is None and exp.tzinfo is not None:
            exp = exp.tz_convert(None)

        tau = (exp - idx) / pd.Timedelta(days=day_count)
        out[col] = np.asarray(tau, dtype=float)

    return pd.DataFrame(out, index=idx)


def _model_restricted(tau: np.ndarray, lambda_: float) -> np.ndarray:
    return 1.0 - np.exp(-lambda_ * tau)


def _model_unrestricted(
    tau: np.ndarray, lambda_: float, amplitude: float
) -> np.ndarray:
    return amplitude * (1.0 - np.exp(-lambda_ * tau))


def fit_lambda_cross_section(
    b_values: pd.Series,
    tau_values: pd.Series,
    weights: Optional[pd.Series] = None,
    amplitude_free: bool = False,
    lambda_start: float = 0.5,
) -> TermStructureFit:
    """
    Ajuste la structure par maturite a une date donnee.

    Modele structurel (restreint):
        b_j = 1 - exp(-lambda * tau_j)

    Modele diagnostic (amplitude libre):
        b_j = A * (1 - exp(-lambda * tau_j))

    Les observations avec tau <= 0 ou valeurs non finies sont exclues.

    weights:
        poids WLS optionnels. Typiquement, on peut utiliser l'inverse de la
        variance estimee de b_tilde si cette variance est disponible.
    """
    df = pd.DataFrame(
        {"b": b_values.astype(float), "tau": tau_values.astype(float)}
    )

    if weights is not None:
        df["w"] = weights.astype(float)

    mask = (
        np.isfinite(df["b"])
        & np.isfinite(df["tau"])
        & (df["tau"] > 0)
    )

    if weights is not None:
        mask &= np.isfinite(df["w"]) & (df["w"] > 0)

    df = df.loc[mask]

    n_params = 2 if amplitude_free else 1
    if len(df) < max(3, n_params + 1):
        return TermStructureFit(np.nan, np.nan, np.nan, len(df), False)

    tau = df["tau"].to_numpy()
    b = df["b"].to_numpy()

    if weights is None:
        sqrt_w = np.ones_like(b)
    else:
        sqrt_w = np.sqrt(df["w"].to_numpy())

    if amplitude_free:
        def residuals(theta):
            lambda_, amplitude = theta
            return sqrt_w * (_model_unrestricted(tau, lambda_, amplitude) - b)

        result = least_squares(
            residuals,
            x0=np.array([lambda_start, 1.0]),
            bounds=(np.array([1e-8, 0.0]), np.array([50.0, 5.0])),
        )
        lambda_hat, amplitude_hat = result.x
        fitted = _model_unrestricted(tau, lambda_hat, amplitude_hat)
    else:
        def residuals(theta):
            lambda_ = theta[0]
            return sqrt_w * (_model_restricted(tau, lambda_) - b)

        result = least_squares(
            residuals,
            x0=np.array([lambda_start]),
            bounds=(np.array([1e-8]), np.array([50.0])),
        )
        lambda_hat = float(result.x[0])
        amplitude_hat = 1.0
        fitted = _model_restricted(tau, lambda_hat)

    rmse = float(np.sqrt(np.mean((b - fitted) ** 2)))

    return TermStructureFit(
        lambda_=float(lambda_hat),
        amplitude=float(amplitude_hat),
        rmse=rmse,
        n_obs=len(df),
        success=bool(result.success),
    )


def fit_term_structure_over_time(
    b_tilde: pd.DataFrame,
    tau: pd.DataFrame,
    weights: Optional[pd.DataFrame] = None,
    amplitude_free: bool = False,
) -> pd.DataFrame:
    """
    Estime lambda_t (et eventuellement A_t) a chaque timestamp.

    Avec 6 maturites, l'ajustement est tres rapide. Pour de tres gros historiques,
    on peut sous-echantillonner lambda_t (par exemple toutes les 5 minutes) si
    l'objectif ne requiert pas une valeur a chaque minute.
    """
    common_cols = [c for c in b_tilde.columns if c in tau.columns]

    records = []
    for t in b_tilde.index:
        w = None if weights is None else weights.loc[t, common_cols]

        fit = fit_lambda_cross_section(
            b_values=b_tilde.loc[t, common_cols],
            tau_values=tau.loc[t, common_cols],
            weights=w,
            amplitude_free=amplitude_free,
        )

        records.append(
            {
                "timestamp": t,
                "lambda": fit.lambda_,
                "amplitude": fit.amplitude,
                "rmse": fit.rmse,
                "n_obs": fit.n_obs,
                "success": fit.success,
            }
        )

    return pd.DataFrame.from_records(records).set_index("timestamp")


def hac_beta_at_timestamp(
    returns_window: pd.DataFrame,
    market_col: str,
    asset_col: str,
    maxlags: int = 30,
) -> Tuple[float, float]:
    """
    Estimation OLS + erreur standard HAC/Newey-West pour UNE fenetre.

    A utiliser pour l'inference / les diagnostics, pas necessairement a chaque
    minute car c'est beaucoup plus couteux que le calcul du beta par covariance.

    Retourne:
        beta_hat, se_HAC
    """
    import statsmodels.api as sm

    df = returns_window[[market_col, asset_col]].dropna()
    y = df[asset_col]
    X = sm.add_constant(df[market_col])

    result = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": maxlags},
    )

    return float(result.params[market_col]), float(result.bse[market_col])


def run_pipeline(
    prices: pd.DataFrame,
    market_col: str,
    dividend_cols: Iterable[str],
    expiries: Dict[str, pd.Timestamp],
    bars_per_session: int,
    n_sessions: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Exemple de pipeline complet.

    Returns:
        beta_ret       : 6 betas de rendement glissants
        b_tilde        : sensibilites transformees vers l'echelle du modele
        restricted_fit : lambda_t sous le modele A = 1
        free_fit       : lambda_t et A_t avec amplitude libre (diagnostic)
    """
    dividend_cols = list(dividend_cols)

    required = [market_col] + dividend_cols
    missing = [c for c in required if c not in prices.columns]
    if missing:
        raise KeyError(f"Colonnes absentes: {missing}")

    returns = compute_log_returns(prices[required])

    window_bars = int(bars_per_session * n_sessions)

    beta_ret = rolling_beta_fixed_bars(
        returns=returns,
        market_col=market_col,
        asset_cols=dividend_cols,
        window_bars=window_bars,
    )

    b_tilde = transform_return_beta_to_level_sensitivity(
        beta_ret=beta_ret,
        prices=prices,
        market_col=market_col,
        asset_cols=dividend_cols,
    )

    tau = time_to_maturity_years(
        index=prices.index,
        expiries=expiries,
    )

    restricted_fit = fit_term_structure_over_time(
        b_tilde=b_tilde,
        tau=tau,
        amplitude_free=False,
    )

    free_fit = fit_term_structure_over_time(
        b_tilde=b_tilde,
        tau=tau,
        amplitude_free=True,
    )

    return beta_ret, b_tilde, restricted_fit, free_fit


if __name__ == "__main__":
    # Exemple d'utilisation.
    #
    # Le fichier CSV doit contenir:
    # timestamp, index_future, div_0, div_1, ..., div_5
    #
    # Adaptez les dates d'echeance et bars_per_session a votre marche.
    #
    # prices = (
    #     pd.read_csv("data.csv", parse_dates=["timestamp"])
    #       .set_index("timestamp")
    #       .sort_index()
    # )
    #
    # dividend_cols = [f"div_{i}" for i in range(6)]
    # expiries = {
    #     "div_0": pd.Timestamp("2026-12-18"),
    #     "div_1": pd.Timestamp("2027-12-17"),
    #     "div_2": pd.Timestamp("2028-12-15"),
    #     "div_3": pd.Timestamp("2029-12-21"),
    #     "div_4": pd.Timestamp("2030-12-20"),
    #     "div_5": pd.Timestamp("2031-12-19"),
    # }
    #
    # beta_ret, b_tilde, fit_struct, fit_diag = run_pipeline(
    #     prices=prices,
    #     market_col="index_future",
    #     dividend_cols=dividend_cols,
    #     expiries=expiries,
    #     bars_per_session=510,  # EXEMPLE UNIQUEMENT: a adapter
    #     n_sessions=20,
    # )
    #
    # print(fit_struct.tail())
    # print(fit_diag.tail())
    pass
