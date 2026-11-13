"""Heat-pump physics features: responses that heat pumps have and electric resistance / gas homes don't.

All within the heating season (Nov-Mar), using day-to-day anomalies (each day minus its 15-day centered
mean) so seasonal trends in lighting and occupancy cancel out:

  hp_frost_coef / hp_frost_t: extra daily use per unit of "frost index" (mean over hours of relative
      humidity on hours between 25 and 40 F). Heat pumps run defrost cycles in humid near-freezing
      weather; resistance heat doesn't care about humidity. Controlled for heating degrees and wind.
  hp_wind_coef / hp_wind_t: extra use per unit of wind x heating degrees (infiltration), a check that
      the home's heating is electric.
  hp_hdd_partial: heating response per degree with frost and wind held fixed.
  hp_mild_band_slope: heating response on days averaging 40-55 F only. Heat pumps (especially
      mini-splits) run a lot in mild cold, when gas homes barely change.
  (All coefficients are divided by the customer's average daily kWh, so homes of different size compare.)

  hp_backup_step: on hours below 12 F, use above what the 12-44 F trend predicts, divided by average use.
      Many heat pumps switch on electric backup strips in deep cold, a step that straight-line
      resistance heating doesn't show.
  hp_heat_cool_log_ratio: log of (heating kW at 20 F) / (AC kW added on hot afternoons). The same heat
      pump does both, so the two are similar in size; resistance heat + window AC gives a mismatch.
  hp_heat_cool_min: the smaller of the two (both present and sizeable suggests one system doing both).
"""

import warnings

import numpy as np
import polars as pl

from .appliances import _rolling_mean


def _masked_ols(Y: np.ndarray, X: np.ndarray):
    """Per-row OLS of Y (customers x days) on X (days x terms), skipping NaN days per row.

    Returns (beta: customers x terms, t: customers x terms).
    """
    ok = ~np.isnan(Y)
    Yz = np.where(ok, Y, 0)
    w = ok.astype(float)
    XtX = np.einsum('dk,dl,cd->ckl', X, X, w)
    XtY = np.einsum('dk,cd->ck', X, Yz)
    ridge = 1e-9 * np.eye(X.shape[1])
    beta = np.linalg.solve(XtX + ridge, XtY[..., None])[..., 0]
    resid = np.where(ok, Y - beta @ X.T, 0)
    dof = np.maximum(ok.sum(1) - X.shape[1], 1)
    s2 = (resid ** 2).sum(1) / dof
    inv_diag = np.diagonal(np.linalg.inv(XtX + ridge), axis1=1, axis2=2)
    t = beta / np.sqrt(np.maximum(s2[:, None] * inv_diag, 1e-12))
    return beta, t


def hp_physics_features(K: np.ndarray, weather: dict, dates: pl.Series, F: dict | None = None,
                        chunk: int = 2000) -> dict[str, np.ndarray]:
    """See module docstring. K: customers x days x 24 kWh; weather: avangrid.weather.load_weather_hourly();
    F: optional avangrid.appliances features (for the heating/cooling size match)."""
    T, RH, WS = weather['temp'], weather['humidity'], weather['windspeed']
    n, nd, _ = K.shape
    month = dates.dt.month().to_numpy()
    season = np.isin(month, [11, 12, 1, 2, 3])
    tday = T.mean(1)
    hdd = np.maximum(55 - tday, 0)
    frost = (((T >= 25) & (T <= 40)) * RH / 100).mean(1)
    wind = (WS * np.maximum(55 - T, 0) / 10).mean(1)
    anom = lambda x: x - _rolling_mean(x[None, :], 15)[0]
    X = np.column_stack([anom(hdd), anom(frost), anom(wind)])[season]
    mild = season & (tday >= 40) & (tday <= 55)
    x_mild = anom(hdd)[mild]
    # Backup heat: hourly bins within the season
    Ts = T[season]
    edges = [-100, 12, 20, 28, 36, 44]
    bins = [(Ts >= lo) & (Ts < hi) for lo, hi in zip(edges[:-1], edges[1:])]
    centers = np.array([Ts[b].mean() if b.any() else np.nan for b in bins])

    names = ['hp_frost_coef', 'hp_frost_t', 'hp_wind_coef', 'hp_wind_t', 'hp_hdd_partial', 'hp_mild_band_slope',
             'hp_backup_step']
    out = {k: np.full(n, np.nan) for k in names}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for s0 in range(0, n, chunk):
            k = np.asarray(K[s0:s0 + chunk], dtype=np.float64)
            o = slice(s0, s0 + len(k))
            y = np.nanmean(k, 2) * 24
            scale = np.nanmean(y, 1)
            ya = y - _rolling_mean(y, 15)
            beta, t = _masked_ols(ya[:, season], X)
            out['hp_hdd_partial'][o] = beta[:, 0] / scale
            out['hp_frost_coef'][o], out['hp_frost_t'][o] = beta[:, 1] / scale, t[:, 1]
            out['hp_wind_coef'][o], out['hp_wind_t'][o] = beta[:, 2] / scale, t[:, 2]
            bm, _ = _masked_ols(ya[:, mild], x_mild[:, None])
            out['hp_mild_band_slope'][o] = bm[:, 0] / scale

            ks = k[:, season]
            means = np.column_stack([np.nanmean(ks[:, b], 1) for b in bins])      # customers x 5 bins
            xs, ys = centers[1:], means[:, 1:]
            mx, my = xs.mean(), ys.mean(1, keepdims=True)
            slope = ((xs - mx) * (ys - my)).sum(1) / ((xs - mx) ** 2).sum()
            pred = my[:, 0] + slope * (centers[0] - mx)
            out['hp_backup_step'][o] = (means[:, 0] - pred) / (scale / 24)
    if F is not None:
        heat = np.maximum(F['heat_kw_at_20f'], 0)
        cool = np.maximum(F['ac_added_kw'], 0)
        out['hp_heat_cool_log_ratio'] = np.log((heat + 0.05) / (cool + 0.05))
        out['hp_heat_cool_min'] = np.minimum(heat, cool)
    for v in out.values():
        v[~np.isfinite(v)] = np.nan
    return out
