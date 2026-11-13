"""NumPy version of notebook 40's ``customer_features`` (23 full-year features per customer).

The table in avangrid.db is built with SQL over ``ami_analysis``, so it can't be recomputed for
simulated customers (e.g. ordinary homes with EV charging added). This computes the same features
from any customers x days x 24 kWh array, so models trained on them can be tested on simulated
homes. Differences from the table are small: imputed hours are missing here (the table used the
imputed values) and daily temperature uses interpolated hourly temperatures.
"""

import warnings

import numpy as np
import polars as pl

NAMES = ['mean_kwh', 'p95_kwh', 'max_kwh', 'winter_summer_ratio', 'cold_night_ratio', 'summer_midday_zero_share',
         'summer_midday_kwh', 'summer_night_kwh', 'winter_midday_kwh', 'night_share', 'morning_share',
         'afternoon_share', 'evening_share', 'summer_midday_night_ratio', 'midday_summer_winter_ratio',
         'high_sessions', 'share_high_hours', 'sharp_rises', 'heating_slope', 'cooling_slope', 'mild_day_kwh',
         'heating_slope_rel', 'cooling_slope_rel']


def _slope(Y, x, mask):
    """Per-row OLS slope of Y[:, mask] on x[mask], ignoring NaN in Y."""
    Y, x = Y[:, mask], x[mask]
    ok = ~np.isnan(Y)
    n = ok.sum(1)
    xb = np.where(ok, x[None, :], 0)
    mx = xb.sum(1) / np.maximum(n, 1)
    my = np.where(ok, Y, 0).sum(1) / np.maximum(n, 1)
    dx = np.where(ok, x[None, :] - mx[:, None], 0)
    sxx = (dx * dx).sum(1)
    sxy = (dx * np.where(ok, Y - my[:, None], 0)).sum(1)
    return np.where((n > 2) & (sxx > 0), sxy / np.where(sxx > 0, sxx, 1), np.nan)


def compute(K: np.ndarray, temp_hourly: np.ndarray, temp_quartile: np.ndarray, dates: pl.Series,
            chunk: int = 2000) -> np.ndarray:
    """customers x 23 feature matrix (columns in ``NAMES`` order) from hourly kWh."""
    n, nd, _ = K.shape
    month = dates.dt.month().to_numpy()
    day_temp = temp_hourly.mean(1)
    m_hour = np.broadcast_to(month[:, None], (nd, 24))
    he = np.broadcast_to(np.arange(1, 25)[None, :], (nd, 24))
    masks = {
        'winter': np.isin(m_hour, [12, 1, 2]), 'summer': np.isin(m_hour, [6, 7, 8]),
        'cold_night': (he <= 6) & (temp_quartile == 1), 'mild_night': (he <= 6) & (temp_quartile == 3),
        'sm_mid': (he >= 11) & (he <= 15) & (m_hour >= 4) & (m_hour <= 9),
        'sm_night': (he <= 6) & (m_hour >= 4) & (m_hour <= 9),
        'w_mid': (he >= 11) & (he <= 15) & np.isin(m_hour, [11, 12, 1]),
        'night': he <= 6, 'morning': (he >= 7) & (he <= 12), 'afternoon': (he >= 13) & (he <= 18), 'evening': he >= 19,
    }
    out = np.full((n, len(NAMES)), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for s0 in range(0, n, chunk):
            k = np.asarray(K[s0:s0 + chunk], dtype=np.float64)
            c = len(k)
            flat = k.reshape(c, -1)
            avg = lambda m: np.nanmean(k[:, m], 1)
            f = {'mean_kwh': np.nanmean(flat, 1), 'p95_kwh': np.nanpercentile(flat, 95, axis=1), 'max_kwh': np.nanmax(flat, 1)}
            f['winter_summer_ratio'] = avg(masks['winter']) / avg(masks['summer'])
            f['cold_night_ratio'] = avg(masks['cold_night']) / avg(masks['mild_night'])
            mid = k[:, masks['sm_mid']]
            f['summer_midday_zero_share'] = np.nanmean(np.where(np.isnan(mid), np.nan, mid < 0.05), 1)
            f['summer_midday_kwh'], f['summer_night_kwh'] = avg(masks['sm_mid']), avg(masks['sm_night'])
            f['winter_midday_kwh'] = avg(masks['w_mid'])
            for tod in ['night', 'morning', 'afternoon', 'evening']:
                f[f'{tod}_share'] = avg(masks[tod]) / f['mean_kwh']
            f['summer_midday_night_ratio'] = f['summer_midday_kwh'] / f['summer_night_kwh']
            f['midday_summer_winter_ratio'] = f['summer_midday_kwh'] / f['winter_midday_kwh']

            base = np.nanmedian(k, 1)
            high = np.nan_to_num(k - base[:, None, :], nan=-1) >= 3
            hf = high.reshape(c, -1)
            starts = hf & ~np.concatenate([np.zeros((c, 1), bool), hf[:, :-1]], 1)
            f['high_sessions'] = starts.sum(1).astype(float)
            f['share_high_hours'] = hf.sum(1) / np.maximum((~np.isnan(flat)).sum(1), 1)
            f['sharp_rises'] = (np.nan_to_num(flat[:, 1:] - flat[:, :-1], nan=0) >= 3).sum(1).astype(float)

            day_kwh = np.nanmean(k, 2) * 24
            f['heating_slope'] = _slope(day_kwh, np.maximum(55 - day_temp, 0), day_temp < 55)
            f['cooling_slope'] = _slope(day_kwh, np.maximum(day_temp - 65, 0), day_temp > 65)
            f['mild_day_kwh'] = np.nanmean(day_kwh[:, (day_temp >= 55) & (day_temp <= 65)], 1)
            f['heating_slope_rel'] = f['heating_slope'] / f['mild_day_kwh']
            f['cooling_slope_rel'] = f['cooling_slope'] / f['mild_day_kwh']
            out[s0:s0 + c] = np.column_stack([f[k_] for k_ in NAMES])
    out[~np.isfinite(out)] = np.nan
    return out
