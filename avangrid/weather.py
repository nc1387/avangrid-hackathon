"""Hourly weather as days x 24 arrays (temperature, humidity, windspeed, temperature quartile).

Missing readings are filled by linear interpolation over the hourly sequence (2026-03-08 to 03-10 have
no readings at all). Small and quick to load, so it isn't cached.
"""

from functools import lru_cache

import numpy as np

from . import db


def _interp(flat: np.ndarray) -> np.ndarray:
    flat = flat.copy()
    ok = ~np.isnan(flat)
    flat[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), flat[ok])
    return flat


@lru_cache(maxsize=1)
def load_weather_hourly() -> dict[str, np.ndarray]:
    """{'temp', 'humidity', 'windspeed': days x 24 float, 'temp_quartile': days x 24 int (0 = missing)}."""
    w = db.sql('SELECT usage_date, hour_ending, temp, humidity, windspeed, temp_quartile '
               'FROM weather_features ORDER BY 1, 2')
    day = (w['usage_date'] - w['usage_date'].min()).dt.total_days().to_numpy()
    hour = w['hour_ending'].to_numpy() - 1
    n_days = int(day.max()) + 1
    out = {}
    for col in ['temp', 'humidity', 'windspeed']:
        a = np.full((n_days, 24), np.nan)
        a[day, hour] = w[col].to_numpy()
        out[col] = _interp(a.ravel()).reshape(n_days, 24)
    q = np.zeros((n_days, 24), dtype=np.int8)
    q[day, hour] = w['temp_quartile'].fill_null(0).to_numpy()
    out['temp_quartile'] = q
    return out
