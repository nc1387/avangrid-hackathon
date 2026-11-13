"""Hourly data as NumPy arrays of shape customers x days x 24 hours, cached on disk.

Building these from DuckDB takes several seconds and about 2 GB of memory; loading them from
the cache is near-instant (memory-mapped). They depend on tables created by notebooks 20-40:
``ami_analysis``, ``customer_peer_groups``, ``peer_index`` and ``weather_features``.

Conventions:
- Customers are the ones in ``customer_peer_groups`` (everyone in ``ami_analysis`` averaging at
  least ``config.MIN_MEAN_KWH``), sorted by CUSTOMER_NUMBER.
- Day 0 is the first date in the data (2025-09-01, a Monday). Hour column 0 is hour_ending 1.
- **Imputed hours are NaN** in both arrays, as is the hour that doesn't exist when clocks go
  forward (2026-03-08, hour_ending 3). Query ``ami_filled`` in DuckDB if you need imputed values.
"""

from dataclasses import dataclass

import duckdb
import numpy as np
import polars as pl

from . import cache, config, db

GRID_VERSION = 'hourly-v2'  # bump when the build logic changes, to invalidate old caches
# v2: daily temperature from interpolated hourly temperatures (was: daily mean, gaps = yearly mean)

# Sums of rounded integers, not floats: DuckDB adds floats in a different order on each run,
# so float sums differ in the last digits and would make the cache rebuild every time.
_FINGERPRINT_QUERIES = {
    'ami_analysis': 'SELECT COUNT(*), SUM(ROUND(hourly_kwh * 1000)::BIGINT), SUM(is_imputed::INT) FROM ami_analysis',
    'customer_peer_groups': 'SELECT COUNT(*), SUM(peer_group), SUM(ROUND(mean_kwh_real * 1e6)::BIGINT) FROM customer_peer_groups',
    'peer_index': 'SELECT COUNT(*), SUM(ROUND(peer_n * 1e6)::BIGINT) FROM peer_index',
    'weather_features': 'SELECT COUNT(*), SUM(ROUND(temp * 10)::BIGINT) FROM weather_features',
}


@dataclass
class HourlyData:
    customers: pl.DataFrame      # CUSTOMER_NUMBER, peer_group, mean_kwh_real (one row per array row)
    dates: pl.Series             # one date per day index
    kwh: np.ndarray              # customers x days x 24, hourly kWh (NaN = imputed or missing)
    resid: np.ndarray            # customers x days x 24, peer-relative residual: kwh / own mean - peer index
    peer_index: np.ndarray       # peer groups x days x 24, average normalized use of each group
    temp: np.ndarray             # days, daily average temperature (deg F), from interpolated hourly temps
    weekend: np.ndarray          # days, True on Saturday and Sunday
    temp_hourly: np.ndarray      # days x 24, hourly temperature (deg F), gaps linearly interpolated

    @property
    def group(self) -> np.ndarray:
        return self.customers['peer_group'].to_numpy()

    @property
    def customer_numbers(self) -> np.ndarray:
        return self.customers['CUSTOMER_NUMBER'].to_numpy()

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.kwh.shape

    def labels(self) -> dict[str, np.ndarray]:
        """EV / HP / PV list membership aligned to the array rows."""
        from .labels import label_arrays
        return label_arrays(self.customer_numbers)


def _build():
    with db.connect() as con:
        customers = con.sql('SELECT CUSTOMER_NUMBER, peer_group, mean_kwh_real FROM customer_peer_groups ORDER BY 1').pl()
        slots = con.sql('SELECT DISTINCT usage_date, hour_ending FROM peer_index ORDER BY 1, 2').pl()
        values = con.sql('''
            SELECT a.hourly_kwh::FLOAT AS kwh,
                   CASE WHEN a.is_imputed THEN NULL
                        ELSE a.hourly_kwh / g.mean_kwh_real - p.peer_n END::FLOAT AS resid
            FROM ami_analysis a
            JOIN customer_peer_groups g USING (CUSTOMER_NUMBER)
            JOIN peer_index p ON p.peer_group = g.peer_group
                             AND p.usage_date = a.usage_date AND p.hour_ending = a.hour_ending
            ORDER BY a.CUSTOMER_NUMBER, a.usage_date, a.hour_ending
        ''').pl()
        pidx = con.sql('SELECT peer_group, usage_date, hour_ending, peer_n FROM peer_index').pl()
        temp = con.sql('SELECT usage_date, hour_ending, temp FROM weather_features ORDER BY 1, 2').pl()

    start = slots['usage_date'].min()
    day = (slots['usage_date'] - start).dt.total_days().to_numpy()
    hour = slots['hour_ending'].to_numpy() - 1
    n_cust, n_days = customers.height, int(day.max()) + 1
    if values.height != n_cust * slots.height:
        raise ValueError('Expected one row per customer per hourly slot; has ami_analysis changed?')

    kwh = np.full((n_cust, n_days, 24), np.nan, dtype=np.float32)
    resid = np.full_like(kwh, np.nan)
    kwh[:, day, hour] = values['kwh'].to_numpy().reshape(n_cust, -1)
    resid[:, day, hour] = values['resid'].to_numpy().reshape(n_cust, -1)
    kwh[np.isnan(resid)] = np.nan  # imputed hours are missing in both
    del values

    peer = np.full((config.N_PEER_GROUPS, n_days, 24), np.nan, dtype=np.float32)
    peer[pidx['peer_group'].to_numpy(), (pidx['usage_date'] - start).dt.total_days().to_numpy(),
         pidx['hour_ending'].to_numpy() - 1] = pidx['peer_n'].to_numpy()

    dates = pl.date_range(start, slots['usage_date'].max(), eager=True).alias('date')
    # Daily temperature = mean of hourly temperatures, with missing hours filled by linear
    # interpolation between neighbouring readings. 2026-03-08 to 03-10 have no readings at all;
    # interpolating across them is much closer than filling with the yearly mean (~50 F in March).
    hourly = np.full((n_days, 24), np.nan)
    hourly[(temp['usage_date'] - start).dt.total_days().to_numpy(), temp['hour_ending'].to_numpy() - 1] = \
        temp['temp'].to_numpy()
    flat = hourly.ravel()
    ok = ~np.isnan(flat)
    flat[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), flat[ok])
    temp_hourly = flat.reshape(n_days, 24)
    t = temp_hourly.mean(axis=1)
    weekend = (dates.dt.weekday() >= 6).to_numpy()  # polars: Monday = 1 ... Sunday = 7

    arrays = {'kwh': kwh, 'resid': resid, 'peer_index': peer, 'temp': t, 'temp_hourly': temp_hourly, 'weekend': weekend}
    tables = {'customers': customers, 'dates': pl.DataFrame({'date': dates})}
    return arrays, tables, {'n_customers': n_cust, 'n_days': n_days}


def source_key() -> dict | None:
    """Fingerprint of the source tables, or None if the database is locked by another process."""
    try:
        return {'version': GRID_VERSION, 'tables': db.fingerprint(_FINGERPRINT_QUERIES)}
    except duckdb.IOException as err:
        if 'lock' not in str(err).lower():
            raise
        return None


def load_hourly(rebuild: bool = False, verbose: bool = True) -> HourlyData:
    """Load the hourly arrays, building (about 10 s) and caching them on first use."""
    arrays, tables, _ = cache.get_or_build('hourly', source_key(), _build, rebuild=rebuild, verbose=verbose)
    return HourlyData(customers=tables['customers'], dates=tables['dates']['date'],
                      kwh=arrays['kwh'], resid=arrays['resid'], peer_index=arrays['peer_index'],
                      temp=np.asarray(arrays['temp']), weekend=np.asarray(arrays['weekend']),
                      temp_hourly=np.asarray(arrays['temp_hourly']))
