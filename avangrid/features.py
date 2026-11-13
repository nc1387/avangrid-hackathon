"""Feature building: weather adjustment, daily peer-relative signals, and features for any date range.

The daily signals and period features match the definitions developed for notebook 60, so
the notebooks and this package agree.

Typical use::

    hd = grids.load_hourly()
    D = load_daily_signals(hd)                 # cached dict of customers x days arrays
    pf = PeriodFeatures(D, hd.temp)            # cumulative sums, built once
    full_year = pf(0, hd.shape[1])             # dict: feature -> one value per customer
    since_change = pf(change_day, hd.shape[1]) # per-customer start days work too
"""

import warnings

import numpy as np
import polars as pl

from . import cache, config, db
from .grids import HourlyData, source_key

DAILY_VERSION = 'daily-v1'  # bump when daily_signals changes

# Period-feature names, in output order.
DAILY_MEANS = ['e_night', 'e_morning', 'e_afternoon', 'e_evening', 'e_midday', 'e_level',
               'zero_mid_rel', 'high_hours_rel', 'sessions_rel', 'sharp_rises_rel']
PERIOD_FEATURES = DAILY_MEANS + ['dip_rel', 'heat_slope_rel', 'cool_slope_rel', 'cold_level_rel']


# --- weather ----------------------------------------------------------------------------

def weather_design(temp: np.ndarray, weekend: np.ndarray, kind: str = 'piecewise') -> np.ndarray:
    """Days x terms design matrix for removing weather effects.

    'linear': intercept, heating degrees below 55 F, cooling degrees above 65 F, weekend.
    'piecewise': adds extra slopes below 35 F and 15 F and above 75 F, so use can climb faster
    in deep cold and extreme heat. It removed more week-to-week variation in testing.
    """
    hdd = np.maximum(config.HEATING_BASE_F - temp, 0)
    cdd = np.maximum(temp - config.COOLING_BASE_F, 0)
    cols = [np.ones_like(temp), hdd, cdd, weekend.astype(float)]
    if kind == 'piecewise':
        cols += [np.maximum(35 - temp, 0), np.maximum(15 - temp, 0), np.maximum(temp - 75, 0)]
    elif kind != 'linear':
        raise ValueError(f'unknown kind {kind!r}')
    return np.column_stack(cols).astype(float)


def remove_weather(Y: np.ndarray, X: np.ndarray, fit_mask: np.ndarray | None = None) -> np.ndarray:
    """Remove each customer's own weather response by least squares.

    Y: customers x days, or customers x days x hours (each hour fitted separately).
    X: days x terms (see ``weather_design``).
    fit_mask: optional customers x days boolean; only those days are used to fit, e.g. the
        period *before* a candidate change, so a new heat pump isn't absorbed into the fit.
    Without ``fit_mask``, missing values are filled with the customer's mean before fitting (fast:
    one shared least-squares solve). With ``fit_mask``, each customer gets its own weighted fit,
    done in chunks to limit memory. Missing values stay missing in the result.
    """
    Y = np.asarray(Y, dtype=np.float64)
    hourly = Y.ndim == 3
    Y3 = Y if hourly else Y[:, :, None]
    missing = np.isnan(Y3)
    if fit_mask is None:
        filled = np.where(missing, np.nanmean(Y3, axis=1, keepdims=True), Y3)
        beta = np.einsum('kd,cdh->ckh', np.linalg.pinv(X), filled)
    else:
        beta = np.empty((Y3.shape[0], X.shape[1], Y3.shape[2]))
        ridge = 1e-9 * np.eye(X.shape[1])
        for s in range(0, Y3.shape[0], 1000):
            y = Y3[s:s + 1000]
            w = ((~np.isnan(y)) & fit_mask[s:s + 1000, :, None]).astype(float)
            XtX = np.einsum('dk,dl,cdh->chkl', X, X, w, optimize=True)
            XtY = np.einsum('dk,cdh->chk', X, np.where(w > 0, y, 0), optimize=True)
            beta[s:s + 1000] = np.linalg.solve(XtX + ridge, XtY[..., None])[..., 0].transpose(0, 2, 1)
    out = Y3 - np.einsum('dk,ckh->cdh', X, beta)
    out[missing] = np.nan
    return out if hourly else out[:, :, 0]


# --- daily signals ------------------------------------------------------------------------

def group_relative(x: np.ndarray, g: np.ndarray, ref: np.ndarray | None = None):
    """Subtract each peer group's daily mean. Returns (relative values, group means)."""
    if ref is None:
        ref = np.vstack([np.nanmean(x[g == k], 0) for k in range(config.N_PEER_GROUPS)])
    return x - ref[g], ref


def daily_signals(kwh: np.ndarray, resid: np.ndarray, g: np.ndarray, peer_index: np.ndarray,
                  refs: dict | None = None) -> tuple[dict, dict]:
    """Daily peer-relative signals (customers x days) from hourly arrays.

    Returns (signals, refs). ``refs`` holds the peer-group daily means used for the *_rel
    count signals; pass them back in to put new rows (e.g. simulated customers) on the same scale.
    """
    with warnings.catch_warnings():
        # Days where every hour was imputed have no real readings; their daily values are NaN.
        warnings.filterwarnings('ignore', message='Mean of empty slice')
        warnings.filterwarnings('ignore', message='All-NaN slice encountered')
        return _daily_signals(kwh, resid, g, peer_index, refs)


def _daily_signals(kwh, resid, g, peer_index, refs):
    n, nd, _ = kwh.shape
    refs = {} if refs is None else dict(refs)
    E = np.asarray(resid)
    K = np.asarray(kwh)
    P = np.asarray(peer_index)
    D = {f'e_{k}': np.nanmean(E[:, :, s], 2) for k, s in config.TOD_SLICES.items()}
    D['e_midday'] = np.nanmean(E[:, :, config.MIDDAY], 2)
    D['e_level'] = np.nanmean(E, 2)
    normalized = E + P[g]
    D['n_mid'] = np.nanmean(normalized[:, :, config.MIDDAY], 2)
    D['n_night'] = np.nanmean(normalized[:, :, config.TOD_SLICES['night']], 2)
    del normalized
    D['p_mid'] = np.nanmean(P[:, :, config.MIDDAY], 2)[g]
    D['p_night'] = np.nanmean(P[:, :, config.TOD_SLICES['night']], 2)[g]

    mid = K[:, :, config.MIDDAY]
    zero = np.nanmean(np.where(np.isnan(mid), np.nan, (mid < 0.05).astype(np.float32)), 2)
    base = np.nanmedian(K, 1)                                    # customer x hour median
    high = (K - base[:, None, :]) >= 3                           # 3+ kWh above usual for that hour
    flat_high = high.reshape(n, -1)
    starts = flat_high & ~np.concatenate([np.zeros((n, 1), bool), flat_high[:, :-1]], 1)
    flat = K.reshape(n, -1)
    rises = np.concatenate([np.zeros((n, 1), bool), (flat[:, 1:] - flat[:, :-1]) >= 3], 1)
    for name, x in [('zero_mid_rel', zero),
                    ('high_hours_rel', high.sum(2).astype(np.float32)),
                    ('sessions_rel', starts.reshape(n, nd, 24).sum(2).astype(np.float32)),
                    ('sharp_rises_rel', rises.reshape(n, nd, 24).sum(2).astype(np.float32))]:
        D[name], refs[name] = group_relative(x, g, refs.get(name))
    return {k: v.astype(np.float32) for k, v in D.items()}, refs


def load_daily_signals(hd: HourlyData, rebuild: bool = False, verbose: bool = True) -> tuple[dict, dict]:
    """Daily signals for every customer in ``hd``, cached (about 15 s to build). Returns (signals, refs)."""
    def build():
        D, refs = daily_signals(hd.kwh, hd.resid, hd.group, hd.peer_index)
        return {**D, **{f'ref__{k}': v for k, v in refs.items()}}, {}, {}
    source = source_key()
    key = None if source is None else {'version': DAILY_VERSION, 'source': source}
    arrays, _, _ = cache.get_or_build('daily_signals', key, build, rebuild=rebuild, verbose=verbose)
    D = {k: v for k, v in arrays.items() if not k.startswith('ref__')}
    refs = {k[5:]: np.asarray(v) for k, v in arrays.items() if k.startswith('ref__')}
    return D, refs


def weekly(x: np.ndarray, n_weeks: int | None = None) -> np.ndarray:
    """Average a customers x days (x hours) array into whole Monday-Sunday weeks."""
    n_weeks = x.shape[1] // 7 if n_weeks is None else n_weeks
    return np.nanmean(x[:, :n_weeks * 7].reshape(x.shape[0], n_weeks, 7, *x.shape[2:]), 2)


# --- features over any date range ---------------------------------------------------------

def _cumsum(x: np.ndarray):
    v = np.where(np.isnan(x), 0, x)
    c = (~np.isnan(x)).astype(float)
    z = np.zeros((x.shape[0], 1))
    return np.concatenate([z, np.cumsum(v, 1)], 1), np.concatenate([z, np.cumsum(c, 1)], 1)


class PeriodFeatures:
    """Peer-relative features over any range of days, with a different range per customer.

    Building takes a few seconds (cumulative sums of every daily signal); each call after
    that is a handful of array lookups, so trying many date ranges is cheap.

    Features (one value per customer):
      e_night/morning/afternoon/evening/midday/level: average residual vs peers
      zero_mid_rel, high_hours_rel, sessions_rel, sharp_rises_rel: daily counts minus peers
      dip_rel: midday/night ratio minus the peer group's (more negative = deeper solar-like dip)
      heat_slope_rel / cool_slope_rel: residual level per degree of heating / cooling
          (needs at least 14 heating / cooling days in the range, else NaN)
      cold_level_rel: residual level on days below 35 F (needs at least 5 such days, else NaN)
    """

    def __init__(self, D: dict, temp: np.ndarray):
        n, nd = D['e_level'].shape
        self.n, self.n_days = n, nd
        hdd = np.maximum(config.HEATING_BASE_F - temp, 0)
        cdd = np.maximum(temp - config.COOLING_BASE_F, 0)
        C = {k: _cumsum(np.asarray(D[k], dtype=np.float64)) for k in DAILY_MEANS + ['n_mid', 'n_night', 'p_mid', 'p_night']}
        level = np.asarray(D['e_level'], dtype=np.float64)
        for name, x in [('heat', hdd), ('cool', cdd)]:
            m = np.where(x > 0, 1.0, np.nan)
            xx = np.broadcast_to(x * m, (n, nd))
            y = level * m
            valid = np.where(np.isnan(y), np.nan, 1.0)
            for suffix, arr in [('n', valid), ('x', xx * valid), ('xx', xx * xx * valid), ('y', y), ('xy', xx * y)]:
                C[f'{name}_{suffix}'] = _cumsum(arr)
        C['cold_y'] = _cumsum(np.where(np.broadcast_to(temp < 35, (n, nd)), level, np.nan))
        self.C = C

    def __call__(self, d0, d1) -> dict[str, np.ndarray]:
        """Features over days [d0, d1). d0/d1 may be scalars or one value per customer."""
        r = np.arange(self.n)
        d0 = np.broadcast_to(d0, (self.n,)).astype(int)
        d1 = np.broadcast_to(d1, (self.n,)).astype(int)
        C = self.C

        def seg(k, i=0):
            return C[k][i][r, d1] - C[k][i][r, d0]

        def mean(k):
            s, c = seg(k), seg(k, 1)
            return np.where(c > 0, s / np.maximum(c, 1), np.nan)

        f = {k: mean(k) for k in DAILY_MEANS}
        f['dip_rel'] = mean('n_mid') / mean('n_night') - mean('p_mid') / mean('p_night')
        for name in ['heat', 'cool']:
            k, sx, sxx, sy, sxy = (seg(f'{name}_{s}') for s in ['n', 'x', 'xx', 'y', 'xy'])
            den = k * sxx - sx ** 2
            f[f'{name}_slope_rel'] = np.where((k >= 14) & (den > 0), (k * sxy - sx * sy) / np.where(den > 0, den, 1), np.nan)
        cy, cn = seg('cold_y'), seg('cold_y', 1)
        f['cold_level_rel'] = np.where(cn >= 5, cy / np.maximum(cn, 1), np.nan)
        return {k: f[k] for k in PERIOD_FEATURES}

    @staticmethod
    def to_matrix(f: dict[str, np.ndarray], extra: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, list[str]]:
        names = list(f) + list(extra or {})
        X = np.column_stack([f[k] for k in f] + [v for v in (extra or {}).values()]).astype(float)
        X[~np.isfinite(X)] = np.nan
        return X, names


def customer_features(customer_numbers=None) -> tuple[np.ndarray, list[str]]:
    """Notebook 40's full-year ``customer_features`` table as a matrix, optionally aligned to given customers."""
    cf = db.sql('SELECT * FROM customer_features ORDER BY CUSTOMER_NUMBER')
    if customer_numbers is not None:
        cf = pl.DataFrame({'CUSTOMER_NUMBER': np.asarray(customer_numbers)}).cast({'CUSTOMER_NUMBER': cf['CUSTOMER_NUMBER'].dtype}) \
            .join(cf, on='CUSTOMER_NUMBER', how='left')
    names = [c for c in cf.columns if c != 'CUSTOMER_NUMBER']
    X = cf.select(names).to_numpy().astype(float)
    X[~np.isfinite(X)] = np.nan
    return X, names
