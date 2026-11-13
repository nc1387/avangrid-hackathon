"""Technology signature features (one value per customer), built without using the EV/HP/PV lists.

Three groups, each aimed at a known weakness of the earlier features:

1. Solar index (PV): does the customer's day-to-day midday dip rise and fall with sunshine?
   Separates solar homes from homes that are simply empty at midday.
2. Charging blocks (EV): flat, sharp-edged blocks of 2-8 hours well above the customer's usual use,
   as a home charger produces, instead of counting every high hour.
3. Cold-response shape (HP): heating slope in mild cold, cold and deep cold separately, plus how
   flat and how morning-peaked cold days are compared with mild days.

None of these use the lists, so they can be used in hidden-label tests without leaking labels.
Results are cached (see ``load_signatures``).
"""

import warnings

import numpy as np
import polars as pl

from . import cache, config
from .grids import HourlyData, source_key

SIGNATURES_VERSION = 'signatures-v3'  # v3: adds peak-load features (v2: HP slopes on own-normalized use)


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean along axis 1, ignoring NaN (customers x days)."""
    v = np.where(np.isnan(x), 0, x)
    c = (~np.isnan(x)).astype(float)
    z = np.zeros((x.shape[0], 1))
    cv, cc = np.concatenate([z, np.cumsum(v, 1)], 1), np.concatenate([z, np.cumsum(c, 1)], 1)
    n = x.shape[1]
    lo = np.clip(np.arange(n) - window // 2, 0, n)
    hi = np.clip(np.arange(n) + window // 2 + 1, 0, n)
    cnt = cc[:, hi] - cc[:, lo]
    return np.where(cnt > 0, (cv[:, hi] - cv[:, lo]) / np.maximum(cnt, 1), np.nan)


def _corr_with(A: np.ndarray, s: np.ndarray):
    """Per-row correlation and slope of A (customers x days) on s (days or customers x days)."""
    s = np.broadcast_to(s, A.shape)
    ok = ~(np.isnan(A) | np.isnan(s))
    n = ok.sum(1)
    a = np.where(ok, A, 0); b = np.where(ok, s, 0)
    ma, mb = a.sum(1) / np.maximum(n, 1), b.sum(1) / np.maximum(n, 1)
    da, db = np.where(ok, A - ma[:, None], 0), np.where(ok, s - mb[:, None], 0)
    cov, va, vb = (da * db).sum(1), (da * da).sum(1), (db * db).sum(1)
    corr = np.where((n > 30) & (va > 0) & (vb > 0), cov / np.sqrt(va * vb), np.nan)
    slope = np.where((n > 30) & (vb > 0), cov / np.where(vb > 0, vb, 1), np.nan)
    return corr, slope


# --- 1. solar index -----------------------------------------------------------------------

def solar_index_features(D: dict, temp_hourly: np.ndarray, dates: pl.Series, top_share: float = 0.01) -> dict:
    """How closely each customer's day-to-day midday dip follows sunshine.

    Daily dip = night residual minus midday residual (higher = deeper midday dip). Its day-to-day
    *anomaly* (minus a 15-day centered rolling mean) removes seasonal change and leaves weather.
    Two sunshine proxies:
      - solar index: average dip anomaly of the ``top_share`` of customers with the deepest
        Apr-Sep dip (chosen from the data, not the lists; each excludes itself from its own index);
      - temperature range: daily max minus min temperature (clear days swing more), as an anomaly.
    """
    dip = np.asarray(D['e_night'], dtype=np.float64) - np.asarray(D['e_midday'], dtype=np.float64)
    anom = dip - _rolling_mean(dip, 15)
    month = dates.dt.month().to_numpy()
    summer = (month >= 4) & (month <= 9)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        summer_dip = np.nanmean(dip[:, summer], 1)
    m = max(10, int(round(top_share * len(dip))))
    ref = np.argsort(-np.nan_to_num(summer_dip, nan=-np.inf))[:m]
    ref_vals = anom[ref]
    total, count = np.nansum(ref_vals, 0), (~np.isnan(ref_vals)).sum(0)
    index = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    corr, slope = _corr_with(anom, index)
    # leave-one-out for the reference customers, so they aren't correlated with themselves
    own = np.where(np.isnan(ref_vals), 0, ref_vals)
    loo = np.where(count - (~np.isnan(ref_vals)) > 0, (total - own) / np.maximum(count - (~np.isnan(ref_vals)), 1), np.nan)
    corr[ref], slope[ref] = _corr_with(anom[ref], loo)

    t_range = temp_hourly.max(1) - temp_hourly.min(1)
    t_anom = t_range - _rolling_mean(t_range[None, :], 15)[0]
    corr_t, _ = _corr_with(anom, t_anom)
    return {'solar_index_corr': corr, 'solar_index_slope': slope, 'dip_vs_temp_range_corr': corr_t}


# --- 2. charging blocks -------------------------------------------------------------------

def charging_block_features(kwh: np.ndarray, temp: np.ndarray, weekend: np.ndarray, chunk: int = 2000,
                            min_excess: float = 2.5, min_len: int = 2, max_len: int = 8,
                            max_cv: float = 0.3, min_edge: float = 2.0) -> dict:
    """Flat, sharp-edged blocks of high use, like a home EV charger.

    A block is a run of consecutive hours (across midnight too) at least ``min_excess`` kWh above the
    customer's median for that hour of day, lasting ``min_len``-``max_len`` hours, with a steady level
    (coefficient of variation of the excess <= ``max_cv``) and use jumping up and down by at least
    ``min_edge`` kWh at its start and end.
    """
    n, nd, _ = kwh.shape
    T = nd * 24
    mild_day = (temp >= 50) & (temp <= 70)
    out = {k: np.full(n, np.nan) for k in ['ev_blocks_per_week', 'ev_block_kwh', 'ev_block_hours',
                                           'ev_block_start_concentration', 'ev_block_mild_day_excess',
                                           'ev_block_weekday_excess']}
    for s0 in range(0, n, chunk):
        K = np.asarray(kwh[s0:s0 + chunk], dtype=np.float64)
        c = K.shape[0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            base = np.nanmedian(K, 1)
        excess = (K - base[:, None, :]).reshape(c, T)
        Kf = K.reshape(c, T)
        high = np.zeros((c, T + 2), bool)
        high[:, 1:T + 1] = np.nan_to_num(excess, nan=-1) >= min_excess
        X = np.zeros((c, T + 2)); X[:, 1:T + 1] = np.nan_to_num(excess)
        Kp = np.full((c, T + 2), np.nan); Kp[:, 1:T + 1] = Kf
        flat_high = high.ravel()
        starts = np.flatnonzero(flat_high[1:] & ~flat_high[:-1]) + 1
        ends = np.flatnonzero(flat_high[:-1] & ~flat_high[1:])
        cs = np.concatenate([[0.0], np.cumsum(X.ravel())])
        cs2 = np.concatenate([[0.0], np.cumsum(X.ravel() ** 2)])
        length = ends - starts + 1
        mean = (cs[ends + 1] - cs[starts]) / length
        var = np.maximum((cs2[ends + 1] - cs2[starts]) / length - mean ** 2, 0)
        cv = np.sqrt(var) / mean
        kp = Kp.ravel()
        rise, fall = kp[starts] - kp[starts - 1], kp[ends] - kp[ends + 1]
        keep = ((length >= min_len) & (length <= max_len) & (cv <= max_cv)
                & (np.nan_to_num(rise) >= min_edge) & (np.nan_to_num(fall) >= min_edge))
        cust = starts[keep] // (T + 2)
        t0 = starts[keep] % (T + 2) - 1
        day, hour = t0 // 24, t0 % 24
        cnt = np.bincount(cust, minlength=c).astype(float)
        has = cnt > 0
        o = slice(s0, s0 + c)
        out['ev_blocks_per_week'][o] = cnt / (nd / 7)
        out['ev_block_kwh'][o] = np.where(has, np.bincount(cust, mean[keep], c) / np.maximum(cnt, 1), np.nan)
        out['ev_block_hours'][o] = np.where(has, np.bincount(cust, length[keep], c) / np.maximum(cnt, 1), np.nan)
        hist = np.bincount(cust * 24 + hour, minlength=c * 24).reshape(c, 24)
        window3 = hist + np.roll(hist, -1, 1) + np.roll(hist, -2, 1)       # 3-hour windows, wrapping midnight
        out['ev_block_start_concentration'][o] = np.where(cnt >= 3, window3.max(1) / np.maximum(cnt, 1), np.nan)
        out['ev_block_mild_day_excess'][o] = np.where(
            cnt >= 3, np.bincount(cust, mild_day[day], c) / np.maximum(cnt, 1) - mild_day.mean(), np.nan)
        out['ev_block_weekday_excess'][o] = np.where(
            cnt >= 3, np.bincount(cust, ~weekend[day], c) / np.maximum(cnt, 1) - (~weekend).mean(), np.nan)
    return out


# --- 3. cold-response shape ---------------------------------------------------------------

def cold_response_features(D: dict, kwh: np.ndarray, temp: np.ndarray, weekend: np.ndarray, chunk: int = 4000) -> dict:
    """Heating response in temperature bands, and the shape of cold days compared with mild days.

    Slopes come from a per-customer fit of daily use divided by the customer's own average (NOT
    peer-relative: peer groups were formed partly on temperature response, so comparing with peers
    cancels much of a heat pump's signal) on
      heating degrees below 55 F, extra below 40 F, extra below 20 F, cooling degrees above 65 F, weekend.
    So hp_slope_40_55 is the response in mild cold (heat pumps run; gas homes barely change),
    hp_slope_20_40 and hp_slope_below_20 the response in colder bands (convex if efficiency drops
    or backup resistance heat kicks in).
    Cold-day shape uses hourly kWh: load factor (daily mean / daily max; heat pumps run steadily) and
    morning ramp (hour_ending 6-9 vs 1-5), each on cold days (< 35 F) minus mild days (50-65 F).
    """
    n = kwh.shape[0]
    X = np.column_stack([np.ones_like(temp), np.maximum(55 - temp, 0), np.maximum(40 - temp, 0),
                         np.maximum(20 - temp, 0), np.maximum(temp - 65, 0), weekend.astype(float)])
    pinv = np.linalg.pinv(X)
    beta = np.full((n, X.shape[1]), np.nan)
    cold, mild = temp < 35, (temp >= 50) & (temp <= 65)
    lf_diff, ramp_diff = np.full(n, np.nan), np.full(n, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for s0 in range(0, n, chunk):
            K = np.asarray(kwh[s0:s0 + chunk], dtype=np.float64)
            daily = np.nanmean(K, 2)
            y = daily / np.nanmean(daily, 1, keepdims=True)              # use relative to own average
            y[~np.isfinite(y)] = np.nan
            y = np.where(np.isnan(y), np.nanmean(y, 1, keepdims=True), y)
            beta[s0:s0 + len(K)] = y @ pinv.T
            lf = np.nanmean(K, 2) / np.nanmax(K, 2)
            ramp = np.nanmean(K[:, :, 5:9], 2) / np.nanmean(K[:, :, 0:5], 2)
            lf[~np.isfinite(lf)] = np.nan; ramp[~np.isfinite(ramp)] = np.nan
            o = slice(s0, s0 + len(K))
            lf_diff[o] = np.nanmean(lf[:, cold], 1) - np.nanmean(lf[:, mild], 1)
            ramp_diff[o] = np.nanmedian(ramp[:, cold], 1) - np.nanmedian(ramp[:, mild], 1)
    out = {'hp_slope_40_55': beta[:, 1], 'hp_slope_20_40': beta[:, 1] + beta[:, 2],
           'hp_slope_below_20': beta[:, 1] + beta[:, 2] + beta[:, 3], 'hp_deep_cold_extra': beta[:, 3],
           'hp_cold_load_factor_change': lf_diff, 'hp_cold_morning_ramp_change': ramp_diff}
    return out


GROUPS = {
    'Solar index': ['solar_index_corr', 'solar_index_slope', 'dip_vs_temp_range_corr'],
    'Charging blocks': ['ev_blocks_per_week', 'ev_block_kwh', 'ev_block_hours', 'ev_block_start_concentration',
                        'ev_block_mild_day_excess', 'ev_block_weekday_excess'],
    'Cold-response shape': ['hp_slope_40_55', 'hp_slope_20_40', 'hp_slope_below_20', 'hp_deep_cold_extra',
                            'hp_cold_load_factor_change', 'hp_cold_morning_ramp_change'],
    'Peak load': ['peak_daily_rel', 'peak_daily_rel_vs_peers', 'peak_daily_cv', 'peak_winter_summer_ratio',
                  'load_factor_annual', 'load_factor_daily', 'peak_hours_near_max',
                  'peak_share_night', 'peak_share_morning', 'peak_share_evening', 'peak_hour_concentration',
                  'grid_peak_ratio', 'grid_peak_ratio_winter', 'grid_peak_ratio_summer', 'grid_peak_kwh'],
}
ORIGINAL_GROUPS = ('Solar index', 'Charging blocks', 'Cold-response shape')
ALL_GROUPS = ORIGINAL_GROUPS + ('Peak load',)


# --- 4. peak load -------------------------------------------------------------------------

def peak_features(kwh: np.ndarray, group: np.ndarray, dates: pl.Series, chunk: int = 2000,
                  n_grid_peak: int = 100, n_season_peak: int = 50) -> dict:
    """How high, how often, when, and how coincident with the grid peak each customer's peaks are.

    Daily peaks (highest hour of each day, divided by the customer's average hourly use):
      peak_daily_rel: median; peak_daily_rel_vs_peers: minus the peer group's median of the same;
      peak_daily_cv: day-to-day variation of the daily peak; peak_winter_summer_ratio: median daily
      peak in Dec-Feb / Jun-Aug.
    Load factor (average / peak): load_factor_annual over the year, load_factor_daily median over days;
      peak_hours_near_max: median hours per day within 80% of that day's peak (broad vs spiky).
    Timing: share of daily peaks at night (hour_ending 1-6), morning (7-10) and evening (17-22), and
      peak_hour_concentration, the largest share of peaks falling in any 3-hour window.
    Grid peak: the hours when all customers together use the most (top ``n_grid_peak`` of the year,
      and top ``n_season_peak`` in Dec-Feb and in Jun-Aug). grid_peak_ratio* = the customer's average
      use in those hours / their average use overall; grid_peak_kwh = their average kWh in the top hours.
    """
    n, nd, _ = kwh.shape
    month = dates.dt.month().to_numpy()
    winter, summer = np.isin(month, [12, 1, 2]), np.isin(month, [6, 7, 8])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        system = np.zeros((nd, 24))
        for s0 in range(0, n, chunk):
            system += np.nansum(np.asarray(kwh[s0:s0 + chunk], dtype=np.float64), 0)
        flat = system.ravel()

        def top_hours(day_mask, k):
            allowed = np.repeat(day_mask, 24)
            idx = np.flatnonzero(allowed)
            return idx[np.argsort(-flat[idx])[:k]]

        peaks = {'grid_peak_ratio': top_hours(np.ones(nd, bool), n_grid_peak),
                 'grid_peak_ratio_winter': top_hours(winter, n_season_peak),
                 'grid_peak_ratio_summer': top_hours(summer, n_season_peak)}

        names = GROUPS['Peak load']
        out = {k: np.full(n, np.nan) for k in names}
        for s0 in range(0, n, chunk):
            K = np.asarray(kwh[s0:s0 + chunk], dtype=np.float64)
            c = len(K); o = slice(s0, s0 + c)
            mean = np.nanmean(K.reshape(c, -1), 1)
            dmax, dmean = np.nanmax(K, 2), np.nanmean(K, 2)
            rel = dmax / mean[:, None]
            rel[~np.isfinite(rel)] = np.nan
            out['peak_daily_rel'][o] = np.nanmedian(rel, 1)
            out['peak_daily_cv'][o] = np.nanstd(dmax, 1) / np.nanmean(dmax, 1)
            out['peak_winter_summer_ratio'][o] = np.nanmedian(dmax[:, winter], 1) / np.nanmedian(dmax[:, summer], 1)
            out['load_factor_annual'][o] = mean / np.nanmax(K.reshape(c, -1), 1)
            lf = dmean / dmax
            lf[~np.isfinite(lf)] = np.nan
            out['load_factor_daily'][o] = np.nanmedian(lf, 1)
            near = (K >= 0.8 * dmax[:, :, None]).sum(2).astype(float)
            near[np.isnan(dmax)] = np.nan
            out['peak_hours_near_max'][o] = np.nanmedian(near, 1)
            valid_day = ~np.isnan(dmax) & (dmax > 0)
            peak_hour = np.argmax(np.nan_to_num(K, nan=-1), 2)
            hist = np.stack([((peak_hour == h) & valid_day).sum(1) for h in range(24)], 1).astype(float)
            days = np.maximum(valid_day.sum(1), 1)
            out['peak_share_night'][o] = hist[:, 0:6].sum(1) / days
            out['peak_share_morning'][o] = hist[:, 6:10].sum(1) / days
            out['peak_share_evening'][o] = hist[:, 16:22].sum(1) / days
            window3 = hist + np.roll(hist, -1, 1) + np.roll(hist, -2, 1)
            out['peak_hour_concentration'][o] = window3.max(1) / days
            Kf = K.reshape(c, -1)
            for name, idx in peaks.items():
                out[name][o] = np.nanmean(Kf[:, idx], 1) / mean
            out['grid_peak_kwh'][o] = np.nanmean(Kf[:, peaks['grid_peak_ratio']], 1)
        ref = np.array([np.nanmedian(out['peak_daily_rel'][group == g]) for g in range(config.N_PEER_GROUPS)])
        out['peak_daily_rel_vs_peers'] = out['peak_daily_rel'] - ref[group]
    for k, v in out.items():
        v[~np.isfinite(v)] = np.nan
    return out


def build_signatures(hd: HourlyData, D: dict) -> pl.DataFrame:
    f = {}
    f.update(solar_index_features(D, hd.temp_hourly, hd.dates))
    f.update(charging_block_features(hd.kwh, hd.temp, hd.weekend))
    f.update(cold_response_features(D, hd.kwh, hd.temp, hd.weekend))
    f.update(peak_features(hd.kwh, hd.group, hd.dates))
    return pl.DataFrame({'CUSTOMER_NUMBER': hd.customer_numbers, **f})


def load_signatures(hd: HourlyData, D: dict, rebuild: bool = False, verbose: bool = True) -> pl.DataFrame:
    """Signature features for every customer in ``hd`` (cached; about a minute to build)."""
    source = source_key()
    key = None if source is None else {'version': SIGNATURES_VERSION, 'source': source}
    _, tables, _ = cache.get_or_build('signatures', key, lambda: ({}, {'signatures': build_signatures(hd, D)}, {}),
                                      rebuild=rebuild, verbose=verbose)
    return tables['signatures']


def to_matrix(sig: pl.DataFrame, groups=ORIGINAL_GROUPS):
    """Feature matrix for the given groups (default: the original three, as used for the earlier
    'all signatures' models; pass ALL_GROUPS to include peak load)."""
    names = [f for g in groups for f in GROUPS[g]]
    X = sig.select(names).to_numpy().astype(float)
    X[~np.isfinite(X)] = np.nan
    return X, names
