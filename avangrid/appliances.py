"""Unlabeled appliance detection from hourly kWh and weather: air conditioning, electric heating,
electric water heaters and pools (Challenge Statement, Part 1).

There are no labels for these, so each detector is a physics-based rule on one or two evidence
features, with a confidence score:

    confidence = sigmoid(k * min_i (x_i - threshold_i) / scale_i)

i.e. how far the weakest piece of evidence sits above its threshold, in units of that feature's
spread across customers (robust SD). confidence >= 0.5 means every condition is met. Thresholds are
set from physical load sizes and checked by injecting simulated appliances into real customers
(see ``inject``), which measures how often a real appliance of that kind would be caught.

All functions take hourly kWh as customers x days x 24 (NaN = missing/imputed), so they work on the
cached grids and on simulated customers alike.
"""

import warnings

import numpy as np
import polars as pl

# --------------------------------------------------------------------------- helpers


def _daily_kwh(K):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(K, 2) * 24


def _regress(Y, X):
    """Per-customer OLS of Y (customers x days) on shared design X (days x terms).

    Missing days are filled with the customer's mean before fitting (few days are missing).
    Returns (beta: customers x terms, t-statistics: customers x terms).
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        Y = np.where(np.isnan(Y), np.nanmean(Y, 1, keepdims=True), Y)
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = Y @ (X @ XtX_inv)
    resid = Y - beta @ X.T
    dof = max(X.shape[0] - X.shape[1], 1)
    s2 = (resid ** 2).sum(1) / dof
    se = np.sqrt(np.maximum(s2[:, None] * np.diag(XtX_inv)[None, :], 1e-12))
    return beta, beta / se


def _rolling_mean(x, window):
    """Centered rolling mean along axis 1, ignoring NaN."""
    v, c = np.where(np.isnan(x), 0, x), (~np.isnan(x)).astype(float)
    z = np.zeros((x.shape[0], 1))
    cv, cc = np.concatenate([z, np.cumsum(v, 1)], 1), np.concatenate([z, np.cumsum(c, 1)], 1)
    n = x.shape[1]
    lo, hi = np.clip(np.arange(n) - window // 2, 0, n), np.clip(np.arange(n) + window // 2 + 1, 0, n)
    cnt = cc[:, hi] - cc[:, lo]
    return np.where(cnt > 0, (cv[:, hi] - cv[:, lo]) / np.maximum(cnt, 1), np.nan)


def _anomaly_slope(Y, x):
    """Per-customer slope and t-statistic of Y (customers x days) on a shared x, ignoring NaN in Y."""
    ok = ~np.isnan(Y)
    n = ok.sum(1)
    xb = np.where(ok, x[None, :], 0)
    mx = xb.sum(1) / np.maximum(n, 1)
    my = np.where(ok, Y, 0).sum(1) / np.maximum(n, 1)
    dx = np.where(ok, x[None, :] - mx[:, None], 0)
    dy = np.where(ok, Y - my[:, None], 0)
    sxx, sxy = (dx * dx).sum(1), (dx * dy).sum(1)
    slope = sxy / np.where(sxx > 0, sxx, np.nan)
    resid = dy - slope[:, None] * dx
    s2 = (resid ** 2).sum(1) / np.maximum(n - 2, 1)
    return slope, slope / np.sqrt(s2 / np.where(sxx > 0, sxx, np.nan))


def _robust_sd(x):
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) or np.std(x) or 1.0


# --------------------------------------------------------------------------- features


def appliance_features(K: np.ndarray, temp_hourly: np.ndarray, dates: pl.Series, weekend: np.ndarray,
                       chunk: int = 2000) -> dict[str, np.ndarray]:
    """Evidence features for each customer.

    Air conditioning (days averaging >= 55 F; daily kWh on cooling degrees above 65 F + weekend):
      ac_slope (kWh/day per degree), ac_t (its t-statistic), ac_share (share of Jun-Aug use explained
      by cooling), ac_added_kw (kWh/h at hour_ending 15-19 on days >= 80 F minus days 60-68 F).
    Electric heating (Nov-Mar; day-to-day anomalies, i.e. each day minus its 15-day centered mean,
      of daily kWh on the same anomalies of heating degrees below 55 F):
      heat_slope (kWh/day per degree), heat_t, heat_kw_at_20f (= slope x 35 / 24: average extra kW
      on a 20 F day), heat_share (share of Dec-Feb use explained), heat_added_kw (average kWh/h on
      days < 25 F minus days 55-65 F, a seasonal comparison kept for reference).
    Electric water heater (mild days 55-70 F, so heating/cooling are off; base = minimum of
      hour_ending 1-5 that day):
      wh_morning_share / wh_evening_share: share of days with an hour >= base + 2 kWh in
      hour_ending 5-10 / 17-22; wh_morning_kwh: median size of that morning spike above base.
    Pool (pump or heater running through summer daytime):
      pool_day_floor_jump: median mid-morning floor (minimum of hour_ending 10-12) on cool summer days
      (Jun 15-Aug 31, hottest hour <= 75 F) minus on mild spring/autumn days (Apr 1-May 20 and October,
      hottest hour <= 75 F, average >= 50 F). Requiring a cool afternoon and using mid-morning keeps
      air conditioning out of the signal;
      pool_day_night_jump: the same for (daytime floor - overnight floor), so loads running all day
      and night (e.g. a summer dehumidifier, or the whole home being occupied) count less.
    """
    n, nd, _ = K.shape
    temp = temp_hourly.mean(1)
    month = dates.dt.month().to_numpy()
    dom = dates.dt.day().to_numpy()
    jja = np.isin(month, [6, 7, 8])
    djf = np.isin(month, [12, 1, 2])
    cdd, hdd = np.maximum(temp - 65, 0), np.maximum(55 - temp, 0)

    warm = temp >= 55
    Xc = np.column_stack([np.ones(warm.sum()), cdd[warm], weekend[warm]])
    heating_season = np.isin(month, [11, 12, 1, 2, 3])
    hdd_anom = hdd - _rolling_mean(hdd[None, :], 15)[0]
    hot, mild_ac = temp >= 80, (temp >= 60) & (temp <= 68)
    very_cold, mild_h = temp < 25, (temp >= 55) & (temp <= 65)
    mild_wh = (temp >= 55) & (temp <= 70)
    # Pool days: the hottest hour stays <= 75 F, so air conditioning is mostly off (a 60-74 F daily
    # average can still reach the mid-80s in the afternoon, and AC then looks like a daytime floor).
    t_max = temp_hourly.max(1)
    summer_cool = (((month == 6) & (dom >= 15)) | np.isin(month, [7, 8])) & (t_max <= 75) & (temp >= 58)
    shoulder = ((month == 4) | ((month == 5) & (dom <= 20)) | (month == 10)) & (t_max <= 75) & (temp >= 50)

    names = ['ac_slope', 'ac_t', 'ac_share', 'ac_added_kw', 'heat_slope', 'heat_t', 'heat_kw_at_20f', 'heat_share',
             'heat_added_kw',
             'wh_morning_share', 'wh_evening_share', 'wh_morning_kwh', 'pool_day_floor_jump', 'pool_day_night_jump']
    out = {k: np.full(n, np.nan) for k in names}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for s0 in range(0, n, chunk):
            k = np.asarray(K[s0:s0 + chunk], dtype=np.float64)
            o = slice(s0, s0 + len(k))
            y = _daily_kwh(k)

            b, t = _regress(y[:, warm], Xc)
            out['ac_slope'][o], out['ac_t'][o] = b[:, 1], t[:, 1]
            out['ac_share'][o] = b[:, 1] * cdd[jja].sum() / np.nansum(y[:, jja], 1)
            aft = k[:, :, 14:19]
            out['ac_added_kw'][o] = np.nanmean(aft[:, hot], (1, 2)) - np.nanmean(aft[:, mild_ac], (1, 2))

            # Heating: respond to day-to-day temperature swings within Nov-Mar, not the seasonal trend.
            # Lighting, time indoors and furnace fans rise through winter for every home; only real
            # electric heating jumps on a cold day relative to the surrounding two weeks.
            ya = y - _rolling_mean(y, 15)
            slope, tstat = _anomaly_slope(ya[:, heating_season], hdd_anom[heating_season])
            out['heat_slope'][o], out['heat_t'][o] = slope, tstat
            out['heat_kw_at_20f'][o] = slope * 35 / 24
            out['heat_share'][o] = slope * hdd[djf].sum() / np.nansum(y[:, djf], 1)
            out['heat_added_kw'][o] = np.nanmean(k[:, very_cold], (1, 2)) - np.nanmean(k[:, mild_h], (1, 2))

            km = k[:, mild_wh]
            base = np.nanmin(km[:, :, 0:5], 2)
            morning = np.nanmax(km[:, :, 4:10], 2) - base
            evening = np.nanmax(km[:, :, 16:22], 2) - base
            spike_m = morning >= 2.0
            out['wh_morning_share'][o] = np.nanmean(np.where(np.isnan(morning), np.nan, spike_m), 1)
            out['wh_evening_share'][o] = np.nanmean(np.where(np.isnan(evening), np.nan, evening >= 2.0), 1)
            out['wh_morning_kwh'][o] = np.nanmedian(np.where(spike_m, morning, np.nan), 1)

            day_floor = np.nanmin(k[:, :, 9:12], 2)       # mid-morning (hour_ending 10-12): pumps on, AC rarely
            night_floor = np.nanmin(k[:, :, 0:5], 2)
            out['pool_day_floor_jump'][o] = (np.nanmedian(day_floor[:, summer_cool], 1)
                                             - np.nanmedian(day_floor[:, shoulder], 1))
            gap = day_floor - night_floor
            out['pool_day_night_jump'][o] = np.nanmedian(gap[:, summer_cool], 1) - np.nanmedian(gap[:, shoulder], 1)

            for name, v in _night_features(k, temp_hourly).items():
                out.setdefault(name, np.full(n, np.nan))[o] = v
            for name, v in _battery_features(k, month).items():
                out.setdefault(name, np.full(n, np.nan))[o] = v
    for v in out.values():
        v[~np.isfinite(v)] = np.nan
    return out


NIGHT_START = 19   # hour index 19 = hour_ending 20 (7-8 pm); the night window runs 11 hours to hour_ending 6
NIGHT_HOURS = 11


def _max_run(b: np.ndarray) -> np.ndarray:
    """Longest run of True along the last axis."""
    cur = np.zeros(b.shape[:-1])
    best = np.zeros(b.shape[:-1])
    for j in range(b.shape[-1]):
        cur = (cur + 1) * b[..., j]
        best = np.maximum(best, cur)
    return best


def _night_features(k: np.ndarray, temp_hourly: np.ndarray) -> dict[str, np.ndarray]:
    """EV charging (Level 1 / Level 2) and hot-tub evidence from overnight windows on mild nights.

    Night window: 8 pm to 7 am (hour_ending 20 to hour_ending 6 of the next day). Mild nights average
    42-65 F, so heating and air conditioning are mostly off. Each window hour is compared with the
    customer's 20th percentile for that hour across mild nights (an "uncharged night" baseline).
      ev_l1_night_share: share of mild nights with a run of >= 5 hours at 0.8-2.5 kWh above baseline
          (Level 1 charging: ~1.2-1.9 kW from a wall socket for many hours); ev_l1_excess_kwh: median
          size of that excess.
      ev_l2_night_share: share of mild nights with a run of >= 2 hours at >= 3 kWh above baseline
          (Level 2 charging: 3.3-11.5 kW for one to a few hours).
      hot_tub_night_share: share of mild nights with a short spike (1-2 hours, >= 1.5 kWh above that
          night's minimum) between midnight and 6 am. A hot-tub heater cycles on through the night all
          year; most other large loads are off at that time.
    """
    c, nd, _ = k.shape
    flat = k.reshape(c, -1)
    nn = nd - 1
    idx = (np.arange(nn) * 24 + NIGHT_START)[:, None] + np.arange(NIGHT_HOURS)[None, :]
    night_temp = temp_hourly.ravel()[idx].mean(1)
    mild = (night_temp >= 42) & (night_temp <= 65)
    w = flat[:, idx[mild]]                                   # customers x mild nights x 11 hours
    valid = ~np.all(np.isnan(w), 2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        base = np.nanpercentile(w, 20, axis=1)
        ex = w - base[:, None, :]
        l1 = (ex >= 0.8) & (ex <= 2.5)
        l1_night = _max_run(l1) >= 5
        l2_night = _max_run(ex >= 3.0) >= 2
        deep = w[:, :, 5:11]                                 # hour_ending 1-6
        spikes = (deep - np.nanmin(deep, 2, keepdims=True)) >= 1.5
        ht_night = spikes.any(2) & (_max_run(spikes) <= 2)
        n_valid = np.maximum(valid.sum(1), 1)
        return {
            'ev_l1_night_share': (l1_night & valid).sum(1) / n_valid,
            'ev_l1_excess_kwh': np.nanmedian(np.where(l1 & l1_night[:, :, None], ex, np.nan), (1, 2)),
            'ev_l2_night_share': (l2_night & valid).sum(1) / n_valid,
            'hot_tub_night_share': (ht_night & valid).sum(1) / n_valid,
        }


def _battery_features(k: np.ndarray, month: np.ndarray) -> dict[str, np.ndarray]:
    """Solar + battery evidence (Apr-Sep).

    solar_midday_zero_share: share of hours 11 am-3 pm (hour_ending 11-15) under 0.05 kWh (panels
      covering the home's use). battery_evening_zero_share: share of days with >= 3 of hour_ending 18-22
      under 0.1 kWh (a battery charged by the panels covering the evening, normally a solar home's
      highest-use time). mean_kwh: average use; summer_morning_kwh: average use at hour_ending 6-9 in
      Apr-Sep, to rule out homes that are empty in summer (near zero all day, not just in the evening).
      Apply the solar_battery rule only to homes with independent solar evidence (PV list or PV model).
    """
    summer = np.isin(month, [4, 5, 6, 7, 8, 9])
    ks = k[:, summer]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mid = ks[:, :, 10:15]
        eve = ks[:, :, 17:22]
        eve_zero_day = (np.where(np.isnan(eve), np.inf, eve) < 0.1).sum(2) >= 3
        return {
            'solar_midday_zero_share': np.nanmean(np.where(np.isnan(mid), np.nan, mid < 0.05), (1, 2)),
            'battery_evening_zero_share': eve_zero_day.mean(1),
            'mean_kwh': np.nanmean(k, (1, 2)),
            # Summer mornings (hour_ending 6-9), when a battery charged by the panels is usually spent:
            # rules out homes that are simply empty in summer, which are near zero all day.
            'summer_morning_kwh': np.nanmean(ks[:, :, 5:9], (1, 2)),
        }


# --------------------------------------------------------------------------- rules

# Each rule: list of (feature, threshold). Confidence uses the weakest feature's margin.
RULES = {
    'air_conditioning': [('ac_t', 4.0), ('ac_share', 0.10)],
    'electric_heating_primary': [('heat_t', 4.0), ('heat_kw_at_20f', 1.5)],
    'electric_heating_any': [('heat_t', 4.0), ('heat_kw_at_20f', 0.5)],
    # The thresholds below were set by injecting realistic simulated appliances into 2,000 real
    # customers on none of the lists: for each rule, the threshold that catches the most simulated
    # appliances while flagging no more than a plausible share of ordinary customers (water heater 15%,
    # pool and hot tub 4%, EV Level 1 and Level 2 2% each, battery 2% of solar homes). See
    # notebook 60 (calibration). A morning water-heater "spike" is already
    # >= 2 kWh above the overnight base, so only the share of days matters for that rule.
    'electric_water_heater': [('wh_morning_share', 0.15)],
    'pool': [('pool_day_floor_jump', 0.10), ('pool_day_night_jump', 0.30)],
    'hot_tub': [('hot_tub_night_share', 0.30)],
    'ev_level1': [('ev_l1_night_share', 0.40)],
    'ev_level2': [('ev_l2_night_share', 0.25)],
    'solar_battery': [('solar_midday_zero_share', 0.20), ('battery_evening_zero_share', 0.20), ('summer_morning_kwh', 0.25)],
}
CONFIDENCE_SHARPNESS = 3.0   # larger = confidence moves from 0 to 1 over a narrower band around the threshold


def fit_scales(F: dict[str, np.ndarray]) -> dict[str, float]:
    """Robust spread of every feature across (real) customers, used to scale rule margins."""
    return {f: _robust_sd(v) for f, v in F.items()}


def classify(F: dict[str, np.ndarray], scales: dict[str, float], thresholds: dict | None = None) -> dict[str, np.ndarray]:
    """Confidence (0-1) for each rule. thresholds: optional overrides {rule: [(feature, threshold), ...]}."""
    rules = {**RULES, **(thresholds or {})}
    out = {}
    for name, conds in rules.items():
        margins = np.column_stack([(np.nan_to_num(F[f], nan=-np.inf) - thr) / scales[f] for f, thr in conds])
        m = margins.min(1)
        out[name] = 1 / (1 + np.exp(-CONFIDENCE_SHARPNESS * np.clip(m, -20, 20)))
    return out


# --------------------------------------------------------------------------- simulation

def inject(K: np.ndarray, kind: str, temp_hourly: np.ndarray, dates: pl.Series, rng: np.random.Generator) -> np.ndarray:
    """Add a simulated appliance load to every customer in K (customers x days x 24). Returns a new array.

    air_conditioning: b * (hourly temp - 68 F), hour_ending 11-23, May-Sep; b ~ U(0.08, 0.20) kWh per
        degree-hour (about 2-4 kW at 90 F, a typical central system).
    electric_heating: b * (60 F - hourly temp) every hour; b ~ U(0.05, 0.15) (about 2-6 kW at 20 F).
    electric_water_heater: on 85% of days a 2-4.5 kWh draw in one hour of hour_ending 6-9, and on 60%
        of days a 1.5-4 kWh draw in hour_ending 18-22 (a 4.5 kW element running 20-60 minutes).
    pool: a 0.8-2.0 kW pump running 8-10 hours from 9 or 10 am, from about Jun 1 to Sep 7.
    """
    n, nd, _ = K.shape
    K = np.array(K, dtype=np.float64)
    month = dates.dt.month().to_numpy()
    doy = dates.dt.ordinal_day().to_numpy()
    if kind in ('air_conditioning', 'room_air_conditioning'):
        lo, hi = (0.08, 0.20) if kind == 'air_conditioning' else (0.02, 0.06)   # central vs one or two room units
        b = rng.uniform(lo, hi, n)[:, None, None]
        season = np.isin(month, [5, 6, 7, 8, 9])[None, :, None]
        hours = np.zeros(24, bool); hours[10:23] = True
        add = b * np.maximum(temp_hourly - 68, 0)[None] * season * hours[None, None, :]
    elif kind == 'electric_heating':
        b = rng.uniform(0.05, 0.15, n)[:, None, None]
        add = b * np.maximum(60 - temp_hourly, 0)[None]
    elif kind == 'heat_pump_heating':   # same heat delivered with ~3x efficiency, so about a third of the kW
        b = rng.uniform(0.015, 0.05, n)[:, None, None]
        add = b * np.maximum(60 - temp_hourly, 0)[None]
    elif kind == 'electric_water_heater':
        # Households differ: how often they draw hot water in the morning/evening, at what hour, how much.
        add = np.zeros((n, nd, 24))
        for (h_lo, h_hi), (p_lo, p_hi) in [((4, 9), (0.4, 0.95)), ((16, 21), (0.2, 0.7))]:
            p = rng.uniform(p_lo, p_hi, n)[:, None]
            usual = rng.integers(h_lo, h_hi + 1, n)[:, None]
            h = np.clip(usual + rng.integers(-1, 2, (n, nd)), 0, 23)          # usual hour +/- 1
            size = rng.uniform(1.0, 4.0, n)[:, None]
            amt = np.minimum(size * rng.uniform(0.6, 1.2, (n, nd)), 4.5) * (rng.random((n, nd)) < p)
            np.put_along_axis(add, h[:, :, None], np.take_along_axis(add, h[:, :, None], 2) + amt[:, :, None], 2)
    elif kind in ('pool', 'pool_variable_speed'):
        start = rng.integers(147, 160, n)[:, None]           # ~May 27 - Jun 8
        end = rng.integers(245, 255, n)[:, None]             # ~Sep 2 - Sep 12
        season = ((doy[None, :] >= start) & (doy[None, :] <= end))[:, :, None]
        hour = np.arange(24)[None, None, :]
        if kind == 'pool':                                    # single-speed pump, daytime
            kw = rng.uniform(0.8, 2.0, n)[:, None, None]
            first = rng.integers(8, 10, n)[:, None, None]
            length = rng.integers(8, 11, n)[:, None, None]
        else:                                                 # variable-speed pump, low power, long hours
            kw = rng.uniform(0.3, 0.8, n)[:, None, None]
            first = rng.integers(0, 9, n)[:, None, None]
            length = rng.integers(12, 25, n)[:, None, None]
        on = ((hour - first) % 24) < length
        add = kw * season * on
    elif kind == 'hot_tub':
        # Heater (1.5-4 kW) cycling on for part of an hour at random through day and night, more often
        # when it's cold, all year; plus 2-3 evening soaks a week with heater and jets running 1-2 hours.
        kw = rng.uniform(1.5, 4.0, n)[:, None, None]
        base_p = rng.uniform(0.08, 0.20, n)[:, None, None]
        p = np.clip(base_p * (1 + np.maximum(50 - temp_hourly, 0)[None] / 25), 0, 0.9)
        add = kw * rng.uniform(0.4, 1.0, (n, nd, 24)) * (rng.random((n, nd, 24)) < p)
        soak = rng.random((n, nd)) < rng.uniform(2, 3, n)[:, None] / 7
        start = rng.integers(18, 22, (n, nd))
        for d_h in range(2):
            h = np.clip(start + d_h, 0, 23)
            extra = np.where(soak, kw[:, :, 0] * 0.8, 0)
            np.put_along_axis(add, h[:, :, None], np.take_along_axis(add, h[:, :, None], 2) + extra[:, :, None], 2)
    elif kind in ('ev_level1', 'ev_level2'):
        # Charging sessions starting in the evening and running across midnight.
        flat = np.zeros((n, nd * 24 + 24))
        if kind == 'ev_level1':
            p, kw = rng.uniform(0.5, 0.9, n), rng.uniform(1.2, 1.9, n)
            dur_lo, dur_hi = 6, 12
        else:
            p, kw = rng.uniform(0.3, 0.8, n), rng.uniform(3.3, 7.6, n)
            dur_lo, dur_hi = 1, 4
        for i in range(n):
            days = np.flatnonzero(rng.random(nd) < p[i])
            for d in days:
                s = d * 24 + rng.integers(17, 23)
                dur = rng.integers(dur_lo, dur_hi + 1)
                flat[i, s:s + dur] += kw[i]
                flat[i, s + dur] += kw[i] * rng.uniform(0, 1)         # last partial hour
        add = flat[:, :nd * 24].reshape(n, nd, 24)
    elif kind == 'battery':
        # A 10-20 kWh battery charged by solar on sunny days (>= 3 midday hours at ~0 grid use; 30% charge
        # otherwise) and discharged from 4 pm to cover the home's use, at up to 5-7.68 kW.
        cap = rng.uniform(10, 20, n) * 0.9
        power = rng.uniform(5.0, 7.68, n)
        sunny = (np.where(np.isnan(K[:, :, 10:15]), np.inf, K[:, :, 10:15]) < 0.05).sum(2) >= 3
        energy = cap[:, None] * np.where(sunny, 1.0, 0.3)
        out = K.copy()
        for h in range(15, 24):
            take = np.minimum(np.nan_to_num(out[:, :, h]), np.minimum(power[:, None], energy))
            out[:, :, h] = out[:, :, h] - take
            energy = energy - take
        return out
    else:
        raise ValueError(kind)
    return np.where(np.isnan(K), np.nan, K + add)
