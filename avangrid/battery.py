"""Behind-the-meter battery simulation and time-of-use bill calculation.

Every customer's actual hourly load is run through a battery, hour by hour, with a transparent
rule-based dispatch, and their bill is computed before and after under a time-of-use tariff with an
optional monthly demand charge. All tariff values are PLACEHOLDERS: replace them with Avangrid's actual
rates before drawing conclusions.

Battery defaults are the Challenge Statement unit: Enphase IQ Battery 10C x 2 = 20.16 kWh, 7.68 kW
continuous, 89.4% round-trip efficiency (applied as sqrt(0.894) on charge and on discharge). Charging is
capped at 3 kW because the Challenge Statement limits a 200-amp residential service to ~3 kVA of
additional load without further evaluation (see Battery).

Dispatch strategies (each hour, in order):
  'tou'        On-peak: discharge to cover the home's load (never exports).
               In the charging window: charge, never pushing the home's load above its monthly threshold.
  'peak_shave' Any hour: discharge whatever load exceeds the monthly threshold.
               Otherwise charge in the charging window (same cap).
  'hybrid'     On-peak: as 'tou'. Off-peak: discharge above the threshold, otherwise charge in the window.
Charging happens in off-peak hours within a window (default midnight - 9 am; see CHARGE_WINDOWS) and is spread
evenly over the remaining hours of each charging block, so batteries don't all recharge at once when
on-peak ends (which would create a new "rebound" peak).
Charging windows can differ by customer (simulate(windows=..., window_of=...)): compare_windows runs every
window for everyone, assign_windows picks each customer's / segment's best, and coordinate_windows picks
each segment's mix of windows that maximizes savings under a system-peak cap (peak_savings_frontier traces
the tradeoff).
The monthly threshold is a percentile (default 95th) of the customer's own hourly load that month, so
charging fills the valleys without creating a new peak. Using the month's own load means the dispatch
knows each month's load in advance (a planning estimate, not a real-time controller).

Limits worth stating: the meter data are net of solar and never negative, so solar homes' exports (and
therefore free solar charging) are invisible; the battery charges from the grid off-peak instead.
"""

from dataclasses import dataclass, field

import numpy as np
import polars as pl


@dataclass
class Battery:
    """Battery and service constraints from the Challenge Statement.

    capacity_kwh, power_kw, round_trip_efficiency: Enphase IQ Battery 10C x 2 (20.16 kWh total, 7.68 kW
        continuous, 89.4% round trip). The 11.52 kW 10-second peak rating isn't modelled: hourly data
        can't resolve 10-second events.
    max_charge_kw: "A residential 200-amp service should not exceed approximately 3 kVA of additional
        incremental load without further evaluation." Charging the battery adds load to the service, so
        charging is capped at 3 kW (3 kVA at unity power factor). Discharging reduces the home's load, so
        it can use the full 7.68 kW.
    reserve_fraction: share of capacity held back for backup power (not used for bill savings).
    """
    capacity_kwh: float = 20.16
    power_kw: float = 7.68
    round_trip_efficiency: float = 0.894
    max_charge_kw: float = 3.0
    reserve_fraction: float = 0.0
    initial_soc_fraction: float = 0.5


@dataclass
class Tariff:
    """Placeholder time-of-use tariff ($/kWh by season and period) with a monthly demand charge."""
    summer_months: tuple = (6, 7, 8, 9)
    on_peak_start_hour_ending: int = 17    # 4-5 pm
    on_peak_end_hour_ending: int = 21      # 8-9 pm (inclusive)
    on_peak_weekdays_only: bool = True
    summer_on_peak: float = 0.30
    summer_off_peak: float = 0.12
    winter_on_peak: float = 0.22
    winter_off_peak: float = 0.11
    demand_charge_per_kw_month: float = 8.0
    demand_basis: str = 'on_peak'          # 'on_peak' = monthly max during on-peak hours; 'any' = any hour

    def hourly(self, dates: pl.Series) -> dict[str, np.ndarray]:
        """Per-hour arrays over the year (days x 24 flattened): rate, on-peak flag, month, summer flag."""
        nd = len(dates)
        month = np.repeat(dates.dt.month().to_numpy(), 24)
        weekday = np.repeat(dates.dt.weekday().to_numpy() <= 5, 24)
        he = np.tile(np.arange(1, 25), nd)
        summer = np.isin(month, self.summer_months)
        on = (he >= self.on_peak_start_hour_ending) & (he <= self.on_peak_end_hour_ending)
        if self.on_peak_weekdays_only:
            on &= weekday
        rate = np.where(summer, np.where(on, self.summer_on_peak, self.summer_off_peak),
                        np.where(on, self.winter_on_peak, self.winter_off_peak))
        return {'rate': rate, 'on_peak': on, 'month': month, 'summer': summer}


def fill_missing(K: np.ndarray, dates: pl.Series, chunk: int = 4000) -> np.ndarray:
    """Hourly kWh with missing hours filled by the customer's average for that hour of day in that month."""
    n, nd, _ = K.shape
    month = dates.dt.month().to_numpy()
    out = np.empty((n, nd, 24), dtype=np.float32)
    for s in range(0, n, chunk):
        k = np.array(K[s:s + chunk], dtype=np.float32)
        for m in range(1, 13):
            d = month == m
            km = k[:, d]
            fill = np.nanmean(km, axis=1, keepdims=True)                         # customer x 1 x 24
            fill = np.where(np.isnan(fill), np.nanmean(k, axis=(1, 2))[:, None, None], fill)
            k[:, d] = np.where(np.isnan(km), fill, km)
        out[s:s + chunk] = np.nan_to_num(k, nan=0.0)
    return out


# Charging windows (hour_ending values; None = any off-peak hour). With the same window at every home
# (all 33,310 customers, hybrid dispatch, 3 kW charging): 'midnight - 9 am' gave the largest cut in the
# grid's 100 highest hours (28.4 MW) with the system peak ~0.8% higher (everyone charges at once);
# 'any off-peak' saved the most ($13.7M vs $12.9M) but cut the grid's highest hours less (23.3 MW); '11 pm -
# 7 am' raised the system peak 4.5%. Mixing windows across customers (see assign_windows) avoids
# everyone charging at once.
CHARGE_WINDOWS = {
    'Midnight - 9 am': (1, 2, 3, 4, 5, 6, 7, 8, 9),
    'Any off-peak hour': None,
    '11 pm - 7 am': (24, 1, 2, 3, 4, 5, 6, 7),
    '9 pm - noon': (22, 23, 24, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
    '9 pm - 3 am': (22, 23, 24, 1, 2, 3),
    '3 am - 9 am': (4, 5, 6, 7, 8, 9),
    '10 am - 4 pm': (11, 12, 13, 14, 15, 16),
}
DEFAULT_CHARGE_WINDOW = CHARGE_WINDOWS['Midnight - 9 am']


def _hours_left_in_window(in_window: np.ndarray) -> np.ndarray:
    """For each hour in a charging window, how many window hours remain in that block (including this one)."""
    left = np.zeros(len(in_window), dtype=np.int32)
    run = 0
    for t in range(len(in_window) - 1, -1, -1):
        run = run + 1 if in_window[t] else 0
        left[t] = run
    return left


def to_hour_major(L: np.ndarray) -> np.ndarray:
    """customers x hours -> hours x customers float32 (contiguous), the layout simulate works in."""
    return np.ascontiguousarray(np.asarray(L, dtype=np.float32).T)


def _window_masks(windows: list, on: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """windows x hours: may the battery charge, and how many window hours are left in the block."""
    he = np.tile(np.arange(1, 25), len(on) // 24)
    iw = np.stack([~on & (np.isin(he, w) if w is not None else True) for w in windows])
    return iw, np.stack([_hours_left_in_window(r) for r in iw])


def simulate(L: np.ndarray, dates: pl.Series, battery: Battery = Battery(), tariff: Tariff = Tariff(),
             strategy: str = 'hybrid', threshold_quantile: float = 0.95, system_peak_hours: int = 100,
             charge_window: tuple | None = DEFAULT_CHARGE_WINDOW, max_charge_kw: float | None = None,
             keep_net_for: np.ndarray | None = None, windows: list | None = None,
             window_of: np.ndarray | None = None, group_of: np.ndarray | None = None,
             hour_major: bool = False) -> dict:
    """Simulate a battery at every customer.

    L: customers x hours (flattened days x 24) kWh, no missing values (see fill_missing); or, with
        hour_major=True, hours x customers float32 (to_hour_major), which is faster and can be shared.
    charge_window: hour_endings when the battery may charge (default midnight - 9 am; see CHARGE_WINDOWS), or
        None for any off-peak hour. Each battery spreads the energy it still needs evenly over
        the remaining hours of the current charging block, so batteries don't all recharge the moment
        on-peak ends (a "rebound" peak). With a short overnight window, many batteries charging together
        can create a new overnight system peak; charging across all off-peak hours avoids that.
    windows, window_of: a different window per customer: ``windows`` is a list of windows (as charge_window)
        and ``window_of`` gives each customer's index into it (overrides charge_window).
    group_of: optional group index per customer; adds 'system_after_by_group' (groups x hours).
    max_charge_kw: optional extra cap on charging power, on top of battery.max_charge_kw (3 kVA service limit).
    Returns per-customer statistics (polars DataFrame under 'customers'), system totals by hour before and
    after ('system_before', 'system_after'), and, for rows in ``keep_net_for``, the net load after the
    battery ('net'). grid_peak_kw_* is the customer's average load during the system's
    ``system_peak_hours`` highest hours (before the batteries): their contribution to the grid peak.
    """
    if strategy not in ('tou', 'peak_shave', 'hybrid'):
        raise ValueError(strategy)
    LT = L if hour_major else to_hour_major(L)             # hours x customers: each hour's loads contiguous
    T, n = LT.shape
    h = tariff.hourly(dates)
    rate, on, month = h['rate'], h['on_peak'], h['month']
    eta = np.sqrt(battery.round_trip_efficiency)
    cap = battery.capacity_kwh
    floor = battery.reserve_fraction * cap
    P = battery.power_kw
    P_charge = min(P, battery.max_charge_kw, max_charge_kw or np.inf)     # 3 kVA service limit by default

    thr = np.empty((n, 12), dtype=np.float32)
    for m in range(1, 13):
        thr[:, m - 1] = np.quantile(LT[month == m], threshold_quantile, axis=0)

    system_before = LT.sum(1, dtype=np.float64)
    peak_idx = np.argsort(-system_before)[:system_peak_hours]
    is_peak_hour = np.zeros(T, bool); is_peak_hour[peak_idx] = True
    grid_peak_load_before = LT[peak_idx].mean(0, dtype=np.float64)
    grid_peak_load_after = np.zeros(n)

    if windows is None:
        windows, window_of = [charge_window], np.zeros(n, dtype=int)
    window_of = np.asarray(window_of, dtype=int)
    single = len(windows) == 1
    IW, LEFT = _window_masks(windows, on)

    soc = np.full(n, battery.initial_soc_fraction * cap, dtype=np.float64)
    keep = np.asarray(keep_net_for if keep_net_for is not None else [], dtype=int)
    net_keep = np.empty((len(keep), T), dtype=np.float32)
    system_after = np.empty(T)
    if group_of is not None:
        group_of = np.asarray(group_of, dtype=int)
        G = int(group_of.max()) + 1
        system_by_group = np.empty((G, T))
    on_kwh_after = np.zeros(n); off_kwh_after = np.zeros(n); cost_after = np.zeros(n)
    on_kwh_before = np.zeros(n); off_kwh_before = np.zeros(n); cost_before = np.zeros(n); annual_max = np.zeros(n)
    discharged = np.zeros(n); charged = np.zeros(n)
    month_max_after = np.zeros((n, 12)); month_max_before = np.zeros((n, 12))
    demand_mask = on if tariff.demand_basis == 'on_peak' else np.ones(T, bool)

    for t in range(T):
        l = LT[t].astype(np.float64)
        cost_before += l * rate[t]
        np.maximum(annual_max, l, out=annual_max)
        th = thr[:, month[t] - 1]
        if on[t] and strategy in ('tou', 'hybrid'):
            want = l
        elif strategy in ('peak_shave', 'hybrid'):
            want = np.maximum(l - th, 0.0)
        else:
            want = np.zeros(n)
        d = np.minimum(np.minimum(want, P), np.maximum(soc - floor, 0.0) * eta)
        soc -= d / eta
        if single:
            iw, lf = IW[0, t], max(LEFT[0, t], 1)
        else:
            iw, lf = IW[:, t][window_of], np.maximum(LEFT[:, t][window_of], 1)
        if np.any(iw):
            need = (cap - soc) / eta                                   # grid kWh to fill the battery
            c = np.where(iw & (want <= 0), np.minimum(np.minimum(P_charge, need / lf), np.maximum(th - l, 0.0)), 0.0)
        else:
            c = np.zeros(n)
        c = np.maximum(c, 0.0)
        soc += c * eta
        net = l - d + c
        discharged += d; charged += c
        cost_after += net * rate[t]
        if on[t]:
            on_kwh_after += net; on_kwh_before += l
        else:
            off_kwh_after += net; off_kwh_before += l
        if demand_mask[t]:
            mi = month[t] - 1
            np.maximum(month_max_after[:, mi], net, out=month_max_after[:, mi])
            np.maximum(month_max_before[:, mi], l, out=month_max_before[:, mi])
        system_after[t] = net.sum()
        if group_of is not None:
            system_by_group[:, t] = np.bincount(group_of, net, minlength=G)
        if is_peak_hour[t]:
            grid_peak_load_after += net / len(peak_idx)
        if len(keep):
            net_keep[:, t] = net[keep]

    months_present = np.unique(month)
    demand_before = tariff.demand_charge_per_kw_month * month_max_before[:, months_present - 1].sum(1)
    demand_after = tariff.demand_charge_per_kw_month * month_max_after[:, months_present - 1].sum(1)

    stats = pl.DataFrame({
        'bill_before': cost_before + demand_before,
        'bill_after': cost_after + demand_after,
        'energy_cost_before': cost_before, 'energy_cost_after': cost_after,
        'demand_charge_before': demand_before, 'demand_charge_after': demand_after,
        'on_peak_kwh_before': on_kwh_before, 'on_peak_kwh_after': on_kwh_after,
        'off_peak_kwh_before': off_kwh_before, 'off_peak_kwh_after': off_kwh_after,
        'avg_monthly_peak_kw_before': month_max_before[:, months_present - 1].mean(1),
        'avg_monthly_peak_kw_after': month_max_after[:, months_present - 1].mean(1),
        'annual_peak_kw_before': annual_max,
        'battery_discharged_kwh': discharged, 'battery_charged_kwh': charged,
        'grid_peak_kw_before': grid_peak_load_before, 'grid_peak_kw_after': grid_peak_load_after,
    }).with_columns(
        (pl.col('grid_peak_kw_before') - pl.col('grid_peak_kw_after')).alias('grid_peak_kw_reduction'),
        (pl.col('bill_before') - pl.col('bill_after')).alias('bill_savings'),
        pl.when(pl.col('bill_before') > 0).then((pl.col('bill_before') - pl.col('bill_after')) / pl.col('bill_before'))
          .alias('bill_savings_pct'),
        (pl.col('energy_cost_before') - pl.col('energy_cost_after')).alias('energy_savings'),
        (pl.col('demand_charge_before') - pl.col('demand_charge_after')).alias('demand_savings'),
        (pl.col('on_peak_kwh_before') - pl.col('on_peak_kwh_after')).alias('on_peak_kwh_decrease'),
        (pl.col('off_peak_kwh_after') - pl.col('off_peak_kwh_before')).alias('off_peak_kwh_increase'),
        (pl.col('avg_monthly_peak_kw_before') - pl.col('avg_monthly_peak_kw_after')).alias('avg_monthly_peak_kw_reduction'),
        (pl.col('battery_charged_kwh') - pl.col('battery_discharged_kwh')).alias('losses_kwh'),
        (pl.col('battery_discharged_kwh') / (battery.capacity_kwh * eta)).alias('equivalent_full_cycles'),
    )
    return {'customers': stats, 'system_before': system_before, 'system_after': system_after,
            'system_peak_hours': peak_idx, 'net': net_keep, 'thresholds': thr, 'window_of': window_of,
            'system_after_by_group': system_by_group if group_of is not None else None}


def system_peak_reduction(result: dict) -> dict[str, float]:
    """System-level peak (sum of all simulated customers): highest hour before and after, in kW.

    'after' is the highest hour after the batteries, wherever it now falls (the peak can move).
    """
    sb, sa = result['system_before'], result['system_after']
    return {'system_peak_kw_before': float(sb.max()), 'system_peak_kw_after': float(sa.max()),
            'system_peak_kw_reduction': float(sb.max() - sa.max())}


# ---------------------------------------------------------------- charging windows per customer / segment

def _window_run(LT, dates, window, group_of, kw):
    r = simulate(LT, dates, charge_window=window, group_of=group_of, hour_major=True, **kw)
    c = r['customers']
    return (c['bill_savings'].to_numpy(), c['grid_peak_kw_reduction'].to_numpy(), r['system_after'],
            r['system_after_by_group'], r['system_before'], float(c['bill_before'].sum()))


def compare_windows(L: np.ndarray, dates: pl.Series, windows: dict | None = None,
                    segments: np.ndarray | None = None, by_best_window: bool = True, n_jobs: int = 8, **kw) -> dict:
    """Run simulate once per window (the same window at every home), windows in parallel.

    Returns names, savings (customers x windows, $/yr), grid_cut (customers x windows, kW in the grid's top
    hours), system_after (windows x hours) and a summary table. If ``segments`` (one label per customer) is
    given, also the hourly system load of each group under each window ('system_by_group': windows x groups x
    hours), used by coordinate_windows. Groups are the segments, or, with by_best_window (default), segment x
    the customer's own best window (needs a second set of runs, but lets the plan keep customers on the window
    that suits them). kw goes to simulate.
    """
    from joblib import Parallel, delayed
    windows = CHARGE_WINDOWS if windows is None else windows
    names = list(windows)
    LT = to_hour_major(L)        # shared with the workers (joblib memory-maps large arrays)

    def run_all(group_of):
        return Parallel(n_jobs=min(n_jobs, len(names)), backend='loky')(
            delayed(_window_run)(LT, dates, windows[nm], group_of, kw) for nm in names)

    seg = np.asarray(segments) if segments is not None else None
    labels = None if seg is None else seg
    out = run_all(None if seg is None else np.unique(labels, return_inverse=True)[1])
    savings = np.column_stack([o[0] for o in out])
    if seg is not None and by_best_window:
        best = np.asarray(names, dtype=object)[savings.argmax(1)]
        labels = np.array([f'{s} | {b}' for s, b in zip(seg, best)], dtype=object)
        out = run_all(np.unique(labels, return_inverse=True)[1])
    group_names, group_idx = (np.unique(labels, return_inverse=True) if labels is not None else (None, None))
    rows = [{'window': nm, 'total_savings': float(o[0].sum()), 'savings_share_of_bill': float(o[0].sum()) / out[0][5],
             'grid_peak_hour_cut_mw': float(o[1].sum()) / 1000,
             'system_peak_mw_after': float(o[2].max()) / 1000} for nm, o in zip(names, out)]
    return {'names': names, 'windows': [windows[n] for n in names], 'savings': savings,
            'grid_cut': np.column_stack([o[1] for o in out]), 'system_after': np.stack([o[2] for o in out]),
            'system_before': out[0][4], 'bill_before_total': out[0][5], 'summary': pl.DataFrame(rows),
            'group_names': group_names, 'group_index': group_idx,
            'system_by_group': np.stack([o[3] for o in out]) if labels is not None else None}


def staggered_shares(system_after: np.ndarray, savings: np.ndarray, peak_tolerance: float = 0.002) -> np.ndarray:
    """Share of customers to put on each window so the combined system load has the lowest possible peak.

    If customers are spread over windows without regard to their load, the system load is (in expectation)
    sum_w share_w * system_after[w], so the shares come from a linear program: minimize the highest hour.
    A second pass keeps the peak within ``peak_tolerance`` of that minimum and maximizes total savings.
    """
    from scipy.optimize import linprog
    W, T = system_after.shape
    # variables: shares (W), z.  min z  s.t.  system_after.T @ s - z <= 0,  sum s = 1
    A = np.hstack([system_after.T, -np.ones((T, 1))])
    eq = np.append(np.ones(W), 0.0)[None]
    r = linprog(np.append(np.zeros(W), 1.0), A_ub=A, b_ub=np.zeros(T), A_eq=eq, b_eq=[1.0],
                bounds=[(0, 1)] * W + [(0, None)], method='highs')
    z = r.x[-1]
    r2 = linprog(-savings.sum(0), A_ub=system_after.T, b_ub=np.full(T, z * (1 + peak_tolerance)),
                 A_eq=np.ones((1, W)), b_eq=[1.0], bounds=[(0, 1)] * W, method='highs')
    s = r2.x if r2.success else r.x[:W]
    s = np.clip(s, 0, None)
    return s / s.sum()


def assign_by_shares(savings: np.ndarray, shares: np.ndarray, prefer_own: bool = True, seed: int = 0) -> np.ndarray:
    """Assign customers to windows in the given shares. prefer_own: customers who lose most by not getting their
    best window choose first (largest gap between best and second-best savings); otherwise random."""
    n, W = savings.shape
    counts = np.floor(shares * n).astype(int)
    counts[np.argsort(-(shares * n - counts))[:n - counts.sum()]] += 1
    rng = np.random.default_rng(seed)
    if not prefer_own:
        return rng.permutation(np.repeat(np.arange(W), counts))
    srt = np.sort(savings, 1)
    regret = srt[:, -1] - srt[:, -2] if W > 1 else np.zeros(n)
    pref = np.argsort(-savings, 1)
    out = np.empty(n, dtype=int)
    left = counts.copy()
    for i in np.argsort(-regret + rng.random(n) * 1e-9):
        for w in pref[i]:
            if left[w] > 0:
                out[i] = w; left[w] -= 1
                break
    return out


def assign_windows(cmp: dict, method: str, groups: np.ndarray | None = None) -> np.ndarray:
    """Each customer's window index (into cmp['names']).

    'customer'  the window that saves that customer the most;
    'segment'   the window that saves each segment (``groups``) the most in total;
    'staggered' window shares chosen to minimize the system peak (staggered_shares), customers placed in
                their own best window where the shares allow (assign_by_shares).
    """
    sav = cmp['savings']
    if method == 'customer':
        return sav.argmax(1)
    if method == 'segment':
        groups = np.asarray(groups)
        out = np.empty(len(groups), dtype=int)
        for g in np.unique(groups):
            m = groups == g
            out[m] = sav[m].sum(0).argmax()
        return out
    if method == 'staggered':
        return assign_by_shares(sav, staggered_shares(cmp['system_after'], sav))
    raise ValueError(method)


def coordinate_windows(cmp: dict, peak_cap_kw: float | None = None, seed: int = 0) -> dict:
    """Choose each group's (segment, or segment x own best window) mix of charging windows to maximize total savings while keeping the system peak
    at or below ``peak_cap_kw`` (default: the lowest peak achievable, + 0.2%).

    Within a group, customers on a window are a random draw, so the system load is (in expectation) linear in
    the shares x[group, window]: the sum over groups and windows of x times that group's hourly load if the
    whole group used that window. That makes it a linear program over x (groups x windows), solved twice: first
    the lowest achievable peak, then the most savings under the cap. Customers are then drawn at random within
    each group in those shares; on the earlier 24,968-customer run the simulated peak and savings matched the linear
    program's to within 0.05 MW and $0.01M.

    Needs compare_windows(..., segments=...). Returns window_of (per customer), shares (groups x windows),
    min_peak_kw and the cap used.
    """
    from scipy.optimize import linprog
    S = cmp['system_by_group']                             # W x G x T
    W, G, T = S.shape
    seg = cmp['group_index']
    sav_g = np.stack([cmp['savings'][seg == g].sum(0) for g in range(G)])   # G x W, $ if the whole group used w
    coef = S.transpose(1, 0, 2).reshape(G * W, T)          # row g*W + w
    eq = np.kron(np.eye(G), np.ones(W))                    # each group's shares sum to 1
    nv = G * W
    r = linprog(np.append(np.zeros(nv), 1.0), A_ub=np.hstack([coef.T, -np.ones((T, 1))]), b_ub=np.zeros(T),
                A_eq=np.hstack([eq, np.zeros((G, 1))]), b_eq=np.ones(G), bounds=[(0, 1)] * nv + [(0, None)],
                method='highs')
    min_peak = float(r.x[-1])
    cap = min_peak * 1.002 if peak_cap_kw is None else max(float(peak_cap_kw), min_peak * 1.002)
    cap = min(cap, 1e12)
    r2 = linprog(-sav_g.reshape(-1), A_ub=coef.T, b_ub=np.full(T, cap), A_eq=eq, b_eq=np.ones(G),
                 bounds=[(0, 1)] * nv, method='highs')
    x = (r2.x if r2.success else r.x[:nv]).reshape(G, W).clip(0)
    x /= x.sum(1, keepdims=True)
    out = np.empty(len(seg), dtype=int)
    rng = np.random.default_rng(seed)
    for g in range(G):
        idx = np.flatnonzero(seg == g)
        out[idx] = assign_by_shares(np.zeros((len(idx), W)), x[g], prefer_own=False, seed=int(rng.integers(1 << 31)))
    return {'window_of': out, 'shares': x, 'min_peak_kw': min_peak, 'peak_cap_kw': cap,
            'expected_peak_kw': float((x.reshape(-1) @ coef).max()), 'expected_savings': float(x.reshape(-1) @ sav_g.reshape(-1))}


def peak_savings_frontier(cmp: dict, points: int = 9) -> pl.DataFrame:
    """Most savings achievable at each system-peak cap, from the lowest achievable peak up to the peak of the
    savings-maximizing plan (linear-program expectations; see coordinate_windows)."""
    hi = coordinate_windows(cmp, np.inf)
    lo = coordinate_windows(cmp, None)
    rows = []
    for cap in np.linspace(lo['min_peak_kw'] * 1.002, hi['expected_peak_kw'], points):
        c = coordinate_windows(cmp, cap)
        rows.append({'peak_cap_kw': float(cap), 'system_peak_kw': c['expected_peak_kw'],
                     'total_savings': c['expected_savings'],
                     'savings_share_of_bill': c['expected_savings'] / cmp['bill_before_total']})
    return pl.DataFrame(rows)
