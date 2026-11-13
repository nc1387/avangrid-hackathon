"""Battery scenarios for the web app (app/server.py): parse settings, run and cache simulations, and shape the
results as JSON-ready dicts.

A scenario is a dict with five parts (missing keys take the defaults in DEFAULTS):
  battery   capacity_kwh, power_kw, round_trip_efficiency, max_charge_kw, reserve_fraction
  tariff    summer_months, on_peak_start / on_peak_end (24-hour clock, e.g. 16 and 21 = 4-9 pm), weekdays_only,
            summer/winter on/off-peak $/kWh, demand_charge ($/kW-month), demand_basis ('on_peak' or 'any')
  dispatch  strategy ('hybrid', 'tou', 'peak_shave'), threshold_quantile, peak_hours (grid hours to report on)
  plan      mode ('coordinated', 'customer', 'segment', 'same'), peak_cut (coordinated: target cut in the system
            peak, as a fraction), window (same), segment_windows ({segment: window}; missing = segment's best)
  filters   segments (list, or null for all), flags (program flags every customer must have)

Data come from the local data folder (avangrid.config.DATA_DIR, or AVANGRID_DATA_DIR) and are loaded once.
Results are cached by scenario, so repeated or shared requests are instant; runs are kept in a small LRU.
"""

import copy
import hashlib
import json
import threading
from collections import OrderedDict

import numpy as np
import polars as pl

from avangrid import battery as bt, db, grids

FLAGS = {'peak_contributor': 'Peak contributors', 'high_electrification': 'High electrification',
         'flexible_load': 'Flexible load', 'hp_retrofit_candidate': 'Heat-pump retrofit candidates',
         'ev_adoption_candidate': 'EV adoption candidates', 'solar': 'Solar', 'seasonal': 'Seasonal'}
MONTH_ORDER = [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8]   # the data year runs Sep 2025 - Aug 2026
MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

DEFAULTS = {
    'battery': {'capacity_kwh': 20.16, 'power_kw': 7.68, 'round_trip_efficiency': 0.894, 'max_charge_kw': 3.0,
                'reserve_fraction': 0.0},
    'tariff': {'summer_months': [6, 7, 8, 9], 'on_peak_start': 16, 'on_peak_end': 21, 'weekdays_only': True,
               'summer_on_peak': 0.30, 'summer_off_peak': 0.12, 'winter_on_peak': 0.22, 'winter_off_peak': 0.11,
               'demand_charge': 8.0, 'demand_basis': 'on_peak'},
    'dispatch': {'strategy': 'hybrid', 'threshold_quantile': 0.95, 'peak_hours': 100},
    'plan': {'mode': 'coordinated', 'peak_cut': 0.075, 'window': 'Midnight - 9 am', 'segment_windows': {}},
    'filters': {'segments': None, 'flags': []},
}

_lock = threading.Lock()          # one heavy computation at a time (each holds ~2 GB of arrays)
_data: dict | None = None
_compare_cache: OrderedDict = OrderedDict()
_frontier_cache: OrderedDict = OrderedDict()
_runs: OrderedDict = OrderedDict()


def _key(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _remember(cache: OrderedDict, key, value, size: int):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > size:
        cache.popitem(last=False)
    return value


# ------------------------------------------------------------------------------------------- data
def load() -> dict:
    """Hourly loads (gaps filled), customer numbers, segments and program flags. Loaded once (~10 s)."""
    global _data
    if _data is None:
        with _lock:
            if _data is None:
                hd = grids.load_hourly(verbose=False)
                L = bt.fill_missing(hd.kwh, hd.dates).reshape(hd.shape[0], -1)
                seg = (pl.DataFrame({'CUSTOMER_NUMBER': hd.customer_numbers})
                       .join(db.sql('SELECT * FROM customer_segments'), on='CUSTOMER_NUMBER', how='left'))
                _data = {'dates': hd.dates, 'customers': np.asarray(hd.customer_numbers), 'L': L,
                         'segment': seg['segment'].fill_null('Unassigned').to_numpy().astype(object),
                         'flags': {f: seg[f].fill_null(False).to_numpy() for f in FLAGS if f in seg.columns},
                         'row_of': {int(c): i for i, c in enumerate(hd.customer_numbers)}}
    return _data


def meta() -> dict:
    d = load()
    segs, counts = np.unique(d['segment'], return_counts=True)
    return {'customers': int(len(d['customers'])), 'first_date': str(d['dates'][0]), 'last_date': str(d['dates'][-1]),
            'segments': [{'segment': str(s), 'customers': int(c)} for s, c in zip(segs, counts)],
            'flags': [{'flag': f, 'label': FLAGS[f], 'customers': int(v.sum())} for f, v in d['flags'].items()],
            'windows': list(bt.CHARGE_WINDOWS), 'defaults': DEFAULTS,
            'plan_modes': {'coordinated': 'Coordinated: most savings under a system-peak target',
                           'customer': 'Best window for each customer', 'segment': 'Chosen per segment',
                           'same': 'Same window for everyone'}}


# ------------------------------------------------------------------------------------------- scenario parts
def normalize(scenario: dict | None) -> dict:
    s = copy.deepcopy(DEFAULTS)
    for part, values in (scenario or {}).items():
        if part in s and isinstance(values, dict):
            s[part].update({k: v for k, v in values.items() if v is not None or k == 'segments'})
    s['tariff']['summer_months'] = sorted(int(m) for m in s['tariff']['summer_months'])
    return s


def _battery_tariff(s: dict):
    b, t = s['battery'], s['tariff']
    battery = bt.Battery(b['capacity_kwh'], b['power_kw'], b['round_trip_efficiency'],
                         max_charge_kw=b['max_charge_kw'], reserve_fraction=b['reserve_fraction'])
    tariff = bt.Tariff(summer_months=tuple(t['summer_months']), on_peak_start_hour_ending=int(t['on_peak_start']) + 1,
                       on_peak_end_hour_ending=int(t['on_peak_end']), on_peak_weekdays_only=bool(t['weekdays_only']),
                       summer_on_peak=t['summer_on_peak'], summer_off_peak=t['summer_off_peak'],
                       winter_on_peak=t['winter_on_peak'], winter_off_peak=t['winter_off_peak'],
                       demand_charge_per_kw_month=t['demand_charge'], demand_basis=t['demand_basis'])
    return battery, tariff


def _rows(filters: dict) -> np.ndarray:
    d = load()
    mask = np.ones(len(d['customers']), bool)
    if filters.get('segments') is not None:
        mask &= np.isin(d['segment'], list(filters['segments']))
    for f in filters.get('flags') or []:
        if f in d['flags']:
            mask &= d['flags'][f]
    return np.flatnonzero(mask)


def _loads(rows: np.ndarray) -> np.ndarray:
    L = load()['L']
    return L if len(rows) == L.shape[0] else L[rows]


def _sim_kwargs(s: dict) -> dict:
    battery, tariff = _battery_tariff(s)
    return {'battery': battery, 'tariff': tariff, 'strategy': s['dispatch']['strategy'],
            'threshold_quantile': s['dispatch']['threshold_quantile'],
            'system_peak_hours': int(s['dispatch']['peak_hours'])}


def _compare(s: dict, rows: np.ndarray) -> dict:
    """Every window applied to everyone (cached by battery/tariff/dispatch/filters)."""
    key = _key([s['battery'], s['tariff'], s['dispatch'], s['filters']])
    if key not in _compare_cache:
        cmp = bt.compare_windows(_loads(rows), load()['dates'], segments=load()['segment'][rows], **_sim_kwargs(s))
        _remember(_compare_cache, key, cmp, 3)
    return _compare_cache[key]


# ------------------------------------------------------------------------------------------- run
def simulate(scenario: dict | None) -> dict:
    """Run (or fetch from cache) a scenario; returns the full results payload, including run_id."""
    s = normalize(scenario)
    run_id = _key(s)
    if run_id in _runs:
        _runs.move_to_end(run_id)
        return _runs[run_id]['payload']
    d = load()
    with _lock:
        rows = _rows(s['filters'])
        if not len(rows):
            raise ValueError('No customers match the filters.')
        plan = s['plan']
        info = {}
        segs = d['segment'][rows]
        if plan['mode'] == 'same':
            if plan['window'] not in bt.CHARGE_WINDOWS:
                raise ValueError(f"Unknown window: {plan['window']}")
            names, windows, window_of = [plan['window']], [bt.CHARGE_WINDOWS[plan['window']]], np.zeros(len(rows), int)
        else:
            cmp = _compare(s, rows)
            names, windows = cmp['names'], cmp['windows']
            if plan['mode'] == 'customer':
                window_of = bt.assign_windows(cmp, 'customer')
            elif plan['mode'] == 'segment':
                best = bt.assign_windows(cmp, 'segment', segs)
                chosen = plan.get('segment_windows') or {}
                window_of = np.array([names.index(chosen[g]) if chosen.get(g) in names else b
                                      for g, b in zip(segs, best)])
            elif plan['mode'] == 'coordinated':
                co = bt.coordinate_windows(cmp, cmp['system_before'].max() * (1 - float(plan['peak_cut'])))
                window_of = co['window_of']
                info = {'min_peak_kw': co['min_peak_kw'], 'peak_cap_kw': co['peak_cap_kw'],
                        'target_peak_kw': float(cmp['system_before'].max() * (1 - float(plan['peak_cut']))),
                        'target_met': bool(co['peak_cap_kw'] <= cmp['system_before'].max() * (1 - float(plan['peak_cut'])) + 1)}
            else:
                raise ValueError(f"Unknown plan mode: {plan['mode']}")
        res = bt.simulate(_loads(rows), d['dates'], windows=windows, window_of=window_of, **_sim_kwargs(s))
    tariff = _sim_kwargs(s)['tariff'].hourly(d['dates'])
    cust = res['customers'].with_columns(
        pl.Series('CUSTOMER_NUMBER', d['customers'][rows]), pl.Series('segment', segs.astype(str)),
        pl.Series('charge_window', np.array(names, dtype=object)[window_of].astype(str)),
        *[pl.Series(f, v[rows]) for f, v in d['flags'].items()])
    payload = _payload(run_id, s, cust, res, tariff, info)
    _remember(_runs, run_id, {'scenario': s, 'rows': rows, 'customers': cust, 'payload': payload}, 8)
    return payload


def _get_run(run_id: str) -> dict:
    if run_id not in _runs:
        raise KeyError(run_id)
    return _runs[run_id]


def _payload(run_id: str, s: dict, c: pl.DataFrame, res: dict, tariff: dict, info: dict) -> dict:
    sb, sa = res['system_before'], res['system_after']
    on, month = tariff['on_peak'], tariff['month']
    tot = lambda col: float(c[col].sum())  # noqa: E731
    summary = {
        'customers': c.height, 'peak_hours': int(s['dispatch']['peak_hours']),
        'bill_before': tot('bill_before'), 'bill_after': tot('bill_after'),
        'total_savings': tot('bill_savings'), 'savings_share_of_bill': tot('bill_savings') / tot('bill_before'),
        'energy_savings': tot('energy_savings'), 'demand_savings': tot('demand_savings'),
        'avg_savings': float(c['bill_savings'].mean()), 'median_savings': float(c['bill_savings'].median()),
        'avg_savings_pct': float(c['bill_savings_pct'].mean()), 'median_savings_pct': float(c['bill_savings_pct'].median()),
        'on_peak_kwh_removed': tot('on_peak_kwh_decrease'),
        'on_peak_share_removed': tot('on_peak_kwh_decrease') / tot('on_peak_kwh_before'),
        'off_peak_kwh_added': tot('off_peak_kwh_increase'),
        'off_peak_share_added': tot('off_peak_kwh_increase') / tot('off_peak_kwh_before'),
        'avg_monthly_peak_cut_kw': float(c['avg_monthly_peak_kw_reduction'].mean()),
        'monthly_peak_cut_share': tot('avg_monthly_peak_kw_reduction') / tot('avg_monthly_peak_kw_before'),
        'grid_peak_cut_kw': tot('grid_peak_kw_reduction'),
        'grid_peak_cut_share': tot('grid_peak_kw_reduction') / tot('grid_peak_kw_before'),
        'system_peak_before_kw': float(sb.max()), 'system_peak_after_kw': float(sa.max()),
        'losses_kwh': tot('losses_kwh'), 'median_cycles': float(c['equivalent_full_cycles'].median()),
    }
    dates = load()['dates']

    def day(series_idx):
        dd = int(series_idx) // 24
        return {'date': str(dates[dd]), 'peak_hour_ending': int(series_idx) % 24 + 1,
                'before': (sb[dd * 24:(dd + 1) * 24] / 1000).round(3).tolist(),
                'after': (sa[dd * 24:(dd + 1) * 24] / 1000).round(3).tolist(),
                'on_peak': on[dd * 24:(dd + 1) * 24].tolist()}

    monthly = []
    for m in MONTH_ORDER:
        sel = month == m
        monthly.append({'month': MONTH_NAMES[m - 1],
                        'on_peak_change_mwh': float((sa[sel & on] - sb[sel & on]).sum()) / 1000,
                        'off_peak_change_mwh': float((sa[sel & ~on] - sb[sel & ~on]).sum()) / 1000})

    def hist(x, bins=40):
        x = x[np.isfinite(x)]
        counts, edges = np.histogram(x, bins=bins)
        return {'edges': edges.round(4).tolist(), 'counts': counts.tolist()}

    segments = (c.group_by('segment').agg(
        pl.len().alias('customers'),
        pl.col('bill_savings').sum().alias('total_savings'),
        pl.col('bill_savings').mean().alias('avg_savings'),
        pl.col('bill_savings').median().alias('median_savings'),
        (pl.col('bill_savings').sum() / pl.col('bill_before').sum()).alias('savings_share_of_bill'),
        pl.col('bill_savings_pct').mean().alias('avg_savings_pct'),
        pl.col('bill_savings_pct').median().alias('median_savings_pct'),
        pl.col('avg_monthly_peak_kw_reduction').mean().alias('avg_monthly_peak_cut_kw'),
        pl.col('grid_peak_kw_reduction').mean().alias('grid_peak_cut_kw_per_customer'),
        (pl.col('grid_peak_kw_reduction').sum() / 1000).alias('grid_peak_cut_mw'),
        (pl.col('on_peak_kwh_decrease').sum() / 1000).alias('on_peak_mwh_removed'),
        (pl.col('off_peak_kwh_increase').sum() / 1000).alias('off_peak_mwh_added'),
    ).sort('avg_savings', descending=True))

    mix = (c.group_by('segment', 'charge_window').agg(pl.len().alias('customers'))
           .with_columns((pl.col('customers') / pl.col('customers').sum().over('segment')).alias('share'))
           .sort('segment', 'charge_window'))
    per_window = (c.group_by('charge_window').agg(
        pl.len().alias('customers'), pl.col('bill_savings').mean().alias('avg_savings'),
        pl.col('bill_savings_pct').median().alias('median_savings_pct'),
        (pl.col('bill_savings').sum() / pl.col('bill_before').sum()).alias('savings_share_of_bill'))
        .sort('customers', descending=True))

    return {'run_id': run_id, 'scenario': s, 'summary': summary,
            'peak_day_before': day(np.argmax(sb)), 'peak_day_after': day(np.argmax(sa)),
            'monthly': monthly,
            'histogram': {'savings': hist(c['bill_savings'].to_numpy()),
                          'savings_pct': hist(c['bill_savings_pct'].to_numpy() * 100)},
            'segments': segments.to_dicts(), 'window_mix': mix.to_dicts(), 'per_window': per_window.to_dicts(),
            'plan_info': info}


# ------------------------------------------------------------------------------------------- per-run views
def plan(run_id: str) -> dict:
    """Charging-plan comparison for a run's battery/tariff/dispatch/filters: every single window, the
    savings vs system-peak frontier, and each segment's best window."""
    r = _get_run(run_id)
    s, rows = r['scenario'], r['rows']
    key = _key([s['battery'], s['tariff'], s['dispatch'], s['filters']])
    with _lock:
        cmp = _compare(s, rows)
        if key not in _frontier_cache:
            _remember(_frontier_cache, key, bt.peak_savings_frontier(cmp, points=9), 3)
    fr = _frontier_cache[key]
    segs = load()['segment'][rows]
    best = {str(g): cmp['names'][int(cmp['savings'][segs == g].sum(0).argmax())] for g in np.unique(segs)}
    return {'run_id': run_id, 'system_peak_before_kw': float(cmp['system_before'].max()),
            'bill_before_total': float(cmp['bill_before_total']),
            'single_windows': cmp['summary'].to_dicts(), 'frontier': fr.to_dicts(), 'best_by_segment': best,
            'current': {'system_peak_kw': r['payload']['summary']['system_peak_after_kw'],
                        'total_savings': r['payload']['summary']['total_savings'],
                        'savings_share_of_bill': r['payload']['summary']['savings_share_of_bill']}}


RANK_COLUMNS = ['CUSTOMER_NUMBER', 'segment', 'charge_window', 'bill_savings', 'bill_savings_pct', 'energy_savings',
                'demand_savings', 'bill_before', 'bill_after', 'avg_monthly_peak_kw_before',
                'avg_monthly_peak_kw_reduction', 'grid_peak_kw_reduction', 'on_peak_kwh_decrease',
                'off_peak_kwh_increase', 'equivalent_full_cycles']
RANK_BY = ['bill_savings', 'bill_savings_pct', 'grid_peak_kw_reduction', 'avg_monthly_peak_kw_reduction']


def ranking(run_id: str, rank_by: str = 'bill_savings', segment: str | None = None,
            customer: int | None = None) -> pl.DataFrame:
    if rank_by not in RANK_BY:
        raise ValueError(f'rank_by must be one of {RANK_BY}')
    c = _get_run(run_id)['customers']
    out = (c.sort(rank_by, descending=True, nulls_last=True).with_row_index('rank', offset=1)
           .select('rank', *RANK_COLUMNS, *[f for f in FLAGS if f in c.columns]))
    if segment:
        out = out.filter(pl.col('segment') == segment)
    if customer is not None:
        out = out.filter(pl.col('CUSTOMER_NUMBER') == customer)
    return out


def customer(run_id: str, customer_number: int, week: int = 45) -> dict:
    """One customer's results and an hourly before/after week (week 1 starts on the first day of data)."""
    r = _get_run(run_id)
    d = load()
    if customer_number not in d['row_of']:
        raise LookupError(f'Unknown customer {customer_number}.')
    row = r['customers'].filter(pl.col('CUSTOMER_NUMBER') == customer_number)
    if not row.height:
        raise LookupError('Customer is not in this run (check the filters).')
    stats = row.row(0, named=True)
    i = d['row_of'][customer_number]
    kw = _sim_kwargs(r['scenario'])
    one = bt.simulate(d['L'][[i]], d['dates'], charge_window=bt.CHARGE_WINDOWS[stats['charge_window']],
                      keep_net_for=np.array([0]), **kw)
    on = kw['tariff'].hourly(d['dates'])['on_peak']
    T = d['L'].shape[1]
    week = int(np.clip(week, 1, T // 168))
    a, b = (week - 1) * 168, min(week * 168, T)
    start = np.datetime64(str(d['dates'][0])) + np.timedelta64(a, 'h')
    times = [str(start + np.timedelta64(h, 'h')) for h in range(b - a)]
    # monthly bill view: kWh by month before/after
    month = kw['tariff'].hourly(d['dates'])['month']
    before, after = d['L'][i].astype(float), one['net'][0].astype(float)
    monthly = [{'month': MONTH_NAMES[m - 1], 'kwh_before': float(before[month == m].sum()),
                'kwh_after': float(after[month == m].sum()),
                'peak_kw_before': float(before[month == m].max()), 'peak_kw_after': float(after[month == m].max())}
               for m in MONTH_ORDER]
    return {'customer': customer_number, 'stats': stats, 'week': week, 'weeks': T // 168, 'times': times,
            'before': before[a:b].round(3).tolist(), 'after': after[a:b].round(3).tolist(),
            'on_peak': on[a:b].tolist(), 'monthly': monthly}
