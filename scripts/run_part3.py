"""Part 3: demand flexibility and behind-the-meter battery opportunity, built only on Part 1 and 2 outputs.

Inputs (written by scripts/export_part3_inputs.py into onedrive-export/Data/part3/):
    hourly_kwh.npy   customers x hours, kWh per hour (= average kW), cleaned and gap-filled by notebooks 10-20
    hours.csv        timestamp (hour start), date, hour_ending, month, weekend, temp_f
    customers.csv    Part 1 appliance inventory and technology scores, Part 2 segments and program flags

Outputs (same folder, git-ignored because they hold customer data):
    app_data.json    everything the planner page reads
    index.html       the planner page (web/part3/index.html) with app_data.json embedded; open it directly
    candidates.csv   one row per customer with the main Part 3 results

Run from the project folder:
    python scripts/run_part3.py
    python scripts/run_part3.py --max-customers 3000     # smaller page: top 3,000 customers listed
                                                           # (summary figures still cover everyone)

Method (also shown on the page's Methodology tab):
 1. System peak: sum all meters by hour. Per season, the 20 highest hours are the coincident-peak (CP) hours;
    the 4-hour block with the most load on the 20 highest days is the discharge window.
 2. Each home at the peak: CP load = mean load at the CP hours; excess = CP load - median home's CP load.
    Excess above 3 kW is past the Challenge Statement's 3 kVA incremental-load guideline.
 3. Flexibility first: one weather regression per home (hour-of-day, weekend, CDH65, HDH60) gives cooling and
    heating kW per degree F. Shed in the discharge window, weighted by Part 1 probabilities:
    EV shift_share x p_EV x EV excess, cooling share x b_cool x CDH x p_AC, heating share x b_heat x HDH x p_heat,
    water heater kW x p_WH, pool kW x p_pool. Only appliances Part 1 rates likely (p >= 0.5) are used.
    Shed energy is moved to the off-peak charge hours (energy-neutral).
 4. Battery: 1 and 2 units simulated on the load left after flex, every day. Discharge levels the top of the
    window (find T: sum_h min(P, max(0, L_h - T)) = stored x sqrt(eff)). Charging uses solar surplus first,
    then the grid at <= 3 kW in the 6 lowest system-load hours before the window.
 4b. Fleet (config fleet.optimize, on by default): the batteries chosen in 5 are dispatched as one fleet instead;
    a daily linear program minimizes the system peak, and the fleet size is the knee of the peak-versus-fleet-size
    curve (see fleet_day_lp and choose_fleet). Bills, relief and economics use each home's fleet-dispatched load.
 5. Size: no battery if flex covers 80% of the excess, the excess is under 0.75 kW, or the home already has
    storage (Part 1 solar + battery). 1 unit if it cuts CP load by >= 0.5 kW; 2 units if the second adds
    >= 0.5 kW or is needed to get back inside 3 kVA.
 6. Value: bills on a TOU rate table (144 buckets: 3 seasons x weekday/weekend x 24 hours); 15-year cash flow
    with escalation, degradation and performance incentives; IRR, NPV, payback, break-even rebate = -NPV.
 7. Channel: grid need (0.6 x pct peak relief + 0.4 x pct excess) vs readiness (0.7 x (1 - pct rebate needed)
    + 0.3 x adoption signal), split at 0.5.
 8. Rank: score = 100 x sum_k w_k x percentile_k over capacity relief, peak persistence, distance over 3 kVA,
    electrification signal and customer economics. Waves at 80 and 60.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get('AVANGRID_DATA_DIR', ROOT / 'onedrive-export' / 'Data'))
DEFAULT_IN = DATA_DIR / 'part3'
DEFAULT_CONFIG = ROOT / 'config' / 'part3_assumptions.json'
TEMPLATE = ROOT / 'web' / 'part3' / 'index.html'

SEASONS = ('summer', 'winter', 'shoulder')     # bucket order used by the page
LEVERS = ('ev', 'cooling', 'heating', 'water_heater', 'pool')
LEVER_TEXT = {'ev': 'managed EV charging', 'cooling': 'thermostat pre-cooling', 'heating': 'heat pump pre-heat',
              'water_heater': 'water heater control', 'pool': 'pool pump scheduling'}
CHANNELS = {
    'Targeted offer': 'Targeted marketing at the standard rebate; installer partners follow up',
    'Utility-owned or lease': 'Utility owns and dispatches the battery; customer receives backup power '
                              '(Green Mountain Power model)',
    'Bring your own battery': 'Performance payments only; marketed through installers '
                              '(CT Energy Storage Solutions model)',
    'Flexibility programs first': 'Managed charging or thermostat program now; reassess storage as costs decline',
}
CHUNK = 1000

# Real residential rates (scripts/fetch_tariffs.py). `--utility` replaces config economics.rates and supply_rate with
# that utility's time-varying rate; without it the config's rates are used unchanged.
TARIFFS = ROOT / 'data' / 'tariffs' / 'residential_tariffs.json'
DEFAULT_PLAN = {'nyseg': 'day_night', 'rge': 'tou', 'cmp': 'tou', 'ui': 'tou'}   # NYSEG's SC 12 needs 35,000 kWh/yr
PAGE_RATE_LABEL = '"CMP TOU delivery rate, effective 1/1/2026"'   # source label in the page's Assumptions table
PAGE_FLEET_TEXT = [                                               # page text replaced when the fleet step runs
    ('"Battery discharge window"', '"Flexibility window"'),
    ('"Charging uses solar surplus first, then the grid at \\u2264 ",n.grid_charge_cap_kw," kW in the ",'
     'n.charge_hours_n," lowest system-load hours before the window."',
     '"This per-home simulation picks candidates and sizes. The batteries are then dispatched as one fleet: each '
     'day a linear program sets the fleet\\u2019s hourly charging (from the grid, \\u2264 ",n.grid_charge_cap_kw,'
     '" kW per home) and discharging (each home up to its own load) to minimize that day\\u2019s system peak. The '
     'fleet size is the knee of the peak-versus-fleet-size curve: the smallest fleet within {pct} of the best '
     'achievable peak reduction, here {homes} of {cands} candidate homes. Candidates outside the fleet get '
     'flexibility only."'),
]


def utility_rates(cfg: dict, utility: str, plan: str | None = None, region: str | None = None,
                  tariffs: Path = TARIFFS) -> dict:
    """A copy of cfg with economics.rates and supply_rate set from a utility's rate in data/tariffs.

    Uses the plan's version in effect at the end of the data year. The page models a rate as, per season, an on-peak
    and an off-peak price plus one flat supply price, so: supply_rate = the rate's off-peak supply price, and the
    on- and off-peak prices are the full price (delivery + supply) minus that. Totals match the utility's rate in
    every hour; exports are credited at the off-peak supply price. Rates with more than two prices in a season, or
    with weekend peak hours that differ from weekday ones, can't be represented and raise an error."""
    t = json.loads(tariffs.read_text())
    u = next((x for x in t['utilities'] if x['id'] == utility), None)
    if u is None:
        raise ValueError(f"unknown utility {utility!r}; choose from {[x['id'] for x in t['utilities']]}")
    plan = plan or DEFAULT_PLAN[utility]
    region = region or u.get('default_region')
    p = next((x for x in u['plans'] if x['id'] == plan and x['region'] in (None, region)), None)
    if p is None:
        raise ValueError(f'{u["name"]} has no plan {plan!r}' + (f' in region {region!r}' if region else ''))
    end = t['meta']['data_year']['end']
    v = [x for x in p['versions'] if (x['start'] or '') <= end][-1]
    periods = v['periods']
    price = [x['rate_per_kwh'] for x in periods]
    split = all('supply_per_kwh' in x for x in periods)
    supply = min(x['supply_per_kwh'] for x in periods) if split else 0.0
    rates = {}
    for season in SEASONS:
        months = cfg['seasons'][season]
        rows = {(tuple(v['weekday_schedule'][m - 1]), tuple(v['weekend_schedule'][m - 1])) for m in months}
        if len(rows) > 1:
            raise ValueError(f'{u["name"]} {p["name"]}: hours differ between months of the {season} season')
        wd, we = next(iter(rows))
        wd_p, we_p = [price[i] for i in wd], [price[i] for i in we]
        levels = sorted(set(wd_p) | set(we_p))
        if len(levels) > 2:
            raise ValueError(f'{u["name"]} {p["name"]}: {len(levels)} prices in the {season} season; the page '
                             'supports on- and off-peak only')
        hi, lo = levels[-1], levels[0]
        on_wd = [h + 1 for h in range(24) if wd_p[h] == hi and hi != lo]
        on_we = [h + 1 for h in range(24) if we_p[h] == hi and hi != lo]
        if on_we and on_we != on_wd:
            raise ValueError(f'{u["name"]} {p["name"]}: weekend peak hours differ from weekday ones')
        rates[season] = {'on_peak_he': on_wd, 'weekdays_only': not on_we,
                         'on_peak': round(hi - supply, 6), 'off_peak': round(lo - supply, 6)}
    out = json.loads(json.dumps(cfg))
    eco = out['economics']
    eco['rates'], eco['supply_rate'] = rates, round(supply, 6)
    eco['tariff'] = {
        'utility': u['id'], 'utility_name': u['name'], 'utility_full_name': u['full_name'], 'state': u['state'],
        'plan': p['id'], 'plan_name': p['name'], 'region': p['region'], 'effective': v['start'],
        'fixed_charge_per_month': v['fixed_charge_per_month'], 'source': v['tariff_source'],
        'openei_label': v['openei']['label'], 'supply_split': split,
        'label': f"{u['name']} {p['name']}" + (f" ({p['region']})" if p['region'] else '')
                 + f", effective {v['start']} (OpenEI; delivery and supply combined)",
    }
    return out


# ---------------------------------------------------------------------------------------------- inputs
def col(df: pd.DataFrame, name: str, default=0.0) -> np.ndarray:
    """A customers.csv column as an array, or the default when the export doesn't have it."""
    if name not in df.columns:
        return np.full(len(df), default)
    s = df[name]
    if isinstance(default, bool):
        return s.astype(str).str.lower().isin(['true', '1']).to_numpy()
    if isinstance(default, str):
        return s.fillna(default).astype(str).to_numpy()
    return pd.to_numeric(s, errors='coerce').fillna(default).to_numpy(float)


def part1_probabilities(c: pd.DataFrame) -> dict:
    """Part 1 outputs -> the probabilities and flags Part 3 uses. Nothing here is re-estimated."""
    on_ev, on_hp, on_pv = col(c, 'on_ev_list', False), col(c, 'on_hp_list', False), col(c, 'on_pv_list', False)
    heating_type = col(c, 'heating_type', 'none detected')
    heat_any = col(c, 'elec_heat_confidence')
    heat_primary = col(c, 'elec_heat_primary_confidence')
    uncertain = heating_type == 'electric (type uncertain)'
    p_hp = np.where(heating_type == 'heat pump likely', np.maximum(heat_any, 0.5), np.where(uncertain, 0.5 * heat_any, 0.0))
    p_hp = np.where(on_hp, 1.0, p_hp)
    p_res = np.where(heating_type == 'resistance likely', np.maximum(heat_primary, 0.5),
                     np.where(uncertain, 0.5 * heat_primary, 0.0))
    p_res = np.where(on_hp, 0.0, p_res)
    p_ev = np.where(on_ev, 1.0, np.maximum(col(c, 'ev_level2_confidence'), col(c, 'ev_level1_confidence')))
    pv_score = col(c, 'pv_score')
    solar_evidence = on_pv | (pv_score >= 0.5) | col(c, 'solar', False)
    p_solar = np.where(solar_evidence, 1.0, pv_score)
    nights = np.maximum(col(c, 'ev_l2_night_share'), col(c, 'ev_l1_night_share'))
    return {
        'p_ev': np.clip(p_ev, 0, 1), 'p_heat_pump': np.clip(p_hp, 0, 1), 'p_resist_heat': np.clip(p_res, 0, 1),
        'p_ac': np.clip(col(c, 'ac_confidence'), 0, 1), 'p_pool': np.clip(col(c, 'pool_confidence'), 0, 1),
        'p_wh': np.clip(col(c, 'water_heater_confidence'), 0, 1), 'p_solar': np.clip(p_solar, 0, 1),
        'on_ev': on_ev, 'on_pv': on_pv, 'storage': col(c, 'solar_battery_flag', False) & solar_evidence,
        'ev_new': col(c, 'chg_ev_flag', False) & ~on_ev, 'ev_onset': col(c, 'chg_ev_change_date', ''),
        'solar_new': col(c, 'chg_pv_flag', False) & solar_evidence, 'solar_onset': col(c, 'chg_pv_change_date', ''),
        'ev_events_yr': np.round(nights * 365).astype(int),
        'segment': col(c, 'segment', 'Unassigned'),
    }


# ---------------------------------------------------------------------------------------------- helpers
def pct(x: np.ndarray) -> np.ndarray:
    """Percentile rank in (0, 1], ties averaged (so many zeros share one middle value)."""
    return pd.Series(x).rank(method='average', pct=True).to_numpy() if len(x) else x


def round_or_none(x, nd):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def cash_flows(units, savings, cp_summer, cp_winter, fin, eco):
    """Same cash flow as the page (year 0 cost, 15 years of savings + incentives, degraded and escalated)."""
    cost = 0 if units <= 0 else units * fin['battery_capex_per_unit'] + fin['install_base'] + fin['install_per_extra_unit'] * (units - 1)
    rows = [-cost + fin['rebate_per_kwh'] * units * 10.08 + fin['tax_credit_pct'] * cost]
    for y in range(1, fin['life_years'] + 1):
        h = (1 - fin['degradation']) ** (y - 1)
        bill = savings * (1 + fin['rate_escalator']) ** (y - 1) * h
        if y <= 5:
            inc = eco['incentive_summer_per_kw'] * cp_summer + eco['incentive_winter_per_kw'] * cp_winter
        elif y <= 10:
            inc = eco['incentive_summer_per_kw_late'] * cp_summer + eco['incentive_winter_per_kw_late'] * cp_winter
        else:
            inc = 0.0
        rows.append(bill + inc * h - fin['om_per_yr'])
    return np.array(rows), cost


def npv(rate, cf):
    return float(sum(v / (1 + rate) ** i for i, v in enumerate(cf)))


def irr(cf):
    lo, hi = -0.99, 1.0
    flo, fhi = npv(lo, cf), npv(hi, cf)
    if np.sign(flo) == np.sign(fhi):
        return None
    for _ in range(100):
        mid = (lo + hi) / 2
        fm = npv(mid, cf)
        if np.sign(fm) == np.sign(flo):
            lo, flo = mid, fm
        else:
            hi = mid
    return (lo + hi) / 2


def payback(cf):
    acc = cf[0]
    for i in range(1, len(cf)):
        nxt = acc + cf[i]
        if nxt >= 0:
            return i - 1 + -acc / max(cf[i], 1e-9)
        acc = nxt
    return None


# ---------------------------------------------------------------------------------------------- model
class Calendar:
    """Hour-level calendar arrays and the per-day discharge/charge hours."""

    def __init__(self, hours: pd.DataFrame, cfg: dict):
        self.H = len(hours)
        assert self.H % 24 == 0, 'hours.csv must hold whole days'
        self.D = self.H // 24
        self.month = hours['month'].to_numpy(int)
        self.he = hours['hour_ending'].to_numpy(int)
        self.weekend = hours['weekend'].astype(str).str.lower().isin(['true', '1']).to_numpy()
        self.temp = pd.to_numeric(hours['temp_f'], errors='coerce').interpolate(limit_direction='both').to_numpy()
        self.stamp = hours['timestamp'].astype(str).str.replace('T', ' ').str.slice(0, 19).to_numpy()
        self.date = hours['date'].astype(str).to_numpy()
        se = cfg['seasons']
        self.season = np.where(np.isin(self.month, se['summer']), 'summer',
                               np.where(np.isin(self.month, se['winter']), 'winter', 'shoulder'))
        self.day_season = self.season[::24]
        wm = cfg['weather_model']
        self.cdh = np.maximum(0, self.temp - wm['cooling_base_f'])
        self.hdh = np.maximum(0, wm['heating_base_f'] - self.temp)
        X = np.zeros((self.H, 27))
        X[np.arange(self.H), self.he - 1] = 1
        X[:, 24], X[:, 25], X[:, 26] = self.weekend, self.cdh, self.hdh
        self.X = X
        self.Xpinv = np.linalg.pinv(X)
        sidx = {s: i for i, s in enumerate(SEASONS)}
        self.bucket = (np.vectorize(sidx.get)(self.season) * 2 + self.weekend) * 24 + (self.he - 1)
        self.onehot = np.zeros((self.H, 144), np.float32)
        self.onehot[np.arange(self.H), self.bucket] = 1
        self.pool_on = np.isin(self.month, cfg['flex'].get('pool_months', [5, 6, 7, 8, 9]))

    def price(self, eco: dict) -> np.ndarray:
        p = np.zeros(self.H)
        for s in SEASONS:
            r = eco['rates'][s]
            m = self.season == s
            on = np.isin(self.he, r['on_peak_he']) & (~self.weekend if r['weekdays_only'] else True)
            p[m] = np.where(on[m], r['on_peak'], r['off_peak']) + eco['supply_rate']
        return p

    def set_windows(self, windows: dict, binding: str):
        """Per-day 0-based hour indices of the discharge window (4) and grid-charge hours (6)."""
        def per_day(key):
            return np.array([np.array(windows[s if s != 'shoulder' else binding][key]) - 1 for s in self.day_season])
        self.win = per_day('window_he')
        self.chg = per_day('charge_he')
        self.win_mask = np.zeros((self.D, 24), bool)
        self.chg_mask = np.zeros((self.D, 24), bool)
        np.put_along_axis(self.win_mask, self.win, True, 1)
        np.put_along_axis(self.chg_mask, self.chg, True, 1)
        start = self.win[:, 0]
        self.before_win = np.arange(24)[None, :] < start[:, None]
        naive = np.zeros((self.D, 24), bool)
        naive[:, :6] = True
        self.naive_mask = naive


def find_windows(S: np.ndarray, cal: Calendar, cfg: dict) -> dict:
    sp = cfg['system_peak']
    n_top, n_days, wl = sp['top_hours_per_season'], sp['top_days_for_window'], sp['window_length_h']
    n_chg = cfg['battery']['charge_hours_n']
    Sd = S.reshape(cal.D, 24)
    out = {}
    for s in ('summer', 'winter'):
        hidx = np.flatnonzero(cal.season == s)
        cp = hidx[np.argsort(S[hidx])[::-1][:n_top]]
        didx = np.flatnonzero(cal.day_season == s)
        top_days = didx[np.argsort(Sd[didx].max(1))[::-1][:n_days]]
        prof = Sd[top_days].mean(0)
        blocks = [prof[i:i + wl].sum() for i in range(24 - wl + 1)]
        st = int(np.argmax(blocks))
        window = list(range(st, st + wl))
        before = [h for h in range(st) if h not in window]
        after = [h for h in range(st + wl, 24)]
        cand = sorted(before, key=lambda h: prof[h])[:n_chg]
        if len(cand) < n_chg:
            cand += sorted(after, key=lambda h: prof[h])[:n_chg - len(cand)]
        out[s] = {'window_he': [h + 1 for h in window], 'charge_he': sorted(h + 1 for h in cand),
                  'peak_mw': round(float(S[hidx].max()) / 1000, 3),
                  'cp_hours': [str(cal.stamp[i]) for i in sorted(cp)],
                  '_cp_idx': np.sort(cp), '_top_days': np.sort(top_days)}
    return out


def weather_fit(L: np.ndarray, cal: Calendar):
    B = L @ cal.Xpinv.T                      # n x 27
    fitted = B @ cal.X.T
    resid = L - fitted
    ss_res = (resid ** 2).sum(1)
    ss_tot = ((L - L.mean(1, keepdims=True)) ** 2).sum(1)
    r2 = np.where(ss_tot > 0, 1 - ss_res / np.maximum(ss_tot, 1e-9), 0)
    return B[:, 25], B[:, 26], np.clip(r2, 0, 1), resid


def flex_shed(L, resid, b_cool, b_heat, P, rows, cal, cfg, full=False):
    """Per-lever shed (n x H) inside the discharge window. full=True gives the technical potential (shares = 1)."""
    fx, sc = cfg['flex'], cfg['screening']
    minp = sc.get('lever_min_p', 0.5)
    win = cal.win_mask.reshape(-1)
    p = {k: P[k][rows] for k in ('p_ev', 'p_ac', 'p_heat_pump', 'p_resist_heat', 'p_wh', 'p_pool')}
    p_heat = np.maximum(p['p_heat_pump'], p['p_resist_heat'])
    gate = lambda q: np.where(q >= minp, q, 0.0)[:, None]  # noqa: E731
    share = (lambda s: 1.0) if full else (lambda s: s)  # noqa: E731
    ev = share(fx['ev_shift_share']) * gate(p['p_ev']) * np.where(resid >= fx['ev_spike_kw'], resid, 0)
    cool = share(fx['cooling_shed_share']) * np.maximum(b_cool, 0)[:, None] * cal.cdh[None, :] * gate(p['p_ac'])
    heat = share(fx['heating_shed_share']) * np.maximum(b_heat, 0)[:, None] * cal.hdh[None, :] * gate(p_heat)
    wh = fx['water_heater_kw'] * gate(p['p_wh']) * np.ones((1, cal.H))
    pool = fx['pool_kw'] * gate(p['p_pool']) * cal.pool_on[None, :]
    out = {}
    room = np.maximum(L, 0) * win[None, :]
    for k, v in zip(LEVERS, (ev, cool, heat, wh, pool)):
        v = np.minimum(np.maximum(v, 0) * win[None, :], room)
        room = room - v
        out[k] = v.astype(np.float32)
    return out


def shift_to_charge_hours(L, shed, cal):
    """Remove shed in the window and put the same energy back evenly in that day's charge hours."""
    n = L.shape[0]
    Ld = L.reshape(n, cal.D, 24) - shed.reshape(n, cal.D, 24)
    daily = shed.reshape(n, cal.D, 24).sum(2)
    Ld += cal.chg_mask[None] * (daily / cal.chg_mask.sum(1))[..., None]
    return Ld.reshape(n, -1)


def battery(Lf, units, cal, cfg):
    """Daily dispatch for `units` batteries on the post-flex load. Returns net load and annual stats."""
    b = cfg['battery']
    n = Lf.shape[0]
    eff_leg = np.sqrt(b['round_trip_eff'])
    usable = units * b['unit_kwh'] * (1 - b['reserve_soc'])
    P = units * b['unit_kw']
    Ld = Lf.reshape(n, cal.D, 24)
    W = np.take_along_axis(Ld, np.broadcast_to(cal.win[None], (n, cal.D, cal.win.shape[1])), 2)
    surplus = np.maximum(-Ld, 0) * cal.before_win[None]
    solar_avail = surplus.sum(2)
    grid_max = b['grid_charge_cap_kw'] * cal.chg_mask.sum(1)[None, :]
    stored_max = np.minimum(usable, (solar_avail + grid_max) * eff_leg)
    target = stored_max * eff_leg
    shave = lambda T: np.minimum(P, np.maximum(0, W - T[..., None])).sum(2)  # noqa: E731
    lo, hi = np.zeros((n, cal.D)), np.maximum(W.max(2), 0)
    need = shave(lo) > target
    for _ in range(40):
        mid = (lo + hi) / 2
        over = shave(mid) > target
        lo = np.where(over, mid, lo)
        hi = np.where(over, hi, mid)
    T = np.where(need, hi, 0)
    dis = np.minimum(P, np.maximum(0, W - T[..., None]))
    delivered = dis.sum(2)
    charged = delivered / b['round_trip_eff']
    solar_used = np.minimum(solar_avail, charged)
    grid = charged - solar_used
    net = Ld.copy()
    np.put_along_axis(net, np.broadcast_to(cal.win[None], W.shape), W - dis, 2)
    frac = np.where(solar_avail > 0, solar_used / np.maximum(solar_avail, 1e-9), 0)
    net += surplus * frac[..., None]
    grid_h = grid / cal.chg_mask.sum(1)[None, :]
    net += cal.chg_mask[None] * grid_h[..., None]
    naive = net - cal.chg_mask[None] * grid_h[..., None] + cal.naive_mask[None] * (grid / 6)[..., None]
    return {'net': net.reshape(n, -1), 'naive': naive.reshape(n, -1), 'discharged': delivered.sum(1),
            'solar_charged': solar_used.sum(1)}


# ---------------------------------------------------------------------------------------------- cluster table
# Notebook 70's eight behavioural clusters (k-means on continuous features only). Each description says what the
# cluster was named for: notebook 70 names clusters in this order, each from the remaining clusters' medians.
CLUSTER_TEXT = {
    'Solar homes': 'Most summer midday hours with near-zero grid use: solar panels cover daytime use.',
    'Seasonal / part-year vacant': 'Months of near-zero use and the largest winter-summer swing: seasonal or '
                                   'part-year homes.',
    'Electric-heated, winter-peaking': 'Strongest heating response for their size: electric heat drives their '
                                       'winter peaks.',
    'Summer peakers (heavy AC)': 'Strongest cooling response for their size: air conditioning drives summer '
                                 'afternoon and evening peaks.',
    'Evening peakers': 'Highest share of use from 5 to 9 pm.',
    'Small, flat, low-AC': 'Lowest average use, with a flat daily profile and little cooling.',
    'Large, steady homes': 'High average use with a high load factor: use stays high around the clock.',
    'Daytime-active, AC': 'The remaining cluster: more daytime use than typical, with air conditioning.',
}
CLUSTER_FLAGS = [   # (label, test on customers.csv) - Part 1 appliance inventory flags (notebook 60)
    ('AC (any)', lambda c: c['ac_flag']),
    ('Central AC', lambda c: c['ac_type'] == 'central likely'),
    ('Electric heat (any)', lambda c: c['elec_heat_flag']),
    ('Electric heat as main source', lambda c: c['elec_heat_primary_flag']),
    ('Heat pump likely', lambda c: c['heating_type'] == 'heat pump likely'),
    ('Water heater', lambda c: c['water_heater_flag']),
    ('Pool', lambda c: c['pool_flag']),
    ('Hot tub', lambda c: c['hot_tub_flag']),
    ('EV charging (Level 1 or 2)', lambda c: c['ev_charging_type'] != 'none detected'),
    ('Solar (PV list or model)', lambda c: c['solar']),
    ('Solar + battery', lambda c: c['solar_battery_flag']),
]


def as_bool(x) -> np.ndarray:
    """A True/False column read from CSV (bool, or the strings 'true'/'false') as a boolean array."""
    x = pd.Series(x)
    return x.to_numpy(bool) if x.dtype == bool else x.astype(str).str.lower().isin(['true', '1']).to_numpy()


def cluster_summary(cust: pd.DataFrame, units: np.ndarray, batt_sav: np.ndarray) -> dict:
    """One row per notebook 70 cluster: description, size, battery homes and their average year-1 savings, and the
    share of homes flagged for each appliance."""
    seg = cust['segment'].astype(str).to_numpy()
    flags = {lab: as_bool(f(cust)) for lab, f in CLUSTER_FLAGS}
    rows = []
    for name in sorted(set(seg), key=lambda s: -(seg == s).sum()):
        m = seg == name
        b = m & (units > 0)
        rows.append({'cluster': name, 'description': CLUSTER_TEXT.get(name, ''), 'homes': int(m.sum()),
                     'share': round(float(m.mean()), 4), 'battery_homes': int(b.sum()),
                     'avg_battery_savings': round(float(batt_sav[b].mean()), 0) if b.any() else None,
                     'flags': {lab: round(float(v[m].mean()), 4) for lab, v in flags.items()}})
    return {'flags': [lab for lab, _ in CLUSTER_FLAGS], 'rows': rows,
            'note': "Notebook 70's eight behavioural clusters (k-means on continuous usage features). Battery "
                    'savings are year-1 bill savings from the battery, averaged over the cluster’s battery homes.'}


# ---------------------------------------------------------------------------------------------- fleet optimization
# Per-home dispatch (battery() above) charges every battery in the same hours, which can create a new system peak.
# Instead the fleet is dispatched as one: each day, a small linear program chooses the fleet's hourly charge c_h and
# discharge d_h to minimize that day's system peak T:
#     min T  s.t.  S_h - d_h + c_h <= T;  0 <= d_h <= A_h (fleet's dischargeable kW: each home up to its own load
#     and power rating);  0 <= c_h <= C (3 kW per home, scaled so no home exceeds it);  stored energy
#     e_{h+1} = e_h + sqrt(eff) c_h - d_h / sqrt(eff), 0 <= e_h <= E (usable kWh), cyclic over the day.
# The fleet size comes from the knee of the peak-versus-size curve: candidate homes are ranked by how much load their
# batteries can cover at the system peak, and the smallest fleet within `knee_share` of the best achievable peak
# reduction is chosen. The fleet's hourly charge and discharge are then split across its homes (each home's
# discharge follows the fleet's hourly shape, up to its own load; its charge follows the fleet's charge shape), and
# every result (system peak, bills, relief) is computed from the homes' own net loads.

def fleet_day_lp(s, A, E, C, eff):
    """One day's min-peak dispatch. s: system load (24), A: dischargeable kW (24), E: usable kWh, C: charge cap kW."""
    from scipy.optimize import linprog
    leg = np.sqrt(eff)
    nv = 73                                               # c (24), d (24), e (24), T
    cost = np.zeros(nv)
    cost[-1] = 1.0
    cost[24:48] = 1e-6                                    # tie-break: no needless cycling
    A_ub = np.zeros((24, nv))
    A_ub[np.arange(24), np.arange(24)] = 1
    A_ub[np.arange(24), 24 + np.arange(24)] = -1
    A_ub[:, -1] = -1
    A_eq = np.zeros((24, nv))
    for h in range(24):
        A_eq[h, 48 + (h + 1) % 24], A_eq[h, 48 + h], A_eq[h, h], A_eq[h, 24 + h] = 1, -1, -leg, 1 / leg
    bounds = [(0, C)] * 24 + [(0, float(a)) for a in A] + [(0, E)] * 24 + [(None, None)]
    r = linprog(cost, A_ub=A_ub, b_ub=-np.asarray(s, float), A_eq=A_eq, b_eq=np.zeros(24), bounds=bounds,
                method='highs')
    if r.status != 0:
        raise RuntimeError(f'fleet dispatch LP failed: {r.message}')
    return r.x[:24], r.x[24:48], r.x[-1]


def fleet_inputs(Lf_c, units_c, bcfg, idx=None):
    """Fleet totals for the homes in idx: dischargeable kW per hour (D x 24), usable kWh, charge cap kW."""
    idx = np.arange(len(units_c)) if idx is None else idx
    D = Lf_c.shape[1] // 24
    A = np.zeros(Lf_c.shape[1])
    for a in range(0, len(idx), 2000):
        j = idx[a:a + 2000]
        A += np.minimum(np.maximum(Lf_c[j], 0), (units_c[j] * bcfg['unit_kw'])[:, None]).sum(0, dtype=np.float64)
    usable = units_c[idx] * bcfg['unit_kwh'] * (1 - bcfg['reserve_soc'])
    C = bcfg['grid_charge_cap_kw'] * usable.sum() / max(usable.max(), 1e-9)
    return A.reshape(D, 24), float(usable.sum()), float(C)


def fleet_peak(S0d, A, E, C, eff, days):
    """Annual system peak (kW) with the fleet dispatched on `days`; other days keep their load."""
    rest = np.setdiff1d(np.arange(len(S0d)), days)
    peaks = [fleet_day_lp(S0d[d], A[d], E, C, eff)[2] for d in days]
    return max(max(peaks), S0d[rest].max() if len(rest) else 0.0)


def choose_fleet(S_flex, Lf_c, units_c, cal, cfg):
    """Rank candidates, trace the peak-versus-fleet-size curve and pick the knee. Returns (order, K, curve)."""
    b, fcfg = cfg['battery'], cfg['fleet']
    eff = b['round_trip_eff']
    S0d = S_flex.reshape(cal.D, 24)
    days = np.sort(np.argsort(S0d.max(1))[::-1][:fcfg.get('curve_days', 60)])
    peak_hour = S0d.argmax(1)                             # each day's peak hour
    cover = np.minimum(np.maximum(Lf_c.reshape(len(units_c), cal.D, 24)[:, days, peak_hour[days]], 0),
                       (units_c * b['unit_kw'])[:, None]).mean(1)
    order = np.lexsort((np.arange(len(units_c)), -cover))  # most kW at the system peak first
    n = len(order)
    cache = {}

    def peak_of(k):
        if k not in cache:
            A, E, C = fleet_inputs(Lf_c, units_c, b, order[:k]) if k else (np.zeros((cal.D, 24)), 0.0, 0.0)
            cache[k] = S0d.max() if k == 0 else fleet_peak(S0d, A, E, C, eff, days)
        return cache[k]

    grid = sorted({0, n} | {int(round(x)) for x in np.geomspace(100, n, 14) if x < n})
    for k in grid:
        peak_of(k)
    base, best = cache[0], min(cache.values())
    share = lambda k: (base - cache[k]) / max(base - best, 1e-9)  # noqa: E731
    hi = next(k for k in grid if share(k) >= fcfg.get('knee_share', 0.95))
    lo = max(k for k in grid if k < hi) if hi > 0 else 0
    while hi - lo > max(50, n // 500):                     # bisect for the smallest fleet past the knee
        mid = (lo + hi) // 2
        peak_of(mid)
        lo, hi = (mid, hi) if share(mid) < fcfg.get('knee_share', 0.95) else (lo, mid)
    curve = []
    for k in sorted(cache):
        sel = order[:k]
        curve.append({'homes': int(k), 'units': int(units_c[sel].sum()),
                      'mwh': round(float((units_c[sel] * b['unit_kwh']).sum()) / 1000, 2),
                      'peak_mw': round(cache[k] / 1000, 3), 'share_of_best_cut': round(share(k), 3)})
    return order, hi, curve


def fleet_dispatch(S_flex, Lf_c, units_c, sel, cal, cfg):
    """Year of fleet dispatch for homes `sel`; returns each selected home's net load (len(sel) x H, float32) and the
    kWh it discharged."""
    b = cfg['battery']
    eff = b['round_trip_eff']
    A, E, C = fleet_inputs(Lf_c, units_c, b, sel)
    S0d = S_flex.reshape(cal.D, 24)
    c_f, d_f = np.zeros((cal.D, 24)), np.zeros((cal.D, 24))
    for d in range(cal.D):
        c_f[d], d_f[d], _ = fleet_day_lp(S0d[d], A[d], E, C, eff)
    d_shape = d_f / np.maximum(d_f.sum(1, keepdims=True), 1e-9)
    c_shape = c_f / np.maximum(c_f.sum(1, keepdims=True), 1e-9)
    usable = units_c * b['unit_kwh'] * (1 - b['reserve_soc'])
    net = np.empty((len(sel), cal.H), np.float32)
    delivered_kwh = np.zeros(len(sel))
    for a in range(0, len(sel), 1000):
        j = sel[a:a + 1000]
        Lf = Lf_c[j].astype(np.float64).reshape(len(j), cal.D, 24)
        Ah = np.minimum(np.maximum(Lf, 0), (units_c[j] * b['unit_kw'])[:, None, None])
        target = d_f.sum(1)[None, :] * (usable[j] / E)[:, None]          # this home's share of each day's discharge
        target = np.minimum(target, (Ah * (d_shape[None] > 0)).sum(2))
        lo, hi = np.zeros(target.shape), np.full(target.shape, 1e4)
        for _ in range(50):                                             # discharge = min(own kW, k x fleet shape)
            mid = (lo + hi) / 2
            over = np.minimum(Ah, mid[..., None] * d_shape[None]).sum(2) > target
            lo, hi = np.where(over, lo, mid), np.where(over, mid, hi)
        dis = np.minimum(Ah, lo[..., None] * d_shape[None])
        chg = np.minimum(dis.sum(2, keepdims=True) / eff * c_shape[None], b['grid_charge_cap_kw'])
        net[a:a + len(j)] = (Lf - dis + chg).reshape(len(j), -1)
        delivered_kwh[a:a + len(j)] = dis.sum((1, 2))
    return net, delivered_kwh


# ---------------------------------------------------------------------------------------------- main run
def run(in_dir: Path, cfg: dict, max_customers: int = 0, template: Path = TEMPLATE, out_dir: Path | None = None,
        synthetic: bool = False):
    t0 = time.time()
    out_dir = out_dir or in_dir
    L_all = np.load(in_dir / 'hourly_kwh.npy', mmap_mode='r')
    hours = pd.read_csv(in_dir / 'hours.csv')
    cust = pd.read_csv(in_dir / 'customers.csv')
    N, H = L_all.shape
    assert len(cust) == N and len(hours) == H, 'customers.csv / hours.csv do not match hourly_kwh.npy'
    cal = Calendar(hours, cfg)
    P = part1_probabilities(cust)
    ids = cust['CUSTOMER_NUMBER'].to_numpy(int)
    sc, bcfg, eco, fin = cfg['screening'], cfg['battery'], cfg['economics'], cfg['finance']
    price = cal.price(eco)
    print(f'{N:,} customers x {H:,} hours')

    # Pass 1: system load, windows, CP loads, persistence inputs.
    S = np.zeros(H)
    for a in range(0, N, CHUNK):
        S += np.asarray(L_all[a:a + CHUNK], dtype=np.float64).sum(0)
    windows = find_windows(S, cal, cfg)
    binding = max(('summer', 'winter'), key=lambda s: windows[s]['peak_mw'])
    cal.set_windows(windows, binding)
    Sd = S.reshape(cal.D, 24)
    top40 = np.sort(np.argsort(Sd.max(1))[::-1][:sc.get('persistence_days', 40)])
    cp_idx = {s: windows[s]['_cp_idx'] for s in ('summer', 'winter')}
    cp = {s: np.zeros(N) for s in ('summer', 'winter')}
    pw = np.zeros((N, len(top40)))
    for a in range(0, N, CHUNK):
        Lc = np.asarray(L_all[a:a + CHUNK], dtype=np.float64)
        for s in cp:
            cp[s][a:a + CHUNK] = Lc[:, cp_idx[s]].mean(1)
        Ld = Lc.reshape(len(Lc), cal.D, 24)[:, top40]
        pw[a:a + CHUNK] = np.take_along_axis(Ld, np.broadcast_to(cal.win[top40][None], (len(Lc), len(top40), 4)), 2).mean(2)
    typical = {s: float(np.median(cp[s])) for s in cp}
    persistence = (pw - np.median(pw, 0)[None, :] > sc.get('persistence_margin_kw', 1.0)).mean(1)
    excess_s = {s: cp[s] - typical[s] for s in cp}
    binding_c = np.where(excess_s['winter'] >= excess_s['summer'], 'winter', 'summer')
    excess = np.maximum(excess_s['winter'], excess_s['summer'])
    print(f'  windows: summer HE {windows["summer"]["window_he"]}, winter HE {windows["winter"]["window_he"]}; '
          f'binding season {binding}  ({time.time() - t0:.0f}s)')

    # Pass 2: per-customer flex, batteries, bills.
    R = {k: np.zeros(N) for k in [
        'b_cool', 'b_heat', 'r2', 'annual_kwh', 'peak_kw', 'avg_import_kw', 'export_kwh', 'load_factor',
        'flex_potential_kw', 'bess1_cp_kw', 'bess2_cp_kw', 'units', 'peak_kw_after', 'kwh_shifted', 'kwh_solar_charged',
        'cycles', 'bill_before', 'bill_flex', 'bill_after', 'hp_conv_kw',
        'cp_summer_after', 'cp_winter_after', 'cp_summer_flex', 'cp_winter_flex', 'flex_kw']}
    lever_kw = {k: np.zeros(N) for k in LEVERS}
    action = np.empty(N, object)
    S_flex, S_after, S_naive = np.zeros(H), np.zeros(H), np.zeros(H)
    fleet_on = cfg.get('fleet', {}).get('optimize', False)
    fleet_rows, fleet_Lf, S_after_cand = [], [], np.zeros(H)
    has_lever_all, shed_kwh = np.zeros(N, bool), np.zeros(N)
    profiles, buckets = {}, {}
    td = {s: windows[s]['_top_days'] for s in ('summer', 'winter')}
    min_need, cover = sc['min_need_kw'], sc['flex_covers_share']
    kva = sc['service_incremental_kva']
    for a in range(0, N, CHUNK):
        rows = np.arange(a, min(a + CHUNK, N))
        n = len(rows)
        L = np.asarray(L_all[rows], dtype=np.float64)
        b_cool, b_heat, r2, resid = weather_fit(L, cal)
        R['b_cool'][rows], R['b_heat'][rows], R['r2'][rows] = b_cool, b_heat, r2
        R['annual_kwh'][rows] = L.sum(1)
        R['peak_kw'][rows] = L.max(1)
        R['avg_import_kw'][rows] = np.maximum(L, 0).mean(1)
        R['export_kwh'][rows] = np.maximum(-L, 0).sum(1)
        R['load_factor'][rows] = np.where(L.max(1) > 0, L.mean(1) / np.maximum(L.max(1), 1e-9), 0)
        sh = flex_shed(L, resid, b_cool, b_heat, P, rows, cal, cfg)
        shp = flex_shed(L, resid, b_cool, b_heat, P, rows, cal, cfg, full=True)
        shed = sum(sh.values())
        Lf = shift_to_charge_hours(L, shed, cal)
        bind = binding_c[rows]
        cpb = lambda X: np.where(bind == 'winter', X[:, cp_idx['winter']].mean(1), X[:, cp_idx['summer']].mean(1))  # noqa: E731
        for k in LEVERS:
            lever_kw[k][rows] = cpb(sh[k])
        R['flex_kw'][rows] = cpb(shed)
        R['flex_potential_kw'][rows] = cpb(sum(shp.values()))
        bt1, bt2 = battery(Lf, 1, cal, cfg), battery(Lf, 2, cal, cfg)
        cpf = cpb(Lf)
        g1, g2 = cpf - cpb(bt1['net']), cpf - cpb(bt2['net'])
        R['bess1_cp_kw'][rows], R['bess2_cp_kw'][rows] = g1, g2
        ex, fl = excess[rows], R['flex_kw'][rows]
        storage = P['storage'][rows]
        solar = P['p_solar'][rows] >= 0.5
        has_lever = np.zeros(n, bool)
        minp = sc.get('lever_min_p', 0.5)
        for k, q in (('ev', P['p_ev']), ('cooling', P['p_ac']), ('heating', np.maximum(P['p_heat_pump'], P['p_resist_heat'])),
                     ('water_heater', P['p_wh']), ('pool', P['p_pool'])):
            has_lever |= (q[rows] >= minp) & (lever_kw[k][rows] > 0.005)
        units = np.zeros(n, int)
        need_batt = (ex >= min_need) & (fl < cover * ex) & ~storage
        one = need_batt & (g1 >= sc['min_battery_cp_kw'])
        units[one] = 1
        rest1 = ex - fl - g1
        two = one & (((g2 - g1) >= sc['second_unit_min_gain_kw']) | ((rest1 > kva) & (ex - fl - g2 <= kva) & (g2 > g1)))
        units[two] = 2
        act = np.where(storage & (ex >= 0), 'Enroll existing battery',
              np.where(units > 0, np.where(solar, 'Solar + BESS ', np.where(has_lever, 'Flex + BESS ', 'BESS ')),
              np.where((ex >= min_need) & has_lever & (fl > 0.05), 'Flex only',
              np.where(ex > 0, 'Monitor', 'No action'))))
        act = np.array([f'{x}{u}u' if x.endswith('BESS ') else x for x, u in zip(act, units)], object)
        action[rows] = act
        R['units'][rows] = units
        has_lever_all[rows] = has_lever
        shed_kwh[rows] = shed.sum(1)
        use_flex = ~np.isin(act, ['No action', 'Monitor'])
        Lflex_used = np.where(use_flex[:, None], Lf, L)
        net = np.where((units == 2)[:, None], bt2['net'], np.where((units == 1)[:, None], bt1['net'], Lflex_used))
        naive = np.where((units == 2)[:, None], bt2['naive'], np.where((units == 1)[:, None], bt1['naive'], Lflex_used))
        S_flex += Lflex_used.sum(0)
        S_after += net.sum(0)
        S_naive += naive.sum(0)
        if fleet_on and (units > 0).any():                # battery candidates, for the fleet step
            fleet_rows.append(rows[units > 0])
            fleet_Lf.append(Lf[units > 0].astype(np.float32))
            S_after_cand += net[units > 0].sum(0)
        for s in ('summer', 'winter'):
            R[f'cp_{s}_after'][rows] = net[:, cp_idx[s]].mean(1)
            R[f'cp_{s}_flex'][rows] = Lflex_used[:, cp_idx[s]].mean(1)
        R['peak_kw_after'][rows] = net.max(1)
        dis = np.where(units == 2, bt2['discharged'], np.where(units == 1, bt1['discharged'], 0))
        R['kwh_shifted'][rows] = np.where(use_flex, shed.sum(1), 0) + dis
        R['kwh_solar_charged'][rows] = np.where(units == 2, bt2['solar_charged'], np.where(units == 1, bt1['solar_charged'], 0))
        R['cycles'][rows] = np.where(units > 0, dis / np.maximum(units * bcfg['unit_kwh'], 1e-9), 0)
        R['bill_before'][rows] = L @ price
        R['bill_flex'][rows] = Lflex_used @ price
        R['bill_after'][rows] = net @ price
        R['hp_conv_kw'][rows] = np.where(P['p_resist_heat'][rows] >= minp,
                                          np.maximum(b_heat, 0) * cal.hdh[cp_idx['winter']].mean() * P['p_resist_heat'][rows]
                                          * (1 - 1 / sc['hp_conversion_cop']), 0)
        # profiles (average over each season's 20 top system days) and bill buckets, for the page
        Lb = {k: X.reshape(n, cal.D, 24) for k, X in (('before', L), ('flex', Lflex_used), ('after', net))}
        bk = {k: X.astype(np.float32) @ cal.onehot for k, X in (('b', L), ('f', Lflex_used), ('a', net))}
        for j, i in enumerate(rows):
            if not use_flex[j]:
                continue
            profiles[int(ids[i])] = {s: {k: np.round(Lb[k][j, td[s]].mean(0), 2).tolist() for k in Lb}
                                     for s in ('summer', 'winter')}
            if units[j] > 0:
                buckets[int(ids[i])] = {k: np.round(v[j], 1).tolist() for k, v in bk.items()}
        print(f'  {rows[-1] + 1:,}/{N:,} customers  ({time.time() - t0:.0f}s)')

    # Fleet step: choose how many batteries and dispatch them together (see fleet_day_lp).
    fleet = None
    if fleet_on and fleet_rows:
        cand = np.concatenate(fleet_rows)
        Lf_c = np.concatenate(fleet_Lf)
        del fleet_Lf
        units_c = R['units'][cand].astype(int)
        S_uncoord = S_after.copy()
        order, K, curve = choose_fleet(S_flex, Lf_c, units_c, cal, cfg)
        sel = np.sort(order[:K])
        drop = np.setdiff1d(np.arange(len(cand)), sel)
        net_sel, dis_sel = fleet_dispatch(S_flex, Lf_c, units_c, sel, cal, cfg)
        S_after = S_after - S_after_cand + net_sel.sum(0, dtype=np.float64) + Lf_c[drop].sum(0, dtype=np.float64)
        for idx, nets in ((sel, net_sel), (drop, None)):
            for a in range(0, len(idx), CHUNK):
                j = idx[a:a + CHUNK]
                rows = cand[j]
                Lf = Lf_c[j].astype(np.float64)
                net = Lf if nets is None else nets[a:a + CHUNK].astype(np.float64)
                if nets is None:                          # not in the fleet: no battery, flexibility only
                    R['units'][rows] = 0
                    ex, fl = excess[rows], R['flex_kw'][rows]
                    action[rows] = np.where((ex >= min_need) & has_lever_all[rows] & (fl > 0.05), 'Flex only',
                                            np.where(ex > 0, 'Monitor', 'No action'))
                    R['kwh_shifted'][rows] = shed_kwh[rows]
                    R['cycles'][rows] = 0
                else:
                    u = R['units'][rows]
                    R['kwh_shifted'][rows] = shed_kwh[rows] + dis_sel[a:a + CHUNK]
                    R['cycles'][rows] = dis_sel[a:a + CHUNK] / np.maximum(u * bcfg['unit_kwh'], 1e-9)
                R['kwh_solar_charged'][rows] = 0              # the fleet charges from the grid in its chosen hours
                for s in ('summer', 'winter'):
                    R[f'cp_{s}_after'][rows] = net[:, cp_idx[s]].mean(1)
                R['peak_kw_after'][rows] = net.max(1)
                R['bill_after'][rows] = net @ price
                L = np.asarray(L_all[rows], dtype=np.float64)
                Lb = {k: X.reshape(len(j), cal.D, 24) for k, X in (('before', L), ('flex', Lf), ('after', net))}
                ba = net.astype(np.float32) @ cal.onehot
                for q, i in enumerate(rows):
                    cid = int(ids[i])
                    if action[i] in ('No action', 'Monitor'):
                        profiles.pop(cid, None)
                    else:
                        profiles[cid] = {s: {k: np.round(Lb[k][q, td[s]].mean(0), 2).tolist() for k in Lb}
                                         for s in ('summer', 'winter')}
                    if nets is None:
                        buckets.pop(cid, None)
                    elif cid in buckets:
                        buckets[cid]['a'] = np.round(ba[q], 1).tolist()
        del Lf_c, net_sel
        fleet = {'candidates': int(len(cand)), 'selected_homes': int(K), 'selected_units': int(units_c[sel].sum()),
                 'knee_share': cfg['fleet'].get('knee_share', 0.95), 'curve': curve,
                 'lp_peak_mw': next(c['peak_mw'] for c in curve if c['homes'] == K),
                 'peak_mw_after': round(S_after.max() / 1000, 3),
                 'uncoordinated': {'homes': int(len(cand)), 'units': int(units_c.sum()),
                                   'peak_mw_after': round(S_uncoord.max() / 1000, 3)}}
        print(f"  fleet: {K:,} of {len(cand):,} candidate homes ({fleet['selected_units']:,} units); system peak "
              f"{S_uncoord.max() / 1000:.2f} MW uncoordinated -> {S_after.max() / 1000:.2f} MW coordinated "
              f"(best possible {min(c['peak_mw'] for c in curve):.2f})  ({time.time() - t0:.0f}s)")

    # Pass 3: economics, scores, channels.
    units = R['units'].astype(int)
    for s in ('summer', 'winter'):
        R[f'batt_cp_{s}'] = np.where(units > 0, R[f'cp_{s}_flex'] - R[f'cp_{s}_after'], 0)
    cp_after_b = np.where(binding_c == 'winter', R['cp_winter_after'], R['cp_summer_after'])
    cp_before_b = np.where(binding_c == 'winter', cp['winter'], cp['summer'])
    actionable = ~np.isin(action, ['No action', 'Monitor'])
    relief = np.where(actionable, np.maximum(cp_before_b - cp_after_b, 0), 0)
    excess_after = excess - relief
    batt_sav = R['bill_flex'] - R['bill_after']
    flex_sav = R['bill_before'] - R['bill_flex']
    inc1 = eco['incentive_summer_per_kw'] * R['batt_cp_summer'] + eco['incentive_winter_per_kw'] * R['batt_cp_winter']
    fin_out = {k: np.full(N, np.nan) for k in ('irr', 'npv', 'payback', 'breakeven', 'cost')}
    for i in np.flatnonzero(units > 0):
        cf, cost = cash_flows(units[i], batt_sav[i], R['batt_cp_summer'][i], R['batt_cp_winter'][i], fin, eco)
        v = npv(fin['discount_rate'], cf)
        r_, pb = irr(cf), payback(cf)
        fin_out['irr'][i] = np.nan if r_ is None else r_
        fin_out['npv'][i] = v
        fin_out['payback'][i] = np.nan if pb is None else pb
        fin_out['breakeven'][i] = max(0.0, -v)
        fin_out['cost'][i] = cost
    p_elec = np.maximum.reduce([P['p_ev'], P['p_heat_pump'], P['p_resist_heat']])
    econ = np.where(units > 0, batt_sav + inc1, 0)
    w = cfg['score_weights']
    s_k = {'capacity_relief': pct(relief), 'persistence': pct(persistence), 'constraint': pct(excess),
           'electrification': pct(p_elec), 'economics': pct(econ)}
    score = np.round(100 * sum(w[k] * s_k[k] for k in w), 1)
    order = np.lexsort((ids, -score))
    rank = np.empty(N, int)
    rank[order] = np.arange(1, N + 1)
    wv = cfg['waves']
    wave = np.where(~actionable, '', np.where(score >= wv['wave1_min_score'], 'Wave 1',
                    np.where(score >= wv['wave2_min_score'], 'Wave 2', 'Wave 3')))
    cand = units > 0
    need = np.full(N, np.nan)
    ready = np.full(N, np.nan)
    quad = np.full(N, '', object)
    adoption = (P['p_solar'] >= 0.5) | P['ev_new'] | P['solar_new']
    if cand.any():
        ci = np.flatnonzero(cand)
        need[ci] = 0.6 * pct(relief[ci]) + 0.4 * pct(excess[ci])
        ready[ci] = 0.7 * (1 - pct(fin_out['breakeven'][ci])) + 0.3 * adoption[ci]
        split = sc['matrix_split']
        hi_n, hi_r = need[ci] >= split, ready[ci] >= split
        quad[ci] = np.where(hi_n & hi_r, 'Targeted offer', np.where(hi_n, 'Utility-owned or lease',
                            np.where(hi_r, 'Bring your own battery', 'Flexibility programs first')))
    avoided = eco['avoided_capacity_per_kw_yr'] * relief
    batt_kw = np.where(units > 0, np.where(binding_c == 'winter', R['batt_cp_winter'], R['batt_cp_summer']), 0)
    hp_kw = sc['hp_whatif_kw']
    winter_relief = np.where(actionable, cp['winter'] - R['cp_winter_after'], 0)
    ex_w_after = excess_s['winter'] - winter_relief
    b2 = min(2 * bcfg['unit_kw'], 2 * bcfg['unit_kwh'] * (1 - bcfg['reserve_soc']) * np.sqrt(bcfg['round_trip_eff'])
             / cfg['system_peak']['window_length_h'])
    hp_needs_eval = ex_w_after + hp_kw > kva
    hp_ok = np.maximum(0, ex_w_after + hp_kw - b2) <= kva

    # ------------------------------------------------------------------ customer records
    ev_flag = np.where(P['on_ev'], 'EV on file', np.where(P['p_ev'] >= 0.5, 'EV detected', ''))
    solar_flag = np.where(P['on_pv'], 'Known solar', np.where(P['p_solar'] >= 0.5, 'Unregistered solar', ''))
    recs = []
    for i in range(N):
        u = int(units[i])
        levers = [k for k in LEVERS if lever_kw[k][i] >= 0.05] if actionable[i] else []
        bs = binding_c[i]
        recs.append({
            'customer': int(ids[i]), 'action': action[i], 'units': u,
            'battery_kwh': round(u * bcfg['unit_kwh'], 2), 'battery_kw': round(u * bcfg['unit_kw'], 2),
            'levers': levers, 'binding_season': bs,
            'annual_kwh': round(R['annual_kwh'][i]), 'peak_kw': round(R['peak_kw'][i], 2),
            'peak_kw_after': round(R['peak_kw_after'][i], 2), 'load_factor': round(R['load_factor'][i], 3),
            'export_kwh': round(R['export_kwh'][i]),
            'cp_summer': round(cp['summer'][i], 2), 'cp_winter': round(cp['winter'][i], 2),
            'cp_summer_after': round(R['cp_summer_after'][i] if actionable[i] else cp['summer'][i], 2),
            'cp_winter_after': round(R['cp_winter_after'][i] if actionable[i] else cp['winter'][i], 2),
            'excess_kw': round(excess[i], 2), 'excess_kw_after': round(excess_after[i], 2),
            'constrained': bool(excess[i] > kva), 'constrained_after': bool(excess_after[i] > kva),
            'flex_kw': round(R['flex_kw'][i], 2), 'flex_potential_kw': round(R['flex_potential_kw'][i], 2),
            'lever_kw': {k: round(lever_kw[k][i], 2) for k in LEVERS},
            'bess1_cp_kw': round(R['bess1_cp_kw'][i], 2), 'bess2_cp_kw': round(R['bess2_cp_kw'][i], 2),
            'cp_relief_kw': round(relief[i], 2), 'cp_relief_binding_kw': round(relief[i], 2),
            'persistence': round(persistence[i], 3),
            'kwh_shifted': round(R['kwh_shifted'][i]), 'kwh_solar_charged': round(R['kwh_solar_charged'][i]),
            'cycles': round(R['cycles'][i]),
            'bill_before': round(R['bill_before'][i]), 'bill_after': round(R['bill_after'][i] if actionable[i] else R['bill_before'][i]),
            'bill_savings': round((R['bill_before'][i] - R['bill_after'][i]) if actionable[i] else 0),
            'bill_savings_flex': round(flex_sav[i]),
            'incentive_yr': round(inc1[i]) if u else 0.0, 'avoided_capacity_yr': round(avoided[i]),
            'battery_cost': int(fin_out['cost'][i]) if u else 0,
            'b_cool': round(R['b_cool'][i], 3), 'b_heat': round(R['b_heat'][i], 3), 'r2': round(R['r2'][i], 2),
            'p_ev': round(P['p_ev'][i], 2), 'p_heat_pump': round(P['p_heat_pump'][i], 2),
            'p_resist_heat': round(P['p_resist_heat'][i], 2), 'p_ac': round(P['p_ac'][i], 2),
            'p_solar': round(P['p_solar'][i], 2), 'p_pool': round(P['p_pool'][i], 2),
            'segment': str(P['segment'][i]),
            'hp_needs_eval': bool(hp_needs_eval[i]), 'hp_ok_with_bess': bool(hp_ok[i]),
            'solar_flag': solar_flag[i], 'solar_new': bool(P['solar_new'][i]),
            'solar_onset': str(P['solar_onset'][i]) if P['solar_new'][i] else '',
            'ev_flag': ev_flag[i], 'ev_new': bool(P['ev_new'][i] and ev_flag[i]),
            'ev_onset': str(P['ev_onset'][i]) if (P['ev_new'][i] and ev_flag[i]) else '',
            'ev_events_yr': int(P['ev_events_yr'][i]) if ev_flag[i] else 0,
            'peak_reduction_kw': round(max(0.0, R['peak_kw'][i] - R['peak_kw_after'][i]) if actionable[i] else 0, 2),
            'hp_conv_kw': round(R['hp_conv_kw'][i], 2), 'avg_import_kw': round(R['avg_import_kw'][i], 3),
            'backup_h': round(u * bcfg['unit_kwh'] / R['avg_import_kw'][i], 1) if u and R['avg_import_kw'][i] > 0 else None,
            'batt_savings': round(batt_sav[i]) if u else 0.0,
            'batt_cp_summer': round(R['batt_cp_summer'][i], 3), 'batt_cp_winter': round(R['batt_cp_winter'][i], 3),
            'irr': round_or_none(fin_out['irr'][i], 4), 'npv': round_or_none(fin_out['npv'][i], 0),
            'payback': round_or_none(fin_out['payback'][i], 1), 'breakeven_rebate': round_or_none(fin_out['breakeven'][i], 0),
            **{f's_{k}': round(float(v[i]), 3) for k, v in s_k.items()},
            'score': float(score[i]), 'wave': wave[i], 'rank': int(rank[i]),
            'reason': reason(i, bs, cp, typical, excess, excess_after, lever_kw, R, units, actionable, action, kva),
            'need_score': round_or_none(need[i], 3), 'ready_score': round_or_none(ready[i], 3),
            'quadrant': quad[i], 'channel': CHANNELS.get(quad[i], ''),
        })

    summary = build_summary(recs, P, R, cp, typical, windows, binding, S, S_flex, S_after, S_naive, cal, cfg, fin_out,
                            relief, units, actionable, batt_kw, p_elec)
    summary['fleet'] = fleet
    summary['clusters'] = cluster_summary(cust, units, batt_sav)
    recs.sort(key=lambda r: r['rank'])
    listed = recs
    if max_customers and len(recs) > max_customers:
        listed = recs[:max_customers]
    keep = {r['customer'] for r in listed}
    max_prof = cfg.get('export', {}).get('max_profiles', 3000)
    prof_ids = [r['customer'] for r in listed if r['customer'] in profiles][:max_prof]
    data = {
        'meta': {'generated': datetime.now().strftime('%Y-%m-%d %H:%M'), 'synthetic': bool(synthetic),
                 'n_customers': N, 'n_valid': N, 'n_listed': len(listed), 'runtime_s': round(time.time() - t0, 1),
                 'inputs': 'Part 1 appliance inventory and technology scores, Part 2 segments, cleaned hourly loads',
                 'tariff': eco.get('tariff')},
        'config': {'data': {'start_date': str(cal.date[0]), 'n_days': cal.D, 'max_customer_number': int(ids.max()),
                            'min_valid_share': 0.8}, **{k: v for k, v in cfg.items() if not k.startswith('_')}},
        'summary': summary,
        'profiles': {str(k): profiles[k] for k in prof_ids},
        'buckets': {str(k): v for k, v in buckets.items() if k in keep},
        'customers': listed,
    }
    write_outputs(data, recs, out_dir, template)
    print(f'Done in {time.time() - t0:.0f}s: {summary["n_bess"]:,} battery candidates, '
          f'{summary["n_actionable"]:,} actionable, system peak {summary["system_peak_mw_before"]} -> '
          f'{summary["system_peak_mw_after"]} MW')
    return data


def reason(i, bs, cp, typical, excess, excess_after, lever_kw, R, units, actionable, action, kva):
    c, t = cp[bs][i], typical[bs]
    txt = f'{bs.capitalize()} system-peak load {c:.1f} kW vs {t:.1f} kW for a typical home ({excess[i]:+.1f} kW).'
    if excess[i] > kva:
        txt += ' Above the 3 kVA incremental guideline.'
    if not actionable[i]:
        return txt
    used = [k for k in LEVERS if lever_kw[k][i] >= 0.05]
    if used:
        txt += f' Flex: {", ".join(LEVER_TEXT[k] for k in used)} sheds {R["flex_kw"][i]:.1f} kW.'
    u = int(units[i])
    if u == 1:
        txt += f' One unit shaves {R["bess1_cp_kw"][i]:.1f} kW.'
    elif u == 2:
        txt += (f' Two units shave {R["bess2_cp_kw"][i]:.1f} kW; the second unit adds '
                f'{R["bess2_cp_kw"][i] - R["bess1_cp_kw"][i]:.1f} kW.')
    if action[i] == 'Enroll existing battery':
        txt += ' Solar and a battery already detected: enroll for performance payments.'
    if excess[i] > kva:
        txt += (' Still above the guideline after action: refer for service evaluation.' if excess_after[i] > kva
                else ' Back inside the guideline after action.')
    return txt


def build_summary(recs, P, R, cp, typical, windows, binding, S, S_flex, S_after, S_naive, cal, cfg, fin_out,
                  relief, units, actionable, batt_kw, p_elec):
    df = pd.DataFrame(recs)
    sc, eco, fin, bcfg = cfg['screening'], cfg['economics'], cfg['finance'], cfg['battery']
    bess = df[df.units > 0]
    mw = lambda x: round(float(np.nansum(x)) / 1000, 3)  # noqa: E731
    ann = sum(1 / (1 + fin['discount_rate']) ** t for t in range(1, fin['life_years'] + 1))
    td = {s: windows[s]['_top_days'] for s in ('summer', 'winter')}
    prof = lambda X, s: (X.reshape(cal.D, 24)[td[s]].mean(0) / 1000).round(3).tolist()  # noqa: E731
    quadrants = {}
    for q, ch in CHANNELS.items():
        g = bess[bess.quadrant == q]
        per_kwh = (g.breakeven_rebate / (g.units * bcfg['unit_kwh'])).dropna()
        quadrants[q] = {'homes': int(len(g)), 'rebate_p75': round(float(per_kwh.quantile(0.75)), 0) if len(per_kwh) else None,
                        'mw': mw(g.cp_relief_binding_kw), 'channel': ch}
    bk = np.where(df.binding_season == 'winter', df.batt_cp_winter, df.batt_cp_summer)
    lev = pd.DataFrame(list(df.lever_kw))
    dr_mask = (df.p_ac >= 0.5) | (df.p_heat_pump >= 0.5)
    programs = [
        {'program': 'Battery storage', 'rule': 'units > 0', 'homes': int((df.units > 0).sum()), 'mw': mw(bk[df.units > 0]),
         'channel': 'Channel assigned by the prioritization matrix'},
        {'program': 'Managed EV charging', 'rule': 'EV detected', 'homes': int((df.ev_flag != '').sum()),
         'mw': mw(lev.ev[df.ev_flag != '']), 'channel': 'Enrollment at EV purchase, charger rebate or via vehicle telematics'},
        {'program': 'Thermostat and heat pump DR', 'rule': 'AC or heat pump', 'homes': int(dr_mask.sum()),
         'mw': mw((lev.cooling + lev.heating)[dr_mask]), 'channel': 'Bring-your-own-thermostat enrollment'},
        {'program': 'Resistance heat to heat pump', 'rule': 'electric resistance heat', 'homes': int((df.p_resist_heat >= 0.5).sum()),
         'mw': mw(df.hp_conv_kw), 'channel': 'Heat pump rebate with a winter peak adder'},
        {'program': 'Solar + storage', 'rule': 'solar and battery candidate',
         'homes': int(((df.p_solar >= 0.5) & (df.units > 0)).sum()), 'mw': mw(bk[(df.p_solar >= 0.5) & (df.units > 0)]),
         'channel': 'Offer at solar interconnection application'},
        {'program': 'Enroll existing batteries', 'rule': 'solar + battery already',
         'homes': int((df.action == 'Enroll existing battery').sum()), 'mw': 0.0, 'channel': 'Performance payments only; no new hardware'},
    ]
    # rebate cost-effective for both parties: breakeven <= value x relief x annuity
    rel_b = bk[df.units > 0]
    be = bess.breakeven_rebate.to_numpy(float)
    curve = []
    for v in range(0, 1001, 25):
        ok = be <= v * rel_b * ann
        curve.append({'value': v, 'homes': int(ok.sum()), 'mw': mw(rel_b[ok])})
    cur = eco['avoided_capacity_per_kw_yr']
    now = be <= cur * rel_b * ann
    first = next((c['value'] for c in curve if c['homes'] > 0), None)
    t_off = quadrants['Targeted offer']['rebate_p75']
    need_kw = be / np.maximum(rel_b * ann, 1e-9)
    inv_items = [('EV', df.p_ev), ('Air conditioning', df.p_ac), ('Heat pump', df.p_heat_pump),
                 ('Electric resistance heat', df.p_resist_heat), ('Pool', df.p_pool)]
    items = [{'tech': t, 'expected': int(round(p.sum())), 'confident': int((p >= 0.5).sum())} for t, p in inv_items]
    known = int(P['on_pv'].sum())
    solar_all = int((P['p_solar'] >= 0.5).sum())
    items += [{'tech': 'Solar', 'expected': known, 'confident': known},
              {'tech': 'Solar (incl. unregistered)', 'expected': solar_all, 'confident': solar_all}]
    waves = []
    for wname in ('Wave 1', 'Wave 2', 'Wave 3'):
        g = df[df.wave == wname]
        waves.append({'wave': wname, 'customers': int(len(g)), 'cp_relief_kw': round(float(g.cp_relief_binding_kw.sum()), 2),
                      'batteries': int((g.units > 0).sum()), 'units': int(g.units.sum())})
    irr_b = bess.irr.dropna()
    return {
        'n_customers': len(df), 'n_valid': len(df),
        'typical_cp_kw': {s: round(typical[s], 2) for s in typical}, 'binding_season': binding,
        'windows': {s: {k: v for k, v in windows[s].items() if not k.startswith('_')} for s in ('summer', 'winter')},
        'system_peak_mw_before': round(S.max() / 1000, 3), 'system_peak_mw_after': round(S_after.max() / 1000, 3),
        'cp_relief_mw': {s: round(float(np.where(actionable, cp[s] - df.sort_index()[f'cp_{s}_after'].to_numpy(), 0).sum()) / 1000, 3)
                         for s in ('summer', 'winter')},
        'n_actionable': int(actionable.sum()), 'n_bess': int(len(bess)), 'n_units': int(bess.units.sum()),
        'n_bess_1u': int((df.units == 1).sum()), 'n_bess_2u': int((df.units == 2).sum()),
        'n_flex_only': int((df.action == 'Flex only').sum()),
        'n_solar_lane': int(df.action.str.startswith('Solar + BESS').sum()),
        'n_existing_storage': int((df.action == 'Enroll existing battery').sum()),
        'n_constrained': int(df.constrained.sum()), 'n_constrained_resolved': int((df.constrained & ~df.constrained_after).sum()),
        'n_hp_needs_eval': int(df.hp_needs_eval.sum()), 'n_hp_ok_with_bess': int(df.hp_ok_with_bess.sum()),
        'storage_mwh': round(float(df.battery_kwh.sum()) / 1000, 2), 'storage_mw': round(float(df.battery_kw.sum()) / 1000, 2),
        'kwh_shifted': int(df.kwh_shifted.sum()), 'kwh_solar_charged': int(df.kwh_solar_charged.sum()),
        'bill_savings_total': int(df.bill_savings.sum()),
        'bill_savings_median_bess': int(bess.bill_savings.median()) if len(bess) else 0,
        'incentive_total': int(df.incentive_yr.sum()), 'avoided_capacity_total': int(df.avoided_capacity_yr.sum()),
        'battery_cost_total': int(df.battery_cost.sum()),
        'service_upgrades_avoided_value': int(eco['service_upgrade_cost'] * int((df.constrained & ~df.constrained_after).sum())),
        'action_mix': {k: int(v) for k, v in df.action.value_counts().items()},
        'customer_peak_reduction_mw': mw(df.peak_reduction_kw),
        'backup_h_median': round(float(bess.backup_h.median()), 1) if len(bess) else None,
        'irr_median': round(float(irr_b.median()), 4) if len(irr_b) else None,
        'n_irr_above_discount': int((irr_b > fin['discount_rate']).sum()),
        'breakeven_rebate_total': int(bess.breakeven_rebate.sum()),
        'breakeven_rebate_median': int(bess.breakeven_rebate.median()) if len(bess) else 0,
        'quadrants': quadrants, 'programs': programs,
        'winwin': {'annuity': round(ann, 3), 'current_value': cur, 'homes_now': int(now.sum()), 'mw_now': mw(rel_b[now]),
                   'median_need_per_kw_yr': round(float(np.median(need_kw[rel_b > 0])), 0) if (rel_b > 0).any() else None,
                   'first_winwin_value': first, 'pilot_rebate_per_kwh': t_off,
                   'pilot_rebate_covers': '75% of targeted-offer customers reach the target return', 'curve': curve},
        'inventory': {'items': items, 'heating_sensitive': int((df.b_heat >= sc.get('sensitive_kw_per_f', 0.02)).sum()),
                      'cooling_sensitive': int((df.b_cool >= sc.get('sensitive_kw_per_f', 0.02)).sum()), 'n': len(df)},
        'flags': {'solar_known': known, 'solar_unregistered': solar_all - known, 'solar_new': int(df.solar_new.sum()),
                  'ev_on_file': int((df.ev_flag == 'EV on file').sum()), 'ev_detected': int((df.ev_flag == 'EV detected').sum()),
                  'ev_new': int(df.ev_new.sum())},
        'waves': waves,
        'system_profile_mw': {s: {'before': prof(S, s), 'flex': prof(S_flex, s), 'after': prof(S_after, s),
                                  'naive': prof(S_naive, s)} for s in ('summer', 'winter')},
        'system_peak_mw_naive': round(S_naive.max() / 1000, 3),
        'system_heatmap': (S.reshape(cal.D, 24) / 1000).round(3).tolist(),
    }


def write_outputs(data, recs, out_dir: Path, template: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, separators=(',', ':'), allow_nan=False,
                         default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    (out_dir / 'app_data.json').write_text(payload)
    cols = ['rank', 'customer', 'segment', 'action', 'units', 'wave', 'score', 'quadrant', 'cp_relief_kw', 'excess_kw',
            'flex_kw', 'bill_savings', 'batt_savings', 'irr', 'breakeven_rebate', 'p_ev', 'p_heat_pump', 'p_resist_heat',
            'p_ac', 'p_pool', 'p_solar', 'reason']
    pd.DataFrame(recs)[cols].sort_values('rank').to_csv(out_dir / 'candidates.csv', index=False)
    if template.exists():
        html = template.read_text()
        start = html.index('window.__APP_DATA__=')
        end = html.index('</script>', start)
        page = html[:start] + 'window.__APP_DATA__=' + payload.replace('</', '<\\/') + ';' + html[end:]
        tariff = data['config']['economics'].get('tariff')
        if tariff and PAGE_RATE_LABEL in page:        # name the rate actually used in the Assumptions table
            page = page.replace(PAGE_RATE_LABEL, json.dumps(tariff['label']))
        fleet = data['summary'].get('fleet')
        if fleet:                                     # describe the fleet dispatch instead of per-home dispatch
            for old, new in PAGE_FLEET_TEXT:
                if old in page:
                    page = page.replace(old, new.replace('{homes}', f"{fleet['selected_homes']:,}").replace(
                        '{cands}', f"{fleet['candidates']:,}").replace('{pct}', f"{fleet['knee_share']:.0%}"))
                else:
                    print(f'  note: page text not found, left as is: {old[:60]}...')
        (out_dir / 'index.html').write_text(page)
        print(f'  page: {out_dir / "index.html"} ({len(page) / 1e6:.1f} MB)')


def main(argv=None):
    ap = argparse.ArgumentParser(description='Part 3 battery and flexibility pipeline (Part 1-2 inputs only).')
    ap.add_argument('--input', type=Path, default=DEFAULT_IN, help='folder written by export_part3_inputs.py')
    ap.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    ap.add_argument('--output', type=Path, default=None, help='output folder (default: the input folder)')
    ap.add_argument('--max-customers', type=int, default=0, help='list only the top N customers on the page')
    ap.add_argument('--synthetic', action='store_true', help='mark the output as demonstration data')
    ap.add_argument('--utility', choices=sorted(DEFAULT_PLAN), help="use this utility's residential time-varying rate "
                    'from data/tariffs (default: the rates in the config)')
    ap.add_argument('--plan', help='rate plan id in data/tariffs (default: the utility\'s time-varying plan)')
    ap.add_argument('--region', help='NYSEG supply region (default: West)')
    args = ap.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    if args.utility:
        cfg = utility_rates(cfg, args.utility, args.plan, args.region)
        print(f"rates: {cfg['economics']['tariff']['label']}")
    run(args.input, cfg, args.max_customers, out_dir=args.output, synthetic=args.synthetic)


if __name__ == '__main__':
    sys.exit(main())
