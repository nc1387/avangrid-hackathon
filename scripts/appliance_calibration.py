"""Check and calibrate the appliance rules by adding simulated appliances to real customers.

For 2,000 random customers on none of the EV/HP/PV lists, each kind of appliance load is added
(avangrid.appliances.inject) and the features recomputed. For each rule we report:
  detected_new: share flagged after injection, among customers NOT flagged before (sensitivity to a
                newly added appliance of that size);
  flag_rate_original: share of the same customers flagged with no injection (real appliances +
                false flags).
For the rarer appliances, thresholds are chosen on a grid to catch as many simulated appliances as
possible while flagging no more than a plausible share of ordinary customers (FLAG_BUDGET; for these
rare appliances most flags on ordinary customers would be false, so the budget caps false flags).
AC and heating keep their physical thresholds (they are too common for this trade-off).
"""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import avangrid  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from avangrid import appliances, cache, grids  # noqa: E402

hd = grids.load_hourly(verbose=False)
lab = hd.labels()
not_listed = np.flatnonzero(~(lab['EV'] | lab['HP'] | lab['PV']))
rng = np.random.default_rng(42)
sample = np.sort(rng.choice(not_listed, 2000, replace=False))
K0 = np.asarray(hd.kwh[sample])

F_all = appliances.appliance_features(hd.kwh, hd.temp_hourly, hd.dates, hd.weekend)
scales = appliances.fit_scales(F_all)
F0 = {k: v[sample] for k, v in F_all.items()}

KINDS = {'air_conditioning': ['air_conditioning'],
         'room_air_conditioning': ['air_conditioning'],
         'electric_heating': ['electric_heating_primary', 'electric_heating_any'],
         'heat_pump_heating': ['electric_heating_primary', 'electric_heating_any'],
         'electric_water_heater': ['electric_water_heater'],
         'pool': ['pool'],
         'pool_variable_speed': ['pool'],
         'hot_tub': ['hot_tub', 'ev_level1', 'ev_level2'],
         'ev_level1': ['ev_level1', 'ev_level2', 'hot_tub'],
         'ev_level2': ['ev_level2', 'ev_level1', 'hot_tub']}

# Solar + battery is only defined for solar homes, so it is checked on the PV-list customers instead.
pv_sample = np.flatnonzero(lab['PV'])
F0_pv = {k: v[pv_sample] for k, v in F_all.items()}
F_pv_battery = appliances.appliance_features(
    appliances.inject(np.asarray(hd.kwh[pv_sample]), 'battery', hd.temp_hourly, hd.dates, np.random.default_rng(7)),
    hd.temp_hourly, hd.dates, hd.weekend)

injected = {}
for kind in KINDS:
    Ki = appliances.inject(K0, kind, hd.temp_hourly, hd.dates, np.random.default_rng(7))
    injected[kind] = appliances.appliance_features(Ki, hd.temp_hourly, hd.dates, hd.weekend)
    print(f'injected {kind}', flush=True)


def rates(rule_conds, F_inj, rule_name, F_base=None):
    F_base = F0 if F_base is None else F_base
    c0 = appliances.classify(F_base, scales, {rule_name: rule_conds})[rule_name] >= 0.5
    c1 = appliances.classify(F_inj, scales, {rule_name: rule_conds})[rule_name] >= 0.5
    return float(c1[~c0].mean()) if (~c0).any() else np.nan, float(c0.mean())


# Grid search for the rare appliances
GRIDS = {
    'electric_water_heater': {'wh_morning_share': [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5], 'wh_morning_kwh': [1.0, 1.25, 1.5, 2.0, 2.5]},
    'pool': {'pool_day_floor_jump': [0.1, 0.2, 0.3, 0.4, 0.5, 0.7], 'pool_day_night_jump': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]},
    'hot_tub': {'hot_tub_night_share': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]},
    'ev_level1': {'ev_l1_night_share': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]},
    'ev_level2': {'ev_l2_night_share': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]},
    'solar_battery': {'solar_midday_zero_share': [0.2, 0.3], 'battery_evening_zero_share': [0.1, 0.15, 0.2, 0.3, 0.4],
                      'summer_morning_kwh': [0.25]},
}
BATTERY = {'solar_battery'}
# Maximum share of ordinary customers a rule may flag (battery: share of PV-list customers).
FLAG_BUDGET = {'electric_water_heater': 0.15, 'pool': 0.04, 'hot_tub': 0.04, 'ev_level1': 0.02, 'ev_level2': 0.02,
               'solar_battery': 0.02}
chosen = {}
grid_rows = []
for rule, grid in GRIDS.items():
    feats = list(grid)
    best = None
    for vals in itertools.product(*grid.values()):
        conds = list(zip(feats, vals))
        if rule in BATTERY:
            det, flag = rates(conds, F_pv_battery, rule, F0_pv)
        else:
            det, flag = rates(conds, injected[rule], rule)
        within = flag <= FLAG_BUDGET[rule]
        key = (within, det if within else -flag)          # best detection within budget, else fewest flags
        grid_rows.append({'rule': rule, **{f: v for f, v in conds}, 'detected_new': det, 'flag_rate_original': flag,
                          'within_budget': within})
        if best is None or key > best[0]:
            best = (key, conds, det, flag)
    chosen[rule] = best[1]
    print(f'{rule}: chosen {best[1]} -> detects {best[2]:.2f}, flags {best[3]:.3f} of ordinary customers', flush=True)
cache.save_results('appliance_threshold_grid', pl.DataFrame(grid_rows, strict=False))

rows = []
for kind, rules in KINDS.items():
    for rule in rules:
        conds = chosen.get(rule, appliances.RULES[rule])
        det, flag = rates(conds, injected[kind], rule)
        rows.append({'injected': kind, 'rule': rule, 'thresholds': str(conds), 'detected_new': det, 'flag_rate_original': flag})
conds = chosen['solar_battery']
det, flag = rates(conds, F_pv_battery, 'solar_battery', F0_pv)
rows.append({'injected': 'battery (on PV-list customers)', 'rule': 'solar_battery', 'thresholds': str(conds),
             'detected_new': det, 'flag_rate_original': flag})
res = pl.DataFrame(rows)
cache.save_results('appliance_calibration', res)
cache.save_results('appliance_chosen_thresholds', pl.DataFrame(
    [{'rule': r, 'feature': f, 'threshold': t} for r, c in chosen.items() for f, t in c]))
with pl.Config(tbl_rows=-1, fmt_str_lengths=80, tbl_width_chars=200, tbl_hide_dataframe_shape=True):
    print(res)
