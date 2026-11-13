"""Classify every customer for AC, electric heating (heat pump vs resistance), electric water heater,
pool, hot tub, EV charging (Level 1 / Level 2) and solar + battery, with confidence scores; save the
inventory and profile charts.

Outputs:
  avangrid.db table appliance_inventory (one row per customer)
  onedrive-export/Data/eda/appliance_inventory.csv
  onedrive-export/Data/eda/appliance_profiles.png    (AC, electric heating, water heater, pool)
  onedrive-export/Data/eda/appliance_profiles_2.png  (EV Level 2, EV Level 1, hot tub, solar + battery)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import avangrid  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from avangrid import appliances, cache, config, db, grids  # noqa: E402

hd = grids.load_hourly(verbose=False)
F = appliances.appliance_features(hd.kwh, hd.temp_hourly, hd.dates, hd.weekend)
conf = appliances.classify(F, appliances.fit_scales(F))
flag = {k: v >= 0.5 for k, v in conf.items()}

# Heat pump vs resistance: combine the electric-heating evidence with the supervised HP model score
# (out-of-fold for listed customers). Resistance heat draws ~3x the kW of a heat pump for the same heat,
# so "resistance likely" also requires a large heating load.
pred = db.sql('SELECT CUSTOMER_NUMBER, hp_percentile, hp_listed FROM technology_predictions')
hp_pct = (pl.DataFrame({'CUSTOMER_NUMBER': hd.customer_numbers}).join(pred, on='CUSTOMER_NUMBER', how='left')
          ['hp_percentile'].to_numpy())
# Top 2% of the HP model (~500 customers), in line with its estimate of ~240 heat pumps in total, or on the HP list.
HP_PCT = 0.98
hp_listed = hd.labels()['HP']
heating_type = np.where(
    flag['electric_heating_any'] & ((hp_pct >= HP_PCT) | hp_listed), 'heat pump likely',
    np.where(flag['electric_heating_primary'], 'resistance likely',
             np.where(flag['electric_heating_any'], 'electric (type uncertain)', 'none detected')))
ac_type = np.where(flag['air_conditioning'] & (F['ac_added_kw'] >= 1.5), 'central likely',
                   np.where(flag['air_conditioning'], 'room / partial', 'none detected'))
# Solar + battery only where there is independent evidence of solar: on the PV list or PV model score >= 0.5.
pv = (pl.DataFrame({'CUSTOMER_NUMBER': hd.customer_numbers})
      .join(db.sql('SELECT CUSTOMER_NUMBER, pv_score, pv_listed FROM technology_predictions'), on='CUSTOMER_NUMBER', how='left'))
solar_evidence = pv['pv_listed'].fill_null(False).to_numpy() | (pv['pv_score'].fill_null(0).to_numpy() >= 0.5)
conf['solar_battery'] = np.where(solar_evidence, conf['solar_battery'], 0.0)
flag['solar_battery'] = conf['solar_battery'] >= 0.5
ev_charging = np.where(flag['ev_level2'], 'Level 2 likely', np.where(flag['ev_level1'], 'Level 1 likely', 'none detected'))

inv = pl.DataFrame({
    'CUSTOMER_NUMBER': hd.customer_numbers,
    'ac_confidence': conf['air_conditioning'], 'ac_flag': flag['air_conditioning'], 'ac_type': ac_type,
    'ac_added_kw': F['ac_added_kw'], 'ac_share_of_summer_use': F['ac_share'],
    'elec_heat_confidence': conf['electric_heating_any'], 'elec_heat_primary_confidence': conf['electric_heating_primary'],
    'elec_heat_flag': flag['electric_heating_any'], 'elec_heat_primary_flag': flag['electric_heating_primary'],
    'heating_type': heating_type, 'heat_kw_at_20f': F['heat_kw_at_20f'], 'hp_model_percentile': hp_pct,
    'water_heater_confidence': conf['electric_water_heater'], 'water_heater_flag': flag['electric_water_heater'],
    'wh_morning_spike_share': F['wh_morning_share'],
    'pool_confidence': conf['pool'], 'pool_flag': flag['pool'],
    'pool_daytime_floor_jump_kw': F['pool_day_floor_jump'], 'pool_day_minus_night_jump_kw': F['pool_day_night_jump'],
    'hot_tub_confidence': conf['hot_tub'], 'hot_tub_flag': flag['hot_tub'], 'hot_tub_night_spike_share': F['hot_tub_night_share'],
    'ev_charging_type': ev_charging, 'ev_level2_confidence': conf['ev_level2'], 'ev_level1_confidence': conf['ev_level1'],
    'ev_l2_night_share': F['ev_l2_night_share'], 'ev_l1_night_share': F['ev_l1_night_share'],
    'solar_battery_confidence': conf['solar_battery'], 'solar_battery_flag': flag['solar_battery'],
    'solar_midday_zero_share': F['solar_midday_zero_share'], 'battery_evening_zero_share': F['battery_evening_zero_share'],
})
with db.connect(read_only=False) as con:
    con.register('inv_df', inv)
    con.execute('CREATE OR REPLACE TABLE appliance_inventory AS SELECT * FROM inv_df ORDER BY CUSTOMER_NUMBER')
inv.write_csv(config.EDA_DIR / 'appliance_inventory.csv')
cache.save_results('appliance_inventory', inv)

lab = hd.labels()
groups = {'All customers': np.ones(len(inv), bool), 'HP listed': lab['HP'], 'EV listed': lab['EV'], 'PV listed': lab['PV']}
summary = pl.DataFrame([{
    'group': g, 'customers': int(m.sum()),
    'AC': flag['air_conditioning'][m].mean(), 'AC central': (ac_type[m] == 'central likely').mean(),
    'elec heat (any)': flag['electric_heating_any'][m].mean(), 'elec heat (primary)': flag['electric_heating_primary'][m].mean(),
    'heat pump likely': (heating_type[m] == 'heat pump likely').mean(),
    'resistance likely': (heating_type[m] == 'resistance likely').mean(),
    'water heater': flag['electric_water_heater'][m].mean(), 'pool': flag['pool'][m].mean(),
    'hot tub': flag['hot_tub'][m].mean(), 'EV L2': flag['ev_level2'][m].mean(),
    'EV L1 only': ((ev_charging == 'Level 1 likely')[m]).mean(), 'solar+battery': flag['solar_battery'][m].mean(),
} for g, m in groups.items()]).with_columns(pl.selectors.float().round(3))
cache.save_results('appliance_summary', summary)
with pl.Config(tbl_cols=-1, tbl_width_chars=220, tbl_hide_dataframe_shape=True):
    print(summary)
    print(pl.DataFrame({'heating_type': heating_type}).group_by('heating_type').len().sort('len', descending=True))
    counts = {name: int(v.sum()) for name, v in flag.items()}
    print('Customers flagged:', counts, '| EV Level 1 only:', int((ev_charging == 'Level 1 likely').sum()))

# Cross-check the rule-based EV charging flags against the supervised EV model (notebook-40 features).
ev_pct = (pl.DataFrame({'CUSTOMER_NUMBER': hd.customer_numbers})
          .join(db.sql('SELECT CUSTOMER_NUMBER, ev_percentile FROM technology_predictions'), on='CUSTOMER_NUMBER', how='left')
          ['ev_percentile'].to_numpy())
xcheck = pl.DataFrame([{'group': name, 'customers': int(m.sum()),
                        'median EV-model percentile': float(np.nanmedian(ev_pct[m])),
                        'share in EV-model top 5%': float(np.nanmean(ev_pct[m] >= 0.95))}
                       for name, m in [('Level 2 flagged', flag['ev_level2']),
                                       ('Level 1 only', ev_charging == 'Level 1 likely'),
                                       ('Hot tub flagged', flag['hot_tub']),
                                       ('Not flagged for charging', ev_charging == 'none detected')]])
cache.save_results('appliance_ev_crosscheck', xcheck)
with pl.Config(tbl_hide_dataframe_shape=True, float_precision=3):
    print(xcheck)

# ---- profile charts: average hourly kWh, flagged vs not, on the days each signature shows ----
temp = hd.temp
month = hd.dates.dt.month().to_numpy()
panels = [
    ('air_conditioning', 'Air conditioning: hot days (>= 80 F)', temp >= 80),
    ('electric_heating_any', 'Electric heating: cold days (< 25 F)', temp < 25),
    ('electric_water_heater', 'Electric water heater: mild days (55-70 F)', (temp >= 55) & (temp <= 70)),
]
t_max = hd.temp_hourly.max(1)
dom = hd.dates.dt.day().to_numpy()
pool_summer = (((month == 6) & (dom >= 15)) | np.isin(month, [7, 8])) & (t_max <= 75) & (temp >= 58)
pool_shoulder = ((month == 4) | ((month == 5) & (dom <= 20)) | (month == 10)) & (t_max <= 75) & (temp >= 50)
fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
hours = np.arange(1, 25)
def mean_profile(sel, days):
    idx = np.flatnonzero(sel)
    prof = np.zeros(24); count = 0
    for s in range(0, len(idx), 3000):
        k = np.asarray(hd.kwh[idx[s:s + 3000]])[:, days]
        prof += np.nansum(np.nanmean(k, 1), 0); count += len(k)
    return prof / count, len(idx)


lines = {rule: [(flag[rule], days, 'Flagged', '#E69F00', '-'), (~flag[rule], days, 'Not flagged', '#8c8c8c', '-')]
         for rule, _, days in panels}
lines['pool'] = [(flag['pool'], pool_summer, 'Flagged, cool summer days', '#E69F00', '-'),
                 (flag['pool'], pool_shoulder, 'Flagged, cool spring/autumn days', '#E69F00', '--'),
                 (~flag['pool'], pool_summer, 'Not flagged, cool summer days', '#8c8c8c', '-')]
panels.append(('pool', 'Pool: days whose hottest hour is <= 75 F (AC mostly off)', None))
for ax, (rule, title, _) in zip(axes.flat, panels):
    for sel, days, label, color, style in lines[rule]:
        prof, count_c = mean_profile(sel, days)
        ax.plot(hours, prof, color=color, linestyle=style, linewidth=2.2, label=f'{label} (n = {count_c:,})')
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(1, 25, 2))
    ax.set_xlim(1, 24)
    ax.set_ylim(bottom=0)
    ax.grid(color='#e5e5e5', linewidth=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
for ax in axes[1]:
    ax.set_xlabel('Hour ending')
for ax in axes[:, 0]:
    ax.set_ylabel('Average kWh per hour')
fig.suptitle('Average daily profile of customers flagged for each appliance', y=1.0)
fig.tight_layout()
fig.savefig(config.EDA_DIR / 'appliance_profiles.png', dpi=150)
print('saved', config.EDA_DIR / 'appliance_profiles.png')


# ---- second chart: EV charging, hot tub, solar + battery ----
mild_night_days = (temp >= 45) & (temp <= 65)
summer_days = np.isin(month, [5, 6, 7, 8])
solar_like = solar_evidence
panels2 = [
    ('EV Level 2 flagged: mild days (45-65 F)', [(flag['ev_level2'], mild_night_days, 'Level 2 flagged', '#0072B2', '-'),
                                                 (ev_charging == 'none detected', mild_night_days, 'No charging detected', '#8c8c8c', '-')]),
    ('EV Level 1 flagged: mild days (45-65 F)', [(ev_charging == 'Level 1 likely', mild_night_days, 'Level 1 flagged', '#0072B2', '-'),
                                                 (ev_charging == 'none detected', mild_night_days, 'No charging detected', '#8c8c8c', '-')]),
    ('Hot tub flagged: mild days (45-65 F)', [(flag['hot_tub'], mild_night_days, 'Hot tub flagged', '#009E73', '-'),
                                              (~flag['hot_tub'], mild_night_days, 'Not flagged', '#8c8c8c', '-')]),
    ('Solar homes, May-Aug: battery flagged vs not', [(flag['solar_battery'], summer_days, 'Solar + battery flagged', '#E69F00', '-'),
                                                      (solar_like & ~flag['solar_battery'], summer_days, 'Other solar homes', '#8c8c8c', '-')]),
]
fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
for ax, (title, spec) in zip(axes.flat, panels2):
    for sel, days, label, color, style in spec:
        prof, count_c = mean_profile(sel, days)
        ax.plot(hours, prof, color=color, linestyle=style, linewidth=2.2, label=f'{label} (n = {count_c:,})')
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(1, 25, 2))
    ax.set_xlim(1, 24)
    ax.set_ylim(bottom=0)
    ax.grid(color='#e5e5e5', linewidth=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
for ax in axes[1]:
    ax.set_xlabel('Hour ending')
for ax in axes[:, 0]:
    ax.set_ylabel('Average kWh per hour')
fig.suptitle('Average daily profile: EV charging, hot tubs, solar + battery', y=1.0)
fig.tight_layout()
fig.savefig(config.EDA_DIR / 'appliance_profiles_2.png', dpi=150)
print('saved', config.EDA_DIR / 'appliance_profiles_2.png')
