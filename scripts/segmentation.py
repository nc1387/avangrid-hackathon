"""Customer segmentation: behavioural clusters plus program-oriented segments.

1. Behavioural clusters: k-means (k = 8) on avangrid.segmentation.FEATURES (size, daily shape, heating and
   cooling response, seasonality, grid-peak coincidence, load factor, variability, solar). k was chosen for
   stability across random starts (adjusted Rand >= 0.98 up to k = 9, dropping sharply at 10) and
   interpretability; silhouette is low (~0.14) at every k, as usual for household load data, which varies
   continuously rather than in crisp groups. Customers with too little use for peer-relative analysis
   (188, average < 0.05 kWh/h) form a ninth group, "Minimal use / vacant".
2. Program segments (flags; a customer can be in several), defined from the appliance inventory and
   technology predictions:
     high_electrification   EV (listed, candidate list or Level 2 charging), primary electric heating,
                            or heat pump (listed or "heat pump likely")
     peak_contributor       top 10% by kWh used during the 100 highest grid hours
     seasonal               2+ months averaging under 30% of the customer's yearly average
     flexible_load          2+ controllable loads among central AC, EV charging, electric water heater,
                            pool, hot tub
     hp_retrofit_candidate  central AC and no electric heating (likely fuel heating with ductwork: a heat
                            pump can replace the AC and the furnace's heating)
     ev_adoption_candidate  solar home (PV list or PV model) with no EV signal: early-adopter households
     solar                  PV list or PV model score >= 0.5
Outputs: avangrid.db table customer_segments, onedrive-export/Data/eda/customer_segments.csv,
segment_summary.csv, segment_profiles.png, segment_peak_share.png.
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
from sklearn.cluster import KMeans  # noqa: E402
from avangrid import appliances, cache, config, db, features, grids, segmentation as seg, signatures  # noqa: E402

K_CLUSTERS = 8
hd = grids.load_hourly(verbose=False)
D, _ = features.load_daily_signals(hd, verbose=False)
F = appliances.appliance_features(hd.kwh, hd.temp_hourly, hd.dates, hd.weekend)
sig = signatures.load_signatures(hd, D, verbose=False)
X = seg.segmentation_features(hd.kwh, hd.dates, F, sig)
Z = seg.robust_standardize(X)
cache.save_results('segmentation_k_choice', seg.choose_k(Z))
labels = KMeans(K_CLUSTERS, n_init=20, random_state=0).fit_predict(Z)
names = seg.name_clusters(X, labels)
cluster_name = np.array([names[c] for c in labels])

# ---------------------------------------------------------------- program segments
cust = pl.DataFrame({'CUSTOMER_NUMBER': hd.customer_numbers})
inv = cust.join(db.sql('SELECT * FROM appliance_inventory'), on='CUSTOMER_NUMBER', how='left')
tp = cust.join(db.sql('SELECT * FROM technology_predictions'), on='CUSTOMER_NUMBER', how='left')
cands = pl.read_csv(config.EDA_DIR / 'technology_candidates.csv')
in_cand = lambda t: np.isin(hd.customer_numbers, cands.filter(pl.col('technology') == t)['CUSTOMER_NUMBER'].to_numpy())

ev = tp['ev_listed'].to_numpy() | in_cand('EV') | (inv['ev_charging_type'] == 'Level 2 likely').to_numpy()
hp = tp['hp_listed'].to_numpy() | (inv['heating_type'] == 'heat pump likely').to_numpy()
solar = tp['pv_listed'].to_numpy() | (tp['pv_score'].fill_null(0).to_numpy() >= 0.5)
elec_heat_primary = inv['elec_heat_primary_flag'].to_numpy()
central_ac = (inv['ac_type'] == 'central likely').to_numpy()
ev_any = ev | (inv['ev_charging_type'] != 'none detected').to_numpy()
flex_count = (central_ac.astype(int) + ev_any + inv['water_heater_flag'].to_numpy() + inv['pool_flag'].to_numpy()
              + inv['hot_tub_flag'].to_numpy())
grid_peak_kwh = sig['grid_peak_kwh'].to_numpy()

segments = cust.with_columns(
    pl.Series('cluster', labels), pl.Series('segment', cluster_name),
    pl.Series('high_electrification', ev | elec_heat_primary | hp),
    pl.Series('peak_contributor', grid_peak_kwh >= np.nanpercentile(grid_peak_kwh, 90)),
    pl.Series('seasonal', X[:, seg.FEATURES.index('empty_months')] >= 2),
    pl.Series('flexible_load', flex_count >= 2),
    pl.Series('hp_retrofit_candidate', central_ac & ~inv['elec_heat_flag'].to_numpy()),
    pl.Series('ev_adoption_candidate', solar & ~ev_any),
    pl.Series('solar', solar),
    pl.Series('flexible_load_count', flex_count),
    pl.Series('grid_peak_kwh', grid_peak_kwh),
    *[pl.Series(f, X[:, i]) for i, f in enumerate(seg.FEATURES)],
)

# The 188 near-zero customers (in ami_analysis, not in the peer groups / hourly arrays)
low = db.sql('''SELECT CUSTOMER_NUMBER FROM ami_analysis GROUP BY 1
                HAVING CUSTOMER_NUMBER NOT IN (SELECT CUSTOMER_NUMBER FROM customer_peer_groups)''')
low = low.with_columns(pl.lit(K_CLUSTERS).cast(segments['cluster'].dtype).alias('cluster'),
                       pl.lit('Minimal use / vacant').alias('segment'))
all_segments = pl.concat([segments, low], how='diagonal').sort('CUSTOMER_NUMBER')
flag_cols = ['high_electrification', 'peak_contributor', 'seasonal', 'flexible_load', 'hp_retrofit_candidate',
             'ev_adoption_candidate', 'solar']
all_segments = all_segments.with_columns([pl.col(c).fill_null(False) for c in flag_cols])
with db.connect(read_only=False) as con:
    con.register('seg_df', all_segments)
    con.execute('CREATE OR REPLACE TABLE customer_segments AS SELECT * FROM seg_df')
all_segments.write_csv(config.EDA_DIR / 'customer_segments.csv')
cache.save_results('customer_segments', all_segments)

# ---------------------------------------------------------------- summary
K = hd.kwh
system = np.zeros((K.shape[1], 24))
for s in range(0, K.shape[0], 3000):
    system += np.nansum(np.asarray(K[s:s + 3000], dtype=np.float64), 0)
flat = system.ravel()
month = np.repeat(hd.dates.dt.month().to_numpy(), 24)
top = lambda mask, k: np.flatnonzero(mask)[np.argsort(-flat[mask])[:k]]
peak_hours = {'all': top(np.ones_like(flat, bool), 100), 'winter': top(np.isin(month, [12, 1, 2]), 50),
              'summer': top(np.isin(month, [6, 7, 8]), 50)}
energy = np.zeros(K.shape[0]); peak_kwh = {k: np.zeros(K.shape[0]) for k in peak_hours}
for s in range(0, K.shape[0], 3000):
    kf = np.asarray(K[s:s + 3000], dtype=np.float64).reshape(-1, K.shape[1] * 24)
    energy[s:s + 3000] = np.nanmean(kf, 1) * kf.shape[1]
    for k, idx in peak_hours.items():
        peak_kwh[k][s:s + 3000] = np.nansum(kf[:, idx], 1)

summ = segments.with_columns(pl.Series('energy', energy), *[pl.Series(f'peak_{k}', v) for k, v in peak_kwh.items()],
                             pl.Series('central_ac', central_ac), pl.Series('ev', ev_any), pl.Series('elec_heat', inv['elec_heat_flag'].to_numpy()))
tot = {c: float(summ[c].sum()) for c in ['energy', 'peak_all', 'peak_winter', 'peak_summer']}
summary = (summ.group_by('segment').agg(
    pl.len().alias('customers'),
    (pl.len() / summ.height).alias('share_customers'),
    (pl.col('energy').sum() / tot['energy']).alias('share_energy'),
    (pl.col('peak_all').sum() / tot['peak_all']).alias('share_grid_peak'),
    (pl.col('peak_winter').sum() / tot['peak_winter']).alias('share_winter_peak'),
    (pl.col('peak_summer').sum() / tot['peak_summer']).alias('share_summer_peak'),
    (pl.col('log_mean_kwh').exp().median()).alias('median_kw'),
    pl.col('load_factor').median(), pl.col('grid_peak_ratio').median(),
    pl.col('central_ac').mean().alias('central_ac'), pl.col('elec_heat').mean(), pl.col('ev').mean().alias('ev_signal'),
    pl.col('solar').mean(), *[pl.col(c).mean() for c in ['high_electrification', 'peak_contributor', 'flexible_load',
                                                         'hp_retrofit_candidate']],
).sort('share_grid_peak', descending=True))
summary.write_csv(config.EDA_DIR / 'segment_summary.csv')
cache.save_results('segment_summary', summary)
overlays = pl.DataFrame([{'segment': c, 'customers': int(all_segments[c].sum()),
                          'share_customers': float(all_segments[c].mean())} for c in flag_cols])
cache.save_results('segment_overlays', overlays)
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, tbl_hide_dataframe_shape=True, float_precision=3, fmt_str_lengths=35):
    print(summary)
    print(overlays)
    print('peak hours (top 100):', sorted({(int(i) // 24, int(i) % 24 + 1) for i in peak_hours['all']})[:5], '...')

# ---------------------------------------------------------------- charts
order = summary['segment'].to_list()
COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00', '#56B4E9', '#F0E442', '#8c8c8c']
color = {s: COLORS[i % len(COLORS)] for i, s in enumerate(order)}
month_d = hd.dates.dt.month().to_numpy()
seasons = {'Winter (Dec-Feb)': np.isin(month_d, [12, 1, 2]), 'Summer (Jun-Aug)': np.isin(month_d, [6, 7, 8])}
fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharex=True)
for ax, s in zip(axes.flat, order):
    idx = np.flatnonzero(cluster_name == s)
    for (lab, days), style in zip(seasons.items(), ['-', '--']):
        prof = np.zeros(24)
        for j in range(0, len(idx), 3000):
            prof += np.nansum(np.nanmean(np.asarray(K[idx[j:j + 3000]])[:, days], 1), 0)
        ax.plot(range(1, 25), prof / len(idx), color=color[s], linestyle=style, linewidth=2.2, label=lab)
    r = summary.filter(pl.col('segment') == s).row(0, named=True)
    ax.set_title(f"{s}\n{r['customers']:,} homes · {r['share_grid_peak']:.0%} of grid peak", fontsize=9)
    ax.set_xticks(range(1, 25, 4))
    ax.set_ylim(bottom=0)
    ax.grid(color='#e5e5e5', linewidth=0.7)
    ax.spines[['top', 'right']].set_visible(False)
for ax in axes[1]:
    ax.set_xlabel('Hour ending')
for ax in axes[:, 0]:
    ax.set_ylabel('Average kWh per hour')
axes[0, 0].legend(frameon=False, fontsize=8)
fig.suptitle('Average daily profile of each customer segment, winter vs summer', y=1.0)
fig.tight_layout()
fig.savefig(config.EDA_DIR / 'segment_profiles.png', dpi=150)

fig, ax = plt.subplots(figsize=(10, 5))
y = np.arange(len(order))
ax.barh(y - 0.2, summary['share_customers'], height=0.38, color='#bbbbbb', label='Share of customers')
ax.barh(y + 0.2, summary['share_grid_peak'], height=0.38, color='#0072B2', label='Share of use in the 100 highest grid hours')
ax.set_yticks(y, order)
ax.invert_yaxis()
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
for i, (a, b) in enumerate(zip(summary['share_customers'], summary['share_grid_peak'])):
    ax.text(b + 0.004, i + 0.2, f'{b:.0%}', va='center', fontsize=8)
ax.set_title('Which segments drive the grid peak?')
ax.grid(axis='x', color='#e5e5e5', linewidth=0.7)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False, loc='lower right')
fig.tight_layout()
fig.savefig(config.EDA_DIR / 'segment_peak_share.png', dpi=150)
print('saved charts')
