"""Step 1: add the appliance-rule features to the EV and HP models, and test whether the EV model
responds to EV charging or mostly to home size.

Feature sets (same hidden-label splits as before: EV 5-fold over the list, HP 5-fold):
  A           notebook-40 features (NumPy version, avangrid.customer_features)
  R           appliance-rule evidence features (avangrid.appliances)
  P           peak-load signatures (avangrid.signatures, 'Peak load')
  A_sizefree  A without absolute size measures (mean/p95/max kWh, absolute kWh levels and slopes)

Charging test: train each EV model on all listed EV customers, add simulated Level 2 (or Level 1)
charging to 500 random customers on no list, recompute their features, and compare their EV scores
before and after (as percentiles of all unlisted customers' scores).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import avangrid  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from avangrid import appliances, cache, customer_features as cfm, features, grids, pu, signatures, weather  # noqa: E402

T0 = time.time()


def log(msg):
    print(f'[{time.time() - T0:5.0f}s] {msg}', flush=True)


hd = grids.load_hourly(verbose=False)
D, _ = features.load_daily_signals(hd, verbose=False)
W = weather.load_weather_hourly()
labels = hd.labels()
not_listed = ~(labels['EV'] | labels['HP'] | labels['PV'])

A = cfm.compute(hd.kwh, W['temp'], W['temp_quartile'], hd.dates)
A_names = list(cfm.NAMES)
F = appliances.appliance_features(hd.kwh, hd.temp_hourly, hd.dates, hd.weekend)
R_names = [k for k in F if k != 'mean_kwh']
R = np.column_stack([F[k] for k in R_names])
sig = signatures.load_signatures(hd, D, verbose=False)
P, P_names = signatures.to_matrix(sig, ('Peak load',))

SIZE = {'mean_kwh', 'p95_kwh', 'max_kwh', 'summer_midday_kwh', 'summer_night_kwh', 'winter_midday_kwh',
        'heating_slope', 'cooling_slope', 'mild_day_kwh'}
keep = [i for i, n in enumerate(A_names) if n not in SIZE]
A_sf, A_sf_names = A[:, keep], [A_names[i] for i in keep]
R_SIZE = {'ac_slope', 'ac_added_kw', 'heat_slope', 'heat_kw_at_20f', 'heat_added_kw', 'wh_morning_kwh',
          'pool_day_floor_jump', 'ev_l1_excess_kwh', 'summer_morning_kwh'}
keep_r = [i for i, n in enumerate(R_names) if n not in R_SIZE]
R_sf, R_sf_names = R[:, keep_r], [R_names[i] for i in keep_r]

SETS = {
    'EV': {'A (current best)': A, 'A + R': np.hstack([A, R]), 'R only': R,
           'A + R, size-free': np.hstack([A_sf, R_sf])},
    'HP': {'A + P': np.hstack([A, P]), 'A + P + R': np.hstack([A, P, R]),
           'A + P + R, size-free': np.hstack([A_sf, P, R_sf])},
}
CFG = {'EV': {'n_models': 50}, 'HP': {}}
names_of = {'A (current best)': A_names, 'A + R': A_names + R_names, 'R only': R_names,
            'A + R, size-free': A_sf_names + R_sf_names}

log('Hidden-label comparison')
res = []
for tech, sets in SETS.items():
    r = pu.evaluate_feature_sets(sets, labels[tech], scheme='kfold', n_splits=5, verbose=False, **CFG[tech])
    res.append(r.with_columns(pl.lit(tech).alias('technology')))
    log(f'  {tech} done')
res = pl.concat(res)
cache.save_results('improve_step1_sets', res)

log('EV charging test')
rng = np.random.default_rng(3)
test = np.sort(rng.choice(np.flatnonzero(not_listed), 500, replace=False))
K_test = np.asarray(hd.kwh[test])


def ev_matrix(K, which):
    a = cfm.compute(K, W['temp'], W['temp_quartile'], hd.dates)
    f = appliances.appliance_features(K, hd.temp_hourly, hd.dates, hd.weekend)
    r = np.column_stack([f[k] for k in R_names])
    return {'A (current best)': a, 'A + R': np.hstack([a, r]), 'R only': r,
            'A + R, size-free': np.hstack([a[:, keep], r[:, keep_r]])}[which]


injected = {kind: appliances.inject(K_test, kind, hd.temp_hourly, hd.dates, np.random.default_rng(11))
            for kind in ['ev_level2', 'ev_level1']}
rows = []
for name, X in SETS['EV'].items():
    models = pu.fit_bagging_models(X, labels['EV'], seed=0, n_models=50)
    base_scores = pu.score_bagging_models(models, X)
    ref = np.sort(base_scores[np.isfinite(base_scores) & ~labels['EV']])
    pct = lambda s: np.searchsorted(ref, s) / len(ref)

    def all_models(Xe):
        return np.mean([m.predict_proba(Xe)[:, 1] for m, _ in models], axis=0)
    before = pct(all_models(ev_matrix(K_test, name)))
    for kind, Kk in injected.items():
        after = pct(all_models(ev_matrix(Kk, name)))
        rows.append({'feature_set': name, 'injected': kind,
                     'median_pct_before': float(np.median(before)), 'median_pct_after': float(np.median(after)),
                     'share_top5pct_before': float((before >= 0.95).mean()), 'share_top5pct_after': float((after >= 0.95).mean())})
    log(f'  {name} done')
inj = pl.DataFrame(rows)
cache.save_results('improve_step1_injection', inj)

with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, tbl_hide_dataframe_shape=True, float_precision=3):
    base = res.filter(pl.col('feature_set').is_in(['A (current best)', 'A + P'])).select('technology', 'split', pl.col('auc').alias('b'))
    print(res.join(base, on=['technology', 'split']).group_by('technology', 'feature_set', maintain_order=True).agg(
        pl.col('auc').mean().alias('auc'), pl.col('auc').std().alias('sd'), (pl.col('auc') - pl.col('b')).mean().alias('gain'),
        ((pl.col('auc') - pl.col('b')) > 0).sum().alias('splits_better'), pl.col('precision_top_k').mean().alias('prec_top_k')))
    print(inj)
log('done')
