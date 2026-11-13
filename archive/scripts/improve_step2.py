"""Step 2: heat-pump physics features and the night-by-night (multiple-instance) EV model.

HP: best set from step 1, with and without avangrid.physics features (5-fold over the HP list).
EV: on the same 5 folds as step 1,
    - the best step-1 EV feature set (bagging PU),
    - the night model alone (customer score = mean of top-20 night probabilities),
    - an ensemble: average of the two scores' percentile ranks.
Final scores for every customer are saved for the best EV and HP models (listed customers out-of-fold).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import avangrid  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from avangrid import (appliances, cache, customer_features as cfm, ev_nights, features, grids, physics, pu,  # noqa: E402
                      signatures, weather)

T0 = time.time()


def log(msg):
    print(f'[{time.time() - T0:5.0f}s] {msg}', flush=True)


hd = grids.load_hourly(verbose=False)
D, _ = features.load_daily_signals(hd, verbose=False)
W = weather.load_weather_hourly()
labels = hd.labels()
n = hd.shape[0]

A = cfm.compute(hd.kwh, W['temp'], W['temp_quartile'], hd.dates)
F = appliances.appliance_features(hd.kwh, hd.temp_hourly, hd.dates, hd.weekend)
R_names = [k for k in F if k != 'mean_kwh']
R = np.column_stack([F[k] for k in R_names])
sig = signatures.load_signatures(hd, D, verbose=False)
P, _ = signatures.to_matrix(sig, ('Peak load',))
H_dict = physics.hp_physics_features(hd.kwh, W, hd.dates, F)
H = np.column_stack(list(H_dict.values()))
log('features ready')

step1 = cache.load_results('improve_step1_sets')
best = {t: step1.filter(pl.col('technology') == t).group_by('feature_set').agg(pl.col('auc').mean())
        .sort('auc', descending=True)['feature_set'][0] for t in ['EV', 'HP']}
log(f'best step-1 sets: {best}')
SIZE = {'mean_kwh', 'p95_kwh', 'max_kwh', 'summer_midday_kwh', 'summer_night_kwh', 'winter_midday_kwh',
        'heating_slope', 'cooling_slope', 'mild_day_kwh'}
R_SIZE = {'ac_slope', 'ac_added_kw', 'heat_slope', 'heat_kw_at_20f', 'heat_added_kw', 'wh_morning_kwh',
          'pool_day_floor_jump', 'ev_l1_excess_kwh', 'summer_morning_kwh'}
A_sf = A[:, [i for i, nm in enumerate(cfm.NAMES) if nm not in SIZE]]
R_sf = R[:, [i for i, nm in enumerate(R_names) if nm not in R_SIZE]]
build = {'A (current best)': A, 'A + R': np.hstack([A, R]), 'R only': R, 'A + R, size-free': np.hstack([A_sf, R_sf]),
         'A + P': np.hstack([A, P]), 'A + P + R': np.hstack([A, P, R]), 'A + P + R, size-free': np.hstack([A_sf, P, R_sf])}
X_ev, X_hp = build[best['EV']], build[best['HP']]

# ---------------------------------------------------------------- HP: physics features
log('HP physics comparison')
hp_sets = {f'{best["HP"]}': X_hp, f'{best["HP"]} + physics': np.hstack([X_hp, H])}
hp_res = pu.evaluate_feature_sets(hp_sets, labels['HP'], scheme='kfold', n_splits=5, verbose=False)
cache.save_results('improve_step2_hp', hp_res)

# ---------------------------------------------------------------- EV: night model
log('night features')
N = ev_nights.load_night_features(hd, verbose=False)
log('EV comparison')
rows = []
for split, lab in pu.hidden_label_splits(labels['EV'], 'kfold', n_splits=5):
    hidden = labels['EV'] & ~lab
    ev_rows = hidden | ~labels['EV']
    y = hidden[ev_rows]
    s_bag, _ = pu.bagging_pu(X_ev, lab, seed=split, n_models=50)
    models = ev_nights.fit_night_models(N, lab, seed=split)
    s_mil = ev_nights.customer_scores(models, N)
    rank = lambda s: pl.Series(s[ev_rows]).rank('average').to_numpy() / ev_rows.sum()
    s_ens = (rank(s_bag) + rank(s_mil)) / 2
    for name, s in [(best['EV'], s_bag[ev_rows]), ('night model', s_mil[ev_rows]), ('ensemble', s_ens)]:
        top = np.argsort(-s)[:y.sum()]
        rows.append({'split': split, 'feature_set': name, 'auc': roc_auc_score(y, s), 'precision_top_k': float(y[top].mean())})
    log(f'  split {split} done')
ev_res = pl.DataFrame(rows)
cache.save_results('improve_step2_ev', ev_res)

summ = lambda r: r.group_by('feature_set', maintain_order=True).agg(
    pl.col('auc').mean().alias('auc'), pl.col('auc').std().alias('sd'), pl.col('precision_top_k').mean().alias('prec_top_k'))
with pl.Config(tbl_hide_dataframe_shape=True, float_precision=3, fmt_str_lengths=60):
    print('HP'); print(summ(hp_res))
    print('EV'); print(summ(ev_res))

# ---------------------------------------------------------------- final scores for every customer
log('final scores')
hp_best_name = summ(hp_res).sort('auc', descending=True)['feature_set'][0]
X_hp_final = hp_sets[hp_best_name]
ev_best_name = summ(ev_res).sort('auc', descending=True)['feature_set'][0]


def final_bagging(X, is_pos, **kw):
    final, _ = pu.bagging_pu(X, is_pos, seed=0, **kw)
    oof = np.full(n, np.nan)
    for split, lab in pu.hidden_label_splits(is_pos, 'kfold', n_splits=5):
        s, _ = pu.bagging_pu(X, lab, seed=split, **kw)
        oof[is_pos & ~lab] = s[is_pos & ~lab]
    return np.where(is_pos, oof, final)


def final_mil(is_pos):
    final = ev_nights.customer_scores(ev_nights.fit_night_models(N, is_pos, seed=0), N)
    oof = np.full(n, np.nan)
    for split, lab in pu.hidden_label_splits(is_pos, 'kfold', n_splits=5):
        s = ev_nights.customer_scores(ev_nights.fit_night_models(N, lab, seed=split), N)
        oof[is_pos & ~lab] = s[is_pos & ~lab]
    return np.where(is_pos, oof, final)


pct = lambda s: pl.Series(s).rank('average').to_numpy() / n
hp_score = final_bagging(X_hp_final, labels['HP'])
ev_bag = final_bagging(X_ev, labels['EV'], n_models=50)
ev_mil = final_mil(labels['EV'])
ev_score = {'ensemble': (pct(ev_bag) + pct(ev_mil)) / 2, 'night model': ev_mil}.get(ev_best_name, ev_bag)
out = hd.customers.select('CUSTOMER_NUMBER').with_columns(
    pl.Series('ev_score_v2', ev_score), pl.Series('ev_bagging_score', ev_bag), pl.Series('ev_night_score', ev_mil),
    pl.Series('hp_score_v2', hp_score), pl.Series('ev_listed', labels['EV']), pl.Series('hp_listed', labels['HP']))
cache.save_results('improve_final_scores', out)
cache.save_results('improve_final_choice', pl.DataFrame({'technology': ['EV', 'HP'], 'model': [ev_best_name, hp_best_name]}))
log(f'final models: EV = {ev_best_name}, HP = {hp_best_name}')
