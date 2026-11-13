"""Test the peak-load features against each technology's current best model; if a model improves,
recompute its importance and final predictions.

Based on predict_with_signatures.py:

Original docstring of the base script follows.

Predict EV, HP and PV customers with the avangrid package.

Stages (results saved with avangrid.cache.save_results so they can be reloaded without rerunning):
  1. Compare feature sets per technology, with and without mid-year change information.
  2. Tune bagging settings for each technology's best feature set.
  3. Permutation importance (single features and feature families) for the best model.
  4. Final model on all listed customers: scores for every customer, Elkan-Noto count estimates.

Run from the project folder:  python scripts/predict_with_peaks.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root, so `import avangrid` works

import avangrid  # noqa: E402  (sets the thread limit before scikit-learn loads)
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from avangrid import cache, config, db, features, grids, pu, signatures, stability  # noqa: E402

T0 = time.time()


def log(msg):
    print(f'[{time.time() - T0:5.0f}s] {msg}', flush=True)


# ---------------------------------------------------------------- data and features
hd = grids.load_hourly(verbose=False)
D, _ = features.load_daily_signals(hd, verbose=False)
pf = features.PeriodFeatures(D, hd.temp)
n, n_days = hd.shape[0], hd.shape[1]
labels = hd.labels()
not_listed = ~(labels['EV'] | labels['HP'] | labels['PV'])
min_weeks = int(np.ceil(db.sql('SELECT ANY_VALUE(stabilization_days) AS d FROM customer_stability')['d'].item() / 7))

A, A_names = features.customer_features(hd.customer_numbers)
log_mean = np.log(hd.customers['mean_kwh_real'].to_numpy())
B, B_names = pf.to_matrix(pf(0, n_days), {'log_mean_kwh': log_mean})
B_names = [f'peer_{k}' for k in B_names]

cp = stability.tech_change_points(D, hd.kwh, hd.temp, min_weeks, not_listed)
Z = np.column_stack([cp[t]['z'] for t in config.TECHS] + [cp[t]['flag'].astype(float) for t in config.TECHS])
Z_names = [f'{t.lower()}_change_z' for t in config.TECHS] + [f'{t.lower()}_change_flag' for t in config.TECHS]


def current_period_block(tech):
    """Current-period features since this technology's flagged change, the before/after differences, period length."""
    flag = cp[tech]['flag']
    change_day = np.where(flag, cp[tech]['week'] * 7, 0)
    cur, bef = pf(change_day, n_days), pf(0, np.maximum(change_day, 1))
    names = list(cur)
    X = np.column_stack([cur[k] for k in names] + [np.where(flag, cur[k] - bef[k], 0) for k in names]
                        + [(n_days - change_day) / 7])
    X[~np.isfinite(X)] = np.nan
    return X, [f'current_{k}' for k in names] + [f'change_in_{k}' for k in names] + ['current_period_weeks']


def build_sets(tech):
    C, C_names = current_period_block(tech)
    return {
        'A: notebook 40': (A, A_names),
        'A+B: + peer-relative': (np.hstack([A, B]), A_names + B_names),
        'A+B+Z: + change scores': (np.hstack([A, B, Z]), A_names + B_names + Z_names),
        'A+B+Z+C: + current period': (np.hstack([A, B, Z, C]), A_names + B_names + Z_names + C_names),
    }


SCHEME = {'PV': 'hide', 'EV': 'kfold', 'HP': 'kfold'}   # PV: keep 25% labeled; EV/HP: 5-fold over the listed


sig = signatures.load_signatures(hd, D, verbose=False)
S_GROUP = {'PV': 'Solar index', 'EV': 'Charging blocks', 'HP': 'Cold-response shape'}
S_all, S_all_names = signatures.to_matrix(sig)
meta = cache.load_results('predict_meta')
best_set = dict(zip(meta['technology'], meta['best_set']))
best_cfg = dict(zip(meta['technology'], meta['best_config']))
CONFIGS = {
    'default (10:1, 200 trees)': {},
    '5:1 sampling': {'neg_ratio': 5},
    '20:1 sampling': {'neg_ratio': 20},
    'regularized trees': {'model_params': {'max_leaf_nodes': 15, 'min_samples_leaf': 40, 'l2_regularization': 1.0}},
    '50 models': {'n_models': 50},
}




def family(name):
    if name in signatures.GROUPS['Peak load']:
        return 'Peak load (new)'
    if name in signatures.GROUPS['Solar index']:
        return 'Solar index (new)'
    if name in signatures.GROUPS['Charging blocks']:
        return 'Charging blocks (new)'
    if name in signatures.GROUPS['Cold-response shape']:
        return 'Cold-response shape (new)'
    base = name.removeprefix('peer_').removeprefix('current_').removeprefix('change_in_')
    if name.endswith('_change_z') or name.endswith('_change_flag') or name == 'current_period_weeks':
        return 'Mid-year change scores'
    if name.startswith('change_in_'):
        return 'Change in features since flagged change'
    if name.startswith('current_'):
        return 'Current-period features'
    if any(s in base for s in ['midday', 'zero', 'dip', 'summer_night', 'midday_summer_winter']):
        return 'Solar / midday signature'
    if any(s in base for s in ['high_', 'session', 'sharp_rise']):
        return 'EV / high-power signature'
    if any(s in base for s in ['heat', 'cold', 'winter', 'cool']):
        return 'Heating & cooling response'
    if any(s in base for s in ['mean_kwh', 'p95', 'max_kwh', 'mild_day', 'log_mean', 'e_level']):
        return 'Size / overall level'
    return 'Daily shape (time of day)'




fmeta = cache.load_results('predict_final_meta')
final_set = dict(zip(fmeta['technology'], fmeta['feature_set']))
P_X, P_names = signatures.to_matrix(sig, ('Peak load',))


def current_best(tech):
    """Rebuild each technology's current best feature matrix from the saved run metadata."""
    X, names = build_sets(tech)[best_set[tech]]
    name = final_set[tech]
    if 'all signatures' in name:
        S, Sn = signatures.to_matrix(sig)
        X, names = np.hstack([X, S]), names + Sn
    elif not name.startswith('previous best ('):
        S, Sn = signatures.to_matrix(sig, (name.split(' + ')[-1],))
        X, names = np.hstack([X, S]), names + Sn
    return X, names


log('Stage 8: testing peak-load features')
stage8, cand = [], {}
for tech in config.TECHS:
    bX, bn = current_best(tech)
    cand[tech] = {
        'current best': (bX, bn),
        'current best + peak load': (np.hstack([bX, P_X]), bn + P_names),
        'A + peak load': (np.hstack([A, P_X]), A_names + P_names),
    }
    res = pu.evaluate_feature_sets({k: v[0] for k, v in cand[tech].items()}, labels[tech], scheme=SCHEME[tech],
                                   n_splits=5, verbose=False, **CONFIGS[best_cfg[tech]])
    stage8.append(res.with_columns(pl.lit(tech).alias('technology')))
    log(f'  {tech} done')
stage8 = pl.concat(stage8)
cache.save_results('predict_stage8_peaks', stage8)
base = stage8.filter(pl.col('feature_set') == 'current best').select('technology', 'split', pl.col('auc').alias('base'))
gain = (stage8.join(base, on=['technology', 'split']).with_columns((pl.col('auc') - pl.col('base')).alias('gain'))
        .group_by('technology', 'feature_set').agg(pl.col('auc').mean().alias('auc'), pl.col('gain').mean().alias('gain'),
                                                   (pl.col('gain') > 0).sum().alias('better')))
# Switch only when a new set wins on average AND in at least 4 of 5 splits (a consistent gain, not noise).
winner = {}
for tech in config.TECHS:
    g = gain.filter((pl.col('technology') == tech) & (pl.col('feature_set') != 'current best')).sort('auc', descending=True)
    top = g.row(0, named=True)
    winner[tech] = top['feature_set'] if (top['gain'] > 0 and top['better'] >= 4) else 'current best'
log(f'Winners: {winner}')
cache.save_results('predict_stage8_winners', pl.DataFrame({'technology': list(winner), 'feature_set': list(winner.values())}))

log('Stage 9: importance for the models with peak features')
importance = []
for tech in config.TECHS:
    key = winner[tech] if winner[tech] != 'current best' else 'current best + peak load'
    X, names = cand[tech][key]
    kw = CONFIGS[best_cfg[tech]]
    fam = {}
    for nm in names:
        fam.setdefault(family(nm), []).append(nm)
    for split, labeled in pu.hidden_label_splits(labels[tech], SCHEME[tech], n_splits=3):
        models = pu.fit_bagging_models(X, labeled, seed=split, **kw)
        hidden = labels[tech] & ~labeled
        rows = np.flatnonzero(hidden | ~labels[tech])
        y = hidden[rows]
        for kind, groups in [('feature', None), ('family', fam)]:
            imp = pu.permutation_importance(models, X, rows, y, names, n_repeats=2, seed=split, groups=groups)
            importance.append(imp.with_columns(pl.lit(tech).alias('technology'), pl.lit(kind).alias('kind'),
                                               pl.lit(split).alias('split'), pl.lit(key).alias('feature_set')))
    log(f'  {tech} done')
cache.save_results('predict_stage9_importance', pl.concat(importance, how='diagonal'))

log('Stage 10: final models for technologies whose winner changed')
scores = cache.load_results('predict_final_scores')
counts = cache.load_results('predict_final_counts')
for tech in config.TECHS:
    if winner[tech] == 'current best':
        continue
    X, names = cand[tech][winner[tech]]
    kw = dict(CONFIGS[best_cfg[tech]])
    kw['n_models'] = max(kw.get('n_models', 20), 50)
    is_pos = labels[tech]
    final, _ = pu.bagging_pu(X, is_pos, seed=0, **kw)
    oof = np.full(n, np.nan)
    for split, lab in pu.hidden_label_splits(is_pos, 'kfold', n_splits=5):
        s, _ = pu.bagging_pu(X, lab, seed=split, **kw)
        oof[is_pos & ~lab] = s[is_pos & ~lab]
    score = np.where(is_pos, oof, final)
    t = tech.lower()
    scores = scores.with_columns(pl.Series(f'{t}_score', score),
                                 pl.Series(f'{t}_percentile', pl.Series(score).rank('average').to_numpy() / n))
    p, c_est, g = pu.elkan_noto(X, is_pos, seed=0)
    row = pl.DataFrame([{
        'technology': tech, 'listed_in_analysis': int(is_pos.sum()), 'estimated_c': float(c_est),
        'est_total_estimated_c': float(p.sum()),
        'est_total_known_c_0.25': float(np.clip(g / 0.25, 0, 1).sum()) if tech in ('EV', 'HP') else None,
        'unlisted_score_ge_0.5': int(((final >= 0.5) & ~is_pos).sum()),
        'unlisted_score_ge_0.7': int(((final >= 0.7) & ~is_pos).sum())}], schema=counts.schema)
    counts = pl.concat([counts.filter(pl.col('technology') != tech), row])
    log(f'  {tech} done')
cache.save_results('predict_final_scores_v2', scores)
cache.save_results('predict_final_counts_v2', counts.sort('technology'))
log('All stages done')
