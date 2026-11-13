"""Positive-unlabeled (PU) learning, with models fitted in parallel processes.

``bagging_pu`` gives the same scores as the sequential version in notebooks 45 and 50 (part 1)
(same random samples and model seeds), but fits the models several at a time in separate
processes, each limited to one thread. On this machine that is several times faster.

Use ``n_jobs=1`` to run sequentially (for debugging, or inside another parallel loop).
"""

import os
import time

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from threadpoolctl import threadpool_limits

DEFAULT_JOBS = max(1, (os.cpu_count() or 2) - 1)  # leave one core free


def _fit_one(X, pos, sample, oob, X_extra, random_state, model_params):
    """Fit one model on positives + sampled unlabeled; return predictions for out-of-bag rows and X_extra."""
    with threadpool_limits(limits=1):
        idx = np.r_[pos, sample]
        y = np.r_[np.ones(len(pos)), np.zeros(len(sample))]
        model = HistGradientBoostingClassifier(random_state=random_state, **model_params).fit(X[idx], y)
        p_oob = model.predict_proba(X[oob])[:, 1]
        p_extra = None if X_extra is None else model.predict_proba(X_extra)[:, 1]
    return oob, p_oob, p_extra


def bagging_pu(X: np.ndarray, labeled: np.ndarray, seed: int, X_extra: np.ndarray | None = None,
               n_models: int = 20, neg_ratio: int = 10, n_jobs: int = DEFAULT_JOBS,
               model_params: dict | None = None):
    """Bagging PU scores.

    Each model is trained on all labeled positives plus a random sample of ``neg_ratio`` times as
    many unlabeled rows (treated as negatives). Every unlabeled row is scored by the models that
    didn't use it, and those scores are averaged. Labeled rows get NaN.

    X_extra: optional extra rows (e.g. simulated customers) scored by every model.
    Returns (scores, extra_scores) where extra_scores is None if X_extra is None.
    """
    model_params = {'max_iter': 200, 'learning_rate': 0.05, **(model_params or {})}
    labeled = np.asarray(labeled, bool)
    rng = np.random.default_rng(seed)
    pos, unl = np.where(labeled)[0], np.where(~labeled)[0]
    k = min(len(unl), neg_ratio * len(pos))
    # Draw samples in the same order as the sequential version so results are identical.
    samples = [rng.choice(unl, k, replace=False) for _ in range(n_models)]
    jobs = (delayed(_fit_one)(X, pos, s, np.setdiff1d(unl, s), X_extra, seed * 1000 + b, model_params)
            for b, s in enumerate(samples))
    results = Parallel(n_jobs=n_jobs, backend='loky')(jobs) if n_jobs != 1 else [j[0](*j[1], **j[2]) for j in jobs]

    total, count = np.zeros(len(labeled)), np.zeros(len(labeled))
    extra = None if X_extra is None else np.zeros(len(X_extra))
    for oob, p, pe in results:
        total[oob] += p
        count[oob] += 1
        if pe is not None:
            extra += pe / n_models
    return np.where(count > 0, total / np.maximum(count, 1), np.nan), extra


def _fit_model(X, pos, sample, random_state, model_params):
    with threadpool_limits(limits=1):
        idx = np.r_[pos, sample]
        y = np.r_[np.ones(len(pos)), np.zeros(len(sample))]
        return HistGradientBoostingClassifier(random_state=random_state, **model_params).fit(X[idx], y)


def fit_bagging_models(X: np.ndarray, labeled: np.ndarray, seed: int, n_models: int = 20, neg_ratio: int = 10,
                       n_jobs: int = DEFAULT_JOBS, model_params: dict | None = None):
    """Like ``bagging_pu`` but returns the fitted models, for reuse (e.g. permutation importance).

    Returns a list of (model, out-of-bag row indices). Same samples and seeds as ``bagging_pu``,
    so ``score_bagging_models(fit_bagging_models(...), X)`` equals ``bagging_pu(...)[0]``.
    """
    model_params = {'max_iter': 200, 'learning_rate': 0.05, **(model_params or {})}
    labeled = np.asarray(labeled, bool)
    rng = np.random.default_rng(seed)
    pos, unl = np.where(labeled)[0], np.where(~labeled)[0]
    k = min(len(unl), neg_ratio * len(pos))
    samples = [rng.choice(unl, k, replace=False) for _ in range(n_models)]
    jobs = [delayed(_fit_model)(X, pos, s, seed * 1000 + b, model_params) for b, s in enumerate(samples)]
    models = Parallel(n_jobs=n_jobs, backend='loky')(jobs) if n_jobs != 1 else [j[0](*j[1], **j[2]) for j in jobs]
    return [(m, np.setdiff1d(unl, s)) for m, s in zip(models, samples)]


def score_bagging_models(models, X: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    """Out-of-bag average score from ``fit_bagging_models`` (NaN for rows no model left out).

    rows: optional subset of row indices to score (others stay NaN), to save time.
    """
    n = len(X)
    total, count = np.zeros(n), np.zeros(n)
    keep = None if rows is None else np.zeros(n, bool)
    if keep is not None:
        keep[rows] = True
    with threadpool_limits(limits=1):
        for model, oob in models:
            r = oob if keep is None else oob[keep[oob]]
            if len(r):
                total[r] += model.predict_proba(X[r])[:, 1]
                count[r] += 1
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _permuted_aucs(models, X, rows, y, key, cols, n_repeats, seed):
    rng = np.random.default_rng(seed + hash(key) % 10_000)
    out = []
    for _ in range(n_repeats):
        Xp = X.copy()
        perm = rng.permutation(rows)
        Xp[np.ix_(rows, cols)] = X[np.ix_(perm, cols)]   # same shuffle for every column in a group
        out.append(roc_auc_score(y, score_bagging_models(models, Xp, rows)[rows]))
    return key, out


def permutation_importance(models, X: np.ndarray, rows: np.ndarray, y: np.ndarray, feature_names: list[str],
                           n_repeats: int = 3, seed: int = 0, n_jobs: int = DEFAULT_JOBS,
                           groups: dict[str, list[str]] | None = None) -> pl.DataFrame:
    """AUC drop when a feature (or group of features) is shuffled across the evaluated rows.

    models: from ``fit_bagging_models``. rows: evaluated row indices (hidden positives + negatives).
    y: 1 for hidden positives among ``rows``. A larger drop means the model relies more on it.
    groups: optional {group name: [feature names]}; each group is shuffled together, which credits
        overlapping features (e.g. several midday measures) jointly instead of splitting the credit.
    Features or groups are scored in parallel.
    """
    rows = np.asarray(rows)
    base = roc_auc_score(y, score_bagging_models(models, X, rows)[rows])
    index = {f: i for i, f in enumerate(feature_names)}
    units = ({g: [index[f] for f in fs if f in index] for g, fs in groups.items()} if groups
             else {f: [i] for f, i in index.items()})
    units = {k: v for k, v in units.items() if v}
    jobs = (delayed(_permuted_aucs)(models, X, rows, y, k, cols, n_repeats, seed) for k, cols in units.items())
    results = Parallel(n_jobs=n_jobs, backend='loky')(jobs)
    out = [{'feature': k, 'n_features': len(units[k]), 'auc_drop': base - float(np.mean(a)),
            'auc_drop_sd': float(np.std(a))} for k, a in results]
    return pl.DataFrame(out).with_columns(pl.lit(base).alias('base_auc')).sort('auc_drop', descending=True)


def elkan_noto(X: np.ndarray, labeled: np.ndarray, seed: int, n_jobs: int = 5):
    """Elkan-Noto probabilities from a labeled-vs-unlabeled logistic regression.

    Returns (probability of being positive, estimated c, raw labeled-vs-unlabeled score g).
    Divide g by a known c instead if you have one (the EV and HP lists are ~25% samples).
    """
    labeled = np.asarray(labeled, int)
    model = make_pipeline(
        SimpleImputer(strategy='median'),
        QuantileTransformer(output_distribution='normal', random_state=seed),
        LogisticRegression(max_iter=3000),
    )
    with threadpool_limits(limits=1):
        g = cross_val_predict(model, X, labeled, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                              method='predict_proba', n_jobs=n_jobs)[:, 1]
    c = g[labeled == 1].mean()
    return np.clip(g / c, 0, 1), c, g


def hidden_label_splits(is_positive: np.ndarray, scheme: str = 'hide', n_splits: int = 5,
                        labeled_share: float = 0.25, seed: int = 0):
    """Yield (split number, labeled mask) for hidden-label tests.

    'hide': keep a random ``labeled_share`` of positives labeled, hide the rest (used for PV,
        to mimic the ~25% EV and HP samples). Each split draws a new random sample.
    'kfold': hide one fold of the positives at a time, label the rest (used for EV and HP,
        where every listed customer is already a sample).
    """
    pos = np.where(is_positive)[0]
    if scheme == 'hide':
        for r in range(n_splits):
            rng = np.random.default_rng(seed + 100 + r)
            labeled = np.zeros(len(is_positive), bool)
            labeled[rng.choice(pos, round(labeled_share * len(pos)), replace=False)] = True
            yield r, labeled
    elif scheme == 'kfold':
        folds = np.array_split(np.random.default_rng(seed + 7).permutation(pos), n_splits)
        for r, fold in enumerate(folds):
            labeled = np.zeros(len(is_positive), bool)
            labeled[np.setdiff1d(pos, fold)] = True
            yield r, labeled
    else:
        raise ValueError(scheme)


def evaluate_feature_sets(feature_sets: dict[str, np.ndarray], is_positive: np.ndarray, scheme: str = 'hide',
                          n_splits: int = 5, subgroup: np.ndarray | None = None, verbose: bool = True,
                          **bagging_kwargs) -> pl.DataFrame:
    """Hidden-label comparison of feature sets with bagging PU, using identical splits for each set.

    Scores are evaluated on hidden positives vs. everyone not on the list. Metrics:
      auc: chance a hidden positive outranks a non-positive
      precision_top_k: share of the top-k that are hidden positives, k = number hidden
      auc_subgroup: AUC using only hidden positives inside ``subgroup`` (e.g. customers with a
          detected change) against all non-positives
    """
    rows = []
    t0 = time.time()
    for r, labeled in hidden_label_splits(is_positive, scheme, n_splits):
        hidden = is_positive & ~labeled
        evaluate = hidden | ~is_positive
        y = hidden[evaluate]
        for name, X in feature_sets.items():
            scores, _ = bagging_pu(X, labeled, seed=r, **bagging_kwargs)
            s = scores[evaluate]
            top = np.argsort(-s)[:y.sum()]
            row = {'split': r, 'feature_set': name, 'auc': roc_auc_score(y, s),
                   'precision_top_k': float(y[top].mean()), 'n_hidden': int(y.sum())}
            if subgroup is not None:
                sg = subgroup[evaluate]
                keep = ~y | sg
                row['n_hidden_subgroup'] = int((y & sg).sum())
                row['auc_subgroup'] = roc_auc_score(y[keep], s[keep]) if (y & sg).sum() >= 3 else None
            rows.append(row)
        if verbose:
            print(f'  split {r + 1}/{n_splits} done ({time.time() - t0:.0f}s)')
    return pl.DataFrame(rows)


def summarize(results: pl.DataFrame) -> pl.DataFrame:
    """Mean and SD of each metric per feature set."""
    metrics = [c for c in ['auc', 'precision_top_k', 'auc_subgroup'] if c in results.columns]
    return results.group_by('feature_set', maintain_order=True).agg(
        *[pl.col(m).mean().round(3).alias(f'{m}_mean') for m in metrics],
        *[pl.col(m).std().round(3).alias(f'{m}_sd') for m in metrics],
        pl.col('n_hidden').mean().alias('n_hidden'),
    )
