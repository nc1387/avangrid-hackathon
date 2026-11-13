"""Night-by-night EV model (multiple-instance learning).

Each customer is treated as a set ("bag") of nights. A listed EV customer has *some* charging nights;
other customers have none (or rarely). A classifier learns what a charging *night* looks like from
nights of labeled EV customers vs nights of other customers, and a customer's score is the average of
their highest-scoring nights. Because it scores individual nights, it is much less affected by how big
the home is than a model trained on yearly summaries.

Night window: 7 pm to 7 am (hour_ending 20 to hour_ending 6 next day, 11 hours). Features per night,
relative to the customer's own 20th-percentile level for each window hour (their "quiet night"):
  max_excess, energy_excess (sum of positive excess), hours_ge3 / hours_ge1 (hours >= 3 / >= 1 kWh above),
  run_ge3 (longest run >= 3), run_l1 (longest run at 0.8-2.5, Level-1-like), start_ge25 (position of the
  first hour >= 2.5 above, -1 if none), sd (variation within the night), step_up / step_down (largest
  hour-to-hour rise / fall), night_kwh (total), temp (night average), weekday (night starts Mon-Fri),
  month.
"""

import warnings

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from . import cache, pu
from .grids import HourlyData, source_key

NIGHT_START, NIGHT_HOURS = 19, 11
FEATURES = ['max_excess', 'energy_excess', 'hours_ge3', 'hours_ge1', 'run_ge3', 'run_l1', 'start_ge25', 'sd',
            'step_up', 'step_down', 'night_kwh', 'temp', 'weekday', 'month']
VERSION = 'ev-nights-v1'


def _max_run(b):
    cur = np.zeros(b.shape[:-1]); best = np.zeros(b.shape[:-1])
    for j in range(b.shape[-1]):
        cur = (cur + 1) * b[..., j]
        best = np.maximum(best, cur)
    return best


def night_features(K: np.ndarray, temp_hourly: np.ndarray, dates: pl.Series, chunk: int = 2000) -> np.ndarray:
    """customers x nights x len(FEATURES) float32."""
    n, nd, _ = K.shape
    nn = nd - 1
    idx = (np.arange(nn) * 24 + NIGHT_START)[:, None] + np.arange(NIGHT_HOURS)[None, :]
    temp = temp_hourly.ravel()[idx].mean(1)
    weekday = (dates.dt.weekday().to_numpy()[:nn] <= 5).astype(float)
    month = dates.dt.month().to_numpy()[:nn].astype(float)
    out = np.full((n, nn, len(FEATURES)), np.nan, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for s0 in range(0, n, chunk):
            k = np.asarray(K[s0:s0 + chunk], dtype=np.float64)
            c = len(k)
            w = k.reshape(c, -1)[:, idx]                          # c x nights x 11
            ex = w - np.nanpercentile(w, 20, axis=1)[:, None, :]
            exz = np.nan_to_num(ex, nan=0.0)
            first = np.where((exz >= 2.5).any(2), np.argmax(exz >= 2.5, 2), -1)
            d = np.diff(np.nan_to_num(w, nan=0.0), axis=2)
            f = [np.nanmax(ex, 2), np.clip(exz, 0, None).sum(2), (exz >= 3).sum(2), (exz >= 1).sum(2),
                 _max_run(exz >= 3), _max_run((exz >= 0.8) & (exz <= 2.5)), first, np.nanstd(w, 2),
                 d.max(2), -d.min(2), np.nansum(w, 2),
                 np.broadcast_to(temp, (c, nn)), np.broadcast_to(weekday, (c, nn)), np.broadcast_to(month, (c, nn))]
            out[s0:s0 + c] = np.stack(f, axis=2).astype(np.float32)
    out[~np.isfinite(out)] = np.nan
    return out


def load_night_features(hd: HourlyData, rebuild: bool = False, verbose: bool = True) -> np.ndarray:
    """Night features for every customer in ``hd`` (cached, memory-mapped)."""
    src = source_key()
    key = None if src is None else {'version': VERSION, 'source': src}
    arrays, _, _ = cache.get_or_build(
        'ev_nights', key, lambda: ({'nights': night_features(hd.kwh, hd.temp_hourly, hd.dates)}, {}, {}),
        rebuild=rebuild, verbose=verbose)
    return arrays['nights']


def _fit(Xn, yn, wn, seed):
    with threadpool_limits(limits=1):
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed).fit(Xn, yn, sample_weight=wn)


def _predict_chunk(models, Xc):
    with threadpool_limits(limits=1):
        return np.mean([m.predict_proba(Xc)[:, 1] for m in models], axis=0)


def fit_night_models(N: np.ndarray, labeled: np.ndarray, seed: int, n_models: int = 5, n_neg_customers: int = 2000,
                     n_jobs: int = pu.DEFAULT_JOBS):
    """Train ``n_models`` night classifiers: nights of labeled customers (positive) vs nights of a random
    sample of unlabeled customers (negative), balanced by weight. Returns the list of models."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(labeled)
    unl = np.flatnonzero(~labeled)
    jobs = []
    for b in range(n_models):
        neg = rng.choice(unl, min(n_neg_customers, len(unl)), replace=False)
        Xp = N[pos].reshape(-1, N.shape[2]); Xq = N[neg].reshape(-1, N.shape[2])
        Xn = np.vstack([Xp, Xq])
        yn = np.r_[np.ones(len(Xp)), np.zeros(len(Xq))]
        wn = np.r_[np.full(len(Xp), len(Xq) / len(Xp)), np.ones(len(Xq))]
        jobs.append(delayed(_fit)(Xn, yn, wn, seed * 100 + b))
    return Parallel(n_jobs=min(n_jobs, n_models), backend='loky')(jobs)


def customer_scores(models, N: np.ndarray, top_k: int = 20, n_jobs: int = pu.DEFAULT_JOBS, chunk: int = 1500) -> np.ndarray:
    """Average of each customer's ``top_k`` highest night probabilities."""
    n, nn, f = N.shape
    parts = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_predict_chunk)(models, np.asarray(N[s:s + chunk]).reshape(-1, f)) for s in range(0, n, chunk))
    p = np.concatenate(parts).reshape(n, nn)
    p = np.sort(p, axis=1)[:, ::-1]
    return p[:, :top_k].mean(1)
