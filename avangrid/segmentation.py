"""Customer segmentation features and clustering.

Features (one row per customer), chosen to describe how and when a home uses electricity in ways that
matter to a utility (size, timing, weather sensitivity, seasonality, contribution to the grid peak):

  log_mean_kwh            size (log of average kWh per hour)
  night/morning/afternoon/evening_share
                          average use in each time of day / overall average (daily shape)
  evening_peak_ratio      average 5-9 pm use / overall average (evening-peak strength)
  heat_rel                extra kW on a 20 F day (day-to-day weather response) / average kW
  cool_rel                extra kW on hot afternoons / average kW
  log_winter_summer       log of Dec-Feb vs Jun-Aug average use
  empty_months            months averaging under 30% of the customer's yearly average (seasonal / vacant)
  grid_peak_ratio         use during the 100 highest grid hours / own average (peak coincidence)
  load_factor             average / highest hour (flat vs peaky)
  daily_cv                day-to-day variation of daily kWh
  solar_midday_zero_share share of Apr-Sep midday hours under 0.05 kWh (solar)

Clustering: k-means on robust-standardized features (each feature capped at its 1st/99th percentile,
centered on the median and divided by its spread), so no single feature or outlier dominates.
"""

import warnings

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

FEATURES = ['log_mean_kwh', 'night_share', 'morning_share', 'afternoon_share', 'evening_share', 'evening_peak_ratio',
            'heat_rel', 'cool_rel', 'log_winter_summer', 'empty_months', 'grid_peak_ratio', 'load_factor', 'daily_cv',
            'solar_midday_zero_share']


def segmentation_features(K: np.ndarray, dates: pl.Series, appliance_F: dict, peak_sig: pl.DataFrame,
                          chunk: int = 2000) -> np.ndarray:
    """customers x len(FEATURES) matrix. appliance_F: avangrid.appliances.appliance_features output;
    peak_sig: signatures table (needs grid_peak_ratio, load_factor_annual), same row order as K."""
    n = K.shape[0]
    month = dates.dt.month().to_numpy()
    winter, summer = np.isin(month, [12, 1, 2]), np.isin(month, [6, 7, 8])
    out = np.full((n, len(FEATURES)), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for s0 in range(0, n, chunk):
            k = np.asarray(K[s0:s0 + chunk], dtype=np.float64)
            c = len(k)
            mean = np.nanmean(k, (1, 2))
            prof = np.nanmean(k, 1)                                   # c x 24
            shares = [np.nanmean(prof[:, s], 1) / mean for s in (slice(0, 6), slice(6, 12), slice(12, 18), slice(18, 24))]
            evening_peak = np.nanmean(prof[:, 17:21], 1) / mean
            daily = np.nanmean(k, 2) * 24
            monthly = np.column_stack([np.nanmean(daily[:, month == m], 1) for m in range(1, 13)])
            empty = (monthly < 0.3 * np.nanmean(daily, 1, keepdims=True)).sum(1)
            cv = np.nanstd(daily, 1) / np.nanmean(daily, 1)
            ws = np.log(np.nanmean(daily[:, winter], 1) / np.nanmean(daily[:, summer], 1))
            o = slice(s0, s0 + c)
            out[o, 0] = np.log(np.maximum(mean, 1e-3))
            out[o, 1:5] = np.column_stack(shares)
            out[o, 5] = evening_peak
            out[o, 8], out[o, 9], out[o, 12] = ws, empty, cv
    mean_all = np.exp(out[:, 0])
    out[:, 6] = np.maximum(appliance_F['heat_kw_at_20f'], 0) / mean_all
    out[:, 7] = np.maximum(appliance_F['ac_added_kw'], 0) / mean_all
    out[:, 10] = peak_sig['grid_peak_ratio'].to_numpy()
    out[:, 11] = peak_sig['load_factor_annual'].to_numpy()
    out[:, 13] = appliance_F['solar_midday_zero_share']
    out[~np.isfinite(out)] = np.nan
    return out


def robust_standardize(X: np.ndarray) -> np.ndarray:
    lo, hi = np.nanpercentile(X, 1, axis=0), np.nanpercentile(X, 99, axis=0)
    Xc = np.clip(X, lo, hi)
    med = np.nanmedian(Xc, axis=0)
    spread = np.nanpercentile(Xc, 75, axis=0) - np.nanpercentile(Xc, 25, axis=0)
    # Fall back to the standard deviation when the interquartile range is degenerate: zero, or (for mostly-zero
    # features such as solar_midday_zero_share) a sliver just above zero that would blow the feature up and let
    # it dominate k-means. For ordinary features the IQR is about 1.0-1.35 standard deviations.
    spread = np.where(spread > 0.25 * np.nanstd(Xc, axis=0), spread, np.nanstd(Xc, axis=0) + 1e-9)
    Z = (Xc - med) / spread
    return np.where(np.isnan(Z), 0.0, Z)


def choose_k(Z: np.ndarray, ks=range(4, 13), sample: int = 6000, seed: int = 0) -> pl.DataFrame:
    """Silhouette, Davies-Bouldin and seed-to-seed stability (adjusted Rand) for each k."""
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), min(sample, len(Z)), replace=False)
    rows = []
    for k in ks:
        labs = [KMeans(k, n_init=5, random_state=s).fit_predict(Z) for s in range(3)]
        ari = np.mean([adjusted_rand_score(labs[0], labs[i]) for i in (1, 2)])
        rows.append({'k': k, 'silhouette': float(silhouette_score(Z[idx], labs[0][idx])),
                     'davies_bouldin': float(davies_bouldin_score(Z, labs[0])), 'stability_ari': float(ari),
                     'smallest_cluster_share': float(np.bincount(labs[0]).min() / len(Z))})
    return pl.DataFrame(rows)


def name_clusters(X: np.ndarray, labels: np.ndarray) -> dict[int, str]:
    """Name clusters from their median features, greedily in a fixed order, so names don't depend on
    k-means' arbitrary cluster numbering. Assumes 8 clusters (as chosen in notebook 70)."""
    f = {name: i for i, name in enumerate(FEATURES)}
    med = {c: np.nanmedian(X[labels == c], axis=0) for c in np.unique(labels)}
    left = set(med)
    names = {}

    def pick(name, key, largest=True):
        c = (max if largest else min)(left, key=lambda c: key(med[c]))
        names[c] = name
        left.discard(c)

    pick('Solar homes', lambda m: m[f['solar_midday_zero_share']])
    pick('Seasonal / part-year vacant', lambda m: m[f['empty_months']] + m[f['log_winter_summer']])
    pick('Electric-heated, winter-peaking', lambda m: m[f['heat_rel']])
    pick('Summer peakers (heavy AC)', lambda m: m[f['cool_rel']])
    pick('Evening peakers', lambda m: m[f['evening_peak_ratio']])
    pick('Small, flat, low-AC', lambda m: m[f['log_mean_kwh']], largest=False)
    pick('Large, steady homes', lambda m: m[f['log_mean_kwh']] + 5 * m[f['load_factor']])
    for c in left:
        names[c] = 'Daytime-active, AC'
    return names


# ------------------------------------------------------------------------------------------------ mixed data
# k-prototypes (Huang 1998): k-means for the continuous features and k-modes for the categorical ones, in one
# objective. Distance from a customer to a cluster = squared Euclidean distance on the (standardized) continuous
# features + gamma x the number of categorical variables on which the customer differs from the cluster's mode.
# Means and modes are updated in turn, as in k-means, until no customer changes cluster.

def kproto_distances(X: np.ndarray, C: np.ndarray, centers: np.ndarray, modes: np.ndarray,
                     gamma: float) -> np.ndarray:
    """customers x clusters distance: squared Euclidean on X + gamma x categorical mismatches of C."""
    num = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(2)
    cat = (C[:, None, :] != modes[None, :, :]).sum(2)
    return num + gamma * cat


def _kproto_update(X, C, labels, k, n_levels):
    centers = np.vstack([X[labels == j].mean(0) for j in range(k)])
    modes = np.vstack([[np.bincount(C[labels == j, v], minlength=n_levels[v]).argmax() for v in range(C.shape[1])]
                       for j in range(k)])
    return centers, modes


def kprototypes(X: np.ndarray, C: np.ndarray, k: int, gamma: float, n_init: int = 10, max_iter: int = 100,
                seed: int = 0) -> dict:
    """Cluster customers on continuous features X (customers x p, already standardized) and categorical
    features C (customers x q, integer codes 0..levels-1). Returns the best of n_init runs (lowest cost):
    {'labels', 'centers', 'modes', 'cost', 'num_cost', 'cat_cost', 'n_iter'}.

    Each run starts from k customers chosen k-means++ style on the combined distance (each new seed drawn with
    probability proportional to its distance from the nearest seed so far). A cluster that empties is
    re-seeded with the customer farthest from its own cluster."""
    X, C = np.asarray(X, float), np.asarray(C, np.int64)
    n_levels = C.max(0) + 1
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        idx = [int(rng.integers(len(X)))]
        d = kproto_distances(X, C, X[idx], C[idx], gamma)[:, 0]
        for _ in range(1, k):
            idx.append(int(rng.choice(len(X), p=d / d.sum())))
            d = np.minimum(d, kproto_distances(X, C, X[idx[-1:]], C[idx[-1:]], gamma)[:, 0])
        centers, modes = X[idx].copy(), C[idx].copy()
        labels = np.full(len(X), -1)
        for it in range(max_iter):
            D = kproto_distances(X, C, centers, modes, gamma)
            new = D.argmin(1)
            for j in set(range(k)) - set(np.unique(new)):          # re-seed empty clusters
                far = int(D[np.arange(len(X)), new].argmax())
                new[far] = j
            if np.array_equal(new, labels):
                break
            labels = new
            centers, modes = _kproto_update(X, C, labels, k, n_levels)
        num = float(((X - centers[labels]) ** 2).sum())
        cat = float((C != modes[labels]).sum())
        cost = num + gamma * cat
        if best is None or cost < best['cost']:
            best = {'labels': labels, 'centers': centers, 'modes': modes, 'cost': cost, 'num_cost': num,
                    'cat_cost': gamma * cat, 'n_iter': it + 1}
    return best


def kproto_total_cost(X: np.ndarray, C: np.ndarray, gamma: float) -> float:
    """Cost with a single cluster (overall mean and mode): the baseline for 'share of cost explained'."""
    X, C = np.asarray(X, float), np.asarray(C, np.int64)
    modes = np.array([np.bincount(C[:, v]).argmax() for v in range(C.shape[1])])
    return float(((X - X.mean(0)) ** 2).sum() + gamma * (C != modes).sum())
