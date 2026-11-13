"""Scree plot for the k-means customer segmentation, as variance accounted for (VAF), on training and test data.

VAF = 1 - within-cluster sum of squares / total sum of squares (= between-cluster SS / total SS): the share of
the spread between customers that the clusters explain, like R-squared. It rises with k, so the "elbow" is
where it stops rising quickly.

Overfitting check, 5-fold cross-validation: for each fold, the robust standardization (1st/99th-percentile
caps, median, interquartile range) and k-means are fitted on the other four folds only. The held-out
customers are then standardized with those training values and assigned to the nearest training centroid.
Test VAF = 1 - their within-cluster SS / their total SS around their own mean. If test VAF tracks training
VAF, the clusters describe the population rather than the particular customers they were fitted on.

Also plotted: seed-to-seed stability (adjusted Rand) from results/segmentation_k_choice, which is what k = 8
was chosen on.

Inputs: the per-customer segmentation features in avangrid.db table customer_segments (run
scripts/segmentation.py first). Outputs: onedrive-export/Data/eda/segmentation_scree.png and
results/segmentation_vaf.parquet. Takes about half a minute.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import avangrid  # noqa: E402,F401
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from avangrid import cache, config, db, segmentation as seg  # noqa: E402

K_CHOSEN = 8
KS = list(range(1, 16))
N_FOLDS = 5


def fit_standardizer(X):
    """Training-set values for avangrid.segmentation.robust_standardize, so test rows can reuse them."""
    lo, hi = np.nanpercentile(X, 1, axis=0), np.nanpercentile(X, 99, axis=0)
    Xc = np.clip(X, lo, hi)
    med = np.nanmedian(Xc, axis=0)
    spread = np.nanpercentile(Xc, 75, axis=0) - np.nanpercentile(Xc, 25, axis=0)
    spread = np.where(spread > 0, spread, np.nanstd(Xc, axis=0) + 1e-9)
    return lo, hi, med, spread


def standardize(X, params):
    lo, hi, med, spread = params
    Z = (np.clip(X, lo, hi) - med) / spread
    return np.where(np.isnan(Z), 0.0, Z)


def vaf(Z, centers):
    """1 - within-cluster SS / total SS, with each row assigned to its nearest center."""
    d2 = ((Z[:, None, :] - centers[None, :, :]) ** 2).sum(2)
    within = d2.min(1).sum()
    total = ((Z - Z.mean(0)) ** 2).sum()
    return 1 - within / total


# The 188 minimal-use customers (cluster == K_CHOSEN) were added after clustering; leave them out.
s = db.sql('SELECT * FROM customer_segments ORDER BY CUSTOMER_NUMBER').filter(pl.col('cluster') < K_CHOSEN)
X = s.select(seg.FEATURES).to_numpy().astype(float)

rows = []
for fold, (tr, te) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=0).split(X)):
    params = fit_standardizer(X[tr])
    Ztr, Zte = standardize(X[tr], params), standardize(X[te], params)
    for k in KS:
        km = KMeans(k, n_init=20, random_state=0).fit(Ztr)
        rows.append({'fold': fold, 'k': k, 'vaf_train': vaf(Ztr, km.cluster_centers_),
                     'vaf_test': vaf(Zte, km.cluster_centers_)})
folds = pl.DataFrame(rows)
res = (folds.group_by('k').agg(
    *[pl.col(c).mean().alias(c) for c in ['vaf_train', 'vaf_test']],
    *[pl.col(c).min().alias(f'{c}_min') for c in ['vaf_train', 'vaf_test']],
    *[pl.col(c).max().alias(f'{c}_max') for c in ['vaf_train', 'vaf_test']],
).sort('k').with_columns((pl.col('vaf_train') - pl.col('vaf_test')).alias('train_minus_test')))
cache.save_results('segmentation_vaf', res)
kc = cache.load_results('segmentation_k_choice')

# Reference palette (light). Slots 1 and 2 of the categorical order; text and grid in neutral tokens.
TRAIN, TEST = '#2a78d6', '#eb6834'
TEXT, TEXT2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e6e5e1', '#fcfcfb'
plt.rcParams.update({'font.size': 10, 'text.color': TEXT, 'axes.labelcolor': TEXT2, 'xtick.color': TEXT2,
                     'ytick.color': TEXT2, 'axes.edgecolor': GRID, 'figure.facecolor': SURFACE,
                     'axes.facecolor': SURFACE})
dot = dict(linewidth=2, marker='o', markersize=6, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
ks = res['k'].to_numpy()
i8 = KS.index(K_CHOSEN)

# The two curves nearly coincide, so training is a wide solid line and held-out a dashed line with hollow markers
# on top: both stay visible where they overlap. (The fold-to-fold range is under 1 point, too thin to draw.)
ax1.plot(ks, res['vaf_train'], color=TRAIN, linewidth=4, marker='o', markersize=8, markerfacecolor=TRAIN,
         markeredgecolor=SURFACE, markeredgewidth=2, zorder=3, label='Training folds (mean of 5)')
ax1.plot(ks, res['vaf_test'], color=TEST, linewidth=1.6, linestyle=(0, (4, 3)), marker='o', markersize=5,
         markerfacecolor=SURFACE, markeredgecolor=TEST, markeredgewidth=1.6, zorder=4,
         label='Held-out fold (mean of 5)')
tr8, te8 = res['vaf_train'][i8], res['vaf_test'][i8]
ax1.plot(K_CHOSEN, tr8, marker='o', markersize=11, markerfacecolor='none', markeredgecolor=TEXT,
         markeredgewidth=1.5, zorder=4)
ax1.annotate(f'k = {K_CHOSEN} (chosen)\ntraining {tr8:.0%}, held-out {te8:.0%}', xy=(K_CHOSEN, tr8),
             xytext=(K_CHOSEN + 0.8, 0.28), fontsize=9, color=TEXT,
             arrowprops=dict(arrowstyle='-', color=TEXT2, linewidth=1))
ax1.set_title('Variance accounted for (VAF), 5-fold cross-validation', loc='left', fontsize=11, color=TEXT)
ax1.set_xlabel('Number of clusters (k)')
ax1.set_ylabel('VAF = 1 - within-cluster SS / total SS')
ax1.set_ylim(0, 0.7)
ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax1.legend(frameon=False, loc='upper left', fontsize=9)

ax2.plot(kc['k'], kc['stability_ari'], color=TRAIN, markerfacecolor=TRAIN, **dot)
a8 = kc.filter(pl.col('k') == K_CHOSEN)['stability_ari'][0]
ax2.plot(K_CHOSEN, a8, marker='o', markersize=11, markerfacecolor='none', markeredgecolor=TEXT,
         markeredgewidth=1.5, zorder=4)
ax2.annotate(f'k = {K_CHOSEN}: {a8:.2f}', xy=(K_CHOSEN, a8), xytext=(K_CHOSEN - 3.2, 0.7),
             fontsize=9, color=TEXT, arrowprops=dict(arrowstyle='-', color=TEXT2, linewidth=1))
ax2.annotate('Unstable from k = 10', xy=(10, kc.filter(pl.col('k') == 10)['stability_ari'][0]),
             xytext=(10.4, 0.5), fontsize=9, color=TEXT2,
             arrowprops=dict(arrowstyle='-', color=TEXT2, linewidth=1))
ax2.set_title('Stability across random starts (adjusted Rand index)', loc='left', fontsize=11, color=TEXT)
ax2.set_xlabel('Number of clusters (k)')
ax2.set_ylabel('Agreement between runs (1 = identical)')
ax2.set_ylim(0, 1.05)

for ax in (ax1, ax2):
    ax.set_xlim(0.5, 15.5)
    ax.set_xticks(KS)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(length=0)

gap = float(res['train_minus_test'].abs().max())
fig.suptitle(f'Choosing the number of customer segments ({s.height:,} customers, {len(seg.FEATURES)} features)',
             x=0.01, ha='left', fontsize=12, color=TEXT)
fig.text(0.01, 0.005, f'Held-out VAF stays within {gap * 100:.1f} percentage points of training VAF at every k '
         '(no sign of overfitting). Stability collapses at k = 10, so k = 8 sits in the stable range.',
         fontsize=8.5, color=TEXT2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
out = config.EDA_DIR / 'segmentation_scree.png'
fig.savefig(out, dpi=200, facecolor=SURFACE)
with pl.Config(tbl_rows=-1, float_precision=4):
    print(res.select('k', 'vaf_train', 'vaf_test', 'train_minus_test'))
print(f'saved {out}')
