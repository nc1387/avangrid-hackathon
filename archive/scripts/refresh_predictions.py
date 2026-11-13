"""Write the final technology scores to avangrid.db (technology_predictions) and the candidate CSV.

EV and HP use the improved models from scripts/improve_step2.py:
  EV: average of the percentile ranks of the bagging-PU model (notebook-40 + appliance-rule features)
      and the night-by-night model;
  HP: bagging PU on notebook-40 + peak-load + appliance-rule + heat-pump physics features.
PV keeps the model from scripts/predict_with_peaks.py. Listed customers get out-of-fold scores.

Candidate list sizes: EV and HP use the estimated number of unlisted owners (Elkan-Noto total with the
known 25% sample share, minus those listed); PV uses unlisted customers scoring >= 0.5.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import avangrid  # noqa: E402
import polars as pl  # noqa: E402
from avangrid import cache, config, db  # noqa: E402

prev = cache.load_results('predict_final_scores_v2')            # PV (and previous EV/HP) scores
new = cache.load_results('improve_final_scores')                 # improved EV/HP scores
counts = cache.load_results('predict_final_counts_v2')
n = prev.height

df = (prev.select('CUSTOMER_NUMBER', 'pv_score', 'pv_listed', 'ev_listed', 'hp_listed')
      .join(new.select('CUSTOMER_NUMBER', pl.col('ev_score_v2').alias('ev_score'), 'ev_bagging_score', 'ev_night_score',
                       pl.col('hp_score_v2').alias('hp_score')), on='CUSTOMER_NUMBER', how='left'))
for t in ['ev', 'hp', 'pv']:
    df = df.with_columns((pl.col(f'{t}_score').rank('average') / n).alias(f'{t}_percentile'))
    df = df.with_columns(pl.when(pl.col(f'{t}_listed')).then(None)
                         .otherwise(pl.col(f'{t}_score').rank('ordinal', descending=True).over(f'{t}_listed'))
                         .alias(f'{t}_rank_among_unlisted'))
df = df.select('CUSTOMER_NUMBER',
               'ev_score', 'ev_percentile', 'ev_listed', 'ev_rank_among_unlisted', 'ev_bagging_score', 'ev_night_score',
               'hp_score', 'hp_percentile', 'hp_listed', 'hp_rank_among_unlisted',
               'pv_score', 'pv_percentile', 'pv_listed', 'pv_rank_among_unlisted').sort('CUSTOMER_NUMBER')

with db.connect(read_only=False) as con:
    con.register('df', df)
    con.execute('CREATE OR REPLACE TABLE technology_predictions AS SELECT * FROM df')

c = {r['technology']: r for r in counts.iter_rows(named=True)}
size = {'EV': round(c['EV']['est_total_known_c_0.25'] - c['EV']['listed_in_analysis']),
        'HP': round(c['HP']['est_total_known_c_0.25'] - c['HP']['listed_in_analysis']),
        'PV': c['PV']['unlisted_score_ge_0.5']}
cands = pl.concat([df.filter(~pl.col(f'{t.lower()}_listed')).sort(f'{t.lower()}_score', descending=True).head(k)
                   .select('CUSTOMER_NUMBER', pl.lit(t).alias('technology'), pl.col(f'{t.lower()}_score').alias('score'),
                           pl.col(f'{t.lower()}_rank_among_unlisted').alias('rank'))
                   for t, k in size.items()])
cands.write_csv(config.EDA_DIR / 'technology_candidates.csv')
print(f'technology_predictions refreshed ({df.height:,} customers); candidates: {size}')
