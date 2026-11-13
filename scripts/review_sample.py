"""Build the blind hand-labelling sample and its review sheets (step 3).

The sample mixes model candidates, listed customers, rule-flagged homes and random homes, shuffled and
given anonymous codes so the reviewer can't tell which is which. The key (code -> customer and why it
was sampled) is saved separately in results/review_key.parquet and should only be opened after labelling.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import avangrid  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from avangrid import cache, config, db, grids, review  # noqa: E402

hd = grids.load_hourly(verbose=False)
lab = hd.labels()
listed = lab['EV'] | lab['HP'] | lab['PV']
s = cache.load_results('improve_final_scores')
inv = (pl.DataFrame({'CUSTOMER_NUMBER': hd.customer_numbers})
       .join(db.sql('SELECT CUSTOMER_NUMBER, ev_charging_type, heating_type FROM appliance_inventory'), on='CUSTOMER_NUMBER', how='left'))
rng = np.random.default_rng(2026)
chosen = {}


def take(rows, k, reason):
    rows = [r for r in rows if r not in chosen]
    for r in rows[:k]:
        chosen[r] = reason


unl = np.flatnonzero(~listed)
take(list(unl[np.argsort(-s['ev_score_v2'].to_numpy()[unl])]), 20, 'EV top candidate')
take(list(rng.permutation(np.flatnonzero(lab['EV']))), 10, 'EV listed')
take(list(rng.permutation(np.flatnonzero((inv['ev_charging_type'] == 'Level 2 likely').to_numpy() & ~listed))), 10, 'Level 2 flagged')
take(list(unl[np.argsort(-s['hp_score_v2'].to_numpy()[unl])]), 20, 'HP top candidate')
take(list(rng.permutation(np.flatnonzero(lab['HP']))), 10, 'HP listed')
take(list(rng.permutation(np.flatnonzero((inv['heating_type'] == 'resistance likely').to_numpy() & ~listed))), 10, 'resistance likely')
take(list(rng.permutation(unl)), 40, 'random unlisted')

rows = np.array(list(chosen))
order = rng.permutation(len(rows))
rows = rows[order]
codes = [f'{i + 1:03d}' for i in range(len(rows))]
key = pl.DataFrame({'code': codes, 'row': rows, 'CUSTOMER_NUMBER': hd.customer_numbers[rows],
                    'reason': [chosen[r] for r in rows]})
cache.save_results('review_key', key)

out = config.EDA_DIR / 'review'
out.mkdir(parents=True, exist_ok=True)
PER_SHEET = 6
for i in range(0, len(rows), PER_SHEET):
    review.review_sheet(hd, rows[i:i + PER_SHEET], codes[i:i + PER_SHEET], out / f'sheet_{i // PER_SHEET + 1:02d}.png')
print(f'{len(rows)} homes on {int(np.ceil(len(rows) / PER_SHEET))} sheets in {out}')
