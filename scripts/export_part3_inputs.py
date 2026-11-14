"""Export the analysis results (Parts 1 and 2) and the real hourly loads for the Part 3 battery and flexibility pipeline.

Run from the project folder after notebooks 10-70 (needs `technology_predictions`, `appliance_inventory`
and `customer_segments` in avangrid.db):

    python scripts/export_part3_inputs.py                 # every customer in the cached hourly arrays
    python scripts/export_part3_inputs.py --sample 800    # random 800 customers, for a quick test

Writes to onedrive-export/Data/part3/ (inside the git-ignored data folder: this is customer data):

    hourly_kwh.npy           float32, customers x hours (gaps filled the same way as the battery simulator)
    hours.csv                one row per column of hourly_kwh: timestamp, date, hour_ending, month, weekday, temp_f
    customers.csv            one row per row of hourly_kwh, in the same order: segment, program flags,
                             appliance types and confidences, technology scores and list flags
    manifest.json            counts, date range, column descriptions, source tables

The Part 3 pipeline (scripts/run_part3.py) reads these instead of generating synthetic customers; the
resulting app_data.json drives the planner in web/part3/.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import avangrid  # noqa: E402,F401  (thread limits before numerical libraries)
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from avangrid import battery as bt, config, db, grids  # noqa: E402

OUT_DIR = config.DATA_DIR / 'part3'

# Columns kept from each table (missing ones are skipped with a warning, so older builds still export).
SEGMENT_COLS = ['cluster', 'segment', 'high_electrification', 'peak_contributor', 'seasonal', 'flexible_load',
                'flexible_load_count', 'hp_retrofit_candidate', 'ev_adoption_candidate', 'solar']
INVENTORY_COLS = ['ac_confidence', 'ac_flag', 'ac_type', 'ac_added_kw',
                  'elec_heat_confidence', 'elec_heat_primary_confidence', 'elec_heat_flag', 'elec_heat_primary_flag',
                  'heating_type', 'heat_kw_at_20f', 'hp_model_percentile',
                  'water_heater_confidence', 'water_heater_flag',
                  'pool_confidence', 'pool_flag', 'hot_tub_confidence', 'hot_tub_flag',
                  'ev_charging_type', 'ev_level2_confidence', 'ev_level1_confidence',
                  'solar_battery_confidence', 'solar_battery_flag']
# technology_predictions: keep every score / percentile / listed column it has.
TECH_PATTERNS = ('_score', '_percentile', '_listed', '_probability')


def _pick(df: pl.DataFrame, wanted: list[str], table: str) -> pl.DataFrame:
    have = [c for c in wanted if c in df.columns]
    missing = sorted(set(wanted) - set(have))
    if missing:
        print(f'  note: {table} has no column(s) {missing}; skipped')
    return df.select('CUSTOMER_NUMBER', *have)


def _table(name: str) -> pl.DataFrame | None:
    try:
        return db.sql(f'SELECT * FROM {name}')
    except Exception as e:  # table not built yet
        print(f'  WARNING: could not read {name} ({e.__class__.__name__}); its columns are left out')
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sample', type=int, default=0, help='random sample of N customers (0 = all)')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    print('Loading hourly arrays (cached)...')
    hd = grids.load_hourly(verbose=False)
    n_cust, n_days, _ = hd.shape
    rows = np.arange(n_cust)
    if args.sample and args.sample < n_cust:
        rows = np.sort(np.random.default_rng(args.seed).choice(n_cust, args.sample, replace=False))
    cust_nums = hd.customer_numbers[rows]

    print(f'Filling gaps for {len(rows):,} customers x {n_days} days...')
    L = bt.fill_missing(hd.kwh[rows], hd.dates).reshape(len(rows), -1).astype(np.float32)
    n_missing = int(np.isnan(L).sum())
    if n_missing:
        print(f'  {n_missing:,} hours still missing after filling; set to 0')
        L = np.nan_to_num(L, nan=0.0)

    # One row per hour column. Hours use hour_ending 1-24 like the source data (hour_ending 1 = 12-1 am).
    dates = hd.dates.to_numpy()
    day_idx = np.repeat(np.arange(n_days), 24)
    he = np.tile(np.arange(1, 25), n_days)
    stamp = [np.datetime64(str(dates[d])) + np.timedelta64(int(h) - 1, 'h') for d, h in zip(day_idx, he)]
    hours = pl.DataFrame({
        'timestamp': [str(s) for s in stamp],
        'date': [str(dates[d]) for d in day_idx],
        'hour_ending': he,
        'month': [int(str(dates[d])[5:7]) for d in day_idx],
        'weekend': hd.weekend[day_idx],
        'temp_f': hd.temp_hourly.reshape(-1).round(1),
    })

    print('Joining Part 1 and Part 2 results...')
    out = pl.DataFrame({'CUSTOMER_NUMBER': cust_nums,
                        'peer_group': hd.customers['peer_group'].to_numpy()[rows],
                        'annual_kwh': L.sum(1).round(1),
                        'peak_kw': L.max(1).round(3)})
    sources = {}
    seg = _table('customer_segments')
    if seg is not None:
        out = out.join(_pick(seg, SEGMENT_COLS, 'customer_segments'), on='CUSTOMER_NUMBER', how='left')
        sources['customer_segments'] = seg.height
    inv = _table('appliance_inventory')
    if inv is not None:
        out = out.join(_pick(inv, INVENTORY_COLS, 'appliance_inventory'), on='CUSTOMER_NUMBER', how='left')
        sources['appliance_inventory'] = inv.height
    tp = _table('technology_predictions')
    if tp is not None:
        cols = [c for c in tp.columns if c.endswith(TECH_PATTERNS)]
        out = out.join(tp.select('CUSTOMER_NUMBER', *cols), on='CUSTOMER_NUMBER', how='left')
        sources['technology_predictions'] = tp.height
    lab = hd.labels()
    out = out.with_columns(*[pl.Series(f'on_{t.lower()}_list', lab[t][rows]) for t in config.TECHS])

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / 'hourly_kwh.npy', L)
    hours.write_csv(args.out / 'hours.csv')
    out.write_csv(args.out / 'customers.csv')
    manifest = {
        'generated': datetime.now().isoformat(timespec='seconds'),
        'source': 'avangrid-hackathon repo (notebooks 10-70)',
        'n_customers': int(len(rows)), 'n_customers_available': int(n_cust), 'sample': bool(args.sample),
        'n_hours': int(L.shape[1]), 'first_date': str(dates[0]), 'last_date': str(dates[-1]),
        'hourly_kwh': {'file': 'hourly_kwh.npy', 'dtype': 'float32', 'shape': list(L.shape),
                       'rows': 'customers.csv order', 'cols': 'hours.csv order', 'units': 'kWh per hour (= average kW)'},
        'source_tables': sources,
        'customers_columns': out.columns,
        'segments': (out.group_by('segment').len().sort('len', descending=True).to_dicts()
                     if 'segment' in out.columns else []),
    }
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    print(f'Wrote {len(rows):,} customers x {L.shape[1]:,} hours to {args.out}')
    print(f"  hourly_kwh.npy {L.nbytes / 1e9:.2f} GB, customers.csv {len(out.columns)} columns")


if __name__ == '__main__':
    main()
