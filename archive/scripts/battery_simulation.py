"""Run the default battery scenario for every customer and save the results.

Defaults: the Challenge Statement battery (20.16 kWh, 7.68 kW discharge, 89.4% round trip) at every home,
charging capped at 3 kW (the Challenge Statement's ~3 kVA limit on additional load for a 200-amp service), hybrid
dispatch, PLACEHOLDER time-of-use tariff (summer Jun-Sep; on-peak 4-9 pm weekdays; $0.30 / $0.12 summer,
$0.22 / $0.11 winter per kWh; $8/kW-month on the monthly on-peak peak).

Charging windows are coordinated (avangrid.battery.coordinate_windows): each segment gets a mix of the
CHARGE_WINDOWS that maximizes total savings while keeping the system peak at least PEAK_CUT below today's
(customers keep their own best window where the mix allows). For other settings use the interactive app:
streamlit run app/battery_simulator.py

Outputs: avangrid.db table battery_simulation (one row per customer, with segment and charge_window), and in
onedrive-export/Data/eda/: battery_simulation.csv, battery_summary_by_segment.csv, battery_summary.csv,
battery_window_comparison.csv (same window for everyone, per window), battery_peak_savings_frontier.csv.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import avangrid  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from avangrid import battery as bt, cache, config, db, grids  # noqa: E402

PEAK_CUT = 0.075

t0 = time.time()
hd = grids.load_hourly(verbose=False)
L = bt.fill_missing(hd.kwh, hd.dates).reshape(hd.shape[0], -1)
seg = db.sql('SELECT CUSTOMER_NUMBER, segment, peak_contributor, high_electrification, flexible_load FROM customer_segments')
labels = (pl.DataFrame({'CUSTOMER_NUMBER': hd.customer_numbers}).join(seg, on='CUSTOMER_NUMBER', how='left')
          ['segment'].fill_null('Unassigned').to_numpy())
cmp = bt.compare_windows(L, hd.dates, segments=labels)
plan = bt.coordinate_windows(cmp, cmp['system_before'].max() * (1 - PEAK_CUT))
res = bt.simulate(L, hd.dates, windows=cmp['windows'], window_of=plan['window_of'])
print(f'simulated {L.shape[0]:,} customers x {len(cmp["names"])} windows + coordinated plan in {time.time() - t0:.0f}s')
cmp['summary'].write_csv(config.EDA_DIR / 'battery_window_comparison.csv')
frontier = bt.peak_savings_frontier(cmp)
frontier.write_csv(config.EDA_DIR / 'battery_peak_savings_frontier.csv')

cust = (res['customers'].with_columns(pl.Series('CUSTOMER_NUMBER', hd.customer_numbers),
                                      pl.Series('charge_window', np.array(cmp['names'])[plan['window_of']]))
        .join(seg, on='CUSTOMER_NUMBER', how='left')
        .with_columns(pl.col('bill_savings').rank('ordinal', descending=True).alias('savings_rank')))
with db.connect(read_only=False) as con:
    con.register('c', cust)
    con.execute('CREATE OR REPLACE TABLE battery_simulation AS SELECT * FROM c ORDER BY savings_rank')
cust.sort('savings_rank').write_csv(config.EDA_DIR / 'battery_simulation.csv')


def summarize(df: pl.DataFrame, by: str | None = None) -> pl.DataFrame:
    aggs = [pl.len().alias('customers'),
            pl.col('bill_savings').sum().alias('total_savings'), pl.col('bill_savings').mean().alias('avg_savings'),
            pl.col('bill_savings').median().alias('median_savings'),
            (pl.col('bill_savings').sum() / pl.col('bill_before').sum()).alias('savings_share_of_bill'),
            pl.col('bill_savings_pct').mean().alias('avg_savings_pct'),
            pl.col('bill_savings_pct').median().alias('median_savings_pct'),
            pl.col('avg_monthly_peak_kw_reduction').mean().alias('avg_monthly_peak_cut_kw'),
            pl.col('grid_peak_kw_reduction').mean().alias('grid_peak_hour_cut_kw_per_customer'),
            (pl.col('grid_peak_kw_reduction').sum() / 1000).alias('grid_peak_hour_cut_mw'),
            (pl.col('on_peak_kwh_decrease').sum() / 1000).alias('on_peak_mwh_removed'),
            (pl.col('on_peak_kwh_decrease').sum() / pl.col('on_peak_kwh_before').sum()).alias('on_peak_share_removed'),
            (pl.col('off_peak_kwh_increase').sum() / 1000).alias('off_peak_mwh_added'),
            (pl.col('losses_kwh').sum() / 1000).alias('losses_mwh'),
            pl.col('equivalent_full_cycles').median().alias('median_cycles')]
    return (df.group_by(by).agg(aggs).sort('avg_savings', descending=True) if by else df.select(aggs))


overall = summarize(cust).with_columns(**{k: pl.lit(v) for k, v in bt.system_peak_reduction(res).items()})
by_seg = summarize(cust, 'segment')
overall.write_csv(config.EDA_DIR / 'battery_summary.csv')
by_seg.write_csv(config.EDA_DIR / 'battery_summary_by_segment.csv')
cache.save_results('battery_summary', overall)
cache.save_results('battery_summary_by_segment', by_seg)
with pl.Config(tbl_cols=-1, tbl_width_chars=250, tbl_hide_dataframe_shape=True, float_precision=2, fmt_str_lengths=35):
    print(overall)
    print(by_seg)
    print(cmp['summary'])
    print(frontier)
    print(cust.group_by('segment', 'charge_window').len().pivot('charge_window', index='segment', values='len').fill_null(0))
    print(cust.sort('savings_rank').head(10).select('savings_rank', 'CUSTOMER_NUMBER', 'segment', 'bill_savings',
                                                    'bill_savings_pct', 'bill_before', 'avg_monthly_peak_kw_reduction', 'grid_peak_kw_reduction'))
