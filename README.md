# Avangrid AMI hackathon analysis

Hourly smart-meter (AMI) data for about 33,000 customers, Sep 2025 – Aug 2026, analysed to
identify customers with electric vehicles (EV), heat pumps (HP) and solar panels (PV).

## Notebooks

Run them in order from this folder. Each builds on tables saved in
`onedrive-export/Data/avangrid.db` by the earlier ones.

| Notebook | What it does |
|---|---|
| `00_setup` | How to replicate the folder structure and environment; checks package versions and the data |
| `10_exploratory-data-analysis` | Loads the meter CSVs with polars, data-quality checks, writes the `ami` and `weather` tables |
| `20_exploratory_data_analysis_p2` | EV/HP/PV list coverage, imputes short gaps, drops still-incomplete customers (`ami_analysis` view) |
| `30_define-variables` | Time of day and weather quartiles (`weather_features`), kWh box plots |
| `40_identify-technology-customers` | Per-customer features (`customer_features`), PU learning tested on PV |
| `50_usage-stabilization` | Peer groups, peer-relative residuals, how long usage patterns take to settle |

## The `avangrid` package: shared code for speed

Code used by several notebooks lives in `avangrid/`, so it's written and tested once and
notebooks only hold the calls, results and explanations.

```python
import avangrid                      # import first: limits scikit-learn to one thread per model
from avangrid import config, db, grids, features, stability, pu, labels, cache
```

| Module | Contents |
|---|---|
| `config` | Paths (resolved from the package, so any working folder works) and constants such as `EXPECTED_ROWS`, `TOD_SLICES`, `MIDDAY` |
| `db` | `connect()` / `sql()` for DuckDB with the progress bar off; read-only by default |
| `labels` | EV/HP/PV lists from the workbook; `label_arrays()` aligned to any customer order |
| `grids` | `load_hourly()`: hourly kWh and peer-relative residuals as customers × days × 24 arrays, **cached** |
| `features` | `load_daily_signals()` (**cached**), `PeriodFeatures` for any date range, `remove_weather()`, `customer_features()` |
| `stability` | `row_corr`, `window_r` (notebook 50 comparisons), `change_scan`, `step_scan` |
| `pu` | `bagging_pu` (**parallel**), `elkan_noto`, `evaluate_feature_sets` for hidden-label tests |
| `status` | `build_customer_status()`: the `customer_status` table (see below) |

### Which customers are in the analysis: `customer_status`

One row for each of the 35,725 customer numbers in the data dictionary, saying whether and where
each was excluded:

| `status` | Customers | Meaning |
|---|---|---|
| `included` | 33,310 | In `customer_peer_groups`: used from notebook 50 on and in the cached arrays |
| `excluded: low usage` | 256 | Average under 0.05 kWh/hour; still in `ami_analysis` (notebooks 20–40) but not in peer groups |
| `dropped: incomplete` | 672 | In `excluded_customers`; `drop_reason` is joined late, left early, or long gap mid-year |
| `no meter data` | 1,487 | Absent from the meter data: 1,402 of them (18,152–18,346 and 18,747–19,953) are in the seven part files missing from the 10,001–20,000 folder |

Other columns: `in_ami`, `in_ami_analysis`, `in_peer_groups`, `dropped_incomplete`, `low_usage`,
`drop_reason`, `missing_hours_after_imputation`, `imputed_hours`, `peer_group`,
and `on_ev_list` / `on_hp_list` / `on_pv_list`. Rebuild it with `python -m avangrid status-build`
after re-running any of notebooks 10–50.

### Example

```python
import avangrid
from avangrid import grids, features, pu

hd = grids.load_hourly()                  # ~0.1 s from cache (customers x 365 days x 24 hours)
D, refs = features.load_daily_signals(hd) # ~0.1 s from cache
pf = features.PeriodFeatures(D, hd.temp)  # ~3 s once; then each date range is ~0.1 s
X, names = pf.to_matrix(pf(0, hd.shape[1]))
is_pv = hd.labels()['PV']

results = pu.evaluate_feature_sets({'full year': X}, is_pv, scheme='hide', n_splits=5)
pu.summarize(results)
```

### What makes things faster

1. **Parallel model training.** `pu.bagging_pu` fits several models at once in separate processes, each on one thread. It gives exactly the same scores as the one-at-a-time version used in notebooks 40 and 60, about **7–8× faster** (30 models: 10.7 s down to 1.4 s). Pass `n_jobs=1` to run sequentially.
2. **Cached arrays.** The hourly arrays (about 6 s and several GB of memory to build) and daily signals (about 9 s) are saved in `onedrive-export/Data/cache/` and reloaded in about 0.1 s, memory-mapped from disk. The cache checks a fingerprint of the source tables plus a code version on each load and rebuilds automatically if anything changed. It's safe to delete.
3. **Cheap date ranges.** `PeriodFeatures` precomputes cumulative sums, so features for any per-customer date range (for example "since each customer's change point") cost one lookup instead of a pass over the hourly data.
4. **Running notebooks without babysitting.** See the commands below.

### Command line

Run from this folder:

```bash
python -m avangrid cache-build
```

```bash
python -m avangrid cache-info
```

```bash
python -m avangrid run 50_usage-stabilization.ipynb
```

```bash
python -m avangrid benchmark
```

```bash
python -m avangrid status-build
```

- `cache-build` builds or refreshes the caches (about 16 s); `cache-clear` deletes them.
- `run` executes notebooks in place from the project folder with the standard `python3` kernel, then puts each notebook's own kernel setting (`conda-base-py`) back. If a cell fails, the original notebook is left untouched, a copy with the partial outputs is saved to `onedrive-export/Data/results/failed_runs/`, and the command exits with an error.
- `benchmark` times parallel against sequential PU training and checks the scores are identical.
- `status-build` creates or refreshes the `customer_status` table. It needs to write to the database, so run it when no notebook has `avangrid.db` open; otherwise it stops with a lock error.

For long experiments, write a script that saves its output with `cache.save_results(name, df)`, run it in the background, and load the results in a notebook with `cache.load_results(name)`.

### Things to know

- **Import `avangrid` before scikit-learn.** It sets `OMP_NUM_THREADS=1` if unset. Without that, gradient boosting starts a thread per core for every small model, and a one-minute job can take more than ten.
- **Database locks.** DuckDB lets only one process open `avangrid.db` for writing, and while it's open nobody else can even read it. Notebooks that create tables hold that lock until they close their connection. Open write connections only around the statements that need them (`with db.connect(read_only=False) as con: ...`). If the database is busy, `load_hourly()` and `load_daily_signals()` fall back to the cached copy with a warning.
- **Imputed hours are NaN** in the cached arrays. Use the `ami_filled` view in DuckDB if you need the imputed values.
- **DuckDB 2.0:** read results with `con.sql(query).pl()`; relations no longer have `fetchone()`.

## Setup

Python 3.13 with the exact package versions in `requirements.txt` (`pip install -r requirements.txt`).
`00_setup.ipynb` walks through a fresh setup: the data folder layout, installing, and checks.

Notebooks and `python -m avangrid ...` commands run from this folder find the package
automatically. A script saved elsewhere won't, because Python looks next to the script. Either
run it with the project folder on the path:

```bash
PYTHONPATH=/Users/nchaturvedi/Documents/Code/avangrid python3 my_script.py
```

or install the package once in editable mode, so code changes take effect without reinstalling:

```bash
pip install -e /Users/nchaturvedi/Documents/Code/avangrid
```
