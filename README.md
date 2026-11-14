# Avangrid AMI hackathon analysis

Hourly smart-meter (AMI) data for about 33,000 customers, Sep 2025 – Aug 2026, analysed to identify customers with
electric vehicles (EV), heat pumps (HP) and solar panels (PV), segment them, and plan a demand-flexibility and
battery program. The result is a dashboard: one web page per utility that opens in any browser, with no server.

## Getting the dashboard running

The hackathon data isn't in git, so the dashboard is built from it on your computer: install Python and the packages,
copy the data in, run the notebooks in order, then open the page. `00_setup.ipynb` walks through each step in detail.

**1. Get the code and install the packages** (Python 3.13; exact versions in `requirements.txt`).

macOS (Terminal):

```bash
git clone https://github.com/nc1387/avangrid-hackathon.git
```

```bash
cd avangrid-hackathon
```

```bash
python3.13 -m venv ~/.venvs/avangrid && source ~/.venvs/avangrid/bin/activate
```

```bash
pip install -r requirements.txt && pip install -e .
```

Windows (Command Prompt):

```bat
git clone https://github.com/nc1387/avangrid-hackathon.git
cd avangrid-hackathon
py -3.13 -m venv %USERPROFILE%\.venvs\avangrid
%USERPROFILE%\.venvs\avangrid\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

**2. Put the data in place.** Copy the hackathon files from OneDrive into `onedrive-export/Data/`: the two workbooks
(`EV_Heat Pump_Solar PV Customers.xlsx` and `Weather Data_Avangrid_Hackathon .xlsx`, with the space before `.xlsx`)
and the meter CSV folders under `onedrive-export/Data/meter-data/`. Run the checks in step 4 of `00_setup.ipynb`
to confirm.

**3. Run the notebooks in order.** They build the database (`onedrive-export/Data/avangrid.db`), the models and, last,
the dashboard. About 40 minutes in all; each notebook is executed in place, so its outputs are saved.

| Order | Notebook | Builds |
|---|---|---|
| 1 | `10_load-data` | Meter data and weather in `avangrid.db` |
| 2 | `20_clean-data` | Gap filling and the customers kept for analysis |
| 3 | `30_time-of-day-and-weather` | Time-of-day and weather features |
| 4 | `40_customer-features` | Customer features, peer groups, cached arrays |
| 5 | `45_change-detection` | Mid-year change detection |
| 6 | `55_technology-models_no_experimentation` | EV, heat pump and solar scores |
| 7 | `60_appliances` | Appliance inventory |
| 8 | `70_segmentation` | Customer segments (the clusters on the dashboard) |
| 9 | `75_segmentation-with-appliances` | Segments with appliance flags |
| 10 | `80_battery-opportunity` | Battery simulation for every customer |
| 11 | `85_peak-reduction-stack` | Least-cost peak reduction |
| 12 | `90_usage-stabilization` | Stabilization analysis |
| 13 | `95_appliance-flag-associations` | Appliance-flag associations |
| 14 | `99_build-planner` | **The dashboard**, one page per utility |

macOS (runs them all, stops at the first failure, logs to `onedrive-export/Data/results/rebuild_logs/`; to resume,
pass the step number, e.g. `scripts/rebuild_all.sh 6`):

```bash
scripts/rebuild_all.sh
```

Windows (Command Prompt; also works on macOS; stops at the first failure):

```bat
python -m avangrid run 10_load-data.ipynb 20_clean-data.ipynb 30_time-of-day-and-weather.ipynb 40_customer-features.ipynb 45_change-detection.ipynb 55_technology-models_no_experimentation.ipynb 60_appliances.ipynb 70_segmentation.ipynb 75_segmentation-with-appliances.ipynb 80_battery-opportunity.ipynb 85_peak-reduction-stack.ipynb 90_usage-stabilization.ipynb 95_appliance-flag-associations.ipynb 99_build-planner.ipynb --timeout 7200
```

You can also open each notebook in Jupyter and run it top to bottom, in the order above.

**4. Open the dashboard.** It's `onedrive-export/Data/part3/index.html`, with a drop-down to switch utilities (NYSEG,
CMP, RG&E, UI). It holds real customer data, so keep it inside the data folder and don't share it outside the event.

macOS:

```bash
open onedrive-export/Data/part3/index.html
```

Windows:

```bat
start "" "onedrive-export\Data\part3\index.html"
```

To rebuild only the dashboard after changing an assumption in `config/part3_assumptions.json`, run notebook 99 again
(`python -m avangrid run 99_build-planner.ipynb`, about 10 minutes). `web/part3/README.md` describes every tab.

## How notebook 99 optimizes the battery program

Notebook 99 builds the dashboard with `scripts/run_part3.py`, which decides which homes get flexibility, which get
batteries and how the batteries run. Three steps are optimizations; the rest are rules set in
`config/part3_assumptions.json`.

1. **Per-home screening (rules, not optimized).** A home needs action if its load at the system peak hours is at least
   0.75 kW above the median home's. Flexibility comes first: EV charging moved out of the peak hours, cooling and
   heating shed, water heater and pool pump, only for appliances the appliance classification rates likely. A home
   gets one battery unit if it cuts the home's peak-hour load by at least the candidate threshold, and a second if
   that adds at least 0.5 kW more. These homes are the battery candidates.
2. **Fleet dispatch: a linear program for each day.** The candidates' batteries run as one fleet. For each day, a
   linear program (`fleet_day_lp`, solved with HiGHS through `scipy.optimize.linprog`) sets the fleet's hourly
   charging and discharging to minimize that day's system peak, within each home's load and battery limits, a 3 kW
   charging limit per home and the batteries' stored energy. A second stage keeps that peak and moves charging to the
   lowest-load hours and discharging to the highest. Candidates are ranked by how much load their batteries can cover
   at the system peak, and the fleet is the knee of the peak-versus-fleet-size curve: the smallest fleet within 5% of
   the best achievable peak. Dispatching as a fleet avoids the new overnight peak that appears when every battery
   charges in the same hours.
3. **The candidate threshold that maximizes project NPV.** Project NPV is the present value of the avoided capacity
   value of the expected peak reduction (batteries and flexibility), less the battery hardware and installation. Each
   candidate counts at its adopter group's acceptance rate (40% if its year-1 savings, bill savings plus the
   performance incentive, are above $100; 20% otherwise). The build tries every threshold from 0.5 kW up and keeps the
   one with the highest NPV. When no threshold makes batteries pay, as with the default $100/kW-yr avoided capacity
   value (batteries start to pay above about $1,100/kW-yr), it keeps the lowest threshold at which NPV is still at least
   zero: the largest battery program flexibility's value can pay for. It then rebuilds the fleet with only those homes.
   With the defaults that is 1.315 kW: 293 candidates, about 118 expected to take a battery. The dashboard repeats this
   calculation whenever an assumption changes.

Expected adoption on the dashboard is the average of 100 random draws with a fixed seed: each candidate takes a battery
with its group's probability (the adopters are dispatched as their own fleet) and half of the homes offered
flexibility enroll. NPV counts only battery hardware, installation and avoided capacity value, not other things
Avangrid may value such as grid flexibility or lower carbon emissions. Section 3 of notebook 99 shows each step:
the rules, the linear program, the fleet curve, the knee's cutoff and the NPV by threshold.

## Notebooks

Each builds on tables saved in `onedrive-export/Data/avangrid.db` by the earlier ones. Every notebook ends with a
**Tried but not used** section: what was tested or dropped at that step, and why.

| Notebook | What it does | Run time |
|---|---|---|
| `00_setup` | How to replicate the folder structure and environment; checks package versions and the data | seconds |
| `10_load-data` | Loads the meter CSVs with polars, data-quality checks, writes the `ami` and `weather` tables | ~1 min |
| `20_clean-data` | EV/HP/PV list coverage, imputes short gaps, drops still-incomplete customers (`ami_analysis` view) | seconds |
| `30_time-of-day-and-weather` | Time of day and weather quartiles (`weather_features`), kWh box plots | ~1 min |
| `40_customer-features` | Full-year features (`customer_features`), labels, peer groups and `peer_index`, cached arrays, `customer_status` | ~1 min |
| `45_change-detection` | Stabilization length (`customer_stability`) and mid-year change detection (`change_points`), used as candidate features | ~12 min |
| `55_technology-models_no_experimentation` | EV, HP and PV scores from positive-unlabeled learning (bagging PU; an ensemble with a night-by-night model for EV); `technology_predictions`, Elkan–Noto count estimates and candidate lists | ~6 min |
| `60_appliances` | Appliance rules calibrated with simulated appliances; `appliance_inventory` | <1 min |
| `70_segmentation` | k-means behavioural clusters, program segments, VAF scree plot; `customer_segments` | ~1 min |
| `75_segmentation-with-appliances` | Clusters on the same features **plus the appliance inventory** (categorical), with k-prototypes; compared with notebook 70; `customer_segments_mixed` | ~2 min |
| `80_battery-opportunity` | Battery dispatch for every customer, coordinated charging windows; `battery_simulation` | <1 min |
| `85_peak-reduction-stack` | Least-cost peak reduction: flexible loads found by the appliance flags (thermostats, pool pumps, water heaters, EV charging) ranked against batteries by cost per kW, by cluster; `peak_reduction_stack` | <1 min |
| `90_usage-stabilization` | The rest of the stabilization analysis (not needed by the models); completes `customer_stability` | <1 min |
| `95_appliance-flag-associations` | How well the appliance flags predict one another (correlation, lift, conditional odds, models), and injection tests of which rules confuse one load for another | <1 min |
| `99_build-planner` | Builds the dashboard (the demand-flexibility and battery planner) from `avangrid.db`, once per utility on its real residential time-varying rate (`data/tariffs/`), and documents the battery program's optimization; open `onedrive-export/Data/part3/index.html` and switch utilities with the drop-down | ~10 min |

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
| `stability` | `row_corr`, `window_r` (stabilization comparisons), `change_scan`, `step_scan`, `tech_change_points` |
| `pu` | `bagging_pu` (**parallel**), `elkan_noto`, `evaluate_feature_sets` for hidden-label tests |
| `segmentation` | Segmentation features, `robust_standardize`, `choose_k`, `name_clusters`; `kprototypes` for continuous + categorical data |
| `status` | `build_customer_status()`: the `customer_status` table (see below) |

### Which customers are in the analysis: `customer_status`

One row for each of the 35,725 customer numbers in the data dictionary, saying whether and where
each was excluded:

| `status` | Customers | Meaning |
|---|---|---|
| `included` | 33,310 | In `customer_peer_groups`: used from the peer groups in notebook 40 on, and in the cached arrays |
| `excluded: low usage` | 256 | Average under 0.05 kWh/hour; still in `ami_analysis` (notebooks 20–40) but not in peer groups |
| `dropped: incomplete` | 672 | In `excluded_customers`; `drop_reason` is joined late, left early, or long gap mid-year |
| `no meter data` | 1,487 | Absent from the meter data: 1,402 of them (18,152–18,346 and 18,747–19,953) are in the seven part files missing from the 10,001–20,000 folder |

Other columns: `in_ami`, `in_ami_analysis`, `in_peer_groups`, `dropped_incomplete`, `low_usage`,
`drop_reason`, `missing_hours_after_imputation`, `imputed_hours`, `peer_group`,
and `on_ev_list` / `on_hp_list` / `on_pv_list`. Rebuild it with `python -m avangrid status-build`
after re-running any of notebooks 10–40 (notebook 40 builds it as its last step).

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

1. **Parallel model training.** `pu.bagging_pu` fits several models at once in separate processes, each on one thread. It gives exactly the same scores as the one-at-a-time version in notebook 45, about **7–8× faster** (30 models: 10.7 s down to 1.4 s). Pass `n_jobs=1` to run sequentially.
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
python -m avangrid run 70_segmentation.ipynb
```

```bash
python -m avangrid benchmark
```

```bash
python -m avangrid status-build
```

- `cache-build` builds or refreshes the caches (about 16 s); `cache-clear` deletes them.
- `run` executes notebooks in place, in the order given, stopping at the first failure (`--timeout` sets the seconds allowed per cell, 3600 by default; `rebuild_all.sh` allows 7200) from the project folder with the standard `python3` kernel, then puts each notebook's own kernel setting (`conda-base-py`) back. If a cell fails, the original notebook is left untouched, a copy with the partial outputs is saved to `onedrive-export/Data/results/failed_runs/`, and the command exits with an error.
- `benchmark` times parallel against sequential PU training and checks the scores are identical.
- `status-build` creates or refreshes the `customer_status` table. It needs to write to the database, so run it when no notebook has `avangrid.db` open; otherwise it stops with a lock error.

Long-running results are saved with `cache.save_results(name, df)`, so later notebooks load them with `cache.load_results(name)` instead of refitting.

### Things to know

- **Import `avangrid` before scikit-learn.** It sets `OMP_NUM_THREADS=1` if unset. Without that, gradient boosting starts a thread per core for every small model, and a one-minute job can take more than ten.
- **Database locks.** DuckDB lets only one process open `avangrid.db` for writing, and while it's open nobody else can even read it. Notebooks that create tables hold that lock until they close their connection. Open write connections only around the statements that need them (`with db.connect(read_only=False) as con: ...`). If the database is busy, `load_hourly()` and `load_daily_signals()` fall back to the cached copy with a warning.
- **Imputed hours are NaN** in the cached arrays. Use the `ami_filled` view in DuckDB if you need the imputed values.
- **DuckDB 2.0:** read results with `con.sql(query).pl()`; relations no longer have `fetchone()`.

## Setup details

Python 3.13 with the exact package versions in `requirements.txt`; see **Getting the dashboard running** above.
`00_setup.ipynb` walks through a fresh setup: the data folder layout, installing (conda or venv), and checks.

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
