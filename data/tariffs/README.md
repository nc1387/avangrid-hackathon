# Residential electric rates

Residential rates for Avangrid's four electric utilities (NYSEG, RG&E, Central Maine Power and United Illuminating),
from the [OpenEI Utility Rate Database](https://openei.org/wiki/Utility_Rate_Database), for toggling between
utilities in the battery simulator.

| File | What it is |
|---|---|
| `residential_tariffs.json` | **The file to load.** Every utility, its rate plans, prices, hours and sources |
| `openei/<label>.json` | OpenEI's full record for each rate used, as fetched (the label is OpenEI's rate ID) |

The Part 3 planner uses these rates: `scripts/run_part3.py --utility <id>`, and notebook 99 builds one planner page per
utility.

Both are committed, so nobody needs an API key to reproduce the results. They are also a fixed snapshot: OpenEI
entries and the rates themselves change over time.

## Rebuilding

```bash
python scripts/fetch_tariffs.py --from-cache
```

rebuilds `residential_tariffs.json` from the saved OpenEI records (no key, no internet).

```bash
python scripts/fetch_tariffs.py
```

fetches fresh records from OpenEI first. It needs a free API key from https://openei.org/services/api/signup/, either
as an `OPENEI_API_KEY` environment variable or in a `.env` file in the project folder: copy `.env.example` to `.env`
and paste the key after `OPENEI_API_KEY=`. `.env` is git-ignored, and the key is never written to any output.

Which OpenEI rates are used, and what was checked against each utility's own tariff, is set in `SELECTION` at the
top of `scripts/fetch_tariffs.py`.

## Format of `residential_tariffs.json`

```text
meta                      source, date pulled, the meter data's year, notes
utilities[]
  id, name, full_name     "nyseg" | "rge" | "cmp" | "ui"; "NYSEG"; "New York State Electric & Gas"
  state, eia_id           "NY"; 13511
  regions, default_region NYSEG only: supply price differs by region ("West" is the default)
  plans[]
    id, name              "standard" | "tou" | "day_night"; "Time-of-use (Rate RT)"
    kind                  "flat" | "time_of_use"
    region                NYSEG: "West" | "East" | "Lower Hudson Valley"; others: null
    default               true for each utility's standard plan (NYSEG: the West one)
    eligibility, checked  who can take the rate; what was verified against the utility's tariff
    versions[]            one per rate change in the data year, oldest first; the last is current
      start, end          "2026-01-01"; null = still in effect
      fixed_charge_per_month
      periods[]           index, name ("Off-peak" | "Mid-peak" | "On-peak" | "All hours"), rate_per_kwh, tiers[],
                          and delivery_per_kwh + supply_per_kwh where OpenEI has a matching delivery-only entry
      weekday_schedule    12 months x 24 hours of period indices (hour 0 = midnight-1 am, local clock time)
      weekend_schedule    the same for Saturdays and Sundays
      hours[]             the schedules in words, grouped by months with the same hours
      demand_charges      false for every rate here
      openei, tariff_source   where the rate came from
```

The price for any hour is `periods[schedule[month - 1][hour]].rate_per_kwh`, using `weekday_schedule` or
`weekend_schedule`. All prices are in dollars per kWh and include the utility's default supply.

**Tiers.** CMP's standard rate charges less for the first 50 kWh each month. `rate_per_kwh` is the price above the
first tier, which is what nearly all of a home's use is charged at; the full tiers are in `tiers`.

## Loading it

In a browser:

```js
const tariffs = await (await fetch('data/tariffs/residential_tariffs.json')).json();
const ui = tariffs.utilities.find(u => u.id === 'ui');
const plan = ui.plans.find(p => p.id === 'tou');
const current = plan.versions.at(-1);
const price = (month, hour, weekend) =>   // month 1-12, hour 0-23
  current.periods[(weekend ? current.weekend_schedule : current.weekday_schedule)[month - 1][hour]].rate_per_kwh;
```

In Python:

```python
import json
tariffs = json.load(open('data/tariffs/residential_tariffs.json'))
```

## Caveats

- **No demand charges.** None of these residential rates has one. The placeholder tariff in notebook 80 has an
  $8/kW-month demand charge, which produces most of its simulated battery savings.
- **Supply prices are snapshots.** NYSEG and RG&E supply prices change monthly, UI's twice a year and CMP's yearly.
  Each rate is OpenEI's entry for the dates shown. NYSEG's and RG&E's entries date from 2025.
- **NYSEG's time-of-use rate (SC 12) is only for homes using 35,000 kWh a year or more**, so its Day/Night rate
  (SC 8) is the realistic time-varying option for ordinary homes. NYSEG also changed SC 12's hours from 8 Mar 2026,
  which OpenEI may not reflect.
- **Daylight saving.** Schedules are in local clock time. NYSEG defines Day/Night hours in Eastern Standard Time
  (night starts at 12:30 am in summer); OpenEI rounds them to clock hours.
- Taxes, most riders and holidays aren't included.
