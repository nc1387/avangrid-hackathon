"""Pull the residential electric rates of Avangrid's four electric utilities from the OpenEI Utility Rate Database
and save them as one JSON file a front end can load directly.

    python scripts/fetch_tariffs.py               # fetch from OpenEI (needs an API key), then build the JSON
    python scripts/fetch_tariffs.py --from-cache  # rebuild the JSON from the saved OpenEI responses (no key needed)

API key: free from https://openei.org/services/api/signup/. Read from the OPENEI_API_KEY environment variable, or
from a line OPENEI_API_KEY=... in a .env file in the project folder (.env is git-ignored). The key is never written
to any output.

Outputs (committed, so the results are reproducible without a key, and pinned: OpenEI entries and the rates
themselves change over time):
    data/tariffs/openei/<label>.json       OpenEI's full record for every rate used
    data/tariffs/residential_tariffs.json  the front-end file (schema in data/tariffs/README.md)

Which rates are used, and what was checked against the utilities' own tariffs, is set in SELECTION below.
"""

import argparse
import datetime as dt
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'data' / 'tariffs'
RAW_DIR = OUT_DIR / 'openei'
API = 'https://api.openei.org/utility_rates'
DATA_YEAR = ('2025-09-01', '2026-08-31')   # the meter data's year

# OpenEI lists two entries for most of these rates: delivery only, and bundled (delivery plus the utility's default
# supply). The bundled ones are used: they are what a customer who hasn't chosen another supplier pays. Where a rate
# changed during the data year, each version is listed with its dates.
SELECTION = [
    {
        'id': 'nyseg', 'name': 'NYSEG', 'full_name': 'New York State Electric & Gas', 'state': 'NY', 'eia_id': 13511,
        'regions': ['West', 'East', 'Lower Hudson Valley'], 'default_region': 'West',
        'region_note': "NYSEG's supply price differs by region; West is OpenEI's default residential rate.",
        'plans': [
            {'id': 'standard', 'name': 'Standard residential (SC 1)', 'kind': 'flat', 'default': True,
             'labels': {'West': ['67bf2b23defa2cd03006bfb3'], 'East': ['67bf2b0501ca3937950b5192'],
                        'Lower Hudson Valley': ['67bf2429defa2cd03006bfb2']},
             'delivery_labels': ['67bf21821a06c5ae710b878d'],   # same delivery rate in every region
             'eligibility': 'All residential customers.',
             'checked': 'Bundled (NYSEG supply) entry; rates not checked against the tariff.'},
            {'id': 'day_night', 'name': 'Day/Night (SC 8)', 'kind': 'time_of_use',
             'labels': {'West': ['67bf920fbf2fbcda3c0efd5f'], 'East': ['67bf9216997764b245041d48'],
                        'Lower Hudson Valley': ['67bf921f776dc2f3380e63e3']},
             'delivery_labels': ['67bf90ab776dc2f3380e63e2'],   # same delivery rate in every region
             'eligibility': "Optional for residential customers. NYSEG's main time-varying rate for ordinary homes.",
             'checked': "Night hours match NYSEG's pricing page (11:30 pm-7 am Eastern Standard Time; OpenEI rounds "
                        'to 11 pm-7 am clock time). Rates not checked.'},
            {'id': 'tou', 'name': 'Time-of-use (SC 12)', 'kind': 'time_of_use',
             'labels': {'West': ['67bf96bb776dc2f3380e63e4'], 'East': ['67bf95b51a06c5ae710b8794'],
                        'Lower Hudson Valley': ['67bf97c776aff5978905380c']},
             'delivery_labels': ['67bf95ae1a06c5ae710b8793'],   # same delivery rate in every region
             'eligibility': 'Only for customers using at least 35,000 kWh a year (about four times a typical home).',
             'checked': "Seasons match NYSEG's page (summer peak Jun-Aug, no peak in spring and autumn) and the summer "
                        'peak matches the older tariff (10 am-6 pm weekdays). NYSEG published new hours effective '
                        '8 Mar 2026 that OpenEI may not reflect. Rates not checked.'},
        ],
    },
    {
        'id': 'rge', 'name': 'RG&E', 'full_name': 'Rochester Gas and Electric', 'state': 'NY', 'eia_id': 16183,
        'plans': [
            {'id': 'standard', 'name': 'Standard residential (SC 1)', 'kind': 'flat', 'default': True,
             'labels': ['68c87662584527c3070c367d'],
             'delivery_labels': ['68c88812dbb826babf06c4d4'],
             'eligibility': 'All residential customers.',
             'checked': 'Bundled (RG&E supply) entry; rates not checked against the tariff.'},
            {'id': 'tou', 'name': 'Time-of-use (SC 4, Schedule I)', 'kind': 'time_of_use',
             'labels': ['68c8894d0ecf0fba6b094e5a'],
             'delivery_labels': ['68c888c6d948d724670ef118'],
             'eligibility': 'Homes using up to 24,750 kWh a year (Schedule II, for larger homes, is in the raw files).',
             'checked': "Peak hours match RG&E's pricing page (Mon-Fri 7 am-9 pm). OpenEI's customer charge ($27.93) "
                        'predates the tariff effective 1 Jun 2026 ($27.00).'},
        ],
    },
    {
        'id': 'cmp', 'name': 'CMP', 'full_name': 'Central Maine Power', 'state': 'ME', 'eia_id': 3266,
        'plans': [
            {'id': 'standard', 'name': 'Standard residential (Rate A, standard offer)', 'kind': 'flat', 'default': True,
             'labels': ['66b11517fdb63f7ea70f62cd', '6993c9d33444dea9740958a9'],
             'delivery_labels': ['66b114fefdb63f7ea70f62cc', '6993c84caf24f32cad0c1778'],
             'eligibility': 'All residential customers.',
             'checked': 'Bundled (standard offer) entry; rates not checked against the tariff.'},
            {'id': 'tou', 'name': 'Time-of-use (Rate A-TOU)', 'kind': 'time_of_use',
             'labels': ['66b11699a650c4a9610c481a', '6993c9629fff77cf2c0a967a'],
             'delivery_labels': ['66b116915bb82009ed07fd1e', '6993c8de530d515912068248'],
             'eligibility': 'Optional for residential customers.',
             'checked': "2026 customer charge and delivery rates match CMP's tariff (Docket 2025-00297: $26.71, "
                        '50.31 and 6.75 cents/kWh); the 2026 on-peak hours (weekdays 5-9 pm) are from OpenEI only.'},
        ],
    },
    {
        'id': 'ui', 'name': 'UI', 'full_name': 'United Illuminating', 'state': 'CT', 'eia_id': 19497,
        'plans': [
            {'id': 'standard', 'name': 'Standard residential (Rate R)', 'kind': 'flat', 'default': True,
             'labels': ['6759cfb6ab9afd98fa09bc98', '69af138baa74f99fce05780a'],
             'delivery_labels': ['6759cfb1631cc469390a4bc4', '69af11280c671e7bff0d1bef'],
             'eligibility': 'All residential customers.',
             'checked': 'Bundled (standard service) entry; rates not checked against the tariff.'},
            {'id': 'tou', 'name': 'Time-of-use (Rate RT)', 'kind': 'time_of_use',
             'labels': ['6759e0fa631cc469390a4bc6', '69af148be15fe96c780a955b'],
             'delivery_labels': ['6759e0f5631cc469390a4bc5', '69af13d3a9608156eb096f1c'],
             'eligibility': 'Optional for residential customers.',
             'checked': "Peak hours match UI's tariff (weekdays noon-8 pm). The 2026 bundled rates equal OpenEI's "
                        "delivery rates plus UI's published Jan-Jun 2026 generation prices (16.26 and 12.76 cents/kWh)."},
        ],
    },
]


# ------------------------------------------------------------------------------------------------ fetching
def api_key() -> str | None:
    key = os.environ.get('OPENEI_API_KEY')
    env = ROOT / '.env'
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith('OPENEI_API_KEY='):
                key = line.split('=', 1)[1].strip().strip('"').strip("'")
    return key or None


def fetch(label: str, key: str) -> dict:
    query = urllib.parse.urlencode({'version': 'latest', 'format': 'json', 'detail': 'full', 'getpage': label,
                                    'api_key': key})
    with urllib.request.urlopen(f'{API}?{query}', timeout=180) as r:
        items = json.load(r).get('items', [])
    if len(items) != 1:
        raise RuntimeError(f'OpenEI returned {len(items)} rates for label {label}')
    return items[0]


# ------------------------------------------------------------------------------------------------ building
def date(ts):
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).date().isoformat() if ts else None


def period_names(rates: list[float]) -> list[str]:
    """Name periods by price: one price 'All hours'; otherwise cheapest 'Off-peak', dearest 'On-peak', any between
    'Mid-peak'. Periods with the same price share a name."""
    prices = sorted(set(rates))
    if len(prices) == 1:
        return ['All hours'] * len(rates)
    name_of = {p: 'Off-peak' if i == 0 else 'On-peak' if i == len(prices) - 1 else 'Mid-peak'
               for i, p in enumerate(prices)}
    return [name_of[r] for r in rates]


def hour_ranges(row: list[int], periods: set[int]) -> str:
    """'10:00-18:00, 22:00-24:00' for the hours of `row` (24 period indices) in any of `periods`."""
    spans, start = [], None
    for h in range(25):
        on = h < 24 and row[h] in periods
        if on and start is None:
            start = h
        if not on and start is not None:
            spans.append(f'{start:02d}:00-{h:02d}:00')
            start = None
    return ', '.join(spans)


def describe(weekday: list[list[int]], weekend: list[list[int]], names: list[str]) -> list[dict]:
    """Group months with identical schedules and describe each group's hours in words."""
    groups: dict[tuple, list[int]] = {}
    for m in range(12):
        groups.setdefault((tuple(weekday[m]), tuple(weekend[m])), []).append(m + 1)
    def by_name(row):
        """Hours of each named period (periods sharing a name are merged), cheapest first."""
        present = sorted({names[p] for p in row}, key=lambda nm: ['Off-peak', 'All hours', 'Mid-peak', 'On-peak'].index(nm))
        return {nm: hour_ranges(list(row), {i for i, n2 in enumerate(names) if n2 == nm}) for nm in present}

    month_abbr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    out = []
    for (wd, we), months in groups.items():
        out.append({
            'months': months, 'months_text': ', '.join(month_abbr[m - 1] for m in months),
            'weekdays': by_name(wd), 'weekends': by_name(we),
        })
    return out


def version(raw: dict, delivery: dict | None = None) -> dict:
    """One OpenEI rate as a compact, front-end-friendly version. `delivery`: OpenEI's delivery-only entry for the
    same rate and dates, used to split each period's price into delivery and supply (bundled - delivery)."""
    structure = raw.get('energyratestructure') or []
    tiers = [[{'up_to_kwh_per_month': t.get('max'), 'rate_per_kwh': round(t.get('rate', 0) + t.get('adj', 0), 6)}
              for t in period] for period in structure]
    # One price per period for display and for simple models: the highest tier (what most use is charged at).
    rates = [p[-1]['rate_per_kwh'] for p in tiers]
    names = period_names(rates)
    weekday, weekend = raw['energyweekdayschedule'], raw['energyweekendschedule']
    fixed = raw.get('fixedchargefirstmeter') or 0.0
    units = raw.get('fixedchargeunits', '$/month')
    per_month = fixed * {'$/month': 1, '$/day': 365 / 12, '$/year': 1 / 12}.get(units, 1)
    split = None
    if delivery is not None:
        d_rates = [round(p[-1].get('rate', 0) + p[-1].get('adj', 0), 6) for p in delivery.get('energyratestructure') or []]
        same_hours = (delivery['energyweekdayschedule'] == weekday and delivery['energyweekendschedule'] == weekend
                      and len(d_rates) == len(rates))
        if same_hours:
            split = [(d, round(r - d, 6)) for r, d in zip(rates, d_rates)]
        elif len(d_rates) == 1:            # delivery the same at all hours; only the supply price varies
            split = [(d_rates[0], round(r - d_rates[0], 6)) for r in rates]
    periods = [{'index': i, 'name': names[i], 'rate_per_kwh': rates[i], 'tiers': tiers[i]}
               | ({'delivery_per_kwh': split[i][0], 'supply_per_kwh': split[i][1]} if split else {})
               for i in range(len(rates))]
    return {
        'start': date(raw.get('startdate')), 'end': date(raw.get('enddate')),
        'fixed_charge_per_month': round(per_month, 2),
        'periods': periods,
        'weekday_schedule': weekday, 'weekend_schedule': weekend,   # 12 months x 24 hours (hour 0 = midnight-1 am)
        'hours': describe(weekday, weekend, names),
        'demand_charges': bool(raw.get('demandratestructure') or raw.get('flatdemandstructure')),
        'openei': {'label': raw['label'], 'name': raw['name'], 'uri': raw.get('uri'), 'approved': raw.get('approved'),
                   'delivery_only_label': delivery['label'] if split else None},
        'tariff_source': raw.get('source'),
    }


def build(raws: dict[str, dict]) -> dict:
    utilities = []
    for u in SELECTION:
        plans = []
        for p in u['plans']:
            by_region = p['labels'] if isinstance(p['labels'], dict) else {None: p['labels']}
            for region, labels in by_region.items():
                # Pair each bundled version with the delivery-only entry starting the same day; a plan with a single
                # delivery-only entry (NYSEG: one for all regions) uses it for every version.
                d_labels = p.get('delivery_labels', [])
                dels = {raws[d].get('startdate'): raws[d] for d in d_labels}
                pick = (lambda r: raws[d_labels[0]]) if len(d_labels) == 1 else (lambda r: dels.get(r.get('startdate')))
                versions = sorted((version(raws[lab], pick(raws[lab])) for lab in labels), key=lambda v: v['start'] or '')
                plans.append({k: p[k] for k in ('id', 'name', 'kind', 'eligibility', 'checked')}
                             | {'region': region, 'default': bool(p.get('default')) and region in (None, u.get('default_region')),
                                'versions': versions})
        utilities.append({k: u[k] for k in ('id', 'name', 'full_name', 'state', 'eia_id')}
                         | {k: u[k] for k in ('regions', 'default_region', 'region_note') if k in u}
                         | {'plans': plans})
    return {
        'meta': {
            'description': 'Residential electric rates for the four Avangrid electric utilities, from the OpenEI '
                           'Utility Rate Database, for toggling between utilities in the battery simulator.',
            'source': 'OpenEI Utility Rate Database (https://openei.org/wiki/Utility_Rate_Database)',
            'pulled_on': max(r['_pulled_on'] for r in raws.values()),
            'data_year': {'start': DATA_YEAR[0], 'end': DATA_YEAR[1]},
            'notes': [
                'Rates are bundled: delivery plus the utility\'s default supply, for customers who have not chosen '
                'another supplier. Supply prices change monthly (NYSEG, RG&E), twice a year (UI) or yearly (CMP); '
                'each rate here is the snapshot in OpenEI for the dates shown.',
                'None of these residential rates has a demand charge.',
                'Schedules are 12 months x 24 hours of indices into `periods`, in local clock time (hour 0 = '
                'midnight-1 am), weekdays and weekends separately. Holidays are not distinguished.',
                'Where a rate changed during the data year (CMP and UI on 1 Jan 2026), each plan lists every '
                'version with its start and end dates; the last version is the current one.',
                'Where OpenEI has a delivery-only entry with the same hours, each period also has delivery_per_kwh '
                'and supply_per_kwh (= rate_per_kwh - delivery_per_kwh).',
                'Taxes and most riders are not included.',
            ],
        },
        'utilities': utilities,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--from-cache', action='store_true', help='rebuild from data/tariffs/openei/ without fetching')
    args = ap.parse_args()
    labels = [lab for u in SELECTION for p in u['plans']
              for labs in (p['labels'].values() if isinstance(p['labels'], dict) else [p['labels']]) for lab in labs]
    labels += [lab for u in SELECTION for p in u['plans'] for lab in p.get('delivery_labels', [])]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not args.from_cache:
        key = api_key()
        if not key:
            sys.exit('No OpenEI API key: set OPENEI_API_KEY, or add it to .env (see data/tariffs/README.md), '
                     'or run with --from-cache.')
        today = dt.date.today().isoformat()
        for lab in labels:
            raw = fetch(lab, key) | {'_pulled_on': today}
            (RAW_DIR / f'{lab}.json').write_text(json.dumps(raw, indent=1, sort_keys=True) + '\n')
            print(f'fetched {lab}: {raw["utility"]} - {raw["name"][:70]}')
    raws = {lab: json.loads((RAW_DIR / f'{lab}.json').read_text()) for lab in labels}
    out = build(raws)
    path = OUT_DIR / 'residential_tariffs.json'
    text = json.dumps(out, indent=1)
    text = re.sub(r'\[\s*((?:-?\d+(?:\.\d+)?,\s*)*-?\d+(?:\.\d+)?)\s*\]',      # lists of numbers on one line
                  lambda m: '[' + re.sub(r',\s*', ', ', m.group(1)) + ']', text)
    path.write_text(text + '\n')
    n_plans = sum(len(u['plans']) for u in out['utilities'])
    print(f'wrote {path.relative_to(ROOT)}: {len(out["utilities"])} utilities, {n_plans} plans, '
          f'{len(raws)} OpenEI rates ({path.stat().st_size / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
