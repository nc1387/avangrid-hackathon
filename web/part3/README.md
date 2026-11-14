# Part 3: demand flexibility and BTM battery planner

`index.html` is the Part 3 front end: one self-contained page (React build, data embedded). Open it directly
in a browser; no server needed. Tabs: Portfolio, Battery Program, Customer List, Scenario Analysis,
Assumptions, Methodology.

The committed page carries a **synthetic demonstration dataset** (800 customers) so it can be shared without
the hackathon data. The banner at the top says so.

## Build it on the real data

Part 3 uses only Part 1 and Part 2 outputs: the hourly loads cleaned by notebooks 10-20, the appliance
inventory and technology scores (notebooks 50-60), mid-year change flags (notebook 45) and the segments
(notebook 70). It does not re-detect appliances.

```bash
# after notebooks 10-70
python scripts/export_part3_inputs.py      # -> onedrive-export/Data/part3/ (hourly_kwh.npy, hours.csv, customers.csv)
python scripts/run_part3.py                # -> onedrive-export/Data/part3/index.html, app_data.json, candidates.csv
```

Open `onedrive-export/Data/part3/index.html`. The full run takes a few minutes for ~33,000 customers.
`--max-customers 3000` lists only the top 3,000 on the page (summary figures still cover everyone) for a
smaller file.

Assumptions (battery spec, 3 kVA screen, flexibility shares, rates, incentives, finance, score weights) are in
`config/part3_assumptions.json`. The Methodology tab documents each step and formula.

## Part 1 and 2 fields used

| Part 3 input | From |
|---|---|
| p_ev | EV list, else max(EV Level 2, Level 1 confidence) (`appliance_inventory`) |
| p_heat_pump, p_resist_heat | `heating_type` and electric-heat confidences; HP list |
| p_ac, p_pool, water heater | AC, pool, water-heater confidences |
| p_solar, existing storage | PV list, PV model score >= 0.5, solar + battery flag |
| new EV / new solar, onset | `change_points` (notebook 45) |
| segment | `customer_segments.segment` (notebook 70) |

Pages built with real customer data stay inside `onedrive-export/Data/` (git-ignored). The challenge data may
not be shared outside the event, so do not commit them.
