# Part 3: demand flexibility and BTM battery planner

`index.html` is the Part 3 front end: one self-contained page (React build, data embedded). Open it directly
in a browser; no server needed. Tabs: Portfolio, Cluster Analysis, Battery Program, Customer List, Scenario
Analysis, Assumptions.

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

Assumptions (battery spec, 3 kVA screen, flexibility shares, rates, finance, adoption, score weights) are in
`config/part3_assumptions.json`.

**Cluster Analysis tab** (second from the left): notebook 70's eight behavioural clusters, with a short description,
size and share of customers, battery candidates, the average of each home's median and 95th percentile hourly kWh over
the year, the cluster's combined peak in summer and winter (its homes' load added up, averaged over the season's 20
highest system days, as in the Portfolio tab's peak-day chart, at the highest hour; today's load, before flexibility
or batteries), and the share of homes flagged for each appliance (customer appliance classification). Built by `cluster_summary` in
`scripts/run_part3.py`.

**Adopter groups.** A battery candidate whose year-1 savings (battery bill savings plus the performance incentive)
exceed the likely adoption threshold ($100 a year) is a **likely adopter**; the rest are **less likely adopters** (`adoption` in the config). The Battery Program
tab plots every candidate's year-1 savings (customer's perspective) against its projected peak reduction, with dotted
lines at the likely adoption threshold and at the 0.5 kW battery candidate threshold (`screening.min_battery_cp_kw`,
a number we set). The Customer List tags each candidate with its group. The threshold and each group's acceptance rate
(40% and 20%) are set on the Scenario Analysis tab, and changing the threshold moves the line and the tags.

**Adoption (Portfolio tab).** The peak-day chart has a third, dashed line: the system load at expected adoption, the
average of 100 random runs with a fixed seed. In each run every candidate takes a battery with its group's
probability (the adopters are dispatched as their own fleet), and each home offered flexibility enrolls with
probability 50%. The other line shows 100% adoption. The top line's upfront investment (battery hardware +
installation) and Project NPV (the avoided capacity value at the binding season's peak hours, × $/kW-yr, over the
battery life at the discount rate, declining with degradation, less the upfront investment) are for expected adoption
and follow the Assumptions tab, as does the lower annual system peak (the average of the runs' peaks). The
Portfolio figures use the config's thresholds and rates; the Scenario Analysis inputs don't change them.

**Battery candidate threshold used by the Portfolio and Battery Program tabs:** the one that maximizes project NPV
or, when no threshold makes batteries pay, the lowest one at which NPV is still at least zero (0.5 kW minimum).
`scripts/run_part3.py` finds it with the default inputs (`npv_threshold`) and rebuilds the fleet with the homes that
clear it, so the peak-day chart, the annual peak and the Customer List match it; the page recalculates it, the
candidates, upfront investment and NPV whenever an input is applied, and notes when the chart and peak are still for
the threshold it was built with. With the defaults on NYSEG no threshold makes batteries pay, and 1.315 kW (293
candidates) is the lowest that keeps NPV at or above zero. The Portfolio tab notes that NPV counts only battery
hardware, installation and avoided capacity value, not other things Avangrid may value such as grid flexibility or
lower carbon emissions.

**Fleet dispatch, second stage.** Each day's linear program first minimizes the peak, then, keeping that peak, moves
charging to the lowest-load hours and discharging to the highest. Without it a small fleet could charge during other
peak-window hours, since only the single highest hour counted.

**NPV-maximizing battery candidate threshold** (Scenario Analysis). Recalculated whenever an Assumptions or Scenario Analysis input is applied. For each threshold, the
candidates are the fleet's homes whose one unit cuts their peak-hour load by at least that much in the per-home
simulation (`battery_points` in the summary carries what the page needs). Each counts at its group's acceptance rate:
the present value of its battery's reduction at the system peak hours, times the avoided capacity value, less its
installed cost. Flexibility adds the same value at every threshold, and the performance incentive, a transfer to the
customer, isn't counted. Thresholds below the configured one aren't tested. At $100/kW-yr no threshold makes batteries
pay; they start to above roughly $1,100/kW-yr. **Scenario Analysis** inputs, like the Assumptions tab's, take effect
when **Re-Calculate** is clicked.

**Performance incentive.** $200/kW of summer and $25/kW of winter peak-hour reduction a year for years 1-5, then
$115 and $15 for years 6-10 (`economics` in the config; editable on the Assumptions tab). The utility pays it to the
customer as a discount amortized over their monthly bills, and it counts in the customer's year-1 savings.

**Removed:** the separate
"Enroll existing battery" action and program, the IRR and rebate columns, the rebate figures on the Portfolio and
Battery Program tabs, and the Methodology tab.

**Scenario Analysis inputs** (`scenario_inputs` in the summary) are measured over every customer, not just those
listed on the page: today's load at the binding season's peak hours, the peak-hour load a new heat pump or EV adds
(median of likely heat-pump or EV homes minus the median of homes with no electric heat, EV or solar), how many homes
could add each, and the flexibility relief. On NYSEG's data a heat pump adds 0.59 kW at the summer peak hours and
3.70 kW at the winter ones. The battery bar still draws only on the battery candidates listed on the page.

**Assumptions tab.** Edits don't take effect until **Re-Calculate** is clicked; it then reruns the financial model and
the ranking for every page. **Discard changes** goes back to the inputs in use.

**Fleet dispatch.** With `fleet.optimize` on (the default), the battery candidates are dispatched as one fleet: a
daily linear program sets the fleet's charging and discharging to minimize the system peak, and the number of
batteries is the knee of the peak-versus-fleet-size curve (the smallest fleet within `knee_share`, 95%, of the best
achievable peak). On the real data that is 7,667 of 15,034 candidate homes, and the annual peak falls from 78.0 to
59.8 MW (74.8 MW when every candidate charges in the same hours). Set `optimize` to false for per-home dispatch.
Only the fleet is optimized: whether a home gets flexibility only, one unit or two is set by the fixed thresholds in
`screening` (step 1), and the optimization then picks which candidates join the fleet. The ranking implies a cutoff:
on the real data, a candidate joins if its batteries can cover at least 2.07 kW at the system-peak hour (averaged over
the 60 highest days). Notebook 99, section 3, sets out the rules, the linear program and the cutoff.

**Real utility rates.** `python scripts/run_part3.py --utility cmp` (or `nyseg`, `rge`, `ui`) replaces the config's
rates with that utility's residential time-varying rate from `data/tariffs/residential_tariffs.json` (NYSEG uses its
Day/Night rate, SC 8, since its time-of-use rate needs 35,000 kWh a year). Delivery and supply are combined: the
page's supply price is the rate's off-peak supply, and on- and off-peak prices carry the rest, so every hour's
total matches the utility's rate. Notebook 99 builds one page per utility into `onedrive-export/Data/part3/<utility>/`
and a starting page, `onedrive-export/Data/part3/index.html`, with a drop-down to switch between them (NYSEG is shown first).

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
