# Part 3: demand flexibility and BTM battery planner

`index.html` is the Part 3 front end: a single self-contained page (React build, data embedded). Open it
directly in a browser; no server needed.

Tabs: Portfolio, Battery Program, Customer List, Scenario Analysis, Assumptions, Methodology.

## Data

The committed page carries a **synthetic demonstration dataset** (800 customers) so it can be shared and
viewed without the hackathon data. The banner at the top says so.

Real results come from the pipeline:

1. Notebooks 10-70 build Parts 1 and 2 in `onedrive-export/Data/avangrid.db`.
2. `python scripts/export_part3_inputs.py` writes the real hourly loads, segments, appliance inventory and
   technology scores to `onedrive-export/Data/part3/`.
3. The Part 3 pipeline (`scripts/run_part3.py`, to be added) reads those files and writes `app_data.json`
   with the same structure as the embedded dataset.

Pages built with real customer data stay inside `onedrive-export/Data/` (git-ignored). The challenge
data may not be shared outside the event, so do not commit them.
