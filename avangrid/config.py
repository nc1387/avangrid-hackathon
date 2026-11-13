"""Paths and constants shared by every notebook.

Paths are resolved from this file's location, so they work whether code runs from the project
folder, a notebook, a script or a background job.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'onedrive-export' / 'Data'
METER_DATA_GLOB = str(DATA_DIR / 'meter-data' / '**' / '*.csv')
DB_PATH = DATA_DIR / 'avangrid.db'
CUSTOMER_GROUPS_PATH = DATA_DIR / 'EV_Heat Pump_Solar PV Customers.xlsx'
WEATHER_PATH = DATA_DIR / 'Weather Data_Avangrid_Hackathon .xlsx'
EDA_DIR = DATA_DIR / 'eda'          # charts and CSV exports
CACHE_DIR = DATA_DIR / 'cache'      # derived arrays; safe to delete, rebuilt on demand
RESULTS_DIR = DATA_DIR / 'results'  # outputs of long-running experiments

# A complete year of hourly data: 365 x 24, minus the hour lost on each daylight-saving
# changeover day in this data (2025-11-02 and 2026-03-08).
EXPECTED_ROWS = 8_758

# Customers whose data folder Avangrid said to disregard (duplicate data).
DISREGARDED_RANGE = (10_001, 20_000)

# Customers averaging less than this (kWh per hour) are left out of peer-relative analysis:
# dividing by an average near zero blows up tiny fluctuations.
MIN_MEAN_KWH = 0.05

N_PEER_GROUPS = 16
TECHS = ('EV', 'HP', 'PV')

# Times of day by hour_ending (hour_ending 1 = 12:00-12:59 am).
TIME_OF_DAY_HOURS = {
    'night': range(1, 7),
    'morning': range(7, 13),
    'afternoon': range(13, 19),
    'evening': range(19, 25),
}
# The same blocks as 0-based slices over a 24-column hour axis (column 0 = hour_ending 1).
TOD_SLICES = {'night': slice(0, 6), 'morning': slice(6, 12), 'afternoon': slice(12, 18), 'evening': slice(18, 24)}
MIDDAY = slice(10, 15)  # hour_ending 11-15, when solar panels generate most

# Weather balance points (deg F) used for heating and cooling degree terms.
HEATING_BASE_F = 55
COOLING_BASE_F = 65
