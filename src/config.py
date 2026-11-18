"""Every project decision that is a number or a name lives here, in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
WEB_DATA = ROOT / "web" / "data"

TZ = "America/Montreal"

# --- What we predict -------------------------------------------------------
HORIZONS = [1, 3, 6]            # total net bikes over the next 1, 3 and 6 hours
SUMMER_MONTHS = (5, 9)          # May through September
N_STATIONS = 100                # the busiest 100, ranked by total summer trips

# --- Which years do what ---------------------------------------------------
TRAIN_YEARS = [2022, 2023]      # the model only ever learns from these
BACKTEST_YEARS = [2024, 2025, 2026]   # scored, never trained on
VALID_YEAR = 2023               # used once, to pick settings, before the final refit

# --- Station identity ------------------------------------------------------
# Two points closer than this are the same station (covers renames and GPS jitter).
SAME_LOCATION_M = 25
# The same name within this distance is also the same station (a seasonal re-install).
SAME_NAME_M = 150
# Greater Montreal box. Drops the Sherbrooke network and test stations at (0, 0).
BBOX = dict(lat_min=45.30, lat_max=45.80, lon_min=-74.10, lon_max=-73.20)
# A station needs this many active summer days in the training years,
# otherwise the map shows it as "not enough data".
MIN_TRAIN_DAYS = 120

# --- Weather ---------------------------------------------------------------
WEATHER_LAT, WEATHER_LON = 45.51, -73.57      # downtown Montreal, one point for all stations
WEATHER_VARS = ["temperature_2m", "precipitation", "wind_speed_10m"]
WEATHER_MODEL = "ecmwf_ifs"
# A forecast run started at 00:00 or 12:00 UTC is not published instantly.
# We only let the model use a run this many hours after it started.
RUN_DELAY_H = 8
RUN_HOURS_UTC = (0, 12)
RUN_FORECAST_HOURS = 30

# --- Scoring ---------------------------------------------------------------
BIG_IMBALANCE_TOP_SHARE = 0.10  # "big-imbalance" = top 10% of |net|, set per horizon
QUANTILES = (0.1, 0.5, 0.9)     # the band between 0.1 and 0.9 should hold 80% of outcomes
BAND_TARGET = 0.80

# --- Map -------------------------------------------------------------------
MAP_DAYS = 7                    # the map shows the last 7 days of the data
