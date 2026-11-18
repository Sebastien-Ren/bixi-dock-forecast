#!/bin/sh
# Rebuild everything from the raw trip files. About 10 minutes on a laptop.
# The weather step needs internet the first time; after that it reads its cache.
set -e
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
PY=.venv/bin/python
[ -x "$PY" ] || { python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt; }

$PY -m pytest -q                 # scoring and no-leakage tests must pass before anything runs
$PY src/s01_trips.py             # raw CSVs        -> one clean trip table
$PY src/s02_stations.py          # names + coords  -> stations, hourly counts, busiest 100
$PY src/s03_weather.py           # Open-Meteo      -> actual weather + forecast runs as issued
$PY src/s03b_capacity.py          # BIXI station list -> docks per station (cached; exploratory view only)
$PY src/s04_features.py          # everything      -> one row per station per hour
$PY src/s05_backtest.py          # train 2022-23   -> forecasts for every hour of 2024-26
$PY src/s05b_retrain_check.py     # would retraining each year help?  -> reports/retrain_check.json
$PY src/s05c_weather_check.py     # how much does weather matter?     -> reports/weather_check.json
$PY src/s05d_decision_check.py   # forecasts -> which K stations to act on, with intervals -> reports/decision_check.json
$PY src/s05e_availability_check.py # past station counts (2025) -> empty/full probability -> reports/availability_check.json
                                 # needs: $PY src/archive/build_inventory.py 2025-05-01 2025-06-01 (and each month to 2025-10-01); skips if absent
$PY src/s05f_model_study.py       # ablation, neighbours, two flows -> reports/model_study.json (about 20 min)
$PY src/s05g_decision_study.py    # which 10 stations to check, on real 2025 counts -> reports/decision_study.json
$PY src/s05h_confirmation.py      # sealed one-time rerun of s05e/s05g on September 2024 -> reports/confirmation_2024_09.json
                                 # needs: $PY src/archive/build_inventory.py 2024-09-01 2024-10-01; never reruns once its file exists (plan/07)
$PY src/s06_report.py            # forecasts       -> reports/RESULTS.md and metrics.json
$PY src/s07_export_map.py        # last week       -> web/data/forecast.json
echo "Done. Run ./run_map.sh to open the map."
