"""Step 3. Download weather from Open-Meteo (free, no key). One point: downtown Montreal.

Two different things are downloaded, on purpose:

  weather_actual.parquet   what the weather turned out to be, hour by hour (2022 to 2026).
                           Used to TRAIN the model on 2022 and 2023.
  weather_runs.parquet     the real ECMWF forecast runs as they were issued (2024 onward),
                           two per day (00:00 and 12:00 UTC), 30 hours each.
                           Used in the BACKTEST, so the model never sees weather
                           it could not have known at the time.

Every download is cached in data/interim/weather_cache, so re-running is instant.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

from config import (BACKTEST_YEARS, INTERIM, RUN_FORECAST_HOURS, RUN_HOURS_UTC, TRAIN_YEARS,
                    WEATHER_LAT, WEATHER_LON, WEATHER_MODEL, WEATHER_VARS)

CACHE = INTERIM / "weather_cache"
COMMON = dict(latitude=WEATHER_LAT, longitude=WEATHER_LON, hourly=",".join(WEATHER_VARS),
              models=WEATHER_MODEL, timezone="GMT")
RENAME = {"temperature_2m": "temp", "precipitation": "precip", "wind_speed_10m": "wind"}
LAST_DATA_DAY = date(2026, 8, 31)


def get(url, params, cache_name):
    """GET with a file cache and a few retries. Returns the JSON, or None if the run does not exist."""
    f = CACHE / cache_name
    if f.exists():
        return json.loads(f.read_text())
    last = "no response"
    for attempt in range(8):
        try:
            r = requests.get(url, params=params, timeout=20)
        except requests.RequestException as e:      # the service sometimes stalls: try again
            last = repr(e)
            continue
        if r.status_code == 200:
            f.write_text(r.text)
            return r.json()
        if r.status_code == 400 and "not available" in r.text:
            f.write_text("null")
            return None
        last = f"{r.status_code} {r.text[:200]}"
        time.sleep(2 + 5 * attempt)                 # rate limit: wait and retry
    raise RuntimeError(f"weather download failed: {last}")


def to_frame(js):
    df = pd.DataFrame(js["hourly"]).rename(columns=RENAME)
    t = pd.to_datetime(df.pop("time"), utc=True)
    df.insert(0, "hour_idx", (t.astype("int64") // 3_600_000_000_000 if t.dt.unit == "ns"
                              else t.astype("int64") // (3600 * {"s": 1, "ms": 1000, "us": 1_000_000}[t.dt.unit])))
    return df


def main():
    CACHE.mkdir(parents=True, exist_ok=True)

    # 1) What the weather actually was. April to mid-October covers summer plus a margin.
    frames = []
    for y in TRAIN_YEARS + BACKTEST_YEARS:
        end = min(date(y, 10, 10), LAST_DATA_DAY + timedelta(days=2))
        js = get("https://archive-api.open-meteo.com/v1/archive",
                 dict(COMMON, start_date=f"{y}-04-10", end_date=end.isoformat()), f"actual_{y}.json")
        frames.append(to_frame(js))
    actual = pd.concat(frames).dropna()
    actual.to_parquet(INTERIM / "weather_actual.parquet", index=False)
    print(f"actual weather: {len(actual):,} hours")

    # 2) The forecast runs as issued, for the backtest years. Four downloads at a time.
    wanted = []
    for y in BACKTEST_YEARS:
        day, end = date(y, 4, 28), min(date(y, 10, 1), LAST_DATA_DAY)
        while day <= end:
            wanted += [datetime(day.year, day.month, day.day, h, tzinfo=timezone.utc) for h in RUN_HOURS_UTC]
            day += timedelta(days=1)

    def fetch(run):
        return get("https://single-runs-api.open-meteo.com/v1/forecast",
                   dict(COMMON, run=f"{run:%Y-%m-%dT%H:%M}", forecast_hours=RUN_FORECAST_HOURS),
                   f"run_{run:%Y%m%d%H}.json")

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, wanted))
    frames, missing = [], []
    for run, js in zip(wanted, results):
        if js is None:
            missing.append(run)
            continue
        df = to_frame(js)
        df.insert(0, "run_idx", int(run.timestamp()) // 3600)
        frames.append(df)
    runs = pd.concat(frames)
    runs["lead_h"] = runs.hour_idx - runs.run_idx
    runs.to_parquet(INTERIM / "weather_runs.parquet", index=False)
    print(f"forecast runs: {runs.run_idx.nunique():,} runs downloaded, {len(missing)} missing")
    if missing:
        print("  missing runs (the previous run is used instead):", [f"{m:%Y-%m-%d %HZ}" for m in missing[:20]])


if __name__ == "__main__":
    main()
