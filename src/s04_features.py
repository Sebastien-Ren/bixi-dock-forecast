"""Step 4. Build the table the models learn from: one row per station per hour.

Vocabulary used everywhere below:
  origin t   = the hour the forecast is made for. Everything before hour t is known.
  window     = the next h hours: t, t+1, ... t+h-1   (h is 1, 3 or 6)
  target y_h = net bikes (arrivals minus departures) summed over the window

The golden rule: a feature for origin t may only use hours before t.
tests/test_no_leakage.py checks that rule by changing the future and confirming no feature moves.
"""
import warnings

import duckdb
import holidays
import numpy as np
import pandas as pd

from config import (BACKTEST_YEARS, HORIZONS, INTERIM, MIN_TRAIN_DAYS, PROCESSED, RUN_DELAY_H,
                    SUMMER_MONTHS, TRAIN_YEARS, TZ)

WEEK = 168


def window_sum(x, start_offset, length):
    """For every origin t, the sum of x over hours [t + start_offset, t + start_offset + length).
    x has shape (stations, hours). Positions that fall outside the data are NaN."""
    n = x.shape[1]
    c = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(x, axis=1)], axis=1)
    a = np.arange(n) + start_offset
    b = a + length
    ok = (a >= 0) & (b <= n)
    out = np.full(x.shape, np.nan)
    out[:, ok] = c[:, b[ok]] - c[:, a[ok]]
    return out


def nanmean(stack):
    """Mean over the first axis, ignoring NaN. All-NaN gives NaN (without the numpy warning)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(stack, axis=0)


def traffic_features(dep, arr, active):
    """All features and targets that come from the trip counts.
    dep, arr: (stations, hours) counts. active: (stations, hours) True when the station was in service."""
    net = arr - dep
    trips = arr + dep
    f = {}
    # What just happened at this station.
    f["net_last1"] = window_sum(net, -1, 1)
    f["net_last3"] = window_sum(net, -3, 3)
    f["net_last6"] = window_sum(net, -6, 6)
    f["dep_last1"] = window_sum(dep, -1, 1)
    f["arr_last1"] = window_sum(arr, -1, 1)
    f["dep_last3"] = window_sum(dep, -3, 3)
    f["arr_last3"] = window_sum(arr, -3, 3)
    f["trips_last24"] = window_sum(trips, -24, 24)
    f["trips_per_day_7d"] = window_sum(trips, -WEEK, WEEK) / 7

    # What this station did in the same window on earlier days. Hours when the station
    # was out of service are skipped, not counted as zero.
    def past_week(x, k, h):
        total = window_sum(np.where(active, x, 0.0), -WEEK * k, h)
        any_off = window_sum((~active).astype(float), -WEEK * k, h) > 0
        return np.where(any_off, np.nan, total)

    for h in HORIZONS:
        f[f"net_yesterday_h{h}"] = window_sum(net, -24, h)
        weeks_net = np.stack([past_week(net, k, h) for k in (1, 2, 3, 4)])
        weeks_trips = np.stack([past_week(trips, k, h) for k in (1, 2, 3, 4)])
        f[f"net_lastweek_h{h}"] = weeks_net[0]
        f[f"net_avg4w_h{h}"] = nanmean(weeks_net)
        f[f"trips_avg4w_h{h}"] = nanmean(weeks_trips)
        # The answer we are trying to predict. Uses hours t .. t+h-1, so it is never a feature.
        f[f"y_h{h}"] = window_sum(net, 0, h)
    return f


def weather_features(hours, actual, runs=None):
    """Weather over each window. `hours` are the origins.
    actual: what the weather was. runs: forecast runs as issued (backtest years only).
    Returns columns <var>_h<h> (what the model may use) and <var>_act_h<h> (what really happened)."""
    out = {}
    act = actual.set_index("hour_idx").reindex(np.arange(hours[0], hours[-1] + max(HORIZONS) + 1))
    for h in HORIZONS:
        for var, how in (("temp", "mean"), ("precip", "sum"), ("wind", "mean")):
            roll = act[var].rolling(h).agg(how).shift(-(h - 1))      # value at t covers t .. t+h-1
            out[f"{var}_act_h{h}"] = roll.reindex(hours).values
    if runs is None:
        # Training years: there are no archived forecast runs, so the model learns from actual weather.
        for h in HORIZONS:
            for var in ("temp", "precip", "wind"):
                out[f"{var}_h{h}"] = out[f"{var}_act_h{h}"]
        return out

    # Backtest years: for each origin, use the newest run that had been published by then.
    run_ids = np.sort(runs.run_idx.unique())
    pos = np.searchsorted(run_ids, hours - RUN_DELAY_H, side="right") - 1    # newest run with run + delay <= t
    chosen = np.where(pos >= 0, run_ids[np.clip(pos, 0, None)], -1)
    table = runs.set_index(["run_idx", "hour_idx"])
    for h in HORIZONS:
        per_hour = {var: [] for var in ("temp", "precip", "wind")}
        for k in range(h):
            idx = pd.MultiIndex.from_arrays([chosen, hours + k])
            got = table.reindex(idx)
            for var in per_hour:
                per_hour[var].append(got[var].values)
        out[f"temp_h{h}"] = nanmean(np.stack(per_hour["temp"]))
        out[f"precip_h{h}"] = np.nansum(np.stack(per_hour["precip"]), axis=0)
        out[f"wind_h{h}"] = nanmean(np.stack(per_hour["wind"]))
        out[f"precip_h{h}"][np.isnan(out[f"temp_h{h}"])] = np.nan
    out["run_age_h"] = np.where(chosen >= 0, hours - chosen, np.nan)
    return out


def main():
    top = pd.read_csv(PROCESSED / "top_stations.csv")
    ids = top.station_id.values
    con = duckdb.connect()
    hourly = con.execute(f"""
        select station_id, hour_idx, dep, arr from '{INTERIM}/hourly.parquet'
        where station_id in ({",".join(map(str, ids))})
    """).df()
    last_departure_hour = int(hourly.hour_idx[hourly.dep > 0].max())
    actual = pd.read_parquet(INTERIM / "weather_actual.parquet")
    runs = pd.read_parquet(INTERIM / "weather_runs.parquet")
    qc_holidays = holidays.Canada(subdiv="QC", years=TRAIN_YEARS + BACKTEST_YEARS)

    frames = []
    for year in TRAIN_YEARS + BACKTEST_YEARS:
        # A dense grid of hours: mid-April (4 weeks of history before May) to early October.
        t0 = int(pd.Timestamp(f"{year}-04-13", tz=TZ).timestamp()) // 3600
        t1 = int(pd.Timestamp(f"{year}-10-03", tz=TZ).timestamp()) // 3600
        t1 = min(t1, last_departure_hour)       # the data ends when trips stop starting
        hours = np.arange(t0, t1 + 1)
        sub = hourly[hourly.hour_idx.between(t0, t1)]
        row = pd.Index(ids).get_indexer(sub.station_id)
        col = (sub.hour_idx - t0).values
        dep = np.zeros((len(ids), len(hours)))
        arr = np.zeros((len(ids), len(hours)))
        dep[row, col], arr[row, col] = sub.dep.values, sub.arr.values

        local = pd.to_datetime(hours * 3600, unit="s", utc=True).tz_convert(TZ)
        # In service = at least one trip started or ended there that local day.
        day_code = pd.factorize(local.date)[0]
        day_trips = np.stack([np.bincount(day_code, weights=dep[s] + arr[s]) for s in range(len(ids))])
        active = (day_trips > 0)[:, day_code]

        f = traffic_features(dep, arr, active)
        w = weather_features(hours, actual, runs if year in BACKTEST_YEARS else None)

        in_summer = (local.month >= SUMMER_MONTHS[0]) & (local.month <= SUMMER_MONTHS[1])
        s_idx, h_idx = np.nonzero(active & in_summer[None, :])
        df = pd.DataFrame({
            "station_id": ids[s_idx],
            "hour_idx": hours[h_idx],
            "year": year,
            "hour": local.hour.values[h_idx],
            "dow": local.day_of_week.values[h_idx],
            "doy": local.day_of_year.values[h_idx],
            "is_holiday": np.array([d in qc_holidays for d in local.date])[h_idx].astype(int),
        })
        df["hour_of_week"] = df.dow * 24 + df.hour
        df["is_weekend"] = (df.dow >= 5).astype(int)
        for name, a in f.items():
            df[name] = a[s_idx, h_idx]
        for name, a in w.items():
            df[name] = a[h_idx]
        frames.append(df)
        print(f"{year}: {len(df):>9,} station-hours in service, "
              f"{(~active[:, in_summer]).mean():.1%} of station-hours out of service")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.merge(top[["station_id", "lat", "lon"]], on="station_id")
    panel = panel.dropna(subset=[f"y_h{max(HORIZONS)}"])        # window must fit inside the data

    # "Enough data" flag: active summer days in the training years.
    train_days = (panel[panel.year.isin(TRAIN_YEARS)]
                  .assign(day=lambda d: d.hour_idx // 24).groupby("station_id").day.nunique())
    top["train_days"] = top.station_id.map(train_days).fillna(0).astype(int)
    top["enough_data"] = top.train_days >= MIN_TRAIN_DAYS
    top.to_csv(PROCESSED / "top_stations.csv", index=False)

    panel.to_parquet(PROCESSED / "panel.parquet", index=False)
    print(f"\npanel: {len(panel):,} rows, {panel.shape[1]} columns -> data/processed/panel.parquet")
    print(f"stations with enough training data: {int(top.enough_data.sum())} of {len(top)}")
    bt = panel[panel.year.isin(BACKTEST_YEARS)]
    print(f"backtest rows missing a forecast run: {bt.temp_h1.isna().mean():.2%}; "
          f"typical run age at forecast time: {bt.run_age_h.median():.0f} h (max {bt.run_age_h.max():.0f} h)")


if __name__ == "__main__":
    main()
