"""Step 5f. What does the model need, and does an obvious alternative do better? (questions fixed before results)

Three small experiments, each scored exactly like the main result (average miss on big windows, 2024 to 2026,
never retrained) against the current Level 2, with day-block paired intervals:
  1. ablation: drop one group of inputs at a time;
  2. neighbours: add what stations within 600 m just did;
  3. two flows: forecast arrivals and departures separately (Poisson loss), then subtract.
Only the forecast (mean) model is refitted, with Level 2's settings. Writes reports/model_study.json.
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import bootstrap as bs
import scoring
from config import BACKTEST_YEARS, HORIZONS, INTERIM, PROCESSED, REPORTS, TRAIN_YEARS, TZ
from models import features_for
from s02_stations import to_metres

REPS, SEED = 2000, 20261118
NEIGHBOUR_M = 600
PARAMS = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1.0,
              early_stopping=False, random_state=0)


def groups(h):
    """The model's inputs, in four plain groups."""
    return {
        "recent state": ["net_last1", "net_last3", "net_last6", "dep_last1", "arr_last1", "dep_last3", "arr_last3",
                         "trips_last24", "trips_per_day_7d"],
        "seasonal memory": [f"net_yesterday_h{h}", f"net_lastweek_h{h}", f"net_avg4w_h{h}", f"trips_avg4w_h{h}"],
        "calendar and place": ["hour", "dow", "is_weekend", "is_holiday", "lat", "lon"],
        "weather": [f"temp_h{h}", f"precip_h{h}", f"wind_h{h}"],
    }


def neighbour_matrix(stations, radius=NEIGHBOUR_M):
    """stations: frame with station_id, lat, lon. Returns (ids, boolean matrix: j is within radius of i, j != i)."""
    xy = to_metres(stations.lat.values, stations.lon.values)
    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    near = (d <= radius) & ~np.eye(len(stations), dtype=bool)
    return stations.station_id.values, near


def add_neighbour_features(df, ids, near, cols):
    """For each row, the sum of `cols` over its neighbours at the same hour. Uses only values already in the row
    set for that hour (all of them are computed from the past), so nothing from the future can enter."""
    out = df.copy()
    pos = {s: i for i, s in enumerate(ids)}
    for c in cols:
        wide = df.pivot_table(index="hour_idx", columns="station_id", values=c, aggfunc="first").reindex(columns=ids)
        summed = np.nan_to_num(wide.values) @ near.T.astype(float)     # [hour, i] = sum over neighbours j of i
        summed = pd.DataFrame(summed, index=wide.index, columns=ids)
        out[f"nb_{c}"] = summed.stack().reindex(pd.MultiIndex.from_arrays([df.hour_idx, df.station_id])).values
    return out


def flow_targets(hourly, df, h):
    """Arrivals and departures over the h hours starting at each row's hour (same window as y_h)."""
    out = {}
    for col in ("arr", "dep"):
        wide = hourly.pivot_table(index="hour_idx", columns="station_id", values=col, aggfunc="sum")
        full = wide.reindex(np.arange(wide.index.min(), wide.index.max() + h + 1)).fillna(0.0)
        win = full.rolling(h).sum().shift(-(h - 1))                    # value at t covers t .. t+h-1
        out[col] = win.stack().reindex(pd.MultiIndex.from_arrays([df.hour_idx, df.station_id])).values
    return out["arr"], out["dep"]


def fit_predict(train, test, cols, target, loss="squared_error"):
    m = HistGradientBoostingRegressor(loss=loss, **PARAMS).fit(train[cols], target)
    return m.predict(test[cols])


def paired(test, big, base_pred, other_pred, h):
    """Big-window miss of `other` minus the base, with a day-block interval (negative = other is better)."""
    y = test[f"y_h{h}"].values
    t = test[big]
    frame = pd.DataFrame({"day": t.day.values, "n": 1.0,
                          "base": np.abs(y[big] - base_pred[big]), "other": np.abs(y[big] - other_pred[big])})
    _, m = bs.day_sums(frame, "day", ["n", "base", "other"])
    boot = bs.resampled_totals(m, REPS, SEED)
    tot = m.sum(axis=0)
    diff = bs.interval((tot[2] - tot[1]) / tot[0], (boot[:, 2] - boot[:, 1]) / boot[:, 0])
    rel = bs.interval(tot[2] / tot[1] - 1, boot[:, 2] / boot[:, 1] - 1)
    return {"mae": float(tot[2] / tot[0]), "diff": diff, "relative": rel,
            "verdict": "better" if diff[2] < 0 else "worse" if diff[1] > 0 else "tie"}


def main():
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    hourly = pd.read_parquet(INTERIM / "hourly.parquet", columns=["station_id", "hour_idx", "dep", "arr"])
    metrics = json.loads((REPORTS / "metrics.json").read_text())
    st = panel.groupby("station_id")[["lat", "lon"]].first().reset_index()
    hourly = hourly[hourly.station_id.isin(st.station_id)]            # only the 100 stations, keeps it small
    ids, near = neighbour_matrix(st)
    print(f"neighbours within {NEIGHBOUR_M} m: median {int(np.median(near.sum(axis=1)))} per station")
    R = {"reps": REPS, "seed": SEED, "neighbour_m": NEIGHBOUR_M,
         "median_neighbours": int(np.median(near.sum(axis=1))), "horizons": {}}
    for h in HORIZONS:
        t0 = time.time()
        cols = features_for(h)
        nb_cols = ["net_last1", "net_last3", f"net_avg4w_h{h}"]
        df = panel[["station_id", "hour_idx", "year", f"y_h{h}"] + list(dict.fromkeys(cols + nb_cols))]
        df = df[df[f"y_h{h}"].notna()]
        df = add_neighbour_features(df, ids, near, nb_cols)
        arr, dep = flow_targets(hourly, df, h)
        assert np.allclose(arr - dep, df[f"y_h{h}"].values, equal_nan=True), "arrivals minus departures must equal net flow"
        df = df.assign(arr=arr, dep=dep)
        train, test = df[df.year.isin(TRAIN_YEARS)], df[df.year.isin(BACKTEST_YEARS)].copy()
        test["day"] = pd.to_datetime(test.hour_idx * 3600, unit="s", utc=True).dt.tz_convert(TZ).dt.strftime("%Y-%m-%d").values
        y = test[f"y_h{h}"].values
        cutoff = metrics["horizons"][str(h)]["cutoff"]
        big = scoring.big_imbalance_mask(y, cutoff)
        base = fit_predict(train, test, cols, train[f"y_h{h}"])
        res = {"cutoff": cutoff, "n_big": int(big.sum()), "base_mae": float(np.mean(np.abs(y[big] - base[big]))),
               "ablation": {}, "neighbours": None, "two_flows": None}
        for name, g in groups(h).items():
            keep = [c for c in cols if c not in g]
            res["ablation"][name] = paired(test, big, base, fit_predict(train, test, keep, train[f"y_h{h}"]), h)
        nb = cols + [f"nb_{c}" for c in nb_cols]
        res["neighbours"] = paired(test, big, base, fit_predict(train, test, nb, train[f"y_h{h}"]), h)
        a = fit_predict(train, test, cols, train.arr, loss="poisson")
        d = fit_predict(train, test, cols, train.dep, loss="poisson")
        res["two_flows"] = paired(test, big, base, a - d, h)
        R["horizons"][str(h)] = res
        print(f"{h}h base {res['base_mae']:.2f}  " +
              "  ".join(f"-{k}: {v['relative'][0]:+.1%}" for k, v in res["ablation"].items()) +
              f"  +neighbours {res['neighbours']['relative'][0]:+.1%} ({res['neighbours']['verdict']})"
              f"  two flows {res['two_flows']['relative'][0]:+.1%} ({res['two_flows']['verdict']})  [{time.time() - t0:.0f}s]")
    out = REPORTS / "model_study.json"
    out.write_text(json.dumps(R, indent=1, sort_keys=True))
    print(f"wrote {out.relative_to(out.parents[1])}")


if __name__ == "__main__":
    main()
