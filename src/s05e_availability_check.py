"""Step 5e. Will a station be empty or full? Scored on real station counts from summer 2025.

The questions, populations, forecasters, baselines and the rule for "beats" were fixed before any result was
seen (dated plan notes). This is a retrospective test on past data, not a prospective one.

Reads  data/interim/inventory_2025-*.parquet (past snapshots of BIXI's station feed, from
       src/archive/build_inventory.py), data/processed/predictions.parquet (Level 2, never trained on 2025),
       and the project's station list (to match feed stations to ours).
Writes reports/availability_check.json. Skips quietly if the snapshots have not been downloaded.
"""
import json
import sys

import numpy as np
import pandas as pd

import availability as av
import bootstrap as bs
from config import HORIZONS, INTERIM, PROCESSED, REPORTS, TZ
from live.station_map import match_stations

YEAR = 2025
TRAIN_MONTHS, TEST_MONTHS = (5, 6, 7), (8, 9)
REPS, SEED = 2000, 20261117
EVENTS = {"empty": ("bikes0", "is_empty1", "is_empty0"), "full": ("docks0", "is_full1", "is_full0")}
START_SLICES = [(1, 2), (3, 5), (6, 10_000)]
HOUR_BLOCKS = [(0, 5, "00-06"), (6, 9, "06-10"), (10, 14, "10-15"), (15, 19, "15-20"), (20, 23, "20-24")]


def load_snapshots(pattern=f"inventory_{YEAR}-*.parquet"):
    files = sorted(INTERIM.glob(pattern))
    if not files:
        return None, []
    return [pd.read_parquet(f) for f in files], files


def match_months(frames, ours):
    """Feed station id -> our station id, matched separately for each monthly file (stations move and get
    renamed), using each month's first snapshot and the project's usual rule."""
    out = []
    for df in frames:
        first = df[df.snap_time == df.snap_time.min()][["station_id", "name", "lat", "lon"]].reset_index(drop=True)
        m = match_stations(ours, first)
        pairs = pd.DataFrame({"our_id": ours.station_id.values,
                              "station_id": [None if j is None else first.station_id.iloc[j] for j in m]}).dropna()
        sub = df[df.station_id.isin(pairs.station_id)].merge(pairs, on="station_id")
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def station_states(snap):
    """One row per (our station, hour mark): the snapshot nearest the start of that hour (within 10 min)."""
    secs = np.sort(snap.snap_time.map(lambda t: t.timestamp()).astype(np.int64).unique())
    hours = np.arange(secs[0] // 3600, secs[-1] // 3600 + 2)
    pick = av.nearest_snapshot(hours * 3600, secs)
    marks = pd.DataFrame({"hour_idx": hours[pick >= 0], "snap_s": secs[pick[pick >= 0]]})
    s = snap.assign(snap_s=snap.snap_time.map(lambda t: t.timestamp()).astype(np.int64))
    s = s.merge(marks, on="snap_s")
    s = s.drop_duplicates(["our_id", "hour_idx"])
    return s[["our_id", "hour_idx", "num_bikes_available", "num_bikes_disabled", "num_docks_available",
              "is_installed", "is_renting", "is_returning", "capacity"]], len(hours), int((pick >= 0).sum())


def build_pairs(states, preds):
    """Join every Level 2 forecast to the station count at the start and at the end of its window."""
    st = states.rename(columns={"our_id": "station_id"})
    start = st.add_suffix("0").rename(columns={"station_id0": "station_id", "hour_idx0": "hour_idx"})
    end = st.add_suffix("1").rename(columns={"station_id1": "station_id", "hour_idx1": "end_idx"})
    d = preds.assign(end_idx=preds.hour_idx + preds.h)
    n_all = len(d)
    d = d.merge(start, on=["station_id", "hour_idx"]).merge(end, on=["station_id", "end_idx"])
    n_both = len(d)
    d["bikes0"], d["docks0"] = d.num_bikes_available0, d.num_docks_available0
    d["is_empty0"] = (d.num_bikes_available0 == 0).astype(int)
    d["is_full0"] = (d.num_docks_available0 == 0).astype(int)
    d["is_empty1"] = (d.num_bikes_available1 == 0).astype(int)
    d["is_full1"] = (d.num_docks_available1 == 0).astype(int)
    local = pd.to_datetime(d.hour_idx * 3600, unit="s", utc=True).dt.tz_convert(TZ)
    d["local_hour"], d["month"], d["day"] = local.dt.hour.values, local.dt.month.values, local.dt.strftime("%Y-%m-%d").values
    return d, {"forecasts_2025": int(n_all), "with_both_snapshots": int(n_both)}


def in_service(d, event):
    """The event is only defined while the station can rent (empty) or take returns (full), at both ends."""
    flag = "is_renting" if event == "empty" else "is_returning"
    return (d.is_installed0 == 1) & (d.is_installed1 == 1) & (d[f"{flag}0"] == 1) & (d[f"{flag}1"] == 1)


def compare(test, cols, y_col, baselines, forecasters):
    """Brier scores with day-block intervals, and paired differences (baseline minus forecaster:
    above zero means the forecaster is better)."""
    y = test[y_col].values
    se = pd.DataFrame({"day": test.day.values, "n": 1.0})
    for c in cols:
        se[c] = (test[c].values - y) ** 2
    days, m = bs.day_sums(se, "day", ["n"] + cols)
    boot = bs.resampled_totals(m, REPS, SEED)
    tot, n, bn = m.sum(axis=0), m[:, 0].sum(), boot[:, 0]
    k = {c: i + 1 for i, c in enumerate(cols)}
    out = {"n": int(n), "days": int(len(days)), "event_rate": float(y.mean()), "brier": {}, "paired": {}}
    for c in cols:
        out["brier"][c] = bs.interval(tot[k[c]] / n, boot[:, k[c]] / bn)
    for f in forecasters:
        for b in baselines:
            diff = bs.interval((tot[k[b]] - tot[k[f]]) / n, (boot[:, k[b]] - boot[:, k[f]]) / bn)
            skill = bs.interval(1 - tot[k[f]] / tot[k[b]], 1 - boot[:, k[f]] / boot[:, k[b]])
            out["paired"][f"{f}_vs_{b}"] = {"diff": diff, "skill": skill, "verdict": bs.verdict(diff[1], diff[2])}
    return out


def event_windows(d, event):
    """In-service windows for one event, with the raw chance read from Level 2's range and the persistence guess."""
    count_col, _, now_col = EVENTS[event]
    e = d[in_service(d, event)].copy()
    direct = av.prob_empty_direct if event == "empty" else av.prob_full_direct
    e["p_direct"] = direct(e[count_col], e.l2, e.l2_lo, e.l2_hi)
    e["p_persist"] = e[now_col].astype(float)
    return e


def fit_event(tr, event):
    """The only fitted parts: the starting-count rate table, and the recalibration on the headline population
    (stations not already empty / full). Both learned from the training windows `tr` of one horizon."""
    count_col, y_col, now_col = EVENTS[event]
    rate = av.rate_table(tr, count_col, y_col)
    cal_fit = tr[tr[now_col] == 0]
    return rate, av.recalibrate(cal_fit.p_direct.values, cal_fit[y_col].values)


def score_event(te, event, rate, cal):
    """Score one horizon's test windows with already-fitted parts. Fits nothing."""
    count_col, y_col, now_col = EVENTS[event]
    te = te.copy()
    te["p_rate"] = rate(te)
    te["p_cal"] = cal(te.p_direct.values)
    head = te[te[now_col] == 0]
    res = {"headline": compare(head, ["p_rate", "p_direct", "p_cal"], y_col,
                               baselines=["p_rate"], forecasters=["p_direct", "p_cal"]),
           "all": compare(te, ["p_persist", "p_rate", "p_direct", "p_cal"], y_col,
                          baselines=["p_persist", "p_rate"], forecasters=["p_direct", "p_cal"]),
           "calibration": {"p_direct": av.calibration_table(head.p_direct, head[y_col]),
                           "p_cal": av.calibration_table(head.p_cal, head[y_col]),
                           "p_rate": av.calibration_table(head.p_rate, head[y_col])},
           "by_start": {}, "by_hour_block": {}}
    for a, b in START_SLICES:
        s = head[head[count_col].between(a, b)]
        if len(s):
            res["by_start"][f"{a}-{b if b < 10_000 else '+'}"] = compare(
                s, ["p_rate", "p_cal"], y_col, baselines=["p_rate"], forecasters=["p_cal"])
    for a, b, name in HOUR_BLOCKS:
        s = head[head.local_hour.between(a, b)]
        if len(s):
            res["by_hour_block"][name] = compare(s, ["p_rate", "p_cal"], y_col, baselines=["p_rate"],
                                                 forecasters=["p_cal"])
    return res


def load_states(frames):
    """Match feed stations to ours and take one count per station per hour mark."""
    stations = pd.read_parquet(INTERIM / "stations.parquet")
    top = pd.read_csv(PROCESSED / "top_stations.csv")
    ours = stations[stations.station_id.isin(top.station_id)].reset_index(drop=True)
    snap = match_months(frames, ours)
    states, n_marks, n_ok = station_states(snap)
    return states, {"matched_stations": int(snap.our_id.nunique()), "of_stations": int(len(ours)),
                    "hour_marks": n_marks, "hour_marks_with_snapshot": n_ok}


def truck_effect(d):
    """Real change in bikes at the station (available + disabled) against the trip-based net flow, same window.
    Uses every matched window, May to September; nothing here is fitted."""
    ok = (d.is_installed0 == 1) & (d.is_installed1 == 1)
    x = d[ok]
    inv = (x.num_bikes_available1 + x.num_bikes_disabled1) - (x.num_bikes_available0 + x.num_bikes_disabled0)
    gap = inv - x.y
    out = {}
    for h in HORIZONS:
        m = (x.h == h).values
        g, i, y = gap[m], inv[m], x.y[m]
        by_block = {}
        for a, b, name in HOUR_BLOCKS:
            mb = m & x.local_hour.between(a, b).values
            by_block[name] = {"n": int(mb.sum()), "mean_gap": float(gap[mb].mean()), "mean_abs_gap": float(gap[mb].abs().mean())}
        out[str(h)] = {"n": int(m.sum()), "corr": float(np.corrcoef(i, y)[0, 1]),
                       "mean_abs_trip_net": float(y.abs().mean()), "mean_abs_gap": float(g.abs().mean()),
                       "mean_gap": float(g.mean()), "share_gap_5plus": float((g.abs() >= 5).mean()),
                       "by_hour_block": by_block}
    return out


def main():
    frames, files = load_snapshots()
    if frames is None:
        print("no past snapshots in data/interim (run src/archive/build_inventory.py); skipping")
        return
    stations = pd.read_parquet(INTERIM / "stations.parquet")
    top = pd.read_csv(PROCESSED / "top_stations.csv")
    ours = stations[stations.station_id.isin(top.station_id)].reset_index(drop=True)
    snap = match_months(frames, ours)
    matched = sorted(snap.our_id.unique().tolist())
    states, n_marks, n_marks_ok = station_states(snap)
    print(f"snapshot files: {[f.name for f in files]}")
    print(f"matched stations: {len(matched)} of {len(ours)}; hour marks with a snapshot: {n_marks_ok} of {n_marks}")

    preds = pd.read_parquet(PROCESSED / "predictions.parquet",
                            columns=["station_id", "hour_idx", "year", "h", "y", "l2", "l2_lo", "l2_hi"])
    preds = preds[preds.year == YEAR]
    d, counts = build_pairs(states, preds)
    print(f"forecast windows with both snapshots: {counts['with_both_snapshots']:,} of {counts['forecasts_2025']:,}")

    R = {"source": "past snapshots of BIXI's station feed (Max Halford's bike-sharing-history), summer 2025",
         "retrospective": True, "train_months": TRAIN_MONTHS, "test_months": TEST_MONTHS,
         "matched_stations": len(matched), "of_stations": int(len(ours)), "hour_marks": n_marks,
         "hour_marks_with_snapshot": n_marks_ok, "counts": counts, "events": {}}

    for event, (count_col, y_col, now_col) in EVENTS.items():
        e = event_windows(d, event)
        train, test = e[e.month.isin(TRAIN_MONTHS)], e[e.month.isin(TEST_MONTHS)]
        R["events"][event] = {"windows_in_service": int(len(e)), "train_rows": int(len(train)), "test_rows": int(len(test)),
                              "horizons": {}}
        for h in HORIZONS:
            tr, te = train[train.h == h], test[test.h == h]
            rate, cal = fit_event(tr, event)                              # learned on training months only
            res = score_event(te, event, rate, cal)
            R["events"][event]["horizons"][str(h)] = res
            hb = res["headline"]
            print(f"{event:5s} {h}h  test n={hb['n']:,} rate={hb['event_rate']:.3f}  Brier rate={hb['brier']['p_rate'][0]:.4f} "
                  f"direct={hb['brier']['p_direct'][0]:.4f} cal={hb['brier']['p_cal'][0]:.4f}  "
                  f"cal vs rate: {hb['paired']['p_cal_vs_p_rate']['verdict']}")

    R["truck_effect"] = truck_effect(d)
    for h, v in R["truck_effect"].items():
        print(f"truck effect {h}h: corr {v['corr']:.2f}, mean |trip net| {v['mean_abs_trip_net']:.1f}, "
              f"mean |gap| {v['mean_abs_gap']:.1f}, mean gap {v['mean_gap']:+.2f}")
    out = REPORTS / "availability_check.json"
    out.write_text(json.dumps(R, indent=1, sort_keys=True))
    print(f"wrote {out.relative_to(out.parents[1])}")


if __name__ == "__main__":
    sys.exit(main())
