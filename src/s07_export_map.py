"""Step 7. Write the one file the map page reads: web/data/forecast.json.

The model and the map stay separate. The page knows nothing about Python or the model;
it only reads this file: station, hour, horizon, forecast, uncertainty range, enough-data flag.

The map shows the last full Monday-to-Sunday week of the data. Every forecast in it comes
straight from the backtest, so it was made with only what was known at that hour.
"""
import json

import numpy as np
import pandas as pd

from config import HORIZONS, MAP_DAYS, PROCESSED, RAW, REPORTS, TZ, WEB_DATA


def dock_counts(top):
    """Dock capacity from the live station list, matched by position (within 150 m). Context only."""
    files = sorted(RAW.glob("station_information_*.json"))
    if not files:
        return [None] * len(top)
    live = pd.DataFrame(json.loads(files[-1].read_text())["data"]["stations"])
    out = []
    for r in top.itertuples():
        d = np.hypot((live.lon - r.lon) * 111_320 * np.cos(np.radians(45.51)), (live.lat - r.lat) * 110_540)
        i = d.idxmin()
        out.append(int(live.capacity[i]) if d[i] <= 150 and live.capacity[i] > 0 else None)
    return out


def findings(metrics, preds=None):
    """The few results the page explains (decision shortlist, empty/full test, truck effect, what the model uses).
    Rounded, so the file stays small. Everything here is already in reports/metrics.json."""
    r = lambda x, d=4: None if x is None else round(float(x), d)
    iv = lambda t: [r(t[0], 5), r(t[1], 5), r(t[2], 5)]
    dc = metrics["decision"]
    k = str(dc["meta"]["headline"]["k"])
    decision = {"k": int(k), "reps": dc["meta"]["bootstrap_reps"], "by_h": {}}
    for h, m in dc["main"].items():
        rk = m["rankings"]
        decision["by_h"][h] = {
            "precision": {n: [r(rk[n][k]["precision"]["v"]), r(rk[n][k]["precision"]["lo"]), r(rk[n][k]["precision"]["hi"])]
                          for n in ("l2", "avg4w", "last_week", "usual", "random")},
            "l2_minus_avg4w": [r(m["paired"]["l2_vs_avg4w"][k]["precision"][x]) for x in ("v", "lo", "hi")]}
    out = {"decision": decision,
           "importance": [{"feature": x["feature"], "mae_up": r(x["mae_increase_when_shuffled"], 3)}
                          for x in metrics["feature_importance"][:8]]}
    if preds is not None:          # how net flow is spread: is "most windows are near zero" true? Measured, not assumed.
        out["distribution"] = {}
        for h, g in preds.groupby("h"):
            a = g.y.abs()
            top = a >= a.quantile(0.9)
            m = metrics["horizons"][str(h)]
            out["distribution"][str(int(h))] = {"median_abs": r(a.median(), 1), "share_le2": r((a <= 2).mean(), 3),
                                                "top10_share_of_movement": r(a[top].sum() / a.sum(), 3),
                                                "mae_zero_all": r(m["mae_zero"], 2), "mae_l2_all": r(m["mae_l2"], 2)}
    ds = metrics.get("decision_study")
    if ds:
        out["decision_study"] = {"k": ds["k"], "by_h": {h: {"precision": {n: iv(v["precision"][n]) for n in ("size", "stock", "both", "random")},
                                                           "both_vs_stock": iv(v["paired"]["both_vs_stock"]["diff"]),
                                                           "both_vs_stock_verdict": v["paired"]["both_vs_stock"]["verdict"],
                                                           "base_rate": r(v["base_rate"])}
                                                       for h, v in ds["horizons"].items()}}
    ms = metrics.get("model_study")
    if ms:
        out["model_study"] = {"neighbour_m": ms["neighbour_m"], "by_h": {
            h: {"base": r(v["base_mae"], 3),
                "rows": [[f"Without {k}", r(x["relative"][0]), r(x["relative"][1]), r(x["relative"][2]), x["verdict"]] for k, x in v["ablation"].items()]
                        + [["With neighbours", r(v["neighbours"]["relative"][0]), r(v["neighbours"]["relative"][1]), r(v["neighbours"]["relative"][2]), v["neighbours"]["verdict"]],
                           ["Two flows (arrivals, departures)", r(v["two_flows"]["relative"][0]), r(v["two_flows"]["relative"][1]), r(v["two_flows"]["relative"][2]), v["two_flows"]["verdict"]]]}
            for h, v in ms["horizons"].items()}}
    av = metrics.get("availability")
    if av:
        a = {"matched": av["matched_stations"], "train_months": av["train_months"], "test_months": av["test_months"],
             "by_h": {}, "truck": {}}
        for h in map(str, HORIZONS):
            a["by_h"][h] = {}
            for e in ("empty", "full"):
                v = av["events"][e]["horizons"][h]
                hd = v["headline"]
                a["by_h"][h][e] = {"n": hd["n"], "rate": r(hd["event_rate"]),
                                   "skill": iv(hd["paired"]["p_cal_vs_p_rate"]["skill"]),
                                   "verdict": hd["paired"]["p_cal_vs_p_rate"]["verdict"],
                                   "direct_skill": r(hd["paired"]["p_direct_vs_p_rate"]["skill"][0]),
                                   "calibration": {c: [[r(b["stated"], 3), r(b["happened"], 3), b["n"]] for b in v["calibration"][c]]
                                                   for c in ("p_direct", "p_cal")},
                                   "by_hour_block": {kk: [r(vv["paired"]["p_cal_vs_p_rate"]["skill"][0]), vv["paired"]["p_cal_vs_p_rate"]["verdict"]]
                                                     for kk, vv in v["by_hour_block"].items()}}
            t = av["truck_effect"][h]
            a["truck"][h] = {"corr": r(t["corr"], 3), "trip": r(t["mean_abs_trip_net"], 2), "gap": r(t["mean_abs_gap"], 2),
                             "gap5": r(t["share_gap_5plus"], 3),
                             "by_block": {kk: [r(vv["mean_abs_gap"], 2), r(vv["mean_gap"], 2)] for kk, vv in t["by_hour_block"].items()}}
        out["availability"] = a
    return out


STOCK_WEEK = "2025-08-04"      # a Monday in the empty/full test months with good snapshot coverage (plan/06)


def running_examples(p, top):
    """Two real stations for the method page, both at Thursday 7 August 2025, 17:00, 3-hour window, each picked by a
    rule fixed in advance. They illustrate; the evidence is the aggregate results.
      hit:   the highest-ranked of the 10 stock + forecast picks that really ended empty or full (the method working).
      limit: among stations whose trips alone would have taken them below zero bikes (bikes now + trip net < 0) yet
             were not empty at the end, the one with the highest chance (the truck effect)."""
    f = PROCESSED.parent / "interim" / "stock_forecast_2025.parquet"
    if not f.exists():
        return None
    from availability import prob_empty_direct, prob_full_direct, sigma_from_band
    t = pd.Timestamp("2025-08-07 17:00", tz=TZ)
    hi = int(t.timestamp()) // 3600
    s = pd.read_parquet(f)
    x = s[(s.hour_idx == hi) & (s.h == 3)].merge(p[["station_id", "hour_idx", "h", "y", "l2", "l2_lo", "l2_hi"]],
                                                 on=["station_id", "hour_idx", "h"])
    x = x.sort_values(["both", "station_id"], ascending=[False, True]).reset_index(drop=True)

    def describe(e, rank):
        full = bool(e.is_full1) or (not e.is_empty1 and e.docks0 < e.bikes0 and e.l2 > 0)
        sig = float(sigma_from_band(e.l2_lo, e.l2_hi))
        if full:
            z, praw = (e.docks0 - 0.5 - e.l2) / sig, float(prob_full_direct(e.docks0, e.l2, e.l2_lo, e.l2_hi))
        else:
            z, praw = (0.5 - e.bikes0 - e.l2) / sig, float(prob_empty_direct(e.bikes0, e.l2, e.l2_lo, e.l2_hi))
        return {"name": top.set_index("station_id").name[e.station_id], "when": "Thursday 7 August 2025, 17:00 to 20:00",
                "rank": int(rank) + 1, "event": "full" if full else "empty",
                "bikes": int(e.bikes0), "docks": int(e.docks0), "mu": round(float(e.l2), 1), "lo": round(float(e.l2_lo), 1),
                "hi": round(float(e.l2_hi), 1), "sigma": round(sig, 1), "z": round(float(z), 2), "p_raw": round(praw, 3),
                "p_cal": round(float(e.both), 3), "trip_net": int(e.y),
                "outcome": "empty" if e.is_empty1 else "full" if e.is_full1 else "ok"}

    out = {}
    top10 = x.head(10)
    hit = top10[(top10.is_empty1 == 1) | (top10.is_full1 == 1)]
    if len(hit):
        out["hit"] = describe(hit.iloc[0], hit.index[0])
    lim = x[(x.bikes0 + x.y < 0) & (x.is_empty1 == 0)]
    if len(lim):
        out["limit"] = describe(lim.iloc[0], lim.index[0])
    return out or None


def week_payload(p, top, start, stock=None):
    """Everything the map needs for one Monday-to-Sunday week starting at `start` (local midnight).
    `stock`: optional table of real station counts and chances (data/interim/stock_forecast_2025.parquet)."""
    h0 = int(start.timestamp()) // 3600
    hours = np.arange(h0, h0 + MAP_DAYS * 24)
    week = p[p.hour_idx.between(hours[0], hours[-1])]
    ids = top.station_id.values

    def grid(d, col, digits):
        """hours x stations table as nested lists, None where there is no value."""
        g = (d.pivot(index="hour_idx", columns="station_id", values=col)
             .reindex(index=hours, columns=ids).round(digits))
        return [[None if np.isnan(v) else (int(v) if digits == 0 else float(v)) for v in row] for row in g.values]

    forecasts, weather = {}, {}
    for h in HORIZONS:
        d = week[week.h == h]
        forecasts[h] = {"forecast": grid(d, "l2", 1), "low": grid(d, "l2_lo", 1), "high": grid(d, "l2_hi", 1),
                        "actual": grid(d, "y", 0)}
        if stock is not None:
            s = stock[(stock.h == h) & stock.hour_idx.between(hours[0], hours[-1])].assign(
                outcome=lambda x: np.where(x.is_empty1 == 1, 1, np.where(x.is_full1 == 1, 2, 0)))
            forecasts[h].update({"bikes": grid(s, "bikes0", 0), "docks": grid(s, "docks0", 0),
                                 "chance": grid(s, "both", 3), "outcome": grid(s, "outcome", 0)})
        w = d.groupby("hour_idx")[["temp_fc", "precip_fc"]].first().reindex(hours)
        weather[h] = {"temp": [None if np.isnan(v) else round(float(v), 1) for v in w.temp_fc],
                      "precip": [None if np.isnan(v) else round(float(v), 1) for v in w.precip_fc]}
    return hours, set(week.station_id.unique()), forecasts, weather


def main():
    WEB_DATA.mkdir(parents=True, exist_ok=True)
    p = pd.read_parquet(PROCESSED / "predictions.parquet")
    top = pd.read_csv(PROCESSED / "top_stations.csv").sort_values("rank").reset_index(drop=True)
    metrics = json.loads((REPORTS / "metrics.json").read_text())
    top["capacity"] = dock_counts(top)

    # The last full Monday-to-Sunday week that has forecasts for every hour.
    local = pd.to_datetime(p.hour_idx.max() * 3600, unit="s", utc=True).tz_convert(TZ)
    last_sunday = local.normalize()
    while not (last_sunday.day_of_week == 6 and last_sunday + pd.Timedelta(hours=23) <= local):
        last_sunday -= pd.Timedelta(days=1)
    latest = last_sunday - pd.Timedelta(days=MAP_DAYS - 1)

    stock_file = PROCESSED.parent / "interim" / "stock_forecast_2025.parquet"
    weeks = [{"id": "latest", "file": "forecast.json", "start": latest.strftime("%Y-%m-%d"), "stock": False}]
    if stock_file.exists():
        weeks.append({"id": STOCK_WEEK, "file": f"week_{STOCK_WEEK}.json", "start": STOCK_WEEK, "stock": True})

    shared = {
        "retrain": {
            str(h): {str(y): {k: r[k] for k in ("never_retrained", "retrained", "last_4_weeks", "zero")}
                     for y, r in metrics["retrain"][str(h)].items()}
            for h in HORIZONS
        },
        "scorecard": {
            str(h): {k: metrics["horizons"][str(h)][k] for k in (
                "cutoff", "n", "n_big", "big_mae_zero", "big_mae_persist", "coverage_l2_raw", "big_mae_last_week", "big_mae_avg4w", "big_mae_l1", "big_mae_l2",
                "big_skill_l2_vs_zero", "big_skill_l2_vs_best_simple", "big_direction_l2",
                "mae_zero", "mae_l2", "coverage_l2", "band_width_l2")}
            for h in HORIZONS
        },
        "findings": {**findings(metrics, p), "examples": running_examples(p, top)},
        "weeks": weeks,
    }
    for wk in weeks:
        start = pd.Timestamp(wk["start"], tz=TZ)
        stock = pd.read_parquet(stock_file) if wk["stock"] else None
        hours, in_service, forecasts, weather = week_payload(p, top, start, stock)
        out = {
            "meta": {
                "timezone": TZ, "week": wk["id"], "has_stock": wk["stock"],
                "first_hour": pd.Timestamp(hours[0] * 3600, unit="s", tz="UTC").tz_convert(TZ).isoformat(),
                "hours": len(hours), "horizons": HORIZONS,
                "trained_on": "summers 2022 and 2023",
                "weather_source": "ECMWF IFS forecast runs as issued, via Open-Meteo",
            },
            "stations": [
                {"id": int(r.station_id), "name": r.name, "lat": round(r.lat, 6), "lon": round(r.lon, 6),
                 "rank": int(r.rank), "capacity": int(r.capacity) if pd.notna(r.capacity) else None,
                 "enough_data": bool(r.enough_data), "in_service": int(r.station_id) in in_service}
                for r in top.itertuples()
            ],
            "forecasts": forecasts,
            "weather": weather,
            **shared,
        }
        f = WEB_DATA / wk["file"]
        f.write_text(json.dumps(out, separators=(",", ":")))
        end = start + pd.Timedelta(days=MAP_DAYS - 1)
        print(f"week {start:%a %d %b %Y} to {end:%a %d %b %Y}: {len(in_service)} stations in service"
              f"{', with station counts' if wk['stock'] else ''}; wrote {f.relative_to(f.parents[2])} ({f.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
