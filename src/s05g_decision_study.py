"""Step 5g. Which 10 stations should a dispatcher check? Three rankings scored on real station counts (plan/06, part A).

  forecast size    largest |Level 2 forecast| (what the map's shortlist does now)
  stock only       closest to empty or full right now (no forecast)
  stock + forecast chance of being empty or full at the end of the window (plan 05's method)
A pick counts if the station really is empty or full at the end of the window. Summer 2025 snapshots; fitted on
May to July, tested on August and September. Writes reports/decision_study.json.

BIXI's own rule is not scored: these counts already include what BIXI's trucks did under that rule, so the
stations it served look fixed. See plan/06.
"""
import json

import numpy as np
import pandas as pd

import availability as av
import bootstrap as bs
import ranking as rk
import s05e_availability_check as chk
from config import HORIZONS, INTERIM, PROCESSED, REPORTS

K = 10
REPS, SEED = 2000, 20261118
NAMES = {"size": "Forecast size", "stock": "Stock only", "both": "Stock + forecast", "random": "Random"}
PAIRS = [("both", "size"), ("both", "stock"), ("stock", "size")]


def stock_score(bikes, docks):
    """Higher = closer to empty or full now: minus the smaller of bikes and free docks."""
    return -np.minimum(np.asarray(bikes, float), np.asarray(docks, float))


def problem_chance(p_empty, p_full):
    """The station needs a truck if it ends empty or full; the two cannot both happen, so take the larger."""
    return np.maximum(p_empty, p_full)


def main():
    frames, _ = chk.load_snapshots()
    if frames is None:
        print("no past snapshots; skipping")
        return
    stations = pd.read_parquet(INTERIM / "stations.parquet")
    top = pd.read_csv(PROCESSED / "top_stations.csv")
    ours = stations[stations.station_id.isin(top.station_id)].reset_index(drop=True)
    states, _, _ = chk.station_states(chk.match_months(frames, ours))
    preds = pd.read_parquet(PROCESSED / "predictions.parquet",
                            columns=["station_id", "hour_idx", "year", "h", "y", "l2", "l2_lo", "l2_hi"])
    d, _ = chk.build_pairs(states, preds[preds.year == chk.YEAR])
    ok = chk.in_service(d, "empty") & chk.in_service(d, "full")
    d = d[ok].copy()
    d["problem"] = ((d.is_empty1 == 1) | (d.is_full1 == 1)).astype(int)
    d["p_e"] = av.prob_empty_direct(d.bikes0, d.l2, d.l2_lo, d.l2_hi)
    d["p_f"] = av.prob_full_direct(d.docks0, d.l2, d.l2_lo, d.l2_hi)

    R = {"k": K, "reps": REPS, "seed": SEED, "test_months": chk.TEST_MONTHS, "horizons": {}}
    keep = []       # every test window's chance and outcome, for the map's 2025 week
    for h in HORIZONS:
        e = d[d.h == h]
        tr, te = e[e.month.isin(chk.TRAIN_MONTHS)], e[e.month.isin(chk.TEST_MONTHS)].copy()
        cal_e = av.recalibrate(tr.p_e.values, tr.is_empty1.values)     # fitted on May to July, every in-service window
        cal_f = av.recalibrate(tr.p_f.values, tr.is_full1.values)
        te["both"] = problem_chance(cal_e(te.p_e.values), cal_f(te.p_f.values))
        te["size"] = te.l2.abs()
        te["stock"] = stock_score(te.bikes0, te.docks0)
        keep.append(te[["station_id", "hour_idx", "h", "bikes0", "docks0", "both", "is_empty1", "is_full1"]])
        te = te[rk.decision_mask(te.local_hour.values, h)]
        tables = {n: rk.pick_table(te.hour_idx.values, te.station_id.values, te[n].values, te.problem.values, (K,))
                  for n in ("size", "stock", "both")}
        base = tables["size"]
        tables["random"] = rk.random_table(base, (K,))
        meta = te.groupby("hour_idx").agg(day=("day", "first"), hour=("local_hour", "first"))
        T = base[["n_cand", "n_pos"]].join(meta)
        T["picks"] = np.minimum(K, T.n_cand)
        for n, tab in tables.items():
            T[n] = tab[f"hits_{K}"]
        cols = ["picks", "n_cand", "n_pos", "size", "stock", "both", "random"]
        _, m = bs.day_sums(T, "day", cols)
        boot = bs.resampled_totals(m, REPS, SEED)
        tot, i = m.sum(axis=0), {c: j for j, c in enumerate(cols)}
        prec = lambda n, src: src[..., i[n]] / src[..., i["picks"]]
        res = {"decision_times": int(len(T)), "median_candidates": float(T.n_cand.median()),
               "base_rate": float(tot[i["n_pos"]] / tot[i["n_cand"]]),
               "precision": {n: bs.interval(prec(n, tot), prec(n, boot)) for n in NAMES},
               "paired": {}, "by_hour": {}}
        for a, b in PAIRS:
            dlt = bs.interval(prec(a, tot) - prec(b, tot), prec(a, boot) - prec(b, boot))
            res["paired"][f"{a}_vs_{b}"] = {"diff": dlt, "verdict": bs.verdict(dlt[1], dlt[2])}
        for hr, g in T.groupby("hour"):
            _, mh = bs.day_sums(g, "day", cols)
            bh = bs.resampled_totals(mh, REPS, SEED)
            th = mh.sum(axis=0)
            dlt = bs.interval(prec("both", th) - prec("stock", th), prec("both", bh) - prec("stock", bh))
            res["by_hour"][f"{int(hr):02d}"] = {"times": int(len(g)),
                                                 "precision": {n: float(prec(n, th)) for n in ("size", "stock", "both")},
                                                 "both_vs_stock": {"diff": dlt, "verdict": bs.verdict(dlt[1], dlt[2])}}
        R["horizons"][str(h)] = res
        p = res["precision"]
        print(f"{h}h  times={len(T)}  base={res['base_rate']:.3f}  size={p['size'][0]:.3f}  stock={p['stock'][0]:.3f}  "
              f"both={p['both'][0]:.3f}  random={p['random'][0]:.3f}  "
              + "  ".join(f"{k}:{v['verdict']}" for k, v in res["paired"].items()))
    pd.concat(keep, ignore_index=True).to_parquet(INTERIM / "stock_forecast_2025.parquet", index=False)
    out = REPORTS / "decision_study.json"
    out.write_text(json.dumps(R, indent=1, sort_keys=True))
    print(f"wrote {out.relative_to(out.parents[1])}")


if __name__ == "__main__":
    main()
