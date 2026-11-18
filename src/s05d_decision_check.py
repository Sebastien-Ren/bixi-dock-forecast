"""Step 5d. Score the forecast as a decision: of the K stations it says to look at, how many were really big?

The questions, metric, K values and the rule for "beats" were fixed in advance (plan/01-decision-metric.md).
Reads  data/processed/predictions.parquet, the 2022-2023 rows of panel.parquet (for the "usual suspects"
       ranking) and data/interim/station_capacity.csv (for the exploratory dock-share view).
Writes reports/decision_check.json. It never changes a forecast.
"""
import json

import numpy as np
import pandas as pd

import bootstrap as bs
import ranking as rk
import scoring
from config import BIG_IMBALANCE_TOP_SHARE, HORIZONS, INTERIM, PROCESSED, REPORTS, TRAIN_YEARS, TZ

KS = (5, 10, 20)
HEADLINE_K, HEADLINE_H = 10, 3
REPS, SEED = 2000, 20261116
DOCK_SHARE = 0.5
RANKINGS = ["l2", "avg4w", "last_week", "usual", "l2_edge", "random"]
EXPLORATORY = {"l2_edge"}
# (a, b): is a better than b? Always a paired comparison on the same days.
PAIRS = [("l2", "avg4w"), ("l2", "last_week"), ("l2", "usual"), ("l2", "random"), ("avg4w", "usual"), ("l2_edge", "l2")]


def usual_suspects(panel, h):
    """Each station's typical |net| and typical net for every hour of the week, from the training years only."""
    tr = panel[panel.year.isin(TRAIN_YEARS)]
    g = tr.assign(a=tr[f"y_h{h}"].abs(), s=tr[f"y_h{h}"]).groupby(["station_id", "hour_of_week"])
    return g.a.mean().rename("usual_abs"), g.s.mean().rename("usual_net")


def add_day_and_hour(d):
    """Local calendar day for every cut-off (the unit the bootstrap resamples)."""
    ids = pd.Series(d.hour_idx.unique())
    day = pd.to_datetime(ids * 3600, unit="s", utc=True).dt.tz_convert(TZ).dt.strftime("%Y-%m-%d")
    return d.assign(day=d.hour_idx.map(dict(zip(ids, day))))


def time_table(d, positive, scores, signs, ks):
    """One row per decision time with everything the statistics need, for every ranking."""
    t = d.hour_idx.values
    tabs, base = {}, None
    for name, sc in scores.items():
        tab = rk.pick_table(t, d.station_id.values, sc["score"], positive, ks,
                            pred=signs.get(name), y=d.y.values, tiebreak=sc.get("tiebreak"))
        tabs[name] = tab
        base = tab if base is None else base
    tabs["random"] = rk.random_table(base, ks)
    meta = d.groupby("hour_idx").agg(day=("day", "first"), year=("year", "first"), hour=("hour", "first"))
    T = base[["n_cand", "n_pos"]].join(meta)
    T["n_times"] = 1.0
    T["has_pos"] = (T.n_pos > 0).astype(float)
    for name, tab in tabs.items():
        for k in ks:
            T[f"hits_{name}_{k}"] = tab[f"hits_{k}"]
            T[f"cap_{name}_{k}"] = tab[f"cap_{k}"]
            if f"dir_{k}" in tab:
                T[f"dir_{name}_{k}"] = tab[f"dir_{k}"]
    return T


def summarize(T, names, ks, pairs):
    """Every metric with a 95% day-block interval, and the paired differences."""
    cols = ["n_times", "has_pos", "n_pos", "n_cand"] + [c for c in T.columns if c.split("_")[0] in ("hits", "cap", "dir")]
    _, m = bs.day_sums(T, "day", cols)
    real = dict(zip(cols, m.sum(axis=0)))
    boot = dict(zip(cols, bs.resampled_totals(m, REPS, SEED).T))

    def iv(fn):
        p, lo, hi = bs.interval(fn(real), fn(boot))
        return {"v": p, "lo": lo, "hi": hi}

    kinds = {"precision": rk.precision, "capture": rk.capture, "direction": rk.direction,
             "lift": lambda t, n, k: rk.precision(t, n, k) / rk.base_rate(t)}
    out = {"n_times": int(real["n_times"]), "n_days": int(T.day.nunique()), "n_times_no_positive": int(real["n_times"] - real["has_pos"]),
           "base_rate": iv(rk.base_rate), "median_positives": float(T.n_pos.median()),
           "median_candidates": float(T.n_cand.median()), "rankings": {}, "paired": {}}
    for n in names:
        out["rankings"][n] = {}
        for k in ks:
            r = {}
            for kind, fn in kinds.items():
                if kind == "direction" and f"dir_{n}_{k}" not in real:
                    continue
                r[kind] = iv(lambda t, fn=fn, n=n, k=k: fn(t, n, k))
            out["rankings"][n][str(k)] = r
    for a, b in pairs:
        key = f"{a}_vs_{b}"
        out["paired"][key] = {}
        for k in ks:
            res = {}
            for kind in ("precision", "capture"):
                fn = kinds[kind]
                d = iv(lambda t, fn=fn, a=a, b=b, k=k: fn(t, a, k) - fn(t, b, k))
                d["verdict"] = bs.verdict(d["lo"], d["hi"])
                res[kind] = d
            out["paired"][key][str(k)] = res
    return out


def main():
    p = add_day_and_hour(pd.read_parquet(PROCESSED / "predictions.parquet"))
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    cap = pd.read_csv(INTERIM / "station_capacity.csv").set_index("station_id").capacity
    p = p.assign(hour_of_week=p.dow * 24 + p.hour)
    R = {"meta": {"ks": list(KS), "headline": {"k": HEADLINE_K, "horizon": HEADLINE_H}, "bootstrap_reps": REPS, "seed": SEED,
                  "big_share": BIG_IMBALANCE_TOP_SHARE, "dock_share_cutoff": DOCK_SHARE, "exploratory_rankings": sorted(EXPLORATORY),
                  "cutoffs": {}, "docks_matched": int(cap.notna().sum())},
         "main": {}, "by_year": {}, "by_hour": {}, "dock_share": {}}

    for h in HORIZONS:
        allrows = p[p.h == h]
        cutoff = scoring.big_imbalance_cutoff(allrows.y.values, BIG_IMBALANCE_TOP_SHARE)      # same cut-off as the main metric
        R["meta"]["cutoffs"][str(h)] = cutoff
        d = allrows[rk.decision_mask(allrows.hour.values, h) & allrows.l2.notna()].reset_index(drop=True)
        ua, un = usual_suspects(panel, h)
        key = pd.MultiIndex.from_arrays([d.station_id, d.hour_of_week])
        d = d.assign(usual_abs=ua.reindex(key).fillna(0).values, usual_net=un.reindex(key).fillna(0).values)
        positive = scoring.big_imbalance_mask(d.y.values, cutoff)

        scores = {"l2": {"score": d.l2.abs().values},
                  "avg4w": {"score": d.avg4w.abs().fillna(0).values},
                  "last_week": {"score": d.last_week.abs().fillna(0).values},
                  "usual": {"score": d.usual_abs.values},
                  "l2_edge": {"score": rk.edge_from_zero(d.l2_lo, d.l2_hi), "tiebreak": d.l2.abs().values}}
        signs = {"l2": d.l2.values, "avg4w": d.avg4w.fillna(0).values, "last_week": d.last_week.fillna(0).values,
                 "usual": d.usual_net.values, "l2_edge": d.l2.values}
        T = time_table(d, positive, scores, signs, KS)
        R["main"][str(h)] = summarize(T, RANKINGS, KS, PAIRS)
        R["main"][str(h)]["cutoff"] = cutoff
        R["by_year"][str(h)] = {str(int(y)): summarize(T[T.year == y], RANKINGS, (HEADLINE_K,), PAIRS[:5])
                                for y in sorted(T.year.unique())}
        if h == HEADLINE_H:
            R["by_hour"][str(h)] = {str(int(hr)): summarize(T[T.hour == hr], RANKINGS, (HEADLINE_K,), PAIRS[:5])
                                    for hr in sorted(T.hour.unique())}

        # exploratory: "how much of the station moves" -- positive = at least half the station's docks
        dc = d.assign(cap=d.station_id.map(cap)).dropna(subset=["cap"]).reset_index(drop=True)
        pos_dock = np.abs(dc.y.values) >= DOCK_SHARE * dc.cap.values
        sc = {n: {"score": (np.abs(dc[n].fillna(0).values) / dc.cap.values)} for n in ("l2", "avg4w", "last_week")}
        sg = {n: dc[n].fillna(0).values for n in sc}
        Td = time_table(dc, pos_dock, sc, sg, (HEADLINE_K,))
        R["dock_share"][str(h)] = summarize(Td, ["l2", "avg4w", "last_week", "random"], (HEADLINE_K,), [("l2", "avg4w"), ("l2", "random")])
        print(f"{h} h: cut-off {cutoff:.0f}, {len(T):,} decision times, {R['main'][str(h)]['n_days']} days")

    (REPORTS / "decision_check.json").write_text(json.dumps(R, indent=1))
    m = R["main"][str(HEADLINE_H)]
    k = str(HEADLINE_K)
    for n in RANKINGS:
        pr = m["rankings"][n][k]["precision"]
        print(f"  3 h, K=10  {n:<10} precision {pr['v']:.1%}  [{pr['lo']:.1%}, {pr['hi']:.1%}]")


if __name__ == "__main__":
    main()
