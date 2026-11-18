"""Step 5h. A sealed confirmation on September 2024 (planned and sealed before its data was downloaded).

The empty/full test (s05e) and the "which stations to check" test (s05g) were built and read on summer 2025.
This reruns both, unchanged, on a month nobody had looked at: September 2024. Level 2 was never trained on 2024.

  fitted parts   the starting-count table and the recalibrations, fitted on May to July 2025 exactly as in
                 s05e and s05g (same rows, same deterministic code). Nothing is fitted on September 2024.
  scored on      September 2024 windows (start of window in September, Montreal time).
  judged at      3 hours: "stock + forecast" and "stock only" both clearly above "forecast size", and the
                 recalibrated chance not worse than the starting-count baseline, for empty and for full.

Run once:  .venv/bin/python src/s05h_confirmation.py --coverage   (step 1: coverage only, reads no outcome)
           .venv/bin/python src/s05h_confirmation.py              (step 3: the one run)
Writes reports/confirmation_2024_09.json and refuses to run again once it exists (the seal).
"""
import argparse
import hashlib
import json
import sys
from datetime import date

import numpy as np
import pandas as pd

import s05e_availability_check as chk
import s05g_decision_study as dg
from config import HORIZONS, PROCESSED, REPORTS, TZ

CONF_PATTERN = "inventory_2024-09-*.parquet"
CONF_YEAR, CONF_MONTHS = 2024, (9,)
OUT = REPORTS / "confirmation_2024_09.json"
PRED_COLS = ["station_id", "hour_idx", "year", "h", "y", "l2", "l2_lo", "l2_hi"]
HEADLINE_H = 3


def coverage(frames):
    """How complete the month's snapshots are. Times only: no counts, events or outcomes."""
    t = pd.concat([f.snap_time.drop_duplicates() for f in frames]).sort_values().reset_index(drop=True)
    gaps = t.diff().dt.total_seconds().div(60).dropna()
    local = t.dt.tz_convert(TZ)
    return {"snapshots": int(len(t)), "first": str(local.iloc[0]), "last": str(local.iloc[-1]),
            "days_with_snapshots": int(local.dt.date.nunique()), "median_gap_min": float(gaps.median()),
            "max_gap_min": float(gaps.max()), "gaps_over_60_min": int((gaps > 60).sum())}


def pairs(states, preds, year, months):
    """Forecast windows of `year` with a station count at both ends, keeping windows that start in `months`."""
    d, _ = chk.build_pairs(states, preds[preds.year == year])
    return d[d.month.isin(months)].reset_index(drop=True)


def fit_2025(d25):
    """Every fitted part, from May to July 2025 windows only, with the same functions s05e and s05g use."""
    d25 = d25[d25.month.isin(chk.TRAIN_MONTHS)]
    fits = {}
    for event in chk.EVENTS:
        e = chk.event_windows(d25, event)
        for h in HORIZONS:
            fits[(event, h)] = chk.fit_event(e[e.h == h], event)
    c = dg.candidate_windows(d25)
    for h in HORIZONS:
        fits[("recal", h)] = dg.fit_recal(c[c.h == h])
    return fits


def score_confirmation(d, fits):
    """Apply already-fitted parts to the confirmation windows `d`. Fits nothing."""
    out = {"empty_full": {}, "decision": {}}
    for event in chk.EVENTS:
        e = chk.event_windows(d, event)
        out["empty_full"][event] = {"windows_in_service": int(len(e)), "horizons": {
            str(h): chk.score_event(e[e.h == h], event, *fits[(event, h)]) for h in HORIZONS}}
    c = dg.candidate_windows(d)
    for h in HORIZONS:
        out["decision"][str(h)] = dg.score_rankings(dg.add_scores(c[c.h == h], *fits[("recal", h)]), h)
    out["truck_effect"] = chk.truck_effect(d)
    return out


def judge(res, h=HEADLINE_H):
    """The two tests written down in the plan before the data was seen."""
    dec = res["decision"][str(h)]["paired"]
    order = {k: dec[k]["verdict"] for k in ("both_vs_size", "stock_vs_size")}
    skill = {ev: res["empty_full"][ev]["horizons"][str(h)]["headline"]["paired"]["p_cal_vs_p_rate"]["verdict"]
             for ev in chk.EVENTS}
    ranking_holds = all(v == "better" for v in order.values())
    skill_holds = all(v != "worse" for v in skill.values())
    return {"horizon": h, "ranking_order": {"verdicts": order, "holds": ranking_holds},
            "empty_full_skill": {"verdicts": skill, "holds": skill_holds},
            "confirmed": bool(ranking_holds and skill_holds)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", action="store_true", help="step 1: check the month's snapshots, read no outcome")
    args = ap.parse_args(argv)

    frames24, files24 = chk.load_snapshots(CONF_PATTERN)
    if frames24 is None:
        print("no September 2024 snapshots (run src/archive/build_inventory.py 2024-09-01 2024-10-01); skipping")
        return 0
    if not args.coverage and OUT.exists():
        print(f"{OUT.relative_to(OUT.parents[1])} exists: the sealed confirmation has already been run once. "
              "Delete it only with a written reason.")
        return 0

    states24, match24 = chk.load_states(frames24)
    cov = coverage(frames24) | match24
    preds = pd.read_parquet(PROCESSED / "predictions.parquet", columns=PRED_COLS)
    d24 = pairs(states24, preds, CONF_YEAR, CONF_MONTHS)
    n_fc = int(((preds.year == CONF_YEAR) & pd.to_datetime(preds.hour_idx * 3600, unit="s", utc=True)
                .dt.tz_convert(TZ).dt.month.isin(CONF_MONTHS)).sum())
    cov |= {"forecast_windows": n_fc, "with_both_snapshots": int(len(d24)),
            "by_horizon": {str(h): int((d24.h == h).sum()) for h in HORIZONS},
            "days_with_windows": int(d24.day.nunique())}
    print(json.dumps(cov, indent=1))
    if args.coverage:
        return 0

    frames25, _ = chk.load_snapshots()
    if frames25 is None:
        print("the 2025 snapshots are needed to rebuild the fitted parts; skipping")
        return 0
    states25, _ = chk.load_states(frames25)
    fits = fit_2025(pairs(states25, preds, chk.YEAR, chk.TRAIN_MONTHS))
    res = score_confirmation(d24, fits)
    R = {"plan": "plan/07-sealed-confirmation-sept-2024.md", "run_date": date.today().isoformat(),
         "confirmation_month": "2024-09", "fitted_on": "May to July 2025 (s05e and s05g, unchanged)",
         "seeds": {"empty_full": chk.SEED, "decision": dg.SEED}, "reps": chk.REPS, "k": dg.K,
         "snapshot_files": [f.name for f in files24], "coverage": cov, **res, "verdict": judge(res)}
    OUT.write_text(json.dumps(R, indent=1, sort_keys=True))

    for h in HORIZONS:
        p = res["decision"][str(h)]["precision"]
        print(f"{h}h  precision@10  size={p['size'][0]:.3f} stock={p['stock'][0]:.3f} both={p['both'][0]:.3f} "
              f"random={p['random'][0]:.3f}")
        for ev in chk.EVENTS:
            s = res["empty_full"][ev]["horizons"][str(h)]["headline"]["paired"]["p_cal_vs_p_rate"]
            print(f"     {ev:5s} skill vs starting count {s['skill'][0]:+.3f} ({s['skill'][1]:+.3f} to {s['skill'][2]:+.3f}) "
                  f"{s['verdict']}")
        print(f"     truck effect corr {res['truck_effect'][str(h)]['corr']:.2f}")
    print("verdict:", json.dumps(R["verdict"]))
    print(f"wrote {OUT.relative_to(OUT.parents[1])}  sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
