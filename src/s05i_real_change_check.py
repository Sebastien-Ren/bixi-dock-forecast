"""Step 5i. How does the forecast do against the real change in bikes, not just against trips? (exploratory)

Everything else in this project scores the forecast against trip net flow (arrivals minus departures), which is
what it was trained to predict. With real station counts for August and September 2025 the real change in bikes at a
station is also known (available plus disabled, end of window minus start). Trucks move bikes without leaving any
trip, so the two differ. This step measures how far the forecast tracks each.

EXPLORATORY: the question was not written down before the numbers were seen (it came from drawing the real change on
the map's 2025 station chart), it covers one period (August and September 2025), and the sealed month (September
2024) is deliberately not used. Day-block bootstrap, 2,000 resamples, as elsewhere.

Reads  data/interim/stock_forecast_2025.parquet (from s05g, with real_change) and data/processed/predictions.parquet.
Writes reports/real_change_check.json. Skips quietly if either is missing.
"""
import json
import sys

import numpy as np
import pandas as pd

import bootstrap as bs
from config import HORIZONS, INTERIM, PROCESSED, REPORTS, TZ

REPS, SEED = 2000, 20261118
BIG_Q = 0.9            # "big" = the 10% of windows with the largest |real change|, per look-ahead (descriptive cut-off)
COLS = ["n", "fc_t", "z_t", "fc_r", "z_r", "nb", "fc_rb", "z_rb", "dir_b", "cov_t", "cov_r",
        "l", "r", "y", "ll", "rr", "yy", "lr", "ly"]


def day_table(e):
    """One row per window with every quantity the statistics are built from (all of them are ratios of sums)."""
    r, y, l = e.real_change.astype(float), e.y.astype(float), e.l2.astype(float)
    big = (r.abs() >= r.abs().quantile(BIG_Q)).astype(float)
    t = pd.DataFrame({"day": e.day.values, "n": 1.0,
                      "fc_t": (y - l).abs(), "z_t": y.abs(), "fc_r": (r - l).abs(), "z_r": r.abs(),
                      "nb": big, "fc_rb": (r - l).abs() * big, "z_rb": r.abs() * big,
                      "dir_b": (np.sign(l) == np.sign(r)).astype(float) * big,
                      "cov_t": ((y >= e.l2_lo) & (y <= e.l2_hi)).astype(float),
                      "cov_r": ((r >= e.l2_lo) & (r <= e.l2_hi)).astype(float),
                      "l": l, "r": r, "y": y, "ll": l * l, "rr": r * r, "yy": y * y, "lr": l * r, "ly": l * y})
    return t, float(r.abs().quantile(BIG_Q))


def corr(t, a, b, aa, bb, ab):
    """Correlation from column totals; works on one row of totals or on many (bootstrap) rows."""
    n = t[..., 0]
    num = n * t[..., ab] - t[..., a] * t[..., b]
    den = np.sqrt((n * t[..., aa] - t[..., a] ** 2) * (n * t[..., bb] - t[..., b] ** 2))
    return num / den


def derive(t):
    """Every reported statistic from column totals (one row, or a bootstrap matrix)."""
    i = {c: k for k, c in enumerate(COLS)}
    g = lambda c: t[..., i[c]]
    return {"skill_vs_zero_trips": 1 - g("fc_t") / g("z_t"),
            "skill_vs_zero_real": 1 - g("fc_r") / g("z_r"),
            "skill_vs_zero_real_big": 1 - g("fc_rb") / g("z_rb"),
            "direction_right_big_real": g("dir_b") / g("nb"),
            "range_held_trips": g("cov_t") / g("n"), "range_held_real": g("cov_r") / g("n"),
            "corr_forecast_trips": corr(t, i["l"], i["y"], i["ll"], i["yy"], i["ly"]),
            "corr_forecast_real": corr(t, i["l"], i["r"], i["ll"], i["rr"], i["lr"])}


def horizon_stats(e):
    """All statistics with 95% day-block intervals for one look-ahead's windows."""
    tab, cutoff = day_table(e)
    days, m = bs.day_sums(tab, "day", COLS)
    point = derive(m.sum(axis=0))
    boot = derive(bs.resampled_totals(m, REPS, SEED))
    out = {"windows": int(len(e)), "days": int(len(days)), "big_cutoff": cutoff,
           "big_share_of_windows": float(tab.nb.mean())}
    out.update({k: bs.interval(point[k], boot[k]) for k in point})
    return out


def main():
    f = INTERIM / "stock_forecast_2025.parquet"
    if not f.exists() or "real_change" not in pd.read_parquet(f).columns:
        print("no stock_forecast_2025.parquet with real_change (run s05g); skipping")
        return 0
    s = pd.read_parquet(f)
    p = pd.read_parquet(PROCESSED / "predictions.parquet", columns=["station_id", "hour_idx", "h", "y", "l2", "l2_lo", "l2_hi"])
    d = s.merge(p, on=["station_id", "hour_idx", "h"])
    d["day"] = pd.to_datetime(d.hour_idx * 3600, unit="s", utc=True).dt.tz_convert(TZ).dt.strftime("%Y-%m-%d")
    R = {"exploratory": True, "period": "August and September 2025", "reps": REPS, "seed": SEED, "big_quantile": BIG_Q,
         "horizons": {str(h): horizon_stats(d[d.h == h]) for h in HORIZONS}}
    for h, v in R["horizons"].items():
        print(f"{h}h  n={v['windows']:,}  corr trips {v['corr_forecast_trips'][0]:.2f} real {v['corr_forecast_real'][0]:.2f}  "
              f"skill vs zero: trips {v['skill_vs_zero_trips'][0]:+.2f} real {v['skill_vs_zero_real'][0]:+.2f} "
              f"real-big {v['skill_vs_zero_real_big'][0]:+.2f}  direction {v['direction_right_big_real'][0]:.2f}  "
              f"range held: trips {v['range_held_trips'][0]:.2f} real {v['range_held_real'][0]:.2f}")
    out = REPORTS / "real_change_check.json"
    out.write_text(json.dumps(R, indent=1, sort_keys=True))
    print(f"wrote {out.relative_to(out.parents[1])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
