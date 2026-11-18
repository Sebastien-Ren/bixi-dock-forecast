"""Step 5. Train on 2022 and 2023, then replay 2024 to 2026 hour by hour and keep every forecast.

Order of work, so that the backtest stays honest:
  1. Settings are chosen using training years only (learn on 2022, check on 2023).
  2. The chosen setting is refit on 2022 + 2023.
  3. The backtest years are predicted once. Nothing is tuned on them.

Every hour of every backtest summer is its own cut-off: the forecast for that hour uses only
trips before it and the weather forecast run that had been published by then.

Outputs
  data/processed/predictions.parquet   one row per station, hour and horizon, all models side by side
  reports/validation.json              what step 1 compared, and what it picked
  reports/feature_importance.csv       which inputs Level 2 leans on
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

import scoring
from config import (BACKTEST_YEARS, BAND_TARGET, BIG_IMBALANCE_TOP_SHARE, HORIZONS, PROCESSED, REPORTS,
                    TRAIN_YEARS, VALID_YEAR)
from models import (Level1, Level2, predict_four_week_average, predict_last_week, predict_persistence, predict_zero,
                    self_correcting_band)

CANDIDATES = [
    dict(with_station_code=False, max_leaf_nodes=31),
    dict(with_station_code=False, max_leaf_nodes=127),
    dict(with_station_code=True, max_leaf_nodes=31),
    dict(with_station_code=True, max_leaf_nodes=127),
]


def big_mae(y, pred):
    mask = scoring.big_imbalance_mask(y, scoring.big_imbalance_cutoff(y, BIG_IMBALANCE_TOP_SHARE))
    return scoring.mae(y[mask], pred[mask])


def choose_setting(panel):
    """Learn on the other training year, score on VALID_YEAR, pick the lowest big-imbalance error
    (summed over the three horizons)."""
    fit_df = panel[panel.year.isin([y for y in TRAIN_YEARS if y != VALID_YEAR])]
    val_df = panel[panel.year == VALID_YEAR]
    rows = []
    for cand in CANDIDATES:
        row = dict(cand)
        for h in HORIZONS:
            y = val_df[f"y_h{h}"].values
            pred = Level2(h, **cand).fit(fit_df, only_mean=True).predict_mean(val_df)
            row[f"big_mae_h{h}"] = round(big_mae(y, pred), 4)
        row["total"] = round(sum(row[f"big_mae_h{h}"] for h in HORIZONS), 4)
        rows.append(row)
        print("  ", row)
    best = min(rows, key=lambda r: r["total"])
    return {k: best[k] for k in ("with_station_code", "max_leaf_nodes")}, rows


def main():
    REPORTS.mkdir(exist_ok=True)
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    panel["station_code"] = pd.factorize(panel.station_id, sort=True)[0]
    train = panel[panel.year.isin(TRAIN_YEARS)]
    test = panel[panel.year.isin(BACKTEST_YEARS)]
    print(f"train rows {len(train):,} ({TRAIN_YEARS}); backtest rows {len(test):,} ({BACKTEST_YEARS})")

    print(f"\n1) choosing Level 2 settings: learn on {[y for y in TRAIN_YEARS if y != VALID_YEAR]}, check on {VALID_YEAR}")
    t0 = time.time()
    setting, table = choose_setting(panel)
    print(f"   picked {setting}  ({time.time() - t0:.0f}s)")
    (REPORTS / "validation.json").write_text(json.dumps(dict(picked=setting, candidates=table), indent=2))

    print("\n2) + 3) refit on all training years, predict the backtest once")
    out = []
    for h in HORIZONS:
        t0 = time.time()
        l1 = Level1().fit(train, h)
        l2 = Level2(h, **setting).fit(train)
        l1_mean, l1_lo, l1_hi = l1.predict(test)
        l2_mean, l2_lo, l2_mid, l2_hi = l2.predict(test)
        cheat_mean, *_ = l2.predict(test, weather="actual")
        # The band then corrects itself day by day from its own recent hits and misses.
        y = test[f"y_h{h}"].values
        adj_lo, adj_hi, stretch = self_correcting_band(test.hour_idx.values, h, y, l2_lo, l2_hi, BAND_TARGET)
        adj_lo, adj_hi = np.minimum(adj_lo, l2_mean), np.maximum(adj_hi, l2_mean)    # the forecast stays inside
        out.append(pd.DataFrame({
            "station_id": test.station_id.values, "hour_idx": test.hour_idx.values, "year": test.year.values,
            "hour": test.hour.values, "dow": test.dow.values, "is_holiday": test.is_holiday.values, "h": h,
            "y": y,
            "precip_fc": test[f"precip_h{h}"].values, "precip_act": test[f"precip_act_h{h}"].values,
            "temp_fc": test[f"temp_h{h}"].values, "temp_act": test[f"temp_act_h{h}"].values,
            "zero": predict_zero(test, h), "persist": predict_persistence(test, h), "last_week": predict_last_week(test, h),
            "avg4w": predict_four_week_average(test, h),
            "l1": l1_mean, "l1_lo": l1_lo, "l1_hi": l1_hi,
            "l2": l2_mean, "l2_mid": l2_mid,
            "l2_raw_lo": l2_lo, "l2_raw_hi": l2_hi,          # the band straight from the model
            "l2_lo": adj_lo, "l2_hi": adj_hi, "band_stretch": stretch,   # the band after self-correction
            "l2_actual_weather": cheat_mean,
        }))
        print(f"   horizon {h}h done ({time.time() - t0:.0f}s)")

        if h == 3:      # which inputs matter, measured on a backtest sample
            sample = test.sample(40_000, random_state=0)
            imp = permutation_importance(l2.mean, sample[l2.cols], sample[f"y_h{h}"], n_repeats=3,
                                         random_state=0, scoring="neg_mean_absolute_error")
            (pd.DataFrame({"feature": l2.cols, "mae_increase_when_shuffled": imp.importances_mean})
             .sort_values("mae_increase_when_shuffled", ascending=False)
             .to_csv(REPORTS / "feature_importance.csv", index=False))

    preds = pd.concat(out, ignore_index=True)
    preds.to_parquet(PROCESSED / "predictions.parquet", index=False)
    print(f"\nsaved {len(preds):,} forecasts -> data/processed/predictions.parquet")


if __name__ == "__main__":
    main()
