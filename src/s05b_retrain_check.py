"""Step 5b. Would retraining every year help?

The main model learns from 2022 and 2023 and is never retrained. Here, for each backtest year after
the first, a second model learns from every earlier year instead (2024 uses 2022 to 2023, 2025 uses
2022 to 2024, 2026 uses 2022 to 2025). Only data from before the test year is used, so this is
something a live system could really do.

Same inputs, same settings, forecast model only (the band is not refit).
Writes reports/retrain_check.json, which s06 turns into tables and the map shows.
"""
import json

import numpy as np
import pandas as pd

import scoring
from config import BACKTEST_YEARS, BIG_IMBALANCE_TOP_SHARE, HORIZONS, PROCESSED, REPORTS
from models import Level2


def main():
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    panel["station_code"] = pd.factorize(panel.station_id, sort=True)[0]
    setting = json.loads((REPORTS / "validation.json").read_text())["picked"]
    preds = pd.read_parquet(PROCESSED / "predictions.parquet")
    out = {}
    for h in HORIZONS:
        p = preds[preds.h == h]
        cutoff = scoring.big_imbalance_cutoff(p.y.values, BIG_IMBALANCE_TOP_SHARE)
        out[h] = {}
        for year in BACKTEST_YEARS:
            d = p[p.year == year]
            big = scoring.big_imbalance_mask(d.y.values, cutoff)
            row = {
                "learns_from": [int(y) for y in sorted(panel.year.unique()) if y < year],
                "zero": scoring.mae(d.y.values[big], d.zero.values[big]),
                "last_4_weeks": scoring.mae(d.y.values[big], d.avg4w.values[big]),
                "never_retrained": scoring.mae(d.y.values[big], d.l2.values[big]),
                "retrained": None,
            }
            if year != BACKTEST_YEARS[0]:
                fit = panel[panel.year < year]
                test = panel[panel.year == year].set_index(["station_id", "hour_idx"]).loc[
                    pd.MultiIndex.from_arrays([d.station_id, d.hour_idx])].reset_index()
                pred = Level2(h, **setting).fit(fit, only_mean=True).predict_mean(test)
                row["retrained"] = scoring.mae(d.y.values[big], pred[big])
            out[h][year] = row
            print(h, year, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    (REPORTS / "retrain_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
