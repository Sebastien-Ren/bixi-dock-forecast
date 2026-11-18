"""Step 5c. How much does weather matter?

Level 2 is refit with every weather input removed, on the same 2022 to 2023 training data, and
scored on the same backtest windows as the full model. Three versions are compared:
  no weather        the model never sees weather
  forecast weather  the main model: the ECMWF forecast as issued (what a live system would have)
  actual weather    the weather that really happened (cheating, shown as a ceiling)
Scored on all windows, on big-imbalance windows, and on rainy windows (1 mm or more over the window).

Writes reports/weather_check.json, which s06 turns into tables.
"""
import json

import pandas as pd

import scoring
from config import BACKTEST_YEARS, BIG_IMBALANCE_TOP_SHARE, HORIZONS, PROCESSED, REPORTS, TRAIN_YEARS
from models import Level2

RAIN_MM = 1.0


def main():
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    panel["station_code"] = pd.factorize(panel.station_id, sort=True)[0]
    setting = json.loads((REPORTS / "validation.json").read_text())["picked"]
    preds = pd.read_parquet(PROCESSED / "predictions.parquet")
    train = panel[panel.year.isin(TRAIN_YEARS)]
    out = {}
    for h in HORIZONS:
        p = preds[preds.h == h].reset_index(drop=True)
        test = (panel[panel.year.isin(BACKTEST_YEARS)].set_index(["station_id", "hour_idx"])
                .loc[pd.MultiIndex.from_arrays([p.station_id, p.hour_idx])].reset_index())
        model = Level2(h, **setting)
        model.cols = [c for c in model.cols if not c.startswith(("temp_", "precip_", "wind_"))]
        p["no_weather"] = model.fit(train, only_mean=True).predict_mean(test)
        cutoff = scoring.big_imbalance_cutoff(p.y.values, BIG_IMBALANCE_TOP_SHARE)
        slices = {"all": pd.Series(True, index=p.index),
                  "big": pd.Series(scoring.big_imbalance_mask(p.y.values, cutoff), index=p.index),
                  "rainy": p.precip_act >= RAIN_MM}
        out[h] = {}
        for name, m in slices.items():
            y = p.y[m].values
            out[h][name] = {"n": int(m.sum()),
                            **{col: scoring.mae(y, p[col][m].values)
                               for col in ("zero", "avg4w", "no_weather", "l2", "l2_actual_weather")}}
        print(h, {k: round(v["l2"], 3) for k, v in out[h].items()}, {k: round(v["no_weather"], 3) for k, v in out[h].items()}, flush=True)
    (REPORTS / "weather_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
