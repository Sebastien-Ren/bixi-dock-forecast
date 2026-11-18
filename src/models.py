"""The forecasters, from simplest to most capable. Each one works on its own.

  predict_zero       Baseline: net flow is always 0.
  predict_last_week  Baseline: the same window, same station, one week ago.
  predict_persistence  Baseline: the window just before this one (same length), same station.
  predict_four_week_average  Baseline: the same window averaged over the last four weeks.
  Level1             Average net flow for this station, hour of week and temperature band.
  Level2             Gradient boosting on recent traffic, calendar and forecast weather.

Every model returns three numbers per row: a forecast, and the low and high edge of an 80% band.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from config import QUANTILES

Q_LO, Q_MID, Q_HI = QUANTILES


# --------------------------------------------------------------------------- baselines
def predict_zero(df, h):
    return np.zeros(len(df))


def predict_persistence(df, h):
    """Whatever just happened is what happens next: the net flow of the h hours before the forecast hour.
    The usual baseline for short horizons. Uses only hours already finished, so there is no leakage."""
    return df[f"net_last{h}"].fillna(0).values


def predict_last_week(df, h):
    """If the station was out of service a week ago there is nothing to copy, so fall back to zero."""
    return df[f"net_lastweek_h{h}"].fillna(0).values


def predict_four_week_average(df, h):
    """The same window averaged over the last four weeks. Smoother than one week, still keeps up with change."""
    return df[f"net_avg4w_h{h}"].fillna(0).values


# --------------------------------------------------------------------------- Level 1
class Level1:
    """A lookup table. For each station, hour of week (Mon 00:00 ... Sun 23:00) and temperature band,
    store the average net flow seen in training, plus the 10th and 90th percentile for the band.
    If a cell has too few examples, use the station + hour-of-week cell without temperature."""

    MIN_EXAMPLES = 8

    def fit(self, df, h):
        self.h = h
        temp = df[f"temp_h{h}"]
        # Three temperature bands with equal numbers of training hours: cool, mild, warm.
        self.edges = np.quantile(temp.dropna(), [1 / 3, 2 / 3])
        d = pd.DataFrame({"station_id": df.station_id.values, "how": df.hour_of_week.values,
                          "band": np.digitize(temp, self.edges), "y": df[f"y_h{h}"].values})
        agg = dict(mean=("y", "mean"), lo=("y", lambda v: np.quantile(v, Q_LO)),
                   hi=("y", lambda v: np.quantile(v, Q_HI)), n=("y", "size"))
        self.fine = d.groupby(["station_id", "how", "band"]).agg(**agg)
        self.coarse = d.groupby(["station_id", "how"]).agg(**agg)
        return self

    def predict(self, df):
        temp = df[f"temp_h{self.h}"]
        band = np.where(temp.isna(), -1, np.digitize(temp, self.edges))       # -1 = no forecast: use coarse
        fine = self.fine.reindex(pd.MultiIndex.from_arrays([df.station_id, df.hour_of_week, band]))
        coarse = self.coarse.reindex(pd.MultiIndex.from_arrays([df.station_id, df.hour_of_week]))
        use_fine = (fine.n.values >= self.MIN_EXAMPLES)
        out = {}
        for col in ("mean", "lo", "hi"):
            v = np.where(use_fine, fine[col].values, coarse[col].values)
            out[col] = np.nan_to_num(v, nan=0.0)        # station never seen at this hour: predict zero
        return out["mean"], out["lo"], out["hi"]


# --------------------------------------------------------------------------- Level 2
SHARED_FEATURES = [
    # calendar
    "hour", "dow", "is_weekend", "is_holiday",
    # where the station is
    "lat", "lon",
    # what just happened at the station
    "net_last1", "net_last3", "net_last6", "dep_last1", "arr_last1", "dep_last3", "arr_last3",
    "trips_last24", "trips_per_day_7d",
]


def features_for(h, with_station_code=False):
    cols = SHARED_FEATURES + [
        # what this station did in the same window before
        f"net_yesterday_h{h}", f"net_lastweek_h{h}", f"net_avg4w_h{h}", f"trips_avg4w_h{h}",
        # weather expected over the window
        f"temp_h{h}", f"precip_h{h}", f"wind_h{h}",
    ]
    return cols + (["station_code"] if with_station_code else [])


class Level2:
    """Four gradient-boosting models per horizon, all on the same features:
       one for the average outcome (the forecast), and one each for the 10th, 50th and 90th percentile."""

    def __init__(self, h, with_station_code=False, max_leaf_nodes=31, max_iter=400, learning_rate=0.06):
        self.h = h
        self.cols = features_for(h, with_station_code)
        self.params = dict(max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
                           min_samples_leaf=100, l2_regularization=1.0, early_stopping=False, random_state=0,
                           categorical_features=["station_code"] if with_station_code else None)

    def _new(self, **loss):
        return HistGradientBoostingRegressor(**self.params, **loss)

    def fit(self, df, only_mean=False):
        X, y = df[self.cols], df[f"y_h{self.h}"]
        self.mean = self._new(loss="squared_error").fit(X, y)
        if not only_mean:
            self.q = {q: self._new(loss="quantile", quantile=q).fit(X, y) for q in QUANTILES}
        return self

    def predict_mean(self, df):
        return self.mean.predict(df[self.cols])

    def predict(self, df, weather="forecast"):
        """weather="actual" swaps in the weather that really happened. That is cheating,
        and is only used to measure how much honest weather costs."""
        X = df[self.cols]
        if weather == "actual":
            X = X.copy()
            for var in ("temp", "precip", "wind"):
                X[f"{var}_h{self.h}"] = df[f"{var}_act_h{self.h}"].values
        mean = self.mean.predict(X)
        lo, mid, hi = (self.q[q].predict(X) for q in QUANTILES)
        # Separate models can cross by a hair. Sort so that lo <= mid <= hi always holds.
        lo, mid, hi = np.sort(np.stack([lo, mid, hi]), axis=0)
        return mean, lo, mid, hi


# --------------------------------------------------------------------------- self-correcting band
def self_correcting_band(hour_idx, h, y, lo, hi, target, lookback_days=28, min_days=7):
    """Stretch or shrink the band using its own recent track record.

    Each day, look at the forecasts from the previous `lookback_days` days of data whose outcome
    is already known, and find the stretch that would have made the band hold `target` of them.
    Apply that stretch to today's bands. Nothing from today or later is used.

    The stretch is measured in half-band-widths: 0 = leave the band alone, 0.2 = push each edge
    out by 20% of half the band's width, negative = pull the edges in.
    For the first `min_days` days there is no track record, so the band is left alone.
    Returns (new_lo, new_hi, stretch_used_per_row).
    """
    hour_idx, y, lo, hi = (np.asarray(a, dtype=float) for a in (hour_idx, y, lo, hi))
    half = np.maximum((hi - lo) / 2, 0.5)
    miss = np.maximum(lo - y, y - hi) / half           # how far outside the band, in half-widths (negative = inside)
    day = (hour_idx // 24).astype(int)                 # the day the forecast was made
    known_from = np.ceil((hour_idx + h) / 24)          # first day on which its outcome is known

    days = np.unique(day)
    rows_of = {d: np.flatnonzero(day == d) for d in days}
    stretch = np.zeros(len(y))
    for k, d in enumerate(days):
        past = days[max(0, k - lookback_days):k]
        if len(past) < min_days:
            continue
        pool = np.concatenate([rows_of[p] for p in past])
        pool = pool[known_from[pool] <= d]
        stretch[rows_of[d]] = max(np.quantile(miss[pool], target), -0.5)
    return lo - stretch * half, hi + stretch * half, stretch
