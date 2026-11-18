"""Leakage tests: a forecast made at hour t must not change if the future changes."""
import numpy as np
import pandas as pd

from config import HORIZONS, RUN_DELAY_H
from s04_features import traffic_features, weather_features, window_sum


def make_counts(seed=0, stations=3, hours=24 * 40):
    rng = np.random.default_rng(seed)
    return rng.poisson(3, (stations, hours)).astype(float), rng.poisson(3, (stations, hours)).astype(float)


def test_window_sum_by_hand():
    x = np.array([[1.0, 2, 3, 4, 5]])
    assert np.array_equal(window_sum(x, 0, 2), [[3, 5, 7, 9, np.nan]], equal_nan=True)    # this hour + next
    assert np.array_equal(window_sum(x, -2, 2), [[np.nan, np.nan, 3, 5, 7]], equal_nan=True)  # two hours before


def test_target_is_net_over_the_next_h_hours():
    dep, arr = make_counts()
    f = traffic_features(dep, arr, np.ones_like(dep, bool))
    t = 500
    for h in HORIZONS:
        expected = (arr - dep)[:, t:t + h].sum(axis=1)
        assert np.allclose(f[f"y_h{h}"][:, t], expected)


def test_features_do_not_move_when_the_future_changes():
    dep, arr = make_counts()
    active = np.ones_like(dep, bool)
    t = 700
    before = traffic_features(dep, arr, active)
    dep2, arr2 = dep.copy(), arr.copy()
    dep2[:, t:] += 50          # rewrite everything from hour t onward
    arr2[:, t:] = 0
    after = traffic_features(dep2, arr2, active)
    for name in before:
        if name.startswith("y_"):
            assert not np.allclose(before[name][:, t], after[name][:, t])      # the target must move
        else:
            assert np.allclose(before[name][:, t], after[name][:, t], equal_nan=True), name


def test_backtest_weather_only_uses_runs_already_published():
    hours = np.arange(1000, 1100)
    actual = pd.DataFrame({"hour_idx": np.arange(900, 1200), "temp": 20.0, "precip": 0.0, "wind": 5.0})
    # Two runs. Each one "forecasts" its own run number as the temperature, so we can see which was used.
    runs = pd.concat([
        pd.DataFrame({"run_idx": r, "hour_idx": np.arange(r, r + 90), "temp": float(r), "precip": 0.0, "wind": 5.0})
        for r in (980, 1040)
    ])
    w = weather_features(hours, actual, runs)
    used = w["temp_h1"]
    for t, temp in zip(hours, used):
        if t < 980 + RUN_DELAY_H:
            assert np.isnan(temp)                     # no run published yet
        elif t < 1040 + RUN_DELAY_H:
            assert temp == 980                        # the 1040 run exists but is not published yet
        else:
            assert temp == 1040


def test_training_weather_is_the_window_average():
    hours = np.arange(100, 110)
    actual = pd.DataFrame({"hour_idx": np.arange(90, 130), "temp": np.arange(90, 130, dtype=float),
                           "precip": 1.0, "wind": 0.0})
    w = weather_features(hours, actual)
    assert w["temp_h3"][0] == np.mean([100, 101, 102])
    assert w["precip_h6"][0] == 6.0


def test_out_of_service_hours_are_skipped_not_counted_as_zero():
    dep, arr = make_counts()
    active = np.ones_like(dep, bool)
    t = 24 * 35
    active[:, t - 168:t - 144] = False            # out of service on the same day one week ago
    f = traffic_features(dep, arr, active)
    net = arr - dep
    assert np.isnan(f["net_lastweek_h3"][:, t]).all()
    expected = np.mean([net[:, t - 168 * k:t - 168 * k + 3].sum(axis=1) for k in (2, 3, 4)], axis=0)
    assert np.allclose(f["net_avg4w_h3"][:, t], expected)       # average of the three weeks that exist
    assert not np.isnan(f["net_avg4w_h3"][:, t + 100]).any()    # and later hours are unaffected


def test_persistence_is_the_window_just_before_and_ignores_the_future():
    from models import predict_persistence
    dep, arr = make_counts()
    active = np.ones_like(dep, bool)
    t = 700
    f = traffic_features(dep, arr, active)
    for h in HORIZONS:
        df = pd.DataFrame({f"net_last{h}": f[f"net_last{h}"][:, t]})
        expected = (arr - dep)[:, t - h:t].sum(axis=1)       # the h hours that ended just before t
        assert np.allclose(predict_persistence(df, h), expected)
    dep2, arr2 = dep.copy(), arr.copy()
    dep2[:, t:] += 50
    arr2[:, t:] = 0
    g = traffic_features(dep2, arr2, active)
    for h in HORIZONS:
        assert np.allclose(f[f"net_last{h}"][:, t], g[f"net_last{h}"][:, t])      # unchanged by the future
