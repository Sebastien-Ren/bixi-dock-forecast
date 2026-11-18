"""Checks for the decision study and the model study (dated plan notes)."""
import numpy as np
import pandas as pd

import s05f_model_study as ms
import s05g_decision_study as ds


def test_stock_score_ranks_nearest_edge_first():
    # (bikes, docks): 0 bikes = empty now; 1 free dock = almost full; 10/10 = comfortable
    s = ds.stock_score([0, 9, 10], [20, 1, 10])
    assert s.tolist() == [0, -1, -10]
    assert np.argsort(-s).tolist() == [0, 1, 2]


def test_problem_chance_takes_the_larger():
    assert ds.problem_chance(np.array([0.1, 0.7]), np.array([0.3, 0.2])).tolist() == [0.3, 0.7]


def test_neighbour_matrix_by_distance():
    st = pd.DataFrame({"station_id": [1, 2, 3], "lat": [45.5, 45.5036, 45.52], "lon": [-73.57, -73.57, -73.57]})
    ids, near = ms.neighbour_matrix(st, radius=600)      # 2 is about 400 m north of 1; 3 is about 2.2 km away
    assert ids.tolist() == [1, 2, 3]
    assert near.tolist() == [[False, True, False], [True, False, False], [False, False, False]]


def test_neighbour_features_use_only_the_same_hour():
    ids = np.array([1, 2])
    near = np.array([[False, True], [True, False]])
    df = pd.DataFrame({"station_id": [1, 2, 1, 2], "hour_idx": [10, 10, 11, 11], "net_last1": [1.0, 5.0, 2.0, 7.0]})
    out = ms.add_neighbour_features(df, ids, near, ["net_last1"])
    assert out.nb_net_last1.tolist() == [5.0, 1.0, 7.0, 2.0]
    # changing hour 11 cannot change hour 10
    df2 = df.copy()
    df2.loc[df2.hour_idx == 11, "net_last1"] = 99.0
    assert ms.add_neighbour_features(df2, ids, near, ["net_last1"]).nb_net_last1.tolist()[:2] == [5.0, 1.0]


def test_flow_targets_by_hand():
    hourly = pd.DataFrame({"station_id": 1, "hour_idx": [0, 1, 2, 3], "arr": [1, 2, 3, 4], "dep": [0, 1, 0, 2]})
    df = pd.DataFrame({"station_id": [1, 1], "hour_idx": [0, 1]})
    arr, dep = ms.flow_targets(hourly, df, 3)
    assert arr.tolist() == [6.0, 9.0] and dep.tolist() == [1.0, 3.0]      # hours 0-2 and 1-3


def test_real_change_counts_disabled_bikes_and_is_end_minus_start():
    d = pd.DataFrame({"num_bikes_available0": [10, 3], "num_bikes_disabled0": [1, 0],
                      "num_bikes_available1": [4, 8], "num_bikes_disabled1": [2, 0]})
    assert ds.real_change(d).tolist() == [-5.0, 5.0]       # (4 + 2) - (10 + 1), and 8 - 3


def _rc_frame(real, trips, fc, lo=None, hi=None):
    n = len(real)
    return pd.DataFrame({"day": ["d1"] * (n // 2) + ["d2"] * (n - n // 2), "real_change": real, "y": trips, "l2": fc,
                         "l2_lo": fc - 1 if lo is None else lo, "l2_hi": fc + 1 if hi is None else hi})


def test_real_change_stats_by_hand():
    import s05i_real_change_check as rc
    real = np.array([10.0, -10.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0, 1.0, -1.0])
    # a forecast equal to the real change: perfect on everything measured against real change
    e = _rc_frame(real, real * 3, real)
    t, _ = rc.day_table(e)
    s = rc.derive(t.drop(columns="day").sum().values)
    assert s["skill_vs_zero_real"] == 1.0 and abs(s["corr_forecast_real"] - 1.0) < 1e-12
    assert s["direction_right_big_real"] == 1.0 and s["range_held_real"] == 1.0
    # the same forecast scored against trips that are 3x bigger misses by 2|r| in total: 2 * 26 = 52 of 78
    assert abs(s["skill_vs_zero_trips"] - (1 - 52 / 78)) < 1e-12
    # a forecast of zero removes none of the error against real change, and the correlation is undefined (0 spread)
    z = _rc_frame(real, real, np.zeros(10))
    tz, _ = rc.day_table(z)
    with np.errstate(invalid="ignore", divide="ignore"):
        assert rc.derive(tz.drop(columns="day").sum().values)["skill_vs_zero_real"] == 0.0
