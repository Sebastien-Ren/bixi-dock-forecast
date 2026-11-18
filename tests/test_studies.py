"""Checks for the decision study and the model study (plan/06)."""
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
