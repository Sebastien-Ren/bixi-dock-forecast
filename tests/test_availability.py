"""Checks for the empty/full probability step (dated plan notes)."""
import numpy as np
import pandas as pd

import availability as av
import s05e_availability_check as chk


def test_nearest_snapshot_within_ten_minutes_only():
    snaps = np.array([0, 900, 3_500, 7_900])
    marks = np.array([0, 3_600, 7_200, 10_800])
    # 0 -> 0 exactly; 3600 -> 3500 (100 s); 7200 -> 7900 is 700 s away, too far; 10800 -> nothing near
    assert av.nearest_snapshot(marks, snaps).tolist() == [0, 2, -1, -1]


def test_direct_probability_by_hand():
    # 3 bikes now, expected change -3, band -3 +/- 1.2816 -> sigma 1. P(end <= 0.5) = P(z <= 0.5) = 0.6915
    p = av.prob_empty_direct([3], [-3], [-3 - av.Z90], [-3 + av.Z90])
    assert abs(p[0] - 0.6915) < 1e-3
    # more bikes now, or a gain expected, makes empty less likely
    assert av.prob_empty_direct([10], [-3], [-5], [-1])[0] < av.prob_empty_direct([3], [-3], [-5], [-1])[0]
    assert av.prob_empty_direct([3], [2], [0], [4])[0] < av.prob_empty_direct([3], [-2], [-4], [0])[0]
    # full mirrors empty: 3 free docks and a gain of 3 is the same as 3 bikes and a loss of 3
    assert abs(av.prob_full_direct([3], [3], [3 - av.Z90], [3 + av.Z90])[0] - p[0]) < 1e-9


def test_tiny_band_does_not_give_certainty():
    p = av.prob_empty_direct([1], [0], [0], [0])
    assert 0 < p[0] < 1


def test_buckets():
    assert av.bucket([0, 1, 5, 6, 8, 9, 12, 13, 20, 21, 300]).tolist() == [0, 1, 5, 6, 6, 7, 7, 8, 8, 9, 9]


def test_brier_by_hand():
    assert av.brier([1, 0, 0.5], [1, 0, 1]) == (0 + 0 + 0.25) / 3
    assert av.brier([0.5] * 4, [0, 1, 0, 1]) == 0.25


def test_rate_table_uses_only_what_it_is_given_and_falls_back():
    rng = np.random.default_rng(0)
    n = 2_000
    train = pd.DataFrame({"bikes0": rng.integers(0, 4, n), "h": 3, "local_hour": rng.integers(0, 24, n)})
    train["e"] = (train.bikes0 == 0).astype(int)              # empty only when starting empty
    predict = av.rate_table(train, "bikes0", "e")
    test = pd.DataFrame({"bikes0": [0, 2, 30], "h": 3, "local_hour": [5, 5, 5]})
    p = predict(test)
    assert p[0] == 1.0 and p[1] == 0.0
    assert p[2] == train.e.mean()                              # bucket never seen in training: overall rate
    # changing the test outcomes cannot change the table
    test2 = test.assign(e=[0, 1, 1])
    assert np.array_equal(predict(test2), p)


def test_recalibration_fixes_a_biased_probability():
    rng = np.random.default_rng(1)
    true = rng.uniform(0.01, 0.6, 20_000)
    y = rng.uniform(size=true.size) < true
    stated = np.clip(true * 1.6, 0, 0.999)                     # says 1.6x too much
    fixed = av.recalibrate(stated, y)(stated)
    assert av.brier(fixed, y) < av.brier(stated, y)


def test_calibration_table_bins():
    rows = av.calibration_table([0.05, 0.05, 0.95], [0, 1, 1])
    assert rows[0]["bin"] == "0.0-0.1" and rows[0]["n"] == 2 and rows[0]["happened"] == 0.5
    assert rows[-1]["bin"] == "0.9-1.0" and rows[-1]["happened"] == 1.0


def test_events_and_in_service_rule():
    states = pd.DataFrame({
        "our_id": [7, 7, 7], "hour_idx": [100, 101, 103],
        "num_bikes_available": [2, 0, 5], "num_bikes_disabled": [0, 0, 0], "num_docks_available": [10, 12, 0],
        "is_installed": [1, 1, 1], "is_renting": [1, 1, 0], "is_returning": [1, 1, 1], "capacity": [12, 12, 12]})
    preds = pd.DataFrame({"station_id": [7, 7, 7], "hour_idx": [100, 100, 101], "year": 2025, "h": [1, 3, 1],
                          "y": [-2, 3, 0], "l2": 0.0, "l2_lo": -1.0, "l2_hi": 1.0})
    d, counts = chk.build_pairs(states, preds)
    assert counts == {"forecasts_2025": 3, "with_both_snapshots": 2}     # 101 + 1 = 102 has no snapshot
    one = d[d.h == 1].iloc[0]
    assert one.is_empty0 == 0 and one.is_empty1 == 1                     # 2 bikes -> 0 bikes
    three = d[d.h == 3].iloc[0]
    assert three.is_full1 == 1                                           # 0 free docks at hour 103
    # at hour 103 the station is not renting, so "empty" is undefined there, but "full" is defined
    assert not chk.in_service(d[d.h == 3], "empty").iloc[0]
    assert chk.in_service(d[d.h == 3], "full").iloc[0]
