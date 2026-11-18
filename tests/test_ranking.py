"""Ranking and decision metrics, checked on a six-station example worked out by hand."""
import numpy as np
import pytest

import ranking as r

T = np.zeros(6, int)                       # one decision time
S = np.arange(1, 7)                        # stations 1..6
SCORE = np.array([5.0, 4, 3, 2, 1, 0])     # forecast size: station 1 is the top pick
POS = np.array([True, False, True, False, False, True])    # stations 1, 3, 6 were really big


def test_hand_worked_example():
    t = r.pick_table(T, S, SCORE, POS, ks=[2, 3])
    assert t.n_cand.iloc[0] == 6 and t.n_pos.iloc[0] == 3
    # K=2 picks stations 1 and 2: one of them (1) was big
    assert t.hits_2.iloc[0] == 1 and t.cap_2.iloc[0] == pytest.approx(1 / 2)       # 1 / min(2, 3)
    # K=3 adds station 3, also big
    assert t.hits_3.iloc[0] == 2 and t.cap_3.iloc[0] == pytest.approx(2 / 3)       # 2 / min(3, 3)


def test_perfect_ranking_captures_everything_and_reversed_the_least():
    pos = np.array([True, True, True, False, False, False])
    assert r.pick_table(T, S, SCORE, pos, [3]).cap_3.iloc[0] == 1.0
    assert r.pick_table(T, S, SCORE[::-1], pos, [3]).cap_3.iloc[0] == 0.0
    # fewer positives than K: catching all of them still scores 1.0
    one = np.array([True, False, False, False, False, False])
    assert r.pick_table(T, S, SCORE, one, [5]).cap_5.iloc[0] == 1.0


def test_times_with_no_positives_are_left_out_of_capture():
    t = np.array([0, 0, 1, 1])
    tab = r.pick_table(t, np.array([1, 2, 1, 2]), np.array([2.0, 1, 2, 1]), np.array([True, False, False, False]), [1])
    tot = {"cap_x_1": tab.cap_1.sum(), "has_pos": float((tab.n_pos > 0).sum())}
    assert r.capture(tot, "x", 1) == 1.0          # only time 0 counts, and it caught its one positive


def test_ties_break_on_station_id_whatever_the_row_order():
    rng = np.random.default_rng(1)
    score = np.array([1.0, 1.0, 1.0, 0.0])
    ids = np.array([30, 10, 20, 40])
    base = r.rank_within_time(np.zeros(4, int), ids, score)
    assert dict(zip(ids, base)) == {10: 0, 20: 1, 30: 2, 40: 3}
    perm = rng.permutation(4)
    shuffled = r.rank_within_time(np.zeros(4, int), ids[perm], score[perm])
    assert dict(zip(ids[perm], shuffled)) == dict(zip(ids, base))


def test_tiebreak_column_is_used_before_station_id():
    rank = r.rank_within_time(np.zeros(3, int), np.array([1, 2, 3]), np.zeros(3), tiebreak=np.array([0.1, 0.9, 0.5]))
    assert list(rank) == [2, 0, 1]


def test_ranks_are_per_time():
    t = np.array([0, 0, 1, 1])
    rank = r.rank_within_time(t, np.array([1, 2, 1, 2]), np.array([1.0, 2, 5, 3]))
    assert list(rank) == [1, 0, 0, 1]


def test_decision_times_line_up_and_never_overlap():
    hours = np.arange(48) % 24
    assert r.decision_mask(hours, 1).all()
    m3 = r.decision_mask(hours, 3)
    assert set(hours[m3]) == {0, 3, 6, 9, 12, 15, 18, 21}
    m6 = r.decision_mask(hours, 6)
    assert set(hours[m6]) == {0, 6, 12, 18}
    assert np.diff(np.flatnonzero(m3)).min() == 3      # windows of 3 h start 3 h apart: no overlap


def test_random_floor_is_exact_expectation():
    base = r.pick_table(T, S, SCORE, POS, [2])
    rnd = r.random_table(base, [2])
    assert rnd.hits_2.iloc[0] == pytest.approx(2 * 3 / 6)
    assert rnd.cap_2.iloc[0] == pytest.approx(1.0 / 2)


def test_rankings_use_only_forecasts_never_the_outcome():
    """Shuffle what really happened: the picks must not move, only their score changes."""
    rng = np.random.default_rng(0)
    n = 60
    t, s = np.repeat(np.arange(6), 10), np.tile(np.arange(10), 6)
    score = rng.normal(size=n)
    before = r.rank_within_time(t, s, score)
    # the ranking function has no outcome argument at all; picks derive from score alone
    assert (before == r.rank_within_time(t, s, score.copy())).all()
    pos1, pos2 = rng.random(n) < .2, rng.random(n) < .2
    a = r.pick_table(t, s, score, pos1, [3])
    b = r.pick_table(t, s, score, pos2, [3])
    picked = r.rank_within_time(t, s, score) < 3
    assert a.hits_3.sum() == (picked & pos1).sum() and b.hits_3.sum() == (picked & pos2).sum()


def test_direction_counts_only_big_picks_with_the_right_sign():
    pred = np.array([5.0, -4, 3, 2, 1, 0])
    y = np.array([9.0, 9, 9, -1, -1, 0])          # station 2 was big but went the other way
    pos = np.array([True, True, True, False, False, False])
    tab = r.pick_table(T, S, np.abs(pred), pos, [3], pred=pred, y=y)
    assert tab.hits_3.iloc[0] == 3 and tab.dir_3.iloc[0] == 2


def test_edge_from_zero():
    e = r.edge_from_zero([2, -3, -1], [6, -1, 4])
    assert list(e) == [2, 1, 0]


def test_usual_suspects_use_only_training_years():
    import pandas as pd
    from s05d_decision_check import usual_suspects
    base = pd.DataFrame({"station_id": [1, 1, 1], "hour_of_week": [5, 5, 5], "year": [2022, 2023, 2024],
                         "y_h3": [4.0, -6.0, 1000.0]})
    a, n = usual_suspects(base, 3)
    assert a.loc[(1, 5)] == 5.0 and n.loc[(1, 5)] == -1.0          # the 2024 row is ignored
