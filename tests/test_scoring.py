"""Scoring checked against numbers worked out by hand."""
import numpy as np
import pytest

import scoring as s


def test_mae_by_hand():
    assert s.mae([2, -4, 0], [0, 0, 0]) == pytest.approx(2.0)      # (2 + 4 + 0) / 3
    assert s.mae([2, -4, 0], [1, -1, 0]) == pytest.approx(4 / 3)   # (1 + 3 + 0) / 3


def test_big_imbalance_is_the_top_ten_percent_of_absolute_net():
    y = np.array([0] * 90 + [5, -6, 7, -8, 9, -10, 11, -12, 13, -14])
    cutoff = s.big_imbalance_cutoff(y, 0.10)
    mask = s.big_imbalance_mask(y, cutoff)
    assert mask.sum() == 10
    assert set(np.abs(y[mask])) == {5, 6, 7, 8, 9, 10, 11, 12, 13, 14}   # sign does not matter


def test_big_imbalance_never_includes_zero_windows():
    y = np.array([0] * 99 + [3])
    mask = s.big_imbalance_mask(y, s.big_imbalance_cutoff(y, 0.10))    # cutoff is 0 here
    assert mask.sum() == 1


def test_predict_zero_scores_the_mean_absolute_net_on_big_windows():
    y = np.array([10, -20])
    assert s.mae(y, np.zeros(2)) == pytest.approx(15.0)


def test_skill():
    assert s.skill(5.0, 10.0) == pytest.approx(0.5)
    assert s.skill(10.0, 10.0) == pytest.approx(0.0)
    assert s.skill(12.0, 10.0) < 0


def test_coverage_counts_edges_as_inside():
    assert s.coverage([1, 2, 3, 10], lo=[0, 2, 4, 0], hi=[2, 2, 5, 9]) == pytest.approx(0.5)


def test_pinball_by_hand():
    # q = 0.9, forecast 10. Outcome 12: too low by 2, costs 0.9 * 2 = 1.8.
    # Outcome 7: too high by 3, costs 0.1 * 3 = 0.3. Mean = 1.05.
    assert s.pinball([12, 7], [10, 10], 0.9) == pytest.approx(1.05)


def test_pinball_at_the_median_is_half_the_mae():
    y, p = np.array([3.0, -1.0, 4.0]), np.array([1.0, 1.0, 1.0])
    assert s.pinball(y, p, 0.5) == pytest.approx(s.mae(y, p) / 2)


def test_direction_hit_rate():
    assert s.direction_hit_rate([5, -5, 5, -5], [2, -1, -3, 0.2]) == pytest.approx(0.5)
