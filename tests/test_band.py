"""The self-correcting band: it must fix a band that is too narrow, and it must not see the future."""
import numpy as np

from models import self_correcting_band
from scoring import coverage


def make(seed=0, days=80, per_hour=20, h=3):
    rng = np.random.default_rng(seed)
    hour_idx = np.repeat(np.arange(days * 24), per_hour) + 480_000
    y = rng.normal(0, 10, len(hour_idx))
    return hour_idx, h, y


def test_a_band_that_is_too_narrow_gets_widened_to_the_target():
    hour_idx, h, y = make()
    lo, hi = np.full(len(y), -6.0), np.full(len(y), 6.0)       # holds about 45% of a spread-10 outcome
    assert coverage(y, lo, hi) < 0.5
    new_lo, new_hi, stretch = self_correcting_band(hour_idx, h, y, lo, hi, target=0.8)
    later = hour_idx >= hour_idx.min() + 24 * 10                # after the warm-up
    assert abs(coverage(y[later], new_lo[later], new_hi[later]) - 0.8) < 0.02
    assert (stretch[later] > 0).all()


def test_a_band_that_is_too_wide_gets_narrowed():
    hour_idx, h, y = make()
    lo, hi = np.full(len(y), -30.0), np.full(len(y), 30.0)
    new_lo, new_hi, _ = self_correcting_band(hour_idx, h, y, lo, hi, target=0.8)
    later = hour_idx >= hour_idx.min() + 24 * 10
    assert (new_hi[later] < 30).all()


def test_the_first_days_are_left_alone():
    hour_idx, h, y = make()
    lo, hi = np.full(len(y), -6.0), np.full(len(y), 6.0)
    _, _, stretch = self_correcting_band(hour_idx, h, y, lo, hi, target=0.8, min_days=7)
    assert (stretch[hour_idx < hour_idx.min() + 24 * 7] == 0).all()


def test_changing_the_future_does_not_change_an_earlier_band():
    hour_idx, h, y = make()
    lo, hi = np.full(len(y), -6.0), np.full(len(y), 6.0)
    _, _, before = self_correcting_band(hour_idx, h, y, lo, hi, target=0.8)
    cut = hour_idx.min() + 24 * 40
    y2 = y.copy()
    # Rewrite every outcome that is not yet known at the start of day 40.
    y2[hour_idx + h > cut] += 500
    _, _, after = self_correcting_band(hour_idx, h, y2, lo, hi, target=0.8)
    same_day_or_earlier = hour_idx < cut + 24
    assert np.array_equal(before[same_day_or_earlier], after[same_day_or_earlier])
    assert not np.array_equal(before[~same_day_or_earlier], after[~same_day_or_earlier])
