"""Station names: spelling differences must not create a second station."""
from s02_stations import normalise, to_metres
import numpy as np


def test_spelling_variants_are_the_same_name():
    assert normalise("Métro Mont-Royal (Rivard / du Mont-Royal)") == normalise("Metro  Mont-Royal (Rivard/du Mont-Royal )")
    assert normalise("Roy / Saint-Laurent") == normalise("Roy / St-Laurent")
    assert normalise("Guy / Sainte-Catherine") == normalise("Guy / Ste-Catherine")
    assert normalise("2e avenue /centrale") == normalise("2e avenue / Centrale")


def test_different_stations_stay_different():
    assert normalise("de Maisonneuve / Mansfield (nord)") != normalise("de Maisonneuve / Mansfield (sud)")
    assert normalise("Monk / Biencourt") != normalise("Monk / de Biencourt")     # a real rename: merged by distance, not name


def test_metres_are_about_right():
    # One thousandth of a degree of latitude is about 111 m; of longitude at Montreal, about 78 m.
    a, b, c = to_metres(np.array([45.510, 45.511, 45.510]), np.array([-73.570, -73.570, -73.569]))
    assert 108 < np.hypot(*(b - a)) < 113
    assert 76 < np.hypot(*(c - a)) < 80
