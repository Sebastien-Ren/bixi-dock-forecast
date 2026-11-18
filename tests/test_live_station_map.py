"""Matching live station ids to our stations, on a tiny made-up map."""
import pandas as pd

from live.station_map import match_stations


def frame(rows):
    return pd.DataFrame(rows, columns=["name", "lat", "lon"])


def test_close_points_match_and_far_ones_do_not():
    ours = frame([("Alpha / Beta", 45.5000, -73.5700), ("Gamma / Delta", 45.5200, -73.5800)])
    live = frame([("Alpha / Beta", 45.50005, -73.5700), ("Somewhere else", 45.5600, -73.6000)])
    assert match_stations(ours, live) == [0, None]


def test_same_name_within_150_m_matches_but_a_different_name_does_not():
    ours = frame([("Rivard / du Mont-Royal", 45.5000, -73.5700), ("Rivard / du Mont-Royal", 45.5100, -73.5700)])
    near = 0.0009                                          # about 100 m north
    live = frame([("Rivard / du Mont-Royal", 45.5000 + near, -73.5700), ("Other name", 45.5100 + near, -73.5700)])
    assert match_stations(ours, live) == [0, None]


def test_saint_spelling_is_ignored_and_nearest_wins():
    ours = frame([("Saint-Denis / Rachel", 45.5000, -73.5700)])
    live = frame([("St-Denis / Rachel", 45.5000 + 0.0005, -73.5700), ("Elsewhere", 45.50001, -73.5700)])
    assert match_stations(ours, live) == [1]               # 1 m away beats a same-name station 55 m away
