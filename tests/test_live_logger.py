"""The live logger, with no network: a trimmed copy of a real feed response and an injected fetch."""
import json
import sqlite3
from pathlib import Path

import pytest

from live import log_status as L

SAMPLE = json.loads((Path(__file__).parent / "fixtures" / "station_status_sample.json").read_text())


def fresh_db():
    db = sqlite3.connect(":memory:")
    db.executescript(L.SCHEMA)
    return db


def feed(updated=1000, mutate=None):
    js = json.loads(json.dumps(SAMPLE))
    js["last_updated"] = updated
    if mutate:
        mutate(js["data"]["stations"])
    return lambda url: (200, js, None)


def test_parser_reads_real_sample():
    updated, rows = L.parse_status(SAMPLE)
    assert len(rows) == 20 and all(len(v) == len(L.FIELDS) for v in rows.values())


def test_parser_survives_missing_field_string_numbers_and_bools():
    js = {"last_updated": "1000", "data": {"stations": [
        {"station_id": 7, "num_bikes_available": "3", "is_renting": True},
        {"station_id": "8"}, {"no_id": 1}, "junk"]}}
    updated, rows = L.parse_status(js)
    assert updated == 1000 and set(rows) == {"7", "8"}
    assert rows["7"][0] == 3 and rows["7"][L.FIELDS.index("is_renting")] == 1 and rows["7"][1] is None
    assert all(v is None for v in rows["8"])


def test_parser_rejects_wrong_shapes():
    for bad in ({}, {"data": {}}, {"data": {"stations": "x"}}, None):
        with pytest.raises(ValueError):
            L.parse_status(bad)


def test_empty_station_list_is_recorded_as_an_error_not_as_zero_bikes():
    db = fresh_db()
    r = L.poll_once(db, lambda u: (200, {"last_updated": 5, "data": {"stations": []}}, None), now=2000)
    assert not r["ok"] and db.execute("select count(*) from status").fetchone()[0] == 0
    assert db.execute("select error from polls").fetchone()[0] == "feed had no stations"


def test_failures_are_recorded_never_silent():
    db = fresh_db()
    r = L.poll_once(db, lambda u: (503, None, "HTTP 503"), now=2000)
    assert not r["ok"]
    assert db.execute("select http_status, error from polls").fetchone() == (503, "HTTP 503")


def test_same_feed_timestamp_twice_writes_status_once():
    db = fresh_db()
    L.poll_once(db, feed(1000), now=2000)
    n = db.execute("select count(*) from status").fetchone()[0]
    r = L.poll_once(db, feed(1000), now=2300)             # feed has not refreshed
    assert db.execute("select count(*) from status").fetchone()[0] == n
    assert r["n_changed"] == 0 and db.execute("select count(*) from polls").fetchone()[0] == 2


def test_only_changed_stations_get_a_new_row():
    db = fresh_db()
    L.poll_once(db, feed(1000), now=2000)
    def bump(st):
        st[3]["num_bikes_available"] = st[3]["num_bikes_available"] + 5
    r = L.poll_once(db, feed(1010, bump), now=2300)
    assert r["n_changed"] == 1
    sid = SAMPLE["data"]["stations"][3]["station_id"]
    rows = db.execute("select bikes from status where station_id=? order by polled_at", (sid,)).fetchall()
    assert len(rows) == 2 and rows[1][0] == rows[0][0] + 5


def test_first_poll_of_each_local_day_writes_every_station():
    db = fresh_db()
    L.poll_once(db, feed(1000), now=1_800_000_000)
    L.poll_once(db, feed(1010), now=1_800_000_300)        # same day, nothing changed
    assert db.execute("select count(*) from status").fetchone()[0] == 20
    L.poll_once(db, feed(1020), now=1_800_000_000 + 86400)   # next day: keyframe, even with no change
    assert db.execute("select count(*) from status").fetchone()[0] == 40


def test_a_failed_poll_does_not_hide_a_later_change():
    db = fresh_db()
    L.poll_once(db, feed(1000), now=2000)
    L.poll_once(db, lambda u: (None, None, "Timeout"), now=2300)
    def bump(st): st[0]["num_docks_available"] += 1
    r = L.poll_once(db, feed(1020, bump), now=2600)
    assert r["ok"] and r["n_changed"] == 1


def test_summary_counts_missed_polls():
    db = fresh_db()
    t0 = 10_000_000
    for i in range(10):
        L.poll_once(db, feed(1000 + i * 10), now=t0 + i * 300)
    s = L.summary(db, hours=1, now=t0 + 9 * 300 + 60)
    assert s["expected"] == 12 and s["ok"] == 10 and s["missed_share"] == pytest.approx(1 - 10 / 12)
    assert s["minutes_since_last_ok"] == pytest.approx(1.0)
