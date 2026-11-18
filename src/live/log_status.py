"""Step 2a. Log BIXI's live station feed, a few minutes apart, forever.

Run it from a scheduler every 5 minutes:   .venv/bin/python src/live/log_status.py
Each run fetches station_status.json once and stores it in live/bixi_live.sqlite.

What is stored (small on purpose, about 1 MB a day instead of ~17):
  polls    one row per run: when, HTTP status, feed timestamp, stations seen, rows changed, error.
           This is the truth about gaps. A time with no poll row is a gap, never "no change".
  status   a station gets a row only when one of its numbers changed since the last row, plus a full
           copy of every station on the first poll of each local day (a keyframe, so one lost row
           cannot corrupt a whole day). To get the value at any time, take the latest row at or before it.
  latest   the most recent state of each station (what "changed" is compared with).
Idempotent: if the feed has not refreshed since the last poll, nothing but a poll row is written.
Also saves BIXI's station list (names, coordinates, docks) once a day to live/info/.
Never crashes silently: every failure is a poll row with the error text.
"""
import gzip
import json
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "live"
STATUS_URL = "https://gbfs.velobixi.com/gbfs/en/station_status.json"
INFO_URL = "https://gbfs.velobixi.com/gbfs/en/station_information.json"
TZ = ZoneInfo("America/Montreal")
POLL_SECONDS = 300
FIELDS = ["num_bikes_available", "num_ebikes_available", "num_bikes_disabled", "num_docks_available",
          "num_docks_disabled", "is_installed", "is_renting", "is_returning", "last_reported"]

SCHEMA = """
create table if not exists polls (polled_at integer primary key, http_status integer, last_updated integer,
                                  n_stations integer, n_changed integer, error text);
create table if not exists status (polled_at integer, station_id text, bikes integer, ebikes integer,
                                   bikes_disabled integer, docks integer, docks_disabled integer, is_installed integer,
                                   is_renting integer, is_returning integer, last_reported integer,
                                   primary key (polled_at, station_id));
create table if not exists latest (station_id text primary key, bikes integer, ebikes integer, bikes_disabled integer,
                                   docks integer, docks_disabled integer, is_installed integer, is_renting integer,
                                   is_returning integer, last_reported integer);
"""


def to_int(v):
    """Feed numbers sometimes arrive as strings or booleans, or not at all. Anything unusable becomes None."""
    if isinstance(v, bool):
        return int(v)
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def parse_status(js):
    """Feed JSON -> (feed timestamp, {station_id: tuple of the nine numbers}). Raises ValueError if the shape is wrong."""
    try:
        stations = js["data"]["stations"]
        updated = to_int(js.get("last_updated"))
    except (KeyError, TypeError, AttributeError) as e:
        raise ValueError(f"unexpected feed shape: {e!r}")
    if not isinstance(stations, list):
        raise ValueError("stations is not a list")
    rows = {}
    for s in stations:
        sid = s.get("station_id") if isinstance(s, dict) else None
        if sid is None:
            continue
        rows[str(sid)] = tuple(to_int(s.get(f)) for f in FIELDS)
    return updated, rows


def connect(path=None):
    LIVE.mkdir(exist_ok=True)
    db = sqlite3.connect(path or LIVE / "bixi_live.sqlite", timeout=30)
    db.execute("pragma journal_mode=wal")
    db.executescript(SCHEMA)
    return db


def local_day(ts):
    return datetime.fromtimestamp(ts, TZ).strftime("%Y-%m-%d")


def fetch_json(url, timeout=20):
    """Returns (http_status, parsed json or None, error text or None). One retry after a short pause."""
    last = None
    for attempt in range(2):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "bixi-net-flow-forecast (open-data logger)"})
            if r.status_code == 200:
                return 200, r.json(), None
            last = (r.status_code, None, f"HTTP {r.status_code}")
        except Exception as e:                                   # network, timeout, bad JSON: all just recorded
            last = (None, None, f"{type(e).__name__}: {e}"[:300])
        time.sleep(2 if attempt == 0 else 0)
    return last


def poll_once(db, fetch=fetch_json, now=None):
    """One poll. Returns a dict describing what happened (also stored in `polls`)."""
    now = int(now if now is not None else time.time())
    code, js, err = fetch(STATUS_URL)
    updated, rows = None, {}
    if err is None:
        try:
            updated, rows = parse_status(js)
            if not rows:
                err = "feed had no stations"
        except ValueError as e:
            err = str(e)
    if err is not None:
        db.execute("insert or replace into polls values (?,?,?,?,?,?)", (now, code, updated, len(rows), 0, err))
        db.commit()
        return {"polled_at": now, "ok": False, "error": err, "n_changed": 0}

    prev_updated = db.execute("select last_updated from polls where error is null order by polled_at desc limit 1").fetchone()
    prev_day = db.execute("select max(polled_at) from polls where error is null").fetchone()[0]
    fresh_feed = prev_updated is None or prev_updated[0] != updated
    keyframe = prev_day is None or local_day(prev_day) != local_day(now)
    changed = 0
    if fresh_feed or keyframe:
        old = {r[0]: tuple(r[1:]) for r in db.execute("select * from latest")}
        for sid, vals in rows.items():
            if keyframe or old.get(sid) != vals:
                db.execute("insert or replace into status values (?,?,?,?,?,?,?,?,?,?,?)", (now, sid) + vals)
                db.execute("insert or replace into latest values (?,?,?,?,?,?,?,?,?,?)", (sid,) + vals)
                changed += 1
    db.execute("insert or replace into polls values (?,?,?,?,?,?)", (now, code, updated, len(rows), changed, None))
    db.commit()
    return {"polled_at": now, "ok": True, "n_stations": len(rows), "n_changed": changed, "fresh_feed": fresh_feed}


def snapshot_info(now=None, fetch=fetch_json):
    """Save the station list once per local day (names, coordinates, docks). Returns the path, or None if already saved or failed."""
    now = int(now if now is not None else time.time())
    out = LIVE / "info" / f"{local_day(now)}.json.gz"
    if out.exists():
        return None
    code, js, err = fetch(INFO_URL)
    if err is not None or not isinstance(js, dict) or "data" not in js:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(json.dumps(js).encode()))
    tmp.replace(out)
    return out


def summary(db, hours=24, now=None):
    """Health at a glance: how many polls in the last `hours`, how many were missed, how stale the newest data is."""
    now = int(now if now is not None else time.time())
    since = now - hours * 3600
    n, ok = db.execute("select count(*), sum(error is null) from polls where polled_at >= ?", (since,)).fetchone()
    last = db.execute("select max(polled_at) from polls where error is null").fetchone()[0]
    expected = hours * 3600 // POLL_SECONDS
    return {"hours": hours, "polls": n, "ok": int(ok or 0), "expected": expected,
            "missed_share": max(0.0, 1 - (ok or 0) / expected), "minutes_since_last_ok": None if last is None else (now - last) / 60}


def main():
    db = connect()
    if "--summary" in sys.argv:
        print(json.dumps(summary(db), indent=1))
        return 0
    res = poll_once(db)
    try:
        snapshot_info()
    except Exception as e:
        print(f"station list not saved: {e!r}", file=sys.stderr)
    print(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), res)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
