"""Past inventory: turn Max Halford's snapshots of BIXI's station feed into one table.

Source: https://github.com/MaxHalford/bike-sharing-history, branch city/montreal. A bot saves the station feed
about every 15 minutes; each saved copy is a git commit. We keep a partial clone (history only, no file contents)
in data/raw/halford_montreal.git and fetch just the copies for the months we ask for.
Credit: BIXI Montréal (open data, attribution) and the archive's author. Keep the downloaded data private.

Run:  .venv/bin/python src/archive/build_inventory.py 2025-06-01 2025-07-01
Writes data/interim/inventory_<since>_<until>.parquet: one row per station per snapshot.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "data" / "raw" / "halford_montreal.git"
FILE = "data/stations/montreal/bixi.geojson"
URL = "https://github.com/MaxHalford/bike-sharing-history.git"
FIELDS = ["station_id", "short_name", "name", "capacity", "num_bikes_available", "num_ebikes_available",
          "num_bikes_disabled", "num_docks_available", "num_docks_disabled", "is_installed", "is_renting",
          "is_returning", "last_reported"]


def git(*args, stdin=None):
    r = subprocess.run(["git", "-C", str(REPO), *args], input=stdin, capture_output=True)
    if r.returncode:
        raise RuntimeError(f"git {' '.join(args[:3])} failed: {r.stderr.decode()[:300]}")
    return r.stdout


def ensure_clone():
    if not REPO.exists():
        REPO.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--bare", "--filter=blob:none", "--single-branch", "--branch", "city/montreal",
                        URL, str(REPO)], check=True)
    else:
        git("fetch", "--no-tags", "-q", "origin", "city/montreal:refs/heads/city/montreal", "--force")


def commits(since, until):
    """(hash, time in UTC) of every saved copy in [since, until), oldest first."""
    out = git("log", "city/montreal", f"--since={since}T00:00:00Z", f"--until={until}T00:00:00Z",
              "--format=%H %cI").decode().split("\n")
    rows = [l.split(" ") for l in out if l]
    return sorted(((h, pd.Timestamp(t).tz_convert("UTC")) for h, t in rows), key=lambda r: r[1])


def blob_ids(hashes, chunk=500):
    ids = []
    for i in range(0, len(hashes), chunk):
        part = hashes[i:i + chunk]
        ids += git("rev-parse", *[f"{h}:{FILE}" for h in part]).decode().split()
    return ids


def fetch_blobs(ids, chunk=200):
    """git skips what we already have, so this is safe to repeat."""
    for i in range(0, len(ids), chunk):
        git("fetch", "--no-tags", "-q", "origin", *ids[i:i + chunk])
        print(f"   fetched {min(i + chunk, len(ids))} / {len(ids)}", flush=True)


def parse(raw, when):
    js = json.loads(raw)
    rows = []
    for f in js["features"]:
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"][:2]
        rows.append([p.get(k) for k in FIELDS] + [lat, lon])
    df = pd.DataFrame(rows, columns=FIELDS + ["lat", "lon"])
    df.insert(0, "snap_time", when)
    return df


def build(since, until):
    ensure_clone()
    cs = commits(since, until)
    if not cs:
        raise SystemExit("no snapshots in that range")
    print(f"{len(cs)} snapshots between {since} and {until}")
    ids = blob_ids([h for h, _ in cs])
    fetch_blobs(sorted(set(ids)))
    # Stream the saved copies in order.
    r = subprocess.run(["git", "-C", str(REPO), "cat-file", "--batch"], input="".join(i + "\n" for i in ids).encode(),
                       capture_output=True, check=True)
    buf, pos, frames = r.stdout, 0, []
    for (h, t), oid in zip(cs, ids):
        end = buf.index(b"\n", pos)
        _, kind, size = buf[pos:end].decode().split()
        size = int(size)
        frames.append(parse(buf[end + 1:end + 1 + size], t))
        pos = end + 1 + size + 1
    df = pd.concat(frames, ignore_index=True)
    for c in ("capacity", "num_bikes_available", "num_ebikes_available", "num_bikes_disabled", "num_docks_available",
              "num_docks_disabled", "is_installed", "is_renting", "is_returning", "last_reported"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["station_id"] = df.station_id.astype(str)
    out = ROOT / "data" / "interim" / f"inventory_{since}_{until}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {out.relative_to(ROOT)}: {len(df):,} rows, {df.station_id.nunique()} stations, "
          f"{out.stat().st_size / 1e6:.0f} MB")
    return df


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
