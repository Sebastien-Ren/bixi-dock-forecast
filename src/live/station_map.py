"""Step 2b. Match BIXI's live station ids to this project's stations.

The trip files have no station id, so this project's stations come from names and coordinates
(src/s02_stations.py). The live feed has ids. Same rule as before: two records are one station if
they are within 25 m, or have the same normalised name within 150 m.
Run:  .venv/bin/python src/live/station_map.py
Writes live/station_map.csv (every one of our stations) and web/data/live_map.json (top 100, for the map).
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED, SAME_LOCATION_M, SAME_NAME_M, WEB_DATA        # noqa: E402
from s02_stations import normalise, to_metres                                # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "live"


def match_stations(ours, live, k=8):
    """For each of our stations, the row index in `live` of its match, or None.
    Both frames need name, lat, lon. Nearest first, so a close candidate wins over a farther one."""
    tree = KDTree(to_metres(live.lat.values, live.lon.values))
    live_names = live.name.map(normalise).values
    d, idx = tree.query(to_metres(ours.lat.values, ours.lon.values), k=min(k, len(live)))
    out = []
    for (_, s), dists, cands in zip(ours.iterrows(), d, idx):
        mine, hit = normalise(s["name"]), None
        for dist, j in zip(dists, cands):
            if dist <= SAME_LOCATION_M or (live_names[j] == mine and dist <= SAME_NAME_M):
                hit = int(j)
                break
        out.append(hit)
    return out


def latest_info():
    files = sorted((LIVE / "info").glob("*.json.gz"))
    if not files:
        raise SystemExit("No station list saved yet: run src/live/log_status.py first.")
    js = json.loads(gzip.decompress(files[-1].read_bytes()))
    return files[-1].name[:10], pd.DataFrame(js["data"]["stations"])[["station_id", "name", "lat", "lon", "capacity"]]


def main():
    day, live = latest_info()
    live["station_id"] = live.station_id.astype(str)
    stations = pd.read_parquet(PROCESSED.parent / "interim" / "stations.parquet")
    top = pd.read_csv(PROCESSED / "top_stations.csv")
    m = match_stations(stations, live)
    full = stations[["station_id", "name"]].rename(columns={"station_id": "our_id", "name": "our_name"}).assign(
        live_id=[None if j is None else live.station_id.iloc[j] for j in m],
        live_name=[None if j is None else live.name.iloc[j] for j in m],
        capacity=[np.nan if j is None else live.capacity.iloc[j] for j in m])
    (LIVE).mkdir(exist_ok=True)
    full.to_csv(LIVE / "station_map.csv", index=False)

    t = top.merge(full, left_on="station_id", right_on="our_id", how="left")
    ok = t[t.live_id.notna()]
    miss = t[t.live_id.isna()]
    out = {"station_list_date": day,
           "to_ours": {r.live_id: int(r.station_id) for r in ok.itertuples()},
           "capacity": {str(int(r.station_id)): int(r.capacity) for r in ok.itertuples()},
           "unmatched": [{"id": int(r.station_id), "name": r.name} for r in miss.itertuples()]}
    WEB_DATA.mkdir(parents=True, exist_ok=True)
    (WEB_DATA / "live_map.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"top 100: matched {len(ok)}, unmatched {len(miss)}; all {len(full)} of our stations: matched {full.live_id.notna().sum()}")
    for r in miss.itertuples():
        print("  unmatched:", r.station_id, r.name)


if __name__ == "__main__":
    main()
