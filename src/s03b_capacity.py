"""Step 3b. Dock counts per station, for the exploratory "share of the station that moves" view.

BIXI's live station list gives today's capacity (docks). It is a snapshot, not history, so the
result is cached in data/interim/station_capacity.csv and reused on every rebuild.
A station is matched with the same rule as s02_stations.py: within 25 m, or the same normalised
name within 150 m. Unmatched stations simply get no capacity.
"""
import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import KDTree

from config import INTERIM, PROCESSED, SAME_LOCATION_M, SAME_NAME_M
from s02_stations import normalise, to_metres

URL = "https://gbfs.velobixi.com/gbfs/en/station_information.json"


def main():
    out = INTERIM / "station_capacity.csv"
    if out.exists():
        print(f"cached: {out.name}")
        return
    js = requests.get(URL, timeout=20).json()
    live = pd.DataFrame(js["data"]["stations"])[["station_id", "name", "lat", "lon", "capacity"]]
    ours = pd.read_csv(PROCESSED / "top_stations.csv")
    tree = KDTree(np.c_[to_metres(live.lat.values, live.lon.values)])
    rows = []
    live_names = live.name.map(normalise).values
    for _, s in ours.iterrows():
        xy = np.c_[to_metres(np.array([s.lat]), np.array([s.lon]))]
        d, i = tree.query(xy, k=8)
        best = None
        for dist, j in zip(d[0], i[0]):
            same_name = live_names[j] == normalise(s["name"]) and dist <= SAME_NAME_M
            if dist <= SAME_LOCATION_M or same_name:
                best = j
                break
        rows.append({"station_id": int(s.station_id),
                     "capacity": float(live.capacity.iloc[best]) if best is not None else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"matched {df.capacity.notna().sum()} of {len(df)} stations; median capacity {df.capacity.median():.0f}")


if __name__ == "__main__":
    main()
