"""Step 2. Decide what "one station" means, then count bikes per station per hour.

The trip files have no station code, only a name and coordinates, and both drift:
names get typos and renames, coordinates jitter by a few metres.
Rule: two points are the same station if
  (a) they are closer than SAME_LOCATION_M metres, or
  (b) they have the same name (ignoring case, accents, spacing and punctuation)
      and are closer than SAME_NAME_M metres. BIXI removes stations for winter and
      often puts them back 30 to 90 m away under the same name; this keeps them as one.
A station that moved further than that, or changed both name and place, is a new station.

Outputs
  data/interim/stations.parquet   one row per station (id, name, lat, lon, every name it had)
  data/interim/hourly.parquet     departures and arrivals per station per hour (all stations)
  data/processed/top_stations.csv the busiest N, ranked on training-year summers only
"""
import re
import unicodedata

import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import KDTree

from config import (BBOX, INTERIM, N_STATIONS, PROCESSED, SAME_LOCATION_M, SAME_NAME_M,
                    SUMMER_MONTHS, TRAIN_YEARS, TZ)

TRIPS = INTERIM / "trips.parquet"


def to_metres(lat, lon):
    """Flat-map approximation, fine at city scale: degrees -> metres from a reference point."""
    x = (lon + 73.57) * 111_320 * np.cos(np.radians(45.51))
    y = (lat - 45.51) * 110_540
    return np.c_[x, y]


def normalise(name):
    """'Métro  Mont-Royal (Rivard/du Mont-Royal)' -> 'metromontroyalrivarddumontroyal'.
    Saint/Sainte are shortened to St/Ste, because the 2026 file spells them out."""
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    plain = re.sub(r"\bsainte\b", "ste", plain)
    plain = re.sub(r"\bsaint\b", "st", plain)
    return re.sub(r"[^a-z0-9]", "", plain)


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"set timezone='{TZ}'")

    # Every distinct (name, coordinates) seen as a trip start or end, with trip counts.
    pts = con.execute(f"""
        select name, lat, lon, sum(n) as n, max(last_ms) as last_ms from (
            select s_name as name, round(s_lat, 6) as lat, round(s_lon, 6) as lon,
                   count(*) as n, max(start_ms) as last_ms
            from '{TRIPS}' where s_lat is not null group by 1, 2, 3
            union all
            select e_name, round(e_lat, 6), round(e_lon, 6), count(*), max(start_ms)
            from '{TRIPS}' where e_lat is not null and e_name is not null group by 1, 2, 3
        ) group by 1, 2, 3
    """).df()
    inside = (pts.lat.between(BBOX["lat_min"], BBOX["lat_max"])
              & pts.lon.between(BBOX["lon_min"], BBOX["lon_max"]))
    print(f"points (name + coordinates): {len(pts):,}; outside greater Montreal: {(~inside).sum()} "
          f"({int(pts.n[~inside].sum()):,} trip ends)")
    # Sorted so the station numbers come out the same on every run.
    pts = pts[inside].sort_values(["lat", "lon", "name"]).reset_index(drop=True)

    # Link pairs of points that pass rule (a) or rule (b), then take connected groups.
    xy = to_metres(pts.lat.values, pts.lon.values)
    norm = pts.name.map(normalise).values
    idx, dist = KDTree(xy).query_radius(xy, r=SAME_NAME_M, return_distance=True)
    i = np.repeat(np.arange(len(pts)), [len(p) for p in idx])
    j, d = np.concatenate(idx), np.concatenate(dist)
    linked = (d <= SAME_LOCATION_M) | (norm[i] == norm[j])
    graph = coo_matrix((np.ones(linked.sum()), (i[linked], j[linked])), shape=(len(pts), len(pts)))
    _, pts["station_id"] = connected_components(graph, directed=False)

    # One row per station: trip-weighted centre, and the name it used most recently.
    def summarise(g):
        latest = g[g.last_ms >= g.last_ms.max() - 30 * 86_400_000]   # names seen in its last 30 days
        names = g.groupby("name").n.sum().sort_values(ascending=False)
        return pd.Series({
            "name": latest.groupby("name").n.sum().idxmax(),
            "lat": np.average(g.lat, weights=g.n),
            "lon": np.average(g.lon, weights=g.n),
            "n_names": len(names),
            "all_names": " | ".join(names.index),
            "spread_m": float(np.ptp(to_metres(g.lat.values, g.lon.values), axis=0).max()),
        })
    stations = pts.groupby("station_id").apply(summarise, include_groups=False).reset_index()
    print(f"stations after merging: {len(stations):,} "
          f"(from {pts.name.nunique():,} distinct names)")
    print(f"stations that carried more than one name: {(stations.n_names > 1).sum()}")

    # Lookup tables: coordinates -> station, and (for trips with no coordinates) name -> station.
    by_coord = pts[["lat", "lon", "station_id"]].drop_duplicates(["lat", "lon"])
    by_name = (pts.sort_values("n", ascending=False).drop_duplicates("name")[["name", "station_id"]])
    con.register("by_coord", by_coord)
    con.register("by_name", by_name)

    # Departures are counted in the hour the trip starts, arrivals in the hour it ends.
    con.execute(f"""
        copy (
            with dep as (
                select coalesce(c.station_id, n.station_id) as station_id,
                       start_ms // 3600000 as hour_idx, count(*) as dep
                from '{TRIPS}' t
                left join by_coord c on round(t.s_lat, 6) = c.lat and round(t.s_lon, 6) = c.lon
                left join by_name n on t.s_lat is null and t.s_name = n.name
                group by 1, 2
            ), arr as (
                select coalesce(c.station_id, n.station_id) as station_id,
                       end_ms // 3600000 as hour_idx, count(*) as arr
                from '{TRIPS}' t
                left join by_coord c on round(t.e_lat, 6) = c.lat and round(t.e_lon, 6) = c.lon
                left join by_name n on t.e_lat is null and t.e_name = n.name
                where t.end_ms is not null and t.e_name is not null
                group by 1, 2
            )
            select station_id, hour_idx,
                   coalesce(dep, 0)::int as dep, coalesce(arr, 0)::int as arr
            from dep full outer join arr using (station_id, hour_idx)
            where station_id is not null
        ) to '{INTERIM}/hourly.parquet' (format parquet)
    """)
    tot = con.execute(f"select sum(dep), sum(arr) from '{INTERIM}/hourly.parquet'").fetchone()
    print(f"departures placed on a station: {int(tot[0]):,}; arrivals placed: {int(tot[1]):,}")

    # Rank on training-year summers only, so nothing from the backtest years leaks into the choice.
    rank = con.execute(f"""
        select station_id, sum(dep + arr) as summer_trips
        from '{INTERIM}/hourly.parquet'
        where year(to_timestamp(hour_idx * 3600)) in ({",".join(map(str, TRAIN_YEARS))})
          and month(to_timestamp(hour_idx * 3600)) between {SUMMER_MONTHS[0]} and {SUMMER_MONTHS[1]}
        group by 1 order by 2 desc limit {N_STATIONS}
    """).df()
    rank["rank"] = np.arange(1, len(rank) + 1)
    top = rank.merge(stations, on="station_id")
    stations.to_parquet(INTERIM / "stations.parquet", index=False)
    top.to_csv(PROCESSED / "top_stations.csv", index=False)
    print(f"\nbusiest {N_STATIONS} stations, summers {TRAIN_YEARS}: "
          f"{int(top.summer_trips.max()):,} down to {int(top.summer_trips.min()):,} trips")
    print(top[["rank", "name", "summer_trips", "n_names", "spread_m"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
