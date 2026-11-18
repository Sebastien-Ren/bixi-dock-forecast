"""Step 1. Read the five raw trip files and write one clean table.

Cleaning rules (each one is counted and printed):
  - drop trips with no start station
  - drop exact duplicate rows
  - a station with placeholder coordinates (-1, -1) or (0, 0) gets no coordinates here;
    step 2 repairs it by name if it can
All files share the same ten columns, so no per-year column mapping is needed.
"""
import duckdb

from config import INTERIM, RAW

INTERIM.mkdir(parents=True, exist_ok=True)
OUT = INTERIM / "trips.parquet"


def main():
    con = duckdb.connect()
    con.execute("set preserve_insertion_order=false")
    con.execute(f"""
        create view raw as
        select * from read_csv('{RAW}/*.csv', all_varchar=true, header=true, union_by_name=true)
    """)
    n_raw = con.execute("select count(*) from raw").fetchone()[0]

    # valid coordinate = a real number that is not the -1 / 0 placeholder
    def coord(col):
        return f"case when try_cast({col} as double) > 1 or try_cast({col} as double) < -1.5 then try_cast({col} as double) end"

    con.execute(f"""
        copy (
            select distinct
                cast(STARTTIMEMS as bigint) as start_ms,
                try_cast(ENDTIMEMS as bigint) as end_ms,
                trim(STARTSTATIONNAME) as s_name,
                {coord('STARTSTATIONLATITUDE')} as s_lat,
                {coord('STARTSTATIONLONGITUDE')} as s_lon,
                nullif(trim(ENDSTATIONNAME), '') as e_name,
                {coord('ENDSTATIONLATITUDE')} as e_lat,
                {coord('ENDSTATIONLONGITUDE')} as e_lon
            from raw
            where STARTSTATIONNAME is not null and trim(STARTSTATIONNAME) <> ''
        ) to '{OUT}' (format parquet)
    """)
    r = con.execute(f"""
        select count(*), sum((e_name is null)::int), sum((s_lat is null)::int),
               min(to_timestamp(start_ms/1000)), max(to_timestamp(start_ms/1000))
        from '{OUT}'
    """).fetchone()
    print(f"raw rows                     {n_raw:>12,}")
    print(f"kept (has start station,     {r[0]:>12,}")
    print(f"      no exact duplicates)")
    print(f"dropped                      {n_raw - r[0]:>12,}")
    print(f"kept, but no end station     {r[1]:>12,}  (counted as a departure only)")
    print(f"kept, start has no coords    {r[2]:>12,}  (step 2 repairs by name)")
    print(f"first / last start           {r[3]}  ->  {r[4]}")


if __name__ == "__main__":
    main()
