# Data

The trip files are about 10 GB, so they are not in git. Put them in `data/raw/` and the pipeline does the rest.

## What to download

1. **Trips.** BIXI Montréal's open-data page, "Historique des déplacements": <https://bixi.com/en/open-data/>. This project used trips from 2022 through August 2026: the newest download is a ZIP covering January to August 2026, and earlier years are on the same page or the Données Québec record for the dataset. Unzip the CSVs into `data/raw/`. File names do not matter: the pipeline reads every `*.csv` in the folder.

   Every file has the same ten columns: `STARTSTATIONNAME`, `STARTSTATIONARRONDISSEMENT`, `STARTSTATIONLATITUDE`, `STARTSTATIONLONGITUDE`, `ENDSTATIONNAME`, `ENDSTATIONARRONDISSEMENT`, `ENDSTATIONLATITUDE`, `ENDSTATIONLONGITUDE`, `STARTTIMEMS`, `ENDTIMEMS` (times are epoch milliseconds). There is no station code, which is why `src/s02_stations.py` decides what one station is from names and coordinates.

2. **Station list (optional).** Today's dock counts, used only for context on the map and in the docs:

   ```sh
   curl -o data/raw/station_information_$(date +%F).json https://gbfs.velobixi.com/gbfs/en/station_information.json
   ```

   Without it the pipeline still runs; the map just omits dock counts.

3. **Weather.** Downloaded automatically by `src/s03_weather.py` from [Open-Meteo](https://open-meteo.com) (no key needed) and cached in `data/interim/weather_cache/`. The first run makes about 900 requests and needs internet.

4. **Past station counts (optional).** Snapshots of BIXI's station feed for May to September 2025, from [Max Halford's bike-sharing-history](https://github.com/MaxHalford/bike-sharing-history). `src/archive/build_inventory.py` keeps a partial clone in `data/raw/halford_montreal.git` (about 260 MB) and writes one table per month to `data/interim/` (about 110 MB in total). Needed only for the empty/full and truck-ranking checks and the map's 2025 week; see the main README for the command.

## What the pipeline writes

| Folder | Contents | In git? |
|---|---|---|
| `data/raw/` | The files above | No |
| `data/interim/` | Clean trips, stations, hourly counts, weather cache (about 1.3 GB) | No |
| `data/processed/` | The feature table and every forecast (about 240 MB) | No, except `top_stations.csv` |

`data/processed/top_stations.csv` lists the 100 busiest stations, ranked on summers 2022 and 2023 only, with every name each station has used.

## Terms

BIXI trip data is published under an attribution licence. Weather data is CC BY 4.0 (Open-Meteo). Keep the attribution if you reuse results. See [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
