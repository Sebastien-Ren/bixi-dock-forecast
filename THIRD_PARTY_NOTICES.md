# Third-party notices

This project's own code is under the [MIT License](LICENSE). It uses the data and software below, each under its own terms.

## Data

| Source | Used for | Terms |
|---|---|---|
| BIXI Montréal, "Historique des déplacements" (<https://bixi.com/en/open-data/>) | Trip history, 2022 to August 2026 | Open-data attribution licence. Attribute BIXI Montréal. The raw files are not redistributed here. |
| [Open-Meteo](https://open-meteo.com), ECMWF IFS | Actual weather and archived forecast runs | CC BY 4.0. The free API is for non-commercial use. |
| BIXI live station list (GBFS) | Dock counts shown as context; the hidden live view | Published by BIXI as open data. |
| Past snapshots of BIXI's station feed, from [Max Halford's bike-sharing-history](https://github.com/MaxHalford/bike-sharing-history) | Real station counts, May to September 2025 and September 2024 (sealed confirmation), for the empty/full and truck-ranking checks | BIXI open data (attribute BIXI Montréal); archive by Max Halford. Raw snapshots are not redistributed; the map's 2025 week file contains derived counts for 85 stations. |
| [OpenFreeMap](https://openfreemap.org), OpenMapTiles, OpenStreetMap | Basemap on the map page | Loaded online at view time. Attribution is shown on the map: OpenFreeMap, © OpenMapTiles, data from OpenStreetMap (ODbL). |
| Instrument Sans (Google Fonts) | Font on the map and method pages | SIL Open Font License. Loaded online at view time. |

## JavaScript committed in `web/vendor/`

Built once by `web/build-vendor.mjs` from the packages pinned in `web/package-lock.json`.

| Package | Licence |
|---|---|
| maplibre-gl 6.18.0 (and its dependencies, such as pbf, earcut, kdbush) | BSD-3-Clause, BSD-2-Clause and ISC |
| deck.gl 9.4.1 (`@deck.gl/core`, `@deck.gl/layers`, `@deck.gl/maplibre`) and the vis.gl modules it uses (luma.gl, math.gl, loaders.gl, probe.gl) | MIT |
| apache-arrow, flatbuffers (pulled in by loaders.gl) | Apache-2.0 |
| tslib | 0BSD |

All of these licences require keeping the copyright notice. Each package's full licence text is in its repository and in the package itself after `npm install`. The MapLibre files keep their licence header.

## Python

The Python packages in `requirements.txt` (DuckDB, pandas, NumPy, SciPy, scikit-learn, PyArrow, holidays, requests, pytz, pytest) are installed from PyPI, not redistributed, and all use permissive licences.
