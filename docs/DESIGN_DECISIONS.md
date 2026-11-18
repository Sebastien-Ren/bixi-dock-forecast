# Design decisions

What was decided, why, and where it lives in the code. Numbers that were measured rather than chosen are marked.

## What is forecast

| Decision | Choice | Why |
|---|---|---|
| Quantity | Net bikes: arrivals minus departures | It is the quantity that empties or fills a station. BIXI's rewards program (AMI BIXI) is built around the same problem: moving bikes from full stations to empty ones. |
| Time step | Hourly | Fine enough to see a commute peak, coarse enough that counts are not mostly zero. |
| Horizons | 1, 3 and 6 hours, as the **total** net flow over the whole window | Someone planning to move bikes cares how many a station gains or loses by then, not what happens in the single hour that far ahead. |
| Stations | The busiest 100, ranked by trips in summers 2022 and 2023 only | Enough activity to learn from. Ranking on later years would leak the test years into the choice of stations. |
| Season | May to September | The summer network. BIXI also runs a smaller winter network that this project leaves out. |

## How it is scored

| Decision | Choice | Why |
|---|---|---|
| Main metric | Average miss on big-imbalance windows: the top 10% of absolute net flow across all station-windows, with the cut-off set separately for each horizon | Net flow is skewed (measured on 2024 to 2026): the busiest 10% of windows carry about 42% of all bike movement, and they are the windows that empty or fill stations. An average over every window is dominated by the ordinary ones. |
| Baselines | Predict zero; the window just before (persistence); the same window last week; the same window averaged over the last 4 weeks; a lookup table by station, hour of week and temperature band (Level 1) | The 4-week average turned out to be the hard one to beat. Persistence is weak here: net flow tends to reverse (commute out, then back), so repeating the last window is worse than last week and, on all windows, worse than predicting zero. |
| Backtest | Train on summers 2022 and 2023, then forecast every hour of summers 2024 to 2026 once. Settings are chosen by learning on 2022 and checking on 2023 only | A forward test, with no tuning on the test years. |
| Uncertainty | An 80% range from the 10th and 90th percentile models, then corrected each day from the previous 28 days of results | Measured coverage was 76.5% to 78.5% without the correction, and 80% with it. |

## Weather

| Decision | Choice | Why |
|---|---|---|
| Test weather | The ECMWF forecast run that had been published at the forecast time (Open-Meteo Single Runs API, from March 2024) | A live system only has a forecast. Observed weather would flatter the model. |
| Run delay | A run is usable 8 hours after it starts | An **assumption**, not a measured publication time. |
| Training weather | Open-Meteo's historical weather service (ECMWF IFS), meaning what the weather turned out to be | No forecast archive exists for 2022 and 2023. Effect measured: forecast weather is within 0.04 bikes of perfect weather on big windows. |
| Where | One point, downtown Montréal, for every station | Simplicity. A known limit. |

## Stations and data quality

| Decision | Choice | Why |
|---|---|---|
| What is one station | Two records are the same station if they are within 25 m, or have the same name (ignoring case, accents, spacing and Saint/St) and are within 150 m | The trip files have names and coordinates but no station code, and both drift. In the data, stations are often put back 30 to 90 m from their old spot after winter. Distances were chosen from the measured spread of same-name and different-name pairs. |
| Out of service | A station-day with zero trips | Those hours are left out of training and scoring instead of counted as zero. |
| Dropped | Trips with no start station, exact duplicates, records outside greater Montréal (the Sherbrooke network), test stations at (0, 0) | 20,418 trips of 57.8 million. |
| Net flow caveat | It is built from trips only | Trucks moving bikes and bikes pulled for repair are not trips, so net flow is not the real change in docked bikes. |

## Models

| Level | What it is | Why it is there |
|---|---|---|
| Level 1 | Average net flow by station, hour of week and temperature band | A model that can be explained in a sentence, and the first thing a forecast must beat. |
| Level 2 | Gradient boosting on recent traffic, calendar, location and forecast weather, with separate models for the 10th, 50th and 90th percentile | The step up: learns interactions a lookup table cannot. |
| Tried and not kept | Adding neighbouring stations (within 600 m); forecasting arrivals and departures separately with a Poisson loss | Measured: neighbours help by under 0.5%, two flows are no better (worse at 3 and 6 hours). The simpler model stays. |

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Shortlist on the 2026 week | The 10 largest forecasts | Scored as precision@10 on big moves (Level 2 43%, 4-week average 41% at 3 hours). Labelled as big moves, not empty stations. |
| Which stations a truck should check | Rank by the chance of ending empty or full: bikes and free docks now plus the forecast | Measured on real 2025 counts: 21% of picks really ended empty or full at 3 hours, against 20% for stock alone and 9% for forecast size. |
| Probability of empty or full | Level 2's forecast and 80% range read as a normal distribution, then a one-variable logistic recalibration fitted on May to July 2025 | The raw reading is far too sure because trips do not see trucks; recalibrated, stated chances match what happens. |
| BIXI's own rule | Described, not scored | The real counts already include what BIXI's trucks did under that rule, so scoring it would be unfair. |
| Confirmation | September 2024, kept aside and run once with the parts fitted on May to July 2025, nothing refitted (`s05h`, plan written before the month was downloaded) | The 2025 checks had no sealed dates. A second, unseen period with a pass rule fixed in advance makes the station-count results harder to explain by luck or by tuning. Both tests held; the forecast's 1-hour gain over stock alone did not repeat. |

## Map and method page

| Decision | Choice | Why |
|---|---|---|
| Shape | Python writes one data file per replay week (`web/data/forecast.json`, `web/data/week_2025-08-04.json`); two static pages read them | The model and the pages never touch each other; any static host works. |
| Weeks | 24 to 30 August 2026 (latest trips) and 4 to 10 August 2025 (real station counts) | The 2025 week is the only one where the map can show real empty and full stations. |
| Sidebar | Fixed controls (clock, look-ahead, legend) and three tabs in order: where to look, is it right, will it empty | One story, read in order, without scrolling the whole sidebar. |
| Libraries | MapLibre GL JS 6.18 with one deck.gl layer through `@deck.gl/maplibre` | Free, runs locally, no account. |
| Basemap | OpenFreeMap | Free and online. |
| Encoding | Blue gains bikes, red loses them, size and strength show how big, fading shows uncertainty, hollow means no forecast | Uncertainty is part of the colour instead of a separate chart. |
| Method page | A technical note: fixed 3-hour examples, tables for all look-aheads, formulas in MathML, two worked examples picked by written rules | Readable top to bottom; numbers come from the same data file as the map. |

## Existing work

Several public BIXI forecasting projects exist; five were read (their READMEs, and one project's code). The closest academic work is by Silva, Aloise and Jena (Omega 2024), with BIXI's data: forecast-based ways to prioritize stations for rebalancing, evaluated by simulation, together with a description of BIXI's location-based practice. This project makes no claim to be first. What it adds is forward testing with weather as forecast, hard baselines with intervals, and checks on real station counts, including the truck effect.
