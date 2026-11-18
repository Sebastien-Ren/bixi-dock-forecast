# Evaluation

Trained on summers 2022 and 2023, never retrained. Tested on summers 2024, 2025 and 2026: 879,339 station-hours per horizon.

The main metric is the average miss, in bikes, on **big-imbalance windows**: the 10% of station-windows with the largest change.

| Look ahead | Big means | Predict zero | Last week | Last 4 weeks | Level 1 (simple average) | Level 2 (gradient boosting) | Level 2 vs best simpler method |
|---|---|---|---|---|---|---|---|
| 1 hour | 9+ bikes | 14.8 | 9.9 | **9.4** | 10.4 | 9.4 | 1% worse |
| 3 hours | 18+ bikes | 31.8 | 15.5 | 14.5 | 17.6 | **13.9** | 4% better |
| 6 hours | 29+ bikes | 52.9 | 21.6 | 20.1 | 25.9 | **18.5** | 8% better |

- **The honest headline:** against predicting zero, Level 2 removes 36%, 56% and 65% of the error. Against a good simple baseline (the same window averaged over the last 4 weeks), the lead is small: none at 1 hour, 4% at 3 hours, 8% at 6 hours.
- **The lead shrinks over time** because the model is never retrained: at 3 hours it is 7% better than the best simple method in 2024, 3% in 2025 and 1% in 2026.
- **Retraining fixes part of it.** A copy refit each year on all earlier years cuts the 2026 miss by 3% (1 hour), 4% (3 hours) and 5% (6 hours). In 2026 that makes it 1%, 6% and 11% better than the 4-week average, against -2%, 1% and 6% for the never-retrained model. The map's sidebar shows this for every year. Section 9 of [reports/RESULTS.md](../reports/RESULTS.md) has the tables.
- **Direction:** on big windows Level 2 gets filling vs emptying right 84%, 97% and 99% of the time.
- **Uncertainty band:** as trained, the 80% band held the outcome 78.5%, 77.5% and 76.5% of the time, and slipped each year. A self-correcting band (it stretches or shrinks each day using the previous 28 days of hits and misses, only outcomes already known) holds 80.0%, 80.0% and 79.9% in every year, at the cost of a band 1 to 2 bikes wider. The map uses the self-corrected band.

## As a decision: which stations to look at

A dispatcher can visit only a few stations, so the forecast is also scored as a shortlist. At each decision time (hourly for 1 hour, every 3 and 6 hours for the longer horizons, so windows never overlap), each ranking picks its top K stations and we count how many really had a big net flow, using the same cut-offs as above. Choices were fixed before looking at results ([reports/RESULTS.md](../reports/RESULTS.md) section 11 has every table). Precision at K = 10, with 95% intervals from resampling whole days:

| Look ahead | Level 2 | Last 4 weeks | Usual suspects | Random | Level 2 minus last 4 weeks |
|---|---|---|---|---|---|
| 1 hour | 30.6% | 27.5% | 27.5% | 10.4% | +3.1 points (2.8 to 3.3) |
| 3 hours | 43.4% | 41.4% | 39.5% | 11.2% | +2.0 points (1.7 to 2.3) |
| 6 hours | 50.6% | 48.8% | 45.9% | 10.8% | +1.8 points (1.4 to 2.2) |

*Usual suspects* ranks stations by their typical |net| for that hour of the week in 2022 and 2023. It shows how much of the result is just "downtown stations are big at rush hour". *Random* draws 10 of the same candidates: every ranking picks from the busiest 100 stations that are in service at that time (about 85), never from the whole network, so "better than random" means better than a random pick among busy stations.

- **Level 2 is better than every simple ranking at every K and horizon**, and the lead holds in 2024, 2025 and 2026 (3 hours, K = 10: +2.1, +2.0, +1.9 points). It is small: about 2 points.
- **Most of the value is knowing which stations are usually big.** The 4-week average is no better than the usual suspects at 1 hour, and only 2 to 3 points better at 3 and 6 hours.
- **Time of day.** Overnight there are almost no big windows, so a shortlist has nothing to find; from 06:00 on, 40% to 74% of the top 10 are big. The model's lead over the 4-week average is largest in the evening (+5.1 points at 21:00) and absent at 06:00.
- **Direction** is almost never the hard part: 98% to 100% of correct picks have the right sign.
- **Exploratory, did not help:** ranking by how far the range's nearer edge is from zero is slightly worse than ranking by forecast size.
- **Exploratory, docks:** counting a window as big when it moves at least half the station's docks (82 of 100 stations matched to today's dock counts) gives the same picture: Level 2 +2.6 points over the 4-week average at 3 hours.
- **Caveats.** A big net flow is not an empty or full station, since net flow ignores trucks and the real stock. K stands in for real limits such as truck capacity. The aggregate backtest had been read once before these choices were fixed.

All tables, including results by year, hour and station, are in [reports/RESULTS.md](../reports/RESULTS.md).

## Will the station be empty or full?

Net flow is not stock, so the forecast was also scored on real station counts: past snapshots of BIXI's station feed, saved about every 15 minutes by a public archive ([Max Halford's bike-sharing-history](https://github.com/MaxHalford/bike-sharing-history)), for summer 2025. 91 of the 100 stations match. This is a retrospective test on past data, not a prospective one; Level 2 never trained on 2025. Questions and baselines were fixed before any result was seen ([reports/RESULTS.md](../reports/RESULTS.md) section 12 has every table).

The baseline to beat knows the real count but has no forecast: how often a station with that many bikes (or free docks) ended up empty (or full) after that many hours, at that hour of day, learned on May to July 2025. The forecast's probability comes from Level 2's forecast and its 80% range, recalibrated on May to July. Tested on August and September 2025; the Brier score, lower is better; skill is the share of the baseline's score removed, with 95% intervals from resampling whole days. Stations not already empty or full:

| Look ahead | Empty: happened | Empty: skill vs baseline | Full: happened | Full: skill vs baseline |
|---|---|---|---|---|
| 1 hour | 3.6% | -0.3% (-0.9 to +0.3), tie | 2.2% | +0.7% (+0.1 to +1.3) |
| 3 hours | 4.7% | +1.1% (+0.3 to +1.7) | 2.9% | +1.7% (+1.1 to +2.2) |
| 6 hours | 5.8% | +2.4% (+1.7 to +3.0) | 3.5% | +2.3% (+1.7 to +2.8) |

- **The forecast adds a little, and more with a longer look-ahead.** The gain is largest in the morning rush (06:00 to 10:00: about +6% for both), absent overnight, and slightly negative at midday for empty stations (-2.2%).
- **Read straight from the model's range, the probability is far too sure.** When it says 90% or more, the station is empty about 20% of the time, and it scores far worse than the baseline. Recalibrated, it is honest but rarely says more than 30%.
- **Why: the truck effect, measured.** Over 3 hours, the real change in bikes at a station (available plus disabled) and the trip-based net flow correlate 0.47 (0.49 at 1 hour, 0.47 at 6). They differ by 2.5, 5.4 and 10.2 bikes on average at 1, 3 and 6 hours, against average trip net flows of 3.3, 7.5 and 13.9, and by 5 or more bikes in a quarter of 3-hour windows. The gap is largest in the morning and evening rush. Timing was checked: the match peaks at zero offset and vanishes an hour either way. So a station the trips say will lose 30 bikes often does not empty: a truck refills it.
- **"Same as now" is a poor baseline here**: stations that are empty usually do not stay empty, so the forecast removes 34% to 51% of its error. The starting-count baseline is the honest one.
- **Confirmed on a sealed month.** The same test, run once on September 2024 with the parts fitted on May to July 2025 and nothing refitted, under a plan written before that month was downloaded: skill at 3 hours +3.6% (+2.6 to +4.6) for empty and +1.7% (+0.7 to +2.7) for full; at 6 hours +2.9% and +3.0%. At 1 hour the two swap (empty better, full a tie), so 1 hour is the least stable look-ahead. The raw reading is still far too sure (says 94%, empty 26% of the time), and the truck-effect correlation is 0.51 to 0.53. Every table: [reports/RESULTS.md](../reports/RESULTS.md) section 15.
- **Caveats.** About 90 days of real counts in two periods (61 test days in 2025, 30 in September 2024). July 2025 is half missing from the archive. Snapshots are matched to the hour within 10 minutes. 9 of the 100 stations do not match.

## Which stations a truck should check

On the 2025 counts, at each decision time, three rankings pick 10 stations; a pick counts if the station really is empty or full at the end of the window (test August and September 2025, 95% day-block intervals):

| Look ahead | Forecast size | Stock only | Stock + forecast | Random | Stock + forecast minus stock only |
|---|---|---|---|---|---|
| 1 hour | 7.4% | 32.2% | 33.2% | 9.7% | +1.0 points (0.3 to 1.7) |
| 3 hours | 9.3% | 20.2% | 20.7% | 9.8% | +0.5 points (-0.9 to 1.9), tie |
| 6 hours | 12.1% | 14.2% | 14.7% | 10.0% | +0.5 points (-1.4 to 2.4), tie |

- The map's shortlist (biggest forecast) is good at finding big moves (section above) but no better than random at finding stations that will be empty or full: a station with many docks can lose 30 bikes and be fine.
- Stock now is what matters; the forecast adds a little on top, clearly only at 1 hour in 2025.
- **Confirmed on September 2024** (sealed, run once, nothing refitted): forecast size 7.6% / 9.1% / 13.3%, stock only 31.5% / 20.0% / 15.6%, stock + forecast 32.2% / 20.7% / 16.3%, random about 10%. The order holds at every look-ahead. The 1-hour gain of stock + forecast over stock only did not repeat: +0.7 points (-0.1 to 1.6), a tie, so the forecast's extra over the stock count is best called within the noise.
- BIXI's rule (mostly location-based, per Silva, Aloise and Jena, Omega 2024) is not scored: the counts include what its trucks did under it. A fair comparison needs BIXI's truck logs.

## Against the real change in bikes (exploratory)

Everything above scores the forecast against trip net flow, which is what it was trained to predict. With the 2025 station counts the real change in bikes at a station (available plus disabled, end minus start) is also known. **Exploratory:** not written down in advance, one period (August and September 2025), the sealed month not used. 95% day-block intervals ([reports/RESULTS.md](../reports/RESULTS.md) section 16):

| Look ahead | Correlation with trips | Correlation with real change | Skill vs no change, on trips | Skill vs no change, on real change | Same, on the biggest real changes | Direction right on those |
|---|---|---|---|---|---|---|
| 1 hour | 0.62 | 0.17 | +13.3% | -2.1% (-2.6 to -1.6) | +5.9% (+5.0 to +6.8) | 56% |
| 3 hours | 0.82 | 0.32 | +32.5% | -11.0% (-12.3 to -9.8) | +16.6% (+14.9 to +18.3) | 68% |
| 6 hours | 0.88 | 0.42 | +46.3% | -27.9% (-29.9 to -26.0) | +18.6% (+16.1 to +20.8) | 80% |

- **The forecast follows trips closely and the real change in bikes loosely.** Over all windows it is worse than predicting no change against the real change: trucks keep stocks steadier than trips alone would.
- **On the biggest real changes it still helps**, more at longer look-aheads. At 1 hour its direction is right only 56% of the time there.
- **The 80% range, calibrated to trips, holds 75%, 70% and 64% of real changes** at 1, 3 and 6 hours. This is the truck effect measured on the forecast itself.
- **It is a forecast of demand imbalance, not of stock**, and cannot be retrained on the real change: counts exist only from 2025, not for the training years.

## What the model needs

Each row refits the forecast with one change (miss on big windows, 2024 to 2026, against Level 2):

| Change | 1 hour | 3 hours | 6 hours |
|---|---|---|---|
| Without seasonal memory | +11.3% | +28.5% | +34.6% |
| Without calendar and place | +4.1% | +5.6% | +5.1% |
| Without recent state | +3.2% | +3.0% | +2.4% |
| Without weather | +0.6% | +2.2% | +4.0% |
| With neighbours (600 m) | -0.4% | -0.3% | tie |
| Arrivals and departures separately | tie | +3.7% | +6.6% |

Seasonal memory carries the model. Neighbours add almost nothing and two flows are worse, so the simpler model stays.

## Where it fails

- **Events.** On the most extreme 1 in 1,000 windows (137+ bikes in 3 hours), the average miss is 97 bikes and only 16% land inside the band. The model has no event calendar.
- **The 1-hour horizon.** No better than a 4-week average. One hour of net flow is mostly noise.
- **Evenings and nights.** From 18:00 to 04:00 it removes 29% to 46% of predict-zero's error, against 67% to 74% in the morning rush. Commutes repeat; leisure trips do not.
- **Balanced residential stations.** At Drolet / Marie-Anne it removes only 15%.
- **Level 1 loses to "last week" and to the 4-week average** at every horizon. A fixed average from 2022 and 2023 goes stale; recent weeks do not.
- **It ages.** Never retrained, the model loses ground each year (see above), and its raw band slips from 77.2% to 75.4% coverage at 6 hours.

## Limits to keep in mind

- Net flow here is built from trips. Trucks moving bikes and bikes pulled for repair are not trips, so this is not the real change in docked bikes. Measured on summer 2025, the two correlate only about 0.47 (see above).
- Only the 100 busiest stations are forecast and scored. Every comparison, random picks included, is within them. In summer 2025 they carried 25% of trips, 20% of all net flow and 37% of the big 3-hour moves (18+ bikes), but only about 9% of empty-station time and 7% of full-station time (August and September 2025 counts): busy and quiet stations are empty about equally often, and there are ten times more quiet ones. At the other ~800 stations the median 3-hour window moves 2 bikes and 62% move 2 or fewer, so a forecast adds little there; a network-wide dispatch list would lean on the stock count, which is what the truck-ranking study found.
- The network changes. Of the 100 stations ranked on 2022 and 2023, 94 were in service in summer 2024, 91 in 2025 and 85 in 2026. Stations that closed or moved more than a block get no forecast.
- The map shows the never-retrained model. The retrained copy is scored on the method page (results section) but does not draw the circles.
- The trip-based backtest (2024 to 2026) has no sealed set of dates: it was read once and nothing was tuned afterwards. The checks on real station counts were confirmed on a sealed month, September 2024, run once under a plan written before its data was downloaded.
