# How weather plays into it

**What goes in.** Three numbers per forecast window: average temperature, total rain, average wind. They come from one point, downtown Montreal, from the ECMWF weather model through Open-Meteo, and every station gets the same values.

**Training and testing use different weather, on purpose.**
- Training (2022, 2023): the weather that actually happened. No archived forecasts exist that far back.
- Testing (2024 onward): the forecast run that had been published at the time. ECMWF starts main runs at 00:00 and 12:00 UTC. I assume a run becomes usable 8 hours after it starts (an assumption, not a measured publication time), so at forecast time the newest usable run is 8 to 20 hours old. In practice the median is 14 hours and the oldest 31. About 0.13% of test hours had no run and were left blank, which gradient boosting handles.
- A test checks that a run is never used before it is published.

**How good the weather forecast was.** The average miss on window temperature was about 0.5 °C. For rain over 3-hour windows, the forecast and the observed totals correlate at about 0.8.

**What weather is worth.** I refit Level 2 with every weather input removed and compared it with the main model (forecast weather) and with a version given the real weather (the best any weather forecast could do). Average miss, in bikes:

| Look ahead | Windows | No weather | Forecast weather (main model) | Real weather (cheating) | Forecast vs no weather |
|---|---|---|---|---|---|
| 1 hour | Big-imbalance | 9.461 | 9.408 | 9.406 | 0.6% better |
| 3 hours | Big-imbalance | 14.210 | 13.907 | 13.889 | 2.1% better |
| 6 hours | Big-imbalance | 19.246 | 18.504 | 18.468 | 3.9% better |
| 1 hour | Rainy (1 mm+) | 2.451 | 2.403 | 2.392 | 1.9% better |
| 3 hours | Rainy (1 mm+) | 4.887 | 4.617 | 4.593 | 5.5% better |
| 6 hours | Rainy (1 mm+) | 7.562 | 7.117 | 7.099 | 5.9% better |

What this says:
- **Weather helps, modestly, and more at longer horizons and in rain.** It is worth up to about 4% on big windows and 6% on rainy windows, and almost nothing at 1 hour.
- **Forecast weather is almost as good as real weather.** The gap is 0.002 to 0.04 bikes, so the "honest weather" test costs very little. That is partly because the weather model is accurate over the next few hours, and partly because weather matters little to begin with.
- **Habit matters far more than weather.** Shuffling one input at a time, the 4-week average of the same window adds 2.9 bikes of error; rain adds 0.08, temperature 0.01, wind 0.002. Rain is the only weather input that counts.
- **The model helps less in the rain.** At 3 hours it removes 33% of predict-zero's error over all windows but 21% over rainy ones. I did not test why. Rainy windows are quieter, so there is less to forecast, but that is a guess.

What this does not cover: one weather point for the whole city; no humidity, snow or events; summer only; and a model trained on observed weather rather than on forecasts (a model trained on forecasts could behave differently). The full tables are in section 10 of [reports/RESULTS.md](../reports/RESULTS.md).
