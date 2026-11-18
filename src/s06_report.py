"""Step 6. Score the backtest and write the numbers down.

Reads data/processed/predictions.parquet (every forecast next to what really happened) and writes
  reports/metrics.json   every number, for the map page and for checking
  reports/RESULTS.md     the same numbers as readable tables
Nothing here changes a forecast. It only measures.
"""
import json

import numpy as np
import pandas as pd

import scoring
from config import BAND_TARGET, BIG_IMBALANCE_TOP_SHARE, HORIZONS, PROCESSED, QUANTILES, REPORTS

MODELS = {"zero": "Predict zero", "persist": "Same as the window just before", "last_week": "Same window last week", "avg4w": "Average of the last 4 weeks",
          "l1": "Level 1: average by hour of week and temperature", "l2": "Level 2: gradient boosting"}


def score_block(d, cutoff):
    """All headline numbers for one slice of forecasts (one horizon, or one horizon + year, ...)."""
    y = d.y.values
    big = scoring.big_imbalance_mask(y, cutoff)
    out = {"n": int(len(d)), "n_big": int(big.sum())}
    for m in list(MODELS) + ["l2_actual_weather"]:
        out[f"mae_{m}"] = scoring.mae(y, d[m].values)
        out[f"big_mae_{m}"] = scoring.mae(y[big], d[m].values[big]) if big.any() else None
    for m in ("l1", "l2_raw", "l2"):          # l2_raw = band straight from the model, l2 = after self-correction
        lo, hi = d[f"{m}_lo"].values, d[f"{m}_hi"].values
        out[f"coverage_{m}"] = scoring.coverage(y, lo, hi)
        out[f"big_coverage_{m}"] = scoring.coverage(y[big], lo[big], hi[big]) if big.any() else None
        out[f"band_width_{m}"] = float(np.mean(hi - lo))
        out[f"pinball_lo_{m}"] = scoring.pinball(y, lo, QUANTILES[0])
        out[f"pinball_hi_{m}"] = scoring.pinball(y, hi, QUANTILES[2])
    if big.any():
        out["big_direction_l2"] = scoring.direction_hit_rate(y[big], d.l2.values[big])
        out["big_direction_l1"] = scoring.direction_hit_rate(y[big], d.l1.values[big])
        out["big_direction_last_week"] = scoring.direction_hit_rate(y[big], d.last_week.values[big])
        out["big_skill_l2_vs_zero"] = scoring.skill(out["big_mae_l2"], out["big_mae_zero"])
        best_simple = min(out["big_mae_zero"], out["big_mae_persist"], out["big_mae_last_week"], out["big_mae_avg4w"], out["big_mae_l1"])
        out["big_skill_l2_vs_best_simple"] = scoring.skill(out["big_mae_l2"], best_simple)
    return out


def table(rows, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


NAMES = {"l2": "Level 2", "avg4w": "Last 4 weeks", "last_week": "Last week", "usual": "Usual suspects",
         "l2_edge": "Level 2, by sureness (exploratory)", "random": "Random"}


def pct(x, digits=1):
    return f"{x['v']:.{digits}%} ({x['lo']:.{digits}%} to {x['hi']:.{digits}%})"


def pts(x):
    """A paired difference in percentage points, with its interval and what it means."""
    return f"{x['v'] * 100:+.1f} pts ({x['lo'] * 100:+.1f} to {x['hi'] * 100:+.1f}), {x['verdict']}"


def decision_section(dc):
    """Section 11: the forecast scored as 'which K stations should I look at?' (src/s05d_decision_check.py)."""
    hk, hh = str(dc["meta"]["headline"]["k"]), str(dc["meta"]["headline"]["horizon"])
    big = {h: dc["main"][h]["cutoff"] for h in dc["main"]}
    md = ["## 11. Turning the forecast into a decision", "",
          "A dispatcher can look at only a few stations at a time. At each decision time (every hour for the 1 h "
          "horizon, every 3 h and every 6 h for the longer ones, so windows never overlap) every ranking picks its "
          "top K stations, and we count how many of them really had a big net flow (the same cut-offs as section 1: "
          + ", ".join(f"±{big[h]:.0f}+ bikes for {h} h" for h in dc["main"]) + "). "
          "*Big net flow is not the same as an empty or full station*: net flow ignores trucks and the real stock "
          "(section 12 scores that on real station counts). K stands in for real limits such as truck capacity. "
          "Choices and the rule for \"better\" were fixed before the numbers were looked at "
          "(the aggregate backtest had been seen once already). All numbers carry 95% day-block bootstrap intervals "
          f"({dc['meta']['bootstrap_reps']:,} resamples of whole days). *Better* only when the interval of the paired "
          "difference is wholly above zero; otherwise it is a tie.", "",
          "- **precision@K**: the share of the K picks that were big.",
          "- **capture@K**: picks that were big, divided by the smaller of K and the number of big windows at that time "
          "(1.0 = a perfect ranking). Times with no big window are left out.",
          "- **Usual suspects**: each station's typical |net| for that hour of the week, from 2022 and 2023 only. "
          "It shows how much of the skill is just 'downtown stations are always big at rush hour'.",
          "- **Random**: the exact expected result of picking at random from the same candidates. Every ranking picks "
          "from the busiest 100 stations in service at that time (median " + f"{dc['main']['3']['median_candidates']:.0f}" + "), not from the "
          "whole network.", ""]

    md += [f"### Headline: top {hk} picks", ""]
    rows = []
    for h in dc["main"]:
        m = dc["main"][h]
        r = m["rankings"]
        rows.append([f"{h} h", f"{m['median_positives']:.0f} of {m['median_candidates']:.0f}"]
                    + [pct(r[n][hk]["precision"], 1) for n in ("l2", "avg4w", "last_week", "usual", "random")]
                    + [pts(m["paired"]["l2_vs_avg4w"][hk]["precision"])])
    md += [table(rows, ["Horizon", "Typical big windows per time (of candidates)", "Level 2", "Last 4 weeks",
                        "Last week", "Usual suspects", "Random", "Level 2 minus last 4 weeks"]), "",
           f"Level 2 against each simple ranking at {hh} h and K={hk}: "
           + "; ".join(f"{NAMES[b]} {pts(dc['main'][hh]['paired'][f'l2_vs_{b}'][hk]['precision'])}"
                       for b in ("avg4w", "last_week", "usual", "random")) + ".", ""]

    md += [f"### Different K ({hh} h horizon)", ""]
    rows = []
    for k in dc["meta"]["ks"]:
        k = str(k)
        r = dc["main"][hh]["rankings"]
        rows.append([k] + [pct(r[n][k]["precision"], 1) for n in ("l2", "avg4w", "usual", "random")]
                    + [f"{r['l2'][k]['capture']['v']:.0%} vs {r['avg4w'][k]['capture']['v']:.0%}",
                       f"{r['l2'][k]['lift']['v']:.1f}x",
                       pts(dc["main"][hh]["paired"]["l2_vs_avg4w"][k]["precision"])])
    md += [table(rows, ["K", "Level 2 precision", "Last 4 weeks", "Usual suspects", "Random",
                        "Capture, Level 2 vs last 4 weeks", "Lift of Level 2 over random", "Level 2 minus last 4 weeks (precision)"]), "",
           "Lift = precision divided by the share of all candidates that were big (what random picking hits).", ""]

    md += [f"### Horizon by horizon: Level 2 minus each baseline at K={hk}, precision", ""]
    rows = []
    for h in dc["main"]:
        pr = dc["main"][h]["paired"]
        rows.append([f"{h} h"] + [pts(pr[f"l2_vs_{b}"][hk]["precision"]) for b in ("avg4w", "last_week", "usual")]
                    + [pts(pr["avg4w_vs_usual"][hk]["precision"])])
    md += [table(rows, ["Horizon", "vs last 4 weeks", "vs last week", "vs usual suspects", "(Last 4 weeks vs usual suspects)"]), ""]

    md += [f"### Year by year ({hh} h, K={hk}, Level 2 precision and lead over last 4 weeks)", ""]
    rows = [[y, pct(v["rankings"]["l2"][hk]["precision"], 1), pts(v["paired"]["l2_vs_avg4w"][hk]["precision"])]
            for y, v in dc["by_year"][hh].items()]
    md += [table(rows, ["Year", "Level 2 precision", "Level 2 minus last 4 weeks"]), ""]

    md += [f"### Time of day ({hh} h, K={hk}, hour the window starts)", ""]
    rows = [[f"{int(hr):02d}:00", f"{v['median_positives']:.0f}", pct(v["rankings"]["l2"][hk]["precision"], 0),
             pct(v["rankings"]["avg4w"][hk]["precision"], 0), pts(v["paired"]["l2_vs_avg4w"][hk]["precision"])]
            for hr, v in dc["by_hour"][hh].items()]
    md += [table(rows, ["Window starts", "Typical big windows", "Level 2", "Last 4 weeks", "Level 2 minus last 4 weeks"]), "",
           "Overnight there are almost no big windows, so a night-time shortlist has nothing to find; the "
           "ranking only matters from the morning on.", ""]

    md += ["### Is the direction right?", "",
           "Of the picks that were big, the share where the forecast said gaining vs losing correctly (K=" + hk + ").", ""]
    rows = [[f"{h} h"] + [pct(dc["main"][h]["rankings"][n][hk]["direction"], 1) for n in ("l2", "avg4w", "last_week", "usual")]
            for h in dc["main"]]
    md += [table(rows, ["Horizon", "Level 2", "Last 4 weeks", "Last week", "Usual suspects"]), "",
           "Direction is almost never the hard part: stations that are big at a given hour of the week nearly always "
           "move the same way. The hard part is knowing which ones.", ""]

    md += ["### Exploratory", "",
           "**Ranking by sureness.** Instead of the size of the forecast, rank by how far the nearer edge of the 80% "
           f"range is from zero. Level 2 by sureness minus Level 2 by size at K={hk}: "
           + "; ".join(f"{h} h {pts(dc['main'][h]['paired']['l2_edge_vs_l2'][hk]['precision'])}" for h in dc["main"]) + ".", "",
           f"**How much of the station moves.** Positive = |net| of at least {dc['meta']['dock_share_cutoff']:.0%} of the "
           f"station's docks, ranked by forecast size divided by docks (today's dock counts, {dc['meta']['docks_matched']} of 100 "
           f"stations matched by location or name, the rest left out). K={hk}.", ""]
    rows = []
    for h, v in dc["dock_share"].items():
        r = v["rankings"]
        rows.append([f"{h} h", f"{v['median_positives']:.0f}"] + [pct(r[n][hk]["precision"], 1) for n in ("l2", "avg4w", "last_week", "random")]
                    + [pts(v["paired"]["l2_vs_avg4w"][hk]["precision"])])
    md += [table(rows, ["Horizon", "Typical positives", "Level 2", "Last 4 weeks", "Last week", "Random", "Level 2 minus last 4 weeks"]), ""]
    return md


def skl(x):
    """A Brier skill with its interval, as a signed percentage."""
    return f"{x[0] * 100:+.1f}% ({x[1] * 100:+.1f} to {x[2] * 100:+.1f})"


def availability_section(av):
    """Section 12: probability a station is empty or full, scored on real station counts (src/s05e_availability_check.py)."""
    md = ["## 12. Will the station be empty or full?", "",
          "Net flow is not stock. Here the forecast is scored on real station counts: past snapshots of BIXI's "
          "station feed, saved about every 15 minutes by a public archive (Max Halford's bike-sharing-history), for "
          f"summer 2025. {av['matched_stations']} of {av['of_stations']} stations match. "
          "**This is a retrospective test on past data, not a prospective one.** Level 2 never trained on 2025. "
          "Questions, baselines and the rule for \"better\" were fixed before any result was seen "
          "(plan/05-empty-full-probability.md).", "",
          "- **Empty**: zero bikes while the station is renting. **Full**: zero free docks while it takes returns. "
          "Measured at the snapshot nearest the end of the window (within 10 minutes).",
          "- **Headline population**: stations that are not already empty (or full) at the start.",
          "- **Starting-count rate** (the baseline to beat): how often a station with that many bikes (or free docks) "
          "ended up empty (or full) after that many hours, at that hour of day, learned on May to July 2025. "
          "It uses the real count but no forecast.",
          "- **Direct**: Level 2's forecast and its 80% band read as a normal distribution. **Calibrated**: the same, "
          "passed through a one-variable logistic recalibration fitted on May to July 2025.",
          "- **Score**: Brier score, lower is better. **Skill** = share of the baseline's Brier score removed. "
          "Tested on August and September 2025, with 95% day-block intervals. *Better* only if the paired interval "
          "is wholly above zero.", ""]
    rows = []
    for e in ("empty", "full"):
        for h, r in av["events"][e]["horizons"].items():
            hd = r["headline"]
            b = hd["brier"]
            rows.append([e, f"{h} h", f"{hd['n']:,}", f"{hd['event_rate']:.1%}", f"{b['p_rate'][0]:.4f}",
                         f"{b['p_direct'][0]:.4f}", f"{b['p_cal'][0]:.4f}",
                         skl(hd["paired"]["p_cal_vs_p_rate"]["skill"]), hd["paired"]["p_cal_vs_p_rate"]["verdict"]])
    md += ["### Headline: stations not already empty or full", "",
           table(rows, ["Event", "Horizon", "Windows", "Happened", "Starting-count rate", "Direct", "Calibrated",
                        "Calibrated vs starting-count rate (skill)", "Verdict"]), ""]
    rows = []
    for e in ("empty", "full"):
        for h, r in av["events"][e]["horizons"].items():
            a = r["all"]
            b = a["brier"]
            rows.append([e, f"{h} h", f"{a['n']:,}", f"{a['event_rate']:.1%}", f"{b['p_persist'][0]:.4f}",
                         f"{b['p_rate'][0]:.4f}", f"{b['p_cal'][0]:.4f}", skl(a["paired"]["p_cal_vs_p_persist"]["skill"]),
                         skl(a["paired"]["p_cal_vs_p_rate"]["skill"]) + ", " + a["paired"]["p_cal_vs_p_rate"]["verdict"]])
    md += ["### All stations, including those already empty or full", "",
           table(rows, ["Event", "Horizon", "Windows", "Happened", "Same as now", "Starting-count rate", "Calibrated",
                        "Calibrated vs same as now", "Calibrated vs starting-count rate"]), "",
           "The calibrated version was fitted only on stations that were not already empty or full, so on those it "
           "knows less than the starting-count rate, which sees them as their own bucket.", ""]
    for e in ("empty", "full"):
        r = av["events"][e]["horizons"]["3"]
        rows = [[k, f"{v['n']:,}", f"{v['event_rate']:.1%}", skl(v["paired"]["p_cal_vs_p_rate"]["skill"]),
                 v["paired"]["p_cal_vs_p_rate"]["verdict"]] for k, v in r["by_hour_block"].items()]
        md += [f"### {e.capitalize()} at 3 h, by hour the window starts", "",
               table(rows, ["Starts", "Windows", "Happened", "Calibrated vs starting-count rate", "Verdict"]), ""]
        what = "bikes" if e == "empty" else "free docks"
        rows = [[k, f"{v['n']:,}", f"{v['event_rate']:.1%}", skl(v["paired"]["p_cal_vs_p_rate"]["skill"]),
                 v["paired"]["p_cal_vs_p_rate"]["verdict"]] for k, v in r["by_start"].items()]
        md += [f"### {e.capitalize()} at 3 h, by {what} at the start", "",
               table(rows, [f"{what.capitalize()} at start", "Windows", "Happened", "Calibrated vs starting-count rate", "Verdict"]), ""]
    rows = []
    for name, label in (("p_direct", "Direct"), ("p_cal", "Calibrated")):
        for c in av["events"]["empty"]["horizons"]["3"]["calibration"][name]:
            rows.append([label, c["bin"], f"{c['n']:,}", f"{c['stated']:.0%}", f"{c['happened']:.0%}"])
    md += ["### Does \"30% chance\" happen 30% of the time? (empty, 3 h)", "",
           table(rows, ["Forecaster", "Stated", "Windows", "Mean stated", "Happened"]), "",
           "The direct reading is far too sure: when it says 90%, the station is empty about 20% of the time. "
           "Its band was built for trip-based net flow, and trucks refill stations that trips alone would empty "
           "(see below). The calibrated version is honest but rarely says more than 30%.", ""]
    rows = []
    for h, v in av["truck_effect"].items():
        rows.append([f"{h} h", f"{v['n']:,}", f"{v['corr']:.2f}", f"{v['mean_abs_trip_net']:.1f}",
                     f"{v['mean_abs_gap']:.1f}", f"{v['mean_gap']:+.2f}", f"{v['share_gap_5plus']:.0%}"])
    md += ["### The truck effect, measured", "",
           "Real change in bikes at the station (available plus disabled) against the trip-based net flow for the "
           "same window, May to September 2025. The gap is everything trips do not see: mostly trucks moving bikes, "
           "plus repairs and snapshot timing. The timing was checked: the match peaks at zero offset and vanishes at "
           "one hour either way.", "",
           table(rows, ["Horizon", "Windows", "Correlation", "Mean |trip net flow|", "Mean |gap|", "Mean gap",
                        "Gap of 5+ bikes"]), ""]
    rows = [[k, f"{v['n']:,}", f"{v['mean_abs_gap']:.1f}", f"{v['mean_gap']:+.2f}"]
            for k, v in av["truck_effect"]["3"]["by_hour_block"].items()]
    md += ["3-hour windows by hour they start (a positive mean gap means the station gained more bikes than trips explain):", "",
           table(rows, ["Starts", "Windows", "Mean |gap|", "Mean gap"]), ""]
    return md


def ivp(x, d=1):
    """A share with its interval, in percent."""
    return f"{x[0] * 100:.{d}f}% ({x[1] * 100:.{d}f} to {x[2] * 100:.{d}f})"


def ivpts(x):
    return f"{x[0] * 100:+.1f} pts ({x[1] * 100:+.1f} to {x[2] * 100:+.1f})"


def decision_study_section(ds):
    """Section 13: which 10 stations to check, scored on real station counts (src/s05g_decision_study.py)."""
    md = ["## 13. Which stations should a truck go to?", "",
          f"At each decision time, three rankings pick {ds['k']} stations, scored on real station counts (August and September "
          "2025, as in section 12). A pick counts if the station really is empty or full at the end of the window. "
          "Fixed before results (plan/06). **Forecast size** is what the map's shortlist does; **stock only** picks the "
          "stations closest to empty or full right now; **stock + forecast** uses the chance of ending empty or full "
          "(section 12's method, recalibrated on May to July). BIXI's own rule is not scored: these counts already include "
          "what its trucks did under that rule. **Random** picks from the same candidates (the busiest 100 stations in service at "
          f"both ends of the window, median {ds['horizons']['3']['median_candidates']:.0f}), not the whole network.", ""]
    rows = []
    for h, r in ds["horizons"].items():
        p, pr = r["precision"], r["paired"]
        rows.append([f"{h} h", r["decision_times"], f"{r['base_rate']:.1%}", ivp(p["size"]), ivp(p["stock"]), ivp(p["both"]),
                     f"{p['random'][0]:.1%}", ivpts(pr["both_vs_stock"]["diff"]) + ", " + pr["both_vs_stock"]["verdict"]])
    md += [table(rows, ["Horizon", "Decision times", "Share of candidates that end up a problem", "Forecast size",
                        "Stock only", "Stock + forecast", "Random", "Stock + forecast minus stock only"]), "",
           "Share of the picks that really were empty or full, with 95% day-block intervals.", ""]
    r = ds["horizons"]["3"]["by_hour"]
    rows = [[f"{hr}:00", v["times"], f"{v['precision']['size']:.0%}", f"{v['precision']['stock']:.0%}",
             f"{v['precision']['both']:.0%}", ivpts(v["both_vs_stock"]["diff"]) + ", " + v["both_vs_stock"]["verdict"]]
            for hr, v in r.items()]
    md += ["### By decision time (3 h)", "", table(rows, ["Decision time", "Times", "Forecast size", "Stock only",
                                                       "Stock + forecast", "Stock + forecast minus stock only"]), ""]
    return md


def model_study_section(ms):
    """Section 14: what the model needs, and two alternatives (src/s05f_model_study.py)."""
    md = ["## 14. What does the model need?", "",
          "Each row refits the forecast model with one change and scores it like section 1 (average miss on big windows, "
          "2024 to 2026) against the current Level 2, with 95% day-block intervals on the paired difference. "
          f"Neighbours are stations within {ms['neighbour_m']} m (median {ms['median_neighbours']} per station). "
          "**Two flows** forecasts arrivals and departures separately (Poisson loss) and subtracts. Fixed before results (plan/06).", ""]
    rows = []
    for h, r in ms["horizons"].items():
        for name, v in list(r["ablation"].items()):
            rows.append([f"{h} h", f"without {name}", f"{v['mae']:.2f}", skl(v["relative"]), v["verdict"]])
        for name, key in (("with neighbours", "neighbours"), ("two flows", "two_flows")):
            v = r[key]
            rows.append([f"{h} h", name, f"{v['mae']:.2f}", skl(v["relative"]), v["verdict"]])
    md += [table(rows, ["Horizon", "Change", "Miss on big windows", "Change in miss vs Level 2", "Verdict"]), "",
           "Level 2's own miss: " + "; ".join(f"{h} h {r['base_mae']:.2f}" for h, r in ms["horizons"].items()) +
           ". A positive change means the variant misses more (worse).", ""]
    return md


def confirmation_section(cf, av, ds):
    """Section 15: sections 12 and 13 rerun once on a sealed month, September 2024 (src/s05h_confirmation.py)."""
    v, cov = cf["verdict"], cf["coverage"]
    md = ["## 15. Sealed confirmation: September 2024", "",
          "Sections 12 and 13 were built and read on summer 2025. Here both are rerun, unchanged and **once**, on a month "
          "nobody had looked at: September 2024, under a plan written before its snapshots were downloaded "
          "(plan/07-sealed-confirmation-sept-2024.md). Nothing is fitted on it: the starting-count table and the "
          "recalibrations are the ones fitted on May to July 2025. Level 2 never trained on 2024. "
          f"{cov['matched_stations']} of {cov['of_stations']} stations match; {cov['snapshots']:,} snapshots on "
          f"{cov['days_with_windows']} days; {cov['with_both_snapshots']:,} forecast windows with a count at both ends.", "",
          "Written down before the run: confirmed if, at 3 h, (1) stock + forecast and stock only are both clearly above "
          "forecast size, and (2) the calibrated chance is not worse than the starting-count rate, for empty and for full.", "",
          f"**Result: {'confirmed' if v['confirmed'] else 'not confirmed'}.** Ranking order: "
          f"{'holds' if v['ranking_order']['holds'] else 'fails'}; empty/full skill: "
          f"{'holds' if v['empty_full_skill']['holds'] else 'fails'}.", ""]
    rows = []
    for h, r in cf["decision"].items():
        p, ref = r["precision"], ds["horizons"][h]["precision"]
        rows.append([f"{h} h", ivp(p["size"]), ivp(p["stock"]), ivp(p["both"]), f"{p['random'][0]:.1%}",
                     ivpts(r["paired"]["both_vs_stock"]["diff"]) + ", " + r["paired"]["both_vs_stock"]["verdict"],
                     f"{ref['size'][0]:.1%} / {ref['stock'][0]:.1%} / {ref['both'][0]:.1%}"])
    md += ["### Which stations should a truck go to? (as section 13)", "",
           table(rows, ["Horizon", "Forecast size", "Stock only", "Stock + forecast", "Random",
                        "Stock + forecast minus stock only", "Aug–Sep 2025 (size / stock / both)"]), ""]
    rows = []
    for e in ("empty", "full"):
        for h, r in cf["empty_full"][e]["horizons"].items():
            hd, ref = r["headline"], av["events"][e]["horizons"][h]["headline"]
            pr = hd["paired"]["p_cal_vs_p_rate"]
            rows.append([e, f"{h} h", f"{hd['n']:,}", f"{hd['event_rate']:.1%}", skl(pr["skill"]), pr["verdict"],
                         skl(ref["paired"]["p_cal_vs_p_rate"]["skill"])])
    md += ["### Will the station be empty or full? (as section 12, stations not already empty or full)", "",
           table(rows, ["Event", "Horizon", "Windows", "Happened", "Calibrated vs starting-count rate (skill)", "Verdict",
                        "Aug–Sep 2025 skill"]), ""]
    top = cf["empty_full"]["empty"]["horizons"]["3"]["calibration"]["p_direct"][-1]
    md += ["The direct reading is still far too sure: in its top bin (mean stated "
           f"{top['stated']:.0%}) the station was empty {top['happened']:.0%} of the time (empty, 3 h). "
           "Truck effect, as in section 12: correlation between the real change in bikes and trip net flow "
           + ", ".join(f"{t['corr']:.2f} at {h} h" for h, t in cf["truck_effect"].items()) + ".", "",
           "What did not repeat: in 2025, stock + forecast beat stock only at 1 h; on September 2024 it is a tie at every "
           "horizon. The forecast's extra over the stock count is within the noise. At 1 h the empty and full skills also "
           "swap between the two periods (each is better in one and a tie in the other), so 1 h is the least stable "
           "horizon; 3 h and 6 h hold in both.", ""]
    return md


def real_change_section(rc):
    """Section 16: forecast against the real change in bikes (src/s05i_real_change_check.py). Exploratory."""
    H = rc["horizons"]
    md = ["## 16. Forecast against the real change in bikes (exploratory)", "",
          "Every section above scores the forecast against **trip net flow**, which is what it was trained to predict. "
          "With real station counts for " + rc["period"] + " the **real change in bikes** at a station is also known "
          "(available plus disabled, end of window minus start). Trucks move bikes without leaving a trip, so the two "
          "differ. **Exploratory:** this question was not written down in advance (it came from drawing the real change "
          "on the map's 2025 station chart), it covers one period, and the sealed month (September 2024) is not used. "
          "\"Big\" means the 10% of windows with the largest |real change| at each look-ahead (cut-offs "
          + ", ".join(f"{v['big_cutoff']:.0f} bikes at {h} h" for h, v in H.items()) + "). \"No change\" is the forecast "
          "of zero. 95% day-block intervals.", ""]
    rows = []
    for h, v in H.items():
        rows.append([f"{h} h", f"{v['windows']:,}", f"{v['corr_forecast_trips'][0]:.2f}", f"{v['corr_forecast_real'][0]:.2f}",
                     skl(v["skill_vs_zero_trips"]), skl(v["skill_vs_zero_real"]), skl(v["skill_vs_zero_real_big"]),
                     ivp(v["direction_right_big_real"], 0), ivp(v["range_held_trips"], 0), ivp(v["range_held_real"], 0)])
    md += [table(rows, ["Horizon", "Windows", "Correlation with trips", "Correlation with real change",
                        "Skill vs no change, scored on trips", "Skill vs no change, scored on real change",
                        "Same, on the biggest real changes", "Direction right on the biggest real changes",
                        "80% range holds, trips", "80% range holds, real change"]), "",
           "Skill = share of the \"no change\" forecast's average miss removed (negative means worse than no change).", "",
           "- The forecast follows trips closely and the real change in bikes loosely.",
           "- Over all windows it is **worse than predicting no change** against the real change: trucks keep stocks "
           "steadier than trips alone would, so \"no change\" is a strong baseline there.",
           "- On the biggest real changes it still helps, more at longer look-aheads, and gets the direction right "
           "56% (1 h) to 80% (6 h) of the time.",
           "- The 80% range, calibrated to trips, holds fewer real changes the longer the look-ahead. This is the truck "
           "effect of section 12 measured on the forecast itself, and why the empty/full chance needed recalibrating.",
           "- The model is a forecast of demand imbalance, not of stock. Training on the real change is not possible: "
           "station counts exist only from 2025, not for the training years 2022 and 2023.", ""]
    return md


def main():
    p = pd.read_parquet(PROCESSED / "predictions.parquet")
    top = pd.read_csv(PROCESSED / "top_stations.csv").set_index("station_id")
    M = {"horizons": {}, "by_year": {}, "by_hour": {}, "by_station": {}, "weather": {}, "extremes": {}}
    md = ["# Backtest results",
          "",
          "Generated by `src/s06_report.py`. Do not edit by hand; re-run the script.",
          "",
          f"Trained on summers 2022 and 2023. Scored on summers {", ".join(str(int(y)) for y in sorted(p.year.unique()))}, "
          f"{p.station_id.nunique()} stations, {p.hour_idx.nunique():,} hourly cut-offs, "
          f"{len(p) // len(HORIZONS):,} station-hours per horizon. All errors are in bikes.",
          ""]

    # ---------------------------------------------------------------- headline, per horizon
    md += ["## 1. The main result: error on big-imbalance windows", "",
           f"Big-imbalance = the top {BIG_IMBALANCE_TOP_SHARE:.0%} of |net bikes| across all station-windows, "
           "with the cut-off set separately for each horizon. Lower error is better.", ""]
    rows_big, rows_all, rows_band = [], [], []
    cut = {}
    for h in HORIZONS:
        d = p[p.h == h]
        cut[h] = scoring.big_imbalance_cutoff(d.y.values, BIG_IMBALANCE_TOP_SHARE)
        s = score_block(d, cut[h])
        s["cutoff"] = cut[h]
        M["horizons"][h] = s
        rows_big.append([f"{h} h", f"±{cut[h]:.0f}+", f"{s['n_big']:,}"]
                        + [f"{s[f'big_mae_{m}']:.2f}" for m in MODELS]
                        + [f"{s['big_skill_l2_vs_zero']:.0%}", f"{s['big_skill_l2_vs_best_simple']:.0%}",
                           f"{s['big_direction_l2']:.0%}"])
        rows_all.append([f"{h} h"] + [f"{s[f'mae_{m}']:.2f}" for m in MODELS])
        rows_band.append([f"{h} h"] + [f"{s[f'coverage_{m}']:.1%}" for m in ("l1", "l2_raw", "l2")]
                         + [f"{s[f'band_width_{m}']:.1f}" for m in ("l1", "l2_raw", "l2")]
                         + [f"{s['big_coverage_l2']:.1%}"])
    md += [table(rows_big, ["Horizon", "Big means", "Windows", "Predict zero", "Just before", "Last week", "Last 4 weeks", "Level 1", "Level 2",
                            "Level 2 vs zero", "Level 2 vs best simple", "Level 2 direction right"]), "",
           "\"Level 2 vs zero\" is the share of predict-zero's error that Level 2 removes. "
           "\"vs best simple\" compares against whichever of the five simpler methods did best. \"Just before\" repeats the net flow of the window of the same length that just ended.", "",
           "## 2. Error on all windows", "",
           "Over all windows, predict-zero's miss is close to the model's at 1 h, so this average cannot tell them apart well. "
           "Net flow is skewed: " + "; ".join(
               f"at {h} h the median window moves {np.median(np.abs(p[p.h == h].y)):.0f} bikes and the biggest 10% carry "
               f"{(lambda a: a[a >= np.quantile(a, 0.9)].sum() / a.sum())(np.abs(p[p.h == h].y.values)):.0%} of all movement"
               for h in HORIZONS) + ". The big windows are where stations empty or fill, which is why the main metric is section 1.", "",
           table(rows_all, ["Horizon", "Predict zero", "Just before", "Last week", "Last 4 weeks", "Level 1", "Level 2"]), "",
           "## 3. Uncertainty bands", "",
           f"Each band is meant to contain {BAND_TARGET:.0%} of outcomes. Width is in bikes; narrower is more useful, "
           "as long as coverage holds.", "",
           "Level 2 has two bands. \"As trained\" is the band straight from the model. \"Self-corrected\" stretches "
           "or shrinks it each day using the hits and misses of the previous 28 days (only outcomes already known).", "",
           table(rows_band, ["Horizon", "Level 1 coverage", "Level 2 as trained", "Level 2 self-corrected",
                             "Level 1 width", "Level 2 as trained width", "Level 2 self-corrected width",
                             "Self-corrected coverage on big windows"]), "",
           "Coverage on big windows is expected to be low: those windows are picked *because* the outcome was extreme.", ""]

    # ---------------------------------------------------------------- by year
    md += ["## 4. Does it hold up year after year?", "",
           "The model is never retrained, so later years are further from what it learned.", ""]
    rows = []
    for h in HORIZONS:
        for yr, d in p[p.h == h].groupby("year"):
            s = score_block(d, cut[h])
            M["by_year"][f"{h}_{yr}"] = s
            rows.append([f"{h} h", yr, d.station_id.nunique(), f"{s['n_big']:,}", f"{s['big_mae_zero']:.2f}", f"{s['big_mae_persist']:.2f}",
                         f"{s['big_mae_last_week']:.2f}", f"{s['big_mae_avg4w']:.2f}", f"{s['big_mae_l1']:.2f}",
                         f"{s['big_mae_l2']:.2f}", f"{s['big_skill_l2_vs_best_simple']:.0%}",
                         f"{s['coverage_l2_raw']:.1%}", f"{s['coverage_l2']:.1%}"])
    md += [table(rows, ["Horizon", "Year", "Stations in service", "Big windows", "Predict zero", "Just before", "Last week",
                        "Last 4 weeks", "Level 1", "Level 2", "Level 2 vs best simple", "Band as trained",
                        "Band self-corrected"]), ""]

    # ---------------------------------------------------------------- honest weather
    md += ["## 5. What honest weather costs", "",
           "Level 2 scored twice on the same windows: once with the forecast that existed at the time, "
           "once with the weather that really happened (which nobody could have known). "
           "The gap is the price of not cheating.", ""]
    rows = []
    for h in HORIZONS:
        d = p[p.h == h]
        s = M["horizons"][h]
        wet = d[d.precip_act >= 1.0]
        s_wet = score_block(wet, cut[h])
        M["weather"][h] = {"wet": s_wet, "temp_forecast_mae": float((d.temp_fc - d.temp_act).abs().mean())}
        rows.append([f"{h} h", f"{s['big_mae_l2']:.3f}", f"{s['big_mae_l2_actual_weather']:.3f}",
                     f"{s_wet['n']:,}", f"{s_wet['mae_l2']:.3f}", f"{s_wet['mae_l2_actual_weather']:.3f}",
                     f"{s_wet['mae_zero']:.3f}"])
    md += [table(rows, ["Horizon", "Big windows: forecast weather", "Big windows: actual weather",
                        "Rainy windows (1 mm+)", "Rainy: forecast weather", "Rainy: actual weather",
                        "Rainy: predict zero"]), ""]

    # ---------------------------------------------------------------- failure analysis
    md += ["## 6. Where it fails", "", "### By hour of day (3-hour horizon, hour the window starts)", ""]
    d3 = p[p.h == 3]
    rows = []
    for hr, d in d3.groupby("hour"):
        s = score_block(d, cut[3])
        M["by_hour"][int(hr)] = s
        rows.append([f"{hr:02d}:00", f"{s['n_big']:,}", f"{s['big_mae_zero']:.1f}", f"{s['big_mae_l2']:.1f}",
                     f"{s['big_skill_l2_vs_zero']:.0%}" if s["n_big"] else "-", f"{s['coverage_l2']:.0%}"])
    md += [table(rows, ["Window starts", "Big windows", "Predict zero", "Level 2", "Level 2 vs zero",
                        "Band coverage"]), ""]

    st_rows = []
    for sid, d in d3.groupby("station_id"):
        s = score_block(d, cut[3])
        if s["n_big"] >= 50:
            M["by_station"][int(sid)] = {k: s[k] for k in ("n", "n_big", "big_mae_zero", "big_mae_l2",
                                                           "big_skill_l2_vs_zero", "coverage_l2", "mae_l2")}
            st_rows.append((s["big_skill_l2_vs_zero"], top.loc[sid, "name"], s))
    st_rows.sort(key=lambda r: r[0])
    fmt = lambda r: [r[1], f"{r[2]['n_big']:,}", f"{r[2]['big_mae_zero']:.1f}", f"{r[2]['big_mae_l2']:.1f}", f"{r[0]:.0%}"]
    head = ["Station", "Big windows", "Predict zero", "Level 2", "Level 2 vs zero"]
    md += ["### Stations where Level 2 helps least (3-hour horizon, at least 50 big windows)", "",
           table([fmt(r) for r in st_rows[:8]], head), "",
           "### Stations where Level 2 helps most", "", table([fmt(r) for r in st_rows[-8:][::-1]], head), ""]

    # The very largest swings: festivals, events, anything the calendar does not know about.
    md += ["### The most extreme windows", "",
           "The top 1 in 1,000 windows by |net bikes|. These are mostly events the model has no input for.", ""]
    rows = []
    for h in HORIZONS:
        d = p[p.h == h]
        x = d[d.y.abs() >= d.y.abs().quantile(0.999)]
        M["extremes"][h] = {"n": int(len(x)), "min_abs": float(x.y.abs().min()),
                            "mae_zero": scoring.mae(x.y, x.zero), "mae_l2": scoring.mae(x.y, x.l2),
                            "coverage_l2": scoring.coverage(x.y, x.l2_lo, x.l2_hi)}
        e = M["extremes"][h]
        rows.append([f"{h} h", e["n"], f"±{e['min_abs']:.0f}+", f"{e['mae_zero']:.1f}", f"{e['mae_l2']:.1f}",
                     f"{e['coverage_l2']:.0%}"])
    md += [table(rows, ["Horizon", "Windows", "Size", "Predict zero", "Level 2", "Inside Level 2 band"]), ""]

    # ---------------------------------------------------------------- how the settings were picked
    val = json.loads((REPORTS / "validation.json").read_text())
    imp = pd.read_csv(REPORTS / "feature_importance.csv")
    md += ["## 7. How the Level 2 settings were picked", "",
           "Learned on 2022, checked on 2023 (big-window error, in bikes). The lowest total won. "
           "The backtest years played no part.", "",
           table([[("yes" if c["with_station_code"] else "no"), c["max_leaf_nodes"]]
                  + [c[f"big_mae_h{h}"] for h in HORIZONS] + [c["total"]] for c in val["candidates"]],
                 ["Station identity as an input", "Tree size (leaves)"] + [f"{h} h" for h in HORIZONS] + ["Total"]),
           "", f"Picked: station identity = {'yes' if val['picked']['with_station_code'] else 'no'}, "
               f"leaves = {val['picked']['max_leaf_nodes']}.", "",
           "## 8. Which inputs Level 2 leans on", "",
           "Shuffle one input at a time and see how much the error grows (3-hour horizon, backtest sample). "
           "Bigger = the model depends on it more.", "",
           table([[r.feature, f"{r.mae_increase_when_shuffled:.3f}"] for r in imp.itertuples()],
                 ["Input", "Error increase when shuffled (bikes)"]), ""]

    retrain = json.loads((REPORTS / "retrain_check.json").read_text())
    weather = json.loads((REPORTS / "weather_check.json").read_text())

    md += ["## 9. Would retraining every year help?", "",
           "The main model learns from 2022 and 2023 and is never retrained. Here, each later year is also scored "
           "with a model refit on every earlier year (2025 uses 2022 to 2024, 2026 uses 2022 to 2025), "
           "using only data that existed before that year. Average miss on big-imbalance windows, in bikes. "
           "(`src/s05b_retrain_check.py`)", ""]
    rows = []
    for h in HORIZONS:
        for yr, r in retrain[str(h)].items():
            best = r["last_4_weeks"]
            if r["retrained"] is None:
                rows.append([f"{h} h", yr, "2022 to 2023", f"{r['never_retrained']:.2f}", "same model", f"{best:.2f}",
                             f"{scoring.skill(r['never_retrained'], best):+.0%}", "-"])
            else:
                rows.append([f"{h} h", yr, f"2022 to {int(yr) - 1}", f"{r['never_retrained']:.2f}", f"{r['retrained']:.2f}",
                             f"{best:.2f}", f"{scoring.skill(r['never_retrained'], best):+.0%}",
                             f"{scoring.skill(r['retrained'], best):+.0%}"])
    md += [table(rows, ["Horizon", "Test year", "Retrained model learns from", "Never retrained", "Retrained",
                        "Last 4 weeks", "Never retrained vs last 4 weeks", "Retrained vs last 4 weeks"]), "",
           "A positive percentage means the model has less error than the 4-week average. "
           "Retraining adds data from the years the network grew into, and it brings back most of the lead the "
           "never-retrained model loses by 2026.", ""]

    md += ["## 10. How much does weather matter?", "",
           "Level 2 refit with every weather input removed (same 2022 to 2023 training, same backtest windows), "
           "against the main model with the forecast weather that existed at the time, and against a model given "
           "the weather that really happened (which no live system could have). Average miss in bikes. "
           "Rainy = 1 mm or more of rain over the window. (`src/s05c_weather_check.py`)", ""]
    rows = []
    for h in HORIZONS:
        for name, label in (("all", "All windows"), ("big", "Big-imbalance windows"), ("rainy", "Rainy windows")):
            r = weather[str(h)][name]
            rows.append([f"{h} h", label, f"{r['n']:,}", f"{r['zero']:.2f}", f"{r['no_weather']:.3f}", f"{r['l2']:.3f}",
                         f"{r['l2_actual_weather']:.3f}", f"{scoring.skill(r['l2'], r['no_weather']):+.1%}"])
    md += [table(rows, ["Horizon", "Windows", "Count", "Predict zero", "No weather", "Forecast weather (main model)",
                        "Actual weather (cheating)", "Forecast weather vs no weather"]), "",
           "A positive last column means the weather forecast lowered the error. Weather is one point (downtown "
           "Montreal) for every station.", ""]

    dc = json.loads((REPORTS / "decision_check.json").read_text())
    md += decision_section(dc)
    M["decision"] = dc

    avf = REPORTS / "availability_check.json"
    if avf.exists():
        av = json.loads(avf.read_text())
        md += availability_section(av)
        M["availability"] = av
    dsf = REPORTS / "decision_study.json"
    if dsf.exists():
        dstudy = json.loads(dsf.read_text())
        md += decision_study_section(dstudy)
        M["decision_study"] = dstudy
    msf = REPORTS / "model_study.json"
    if msf.exists():
        mstudy = json.loads(msf.read_text())
        md += model_study_section(mstudy)
        M["model_study"] = mstudy
    cff = REPORTS / "confirmation_2024_09.json"
    if cff.exists() and avf.exists() and dsf.exists():
        cf = json.loads(cff.read_text())
        md += confirmation_section(cf, av, dstudy)
        M["confirmation"] = cf
    rcf = REPORTS / "real_change_check.json"
    if rcf.exists():
        rcc = json.loads(rcf.read_text())
        md += real_change_section(rcc)
        M["real_change"] = rcc

    M["validation"] = val
    M["retrain"] = retrain
    M["weather_check"] = weather
    M["feature_importance"] = imp.to_dict("records")
    (REPORTS / "metrics.json").write_text(json.dumps(M, indent=1, default=float))
    (REPORTS / "RESULTS.md").write_text("\n".join(md))
    print("\n".join(md[:60]))


if __name__ == "__main__":
    main()
