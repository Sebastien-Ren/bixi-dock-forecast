"""Turn forecasts into a decision: which K stations to look at, and how many were really big?

Everything here is a plain function on arrays, so it can be checked by hand on a small example.
Rankings only ever see forecast numbers. The real outcome is used afterwards, to score the picks.
"""
import numpy as np
import pandas as pd


def decision_mask(hour, h):
    """True for the hours a decision is made. Hourly for the 1 h horizon; every h hours (00:00, 03:00 ...)
    for longer ones, so windows never overlap and no outcome is counted twice."""
    hour = np.asarray(hour)
    return np.ones(len(hour), bool) if h == 1 else (hour % h == 0)


def rank_within_time(time_id, station_id, score, tiebreak=None):
    """0 for the best pick at each time, 1 for the next, ... Highest score first.
    Ties break on `tiebreak` (higher first, if given) and then on the lower station id,
    so the answer never depends on row order."""
    time_id, station_id = np.asarray(time_id), np.asarray(station_id)
    score = np.asarray(score, float)
    tb = np.zeros(len(score)) if tiebreak is None else np.asarray(tiebreak, float)
    order = np.lexsort((station_id, -tb, -score, time_id))
    t_sorted = time_id[order]
    start = np.r_[True, t_sorted[1:] != t_sorted[:-1]]
    first = np.maximum.accumulate(np.where(start, np.arange(len(order)), 0))
    rank = np.empty(len(order), int)
    rank[order] = np.arange(len(order)) - first
    return rank


def pick_table(time_id, station_id, score, positive, ks, pred=None, y=None, tiebreak=None):
    """One row per decision time: candidates, positives, and for each K the number of picks that were
    positive (`hits_K`), the capture ratio (`cap_K`) and how many of those hits had the right direction
    (`dir_K`, only if pred and y are given)."""
    time_id = np.asarray(time_id)
    positive = np.asarray(positive, bool)
    rank = rank_within_time(time_id, station_id, score, tiebreak)
    df = pd.DataFrame({"t": time_id, "pos": positive})
    g = df.groupby("t")
    out = pd.DataFrame({"n_cand": g.size(), "n_pos": g.pos.sum()})
    if pred is not None:
        right = positive & (np.sign(np.round(np.asarray(pred, float))) == np.sign(np.asarray(y, float)))
    for k in ks:
        pick = rank < k
        out[f"hits_{k}"] = df.assign(v=pick & positive).groupby("t").v.sum()
        out[f"cap_{k}"] = np.where(out.n_pos > 0, out[f"hits_{k}"] / np.minimum(k, out.n_pos.clip(lower=1)), 0.0)
        if pred is not None:
            out[f"dir_{k}"] = df.assign(v=pick & right).groupby("t").v.sum()
    return out


def random_table(base, ks):
    """The random floor, in expectation: picking K at random out of n candidates with p positives
    catches K * p / n on average. Exact, so no luck of a random draw is added to the comparison."""
    out = base[["n_cand", "n_pos"]].copy()
    for k in ks:
        kk = np.minimum(k, out.n_cand)
        out[f"hits_{k}"] = kk * out.n_pos / out.n_cand
        out[f"cap_{k}"] = np.where(out.n_pos > 0, out[f"hits_{k}"] / np.minimum(k, out.n_pos.clip(lower=1)), 0.0)
    return out


def edge_from_zero(lo, hi):
    """How far the nearer edge of the uncertainty range is from zero (0 if the range includes zero).
    Large = the model is sure the window is not quiet."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return np.where(lo > 0, lo, np.where(hi < 0, -hi, 0.0))


# --- statistics from summed columns (work on the real data and on bootstrap resamples) -------------

def precision(tot, name, k):
    """Of all picks, the share that were big."""
    return tot[f"hits_{name}_{k}"] / (k * tot["n_times"])


def capture(tot, name, k):
    """Average over times that had at least one positive: hits / min(K, positives). 1.0 = perfect."""
    return tot[f"cap_{name}_{k}"] / tot["has_pos"]


def direction(tot, name, k):
    """Of the picks that were big, the share where the forecast got gaining-vs-losing right."""
    return tot[f"dir_{name}_{k}"] / tot[f"hits_{name}_{k}"]


def base_rate(tot):
    """Share of all candidates that were big. Picking at random hits this."""
    return tot["n_pos"] / tot["n_cand"]
