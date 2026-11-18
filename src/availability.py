"""Will a station be empty or full? Plain functions, checked by hand in tests/test_availability.py.

Empty = zero bikes available while the station is renting. Full = zero free docks while it is returning.
A forecast here is a probability. The score is the Brier score: the average of (probability - outcome)^2,
where the outcome is 1 if it happened and 0 if not. Lower is better; always saying 0.5 scores 0.25.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

Z90 = norm.ppf(0.9)                      # the 80% band runs from the 10th to the 90th percentile
MIN_SIGMA = 0.5                          # bikes; keeps a near-zero band from giving certainties
EDGES = [0, 1, 2, 3, 4, 5, 6, 9, 13, 21, np.inf]    # starting-count buckets: 0, 1, 2, 3, 4, 5, 6-8, 9-12, 13-20, 21+
MIN_CELL = 30                            # fewer training cases than this in a cell: fall back to a coarser cell
TOL_S = 600                              # a snapshot counts for an hour mark if it is within 10 minutes of it


def nearest_snapshot(marks, snaps, tol=TOL_S):
    """For each time mark (seconds), the index of the nearest snapshot time (seconds, sorted), or -1 if none
    is within `tol` seconds."""
    marks, snaps = np.asarray(marks), np.asarray(snaps)
    pos = np.clip(np.searchsorted(snaps, marks), 1, len(snaps) - 1)
    left, right = snaps[pos - 1], snaps[pos]
    pick = np.where(np.abs(marks - left) <= np.abs(right - marks), pos - 1, pos)
    return np.where(np.abs(snaps[pick] - marks) <= tol, pick, -1)


def sigma_from_band(lo, hi):
    return np.maximum((np.asarray(hi, float) - np.asarray(lo, float)) / (2 * Z90), MIN_SIGMA)


def prob_empty_direct(bikes0, mu, lo, hi):
    """P(bikes at the end <= 0) if the change is normal with mean `mu` and the 80% band [lo, hi].
    The 0.5 treats the count as whole bikes."""
    return norm.cdf((0.5 - np.asarray(bikes0, float) - np.asarray(mu, float)) / sigma_from_band(lo, hi))


def prob_full_direct(docks0, mu, lo, hi):
    """P(free docks at the end <= 0): every bike gained takes one free dock."""
    return norm.sf((np.asarray(docks0, float) - 0.5 - np.asarray(mu, float)) / sigma_from_band(lo, hi))


def bucket(n):
    """Index of the starting-count bucket (see EDGES)."""
    return np.searchsorted(EDGES, np.asarray(n, float), side="right") - 1


def rate_table(train, count_col, event_col):
    """The starting-count baseline: share of training cases that ended in the event, by starting-count bucket,
    horizon and local hour of day, with a fallback to bucket and horizon for thin cells.
    `train` must hold only training months. Returns a function frame -> probabilities."""
    t = train.assign(b=bucket(train[count_col]))
    fine = t.groupby(["b", "h", "local_hour"])[event_col].agg(["mean", "size"])
    coarse = t.groupby(["b", "h"])[event_col].agg(["mean", "size"])
    overall = float(t[event_col].mean())
    fine = fine[fine["size"] >= MIN_CELL]["mean"]
    coarse = coarse[coarse["size"] >= MIN_CELL]["mean"]

    def predict(frame):
        b = bucket(frame[count_col].values)
        k_fine = pd.MultiIndex.from_arrays([b, frame.h.values, frame.local_hour.values])
        k_coarse = pd.MultiIndex.from_arrays([b, frame.h.values])
        p = fine.reindex(k_fine).values
        p = np.where(np.isnan(p), coarse.reindex(k_coarse).values, p)
        return np.where(np.isnan(p), overall, p)
    return predict


def logit(p, eps=1e-4):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def recalibrate(p_train, y_train):
    """One-variable logistic recalibration of a probability, fitted on training months only."""
    m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p_train).reshape(-1, 1), np.asarray(y_train, int))
    return lambda p: m.predict_proba(logit(p).reshape(-1, 1))[:, 1]


def brier(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def calibration_table(p, y, bins=10):
    """Equal-width bins of the stated probability: mean stated, share that happened, count."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    k = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for i in range(bins):
        m = k == i
        if m.any():
            rows.append({"bin": f"{i / bins:.1f}-{(i + 1) / bins:.1f}", "n": int(m.sum()),
                         "stated": float(p[m].mean()), "happened": float(y[m].mean())})
    return rows
