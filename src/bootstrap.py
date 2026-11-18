"""Day-block bootstrap: how much could a number move by luck?

Neighbouring hours resemble each other (the same rain, the same festival), so we do not shuffle rows.
We shuffle whole days: draw as many days as there are, with replacement, and recompute the number.
Do that thousands of times; the middle 95% of the results is the 95% interval.

Every statistic here is a ratio of sums (hits / picks, bikes missed / windows ...), so each day is
summarised once as a row of sums and a resample is a weighted sum of those rows. That keeps it fast.
"""
import numpy as np


def day_sums(frame, day_col, cols):
    """One row of sums per day. Returns (days, matrix of shape (n_days, len(cols)))."""
    g = frame.groupby(day_col)[list(cols)].sum().sort_index()
    return g.index.values, g.values.astype(float)


def resampled_totals(matrix, n, seed):
    """n resamples of the whole period. Each one is the column totals over a with-replacement draw of days.
    Returns an array of shape (n, n_columns). The same draws are reused for every statistic computed
    from it, which is what makes paired differences fair."""
    rng = np.random.default_rng(seed)
    d = matrix.shape[0]
    counts = rng.multinomial(d, np.full(d, 1.0 / d), size=n)    # how many times each day was drawn
    return counts @ matrix


def interval(point, boot, level=0.95):
    """(point, low, high) from the bootstrap results of one statistic."""
    a = (1 - level) / 2
    lo, hi = np.quantile(boot, [a, 1 - a])
    return float(point), float(lo), float(hi)


def day_block_bootstrap(frame, day_col, cols, stat_fn, n=2000, seed=0):
    """Convenience wrapper. `stat_fn(totals)` receives a dict {column: total} where each total is a float
    (the real data) or an array of n values (the resamples), and returns the statistic."""
    _, m = day_sums(frame, day_col, cols)
    names = list(cols)
    point = stat_fn(dict(zip(names, m.sum(axis=0))))
    boot = stat_fn(dict(zip(names, resampled_totals(m, n, seed).T)))
    return interval(point, boot)


def verdict(lo, hi):
    """Read a paired-difference interval: 'better' if wholly above zero, 'worse' if wholly below, else 'tie'."""
    return "better" if lo > 0 else "worse" if hi < 0 else "tie"
