"""The scoring rules. Written and tested before any model, and never changed to suit one.

y      = what actually happened (net bikes over the window)
pred   = the forecast
lo, hi = the lower and upper edge of the uncertainty band
"""
import numpy as np


def mae(y, pred):
    """Mean absolute error: on average, how many bikes the forecast was off by."""
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


def big_imbalance_cutoff(y, top_share):
    """The |net| value that separates the top `top_share` of windows from the rest."""
    return float(np.quantile(np.abs(np.asarray(y)), 1 - top_share))


def big_imbalance_mask(y, cutoff):
    """True for the windows that count as big-imbalance. A cutoff of 0 would keep everything,
    so windows must also be non-zero."""
    a = np.abs(np.asarray(y))
    return (a >= cutoff) & (a > 0)


def skill(model_error, baseline_error):
    """Share of the baseline's error the model removed. 0 = no better, 1 = perfect, below 0 = worse."""
    return float(1 - model_error / baseline_error)


def coverage(y, lo, hi):
    """Share of outcomes that landed inside the band. An 80% band should score close to 0.80."""
    y = np.asarray(y)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))


def pinball(y, q_pred, q):
    """Pinball loss for a quantile forecast. Lower is better.
    Being too low costs q per bike, being too high costs (1 - q) per bike."""
    diff = np.asarray(y) - np.asarray(q_pred)
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def direction_hit_rate(y, pred):
    """Share of windows where the forecast got the sign right (filling up vs emptying)."""
    y, pred = np.asarray(y), np.asarray(pred)
    return float(np.mean(np.sign(y) == np.sign(np.round(pred))))
