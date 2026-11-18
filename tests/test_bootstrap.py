"""The day-block bootstrap: repeatable, and honest about how much data it has."""
import numpy as np
import pandas as pd

import bootstrap as b


def frame(n_days, seed=0):
    rng = np.random.default_rng(seed)
    d = np.repeat(np.arange(n_days), 8)
    return pd.DataFrame({"day": d, "hit": rng.integers(0, 2, len(d)), "n": 1})


def stat(t):
    return t["hit"] / t["n"]


def test_same_seed_gives_identical_output():
    f = frame(60)
    a = b.day_block_bootstrap(f, "day", ["hit", "n"], stat, n=500, seed=3)
    c = b.day_block_bootstrap(f, "day", ["hit", "n"], stat, n=500, seed=3)
    assert a == c
    assert a != b.day_block_bootstrap(f, "day", ["hit", "n"], stat, n=500, seed=4)


def test_interval_narrows_with_more_days():
    w = lambda iv: iv[2] - iv[1]
    assert w(b.day_block_bootstrap(frame(400), "day", ["hit", "n"], stat, seed=1)) < \
           w(b.day_block_bootstrap(frame(25), "day", ["hit", "n"], stat, seed=1))


def test_point_estimate_is_the_plain_ratio_and_sits_inside_its_interval():
    f = frame(100)
    p, lo, hi = b.day_block_bootstrap(f, "day", ["hit", "n"], stat, seed=0)
    assert p == f.hit.sum() / len(f) and lo <= p <= hi


def test_paired_difference_of_identical_series_is_zero_and_a_tie():
    f = frame(80)
    f["hit2"] = f.hit
    p, lo, hi = b.day_block_bootstrap(f, "day", ["hit", "hit2", "n"], lambda t: (t["hit"] - t["hit2"]) / t["n"], seed=0)
    assert (p, lo, hi) == (0, 0, 0) and b.verdict(lo, hi) == "tie"


def test_verdict_reads_the_interval():
    assert b.verdict(0.01, 0.05) == "better" and b.verdict(-0.05, -0.01) == "worse" and b.verdict(-0.01, 0.02) == "tie"


def test_a_clearly_better_series_is_called_better():
    f = frame(200)
    f["win"] = 1
    f["lose"] = 0
    p, lo, hi = b.day_block_bootstrap(f, "day", ["win", "lose", "n"], lambda t: (t["win"] - t["lose"]) / t["n"], seed=0)
    assert b.verdict(lo, hi) == "better"
