"""Checks for the sealed confirmation (sealed plan): it only applies parts fitted on May to July 2025."""
import numpy as np
import pandas as pd
import pytest

import availability as av
import s05e_availability_check as chk
import s05h_confirmation as conf


def fake_pairs(start, hours, seed=0):
    """A few stations' hourly counts and Level 2 forecasts, joined the way the real pipeline joins them."""
    rng = np.random.default_rng(seed)
    t0 = int(pd.Timestamp(start, tz="America/Montreal").timestamp() // 3600)
    idx = np.arange(t0, t0 + hours + 7)
    rows = []
    for s in (1, 2, 3, 4):
        bikes = rng.integers(0, 6, len(idx))
        docks = rng.integers(0, 6, len(idx))
        rows.append(pd.DataFrame({"our_id": s, "hour_idx": idx, "num_bikes_available": bikes,
                                  "num_bikes_disabled": 0, "num_docks_available": docks, "is_installed": 1,
                                  "is_renting": 1, "is_returning": 1, "capacity": 20}))
    states = pd.concat(rows, ignore_index=True)
    preds = []
    for h in (1, 3, 6):
        for s in (1, 2, 3, 4):
            mu = rng.normal(0, 3, hours)
            preds.append(pd.DataFrame({"station_id": s, "hour_idx": idx[:hours], "year": pd.Timestamp(start).year,
                                       "h": h, "y": rng.normal(0, 3, hours), "l2": mu, "l2_lo": mu - 3, "l2_hi": mu + 3}))
    d, _ = chk.build_pairs(states, pd.concat(preds, ignore_index=True))
    return d


def test_confirmation_scoring_fits_nothing(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the confirmation must not fit anything")
    fits = {}
    for h in (1, 3, 6):
        fits[("empty", h)] = (lambda f: np.full(len(f), 0.2), lambda p: p)
        fits[("full", h)] = (lambda f: np.full(len(f), 0.2), lambda p: p)
        fits[("recal", h)] = (lambda p: p, lambda p: p)
    monkeypatch.setattr(av, "rate_table", boom)
    monkeypatch.setattr(av, "recalibrate", boom)
    res = conf.score_confirmation(fake_pairs("2024-09-02", 72), fits)
    assert set(res) == {"empty_full", "decision", "truck_effect"}
    v = conf.judge(res)
    assert set(v["ranking_order"]["verdicts"]) == {"both_vs_size", "stock_vs_size"}
    assert v["confirmed"] == (v["ranking_order"]["holds"] and v["empty_full_skill"]["holds"])


def test_fit_uses_only_may_to_july_2025():
    jul = fake_pairs("2025-07-27", 96, seed=1)
    aug = fake_pairs("2025-08-04", 96, seed=2)
    flipped = aug.assign(is_empty1=1 - aug.is_empty1, is_full1=1 - aug.is_full1)
    a = conf.fit_2025(pd.concat([jul, aug], ignore_index=True))
    b = conf.fit_2025(pd.concat([jul, flipped], ignore_index=True))
    probe = jul.assign(p_direct=np.linspace(0.01, 0.99, len(jul)))
    for key in a:
        if key[0] == "recal":
            for fa, fb in zip(a[key], b[key]):
                assert np.allclose(fa(probe.p_direct.values), fb(probe.p_direct.values))
        else:
            assert np.allclose(a[key][0](probe), b[key][0](probe))
            assert np.allclose(a[key][1](probe.p_direct.values), b[key][1](probe.p_direct.values))


def test_judge_reads_the_frozen_rule():
    def res(both, stock, empty, full):
        ef = lambda v: {"horizons": {"3": {"headline": {"paired": {"p_cal_vs_p_rate": {"verdict": v}}}}}}
        return {"decision": {"3": {"paired": {"both_vs_size": {"verdict": both}, "stock_vs_size": {"verdict": stock}}}},
                "empty_full": {"empty": ef(empty), "full": ef(full)}}
    assert conf.judge(res("better", "better", "tie", "better"))["confirmed"]
    assert not conf.judge(res("better", "tie", "better", "better"))["confirmed"]
    assert not conf.judge(res("better", "better", "worse", "better"))["confirmed"]
