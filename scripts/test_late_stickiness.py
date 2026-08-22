#!/usr/bin/env python
"""Smoke for per-session quintile mean_run matching (no torch / ActionKernel).

Signed mix: copy-last when the target μ is higher than this sequence,
break-repeat when it is lower. Geometric p_repeat = 1 − 1/μ.

  python scripts/test_late_stickiness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from late_choice_stickiness import (  # noqa: E402
    apply_late_stickiness,
    extra_repeat_probs,
    quintile_mean_run,
)


def _markov_choices(n, p_repeat, rng, start=1.0):
    ch = np.empty(n, dtype=float)
    ch[0] = start
    for t in range(1, n):
        ch[t] = ch[t - 1] if rng.random() < p_repeat else -ch[t - 1]
    return ch


def _pleft_session(n, n_half=90):
    p = np.full(n, 0.8)
    p[:n_half] = 0.5
    blk = n_half
    side = 0.8
    while blk < n:
        p[blk:blk + 60] = side
        blk += 60
        side = 0.2 if side > 0.5 else 0.8
    return p


def test_target_below_lowers_mean_run():
    rng = np.random.default_rng(0)
    n = 90 + 5 * 80
    n_sess = 60
    target = np.full(5, 2.2)
    err_ak, err_st = [], []
    for _ in range(n_sess):
        ch = _markov_choices(n, 0.75, rng)
        pleft = _pleft_session(n)
        mu0 = quintile_mean_run(ch, pleft)
        out = apply_late_stickiness(ch, pleft, rng, target_mean_run=target)
        mu1 = quintile_mean_run(out, pleft)
        err_ak.append(np.nanmean(np.abs(mu0 - target)))
        err_st.append(np.nanmean(np.abs(mu1 - target)))
    e0 = float(np.nanmedian(err_ak))
    e1 = float(np.nanmedian(err_st))
    assert e1 < e0 - 0.3, (e0, e1)
    rho = extra_repeat_probs(ch, pleft, target_mean_run=target)
    assert np.all(rho <= 0.0), rho


def test_matches_higher_targets():
    rng = np.random.default_rng(1)
    p_rep = 0.5  # μ ≈ 2
    n = 90 + 5 * 80
    n_sess = 80
    target = np.array([2.0, 2.2, 2.6, 3.0, 2.9])
    err_ak = []
    err_st = []
    d_ak = []
    d_st = []
    align_shift = []
    for _ in range(n_sess):
        ch = _markov_choices(n, p_rep, rng)
        pleft = _pleft_session(n)
        mu0 = quintile_mean_run(ch, pleft)
        out = apply_late_stickiness(ch, pleft, rng, target_mean_run=target)
        mu1 = quintile_mean_run(out, pleft)
        err_ak.append(np.nanmean(np.abs(mu0 - target)))
        err_st.append(np.nanmean(np.abs(mu1 - target)))
        d_ak.append(mu0[3] - mu0[0])
        d_st.append(mu1[3] - mu1[0])
        block = np.where(np.isclose(pleft, 0.8), 1.0, -1.0)
        nobias = ~np.isclose(pleft, 0.5)
        a0 = float(np.mean(ch[nobias] == block[nobias]))
        a1 = float(np.mean(out[nobias] == block[nobias]))
        align_shift.append(a1 - a0)
    e0 = float(np.nanmedian(err_ak))
    e1 = float(np.nanmedian(err_st))
    d0 = float(np.nanmedian(d_ak))
    d1 = float(np.nanmedian(d_st))
    da = float(np.nanmedian(align_shift))
    assert e1 < e0 - 0.15, (e0, e1)
    assert d1 > d0 + 0.4, (d0, d1)
    assert abs(da) < 0.05, da


def test_q1_rises_when_target_higher():
    rng = np.random.default_rng(2)
    n = 90 + 5 * 80
    ch = _markov_choices(n, 0.5, rng)
    pleft = _pleft_session(n)
    mu0 = quintile_mean_run(ch, pleft)
    target = np.full(5, 3.0)
    out = apply_late_stickiness(ch, pleft, rng, target_mean_run=target)
    mu1 = quintile_mean_run(out, pleft)
    assert mu1[0] > mu0[0] + 0.2, (mu0[0], mu1[0])
    rho = extra_repeat_probs(ch, pleft, target_mean_run=target)
    assert rho[0] > 0.0


def test_timeouts_untouched():
    rng = np.random.default_rng(3)
    ch = np.array([1.0, 0.0, 1.0, -1.0, 0.0, -1.0])
    pleft = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
    out = apply_late_stickiness(ch, pleft, rng, target_mean_run=(3.0,) * 5)
    assert np.array_equal(out, ch)


if __name__ == '__main__':
    test_target_below_lowers_mean_run()
    test_matches_higher_targets()
    test_q1_rises_when_target_higher()
    test_timeouts_untouched()
    print('late_stickiness smoke OK')
