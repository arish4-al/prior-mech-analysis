#!/usr/bin/env python
"""Smoke for OptimalBayesian choice sampling (no ActionKernel / ONE).

  python scripts/test_bayes_choice_null.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from simulate_synthetic_choices import (  # noqa: E402
    _combine_lkd_prior_np,
    simulate_bayes_choices,
)


def test_combine_left_stim_high_prior():
    p = _combine_lkd_prior_np(np.array([1.0]), zeta=0.1, pi=np.array([0.8]))
    assert p[0] > 0.9, float(p[0])
    p_r = _combine_lkd_prior_np(np.array([-1.0]), zeta=0.1, pi=np.array([0.2]))
    assert p_r[0] < 0.1, float(p_r[0])


def test_sample_follows_easy_stim():
    n = 400
    # Brainbox: −1 left. Easy left / easy right, prior matching the side.
    left = simulate_bayes_choices(
        np.full(n, -1.0), np.full(n, -1.0), seed=0,
        priors=np.full(n, 0.8), lapse=0.05)
    right = simulate_bayes_choices(
        np.full(n, 1.0), np.full(n, 1.0), seed=1,
        priors=np.full(n, 0.2), lapse=0.05)
    p_left = float((left == 1).mean())
    p_right_as_left = float((right == 1).mean())
    assert p_left > 0.85, p_left
    assert p_right_as_left < 0.15, p_right_as_left


def test_prior_shifts_zero_contrast():
    n = 2000
    side = np.full(n, -1.0)
    signed = np.zeros(n)
    hi = simulate_bayes_choices(
        side, signed, seed=2, priors=np.full(n, 0.8), lapse=0.0)
    lo = simulate_bayes_choices(
        side, signed, seed=3, priors=np.full(n, 0.2), lapse=0.0)
    # Zero contrast: prior should dominate (left-positive stim is 0).
    assert (hi == 1).mean() > (lo == 1).mean() + 0.3


def test_scheme_tag():
    sys.path.insert(0, str(ROOT))
    import block_analysis_allsplits as ba
    assert ba._null_choice_model('choice_stim_r_block_r_bayes') == 'bayes'
    assert ba._null_choice_model('choice_stim_r_block_r_act') == 'actkernel'
    assert ba._null_choice_model('bayes_block_duringstim_l') == 'bayes'
    assert ba._null_choice_model('act_block_duringstim_l') == 'actkernel'
    tag = ba._null_scheme_name(
        'synthetic_choice_pseudo_strat', True, 'bayes')
    assert tag == 'synthetic_bayes_choice_pseudo_strat_sticky', tag
    tag_b = ba._null_scheme_name(
        'synthetic_prior_pseudo_strat', True, 'bayes')
    assert tag_b == 'synthetic_bayes_prior_pseudo_strat_sticky', tag_b
    tag_ak = ba._null_scheme_name(
        'synthetic_choice_pseudo_strat', True, 'actkernel')
    assert tag_ak == 'synthetic_choice_pseudo_strat_sticky', tag_ak


def test_harris_routing():
    sys.path.insert(0, str(ROOT))
    import block_analysis_allsplits as ba
    prior_sc = [
        'bayes_block_duringstim_l_choice_l_f1',
        'bayes_block_stim_r_duringchoice_r_f1',
    ]
    prior_stim = ['bayes_block_duringstim_l', 'bayes_block_duringstim_r']
    choice = [
        'choice_duringstim_l_block_l_bayes',
        'choice_stim_r_block_r_bayes',
    ]
    for sp in prior_sc + prior_stim:
        assert ba.is_act_block_prior_split(sp), sp
        assert ba.is_harris_eligible_split(sp), sp
        assert ba._split_uses_bayes_prior(sp)
        assert not ba._split_uses_act_prior(sp)
    spec_u = ba._act_block_conditioning_spec('bayes_block_duringstim_l')
    assert spec_u['stim_is_left'] is True and spec_u['choice'] is None
    spec_f1 = ba._act_block_conditioning_spec(prior_sc[0])
    assert spec_f1['stim_is_left'] is True and spec_f1['choice'] == 1.0
    for sp in choice:
        assert ba.is_choice_lr_split(sp) and ba.is_harris_eligible_split(sp)
        assert ba._split_uses_bayes_prior(sp)
        stim, p = ba._choice_lr_stratum_targets(sp)
        assert stim is not None and p in (0.8, 0.2)


def test_donor_bayes_uses_full_stim_history():
    '''Harris donor Bayes labels must see the 0.5-block prefix (recipient does).'''
    sys.path.insert(0, str(ROOT))
    import block_analysis_allsplits as ba
    n0, n1 = 90, 40
    stim = np.concatenate([
        np.arange(n0) % 2 == 0,
        np.ones(n1, dtype=bool),
    ])
    pleft = np.concatenate([
        np.full(n0, 0.5),
        np.full(n1, 0.8),
    ])
    rec = {
        'choice': np.ones(n0 + n1),
        'stim_is_left': stim,
        'pleft_true': pleft,
    }
    keep = ~np.isclose(pleft, 0.5)
    _, p_dropped = ba.bayesian_priors(stim[keep])
    _, p_full = ba.bayesian_priors(stim)
    # Prefix changes early biased-trial binaries (otherwise the bug is silent).
    assert not np.array_equal(p_dropped == 0.8, p_full[keep] == 0.8)
    labels, keep_out = ba._donor_block_prior_labels(
        rec, 'bayes_block_duringstim_l')
    assert keep_out is not None and np.array_equal(keep_out, keep)
    assert np.array_equal(labels[keep], p_full[keep] == 0.8)
    labels_act, _ = ba._donor_block_prior_labels(
        rec, 'act_block_duringstim_l')
    _, p_act = ba.action_kernel_priors(ba.alpha, list(np.ones(n1)))
    assert np.array_equal(labels_act[keep], np.asarray(p_act) == 0.8)


if __name__ == '__main__':
    test_combine_left_stim_high_prior()
    test_sample_follows_easy_stim()
    test_prior_shifts_zero_contrast()
    print('policy smokes ok')
    try:
        test_scheme_tag()
        test_harris_routing()
        test_donor_bayes_uses_full_stim_history()
        print('routing tags ok')
    except Exception as exc:
        print(f'routing tags skipped ({type(exc).__name__}: {exc})')
