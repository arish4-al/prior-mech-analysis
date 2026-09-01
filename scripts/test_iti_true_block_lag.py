#!/usr/bin/env python
"""ITI prior timing: true-block lag, causal act kernel, causal Bayes.

  python scripts/test_iti_true_block_lag.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import block_analysis_allsplits as ba  # noqa: E402


def test_split_gate():
    assert ba._is_true_block_iti_split('block_only')
    assert ba._is_true_block_iti_split('block_only_0.0')
    assert not ba._is_true_block_iti_split('act_block_only')
    assert not ba._is_true_block_iti_split('bayes_block_only')
    assert not ba._is_true_block_iti_split('block_duringstim_l')
    assert ba.is_harris_eligible_split('block_only')
    assert ba.is_harris_eligible_split('act_block_only')
    assert ba.is_harris_eligible_split('bayes_block_only')
    for name in ('block_only', 'act_block_only', 'bayes_block_only'):
        assert ba._is_iti_prior_only_split(name)
    assert not ba._is_iti_prior_only_split('act_block_duringstim_l')
    assert not ba._is_iti_prior_only_split('block_duringstim_l')


def test_true_block_lag_on_trials():
    df = pd.DataFrame({
        'probabilityLeft': [0.8, 0.8, 0.2, 0.2],
        'choice': [1, 1, -1, -1],
        'stimOn_times': [1.0, 2.0, 3.0, 4.0],
    })
    out = ba._apply_true_block_iti_previous_trial_label(df)
    assert len(out) == 3
    np.testing.assert_array_equal(
        np.asarray(out['probabilityLeft'], dtype=float),
        np.array([0.8, 0.8, 0.2]))
    # Alignment events stay trial t (drop trial 0 only).
    np.testing.assert_array_equal(
        np.asarray(out['stimOn_times'], dtype=float),
        np.array([2.0, 3.0, 4.0]))


def test_true_block_lag_at_switch():
    # Last trial of old block (0.8) then first of new block (0.2):
    # ITI before the 0.2 trial must still be labeled 0.8.
    df = pd.DataFrame({
        'probabilityLeft': [0.8, 0.2],
        'choice': [1, -1],
        'stimOn_times': [10.0, 11.0],
    })
    out = ba._apply_true_block_iti_previous_trial_label(df)
    assert len(out) == 1
    assert float(out['probabilityLeft'].iloc[0]) == 0.8
    assert float(out['stimOn_times'].iloc[0]) == 11.0


def test_donor_lag():
    pleft = np.array([0.5, 0.8, 0.8, 0.2])
    keep = ~np.isclose(pleft, 0.5)
    labels, keep2 = ba._lag_true_block_iti_donor_labels(pleft, keep)
    assert keep2[0] is np.False_ or keep2[0] == False
    assert keep2[1] == False  # first biased trial dropped
    assert bool(keep2[2]) and bool(keep2[3])
    # remaining trials 2,3 get pLeft of 1,2 → 0.8, 0.8
    assert labels[2] and labels[3]


def test_act_kernel_includes_previous_action_not_current():
    actions = [1, 1, -1]
    cont, binary = ba.action_kernel_priors(0.2, actions)
    assert cont[0] == 0.5
    assert abs(cont[1] - (0.2 * 1 + 0.8 * 0.5)) < 1e-12
    # priors[2] uses actions[0] and [1], not actions[2]
    expected = 0.2 * 1 + 0.8 * cont[1]
    assert abs(cont[2] - expected) < 1e-12
    assert len(cont) == 3


def test_bayes_includes_previous_stim_not_current():
    # Needs a long run of one side (min block length 20; journal smoke: 80 left → ~0.77).
    stim = np.ones(82, dtype=bool)
    cont, _ = ba.bayesian_priors(stim)
    assert abs(cont[0] - 0.5) < 1e-6
    assert cont[80] > 0.7
    stim2 = stim.copy()
    stim2[80] = False
    cont2, _ = ba.bayesian_priors(stim2)
    assert abs(cont[80] - cont2[80]) < 1e-12
    assert abs(cont[81] - cont2[81]) > 1e-6


def test_lag_resets_index():
    df = pd.DataFrame({
        'probabilityLeft': [0.8, 0.2, 0.2],
        'choice': [1, -1, -1],
        'stimOn_times': [1.0, 2.0, 3.0],
    }, index=[10, 11, 12])
    out = ba._apply_true_block_iti_previous_trial_label(df)
    np.testing.assert_array_equal(out.index.to_numpy(), np.array([0, 1]))


def test_null_labels_length():
    ntr = 40
    dx = np.zeros((ntr, 2))
    ys = ba._null_labels('block_only', ntr, dx)
    assert ys.shape == (ntr,)
    assert ys.dtype == bool
    ys_act = ba._null_labels('act_block_only', ntr, dx)
    assert ys_act.shape == (ntr,)
    ys_bayes = ba._null_labels('bayes_block_only', ntr, dx)
    assert ys_bayes.shape == (ntr,)


def _visp_atlas_id():
    ids = np.asarray(ba.br.id[ba.br.acronym == 'VISp']).reshape(-1)
    if len(ids) == 0:
        raise RuntimeError('no VISp atlas id')
    return int(ids[0])


def _synthetic_trials(n_half=25, n_unbiased=20, t0=10.0, dt=2.0):
    '''Calendar session: unbiased 0.5, then left block, then right block.'''
    n_b = n_half * 2
    n = n_unbiased + n_b
    pleft = np.concatenate([
        np.full(n_unbiased, 0.5),
        np.full(n_half, 0.8),
        np.full(n_half, 0.2),
    ])
    # Stim matches block side on biased trials; mixed in 0.5.
    stim_left = np.concatenate([
        np.arange(n_unbiased) % 2 == 0,
        np.ones(n_half, dtype=bool),
        np.zeros(n_half, dtype=bool),
    ])
    choice = np.where(stim_left, 1.0, -1.0)
    cl = np.where(stim_left, 1.0, np.nan)
    cr = np.where(stim_left, np.nan, 1.0)
    stim_on = t0 + dt * np.arange(n)
    return pd.DataFrame({
        'probabilityLeft': pleft,
        'choice': choice,
        'contrastLeft': cl,
        'contrastRight': cr,
        'stimOn_times': stim_on,
        'firstMovement_times': stim_on + 0.2,
        'feedbackType': np.ones(n),
    })


def _synthetic_cache(trials, eid='eidA', n_clus=4, rng=None):
    rng = np.random.default_rng(0 if rng is None else rng)
    atlas_id = _visp_atlas_id()
    n = len(trials)
    t_max = float(trials['stimOn_times'].max()) + 1.0
    n_spk = 400
    spikes = {
        'times': np.sort(rng.uniform(0.0, t_max, n_spk)),
        'clusters': rng.integers(0, n_clus, n_spk),
    }
    clusters = {
        'cluster_id': np.arange(n_clus),
        'atlas_id': np.full(n_clus, atlas_id),
    }
    return {
        'pid': 'fake-pid',
        'eid': eid,
        'probe': 'probe00',
        'spikes': spikes,
        'clusters': clusters,
        'trials': {'saturation_stim_plus04': trials.copy()},
    }


def test_get_d_vars_iti_three_splits_tiny():
    '''One fake insertion, nrand=3, default (pseudo-block) null.'''
    trials = _synthetic_trials()
    cached = _synthetic_cache(trials)
    n_unb, n_half = 20, 25
    # After 0.5 drop: 50 biased. True-block ITI drops first → 49.
    # Lagged labels: 24 remaining 0.8-on-0.8, plus first 0.2 labeled 0.8, then 24 of 0.2.
    expected_l, expected_r = n_half, n_half - 1
    nrand = 3
    for split in ('block_only', 'act_block_only', 'bayes_block_only'):
        D = ba.get_d_vars(
            split, cached['pid'], control=True, nrand=nrand, cached=cached)
        assert isinstance(D, dict), split
        assert 'D' in D and len(D['D']) >= 1, split
        reg = next(iter(D['D']))
        n_stored = len(D['D'][reg]['d_eucs'])
        assert n_stored == nrand + 1, (split, n_stored)
        assert np.isfinite(np.asarray(D['D'][reg]['d_eucs'][0])).any(), split
        print(f'  get_d_vars {split}: n_regs={len(D["D"])} n_curves={n_stored}')
        if split == 'block_only':
            # Recompute lagged labels on the same 0.5-dropped table.
            biased = trials[trials['probabilityLeft'] != 0.5].reset_index(drop=True)
            lagged = ba._apply_true_block_iti_previous_trial_label(biased)
            n_l = int(np.isclose(lagged['probabilityLeft'], 0.8).sum())
            n_r = int(np.isclose(lagged['probabilityLeft'], 0.2).sum())
            assert n_l == expected_l and n_r == expected_r, (n_l, n_r)


def test_harris_block_only_tiny():
    '''Harris unique-null on fake recipient + 2 donors.'''
    rec_trials = _synthetic_trials(n_half=25)
    cached = _synthetic_cache(rec_trials, eid='eidA')
    donors = {}
    for eid in ('eidB', 'eidC'):
        dtr = _synthetic_trials(n_half=30)
        donors[eid] = {
            'choice': dtr['choice'].to_numpy(dtype=float),
            'stim_is_left': np.isnan(dtr['contrastRight'].astype(float)),
            'pleft_true': dtr['probabilityLeft'].to_numpy(dtype=float),
            'contrast_left': dtr['contrastLeft'].to_numpy(dtype=float),
            'contrast_right': dtr['contrastRight'].to_numpy(dtype=float),
        }
    labels, keep = ba._donor_block_prior_labels(donors['eidB'], 'block_only')
    assert keep is not None and int(keep.sum()) == 59  # 60 biased − first
    nrand = 3
    D = ba.get_d_vars(
        'block_only', cached['pid'], control=True, nrand=nrand,
        cached=cached, donor_bank=donors, session_shuffle_null=True)
    assert D.get('null_scheme') == 'harris_session_permutation_unique'
    assert D.get('harris_n_stratum_donors') == 2
    reg = next(iter(D['D']))
    n_stored = max(len(D['D'][reg]['d_eucs']) - 1, 0)
    assert n_stored == D.get('harris_n_unique_nulls')
    assert n_stored >= 1
    print(f'  harris block_only: donors={D["harris_n_stratum_donors"]} '
          f'unique={n_stored}')


def test_iti_pseudosession_labels_per_prior():
    '''True = lagged pLeft; Bayes = stim history; act = choices. No stratum.'''
    pleft = np.concatenate([np.full(10, 0.5), np.full(15, 0.8), np.full(15, 0.2)])
    # Stim follows the opposite of the block so Bayes ≠ true-block after drop.
    stim_side = np.where(pleft > 0.5, 1.0, -1.0)
    stim_side[:10] = np.where(np.arange(10) % 2 == 0, -1.0, 1.0)
    # Choices follow the block (so act ≈ true after drop, ≠ Bayes).
    choice = np.where(pleft > 0.5, 1.0, -1.0)
    choice[:10] = 1.0

    ys_true = ba._iti_pseudosession_prior_left('block_only', pleft)
    assert ys_true is not None and ys_true.shape == (29,)  # 30 biased − first
    # First remaining biased trial dropped; next 14 of 0.8 + first 0.2 labeled 0.8.
    assert int(ys_true.sum()) == 15

    ys_bayes = ba._iti_pseudosession_prior_left(
        'bayes_block_only', pleft, stim_side=stim_side)
    assert ys_bayes is not None and ys_bayes.shape == (30,)
    ys_act = ba._iti_pseudosession_prior_left(
        'act_block_only', pleft, choice=choice)
    assert ys_act is not None and ys_act.shape == (30,)
    # Three generative labels must not collapse to one another.
    assert not np.array_equal(ys_true, ys_bayes[:29])
    assert not np.array_equal(ys_act, ys_bayes)


def test_get_d_vars_iti_pseudosession_three_splits():
    '''Unconstrained ITI: block + Bayes need no AK; act uses a fake choice model.'''
    trials = _synthetic_trials()
    cached = _synthetic_cache(trials)
    nrand = 3

    class _FakeSyn:
        def synthetic_sessions_from_trials(self, trials_df, n=3, **kw):
            n_tr = int(kw.get('n_trials') or len(trials_df))
            rng = np.random.default_rng(1)
            pleft = np.stack([
                ba.generate_pseudo_blocks(n_tr) for _ in range(int(n))])
            side = np.where(rng.random(pleft.shape) < pleft, -1.0, 1.0)
            choice = np.where(side < 0, 1.0, -1.0)
            return dict(
                choice=choice, stim_side=side,
                signed_contrast=np.abs(side), probabilityLeft=pleft)

    orig_syn = ba._syn
    orig_fit = ba._null_choice_fit
    ba._syn = lambda: _FakeSyn()
    ba._null_choice_fit = lambda *a, **k: {
        'eid': 'eidA', 'params': np.array([0.2, 0.1, 0.05, 0.05]),
        'mode': 'fake', 'choice_model': 'actkernel', 'sim_model': None,
    }
    try:
        expected = {
            'block_only': 'pseudo_block_pseudosession',
            'bayes_block_only': 'synthetic_bayes_prior_pseudosession',
            'act_block_only': 'synthetic_prior_pseudosession',
        }
        for split, scheme in expected.items():
            D = ba.get_d_vars(
                split, cached['pid'], control=True, nrand=nrand,
                cached=cached, actkernel_null_mode='unconstrained')
            assert D.get('null_scheme') == scheme, (split, D.get('null_scheme'))
            assert D.get('actkernel_null_mode') == 'unconstrained', split
            if split == 'act_block_only':
                assert D.get('null_choice_model') == 'actkernel'
            else:
                assert D.get('null_choice_model') == 'none'
            reg = next(iter(D['D']))
            n_stored = len(D['D'][reg]['d_eucs'])
            assert n_stored == nrand + 1, (split, n_stored)
            print(f'  unconstrained {split}: scheme={scheme} n_curves={n_stored}')
        dummy_b = np.zeros((10, 2, 3))
        dummy_acs = np.array(['VISp', 'VISp'])
        try:
            ba._compute_control_D_actkernel_block(
                dummy_b, dummy_acs, dummy_acs,
                np.array([True] * 5 + [False] * 5),
                np.ones(10, dtype=bool), np.zeros(10, dtype=bool),
                10, 2, 'act_block_duringstim_l',
                trials=None, elig_idx=np.arange(10), eid='x',
                actkernel_null_mode='unconstrained')
            raise AssertionError('during-trial unconstrained should raise')
        except ValueError as exc:
            assert 'ITI-only' in str(exc)
    finally:
        ba._syn = orig_syn
        ba._null_choice_fit = orig_fit


if __name__ == '__main__':
    test_split_gate()
    test_true_block_lag_on_trials()
    test_true_block_lag_at_switch()
    test_lag_resets_index()
    test_donor_lag()
    test_null_labels_length()
    test_act_kernel_includes_previous_action_not_current()
    test_bayes_includes_previous_stim_not_current()
    test_iti_pseudosession_labels_per_prior()
    print('unit ok — running get_d_vars smoke')
    test_get_d_vars_iti_three_splits_tiny()
    test_harris_block_only_tiny()
    test_get_d_vars_iti_pseudosession_three_splits()
    print('SMOKE PASSED')

