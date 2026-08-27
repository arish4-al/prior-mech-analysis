#!/usr/bin/env python
"""Localize the pseudo over-perseveration: choice run-length (what copy-last
targets) and act-prior run-length, for real vs base-AK vs copy-last."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from one.api import ONE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import block_analysis_allsplits as ba  # noqa: E402
sys.path.insert(0, str(ROOT / 'scripts'))
from late_choice_stickiness import (  # noqa: E402
    mean_run, quintile_mean_run)

os.environ.setdefault('ACTKERNEL_NB_STEPS', '80')
PREFERRED_EID = '4364a246-f8d7-4ce7-ba23-a098104b96e4'
SPLIT = 'act_block_duringstim_fully_unsplit'
N = int(os.environ.get('N', '200'))


def prior_run(choice):
    pbin = ba._act_prior_binary_from_choice(np.asarray(choice, dtype=float))
    lab = np.isclose(pbin, 0.8).astype(int)
    if len(lab) < 2:
        return float('nan')
    return len(lab) / (int(np.sum(np.diff(lab) != 0)) + 1)


def summ(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return f'{np.median(x):.2f} [{np.percentile(x,10):.2f},{np.percentile(x,90):.2f}]'


def main():
    cache_root = Path.home() / 'Downloads/ONE/alyx.internationalbrainlab.org'
    ba.one = ONE(cache_dir=str(cache_root), mode='local')
    caches = sorted((cache_root / 'manifold' / 'insertion_cache').glob('*.npy'))
    c = None
    for f in caches:
        cc = np.load(f, allow_pickle=True).item()
        if str(cc.get('eid')) == PREFERRED_EID and cc.get('pid'):
            c = cc
            break
    assert c is not None
    eid = c['eid']
    trials_full = c['trials'][ba.saturation_for_split(SPLIT)]
    true_pleft = trials_full['probabilityLeft'].to_numpy().astype(float)
    keep = ~np.isclose(true_pleft, 0.5)
    trials_k = trials_full[keep].reset_index(drop=True)
    ch_real = trials_full['choice'].to_numpy().astype(float)
    pleft_real = true_pleft

    real_choice_run = mean_run(ch_real[keep])
    real_qmr = quintile_mean_run(ch_real, pleft_real)
    real_prior_run = prior_run(ch_real[keep])
    print(f'insertion {eid}  n_biased={int(keep.sum())}')
    print(f'REAL   choice mean_run={real_choice_run:.2f}  '
          f'quintile={np.array2string(real_qmr, precision=2)}  '
          f'prior_run={real_prior_run:.2f}\n')

    fit = ba._null_choice_fit(SPLIT, eid, trials_k)
    theta = fit['params']
    cm = fit.get('choice_model', 'actkernel')
    syn = ba._syn()
    n_pseudo = ba._strat_pseudo_n_trials(trials_k, None)

    for sticky in (False, True):
        out = syn.synthetic_sessions_from_trials(
            trials_k, n=N, eid=str(eid), subject='bwm', params=theta,
            seed=777, fast=True, n_trials=n_pseudo,
            late_sticky=sticky, choice_model=cm)
        ch_mat = np.asarray(out['choice'], dtype=float)
        pl_mat = np.asarray(out['probabilityLeft'], dtype=float)
        c_runs, p_runs = [], []
        for i in range(ch_mat.shape[0]):
            kp = ~np.isclose(pl_mat[i], 0.5)
            c_runs.append(mean_run(ch_mat[i][kp]))
            p_runs.append(prior_run(ch_mat[i][kp]))
        tag = 'copy-last' if sticky else 'base AK '
        print(f'PSEUDO {tag}: choice mean_run={summ(c_runs)}  '
              f'prior_run={summ(p_runs)}  '
              f'(real choice={real_choice_run:.2f}, prior={real_prior_run:.2f})')


if __name__ == '__main__':
    main()
