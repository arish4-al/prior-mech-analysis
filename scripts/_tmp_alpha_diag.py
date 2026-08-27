#!/usr/bin/env python
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
from one.api import ONE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import block_analysis_allsplits as ba  # noqa
from late_choice_stickiness import mean_run  # noqa

os.environ.setdefault('ACTKERNEL_NB_STEPS', '80')
EID = '4364a246-f8d7-4ce7-ba23-a098104b96e4'
SPLIT = 'act_block_duringstim_fully_unsplit'


def prior_run_alpha(choice, a):
    """Binarized leaky-prior run length using integration rate a."""
    ch = np.asarray(choice, dtype=float)
    _, pbin = ba.action_kernel_priors(a, list(ch))
    lab = np.isclose(pbin, 0.8).astype(int)
    return len(lab) / (int(np.sum(np.diff(lab) != 0)) + 1)


def main():
    root = Path.home() / 'Downloads/ONE/alyx.internationalbrainlab.org'
    ba.one = ONE(cache_dir=str(root), mode='local')
    for f in sorted((root / 'manifold' / 'insertion_cache').glob('*.npy')):
        cc = np.load(f, allow_pickle=True).item()
        if str(cc.get('eid')) == EID and cc.get('pid'):
            c = cc; break
    trials_full = c['trials'][ba.saturation_for_split(SPLIT)]
    tp = trials_full['probabilityLeft'].to_numpy().astype(float)
    keep = ~np.isclose(tp, 0.5)
    trials_k = trials_full[keep].reset_index(drop=True)
    ch_real = trials_full['choice'].to_numpy().astype(float)[keep]

    fit = ba._null_choice_fit(SPLIT, EID, trials_k)
    theta = np.asarray(fit['params'], dtype=float)
    print(f'label alpha (global) = {ba.alpha}')
    print(f'fitted theta [alpha, zeta, lapse+, lapse-] = '
          f'{np.array2string(theta, precision=4)}')
    print(f'fitted generative alpha = {theta[0]:.4f}  '
          f'(kernel memory ~ {1/theta[0]:.1f} trials vs label 5.0)\n')

    print(f'REAL prior_run @label-alpha(0.2) = {prior_run_alpha(ch_real,0.2):.2f}')
    print(f'REAL prior_run @fitted-alpha({theta[0]:.3f}) = '
          f'{prior_run_alpha(ch_real, theta[0]):.2f}')

    syn = ba._syn()
    n_pseudo = ba._strat_pseudo_n_trials(trials_k, None)
    out = syn.synthetic_sessions_from_trials(
        trials_k, n=200, eid=EID, subject='bwm', params=theta, seed=777,
        fast=True, n_trials=n_pseudo, late_sticky=True,
        choice_model=fit.get('choice_model', 'actkernel'))
    chm = np.asarray(out['choice'], dtype=float)
    plm = np.asarray(out['probabilityLeft'], dtype=float)
    pr02, prfit = [], []
    for i in range(chm.shape[0]):
        kp = ~np.isclose(plm[i], 0.5)
        pr02.append(prior_run_alpha(chm[i][kp], 0.2))
        prfit.append(prior_run_alpha(chm[i][kp], theta[0]))
    print(f'PSEUDO prior_run @label-alpha(0.2)  med={np.median(pr02):.2f}')
    print(f'PSEUDO prior_run @fitted-alpha({theta[0]:.3f}) med={np.median(prfit):.2f}')


if __name__ == '__main__':
    main()
