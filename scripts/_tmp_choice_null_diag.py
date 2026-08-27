#!/usr/bin/env python
"""Measure real vs pseudo within-stratum choice L/R imbalance + run structure.

Answers: are choice runs short? does the pseudo null carry the same L/R
imbalance as real (it is conditioned on the same stim x act-prior stratum)?
Both real and pseudo are gated by min5.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from one.api import ONE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import block_analysis_allsplits as ba  # noqa: E402

os.environ.setdefault('ACTKERNEL_NB_STEPS', '80')
PREFERRED_EID = '4364a246-f8d7-4ce7-ba23-a098104b96e4'
SPLITS = [
    'choice_duringstim_r_block_r_act',  # stim_r, act-prior right (p=0.2)
    'choice_duringstim_l_block_l_act',  # stim_l, act-prior left  (p=0.8)
    'choice_stim_r_block_r_act',        # duringchoice window
]
N_ACC = int(os.environ.get('N_ACC', '300'))


def runs(x):
    """Mean run length of a boolean/int sequence (consecutive equal values)."""
    x = np.asarray(x).reshape(-1)
    if len(x) == 0:
        return 0.0
    change = np.flatnonzero(np.diff(x) != 0)
    n_runs = len(change) + 1
    return len(x) / n_runs


def minority_run(ys):
    """Mean run length of the minority label within the sequence order."""
    ys = np.asarray(ys, dtype=bool)
    minlab = ys if ys.sum() <= (~ys).sum() else ~ys
    # runs of True in minlab
    idx = np.flatnonzero(minlab)
    if len(idx) == 0:
        return 0.0
    splits = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
    return float(np.mean([len(s) for s in splits]))


def real_stratum(split, trials):
    tr = trials.copy()
    ch = tr['choice'].to_numpy().astype(float)
    if ba._split_uses_act_prior(split):
        _, pbin = ba.action_kernel_priors(ba.alpha, list(ch))
        tr = tr.copy()
        tr['probabilityLeft'] = np.asarray(pbin, dtype=float)
    stim_side = ba._trials_stim_side(tr)
    pleft = tr['probabilityLeft'].to_numpy().astype(float)
    elig = ba._stratum_mask_stim_pleft(stim_side, pleft, split)
    idx = np.flatnonzero(elig)
    ch_elig = ch[idx]
    ys = ch_elig == 1
    return idx, ys, ch


def main():
    cache_root = Path.home() / 'Downloads/ONE/alyx.internationalbrainlab.org'
    ba.one = ONE(cache_dir=str(cache_root), mode='local')
    caches = sorted((cache_root / 'manifold' / 'insertion_cache').glob('*.npy'))
    picked = None
    for f in caches:
        c = np.load(f, allow_pickle=True).item()
        if str(c.get('eid')) == PREFERRED_EID and c.get('pid'):
            picked = (f, c)
            break
    assert picked, 'preferred insertion not found'
    fpath, c = picked
    eid = c['eid']
    print(f'insertion {fpath.name} eid={eid}\n')

    syn = ba._syn()
    for split in SPLITS:
        sat = ba.saturation_for_split(split)
        trials = c['trials'][sat]
        idx, ys_real, ch_full = real_stratum(split, trials)
        n_elig = len(idx)
        nL = int(ys_real.sum())
        nR = int((~ys_real).sum())
        if nL < 5 or nR < 5:
            print(f'== {split}: real fails min5 (nL={nL} nR={nR}); skip\n')
            continue
        # run length of full choice sequence and within-stratum (stratum order)
        run_full = runs(ch_full)
        run_strat = runs(ch_full[idx])
        minor_real = minority_run(ys_real)
        print(f'== {split}  n_elig={n_elig}')
        print(f'   REAL within-stratum: nL={nL} nR={nR} '
              f'minority_frac={min(nL, nR)/n_elig:.3f}  '
              f'(1/nL+1/nR)={1/nL + 1/nR:.4f}')
        print(f'   REAL choice mean_run: full session={run_full:.2f}  '
              f'within-stratum(order)={run_strat:.2f}  '
              f'minority-run(in stratum)={minor_real:.2f}')

        # pseudo strat + sticky draws (mirror _compute_control_D_actkernel_choice)
        fit = ba._null_choice_fit(split, eid, trials)
        theta = fit['params']
        sim_model = fit.get('sim_model')
        cm = fit.get('choice_model', 'actkernel')
        n_pseudo = ba._strat_pseudo_n_trials(trials, None)  # factor 1 + pad
        rng = np.random.default_rng(0)
        acc_nL, acc_nR, acc_minor = [], [], []
        n_acc = 0
        gen = 0
        seed0 = 12345
        while n_acc < N_ACC and gen < N_ACC * 40:
            out = syn.synthetic_sessions_from_trials(
                trials, n=64, eid=str(eid), subject='bwm', params=theta,
                seed=seed0 + gen, fast=True, n_trials=n_pseudo,
                late_sticky=True, choice_model=cm)
            ch_mat = np.asarray(out['choice'], dtype=float)
            side_mat = np.asarray(out['stim_side'], dtype=float)
            pl_mat = np.asarray(out['probabilityLeft'], dtype=float)
            gen += ch_mat.shape[0]
            for i in range(ch_mat.shape[0]):
                mask = ba._stratum_mask_for_stream(
                    split, side_mat[i], ch_mat[i], pleft_schedule=pl_mat[i])
                y = ba._ys_from_stratum_choices(ch_mat[i], mask, n_elig, rng)
                if y is None:
                    continue
                s = int(np.asarray(y).sum())
                if s < 5 or (n_elig - s) < 5:
                    continue
                acc_nL.append(s)
                acc_nR.append(n_elig - s)
                acc_minor.append(minority_run(y))
                n_acc += 1
                if n_acc >= N_ACC:
                    break
        acc_nL = np.array(acc_nL)
        acc_nR = np.array(acc_nR)
        if len(acc_nL) == 0:
            print('   PSEUDO: 0 accepted draws\n')
            continue
        minfrac = np.minimum(acc_nL, acc_nR) / n_elig
        inv = 1.0 / acc_nL + 1.0 / acc_nR
        acc_rate = n_acc / gen
        print(f'   PSEUDO accepted={len(acc_nL)}/{gen} (acc_rate={acc_rate:.2f})')
        print(f'   PSEUDO within-stratum: nL med={np.median(acc_nL):.0f} '
              f'nR med={np.median(acc_nR):.0f}  '
              f'minority_frac med={np.median(minfrac):.3f} '
              f'[{np.percentile(minfrac, 10):.3f},{np.percentile(minfrac, 90):.3f}]')
        print(f'   PSEUDO (1/nL+1/nR) med={np.median(inv):.4f}  '
              f'vs REAL {1/nL + 1/nR:.4f}  '
              f'(smaller ⇒ narrower null amplitude)')
        print(f'   PSEUDO minority-run med={np.median(acc_minor):.2f}  '
              f'vs REAL {minor_real:.2f}\n')


if __name__ == '__main__':
    main()
