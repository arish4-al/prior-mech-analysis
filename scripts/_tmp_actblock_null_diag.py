#!/usr/bin/env python
"""Same metrics as the choice diagnostic, but for act_block prior L vs R.

For prior splits the LABEL is the act-binary prior (a leaky function of choice
with ~block-length runs), so we expect long label runs (drift alignment) and
we check whether the pseudo (strat+sticky) null reproduces the real prior-L/R
imbalance + run structure. Both real and pseudo are gated by min5.
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
    'act_block_duringstim_l_choice_l_f1',    # stratified (stim_l x choice_l)
    'act_block_duringstim_r_choice_r_f1',    # stratified (stim_r x choice_r)
    'act_block_duringstim_fully_unsplit',    # no stratum (all biased)
]
N_ACC = int(os.environ.get('N_ACC', '300'))


def runs(x):
    x = np.asarray(x).reshape(-1)
    if len(x) == 0:
        return 0.0
    change = np.flatnonzero(np.diff(x) != 0)
    return len(x) / (len(change) + 1)


def minority_run(ys):
    ys = np.asarray(ys, dtype=bool)
    minlab = ys if ys.sum() <= (~ys).sum() else ~ys
    idx = np.flatnonzero(minlab)
    if len(idx) == 0:
        return 0.0
    splits = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
    return float(np.mean([len(s) for s in splits]))


def real_elig(split, trials):
    """Replicate get_d_vars act_block: drop true 0.5, act-binary prior,
    stim x choice conditioning."""
    spec = ba._act_block_conditioning_spec(split)
    ch = trials['choice'].to_numpy().astype(float)
    cr = trials['contrastRight'].to_numpy().astype(float)
    cl = trials['contrastLeft'].to_numpy().astype(float)
    true_pleft = trials['probabilityLeft'].to_numpy().astype(float)
    keep = ~np.isclose(true_pleft, 0.5)
    ch_k, cl_k, cr_k = ch[keep], cl[keep], cr[keep]
    stim_is_left_k = np.isnan(cr_k)
    pbin_k = ba._act_prior_binary_from_choice(ch_k)
    prior_l_k = np.isclose(pbin_k, 0.8)
    mask = ba._block_conditioning_mask(
        spec, stim_is_left=stim_is_left_k, choice=ch_k,
        contrast_left=cl_k, contrast_right=cr_k)
    elig_idx = np.flatnonzero(mask)
    labels = prior_l_k[elig_idx]
    return elig_idx, labels, keep


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
        trials_full = c['trials'][sat]
        elig_idx, labels_real, keep = real_elig(split, trials_full)
        # 0.5-dropped trials, as passed to the null functions
        trials_k = trials_full[keep].reset_index(drop=True)
        n_elig = len(elig_idx)
        nL = int(labels_real.sum())       # prior-left
        nR = int((~labels_real).sum())    # prior-right
        if nL < 5 or nR < 5:
            print(f'== {split}: real fails min5 (priorL={nL} priorR={nR}); skip\n')
            continue
        run_lab = runs(labels_real)
        minor_real = minority_run(labels_real)
        print(f'== {split}  n_elig={n_elig}')
        print(f'   REAL prior L/R: nL={nL} nR={nR} '
              f'minority_frac={min(nL, nR)/n_elig:.3f}  '
              f'(1/nL+1/nR)={1/nL + 1/nR:.4f}')
        print(f'   REAL prior-label mean_run(in elig order)={run_lab:.2f}  '
              f'minority-run={minor_real:.2f}')

        fit = ba._null_choice_fit(split, eid, trials_k)
        theta = fit['params']
        cm = fit.get('choice_model', 'actkernel')
        n_pseudo = ba._strat_pseudo_n_trials(trials_k, None)
        rng = np.random.default_rng(0)
        acc_nL, acc_run, acc_minor = [], [], []
        n_acc, gen = 0, 0
        seed0 = 12345
        while n_acc < N_ACC and gen < N_ACC * 60:
            out = syn.synthetic_sessions_from_trials(
                trials_k, n=64, eid=str(eid), subject='bwm', params=theta,
                seed=seed0 + gen, fast=True, n_trials=n_pseudo,
                late_sticky=True, choice_model=cm)
            ch_mat = np.asarray(out['choice'], dtype=float)
            side_mat = np.asarray(out['stim_side'], dtype=float)
            signed_mat = np.asarray(out['signed_contrast'], dtype=float)
            pl_mat = np.asarray(out['probabilityLeft'], dtype=float)
            gen += ch_mat.shape[0]
            for i in range(ch_mat.shape[0]):
                keepp = ~np.isclose(pl_mat[i], 0.5)
                ch_kk = ch_mat[i][keepp]
                side_kk = side_mat[i][keepp]
                signed_kk = signed_mat[i][keepp]
                prior_l = ba._block_null_prior_left(split, ch_kk, side_kk)
                mask = ba._act_block_stream_mask(
                    split, side_kk, ch_kk, signed_contrast=signed_kk)
                ys = ba._ys_from_stratum_labels(prior_l, mask, n_elig, rng)
                if ys is None:
                    continue
                s = int(np.asarray(ys).sum())
                if s < 5 or (n_elig - s) < 5:
                    continue
                acc_nL.append(s)
                acc_run.append(runs(ys))
                acc_minor.append(minority_run(ys))
                n_acc += 1
                if n_acc >= N_ACC:
                    break
        acc_nL = np.array(acc_nL)
        if len(acc_nL) == 0:
            print(f'   PSEUDO: 0 accepted draws in {gen} gens\n')
            continue
        acc_nR = n_elig - acc_nL
        minfrac = np.minimum(acc_nL, acc_nR) / n_elig
        inv = 1.0 / acc_nL + 1.0 / acc_nR
        print(f'   PSEUDO accepted={len(acc_nL)}/{gen} '
              f'(acc_rate={n_acc/gen:.2f})')
        print(f'   PSEUDO prior L/R: nL med={np.median(acc_nL):.0f} '
              f'nR med={np.median(acc_nR):.0f}  '
              f'minority_frac med={np.median(minfrac):.3f} '
              f'[{np.percentile(minfrac,10):.3f},{np.percentile(minfrac,90):.3f}]')
        print(f'   PSEUDO (1/nL+1/nR) med={np.median(inv):.4f}  '
              f'vs REAL {1/nL + 1/nR:.4f}')
        print(f'   PSEUDO prior-label mean_run med={np.median(acc_run):.2f}  '
              f'vs REAL {run_lab:.2f}   '
              f'minority-run med={np.median(acc_minor):.2f} vs REAL {minor_real:.2f}\n')


if __name__ == '__main__':
    main()
