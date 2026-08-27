#!/usr/bin/env python
"""Cohort-wide prior-run heterogeneity (act_block_duringstim_fully_unsplit).

Part 1 (all donor-bank sessions): distribution of each session's OWN real
act-prior run length + minority_frac -> quantifies how mis-calibrated a
population-transplant (Harris) null is per session.

Part 2 (local full-trials sessions): real prior_run vs AK-sim prior_run,
to check whether the AK over-persistence generalizes beyond the one outlier.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from one.api import ONE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import block_analysis_allsplits as ba  # noqa: E402

os.environ.setdefault('ACTKERNEL_NB_STEPS', '80')
SPLIT = 'act_block_duringstim_fully_unsplit'
N_AK = int(os.environ.get('N_AK', '100'))


def runs(x):
    x = np.asarray(x).reshape(-1)
    if len(x) < 2:
        return float('nan')
    return len(x) / (int(np.sum(np.diff(x) != 0)) + 1)


def qsumm(x, name):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    print(f'   {name}: n={len(x)}  median={np.median(x):.2f}  '
          f'IQR=[{np.percentile(x,25):.2f},{np.percentile(x,75):.2f}]  '
          f'10/90=[{np.percentile(x,10):.2f},{np.percentile(x,90):.2f}]  '
          f'range=[{np.min(x):.2f},{np.max(x):.2f}]')
    return x


def main():
    root = Path.home() / 'Downloads/ONE/alyx.internationalbrainlab.org'
    ba.one = ONE(cache_dir=str(root), mode='local')
    bank = ba.load_choice_donor_bank()
    print(f'donor bank: {len(bank)} eids\n')

    print('== PART 1: cohort real prior-run heterogeneity (fully-unsplit) ==')
    runs_all, minf_all, n_all = [], [], []
    for e, rec in bank.items():
        labels, keep = ba._donor_block_prior_labels(rec, SPLIT)
        if labels is None:
            continue
        mask = ba._donor_block_conditioning_mask(rec, SPLIT, keep=keep)
        if mask is None:
            continue
        lab = labels[mask]
        if len(lab) < 2 * ba.min_trials_per_side:
            continue
        s = int(lab.sum())
        if s < ba.min_trials_per_side or (len(lab) - s) < ba.min_trials_per_side:
            continue
        runs_all.append(runs(lab))
        minf_all.append(min(s, len(lab) - s) / len(lab))
        n_all.append(len(lab))
    r = qsumm(runs_all, 'own prior_run')
    qsumm(minf_all, 'own minority_frac')
    qsumm(n_all, 'n biased trials')

    pop_med = float(np.median(r))
    lo, hi = np.percentile(r, 10), np.percentile(r, 90)
    # A population-transplant (Harris) null draws its run from this population,
    # ~independent of the recipient. Recipient mis-calibration = own vs pop.
    below = np.mean(r < pop_med) * 100
    tail_lo = np.mean(r < lo) * 100
    tail_hi = np.mean(r > hi) * 100
    log2r = np.log2(r / pop_med)
    print(f'\n   population median prior_run = {pop_med:.2f} '
          f'(this is ~what Harris imposes on every recipient)')
    print(f'   sessions with own_run < pop_median (Harris over-clusters '
          f'-> conservative): {below:.0f}%')
    print(f'   sessions in population tails: <10th pct={tail_lo:.0f}%  '
          f'>90th pct={tail_hi:.0f}%  (strongly mis-nulled either way)')
    print(f'   |log2(own/pop_median)| median={np.median(np.abs(log2r)):.2f}  '
          f'90th pct={np.percentile(np.abs(log2r),90):.2f}  '
          f'(0=perfect; 1=2x off)')
    frac_2x = np.mean(np.abs(log2r) >= 1.0) * 100
    print(f'   sessions >=2x off from pop median (either way): {frac_2x:.0f}%')
    sys.stdout.flush()

    print('\n== PART 2: real vs AK-sim prior_run (local full-trials eids) ==')
    seen = set()
    syn = ba._syn()
    for f in sorted((root / 'manifold' / 'insertion_cache').glob('*.npy')):
        c = np.load(f, allow_pickle=True).item()
        eid = str(c.get('eid'))
        if not eid or eid in seen or not c.get('pid'):
            continue
        seen.add(eid)
        try:
            trials = c['trials'][ba.saturation_for_split(SPLIT)]
            tp = trials['probabilityLeft'].to_numpy().astype(float)
            keep = ~np.isclose(tp, 0.5)
            trials_k = trials[keep].reset_index(drop=True)
            ch_real = trials['choice'].to_numpy().astype(float)[keep]
            _, pbin = ba.action_kernel_priors(ba.alpha, list(ch_real))
            real_run = runs(np.isclose(pbin, 0.8).astype(int))

            fit = ba._null_choice_fit(SPLIT, eid, trials_k)
            theta = fit['params']
            n_pseudo = ba._strat_pseudo_n_trials(trials_k, None)
            out = syn.synthetic_sessions_from_trials(
                trials_k, n=N_AK, eid=eid, subject='bwm', params=theta,
                seed=123, fast=True, n_trials=n_pseudo, late_sticky=True,
                choice_model=fit.get('choice_model', 'actkernel'))
            chm = np.asarray(out['choice'], dtype=float)
            plm = np.asarray(out['probabilityLeft'], dtype=float)
            ak_runs = []
            for i in range(chm.shape[0]):
                kp = ~np.isclose(plm[i], 0.5)
                _, pb = ba.action_kernel_priors(ba.alpha, list(chm[i][kp]))
                ak_runs.append(runs(np.isclose(pb, 0.8).astype(int)))
            ak_med = float(np.nanmedian(ak_runs))
            pctile = 100.0 * np.mean(r <= real_run)
            print(f'   {eid[:8]}: real_run={real_run:.2f} (pop pctile {pctile:.0f}%)  '
                  f'AK-sim_run={ak_med:.2f}  '
                  f'AK/real={ak_med/real_run:.2f}  alpha={theta[0]:.3f}')
            sys.stdout.flush()
        except Exception as exc:
            print(f'   {eid[:8]}: FAILED ({type(exc).__name__}: {exc})')
            sys.stdout.flush()


if __name__ == '__main__':
    main()
