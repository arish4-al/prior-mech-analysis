#!/usr/bin/env python
"""Harris (donor-transplant) null stats for act_block prior L vs R.

Checks: (a) do Harris null labels match the recipient's real prior run/imbalance
better than the AK pseudo? (b) how heterogeneous are the donors (each = a
different mouse/session)? A valid transplant null wants the recipient's real
label stats to sit inside the donor distribution, not in its tail.
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
PREFERRED_EID = '4364a246-f8d7-4ce7-ba23-a098104b96e4'
SPLITS = [
    'act_block_duringstim_fully_unsplit',
    'act_block_duringstim_l_choice_l_f1',
    'act_block_duringstim_r_choice_r_f1',
]
N_DRAW = int(os.environ.get('N_DRAW', '400'))


def runs(x):
    x = np.asarray(x).reshape(-1)
    if len(x) < 2:
        return float('nan')
    return len(x) / (int(np.sum(np.diff(x) != 0)) + 1)


def real_elig(split, trials):
    spec = ba._act_block_conditioning_spec(split)
    ch = trials['choice'].to_numpy().astype(float)
    cr = trials['contrastRight'].to_numpy().astype(float)
    cl = trials['contrastLeft'].to_numpy().astype(float)
    keep = ~np.isclose(trials['probabilityLeft'].to_numpy().astype(float), 0.5)
    ch_k, cl_k, cr_k = ch[keep], cl[keep], cr[keep]
    stim_is_left_k = np.isnan(cr_k)
    pbin_k = ba._act_prior_binary_from_choice(ch_k)
    prior_l_k = np.isclose(pbin_k, 0.8)
    mask = ba._block_conditioning_mask(
        spec, stim_is_left=stim_is_left_k, choice=ch_k,
        contrast_left=cl_k, contrast_right=cr_k)
    elig_idx = np.flatnonzero(mask)
    return elig_idx, prior_l_k[elig_idx]


def pct(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 'n/a'
    return f'{np.median(x):.2f} [{np.percentile(x,10):.2f},{np.percentile(x,90):.2f}]'


def main():
    root = Path.home() / 'Downloads/ONE/alyx.internationalbrainlab.org'
    ba.one = ONE(cache_dir=str(root), mode='local')
    bank = ba.load_choice_donor_bank()
    print(f'donor bank: {len(bank)} eids\n')

    c = None
    for f in sorted((root / 'manifold' / 'insertion_cache').glob('*.npy')):
        cc = np.load(f, allow_pickle=True).item()
        if str(cc.get('eid')) == PREFERRED_EID and cc.get('pid'):
            c = cc
            break
    assert c is not None
    eid = str(c['eid'])

    for split in SPLITS:
        trials = c['trials'][ba.saturation_for_split(split)]
        elig_idx, labels_real = real_elig(split, trials)
        n_elig = len(elig_idx)
        nL, nR = int(labels_real.sum()), int((~labels_real).sum())
        if nL < 5 or nR < 5:
            print(f'== {split}: recipient fails min5 (L={nL} R={nR}); skip\n')
            continue
        real_minfrac = min(nL, nR) / n_elig
        real_run = runs(labels_real)
        print(f'== {split}  n_elig={n_elig}')
        print(f'   RECIPIENT real: L={nL} R={nR} minority_frac={real_minfrac:.3f} '
              f'prior_run={real_run:.2f}')

        cands, stats = ba._harris_block_donor_candidates(n_elig, bank, eid, split)
        print(f'   donors with cond>={n_elig}: {len(cands)} '
              f'(short={stats["n_short"]}, legacy={stats["n_legacy"]}, '
              f'no_labels={stats["n_no_labels"]})')
        if not cands:
            print('   no donors; skip\n')
            continue

        # Per-donor heterogeneity: full within-stratum prior run + minority frac.
        donor_runs, donor_minf = [], []
        for labels, mask in cands:
            lab = labels[mask]
            donor_runs.append(runs(lab))
            s = int(lab.sum())
            donor_minf.append(min(s, len(lab) - s) / len(lab))
        print(f'   DONOR heterogeneity (per mouse, full stratum): '
              f'prior_run={pct(donor_runs)}  minority_frac={pct(donor_minf)}')

        # Harris null draws (windowed to n_elig + min5), as the pipeline uses.
        rng = np.random.default_rng(0)
        d_run, d_minf, d_inv = [], [], []
        n_ok = 0
        for _ in range(N_DRAW * 6):
            ys = ba._sample_harris_block_ys(
                n_elig, bank, eid, split, rng=rng, candidates=cands, max_tries=1)
            if ys is None:
                continue
            s = int(ys.sum())
            if s < 5 or (n_elig - s) < 5:
                continue
            d_run.append(runs(ys))
            d_minf.append(min(s, n_elig - s) / n_elig)
            d_inv.append(1.0 / s + 1.0 / (n_elig - s))
            n_ok += 1
            if n_ok >= N_DRAW:
                break
        print(f'   HARRIS null draws (n={n_ok}): prior_run={pct(d_run)} '
              f'minority_frac={pct(d_minf)} (1/nL+1/nR)={pct(d_inv)}')
        print(f'      vs RECIPIENT real: prior_run={real_run:.2f} '
              f'minority_frac={real_minfrac:.3f} (1/nL+1/nR)={1/nL+1/nR:.4f}')
        # Where does recipient real run sit in the donor distribution?
        dr = np.asarray(donor_runs, dtype=float)
        dr = dr[np.isfinite(dr)]
        pctile = 100.0 * np.mean(dr <= real_run)
        print(f'      recipient real prior_run is at donor percentile '
              f'{pctile:.0f}%\n')


if __name__ == '__main__':
    main()
