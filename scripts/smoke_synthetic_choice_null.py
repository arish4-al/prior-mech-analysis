#!/usr/bin/env python
"""
Smoke-test sticky psychometric synthetic-choice nulls.

Runs get_d_vars on the first insertion_cache entry that has ≥5 trials/side
for a choice_stim* / choice_duringstim* split.

  python scripts/smoke_synthetic_choice_null.py
  ONE_CACHE_DIR=/path/to/alyx python scripts/smoke_synthetic_choice_null.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from one.api import ONE

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import block_analysis_allsplits as ba  # noqa: E402

SPLITS = [
    'choice_stim_l', 'choice_stim_r',
    'choice_stim_l_block_l', 'choice_stim_l_block_r',
    'choice_stim_r_block_l', 'choice_stim_r_block_r',
    'choice_duringstim_l', 'choice_duringstim_r',
    'choice_duringstim_l_block_l', 'choice_duringstim_r_block_r',
]


def _self_test_fit_sample():
    '''Lightweight unit check of fit/sample without neural data.'''
    n = 200
    rng = np.random.default_rng(0)
    stim_left = rng.random(n) < 0.5
    contrast = rng.choice([0.0, 0.0625, 0.125, 0.25, 1.0], size=n)
    cl = np.where(stim_left, contrast, np.nan)
    cr = np.where(stim_left, np.nan, contrast)
    pleft = np.where(rng.random(n) < 0.5, 0.8, 0.2)
    # Sticky + stim-biased choices.
    choice = np.zeros(n, dtype=float)
    a_prev = 0.0
    for t in range(n):
        s = 1.0 if stim_left[t] else -1.0
        logit = 0.5 * s + 1.5 * s * contrast[t] + 0.8 * (pleft[t] - 0.5) + 0.6 * a_prev
        p_l = 1.0 / (1.0 + np.exp(-logit))
        choice[t] = 1.0 if rng.random() < p_l else -1.0
        a_prev = choice[t]
    trials = {
        'contrastLeft': cl,
        'contrastRight': cr,
        'probabilityLeft': pleft,
        'choice': choice,
    }
    params = ba.fit_sticky_choice_model(trials)
    assert params.get('mode') in ('mle', 'empirical'), params
    syn = ba.sample_synthetic_choices(trials, params, rng=np.random.default_rng(1))
    assert syn.shape == (n,)
    assert set(np.unique(syn)).issubset({-1.0, 1.0})
    print(f'  fit/sample self-test OK (mode={params["mode"]})', flush=True)


def main():
    print('Self-test sticky fit/sample …', flush=True)
    _self_test_fit_sample()

    cache = Path(os.environ.get(
        'ONE_CACHE_DIR',
        Path.home() / 'Downloads/ONE/alyx.internationalbrainlab.org',
    ))
    ba.one = ONE(cache_dir=str(cache), mode='local')
    ba.pth_res = Path(ba.one.cache_dir, 'manifold', 'res')
    ba.pth_res.mkdir(parents=True, exist_ok=True)
    ba._sticky_choice_fit_cache.clear()

    caches = sorted(
        Path(ba.one.cache_dir, 'manifold', 'insertion_cache').glob('*.npy'))
    if not caches:
        print('No insertion_cache/*.npy — fit/sample self-test only.', flush=True)
        print('SMOKE PASSED (self-test)', flush=True)
        return

    nrand = int(os.environ.get('SMOKE_NRAND', '15'))
    for fpath in caches:
        c = np.load(fpath, allow_pickle=True).item()
        pid, eid = c.get('pid'), c.get('eid')
        if not pid:
            continue
        for split in SPLITS:
            try:
                D = ba.get_d_vars(
                    split, pid, control=True, nrand=nrand, cached=c,
                    synthetic_choice_null=True)
            except ba.InsufficientTrials as exc:
                print(f'  skip {fpath.name} {split}: {exc}', flush=True)
                continue
            if not isinstance(D, dict) or D.get('null_scheme') != (
                    'synthetic_choice_sticky'):
                raise SystemExit(
                    f'Unexpected return for {split}: '
                    f'{D.get("null_scheme") if isinstance(D, dict) else type(D)}')
            print(f'OK {split} eid={eid}', flush=True)
            print(f'  null_scheme={D["null_scheme"]} '
                  f'fit={D.get("sticky_fit_mode")} '
                  f'n_regs={len(D.get("D", {}))} uperms={D.get("uperms")}',
                  flush=True)
            print('SMOKE PASSED', flush=True)
            return
    raise SystemExit('SMOKE FAILED: no insertion × split completed')


if __name__ == '__main__':
    main()
