#!/usr/bin/env python
"""
Smoke-test BWM-style ActionKernel synthetic-session nulls.

Runs get_d_vars with --actkernel-choice-null on the first insertion_cache
entry that has enough trials for a choice_stim* / choice_duringstim* split.

Uses a short MCMC (ACTKERNEL_NB_STEPS, default 80) so the fit finishes
quickly; short fits write under a separate actkernel_fits/*_nbN dir and do
not overwrite paper-length pickles.

  conda activate iblenv   # needs torch; behavior_models from third_party/ submodule
  # fresh clone: git submodule update --init --recursive
  python scripts/smoke_choice_actkernel_null.py
  ONE_CACHE_DIR=/orcd/data/.../alyx SMOKE_NRAND=8 ACTKERNEL_NB_STEPS=50 \\
    python scripts/smoke_choice_actkernel_null.py
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
    'choice_duringstim_l', 'choice_duringstim_r',
]


def main():
    # Short MCMC for smoke only (separate pickle path via get_actkernel_choice_fit).
    os.environ.setdefault('ACTKERNEL_NB_STEPS', '80')

    cache = Path(os.environ.get(
        'ONE_CACHE_DIR',
        Path.home() / 'Downloads/ONE/alyx.internationalbrainlab.org',
    ))
    ba.one = ONE(cache_dir=str(cache), mode='local')
    ba.pth_res = Path(ba.one.cache_dir, 'manifold', 'res')
    ba.pth_res.mkdir(parents=True, exist_ok=True)

    caches = sorted(
        Path(ba.one.cache_dir, 'manifold', 'insertion_cache').glob('*.npy'))
    if not caches:
        raise SystemExit('No insertion_cache/*.npy')

    nrand = int(os.environ.get('SMOKE_NRAND', '8'))
    print(f'actkernel smoke: nrand={nrand} '
          f'ACTKERNEL_NB_STEPS={os.environ.get("ACTKERNEL_NB_STEPS")}',
          flush=True)

    for fpath in caches:
        c = np.load(fpath, allow_pickle=True).item()
        pid, eid = c.get('pid'), c.get('eid')
        if not pid:
            continue
        for split in SPLITS:
            try:
                D = ba.get_d_vars(
                    split, pid, control=True, nrand=nrand, cached=c,
                    actkernel_choice_null=True)
            except ba.InsufficientTrials as exc:
                print(f'  skip {fpath.name} {split}: {exc}', flush=True)
                continue
            except Exception as exc:
                print(f'  fail {fpath.name} {split}: {type(exc).__name__}: {exc}',
                      flush=True)
                continue
            if not isinstance(D, dict) or D.get('null_scheme') != (
                    'synthetic_choice_pseudosession'):
                raise SystemExit(
                    f'Unexpected return for {split}: '
                    f'{D.get("null_scheme") if isinstance(D, dict) else type(D)}')
            print(f'OK {split} eid={eid}', flush=True)
            print(f'  null_scheme={D["null_scheme"]} '
                  f'n_regs={len(D.get("D", {}))} uperms={D.get("uperms")}',
                  flush=True)
            print(f'  actkernel_params={D.get("actkernel_params")}', flush=True)
            print('SMOKE PASSED', flush=True)
            return
    raise SystemExit('SMOKE FAILED: no insertion × split completed')


if __name__ == '__main__':
    main()
