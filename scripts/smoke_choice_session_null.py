#!/usr/bin/env python
"""
Smoke-test stratified Harris session-permutation nulls.

Builds / loads choice_donors.npy, then runs get_d_vars on the first insertion
cache entry that has ≥5 trials/side for a choice_stim* / choice_duringstim* split.

  python scripts/smoke_choice_session_null.py
  ONE_CACHE_DIR=/orcd/data/.../alyx python scripts/smoke_choice_session_null.py
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


def main():
    cache = Path(os.environ.get(
        'ONE_CACHE_DIR',
        Path.home() / 'Downloads/ONE/alyx.internationalbrainlab.org',
    ))
    ba.one = ONE(cache_dir=str(cache), mode='local')
    ba.pth_res = Path(ba.one.cache_dir, 'manifold', 'res')
    ba.pth_res.mkdir(parents=True, exist_ok=True)

    print('Building / rebuilding choice donor bank …', flush=True)
    bank = ba.build_choice_donor_bank(restart=False)
    if not bank:
        raise SystemExit('Empty donor bank — need manifold/insertion_cache')
    rec0 = next(iter(bank.values()))
    if not isinstance(rec0, dict) or 'stim_is_left' not in rec0:
        raise SystemExit('Donor bank is legacy format; rebuild failed')
    print(f'  {len(bank)} eids -> {ba._choice_donors_path()}', flush=True)

    caches = sorted(
        Path(ba.one.cache_dir, 'manifold', 'insertion_cache').glob('*.npy'))
    if not caches:
        raise SystemExit('No insertion_cache/*.npy')

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
                    donor_bank=bank, session_shuffle_null=True)
            except ba.InsufficientTrials as exc:
                print(f'  skip {fpath.name} {split}: {exc}', flush=True)
                continue
            if not isinstance(D, dict) or D.get('null_scheme') != (
                    'session_shuffle_stratified'):
                raise SystemExit(
                    f'Unexpected return for {split}: '
                    f'{D.get("null_scheme") if isinstance(D, dict) else type(D)}')
            n_long = sum(
                1 for e, rec in bank.items()
                if e != str(eid)
                and len(ba._donor_choice_stream(rec, split)) >= 10)
            print(f'OK {split} eid={eid}', flush=True)
            print(f'  null_scheme={D["null_scheme"]} '
                  f'n_regs={len(D.get("D", {}))} uperms={D.get("uperms")}',
                  flush=True)
            print(f'  donors with stream≥10: {n_long}/{max(len(bank) - 1, 0)}',
                  flush=True)
            print('SMOKE PASSED', flush=True)
            return
    raise SystemExit('SMOKE FAILED: no insertion × split completed')


if __name__ == '__main__':
    main()
