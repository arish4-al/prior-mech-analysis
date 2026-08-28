#!/usr/bin/env python
"""
Smoke-test Harris session-permutation nulls (donor stratum re-filter).

Builds / loads choice_donors.npy, then runs get_d_vars on the first insertion
cache entry that works for a choice_stim* / choice_duringstim* split.

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
    # Prefer production act splits.
    'choice_stim_l_block_l_act', 'choice_stim_r_block_r_act',
    'choice_duringstim_l_block_l_act', 'choice_duringstim_r_block_r_act',
    'choice_stim_l_block_l_bayes', 'choice_stim_r_block_r_bayes',
    'choice_duringstim_l_block_l_bayes', 'choice_duringstim_r_block_r_bayes',
    'choice_stim_l', 'choice_stim_r',
    'choice_stim_l_block_l', 'choice_stim_l_block_r',
    'choice_duringstim_l', 'choice_duringstim_r',
]


def main():
    cache = Path(os.environ.get(
        'ONE_CACHE_DIR',
        Path.home() / 'Downloads/ONE/alyx.internationalbrainlab.org',
    ))
    ba.one = ONE(cache_dir=str(cache), mode='local')
    ba.pth_res = Path(ba.one.cache_dir, 'manifold', 'res')
    ba.pth_res.mkdir(parents=True, exist_ok=True)

    ba.configure_null_file_suffix(session_shuffle_null=True)
    print(f'Configured RES_FILE_SUFFIX={ba.RES_FILE_SUFFIX!r}', flush=True)
    print('Loading choice donor bank (prefers largest on-disk copy) …', flush=True)
    bank = ba.load_choice_donor_bank()
    if not bank:
        raise SystemExit('Empty donor bank — need manifold/insertion_cache')
    rec0 = next(iter(bank.values()))
    if not isinstance(rec0, dict) or 'choice' not in rec0:
        raise SystemExit('Donor bank missing choice field; rebuild failed')
    if 'stim_is_left' not in rec0 or 'pleft_true' not in rec0:
        raise SystemExit('Donor bank missing stim/pleft; rebuild required')
    print(f'  {len(bank)} eids', flush=True)

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
            if not isinstance(D, dict) or D.get('null_scheme') not in (
                    'harris_session_permutation_unique',
                    'harris_session_permutation',  # legacy tag
            ):
                raise SystemExit(
                    f'Unexpected return for {split}: '
                    f'{D.get("null_scheme") if isinstance(D, dict) else type(D)}')
            if ba.RES_FILE_SUFFIX != '_harris_unique':
                raise SystemExit(
                    f'Expected RES_FILE_SUFFIX=_harris_unique, got '
                    f'{ba.RES_FILE_SUFFIX!r}')
            n_donors = D.get('harris_n_stratum_donors')
            n_unique = D.get('harris_n_unique_nulls')
            # Unique-null mode: stored null curves == unique labels (no dups).
            n_stored = None
            if D.get('D'):
                reg0 = next(iter(D['D']))
                n_stored = max(len(D['D'][reg0]['d_eucs']) - 1, 0)
            print(f'OK {split} eid={eid}', flush=True)
            print(f'  null_scheme={D["null_scheme"]} '
                  f'suffix={ba.RES_FILE_SUFFIX} '
                  f'n_regs={len(D.get("D", {}))} uperms={D.get("uperms")}',
                  flush=True)
            print(f'  stratum-matched donors: {n_donors}; '
                  f'unique_nulls={n_unique} stored_nulls={n_stored}',
                  flush=True)
            if n_unique is not None and n_stored is not None and n_unique != n_stored:
                raise SystemExit(
                    f'Unique-null mismatch: harris_n_unique_nulls={n_unique} '
                    f'vs stored={n_stored}')
            print('SMOKE PASSED', flush=True)
            return
    raise SystemExit('SMOKE FAILED: no insertion × split completed')


if __name__ == '__main__':
    main()
