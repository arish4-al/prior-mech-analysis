#!/usr/bin/env python
"""
Smoke-test Harris unique-null for act_block prior L–R splits.

  python scripts/smoke_act_block_harris_null.py
  ONE_CACHE_DIR=/orcd/data/.../alyx python scripts/smoke_act_block_harris_null.py
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
    'act_block_duringstim_l',
    'act_block_duringstim_r',
    'act_block_duringchoice_l',
    'act_block_duringchoice_r',
    'act_block_duringstim_l_choice_l_f1',
    'act_block_duringstim_r_choice_r_f1',
    'act_block_only',
    'act_block_stim_l_duringchoice_l_f1',
    'act_block_stim_r_duringchoice_r_f1',
    'bayes_block_duringstim_l',
    'bayes_block_duringstim_r',
    'bayes_block_duringstim_l_choice_l_f1',
    'bayes_block_duringstim_r_choice_r_f1',
    'bayes_block_stim_l_duringchoice_l_f1',
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
    print('Loading donor bank …', flush=True)
    bank = ba.load_choice_donor_bank()
    if not bank:
        raise SystemExit('Empty donor bank — need manifold/insertion_cache')
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
            if not ba.is_act_block_prior_split(split):
                raise SystemExit(f'not act_block: {split}')
            try:
                D = ba.get_d_vars(
                    split, pid, control=True, nrand=nrand, cached=c,
                    donor_bank=bank, session_shuffle_null=True)
            except ba.InsufficientTrials as exc:
                print(f'  skip {fpath.name} {split}: {exc}', flush=True)
                continue
            if not isinstance(D, dict) or D.get('null_scheme') != (
                    'harris_session_permutation_unique'):
                raise SystemExit(
                    f'Unexpected return for {split}: '
                    f'{D.get("null_scheme") if isinstance(D, dict) else type(D)}')
            if ba.RES_FILE_SUFFIX != '_harris_unique':
                raise SystemExit(
                    f'Expected RES_FILE_SUFFIX=_harris_unique, got '
                    f'{ba.RES_FILE_SUFFIX!r}')
            n_donors = D.get('harris_n_stratum_donors')
            n_unique = D.get('harris_n_unique_nulls')
            n_stored = None
            if D.get('D'):
                reg0 = next(iter(D['D']))
                n_stored = max(len(D['D'][reg0]['d_eucs']) - 1, 0)
            print(f'OK {split} eid={eid}', flush=True)
            print(f'  null_scheme={D["null_scheme"]} '
                  f'suffix={ba.RES_FILE_SUFFIX} '
                  f'n_regs={len(D.get("D", {}))} uperms={D.get("uperms")}',
                  flush=True)
            print(f'  cond-matched donors: {n_donors}; '
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
