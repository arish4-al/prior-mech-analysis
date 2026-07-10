#!/usr/bin/env python
"""
Goal 2 parity check: cached vs uncached get_d_vars on BWM insertions.

Run from repo root (iblenv / ibl conda env, ONE cache available):

    python scripts/validate_goal2_cache.py
    python scripts/validate_goal2_cache.py --n-pids 20 --nrand 50

With control=True (default), random.seed is reset before each paired call so
null shuffles match. Use --control-only for a faster deterministic check
(true split only, no null loop).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import block_analysis_allsplits as ba  # noqa: E402
from brainwidemap import bwm_query  # noqa: E402

DEFAULT_SPLITS = [
    'block_only',
    'block_duringstim_l_choice_l_f1',
    'block_stim_l_duringchoice_l_f1',
    'act_block_only',
]

TRIAL_COLS = (
    'stimOn_times',
    'choice',
    'probabilityLeft',
    'contrastLeft',
    'contrastRight',
)


def compare_arrays(a, b, path, rtol=0, atol=0):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return f'{path}: shape {a.shape} vs {b.shape}'
    if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
        d = np.nanmax(np.abs(a.astype(float) - b.astype(float)))
        return f'{path}: max |diff| = {d}'
    return None


def compare_d(D_ref, D_test, label):
    errs = []
    if set(D_ref.keys()) != set(D_test.keys()):
        return [f'{label}: top-level keys differ: {set(D_ref)} vs {set(D_test)}']
    if not np.array_equal(D_ref['acs'], D_test['acs']):
        errs.append(f'{label}: acs differ')
    if not np.array_equal(D_ref['acs1'], D_test['acs1']):
        errs.append(f'{label}: acs1 differ')
    for reg in D_ref.get('D', {}):
        if reg not in D_test.get('D', {}):
            errs.append(f'{label}/{reg}: missing in test')
            continue
        for key in ('d_vars', 'd_eucs', 'd_xnobis'):
            for i, (a, b) in enumerate(zip(D_ref['D'][reg][key], D_test['D'][reg][key])):
                e = compare_arrays(a, b, f'{label}/{reg}/{key}[{i}]')
                if e:
                    errs.append(e)
    if D_ref.get('uperms') != D_test.get('uperms'):
        errs.append(
            f'{label}: uperms {D_ref.get("uperms")} vs {D_test.get("uperms")}'
        )
    return errs


def validate_cache_trials(pids, n_cache_pids):
    print(f'=== (A) insertion cache trials == direct ONE load ({n_cache_pids} pids) ===')
    for pid in pids[:n_cache_pids]:
        cache = ba.build_insertion_cache(pid, save=False, restart=False)
        eid, _ = ba.one.pid2eid(pid)
        for st in ba.SATURATION_TYPES:
            t_direct, mask = ba.load_trials_for_saturation(ba.one, eid, st)
            t_direct = t_direct[mask]
            t_cached = cache['trials'][st]
            if len(t_direct) != len(t_cached):
                raise AssertionError(
                    f'{pid} {st}: n trials {len(t_direct)} vs {len(t_cached)}'
                )
            for col in TRIAL_COLS:
                if col not in t_direct.columns:
                    continue
                if not np.allclose(t_direct[col], t_cached[col], equal_nan=True):
                    raise AssertionError(f'{pid} {st}: column {col} differs')
        print('  cache trials OK', pid)


def validate_get_d_vars(pids, splits, control, nrand, seed):
    label = 'control=False (true split)' if not control else f'control=True nrand={nrand} seed={seed}'
    print(f'=== (B) get_d_vars cached vs uncached — {label} ===')
    failures = []
    for pid in pids:
        cache = ba.build_insertion_cache(pid, save=False, restart=False)
        for split in splits:
            if control:
                random.seed(seed)
                np.random.seed(seed)
            D_unc = ba.get_d_vars(
                split, pid, control=control, nrand=nrand, cached=None,
            )
            if control:
                random.seed(seed)
                np.random.seed(seed)
            D_cached = ba.get_d_vars(
                split, pid, control=control, nrand=nrand, cached=cache,
            )
            errs = compare_d(D_unc, D_cached, f'{pid}/{split}')
            if errs:
                failures.extend(errs)
                print('  FAIL', pid, split)
                for e in errs[:5]:
                    print('    ', e)
            else:
                print('  OK  ', pid, split)
    return failures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n-pids', type=int, default=5,
                   help='insertions to test in get_d_vars parity (default 5)')
    p.add_argument('--n-cache-pids', type=int, default=2,
                   help='insertions for raw cache trials check (default 2)')
    p.add_argument('--nrand', type=int, default=50,
                   help='null shuffles when control=True (default 50)')
    p.add_argument('--seed', type=int, default=123,
                   help='RNG seed for paired control=True calls (default 123)')
    p.add_argument('--control-only', action='store_true',
                   help='use control=False (faster, no null loop)')
    p.add_argument('--splits', nargs='*', default=DEFAULT_SPLITS,
                   help='split names to test')
    p.add_argument('--skip-cache-trials', action='store_true',
                   help='skip step (A) trials table check')
    args = p.parse_args()

    df = bwm_query(ba.one)
    pids = df['pid'].values
    print('ONE cache:', ba.one.cache_dir)
    print('BWM insertions:', len(pids))
    print('Test splits:', args.splits)

    if not args.skip_cache_trials:
        validate_cache_trials(pids, args.n_cache_pids)

    failures = validate_get_d_vars(
        pids[: args.n_pids],
        args.splits,
        control=not args.control_only,
        nrand=args.nrand,
        seed=args.seed,
    )

    print('---')
    if failures:
        print(f'FAILED: {len(failures)} issue(s)')
        return 1
    print('All Goal 2 parity checks passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
