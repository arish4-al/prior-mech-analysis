#!/usr/bin/env python
"""Check that standard-split jobs can run with Alyx blocked.

Run before submitting cluster jobs (``submit_standard_alt_nulls_orcd.sh`` runs it
unless ``PREFLIGHT=0``). It opens ONE in local mode with Alyx blocked, the same
way a Slurm job does (see ``one_offline.py``). For every BWM session it then
loads the inputs a shard reads:

  * the unmasked trial table (standard act prior), and with ``--sticky`` the
    ``truncate_to_pass=False`` table for the θ fit. Each is saved under
    ``manifold/session_trials/`` the first time, so jobs read the saved file;
  * ``bwm_tables/trials.pqt`` (saturation masks);
  * ``manifold/insertion_cache/{eid}_{probe}.npy`` (a missing one is rebuilt
    by the job from local files, so it is a warning, not an error);
  * with ``--sticky``, the ``actkernel_fits/{eid8}_fullsess`` pickle;
  * with ``--harris``, ``choice_donors_standard.npy`` (a warning: the donor job
    builds it offline).

Exits 1 when any session would need Alyx, or when an Alyx request was attempted.

    ONE_CACHE_DIR=... python scripts/check_offline_inputs.py --sticky --harris
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('BLOCK_ALYX', '1')
os.environ.setdefault('ONE_MODE', 'local')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import block_analysis_allsplits as ba  # noqa: E402
from one_offline import AlyxBlockedError  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sticky', action='store_true',
                    help='also check the full-session θ table and pickle')
    ap.add_argument('--no-pickle-check', action='store_true',
                    help='with --sticky, skip the θ pickle check (prefit runs first)')
    ap.add_argument('--harris', action='store_true',
                    help='also check the standard Harris donor bank')
    ap.add_argument('--limit', type=int, default=None,
                    help='check only the first N sessions (smoke test)')
    ap.add_argument('--one-cache-dir', default=os.environ.get('ONE_CACHE_DIR'),
                    help='ONE cache the jobs use (default: $ONE_CACHE_DIR)')
    args = ap.parse_args()
    if args.one_cache_dir:
        ba.one = ba.make_one(cache_dir=args.one_cache_dir)

    t0 = time.perf_counter()
    print(f'ONE: {ba.one} mode={ba.one.mode}')
    errors, warnings = [], []

    agg = Path(ba.one.cache_dir, 'bwm_tables', 'trials.pqt')
    if not agg.exists():
        errors.append(f'missing aggregate trials table {agg}')

    df = ba.bwm_query(ba.one)
    icache = Path(ba.one.cache_dir, 'manifold', 'insertion_cache')
    missing_ic = [f'{e}_{p}' for e, p in df[['eid', 'probe_name']].values
                  if not (icache / f'{e}_{p}.npy').exists()]
    if missing_ic:
        warnings.append(f'{len(missing_ic)}/{len(df)} insertion caches missing; '
                        f'jobs rebuild them from local files (first: {missing_ic[:3]})')

    eids = sorted(df['eid'].astype(str).unique())
    if args.limit:
        eids = eids[:args.limit]
    variants = [True] + ([False] if args.sticky else [])
    bad = {}
    n_saved = n_present = 0
    for i, eid in enumerate(eids, 1):
        for trunc in variants:
            had = ba._session_trials_path(eid, trunc).exists()
            n_present += had
            try:
                ba._session_trials_full(eid, truncate_to_pass=trunc)
                n_saved += not had
            except AlyxBlockedError as exc:
                bad[eid] = f'needed Alyx: {exc}'
            except Exception as exc:  # noqa: BLE001
                bad[eid] = f'{type(exc).__name__}: {exc}'
        ba._full_trials_memo.clear()
        if args.sticky and not args.no_pickle_check:
            fit_dir = ba._actkernel_model_dir() / ba._actkernel_fit_tag(
                eid, full_session=True)
            if not any(fit_dir.rglob('train_*.pkl')):
                bad.setdefault(eid, f'no θ pickle in {fit_dir}')
        if i % 50 == 0:
            print(f'  {i}/{len(eids)} sessions checked, {len(bad)} bad', flush=True)
    print(f'session trial tables: {n_present} already saved, '
          f'{n_saved} built and saved under manifold/session_trials/')
    for eid, why in bad.items():
        errors.append(f'{eid}: {why}')

    if args.harris and not ba._choice_donors_standard_path().exists():
        warnings.append('no standard donor bank yet; the std_donors job builds it offline')

    for w in warnings:
        print('WARNING:', w)
    for e in errors[:40]:
        print('ERROR:', e)
    if len(errors) > 40:
        print(f'ERROR: ... {len(errors) - 40} more')
    dt = time.perf_counter() - t0
    print(f'{len(eids)} sessions, {len(errors)} errors, {len(warnings)} warnings '
          f'({dt:.0f} s)')
    sys.exit(1 if errors else 0)


if __name__ == '__main__':
    main()
