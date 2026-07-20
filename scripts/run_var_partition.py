#!/usr/bin/env python
"""
Run single-neuron stim/choice/prior variance partition on BWM insertions.

Region list (git-tracked)
-------------------------
Default ``--regtype-csv``::

  data/stimchoice_act_regtype_regions_p_mean_c_0.01.csv

Remote / local fit (uses default ONE cache; override with ``--one-cache-dir``
or ``$ONE_CACHE_DIR`` only if needed)::

  python scripts/run_var_partition.py --target mixed
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_REGTYPE_CSV = (
    ROOT / 'data' / 'stimchoice_act_regtype_regions_p_mean_c_0.01.csv'
)

# Avoid live Alyx connect on import of block_analysis_allsplits.
import one.api as _one_api  # noqa: E402

_real_ONE = _one_api.ONE


def _deferred_ONE(*args, **kwargs):
    '''Placeholder replaced in _configure_one; keeps import offline-safe.'''
    kwargs.setdefault('mode', 'local')
    kwargs.setdefault('silent', True)
    return _real_ONE(*args, **kwargs)


_one_api.ONE = _deferred_ONE

import block_analysis_allsplits as ba  # noqa: E402


def _configure_one(cache_dir: str | None, base_url: str | None,
                   local: bool = False):
    '''Use default ONE cache when cache_dir is unset (same as ``ONE()``).'''
    kwargs = {'silent': True}
    if cache_dir:
        kwargs['cache_dir'] = cache_dir
    if local:
        kwargs['mode'] = 'local'
    elif base_url:
        kwargs['base_url'] = base_url
    ba.one = _real_ONE(**kwargs)
    ba.pth_res = Path(ba.one.cache_dir, 'manifold', 'res')
    ba.pth_res.mkdir(parents=True, exist_ok=True)


def _load_target_regions(regtype_csv: Path, regtypes: list[float] | None,
                         target: str) -> list[str]:
    import pandas as pd
    if not regtype_csv.exists():
        raise SystemExit(
            f'Missing regtype CSV: {regtype_csv}\n'
            'Pull repo data/ or pass --regtype-csv.'
        )
    df = pd.read_csv(regtype_csv)
    if target == 'mixed':
        if 'mixed_stim_choice' not in df.columns:
            raise SystemExit(
                f'{regtype_csv} missing mixed_stim_choice — re-run '
                'export_stimchoice_regtypes.py'
            )
        mixed = df['mixed_stim_choice']
        if mixed.dtype == object:
            mixed = mixed.astype(str).str.lower().isin(('true', '1', '1.0'))
        else:
            mixed = mixed.astype(bool)
        regs = sorted(df.loc[mixed, 'region'].dropna().unique().tolist())
        label = 'mixed_stim_choice'
    elif target == 'stim_or_early':
        # Strict stim (0) + early block-only stim (0.1) + optional regtypes
        mask = df['sc_duringstim_regtype'].isin([0.0, 0.1])
        if 'stim_processor_loose' in df.columns:
            loose = df['stim_processor_loose']
            if loose.dtype == object:
                loose = loose.astype(str).str.lower().isin(('true', '1', '1.0'))
            else:
                loose = loose.fillna(False).astype(bool)
            mask = mask | loose
        regs = sorted(df.loc[mask, 'region'].dropna().unique().tolist())
        label = 'stim_or_early'
    else:
        # legacy: filter by sc_duringstim_regtype codes
        if regtypes is None:
            regtypes = [0.0, 0.1, 0.5, 1.0]
        allowed = set(float(x) for x in regtypes)
        mask = df['sc_duringstim_regtype'].isin(allowed)
        regs = sorted(df.loc[mask, 'region'].dropna().unique().tolist())
        label = f'regtypes={sorted(allowed)}'
    if not regs:
        raise SystemExit(f'No regions for target={label} from {regtype_csv}')
    return regs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--one-cache-dir',
        default=os.environ.get('ONE_CACHE_DIR'),
        help='ONE cache root for insertion_cache + outputs '
             '(default: ONE() default / $ONE_CACHE_DIR if set)',
    )
    p.add_argument(
        '--one-base-url',
        default=os.environ.get(
            'ONE_BASE_URL', 'https://alyx.internationalbrainlab.org'),
    )
    p.add_argument(
        '--regtype-csv',
        type=Path,
        default=DEFAULT_REGTYPE_CSV,
        help='Region-type CSV (default: repo '
             'data/stimchoice_act_regtype_regions_p_mean_c_0.01.csv)',
    )
    p.add_argument(
        '--regtypes',
        default='0,0.1,0.5,1',
        help='Comma-separated sc_duringstim_regtype values when '
             '--target=regtype (0=stim, 0.1=early, 0.5=integrator, 1=move)',
    )
    p.add_argument(
        '--target',
        choices=['mixed', 'regtype', 'stim_or_early'],
        default='mixed',
        help='Region set: mixed=has_stim∧has_choice (default); '
             'regtype=filter by --regtypes; stim_or_early=Σ′ processors',
    )
    p.add_argument('--window', type=float, default=0.08,
                   help='Post-stimOn window length in seconds '
                        '(default: 0.08 = early stim)')
    p.add_argument('--prior-type', choices=['act', 'block'], default='act',
                   help="Encoding prior: 'act' = action-kernel EMA (default); "
                        "'block' = true probabilityLeft")
    p.add_argument('--min-trials', type=int, default=30)
    p.add_argument('--min-neurons', type=int, default=5,
                   help='Min neurons per region for stacked summary')
    p.add_argument('--restart', action='store_true', default=True)
    p.add_argument('--no-restart', dest='restart', action='store_false')
    p.add_argument('--use-cache', action='store_true', default=True)
    p.add_argument('--no-use-cache', dest='use_cache', action='store_false')
    p.add_argument('--n-insertions', type=int, default=None,
                   help='Smoke: only first N BWM insertions')
    p.add_argument('--pids', nargs='*', default=None,
                   help='Optional explicit pid list')
    p.add_argument('--stack-only', action='store_true',
                   help='Only aggregate existing var_partition/*.npy')
    p.add_argument('--all-regions', action='store_true',
                   help='Fit all non-void/root regions (ignore regtype filter)')
    p.add_argument(
        '--cached-only',
        action='store_true',
        help='Only use insertions already in manifold/insertion_cache',
    )
    p.add_argument(
        '--local',
        action='store_true',
        help='ONE mode=local (no network; required with --cached-only smoke)',
    )
    p.add_argument('--shard-idx', type=int, default=None,
                   help='0-based insertion shard index (with --n-shards)')
    p.add_argument('--n-shards', type=int, default=1,
                   help='Number of insertion shards (default: 1 = all)')
    p.add_argument('--no-stack', action='store_true',
                   help='Skip region stacking after fit (use with shards; '
                        'finalize with --stack-only)')
    args = p.parse_args()

    cache_dir = None
    if args.one_cache_dir:
        cache_dir = Path(args.one_cache_dir).expanduser().resolve()
        if 'openalyx' in str(cache_dir).lower():
            raise SystemExit(
                f'Refusing to write into openalyx: {cache_dir}\n'
                'Point --one-cache-dir at your working ONE cache.'
            )

    use_local = args.local or args.cached_only or args.stack_only
    _configure_one(
        str(cache_dir) if cache_dir is not None else None,
        args.one_base_url,
        local=use_local,
    )
    print(f'ONE cache: {ba.one.cache_dir} (local={use_local})')

    regtype_csv = args.regtype_csv.expanduser().resolve()
    if not regtype_csv.exists():
        raise SystemExit(
            f'Missing regtype CSV: {regtype_csv}\n'
            'Pull repo data/ or pass --regtype-csv.'
        )
    print(f'Regtype CSV: {regtype_csv}')
    print(f'prior_type={args.prior_type}  window={args.window}')
    regtypes = [float(x) for x in args.regtypes.split(',') if x.strip() != '']

    if args.stack_only:
        ba.var_partition_stacked(
            regtype_csv=regtype_csv if regtype_csv.exists() else None,
            regtypes=None if args.target == 'mixed' else regtypes,
            min_neurons=args.min_neurons,
            mixed_only=(args.target == 'mixed'),
        )
        return

    if args.all_regions:
        regions = None
        print('Fitting all non-void/root regions')
    else:
        regions = _load_target_regions(regtype_csv, regtypes, args.target)
        print(
            f'Target={args.target} regions ({len(regions)}): {regions[:20]}'
            + ('...' if len(regions) > 20 else '')
        )

    if args.pids:
        eids_plus = []
        for pid in args.pids:
            eid, probe = ba.one.pid2eid(pid)
            eids_plus.append([eid, probe, pid])
        eids_plus = np.array(eids_plus, dtype=object)
    elif args.cached_only:
        icache = Path(ba.one.cache_dir, 'manifold', 'insertion_cache')
        files = sorted(icache.glob('*.npy'))
        if not files:
            raise SystemExit(f'No insertion caches in {icache}')
        eids_plus = []
        for f in files:
            D = np.load(f, allow_pickle=True).item()
            eids_plus.append([D['eid'], D['probe'], D['pid']])
            if args.n_insertions is not None and len(eids_plus) >= args.n_insertions:
                break
        eids_plus = np.array(eids_plus, dtype=object)
        print(f'Using {len(eids_plus)} cached insertions')
    else:
        from brainwidemap import bwm_query
        df = bwm_query(ba.one)
        eids_plus = df[['eid', 'probe_name', 'pid']].values
        if args.n_insertions is not None:
            eids_plus = eids_plus[: args.n_insertions]

    n_shards = max(1, int(args.n_shards))
    if args.shard_idx is not None:
        if not (0 <= args.shard_idx < n_shards):
            raise SystemExit(
                f'--shard-idx must be in [0, {n_shards}), got {args.shard_idx}')
        eids_plus = eids_plus[args.shard_idx::n_shards]
        print(f'Shard {args.shard_idx}/{n_shards}: {len(eids_plus)} insertions')

    ba.get_all_var_partition(
        eids_plus=eids_plus,
        regions=regions,
        window=(0.0, args.window),
        restart=args.restart,
        use_cache=args.use_cache,
        min_trials=args.min_trials,
        prior_type=args.prior_type,
    )
    if args.no_stack or (args.shard_idx is not None and n_shards > 1):
        print('Skipping stack (shard or --no-stack); run --stack-only to finalize')
        return
    ba.var_partition_stacked(
        regtype_csv=regtype_csv if regtype_csv.exists() else None,
        regtypes=None if args.all_regions or args.target == 'mixed' else regtypes,
        min_neurons=args.min_neurons,
        mixed_only=(args.target == 'mixed' and not args.all_regions),
    )


if __name__ == '__main__':
    main()
