#!/usr/bin/env python
"""Contrast-response slope vs true-block prior, per BWM insertion.

Statistic: OLS slope of region-mean spike count (one 0–150 ms bin) against
raw contrast, concordant block minus discordant block, averaged over stim
sides. Concordant = true ``probabilityLeft`` favors that stim side.
0.5-blocks are dropped. 0% contrast stays in, on its nominal side.

  python scripts/run_crf_slope.py --cached-only --local --nrand 10000

Sharded (finalize separately with --stack-only):

  python scripts/run_crf_slope.py --cached-only --local --nrand 10000 \\
      --shard-idx 0 --n-shards 4 --no-stack
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import one.api as _one_api  # noqa: E402

_real_ONE = _one_api.ONE


def _deferred_ONE(*args, **kwargs):
    kwargs.setdefault('mode', 'local')
    kwargs.setdefault('silent', True)
    return _real_ONE(*args, **kwargs)


_one_api.ONE = _deferred_ONE

import block_analysis_allsplits as ba  # noqa: E402


def _configure_one(cache_dir: str | None, base_url: str | None, local: bool):
    kwargs = {'silent': True}
    if cache_dir:
        kwargs['cache_dir'] = cache_dir
    if local:
        kwargs['mode'] = 'local'
    elif base_url:
        kwargs['base_url'] = base_url
    ba.one = _real_ONE(**kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--one-cache-dir', default=os.environ.get('ONE_CACHE_DIR'))
    p.add_argument('--one-base-url', default=os.environ.get('ONE_BASE_URL'))
    p.add_argument('--nrand', type=int, default=10000)
    p.add_argument('--window', type=float, default=0.15,
                   help='Post-stim bin end in seconds (start is 0)')
    p.add_argument('--restart', action=argparse.BooleanOptionalAction, default=True,
                   help='Skip insertions that already have a crf_slope npy')
    p.add_argument('--cached-only', action='store_true')
    p.add_argument('--local', action='store_true')
    p.add_argument('--n-insertions', type=int, default=None)
    p.add_argument('--shard-idx', type=int, default=None)
    p.add_argument('--n-shards', type=int, default=1)
    p.add_argument('--no-stack', action='store_true')
    p.add_argument('--stack-only', action='store_true')
    args = p.parse_args()

    cache_dir = None
    if args.one_cache_dir:
        cache_dir = Path(args.one_cache_dir).expanduser().resolve()
        if 'openalyx' in str(cache_dir).lower():
            raise SystemExit(
                f'Refusing to write into openalyx: {cache_dir}\n'
                'Point --one-cache-dir at the alyx working cache.'
            )
    use_local = args.local or args.cached_only or args.stack_only
    _configure_one(
        str(cache_dir) if cache_dir is not None else None,
        args.one_base_url,
        local=use_local,
    )
    print(f'ONE cache: {ba.one.cache_dir} (local={use_local})')
    print(
        'CRF: true-block prior, window [0, '
        f'{args.window}] s, raw-contrast OLS, nrand={args.nrand}'
    )
    if args.stack_only:
        ba.crf_slope_stacked()
        return

    if args.cached_only:
        icache = Path(ba.one.cache_dir, 'manifold', 'insertion_cache')
        files = sorted(icache.glob('*.npy'))
        if not files:
            raise SystemExit(f'No insertion caches in {icache}')
        import numpy as np
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
        import numpy as np
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

    ba.get_all_crf_slope(
        eids_plus=eids_plus,
        nrand=args.nrand,
        restart=args.restart,
        use_cache=True,
        window=(0.0, float(args.window)),
    )
    if args.no_stack or (args.shard_idx is not None and n_shards > 1):
        print('Skipping stack; run --stack-only after all shards')
        return
    ba.crf_slope_stacked()


if __name__ == '__main__':
    main()
