#!/usr/bin/env python
"""
Run Goal 2 BWM pipeline (insertion cache + stream_pool) for an explicit split list.

  python scripts/run_goal2_splits.py --splits block_only act_block_stim_l
  python scripts/run_goal2_splits.py --preset stimOn_times_act
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import block_analysis_allsplits as ba  # noqa: E402
from one.api import ONE  # noqa: E402


# Mirrors block_analysis_allsplits.run_align (keep in sync for presets).
RUN_ALIGN = {
    'stimOn_times': [
        'act_block_duringstim_r_choice_r_f1',
        'act_block_duringstim_l_choice_l_f1',
        'act_block_duringstim_l_choice_r_f2',
        'act_block_duringstim_r_choice_l_f2',
    ],
    'stimOn_times1': [
        'block_duringstim_r_choice_r_f1',
        'block_duringstim_l_choice_l_f1',
        'block_duringstim_l_choice_r_f2',
        'block_duringstim_r_choice_l_f2',
    ],
    'firstMovement_times': [
        'act_block_stim_r_duringchoice_r_f1',
        'act_block_stim_l_duringchoice_l_f1',
        'act_block_stim_l_duringchoice_r_f2',
        'act_block_stim_r_duringchoice_l_f2',
    ],
    'firstMovement_times1': [
        'block_stim_r_duringchoice_r_f1',
        'block_stim_l_duringchoice_l_f1',
        'block_stim_l_duringchoice_r_f2',
        'block_stim_r_duringchoice_l_f2',
    ],
}

PRESETS = {
    # User cluster job: run_align['stimOn_times'] + new action-kernel stim-side splits.
    'stimOn_times_act': RUN_ALIGN['stimOn_times'] + ['act_block_stim_l', 'act_block_stim_r'],
}


def _configure_one(cache_dir: str | None, base_url: str | None):
    kwargs = {}
    if cache_dir:
        kwargs['cache_dir'] = cache_dir
    if base_url:
        kwargs['base_url'] = base_url
    ba.one = ONE(**kwargs) if kwargs else ONE()
    ba.pth_res = Path(ba.one.cache_dir, 'manifold', 'res')
    ba.pth_res.mkdir(parents=True, exist_ok=True)
    ba.pth_stream_acc = ba.pth_res / '_stream_acc'
    ba.pth_stream_acc.mkdir(parents=True, exist_ok=True)


def _validate_splits(splits: list[str]) -> list[str]:
    missing = [s for s in splits if s not in ba.align]
    if missing:
        raise SystemExit(
            f'Splits missing from align/pre_post: {missing}\n'
            f'Add to align_old in block_analysis_allsplits.py first.'
        )
    return splits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset', choices=sorted(PRESETS), default=None,
                   help='Named split list (overrides --splits if set)')
    p.add_argument('--splits', nargs='*', default=None, help='Explicit split names')
    p.add_argument('--nrand', type=int, default=ba.nrand)
    p.add_argument('--restart', action='store_true', default=True)
    p.add_argument('--no-restart', dest='restart', action='store_false')
    p.add_argument('--stream-pool', action='store_true', default=True)
    p.add_argument('--no-stream-pool', dest='stream_pool', action='store_false')
    p.add_argument('--save-cache', action='store_true', default=True)
    p.add_argument('--no-save-cache', dest='save_cache', action='store_false')
    p.add_argument('--one-cache-dir', default=os.environ.get('ONE_CACHE_DIR'))
    p.add_argument('--one-base-url', default=os.environ.get('ONE_BASE_URL', 'https://alyx.internationalbrainlab.org'))
    p.add_argument('--cache-only', action='store_true',
                   help='Only build manifold/insertion_cache (no split analysis)')
    args = p.parse_args()

    _configure_one(args.one_cache_dir, args.one_base_url)

    if args.cache_only:
        print('ONE cache:', ba.one.cache_dir)
        print('cache_only restart:', args.restart)
        ba.cache_all_insertions(restart=args.restart)
        print('Done. Insertion cache under:', Path(ba.one.cache_dir, 'manifold', 'insertion_cache'))
        return

    if args.preset:
        splits = PRESETS[args.preset]
    elif args.splits:
        splits = args.splits
    else:
        splits = PRESETS['stimOn_times_act']

    splits = _validate_splits(splits)

    print('ONE cache:', ba.one.cache_dir)
    print('Splits:', splits)
    print('nrand:', args.nrand, 'restart:', args.restart,
          'stream_pool:', args.stream_pool, 'save_cache:', args.save_cache)

    ba.get_all_d_vars_allsplits(
        splits,
        control=True,
        nrand=args.nrand,
        bycontrast=False,
        restart=args.restart,
        use_cache=True,
        save_cache=args.save_cache,
        stream_pool=args.stream_pool,
        save_per_insertion=not args.stream_pool,
    )
    print('Done. Final outputs under:', ba.pth_res)
    for sp in splits:
        for name in (f'{sp}.npy', f'{sp}_regde.npy'):
            fp = ba.pth_res / name
            if fp.exists():
                print(f'  OK {fp} ({fp.stat().st_size/1e6:.1f} MB)')
            else:
                print(f'  MISSING {fp}')


if __name__ == '__main__':
    main()
