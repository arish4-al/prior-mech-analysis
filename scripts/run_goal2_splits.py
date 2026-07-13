#!/usr/bin/env python
"""
Run Goal 2 BWM pipeline (insertion cache + stream_pool) for an explicit split list.

  python scripts/run_goal2_splits.py --preset stimOn_times_act
  python scripts/run_goal2_splits.py --preset goal3_duringstim_act --contrasts 0.0 0.125 1.0
  python scripts/run_goal2_splits.py --preset goal3_duringstim_act --list-splits
  python scripts/run_goal2_splits.py --splits act_block_duringstim_l --shard-idx 0 --n-shards 4
  python scripts/run_goal2_splits.py --finalize-only --splits act_block_duringstim_l
  python scripts/run_goal2_splits.py --cache-only
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


def _goal3_presets(contrasts=None):
    '''Contrast × duringstim/duringchoice bases (act / true-block / bayes).'''
    act_stim = [s for s in ba.GOAL3_DURINGSTIM_BASES if s.startswith('act_')]
    block_stim = [s for s in ba.GOAL3_DURINGSTIM_BASES if not s.startswith('act_')]
    act_choice = [s for s in ba.GOAL3_DURINGCHOICE_BASES if s.startswith('act_')]
    block_choice = [s for s in ba.GOAL3_DURINGCHOICE_BASES if not s.startswith('act_')]
    return {
        'goal3_duringstim': ba.expand_contrast_splits(ba.GOAL3_DURINGSTIM_BASES, contrasts),
        'goal3_duringchoice': ba.expand_contrast_splits(ba.GOAL3_DURINGCHOICE_BASES, contrasts),
        'goal3_duringstim_act': ba.expand_contrast_splits(act_stim, contrasts),
        'goal3_duringstim_block': ba.expand_contrast_splits(block_stim, contrasts),
        'goal3_duringchoice_act': ba.expand_contrast_splits(act_choice, contrasts),
        'goal3_duringchoice_block': ba.expand_contrast_splits(block_choice, contrasts),
        'goal3_duringstim_bayes': ba.expand_contrast_splits(
            ba.GOAL3_BAYES_DURINGSTIM_BASES, contrasts),
        'goal3_duringchoice_bayes': ba.expand_contrast_splits(
            ba.GOAL3_BAYES_DURINGCHOICE_BASES, contrasts),
        'goal3_bayes_all': ba.expand_contrast_splits(ba.GOAL3_BAYES_BASE_SPLITS, contrasts),
        'goal3_all': ba.expand_contrast_splits(ba.GOAL3_BASE_SPLITS, contrasts),
    }


PRESETS = {
    'stimOn_times_act': RUN_ALIGN['stimOn_times'] + [
        'act_block_duringstim_l', 'act_block_duringstim_r'],
    'stimOn_times_bayes': [
        'bayes_block_duringstim_r_choice_r_f1',
        'bayes_block_duringstim_l_choice_l_f1',
        'bayes_block_duringstim_l_choice_r_f2',
        'bayes_block_duringstim_r_choice_l_f2',
        'bayes_block_duringstim_l',
        'bayes_block_duringstim_r',
    ],
    # Stim L vs R under Bayes prior (from choicestim; cached pipeline)
    'stim_duringstim_bayes': [
        'stim_choice_r_block_r_bayes',
        'stim_choice_l_block_l_bayes',
        'stim_choice_r_block_l_bayes',
        'stim_choice_l_block_r_bayes',
    ],
    'stim_duringstim1_bayes': [
        'stim_block_l_bayes',
        'stim_block_r_bayes',
    ],
    'stim_lr_bayes_all': [
        'stim_choice_r_block_r_bayes',
        'stim_choice_l_block_l_bayes',
        'stim_choice_r_block_l_bayes',
        'stim_choice_l_block_r_bayes',
        'stim_block_l_bayes',
        'stim_block_r_bayes',
    ],
    **_goal3_presets(),
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
            f'Add to align_old / register_contrast_splits in '
            f'block_analysis_allsplits.py first.'
        )
    return splits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset', choices=sorted(PRESETS), default=None)
    p.add_argument('--splits', nargs='*', default=None)
    p.add_argument('--contrasts', nargs='*', type=float, default=None,
                   help='Subset of contrasts for goal3 presets (default: all)')
    p.add_argument('--nrand', type=int, default=ba.nrand)
    p.add_argument('--restart', action='store_true', default=True)
    p.add_argument('--no-restart', dest='restart', action='store_false')
    p.add_argument('--stream-pool', action='store_true', default=True)
    p.add_argument('--no-stream-pool', dest='stream_pool', action='store_false')
    p.add_argument('--save-cache', action='store_true', default=True)
    p.add_argument('--no-save-cache', dest='save_cache', action='store_false')
    p.add_argument('--one-cache-dir', default=os.environ.get('ONE_CACHE_DIR'))
    p.add_argument('--one-base-url',
                   default=os.environ.get('ONE_BASE_URL', 'https://alyx.internationalbrainlab.org'))
    p.add_argument('--cache-only', action='store_true',
                   help='Only build manifold/insertion_cache')
    p.add_argument('--shard-idx', type=int, default=None,
                   help='0-based shard index (with --n-shards)')
    p.add_argument('--n-shards', type=int, default=None,
                   help='Number of insertion shards for this split')
    p.add_argument('--finalize-only', action='store_true',
                   help='Merge stream_acc shards and write res/{split}*.npy')
    p.add_argument('--no-finalize', dest='finalize', action='store_false', default=True,
                   help='Skip finalize (shard workers should use this)')
    p.add_argument('--list-splits', action='store_true',
                   help='Print resolved split names (one per line) and exit')
    args = p.parse_args()

    presets = dict(PRESETS)
    if args.contrasts is not None:
        presets.update(_goal3_presets(args.contrasts))

    if args.preset:
        splits = presets[args.preset]
    elif args.splits:
        splits = args.splits
    else:
        splits = presets['stimOn_times_act']

    if args.list_splits:
        for sp in splits:
            print(sp)
        return

    _configure_one(args.one_cache_dir, args.one_base_url)

    if args.contrasts is not None:
        ba.register_contrast_splits(contrasts=args.contrasts)

    if args.cache_only:
        print('ONE cache:', ba.one.cache_dir)
        ba.cache_all_insertions(restart=args.restart)
        print('Done. Insertion cache under:',
              Path(ba.one.cache_dir, 'manifold', 'insertion_cache'))
        return

    splits = _validate_splits(splits)

    if args.finalize_only:
        print('ONE cache:', ba.one.cache_dir)
        print('Finalize splits:', splits)
        for sp in splits:
            ba.finalize_stream_shards(sp)
            for name in (f'{sp}.npy', f'{sp}_regde.npy'):
                fp = ba.pth_res / name
                tag = 'OK' if fp.exists() else 'MISSING'
                size = f'{fp.stat().st_size/1e6:.1f} MB' if fp.exists() else ''
                print(f'  {tag} {fp} {size}')
        return

    # Shard workers must not finalize (merge job does that).
    finalize = args.finalize
    if args.n_shards is not None and args.n_shards > 1:
        finalize = False

    print('ONE cache:', ba.one.cache_dir)
    print('Splits:', splits)
    print('nrand:', args.nrand, 'restart:', args.restart,
          'stream_pool:', args.stream_pool, 'save_cache:', args.save_cache,
          'shard:', args.shard_idx, '/', args.n_shards, 'finalize:', finalize)

    # bycontrast=False: contrast is read from the split name (..._0.125).
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
        shard_idx=args.shard_idx,
        n_shards=args.n_shards,
        finalize=finalize,
    )
    print('Done. Outputs under:', ba.pth_res)
    if finalize:
        for sp in splits:
            for name in (f'{sp}.npy', f'{sp}_regde.npy'):
                fp = ba.pth_res / name
                if fp.exists():
                    print(f'  OK {fp} ({fp.stat().st_size/1e6:.1f} MB)')
                else:
                    print(f'  MISSING {fp}')
    elif args.n_shards is not None:
        for sp in splits:
            fp = ba.pth_stream_acc / f'{sp}.shard{args.shard_idx}.npy'
            if fp.exists():
                print(f'  OK shard {fp} ({fp.stat().st_size/1e6:.1f} MB)')
            else:
                print(f'  MISSING shard {fp}')


if __name__ == '__main__':
    main()
