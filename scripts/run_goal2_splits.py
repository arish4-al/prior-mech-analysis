#!/usr/bin/env python
"""
Run Goal 2 BWM pipeline (insertion cache + stream_pool) for an explicit split list.

  python scripts/run_goal2_splits.py --preset stimOn_times_act
  python scripts/run_goal2_splits.py --preset goal3_c0_choice
  python scripts/run_goal2_splits.py --preset goal3_duringstim_act --contrasts 0.0 0.125 1.0
  python scripts/run_goal2_splits.py --preset goal3_duringstim_act --list-splits
  python scripts/run_goal2_splits.py --splits act_block_duringstim_l --shard-idx 0 --n-shards 4
  python scripts/run_goal2_splits.py --finalize-only --splits act_block_duringstim_l
  python scripts/run_goal2_splits.py --cache-only
  python scripts/run_goal2_splits.py --build-choice-donors
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

# Choice L vs R under fixed stim×block (Harris session-permutation null).
CHOICE_DURINGCHOICE = [
    'choice_stim_r_block_r',
    'choice_stim_l_block_l',
    'choice_stim_r_block_l',
    'choice_stim_l_block_r',
]
CHOICE_DURINGCHOICE_ACT = [
    'choice_stim_r_block_r_act',
    'choice_stim_l_block_l_act',
    'choice_stim_r_block_l_act',
    'choice_stim_l_block_r_act',
]
CHOICE_DURINGSTIM = [
    'choice_duringstim_r_block_r',
    'choice_duringstim_l_block_l',
    'choice_duringstim_r_block_l',
    'choice_duringstim_l_block_r',
]
CHOICE_DURINGSTIM_ACT = [
    'choice_duringstim_r_block_r_act',
    'choice_duringstim_l_block_l_act',
    'choice_duringstim_r_block_l_act',
    'choice_duringstim_l_block_r_act',
]
CHOICE_DURINGCHOICE_BAYES = [
    'choice_stim_r_block_r_bayes',
    'choice_stim_l_block_l_bayes',
    'choice_stim_r_block_l_bayes',
    'choice_stim_l_block_r_bayes',
]
CHOICE_DURINGSTIM_BAYES = [
    'choice_duringstim_r_block_r_bayes',
    'choice_duringstim_l_block_l_bayes',
    'choice_duringstim_r_block_l_bayes',
    'choice_duringstim_l_block_r_bayes',
]


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
    # Choice L vs R (Harris session-permutation null)
    'choice_duringchoice': CHOICE_DURINGCHOICE,
    'choice_duringchoice_act': CHOICE_DURINGCHOICE_ACT,
    'choice_duringstim': CHOICE_DURINGSTIM,
    'choice_duringstim_act': CHOICE_DURINGSTIM_ACT,
    # Default session-null preset: act only (8 splits)
    'choice_lr_session_null_all': (
        CHOICE_DURINGCHOICE_ACT + CHOICE_DURINGSTIM_ACT
    ),
    'choice_lr_session_null_true': (
        CHOICE_DURINGCHOICE + CHOICE_DURINGSTIM
    ),
    'choice_lr_session_null_bayes': (
        CHOICE_DURINGCHOICE_BAYES + CHOICE_DURINGSTIM_BAYES
    ),
    # Late+perseveration exclusion + label-shuffle null (stim×block splits)
    'choice_lr_excl_sticky_act': (
        CHOICE_DURINGCHOICE_ACT + CHOICE_DURINGSTIM_ACT
    ),
    'choice_lr_excl_sticky_true': (
        CHOICE_DURINGCHOICE + CHOICE_DURINGSTIM
    ),
    'choice_lr_excl_sticky_bayes': (
        CHOICE_DURINGCHOICE_BAYES + CHOICE_DURINGSTIM_BAYES
    ),
    # Revised Goal 3: true block L vs R at 0% contrast, separately by choice.
    'goal3_c0_choice': ba.GOAL3_C0_CHOICE_SPLITS,
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
    p.add_argument('--build-choice-donors', action='store_true',
                   help='Rebuild manifold/choice_donors.npy from insertion_cache '
                        '(full-session choice sequences for Harris nulls)')
    p.add_argument('--session-shuffle-null', action='store_true', default=False,
                   help='Harris session-permutation nulls for '
                        'choice_stim*/choice_duringstim*: stratify stim×prior '
                        'on the real session only; null labels = another eid\'s '
                        'choices at the same trial indices (default: label shuffle)')
    p.add_argument('--actkernel-choice-null', action='store_true', default=False,
                   help='Enable ActionKernel synthetic-choice nulls for '
                        'choice_stim*/choice_duringstim* (default mode: strat). '
                        'Takes precedence over --session-shuffle-null')
    p.add_argument('--actkernel-null-mode', default=None,
                   choices=['strat', 'fixedstim', 'unconstrained'],
                   help='AK null variant: strat=stim×block–stratified pseudo '
                        '(opt 1); fixedstim=real stim×block sequence (opt 2); '
                        'unconstrained=legacy calendar-index pseudo. '
                        'Implies AK null even without --actkernel-choice-null')
    p.add_argument('--actkernel-pseudo-len-factor', type=float, default=None,
                   help='Strat only: BWM pseudo length = factor × real n_trials '
                        '(default 3, or env ACTKERNEL_PSEUDO_LEN_FACTOR). '
                        'On low accept rate doubles up to 16; always '
                        'writes _pseudo_strat outputs')
    p.add_argument('--exclude-sticky-trials', action='store_true', default=False,
                   help='Drop last 20%% of session and tails of perseveration '
                        'runs (≥10 same choice poorly explained by non-0 '
                        'contrast stim; keep first 9 of each run); outputs go '
                        'to manifold/res_excl_sticky')
    p.add_argument('--sticky-late-frac', type=float, default=0.2,
                   help='Fraction of late-session trials to drop when '
                        '--exclude-sticky-trials (default 0.2)')
    p.add_argument('--sticky-min-run', type=int, default=10,
                   help='Min same-choice run length; drop only the run tail '
                        'from this position onward (default 10)')
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

    if (
        args.preset == 'goal3_c0_choice'
        and args.contrasts is not None
        and args.contrasts != [0.0]
    ):
        p.error('goal3_c0_choice is fixed at 0% contrast; omit --contrasts')

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

    if args.build_choice_donors:
        print('ONE cache:', ba.one.cache_dir)
        bank = ba.build_choice_donor_bank(restart=False)
        print(f'Done. {len(bank)} donor eids ->', ba._choice_donors_path())
        return

    if args.cache_only:
        print('ONE cache:', ba.one.cache_dir)
        ba.cache_all_insertions(restart=args.restart)
        print('Done. Insertion cache under:',
              Path(ba.one.cache_dir, 'manifold', 'insertion_cache'))
        return

    splits = _validate_splits(splits)

    if args.finalize_only:
        if args.exclude_sticky_trials:
            ba.configure_excl_sticky_output_dirs(args.one_cache_dir)
        ak_mode = args.actkernel_null_mode
        ba.configure_null_file_suffix(
            actkernel_choice_null=args.actkernel_choice_null,
            session_shuffle_null=(
                args.session_shuffle_null
                and ba._resolve_actkernel_null_mode(
                    args.actkernel_choice_null, ak_mode) is None),
            actkernel_null_mode=ak_mode,
            actkernel_pseudo_len_factor=args.actkernel_pseudo_len_factor,
        )
        print('ONE cache:', ba.one.cache_dir)
        print('Finalize splits:', splits, 'res=', ba.pth_res,
              'suffix=', repr(ba.RES_FILE_SUFFIX))
        for sp in splits:
            ba.finalize_stream_shards(sp)
            out = ba.output_split_name(sp)
            for name in (
                f'{out}.npy', f'{out}_regde.npy',
                f'{out}_all.npy', f'{out}_all_regde.npy',
            ):
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
          'shard:', args.shard_idx, '/', args.n_shards, 'finalize:', finalize,
          'session_shuffle_null:', args.session_shuffle_null,
          'actkernel_choice_null:', args.actkernel_choice_null,
          'actkernel_null_mode:', args.actkernel_null_mode,
          'actkernel_pseudo_len_factor:', args.actkernel_pseudo_len_factor,
          'exclude_sticky_trials:', args.exclude_sticky_trials)

    if args.exclude_sticky_trials:
        ba.configure_excl_sticky_output_dirs(args.one_cache_dir)

    ba.configure_null_file_suffix(
        actkernel_choice_null=args.actkernel_choice_null,
        session_shuffle_null=(
            args.session_shuffle_null
            and ba._resolve_actkernel_null_mode(
                args.actkernel_choice_null, args.actkernel_null_mode) is None),
        actkernel_null_mode=args.actkernel_null_mode,
        actkernel_pseudo_len_factor=args.actkernel_pseudo_len_factor,
    )

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
        session_shuffle_null=args.session_shuffle_null,
        actkernel_choice_null=args.actkernel_choice_null,
        actkernel_null_mode=args.actkernel_null_mode,
        actkernel_pseudo_len_factor=args.actkernel_pseudo_len_factor,
        exclude_sticky_trials=args.exclude_sticky_trials,
        sticky_late_frac=args.sticky_late_frac,
        sticky_min_run=args.sticky_min_run,
    )
    print('Done. Outputs under:', ba.pth_res,
          'suffix=', repr(ba.RES_FILE_SUFFIX))
    if finalize:
        for sp in splits:
            out = ba.output_split_name(sp)
            for name in (
                f'{out}.npy', f'{out}_regde.npy',
                f'{out}_all.npy', f'{out}_all_regde.npy',
            ):
                fp = ba.pth_res / name
                if fp.exists():
                    print(f'  OK {fp} ({fp.stat().st_size/1e6:.1f} MB)')
                else:
                    print(f'  MISSING {fp}')
    elif args.n_shards is not None:
        for sp in splits:
            fp = ba.pth_stream_acc / f'{ba.output_split_name(sp)}.shard{args.shard_idx}.npy'
            if fp.exists():
                print(f'  OK shard {fp} ({fp.stat().st_size/1e6:.1f} MB)')
            else:
                print(f'  MISSING shard {fp}')


if __name__ == '__main__':
    main()
