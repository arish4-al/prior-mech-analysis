#!/usr/bin/env python
"""
Re-finalize Harris unique-null splits from stream_acc.

Looks for ``_stream_acc/{split}_harris_unique*.npy``. Drops exact-duplicate
null distance rows within each insertion, then writes pooled
``{split}_harris_unique*.npy`` (never touches legacy ``_harris``).

  conda activate iblenv
  python scripts/recompute_harris_unique_from_stream.py \\
    --res-dir ~/Downloads/ONE/alyx.internationalbrainlab.org/manifold/res

Then replot:
  python scripts/plot_choice_null_comparison_table.py \\
    --arm-res .../res --arm-tag harris_unique --force-combine --alpha 0.01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import block_analysis_allsplits as ba  # noqa: E402

HARRIS_SPLITS = [
    'choice_duringstim_r_block_r_act',
    'choice_duringstim_l_block_l_act',
    'choice_duringstim_r_block_l_act',
    'choice_duringstim_l_block_r_act',
    'choice_stim_r_block_r_act',
    'choice_stim_l_block_l_act',
    'choice_stim_r_block_l_act',
    'choice_stim_l_block_r_act',
]

SUFFIX = '_harris_unique'


def _dedupe_null_rows(arr: np.ndarray) -> np.ndarray:
    """Keep obs row + unique null rows (exact float equality)."""
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return arr
    out = [arr[0]]
    seen = {np.ascontiguousarray(arr[0]).tobytes()}
    for row in arr[1:]:
        key = np.ascontiguousarray(row).tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return np.asarray(out)


def _dedupe_accumulator(acc: ba.SplitPoolAccumulator) -> dict:
    stats = {'insertions': len(acc.acs), 'regs': 0, 'before': 0, 'after': 0}
    for store in (acc.regdv0, acc.regde0):
        for reg, arrs in store.items():
            new_arrs = []
            for a in arrs:
                a = np.asarray(a)
                stats['before'] += max(a.shape[0] - 1, 0)
                d = _dedupe_null_rows(a)
                stats['after'] += max(d.shape[0] - 1, 0)
                new_arrs.append(d)
            store[reg] = new_arrs
            stats['regs'] += 1
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--res-dir', type=Path,
        default=Path.home() / (
            'Downloads/ONE/alyx.internationalbrainlab.org/manifold/res'),
        help='manifold/res[/new] containing _stream_acc/',
    )
    ap.add_argument('--no-cleanup', action='store_true',
                    help='Keep stream_acc after rewrite')
    args = ap.parse_args()

    ba.pth_res = args.res_dir
    ba.pth_stream_acc = ba.pth_res / '_stream_acc'
    ba.pth_stream_acc.mkdir(parents=True, exist_ok=True)
    ba.RES_FILE_SUFFIX = SUFFIX

    any_ok = False
    for split in HARRIS_SPLITS:
        paths = []
        p0 = ba._stream_acc_path(split)
        if p0.exists():
            paths.append(p0)
        paths.extend(ba._stream_acc_shard_paths(split))
        if not paths:
            print(f'SKIP {split}: no stream_acc under {ba.pth_stream_acc}')
            continue
        print(f'Loading {split}: {len(paths)} stream file(s)')
        acc = ba.merge_stream_acc_shards(split)
        stats = _dedupe_accumulator(acc)
        print(f'  insertions={stats["insertions"]}  '
              f'null rows before={stats["before"]} after={stats["after"]}')
        acc.finalize(save=True, cleanup_checkpoint=not args.no_cleanup)
        out = ba.pth_res / f'{ba.output_split_name(split)}.npy'
        print(f'  wrote {out}')
        # Safety: never write legacy _harris basename
        if out.name.endswith('_harris.npy') and not out.name.endswith(
                '_harris_unique.npy'):
            raise SystemExit(f'Refused to write legacy path {out}')
        any_ok = True

    if not any_ok:
        raise SystemExit(
            f'No {SUFFIX} stream_acc found under {ba.pth_stream_acc}. '
            'Re-run Harris unique-null job '
            '(NULL_SCHEME=harris_unique); legacy _harris is left alone.'
        )
    print('Done. Replot with --arm-tag harris_unique --force-combine')


if __name__ == '__main__':
    main()
