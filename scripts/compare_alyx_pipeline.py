#!/usr/bin/env python
"""
Compare original vs new stream_pool pipeline on alyx ONE cache.

  # Second split only + 2-split aggregate (block_only from prior run):
  python scripts/compare_alyx_pipeline.py --split block_duringstim_l_choice_l_f1

  # Re-run a single split from scratch:
  python scripts/compare_alyx_pipeline.py --split block_only --run-both
"""
from __future__ import annotations

import argparse
import gc
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from one.api import ONE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import block_analysis_allsplits as ba  # noqa: E402
from brainwidemap import bwm_query  # noqa: E402

SEED = 123
RES_NEW = '_compare_stream'
SPLIT1 = 'block_only'
SPLIT2_DEFAULT = 'block_duringstim_l_choice_l_f1'

# Prior alyx runs (nrand=2000, 5 insertions, seed=123).
SPLIT1_ORIG = {
    'split': SPLIT1, 'wall_s': 415.1, 'per_ins_mean_s': 82.8, 'finalize_s': 0.7,
    'peak_rss_mb': 3735.4, 'per_ins_bytes': 231231375, 'res_bytes': 24098552,
}
SPLIT1_NEW = {
    'split': SPLIT1, 'wall_s': 488.8, 'per_ins_mean_s': 97.7, 'finalize_s': 0.1,
    'peak_rss_mb': 2422.5, 'ins_bytes': 322400000, 'stream_acc_bytes': 135000000,
    'res_bytes': 24098552, 'parity_ok': True,
}


def _use_alyx():
    ba.one = ONE(base_url='https://alyx.internationalbrainlab.org')
    ba.pth_res = Path(ba.one.cache_dir, 'manifold', 'res')
    ba.pth_res.mkdir(parents=True, exist_ok=True)
    ba.pth_stream_acc = ba.pth_res / '_stream_acc'
    ba.pth_stream_acc.mkdir(parents=True, exist_ok=True)


def _rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())


def default_pids() -> list[str]:
    return [
        '8dfb86c8-d45c-46c4-90ec-33078014d434',
        '6be21156-33b0-4f70-9a0f-65b3e3cd6d4a',
        '47be9ae4-290f-46ab-b047-952bc3a1a509',
        '56f2a378-78d2-4132-b3c8-8c1ba82be598',
        'c893c0a3-5597-49cf-baa1-60efdfdef542',
    ]


def compare_final(orig_res: Path, new_res: Path, split: str) -> list[str]:
    fa, fb = orig_res / f'{split}.npy', new_res / f'{split}.npy'
    da, db = orig_res / f'{split}_regde.npy', new_res / f'{split}_regde.npy'
    ra = np.load(fa, allow_pickle=True).item()
    rb = np.load(fb, allow_pickle=True).item()
    regdea = np.load(da, allow_pickle=True).item()
    regdeb = np.load(db, allow_pickle=True).item()

    errs = []
    common = sorted(set(ra) & set(rb))
    for reg in common:
        for key in ('d_euc', 'amp_euc', 'p_euc', 'nclus'):
            va, vb = ra[reg][key], rb[reg][key]
            if isinstance(va, np.ndarray):
                if not np.allclose(va, vb, rtol=1e-5, atol=1e-8, equal_nan=True):
                    errs.append(f'{reg}/{key} max={np.nanmax(np.abs(va-vb)):.2e}')
            elif not (np.isnan(va) and np.isnan(vb)) and va != vb:
                if not np.allclose(float(va), float(vb), rtol=1e-5, atol=1e-8):
                    errs.append(f'{reg}/{key}: {va} vs {vb}')
        la, lb = ra[reg]['lat_euc'], rb[reg]['lat_euc']
        if not ((np.isnan(la) and np.isnan(lb)) or np.allclose(la, lb, rtol=1e-5, atol=1e-8, equal_nan=True)):
            errs.append(f'{reg}/lat_euc: {la} vs {lb}')
        if reg in regdea and reg in regdeb:
            if not np.allclose(regdea[reg], regdeb[reg], rtol=1e-5, atol=1e-8, equal_nan=True):
                errs.append(f'{reg}/regde max={np.nanmax(np.abs(regdea[reg]-regdeb[reg])):.2e}')

    print(f'=== FINAL COMPARE {split} (res/ vs _compare_stream/) ===')
    if errs:
        print(f'FAIL — {len(errs)} differences')
        for e in errs[:15]:
            print(' ', e)
    else:
        print(f'OK — {len(common)} regions match')
    return errs


def run_original_pipeline(pids, split: str, nrand: int, seed: int) -> dict:
    print(f'=== ORIGINAL {split} ===')
    worker = ROOT / 'scripts' / '_original_pipeline_worker.py'
    cmd = [
        sys.executable, str(worker),
        '--split', split, '--pids', *pids,
        '--nrand', str(nrand), '--seed', str(seed), '--res-tag', 'default',
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    wall = time.perf_counter() - t0
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError(f'original worker failed: {proc.returncode}')

    stats = {'split': split, 'wall_s': wall, 'parity_ok': None}
    for line in proc.stdout.splitlines():
        if '=' in line and line[0].isalpha():
            k, v = line.split('=', 1)
            try:
                stats[k] = float(v)
            except ValueError:
                stats[k] = v
    stats['per_ins_bytes'] = stats.get('per_ins_files_bytes', 0)
    stats['res_bytes'] = stats.get('res_files_bytes', 0)
    stats['finalize_s'] = stats.get('d_var_stacked_s', 0)
    return stats


def run_new_pipeline(pids, split: str, nrand: int, seed: int, *, reuse_ins_cache: bool) -> dict:
    print(f'=== NEW (stream_pool) {split}  reuse_cache={reuse_ins_cache} ===')
    cache_root = Path(ba.one.cache_dir, 'manifold')
    ins_cache = cache_root / 'insertion_cache'
    pth_split = cache_root / split

    acc = ba._stream_acc_path(split)
    if acc.exists():
        acc.unlink()
    out_res = ba.pth_res / RES_NEW
    out_res.mkdir(parents=True, exist_ok=True)
    for name in (f'{split}.npy', f'{split}_regde.npy'):
        fp = out_res / name
        if fp.exists():
            fp.unlink()

    if not reuse_ins_cache:
        df = bwm_query(ba.one)
        for eid, probe, pid in df[df['pid'].isin(pids)][['eid', 'probe_name', 'pid']].values:
            fp = ins_cache / f'{eid}_{probe}.npy'
            if fp.exists():
                fp.unlink()

    df = bwm_query(ba.one)
    eids_plus = df[df['pid'].isin(pids)][['eid', 'probe_name', 'pid']].values

    accum = ba.SplitPoolAccumulator(split)
    peak = _rss_mb()
    per_ins_times = []
    cache_load_times = []
    t0 = time.perf_counter()

    for eid, probe, pid in eids_plus:
        random.seed(seed)
        np.random.seed(seed)
        t_ins = time.perf_counter()
        t_cache = time.perf_counter()
        cache = ba.build_insertion_cache(pid, save=True, restart=reuse_ins_cache)
        cache_load_times.append(time.perf_counter() - t_cache)
        D_ = ba.get_d_vars(split, pid, control=True, nrand=nrand, bycontrast=False, cached=cache)
        accum.add(f'{eid}_{probe}', D_)
        accum.save()
        per_ins_times.append(time.perf_counter() - t_ins)
        del cache
        gc.collect()
        peak = max(peak, _rss_mb())

    orig_res = ba.pth_res
    ba.pth_res = out_res
    t_fin = time.perf_counter()
    accum.finalize()
    finalize_s = time.perf_counter() - t_fin
    ba.pth_res = orig_res
    wall = time.perf_counter() - t0
    peak = max(peak, _rss_mb())

    ins_bytes = sum(
        (ins_cache / f'{eid}_{probe}.npy').stat().st_size
        for eid, probe, _ in eids_plus
        if (ins_cache / f'{eid}_{probe}.npy').exists()
    )
    stream_acc_bytes = sum(
        f.stat().st_size for f in ba.pth_stream_acc.glob('*.npy') if f.exists()
    )
    res_bytes = sum(
        (out_res / n).stat().st_size
        for n in (f'{split}.npy', f'{split}_regde.npy')
        if (out_res / n).exists()
    )

    stats = {
        'split': split,
        'wall_s': wall,
        'per_ins_mean_s': float(np.mean(per_ins_times)),
        'per_ins_total_s': float(sum(per_ins_times)),
        'cache_load_mean_s': float(np.mean(cache_load_times)),
        'finalize_s': finalize_s,
        'peak_rss_mb': peak,
        'ins_bytes': ins_bytes,
        'stream_acc_bytes': stream_acc_bytes,
        'res_bytes': res_bytes,
        'per_ins_files': len(list(pth_split.glob('*.npy'))),
        'res_dir': out_res,
    }
    print(f'Wall: {wall:.1f} s | per-ins mean: {stats["per_ins_mean_s"]:.1f} s | cache load mean: {stats["cache_load_mean_s"]:.1f} s')
    print(f'Finalize: {finalize_s:.1f} s | Peak RSS: {peak:.1f} MB')
    print(f'Cache: {ins_bytes/1e6:.1f} MB | stream_acc (all splits): {stream_acc_bytes/1e6:.1f} MB | res: {res_bytes/1e6:.1f} MB')
    return stats


def print_two_split_aggregate(orig1, new1, orig2, new2):
    print()
    print('=== TWO-SPLIT AGGREGATE (same 5 insertions, nrand=2000, alyx) ===')
    print(f'Splits: {orig1["split"]} + {orig2["split"]}')
    print()
    print('| Metric | Original | New (stream_pool) |')
    print('|--------|----------|-------------------|')
    o_time = orig1['wall_s'] + orig2['wall_s']
    n_time = new1['wall_s'] + new2['wall_s']
    print(f'| Total wall time | {o_time:.1f} s ({o_time/60:.1f} min) | {n_time:.1f} s ({n_time/60:.1f} min) |')
    print(f'| Speedup (new) | — | {o_time/n_time:.2f}x |')
    print(f'| Peak RSS (max of runs) | {max(orig1["peak_rss_mb"], orig2["peak_rss_mb"]):.0f} MB | {max(new1["peak_rss_mb"], new2["peak_rss_mb"]):.0f} MB |')

    o_disk = orig1['per_ins_bytes'] + orig2['per_ins_bytes'] + orig1['res_bytes'] + orig2['res_bytes']
    # New: shared insertion cache + stream_acc for both splits + 2 final res
    n_disk = new2['ins_bytes'] + new2['stream_acc_bytes'] + new1['res_bytes'] + new2['res_bytes']
    print(f'| Total disk | {o_disk/1e6:.1f} MB | {n_disk/1e6:.1f} MB |')
    print(f'| Per-split insertion files | 10 (5×2) | 5 (split1 orig only; new writes 0) |')
    print()
    print('Per-split breakdown:')
    for label, o, n in [('Split 1', orig1, new1), ('Split 2', orig2, new2)]:
        print(f'  {label} ({o["split"]}): orig {o["wall_s"]:.1f}s peak {o["peak_rss_mb"]:.0f}MB | '
              f'new {n["wall_s"]:.1f}s peak {n["peak_rss_mb"]:.0f}MB')
    print()
    print('New path split-2 cache load mean:', f'{new2.get("cache_load_mean_s", 0):.1f}s',
          '(restart=True reuses insertion_cache from split 1)')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', default=SPLIT2_DEFAULT)
    p.add_argument('--nrand', type=int, default=2000)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--run-original', action='store_true')
    p.add_argument('--run-new', action='store_true')
    p.add_argument('--run-both', action='store_true')
    p.add_argument('--two-split-summary', action='store_true', default=True)
    p.add_argument('--no-two-split-summary', dest='two_split_summary', action='store_false')
    args = p.parse_args()

    run_orig = args.run_original or args.run_both or (not args.run_original and not args.run_new)
    run_new = args.run_new or args.run_both or (not args.run_original and not args.run_new)

    _use_alyx()
    pids = default_pids()
    print('ONE cache (alyx):', ba.one.cache_dir)
    print('Split:', args.split, '| pids:', len(pids), '| nrand:', args.nrand)
    print()

    orig2 = new2 = None
    errs = []

    if run_orig:
        orig2 = run_original_pipeline(pids, args.split, args.nrand, args.seed)
        print()

    if run_new:
        reuse = args.split != SPLIT1  # split2 reuses cache built for split1
        new2 = run_new_pipeline(pids, args.split, args.nrand, args.seed, reuse_ins_cache=reuse)
        print()

    if run_orig and run_new:
        errs = compare_final(ba.pth_res, new2['res_dir'], args.split)
        new2['parity_ok'] = len(errs) == 0
        print()

    if args.two_split_summary and orig2 and new2:
        print_two_split_aggregate(SPLIT1_ORIG, SPLIT1_NEW, orig2, new2)

    return 1 if errs else 0


if __name__ == '__main__':
    raise SystemExit(main())
