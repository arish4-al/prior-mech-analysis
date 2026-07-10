#!/usr/bin/env python
"""
Goal 2 integration test: nrand=2000 parity (cached vs uncached) + time/storage.

    python scripts/test_goal2_nrand2000.py
    python scripts/test_goal2_nrand2000.py --parity-splits block_only
"""
from __future__ import annotations

import argparse
import gc
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import block_analysis_allsplits as ba  # noqa: E402
from brainwidemap import bwm_query  # noqa: E402
from validate_goal2_cache import compare_d  # noqa: E402

PARITY_SPLITS = ['block_only', 'block_duringstim_l_choice_l_f1']
BENCH_SPLITS = [
    'block_only',
    'act_block_only',
    'block_duringstim_l_choice_l_f1',
    'block_duringstim_r_choice_l_f2',
    'block_stim_l_duringchoice_l_f1',
    'block_stim_r_duringchoice_l_f2',
    'act_block_duringstim_l_choice_l_f1',
    'act_block_duringstim_r_choice_l_f2',
    'act_block_stim_l_duringchoice_l_f1',
    'block_stim_l_choice_l_f1',
]
NRAND = 2000
SEED = 123


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())


def count_files(path: Path, pattern='*.npy') -> int:
    if not path.exists():
        return 0
    return len(list(path.rglob(pattern)))


def run_parity(pids, splits, nrand, seed):
    print(f'=== PARITY cached vs uncached (nrand={nrand}, {len(pids)} pids, {len(splits)} splits) ===')
    t0 = time.perf_counter()
    failures = []
    for i, pid in enumerate(pids, 1):
        print(f'--- pid {i}/{len(pids)} {pid} ---')
        cache = ba.build_insertion_cache(pid, save=False, restart=False)
        for split in splits:
            random.seed(seed)
            np.random.seed(seed)
            t_a = time.perf_counter()
            D_unc = ba.get_d_vars(split, pid, control=True, nrand=nrand, cached=None)
            t_unc = time.perf_counter() - t_a
            random.seed(seed)
            np.random.seed(seed)
            t_b = time.perf_counter()
            D_cached = ba.get_d_vars(split, pid, control=True, nrand=nrand, cached=cache)
            t_cached = time.perf_counter() - t_b
            errs = compare_d(D_unc, D_cached, f'{pid}/{split}')
            tag = 'OK  ' if not errs else 'FAIL'
            print(f'  {tag} {split}  uncached={t_unc:.1f}s cached={t_cached:.1f}s  speedup={t_unc/t_cached:.2f}x')
            if errs:
                failures.extend(errs[:3])
                print('    ', errs[0])
        del cache
        gc.collect()
    elapsed = time.perf_counter() - t0
    print(f'Parity wall time: {elapsed/60:.2f} min, failures={len(failures)}')
    return failures, elapsed


def bench_insertion(pid, splits, nrand, mode):
    '''mode: uncached | cached | cached_stream'''
    t0 = time.perf_counter()
    if mode == 'uncached':
        for split in splits:
            ba.get_d_vars(split, pid, control=True, nrand=nrand, cached=None)
            gc.collect()
        return time.perf_counter() - t0, 0.0

    t_load = time.perf_counter()
    cache = ba.build_insertion_cache(pid, save=False, restart=False)
    t_load = time.perf_counter() - t_load
    for split in splits:
        ba.get_d_vars(split, pid, control=True, nrand=nrand, cached=cache)
    del cache
    gc.collect()
    return time.perf_counter() - t0, t_load


def run_time_benchmark(pids, splits, nrand):
    print(f'=== TIME BENCHMARK (nrand={nrand}, {len(splits)} splits, {len(pids)} pids) ===')
    unc, cached, load_only = [], [], []
    for i, pid in enumerate(pids, 1):
        print(f'--- pid {i}/{len(pids)} ---')
        t_u, _ = bench_insertion(pid, splits, nrand, 'uncached')
        t_c, t_l = bench_insertion(pid, splits, nrand, 'cached')
        unc.append(t_u)
        cached.append(t_c)
        load_only.append(t_l)
        print(f'  uncached={t_u:.1f}s  cached={t_c:.1f}s (load={t_l:.1f}s)  speedup={t_u/t_c:.2f}x')
    mean_u, mean_c = np.mean(unc), np.mean(cached)
    print(f'Mean per insertion: uncached={mean_u:.1f}s cached={mean_c:.1f}s speedup={mean_u/mean_c:.2f}x')
    print(f'Total {len(pids)} insertions: {sum(unc)/60:.1f} min vs {sum(cached)/60:.1f} min')
    return {'uncached': unc, 'cached': cached, 'load': load_only}


def run_storage_test(pids, splits, nrand):
    print(f'=== STORAGE (stream_pool, nrand={nrand}, {len(pids)} pids, {len(splits)} splits) ===')
    cache_root = Path(ba.one.cache_dir, 'manifold')
    ins_cache = cache_root / 'insertion_cache'
    stream_acc = ba.pth_stream_acc
    res_dir = ba.pth_res

    # clean prior test artifacts for these splits
    for split in splits:
        acc = ba._stream_acc_path(split)
        if acc.exists():
            acc.unlink()
        for name in (f'{split}.npy', f'{split}_regde.npy'):
            p = res_dir / name
            if p.exists():
                p.unlink()
        split_dir = cache_root / split
        if split_dir.exists():
            for f in split_dir.glob('*.npy'):
                f.unlink()

    df = bwm_query(ba.one)
    sub = df[df['pid'].isin(pids)]
    eids_plus = sub[['eid', 'probe_name', 'pid']].values

    t0 = time.perf_counter()
    ba.get_all_d_vars_allsplits(
        splits, eids_plus=eids_plus, control=True, nrand=nrand,
        restart=False, stream_pool=True, save_per_insertion=False,
        save_cache=True,
    )
    wall = time.perf_counter() - t0

    n_ins = len(pids)
    n_splits = len(splits)
    per_ins_files = sum(count_files(cache_root / sp) for sp in splits)
    sizes = {
        'insertion_cache_total': dir_size(ins_cache),
        'insertion_cache_new': n_ins * 30e6,  # estimate if mixed with prior
        'stream_acc': dir_size(stream_acc),
        'res_outputs': sum(
            (res_dir / f'{sp}.npy').stat().st_size
            + (res_dir / f'{sp}_regde.npy').stat().st_size
            for sp in splits
            if (res_dir / f'{sp}.npy').exists()
        ),
        'per_split_insertion_files': per_ins_files,
    }
    # measure actual insertion cache files for our pids only
    ins_bytes = 0
    for eid, probe, pid in eids_plus:
        p = ins_cache / f'{eid}_{probe}.npy'
        if p.exists():
            ins_bytes += p.stat().st_size

    legacy_per_split_est = per_ins_files  # 0 with stream_pool
    legacy_hypothetical = ins_bytes * 0  # no per-split if stream

    print(f'Wall time (stream_pool): {wall/60:.2f} min')
    print(f'Insertion cache (test pids): {ins_bytes/1e6:.1f} MB ({n_ins} files)')
    print(f'Stream acc checkpoints: {sizes["stream_acc"]/1e6:.1f} MB')
    print(f'Final res/ outputs: {sizes["res_outputs"]/1e6:.1f} MB ({n_splits} splits)')
    print(f'Per-split insertion files written: {per_ins_files} (0 = stream_pool OK)')
    print()
    print('Storage comparison (test scope):')
    print(f'  WITH stream_pool:    {ins_bytes + sizes["stream_acc"] + sizes["res_outputs"]:.0f} bytes')
    print(f'    insertion_cache + stream_acc + res only')
    if per_ins_files == 0:
        print(f'  WITHOUT stream_pool: would add ~{n_ins * n_splits} per-insertion .npy files')
        print(f'    (each small vs 30MB cache, but {n_ins*n_splits} files = inode bloat)')
    return sizes, wall


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n-pids', type=int, default=5)
    p.add_argument('--nrand', type=int, default=NRAND)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--parity-splits', nargs='*', default=PARITY_SPLITS)
    p.add_argument('--bench-splits', nargs='*', default=BENCH_SPLITS)
    p.add_argument('--skip-parity', action='store_true')
    p.add_argument('--skip-bench', action='store_true')
    p.add_argument('--skip-storage', action='store_true')
    args = p.parse_args()

    df = bwm_query(ba.one)
    pids = list(df['pid'].values[: args.n_pids])
    print('ONE cache:', ba.one.cache_dir)
    print('Test pids:', len(pids), 'nrand:', args.nrand)
    print('null_batch_size:', ba.NULL_BATCH_SIZE)
    print()

    if not args.skip_parity:
        failures, _ = run_parity(pids, args.parity_splits, args.nrand, args.seed)
        if failures:
            print('PARITY FAILED')
            return 1
        print('PARITY PASSED')
        print()

    if not args.skip_bench:
        run_time_benchmark(pids, args.bench_splits, args.nrand)
        print()

    if not args.skip_storage:
        run_storage_test(pids, args.bench_splits, args.nrand)

    print('=== ALL TESTS DONE ===')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
