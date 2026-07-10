#!/usr/bin/env python
"""Benchmark Goal 2 cache: old (reload per split) vs new (load once, 10 splits)."""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import block_analysis_allsplits as ba  # noqa: E402
from brainwidemap import bwm_query  # noqa: E402

# 10 representative splits (mix of stim / choice / act / block_only)
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


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())


def bench_old(pid: str, splits: list[str]) -> float:
    """Simulate old driver: full reload for every split."""
    t0 = time.perf_counter()
    for split in splits:
        ba.get_d_vars(split, pid, control=False, cached=None)
        gc.collect()
    return time.perf_counter() - t0


def bench_new(pid: str, splits: list[str], save_cache: bool) -> tuple[float, float]:
    """New driver: one load, then all splits from cache."""
    t0 = time.perf_counter()
    cache = ba.build_insertion_cache(pid, save=save_cache, restart=False)
    t_load = time.perf_counter() - t0
    t1 = time.perf_counter()
    for split in splits:
        ba.get_d_vars(split, pid, control=False, cached=cache)
    t_compute = time.perf_counter() - t1
    del cache
    gc.collect()
    return t_load, t_compute


def main():
    n_insertions = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    df = bwm_query(ba.one)
    pids = list(df['pid'].values[:n_insertions])
    n_splits = len(BENCH_SPLITS)

    print('ONE cache:', ba.one.cache_dir)
    print(f'Benchmark: {n_insertions} insertions x {n_splits} splits, control=False')
    print('Splits:', BENCH_SPLITS)
    print()

    old_times, load_times, compute_times = [], [], []
    for i, pid in enumerate(pids, 1):
        print(f'--- insertion {i}/{n_insertions} {pid} ---')
        t_old = bench_old(pid, BENCH_SPLITS)
        t_load, t_compute = bench_new(pid, BENCH_SPLITS, save_cache=(i == 1))
        t_new = t_load + t_compute
        old_times.append(t_old)
        load_times.append(t_load)
        compute_times.append(t_compute)
        speedup = t_old / t_new if t_new > 0 else float('inf')
        print(f'  old (reload/split): {t_old:6.1f}s')
        print(f'  new load once:      {t_load:6.1f}s')
        print(f'  new {n_splits} splits:     {t_compute:6.1f}s  (total {t_new:6.1f}s)')
        print(f'  speedup:            {speedup:6.2f}x')
        print()

    tot_old = sum(old_times)
    tot_new = sum(load_times) + sum(compute_times)
    mean_old = np.mean(old_times)
    mean_new = np.mean(load_times) + np.mean(compute_times)

    cache_dir = Path(ba.one.cache_dir, 'manifold', 'insertion_cache')
    cache_bytes = dir_size(cache_dir)
    n_cached = len(list(cache_dir.glob('*.npy'))) if cache_dir.exists() else 0

    # Per-split output size (one insertion, one split) for reference
    sample_out = None
    eid, probe = ba.one.pid2eid(pids[0])
    for split in BENCH_SPLITS:
        p = Path(ba.one.cache_dir, 'manifold', split, f'{eid}_{probe}.npy')
        if p.exists():
            sample_out = p.stat().st_size
            break

    print('=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'Per insertion (mean):')
    print(f'  Old pattern ({n_splits} full reloads): {mean_old:6.1f}s')
    print(f'  New pattern (1 load + {n_splits} splits): {mean_new:6.1f}s')
    print(f'    load component:    {np.mean(load_times):6.1f}s')
    print(f'    compute component: {np.mean(compute_times):6.1f}s')
    print(f'  Mean speedup: {mean_old / mean_new:.2f}x')
    print()
    print(f'Total ({n_insertions} insertions):')
    print(f'  Old: {tot_old / 60:.2f} min')
    print(f'  New: {tot_new / 60:.2f} min')
    print(f'  Saved: {(tot_old - tot_new) / 60:.2f} min ({100 * (1 - tot_new / tot_old):.1f}%)')
    print()
    print('STORAGE (insertion cache vs redundant loading)')
    print(f'  insertion_cache dir: {cache_bytes / 1e6:.2f} MB ({n_cached} files)')
    if n_cached:
        print(f'  per insertion cache: {cache_bytes / n_cached / 1e6:.2f} MB')
    print(f'  Old pattern re-reads same raw data {n_splits}x per insertion')
    print(f'  Effective redundant I/O avoided: ~{(n_splits - 1) / n_splits * 100:.0f}% of load time')
  # Extrapolate to full BWM (~699 insertions)
    bwm_n = len(df)
    est_old_h = tot_old / n_insertions * bwm_n / 3600
    est_new_h = tot_new / n_insertions * bwm_n / 3600
    est_cache_gb = (cache_bytes / max(n_cached, 1)) * bwm_n / 1e9
    print()
    print(f'EXTRAPOLATION to full BWM ({bwm_n} insertions, {n_splits} splits):')
    print(f'  Old driver est: {est_old_h:.1f} hours')
    print(f'  New driver est: {est_new_h:.1f} hours')
    print(f'  Insertion cache est total: {est_cache_gb:.2f} GB')
    if sample_out:
        out_gb = sample_out * bwm_n * n_splits / 1e9
        print(f'  Per-split outputs (unchanged): ~{out_gb:.2f} GB for all splits')


if __name__ == '__main__':
    main()
