#!/usr/bin/env python
"""
Goal 2 end-to-end comparison: storage test + original vs stream_pool pipeline.

  python scripts/compare_goal2_pipeline.py
  python scripts/compare_goal2_pipeline.py --nrand 500   # faster smoke test
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import block_analysis_allsplits as ba  # noqa: E402
from brainwidemap import bwm_query  # noqa: E402

SPLIT = 'block_only'
SEED = 123
RES_ORIG = '_compare_orig'
RES_STREAM = '_compare_stream'


def _rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())


def count_files(path: Path, pattern='*.npy') -> int:
    if not path.exists():
        return 0
    return len(list(path.rglob(pattern)))


def clean_artifacts(split: str, pids: list, *, keep_compare: str | None = None):
    '''keep_compare: None | 'orig' | 'stream' — do not delete that compare res dir.'''
    cache_root = Path(ba.one.cache_dir, 'manifold')
    for tag in (RES_ORIG, RES_STREAM):
        if keep_compare == 'orig' and tag == RES_ORIG:
            continue
        if keep_compare == 'stream' and tag == RES_STREAM:
            continue
        d = ba.pth_res / tag
        if d.exists():
            for f in d.glob('*.npy'):
                f.unlink()
    acc = ba._stream_acc_path(split)
    if acc.exists():
        acc.unlink()
    d = cache_root / split
    if d.exists():
        for f in d.glob('*.npy'):
            f.unlink()
    df = bwm_query(ba.one)
    subdf = df[df['pid'].isin(pids)]
    ins = cache_root / 'insertion_cache'
    for eid, probe, _ in subdf[['eid', 'probe_name', 'pid']].values:
        fp = ins / f'{eid}_{probe}.npy'
        if fp.exists():
            fp.unlink()


def compare_final(res_a: Path, res_b: Path, split: str, label_a: str, label_b: str):
    fa, fb = res_a / f'{split}.npy', res_b / f'{split}.npy'
    da, db = res_a / f'{split}_regde.npy', res_b / f'{split}_regde.npy'
    if not all(p.exists() for p in (fa, fb, da, db)):
        missing = [str(p) for p in (fa, fb, da, db) if not p.exists()]
        return [f'missing output: {missing}']

    ra = np.load(fa, allow_pickle=True).item()
    rb = np.load(fb, allow_pickle=True).item()
    regdea = np.load(da, allow_pickle=True).item()
    regdeb = np.load(db, allow_pickle=True).item()

    errs = []
    regs_a, regs_b = set(ra), set(rb)
    if regs_a != regs_b:
        errs.append(f'region keys differ: only_a={regs_a-regs_b} only_b={regs_b-regs_a}')

    for reg in sorted(regs_a & regs_b):
        for key in ('d_euc', 'amp_euc', 'p_euc', 'nclus'):
            va, vb = ra[reg].get(key), rb[reg].get(key)
            if isinstance(va, np.ndarray):
                if not np.allclose(va, vb, rtol=1e-5, atol=1e-8, equal_nan=True):
                    errs.append(f'{reg}/{key} max_diff={np.nanmax(np.abs(va-vb)):.2e}')
            elif isinstance(va, (int, float, np.floating)):
                if not (np.isnan(va) and np.isnan(vb)) and not np.allclose(va, vb, rtol=1e-5, atol=1e-8):
                    errs.append(f'{reg}/{key}: {va} vs {vb}')
            elif va != vb:
                errs.append(f'{reg}/{key}: {va} vs {vb}')
        la, lb = ra[reg].get('lat_euc'), rb[reg].get('lat_euc')
        if (np.isnan(la) and np.isnan(lb)) or np.allclose(la, lb, rtol=1e-5, atol=1e-8, equal_nan=True):
            pass
        else:
            errs.append(f'{reg}/lat_euc: {la} vs {lb}')
        if reg in regdea and reg in regdeb:
            if not np.allclose(regdea[reg], regdeb[reg], rtol=1e-5, atol=1e-8, equal_nan=True):
                errs.append(f'{reg}/regde max_diff={np.nanmax(np.abs(regdea[reg]-regdeb[reg])):.2e}')

    print(f'=== FINAL OUTPUT COMPARE ({label_a} vs {label_b}) ===')
    if errs:
        print(f'FAIL ({len(errs)} diffs)')
        for e in errs[:15]:
            print(' ', e)
    else:
        print('OK — final res/{split}*.npy match within tolerance')
    return errs


def run_storage_test(pids, split: str, nrand: int):
    print(f'=== STORAGE TEST (stream_pool, {split}, {len(pids)} pids, nrand={nrand}) ===')
    clean_artifacts(split, pids)
    cache_root = Path(ba.one.cache_dir, 'manifold')
    ins_cache = cache_root / 'insertion_cache'

    df = bwm_query(ba.one)
    eids_plus = df[df['pid'].isin(pids)][['eid', 'probe_name', 'pid']].values

    peak = _rss_mb()
    t0 = time.perf_counter()
    ba.get_all_d_vars_allsplits(
        [split], eids_plus=eids_plus, control=True, nrand=nrand,
        restart=False, stream_pool=True, save_per_insertion=False,
        save_cache=True,
    )
    wall = time.perf_counter() - t0
    peak = max(peak, _rss_mb())

    ins_bytes = sum(
        (ins_cache / f'{eid}_{probe}.npy').stat().st_size
        for eid, probe, _ in eids_plus
        if (ins_cache / f'{eid}_{probe}.npy').exists()
    )
    stream_acc_bytes = dir_size(ba.pth_stream_acc)
    res_bytes = sum(
        (ba.pth_res / n).stat().st_size
        for n in (f'{split}.npy', f'{split}_regde.npy')
        if (ba.pth_res / n).exists()
    )
    per_ins_files = count_files(cache_root / split)

    print(f'Wall time: {wall:.1f} s ({wall/60:.2f} min)')
    print(f'Peak RSS (parent): {peak:.1f} MB')
    print(f'Insertion cache: {ins_bytes/1e6:.1f} MB ({len(pids)} files)')
    print(f'Stream acc checkpoint: {stream_acc_bytes/1e6:.1f} MB')
    print(f'Final res outputs: {res_bytes/1e6:.1f} MB')
    print(f'Per-split insertion files: {per_ins_files} (expect 0)')
    total = ins_bytes + stream_acc_bytes + res_bytes
    print(f'Total on disk (stream path): {total/1e6:.1f} MB')
    print(f'Legacy would also write ~{len(pids)} per-insertion .npy in manifold/{split}/')
    return {
        'wall_s': wall, 'ins_bytes': ins_bytes, 'stream_acc_bytes': stream_acc_bytes,
        'res_bytes': res_bytes, 'per_ins_files': per_ins_files, 'peak_mb': peak,
    }


def run_stream_pipeline(pids, split: str, nrand: int, seed: int):
    print(f'=== STREAM PIPELINE ({split}, {len(pids)} pids, nrand={nrand}) ===')
    clean_artifacts(split, pids, keep_compare='orig')

    df = bwm_query(ba.one)
    eids_plus = df[df['pid'].isin(pids)][['eid', 'probe_name', 'pid']].values

    orig_res = ba.pth_res
    out_res = Path(orig_res.parent, RES_STREAM)
    out_res.mkdir(parents=True, exist_ok=True)
    ba.pth_stream_acc.mkdir(parents=True, exist_ok=True)

    accum = ba.SplitPoolAccumulator(split)
    peak = _rss_mb()
    t0 = time.perf_counter()

    for eid, probe, pid in eids_plus:
        eid_probe = f'{eid}_{probe}'
        random.seed(seed)
        np.random.seed(seed)
        cache = ba.build_insertion_cache(pid, save=True, restart=False)
        D_ = ba.get_d_vars(
            split, pid, control=True, nrand=nrand, bycontrast=False, cached=cache,
        )
        accum.add(eid_probe, D_)
        accum.save()
        del cache
        gc.collect()
        peak = max(peak, _rss_mb())

    ba.pth_res = out_res
    accum.finalize()
    ba.pth_res = orig_res
    wall = time.perf_counter() - t0
    peak = max(peak, _rss_mb())

    cache_root = Path(ba.one.cache_dir, 'manifold')
    ins_bytes = sum(
        (cache_root / 'insertion_cache' / f'{eid}_{probe}.npy').stat().st_size
        for eid, probe, _ in eids_plus
        if (cache_root / 'insertion_cache' / f'{eid}_{probe}.npy').exists()
    )
    stream_acc_bytes = dir_size(ba.pth_stream_acc)
    res_bytes = dir_size(out_res)
    per_ins_files = count_files(cache_root / split)

    print(f'Wall time: {wall:.1f} s')
    print(f'Peak RSS (parent): {peak:.1f} MB')
    print(f'Insertion cache: {ins_bytes/1e6:.1f} MB')
    print(f'Stream acc: {stream_acc_bytes/1e6:.1f} MB')
    print(f'Final res: {res_bytes/1e6:.1f} MB')
    print(f'Per-split insertion files: {per_ins_files}')

    ba.pth_res = orig_res
    gc.collect()
    return {
        'wall_s': wall, 'ins_bytes': ins_bytes, 'stream_acc_bytes': stream_acc_bytes,
        'res_bytes': res_bytes, 'per_ins_files': per_ins_files, 'peak_mb': peak,
        'res_dir': out_res,
    }


def run_original_pipeline(pids, split: str, nrand: int, seed: int):
    print(f'=== ORIGINAL PIPELINE ({split}, {len(pids)} pids, nrand={nrand}) ===')
    clean_artifacts(split, pids)

    worker = ROOT / 'scripts' / '_original_pipeline_worker.py'
    cmd = [
        sys.executable, str(worker),
        '--split', split,
        '--pids', *pids,
        '--nrand', str(nrand),
        '--seed', str(seed),
        '--res-tag', RES_ORIG,
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    wall = time.perf_counter() - t0
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError(f'original worker failed (exit {proc.returncode})')

    stats = {}
    for line in proc.stdout.splitlines():
        if '=' in line and not line.startswith(' '):
            k, v = line.split('=', 1)
            try:
                stats[k] = float(v)
            except ValueError:
                stats[k] = v

    stats['wall_s'] = wall
    stats['res_dir'] = Path(ba.one.cache_dir, 'manifold', 'res', RES_ORIG)
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n-pids', type=int, default=5)
    p.add_argument('--nrand', type=int, default=2000)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--split', default=SPLIT)
    p.add_argument('--skip-storage', action='store_true')
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    df = bwm_query(ba.one)
    pids = [str(x) for x in df['pid'].values[: args.n_pids]]
    print('ONE cache:', ba.one.cache_dir)
    print('Split:', args.split, '| pids:', len(pids), '| nrand:', args.nrand)
    print()

    storage_stats = None
    if not args.skip_storage:
        storage_stats = run_storage_test(pids, args.split, args.nrand)
        print()

    orig = run_original_pipeline(pids, args.split, args.nrand, args.seed)
    print()
    stream = run_stream_pipeline(pids, args.split, args.nrand, args.seed)
    print()

    errs = compare_final(
        Path(orig['res_dir']), Path(stream['res_dir']),
        args.split, 'original', 'stream_pool',
    )
    print()

    print('=== SUMMARY ===')
    print(f'Original wall: {orig.get("wall_s", orig.get("per_ins_total_s", 0)):.1f} s'
          f'  (per-ins mean {orig.get("per_ins_mean_s", float("nan")):.1f} s,'
          f'  d_var_stacked {orig.get("d_var_stacked_s", float("nan")):.1f} s)')
    print(f'Stream wall:   {stream["wall_s"]:.1f} s')
    if orig.get('wall_s') and stream['wall_s']:
        ratio = orig['wall_s'] / stream['wall_s']
        print(f'Speedup (stream vs original): {ratio:.2f}x' if ratio > 1 else
              f'Slowdown (stream vs original): {1/ratio:.2f}x')

    print()
    print('Disk (intermediate + final):')
    print(f'  Original: {orig.get("per_ins_files_bytes", 0)/1e6:.1f} MB per-ins files +'
          f' {orig.get("res_files_bytes", 0)/1e6:.1f} MB res'
          f'  ({int(orig.get("n_per_ins_files", 0))} per-ins files)')
    print(f'  Stream:   {stream["ins_bytes"]/1e6:.1f} MB insertion_cache +'
          f' {stream["stream_acc_bytes"]/1e6:.1f} MB stream_acc +'
          f' {stream["res_bytes"]/1e6:.1f} MB res'
          f'  ({stream["per_ins_files"]} per-ins files)')
    orig_total = orig.get('per_ins_files_bytes', 0) + orig.get('res_files_bytes', 0)
    stream_total = stream['ins_bytes'] + stream['stream_acc_bytes'] + stream['res_bytes']
    print(f'  Total bytes: original {orig_total/1e6:.1f} MB vs stream {stream_total/1e6:.1f} MB')

    print()
    print(f'Peak RSS: original worker {orig.get("peak_rss_mb", float("nan")):.1f} MB,'
          f' stream parent {stream["peak_mb"]:.1f} MB')

    if storage_stats:
        print()
        print(f'Storage test (stream only): {storage_stats["wall_s"]:.1f} s,'
              f' {storage_stats["ins_bytes"]/1e6:.1f}+{storage_stats["stream_acc_bytes"]/1e6:.1f}+'
              f'{storage_stats["res_bytes"]/1e6:.1f} MB')

    return 1 if errs else 0


if __name__ == '__main__':
    raise SystemExit(main())
