#!/usr/bin/env python
'''Isolated worker: original (ee849e0) per-insertion + d_var_stacked pipeline.'''
from __future__ import annotations

import argparse
import gc
import importlib.util
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_original_module():
    src = subprocess.check_output(
        ['git', 'show', 'ee849e0:block_analysis_allsplits.py'],
        text=True, cwd=ROOT,
    )
    cut = src.find('\nrestart = True')
    if cut > 0:
        src = src[:cut]
    path = ROOT / 'scripts' / '_original_ba_stripped.py'
    path.write_text(src)
    spec = importlib.util.spec_from_file_location('original_ba', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', required=True)
    p.add_argument('--pids', nargs='+', required=True)
    p.add_argument('--nrand', type=int, default=2000)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--res-tag', default='default',
                   help='subdir under manifold/res/ (default: standard res/)')
    args = p.parse_args()

    from brainwidemap import bwm_query

    oba = _load_original_module()
    df = bwm_query(oba.one)
    sub = df[df['pid'].isin(args.pids)]
    eids_plus = sub[['eid', 'probe_name', 'pid']].values

    pth_split = Path(oba.one.cache_dir, 'manifold', args.split)
    pth_split.mkdir(parents=True, exist_ok=True)
    for f in pth_split.glob('*.npy'):
        f.unlink()

    if args.res_tag == 'default':
        oba.pth_res = Path(oba.one.cache_dir, 'manifold', 'res')
    else:
        oba.pth_res = Path(oba.one.cache_dir, 'manifold', 'res', args.res_tag)
    oba.pth_res.mkdir(parents=True, exist_ok=True)
    for name in (f'{args.split}.npy', f'{args.split}_regde.npy'):
        fp = oba.pth_res / name
        if fp.exists():
            fp.unlink()

    t0 = time.perf_counter()
    peak_rss = _rss_mb()
    per_ins_times = []

    for eid, probe, pid in eids_plus:
        random.seed(args.seed)
        np.random.seed(args.seed)
        t_ins = time.perf_counter()
        D_ = oba.get_d_vars(
            args.split, pid, control=True, nrand=args.nrand,
            bycontrast=False,
        )
        eid_probe = f'{eid}_{probe}'
        np.save(pth_split / f'{eid_probe}.npy', D_, allow_pickle=True)
        per_ins_times.append(time.perf_counter() - t_ins)
        peak_rss = max(peak_rss, _rss_mb())
        gc.collect()

    t_stack0 = time.perf_counter()
    oba.d_var_stacked(args.split)
    stack_time = time.perf_counter() - t_stack0
    peak_rss = max(peak_rss, _rss_mb())
    wall = time.perf_counter() - t0

    per_ins_bytes = sum(f.stat().st_size for f in pth_split.glob('*.npy'))
    res_bytes = sum(
        (oba.pth_res / n).stat().st_size
        for n in (f'{args.split}.npy', f'{args.split}_regde.npy')
        if (oba.pth_res / n).exists()
    )

    print('PIPELINE_ORIGINAL_DONE')
    print(f'wall_s={wall:.1f}')
    print(f'per_ins_mean_s={np.mean(per_ins_times):.1f}')
    print(f'per_ins_total_s={sum(per_ins_times):.1f}')
    print(f'd_var_stacked_s={stack_time:.1f}')
    print(f'peak_rss_mb={peak_rss:.1f}')
    print(f'per_ins_files_bytes={per_ins_bytes}')
    print(f'res_files_bytes={res_bytes}')
    print(f'n_per_ins_files={len(list(pth_split.glob("*.npy")))}')
    print(f'res_dir={oba.pth_res}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
