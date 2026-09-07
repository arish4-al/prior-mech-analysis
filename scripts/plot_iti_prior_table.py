#!/usr/bin/env python
"""Single-split ITI prior table: BH-FDR on p_mean from *_regde.

Default = shuffle (plain) act_block_only at FDR 0.05.

  conda activate iblenv
  python scripts/plot_iti_prior_table.py
  python scripts/plot_iti_prior_table.py --split act_block_only --alpha 0.05
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import analysis_functions as af  # noqa: E402


ALYX = Path(
    '/Users/ariliu/Downloads/ONE/alyx.internationalbrainlab.org'
)
DEFAULT_RES = ALYX / 'manifold' / 'res' / 'new'
DEFAULT_META = ALYX / 'meta'


def _p_mean_from_regde(curves) -> float:
    r = np.asarray(curves, dtype=float)
    if r.ndim == 1:
        r = r.reshape(1, -1)
    return float(np.mean(np.mean(r, axis=1) >= np.mean(r[0])))


def _amp_euc(curves, pooled=None) -> float:
    if pooled is not None and 'amp_euc' in pooled:
        return float(pooled['amp_euc'])
    obs = np.asarray(curves[0], dtype=float).reshape(-1)
    return float(np.max(obs) - np.min(obs))


def region_table(pth_res: Path, split: str, alpha: float) -> pd.DataFrame:
    regde = np.load(pth_res / f'{split}_regde.npy', allow_pickle=True).item()
    pooled_path = pth_res / f'{split}.npy'
    pooled = {}
    if pooled_path.exists():
        pooled = np.load(pooled_path, allow_pickle=True).item()
    rows = []
    for reg, curves in regde.items():
        rec = pooled.get(reg, {})
        rows.append({
            'region': reg,
            'nclus': int(rec['nclus']) if rec.get('nclus') is not None else np.nan,
            'p_mean': _p_mean_from_regde(curves),
            'amp_euc': _amp_euc(curves, rec),
        })
    df = pd.DataFrame(rows)
    _, p_c, _, _ = multipletests(df['p_mean'].to_numpy(), alpha=alpha, method='fdr_bh')
    df['p_mean_c'] = p_c
    df['significant'] = (df['p_mean_c'] <= alpha).astype(int)
    return df.sort_values(['p_mean_c', 'p_mean', 'region']).reset_index(drop=True)


def _region_order(meta_dir: Path, regions: list[str]) -> list[str]:
    for path in (meta_dir / 'region_order.txt',
                 Path('/Users/ariliu/Downloads/ONE/openalyx.internationalbrainlab.org/meta/region_order.txt')):
        if path.exists():
            order = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
            present = set(regions)
            return [r for r in order if r in present] + [r for r in regions if r not in set(order)]
    return list(regions)


def plot_simple_amp_table(df: pd.DataFrame, out_path: Path, col_name: str):
    plot = df.copy()
    amp = plot['amp_euc'].to_numpy(dtype=float)
    lo, hi = float(np.nanmin(amp)), float(np.nanmax(amp))
    denom = (hi - lo) if hi > lo else 1.0
    plot[col_name] = ((amp - lo) / denom + 1e-4) * plot['significant']
    plot['beryl_hex'] = plot['region'].apply(af.swanson_to_beryl_hex, args=[af.br])
    beryl_palette = dict(zip(plot['region'], plot['beryl_hex']))
    order = _region_order(out_path.parent, plot['region'].tolist())
    plot['region'] = pd.Categorical(plot['region'], categories=order, ordered=True)
    plot = plot.sort_values('region')
    df_to_plot = plot[['region', col_name]].reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    af.plot_table_with_styles(
        df=df_to_plot,
        beryl_palette=beryl_palette,
        colormap_lookup={col_name: af.get_cmap_(col_name)},
        out_path=out_path,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='act_block_only')
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--res-dir', type=Path, default=DEFAULT_RES)
    p.add_argument('--meta-dir', type=Path, default=DEFAULT_META)
    p.add_argument('--tag', default='shuffle',
                   help='null tag in the filename (shuffle = plain {split}.npy)')
    args = p.parse_args()

    df = region_table(args.res_dir, args.split, args.alpha)
    n_sig = int(df['significant'].sum())
    stem = f'table_{args.split}_{args.tag}_p_mean_c_{args.alpha:g}'
    csv_path = args.meta_dir / f'{stem}.csv'
    png_path = args.meta_dir / f'{stem}.png'
    args.meta_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    plot_simple_amp_table(df, png_path, col_name=args.split)
    sig = df.loc[df['significant'] == 1, 'region'].tolist()
    print(f'{args.split} {args.tag}: {n_sig}/{len(df)} FDR p_mean_c ≤ {args.alpha:g}')
    print('  sig:', ', '.join(sig) if sig else '(none)')
    print(f'  {png_path}')
    print(f'  {csv_path}')


if __name__ == '__main__':
    main()
