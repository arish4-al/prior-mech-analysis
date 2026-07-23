#!/usr/bin/env python
"""
Plot Swanson-style region tables for Goal-1 var-partition results.

Writes two PNGs (mixed regions colored; others gray):

1. ``table_var_partition_sxp_regtype.png`` — compact:
   region | sxp_stim (green) | regtype
2. ``table_var_partition_mixed.png`` — full:
   region | regtype | unique prior (purple) | stim (blue) |
   choice (orange) | sxp_stim (green)

  conda activate iblenv
  python scripts/plot_var_partition_table.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_ALYX = Path.home() / 'Downloads' / 'ONE' / 'alyx.internationalbrainlab.org'
DEFAULT_OPENALYX = (
    Path.home() / 'Downloads' / 'ONE' / 'openalyx.internationalbrainlab.org'
)
DEFAULT_REGTYPE = ROOT / 'data' / 'stimchoice_act_regtype_regions_p_mean_c_0.01.csv'

# Compact (previous): sxp + regtype only.
COMPACT_COLS = [
    'region',
    'sxp_stim',
    'sc_duringstim_regtype',
]

# Full (current): partition components + ratio.
FULL_COLS = [
    'region',
    'sc_duringstim_regtype',
    'r2_unique_prior',
    'r2_unique_stim',
    'r2_unique_choice',
    'sxp_stim',
]


def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        return s.astype(str).str.lower().isin(('true', '1', '1.0'))
    return s.fillna(False).astype(bool)


def _load_region_order(meta_dir: Path, openalyx_meta: Path) -> list[str]:
    for path in (meta_dir / 'region_order.txt', openalyx_meta / 'region_order.txt'):
        if path.exists():
            return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return []


def build_table(
    vp: pd.DataFrame,
    regtypes: pd.DataFrame,
    region_order: list[str],
    sxp_vmax: float,
    r2_vmax: float,
) -> pd.DataFrame:
    mixed = _as_bool(regtypes['mixed_stim_choice'])
    mixed_regs = set(regtypes.loc[mixed, 'region'].astype(str))

    vp = vp.copy()
    vp['region'] = vp['region'].astype(str)
    stim = vp['r2_unique_stim_mean'].astype(float)
    sxp = vp['r2_stim_x_prior_mean'].astype(float)
    ratio = np.where(stim > 1e-12, sxp / stim, np.nan)
    vp['sxp_stim_raw'] = ratio
    # Color scales: map [0, vmax] → [0, 1] (numbers hidden in the PNG).
    vp['sxp_stim'] = np.clip(ratio / float(sxp_vmax), 0.0, 1.0)
    vp['r2_unique_prior'] = np.clip(
        vp['r2_unique_prior_mean'].astype(float) / float(r2_vmax), 0.0, 1.0)
    vp['r2_unique_stim'] = np.clip(
        vp['r2_unique_stim_mean'].astype(float) / float(r2_vmax), 0.0, 1.0)
    vp['r2_unique_choice'] = np.clip(
        vp['r2_unique_choice_mean'].astype(float) / float(r2_vmax), 0.0, 1.0)

    rt = regtypes.set_index(regtypes['region'].astype(str))
    present = [r for r in region_order if r in set(rt.index)]
    extras = [r for r in rt.index.astype(str) if r not in set(present)]
    ordered = present + extras

    vp_by = vp.set_index('region')
    rows = []
    for reg in ordered:
        if reg in mixed_regs and reg in vp_by.index:
            rt_val = rt.at[reg, 'sc_duringstim_regtype']
            try:
                rt_val = float(rt_val)
            except (TypeError, ValueError):
                rt_val = np.nan
            if pd.isna(rt_val):
                rt_val = np.nan
            row = {
                'region': reg,
                'sc_duringstim_regtype': rt_val,
                'r2_unique_prior': float(vp_by.at[reg, 'r2_unique_prior']),
                'r2_unique_stim': float(vp_by.at[reg, 'r2_unique_stim']),
                'r2_unique_choice': float(vp_by.at[reg, 'r2_unique_choice']),
                'sxp_stim': float(vp_by.at[reg, 'sxp_stim']),
            }
        else:
            row = {
                'region': reg,
                'sc_duringstim_regtype': np.nan,
                'r2_unique_prior': np.nan,
                'r2_unique_stim': np.nan,
                'r2_unique_choice': np.nan,
                'sxp_stim': np.nan,
            }
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--one-cache-dir', type=Path, default=DEFAULT_ALYX)
    p.add_argument('--openalyx-meta', type=Path,
                   default=DEFAULT_OPENALYX / 'meta')
    p.add_argument('--var-partition-csv', type=Path, default=None)
    p.add_argument('--regtype-csv', type=Path, default=DEFAULT_REGTYPE)
    p.add_argument('--sxp-vmax', type=float, default=2.0,
                   help='sxp/stim color ceiling (ratio ≥ this → max color)')
    p.add_argument('--r2-vmax', type=float, default=0.02,
                   help='shared unique-R² color ceiling for prior/stim/choice')
    p.add_argument('--out-dir', type=Path, default=None,
                   help='output directory (default: <cache>/meta)')
    args = p.parse_args()

    cache = args.one_cache_dir.expanduser().resolve()
    meta_dir = cache / 'meta'
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_dir = (args.out_dir or meta_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    vp_path = args.var_partition_csv or (meta_dir / 'var_partition_by_region.csv')
    vp_path = vp_path.expanduser().resolve()
    if not vp_path.exists():
        raise SystemExit(f'missing var-partition CSV: {vp_path}')

    reg_path = args.regtype_csv.expanduser().resolve()
    if not reg_path.exists():
        raise SystemExit(f'missing regtype CSV: {reg_path}')

    print(f'loading {vp_path}', flush=True)
    vp = pd.read_csv(vp_path)
    regtypes = pd.read_csv(reg_path)
    region_order = _load_region_order(meta_dir, args.openalyx_meta.expanduser())

    df = build_table(
        vp, regtypes, region_order,
        sxp_vmax=args.sxp_vmax, r2_vmax=args.r2_vmax,
    )
    n_mixed = int(df['sxp_stim'].notna().sum())
    print(f'regions={len(df)}  mixed colored={n_mixed}  '
          f'sxp_vmax={args.sxp_vmax}  r2_vmax={args.r2_vmax}', flush=True)

    import analysis_functions as af
    from matplotlib.colors import LinearSegmentedColormap

    af.meta_pth = meta_dir
    df_plot = df.copy()
    df_plot['beryl_hex'] = df_plot['region'].apply(
        af.swanson_to_beryl_hex, args=[af.br])
    beryl_palette = dict(zip(df_plot['region'], df_plot['beryl_hex']))

    sxp_green = LinearSegmentedColormap.from_list(
        'sxp_green',
        ['#EAF4B3', '#D5E1A0', '#A3C968', '#86AF40', '#517146', '#33492E'],
    )
    cmap_common = {
        'sc_duringstim_regtype': af.get_cmap_('sc_duringstim_regtype'),
        'sxp_stim': sxp_green,
    }
    cmap_full = {
        **cmap_common,
        'r2_unique_prior': af.get_cmap_('prior'),       # purple
        'r2_unique_stim': af.get_cmap_('stim_d'),       # blue
        'r2_unique_choice': af.get_cmap_('choice_d'),   # orange
    }

    outs = [
        (out_dir / 'table_var_partition_sxp_regtype.png', COMPACT_COLS, cmap_common),
        (out_dir / 'table_var_partition_mixed.png', FULL_COLS, cmap_full),
    ]
    for out, cols, cmaps in outs:
        print(f'plotting → {out}', flush=True)
        af.plot_table_with_styles(
            df=df_plot[cols].copy(),
            beryl_palette=beryl_palette,
            colormap_lookup=cmaps,
            out_path=out,
        )
        print(f'wrote {out}', flush=True)

    stim = vp['r2_unique_stim_mean'].astype(float)
    sxp = vp['r2_stim_x_prior_mean'].astype(float)
    vp_out = vp.copy()
    vp_out['sxp_stim'] = np.where(stim > 1e-12, sxp / stim, np.nan)
    csv_cols = [
        'region', 'n_neurons', 'n_insertions',
        'sc_duringstim_regtype',
        'r2_unique_prior_mean', 'r2_unique_stim_mean', 'r2_unique_choice_mean',
        'r2_stim_x_prior_mean', 'sxp_stim',
    ]
    csv_cols = [c for c in csv_cols if c in vp_out.columns]
    csv_out = out_dir / 'table_var_partition_mixed.csv'
    vp_out[csv_cols].sort_values('sxp_stim', ascending=False).to_csv(
        csv_out, index=False)
    print(f'wrote {csv_out}', flush=True)


if __name__ == '__main__':
    main()
