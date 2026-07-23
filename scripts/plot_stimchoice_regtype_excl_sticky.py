#!/usr/bin/env python
"""
Rebuild ``table_stimchoice_act_regtype_*`` using excl-sticky choice L–R
(combined four-split) while keeping stim / short-stim from openalyx.

Writes a **new** PNG (does not overwrite the openalyx/alyx originals):

  meta/table_stimchoice_act_regtype_excl_sticky_{ptype}_{alpha}.png

Staging res mixes:
  openalyx manifold/res  — stim*, short, stim1, act_block_only.csv
  alyx res_excl_sticky   — choice_duringstim_act + choice_duringchoice_act
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OPENALYX = Path.home() / 'Downloads/ONE/openalyx.internationalbrainlab.org'
DEFAULT_ALYX = Path.home() / 'Downloads/ONE/alyx.internationalbrainlab.org'

SC_TIMES = [
    'stim_duringstim_act',
    'choice_duringstim_act',
    'stim_duringchoice_act',
    'choice_duringchoice_act',
]

CHOICE_TIMES = [
    'choice_duringstim_act',
    'choice_duringchoice_act',
]

# Extra stim windows loaded inside get_sc_table
EXTRA_STIM_TIMES = [
    'stim_duringstim_short_act',
    'stim_duringstim1_act',
]


def _patch_one_local(openalyx_cache: Path):
    import one.api as one_api
    _real = one_api.ONE

    def _local_one(*args, **kwargs):
        return _real(
            cache_dir=str(openalyx_cache),
            mode='local',
            silent=True,
        )

    one_api.ONE = _local_one


def _combined_stems(af, timeframe: str) -> list[str]:
    splits = af.run_align[timeframe]
    if len(splits) == 1:
        return [splits[0], f'{splits[0]}_regde']
    joined = '_'.join(splits)
    return [f'combined_{joined}', f'combined_regde_{joined}']


def _link_or_copy(src: Path, dst: Path, copy: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def build_staging(
    openalyx_res: Path,
    excl_res: Path,
    staging: Path,
    af,
) -> None:
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # Region alignment sample
    sample = openalyx_res / 'act_block_only.csv'
    if sample.exists():
        _link_or_copy(sample, staging / sample.name)

    # Stim (+ short / stim1) from openalyx
    for tf in SC_TIMES + EXTRA_STIM_TIMES:
        if tf in CHOICE_TIMES:
            continue
        for stem in _combined_stems(af, tf):
            src = openalyx_res / f'{stem}.npy'
            if not src.exists():
                raise FileNotFoundError(f'missing openalyx {src}')
            _link_or_copy(src, staging / f'{stem}.npy')

    # Choice from excl-sticky (copy so amp_slope writes stay local to staging)
    for tf in CHOICE_TIMES:
        for stem in _combined_stems(af, tf):
            src = excl_res / f'{stem}.npy'
            if not src.exists():
                raise FileNotFoundError(
                    f'missing excl {src} — run plot_choice_excl_sticky_comparison_table.py first'
                )
            _link_or_copy(src, staging / f'{stem}.npy', copy=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--openalyx-cache-dir', type=Path, default=DEFAULT_OPENALYX)
    ap.add_argument('--alyx-cache-dir', type=Path, default=DEFAULT_ALYX)
    ap.add_argument('--excl-res', type=Path, default=None,
                    help='Default: <alyx>/manifold/res_excl_sticky')
    ap.add_argument('--ptype', default='p_mean_c')
    ap.add_argument('--alpha', type=float, default=0.01)
    ap.add_argument('--sc-threshold', type=float, default=0.0)
    ap.add_argument('--slope-threshold', type=float, default=0.05)
    ap.add_argument('--amp-loc-threshold', type=int, default=67)
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--stim-restr', action='store_true', default=True)
    ap.add_argument('--no-stim-restr', dest='stim_restr', action='store_false')
    args = ap.parse_args()

    openalyx = args.openalyx_cache_dir.expanduser().resolve()
    alyx = args.alyx_cache_dir.expanduser().resolve()
    excl_res = (
        args.excl_res.expanduser().resolve()
        if args.excl_res
        else alyx / 'manifold' / 'res_excl_sticky'
    )
    openalyx_res = openalyx / 'manifold' / 'res'
    staging = alyx / 'manifold' / 'res_sc_choice_excl_sticky'
    meta_out = alyx / 'meta'
    meta_out.mkdir(parents=True, exist_ok=True)

    _patch_one_local(openalyx)
    import analysis_functions as af  # noqa: E402

    build_staging(openalyx_res, excl_res, staging, af)

    # Point AF at staging for loads; write PNG to alyx meta
    af.pth_res = staging
    af.meta_pth = meta_out

    import numpy as np
    from statsmodels.stats.multitest import multipletests

    p_base = args.ptype[:-2] if args.ptype.endswith('_c') else args.ptype

    # Shape metrics for excl choice (needed for integrator / move_init)
    for tf in CHOICE_TIMES:
        print(f'compute_amp_slope({tf}) …')
        af.compute_amp_slope(tf, n=args.n)
        d = np_load_combined(af, tf)
        if f'{p_base}_c' not in next(iter(d.values())):
            regs = list(d.keys())
            pvals = [d[r][p_base] for r in regs]
            _, pvals_c, _, _ = multipletests(pvals, args.alpha, method='fdr_bh')
            for i, reg in enumerate(regs):
                d[reg][f'{p_base}_c'] = float(pvals_c[i])
            name = 'combined_' + '_'.join(af.run_align[tf])
            np.save(staging / f'{name}.npy', d, allow_pickle=True)

    # Avoid float×bool & crash when amp_last5 has NaNs for sample-only regions
    _orig_mtc = af.manifold_to_csv

    def _manifold_to_csv(split, sigl, p_type, sample=True):
        df = _orig_mtc(split, sigl, p_type, sample=sample)
        if 'amp_last5_is_global_max' in df.columns:
            df['amp_last5_is_global_max'] = (
                df['amp_last5_is_global_max'].fillna(0).astype(int)
            )
        return df

    af.manifold_to_csv = _manifold_to_csv

    print('get_sc_table + plot …')
    res = af.get_sc_table(
        SC_TIMES, args.ptype, alpha=args.alpha, combined_p=True,
        sc_threshold=args.sc_threshold, slope_threshold=args.slope_threshold,
        amp_loc_threshold=args.amp_loc_threshold, n=args.n,
        stim_restr=args.stim_restr,
    )

    # Mirror plot_sc_table(metric='regtype') but custom filename
    sc_splits = (
        ['sc_duringchoice_regtype', 'sc_duringstim_regtype']
        if args.stim_restr
        else ['sc_duringchoice_regtype', 'sc_duringstim_regtype', 'sc_stim_regtype']
    )
    res = res.copy()
    res['beryl_hex'] = res.region.apply(af.swanson_to_beryl_hex, args=[af.br])
    beryl_palette = dict(zip(res.region, res.beryl_hex))
    names = ['region'] + sc_splits
    res = res[names]
    res['sum'] = res[names[1:]].sum(axis=1, skipna=True)
    res['cosmos'] = res['region'].apply(lambda r: af.beryl_to_cosmos(r, af.br))

    ordering_path = meta_out / 'region_order.txt'
    if not ordering_path.exists():
        ordering_path = openalyx / 'meta' / 'region_order.txt'
    if ordering_path.exists():
        region_order = [line.strip() for line in ordering_path.read_text().splitlines() if line.strip()]
    else:
        res = res.sort_values(['cosmos', 'sum'], ascending=[True, False])
        region_order = res['region'].tolist()

    import pandas as pd
    res['region'] = pd.Categorical(res['region'], categories=region_order, ordered=True)
    res = res.sort_values('region')
    df_to_plot = res.drop(columns=['cosmos', 'sum']).reset_index(drop=True)
    cols = ['region'] + [c for c in df_to_plot.columns if c != 'region']
    df_to_plot = df_to_plot[cols]
    colormap_lookup = {col: af.get_cmap_(col) for col in df_to_plot.columns if col != 'region'}

    out_name = f'table_stimchoice_act_regtype_excl_sticky_{args.ptype}_{args.alpha}.png'
    out_path = meta_out / out_name
    af.plot_table_with_styles(
        df=df_to_plot,
        beryl_palette=beryl_palette,
        colormap_lookup=colormap_lookup,
        out_path=out_path,
    )

    csv_path = meta_out / out_name.replace('.png', '.csv')
    df_to_plot.to_csv(csv_path, index=False)

    # Quick counts
    for col in sc_splits:
        vc = df_to_plot[col].value_counts(dropna=False)
        print(f'{col}: {vc.to_dict()}')
    print(f'Wrote {out_path}')
    print(f'Wrote {csv_path}')
    print('Columns L→R: region | sc_duringchoice_regtype | sc_duringstim_regtype'
          + (' | sc_stim_regtype' if not args.stim_restr else ''))
    print('  (choice amps from excl-sticky; stim amps from openalyx)')


def np_load_combined(af, timeframe: str) -> dict:
    import numpy as np
    name = 'combined_' + '_'.join(af.run_align[timeframe])
    return np.load(af.pth_res / f'{name}.npy', allow_pickle=True).item()


if __name__ == '__main__':
    main()
