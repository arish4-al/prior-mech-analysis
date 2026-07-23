#!/usr/bin/env python
"""
Compare choice L–R sensitivity: openalyx label-shuffle (within stim×block) vs
a second arm (excl-sticky, Harris, actkernel, or any folder with the 8 act
choice ``*_regde.npy`` splits).

Same path as journal 2026-07-12 / ``plot_choice_excl_sticky_comparison_table``:
sum four-split ``*_regde`` → amp / ``p_mean`` → BH-FDR → normalized amp × sig.

Examples
--------
# original excl-sticky comparison
python scripts/plot_choice_null_comparison_table.py --alpha 0.05

# alyx res/new vs openalyx shuffle
python scripts/plot_choice_null_comparison_table.py \\
  --arm-res ~/Downloads/ONE/alyx.internationalbrainlab.org/manifold/res/new \\
  --arm-tag new --force-combine --alpha 0.05
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

CHOICE_DURINGCHOICE_ACT = [
    'choice_stim_r_block_r_act',
    'choice_stim_l_block_l_act',
    'choice_stim_r_block_l_act',
    'choice_stim_l_block_r_act',
]
CHOICE_DURINGSTIM_ACT = [
    'choice_duringstim_r_block_r_act',
    'choice_duringstim_l_block_l_act',
    'choice_duringstim_r_block_l_act',
    'choice_duringstim_l_block_r_act',
]

TIMEFRAMES = {
    'choice_duringstim_act': CHOICE_DURINGSTIM_ACT,
    'choice_duringchoice_act': CHOICE_DURINGCHOICE_ACT,
}


def _combined_name(splits: list[str]) -> str:
    return 'combined_' + '_'.join(splits)


def _default_openalyx_res() -> Path:
    return Path.home() / (
        'Downloads/ONE/openalyx.internationalbrainlab.org/manifold/res'
    )


def _default_arm_res() -> Path:
    return Path.home() / (
        'Downloads/ONE/alyx.internationalbrainlab.org/manifold/res_excl_sticky'
    )


def _default_meta() -> Path:
    return Path.home() / 'Downloads/ONE/alyx.internationalbrainlab.org/meta'


def combine_four_splits(
    pth_res: Path,
    splits: list[str],
    force: bool = False,
    split_suffix: str = '',
) -> tuple[str, dict]:
    """Sum regde across splits (same as d_var_stacked_multi combine) → amp/p_euc.

    ``split_suffix`` is appended to each split basename on disk
    (e.g. ``_pseudosession`` → ``choice_stim_l_block_l_act_pseudosession_regde.npy``).
    Combined output names also include the suffix so they do not overwrite
    label-shuffle combines.
    """
    disk_splits = [f'{s}{split_suffix}' for s in splits]
    combined_name = _combined_name(disk_splits)
    out_npy = pth_res / f'{combined_name}.npy'
    out_regde = pth_res / f'combined_regde_{"_".join(disk_splits)}.npy'

    if out_npy.exists() and out_regde.exists() and not force:
        d = np.load(out_npy, allow_pickle=True).item()
        return combined_name, d

    combined_regde: dict = {}
    for split in disk_splits:
        path = pth_res / f'{split}_regde.npy'
        if not path.exists():
            raise FileNotFoundError(path)
        split_regde = np.load(path, allow_pickle=True).item()
        for reg, curves in split_regde.items():
            curves = np.asarray(curves)
            if reg not in combined_regde:
                combined_regde[reg] = [curves[0].copy(), np.array(curves[1:])]
            else:
                combined_regde[reg][0] += curves[0]
                combined_regde[reg][1] += np.array(curves[1:])

    r = {}
    for reg, (sum_real, controls) in combined_regde.items():
        amp_real = float(np.max(sum_real) - np.min(sum_real))
        amp_controls = [float(np.max(c) - np.min(c)) for c in controls]
        p_euc = float(np.mean(np.asarray(amp_controls) >= amp_real))
        d_euc = sum_real - np.min(sum_real)
        amp_euc = float(np.max(d_euc))
        stacked = np.concatenate([sum_real.reshape(1, -1), controls], axis=0)
        p_mean = float(np.mean(np.mean(stacked, axis=1) >= np.mean(stacked[0])))
        p_amp = float(np.mean(np.asarray(amp_controls) >= amp_real))
        r[reg] = {
            'd_euc': d_euc,
            'amp_euc': amp_euc,
            'p_euc': p_euc,
            'p_mean': p_mean,
            'p_amp': p_amp,
            'lat_euc': np.nan,
            'p_gain': np.nan,
            'p_offset': np.nan,
            'p_gain_effect': np.nan,
            'p_offset_effect': np.nan,
            'amp_slope': np.nan,
            'slope_last': np.nan,
            'amp_loc': np.nan,
            'slope_last_5': np.nan,
            'slope_last_10': np.nan,
            'amp_last5_is_global_max': np.nan,
        }

    np.save(out_npy, r, allow_pickle=True)
    np.save(out_regde, combined_regde, allow_pickle=True)
    return combined_name, r


def fdr_on_dict(d: dict, ptype: str, alpha: float) -> dict:
    regs = list(d.keys())
    pvals = [d[reg][ptype] for reg in regs]
    _, pvals_c, _, _ = multipletests(pvals, alpha, method='fdr_bh')
    for i, reg in enumerate(regs):
        d[reg][f'{ptype}_c'] = float(pvals_c[i])
    return d


def load_or_build_combined(
    pth_res: Path,
    timeframe: str,
    ptype: str,
    alpha: float,
    force_combine: bool = False,
    split_suffix: str = '',
) -> tuple[str, dict]:
    splits = TIMEFRAMES[timeframe]
    combined_name, d = combine_four_splits(
        pth_res, splits, force=force_combine, split_suffix=split_suffix)
    need_fdr = any(f'{ptype}_c' not in d[reg] for reg in d)
    if need_fdr or force_combine:
        if ptype not in next(iter(d.values())):
            raise KeyError(f'{ptype} missing in {combined_name}')
        d = fdr_on_dict(d, ptype, alpha)
        np.save(pth_res / f'{combined_name}.npy', d, allow_pickle=True)
    return combined_name, d


def amp_sig_series(
    d: dict,
    regions: list[str],
    ptype: str,
    alpha: float,
) -> pd.Series:
    """Normalized amp_euc × significant (same as plot_table)."""
    amps = []
    sigs = []
    for reg in regions:
        if reg not in d:
            amps.append(np.nan)
            sigs.append(0)
            continue
        amps.append(float(d[reg]['amp_euc']))
        p = d[reg].get(ptype, d[reg].get(ptype.replace('_c', ''), np.nan))
        sigs.append(int(p <= alpha) if np.isfinite(p) else 0)
    s = pd.Series(amps, index=regions, dtype=float)
    min_val, max_val = s.min(skipna=True), s.max(skipna=True)
    if not np.isfinite(min_val) or not np.isfinite(max_val) or max_val == min_val:
        norm = s.fillna(0) * 0 + 1e-4
    else:
        norm = (s - min_val) / (max_val - min_val) + 1e-4
    norm = norm.fillna(0) * pd.Series(sigs, index=regions)
    return norm


def build_comparison_table(
    openalyx_res: Path,
    arm_res: Path,
    meta_dir: Path,
    arm_tag: str = 'excl',
    ptype: str = 'p_mean_c',
    alpha: float = 0.05,
    force_combine: bool = False,
    out_prefix: str | None = None,
    arm_split_suffix: str = '',
) -> pd.DataFrame:
    import analysis_functions as af

    br = af.br
    swanson_to_beryl_hex = af.swanson_to_beryl_hex
    beryl_to_cosmos = af.beryl_to_cosmos
    get_cmap_ = af.get_cmap_
    plot_table_with_styles = af.plot_table_with_styles

    p_base = ptype[:-2] if ptype.endswith('_c') else ptype
    tag = arm_tag.strip().replace(' ', '_')
    col_s_arm = f'choice_s_{tag}'
    col_m_arm = f'choice_m_{tag}'
    if out_prefix is None:
        out_prefix = f'table_choice_{tag}_vs_shuffle'

    loaded = {}
    for label, pth, force, suffix in [
        ('oa', openalyx_res, False, ''),
        ('arm', arm_res, force_combine, arm_split_suffix),
    ]:
        for tf in TIMEFRAMES:
            name, d = load_or_build_combined(
                pth, tf, p_base, alpha, force_combine=force,
                split_suffix=suffix,
            )
            if f'{p_base}_c' not in next(iter(d.values())):
                d = fdr_on_dict(d, p_base, alpha)
                np.save(pth / f'{name}.npy', d, allow_pickle=True)
            loaded[(label, tf)] = d

    ordering_path = meta_dir / 'region_order.txt'
    oa_meta = Path.home() / 'Downloads/ONE/openalyx.internationalbrainlab.org/meta'
    if not ordering_path.exists():
        ordering_path = oa_meta / 'region_order.txt'
    sample_csv = openalyx_res / 'act_block_only.csv'
    if sample_csv.exists():
        regions = pd.read_csv(sample_csv).region.tolist()
    elif ordering_path.exists():
        regions = [line.strip() for line in ordering_path.read_text().splitlines() if line.strip()]
    else:
        regions = sorted(set().union(*(set(d) for d in loaded.values())))

    cols = {
        'choice_s': amp_sig_series(
            loaded[('oa', 'choice_duringstim_act')], regions, f'{p_base}_c', alpha
        ),
        col_s_arm: amp_sig_series(
            loaded[('arm', 'choice_duringstim_act')], regions, f'{p_base}_c', alpha
        ),
        'choice_m': amp_sig_series(
            loaded[('oa', 'choice_duringchoice_act')], regions, f'{p_base}_c', alpha
        ),
        col_m_arm: amp_sig_series(
            loaded[('arm', 'choice_duringchoice_act')], regions, f'{p_base}_c', alpha
        ),
    }
    table = pd.DataFrame(cols)
    table['region'] = regions
    table['beryl_hex'] = table['region'].apply(lambda r: swanson_to_beryl_hex(r, br))
    beryl_palette = dict(zip(table['region'], table['beryl_hex']))
    table['sum'] = table[['choice_s', col_s_arm, 'choice_m', col_m_arm]].sum(axis=1)
    table['cosmos'] = table['region'].apply(lambda r: beryl_to_cosmos(r, br))

    if ordering_path.exists():
        region_order = [line.strip() for line in ordering_path.read_text().splitlines() if line.strip()]
    else:
        table = table.sort_values(['cosmos', 'sum'], ascending=[True, False])
        region_order = table['region'].tolist()
        meta_dir.mkdir(parents=True, exist_ok=True)
        ordering_path.write_text('\n'.join(region_order) + '\n')

    table['region'] = pd.Categorical(table['region'], categories=region_order, ordered=True)
    table = table.sort_values('region')

    df_to_plot = table.drop(columns=['beryl_hex', 'sum', 'cosmos']).reset_index(drop=True)
    df_to_plot = df_to_plot[['region', 'choice_s', col_s_arm, 'choice_m', col_m_arm]]

    colormap_lookup = {
        'choice_s': get_cmap_('choice_duringstim_act'),
        col_s_arm: get_cmap_('choice_duringstim_act'),
        'choice_m': get_cmap_('choice_duringchoice_act'),
        col_m_arm: get_cmap_('choice_duringchoice_act'),
    }

    meta_dir.mkdir(parents=True, exist_ok=True)
    out_path = meta_dir / f'{out_prefix}_{ptype}_{alpha}.png'
    plot_table_with_styles(
        df=df_to_plot,
        colormap_lookup=colormap_lookup,
        beryl_palette=beryl_palette,
        out_path=out_path,
    )

    df2 = df_to_plot[['region', 'choice_m', col_m_arm]].copy()
    df2 = df2.rename(columns={
        'choice_m': 'choice_shuffle',
        col_m_arm: f'choice_{tag}',
    })
    out2 = meta_dir / f'{out_prefix}_duringchoice_{ptype}_{alpha}.png'
    plot_table_with_styles(
        df=df2,
        colormap_lookup={
            'choice_shuffle': get_cmap_('choice_duringchoice_act'),
            f'choice_{tag}': get_cmap_('choice_duringchoice_act'),
        },
        beryl_palette=beryl_palette,
        out_path=out2,
    )

    rows = []
    for reg in region_order:
        row = {'region': reg}
        for label, tf, key in [
            ('oa', 'choice_duringstim_act', 's'),
            ('arm', 'choice_duringstim_act', f's_{tag}'),
            ('oa', 'choice_duringchoice_act', 'm'),
            ('arm', 'choice_duringchoice_act', f'm_{tag}'),
        ]:
            d = loaded[(label, tf)]
            if reg not in d:
                row[f'amp_{key}'] = np.nan
                row[f'p_{key}'] = np.nan
                row[f'sig_{key}'] = 0
                continue
            p = d[reg].get(f'{p_base}_c', np.nan)
            row[f'amp_{key}'] = d[reg]['amp_euc']
            row[f'p_{key}'] = p
            row[f'sig_{key}'] = int(p <= alpha) if np.isfinite(p) else 0
        rows.append(row)
    summary = pd.DataFrame(rows)
    csv_path = meta_dir / f'{out_prefix}_{ptype}_{alpha}.csv'
    summary.to_csv(csv_path, index=False)

    # Overlap stats among regions present in both
    both = summary.dropna(subset=[f'p_s', f'p_s_{tag}', f'p_m', f'p_m_{tag}'])
    print(f'Wrote {out_path}')
    print(f'Wrote {out2}')
    print(f'Wrote {csv_path}')
    print(f'Arm: {arm_res}  tag={tag}  split_suffix={arm_split_suffix!r}')
    for epoch, sk, ak in [
        ('duringstim', 'sig_s', f'sig_s_{tag}'),
        ('duringchoice', 'sig_m', f'sig_m_{tag}'),
    ]:
        n_sh = int(both[sk].sum())
        n_arm = int(both[ak].sum())
        lost = int(((both[sk] == 1) & (both[ak] == 0)).sum())
        gained = int(((both[sk] == 0) & (both[ak] == 1)).sum())
        kept = int(((both[sk] == 1) & (both[ak] == 1)).sum())
        print(f'  {epoch}: shuffle={n_sh}  {tag}={n_arm}  '
              f'lost={lost} gained={gained} kept={kept}  (n={len(both)})')
    return df_to_plot


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--openalyx-res', type=Path, default=_default_openalyx_res())
    ap.add_argument('--arm-res', type=Path, default=_default_arm_res(),
                    help='Second arm res/ folder (default: res_excl_sticky)')
    ap.add_argument('--excl-res', type=Path, default=None,
                    help='Alias for --arm-res (back-compat)')
    ap.add_argument('--arm-tag', default='excl',
                    help='Short label for second arm (column / file names)')
    ap.add_argument('--arm-split-suffix', default='',
                    help='On-disk suffix for arm split files '
                         '(e.g. _pseudosession or _harris; empty = plain shuffle names)')
    ap.add_argument('--out-prefix', default=None,
                    help='Filename prefix under meta/ (default: table_choice_{tag}_vs_shuffle)')
    ap.add_argument('--meta-dir', type=Path, default=_default_meta())
    ap.add_argument('--ptype', default='p_mean_c')
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--force-combine', action='store_true')
    args = ap.parse_args()
    arm_res = args.excl_res if args.excl_res is not None else args.arm_res
    # Infer suffix from tag if not set
    suffix = args.arm_split_suffix
    tag_aliases = {
        'pseudosession': '_pseudosession',
        'actkernel': '_pseudosession',  # legacy unconstrained
        'pseudo_strat': '_pseudo_strat',
        'strat': '_pseudo_strat',
        'pseudo_fixed': '_pseudo_fixed',
        'fixedstim': '_pseudo_fixed',
        'harris': '_harris',
    }
    if not suffix and args.arm_tag in tag_aliases:
        suffix = tag_aliases[args.arm_tag]
    build_comparison_table(
        openalyx_res=args.openalyx_res,
        arm_res=arm_res,
        meta_dir=args.meta_dir,
        arm_tag=('pseudosession' if args.arm_tag == 'actkernel' else args.arm_tag),
        ptype=args.ptype,
        alpha=args.alpha,
        force_combine=args.force_combine,
        out_prefix=args.out_prefix,
        arm_split_suffix=suffix,
    )


if __name__ == '__main__':
    main()
