#!/usr/bin/env python
"""
Summarize revised Goal 3: true block L-vs-R at 0% contrast, separately within
choice-L and choice-R trials. Write per-region CSVs (with BH-FDR) and a separate
all-region population summary.

Legacy options retain the earlier four-way contrast/feedback analyses.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CONTRASTS = [1.0, 0.25, 0.125, 0.0625, 0.0]
GOAL3_C0_CHOICE_SPLITS = [
    'block_duringstim_choice_l_0.0',
    'block_duringstim_choice_r_0.0',
]

SPLIT_BASES = [
    'act_block_duringstim_r_choice_r_f1',
    'act_block_duringstim_l_choice_l_f1',
    'act_block_duringstim_l_choice_r_f2',
    'act_block_duringstim_r_choice_l_f2',
]

F1_BASES = [
    'act_block_duringstim_l_choice_l_f1',
    'act_block_duringstim_r_choice_r_f1',
]
F2_BASES = [
    'act_block_duringstim_l_choice_r_f2',
    'act_block_duringstim_r_choice_l_f2',
]

# Stim-side only (no choice×feedback restriction) — UNSPLIT_PRIOR style
STIM_SIDE_SPLITS = [
    'act_block_duringstim_r',
    'act_block_duringstim_l',
]

BAYES_CHOICE_SPLITS = [
    'bayes_block_duringstim_r_choice_r_f1',
    'bayes_block_duringstim_l_choice_l_f1',
    'bayes_block_duringstim_l_choice_r_f2',
    'bayes_block_duringstim_r_choice_l_f2',
]

BAYES_STIM_SIDE_SPLITS = [
    'bayes_block_duringstim_r',
    'bayes_block_duringstim_l',
]


def _default_res_dir() -> Path:
    return Path(
        '/Users/ariliu/Downloads/ONE/alyx.internationalbrainlab.org'
        '/manifold/res/new'
    )


def _default_meta_dir() -> Path:
    return Path(
        '/Users/ariliu/Downloads/ONE/alyx.internationalbrainlab.org/meta'
    )


def _openalyx_meta_dir() -> Path:
    return Path('/Users/ariliu/Downloads/ONE/openalyx.internationalbrainlab.org/meta')


def summarize_goal3_c0_choice(pth_res: Path, meta_dir: Path, alpha: float):
    '''Write revised Goal-3 regional and literal all-region summaries.'''
    from statsmodels.stats.multitest import multipletests

    meta_dir.mkdir(parents=True, exist_ok=True)
    aggregate_rows = []
    regional_outputs = []
    for split in GOAL3_C0_CHOICE_SPLITS:
        regional_path = pth_res / f'{split}.npy'
        all_path = pth_res / f'{split}_all.npy'
        missing = [p for p in (regional_path, all_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(f'missing finalized Goal-3 outputs: {missing}')

        regional = np.load(regional_path, allow_pickle=True).item()
        rows = [
            {
                'region': reg,
                'nclus': int(result['nclus']),
                'p_euc': float(result['p_euc']),
                'amp_euc': float(result['amp_euc']),
                'lat_euc': float(result['lat_euc']),
            }
            for reg, result in regional.items()
        ]
        if rows:
            pvals = [row['p_euc'] for row in rows]
            _, corrected, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
            for row, p_fdr in zip(rows, corrected):
                row['p_euc_fdr'] = float(p_fdr)
                row['significant_fdr'] = bool(p_fdr <= alpha)
        regional_df = pd.DataFrame(rows)
        if rows:
            regional_df = regional_df.sort_values(['p_euc_fdr', 'region'])
        regional_out = meta_dir / f'{split}_regions_a{alpha:g}.csv'
        regional_df.to_csv(regional_out, index=False)
        regional_outputs.append(regional_out)

        all_result = np.load(all_path, allow_pickle=True).item()
        aggregate_rows.append({
            'choice': 'left' if '_choice_l_' in split else 'right',
            'split': split,
            'nclus': int(all_result['nclus']),
            'n_regions': int(all_result['n_regions']),
            'p_euc': float(all_result['p_euc']),
            'amp_euc': float(all_result['amp_euc']),
            'lat_euc': float(all_result['lat_euc']),
        })
        n_sig = int(regional_df.get(
            'significant_fdr', pd.Series(dtype=bool)).sum())
        print(
            f'{split}: {n_sig}/{len(regional_df)} regions FDR≤{alpha:g}; '
            f'all-region p={all_result["p_euc"]:.4g}, '
            f'n={all_result["nclus"]:,} neurons'
        )

    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_out = meta_dir / 'goal3_c0_choice_all_regions.csv'
    aggregate_df.to_csv(aggregate_out, index=False)
    print(f'wrote {aggregate_out}')
    for path in regional_outputs:
        print(f'wrote {path}')
    return regional_outputs, aggregate_out


def plot_goal3_c0_choice_tables(pth_res: Path, meta_dir: Path, alphas: list[float]):
    '''
    Mirror the legacy per-contrast workflow for revised Goal 3:
    combine choice-L + choice-R 0% splits → p_mean/p_gain/p_offset → BH-FDR →
    gain/offset summary table at each alpha.
    '''
    import analysis_functions as af

    af.pth_res = pth_res
    af.meta_pth = meta_dir
    af.meta_pth.mkdir(parents=True, exist_ok=True)
    open_order = _openalyx_meta_dir() / 'region_order.txt'
    local_order = af.meta_pth / 'region_order.txt'
    if open_order.exists() and not local_order.exists():
        local_order.write_text(open_order.read_text())

    timeframe = 'block_duringstim_choice_c0'
    outs = []
    for alpha in alphas:
        print(f'\n======== revised Goal 3 c0 choice combine @ α={alpha:g} ========')
        outs.append(
            run_splits(
                af,
                pth_res,
                GOAL3_C0_CHOICE_SPLITS,
                timeframe=timeframe,
                alpha=alpha,
                meta_dir=meta_dir,
                file_tag='c0_choice',
                prior='block',
            )
        )
    return outs


def contrast_tag(c: float) -> str:
    return f'{float(c)}'


def contrast_label(c: float) -> str:
    """Short label for filenames / timeframe keys."""
    if c == 0.0:
        return 'c0'
    if c == 1.0:
        return 'c1'
    if c == 0.25:
        return 'c025'
    if c == 0.125:
        return 'c0125'
    if c == 0.0625:
        return 'c00625'
    return f'c{c}'.replace('.', '')


def split_name(base: str, contrast: float | None) -> str:
    if contrast is None:
        return base
    return f'{base}_{contrast_tag(contrast)}'


def available_splits(pth_res: Path, contrast: float | None) -> list[str]:
    out = []
    for base in SPLIT_BASES:
        name = split_name(base, contrast)
        if (
            Path(pth_res, f'{name}.npy').exists()
            and Path(pth_res, f'{name}_regde.npy').exists()
        ):
            out.append(name)
    return out


def load_nclus_by_reg(path: Path) -> dict[str, int] | None:
    if not path.exists():
        return None
    r = np.load(path, allow_pickle=True).flat[0]
    return {reg: int(v.get('nclus', 0)) for reg, v in r.items()}


def retention_row(pth_res: Path, bases: list[str], contrast: float | None):
    maps = []
    labels = []
    for b in bases:
        m = load_nclus_by_reg(pth_res / f'{split_name(b, contrast)}.npy')
        side = 'L' if '_l_choice_' in b or b.endswith('_l') else 'R'
        if m is None:
            maps.append(None)
            labels.append(side)
            continue
        maps.append(m)
        labels.append(side)
    present = [m for m in maps if m is not None]
    if not present:
        return None
    totals = [sum(m.values()) if m is not None else 0 for m in maps]
    mean_cells = float(np.mean([t for t in totals if t > 0]))
    regs = set().union(*[m.keys() for m in present])
    return {
        'contrast': 'all' if contrast is None else contrast_tag(contrast),
        'nclus_by_side': dict(zip(labels, totals)),
        'mean_nclus': mean_cells,
        'nreg': len(regs),
        'n_files': len(present),
    }


def print_retention_tables(pth_res: Path):
    """Per stim-side nclus; % = mean(L,R) / mean(all-contrast L,R)."""
    tables = {}
    for fb, bases in [('f1', F1_BASES), ('f2', F2_BASES)]:
        rows = []
        base = retention_row(pth_res, bases, None)
        if base is None:
            print(f'{fb}: missing all-contrast baseline')
            continue
        for c in [None] + CONTRASTS:
            row = retention_row(pth_res, bases, c)
            if row is None:
                rows.append({
                    'contrast': 'all' if c is None else contrast_tag(c),
                    'L': '—', 'R': '—', 'mean': '—',
                    'pct': '—', 'nreg': '—',
                })
                continue
            sides = row['nclus_by_side']
            pct = 100.0 * row['mean_nclus'] / base['mean_nclus']
            rows.append({
                'contrast': row['contrast'],
                'L': sides.get('L', 0),
                'R': sides.get('R', 0),
                'mean': int(round(row['mean_nclus'])),
                'pct': pct,
                'nreg': row['nreg'],
            })
        tables[fb] = (base, rows)
        print(f'\n=== {fb} cell retention (vs all-contrast mean ≈ {base["mean_nclus"]:,.0f}) ===')
        print(f'{"contrast":>8} {"L":>8} {"R":>8} {"mean":>8} {"% kept":>8} {"nreg":>6}')
        for r in rows:
            if r['pct'] == '—':
                print(f'{r["contrast"]:>8} {"—":>8} {"—":>8} {"—":>8} {"—":>8} {"—":>6}')
            else:
                print(
                    f'{r["contrast"]:>8} {r["L"]:>8,} {r["R"]:>8,} '
                    f'{r["mean"]:>8,} {r["pct"]:>7.1f}% {r["nreg"]:>6}'
                )
    return tables


def combine_splits(pth_res: Path, splits: list[str], pre_post=(0.0, 0.15)) -> str:
    """Union-sum combine of per-split regde (same logic as d_var_stacked_multi)."""
    combined_regde: dict = {}
    for split in splits:
        split_regde = np.load(
            Path(pth_res, f'{split}_regde.npy'), allow_pickle=True
        ).item()
        for reg, curves in split_regde.items():
            true = np.asarray(curves[0], dtype=float)
            nulls = np.asarray(curves[1:], dtype=float)
            if reg not in combined_regde:
                combined_regde[reg] = [true.copy(), nulls.copy()]
            else:
                combined_regde[reg][0] += true
                combined_regde[reg][1] += nulls

    r = {}
    for reg, (sum_real, controls) in combined_regde.items():
        amp_real = float(np.max(sum_real) - np.min(sum_real))
        amp_controls = [float(np.max(c) - np.min(c)) for c in controls]
        p_euc = float(np.mean(np.array(amp_controls) >= amp_real))
        d_euc = sum_real - np.min(sum_real)
        amp_euc = float(np.max(d_euc))
        loc = np.where(d_euc > 0.7 * amp_euc)[0]
        if len(loc) == 0:
            lat_euc = np.nan
        else:
            t = np.linspace(-pre_post[0], pre_post[1], len(d_euc))
            lat_euc = float(t[loc[0]])
        r[reg] = {
            'd_euc': d_euc,
            'amp_euc': amp_euc,
            'p_euc': p_euc,
            'lat_euc': lat_euc,
        }

    combined_name = 'combined_' + '_'.join(splits)
    np.save(Path(pth_res, f'{combined_name}.npy'), r, allow_pickle=True)
    np.save(
        Path(pth_res, f'combined_regde_{"_".join(splits)}.npy'),
        combined_regde,
        allow_pickle=True,
    )
    print(f'combined {len(r)} regions from {len(splits)} splits -> {combined_name}.npy')
    return combined_name


def compute_p_and_fdr_combined(af, timeframe: str, alpha: float):
    """Combined-only p-values + FDR (skip per-split pass)."""
    from statsmodels.stats.multitest import multipletests

    d, r_all, combined_name, _ = af.load_combined_data(timeframe, dist='de')
    regs = list(d)

    def _stack(reg):
        r = r_all[reg]
        return np.concatenate([r[0].reshape(1, -1), r[1]], axis=0)

    for reg in regs:
        r = _stack(reg)
        d[reg]['p_mean'] = float(np.mean(np.mean(r, axis=1) >= np.mean(r[0])))

    for reg in regs:
        r = _stack(reg)
        mean_first5 = np.mean(r[:, :5], axis=1)
        d[reg]['p_offset'] = float(np.mean(mean_first5 >= mean_first5[0]))
        d[reg]['p_offset_effect'] = float(mean_first5[0] - np.mean(mean_first5[1:]))

    for reg in regs:
        r = _stack(reg)
        mean_first5 = np.mean(r[:, :5], axis=1)
        p_val_offset = float(np.mean(mean_first5 >= mean_first5[0]))
        offset = (
            float(mean_first5[0] - np.mean(mean_first5[1:]))
            if p_val_offset < alpha
            else 0.0
        )
        r_shifted = r[0] - offset
        r_new = r[:, 4:].copy()
        r_new[0] = r_shifted[4:]
        max_idx = int(np.argmax(r_new[0]))
        d[reg]['p_gain'] = float(np.mean(np.mean(r_new, axis=1) >= np.mean(r_new[0])))
        d[reg]['p_gain_effect'] = float(
            np.max(r_new[0]) - np.mean(r_new[:, max_idx])
        )

    np.save(Path(af.pth_res, f'{combined_name}.npy'), d, allow_pickle=True)

    for ptype in ['p_mean', 'p_gain', 'p_offset']:
        d = np.load(Path(af.pth_res, f'{combined_name}.npy'), allow_pickle=True).flat[0]
        regs = list(d)
        pvals = [d[x][ptype] for x in regs]
        _, pvals_c, _, _ = multipletests(pvals, alpha, method='fdr_bh')
        for i, reg in enumerate(regs):
            d[reg][f'{ptype}_c'] = float(pvals_c[i])
        np.save(Path(af.pth_res, f'{combined_name}.npy'), d, allow_pickle=True)
        n_sig = sum(1 for p in pvals_c if p <= alpha)
        print(f'  {ptype}: FDR {n_sig}/{len(regs)} ≤ {alpha}')


def plot_gain_offset_table(af, timeframe: str, alpha: float, out_path: Path):
    """Gain/offset columns only (no SC), same styling as plot_combined_table_summary."""
    splits = af.run_align[timeframe]
    split_name_ = 'combined_' + '_'.join(splits)
    ptype = 'p_mean_c'

    af.compute_amp_slope(timeframe, n=20)
    res = af.manifold_to_csv(split_name_, alpha, ptype, sample=False)
    res = res.fillna(0)

    gain_sig = (
        (res['p_gain'] < alpha).astype(float)
        * res['significant'].astype(float)
        * res['p_gain_effect'].astype(float)
    )
    offset_sig = (
        (res['p_offset'] < alpha).astype(float)
        * res['significant'].astype(float)
        * res['p_offset_effect'].astype(float)
    )

    df = pd.DataFrame({
        'region': res['region'],
        f'{timeframe}_gain_sig': gain_sig,
        f'{timeframe}_offset_sig': offset_sig,
    })
    df['beryl_hex'] = df['region'].apply(af.swanson_to_beryl_hex, args=[af.br])
    beryl_palette = dict(zip(df['region'], df['beryl_hex']))
    df['cosmos'] = df['region'].apply(lambda r: af.beryl_to_cosmos(r, af.br))
    df['sum'] = (
        df[f'{timeframe}_gain_sig'].abs() + df[f'{timeframe}_offset_sig'].abs()
    )

    ordering_path = Path(af.meta_pth, 'region_order.txt')
    if not ordering_path.exists():
        ordering_path = _openalyx_meta_dir() / 'region_order.txt'
    if ordering_path.exists():
        with open(ordering_path) as f:
            region_order = [line.strip() for line in f if line.strip()]
        present = set(df['region'])
        ordered = [r for r in region_order if r in present]
        extras = [r for r in df['region'] if r not in set(ordered)]
        region_order = ordered + extras
    else:
        df = df.sort_values(['cosmos', 'sum'], ascending=[True, False])
        region_order = df['region'].tolist()

    df['region'] = pd.Categorical(df['region'], categories=region_order, ordered=True)
    df = df.sort_values('region')
    display_cols = [
        'region',
        f'{timeframe}_gain_sig',
        f'{timeframe}_offset_sig',
    ]
    df_to_plot = df[display_cols].reset_index(drop=True)
    colormap_lookup = {
        name: af.get_cmap_(name) for name in display_cols if name != 'region'
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    af.plot_table_with_styles(
        df=df_to_plot,
        beryl_palette=beryl_palette,
        colormap_lookup=colormap_lookup,
        out_path=out_path,
    )
    n_sig = int(res['significant'].sum())
    n_gain = int(((res['p_gain'] < alpha) & (res['significant'] == 1)).sum())
    n_off = int(((res['p_offset'] < alpha) & (res['significant'] == 1)).sum())
    print(
        f'wrote {out_path.name}  '
        f'(nreg={len(res)}, p_mean_c sig@α={alpha}: {n_sig}, '
        f'gain∩sig: {n_gain}, offset∩sig: {n_off})'
    )
    return df, res


def run_splits(
    af,
    pth_res: Path,
    splits: list[str],
    timeframe: str,
    alpha: float,
    meta_dir: Path,
    file_tag: str | None = None,
    prior: str = 'act',
):
    """prior: 'act' or 'bayes' — used in output filename (replaces act↔bayes)."""
    missing = [
        s for s in splits
        if not (
            Path(pth_res, f'{s}.npy').exists()
            and Path(pth_res, f'{s}_regde.npy').exists()
        )
    ]
    if missing:
        raise FileNotFoundError(f'missing finalized splits: {missing}')

    combine_splits(pth_res, splits)
    af.run_align[timeframe] = list(splits)
    print(f'[{timeframe}] p-values + FDR...')
    compute_p_and_fdr_combined(af, timeframe, alpha)

    # Match analysis_figs naming: table_{prior}_block_combined_summary_{prior}_...
    # Optional suffix only for variants (stim_lr, c1, …) — never repeat prior name.
    out_name = (
        f'table_{prior}_block_combined_summary_{prior}_p_mean_c_'
        f'combinedpTrue_{alpha:g}_gain_offset'
    )
    if file_tag:
        out_name = f'{out_name}_{file_tag}'
    out_name = f'{out_name}.png'
    out_path = meta_dir / out_name
    plot_gain_offset_table(af, timeframe, alpha, out_path)
    return out_path


def run_contrast(
    af,
    pth_res: Path,
    contrast: float | None,
    alpha: float,
    meta_dir: Path,
    tag: str | None = None,
):
    splits = available_splits(pth_res, contrast)
    if len(splits) < 1:
        label = 'all' if contrast is None else f'c={contrast}'
        print(f'skip {label}: no finalized splits')
        return None
    if contrast is not None and len(splits) < 4:
        print(f'warning: only {len(splits)}/4 splits: {splits}')

    if contrast is None:
        timeframe = 'act_block_duringstim_all'
        file_tag = tag  # None → plain …_gain_offset.png like the reference
    else:
        timeframe = f'act_block_duringstim_{contrast_label(contrast)}'
        file_tag = contrast_label(contrast)
        if tag:
            file_tag = f'{file_tag}_{tag}'

    return run_splits(
        af, pth_res, splits, timeframe, alpha, meta_dir,
        file_tag=file_tag, prior='act',
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res-dir', type=Path, default=_default_res_dir())
    ap.add_argument('--meta-dir', type=Path, default=_default_meta_dir())
    ap.add_argument('--alpha', type=float, default=None,
                    help='Single FDR alpha (default revised path: both 0.05 and 0.01)')
    ap.add_argument(
        '--alphas',
        nargs='+',
        type=float,
        default=None,
        help='FDR alphas for revised Goal-3 tables (default: 0.05 0.01)',
    )
    ap.add_argument(
        '--csv-only',
        action='store_true',
        help='Revised Goal 3: write regional/all CSVs only (no combine/table plots)',
    )
    ap.add_argument(
        '--legacy-contrast',
        action='store_true',
        help='Run the historical stim-side × choice/feedback contrast analysis',
    )
    ap.add_argument(
        '--contrasts',
        nargs='+',
        type=float,
        default=None,
        help='Contrasts to plot (default: all Goal-3 contrasts; ignored with --all-contrast/--stim-side)',
    )
    ap.add_argument(
        '--all-contrast',
        action='store_true',
        help='Combine the four unconditioned choice×feedback splits (no contrast suffix)',
    )
    ap.add_argument(
        '--stim-side',
        action='store_true',
        help='Combine act_block_duringstim_{l,r} (no choice×feedback restriction)',
    )
    ap.add_argument(
        '--bayes-choice',
        action='store_true',
        help='Combine four bayes_block_duringstim choice×feedback splits',
    )
    ap.add_argument(
        '--bayes-stim-side',
        action='store_true',
        help='Combine bayes_block_duringstim_{l,r} (no choice×feedback)',
    )
    ap.add_argument(
        '--tag',
        default=None,
        help='Optional output filename tag (e.g. openalyx_ref)',
    )
    ap.add_argument(
        '--retention-only',
        action='store_true',
        help='Only print f1/f2 cell-retention tables',
    )
    ap.add_argument(
        '--skip-retention',
        action='store_true',
        help='Skip retention tables (useful for isolated ref dirs)',
    )
    args = ap.parse_args()

    pth_res = args.res_dir
    legacy_requested = any([
        args.legacy_contrast,
        args.contrasts is not None,
        args.all_contrast,
        args.stim_side,
        args.bayes_choice,
        args.bayes_stim_side,
        args.retention_only,
        args.skip_retention,
    ])
    if not legacy_requested:
        if args.alphas is not None:
            alphas = args.alphas
        elif args.alpha is not None:
            alphas = [args.alpha]
        else:
            alphas = [0.05, 0.01]
        # Per-split regional CSVs use the first / primary alpha for FDR marking.
        summarize_goal3_c0_choice(pth_res, args.meta_dir, alphas[0])
        for a in alphas[1:]:
            summarize_goal3_c0_choice(pth_res, args.meta_dir, a)
        if args.csv_only:
            return
        return plot_goal3_c0_choice_tables(pth_res, args.meta_dir, alphas)

    tables = None
    if not args.skip_retention:
        tables = print_retention_tables(pth_res)
    if args.retention_only:
        return tables

    import analysis_functions as af

    af.pth_res = pth_res
    af.meta_pth = args.meta_dir
    af.meta_pth.mkdir(parents=True, exist_ok=True)
    open_order = _openalyx_meta_dir() / 'region_order.txt'
    local_order = af.meta_pth / 'region_order.txt'
    if open_order.exists() and not local_order.exists():
        local_order.write_text(open_order.read_text())

    alpha = 0.01 if args.alpha is None else args.alpha
    if args.stim_side:
        print('\n======== stim-side only (no choice restriction) ========')
        run_splits(
            af, pth_res, STIM_SIDE_SPLITS,
            timeframe='act_block_duringstim_stimlr',
            alpha=alpha,
            meta_dir=af.meta_pth,
            file_tag=args.tag or 'stim_lr',
            prior='act',
        )
    elif args.bayes_choice:
        print('\n======== bayes choice×feedback 4-split ========')
        run_splits(
            af, pth_res, BAYES_CHOICE_SPLITS,
            timeframe='bayes_block_duringstim_choice',
            alpha=alpha,
            meta_dir=af.meta_pth,
            file_tag=args.tag,  # default None → …_gain_offset.png
            prior='bayes',
        )
    elif args.bayes_stim_side:
        print('\n======== bayes stim-side only ========')
        run_splits(
            af, pth_res, BAYES_STIM_SIDE_SPLITS,
            timeframe='bayes_block_duringstim_stimlr',
            alpha=alpha,
            meta_dir=af.meta_pth,
            file_tag=args.tag or 'stim_lr',
            prior='bayes',
        )
    elif args.all_contrast:
        print('\n======== all-contrast (unconditioned 4-split) ========')
        run_contrast(
            af, pth_res, None, alpha, af.meta_pth,
            tag=args.tag,  # None → plain …_gain_offset.png
        )
    else:
        contrasts = args.contrasts if args.contrasts is not None else CONTRASTS
        for c in contrasts:
            print(f'\n======== contrast {c} ========')
            run_contrast(af, pth_res, c, alpha, af.meta_pth, tag=args.tag)

    return tables


if __name__ == '__main__':
    main()
