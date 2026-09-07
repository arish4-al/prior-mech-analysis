#!/usr/bin/env python
"""Score Bayes Harris unique (6 local splits) and Bayes-stratum shuffles.

Writes Harris / stratum-shuffle combines into alyx ``res/new`` and tables
into alyx ``meta/``. Does not rebuild the 07-14 prior-L–R shuffle combines.

  conda activate iblenv
  python scripts/plot_bayes_harris_stratum_shuffle.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for p in (REPO_ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from plot_choice_null_comparison_table import (  # noqa: E402
    _split_coverage,
    amp_sig_series,
    combine_four_splits,
    fdr_on_dict,
)
from plot_goal3_c0_summary_table import (  # noqa: E402
    compute_p_and_fdr_combined,
    plot_gain_offset_table,
)

RES = Path.home() / (
    'Downloads/ONE/alyx.internationalbrainlab.org/manifold/res/new'
)
META = Path.home() / 'Downloads/ONE/alyx.internationalbrainlab.org/meta'

BAYES_4 = [
    'bayes_block_duringstim_r_choice_r_f1',
    'bayes_block_duringstim_l_choice_l_f1',
    'bayes_block_duringstim_l_choice_r_f2',
    'bayes_block_duringstim_r_choice_l_f2',
]
BAYES_F1 = BAYES_4[:2]
BAYES_F2 = BAYES_4[2:]
BAYES_UNSPLIT = [
    'bayes_block_duringstim_r',
    'bayes_block_duringstim_l',
]
CHOICE_BAYES = [
    'choice_duringstim_r_block_r_bayes',
    'choice_duringstim_l_block_l_bayes',
    'choice_duringstim_r_block_l_bayes',
    'choice_duringstim_l_block_r_bayes',
]
STIM_BAYES = [
    'stim_choice_r_block_r_bayes',
    'stim_choice_l_block_l_bayes',
    'stim_choice_r_block_l_bayes',
    'stim_choice_l_block_r_bayes',
]


def _bh(d: dict, ptype: str, alpha: float) -> dict:
    out = {reg: dict(rec) for reg, rec in d.items()}
    return fdr_on_dict(out, ptype, alpha)


def _counts(d: dict, ptype: str, alpha: float) -> dict:
    regs = list(d)
    raw = [float(d[r][ptype]) for r in regs]
    n_raw = sum(1 for p in raw if p <= alpha)
    _, pc, _, _ = multipletests(raw, alpha, method='fdr_bh')
    n_fdr = sum(1 for p in pc if p <= alpha)
    floor = min(raw) if raw else np.nan
    n_floor = sum(1 for p in raw if np.isclose(p, floor)) if raw else 0
    amps = [float(d[r]['amp_euc']) for r in regs]
    return {
        'nreg': len(regs),
        'n_raw': n_raw,
        'n_fdr': n_fdr,
        'median_p': float(np.median(raw)) if raw else np.nan,
        'median_amp': float(np.median(amps)) if amps else np.nan,
        'p_floor': float(floor) if raw else np.nan,
        'n_floor': n_floor,
        'hits': [r for r, p in zip(regs, pc) if p <= alpha],
        'raw_hits': [r for r, p in zip(regs, raw) if p <= alpha],
    }


def _overlap(a: dict, b: dict, alpha: float, ptype: str = 'p_mean') -> dict:
    common = sorted(set(a) & set(b))
    _, pa, _, _ = multipletests(
        [float(a[r][ptype]) for r in common], alpha, method='fdr_bh')
    _, pb, _, _ = multipletests(
        [float(b[r][ptype]) for r in common], alpha, method='fdr_bh')
    sa = {r for r, p in zip(common, pa) if p <= alpha}
    sb = {r for r, p in zip(common, pb) if p <= alpha}
    amps = []
    for r in common:
        den = float(a[r]['amp_euc'])
        if den > 0:
            amps.append(float(b[r]['amp_euc']) / den)
    return {
        'n_common': len(common),
        'n_a': len(sa),
        'n_b': len(sb),
        'lost': len(sa - sb),
        'gained': len(sb - sa),
        'kept': len(sa & sb),
        'amp_ratio_med': float(np.median(amps)) if amps else np.nan,
        'kept_regs': sorted(sa & sb),
        'gained_regs': sorted(sb - sa),
        'lost_example': sorted(sa - sb)[:12],
    }


def _lowest(d: dict, ptype: str, n: int = 10) -> list[tuple[str, float]]:
    rows = [(r, float(d[r][ptype])) for r in d]
    rows.sort(key=lambda x: (x[1], x[0]))
    return rows[:n]


def _load_or_combine(
    splits: list[str],
    suffix: str = '',
    force: bool = False,
) -> dict:
    name, d = combine_four_splits(RES, splits, force=force, split_suffix=suffix)
    print(f'  combine {name}: {len(d)} regions')
    return d


def _print_block(title: str, d: dict) -> None:
    print(f'\n=== {title} ===')
    for a in (0.01, 0.05):
        c = _counts(d, 'p_mean', a)
        print(
            f'  α={a:g}: raw {c["n_raw"]}/{c["nreg"]}  '
            f'FDR {c["n_fdr"]}/{c["nreg"]}  '
            f'median p={c["median_p"]:.3f}  median amp={c["median_amp"]:.3f}  '
            f'p-floor={c["p_floor"]:.4g} (n={c["n_floor"]})'
        )
        if c['hits']:
            print(f'    FDR hits: {", ".join(c["hits"])}')
        elif c['raw_hits']:
            print(f'    uncorr only: {", ".join(c["raw_hits"])}')


def _write_vs_csv(
    shuffle: dict,
    harris: dict,
    out: Path,
    alpha: float,
) -> pd.DataFrame:
    common = sorted(set(shuffle) & set(harris))
    sh = _bh(shuffle, 'p_mean', alpha)
    ha = _bh(harris, 'p_mean', alpha)
    rows = []
    for reg in common:
        ps = float(sh[reg]['p_mean_c'])
        ph = float(ha[reg]['p_mean_c'])
        rows.append({
            'region': reg,
            'amp_shuffle': float(sh[reg]['amp_euc']),
            'p_shuffle': ps,
            'sig_shuffle': int(ps <= alpha),
            'amp_harris': float(ha[reg]['amp_euc']),
            'p_harris': ph,
            'sig_harris': int(ph <= alpha),
        })
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f'  wrote {out}')
    return df


def _plot_vs(shuffle: dict, harris: dict, out_png: Path, alpha: float,
             cmap_name: str, left: str, right: str) -> None:
    import analysis_functions as af

    ordering_path = META / 'region_order.txt'
    if not ordering_path.exists():
        ordering_path = Path.home() / (
            'Downloads/ONE/openalyx.internationalbrainlab.org/meta/'
            'region_order.txt'
        )
    if ordering_path.exists():
        regions = [ln.strip() for ln in ordering_path.read_text().splitlines()
                   if ln.strip()]
    else:
        regions = sorted(set(shuffle) | set(harris))
    sh = _bh(shuffle, 'p_mean', alpha)
    ha = _bh(harris, 'p_mean', alpha)
    table = pd.DataFrame({
        left: amp_sig_series(sh, regions, 'p_mean_c', alpha),
        right: amp_sig_series(ha, regions, 'p_mean_c', alpha),
    })
    table['region'] = regions
    table['beryl_hex'] = table['region'].apply(
        lambda r: af.swanson_to_beryl_hex(r, af.br))
    beryl_palette = dict(zip(table['region'], table['beryl_hex']))
    table['cosmos'] = table['region'].apply(
        lambda r: af.beryl_to_cosmos(r, af.br))
    present = set(shuffle) | set(harris)
    ordered = [r for r in regions if r in present]
    extras = [r for r in present if r not in set(ordered)]
    region_order = ordered + extras
    table = table[table['region'].isin(region_order)].copy()
    table['region'] = pd.Categorical(
        table['region'], categories=region_order, ordered=True)
    table = table.sort_values('region')
    df_to_plot = table[['region', left, right]].reset_index(drop=True)
    META.mkdir(parents=True, exist_ok=True)
    af.plot_table_with_styles(
        df=df_to_plot,
        colormap_lookup={left: af.get_cmap_(cmap_name),
                         right: af.get_cmap_(cmap_name)},
        beryl_palette=beryl_palette,
        out_path=out_png,
    )
    print(f'  wrote {out_png}')


def score_harris() -> None:
    print('\n######## 1. Harris unique vs 07-14 shuffle (prior L–R) ########')
    print('Coverage shuffle 4-split')
    _split_coverage(RES, BAYES_4, '')
    print('Coverage Harris 4-split')
    _split_coverage(RES, BAYES_4, '_harris_unique')
    print('Coverage shuffle stim-side')
    _split_coverage(RES, BAYES_UNSPLIT, '')
    print('Coverage Harris stim-side')
    _split_coverage(RES, BAYES_UNSPLIT, '_harris_unique')

    sh4 = _load_or_combine(BAYES_4, '', force=False)
    ha4 = _load_or_combine(BAYES_4, '_harris_unique', force=True)
    shu = _load_or_combine(BAYES_UNSPLIT, '', force=False)
    hau = _load_or_combine(BAYES_UNSPLIT, '_harris_unique', force=True)

    _print_block('4-split shuffle (existing 07-14 combine)', sh4)
    _print_block('4-split Harris unique', ha4)
    print('\n--- 4-split overlap ---')
    for a in (0.01, 0.05):
        ov = _overlap(sh4, ha4, a)
        print(
            f'  α={a:g}: shuffle {ov["n_a"]} → Harris {ov["n_b"]}  '
            f'lost={ov["lost"]} gained={ov["gained"]} kept={ov["kept"]}  '
            f'(n={ov["n_common"]})  amp ratio med={ov["amp_ratio_med"]:.3f}'
        )
        if ov['kept_regs']:
            print(f'    kept: {", ".join(ov["kept_regs"])}')
        if ov['gained_regs']:
            print(f'    gained: {", ".join(ov["gained_regs"])}')
        if ov['lost_example']:
            print(f'    lost e.g.: {", ".join(ov["lost_example"])}')

    print('\n--- f1 / f2 only (in-memory, product-MC) ---')
    for label, splits in (('f1', BAYES_F1), ('f2', BAYES_F2)):
        sh = combine_four_splits(RES, splits, force=True, split_suffix='')[1]
        ha = combine_four_splits(
            RES, splits, force=True, split_suffix='_harris_unique')[1]
        # force=True wrote f1/f2-only combines; fine (distinct names)
        print(f'  {label} shuffle / Harris:')
        for a in (0.01, 0.05):
            cs, ch = _counts(sh, 'p_mean', a), _counts(ha, 'p_mean', a)
            ov = _overlap(sh, ha, a)
            print(
                f'    α={a:g}: shuffle FDR {cs["n_fdr"]}/{cs["nreg"]}  '
                f'Harris FDR {ch["n_fdr"]}/{ch["nreg"]}  '
                f'raw Harris {ch["n_raw"]}  lost={ov["lost"]} kept={ov["kept"]}'
            )
        print(f'    Harris lowest p: {_lowest(ha, "p_mean", 6)}')

    _print_block('stim-side shuffle (existing 07-14 combine)', shu)
    _print_block('stim-side Harris unique', hau)
    print('\n--- stim-side overlap ---')
    for a in (0.01, 0.05):
        ov = _overlap(shu, hau, a)
        print(
            f'  α={a:g}: shuffle {ov["n_a"]} → Harris {ov["n_b"]}  '
            f'lost={ov["lost"]} gained={ov["gained"]} kept={ov["kept"]}  '
            f'(n={ov["n_common"]})  amp ratio med={ov["amp_ratio_med"]:.3f}'
        )
        if ov['kept_regs']:
            print(f'    kept: {", ".join(ov["kept_regs"])}')
        if ov['gained_regs']:
            print(f'    gained: {", ".join(ov["gained_regs"])}')

    print('\n4-split Harris lowest p:', _lowest(ha4, 'p_mean', 10))
    print('stim-side Harris lowest p:', _lowest(hau, 'p_mean', 10))

    _write_vs_csv(
        sh4, ha4,
        META / 'table_bayes_block_harris_unique_vs_shuffle_p_mean_c_0.01.csv',
        0.01,
    )
    _write_vs_csv(
        shu, hau,
        META / 'table_bayes_block_unsplit_harris_unique_vs_shuffle_p_mean_c_0.01.csv',
        0.01,
    )
    _plot_vs(
        sh4, ha4,
        META / 'table_bayes_block_harris_unique_vs_shuffle_p_mean_c_0.01.png',
        0.01, 'bayes_block_duringstim', 'prior_shuffle', 'prior_harris_unique',
    )
    _plot_vs(
        shu, hau,
        META / 'table_bayes_block_unsplit_harris_unique_vs_shuffle_p_mean_c_0.01.png',
        0.01, 'bayes_block_duringstim', 'prior_shuffle', 'prior_harris_unique',
    )


def score_stratum_shuffle() -> None:
    print('\n######## 2. Label shuffle inside Bayes stratum (duringstim) ########')
    import analysis_functions as af

    af.pth_res = RES
    af.meta_pth = META
    META.mkdir(parents=True, exist_ok=True)
    open_order = Path.home() / (
        'Downloads/ONE/openalyx.internationalbrainlab.org/meta/region_order.txt'
    )
    local_order = META / 'region_order.txt'
    if open_order.exists() and not local_order.exists():
        local_order.write_text(open_order.read_text())

    for title, splits, tf, tag in [
        ('choice L–R (stim × Bayes)', CHOICE_BAYES,
         'choice_duringstim_bayes', 'choice_lr'),
        ('stim L–R (choice × Bayes)', STIM_BAYES,
         'stim_duringstim_bayes', 'stim_choice_lr'),
    ]:
        print(f'\n--- {title} ---')
        _split_coverage(RES, splits, '')
        # Product-MC combine (aligned here: n_null=2000 everywhere).
        d = _load_or_combine(splits, '', force=True)
        _print_block(title, d)
        print(f'  lowest p: {_lowest(d, "p_mean", 10)}')

        af.run_align[tf] = list(splits)
        for alpha in (0.01, 0.05):
            print(f'  gain/offset FDR @ α={alpha:g}')
            compute_p_and_fdr_combined(af, tf, alpha)
            plot_gain_offset_table(
                af, tf, alpha,
                META / (
                    f'table_bayes_block_combined_summary_bayes_p_mean_c_'
                    f'combinedpTrue_{alpha:g}_gain_offset_{tag}.png'
                ),
            )


def main() -> None:
    if not RES.is_dir():
        raise SystemExit(f'missing {RES}')
    score_harris()
    score_stratum_shuffle()
    print('\nDone.')


if __name__ == '__main__':
    main()
