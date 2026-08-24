#!/usr/bin/env python
"""
Compare choice L–R or act_block prior L–R sensitivity: label-shuffle vs a
second arm (excl-sticky, Harris, actkernel, or any folder with the matching
``*_regde.npy`` splits).

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

# min5 shuffle in res/new vs harris_unique (journal 2026-07-27c)
python scripts/plot_choice_null_comparison_table.py \\
  --openalyx-res ~/Downloads/ONE/alyx.internationalbrainlab.org/manifold/res/new \\
  --arm-res ~/Downloads/ONE/alyx.internationalbrainlab.org/manifold/res/new \\
  --arm-tag harris_unique --force-combine-shuffle --alpha 0.01

# act_block prior L–R Harris unique vs shuffle (journal 2026-07-27e)
python scripts/plot_choice_null_comparison_table.py --family act_block \\
  --openalyx-res ~/Downloads/ONE/alyx.internationalbrainlab.org/manifold/res/new \\
  --shuffle-res-duringchoice ~/Downloads/ONE/openalyx.internationalbrainlab.org/manifold/res \\
  --arm-res ~/Downloads/ONE/alyx.internationalbrainlab.org/manifold/res/new \\
  --arm-tag harris_unique --force-combine --alpha 0.01

# unsplit analog (stim-side at stimOn; choice-side at movement; no f1/f2)
python scripts/plot_choice_null_comparison_table.py --family act_block_unsplit \\
  --openalyx-res ~/Downloads/ONE/alyx.internationalbrainlab.org/manifold/res/new \\
  --arm-res ~/Downloads/ONE/alyx.internationalbrainlab.org/manifold/res/new \\
  --arm-tag harris_unique --force-combine --alpha 0.01
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

ACT_BLOCK_DURINGSTIM = [
    'act_block_duringstim_r_choice_r_f1',
    'act_block_duringstim_l_choice_l_f1',
    'act_block_duringstim_l_choice_r_f2',
    'act_block_duringstim_r_choice_l_f2',
]
ACT_BLOCK_DURINGCHOICE = [
    'act_block_stim_r_duringchoice_r_f1',
    'act_block_stim_l_duringchoice_l_f1',
    'act_block_stim_l_duringchoice_r_f2',
    'act_block_stim_r_duringchoice_l_f2',
]

FAMILIES = {
    'choice': {
        'timeframes': {
            'choice_duringstim_act': CHOICE_DURINGSTIM_ACT,
            'choice_duringchoice_act': CHOICE_DURINGCHOICE_ACT,
        },
        'tf_s': 'choice_duringstim_act',
        'tf_m': 'choice_duringchoice_act',
        'col_prefix': 'choice',
        'cmap_s': 'choice_duringstim_act',
        'cmap_m': 'choice_duringchoice_act',
        'out_stem': 'choice',
    },
    'act_block': {
        'timeframes': {
            'act_block_duringstim': ACT_BLOCK_DURINGSTIM,
            'act_block_duringchoice': ACT_BLOCK_DURINGCHOICE,
        },
        'tf_s': 'act_block_duringstim',
        'tf_m': 'act_block_duringchoice',
        'col_prefix': 'prior',
        'cmap_s': 'act_block_duringstim',
        'cmap_m': 'act_block_duringchoice',
        'out_stem': 'act_block',
    },
    'act_block_unsplit': {
        'timeframes': {
            'act_block_duringstim_unsplit': [
                'act_block_duringstim_l',
                'act_block_duringstim_r',
            ],
            'act_block_duringchoice_unsplit': [
                'act_block_duringchoice_l',
                'act_block_duringchoice_r',
            ],
        },
        'tf_s': 'act_block_duringstim_unsplit',
        'tf_m': 'act_block_duringchoice_unsplit',
        'col_prefix': 'prior',
        'cmap_s': 'act_block_duringstim',
        'cmap_m': 'act_block_duringchoice',
        'out_stem': 'act_block_unsplit',
    },
    'act_block_fully_unsplit': {
        'timeframes': {
            'act_block_duringstim_fully_unsplit': [
                'act_block_duringstim_fully_unsplit',
            ],
            'act_block_duringchoice_fully_unsplit': [
                'act_block_duringchoice_fully_unsplit',
            ],
        },
        'tf_s': 'act_block_duringstim_fully_unsplit',
        'tf_m': 'act_block_duringchoice_fully_unsplit',
        'col_prefix': 'prior',
        'cmap_s': 'act_block_duringstim',
        'cmap_m': 'act_block_duringchoice',
        'out_stem': 'act_block_fully_unsplit',
    },
}

# Back-compat alias used by load_or_build_combined callers / tests.
TIMEFRAMES = FAMILIES['choice']['timeframes']


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


def _combine_split_curve_stacks(
    stacks: list[np.ndarray],
    rng_seed: int = 0,
    n_mc_null: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum obs across splits; combine nulls (aligned or product-MC if ragged).

    ``stacks``: list of (1+U_s, T) arrays. Returns ``(obs_sum, nulls)`` with
    ``nulls`` shape (n_null, T).

    When unique-null counts differ across splits, each pooled null is one draw
    from the product of per-split unique sets (uniform over each split's
    unique curves). ``n_mc_null`` defaults to ``max(min(U_s), 2000)`` so
    p-value resolution is not capped by the scarcest split's unique count.
    """
    stacks = [np.asarray(s, dtype=float) for s in stacks]
    obs = np.sum([s[0] for s in stacks], axis=0)
    null_counts = [max(s.shape[0] - 1, 0) for s in stacks]
    if any(c < 1 for c in null_counts):
        raise ValueError('each split needs ≥1 null curve to combine')
    lengths = [s.shape[0] for s in stacks]
    if len(set(lengths)) == 1:
        nulls = np.sum([s[1:] for s in stacks], axis=0)
        return obs, nulls
    n_mc = int(n_mc_null) if n_mc_null is not None else max(min(null_counts), 2000)
    n_mc = max(n_mc, 1)
    T = obs.shape[0]
    rng = np.random.default_rng(rng_seed)
    nulls = np.zeros((n_mc, T), dtype=float)
    for k in range(n_mc):
        acc = np.zeros(T, dtype=float)
        for s, n_u in zip(stacks, null_counts):
            acc += s[1 + int(rng.integers(0, n_u))]
        nulls[k] = acc
    return obs, nulls


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

    Handles ragged unique-null counts across splits (Harris unique): product-MC
    over each split's null set with ``n_mc = max(min(U), 2000)``.
    """
    disk_splits = [f'{s}{split_suffix}' for s in splits]
    combined_name = _combined_name(disk_splits)
    out_npy = pth_res / f'{combined_name}.npy'
    out_regde = pth_res / f'combined_regde_{"_".join(disk_splits)}.npy'

    if out_npy.exists() and out_regde.exists() and not force:
        d = np.load(out_npy, allow_pickle=True).item()
        return combined_name, d

    # reg -> list of per-split (1+U, T) stacks
    per_reg_stacks: dict[str, list[np.ndarray]] = {}
    for split in disk_splits:
        path = pth_res / f'{split}_regde.npy'
        if not path.exists():
            raise FileNotFoundError(path)
        split_regde = np.load(path, allow_pickle=True).item()
        for reg, curves in split_regde.items():
            per_reg_stacks.setdefault(reg, []).append(np.asarray(curves))

    combined_regde: dict = {}
    r = {}
    for reg, stacks in per_reg_stacks.items():
        import hashlib
        seed = int(hashlib.md5(
            f'{reg}|{split_suffix}|{"|".join(disk_splits)}'.encode()
        ).hexdigest()[:8], 16)
        sum_real, controls = _combine_split_curve_stacks(stacks, rng_seed=seed)
        # Drop exact-duplicate null curves after combine
        if controls.size and controls.ndim == 2 and len(controls) > 1:
            uniq_rows = []
            seen = set()
            for row in controls:
                key = np.ascontiguousarray(row).tobytes()
                if key in seen:
                    continue
                seen.add(key)
                uniq_rows.append(row)
            controls = np.asarray(uniq_rows)
        combined_regde[reg] = [sum_real, controls]
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
            'n_null': int(len(controls)),
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
    timeframes: dict[str, list[str]] | None = None,
) -> tuple[str, dict]:
    splits = (timeframes or TIMEFRAMES)[timeframe]
    combined_name, d = combine_four_splits(
        pth_res, splits, force=force_combine, split_suffix=split_suffix)
    need_fdr = any(f'{ptype}_c' not in d[reg] for reg in d)
    if need_fdr or force_combine:
        if ptype not in next(iter(d.values())):
            raise KeyError(f'{ptype} missing in {combined_name}')
        d = fdr_on_dict(d, ptype, alpha)
        np.save(pth_res / f'{combined_name}.npy', d, allow_pickle=True)
    return combined_name, d


def _split_coverage(pth_res: Path, splits: list[str], split_suffix: str = '') -> None:
    """Print per-split n_regions / nclus from the non-regde dict (one file at a time)."""
    for split in splits:
        disk = f'{split}{split_suffix}'
        path = pth_res / f'{disk}.npy'
        if not path.exists():
            print(f'  {disk}: MISSING {path}')
            continue
        d = np.load(path, allow_pickle=True).item()
        nclus = 0
        n_nulls = []
        for rec in d.values():
            nclus += int(rec.get('nclus', 0) or 0)
            n_u = rec.get('n_null', rec.get('uperms'))
            if n_u is not None:
                n_nulls.append(int(n_u))
        extra = ''
        if n_nulls:
            extra = (f'  n_null min/med/max='
                     f'{min(n_nulls)}/{int(np.median(n_nulls))}/{max(n_nulls)}')
        print(f'  {disk}: {len(d)} regions / {nclus:,} cells{extra}')
        del d


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
    family: str = 'choice',
    force_combine_shuffle: bool = False,
    shuffle_res_duringchoice: Path | None = None,
) -> pd.DataFrame:
    import analysis_functions as af

    if family not in FAMILIES:
        raise ValueError(f'unknown family {family!r}; expected {list(FAMILIES)}')
    spec = FAMILIES[family]
    timeframes = spec['timeframes']
    tf_s, tf_m = spec['tf_s'], spec['tf_m']
    prefix = spec['col_prefix']

    br = af.br
    swanson_to_beryl_hex = af.swanson_to_beryl_hex
    beryl_to_cosmos = af.beryl_to_cosmos
    get_cmap_ = af.get_cmap_
    plot_table_with_styles = af.plot_table_with_styles

    p_base = ptype[:-2] if ptype.endswith('_c') else ptype
    tag = arm_tag.strip().replace(' ', '_')
    col_s = f'{prefix}_s'
    col_m = f'{prefix}_m'
    col_s_arm = f'{prefix}_s_{tag}'
    col_m_arm = f'{prefix}_m_{tag}'
    if out_prefix is None:
        out_prefix = f'table_{spec["out_stem"]}_{tag}_vs_shuffle'

    shuffle_res_m = shuffle_res_duringchoice or openalyx_res
    shuffle_by_tf = {tf_s: openalyx_res, tf_m: shuffle_res_m}

    print(f'Coverage shuffle ({tf_s}) {openalyx_res}')
    _split_coverage(openalyx_res, timeframes[tf_s], '')
    print(f'Coverage shuffle ({tf_m}) {shuffle_res_m}')
    _split_coverage(shuffle_res_m, timeframes[tf_m], '')
    print(f'Coverage {tag} ({tf_s}) {arm_res}')
    _split_coverage(arm_res, timeframes[tf_s], arm_split_suffix)
    print(f'Coverage {tag} ({tf_m}) {arm_res}')
    _split_coverage(arm_res, timeframes[tf_m], arm_split_suffix)

    loaded = {}
    for tf in timeframes:
        for label, pth, force, suffix in [
            ('oa', shuffle_by_tf[tf], force_combine_shuffle, ''),
            ('arm', arm_res, force_combine, arm_split_suffix),
        ]:
            name, d = load_or_build_combined(
                pth, tf, p_base, alpha, force_combine=force,
                split_suffix=suffix, timeframes=timeframes,
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
    if not sample_csv.exists():
        sample_csv = oa_meta.parent / 'manifold' / 'res' / 'act_block_only.csv'
    if sample_csv.exists():
        regions = pd.read_csv(sample_csv).region.tolist()
    elif ordering_path.exists():
        regions = [line.strip() for line in ordering_path.read_text().splitlines() if line.strip()]
    else:
        regions = sorted(set().union(*(set(d) for d in loaded.values())))

    cols = {
        col_s: amp_sig_series(loaded[('oa', tf_s)], regions, f'{p_base}_c', alpha),
        col_s_arm: amp_sig_series(loaded[('arm', tf_s)], regions, f'{p_base}_c', alpha),
        col_m: amp_sig_series(loaded[('oa', tf_m)], regions, f'{p_base}_c', alpha),
        col_m_arm: amp_sig_series(loaded[('arm', tf_m)], regions, f'{p_base}_c', alpha),
    }
    table = pd.DataFrame(cols)
    table['region'] = regions
    table['beryl_hex'] = table['region'].apply(lambda r: swanson_to_beryl_hex(r, br))
    beryl_palette = dict(zip(table['region'], table['beryl_hex']))
    table['sum'] = table[[col_s, col_s_arm, col_m, col_m_arm]].sum(axis=1)
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
    df_to_plot = df_to_plot[['region', col_s, col_s_arm, col_m, col_m_arm]]

    colormap_lookup = {
        col_s: get_cmap_(spec['cmap_s']),
        col_s_arm: get_cmap_(spec['cmap_s']),
        col_m: get_cmap_(spec['cmap_m']),
        col_m_arm: get_cmap_(spec['cmap_m']),
    }

    meta_dir.mkdir(parents=True, exist_ok=True)
    out_path = meta_dir / f'{out_prefix}_{ptype}_{alpha}.png'
    plot_table_with_styles(
        df=df_to_plot,
        colormap_lookup=colormap_lookup,
        beryl_palette=beryl_palette,
        out_path=out_path,
    )

    df2 = df_to_plot[['region', col_m, col_m_arm]].copy()
    df2 = df2.rename(columns={
        col_m: f'{prefix}_shuffle',
        col_m_arm: f'{prefix}_{tag}',
    })
    out2 = meta_dir / f'{out_prefix}_duringchoice_{ptype}_{alpha}.png'
    plot_table_with_styles(
        df=df2,
        colormap_lookup={
            f'{prefix}_shuffle': get_cmap_(spec['cmap_m']),
            f'{prefix}_{tag}': get_cmap_(spec['cmap_m']),
        },
        beryl_palette=beryl_palette,
        out_path=out2,
    )

    rows = []
    for reg in region_order:
        row = {'region': reg}
        for label, tf, key in [
            ('oa', tf_s, 's'),
            ('arm', tf_s, f's_{tag}'),
            ('oa', tf_m, 'm'),
            ('arm', tf_m, f'm_{tag}'),
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

    both = summary.dropna(subset=[f'p_s', f'p_s_{tag}', f'p_m', f'p_m_{tag}'])
    print(f'Wrote {out_path}')
    print(f'Wrote {out2}')
    print(f'Wrote {csv_path}')
    print(f'Family={family}  Arm: {arm_res}  tag={tag}  '
          f'split_suffix={arm_split_suffix!r}')
    for epoch, sk, ak, pk, pak in [
        ('duringstim', 'sig_s', f'sig_s_{tag}', 'p_s', f'p_s_{tag}'),
        ('duringchoice', 'sig_m', f'sig_m_{tag}', 'p_m', f'p_m_{tag}'),
    ]:
        n_sh = int(both[sk].sum())
        n_arm = int(both[ak].sum())
        lost = int(((both[sk] == 1) & (both[ak] == 0)).sum())
        gained = int(((both[sk] == 0) & (both[ak] == 1)).sum())
        kept = int(((both[sk] == 1) & (both[ak] == 1)).sum())
        med_sh = float(both[pk].median())
        med_arm = float(both[pak].median())
        print(f'  {epoch}: shuffle={n_sh}  {tag}={n_arm}  '
              f'lost={lost} gained={gained} kept={kept}  (n={len(both)})  '
              f'median p shuffle={med_sh:.3f} / {tag}={med_arm:.3f}')
    return df_to_plot


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--family', choices=sorted(FAMILIES), default='choice',
                    help='choice L–R (default), act_block prior L–R (f1/f2), '
                         'act_block_unsplit (stim-side / choice-side), or '
                         'act_block_fully_unsplit (no stratum)')
    ap.add_argument('--openalyx-res', type=Path, default=_default_openalyx_res())
    ap.add_argument('--arm-res', type=Path, default=_default_arm_res(),
                    help='Second arm res/ folder (default: res_excl_sticky)')
    ap.add_argument('--excl-res', type=Path, default=None,
                    help='Alias for --arm-res (back-compat)')
    ap.add_argument('--shuffle-res-duringchoice', type=Path, default=None,
                    help='Shuffle folder for the duringchoice four (default: --openalyx-res)')
    ap.add_argument('--arm-tag', default='excl',
                    help='Short label for second arm (column / file names)')
    ap.add_argument('--arm-split-suffix', default='',
                    help='On-disk suffix for arm split files '
                         '(e.g. _pseudosession or _harris; empty = plain shuffle names)')
    ap.add_argument('--out-prefix', default=None,
                    help='Filename prefix under meta/ '
                         '(default: table_{family}_{tag}_vs_shuffle)')
    ap.add_argument('--meta-dir', type=Path, default=_default_meta())
    ap.add_argument('--ptype', default='p_mean_c')
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--force-combine', action='store_true',
                    help='Rebuild the arm four-split combine')
    ap.add_argument('--force-combine-shuffle', action='store_true',
                    help='Rebuild the shuffle four-split combine (needed when '
                         'plain {split}.npy in that folder was re-finalized)')
    args = ap.parse_args()
    arm_res = args.excl_res if args.excl_res is not None else args.arm_res
    suffix = args.arm_split_suffix
    tag_aliases = {
        'pseudosession': '_pseudosession',
        'actkernel': '_pseudosession',  # legacy unconstrained
        'pseudo_strat': '_pseudo_strat',
        'strat': '_pseudo_strat',
        'pseudo_strat_sticky': '_pseudo_strat_sticky',
        'strat_sticky': '_pseudo_strat_sticky',
        'pseudo_fixed': '_pseudo_fixed',
        'fixedstim': '_pseudo_fixed',
        'pseudo_fixed_sticky': '_pseudo_fixed_sticky',
        'fixed_sticky': '_pseudo_fixed_sticky',
        'sticky': '_pseudo_strat_sticky',
        'harris': '_harris',  # legacy with-replacement (2026-07-24e)
        'harris_unique': '_harris_unique',
        'harris_u': '_harris_unique',
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
        family=args.family,
        force_combine_shuffle=args.force_combine_shuffle,
        shuffle_res_duringchoice=args.shuffle_res_duringchoice,
    )


if __name__ == '__main__':
    main()
