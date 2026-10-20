#!/usr/bin/env python
"""Alternative Figure 2 and Figure 4 from alyx manifold/res_standard.

Writes a res_standard version next to the existing paper figures:

  <oa>/manifold/paper_figs/fig2/res_standard/fig2_res_standard.{svg,png}
  <oa>/manifold/paper_figs/fig4/res_standard/figure4_res_standard.{pdf,svg,png}

The 80 ms stim×choice term in the Figure 2 denominator is the first 80 ms
of the 150 ms stim_choice curves (those short splits are not in this run).
act_block_only.csv from openalyx is used only as a region-order list.

  conda activate iblenv   # outside the sandbox
  python scripts/paper_figs/build_standard_figs.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OPENALYX = Path.home() / 'Downloads/ONE/openalyx.internationalbrainlab.org'
ALYX = Path.home() / 'Downloads/ONE/alyx.internationalbrainlab.org'
SRC = ALYX / 'manifold' / 'res_standard'
PAPER = OPENALYX / 'manifold' / 'paper_figs'
OUT2 = PAPER / 'fig2' / 'res_standard'
OUT4 = PAPER / 'fig4' / 'res_standard'
STAGE = OUT2 / 'staging'
OA_CSV = OPENALYX / 'manifold' / 'res' / 'act_block_only.csv'
PANEL_D = OPENALYX / 'manifold' / 'paper_figs' / 'fig2' / 'original' / 'panel_d_original.png'
ALPHA = 0.01


def _import_af():
    import one.api as one_api
    real = one_api.ONE

    def _local(*args, **kwargs):
        kwargs['mode'] = 'local'
        kwargs['silent'] = True
        kwargs.setdefault('cache_dir', str(OPENALYX))
        return real(*args, **kwargs)

    one_api.ONE = _local
    import analysis_functions as af

    def _fast_amp_slope(timeframe, n=20):
        """Same fields as analysis_functions.compute_amp_slope, one regde load."""
        splits = af.run_align[timeframe]
        joined = '_'.join(splits)
        combined_name = f'combined_{joined}'
        regde_name = f'combined_regde_{joined}'
        d = np.load(Path(af.pth_res) / f'{combined_name}.npy', allow_pickle=True).flat[0]
        regde = np.load(Path(af.pth_res) / f'{regde_name}.npy', allow_pickle=True).flat[0]
        for reg in list(d):
            r = np.asarray(regde[reg][0], dtype=float)
            d[reg]['amp_slope'] = float(np.polyfit(np.linspace(0, 0.15, len(r)), r, 1)[0])

            def _last_slope(curve, k):
                k = min(k, len(curve))
                if k < 2:
                    return np.nan
                return float(np.polyfit(np.arange(k), curve[-k:], 1)[0])

            d[reg]['slope_last'] = _last_slope(r, n)
            d[reg]['amp_loc'] = int(np.argmax(r))
            d[reg]['slope_last_5'] = _last_slope(r, 5)
            d[reg]['slope_last_10'] = _last_slope(r, 10)
            if len(r) >= 5:
                win5 = np.convolve(r, np.ones(5) / 5.0, mode='valid')
                last5 = win5[-1]
                prev = win5[:-1]
                is_max = int(last5 > np.max(prev)) if prev.size else 1
                prev_max = float(np.max(prev)) if prev.size else np.nan
            else:
                last5, prev_max, is_max = np.nan, np.nan, 0
            d[reg]['amp_last5_mean'] = float(last5) if last5 == last5 else np.nan
            d[reg]['amp_prev5_max_mean'] = prev_max
            d[reg]['amp_last5_is_global_max'] = int(is_max)
        np.save(Path(af.pth_res) / f'{combined_name}.npy', d, allow_pickle=True)

    af.compute_amp_slope = _fast_amp_slope
    return af


def bh(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out


def n_bins_80(n_bins, window_ms=150.0):
    t = np.linspace(0.0, window_ms, int(n_bins))
    return int(np.sum(t <= 80.0))


def load_regde(split):
    path = SRC / f'{split}_regde.npy'
    d = np.load(path, allow_pickle=True).item()
    out = {reg: np.asarray(curves, dtype=float) for reg, curves in d.items()}
    del d
    return out


def sum_splits(splits, n_keep=None):
    acc = {}
    for split in splits:
        d = load_regde(split)
        for reg, curves in d.items():
            if n_keep is not None:
                curves = curves[:, :n_keep]
            if reg not in acc:
                acc[reg] = curves.copy()
            else:
                acc[reg] = acc[reg] + curves
        del d
    return acc


def summarize(arr, stim_arr=None, alpha=ALPHA):
    """p_mean / amp / offset / gain on a (1+n_null, T) stack."""
    obs = arr[0]
    null = arr[1:]
    means = arr.mean(axis=1)
    p_mean = float(np.mean(means >= means[0]))
    excess = obs - null.mean(axis=0)
    excess = excess - excess.min()
    amp = float(excess.max())
    src = stim_arr if stim_arr is not None else arr
    m5 = src[:, :5].mean(axis=1)
    p_off = float(np.mean(m5 >= m5[0]))
    effect_off = float(m5[0] - m5[1:].mean())
    shifted = obs - effect_off if p_off <= alpha else obs
    late = shifted[4:]
    late_null = arr[1:, 4:]
    p_gain = float(np.mean(late_null.mean(axis=1) >= late.mean()))
    max_idx = int(np.argmax(late)) if late.size else 0
    effect_gain = float(late[max_idx] - late_null[:, max_idx].mean()) if late.size else np.nan
    return {
        'p_mean': p_mean,
        'p_euc': p_mean,
        'amp_euc': amp,
        'lat_euc': np.nan,
        'd_euc': excess,
        'n_null': int(arr.shape[0] - 1),
        'p_offset': p_off,
        'p_offset_effect': effect_off,
        'p_gain': p_gain,
        'p_gain_effect': effect_gain,
    }


def to_regde_list(acc):
    return {reg: [arr[0], np.asarray(arr[1:])] for reg, arr in acc.items()}


def attach_fdr(records):
    regs = list(records)
    pc = bh([records[r]['p_mean'] for r in regs])
    for reg, p_c in zip(regs, pc):
        records[reg]['p_mean_c'] = float(p_c)
    return records


def save_pair(af, name, regde_name, acc, stim_acc=None):
    recs = {}
    for reg, arr in acc.items():
        stim = None if stim_acc is None else stim_acc.get(reg)
        rec = summarize(arr, stim_arr=stim)
        rec.pop('d_euc', None)
        recs[reg] = rec
    attach_fdr(recs)
    af_pth = Path(af.pth_res)
    np.save(af_pth / f'{name}.npy', recs, allow_pickle=True)
    np.save(af_pth / f'{regde_name}.npy', to_regde_list(acc), allow_pickle=True)
    ps = np.array([recs[r]['p_mean_c'] for r in recs])
    print(f'  {name}: {len(recs)} regions  FDR01={(ps <= 0.01).sum()}  '
          f'FDR05={(ps <= 0.05).sum()}  med p={np.median([recs[r]["p_mean"] for r in recs]):.2f}')
    return recs


def combined_names(af, timeframe):
    splits = af.run_align[timeframe]
    joined = '_'.join(splits)
    return splits, f'combined_{joined}', f'combined_regde_{joined}'


def build(af):
    STAGE.mkdir(parents=True, exist_ok=True)
    af.pth_res = STAGE
    if OA_CSV.exists():
        shutil.copy2(OA_CSV, STAGE / 'act_block_only.csv')
    for split_npy in SRC.glob('*.npy'):
        name = split_npy.name
        if ('_regde' in name or '_all' in name or '_pseudo' in name
                or '_harris' in name):
            continue
        dst = STAGE / name
        if not dst.exists():
            dst.symlink_to(split_npy)

    groups = [
        'stim_duringstim_act',
        'stim_duringchoice_act',
        'choice_duringstim_act',
        'choice_duringchoice_act',
        'stim_duringstim1_act',
        'act_block_duringstim',
        'act_block_duringchoice',
        'act_block_duringstim_unsplit',
    ]
    stacks = {}
    for tf in groups:
        splits, name, regde = combined_names(af, tf)
        print('combine', tf)
        acc = sum_splits(splits)
        stim = None
        if 'duringchoice' in tf and tf.startswith('act_block'):
            stim = stacks['act_block_duringstim']
        elif tf == 'choice_duringchoice_act':
            stim = stacks['choice_duringstim_act']
        elif tf == 'stim_duringchoice_act':
            stim = stacks['stim_duringstim_act']
        save_pair(af, name, regde, acc, stim_acc=stim)
        stacks[tf] = acc

    # 80 ms prefix of the 150 ms stim×choice curves (stands in for *_short_act).
    stim150 = stacks['stim_duringstim_act']
    n80 = n_bins_80(next(iter(stim150.values())).shape[1])
    print(f'short prefix: first {n80} bins of stim_duringstim_act')
    short = {reg: arr[:, :n80] for reg, arr in stim150.items()}
    _, short_name, short_regde = combined_names(af, 'stim_duringstim_short_act')
    save_pair(af, short_name, short_regde, short)

    unsplit = stacks['act_block_duringstim_unsplit']
    n80u = n_bins_80(next(iter(unsplit.values())).shape[1])
    early = {}
    for reg, arr in unsplit.items():
        rec = summarize(arr[:, :n80u])
        rec.pop('d_euc', None)
        early[reg] = rec
    attach_fdr(early)
    early_name = (
        'combined_act_block_duringstim_r_act_block_duringstim_l_earlystim80'
    )
    np.save(STAGE / f'{early_name}.npy', early, allow_pickle=True)
    ps = np.array([early[r]['p_mean_c'] for r in early])
    print(f'  earlystim80: {len(early)} regions  FDR01={(ps <= 0.01).sum()}  '
          f'FDR05={(ps <= 0.05).sum()}')

    for tf in ('stim_duringstim_short_act', 'stim_duringstim1_act'):
        af.compute_amp_slope(tf, n=20)
    _write_region_sample(af)
    return early_name


def _write_region_sample(af):
    """Region list shared by every combined file get_sc_table reads."""
    import pandas as pd
    names = []
    for tf in (
        'stim_duringstim_act', 'choice_duringstim_act',
        'stim_duringchoice_act', 'choice_duringchoice_act',
        'stim_duringstim_short_act', 'stim_duringstim1_act',
    ):
        _, name, _ = combined_names(af, tf)
        d = np.load(STAGE / f'{name}.npy', allow_pickle=True).item()
        names.append(set(d))
        del d
    keep = set.intersection(*names)
    order_path = REPO / 'region_order.txt'
    order = [ln.strip() for ln in order_path.read_text().splitlines() if ln.strip()]
    regs = [r for r in order if r in keep] + sorted(keep - set(order))
    pd.DataFrame({'region': regs}).to_csv(STAGE / 'act_block_only.csv', index=False)
    print(f'region sample: {len(regs)} (intersection)')


def sc_and_pvals(af):
    import pandas as pd
    times = [
        'stim_duringstim_act', 'choice_duringstim_act',
        'stim_duringchoice_act', 'choice_duringchoice_act',
    ]
    extra = ['stim_duringstim_short_act', 'stim_duringstim1_act']
    res = af.get_sc_table(
        times, 'p_mean_c', alpha=ALPHA, combined_p=True,
        sc_threshold=0.0, slope_threshold=0.05,
        amp_loc_threshold=67, n=20, stim_restr=True)
    res = res.copy()
    res['sigma_stim_s'] = 1 - res['sc_duringstim']
    res['sigma_stim_m'] = 1 - res['sc_duringchoice']
    res['monotonicity'] = res['sc_duringchoice_move_shape'].fillna(0).astype(int)
    OUT2.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT2 / 'sc_table.csv', index=False)

    rows = {}
    for tf in times + extra:
        _, name, _ = combined_names(af, tf)
        d = np.load(STAGE / f'{name}.npy', allow_pickle=True).item()
        rows[tf] = pd.Series({r: d[r]['p_mean_c'] for r in d})
        rows[f'{tf}_amp'] = pd.Series({r: d[r]['amp_euc'] for r in d})
    pv = pd.DataFrame(rows)
    pv.index.name = 'region'
    pv.to_csv(OUT2 / 'pvals.csv')
    print('region types stim', res['sc_duringstim_regtype'].value_counts(dropna=False).to_dict())
    print('region types choice', res['sc_duringchoice_regtype'].value_counts(dropna=False).to_dict())
    return res, pv


def plot_fig2(pv):
    sys.path.insert(0, str(REPO / 'scripts' / 'paper_figs' / 'fig2'))
    import fig2_common as fc
    import fig2_plot as p2

    fc.FIG_DIR = OUT2
    fc.VERSIONS['res_standard'] = {
        'choice_res': SRC,
        'desc': 'alyx res_standard full-session act prior, plain shuffle',
    }
    fc.version_dir = lambda version: OUT2
    fc.staging_dir = lambda version: STAGE
    orig = OUT2 / 'original'
    orig.mkdir(parents=True, exist_ok=True)
    link = orig / 'panel_d_original.png'
    if not link.exists():
        link.symlink_to(PANEL_D)

    examples = dict(p2.EXAMPLES)
    data = p2.Data('res_standard')
    examples = _examples_present(data, p2, examples)
    used, e_info, n_f = p2.composite(data, examples, OUT2)
    p2.single_panels(data, examples, OUT2 / 'panels')
    summary = {
        'version': 'res_standard',
        'source': str(SRC),
        'examples': used,
        'e': e_info,
        'f_n_regions': n_f,
        'short': 'first 80 ms of stim_duringstim_act, not a separate short job',
        'regtype_counts': {
            c: data.sc[c].value_counts().sort_index().to_dict()
            for c in ('sc_duringstim_regtype', 'sc_duringchoice_regtype')
        },
    }
    (OUT2 / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
    print('wrote', OUT2 / 'fig2_res_standard.svg')
    return examples


def _examples_present(data, p2, examples):
    """Keep a published example when its curves exist; otherwise the first region that does."""
    used = {}
    for panel, regs in examples.items():
        tfs = [tf for tf, _ in p2.PANELS[panel]]
        pools = []
        for tf in tfs:
            if tf not in data._regde:
                _, regde = p2.fc.combined_stems(data.af, tf)
                data._regde[tf] = np.load(
                    p2.fc.staging_dir(data.version) / f'{regde}.npy',
                    allow_pickle=True).item()
            pools.append(set(data._regde[tf]))
        have = set.intersection(*pools) & set(data.pv.index)
        picked = []
        for reg in regs:
            if reg in have:
                picked.append(reg)
            else:
                sub = next(iter(sorted(have - set(picked))))
                print(f'note: fig2 panel {panel} example {reg} has no curve; using {sub}')
                picked.append(sub)
        used[panel] = tuple(picked)
    return used


def plot_fig4(early_name):
    sys.path.insert(0, str(REPO / 'scripts' / 'paper_figs' / 'fig4'))
    import fig4_common as fc4
    import fig4_plot as p4

    fc4.FIG_DIR = OUT4
    fc4.OA_RES = STAGE
    fc4.ALYX_RES = STAGE
    fc4.SC_TABLE = OUT2 / 'sc_table.csv'
    fc4.EARLYSTIM_NPY = STAGE / f'{early_name}.npy'
    fc4.ORIGINAL_PDF = Path.home() / 'Downloads' / 'figure4.pdf'
    # Draw even if a published example leaves the sensory overlap.
    data = p4.Data()
    if not data.overlap:
        raise RuntimeError('sensory ∩ prior_80ms is empty; figure 4 panel b/c has no regions')
    if fc4.EXAMPLE_S not in data.overlap:
        print(f'note: {fc4.EXAMPLE_S} not in sensory ∩ prior_80ms; using {data.overlap[0]}')
        fc4.EXAMPLE_S = data.overlap[0]
    c_ok = [r for r in fc4.EXAMPLE_C if r in data.stim_d]
    if len(c_ok) < 2:
        pool = data.c_strip or [r for r in data.regs if r in data.stim_d]
        c_ok = (c_ok + [r for r in pool if r not in c_ok])[:2]
        print(f'note: C examples -> {c_ok}')
        fc4.EXAMPLE_C = tuple(c_ok)
    if fc4.EXAMPLE_M not in data.stim_d or fc4.EXAMPLE_M not in data.choice_d:
        pool = data.m_strip or [r for r in data.regs if r in data.stim_d and r in data.choice_d]
        print(f'note: M example {fc4.EXAMPLE_M} missing; using {pool[0]}')
        fc4.EXAMPLE_M = pool[0]
    OUT4.mkdir(parents=True, exist_ok=True)
    (OUT4 / 'original').mkdir(parents=True, exist_ok=True)
    (OUT4 / 'panels').mkdir(parents=True, exist_ok=True)
    if fc4.ORIGINAL_PDF.exists():
        shutil.copy2(fc4.ORIGINAL_PDF, OUT4 / 'original' / 'figure4.pdf')
    info = p4.composite(data, OUT4)
    p4.write_panel_strips(data, OUT4 / 'panels')
    for ext in ('.pdf', '.svg', '.png'):
        src = OUT4 / f'figure4{ext}'
        src.replace(OUT4 / f'figure4_res_standard{ext}')
    summary = {
        'source': str(SRC),
        'sc_table': str(fc4.SC_TABLE),
        'overlap': data.overlap,
        'n_overlap': len(data.overlap),
        'c_strip': data.c_strip,
        'm_strip': data.m_strip,
        'n_prior_stim': len(data.prior_stim),
        'n_prior_choice': len(data.prior_choice),
        'n_prior_80': len(data.prior80),
        'example_s': fc4.EXAMPLE_S,
        'example_s_published': 'LGd',
        'example_s_in_overlap': fc4.EXAMPLE_S in data.overlap,
        'panels': info,
        'short': 'prior_80ms is the first 80 ms of the unsplit 150 ms curves',
    }
    def _conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, dict):
            return {str(k): _conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_conv(v) for v in o]
        return o
    (OUT4 / 'summary.json').write_text(json.dumps(_conv(summary), indent=2))
    print(f'overlap ({len(data.overlap)}): {", ".join(data.overlap)}')
    print('wrote', OUT4 / 'figure4_res_standard.pdf')


def main():
    af = _import_af()
    early_name = build(af)
    sc_and_pvals(af)
    plot_fig2(None)
    plot_fig4(early_name)


if __name__ == '__main__':
    main()
