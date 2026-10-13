"""Figure 4.5 panels b, c, e: block-L vs block-R population distance for
rastermap cell types, against block-label shuffles.

Re-implements (same math, same styling choices) from paper-brain-wide-map
``analysis_functions``:
  - ``plot_average_distance_across_all_pairs(..., plot_offset=True,
    plot_gain=True, plot_inset=True)``  -> panels b (C, M) and c (S)
  - ``plot_averaged_distances_comparison(vers='concat_block_pairs_act')``
                                        -> panel e (C vs M, shuffle-mean subtracted)

Inputs are stacked files ``{vers}_normFalse_shuffleTrue_{raster_type}.npy``
(``stack_concat`` / ``fig45_block_peths.py stack``). Plots go next to the
inputs (``<dmn>/res/fig45/<tag>/`` for the 2025 files, else
``<root>/plots/<tag>/``).

Usage (iblenv, outside sandbox):
  python scripts/fig45_plot_distances.py --tag old_crossval     # existing 2025 files
  python scripts/fig45_plot_distances.py \
      --root ~/Downloads/ONE/openalyx.internationalbrainlab.org/dmn/res/fig45/cvTrue --tag act
  # S panel as published used vis_subset0:
  python scripts/fig45_plot_distances.py --vers concat_block_pairs_act_vis \
      --types vis_subset0 --tag old_crossval_S
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from fig45_common import (COLORS, DMN_RES, load_stacked, pairs_for_vers,
                          normalized_distance, savefig_svg_png)

OFFSET_WINDOW = 5      # bins used for the early-offset test (stim side)
GAIN_SKIP = 4          # bins dropped before the gain test
STIM_PLOT_MS = 150


def split_pairs(pairs):
    return ([p for p in pairs if 'stim' in p[0]],
            [p for p in pairs if 'choice' in p[0]])


def averaged_traces(d, pairs):
    dists, ctrls = zip(*(normalized_distance(d, p) for p in pairs))
    mean_distance = np.mean(np.stack(dists), axis=0)
    stacked_controls = np.mean(np.stack(ctrls), axis=0)   # (nrand, T)
    if 'stim' in pairs[0][0]:
        x = np.linspace(0, 200, len(mean_distance))
        keep = x <= STIM_PLOT_MS
        return x[keep], mean_distance[keep], stacked_controls[:, keep]
    x = np.linspace(-150, 0, len(mean_distance))
    return x, mean_distance, stacked_controls


def distance_stats(mean_distance, stacked_controls, alpha=0.05):
    """p-values as in plot_average_distance_across_all_pairs (ptype='mean')."""
    distances = np.concatenate([mean_distance[None], stacked_controls], axis=0)
    mean_vals = distances.mean(1)
    p_mean = float(np.mean(mean_vals >= mean_vals[0]))
    first = distances[:, :OFFSET_WINDOW].mean(1)
    p_offset = float(np.mean(first >= first[0]))
    if p_offset < alpha:
        offset = first[0] - first[1:].mean()
    else:
        offset = 0.0
    shifted = mean_distance - offset
    r_gain = np.concatenate([shifted[GAIN_SKIP:][None],
                             stacked_controls[:, GAIN_SKIP:]], axis=0)
    gm = r_gain.mean(1)
    p_gain = float(np.mean(gm >= gm[0]))
    amp = distances.max(1) - distances.min(1)
    p_per_time = np.mean(stacked_controls >= mean_distance[None], axis=0)
    return dict(p_mean=p_mean, p_offset=p_offset, p_gain=p_gain,
                offset=float(offset), amp=float(amp[0]),
                p_amp=float(np.mean(amp >= amp[0])),
                n_sig_bins=int(np.sum(p_per_time < alpha)),
                shifted=shifted)


def plot_type_side(d, pairs, raster_type, side, out_dir, alpha=0.05):
    x, md, sc = averaged_traces(d, pairs)
    st = distance_stats(md, sc, alpha)
    col = COLORS.get(raster_type, 'k')
    is_stim = side == 'stim'

    fig = plt.figure(figsize=(5, 3))
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 0.7], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    axh = fig.add_subplot(gs[0, 1], sharey=ax)

    for i in range(min(100, sc.shape[0])):
        if is_stim:
            ax.plot(x[:OFFSET_WINDOW], sc[i, :OFFSET_WINDOW], color='#5f7ea3', lw=0.2, alpha=0.5)
            ax.plot(x[OFFSET_WINDOW - 1:], sc[i, OFFSET_WINDOW - 1:], color='silver', lw=0.2, alpha=0.5)
        else:
            ax.plot(x, sc[i], color='silver', lw=0.2, alpha=0.5)
    if is_stim:
        ax.plot(x[:OFFSET_WINDOW], md[:OFFSET_WINDOW], color='blue', ls='--', lw=2)
        ax.plot(x[OFFSET_WINDOW - 1:], md[OFFSET_WINDOW - 1:], color=col, ls='--', lw=2)
    else:
        ax.plot(x, md, color=col, ls='--', lw=2)
    if st['offset'] != 0.0:
        ax.plot(x, st['shifted'], color=col, lw=2)

    ins = inset_axes(ax, width='25%', height='25%', loc='upper left', borderpad=2.0)
    cm = sc.mean(0)
    lo, hi = np.percentile(sc - cm, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    ins.fill_between(x, lo, hi, color='gray', alpha=0.3, lw=0)
    ins.plot(x, md - cm, color=col, lw=1)
    ins.set_xticks([])
    ins.set_facecolor('none')

    axh.hist(sc.mean(1), bins=20, orientation='horizontal', color='gray', alpha=0.6)
    axh.axhline(md.mean(), ls='--', lw=2, color=col)
    if is_stim:
        axh.hist(sc[:, :OFFSET_WINDOW].mean(1), bins=20, orientation='horizontal',
                 color='#5f7ea3', alpha=0.5)
        axh.axhline(md[:OFFSET_WINDOW].mean(), color='blue', ls='--', lw=2)
    axh.hist(sc[:, GAIN_SKIP:].mean(1), bins=20, density=True, orientation='horizontal',
             color='#c8a2d6', alpha=0.5)
    axh.axhline(st['shifted'][GAIN_SKIP:].mean(), color=col, lw=2)
    for s in ('top', 'right', 'bottom'):
        axh.spines[s].set_visible(False)
    axh.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.set_xticks([0, 40, 80, 120] if is_stim else [0, -40, -80, -120])
    ax.axvline(0, color='#57C1EB' if is_stim else 'tomato', lw=1.5)
    ax.set_xlabel('time (ms)')
    ax.set_ylabel(f'd_prior ({raster_type}, {side})')
    txt = f"p_mean {st['p_mean']:.4f}"
    if is_stim:
        txt += f"\np_offset {st['p_offset']:.4f}"
    txt += f"\np_gain {st['p_gain']:.4f}\nn={len(d['uuids'])}"
    ax.text(0.98, 0.04, txt, transform=ax.transAxes, ha='right', va='bottom', fontsize=7)
    savefig_svg_png(fig, out_dir / f'dist_{raster_type}_{side}')
    plt.close(fig)
    return {k: v for k, v in st.items() if k != 'shifted'}


def plot_comparison(ds, pairs, out_dir, rts=('integrator', 'move_init_movement')):
    """Panel e: mean over pairs of (distance - shuffle mean), C vs M."""
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.5), dpi=150, sharey=True)
    out = {}
    for ax, side_pairs, side in zip(axs, split_pairs(pairs), ('stim', 'choice')):
        amp = {}
        for rt in rts:
            corr = []
            for p in side_pairs:
                dist, ctrl = normalized_distance(ds[rt], p)
                corr.append(dist - ctrl.mean(0))
            tr = np.mean(corr, axis=0)
            if side == 'stim':
                t = np.linspace(0, 200, len(tr))
                tr, t = tr[t <= STIM_PLOT_MS], t[t <= STIM_PLOT_MS]
            else:
                t = np.linspace(-150, 0, len(tr))
            amp[rt] = float(tr.max() - tr.min())
            ax.plot(t, tr, color=COLORS[rt], lw=2, label=rt)
        out[side] = {'amp': amp, 'amp_ratio_M_over_C': amp[rts[1]] / amp[rts[0]]}
        ax.text(0.05, 0.95, f"amp ratio M/C: {out[side]['amp_ratio_M_over_C']:.2f}",
                transform=ax.transAxes, va='top', fontsize=8)
        ax.set_xticks([0, 40, 80, 120] if side == 'stim' else [0, -40, -80, -120])
        ax.set_xlabel('time (ms)')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axs[0].set_ylabel('d_prior - shuffle mean')
    axs[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    savefig_svg_png(fig, out_dir / 'dist_C_vs_M_shuffle_subtracted')
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=DMN_RES,
                    help='dir holding {vers}_normFalse_shuffleTrue_{type}.npy')
    ap.add_argument('--vers', default='concat_block_pairs_act')
    ap.add_argument('--types', nargs='+',
                    default=['integrator', 'move_init_movement', 'stim_res'])
    ap.add_argument('--tag', required=True,
                    help='output subdir: <root>/fig45/<tag> for the 2025 res dir, '
                         'else <root>/plots/<tag>')
    args = ap.parse_args()

    pairs = pairs_for_vers(args.vers)
    out_dir = (args.root / 'fig45' / args.tag if args.root.resolve() == DMN_RES.resolve()
               else args.root / 'plots' / args.tag)
    ds = {rt: load_stacked(args.vers, rt, args.root) for rt in args.types}
    summary = {'vers': args.vers, 'root': str(args.root), 'types': {}}
    stim_pairs, choice_pairs = split_pairs(pairs)
    for rt, d in ds.items():
        summary['types'][rt] = {'n_cells': int(len(d['uuids']))}
        for side, sp in (('stim', stim_pairs), ('choice', choice_pairs)):
            summary['types'][rt][side] = plot_type_side(d, sp, rt, side, out_dir)
    m_name = next((m for m in ('move_all', 'move_init_movement') if m in ds), None)
    if 'integrator' in ds and m_name:
        summary['C_vs_M'] = plot_comparison(ds, pairs, out_dir, rts=('integrator', m_name))
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print('wrote', out_dir)


if __name__ == '__main__':
    main()
