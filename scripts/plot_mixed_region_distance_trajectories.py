#!/usr/bin/env python
"""
Goal 1 (2026-07-20): per mixed stim×choice region, plot three duringstim
distance trajectories vs label shuffles:

  1. early stim  — ``stim_duringstim1_act``  (d^{stim,se'}, stim_block_{l,r}_act)
  2. prior       — ``act_block_duringstim``
  3. choice      — ``choice_duringstim_act``

Reuses ``analysis_functions.load_combined_data`` / ``_time_axis_for_plot`` and
the shuffle-overlay style of ``plot_regional_distance``.

  conda activate iblenv
  python scripts/plot_mixed_region_distance_trajectories.py
  python scripts/plot_mixed_region_distance_trajectories.py --regions MRN SCm CP
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_OPENALYX = (
    Path.home() / 'Downloads' / 'ONE' / 'openalyx.internationalbrainlab.org'
)
DEFAULT_REGIONS_CSV = ROOT / 'data' / 'var_partition_mixed_stim_choice_regions.csv'

# (timeframe, title, plot_offset, plot_gain)
PANELS = [
    ('stim_duringstim1_act', r'early stim $d^{\mathrm{stim},se\prime}$', True, True),
    ('act_block_duringstim', r'prior $d^{\mathrm{prior},s}$', True, True),
    ('choice_duringstim_act', r'choice $d^{\mathrm{choice},s}$', True, False),
]


def _patch_one_local(cache_dir: Path):
    import one.api as one_api

    _real = one_api.ONE

    def _local_one(*args, **kwargs):
        return _real(cache_dir=str(cache_dir), mode='local', silent=True)

    one_api.ONE = _local_one


def _load_region_curves(af, timeframe: str, reg: str, dist: str = 'de'):
    """Return (times_ms, r [nrand+1, T], d_dict) averaged across combined splits."""
    d_all, r_all, _, _ = af.load_combined_data(timeframe, dist=dist)
    if reg not in r_all:
        raise KeyError(f'{reg!r} missing from {timeframe} regde')
    r = r_all[reg]
    r = np.concatenate([np.asarray(r[0]).reshape(1, -1), np.asarray(r[1])], axis=0)
    splits = af.run_align[timeframe]
    r = r / len(splits)
    times = af._time_axis_for_plot(timeframe, r.shape[1])
    return times, r, d_all[reg]


def _compute_p_gain(r, d, alpha: float):
    """Offset-subtracted late-window gain p (plot_regional_distance)."""
    if r.shape[1] < 5:
        return np.nan, r[0]
    mean_first5 = np.mean(r[:, :5], axis=1)
    p_off = float(d.get('p_offset', np.nan))
    if np.isfinite(p_off) and p_off <= alpha:
        offset = mean_first5[0] - np.mean(mean_first5[1:])
        r_shifted = r[0] - offset
    else:
        r_shifted = r[0]
    p_gain = float(np.mean(np.mean(r[1:, 4:], axis=1) >= np.mean(r_shifted[4:])))
    return p_gain, r_shifted


def _draw_panel(
    ax,
    times,
    r,
    d,
    title: str,
    *,
    ptype: str = 'p_mean_c',
    alpha: float = 0.01,
    n_shuf_show: int = 40,
    plot_offset: bool = True,
    plot_gain: bool = False,
    show_y: bool = True,
):
    """One trajectory panel: shuffles + observed, matching plot_regional_distance."""
    n_show = min(n_shuf_show, r.shape[0] - 1)
    for i in range(1, n_show + 1):
        if plot_offset and times[0] >= 0 and r.shape[1] >= 5:
            ax.plot(times[:5], r[i][:5], c='#5f7ea3', alpha=0.5, linewidth=0.5)
            ax.plot(times[4:], r[i][4:], c='gray', alpha=0.2, linewidth=0.5)
        else:
            ax.plot(times, r[i], c='gray', alpha=0.2, linewidth=0.5)

    ls_obs = '--' if plot_gain else '-'
    ax.plot(times, r[0], c='black', linewidth=1.5, linestyle=ls_obs)

    p_gain = np.nan
    if plot_gain:
        p_gain, r_shifted = _compute_p_gain(r, d, alpha)
        p_off = float(d.get('p_offset', np.nan))
        if np.isfinite(p_off) and p_off <= alpha:
            ax.plot(times, r_shifted, c='black', linewidth=1.2, linestyle='-')

    # Per-bin significance vs shuffle (same as plot_regional_distance)
    p_per_time = np.mean(r >= r[0], axis=0)
    sig_mask = p_per_time <= alpha
    if np.any(sig_mask):
        y0, y1 = ax.get_ylim()
        ax.scatter(
            times[sig_mask],
            np.full(int(np.sum(sig_mask)), y0),
            marker='v',
            color='blue',
            s=22,
            zorder=5,
        )
        ax.set_ylim(y0, y1)

    p_val = float(d.get(ptype, np.nan))
    ax.text(
        0.04,
        0.96,
        f'{ptype}={p_val:.4f}',
        transform=ax.transAxes,
        color='red' if (np.isfinite(p_val) and p_val <= alpha) else 'black',
        fontsize=11,
        ha='left',
        va='top',
    )
    y_ann = 0.84
    if plot_gain and np.isfinite(p_gain):
        ax.text(
            0.04,
            y_ann,
            f'p_gain={p_gain:.4f}',
            transform=ax.transAxes,
            color='red' if p_gain <= alpha else 'purple',
            fontsize=9,
            ha='left',
            va='top',
        )
        y_ann -= 0.12
    if plot_offset and 'p_offset' in d:
        p_off = float(d['p_offset'])
        if np.isfinite(p_off):
            ax.text(
                0.04,
                y_ann,
                f'p_offset={p_off:.4f}',
                transform=ax.transAxes,
                color='red' if p_off <= alpha else '#5f7ea3',
                fontsize=9,
                ha='left',
                va='top',
            )

    ax.axvline(0, color='k', lw=0.6, alpha=0.4)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('time from stimOn (ms)', fontsize=10)
    if show_y:
        ax.set_ylabel(r'$d_{\mathrm{euc}}$', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_region_three_panel(
    af,
    reg: str,
    out_path: Path,
    *,
    ptype: str = 'p_mean_c',
    alpha: float = 0.01,
    n_shuf_show: int = 40,
    dpi: int = 200,
):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), dpi=dpi)
    missing = []
    for ax, (timeframe, title, plot_offset, plot_gain) in zip(axes, PANELS):
        try:
            times, r, d = _load_region_curves(af, timeframe, reg)
        except (KeyError, FileNotFoundError, OSError) as exc:
            missing.append(f'{timeframe}: {exc}')
            ax.set_title(f'{title}\n(missing)', fontsize=11, color='gray')
            ax.axis('off')
            continue
        _draw_panel(
            ax,
            times,
            r,
            d,
            title,
            ptype=ptype,
            alpha=alpha,
            n_shuf_show=n_shuf_show,
            plot_offset=plot_offset,
            plot_gain=plot_gain,
            show_y=(ax is axes[0]),
        )

    fig.suptitle(f'{reg} — mixed stim×choice duringstim distances', fontsize=13, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return missing


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir', type=Path, default=DEFAULT_OPENALYX)
    p.add_argument('--res-dir', type=Path, default=None,
                   help='Override manifold/res (default: <cache>/manifold/res)')
    p.add_argument('--regions-csv', type=Path, default=DEFAULT_REGIONS_CSV)
    p.add_argument('--regions', nargs='*', default=None,
                   help='Optional subset of region acronyms')
    p.add_argument('--out-dir', type=Path, default=None,
                   help='Default: <cache>/manifold/figs/mixed_stim_choice_trajectories')
    p.add_argument('--ptype', default='p_mean_c')
    p.add_argument('--alpha', type=float, default=0.01)
    p.add_argument('--n-shuf-show', type=int, default=40)
    args = p.parse_args()

    _patch_one_local(args.cache_dir)

    import analysis_functions as af

    res_dir = args.res_dir or (args.cache_dir / 'manifold' / 'res')
    af.pth_res = Path(res_dir)
    out_dir = args.out_dir or (
        args.cache_dir / 'manifold' / 'figs' / 'mixed_stim_choice_trajectories'
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    regs = list(pd.read_csv(args.regions_csv)['region'].astype(str))
    if args.regions:
        want = set(args.regions)
        regs = [r for r in regs if r in want]
        missing_req = sorted(want - set(regs))
        if missing_req:
            print(f'[warn] requested regions not in CSV: {missing_req}')

    print(f'res: {af.pth_res}')
    print(f'out: {out_dir}')
    print(f'regions ({len(regs)}): {regs}')

    n_ok = 0
    for reg in regs:
        out = out_dir / f'{reg}_stim1_prior_choice_duringstim.png'
        missing = plot_region_three_panel(
            af,
            reg,
            out,
            ptype=args.ptype,
            alpha=args.alpha,
            n_shuf_show=args.n_shuf_show,
        )
        if missing:
            print(f'[warn] {reg}: {missing}')
        else:
            n_ok += 1
            print(f'wrote {out}')

    print(f'done: {n_ok}/{len(regs)} regions')


if __name__ == '__main__':
    main()
