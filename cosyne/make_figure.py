#!/usr/bin/env python
"""COSYNE figure, second version.

  A  Task and question (paper 1a–c). Left empty.
  B  Prior distance pooled by function: Fig 4c, 4e, 4g insets.
     S is the sensory overlap, 0–80 ms, stimulus-aligned.
     C is every integrator, stimulus-aligned. M is every movement
     region, movement-aligned. Gray is the shuffle percentile band.
  C  C vs M after subtracting the shuffle mean (Fig 4i), stimulus-
     aligned and movement-aligned.
  D  Circuit left empty (paper 6a). Prior-distance fit is Fig 5d
     (paper 6d): solid data, dashed model.
  E  Behavior predicted with no fit to behavior (paper 6i): percent
     correct and reaction time.
  F  Summary of conclusions (paper 6j). Left empty.

Fig 4 is the 10-15 rebuild, not res_standard. The model panels are the
cached Fig 5 s89 series. Paper Fig 5 is the model figure.

  conda activate iblenv   # outside the sandbox
  python cosyne/make_figure.py
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for sub in ('scripts/paper_figs/fig4', 'scripts/paper_figs/fig5'):
    sys.path.insert(0, str(REPO / sub))

os.environ.setdefault('PRIOR_MECH_NO_ONE', '1')
os.environ.setdefault(
    'ONE_CACHE_DIR',
    str(Path.home() / 'Downloads/ONE/openalyx.internationalbrainlab.org'),
)

import fig4_plot as f4
import fig4_common as f4c
import fig5_plot as f5
import fig5_common as f5c

W, H = 6.2, 8.02
C_C = '#E8AC22'
C_M = 'tomato'
FS_LETTER = 14
FS_TITLE = 12
FS_LABEL = 11
FS_TICK = 10
FS_LEGEND = 10
FS_R2 = 9

plt.rcParams.update({
    'font.family': 'Helvetica',
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.7,
    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_TICK,
    'axes.labelsize': FS_LABEL,
})


def add(fig, x, y, w, h):
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])


def letter(fig, x, y, s):
    fig.text(x / W, 1 - y / H, s, fontsize=FS_LETTER, fontweight='bold',
             ha='left', va='top', color='#231F20')


def empty_box(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(Rectangle((0.01, 0.01), 0.98, 0.98, fill=False,
                           edgecolor='#c8c8c8', linestyle=(0, (2, 2)), lw=0.7,
                           transform=ax.transAxes))
    ax.axis('off')


def _shrink(ax, labels=FS_TICK):
    ax.tick_params(labelsize=labels, length=3.0, width=0.6, pad=2)
    ax.xaxis.label.set_fontsize(FS_LABEL)
    ax.yaxis.label.set_fontsize(FS_LABEL)


def thin_events(ax, lw=1.0):
    for ln in ax.lines:
        x = np.asarray(ln.get_xdata(), float)
        if x.size >= 2 and np.nanmax(x) - np.nanmin(x) < 1e-3:
            ln.set_linewidth(lw)
            ln.set_zorder(0)


def inset_pack(t, r):
    """Fig 4 c/e/g inset: observed minus shuffle mean, with the shuffle band."""
    diff = r[0] - np.mean(r[1:], axis=0)
    centered = r[1:] - np.mean(r[1:], axis=0, keepdims=True)
    lo = np.percentile(centered, 100 * (f4c.ALPHA / 2), axis=0)
    hi = np.percentile(centered, 100 * (1 - f4c.ALPHA / 2), axis=0)
    return dict(t=np.asarray(t, float), diff=np.asarray(diff, float),
                lo=np.asarray(lo, float), hi=np.asarray(hi, float))


def _ylim_insets(packs):
    lo = min(min(p['diff'].min(), p['lo'].min()) for p in packs.values())
    hi = max(max(p['diff'].max(), p['hi'].max()) for p in packs.values())
    paper = (-0.05, 0.2)
    if lo >= paper[0] and hi <= paper[1]:
        return paper, [0.0, 0.1, 0.2]
    pad = 0.08 * (hi - lo + 1e-6)
    return (min(paper[0], lo - pad), max(paper[1], hi + pad)), [0.0, 0.1, 0.2]


def style_time(ax, align, tmax):
    ev = f4c.C_STIM_ON if align == 'stim' else f4c.C_MOVE_ON
    ax.axvline(0, color=ev, lw=1.0, zorder=0)
    span = abs(tmax)
    pad = max(2.0, 0.03 * span)
    if align == 'stim':
        ax.set_xlim(-pad, tmax + pad)
        if tmax <= 90:
            ax.set_xticks([0, 40, 80])
        else:
            ax.set_xticks([0, 50, 100, 150])
    else:
        ax.set_xlim(-span - pad, pad)
        ax.set_xticks([-150, -100, -50, 0])
    _shrink(ax)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def draw_b(fig, packs, ylim, yticks):
    specs = (
        ('S', f4c.C_STIM, 'stim', 0.74),
        ('C', C_C, 'stim', 2.60),
        ('M', C_M, 'move', 4.46),
    )
    for i, (name, color, align, x) in enumerate(specs):
        ax = add(fig, x, 1.32, 1.44, 0.90)
        pack = packs[name]
        ax.fill_between(pack['t'], pack['lo'], pack['hi'],
                        color='0.75', lw=0, zorder=1)
        ax.plot(pack['t'], pack['diff'], color=color, lw=1.4, zorder=3)
        ax.axhline(0, color='0.6', lw=0.4, zorder=0)
        style_time(ax, align, float(np.nanmax(np.abs(pack['t']))))
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.set_title(name, fontsize=FS_TITLE, color=color, pad=2, loc='left')
        if i == 0:
            ax.set_ylabel('prior distance')
        else:
            ax.tick_params(labelleft=False)
        ax.set_xlabel('time (ms)')


def _panel_i_ylim(curves):
    lo = min(y.min() for y in curves)
    hi = max(y.max() for y in curves)
    ylim = (min(0.03, lo - 0.01), max(0.16, hi + 0.01))
    return ylim, [0.06, 0.09, 0.12, 0.15]


def draw_c(fig, stim, move, ylim, yticks):
    specs = (
        (0.82, stim, 'stim', True),
        (3.63, move, 'move', False),
    )
    for x, pair, align, show_y in specs:
        ax = add(fig, x, 2.68, 2.41, 1.46)
        t, y_c, y_m = pair
        ax.plot(t, y_c, color=C_C, lw=1.5, label='C', zorder=3)
        ax.plot(t, y_m, color=C_M, lw=1.5, label='M', zorder=3)
        style_time(ax, align, 150)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        if show_y:
            ax.set_ylabel('prior distance')
            ax.legend(frameon=False, fontsize=FS_LEGEND, loc='upper left')
        ax.set_xlabel('time (ms)')


def draw_d_fits(fig, series):
    specs = (
        (2.42, 1.58, 'd_stim', f5c.LIM_D_STIM, f5c.TICK_D_X, f5c.C_STIM_ON, True),
        (4.36, 1.64, 'd_move', f5c.LIM_D_MOVE, f5c.TICK_D_X_MOVE, f5c.C_MOVE_ON, False),
    )
    for x, w, key, lim, xt, event, show_y in specs:
        ax = add(fig, x, 4.58, w, 1.18)
        f5._style_ax(ax, lim[0], lim[1], xt, f5c.TICK_D_Y, series[key]['lines'],
                     event, r'$d(t)$' if show_y else None)
        for ln in ax.lines:
            if ln.get_linewidth() >= 1.5:
                ln.set_linewidth(1.15)
        thin_events(ax, 1.0)
        if not show_y:
            ax.tick_params(labelleft=False)
            ax.set_ylabel('')
        ax.set_xlabel('time (ms)')
        _shrink(ax)
    ax = fig.axes[-2]
    ax.text(0.04, 0.97, f"R$^2$ {series['d_stim']['r2']:.2f}",
            transform=ax.transAxes, fontsize=FS_R2, color='#030505',
            ha='left', va='top')


def _smaller_points(ax, size=3):
    for ln in ax.lines:
        if ln.get_marker() not in (None, 'None', 'none', ''):
            ln.set_markersize(size)


def draw_e(fig, series):
    ax = add(fig, 0.68, 6.46, 1.46, 1.00)
    f5.draw_i(ax, series['i_perf']['lines'], f5c.LIM_I_PERF[0], f5c.LIM_I_PERF[1],
              f5c.TICK_CONTRAST, f5c.TICK_PERF_Y)
    ys = np.concatenate([np.asarray(r['y'], float) for r in series['i_perf']['lines']])
    ymin = float(np.nanmin(ys))
    yhi = f5c.LIM_I_PERF[1][1]
    ax.set_ylim(ymin - 0.08 * (yhi - ymin), yhi)
    _smaller_points(ax)
    ax.set_ylabel('% correct')
    ax.set_xlabel('stimulus contrast')
    _shrink(ax)
    # White gap at zero contrast, between the concordant and discordant curves.
    ax.text(0.50, 0.52, f"R$^2$ con {series['i_perf']['r2_con']:.2f}",
            transform=ax.transAxes, fontsize=FS_R2, color='#030505',
            ha='center', va='center')
    ax.text(0.50, 0.38, f"R$^2$ incon {series['i_perf']['r2_inc']:.2f}",
            transform=ax.transAxes, fontsize=FS_R2, color='#808181',
            ha='center', va='center')

    ax = add(fig, 2.66, 6.46, 1.36, 1.00)
    f5.draw_i(ax, series['i_rt']['lines'], f5c.LIM_I_RT[0], f5c.LIM_I_RT[1],
              f5c.TICK_CONTRAST, f5c.TICK_RT_Y)
    _smaller_points(ax)
    ax.set_ylabel('reaction time (s)')
    ax.set_xlabel('stimulus contrast')
    _shrink(ax)
    ax.text(0.50, 0.08, f"R$^2$ {series['i_rt']['r2']:.2f}",
            transform=ax.transAxes, fontsize=FS_R2, color='#030505',
            ha='center', va='bottom')


def build():
    plt.close('all')
    fig = plt.figure(figsize=(W, H), dpi=150)
    fig.patch.set_facecolor('white')

    letter(fig, 0.04, 0.02, 'A')
    empty_box(add(fig, 0.24, 0.08, 5.80, 0.88))

    data = f4.Data()
    c_regs = sorted(data.integ_s | data.integ_c)
    m_regs = sorted(data.move_s | data.move_c)
    t_s, r_s, used_s, n_s = data.weighted_average(
        data.overlap, f4.UNSPLIT_TF, f4c.ALYX_RES, t_max=f4c.T_MAX_MS)
    t_c, r_c, used_c, n_c = data.weighted_average(c_regs, f4.STIM_TF)
    t_m, r_m, used_m, n_m = data.weighted_average(m_regs, f4.CHOICE_TF)
    packs = {
        'S': inset_pack(t_s, r_s),
        'C': inset_pack(t_c, r_c),
        'M': inset_pack(t_m, r_m),
    }
    ylim_b, yt_b = _ylim_insets(packs)
    letter(fig, 0.04, 1.16, 'B')
    draw_b(fig, packs, ylim_b, yt_b)

    data.af.pth_res = f4c.OA_RES
    y_c_s = data.load_group_sub(c_regs, f4.STIM_TF)
    y_m_s = data.load_group_sub(m_regs, f4.STIM_TF)
    y_c_m = data.load_group_sub(c_regs, f4.CHOICE_TF)
    y_m_m = data.load_group_sub(m_regs, f4.CHOICE_TF)
    stim = (np.linspace(0, 150, len(y_c_s)), y_c_s, y_m_s)
    move = (np.linspace(-150, 0, len(y_c_m)), y_c_m, y_m_m)
    ylim_c, yt_c = _panel_i_ylim([y_c_s, y_m_s, y_c_m, y_m_m])
    letter(fig, 0.04, 2.52, 'C')
    draw_c(fig, stim, move, ylim_c, yt_c)

    letter(fig, 0.04, 4.42, 'D')
    empty_box(add(fig, 0.24, 4.58, 1.78, 1.18))
    with f5.CACHE.open('rb') as handle:
        series = pickle.load(handle)
    draw_d_fits(fig, series)

    letter(fig, 0.04, 6.30, 'E')
    draw_e(fig, series)

    letter(fig, 4.18, 6.30, 'F')
    empty_box(add(fig, 4.40, 6.46, 1.62, 1.00))

    out = HERE / 'figure_v2'
    fig.savefig(out.with_suffix('.pdf'), facecolor='white')
    fig.savefig(out.with_suffix('.svg'), facecolor='white')
    fig.savefig(out.with_suffix('.png'), dpi=200, facecolor='white')
    plt.close(fig)
    print(f"S n={len(used_s)} cells={n_s}  "
          f"C n={len(used_c)} cells={n_c}  "
          f"M n={len(used_m)} cells={n_m}")
    print(f'inset ylim {ylim_b}  panel-i ylim {ylim_c}')
    print(f'wrote {out.with_suffix(".pdf")}')


if __name__ == '__main__':
    build()
