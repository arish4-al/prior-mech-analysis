#!/usr/bin/env python
"""Figure 5 on the published page, redrawn for the 80 ms full model.

Panels a and j stay empty. b–i use
``stageB_hold_s89_full_mpre3_meancell`` seed 89 (best full model ranked
on 80 ms tot; journals/stage_b_fit_variants.md).

  b  S pop-diff by contrast          plot_S_diff_by_contrast_side_with_data
  c  I/M trajectories, stim and move loss_plot_diff_by_condition_with_data
  d  I/M prior distance, 80 ms       loss_prior_effect (plot_stim=False)
  e  S distance by contrast          plot_S_dist_by_contrast
  f  C choice distance, stim         plot_dist_I_M_P  (dc_I)
  g  M choice distance, move         plot_dist_I_M_P  (dc_M), time to 0
  h  one block switch, ~9 trials     same traces as plot_overall_traj
  i  act-prior psychometric + RT     loss_perf_with_data on the cached run

Shared stim for b–h: bps=20, seed 12345. ΔAIC uses m_pre_weight=1
(the ranking protocol) against g=d=0 on the same stimuli.

  conda activate iblenv   # outside the sandbox
  python scripts/paper_figs/fig5/fig5_plot.py
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig5_common as fc

os.environ['PRIOR_MECH_NO_ONE'] = '1'
os.environ['ONE_CACHE_DIR'] = str(fc.OPENALYX)
os.chdir(fc.REPO)
sys.path.insert(0, str(fc.REPO))
sys.path.insert(0, str(fc.REPO / 'scripts'))

import one.api as one_api
_ONE = one_api.ONE


def _local_one(*args, **kwargs):
    return _ONE(cache_dir=str(fc.OPENALYX), mode='local', silent=True)


one_api.ONE = _local_one

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

import analysis_functions as af
import model_functions as mf
from model_functions import (
    compute_sse_stim_right,
    corrected_aicc_from_total_loss,
    int_regs,
    load_and_plot_stim_dist_data,
    loss_perf_with_data,
    loss_plot_diff_by_condition_with_data,
    loss_prior_effect,
    mean_S_by_contrast,
    mean_by_condition,
    move_regs,
    plot_S_diff_by_contrast_side_with_data,
    plot_S_dist_by_contrast,
    plot_dist_I_M_P,
    stim_regs,
)
from plot_best_fit_results import (
    ensure_fit_data_links_paper,
    load_mean_data_results,
    load_plot_model,
    make_shared_stimuli,
)
from _fit_data import load_avg_mean_r

plt.rcParams.update({
    'font.family': 'Helvetica',
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.labelsize': 14,
})

FIG_W = fc.PAGE_WH[0] / 72
FIG_H = fc.PAGE_WH[1] / 72
SCRATCH = fc.FIG_DIR / '_scratch'
CACHE = fc.FIG_DIR / 'cache' / 'series.pkl'

# Least-squares scales of this model's distance curves on the plotted window.
# Notebook cell 72 used 1.05 on every S contrast; cell 76 used
# {'int': 0.67, 'move': 0.88}. Those were for the published weights.
# e uses one factor for every contrast. f is 0–120 ms. g is the full
# move-aligned curve, which sits inside the panel.
F_SCALE = 0.590
G_SCALE = 0.795
E_SCALE = 0.906
S_DIST_SCALE = {c: E_SCALE for c in (0.0, 0.0625, 0.125, 0.25, 1.0)}
DIST_SCALE = {'int': F_SCALE, 'move': G_SCALE}
S_DIST_SHIFT = {0.0: 0, 0.0625: 0, 0.125: 0, 0.25: 0, 1.0: 0}
STIM_SPLITS = ['stim_1.0', 'stim_0.25', 'stim_0.125', 'stim_0.0625', 'stim_0.0']
PRIOR_KEYS = ('g_i', 'd_i', 'g_m', 'd_m', 'g_s', 'd_s')
K_FIT, K_NULL = 21, 15  # train_mask is 21 True; null freezes the six g/d

PRIOR_REGIONS = {
    'int_regs_choice': int_regs,
    'int_regs_stim': int_regs,
    'move_regs_choice': move_regs,
    'move_regs_stim': move_regs,
    'stim_regs': ['VISpm', 'FRP', 'VISal'],
}


def _sp_sign(x: float) -> float:
    """P_L − P_R < 0 → +1 (right). Same rule as scripts/_tmp_perf_rt_model_vs_data.py."""
    return 1.0 if float(x) < 0.0 else -1.0


def results_with_subjective_prior(results: dict) -> dict:
    out = dict(results)
    new_blocks = []
    for i, sp in enumerate(results['sub_prior']):
        arr = np.asarray(sp, dtype=float).reshape(-1)
        sign = _sp_sign(arr[0])
        n = len(results['block_sides'][i])
        new_blocks.append(np.full(n, sign, dtype=float))
    out['block_sides'] = new_blocks
    return out


def _new_figs(before):
    nums = [n for n in plt.get_fignums() if n not in before]
    return [plt.figure(n) for n in nums]


def _harvest(ax):
    rows = []
    for ln in ax.lines:
        x = np.asarray(ln.get_xdata(), dtype=float)
        y = np.asarray(ln.get_ydata(), dtype=float)
        if x.size < 2 or not np.isfinite(x).any():
            continue
        if float(np.nanmax(x) - np.nanmin(x)) < 1e-6:
            continue
        if np.allclose(y, 0.0, atol=1e-12):
            continue
        rows.append({
            'x': x,
            'y': y,
            'color': ln.get_color(),
            'ls': ln.get_linestyle(),
            'lw': float(ln.get_linewidth() or 2),
            'marker': ln.get_marker(),
            'ms': float(ln.get_markersize() or 0),
            'alpha': ln.get_alpha(),
        })
    return rows


def _draw_lines(ax, lines, lw=None):
    # Solid data under dashed model.
    ordered = sorted(lines, key=lambda r: 0 if r['ls'] in ('-', 'solid') else 1)
    for r in ordered:
        kw = dict(color=r['color'], linestyle=r['ls'], linewidth=lw or r['lw'],
                  solid_capstyle='round', zorder=3)
        marker = r.get('marker') or 'None'
        if marker not in ('None', 'none', '', None):
            kw['marker'] = marker
            kw['markersize'] = min(r.get('ms') or 5, 5)
        if r.get('alpha') is not None:
            kw['alpha'] = r['alpha']
        ax.plot(r['x'], r['y'], **kw)


def _maybe_widen(ax, lines, ylim):
    ys = np.concatenate([np.asarray(r['y'], float) for r in lines]) if lines else None
    lo, hi = ylim
    if ys is not None and np.isfinite(ys).any():
        ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
        span = hi - lo
        if ymin < lo:
            lo = ymin - 0.04 * span
        if ymax > hi:
            hi = ymax + 0.04 * span
    ax.set_ylim(lo, hi)


def _style_ax(ax, xlim, ylim, xticks, yticks, lines, event, ylabel=None):
    _draw_lines(ax, lines)
    ax.axvline(0, color=event, linewidth=2, zorder=2, solid_capstyle='butt')
    ax.set_xlim(*xlim)
    _maybe_widen(ax, lines, ylim)
    ax.set_xticks(list(xticks))
    ax.set_yticks(list(yticks))
    ax.tick_params(length=3, width=0.6, colors='#030505', pad=1)
    for sp in ax.spines.values():
        sp.set_color('black')
        sp.set_linewidth(0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('time (ms)', fontsize=14, color='#030505', labelpad=1)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, color='#030505', labelpad=2)
    ax.set_facecolor('white')


def _r2(ax, text, x, y, color, size=10, ha='left'):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=size, color=color,
            ha=ha, va='top')


def _pt(fig, x, y, s, size, color='#231F20', ha='left', va='top', weight='regular'):
    fig.text(x / fc.PAGE_WH[0], 1 - y / fc.PAGE_WH[1], s,
             fontsize=size, color=color, ha=ha, va=va, fontweight=weight)


def letter(fig, x, y, s, size=21):
    _pt(fig, x, max(y, 1.0), s, size)


def badge_layer(fig):
    ax = fig.add_axes([0, 0, 1, 1], zorder=6)
    ax.set_xlim(0, fc.PAGE_WH[0])
    ax.set_ylim(fc.PAGE_WH[1], 0)
    ax.axis('off')
    ax.patch.set_alpha(0)
    return ax


def badge(ax, x, y, letter_, color):
    ax.add_patch(Circle((x, y), 9.5, facecolor=color, edgecolor='#231F20',
                        linewidth=0.5, clip_on=False, zorder=6))
    ax.text(x, y, letter_, fontsize=11.5, ha='center', va='center',
            color='#231F20', zorder=7)


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------
def _loss_pair(results, steps_before_obs, sim_out, mp, plot):
    _, mean_data = load_mean_data_results()
    traj = loss_plot_diff_by_condition_with_data(
        sim_out, mp, var_names=('I', 'M'),
        mean_data_results=mean_data,
        plot=plot, save_dir=str(SCRATCH),
    )
    prior = loss_prior_effect(
        regions=PRIOR_REGIONS, results=results, model_params=mp,
        steps_before_obs=steps_before_obs, T=72, model_metric='l2',
        timeframes=('act_block_duringstim', 'act_block_duringchoice'),
        ptype='p_mean_c', plot_window=80, reload=False,
        label_A='integrator', label_B='move', do_plot=plot,
        plot_shifted=False, ylim=None, scale_factors=[1, 1, 1],
        include_all_trials=True, save_dir=str(SCRATCH),
        plot_stim=False, lump_all=False, include_stim=False,
    )
    return traj, prior


def _flip_move(lines):
    """Pre-move samples are stored earliest-first; the paper ends them at 0."""
    out = []
    for r in lines:
        x = np.asarray(r['x'], float)
        t = x - x[-1]
        out.append({**r, 'x': t})
    return out


def _single_trial(results, steps_before_obs):
    sides = []
    for bs in results['block_sides']:
        b = np.asarray(bs, dtype=float).reshape(-1)
        sides.append(int(np.sign(b[0])) if b.size else 0)
    n = len(sides)
    switch = None
    for i in range(1, n):
        if sides[i] != sides[i - 1] and sides[i] != 0 and sides[i - 1] != 0:
            if i - 4 >= 0 and i + 5 <= n:
                switch = i
                break
    if switch is None:
        i0, i1 = 0, min(9, n)
    else:
        i0, i1 = switch - 4, switch + 5
    lens = [len(ts) for ts in results['trial_sides']]
    starts = np.cumsum([0] + lens[:-1]).astype(int)
    start = int(starts[i0])
    length = int(starts[i1 - 1] + lens[i1 - 1] - start)

    def sl(key, scale=1.0):
        return np.asarray(results[key][start:start + length], dtype=float) * scale

    trial_sides = np.concatenate(results['trial_sides'])[start:start + length]
    trial_strengths = np.concatenate(results['trial_strengths'])[start:start + length]
    weighted = np.asarray(trial_sides * trial_strengths + 0.05, dtype=float)
    block = np.concatenate([np.asarray(b, dtype=float).reshape(-1)
                            for b in results['block_sides']])[start:start + length]
    masked = np.zeros(length, dtype=float)
    action = np.asarray(results.get('action_time', []), dtype=float)
    action = action[np.isfinite(action)] if action.size else action
    obs_x, act_x, bracket = [], [], None
    for ts, te in zip(starts, starts + np.asarray(lens)):
        t_obs = int(ts + steps_before_obs)
        if action.size == 0:
            continue
        in_trial = action[(action >= ts) & (action < te)]
        if in_trial.size == 0:
            continue
        t_act = float(in_trial[0])
        lo = max(int(t_obs - start), 0)
        hi = min(int(t_act - start), length)
        if hi > lo:
            masked[lo:hi] = weighted[lo:hi]
        if start <= t_obs < start + length:
            obs_x.append(int(t_obs - start))
        if start <= t_act < start + length:
            act_x.append(float(t_act - start))
        if bracket is None and hi > lo and lo > length * 0.25:
            bracket = (lo, hi)
    masked[masked == 0] = np.nan
    return {
        'S': sl('S'), 'I': sl('I'), 'M': sl('M'), 'P': sl('P', 10.0),
        'trial': masked, 'block': np.asarray(block, dtype=float),
        'obs': np.asarray(obs_x, dtype=float),
        'act': np.asarray(act_x, dtype=float),
        'bracket': bracket,
        'start': start, 'length': length, 'i0': i0, 'i1': i1,
    }


def _total_r2_from_lines(lines, xmin=None, xmax=None):
    """Pooled R², solid = data, dashed = model. Optional x window."""
    data, model = [], []
    solids = [r for r in lines if r['ls'] in ('-', 'solid')]
    dashes = [r for r in lines if r['ls'] not in ('-', 'solid')]
    for s, m in zip(solids, dashes):
        n = min(len(s['y']), len(m['y']), len(m['x']))
        x = np.asarray(m['x'][:n], float)
        keep = np.ones(n, dtype=bool)
        if xmin is not None:
            keep &= x >= xmin
        if xmax is not None:
            keep &= x <= xmax
        data.append(np.asarray(s['y'][:n], float)[keep])
        model.append(np.asarray(m['y'][:n], float)[keep])
    if not data:
        return float('nan')
    d = np.concatenate(data)
    m = np.concatenate(model)
    sse = float(np.sum((m - d) ** 2))
    sst = float(np.sum((d - np.mean(d)) ** 2) + 1e-12)
    return 1.0 - sse / sst


def compute(mp):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    mf.save_dir = SCRATCH
    mf.pth_res = fc.OPENALYX / 'manifold' / 'res'
    af.save_dir = SCRATCH
    af.pth_res = mf.pth_res
    ensure_fit_data_links_paper()
    _, mean_data = load_mean_data_results()
    _, avg_mean_R = load_avg_mean_r()

    print('loading S distance and choice-distance curves')
    plt.close('all')
    avg_stim = load_and_plot_stim_dist_data(stim_regs, STIM_SPLITS)
    real = {}
    data_regs = {
        'int_stim': int_regs, 'int': int_regs,
        'move': move_regs, 'prior': mf.prior_regs_int,
    }
    data_times = {
        'int_stim': 'stim_duringstim_act', 'int': 'choice_duringstim_act',
        'move': 'choice_duringchoice_act', 'prior': 'act_intertrial0',
    }
    for name, regs in data_regs.items():
        real[name] = af.plot_average_distance_over_regions(
            regs, timewindow=data_times[name], name=name,
            ptype='p_mean_c', show_y=True, alpha=0.01,
            plot_p_per_time=False, plot_gain=True, return_mean=True,
        )
    plt.close('all')

    print('simulating bps=20 seed 12345')
    bundle = make_shared_stimuli(mp, bps=20, seed=12345)
    stimuli, trial_strengths, trial_sides, block_sides, steps_before_obs, bps = bundle
    results = mf.run_model(
        'data', stimuli, trial_strengths, trial_sides, block_sides, bps,
        steps_before_obs=steps_before_obs, verbose=False, backend='numba', **mp,
    )
    T_post, T_pre = mf.im_traj_T_of(mp)
    sim_out = mean_by_condition(results, steps_before_obs, T=T_post, T_pre=T_pre)
    S_avg = mean_S_by_contrast(results, steps_before_obs)

    mp_loss = dict(mp)
    mp_loss['m_pre_weight'] = 1.0

    plt.close('all')
    before = set(plt.get_fignums())
    sse_s = compute_sse_stim_right(S_avg, avg_mean_R, baseline_R=0)
    plot_S_diff_by_contrast_side_with_data(
        S_avg, {}, avg_mean_R, baseline=0, save_dir=str(SCRATCH),
        ylim=[-0.14, 0.75],
    )
    b_figs = _new_figs(before)
    b_lines = _harvest(b_figs[0].axes[0]) if b_figs else []

    before = set(plt.get_fignums())
    loss_im = loss_plot_diff_by_condition_with_data(
        sim_out, mp_loss, var_names=('I', 'M'),
        mean_data_results=mean_data, plot=True, save_dir=str(SCRATCH),
    )
    c_figs = _new_figs(before)
    # post figure is created first, then pre.
    c_stim = _harvest(c_figs[0].axes[0]) if len(c_figs) > 0 else []
    c_move = _harvest(c_figs[1].axes[0]) if len(c_figs) > 1 else []
    gof = loss_im.get('gof') or {}

    before = set(plt.get_fignums())
    loss_prior = loss_prior_effect(
        regions=PRIOR_REGIONS, results=results, model_params=mp_loss,
        steps_before_obs=steps_before_obs, T=72, model_metric='l2',
        timeframes=('act_block_duringstim', 'act_block_duringchoice'),
        ptype='p_mean_c', plot_window=80, reload=False,
        label_A='integrator', label_B='move', do_plot=True,
        plot_shifted=False, ylim=None, scale_factors=[1, 1, 1],
        include_all_trials=True, save_dir=str(SCRATCH),
        plot_stim=False, lump_all=False, include_stim=False,
    )
    d_figs = _new_figs(before)
    d_axes = d_figs[0].axes if d_figs else []
    d_stim = _harvest(d_axes[0]) if len(d_axes) > 0 else []
    d_move = _harvest(d_axes[1]) if len(d_axes) > 1 else []

    before = set(plt.get_fignums())
    plot_S_dist_by_contrast(
        S_avg, avg_stim, scale_factors=S_DIST_SCALE, time_shifts_steps=S_DIST_SHIFT,
        save_dir=str(SCRATCH), ylim=[-0.05, 1.2], yticks=[0, 0.4, 0.8, 1.2],
    )
    e_figs = _new_figs(before)
    e_lines = []
    for fig in e_figs:
        got = _harvest(fig.axes[0])
        if len(got) > len(e_lines):
            e_lines = got

    before = set(plt.get_fignums())
    plot_dist_I_M_P(sim_out, real, scale_factors=DIST_SCALE, save_dir=str(SCRATCH))
    dist_figs = _new_figs(before)
    golds, tomato = [], []
    for fig in dist_figs:
        lines = _harvest(fig.axes[0])
        col = str(lines[0]['color']) if lines else ''
        if col == 'tomato':
            tomato = lines
        elif col == 'gold':
            golds.append(lines)
    # creation order: ds_I (trial side), dc_I (choice). Panel f is the choice one.
    f_lines = golds[1] if len(golds) > 1 else (golds[0] if golds else [])
    g_lines = _flip_move(tomato)

    h = _single_trial(results, steps_before_obs)

    print('null model (g = d = 0) for ΔAIC')
    mp_null = dict(mp_loss)
    for k in PRIOR_KEYS:
        mp_null[k] = 0.0
    results_null = mf.run_model(
        'data', stimuli, trial_strengths, trial_sides, block_sides, bps,
        steps_before_obs=steps_before_obs, verbose=False, backend='numba',
        **mp_null,
    )
    sim_null = mean_by_condition(
        results_null, steps_before_obs, T=T_post, T_pre=T_pre)
    loss_im_null, loss_prior_null = _loss_pair(
        results_null, steps_before_obs, sim_null, mp_null, plot=False)
    aic_fit = corrected_aicc_from_total_loss(loss_prior, loss_im, K_FIT)
    aic_null = corrected_aicc_from_total_loss(loss_prior_null, loss_im_null, K_NULL)
    delta = float(aic_fit['aic'] - aic_null['aic'])
    print(f"ΔAIC = {delta:.2f}  (fit {aic_fit['aic']:.2f}, null {aic_null['aic']:.2f})")

    print('act-prior psychometric from cached sessions')
    beh = np.load(fc.BEHAVIOR_ACT, allow_pickle=True).flat[0]
    loaded = np.load(fc.PSYCH_CACHE, allow_pickle=True).item()
    psych = results_with_subjective_prior(loaded)
    plt.close('all')
    before = set(plt.get_fignums())
    sse_perf = loss_perf_with_data(
        psych, beh, mp, metric='correct', dt=1.0, do_plot=True,
        save_dir=None, log_xaxis=True,
    )
    perf_figs = _new_figs(before)
    before = set(plt.get_fignums())
    dt_s = mf.model_step_s()  # one model step = one data bin (12.5/6 ms)
    sse_rt = loss_perf_with_data(
        psych, beh, mp, metric='rt', dt=dt_s, do_plot=True,
        save_dir=None, log_xaxis=True, rt_mode='combined_all',
    )
    rt_figs = _new_figs(before)
    perf_lines = _harvest(perf_figs[0].axes[0]) if perf_figs else []
    rt_lines = _harvest(rt_figs[0].axes[0]) if rt_figs else []
    plt.close('all')

    series = {
        'b': {'lines': b_lines, 'r2': float(sse_s.get('total_gof_r2', np.nan))},
        'c_stim': {
            'lines': c_stim,
            'r2_I': float((gof.get('I') or {}).get('post', np.nan)),
            'r2_M': float((gof.get('M') or {}).get('post', np.nan)),
        },
        'c_move': {
            'lines': c_move,
            'r2_I': float((gof.get('I') or {}).get('pre', np.nan)),
            'r2_M': float((gof.get('M') or {}).get('pre', np.nan)),
        },
        'd_stim': {'lines': d_stim, 'r2': float(loss_prior.get('gof', np.nan))},
        'd_move': {'lines': d_move},
        'e': {'lines': e_lines, 'r2': _total_r2_from_lines(e_lines)},
        'f': {'lines': f_lines, 'r2': _total_r2_from_lines(f_lines, xmin=0.0, xmax=120.0)},
        'g': {'lines': g_lines, 'r2': _total_r2_from_lines(g_lines)},
        'h': h,
        'i_perf': {
            'lines': perf_lines,
            'r2_con': float((sse_perf.get('gof') or {}).get('congruent', np.nan)),
            'r2_inc': float((sse_perf.get('gof') or {}).get('incongruent', np.nan)),
        },
        'i_rt': {
            'lines': rt_lines,
            'r2': float((sse_rt.get('gof') or {}).get('total', np.nan)),
        },
        'delta_aic': delta,
        'aic_fit': float(aic_fit['aic']),
        'aic_null': float(aic_null['aic']),
        'loss_fit': float(aic_fit['components']['total_loss']),
        'loss_null': float(aic_null['components']['total_loss']),
    }
    return series


# ---------------------------------------------------------------------------
# draw
# ---------------------------------------------------------------------------
Y_B = r'$|r_{S}^{\mathrm{stim},s}(t;c)|$'
Y_C = r'$r_{\{C,M\}}^{\mathrm{choice},\{s,m\}}(t)$'
Y_D = r'$d_{\{C,M\}}^{\mathrm{prior},\{s,m\}}(t)$'
Y_E = r'$d_{S}^{\mathrm{stim},s}(t)$'
Y_F = r'$d_{C}^{\mathrm{choice},s}(t)$'
Y_G = r'$d_{M}^{\mathrm{choice},m}(t)$'


def _ax(fig, box):
    ax = fig.add_axes(fc.fig_box(*box))
    ax.set_facecolor('white')
    return ax


def draw_h(fig, h):
    ax = _ax(fig, fc.AX_H)
    for x in h['obs']:
        ax.axvline(x, color=fc.C_H_STIM_LINE, ls='--', lw=0.8, alpha=0.9, zorder=1)
    for x in h['act']:
        ax.axvline(x, color=fc.C_H_MOVE_LINE, ls='--', lw=0.8, alpha=0.9, zorder=1)
    specs = (
        ('S', 1, fc.C_S_TRACE, 1.0, 's$_R$', '#1d78b5'),
        ('I', 1, fc.C_I_TRACE, 1.0, 'c$_R$', '#fbab1a'),
        ('M', 1, fc.C_M_TRACE, 1.0, 'm$_R$', '#ed2124'),
        ('P', 1, fc.C_P_TRACE, 1.0, 'p$_R$', '#7850a1'),
        ('S', 0, fc.C_S_TRACE, 0.5, 's$_L$', '#558dca'),
        ('I', 0, fc.C_I_TRACE, 0.5, 'c$_L$', '#ffc50f'),
        ('M', 0, fc.C_M_TRACE, 0.5, 'm$_L$', '#ed2124'),
        ('P', 0, fc.C_P_TRACE, 0.5, 'p$_L$', '#7850a1'),
    )
    labeled = []
    for key, col, color, alpha, name, tcol in specs:
        y = np.asarray(h[key][:, col], float)
        ax.plot(y, color=color, lw=0.5, alpha=alpha, zorder=2)
        labeled.append((name, tcol, y))
    ax.plot(h['trial'], color='black', lw=0.5, zorder=3)
    ax.plot(h['block'], color=fc.C_BLOCK, lw=2, alpha=0.7, zorder=1)
    n = len(h['block'])
    ax.set_xlim(-0.01 * n, 1.03 * n)
    ax.margins(y=0.08)
    ax.axis('off')
    if len(h['obs']) and len(h['act']):
        x0 = int(h['obs'][0])
        x1 = int(min(h['act'][0], n - 1))
    else:
        x0, x1 = 0, min(20, n - 1)
    # Names in the ITI just after the first pulse, stacked in axes y so the
    # ±1 block step does not crush them onto the zero line. Top = highest trace.
    peaks = []
    for name, tcol, y in labeled:
        seg = y[x0:x1 + 1]
        yi = float(np.nanmax(np.abs(seg))) if np.isfinite(seg).any() else 0.0
        peaks.append((yi, name, tcol))
    peaks.sort(key=lambda t: -t[0])
    lab_trans = ax.get_xaxis_transform()
    x_lab = x1 + 36
    ys = np.linspace(0.90, 0.28, len(peaks))
    for (_, name, tcol), yf in zip(peaks, ys):
        ax.text(x_lab, yf, name, color=tcol, fontsize=11.3, ha='left', va='center',
                transform=lab_trans)
    # "stim" bracket under the first trial. "block side" on the first grey step,
    # clear of that bracket.
    trans = ax.get_xaxis_transform()
    ax.annotate(
        '', xy=(x1, 0.08), xytext=(x0, 0.08), xycoords=trans, textcoords=trans,
        arrowprops=dict(arrowstyle='-', color='#231F20', lw=0.8),
    )
    ax.plot([x0, x0], [0.05, 0.11], color='#231F20', lw=0.8, transform=trans, clip_on=False)
    ax.plot([x1, x1], [0.05, 0.11], color='#231F20', lw=0.8, transform=trans, clip_on=False)
    ax.text((x0 + x1) / 2, 0.02, 'stim', color='#231F20', fontsize=13,
            ha='center', va='bottom', transform=trans)
    mid_block = int(np.clip(0.28 * n, 0, n - 1))
    b0 = float(h['block'][mid_block])
    ax.text(mid_block, b0, 'block side', color='#737477', fontsize=13,
            ha='center', va='top' if b0 > 0 else 'bottom')
    ax.text(0.995, 0.50, '0', transform=ax.transAxes, fontsize=12,
            color='#231F20', ha='right', va='center')


def draw_i(ax, lines, xlim, ylim, xticks, yticks):
    _draw_lines(ax, lines, lw=1.6)
    ax.set_xscale('symlog', linthresh=0.03125, linscale=1.0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(list(xticks))
    ax.set_xticklabels(['-1', '-0.1', '0', '0.1', '1'])
    ax.set_yticks(list(yticks))
    ax.tick_params(length=3, width=0.6, colors='#030505', labelsize=12, pad=1)
    for sp in ax.spines.values():
        sp.set_color('black')
        sp.set_linewidth(0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('stimulus contrast', fontsize=13.8, color='#030505', labelpad=2)
    ax.set_facecolor('white')


def draw(series):
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=72)
    fig.patch.set_facecolor('white')

    ax = _ax(fig, fc.AX_B)
    _style_ax(ax, fc.LIM_B[0], fc.LIM_B[1], fc.TICK_B_X, fc.TICK_B_Y,
              series['b']['lines'], fc.C_STIM_ON, Y_B)
    _r2(ax, f"R$^2$ {series['b']['r2']:.2f}", 0.98, 0.98, fc.C_R2_S, ha='right')
    ax.text(0.98, 0.90, 'contrast', transform=ax.transAxes, fontsize=11.6,
            color='#231F20', ha='right', va='top')
    ax.annotate(
        '', xy=(0.97, 0.78), xytext=(0.97, 0.86),
        xycoords=ax.transAxes, textcoords=ax.transAxes,
        arrowprops=dict(arrowstyle='->', color='#231F20', lw=0.8),
    )

    ax = _ax(fig, fc.AX_C_STIM)
    _style_ax(ax, fc.LIM_C_STIM[0], fc.LIM_C_STIM[1], fc.TICK_B_X, fc.TICK_C_STIM_Y,
              series['c_stim']['lines'], fc.C_STIM_ON, Y_C)
    _r2(ax, f"R$^2$ {series['c_stim']['r2_I']:.3f}", 0.04, 0.98, fc.C_R2_I)
    _r2(ax, f"R$^2$ {series['c_stim']['r2_M']:.3f}", 0.04, 0.86, fc.C_R2_M)

    ax = _ax(fig, fc.AX_C_MOVE)
    _style_ax(ax, fc.LIM_C_MOVE[0], fc.LIM_C_MOVE[1], fc.TICK_C_MOVE_X,
              fc.TICK_C_MOVE_Y, series['c_move']['lines'], fc.C_MOVE_ON)
    _r2(ax, f"R$^2$ {series['c_move']['r2_I']:.3f}", 0.04, 0.98, fc.C_R2_I)
    _r2(ax, f"R$^2$ {series['c_move']['r2_M']:.3f}", 0.04, 0.86, fc.C_R2_M)

    ax = _ax(fig, fc.AX_D_STIM)
    _style_ax(ax, fc.LIM_D_STIM[0], fc.LIM_D_STIM[1], fc.TICK_D_X, fc.TICK_D_Y,
              series['d_stim']['lines'], fc.C_STIM_ON, Y_D)
    _r2(ax, f"R$^2$: {series['d_stim']['r2']:.3f}", 0.02, 0.98, '#030505')

    ax = _ax(fig, fc.AX_D_MOVE)
    _style_ax(ax, fc.LIM_D_MOVE[0], fc.LIM_D_MOVE[1], fc.TICK_D_X_MOVE, fc.TICK_D_Y,
              series['d_move']['lines'], fc.C_MOVE_ON)
    ax.tick_params(labelleft=False)
    _pt(fig, 927, 2, f"ΔAIC = {series['delta_aic']:.2f}", 13)

    ax = _ax(fig, fc.AX_E)
    _style_ax(ax, fc.LIM_E[0], fc.LIM_E[1], fc.TICK_E_X, fc.TICK_E_Y,
              series['e']['lines'], fc.C_STIM_ON, Y_E)
    _r2(ax, f"R$^2$ {series['e']['r2']:.3f}", 0.98, 0.98, fc.C_R2_S, ha='right')

    ax = _ax(fig, fc.AX_F)
    _style_ax(ax, fc.LIM_F[0], fc.LIM_F[1], fc.TICK_F_X, fc.TICK_F_Y,
              series['f']['lines'], fc.C_STIM_ON, Y_F)
    _r2(ax, f"R$^2$ = {series['f']['r2']:.3f}", 0.98, 0.98, fc.C_R2_F, ha='right')

    ax = _ax(fig, fc.AX_G)
    g_ticks = [-144, -120, -80, -40, 0]
    _style_ax(ax, fc.LIM_G[0], fc.LIM_G[1], g_ticks, fc.TICK_G_Y,
              series['g']['lines'], fc.C_MOVE_ON, Y_G)
    ax.set_xticklabels(['', '−120', '−80', '−40', '0'])
    # Upper-left: the curve fills the upper right, where the PDF puts this R².
    _r2(ax, f"R$^2$ = {series['g']['r2']:.3f}", 0.06, 0.72, fc.C_R2_G, ha='left')

    draw_h(fig, series['h'])

    ax = _ax(fig, fc.AX_I_PERF)
    draw_i(ax, series['i_perf']['lines'], fc.LIM_I_PERF[0], fc.LIM_I_PERF[1],
           fc.TICK_CONTRAST, fc.TICK_PERF_Y)
    ax.set_ylabel('% correct', fontsize=13.8, color='#030505')
    _r2(ax, f"R$^2$ con: {series['i_perf']['r2_con']:.3f}",
        0.98, 0.24, '#030505', size=8.4, ha='right')
    _r2(ax, f"R$^2$ incon: {series['i_perf']['r2_inc']:.3f}",
        0.98, 0.14, '#808181', size=8.4, ha='right')

    ax = _ax(fig, fc.AX_I_RT)
    draw_i(ax, series['i_rt']['lines'], fc.LIM_I_RT[0], fc.LIM_I_RT[1],
           fc.TICK_CONTRAST, fc.TICK_RT_Y)
    ax.set_ylabel('reaction time (s)', fontsize=13.8, color='#030505')
    _r2(ax, f"R$^2$: {series['i_rt']['r2']:.3f}", 0.04, 0.20, '#030505', size=8.5)

    for x, y, s in (
        (5.4, 0, 'a'), (192.0, 0, 'b'), (393.4, 0, 'c'), (766.9, 0, 'd'),
        (35.0, 172.6, 'e'), (402.1, 172.6, 'f'), (762.8, 173.3, 'g'),
        (4.9, 343.3, 'h'), (8.3, 485.6, 'i'), (744.2, 484.6, 'j'),
    ):
        letter(fig, x, y, s, 22 if s == 'a' else (19 if s == 'g' else 21))

    layer = badge_layer(fig)
    for x, y, s, col in fc.BADGES:
        badge(layer, x, y, s, col)
    return fig


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recompute', action='store_true',
                    help='ignore the cached series and resimulate')
    args = ap.parse_args()
    fc.FIG_DIR.mkdir(parents=True, exist_ok=True)
    (fc.FIG_DIR / 'original').mkdir(parents=True, exist_ok=True)
    (fc.FIG_DIR / 'panels').mkdir(parents=True, exist_ok=True)
    if fc.ORIGINAL_PDF.is_file():
        dest = fc.FIG_DIR / 'original' / 'figure5.pdf'
        if not dest.is_file():
            shutil.copy2(fc.ORIGINAL_PDF, dest)

    mp, meta = load_plot_model(fc.WEIGHTS_JSON)
    if args.recompute or not CACHE.is_file():
        series = compute(mp)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with CACHE.open('wb') as f:
            pickle.dump(series, f, protocol=4)
        print(f'cached {CACHE}')
    else:
        with CACHE.open('rb') as f:
            series = pickle.load(f)
        print(f'loaded {CACHE}')

    fig = draw(series)
    fc.savefig_svg_png_pdf(fig, fc.FIG_DIR / 'figure5', dpi=150)
    plt.close(fig)

    summary = {
        'model': 'stageB_hold_s89_full_mpre3_meancell',
        'seed': 89,
        'json': str(fc.WEIGHTS_JSON),
        'ranked_on': '80 ms tot',
        'tot80': 1.054,
        'm_pre_weight_in_json': meta.get('model_params', {}).get('m_pre_weight',
                                                                  mp.get('m_pre_weight')),
        'm_pre_weight_for_aic': 1.0,
        'k_fit': K_FIT,
        'k_null': K_NULL,
        'delta_aic': series['delta_aic'],
        'aic_fit': series['aic_fit'],
        'aic_null': series['aic_null'],
        'r2': {
            'b': series['b']['r2'],
            'c_stim_I': series['c_stim']['r2_I'],
            'c_stim_M': series['c_stim']['r2_M'],
            'c_move_I': series['c_move']['r2_I'],
            'c_move_M': series['c_move']['r2_M'],
            'd': series['d_stim']['r2'],
            'e': series['e']['r2'],
            'f': series['f']['r2'],
            'g': series['g']['r2'],
            'i_con': series['i_perf']['r2_con'],
            'i_inc': series['i_perf']['r2_inc'],
            'i_rt': series['i_rt']['r2'],
        },
        'h_window': {
            'start': series['h']['start'], 'length': series['h']['length'],
            'trials': [series['h']['i0'], series['h']['i1']],
        },
        'scales': {
            'S_dist': S_DIST_SCALE,
            'dc_I': F_SCALE,
            'dc_M': G_SCALE,
            'notebook_S_dist': 1.05,
            'notebook_dc_I': 0.67,
            'notebook_dc_M': 0.88,
        },
    }
    (fc.FIG_DIR / 'summary.json').write_text(json.dumps(_jsonable(summary), indent=2))
    print(json.dumps(_jsonable(summary['r2']), indent=2))
    print(f"ΔAIC = {series['delta_aic']:.2f}")
    print(f"wrote {fc.FIG_DIR / 'figure5.pdf'}")


if __name__ == '__main__':
    main()
