#!/usr/bin/env python
"""SI figure: recovery of cell class and sensory prior on simulated data.

Reads the 2026-08-13 Stage B runs (regular s101, sensory s23). Does not
re-simulate. Two ground truths:

  no sensory prior   g_s = d_s = 0, fitted I/M prior   (regular s101)
  full model         d_s fitted, g_s ≈ 0, I/M prior on (full mpre3 s89)

  a  BWM classifier on each simulation (S / I / M).
  b  S prior distance under stim × choice vs stim side only.

  python scripts/paper_figs/si_recovery/si_recovery_plot.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import si_recovery_common as sc

plt.rcParams.update({
    'font.family': 'Helvetica',
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.labelsize': 13,
})

PAGE = (7.55, 5.35)  # inches
POP_ORDER = ('S', 'I', 'M')


def _bool(value):
    return str(value).strip().lower() in ('1', 'true', 'yes')


def _read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def _unsplit_csv(spec):
    path = spec['unsplit_csv']
    if path is not None:
        return Path(path)
    matches = sorted(
        p for p in sc.FULL.glob('unsplit_prior/seed_123/presence_*_unsplit/figs/*_summary.csv')
        if 'harris' not in p.name
    )
    if len(matches) != 1:
        raise SystemExit(f'expected one presence unsplit summary, found {matches}')
    return matches[0]


def load_condition(spec):
    rows = {r['region']: r for r in _read_csv(spec['class_csv'])}
    classes = []
    n_correct = 0
    for pop in POP_ORDER:
        row = rows[pop]
        correct = _bool(row['correct'])
        n_correct += int(correct)
        classes.append({
            'pop': pop,
            'true': row['true'],
            'pred': row['pred'],
            'correct': correct,
            'sigma_stim_s': float(row['sigma_stim_s']),
            'sigma_stim_m': float(row['sigma_stim_m']),
            'monotonicity': int(float(row['monotonicity'])),
        })
    split = _read_csv(spec['split_s_csv'])
    split_s = next(r for r in split if r['population'] == 'S')
    unsplit = {r['population']: r for r in _read_csv(_unsplit_csv(spec))}
    strata = {}
    for name, row in (
        ('stim_x_choice', split_s),
        ('stim_only', unsplit['S']),
    ):
        strata[name] = {
            'curve_mean': float(row['curve_mean']),
            'p_mean': float(row['p_mean']),
            'significant': _bool(row['significant_p_mean']),
        }
    im_stim_only = {
        pop: {
            'curve_mean': float(unsplit[pop]['curve_mean']),
            'p_mean': float(unsplit[pop]['p_mean']),
            'significant': _bool(unsplit[pop]['significant_p_mean']),
        }
        for pop in ('I', 'M')
    }
    return {
        'key': spec['key'],
        'title': spec['title'],
        'model': spec['model'],
        'ground_truth': spec['ground_truth'],
        'n_correct': n_correct,
        'classes': classes,
        'strata': strata,
        'im_stim_only': im_stim_only,
    }


def _zone_shades(ax):
    """Classifier regions: stimulus if Σ > 0.8, else movement if monotonic."""
    x0, x1, y0, y1 = -0.02, 1.16, -0.42, 1.42
    zones = (
        (x0, sc.SIGMA_STIM, y0, 0.5, sc.C_I, 0.18),
        (x0, sc.SIGMA_STIM, 0.5, y1, sc.C_M, 0.14),
        (sc.SIGMA_STIM, x1, y0, y1, sc.C_S, 0.18),
    )
    for xa, xb, ya, yb, color, alpha in zones:
        ax.add_patch(Rectangle(
            (xa, ya), xb - xa, yb - ya, facecolor=color, edgecolor='none',
            alpha=alpha, zorder=0, clip_on=True,
        ))
    ax.axvline(sc.SIGMA_STIM, color='0.35', lw=0.7, ls='--', zorder=1)
    ax.text(0.97, 1.22, 'stimulus', ha='right', va='top', fontsize=9,
            color='0.35')
    ax.text(0.04, 1.22, 'movement', ha='left', va='top', fontsize=9,
            color='0.35')
    ax.text(0.04, -0.28, 'integrator', ha='left', va='bottom', fontsize=9,
            color='0.35')


def draw_classification(ax, cond):
    _zone_shades(ax)
    # Per-condition nudges. Sensory S sits on the 0.8 line, next to I.
    offsets = {
        'no_sensory_prior': {
            'S': (0, 8, 'center', 'bottom'),
            'I': (0, 8, 'center', 'bottom'),
            'M': (-11, 2, 'right', 'center'),
        },
        'full_model': {
            'S': (0, 8, 'center', 'bottom'),
            'I': (0, 8, 'center', 'bottom'),
            'M': (-11, 2, 'right', 'center'),
        },
    }
    for row in cond['classes']:
        x = row['sigma_stim_s']
        y = row['monotonicity']
        ax.scatter(
            [x], [y], s=78, color=sc.POP_COLOR[row['pop']],
            edgecolor=sc.C_INK, linewidth=0.6, zorder=3,
        )
        if row['correct']:
            dx, dy, ha, va = offsets[cond['key']][row['pop']]
            ax.annotate(
                row['pop'], (x, y), textcoords='offset points',
                xytext=(dx, dy), ha=ha, va=va, fontsize=11,
                color=sc.C_INK, zorder=4,
            )
        else:
            # Empty integrator region, left of the cluster.
            pred = sc.PRED_NAME.get(row['pred'], row['pred'])
            ax.annotate(
                f"{row['pop']}, {pred}",
                xy=(x, y), xytext=(0.22, 0.32),
                textcoords='data', ha='center', va='bottom', fontsize=11,
                color=sc.C_INK, zorder=4,
                arrowprops=dict(arrowstyle='-', color='0.45', lw=0.6,
                                shrinkA=2, shrinkB=4),
            )
    ax.set_xlim(-0.02, 1.16)
    ax.set_ylim(-0.42, 1.42)
    ax.set_xticks([0, 0.4, 0.8])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['0', '1'])
    ax.set_xlabel(r'$\Sigma^{\mathrm{stim},s}$')
    ax.set_ylabel('monotonicity')
    ax.set_title(cond['title'], fontsize=14, color=sc.C_INK, pad=8)
    ax.text(
        0.98, 0.04, f"{cond['n_correct']}/3", transform=ax.transAxes,
        ha='right', va='bottom', fontsize=12, color=sc.C_INK,
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(length=3, width=0.8, color=sc.C_INK)


def draw_prior(ax, conditions):
    width = 0.36
    centers = [0.0, 2.15]
    series = (
        ('stim_x_choice', -width / 2, sc.C_SPLIT, 'stim × choice'),
        ('stim_only', width / 2, sc.C_STIMONLY, 'stim only'),
    )
    ymax = max(
        c['strata'][k]['curve_mean'] for c in conditions for k, _, _, _ in series
    )
    legend_bars = {}
    for cond, center in zip(conditions, centers):
        for key, shift, color, label in series:
            row = cond['strata'][key]
            x = center + shift
            bar = ax.bar(
                x, row['curve_mean'], width=width * 0.92, color=color,
                edgecolor=sc.C_INK, linewidth=0.5, zorder=2, label=label,
            )
            legend_bars.setdefault(label, bar)
            if row['significant']:
                ax.plot(
                    x, row['curve_mean'] + 0.06 * max(ymax, 1e-6),
                    marker='*', color=sc.C_INK, markersize=9, linestyle='None',
                    zorder=3,
                )
            else:
                ax.text(
                    x, row['curve_mean'] + 0.04 * max(ymax, 1e-6),
                    'n.s.', ha='center', va='bottom', fontsize=10, color=sc.C_INK,
                )
        ax.text(
            center, -0.20, cond['title'], transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=12, color=sc.C_INK,
        )
    ax.set_xlim(-0.7, 2.85)
    ax.set_ylim(0, ymax * 1.38 if ymax > 0 else 1.0)
    ax.set_xticks([])
    ax.set_ylabel('S prior distance')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(length=3, width=0.8, color=sc.C_INK)
    ax.legend(legend_bars.values(), list(legend_bars), frameon=False,
              fontsize=11, loc='upper right', borderaxespad=0.2)


def _letter(fig, x, y, letter):
    fig.text(x, y, letter, fontsize=16, fontweight='regular', color=sc.C_INK,
             ha='left', va='top')


def save_all(fig, path_no_ext, dpi=200):
    path_no_ext = Path(path_no_ext)
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(facecolor='white')
    fig.savefig(path_no_ext.with_suffix('.pdf'), **kw)
    fig.savefig(path_no_ext.with_suffix('.svg'), **kw)
    fig.savefig(path_no_ext.with_suffix('.png'), dpi=dpi, **kw)


def write_tables(conditions, out_dir):
    class_path = out_dir / 'classification.csv'
    with class_path.open('w', newline='') as handle:
        fields = ['condition', 'model', 'population', 'true', 'pred', 'correct',
                  'sigma_stim_s', 'sigma_stim_m', 'monotonicity']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cond in conditions:
            for row in cond['classes']:
                writer.writerow({
                    'condition': cond['key'],
                    'model': cond['model'],
                    'population': row['pop'],
                    'true': row['true'],
                    'pred': row['pred'],
                    'correct': row['correct'],
                    'sigma_stim_s': f"{row['sigma_stim_s']:.6f}",
                    'sigma_stim_m': f"{row['sigma_stim_m']:.6f}",
                    'monotonicity': row['monotonicity'],
                })
    prior_path = out_dir / 'prior_by_stratum.csv'
    with prior_path.open('w', newline='') as handle:
        fields = ['condition', 'model', 'population', 'stratum', 'curve_mean',
                  'p_mean', 'significant']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cond in conditions:
            for stratum, row in cond['strata'].items():
                writer.writerow({
                    'condition': cond['key'],
                    'model': cond['model'],
                    'population': 'S',
                    'stratum': stratum,
                    'curve_mean': f"{row['curve_mean']:.6g}",
                    'p_mean': row['p_mean'],
                    'significant': row['significant'],
                })
            for pop, row in cond['im_stim_only'].items():
                writer.writerow({
                    'condition': cond['key'],
                    'model': cond['model'],
                    'population': pop,
                    'stratum': 'stim_only',
                    'curve_mean': f"{row['curve_mean']:.6g}",
                    'p_mean': row['p_mean'],
                    'significant': row['significant'],
                })
    summary = {
        'figure': 'si_recovery',
        'seed': 123,
        'n_sessions': 40,
        'nrand': 100,
        'null': 'contrast-matched label shuffle',
        'alpha': sc.ALPHA,
        'sigma_stim_threshold': sc.SIGMA_STIM,
        'conditions': [
            {
                'key': c['key'],
                'model': c['model'],
                'ground_truth': c['ground_truth'],
                'classification': f"{c['n_correct']}/3",
                'assignments': [
                    {'pop': r['pop'], 'pred': r['pred'],
                     'sigma_stim_s': r['sigma_stim_s'],
                     'monotonicity': r['monotonicity']}
                    for r in c['classes']
                ],
                'S_prior': c['strata'],
                'IM_stim_only': c['im_stim_only'],
            }
            for c in conditions
        ],
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


def build(conditions):
    fig = plt.figure(figsize=PAGE, dpi=150)
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.15, 1.0],
        left=0.09, right=0.98, top=0.90, bottom=0.14,
        hspace=0.48, wspace=0.32,
    )
    axes_a = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    ax_b = fig.add_subplot(gs[1, :])
    for ax, cond in zip(axes_a, conditions):
        draw_classification(ax, cond)
    draw_prior(ax_b, conditions)
    _letter(fig, 0.012, 0.985, 'a')
    _letter(fig, 0.012, 0.46, 'b')
    return fig


def build_panel(kind, conditions):
    if kind == 'a':
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), dpi=150)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.16, wspace=0.34)
        for ax, cond in zip(axes, conditions):
            draw_classification(ax, cond)
        return fig
    fig, ax = plt.subplots(figsize=(7.2, 2.7), dpi=150)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.24)
    draw_prior(ax, conditions)
    return fig


def main():
    missing = []
    for spec in sc.CONDITIONS:
        for key in ('class_csv', 'split_s_csv'):
            if not Path(spec[key]).is_file():
                missing.append(str(spec[key]))
        try:
            unsplit = _unsplit_csv(spec)
        except SystemExit:
            missing.append(f"unsplit summary for {spec['key']}")
        else:
            if not unsplit.is_file():
                missing.append(str(unsplit))
    if missing:
        raise SystemExit('missing recovery outputs:\n' + '\n'.join(missing))
    conditions = [load_condition(spec) for spec in sc.CONDITIONS]
    out = sc.FIG_DIR
    panels = out / 'panels'
    panels.mkdir(parents=True, exist_ok=True)
    fig = build(conditions)
    save_all(fig, out / 'si_recovery')
    plt.close(fig)
    fig_a = build_panel('a', conditions)
    save_all(fig_a, panels / 'panel_a')
    plt.close(fig_a)
    fig_b = build_panel('b', conditions)
    save_all(fig_b, panels / 'panel_b')
    plt.close(fig_b)
    write_tables(conditions, out)
    print(f'wrote {out / "si_recovery.pdf"}')


if __name__ == '__main__':
    main()
