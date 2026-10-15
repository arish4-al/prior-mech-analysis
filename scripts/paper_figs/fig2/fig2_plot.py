#!/usr/bin/env python
"""Figure 2 composite and panels in the published layout.

  a  early stim sensitivity  d^{stim,se'}  stim_duringstim1_act (act-block strata, 0-80 ms)
  b  stim sensitivity        d^{stim,s/m}  stim_duringstim_act / stim_duringchoice_act
                                           (choice x act-block strata)
  c  choice sensitivity      d^{choice,s/m} choice_duringstim_act / choice_duringchoice_act
                                           (stim x act-block strata)
  d  schematic, cropped from the original PDF
  e  Sigma^{stim,s} histogram, Sigma^{stim,m} vs monotonicity (analysis_figs.ipynb cell 6)
  f  s / m region types over Beryl regions (plot_sc_table regtype columns)

Reads <fig2>/<version>/{sc_table.csv, pvals.csv, staging/} from fig2_data.py.
Writes <fig2>/<version>/fig2_<version>.{svg,png} and panels/.

  conda activate iblenv   # outside the sandbox
  python scripts/paper_figs/fig2/fig2_plot.py --version openalyx
  python scripts/paper_figs/fig2/fig2_plot.py --version min5 --examples b=VeCB,ORBl
"""
import argparse
import json
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from matplotlib.patches import Rectangle

import fig2_common as fc

FIG_W, FIG_H = 858.748 / 72, 554.773 / 72      # original page size [in]
EXAMPLES = {'a': ('VISp', 'MOs'), 'b': ('VeCB', 'ORBl'), 'c': ('GRN', 'RSPd')}
PANELS = {   # panel -> [(timeframe, aligned to)]
    'a': [('stim_duringstim1_act', 'stim')],
    'b': [('stim_duringstim_act', 'stim'), ('stim_duringchoice_act', 'move')],
    'c': [('choice_duringstim_act', 'stim'), ('choice_duringchoice_act', 'move')],
}
VAR = {'a': ('stim', "se^\\prime"), 'b': ('stim', None), 'c': ('choice', None)}
TITLES = {'a': 'early stim sensitivity', 'b': 'stim sensitivity', 'c': 'choice sensitivity'}
N_SHOW = 40

# axes boxes measured on the original page: (x0, x1) for data and histogram
ROW_Y = {0: (0.060, 0.213), 1: (0.250, 0.405)}       # from the top
COLS = {('a', 0): ((0.078, 0.203), (0.206, 0.228)),
        ('b', 0): ((0.277, 0.407), (0.411, 0.433)),
        ('b', 1): ((0.450, 0.581), (0.584, 0.606)),
        ('c', 0): ((0.658, 0.790), (0.794, 0.816)),
        ('c', 1): ((0.841, 0.971), (0.974, 0.996))}
E_BOX = {'hist': (0.5895, 0.6725), 'strip': (0.7103, 0.8755), 'y': (0.509, 0.740)}
D_CROP = (0.129, 0.480, 0.470, 0.800)                 # x0, x1, ytop0, ytop1
F_BOX = {'x': (0.008, 0.945), 'y': (0.895, 0.970), 'split': (0.913, 0.932)}

plt.rcParams.update({'font.family': 'Helvetica', 'svg.fonttype': 'none',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.labelsize': 9})


def rect(x, y):
    return [x[0], 1 - y[1], x[1] - x[0], y[1] - y[0]]


class Data:
    def __init__(self, version):
        self.version = version
        self.vdir = fc.version_dir(version)
        self.sc = pd.read_csv(self.vdir / 'sc_table.csv')
        self.pv = pd.read_csv(self.vdir / 'pvals.csv', index_col='region')
        self.af = fc.load_af(fc.staging_dir(version))
        self._regde = {}

    def curves(self, tf, reg):
        if tf not in self._regde:
            _, regde = fc.combined_stems(self.af, tf)
            self._regde[tf] = np.load(fc.staging_dir(self.version) / f'{regde}.npy',
                                      allow_pickle=True).item()
        r = self._regde[tf][reg]
        r = np.concatenate([np.asarray(r[0]).reshape(1, -1), np.asarray(r[1])], axis=0)
        r = r / len(self.af.run_align[tf])
        return self.af._time_axis_for_plot(tf, r.shape[1]), r


def nice_ylim(rs):
    lo = min(r[:N_SHOW + 1].min() for r in rs)
    hi = max(r[:N_SHOW + 1].max() for r in rs)
    step = 0.05 if hi - lo < 0.6 else 0.1
    lo, hi = np.floor(lo / step) * step, np.ceil(hi / step) * step
    if round((hi - lo) / step) % 2:
        hi += step
    return (lo, hi), [lo, (lo + hi) / 2, hi]


def ylabel(panel, align, reg):
    var, sup = VAR[panel]
    sup = sup or ('s' if align == 'stim' else 'm')
    return rf'$d^{{\mathrm{{{var}}},{sup}}}_{{\mathrm{{{reg}}}}}(t)$'


def draw_dist(ax, hax, t, r, p, align, ylim, yticks, first_col, bottom_row, label):
    ev = fc.C_STIM_ON if align == 'stim' else fc.C_MOVE_ON
    t0 = 0
    ax.axvline(t0, color=ev, lw=3.2, zorder=0)
    for i in range(1, min(N_SHOW + 1, len(r))):
        ax.plot(t, r[i], color='gray', alpha=0.3, lw=0.5)
    ax.plot(t, r[0], color='k', lw=1.4)
    ax.text(0.12, 0.97, f'p_val {p:.4f}', transform=ax.transAxes, fontsize=6.5,
            ha='left', va='top')
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{v:g}' for v in np.round(yticks, 2)])
    ax.set_ylabel(label, fontsize=11, labelpad=0)
    if not first_col:
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', left=False, labelleft=False)
    if align == 'stim':
        ax.set_xlim(t[0] - 0.02 * (t[-1] - t[0]), t[-1])
        xt = [0, 40, 80] if t[-1] <= 80 else [0, 50, 100, 150]
    else:
        ax.set_xlim(t[0], t[-1] + 0.02 * (t[-1] - t[0]))
        xt = [-150, -100, -50, 0]
    ax.set_xticks(xt)
    if not bottom_row:
        ax.tick_params(axis='x', labelbottom=False)
        ax.text(0 if align == 'stim' else 1, -0.04, 'stim on' if align == 'stim' else 'mvt on',
                transform=ax.transAxes, color=ev, fontsize=9,
                ha='left' if align == 'stim' else 'right', va='top')
    hax.hist(np.mean(r[1:], axis=1), bins=20, density=True, color='silver',
             orientation='horizontal')
    hax.axhline(np.mean(r[0]), color='k', lw=1.4)
    hax.set_ylim(ylim)
    hax.tick_params(axis='y', left=False, labelleft=False)
    hax.tick_params(axis='x', bottom=False, labelbottom=False)
    hax.spines['bottom'].set_visible(False)


def panel_axes_single(n_cols):
    fig = plt.figure(figsize=(2.3 * n_cols, 1.6))
    axs = []
    for j in range(n_cols):
        x0 = 0.12 / n_cols + j / n_cols
        ax = fig.add_axes([x0, 0.18, 0.62 / n_cols, 0.75])
        hax = fig.add_axes([x0 + 0.63 / n_cols, 0.18, 0.11 / n_cols, 0.75])
        axs.append((ax, hax))
    return fig, axs


def draw_example(data, panel, row, reg, axes_pairs):
    tfs = PANELS[panel]
    cur = [data.curves(tf, reg) for tf, _ in tfs]
    ylim, yticks = nice_ylim([r for _, r in cur])
    for j, ((tf, align), (t, r), (ax, hax)) in enumerate(zip(tfs, cur, axes_pairs)):
        draw_dist(ax, hax, t, r, float(data.pv.loc[reg, tf]), align, ylim, yticks,
                  first_col=j == 0, bottom_row=row == 1, label=ylabel(panel, align, reg))
    return {tf: float(data.pv.loc[reg, tf]) for tf, _ in tfs}


def draw_e(data, ax_h, ax_s):
    res = data.sc
    rs, rc = res['sc_duringstim_regtype'], res['sc_duringchoice_regtype']
    centers = np.linspace(0, 1, 21)
    w = centers[1] - centers[0]
    edges = np.concatenate(([centers[0] - w / 2], centers + w / 2))
    inv = 1 - res['sc_duringstim']
    ax_h.hist(inv[rs.isin([0.5, 1.0])], bins=edges, orientation='horizontal',
              color=fc.C_INTEG_HIST, alpha=0.7)
    ax_h.hist(inv[rs == 0], bins=edges, orientation='horizontal', color=fc.C_STIM, alpha=0.7)
    ax_h.hist(inv[rs == 0.1] + w / 4, bins=edges, orientation='horizontal',
              color=fc.C_STIM_EARLY, alpha=0.7)
    ax_h.axhline(0.8, color=fc.C_STIM, ls='--', lw=1)
    rng = np.random.default_rng(0)
    mono = res['monotonicity'].astype(float)
    sig_m = 1 - res['sc_duringchoice']
    for val, color in ((0.5, fc.C_INTEG), (1.0, fc.C_MOVE), (0.0, fc.C_STIM)):
        m = rc == val
        x = mono[m] + rng.uniform(-0.25, 0.25, m.sum())
        ax_s.scatter(x, sig_m[m], s=6, color=color, lw=0)
    ax_s.axvline(0.5, color=fc.C_MOVE_ON, ls='--', lw=1)
    for ax in (ax_h, ax_s):
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 0.5, 1])
    ax_h.set_yticklabels(['0', '0.5', '1'])
    ax_s.set_yticklabels([])
    ax_s.set_xlim(-0.5, 1.5)
    ax_s.set_xticks([0, 1])
    ax_h.set_xlabel('count', fontsize=10)
    ax_s.set_xlabel('monotonicity', fontsize=10)
    ax_h.set_ylabel(r'$\Sigma^{\mathrm{stim},s}$', fontsize=12)
    ax_s.set_ylabel(r'$\Sigma^{\mathrm{stim},m}$', fontsize=12)
    return {'n_hist': int(inv.notna().sum()),
            'sensory_gt_0.8': int((inv > 0.8).sum()),
            'n_strip': int(rc.notna().sum()),
            'move': int((rc == 1).sum()), 'integ_m': int((rc == 0.5).sum())}


def region_order(regions):
    order = [l.strip() for l in (fc.REPO / 'region_order.txt').read_text().splitlines()
             if l.strip()]
    return [r for r in order if r in set(regions)]


def draw_f(data, ax):
    af = data.af
    res = data.sc.set_index('region')
    regs = region_order(res.index)
    cmap = af.get_cmap_('sc_duringstim_regtype')
    rows = (('sc_duringstim_regtype', 2), ('sc_duringchoice_regtype', 1))
    for i, reg in enumerate(regs):
        for col, y in rows:
            v = res.loc[reg, col]
            face = '#f2f2f2' if pd.isna(v) else to_hex(cmap(float(v)))
            ax.add_patch(Rectangle((i, y), 1, 1, facecolor=face, edgecolor='white', lw=0.4))
        ax.add_patch(Rectangle((i, -0.15), 1, 1.15, edgecolor='white', lw=0.4,
                               facecolor=af.swanson_to_beryl_hex(reg, af.br)))
        ax.text(i + 0.5, 0.92, reg, rotation=90, ha='center', va='top', fontsize=2.8)
    ax.set_xlim(0, len(regs))
    ax.set_ylim(-0.15, 3)
    ax.axis('off')
    return len(regs)


def colored_title(fig, x, y, pieces, fontsize=13):
    """Left-to-right text pieces [(text, color)], centred on x (figure coords)."""
    r = fig.canvas.get_renderer()
    texts = [fig.text(0, y, s, color=c, fontsize=fontsize, va='center') for s, c in pieces]
    widths = [t.get_window_extent(r).width / fig.bbox.width for t in texts]
    x0 = x - sum(widths) / 2
    for t, w in zip(texts, widths):
        t.set_x(x0)
        x0 += w


def crop_original_d(out_png):
    if out_png.exists():
        return out_png
    full = out_png.with_name('figure2_600dpi.png')
    subprocess.run(['gs', '-q', '-dNOPAUSE', '-dBATCH', '-sDEVICE=png16m', '-r600',
                    f'-sOutputFile={full}', str(fc.ORIGINAL_PDF)], check=True)
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(full)
    W, H = im.size
    x0, x1, y0, y1 = D_CROP
    im.crop((int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H))).save(out_png)
    full.unlink()
    return out_png


def letter(fig, x, ytop, s):
    fig.text(x, 1 - ytop, s, fontsize=24, va='center', ha='left')


def composite(data, examples, out):
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor('white')
    used = {}
    for panel in ('a', 'b', 'c'):
        for row, reg in enumerate(examples[panel]):
            pairs = []
            for j in range(len(PANELS[panel])):
                dx, hx = COLS[(panel, j)]
                pairs.append((fig.add_axes(rect(dx, ROW_Y[row])),
                              fig.add_axes(rect(hx, ROW_Y[row]))))
            used[f'{panel}_{"sig" if row == 0 else "nonsig"}'] = {
                'region': reg, 'p': draw_example(data, panel, row, reg, pairs)}
    for x, s in ((0.140, TITLES['a']), (0.440, TITLES['b']), (0.815, TITLES['c'])):
        fig.text(x, 1 - 0.030, s, fontsize=12, ha='center', va='center')
    for x, s in ((0.140, 'time (ms)'), (0.440, 'time (ms)'), (0.815, 'time (ms)')):
        fig.text(x, 1 - 0.458, s, fontsize=10, ha='center', va='center')
    for yt, s in ((0.137, 'significant'), (0.328, 'non-significant')):
        fig.text(0.012, 1 - yt, s, fontsize=13, rotation=90, ha='center', va='center')
    for x, yt, s in ((0.003, 0.035, 'a'), (0.236, 0.035, 'b'), (0.613, 0.035, 'c'),
                     (0.105, 0.492, 'd'), (0.527, 0.492, 'e'), (0.003, 0.862, 'f')):
        letter(fig, x, yt, s)

    d_png = crop_original_d(fc.FIG_DIR / 'original' / 'panel_d_original.png')
    ax_d = fig.add_axes(rect(D_CROP[:2], D_CROP[2:]))
    ax_d.imshow(plt.imread(d_png), interpolation='lanczos')
    ax_d.axis('off')

    ax_h = fig.add_axes(rect(E_BOX['hist'], E_BOX['y']))
    ax_s = fig.add_axes(rect(E_BOX['strip'], E_BOX['y']))
    e_info = draw_e(data, ax_h, ax_s)
    fig.text(0.71, 1 - 0.494, 'function assignments', fontsize=12, ha='center', va='center')

    ax_f = fig.add_axes(rect(F_BOX['x'], F_BOX['y']))
    n_f = draw_f(data, ax_f)
    for yt, s in ((0.904, 's'), (0.922, 'm')):
        fig.text(0.955, 1 - yt, s, fontsize=15, style='italic', family='serif',
                 ha='left', va='center')
    colored_title(fig, 0.5, 1 - 0.862, [('stim', fc.C_STIM), (', ', 'k'), ('integ', fc.C_INTEG),
                                        (', ', 'k'), ('move', fc.C_MOVE),
                                        (': Beryl regions', 'k')])
    fc.savefig_svg_png(fig, out / f'fig2_{data.version}', dpi=200)
    plt.close(fig)
    return used, e_info, n_f


def single_panels(data, examples, pdir):
    for panel in ('a', 'b', 'c'):
        for row, reg in enumerate(examples[panel]):
            fig, pairs = panel_axes_single(len(PANELS[panel]))
            draw_example(data, panel, row, reg, pairs)
            tag = 'sig' if row == 0 else 'nonsig'
            fc.savefig_svg_png(fig, pdir / f'panel_{panel}_{tag}_{reg}', tight=True)
            plt.close(fig)
    fig = plt.figure(figsize=(3.6, 2.4))
    draw_e(data, fig.add_axes([0.16, 0.2, 0.28, 0.7]), fig.add_axes([0.6, 0.2, 0.36, 0.7]))
    fc.savefig_svg_png(fig, pdir / 'panel_e', tight=True)
    plt.close(fig)
    fig = plt.figure(figsize=(FIG_W, 0.8))
    draw_f(data, fig.add_axes([0.01, 0.02, 0.98, 0.96]))
    fc.savefig_svg_png(fig, pdir / 'panel_f', dpi=300, tight=True)
    plt.close(fig)


def parse_examples(items):
    ex = dict(EXAMPLES)
    for it in items or []:
        k, v = it.split('=')
        ex[k] = tuple(v.split(','))
    return ex


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--version', choices=list(fc.VERSIONS), default='openalyx')
    ap.add_argument('--examples', nargs='*', help='panel=sig,nonsig, e.g. b=VeCB,ORBl')
    ap.add_argument('--no-panels', action='store_true')
    args = ap.parse_args()
    examples = parse_examples(args.examples)
    data = Data(args.version)
    out = data.vdir
    used, e_info, n_f = composite(data, examples, out)
    if not args.no_panels:
        single_panels(data, examples, out / 'panels')
    summary = {'version': args.version, 'desc': fc.VERSIONS[args.version]['desc'],
               'examples': used, 'e': e_info, 'f_n_regions': n_f,
               'regtype_counts': {c: data.sc[c].value_counts().sort_index().to_dict()
                                  for c in ('sc_duringstim_regtype', 'sc_duringchoice_regtype')}}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
