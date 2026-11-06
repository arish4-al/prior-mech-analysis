"""Figure 4.5 panel a: whole-population rastermap (top) and a zoom on the
M + C cell-type rows (bottom).

Published panel: old 26-type concat (``cross_val_test.npy`` sorted by the
``cross_val_train`` isort), columns quiescence -> R_move, with the 8 x 2
stim x choice x block types. The new stack pools mistakes
(``mistake_s`` / ``mistake_m``) and has only the 4 correct stim x block types per
alignment, so columns differ; we show quiescence -> choiceR.

Rows: held-out half (``concat_z``) in ``isort`` order (fit on the train half).
Zoom rows: rm clusters picked for move_all (M) / integrator (C) (from
``fig45_define_cell_types.py save``), in isort order.

Also writes ``panel_a_<half>.npz`` (row-binned rasters, column segments,
S / C / M row spans) for the composite in ``fig45_sensitivity.py fig45``.

``--rm-select <stageRM>/select_rm`` takes S / C / M from the stage RM cluster
selection (``clusters.csv``, ``--move-rule``) instead of the picks and writes
the raster and cache into ``<select_rm>/fig45``.

Usage (iblenv, outside sandbox):
  python scripts/fig45_panel_a.py [--half test|train] [--rm-select DIR]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

from fig45_common import NEW_STACK, OUT_ROOT, TYPES_DIR, savefig_svg_png, sum_for_key

OUT = OUT_ROOT / 'cvTrue'
FIRST, LAST = 'quiescence', 'choiceR'
ZOOM_TYPES = {'move_all': 'tomato', 'integrator': 'gold'}
BAND_TYPES = {'move_init': 'tomato', 'move': 'darkred', 'move_fback': 'mediumpurple',
              'integrator': 'gold', 'stim_res': '#57C1EB'}
SHORT = {'quiescence': 'quies', 'block_change_s': 'change_b,s', 'stimLbLcL': 'LLL,s',
         'stimLbRcL': 'LLR,s', 'stimRbRcR': 'RRR,s', 'stimRbLcR': 'RRL,s',
         'mistake_s': 'mistake,s', 'motor_init': 'm', 'block_change_m': 'change_b,m',
         'sLbLchoiceL': 'LLL,m', 'sLbRchoiceL': 'LLR,m', 'sRbRchoiceR': 'RRR,m',
         'sRbLchoiceR': 'RRL,m', 'mistake_m': 'mistake,m', 'choiceL': 'L move',
         'choiceR': 'R move'}
# composite colour strips: S pool, C, M (one colour each)
STRIP_TYPES = {'S': 'stimint', 'C': 'integrator', 'M': 'move_all'}
MAX_ROWS = 2000


def cache_path(half):
    return OUT / 'plots' / f'panel_a_{half}.npz'


def rm_picks(csv_f, move_rule):
    """Stage RM S / C / M clusters as picks (``STRIP_TYPES`` names)."""
    import csv
    with open(csv_f) as f:
        rows = list(csv.DictReader(f))
    col = {'stimint': 'S', 'integrator': f'C_{move_rule}', 'move_all': f'M_{move_rule}'}
    return {t: [int(r['cluster']) for r in rows if r[k] == 'True'] for t, k in col.items()}


def row_bin(X, max_rows=MAX_ROWS):
    """Mean over consecutive row groups so at most ``max_rows`` remain."""
    k = int(np.ceil(len(X) / max_rows))
    n = len(X) // k * k
    out = X[:n].reshape(-1, k, X.shape[1]).mean(1)
    return np.vstack([out, X[n:].mean(0, keepdims=True)]) if n < len(X) else out


def runs(mask):
    """(start, stop) row spans where ``mask`` is True."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.c_[np.flatnonzero(d == 1), np.flatnonzero(d == -1)]


def column_slice(lens):
    keys = list(lens)
    segs = keys[keys.index(FIRST):keys.index(LAST) + 1]
    return sum_for_key(lens, FIRST), sum_for_key(lens, LAST, after=True), {k: lens[k] for k in segs}


def draw(ax, img, seg_lens, norm, labels=True):
    ax.imshow(img, cmap='gray_r', norm=norm, aspect='auto', interpolation='antialiased')
    h = 0
    trans = ax.get_xaxis_transform()
    for k, n in seg_lens.items():
        ax.axvline(h + n - 0.5, ls='--', lw=0.4, color='grey')
        if labels:
            ax.text(h + n / 2, 1.01, SHORT.get(k, k), rotation=90, ha='center', va='bottom',
                    fontsize=6, transform=trans)
        h += n
    ax.set_xticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--half', choices=['test', 'train'], default='test')
    ap.add_argument('--rm-select', type=Path, default=None)
    ap.add_argument('--move-rule', default='late_mean')
    args = ap.parse_args()
    raw = np.load(NEW_STACK, allow_pickle=True).flat[0]
    key = 'concat_z' if args.half == 'test' else 'concat_z_train'
    c0, c1, seg_lens = column_slice(raw['len'])
    isort = raw['isort']
    lab_sorted = raw['rm_labels'][isort]
    X = raw[key][isort, c0:c1]
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=1.5)

    if args.rm_select:
        picks = rm_picks(args.rm_select / 'clusters.csv', args.move_rule)
        odir, cache_f = args.rm_select / 'fig45', args.rm_select / 'fig45' / f'panel_a_{args.half}.npz'
    else:
        picks_f = TYPES_DIR / 'cell_types_picks.json'
        picks = json.loads(picks_f.read_text()) if picks_f.exists() else {}
        odir, cache_f = OUT / 'plots', cache_path(args.half)

    fig = plt.figure(figsize=(8, 9))
    ax_top = fig.add_axes([0.08, 0.45, 0.8, 0.48])
    draw(ax_top, X, seg_lens, norm)
    ax_top.set_ylabel('cell #')
    ax_top.set_yticks([0, 50000], ['1', '50K'])
    for t, col in BAND_TYPES.items():
        if t in picks:
            rows = np.flatnonzero(np.isin(lab_sorted, picks[t]))
            ax_top.plot([X.shape[1] + 3] * len(rows), rows, '|', color=col, ms=1, clip_on=False)
    ax_top.set_xlim(-0.5, X.shape[1] + 6)

    zoom = [c for t in ZOOM_TYPES if t in picks for c in picks[t]]
    if zoom:
        rows = np.flatnonzero(np.isin(lab_sorted, zoom))
        ax_z = fig.add_axes([0.08, 0.05, 0.8, 0.36])
        draw(ax_z, X[rows], seg_lens, norm, labels=False)
        ax_z.set_ylabel('cell # (M + C clusters)')
        zl = lab_sorted[rows]
        for t, col in ZOOM_TYPES.items():
            r = np.flatnonzero(np.isin(zl, picks.get(t, [])))
            if len(r):
                ax_z.plot([X.shape[1] + 3] * len(r), r, '|', color=col, ms=1, clip_on=False)
        ax_z.set_xlim(-0.5, X.shape[1] + 6)
    savefig_svg_png(fig, odir / f'panel_a_raster_{args.half}', dpi=200)
    plt.close(fig)

    zl = lab_sorted[rows] if zoom else lab_sorted[:0]
    cache = dict(top=row_bin(X).astype(np.float32), n_top=len(X),
                 zoom=row_bin(X[rows]).astype(np.float32) if zoom else np.zeros((0, X.shape[1])),
                 n_zoom=len(zl), seg_keys=np.array(list(seg_lens)),
                 seg_lens=np.array(list(seg_lens.values())),
                 seg_short=np.array([SHORT.get(k, k) for k in seg_lens]))
    for L, t in STRIP_TYPES.items():
        cache[f'top_{L}'] = runs(np.isin(lab_sorted, picks.get(t, [])))
        cache[f'zoom_{L}'] = runs(np.isin(zl, picks.get(t, [])))
    cache_f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_f, **cache)
    print('wrote', odir)


if __name__ == '__main__':
    main()
