"""Define Figure 4.5 cell types (S / C / M) on the new rastermap stack
``~/Downloads/concat_cvTrue.npy`` (54,569 cells, odd/even CV split, its own
``isort`` + ``rm_labels``).

The original types were index ranges along the old ``cross_val_test.npy``
isort (``dmn_avgbystim.get_uuids_raster_types``):
  stim_res [6000, 6700), integrator [4500, 6000), move_init [750, 4500),
  movement [43720, 44880)
plus the hard-coded 67-cell ``vis_subset0`` used for the S panel. That file
is gone, but the stacked 2025 outputs keep each type's ``uuids``, so we map
old types onto new rm clusters by uuid overlap.

Subcommands
  overlap  per-cluster counts of old-type cells -> CSV, enrichment heatmap,
           train/test cluster rasters (for picking by eye), suggested picks.
  save     write uuid lists for picked clusters:
           --pick integrator=22-27 move_init=... movement=... stim_res=28

Outputs: <dmn>/res/fig45/cvTrue/
"""
import argparse
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

from fig45_common import CLUSTERS_TABLE, DMN_RES, NEW_STACK, OUT_ROOT, savefig_svg_png

OLD_TYPE_FILES = {
    'stim_res': 'concat_block_pairs_act_normFalse_shuffleTrue_stim_res.npy',
    'integrator': 'concat_block_pairs_act_normFalse_shuffleTrue_integrator.npy',
    'move_init': 'concat_block_pairs_act_normFalse_shuffleTrue_move_init.npy',
    'movement': 'concat_block_pairs_act_normFalse_shuffleTrue_movement.npy',
    'vis_subset0': 'concat_block_pairs_act_vis_normFalse_shuffleTrue_vis_subset0.npy',
}
TYPE_COLORS = {'stim_res': '#57C1EB', 'vis_subset0': 'navy', 'integrator': 'goldenrod',
               'move_init': 'tomato', 'movement': 'darkred'}
OUT = OUT_ROOT / 'cvTrue'


def load_new(keys=('uuids', 'pid', 'ids', 'isort', 'rm_labels', 'len',
                   'concat_z', 'concat_z_train')):
    raw = np.load(NEW_STACK, allow_pickle=True).flat[0]
    return {k: raw[k] for k in keys}


def load_old_types():
    return {t: set(np.load(DMN_RES / f, allow_pickle=True).flat[0]['uuids'])
            for t, f in OLD_TYPE_FILES.items()}


def parse_ranges(s):
    out = []
    for part in s.split(','):
        if '-' in part:
            a, b = map(int, part.split('-'))
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return out


def cluster_table(new, old):
    labels = new['rm_labels']
    uu = new['uuids']
    clusters = np.unique(labels)
    rows = []
    for c in clusters:
        m = labels == c
        cu = set(uu[m])
        row = {'cluster': int(c), 'n': int(m.sum())}
        for t, s in old.items():
            row[f'n_{t}'] = len(cu & s)
            row[f'frac_{t}'] = row[f'n_{t}'] / row['n']
        rows.append(row)
    return rows


def suggest(rows, old, new_uuids, min_frac=0.15, min_cover=0.03):
    """Clusters where an old type is >= min_frac of the cluster AND the
    cluster holds >= min_cover of that type's surviving cells."""
    present = {t: len(s & new_uuids) for t, s in old.items()}
    sug = {}
    for t in old:
        picks = [r['cluster'] for r in rows
                 if r[f'frac_{t}'] >= min_frac and r[f'n_{t}'] / max(present[t], 1) >= min_cover]
        cover = sum(r[f'n_{t}'] for r in rows if r['cluster'] in picks) / max(present[t], 1)
        n = sum(r['n'] for r in rows if r['cluster'] in picks)
        sug[t] = {'clusters': picks, 'n_cells': n, 'old_cells_in_stack': present[t],
                  'coverage_of_old': cover}
    return sug


def plot_enrichment(rows, out_dir):
    types = list(OLD_TYPE_FILES)
    M = np.array([[r[f'frac_{t}'] for t in types] for r in rows])
    fig, ax = plt.subplots(figsize=(14, 3))
    im = ax.imshow(M.T, aspect='auto', cmap='magma_r', vmin=0, vmax=max(0.5, M.max()))
    ax.set_yticks(range(len(types)), types)
    ax.set_xticks(range(len(rows)), [r['cluster'] for r in rows], fontsize=5, rotation=90)
    ax.set_xlabel('new rm cluster (isort order)')
    fig.colorbar(im, ax=ax, label='fraction of cluster cells in old type')
    savefig_svg_png(fig, out_dir / 'old_type_fraction_per_cluster')
    plt.close(fig)


def plot_cluster_rasters(new, rows, out_dir, ranges):
    """Train / test rasters (rastermap order) with each cluster's tick
    coloured by its dominant old type (if any type is >= 15% of the cluster)."""
    isort = new['isort']
    ls = new['rm_labels'][isort]
    edges = np.concatenate([[0], np.flatnonzero(ls[1:] != ls[:-1]) + 1, [len(ls)]])
    lab_of_seg = ls[edges[:-1]]
    seg_edges = np.cumsum(list(new['len'].values()))
    rowd = {r['cluster']: r for r in rows}
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=1.5)
    for lo, hi in ranges:
        r0, r1 = edges[lo], edges[hi]
        fig, axs = plt.subplots(2, 1, figsize=(24, 14), sharex=True)
        for ax, key in zip(axs, ['concat_z_train', 'concat_z']):
            img = new[key][isort[r0:r1]]
            ax.imshow(img, cmap='gray_r', norm=norm, aspect='auto',
                      extent=(0, img.shape[1], r1, r0), interpolation='nearest')
            for b in seg_edges[:-1]:
                ax.axvline(b, color='r', lw=0.6)
            for e in edges[lo + 1:hi]:
                ax.axhline(e, color='tab:blue', lw=0.4)
            mids = (edges[lo:hi] + edges[lo + 1:hi + 1]) / 2
            labs = lab_of_seg[lo:hi]
            ax.set_yticks(mids, [str(c) for c in labs], fontsize=9)
            for t, c in zip(ax.get_yticklabels(), labs):
                r = rowd[int(c)]
                best = max(OLD_TYPE_FILES, key=lambda k: r[f'frac_{k}'])
                if r[f'frac_{best}'] >= 0.15:
                    t.set_color(TYPE_COLORS[best])
                    t.set_fontweight('bold')
            ax.set_ylabel(f'{key} (rm cluster)')
        axs[1].set_xticks(seg_edges - np.array(list(new['len'].values())) / 2,
                          list(new['len']), rotation=60, ha='right', fontsize=8)
        fig.suptitle(f'clusters {lo}-{hi - 1}; tick colour = dominant old type '
                     '(cyan stim_res, navy vis_subset0, gold integrator, '
                     'tomato move_init, darkred movement)')
        fig.tight_layout()
        fig.savefig(out_dir / 'cluster_id' / f'rm_clusters_{lo}_{hi - 1}.png', dpi=90,
                    facecolor='white')
        plt.close(fig)


def plot_cluster_means(new, rows, out_dir, clusters, name):
    """One row per cluster: mean train (grey) and test (black) concat_z,
    trial-type boundaries in red; label coloured by dominant old type."""
    seg_len = np.array(list(new['len'].values()))
    seg_edges = np.cumsum(seg_len)
    rowd = {r['cluster']: r for r in rows}
    fig, axs = plt.subplots(len(clusters), 1, figsize=(24, 1.1 * len(clusters) + 1.5),
                            sharex=True, squeeze=False)
    for ax, c in zip(axs[:, 0], clusters):
        m = new['rm_labels'] == c
        ax.plot(new['concat_z_train'][m].mean(0), color='0.6', lw=0.8)
        ax.plot(new['concat_z'][m].mean(0), color='k', lw=0.8)
        for b in seg_edges[:-1]:
            ax.axvline(b, color='r', lw=0.5)
        r = rowd[int(c)]
        best = max(OLD_TYPE_FILES, key=lambda k: r[f'frac_{k}'])
        col = TYPE_COLORS[best] if r[f'frac_{best}'] >= 0.15 else 'k'
        ax.set_ylabel(f'{c}\nn={r["n"]}', rotation=0, ha='right', va='center',
                      fontsize=9, color=col)
        ax.set_yticks([])
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axs[-1, 0].set_xticks(seg_edges - seg_len / 2, list(new['len']), rotation=60,
                          ha='right', fontsize=8)
    axs[-1, 0].set_xlim(0, seg_edges[-1])
    fig.suptitle(f'{name}: cluster-mean concat_z, train (grey) vs test (black); '
                 'label colour = dominant old type (tomato move_init, darkred movement, '
                 'gold integrator, cyan stim_res)')
    fig.tight_layout()
    fig.savefig(out_dir / 'cluster_id' / f'{name}_means.png', dpi=90, facecolor='white')
    plt.close(fig)


def cmd_rasters(args):
    (OUT / 'cluster_id').mkdir(parents=True, exist_ok=True)
    new = load_new()
    rows = cluster_table(new, load_old_types())
    ranges = [tuple(map(int, s.split('-'))) for s in args.ranges]
    plot_cluster_rasters(new, rows, OUT, [(a, b + 1) for a, b in ranges])
    clusters = sorted({c for a, b in ranges for c in range(a, b + 1)})
    plot_cluster_means(new, rows, OUT, clusters, args.name)
    print('wrote', OUT / 'cluster_id')


REVIEW_COLORS = {'integrator': 'goldenrod', 'move_init': 'tomato', 'move': 'darkred',
                 'move_fback': 'mediumpurple'}


def cmd_review(args):
    """Train | test rastermap of the whole new stack (isort order, all trial
    types), side band + cluster-id ticks coloured by picked type."""
    picks = json.loads((OUT / 'cell_types_picks.json').read_text())
    new = load_new(keys=('isort', 'rm_labels', 'len', 'concat_z', 'concat_z_train'))
    isort = new['isort']
    ls = new['rm_labels'][isort]
    edges = np.concatenate([[0], np.flatnonzero(ls[1:] != ls[:-1]) + 1, [len(ls)]])
    seg_len = np.array(list(new['len'].values()))
    seg_edges = np.cumsum(seg_len)
    type_of = {c: t for t in REVIEW_COLORS if t in picks for c in picks[t]}
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=1.5)
    fig, axs = plt.subplots(1, 2, figsize=(26, 18), sharey=True)
    for ax, key in zip(axs, ['concat_z_train', 'concat_z']):
        img = new[key][isort]
        ax.imshow(img, cmap='gray_r', norm=norm, aspect='auto', interpolation='antialiased',
                  extent=(0, img.shape[1], len(ls), 0))
        for b in seg_edges[:-1]:
            ax.axvline(b, color='r', lw=0.4)
        for e in edges[1:-1]:
            ax.axhline(e, color='0.7', lw=0.2)
        for lo, hi, c in zip(edges[:-1], edges[1:], ls[edges[:-1]]):
            t = type_of.get(int(c))
            if t:
                ax.add_patch(plt.Rectangle((img.shape[1] + 5, lo), 25, hi - lo,
                                           color=REVIEW_COLORS[t], clip_on=False))
        ax.set_xlim(0, img.shape[1] + 32)
        ax.set_xticks(seg_edges - seg_len / 2, list(new['len']), rotation=60, ha='right',
                      fontsize=8)
        ax.set_title(f'{key} ({"rastermap fit" if key.endswith("train") else "held-out"})')
    mids = (edges[:-1] + edges[1:]) / 2
    labs = ls[edges[:-1]]
    axs[0].set_yticks(mids, [str(c) for c in labs], fontsize=6)
    for tl, c in zip(axs[0].get_yticklabels(), labs):
        t = type_of.get(int(c))
        if t:
            tl.set_color(REVIEW_COLORS[t])
            tl.set_fontweight('bold')
    axs[0].set_ylabel('rm cluster (isort order)')
    counts = {t: int(np.isin(new['rm_labels'], picks[t]).sum()) for t in REVIEW_COLORS
              if t in picks}
    fig.suptitle('concat_cvTrue picks: ' + ', '.join(
        f'{t} ({REVIEW_COLORS[t]}, n={n})' for t, n in counts.items()), fontsize=12)
    fig.tight_layout()
    (OUT / 'plots').mkdir(parents=True, exist_ok=True)
    savefig_svg_png(fig, OUT / 'plots' / 'review_C_M_picks', dpi=150)
    plt.close(fig)
    print('wrote', OUT / 'plots' / 'review_C_M_picks.png')


def cmd_overlap(args):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'cluster_id').mkdir(exist_ok=True)
    new = load_new()
    lab_sorted = new['rm_labels'][new['isort']]
    assert np.all(np.diff(lab_sorted) >= 0), 'rm_labels not monotonic along isort'
    old = load_old_types()
    rows = cluster_table(new, old)
    with open(OUT / 'old_type_overlap_per_cluster.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot_enrichment(rows, OUT)
    sug = suggest(rows, old, set(new['uuids']))
    (OUT / 'suggested_clusters.json').write_text(json.dumps(sug, indent=2))
    print(json.dumps(sug, indent=2))
    if not args.no_rasters:
        n = int(new['rm_labels'].max()) + 1
        step = 20
        plot_cluster_rasters(new, rows, OUT,
                             [(a, min(a + step, n)) for a in range(0, n, step)])
    print('wrote', OUT)


def cluster_ids_and_insertions(uuids, pids):
    """cluster_id per uuid (BWM clusters.pqt) and pid -> (eid, probe) (bwm_query freeze).

    Stored in cell_types.npy so the ORCD insertion-cache path needs neither
    clusters.pqt nor an Alyx connection.
    """
    import pandas as pd
    from brainwidemap import bwm_query
    tab = pd.read_parquet(CLUSTERS_TABLE, columns=['uuids', 'cluster_id', 'pid'])
    cid = dict(zip(tab['uuids'], tab['cluster_id']))
    missing = [u for u in uuids if u not in cid]
    if missing:
        raise KeyError(f'{len(missing)} uuids not in {CLUSTERS_TABLE}, e.g. {missing[:3]}')
    q = bwm_query().set_index('pid')
    ins = {p: (q.loc[p, 'eid'], q.loc[p, 'probe_name']) for p in sorted(set(pids))}
    return np.array([cid[u] for u in uuids], dtype=np.int64), ins


def cmd_save(args):
    new = load_new(keys=('uuids', 'pid', 'ids', 'rm_labels'))
    picks = {}
    for item in args.pick:
        name, rng = item.split('=')
        picks[name] = parse_ranges(rng)
    out = {'source': str(NEW_STACK), 'picks': picks, 'types': {}}
    for item in args.union or []:
        name, parts = item.split('=')
        picks[name] = sorted(c for p in parts.split('+') for c in picks[p])
    masks = {name: np.isin(new['rm_labels'], cl) for name, cl in picks.items()}
    any_m = np.logical_or.reduce(list(masks.values()))
    cids = np.full(len(any_m), -1, dtype=np.int64)
    cids[any_m], out['insertions'] = cluster_ids_and_insertions(
        new['uuids'][any_m], new['pid'][any_m])
    for name, m in masks.items():
        out['types'][name] = {'uuids': new['uuids'][m], 'pid': new['pid'][m],
                              'ids': new['ids'][m], 'cluster_id': cids[m]}
        print(f'{name}: clusters {picks.get(name, "union")} -> {m.sum()} cells, '
              f'{len(set(new["pid"][m]))} insertions')
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / 'cell_types.npy', out, allow_pickle=True)
    (OUT / 'cell_types_picks.json').write_text(json.dumps(picks, indent=2))
    print('wrote', OUT / 'cell_types.npy')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('overlap')
    p.add_argument('--no-rasters', action='store_true')
    p.set_defaults(func=cmd_overlap)
    p = sub.add_parser('rasters', help='zoomed train/test rasters + cluster means')
    p.add_argument('--ranges', nargs='+', required=True, help='e.g. 15-22 29-33')
    p.add_argument('--name', default='zoom')
    p.set_defaults(func=cmd_rasters)
    p = sub.add_parser('save')
    p.add_argument('--pick', nargs='+', required=True,
                   help='TYPE=clusters, e.g. integrator=22-27 stim_res=28')
    p.add_argument('--union', nargs='+',
                   help='NAME=a+b+c over picked types, e.g. move_all=move_init+move+move_fback')
    p.set_defaults(func=cmd_save)
    p = sub.add_parser('review', help='full rastermap with picked C / M clusters marked')
    p.set_defaults(func=cmd_review)
    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
