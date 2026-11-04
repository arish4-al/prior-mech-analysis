"""Figure 4.5 follow-up on concat_cvTrue: stimulus / choice / act-prior
sensitivity of rastermap cell types, each against a label shuffle inside the
strata of the other factors.

Labels: stim side (contrastLeft/Right not NaN), choice (+-1), act-kernel block
(alpha 0.2 over the full choice sequence, binarised 0.8 / 0.2, as in
fig45_block_peths). Trials: that kernel, then the BWM bad-trial mask and one
saturation key for the alignment, then a drop of true probabilityLeft == 0.5.

Analyses (window = (pre, post) s around the align event; same overlapping
12.5 ms / 2 ms bins as the regional pipeline):

  S_stim       stimOn [0, 80]   stim L vs R     strata: act block
  S_prior      stimOn [0, 80]   block L vs R    strata: stim side
  choice_stim  stimOn [0, 150]  choice L vs R   strata: stim x act block
  choice_move  move [-150, 0]   choice L vs R   strata: stim x act block
  prior_stim   stimOn [0, 150]  block L vs R    strata: stim x choice
  prior_move   move [-150, 0]   block L vs R    strata: stim x choice

Each stratum is one pair; its null permutes the contrast label between the
pair's two trial sets (sizes kept), NRAND times. A pair enters an insertion
only if both sides have >= MIN_TRIALS trials (regional-pipeline gate).

Population statistic per pair: sum over active cells of (mean_a - mean_b)^2
per bin, normalised sqrt(. / T_BIN / n_pair) and averaged over pairs as in
the Fig 4.5 plots; p from fig45_plot_distances.distance_stats.
Per-cell statistic (selection): mean over bins of diff^2, averaged over the
insertion's valid pairs; p = (1 + #{null >= obs}) / (1 + nrand).

Workflow (same trials throughout, no odd/even split):
  stage A  compute  S_stim on the clusters 22-28 pool (whole pool, per cluster,
                    per cell); S_prior on the whole pool; choice_* / prior_*
                    on all of C and all of M (per-cell choice_*)
  select            S_cell  = BH(q) over pool cells on S_stim (--s-cell-p P:
                    uncorrected p < P instead)
                    S_clus  = clusters with BH(q) population S_stim p_mean
                    C_choice / M_choice = cells with BH(q) choice p (stim or
                    move window, BH over both windows jointly)
                    -> <out>/stageB_cell_types.npy
  stage B  compute  S_stim / S_prior on S_cell, S_clus; choice_* / prior_* on
                    C_choice, M_choice (populations only)
  stage BS compute  S_stim / S_prior on S_cell only (re-selected S cells,
                    without redoing the other stage-B populations)
  cluster-types     one type per C / M rastermap cluster (needs NEW_STACK)
                    -> data/fig45/cell_types_cm_clusters.npy
  stage CL compute  choice_* on each C / M cluster (--cell-types that file)
  choice-clusters   C vs M choice sensitivity (stage A), and one figure per
                    C / M cluster (stage CL)
  select-cm         C_sel = C clusters with choice p < P (stim or move window),
                    M_sel = M clusters with choice p < P (move window), from
                    stage A per-cell p -> data/fig45/cell_types_cm_choice_p<P>.npy
  stage CS compute  choice_* and prior_* on C_sel / M_sel (--cell-types that file)
  choice-selected   C_sel vs M_sel choice curves (end peak, slopes) and prior
                    curves (offset / gain)
  rm-types          one type per rastermap cluster, all 100 (needs NEW_STACK)
                    -> data/fig45/cell_types_rm100.npy
  stage RM compute  all six analyses on each of the 100 clusters, as populations
                    (clusters treated as regions). Shards split the clusters,
                    not the insertions, so each accumulator holds only its
                    clusters (rm<k>, k = shard mod n_shards)
  select-rm         Figure 2 / 4 region rules on the stage RM clusters:
                    S = 80 ms stim BH <= 0.01, no onset-offset confound, no
                    early choice run; C / M = choice BH <= 0.01 per window,
                    integrator / movement shape (three movement-slope rules);
                    prior BH <= 0.01 in 0-80 ms, stim and move windows
  stack             merge shard accumulators + per-cell files, plots, summary
  compare           S: pool vs S_cell vs S_clus (stim, prior; 0-80 ms);
                    C / M: all vs choice-selected (choice, prior; stim and
                    move windows) -> <out>/compare/

Shards keep one additive accumulator (population sums) checkpointed every
CKPT insertions; per-insertion files hold only per-cell statistics.

Usage (iblenv, outside sandbox; ORCD: scripts/submit_fig45_sensitivity.sh):
  python scripts/fig45_sensitivity.py compute --stage A --shard 0 --n-shards 1
  python scripts/fig45_sensitivity.py stack --stage A
  python scripts/fig45_sensitivity.py select
  python scripts/fig45_sensitivity.py compute --stage B ...
  python scripts/fig45_sensitivity.py stack --stage B
"""
import argparse
import gc
import json
import os
import time
import traceback
import uuid
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from fig45_block_peths import (DEFAULT_ONE_CACHE, Loader, binned, insertion_list,
                               load_cell_types, shuffled_diffs, typed_spikes)
from fig45_common import (CELL_TYPES, COLORS, NEW_STACK, NRAND, OUT_ROOT, PAPER_FIGS, STS,
                          T_BIN, TYPES_DIR, savefig_svg_png)

OUT = OUT_ROOT / 'cvTrue' / 'sensitivity'
MIN_TRIALS = 5
CKPT = 5
Q_FDR = 0.05
ALIGN_EVENT = {'stim': 'stimOn_times', 'move': 'firstMovement_times'}
ANALYSES = {
    'S_stim': dict(align='stim', win=(0, 0.08), contrast='s', strata=('b',), group='S'),
    'S_prior': dict(align='stim', win=(0, 0.08), contrast='b', strata=('s',), group='S'),
    'choice_stim': dict(align='stim', win=(0, 0.15), contrast='c', strata=('s', 'b'),
                        group='CM'),
    'choice_move': dict(align='move', win=(0.15, 0), contrast='c', strata=('s', 'b'),
                        group='CM'),
    'prior_stim': dict(align='stim', win=(0, 0.15), contrast='b', strata=('s', 'c'),
                       group='CM'),
    'prior_move': dict(align='move', win=(0.15, 0), contrast='b', strata=('s', 'c'),
                       group='CM'),
}
S_POOL_CLUSTERS = list(range(22, 29))
CM = ['integrator', 'move_all']
CM_SEL = ['C_choice', 'M_choice']
S_SEL = ['S_cell', 'S_clus']
# rastermap clusters picked for C (integrator) and M (move_all), one type each
_PICKS = json.loads((TYPES_DIR / 'cell_types_picks.json').read_text())
CM_CLUSTERS = {'C': _PICKS['integrator'], 'M': _PICKS['move_all']}
CL_TYPES = [f'rm{c}' for c in sorted(set(CM_CLUSTERS['C']) | set(CM_CLUSTERS['M']))]
CL_CELL_TYPES = TYPES_DIR / 'cell_types_cm_clusters.npy'
# Figure-2-like choice selection: C clusters with choice p < P in either window,
# M clusters with choice p < P around movement (uncorrected, stage A per-cell p)
CS_TYPES = ['C_sel', 'M_sel']
# every rastermap cluster of concat_cvTrue (rm_labels 0-99), one type each
RM_TYPES = [f'rm{c}' for c in range(100)]
RM_CELL_TYPES = TYPES_DIR / 'cell_types_rm100.npy'
# stage -> analysis -> (population types, per-cell test pool or None)
STAGES = {
    'A': {'S_stim': (['stimint'] + [f'rm{c}' for c in S_POOL_CLUSTERS], ['stimint']),
          'S_prior': (['stimint'], None),
          'choice_stim': (CM, CM), 'choice_move': (CM, CM),
          'prior_stim': (CM, None), 'prior_move': (CM, None)},
    'B': {'S_stim': (S_SEL, None), 'S_prior': (S_SEL, None),
          'choice_stim': (CM_SEL, None), 'choice_move': (CM_SEL, None),
          'prior_stim': (CM_SEL, None), 'prior_move': (CM_SEL, None)},
    'BS': {'S_stim': (['S_cell'], None), 'S_prior': (['S_cell'], None)},
    'CL': {'choice_stim': (CL_TYPES, None), 'choice_move': (CL_TYPES, None)},
    'CS': {'choice_stim': (CS_TYPES, None), 'choice_move': (CS_TYPES, None),
           'prior_stim': (CS_TYPES, None), 'prior_move': (CS_TYPES, None)},
    'RM': {name: (RM_TYPES, None) for name in ANALYSES},
}
# comparison figure: population -> (stage, label, colour, line style)
COMPARE = {
    'S': {'stimint': ('A', 'S pool (22-28)', '#57C1EB', '-'),
          'S_cell': ('B', 'S per cell', 'navy', '-'),
          'S_clus': ('B', 'S per cluster', 'steelblue', '--')},
    'CM': {'integrator': ('A', 'C all', 'goldenrod', '-'),
           'C_choice': ('B', 'C choice-selected', 'goldenrod', '--'),
           'move_all': ('A', 'M all', 'tomato', '-'),
           'M_choice': ('B', 'M choice-selected', 'tomato', '--')},
}
STAGE_COLORS = {t: v[2] for g in COMPARE.values() for t, v in g.items()}
FNAME = {'s': 'stim', 'b': 'block', 'c': 'choice'}


def cond_name(align, d):
    return align + ''.join(f'_{k}{v}' for k, v in sorted(d.items()))


def pairs_of(spec):
    out = []
    for vals in product('LR', repeat=len(spec['strata'])):
        st = dict(zip(spec['strata'], vals))
        a = cond_name(spec['align'], {**st, spec['contrast']: 'L'})
        b = cond_name(spec['align'], {**st, spec['contrast']: 'R'})
        out.append((a, b, {**st}))
    return out


def pair_key(a, b):
    return f'{a}|{b}'


def factor_masks(trials):
    pl = trials['probabilityLeft'].to_numpy()
    ch = trials['choice'].to_numpy()
    return {'s': {'L': ~np.isnan(trials['contrastLeft'].to_numpy(dtype=float)),
                  'R': ~np.isnan(trials['contrastRight'].to_numpy(dtype=float))},
            'b': {'L': pl == 0.8, 'R': pl == 0.2},
            'c': {'L': ch == 1, 'R': ch == -1}}


def cond_mask(fm, mask, align, name):
    parts = name[len(align) + 1:].split('_')
    return np.logical_and.reduce([mask] + [fm[p[0]][p[1]] for p in parts])


def time_axis_ms(win, T):
    """Bin centres (ms from the align event) of the overlapping-bin layout."""
    st = int(T_BIN // STS)
    j = np.arange(T)
    return (-win[0] + (j // st) * T_BIN + (j % st) * STS + T_BIN / 2) * 1000


def stage_types(stage):
    return list(dict.fromkeys(t for types, _ in STAGES[stage].values() for t in types))


def analysis_types(stage, name):
    return STAGES[stage][name][0]


def cell_pool(stage, name):
    """Types whose cells get a per-cell test for this analysis, or None."""
    return STAGES[stage][name][1]


def compute_insertion(loaded, cell_types, stage, nrand, rng, br, keep=None):
    """keep: types to compute (default every type of the stage)."""
    src, eid, trials, masks, prior_l, spikes, clusters, in_any, uu_in = loaded
    trials = trials.copy()
    if prior_l is not None:
        trials['probabilityLeft'] = prior_l
    fm = factor_masks(trials)
    acr = np.array(br.id2acronym(np.asarray(clusters['atlas_id'])[in_any], mapping='Beryl'))
    sp = typed_spikes(spikes, clusters, in_any)
    keep = stage_types(stage) if keep is None else keep
    tmask = {t: np.isin(uu_in, list(cell_types.get(t, ()))) for t in keep}
    pop, cells = {}, {}
    for name in STAGES[stage]:
        spec = ANALYSES[name]
        ev_all = trials[ALIGN_EVENT[spec['align']]].to_numpy()
        types = [t for t in analysis_types(stage, name) if t in tmask]
        pool = cell_pool(stage, name)
        pool_m = np.logical_or.reduce([tmask[t] for t in pool]) if pool else None
        res = {t: {'n': {}, 'dist': {}, 'ctrl': {}} for t in types}
        ntr = {}
        stat = np.zeros(len(uu_in))
        null = np.zeros((nrand, len(uu_in)), dtype=np.float32) if pool else None
        n_ok = 0
        align_mask = masks[spec['align']]
        for a, b, _ in pairs_of(spec):
            key = pair_key(a, b)
            ma = cond_mask(fm, align_mask, spec['align'], a)
            mb = cond_mask(fm, align_mask, spec['align'], b)
            ntr[key] = (int(ma.sum()), int(mb.sum()))
            if min(ntr[key]) < MIN_TRIALS:
                continue
            A = binned(sp, ev_all[ma], spec['win'])
            B = binned(sp, ev_all[mb], spec['win'])
            diff = A.mean(0) - B.mean(0)
            D = shuffled_diffs(A, B, nrand, rng)
            active = (A.sum((0, 2)) + B.sum((0, 2))) > 0
            for t in types:
                m = tmask[t] & active
                if not m.any():
                    continue
                res[t]['n'][key] = int(m.sum())
                res[t]['dist'][key] = (diff[m] ** 2).sum(0)
                res[t]['ctrl'][key] = (D[:, m] ** 2).sum(1)
            if pool:
                stat += (diff ** 2).mean(1)
                null += (D ** 2).mean(2)
                n_ok += 1
            del A, B, D
        pop[name] = res
        if pool and n_ok and pool_m.any():
            cells[name] = dict(uuids=uu_in[pool_m], stat=(stat / n_ok)[pool_m],
                               null=(null / n_ok)[:, pool_m], n_pairs=n_ok,
                               acronym=acr[pool_m])
        cells.setdefault('_ntr', {})[name] = ntr
    return pop, cells


def add_to_acc(acc, pop):
    for name, res in pop.items():
        for t, r in res.items():
            a = acc.setdefault(name, {}).setdefault(t, {'n': {}, 'dist': {}, 'ctrl': {}})
            for key in r['n']:
                a['n'][key] = a['n'].get(key, 0) + r['n'][key]
                a['dist'][key] = a['dist'].get(key, 0) + r['dist'][key].astype(np.float64)
                a['ctrl'][key] = a['ctrl'].get(key, 0) + r['ctrl'][key].astype(np.float64)


def save_atomic(obj, path):
    tmp = path.with_name(f'.{path.stem}.tmp.npy')
    np.save(tmp, obj, allow_pickle=True)
    tmp.replace(path)


def stage_dir(out, stage):
    return out / f'stage{stage}'


def cmd_compute(args):
    from iblatlas.regions import BrainRegions
    br = BrainRegions()
    ct_path = args.cell_types or {'A': CELL_TYPES, 'RM': RM_CELL_TYPES}.get(
        args.stage, args.out / 'stageB_cell_types.npy')
    cell_types, ct_raw = load_cell_types(ct_path)
    types = [t for t in stage_types(args.stage) if t in ct_raw['types']]
    missing = set(stage_types(args.stage)) - set(types)
    if missing:
        raise KeyError(f'{ct_path} lacks types {sorted(missing)}')
    if args.stage == 'RM':  # shard over clusters: every insertion of this shard's clusters
        types = types[args.shard::args.n_shards]
    pids = insertion_list(ct_raw, types)
    if args.pids:
        pids = [p for p in pids if p in set(args.pids)]
    if args.stage != 'RM':
        pids = pids[args.shard::args.n_shards]
    sdir = stage_dir(args.out, args.stage)
    cdir = sdir / 'cells'
    cdir.mkdir(parents=True, exist_ok=True)
    acc_f = sdir / f'acc_shard{args.shard}of{args.n_shards}.npy'
    acc = np.load(acc_f, allow_pickle=True).item() if acc_f.exists() and not args.overwrite \
        else {'done': [], 'pop': {}, 'nrand': args.nrand, 'cell_types': str(ct_path)}
    done = set(acc['done'])
    load = Loader(args, ct_raw)
    print(f'stage {args.stage} shard {args.shard}/{args.n_shards}: {len(pids)} insertions, '
          f'{len(done)} done; types {types}', flush=True)
    fails, since = [], 0
    for k, pid in enumerate(pids):
        if pid in done:
            continue
        t0 = time.perf_counter()
        rng = np.random.default_rng([args.seed, uuid.UUID(pid).int % (2 ** 32),
                                     int.from_bytes(args.stage.encode(), "big")])
        try:
            loaded = load(pid, cell_types, types)
            if loaded[7].any():
                pop, cells = compute_insertion(loaded, cell_types, args.stage, args.nrand,
                                               rng, br, keep=types)
                add_to_acc(acc['pop'], pop)
                cells['pid'] = pid
                save_atomic(cells, cdir / f'{pid}.npy')
            acc['done'].append(pid)
            since += 1
            if since >= CKPT:
                save_atomic(acc, acc_f)
                since = 0
            print(f'{k + 1}/{len(pids)} {pid}: {int(loaded[7].sum())} cells, {loaded[0]}, '
                  f'{time.perf_counter() - t0:.0f}s', flush=True)
        except Exception:
            fails.append(pid)
            print(f'{k + 1}/{len(pids)} {pid}: FAIL\n{traceback.format_exc()}', flush=True)
        gc.collect()
    save_atomic(acc, acc_f)
    print(f'{len(fails)} failures: {fails}')


def bh(p, q=Q_FDR):
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    sig = np.zeros(len(p), bool)
    if ok.any():
        from statsmodels.stats.multitest import multipletests
        sig[ok] = multipletests(p[ok], q, method='fdr_bh')[0]
    return sig


def merged(sdir):
    pop, nrand, n_done = {}, None, 0
    for f in sorted(sdir.glob('acc_shard*.npy')):
        a = np.load(f, allow_pickle=True).item()
        nrand = a['nrand']
        n_done += len(a['done'])
        for name, res in a['pop'].items():
            for t, r in res.items():
                add_to_acc(pop, {name: {t: r}})
    cells = {}
    for f in sorted((sdir / 'cells').glob('*.npy')):
        c = np.load(f, allow_pickle=True).item()
        for name, v in c.items():
            if name in ('pid', '_ntr'):
                continue
            cells.setdefault(name, []).append({**v, 'pid': c['pid']})
    return pop, cells, nrand, n_done


def population_traces(r):
    keys = sorted(r['n'])
    d = [np.sqrt(r['dist'][k] / T_BIN / r['n'][k]) for k in keys]
    c = [np.sqrt(r['ctrl'][k] / T_BIN / r['n'][k]) for k in keys]
    return keys, np.mean(d, 0), np.mean(c, 0)


# Regional offset/gain, on the 12.5 ms bins (mean of the 6 overlapping sub-bins).
# Offset: mean of the first 5 stim-aligned bins (0-62.5 ms) minus the shuffle mean.
# Gain: after removing that offset when it is significant, the peak of the remaining
# bins (from 50 ms, or from -100 ms on the move trace) minus the shuffle at that time.
PRIOR_OFFSET_BINS = 5
PRIOR_GAIN_SKIP = 4
PRIOR_ALPHA = 0.05


def coarse_bins(tr):
    st = int(T_BIN // STS)
    n = tr.shape[-1] // st
    return tr.reshape(tr.shape[:-1] + (n, st)).mean(-1)


def prior_offset_gain(md, sc, win, alpha=PRIOR_ALPHA, offset_from=None):
    """Offset and gain of one prior-distance curve against its shuffles.

    offset_from=(md, sc) uses another curve's stim-aligned onset for the offset
    removed before the gain test, as the move-aligned regional test does.
    Sizes are the real value minus the shuffle mean (early-window mean for the
    offset; the curve's peak for the gain).
    """
    md_c, sc_c = coarse_bins(md), coarse_bins(sc)
    if offset_from is None:
        off_md, off_sc = md_c, sc_c
    else:
        off_md, off_sc = coarse_bins(offset_from[0]), coarse_bins(offset_from[1])
    early = np.concatenate([[off_md[:PRIOR_OFFSET_BINS].mean()],
                            off_sc[:, :PRIOR_OFFSET_BINS].mean(1)])
    p_offset = float(np.mean(early >= early[0]))
    offset = float(early[0] - early[1:].mean())
    shifted = md_c - (offset if p_offset < alpha else 0.0)
    real_g = shifted[PRIOR_GAIN_SKIP:]
    shuf_g = sc_c[:, PRIOR_GAIN_SKIP:]
    joined = np.concatenate([[real_g.mean()], shuf_g.mean(1)])
    p_gain = float(np.mean(joined >= joined[0]))
    peak = int(np.argmax(real_g))
    centers = (-win[0] + (np.arange(len(md_c)) + 0.5) * T_BIN) * 1000
    return {'p_offset': p_offset,
            'offset': offset,
            'offset_real': float(early[0]),
            'offset_shuffle': float(early[1:].mean()),
            'offset_sig': bool(p_offset < alpha),
            'p_gain': p_gain,
            'gain': float(real_g[peak] - shuf_g[:, peak].mean()),
            'gain_real': float(real_g[peak]),
            'gain_shuffle': float(shuf_g[:, peak].mean()),
            'gain_t_ms': float(centers[PRIOR_GAIN_SKIP + peak]),
            'gain_sig': bool(p_gain < alpha)}


def plot_pop(name, t, x, md, sc, st, n, out_dir):
    spec = ANALYSES[name]
    col = STAGE_COLORS.get(t, COLORS.get(t, 'k'))
    fig, ax = plt.subplots(figsize=(4, 2.8))
    for i in range(min(100, len(sc))):
        ax.plot(x, sc[i], color='silver', lw=0.2, alpha=0.5)
    ax.plot(x, md, color=col, lw=2)
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel(f'time from {spec["align"]} (ms)')
    ax.set_ylabel(f'd_{FNAME[spec["contrast"]]}')
    strata = ' x '.join(FNAME[s] for s in spec['strata'])
    ax.set_title(f'{t}: {name} (strata {strata})', fontsize=8)
    ax.text(0.98, 0.04, f"p_mean {st['p_mean']:.4f}\np_gain {st['p_gain']:.4f}\nn~{n}",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=7)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    savefig_svg_png(fig, out_dir / f'{name}_{t}')
    plt.close(fig)


def cell_pvals(rows):
    uu = np.concatenate([r['uuids'] for r in rows])
    pid = np.concatenate([np.repeat(r['pid'], len(r['uuids'])) for r in rows])
    stat = np.concatenate([r['stat'] for r in rows])
    null = np.concatenate([r['null'] for r in rows], axis=1)
    p = (1 + (null >= stat[None]).sum(0)) / (1 + null.shape[0])
    return uu, pid, stat, p


def cmd_stack(args):
    from fig45_plot_distances import distance_stats
    sdir = stage_dir(args.out, args.stage)
    pop, cells, nrand, n_done = merged(sdir)
    pdir = sdir / 'plots'
    summary = {'stage': args.stage, 'nrand': nrand, 'insertions_done': n_done,
               'min_trials': MIN_TRIALS, 'population': {}, 'cells': {}}
    for name, res in pop.items():
        spec = ANALYSES[name]
        for t, r in res.items():
            if not r['n']:
                continue
            keys, md, sc = population_traces(r)
            st = distance_stats(md, sc)
            x = time_axis_ms(spec['win'], len(md))
            n = int(np.median([r['n'][k] for k in keys]))
            plot_pop(name, t, x, md, sc, st, n, pdir)
            summary['population'].setdefault(name, {})[t] = {
                **{k: v for k, v in st.items() if k != 'shifted'},
                'n_per_pair': {k: r['n'][k] for k in keys}}
    for name, rows in cells.items():
        uu, pid, stat, p = cell_pvals(rows)
        summary['cells'][name] = {'n_cells': len(uu), 'n_p05': int((p < 0.05).sum()),
                                  'n_bh': int(bh(p).sum())}
        np.save(sdir / f'cells_{name}.npy', dict(uuids=uu, pid=pid, stat=stat, p=p),
                allow_pickle=True)
    np.save(sdir / 'population.npy', pop, allow_pickle=True)
    if args.stage == 'RM':  # pair-averaged curves only (~0.3 GB), what select-rm reads
        tr = {}
        for name, res in pop.items():
            for t, r in res.items():
                if r['n']:
                    keys, md, sc = population_traces(r)
                    tr.setdefault(name, {})[t] = dict(
                        md=md.astype(np.float32), sc=sc.astype(np.float32),
                        n=int(np.median([r['n'][k] for k in keys])))
        np.save(sdir / 'traces.npy', tr, allow_pickle=True)
    (sdir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'population'}, indent=2))
    for name, v in summary['population'].items():
        for t, st in v.items():
            print(f'{name:12s} {t:12s} p_mean={st["p_mean"]:.4f} p_gain={st["p_gain"]:.4f}')


def cmd_compare(args):
    """Shuffle-mean-subtracted population distances, one panel per analysis:
    S pool / per cell / per cluster (0-80 ms), C / M all vs choice-selected."""
    from fig45_plot_distances import distance_stats
    pops = {s: np.load(stage_dir(args.out, s) / 'population.npy', allow_pickle=True).item()
            for s in STAGES if (stage_dir(args.out, s) / 'population.npy').exists()}
    layout = {'S': [['S_stim', 'S_prior']],
              'CM': [['choice_stim', 'choice_move'], ['prior_stim', 'prior_move']]}
    cdir = args.out / 'compare'
    summary = {}
    for group, grid in layout.items():
        fig, axs = plt.subplots(len(grid), len(grid[0]), squeeze=False,
                                figsize=(4.2 * len(grid[0]), 3.0 * len(grid)))
        for row, names in zip(axs, grid):
            for ax, name in zip(row, names):
                spec = ANALYSES[name]
                for t, (stage, label, col, ls) in COMPARE[group].items():
                    r = pops[stage].get(name, {}).get(t)
                    if not r or not r['n']:
                        continue
                    keys, md, sc = population_traces(r)
                    st = distance_stats(md, sc)
                    x = time_axis_ms(spec['win'], len(md))
                    cm = sc.mean(0)
                    lo, hi = np.percentile(sc - cm, [2.5, 97.5], axis=0)
                    n = int(np.median([r['n'][k] for k in keys]))
                    og = None
                    if spec['contrast'] == 'b':
                        off_from = None
                        if name == 'prior_move':
                            _, smd, ssc = population_traces(pops[stage]['prior_stim'][t])
                            off_from = (smd, ssc)
                        og = prior_offset_gain(md, sc, spec['win'], offset_from=off_from)
                    ax.fill_between(x, lo, hi, color=col, alpha=0.12, lw=0)
                    ax.plot(x, md - cm, color=col, ls=ls, lw=1.8,
                            label=f"{label} ({n} cells/pair) p={st['p_mean']:.4f}")
                    summary.setdefault(name, {})[t] = {
                        'label': label, 'stage': stage, 'n_median': n,
                        **{k: v for k, v in st.items() if k != 'shifted'},
                        **({'offset_gain': og} if og else {})}
                ax.axhline(0, color='k', lw=0.5)
                ax.axvline(0, color='k', lw=0.5)
                strata = ' x '.join(FNAME[s] for s in spec['strata'])
                ax.set_title(f'{FNAME[spec["contrast"]]} sensitivity, {spec["align"]}-aligned'
                             f' (strata {strata})', fontsize=8)
                ax.set_xlabel(f'time from {spec["align"]} (ms)')
                ax.set_ylabel('d - shuffle mean')
                ax.legend(frameon=False, fontsize=6)
                for s in ('top', 'right'):
                    ax.spines[s].set_visible(False)
        fig.tight_layout()
        savefig_svg_png(fig, cdir / f'compare_{group}')
        plt.close(fig)
    (cdir / 'summary.json').write_text(json.dumps(summary, indent=2))
    for name, v in summary.items():
        for t, st in v.items():
            print(f'{name:12s} {st["label"]:20s} n~{st["n_median"]:5d} '
                  f'p_mean={st["p_mean"]:.4f}')
            og = st.get('offset_gain')
            if not og:
                continue
            off_label = 'offset (from stim)' if name == 'prior_move' else 'offset'
            off = 'SIG' if og['offset_sig'] else 'ns'
            gn = 'SIG' if og['gain_sig'] else 'ns'
            removed = 'removed' if og['offset_sig'] else 'kept'
            print(f'{"":12s} {off_label:20s} {og["offset_real"]:.4f} vs shuffle '
                  f'{og["offset_shuffle"]:.4f}  excess {og["offset"]:+.4f}  '
                  f'p={og["p_offset"]:.4f} {off}')
            print(f'{"":12s} {"gain peak":20s} {og["gain_real"]:.4f} vs shuffle '
                  f'{og["gain_shuffle"]:.4f}  excess {og["gain"]:+.4f}  '
                  f'at {og["gain_t_ms"]:.0f} ms  p={og["p_gain"]:.4f} {gn}'
                  f'  (offset {removed})')
    print('wrote', cdir)


def subset_types(ct_raw, wanted):
    """{name: uuid set} -> same per-cell fields as cell_types.npy, from its union."""
    allrows = {}
    for v in ct_raw['types'].values():
        for i, u in enumerate(v['uuids']):
            allrows.setdefault(u, (v['pid'][i], v['ids'][i], v['cluster_id'][i]))
    out = {}
    for name, uus in wanted.items():
        uus = [u for u in uus if u in allrows]
        rows = [allrows[u] for u in uus]
        out[name] = {'uuids': np.array(uus, dtype=object),
                     'pid': np.array([r[0] for r in rows], dtype=object),
                     'ids': np.array([r[1] for r in rows]),
                     'cluster_id': np.array([r[2] for r in rows], dtype=np.int64)}
    return out


def cmd_select(args):
    sdir = stage_dir(args.out, 'A')
    summ = json.loads((sdir / 'summary.json').read_text())
    _, ct_raw = load_cell_types(args.cell_types or CELL_TYPES)
    sel, info = {}, {}
    c = np.load(sdir / 'cells_S_stim.npy', allow_pickle=True).item()
    if args.s_cell_p is None:
        sig, rule = bh(c['p'], args.q), f'BH q={args.q}'
    else:
        sig, rule = c['p'] < args.s_cell_p, f'uncorrected p<{args.s_cell_p}'
    sel['S_cell'] = set(c['uuids'][sig])
    info['S_cell'] = {'pool': len(c['uuids']), 'selected': int(sig.sum()), 'rule': rule}
    s_pop = summ['population']['S_stim']
    pc = {k: s_pop[f'rm{k}']['p_mean'] for k in S_POOL_CLUSTERS if f'rm{k}' in s_pop}
    ks = list(pc)
    sig_cl = bh([pc[k] for k in ks], args.q)
    clus = [k for k, s in zip(ks, sig_cl) if s]
    sel['S_clus'] = set().union(*[set(ct_raw['types'][f'rm{k}']['uuids']) for k in clus]) \
        if clus else set()
    info['S_clus'] = {'p_mean': pc, 'clusters': clus, 'selected': len(sel['S_clus'])}
    cs = np.load(sdir / 'cells_choice_stim.npy', allow_pickle=True).item()
    cm = np.load(sdir / 'cells_choice_move.npy', allow_pickle=True).item()
    for out_name, src in (('C_choice', 'integrator'), ('M_choice', 'move_all')):
        members = set(ct_raw['types'][src]['uuids'])
        ms = np.isin(cs['uuids'], list(members))
        mm = np.isin(cm['uuids'], list(members))
        sig = bh(np.concatenate([cs['p'][ms], cm['p'][mm]]), args.q)
        chosen = set(cs['uuids'][ms][sig[:ms.sum()]]) | set(cm['uuids'][mm][sig[ms.sum():]])
        sel[out_name] = chosen
        info[out_name] = {'from': src, 'tested_stim': int(ms.sum()),
                          'tested_move': int(mm.sum()), 'selected': len(chosen)}
    out = {'source': str(args.cell_types or CELL_TYPES), 'q': args.q, 'info': info,
           'types': subset_types(ct_raw, sel), 'insertions': ct_raw['insertions']}
    save_atomic(out, args.out / 'stageB_cell_types.npy')
    (args.out / 'stageB_selection.json').write_text(json.dumps(info, indent=2, default=str))
    print(json.dumps(info, indent=2, default=str))


def _block_curve(pops, stage, name, typ):
    r = pops[stage][name][typ]
    keys, md, sc = population_traces(r)
    n = int(np.median([r['n'][k] for k in keys]))
    return md, sc, n


# Figure 4 house style (scripts/paper_figs/fig4): Helvetica, 11 pt ticks, Times
# y labels, 26 pt panel letters, filled S / C / M circles, layout in points.
FIG4_RC = {'font.family': 'Helvetica', 'svg.fonttype': 'none', 'pdf.fonttype': 42,
           'axes.spines.top': False, 'axes.spines.right': False,
           'xtick.labelsize': 11, 'ytick.labelsize': 11, 'axes.labelsize': 11}
PAGE_W, PAGE_H = 759.276, 600.0  # pt; Figure 4 page width
BADGE = {'S': '#57C1EB', 'C': '#F3D37A', 'M': '#E24A33'}
INSET_BAND = 0.30  # top fraction of each distance panel kept free for badge + inset


def _page(fig):
    return tuple(fig.get_size_inches() * 72)


def _box(fig, x0, y0, x1, y1):
    """Axes rect from a top-left-origin box in points."""
    w, h = _page(fig)
    return [x0 / w, 1 - y1 / h, (x1 - x0) / w, (y1 - y0) / h]


def _overlay(fig):
    w, h = _page(fig)
    ax = fig.add_axes([0, 0, 1, 1], zorder=5)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis('off')
    ax.patch.set_alpha(0)
    return ax


def _badge(ov, x, y, letter, radius=9):
    """Figure 4 circle label at (x, y) pt."""
    ov.add_patch(plt.Circle((x, y), radius, facecolor=BADGE[letter], edgecolor='none',
                            clip_on=False))
    ov.text(x, y, letter, fontsize=12, ha='center', va='center',
            color='white' if letter == 'M' else 'k', zorder=6)


def _letter(fig, x, y, s):
    w, h = _page(fig)
    fig.text(x / w, 1 - y / h, s, fontsize=26, va='center', ha='left')


def _xlabel(fig, x, y, s='time (ms)'):
    w, h = _page(fig)
    fig.text(x / w, 1 - y / h, s, fontsize=14, ha='center', va='top')


def _style_ax(ax, yticks, ylabel=None):
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{v:g}'.replace('-', '\u2212') for v in np.round(yticks, 3)])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, fontfamily='Times New Roman', labelpad=0)
    for s in ('left', 'bottom'):
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(length=3, width=0.8)


def _data_ticks(lo, hi, n=4):
    """At most n round ticks inside [lo, hi]."""
    from matplotlib.ticker import MaxNLocator
    t = MaxNLocator(nbins=n, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
    return t[(t >= lo - 1e-9) & (t <= hi + 1e-9)]


def _draw_block_panel(ax, md, sc, color, win, stim=True, hist=None, ylabel=None,
                      show_x=True):
    """Fig 4.5 distance panel. The blue offset mark is the first 5 samples.

    Curves fill the lower 1 - INSET_BAND of the axes; the band above them holds
    the badge and the shuffle-subtracted inset, so neither covers a curve.
    """
    from fig45_plot_distances import GAIN_SKIP, OFFSET_WINDOW, distance_stats
    st = distance_stats(md, sc)
    x = time_axis_ms(win, len(md))
    early, gain0 = OFFSET_WINDOW, GAIN_SKIP
    for i in range(min(80, len(sc))):
        if stim:
            ax.plot(x[:early], sc[i, :early], color='#5f7ea3', lw=0.2, alpha=0.35)
            ax.plot(x[early - 1:], sc[i, early - 1:], color='0.75', lw=0.2, alpha=0.35)
        else:
            ax.plot(x, sc[i], color='0.75', lw=0.2, alpha=0.35)
    if stim:
        ax.plot(x[:early], md[:early], color='blue', ls='--', lw=1.3)
        ax.plot(x[early - 1:], md[early - 1:], color=color, ls='--', lw=1.3)
    else:
        ax.plot(x, md, color=color, ls='--', lw=1.3)
    curves = [md, sc[:80].ravel()]
    if st['offset'] != 0.0:
        ax.plot(x, st['shifted'], color=color, lw=1.5)
        curves.append(st['shifted'])
    lo = min(c.min() for c in curves)
    hi = max(c.max() for c in curves)
    pad = 0.04 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    ax.set_ylim(lo, lo + (hi - lo) / (1 - INSET_BAND))
    _style_ax(ax, _data_ticks(lo, hi), ylabel)
    # Starts right of the badge so the inset's y tick labels clear it.
    ins = ax.inset_axes([0.48, 1 - INSET_BAND + 0.02, 0.50, INSET_BAND - 0.04])
    cm = sc.mean(0)
    qlo, qhi = np.percentile(sc - cm, [2.5, 97.5], axis=0)
    ins.fill_between(x, qlo, qhi, color='0.75', alpha=0.7, lw=0)
    ins.plot(x, md - cm, color=color, lw=0.8)
    ins.axhline(0, color='k', lw=0.4)
    ins.set_xlim(x[0], x[-1])
    ins.set_xticks([])
    from matplotlib.ticker import MaxNLocator
    ins.yaxis.set_major_locator(MaxNLocator(3))
    ins.tick_params(axis='y', labelsize=5.5, length=2, width=0.6, pad=1)
    ins.patch.set_alpha(0)
    for s in ('top', 'right'):
        ins.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ins.spines[s].set_linewidth(0.6)
    if hist is not None:
        hist.hist(sc.mean(1), bins=18, density=True, orientation='horizontal',
                  color='0.6', alpha=0.7)
        hist.axhline(md.mean(), ls='--', lw=1.1, color=color)
        rows = [(md.mean(), st['p_mean'] < PRIOR_ALPHA)]
        if stim:
            hist.hist(sc[:, :early].mean(1), bins=18, density=True, orientation='horizontal',
                      color='#5f7ea3', alpha=0.45)
            hist.axhline(md[:early].mean(), color='blue', ls='--', lw=1.1)
            rows.append((md[:early].mean(), st['p_offset'] < PRIOR_ALPHA))
        hist.hist(sc[:, gain0:].mean(1), bins=18, density=True, orientation='horizontal',
                  color='#c8a2d6', alpha=0.55)
        hist.axhline(st['shifted'][gain0:].mean(), color=color, lw=1.1)
        rows.append((st['shifted'][gain0:].mean(), st['p_gain'] < PRIOR_ALPHA))
        xright = hist.get_xlim()[1]
        dy = 0.028 * (hist.get_ylim()[1] - hist.get_ylim()[0])
        for y, _sig in rows:
            hist.plot([xright, xright], [y - dy, y + dy], color='k', lw=0.8,
                      clip_on=False, zorder=5, solid_capstyle='butt')
        _stars(hist, rows)
        hist.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        for s in ('top', 'right', 'bottom'):
            hist.spines[s].set_visible(False)
        hist.spines['left'].set_visible(True)
        hist.spines['left'].set_linewidth(0.8)
    ax.axvline(0, color='#57C1EB' if stim else '#e74c3c', lw=1.4)
    ax.set_xticks([0, 40, 80] if (stim and win[1] <= 0.1) else
                  ([0, 50, 100, 150] if stim else [-150, -100, -50, 0]))
    if not stim:
        ax.set_xticklabels(['−150', '−100', '−50', '0'])
    if not show_x:
        ax.tick_params(axis='x', labelbottom=False)
    return st


def _stars(hist, rows, min_gap=0.07):
    """rows: (y, sig). A star sits to the right of a significant reference line;
    stars closer than ``min_gap`` (axes fraction) are spread apart."""
    lo, hi = hist.get_ylim()
    ys = sorted((y - lo) / (hi - lo) for y, sig in rows if sig)
    groups = []
    for y in ys:
        if groups and y - groups[-1][-1] < min_gap:
            groups[-1].append(y)
        else:
            groups.append([y])
    ys = [np.mean(g) + min_gap * (i - (len(g) - 1) / 2)
          for g in groups for i in range(len(g))]
    for y in ys:
        hist.text(1.08, y, '*', transform=hist.transAxes, ha='left', va='center',
                  fontsize=11, clip_on=False)


def _place_ylabels(fig, gap_pt=3):
    """Put each y label ``gap_pt`` left of its tick labels (mathtext labels in
    Times otherwise land on the tick numbers)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    px = fig.dpi / 72
    for ax in fig.axes:
        lab = ax.yaxis.label
        if not lab.get_text():
            continue
        ticks = [t.get_window_extent(r) for t in ax.get_yticklabels()
                 if t.get_visible() and t.get_text()]
        left = min(b.x0 for b in ticks) if ticks else ax.bbox.x0
        w = lab.get_window_extent(r).width
        lab.set_va('center')
        lab.set_ha('center')
        ax.yaxis.set_label_coords((left - gap_pt * px - w / 2 - ax.bbox.x0) / ax.bbox.width,
                                  0.5)


# Panel a: each C / M row range is outlined over these raster columns (first, last).
TYPE_COLS = {'C': ('LLL,s', 'RRR,s'), 'M': ('LLL,m', 'R move')}


def _draw_raster(fig, ov, cache, box, letters, labels):
    """Row-binned raster in ``box`` (pt); each letter's rows outlined over its TYPE_COLS."""
    from matplotlib.colors import PowerNorm
    from matplotlib.patches import Rectangle
    part = 'top' if labels else 'zoom'
    img, n = cache[part], int(cache[f'n_{part}'])
    ax = fig.add_axes(_box(fig, *box))
    ax.imshow(img, cmap='gray_r', norm=PowerNorm(gamma=0.5, vmin=0, vmax=1.5),
              aspect='auto', interpolation='antialiased', extent=(0, img.shape[1], n, 0))
    h = 0
    for short, w in zip(cache['seg_short'], cache['seg_lens']):
        if h:
            ax.axvline(h, ls='--', lw=0.4, color='grey')
        if labels:
            ax.text(h + w / 2, 1.02, short, rotation=90, ha='center', va='bottom',
                    fontsize=7, transform=ax.get_xaxis_transform())
        h += w
    ax.set_xticks([])
    ax.tick_params(axis='y', length=3, width=0.8, labelsize=9)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    starts = np.cumsum([0] + list(cache['seg_lens'][:-1]))
    col = {str(k): (int(a), int(a + w)) for k, a, w in
           zip(cache['seg_short'], starts, cache['seg_lens'])}
    for L in letters:
        c0, c1 = col[TYPE_COLS[L][0]][0], col[TYPE_COLS[L][1]][1]
        for a, b in cache[f'{part}_{L}']:
            ax.add_patch(Rectangle((c0, a), c1 - c0, b - a, facecolor='none',
                                   edgecolor=BADGE[L], lw=1.2, zorder=3))
    return ax, n


def _panel_a(fig, ov, cache_f):
    if not cache_f.exists():
        raise FileNotFoundError(f'{cache_f}: run scripts/fig45_panel_a.py first')
    cache = np.load(cache_f)
    ax, n = _draw_raster(fig, ov, cache, (52, 62, 300, 262), 'CM', True)
    ax.set_yticks([0, 50000], ['1', '50K'])
    ax.set_ylabel('cell #', fontsize=11, labelpad=2)
    for L, x in {'C': 338, 'M': 358}.items():
        _badge(ov, x, 50, L, radius=8)
    az, nz = _draw_raster(fig, ov, cache, (52, 276, 300, 372), 'CM', False)
    az.set_yticks([0, 10000, 20000], ['1', '10K', '20K'])
    az.set_ylabel('C + M cells', fontsize=11, labelpad=2)


def _draw_panel_e(fig, ov, curves, wins, colors, y0, y1, boxes, xlabel_y):
    """Panel e: prior distance minus shuffle mean; S (if present) and C, M on the
    stim side, C and M on the move side. Returns peak-to-trough amplitudes."""
    amps = {}
    stim_keys = tuple(k for k in ('S', 'C_stim', 'M_stim') if k in curves)
    for side, keys in (('stim', stim_keys), ('move', ('C_move', 'M_move'))):
        ax = fig.add_axes(_box(fig, boxes[side][0], y0, boxes[side][1], y1))
        trs = []
        for key in keys:
            md, sc, _ = curves[key]
            tr = md - sc.mean(0)
            x = time_axis_ms(wins[key], len(tr))
            ax.plot(x, tr, color=colors[key], lw=1.3)
            trs.append(tr)
            amps.setdefault(side, {})[key[0]] = float(tr.max() - tr.min())
        amps[side]['M_over_C'] = amps[side]['M'] / amps[side]['C']
        lo, hi = min(t.min() for t in trs), max(t.max() for t in trs)
        pad = 0.04 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        ax.set_ylim(lo, lo + (hi - lo) / (1 - INSET_BAND))
        tag = 's' if side == 'stim' else 'm'
        grp = 'S,C,M' if side == 'stim' and 'S' in curves else 'C,M'
        _style_ax(ax, _data_ticks(lo, hi),
                  rf'$d^{{\mathrm{{prior}},{tag}}}_{{\{{{grp}\}}}}(t)$')
        ax.axvline(0, color='#57C1EB' if side == 'stim' else '#e74c3c', lw=1.4)
        if side == 'stim':
            ax.set_xticks([0, 50, 100, 150])
        else:
            ax.set_xticks([-150, -100, -50, 0])
            ax.set_xticklabels(['−150', '−100', '−50', '0'])
        _xlabel(fig, sum(boxes[side]) / 2, xlabel_y)
    x0 = boxes['stim'][0]
    for i, L in enumerate('SCM' if 'S' in curves else 'CM'):
        _badge(ov, x0 + 14 + 22 * i, y0 + 0.15 * (y1 - y0), L)
    return amps


def cmd_fig45(args):
    """Figure 4.5 in the published layout: a raster, b C/M 2x2, c S, d schematic, e C vs M.

    C and M are the act-block distance within stim x choice. S per cell is its
    act-block distance within stim side, 0-80 ms. Fonts, sizes and S / C / M
    circles follow Figure 4 (scripts/paper_figs/fig4/fig4_plot.py).
    """
    from fig45_plot_distances import distance_stats
    pops = {s: np.load(stage_dir(args.out, s) / 'population.npy', allow_pickle=True).item()
            for s in STAGES if (stage_dir(args.out, s) / 'population.npy').exists()}
    spec = {name: ANALYSES[name] for name in ('S_prior', 'prior_stim', 'prior_move')}
    curves = {
        'S': _block_curve(pops, 'BS' if 'BS' in pops else 'B', 'S_prior', 'S_cell'),
        'C_stim': _block_curve(pops, 'A', 'prior_stim', 'integrator'),
        'C_move': _block_curve(pops, 'A', 'prior_move', 'integrator'),
        'M_stim': _block_curve(pops, 'A', 'prior_stim', 'move_all'),
        'M_move': _block_curve(pops, 'A', 'prior_move', 'move_all'),
    }
    og = {}
    for key, (md, sc, _) in curves.items():
        if key == 'S':
            win, off = spec['S_prior']['win'], None
        elif key.endswith('move'):
            win = spec['prior_move']['win']
            off = curves[key.replace('move', 'stim')][:2]
        else:
            win, off = spec['prior_stim']['win'], None
        og[key] = prior_offset_gain(md, sc, win, offset_from=off)
    odir = args.fig_dir
    wins = {'S': spec['S_prior']['win'], 'C_stim': spec['prior_stim']['win'],
            'C_move': spec['prior_move']['win'], 'M_stim': spec['prior_stim']['win'],
            'M_move': spec['prior_move']['win']}
    colors = {'S': '#5BA4D9', 'C_stim': '#f4d03f', 'C_move': '#f4d03f',
              'M_stim': '#e07a5f', 'M_move': '#e07a5f'}
    ylabels = {
        'S': r'$d^{\mathrm{prior},s}_{S}(t)$',
        'C_stim': r'$d^{\mathrm{prior},s}_{C}(t)$',
        'C_move': r'$d^{\mathrm{prior},m}_{C}(t)$',
        'M_stim': r'$d^{\mathrm{prior},s}_{M}(t)$',
        'M_move': r'$d^{\mathrm{prior},m}_{M}(t)$',
    }
    summary = {k: {'n': n, 'p_mean': distance_stats(md, sc)['p_mean'], 'offset_gain': o}
               for k, ((md, sc, n), o) in ((k, (curves[k], og[k])) for k in curves)}

    with plt.rc_context(FIG4_RC):
        fig = plt.figure(figsize=(PAGE_W / 72, PAGE_H / 72))
        fig.patch.set_facecolor('white')
        ov = _overlay(fig)

        # --- a: raster, colour strips S / C / M ---
        _panel_a(fig, ov, OUT_ROOT / 'cvTrue' / 'plots' / 'panel_a_test.npz')
        _letter(fig, 2, 18, 'a')

        # --- b: C / M, stim then move ---
        cols = {'stim': (440, 538, 541, 556), 'move': (630, 728, 731, 746)}
        rows = {'C': (40, 160), 'M': (196, 316)}
        for key in ('C_stim', 'C_move', 'M_stim', 'M_move'):
            L, side = key.split('_')
            ax0, ax1, h0, h1 = cols[side]
            y0, y1 = rows[L]
            ax = fig.add_axes(_box(fig, ax0, y0, ax1, y1))
            hist = fig.add_axes(_box(fig, h0, y0, h1, y1), sharey=ax)
            md, sc, _ = curves[key]
            _draw_block_panel(ax, md, sc, colors[key], wins[key], stim=side == 'stim',
                              hist=hist, ylabel=ylabels[key], show_x=L == 'M')
            _badge(ov, ax0 + 14, y0 + 0.15 * (y1 - y0), L)
        for side, (ax0, ax1, _, _) in cols.items():
            _xlabel(fig, (ax0 + ax1) / 2, 338)
        _letter(fig, 392, 18, 'b')

        # --- c: S ---
        y0, y1 = 430, 550
        ax = fig.add_axes(_box(fig, 62, y0, 162, y1))
        hist = fig.add_axes(_box(fig, 165, y0, 180, y1), sharey=ax)
        md, sc, _ = curves['S']
        _draw_block_panel(ax, md, sc, colors['S'], wins['S'], stim=True, hist=hist,
                          ylabel=ylabels['S'])
        _badge(ov, 62 + 14, y0 + 0.15 * (y1 - y0), 'S')
        _xlabel(fig, 112, 572)
        _letter(fig, 2, 410, 'c')

        # --- e: S (stim side only), C and M, shuffle mean removed ---
        amps = _draw_panel_e(fig, ov, curves, wins, colors, y0, y1,
                             {'stim': (485, 595), 'move': (650, 750)}, 572)
        _letter(fig, 425, 410, 'e')

        _place_ylabels(fig)
        savefig_svg_png(fig, odir / 'fig45_Spercell_Call_Mall', dpi=200)
        overlaps = _panel_overlaps(fig, ov)
        plt.close(fig)
    summary['C_vs_M_amp'] = amps
    summary['layout_overlaps'] = overlaps
    (odir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: {'n': v['n'], 'p_mean': v['p_mean'],
                          'p_offset': v['offset_gain']['p_offset'],
                          'p_gain': v['offset_gain']['p_gain']}
                      for k, v in summary.items() if isinstance(v, dict) and 'n' in v},
                     indent=2))
    print('amp', amps)
    print('overlapping axes (with labels):', overlaps or 'none')
    print('wrote', odir)


def _panel_overlaps(fig, ov):
    """Pairs of axes whose full extents (tick labels, axis labels) intersect.

    Each axis is checked against axes outside its own panel unit (main + its
    histogram, which share a y axis by design)."""
    r = fig.canvas.get_renderer()
    axes = [a for a in fig.axes if a is not ov and a.get_label() != '<inset>']
    axes = [a for a in axes if not any(a is c for p in fig.axes for c in p.child_axes)]
    ext = [a.get_tightbbox(r) for a in axes]
    shared = lambda a, b: a.get_shared_y_axes().joined(a, b)
    out = []
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            if shared(axes[i], axes[j]):
                continue
            bi, bj = ext[i], ext[j]
            w = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
            h = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
            if w > 0.5 and h > 0.5:
                out.append([i, j, round(float(w), 1), round(float(h), 1)])
    return out


def _draw_schematic(ax):
    """Panel d, black line drawing. Marks and fills are left for later."""
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(0.2, 4.2)
    ax.axis('off')
    nodes = [(1.1, 1.7, 'P'), (4.0, 1.7, 'S'), (6.7, 1.7, 'C'), (9.2, 1.7, 'M')]
    for x, y, lab in nodes:
        ax.add_patch(plt.Circle((x, y), 0.52, fc='white', ec='k', lw=1.0, zorder=3))
        ax.text(x, y, lab, ha='center', va='center', fontsize=10, color='k', zorder=4)
    for x0, x1 in ((1.7, 3.4), (4.6, 6.1), (7.3, 8.6)):
        ax.annotate('', xy=(x1, 1.55), xytext=(x0, 1.55),
                    arrowprops=dict(arrowstyle='-|>', color='k', lw=1.0,
                                    mutation_scale=8))
    ax.text(2.55, 1.15, 'stim', ha='center', fontsize=7, color='k')
    for x0, x1 in ((1.1, 4.0), (4.0, 6.7), (6.7, 9.2)):
        ax.annotate('', xy=(x1 - 0.15, 2.35), xytext=(x0 + 0.15, 2.35),
                    arrowprops=dict(arrowstyle='-', color='k', lw=0.7,
                                    connectionstyle='arc3,rad=-0.35'))


def cmd_cluster_types(args):
    """Split integrator / move_all into one type per rastermap cluster."""
    raw = np.load(NEW_STACK, allow_pickle=True).flat[0]
    rm = dict(zip(raw['uuids'], raw['rm_labels']))
    del raw
    _, ct_raw = load_cell_types(args.cell_types or CELL_TYPES)
    types = {}
    for src in ('integrator', 'move_all'):
        v = ct_raw['types'][src]
        lab = np.array([rm[u] for u in v['uuids']])
        for c in np.unique(lab):
            m = lab == c
            types[f'rm{c}'] = {k: v[k][m] for k in ('uuids', 'pid', 'ids', 'cluster_id')}
    missing = set(CL_TYPES) - set(types)
    if missing:
        raise KeyError(f'clusters without cells: {sorted(missing)}')
    out = {'source': str(NEW_STACK), 'picks': CM_CLUSTERS,
           'types': {t: types[t] for t in CL_TYPES}, 'insertions': ct_raw['insertions']}
    save_atomic(out, CL_CELL_TYPES)
    for t in CL_TYPES:
        print(t, len(out['types'][t]['uuids']), 'cells',
              len(set(out['types'][t]['pid'])), 'insertions')
    print('wrote', CL_CELL_TYPES)


def _choice_trace(r, win):
    """Distance minus shuffle mean and the shuffle 2.5-97.5 % band around it."""
    from fig45_plot_distances import distance_stats
    keys, md, sc = population_traces(r)
    cm = sc.mean(0)
    lo, hi = np.percentile(sc - cm, [2.5, 97.5], axis=0)
    return dict(x=time_axis_ms(win, len(md)), y=md - cm, lo=lo, hi=hi,
                n=int(np.median([r['n'][k] for k in keys])),
                p_mean=distance_stats(md, sc)['p_mean'])


def _choice_axes(fig, curves, colors, kind='choice'):
    """Two panels (stim, move) of ``kind`` (choice or prior); each curve gets
    its shuffle band."""
    axs = [fig.add_axes(_box(fig, *b)) for b in ((70, 40, 235, 170), (300, 40, 465, 170))]
    for ax, name in zip(axs, (f'{kind}_stim', f'{kind}_move')):
        stim = name.endswith('_stim')
        for key, tr in curves[name].items():
            ax.fill_between(tr['x'], tr['lo'], tr['hi'], color=colors[key], alpha=0.18, lw=0)
            ax.plot(tr['x'], tr['y'], color=colors[key], lw=1.5)
        ax.axhline(0, color='k', lw=0.5)
        ax.axvline(0, color='#57C1EB' if stim else '#e74c3c', lw=1.4)
        ys = np.concatenate([np.r_[t['y'], t['lo'], t['hi']] for t in curves[name].values()])
        lo, hi = ys.min(), ys.max()
        pad = 0.04 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        ax.set_ylim(lo, lo + (hi - lo) / (1 - INSET_BAND))
        tag = 's' if stim else 'm'
        _style_ax(ax, _data_ticks(lo, hi), rf'$d^{{\mathrm{{{kind}}},{tag}}}(t)$')
        if stim:
            ax.set_xticks([0, 50, 100, 150])
        else:
            ax.set_xticks([-150, -100, -50, 0])
            ax.set_xticklabels(['\u2212150', '\u2212100', '\u221250', '0'])
    for ax in axs:
        pos = ax.get_position()
        _xlabel(fig, (pos.x0 + pos.x1) / 2 * _page(fig)[0], 192)
    return axs


def cmd_choice_clusters(args):
    """Choice sensitivity (choice L vs R within stim x act block; stim 0-150 ms,
    move -150-0 ms), distance minus shuffle mean: C vs M groups from stage A,
    then one figure per C / M rastermap cluster from stage CL."""
    odir = args.fig_dir or args.out / 'choice_clusters'
    odir.mkdir(parents=True, exist_ok=True)
    wins = {n: ANALYSES[n]['win'] for n in ('choice_stim', 'choice_move')}
    size = (490 / 72, 215 / 72)
    summary = {}
    with plt.rc_context(FIG4_RC):
        pop_a = np.load(stage_dir(args.out, 'A') / 'population.npy', allow_pickle=True).item()
        curves = {n: {L: _choice_trace(pop_a[n][t], wins[n])
                      for L, t in (('C', 'integrator'), ('M', 'move_all'))} for n in wins}
        fig = plt.figure(figsize=size)
        fig.patch.set_facecolor('white')
        ov = _overlay(fig)
        _choice_axes(fig, curves, {'C': '#f4d03f', 'M': '#e07a5f'})
        _badge(ov, 84, 58, 'C')
        _badge(ov, 106, 58, 'M')
        _place_ylabels(fig)
        savefig_svg_png(fig, odir / 'choice_C_vs_M', dpi=200)
        plt.close(fig)
        summary['groups'] = {n: {L: {'n_median': v['n'], 'p_mean': v['p_mean']}
                                 for L, v in c.items()} for n, c in curves.items()}
        f_cl = stage_dir(args.out, 'CL') / 'population.npy'
        if not f_cl.exists():
            print(f'{f_cl} missing: per-cluster figures skipped')
        else:
            pop = np.load(f_cl, allow_pickle=True).item()
            cdir = odir / 'clusters'
            for t in CL_TYPES:
                c = int(t[2:])
                L = 'C' if c in CM_CLUSTERS['C'] else 'M'
                if not all(pop.get(n, {}).get(t, {}).get('n') for n in wins):
                    print(f'{t}: no valid pairs, skipped')
                    continue
                cc = {n: {L: _choice_trace(pop[n][t], wins[n])} for n in wins}
                fig = plt.figure(figsize=size)
                fig.patch.set_facecolor('white')
                ov = _overlay(fig)
                _choice_axes(fig, cc, {L: '#f4d03f' if L == 'C' else '#e07a5f'})
                _badge(ov, 84, 58, L)
                fig.text(0.5, 1 - 14 / 215, f'cluster {c}', fontsize=12,
                         ha='center', va='center')
                _place_ylabels(fig)
                savefig_svg_png(fig, cdir / f'choice_cluster{c:03d}_{L}', dpi=200)
                plt.close(fig)
                summary.setdefault('clusters', {})[t] = {
                    'type': L, **{n: {'n_median': cc[n][L]['n'],
                                      'p_mean': cc[n][L]['p_mean']} for n in wins}}
    (odir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary.get('groups'), indent=2))
    print(len(summary.get('clusters', {})), 'cluster figures; wrote', odir)


def cmd_select_cm(args):
    """C / M cells with choice sensitivity, Figure 2 style but per cell."""
    sdir = stage_dir(args.out, 'A')
    _, ct_raw = load_cell_types(args.cell_types or CELL_TYPES)
    cs = np.load(sdir / 'cells_choice_stim.npy', allow_pickle=True).item()
    cm = np.load(sdir / 'cells_choice_move.npy', allow_pickle=True).item()
    ps, pm = dict(zip(cs['uuids'], cs['p'])), dict(zip(cm['uuids'], cm['p']))
    sel, info = {}, {'rule': f'uncorrected choice p < {args.p}'}
    for name, src, wins in (('C_sel', 'integrator', (ps, pm)), ('M_sel', 'move_all', (pm,))):
        uu = [u for u in ct_raw['types'][src]['uuids'] if all(u in w for w in wins)]
        hit = [u for u in uu if any(w[u] < args.p for w in wins)]
        sel[name] = set(hit)
        info[name] = {'from': src, 'windows': ['stim', 'move'] if len(wins) == 2 else ['move'],
                      'tested': len(uu), 'selected': len(hit),
                      'expected_by_chance': round(len(uu) * (1 - (1 - args.p) ** len(wins)), 1)}
    out = {'source': str(args.cell_types or CELL_TYPES), 'info': info,
           'types': subset_types(ct_raw, sel), 'insertions': ct_raw['insertions']}
    tag = f'{args.p:g}'.replace('0.', 'p')
    dst = args.dst or TYPES_DIR / f'cell_types_cm_choice_{tag}.npy'
    save_atomic(out, dst)
    print(json.dumps(info, indent=2))
    print('wrote', dst)


def _shape(md, win):
    """Figure 2 curve shape (analysis_functions.compute_amp_slope) on the raw
    distance: end peak = last 5-bin window above every earlier one; late slope
    = per-bin slope of the last 20 bins; overall slope = per-s fit over the window."""
    w5 = np.convolve(md, np.ones(5) / 5, mode='valid')
    x = time_axis_ms(win, len(md)) / 1000
    return {'end_peak': bool(w5[-1] > w5[:-1].max()),
            'slope_last20': float(np.polyfit(np.arange(20), md[-20:], 1)[0]),
            'slope_overall_per_s': float(np.polyfit(x, md, 1)[0]),
            'peak_t_ms': float(time_axis_ms(win, len(md))[int(np.argmax(md))])}


def cmd_choice_selected(args):
    """Choice and prior sensitivity of C_sel / M_sel (stage CS), next to C all /
    M all (stage A): overlaid curves; end peak / slopes of each raw choice
    curve; prior offset / gain as in Figure 4.5 (move offset from stim)."""
    odir = args.fig_dir or args.out / 'choice_selected'
    odir.mkdir(parents=True, exist_ok=True)
    wins = {n: ANALYSES[n]['win'] for n in ('choice_stim', 'choice_move')}
    pops = {s: np.load(stage_dir(args.out, s) / 'population.npy', allow_pickle=True).item()
            for s in ('A', 'CS')}
    sets = {'C_sel': ('CS', 'C_sel'), 'M_sel': ('CS', 'M_sel'),
            'C_all': ('A', 'integrator'), 'M_all': ('A', 'move_all')}
    summary = {}
    for key, (stage, t) in sets.items():
        for n, win in wins.items():
            r = pops[stage].get(n, {}).get(t)
            if not r or not r['n']:
                print(f'{key} {n}: no valid pairs, skipped')
                continue
            keys, md, sc = population_traces(r)
            summary.setdefault(key, {})[n] = {
                'n_median': int(np.median([r['n'][k] for k in keys])), **_shape(md, win)}
        for n in ('prior_stim', 'prior_move'):
            r = pops[stage].get(n, {}).get(t)
            if not r or not r['n']:
                print(f'{key} {n}: no valid pairs, skipped')
                continue
            keys, md, sc = population_traces(r)
            off = None
            if n == 'prior_move' and pops[stage].get('prior_stim', {}).get(t, {}).get('n'):
                off = population_traces(pops[stage]['prior_stim'][t])[1:]
            og = prior_offset_gain(md, sc, ANALYSES[n]['win'], offset_from=off)
            summary.setdefault(key, {})[n] = {
                'n_median': int(np.median([r['n'][k] for k in keys])),
                **{k: og[k] for k in ('p_offset', 'offset', 'p_gain', 'gain')}}
    with plt.rc_context(FIG4_RC):
        curves = {n: {L: _choice_trace(pops['CS'][n][f'{L}_sel'], wins[n]) for L in 'CM'
                      if pops['CS'][n].get(f'{L}_sel', {}).get('n')} for n in wins}
        fig = plt.figure(figsize=(490 / 72, 215 / 72))
        fig.patch.set_facecolor('white')
        ov = _overlay(fig)
        _choice_axes(fig, curves, {'C': '#f4d03f', 'M': '#e07a5f'})
        _badge(ov, 84, 58, 'C')
        _badge(ov, 106, 58, 'M')
        _place_ylabels(fig)
        savefig_svg_png(fig, odir / 'choice_Csel_vs_Msel', dpi=200)
        plt.close(fig)
        pwins = {n: ANALYSES[n]['win'] for n in ('prior_stim', 'prior_move')}
        curves = {n: {L: _choice_trace(pops['CS'][n][f'{L}_sel'], pwins[n]) for L in 'CM'
                      if pops['CS'].get(n, {}).get(f'{L}_sel', {}).get('n')} for n in pwins}
        fig = plt.figure(figsize=(490 / 72, 215 / 72))
        fig.patch.set_facecolor('white')
        ov = _overlay(fig)
        _choice_axes(fig, curves, {'C': '#f4d03f', 'M': '#e07a5f'}, kind='prior')
        _badge(ov, 84, 58, 'C')
        _badge(ov, 106, 58, 'M')
        _place_ylabels(fig)
        savefig_svg_png(fig, odir / 'prior_Csel_vs_Msel', dpi=200)
        plt.close(fig)
    (odir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'{"set":6s} {"window":12s} {"n":>6s} {"end_peak":>8s} {"slope20":>9s} '
          f'{"slope/s":>8s} {"peak_ms":>8s}')
    for key, v in summary.items():
        for n, r in v.items():
            if n.startswith('choice'):
                print(f'{key:6s} {n:12s} {r["n_median"]:6d} {str(r["end_peak"]):>8s} '
                      f'{r["slope_last20"]:9.4f} {r["slope_overall_per_s"]:8.3f} '
                      f'{r["peak_t_ms"]:8.1f}')
    print(f'{"set":6s} {"window":12s} {"n":>6s} {"offset":>8s} {"p_off":>7s} {"gain":>8s} '
          f'{"p_gain":>7s}')
    for key, v in summary.items():
        for n, r in v.items():
            if n.startswith('prior'):
                print(f'{key:6s} {n:12s} {r["n_median"]:6d} {r["offset"]:8.4f} '
                      f'{r["p_offset"]:7.4f} {r["gain"]:8.4f} {r["p_gain"]:7.4f}')
    print('wrote', odir)


def cmd_panel_e_sel(args):
    """Figure 4.5 e with the choice-selected C_sel / M_sel (stage CS) for C / M.
    S is the stage BS (else B) S_cell curve when <out>/stageBS or stageB exists."""
    pops = {s: np.load(stage_dir(args.out, s) / 'population.npy', allow_pickle=True).item()
            for s in STAGES if (stage_dir(args.out, s) / 'population.npy').exists()}
    curves = {f'{L}_{side}': _block_curve(pops, 'CS', f'prior_{side}', f'{L}_sel')
              for L in 'CM' for side in ('stim', 'move')}
    s_stage = 'BS' if 'BS' in pops else ('B' if 'B' in pops else None)
    if s_stage:
        curves['S'] = _block_curve(pops, s_stage, 'S_prior', 'S_cell')
    wins = {'S': ANALYSES['S_prior']['win'],
            **{f'{L}_{side}': ANALYSES[f'prior_{side}']['win']
               for L in 'CM' for side in ('stim', 'move')}}
    colors = {'S': '#5BA4D9', 'C_stim': '#f4d03f', 'C_move': '#f4d03f',
              'M_stim': '#e07a5f', 'M_move': '#e07a5f'}
    odir = args.fig_dir or args.out / 'fig45'
    with plt.rc_context(FIG4_RC):
        fig = plt.figure(figsize=(320 / 72, 190 / 72))
        fig.patch.set_facecolor('white')
        ov = _overlay(fig)
        amps = _draw_panel_e(fig, ov, curves, wins, colors, 20, 140,
                             {'stim': (55, 165), 'move': (215, 315)}, 162)
        _place_ylabels(fig)
        savefig_svg_png(fig, odir / 'panel_e_Csel_Msel', dpi=200)
        overlaps = _panel_overlaps(fig, ov)
        plt.close(fig)
    out = {'S_stage': s_stage, 'n': {k: v[2] for k, v in curves.items()}, 'amp': amps,
           'layout_overlaps': overlaps}
    (odir / 'panel_e_Csel_Msel.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print('wrote', odir)


def cmd_rm_types(args):
    """One type per rastermap cluster, all of rm_labels (stage RM)."""
    from fig45_define_cell_types import cluster_ids_and_insertions
    raw = np.load(NEW_STACK, allow_pickle=True).flat[0]
    new = {k: raw[k] for k in ('uuids', 'pid', 'ids', 'rm_labels')}
    del raw
    cids, ins = cluster_ids_and_insertions(new['uuids'], new['pid'])
    types = {}
    for t in RM_TYPES:
        m = new['rm_labels'] == int(t[2:])
        if not m.any():
            raise KeyError(f'{t} has no cells')
        types[t] = {'uuids': new['uuids'][m], 'pid': new['pid'][m], 'ids': new['ids'][m],
                    'cluster_id': cids[m]}
    out = {'source': str(NEW_STACK), 'picks': {t: [int(t[2:])] for t in RM_TYPES},
           'types': types, 'insertions': ins}
    save_atomic(out, RM_CELL_TYPES)
    print(len(new['uuids']), 'cells,', len(ins), 'insertions,', len(types), 'clusters')
    print('wrote', RM_CELL_TYPES)


# Figure 2 / 4 region rules (build_standard_figs, sensory_responding_areas)
RM_ALPHA = 0.01          # BH over the 100 clusters, per analysis
EARLY_CHOICE_MS = 80     # S: no run of choice bins (uncorrected p <= RM_ALPHA) ...
EARLY_CHOICE_RUN = 5     # ... this long in the first 80 ms of choice_stim
SLOPE_CUT_FIXED = 0.05   # get_sc_table's late-slope cut (regional distance units)
MOVE_RULES = ('late_mean', 'late_fixed', 'end_peak')


def bh_adjusted(p):
    from statsmodels.stats.multitest import multipletests
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        q[ok] = multipletests(p[ok], method='fdr_bh')[1]
    return q


def _longest_run(sig):
    best = cur = 0
    for s in sig:
        cur = cur + 1 if s else 0
        best = max(best, cur)
    return best


def cmd_select_rm(args):
    """S / C / M clusters from stage RM, each cluster tested like a region.

    S: S_stim (0-80 ms, stim L vs R, act-prior strata) BH <= 0.01; not an
    onset-offset confound (offset p <= 0.01 and no gain after subtracting it);
    no early choice (a run of >= 5 choice_stim bins with p <= 0.01 at t <= 80 ms).
    Choice: BH <= 0.01 per window (choice_stim -> stim row, choice_move -> move row).
    Movement: move-row choice and a movement shape on the move-aligned choice
    curve: end peak and late slope > the mean over choice clusters (late_mean,
    res_standard default), > 0.05 (late_fixed, old rule), or end peak only.
    Integrator, per row: choice on that row, rising (overall slope > 0), not movement.
    C = integrator on either row, M = movement. Prior: BH <= 0.01 per window, with
    offset / gain as in Figure 4.5 (move offset from the stim curve).
    """
    import csv
    from fig45_plot_distances import distance_stats
    sdir = stage_dir(args.out, 'RM')
    traces = np.load(sdir / 'traces.npy', allow_pickle=True).item()
    odir = sdir / 'select_rm'
    odir.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in RM_TYPES:
        row = {'cluster': int(t[2:]), 'type': t}
        tr = {}
        for name, spec in ANALYSES.items():
            r = traces.get(name, {}).get(t)
            if r is None:
                row[f'{name}_p'] = np.nan
                continue
            md, sc = r['md'].astype(float), r['sc'].astype(float)
            tr[name] = (md, sc)
            st = distance_stats(md, sc, alpha=RM_ALPHA)
            row[f'{name}_p'] = st['p_mean']
            row[f'{name}_n'] = r['n']
            row[f'{name}_amp'] = float((md - sc.mean(0)).mean())
            if name == 'S_stim':
                row['S_p_offset'], row['S_p_gain'] = st['p_offset'], st['p_gain']
        if 'choice_stim' in tr:
            md, sc = tr['choice_stim']
            p_bin = (1 + (sc >= md[None]).sum(0)) / (1 + len(sc))
            early = time_axis_ms(ANALYSES['choice_stim']['win'], len(md)) <= EARLY_CHOICE_MS
            row['early_choice_run'] = _longest_run(p_bin[early] <= RM_ALPHA)
            row['slope_overall_stim'] = _shape(md, ANALYSES['choice_stim']['win'])[
                'slope_overall_per_s']
        if 'choice_move' in tr:
            sh = _shape(tr['choice_move'][0], ANALYSES['choice_move']['win'])
            row.update(end_peak=sh['end_peak'], slope_last20=sh['slope_last20'],
                       slope_overall_move=sh['slope_overall_per_s'])
        for pre, ref in (('prior_stim', None), ('prior_move', 'prior_stim'),
                         ('S_prior', None)):
            if pre in tr and (ref is None or ref in tr):
                og = prior_offset_gain(*tr[pre], ANALYSES[pre]['win'],
                                       offset_from=tr[ref] if ref else None)
                row[f'{pre}_p_offset'], row[f'{pre}_p_gain'] = og['p_offset'], og['p_gain']
                row[f'{pre}_offset'], row[f'{pre}_gain'] = og['offset'], og['gain']
        rows.append(row)
    for name in ANALYSES:
        q = bh_adjusted([r[f'{name}_p'] for r in rows])
        for r, v in zip(rows, q):
            r[f'{name}_q'] = float(v)
    get = lambda r, k, d=np.nan: r.get(k, d)
    for r in rows:
        r['s80'] = bool(get(r, 'S_stim_q') <= RM_ALPHA)
        r['onset_confound'] = bool(get(r, 'S_p_offset', 1) <= RM_ALPHA
                                   and get(r, 'S_p_gain', 1) > RM_ALPHA)
        r['early_choice'] = bool(get(r, 'early_choice_run', 0) >= EARLY_CHOICE_RUN)
        r['S'] = r['s80'] and not r['onset_confound'] and not r['early_choice']
        r['choice_s'] = bool(get(r, 'choice_stim_q') <= RM_ALPHA)
        r['choice_m'] = bool(get(r, 'choice_move_q') <= RM_ALPHA)
        for name in ('S_prior', 'prior_stim', 'prior_move'):
            r[f'{name}_sig'] = bool(get(r, f'{name}_q') <= RM_ALPHA)
    choice = [r for r in rows if r['choice_s'] or r['choice_m']]
    cut_late = float(np.nanmean([get(r, 'slope_last20') for r in choice])) if choice else np.nan
    cuts = {'late_mean': cut_late, 'late_fixed': SLOPE_CUT_FIXED, 'end_peak': -np.inf}
    for rule, cut in cuts.items():
        for r in rows:
            shape = bool(get(r, 'end_peak', False)) and get(r, 'slope_last20', -np.inf) > cut
            move = shape and r['choice_m']
            integ_s = (r['choice_s'] and get(r, 'slope_overall_stim', 0) > 0) and not move
            integ_m = (r['choice_m'] and get(r, 'slope_overall_move', 0) > 0) and not move
            r[f'M_{rule}'] = move
            r[f'C_{rule}'] = integ_s or integ_m
            r[f'C_stimrow_{rule}'] = integ_s
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(odir / 'clusters.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    sel = lambda k: [r['cluster'] for r in rows if r.get(k)]
    summary = {
        'alpha_bh': RM_ALPHA, 'n_clusters': len(rows),
        'move_slope_cuts': cuts | {'end_peak': None},
        'S': sel('S'), 's80_before_cuts': sel('s80'),
        'onset_confound': [r['cluster'] for r in rows if r['s80'] and r['onset_confound']],
        'early_choice': [r['cluster'] for r in rows if r['s80'] and r['early_choice']],
        'choice_stim': sel('choice_s'), 'choice_move': sel('choice_m'),
        **{f'{g}_{rule}': sel(f'{g}_{rule}') for rule in MOVE_RULES for g in ('C', 'M')},
        **{f'{n}_sig': sel(f'{n}_sig') for n in ('S_prior', 'prior_stim', 'prior_move')},
    }
    (odir / 'selection.json').write_text(json.dumps(summary, indent=1))
    for k, v in summary.items():
        if isinstance(v, list):
            print(f'{k:22s} {len(v):3d}  {v}')
    print('movement slope cuts', cuts)
    print('wrote', odir)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name, fn in (('compute', cmd_compute), ('stack', cmd_stack), ('select', cmd_select),
                     ('compare', cmd_compare), ('fig45', cmd_fig45),
                     ('cluster-types', cmd_cluster_types),
                     ('choice-clusters', cmd_choice_clusters),
                     ('select-cm', cmd_select_cm),
                     ('choice-selected', cmd_choice_selected),
                     ('panel-e-sel', cmd_panel_e_sel),
                     ('rm-types', cmd_rm_types), ('select-rm', cmd_select_rm)):
        p = sub.add_parser(name)
        p.add_argument('--out', type=Path, default=OUT)
        p.add_argument('--cell-types', type=Path, default=None)
        p.set_defaults(func=fn)
        if name in ('compute', 'stack'):
            p.add_argument('--stage', choices=list(STAGES), required=True)
        if name == 'select':
            p.add_argument('--q', type=float, default=Q_FDR)
            p.add_argument('--s-cell-p', type=float, default=None,
                           help='S_cell = uncorrected S_stim p below this (no BH)')
        if name == 'fig45':
            p.add_argument('--fig-dir', type=Path, default=PAPER_FIGS / 'fig45')
        if name == 'select-cm':
            p.add_argument('--p', type=float, default=0.01)
            p.add_argument('--dst', type=Path, default=None)
        if name == 'panel-e-sel':
            p.add_argument('--fig-dir', type=Path, default=None, help='default <out>/fig45')
        if name == 'choice-selected':
            p.add_argument('--fig-dir', type=Path, default=None,
                           help='default <out>/choice_selected')
        if name == 'choice-clusters':
            p.add_argument('--fig-dir', type=Path, default=None,
                           help='default <out>/choice_clusters')
        if name == 'compute':
            p.add_argument('--shard', type=int, default=0)
            p.add_argument('--n-shards', type=int, default=1)
            p.add_argument('--nrand', type=int, default=NRAND)
            p.add_argument('--seed', type=int, default=0)
            p.add_argument('--prior', choices=['act', 'true'], default='act')
            p.add_argument('--one-cache-dir', type=Path,
                           default=Path(os.environ.get('ONE_CACHE_DIR', DEFAULT_ONE_CACHE)))
            p.add_argument('--base-url', default=os.environ.get(
                'ONE_BASE_URL', 'https://openalyx.internationalbrainlab.org'))
            p.add_argument('--source', choices=['auto', 'cache', 'one'], default='auto')
            p.add_argument('--overwrite', action='store_true')
            p.add_argument('--pids', nargs='+', default=None, help='restrict (testing)')
    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
