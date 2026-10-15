"""Figure 4.5 follow-up on concat_cvTrue: stimulus / choice / act-prior
sensitivity of rastermap cell types, each against a label shuffle inside the
strata of the other factors.

Labels: stim side (contrastLeft/Right not NaN), choice (+-1), act-kernel block
(alpha 0.2 over the full choice sequence, binarised 0.8 / 0.2, as in
fig45_block_peths). Trials: BWM base mask + the six Fig 4.5 saturation keys.

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
  select            S_cell  = BH(q) over pool cells on S_stim
                    S_clus  = clusters with BH(q) population S_stim p_mean
                    C_choice / M_choice = cells with BH(q) choice p (stim or
                    move window, BH over both windows jointly)
                    -> <out>/stageB_cell_types.npy
  stage B  compute  S_stim / S_prior on S_cell, S_clus; choice_* / prior_* on
                    C_choice, M_choice (populations only)
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
from fig45_common import (CELL_TYPES, COLORS, NRAND, OUT_ROOT, PAPER_FIGS, STS, T_BIN,
                          savefig_svg_png)

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
# stage -> analysis -> (population types, per-cell test pool or None)
STAGES = {
    'A': {'S_stim': (['stimint'] + [f'rm{c}' for c in S_POOL_CLUSTERS], ['stimint']),
          'S_prior': (['stimint'], None),
          'choice_stim': (CM, CM), 'choice_move': (CM, CM),
          'prior_stim': (CM, None), 'prior_move': (CM, None)},
    'B': {'S_stim': (S_SEL, None), 'S_prior': (S_SEL, None),
          'choice_stim': (CM_SEL, None), 'choice_move': (CM_SEL, None),
          'prior_stim': (CM_SEL, None), 'prior_move': (CM_SEL, None)},
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


def compute_insertion(loaded, cell_types, stage, nrand, rng, br):
    src, eid, trials, mask, prior_l, spikes, clusters, in_any, uu_in = loaded
    trials = trials.copy()
    if prior_l is not None:
        trials['probabilityLeft'] = prior_l
    fm = factor_masks(trials)
    acr = np.array(br.id2acronym(np.asarray(clusters['atlas_id'])[in_any], mapping='Beryl'))
    sp = typed_spikes(spikes, clusters, in_any)
    tmask = {t: np.isin(uu_in, list(cell_types.get(t, ()))) for t in stage_types(stage)}
    pop, cells = {}, {}
    for name in STAGES[stage]:
        spec = ANALYSES[name]
        ev_all = trials[ALIGN_EVENT[spec['align']]].to_numpy()
        types = analysis_types(stage, name)
        pool = cell_pool(stage, name)
        pool_m = np.logical_or.reduce([tmask[t] for t in pool]) if pool else None
        res = {t: {'n': {}, 'dist': {}, 'ctrl': {}} for t in types}
        ntr = {}
        stat = np.zeros(len(uu_in))
        null = np.zeros((nrand, len(uu_in)), dtype=np.float32) if pool else None
        n_ok = 0
        for a, b, _ in pairs_of(spec):
            key = pair_key(a, b)
            ma = cond_mask(fm, mask, spec['align'], a)
            mb = cond_mask(fm, mask, spec['align'], b)
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
    ct_path = args.cell_types or (CELL_TYPES if args.stage == 'A'
                                  else args.out / 'stageB_cell_types.npy')
    cell_types, ct_raw = load_cell_types(ct_path)
    types = [t for t in stage_types(args.stage) if t in ct_raw['types']]
    missing = set(stage_types(args.stage)) - set(types)
    if missing:
        raise KeyError(f'{ct_path} lacks types {sorted(missing)}')
    pids = insertion_list(ct_raw, types)
    if args.pids:
        pids = [p for p in pids if p in set(args.pids)]
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
                                     ord(args.stage)])
        try:
            loaded = load(pid, cell_types, types)
            if loaded[7].any():
                pop, cells = compute_insertion(loaded, cell_types, args.stage, args.nrand,
                                               rng, br)
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
            for s in STAGES}
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
    sig = bh(c['p'], args.q)
    sel['S_cell'] = set(c['uuids'][sig])
    info['S_cell'] = {'pool': len(c['uuids']), 'selected': int(sig.sum())}
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


def _badge(ax, letter, color, xy=(0.1, 0.9)):
    ax.scatter(*xy, s=220, c=[color], transform=ax.transAxes, zorder=6,
               clip_on=False, edgecolors='none')
    ax.text(*xy, letter, transform=ax.transAxes, ha='center', va='center',
            fontsize=8, zorder=7, clip_on=False)


def _stars(hist, rows):
    """rows: (y, sig). A star sits to the right of a significant reference line."""
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(hist.transAxes, hist.transData)
    for y, sig in rows:
        if sig:
            hist.text(1.08, y, '*', transform=trans, ha='left', va='center',
                      fontsize=11, clip_on=False)


def _draw_block_panel(ax, md, sc, color, win, stim=True, hist=None,
                      badge=None, inset_loc='upper left'):
    """Fig 4.5 distance panel. The blue offset mark is the first 5 samples."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
    if st['offset'] != 0.0:
        ax.plot(x, st['shifted'], color=color, lw=1.5)
    ins = inset_axes(ax, width='32%', height='30%', loc=inset_loc, borderpad=0.6)
    cm = sc.mean(0)
    lo, hi = np.percentile(sc - cm, [2.5, 97.5], axis=0)
    ins.fill_between(x, lo, hi, color='0.75', alpha=0.7, lw=0)
    ins.plot(x, md - cm, color=color, lw=0.8)
    ins.axhline(0, color='k', lw=0.4)
    ins.set_xticks([])
    ins.set_yticks([])
    for s in ins.spines.values():
        s.set_visible(False)
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
                  ([0, 40, 80, 120] if stim else [0, -40, -80, -120]))
    ax.tick_params(labelsize=10)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if badge:
        _badge(ax, *badge)
    return st


def _panel_letter(ax, letter):
    ax.text(0.0, 1.01, letter, transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='bottom', ha='left', clip_on=False)


def _cap_ticks(ax):
    """At most five labeled ticks, at the original's size."""
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    ax.tick_params(axis='both', labelsize=10)


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


def cmd_fig45(args):
    """Figure 4.5 in the published layout: a raster, b C/M 2x2, c S, d schematic, e C vs M.

    C and M are the act-block distance within stim x choice. S per cell is its
    act-block distance within stim side, 0-80 ms.
    """
    import matplotlib.image as mpimg
    from fig45_plot_distances import distance_stats
    pops = {s: np.load(stage_dir(args.out, s) / 'population.npy', allow_pickle=True).item()
            for s in STAGES}
    spec = {name: ANALYSES[name] for name in ('S_prior', 'prior_stim', 'prior_move')}
    curves = {
        'S': _block_curve(pops, 'B', 'S_prior', 'S_cell'),
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
    badges = {'S': ('S', '#5BA4D9'), 'C_stim': ('C', '#f4d03f'), 'C_move': ('C', '#f4d03f'),
              'M_stim': ('M', '#e07a5f'), 'M_move': ('M', '#e07a5f')}
    ylabels = {
        'S': r'$d^{\mathrm{prior},s}_{S}(t)$',
        'C_stim': r'$d^{\mathrm{prior},s}_{C}(t)$',
        'C_move': r'$d^{\mathrm{prior},m}_{C}(t)$',
        'M_stim': r'$d^{\mathrm{prior},s}_{M}(t)$',
        'M_move': r'$d^{\mathrm{prior},m}_{M}(t)$',
    }
    insets = {'S': 'lower right', 'C_stim': 'center right', 'C_move': 'lower left',
              'M_stim': 'center right', 'M_move': 'upper left'}
    summary = {k: {'n': n, 'p_mean': distance_stats(md, sc)['p_mean'], 'offset_gain': o}
               for k, ((md, sc, n), o) in ((k, (curves[k], og[k])) for k in curves)}

    fig = plt.figure(figsize=(13.2, 7.3))
    grid = fig.add_gridspec(2, 1, height_ratios=[2.2, 1], hspace=0.38,
                            left=0.02, right=0.99, top=0.96, bottom=0.06)
    upper = grid[0].subgridspec(1, 2, width_ratios=[1.05, 1.05], wspace=0.16)
    lower = grid[1].subgridspec(1, 3, width_ratios=[1.0, 2.2, 1.65], wspace=0.22)

    ax_a = fig.add_subplot(upper[0])
    raster = mpimg.imread(OUT_ROOT / 'cvTrue' / 'plots' / 'panel_a_raster_test.png')
    rgb = raster[..., :3] if raster.ndim == 3 else raster
    ink = (rgb < 0.98).any(2) if rgb.ndim == 3 else rgb < 0.98
    ys, xs = ink.any(1), ink.any(0)
    y0, y1 = int(ys.argmax()), int(len(ys) - ys[::-1].argmax())
    x0, x1 = int(xs.argmax()), int(len(xs) - xs[::-1].argmax())
    ax_a.imshow(raster[y0:y1, x0:x1], aspect='auto')
    ax_a.axis('off')
    _panel_letter(ax_a, 'a')

    block = upper[1].subgridspec(2, 2, hspace=0.42, wspace=0.42)
    slots = {'C_stim': block[0, 0], 'C_move': block[0, 1],
             'M_stim': block[1, 0], 'M_move': block[1, 1]}
    first_b = None
    for key, slot in slots.items():
        inner = slot.subgridspec(1, 2, width_ratios=[5.2, 0.9], wspace=0.12)
        ax = fig.add_subplot(inner[0])
        hist = fig.add_subplot(inner[1], sharey=ax)
        md, sc, _ = curves[key]
        _draw_block_panel(ax, md, sc, colors[key], wins[key],
                          stim=not key.endswith('move'), hist=hist, badge=badges[key],
                          inset_loc=insets[key])
        ax.set_ylabel(ylabels[key], fontsize=10)
        ax.set_xlabel('time (ms)', fontsize=10)
        _cap_ticks(ax)
        if first_b is None:
            first_b = ax
    _panel_letter(first_b, 'b')
    apos = ax_a.get_position()
    bpos = first_b.get_position()
    # Panel c's ylabel sticks out of the figure and bbox-tight then pads a.
    # Start a left of that so the raster sits on the left edge, short of b.
    right = bpos.x0 - 0.14
    x0 = -0.03
    ax_a.set_position([x0, apos.y0, right - x0, apos.height])

    inner = lower[0].subgridspec(1, 2, width_ratios=[5.2, 0.9], wspace=0.12)
    ax_c = fig.add_subplot(inner[0])
    hist_c = fig.add_subplot(inner[1], sharey=ax_c)
    md, sc, _ = curves['S']
    _draw_block_panel(ax_c, md, sc, colors['S'], wins['S'], stim=True,
                      hist=hist_c, badge=badges['S'], inset_loc=insets['S'])
    ax_c.set_ylabel(ylabels['S'], fontsize=10)
    ax_c.set_xlabel('time (ms)', fontsize=10)
    _cap_ticks(ax_c)
    _panel_letter(ax_c, 'c')

    egs = lower[2].subgridspec(1, 2, wspace=0.28)
    amps = {}
    ax_e0 = None
    for i, (side, akey, bkey) in enumerate((('stim', 'C_stim', 'M_stim'),
                                            ('move', 'C_move', 'M_move'))):
        ax = fig.add_subplot(egs[i])
        if ax_e0 is None:
            ax_e0 = ax
        for key in (akey, bkey):
            md, sc, _ = curves[key]
            tr = md - sc.mean(0)
            x = time_axis_ms(wins[key], len(tr))
            ax.plot(x, tr, color=colors[key], lw=1.3)
            amps.setdefault(side, {})[key[0]] = float(tr.max() - tr.min())
        amps[side]['M_over_C'] = amps[side]['M'] / amps[side]['C']
        ax.axvline(0, color='#57C1EB' if side == 'stim' else '#e74c3c', lw=1.4)
        ax.set_xlabel('time (ms)', fontsize=10)
        ax.set_xticks([0, 40, 80, 120] if side == 'stim' else [0, -40, -80, -120])
        _cap_ticks(ax)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        tag = 's' if side == 'stim' else 'm'
        ax.set_ylabel(rf'$d^{{\mathrm{{prior}},{tag}}}_{{\{{C,M\}}}}(t)$', fontsize=10)
    _badge(ax_e0, 'C', '#f4d03f', (0.16, 0.88))
    _badge(ax_e0, 'M', '#e07a5f', (0.30, 0.88))
    _panel_letter(ax_e0, 'e')

    savefig_svg_png(fig, odir / 'fig45_Spercell_Call_Mall', dpi=160)
    plt.close(fig)
    summary['C_vs_M_amp'] = amps
    (odir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: {'n': v['n'], 'p_mean': v['p_mean'],
                          'p_offset': v['offset_gain']['p_offset'],
                          'p_gain': v['offset_gain']['p_gain']}
                      for k, v in summary.items() if k != 'C_vs_M_amp'}, indent=2))
    print('amp', amps)
    print('wrote', odir)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name, fn in (('compute', cmd_compute), ('stack', cmd_stack), ('select', cmd_select),
                     ('compare', cmd_compare), ('fig45', cmd_fig45)):
        p = sub.add_parser(name)
        p.add_argument('--out', type=Path, default=OUT)
        p.add_argument('--cell-types', type=Path, default=None)
        p.set_defaults(func=fn)
        if name in ('compute', 'stack'):
            p.add_argument('--stage', choices=list(STAGES), required=True)
        if name == 'select':
            p.add_argument('--q', type=float, default=Q_FDR)
        if name == 'fig45':
            p.add_argument('--fig-dir', type=Path, default=PAPER_FIGS / 'fig45')
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
