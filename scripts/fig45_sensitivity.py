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
from fig45_common import COLORS, NRAND, OUT_ROOT, STS, T_BIN, savefig_svg_png

OUT = OUT_ROOT / 'cvTrue' / 'sensitivity'
CELL_TYPES = OUT_ROOT / 'cvTrue' / 'cell_types.npy'
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
                    ax.fill_between(x, lo, hi, color=col, alpha=0.12, lw=0)
                    ax.plot(x, md - cm, color=col, ls=ls, lw=1.8,
                            label=f"{label} ({n} cells/pair) p={st['p_mean']:.4f}")
                    summary.setdefault(name, {})[t] = {
                        'label': label, 'stage': stage, 'n_median': n,
                        **{k: v for k, v in st.items() if k != 'shifted'}}
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
                  f'p_mean={st["p_mean"]:.4f} p_gain={st["p_gain"]:.4f}')
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


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name, fn in (('compute', cmd_compute), ('stack', cmd_stack), ('select', cmd_select),
                     ('compare', cmd_compare)):
        p = sub.add_parser(name)
        p.add_argument('--out', type=Path, default=OUT)
        p.add_argument('--cell-types', type=Path, default=None)
        p.set_defaults(func=fn)
        if name in ('compute', 'stack'):
            p.add_argument('--stage', choices=list(STAGES), required=True)
        if name == 'select':
            p.add_argument('--q', type=float, default=Q_FDR)
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
