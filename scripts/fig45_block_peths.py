"""Figure 4.5 rerun: per-insertion block-pair PETHs + block-label shuffles for
rastermap cell types, then stack into the files the distance plots read.

Re-implements paper-brain-wide-map ``dmn_avgbystim.concat_PETHs`` (vers
``concat_block_pairs_act``) + ``stack_concat(shuffling=True)``, with the
standard-splits trial table:
  - action kernel (alpha=0.2) over the full choice sequence, binarised
    (>=0.5 -> 0.8 else 0.2), then written onto ``probabilityLeft``
  - bad-trial mask, then one saturation key for that alignment
    (stimOn: saturation_stim_plus04; firstMovement: saturation_move_minus02)
  - then drop trials whose true ``probabilityLeft`` is 0.5. The kernel
    has already seen them. The drop uses the saved true block.
  - 16 trial types = stim side x choice x block, stimOn [0, 200] ms and
    firstMovement [-150, 0] ms, 12.5 ms bins at 2 ms stride (6 sub-bins)
  - real PETH = trial-mean spike count per bin
  - null: for each block pair, pool both conditions' trials and permute the
    block label NRAND times (sizes kept)
  - distance = sum over cells of (PETH_a - PETH_b)^2 per bin; controls the
    same on each shuffle; plotted as sqrt(. / T_BIN / n_cells)

Differences from the original (numerically equivalent in expectation):
  - shuffled-mean differences are computed with an assignment matrix
    (one matmul per pair) instead of a Python loop over permutations;
  - only the per-type sum of squared shuffle differences is kept
    (``stack_concat`` discards per-cell shuffles anyway);
  - AUDp cells are dropped from stim_res in both real and control sums
    (``stack_concat`` dropped them from the real sum only);
  - per-cell shuffle p-values (``p_vals``) are not computed.

Subcommands
  compute  --shard i --n-shards n   per-insertion files -> <out>/peths_<vers>/
  stack    --types ...              -> <out>/{vers}_normFalse_shuffleTrue_{type}.npy

Inputs per insertion come from this repo's ``manifold/insertion_cache``
(``block_analysis_allsplits.build_insertion_cache``) + ``bwm_tables/trials.pqt``
under ``--one-cache-dir``; ONE is used only on a cache miss (``--source auto``).
``cell_types.npy`` carries cluster_id and (eid, probe) per typed cell so the
cache path needs no Alyx / clusters.pqt. Cache vs ONE: bit-identical PETHs and
controls (pid 8dfb86c8, nrand=2000).

Usage (iblenv, outside sandbox; ORCD: scripts/submit_fig45_peths_sharded.sh):
  python scripts/fig45_block_peths.py compute --shard 0 --n-shards 1
  python scripts/fig45_block_peths.py stack
  python scripts/fig45_plot_distances.py --root <out> --tag act
"""
import argparse
import gc
import os
import time
import traceback
import uuid
from pathlib import Path

import numpy as np

from fig45_common import (ALIGN_SATURATION, ALPHA_AK, BLOCK_PAIRS, CELL_TYPES, MOVE_WIN,
                          NRAND, OUT_ROOT, STIM_WIN, STS, T_BIN, TTYPES, sum_for_key)

VERS = 'concat_block_pairs_act'
OUT = OUT_ROOT / 'cvTrue'
DEFAULT_ONE_CACHE = Path.home() / 'Downloads/ONE/openalyx.internationalbrainlab.org'
DEFAULT_TYPES = ['integrator', 'move_all', 'move_init', 'move', 'move_fback']
EXCLUDE_REGIONS = {'stim_res': {'AUDp'}}


def action_kernel_priors(alpha, actions):
    '''Left prior. 0 / NaN holds; left (>0) steps up; right (<0) steps down.'''
    prior = 0.5
    priors = [prior]
    for t in range(len(actions) - 1):
        action = float(np.asarray(actions[t], dtype=float).reshape(-1)[0])
        if (not np.isfinite(action)) or action == 0.0:
            priors.append(prior)
            continue
        prior = alpha * int(action > 0) + (1 - alpha) * prior
        priors.append(prior)
    binary = np.double(np.double(priors) >= 0.5) * 0.6 + 0.2
    return priors, binary


def not_true_half(probability_left):
    """Rows whose true block is not 0.5. Call this before overwriting the column."""
    return ~np.isclose(np.asarray(probability_left, dtype=float), 0.5)


def trial_types(trials, masks):
    """{name: (align event, bool mask, (pre, post))} for the 16 block-pair types.

    ``masks['stim']`` is the stimOn keep (bad-trial + stim saturation + true-0.5
    drop). ``masks['move']`` is the same for firstMovement.
    """
    pl, ch = trials['probabilityLeft'], trials['choice']
    side = {'L': ~np.isnan(trials['contrastLeft']), 'R': ~np.isnan(trials['contrastRight'])}
    block = {'L': pl == 0.8, 'R': pl == 0.2}
    choice = {'L': ch == 1, 'R': ch == -1}
    tts = {}
    for s in 'LR':
        for b in 'LR':
            for c in 'LR':
                common = [np.asarray(side[s]), np.asarray(block[b]), np.asarray(choice[c])]
                tts[f'stim{s}b{b}c{c}'] = (
                    'stimOn_times', np.bitwise_and.reduce([masks['stim'], *common]), STIM_WIN)
                tts[f's{s}b{b}choice{c}'] = (
                    'firstMovement_times', np.bitwise_and.reduce([masks['move'], *common]),
                    MOVE_WIN)
    return tts


def typed_spikes(spikes, clusters, in_any):
    """Spike times / cluster ids / sorted cluster ids restricted to typed cells.

    bin_spikes2D requires every spike's cluster to be in ``cluster_ids``, so
    spikes are filtered first; counts are identical to binning all cells and
    slicing afterwards.
    """
    cid = np.asarray(clusters['cluster_id'])
    keep = in_any[spikes['clusters']]
    return spikes['times'][keep], cid[spikes['clusters'][keep]], cid[in_any]


def binned(sp, events, win):
    """(ntr, ncells, 6 * nbin) overlapping-bin spike counts, as concat_PETHs."""
    from brainbox.singlecell import bin_spikes2D
    times, spk_cl, cl_ids = sp
    st = int(T_BIN // STS)
    bis = []
    for ts in range(st):
        bi, _ = bin_spikes2D(times, spk_cl, cl_ids, np.asarray(events) + ts * STS,
                             win[0], win[1], T_BIN)
        bis.append(bi)
    ntr, nn, nbin = bis[0].shape
    ar = np.zeros((ntr, nn, st * nbin))
    for ts in range(st):
        ar[:, :, ts::st] = bis[ts]
    return ar


def shuffled_diffs(ar0, ar1, nrand, rng, chunk=250):
    """(nrand, ncells, T) of mean(shuffled group0) - mean(shuffled group1)."""
    n0, n1 = len(ar0), len(ar1)
    C = np.concatenate([ar0, ar1]).reshape(n0 + n1, -1).astype(np.float32)
    out = np.empty((nrand, C.shape[1]), dtype=np.float32)
    for lo in range(0, nrand, chunk):
        k = min(chunk, nrand - lo)
        perm = np.argsort(rng.random((k, n0 + n1)), axis=1)
        W = np.full((k, n0 + n1), -1.0 / n1, dtype=np.float32)
        np.put_along_axis(W, perm[:, :n0], 1.0 / n0, axis=1)
        out[lo:lo + k] = W @ C
    return out.reshape(nrand, ar0.shape[1], ar0.shape[2])


def load_cell_types(path):
    ct = np.load(path, allow_pickle=True).flat[0]
    return {t: set(v['uuids']) for t, v in ct['types'].items()}, ct


def insertion_list(ct_raw, types):
    pids = set()
    for t in types:
        pids |= set(ct_raw['types'][t]['pid'])
    return sorted(pids)


def uuid_by_cluster(ct_raw, types, pid):
    """{cluster_id: uuid} for this pid's typed cells (from cell_types.npy)."""
    out = {}
    for t in types:
        v = ct_raw['types'][t]
        m = v['pid'] == pid
        out.update(zip(v['cluster_id'][m].tolist(), v['uuids'][m]))
    return out


def _sat_column(sess_trials, n_trials, key):
    """Saturation flag aligned to the truncated session-loader table."""
    sess = sess_trials.iloc[:n_trials]
    if len(sess) < n_trials:
        raise AssertionError(
            f'trials.pqt ({len(sess)}) shorter than the session table ({n_trials})')
    return sess[key].to_numpy() == True


def load_from_one(one, eid, pid, sess_trials, cell_types, types):
    """Full loader table for the kernel. Masks are applied later, per alignment.

    Saturation flags come from the aggregate table by row, so a session cut by
    truncate_to_pass does not hit load_trials_and_mask's length assert.
    """
    from brainwidemap import load_good_units, load_trials_and_mask
    trials, base = load_trials_and_mask(one, eid, saturation_intervals=None)
    base = np.asarray(base)
    keep = not_true_half(trials['probabilityLeft'])
    masks = {}
    for align, key in ALIGN_SATURATION.items():
        sat = _sat_column(sess_trials, len(trials), key)
        masks[align] = base & ~sat & keep
    choices = list(trials['choice'])
    spikes, clusters = load_good_units(one, pid)
    uu = np.asarray(clusters['uuids'])
    in_any = np.isin(uu, list(set().union(*(cell_types[t] for t in types))))
    return trials, masks, choices, spikes, clusters, in_any, uu[in_any]


def load_from_cache(cache, sess_trials, cid2uuid):
    """Same trial rules as load_from_one, from manifold/insertion_cache + trials.pqt.

    Each cache table is already the bad-trial mask plus one saturation key, with
    the original trial index kept. StimOn uses the stim table; firstMovement
    uses the move table. The action-kernel prior runs over the session's full
    choice sequence from trials.pqt, then those labels are indexed onto the
    union of the two tables. True probabilityLeft == 0.5 is removed after that.
    """
    import pandas as pd
    tabs = cache['trials']
    missing = [k for k in ALIGN_SATURATION.values() if k not in tabs]
    if missing:
        raise KeyError(f'insertion cache lacks saturation tables {missing}')
    stim_df = tabs[ALIGN_SATURATION['stim']]
    move_df = tabs[ALIGN_SATURATION['move']]
    both = pd.concat([stim_df, move_df])
    both = both[~both.index.duplicated(keep='first')].sort_index()
    both = both.iloc[np.flatnonzero(not_true_half(both['probabilityLeft']))]
    masks = {
        'stim': np.asarray(both.index.isin(stim_df.index)),
        'move': np.asarray(both.index.isin(move_df.index)),
    }
    choices = list(sess_trials['choice'])
    clusters = cache['clusters']
    cid = np.asarray(clusters['cluster_id'])
    in_any = np.isin(cid, list(cid2uuid))
    uu_in = np.array([cid2uuid[c] for c in cid[in_any]], dtype=object)
    return both, masks, choices, np.asarray(both.index), cache['spikes'], clusters, in_any, uu_in


def compute_core(pid, eid, trials, masks, prior_l, spikes, clusters, in_any, uu_in,
                 cell_types, types, nrand, prior, rng, br):
    if prior_l is not None:
        trials = trials.copy()
        trials['probabilityLeft'] = prior_l
    tts = trial_types(trials, masks)
    acr = np.array(br.id2acronym(np.asarray(clusters['atlas_id'])[in_any], mapping='Beryl'))
    sp = typed_spikes(spikes, clusters, in_any)
    type_masks = {}
    for t in types:
        m = np.isin(uu_in, list(cell_types[t]))
        if t in EXCLUDE_REGIONS:
            m &= ~np.isin(acr, list(EXCLUDE_REGIONS[t]))
        type_masks[t] = m

    ws, tls, diffs = {}, {}, {}
    for a, b in BLOCK_PAIRS:
        ars = []
        for tt in (a, b):
            ev_name, m, win = tts[tt]
            ev = trials[ev_name][m]
            tls[tt] = int(len(ev))
            ars.append(binned(sp, ev, win) if len(ev) else None)
        for tt, ar in zip((a, b), ars):
            nbin = int(round((tts[tt][2][0] + tts[tt][2][1]) / T_BIN)) * int(T_BIN // STS)
            ws[tt] = ar.mean(0) if ar is not None else np.full((in_any.sum(), nbin), np.nan)
        if ars[0] is not None and ars[1] is not None:
            diffs[f'{a}_{b}'] = shuffled_diffs(ars[0], ars[1], nrand, rng)
        else:
            diffs[f'{a}_{b}'] = None

    concat = np.concatenate([ws[t] for t in TTYPES], axis=1)
    good = ~np.isnan(concat).any(axis=1) & np.any(concat, axis=1)
    ctrl = {t: {} for t in types}
    for key, D in diffs.items():
        for t in types:
            m = type_masks[t] & good
            ctrl[t][key] = (np.nansum(D[:, m] ** 2, axis=1) if D is not None
                            else np.full((nrand, ws[key.split('_')[0]].shape[1]), np.nan))
    return dict(pid=pid, eid=eid, uuids=uu_in, ids=np.asarray(clusters['atlas_id'])[in_any],
                acronym=acr, concat=concat.astype(np.float32), good=good, tls=tls,
                type_masks=type_masks, ctrl=ctrl, ttypes=TTYPES, prior=prior, nrand=nrand,
                lens={t: ws[t].shape[1] for t in TTYPES})


class Loader:
    """Insertion cache first (``<one-cache>/manifold/insertion_cache``), ONE on miss."""

    def __init__(self, args, ct_raw):
        self.args, self.ct_raw = args, ct_raw
        self.root = args.one_cache_dir
        self.icache = self.root / 'manifold' / 'insertion_cache'
        self._one = self._trials = None

    @property
    def one(self):
        if self._one is None:
            from one.api import ONE
            kw = dict(base_url=self.args.base_url, silent=True, cache_dir=str(self.root))
            if 'openalyx' in self.args.base_url:
                kw['password'] = 'international'
            self._one = ONE(**kw)
        return self._one

    def sess_trials(self, eid):
        if self._trials is None:
            import pandas as pd
            self._trials = pd.read_parquet(self.root / 'bwm_tables' / 'trials.pqt')
        t = self._trials[self._trials['eid'] == eid]
        return t.reset_index(drop=True)

    def __call__(self, pid, cell_types, types):
        """(source, eid, trials, masks, prior_l, spikes, clusters, in_any, uu_in).

        ``masks`` is ``{'stim', 'move'}``: bad-trial and that alignment's
        saturation key, then true probabilityLeft == 0.5 removed. ``prior_l``
        is the full-sequence act label on the returned rows (None for the true
        block). On the cache path the returned frame is the union of the two
        alignment tables, so ``prior_l`` is indexed by the original trial index.
        """
        alpha = ALPHA_AK if self.args.prior == 'act' else None
        eid, probe = self.ct_raw['insertions'][pid]
        cpath = self.icache / f'{eid}_{probe}.npy'
        if self.args.source != 'one' and cpath.exists():
            cache = np.load(cpath, allow_pickle=True).item()
            assert cache['pid'] == pid, (cpath, cache['pid'])
            trials, masks, choices, idx, spikes, clusters, in_any, uu_in = load_from_cache(
                cache, self.sess_trials(eid), uuid_by_cluster(self.ct_raw, types, pid))
            trials = trials.reset_index(drop=True)
            prior_l = None if alpha is None else np.asarray(
                action_kernel_priors(alpha, choices)[1])[idx]
            return 'cache', eid, trials, masks, prior_l, spikes, clusters, in_any, uu_in
        if self.args.source == 'cache':
            raise FileNotFoundError(cpath)
        trials, masks, choices, spikes, clusters, in_any, uu_in = load_from_one(
            self.one, eid, pid, self.sess_trials(eid), cell_types, types)
        prior_l = None if alpha is None else action_kernel_priors(alpha, choices)[1]
        return 'one', eid, trials, masks, prior_l, spikes, clusters, in_any, uu_in


def cmd_compute(args):
    from iblatlas.regions import BrainRegions
    br = BrainRegions()
    cell_types, ct_raw = load_cell_types(args.cell_types)
    pids = insertion_list(ct_raw, args.types)
    pids = pids[args.shard::args.n_shards]
    pdir = args.out / f'peths_{VERS}' if args.prior == 'act' else args.out / f'peths_{VERS}_trueprior'
    pdir.mkdir(parents=True, exist_ok=True)
    load = Loader(args, ct_raw)
    print(f'shard {args.shard}/{args.n_shards}: {len(pids)} insertions -> {pdir}\n'
          f'  source={args.source} insertion_cache={load.icache}', flush=True)
    fails, n_src = [], {'cache': 0, 'one': 0}
    for k, pid in enumerate(pids):
        f = pdir / f'{pid}.npy'
        if f.exists() and not args.overwrite:
            continue
        t0 = time.perf_counter()
        rng = np.random.default_rng([args.seed, uuid.UUID(pid).int % (2 ** 32)])
        try:
            src, eid, trials, masks, prior_l, spikes, clusters, in_any, uu_in = load(
                pid, cell_types, args.types)
            t_load = time.perf_counter() - t0
            n_src[src] += 1
            if not in_any.any():
                print(f'{k + 1}/{len(pids)} {pid}: no typed cells ({src})', flush=True)
                continue
            D = compute_core(pid, eid, trials, masks, prior_l, spikes, clusters, in_any, uu_in,
                             cell_types, args.types, args.nrand, args.prior, rng, br)
            D['source'] = src
            tmp = f.with_name(f'.{f.stem}.tmp.npy')
            np.save(tmp, D, allow_pickle=True)
            tmp.replace(f)
            print(f'{k + 1}/{len(pids)} {pid}: {len(D["uuids"])} cells, {src}, '
                  f'load {t_load:.0f}s, total {time.perf_counter() - t0:.0f}s', flush=True)
        except Exception:
            fails.append(pid)
            print(f'{k + 1}/{len(pids)} {pid}: FAIL\n{traceback.format_exc()}', flush=True)
        gc.collect()
    print(f'sources: {n_src}; {len(fails)} failures: {fails}')


def cmd_stack(args):
    pdir = args.out / (f'peths_{VERS}' if args.prior == 'act' else f'peths_{VERS}_trueprior')
    vers = VERS if args.prior == 'act' else 'concat_block_pairs'
    files = sorted(pdir.glob('*.npy'))
    print(f'stacking {len(files)} insertions from {pdir}')
    acc = {t: dict(uuids=[], ids=[], pid=[], concat=[], ctrl={}) for t in args.types}
    lens = None
    n_skip = 0
    for f in files:
        D = np.load(f, allow_pickle=True).flat[0]
        if sum(v == 0 for v in D['tls'].values()) > 2:
            n_skip += 1
            continue
        lens = D['lens']
        for t in args.types:
            m = D['type_masks'][t] & D['good']
            if not m.any():
                continue
            a = acc[t]
            a['uuids'].append(D['uuids'][m])
            a['ids'].append(D['ids'][m])
            a['pid'].append(np.repeat(D['pid'], m.sum()))
            a['concat'].append(D['concat'][m])
            for key, c in D['ctrl'][t].items():
                a['ctrl'][key] = a['ctrl'].get(key, 0) + c
    print(f'skipped {n_skip} insertions with >2 empty trial types')
    for t in args.types:
        a = acc[t]
        cs = np.concatenate(a['concat']).astype(np.float64)
        r = dict(uuids=np.concatenate(a['uuids']), ids=np.concatenate(a['ids']),
                 pid=np.concatenate(a['pid']), concat=cs, len=lens,
                 distance_controls=a['ctrl'])
        r['distance'] = {}
        for x, y in BLOCK_PAIRS:
            s1, e1 = sum_for_key(lens, x), sum_for_key(lens, x, after=True)
            s2, e2 = sum_for_key(lens, y), sum_for_key(lens, y, after=True)
            r['distance'][f'{x}_{y}'] = np.nansum((cs[:, s1:e1] - cs[:, s2:e2]) ** 2, axis=0)
        sd = cs.std(axis=1, keepdims=True)
        r['concat_z'] = (cs - cs.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, np.nan)
        f = args.out / f'{vers}_normFalse_shuffleTrue_{t}.npy'
        np.save(f, r, allow_pickle=True)
        print(f'{t}: {len(cs)} cells -> {f}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name, fn in (('compute', cmd_compute), ('stack', cmd_stack)):
        p = sub.add_parser(name)
        p.add_argument('--out', type=Path, default=OUT)
        p.add_argument('--types', nargs='+', default=DEFAULT_TYPES)
        p.add_argument('--prior', choices=['act', 'true'], default='act')
        p.set_defaults(func=fn)
        if name == 'compute':
            p.add_argument('--cell-types', type=Path, default=CELL_TYPES)
            p.add_argument('--shard', type=int, default=0)
            p.add_argument('--n-shards', type=int, default=1)
            p.add_argument('--nrand', type=int, default=NRAND)
            p.add_argument('--seed', type=int, default=0)
            p.add_argument('--one-cache-dir', type=Path,
                           default=Path(os.environ.get('ONE_CACHE_DIR', DEFAULT_ONE_CACHE)),
                           help='ONE root holding manifold/insertion_cache and bwm_tables/')
            p.add_argument('--base-url', default=os.environ.get(
                'ONE_BASE_URL', 'https://openalyx.internationalbrainlab.org'))
            p.add_argument('--source', choices=['auto', 'cache', 'one'], default='auto',
                           help='auto: insertion cache, ONE on miss')
            p.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
