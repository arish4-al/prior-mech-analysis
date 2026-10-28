#!/usr/bin/env python
"""I/M traj and S RMS fit targets from the insertion cache, per region set.

Rebuilds the two neural fit targets that ``build_fit_targets_integ.py``
cannot make on the laptop, in the notebook format the fitter reads:

  mean_data_results.npy   I/M ``mean_traj`` {stim: 8 keys, choice: 8 keys}
                          (``model_test.ipynb`` / ``get_data_for_fitting.py``)
  avg_mean_R.npy          S, stim-R correct trials, per |c|: mean over the two
                          prior keys of the RMS, minus its first-7-bin mean

Keys (``dmn_avgbystim.concat_PETHs`` with the act prior): stim{L,R} = side
with a finite contrast; b{L,R} = act-prior 0.8 / 0.2; c / choice{L,R} =
choice 1 / −1. Stim window 0–200 ms (96 bins), move window −150–0 ms (72).
RMS over all cells of the set: ``sqrt(Σ cell_mean² / n / T_BIN)``.

Prior labels are the standard ones (``attach_standard_act_prior`` /
``apply_standard_act_prior``): α=0.2 kernel on the full session, looked up
onto the cached masked trials, then true-0.5 trials dropped.

One pass over the cache fills every set at once. ``legacy`` uses the current
``fit_targets`` I/M lists and the old ``stim_regs``, and is compared key by
key against ``fit_targets`` (correlation, scale) as the recipe check. Each
``--target-dir`` must hold ``regions.json`` (I, M, S, from
build_fit_targets_integ.py); its traj files are written there and P is
copied from ``fit_targets``. Plots and the check go to ``--plot-dir``.

  python scripts/build_fit_traj_targets_from_cache.py \
    --target-dir fit_targets_integ/opt1_rawslope \
    --target-dir fit_targets_integ/opt2_shuffleslope

Run on ORCD via scripts/submit_fit_targets_integ.sh (needs ONE for the
full-session trial tables).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from iblatlas.regions import BrainRegions

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from build_mean_data_im_from_cache import (  # noqa: E402
    DEFAULT_CACHE, T_BIN, RmsAcc, bin_aligned, n_bins, subset_spikes,
)

OLD = ROOT / 'fit_targets'
OLD_STIM_REGS = ['VISpm', 'VISam', 'FRP', 'VISp', 'VISli', 'LGd', 'LP', 'NOT']
CONTRASTS = (1.0, 0.25, 0.125, 0.0625, 0.0)
SIDES = (('L', 'contrastLeft'), ('R', 'contrastRight'))
BLOCKS = (('L', 0.8), ('R', 0.2))
CHOICES = (('L', 1), ('R', -1))
STIM_PRE_POST = (0.0, 0.2)
MOVE_PRE_POST = (0.15, 0.0)
S_BASE_BINS = 7


def stim_key(s, b, c):
    return f'stim{s}b{b}c{c}'


def choice_key(s, b, c):
    return f's{s}b{b}choice{c}'


def trial_masks(trials):
    pl = trials['probabilityLeft'].to_numpy(float)
    ch = trials['choice'].to_numpy(float)
    out = {}
    for (s, col), (b, p), (c, cv) in product(SIDES, BLOCKS, CHOICES):
        out[(s, b, c)] = (np.isfinite(trials[col].to_numpy(float))
                          & np.isclose(pl, p) & (ch == cv))
    return out


def process(cache, sets, br, acc, min_trials):
    from block_analysis_allsplits import attach_standard_act_prior, apply_standard_act_prior
    acs = np.asarray(br.id2acronym(cache['clusters']['atlas_id'], mapping='Beryl'))
    member = {name: np.isin(acs, list(regs)) for name, regs in sets.items()}
    keep = np.any(list(member.values()), axis=0)
    if not keep.any():
        return 0
    attach_standard_act_prior(cache)
    pack = cache['standard_act']
    times, spk_cid, cid = subset_spikes(cache['spikes'], cache['clusters'], keep)
    member = {name: m[keep] for name, m in member.items()}

    def binned(trials, ev_col, pre_post):
        trials = apply_standard_act_prior(trials, pack)
        ev = trials[ev_col].to_numpy(float)
        ok = np.isfinite(ev)
        return bin_aligned(times, spk_cid, cid, ev[ok], *pre_post), \
            trials.iloc[np.flatnonzero(ok)]

    b_stim, t_stim = binned(cache['trials']['saturation_stim_plus04'],
                            'stimOn_times', STIM_PRE_POST)
    b_move, t_move = binned(cache['trials']['saturation_move_minus02'],
                            'firstMovement_times', MOVE_PRE_POST)

    def add(win, b, mask, key):
        if int(mask.sum()) < min_trials:
            return
        mean = b[mask].mean(axis=0)
        for name, sel in member.items():
            if sel.any():
                acc[name][win].add(key, mean[sel])

    m_stim, m_move = trial_masks(t_stim), trial_masks(t_move)
    cr = t_stim['contrastRight'].to_numpy(float)
    for (s, bl, c) in m_stim:
        add('stim', b_stim, m_stim[(s, bl, c)], stim_key(s, bl, c))
        add('choice', b_move, m_move[(s, bl, c)], choice_key(s, bl, c))
    for (bl, _), cval in product(BLOCKS, CONTRASTS):
        mask = m_stim[('R', bl, 'R')] & np.isclose(cr, cval)
        add('stim', b_stim, mask, f'{stim_key("R", bl, "R")}_{cval}')
    return 1


def traj(acc_set, keys_stim, keys_choice):
    return {'stim': {k: acc_set['stim'].rms(k) for k in keys_stim if acc_set['stim'].rms(k) is not None},
            'choice': {k: acc_set['choice'].rms(k) for k in keys_choice
                       if acc_set['choice'].rms(k) is not None}}


def avg_mean_r(acc_set):
    out = {}
    for cval in CONTRASTS:
        rs = [acc_set['stim'].rms(f'{stim_key("R", bl, "R")}_{cval}') for bl, _ in BLOCKS]
        rs = [r for r in rs if r is not None]
        if rs:
            r = np.mean(rs, axis=0)
            out[cval] = r - np.mean(r[:S_BASE_BINS])
    return out


def check(old, new):
    """Correlation and mean ratio per key, new vs old."""
    out = {}
    for k in old:
        if k in new and new[k] is not None:
            a, b = np.asarray(old[k], float), np.asarray(new[k], float)
            n = min(a.size, b.size)
            out[str(k)] = {'r': float(np.corrcoef(a[:n], b[:n])[0, 1]),
                           'scale': float(np.mean(b[:n]) / np.mean(a[:n]))
                           if np.mean(a[:n]) else float('nan')}
    return out


def plot(payloads, out_png):
    fig, axs = plt.subplots(2, 3, figsize=(11, 5.5), layout='constrained')
    for name, (md, amr) in payloads.items():
        ls = ':' if name == 'fit_targets' else '-'
        for i, vn in enumerate(('I', 'M')):
            for j, win in enumerate(('stim', 'choice')):
                mt = md[vn]['mean_traj'][win]
                cL = [np.asarray(v, float) for k, v in mt.items() if k.endswith('L')]
                if cL:
                    T = min(c.size for c in cL)
                    axs[i, j].plot(np.mean([c[:T] for c in cL], 0), ls=ls, label=name)
                axs[i, j].set_title(f'{vn} {win} (choice L, mean of keys)', fontsize=8)
        for cval, r in sorted(amr.items()):
            axs[0, 2].plot(r, ls=ls, lw=1, label=f'{name} c={cval}' if cval == 1.0 else None)
        axs[0, 2].set_title('S avg_mean_R (all |c|)', fontsize=8)
    axs[1, 2].axis('off')
    axs[0, 0].legend(fontsize=6, frameon=False)
    axs[0, 2].legend(fontsize=6, frameon=False)
    for ext in ('svg', 'png'):
        fig.savefig(out_png.with_suffix(f'.{ext}'), dpi=200, facecolor='white')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE)
    ap.add_argument('--target-dir', type=Path, action='append', required=True)
    ap.add_argument('--plot-dir', type=Path, default=None)
    ap.add_argument('--n-insertions', type=int, default=None)
    ap.add_argument('--min-trials', type=int, default=1)
    ap.add_argument('--max-skip-frac', type=float, default=0.05)
    args = ap.parse_args()

    old_md = np.load(OLD / 'mean_data_results.npy', allow_pickle=True).flat[0]
    old_amr = np.load(OLD / 'avg_mean_R.npy', allow_pickle=True).flat[0]
    targets = {}
    for td in args.target_dir:
        td = (ROOT / td).resolve() if not td.is_absolute() else td
        targets[td.name] = (td, json.loads((td / 'regions.json').read_text()))
    sets = {'legacy_I': set(old_md['I']['regs']), 'legacy_M': set(old_md['M']['regs']),
            'legacy_S': set(OLD_STIM_REGS)}
    for name, (_, regs) in targets.items():
        for vn in ('I', 'M', 'S'):
            sets[f'{name}_{vn}'] = set(regs[vn])
    for name, regs in sets.items():
        print(f'{name}: {len(regs)} regions', flush=True)

    files = sorted(args.cache_dir.expanduser().glob('*.npy'))[:args.n_insertions]
    if not files:
        raise SystemExit(f'no insertion cache in {args.cache_dir}')
    nb = {'stim': n_bins(*STIM_PRE_POST), 'choice': n_bins(*MOVE_PRE_POST)}
    acc = {name: {w: RmsAcc(n) for w, n in nb.items()} for name in sets}
    br = BrainRegions()
    t0, used, skipped = time.time(), 0, []
    for i, f in enumerate(files, 1):
        cache = np.load(f, allow_pickle=True).item()
        try:
            used += process(cache, sets, br, acc, args.min_trials)
        except Exception as e:  # one bad session should not kill the job
            print(f'  [{i}/{len(files)}] {f.name} SKIP {e!r}', flush=True)
            skipped.append(f.name)
            continue
        print(f'  [{i}/{len(files)}] {f.name}', flush=True)

    keys = {vn: (list(old_md[vn]['mean_traj']['stim']), list(old_md[vn]['mean_traj']['choice']))
            for vn in ('I', 'M')}
    # Fail closed: the fit step only checks that the files exist.
    if len(skipped) > args.max_skip_frac * len(files):
        raise SystemExit(f'{len(skipped)}/{len(files)} insertions failed; no targets written')
    for name in sets:
        if not name.endswith('_S'):
            vn = name[-1]
            for j, win in enumerate(('stim', 'choice')):
                missing = [k for k in keys[vn][j] if acc[name][win].rms(k) is None]
                if missing:
                    raise SystemExit(f'{name}: no cells for {missing}; no targets written')
        else:
            missing = [c for c in CONTRASTS if c not in avg_mean_r(acc[name])]
            if missing:
                raise SystemExit(f'{name}: no S curve at |c| {missing}; no targets written')
    n_cells = {name: {w: dict(a.n) for w, a in acc[name].items()} for name in sets}

    def build(prefix, regs):
        md = {'P': old_md['P']}
        for vn in ('I', 'M'):
            md[vn] = {'regs': sorted(regs[vn]), 'comparison': old_md[vn]['comparison'],
                      'mean_traj': traj(acc[f'{prefix}_{vn}'], *keys[vn])}
        return md, avg_mean_r(acc[f'{prefix}_S'])

    payloads = {'fit_targets': (old_md, old_amr)}
    leg_md, leg_amr = build('legacy', {'I': sets['legacy_I'], 'M': sets['legacy_M']})
    payloads['legacy'] = (leg_md, leg_amr)
    report = {'n_insertions': used, 'n_listed': len(files), 'skipped': skipped,
              'elapsed_s': time.time() - t0,
              'legacy_vs_fit_targets': {
                  vn: {w: check(old_md[vn]['mean_traj'][w], leg_md[vn]['mean_traj'][w])
                       for w in ('stim', 'choice')} for vn in ('I', 'M')},
              'legacy_vs_fit_targets_S': check(old_amr, leg_amr),
              'n_trials_x_cells': {k: {w: {kk: int(vv) for kk, vv in d.items()}
                                       for w, d in v.items()} for k, v in n_cells.items()}}
    for name, (td, regs) in targets.items():
        md, amr = build(name, regs)
        md['meta'] = {'recipe': 'insertion cache, notebook keys, standard act prior',
                      'n_insertions': used, 'source': str(args.cache_dir)}
        np.save(td / 'mean_data_results.npy', md, allow_pickle=True)
        np.save(td / 'avg_mean_R.npy', amr, allow_pickle=True)
        payloads[name] = (md, amr)
        print(f'wrote {td}', flush=True)

    plot_dir = args.plot_dir or args.cache_dir.expanduser().parent / 'fit_targets_integ'
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot(payloads, plot_dir / 'traj_targets.png')
    (plot_dir / 'traj_targets_check.json').write_text(json.dumps(report, indent=1, default=str))
    for vn in ('I', 'M'):
        for w in ('stim', 'choice'):
            c = report['legacy_vs_fit_targets'][vn][w]
            print(f'legacy vs fit_targets {vn} {w}: r min {min(v["r"] for v in c.values()):.3f}  '
                  f'scale {np.median([v["scale"] for v in c.values()]):.3f}')
    c = report['legacy_vs_fit_targets_S']
    print(f'legacy vs fit_targets S: r min {min(v["r"] for v in c.values()):.3f}  '
          f'scale {np.median([v["scale"] for v in c.values()]):.3f}')
    print(f'plots + check in {plot_dir}')


if __name__ == '__main__':
    main()
