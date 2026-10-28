#!/usr/bin/env python
"""Region lists + prior-distance fit targets for the integrator-rule comparison.

Two Stage B target sets, same movement and sensory regions, different
integrator rule (journals/paper_main_figures.md 2026-10-28):

  opt1_rawslope      integrator = choice significant & raw overall slope > 0
                     (current labels; Figure 4 i C = 84, M = 31)
  opt2_shuffleslope  same, and the whole-curve (72-bin) least-squares slope
                     of the choice curve beats its 2000 shuffles at BH-FDR
                     ≤ 0.01 over all regions, on that row (C = 57, M = 31)

Groups follow Figure 4 i: I = during-stim integrator ∪ move-aligned
integrator, M = during-stim movement ∪ move-aligned movement (get_sc_table
labels after apply_move_rule). S = the 10 sensory regions. S prior = S ∩
prior FDR ≤ 0.01 in 0–80 ms (VISal, VISam, LP, PO, ZI).

Prior curves come from the res-standard staging (act-kernel prior, 2000
shuffles), the source of Figure 4 i:

  data_act_block_duringstim.npy / data_act_block_duringchoice.npy
      ``load_group(regs, tf, alpha=1.0, correction='simple')`` on the
      stim × choice f1/f2 splits — the recipe of the old fit_targets files.
  data_act_block_duringstim_s_unsplit80.npy
      unsplit l/r, (obs − mean(nulls)) / 2, cell-weighted, t ≤ 80 ms — the
      recipe of build_s_prior_curve_unsplit80.py.

Writes into ``<repo>/fit_targets_integ/<opt>/`` (picked up by the fit with
FIT_TARGETS_DIR) plus ``regions.json``. The I/M traj and S RMS targets
(``mean_data_results.npy``, ``avg_mean_R.npy``) are added on ORCD by
scripts/build_fit_traj_targets_from_cache.py. Plot goes to
``<oa>/manifold/paper_figs/fit_targets_integ/``.

  conda activate iblenv   # outside the sandbox
  python scripts/build_fit_targets_integ.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'scripts/paper_figs'),
                str(ROOT / 'scripts/paper_figs/fig4')]
import build_standard_figs as b  # noqa: E402
import fig4_common as fc  # noqa: E402

fc.OA_RES = b.STAGE
fc.ALYX_RES = b.STAGE
fc.SC_TABLE = b.OUT2 / 'sc_table.csv'
fc.EARLYSTIM_NPY = b.STAGE / 'combined_act_block_duringstim_r_act_block_duringstim_l_earlystim80.npy'
import fig4_plot as p4  # noqa: E402
import model_functions as mf  # noqa: E402

OUT = ROOT / 'fit_targets_integ'
PLOT_DIR = b.OUT2.parents[1] / 'fit_targets_integ'
OLD = ROOT / 'fit_targets'
UNSPLIT = ['act_block_duringstim_l', 'act_block_duringstim_r']
T_MAX_MS, WINDOW_MS = 80.0, 150.0
OPTS = ('opt1_rawslope', 'opt2_shuffleslope')


def _slope(y):
    """Least-squares slope over all bins (per bin), for each row of y."""
    y = np.asarray(y, float)
    x = np.arange(y.shape[-1]) - (y.shape[-1] - 1) / 2
    return (y * x).sum(-1) / (x ** 2).sum()


def whole_curve_q(af, timeframe, regs):
    """BH-FDR q of the whole-curve slope against the shuffles, per region."""
    from statsmodels.stats.multitest import multipletests
    _, _, regde = b.combined_names(af, timeframe)
    rd = np.load(b.STAGE / f'{regde}.npy', allow_pickle=True).flat[0]
    names = [r for r in regs if r in rd]
    p = np.array([(1 + (_slope(rd[r][1]) >= _slope(rd[r][0])).sum()) / (1 + len(rd[r][1]))
                  for r in names])
    return dict(zip(names, multipletests(p, method='fdr_bh')[1]))


def s_prior_80(regs):
    """S prior-distance curve, as build_s_prior_curve_unsplit80.build_curve."""
    regde = np.load(b.STAGE / f"combined_regde_{'_'.join(UNSPLIT)}.npy",
                    allow_pickle=True).item()
    split = np.load(b.STAGE / f'{UNSPLIT[0]}.npy', allow_pickle=True).flat[0]
    n_full = len(np.asarray(next(iter(regde.values()))[0]).reshape(-1))
    n_keep = int(np.sum(np.linspace(0, WINDOW_MS, n_full) <= T_MAX_MS))
    wd = wo = wn = 0.0
    n_cells, used = 0, []
    for reg in regs:
        n = int(split.get(reg, {}).get('nclus', 0)) if reg in regde else 0
        if n <= 0:
            continue
        obs = np.asarray(regde[reg][0], float).reshape(-1) / len(UNSPLIT)
        null = np.asarray(regde[reg][1], float).mean(0) / len(UNSPLIT)
        wd, wo, wn = wd + (obs - null) * n, wo + obs * n, wn + null * n
        n_cells += n
        used.append(reg)
    r150 = wd / n_cells
    return {
        'regs_stim': used, 'regs_stim_requested': list(regs),
        'r_stim': r150[:n_keep], 'r_stim_150ms': r150,
        'r_obs': wo / n_cells, 'r_null': wn / n_cells,
        'n_keep': n_keep, 'n_full': n_full, 't_max_ms': T_MAX_MS,
        'window_ms': WINDOW_MS, 'alpha': fc.ALPHA, 'n_cells': n_cells,
        'n_regions': len(used), 'conditioning': 'unsplit', 'null': 'shuffle',
        'prior': 'act_standard', 'splits': UNSPLIT, 'n_splits': len(UNSPLIT),
        'source_regde': str(b.STAGE), 'region_rule': 'sensory 10 ∩ prior 0–80 ms FDR',
    }


def main():
    d = p4.Data()
    q_s = whole_curve_q(d.af, 'choice_duringstim_act', d.regs)
    q_m = whole_curve_q(d.af, 'choice_duringchoice_act', d.regs)
    ok = lambda q, r: q.get(r, 1.0) <= fc.ALPHA  # noqa: E731
    move = sorted(d.move_s | d.move_c)
    integ = {
        'opt1_rawslope': sorted(d.integ_s | d.integ_c),
        'opt2_shuffleslope': sorted({r for r in d.integ_s if ok(q_s, r)}
                                    | {r for r in d.integ_c if ok(q_m, r)}),
    }
    sensory = sorted(d.sensory)
    s_prior = list(d.overlap)
    if set(integ['opt1_rawslope']) & set(move):
        raise RuntimeError('I/M overlap')
    print(f"I opt1 {len(integ['opt1_rawslope'])}  opt2 {len(integ['opt2_shuffleslope'])}  "
          f"M {len(move)}  S {len(sensory)}  S prior {s_prior}")

    mf.pth_res = b.STAGE
    curves = {}
    for opt in OPTS:
        out = OUT / opt
        out.mkdir(parents=True, exist_ok=True)
        for tf in ('act_block_duringstim', 'act_block_duringchoice'):
            is_choice = 'duringchoice' in tf
            save = {'regs_int': integ[opt], 'regs_move': move, 'regs_stim': sensory,
                    'r_int': mf.load_group(integ[opt], tf, alpha=1.0, correction='simple'),
                    'r_move': mf.load_group(move, tf, alpha=1.0, correction='simple'),
                    'r_stim': None if is_choice else mf.load_group(
                        sensory, tf, is_stim=True, correction='simple'),
                    'source': f'res_standard staging {b.STAGE}'}
            np.save(out / f'data_{tf}.npy', save, allow_pickle=True)
            curves[(opt, tf)] = save
        s = s_prior_80(s_prior)
        s['regs_sensory'] = sensory
        np.save(out / 'data_act_block_duringstim_s_unsplit80.npy', s, allow_pickle=True)
        curves[(opt, 's')] = s
        (out / 'regions.json').write_text(json.dumps({
            'rule': opt, 'I': integ[opt], 'M': move, 'S': sensory, 'S_prior': s_prior,
            'P': list(np.load(OLD / 'mean_data_results.npy', allow_pickle=True)
                      .flat[0]['P']['regs']),
            'source_sc_table': str(fc.SC_TABLE),
        }, indent=1))
        print(f'wrote {out}')

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3), layout='constrained')
    for ax, tf in zip(axs[:2], ('act_block_duringstim', 'act_block_duringchoice')):
        old = np.load(OLD / f'data_{tf}.npy', allow_pickle=True).flat[0]
        t = np.linspace(0, 150, 72) if 'stim' in tf else np.linspace(-150, 0, 72)
        ax.plot(t, old['r_int'], color='gold', ls=':', label='I old fit_targets')
        ax.plot(t, old['r_move'], color='tomato', ls=':', label='M old fit_targets')
        for opt, ls in zip(OPTS, ('-', '--')):
            c = curves[(opt, tf)]
            ax.plot(t, c['r_int'], color='gold', ls=ls, label=f'I {opt} ({len(c["regs_int"])})')
        ax.plot(t, curves[(OPTS[0], tf)]['r_move'], color='tomato', label=f'M ({len(move)})')
        ax.set_title(tf, fontsize=9)
        ax.set_xlabel('ms')
    axs[0].set_ylabel('prior distance − shuffle mean')
    axs[0].legend(fontsize=6, frameon=False)
    old_s = np.load(OLD / 'data_act_block_duringstim_s_unsplit80.npy', allow_pickle=True).flat[0]
    new_s = curves[(OPTS[0], 's')]
    axs[2].plot(np.linspace(0, 80, len(old_s['r_stim'])), old_s['r_stim'], color='C0', ls=':',
                label=f"old ({len(old_s['regs_stim'])})")
    axs[2].plot(np.linspace(0, 80, len(new_s['r_stim'])), new_s['r_stim'], color='C0',
                label=f"new ({len(new_s['regs_stim'])})")
    axs[2].set_title('S prior 0–80 ms (full arm)', fontsize=9)
    axs[2].legend(fontsize=7, frameon=False)
    for ax in axs:
        ax.spines[['top', 'right']].set_visible(False)
    for ext in ('svg', 'png'):
        fig.savefig(PLOT_DIR / f'prior_targets.{ext}', dpi=200, facecolor='white')
    plt.close(fig)
    print(f'wrote {PLOT_DIR}/prior_targets.png')


if __name__ == '__main__':
    main()
