#!/usr/bin/env python
"""Figure 4 i with the movement slope cut at the mean slope (2026-11-02).

Movement = end peak (move-aligned choice curve) & slope > cut & choice
significant around movement; integrator = choice significant and rising, minus
movement, on each row; C / M = union over the stim and choice rows. This is
get_sc_table's labelling (the fixed rule, build_standard_figs.MOVE_RULE =
'fixed'), re-run on the res_standard type table with three cuts:
  - current: late slope (last 20 bins, per bin) > 0.05 (= fit_targets_integ/oldrule)
  - late mean: late slope > its mean over the 118 choice-significant regions
  - overall mean: overall slope (amp_slope, 0–150 ms, per s) > its mean over them
Stim-row sensory labels (0 / 0.1) are kept, as in apply_move_rule.

Output: <oa>/manifold/paper_figs/fig4/res_standard/panel_i_slope_mean_cut.{svg,png,json}

  conda activate iblenv   # outside the sandbox
  python scripts/paper_figs/fig4/panel_i_slope_mean_cut.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE), str(HERE.parent / 'fig2')]
import build_standard_figs as b  # noqa: E402
import fig4_common as fc  # noqa: E402

fc.OA_RES = b.STAGE
fc.ALYX_RES = b.STAGE
fc.SC_TABLE = b.OUT2 / 'sc_table.csv'
fc.EARLYSTIM_NPY = b.STAGE / 'combined_act_block_duringstim_r_act_block_duringstim_l_earlystim80.npy'
import fig4_plot as p4  # noqa: E402

OUT = b.OUT4
ROWS = ('sc_duringstim', 'sc_duringchoice')


def label_sets(res, shape):
    """C / M sets for a movement-shape mask, get_sc_table style."""
    move = shape & (res['sc_duringchoice'] > 0)
    c, m = set(), set()
    for row in ROWS:
        integ = (res[row] > 0) & res[f'{row}_amp_slope'].astype(bool) & ~move
        old = res[f'{row}_regtype_old' if row == 'sc_duringstim' else f'{row}_regtype']
        t = old.to_numpy(dtype=float).copy()
        redo = np.isnan(t) | np.isin(t, [0.5, 1.0])
        new = np.full(len(res), np.nan)
        new[move.to_numpy()] = 1.0
        new[integ.to_numpy()] = 0.5
        t[redo] = new[redo]
        c |= set(res['region'][t == 0.5])
        m |= set(res['region'][t == 1.0])
    return sorted(c), sorted(m)


def mean_cut_rules():
    """Type table, the two mean cuts and the three movement-shape masks."""
    res = pd.read_csv(b.OUT2 / 'sc_table.csv')
    cc = np.load(next(b.STAGE.glob('combined_choice_stim_r_block_r_act*.npy')), allow_pickle=True).item()
    res['amp_slope_m'] = [float(cc[r]['amp_slope']) if r in cc else np.nan for r in res['region']]
    choice = (res['choice_s_fdr'] <= b.ALPHA) | (res['choice_m_fdr'] <= b.ALPHA)
    end_peak = res['end_peak_m'] > 0
    cut_late = float(res.loc[choice, 'slope_last_m'].mean())
    cut_overall = float(res.loc[choice, 'amp_slope_m'].mean())
    rules = {
        f'current: late > {b.SLOPE_THRESHOLD}': end_peak & (res['slope_last_m'] > b.SLOPE_THRESHOLD),
        f'late > mean ({cut_late:.3f})': end_peak & (res['slope_last_m'] > cut_late),
        f'overall > mean ({cut_overall:.1f}/s)': end_peak & (res['amp_slope_m'] > cut_overall),
    }
    return res, choice, cut_late, cut_overall, rules


def mean_cut_sets():
    """C / M for the late-mean (opt3) and overall-mean (opt4) cuts, with the cut values."""
    res, _, cut_late, cut_overall, rules = mean_cut_rules()
    _, late, overall = rules.values()
    return {'opt3_latemean': (*label_sets(res, late), 'slope_last_m', cut_late),
            'opt4_overallmean': (*label_sets(res, overall), 'amp_slope_m', cut_overall)}


def main():
    res, choice, cut_late, cut_overall, rules = mean_cut_rules()
    sets = {k: label_sets(res, v) for k, v in rules.items()}
    fit = json.loads((b.REPO / 'fit_targets_integ' / 'oldrule' / 'regions.json').read_text())
    cur = sets[next(iter(sets))]
    assert (cur[0], cur[1]) == (sorted(fit['I']), sorted(fit['M'])), 'current rule != oldrule regions'

    d = p4.Data()
    t_s, y_s = d.load_group_sub(d.overlap, p4.UNSPLIT_TF, fc.ALYX_RES, t_max=fc.T_MAX_MS)
    t_s = np.asarray(t_s) * (1000 if np.max(t_s) < 1 else 1)
    curves = {k: {g: {tf: d.load_group_sub(regs, tf) for tf in (p4.STIM_TF, p4.CHOICE_TF)}
                  for g, regs in zip('CM', v)} for k, v in sets.items()}
    allv = np.concatenate([np.ravel(y) for s in curves.values() for g in s.values()
                           for y in g.values()] + [y_s])
    ylim = (min(0.0, allv.min() - 0.01), allv.max() + 0.01)

    fig, axs = plt.subplots(len(sets), 2, figsize=(7, 2.5 * len(sets)), sharey=True,
                            layout='constrained')
    for i, (title, (cr, mr)) in enumerate(sets.items()):
        for j, (tf, align) in enumerate(((p4.STIM_TF, 'stim'), (p4.CHOICE_TF, 'move'))):
            ax = axs[i, j]
            yc = curves[title]['C'][tf]
            t = np.linspace(0, 150, len(yc)) if align == 'stim' else np.linspace(-150, 0, len(yc))
            if align == 'stim':
                ax.plot(t_s, y_s, color=fc.C_STIM, lw=1.5, label=f'S ({len(d.overlap)})')
            ax.plot(t, yc, color='gold', lw=1.5, label=f'C ({len(cr)})')
            ax.plot(t, curves[title]['M'][tf], color='tomato', lw=1.5, label=f'M ({len(mr)})')
            ax.axvline(0, color=fc.C_STIM_ON if align == 'stim' else fc.C_MOVE_ON, lw=2)
            ax.set_ylim(ylim)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_xlabel('time from stim on (ms)' if align == 'stim' else 'time to mvt on (ms)')
            if j == 0:
                ax.set_ylabel(f'{title}\nprior distance − shuffle mean', fontsize=8.5)
                ax.legend(fontsize=7, frameon=False, loc='upper left')
    for ext in ('svg', 'png'):
        fig.savefig(OUT / f'panel_i_slope_mean_cut.{ext}', dpi=200, facecolor='white')
    plt.close(fig)

    def summ(y):
        y = np.asarray(y)
        return {'mean': float(np.mean(y)), 'first5': float(np.mean(y[:5])),
                'last5': float(np.mean(y[-5:]))}
    base_c, base_m = map(set, cur)
    info = {'cut_late_mean': cut_late, 'cut_overall_mean': cut_overall, 'n_choice_sig': int(choice.sum()),
            'S': d.overlap, 'rules': {}}
    for k, (cr, mr) in sets.items():
        info['rules'][k] = {
            'n_C': len(cr), 'n_M': len(mr),
            'M_added': sorted(set(mr) - base_m), 'M_removed': sorted(base_m - set(mr)),
            'C_added': sorted(set(cr) - base_c), 'C_removed': sorted(base_c - set(cr)),
            'curves': {g: {tf: summ(y) for tf, y in v.items()} for g, v in curves[k].items()},
            'C': cr, 'M': mr,
        }
    (OUT / 'panel_i_slope_mean_cut.json').write_text(json.dumps(info, indent=1))
    print(f'cuts: late mean {cut_late:.4f}, overall mean {cut_overall:.2f}/s (n={int(choice.sum())})')
    for k, r in info['rules'].items():
        cv = r['curves']
        print(f"\n{k}: C {r['n_C']} M {r['n_M']}")
        print(f"  M +{r['M_added']}  −{r['M_removed']}")
        print(f"  C +{r['C_added']}  −{r['C_removed']}")
        for g in 'CM':
            print(f"  {g} stim mean {cv[g][p4.STIM_TF]['mean']:.4f} last5 {cv[g][p4.STIM_TF]['last5']:.4f} | "
                  f"move mean {cv[g][p4.CHOICE_TF]['mean']:.4f} last5 {cv[g][p4.CHOICE_TF]['last5']:.4f}")
    print('wrote', OUT / 'panel_i_slope_mean_cut.png')


if __name__ == '__main__':
    main()
