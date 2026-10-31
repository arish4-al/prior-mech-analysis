#!/usr/bin/env python
"""Figure 4 i on the standard data with the original C/M rules (2026-10-30).

C and M come straight from get_sc_table with the published parameters
(fig2_common: alpha 0.01, sc_threshold 0, fixed late-slope cut 0.05,
amp_loc 67, n 20), run on the res-standard staging. apply_move_rule (late
slope against the shuffles, 10-27) and the integrator slope tests (10-28)
are not applied. Integrator = choice significant and rising; movement =
end peak, late slope > 0.05 and choice significant; on either row.
S is the same five sensory ∩ prior 0–80 ms regions as panel i.

Rows: original rules; current opt1 (84 / 31); opt2 (57 / 31).

Output: <oa>/manifold/paper_figs/fig4/res_standard/panel_i_old_rules.{svg,png,json}

  conda activate iblenv   # outside the sandbox
  python scripts/paper_figs/fig4/panel_i_old_rules.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE), str(HERE.parent / 'fig2')]
import build_standard_figs as b  # noqa: E402
import fig2_common as f2  # noqa: E402
import fig4_common as fc  # noqa: E402

fc.OA_RES = b.STAGE
fc.ALYX_RES = b.STAGE
fc.SC_TABLE = b.OUT2 / 'sc_table.csv'
fc.EARLYSTIM_NPY = b.STAGE / 'combined_act_block_duringstim_r_act_block_duringstim_l_earlystim80.npy'
import fig4_plot as p4  # noqa: E402

OUT = b.OUT4
INTEG = b.REPO / 'fit_targets_integ'


def old_rule_sets(af):
    af.pth_res = b.STAGE
    res = af.get_sc_table(
        f2.SC_TIMES, f2.PTYPE, alpha=f2.ALPHA, combined_p=True,
        sc_threshold=f2.SC_THRESHOLD, slope_threshold=f2.SLOPE_THRESHOLD,
        amp_loc_threshold=f2.AMP_LOC_THRESHOLD, n=f2.N_SLOPE, stim_restr=True)
    res = res.set_index('region')
    c, m = set(), set()
    for row in ('sc_duringstim_regtype', 'sc_duringchoice_regtype'):
        c |= set(res.index[res[row] == 0.5])
        m |= set(res.index[res[row] == 1.0])
    res.reset_index().to_csv(OUT / 'panel_i_old_rules_sc_table.csv', index=False)
    return sorted(c), sorted(m)


def main():
    d = p4.Data()
    c_old, m_old = old_rule_sets(d.af)
    sets = {'original rules': (c_old, m_old)}
    for opt in ('opt1_rawslope', 'opt2_shuffleslope'):
        r = json.loads((INTEG / opt / 'regions.json').read_text())
        sets[f'{opt} (current fit)'] = (sorted(r['I']), sorted(r['M']))
    old_fit = json.loads((INTEG / 'opt1_rawslope' / 'regions.json').read_text())

    d.af.pth_res = fc.OA_RES
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
                ax.set_ylabel(f'{title}\nprior distance − shuffle mean', fontsize=9)
                ax.legend(fontsize=7, frameon=False, loc='upper left')
    for ext in ('svg', 'png'):
        fig.savefig(OUT / f'panel_i_old_rules.{ext}', dpi=200, facecolor='white')
    plt.close(fig)

    def summ(y):
        return {'mean': float(np.mean(y)), 'first5': float(np.mean(y[:5])),
                'last5': float(np.mean(y[-5:]))}
    cur_c, cur_m = set(old_fit['I']), set(old_fit['M'])
    info = {
        'S': d.overlap, 'S_list': sorted(old_fit['S']),
        'C_old': c_old, 'M_old': m_old, 'n_C_old': len(c_old), 'n_M_old': len(m_old),
        'C_old_minus_opt1': sorted(set(c_old) - cur_c), 'opt1_minus_C_old': sorted(cur_c - set(c_old)),
        'M_old_minus_opt1': sorted(set(m_old) - cur_m), 'opt1_minus_M_old': sorted(cur_m - set(m_old)),
        'C_old_and_M_old': sorted(set(c_old) & set(m_old)),
        'S_and_C_old': sorted(set(old_fit['S']) & set(c_old)),
        'S_and_M_old': sorted(set(old_fit['S']) & set(m_old)),
        'curves': {k: {g: {tf: summ(y) for tf, y in v.items()} for g, v in s.items()}
                   for k, s in curves.items()},
    }
    (OUT / 'panel_i_old_rules.json').write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k not in ('C_old', 'curves')}, indent=1))
    for k, s in info['curves'].items():
        print(k, {g: {tf.split('_')[0]: round(v[tf]['mean'], 4) for tf in v} for g, v in s.items()})
    print('wrote', OUT / 'panel_i_old_rules.png')


if __name__ == '__main__':
    main()
