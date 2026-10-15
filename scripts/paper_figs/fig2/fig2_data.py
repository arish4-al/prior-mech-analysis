#!/usr/bin/env python
"""Figure 2 data: region types, Sigma / monotonicity and per-region p-values.

For each version in fig2_common.VERSIONS:
  1. stage the combined act splits (stim from openalyx; choice from the
     version's choice res),
  2. run analysis_functions.get_sc_table with the analysis_figs.ipynb
     parameters (p_mean_c, alpha 0.01, slope 0.05, n 20),
  3. write <fig2>/<version>/sc_table.csv and pvals.csv,
  4. print the status of the a-c example regions and non-significant
     replacement candidates.

  conda activate iblenv   # outside the sandbox
  python scripts/paper_figs/fig2/fig2_data.py [--versions openalyx min5]
"""
import argparse
import json

import numpy as np
import pandas as pd

import fig2_common as fc

EXAMPLES = {   # published fig2 regions: (significant, non-significant)
    'a': (('stim_duringstim1_act',), 'VISp', 'MOs'),
    'b': (('stim_duringstim_act', 'stim_duringchoice_act'), 'VeCB', 'MOs'),
    'c': (('choice_duringstim_act', 'choice_duringchoice_act'), 'GRN', 'RSPd'),
}


def sc_table(af):
    orig = af.manifold_to_csv

    def _manifold_to_csv(split, sigl, p_type, sample=True):
        df = orig(split, sigl, p_type, sample=sample)
        if 'amp_last5_is_global_max' in df.columns:
            df['amp_last5_is_global_max'] = df['amp_last5_is_global_max'].fillna(0).astype(int)
        return df

    af.manifold_to_csv = _manifold_to_csv
    res = af.get_sc_table(fc.SC_TIMES, fc.PTYPE, alpha=fc.ALPHA, combined_p=True,
                          sc_threshold=fc.SC_THRESHOLD, slope_threshold=fc.SLOPE_THRESHOLD,
                          amp_loc_threshold=fc.AMP_LOC_THRESHOLD, n=fc.N_SLOPE,
                          stim_restr=True)
    res = res.copy()
    res['sigma_stim_s'] = 1 - res['sc_duringstim']
    res['sigma_stim_m'] = 1 - res['sc_duringchoice']
    res['monotonicity'] = res['sc_duringchoice_move_shape'].fillna(0).astype(int)
    return res


def pval_table(af):
    rows = {}
    for tf in fc.SC_TIMES + fc.STIM_EXTRA:
        name, _ = fc.combined_stems(af, tf)
        d = np.load(af.pth_res / f'{name}.npy', allow_pickle=True).item()
        rows[tf] = pd.Series({r: d[r][fc.PTYPE] for r in d})
        rows[f'{tf}_amp'] = pd.Series({r: d[r]['amp_euc'] for r in d})
    df = pd.DataFrame(rows)
    df.index.name = 'region'
    return df


def example_report(pv):
    out = {}
    for panel, (tfs, sig, nonsig) in EXAMPLES.items():
        ps = {reg: [float(pv.loc[reg, tf]) for tf in tfs] for reg in (sig, nonsig)}
        ok_sig = all(p <= fc.ALPHA for p in ps[sig])
        ok_non = all(p > 0.05 for p in ps[nonsig])
        others = pv[list(tfs)].dropna()
        cand = others[(others > 0.3).all(axis=1)]
        cand = cand.assign(pmin=cand.min(axis=1)).sort_values('pmin', ascending=False)
        out[panel] = {'timeframes': tfs, 'sig': sig, 'nonsig': nonsig, 'p': ps,
                      'sig_holds': ok_sig, 'nonsig_holds': ok_non,
                      'nonsig_candidates': cand.index[:25].tolist()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--versions', nargs='+', default=list(fc.VERSIONS))
    args = ap.parse_args()
    af = fc.load_af(fc.OA_RES)
    for version in args.versions:
        stage = fc.build_staging(af, version)
        af.pth_res = stage
        af.meta_pth = stage
        res = sc_table(af)
        pv = pval_table(af)
        vdir = fc.version_dir(version)
        res.to_csv(vdir / 'sc_table.csv', index=False)
        pv.to_csv(vdir / 'pvals.csv')
        rep = example_report(pv)
        (vdir / 'examples_check.json').write_text(json.dumps(rep, indent=2))
        print(f'== {version}: {fc.VERSIONS[version]["desc"]}')
        for col in ('sc_duringstim_regtype', 'sc_duringchoice_regtype'):
            print(f'  {col}:', res[col].value_counts(dropna=True).sort_index().to_dict())
        print('  monotonicity=1:', int(res['monotonicity'].sum()))
        for panel, r in rep.items():
            print(f'  {panel}: {r["sig"]} {r["p"][r["sig"]]} holds={r["sig_holds"]} | '
                  f'{r["nonsig"]} {r["p"][r["nonsig"]]} holds={r["nonsig_holds"]}')
            if not r['nonsig_holds']:
                print('     non-sig candidates:', ', '.join(r['nonsig_candidates']))
        print('  wrote', vdir)


if __name__ == '__main__':
    main()
