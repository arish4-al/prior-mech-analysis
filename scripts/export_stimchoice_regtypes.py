#!/usr/bin/env python
"""
Export stim×choice region types from local SC results into **repo** ``data/``
(for git push/pull to remote runs).

Local-only helper: reads openalyx ``manifold/res`` act splits. Remote fitting
does **not** use this script — it reads the committed CSV via
``scripts/run_var_partition.py --regtype-csv data/...``.

  conda activate iblenv
  python scripts/export_stimchoice_regtypes.py --skip-out-cache
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_OPENALYX = (
    Path.home() / 'Downloads' / 'ONE' / 'openalyx.internationalbrainlab.org'
)
DEFAULT_ALYX = Path.home() / 'Downloads' / 'ONE' / 'alyx.internationalbrainlab.org'

SC_TIMES = [
    'stim_duringstim_act',
    'choice_duringstim_act',
    'stim_duringchoice_act',
    'choice_duringchoice_act',
]

REGTYPE_LABEL = {
    0.0: 'stim',
    0.1: 'stim_early',
    0.5: 'integrator',
    1.0: 'move',
}


def _patch_one_local(openalyx_cache: Path):
    import one.api as one_api
    _real = one_api.ONE

    def _local_one(*args, **kwargs):
        return _real(
            cache_dir=str(openalyx_cache),
            mode='local',
            silent=True,
        )

    one_api.ONE = _local_one


def _sig_amp(af, timeframe: str, alpha: float, ptype: str) -> pd.Series:
    splits = af.run_align[timeframe]
    split_name = 'combined_' + '_'.join(splits)
    df = af.manifold_to_csv(split_name, alpha, ptype)
    amp = df['amp_euc'].to_numpy(dtype=float) * df['significant'].to_numpy(dtype=float)
    return pd.Series(amp, index=df['region'].to_numpy(), name=timeframe)


def _sigma(num_parts, choice: pd.Series) -> pd.Series:
    num = sum(num_parts)
    den = num + choice
    return (num / den).where(den > 0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--openalyx-cache-dir', type=Path, default=DEFAULT_OPENALYX)
    p.add_argument('--out-cache-dir', type=Path, default=DEFAULT_ALYX,
                   help='Optional local ONE meta write')
    p.add_argument('--ptype', default='p_mean_c')
    p.add_argument('--alpha', type=float, default=0.01)
    p.add_argument('--sc-threshold', type=float, default=0.0)
    p.add_argument('--slope-threshold', type=float, default=0.05)
    p.add_argument('--amp-loc-threshold', type=int, default=67)
    p.add_argument('--n', type=int, default=20)
    p.add_argument('--stim-restr', action='store_true', default=True)
    p.add_argument('--no-stim-restr', dest='stim_restr', action='store_false')
    p.add_argument('--sigma-threshold', type=float, default=0.8)
    p.add_argument('--repo-data-dir', type=Path, default=ROOT / 'data',
                   help='Git-tracked output dir (default: data/)')
    p.add_argument('--no-repo-data', action='store_true')
    p.add_argument('--skip-out-cache', action='store_true',
                   help='Only write repo data/ (recommended for committing)')
    p.add_argument('--recreate-table', action='store_true')
    p.add_argument('--copy-table-png', action='store_true')
    args = p.parse_args()

    openalyx = args.openalyx_cache_dir.expanduser().resolve()
    if not (openalyx / 'manifold' / 'res').exists():
        raise SystemExit(f'Missing openalyx manifold/res under {openalyx}')

    out_root = None
    if not args.skip_out_cache:
        out_root = args.out_cache_dir.expanduser().resolve()
        if 'openalyx' in str(out_root).lower():
            raise SystemExit(f'Refusing openalyx write path: {out_root}')

    _patch_one_local(openalyx)
    import analysis_functions as af  # noqa: E402

    print(f'Reading SC results from: {af.one.cache_dir}')

    table = af.get_sc_table(
        SC_TIMES, args.ptype, alpha=args.alpha, combined_p=True,
        sc_threshold=args.sc_threshold, slope_threshold=args.slope_threshold,
        amp_loc_threshold=args.amp_loc_threshold, n=args.n,
        stim_restr=args.stim_restr,
    )

    amp_stim_s = _sig_amp(af, 'stim_duringstim_act', args.alpha, args.ptype)
    amp_choice_s = _sig_amp(af, 'choice_duringstim_act', args.alpha, args.ptype)
    amp_stim_se = _sig_amp(af, 'stim_duringstim_short_act', args.alpha, args.ptype)
    amp_stim_se1 = _sig_amp(af, 'stim_duringstim1_act', args.alpha, args.ptype)

    out = table.copy().set_index('region')
    for name, ser in [
        ('amp_stim_s', amp_stim_s),
        ('amp_choice_s', amp_choice_s),
        ('amp_stim_se', amp_stim_se),
        ('amp_stim_se1', amp_stim_se1),
    ]:
        out[name] = ser.reindex(out.index).fillna(0.0)

    out['sigma_stim_s'] = _sigma(
        [out['amp_stim_s'], out['amp_stim_se']], out['amp_choice_s'])
    out['sigma_stim_s_prime'] = _sigma(
        [out['amp_stim_s'], out['amp_stim_se1']], out['amp_choice_s'])

    thr = args.sigma_threshold
    out['stim_processor'] = (out['sigma_stim_s'] > thr).fillna(False)
    out['stim_processor_loose'] = (out['sigma_stim_s_prime'] > thr).fillna(False)
    out['has_stim'] = (
        (out['amp_stim_s'] > 0)
        | (out['amp_stim_se'] > 0)
        | (out['amp_stim_se1'] > 0)
    )
    out['has_choice'] = out['amp_choice_s'] > 0
    out['mixed_stim_choice'] = out['has_stim'] & out['has_choice']
    out['duringstim_label'] = out['sc_duringstim_regtype'].map(REGTYPE_LABEL)
    out['duringchoice_label'] = out['sc_duringchoice_regtype'].map(REGTYPE_LABEL)
    out = out.reset_index()

    csv_name = f'stimchoice_act_regtype_regions_{args.ptype}_{args.alpha}.csv'
    written = []

    if out_root is not None:
        meta_out = out_root / 'meta'
        meta_out.mkdir(parents=True, exist_ok=True)
        csv_path = meta_out / csv_name
        out.to_csv(csv_path, index=False)
        written.append(csv_path)
        if args.recreate_table:
            af.meta_pth = meta_out
            af.plot_sc_table(
                SC_TIMES, args.ptype, alpha=args.alpha, metric='regtype',
                sc_threshold=args.sc_threshold,
                slope_threshold=args.slope_threshold,
                amp_loc_threshold=args.amp_loc_threshold, n=args.n,
                stim_restr=args.stim_restr,
            )
            print(f'Recreated table PNG under {meta_out}')
        if args.copy_table_png:
            src = (
                openalyx / 'meta'
                / f'table_stimchoice_act_regtype_{args.ptype}_{args.alpha}.png'
            )
            if src.exists():
                dst = meta_out / f'{src.stem}_openalyx_copy{src.suffix}'
                shutil.copy2(src, dst)

    if not args.no_repo_data:
        repo_data = args.repo_data_dir.expanduser().resolve()
        repo_data.mkdir(parents=True, exist_ok=True)
        repo_csv = repo_data / csv_name
        out.to_csv(repo_csv, index=False)
        mixed = sorted(out.loc[out['mixed_stim_choice'], 'region'].tolist())
        compact = repo_data / 'var_partition_mixed_stim_choice_regions.csv'
        pd.DataFrame({'region': mixed}).to_csv(compact, index=False)
        written.extend([repo_csv, compact])

    n_mixed = int(out['mixed_stim_choice'].sum())
    print(f'Wrote: {written}')
    print(
        f'  stim_processor={int(out.stim_processor.sum())}, '
        f'stim_processor_loose={int(out.stim_processor_loose.sum())}, '
        f'mixed_stim_choice={n_mixed}'
    )
    print('  mixed:', ', '.join(
        sorted(out.loc[out.mixed_stim_choice, 'region'].tolist())))


if __name__ == '__main__':
    main()
