#!/usr/bin/env python
"""
Count perseveration / late-session exclusions over all BWM sessions and plot
distributions.

Uses ``bwm_tables/trials.pqt`` (or rebuilds from insertion_cache / ONE).
Perseveration = **tail** of same-choice run ≥ ``min_run`` poorly explained by
non-0 contrast stim (keep first ``min_run - 1`` trials; see
``block_analysis_allsplits.perseveration_run_mask``).

  conda activate iblenv
  python scripts/analyze_perseveration_counts.py
  python scripts/analyze_perseveration_counts.py \\
    --cache-dir ~/Downloads/ONE/openalyx.internationalbrainlab.org
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import exclusion helpers without triggering a live Alyx ONE() connect.
import one.api as _one_api  # noqa: E402
_one_api.ONE = lambda *a, **k: type('ONE', (), {'cache_dir': Path('/tmp')})()
import block_analysis_allsplits as ba  # noqa: E402
ba.STICKY_LATE_FRAC = getattr(ba, 'STICKY_LATE_FRAC', 0.2)
ba.STICKY_MIN_RUN = getattr(ba, 'STICKY_MIN_RUN', 10)


def _session_frame(df: pd.DataFrame, eid: str, use_bwm_include: bool) -> pd.DataFrame | None:
    d = df[df['eid'] == eid]
    if use_bwm_include and 'bwm_include' in d.columns:
        d = d[d['bwm_include'].astype(bool)]
    if len(d) < 10:
        return None
    # Temporal order as stored (aggregate table is already trial order per eid).
    return d.reset_index(drop=True)


def _counts_for_session(trials: pd.DataFrame, late_frac: float, min_run: int) -> dict:
    drop, info = ba.sticky_trial_exclusion_mask(
        trials, late_frac=late_frac, min_run=min_run)
    choice = np.asarray(trials['choice'].to_numpy(), dtype=float)
    valid = np.isin(choice, [-1.0, 1.0])
    return {
        **info,
        'n_valid_choice': int(valid.sum()),
        'frac_perseveration': (
            info['n_perseveration'] / info['n_trials']
            if info['n_trials'] else np.nan),
        'frac_late': (
            info['n_late'] / info['n_trials'] if info['n_trials'] else np.nan),
        'frac_drop': (
            info['n_drop'] / info['n_trials'] if info['n_trials'] else np.nan),
        'frac_pers_among_valid': (
            info['n_perseveration'] / valid.sum() if valid.sum() else np.nan),
    }


def load_trials_table(cache_dir: Path) -> pd.DataFrame:
    pqt = cache_dir / 'bwm_tables' / 'trials.pqt'
    if not pqt.exists():
        raise SystemExit(
            f'Missing {pqt}. Point --cache-dir at an ONE cache with '
            'bwm_tables/trials.pqt (openalyx BWM release).')
    return pd.read_parquet(pqt)


def plot_distributions(rows: pd.DataFrame, out_dir: Path, min_run: int,
                       late_frac: float) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax = axes[0, 0]
    ax.hist(rows['n_perseveration'], bins=40, color='steelblue', edgecolor='k',
            linewidth=0.3)
    ax.axvline(rows['n_perseveration'].median(), color='crimson', ls='--',
               label=f"median={rows['n_perseveration'].median():.0f}")
    ax.set_xlabel(
        f'# perseveration tail trials (run≥{min_run}, keep first {min_run - 1})')
    ax.set_ylabel('# sessions')
    ax.set_title('Perseveration trial counts')
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.hist(rows['frac_perseveration'], bins=40, color='steelblue',
            edgecolor='k', linewidth=0.3, range=(0, 1))
    ax.axvline(rows['frac_perseveration'].median(), color='crimson', ls='--',
               label=f"median={rows['frac_perseveration'].median():.3f}")
    ax.set_xlabel('Fraction of session trials in perseveration')
    ax.set_ylabel('# sessions')
    ax.set_title('Perseveration fraction')
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    ax.hist(rows['n_drop'], bins=40, color='darkseagreen', edgecolor='k',
            linewidth=0.3)
    ax.axvline(rows['n_drop'].median(), color='crimson', ls='--',
               label=f"median={rows['n_drop'].median():.0f}")
    ax.set_xlabel(f'# dropped (late {late_frac:.0%} ∪ perseveration)')
    ax.set_ylabel('# sessions')
    ax.set_title('Total exclusion counts')
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.hist(rows['frac_drop'], bins=40, color='darkseagreen', edgecolor='k',
            linewidth=0.3, range=(0, 1))
    ax.axvline(rows['frac_drop'].median(), color='crimson', ls='--',
               label=f"median={rows['frac_drop'].median():.3f}")
    ax.set_xlabel('Fraction dropped (late ∪ perseveration)')
    ax.set_ylabel('# sessions')
    ax.set_title('Total exclusion fraction')
    ax.legend(fontsize=9)

    fig.suptitle(
        f'BWM sessions (n={len(rows)}): sticky exclusion diagnostics',
        fontsize=12)
    fig.tight_layout()
    fig_path = out_dir / 'perseveration_exclusion_distributions.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--cache-dir',
        default=os.environ.get(
            'ONE_CACHE_DIR',
            str(Path.home() / 'Downloads/ONE/openalyx.internationalbrainlab.org'),
        ))
    p.add_argument('--late-frac', type=float, default=ba.STICKY_LATE_FRAC)
    p.add_argument('--min-run', type=int, default=ba.STICKY_MIN_RUN)
    p.add_argument('--no-bwm-include', action='store_true',
                   help='Do not filter to bwm_include==True')
    p.add_argument('--out-dir', default=None,
                   help='Default: <cache>/manifold/choice_epoch_diag')
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = (Path(args.out_dir) if args.out_dir
               else cache_dir / 'manifold' / 'choice_epoch_diag')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading trials from {cache_dir / "bwm_tables" / "trials.pqt"} …')
    df = load_trials_table(cache_dir)
    eids = df['eid'].astype(str).unique()
    print(f'{len(df)} trials, {len(eids)} sessions')

    rows = []
    n_skip = 0
    for i, eid in enumerate(eids):
        sess = _session_frame(df, eid, use_bwm_include=not args.no_bwm_include)
        if sess is None:
            n_skip += 1
            continue
        rec = _counts_for_session(sess, args.late_frac, args.min_run)
        rec['eid'] = eid
        rows.append(rec)
        if (i + 1) % 50 == 0:
            print(f'  {i + 1}/{len(eids)} …', flush=True)

    rows = pd.DataFrame(rows)
    csv_path = out_dir / 'perseveration_exclusion_by_session.csv'
    rows.to_csv(csv_path, index=False)

    # Summary stats
    def _summ(col):
        x = rows[col].to_numpy(dtype=float)
        return {
            'mean': float(np.nanmean(x)),
            'median': float(np.nanmedian(x)),
            'p25': float(np.nanpercentile(x, 25)),
            'p75': float(np.nanpercentile(x, 75)),
            'min': float(np.nanmin(x)),
            'max': float(np.nanmax(x)),
        }

    print(f'\nSessions scored: {len(rows)} (skipped {n_skip})')
    print(f'min_run={args.min_run}, late_frac={args.late_frac}, '
          f'bwm_include={not args.no_bwm_include}')
    for col, label in [
        ('n_perseveration', '# perseveration trials'),
        ('frac_perseveration', 'frac perseveration'),
        ('n_late', '# late trials'),
        ('n_drop', '# dropped (union)'),
        ('frac_drop', 'frac dropped'),
        ('n_keep', '# kept'),
    ]:
        s = _summ(col)
        print(f'  {label}: median={s["median"]:.3g}  '
              f'mean={s["mean"]:.3g}  '
              f'IQR=[{s["p25"]:.3g}, {s["p75"]:.3g}]  '
              f'range=[{s["min"]:.3g}, {s["max"]:.3g}]')

    n_zero_pers = int((rows['n_perseveration'] == 0).sum())
    print(f'  sessions with 0 perseveration trials: '
          f'{n_zero_pers}/{len(rows)} ({100 * n_zero_pers / max(len(rows), 1):.1f}%)')

    fig_path = plot_distributions(rows, out_dir, args.min_run, args.late_frac)
    print(f'\nWrote {csv_path}')
    print(f'Wrote {fig_path}')


if __name__ == '__main__':
    main()
