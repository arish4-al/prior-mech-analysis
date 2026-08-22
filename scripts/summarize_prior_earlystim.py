#!/usr/bin/env python
"""Slice existing duringstim prior-distance curves to the early-stim window.

The BWM ``act_block_duringstim*`` analyses were binned on ``[0, 150]`` ms
(72 bins). This recomputes ``p_mean`` / BH-FDR on the ``t <= t_max`` prefix
(default 80 ms, matching canonical sim S / ``SHORT_DURINGSTIM_WINDOW_S``).

Does **not** re-bin spikes. Per-bin Euclidean distance is independent of later
bins, so the prefix of a 150 ms curve is the early-stim distance. Does **not**
overwrite ``res/`` combines; writes a small CSV under ``meta/``.

  conda activate iblenv
  python scripts/summarize_prior_earlystim.py --alpha 0.01
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

WINDOW_MS = 150.0
DEFAULT_T_MAX_MS = 80.0

SPLIT_SHUFFLE = [
    'act_block_duringstim_r_choice_r_f1',
    'act_block_duringstim_l_choice_l_f1',
    'act_block_duringstim_l_choice_r_f2',
    'act_block_duringstim_r_choice_l_f2',
]
UNSPLIT_SHUFFLE = ['act_block_duringstim_l', 'act_block_duringstim_r']
SPLIT_HARRIS = [f'{s}_harris_unique' for s in SPLIT_SHUFFLE]
UNSPLIT_HARRIS = [f'{s}_harris_unique' for s in UNSPLIT_SHUFFLE]


def _combined_name(splits: list[str]) -> str:
    return 'combined_' + '_'.join(splits)

# 08-17 unsplit combine used l then r; Jul 14 four-split used f1/f2 order above.
ARMS = [
    ('split', 'shuffle', SPLIT_SHUFFLE),
    ('split', 'harris_unique', SPLIT_HARRIS),
    ('unsplit', 'shuffle', UNSPLIT_SHUFFLE),
    ('unsplit', 'harris_unique', UNSPLIT_HARRIS),
]


def _default_res() -> Path:
    return Path.home() / (
        'Downloads/ONE/alyx.internationalbrainlab.org/manifold/res/new'
    )


def _default_meta() -> Path:
    return Path.home() / 'Downloads/ONE/alyx.internationalbrainlab.org/meta'


def n_bins_le_tmax(n_bins: int, window_ms: float, t_max_ms: float) -> int:
    """Keep bins whose linspace(0, window_ms, n_bins) time is <= t_max_ms."""
    if n_bins < 1:
        return 0
    t = np.linspace(0.0, float(window_ms), int(n_bins))
    return int(np.sum(t <= float(t_max_ms)))


def _obs_nulls(entry) -> tuple[np.ndarray, np.ndarray]:
    """Accept combined [obs, nulls] or stacked (1+U, T) per-split arrays."""
    if isinstance(entry, (list, tuple)) and len(entry) == 2:
        obs = np.asarray(entry[0], dtype=float).reshape(-1)
        nulls = np.asarray(entry[1], dtype=float)
        if nulls.ndim == 1:
            nulls = nulls.reshape(1, -1)
        return obs, nulls
    stacked = np.asarray(entry, dtype=float)
    if stacked.ndim != 2 or stacked.shape[0] < 2:
        raise ValueError(f'expected (1+U, T) or [obs, nulls], got {stacked.shape}')
    return stacked[0], stacked[1:]


def stats_from_curves(obs: np.ndarray, nulls: np.ndarray, n_keep: int | None):
    if n_keep is not None:
        obs = np.asarray(obs, dtype=float)[:n_keep]
        nulls = np.asarray(nulls, dtype=float)[:, :n_keep]
    else:
        obs = np.asarray(obs, dtype=float)
        nulls = np.asarray(nulls, dtype=float)
    stacked = np.concatenate([obs.reshape(1, -1), nulls], axis=0)
    p_mean = float(np.mean(np.mean(stacked, axis=1) >= np.mean(stacked[0])))
    d_euc = obs - np.min(obs)
    amp_euc = float(np.max(d_euc))
    return {
        'p_mean': p_mean,
        'amp_euc': amp_euc,
        'n_null': int(nulls.shape[0]),
        'n_bins': int(obs.shape[0]),
    }


def load_combined_regde(pth_res: Path, disk_splits: list[str]) -> dict:
    path = pth_res / f'combined_regde_{"_".join(disk_splits)}.npy'
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True).item()


def summarize_arm(regde: dict, n_keep: int | None, alpha: float) -> dict:
    rows = {}
    for reg, entry in regde.items():
        obs, nulls = _obs_nulls(entry)
        rows[reg] = stats_from_curves(obs, nulls, n_keep)
    pvals = np.array([rows[r]['p_mean'] for r in rows], dtype=float)
    _, p_c, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
    for i, reg in enumerate(rows):
        rows[reg]['p_mean_c'] = float(p_c[i])
    ps = np.array([rows[r]['p_mean'] for r in rows])
    pcs = np.array([rows[r]['p_mean_c'] for r in rows])
    amps = np.array([rows[r]['amp_euc'] for r in rows])
    fdr_hits = sorted(r for r in rows if rows[r]['p_mean_c'] <= alpha)
    return {
        'n_reg': len(rows),
        'n_bins': next(iter(rows.values()))['n_bins'] if rows else 0,
        'n_null_med': float(np.median([rows[r]['n_null'] for r in rows])) if rows else np.nan,
        'uncorr_le_0.01': int(np.sum(ps <= 0.01)),
        'uncorr_le_0.05': int(np.sum(ps <= 0.05)),
        'fdr_le_alpha': int(np.sum(pcs <= alpha)),
        'fdr_le_0.01': int(np.sum(pcs <= 0.01)),
        'fdr_le_0.05': int(np.sum(pcs <= 0.05)),
        'median_p': float(np.median(ps)) if len(ps) else np.nan,
        'median_amp': float(np.median(amps)) if len(amps) else np.nan,
        'fdr_hits': fdr_hits,
        'per_reg': rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--res', type=Path, default=_default_res())
    ap.add_argument('--meta-dir', type=Path, default=_default_meta())
    ap.add_argument('--t-max-ms', type=float, default=DEFAULT_T_MAX_MS)
    ap.add_argument('--window-ms', type=float, default=WINDOW_MS)
    ap.add_argument('--alpha', type=float, default=0.01)
    args = ap.parse_args()

    print(f'res={args.res}')
    print(f'early-stim = t <= {args.t_max_ms:g} ms of the {args.window_ms:g} ms '
          f'duringstim curves (linspace time axis)')

    summary_rows = []
    hit_rows = []
    for cond, null, splits in ARMS:
        path = args.res / f'combined_regde_{"_".join(splits)}.npy'
        print(f'\n=== {cond} / {null} ===')
        print(f'  {path.name}  exists={path.exists()}')
        if not path.exists():
            continue
        regde = load_combined_regde(args.res, splits)
        sample = _obs_nulls(next(iter(regde.values())))[0]
        n_full = int(sample.shape[0])
        n_keep = n_bins_le_tmax(n_full, args.window_ms, args.t_max_ms)
        t_axis = np.linspace(0.0, args.window_ms, n_full)
        t_last = float(t_axis[n_keep - 1]) if n_keep else np.nan
        print(f'  n_bins {n_full} → {n_keep}  (last kept t={t_last:.1f} ms)')

        for label, keep in (
            (f'{args.window_ms:g}ms', None),
            (f'{args.t_max_ms:g}ms_earlystim', n_keep),
        ):
            s = summarize_arm(regde, keep, args.alpha)
            print(
                f'  {label}: nreg={s["n_reg"]}  bins={s["n_bins"]}  '
                f'uncorr≤0.01={s["uncorr_le_0.01"]}  ≤0.05={s["uncorr_le_0.05"]}  '
                f'FDR@{args.alpha:g}={s["fdr_le_alpha"]}  '
                f'FDR@0.01={s["fdr_le_0.01"]}  FDR@0.05={s["fdr_le_0.05"]}  '
                f'median p={s["median_p"]:.3f}  median amp={s["median_amp"]:.3f}'
            )
            if s['fdr_hits'] and (keep is not None or args.alpha >= 0.05):
                hits = ', '.join(s['fdr_hits'][:20])
                extra = '' if len(s['fdr_hits']) <= 20 else f' … +{len(s["fdr_hits"])-20}'
                print(f'    FDR@{args.alpha:g} hits: {hits}{extra}')
            summary_rows.append({
                'conditioning': cond,
                'null': null,
                'window': label,
                'n_reg': s['n_reg'],
                'n_bins': s['n_bins'],
                'uncorr_le_0.01': s['uncorr_le_0.01'],
                'uncorr_le_0.05': s['uncorr_le_0.05'],
                'fdr_0.01': s['fdr_le_0.01'],
                'fdr_0.05': s['fdr_le_0.05'],
                'median_p': s['median_p'],
                'median_amp': s['median_amp'],
                'combined': _combined_name(splits),
            })
            if keep is not None:
                for reg, rec in s['per_reg'].items():
                    hit_rows.append({
                        'conditioning': cond,
                        'null': null,
                        'region': reg,
                        'p_mean': rec['p_mean'],
                        'p_mean_c': rec['p_mean_c'],
                        'amp_euc': rec['amp_euc'],
                        'n_null': rec['n_null'],
                    })
        del regde

    args.meta_dir.mkdir(parents=True, exist_ok=True)
    stem = f'table_act_block_earlystim_{args.t_max_ms:g}ms'
    csv_path = args.meta_dir / f'{stem}_summary.csv'
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    hits_path = args.meta_dir / f'{stem}_p_mean.csv'
    pd.DataFrame(hit_rows).to_csv(hits_path, index=False)
    print(f'\nWrote {csv_path}')
    print(f'Wrote {hits_path}')


if __name__ == '__main__':
    main()
