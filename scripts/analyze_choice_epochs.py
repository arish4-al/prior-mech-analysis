#!/usr/bin/env python
"""
Check whether BWM choice sequences show one-sided response epochs,
especially late in the session (motivation for Harris session-permutation nulls).

Mean run length = mean number of consecutive trials with the same choice side.

Pipeline:
  1. Trial inclusion: aggregate ``bwm_include`` + same RT/NaN mask as
     ``load_trials_and_mask`` (use ``--prefer-alf`` when session ALF is cached).
  2. Drop unbiased (probabilityLeft == 0.5) trials — typically the session start.
  3. Keep choice ∈ {+1, −1} only.
  4. True *and* null stickiness scored **within** stim×block strata (each
     stratum's trials in temporal order, then pooled) so the comparison matches
     the ephys cell conditioning.
  5. Tertiles of remaining trials; nulls shuffle within stim×block (true-block
     and action-kernel α=0.2).

  python scripts/analyze_choice_epochs.py \\
    --cache-dir ~/Downloads/ONE/alyx.internationalbrainlab.org
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from one.api import ONE

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brainwidemap import bwm_query, load_trials_and_mask  # noqa: E402

ACT_ALPHA = 0.2


def run_lengths(choices: np.ndarray) -> np.ndarray:
    '''Lengths of consecutive same-side choice runs (±1).'''
    choices = np.asarray(choices)
    if len(choices) == 0:
        return np.array([], dtype=int)
    change = np.flatnonzero(choices[1:] != choices[:-1]) + 1
    bounds = np.concatenate(([0], change, [len(choices)]))
    return np.diff(bounds)


def lag1_autocorr(choices: np.ndarray) -> float:
    '''Pearson corr of consecutive choices coded ±1.'''
    if len(choices) < 3:
        return np.nan
    x = choices[:-1].astype(float)
    y = choices[1:].astype(float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def frac_in_long_runs(choices: np.ndarray, min_len: int = 5) -> float:
    '''Fraction of trials that fall inside a same-side run of length ≥ min_len.'''
    if len(choices) == 0:
        return np.nan
    rl = run_lengths(choices)
    # sum of lengths of long runs / n trials
    return float(np.sum(rl[rl >= min_len]) / len(choices))


def _empty_metrics(n: int = 0) -> dict:
    return {
        'mean_run': np.nan, 'median_run': np.nan, 'max_run': np.nan,
        'lag1': np.nan, 'frac_run5': np.nan, 'frac_run10': np.nan,
        'side_dom': np.nan, 'n': int(n),
    }


def choice_metrics(choices: np.ndarray) -> dict:
    '''Session-order stickiness (no stratum restriction).'''
    rl = run_lengths(choices)
    if len(rl) == 0:
        return _empty_metrics(0)
    p_left = float(np.mean(choices == 1))
    return {
        'mean_run': float(np.mean(rl)),
        'median_run': float(np.median(rl)),
        'max_run': float(np.max(rl)),
        'lag1': lag1_autocorr(choices),
        'frac_run5': frac_in_long_runs(choices, 5),
        'frac_run10': frac_in_long_runs(choices, 10),
        'side_dom': abs(2 * p_left - 1),
        'n': int(len(choices)),
    }


def choice_metrics_within_strata(
        choices: np.ndarray, strata: np.ndarray) -> dict:
    '''
    Stickiness within stim×block cells (matches ephys conditioning).

    For each non-empty stratum, take that stratum's choices in temporal order,
    compute runs / lag pairs there, then pool across strata.
    '''
    choices = np.asarray(choices)
    strata = np.asarray(strata)
    all_rl = []
    lag_x, lag_y = [], []
    n_total = 0
    n_long5 = 0
    n_long10 = 0
    side_choices = []

    for key in np.unique(strata):
        if key == '' or key is None:
            continue
        idx = np.flatnonzero(strata == key)
        if len(idx) == 0:
            continue
        ch = choices[idx].astype(float)
        side_choices.append(ch)
        n_total += len(ch)
        rl = run_lengths(ch)
        if len(rl):
            all_rl.append(rl)
            n_long5 += int(np.sum(rl[rl >= 5]))
            n_long10 += int(np.sum(rl[rl >= 10]))
        if len(ch) >= 2:
            lag_x.append(ch[:-1])
            lag_y.append(ch[1:])

    if n_total == 0 or not all_rl:
        return _empty_metrics(n_total)

    rl = np.concatenate(all_rl)
    if lag_x:
        x = np.concatenate(lag_x)
        y = np.concatenate(lag_y)
        if len(x) >= 2 and x.std() >= 1e-12 and y.std() >= 1e-12:
            lag1 = float(np.corrcoef(x, y)[0, 1])
        else:
            lag1 = np.nan
    else:
        lag1 = np.nan

    sc = np.concatenate(side_choices)
    p_left = float(np.mean(sc == 1))
    return {
        'mean_run': float(np.mean(rl)),
        'median_run': float(np.median(rl)),
        'max_run': float(np.max(rl)),
        'lag1': lag1,
        'frac_run5': float(n_long5 / n_total),
        'frac_run10': float(n_long10 / n_total),
        'side_dom': abs(2 * p_left - 1),
        'n': int(n_total),
    }


def tertile_slices(n: int) -> dict[str, slice]:
    if n < 9:
        return {}
    a, b = n // 3, 2 * n // 3
    return {
        'early': slice(0, a),
        'mid': slice(a, b),
        'late': slice(b, None),
    }


def action_kernel_binary(actions: np.ndarray, alpha: float = ACT_ALPHA) -> np.ndarray:
    '''Match block_analysis_allsplits.action_kernel_priors → 0.8/0.2 labels.'''
    prior = 0.5
    priors = [prior]
    for t in range(len(actions) - 1):
        prior = alpha * int(actions[t] > 0) + (1 - alpha) * prior
        priors.append(prior)
    binary = (np.asarray(priors, dtype=float) >= 0.5).astype(float)
    return binary * 0.6 + 0.2


def stim_side_labels(contrast_left, contrast_right) -> np.ndarray:
    ''' 'L' / 'R' / '' per trial.'''
    cl = np.asarray(contrast_left, dtype=float)
    cr = np.asarray(contrast_right, dtype=float)
    out = np.full(len(cl), '', dtype=object)
    left = ~np.isnan(cl)
    right = ~np.isnan(cr)
    out[left & ~right] = 'L'
    out[right & ~left] = 'R'
    # rare both present: prefer non-nan magnitude side (shouldn't happen in IBL)
    both = left & right
    out[both] = 'L'  # unused in practice
    return out


def block_side_labels(pleft: np.ndarray) -> np.ndarray:
    ''' 'L' if 0.8, 'R' if 0.2, else ''.'''
    p = np.asarray(pleft, dtype=float)
    out = np.full(len(p), '', dtype=object)
    out[np.isclose(p, 0.8)] = 'L'
    out[np.isclose(p, 0.2)] = 'R'
    return out


def strata_keys(stim: np.ndarray, block: np.ndarray) -> np.ndarray:
    keys = np.empty(len(stim), dtype=object)
    for i in range(len(stim)):
        if stim[i] and block[i]:
            keys[i] = f'{stim[i]}_{block[i]}'
        else:
            keys[i] = ''
    return keys


def stratified_choice_shuffle(choices: np.ndarray, strata: np.ndarray, rng) -> np.ndarray:
    '''Permute choices within each non-empty stim×block stratum.'''
    out = np.array(choices, copy=True, dtype=float)
    for key in np.unique(strata):
        if key == '' or key is None:
            continue
        idx = np.where(strata == key)[0]
        if len(idx) < 2:
            continue
        out[idx] = rng.permutation(out[idx])
    return out


def stratified_null_stats(
        choices: np.ndarray, strata: np.ndarray, nrand: int, rng,
        prefix: str, true_metrics: dict | None = None) -> dict:
    '''
    Null: shuffle choice labels within stim×block strata, then recompute
    within-strata mean_run / lag1 / frac_run5 (same metric as true).

    Returns mean and median over null draws. If ``true_metrics`` is given,
    also one-sided p-values P(null ≥ true) with +1 correction:
    ``(1 + count(null >= true)) / (1 + nrand)``.
    '''
    means, lags, fracs = [], [], []
    for _ in range(nrand):
        sh = stratified_choice_shuffle(choices, strata, rng)
        m = choice_metrics_within_strata(sh, strata)
        means.append(m['mean_run'])
        lags.append(m['lag1'])
        fracs.append(m['frac_run5'])
    means = np.asarray(means, dtype=float)
    lags = np.asarray(lags, dtype=float)
    fracs = np.asarray(fracs, dtype=float)
    out = {
        f'{prefix}_null_mean_run_mean': float(np.nanmean(means)),
        f'{prefix}_null_mean_run_median': float(np.nanmedian(means)),
        f'{prefix}_null_lag1_mean': float(np.nanmean(lags)),
        f'{prefix}_null_lag1_median': float(np.nanmedian(lags)),
        f'{prefix}_null_frac_run5_mean': float(np.nanmean(fracs)),
        f'{prefix}_null_frac_run5_median': float(np.nanmedian(fracs)),
        # aliases used by older print/plot code (= mean of null draws)
        f'{prefix}_null_mean_run': float(np.nanmean(means)),
        f'{prefix}_null_lag1': float(np.nanmean(lags)),
        f'{prefix}_null_frac_run5': float(np.nanmean(fracs)),
    }
    if true_metrics is not None:
        def _p(null_arr, true_val):
            if not np.isfinite(true_val):
                return np.nan
            return float((1 + np.nansum(null_arr >= true_val)) / (1 + nrand))

        out[f'{prefix}_p_mean_run'] = _p(means, true_metrics['mean_run'])
        out[f'{prefix}_p_lag1'] = _p(lags, true_metrics['lag1'])
        out[f'{prefix}_p_frac_run5'] = _p(fracs, true_metrics['frac_run5'])
    return out


def _pack_true(prefix: str, m: dict) -> dict:
    return {f'{prefix}_{k}': v for k, v in m.items()}


def prepare_session(df: pd.DataFrame) -> dict | None:
    '''
    Drop pLeft==0.5 and non-±1 choices; build true-block and act-kernel strata.
    Returns None if too few trials remain.
    '''
    d = df.copy()
    # Action kernel on full session order (incl. 0.5 / no-go) for EMA continuity
    actions_full = np.asarray(d['choice'].to_numpy(), dtype=float)
    actions_full = np.nan_to_num(actions_full, nan=0.0)
    act_pleft_full = action_kernel_binary(actions_full)

    # Analysis set: biased blocks + valid L/R choice
    pleft = np.asarray(d['probabilityLeft'].to_numpy(), dtype=float)
    choice = np.asarray(d['choice'].to_numpy(), dtype=float)
    keep = (~np.isclose(pleft, 0.5)) & np.isfinite(choice) & (np.abs(choice) == 1)
    n_keep = int(np.sum(keep))
    if n_keep < 30:
        return None

    d = d.loc[keep].reset_index(drop=True)
    choice = choice[keep]
    pleft = pleft[keep]
    act_pleft = act_pleft_full[keep]
    stim = stim_side_labels(d['contrastLeft'], d['contrastRight'])
    block_true = block_side_labels(pleft)
    block_act = block_side_labels(act_pleft)
    strata_true = strata_keys(stim, block_true)
    strata_act = strata_keys(stim, block_act)

    return {
        'choice': choice,
        'strata_true': strata_true,
        'strata_act': strata_act,
        'n': n_keep,
        'frac_stratified_true': float(np.mean(strata_true != '')),
        'frac_stratified_act': float(np.mean(strata_act != '')),
    }


def apply_bwm_style_mask(trials: pd.DataFrame) -> pd.Series:
    '''
    Replicate ``load_trials_and_mask`` defaults on a trials DataFrame
    (min_rt=0.08, max_rt=2, default nan_exclude). No truncate_to_pass.
    '''
    rt = (trials['firstMovement_times'].astype(float)
          - trials['stimOn_times'].astype(float))
    mask = (rt >= 0.08) & (rt <= 2.0)
    for col in (
        'stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
        'firstMovement_times', 'feedbackType',
    ):
        if col in trials.columns:
            mask = mask & trials[col].notna()
    return mask


def load_masked_trials(one, eid: str) -> pd.DataFrame | None:
    '''
    Same base trial inclusion as block_analysis_allsplits.get_d_vars:
    ``load_trials_and_mask(..., saturation_intervals=None)`` then keep mask==True.

    Saturation (stim/move/feedback) is split-specific in the neural pipeline and
    is not applied here.
    '''
    try:
        trials, mask = load_trials_and_mask(
            one, eid, saturation_intervals=None)
    except Exception as exc:  # noqa: BLE001 — skip broken eids, continue cohort
        print(f'  skip {eid}: load_trials_and_mask failed: {exc}', flush=True)
        return None
    return trials.loc[mask.to_numpy()].reset_index(drop=True)


def load_sessions_from_aggregate(
        one_root: Path, eids: list[str] | None = None,
        max_sessions: int | None = None) -> dict[str, pd.DataFrame]:
    '''
    Full-cohort path when session ALF is not cached: ``bwm_tables/trials.pqt``
    with ``bwm_include`` plus the same RT/NaN criteria as load_trials_and_mask.
    '''
    path = one_root / 'bwm_tables' / 'trials.pqt'
    if not path.exists():
        raise SystemExit(f'Missing aggregate trials table: {path}')
    df = pd.read_parquet(path)
    need = {
        'eid', 'choice', 'probabilityLeft', 'contrastLeft', 'contrastRight',
        'stimOn_times', 'firstMovement_times', 'feedback_times', 'feedbackType',
    }
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f'{path} missing columns {missing}')
    if 'bwm_include' in df.columns:
        df = df[df['bwm_include'] == True]  # noqa: E712
    df = df.loc[apply_bwm_style_mask(df)].copy()
    if eids is not None:
        eids_set = set(map(str, eids))
        df = df[df['eid'].astype(str).isin(eids_set)]
    out: dict[str, pd.DataFrame] = {}
    for eid, g in df.groupby(df['eid'].astype(str), sort=False):
        out[str(eid)] = g.reset_index(drop=True)
        if max_sessions is not None and len(out) >= max_sessions:
            break
    return out


def load_sessions_via_mask(
        one, eids: list[str] | None = None,
        max_sessions: int | None = None,
        prefer_alf: bool = False) -> dict[str, pd.DataFrame]:
    '''
    Default: aggregate ``bwm_include`` + BWM-style mask (all BWM eids offline).

    ``prefer_alf=True``: try ``load_trials_and_mask`` per eid (needs session ALF);
    fall back to aggregate for failures.
    '''
    one_root = Path(one.cache_dir)

    if not prefer_alf:
        print('Loading via bwm_tables/trials.pqt (bwm_include + RT/NaN mask) …',
              flush=True)
        return load_sessions_from_aggregate(
            one_root, eids=eids, max_sessions=max_sessions)

    if eids is None:
        eids = list(bwm_query(one)['eid'].astype(str).unique())
    if max_sessions is not None:
        eids = eids[:max_sessions]

    print('Loading via load_trials_and_mask (ALF); aggregate fallback …',
          flush=True)
    agg = load_sessions_from_aggregate(one_root, eids=None, max_sessions=None)
    out: dict[str, pd.DataFrame] = {}
    t0 = time.perf_counter()
    n_alf = 0
    for i, eid in enumerate(eids, 1):
        trials = load_masked_trials(one, eid)
        if trials is not None and len(trials):
            out[str(eid)] = trials
            n_alf += 1
        elif eid in agg:
            out[str(eid)] = agg[eid]
        if i % 25 == 0 or i == len(eids):
            print(
                f'  loaded {i}/{len(eids)} eids '
                f'({len(out)} ok, {n_alf} alf, {time.perf_counter() - t0:.1f}s)',
                flush=True,
            )
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--cache-dir',
        default=os.environ.get(
            'ONE_CACHE_DIR',
            str(Path.home() / 'Downloads/ONE/alyx.internationalbrainlab.org'),
        ),
    )
    p.add_argument('--nrand', type=int, default=200)
    p.add_argument('--out-dir', default=None,
                   help='Default: <cache>/manifold/choice_epoch_diag')
    p.add_argument('--max-sessions', type=int, default=None,
                   help='Optional cap for smoke tests')
    p.add_argument(
        '--one-mode', default='local',
        choices=('local', 'remote', 'auto'),
        help='ONE mode (default local = cache only)',
    )
    p.add_argument(
        '--prefer-alf', action='store_true',
        help='Use load_trials_and_mask when session ALF is cached '
             '(fallback: aggregate bwm_include)',
    )
    args = p.parse_args()

    one_root = Path(args.cache_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (
        one_root / 'manifold' / 'choice_epoch_diag')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'ONE cache_dir={one_root} mode={args.one_mode}', flush=True)
    one = ONE(cache_dir=str(one_root), mode=args.one_mode)
    sessions = load_sessions_via_mask(
        one, max_sessions=args.max_sessions, prefer_alf=args.prefer_alf)
    print(f'masked sessions: {len(sessions)} eids', flush=True)
    if not sessions:
        raise SystemExit('No sessions loaded')

    rng = np.random.default_rng(0)
    rows = []
    eids = list(sessions.items())
    t_loop = time.perf_counter()
    for i, (eid, sdf) in enumerate(eids, 1):
        prep = prepare_session(sdf)
        if prep is None:
            continue
        ch = prep['choice']
        st_block = prep['strata_true']
        st_act = prep['strata_act']
        m_block = choice_metrics_within_strata(ch, st_block)
        m_act = choice_metrics_within_strata(ch, st_act)
        row = {
            'eid': eid,
            'n_trials': prep['n'],
            'frac_stratified_true': prep['frac_stratified_true'],
            'frac_stratified_act': prep['frac_stratified_act'],
            **_pack_true('block_true', m_block),
            **_pack_true('act_true', m_act),
            # aliases: primary "true" = within stim×true-block
            **{f'true_{k}': v for k, v in m_block.items() if k != 'n'},
        }
        # Tertiles of post-0.5 trials; within-strata metrics per tertile
        for name, sl in tertile_slices(len(ch)).items():
            ch_p, sb_p, sa_p = ch[sl], st_block[sl], st_act[sl]
            mb = choice_metrics_within_strata(ch_p, sb_p)
            ma = choice_metrics_within_strata(ch_p, sa_p)
            row.update(_pack_true(f'{name}_block_true', mb))
            row.update(_pack_true(f'{name}_act_true', ma))
            # aliases for plots / late−early (true-block strata)
            for k, v in mb.items():
                row[f'{name}_{k}' if k != 'n' else f'{name}_n'] = v
            row.update(stratified_null_stats(
                ch_p, sb_p, args.nrand, rng, prefix=f'{name}_block',
                true_metrics=mb))
            row.update(stratified_null_stats(
                ch_p, sa_p, args.nrand, rng, prefix=f'{name}_act',
                true_metrics=ma))

        row.update(stratified_null_stats(
            ch, st_block, args.nrand, rng, prefix='block',
            true_metrics=m_block))
        row.update(stratified_null_stats(
            ch, st_act, args.nrand, rng, prefix='act',
            true_metrics=m_act))
        rows.append(row)
        if i % 50 == 0 or i == len(eids):
            elapsed = time.perf_counter() - t_loop
            print(f'  analyzed {i}/{len(eids)} sessions  ({elapsed:.1f}s)', flush=True)

    df = pd.DataFrame(rows)
    csv_path = out_dir / 'choice_epoch_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f'Saved {csv_path} ({len(df)} sessions)')
    print('Note: mean_run / lag1 = within stim×block strata (pooled), '
          'matching null')
    print(f'Null: {args.nrand} shuffles/session within stim×block strata; '
          'reported null = mean over shuffles; '
          'p = (1 + #{null≥true}) / (1+nrand)  [one-sided]')
    print('block = true probabilityLeft; act = action-kernel 0.8/0.2')

    def _med(col):
        return float(np.nanmedian(df[col])) if col in df.columns else np.nan

    def _sig_frac(pcol, alpha=0.01):
        if pcol not in df.columns:
            return np.nan
        p = df[pcol].to_numpy(dtype=float)
        p = p[np.isfinite(p)]
        return float(np.mean(p < alpha)) if len(p) else np.nan

    print('\n=== Within stim×block (median across sessions) ===')
    print(f'n_sessions={len(df)}  median n_trials={_med("n_trials"):.0f}  '
          f'nrand={args.nrand}')
    print(f"mean_run|block: true={_med('block_true_mean_run'):.2f}  "
          f"null={_med('block_null_mean_run_mean'):.2f}  "
          f"p<0.01={_sig_frac('block_p_mean_run'):.3f}  "
          f"(median p={_med('block_p_mean_run'):.4f})")
    print(f"mean_run|act:   true={_med('act_true_mean_run'):.2f}  "
          f"null={_med('act_null_mean_run_mean'):.2f}  "
          f"p<0.01={_sig_frac('act_p_mean_run'):.3f}  "
          f"(median p={_med('act_p_mean_run'):.4f})")
    print(f"lag1|block: true={_med('block_true_lag1'):.3f}  "
          f"null={_med('block_null_lag1_mean'):.3f}  "
          f"p<0.01={_sig_frac('block_p_lag1'):.3f}  "
          f"(median p={_med('block_p_lag1'):.4f})")
    print(f"lag1|act:   true={_med('act_true_lag1'):.3f}  "
          f"null={_med('act_null_lag1_mean'):.3f}  "
          f"p<0.01={_sig_frac('act_p_lag1'):.3f}  "
          f"(median p={_med('act_p_lag1'):.4f})")
    print(f"frac_run≥5|block: true={_med('block_true_frac_run5'):.3f}  "
          f"null={_med('block_null_frac_run5_mean'):.3f}  "
          f"p<0.01={_sig_frac('block_p_frac_run5'):.3f}")
    print(f"frac_run≥5|act:   true={_med('act_true_frac_run5'):.3f}  "
          f"null={_med('act_null_frac_run5_mean'):.3f}  "
          f"p<0.01={_sig_frac('act_p_frac_run5'):.3f}")

    print('\n--- Tertiles within stim×true-block (median) ---')
    for part in ('early', 'mid', 'late'):
        print(
            f"{part}: mean_run={_med(f'{part}_block_true_mean_run'):.2f}  "
            f"(null={_med(f'{part}_block_null_mean_run_mean'):.2f}; "
            f"p<0.01={_sig_frac(f'{part}_block_p_mean_run'):.2f})  "
            f"lag1={_med(f'{part}_block_true_lag1'):.3f}  "
            f"(null={_med(f'{part}_block_null_lag1_mean'):.3f}; "
            f"p<0.01={_sig_frac(f'{part}_block_p_lag1'):.2f})"
        )

    if 'early_mean_run' in df.columns and 'late_mean_run' in df.columns:
        d_run = df['late_mean_run'] - df['early_mean_run']
        d_lag = df['late_lag1'] - df['early_lag1']
        print('\n--- Late − early within stim×true-block (median [IQR]) ---')
        print(f"Δ mean_run: {np.nanmedian(d_run):.2f} "
              f"[{np.nanpercentile(d_run, 25):.2f}, {np.nanpercentile(d_run, 75):.2f}]  "
              f"frac late>early={np.nanmean(d_run > 0):.2f}")
        print(f"Δ lag1: {np.nanmedian(d_lag):.3f} "
              f"[{np.nanpercentile(d_lag, 25):.3f}, {np.nanpercentile(d_lag, 75):.3f}]  "
              f"frac late>early={np.nanmean(d_lag > 0):.2f}")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    axes[0].hist(df['block_true_mean_run'], bins=30, color='C0', alpha=0.75,
                 label='true within stim×block')
    axes[0].axvline(_med('block_null_mean_run'), color='C3', ls='--',
                    label='null stim×block')
    axes[0].axvline(_med('act_null_mean_run'), color='C2', ls=':',
                    label='null stim×act')
    axes[0].set_xlabel('mean run length (within strata)')
    axes[0].set_ylabel('# sessions')
    axes[0].legend(fontsize=7)
    axes[0].set_title('Within stim×block')

    axes[1].hist(df['block_true_lag1'], bins=30, color='C1', alpha=0.75)
    axes[1].axvline(_med('block_null_lag1'), color='C3', ls='--')
    axes[1].axvline(_med('act_null_lag1'), color='C2', ls=':')
    axes[1].set_xlabel('lag-1 (within strata)')
    axes[1].set_title('Stickiness')

    vals = [df['early_mean_run'], df['mid_mean_run'], df['late_mean_run']]
    axes[2].boxplot(vals, tick_labels=['early', 'mid', 'late'], showfliers=False)
    axes[2].set_ylabel('mean run length')
    axes[2].set_title('Tertiles (within stim×block)')
    fig.tight_layout()
    fig_path = out_dir / 'choice_epoch_summary.png'
    fig.savefig(fig_path, dpi=150)
    print(f'Saved {fig_path}')


if __name__ == '__main__':
    main()
