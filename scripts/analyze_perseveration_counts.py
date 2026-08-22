#!/usr/bin/env python
"""
Count perseveration / late-session exclusions over all BWM sessions, report
overlap of the two drop rules, and compare behaviour in the last 20 % vs the
early 80 % (and vs perseveration-tail trials).

Uses ``bwm_tables/trials.pqt``. Perseveration = **tail** of same-choice run
≥ ``min_run`` poorly explained by non-0 contrast stim (keep first
``min_run - 1`` trials; see
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
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import exclusion helpers without triggering a live Alyx ONE() connect.
import one.api as _one_api  # noqa: E402
_one_api.ONE = lambda *a, **k: type('ONE', (), {'cache_dir': Path('/tmp')})()
import block_analysis_allsplits as ba  # noqa: E402
ba.STICKY_LATE_FRAC = getattr(ba, 'STICKY_LATE_FRAC', 0.2)
ba.STICKY_MIN_RUN = getattr(ba, 'STICKY_MIN_RUN', 10)

HIGH_CONTRAST = 0.25
EASY_CONTRAST = 1.0

SLICE_PREFIXES = ('early', 'late', 'pers', 'late_not_pers', 'pers_not_late')
BEHAV_KEYS = (
    'n', 'frac_nochoice', 'acc', 'acc_c0', 'acc_high', 'acc_easy',
    'n_c0', 'n_high', 'n_easy', 'rt_median', 'rt_iqr', 'frac_rt_missing',
    'n_block', 'n_block_high', 'n_block_low',
    'frac_block_match', 'frac_block_match_high', 'frac_block_match_low',
)


def _session_frame(df: pd.DataFrame, eid: str, use_bwm_include: bool) -> pd.DataFrame | None:
    d = df[df['eid'] == eid]
    if use_bwm_include and 'bwm_include' in d.columns:
        d = d[d['bwm_include'].astype(bool)]
    if len(d) < 10:
        return None
    # Temporal order as stored (aggregate table is already trial order per eid).
    return d.reset_index(drop=True)


def _contrast_mag(trials: pd.DataFrame) -> np.ndarray:
    cl = np.asarray(trials['contrastLeft'].to_numpy(), dtype=float)
    cr = np.asarray(trials['contrastRight'].to_numpy(), dtype=float)
    stim_is_left = np.isnan(cr)
    mag = np.zeros(len(trials), dtype=float)
    mag[stim_is_left] = np.nan_to_num(cl[stim_is_left], nan=0.0)
    mag[~stim_is_left] = np.nan_to_num(cr[~stim_is_left], nan=0.0)
    return mag


def _run_lengths(choices: np.ndarray) -> np.ndarray:
    choices = np.asarray(choices)
    if len(choices) == 0:
        return np.array([], dtype=int)
    change = np.flatnonzero(choices[1:] != choices[:-1]) + 1
    bounds = np.concatenate(([0], change, [len(choices)]))
    return np.diff(bounds)


def _lag1(choices: np.ndarray) -> float:
    if len(choices) < 3:
        return np.nan
    x = choices[:-1].astype(float)
    y = choices[1:].astype(float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _calendar_stickiness(choices: np.ndarray, prefix: str, suffix: str = '') -> dict:
    """Run lengths on a contiguous calendar slice (no stim×block skip)."""
    ch = np.asarray(choices, dtype=float)
    ch = ch[np.isin(ch, [-1.0, 1.0])]
    tag = f'{prefix}{suffix}'
    out = {
        f'{tag}_mean_run': np.nan,
        f'{tag}_lag1': np.nan,
        f'{tag}_frac_run5': np.nan,
        f'{tag}_frac_run10': np.nan,
        f'{tag}_n_stick': int(len(ch)),
    }
    if len(ch) == 0:
        return out
    rl = _run_lengths(ch)
    out[f'{tag}_mean_run'] = float(np.mean(rl))
    out[f'{tag}_lag1'] = _lag1(ch)
    out[f'{tag}_frac_run5'] = float(np.sum(rl[rl >= 5]) / len(ch))
    out[f'{tag}_frac_run10'] = float(np.sum(rl[rl >= 10]) / len(ch))
    return out


def _prior_align(choice: np.ndarray, prior_bin: np.ndarray,
                 mask: np.ndarray, key: str) -> dict:
    """P(choice = prior side) on valid ±1 trials in mask. prior_bin is 0.8/0.2."""
    choice = np.asarray(choice, dtype=float)
    prior_bin = np.asarray(prior_bin, dtype=float)
    mask = np.asarray(mask, dtype=bool) & np.isin(choice, [-1.0, 1.0])
    if not mask.any():
        return {key: np.nan}
    ch = choice[mask]
    pr = prior_bin[mask]
    match = ((np.isclose(pr, 0.8) & (ch == 1.0))
             | (np.isclose(pr, 0.2) & (ch == -1.0)))
    return {key: float(np.mean(match))}


N_QUINTILES = 5
MIN_TRIALS_PER_QUINTILE = 10


def _quintile_masks(n_tr: int, nobias: np.ndarray, n_groups: int = N_QUINTILES,
                    min_per_q: int = MIN_TRIALS_PER_QUINTILE):
    """Masks for equal-count quintiles of the drop-0.5 sequence (temporal order)."""
    idx = np.flatnonzero(np.asarray(nobias, dtype=bool))
    n = len(idx)
    if n < n_groups * min_per_q:
        return None, n
    edges = np.linspace(0, n, n_groups + 1).astype(int)
    masks = []
    for q in range(n_groups):
        m = np.zeros(n_tr, dtype=bool)
        m[idx[edges[q]:edges[q + 1]]] = True
        masks.append(m)
    return masks, n


def _quintile_rec(trials, choice, pleft, act_bin, n_groups=N_QUINTILES,
                  min_per_q=MIN_TRIALS_PER_QUINTILE) -> dict:
    nobias = ~np.isclose(np.asarray(pleft, dtype=float), 0.5)
    masks, n_nobias = _quintile_masks(len(choice), nobias, n_groups, min_per_q)
    rec = {'n_nobias': int(n_nobias), 'n_quintiles_ok': int(masks is not None)}
    empty = {
        'n': 0, 'acc': np.nan, 'rt_median': np.nan,
        'frac_block_match': np.nan, 'frac_act_match': np.nan,
        'mean_run': np.nan, 'lag1': np.nan,
        'frac_run5': np.nan, 'frac_run10': np.nan,
    }
    for q in range(n_groups):
        prefix = f'q{q + 1}'
        if masks is None:
            for k, v in empty.items():
                rec[f'{prefix}_{k}'] = v
            continue
        m = masks[q]
        beh = _behavior_for_mask(trials, m, prefix)
        rec[f'{prefix}_n'] = beh[f'{prefix}_n']
        rec[f'{prefix}_acc'] = beh[f'{prefix}_acc']
        rec[f'{prefix}_rt_median'] = beh[f'{prefix}_rt_median']
        rec[f'{prefix}_frac_block_match'] = beh[f'{prefix}_frac_block_match']
        rec.update(_prior_align(choice, act_bin, m, f'{prefix}_frac_act_match'))
        rec.update(_calendar_stickiness(choice[m], prefix))
    return rec


def _nan_behav(prefix: str) -> dict:
    return {f'{prefix}_{k}': (0 if k == 'n' or k.startswith('n_') else np.nan)
            for k in BEHAV_KEYS}


def _behavior_for_mask(trials: pd.DataFrame, mask: np.ndarray, prefix: str) -> dict:
    """Accuracy / RT / no-choice on a boolean trial mask (session temporal order)."""
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    if n == 0:
        return _nan_behav(prefix)

    sub = trials.iloc[np.where(mask)[0]]
    choice = np.asarray(sub['choice'].to_numpy(), dtype=float)
    fb = np.asarray(sub['feedbackType'].to_numpy(), dtype=float)
    mag = _contrast_mag(sub)
    valid = np.isin(choice, [-1.0, 1.0])
    scored = valid & np.isfinite(fb)

    out = {
        f'{prefix}_n': n,
        f'{prefix}_frac_nochoice': float((~valid).mean()),
    }
    out[f'{prefix}_acc'] = (
        float(np.mean(fb[scored] == 1.0)) if scored.any() else np.nan)

    c0 = scored & (mag == 0.0)
    high = scored & (mag >= HIGH_CONTRAST)
    easy = scored & (np.isclose(mag, EASY_CONTRAST))
    out[f'{prefix}_n_c0'] = int(c0.sum())
    out[f'{prefix}_n_high'] = int(high.sum())
    out[f'{prefix}_n_easy'] = int(easy.sum())
    out[f'{prefix}_acc_c0'] = (
        float(np.mean(fb[c0] == 1.0)) if c0.any() else np.nan)
    out[f'{prefix}_acc_high'] = (
        float(np.mean(fb[high] == 1.0)) if high.any() else np.nan)
    out[f'{prefix}_acc_easy'] = (
        float(np.mean(fb[easy] == 1.0)) if easy.any() else np.nan)

    pleft = np.asarray(sub['probabilityLeft'].to_numpy(), dtype=float)
    block_left = np.isclose(pleft, 0.8)
    block_right = np.isclose(pleft, 0.2)
    biased = (block_left | block_right) & valid
    match = (block_left & (choice == 1.0)) | (block_right & (choice == -1.0))
    high_b = biased & (mag >= HIGH_CONTRAST)
    low_b = biased & (mag < HIGH_CONTRAST)
    out[f'{prefix}_n_block'] = int(biased.sum())
    out[f'{prefix}_n_block_high'] = int(high_b.sum())
    out[f'{prefix}_n_block_low'] = int(low_b.sum())
    out[f'{prefix}_frac_block_match'] = (
        float(np.mean(match[biased])) if biased.any() else np.nan)
    out[f'{prefix}_frac_block_match_high'] = (
        float(np.mean(match[high_b])) if high_b.any() else np.nan)
    out[f'{prefix}_frac_block_match_low'] = (
        float(np.mean(match[low_b])) if low_b.any() else np.nan)

    stim = np.asarray(sub['stimOn_times'].to_numpy(), dtype=float)
    move = np.asarray(sub['firstMovement_times'].to_numpy(), dtype=float)
    rt = move - stim
    finite_rt = np.isfinite(rt) & (rt > 0)
    out[f'{prefix}_frac_rt_missing'] = float((~finite_rt).mean())
    if finite_rt.any():
        rt_ok = rt[finite_rt]
        out[f'{prefix}_rt_median'] = float(np.median(rt_ok))
        q25, q75 = np.percentile(rt_ok, [25, 75])
        out[f'{prefix}_rt_iqr'] = float(q75 - q25)
    else:
        out[f'{prefix}_rt_median'] = np.nan
        out[f'{prefix}_rt_iqr'] = np.nan
    return out


def _counts_for_session(trials: pd.DataFrame, late_frac: float, min_run: int) -> dict:
    _drop, info = ba.sticky_trial_exclusion_mask(
        trials, late_frac=late_frac, min_run=min_run)
    choice = np.asarray(trials['choice'].to_numpy(), dtype=float)
    valid = np.isin(choice, [-1.0, 1.0])
    n_tr = info['n_trials']
    n_pers = info['n_perseveration']
    n_both = info['n_drop_both']
    n_pers_only = info['n_drop_pers_only']

    if n_pers == 0:
        pers_loc = 'none'
        frac_pers_in_late = np.nan
    elif n_pers_only == 0:
        pers_loc = 'all_late'
        frac_pers_in_late = 1.0
    elif n_both == 0:
        pers_loc = 'all_early'
        frac_pers_in_late = 0.0
    else:
        pers_loc = 'mixed'
        frac_pers_in_late = n_both / n_pers

    rec = {
        **info,
        'n_valid_choice': int(valid.sum()),
        'frac_perseveration': n_pers / n_tr if n_tr else np.nan,
        'frac_late': info['n_late'] / n_tr if n_tr else np.nan,
        'frac_drop': info['n_drop'] / n_tr if n_tr else np.nan,
        'frac_pers_among_valid': (
            n_pers / valid.sum() if valid.sum() else np.nan),
        'frac_pers_in_late': frac_pers_in_late,
        'frac_extra_drop': n_pers_only / n_tr if n_tr else np.nan,
        'pers_location': pers_loc,
    }

    late = ba.late_session_mask(n_tr, late_frac=late_frac)
    cr = np.asarray(trials['contrastRight'].to_numpy(), dtype=float)
    pers = ba.perseveration_run_mask(
        choice, np.isnan(cr), _contrast_mag(trials), min_run=min_run)
    slices = {
        'early': ~late,
        'late': late,
        'pers': pers,
        'late_not_pers': late & ~pers,
        'pers_not_late': pers & ~late,
    }
    for prefix, m in slices.items():
        rec.update(_behavior_for_mask(trials, m, prefix))
    # Calendar-order stickiness only on contiguous early/late windows
    # (pers / late_not_pers are not contiguous — concatenating would glue runs).
    pleft = np.asarray(trials['probabilityLeft'].to_numpy(), dtype=float)
    rec.update(_calendar_stickiness(choice[~late], 'early'))
    rec.update(_calendar_stickiness(choice[late], 'late'))
    rec.update(_calendar_stickiness(choice[(~late) & ~np.isclose(pleft, 0.5)],
                                    'early', suffix='_nobias'))
    rec.update(_calendar_stickiness(choice[late & ~np.isclose(pleft, 0.5)],
                                    'late', suffix='_nobias'))
    rec.update(_behavior_for_mask(
        trials, (~late) & ~np.isclose(pleft, 0.5), 'early_nobias'))
    rec.update(_behavior_for_mask(
        trials, late & ~np.isclose(pleft, 0.5), 'late_nobias'))
    # Action-kernel (α=0.2) on the full session choice sequence, then alignment
    # scored on the same drop-0.5 early/late windows as the true-block row.
    actions = np.nan_to_num(choice, nan=0.0)
    _, act_bin = ba.action_kernel_priors(0.2, list(actions))
    act_bin = np.asarray(act_bin, dtype=float)
    rec.update(_prior_align(choice, act_bin, (~late) & ~np.isclose(pleft, 0.5),
                            'early_nobias_frac_act_match'))
    rec.update(_prior_align(choice, act_bin, late & ~np.isclose(pleft, 0.5),
                            'late_nobias_frac_act_match'))
    rec.update(_quintile_rec(trials, choice, pleft, act_bin, n_groups=5,
                             min_per_q=10))
    return rec


def load_trials_table(cache_dir: Path) -> pd.DataFrame:
    pqt = cache_dir / 'bwm_tables' / 'trials.pqt'
    if not pqt.exists():
        raise SystemExit(
            f'Missing {pqt}. Point --cache-dir at an ONE cache with '
            'bwm_tables/trials.pqt (openalyx BWM release).')
    return pd.read_parquet(pqt)


def _summ(x) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {k: np.nan for k in ('mean', 'median', 'p25', 'p75', 'min', 'max', 'n')}
    return {
        'mean': float(np.mean(x)),
        'median': float(np.median(x)),
        'p25': float(np.percentile(x, 25)),
        'p75': float(np.percentile(x, 75)),
        'min': float(np.min(x)),
        'max': float(np.max(x)),
        'n': int(x.size),
    }


def _fmt(s: dict, digits: str = '.3g') -> str:
    return (
        f'median={s["median"]:{digits}}  mean={s["mean"]:{digits}}  '
        f'IQR=[{s["p25"]:{digits}}, {s["p75"]:{digits}}]  '
        f'range=[{s["min"]:{digits}}, {s["max"]:{digits}}]  n={s["n"]}'
    )


def _paired(rows: pd.DataFrame, a: str, b: str) -> dict:
    """Paired a-minus-b (e.g. late − early). Wilcoxon two-sided on finite pairs."""
    da = rows[a].to_numpy(dtype=float)
    db = rows[b].to_numpy(dtype=float)
    d = da - db
    ok = np.isfinite(d)
    d_ok = d[ok]
    out = {
        'n': int(ok.sum()),
        'median_delta': float(np.median(d_ok)) if ok.any() else np.nan,
        'mean_delta': float(np.mean(d_ok)) if ok.any() else np.nan,
        'frac_a_lower': float(np.mean(d_ok < 0)) if ok.any() else np.nan,
        'frac_a_higher': float(np.mean(d_ok > 0)) if ok.any() else np.nan,
        'wilcoxon_p': np.nan,
    }
    if ok.sum() >= 10 and np.any(d_ok != 0):
        try:
            out['wilcoxon_p'] = float(
                stats.wilcoxon(d_ok, alternative='two-sided', zero_method='wilcox').pvalue)
        except ValueError:
            out['wilcoxon_p'] = np.nan
    return out


def _paired_line(tag: str, rows: pd.DataFrame, a: str, b: str, worse: str,
                 frac_label: str = 'frac worse') -> str:
    p = _paired(rows, a, b)
    a_med = float(np.nanmedian(rows[a].to_numpy(dtype=float)))
    b_med = float(np.nanmedian(rows[b].to_numpy(dtype=float)))
    frac = p['frac_a_lower'] if worse == 'a_lower' else p['frac_a_higher']
    return (
        f'  {tag}: median {a_med:.3g} vs {b_med:.3g}; '
        f'Δ median={p["median_delta"]:.3g}  {frac_label}={frac:.1%}  '
        f'Wilcoxon p={p["wilcoxon_p"]:.3g}  n={p["n"]}'
    )


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


def plot_overlap(rows: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    ax = axes[0]
    x = rows['frac_pers_in_late'].to_numpy(dtype=float)
    ax.hist(x[np.isfinite(x)], bins=20, range=(0, 1), color='steelblue',
            edgecolor='k', linewidth=0.3)
    ax.axvline(np.nanmedian(x), color='crimson', ls='--',
               label=f'median={np.nanmedian(x):.2f}')
    ax.set_xlabel('Fraction of sticky-tail trials in last 20%')
    ax.set_ylabel('# sessions')
    ax.set_title('Where sticky tails sit')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(rows['n_drop_pers_only'], bins=30, color='darkorange',
            edgecolor='k', linewidth=0.3)
    ax.axvline(rows['n_drop_pers_only'].median(), color='crimson', ls='--',
               label=f"median={rows['n_drop_pers_only'].median():.0f}")
    ax.set_xlabel('# sticky tails outside last 20%')
    ax.set_ylabel('# sessions')
    ax.set_title('Extra drop beyond late window')
    ax.legend(fontsize=8)

    ax = axes[2]
    loc = rows['pers_location'].value_counts()
    order = ['none', 'all_late', 'mixed', 'all_early']
    counts = [int(loc.get(k, 0)) for k in order]
    ax.bar(order, counts, color=['0.7', 'steelblue', 'darkseagreen', 'darkorange'])
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('# sessions')
    ax.set_title('Sticky-tail location vs last 20%')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=20)

    fig.suptitle(f'BWM sessions (n={len(rows)}): sticky ∩ late overlap', fontsize=12)
    fig.tight_layout()
    fig_path = out_dir / 'sticky_late_overlap.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


def _box_group(ax, rows: pd.DataFrame, cols: list[str], labels: list[str],
               ylabel: str, title: str, ylim=None):
    data = [rows[c].to_numpy(dtype=float) for c in cols]
    data = [d[np.isfinite(d)] for d in data]
    try:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    except TypeError:
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    colors = ['0.75', 'steelblue', 'darkorange', 'darkseagreen']
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(axis='x', rotation=20)


def plot_performance(rows: pd.DataFrame, out_dir: Path) -> Path:
    cols = ['early', 'late', 'late_not_pers', 'pers']
    labels = ['early 80%', 'last 20%', 'late not pers', 'pers tail']
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    _box_group(axes[0, 0], rows, [f'{c}_acc' for c in cols], labels,
               'P(correct)', 'Accuracy (all contrasts)', (0.4, 1.0))
    _box_group(axes[0, 1], rows, [f'{c}_acc_c0' for c in cols], labels,
               'P(correct)', 'Accuracy (0% contrast)', (0.2, 1.0))
    _box_group(axes[0, 2], rows, [f'{c}_acc_high' for c in cols], labels,
               'P(correct)', f'Accuracy (|c|≥{HIGH_CONTRAST:g})', (0.4, 1.0))
    _box_group(axes[1, 0], rows, [f'{c}_acc_easy' for c in cols], labels,
               'P(correct)', 'Accuracy (c=1)', (0.4, 1.0))
    _box_group(axes[1, 1], rows, [f'{c}_rt_median' for c in cols], labels,
               'RT (s)', 'Median RT (move − stimOn)', (0, 1.2))
    _box_group(axes[1, 2], rows, [f'{c}_frac_nochoice' for c in cols], labels,
               'Fraction', 'No-choice (choice=0)', (0, 0.15))

    fig.suptitle(
        f'BWM sessions (n={len(rows)}): early 80% vs last 20% vs perseveration',
        fontsize=12)
    fig.tight_layout()
    fig_path = out_dir / 'sticky_late_performance.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


def plot_performance_deltas(rows: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    specs = [
        ('acc', 'Δ accuracy (all)', axes[0, 0]),
        ('acc_c0', 'Δ accuracy (0% contrast)', axes[0, 1]),
        ('acc_high', f'Δ accuracy (|c|≥{HIGH_CONTRAST:g})', axes[0, 2]),
        ('acc_easy', 'Δ accuracy (c=1)', axes[1, 0]),
        ('rt_median', 'Δ median RT (s)', axes[1, 1]),
        ('frac_nochoice', 'Δ no-choice fraction', axes[1, 2]),
    ]
    for key, title, ax in specs:
        d_late = rows[f'late_{key}'] - rows[f'early_{key}']
        d_lnp = rows[f'late_not_pers_{key}'] - rows[f'early_{key}']
        d_pers = rows[f'pers_{key}'] - rows[f'early_{key}']
        data = [
            d_late.to_numpy(dtype=float),
            d_lnp.to_numpy(dtype=float),
            d_pers.to_numpy(dtype=float),
        ]
        data = [d[np.isfinite(d)] for d in data]
        tick = ['late−early', '(late not pers)−early', 'pers−early']
        try:
            bp = ax.boxplot(data, tick_labels=tick, patch_artist=True, showfliers=False)
        except TypeError:
            bp = ax.boxplot(data, labels=tick, patch_artist=True, showfliers=False)
        for patch, c in zip(bp['boxes'], ['steelblue', 'darkseagreen', 'darkorange']):
            patch.set_facecolor(c)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=15)
    fig.suptitle(
        f'BWM sessions (n={len(rows)}): paired deltas vs early 80%',
        fontsize=12)
    fig.tight_layout()
    fig_path = out_dir / 'sticky_late_performance_deltas.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


def plot_block_match(rows: pd.DataFrame, out_dir: Path) -> Path:
    cols = ['early', 'late', 'late_not_pers', 'pers']
    labels = ['early 80%', 'last 20%', 'late not pers', 'pers tail']
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    _box_group(axes[0, 0], rows, [f'{c}_frac_block_match' for c in cols], labels,
               'P(choice = block side)', 'All contrasts (biased blocks)', (0.4, 1.0))
    _box_group(axes[0, 1], rows,
               [f'{c}_frac_block_match_high' for c in cols], labels,
               'P(choice = block side)', f'High contrast (|c|≥{HIGH_CONTRAST:g})',
               (0.4, 1.0))
    _box_group(axes[0, 2], rows,
               [f'{c}_frac_block_match_low' for c in cols], labels,
               'P(choice = block side)', f'Low contrast (|c|<{HIGH_CONTRAST:g})',
               (0.3, 1.0))

    specs = [
        ('frac_block_match', 'Δ all contrasts', axes[1, 0]),
        ('frac_block_match_high', f'Δ high (|c|≥{HIGH_CONTRAST:g})', axes[1, 1]),
        ('frac_block_match_low', f'Δ low (|c|<{HIGH_CONTRAST:g})', axes[1, 2]),
    ]
    for key, title, ax in specs:
        data = [
            (rows[f'late_{key}'] - rows[f'early_{key}']).to_numpy(dtype=float),
            (rows[f'late_not_pers_{key}'] - rows[f'early_{key}']).to_numpy(dtype=float),
            (rows[f'pers_{key}'] - rows[f'early_{key}']).to_numpy(dtype=float),
        ]
        data = [d[np.isfinite(d)] for d in data]
        tick = ['late−early', '(late not pers)−early', 'pers−early']
        try:
            bp = ax.boxplot(data, tick_labels=tick, patch_artist=True, showfliers=False)
        except TypeError:
            bp = ax.boxplot(data, labels=tick, patch_artist=True, showfliers=False)
        for patch, c in zip(bp['boxes'], ['steelblue', 'darkseagreen', 'darkorange']):
            patch.set_facecolor(c)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=15)
        ax.set_ylabel('Δ P(match)')

    fig.suptitle(
        f'BWM sessions (n={len(rows)}): choice matches true-block side',
        fontsize=12)
    fig.tight_layout()
    fig_path = out_dir / 'sticky_late_block_match.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


def plot_calendar_stickiness(rows: pd.DataFrame, out_dir: Path) -> Path:
    labels = ['early 80%', 'last 20%']
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    specs = [
        ('mean_run', 'mean run length', axes[0, 0], None),
        ('lag1', 'lag-1 choice corr', axes[0, 1], (-0.2, 1.0)),
        ('frac_run5', 'frac trials in runs ≥5', axes[0, 2], (0, 1)),
        ('nobias_mean_run', 'mean run (drop pLeft=0.5)', axes[1, 0], None),
        ('nobias_lag1', 'lag-1 (drop pLeft=0.5)', axes[1, 1], (-0.2, 1.0)),
        ('nobias_frac_run5', 'frac run≥5 (drop 0.5)', axes[1, 2], (0, 1)),
    ]
    for key, title, ax, ylim in specs:
        _box_group(ax, rows, [f'early_{key}', f'late_{key}'], labels,
                   title, title, ylim)
    fig.suptitle(
        f'BWM sessions (n={len(rows)}): calendar-order choice stickiness '
        '(no stim×block skip)',
        fontsize=12)
    fig.tight_layout()
    fig_path = out_dir / 'sticky_late_calendar_stickiness.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


Q_METRIC_LABELS = [
    ('frac_block_match', 'block alignment'),
    ('frac_act_match', 'act-kernel alignment'),
    ('rt_median', 'median RT (s)'),
    ('acc', 'P(correct)'),
    ('mean_run', 'mean run length'),
    ('lag1', 'lag-1 choice corr'),
    ('frac_run5', 'frac in runs ≥5'),
    ('frac_run10', 'frac in runs ≥10'),
]


def plot_quintiles(rows: pd.DataFrame, out_dir: Path) -> Path:
    ok = rows[rows['n_quintiles_ok'] == 1]
    labels = [f'Q{i}' for i in range(1, 6)]
    colors = ['0.8', '0.7', '0.6', 'steelblue', 'darkorange']
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (key, title) in zip(axes.ravel(), Q_METRIC_LABELS):
        data = [ok[f'q{i}_{key}'].to_numpy(dtype=float) for i in range(1, 6)]
        data = [d[np.isfinite(d)] for d in data]
        try:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                            showfliers=False)
        except TypeError:
            bp = ax.boxplot(data, labels=labels, patch_artist=True,
                            showfliers=False)
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
        ax.set_title(title)
        ax.axhline(np.nanmedian(ok[f'q5_{key}'].to_numpy(dtype=float)),
                   color='darkorange', ls=':', lw=0.8)
    fig.suptitle(
        f'BWM sessions (n={len(ok)}): post-0.5 quintiles '
        '(Q5 = last 20% of biased-block trials)',
        fontsize=12)
    fig.tight_layout()
    fig_path = out_dir / 'sticky_post05_quintiles.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


def _q5_vs_prev(rows: pd.DataFrame, key: str) -> dict:
    """Q5 vs the four earlier quintiles (per-session)."""
    ok = rows[rows['n_quintiles_ok'] == 1]
    q5 = ok[f'q5_{key}'].to_numpy(dtype=float)
    prev = np.column_stack(
        [ok[f'q{i}_{key}'].to_numpy(dtype=float) for i in range(1, 5)])
    finite_row = np.isfinite(q5) & np.all(np.isfinite(prev), axis=1)
    q5 = q5[finite_row]
    prev = prev[finite_row]
    if len(q5) == 0:
        return {'n': 0, 'median_q5': np.nan, 'median_prev_med': np.nan,
                'median_delta': np.nan, 'frac_gt_prev_med': np.nan,
                'frac_gt_prev_max': np.nan, 'wilcoxon_p': np.nan}
    prev_med = np.median(prev, axis=1)
    prev_max = np.max(prev, axis=1)
    d = q5 - prev_med
    pval = np.nan
    if len(d) >= 10 and np.any(d != 0):
        try:
            pval = float(stats.wilcoxon(
                d, alternative='two-sided', zero_method='wilcox').pvalue)
        except ValueError:
            pval = np.nan
    return {
        'n': int(len(q5)),
        'median_q5': float(np.median(q5)),
        'median_prev_med': float(np.median(prev_med)),
        'median_delta': float(np.median(d)),
        'frac_gt_prev_med': float(np.mean(q5 > prev_med)),
        'frac_gt_prev_max': float(np.mean(q5 > prev_max)),
        'wilcoxon_p': pval,
    }


def write_summary(lines: list[str], out_dir: Path) -> Path:
    path = out_dir / 'sticky_late_overlap_performance_summary.txt'
    path.write_text('\n'.join(lines) + '\n')
    return path


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

    lines: list[str] = []

    def log(msg: str = ''):
        print(msg)
        lines.append(msg)

    log(f'Sessions scored: {len(rows)} (skipped {n_skip})')
    log(f'min_run={args.min_run}, late_frac={args.late_frac}, '
        f'bwm_include={not args.no_bwm_include}')
    log('')
    log('=== 1. Exclusion counts and overlap ===')
    for col, label in [
        ('n_perseveration', '# perseveration tail trials'),
        ('frac_perseveration', 'frac perseveration'),
        ('n_late', '# late trials'),
        ('n_drop_late_only', '# late-only (not pers)'),
        ('n_drop_pers_only', '# pers-only (not late)'),
        ('n_drop_both', '# both (pers ∩ late)'),
        ('n_drop', '# dropped (union)'),
        ('frac_drop', 'frac dropped'),
        ('frac_pers_in_late', 'frac of pers tails that sit in last 20%'),
        ('frac_extra_drop', 'frac session that is pers-only (extra drop)'),
        ('n_keep', '# kept'),
    ]:
        log(f'  {label}: {_fmt(_summ(rows[col]))}')

    n_zero_pers = int((rows['n_perseveration'] == 0).sum())
    n_has_pers = int((rows['n_perseveration'] > 0).sum())
    loc = rows['pers_location'].value_counts()
    log(f'  sessions with 0 perseveration trials: '
        f'{n_zero_pers}/{len(rows)} ({100 * n_zero_pers / max(len(rows), 1):.1f}%)')
    log(f'  among sessions with pers tails (n={n_has_pers}):')
    for k, desc in [
        ('all_late', 'entirely inside last 20%'),
        ('mixed', 'spread across early and late'),
        ('all_early', 'entirely outside last 20%'),
    ]:
        n = int(loc.get(k, 0))
        log(f'    {desc}: {n}/{n_has_pers} '
            f'({100 * n / max(n_has_pers, 1):.1f}%)')

    log('')
    log('=== 2. Behaviour: last 20% vs early 80% (and pers tails) ===')
    log('Paired per session; Δ = first named slice minus early. '
        '"frac late worse" = fraction of sessions with lower accuracy / '
        'higher RT / higher no-choice in the late (or pers) slice.')
    for metric, label, late_worse in [
        ('acc', 'accuracy (all contrasts)', 'a_lower'),
        ('acc_c0', 'accuracy (0% contrast)', 'a_lower'),
        ('acc_high', f'accuracy (|c|≥{HIGH_CONTRAST:g})', 'a_lower'),
        ('acc_easy', 'accuracy (c=1)', 'a_lower'),
        ('rt_median', 'median RT (s)', 'a_higher'),
        ('frac_nochoice', 'no-choice fraction', 'a_higher'),
    ]:
        log('')
        log(f'-- {label} --')
        log(_paired_line(
            'late vs early', rows, f'late_{metric}', f'early_{metric}', late_worse))
        log(_paired_line(
            'late not pers vs early', rows,
            f'late_not_pers_{metric}', f'early_{metric}', late_worse))
        log(_paired_line(
            'pers vs early', rows, f'pers_{metric}', f'early_{metric}', late_worse))

    log('')
    log('=== 3. Choice matches true-block side (pLeft 0.8/0.2; drop 0.5) ===')
    log('P(choice = block side) among valid ±1 choices on biased blocks. '
        'High = |c|≥0.25; low = |c|<0.25 (includes 0). '
        '"frac late higher" = sessions with more block-aligned choices late.')
    for metric, label in [
        ('frac_block_match', 'block-match (all contrasts)'),
        ('frac_block_match_high', f'block-match (|c|≥{HIGH_CONTRAST:g})'),
        ('frac_block_match_low', f'block-match (|c|<{HIGH_CONTRAST:g})'),
    ]:
        log('')
        log(f'-- {label} --')
        log(_paired_line(
            'late vs early', rows, f'late_{metric}', f'early_{metric}',
            'a_higher', frac_label='frac late higher'))
        log(_paired_line(
            'late not pers vs early', rows,
            f'late_not_pers_{metric}', f'early_{metric}',
            'a_higher', frac_label='frac late higher'))
        log(_paired_line(
            'pers vs early', rows, f'pers_{metric}', f'early_{metric}',
            'a_higher', frac_label='frac late higher'))
    for col, label in [
        ('early_n_block', '# biased-block trials (early)'),
        ('late_n_block', '# biased-block trials (late)'),
        ('early_n_block_low', '# low-c biased (early)'),
        ('late_n_block_low', '# low-c biased (late)'),
        ('early_n_block_high', '# high-c biased (early)'),
        ('late_n_block_high', '# high-c biased (late)'),
    ]:
        log(f'  {label}: {_fmt(_summ(rows[col]))}')

    log('')
    log('=== 4. Calendar-order choice stickiness (no stim×block skip) ===')
    log('Run lengths on valid ±1 choices in session temporal order inside '
        'the contiguous early-80% / last-20% windows. nobias = drop pLeft=0.5 '
        'inside the window (0.5 lives in early).')
    for metric, label in [
        ('mean_run', 'mean run length (all valid choices)'),
        ('lag1', 'lag-1 choice corr (all valid)'),
        ('frac_run5', 'frac trials in runs ≥5 (all valid)'),
        ('frac_run10', 'frac trials in runs ≥10 (all valid)'),
        ('nobias_mean_run', 'mean run (drop pLeft=0.5)'),
        ('nobias_lag1', 'lag-1 (drop pLeft=0.5)'),
        ('nobias_frac_run5', 'frac run≥5 (drop 0.5)'),
        ('nobias_frac_run10', 'frac run≥10 (drop 0.5)'),
    ]:
        log('')
        log(f'-- {label} --')
        log(_paired_line(
            'late vs early', rows, f'late_{metric}', f'early_{metric}',
            'a_higher', frac_label='frac late higher'))

    log('')
    log('=== 5. Overall table: early vs late, drop pLeft=0.5 ===')
    log(f'Last {args.late_frac:.0%} of the full bwm_include session vs the '
        f'complementary early window; pLeft=0.5 dropped inside each window. '
        'Cell values = median across sessions. p = paired Wilcoxon. '
        'frac late > early = sign count.')
    log('metric\tearly\tlate\tΔ median\tfrac late > early\tp')
    for metric, label, worse in [
        ('nobias_frac_block_match', 'block alignment', 'a_higher'),
        ('nobias_frac_act_match', 'act-kernel alignment', 'a_higher'),
        ('nobias_rt_median', 'median RT (s)', 'a_higher'),
        ('nobias_acc', 'P(correct)', 'a_higher'),
        ('nobias_mean_run', 'mean run length', 'a_higher'),
        ('nobias_lag1', 'lag-1 choice corr', 'a_higher'),
        ('nobias_frac_run5', 'frac in runs ≥5', 'a_higher'),
        ('nobias_frac_run10', 'frac in runs ≥10', 'a_higher'),
    ]:
        pstat = _paired(rows, f'late_{metric}', f'early_{metric}')
        a_med = float(np.nanmedian(rows[f'late_{metric}'].to_numpy(dtype=float)))
        b_med = float(np.nanmedian(rows[f'early_{metric}'].to_numpy(dtype=float)))
        line = (
            f'{label}\t{b_med:.4g}\t{a_med:.4g}\t{pstat["median_delta"]:.4g}\t'
            f'{pstat["frac_a_higher"]:.1%}\t{pstat["wilcoxon_p"]:.3g}'
        )
        log(line)

    log('')
    log('=== 6. Post-0.5 quintiles (Q1–Q4 vs last 20% = Q5) ===')
    n_ok = int((rows['n_quintiles_ok'] == 1).sum())
    log(f'Sessions with ≥{MIN_TRIALS_PER_QUINTILE} trials per quintile: '
        f'{n_ok}/{len(rows)}. Quintiles are equal-count slices of the '
        'drop-pLeft=0.5 sequence (not last 20% of the full session).')
    log('n trials per quintile (median / IQR):')
    for i in range(1, 6):
        log(f'  Q{i}: {_fmt(_summ(rows.loc[rows["n_quintiles_ok"]==1, f"q{i}_n"]))}')
    log('metric\tQ1\tQ2\tQ3\tQ4\tQ5\tQ5−med(Q1–4)\tfrac Q5>med(Q1–4)\t'
        'frac Q5>max(Q1–4)\tp Q5 vs med(Q1–4)')
    ok = rows[rows['n_quintiles_ok'] == 1]
    for key, label in Q_METRIC_LABELS:
        meds = [float(np.nanmedian(ok[f'q{i}_{key}'].to_numpy(dtype=float)))
                for i in range(1, 6)]
        vs = _q5_vs_prev(rows, key)
        log(
            f'{label}\t' + '\t'.join(f'{m:.4g}' for m in meds) +
            f'\t{vs["median_delta"]:.4g}\t{vs["frac_gt_prev_med"]:.1%}\t'
            f'{vs["frac_gt_prev_max"]:.1%}\t{vs["wilcoxon_p"]:.3g}'
        )
    log('Paired Wilcoxon Q5 vs each earlier quintile:')
    for key, label in Q_METRIC_LABELS:
        bits = []
        for i in range(1, 5):
            pstat = _paired(ok, f'q5_{key}', f'q{i}_{key}')
            bits.append(f'Q{i} p={pstat["wilcoxon_p"]:.3g} Δ={pstat["median_delta"]:.3g}')
        log(f'  {label}: ' + '; '.join(bits))

    fig_path = plot_distributions(rows, out_dir, args.min_run, args.late_frac)
    overlap_path = plot_overlap(rows, out_dir)
    perf_path = plot_performance(rows, out_dir)
    delta_path = plot_performance_deltas(rows, out_dir)
    match_path = plot_block_match(rows, out_dir)
    cal_path = plot_calendar_stickiness(rows, out_dir)
    q_path = plot_quintiles(rows, out_dir)
    summary_path = write_summary(lines, out_dir)
    print(f'\nWrote {csv_path}')
    print(f'Wrote {fig_path}')
    print(f'Wrote {overlap_path}')
    print(f'Wrote {perf_path}')
    print(f'Wrote {delta_path}')
    print(f'Wrote {match_path}')
    print(f'Wrote {cal_path}')
    print(f'Wrote {q_path}')
    print(f'Wrote {summary_path}')


if __name__ == '__main__':
    main()
