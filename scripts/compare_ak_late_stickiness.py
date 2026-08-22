#!/usr/bin/env python
"""Compare real BWM session stickiness vs ActionKernel ± per-session copy-last.

For each session: score post-0.5 quintile mean_run (and lag-1 / block-align /
accuracy) on (1) real choices, (2) stationary AK on the *real* stim stream,
(3) the same AK draw after a signed mix targeting that session's real μ_q
(copy-last if AK is too switchy, break-repeat if AK is too sticky).

Fixed stim so quintile windows match the real session. Default AK θ is the
analysis-kernel α=0.2 (not a per-eid MCMC fit).

  conda activate iblenv
  python scripts/compare_ak_late_stickiness.py
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from late_choice_stickiness import (  # noqa: E402
    N_QUINTILES,
    apply_late_stickiness,
    mean_run,
    post05_quintile_index,
    quintile_mean_run,
)


def load_trials_table(cache_dir: Path) -> pd.DataFrame:
    pqt = cache_dir / 'bwm_tables' / 'trials.pqt'
    if not pqt.exists():
        raise SystemExit(
            f'Missing {pqt}. Point --cache-dir at an ONE cache with '
            'bwm_tables/trials.pqt.')
    return pd.read_parquet(pqt)


def session_frame(df: pd.DataFrame, eid: str) -> pd.DataFrame | None:
    d = df[df['eid'] == eid]
    if 'bwm_include' in d.columns:
        d = d[d['bwm_include'].astype(bool)]
    if len(d) < 10:
        return None
    return d.reset_index(drop=True)


def _lag1(choices: np.ndarray) -> float:
    ch = np.asarray(choices, dtype=float)
    ch = ch[np.isin(ch, (-1.0, 1.0))]
    if len(ch) < 3:
        return float('nan')
    x, y = ch[:-1], ch[1:]
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def _block_align(choice, pleft) -> float:
    choice = np.asarray(choice, dtype=float)
    pleft = np.asarray(pleft, dtype=float)
    valid = np.isin(choice, (-1.0, 1.0))
    left = np.isclose(pleft, 0.8)
    right = np.isclose(pleft, 0.2)
    biased = valid & (left | right)
    if not biased.any():
        return float('nan')
    match = (left & (choice == 1.0)) | (right & (choice == -1.0))
    return float(np.mean(match[biased]))


def _acc_real(trials: pd.DataFrame) -> float:
    fb = np.asarray(trials['feedbackType'].to_numpy(), dtype=float)
    choice = np.asarray(trials['choice'].to_numpy(), dtype=float)
    scored = np.isin(choice, (-1.0, 1.0)) & np.isfinite(fb)
    if not scored.any():
        return float('nan')
    return float(np.mean(fb[scored] == 1.0))


def quintile_metric(choice, pleft, fn) -> np.ndarray:
    q_idx = post05_quintile_index(pleft)
    out = np.full(N_QUINTILES, np.nan)
    choice = np.asarray(choice, dtype=float)
    for q in range(N_QUINTILES):
        out[q] = fn(choice[q_idx == q])
    return out


def _acc_ak(choice, stim_side) -> float:
    """AK / format_data coding: choice sign matches stim_side when correct."""
    choice = np.asarray(choice, dtype=float)
    stim_side = np.asarray(stim_side, dtype=float)
    valid = np.isin(choice, (-1.0, 1.0))
    if not valid.any():
        return float('nan')
    return float(np.mean(choice[valid] == stim_side[valid]))


def score_sequence(choice, pleft, stim_side, prefix: str, acc: float) -> dict:
    mu = quintile_mean_run(choice, pleft)
    lag = quintile_metric(choice, pleft, _lag1)
    rec = {}
    for q in range(N_QUINTILES):
        rec[f'{prefix}_q{q + 1}_mean_run'] = float(mu[q])
        rec[f'{prefix}_q{q + 1}_lag1'] = float(lag[q])
    rec[f'{prefix}_mean_run'] = mean_run(choice[~np.isclose(np.asarray(pleft, float), 0.5)])
    rec[f'{prefix}_lag1'] = _lag1(choice[~np.isclose(np.asarray(pleft, float), 0.5)])
    rec[f'{prefix}_block_align'] = _block_align(choice, pleft)
    rec[f'{prefix}_acc'] = float(acc)
    rec[f'{prefix}_q4_minus_q1'] = float(mu[3] - mu[0])
    return rec, mu


def _fmt(x) -> str:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 'nan'
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    return f'median={q50:.3f}  IQR=[{q25:.3f}, {q75:.3f}]  n={x.size}'


def plot_quintile_mean_run(rows: pd.DataFrame, out_path: Path, styles, title) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    xs = np.arange(1, N_QUINTILES + 1)
    for prefix, color, marker, label in styles:
        med = [rows[f'{prefix}_q{q}_mean_run'].median() for q in range(1, 6)]
        ax.plot(xs, med, color=color, marker=marker, label=label)
    ax.set_xticks(xs)
    ax.set_xlabel('post-0.5 quintile')
    ax.set_ylabel('median mean_run')
    ax.legend(frameon=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def post05_mean_run(choice, pleft) -> float:
    pleft = np.asarray(pleft, dtype=float)
    choice = np.asarray(choice, dtype=float)
    return mean_run(choice[~np.isclose(pleft, 0.5)])


def _pad_sessions(arrs):
    n = len(arrs)
    max_t = max(len(a) for a in arrs)
    out = np.zeros((n, max_t), dtype=float)
    for i, a in enumerate(arrs):
        out[i, :len(a)] = np.asarray(a, dtype=float)
    return out, np.array([len(a) for a in arrs], dtype=int)


def batched_simulate(syn, model, stim_pad, side_pad, theta, seed):
    import torch
    torch.manual_seed(int(seed))
    arr_params = np.asarray(theta, dtype=float)[None]
    act_sim, _, _ = model.simulate_parallel(
        arr_params, stim_pad, side_pad, nb_simul=1)
    return np.asarray(act_sim.squeeze(-1), dtype=float)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--cache-dir',
        default=os.environ.get(
            'ONE_CACHE_DIR',
            str(Path.home() / 'Downloads/ONE/openalyx.internationalbrainlab.org'),
        ))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n-sess', type=int, default=None,
                   help='Cap the number of sessions (default: all BWM)')
    p.add_argument('--ak-params', nargs=4, type=float, default=None,
                   metavar=('ALPHA', 'ZETA', 'LAPSE_POS', 'LAPSE_NEG'),
                   help='AK θ (default 0.2 0.5 0.05 0.05 = analysis α)')
    p.add_argument(
        '--fit-ak', action='store_true',
        help='Per-session MCMC posterior-mean θ (option 1), then copy-last')
    p.add_argument('--nb-steps', type=int, default=200,
                   help='MCMC steps when --fit-ak (default 200)')
    p.add_argument('--nb-chains', type=int, default=4)
    p.add_argument('--out-dir', default=None)
    p.add_argument(
        '--tune-ak', action='store_true',
        help='Grid-search α,ζ per session to match post-0.5 mean_run '
             '(instead of post-hoc copy-last / break-repeat)')
    p.add_argument(
        '--alpha-grid', type=float, nargs='+',
        default=[0.05, 0.10, 0.20, 0.35, 0.50])
    p.add_argument(
        '--zeta-grid', type=float, nargs='+',
        default=[0.05, 0.08, 0.12, 0.16, 0.20, 0.30, 0.50])
    p.add_argument('--lapse', type=float, default=0.05)
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.fit_ak:
        out_dir = (cache_dir / 'manifold' / 'choice_epoch_diag'
                   / 'ak_late_sticky' / 'fitted')
    else:
        out_dir = cache_dir / 'manifold' / 'choice_epoch_diag' / 'ak_late_sticky'
    out_dir.mkdir(parents=True, exist_ok=True)

    import simulate_synthetic_choices as syn

    theta = np.asarray(
        args.ak_params if args.ak_params is not None else (0.2, 0.5, 0.05, 0.05),
        dtype=float)
    sim_model = syn._sim_model()
    if args.tune_ak:
        return run_tune(args, cache_dir, out_dir, syn, sim_model, theta)
    if args.fit_ak:
        print(
            f'AK θ = per-session MCMC  nb_steps={args.nb_steps}  '
            f'nb_chains={args.nb_chains}  (fixed-stim, copy-last to real μ_q)')
        fit_root = out_dir / 'fits'
        fit_root.mkdir(parents=True, exist_ok=True)
    else:
        print(f'AK θ = {theta}  (fixed-stim, per-session copy-last targets)')
        fit_root = None

    df = load_trials_table(cache_dir)
    eids = df['eid'].astype(str).unique()
    if args.n_sess is not None:
        eids = eids[: int(args.n_sess)]
    print(f'{len(eids)} sessions from {cache_dir}')

    rows = []
    n_skip = 0
    rng_master = np.random.default_rng(args.seed)
    for i, eid in enumerate(eids):
        sess = session_frame(df, eid)
        if sess is None:
            n_skip += 1
            continue
        choice_r = np.asarray(sess['choice'].to_numpy(), dtype=float)
        pleft = np.asarray(sess['probabilityLeft'].to_numpy(), dtype=float)
        mu_real = quintile_mean_run(choice_r, pleft)
        if not np.isfinite(mu_real).any():
            n_skip += 1
            continue
        seed_i = int(rng_master.integers(0, 2**31 - 1))
        rec_fit = {}
        try:
            if args.fit_ak:
                tag = str(eid).replace('-', '')[:8]
                _, theta_i = syn.fit_action_kernel(
                    sess, eid=str(eid), subject='bwm',
                    model_dir=fit_root / tag,
                    nb_steps=args.nb_steps, nb_chains=args.nb_chains)
                theta_i = np.asarray(theta_i, dtype=float).reshape(-1)
                rec_fit = {
                    'fit_alpha': float(theta_i[0]),
                    'fit_zeta': float(theta_i[1]),
                    'fit_lapse_pos': float(theta_i[2]),
                    'fit_lapse_neg': float(theta_i[3]),
                }
            else:
                theta_i = theta
            out = syn.synthetic_choices_fixed_stim(
                sess, params=theta_i, n=1, seed=seed_i, model=sim_model,
                late_sticky=False)
        except Exception as exc:  # noqa: BLE001
            print(f'  skip {eid}: AK fit/simulate failed ({exc})')
            n_skip += 1
            continue
        ch_ak = np.asarray(out['choice'], dtype=float).reshape(-1)
        side = np.asarray(out['side'], dtype=float).reshape(-1)
        rng = np.random.default_rng(seed_i + 10007)
        ch_st = apply_late_stickiness(
            ch_ak, pleft, rng, target_mean_run=mu_real)
        rec = {'eid': eid, 'n_trials': int(len(sess))}
        rec.update(rec_fit)
        r_real, _ = score_sequence(
            choice_r, pleft, None, 'real', acc=_acc_real(sess))
        r_ak, _ = score_sequence(
            ch_ak, pleft, side, 'ak', acc=_acc_ak(ch_ak, side))
        r_st, _ = score_sequence(
            ch_st, pleft, side, 'sticky', acc=_acc_ak(ch_st, side))
        rec.update(r_real)
        rec.update(r_ak)
        rec.update(r_st)
        err_ak = np.abs(quintile_mean_run(ch_ak, pleft) - mu_real)
        err_st = np.abs(quintile_mean_run(ch_st, pleft) - mu_real)
        rec['mae_ak'] = float(np.nanmean(err_ak))
        rec['mae_sticky'] = float(np.nanmean(err_st))
        rec['sticky_closer'] = int(rec['mae_sticky'] < rec['mae_ak'])
        rows.append(rec)
        step = 5 if args.fit_ak else 50
        if (i + 1) % step == 0:
            print(f'  {i + 1}/{len(eids)} …', flush=True)

    rows = pd.DataFrame(rows)
    csv_path = out_dir / 'ak_late_sticky_by_session.csv'
    rows.to_csv(csv_path, index=False)
    print(f'Wrote {csv_path}  ({len(rows)} sessions, skipped {n_skip})')

    lines = []

    def log(msg=''):
        print(msg)
        lines.append(msg)

    if args.fit_ak:
        log(f'AK θ = per-session MCMC  nb_steps={args.nb_steps}  '
            f'n={len(rows)}  skipped={n_skip}')
        if 'fit_alpha' in rows.columns:
            log(f'  fitted α: {_fmt(rows["fit_alpha"])}')
            log(f'  fitted ζ: {_fmt(rows["fit_zeta"])}')
    else:
        log(f'AK θ = {theta.tolist()}  n={len(rows)}  skipped={n_skip}')
    log('Fixed real stim; copy-last targets = that session\'s real quintile mean_run')
    log('')
    log('=== Quintile mean_run (median) ===')
    hdr = '        ' + ''.join(f'{"Q" + str(q):>8}' for q in range(1, 6)) + f'{"Q4−Q1":>10}'
    log(hdr)
    for prefix, name in [('real', 'real'), ('ak', 'AK'), ('sticky', 'sticky')]:
        meds = [rows[f'{prefix}_q{q}_mean_run'].median() for q in range(1, 6)]
        d = rows[f'{prefix}_q4_minus_q1'].median()
        log(f'{name:8}' + ''.join(f'{m:8.3f}' for m in meds) + f'{d:10.3f}')
    log('')
    log('=== |mean_run − real| MAE across quintiles ===')
    log(f'  AK:     {_fmt(rows["mae_ak"])}')
    log(f'  sticky: {_fmt(rows["mae_sticky"])}')
    n_closer = int(rows['sticky_closer'].sum())
    log(f'  sticky closer than AK: {n_closer}/{len(rows)} '
        f'({100 * n_closer / max(len(rows), 1):.1f}%)')
    log('')
    log('=== Paired Wilcoxon (sticky − real) vs (AK − real) on mean_run ===')
    for q in range(1, 6):
        real = rows[f'real_q{q}_mean_run']
        d_ak = rows[f'ak_q{q}_mean_run'] - real
        d_st = rows[f'sticky_q{q}_mean_run'] - real
        try:
            p_ak = stats.wilcoxon(d_ak.dropna(), alternative='two-sided').pvalue
        except ValueError:
            p_ak = np.nan
        try:
            p_st = stats.wilcoxon(d_st.dropna(), alternative='two-sided').pvalue
        except ValueError:
            p_st = np.nan
        log(f'  Q{q}: AK−real median={d_ak.median():+.3f} p={p_ak:.3g}   '
            f'sticky−real median={d_st.median():+.3f} p={p_st:.3g}')
    log('')
    log('=== Other session stats (median) ===')
    for key, label in [
        ('block_align', 'block align'),
        ('acc', 'P(correct)'),
        ('lag1', 'lag-1 (post-0.5)'),
        ('q4_minus_q1', 'mean_run Q4−Q1'),
    ]:
        log(f'  {label}:')
        for prefix in ('real', 'ak', 'sticky'):
            log(f'    {prefix:7} {_fmt(rows[f"{prefix}_{key}"])}')
    log('')
    log('=== Quintile lag-1 (median) — not the matching target ===')
    log(hdr)
    for prefix, name in [('real', 'real'), ('ak', 'AK'), ('sticky', 'sticky')]:
        meds = [rows[f'{prefix}_q{q}_lag1'].median() for q in range(1, 6)]
        log(f'{name:8}' + ''.join(f'{m:8.3f}' for m in meds))

    plot_path = out_dir / 'ak_late_sticky_quintile_mean_run.png'
    plot_quintile_mean_run(
        rows, plot_path,
        styles=[
            ('real', 'k', 'o', 'real'),
            ('ak', 'C0', 's', 'AK (stationary)'),
            ('sticky', 'C3', 'D', 'AK + session mix'),
        ],
        title=('Calendar mean_run: real vs fitted AK vs fitted+copy-last'
               if args.fit_ak else
               'Calendar mean_run: real vs AK vs session-matched mix'))

    txt = out_dir / 'ak_late_sticky_summary.txt'
    txt.write_text('\n'.join(lines) + '\n')
    print(f'Wrote {txt}')


def run_tune(args, cache_dir, out_dir, syn, sim_model, theta_default):
    """Per-session α,ζ grid to match post-0.5 mean_run on the real stim stream."""
    out_dir = out_dir / 'tune' if out_dir.name != 'tune' else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_trials_table(cache_dir)
    eids = df['eid'].astype(str).unique()
    if args.n_sess is not None:
        eids = eids[: int(args.n_sess)]

    sessions = []
    for eid in eids:
        sess = session_frame(df, eid)
        if sess is None:
            continue
        try:
            stim, side, pleft = syn.stim_side_from_trials(sess)
        except Exception as exc:  # noqa: BLE001
            print(f'  skip {eid}: format_data failed ({exc})')
            continue
        choice_r = np.asarray(sess['choice'].to_numpy(), dtype=float)
        mu_q = quintile_mean_run(choice_r, pleft)
        if not np.isfinite(mu_q).any():
            continue
        sessions.append(dict(
            eid=str(eid), trials=sess, stim=stim, side=side, pleft=pleft,
            choice_r=choice_r, n=int(len(stim)),
            mu_real=post05_mean_run(choice_r, pleft),
            mu_q=mu_q, acc_real=_acc_real(sess),
        ))
    print(f'Tuning {len(sessions)} sessions; default θ={theta_default.tolist()}')

    stim_pad, lengths = _pad_sessions([s['stim'] for s in sessions])
    side_pad, _ = _pad_sessions([s['side'] for s in sessions])
    plefts = [s['pleft'] for s in sessions]

    alphas = [float(a) for a in args.alpha_grid]
    zetas = [float(z) for z in args.zeta_grid]
    lapse = float(args.lapse)
    grid = [(a, z) for a in alphas for z in zetas]
    print(f'Grid {len(alphas)} α × {len(zetas)} ζ = {len(grid)} cells')

    # choices_by_cell[k][i] = length-n_i array
    best_err = np.full(len(sessions), np.inf)
    best_az = [(np.nan, np.nan)] * len(sessions)
    best_ch = [None] * len(sessions)
    default_ch = [None] * len(sessions)

    for k, (a, z) in enumerate(grid):
        theta = np.array([a, z, lapse, lapse], dtype=float)
        seed = int(args.seed) + 17 * k
        ch_pad = batched_simulate(syn, sim_model, stim_pad, side_pad, theta, seed)
        for i, s in enumerate(sessions):
            n = lengths[i]
            ch = ch_pad[i, :n]
            mu = post05_mean_run(ch, plefts[i])
            err = abs(mu - s['mu_real']) if np.isfinite(mu) else np.inf
            if err < best_err[i]:
                best_err[i] = err
                best_az[i] = (a, z)
                best_ch[i] = ch.copy()
        if np.allclose(theta, theta_default):
            for i, s in enumerate(sessions):
                default_ch[i] = ch_pad[i, :lengths[i]].copy()
        print(f'  cell {k + 1}/{len(grid)}  α={a:.2f} ζ={z:.2f}', flush=True)

    if any(c is None for c in default_ch):
        ch_pad = batched_simulate(
            syn, sim_model, stim_pad, side_pad, theta_default, int(args.seed))
        for i in range(len(sessions)):
            default_ch[i] = ch_pad[i, :lengths[i]].copy()

    rows = []
    for i, s in enumerate(sessions):
        ch_d = default_ch[i]
        ch_t = best_ch[i]
        rec = {
            'eid': s['eid'], 'n_trials': s['n'],
            'tuned_alpha': best_az[i][0], 'tuned_zeta': best_az[i][1],
            'tuned_mean_run_err': float(best_err[i]),
        }
        r_real, _ = score_sequence(
            s['choice_r'], s['pleft'], None, 'real', acc=s['acc_real'])
        r_d, _ = score_sequence(
            ch_d, s['pleft'], s['side'], 'ak', acc=_acc_ak(ch_d, s['side']))
        r_t, _ = score_sequence(
            ch_t, s['pleft'], s['side'], 'tuned', acc=_acc_ak(ch_t, s['side']))
        rec.update(r_real)
        rec.update(r_d)
        rec.update(r_t)
        rec['mae_ak'] = float(np.nanmean(np.abs(
            quintile_mean_run(ch_d, s['pleft']) - s['mu_q'])))
        rec['mae_tuned'] = float(np.nanmean(np.abs(
            quintile_mean_run(ch_t, s['pleft']) - s['mu_q'])))
        rec['tuned_closer'] = int(rec['mae_tuned'] < rec['mae_ak'])
        rows.append(rec)

    rows = pd.DataFrame(rows)
    csv_path = out_dir / 'ak_tune_by_session.csv'
    rows.to_csv(csv_path, index=False)
    print(f'Wrote {csv_path}  ({len(rows)} sessions)')

    lines = []

    def log(msg=''):
        print(msg)
        lines.append(msg)

    log(f'Default θ = {theta_default.tolist()}  n={len(rows)}')
    log(f'Grid α={alphas}  ζ={zetas}  lapse={lapse}')
    log('Fixed real stim; per-session (α,ζ) min |post-0.5 mean_run − real|')
    log('')
    log('=== Chosen (α, ζ) ===')
    log(f'  α: {_fmt(rows["tuned_alpha"])}')
    log(f'  ζ: {_fmt(rows["tuned_zeta"])}')
    log(f'  |μ_tuned − μ_real|: {_fmt(rows["tuned_mean_run_err"])}')
    log('')
    log('=== Quintile mean_run (median) ===')
    hdr = '        ' + ''.join(f'{"Q" + str(q):>8}' for q in range(1, 6)) + f'{"Q4−Q1":>10}'
    log(hdr)
    for prefix, name in [('real', 'real'), ('ak', 'default'), ('tuned', 'tuned')]:
        meds = [rows[f'{prefix}_q{q}_mean_run'].median() for q in range(1, 6)]
        d = rows[f'{prefix}_q4_minus_q1'].median()
        log(f'{name:8}' + ''.join(f'{m:8.3f}' for m in meds) + f'{d:10.3f}')
    log('')
    log('=== |mean_run − real| MAE across quintiles ===')
    log(f'  default: {_fmt(rows["mae_ak"])}')
    log(f'  tuned:   {_fmt(rows["mae_tuned"])}')
    n_closer = int(rows['tuned_closer'].sum())
    log(f'  tuned closer: {n_closer}/{len(rows)} '
        f'({100 * n_closer / max(len(rows), 1):.1f}%)')
    log('')
    log('=== Paired Wilcoxon on post-0.5 mean_run (model − real) ===')
    for prefix, name in [('ak', 'default'), ('tuned', 'tuned')]:
        d = rows[f'{prefix}_mean_run'] - rows['real_mean_run']
        try:
            p = stats.wilcoxon(d.dropna(), alternative='two-sided').pvalue
        except ValueError:
            p = np.nan
        log(f'  {name}: median Δ={d.median():+.3f}  p={p:.3g}')
    log('')
    log('=== Other session stats (median) ===')
    for key, label in [
        ('block_align', 'block align'),
        ('acc', 'P(correct)'),
        ('lag1', 'lag-1 (post-0.5)'),
        ('q4_minus_q1', 'mean_run Q4−Q1'),
        ('mean_run', 'post-0.5 mean_run'),
    ]:
        log(f'  {label}:')
        for prefix in ('real', 'ak', 'tuned'):
            log(f'    {prefix:7} {_fmt(rows[f"{prefix}_{key}"])}')

    plot_path = out_dir / 'ak_tune_quintile_mean_run.png'
    plot_quintile_mean_run(
        rows, plot_path,
        styles=[
            ('real', 'k', 'o', 'real'),
            ('ak', 'C0', 's', 'AK default ζ=0.5'),
            ('tuned', 'C2', 'D', 'AK per-session α,ζ'),
        ],
        title='Calendar mean_run: real vs default AK vs stickiness-tuned AK')
    log(f'\nPlot: {plot_path}')
    txt = out_dir / 'ak_tune_summary.txt'
    txt.write_text('\n'.join(lines) + '\n')
    print(f'Wrote {txt}')
    return 0


if __name__ == '__main__':
    main()
