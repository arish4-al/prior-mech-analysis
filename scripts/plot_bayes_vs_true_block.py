#!/usr/bin/env python
"""Check Bayes-block routing and plot Bayes vs true-block on a few BWM sessions.

  conda activate iblenv
  python scripts/plot_bayes_vs_true_block.py
  python scripts/plot_bayes_vs_true_block.py --check-only
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import block_analysis_allsplits as ba  # noqa: E402

F1_BAYES = [
    'bayes_block_duringstim_r_choice_r_f1',
    'bayes_block_duringstim_l_choice_l_f1',
    'bayes_block_stim_r_duringchoice_r_f1',
    'bayes_block_stim_l_duringchoice_l_f1',
]
F1_ACT = [s.replace('bayes_block', 'act_block') for s in F1_BAYES]
DEFAULT_EIDS = [
    '6713a4a7-faed-4df2-acab-ee4e63326f8d',
    'b182b754-3c3e-4942-8144-6ee790926b58',
    '5ec72172-c398-4e0e-8c53-e1d1cf0b427d',  # may fail; fallback below
    'a8a8af78-16de-4841-ab07-fde4b5281a03',
]


def check_pipeline() -> None:
    print('=== routing ===')
    for sp in F1_BAYES:
        assert ba.is_act_block_prior_split(sp), sp
        assert ba._split_uses_bayes_prior(sp), sp
        assert not ba._split_uses_act_prior(sp), f'act steal: {sp}'
        assert sp in ba.align, f'missing align: {sp}'
        spec = ba._act_block_conditioning_spec(sp)
        print(f'  {sp}')
        print(f'    align={ba.align[sp]} window={ba.pre_post[sp]} spec={spec}')
    for sp in F1_ACT:
        assert ba._split_uses_act_prior(sp) and not ba._split_uses_bayes_prior(sp)
    # Same-side f1: stim and choice on the same side.
    assert ba._act_block_conditioning_spec(F1_BAYES[0])['stim_is_left'] is False
    assert ba._act_block_conditioning_spec(F1_BAYES[0])['choice'] == -1.0
    assert ba._act_block_conditioning_spec(F1_BAYES[1])['stim_is_left'] is True
    assert ba._act_block_conditioning_spec(F1_BAYES[1])['choice'] == 1.0

    print('=== bayesian_priors smoke (journal prior_definitions) ===')
    left = np.ones(80, dtype=bool)
    right = np.zeros(80, dtype=bool)
    p_l, b_l = ba.bayesian_priors(left)
    p_r, b_r = ba.bayesian_priors(right)
    print(f'  80 left:  P(left) last={p_l[-1]:.3f}  binary last={b_l[-1]}')
    print(f'  80 right: P(left) last={p_r[-1]:.3f}  binary last={b_r[-1]}')
    if not (0.70 <= p_l[-1] <= 0.85):
        raise SystemExit(f'expected ~0.77 after 80 left, got {p_l[-1]}')
    if not (0.15 <= p_r[-1] <= 0.30):
        raise SystemExit(f'expected ~0.23 after 80 right, got {p_r[-1]}')
    mid = np.concatenate([np.ones(40, dtype=bool), np.zeros(40, dtype=bool)])
    _, b_m = ba.bayesian_priors(mid)
    print(f'  40L then 40R: binary {b_m[0]} → {b_m[39]} → {b_m[-1]}')

    side = np.where(left, -1.0, 1.0)
    dummy_ch = np.ones(80)
    ys = ba._block_null_prior_left(F1_BAYES[0], dummy_ch, side)
    assert ys.dtype == bool and ys[-1]  # late left-stim → prior L
    ys_act = ba._block_null_prior_left(F1_ACT[0], dummy_ch, side)
    assert ys_act[-1]  # all left choices → act prior L
    print('  _block_null_prior_left: bayes uses stim; act uses choice. OK')

    print('=== Harris unique routing (stim×choice / stim-only / choice) ===')
    unsplit = ['bayes_block_duringstim_l', 'bayes_block_duringstim_r']
    for sp in unsplit:
        assert ba.is_harris_eligible_split(sp), sp
        spec = ba._act_block_conditioning_spec(sp)
        assert spec['stim_is_left'] is not None, sp
        assert spec['choice'] is None, f'stim-only must drop choice: {sp} {spec}'
        print(f'  {sp} spec={spec}')
    for sp in F1_BAYES:
        assert ba.is_harris_eligible_split(sp), sp
        spec = ba._act_block_conditioning_spec(sp)
        assert spec['stim_is_left'] is not None and spec['choice'] is not None, sp
    choice_bayes = [
        'choice_duringstim_l_block_l_bayes',
        'choice_stim_r_block_r_bayes',
    ]
    for sp in choice_bayes:
        assert ba.is_choice_lr_split(sp) and ba.is_harris_eligible_split(sp), sp
        assert ba._split_uses_bayes_prior(sp) and not ba._split_uses_act_prior(sp)
        stim, p = ba._choice_lr_stratum_targets(sp)
        assert stim is not None and p is not None, sp
        print(f'  {sp} stim_left={stim} pleft={p}')
    print('pipeline checks passed')


def _trials_from_pqt(path: Path):
    import pandas as pd
    df = pd.read_parquet(path)
    if 'contrastLeft' not in df.columns or 'probabilityLeft' not in df.columns:
        return None
    return df


def _load_trials(eid: str):
    from one.api import ONE
    from brainbox.io.one import SessionLoader
    one = ONE(mode='local')
    sl = SessionLoader(one=one, eid=eid)
    sl.load_trials()
    return sl.trials


def _discover_local_trial_tables() -> list[tuple[str, Path]]:
    """(label, pqt) for complete ``_ibl_trials.table.pqt`` in the local ONE cache."""
    from one.api import ONE
    root = Path(ONE(mode='local').cache_dir)
    found = []
    for p in sorted(root.rglob('_ibl_trials.table.pqt')):
        if '#2025' in str(p):
            continue
        parts = p.parts
        try:
            i = parts.index('Subjects')
            label = f'{parts[i + 1]}_{parts[i + 2]}_{parts[i + 3]}'
        except (ValueError, IndexError):
            label = p.parent.parent.name
        found.append((label, p))
    return found


def _session_priors(trials):
    stim_is_left = ~np.isnan(trials['contrastLeft'].astype(float).to_numpy())
    pleft = trials['probabilityLeft'].to_numpy().astype(float)
    choice = trials['choice'].to_numpy().astype(float)
    bayes_c, bayes_b = ba.bayesian_priors(stim_is_left)
    act_c, act_b = ba.action_kernel_priors(ba.alpha, list(choice))
    act_c = np.asarray(act_c, dtype=float)
    act_b = np.asarray(act_b, dtype=float)
    biased = ~np.isclose(pleft, 0.5)
    return {
        'pleft': pleft,
        'bayes_c': bayes_c,
        'bayes_b': bayes_b,
        'act_c': act_c,
        'act_b': act_b,
        'biased': biased,
        'n': len(pleft),
    }


def _agree(a, b, mask) -> float:
    if int(mask.sum()) == 0:
        return float('nan')
    return float(np.mean(np.isclose(a[mask], b[mask])))


def plot_sessions(eids, out_path: Path, n_want: int) -> None:
    import matplotlib.pyplot as plt

    rows = []
    sources = [(eid, None) for eid in eids]
    for label, pqt in _discover_local_trial_tables():
        sources.append((label, pqt))

    for label, pqt in sources:
        if len(rows) >= n_want:
            break
        try:
            trials = (_trials_from_pqt(pqt) if pqt is not None
                      else _load_trials(label))
        except Exception as exc:
            print(f'  skip {str(label)[:12]} load: {exc}')
            continue
        if trials is None or len(trials) < 80:
            print(f'  skip {str(label)[:12]} short/empty/incomplete')
            continue
        if 'contrastLeft' not in trials.columns or 'probabilityLeft' not in trials.columns:
            print(f'  skip {str(label)[:12]} missing columns')
            continue
        rec = _session_priors(trials)
        rec['eid'] = str(label)
        sig = (rec['n'], round(float(rec['pleft'][:5].sum()), 5))
        if any((r['n'], round(float(r['pleft'][:5].sum()), 5)) == sig for r in rows):
            continue
        rows.append(rec)
        m = rec['biased']
        print(
            f'  {rec["eid"][:28]}  n={rec["n"]} biased={int(m.sum())}  '
            f'bayes≡block={_agree(rec["bayes_b"], rec["pleft"], m):.3f}  '
            f'act≡block={_agree(rec["act_b"], rec["pleft"], m):.3f}'
        )

    if not rows:
        raise SystemExit('no sessions loaded from local ONE cache')

    fig, axes = plt.subplots(len(rows), 1, figsize=(11, 2.6 * len(rows)),
                             sharex=False, squeeze=False)
    for ax, rec in zip(axes[:, 0], rows):
        t = np.arange(rec['n'])
        ax.step(t, rec['pleft'], where='mid', color='0.25', lw=1.6,
                label='true block')
        ax.plot(t, rec['bayes_c'], color='C0', lw=1.1, alpha=0.9,
                label='Bayes P(left)')
        ax.step(t, rec['bayes_b'], where='mid', color='C0', lw=1.0, ls='--',
                alpha=0.85, label='Bayes 0.8/0.2')
        ax.plot(t, rec['act_c'], color='C1', lw=0.8, alpha=0.55,
                label='act-kernel (α=0.2)')
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel('P(left)')
        m = rec['biased']
        ax.set_title(
            f'{rec["eid"][:8]}  n={rec["n"]}  '
            f'Bayes≡block {100*_agree(rec["bayes_b"], rec["pleft"], m):.1f}%  '
            f'act≡block {100*_agree(rec["act_b"], rec["pleft"], m):.1f}% '
            f'(biased trials)',
            loc='left', fontsize=10)
        ax.legend(loc='upper right', fontsize=8, ncol=2, frameon=False)
    axes[-1, 0].set_xlabel('trial')
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f'wrote {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--n-sessions', type=int, default=4)
    p.add_argument('--eids', nargs='*', default=None)
    p.add_argument('--out', default=None)
    args = p.parse_args()
    check_pipeline()
    if args.check_only:
        return
    cache = Path(os.environ.get(
        'ONE_CACHE_DIR',
        str(Path.home() / 'Downloads/ONE/alyx.internationalbrainlab.org')))
    out = (Path(args.out) if args.out
           else cache / 'manifold' / 'figs' / 'bayes_vs_true_block.png')
    eids = list(args.eids) if args.eids else list(DEFAULT_EIDS)
    # Extra fixture eids if some fail.
    fx = ROOT / 'brainwidemap' / 'fixtures' / '2023_12_bwm_release.csv'
    if fx.exists():
        import csv
        with fx.open() as f:
            for row in csv.DictReader(f):
                eid = row.get('eid')
                if eid and eid not in eids:
                    eids.append(eid)
                if len(eids) > 40:
                    break
    print('=== sessions ===')
    plot_sessions(eids, out, args.n_sessions)


if __name__ == '__main__':
    main()
