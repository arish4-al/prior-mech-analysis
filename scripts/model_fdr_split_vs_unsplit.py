#!/usr/bin/env python
"""Score model prior distance the mouse BWM way: per-session units + BH-FDR.

The published model metric is one pooled curve_mean per population (40 sessions
stacked, then 4-split or 2-split combine). The mouse metric is BH-FDR *counts*
across ~200 regions after the same combine. This script puts the model on the
mouse readout:

  * each simulated session is one "region" (independent draw of the same circuit)
  * 4-split family = f1/f2 duringstim (mouse ``act_block_duringstim`` four)
  * 2-split family = stim-side duringstim, no f1/f2 (mouse ``act_block_duringstim_{l,r}``)
  * M stays stim-aligned in both arms — the mouse 126 vs 42 table is duringstim
  * combine = SUM of per-split curves (mouse ``combine_four_splits``), not the
    model's average. ``p_mean`` is invariant to that scale; ``amp_euc`` is not
  * BH-FDR on ``p_mean`` across sessions, per population and pooled (S+I+M)

Canonical analysis (80 ms S / 150 ms I/M, fill-from-next-ITI, contrast-matched
null, seed 123, 40 sessions, nrand=100). Phase 4b is the no-coupling control.

  conda activate iblenv
  python scripts/model_fdr_split_vs_unsplit.py --cases absence phase4 --n-jobs 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulate_recovery import (  # noqa: E402
    BLOCKS_PER_SESSION_DEFAULT,
    N_SESSIONS_DEFAULT,
    UNSPLIT_PRIOR_SPLITS,
    _experiment_case_spec,
    build_split_results,
    default_output_dir,
    s_prior_splits,
    simulate_condition_sessions,
)


POPS = ("S", "I", "M")
ALPHA = 0.01


def _p_mean_model(obs: np.ndarray, nulls: np.ndarray) -> float:
    """Fraction of null curve-means ≥ observed (model ``population_prior_test``)."""
    return float(np.mean(np.mean(nulls, axis=1) >= float(np.mean(obs))))


def _p_mean_mouse(obs: np.ndarray, nulls: np.ndarray) -> float:
    """Mouse combine: include the observed curve in the comparison set."""
    stacked = np.concatenate([obs.reshape(1, -1), nulls], axis=0)
    return float(np.mean(np.mean(stacked, axis=1) >= float(np.mean(stacked[0]))))


def _amp_euc(curve: np.ndarray) -> float:
    return float(np.max(curve) - np.min(curve))


def _combine_sum(stacks: list[np.ndarray], rng_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Mouse SUM combine (``plot_choice_null_comparison_table._combine_split_curve_stacks``)."""
    stacks = [np.asarray(s, dtype=float) for s in stacks]
    obs = np.sum([s[0] for s in stacks], axis=0)
    null_counts = [max(s.shape[0] - 1, 0) for s in stacks]
    if any(c < 1 for c in null_counts):
        raise ValueError("each split needs ≥1 null curve to combine")
    if len(set(s.shape[0] for s in stacks)) == 1:
        return obs, np.sum([s[1:] for s in stacks], axis=0)
    n_mc = max(min(null_counts), 2000)
    rng = np.random.default_rng(int(rng_seed) & 0xFFFFFFFF)
    nulls = np.zeros((n_mc, obs.shape[0]), dtype=float)
    for k in range(n_mc):
        acc = np.zeros(obs.shape[0], dtype=float)
        for s, n_u in zip(stacks, null_counts):
            acc += s[1 + int(rng.integers(0, n_u))]
        nulls[k] = acc
    return obs, nulls


def _fdr_counts(pvals: np.ndarray, alpha: float) -> dict:
    pvals = np.asarray(pvals, dtype=float)
    n = int(pvals.size)
    if n == 0:
        return {
            "n": 0,
            "uncorr_le_alpha": 0,
            "uncorr_le_0.05": 0,
            "fdr_le_alpha": 0,
            "fdr_le_0.05": 0,
            "median_p": np.nan,
        }
    uncorr_a = int(np.sum(pvals <= alpha))
    uncorr_05 = int(np.sum(pvals <= 0.05))
    _, pc, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    _, pc05, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    return {
        "n": n,
        "uncorr_le_alpha": uncorr_a,
        "uncorr_le_0.05": uncorr_05,
        "fdr_le_alpha": int(np.sum(pc <= alpha)),
        "fdr_le_0.05": int(np.sum(pc05 <= 0.05)),
        "median_p": float(np.median(pvals)),
    }


def _session_stacks(df, steps_before_obs, splits, nrand, rng, n_jobs, contrast_matched):
    """pop -> {split: (1+U, T) stack} for one session. Missing splits omitted."""
    out = {p: {} for p in POPS}
    for split in splits:
        built = build_split_results(
            df,
            split,
            steps_before_obs,
            nrand,
            rng,
            populations=POPS,
            n_jobs=n_jobs,
            contrast_matched_null=contrast_matched,
        )
        if built is None:
            continue
        _r, regde, _xn = built
        for pop, curves in regde.items():
            out[pop][split] = np.asarray(curves, dtype=float)
    return out


def _unit_row(case, family, sess, pop, splits, stacks, rng_seed):
    """One session × population × family after mouse SUM combine."""
    present = [s for s in splits if s in stacks]
    if len(present) != len(splits):
        return {
            "case": case,
            "family": family,
            "session": sess,
            "population": pop,
            "n_splits_kept": len(present),
            "n_splits_required": len(splits),
            "complete": False,
            "p_mean_model": np.nan,
            "p_mean_mouse": np.nan,
            "curve_mean": np.nan,
            "amp_euc_sum": np.nan,
            "amp_euc_mean": np.nan,
            "n_null": 0,
        }
    pop_i = POPS.index(pop) if pop in POPS else 0
    fam_i = 0 if family == "split4" else 1
    seed = int(rng_seed) + 17 * int(sess) + 101 * pop_i + 1009 * fam_i
    obs, nulls = _combine_sum([stacks[s] for s in present], rng_seed=seed)
    n_split = len(present)
    return {
        "case": case,
        "family": family,
        "session": sess,
        "population": pop,
        "n_splits_kept": n_split,
        "n_splits_required": len(splits),
        "complete": True,
        "p_mean_model": _p_mean_model(obs, nulls),
        "p_mean_mouse": _p_mean_mouse(obs, nulls),
        "curve_mean": float(np.mean(obs)),
        "amp_euc_sum": _amp_euc(obs),
        "amp_euc_mean": _amp_euc(obs) / n_split,
        "n_null": int(nulls.shape[0]),
    }


def _summarize(units: pd.DataFrame, pcol: str, alpha: float) -> pd.DataFrame:
    rows = []
    for (case, family, pop), g in units.groupby(["case", "family", "population"], sort=False):
        gg = g[g["complete"]]
        fd = _fdr_counts(gg[pcol].to_numpy(), alpha)
        rows.append(
            {
                "case": case,
                "family": family,
                "population": pop,
                "p_col": pcol,
                "n_complete": fd["n"],
                "n_dropped": int((~g["complete"]).sum()),
                "uncorr_le_0.01": fd["uncorr_le_alpha"] if alpha == 0.01 else int(np.sum(gg[pcol] <= 0.01)),
                "uncorr_le_0.05": fd["uncorr_le_0.05"],
                "fdr_le_0.01": fd["fdr_le_alpha"] if alpha == 0.01 else int(
                    np.sum(multipletests(gg[pcol], 0.01, method="fdr_bh")[1] <= 0.01)
                ) if fd["n"] else 0,
                "fdr_le_0.05": fd["fdr_le_0.05"],
                "median_p": fd["median_p"],
                "median_amp_sum": float(gg["amp_euc_sum"].median()) if fd["n"] else np.nan,
                "median_amp_mean": float(gg["amp_euc_mean"].median()) if fd["n"] else np.nan,
                "median_curve_mean": float(gg["curve_mean"].median()) if fd["n"] else np.nan,
            }
        )
    # Pooled S+I+M = all-region analog
    for (case, family), g in units.groupby(["case", "family"], sort=False):
        gg = g[g["complete"]]
        fd = _fdr_counts(gg[pcol].to_numpy(), alpha)
        rows.append(
            {
                "case": case,
                "family": family,
                "population": "S+I+M",
                "p_col": pcol,
                "n_complete": fd["n"],
                "n_dropped": int((~g["complete"]).sum()),
                "uncorr_le_0.01": int(np.sum(gg[pcol] <= 0.01)) if fd["n"] else 0,
                "uncorr_le_0.05": fd["uncorr_le_0.05"],
                "fdr_le_0.01": int(np.sum(multipletests(gg[pcol], 0.01, method="fdr_bh")[1] <= 0.01))
                if fd["n"]
                else 0,
                "fdr_le_0.05": fd["fdr_le_0.05"],
                "median_p": fd["median_p"],
                "median_amp_sum": float(gg["amp_euc_sum"].median()) if fd["n"] else np.nan,
                "median_amp_mean": float(gg["amp_euc_mean"].median()) if fd["n"] else np.nan,
                "median_curve_mean": float(gg["curve_mean"].median()) if fd["n"] else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _run_case(
    case,
    weights_json,
    n_sessions,
    blocks_per_session,
    max_obs,
    rng_seed,
    nrand,
    n_jobs,
    contrast_matched,
):
    mp, exp_tag, cond_desc = _experiment_case_spec(case, weights_json)
    print(f"\n=== {exp_tag}: {cond_desc} ===")
    t0 = time.perf_counter()
    session_dfs, steps_before_obs, _meta = simulate_condition_sessions(
        mp, n_sessions, blocks_per_session, max_obs, rng_seed
    )
    print(f"  sessions ready ({len(session_dfs)}) in {time.perf_counter() - t0:.1f}s")

    split4 = list(s_prior_splits())
    split2 = list(UNSPLIT_PRIOR_SPLITS)
    families = {"split4": split4, "unsplit2": split2}

    units = []
    for i, df in enumerate(session_dfs):
        t1 = time.perf_counter()
        rng = np.random.RandomState(rng_seed + 1000 * (i + 1))
        stacks4 = _session_stacks(
            df, steps_before_obs, split4, nrand, rng, n_jobs, contrast_matched
        )
        rng_u = np.random.RandomState(rng_seed + 2000 * (i + 1))
        stacks2 = _session_stacks(
            df, steps_before_obs, split2, nrand, rng_u, n_jobs, contrast_matched
        )
        for pop in POPS:
            units.append(_unit_row(exp_tag, "split4", i, pop, split4, stacks4[pop], rng_seed))
            units.append(_unit_row(exp_tag, "unsplit2", i, pop, split2, stacks2[pop], rng_seed))
        if (i + 1) % 5 == 0 or i + 1 == len(session_dfs):
            print(
                f"  session {i + 1}/{len(session_dfs)} "
                f"({time.perf_counter() - t1:.1f}s this, "
                f"{time.perf_counter() - t0:.1f}s total)"
            )
    return pd.DataFrame(units), {
        "case": exp_tag,
        "condition": cond_desc,
        "n_sessions": len(session_dfs),
        "split4": split4,
        "unsplit2": split2,
        "elapsed_s": time.perf_counter() - t0,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", nargs="+", default=["absence", "phase4"])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n-sessions", type=int, default=N_SESSIONS_DEFAULT)
    p.add_argument("--nrand", type=int, default=100)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--blocks-per-session", type=int, default=BLOCKS_PER_SESSION_DEFAULT)
    p.add_argument("--weights-json", default=None)
    p.add_argument("--label-shuffle-null", action="store_true")
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    max_obs = 400

    out_dir = Path(args.output_dir) if args.output_dir else (
        default_output_dir() / "unsplit_prior" / f"seed_{args.seed}" / "fdr_split_vs_unsplit"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    contrast_matched = not args.label_shuffle_null
    all_units = []
    meta = {
        "seed": args.seed,
        "n_sessions": args.n_sessions,
        "nrand": args.nrand,
        "n_jobs": args.n_jobs,
        "blocks_per_session": args.blocks_per_session,
        "contrast_matched_null": contrast_matched,
        "alpha": args.alpha,
        "s_window_s": 0.08,
        "im_window_s": 0.15,
        "unit": "one simulated session = one FDR unit (region analog)",
        "combine": "sum of per-split curves (mouse combine_four_splits)",
        "m_alignment": "stimOn duringstim in both arms (mouse 126 vs 42 table)",
        "p_mean_model": "P(null mean >= obs mean)",
        "p_mean_mouse": "P(mean of {obs}+nulls >= obs mean)  [includes observed]",
        "cases": [],
    }

    for case in args.cases:
        units, case_meta = _run_case(
            case,
            args.weights_json,
            args.n_sessions,
            args.blocks_per_session,
            max_obs,
            args.seed,
            args.nrand,
            args.n_jobs,
            contrast_matched,
        )
        all_units.append(units)
        meta["cases"].append(case_meta)

    units = pd.concat(all_units, ignore_index=True)
    units_path = out_dir / "per_session_units.csv"
    units.to_csv(units_path, index=False)

    summaries = []
    for pcol in ("p_mean_model", "p_mean_mouse"):
        summaries.append(_summarize(units, pcol, args.alpha))
    summary = pd.concat(summaries, ignore_index=True)
    summary_path = out_dir / "fdr_summary.csv"
    summary.to_csv(summary_path, index=False)

    meta_path = out_dir / "setup.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("\n=== FDR summary (p_mean_mouse, mouse formula) ===")
    show = summary[summary["p_col"] == "p_mean_mouse"]
    cols = [
        "case",
        "family",
        "population",
        "n_complete",
        "n_dropped",
        "uncorr_le_0.01",
        "fdr_le_0.01",
        "fdr_le_0.05",
        "median_p",
        "median_amp_mean",
        "median_curve_mean",
    ]
    print(show[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\n=== FDR summary (p_mean_model) ===")
    show = summary[summary["p_col"] == "p_mean_model"]
    print(show[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"\nWrote {units_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
