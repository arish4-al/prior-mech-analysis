#!/usr/bin/env python
"""Run Phase 4b (no prior mod) across seeds and aggregate split×contrast results."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import simulate_recovery as sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/phase4_multiseed"),
        help="Root dir; each seed writes to {output-dir}/seed_{seed}/",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123, 999, 2024])
    parser.add_argument("--nrand", type=int, default=100)
    parser.add_argument("--n-sessions", type=int, default=40)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        seed_root = args.output_dir / f"seed_{seed}"
        print(f"\n>>> seed {seed} -> {seed_root}")
        summary = sr.run_phase4_no_prior_mod_analysis(
            seed_root,
            rng_seed=seed,
            nrand=args.nrand,
            n_sessions=args.n_sessions,
            n_jobs=args.n_jobs,
        )
        sc_path = (
            seed_root
            / "absence"
            / "figs"
            / "phase4_no_prior_mod"
            / "phase4_no_prior_mod_split_contrast.csv"
        )
        df = pd.read_csv(sc_path)
        df["seed"] = seed
        rows.append(df)
        combined = summary.get("combined", [])
        print(f"    combined p_mean: " + ", ".join(
            f"{r['population']}={r.get('p_mean', '?'):.4f}" for r in combined
        ))

    all_df = pd.concat(rows, ignore_index=True)
    out_dir = args.output_dir / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_dir / "phase4_multiseed_split_contrast.csv", index=False)

    # Per (split, population, contrast): how often p=0, median true, median null
    def agg_group(g):
        p = g["p_shuffle_gte_true"].dropna()
        return pd.Series(
            {
                "n_seeds": len(g),
                "n_with_data": int(p.notna().sum()),
                "n_sig_p005": int((p < 0.05).sum()),
                "n_sig_p0": int((p == 0).sum()),
                "true_mean_median": g["true_curve_mean"].median(),
                "null_median_median": g["shuffle_curve_mean_median"].median(),
                "p_values": ",".join(f"{x:.3g}" for x in p),
            }
        )

    summary_df = (
        all_df.groupby(["split", "population", "contrast"], dropna=False)
        .apply(agg_group)
        .reset_index()
    )
    summary_df.to_csv(out_dir / "phase4_multiseed_summary.csv", index=False)

    # Per-split pooled significance across seeds
    by_split = (
        all_df.groupby(["split", "population", "seed"])["p_shuffle_gte_true"]
        .first()
        .reset_index()
    )
    split_agg = (
        by_split.groupby(["split", "population"])["p_shuffle_gte_true"]
        .agg(n_seeds="count", n_sig=lambda s: (s < 0.05).sum(), p_values=lambda s: list(s))
        .reset_index()
    )
    split_agg.to_csv(out_dir / "phase4_multiseed_by_split.csv", index=False)

    print(f"\nWrote aggregate to {out_dir}")
    print("\n=== Split-level (all contrasts pooled in run; per-split from by_split csv) ===")
    bs_path = seed_root / "absence" / "figs" / "phase4_no_prior_mod" / "phase4_no_prior_mod_by_split.csv"
    # re-read per-seed by_split
    bs_rows = []
    for seed in args.seeds:
        p = (
            args.output_dir
            / f"seed_{seed}"
            / "absence"
            / "figs"
            / "phase4_no_prior_mod"
            / "phase4_no_prior_mod_by_split.csv"
        )
        if p.exists():
            b = pd.read_csv(p)
            b["seed"] = seed
            bs_rows.append(b)
    if bs_rows:
        bs_all = pd.concat(bs_rows, ignore_index=True)
        bs_all.to_csv(out_dir / "phase4_multiseed_by_split_all_seeds.csv", index=False)
        for (split, pop), g in bs_all.groupby(["split", "population"]):
            ps = g["p_mean"].tolist()
            n_sig = sum(p < 0.05 for p in ps)
            print(f"  {split} {pop}: p={ps} ({n_sig}/{len(ps)} sig)")

    print("\n=== Contrast bins with data in all seeds (n_sig / n_seeds) ===")
    for _, r in summary_df.sort_values(["split", "population", "contrast"]).iterrows():
        if r["n_with_data"] == 0:
            continue
        flag = "***" if r["n_sig_p005"] == r["n_with_data"] else (
            "ns" if r["n_sig_p005"] == 0 else f"{int(r['n_sig_p005'])}/{int(r['n_with_data'])}"
        )
        print(
            f"  {r['split']} {r['population']} c={r['contrast']}: {flag} "
            f"(true~{r['true_mean_median']:.4f}, null~{r['null_median_median']:.4f}) "
            f"p=[{r['p_values']}]"
        )

    meta = {
        "seeds": args.seeds,
        "nrand": args.nrand,
        "n_sessions": args.n_sessions,
        "output_dir": str(args.output_dir),
    }
    with open(out_dir / "phase4_multiseed_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
