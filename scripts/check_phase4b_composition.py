#!/usr/bin/env python
"""Covariate balance by ITI-P label at c=1.0 (Phase 4b, no prior mod)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import simulate_recovery as sr

N_SESSIONS = 10
SEED = 42


def contrast_for_row(row, split):
    if "stim_r" in split:
        return float(row["contrastRight"])
    if "stim_l" in split:
        return float(row["contrastLeft"])
    return np.nan


def main():
    mp, _ = sr.load_fitted_model(zero_all_prior_mod=True)
    dfs, sbo, _ = sr.simulate_condition_sessions(mp, N_SESSIONS, 6, 400, SEED)
    all_df = pd.concat(dfs, ignore_index=True)
    pcol = sr.PRIOR_COLUMN

    print(f"Phase 4b composition check (seed={SEED}, n_sessions={N_SESSIONS}, all g_*=d_*=0)\n")

    for split in sr.s_prior_splits():
        m = sr.split_fixed_condition_mask(all_df, split)
        sub = all_df.loc[m].copy()
        sub["contrast_stim"] = sub.apply(lambda r: contrast_for_row(r, split), axis=1)
        sub = sub.loc[sub["contrast_stim"] == 1.0]
        if len(sub) < 30:
            print(f"=== {split} c=1.0: n={len(sub)} (skip)\n")
            continue

        hi = sub[pcol].values >= 0.5
        ts = sub["trial_side"].values
        bs = sub["block_side"].values
        tib = sub["trial_in_block"].values.astype(float)

        sdiff = []
        s0l, s0r = [], []
        for _, row in sub.iterrows():
            tr = row["traces"]["S"]
            stim_idx = min(sbo, row["length"] - 1)
            sdiff.append(tr[stim_idx, 1] - tr[stim_idx, 0])
            s0 = row["traces"]["S"]  # use perceived stim from trace window - actually S not S0
            # S at first post-stim step
            if stim_idx + 1 < row["length"]:
                sdiff[-1] = tr[stim_idx + 1, 1] - tr[stim_idx + 1, 0]

        sdiff = np.array(sdiff)

        print(f"=== {split} c=1.0 n={len(sub)} (high={hi.sum()}, low={(~hi).sum()}) ===")
        for label, x in [
            ("P(left-signal)", ts == -1),
            ("P(block_left)", bs == -1),
            ("trial_in_block", tib),
            ("S_r-S_l early stim", sdiff),
        ]:
            if x.dtype == bool:
                print(f"  {label}: high={x[hi].mean():.3f} low={x[~hi].mean():.3f}")
            else:
                print(
                    f"  {label}: high med={np.median(x[hi]):.1f} "
                    f"low med={np.median(x[~hi]):.1f}"
                )
        print()


if __name__ == "__main__":
    main()
