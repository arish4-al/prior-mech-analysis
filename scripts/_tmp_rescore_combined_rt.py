"""Rescore cached act-prior RT with pooled (not con/inc-averaged) combined R²."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _tmp_perf_rt_model_vs_data import (  # noqa: E402
    BEHAVIOR_ACT,
    WEIGHTS_REL,
    latest_final,
    plot_one_json,
)

BASE = Path.home() / (
    "Downloads/ONE/openalyx.internationalbrainlab.org/models"
)
SEEDS = (7, 12, 34, 45, 89, 101, 303, 333)
MPRE = "weights_run_fj_stageB_hold_s89_mpre3_regular_mask12-13"
REG = "weights_run_fj_stageB_hold_s89_regular_mask12-13"


def main():
    behavior = np.load(BEHAVIOR_ACT, allow_pickle=True).item()
    print("loaded", BEHAVIOR_ACT)

    mpre_rows, reg_rows = [], []
    print(f"{'model':<16} {'comb':>7} {'con':>7} {'inc':>7}", flush=True)
    for seed in SEEDS:
        run = BASE / f"{MPRE}_s{seed}"
        r = plot_one_json(f"mpre3_s{seed}", latest_final(run), run, behavior)
        mpre_rows.append(r)
        print(
            f"mpre3 s{seed:<3d}     {r['rt_r2']:7.3f} {r['rt_split_r2_con']:7.3f} "
            f"{r['rt_split_r2_inc']:7.3f}",
            flush=True,
        )
    for seed in SEEDS:
        run = BASE / f"{REG}_s{seed}"
        r = plot_one_json(f"base_s{seed}", latest_final(run), run, behavior)
        reg_rows.append(r)
        print(
            f"base  s{seed:<3d}     {r['rt_r2']:7.3f} {r['rt_split_r2_con']:7.3f} "
            f"{r['rt_split_r2_inc']:7.3f}",
            flush=True,
        )
    wr = plot_one_json("WEIGHTS_REL", WEIGHTS_REL, WEIGHTS_REL.parent, behavior)
    print(
        f"{'WEIGHTS_REL':<16} {wr['rt_r2']:7.3f} {wr['rt_split_r2_con']:7.3f} "
        f"{wr['rt_split_r2_inc']:7.3f}",
        flush=True,
    )

    mpre_sum = BASE / "stageB_hold_s89_mpre3_plot_summary.json"
    prev = json.loads(mpre_sum.read_text()) if mpre_sum.is_file() else {}
    prev["rt"] = mpre_rows
    prev["combined_rt_pooled"] = True
    mpre_sum.write_text(json.dumps(prev, indent=2, default=str))

    (BASE / "stageB_hold_s89_regular_actprior_rt.json").write_text(
        json.dumps(reg_rows + [wr], indent=2, default=str)
    )
    print("updated summaries")


if __name__ == "__main__":
    main()
