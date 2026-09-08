"""Act-prior perf/RT overlays for Stage B im150 (Test 6).

Same protocol as ``_tmp_regular_actprior_rt.py`` / mpre3: 10 sessions × 20
blocks, stim seed 12345, data = action-kernel α=0.2, model = binarized
trial-average P. Plots go in each run's
``psychometric_model_vs_data_actprior/``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _tmp_perf_rt_model_vs_data import (  # noqa: E402
    ALYX,
    BEHAVIOR_ACT,
    build_actprior_behavior,
    latest_final,
    plot_one_json,
)
from analyze_choice_epochs import load_sessions_from_aggregate  # noqa: E402

BASE = Path.home() / (
    "Downloads/ONE/openalyx.internationalbrainlab.org/models"
)
SEEDS = (7, 12, 34, 45, 89, 101, 303, 333)
PREFIX = "weights_run_fj_stageB_hold_s89_im150_regular_mask12-13"
OUT = BASE / "stageB_hold_s89_im150_actprior_rt.json"


def main():
    if BEHAVIOR_ACT.is_file():
        behavior = np.load(BEHAVIOR_ACT, allow_pickle=True).item()
        print(f"loaded {BEHAVIOR_ACT}")
    else:
        print("building act-prior behavior from BWM sessions …")
        behavior = build_actprior_behavior(load_sessions_from_aggregate(ALYX))
        BEHAVIOR_ACT.parent.mkdir(parents=True, exist_ok=True)
        np.save(BEHAVIOR_ACT, behavior, allow_pickle=True)
        print(f"wrote {BEHAVIOR_ACT}")

    rows = []
    print(
        f"{'model':<14} {'perf':>6} {'RTcomb':>7} {'RTspl':>6} "
        f"{'con':>6} {'inc':>6}",
        flush=True,
    )
    for seed in SEEDS:
        run = BASE / f"{PREFIX}_s{seed}"
        r = plot_one_json(f"im150_s{seed}", latest_final(run), run, behavior)
        rows.append(r)
        print(
            f"s{seed:<3d}           {r['perf_r2']:.3f}  {r['rt_r2']:7.3f}  "
            f"{r['rt_split_r2']:.3f}  "
            f"({r['rt_split_r2_con']:.3f} / {r['rt_split_r2_inc']:.3f})",
            flush=True,
        )
    OUT.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
