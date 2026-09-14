"""Act-prior perf/RT for gm0_im150. Same 10×20, stim seed 12345."""
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
OUT = BASE / "stageB_hold_s89_gm0_im150_eval" / "actprior_rt.json"
SEEDS = (7, 12, 34, 45, 89, 101, 303, 333)
PREFIX = "weights_run_fj_stageB_hold_s89_gm0_im150_regular_mask7-9-12-13"


def main():
    if BEHAVIOR_ACT.is_file():
        behavior = np.load(BEHAVIOR_ACT, allow_pickle=True).item()
        print(f"loaded {BEHAVIOR_ACT}")
    else:
        behavior = build_actprior_behavior(load_sessions_from_aggregate(ALYX))
        BEHAVIOR_ACT.parent.mkdir(parents=True, exist_ok=True)
        np.save(BEHAVIOR_ACT, behavior, allow_pickle=True)
    rows = []
    print(
        f"{'seed':>4} {'perf':>6} {'RTcomb':>7} {'RTspl':>6} "
        f"{'con':>6} {'inc':>6}",
        flush=True,
    )
    for seed in SEEDS:
        run = BASE / f"{PREFIX}_s{seed}"
        r = plot_one_json(f"gm0im150_s{seed}", latest_final(run), run, behavior)
        r["arm"] = "gm0im150"
        r["seed"] = seed
        rows.append(r)
        print(
            f"{seed:4d} {r['perf_r2']:.3f}  {r['rt_r2']:7.3f}  "
            f"{r['rt_split_r2']:.3f}  "
            f"({r['rt_split_r2_con']:.3f} / {r['rt_split_r2_inc']:.3f})",
            flush=True,
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
