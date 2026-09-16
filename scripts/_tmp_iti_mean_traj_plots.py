"""ITI S/I/M mean-trajectory overlays for noiti + production regular.

Same shared stim as the eval (bps=20, seed 12345, from regular s101).
Writes ``S_iti`` / ``I_iti`` / ``M_iti`` (.svg+.png) into each run dir.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from plot_best_fit_results import (  # noqa: E402
    load_plot_model,
    make_shared_stimuli,
    plot_iti_mean_trajectories,
)
import model_functions as mf  # noqa: E402
from model_functions import run_model  # noqa: E402

SEEDS = (7, 12, 34, 45, 89, 101, 303, 333)
BASE = Path.home() / (
    "Downloads/ONE/openalyx.internationalbrainlab.org/models"
)
ARMS = {
    "noiti": "weights_run_fj_stageB_hold_s89_noiti_regular_mask12-13",
    "regular": "weights_run_fj_stageB_hold_s89_regular_mask12-13",
}
TS = re.compile(r"(\d{8}-\d{6})")
STIM_REF = BASE / "weights_run_fj_stageB_hold_s89_regular_mask12-13_s101"


def newest_final(d: Path) -> Path:
    finals = list(d.glob("weights_final_*.json"))
    if not finals:
        raise FileNotFoundError(d)

    def key(p):
        m = TS.search(p.name)
        return (m.group(1) if m else "", p.stat().st_mtime)

    return max(finals, key=key)


def main():
    jobs = []
    for arm, prefix in ARMS.items():
        for seed in SEEDS:
            d = BASE / f"{prefix}_s{seed}"
            jobs.append((arm, seed, newest_final(d), d))

    mp0, _ = load_plot_model(newest_final(STIM_REF))
    stim_bundle = make_shared_stimuli(mp0, bps=20, seed=12345)
    (
        stimuli,
        trial_strengths,
        trial_sides,
        block_sides,
        steps_before_obs,
        bps,
    ) = stim_bundle
    print(
        f"HAVE_NUMBA={mf._HAVE_NUMBA}  n={len(jobs)}  "
        f"stim from {STIM_REF.name}"
    )

    for i, (arm, seed, jp, run_dir) in enumerate(jobs, 1):
        mp, _ = load_plot_model(jp)
        results = run_model(
            "data",
            stimuli,
            trial_strengths,
            trial_sides,
            block_sides,
            bps,
            steps_before_obs=steps_before_obs,
            verbose=False,
            backend="numba",
            **mp,
        )
        written = plot_iti_mean_trajectories(
            results, steps_before_obs, mp, run_dir
        )
        names = " ".join(p.name for p in written)
        print(f"[{i}/{len(jobs)}] {arm} s{seed}  {names} -> {run_dir.name}", flush=True)


if __name__ == "__main__":
    main()
