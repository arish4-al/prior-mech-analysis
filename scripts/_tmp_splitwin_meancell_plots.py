"""S / I/M / prior / ITI overlays + act-prior RT for stim150/choice80 meancell.

Same shared stim as the eval (bps=20, seed 12345, from regular s101).
Plots go in each run dir (SVG+PNG).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _tmp_mpre3_meancell_plots import (  # noqa: E402
    ALYX,
    BASE,
    BEHAVIOR_ACT,
    SEEDS,
    alias_svgs,
    fmt,
    newest_final,
)
from plot_best_fit_results import (  # noqa: E402
    ensure_fit_data_links_paper,
    load_avg_mean_r,
    load_mean_data_results,
    load_plot_model,
    make_shared_stimuli,
    plot_one,
)
from _tmp_perf_rt_model_vs_data import (  # noqa: E402
    build_actprior_behavior,
    plot_one_json,
)
from _fit_data import load_s_unsplit80  # noqa: E402
from analyze_choice_epochs import load_sessions_from_aggregate  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import model_functions as mf  # noqa: E402
from model_functions import int_regs, move_regs  # noqa: E402

ARMS = {
    "reg_split": (
        "weights_run_fj_stageB_hold_s89_stim150_choice80_meancell_regular_mask12-13",
        False,
    ),
    "reg_split_mpre3": (
        "weights_run_fj_stageB_hold_s89_mpre3_stim150_choice80_meancell_regular_mask12-13",
        False,
    ),
    "full_split": (
        "weights_run_fj_stageB_hold_s89_full_stim150_choice80_meancell_full_masknone",
        True,
    ),
    "full_split_mpre3": (
        "weights_run_fj_stageB_hold_s89_full_mpre3_stim150_choice80_meancell_full_masknone",
        True,
    ),
}
OUT = Path(os.environ.get(
    "PLOT_OUT",
    str(BASE / "stageB_hold_s89_splitwin_meancell_plot_summary.json"),
))


def main():
    ensure_fit_data_links_paper()
    _, mean_data = load_mean_data_results()
    _, avg_mean_R = load_avg_mean_r()
    stim_curve, payload = load_s_unsplit80()
    print(f"HAVE_NUMBA={mf._HAVE_NUMBA}  S sidecar={stim_curve.name}")
    prior_regions = {
        "int_regs_choice": int_regs,
        "int_regs_stim": int_regs,
        "move_regs_choice": move_regs,
        "move_regs_stim": move_regs,
        "stim_regs": list(payload.get("regs_stim") or ["VISpm", "FRP", "VISal"]),
    }
    stim_ref = newest_final(
        BASE / "weights_run_fj_stageB_hold_s89_regular_mask12-13_s101"
    )
    mp0, _ = load_plot_model(stim_ref)
    stim_bundle = make_shared_stimuli(mp0, bps=20, seed=12345)
    print(f"shared stim from {stim_ref.parent.name}")

    only = [a for a in os.environ.get("ONLY_ARMS", "").split() if a]
    arms = {k: v for k, v in ARMS.items() if (not only or k in only)}
    jobs = []
    for arm, (prefix, include_stim) in arms.items():
        for seed in SEEDS:
            d = BASE / f"{prefix}_s{seed}"
            if d.is_dir():
                jobs.append((arm, seed, d, include_stim))
            else:
                print(f"SKIP missing {prefix}_s{seed}")

    traj_rows = []
    for i, (arm, seed, run, include_stim) in enumerate(jobs, 1):
        jp = newest_final(run)
        print(
            f"\n[traj {i}/{len(jobs)}] {arm} s{seed}  include_stim={include_stim}",
            flush=True,
        )
        s = plot_one(
            jp, stim_bundle, mean_data, prior_regions, run,
            avg_mean_R=avg_mean_R, include_stim=include_stim,
        )
        plt.close("all")
        alias_svgs(run)
        s["arm"] = arm
        s["seed"] = seed
        traj_rows.append(s)
        print(
            f"  {arm:16} s{seed:3d}  I {fmt(s.get('gof_I_pre'))}/"
            f"{fmt(s.get('gof_I_post'))}  "
            f"M {fmt(s.get('gof_M_pre'))}/{fmt(s.get('gof_M_post'))}  "
            f"S {fmt(s.get('gof_S'))}  priorS {fmt(s.get('prior_S'))}",
            flush=True,
        )

    if BEHAVIOR_ACT.is_file():
        behavior = np.load(BEHAVIOR_ACT, allow_pickle=True).item()
        print(f"\nloaded {BEHAVIOR_ACT}")
    else:
        print("\nloading BWM trials.pqt for act-prior data …")
        behavior = build_actprior_behavior(load_sessions_from_aggregate(ALYX))
        BEHAVIOR_ACT.parent.mkdir(parents=True, exist_ok=True)
        np.save(BEHAVIOR_ACT, behavior, allow_pickle=True)

    rt_rows = []
    for i, (arm, seed, run, _include_stim) in enumerate(jobs, 1):
        print(f"\n[rt {i}/{len(jobs)}] {arm} s{seed}", flush=True)
        r = plot_one_json(
            f"{arm}_s{seed}", newest_final(run), run, behavior,
        )
        r["arm"] = arm
        r["seed"] = seed
        rt_rows.append(r)
        print(
            f"  {arm:16} s{seed:3d}  perf {r['perf_r2']:.3f}  "
            f"RT {r['rt_r2']:.3f}  split {r['rt_split_r2']:.3f}  "
            f"({r['rt_split_r2_con']:.3f} / {r['rt_split_r2_inc']:.3f})",
            flush=True,
        )

    payload = {"traj": traj_rows, "rt": rt_rows}
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
