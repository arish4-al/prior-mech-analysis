"""S / I/M / prior / ITI overlays + act-prior behavior for mpre3 meancell.

Same shared stim as the eval (bps=20, seed 12345, from regular s101).
Full arms overlay the unsplit-80 S sidecar. Plots go in each run dir.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from plot_best_fit_results import (  # noqa: E402
    ensure_fit_data_links_paper,
    load_avg_mean_r,
    load_mean_data_results,
    load_plot_model,
    make_shared_stimuli,
    plot_one,
)
from _tmp_perf_rt_model_vs_data import (  # noqa: E402
    ALYX,
    BEHAVIOR_ACT,
    build_actprior_behavior,
    plot_one_json,
)
from _fit_data import load_s_unsplit80  # noqa: E402
from analyze_choice_epochs import load_sessions_from_aggregate  # noqa: E402
import model_functions as mf  # noqa: E402
from model_functions import int_regs, move_regs  # noqa: E402

BASE = Path.home() / (
    "Downloads/ONE/openalyx.internationalbrainlab.org/models"
)
SEEDS = (7, 12, 34, 45, 89, 101, 303, 333)
ARMS = {
    "full80": (
        "weights_run_fj_stageB_hold_s89_full_mpre3_meancell_full_masknone",
        True,
    ),
    "reg150": (
        "weights_run_fj_stageB_hold_s89_mpre3_im150_meancell_regular_mask12-13",
        False,
    ),
    "full150": (
        "weights_run_fj_stageB_hold_s89_full_mpre3_im150_meancell_full_masknone",
        True,
    ),
}
OUT = Path(os.environ.get(
    "PLOT_OUT",
    str(BASE / "stageB_hold_s89_mpre3_meancell_plot_summary.json"),
))
TS = re.compile(r"(\d{8}-\d{6})")


def newest_final(d: Path) -> Path:
    finals = list(d.glob("weights_final_*.json"))
    if not finals:
        raise FileNotFoundError(d)

    def key(p):
        m = TS.search(p.name)
        return (m.group(1) if m else "", p.stat().st_mtime)

    return max(finals, key=key)


def alias_svgs(out_dir: Path) -> None:
    mapping = {
        "IM_pre.svg": "IM_pre_fit_*.svg",
        "IM_post.svg": "IM_post_fit_*.svg",
        "P_fit.svg": "P_fit_*.svg",
        "prior_effects.svg": "prior_effects_*.svg",
    }
    for dest, glob in mapping.items():
        hits = [p for p in out_dir.glob(glob) if p.name != dest]
        if not hits:
            continue
        newest = max(hits, key=lambda p: p.stat().st_mtime)
        shutil.copy2(newest, out_dir / dest)
        png = newest.with_suffix(".png")
        if png.is_file():
            shutil.copy2(png, out_dir / dest.replace(".svg", ".png"))


def fmt(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "nan"
    return f"{float(x):.3f}"


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
    print(
        "\n=== S / I/M / prior  "
        f"{'arm':8} {'seed':>4} {'Ipre':>6} {'Ipost':>6} {'Mpre':>6} "
        f"{'Mpost':>6} {'S':>6} {'priorS':>7}",
        flush=True,
    )
    for i, (arm, seed, run, include_stim) in enumerate(jobs, 1):
        jp = newest_final(run)
        print(
            f"\n[traj {i}/{len(jobs)}] {arm} s{seed}  include_stim={include_stim}  "
            f"{run.name}",
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
            f"  {arm:8} s{seed:3d}  I {fmt(s.get('gof_I_pre'))}/"
            f"{fmt(s.get('gof_I_post'))}  "
            f"M {fmt(s.get('gof_M_pre'))}/{fmt(s.get('gof_M_post'))}  "
            f"S {fmt(s.get('gof_S'))}  priorS {fmt(s.get('prior_S'))}  "
            f"-> {run.name}",
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
    print(
        "\n=== act-prior behavior  "
        f"{'arm':8} {'seed':>4} {'perf':>6} {'RTcomb':>7} {'RTspl':>6} "
        f"{'con':>6} {'inc':>6}",
        flush=True,
    )
    for i, (arm, seed, run, _include_stim) in enumerate(jobs, 1):
        print(f"\n[rt {i}/{len(jobs)}] {arm} s{seed}", flush=True)
        r = plot_one_json(
            f"{arm}_s{seed}", newest_final(run), run, behavior,
        )
        r["arm"] = arm
        r["seed"] = seed
        rt_rows.append(r)
        print(
            f"  {arm:8} s{seed:3d}  perf {r['perf_r2']:.3f}  "
            f"RT {r['rt_r2']:.3f}  split {r['rt_split_r2']:.3f}  "
            f"({r['rt_split_r2_con']:.3f} / {r['rt_split_r2_inc']:.3f})",
            flush=True,
        )

    if only and OUT.is_file() and "PLOT_OUT" not in os.environ:
        prev = json.loads(OUT.read_text())
        traj_rows = (
            [r for r in prev.get("traj", []) if r.get("arm") not in only]
            + traj_rows
        )
        rt_rows = (
            [r for r in prev.get("rt", []) if r.get("arm") not in only]
            + rt_rows
        )
        print(f"merged {only} into existing plot summary")
    payload = {"traj": traj_rows, "rt": rt_rows}
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
