"""S / I/M / prior overlays + act-prior RT for Stage B mpre3 regular seeds.

Plots go in each run dir. Act-prior RT uses the same protocol as
``_tmp_perf_rt_model_vs_data.py`` (10×20, subj-P on model, AK on data).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

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
from _tmp_ablation_plots import alias_svgs  # noqa: E402
from _tmp_perf_rt_model_vs_data import (  # noqa: E402
    ALYX,
    BEHAVIOR_ACT,
    build_actprior_behavior,
    plot_one_json,
)
from analyze_choice_epochs import load_sessions_from_aggregate  # noqa: E402
import model_functions as mf  # noqa: E402
from model_functions import int_regs, move_regs  # noqa: E402

BASE = Path.home() / (
    "Downloads/ONE/openalyx.internationalbrainlab.org/models"
)
SEEDS = (7, 12, 34, 45, 89, 101, 303, 333)
PREFIX = "weights_run_fj_stageB_hold_s89_mpre3_regular_mask12-13"
OUT = BASE / "stageB_hold_s89_mpre3_plot_summary.json"


def latest_final(d: Path) -> Path:
    finals = sorted(d.glob("weights_final_*.json"))
    if not finals:
        raise FileNotFoundError(d)
    return finals[-1]


def fmt(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "nan"
    return f"{float(x):.3f}"


def main():
    ensure_fit_data_links_paper()
    _, mean_data = load_mean_data_results()
    _, avg_mean_R = load_avg_mean_r()
    print(f"HAVE_NUMBA={mf._HAVE_NUMBA}")
    prior_regions = {
        "int_regs_choice": int_regs,
        "int_regs_stim": int_regs,
        "move_regs_choice": move_regs,
        "move_regs_stim": move_regs,
        "stim_regs": ["VISpm", "FRP", "VISal"],
    }
    stim_ref = latest_final(
        BASE / "weights_run_fj_stageB_hold_s89_regular_mask12-13_s101"
    )
    mp0, _ = load_plot_model(stim_ref)
    stim_bundle = make_shared_stimuli(mp0, bps=20, seed=12345)
    print(f"shared stim from {stim_ref.parent.name}")

    traj_rows = []
    print(
        "\n=== S / I/M / prior  "
        f"{'seed':>4} {'Ipre':>6} {'Ipost':>6} {'Mpre':>6} {'Mpost':>6} "
        f"{'S':>6} {'prior':>6}",
        flush=True,
    )
    for seed in SEEDS:
        run = BASE / f"{PREFIX}_s{seed}"
        jp = latest_final(run)
        print(f"\n[traj {seed}] {run.name}", flush=True)
        s = plot_one(
            jp, stim_bundle, mean_data, prior_regions, run,
            avg_mean_R=avg_mean_R,
        )
        plt.close("all")
        alias_svgs(run)
        s["seed"] = seed
        traj_rows.append(s)
        print(
            f"  s{seed:3d}  I {fmt(s.get('gof_I_pre'))}/{fmt(s.get('gof_I_post'))}  "
            f"M {fmt(s.get('gof_M_pre'))}/{fmt(s.get('gof_M_post'))}  "
            f"S {fmt(s.get('gof_S'))}  prior {fmt(s.get('gof_prior'))}",
            flush=True,
        )

    if BEHAVIOR_ACT.is_file():
        behavior = np.load(BEHAVIOR_ACT, allow_pickle=True).item()
        print(f"\nloaded {BEHAVIOR_ACT}")
    else:
        print("\nloading BWM trials.pqt for act-prior data …")
        behavior = build_actprior_behavior(load_sessions_from_aggregate(ALYX))
        np.save(BEHAVIOR_ACT, behavior, allow_pickle=True)

    rt_rows = []
    print(
        "\n=== act-prior RT  "
        f"{'seed':>4} {'perf':>6} {'RTcomb':>7} {'RTspl':>6} {'con':>6} {'inc':>6}",
        flush=True,
    )
    for seed in SEEDS:
        run = BASE / f"{PREFIX}_s{seed}"
        r = plot_one_json(f"mpre3_s{seed}", latest_final(run), run, behavior)
        rt_rows.append(r)
        print(
            f"  s{seed:3d}  perf {r['perf_r2']:.3f}  RT {r['rt_r2']:.3f}  "
            f"split {r['rt_split_r2']:.3f}  "
            f"({r['rt_split_r2_con']:.3f} / {r['rt_split_r2_inc']:.3f})",
            flush=True,
        )

    payload = {"traj": traj_rows, "rt": rt_rows}
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {OUT}")
    print("\nseed   I pre/post     M pre/post     S      prior   perf   RT comb  RT con/inc")
    by_seed = {r["seed"]: r for r in traj_rows}
    for r in rt_rows:
        seed = int(str(r["label"]).rsplit("_s", 1)[-1])
        t = by_seed[seed]
        print(
            f"s{seed:<3d}  {fmt(t.get('gof_I_pre'))}/{fmt(t.get('gof_I_post'))}   "
            f"{fmt(t.get('gof_M_pre'))}/{fmt(t.get('gof_M_post'))}   "
            f"{fmt(t.get('gof_S'))}  {fmt(t.get('gof_prior'))}  "
            f"{r['perf_r2']:.3f}  {r['rt_r2']:7.3f}  "
            f"{r['rt_split_r2_con']:.3f}/{r['rt_split_r2_inc']:.3f}"
        )


if __name__ == "__main__":
    main()
