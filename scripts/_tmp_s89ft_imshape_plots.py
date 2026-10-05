"""S / I/M / prior / ITI overlays + act-prior RT for s89ft imshape.

Same shared stim as the eval (bps=20, seed 12345, from regular s101).
Full include_stim. plot_one writes the 150 ms display prior_effects;
this also writes prior_effects_80ms.svg/png (fit window) without replacing
the 150 ms alias. Plots go in each run dir.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _tmp_mpre3_meancell_plots import (  # noqa: E402
    ALYX,
    BASE,
    BEHAVIOR_ACT,
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
import numpy as np  # noqa: E402
import model_functions as mf  # noqa: E402
from model_functions import (  # noqa: E402
    int_regs,
    loss_prior_effect,
    move_regs,
    resolve_prior_distance_window,
    run_model,
    savefig_svg_png,
)

ARMS = {
    "nowii": (
        "weights_run_fj_stageB_hold_s89_full_s89ft_imshape_meancell_full_mask0-1-3-4-6-10-11-12-13-14-15-16-17-18-19-20",
        True,
    ),
    "wii": (
        "weights_run_fj_stageB_hold_s89_full_s89ft_imshape_wii_meancell_full_mask1-3-4-6-10-11-12-13-14-15-16-17-18-19-20",
        True,
    ),
}
SEEDS = (89, 7, 12, 45)
OUT = Path(os.environ.get(
    "PLOT_OUT",
    str(BASE / "stageB_hold_s89_s89ft_imshape_plot_summary.json"),
))


def plot_prior_80ms(jp: Path, stim_bundle, prior_regions, run: Path, stim_curve):
    mp, _ = load_plot_model(jp)
    (
        stimuli, trial_strengths, trial_sides, block_sides,
        steps_before_obs, bps,
    ) = stim_bundle
    results = run_model(
        "data", stimuli, trial_strengths, trial_sides, block_sides, bps,
        steps_before_obs=steps_before_obs, verbose=False,
        backend="numba", **mp,
    )
    T_prior, _, _ = resolve_prior_distance_window(mp, T=72, plot_window=80)
    loss_prior_effect(
        regions=prior_regions, results=results, model_params=mp,
        steps_before_obs=steps_before_obs, T=T_prior, model_metric="l2",
        timeframes=("act_block_duringstim", "act_block_duringchoice"),
        ptype="p_mean_c", plot_window=80, reload=False,
        label_A="integrator", label_B="move", do_plot=True,
        plot_shifted=False, ylim=None, scale_factors=[1, 1, 1],
        include_all_trials=True, save_dir=None, plot_stim=True,
        lump_all=False, include_stim=True, stim_curve_path=str(stim_curve),
    )
    fig = plt.gcf()
    savefig_svg_png(
        fig, str(run / "prior_effects_80ms.svg"),
        dpi=150, bbox_inches="tight", facecolor="white", transparent=False,
    )
    plt.close("all")


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
        f"{'arm':6} {'seed':>4} {'Ipre':>6} {'Ipost':>6} {'Mpre':>6} "
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
        print(f"  80ms prior overlay -> {run / 'prior_effects_80ms.svg'}", flush=True)
        plot_prior_80ms(jp, stim_bundle, prior_regions, run, stim_curve)
        s["arm"] = arm
        s["seed"] = seed
        traj_rows.append(s)
        print(
            f"  {arm:6} s{seed:3d}  I {fmt(s.get('gof_I_pre'))}/"
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
        f"{'arm':6} {'seed':>4} {'perf':>6} {'RTcomb':>7} {'RTspl':>6} "
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
            f"  {arm:6} s{seed:3d}  perf {r['perf_r2']:.3f}  "
            f"RT {r['rt_r2']:.3f}  split {r['rt_split_r2']:.3f}  "
            f"({r['rt_split_r2_con']:.3f} / {r['rt_split_r2_inc']:.3f})",
            flush=True,
        )

    payload = {"traj": traj_rows, "rt": rt_rows}
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
