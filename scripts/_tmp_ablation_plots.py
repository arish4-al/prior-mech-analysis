"""I/M traj + prior overlays for Stage B model-detail ablations.

Same shared stim as the eval (bps=20, seed 12345, from baseline s101).
Writes into each run’s model folder (openalyx models / models/new).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt

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
import model_functions as mf  # noqa: E402
from model_functions import int_regs, move_regs  # noqa: E402

SEEDS = (7, 12, 34, 45, 89, 101, 303, 333)
BASE = Path.home() / "Downloads/ONE/openalyx.internationalbrainlab.org/models"
NEW = BASE / "new"


def latest_final(d: Path) -> Path:
    finals = sorted(d.glob("weights_final_*.json"))
    if not finals:
        raise FileNotFoundError(f"no weights_final in {d}")
    return finals[-1]


def arm_dir(arm: str, seed: int) -> Path:
    if arm == "baseline":
        return BASE / f"weights_run_fj_stageB_hold_s89_regular_mask12-13_s{seed}"
    if arm == "onethr":
        return NEW / (
            f"weights_run_fj_stageB_hold_s89_onethr_regular_mask11-12-13_s{seed}"
        )
    if arm in ("poffset", "noiti"):
        return BASE / (
            f"weights_run_fj_stageB_hold_s89_{arm}_regular_mask12-13_s{seed}"
        )
    return NEW / (
        f"weights_run_fj_stageB_hold_s89_{arm}_regular_mask12-13_s{seed}"
    )


def alias_svgs(out_dir: Path) -> None:
    """Stable names: IM_pre.svg, IM_post.svg, P_fit.svg, prior_effects.svg."""
    mapping = {
        "IM_pre.svg": "IM_pre_fit_*.svg",
        "IM_post.svg": "IM_post_fit_*.svg",
        "P_fit.svg": "P_fit_*.svg",
        "prior_effects.svg": "prior_effects_*.svg",
    }
    for dest, glob in mapping.items():
        hits = [p for p in out_dir.glob(glob) if p.name != dest]
        if len(hits) == 1:
            shutil.copy2(hits[0], out_dir / dest)


def main():
    arms = (
        "baseline",
        "poffset",
        "noiti",
        "wpplarge",
        "wppopen",
        "wppsmall",
        "onethr",
    )
    jobs = [("baseline", 101)]
    for arm in arms:
        for seed in SEEDS:
            if arm == "baseline" and seed == 101:
                continue
            jobs.append((arm, seed))

    jsons = []
    for arm, seed in jobs:
        d = arm_dir(arm, seed)
        jp = latest_final(d)
        jsons.append((arm, seed, jp, d))

    ensure_fit_data_links_paper()
    mean_path, mean_data = load_mean_data_results()
    avg_path, avg_mean_R = load_avg_mean_r()
    print(f"mean_data: {mean_path}")
    print(f"avg_mean_R: {avg_path}")
    print(f"HAVE_NUMBA={mf._HAVE_NUMBA}")
    print(f"n_models={len(jsons)}  plots -> each model run dir")

    prior_regions = {
        "int_regs_choice": int_regs,
        "int_regs_stim": int_regs,
        "move_regs_choice": move_regs,
        "move_regs_stim": move_regs,
        "stim_regs": ["VISpm", "FRP", "VISal"],
    }
    stim_ref = jsons[0][2]
    mp0, _ = load_plot_model(stim_ref)
    stim_bundle = make_shared_stimuli(mp0, bps=20, seed=12345)
    print(f"shared stim from {stim_ref.parent.name}")

    summaries = []
    for i, (arm, seed, jp, run_dir) in enumerate(jsons, 1):
        out_dir = run_dir
        print(f"\n[{i}/{len(jsons)}] {arm} s{seed}  {run_dir.name}", flush=True)
        s = plot_one(
            jp, stim_bundle, mean_data, prior_regions, out_dir,
            avg_mean_R=avg_mean_R,
        )
        plt.close("all")
        alias_svgs(out_dir)
        s["arm"] = arm
        s["seed"] = seed
        summaries.append(s)
        print(
            f"  rec={s['recorded_loss']:.3f} eval_traj+prior={s['eval_total']:.3f} "
            f"(traj={s['traj']:.3f}+prior={s['prior']:.3f}) -> {out_dir}",
            flush=True,
        )

    payload = NEW / "modeling_details_plot_summary.json"
    payload.write_text(json.dumps(summaries, indent=2, default=str))
    print(f"\nwrote {payload}")


if __name__ == "__main__":
    main()
