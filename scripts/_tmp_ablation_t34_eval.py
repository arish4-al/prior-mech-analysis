"""Shared-stim fair eval for Stage B tests 3–4 (W_pp box / tied θ)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _tmp_ablation_fair_eval import (  # noqa: E402
    BASE,
    NEW,
    SEEDS,
    eval_one,
    extract_row,
    fmt,
    tau_delta,
)
from plot_best_fit_results import (  # noqa: E402
    ensure_fit_data_links_paper,
    load_mean_data_results,
    load_plot_model,
    make_shared_stimuli,
)
from _fit_data import load_avg_mean_r  # noqa: E402
import model_functions as mf  # noqa: E402
from model_functions import int_regs, move_regs  # noqa: E402


def arm_dir(arm: str, seed: int) -> Path:
    if arm == "baseline":
        return BASE / f"weights_run_fj_stageB_hold_s89_regular_mask12-13_s{seed}"
    if arm == "onethr":
        return NEW / (
            f"weights_run_fj_stageB_hold_s89_onethr_regular_mask11-12-13_s{seed}"
        )
    return NEW / (
        f"weights_run_fj_stageB_hold_s89_{arm}_regular_mask12-13_s{seed}"
    )


def main():
    arms = ("baseline", "wpplarge", "wppopen", "wppsmall", "onethr")
    rows = []
    for arm in arms:
        for seed in SEEDS:
            r = extract_row(arm_dir(arm, seed), arm, seed)
            meta = json.loads(Path(r["path"]).read_text())
            mp = meta.get("model_params") or {}
            report = {}
            rp = arm_dir(arm, seed) / "run_fit_joint_report.json"
            if rp.is_file():
                report = json.loads(rp.read_text())
            r["tied_thresholds"] = bool(mp.get("tied_thresholds", False))
            r["w_pp_bounds"] = report.get("w_pp_bounds")
            r["set_w_pp"] = report.get("set_w_pp")
            rows.append(r)

    print("=== recorded params ===")
    print(
        f"{'arm':9} {'seed':>4} {'rec':>7} {'W_pp':>8} {'tP_s':>8} "
        f"{'th_c':>6} {'th_d':>6} {'g_i':>8} {'d_i':>8} tied wall_m"
    )
    for r in rows:
        tp = r["tau_P"]
        tp_s = tp / 1000.0 if np.isfinite(tp) else float("inf")
        print(
            f"{r['arm']:9} {r['seed']:4d} {r['recorded']:7.3f} "
            f"{r['W_pp']:8.5f} {tp_s:8.2f} "
            f"{r['theta_c']:6.3f} {r['theta_d']:6.3f} "
            f"{r['g_i']:8.3g} {r['d_i']:8.3g} "
            f"{int(r['tied_thresholds'])} {(r['wall_s'] or 0)/60:6.1f}"
        )

    ensure_fit_data_links_paper()
    mean_path, mean_data = load_mean_data_results()
    avg_path, avg_mean_R = load_avg_mean_r()
    print(f"\nmean_data: {mean_path}")
    print(f"avg_mean_R: {avg_path}")
    print(f"HAVE_NUMBA={mf._HAVE_NUMBA}")

    prior_regions = {
        "int_regs_choice": int_regs,
        "int_regs_stim": int_regs,
        "move_regs_choice": move_regs,
        "move_regs_stim": move_regs,
        "stim_regs": ["VISpm", "FRP", "VISal"],
    }
    stim_ref = Path(
        next(r["path"] for r in rows if r["arm"] == "baseline" and r["seed"] == 101)
    )
    mp0, _ = load_plot_model(stim_ref)
    stim_bundle = make_shared_stimuli(mp0, bps=20, seed=12345)
    print(f"shared stim from {stim_ref.parent.name}")

    print(f"\n=== shared-stim eval ({len(rows)} runs) ===")
    print(
        f"{'arm':9} {'seed':>4} {'rec':>7} {'traj':>7} {'prior':>7} "
        f"{'Lw':>7} {'LS':>7} {'fair':>7} {'W_pp':>8} {'tP_s':>8}"
    )
    evals = []
    for r in rows:
        ev = eval_one(
            Path(r["path"]), stim_bundle, mean_data, prior_regions, avg_mean_R,
        )
        rec = {**r, **ev}
        evals.append(rec)
        tp_s = r["tau_P"] / 1000.0 if np.isfinite(r["tau_P"]) else float("inf")
        print(
            f"{r['arm']:9} {r['seed']:4d} {ev['recorded']:7.3f} "
            f"{ev['eval_traj']:7.3f} {ev['eval_prior']:7.3f} "
            f"{ev['eval_Lw']:7.3f} {fmt(ev['eval_LS'])} {fmt(ev['eval_fair'])} "
            f"{r['W_pp']:8.5f} {tp_s:8.2f}",
            flush=True,
        )

    out = ROOT / "scripts" / "_tmp_ablation_t34_eval.json"
    out.write_text(json.dumps({"params": rows, "evals": evals}, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
