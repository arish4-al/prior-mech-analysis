"""Shared-stim fair eval for Stage B model-detail ablations (2026-08-27).

Protocol matches journals/retinal_then_joint_fitting.md 2026-08-12g/13:
bps=20, stim seed 12345, nested fit_targets/, L_w = traj+prior, fair = L_w+L_S.

Stimuli are built from the Stage B regular s101 hybrid retinal (same as
baseline), not from each ablation JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from plot_best_fit_results import (  # noqa: E402
    ensure_fit_data_links_paper,
    load_mean_data_results,
    load_plot_model,
    make_shared_stimuli,
)
from _fit_data import load_avg_mean_r  # noqa: E402
import model_functions as mf  # noqa: E402
from model_functions import (  # noqa: E402
    compute_sse_stim_right,
    int_regs,
    loss_plot_diff_by_condition_with_data,
    loss_prior_effect,
    mean_by_condition,
    mean_S_by_contrast,
    move_regs,
    run_model,
)

NEW = Path.home() / (
    "Downloads/ONE/alyx.internationalbrainlab.org/models/new"
)
BASE = Path.home() / (
    "Downloads/ONE/openalyx.internationalbrainlab.org/models"
)
SEEDS = (7, 12, 34, 45, 89, 101, 303, 333)
TAU_P = 20.0


def tau_delta(w, tau=TAU_P):
    den = 1.0 - 2.0 * float(w)
    if den <= 0:
        return float("inf")
    return tau / den


def latest_final(d: Path) -> Path:
    finals = sorted(d.glob("weights_final_*.json"))
    if not finals:
        raise FileNotFoundError(f"no weights_final in {d}")
    return finals[-1]


def parse_arm_seed(name: str):
    if "_poffset_" in name:
        arm = "poffset"
    elif "_noiti_" in name:
        arm = "noiti"
    elif "_regular_" in name:
        arm = "baseline"
    else:
        arm = "other"
    seed = int(name.rsplit("_s", 1)[-1])
    return arm, seed


def extract_row(d: Path, arm: str, seed: int) -> dict:
    jp = latest_final(d)
    meta = json.loads(jp.read_text())
    report = {}
    rp = d / "run_fit_joint_report.json"
    if rp.is_file():
        report = json.loads(rp.read_text())
    W = meta["W"]
    g = meta["g"]
    dd = meta["d"]
    th = meta["theta"]
    mp = meta.get("model_params") or {}
    wii, wmm, wpp = W["W_ii"], W["W_mm"], W["W_pp"]
    return {
        "arm": arm,
        "seed": seed,
        "dir": d.name,
        "json": jp.name,
        "recorded": float(meta.get("loss", report.get("final_loss", np.nan))),
        "wall_s": report.get("wall_s"),
        "fit_status": report.get("fit_status"),
        "p_offset_always_on": bool(mp.get("p_offset_always_on", False)),
        "iti_penalty": bool(mp.get("iti_penalty", True)),
        "W_ii": wii,
        "W_mm": wmm,
        "W_pp": wpp,
        "W_is": W["W_is"],
        "W_pi": W["W_pi"],
        "W_mi": W["W_mi"],
        "tau_I": tau_delta(wii),
        "tau_M": tau_delta(wmm),
        "tau_P": tau_delta(wpp),
        "g_i": g["g_i"],
        "g_m": g["g_m"],
        "d_i": dd["d_i"],
        "d_m": dd["d_m"],
        "g_s": g.get("g_s", meta.get("g_s")),
        "d_s": dd.get("d_s", meta.get("d_s")),
        "theta_c": th["theta_c"],
        "theta_d": th["theta_d"],
        "path": str(jp),
    }


def eval_one(jp: Path, stim_bundle, mean_data, prior_regions, avg_mean_R,
             *, force_iti_penalty=None, force_p_offset=None):
    mp, meta = load_plot_model(jp)
    if force_iti_penalty is not None:
        mp["iti_penalty"] = bool(force_iti_penalty)
    if force_p_offset is not None:
        mp["p_offset_always_on"] = bool(force_p_offset)
    (
        stimuli, trial_strengths, trial_sides, block_sides,
        steps_before_obs, bps,
    ) = stim_bundle
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
    sim_out = mean_by_condition(results, steps_before_obs)
    loss_traj = loss_plot_diff_by_condition_with_data(
        sim_out, mp, var_names=("I", "P", "M"),
        mean_data_results=mean_data, plot=False,
    )
    loss_prior = loss_prior_effect(
        regions=prior_regions,
        results=results,
        model_params=mp,
        steps_before_obs=steps_before_obs,
        T=72,
        model_metric="l2",
        timeframes=("act_block_duringstim", "act_block_duringchoice"),
        ptype="p_mean_c",
        plot_window=80,
        reload=False,
        label_A="integrator",
        label_B="move",
        do_plot=False,
        plot_shifted=False,
        ylim=None,
        scale_factors=[1, 1, 1],
        include_all_trials=True,
        plot_stim=False,
        lump_all=False,
    )
    traj = float(loss_traj["total"])
    prior = float(loss_prior["total"])
    iti_pen = loss_traj.get("debug", {}).get("iti_penalty")
    iti_sum = None
    if iti_pen is not None:
        try:
            iti_sum = float(np.nansum([float(x) for x in np.ravel(iti_pen)]))
        except Exception:
            iti_sum = None
    S_avg = mean_S_by_contrast(results, steps_before_obs)
    sse = compute_sse_stim_right(S_avg, avg_mean_R, baseline_R=0)
    raw_ls = sse["total_loss"]
    L_S = float(raw_ls) if np.isfinite(raw_ls) else None
    return {
        "eval_traj": traj,
        "eval_prior": prior,
        "eval_Lw": traj + prior,
        "eval_LS": L_S,
        "eval_fair": (traj + prior + L_S) if L_S is not None else None,
        "eval_iti_pen_sum": iti_sum,
        "eval_p_offset": bool(mp.get("p_offset_always_on", False)),
        "eval_iti_penalty": bool(mp.get("iti_penalty", True)),
        "recorded": float(meta.get("loss", np.nan)),
    }


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "nan"
    return f"{x:.{nd}f}"


def main():
    rows = []
    for seed in SEEDS:
        d = BASE / f"weights_run_fj_stageB_hold_s89_regular_mask12-13_s{seed}"
        rows.append(extract_row(d, "baseline", seed))
    for arm in ("poffset", "noiti"):
        for seed in SEEDS:
            d = NEW / (
                f"weights_run_fj_stageB_hold_s89_{arm}_regular_mask12-13_s{seed}"
            )
            rows.append(extract_row(d, arm, seed))

    print("=== recorded params ===")
    hdr = (
        f"{'arm':8} {'seed':>4} {'rec':>7} {'W_ii':>7} {'tI_s':>6} "
        f"{'W_mm':>7} {'tM_ms':>6} {'W_pp':>7} {'g_i':>8} {'d_i':>8} "
        f"{'g_m':>7} {'d_m':>8} {'th_c':>6} {'th_d':>6} poff iti wall_m"
    )
    print(hdr)
    for r in rows:
        print(
            f"{r['arm']:8} {r['seed']:4d} {r['recorded']:7.3f} "
            f"{r['W_ii']:7.4f} {r['tau_I']/1000:6.2f} "
            f"{r['W_mm']:7.4f} {r['tau_M']:6.1f} "
            f"{r['W_pp']:7.5f} {r['g_i']:8.3g} {r['d_i']:8.3g} "
            f"{r['g_m']:7.3g} {r['d_m']:8.3g} "
            f"{r['theta_c']:6.3f} {r['theta_d']:6.3f} "
            f"{int(r['p_offset_always_on'])} {int(r['iti_penalty'])} "
            f"{(r['wall_s'] or 0)/60:6.1f}"
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
    stim_ref = Path(next(r["path"] for r in rows if r["arm"] == "baseline" and r["seed"] == 101))
    mp0, _ = load_plot_model(stim_ref)
    stim_bundle = make_shared_stimuli(mp0, bps=20, seed=12345)
    print(f"shared stim from {stim_ref.parent.name}")

    evals = []
    jobs = []
    for r in rows:
        jobs.append((r, None, None, "own"))
        if r["arm"] == "noiti":
            jobs.append((r, True, None, "canon_iti"))
        if r["arm"] == "poffset":
            jobs.append((r, None, False, "gated"))

    print(f"\n=== shared-stim eval ({len(jobs)} runs) ===")
    print(
        f"{'tag':12} {'seed':>4} {'rec':>7} {'traj':>7} {'prior':>7} "
        f"{'Lw':>7} {'LS':>7} {'fair':>7} poff iti"
    )
    for r, force_iti, force_po, tag in jobs:
        ev = eval_one(
            Path(r["path"]), stim_bundle, mean_data, prior_regions, avg_mean_R,
            force_iti_penalty=force_iti, force_p_offset=force_po,
        )
        rec = {**r, **ev, "eval_tag": tag}
        evals.append(rec)
        print(
            f"{r['arm']+':'+tag:12} {r['seed']:4d} {ev['recorded']:7.3f} "
            f"{ev['eval_traj']:7.3f} {ev['eval_prior']:7.3f} "
            f"{ev['eval_Lw']:7.3f} {fmt(ev['eval_LS'])} {fmt(ev['eval_fair'])} "
            f"{int(ev['eval_p_offset'])} {int(ev['eval_iti_penalty'])}",
            flush=True,
        )

    out = ROOT / "scripts" / "_tmp_ablation_fair_eval.json"
    payload = {"params": rows, "evals": evals}
    # paths only; drop non-serializable
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
