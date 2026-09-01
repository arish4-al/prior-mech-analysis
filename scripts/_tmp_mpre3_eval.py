"""Shared-stim eval for Stage B mpre3 vs baseline regular.

bps=20, stim seed 12345, from baseline s101 retinal (same as ablation fair eval).
Each model is scored at m_pre_weight=1 (old traj metric) and =3 (as fitted).
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

BASE = Path.home() / (
    "Downloads/ONE/openalyx.internationalbrainlab.org/models"
)
SEEDS = (7, 12, 34, 45, 89, 101, 303, 333)
OUT = BASE / "stageB_hold_s89_mpre3_eval.json"


def latest_final(d: Path) -> Path:
    finals = sorted(d.glob("weights_final_*.json"))
    if not finals:
        raise FileNotFoundError(d)
    return finals[-1]


def gof_win(loss_traj, vn, win):
    g = (loss_traj.get("gof") or {}).get(vn) or {}
    v = g.get(win)
    return float(v) if v is not None and np.isfinite(v) else float("nan")


def score(mp, results, steps_before_obs, mean_data, prior_regions, avg_mean_R, m_pre_w):
    mp = dict(mp)
    mp["m_pre_weight"] = float(m_pre_w)
    sim_out = mean_by_condition(results, steps_before_obs)
    loss_traj = loss_plot_diff_by_condition_with_data(
        sim_out, mp, var_names=("I", "P", "M"),
        mean_data_results=mean_data, plot=False,
    )
    loss_prior = loss_prior_effect(
        regions=prior_regions, results=results, model_params=mp,
        steps_before_obs=steps_before_obs, T=72, model_metric="l2",
        timeframes=("act_block_duringstim", "act_block_duringchoice"),
        ptype="p_mean_c", plot_window=80, reload=False,
        label_A="integrator", label_B="move", do_plot=False,
        plot_shifted=False, ylim=None, scale_factors=[1, 1, 1],
        include_all_trials=True, plot_stim=False, lump_all=False,
    )
    traj = float(loss_traj["total"])
    prior = float(loss_prior["total"])
    S_avg = mean_S_by_contrast(results, steps_before_obs)
    sse = compute_sse_stim_right(S_avg, avg_mean_R, baseline_R=0)
    raw_ls = sse["total_loss"]
    L_S = float(raw_ls) if np.isfinite(raw_ls) else None
    return {
        "m_pre_weight": float(m_pre_w),
        "eval_traj": traj,
        "eval_prior": prior,
        "eval_Lw": traj + prior,
        "eval_LS": L_S,
        "eval_fair": (traj + prior + L_S) if L_S is not None else None,
        "gof_M_pre": gof_win(loss_traj, "M", "pre"),
        "gof_M_post": gof_win(loss_traj, "M", "post"),
        "gof_I_pre": gof_win(loss_traj, "I", "pre"),
        "gof_I_post": gof_win(loss_traj, "I", "post"),
    }


def sim_one(jp: Path, stim_bundle):
    mp, meta = load_plot_model(jp)
    (
        stimuli, trial_strengths, trial_sides, block_sides,
        steps_before_obs, bps,
    ) = stim_bundle
    results = run_model(
        "data",
        stimuli, trial_strengths, trial_sides, block_sides, bps,
        steps_before_obs=steps_before_obs, verbose=False, backend="numba",
        **mp,
    )
    th = mp["action_thresholds"]
    return mp, meta, results, steps_before_obs, {
        "recorded": float(meta.get("loss", np.nan)),
        "g_i": float(mp["g_i"]),
        "d_i": float(mp["d_i"]),
        "theta_c": float(next(iter(th["concordant"].values()))),
        "theta_d": float(next(iter(th["discordant"].values()))),
        "json": jp.name,
    }


def main():
    ensure_fit_data_links_paper()
    _, mean_data = load_mean_data_results()
    _, avg_mean_R = load_avg_mean_r()
    prior_regions = {
        "int_regs_choice": int_regs,
        "int_regs_stim": int_regs,
        "move_regs_choice": move_regs,
        "move_regs_stim": move_regs,
        "stim_regs": ["VISpm", "FRP", "VISal"],
    }
    stim_ref = latest_final(BASE / "weights_run_fj_stageB_hold_s89_regular_mask12-13_s101")
    mp0, _ = load_plot_model(stim_ref)
    stim_bundle = make_shared_stimuli(mp0, bps=20, seed=12345)
    print(f"HAVE_NUMBA={mf._HAVE_NUMBA}  stim from {stim_ref.parent.name}")

    rows = []
    print(
        f"{'arm':7} {'seed':>4} {'w':>3} {'rec':>7} {'traj':>7} {'prior':>7} "
        f"{'Lw':>7} {'LS':>7} {'fair':>7} {'Mpre':>6} {'Mpost':>6} "
        f"{'gi':>6} {'θc':>5} {'θd':>5}",
        flush=True,
    )
    for arm, prefix in (
        ("mpre3", "weights_run_fj_stageB_hold_s89_mpre3_regular_mask12-13"),
        ("base", "weights_run_fj_stageB_hold_s89_regular_mask12-13"),
    ):
        for seed in SEEDS:
            jp = latest_final(BASE / f"{prefix}_s{seed}")
            mp, meta, results, sbo, info = sim_one(jp, stim_bundle)
            ev3 = {**info, **score(
                mp, results, sbo, mean_data, prior_regions, avg_mean_R, 3.0,
            )}
            ev1 = {**info, **score(
                mp, results, sbo, mean_data, prior_regions, avg_mean_R, 1.0,
            )}
            for w, ev in ((3, ev3), (1, ev1)):
                rec = {"arm": arm, "seed": seed, **ev}
                rows.append(rec)
                ls = ev["eval_LS"]
                fair = ev["eval_fair"]
                print(
                    f"{arm:7} {seed:4d} {w:3d} {ev['recorded']:7.3f} "
                    f"{ev['eval_traj']:7.3f} {ev['eval_prior']:7.3f} "
                    f"{ev['eval_Lw']:7.3f} "
                    f"{ls if ls is None else f'{ls:7.3f}'} "
                    f"{fair if fair is None else f'{fair:7.3f}'} "
                    f"{ev['gof_M_pre']:6.3f} {ev['gof_M_post']:6.3f} "
                    f"{ev['g_i']:6.1f} {ev['theta_c']:5.3f} {ev['theta_d']:5.3f}",
                    flush=True,
                )

    OUT.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
