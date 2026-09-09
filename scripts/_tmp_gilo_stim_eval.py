"""Shared-stim fair eval for g_i-floor rerun (im150stim + stimonly).

bps=20, stim seed 12345, stim from baseline s101.
Score each arm as fitted and with flags cleared (production stim×choice
T=72 / 40-bin). Baseline regular is scored at production, 150+stim, and
legacy+stim.
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
    prior_stratum_of,
    prior_window_ms_of,
    run_model,
)

BASE = Path.home() / (
    "Downloads/ONE/openalyx.internationalbrainlab.org/models"
)
SEEDS = (7, 12, 34, 45, 89, 101, 303, 333)
OUT = BASE / "stageB_hold_s89_gilo_stim_eval.json"


def latest_final(d: Path) -> Path:
    finals = sorted(d.glob("weights_final_*.json"))
    if not finals:
        raise FileNotFoundError(d)
    return finals[-1]


def score(mp, results, steps_before_obs, mean_data, prior_regions, avg_mean_R):
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
    raw_ls = compute_sse_stim_right(S_avg, avg_mean_R, baseline_R=0)["total_loss"]
    L_S = float(raw_ls) if np.isfinite(raw_ls) else None
    return {
        "prior_window_ms": prior_window_ms_of(mp),
        "prior_stratum": prior_stratum_of(mp),
        "eval_traj": traj,
        "eval_prior": prior,
        "eval_Lw": traj + prior,
        "eval_LS": L_S,
        "eval_fair": (traj + prior + L_S) if L_S is not None else None,
        "gof_prior": float(loss_prior.get("gof", float("nan"))),
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
    W = meta.get("W") or {}
    return mp, meta, results, steps_before_obs, {
        "recorded": float(meta.get("loss", np.nan)),
        "g_i": float(mp["g_i"]),
        "d_i": float(mp["d_i"]),
        "g_m": float(mp["g_m"]),
        "W_ii": float(W.get("W_ii", mp.get("W_ii", np.nan))),
        "W_mm": float(W.get("W_mm", mp.get("W_mm", np.nan))),
        "W_pp": float(W.get("W_pp", mp.get("W_pp", np.nan))),
        "theta_c": float(next(iter(th["concordant"].values()))),
        "theta_d": float(next(iter(th["discordant"].values()))),
        "json": jp.name,
        "json_prior_window_ms": (meta.get("model_params") or {}).get(
            "prior_window_ms"
        ),
        "json_prior_stratum": (meta.get("model_params") or {}).get(
            "prior_stratum"
        ),
    }


def settings_for(arm):
    if arm == "im150stim":
        return (("150stim", 150.0, "stim"), ("prod", None, None))
    if arm == "stimonly":
        return (("stimonly", None, "stim"), ("prod", None, None))
    return (
        ("prod", None, None),
        ("150stim", 150.0, "stim"),
        ("stimonly", None, "stim"),
    )


def main():
    ensure_fit_data_links_paper()
    _, mean_data = load_mean_data_results()
    _, avg_mean_R = load_avg_mean_r()
    prior_regions = {
        "int_regs_choice": int_regs, "int_regs_stim": int_regs,
        "move_regs_choice": move_regs, "move_regs_stim": move_regs,
        "stim_regs": ["VISpm", "FRP", "VISal"],
    }
    stim_ref = latest_final(
        BASE / "weights_run_fj_stageB_hold_s89_regular_mask12-13_s101"
    )
    mp0, _ = load_plot_model(stim_ref)
    stim_bundle = make_shared_stimuli(mp0, bps=20, seed=12345)
    print(f"HAVE_NUMBA={mf._HAVE_NUMBA}  stim from {stim_ref.parent.name}",
          flush=True)

    rows = []
    print(
        f"{'arm':10} {'seed':>4} {'set':>8} {'rec':>7} {'traj':>7} {'prior':>7} "
        f"{'fair':>7} {'gi':>8} {'di':>6} {'θc':>5} {'θd':>5}",
        flush=True,
    )
    for arm, prefix in (
        ("im150stim", "weights_run_fj_stageB_hold_s89_im150stim_regular_mask12-13"),
        ("stimonly", "weights_run_fj_stageB_hold_s89_stimonly_regular_mask12-13"),
        ("base", "weights_run_fj_stageB_hold_s89_regular_mask12-13"),
    ):
        for seed in SEEDS:
            jp = latest_final(BASE / f"{prefix}_s{seed}")
            mp, meta, results, sbo, info = sim_one(jp, stim_bundle)
            for label, win, stratum in settings_for(arm):
                mp_sc = dict(mp)
                mp_sc["prior_window_ms"] = win
                mp_sc["prior_stratum"] = stratum
                ev = {**info, **score(
                    mp_sc, results, sbo, mean_data, prior_regions, avg_mean_R,
                )}
                rec = {"arm": arm, "seed": seed, "setting": label, **ev}
                rows.append(rec)
                print(
                    f"{arm:10} {seed:4d} {label:>8} {ev['recorded']:7.3f} "
                    f"{ev['eval_traj']:7.3f} {ev['eval_prior']:7.3f} "
                    f"{ev['eval_fair']:7.3f} "
                    f"{ev['g_i']:8.3g} {ev['d_i']:6.1f} "
                    f"{ev['theta_c']:5.3f} {ev['theta_d']:5.3f}",
                    flush=True,
                )

    OUT.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
