"""Shared-stim eval for s89ft d_i punch (di / wiigi) vs wii s7.

bps=20, stim seed 12345, stim from regular s101. Rank at extras=0,
m_pre_weight=1, include_stim, current mean_c ‖Δ‖, window unset (~80 ms).
JSON recorded loss includes the late-I hinge — not comparable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _tmp_s89ft_imshape_eval import (  # noqa: E402
    BASE,
    _finite0,
    gof_win,
    newest_final,
    overshoot_nsse,
    stim_IM_at,
)
from plot_best_fit_results import (  # noqa: E402
    ensure_fit_data_links_paper,
    load_mean_data_results,
    load_plot_model,
    make_shared_stimuli,
)
from _fit_data import load_avg_mean_r, load_s_unsplit80  # noqa: E402
from fit_joint import _choice_i_late_undershoot_nsse  # noqa: E402
import model_functions as mf  # noqa: E402
from model_functions import (  # noqa: E402
    compute_sse_stim_right,
    int_regs,
    loss_plot_diff_by_condition_with_data,
    loss_prior_effect,
    mean_by_condition,
    mean_S_by_contrast,
    move_regs,
    prior_distance_I_M_both_alignments,
    prior_stratum_of,
    prior_window_ms_of,
    resolve_prior_distance_window,
    run_model,
)

ARMS = {
    "wii7": (
        "weights_run_fj_stageB_hold_s89_full_s89ft_imshape_wii_meancell_full_mask1-3-4-6-10-11-12-13-14-15-16-17-18-19-20",
        (7,),
    ),
    "di": (
        "weights_run_fj_stageB_hold_s89_full_s89ft_di_choicei_meancell_full_mask0-1-2-3-4-5-6-7-9-10-11-12-13-14-15-16-17-18-19-20",
        (89, 7, 12, 45),
    ),
    "wiigi": (
        "weights_run_fj_stageB_hold_s89_full_s89ft_wiigi_choicei_meancell_full_mask1-3-4-10-11-12-13-14-15-16-17-18-19-20",
        (89, 7, 12, 45),
    ),
}
OUT = BASE / "stageB_hold_s89_s89ft_di_choicei_eval.json"


def choice_I_at(results, steps_before_obs, mp, tqs=(-80, -40, -20, 0)):
    T, _, _ = resolve_prior_distance_window(mp, T=72, plot_window=80)
    out = prior_distance_I_M_both_alignments(
        results, steps_before_obs, T=T, metric="l2",
        include_all_trials=True, lump_all=False,
    )
    I = np.asarray(out["I"]["action"], float)
    win = prior_window_ms_of(mp)
    if win is None:
        dt = float(mp.get("dt", 2.0) or 2.0)
        t = (np.arange(len(I), dtype=float) - (len(I) - 1)) * dt
    else:
        t = np.linspace(-float(win), 0.0, len(I))
    pts = {}
    for tq in tqs:
        key = f"Iact{abs(int(tq))}" if tq != 0 else "Iact0"
        pts[key] = float(np.interp(tq, t, I))
    return pts


def score(mp, results, steps_before_obs, mean_data, prior_regions, avg_mean_R,
          stim_curve):
    mp = dict(mp)
    mp["m_pre_weight"] = 1.0
    mp["choice_im_extra_weight"] = 0.0
    mp["m_stim_overshoot_weight"] = 0.0
    mp["choice_i_late_weight"] = 0.0
    sim_out = mean_by_condition(results, steps_before_obs)
    loss_traj = loss_plot_diff_by_condition_with_data(
        sim_out, mp, var_names=("I", "P", "M"),
        mean_data_results=mean_data, plot=False,
    )
    T_prior, plot_win, _ = resolve_prior_distance_window(mp, T=72, plot_window=80)
    loss_prior = loss_prior_effect(
        regions=prior_regions, results=results, model_params=mp,
        steps_before_obs=steps_before_obs, T=T_prior, model_metric="l2",
        timeframes=("act_block_duringstim", "act_block_duringchoice"),
        ptype="p_mean_c", plot_window=plot_win, reload=False,
        label_A="integrator", label_B="move", do_plot=False,
        plot_shifted=False, ylim=None, scale_factors=[1, 1, 1],
        include_all_trials=True, plot_stim=True, lump_all=False,
        include_stim=True, stim_curve_path=str(stim_curve),
    )
    ds = loss_prior.get("act_block_duringstim") or {}
    ch = loss_prior.get("act_block_duringchoice") or {}
    traj = float(loss_traj["total"])
    prior = float(loss_prior["total"])
    s_nsse = float(ds.get("stim", 0.0) or 0.0)
    i_stim = _finite0(ds.get("integrator"))
    m_stim = _finite0(ds.get("move"))
    i_ch = _finite0(ch.get("integrator"))
    m_ch = _finite0(ch.get("move"))
    im = prior - s_nsse
    ov = overshoot_nsse(loss_prior, from_ms=40.0)
    late = _choice_i_late_undershoot_nsse(loss_prior, last_ms=40.0)
    S_avg = mean_S_by_contrast(results, steps_before_obs)
    raw_ls = compute_sse_stim_right(S_avg, avg_mean_R, baseline_R=0)["total_loss"]
    L_S = float(raw_ls) if np.isfinite(raw_ls) else None
    fair = (traj + prior + L_S) if L_S is not None else None
    tot_late = (
        fair + late if fair is not None and np.isfinite(late) else None
    )
    return {
        "m_pre_weight": 1.0,
        "score_prior_window_ms": prior_window_ms_of(mp),
        "score_prior_stratum": prior_stratum_of(mp),
        "eval_traj": traj,
        "eval_prior": prior,
        "eval_prior_S": s_nsse,
        "eval_IM": im,
        "eval_I_stim": i_stim,
        "eval_M_stim": m_stim,
        "eval_I_choice": i_ch,
        "eval_M_choice": m_ch,
        "eval_M_overshoot40": ov,
        "eval_I_late40": late,
        "eval_LS": L_S,
        "eval_fair": fair,
        "eval_tot_late1": tot_late,
        "gof_M_pre": gof_win(loss_traj, "M", "pre"),
        "gof_M_post": gof_win(loss_traj, "M", "post"),
        "gof_I_pre": gof_win(loss_traj, "I", "pre"),
        "gof_I_post": gof_win(loss_traj, "I", "post"),
        "gof_prior": float(loss_prior.get("gof", float("nan"))),
        "include_stim": True,
    }


def main():
    ensure_fit_data_links_paper()
    _, mean_data = load_mean_data_results()
    _, avg_mean_R = load_avg_mean_r()
    stim_curve, payload = load_s_unsplit80()
    prior_regions = {
        "int_regs_choice": int_regs, "int_regs_stim": int_regs,
        "move_regs_choice": move_regs, "move_regs_stim": move_regs,
        "stim_regs": list(payload.get("regs_stim") or ["VISpm", "FRP", "VISal"]),
    }
    stim_ref = newest_final(
        BASE / "weights_run_fj_stageB_hold_s89_regular_mask12-13_s101"
    )
    mp0, _ = load_plot_model(stim_ref)
    stim_bundle = make_shared_stimuli(mp0, bps=20, seed=12345)
    (
        stimuli, trial_strengths, trial_sides, block_sides,
        steps_before_obs, bps,
    ) = stim_bundle
    print(
        f"HAVE_NUMBA={mf._HAVE_NUMBA}  stim from {stim_ref.parent.name}  "
        f"S sidecar={stim_curve.name}",
        flush=True,
    )
    rows = []
    print(
        f"{'arm':6} {'seed':>4} {'rec':>7} {'fair':>7} {'late1':>7} "
        f"{'traj':>7} {'IM':>7} {'S':>6} {'LS':>7} "
        f"{'Ich':>6} {'Ilate':>6} {'I0':>6} {'gi':>8} {'di':>7}",
        flush=True,
    )
    for arm, (prefix, seeds) in ARMS.items():
        for seed in seeds:
            d = BASE / f"{prefix}_s{seed}"
            jp = newest_final(d)
            mp, meta = load_plot_model(jp)
            results = run_model(
                "data", stimuli, trial_strengths, trial_sides, block_sides, bps,
                steps_before_obs=steps_before_obs, verbose=False,
                backend="numba", **mp,
            )
            th = mp["action_thresholds"]
            info = {
                "recorded": float(meta.get("loss", np.nan)),
                "W_ii": float(mp["W_ii"]),
                "W_mm": float(mp["W_mm"]),
                "W_mi": float(mp["W_mi"]),
                "g_i": float(mp["g_i"]),
                "g_m": float(mp["g_m"]),
                "g_s": float(mp["g_s"]),
                "d_i": float(mp["d_i"]),
                "d_m": float(mp["d_m"]),
                "d_s": float(mp["d_s"]),
                "theta_c": float(next(iter(th["concordant"].values()))),
                "theta_d": float(next(iter(th["discordant"].values()))),
                "json": jp.name,
                "frozen_idx": meta.get("frozen_idx"),
            }
            shape = stim_IM_at(results, steps_before_obs, mp)
            act = choice_I_at(results, steps_before_obs, mp)
            ev = score(
                mp, results, steps_before_obs, mean_data, prior_regions,
                avg_mean_R, stim_curve,
            )
            rec = {"arm": arm, "seed": seed, **info, **ev, **shape, **act}
            rows.append(rec)
            print(
                f"{arm:6} {seed:4d} {info['recorded']:7.3f} "
                f"{ev['eval_fair']:7.3f} {ev['eval_tot_late1']:7.3f} "
                f"{ev['eval_traj']:7.3f} {ev['eval_IM']:7.3f} "
                f"{ev['eval_prior_S']:6.3f} {ev['eval_LS']:7.3f} "
                f"{ev['eval_I_choice']:6.3f} {ev['eval_I_late40']:6.3f} "
                f"{act['Iact0']:6.3f} {info['g_i']:8.3g} {info['d_i']:7.3f}",
                flush=True,
            )
            OUT.write_text(json.dumps({"eval": rows}, indent=2, default=str))
    print(f"\nwrote {OUT}  n={len(rows)}")


if __name__ == "__main__":
    main()
