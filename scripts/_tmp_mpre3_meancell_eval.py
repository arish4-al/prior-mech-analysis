"""Shared-stim eval for mpre3 × meancell (80 ms full, 150 regular, 150 full).

bps=20, stim seed 12345, stim from regular s101. Rank at m_pre_weight=1
(same as test 5). As-fitted traj uses m_pre=3. Full arms include unsplit-80 S.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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
from _fit_data import load_avg_mean_r, load_s_unsplit80  # noqa: E402
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
OUT = BASE / "stageB_hold_s89_mpre3_meancell_eval.json"
TS = re.compile(r"(\d{8}-\d{6})")


def newest_final(d: Path) -> Path:
    finals = list(d.glob("weights_final_*.json"))
    if not finals:
        raise FileNotFoundError(d)

    def key(p):
        m = TS.search(p.name)
        return (m.group(1) if m else "", p.stat().st_mtime)

    return max(finals, key=key)


def gof_win(loss_traj, vn, win):
    g = (loss_traj.get("gof") or {}).get(vn) or {}
    v = g.get(win)
    return float(v) if v is not None and np.isfinite(v) else float("nan")


def stim_IM_at(results, steps_before_obs, mp, tqs=(0, 40, 60, 70, 80, 110, 150)):
    T, _, _ = resolve_prior_distance_window(mp, T=72, plot_window=80)
    out = prior_distance_I_M_both_alignments(
        results, steps_before_obs, T=T, metric="l2",
        include_all_trials=True, lump_all=False,
    )
    I = np.asarray(out["I"]["start"], float)
    M = np.asarray(out["M"]["start"], float)
    win = prior_window_ms_of(mp)
    if win is None:
        dt = float(mp.get("dt", 2.0) or 2.0)
        t = np.arange(len(I), dtype=float) * dt
    else:
        t = np.linspace(0.0, float(win), len(I))
    pts = {}
    for tq in tqs:
        pts[f"I{int(tq)}"] = float(np.interp(tq, t, I))
        pts[f"M{int(tq)}"] = float(np.interp(tq, t, M))
    return pts


def score(mp, results, steps_before_obs, mean_data, prior_regions, avg_mean_R,
          stim_curve, include_stim, m_pre_w):
    mp = dict(mp)
    mp["m_pre_weight"] = float(m_pre_w)
    sim_out = mean_by_condition(results, steps_before_obs)
    loss_traj = loss_plot_diff_by_condition_with_data(
        sim_out, mp, var_names=("I", "P", "M"),
        mean_data_results=mean_data, plot=False,
    )
    T_prior, plot_win, _ = resolve_prior_distance_window(mp, T=72, plot_window=80)
    kw = dict(
        regions=prior_regions, results=results, model_params=mp,
        steps_before_obs=steps_before_obs, T=T_prior, model_metric="l2",
        timeframes=("act_block_duringstim", "act_block_duringchoice"),
        ptype="p_mean_c", plot_window=plot_win, reload=False,
        label_A="integrator", label_B="move", do_plot=False,
        plot_shifted=False, ylim=None, scale_factors=[1, 1, 1],
        include_all_trials=True, plot_stim=include_stim, lump_all=False,
        include_stim=include_stim,
    )
    if include_stim:
        kw["stim_curve_path"] = str(stim_curve)
    loss_prior = loss_prior_effect(**kw)
    ds = loss_prior.get("act_block_duringstim") or {}
    traj = float(loss_traj["total"])
    prior = float(loss_prior["total"])
    s_nsse = float(ds.get("stim", 0.0) or 0.0) if include_stim else 0.0
    im = prior - s_nsse if include_stim else prior
    S_avg = mean_S_by_contrast(results, steps_before_obs)
    raw_ls = compute_sse_stim_right(S_avg, avg_mean_R, baseline_R=0)["total_loss"]
    L_S = float(raw_ls) if np.isfinite(raw_ls) else None
    fair = (traj + prior + L_S) if L_S is not None else None
    return {
        "m_pre_weight": float(m_pre_w),
        "score_prior_window_ms": prior_window_ms_of(mp),
        "score_prior_stratum": prior_stratum_of(mp),
        "eval_traj": traj,
        "eval_prior": prior,
        "eval_prior_S": s_nsse,
        "eval_IM": im,
        "eval_LS": L_S,
        "eval_fair": fair,
        "gof_M_pre": gof_win(loss_traj, "M", "pre"),
        "gof_M_post": gof_win(loss_traj, "M", "post"),
        "gof_I_pre": gof_win(loss_traj, "I", "pre"),
        "gof_I_post": gof_win(loss_traj, "I", "post"),
        "gof_prior": float(loss_prior.get("gof", float("nan"))),
        "include_stim": include_stim,
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
        f"{'arm':8} {'seed':>4} {'w':>3} {'rec':>7} {'traj':>7} {'IM':>7} "
        f"{'S':>7} {'LS':>7} {'fair':>7} {'gi':>8} {'gs':>8} {'ds':>8}",
        flush=True,
    )
    for arm, (prefix, include_stim) in ARMS.items():
        for seed in SEEDS:
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
                "g_i": float(mp["g_i"]),
                "g_m": float(mp["g_m"]),
                "g_s": float(mp["g_s"]),
                "d_i": float(mp["d_i"]),
                "d_m": float(mp["d_m"]),
                "d_s": float(mp["d_s"]),
                "theta_c": float(next(iter(th["concordant"].values()))),
                "theta_d": float(next(iter(th["discordant"].values()))),
                "json": jp.name,
                "json_m_pre_weight": (meta.get("model_params") or {}).get(
                    "m_pre_weight"
                ),
                "json_prior_window_ms": (meta.get("model_params") or {}).get(
                    "prior_window_ms"
                ),
            }
            shape = stim_IM_at(results, steps_before_obs, mp)
            for w in (3.0, 1.0):
                ev = score(
                    mp, results, steps_before_obs, mean_data, prior_regions,
                    avg_mean_R, stim_curve, include_stim, w,
                )
                rec = {"arm": arm, "seed": seed, **info, **ev}
                if w == 1.0:
                    rec.update(shape)
                rows.append(rec)
                print(
                    f"{arm:8} {seed:4d} {int(w):3d} {info['recorded']:7.3f} "
                    f"{ev['eval_traj']:7.3f} {ev['eval_IM']:7.3f} "
                    f"{ev['eval_prior_S']:7.3f} {ev['eval_LS']:7.3f} "
                    f"{ev['eval_fair']:7.3f} {info['g_i']:8.3g} "
                    f"{info['g_s']:8.3g} {info['d_s']:8.3g}",
                    flush=True,
                )
            OUT.write_text(json.dumps({"eval": rows}, indent=2, default=str))
    print(f"\nwrote {OUT}  n={len(rows)}")


if __name__ == "__main__":
    main()
