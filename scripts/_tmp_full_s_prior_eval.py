"""Shared-stim fair eval for Stage B full+S 1e-12-floor campaign.

bps=20, stim seed 12345, stim from baseline regular s101. Score each arm
as fitted and on production (legacy T=72 / stim×choice). S prior nSSE
always uses the unsplit-80 sidecar (stratum_s=stim).
"""
from __future__ import annotations

import json
import os
import re
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
from _fit_data import FIT_S_UNSPLIT80, load_avg_mean_r, load_s_unsplit80  # noqa: E402
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
    "full": "weights_run_fj_stageB_hold_s89_full_full_masknone",
    "im150": "weights_run_fj_stageB_hold_s89_full_im150_full_masknone",
    "im150stim": "weights_run_fj_stageB_hold_s89_full_im150stim_full_masknone",
    "stimonly": "weights_run_fj_stageB_hold_s89_full_stimonly_full_masknone",
    "regular": "weights_run_fj_stageB_hold_s89_regular_mask12-13",
    "im150_meancell": (
        "weights_run_fj_stageB_hold_s89_full_im150_meancell_full_masknone"
    ),
}
NEW = BASE / "new"


def resolve_run(prefix: str, seed: int) -> Path | None:
    for root in (BASE, NEW):
        d = root / f"{prefix}_s{seed}"
        if d.is_dir():
            return d
    return None
OUT = Path(os.environ.get(
    "EVAL_OUT",
    str(BASE / "stageB_hold_s89_full_s_prior_1e12_eval.json"),
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


def score(mp, results, steps_before_obs, mean_data, prior_regions, avg_mean_R,
          stim_curve):
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
        include_stim=True, stim_curve_path=str(stim_curve),
    )
    ds = loss_prior.get("act_block_duringstim") or {}
    dc = loss_prior.get("act_block_duringchoice") or {}
    traj = float(loss_traj["total"])
    prior = float(loss_prior["total"])
    s_nsse = float(ds.get("stim", 0.0) or 0.0)
    im = prior - s_nsse
    S_avg = mean_S_by_contrast(results, steps_before_obs)
    raw_ls = compute_sse_stim_right(S_avg, avg_mean_R, baseline_R=0)["total_loss"]
    L_S = float(raw_ls) if np.isfinite(raw_ls) else None
    return {
        "score_prior_window_ms": prior_window_ms_of(mp),
        "score_prior_stratum": prior_stratum_of(mp),
        "eval_traj": traj,
        "eval_prior": prior,
        "eval_prior_I_stim": float(ds.get("integrator", float("nan"))),
        "eval_prior_M_stim": float(ds.get("move", float("nan"))),
        "eval_prior_S": s_nsse,
        "eval_prior_I_choice": float(dc.get("integrator", float("nan"))),
        "eval_prior_M_choice": float(dc.get("move", float("nan"))),
        "eval_IM": im,
        "eval_LS": L_S,
        "eval_Lw_noS": traj + im,
        "eval_fair": (traj + prior + L_S) if L_S is not None else None,
        "eval_fair_noS": (traj + im + L_S) if L_S is not None else None,
        "gof_prior": float(loss_prior.get("gof", float("nan"))),
    }


def stim_IM_at(results, steps_before_obs, mp, tqs=(0, 40, 60, 70, 80, 110, 150)):
    T, _, _ = resolve_prior_distance_window(mp, T=72, plot_window=80)
    out = prior_distance_I_M_both_alignments(
        results, steps_before_obs, T=T, metric="l2",
        include_all_trials=True, lump_all=False,
    )
    I = np.asarray(out["I"]["start"], float)
    M = np.asarray(out["M"]["start"], float)
    win = float(prior_window_ms_of(mp) or 144.0)
    t = np.linspace(0.0, win, len(I))
    pts = {}
    for tq in tqs:
        pts[f"I{int(tq)}"] = float(np.interp(tq, t, I))
        pts[f"M{int(tq)}"] = float(np.interp(tq, t, M))
    return pts


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
    g = meta.get("g") or {}
    dd = meta.get("d") or {}
    jmp = meta.get("model_params") or {}
    return mp, meta, results, steps_before_obs, {
        "recorded": float(meta.get("loss", np.nan)),
        "g_i": float(mp["g_i"]),
        "g_m": float(mp["g_m"]),
        "g_s": float(mp["g_s"]),
        "d_i": float(mp["d_i"]),
        "d_m": float(mp["d_m"]),
        "d_s": float(mp["d_s"]),
        "W_ii": float(W.get("W_ii", mp.get("W_ii", np.nan))),
        "W_pp": float(W.get("W_pp", mp.get("W_pp", np.nan))),
        "W_mm": float(W.get("W_mm", mp.get("W_mm", np.nan))),
        "theta_c": float(next(iter(th["concordant"].values()))),
        "theta_d": float(next(iter(th["discordant"].values()))),
        "json": jp.name,
        "json_prior_window_ms": jmp.get("prior_window_ms"),
        "json_prior_stratum": jmp.get("prior_stratum"),
        "json_g_i": float(g.get("g_i", mp["g_i"])),
        "json_g_s": float(meta.get("g_s", g.get("g_s", mp["g_s"]))),
        "json_d_s": float(meta.get("d_s", dd.get("d_s", mp["d_s"]))),
    }


def settings_for(arm, mp):
    fitted_win = prior_window_ms_of(mp)
    fitted_str = prior_stratum_of(mp)
    out = [("asfit", fitted_win, fitted_str)]
    if fitted_win is not None or fitted_str not in (None, "stim_choice"):
        out.append(("prod", None, None))
    return out


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
    print(
        f"HAVE_NUMBA={mf._HAVE_NUMBA}  stim from {stim_ref.parent.name}  "
        f"S sidecar={stim_curve.name} nreg={len(prior_regions['stim_regs'])}",
        flush=True,
    )

    only = [a for a in os.environ.get("ONLY_ARMS", "").split() if a]
    arms = {k: v for k, v in ARMS.items() if (not only or k in only)}
    rows = []
    print(
        f"{'arm':10} {'seed':>4} {'set':>6} {'rec':>7} {'traj':>7} {'IM':>7} "
        f"{'S':>7} {'LS':>7} {'fair':>7} {'noS':>7} {'gs':>8} {'ds':>8} {'gi':>7}",
        flush=True,
    )
    for arm, prefix in arms.items():
        for seed in SEEDS:
            d = resolve_run(prefix, seed)
            if d is None:
                print(f"{arm:10} {seed:4d} MISSING", flush=True)
                continue
            jp = newest_final(d)
            mp, meta, results, sbo, info = sim_one(jp, stim_bundle)
            shape = stim_IM_at(results, sbo, mp)
            for label, win, stratum in settings_for(arm, mp):
                mp_sc = dict(mp)
                mp_sc["prior_window_ms"] = win
                mp_sc["prior_stratum"] = stratum
                ev = {**info, **score(
                    mp_sc, results, sbo, mean_data, prior_regions, avg_mean_R,
                    stim_curve,
                )}
                rec = {"arm": arm, "seed": seed, "setting": label, **ev}
                if label == "asfit":
                    rec.update(shape)
                rows.append(rec)
                extra = ""
                if label == "asfit":
                    extra = (
                        f"  M40={shape['M40']:.3f} M70={shape['M70']:.3f} "
                        f"M80={shape['M80']:.3f} M150={shape['M150']:.3f}"
                    )
                print(
                    f"{arm:10} {seed:4d} {label:>6} {ev['recorded']:7.3f} "
                    f"{ev['eval_traj']:7.3f} {ev['eval_IM']:7.3f} "
                    f"{ev['eval_prior_S']:7.3f} {ev['eval_LS']:7.3f} "
                    f"{ev['eval_fair']:7.3f} {ev['eval_fair_noS']:7.3f} "
                    f"{ev['g_s']:8.3g} {ev['d_s']:8.3g} {ev['g_i']:7.2f}"
                    f"{extra}",
                    flush=True,
                )

    if only and OUT.is_file() and "EVAL_OUT" not in os.environ:
        prev = json.loads(OUT.read_text())
        kept = [r for r in prev.get("eval", []) if r.get("arm") not in only]
        rows = kept + rows
        print(f"merged {only} into existing eval ({len(kept)} kept)")
    have = {r.get("arm") for r in rows}
    payload_out = {
        "protocol": (
            "bps=20 stim_seed=12345 stim_from=regular_s101 "
            "include_stim=True stratum_s=stim floor=1e-12"
        ),
        "stim_curve": str(stim_curve),
        "n_reg_S": len(prior_regions["stim_regs"]),
        "missing_arms": [a for a in ARMS if a not in have],
        "eval": rows,
    }
    OUT.write_text(json.dumps(payload_out, indent=2, default=str))
    print(f"\nwrote {OUT}  n={len(rows)}")


if __name__ == "__main__":
    main()
