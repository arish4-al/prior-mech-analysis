"""Shared-stim fair eval for Stage B im150_meancell (mean_c ‖Δ‖).

bps=20, stim seed 12345, stim from baseline s101. Score meancell and the
09-07 im150 as fitted (150 ms) and on the production window; baseline at
production and at 150 ms. Also dump I/M curve 80 vs end and an S-metric audit.
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

from plot_best_fit_results import (  # noqa: E402
    alias_prior_effects,
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
    plot_S_diff_by_contrast_side_with_data,
    prior_distance_I_M_both_alignments,
    prior_window_ms_of,
    resolve_prior_distance_window,
    run_model,
)

BASE = Path.home() / (
    "Downloads/ONE/openalyx.internationalbrainlab.org/models"
)
SEEDS = (7, 12, 34, 45, 89, 101, 303, 333)
OUT = BASE / "stageB_hold_s89_im150_meancell_eval.json"
AUDIT_OUT = BASE / "stageB_hold_s89_im150_meancell_s_audit.json"


def latest_final(d: Path) -> Path:
    finals = sorted(d.glob("weights_final_*.json"))
    if not finals:
        raise FileNotFoundError(d)
    return finals[-1]


def _finite(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def curve_at(y, idx):
    y = np.asarray(y, dtype=float)
    if y.size == 0 or not np.isfinite(y).any():
        return None
    i = idx if idx >= 0 else y.size + idx
    i = min(max(i, 0), y.size - 1)
    v = float(y[i])
    return v if np.isfinite(v) else None


def im_curve_stats(results, steps_before_obs, mp):
    T, plot_win, _ = resolve_prior_distance_window(mp, T=72, plot_window=80)
    out = prior_distance_I_M_both_alignments(
        results, steps_before_obs, T=T, metric="l2",
        include_all_trials=True, lump_all=False,
    )
    dt = float(mp.get("dt", 2.0) or 2.0)
    i80 = int(round(80.0 / dt))
    stats = {
        "prior_T": int(T),
        "plot_window": float(plot_win),
        "idx80": i80,
    }
    for vn, align in (("I", "start"), ("M", "start"), ("I", "action"), ("M", "action")):
        y = out[vn][align]
        prefix = f"{vn}_{align}"
        stats[f"{prefix}_80"] = curve_at(y, i80)
        stats[f"{prefix}_end"] = curve_at(y, -1)
        a80, aend = stats[f"{prefix}_80"], stats[f"{prefix}_end"]
        if a80 is not None and aend is not None:
            stats[f"{prefix}_dend"] = aend - a80
        else:
            stats[f"{prefix}_dend"] = None
    return stats


def score(mp, results, steps_before_obs, mean_data, prior_regions, avg_mean_R,
          plot_dir=None):
    sim_out = mean_by_condition(results, steps_before_obs)
    loss_traj = loss_plot_diff_by_condition_with_data(
        sim_out, mp, var_names=("I", "P", "M"),
        mean_data_results=mean_data, plot=bool(plot_dir),
        save_dir=str(plot_dir) if plot_dir else None,
    )
    T_prior, plot_win, _ = resolve_prior_distance_window(mp, T=72, plot_window=80)
    loss_prior = loss_prior_effect(
        regions=prior_regions, results=results, model_params=mp,
        steps_before_obs=steps_before_obs, T=T_prior, model_metric="l2",
        timeframes=("act_block_duringstim", "act_block_duringchoice"),
        ptype="p_mean_c", plot_window=plot_win, reload=False,
        label_A="integrator", label_B="move", do_plot=bool(plot_dir),
        plot_shifted=False, ylim=None, scale_factors=[1, 1, 1],
        include_all_trials=True, plot_stim=False, lump_all=False,
        save_dir=str(plot_dir) if plot_dir else None,
    )
    traj = float(loss_traj["total"])
    prior = float(loss_prior["total"])
    S_avg = mean_S_by_contrast(results, steps_before_obs)
    sse = compute_sse_stim_right(S_avg, avg_mean_R, baseline_R=0)
    raw_ls = sse["total_loss"]
    L_S = float(raw_ls) if np.isfinite(raw_ls) else None
    if plot_dir is not None:
        import matplotlib.pyplot as plt
        prior_fig = plt.gcf()
        prior_fig.savefig(
            plot_dir / "prior_effects.png",
            dpi=150, bbox_inches="tight", facecolor="white", transparent=False,
        )
        alias_prior_effects(plot_dir)
        plt.close("all")
        plot_S_diff_by_contrast_side_with_data(
            S_avg, {}, avg_mean_R, baseline=0,
            save_dir=str(plot_dir), ylim=[-0.14, 0.75], yticks=None,
        )
        for n in list(plt.get_fignums()):
            fig = plt.figure(n)
            fig.savefig(
                plot_dir / "S_fit.png",
                dpi=150, bbox_inches="tight", transparent=False,
            )
        plt.close("all")
    stim_tf = loss_prior.get("act_block_duringstim") or {}
    ch_tf = loss_prior.get("act_block_duringchoice") or {}
    per = sse.get("per_contrast") or {}
    ls_per = {}
    if isinstance(per, dict):
        for c, d in per.items():
            ls_per[str(c)] = _finite((d or {}).get("loss"))
    return {
        "prior_window_ms": prior_window_ms_of(mp),
        "eval_traj": traj,
        "eval_prior": prior,
        "eval_Lw": traj + prior,
        "eval_LS": L_S,
        "eval_fair": (traj + prior + L_S) if L_S is not None else None,
        "gof_prior": _finite(loss_prior.get("gof")),
        "gof_S": _finite(sse.get("total_gof_r2")),
        "prior_stim": _finite(stim_tf.get("total")),
        "prior_choice": _finite(ch_tf.get("total")),
        "prior_stim_I": _finite(stim_tf.get("integrator")),
        "prior_stim_M": _finite(stim_tf.get("move")),
        "prior_choice_I": _finite(ch_tf.get("integrator")),
        "prior_choice_M": _finite(ch_tf.get("move")),
        "LS_per_contrast": ls_per,
        **im_curve_stats(results, steps_before_obs, mp),
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
    g = meta.get("g") or {}
    d = meta.get("d") or {}
    return mp, meta, results, steps_before_obs, {
        "recorded": float(meta.get("loss", np.nan)),
        "g_i": float(mp["g_i"]),
        "d_i": float(mp["d_i"]),
        "g_m": float(mp["g_m"]),
        "d_m": float(mp["d_m"]),
        "g_s": float(mp["g_s"]),
        "d_s": float(mp["d_s"]),
        "g_i_json_g": _finite(g.get("g_i")),
        "g_i_model_params": _finite((meta.get("model_params") or {}).get("g_i")),
        "W_ii": float(W.get("W_ii", mp.get("W_ii", np.nan))),
        "W_mm": float(W.get("W_mm", mp.get("W_mm", np.nan))),
        "W_pp": float(W.get("W_pp", mp.get("W_pp", np.nan))),
        "W_is": float(W.get("W_is", mp.get("W_is", np.nan))),
        "theta_c": float(next(iter(th["concordant"].values()))),
        "theta_d": float(next(iter(th["discordant"].values()))),
        "json": jp.name,
        "json_prior_window_ms": (meta.get("model_params") or {}).get("prior_window_ms"),
        "json_prior_stratum": (meta.get("model_params") or {}).get("prior_stratum"),
    }


def audit_s(results, steps_before_obs, avg_mean_R, mp, prior_regions):
    """Retinal L_S vs S prior-distance: shapes, NaN poison, T used, pooling."""
    S_avg = mean_S_by_contrast(results, steps_before_obs)
    buckets = {}
    any_nan = False
    for k, v in S_avg.items():
        a = np.asarray(v)
        nan = bool(np.isnan(a).any())
        any_nan = any_nan or nan
        buckets[str(k)] = {
            "side": int(k[0]), "contrast": float(k[1]),
            "shape": list(a.shape), "nan": nan,
        }
    sse_all = compute_sse_stim_right(S_avg, avg_mean_R, baseline_R=0)
    S_right = {k: v for k, v in S_avg.items() if k[0] == 1}
    sse_right = compute_sse_stim_right(S_right, avg_mean_R, baseline_R=0)
    S_left = {k: v for k, v in S_avg.items() if k[0] == -1}
    left_nan = any(np.isnan(np.asarray(v)).any() for v in S_left.values())
    right_nan = any(np.isnan(np.asarray(v)).any() for v in S_right.values())

    data_shapes = {str(c): list(np.asarray(v).shape) for c, v in avg_mean_R.items()}
    per = sse_right.get("per_contrast") or {}
    per_T = {}
    if isinstance(per, dict):
        for c, d in per.items():
            per_T[str(c)] = {
                "T": (d or {}).get("T"),
                "sse": _finite((d or {}).get("sse")),
                "snr_loss": _finite((d or {}).get("snr_loss")),
                "loss": _finite((d or {}).get("loss")),
                "gof_r2": _finite((d or {}).get("gof_r2")),
            }

    T, plot_win, _ = resolve_prior_distance_window(mp, T=72, plot_window=80)
    s_stim = prior_distance_I_M_both_alignments(
        results, steps_before_obs, T=T, metric="l2",
        include_all_trials=True, lump_all=False, stratum="stim_choice",
        stratum_s="stim",
    )["S"]["start"]
    s_all = prior_distance_I_M_both_alignments(
        results, steps_before_obs, T=T, metric="l2",
        include_all_trials=True, lump_all=True, stratum="all",
        stratum_s="all",
    )["S"]["start"]
    sidecar = ROOT / "fit_targets" / "data_act_block_duringstim_s_unsplit80.npy"
    s_prior = None
    if sidecar.is_file():
        s_prior = loss_prior_effect(
            regions=prior_regions, results=results, model_params=mp,
            steps_before_obs=steps_before_obs, T=T, model_metric="l2",
            timeframes=("act_block_duringstim",),
            ptype="p_mean_c", plot_window=plot_win, reload=False,
            label_A="integrator", label_B="move", do_plot=False,
            plot_shifted=False, ylim=None, scale_factors=[1, 1, 1],
            include_all_trials=True, plot_stim=False, lump_all=False,
            include_stim=True, stim_curve_path=str(sidecar),
        )

    dt = float(mp.get("dt", 2.0) or 2.0)
    i80 = int(round(80.0 / dt))
    return {
        "mean_S_T": 65,
        "avg_mean_R_shapes": data_shapes,
        "buckets": buckets,
        "any_bucket_nan": any_nan,
        "left_nan": left_nan,
        "right_nan": right_nan,
        "L_S_all_buckets": _finite(sse_all.get("total_loss")),
        "L_S_right_only": _finite(sse_right.get("total_loss")),
        "nan_poison_if_left_nan": bool(
            left_nan and not right_nan
            and sse_all.get("total_loss") != sse_all.get("total_loss")
        ),
        "per_contrast_right": per_T,
        "S_prior_stim_80": curve_at(s_stim, i80),
        "S_prior_stim_end": curve_at(s_stim, -1),
        "S_prior_all_80": curve_at(s_all, i80),
        "S_prior_all_end": curve_at(s_all, -1),
        "include_stim_nSSE": None if s_prior is None else _finite(
            (s_prior.get("act_block_duringstim") or {}).get("stim")
        ),
        "note": (
            "L_S is retinal right-stim (R−L) vs avg_mean_R, not S prior-distance. "
            "Regular include_stim=False so S sidecar is not in the fit. "
            "compute_sse_stim_right NaNs the whole L_S if any (side, contrast) "
            "bucket is NaN, including left stim which is not scored."
        ),
    }


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
        f"{'arm':8} {'seed':>4} {'win':>4} {'rec':>7} {'traj':>7} {'prior':>7} "
        f"{'LS':>7} {'fair':>7} {'gi':>6} {'I80':>6} {'Iend':>6}",
        flush=True,
    )
    audited = False
    for arm, prefix in (
        ("meancell", "weights_run_fj_stageB_hold_s89_im150_meancell_regular_mask12-13"),
        ("im150old", "weights_run_fj_stageB_hold_s89_im150_regular_mask12-13"),
        ("base", "weights_run_fj_stageB_hold_s89_regular_mask12-13"),
    ):
        for seed in SEEDS:
            run = BASE / f"{prefix}_s{seed}"
            jp = latest_final(run)
            mp, meta, results, sbo, info = sim_one(jp, stim_bundle)
            if arm == "meancell" and not audited:
                audit = audit_s(results, sbo, avg_mean_R, mp, prior_regions)
                AUDIT_OUT.write_text(json.dumps(audit, indent=2, default=str))
                print(f"wrote S audit {AUDIT_OUT}", flush=True)
                audited = True
            windows = (150.0, None) if arm != "base" else (None, 150.0)
            for win in windows:
                mp_sc = dict(mp)
                mp_sc["prior_window_ms"] = win
                plot_dir = None
                if arm == "meancell" and win == 150.0:
                    plot_dir = run
                ev = {**info, **score(
                    mp_sc, results, sbo, mean_data, prior_regions, avg_mean_R,
                    plot_dir=plot_dir,
                )}
                rec = {"arm": arm, "seed": seed, **ev}
                rows.append(rec)
                wlab = "150" if win == 150.0 else "leg"
                print(
                    f"{arm:8} {seed:4d} {wlab:>4} {ev['recorded']:7.3f} "
                    f"{ev['eval_traj']:7.3f} {ev['eval_prior']:7.3f} "
                    f"{(ev['eval_LS'] or float('nan')):7.3f} "
                    f"{(ev['eval_fair'] or float('nan')):7.3f} "
                    f"{ev['g_i']:6.1f} "
                    f"{(ev.get('I_start_80') or float('nan')):6.3f} "
                    f"{(ev.get('I_start_end') or float('nan')):6.3f}",
                    flush=True,
                )

    OUT.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
