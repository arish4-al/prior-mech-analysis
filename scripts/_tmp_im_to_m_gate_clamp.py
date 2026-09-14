"""Clamp I→M gate on regular s303 (best 150 ms eval tot). Not a refit.

Zero W_mi and g_m until T ms after stimOn, optionally also during
prestim. Score at prior_window_ms=150. Plots in the s303 run dir.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from model_functions import (  # noqa: E402
    int_regs,
    move_regs,
    prior_distance_I_M_both_alignments,
    resolve_prior_distance_window,
    run_model,
    _resample_to_len,
)
from _tmp_im150_meancell_eval import latest_final, score  # noqa: E402
from _tmp_perf_rt_model_vs_data import (  # noqa: E402
    BEHAVIOR_ACT,
    BPS,
    N_SESSIONS,
    STIM_SEED0,
    gof,
    results_with_subjective_prior,
    run_many_sessions,
    save_current,
)
from model_functions import loss_perf_with_data  # noqa: E402

BASE = Path.home() / (
    "Downloads/ONE/openalyx.internationalbrainlab.org/models"
)
RUN = BASE / "weights_run_fj_stageB_hold_s89_regular_mask12-13_s303"
OUT = RUN / "im_to_m_gate_clamp"
WIN = 150.0
TQ = (0, 40, 50, 60, 70, 80, 110, 150)
CONDS = (
    ("baseline", 0.0, False),
    ("prestim", 0.0, True),
    ("post60", 60.0, False),
    ("post80", 80.0, False),
    ("full60", 60.0, True),
    ("full80", 80.0, True),
)


def at(y, t, tq):
    return float(np.interp(tq, t, np.asarray(y, float)))


def apply_gate(mp_fit, until_ms, off_prestim):
    mp = dict(mp_fit)
    mp["prior_window_ms"] = WIN
    mp["w_mi_off_until_ms"] = float(until_ms)
    mp["w_mi_off_prestim"] = bool(off_prestim)
    return mp


def main():
    ensure_fit_data_links_paper()
    OUT.mkdir(parents=True, exist_ok=True)
    _, mean_data = load_mean_data_results()
    _, avg_mean_R = load_avg_mean_r()
    prior_regions = {
        "int_regs_choice": int_regs, "int_regs_stim": int_regs,
        "move_regs_choice": move_regs, "move_regs_stim": move_regs,
        "stim_regs": ["VISpm", "FRP", "VISal"],
    }
    data = np.load("data_act_block_duringstim.npy", allow_pickle=True).flat[0]
    I_d = np.asarray(data["r_int"], float)
    M_d = np.asarray(data["r_move"], float)
    t = np.linspace(0.0, 150.0, len(M_d))

    jp = latest_final(RUN)
    mp_fit, _ = load_plot_model(jp)
    stim_ref = latest_final(BASE / "weights_run_fj_stageB_hold_s89_regular_mask12-13_s101")
    mp0, _ = load_plot_model(stim_ref)
    stim_bundle = make_shared_stimuli(mp0, bps=20, seed=12345)
    (
        stimuli, trial_strengths, trial_sides, block_sides,
        steps_before_obs, bps,
    ) = stim_bundle

    print(
        f"clamp regular s303 @ {WIN:.0f} ms  W_mi={mp_fit['W_mi']:.3f} "
        f"g_m={mp_fit['g_m']:.3g}  d_m={mp_fit['d_m']:.3g}",
        flush=True,
    )
    print(
        f"{'tag':8} {'until':>5} {'pre':>3} {'traj':>7} {'prior':>7} "
        f"{'fair':>7} {'M40':>6} {'M70':>6} {'M80':>6} {'M150':>6} "
        f"{'d40-70':>7}",
        flush=True,
    )

    rows = []
    curves = {"t": t.tolist(), "data_I": I_d.tolist(), "data_M": M_d.tolist(),
              "models": {}}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharex=True)
    for ax, yd, lab in zip(axes, (I_d, M_d), ("I", "M")):
        ax.plot(t, yd, color="tomato", lw=2.0, label="data")
        ax.set_title(f"{lab} prior distance (150 ms)")
        ax.set_xlim(0, 150)
        ax.set_xlabel("time from stim (ms)")
    axes[0].set_ylabel("prior distance")

    colors = {
        "baseline": "0.35", "prestim": "C1", "post60": "C0",
        "post80": "C2", "full60": "C3", "full80": "C4",
    }
    for tag, until_ms, off_pre in CONDS:
        mp = apply_gate(mp_fit, until_ms, off_pre)
        results = run_model(
            "data",
            stimuli, trial_strengths, trial_sides, block_sides, bps,
            steps_before_obs=steps_before_obs, verbose=False,
            backend="numba", **mp,
        )
        ev = score(mp, results, steps_before_obs, mean_data, prior_regions,
                   avg_mean_R, plot_dir=None)
        T, _, _ = resolve_prior_distance_window(mp, T=72, plot_window=80)
        out = prior_distance_I_M_both_alignments(
            results, steps_before_obs, T=T, metric="l2",
            include_all_trials=True, lump_all=False,
        )
        I = _resample_to_len(out["I"]["start"], len(I_d))
        M = _resample_to_len(out["M"]["start"], len(M_d))
        shape = {f"I{tq}": at(I, t, tq) for tq in TQ}
        shape.update({f"M{tq}": at(M, t, tq) for tq in TQ})
        rec = {
            "tag": tag, "until_ms": until_ms, "off_prestim": off_pre,
            "W_mi": float(mp_fit["W_mi"]), "g_m": float(mp_fit["g_m"]),
            **ev, **shape,
        }
        rows.append(rec)
        curves["models"][tag] = {"I": I.tolist(), "M": M.tolist()}
        print(
            f"{tag:8} {until_ms:5.0f} {int(off_pre):3d} "
            f"{ev['eval_traj']:7.3f} {ev['eval_prior']:7.3f} "
            f"{(ev['eval_fair'] or float('nan')):7.3f} "
            f"{shape['M40']:6.3f} {shape['M70']:6.3f} "
            f"{shape['M80']:6.3f} {shape['M150']:6.3f} "
            f"{shape['M70'] - shape['M40']:+7.3f}",
            flush=True,
        )
        ls = "--" if tag == "baseline" else "-"
        axes[0].plot(t, I, ls, color=colors[tag], lw=1.4, label=tag)
        axes[1].plot(t, M, ls, color=colors[tag], lw=1.4, label=tag)

    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "IM_gate_overlay.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    fig.savefig(OUT / "IM_gate_overlay.svg", bbox_inches="tight",
                transparent=True)
    plt.close(fig)

    (OUT / "eval.json").write_text(json.dumps(rows, indent=2, default=str))
    (OUT / "curves.json").write_text(json.dumps(curves, indent=2))
    print(f"wrote {OUT / 'eval.json'}", flush=True)

    if not BEHAVIOR_ACT.is_file():
        print("no behavior_actprior.npy; skip RT", flush=True)
        return
    behavior = np.load(BEHAVIOR_ACT, allow_pickle=True).item()
    print("\nact-prior RT (10×20, seed 12345)", flush=True)
    print(
        f"{'tag':8} {'perf':>6} {'RTcomb':>7} {'RTspl':>6} "
        f"{'con':>6} {'inc':>6}",
        flush=True,
    )
    rt_rows = []
    for tag, until_ms, off_pre in CONDS:
        mp = apply_gate(mp_fit, until_ms, off_pre)
        out_dir = OUT / f"actprior_{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cache = out_dir / "combined_results.npy"
        if cache.is_file():
            results = np.load(cache, allow_pickle=True).item()
            n_trials = int(np.sum(np.asarray(results["choices"]) != 0))
            print(f"loaded {cache}", flush=True)
        else:
            print(f"sim {tag} {N_SESSIONS}×{BPS} seed0={STIM_SEED0}", flush=True)
            results, n_trials = run_many_sessions(mp, N_SESSIONS, BPS, STIM_SEED0)
            slim = {
                k: results[k]
                for k in (
                    "trial_strengths", "trial_sides", "block_sides",
                    "correct_action_taken", "reaction_time", "choices",
                    "sub_prior",
                )
            }
            np.save(cache, slim, allow_pickle=True)
        results = results_with_subjective_prior(results)
        dt_s = float(mp["dt"]) / 1000.0
        sse_c = loss_perf_with_data(
            results, behavior, mp, metric="correct", dt=1.0,
            do_plot=True, save_dir=None, log_xaxis=True,
        )
        save_current(out_dir / "performance_model_vs_data",
                     f"performance ({tag})")
        sse_rt = loss_perf_with_data(
            results, behavior, mp, metric="rt", dt=dt_s,
            do_plot=True, save_dir=None, log_xaxis=True, rt_mode="combined_all",
        )
        save_current(out_dir / "rt_model_vs_data", f"RT ({tag})")
        sse_sp = loss_perf_with_data(
            results, behavior, mp, metric="rt", dt=dt_s,
            do_plot=True, save_dir=None, log_xaxis=True, rt_mode="split_all",
        )
        save_current(out_dir / "rt_split_model_vs_data", f"RT split ({tag})")
        rec = {
            "tag": tag, "until_ms": until_ms, "off_prestim": off_pre,
            "n_trials": n_trials,
            "perf_r2": gof(sse_c, "total"),
            "rt_r2": gof(sse_rt, "total"),
            "rt_split_r2": gof(sse_sp, "total"),
            "rt_split_r2_con": gof(sse_sp, "congruent"),
            "rt_split_r2_inc": gof(sse_sp, "incongruent"),
        }
        rt_rows.append(rec)
        print(
            f"{tag:8} {rec['perf_r2']:.3f}  {rec['rt_r2']:7.3f}  "
            f"{rec['rt_split_r2']:.3f}  "
            f"({rec['rt_split_r2_con']:.3f} / {rec['rt_split_r2_inc']:.3f})",
            flush=True,
        )
    (OUT / "actprior_rt.json").write_text(json.dumps(rt_rows, indent=2))
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
