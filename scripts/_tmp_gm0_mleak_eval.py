"""Shared-stim fair eval for Stage B gm0 / mleak (options 2+3).

bps=20, stim seed 12345, stim from regular s101. Production prior window
(unset). Plots live in each run dir + campaign overlay in
models/stageB_hold_s89_gm0_mleak_eval/.
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
import model_functions as mf  # noqa: E402
from model_functions import (  # noqa: E402
    int_regs,
    move_regs,
    prior_distance_I_M_both_alignments,
    resolve_prior_distance_window,
    _resample_to_len,
)
from _tmp_im150_meancell_eval import latest_final, score, sim_one  # noqa: E402

BASE = Path.home() / (
    "Downloads/ONE/openalyx.internationalbrainlab.org/models"
)
OUT_DIR = BASE / "stageB_hold_s89_gm0_mleak_eval"
OUT = OUT_DIR / "eval.json"
SEEDS = (7, 12, 34, 45, 89, 101, 303, 333)
ARMS = (
    ("gm0", "weights_run_fj_stageB_hold_s89_gm0_regular_mask7-9-12-13"),
    ("mleak", "weights_run_fj_stageB_hold_s89_mleak_regular_mask7-9-12-13"),
    ("base", "weights_run_fj_stageB_hold_s89_regular_mask12-13"),
)
TQ = (0, 40, 60, 70, 80, 110, 150)


def at(y, t, tq):
    return float(np.interp(tq, t, np.asarray(y, float)))


def main():
    ensure_fit_data_links_paper()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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

    stim_ref = latest_final(BASE / "weights_run_fj_stageB_hold_s89_regular_mask12-13_s101")
    mp0, _ = load_plot_model(stim_ref)
    stim_bundle = make_shared_stimuli(mp0, bps=20, seed=12345)
    print(f"HAVE_NUMBA={mf._HAVE_NUMBA}  stim from {stim_ref.parent.name}", flush=True)

    rows = []
    curves = {"data_I": I_d.tolist(), "data_M": M_d.tolist(), "t": t.tolist(),
              "models": {}}
    print(
        f"{'arm':6} {'seed':>4} {'rec':>7} {'traj':>7} {'prior':>7} "
        f"{'LS':>7} {'fair':>7} {'Wmm':>6} {'Wmi':>6} {'gi':>7} "
        f"{'M40':>6} {'M70':>6} {'M80':>6} {'M150':>6}",
        flush=True,
    )
    for arm, prefix in ARMS:
        for seed in SEEDS:
            run = BASE / f"{prefix}_s{seed}"
            if not run.is_dir():
                print(f"skip missing {run.name}", flush=True)
                continue
            jp = latest_final(run)
            mp, meta, results, sbo, info = sim_one(jp, stim_bundle)
            W = meta.get("W") or {}
            info["W_mi"] = float(W.get("W_mi", mp.get("W_mi", np.nan)))
            plot_dir = run if arm != "base" else None
            ev = {**info, **score(
                mp, results, sbo, mean_data, prior_regions, avg_mean_R,
                plot_dir=plot_dir,
            )}
            T, _, _ = resolve_prior_distance_window(mp, T=72, plot_window=80)
            out = prior_distance_I_M_both_alignments(
                results, sbo, T=T, metric="l2",
                include_all_trials=True, lump_all=False,
            )
            I = _resample_to_len(out["I"]["start"], len(I_d))
            M = _resample_to_len(out["M"]["start"], len(M_d))
            shape = {f"I{tq}": at(I, t, tq) for tq in TQ}
            shape.update({f"M{tq}": at(M, t, tq) for tq in TQ})
            rec = {"arm": arm, "seed": seed, **ev, **shape}
            rows.append(rec)
            key = f"{arm}_s{seed}"
            curves["models"][key] = {"I": I.tolist(), "M": M.tolist(),
                                     "W_mm": rec["W_mm"], "W_mi": rec["W_mi"]}
            print(
                f"{arm:6} {seed:4d} {ev['recorded']:7.3f} "
                f"{ev['eval_traj']:7.3f} {ev['eval_prior']:7.3f} "
                f"{(ev['eval_LS'] or float('nan')):7.3f} "
                f"{(ev['eval_fair'] or float('nan')):7.3f} "
                f"{ev['W_mm']:6.3f} {info['W_mi']:6.3f} {ev['g_i']:7.2g} "
                f"{shape['M40']:6.3f} {shape['M70']:6.3f} "
                f"{shape['M80']:6.3f} {shape['M150']:6.3f}",
                flush=True,
            )

    OUT.write_text(json.dumps(rows, indent=2, default=str))
    (OUT_DIR / "curves.json").write_text(json.dumps(curves, indent=2))

    # Overlay: data + best-ish seeds (s12, s34, s101) per arm
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharex=True)
    for ax, pop, yd in zip(axes, ("I", "M"), (I_d, M_d)):
        ax.plot(t, yd, color="tomato", lw=2.0, label="data")
        for arm, col, ls in (("base", "0.45", "--"), ("gm0", "C0", "-"),
                             ("mleak", "C2", "-")):
            for seed, alpha in ((12, 0.45), (34, 0.7), (101, 1.0)):
                key = f"{arm}_s{seed}"
                if key not in curves["models"]:
                    continue
                y = curves["models"][key][pop]
                lab = f"{arm} s{seed}" if seed == 101 else None
                ax.plot(t, y, ls, color=col, lw=1.3, alpha=alpha, label=lab)
        ax.set_xlim(0, 150)
        ax.set_xlabel("time from stim (ms)")
        ax.set_title(f"{pop} prior distance")
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("prior distance")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "IM_overlay_s12_s34_s101.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / "IM_overlay_s12_s34_s101.svg", bbox_inches="tight",
                transparent=True)
    plt.close(fig)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
