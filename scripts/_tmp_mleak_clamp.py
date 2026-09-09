"""Clamp g_m=d_m=0 and weaken W_mm on existing finals (options 2+3).

tau_* stay 20 ms. No I→M gate. Plots + JSON in
openalyx models/stageB_hold_s89_mleak_clamp/.
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
    load_plot_model,
    make_shared_stimuli,
)
from model_functions import (  # noqa: E402
    prior_distance_I_M_both_alignments,
    resolve_prior_distance_window,
    run_model,
    _resample_to_len,
)

BASE = Path.home() / (
    "Downloads/ONE/openalyx.internationalbrainlab.org/models"
)
OUT = BASE / "stageB_hold_s89_mleak_clamp"
W_MM_SWEEP = (None, 0.20, 0.15, 0.10)  # None = fitted W_mm
MODELS = (
    ("meancell_s12", BASE / "weights_run_fj_stageB_hold_s89_im150_meancell_regular_mask12-13_s12", 150.0),
    ("meancell_s34", BASE / "weights_run_fj_stageB_hold_s89_im150_meancell_regular_mask12-13_s34", 150.0),
    ("regular_s101", BASE / "weights_run_fj_stageB_hold_s89_regular_mask12-13_s101", None),
)


def latest_final(d: Path) -> Path:
    finals = sorted(d.glob("weights_final_*.json"))
    if not finals:
        raise FileNotFoundError(d)
    return finals[-1]


def at(y, t, tq):
    return float(np.interp(tq, t, np.asarray(y, float)))


def main():
    ensure_fit_data_links_paper()
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.load("data_act_block_duringstim.npy", allow_pickle=True).flat[0]
    I_d = np.asarray(data["r_int"], float)
    M_d = np.asarray(data["r_move"], float)
    t = np.linspace(0.0, 150.0, len(M_d))

    stim_ref = latest_final(BASE / "weights_run_fj_stageB_hold_s89_regular_mask12-13_s101")
    mp0, _ = load_plot_model(stim_ref)
    stim_bundle = make_shared_stimuli(mp0, bps=20, seed=12345)
    (
        stimuli, trial_strengths, trial_sides, block_sides,
        steps_before_obs, bps,
    ) = stim_bundle

    rows = []
    print(
        f"{'model':16} {'tag':12} {'Wmm':>5} {'gm':>6} {'M0':>6} {'M40':>6} "
        f"{'M70':>6} {'M80':>6} {'M150':>6}",
        flush=True,
    )
    for label, run, win in MODELS:
        jp = latest_final(run)
        mp_fit, _ = load_plot_model(jp)
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.plot(t, M_d, color="tomato", lw=2.0, label="data M")
        ax.plot(t, I_d, color="gold", lw=1.2, alpha=0.7, label="data I")
        for wmm in W_MM_SWEEP:
            mp = dict(mp_fit)
            mp["g_m"] = 1e-12
            mp["d_m"] = 1e-12
            if win is not None:
                mp["prior_window_ms"] = win
            w_use = float(mp_fit["W_mm"] if wmm is None else wmm)
            mp["W_mm"] = w_use
            results = run_model(
                "data",
                stimuli, trial_strengths, trial_sides, block_sides, bps,
                steps_before_obs=steps_before_obs, verbose=False,
                backend="numba", **mp,
            )
            T, _, _ = resolve_prior_distance_window(mp, T=72, plot_window=80)
            out = prior_distance_I_M_both_alignments(
                results, steps_before_obs, T=T, metric="l2",
                include_all_trials=True, lump_all=False,
            )
            M = _resample_to_len(out["M"]["start"], len(M_d))
            I = _resample_to_len(out["I"]["start"], len(I_d))
            tag = "gm0_fitW" if wmm is None else f"gm0_Wmm{wmm:.2f}"
            rec = {
                "model": label, "tag": tag,
                "W_mm": w_use, "W_mm_fitted": float(mp_fit["W_mm"]),
                "g_m_fitted": float(mp_fit["g_m"]),
                "d_m_fitted": float(mp_fit["d_m"]),
                "g_m": 1e-12, "d_m": 1e-12,
                "tau_m": float(mp.get("tau_m", 20.0)),
                "M0": at(M, t, 0), "M40": at(M, t, 40), "M60": at(M, t, 60),
                "M70": at(M, t, 70), "M80": at(M, t, 80), "M150": at(M, t, 150),
                "I80": at(I, t, 80),
            }
            rows.append(rec)
            print(
                f"{label:16} {tag:12} {w_use:5.3f} {mp_fit['g_m']:6.2g} "
                f"{rec['M0']:6.3f} {rec['M40']:6.3f} {rec['M70']:6.3f} "
                f"{rec['M80']:6.3f} {rec['M150']:6.3f}",
                flush=True,
            )
            ls = "--" if wmm is None else "-"
            ax.plot(t, M, ls, lw=1.6, label=f"M {tag} ({w_use:.2f})")
        ax.set_xlim(0, 150)
        ax.set_xlabel("time from stim (ms)")
        ax.set_ylabel("prior distance")
        ax.set_title(f"{label}  g_m=d_m=0, W_mm sweep (τ=20)")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / f"{label}_M_wmm_sweep.png", dpi=150,
                    bbox_inches="tight", facecolor="white")
        fig.savefig(OUT / f"{label}_M_wmm_sweep.svg", bbox_inches="tight",
                    transparent=True)
        plt.close(fig)

    (OUT / "summary.json").write_text(json.dumps({
        "note": "g_m=d_m=1e-12; tau_m left at 20; W_mm swept. Not a refit.",
        "data_M": {str(tq): at(M_d, t, tq) for tq in (0, 40, 60, 70, 80, 150)},
        "rows": rows,
    }, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
