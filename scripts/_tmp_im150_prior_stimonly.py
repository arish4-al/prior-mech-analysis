"""Overlay stim-only vs stim×choice I/M prior-distance for im150 fits.

Does not change the fitted loss (still stim×choice). Data overlay is the
cached 4-split ``act_block_duringstim`` / ``duringchoice`` curves.
"""
from __future__ import annotations

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
    _resample_to_len,
    prior_distance_I_M_both_alignments,
    resolve_prior_distance_window,
    run_model,
)

BASE = Path.home() / (
    "Downloads/ONE/openalyx.internationalbrainlab.org/models"
)
SEEDS = (7, 12, 34, 45, 89, 101, 303, 333)
PREFIX = "weights_run_fj_stageB_hold_s89_im150_regular_mask12-13"


def latest_final(d: Path) -> Path:
    finals = sorted(d.glob("weights_final_*.json"))
    if not finals:
        raise FileNotFoundError(d)
    return finals[-1]


def load_data_curves():
    stim = np.load("data_act_block_duringstim.npy", allow_pickle=True).flat[0]
    ch = np.load("data_act_block_duringchoice.npy", allow_pickle=True).flat[0]
    return (
        np.asarray(stim["r_int"], float),
        np.asarray(stim["r_move"], float),
        np.asarray(ch["r_int"], float),
        np.asarray(ch["r_move"], float),
    )


def model_curve(dists, align, n_data, resample):
    out = {}
    for vn in ("I", "M"):
        y = np.asarray(dists[vn][align], float)
        if resample:
            y = _resample_to_len(y, n_data)
        out[vn] = y
    return out


def plot_seed(seed, results, sbo, mp, data, out_dir: Path):
    T, plot_win, resample = resolve_prior_distance_window(mp, T=72, plot_window=80)
    d_i, d_m, d_i_c, d_m_c = data
    n_stim, n_ch = len(d_i), len(d_i_c)
    kw = dict(
        results=results, steps_before_obs=sbo, T=T, metric="l2",
        include_all_trials=True,
    )
    d_sc = prior_distance_I_M_both_alignments(**kw, stratum="stim_choice")
    d_st = prior_distance_I_M_both_alignments(**kw, stratum="stim")
    m_sc_st = model_curve(d_sc, "start", n_stim, resample)
    m_st_st = model_curve(d_st, "start", n_stim, resample)
    m_sc_ac = model_curve(d_sc, "action", n_ch, resample)
    m_st_ac = model_curve(d_st, "action", n_ch, resample)

    t_stim = np.linspace(0.0, plot_win, n_stim)
    t_ch = np.linspace(-plot_win, 0.0, n_ch)

    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(7.2, 2.6), dpi=150)
    # stim-aligned
    axs[0].plot(t_stim, d_i, color="gold", lw=2, label="I data (stim×choice)")
    axs[0].plot(t_stim, d_m, color="tomato", lw=2, label="M data (stim×choice)")
    axs[0].plot(t_stim, m_sc_st["I"], "--", color="gold", lw=1.8, alpha=0.95,
                label="I model stim×choice")
    axs[0].plot(t_stim, m_sc_st["M"], "--", color="tomato", lw=1.8, alpha=0.95,
                label="M model stim×choice")
    axs[0].plot(t_stim, m_st_st["I"], ":", color="gold", lw=2.2,
                label="I model stim only")
    axs[0].plot(t_stim, m_st_st["M"], ":", color="tomato", lw=2.2,
                label="M model stim only")
    axs[0].set_xlim(0.0, plot_win)
    axs[0].set_xlabel("ms after stimOn")
    # movement-aligned
    axs[1].plot(t_ch, d_i_c, color="gold", lw=2)
    axs[1].plot(t_ch, d_m_c, color="tomato", lw=2)
    axs[1].plot(t_ch, m_sc_ac["I"], "--", color="gold", lw=1.8, alpha=0.95)
    axs[1].plot(t_ch, m_sc_ac["M"], "--", color="tomato", lw=1.8, alpha=0.95)
    axs[1].plot(t_ch, m_st_ac["I"], ":", color="gold", lw=2.2)
    axs[1].plot(t_ch, m_st_ac["M"], ":", color="tomato", lw=2.2)
    axs[1].set_xlim(-plot_win, 0.0)
    axs[1].set_xlabel("ms before movement")
    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10)
    axs[0].set_ylabel("prior distance")
    axs[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.suptitle(f"im150 s{seed}  dashed=stim×choice  dotted=stim only",
                 fontsize=10)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "prior_effects_stimonly.png"
    svg = out_dir / "prior_effects_stimonly.svg"
    fig.savefig(png, dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", transparent=True)
    plt.close(fig)

    def at(y, i):
        return float(y[i]) if y.size else float("nan")

    i80 = min(n_stim - 1, int(round(80.0 / plot_win * (n_stim - 1))))
    return {
        "seed": seed,
        "I_sc_0": at(m_sc_st["I"], 0),
        "I_sc_80": at(m_sc_st["I"], i80),
        "I_sc_end": at(m_sc_st["I"], -1),
        "I_st_0": at(m_st_st["I"], 0),
        "I_st_80": at(m_st_st["I"], i80),
        "I_st_end": at(m_st_st["I"], -1),
        "M_sc_0": at(m_sc_st["M"], 0),
        "M_sc_80": at(m_sc_st["M"], i80),
        "M_sc_end": at(m_sc_st["M"], -1),
        "M_st_0": at(m_st_st["M"], 0),
        "M_st_80": at(m_st_st["M"], i80),
        "M_st_end": at(m_st_st["M"], -1),
        "png": str(png),
    }


def main():
    ensure_fit_data_links_paper()
    data = load_data_curves()
    stim_ref = latest_final(
        BASE / "weights_run_fj_stageB_hold_s89_regular_mask12-13_s101"
    )
    mp0, _ = load_plot_model(stim_ref)
    stim_bundle = make_shared_stimuli(mp0, bps=20, seed=12345)
    stimuli, tstr, tsides, bsides, sbo, bps = stim_bundle
    print(
        f"{'seed':>4} {'I_sc 0/80/end':>18} {'I_st 0/80/end':>18} "
        f"{'M_sc 0/80/end':>18} {'M_st 0/80/end':>18}",
        flush=True,
    )
    for seed in SEEDS:
        run = BASE / f"{PREFIX}_s{seed}"
        mp, _ = load_plot_model(latest_final(run))
        results = run_model(
            "data", stimuli, tstr, tsides, bsides, bps,
            steps_before_obs=sbo, verbose=False, backend="numba", **mp,
        )
        rec = plot_seed(seed, results, sbo, mp, data, run)
        print(
            f"{seed:4d} "
            f"{rec['I_sc_0']:.3f}/{rec['I_sc_80']:.3f}/{rec['I_sc_end']:.3f}   "
            f"{rec['I_st_0']:.3f}/{rec['I_st_80']:.3f}/{rec['I_st_end']:.3f}   "
            f"{rec['M_sc_0']:.3f}/{rec['M_sc_80']:.3f}/{rec['M_sc_end']:.3f}   "
            f"{rec['M_st_0']:.3f}/{rec['M_st_80']:.3f}/{rec['M_st_end']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
