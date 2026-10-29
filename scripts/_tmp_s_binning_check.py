"""S trajectory: does data binning / time axis explain the c=1 overshoot? (2026-10-29)

ARM=opt1_rawslope | opt2_shuffleslope; one process per arm (targets + cwd via
_tmp_integ_opt_eval_plots). Same shared stim as the report (bps=20, seed 12345,
stim from regular s101).

Data S bin i (build_fit_traj_targets_from_cache: 12.5 ms bins, 6 sub-offsets of
2 ms, brainbox bin_spikes2D with pre=0) covers [12.5*(i//6) + 2*(i%6), +12.5) ms
after stimOn. The fit puts it at 2i ms against an unsmoothed model step. Here the
model step k (covering [2k, 2k+2) ms) is box-averaged over each data bin's window
and zeroed on its first 7 bins like the data ("binned model", on the data's own
index), and both are drawn at true bin centres.
Reports the c=1 bump and L_S (current defaults: 0-100 ms, c>0) for raw vs binned
model. Figures go to the report folder (figs/s_binning_check/).
"""
from __future__ import annotations

import json

import _tmp_integ_opt_eval_plots as E  # sets FIT_TARGETS_DIR + cwd for ARM
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from model_functions import compute_sse_stim_right, mean_S_by_contrast, run_model, savefig_svg_png

PICKS = {"opt1_rawslope": ((45, 45), (7, 7)), "opt2_shuffleslope": ((12, 7), (303, 7))}[E.ARM]
MASK = "0-1-3-4-7-9-10-11-14-15-16-17-18-19-20"
REPORT = E.ALYX_MODELS / "stageB_hold_newS34_integ_ft_gm0_report"
OUT = REPORT / "figs" / "s_binning_check"
T_BIN, SUB, DT = 12.5, 2.0, 2.0


def bin_start(i):
    i = np.asarray(i)
    return T_BIN * (i // 6) + SUB * (i % 6)


def bin_model(m, n_bins):
    """Box-average model steps over each data bin's 12.5 ms window."""
    k_t = np.arange(m.size) * DT  # step start times
    out = np.full(n_bins, np.nan)
    for i in range(n_bins):
        s = bin_start(i)
        w = np.clip(np.minimum(k_t + DT, s + T_BIN) - np.maximum(k_t, s), 0, None)
        if w.sum() >= T_BIN - 1e-9:
            out[i] = np.sum(w * m) / w.sum()
    return out


def bump(t, y, lo=0.0, hi=80.0):
    sel = (t >= lo) & (t <= hi)
    j = int(np.nanargmax(np.where(sel, y, -np.inf)))
    return float(y[j]), float(t[j])


def main():
    _, R = E.load_avg_mean_r()
    mp0, _ = E.load_plot_model(E.newest_final(
        E.OA_MODELS / "weights_run_fj_stageB_hold_s89_regular_mask12-13_s101"))
    stim, ts, tsides, bsides, sbo, bps = E.make_shared_stimuli(mp0, bps=20, seed=12345)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for src, ft in PICKS:
        run = E.ALYX_MODELS / (
            f"weights_run_fj_stageB_hold_newS34_integ_{E.ARM}_ft{src}_gm0_full_mask{MASK}_s{ft}")
        mp, _ = E.load_plot_model(E.newest_final(run))
        res = run_model("data", stim, ts, tsides, bsides, bps, steps_before_obs=sbo,
                        verbose=False, backend="numba", **mp)
        S = mean_S_by_contrast(res, sbo)
        cs = sorted(c for (s, c) in S if s == 1)
        binned = {}
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        colors = cm.Blues(np.linspace(0.4, 1.0, len(cs)))[::-1]
        for c, col in zip(cs, colors):
            m = S[(1, c)][1] - S[(1, c)][0]
            d = np.asarray(R[c], float)
            n = d.size
            mb = bin_model(m, n)
            mb = mb - np.nanmean(mb[:7])  # same zeroing as the data (first 7 bins)
            ok = np.isfinite(mb)
            mb_f = np.where(ok, mb, 0.0)
            binned[(1, c)] = np.vstack([np.zeros(n), mb_f])
            tc = bin_start(np.arange(n)) + T_BIN / 2  # true bin centres
            if c > 0:
                ax.plot(tc, d, "-", color=col, lw=2)
                ax.plot(np.arange(m.size) * DT + DT / 2, m, ":", color=col, lw=1.2)
                ax.plot(tc[ok], mb[ok], "--", color=col, lw=2)
            if abs(c - 1.0) < 1e-9:
                tm = np.arange(m.size) * DT + DT / 2
                d_amp, d_t = bump(tc, d)
                d_amp_old, d_t_old = bump(np.arange(n) * DT, d)
                r_amp, r_t = bump(tm, m)
                b_amp, b_t = bump(tc[ok], mb[ok])
        ls_raw = compute_sse_stim_right(S, R, baseline_R=0)
        ls_bin = compute_sse_stim_right(binned, R, baseline_R=0)
        row = {"fit": f"{E.ARM} s{src}->s{ft}",
               "c1_data_peak": d_amp, "c1_data_t_true": d_t, "c1_data_t_as_fit": d_t_old,
               "c1_model_raw_peak": r_amp, "c1_model_raw_t": r_t,
               "c1_model_binned_peak": b_amp, "c1_model_binned_t": b_t,
               "L_S_raw": ls_raw["total_loss"], "R2_raw": ls_raw["total_gof_r2"],
               "L_S_binned": ls_bin["total_loss"], "R2_binned": ls_bin["total_gof_r2"],
               "per_c_R2_raw": {str(c): v["gof_r2"] for c, v in ls_raw["per_contrast"].items()},
               "per_c_R2_binned": {str(c): v["gof_r2"] for c, v in ls_bin["per_contrast"].items()}}
        rows.append(row)
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.set_xlim(0, 130)
        ax.set_xlabel("time after stim onset (ms, true bin centre)")
        ax.set_title(f"{E.ARM[:4]} s{src}->s{ft}: data (solid), model raw (dotted), "
                     "model binned (dashed)", fontsize=7)
        ax.text(0.98, 0.98, f"R$^2$ raw {row['R2_raw']:.2f} / binned {row['R2_binned']:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        savefig_svg_png(fig, str(OUT / f"S_binning_{E.ARM[:4]}_s{src}_s{ft}.svg"),
                        dpi=150, transparent=False)
        plt.close(fig)
        print(json.dumps(row), flush=True)
    (OUT / f"s_binning_check_{E.ARM}.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
