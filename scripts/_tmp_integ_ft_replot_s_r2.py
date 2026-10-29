"""Re-plot the four report fine-tunes (S R², bin-by-bin windows, RT at 12.5/6 ms; 2026-10-29).

ARM=opt1_rawslope | opt2_shuffleslope; one process per arm (targets + cwd via
_tmp_integ_opt_eval_plots). Same shared stim as _tmp_integ_ft_eval_plots
(bps=20, seed 12345, stim from regular s101). Redraws S_fit (overall R²) and
prior_effects, I/M traj and act-prior behaviour (RT uses model_step_s) in each
picked run dir and updates the `traj` and `rt` entries of that arm's
plot_summary JSON.
"""
from __future__ import annotations

import json

import numpy as np

import _tmp_integ_opt_eval_plots as E  # sets FIT_TARGETS_DIR + cwd for ARM
import matplotlib.pyplot as plt

PICKS = {"opt1_rawslope": ((45, 45), (7, 7)), "opt2_shuffleslope": ((12, 7), (303, 7))}[E.ARM]
FT_MASK = "0-1-3-4-7-9-10-11-14-15-16-17-18-19-20"
SUMMARY = E.ALYX_MODELS / f"stageB_hold_newS34_integ_{E.ARM}_ft_gm0_plot_summary.json"


def main():
    E.ensure_fit_data_links_paper()
    _, mean_data = E.load_mean_data_results()
    _, avg_mean_R = E.load_avg_mean_r()
    _, payload = E.load_s_unsplit80()
    prior_regions = {"int_regs_choice": E.int_regs, "int_regs_stim": E.int_regs,
                     "move_regs_choice": E.move_regs, "move_regs_stim": E.move_regs,
                     "stim_regs": list(payload["regs_stim"])}
    mp0, _ = E.load_plot_model(E.newest_final(
        E.OA_MODELS / "weights_run_fj_stageB_hold_s89_regular_mask12-13_s101"))
    bundle = E.make_shared_stimuli(mp0, bps=20, seed=12345)
    from _tmp_perf_rt_model_vs_data import BEHAVIOR_ACT, plot_one_json
    behavior = np.load(BEHAVIOR_ACT, allow_pickle=True).item()

    summary = json.loads(SUMMARY.read_text())
    for src, ft in PICKS:
        run = E.ALYX_MODELS / (
            f"weights_run_fj_stageB_hold_newS34_integ_{E.ARM}_ft{src}_gm0_full_mask{FT_MASK}_s{ft}")
        t = E.plot_one(E.newest_final(run), bundle, mean_data, prior_regions, run,
                       avg_mean_R=avg_mean_R, include_stim=True, prior_plot_window=80)
        plt.close("all")
        E.alias_svgs(run)
        r = plot_one_json(f"{E.ARM}_ft{src}_s{ft}", E.newest_final(run), run, behavior)
        plt.close("all")
        entry = next(p for p in summary if p["src"] == src and p["ft_seed"] == ft)
        old_rt = entry["rt"]
        entry["traj"], entry["rt"] = t, r
        print(f"src s{src} ft s{ft}: S traj R2 {t['gof_S']:.3f}  "
              f"L_S {t['L_S']:.3f}  prior_S {t['prior_S']:.3f} | perf R2 {r['perf_r2']:.3f} "
              f"RT R2 {old_rt['rt_r2']:.3f} -> {r['rt_r2']:.3f}  split {r['rt_split_r2']:.3f} "
              f"({r['rt_split_r2_con']:.3f} / {r['rt_split_r2_inc']:.3f})", flush=True)
    SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
