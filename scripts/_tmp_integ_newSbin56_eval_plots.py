"""Stage B on Stage A newSbin s56: multi-draw eval + overlays (2026-10-30).

ARM=opt1_rawslope | opt2_shuffleslope; one process per arm (sets that arm's
targets and cwd via _tmp_integ_opt_eval_plots). Scores, at move-period weight 1
with the per-term breakdown of the gm0 report (`terms`), on DRAWS shared-stim
draws (bps=20, stim from regular s101):
  - the 8 stageB_hold_newSbin56_full_mpre3_meancell_integ_<ARM> seeds,
  - the 8 newS34 full seeds of the same arm (reference, rescored under the
    current S loss),
  - the two gm0 fine-tunes of this arm that are in the report.
Overlays (prior_effects on the scored 80 ms window) + act-prior behaviour go in
each newSbin56 run dir (draw 12345). PLOTS=0 skips them.

OWN_STIM=1 (2026-10-30): each fit's stimuli are built under its own α_w/β_w
(same draw seeds), as in fitting (stage2_restim), instead of under the regular
s101 reference. Writes *_eval_ownstim.json and redraws the S / I/M / prior
overlays in every scored run dir (behaviour already simulates under the fit's
own params, so it is not redrawn).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import _tmp_integ_ft_eval_plots as F  # sets FIT_TARGETS_DIR + cwd for ARM
import matplotlib.pyplot as plt
import numpy as np
from model_functions import compute_sse_stim_right, mean_S_by_contrast, run_model

E = F.E
DRAWS = tuple(int(x) for x in os.environ.get("DRAWS", "12345 1 2 3").split())
NEW = E.ALYX_MODELS / (
    f"weights_run_fj_stageB_hold_newSbin56_full_mpre3_meancell_integ_{E.ARM}_full_masknone")
FT_PICKS = {"opt1_rawslope": ((45, 45), (7, 7)), "opt2_shuffleslope": ((12, 7), (303, 7))}[E.ARM]
OWN_STIM = os.environ.get("OWN_STIM", "0") == "1"
_SUF = "_ownstim" if OWN_STIM else ""
OUT = E.ALYX_MODELS / f"stageB_hold_newSbin56_integ_{E.ARM}_eval{_SUF}.json"
TERMS = ("fair", "traj", "traj_I", "traj_M", "traj_P", "prior", "prior_S", "prior_I_stim",
         "prior_I_choice", "prior_M_stim", "prior_M_choice", "L_S", "S_r2")


def runs():
    for s in E.SEEDS:
        yield "newSbin56", s, Path(f"{NEW}_s{s}")
    for s in E.SEEDS:
        yield "newS34", s, Path(f"{E.RUNS[E.ARM]}_s{s}")
    for src, ft in FT_PICKS:
        yield "newS34_ft_gm0", f"{src}->{ft}", E.ALYX_MODELS / (
            f"weights_run_fj_stageB_hold_newS34_integ_{E.ARM}_ft{src}_gm0_full_mask{F.FT_MASK}_s{ft}")


def main():
    E.ensure_fit_data_links_paper()
    _, mean_data = E.load_mean_data_results()
    _, avg_mean_R = E.load_avg_mean_r()
    stim_curve, payload = E.load_s_unsplit80()
    prior_regions = {"int_regs_choice": E.int_regs, "int_regs_stim": E.int_regs,
                     "move_regs_choice": E.move_regs, "move_regs_stim": E.move_regs,
                     "stim_regs": list(payload["regs_stim"])}
    mp0, _ = E.load_plot_model(E.newest_final(
        E.OA_MODELS / "weights_run_fj_stageB_hold_s89_regular_mask12-13_s101"))
    ref_bundles = {d: E.make_shared_stimuli(mp0, bps=20, seed=d) for d in DRAWS}

    def bundles_for(mp):
        if not OWN_STIM:
            return ref_bundles
        return {d: E.make_shared_stimuli(mp, bps=20, seed=d) for d in DRAWS}

    rows = []
    for group, s, run in runs():
        jp = E.newest_final(run)
        mp, meta = E.load_plot_model(jp)
        per = []
        for d, (stim, ts, tsides, bsides, sbo, bps) in bundles_for(mp).items():
            res = run_model("data", stim, ts, tsides, bsides, bps, steps_before_obs=sbo,
                            verbose=False, backend="numba", **mp)
            t = F.terms(mp, res, sbo, mean_data, prior_regions, avg_mean_R, stim_curve)
            t["S_r2"] = float(compute_sse_stim_right(mean_S_by_contrast(res, sbo), avg_mean_R,
                                                     baseline_R=0)["total_gof_r2"])
            t["draw"] = d
            if d == DRAWS[0]:
                t.update(E.stim_IM_at(res, sbo, mp))
            per.append(t)
        rec = {"group": group, "seed": s, "run": run.name, "json": jp.name,
               "recorded": float(meta.get("loss", np.nan)),
               **{k: float(mp[k]) for k in ("g_i", "d_i", "g_m", "d_m", "g_s", "d_s",
                                             "W_mm", "W_mi")},
               "per_draw": per}
        for k in TERMS:
            v = np.array([p[k] for p in per], float)
            rec[k], rec[k + "_sd"] = float(v.mean()), float(v.std())
        rows.append(rec)
        print(f"{group:14} s{s!s:7} fair {rec['fair']:.3f}±{rec['fair_sd']:.3f} "
              f"traj {rec['traj']:.3f} (I {rec['traj_I']:.3f} M {rec['traj_M']:.3f}) "
              f"prior {rec['prior']:.3f} (S {rec['prior_S']:.3f}) LS {rec['L_S']:.3f}"
              f"±{rec['L_S_sd']:.3f} R2 {rec['S_r2']:.3f} gi {rec['g_i']:.3g} di {rec['d_i']:.3g} "
              f"gs {rec['g_s']:.3g} ds {rec['d_s']:.3g} gm {rec['g_m']:.3g} dm {rec['d_m']:.3g}",
              flush=True)
        OUT.write_text(json.dumps({"targets": str(E.FIT_DATA_DIR), "draws": DRAWS,
                                   "own_stim": OWN_STIM, "eval": rows}, indent=2, default=str))
    print(f"wrote {OUT}", flush=True)
    if os.environ.get("PLOTS", "1") == "0":
        return

    if OWN_STIM:
        summary = []
        for group, s, run in runs():
            jp = E.newest_final(run)
            mp, _ = E.load_plot_model(jp)
            t = E.plot_one(jp, E.make_shared_stimuli(mp, bps=20, seed=DRAWS[0]), mean_data,
                           prior_regions, run, avg_mean_R=avg_mean_R, include_stim=True,
                           prior_plot_window=80)
            plt.close("all")
            E.alias_svgs(run)
            summary.append({"group": group, "seed": s, "traj": t})
            print(f"[plots] {group} s{s} S R2 {t.get('gof_S')}", flush=True)
        (E.ALYX_MODELS / f"stageB_hold_newSbin56_integ_{E.ARM}_plot_summary{_SUF}.json").write_text(
            json.dumps(summary, indent=2, default=str))
        return

    from _tmp_perf_rt_model_vs_data import BEHAVIOR_ACT, plot_one_json

    behavior = np.load(BEHAVIOR_ACT, allow_pickle=True).item()
    summary = []
    for group, s, run in runs():
        if group != "newSbin56":
            continue
        jp = E.newest_final(run)
        t = E.plot_one(jp, ref_bundles[DRAWS[0]], mean_data, prior_regions, run,
                       avg_mean_R=avg_mean_R, include_stim=True, prior_plot_window=80)
        plt.close("all")
        E.alias_svgs(run)
        r = plot_one_json(f"{E.ARM}_newSbin56_s{s}", jp, run, behavior)
        summary.append({"seed": s, "traj": t, "rt": r})
        print(f"[plots] s{s} S R2 {t.get('gof_S')} perf {r['perf_r2']:.3f} RT {r['rt_r2']:.3f} "
              f"split {r['rt_split_r2']:.3f}", flush=True)
    (E.ALYX_MODELS / f"stageB_hold_newSbin56_integ_{E.ARM}_plot_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
