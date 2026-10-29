"""Shared-stim eval + overlays for the integrator-rule Stage B arms (2026-10-28).

One process per target set: ARM=opt1_rawslope | opt2_shuffleslope.
FIT_TARGETS_DIR=fit_targets_integ/<ARM> (set here before the target loaders
import), cwd = fit_workdirs/eval_<ARM> so loss_prior_effect reads that arm's
data_*.npy. bps=20, stim seed 12345, stim from regular s101 (as mpre3 meancell).

Scores the 8 arm seeds and, as a reference, the production
stageB_hold_s89_full_mpre3_meancell seeds on the same targets (eval only).
Overlays + act-prior behavior go in each arm run dir (PLOTS=0 skips them).
prior_effects is drawn on the scored 80 ms windows only.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ARM = os.environ["ARM"]
ROOT = Path(__file__).resolve().parents[1]
os.environ["FIT_TARGETS_DIR"] = f"fit_targets_integ/{ARM}"
WORK = ROOT / "fit_workdirs" / f"eval_{ARM}"
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plot_best_fit_results import (  # noqa: E402
    FIT_DATA_DIR,
    ensure_fit_data_links_paper,
    load_mean_data_results,
    load_plot_model,
    make_shared_stimuli,
    plot_one,
)
from _fit_data import load_avg_mean_r, load_s_unsplit80  # noqa: E402
from _tmp_mpre3_meancell_eval import newest_final, score, stim_IM_at  # noqa: E402
from _tmp_mpre3_meancell_plots import alias_svgs, fmt  # noqa: E402
import model_functions as mf  # noqa: E402
from model_functions import int_regs, move_regs, run_model  # noqa: E402

ALYX_MODELS = Path.home() / "Downloads/ONE/alyx.internationalbrainlab.org/models"
OA_MODELS = Path.home() / "Downloads/ONE/openalyx.internationalbrainlab.org/models"
SEEDS = (7, 12, 34, 45, 89, 101, 303, 333)
RUNS = {
    ARM: ALYX_MODELS / (
        f"weights_run_fj_stageB_hold_newS34_full_mpre3_meancell_integ_{ARM}_full_masknone"
    ),
    "prod_s89": OA_MODELS / "weights_run_fj_stageB_hold_s89_full_mpre3_meancell_full_masknone",
}
OUT = ALYX_MODELS / f"stageB_hold_newS34_integ_{ARM}_eval.json"


def main():
    assert FIT_DATA_DIR.name == ARM, FIT_DATA_DIR
    ensure_fit_data_links_paper()
    _, mean_data = load_mean_data_results()
    _, avg_mean_R = load_avg_mean_r()
    stim_curve, payload = load_s_unsplit80()
    prior_regions = {
        "int_regs_choice": int_regs, "int_regs_stim": int_regs,
        "move_regs_choice": move_regs, "move_regs_stim": move_regs,
        "stim_regs": list(payload["regs_stim"]),
    }
    stim_ref = newest_final(OA_MODELS / "weights_run_fj_stageB_hold_s89_regular_mask12-13_s101")
    mp0, _ = load_plot_model(stim_ref)
    stim_bundle = make_shared_stimuli(mp0, bps=20, seed=12345)
    stimuli, trial_strengths, trial_sides, block_sides, steps_before_obs, bps = stim_bundle
    print(f"ARM={ARM} targets={FIT_DATA_DIR} cwd={Path.cwd()} HAVE_NUMBA={mf._HAVE_NUMBA} "
          f"S prior regs={payload['regs_stim']}", flush=True)

    rows = []
    for name, prefix in RUNS.items():
        for seed in SEEDS:
            jp = newest_final(Path(f"{prefix}_s{seed}"))
            mp, meta = load_plot_model(jp)
            results = run_model(
                "data", stimuli, trial_strengths, trial_sides, block_sides, bps,
                steps_before_obs=steps_before_obs, verbose=False, backend="numba", **mp,
            )
            info = {k: float(mp[k]) for k in ("g_i", "g_m", "g_s", "d_i", "d_m", "d_s")}
            info.update(recorded=float(meta.get("loss", np.nan)), json=str(jp))
            shape = stim_IM_at(results, steps_before_obs, mp)
            for w in (3.0, 1.0):
                ev = score(mp, results, steps_before_obs, mean_data, prior_regions,
                           avg_mean_R, stim_curve, True, w)
                rec = {"run": name, "seed": seed, **info, **ev}
                if w == 1.0:
                    rec.update(shape)
                rows.append(rec)
                print(f"{name:18} s{seed:<4} w{int(w)} rec {info['recorded']:.3f} "
                      f"traj {ev['eval_traj']:.3f} IM {ev['eval_IM']:.3f} "
                      f"S {ev['eval_prior_S']:.3f} LS {ev['eval_LS']:.3f} "
                      f"fair {ev['eval_fair']:.3f} gi {info['g_i']:.3g} gs {info['g_s']:.3g}",
                      flush=True)
            OUT.write_text(json.dumps({"targets": str(FIT_DATA_DIR), "eval": rows},
                                      indent=2, default=str))
    print(f"wrote {OUT}", flush=True)
    if os.environ.get("PLOTS", "1") == "0":
        return

    from _tmp_perf_rt_model_vs_data import (  # noqa: E402
        ALYX, BEHAVIOR_ACT, build_actprior_behavior, plot_one_json,
    )
    from analyze_choice_epochs import load_sessions_from_aggregate  # noqa: E402

    traj_rows, rt_rows = [], []
    for seed in SEEDS:
        run = Path(f"{RUNS[ARM]}_s{seed}")
        s = plot_one(newest_final(run), stim_bundle, mean_data, prior_regions, run,
                     avg_mean_R=avg_mean_R, include_stim=True, prior_plot_window=80)
        plt.close("all")
        alias_svgs(run)
        s["seed"] = seed
        traj_rows.append(s)
        print(f"[plots] s{seed} I {fmt(s.get('gof_I_pre'))}/{fmt(s.get('gof_I_post'))} "
              f"M {fmt(s.get('gof_M_pre'))}/{fmt(s.get('gof_M_post'))} "
              f"S {fmt(s.get('gof_S'))} priorS {fmt(s.get('prior_S'))}", flush=True)

    if BEHAVIOR_ACT.is_file():
        behavior = np.load(BEHAVIOR_ACT, allow_pickle=True).item()
    else:
        behavior = build_actprior_behavior(load_sessions_from_aggregate(ALYX))
        BEHAVIOR_ACT.parent.mkdir(parents=True, exist_ok=True)
        np.save(BEHAVIOR_ACT, behavior, allow_pickle=True)
    for seed in SEEDS:
        run = Path(f"{RUNS[ARM]}_s{seed}")
        r = plot_one_json(f"{ARM}_s{seed}", newest_final(run), run, behavior)
        r["seed"] = seed
        rt_rows.append(r)
        print(f"[rt] s{seed} perf {r['perf_r2']:.3f} RT {r['rt_r2']:.3f} "
              f"split {r['rt_split_r2']:.3f}", flush=True)
    (ALYX_MODELS / f"stageB_hold_newS34_integ_{ARM}_plot_summary.json").write_text(
        json.dumps({"traj": traj_rows, "rt": rt_rows}, indent=2, default=str))


if __name__ == "__main__":
    main()
