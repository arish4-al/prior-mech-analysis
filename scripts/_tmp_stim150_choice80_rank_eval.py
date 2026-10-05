"""Rank all 6 regular + 6 full tags at stim150 / choice80.

Shared stim bps=20 seed 12345 from regular s101. m_pre_weight=1.
Traj + I/M prior use im_window_stim_ms=150 / im_window_choice_ms=80
(split keys win). Full tot includes unsplit-80 S nSSE; tot_noS drops
that S prior term so it can sit next to regular tot.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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
from _fit_data import load_avg_mean_r, load_s_unsplit80  # noqa: E402
import model_functions as mf  # noqa: E402
from model_functions import (  # noqa: E402
    compute_sse_stim_right,
    im_traj_T_of,
    int_regs,
    loss_plot_diff_by_condition_with_data,
    loss_prior_effect,
    mean_by_condition,
    mean_S_by_contrast,
    move_regs,
    run_model,
)

BASE = Path.home() / (
    "Downloads/ONE/openalyx.internationalbrainlab.org/models"
)
NEW = BASE / "new"
SEEDS = (7, 12, 34, 45, 89, 101, 303, 333)
FAMILIES = {
    "regular 80": (
        "weights_run_fj_stageB_hold_s89_regular_mask12-13", False,
    ),
    "regular 80 mpre3": (
        "weights_run_fj_stageB_hold_s89_mpre3_regular_mask12-13", False,
    ),
    "regular 150": (
        "weights_run_fj_stageB_hold_s89_im150_meancell_regular_mask12-13",
        False,
    ),
    "regular 150 mpre3": (
        "weights_run_fj_stageB_hold_s89_mpre3_im150_meancell_regular_mask12-13",
        False,
    ),
    "regular split 150/80": (
        "weights_run_fj_stageB_hold_s89_stim150_choice80_meancell_regular_mask12-13",
        False,
    ),
    "regular split mpre3": (
        "weights_run_fj_stageB_hold_s89_mpre3_stim150_choice80_meancell_regular_mask12-13",
        False,
    ),
    "`full` 80": (
        "weights_run_fj_stageB_hold_s89_full_full_masknone", True,
    ),
    "`full` 80 mpre3": (
        "weights_run_fj_stageB_hold_s89_full_mpre3_meancell_full_masknone",
        True,
    ),
    "`full` 150": (
        "weights_run_fj_stageB_hold_s89_full_im150_meancell_full_masknone",
        True,
    ),
    "`full` 150 mpre3": (
        "weights_run_fj_stageB_hold_s89_full_mpre3_im150_meancell_full_masknone",
        True,
    ),
    "`full` split 150/80": (
        "weights_run_fj_stageB_hold_s89_full_stim150_choice80_meancell_full_masknone",
        True,
    ),
    "`full` split mpre3": (
        "weights_run_fj_stageB_hold_s89_full_mpre3_stim150_choice80_meancell_full_masknone",
        True,
    ),
}
REGULAR = [k for k in FAMILIES if k.startswith("regular")]
FULL = [k for k in FAMILIES if k.startswith("`full`")]
OUT = BASE / "stageB_hold_s89_stim150_choice80_rank_eval.json"
TS = re.compile(r"(\d{8}-\d{6})")


def newest_final(d: Path) -> Path:
    finals = list(d.glob("weights_final_*.json"))
    if not finals:
        raise FileNotFoundError(d)

    def key(p):
        m = TS.search(p.name)
        return (m.group(1) if m else "", p.stat().st_mtime)

    return max(finals, key=key)


def resolve_run(prefix: str, seed: int) -> Path:
    for root in (BASE, NEW):
        d = root / f"{prefix}_s{seed}"
        if d.is_dir() and list(d.glob("weights_final_*.json")):
            return d
    raise FileNotFoundError(f"{prefix}_s{seed}")


def mp_split(mp):
    out = dict(mp)
    out["m_pre_weight"] = 1.0
    out["prior_window_ms"] = None
    out["im_window_stim_ms"] = 150.0
    out["im_window_choice_ms"] = 80.0
    return out


def fmt_g(x):
    v = float(x)
    if abs(v) < 1e-6:
        return "~0"
    return f"{v:.3g}"


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
    (
        stimuli, trial_strengths, trial_sides, block_sides,
        steps_before_obs, bps,
    ) = stim_bundle
    print(
        f"HAVE_NUMBA={mf._HAVE_NUMBA}  stim from {stim_ref.parent.name}  "
        f"window=stim150/choice80  m_pre=1",
        flush=True,
    )
    rows = []
    print(
        f"{'tag':24} {'seed':>4} {'tot':>7} {'noS':>7} {'traj':>7} "
        f"{'IM':>7} {'S':>7} {'LS':>7} {'gi':>8} {'ds':>8}",
        flush=True,
    )
    for tag, (prefix, include_stim) in FAMILIES.items():
        for seed in SEEDS:
            d = resolve_run(prefix, seed)
            jp = newest_final(d)
            mp, _meta = load_plot_model(jp)
            results = run_model(
                "data", stimuli, trial_strengths, trial_sides, block_sides, bps,
                steps_before_obs=steps_before_obs, verbose=False,
                backend="numba", **mp,
            )
            mpw = mp_split(mp)
            T_post, T_pre = im_traj_T_of(mpw)
            sim = mean_by_condition(
                results, steps_before_obs, T=T_post, T_pre=T_pre,
            )
            loss_traj = loss_plot_diff_by_condition_with_data(
                sim, mpw, var_names=("I", "P", "M"),
                mean_data_results=mean_data, plot=False,
            )
            traj = float(loss_traj["total"])
            T_stim, win_stim, _ = mf.resolve_im_window(
                mpw, "stim", T=72, plot_window=80)
            kw = dict(
                regions=prior_regions, results=results, model_params=mpw,
                steps_before_obs=steps_before_obs, T=T_stim, model_metric="l2",
                timeframes=("act_block_duringstim", "act_block_duringchoice"),
                ptype="p_mean_c", plot_window=win_stim, reload=False,
                label_A="integrator", label_B="move", do_plot=False,
                plot_shifted=False, ylim=None, scale_factors=[1, 1, 1],
                include_all_trials=True, plot_stim=include_stim,
                lump_all=False, include_stim=include_stim,
            )
            if include_stim:
                kw["stim_curve_path"] = str(stim_curve)
            loss_prior = loss_prior_effect(**kw)
            prior = float(loss_prior["total"])
            ds = loss_prior.get("act_block_duringstim") or {}
            s_nsse = float(ds.get("stim", 0.0) or 0.0) if include_stim else 0.0
            im = prior - s_nsse if include_stim else prior
            raw_ls = compute_sse_stim_right(
                mean_S_by_contrast(results, steps_before_obs),
                avg_mean_R, baseline_R=0,
            )["total_loss"]
            L_S = float(raw_ls) if np.isfinite(raw_ls) else float("nan")
            tot = traj + prior + L_S
            tot_noS = traj + im + L_S
            rec = {
                "tag": tag,
                "include_stim": include_stim,
                "seed": seed,
                "g_i": float(mp["g_i"]),
                "g_m": float(mp["g_m"]),
                "g_s": float(mp["g_s"]),
                "d_i": float(mp["d_i"]),
                "d_m": float(mp["d_m"]),
                "d_s": float(mp["d_s"]),
                "traj": traj,
                "IM": im,
                "S": s_nsse,
                "LS": L_S,
                "tot": tot,
                "tot_noS": tot_noS,
            }
            rows.append(rec)
            print(
                f"{tag:24} {seed:4d} {tot:7.3f} {tot_noS:7.3f} {traj:7.3f} "
                f"{im:7.3f} {s_nsse:7.3f} {L_S:7.3f} "
                f"{rec['g_i']:8.3g} {rec['d_s']:8.3g}",
                flush=True,
            )
            OUT.write_text(json.dumps({"eval": rows}, indent=2, default=str))

    def bests(tags):
        out = []
        for tag in tags:
            rs = [r for r in rows if r["tag"] == tag]
            out.append(min(rs, key=lambda r: r["tot"]))
        out.sort(key=lambda r: r["tot"])
        return out

    def print_reg(title, recs):
        print(f"\n=== {title} ===")
        print(
            f"{'tag':24} {'seed':>4} {'tot150/80':>9} "
            f"{'gi':>8} {'gm':>8} {'gs':>8} {'di':>8} {'dm':>8} {'ds':>8}"
        )
        for r in recs:
            print(
                f"{r['tag']:24} {r['seed']:4d} {r['tot']:9.3f} "
                f"{fmt_g(r['g_i']):>8} {fmt_g(r['g_m']):>8} "
                f"{fmt_g(r['g_s']):>8} {fmt_g(r['d_i']):>8} "
                f"{fmt_g(r['d_m']):>8} {fmt_g(r['d_s']):>8}"
            )

    def print_full(title, recs):
        print(f"\n=== {title} ===")
        print(
            f"{'tag':24} {'seed':>4} {'tot150/80':>9} {'tot_noS':>8} "
            f"{'gi':>8} {'gm':>8} {'gs':>8} {'di':>8} {'dm':>8} {'ds':>8}"
        )
        for r in recs:
            print(
                f"{r['tag']:24} {r['seed']:4d} {r['tot']:9.3f} "
                f"{r['tot_noS']:8.3f} "
                f"{fmt_g(r['g_i']):>8} {fmt_g(r['g_m']):>8} "
                f"{fmt_g(r['g_s']):>8} {fmt_g(r['d_i']):>8} "
                f"{fmt_g(r['d_m']):>8} {fmt_g(r['d_s']):>8}"
            )

    print_reg("regular, ranked at stim150/choice80 tot", bests(REGULAR))
    print_full(
        "full, ranked at stim150/choice80 tot  (tot_noS = tot minus S prior nSSE)",
        bests(FULL),
    )
    OUT.write_text(json.dumps({"eval": rows}, indent=2, default=str))
    print(f"\nwrote {OUT}  n={len(rows)}")


if __name__ == "__main__":
    main()
