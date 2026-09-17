"""Fair 80 vs 150 ms I/M window for regular / full ± mpre3 (current meancell).

Shared stim bps=20 seed 12345 from regular s101. Rank at m_pre_weight=1.
Full arms include unsplit-80 S nSSE. Window is an eval override, not JSON.
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
    int_regs,
    loss_plot_diff_by_condition_with_data,
    loss_prior_effect,
    mean_by_condition,
    mean_S_by_contrast,
    move_regs,
    prior_stratum_of,
    resolve_prior_distance_window,
    run_model,
)

BASE = Path.home() / (
    "Downloads/ONE/openalyx.internationalbrainlab.org/models"
)
NEW = BASE / "new"
SEEDS = (7, 12, 34, 45, 89, 101, 303, 333)
FAMILIES = {
    "reg80": (
        "weights_run_fj_stageB_hold_s89_regular_mask12-13", False,
    ),
    "reg80_mpre3": (
        "weights_run_fj_stageB_hold_s89_mpre3_regular_mask12-13", False,
    ),
    "reg150": (
        "weights_run_fj_stageB_hold_s89_im150_meancell_regular_mask12-13",
        False,
    ),
    "reg150_mpre3": (
        "weights_run_fj_stageB_hold_s89_mpre3_im150_meancell_regular_mask12-13",
        False,
    ),
    "full80": (
        "weights_run_fj_stageB_hold_s89_full_full_masknone", True,
    ),
    "full80_mpre3": (
        "weights_run_fj_stageB_hold_s89_full_mpre3_meancell_full_masknone",
        True,
    ),
    "full150": (
        "weights_run_fj_stageB_hold_s89_full_im150_meancell_full_masknone",
        True,
    ),
    "full150_mpre3": (
        "weights_run_fj_stageB_hold_s89_full_mpre3_im150_meancell_full_masknone",
        True,
    ),
}
POOL = {
    "regular": ("reg80", "reg150"),
    "regular_mpre3": ("reg80_mpre3", "reg150_mpre3"),
    "full": ("full80", "full150"),
    "full_mpre3": ("full80_mpre3", "full150_mpre3"),
}
OUT = BASE / "stageB_hold_s89_mpre3_crosswindow_meancell_eval.json"
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


def mp_for_window(mp, window_ms):
    out = dict(mp)
    out["m_pre_weight"] = 1.0
    if window_ms is None:
        out.pop("prior_window_ms", None)
    else:
        out["prior_window_ms"] = float(window_ms)
    return out


def score_prior(mp, results, steps_before_obs, prior_regions, stim_curve,
                include_stim):
    T_prior, plot_win, _ = resolve_prior_distance_window(mp, T=72, plot_window=80)
    kw = dict(
        regions=prior_regions, results=results, model_params=mp,
        steps_before_obs=steps_before_obs, T=T_prior, model_metric="l2",
        timeframes=("act_block_duringstim", "act_block_duringchoice"),
        ptype="p_mean_c", plot_window=plot_win, reload=False,
        label_A="integrator", label_B="move", do_plot=False,
        plot_shifted=False, ylim=None, scale_factors=[1, 1, 1],
        include_all_trials=True, plot_stim=include_stim, lump_all=False,
        include_stim=include_stim,
    )
    if include_stim:
        kw["stim_curve_path"] = str(stim_curve)
    loss_prior = loss_prior_effect(**kw)
    ds = loss_prior.get("act_block_duringstim") or {}
    prior = float(loss_prior["total"])
    s_nsse = float(ds.get("stim", 0.0) or 0.0) if include_stim else 0.0
    return prior, s_nsse, prior - s_nsse if include_stim else prior


def fmt_g(x):
    v = float(x)
    if abs(v) < 1e-6:
        return "~0"
    if abs(v) >= 10:
        return f"{v:.3g}"
    return f"{v:.3g}"


def best_of(rows, key):
    return min(rows, key=lambda r: r[key])


def print_best_table(title, groups):
    print(f"\n=== {title} ===")
    hdr = (
        f"{'family':16} {'fit':12} {'seed':>4} {'tot80':>7} {'tot150':>7} "
        f"{'gi':>8} {'gm':>8} {'gs':>8} {'di':>8} {'dm':>8} {'ds':>8}"
    )
    print(hdr)
    for fam, row in groups:
        print(
            f"{fam:16} {row['family']:12} {row['seed']:4d} "
            f"{row['tot80']:7.3f} {row['tot150']:7.3f} "
            f"{fmt_g(row['g_i']):>8} {fmt_g(row['g_m']):>8} "
            f"{fmt_g(row['g_s']):>8} {fmt_g(row['d_i']):>8} "
            f"{fmt_g(row['d_m']):>8} {fmt_g(row['d_s']):>8}"
        )


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
        f"S sidecar={stim_curve.name}  m_pre_weight=1  metric=mean_c",
        flush=True,
    )
    rows = []
    print(
        f"{'fam':16} {'seed':>4} {'tot80':>7} {'tot150':>7} {'traj':>7} "
        f"{'IM80':>7} {'IM150':>7} {'S':>7} {'LS':>7} {'gi':>8} {'ds':>8}",
        flush=True,
    )
    for fam, (prefix, include_stim) in FAMILIES.items():
        for seed in SEEDS:
            d = resolve_run(prefix, seed)
            jp = newest_final(d)
            mp, meta = load_plot_model(jp)
            results = run_model(
                "data", stimuli, trial_strengths, trial_sides, block_sides, bps,
                steps_before_obs=steps_before_obs, verbose=False,
                backend="numba", **mp,
            )
            mp1 = dict(mp)
            mp1["m_pre_weight"] = 1.0
            sim_out = mean_by_condition(results, steps_before_obs)
            loss_traj = loss_plot_diff_by_condition_with_data(
                sim_out, mp1, var_names=("I", "P", "M"),
                mean_data_results=mean_data, plot=False,
            )
            traj = float(loss_traj["total"])
            S_avg = mean_S_by_contrast(results, steps_before_obs)
            raw_ls = compute_sse_stim_right(
                S_avg, avg_mean_R, baseline_R=0,
            )["total_loss"]
            L_S = float(raw_ls) if np.isfinite(raw_ls) else float("nan")
            wins = {}
            for label, wms in (("80", None), ("150", 150.0)):
                mpw = mp_for_window(mp, wms)
                prior, s_nsse, im = score_prior(
                    mpw, results, steps_before_obs, prior_regions,
                    stim_curve, include_stim,
                )
                wins[label] = {
                    "prior": prior, "S": s_nsse, "IM": im,
                    "tot": traj + prior + L_S,
                }
            rec = {
                "family": fam,
                "pool": next(k for k, v in POOL.items() if fam in v),
                "seed": seed,
                "include_stim": include_stim,
                "json": jp.name,
                "json_m_pre_weight": (meta.get("model_params") or {}).get(
                    "m_pre_weight"
                ),
                "json_prior_window_ms": (meta.get("model_params") or {}).get(
                    "prior_window_ms"
                ),
                "score_stratum": prior_stratum_of(mp),
                "g_i": float(mp["g_i"]),
                "g_m": float(mp["g_m"]),
                "g_s": float(mp["g_s"]),
                "d_i": float(mp["d_i"]),
                "d_m": float(mp["d_m"]),
                "d_s": float(mp["d_s"]),
                "traj": traj,
                "LS": L_S,
                "S": wins["80"]["S"],
                "IM80": wins["80"]["IM"],
                "IM150": wins["150"]["IM"],
                "tot80": wins["80"]["tot"],
                "tot150": wins["150"]["tot"],
            }
            rows.append(rec)
            print(
                f"{fam:16} {seed:4d} {rec['tot80']:7.3f} {rec['tot150']:7.3f} "
                f"{traj:7.3f} {rec['IM80']:7.3f} {rec['IM150']:7.3f} "
                f"{rec['S']:7.3f} {L_S:7.3f} {rec['g_i']:8.3g} {rec['d_s']:8.3g}",
                flush=True,
            )
            OUT.write_text(json.dumps({"eval": rows}, indent=2, default=str))

    tag_best_80 = [(fam, best_of(
        [r for r in rows if r["family"] == fam], "tot80",
    )) for fam in FAMILIES]
    tag_best_150 = [(fam, best_of(
        [r for r in rows if r["family"] == fam], "tot150",
    )) for fam in FAMILIES]
    pool_best_80 = [(pool, best_of(
        [r for r in rows if r["pool"] == pool], "tot80",
    )) for pool in POOL]
    pool_best_150 = [(pool, best_of(
        [r for r in rows if r["pool"] == pool], "tot150",
    )) for pool in POOL]
    print_best_table("best seed per fit tag, ranked at 80 ms tot", tag_best_80)
    print_best_table("best seed per fit tag, ranked at 150 ms tot", tag_best_150)
    print_best_table(
        "best seed per family (80-fit ∪ 150-fit), ranked at 80 ms tot",
        pool_best_80,
    )
    print_best_table(
        "best seed per family (80-fit ∪ 150-fit), ranked at 150 ms tot",
        pool_best_150,
    )
    OUT.write_text(json.dumps({"eval": rows}, indent=2, default=str))
    print(f"\nwrote {OUT}  n={len(rows)}")


if __name__ == "__main__":
    main()
