"""Fair 80 vs 150 ms I/M window for stim150/choice80 ± mpre3 (current meancell).

Shared stim bps=20 seed 12345 from regular s101. Rank at m_pre_weight=1.
Full arms include unsplit-80 S nSSE.

Fair 80/150 **clears** ``im_window_*`` so split keys cannot win over the
eval override (same tot as 09-16b). Traj stays default T=72 for those
tots. As-fitted tot uses JSON 150/80 split for traj + prior.
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
    prior_stratum_of,
    resolve_im_window,
    run_model,
)

BASE = Path.home() / (
    "Downloads/ONE/openalyx.internationalbrainlab.org/models"
)
SEEDS = (7, 12, 34, 45, 89, 101, 303, 333)
FAMILIES = {
    "reg_split": (
        "weights_run_fj_stageB_hold_s89_stim150_choice80_meancell_regular_mask12-13",
        False,
    ),
    "reg_split_mpre3": (
        "weights_run_fj_stageB_hold_s89_mpre3_stim150_choice80_meancell_regular_mask12-13",
        False,
    ),
    "full_split": (
        "weights_run_fj_stageB_hold_s89_full_stim150_choice80_meancell_full_masknone",
        True,
    ),
    "full_split_mpre3": (
        "weights_run_fj_stageB_hold_s89_full_mpre3_stim150_choice80_meancell_full_masknone",
        True,
    ),
}
OLD_DUMP = BASE / "stageB_hold_s89_mpre3_crosswindow_meancell_eval.json"
OUT = BASE / "stageB_hold_s89_splitwin_crosswindow_meancell_eval.json"
TS = re.compile(r"(\d{8}-\d{6})")
OLD_LABEL = {
    "reg80": "regular 80",
    "reg80_mpre3": "regular 80 mpre3",
    "reg150": "regular 150 meancell",
    "reg150_mpre3": "regular 150 mpre3",
    "full80": "`full` 80",
    "full80_mpre3": "`full` 80 mpre3",
    "full150": "`full` 150 meancell",
    "full150_mpre3": "`full` 150 mpre3",
}
NEW_LABEL = {
    "reg_split": "regular split",
    "reg_split_mpre3": "regular split mpre3",
    "full_split": "`full` split",
    "full_split_mpre3": "`full` split mpre3",
}


def newest_final(d: Path) -> Path:
    finals = list(d.glob("weights_final_*.json"))
    if not finals:
        raise FileNotFoundError(d)

    def key(p):
        m = TS.search(p.name)
        return (m.group(1) if m else "", p.stat().st_mtime)

    return max(finals, key=key)


def mp_fair(mp, window_ms):
    """Shared-window fair score: drop split keys so they cannot win."""
    out = dict(mp)
    out["m_pre_weight"] = 1.0
    out["im_window_stim_ms"] = None
    out["im_window_choice_ms"] = None
    if window_ms is None:
        out["prior_window_ms"] = None
    else:
        out["prior_window_ms"] = float(window_ms)
    return out


def score_prior(mp, results, steps_before_obs, prior_regions, stim_curve,
                include_stim):
    T_stim, win_stim, _ = resolve_im_window(mp, "stim", T=72, plot_window=80)
    kw = dict(
        regions=prior_regions, results=results, model_params=mp,
        steps_before_obs=steps_before_obs, T=T_stim, model_metric="l2",
        timeframes=("act_block_duringstim", "act_block_duringchoice"),
        ptype="p_mean_c", plot_window=win_stim, reload=False,
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
    return f"{v:.3g}"


def best_of(rows, key):
    return min(rows, key=lambda r: r[key])


def print_best_table(title, groups, label_of):
    print(f"\n=== {title} ===")
    hdr = (
        f"{'tag':24} {'seed':>4} {'tot80':>7} {'tot150':>7} "
        f"{'gi':>8} {'gm':>8} {'gs':>8} {'di':>8} {'dm':>8} {'ds':>8}"
    )
    print(hdr)
    for fam, row in groups:
        print(
            f"{label_of[fam]:24} {row['seed']:4d} "
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
        f"{'fam':18} {'seed':>4} {'tot80':>7} {'tot150':>7} {'asfit':>7} "
        f"{'traj72':>7} {'tr_af':>7} {'IM80':>7} {'IM150':>7} {'S':>7} "
        f"{'LS':>7} {'gi':>8} {'ds':>8}",
        flush=True,
    )
    for fam, (prefix, include_stim) in FAMILIES.items():
        for seed in SEEDS:
            d = BASE / f"{prefix}_s{seed}"
            jp = newest_final(d)
            mp, meta = load_plot_model(jp)
            results = run_model(
                "data", stimuli, trial_strengths, trial_sides, block_sides, bps,
                steps_before_obs=steps_before_obs, verbose=False,
                backend="numba", **mp,
            )
            mp1 = dict(mp)
            mp1["m_pre_weight"] = 1.0
            sim72 = mean_by_condition(results, steps_before_obs)
            loss_traj72 = loss_plot_diff_by_condition_with_data(
                sim72, mp1, var_names=("I", "P", "M"),
                mean_data_results=mean_data, plot=False,
            )
            traj72 = float(loss_traj72["total"])
            T_post, T_pre = im_traj_T_of(mp1)
            sim_af = mean_by_condition(
                results, steps_before_obs, T=T_post, T_pre=T_pre,
            )
            loss_traj_af = loss_plot_diff_by_condition_with_data(
                sim_af, mp1, var_names=("I", "P", "M"),
                mean_data_results=mean_data, plot=False,
            )
            traj_af = float(loss_traj_af["total"])
            S_avg = mean_S_by_contrast(results, steps_before_obs)
            raw_ls = compute_sse_stim_right(
                S_avg, avg_mean_R, baseline_R=0,
            )["total_loss"]
            L_S = float(raw_ls) if np.isfinite(raw_ls) else float("nan")
            wins = {}
            for label, wms in (("80", None), ("150", 150.0)):
                mpw = mp_fair(mp, wms)
                prior, s_nsse, im = score_prior(
                    mpw, results, steps_before_obs, prior_regions,
                    stim_curve, include_stim,
                )
                wins[label] = {
                    "prior": prior, "S": s_nsse, "IM": im,
                    "tot": traj72 + prior + L_S,
                }
            prior_af, s_af, im_af = score_prior(
                mp1, results, steps_before_obs, prior_regions,
                stim_curve, include_stim,
            )
            tot_af = traj_af + prior_af + L_S
            jmp = meta.get("model_params") or {}
            rec = {
                "family": fam,
                "seed": seed,
                "include_stim": include_stim,
                "json": jp.name,
                "json_m_pre_weight": jmp.get("m_pre_weight"),
                "json_im_window_stim_ms": jmp.get("im_window_stim_ms"),
                "json_im_window_choice_ms": jmp.get("im_window_choice_ms"),
                "score_stratum": prior_stratum_of(mp),
                "T_post": int(T_post),
                "T_pre": int(T_pre),
                "g_i": float(mp["g_i"]),
                "g_m": float(mp["g_m"]),
                "g_s": float(mp["g_s"]),
                "d_i": float(mp["d_i"]),
                "d_m": float(mp["d_m"]),
                "d_s": float(mp["d_s"]),
                "traj72": traj72,
                "traj_asfit": traj_af,
                "LS": L_S,
                "S": wins["80"]["S"],
                "IM80": wins["80"]["IM"],
                "IM150": wins["150"]["IM"],
                "IMaf": im_af,
                "tot80": wins["80"]["tot"],
                "tot150": wins["150"]["tot"],
                "tot_asfit": tot_af,
            }
            rows.append(rec)
            print(
                f"{fam:18} {seed:4d} {rec['tot80']:7.3f} {rec['tot150']:7.3f} "
                f"{tot_af:7.3f} {traj72:7.3f} {traj_af:7.3f} "
                f"{rec['IM80']:7.3f} {rec['IM150']:7.3f} "
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
    tag_best_af = [(fam, best_of(
        [r for r in rows if r["family"] == fam], "tot_asfit",
    )) for fam in FAMILIES]
    print_best_table(
        "splitwin best seed per tag, ranked at 80 ms tot (fair shared window)",
        tag_best_80, NEW_LABEL,
    )
    print_best_table(
        "splitwin best seed per tag, ranked at 150 ms tot (fair shared window)",
        tag_best_150, NEW_LABEL,
    )
    print("\n=== splitwin best seed per tag, as-fitted 150/80 ===")
    print(
        f"{'tag':24} {'seed':>4} {'asfit':>7} {'tot80':>7} {'tot150':>7} "
        f"{'gi':>8} {'ds':>8}"
    )
    for fam, row in tag_best_af:
        print(
            f"{NEW_LABEL[fam]:24} {row['seed']:4d} "
            f"{row['tot_asfit']:7.3f} {row['tot80']:7.3f} {row['tot150']:7.3f} "
            f"{fmt_g(row['g_i']):>8} {fmt_g(row['d_s']):>8}"
        )

    combined = list(rows)
    if OLD_DUMP.is_file():
        old = json.loads(OLD_DUMP.read_text()).get("eval") or []
        combined = old + rows
        label_of = {**OLD_LABEL, **NEW_LABEL}
        fams = list(OLD_LABEL) + list(NEW_LABEL)
        comb_80 = [(fam, best_of(
            [r for r in combined if r["family"] == fam], "tot80",
        )) for fam in fams]
        comb_150 = [(fam, best_of(
            [r for r in combined if r["family"] == fam], "tot150",
        )) for fam in fams]
        print_best_table(
            "ALL tags (09-16b ∪ splitwin), ranked at 80 ms tot",
            comb_80, label_of,
        )
        print_best_table(
            "ALL tags (09-16b ∪ splitwin), ranked at 150 ms tot",
            comb_150, label_of,
        )

    OUT.write_text(json.dumps({
        "eval": rows,
        "combined_n": len(combined),
    }, indent=2, default=str))
    print(f"\nwrote {OUT}  n={len(rows)}")


if __name__ == "__main__":
    main()
