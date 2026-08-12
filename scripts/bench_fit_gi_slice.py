"""
Phase 2c: 1-D loss slice over g_i with other params fixed at checkpoint (and paper).

Stage-2 protocol: bps=20, numba, optional multi-seed stim. Reports traj vs prior.

Usage (iblenv, outside sandbox):
  PYTHONPATH=. python scripts/bench_fit_gi_slice.py
  PYTHONPATH=. python scripts/bench_fit_gi_slice.py --seeds 0,1,2 --n-grid 21
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from simulate_recovery import resolve_weights_json
import model_functions as mf
from model_functions import (
    create_stimuli,
    run_model,
    mean_by_condition,
    loss_plot_diff_by_condition_with_data,
    loss_prior_effect,
    pth_res,
    int_regs,
    move_regs,
    trials_per_block_param,
    block_side_probs,
    num_stimulus_strength,
    min_stimulus_strength,
    max_stimulus_strength,
    min_trials_per_block,
    max_trials_per_block,
)

# Paper hand-tweaks (c → i naming)
PAPER = {
    "W_ii": 0.43,
    "W_pp": 0.496,
    "W_mm": 0.27,
    "W_is": 0.17,
    "W_pi": 1.63e-5,  # not in paper; use ckpt scale
    "W_mi": 0.50,
    "g_i": 163.0,
    "g_m": 1e-12,
    "d_i": 21.4,
    "d_m": 1e-12,
    "theta_c": 0.76,
    "theta_d": 0.40,
}

STEPS_BEFORE_OBS = 500
MAX_OBS = 1000
BPS = 20


def _ensure_data_links():
    try:
        from _fit_data import ensure_fit_data_links
    except ImportError:
        from scripts._fit_data import ensure_fit_data_links
    ensure_fit_data_links(require_avg_mean_r=False, mean_and_prior=True)


def load_ckpt_params(weights: Path):
    raw = json.loads(weights.read_text())
    ckpt = {
        "W_ii": float(raw["W"]["W_ii"]),
        "W_pp": float(raw["W"]["W_pp"]),
        "W_mm": float(raw["W"]["W_mm"]),
        "W_is": float(raw["W"]["W_is"]),
        "W_pi": float(raw["W"]["W_pi"]),
        "W_mi": float(raw["W"]["W_mi"]),
        "g_i": float(raw["g"]["g_i"]),
        "g_m": max(float(raw["g"]["g_m"]), 1e-12),
        "d_i": float(raw["d"]["d_i"]),
        "d_m": max(float(raw["d"]["d_m"]), 1e-12),
        "theta_c": float(raw["theta"]["theta_c"]),
        "theta_d": float(raw["theta"]["theta_d"]),
    }
    return ckpt, float(raw["loss"]), raw.get("model_params") or {}


def mp_from_base(base: dict, g_i: float) -> dict:
    mp = dict(mf.model_params)
    mp.update(
        {
            "W_ii": base["W_ii"],
            "W_pp": base["W_pp"],
            "W_mm": base["W_mm"],
            "W_is": base["W_is"],
            "W_pi": base["W_pi"],
            "W_mi": base["W_mi"],
            "g_i": float(g_i),
            "g_m": base["g_m"],
            "d_i": base["d_i"],
            "d_m": base["d_m"],
            "g_s": 0.0,
            "d_s": 0.0,
            "direct_offset": False,
            "action_thresholds": {
                "concordant": {
                    c: base["theta_c"] for c in [1.0, 0.25, 0.125, 0.0625, 0.0]
                },
                "discordant": {
                    c: base["theta_d"] for c in [1.0, 0.25, 0.125, 0.0625, 0.0]
                },
            },
        }
    )
    mp["dt"] = 2.0
    mf._update_model_params_for_dt(mp, 2.0)
    return mp


def eval_loss(mp, mean_data, prior_regions, seed: int):
    stim_rng = np.random.default_rng(seed)
    stimuli, trial_strengths, _, trial_sides, block_sides = create_stimuli(
        BPS,
        trials_per_block_param,
        block_side_probs,
        num_stimulus_strength,
        min_stimulus_strength,
        max_stimulus_strength,
        min_trials_per_block,
        max_trials_per_block,
        MAX_OBS,
        STEPS_BEFORE_OBS,
        rng=stim_rng,
        **mp,
    )
    results = run_model(
        "data",
        stimuli,
        trial_strengths,
        trial_sides,
        block_sides,
        BPS,
        steps_before_obs=STEPS_BEFORE_OBS,
        verbose=False,
        backend="numba",
        **mp,
    )
    sim_out = mean_by_condition(
        results, STEPS_BEFORE_OBS, T=72, var_names=("I", "P", "M")
    )
    loss_traj = loss_plot_diff_by_condition_with_data(
        sim_out, mp, var_names=("I", "P", "M"), mean_data_results=mean_data, plot=False
    )
    loss_prior = loss_prior_effect(
        regions=prior_regions,
        results=results,
        model_params=mp,
        steps_before_obs=STEPS_BEFORE_OBS,
        T=72,
        timeframes=("act_block_duringstim", "act_block_duringchoice"),
        alpha=0.05,
        ptype="p_mean_c",
        label_A="integrator",
        label_B="move",
        do_plot=False,
        scale_factors=[1, 1, 1],
        include_all_trials=True,
    )
    return {
        "total": float(loss_traj["total"] + loss_prior["total"]),
        "traj": float(loss_traj["total"]),
        "prior": float(loss_prior["total"]),
    }


def slice_base(base, label, g_grid, seeds, mean_data, prior_regions):
    rows = []
    t0 = time.perf_counter()
    for g_i in g_grid:
        mp = mp_from_base(base, g_i)
        per_seed = [eval_loss(mp, mean_data, prior_regions, s) for s in seeds]
        tot = np.array([r["total"] for r in per_seed])
        trj = np.array([r["traj"] for r in per_seed])
        pri = np.array([r["prior"] for r in per_seed])
        rows.append(
            {
                "base": label,
                "g_i": float(g_i),
                "total_mean": float(np.mean(tot)),
                "total_std": float(np.std(tot)),
                "traj_mean": float(np.mean(trj)),
                "prior_mean": float(np.mean(pri)),
                "per_seed_total": tot.tolist(),
            }
        )
        print(
            f"[{label}] g_i={g_i:7.2f}  total={np.mean(tot):.6f}±{np.std(tot):.4f}  "
            f"traj={np.mean(trj):.6f}  prior={np.mean(pri):.6f}"
        )
    print(f"[{label}] wall={time.perf_counter() - t0:.1f}s")
    return rows


def summarize(rows, paper_gi, ckpt_gi):
    totals = np.array([r["total_mean"] for r in rows])
    gis = np.array([r["g_i"] for r in rows])
    i_min = int(np.argmin(totals))
    # nearest grid points to paper / ckpt
    i_paper = int(np.argmin(np.abs(gis - paper_gi)))
    i_ckpt = int(np.argmin(np.abs(gis - ckpt_gi)))
    return {
        "min_g_i": float(gis[i_min]),
        "min_total": float(totals[i_min]),
        "at_paper_g_i": {
            "g_i": float(gis[i_paper]),
            "total": float(totals[i_paper]),
            "traj": rows[i_paper]["traj_mean"],
            "prior": rows[i_paper]["prior_mean"],
        },
        "at_ckpt_g_i": {
            "g_i": float(gis[i_ckpt]),
            "total": float(totals[i_ckpt]),
            "traj": rows[i_ckpt]["traj_mean"],
            "prior": rows[i_ckpt]["prior_mean"],
        },
        "prefers_paper": bool(totals[i_paper] < totals[i_ckpt] - 1e-9),
        "delta_paper_minus_ckpt": float(totals[i_paper] - totals[i_ckpt]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g-min", type=float, default=120.0)
    ap.add_argument("--g-max", type=float, default=220.0)
    ap.add_argument("--n-grid", type=int, default=21)
    ap.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="Comma-separated stim seeds (multi-seed Stage-2 audit)",
    )
    ap.add_argument("--weights-json", type=str, default=None)
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Report JSON path (default: ONE models/gi_slice_report.json)",
    )
    args = ap.parse_args()

    _ensure_data_links()
    if not mf._HAVE_NUMBA:
        raise RuntimeError("numba required for fit-path slice")

    weights = Path(args.weights_json) if args.weights_json else resolve_weights_json()
    ckpt, json_loss, mp_nested = load_ckpt_params(weights)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    g_grid = np.linspace(args.g_min, args.g_max, args.n_grid)

    # Ensure paper/ckpt landmarks are on the grid
    landmarks = {PAPER["g_i"], ckpt["g_i"]}
    g_grid = np.unique(np.sort(np.concatenate([g_grid, list(landmarks)])))

    try:
        from _fit_data import load_validated_mean_data
    except ImportError:
        from scripts._fit_data import load_validated_mean_data
    _, mean_data = load_validated_mean_data()
    prior_regions = {
        "int_regs_choice": int_regs,
        "int_regs_stim": int_regs,
        "move_regs_choice": move_regs,
        "move_regs_stim": move_regs,
        "stim_regs": ["VISpm", "FRP", "VISal"],
    }

    print(f"weights: {weights}")
    print(f"JSON loss={json_loss:.6f}  ckpt g_i={ckpt['g_i']:.4f}  paper g_i={PAPER['g_i']}")
    print(f"nested model_params g_i={mp_nested.get('g_i')} d_i={mp_nested.get('d_i')}")
    print(f"bps={BPS}  seeds={seeds}  n_grid={len(g_grid)}  HAVE_NUMBA={mf._HAVE_NUMBA}")

    # Warmup numba
    _ = eval_loss(mp_from_base(ckpt, ckpt["g_i"]), mean_data, prior_regions, seeds[0])

    rows_ckpt = slice_base(ckpt, "ckpt_fixed", g_grid, seeds, mean_data, prior_regions)
    paper_base = dict(PAPER)
    paper_base["W_pi"] = ckpt["W_pi"]  # paper omitted W_pi
    paper_base["theta_c"] = ckpt["theta_c"]
    paper_base["theta_d"] = ckpt["theta_d"]
    rows_paper = slice_base(
        paper_base, "paper_fixed", g_grid, seeds, mean_data, prior_regions
    )

    sum_ckpt = summarize(rows_ckpt, PAPER["g_i"], ckpt["g_i"])
    sum_paper = summarize(rows_paper, PAPER["g_i"], ckpt["g_i"])

    report = {
        "weights_json": str(weights),
        "json_loss": json_loss,
        "ckpt_g_i": ckpt["g_i"],
        "paper_g_i": PAPER["g_i"],
        "bps": BPS,
        "seeds": seeds,
        "g_grid": g_grid.tolist(),
        "ckpt_fixed": {"rows": rows_ckpt, "summary": sum_ckpt},
        "paper_fixed": {"rows": rows_paper, "summary": sum_paper},
        "verdict": {
            "ckpt_base_prefers_paper_gi": sum_ckpt["prefers_paper"],
            "ckpt_base_delta_paper_minus_ckpt": sum_ckpt["delta_paper_minus_ckpt"],
            "ckpt_base_argmin_gi": sum_ckpt["min_g_i"],
            "paper_base_argmin_gi": sum_paper["min_g_i"],
            "note": (
                "If ckpt_base prefers paper g_i (delta<0), local refine should pull "
                "g_i down. If prefers ckpt g_i (delta>0), loss design / MC noise is "
                "the issue — consider multi-seed Stage-2 averaging."
            ),
        },
    }

    out = (
        Path(args.out)
        if args.out
        else Path(weights).parent / "gi_slice_report.json"
    )
    out.write_text(json.dumps(report, indent=2))

    print("\n=== Verdict (other params = ckpt) ===")
    print(f"argmin g_i={sum_ckpt['min_g_i']:.2f}  min_total={sum_ckpt['min_total']:.6f}")
    print(f"at paper g_i≈{sum_ckpt['at_paper_g_i']}")
    print(f"at ckpt  g_i≈{sum_ckpt['at_ckpt_g_i']}")
    print(
        f"prefers_paper={sum_ckpt['prefers_paper']}  "
        f"Δ(paper-ckpt)={sum_ckpt['delta_paper_minus_ckpt']:+.6f}"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
