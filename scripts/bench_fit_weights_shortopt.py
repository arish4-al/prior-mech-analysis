"""
Phase 2a baseline: short Stage-2 CMA from a small perturbation of fitted weights.

Compares wall-clock / loss trajectory of the *current* fit_weights pipeline
(numba hard-required) against the historical Stage-2 log for the first N gens.

Usage (iblenv, outside sandbox):
  PYTHONPATH=. python scripts/bench_fit_weights_shortopt.py --n-gens 10 --n-jobs 8

Re-run after pipeline edits; keep --tag / seed fixed to compare apples-to-apples.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from simulate_recovery import resolve_weights_json
from model_functions import (
    pth_res,
    int_regs,
    move_regs,
    model_params,
    save_dir,
    _update_model_params_for_dt,
)
import fit_weights as fw
from fit_weights import (
    reconstruct_theta_log_from_json,
    fit_weights_two_stage_v2,
    _log_bounds_weights_v2,
    disable_realtime_plot,
    loss_history,
    _eval_counter,
)

PARAM_NAMES = [
    "W_ii", "W_pp", "W_mm", "W_is", "W_pi", "W_mi",
    "g_i", "g_m", "d_i", "d_m", "theta_c", "theta_d",
]
# Fitted g_m/d_m sit below log-bounds (~0); freeze them so resume clamp → LOG_ZERO.
FREEZE_NEAR_ZERO = (7, 9)  # g_m, d_m

HIST_FIT_LOG = Path(
    "/Users/ariliu/Downloads/ONE/openalyx.internationalbrainlab.org/models/"
    "weights_run_20251125_182058/fit_log.jsonl"
)


def _ensure_data_links():
    mean_src = Path(pth_res) / "mean_data_results.npy"
    mean_dst = Path.cwd() / "mean_data_results.npy"
    if mean_src.is_file() and not mean_dst.exists():
        mean_dst.symlink_to(mean_src)
    figs = Path(pth_res).parent / "figs"
    for name in ("data_act_block_duringstim.npy", "data_act_block_duringchoice.npy"):
        src, dst = figs / name, Path.cwd() / name
        if src.is_file() and not dst.exists():
            dst.symlink_to(src)


def load_fitted_theta(weights_path: Path):
    meta = json.loads(weights_path.read_text())
    # g_m/d_m may be below positive bounds; bump tiny positives for log, then freeze.
    for k in ("g_m", "d_m"):
        if float(meta["g" if k.startswith("g") else "d"][k]) <= 0:
            meta["g" if k.startswith("g") else "d"][k] = 1e-12
    # reconstruct needs all > 0
    g_m = float(meta["g"]["g_m"])
    d_m = float(meta["d"]["d_m"])
    if g_m <= 0:
        meta["g"]["g_m"] = 1e-12
    if d_m <= 0:
        meta["d"]["d_m"] = 1e-12
    # If still below bound lower edge used by reconstruct, keep as-is; freeze later.
    try:
        theta = reconstruct_theta_log_from_json(meta)
    except ValueError:
        # force positive for log
        meta["g"]["g_m"] = max(float(meta["g"]["g_m"]), 1e-12)
        meta["d"]["d_m"] = max(float(meta["d"]["d_m"]), 1e-12)
        theta = reconstruct_theta_log_from_json(meta)
    return theta, float(meta["loss"]), meta


def perturb_theta(theta_log, train_mask, rng, scale=0.05):
    """Additive Gaussian noise in log-space on free coords; clip to bounds."""
    bounds = _log_bounds_weights_v2()
    Lb = np.array([lo for lo, _ in bounds], float)
    Ub = np.array([hi for _, hi in bounds], float)
    out = np.asarray(theta_log, float).copy()
    noise = rng.normal(scale=scale, size=out.shape)
    free = np.asarray(train_mask, bool)
    out[free] = out[free] + noise[free]
    out[free] = np.minimum(Ub[free], np.maximum(Lb[free], out[free]))
    # frozen stay at LOG_ZERO after resume (~0 actual)
    out[~free] = -30.0
    return out


def parse_cma_gens(log_path: Path, n_gens: int | None = None):
    gens = []
    early_stop = None
    if not log_path.is_file():
        return gens, early_stop
    for line in log_path.read_text().strip().splitlines():
        ev = json.loads(line)
        msg = ev.get("message") or ""
        if ev.get("early_stop") or "early stop" in msg:
            early_stop = {
                "gen": ev.get("gen"),
                "plateau_count": ev.get("plateau_count"),
                "patience": ev.get("patience"),
                "best_overall": ev.get("best_overall"),
                "message": msg,
            }
        if ev.get("stage") == "cma" and "best_in_gen" in ev:
            gens.append(
                {
                    "ts": ev["ts"],
                    "gen": int(ev["gen"]),
                    "best_in_gen": float(ev["best_in_gen"]),
                    "best_overall": float(ev["best_overall"]),
                }
            )
        elif "[CMA-ES] gen" in msg and "best_overall" in ev:
            gens.append(
                {
                    "ts": ev["ts"],
                    "gen": int(ev["gen"]),
                    "best_in_gen": float(ev["best_in_gen"]),
                    "best_overall": float(ev["best_overall"]),
                }
            )
    gens = sorted(gens, key=lambda g: g["gen"])
    if n_gens is not None:
        gens = gens[:n_gens]
    if len(gens) >= 2:
        t0 = datetime.fromisoformat(gens[0]["ts"])
        for i, g in enumerate(gens):
            t = datetime.fromisoformat(g["ts"])
            g["t_from_first_s"] = (t - t0).total_seconds()
            if i == 0:
                g["dt_s"] = None
            else:
                g["dt_s"] = (
                    datetime.fromisoformat(g["ts"])
                    - datetime.fromisoformat(gens[i - 1]["ts"])
                ).total_seconds()
    return gens, early_stop


def summarize_gens(gens):
    if not gens:
        return {}
    dts = [g["dt_s"] for g in gens if g.get("dt_s") is not None]
    return {
        "n_gens": len(gens),
        "best_overall_first": gens[0]["best_overall"],
        "best_overall_last": gens[-1]["best_overall"],
        "wall_to_last_s": gens[-1].get("t_from_first_s"),
        "mean_s_per_gen": float(np.mean(dts)) if dts else None,
        "median_s_per_gen": float(np.median(dts)) if dts else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gens", type=int, default=10)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--popsize", type=int, default=16)
    ap.add_argument("--patience", type=int, default=8,
                    help="CMA early-stop patience after beat-loss gate; 0=off")
    ap.add_argument("--beat-loss", type=float, default=None,
                    help="Arm early-stop only after best_overall < this "
                         "(default: recorded loss from weights JSON)")
    ap.add_argument("--no-beat-gate", action="store_true",
                    help="Disable beat-loss gate (patience alone; can freeze worse plateau)")
    ap.add_argument("--perturb-scale", type=float, default=0.02,
                    help="Gaussian σ in log-space on free params (resampled if loss NaN)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--tag", type=str, default="phase2a_baseline_numba")
    ap.add_argument("--weights-json", type=str, default=None)
    args = ap.parse_args()

    _ensure_data_links()
    weights = Path(args.weights_json) if args.weights_json else resolve_weights_json()
    theta0, json_loss, meta = load_fitted_theta(weights)
    if args.no_beat_gate:
        beat_loss = None
    elif args.beat_loss is not None:
        beat_loss = float(args.beat_loss)
    else:
        beat_loss = float(json_loss)
    print(f"fitted JSON: {weights}")
    print(f"recorded loss={json_loss:.6f}")
    print(
        f"early-stop: patience={args.patience}, "
        f"beat_loss={beat_loss if beat_loss is not None else 'None (ungated)'}"
    )

    train_mask = np.ones(12, dtype=bool)
    for i in FREEZE_NEAR_ZERO:
        train_mask[i] = False
    print(f"freeze indices {FREEZE_NEAR_ZERO}: {[PARAM_NAMES[i] for i in FREEZE_NEAR_ZERO]}")

    mean_data = np.load("mean_data_results.npy", allow_pickle=True).flat[0]
    behavior = np.load(Path(pth_res, "behavior.npy"), allow_pickle=True).flat[0]
    prior_regions = {
        "int_regs_choice": int_regs,
        "int_regs_stim": int_regs,
        "move_regs_choice": move_regs,
        "move_regs_stim": move_regs,
        "stim_regs": ["VISpm", "FRP", "VISal"],
    }

    # Align front-end + dt with the fitted JSON (same object as fit_weights.model_params).
    for k, v in meta.get("model_params", {}).items():
        if isinstance(v, (int, float, np.floating)):
            model_params[k] = float(v)
    model_params["direct_offset"] = False
    model_params["dt"] = 2.0
    _update_model_params_for_dt(model_params, 2.0)
    import model_functions as mf
    mf.blocks_per_session = 5
    if hasattr(fw, "blocks_per_session"):
        fw.blocks_per_session = 5
    fw._STIMULI_BUNDLE_CACHE = None

    disable_realtime_plot()
    loss_history.clear()
    _eval_counter["n"] = 0

    det_stim_seed = int(args.seed) + 100003

    def eval_bps20(theta_log, debug=False):
        fw._STIMULI_BUNDLE_CACHE = None
        return float(
            fw.loss_weights_core_v2(
                theta_log,
                mean_data,
                prior_regions,
                behavior,
                model_type="data",
                plot=False,
                debug=debug,
                verbose=False,
                blocks_per_session_override=20,
                stim_rng=np.random.RandomState(det_stim_seed),
            )
        )

    # Sanity: fitted + freeze only (no perturb) must be finite.
    theta_freeze = theta0.copy()
    theta_freeze[~train_mask] = -30.0
    L_ref = eval_bps20(theta_freeze, debug=True)
    print(f"ref loss (fitted, freeze g_m/d_m, bps=20): {L_ref:.6f}")
    if not np.isfinite(L_ref) or L_ref >= 1e11:
        raise RuntimeError(f"Reference fitted loss invalid ({L_ref}); check data links.")

    # Resample log-space perturbation until Stage-2 loss is finite.
    theta_pert = None
    start_loss = None
    t_start_eval = None
    for attempt in range(32):
        rng = np.random.RandomState(args.seed + attempt)
        cand = perturb_theta(theta0, train_mask, rng, scale=args.perturb_scale)
        t0 = time.perf_counter()
        L = eval_bps20(cand, debug=(attempt == 0))
        dt_eval = time.perf_counter() - t0
        if np.isfinite(L) and L < 1e11:
            theta_pert, start_loss, t_start_eval = cand, L, dt_eval
            print(
                f"perturb ok attempt={attempt} scale={args.perturb_scale} "
                f"||Δlog||_free={np.linalg.norm((cand - theta0)[train_mask]):.4f} "
                f"loss={L:.6f} wall={dt_eval:.2f}s"
            )
            break
        print(f"perturb reject attempt={attempt} loss={L}")
    if theta_pert is None:
        raise RuntimeError(
            "Could not find a finite-loss perturbation; try --perturb-scale 0.02"
        )

    CMA_stds2 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1])
    run_root = Path(save_dir) / f"weights_run_{args.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    fw._ensure_run_dirs(run_dir=run_root)
    print(f"run_dir: {run_root}")

    # Sidecar so resume uses this loss for Stage-2 gating (avoids a bps=5 recompute
    # that can disagree with the Stage-2 start loss / hit the discard threshold).
    resume_sidecar = run_root / "resume_perturbed.json"
    resume_sidecar.write_text(
        json.dumps(
            {
                "loss": float(start_loss),
                "frozen_idx": [int(i) for i in FREEZE_NEAR_ZERO],
                "train_mask": train_mask.tolist(),
                "theta_log": theta_pert.tolist(),
                "note": "phase2a short-opt start (perturbed fitted)",
            },
            indent=2,
        )
    )

    wall0 = time.perf_counter()
    best = fit_weights_two_stage_v2(
        mean_data,
        prior_regions,
        behavior,
        model_type="data",
        random_state=args.seed,
        top_k=0,
        sobol_count=0,
        global_method_stage1="cma",  # skip DE post-stage1 scoring when resuming
        global_method_stage2="cma",
        cma_sigma_scale=0.25,
        cma_sigma_scale_stage2=0.02,
        cma_opts_stage2={
            "popsize": int(args.popsize),
            "tolfun": 5e-4,
            "tolx": 5e-5,
            "CMA_stds": list(CMA_stds2[train_mask]),
            "CMA_diagonal": False,
        },
        de1_maxiter=0,
        de2_maxiter=int(args.n_gens),
        train_mask=train_mask,
        resume_from="de2",
        resume_theta_log=theta_pert,
        resume_path=str(resume_sidecar),
        blocks_per_session_stage2=20,
        n_jobs=int(args.n_jobs),
        parallel_backend="loky",
        deterministic_stage2=True,
        L_threshold=2.0,
        cma_early_stop_patience=int(args.patience),
        cma_early_stop_beat_loss=beat_loss,
    )
    wall_total = time.perf_counter() - wall0

    log_path = Path(best.get("log_path") or (run_root / "fit_log.jsonl"))
    gens_now, early_stop = parse_cma_gens(log_path, n_gens=None)
    gens_hist, _ = parse_cma_gens(HIST_FIT_LOG, n_gens=int(args.n_gens))
    sum_now = summarize_gens(gens_now)
    sum_hist = summarize_gens(gens_hist)

    report = {
        "tag": args.tag,
        "weights_json": str(weights),
        "json_loss": json_loss,
        "perturb_scale": args.perturb_scale,
        "seed": args.seed,
        "n_gens": args.n_gens,
        "popsize": args.popsize,
        "n_jobs": args.n_jobs,
        "cma_early_stop_patience": int(args.patience),
        "cma_early_stop_beat_loss": beat_loss,
        "early_stop": early_stop,
        "freeze": [PARAM_NAMES[i] for i in FREEZE_NEAR_ZERO],
        "start_loss_bps20": start_loss,
        "start_eval_wall_s": t_start_eval,
        "final_loss": float(best["loss"]),
        "wall_total_s": wall_total,
        "run_dir": str(run_root),
        "backend": "numba",
        "current": {"gens": gens_now, "summary": sum_now},
        "historical_first_N_gens": {"gens": gens_hist, "summary": sum_hist},
        "note": (
            "Historical log is Stage-2 CMA from a different start (stage1_loss≈0.80), "
            "n_jobs=16, popsize=16. Current run starts from perturbed fitted θ. "
            "Compare s/gen and parallel efficiency; loss trajectories are not matched starts."
        ),
    }
    out = run_root / "phase2a_shortopt_report.json"
    out.write_text(json.dumps(report, indent=2))
    print("\n=== Phase 2a short-opt report ===")
    print(f"start_loss={start_loss:.6f}  final_loss={best['loss']:.6f}  wall={wall_total:.1f}s")
    if early_stop:
        print(f"early_stop: {early_stop}")
    print(f"current  summary: {sum_now}")
    print(f"history  summary (first {args.n_gens} gens): {sum_hist}")
    if sum_now.get("mean_s_per_gen") and sum_hist.get("mean_s_per_gen"):
        print(
            f"mean s/gen  current={sum_now['mean_s_per_gen']:.1f}  "
            f"hist={sum_hist['mean_s_per_gen']:.1f}  "
            f"ratio={sum_now['mean_s_per_gen']/sum_hist['mean_s_per_gen']:.2f}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
