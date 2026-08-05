"""
Phase 2a/2c: compare fit_weights pipeline variants under an equal short budget.

Answers three questions the plan raises:
  1. Is DE Stage 1 needed?           -> cold-start `de_cma` vs `cma_only`
  2. Is post-CMA local refine worth it? -> warm `cma_only_local` vs `cma_only_nolocal`
  3. loky vs threading for CMA evals?   -> warm `..._loky` vs `..._threading`

Metrics per variant: final_loss, wall_s, mean_s_per_gen, recovered g_i (paper 163)
and d_i. Warm variants start from a fixed log-space perturbation of the fitted
weights (same procedure as bench_fit_weights_shortopt.py). Cold variants start
from a random-in-bounds init (reproducible via --seed) with L_threshold raised
so Stage 2 always runs.

Usage (iblenv, OUTSIDE the Cursor sandbox):
  PYTHONPATH=. python scripts/bench_fit_pipeline_variants.py --n-jobs 8
  PYTHONPATH=. python scripts/bench_fit_pipeline_variants.py --n-jobs 8 --include-cold

Keep --seed fixed to compare against bench_fit_weights_shortopt.py reports.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Skip ONE construction in the fit path (avoids the loky/ONE params race).
try:
    import _one_bypass  # noqa: F401
except ImportError:
    from scripts import _one_bypass  # noqa: F401

from model_functions import (
    pth_res,
    int_regs,
    move_regs,
    model_params,
    save_dir,
    _update_model_params_for_dt,
)
from simulate_recovery import resolve_weights_json
import fit_weights as fw
from fit_weights import (
    fit_weights_two_stage_v2,
    disable_realtime_plot,
    loss_history,
    _eval_counter,
)

# Reuse the short-opt helpers (guarded __main__, safe to import).
try:
    from bench_fit_weights_shortopt import (
        PARAM_NAMES,
        FREEZE_NEAR_ZERO,
        _ensure_data_links,
        load_fitted_theta,
        perturb_theta,
        parse_cma_gens,
        summarize_gens,
    )
except ImportError:  # when run without scripts/ on sys.path[0]
    from scripts.bench_fit_weights_shortopt import (  # type: ignore
        PARAM_NAMES,
        FREEZE_NEAR_ZERO,
        _ensure_data_links,
        load_fitted_theta,
        perturb_theta,
        parse_cma_gens,
        summarize_gens,
    )

# Stage-2 per-dim CMA step sizes (matches bench_fit_weights_shortopt / __main__).
CMA_STDS2 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1])


def _cma_opts_stage2(popsize, train_mask):
    return {
        "popsize": int(popsize),
        "tolfun": 5e-4,
        "tolx": 5e-5,
        "CMA_stds": list(CMA_STDS2[train_mask]),
        "CMA_diagonal": False,
    }


def _extract(best):
    g_i, g_m = best["g"]
    d_i, d_m = best["d"]
    return {
        "final_loss": float(best["loss"]),
        "g_i": float(g_i),
        "d_i": float(d_i),
        "run_dir": str(best.get("run_dir")),
    }


def run_variant(label, common, kwargs, seed, run_parent):
    """Run one variant in its own run dir; return metrics dict."""
    run_dir = run_parent / label
    fw._ensure_run_dirs(run_dir=run_dir)
    fw._STIMULI_BUNDLE_CACHE = None
    loss_history.clear()
    _eval_counter["n"] = 0
    print(f"\n{'='*70}\n=== variant: {label} ===\n{'='*70}")
    t0 = time.perf_counter()
    best = fit_weights_two_stage_v2(
        common["mean_data"],
        common["prior_regions"],
        common["behavior"],
        model_type="data",
        random_state=seed,
        **kwargs,
    )
    wall = time.perf_counter() - t0
    log_path = Path(best.get("log_path") or (run_dir / "fit_log.jsonl"))
    gens, early = parse_cma_gens(log_path, n_gens=None)
    summ = summarize_gens(gens)
    out = _extract(best)
    out.update(
        {
            "wall_s": wall,
            "n_cma_gens": summ.get("n_gens"),
            "mean_s_per_gen": summ.get("mean_s_per_gen"),
            "best_overall_last": summ.get("best_overall_last"),
            "early_stop": early,
            "backend": kwargs.get("parallel_backend"),
            "local_refine": kwargs.get("local_refine_after_cma"),
            "stage1": kwargs.get("global_method_stage1"),
            "de1_maxiter": kwargs.get("de1_maxiter"),
        }
    )
    print(
        f"[{label}] final_loss={out['final_loss']:.6f} g_i={out['g_i']:.1f} "
        f"d_i={out['d_i']:.2f} wall={wall:.1f}s s/gen={out['mean_s_per_gen']}"
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gens", type=int, default=10, help="Stage-2 CMA gens per variant")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--popsize", type=int, default=16)
    ap.add_argument("--patience", type=int, default=8, help="CMA early-stop patience; 0=off")
    ap.add_argument("--perturb-scale", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--tag", type=str, default="phase2a_variants")
    ap.add_argument("--weights-json", type=str, default=None)
    ap.add_argument("--include-cold", action="store_true",
                    help="Also run cold-start de_cma vs cma_only (DE-needed test)")
    ap.add_argument("--cold-de1-iters", type=int, default=12,
                    help="DE/CMA Stage-1 iterations for cold-start variants")
    ap.add_argument("--variants", type=str, default=None,
                    help="Comma list to restrict warm variants (default: all)")
    args = ap.parse_args()

    _ensure_data_links()
    weights = Path(args.weights_json) if args.weights_json else resolve_weights_json()
    theta0, json_loss, meta = load_fitted_theta(weights)
    beat_loss = float(json_loss)
    print(f"fitted JSON: {weights}  recorded loss={json_loss:.6f}")

    train_mask = np.ones(12, dtype=bool)
    for i in FREEZE_NEAR_ZERO:
        train_mask[i] = False
    print(f"freeze {FREEZE_NEAR_ZERO}: {[PARAM_NAMES[i] for i in FREEZE_NEAR_ZERO]}")

    mean_data = np.load("mean_data_results.npy", allow_pickle=True).flat[0]
    behavior = np.load(Path(pth_res, "behavior.npy"), allow_pickle=True).flat[0]
    prior_regions = {
        "int_regs_choice": int_regs,
        "int_regs_stim": int_regs,
        "move_regs_choice": move_regs,
        "move_regs_stim": move_regs,
        "stim_regs": ["VISpm", "FRP", "VISal"],
    }

    # Align front-end + dt with the fitted JSON (same as short-opt bench).
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

    det_stim_seed = int(args.seed) + 100003

    def eval_bps20(theta_log, debug=False):
        fw._STIMULI_BUNDLE_CACHE = None
        return float(
            fw.loss_weights_core_v2(
                theta_log, mean_data, prior_regions, behavior,
                model_type="data", plot=False, debug=debug, verbose=False,
                blocks_per_session_override=20,
                stim_rng=np.random.RandomState(det_stim_seed),
            )
        )

    # ---- fixed warm perturbation (same recipe as short-opt bench) ----
    theta_pert = None
    start_loss = None
    for attempt in range(32):
        rng = np.random.RandomState(args.seed + attempt)
        cand = perturb_theta(theta0, train_mask, rng, scale=args.perturb_scale)
        L = eval_bps20(cand, debug=(attempt == 0))
        if np.isfinite(L) and L < 1e11:
            theta_pert, start_loss = cand, L
            print(f"perturb ok attempt={attempt} loss={L:.6f}")
            break
        print(f"perturb reject attempt={attempt} loss={L}")
    if theta_pert is None:
        raise RuntimeError("Could not find a finite-loss perturbation; try --perturb-scale 0.02")

    run_parent = Path(save_dir) / f"weights_run_{args.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_parent.mkdir(parents=True, exist_ok=True)
    print(f"run_parent: {run_parent}")

    # Sidecar for warm resume gating (loss used by Stage-2 threshold logic).
    resume_sidecar = run_parent / "resume_perturbed.json"
    resume_sidecar.write_text(
        json.dumps(
            {
                "loss": float(start_loss),
                "frozen_idx": [int(i) for i in FREEZE_NEAR_ZERO],
                "train_mask": train_mask.tolist(),
                "theta_log": theta_pert.tolist(),
                "note": "phase2a variants warm start (perturbed fitted)",
            },
            indent=2,
        )
    )

    common = {"mean_data": mean_data, "prior_regions": prior_regions, "behavior": behavior}

    def warm_kwargs(local, backend):
        return dict(
            top_k=0,
            sobol_count=0,
            global_method_stage1="cma",   # skip DE post-stage1 scoring on resume
            global_method_stage2="cma",
            cma_sigma_scale=0.25,
            cma_sigma_scale_stage2=0.02,
            cma_opts_stage2=_cma_opts_stage2(args.popsize, train_mask),
            de1_maxiter=0,
            de2_maxiter=int(args.n_gens),
            train_mask=train_mask,
            resume_from="de2",
            resume_theta_log=theta_pert,
            resume_path=str(resume_sidecar),
            blocks_per_session_stage2=20,
            n_jobs=int(args.n_jobs),
            parallel_backend=backend,
            deterministic_stage2=True,
            L_threshold=2.0,
            cma_early_stop_patience=int(args.patience),
            cma_early_stop_beat_loss=beat_loss,
            local_refine_after_cma=bool(local),
            local_refine_patience=8,
        )

    warm_registry = {
        # backend comparison (no local refine so CMA-backend timing is isolated)
        "cma_only_nolocal_loky": warm_kwargs(local=False, backend="loky"),
        "cma_only_nolocal_threading": warm_kwargs(local=False, backend="threading"),
        # local-refine value (loky), compare against cma_only_nolocal_loky
        "cma_only_local_loky": warm_kwargs(local=True, backend="loky"),
    }

    if args.variants:
        keep = {s.strip() for s in args.variants.split(",") if s.strip()}
        warm_registry = {k: v for k, v in warm_registry.items() if k in keep}

    results = {}
    for label, kwargs in warm_registry.items():
        results[label] = run_variant(label, common, kwargs, args.seed, run_parent)

    # ---- optional cold-start: is DE Stage 1 needed? ----
    if args.include_cold:
        def cold_kwargs(stage1, local):
            return dict(
                top_k=0,
                sobol_count=(8 if stage1 == "de" else 0),
                de_popsize=8,
                global_method_stage1=stage1,
                global_method_stage2="cma",
                cma_sigma_scale=0.25,
                cma_sigma_scale_stage2=0.02,
                cma_opts_stage1=_cma_opts_stage2(args.popsize, train_mask) if stage1 == "cma" else None,
                cma_opts_stage2=_cma_opts_stage2(args.popsize, train_mask),
                de1_maxiter=int(args.cold_de1_iters),
                de2_maxiter=int(args.n_gens),
                train_mask=train_mask,
                resume_from="none",
                blocks_per_session_stage2=20,
                n_jobs=int(args.n_jobs),
                parallel_backend="loky",  # loky ~5x faster than threading for CMA here
                deterministic_stage2=True,
                # Raise threshold so a random-init Stage 1 always proceeds to Stage 2.
                L_threshold=1.0e9,
                cma_early_stop_patience=int(args.patience),
                cma_early_stop_beat_loss=beat_loss,
                local_refine_after_cma=bool(local),
                local_refine_patience=8,
            )

        for label, kwargs in {
            "cold_de_cma_local": cold_kwargs(stage1="de", local=True),
            "cold_cma_only_local": cold_kwargs(stage1="cma", local=True),
        }.items():
            results[label] = run_variant(label, common, kwargs, args.seed, run_parent)

    report = {
        "tag": args.tag,
        "weights_json": str(weights),
        "json_loss": json_loss,
        "seed": args.seed,
        "n_gens": args.n_gens,
        "popsize": args.popsize,
        "n_jobs": args.n_jobs,
        "patience": args.patience,
        "perturb_scale": args.perturb_scale,
        "start_loss_bps20": start_loss,
        "freeze": [PARAM_NAMES[i] for i in FREEZE_NEAR_ZERO],
        "run_parent": str(run_parent),
        "variants": results,
        "note": (
            "Warm variants resume from a perturbed fitted theta (Stage-2 CMA only). "
            "Cold variants random-init with L_threshold raised so Stage 2 runs. "
            "Compare mean_s_per_gen (loky vs threading), final_loss (local on/off), "
            "and g_i vs paper 163."
        ),
    }
    out = run_parent / "bench_pipeline_variants_report.json"
    out.write_text(json.dumps(report, indent=2))

    print("\n=== pipeline variants summary ===")
    hdr = f"{'variant':30s} {'loss':>8s} {'g_i':>7s} {'d_i':>6s} {'wall_s':>8s} {'s/gen':>7s}"
    print(hdr)
    for label, r in results.items():
        spg = r.get("mean_s_per_gen")
        spg_s = f"{spg:.1f}" if isinstance(spg, (int, float)) else "  -  "
        print(
            f"{label:30s} {r['final_loss']:8.4f} {r['g_i']:7.1f} {r['d_i']:6.2f} "
            f"{r['wall_s']:8.1f} {spg_s:>7s}"
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
