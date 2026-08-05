"""
Smoke: post-CMA local refine from WEIGHTS_REL checkpoint (skip Stage-2 CMA).

Usage (iblenv, outside sandbox):
  PYTHONPATH=. python scripts/bench_fit_local_after_cma.py --local-maxiter 15
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Skip ONE construction in the fit path (avoids the loky/ONE params race in FD workers).
try:
    import _one_bypass  # noqa: F401
except ImportError:
    from scripts import _one_bypass  # noqa: F401

from simulate_recovery import resolve_weights_json
from model_functions import pth_res, int_regs, move_regs, save_dir, model_params
import fit_weights as fw
from fit_weights import reconstruct_theta_log_from_json, fit_weights_two_stage_v2


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-maxiter", type=int, default=40)
    ap.add_argument("--n-jobs", type=int, default=8,
                    help="Parallel workers for optional CMA-polish candidate evals")
    ap.add_argument("--patience", type=int, default=8,
                    help="Stop polish after this many iters/gens with no best improve; 0=off")
    ap.add_argument("--n-stim-seeds", type=int, default=3,
                    help="number of fixed stim bundles for polish train (default 3; "
                         "fit_weights default aggregate=sample → 1-of-K per eval)")
    ap.add_argument("--seed", type=int, default=123, help="TRAINING seed (refine bundle)")
    ap.add_argument("--eval-seed", type=int, default=None,
                    help="HELD-OUT seed for in-polish gate + final score; default = seed+7777")
    ap.add_argument("--backend", choices=["loky", "threading"], default="loky",
                    help="joblib backend (CMA polish uses threading for closures)")
    ap.add_argument("--local-refine-idx", type=str, default="prior",
                    help="prior ([6,8,10,11], default) | active (all 12) | comma list")
    ap.add_argument("--method", choices=["powell", "cma"], default="powell",
                    help="polish optimizer (default powell; lbfgs removed)")
    ap.add_argument("--cma-sigma", type=float, default=0.05,
                    help="initial sigma (fraction of bound span) for --method cma")
    ap.add_argument("--powell-fallback", action="store_true",
                    help="if method=cma, fall back to Powell when CMA doesn't beat start loss")
    ap.add_argument("--paper", action="store_true",
                    help="override params with the paper-reported values (g_i=163 etc.) before eval")
    ap.add_argument("--eval-only", action="store_true",
                    help="just evaluate the (possibly --paper) params on the deterministic "
                         "bundle and print the loss; no real refinement")
    ap.add_argument("--params-json", type=str, default=None,
                    help="path to a saved weights JSON to load instead of the 0.404 ckpt "
                         "(used with --eval-only to score already-fit params held-out)")
    args = ap.parse_args()

    # Paper-reported params (i->c in the paper), mapped to the 12-d log vector layout
    # [W_ii,W_pp,W_mm,W_is,W_pi,W_mi, g_i,g_m,d_i,d_m, theta_c,theta_d].
    # NOTE: W_pi (index 4) is load-bearing — zeroing it blows the loss to ~2.7. Paper
    # value W_pi=1.6e-5. g_m/d_m "negligible" per paper.
    PAPER = {0: 0.43, 1: 0.496, 2: 0.27, 3: 0.17, 4: 1.6e-5, 5: 0.50,
             6: 163.0, 7: 1e-12, 8: 21.4, 9: 1e-12, 10: 0.76, 11: 0.40}

    def _parse_lri(spec):
        s = str(spec).strip().lower()
        if s in ("prior", "focused", ""):
            return None  # fit_weights default = [6,8,10,11]
        if s in ("active", "all"):
            return list(range(12))
        return [int(t) for t in s.replace(" ", "").split(",") if t != ""]

    _ensure_data_links()
    weights = args.params_json if args.params_json else resolve_weights_json()
    meta = json.loads(Path(weights).read_text())
    # bump near-zero for log
    for k in ("g_m", "d_m"):
        key = "g" if k.startswith("g") else "d"
        if float(meta[key][k]) <= 0:
            meta[key][k] = 1e-12
    theta = reconstruct_theta_log_from_json(meta)
    if args.paper:
        for i, v in PAPER.items():
            theta[i] = float(np.log(v))
        print(f"[paper] overrode params -> {PAPER}")
    g_i0 = float(np.exp(theta[6]))
    d_i0 = float(np.exp(theta[8]))
    train_seed = int(args.seed)
    eval_seed = (int(args.eval_seed) if args.eval_seed is not None
                 else train_seed + 7777)
    print(f"ckpt: {weights}")
    print(f"start g_i={g_i0:.4f} d_i={d_i0:.4f} json_loss={meta['loss']:.6f} "
          f"paper={args.paper} train_seed={train_seed} eval_seed={eval_seed} "
          f"n_stim={args.n_stim_seeds}")

    mean_data = np.load("mean_data_results.npy", allow_pickle=True).flat[0]
    behavior = np.load(Path(pth_res, "behavior.npy"), allow_pickle=True).flat[0]
    prior_regions = {
        "int_regs_choice": int_regs,
        "int_regs_stim": int_regs,
        "move_regs_choice": move_regs,
        "move_regs_stim": move_regs,
        "stim_regs": ["VISpm", "FRP", "VISal"],
    }

    def eval_theta_on_seed(theta_log, seed, bps=20):
        """Exact Stage-2 loss on the deterministic bundle for `seed` (matches the internal
        stage2_stim_seed = random_state + 100003)."""
        bundle = fw.build_stimuli_bundle(
            bps, stim_rng=np.random.RandomState(int(seed) + 100003), **model_params)
        return float(fw._safe_loss_weights_v2(
            np.asarray(theta_log, float), mean_data, prior_regions, behavior,
            model_type="data", plot=False, debug=False,
            blocks_per_session_override=bps, verbose=False, stimuli_bundle=bundle))

    # --eval-only: no refinement — just score the (possibly --paper) params in-sample
    # (train_seed) and held-out (eval_seed).
    if args.eval_only:
        l_train = eval_theta_on_seed(theta, train_seed)
        l_eval = eval_theta_on_seed(theta, eval_seed)
        tag = "paper" if args.paper else "ckpt"
        print(f"done method={tag} refine_idx=eval "
              f"loss_train(seed={train_seed})={l_train:.6f} "
              f"loss_heldout(seed={eval_seed})={l_eval:.6f} "
              f"g_i={g_i0:.4f} d_i={d_i0:.4f}")
        return

    train_mask = np.ones(12, dtype=bool)
    run_root = (
        Path(save_dir)
        / f"weights_run_local_after_cma_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    fw._ensure_run_dirs(run_dir=run_root)
    sidecar = run_root / "resume_ckpt.json"
    sidecar.write_text(
        json.dumps(
            {
                "loss": float(meta["loss"]),
                "train_mask": train_mask.tolist(),
                "theta_log": theta.tolist(),
            },
            indent=2,
        )
    )

    t0 = time.perf_counter()
    best = fit_weights_two_stage_v2(
        mean_data,
        prior_regions,
        behavior,
        model_type="data",
        random_state=train_seed,
        top_k=0,
        sobol_count=0,
        global_method_stage1="cma",
        global_method_stage2="cma",
        de1_maxiter=0,
        de2_maxiter=0,  # skip CMA; go straight to local-after-cma
        train_mask=train_mask,
        resume_from="de2",
        resume_theta_log=theta,
        resume_path=str(sidecar),
        blocks_per_session_stage2=20,
        n_jobs=int(args.n_jobs),
        parallel_backend=args.backend,
        deterministic_stage2=True,
        L_threshold=2.0,
        local_refine_after_cma=True,
        local_refine_idx=_parse_lri(args.local_refine_idx),
        local_refine_method=args.method,
        local_refine_cma_sigma=float(args.cma_sigma),
        local_refine_use_powell=bool(args.powell_fallback),
        local_refine_patience=int(args.patience),
        local_maxiter=int(args.local_maxiter),
        stage2_n_stim_seeds=int(args.n_stim_seeds),
        val_stim_seed=eval_seed,
    )
    dt = time.perf_counter() - t0
    g = best["g"]
    d = best["d"]
    theta_final = np.asarray(best["theta_log"], float)
    # In-sample sanity (should match best['loss']) + held-out generalization score.
    l_train = eval_theta_on_seed(theta_final, train_seed)
    l_eval = eval_theta_on_seed(theta_final, eval_seed)
    print(
        f"done method={args.method} refine_idx={args.local_refine_idx} "
        f"wall={dt:.1f}s loss_train(seed={train_seed})={l_train:.6f} "
        f"loss_heldout(seed={eval_seed})={l_eval:.6f} "
        f"g_i {g_i0:.4f}→{g[0]:.4f}  d_i {d_i0:.4f}→{d[0]:.4f}"
    )
    print(f"run_dir={best.get('run_dir')}")


if __name__ == "__main__":
    main()
