"""
CLI driver for weights fitting (retinal frozen) — local or ORCD cluster.

Refactors the guarded ``fit_weights.py`` ``__main__`` block into a callable
``main(argv)`` with flags for model variant, seed, parallelism, and pipeline, so
a variant x seed sweep can be launched from a SLURM submitter (see
scripts/run_fit_weights_slurm.sh + scripts/submit_fit_weights_sharded.sh).

Model variant = (mtype label, frozen params). ``--freeze`` lists 0-based indices
of the 12-d vector to hold fixed (excluded from optimization):

    0 W_ii  1 W_pp  2 W_mm  3 W_is  4 W_pi  5 W_mi
    6 g_i   7 g_m   8 d_i   9 d_m   10 theta_c  11 theta_d

e.g. ``--mtype none --freeze ""``  or  ``--mtype gain --freeze 7,9`` (g_m,d_m ~0).

Each (mtype, mask, seed) gets its own deterministic run dir:
    <save_dir>/weights_run_fw[_<tag>]_<mtype>_mask<slug>_s<seed>/
Re-running the same command restarts that fit: with ``--resume auto`` (default) it
loads the *latest mtime* checkpoint (rolling Stage-2 / polish / final / de1) and
continues appropriately. A successful fit writes ``FIT_DONE`` (skipped unless
``--force``); Stage-1 failure writes ``FIT_FAILED`` without ``FIT_DONE``.

Pipelines (journals/simulation_fit_speedups.md, Phase 2a/2c):
  de_cma_local  DE Stage 1 -> CMA Stage 2 -> local refine (default; cold start)
  de_cma        DE Stage 1 -> CMA Stage 2 (no local refine)
  cma_only      CMA Stage 2 only from a warm start (--resume-json); skips DE Stage 1

Usage (conda env `ibl`/`iblenv`, OUTSIDE the Cursor sandbox):
  PYTHONPATH=. python scripts/run_fit_weights.py --mtype none  --freeze ""    --seed 56 --n-jobs 16
  PYTHONPATH=. python scripts/run_fit_weights.py --mtype gain  --freeze 7,9   --seed 56 --n-jobs 16
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Must run before importing model_functions: skip ONE construction in the fit path
# (avoids the loky/ONE params race) by resolving cache_dir once and setting env.
try:
    import _one_bypass  # noqa: F401
except ImportError:
    from scripts import _one_bypass  # noqa: F401

try:
    from _fit_data import ensure_fit_data_links, load_validated_mean_data
except ImportError:
    from scripts._fit_data import ensure_fit_data_links, load_validated_mean_data

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
    fit_weights_two_stage_v2,
    reconstruct_theta_log_from_json,
    disable_realtime_plot,
    loss_history,
    _eval_counter,
    _save_params_v2,
)

# Per-dim CMA step sizes (g_i / g_m get larger steps); matches __main__.
CMA_STDS = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1])
PARAM_NAMES = [
    "W_ii", "W_pp", "W_mm", "W_is", "W_pi", "W_mi",
    "g_i", "g_m", "d_i", "d_m", "theta_c", "theta_d",
]


def _default_n_jobs():
    for env in ("SLURM_CPUS_PER_TASK", "JOBLIB_N_JOBS"):
        v = os.environ.get(env)
        if v and v.isdigit() and int(v) > 0:
            return int(v)
    return 8


def _parse_freeze(spec):
    if not spec:
        return []
    return sorted({int(s) for s in str(spec).replace(" ", "").split(",") if s != ""})


def _mask_slug(freeze_idx):
    return "none" if not freeze_idx else "-".join(str(i) for i in freeze_idx)


def _find_resume(run_dir: Path):
    """Return (theta_log_full, loss, model_params_dict, source, kind) from the *latest*
    in-folder checkpoint (by mtime), or None.

    kind ∈ {'stage2','polish','de1','other'} steers resume_from so a polished final is
    not discarded in favour of an older Stage-2 rolling file, and a Stage-1 DE ckpt can
    warm-start Stage 2 if Stage 2 never wrote a rolling file.
    """
    candidates = []
    # Note: resume_start.json is a driver sidecar written *at restart time* — exclude
    # it so its fresh mtime cannot outrank a real Stage-2 / polish checkpoint.
    last = run_dir / "weights_stage2_last.json"
    if last.is_file():
        candidates.append(last)
    for pat in (
        "weights_2stagelocalrefine_*.json",
        "weights_final_*.json",
        "weights_localrefine_*.json",
        # Exclude weights_v2_*.json: opportunistic mid-fit dumps (loss<0.4), not
        # selection incumbents — their mtime can outrank the held-out rolling ckpt.
    ):
        candidates += list(run_dir.glob(pat))
    ckpt_dir = run_dir / "ckpts"
    if ckpt_dir.is_dir():
        candidates += list(ckpt_dir.glob("v2_de1*.json"))
        candidates += list(ckpt_dir.glob("v2_de2*.json"))

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    for c in candidates:
        try:
            meta = json.loads(c.read_text())
            if "theta_log" in meta:
                theta = np.array(meta["theta_log"], float)
                # Stage-1 DE ckpts store active coords only — expand via train_mask/fit_idx.
                if theta.size != 12:
                    theta_full, _ = fw.load_theta_from_ckpt(c)
                    theta = np.asarray(theta_full, float)
            else:
                for k in ("g_m", "d_m"):
                    grp = "g" if k.startswith("g") else "d"
                    if float(meta[grp][k]) <= 0:
                        meta[grp][k] = 1e-12
                theta = reconstruct_theta_log_from_json(meta)
            if theta.size != 12 or not np.all(np.isfinite(theta)):
                continue
            name = c.name
            sel = meta.get("selection")
            # Final/polish snapshots may live in weights_stage2_last (selection=final)
            # after a completed polish — treat those as polish so we don't re-CMA.
            if sel == "final" or any(tag in name for tag in (
                "2stagelocalrefine", "localrefine", "weights_final",
            )):
                kind = "polish"
            elif "stage2_last" in name or name.startswith("v2_de2") or sel in (
                "held_out", "train",
            ):
                kind = "stage2"
            elif name.startswith("v2_de1"):
                kind = "de1"
            else:
                kind = "other"
            return (
                theta,
                float(meta.get("loss", 1.0)),
                meta.get("model_params", {}),
                name,
                kind,
            )
        except Exception:
            continue
    return None


def _resume_from_kind(kind):
    """Map checkpoint kind → fit_weights_two_stage_v2 resume_from."""
    if kind == "polish":
        return "local"   # skip Stage-2 CMA; re-enter polish / finalize
    if kind == "de1":
        return "de1"     # treat as Stage-1 best; enter Stage 2
    return "de2"         # stage2 / other: warm-restart Stage-2 CMA


def build_args(argv=None):
    ap = argparse.ArgumentParser(description="Fit network weights (retinal frozen).")
    ap.add_argument("--mtype", type=str, default="none",
                    help="model variant label (folder name); e.g. none, gain")
    ap.add_argument("--freeze", type=str, default="",
                    help="comma indices to freeze (0..11); e.g. '7,9' for g_m,d_m")
    ap.add_argument("--seed", type=int, default=0, help="random_state / multi-start seed")
    ap.add_argument("--n-jobs", type=int, default=_default_n_jobs(),
                    help="parallel workers (default: SLURM_CPUS_PER_TASK or 8)")
    ap.add_argument("--pipeline", choices=["de_cma_local", "de_cma", "cma_only"],
                    default="de_cma_local")
    ap.add_argument("--out-tag", type=str, default="",
                    help="optional campaign tag prepended to the run-dir name")
    ap.add_argument("--resume", choices=["auto", "off"], default="auto",
                    help="auto: continue from in-folder checkpoint if present")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if a FIT_DONE marker exists")
    ap.add_argument("--backend", choices=["loky", "threading"], default="loky",
                    help="joblib backend for CMA candidate evals. loky is ~5x faster "
                         "than threading here (numba holds the GIL); see "
                         "journals/simulation_fit_speedups.md 2026-08-03l.")
    ap.add_argument("--bps-stage1", type=int, default=5)
    ap.add_argument("--bps-stage2", type=int, default=20)
    ap.add_argument("--de1-maxiter", type=int, default=40, help="DE Stage-1 iterations")
    ap.add_argument("--de2-maxiter", type=int, default=40, help="CMA Stage-2 generations")
    ap.add_argument("--de-popsize", type=int, default=8, help="DE Stage-1 popsize factor")
    ap.add_argument("--popsize", type=int, default=16, help="CMA Stage-2 popsize")
    ap.add_argument("--sobol-count", type=int, default=8)
    ap.add_argument("--patience", type=int, default=8, help="CMA early-stop patience; 0=off")
    ap.add_argument("--beat-loss", type=float, default=0.4044,
                    help="arm early-stop only after best_overall < this (neg = disable gate)")
    ap.add_argument("--resume-json", type=str, default=None,
                    help="external warm-start weights JSON (required for cma_only fresh)")
    ap.add_argument("--stage2-n-stim-seeds", type=int, default=3,
                    help="number of fixed Stage-2 stim bundles (default 3). How they enter "
                         "the train loss is controlled by --stage2-stim-aggregate.")
    ap.add_argument("--stage2-stim-aggregate", choices=["sample", "mean"], default="sample",
                    help="sample (default): each eval draws 1 of K fixed bundles (~1× wall, "
                         "anti-overfit across bundles). mean: average all K (~K× wall).")
    ap.add_argument("--val-seed", type=int, default=None,
                    help="held-out stim seed for in-training selection + early-stop + "
                         "polish gate (matches bench --eval-seed). Must differ from "
                         "--seed .. --seed+stage2-n-stim-seeds-1. Default: seed+7777.")
    ap.add_argument("--dt", type=float, default=2.0)
    ap.add_argument("--local-refine-idx", type=str, default="prior",
                    help="post-CMA polish target: 'prior' (default; [6,8,10,11]=g_i,d_i,"
                         "theta_c,theta_d focused polish — best held-out), 'active' (all "
                         "unfrozen params; lower train but overfits), or a comma list")
    ap.add_argument("--local-refine-method", choices=["powell", "cma"], default="powell",
                    help="post-CMA polish optimizer (default powell; cma = small-sigma restart). "
                         "Uses the same stage2 stim protocol + val-seed held-out gate.")
    ap.add_argument("--local-refine-max-wall-s", type=float, default=1800.0,
                    help="safety cap (seconds) on local-refine wall time; keeps best-so-far "
                         "on exceed. Default 1800 (~30 min): covers Powell/prior (~15 min) "
                         "and CMA/prior (~11 min) under sample aggregate; set 0 to disable.")
    return ap.parse_args(argv)


def _parse_local_refine_idx(spec):
    """Map --local-refine-idx to the fit_weights_two_stage_v2 `local_refine_idx` arg.

    'prior'  -> None (function default = [6, 8, 10, 11] focused polish)
    'active' -> list(range(12)) (all dims; intersected with train_mask inside the fitter)
    'a,b,c'  -> [a, b, c]
    """
    if spec is None:
        return None
    s = str(spec).strip().lower()
    if s in ("prior", "focused", ""):
        return None
    if s in ("active", "all"):
        return list(range(12))
    return [int(tok) for tok in s.replace(" ", "").split(",") if tok != ""]


def main(argv=None):
    args = build_args(argv)
    freeze_idx = _parse_freeze(args.freeze)
    slug = _mask_slug(freeze_idx)

    # --- deterministic per-variant/seed run dir ---
    prefix = f"fw_{args.out_tag}_" if args.out_tag else "fw_"
    run_dir = Path(save_dir) / f"weights_run_{prefix}{args.mtype}_mask{slug}_s{args.seed}"
    done_marker = run_dir / "FIT_DONE"
    if done_marker.exists() and not args.force:
        print(f"[skip] {run_dir.name} already complete (FIT_DONE). Use --force to redo.")
        return {"skipped": True, "run_dir": str(run_dir)}

    ensure_fit_data_links(pth_res=pth_res, require_avg_mean_r=False)
    disable_realtime_plot()
    loss_history.clear()
    _eval_counter["n"] = 0

    # --- data / regions ---
    mean_path, mean_data_results = load_validated_mean_data()
    print(f"[fit-data] mean_data_results={mean_path}")
    behavior = np.load(Path(pth_res, "behavior.npy"), allow_pickle=True).flat[0]
    prior_regions = {
        "int_regs_choice": int_regs, "int_regs_stim": int_regs,
        "move_regs_choice": move_regs, "move_regs_stim": move_regs,
        "stim_regs": ["VISpm", "FRP", "VISal"],
    }

    # --- decide resume source ---
    # Prefer a *good* in-folder checkpoint for kill/restart. Skip penalty dumps
    # (loss>=1e10) so a failed smoke cannot block a later --resume-json warm start.
    # With --force and an explicit --resume-json, always use the external JSON
    # (ignores in-folder junk from prior runs in the same run dir).
    resume_meta_mp = {}
    resume_theta = None
    resume_loss = None
    resume_source = None
    resume_kind = None
    prefer_external = bool(args.force and args.resume_json)
    if args.resume == "auto" and run_dir.exists() and not prefer_external:
        found = _find_resume(run_dir)
        if found is not None:
            th, loss, mp, src, kind = found
            if np.isfinite(loss) and float(loss) < 1e10:
                resume_theta, resume_loss, resume_meta_mp = th, loss, mp
                resume_source, resume_kind = src, kind
                print(f"[resume] latest checkpoint '{resume_source}' "
                      f"(kind={resume_kind}) loss={resume_loss:.6f}")
            else:
                print(f"[resume] ignoring in-folder '{src}' (loss={loss:.6g} looks "
                      f"like a penalty dump); will try --resume-json / cold start")

    if resume_theta is None and args.resume_json:
        meta = json.loads(Path(args.resume_json).read_text())
        for k in ("g_m", "d_m"):
            grp = "g" if k.startswith("g") else "d"
            if float(meta[grp][k]) <= 0:
                meta[grp][k] = 1e-12
        resume_theta = reconstruct_theta_log_from_json(meta)
        resume_loss = float(meta.get("loss", 1.0))
        resume_meta_mp = meta.get("model_params", {})
        resume_source = f"external:{Path(args.resume_json).name}"
        resume_kind = "stage2"
        why = " (--force prefers external)" if prefer_external else ""
        print(f"[resume] external warm start '{resume_source}' "
              f"loss={resume_loss:.6f}{why}")

    if args.pipeline == "cma_only" and resume_theta is None:
        raise SystemExit("--pipeline cma_only needs a warm start (--resume-json) or an "
                         "existing checkpoint in the run dir.")

    # --- model_params (retinal front-end) alignment ---
    for k, v in (resume_meta_mp or {}).items():
        if isinstance(v, (int, float, np.floating)):
            model_params[k] = float(v)
    model_params["direct_offset"] = False
    model_params["dt"] = float(args.dt)
    _update_model_params_for_dt(model_params, float(args.dt))
    import model_functions as mf
    mf.blocks_per_session = int(args.bps_stage1)
    if hasattr(fw, "blocks_per_session"):
        fw.blocks_per_session = int(args.bps_stage1)
    fw._STIMULI_BUNDLE_CACHE = None

    # --- train mask ---
    train_mask = np.ones(12, dtype=bool)
    for i in freeze_idx:
        train_mask[i] = False
    frozen = [PARAM_NAMES[i] for i in freeze_idx]
    print(f"variant mtype={args.mtype} mask={slug} ({frozen or 'none'}) "
          f"pipeline={args.pipeline} seed={args.seed} n_jobs={args.n_jobs} backend={args.backend}")

    fw._ensure_run_dirs(run_dir=run_dir)
    print(f"run_dir: {run_dir}")

    # --- schedule: restart or external warm start skips DE Stage 1 ---
    local = args.pipeline in ("de_cma_local", "cma_only")
    if resume_theta is not None:
        stage1 = "cma"
        de1_maxiter = 0
        resume_from = _resume_from_kind(resume_kind)
        # Polished resume under de_cma (no local) should still skip re-CMA if the
        # latest file is already a polish/final — treat as de2 warm vector instead.
        if resume_from == "local" and not local:
            resume_from = "de2"
        sidecar = run_dir / "resume_start.json"
        sidecar.write_text(json.dumps({
            "loss": float(resume_loss if resume_loss is not None else 1.0),
            "train_mask": train_mask.tolist(),
            "theta_log": np.asarray(resume_theta, float).tolist(),
            "source": resume_source,
            "kind": resume_kind,
            "resume_from": resume_from,
        }, indent=2))
        resume_path = str(sidecar)
    else:
        stage1 = "de"
        de1_maxiter = int(args.de1_maxiter)
        resume_from = "none"
        resume_path = None

    cma_opts_stage2 = {
        "popsize": int(args.popsize),
        "tolfun": 5e-4,
        "tolx": 5e-5,
        "CMA_stds": list(CMA_STDS[train_mask]),
        "CMA_diagonal": False,
    }

    # Held-out seed: default seed+7777 so it never collides with seed..seed+n_stim-1
    # for any reasonable n_stim (<< 7777). Override with --val-seed.
    val_seed = (int(args.val_seed) if args.val_seed is not None
                else int(args.seed) + 7777)

    wall0 = time.perf_counter()
    best = fit_weights_two_stage_v2(
        mean_data_results, prior_regions, behavior,
        model_type="data",
        random_state=int(args.seed),
        top_k=0,
        global_method_stage1=stage1,
        de_popsize=int(args.de_popsize),
        de1_maxiter=de1_maxiter,
        sobol_count=int(args.sobol_count),
        global_method_stage2="cma",
        cma_sigma_scale=0.25,
        cma_sigma_scale_stage2=0.02,
        cma_opts_stage2=cma_opts_stage2,
        de2_maxiter=int(args.de2_maxiter),
        train_mask=train_mask,
        resume_from=resume_from,
        resume_theta_log=resume_theta,
        resume_path=resume_path,
        blocks_per_session_stage2=int(args.bps_stage2),
        n_jobs=int(args.n_jobs),
        parallel_backend=args.backend,
        deterministic_stage2=True,
        L_threshold=2.0,
        cma_early_stop_patience=int(args.patience),
        cma_early_stop_beat_loss=(None if args.beat_loss < 0 else float(args.beat_loss)),
        local_refine_after_cma=local,
        local_refine_idx=_parse_local_refine_idx(args.local_refine_idx),
        local_refine_method=args.local_refine_method,
        local_refine_patience=8,
        local_refine_max_wall_s=(
            None if (args.local_refine_max_wall_s is not None
                     and float(args.local_refine_max_wall_s) <= 0)
            else args.local_refine_max_wall_s
        ),
        stage2_n_stim_seeds=int(args.stage2_n_stim_seeds),
        stage2_stim_aggregate=args.stage2_stim_aggregate,
        val_stim_seed=val_seed,
    )
    wall = time.perf_counter() - wall0

    fit_status = best.get("fit_status", "ok")
    report = {
        "mtype": args.mtype,
        "mask": slug,
        "frozen": frozen,
        "pipeline": args.pipeline,
        "seed": args.seed,
        "val_seed": val_seed,
        "stage2_n_stim_seeds": int(args.stage2_n_stim_seeds),
        "stage2_stim_aggregate": args.stage2_stim_aggregate,
        "n_jobs": args.n_jobs,
        "backend": args.backend,
        "local_refine_idx": args.local_refine_idx,
        "local_refine_method": args.local_refine_method,
        "local_refine_max_wall_s": args.local_refine_max_wall_s,
        "resumed_from": resume_source,
        "resume_kind": resume_kind,
        "final_loss": float(best["loss"]),
        "g_i": float(best["g"][0]),
        "d_i": float(best["d"][0]),
        "wall_s": wall,
        "run_dir": str(run_dir),
        "log_path": best.get("log_path"),
        "fit_status": fit_status,
        "fail_reason": best.get("fail_reason"),
    }
    (run_dir / "run_fit_weights_report.json").write_text(json.dumps(report, indent=2))

    if fit_status != "ok":
        # Do NOT write FIT_DONE — re-submit can retry (or inspect FIT_FAILED).
        failed = run_dir / "FIT_FAILED"
        failed.write_text(json.dumps({
            "loss": float(best["loss"]),
            "fit_status": fit_status,
            "fail_reason": best.get("fail_reason"),
            "ts": datetime.now().isoformat(),
        }, indent=2))
        print(f"\n=== fit FAILED: status={fit_status} reason={best.get('fail_reason')} "
              f"loss={best['loss']:.6f} wall={wall:.1f}s ===")
        print(f"run_dir: {run_dir}")
        raise SystemExit(2)

    # Guarantee a final weights JSON regardless of which internal path saved.
    _save_params_v2(np.asarray(best["theta_log"], float), float(best["loss"]),
                    tag="final", random_state=int(args.seed), train_mask=train_mask)

    done_marker.write_text(json.dumps({"loss": float(best["loss"]), "ts": datetime.now().isoformat()}))
    failed_marker = run_dir / "FIT_FAILED"
    if failed_marker.exists():
        failed_marker.unlink()
    print(f"\n=== fit done: loss={best['loss']:.6f} g_i={best['g'][0]:.1f} "
          f"d_i={best['d'][0]:.2f} wall={wall:.1f}s ===")
    print(f"run_dir: {run_dir}")
    return report


if __name__ == "__main__":
    main()
