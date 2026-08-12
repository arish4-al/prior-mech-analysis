"""
CLI driver for joint fitting: retinal + g_s/d_s + network weights/θ.

21-d mixed vector (see fit_joint.py). Modeled on run_fit_weights.py.

Default sensory-prior ablation freezes g_i,g_m,d_i,d_m (indices 6–9):

  PYTHONPATH=. python scripts/run_fit_joint.py --mtype sensory --freeze 6,7,8,9 --seed 56

Warm-start DE (external JSON + de_cma*): Stage-1 DE holds Stage-A retinal
(dims 14–20) at the loaded values and searches the rest; Stage-2 CMA and
polish unfreeze retinal. Pass --no-stage1-hold-retinal to search retinal in DE.

  PYTHONPATH=. python scripts/run_fit_joint.py --mtype regular --freeze 12,13 --seed 56 \\
      --pipeline de_cma_local --resume-json path/to/weights_*.json --stage1-hold-retinal

ORCD: scripts/run_fit_joint_slurm.sh + submit_fit_joint_sharded.sh
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import _one_bypass  # noqa: F401
except ImportError:
    from scripts import _one_bypass  # noqa: F401

try:
    from _fit_data import (
        ensure_fit_data_links,
        load_validated_mean_data,
        load_avg_mean_r,
    )
except ImportError:
    from scripts._fit_data import (
        ensure_fit_data_links,
        load_validated_mean_data,
        load_avg_mean_r,
    )

from model_functions import (
    pth_res,
    int_regs,
    move_regs,
    model_params,
    save_dir,
    _update_model_params_for_dt,
)
import fit_weights as fw
from fit_weights import disable_realtime_plot, loss_history, _eval_counter
import fit_joint as fj
from fit_joint import (
    D_JOINT,
    PARAM_NAMES,
    CMA_STDS,
    DEFAULT_REFINE_IDX,
    SENSORY_REFINE_IDX,
    RETINAL_REFINE_IDX,
    fit_joint_two_stage,
    reconstruct_theta_joint_from_json,
    _save_params_joint,
)


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
    """Latest mtime checkpoint with a valid 21-d (or expandable) theta."""
    candidates = []
    last = run_dir / "weights_stage2_last.json"
    if last.is_file():
        candidates.append(last)
    for pat in (
        "weights_2stagelocalrefine_*.json",
        "weights_final_*.json",
        "weights_localrefine_*.json",
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
                if theta.size != D_JOINT:
                    # Active-only DE/CMA ckpt: expand via fit_idx + joint freeze fill.
                    fit_idx = meta.get("fit_idx", meta.get("fit_id", None))
                    if (
                        fit_idx is not None
                        and len(fit_idx) == theta.size
                        and max(int(i) for i in fit_idx) < D_JOINT
                    ):
                        # Prefer the warm-start full vector so Stage-1-held retinal
                        # (not in fit_idx) is not zero-filled.
                        full = fj.freeze_fill_joint()
                        start = run_dir / "resume_start.json"
                        if start.is_file():
                            try:
                                st = json.loads(start.read_text())
                                tl = st.get("theta_log")
                                if tl is not None and len(tl) == D_JOINT:
                                    full = np.asarray(tl, float)
                            except Exception:
                                pass
                        full[np.asarray(fit_idx, int)] = theta
                        theta = full
                    else:
                        try:
                            theta = reconstruct_theta_joint_from_json(meta)
                        except Exception:
                            continue
            else:
                theta = reconstruct_theta_joint_from_json(meta)
            if theta.size != D_JOINT or not np.all(np.isfinite(theta)):
                continue
            raw_loss = meta.get("loss", 1.0)
            try:
                loss = float(1.0 if raw_loss is None else raw_loss)
            except (TypeError, ValueError):
                loss = 1.0
            # Skip penalty dumps unless nothing else exists (caller may FORCE+external).
            name = c.name
            sel = meta.get("selection")
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
            return (theta, loss, meta.get("model_params", {}), name, kind)
        except Exception:
            continue
    return None


def _resume_from_kind(kind):
    if kind == "polish":
        return "local"
    if kind == "de1":
        return "de1"
    return "de2"


def _parse_local_refine_idx(spec):
    if spec is None:
        return None
    s = str(spec).strip().lower()
    if s in ("prior", "focused", ""):
        return None  # → DEFAULT_REFINE_IDX via fitter hook
    if s in ("sensory",):
        return list(SENSORY_REFINE_IDX)
    if s in ("retinal",):
        return list(RETINAL_REFINE_IDX)
    if s in ("active", "all"):
        return list(range(D_JOINT))
    return [int(tok) for tok in s.replace(" ", "").split(",") if tok != ""]


def build_args(argv=None):
    ap = argparse.ArgumentParser(description="Joint fit: retinal + g_s/d_s + weights.")
    ap.add_argument("--mtype", type=str, default="sensory")
    ap.add_argument("--freeze", type=str, default="6,7,8,9",
                    help="comma indices to freeze (0..20); default 6,7,8,9 = g_i,g_m,d_i,d_m")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=_default_n_jobs())
    ap.add_argument("--pipeline", choices=["de_cma_local", "de_cma", "cma_only"],
                    default="de_cma_local",
                    help="de_cma*: Stage-1 DE→CMA[(→polish)]. With --resume-json, "
                         "DE is warm-seeded (x0 inject) unless cma_only (skip DE).")
    ap.add_argument("--out-tag", type=str, default="")
    ap.add_argument("--resume", choices=["auto", "off"], default="auto")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--backend", choices=["loky", "threading"], default="loky")
    ap.add_argument("--bps-stage1", type=int, default=10,
                    help="blocks/session in Stage 1 (default 10; matches fit_retinal)")
    ap.add_argument("--bps-stage2", type=int, default=20)
    ap.add_argument("--de1-maxiter", type=int, default=40)
    ap.add_argument("--de2-maxiter", type=int, default=40)
    ap.add_argument("--de-popsize", type=int, default=8)
    ap.add_argument("--popsize", type=int, default=16)
    ap.add_argument("--sobol-count", type=int, default=8)
    ap.add_argument("--patience", type=int, default=8)
    # Joint loss = Lw + LS (~0.4 + ~0.8); higher beat bar than weights-only 0.4044.
    ap.add_argument("--beat-loss", type=float, default=1.2)
    ap.add_argument("--l-threshold", type=float, default=10.0,
                    help="Stage-1 gate on joint loss (Lw+LS); default 10 "
                         "(joint scale is higher than weights-only 3.5)")
    ap.add_argument("--resume-json", type=str, default=None,
                    help="Warm-start JSON (weights or joint). With de_cma*: seeds "
                         "Stage-1 DE. With cma_only: skip DE, start Stage-2 CMA.")
    ap.add_argument("--stage1-hold-retinal", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Stage-1 DE holds retinal dims 14–20 at the loaded "
                         "theta_log0 (Stage-A values); Stage-2 CMA and polish "
                         "unfreeze them. Default off; Stage B submit turns it on.")
    ap.add_argument("--stage2-n-stim-seeds", type=int, default=3)
    ap.add_argument("--stage2-stim-aggregate", choices=["sample", "mean"], default="sample")
    ap.add_argument("--val-seed", type=int, default=None)
    ap.add_argument("--dt", type=float, default=2.0)
    ap.add_argument("--local-refine-idx", type=str, default="prior",
                    help="prior|sensory|retinal|active|comma indices")
    ap.add_argument("--local-refine-method", choices=["powell", "cma"], default="powell")
    ap.add_argument("--local-refine-max-wall-s", type=float, default=1800.0)
    return ap.parse_args(argv)


def main(argv=None):
    args = build_args(argv)
    freeze_idx = _parse_freeze(args.freeze)
    slug = _mask_slug(freeze_idx)

    prefix = f"fj_{args.out_tag}_" if args.out_tag else "fj_"
    run_dir = Path(save_dir) / f"weights_run_{prefix}{args.mtype}_mask{slug}_s{args.seed}"
    done_marker = run_dir / "FIT_DONE"
    if done_marker.exists() and not args.force:
        print(f"[skip] {run_dir.name} already complete (FIT_DONE). Use --force to redo.")
        return {"skipped": True, "run_dir": str(run_dir)}

    ensure_fit_data_links(pth_res=pth_res, require_avg_mean_r=True)
    disable_realtime_plot()
    loss_history.clear()
    _eval_counter["n"] = 0

    mean_path, mean_data_results = load_validated_mean_data()
    print(f"[fit-data] mean_data_results={mean_path}")
    avg_path, avg_data_R = load_avg_mean_r()
    print(f"[fit-data] avg_mean_R={avg_path}")
    behavior = np.load(Path(pth_res, "behavior.npy"), allow_pickle=True).flat[0]
    prior_regions = {
        "int_regs_choice": int_regs, "int_regs_stim": int_regs,
        "move_regs_choice": move_regs, "move_regs_stim": move_regs,
        "stim_regs": ["VISpm", "FRP", "VISal"],
    }

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
                print(f"[resume] ignoring in-folder '{src}' (loss={loss:.6g})")

    if resume_theta is None and args.resume_json:
        meta = json.loads(Path(args.resume_json).read_text())
        resume_theta = reconstruct_theta_joint_from_json(meta)
        raw_loss = meta.get("loss", 1.0)
        try:
            resume_loss = float(1.0 if raw_loss is None else raw_loss)
        except (TypeError, ValueError):
            resume_loss = 1.0
        if not np.isfinite(resume_loss):
            resume_loss = 1.0
        resume_meta_mp = meta.get("model_params", {})
        resume_source = f"external:{Path(args.resume_json).name}"
        resume_kind = "stage2"
        why = " (--force prefers external)" if prefer_external else ""
        print(f"[resume] external warm start '{resume_source}' "
              f"loss={resume_loss:.6f} size={resume_theta.size}{why}")

    if args.pipeline == "cma_only" and resume_theta is None:
        raise SystemExit("--pipeline cma_only needs --resume-json or an in-folder ckpt")

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
    fj.blocks_per_session = int(args.bps_stage1)
    fw._STIMULI_BUNDLE_CACHE = None

    train_mask = np.ones(D_JOINT, dtype=bool)
    for i in freeze_idx:
        if i < 0 or i >= D_JOINT:
            raise SystemExit(f"--freeze index {i} out of range 0..{D_JOINT-1}")
        train_mask[i] = False
    frozen = [PARAM_NAMES[i] for i in freeze_idx]
    train_mask_stage1 = None
    if args.stage1_hold_retinal:
        train_mask_stage1 = train_mask.copy()
        train_mask_stage1[14:21] = False
        held = [PARAM_NAMES[i] for i in range(14, 21) if train_mask[i]]
        print(f"[stage1-hold-retinal] DE holds {held or 'none (already frozen)'}; "
              f"CMA/polish unfreeze retinal")
    print(f"variant mtype={args.mtype} mask={slug} ({frozen or 'none'}) "
          f"pipeline={args.pipeline} seed={args.seed} n_jobs={args.n_jobs} "
          f"backend={args.backend} D={D_JOINT} "
          f"bps1={args.bps_stage1} bps2={args.bps_stage2} "
          f"hold_retinal={bool(args.stage1_hold_retinal)}")

    fw._ensure_run_dirs(run_dir=run_dir)
    print(f"run_dir: {run_dir}")

    local = args.pipeline in ("de_cma_local", "cma_only")
    # External --resume-json + de_cma*: warm-start DE (inject θ into DE init).
    # In-folder auto-resume of an ongoing run still skips completed stages.
    # cma_only: skip DE, jump to Stage-2 CMA (unchanged).
    external_warm = (
        resume_theta is not None
        and resume_source is not None
        and str(resume_source).startswith("external:")
    )
    warm_de = (
        external_warm
        and args.pipeline in ("de_cma_local", "de_cma")
    )

    theta_log0 = None
    resume_theta_for_fit = resume_theta
    if warm_de:
        stage1 = "de"
        de1_maxiter = int(args.de1_maxiter)
        resume_from = "none"
        theta_log0 = np.asarray(resume_theta, float)
        resume_theta_for_fit = None
        resume_path = None
        sidecar = run_dir / "resume_start.json"
        sidecar.write_text(json.dumps({
            "loss": float(resume_loss if resume_loss is not None else 1.0),
            "train_mask": train_mask.tolist(),
            "theta_log": theta_log0.tolist(),
            "source": resume_source,
            "kind": "warm_de",
            "resume_from": "none",
            "layout": "joint21",
            "note": "Stage-1 DE seeded from warm θ (x0 inject); retinal held if --stage1-hold-retinal",
        }, indent=2))
        print(f"[warm-DE] Stage-1 DE from '{resume_source}' "
              f"(de1_maxiter={de1_maxiter}, x0 inject + random pop)")
    elif resume_theta is not None:
        stage1 = "cma"
        de1_maxiter = 0
        resume_from = _resume_from_kind(resume_kind)
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
            "layout": "joint21",
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
    val_seed = (int(args.val_seed) if args.val_seed is not None
                else int(args.seed) + 7777)

    refine_idx = _parse_local_refine_idx(args.local_refine_idx)
    if args.stage1_hold_retinal:
        if refine_idx is None:
            refine_idx = list(DEFAULT_REFINE_IDX)
        refine_idx = sorted(set(int(i) for i in refine_idx) | set(RETINAL_REFINE_IDX))

    wall0 = time.perf_counter()
    best = fit_joint_two_stage(
        mean_data_results, prior_regions, behavior, avg_data_R,
        model_type="data",
        random_state=int(args.seed),
        top_k=0,
        theta_log0=theta_log0,
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
        train_mask_stage1=train_mask_stage1,
        resume_from=resume_from,
        resume_theta_log=resume_theta_for_fit,
        resume_path=resume_path,
        blocks_per_session_stage2=int(args.bps_stage2),
        n_jobs=int(args.n_jobs),
        parallel_backend=args.backend,
        deterministic_stage2=True,
        L_threshold=float(args.l_threshold),
        cma_early_stop_patience=int(args.patience),
        cma_early_stop_beat_loss=(None if args.beat_loss < 0 else float(args.beat_loss)),
        local_refine_after_cma=local,
        local_refine_idx=refine_idx,
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
        "layout": "joint21",
        "warm_de": bool(warm_de),
        "stage1_hold_retinal": bool(args.stage1_hold_retinal),
        "resume_source": resume_source,
        "val_seed": val_seed,
        "n_jobs": args.n_jobs,
        "backend": args.backend,
        "final_loss": float(best["loss"]),
        "g_i": float(best["g"][0]),
        "d_i": float(best["d"][0]),
        "g_s": float(best.get("g_s", float("nan"))),
        "d_s": float(best.get("d_s", float("nan"))),
        "wall_s": wall,
        "run_dir": str(run_dir),
        "fit_status": fit_status,
        "fail_reason": best.get("fail_reason"),
    }
    (run_dir / "run_fit_joint_report.json").write_text(json.dumps(report, indent=2))

    if fit_status != "ok":
        failed = run_dir / "FIT_FAILED"
        failed.write_text(json.dumps({
            "loss": float(best["loss"]),
            "fit_status": fit_status,
            "fail_reason": best.get("fail_reason"),
            "ts": datetime.now().isoformat(),
        }, indent=2))
        print(f"\n=== joint fit FAILED: status={fit_status} reason={best.get('fail_reason')} "
              f"loss={best['loss']:.6f} wall={wall:.1f}s ===")
        print(f"run_dir: {run_dir}")
        raise SystemExit(2)

    _save_params_joint(np.asarray(best["theta_log"], float), float(best["loss"]),
                       tag="final", random_state=int(args.seed), train_mask=train_mask)
    done_marker.write_text(json.dumps({
        "loss": float(best["loss"]), "ts": datetime.now().isoformat(), "layout": "joint21",
    }))
    failed_marker = run_dir / "FIT_FAILED"
    if failed_marker.exists():
        failed_marker.unlink()
    print(f"\n=== joint fit done: loss={best['loss']:.6f} "
          f"g_s={best.get('g_s', float('nan')):.3g} d_s={best.get('d_s', float('nan')):.3g} "
          f"wall={wall:.1f}s ===")
    print(f"run_dir: {run_dir}")
    return report


if __name__ == "__main__":
    main()
