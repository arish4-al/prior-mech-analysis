"""
Plot / evaluate retinal Stage-A fits vs avg_mean_R (notebook diagnostic).

Fair multi-seed compare: shared ``--bps`` + ``--seed``; restim under each model's
α_w/β_w (same protocol as Stage-A ``stage2_restim``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import model_functions as mf
from model_functions import (
    create_stimuli,
    run_model,
    mean_S_by_contrast,
    plot_S_diff_by_contrast_side_with_data,
    compute_sse_stim_right,
    trials_per_block_param,
    block_side_probs,
    num_stimulus_strength,
    min_stimulus_strength,
    max_stimulus_strength,
    min_trials_per_block,
    max_trials_per_block,
)
import fit_retinal as fr

try:
    from _fit_data import FIT_TARGETS_DIR, load_avg_mean_r
except ImportError:
    from scripts._fit_data import FIT_TARGETS_DIR, load_avg_mean_r

MODELS = Path.home() / "Downloads/ONE/openalyx.internationalbrainlab.org/models"
DEFAULT_JSONS = sorted(
    MODELS.glob("retinal_run_fr_retinal_masknone_s*/retinal_final_*.json")
)


def mp_from_retinal_json(json_path: Path):
    """Stage-A anchors + fitted retinal. Do not call _update_model_params_for_dt
    after retinal apply — it hard-resets tau_a to 222.68."""
    meta = json.loads(Path(json_path).read_text())
    fr.apply_retinal_stage_a_defaults(mf.model_params)
    ret = meta.get("retinal") or {}
    if not ret and "theta_log" in meta:
        ret = fr.unpack_retinal(fr.reconstruct_theta_retinal_from_json(meta))
    mp = dict(mf.model_params)
    # Keep fitted retinal after any dt scaffolding (dt only; preserve tau_a).
    mp["dt"] = float(mp.get("dt", 2.0))
    mp["post_action_steps"] = int(40 / mp["dt"])
    mp["prestim_offset_start"] = int(100 / mp["dt"])
    mp.update({k: float(ret[k]) for k in fr.PARAMS if k in ret})
    # Force zero priors (Stage A).
    for k in ("g_i", "g_m", "d_i", "d_m", "g_s", "d_s"):
        mp[k] = 0.0
    return mp, meta, {k: float(ret[k]) for k in fr.PARAMS if k in ret}


def eval_one(json_path: Path, avg_mean_R, bps: int, seed: int, out_dir: Path, ylim):
    mp, meta, ret = mp_from_retinal_json(json_path)
    steps_before_obs = int(mf.STEPS_BEFORE_OBS_DURATION_MS / mp["dt"])
    max_obs = int(mf.MAX_OBS_DURATION_MS / mp["dt"])

    stim_rng = np.random.RandomState(int(seed))
    stimuli, trial_strengths, _, trial_sides, block_sides = create_stimuli(
        int(bps), trials_per_block_param, block_side_probs,
        num_stimulus_strength, min_stimulus_strength, max_stimulus_strength,
        min_trials_per_block, max_trials_per_block,
        max_obs, steps_before_obs, rng=stim_rng, **mp,
    )
    results = run_model(
        "data", stimuli, trial_strengths, trial_sides, block_sides, int(bps),
        steps_before_obs=steps_before_obs, verbose=False, backend="numba", **mp,
    )
    S_avg = mean_S_by_contrast(results, steps_before_obs)
    loss = compute_sse_stim_right(S_avg, avg_mean_R, baseline_R=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_S_diff_by_contrast_side_with_data(
        S_avg, {}, avg_mean_R, baseline=0,
        save_dir=str(out_dir), ylim=list(ylim), yticks=None,
    )
    for n in plt.get_fignums():
        fig = plt.figure(n)
        fig.savefig(
            out_dir / "model_vs_data_stim_Right_(+1).png",
            dpi=150, bbox_inches="tight", transparent=False,
        )
    plt.close("all")

    def _f(x):
        x = float(x)
        return x if np.isfinite(x) else None

    report = {
        "tag": json_path.parent.name,
        "json": str(json_path),
        "recorded_fit_loss": _f(meta.get("loss", np.nan)),
        "eval_bps": int(bps),
        "eval_seed": int(seed),
        "total_loss": _f(loss["total_loss"]),
        "total_sse": _f(loss.get("total_sse", np.nan)),
        "total_snr_loss": _f(loss.get("total_snr_loss", np.nan)),
        "total_gof_r2": _f(loss.get("total_gof_r2", np.nan)),
        "retinal": ret,
        "out_dir": str(out_dir),
    }
    (out_dir / "s_fit_eval.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate/plot retinal Stage-A fits.")
    ap.add_argument(
        "--json", type=Path, nargs="*", default=None,
        help="retinal_final_*.json paths (default: production masknone finals)",
    )
    ap.add_argument("--bps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=12345,
                    help="shared stim seed (restim under each mp)")
    ap.add_argument("--avg-mean-r", type=Path, default=None)
    ap.add_argument(
        "--out-root", type=Path, default=None,
        help="default: <MODELS>/retinal_s_fit_plots_bps<bps>_seed<seed>",
    )
    ap.add_argument("--ylim", type=float, nargs=2, default=[-0.14, 0.75])
    args = ap.parse_args(argv)

    jsons = [Path(p) for p in (args.json or DEFAULT_JSONS)]
    if not jsons:
        raise SystemExit("no retinal_final_*.json found")
    for p in jsons:
        if not p.is_file():
            raise SystemExit(f"missing {p}")

    out_root = args.out_root or (
        MODELS / f"retinal_s_fit_plots_bps{args.bps}_seed{args.seed}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    avg_path, avg_mean_R = load_avg_mean_r(args.avg_mean_r)
    print(f"avg_mean_R={avg_path}")
    print(f"shared eval: bps={args.bps} seed={args.seed} n={len(jsons)}")
    print(f"HAVE_NUMBA={mf._HAVE_NUMBA}  out_root={out_root}")

    rows = []
    for jp in jsons:
        tag = jp.parent.name
        out_dir = out_root / tag
        print(f"\n=== {tag} ===")
        r = eval_one(jp, avg_mean_R, args.bps, args.seed, out_dir, args.ylim)
        rows.append(r)
        print(
            f"  recorded={r['recorded_fit_loss']:.4f}  "
            f"shared_L_S={r['total_loss']:.4f}  "
            f"R2={r['total_gof_r2']}"
        )

    rows_sorted = sorted(
        rows,
        key=lambda r: (r["total_loss"] is None, r["total_loss"] or 1e99),
    )
    summary = {
        "avg_mean_R": str(avg_path),
        "eval_bps": int(args.bps),
        "eval_seed": int(args.seed),
        "protocol": "shared_seed_restim_under_each_mp",
        "n": len(rows_sorted),
        "best_tag": rows_sorted[0]["tag"] if rows_sorted else None,
        "best_shared_L_S": rows_sorted[0]["total_loss"] if rows_sorted else None,
        "rows": rows_sorted,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))

    # Markdown table for journal paste.
    lines = [
        f"| seed / tag | recorded fit `L_S` | shared `L_S` (bps={args.bps}, seed={args.seed}) | R² |",
        "|---|---:|---:|---:|",
    ]
    for r in rows_sorted:
        seed = "".join(ch for ch in r["tag"].split("_s")[-1] if ch.isdigit()) or "?"
        lines.append(
            f"| s{seed} (`{r['tag']}`) | "
            f"{r['recorded_fit_loss']:.4f} | {r['total_loss']:.4f} | "
            f"{r['total_gof_r2']:.3f} |"
        )
    md = "\n".join(lines) + "\n"
    (out_root / "summary.md").write_text(md)
    print("\n" + md)
    print(f"wrote {out_root / 'summary.json'}")
    return summary


if __name__ == "__main__":
    main()
