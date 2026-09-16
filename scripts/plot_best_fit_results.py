"""
Plot traj + prior fit diagnostics for the best ORCD weight finals on one shared
bps=20 session (same stimuli for every model).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulate_recovery import load_fitted_model
import model_functions as mf
from model_functions import (
    create_stimuli,
    run_model,
    mean_by_condition,
    mean_S_by_contrast,
    plot_S_diff_by_contrast_side_with_data,
    compute_sse_stim_right,
    loss_plot_diff_by_condition_with_data,
    loss_prior_effect,
    savefig_svg_png,
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

try:
    from _fit_data import (
        ensure_fit_data_links,
        load_validated_mean_data,
        load_avg_mean_r,
        resolve_fit_targets_dir,
        FIT_TARGETS_DIR,
    )
except ImportError:
    from scripts._fit_data import (
        ensure_fit_data_links,
        load_validated_mean_data,
        load_avg_mean_r,
        resolve_fit_targets_dir,
        FIT_TARGETS_DIR,
    )

REMOTE = Path.home() / (
    "Downloads/ONE/openalyx.internationalbrainlab.org/models/remote"
)
# Canonical fit targets: repo fit_targets/ (notebook nested mean_data + prior).
FIT_DATA_DIR = resolve_fit_targets_dir()

# Best-of ORCD batch (journals/simulation_fit_speedups.md 2026-08-06a)
DEFAULT_MODELS = [
    REMOTE
    / "weights_run_fw_gain_mask7-9_s89"
    / "weights_final_loss0p2167_20260805-200013.json",
    REMOTE
    / "weights_run_fw_gain_mask7-9_s78"
    / "weights_final_loss0p249_20260805-195936.json",
]


def ensure_fit_data_links_paper(data_dir: Path | None = None):
    """Refresh cwd links from repo fit_targets/ (2026-08-12c)."""
    targets = Path(data_dir) if data_dir is not None else FIT_DATA_DIR
    return ensure_fit_data_links(
        fit_targets_dir=targets, require_avg_mean_r=False, mean_and_prior=True,
    )


def load_mean_data_results(data_dir: Path | None = None):
    """Load nested stim/choice mean_data_results from repo fit_targets/."""
    targets = Path(data_dir) if data_dir is not None else FIT_DATA_DIR
    ensure_fit_data_links_paper(targets)
    return load_validated_mean_data(targets / "mean_data_results.npy")


def make_shared_stimuli(mp_ref, bps: int, seed: int):
    steps_before_obs = int(mf.STEPS_BEFORE_OBS_DURATION_MS / mp_ref["dt"])
    max_obs_per_trial = int(mf.MAX_OBS_DURATION_MS / mp_ref["dt"])
    stim_rng = np.random.default_rng(seed)
    stimuli, trial_strengths, _, trial_sides, block_sides = create_stimuli(
        bps,
        trials_per_block_param,
        block_side_probs,
        num_stimulus_strength,
        min_stimulus_strength,
        max_stimulus_strength,
        min_trials_per_block,
        max_trials_per_block,
        max_obs_per_trial,
        steps_before_obs,
        rng=stim_rng,
        **mp_ref,
    )
    return (
        stimuli,
        trial_strengths,
        trial_sides,
        block_sides,
        steps_before_obs,
        bps,
    )


def load_plot_model(json_path: Path):
    """Load weights or joint JSON for diagnostics.

    Re-apply retinal after ``_update_model_params_for_dt`` (that helper
    hard-resets ``tau_a``). Use JSON ``g_s``/``d_s`` when present (joint).
    """
    mp, meta = load_fitted_model(g_s=0.0, d_s=0.0, json_path=json_path)
    ret = meta.get("retinal") or {}
    for k in (
        "alpha_w", "beta_w", "alpha_d", "beta_d", "tau_a", "W_as", "W_ss",
    ):
        if k in ret:
            mp[k] = float(ret[k])
    g = meta.get("g") or {}
    d = meta.get("d") or {}
    if "g_s" in meta or "g_s" in g:
        mp["g_s"] = float(meta.get("g_s", g.get("g_s", 0.0)))
    if "d_s" in meta or "d_s" in d:
        mp["d_s"] = float(meta.get("d_s", d.get("d_s", 0.0)))
    return mp, meta


def alias_prior_effects(out_dir: Path) -> None:
    """Copy the long param-name SVG/PNG onto stable ``prior_effects.*`` names."""
    hits = [
        p for p in out_dir.glob("prior_effects_*.svg") if p.name != "prior_effects.svg"
    ]
    if not hits:
        return
    newest = max(hits, key=lambda p: p.stat().st_mtime)
    shutil.copy2(newest, out_dir / "prior_effects.svg")
    png = newest.with_suffix(".png")
    if png.is_file():
        shutil.copy2(png, out_dir / "prior_effects.png")


ITI_COLORS = {
    "S": {-1: "#1f4e79", 1: "#6baed6"},
    "I": {-1: "#DAA520", 1: "#FFD700"},
    "M": {-1: "#CC5500", 1: "#FF7F0E"},
}
ITI_LABELS = {
    "S": {-1: "stim L", 1: "stim R"},
    "I": {-1: "choice L", 1: "choice R"},
    "M": {-1: "choice L", 1: "choice R"},
}


def _iti_signed_means(results, steps_before_obs, vn, group_by, dt, min_valid=10):
    """Signed pop difference in the ITI window, split L vs R.

    Window is ``[−400, −100)`` ms before stimOn of trial *t*, same as
    ``avg_intertrial_by_prev_ch``. Labels are trial *t−1*: stim side for
    ``group_by='stim'``, choice for ``group_by='choice'``. Each trace is
    ``unit[1]−unit[0]`` (pop R − pop L) so L and R sit on opposite sides.
    """
    var = np.asarray(results[vn], dtype=float)
    choices = results["choices"]
    trial_sides = results["trial_sides"]
    n = len(choices)
    lens = [len(trial_sides[i]) for i in range(n)]
    offsets = np.cumsum([0] + lens[:-1])
    start_before = mf._iti_start_before_steps(dt)
    end_before = mf._iti_end_before_steps(dt)
    length = int(start_before - end_before)
    hard_need = steps_before_obs + mf._min_trial_steps(dt)
    buckets = {-1: [], 1: []}
    if (
        var.ndim != 2
        or var.shape[1] != 2
        or steps_before_obs < start_before
        or steps_before_obs <= end_before
        or length <= 0
    ):
        return {k: None for k in (-1, 1)}, length
    for i in range(1, n):
        if group_by == "stim":
            lab = int(np.sign(trial_sides[i - 1][0]))
        else:
            lab = int(choices[i - 1])
        if lab not in (-1, 1):
            continue
        if lens[i] < hard_need:
            continue
        start = offsets[i] + steps_before_obs - start_before
        stop = offsets[i] + steps_before_obs - end_before
        if start < offsets[i] or stop > offsets[i] + steps_before_obs:
            continue
        seg = var[start:stop, :]
        if seg.shape[0] != length:
            continue
        buckets[lab].append(seg)
    out = {}
    for lab in (-1, 1):
        segs = buckets[lab]
        if len(segs) < min_valid:
            out[lab] = None
            continue
        mean_t2 = np.mean(np.stack(segs, axis=0), axis=0)
        arr = mean_t2.T
        out[lab] = arr[1] - arr[0]
    return out, length


def plot_iti_mean_trajectories(results, steps_before_obs, model_params, save_dir):
    """Three ITI mean-trajectory overlays: S (stim L/R), I and M (choice L/R)."""
    dt = float(model_params.get("dt", mf._DEFAULT_DT))
    t_ms = np.arange(
        -mf.ITI_START_BEFORE_MS, -mf.ITI_END_BEFORE_MS, dt
    )
    specs = (("S", "stim"), ("I", "choice"), ("M", "choice"))
    written = []
    for vn, group_by in specs:
        traces, length = _iti_signed_means(
            results, steps_before_obs, vn, group_by, dt
        )
        t = t_ms[:length] if length else t_ms
        fig, ax = plt.subplots(figsize=(4.0, 3.2))
        for lab in (-1, 1):
            y = traces.get(lab)
            if y is None or y.size == 0:
                continue
            n = min(len(t), y.shape[0])
            ax.plot(
                t[:n],
                y[:n],
                "-",
                linewidth=2,
                color=ITI_COLORS[vn][lab],
                label=ITI_LABELS[vn][lab],
            )
        ax.axhline(0, color="k", linewidth=0.8)
        ax.axvline(-mf.ITI_END_BEFORE_MS, color="k", linewidth=1)
        ax.set_xlim(-mf.ITI_START_BEFORE_MS, -mf.ITI_END_BEFORE_MS)
        ax.set_xlabel("time before stimOn (ms)")
        ax.set_ylabel(f"{vn} pop R − pop L")
        ax.set_title(f"{vn} ITI ({group_by} L/R)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        fig.tight_layout()
        out = Path(save_dir) / f"{vn}_iti.svg"
        savefig_svg_png(fig, str(out), transparent=True)
        plt.close(fig)
        written.append(out)
    return written


def plot_one(json_path: Path, stim_bundle, mean_data, prior_regions, out_dir: Path,
             avg_mean_R=None, include_stim=False):
    mp, meta = load_plot_model(json_path)
    (
        stimuli,
        trial_strengths,
        trial_sides,
        block_sides,
        steps_before_obs,
        bps,
    ) = stim_bundle

    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_model(
        "data",
        stimuli,
        trial_strengths,
        trial_sides,
        block_sides,
        bps,
        steps_before_obs=steps_before_obs,
        verbose=False,
        backend="numba",
        **mp,
    )
    plot_iti_mean_trajectories(results, steps_before_obs, mp, out_dir)
    # Match paper-brain-wide-map/model_test.ipynb diagnostic cell.
    sim_out = mean_by_condition(results, steps_before_obs)

    loss_traj = loss_plot_diff_by_condition_with_data(
        sim_out,
        mp,
        var_names=("I", "P", "M"),
        mean_data_results=mean_data,
        plot=True,
        save_dir=str(out_dir),
    )
    T_prior, plot_win, custom_prior_win = mf.resolve_prior_distance_window(
        mp, T=72, plot_window=80)
    if include_stim and not custom_prior_win:
        plot_win = 150
    loss_prior = loss_prior_effect(
        regions=prior_regions,
        results=results,
        model_params=mp,
        steps_before_obs=steps_before_obs,
        T=T_prior,
        model_metric="l2",
        timeframes=("act_block_duringstim", "act_block_duringchoice"),
        ptype="p_mean_c",
        plot_window=plot_win,
        reload=False,
        label_A="integrator",
        label_B="move",
        do_plot=True,
        plot_shifted=False,
        ylim=None,
        scale_factors=[1, 1, 1],
        include_all_trials=True,
        save_dir=str(out_dir),
        plot_stim=include_stim,
        lump_all=False,
        include_stim=include_stim,
    )
    prior_fig = plt.gcf()
    prior_fig.savefig(
        out_dir / "prior_effects.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    alias_prior_effects(out_dir)
    total = float(loss_traj["total"] + loss_prior["total"])
    L_S = None
    S_r2 = None
    if avg_mean_R is not None:
        plt.close("all")
        S_avg = mean_S_by_contrast(results, steps_before_obs)
        sse = compute_sse_stim_right(S_avg, avg_mean_R, baseline_R=0)
        raw_ls = sse["total_loss"]
        L_S = float(raw_ls) if np.isfinite(raw_ls) else None
        sr = sse.get("total_gof_r2")
        S_r2 = float(sr) if sr is not None and np.isfinite(sr) else None
        plot_S_diff_by_contrast_side_with_data(
            S_avg, {}, avg_mean_R, baseline=0,
            save_dir=str(out_dir), ylim=[-0.14, 0.75], yticks=None,
        )
        for n in list(plt.get_fignums()):
            fig = plt.figure(n)
            fig.savefig(
                out_dir / "S_fit.png",
                dpi=150, bbox_inches="tight", transparent=False,
            )
        plt.close("all")
    gI = (loss_traj.get("gof") or {}).get("I") or {}
    gM = (loss_traj.get("gof") or {}).get("M") or {}
    pg = loss_prior.get("gof")
    try:
        prior_r2 = float(pg) if pg is not None and np.isfinite(float(pg)) else None
    except (TypeError, ValueError):
        prior_r2 = None
    def _g(d, k):
        v = d.get(k)
        return float(v) if v is not None and np.isfinite(v) else None
    summary = {
        "json": str(json_path),
        "recorded_loss": float(meta.get("loss", np.nan)),
        "eval_total": total,
        "traj": float(loss_traj["total"]),
        "prior": float(loss_prior["total"]),
        "L_S": L_S,
        "gof_I_pre": _g(gI, "pre"),
        "gof_I_post": _g(gI, "post"),
        "gof_M_pre": _g(gM, "pre"),
        "gof_M_post": _g(gM, "post"),
        "gof_S": S_r2,
        "gof_prior": prior_r2,
        "g_i": float(mp["g_i"]),
        "d_i": float(mp["d_i"]),
        "g_m": float(mp["g_m"]),
        "d_m": float(mp["d_m"]),
        "theta_c": float(mp["action_thresholds"]["concordant"][0.0]),
        "theta_d": float(mp["action_thresholds"]["discordant"][0.0]),
        "out_dir": str(out_dir),
    }
    if include_stim:
        stim_tf = loss_prior.get("act_block_duringstim") or {}
        summary["prior_S"] = stim_tf.get("stim")
        summary["prior_I_stim"] = stim_tf.get("integrator")
        summary["prior_M_stim"] = stim_tf.get("move")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=12345, help="shared stim seed")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="optional dump root (out-root/<run_name>/). Default: the JSON's parent model dir.",
    )
    ap.add_argument(
        "--weights-json",
        type=Path,
        nargs="*",
        default=None,
        help="override default best finals",
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=FIT_DATA_DIR,
        help="dir with mean_data_results.npy + data_act_block_*.npy (default: repo fit_targets/)",
    )
    ap.add_argument(
        "--include-stim-prior",
        action="store_true",
        help="overlay the unsplit-80 ms S prior-distance sidecar (data + model S)",
    )
    args = ap.parse_args()

    ensure_fit_data_links_paper(args.data_dir)
    mean_path, mean_data = load_mean_data_results(args.data_dir)
    print(f"mean_data_results: {mean_path}")
    avg_path, avg_mean_R = load_avg_mean_r()
    print(f"avg_mean_R: {avg_path}")
    print(
        "prior data cwd links:",
        Path("data_act_block_duringstim.npy").resolve(),
        Path("data_act_block_duringchoice.npy").resolve(),
    )
    prior_regions = {
        "int_regs_choice": int_regs,
        "int_regs_stim": int_regs,
        "move_regs_choice": move_regs,
        "move_regs_stim": move_regs,
        "stim_regs": ["VISpm", "FRP", "VISal"],
    }

    jsons = args.weights_json or DEFAULT_MODELS
    jsons = [Path(p) for p in jsons]
    for p in jsons:
        if not p.is_file():
            raise FileNotFoundError(p)

    mp0, _ = load_plot_model(jsons[0])
    stim_bundle = make_shared_stimuli(mp0, bps=args.bps, seed=args.seed)
    print(
        f"shared stim: bps={args.bps} seed={args.seed} "
        f"HAVE_NUMBA={mf._HAVE_NUMBA}"
    )

    for jp in jsons:
        tag = jp.parent.name
        out_dir = (args.out_root / tag) if args.out_root is not None else jp.parent
        print(f"\n=== {tag} ===")
        s = plot_one(jp, stim_bundle, mean_data, prior_regions, out_dir,
                     avg_mean_R=avg_mean_R, include_stim=args.include_stim_prior)
        ls = s.get("L_S")
        ls_txt = f" L_S={ls:.4f}" if ls is not None else ""
        print(
            f"recorded={s['recorded_loss']:.4f}  "
            f"eval={s['eval_total']:.4f} "
            f"(traj={s['traj']:.4f}+prior={s['prior']:.4f}{ls_txt})  "
            f"g_i={s['g_i']:.3g} d_i={s['d_i']:.3g}  "
            f"plots -> {s['out_dir']}"
        )
        if s.get("prior_S") is not None:
            print(
                f"  duringstim nSSE  S={s['prior_S']:.4f}  "
                f"I={s['prior_I_stim']:.4f}  M={s['prior_M_stim']:.4f}"
            )
        for f in sorted(out_dir.glob("*.svg")):
            print(f"  {f.name}")
        for f in sorted(out_dir.glob("prior_effects.png")):
            print(f"  {f.name}")


if __name__ == "__main__":
    main()
