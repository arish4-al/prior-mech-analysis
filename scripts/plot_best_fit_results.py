"""
Plot traj + prior fit diagnostics for the best ORCD weight finals on one shared
bps=20 session (same stimuli for every model).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from simulate_recovery import load_fitted_model
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

REMOTE = Path.home() / (
    "Downloads/ONE/openalyx.internationalbrainlab.org/models/remote"
)

# Best-of ORCD batch (journals/simulation_fit_speedups.md 2026-08-06a)
DEFAULT_MODELS = [
    REMOTE
    / "weights_run_fw_gain_mask7-9_s89"
    / "weights_final_loss0p2167_20260805-200013.json",
    REMOTE
    / "weights_run_fw_gain_mask7-9_s78"
    / "weights_final_loss0p249_20260805-195936.json",
]


def ensure_prior_data_links():
    figs = Path(pth_res).parent / "figs"
    for name in ("data_act_block_duringstim.npy", "data_act_block_duringchoice.npy"):
        src = figs / name
        dst = Path.cwd() / name
        if src.is_file() and not dst.exists():
            dst.symlink_to(src)


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


def plot_one(json_path: Path, stim_bundle, mean_data, prior_regions, out_dir: Path):
    mp, meta = load_fitted_model(g_s=0.0, d_s=0.0, json_path=json_path)
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
    sim_out = mean_by_condition(
        results, steps_before_obs, T=72, var_names=("I", "P", "M")
    )

    loss_traj = loss_plot_diff_by_condition_with_data(
        sim_out,
        mp,
        var_names=("I", "P", "M"),
        mean_data_results=mean_data,
        plot=True,
        save_dir=str(out_dir),
    )
    loss_prior = loss_prior_effect(
        regions=prior_regions,
        results=results,
        model_params=mp,
        steps_before_obs=steps_before_obs,
        T=72,
        timeframes=("act_block_duringstim", "act_block_duringchoice"),
        alpha=0.05,
        ptype="p_mean_c",
        label_A="integrator",
        label_B="move",
        do_plot=True,
        save_dir=str(out_dir),
        scale_factors=[1, 1, 1],
        include_all_trials=True,
    )
    total = float(loss_traj["total"] + loss_prior["total"])
    summary = {
        "json": str(json_path),
        "recorded_loss": float(meta.get("loss", np.nan)),
        "eval_total": total,
        "traj": float(loss_traj["total"]),
        "prior": float(loss_prior["total"]),
        "g_i": float(mp["g_i"]),
        "d_i": float(mp["d_i"]),
        "g_m": float(mp["g_m"]),
        "d_m": float(mp["d_m"]),
        "theta_c": float(mp["action_thresholds"]["concordant"][0.0]),
        "theta_d": float(mp["action_thresholds"]["discordant"][0.0]),
        "out_dir": str(out_dir),
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=12345, help="shared stim seed")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=REMOTE / "fit_result_plots_bps20",
    )
    ap.add_argument(
        "--weights-json",
        type=Path,
        nargs="*",
        default=None,
        help="override default best finals",
    )
    args = ap.parse_args()

    ensure_prior_data_links()
    mean_path = Path(pth_res) / "mean_data_results.npy"
    if not mean_path.is_file():
        mean_path = Path("mean_data_results.npy")
    mean_data = np.load(mean_path, allow_pickle=True).flat[0]
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

    mp0, _ = load_fitted_model(g_s=0.0, d_s=0.0, json_path=jsons[0])
    stim_bundle = make_shared_stimuli(mp0, bps=args.bps, seed=args.seed)
    print(
        f"shared stim: bps={args.bps} seed={args.seed} "
        f"HAVE_NUMBA={mf._HAVE_NUMBA}"
    )

    args.out_root.mkdir(parents=True, exist_ok=True)
    for jp in jsons:
        tag = jp.parent.name
        out_dir = args.out_root / tag
        print(f"\n=== {tag} ===")
        s = plot_one(jp, stim_bundle, mean_data, prior_regions, out_dir)
        print(
            f"recorded={s['recorded_loss']:.4f}  "
            f"eval={s['eval_total']:.4f} "
            f"(traj={s['traj']:.4f}+prior={s['prior']:.4f})  "
            f"g_i={s['g_i']:.3g} d_i={s['d_i']:.3g}  "
            f"plots -> {s['out_dir']}"
        )
        for f in sorted(out_dir.glob("*.svg")):
            print(f"  {f.name}")


if __name__ == "__main__":
    main()
