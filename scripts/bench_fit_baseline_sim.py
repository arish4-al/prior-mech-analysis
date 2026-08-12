"""
Baseline: time one fit-like session sim + one loss eval at the saved fitted weights.

Does not import fit_weights.py (its bottom is unguarded and starts a full fit).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from simulate_recovery import resolve_weights_json, load_fitted_model, simulate_session
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


def reconstruct_theta_log(meta: dict) -> np.ndarray:
    vec = np.array(
        [
            float(meta["W"]["W_ii"]),
            float(meta["W"]["W_pp"]),
            float(meta["W"]["W_mm"]),
            float(meta["W"]["W_is"]),
            float(meta["W"]["W_pi"]),
            float(meta["W"]["W_mi"]),
            float(meta["g"]["g_i"]),
            float(meta["g"]["g_m"]),
            float(meta["d"]["d_i"]),
            float(meta["d"]["d_m"]),
            float(meta["theta"]["theta_c"]),
            float(meta["theta"]["theta_d"]),
        ],
        dtype=float,
    )
    if np.any(vec <= 0):
        raise ValueError(f"nonpositive params at {np.where(vec <= 0)[0].tolist()}")
    return np.log(vec)


def unpack_theta_log(theta_log: np.ndarray):
    t = np.asarray(theta_log, float)
    return np.exp(t)


def mp_from_theta(theta_log: np.ndarray) -> dict:
    (
        W_ii,
        W_pp,
        W_mm,
        W_is,
        W_pi,
        W_mi,
        g_i,
        g_m,
        d_i,
        d_m,
        theta_c,
        theta_d,
    ) = unpack_theta_log(theta_log)
    mp = dict(mf.model_params)
    mp.update(
        dict(
            W_ii=W_ii,
            W_pp=W_pp,
            W_mm=W_mm,
            W_is=W_is,
            W_pi=W_pi,
            W_mi=W_mi,
            g_i=g_i,
            g_m=g_m,
            d_i=d_i,
            d_m=d_m,
            g_s=0.0,
            d_s=0.0,
            direct_offset=False,
            action_thresholds={
                "concordant": {c: theta_c for c in [1.0, 0.25, 0.125, 0.0625, 0.0]},
                "discordant": {c: theta_d for c in [1.0, 0.25, 0.125, 0.0625, 0.0]},
            },
        )
    )
    mp["dt"] = 2.0
    mf._update_model_params_for_dt(mp, 2.0)
    return mp


def section_time_loss(mp, mean_data, prior_regions, bps: int, seed: int):
    steps_before_obs = 500
    max_obs_per_trial = 1000
    stim_rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
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
        **mp,
    )
    t_stim = time.perf_counter() - t0

    t0 = time.perf_counter()
    results = run_model(
        "data",
        stimuli,
        trial_strengths,
        trial_sides,
        block_sides,
        bps,
        steps_before_obs=steps_before_obs,
        verbose=False,
        backend="auto",
        **mp,
    )
    t_sim = time.perf_counter() - t0

    t0 = time.perf_counter()
    sim_out = mean_by_condition(results, steps_before_obs, T=72, var_names=("I", "P", "M"))
    t_avg = time.perf_counter() - t0

    t0 = time.perf_counter()
    loss_traj = loss_plot_diff_by_condition_with_data(
        sim_out, mp, var_names=("I", "P", "M"), mean_data_results=mean_data, plot=False
    )
    t_traj = time.perf_counter() - t0

    t0 = time.perf_counter()
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
        do_plot=False,
        scale_factors=[1, 1, 1],
        include_all_trials=True,
    )
    t_prior = time.perf_counter() - t0

    total = float(loss_traj["total"] + loss_prior["total"])
    return {
        "loss": total,
        "traj": float(loss_traj["total"]),
        "prior": float(loss_prior["total"]),
        "t_stim": t_stim,
        "t_sim": t_sim,
        "t_avg": t_avg,
        "t_traj": t_traj,
        "t_prior": t_prior,
        "t_sum": t_stim + t_sim + t_avg + t_traj + t_prior,
    }


def main():
    weights = resolve_weights_json()
    print("weights:", weights)
    mp_fit, meta = load_fitted_model(g_s=0.0, d_s=0.0)
    print(f"JSON loss={meta['loss']:.6f}  HAVE_NUMBA={mf._HAVE_NUMBA}")

    # --- A. session sim only (fit-like settings) ---
    max_obs = 1000
    # warmup
    simulate_session(mp_fit, 5, np.random.default_rng(0), max_obs)

    print("\n=== A. simulate_session (fitted params, backend=auto) ===")
    for label, bps, n in (("stage1_bps5", 5, 5), ("stage2_bps20", 20, 3)):
        times = []
        for i in range(n):
            t0 = time.perf_counter()
            simulate_session(mp_fit, bps, np.random.default_rng(1000 + i), max_obs)
            times.append(time.perf_counter() - t0)
        print(
            f"{label}: mean={np.mean(times):.3f}s  "
            f"range=[{min(times):.3f},{max(times):.3f}]  n={n}"
        )

    print("\n=== A2. numpy vs numba (bps=5, identical stim, seed=42) ===")
    rng = np.random.default_rng(42)
    sbo = int(mf.STEPS_BEFORE_OBS_DURATION_MS / mp_fit["dt"])
    stim = create_stimuli(
        5,
        trials_per_block_param,
        block_side_probs,
        num_stimulus_strength,
        min_stimulus_strength,
        max_stimulus_strength,
        min_trials_per_block,
        max_trials_per_block,
        max_obs,
        sbo,
        rng=rng,
        **mp_fit,
    )
    stimuli, trial_strengths, _, trial_sides, block_sides = stim
    run_model(
        "data",
        stimuli,
        trial_strengths,
        trial_sides,
        block_sides,
        5,
        steps_before_obs=sbo,
        verbose=False,
        backend="numba",
        **mp_fit,
    )
    for backend in ("numpy", "numba"):
        t0 = time.perf_counter()
        run_model(
            "data",
            stimuli,
            trial_strengths,
            trial_sides,
            block_sides,
            5,
            steps_before_obs=sbo,
            verbose=False,
            backend=backend,
            **mp_fit,
        )
        print(f"  {backend}: {time.perf_counter() - t0:.3f}s")

    # --- B. full loss eval breakdown ---
    try:
        from _fit_data import ensure_fit_data_links, load_validated_mean_data
    except ImportError:
        from scripts._fit_data import ensure_fit_data_links, load_validated_mean_data
    ensure_fit_data_links(require_avg_mean_r=False, mean_and_prior=True)
    mean_path, mean_data = load_validated_mean_data()
    behavior_path = Path(pth_res) / "behavior.npy"
    _ = np.load(behavior_path, allow_pickle=True).flat[0]  # ensure present
    prior_regions = {
        "int_regs_choice": int_regs,
        "int_regs_stim": int_regs,
        "move_regs_choice": move_regs,
        "move_regs_stim": move_regs,
        "stim_regs": ["VISpm", "FRP", "VISal"],
    }
    theta_log = reconstruct_theta_log(meta)
    mp = mp_from_theta(theta_log)

    # warmup
    section_time_loss(mp, mean_data, prior_regions, bps=5, seed=0)

    print("\n=== B. loss eval sections at baseline theta ===")
    for bps in (5, 20):
        print(f"-- bps={bps} --")
        for i in range(3):
            r = section_time_loss(mp, mean_data, prior_regions, bps=bps, seed=100 + i)
            print(
                f"  eval{i}: loss={r['loss']:.6f} "
                f"(traj={r['traj']:.4f}+prior={r['prior']:.4f})  "
                f"wall={r['t_sum']:.3f}s  "
                f"stim={r['t_stim']:.3f} sim={r['t_sim']:.3f} "
                f"avg={r['t_avg']:.3f} trajL={r['t_traj']:.3f} priorL={r['t_prior']:.3f}"
            )


if __name__ == "__main__":
    main()
