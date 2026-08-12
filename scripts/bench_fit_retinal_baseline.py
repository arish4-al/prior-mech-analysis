"""
Baseline: time retinal loss evals at frozen front-end params (from weights JSON)
and at the best saved DE2 checkpoint. Avoids importing fit_retinal.py so the bench
stays independent of the modernized fitter (fit_retinal is now import-safe).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from simulate_recovery import resolve_weights_json
import model_functions as mf
from model_functions import (
    create_stimuli,
    run_model,
    mean_S_by_contrast,
    compute_sse_stim_right,
    trials_per_block_param,
    block_side_probs,
    num_stimulus_strength,
    min_stimulus_strength,
    max_stimulus_strength,
    min_trials_per_block,
    max_trials_per_block,
)

RETINAL_KEYS = ("alpha_w", "beta_w", "alpha_d", "beta_d", "tau_a", "W_as", "W_ss")
FIT_RUN_DE2 = Path(
    "/Users/ariliu/Downloads/ONE/openalyx.internationalbrainlab.org/models/"
    "fit_run_20251013_212918/ckpts/de2_ckpt_2025-10-13T23-22-07.npz"
)

try:
    from _fit_data import load_avg_mean_r as _load_avg_mean_r
except ImportError:
    from scripts._fit_data import load_avg_mean_r as _load_avg_mean_r


def load_avg_mean_R():
    path, data = _load_avg_mean_r()
    return data, path


def unpack_mixed(theta: np.ndarray):
    t = np.asarray(theta, float)
    return (
        np.exp(t[0]),
        t[1],
        np.exp(t[2]),
        np.exp(t[3]),
        np.exp(t[4]),
        np.exp(t[5]),
        np.exp(t[6]),
    )


def pack_mixed_from_params(p: dict) -> np.ndarray:
    return np.array(
        [
            np.log(float(p["alpha_w"])),
            float(p["beta_w"]),
            np.log(float(p["alpha_d"])),
            np.log(float(p["beta_d"])),
            np.log(float(p["tau_a"])),
            np.log(float(p["W_as"])),
            np.log(float(p["W_ss"])),
        ],
        dtype=float,
    )


def base_mp_retinal(retinal: dict, bps_unused=None) -> dict:
    """Match fit_retinal.py: prior gains off; network weights at script defaults."""
    mp = dict(mf.model_params)
    mp.update(
        dict(
            direct_offset=False,
            W_pp=0.45,
            W_ii=0.375,
            W_mm=0.139,
            W_is=0.119,
            W_pi=0.00107,
            W_mi=1.471,
            g_i=0.0,
            d_i=0.0,
            g_m=0.0,
            d_m=0.0,
            g_s=0.0,
            d_s=0.0,
            dt=2.0,
        )
    )
    mp.update({k: float(retinal[k]) for k in RETINAL_KEYS})
    theta = [0.78, 0.54]
    mp["action_thresholds"] = {
        "concordant": {c: theta[0] for c in [1.0, 0.25, 0.125, 0.0625, 0.0]},
        "discordant": {c: theta[1] for c in [1.0, 0.25, 0.125, 0.0625, 0.0]},
    }
    mf._update_model_params_for_dt(mp, 2.0)
    return mp


def section_time(mp, avg_data_R, bps: int, seed: int, baseline=0):
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
    S_avg = mean_S_by_contrast(results, steps_before_obs)
    t_avg = time.perf_counter() - t0

    t0 = time.perf_counter()
    loss_results = compute_sse_stim_right(S_avg, avg_data_R, baseline)
    loss = float(loss_results["total_loss"])
    t_sse = time.perf_counter() - t0

    return {
        "loss": loss,
        "t_stim": t_stim,
        "t_sim": t_sim,
        "t_avg": t_avg,
        "t_sse": t_sse,
        "t_sum": t_stim + t_sim + t_avg + t_sse,
    }


def main():
    weights = resolve_weights_json()
    meta = json.loads(Path(weights).read_text())
    frozen = {k: float(meta["model_params"][k]) for k in RETINAL_KEYS}
    print("weights:", weights)
    print("frozen retinal (used in loss0p4044 weights fit):", frozen)
    print(f"HAVE_NUMBA={mf._HAVE_NUMBA}")

    avg_data_R, avg_path = load_avg_mean_R()
    print("avg_mean_R:", avg_path)

    bps = 10  # fit_retinal.py default
    configs = [("frozen_weights_json", frozen)]

    if FIT_RUN_DE2.is_file():
        z = np.load(FIT_RUN_DE2, allow_pickle=True)
        theta = np.asarray(z["de2_best"], float)
        unpacked = unpack_mixed(theta)
        de2 = dict(zip(RETINAL_KEYS, unpacked))
        configs.append(("de2_ckpt_20251013_loss2p076", de2))
        print("DE2 best params:", {k: float(v) for k, v in de2.items()})
        print("DE2 recorded best_loss (log): 2.0759")

    # warmup
    mp0 = base_mp_retinal(frozen)
    section_time(mp0, avg_data_R, bps=bps, seed=0)

    print(f"\n=== retinal loss sections (bps={bps}, fit_mode=rms) ===")
    for label, retinal in configs:
        mp = base_mp_retinal(retinal)
        print(f"-- {label} --")
        for i in range(3):
            r = section_time(mp, avg_data_R, bps=bps, seed=200 + i)
            print(
                f"  eval{i}: loss={r['loss']:.6f}  wall={r['t_sum']:.3f}s  "
                f"stim={r['t_stim']:.3f} sim={r['t_sim']:.3f} "
                f"avg={r['t_avg']:.3f} sse={r['t_sse']:.3f}"
            )

    # numpy vs numba on one retinal session
    print("\n=== numpy vs numba (bps=10, frozen retinal, seed=42) ===")
    mp = base_mp_retinal(frozen)
    sbo = 500
    stim = create_stimuli(
        bps,
        trials_per_block_param,
        block_side_probs,
        num_stimulus_strength,
        min_stimulus_strength,
        max_stimulus_strength,
        min_trials_per_block,
        max_trials_per_block,
        1000,
        sbo,
        rng=np.random.default_rng(42),
        **mp,
    )
    stimuli, trial_strengths, _, trial_sides, block_sides = stim
    run_model(
        "data",
        stimuli,
        trial_strengths,
        trial_sides,
        block_sides,
        bps,
        steps_before_obs=sbo,
        verbose=False,
        backend="numba",
        **mp,
    )
    for backend in ("numpy", "numba"):
        t0 = time.perf_counter()
        run_model(
            "data",
            stimuli,
            trial_strengths,
            trial_sides,
            block_sides,
            bps,
            steps_before_obs=sbo,
            verbose=False,
            backend=backend,
            **mp,
        )
        print(f"  {backend}: {time.perf_counter() - t0:.3f}s")


if __name__ == "__main__":
    main()
