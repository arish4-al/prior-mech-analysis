"""
Profile fit-path sim: stim build vs numba flatten / kernel / reassembly.

Also estimates cost of unused outputs (a, perceived, action_signal, tiled
metadata arrays) by timing reassembly with/without those steps.

Does not import fit_weights / fit_retinal.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from simulate_recovery import load_fitted_model
import model_functions as mf
from model_functions import (
    create_stimuli,
    run_model,
    _run_model_kernel,
    _NONLIN_CODES,
    set_model_parameters,
    trials_per_block_param,
    block_side_probs,
    num_stimulus_strength,
    min_stimulus_strength,
    max_stimulus_strength,
    min_trials_per_block,
    max_trials_per_block,
)


def build_stim(mp, bps, seed, max_obs=1000, sbo=500):
    rng = np.random.default_rng(seed)
    return create_stimuli(
        bps,
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
        **mp,
    )


def flatten(stimuli, trial_strengths, trial_sides, block_sides, bps, mp):
    action_thresholds = mp["action_thresholds"]
    is_thr_dict = isinstance(action_thresholds, dict)
    n_blocks = int(bps)
    L = int(np.asarray(stimuli[0]).shape[1])
    Ntr = sum(np.asarray(stimuli[i]).shape[0] for i in range(n_blocks))

    stim = np.empty((Ntr, L, 2), dtype=np.float64)
    contrast_mag = np.empty(Ntr, dtype=np.float64)
    trial_side = np.empty(Ntr, dtype=np.float64)
    block_side = np.empty(Ntr, dtype=np.float64)
    theta_c_tr = np.empty(Ntr, dtype=np.float64)
    theta_d_tr = np.empty(Ntr, dtype=np.float64)

    idx = 0
    for i in range(n_blocks):
        blk = np.ascontiguousarray(stimuli[i], dtype=np.float64)
        nt = blk.shape[0]
        for j in range(nt):
            stim[idx] = blk[j]
            c = float(trial_strengths[i][j][0])
            cm = abs(c)
            contrast_mag[idx] = cm
            trial_side[idx] = float(trial_sides[i][j][0])
            block_side[idx] = float(block_sides[i][j][0])
            if is_thr_dict:
                theta_c_tr[idx] = float(action_thresholds["concordant"][cm])
                theta_d_tr[idx] = float(action_thresholds["discordant"][cm])
            else:
                theta_c_tr[idx] = float(action_thresholds)
                theta_d_tr[idx] = float(action_thresholds)
            idx += 1
    return stim, contrast_mag, trial_side, block_side, theta_c_tr, theta_d_tr, L, Ntr


def reassemble_full(kernel_out, trial_side, block_side, contrast_mag, Ntr):
    (
        Sout,
        Iout,
        Pout,
        Mout,
        aout,
        perceived,
        actionsig,
        trial_len,
        choice_arr,
        correct_arr,
        rt_arr,
        atime_arr,
        subprior_mean,
        finite_ok,
        ntot,
    ) = kernel_out
    S = Sout[:ntot]
    I = Iout[:ntot]
    P = Pout[:ntot]
    M = Mout[:ntot]
    a = aout[:ntot]
    perceived_stim = perceived[:ntot]
    action_signal = actionsig[:ntot]

    trial_sides_for_plot = []
    block_sides_for_plot = []
    choice_sides_for_plot = []
    trial_strengths_for_plot = []
    sub_prior = []
    choices = []
    correct_action_taken = []
    reaction_time = []
    action_time = []

    for tr in range(Ntr):
        k = int(trial_len[tr])
        trial_sides_for_plot.append(np.tile(trial_side[tr], k))
        block_sides_for_plot.append(np.tile(block_side[tr], k))
        choice_sides_for_plot.append(np.tile(float(choice_arr[tr]), k))
        trial_strengths_for_plot.append(np.tile(contrast_mag[tr], k))
        sub_prior.append(np.tile(subprior_mean[tr], k))
        choices.append(int(choice_arr[tr]))
        correct_action_taken.append(int(correct_arr[tr]))
        reaction_time.append(int(rt_arr[tr]))
        if atime_arr[tr] >= 0:
            action_time.append(int(atime_arr[tr]))

    return {
        "S": S,
        "I": I,
        "P": P,
        "M": M,
        "a": a,
        "choices": choices,
        "correct_action_taken": correct_action_taken,
        "reaction_time": reaction_time,
        "trial_sides": trial_sides_for_plot,
        "block_sides": block_sides_for_plot,
        "choice_sides": choice_sides_for_plot,
        "trial_strengths": trial_strengths_for_plot,
        "perceived_stim": perceived_stim,
        "sub_prior": sub_prior,
        "action_time": action_time,
        "action_signal": action_signal,
    }


def reassemble_fit_minimal(kernel_out, trial_side, block_side, Ntr):
    """Only fields needed by weights loss: I/P/M (+S for prior), choices,
    trial_sides/block_sides/sub_prior lengths + first values, reaction_time."""
    (
        Sout,
        Iout,
        Pout,
        Mout,
        _aout,
        _perceived,
        _actionsig,
        trial_len,
        choice_arr,
        _correct_arr,
        rt_arr,
        _atime_arr,
        subprior_mean,
        _finite_ok,
        ntot,
    ) = kernel_out
    trial_sides = []
    block_sides = []
    sub_prior = []
    choices = []
    reaction_time = []
    for tr in range(Ntr):
        k = int(trial_len[tr])
        # keep length-k arrays so mean_by_condition lens/offsets stay valid
        trial_sides.append(np.full(k, trial_side[tr]))
        block_sides.append(np.full(k, block_side[tr]))
        sub_prior.append(np.full(k, subprior_mean[tr]))
        choices.append(int(choice_arr[tr]))
        reaction_time.append(int(rt_arr[tr]))
    return {
        "S": Sout[:ntot],
        "I": Iout[:ntot],
        "P": Pout[:ntot],
        "M": Mout[:ntot],
        "choices": choices,
        "reaction_time": reaction_time,
        "trial_sides": trial_sides,
        "block_sides": block_sides,
        "sub_prior": sub_prior,
    }


def call_kernel(mp, stim, contrast_mag, trial_side, theta_c_tr, theta_d_tr, L, sbo=500):
    dt = float(mp["dt"])
    d_s, d_i, d_m, g_s, g_i, g_m = set_model_parameters("data", **mp)
    nonlin_code = _NONLIN_CODES[mp["nonlin_type"]]
    return _run_model_kernel(
        stim,
        contrast_mag,
        trial_side,
        theta_c_tr,
        theta_d_tr,
        L,
        int(sbo),
        int(mf._min_trial_steps(dt)),
        dt,
        float(mp["tau_s"]),
        float(mp["tau_i"]),
        float(mp["tau_p"]),
        float(mp["tau_m"]),
        float(mp["tau_a"]),
        float(mp["W_ss"]),
        float(mp["W_ii"]),
        float(mp["W_pp"]),
        float(mp["W_mm"]),
        float(mp["W_is"]),
        float(mp["W_mi"]),
        float(mp["W_pi"]),
        float(mp["W_as"]),
        float(g_s),
        float(g_i),
        float(g_m),
        float(d_s),
        float(d_i),
        float(d_m),
        float(mp["alpha_d"]),
        float(mp["beta_d"]),
        float(mf._DEFAULT_DT),
        float(mp["baseline"]),
        bool(mp["stim_adap"]),
        bool(mp["direct_offset"]),
        int(nonlin_code),
        int(mp["prestim_offset_start"]),
        int(mp["post_action_steps"]),
        bool(mp.get("gs_outside_adaptation", False)),
    )


def profile_once(mp, bps, seed):
    t0 = time.perf_counter()
    stimuli, trial_strengths, _, trial_sides, block_sides = build_stim(mp, bps, seed)
    t_stim = time.perf_counter() - t0

    t0 = time.perf_counter()
    stim, contrast_mag, trial_side, block_side, theta_c_tr, theta_d_tr, L, Ntr = flatten(
        stimuli, trial_strengths, trial_sides, block_sides, bps, mp
    )
    t_flat = time.perf_counter() - t0

    t0 = time.perf_counter()
    kout = call_kernel(mp, stim, contrast_mag, trial_side, theta_c_tr, theta_d_tr, L)
    t_kern = time.perf_counter() - t0
    ntot = int(kout[-1])
    fill = ntot / (Ntr * L) if Ntr and L else 0.0

    t0 = time.perf_counter()
    reassemble_full(kout, trial_side, block_side, contrast_mag, Ntr)
    t_re_full = time.perf_counter() - t0

    t0 = time.perf_counter()
    reassemble_fit_minimal(kout, trial_side, block_side, Ntr)
    t_re_min = time.perf_counter() - t0

    t0 = time.perf_counter()
    run_model(
        "data",
        stimuli,
        trial_strengths,
        trial_sides,
        block_sides,
        bps,
        steps_before_obs=500,
        verbose=False,
        backend="numba",
        **mp,
    )
    t_run = time.perf_counter() - t0

    return dict(
        t_stim=t_stim,
        t_flat=t_flat,
        t_kern=t_kern,
        t_re_full=t_re_full,
        t_re_min=t_re_min,
        t_run=t_run,
        Ntr=Ntr,
        L=L,
        ntot=ntot,
        fill=fill,
    )


def main():
    mp, meta = load_fitted_model(g_s=0.0, d_s=0.0)
    print("weights loss", meta["loss"], "HAVE_NUMBA", mf._HAVE_NUMBA)

    # warmup JIT
    profile_once(mp, 5, 0)

    for bps in (5, 20):
        print(f"\n=== bps={bps} (n=4 seeds) ===")
        rows = [profile_once(mp, bps, 100 + i) for i in range(4)]
        def mean(key):
            return float(np.mean([r[key] for r in rows]))

        print(
            f"  stim={mean('t_stim'):.3f}s  flatten={mean('t_flat'):.3f}s  "
            f"kernel={mean('t_kern'):.3f}s  re_full={mean('t_re_full'):.3f}s  "
            f"re_min={mean('t_re_min'):.3f}s  run_model={mean('t_run'):.3f}s"
        )
        print(
            f"  Ntr≈{mean('Ntr'):.0f}  L={rows[0]['L']}  "
            f"ntot/NtrL fill≈{mean('fill'):.2f}  "
            f"reassembly save if minimal≈{mean('t_re_full')-mean('t_re_min'):.3f}s"
        )


if __name__ == "__main__":
    main()
