#!/usr/bin/env python
"""Smoke for in-kernel stochastic/soft threshold (T=0 identity, T>0 changes RT)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import model_functions as mf
from model_functions import create_stimuli, run_model
from simulate_recovery import (
    attach_threshold_temperature,
    threshold_temp_dir_tag,
)


def _tiny_stim(seed=0, bps=2):
    mp = copy.deepcopy(mf.model_params)
    mp["direct_offset"] = False
    mp["nonlin_type"] = "linear"
    mp["g_s"] = 0.0
    mp["d_s"] = 0.0
    dt = 2.0
    mp["dt"] = dt
    mf._update_model_params_for_dt(mp, dt)
    steps_before_obs = 500
    max_obs = 400
    rng = np.random.RandomState(seed)
    stimuli, trial_strengths, _, trial_sides, block_sides = create_stimuli(
        bps,
        mf.trials_per_block_param,
        mf.block_side_probs,
        mf.num_stimulus_strength,
        mf.min_stimulus_strength,
        mf.max_stimulus_strength,
        mf.min_trials_per_block,
        mf.max_trials_per_block,
        max_obs,
        steps_before_obs,
        rng=rng,
        **mp,
    )
    return mp, stimuli, trial_strengths, trial_sides, block_sides, bps, steps_before_obs


def _run(backend, mp, stimuli, trial_strengths, trial_sides, block_sides, bps, sbo, **extra):
    kw = dict(mp)
    kw.update(extra)
    return run_model(
        "data",
        stimuli,
        trial_strengths,
        trial_sides,
        block_sides,
        bps,
        steps_before_obs=sbo,
        verbose=False,
        backend=backend,
        **kw,
    )


def _choices_match_action_sign(results):
    """Committed ±1 choice follows sign(action)=sign(tanh(M0−M1)); left is −1."""
    choices = np.asarray(results["choices"])
    atime = np.asarray(results["action_time"])
    asig = np.asarray(results["action_signal"])
    committed = [i for i, ch in enumerate(choices) if ch in (-1, 1)]
    assert len(atime) == len(committed), (len(atime), len(committed))
    for i, t in zip(committed, atime):
        act = float(asig[int(t)])
        expected = -1 if act >= 0.0 else 1
        assert int(choices[i]) == expected, (i, int(choices[i]), act, expected)


def test_cache_tag():
    mp = {"g_s": 0.0}
    attach_threshold_temperature(mp, 0.0)
    assert "threshold_temperature" not in mp
    assert threshold_temp_dir_tag(0.0) == ""
    attach_threshold_temperature(mp, 0.05)
    assert mp["threshold_temperature"] == 0.05
    assert threshold_temp_dir_tag(0.05) == "_softthr0.05"


def test_t0_numpy_numba_match():
    mp, stim, ts, tside, bside, bps, sbo = _tiny_stim(seed=7)
    r_np = _run("numpy", mp, stim, ts, tside, bside, bps, sbo)
    r_nb = _run("numba", mp, stim, ts, tside, bside, bps, sbo)
    for key in ("choices", "correct_action_taken", "reaction_time"):
        an = np.asarray(r_np[key])
        bn = np.asarray(r_nb[key])
        assert an.shape == bn.shape and np.array_equal(an, bn), key
    for key in ("S", "I", "P", "M", "action_signal"):
        an = np.asarray(r_np[key], dtype=float)
        bn = np.asarray(r_nb[key], dtype=float)
        assert np.allclose(an, bn, rtol=1e-12, atol=1e-12, equal_nan=True), key


def test_t_positive_changes_rt_and_matches_across_backends():
    mp, stim, ts, tside, bside, bps, sbo = _tiny_stim(seed=7)
    extra = {"threshold_temperature": 0.05, "threshold_rng_seed": 12345}
    r_hard = _run("numpy", mp, stim, ts, tside, bside, bps, sbo)
    r_np = _run("numpy", mp, stim, ts, tside, bside, bps, sbo, **extra)
    r_nb = _run("numba", mp, stim, ts, tside, bside, bps, sbo, **extra)
    rt_hard = np.asarray(r_hard["reaction_time"])
    rt_soft = np.asarray(r_np["reaction_time"])
    assert rt_hard.shape == rt_soft.shape
    assert not np.array_equal(rt_hard, rt_soft), "T=0.05 should change some RTs"
    for key in ("choices", "correct_action_taken", "reaction_time"):
        an = np.asarray(r_np[key])
        bn = np.asarray(r_nb[key])
        assert an.shape == bn.shape and np.array_equal(an, bn), key
    _choices_match_action_sign(r_np)
    _choices_match_action_sign(r_nb)


if __name__ == "__main__":
    test_cache_tag()
    test_t0_numpy_numba_match()
    test_t_positive_changes_rt_and_matches_across_backends()
    print("soft_threshold smoke OK")
