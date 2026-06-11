#!/usr/bin/env python3
"""
Generative-model recovery demo for the prior-mechanism circuit model.

Simulates trial-level model population trajectories (S/I/M/P), computes prior
distances with label-shuffle nulls per split (as in d_var_stacked_multi), then
runs the classification + prior-modulation analysis pipeline to check recovery
of population types and sensory prior modulation (g_s/d_s).

Run with the iblenv conda environment, e.g.:
    ~/opt/anaconda3/envs/iblenv/bin/python simulate_recovery.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project imports (iblenv must have ibllib / brainwidemap / one-api installed)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import model_functions as mf  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (mirroring block_analysis_allsplits.py)
# ---------------------------------------------------------------------------
B_SIZE = 0.0125
STS = 0.002
NRAND_DEFAULT = 2000
N_SESSIONS_DEFAULT = 40
BLOCKS_PER_SESSION_DEFAULT = 6
MIN_TRIALS_PER_SESSION_DEFAULT = 600
SHUFFLE_PLOT_N_SAMPLE = 100
ALPHA_ACT = 0.2
DT_MS = 2.0
PRIOR_COLUMN = "p_subjective_probabilityLeft"

WEIGHTS_JSON = (
    Path(mf.one.cache_dir)
    / "models"
    / "weights_run_20251125_182058"
    / "weights_2stagelocalrefine_loss0p4044_20251125-195255.json"
)

# Model populations used directly as "regions" (no synthetic neurons / atlas).
MODEL_POPULATIONS = ("S", "I", "M", "P")
POPULATION_TYPE = {"S": "S", "I": "I", "M": "M", "P": "P"}
PRIOR_OFFSET_BINS = 5


def fitted_integrator_scales(json_path=None):
    """Return fitted g_i, d_i to use as comparable g_s, d_s for sensory prior presence."""
    json_path = Path(json_path) if json_path is not None else WEIGHTS_JSON
    with open(json_path) as f:
        meta = json.load(f)
    return float(meta["g"]["g_i"]), float(meta["d"]["d_i"])


GAIN_PRESENCE_DEFAULT, OFFSET_PRESENCE_DEFAULT = fitted_integrator_scales()

SC_TIMES = [
    "stim_duringstim_act",
    "choice_duringstim_act",
    "stim_duringchoice_act",
    "choice_duringchoice_act",
]

TIMING_SPLITS = ["act_block_duringstim", "act_block_duringchoice"]
S_PRIOR_TIMEFRAME = "act_block_duringstim"

FOCUSED_TIMEFRAMES = SC_TIMES + TIMING_SPLITS + [
    "stim_duringstim1_act",
    "stim_duringstim_short_act",
]


def build_align_pre_post():
    """Replicate align / pre_post dict construction from block_analysis_allsplits."""
    align_old = {
        "block_duringstim_r_choice_r_f1": "stimOn_times",
        "block_duringstim_l_choice_l_f1": "stimOn_times",
        "block_duringstim_l_choice_r_f2": "stimOn_times",
        "block_duringstim_r_choice_l_f2": "stimOn_times",
        "block_stim_r_duringchoice_r_f1": "firstMovement_times",
        "block_stim_l_duringchoice_l_f1": "firstMovement_times",
        "block_stim_l_duringchoice_r_f2": "firstMovement_times",
        "block_stim_r_duringchoice_l_f2": "firstMovement_times",
        "block_stim_r_choice_r_f1": "stimOn_times",
        "block_stim_l_choice_l_f1": "stimOn_times",
        "block_stim_l_choice_r_f2": "stimOn_times",
        "block_stim_r_choice_l_f2": "stimOn_times",
        "block_only": "stimOn_times",
        "act_block_stim_r_choice_r_f1": "stimOn_times",
        "act_block_stim_l_choice_l_f1": "stimOn_times",
        "act_block_stim_l_choice_r_f2": "stimOn_times",
        "act_block_stim_r_choice_l_f2": "stimOn_times",
        "act_block_duringstim_r_choice_r_f1": "stimOn_times",
        "act_block_duringstim_l_choice_l_f1": "stimOn_times",
        "act_block_duringstim_l_choice_r_f2": "stimOn_times",
        "act_block_duringstim_r_choice_l_f2": "stimOn_times",
        "act_block_stim_r_duringchoice_r_f1": "firstMovement_times",
        "act_block_stim_l_duringchoice_l_f1": "firstMovement_times",
        "act_block_stim_l_duringchoice_r_f2": "firstMovement_times",
        "act_block_stim_r_duringchoice_l_f2": "firstMovement_times",
        "act_block_only": "stimOn_times",
    }
    align = {}
    pre_post = {}
    for split in align_old:
        if "durings" in split:
            align[split] = "stimOn_times"
            pre_post[split] = [0, 0.15]
        elif "duringc" in split:
            align[split] = "firstMovement_times"
            pre_post[split] = [0.15, 0]
        else:
            align[split] = align_old[split]
            pre_post[split] = [0.4, -0.1]

    extra_splits = [
        "stim_choice_r_block_r",
        "stim_choice_l_block_l",
        "stim_choice_r_block_l",
        "stim_choice_l_block_r",
        "choice_stim_r_block_r",
        "choice_stim_l_block_l",
        "choice_stim_r_block_l",
        "choice_stim_l_block_r",
        "stim_duringchoice_r_block_r",
        "stim_duringchoice_l_block_l",
        "stim_duringchoice_r_block_l",
        "stim_duringchoice_l_block_r",
        "choice_duringstim_r_block_r",
        "choice_duringstim_l_block_l",
        "choice_duringstim_r_block_l",
        "choice_duringstim_l_block_r",
        "stim_block_l",
        "stim_block_r",
    ]
    for base in extra_splits:
        for suffix in ("", "_act", "_short", "_short_act"):
            split = f"{base}{suffix}"
            if "duringchoice" in split and "choice_during" not in split:
                align[split] = "firstMovement_times"
            elif "choice_duringchoice" in split or (
                split.startswith("choice_") and "duringstim" not in split
            ):
                align[split] = "firstMovement_times"
            else:
                align[split] = "stimOn_times"
            if "short" in split:
                pre_post[split] = [0, 0.15]
            elif "duringchoice" in split or (
                split.startswith("choice_") and "duringstim" not in split
            ):
                pre_post[split] = [0.15, 0]
            elif "block" in split:
                pre_post[split] = [0.4, -0.1]
            else:
                pre_post[split] = [0, 0.15]
    return align, pre_post


ALIGN, PRE_POST = build_align_pre_post()


def action_kernel_priors(alpha, actions):
    prior = 0.5
    priors = [prior]
    for t in range(len(actions) - 1):
        action = actions[t]
        prior = alpha * int(action > 0) + (1 - alpha) * prior
        priors.append(prior)
    binary_priors = np.double(list(np.double(priors) >= 0.5))
    binary_priors = binary_priors * 0.6 + 0.2
    return priors, binary_priors


def p_subjective_probability_left(p_trace):
    """
  Binarize model P population to 0.8/0.2 probabilityLeft for trial grouping.

  Matches model_functions prior_distance helpers: sp = +1 if mean(P_L - P_R) < 0.
  """
    pdiff = float(np.mean(p_trace[:, 0] - p_trace[:, 1]))
    return 0.2 if pdiff < 0 else 0.8


def prior_column_for_split(split):
    """
  Prior column for high/low trial groups in simulated data.

  Uses binarized model P population (mean P_L - P_R per trial), matching
  model_functions.prior_distance_* subjective-prior convention.
  """
    return PRIOR_COLUMN


def sample_num_trials_per_block(rng):
    """One block length from Geometric(p=1/60), clamped to [20, 100] (create_stimuli)."""
    while True:
        sample = int(rng.geometric(p=mf.trials_per_block_param))
        if mf.min_trials_per_block <= sample <= mf.max_trials_per_block:
            return sample


def blocks_for_min_trials(min_trials, rng):
    """Number of blocks needed so expected trial count exceeds min_trials."""
    total = 0
    n_blocks = 0
    while total < min_trials:
        total += sample_num_trials_per_block(rng)
        n_blocks += 1
    return n_blocks, total


def load_fitted_model(g_s=0.0, d_s=0.0, json_path=None):
    json_path = Path(json_path) if json_path is not None else WEIGHTS_JSON
    with open(json_path) as f:
        meta = json.load(f)

    mp = deepcopy(mf.model_params)
    mp.update(meta.get("model_params", {}))
    mp.update(meta["W"])
    mp["g_i"] = meta["g"]["g_i"]
    mp["g_m"] = meta["g"]["g_m"]
    mp["d_i"] = meta["d"]["d_i"]
    mp["d_m"] = meta["d"]["d_m"]
    mp["g_s"] = float(g_s)
    mp["d_s"] = float(d_s)
    theta_c = meta["theta"]["theta_c"]
    theta_d = meta["theta"]["theta_d"]
    mp["action_thresholds"] = {
        "concordant": {c: theta_c for c in [1.0, 0.25, 0.125, 0.0625, 0.0]},
        "discordant": {c: theta_d for c in [1.0, 0.25, 0.125, 0.0625, 0.0]},
    }
    mp["dt"] = DT_MS
    mf._update_model_params_for_dt(mp, DT_MS)
    return mp, meta


def simulate_session(model_params, blocks_per_session, rng, max_obs_per_trial):
    steps_before_obs = int(mf.STEPS_BEFORE_OBS_DURATION_MS / DT_MS)
    stimuli, trial_strengths, _, trial_sides, block_sides = mf.create_stimuli(
        blocks_per_session,
        mf.trials_per_block_param,
        mf.block_side_probs,
        mf.num_stimulus_strength,
        mf.min_stimulus_strength,
        mf.max_stimulus_strength,
        mf.min_trials_per_block,
        mf.max_trials_per_block,
        max_obs_per_trial,
        steps_before_obs,
        rng=rng,
        **model_params,
    )
    results = mf.run_model(
        "data",
        stimuli,
        trial_strengths,
        trial_sides,
        block_sides,
        blocks_per_session,
        steps_before_obs=steps_before_obs,
        verbose=False,
        backend="auto",
        **model_params,
    )
    return results, steps_before_obs


def extract_trial_table(results, steps_before_obs):
    """Build per-trial metadata table aligned with IBL conventions."""
    rows = []
    pops = {k: np.asarray(results[k], float) for k in ("S", "I", "P", "M")}
    n_trials = len(results["trial_sides"])
    lens = [len(results["trial_sides"][i]) for i in range(n_trials)]
    offsets = np.cumsum([0] + lens[:-1])

    for i in range(n_trials):
        m_i = lens[i]
        off = offsets[i]
        trial_side = int(np.sign(results["trial_sides"][i][0])) or 1
        block_side = int(np.sign(results["block_sides"][i][0]))
        pleft = 0.8 if block_side == -1 else 0.2
        contrast = float(abs(results["trial_strengths"][i][0]))
        model_choice = int(results["choices"][i]) if i < len(results["choices"]) else 0
        ibl_choice = -model_choice if model_choice != 0 else 0
        correct = int(results["correct_action_taken"][i]) if i < len(results["correct_action_taken"]) else 0
        feedback = 1 if correct else -1
        rt = int(results["reaction_time"][i]) if i < len(results["reaction_time"]) else m_i - steps_before_obs

        cl = contrast if trial_side == -1 else np.nan
        cr = contrast if trial_side == 1 else np.nan

        p_trace = pops["P"][off : off + m_i]
        rows.append(
            {
                "trial_idx": i,
                "offset": off,
                "length": m_i,
                "trial_side": trial_side,
                "block_side": block_side,
                "probabilityLeft": pleft,
                "p_subjective_probabilityLeft": p_subjective_probability_left(p_trace),
                "contrastLeft": cl,
                "contrastRight": cr,
                "choice": ibl_choice,
                "feedbackType": feedback,
                "reaction_time": rt,
                "correct": correct,
                "traces": {k: pops[k][off : off + m_i].copy() for k in pops},
            }
        )

    return pd.DataFrame(rows)


def apply_act_prior(df):
    """Add action-kernel prior; keep true block prior in probabilityLeft (as in get_d_vars)."""
    actions = [max(0, c) for c in df["choice"].values]
    _, act_binary = action_kernel_priors(ALPHA_ACT, actions)
    df = df.copy()
    df["true_probabilityLeft"] = df["probabilityLeft"].values
    df["act_probabilityLeft"] = act_binary
    return df


def split_n_bins(split):
    pre, post = PRE_POST[split]
    duration = pre + (abs(post) if post < 0 else post)
    n_coarse = max(1, int(round(duration / B_SIZE)))
    st = max(1, int(B_SIZE // STS))
    return n_coarse * st


def window_step_bounds(align_kind, trial_len, rt, steps_before_obs, pre, post):
    if align_kind == "stimOn_times":
        anchor = steps_before_obs
    else:
        anchor = steps_before_obs + max(1, rt)
    pre_steps = int(round(pre * 1000.0 / DT_MS))
    if post < 0:
        post_steps = -int(round(abs(post) * 1000.0 / DT_MS))
        start = anchor - pre_steps
        end = anchor + post_steps
    else:
        post_steps = int(round(post * 1000.0 / DT_MS))
        start = anchor - pre_steps
        end = anchor + post_steps
    start = max(0, start)
    end = min(trial_len, end)
    if end <= start:
        return None
    return start, end


def bin_trace_segment(trace_2d, start, end, n_bins):
    """Average 2-channel trace into n_bins (simple uniform binning)."""
    seg = trace_2d[start:end]
    if seg.size == 0:
        return np.zeros((n_bins, 2))
    edges = np.linspace(0, seg.shape[0], n_bins + 1).astype(int)
    out = np.zeros((n_bins, 2))
    for b in range(n_bins):
        sl = seg[edges[b] : edges[b + 1]]
        out[b] = sl.mean(axis=0) if sl.size else 0.0
    return out


def trial_masks_for_split(df, split):
    """
  Return boolean masks for high- vs low-prior trial groups (pleft >= 0.5 vs < 0.5).

  All populations use model P subjective prior (p_subjective_probabilityLeft).
  """
    pcol = prior_column_for_split(split)
    if pcol not in df.columns:
        raise KeyError(f"Missing prior column {pcol!r} for split {split!r}")

    m = np.ones(len(df), dtype=bool)

    if "block_only" in split or "act_block_only" == split:
        pass
    elif "stim_block_l" in split:
        m &= ~np.isnan(df["contrastLeft"].values)
    elif "stim_block_r" in split:
        m &= ~np.isnan(df["contrastRight"].values)
    elif "stim_l" in split:
        m &= ~np.isnan(df["contrastLeft"].values)
        if "choice_l" in split and "f1" in split:
            m &= df["choice"].values == 1
            m &= df["feedbackType"].values == 1
        elif "choice_r" in split and "f2" in split:
            m &= df["choice"].values == -1
            m &= df["feedbackType"].values == -1
    elif "stim_r" in split:
        m &= ~np.isnan(df["contrastRight"].values)
        if "choice_r" in split and "f1" in split:
            m &= df["choice"].values == -1
            m &= df["feedbackType"].values == 1
        elif "choice_l" in split and "f2" in split:
            m &= df["choice"].values == 1
            m &= df["feedbackType"].values == -1
    elif split.startswith("stim_choice_") or split.startswith("stim_duringchoice_"):
        if "choice_r" in split:
            m &= df["choice"].values == -1
        if "choice_l" in split:
            m &= df["choice"].values == 1
        if "_block_r" in split or split.endswith("block_r"):
            m &= df["trial_side"].values == 1
        if "_block_l" in split or split.endswith("block_l"):
            m &= df["trial_side"].values == -1
    elif split.startswith("choice_") or split.startswith("choice_duringstim_"):
        if "stim_r" in split:
            m &= ~np.isnan(df["contrastRight"].values)
        if "stim_l" in split:
            m &= ~np.isnan(df["contrastLeft"].values)
        if "choice_r" in split:
            m &= df["choice"].values == -1
        if "choice_l" in split:
            m &= df["choice"].values == 1
    else:
        raise ValueError(f"Unrecognized split: {split}")

    cond0 = m & (df[pcol].values >= 0.5)
    cond1 = m & (df[pcol].values < 0.5)
    return cond0, cond1


def build_population_b_for_split(df, split, population, steps_before_obs):
    """Per-trial binned model trajectories for one population: (trials, 2, bins)."""
    align_kind = ALIGN.get(split, "stimOn_times")
    pre, post = PRE_POST[split]
    n_bins = split_n_bins(split)
    cond0, cond1 = trial_masks_for_split(df, split)
    idx0 = np.where(cond0)[0]
    idx1 = np.where(cond1)[0]
    if len(idx0) < 2 or len(idx1) < 2:
        return None

    def trials_to_stack(idxs):
        chunks = []
        for ti in idxs:
            row = df.iloc[ti]
            bounds = window_step_bounds(
                align_kind, row["length"], row["reaction_time"], steps_before_obs, pre, post
            )
            if bounds is None:
                continue
            s, e = bounds
            trace = row["traces"][population]
            seg = bin_trace_segment(trace, s, e, n_bins)
            chunks.append(seg.T)
        if not chunks:
            return None
        return np.stack(chunks, axis=0)

    b0 = trials_to_stack(idx0)
    b1 = trials_to_stack(idx1)
    if b0 is None or b1 is None:
        return None
    return np.concatenate([b0, b1], axis=0), len(b0)


def _null_shuffle_executor(n_workers):
    """Process pool on Linux (fork); threads on macOS (spawn re-imports this script)."""
    import multiprocessing as mp

    n_workers = max(1, int(n_workers))
    if sys.platform == "darwin":
        return ThreadPoolExecutor(max_workers=n_workers)
    try:
        ctx = mp.get_context("fork")
        return ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)
    except (ValueError, AttributeError):
        return ThreadPoolExecutor(max_workers=n_workers)


def compute_population_distances(b, n0, nrand, rng, n_jobs=1):
    """
    Euclidean prior-distance curves for one population.

    Nulls: shuffle high/low block labels before trial averaging (same logic as get_d_vars).
    b shape: (n_trials, n_channels, n_bins).
    """
    ntr = b.shape[0]
    if ntr < 4 or n0 < 2 or (ntr - n0) < 2:
        return None

    ys_true = np.zeros(ntr, dtype=bool)
    ys_true[:n0] = True
    means = [b[:n0].mean(axis=0), b[n0:].mean(axis=0)]

    n_jobs = max(1, int(n_jobs))
    if n_jobs > 1 and nrand >= n_jobs:
        from null_shuffle_worker import null_shuffle_chunk

        chunk = int(np.ceil(nrand / n_jobs))
        tasks = []
        start = 0
        while start < nrand:
            n_chunk = min(chunk, nrand - start)
            tasks.append((n_chunk, int(rng.randint(0, 2**31 - 1))))
            start += n_chunk
        with _null_shuffle_executor(min(n_jobs, len(tasks))) as pool:
            futures = [
                pool.submit(null_shuffle_chunk, b, n0, ys_true, nc, seed)
                for nc, seed in tasks
            ]
            for fut in futures:
                means.extend(fut.result())
    else:
        for _ in range(nrand):
            perm = ys_true.copy()
            rng.shuffle(perm)
            means.append(b[perm].mean(axis=0))
            means.append(b[~perm].mean(axis=0))

    means_arr = np.stack(means, axis=0)
    diff = means_arr[0::2] - means_arr[1::2]
    d_eucs = np.sum(diff**2, axis=-1)
    ws = means_arr[:2]
    return d_eucs, ws


def _metrics_from_regde(regde_curves, split):
    """Split-level metrics from real + null distance curves (d_var_stacked core)."""
    curves = np.asarray(regde_curves, dtype=float)
    ampse = [np.max(x) - np.min(x) for x in curves]
    pre, post = PRE_POST[split]
    d_euc = curves[0] - np.mean(curves[1:], axis=0)
    d_euc = d_euc - np.min(d_euc)
    loc = np.where(d_euc > 0.7 * np.max(d_euc))[0]
    return {
        "nclus": 2,
        "p_euc": float(np.mean(np.array(ampse) >= ampse[0])),
        "d_euc": d_euc,
        "amp_euc": float(np.max(d_euc)),
        "lat_euc": (
            float(np.linspace(-pre, post if post > 0 else -abs(post), len(d_euc))[loc[0]])
            if len(loc)
            else np.nan
        ),
        "p_gain": np.nan,
        "p_offset": np.nan,
        "p_gain_effect": np.nan,
        "p_offset_effect": np.nan,
        "p_xnobis": np.nan,
        "d_xnobis": None,
        "amp_xnobis": np.nan,
    }


def build_split_results(df, split, steps_before_obs, nrand, rng, populations=None, n_jobs=1):
    """Build {split}.npy / {split}_regde.npy outputs from pooled model trajectories."""
    populations = populations or MODEL_POPULATIONS
    regde = {}
    regxn = {}
    r = {}
    for pop in populations:
        built = build_population_b_for_split(df, split, pop, steps_before_obs)
        if built is None:
            continue
        b, n0 = built
        dist = compute_population_distances(b, n0, nrand, rng, n_jobs=n_jobs)
        if dist is None:
            continue
        d_eucs, ws = dist
        regde[pop] = np.asarray(d_eucs, dtype=float) / B_SIZE
        regxn[pop] = None
        res = _metrics_from_regde(regde[pop], split)
        res["ws"] = ws
        r[pop] = res
    if not r:
        return None
    return r, regde, regxn


def s_prior_splits():
    """Splits for S act_block_duringstim prior distance (subjective prior grouping)."""
    import analysis_functions as af

    return list(af.run_align[S_PRIOR_TIMEFRAME])


def build_res_from_trajectories(
    session_dfs, splits, steps_before_obs, nrand, rng, pth_res, populations=None, n_jobs=1
):
    """Write per-split res files by pooling trials across simulated sessions."""
    pth_res.mkdir(parents=True, exist_ok=True)
    all_df = pd.concat(session_dfs, ignore_index=True)
    n_saved = 0
    t0 = time.perf_counter()
    for i, split in enumerate(splits):
        out = build_split_results(
            all_df, split, steps_before_obs, nrand, rng, populations=populations, n_jobs=n_jobs
        )
        if out is None:
            continue
        r, regde, regxn = out
        np.save(pth_res / f"{split}_regde.npy", regde, allow_pickle=True)
        np.save(pth_res / f"{split}_regxn.npy", regxn, allow_pickle=True)
        np.save(pth_res / f"{split}.npy", r, allow_pickle=True)
        n_saved += 1
        if (i + 1) % max(1, len(splits) // 4) == 0 or i + 1 == len(splits):
            print(f"    split {i + 1}/{len(splits)} ({split}) — {time.perf_counter() - t0:.1f}s")
    return n_saved


def collect_all_splits():
    import analysis_functions as af

    splits = set()
    for tf in FOCUSED_TIMEFRAMES:
        splits.update(af.run_align.get(tf, []))
    return sorted(splits)


def stack_combined_timeframes(pth_res, timeframes):
    """Build combined_{splits}.npy and combined_regde_{splits}.npy per timeframe."""
    import analysis_functions as af

    for timeframe in timeframes:
        splits = af.run_align.get(timeframe, [])
        if not splits:
            continue
        combined_regde = {}
        combined_regxn = {}
        for split in splits:
            regde_path = pth_res / f"{split}_regde.npy"
            regxn_path = pth_res / f"{split}_regxn.npy"
            if not regde_path.exists():
                continue
            split_regde = np.load(regde_path, allow_pickle=True).item()
            split_regxn = np.load(regxn_path, allow_pickle=True).item() if regxn_path.exists() else {}
            for reg, curves in split_regde.items():
                if reg not in combined_regde:
                    combined_regde[reg] = [curves[0], np.array(curves[1:])]
                else:
                    combined_regde[reg][0] = combined_regde[reg][0] + curves[0]
                    combined_regde[reg][1] = combined_regde[reg][1] + np.array(curves[1:])
            for reg, curves in split_regxn.items():
                if curves is None:
                    continue
                if reg not in combined_regxn:
                    combined_regxn[reg] = [curves[0], np.array(curves[1:]) if len(curves) > 1 else np.empty((0, len(curves[0])))]
                else:
                    combined_regxn[reg][0] = combined_regxn[reg][0] + curves[0]
                    if len(curves) > 1:
                        combined_regxn[reg][1] = combined_regxn[reg][1] + np.array(curves[1:])

        if not combined_regde:
            continue

        r = {}
        pre0, post0 = PRE_POST.get(splits[0], [0.4, -0.1])
        for reg, (sum_real_curve, control_curves) in combined_regde.items():
            amp_real = np.max(sum_real_curve) - np.min(sum_real_curve)
            amp_controls = [np.max(c) - np.min(c) for c in control_curves]
            p_euc = float(np.mean(np.array(amp_controls) >= amp_real))
            d_euc = sum_real_curve - np.min(sum_real_curve)
            amp_euc = float(np.max(d_euc))
            loc = np.where(d_euc > 0.7 * amp_euc)[0]
            lat_euc = (
                float(np.linspace(-pre0, post0 if post0 > 0 else -abs(post0), len(d_euc))[loc[0]])
                if len(loc)
                else np.nan
            )
            res = {
                "d_euc": d_euc,
                "amp_euc": amp_euc,
                "p_euc": p_euc,
                "lat_euc": lat_euc,
                "p_gain": np.nan,
                "p_offset": np.nan,
                "p_gain_effect": np.nan,
                "p_offset_effect": np.nan,
                "p_xnobis": np.nan,
                "amp_xnobis": np.nan,
            }
            if reg in combined_regxn:
                xn_real, xn_ctrl = combined_regxn[reg]
                amp_x_real = np.max(xn_real) - np.min(xn_real)
                amp_x_ctrl = [np.max(c) - np.min(c) for c in xn_ctrl] if xn_ctrl.size else []
                p_x = float(np.mean(np.array(amp_x_ctrl) >= amp_x_real)) if len(amp_x_ctrl) else np.nan
                d_x = xn_real - np.min(xn_real)
                res.update(
                    {
                        "d_xnobis": d_x,
                        "amp_xnobis": float(np.max(d_x)),
                        "p_xnobis": p_x,
                    }
                )
            r[reg] = res

        combined_name = splits[0] if len(splits) == 1 else "combined_" + "_".join(splits)
        regde_name = (
            f"{combined_name}_regde"
            if len(splits) == 1
            else f"combined_regde_{'_'.join(splits)}"
        )
        regxn_name = (
            f"{combined_name}_regxn"
            if len(splits) == 1
            else f"combined_regxn_{'_'.join(splits)}"
        )
        np.save(pth_res / f"{combined_name}.npy", r, allow_pickle=True)
        np.save(pth_res / f"{regde_name}.npy", combined_regde, allow_pickle=True)
        np.save(pth_res / f"{regxn_name}.npy", combined_regxn, allow_pickle=True)


def setup_sim_meta(pth_res):
    meta = pth_res.parent / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "region_order.txt").write_text("\n".join(MODEL_POPULATIONS) + "\n")
    return meta


def run_analysis(pth_res, alpha=0.01, ptype="p_mean"):
    import analysis_functions as af

    af.pth_res = pth_res
    af.meta_pth = pth_res.parent / "meta"
    af.meta_pth.mkdir(parents=True, exist_ok=True)
    setup_sim_meta(pth_res)
    pd.DataFrame({"region": list(MODEL_POPULATIONS)}).to_csv(
        pth_res / "act_block_only.csv", index=False
    )

    # Disable atlas lookup / plotting in classification helpers.
    af.plot_table_with_styles = lambda **kwargs: None
    af.swanson_to_beryl_hex = lambda region, br=None: "#cccccc"

    for timeframe in FOCUSED_TIMEFRAMES:
        if timeframe not in af.run_align:
            continue
        try:
            af.compute_p_value(timeframe, ptype=ptype, alpha=alpha, dist="de")
            af.compute_p_value(timeframe, ptype="p_offset", alpha=alpha, dist="de")
            af.fdr_combined(timeframe, ptype=ptype, sigl=alpha)
            af.fdr_combined(timeframe, ptype="p_offset", sigl=alpha)
        except Exception as exc:
            print(f"[warn] p-value step failed for {timeframe}: {exc}")

    for timeframe in FOCUSED_TIMEFRAMES:
        if timeframe not in af.run_align:
            continue
        try:
            af.compute_amp_slope(timeframe, n=72)
        except Exception as exc:
            print(f"[warn] amp_slope failed for {timeframe}: {exc}")

    sc_table = af.get_sc_table(
        SC_TIMES,
        ptype=f"{ptype}_c",
        alpha=alpha,
        combined_p=True,
        sc_threshold=0.0,
        n=72,
    )
    return sc_table, af


def _combined_names(timeframe):
    import analysis_functions as af

    splits = af.run_align[timeframe]
    combined = splits[0] if len(splits) == 1 else "combined_" + "_".join(splits)
    regde = (
        f"{combined}_regde"
        if len(splits) == 1
        else f"combined_regde_{'_'.join(splits)}"
    )
    return combined, regde, splits


def _load_combined_results(pth_res, combined):
    """Load combined split results from .npy or exported .csv."""
    npy_path = pth_res / f"{combined}.npy"
    if npy_path.exists():
        return np.load(npy_path, allow_pickle=True).flat[0]
    csv_path = pth_res / f"{combined}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path).set_index("region")
    return {
        reg: {
            "amp_euc": float(row["amp_euc"]),
            "p_euc": np.nan,
            "lat_euc": float(row.get("lat_euc", np.nan)),
            "p_mean": float(row.get("p_mean_c", np.nan)),
            "d_euc": None,
        }
        for reg, row in df.iterrows()
    }


def load_population_features(pth_res):
    """Amp/d_euc features per population across focused timeframes."""
    timeframes = SC_TIMES + TIMING_SPLITS
    features = {pop: {} for pop in MODEL_POPULATIONS}
    for tf in timeframes:
        combined, regde_name, _ = _combined_names(tf)
        d = _load_combined_results(pth_res, combined)
        if d is None:
            continue
        regde_path = pth_res / f"{regde_name}.npy"
        regde = np.load(regde_path, allow_pickle=True).item() if regde_path.exists() else {}
        for pop in MODEL_POPULATIONS:
            if pop not in d:
                continue
            entry = {
                "amp_euc": float(d[pop]["amp_euc"]),
                "p_euc": float(d[pop].get("p_euc", np.nan)),
                "lat_euc": float(d[pop].get("lat_euc", np.nan)),
            }
            if pop in regde:
                real, nulls = regde[pop]
                nulls = np.atleast_2d(np.asarray(nulls, dtype=float))
                real = np.asarray(real, dtype=float)
                entry["prior_real"] = real
                entry["prior_null_mean"] = np.mean(nulls, axis=0)
            features[pop][tf] = entry
    return features


def classify_populations(features):
    """
    Population-specific classifier for the three circuit populations.

    Uses competitive assignment on timing-specific signatures:
    - M: max choice_duringchoice_act (movement-aligned coding)
    - S: max act_block_duringstim among remaining (sensory block prior)
    - I: remaining population (integrator)
    """
    rows = []
    recovered = {t: [] for t in ("S", "I", "M")}
    sim_pops = [p for p in MODEL_POPULATIONS if p != "P"]

    block_stim = {
        p: features[p].get("act_block_duringstim", {}).get("amp_euc", 0.0) for p in sim_pops
    }
    choice_dc = {
        p: features[p].get("choice_duringchoice_act", {}).get("amp_euc", 0.0) for p in sim_pops
    }

    m_pop = max(sim_pops, key=lambda p: choice_dc[p])
    remaining = [p for p in sim_pops if p != m_pop]
    s_pop = max(remaining, key=lambda p: block_stim[p])
    i_pop = next(p for p in remaining if p != s_pop)

    pred_map = {m_pop: "M", s_pop: "S", i_pop: "I", "P": "P"}

    for pop in MODEL_POPULATIONS:
        pred = pred_map[pop]
        true_t = POPULATION_TYPE[pop]
        if pop in recovered and pred in recovered:
            recovered[pred].append(pop)
        rows.append(
            {
                "region": pop,
                "true": true_t,
                "pred": pred,
                "correct": pred == true_t,
                "block_stim_amp": float(block_stim.get(pop, np.nan)),
                "choice_duringchoice_amp": float(choice_dc.get(pop, np.nan)),
            }
        )

    df = pd.DataFrame(rows)
    sim = df[df["true"].isin(["S", "I", "M"])]
    acc = float(sim["correct"].mean()) if len(sim) else 0.0
    cm = pd.crosstab(sim["true"], sim["pred"], dropna=False) if len(sim) else pd.DataFrame()
    return recovered, df, cm, acc


def _prior_offset_stats(real, nulls, n_early=PRIOR_OFFSET_BINS):
    """Early-bin offset test (matches analysis_functions p_offset logic)."""
    n_early = min(n_early, len(real))
    if n_early < 1:
        return {"p_offset": np.nan, "offset_effect": np.nan, "offset_mean": np.nan}
    real_early = float(np.mean(real[:n_early]))
    null_early = np.array([np.mean(n[:n_early]) for n in nulls])
    p_offset = float(np.mean(null_early >= real_early))
    offset_effect = real_early - float(np.mean(null_early))
    return {
        "p_offset": p_offset,
        "offset_effect": offset_effect,
        "offset_mean": real_early,
    }


def _prior_gain_stats(real, nulls, alpha=0.01, n_early=PRIOR_OFFSET_BINS):
    """
    Gain test after early-offset removal (matches analysis_functions p_gain logic).

    If p_offset < alpha, subtract early offset from the real curve, then compare
    mean of bins 4+ against label-shuffle nulls on the same offset-corrected bins.
    """
    real = np.asarray(real, dtype=float)
    nulls = np.atleast_2d(np.asarray(nulls, dtype=float))
    n_early = min(n_early, real.shape[0])
    if n_early < 1 or real.shape[0] <= 4:
        return {
            "p_gain": np.nan,
            "gain_effect": np.nan,
            "gain_late_mean": np.nan,
            "gain_offset_subtracted": np.nan,
        }

    stacked = np.vstack([real.reshape(1, -1), nulls])
    mean_first5 = np.mean(stacked[:, :n_early], axis=1)
    p_offset = float(np.mean(mean_first5 >= mean_first5[0]))
    offset = (
        float(mean_first5[0] - np.mean(mean_first5[1:]))
        if p_offset < alpha
        else 0.0
    )
    shifted = real - offset
    gain_late_mean = float(np.mean(shifted[4:]))

    null_gain_means = []
    for n in nulls:
        n_shifted = n - offset
        if len(n_shifted) > 4:
            null_gain_means.append(float(np.mean(n_shifted[4:])))
    if not null_gain_means:
        p_gain = np.nan
        gain_effect = np.nan
    else:
        p_gain = float(np.mean(np.array(null_gain_means) >= gain_late_mean))
        gain_effect = gain_late_mean - float(np.mean(null_gain_means))

    return {
        "p_gain": p_gain,
        "gain_effect": gain_effect,
        "gain_late_mean": gain_late_mean,
        "gain_offset_subtracted": offset,
    }


def _gain_late_mean_for_curve(curve, nulls, alpha=0.01, n_early=PRIOR_OFFSET_BINS):
    """Mean of bins 4+ after p_gain-style early offset removal for one curve."""
    return _prior_gain_stats(curve, nulls, alpha=alpha, n_early=n_early)["gain_late_mean"]


def s_only_prior_test(pth_res, alpha=0.01, timeframe="act_block_duringstim"):
    """
    S-only block prior test using act_block_duringstim curves (no FDR across populations).
    """
    combined, regde_name, splits = _combined_names(timeframe)
    regde_path = pth_res / f"{regde_name}.npy"
    d = _load_combined_results(pth_res, combined)
    if not regde_path.exists() or d is None:
        return None

    regde = np.load(regde_path, allow_pickle=True).item()
    if "S" not in regde or "S" not in d:
        return None

    real, nulls = regde["S"]
    nulls = np.atleast_2d(np.asarray(nulls, dtype=float))
    real = np.asarray(real, dtype=float)
    real_mean = float(np.mean(real))
    null_means = np.mean(nulls, axis=1)
    p_mean = float(np.mean(null_means >= real_mean))
    amp_real = float(np.max(real) - np.min(real))
    amp_null = np.array([np.max(n) - np.min(n) for n in nulls])
    p_amp = float(np.mean(amp_null >= amp_real))

    pre, post = PRE_POST[splits[0]]
    t_axis = np.linspace(-pre, post if post > 0 else -abs(post), len(real))
    offset = _prior_offset_stats(real, nulls)
    gain = _prior_gain_stats(real, nulls, alpha=alpha)

    return {
        "population": "S",
        "timeframe": timeframe,
        "p_mean": p_mean,
        "p_amp": p_amp,
        "p_offset": offset["p_offset"],
        "p_gain": gain["p_gain"],
        "offset_effect": offset["offset_effect"],
        "gain_effect": gain["gain_effect"],
        "offset_mean": offset["offset_mean"],
        "gain_late_mean": gain["gain_late_mean"],
        "amp_euc": float(d["S"]["amp_euc"]),
        "curve_mean": real_mean,
        "curve_amp": amp_real,
        "significant_p_mean": p_mean < alpha,
        "significant_p_amp": p_amp < alpha,
        "significant_p_offset": offset["p_offset"] < alpha,
        "significant_p_gain": gain["p_gain"] < alpha if not np.isnan(gain["p_gain"]) else False,
        "curve_real": real,
        "null_curves": nulls,
        "curve_null_mean": np.mean(nulls, axis=0),
        "t_axis": t_axis,
    }


def s_prior_presence_vs_absence_pvalues(abs_s, pres_s, alpha=0.01):
    """
    Cross-condition p-values for S prior curves (presence vs absence).

    Uses paired label-shuffle null curves: for each permutation index, compares
    presence vs absence on curve mean, early-bin mean, and p_gain late-bin mean.
  """
    if abs_s is None or pres_s is None:
        return None

    nulls_abs = np.atleast_2d(np.asarray(abs_s["null_curves"], dtype=float))
    nulls_pres = np.atleast_2d(np.asarray(pres_s["null_curves"], dtype=float))
    n = min(nulls_abs.shape[0], nulls_pres.shape[0])
    if n == 0:
        return None

    real_mean_diff = float(pres_s["curve_mean"] - abs_s["curve_mean"])
    real_amp_diff = float(pres_s["curve_amp"] - abs_s["curve_amp"])
    real_offset_diff = float(pres_s["offset_mean"] - abs_s["offset_mean"])

    null_mean_diffs = np.array(
        [np.mean(nulls_pres[i]) - np.mean(nulls_abs[i]) for i in range(n)]
    )
    null_amp_diffs = np.array(
        [
            (np.max(nulls_pres[i]) - np.min(nulls_pres[i]))
            - (np.max(nulls_abs[i]) - np.min(nulls_abs[i]))
            for i in range(n)
        ]
    )
    n_early = min(PRIOR_OFFSET_BINS, nulls_abs.shape[1], nulls_pres.shape[1])
    null_offset_diffs = np.array(
        [
            np.mean(nulls_pres[i, :n_early]) - np.mean(nulls_abs[i, :n_early])
            for i in range(n)
        ]
    )

    real_gain_diff = float(
        _gain_late_mean_for_curve(pres_s["curve_real"], nulls_pres, alpha=alpha)
        - _gain_late_mean_for_curve(abs_s["curve_real"], nulls_abs, alpha=alpha)
    )
    null_gain_diffs = np.array(
        [
            _gain_late_mean_for_curve(nulls_pres[i], nulls_pres, alpha=alpha)
            - _gain_late_mean_for_curve(nulls_abs[i], nulls_abs, alpha=alpha)
            for i in range(n)
        ]
    )

    p_mean_diff = float(np.mean(null_mean_diffs >= real_mean_diff))
    p_amp_diff = float(np.mean(null_amp_diffs >= real_amp_diff))
    p_offset_diff = float(np.mean(null_offset_diffs >= real_offset_diff))
    p_gain_diff = float(np.mean(null_gain_diffs >= real_gain_diff))

    return {
        "real_mean_diff": real_mean_diff,
        "real_amp_diff": real_amp_diff,
        "real_offset_diff": real_offset_diff,
        "real_gain_diff": real_gain_diff,
        "p_mean_pres_vs_abs": p_mean_diff,
        "p_amp_pres_vs_abs": p_amp_diff,
        "p_offset_pres_vs_abs": p_offset_diff,
        "p_gain_pres_vs_abs": p_gain_diff,
        "null_mean_diffs": null_mean_diffs,
        "null_amp_diffs": null_amp_diffs,
        "null_offset_diffs": null_offset_diffs,
        "null_gain_diffs": null_gain_diffs,
    }


def population_prior_tests(pth_res, alpha=0.01):
    """Per-population act_block_duringstim prior tests (uncorrected)."""
    rows = []
    for pop in ("S", "I", "M"):
        combined, regde_name, _ = _combined_names("act_block_duringstim")
        regde_path = pth_res / f"{regde_name}.npy"
        res_path = pth_res / f"{combined}.npy"
        if not regde_path.exists():
            continue
        regde = np.load(regde_path, allow_pickle=True).item()
        d = np.load(res_path, allow_pickle=True).flat[0]
        if pop not in regde:
            continue
        real, nulls = regde[pop]
        nulls = np.atleast_2d(np.asarray(nulls, dtype=float))
        real = np.asarray(real, dtype=float)
        p_mean = float(np.mean(np.mean(nulls, axis=1) >= np.mean(real)))
        rows.append(
            {
                "population": pop,
                "group": POPULATION_TYPE[pop],
                "p_mean": p_mean,
                "amp_euc": float(d[pop]["amp_euc"]),
                "significant": p_mean < alpha,
            }
        )
    return pd.DataFrame(rows)


def classify_regions(sc_table, af, alpha=0.01, ptype="p_mean_c"):
    recovered = {t: [] for t in ("S", "I", "M")}
    stim_regs = af.plot_combined_onetype(
        SC_TIMES,
        "stim",
        ["act_block_duringstim"],
        ptype=ptype,
        alpha=alpha,
        combined_p=True,
        n=72,
        sc_threshold=0.0,
    )
    int_regs = af.plot_combined_onetype(
        SC_TIMES,
        "integrator",
        ["act_block_duringstim", "act_block_duringchoice"],
        ptype=ptype,
        alpha=alpha,
        combined_p=True,
        n=72,
        sc_threshold=0.0,
    )
    move_regs = af.plot_combined_onetype(
        SC_TIMES,
        "move",
        ["act_block_duringstim", "act_block_duringchoice"],
        ptype=ptype,
        alpha=alpha,
        combined_p=True,
        n=72,
        sc_threshold=0.0,
    )
    recovered["S"] = stim_regs
    recovered["I"] = int_regs
    recovered["M"] = move_regs
    return recovered


def prior_modulation_table(pth_res, regions_by_type, timeframe="act_block_duringstim", ptype="p_mean_c"):
    import analysis_functions as af

    splits = af.run_align[timeframe]
    combined = "combined_" + "_".join(splits)
    d = np.load(pth_res / f"{combined}.npy", allow_pickle=True).flat[0]
    rows = []
    for gtype, regs in regions_by_type.items():
        for reg in regs:
            if reg not in d:
                continue
            rows.append(
                {
                    "region": reg,
                    "group": gtype,
                    "p_mean": d[reg].get("p_mean", np.nan),
                    "p_mean_c": d[reg].get(ptype, np.nan),
                    "amp_euc": d[reg].get("amp_euc", np.nan),
                    "significant": float(d[reg].get(ptype, 1.0)) <= 0.01,
                }
            )
    return pd.DataFrame(rows)


def recovery_classification_metrics(recovered):
    gt = POPULATION_TYPE
    types = ["S", "I", "M"]
    assign = {}
    for t in types:
        for r in recovered.get(t, []):
            assign[r] = t
    rows = []
    for reg, true_t in gt.items():
        if true_t not in types:
            continue
        pred = assign.get(reg, "none")
        rows.append({"region": reg, "true": true_t, "pred": pred, "correct": pred == true_t})
    df = pd.DataFrame(rows)
    acc = df["correct"].mean() if len(df) else 0.0
    cm = pd.crosstab(df["true"], df["pred"], dropna=False)
    return df, cm, acc


def _draw_shuffle_control_panel(ax, s_prior, condition, n_sample=SHUFFLE_PLOT_N_SAMPLE, rng_seed=0):
    """Draw one shuffle-control panel on ax."""
    real = np.asarray(s_prior["curve_real"], dtype=float)
    nulls = np.atleast_2d(np.asarray(s_prior["null_curves"], dtype=float))
    n_plot = min(n_sample, nulls.shape[0])
    rng = np.random.RandomState(rng_seed)
    idx = rng.choice(nulls.shape[0], size=n_plot, replace=False)

    t = s_prior["t_axis"]
    for j in idx:
        ax.plot(t, nulls[j], color="0.78", lw=0.7, alpha=0.28, zorder=1)
    ax.plot(
        t,
        s_prior["curve_null_mean"],
        color="#4C72B0",
        ls="--",
        lw=1.4,
        alpha=0.9,
        label="null mean",
        zorder=8,
    )
    ax.plot(t, real, color="#C44E52", lw=2.5, label="true labels", zorder=10)
    if len(t) >= PRIOR_OFFSET_BINS:
        ax.axvspan(t[0], t[PRIOR_OFFSET_BINS - 1], color="gray", alpha=0.1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("S prior distance (raw)")
    ax.set_title(
        f"{condition}\n"
        f"true curve vs {n_plot} sampled nulls (of {nulls.shape[0]} shuffles)"
    )
    p_gain = s_prior.get("p_gain", np.nan)
    ax.text(
        0.98,
        0.98,
        f"p_mean={s_prior['p_mean']:.4f}\n"
        f"p_offset={s_prior['p_offset']:.4f}\n"
        f"p_gain={p_gain:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )
    ax.legend(fontsize=7, loc="upper left")


def plot_s_shuffle_control(condition, s_prior, fig_dir, n_sample=SHUFFLE_PLOT_N_SAMPLE, rng_seed=0):
    """True S prior-distance curve vs sampled label-shuffle null curves."""
    if s_prior is None:
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    _draw_shuffle_control_panel(ax, s_prior, condition, n_sample, rng_seed)
    fig.tight_layout()
    out = fig_dir / "s_shuffle_control.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved shuffle control figure to {out}")


def plot_combined_shuffle_controls(base_dir, abs_s, pres_s, rng_seed=0):
    """Side-by-side absence/presence shuffle-control panels with within-condition p-values."""
    if abs_s is None and pres_s is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    panels = [("absence", abs_s, rng_seed), ("presence", pres_s, rng_seed + 1)]
    for ax, (cond, s_prior, seed) in zip(axes, panels):
        if s_prior is None:
            ax.set_title(f"{cond} (no data)")
            ax.axis("off")
            continue
        _draw_shuffle_control_panel(ax, s_prior, cond, rng_seed=seed)

    fig.suptitle("S prior distance vs label-shuffle nulls (model P subjective prior)", y=1.02)
    fig.tight_layout()
    out = base_dir / "s_shuffle_control_combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined shuffle control figure to {out}")


def plot_s_prior_curve(condition, s_prior, fig_dir):
    """S-only act_block_duringstim prior curve (raw, baseline not removed)."""
    if s_prior is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4))

    t = s_prior["t_axis"]
    real = s_prior["curve_real"]
    null_mean = s_prior["curve_null_mean"]
    ax.plot(t, real, "C0", lw=2, label="S real")
    ax.plot(t, null_mean, "C0", ls="--", alpha=0.6, label="null mean")
    if len(t) >= PRIOR_OFFSET_BINS:
        ax.axvspan(t[0], t[PRIOR_OFFSET_BINS - 1], color="gray", alpha=0.12, label="offset window")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Prior distance (raw)")
    ax.set_title(
        f"S prior curve ({condition}) — "
        f"p_mean={s_prior['p_mean']:.4f}, p_offset={s_prior['p_offset']:.4f}, "
        f"p_gain={s_prior.get('p_gain', np.nan):.4f}"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "s_prior_curve.png", dpi=150)
    plt.close(fig)


def plot_recovery_figures(
    condition,
    pth_res,
    fig_dir,
    class_df,
    cm,
    prior_df,
    regde_stim,
    pop_class_df=None,
    pop_cm=None,
    s_prior=None,
):
    fig_dir.mkdir(parents=True, exist_ok=True)

    if s_prior is not None:
        plot_s_prior_curve(condition, s_prior, fig_dir)
        plot_s_shuffle_control(condition, s_prior, fig_dir)

    # Population-specific confusion matrix (preferred)
    use_cm = pop_cm if pop_cm is not None and len(pop_cm) else cm
    use_df = pop_class_df if pop_class_df is not None and len(pop_class_df) else class_df
    cm_title = "Population classifier" if pop_cm is not None and len(pop_cm) else "BWM classifier"

    fig, ax = plt.subplots(figsize=(5, 4))
    cm_vals = use_cm.reindex(index=["S", "I", "M"], columns=["S", "I", "M", "none"], fill_value=0)
    im = ax.imshow(cm_vals.values, cmap="Blues")
    ax.set_xticks(range(len(cm_vals.columns)))
    ax.set_yticks(range(len(cm_vals.index)))
    ax.set_xticklabels(cm_vals.columns)
    ax.set_yticklabels(cm_vals.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Classification recovery ({condition}) — {cm_title}")
    for i in range(cm_vals.shape[0]):
        for j in range(cm_vals.shape[1]):
            ax.text(j, i, int(cm_vals.values[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(fig_dir / "classification_confusion.png", dpi=150)
    plt.close(fig)

    # Prior distance curves for S/I/M groups (act_block_duringstim combined)
    if regde_stim:
        fig, ax = plt.subplots(figsize=(7, 4))
        for gtype, color in zip(["S", "I", "M"], ["C0", "C1", "C2"]):
            regs = [r for r, t in POPULATION_TYPE.items() if t == gtype and r in regde_stim]
            if not regs:
                continue
            curves = [regde_stim[r][0] for r in regs if len(regde_stim[r])]
            if curves:
                mean_c = np.mean(curves, axis=0)
                ax.plot(mean_c, label=f"{gtype} (n={len(curves)})", color=color)
        ax.set_xlabel("Time bin")
        ax.set_ylabel("Prior distance (d_euc)")
        ax.set_title(f"Pooled prior distance ({condition})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "prior_distance_by_group.png", dpi=150)
        plt.close(fig)

    # Prior modulation table figure
    if len(prior_df):
        fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(prior_df))))
        ax.axis("off")
        tbl = prior_df.round(4)
        ax.table(cellText=tbl.values, colLabels=tbl.columns, loc="center")
        ax.set_title(f"Sensory prior modulation p-values ({condition})")
        fig.tight_layout()
        fig.savefig(fig_dir / "prior_modulation_table.png", dpi=150)
        plt.close(fig)

    use_df.to_csv(fig_dir / "classification_details.csv", index=False)
    if pop_class_df is not None and len(pop_class_df):
        pop_class_df.to_csv(fig_dir / "population_classification.csv", index=False)
    prior_df.to_csv(fig_dir / "prior_modulation.csv", index=False)


def plot_s_prior_figures(condition, s_prior, fig_dir):
    """S-prior-only figure outputs (curve + shuffle control)."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    if s_prior is not None:
        plot_s_prior_curve(condition, s_prior, fig_dir)
        plot_s_shuffle_control(condition, s_prior, fig_dir)
        pd.DataFrame(
            [{k: v for k, v in s_prior.items() if not isinstance(v, np.ndarray)}]
        ).to_csv(fig_dir / "s_prior_stats.csv", index=False)


def _default_n_jobs():
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return max(1, int(os.environ["SLURM_CPUS_PER_TASK"]))
    return max(1, (os.cpu_count() or 1))


def process_condition(
    condition_name,
    g_s,
    d_s,
    n_sessions,
    nrand,
    blocks_per_session,
    max_obs_per_trial,
    base_dir,
    rng_seed=0,
    weights_json=None,
    min_trials_per_session=MIN_TRIALS_PER_SESSION_DEFAULT,
    s_prior_only=True,
    n_jobs=1,
):
    print(f"\n=== Condition: {condition_name} (g_s={g_s}, d_s={d_s}) ===")
    t0 = time.time()
    mp, meta = load_fitted_model(g_s=g_s, d_s=d_s, json_path=weights_json)

    cond_dir = base_dir / condition_name
    res_dir = cond_dir / "res"
    fig_dir = cond_dir / "figs"
    if res_dir.exists():
        shutil.rmtree(res_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    splits = s_prior_splits() if s_prior_only else collect_all_splits()
    populations = ("S",) if s_prior_only else None
    timeframes = [S_PRIOR_TIMEFRAME] if s_prior_only else FOCUSED_TIMEFRAMES
    n_jobs = max(1, int(n_jobs))
    rng = np.random.RandomState(rng_seed)
    session_dfs = []
    steps_before_obs = int(mf.STEPS_BEFORE_OBS_DURATION_MS / DT_MS)

    session_meta = []
    for sess in range(n_sessions):
        sess_rng = np.random.RandomState(rng_seed + sess)
        if blocks_per_session is None:
            n_blocks, planned_trials = blocks_for_min_trials(min_trials_per_session, sess_rng)
        else:
            n_blocks = blocks_per_session
            planned_trials = None
        results, sbo = simulate_session(mp, n_blocks, sess_rng, max_obs_per_trial)
        df = extract_trial_table(results, sbo)
        session_dfs.append(apply_act_prior(df))
        session_meta.append(
            {
                "session": sess,
                "n_blocks": n_blocks,
                "n_trials": len(df),
                "planned_trials_min_target": planned_trials,
            }
        )
        print(
            f"  session {sess + 1}/{n_sessions}: {n_blocks} blocks, "
            f"{len(df)} trials"
            + (f" (target >={min_trials_per_session})" if blocks_per_session is None else "")
        )

    print(
        f"  distance stage: {len(splits)} splits, populations="
        f"{populations or MODEL_POPULATIONS}, nrand={nrand}, n_jobs={n_jobs}"
    )
    t_dist = time.perf_counter()
    n_splits = build_res_from_trajectories(
        session_dfs,
        splits,
        steps_before_obs,
        nrand,
        rng,
        res_dir,
        populations=populations,
        n_jobs=n_jobs,
    )
    print(f"  wrote {n_splits} split files in {time.perf_counter() - t_dist:.1f}s")
    stack_combined_timeframes(res_dir, timeframes)

    s_prior = s_only_prior_test(res_dir)

    if s_prior_only:
        plot_s_prior_figures(condition_name, s_prior, fig_dir)
        n_trials_total = int(sum(m["n_trials"] for m in session_meta))
        summary = {
            "condition": condition_name,
            "g_s": g_s,
            "d_s": d_s,
            "mode": "s_prior_only",
            "prior_conditioning": {
                "prior_column": PRIOR_COLUMN,
                "timeframe": S_PRIOR_TIMEFRAME,
                "splits": splits,
                "populations": ["S"],
                "null_scheme": f"label shuffle, nrand={nrand}, n_jobs={n_jobs}",
            },
            "sessions": session_meta,
            "n_trials_total": n_trials_total,
            "g_i_fitted": float(meta["g"]["g_i"]),
            "d_i_fitted": float(meta["d"]["d_i"]),
            "s_prior_p_mean": s_prior["p_mean"] if s_prior else np.nan,
            "s_prior_p_offset": s_prior["p_offset"] if s_prior else np.nan,
            "s_prior_p_gain": s_prior["p_gain"] if s_prior else np.nan,
            "s_prior_gain_late_mean": s_prior["gain_late_mean"] if s_prior else np.nan,
            "s_prior_offset_mean": s_prior["offset_mean"] if s_prior else np.nan,
            "s_prior_amp_euc": s_prior["amp_euc"] if s_prior else np.nan,
            "s_prior_significant": bool(s_prior["significant_p_mean"]) if s_prior else False,
            "n_sessions": n_sessions,
            "nrand": nrand,
            "n_jobs": n_jobs,
            "weights_loss": meta.get("loss"),
            "runtime_sec": time.time() - t0,
        }
        with open(cond_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
        return summary

    sc_table, af = run_analysis(res_dir)
    bwm_recovered = classify_regions(sc_table, af)
    bwm_class_df, bwm_cm, bwm_acc = recovery_classification_metrics(bwm_recovered)

    pop_features = load_population_features(res_dir)
    pop_recovered, pop_class_df, pop_cm, pop_acc = classify_populations(pop_features)
    s_prior = s_only_prior_test(res_dir)
    pop_prior_df = population_prior_tests(res_dir)

    regions_by_type = defaultdict(list)
    for reg, t in POPULATION_TYPE.items():
        if t in ("S", "I", "M"):
            regions_by_type[t].append(reg)
    prior_df = prior_modulation_table(res_dir, regions_by_type)

    combined_name = "combined_" + "_".join(af.run_align["act_block_duringstim"])
    regde_path = res_dir / f"combined_regde_{'_'.join(af.run_align['act_block_duringstim'])}.npy"
    regde_stim = np.load(regde_path, allow_pickle=True).item() if regde_path.exists() else {}

    plot_recovery_figures(
        condition_name,
        res_dir,
        fig_dir,
        bwm_class_df,
        bwm_cm,
        prior_df,
        regde_stim,
        pop_class_df=pop_class_df,
        pop_cm=pop_cm,
        s_prior=s_prior,
    )
    if s_prior is not None:
        pd.DataFrame([{k: v for k, v in s_prior.items() if not isinstance(v, np.ndarray)}]).to_csv(
            fig_dir / "s_prior_stats.csv", index=False
        )
    pop_prior_df.to_csv(fig_dir / "population_prior_tests.csv", index=False)

    n_trials_total = int(sum(m["n_trials"] for m in session_meta))
    summary = {
        "condition": condition_name,
        "g_s": g_s,
        "d_s": d_s,
        "prior_conditioning": {
            "prior_column": PRIOR_COLUMN,
            "description": "binarized model P subjective prior (mean P_L-P_R per trial)",
            "populations": "S, I, M, P share the same prior column per split",
            "null_scheme": f"label shuffle, nrand={nrand}",
        },
        "sessions": session_meta,
        "n_trials_total": n_trials_total,
        "classification_accuracy": float(pop_acc),
        "bwm_classification_accuracy": float(bwm_acc),
        "recovered_S": pop_recovered["S"],
        "recovered_I": pop_recovered["I"],
        "recovered_M": pop_recovered["M"],
        "bwm_recovered_S": bwm_recovered["S"],
        "bwm_recovered_I": bwm_recovered["I"],
        "bwm_recovered_M": bwm_recovered["M"],
        "g_i_fitted": float(meta["g"]["g_i"]),
        "d_i_fitted": float(meta["d"]["d_i"]),
        "s_prior_p_mean": s_prior["p_mean"] if s_prior else np.nan,
        "s_prior_p_offset": s_prior["p_offset"] if s_prior else np.nan,
        "s_prior_p_gain": s_prior["p_gain"] if s_prior else np.nan,
        "s_prior_gain_late_mean": s_prior["gain_late_mean"] if s_prior else np.nan,
        "s_prior_offset_mean": s_prior["offset_mean"] if s_prior else np.nan,
        "s_prior_amp_euc": s_prior["amp_euc"] if s_prior else np.nan,
        "s_prior_significant": bool(s_prior["significant_p_mean"]) if s_prior else False,
        "population_prior_tests": pop_prior_df.to_dict(orient="records"),
        "S_prior_significant_frac": float(
            prior_df.loc[prior_df["group"] == "S", "significant"].mean()
        )
        if len(prior_df)
        else np.nan,
        "I_prior_significant_frac": float(
            prior_df.loc[prior_df["group"] == "I", "significant"].mean()
        )
        if len(prior_df)
        else np.nan,
        "M_prior_significant_frac": float(
            prior_df.loc[prior_df["group"] == "M", "significant"].mean()
        )
        if len(prior_df)
        else np.nan,
        "prior_by_population": {
            row["region"]: {
                "p_mean": float(row["p_mean"]),
                "p_mean_c": float(row["p_mean_c"]),
                "amp_euc": float(row["amp_euc"]),
                "significant": bool(row["significant"]),
            }
            for _, row in prior_df.iterrows()
        },
        "n_sessions": n_sessions,
        "nrand": nrand,
        "weights_loss": meta.get("loss"),
        "runtime_sec": time.time() - t0,
    }
    with open(cond_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def plot_s_prior_comparison(base_dir, abs_s, pres_s, cross_p, sensory_prior_recovery):
    """Dedicated S prior comparison figure (raw curves, baseline not removed)."""
    if abs_s is None or pres_s is None:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    t = abs_s["t_axis"]

    ax.plot(t, abs_s["curve_real"], color="#4C72B0", lw=2, label="absence")
    ax.plot(t, pres_s["curve_real"], color="#DD8452", lw=2, label="presence")
    ax.plot(t, abs_s["curve_null_mean"], color="#4C72B0", ls="--", alpha=0.5, label="abs null")
    ax.plot(t, pres_s["curve_null_mean"], color="#DD8452", ls="--", alpha=0.5, label="pres null")
    if len(t) >= PRIOR_OFFSET_BINS:
        ax.axvspan(t[0], t[PRIOR_OFFSET_BINS - 1], color="gray", alpha=0.12, label="offset window")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("S prior distance (raw)")
    ax.set_title("S prior curves (raw, baseline retained)")
    ax.legend(fontsize=8)

    if cross_p:
        g_s = sensory_prior_recovery.get("g_s_presence", "?")
        d_s = sensory_prior_recovery.get("d_s_presence", "?")
        g_i = sensory_prior_recovery.get("g_i_fitted", "?")
        d_i = sensory_prior_recovery.get("d_i_fitted", "?")
        txt = (
            f"fitted integrator: g_i={g_i}, d_i={d_i} | presence sensory: g_s={g_s}, d_s={d_s}\n"
            f"p_mean(pres>abs)={cross_p['p_mean_pres_vs_abs']:.4f}  "
            f"p_offset(pres>abs)={cross_p['p_offset_pres_vs_abs']:.4f}  "
            f"p_gain(pres>abs)={cross_p.get('p_gain_pres_vs_abs', np.nan):.4f}"
        )
        fig.text(0.5, -0.02, txt, ha="center", va="top", fontsize=9, family="monospace")

    fig.tight_layout()
    out = base_dir / "s_prior_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved S prior comparison figure to {out}")


def _plot_pres_abs_bar_comparison(ax, abs_val, pres_val, ylabel, metric_name, p_cross):
    """Grouped absence vs presence bar chart for one scalar metric."""
    x = [0, 1]
    ax.bar(x, [abs_val, pres_val], color=["#4C72B0", "#DD8452"], width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(["absence", "presence"])
    ax.set_ylabel(ylabel)
    p_txt = f"p(pres>abs)={p_cross:.4f}" if not np.isnan(p_cross) else "p(pres>abs)=n/a"
    ax.set_title(f"{metric_name}\n{p_txt}")


def plot_comparison_metrics_summary(
    sensory_prior_recovery,
    presence,
    out_path,
):
    """Three-panel absence vs presence metric bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    _plot_pres_abs_bar_comparison(
        axes[0],
        sensory_prior_recovery["s_curve_mean_absence"],
        sensory_prior_recovery["s_curve_mean_presence"],
        "mean prior distance",
        "S curve overall mean (p_mean)",
        sensory_prior_recovery["p_mean_pres_vs_abs"],
    )
    _plot_pres_abs_bar_comparison(
        axes[1],
        sensory_prior_recovery["s_offset_mean_absence"],
        sensory_prior_recovery["s_offset_mean_presence"],
        f"mean first {PRIOR_OFFSET_BINS} bins",
        "S early-bin mean (p_offset)",
        sensory_prior_recovery["p_offset_pres_vs_abs"],
    )
    _plot_pres_abs_bar_comparison(
        axes[2],
        sensory_prior_recovery["s_gain_late_mean_absence"],
        sensory_prior_recovery["s_gain_late_mean_presence"],
        "mean bins 4+ after early offset removed",
        "S offset-corrected late mean (p_gain)",
        sensory_prior_recovery["p_gain_pres_vs_abs"],
    )

    gs = presence.get("g_s", "?")
    ds = presence.get("d_s", "?")
    fig.suptitle(
        f"S sensory prior recovery: "
        f"absence (g_s=0,d_s=0) vs presence (g_s={gs}, d_s={ds})",
        y=1.04,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison metrics figure to {out_path}")


def compare_conditions(base_dir, alpha=0.01, s_amp_ratio_thresh=1.1):
    absence = json.loads((base_dir / "absence" / "summary.json").read_text())
    presence = json.loads((base_dir / "presence" / "summary.json").read_text())

    abs_s = s_only_prior_test(base_dir / "absence" / "res", alpha=alpha)
    pres_s = s_only_prior_test(base_dir / "presence" / "res", alpha=alpha)
    cross_p = s_prior_presence_vs_absence_pvalues(abs_s, pres_s, alpha=alpha)
    s_amp_ratio = (
        pres_s["amp_euc"] / (abs_s["amp_euc"] + 1e-12) if abs_s and pres_s else np.nan
    )
    g_i_fit, d_i_fit = fitted_integrator_scales()
    sensory_prior_recovery = {
        "g_i_fitted": g_i_fit,
        "d_i_fitted": d_i_fit,
        "g_s_presence": presence.get("g_s"),
        "d_s_presence": presence.get("d_s"),
        "s_offset_mean_absence": abs_s["offset_mean"] if abs_s else np.nan,
        "s_offset_mean_presence": pres_s["offset_mean"] if pres_s else np.nan,
        "s_p_offset_absence": abs_s["p_offset"] if abs_s else np.nan,
        "s_p_offset_presence": pres_s["p_offset"] if pres_s else np.nan,
        "s_curve_mean_absence": abs_s["curve_mean"] if abs_s else np.nan,
        "s_curve_mean_presence": pres_s["curve_mean"] if pres_s else np.nan,
        "s_curve_amp_absence": abs_s["curve_amp"] if abs_s else np.nan,
        "s_curve_amp_presence": pres_s["curve_amp"] if pres_s else np.nan,
        "s_amp_absence": abs_s["amp_euc"] if abs_s else np.nan,
        "s_amp_presence": pres_s["amp_euc"] if pres_s else np.nan,
        "s_amp_ratio": float(s_amp_ratio),
        "s_p_mean_absence": abs_s["p_mean"] if abs_s else np.nan,
        "s_p_mean_presence": pres_s["p_mean"] if pres_s else np.nan,
        "s_gain_late_mean_absence": abs_s["gain_late_mean"] if abs_s else np.nan,
        "s_gain_late_mean_presence": pres_s["gain_late_mean"] if pres_s else np.nan,
        "p_mean_pres_vs_abs": cross_p["p_mean_pres_vs_abs"] if cross_p else np.nan,
        "p_amp_pres_vs_abs": cross_p["p_amp_pres_vs_abs"] if cross_p else np.nan,
        "p_offset_pres_vs_abs": cross_p["p_offset_pres_vs_abs"] if cross_p else np.nan,
        "p_gain_pres_vs_abs": cross_p["p_gain_pres_vs_abs"] if cross_p else np.nan,
        "s_significant_absence": bool(abs_s["significant_p_mean"]) if abs_s else False,
        "s_significant_presence": bool(pres_s["significant_p_mean"]) if pres_s else False,
        "recovered_sensory_gain": bool(
            cross_p
            and cross_p.get("p_gain_pres_vs_abs", 1.0) <= alpha
            and cross_p.get("real_gain_diff", 0) > 0
        ),
        "recovered_sensory_offset": bool(
            cross_p
            and cross_p["p_offset_pres_vs_abs"] <= alpha
            and cross_p["real_offset_diff"] > 0
        ),
    }
    out_json = base_dir / "sensory_prior_recovery.json"
    out_json.write_text(json.dumps(sensory_prior_recovery, indent=2))

    plot_comparison_metrics_summary(
        sensory_prior_recovery,
        presence,
        base_dir / "comparison_summary_metrics.png",
    )
    plot_s_prior_comparison(base_dir, abs_s, pres_s, cross_p, sensory_prior_recovery)
    plot_combined_shuffle_controls(base_dir, abs_s, pres_s)
    print(json.dumps(sensory_prior_recovery, indent=2))
    return sensory_prior_recovery


def run_recovery_only(base_dir, alpha=0.01, s_prior_only=True):
    """Re-run analysis on existing res/ outputs."""
    base_dir = Path(base_dir)
    summaries = {}
    for cond in ("absence", "presence"):
        res_dir = base_dir / cond / "res"
        fig_dir = base_dir / cond / "figs"
        if not res_dir.exists():
            continue
        s_prior = s_only_prior_test(res_dir, alpha=alpha)
        if s_prior_only:
            plot_s_prior_figures(cond, s_prior, fig_dir)
            summaries[cond] = {
                "s_prior": {
                    k: v for k, v in (s_prior or {}).items() if not isinstance(v, np.ndarray)
                },
            }
            continue
        pop_features = load_population_features(res_dir)
        pop_recovered, pop_class_df, pop_cm, pop_acc = classify_populations(pop_features)
        pop_prior_df = population_prior_tests(res_dir, alpha=alpha)
        _, regde_name, _ = _combined_names("act_block_duringstim")
        regde_path = res_dir / f"{regde_name}.npy"
        regde_stim = np.load(regde_path, allow_pickle=True).item() if regde_path.exists() else {}
        prior_df = prior_modulation_table(res_dir, {"S": ["S"], "I": ["I"], "M": ["M"]})
        plot_recovery_figures(
            cond, res_dir, fig_dir, pop_class_df, pop_cm, prior_df, regde_stim,
            pop_class_df=pop_class_df, pop_cm=pop_cm, s_prior=s_prior,
        )
        summaries[cond] = {
            "classification_accuracy": pop_acc,
            "recovered": pop_recovered,
            "s_prior": {k: v for k, v in (s_prior or {}).items() if not isinstance(v, np.ndarray)},
            "population_prior_tests": pop_prior_df.to_dict(orient="records"),
        }
    if (base_dir / "absence").exists() and (base_dir / "presence").exists():
        compare_conditions(base_dir, alpha=alpha)
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Circuit generative model recovery demo")
    parser.add_argument("--n-sessions", type=int, default=N_SESSIONS_DEFAULT)
    parser.add_argument("--nrand", type=int, default=NRAND_DEFAULT)
    parser.add_argument(
        "--blocks-per-session",
        type=int,
        default=BLOCKS_PER_SESSION_DEFAULT,
        help="Blocks per session (geometric trial counts per block as in create_stimuli)",
    )
    parser.add_argument(
        "--min-trials-per-session",
        type=int,
        default=MIN_TRIALS_PER_SESSION_DEFAULT,
        help="Target min trials when --blocks-per-session is not set",
    )
    parser.add_argument("--max-obs-per-trial", type=int, default=400)
    parser.add_argument("--g-s-presence", type=float, default=GAIN_PRESENCE_DEFAULT)
    parser.add_argument("--d-s-presence", type=float, default=OFFSET_PRESENCE_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(mf.one.cache_dir) / "manifold_sim"),
    )
    parser.add_argument("--weights-json", type=str, default=str(WEIGHTS_JSON))
    parser.add_argument(
        "--recovery-only",
        action="store_true",
        help="Skip simulation; re-run analysis on existing res/ outputs",
    )
    parser.add_argument(
        "--full-analysis",
        action="store_true",
        help="All splits/populations + classifiers (slow). Default: S-prior distance only.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Parallel workers for label-shuffle nulls (default: SLURM_CPUS_PER_TASK or CPU count)",
    )
    args = parser.parse_args()

    weights_json = Path(args.weights_json)
    n_jobs = args.n_jobs if args.n_jobs is not None else _default_n_jobs()
    s_prior_only = not args.full_analysis

    base_dir = Path(args.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if args.recovery_only:
        run_recovery_only(base_dir, s_prior_only=s_prior_only)
        return

    common_kw = dict(
        n_sessions=args.n_sessions,
        nrand=args.nrand,
        blocks_per_session=args.blocks_per_session,
        max_obs_per_trial=args.max_obs_per_trial,
        base_dir=base_dir,
        weights_json=weights_json,
        min_trials_per_session=args.min_trials_per_session,
        s_prior_only=s_prior_only,
        n_jobs=n_jobs,
    )
    process_condition(
        "absence",
        g_s=0.0,
        d_s=0.0,
        rng_seed=args.seed,
        **common_kw,
    )
    process_condition(
        "presence",
        g_s=args.g_s_presence,
        d_s=args.d_s_presence,
        rng_seed=args.seed + 1000,
        **common_kw,
    )
    compare_conditions(base_dir)


if __name__ == "__main__":
    main()
