#!/usr/bin/env python3
"""
Generative-model recovery demo for the prior-mechanism circuit model.

Simulates trial-level model population trajectories (S/I/M/P), computes prior
distances with label-shuffle nulls per split (as in d_var_stacked_multi), then
runs the classification + prior-modulation analysis pipeline to check recovery
of population types and sensory prior modulation (g_s/d_s).

Canonical prior-distance analysis (required for all experiments since 2026-06-19):
  - Trial windows: fill-from-next-ITI when RT < window end (never zero-pad).
  - S population: 80 ms post-stim window (S_DURINGSTIM_WINDOW_S).
  - I/M populations: 150 ms post-stim window (PRE_POST duringstim splits).
  - Null: contrast-matched label shuffle (default).
  - Phase 4b sanity check (split, seed 123): S curve_mean≈0.012, p≈0.78.
See AGENTS.md and .cursor/rules/prior-distance-analysis.mdc for agent guidance.

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
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
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
ABSENCE_PRES_NULL_REPLICATES_DEFAULT = 100
NRAND_PRES_ABS_NULL = 1
BLOCKS_PER_SESSION_DEFAULT = 6
MIN_TRIALS_PER_SESSION_DEFAULT = 600
SHUFFLE_PLOT_N_SAMPLE = 100
ALPHA_ACT = 0.2
DT_MS = 2.0
PRIOR_COLUMN = "p_subjective_probabilityLeft"
PRIOR_COLUMN_BLOCK = "probabilityLeft"
RANDOM_PRIOR_COLUMN = "p_random_probabilityLeft"
NULL_SCHEME_CONTRAST_MATCHED = (
    "contrast-matched label shuffle (preserves per-contrast counts in high/low groups)"
)
NULL_SCHEME_LABEL_SHUFFLE = "unrestricted label shuffle"
PRIOR_LABEL_VARIANTS = {
    "subjective": PRIOR_COLUMN,
    "block": PRIOR_COLUMN_BLOCK,
}
_prior_column_override = None

WEIGHTS_REL = Path(
    "models/weights_run_20251125_182058/"
    "weights_2stagelocalrefine_loss0p4044_20251125-195255.json"
)

# Model populations used directly as "regions" (no synthetic neurons / atlas).
MODEL_POPULATIONS = ("S", "I", "M", "P")
# S / I / M populations for multi-population prior-distance controls.
SIM_POPULATIONS = ("S", "I", "M")
POPULATION_TYPE = {"S": "S", "I": "I", "M": "M", "P": "P"}
PRIOR_OFFSET_BINS = 5
# g_s for presence matches g_i (fitted integrator scale) so S receives the same
# prior drive as the I population.  Loaded from weights JSON at runtime.
TUNED_G_S_PRESENCE = None  # sentinel: replaced by g_i at runtime


def time_axis_for_split(split, n_bins):
    """
    Plot time axis in ms, aligned to the split's event (stimOn or movement).

    Matches analysis_functions.plot_regional_distance: t=0 is stimulus onset for
    duringstim splits and choice/movement onset for duringchoice splits.
    """
    if "duringchoice" in split:
        return np.linspace(-150, 0, n_bins)
    if "duringstim" in split and "short" in split:
        return np.linspace(0, 80, n_bins)
    if "duringstim" in split:
        return np.linspace(0, 150, n_bins)
    return np.linspace(-400, -100, n_bins)


def _time_xlabel(split):
    if "duringstim" in split:
        return "Time from stim onset (ms)"
    if "duringchoice" in split:
        return "Time from choice onset (ms)"
    return "Time (ms)"


def _mark_align_event(ax, split):
    """Vertical line at t=0 (stim or choice onset)."""
    ax.axvline(0, color="0.35", lw=0.9, ls=":", zorder=2)


def _shade_offset_window(ax, t, n_early=PRIOR_OFFSET_BINS, **kwargs):
    """Highlight first n_early bins (offset test window), anchored at align time."""
    if len(t) < n_early:
        return
    kw = {"color": "gray", "alpha": 0.12, "zorder": 0}
    kw.update(kwargs)
    ax.axvspan(t[0], t[n_early - 1], **kw)


def _one_cache_dir_candidates():
    """ONE cache roots: ONE_CACHE_DIR env, one.cache_dir, and one.cache_dir/alyx."""
    bases = []
    if "ONE_CACHE_DIR" in os.environ:
        bases.append(Path(os.environ["ONE_CACHE_DIR"]))
    cache = Path(mf.one.cache_dir)
    bases.extend([cache, cache / "alyx"])
    seen = set()
    for base in bases:
        base = base.resolve()
        if base in seen:
            continue
        seen.add(base)
        yield base


def resolve_one_cache_dir():
    """
    Resolve the active ONE cache directory.

    On ORCD, fitted weights and analysis outputs often live under .../ONE/alyx while
    one.cache_dir may point at the parent .../ONE.
    """
    candidates = list(_one_cache_dir_candidates())
    for base in candidates:
        if (base / "models").is_dir() or (base / "manifold").is_dir():
            return base
    return candidates[0] if candidates else Path(mf.one.cache_dir)


def default_output_dir():
    return resolve_one_cache_dir() / "manifold_sim"


def _project_root():
    """Git repo root containing simulate_recovery.py."""
    root = Path(__file__).resolve().parent
    for candidate in (root, *root.parents):
        if (candidate / ".git").is_dir():
            return candidate
    return root


def resolve_output_dir(explicit=None, allow_repo_output=False):
    """
    Resolve simulation output root.

    Default: <ONE cache>/manifold_sim. Paths inside the git repo (especially
    ``output/...`` copied from journals) are redirected to the same relative
    path under manifold_sim so results stay in the ONE cache.
    """
    canonical = default_output_dir()
    if explicit is None:
        print(f"Output directory: {canonical}")
        return canonical

    path = Path(explicit).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    project = _project_root().resolve()
    try:
        rel = path.relative_to(project)
    except ValueError:
        print(f"Output directory: {path}")
        return path

    if allow_repo_output:
        print(f"Output directory (repo, --allow-repo-output): {path}")
        return path

    suffix = rel
    if rel.parts and rel.parts[0] == "output":
        suffix = Path(*rel.parts[1:]) if len(rel.parts) > 1 else Path()

    redirected = canonical / suffix
    print(
        f"WARNING: --output-dir {explicit!r} is inside the git repo ({project}).\n"
        f"         Redirecting to: {redirected}\n"
        f"         (Omit --output-dir to use {canonical}; "
        f"pass --allow-repo-output to write under the repo.)",
        flush=True,
    )
    return redirected


def _weights_json_candidates():
    """Search ONE cache roots (incl. .../alyx) for the fitted weights file."""
    for base in _one_cache_dir_candidates():
        yield base / WEIGHTS_REL


def resolve_weights_json(explicit=None):
    """Resolve weights JSON path; explicit must exist if provided."""
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Weights JSON not found: {path}")
        return path
    for candidate in _weights_json_candidates():
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(p) for p in _weights_json_candidates())
    raise FileNotFoundError(
        f"Weights JSON not found. Tried: {tried}. "
        "Pass --weights-json or set ONE_CACHE_DIR."
    )


def fitted_integrator_scales(json_path=None):
    """Return fitted g_i, d_i to use as comparable g_s, d_s for sensory prior presence."""
    json_path = resolve_weights_json(json_path)
    with open(json_path) as f:
        meta = json.load(f)
    return float(meta["g"]["g_i"]), float(meta["d"]["d_i"])


def _trace_asymmetry(vec):
    """Signed left-right asymmetry (positive = right-larger)."""
    v = np.asarray(vec, float).reshape(-1)
    return float(v[1] - v[0])


def estimate_s_s0_magnitude_ratio(
    weights_json=None,
    n_sessions=10,
    blocks_per_session=BLOCKS_PER_SESSION_DEFAULT,
    max_obs_per_trial=400,
    rng_seed=42,
):
    """
    Median |S| / |S0| during the prior-modulation window (prestim through early stim).

    Matches the drive scaling in the generative model: S receives ``g_s * P @ S0`` while
    I receives ``g_i * P @ S``. Setting ``g_s = g_i * (|S|/|S0|)`` makes the two drives
    comparable; ``d_s = d_i`` matches offset terms directly.
    """
    mp, _ = load_fitted_model(zero_all_prior_mod=True, json_path=weights_json)
    rng = np.random.default_rng(rng_seed)
    steps_before_obs = int(mf.STEPS_BEFORE_OBS_DURATION_MS / DT_MS)
    # S lags S0 by tens of ms; use 50–80 ms post-stim (matches S prior window).
    lag_start = int(50 / DT_MS)
    lag_end = int(80 / DT_MS)
    min_contrast = 0.0625

    s_abs, s0_abs = [], []
    for _ in range(n_sessions):
        results, _ = simulate_session(
            mp, blocks_per_session, rng, max_obs_per_trial, constant_s0=False
        )
        S = np.asarray(results["S"], float)
        S0 = np.asarray(results["perceived_stim"], float)
        n_trials = len(results["trial_sides"])
        lens = [len(results["trial_sides"][i]) for i in range(n_trials)]
        offsets = np.cumsum([0] + lens[:-1])
        for i in range(n_trials):
            off = int(offsets[i])
            m_i = lens[i]
            side = int(np.sign(results["trial_sides"][i][0])) or 1
            contrast = abs(float(np.asarray(results["trial_strengths"][i][0])))
            if contrast < min_contrast:
                continue
            for lag in range(lag_start, lag_end + 1):
                k = steps_before_obs + lag
                if k >= m_i:
                    break
                g = off + k
                s0_asym = side * _trace_asymmetry(S0[g])
                s_asym = side * _trace_asymmetry(S[g])
                if abs(s0_asym) < 1e-6:
                    continue
                s0_abs.append(abs(s0_asym))
                s_abs.append(abs(s_asym))

    s0_med = float(np.median(s0_abs))
    s_med = float(np.median(s_abs))
    ratio = s_med / s0_med if s0_med > 0 else float("nan")
    return {
        "s_median_abs": s_med,
        "s0_median_abs": s0_med,
        "s_over_s0": ratio,
        "n_samples": len(s_abs),
        "rng_seed": rng_seed,
    }


def integrator_comparable_s_params(
    weights_json=None,
    rng_seed=42,
    n_sessions=10,
    blocks_per_session=BLOCKS_PER_SESSION_DEFAULT,
    max_obs_per_trial=400,
):
    """g_s = g_i * (|S|/|S0|), d_s = d_i so S and I prior drives are comparable."""
    g_i, d_i = fitted_integrator_scales(weights_json)
    est = estimate_s_s0_magnitude_ratio(
        weights_json=weights_json,
        n_sessions=n_sessions,
        blocks_per_session=blocks_per_session,
        max_obs_per_trial=max_obs_per_trial,
        rng_seed=rng_seed,
    )
    ratio = est["s_over_s0"]
    g_s = g_i * ratio
    return {
        "g_s": float(g_s),
        "d_s": float(d_i),
        "g_i_fitted": g_i,
        "d_i_fitted": d_i,
        **est,
    }

SC_TIMES = [
    "stim_duringstim_act",
    "choice_duringstim_act",
    "stim_duringchoice_act",
    "choice_duringchoice_act",
]

TIMING_SPLITS = ["act_block_duringstim", "act_block_duringchoice"]
S_PRIOR_TIMEFRAME = "act_block_duringstim"
# Unsplit = no choice×feedback (f1/f2) conditioning; still split by stim side (L/R).
UNSPLIT_PRIOR_TIMEFRAME = "act_block_duringstim_unsplit"
UNSPLIT_PRIOR_SPLITS = (
    "act_block_duringstim_stim_l_unsplit",
    "act_block_duringstim_stim_r_unsplit",
)
# Fully unsplit: all duringstim trials in one pool (L+R stim; S channel misalignment risk).
FULLY_UNSPLIT_PRIOR_SPLIT = "act_block_duringstim_fully_unsplit"
FULLY_UNSPLIT_PRIOR_TIMEFRAME = FULLY_UNSPLIT_PRIOR_SPLIT
# S dynamics are fast; cap the S analysis window at 80ms to avoid borrowing
# large chunks of next-trial ITI at high contrast (where RT << 150ms).
# I/M use the full 150ms window defined in PRE_POST.
S_DURINGSTIM_WINDOW_S = 0.08
IM_DURINGSTIM_WINDOW_S = 0.15

# Canonical analysis defaults (2026-06-19 retest; see research_journal_2026-06-18.md).
CANONICAL_PRIOR_DISTANCE_ANALYSIS = {
    "s_window_s": S_DURINGSTIM_WINDOW_S,
    "im_window_s": IM_DURINGSTIM_WINDOW_S,
    "truncation": "fill_from_next_iti",  # never zero-pad truncated stimOn windows
    "contrast_matched_null": True,
    "phase4b_sanity_seed123": {"s_curve_mean": 0.0124, "s_p_mean": 0.78},
}


def log_canonical_analysis_banner():
    """Print canonical analysis settings at the start of each analysis run."""
    c = CANONICAL_PRIOR_DISTANCE_ANALYSIS
    print(
        "Canonical prior-distance analysis: "
        f"S={c['s_window_s']*1000:.0f}ms, I/M={c['im_window_s']*1000:.0f}ms, "
        f"truncation={c['truncation']}, "
        f"null={'contrast-matched' if c['contrast_matched_null'] else 'label-shuffle'}"
    )

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
for _unsplit in (*UNSPLIT_PRIOR_SPLITS, FULLY_UNSPLIT_PRIOR_SPLIT):
    ALIGN[_unsplit] = "stimOn_times"
    PRE_POST[_unsplit] = [0, IM_DURINGSTIM_WINDOW_S]
SIM_TIMEFRAME_SPLITS = {
    UNSPLIT_PRIOR_TIMEFRAME: list(UNSPLIT_PRIOR_SPLITS),
    FULLY_UNSPLIT_PRIOR_TIMEFRAME: [FULLY_UNSPLIT_PRIOR_SPLIT],
}


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


def p_subjective_probability_left_trial_mean(p_trace):
    """Binarize from trial-averaged P (legacy; includes during-stim P)."""
    pdiff = float(np.mean(p_trace[:, 0] - p_trace[:, 1]))
    return 0.2 if pdiff < 0 else 0.8


def p_subjective_probability_left(p_trace, steps_before_obs):
    """
    Binarize model P to 0.8/0.2 probabilityLeft from the pre-stimulus intertrial window.

    Uses the same ITI window as model_functions / act_intertrial splits:
    [-ITI_START_BEFORE_MS, -ITI_END_BEFORE_MS) relative to stimulus onset
    (default −400 ms to −100 ms before stimOn).
    """
    start_before = int(mf.ITI_START_BEFORE_MS / DT_MS)
    end_before = int(mf.ITI_END_BEFORE_MS / DT_MS)
    s = max(0, steps_before_obs - start_before)
    e = min(len(p_trace), steps_before_obs - end_before)
    if e <= s:
        pre = p_trace[:steps_before_obs]
        seg = pre if len(pre) else p_trace
    else:
        seg = p_trace[s:e]
    pdiff = float(np.mean(seg[:, 0] - seg[:, 1]))
    return 0.2 if pdiff < 0 else 0.8


@contextmanager
def use_prior_column(column):
    """Temporarily override trial high/low grouping column."""
    global _prior_column_override
    prev = _prior_column_override
    _prior_column_override = column
    try:
        yield
    finally:
        _prior_column_override = prev


def prior_column_for_split(split):
    """
  Prior column for high/low trial groups in simulated data.

  Default: binarized model P population (mean P_L - P_R per trial).
  """
    if _prior_column_override is not None:
        return _prior_column_override
    return PRIOR_COLUMN


def _prior_label_title(prior_label):
    if not prior_label:
        return ""
    labels = {
        "subjective": "subjective prior (pre-stim P, ITI window)",
        "block": "true block prior (probabilityLeft)",
        "contrast_null": "contrast-matched null",
        "label_null": "unrestricted label null",
    }
    return labels.get(prior_label, prior_label)


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


def load_fitted_model(
    g_s=0.0,
    d_s=0.0,
    zero_all_prior_mod=False,
    zero_im_prior_mod=False,
    json_path=None,
    gs_outside_adaptation=False,
):
    json_path = resolve_weights_json(json_path)
    with open(json_path) as f:
        meta = json.load(f)

    mp = deepcopy(mf.model_params)
    mp.update(meta.get("model_params", {}))
    mp.update(meta["W"])
    if zero_all_prior_mod:
        mp["g_i"] = 0.0
        mp["g_m"] = 0.0
        mp["d_i"] = 0.0
        mp["d_m"] = 0.0
        g_s, d_s = 0.0, 0.0
    elif zero_im_prior_mod:
        mp["g_i"] = 0.0
        mp["g_m"] = 0.0
        mp["d_i"] = 0.0
        mp["d_m"] = 0.0
    else:
        mp["g_i"] = meta["g"]["g_i"]
        mp["g_m"] = meta["g"]["g_m"]
        mp["d_i"] = meta["d"]["d_i"]
        mp["d_m"] = meta["d"]["d_m"]
    mp["g_s"] = float(g_s)
    mp["d_s"] = float(d_s)
    mp["gs_outside_adaptation"] = bool(gs_outside_adaptation)
    theta_c = meta["theta"]["theta_c"]
    theta_d = meta["theta"]["theta_d"]
    mp["action_thresholds"] = {
        "concordant": {c: theta_c for c in [1.0, 0.25, 0.125, 0.0625, 0.0]},
        "discordant": {c: theta_d for c in [1.0, 0.25, 0.125, 0.0625, 0.0]},
    }
    mp["dt"] = DT_MS
    mf._update_model_params_for_dt(mp, DT_MS)
    return mp, meta


def apply_constant_s0_stimuli(stimuli, trial_strengths, trial_sides, steps_before_obs):
    """
    Replace stochastic S0 with deterministic contrast on the signal side.

    From stim onset onward: signal channel = |contrast|, other channel = 0.
    Pre-stim steps are zero on both channels.
    """
    for i, block_stim in enumerate(stimuli):
        for j, trial in enumerate(block_stim):
            c = abs(float(np.asarray(trial_strengths[i][j]).reshape(-1)[0]))
            side = int(np.sign(trial_sides[i][j][0]))
            if side == 0:
                side = 1
            trial[:] = 0.0
            stim_start = min(int(steps_before_obs), trial.shape[0])
            if side < 0:
                trial[stim_start:, 0] = c
            else:
                trial[stim_start:, 1] = c
    return stimuli


def simulate_session(
    model_params,
    blocks_per_session,
    rng,
    max_obs_per_trial,
    constant_s0=False,
):
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
    if constant_s0:
        apply_constant_s0_stimuli(stimuli, trial_strengths, trial_sides, steps_before_obs)
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


def _iti_step_bounds(steps_before_obs, trial_len):
    """Step indices for the pre-stim ITI window [-ITI_START, -ITI_END) before stimOn."""
    start_before = int(mf.ITI_START_BEFORE_MS / DT_MS)
    end_before = int(mf.ITI_END_BEFORE_MS / DT_MS)
    s = max(0, steps_before_obs - start_before)
    e = min(trial_len, steps_before_obs - end_before)
    if e <= s:
        pre_end = min(steps_before_obs, trial_len)
        return 0, pre_end
    return s, e


def _trace_iti_mean_norm(trace_2d, steps_before_obs):
    """Mean L2 norm of a population trace over the ITI window."""
    s, e = _iti_step_bounds(steps_before_obs, len(trace_2d))
    seg = trace_2d[s:e]
    if seg.size == 0:
        return np.nan
    return float(np.mean(np.linalg.norm(seg, axis=1)))


def extract_trial_table(results, steps_before_obs):
    """Build per-trial metadata table aligned with IBL conventions."""
    rows = []
    pops = {k: np.asarray(results[k], float) for k in ("S", "I", "P", "M")}
    a_flat = np.asarray(results["a"], float) if "a" in results and not isinstance(results["a"], float) else None
    n_trials = len(results["trial_sides"])
    lens = [len(results["trial_sides"][i]) for i in range(n_trials)]
    offsets = np.cumsum([0] + lens[:-1])
    prev_block_side = None
    trial_in_block = 0

    for i in range(n_trials):
        m_i = lens[i]
        off = offsets[i]
        trial_side = int(np.sign(results["trial_sides"][i][0])) or 1
        block_side = int(np.sign(results["block_sides"][i][0]))
        if block_side != prev_block_side:
            trial_in_block = 0
            prev_block_side = block_side
        else:
            trial_in_block += 1
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
        s_trace = pops["S"][off : off + m_i]
        stim_idx = min(steps_before_obs, m_i - 1)
        a_at_stim_mean = np.nan
        a_at_stim_asym = np.nan
        if a_flat is not None and a_flat.shape[0] > off + stim_idx:
            a_step = a_flat[off + stim_idx]
            a_at_stim_mean = float(np.mean(a_step))
            a_at_stim_asym = float(a_step[0] - a_step[1])

        rows.append(
            {
                "trial_idx": i,
                "offset": off,
                "length": m_i,
                "trial_side": trial_side,
                "block_side": block_side,
                "trial_in_block": trial_in_block,
                "probabilityLeft": pleft,
                "p_subjective_probabilityLeft": p_subjective_probability_left(
                    p_trace, steps_before_obs
                ),
                "p_subjective_probabilityLeft_trialmean": p_subjective_probability_left_trial_mean(
                    p_trace
                ),
                "contrastLeft": cl,
                "contrastRight": cr,
                "choice": ibl_choice,
                "feedbackType": feedback,
                "reaction_time": rt,
                "correct": correct,
                "a_at_stim_mean": a_at_stim_mean,
                "a_at_stim_asym": a_at_stim_asym,
                "s_norm_iti_mean": _trace_iti_mean_norm(s_trace, steps_before_obs),
                "prev_feedback": rows[-1]["feedbackType"] if rows else np.nan,
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


def split_fixed_condition_mask(df, split):
    """Boolean mask for trials matching split stim/choice/feedback (ignores prior)."""
    m = np.ones(len(df), dtype=bool)

    if "fully_unsplit" in split:
        cl = df["contrastLeft"].values
        cr = df["contrastRight"].values
        m &= ~np.isnan(cl) | ~np.isnan(cr)
    elif "unsplit" in split:
        if "stim_l" in split:
            m &= ~np.isnan(df["contrastLeft"].values)
        elif "stim_r" in split:
            m &= ~np.isnan(df["contrastRight"].values)
    elif "block_only" in split or "act_block_only" == split:
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
    return m


def trial_masks_for_split(df, split):
    """
  Return boolean masks for high- vs low-prior trial groups (pleft >= 0.5 vs < 0.5).

  All populations use model P subjective prior (p_subjective_probabilityLeft).
  """
    pcol = prior_column_for_split(split)
    if pcol not in df.columns:
        raise KeyError(f"Missing prior column {pcol!r} for split {split!r}")

    m = split_fixed_condition_mask(df, split)
    cond0 = m & (df[pcol].values >= 0.5)
    cond1 = m & (df[pcol].values < 0.5)
    return cond0, cond1


def _trial_contrast_value(row, split):
    """Contrast on the stimulated side for this split."""
    if "unsplit" in split:
        trial_side = int(row["trial_side"])
        if trial_side == -1:
            return float(row["contrastLeft"])
        return float(row["contrastRight"])
    if "stim_r" in split:
        return float(row["contrastRight"])
    if "stim_l" in split:
        return float(row["contrastLeft"])
    return np.nan


def trial_s_peak_time_ms(row, split, steps_before_obs, population="S"):
    """Time (ms) of peak ||population|| in the split analysis window (plot time axis)."""
    align_kind = ALIGN.get(split, "stimOn_times")
    pre, post = PRE_POST[split]
    n_bins = split_n_bins(split)
    bounds = window_step_bounds(
        align_kind, row["length"], row["reaction_time"], steps_before_obs, pre, post
    )
    if bounds is None:
        return np.nan
    trace = row["traces"][population]
    seg = bin_trace_segment(trace, bounds[0], bounds[1], n_bins)
    mag = np.linalg.norm(seg, axis=1)
    if not np.any(np.isfinite(mag)):
        return np.nan
    peak_bin = int(np.argmax(mag))
    t_axis = time_axis_for_split(split, n_bins)
    return float(t_axis[peak_bin])


def trial_s_binned_signed(row, split, steps_before_obs, population="S"):
    """
    Binned S left/right channels in the split analysis window.

    Returns (t_axis, s_l, s_r) each shape (n_bins,). Model S is antisymmetric
    (channel 0 = S_l < 0, channel 1 = S_r > 0); do not negate S_l again.

    For S population on stimOn-aligned duringstim splits, applies the same 80 ms
    window cap as ``build_population_b_for_split`` so that trajectory plots are
    consistent with the prior-distance analysis and avoid the zero-padding
    artefact that masks g_s/d_s modulation.
    """
    align_kind = ALIGN.get(split, "stimOn_times")
    pre, post = PRE_POST[split]
    # Apply the same 80 ms cap for S that build_population_b_for_split uses so
    # that concordant-trial zero-padding does not cancel the g_s/d_s boost.
    if population == "S" and align_kind == "stimOn_times" and post > 0:
        post = min(post, S_DURINGSTIM_WINDOW_S)
        n_coarse = max(1, int(round(post / B_SIZE)))
        n_bins = n_coarse * max(1, int(B_SIZE // STS))
        t_axis = np.linspace(0.0, post * 1000.0, n_bins)
    else:
        n_bins = split_n_bins(split)
        t_axis = time_axis_for_split(split, n_bins)
    bounds = window_step_bounds(
        align_kind, row["length"], row["reaction_time"], steps_before_obs, pre, post
    )
    if bounds is None:
        return None
    trace = row["traces"][population]
    seg = bin_trace_segment(trace, bounds[0], bounds[1], n_bins)
    return t_axis, seg[:, 0], seg[:, 1]


def _p_block_left_mask(p_values):
    """High ITI subjective P -> P-block-left group (matches prior-distance analysis)."""
    return np.asarray(p_values, dtype=float) >= 0.5


def _split_short_label(split):
    return split.replace("act_block_duringstim_", "stim_")


def _overlay_block_histograms(
    ax, left_vals, right_vals, bins, xlabel, title, xlim=None,
    left_label="P block L", right_label="P block R",
):
    """Overlaid normalized histograms for left vs right block trials."""
    lv = np.asarray(left_vals, dtype=float)
    rv = np.asarray(right_vals, dtype=float)
    lv = lv[np.isfinite(lv)]
    rv = rv[np.isfinite(rv)]
    ax.hist(
        lv,
        bins=bins,
        density=True,
        alpha=0.45,
        color="#4C72B0",
        label=f"{left_label} (n={len(lv)})",
    )
    ax.hist(
        rv,
        bins=bins,
        density=True,
        alpha=0.45,
        color="#DD8452",
        label=f"{right_label} (n={len(rv)})",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.legend(fontsize=7, loc="upper right")


def plot_block_confound_distributions(
    session_dfs,
    steps_before_obs,
    out_dir,
    splits=None,
    condition_name="absence",
    rng_seed=42,
    prior_column=None,
):
    """
    Per-split overlays of P-block-L vs P-block-R for RT, contrast, and S peak time.

    Block groups use subjective ITI P (p_subjective_probabilityLeft >= 0.5 vs < 0.5),
    matching the prior-distance analysis grouping.

    Writes three 2x2 figures (one panel per act_block_duringstim split).
    """
    from scipy import stats

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = splits or s_prior_splits()
    prior_column = prior_column or PRIOR_COLUMN
    all_df = pd.concat(
        [apply_act_prior(df) for df in session_dfs],
        ignore_index=True,
    )

    rt_by_split = {}
    contrast_by_split = {}
    speak_by_split = {}
    for split in splits:
        m = split_fixed_condition_mask(all_df, split)
        sub = all_df.loc[m]
        left = _p_block_left_mask(sub[prior_column].values)
        rt_by_split[split] = (
            sub.loc[left, "reaction_time"].astype(float).values * DT_MS,
            sub.loc[~left, "reaction_time"].astype(float).values * DT_MS,
        )
        c_left, c_right = [], []
        s_left, s_right = [], []
        for idx in sub.index[left]:
            row = sub.loc[idx]
            c_left.append(_trial_contrast_value(row, split))
            s_left.append(trial_s_peak_time_ms(row, split, steps_before_obs))
        for idx in sub.index[~left]:
            row = sub.loc[idx]
            c_right.append(_trial_contrast_value(row, split))
            s_right.append(trial_s_peak_time_ms(row, split, steps_before_obs))
        contrast_by_split[split] = (np.asarray(c_left), np.asarray(c_right))
        speak_by_split[split] = (np.asarray(s_left), np.asarray(s_right))

    def _mannwhitney_p(a, b):
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 5 or len(b) < 5:
            return np.nan
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(p)

    fig_configs = [
        (
            "p_block_confound_rt_by_split.png",
            rt_by_split,
            np.linspace(80, 320, 25),
            "reaction time (ms)",
            (80, 320),
        ),
        (
            "p_block_confound_contrast_by_split.png",
            contrast_by_split,
            np.linspace(-0.03, 1.03, 22),
            "stimulus contrast",
            (-0.05, 1.05),
        ),
        (
            "p_block_confound_s_peak_by_split.png",
            speak_by_split,
            np.linspace(0, 150, 31),
            f"S peak time ({_time_xlabel(splits[0])})",
            (0, 150),
        ),
    ]

    for fname, data_by_split, bins, xlabel, xlim in fig_configs:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, split in zip(axes.ravel(), splits):
            lv, rv = data_by_split[split]
            p = _mannwhitney_p(lv, rv)
            p_txt = f"p={p:.3g}" if np.isfinite(p) else "p=n/a"
            _overlay_block_histograms(
                ax,
                lv,
                rv,
                bins=bins,
                xlabel=xlabel,
                title=f"{_split_short_label(split)}\n{p_txt}",
                xlim=xlim,
            )
            if "peak" in fname:
                _mark_align_event(ax, split)
        fig.suptitle(
            f"{condition_name}: P block L vs R (ITI {prior_column} >= 0.5 vs < 0.5)\n"
            f"seed={rng_seed}, n_sessions={len(session_dfs)}",
            fontsize=11,
        )
        fig.tight_layout()
        out_path = out_dir / fname
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

    return {"rt": rt_by_split, "contrast": contrast_by_split, "s_peak": speak_by_split}


CONTRAST_LEVELS = (1.0, 0.25, 0.125, 0.0625, 0.0)
CONTRAST_COLORS = {
    1.0: "#1b9e77",
    0.25: "#d95f02",
    0.125: "#7570b3",
    0.0625: "#e7298a",
    0.0: "#666666",
}


def _collect_s_traces_by_contrast(rows, split, steps_before_obs, population="S"):
    """Group binned ch0/ch1 traces by stimulated-side contrast for any population."""
    by_contrast = {c: {"s_l": [], "s_r": []} for c in CONTRAST_LEVELS}
    for _, row in rows.iterrows():
        contrast = _trial_contrast_value(row, split)
        if contrast not in by_contrast:
            continue
        out = trial_s_binned_signed(row, split, steps_before_obs, population=population)
        if out is None:
            continue
        _, s_l, s_r = out
        by_contrast[contrast]["s_l"].append(s_l)
        by_contrast[contrast]["s_r"].append(s_r)
    return by_contrast


def plot_p_block_s_trajectories(
    session_dfs,
    steps_before_obs,
    out_dir,
    splits=None,
    condition_name="absence",
    rng_seed=42,
    prior_column=None,
    min_trials_per_contrast=3,
    population="S",
):
    """
    One figure per split: 5 panels (one per contrast), each overlaying
    P-block-L vs P-block-R ch0 (solid) and ch1 (dashed) for any population.

    Blue = P-block-L (high prior).  Red = P-block-R (low prior).
    Solid = left channel (ch0), dashed = right channel (ch1).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = splits or s_prior_splits()
    prior_column = prior_column or PRIOR_COLUMN
    all_df = pd.concat(
        [apply_act_prior(df) for df in session_dfs],
        ignore_index=True,
    )

    GROUP_COLORS = {"left": "#2166ac", "right": "#d6604d"}  # blue=high, red=low
    GROUP_LABELS = {"left": "P-block-L", "right": "P-block-R"}
    ch_l_label = f"{population}_l"
    ch_r_label = f"{population}_r"

    for split in splits:
        m = split_fixed_condition_mask(all_df, split)
        sub = all_df.loc[m]
        left_mask = _p_block_left_mask(sub[prior_column].values)

        # Use population-specific time axis: S uses 80 ms cap, I/M use full 150 ms.
        align_kind = ALIGN.get(split, "stimOn_times")
        _, post_full = PRE_POST[split]
        if population == "S" and align_kind == "stimOn_times" and post_full > 0:
            post_eff = min(post_full, S_DURINGSTIM_WINDOW_S)
            n_coarse = max(1, int(round(post_eff / B_SIZE)))
            n_bins_pop = n_coarse * max(1, int(B_SIZE // STS))
            t_axis = np.linspace(0.0, post_eff * 1000.0, n_bins_pop)
        else:
            n_bins_pop = split_n_bins(split)
            t_axis = time_axis_for_split(split, n_bins_pop)

        groups = {
            "left":  _collect_s_traces_by_contrast(sub.loc[left_mask],  split, steps_before_obs, population=population),
            "right": _collect_s_traces_by_contrast(sub.loc[~left_mask], split, steps_before_obs, population=population),
        }

        valid_contrasts = [
            c for c in CONTRAST_LEVELS
            if max(len(groups["left"][c]["s_l"]), len(groups["right"][c]["s_l"])) >= min_trials_per_contrast
        ]
        n_panels = len(valid_contrasts)
        if n_panels == 0:
            continue

        fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), sharey=True)
        if n_panels == 1:
            axes = [axes]

        for ax, contrast in zip(axes, valid_contrasts):
            for gk in ("left", "right"):
                color = GROUP_COLORS[gk]
                s_l_t = groups[gk][contrast]["s_l"]
                s_r_t = groups[gk][contrast]["s_r"]
                n_c = len(s_l_t)
                if n_c < min_trials_per_contrast:
                    continue
                s_l_mean = np.mean(np.stack(s_l_t, axis=0), axis=0)
                s_r_mean = np.mean(np.stack(s_r_t, axis=0), axis=0)
                ax.plot(t_axis, s_l_mean, color=color, lw=1.8, ls="-",
                        label=f"{GROUP_LABELS[gk]} {ch_l_label} (n={n_c})")
                ax.plot(t_axis, s_r_mean, color=color, lw=1.8, ls="--",
                        label=f"{GROUP_LABELS[gk]} {ch_r_label}")
            _shade_offset_window(ax, t_axis)
            _mark_align_event(ax, split)
            ax.axhline(0, color="0.5", lw=0.8, ls=":")
            ax.set_xlabel(_time_xlabel(split))
            ax.set_title(f"c={contrast:g}", fontsize=10)
            ax.legend(fontsize=6, loc="best")

        axes[0].set_ylabel(f"{population} activity ({ch_l_label} solid, {ch_r_label} dashed)")
        fig.suptitle(
            f"{condition_name} — {_split_short_label(split)}\n"
            f"blue=P-block-L  red=P-block-R  "
            f"(solid={ch_l_label}  dashed={ch_r_label})  seed={rng_seed}",
            fontsize=10,
        )
        fig.tight_layout()
        out_path = out_dir / f"p_block_{population.lower()}_trajectory_{_split_short_label(split)}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


def run_block_confound_plots(
    base_dir,
    condition="absence",
    g_s=0.0,
    d_s=0.0,
    weights_json=None,
    n_sessions=N_SESSIONS_DEFAULT,
    blocks_per_session=BLOCKS_PER_SESSION_DEFAULT,
    max_obs_per_trial=400,
    rng_seed=42,
    zero_im_prior_mod=False,
):
    """Simulate one condition and write block-confound distribution figures."""
    base_dir = Path(base_dir)
    mp, _ = load_fitted_model(
        g_s=g_s,
        d_s=d_s,
        zero_im_prior_mod=zero_im_prior_mod,
        json_path=weights_json,
    )
    session_dfs, steps_before_obs, _ = simulate_condition_sessions(
        mp, n_sessions, blocks_per_session, max_obs_per_trial, rng_seed
    )
    out_dir = base_dir / condition / "figs" / "block_confounds"
    plot_block_confound_distributions(
        session_dfs,
        steps_before_obs,
        out_dir,
        condition_name=f"{condition} (g_s={g_s})",
        rng_seed=rng_seed,
    )
    for pop in ("S", "I"):
        plot_p_block_s_trajectories(
            session_dfs,
            steps_before_obs,
            out_dir,
            condition_name=f"{condition} (g_s={g_s})",
            rng_seed=rng_seed,
            population=pop,
        )
    return out_dir


def _mannwhitney_two_sided(a, b, min_n=5):
    from scipy import stats

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < min_n or len(b) < min_n:
        return np.nan, len(a), len(b), np.nan, np.nan
    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(p), len(a), len(b), float(np.median(a)), float(np.median(b))


def _prior_high_mask(values, prior_column=None):
    col = prior_column or PRIOR_COLUMN
    return np.asarray(values, dtype=float) >= 0.5


def compute_phase2_covariate_table(
    session_dfs,
    splits=None,
    prior_column=None,
):
    """
    Per-split Mann–Whitney p: high vs low ITI subjective P on adaptation / history covariates.
    """
    splits = splits or s_prior_splits()
    prior_column = prior_column or PRIOR_COLUMN
    all_df = pd.concat(
        [df.assign(session_id=s) for s, df in enumerate(session_dfs)],
        ignore_index=True,
    )
    metrics = (
        ("a_at_stim_mean", "adaptation a at stim (mean)"),
        ("a_at_stim_asym", "adaptation a at stim (L−R)"),
        ("s_norm_iti_mean", "ITI ||S|| mean"),
        ("trial_in_block", "trial index in block"),
        ("reaction_time", "reaction time (steps)"),
    )
    rows = []
    for split in splits:
        m = split_fixed_condition_mask(all_df, split)
        sub = all_df.loc[m]
        if sub.empty:
            continue
        high = _prior_high_mask(sub[prior_column].values, prior_column)
        for key, label in metrics:
            vals = sub[key].astype(float).values
            p, n_hi, n_lo, med_hi, med_lo = _mannwhitney_two_sided(vals[high], vals[~high])
            rows.append(
                {
                    "split": split,
                    "metric": key,
                    "metric_label": label,
                    "p_mannwhitney": p,
                    "n_high_prior": n_hi,
                    "n_low_prior": n_lo,
                    "median_high_prior": med_hi,
                    "median_low_prior": med_lo,
                    "median_diff_hi_minus_lo": (
                        med_hi - med_lo if np.isfinite(med_hi) and np.isfinite(med_lo) else np.nan
                    ),
                }
            )
        for block_val, block_name in ((-1, "block_left"), (1, "block_right")):
            bm = sub["block_side"].values == block_val
            if bm.sum() < 10:
                continue
            frac_high = float(np.mean(high[bm]))
            rows.append(
                {
                    "split": split,
                    "metric": f"p_high_prior_given_{block_name}",
                    "metric_label": f"frac high prior | {block_name}",
                    "p_mannwhitney": np.nan,
                    "n_high_prior": int(bm.sum()),
                    "n_low_prior": np.nan,
                    "median_high_prior": frac_high,
                    "median_low_prior": np.nan,
                    "median_diff_hi_minus_lo": np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_phase2_adaptation_confounds(
    session_dfs,
    steps_before_obs,
    out_dir,
    splits=None,
    condition_name="absence",
    rng_seed=42,
    prior_column=None,
):
    """Histograms of a_at_stim and ITI ||S|| for P-block-L vs P-block-R per split."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = splits or s_prior_splits()
    prior_column = prior_column or PRIOR_COLUMN
    all_df = pd.concat(session_dfs, ignore_index=True)

    fig_configs = [
        (
            "phase2_a_at_stim_by_prior_split.png",
            "a_at_stim_mean",
            np.linspace(0.2, 1.4, 25),
            "adaptation a at stim onset (mean)",
            (0.2, 1.4),
        ),
        (
            "phase2_s_norm_iti_by_prior_split.png",
            "s_norm_iti_mean",
            np.linspace(0, 0.35, 25),
            "ITI mean ||S||",
            (0, 0.35),
        ),
        (
            "phase2_trial_in_block_by_prior_split.png",
            "trial_in_block",
            np.linspace(-0.5, 80.5, 28),
            "trial index within block",
            (-0.5, 80.5),
        ),
    ]

    for fname, col, bins, xlabel, xlim in fig_configs:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, split in zip(axes.ravel(), splits):
            m = split_fixed_condition_mask(all_df, split)
            sub = all_df.loc[m]
            left = _prior_high_mask(sub[prior_column].values, prior_column)
            lv = sub.loc[left, col].astype(float).values
            rv = sub.loc[~left, col].astype(float).values
            p, _, _, _, _ = _mannwhitney_two_sided(lv, rv)
            p_txt = f"p={p:.3g}" if np.isfinite(p) else "p=n/a"
            _overlay_block_histograms(
                ax,
                lv,
                rv,
                bins=bins,
                xlabel=xlabel,
                title=f"{_split_short_label(split)}\n{p_txt}",
                xlim=xlim,
            )
        fig.suptitle(
            f"{condition_name}: {xlabel}\n"
            f"P block L vs R (ITI {prior_column} >= 0.5); seed={rng_seed}",
            fontsize=11,
        )
        fig.tight_layout()
        out_path = out_dir / fname
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


def _trial_bin_key(row, split, steps_before_obs):
    """Match key: block_side, contrast, trial_in_block decile within session-block."""
    contrast = _trial_contrast_value(row, split)
    if np.isnan(contrast):
        return None
    decile = int(min(9, row["trial_in_block"] // 10))
    return (int(row["block_side"]), float(contrast), decile, int(row.get("session_id", 0)))


def compute_matched_history_s_distances(
    session_dfs,
    steps_before_obs,
    splits=None,
    prior_column=None,
    min_trials_per_arm=2,
):
    """
    S prior-distance (curve mean) within matched (block_side, contrast, trial-in-block decile) bins.

    Compares full-split prior distance vs distance after averaging only bins with both
    high and low prior arms (matched on block_side, contrast, trial decile, session).
    """
    splits = splits or s_prior_splits()
    prior_column = prior_column or PRIOR_COLUMN
    all_df = pd.concat(
        [df.assign(session_id=s) for s, df in enumerate(session_dfs)],
        ignore_index=True,
    )
    summary_rows = []
    all_bin_rows = []
    for split in splits:
        m = split_fixed_condition_mask(all_df, split)
        sub = all_df.loc[m].copy().reset_index(drop=True)
        if len(sub) < 8:
            continue

        built = build_population_b_for_split(sub, split, "S", steps_before_obs)
        if built is None:
            continue
        b, n0, contrasts = built
        ys_true = np.zeros(b.shape[0], dtype=bool)
        ys_true[:n0] = True
        m0 = b[:n0].mean(axis=0)
        m1 = b[n0:].mean(axis=0)
        full_curve = np.sum((m0 - m1) ** 2, axis=0) / B_SIZE
        full_mean = float(np.mean(full_curve))

        cond0, cond1 = trial_masks_for_split(sub, split)
        trial_order = np.concatenate([np.where(cond0)[0], np.where(cond1)[0]])
        if len(trial_order) != b.shape[0]:
            continue

        bin_keys = []
        for ti in trial_order:
            bin_keys.append(_trial_bin_key(sub.iloc[int(ti)], split, steps_before_obs))

        matched_hi = []
        matched_lo = []
        split_bin_rows = []
        for key in sorted({k for k in bin_keys if k is not None}, key=str):
            in_bin = np.array([k == key for k in bin_keys], dtype=bool)
            if in_bin.sum() < 4:
                continue
            hi = in_bin & ys_true
            lo = in_bin & ~ys_true
            if hi.sum() < min_trials_per_arm or lo.sum() < min_trials_per_arm:
                continue
            hi_mean = b[hi].mean(axis=0)
            lo_mean = b[lo].mean(axis=0)
            matched_hi.append(hi_mean)
            matched_lo.append(lo_mean)
            mcurve = np.sum((hi_mean - lo_mean) ** 2, axis=0) / B_SIZE
            split_bin_rows.append(
                {
                    "split": split,
                    "block_side": key[0],
                    "contrast": key[1],
                    "trial_decile": key[2],
                    "session_id": key[3],
                    "n_high": int(hi.sum()),
                    "n_low": int(lo.sum()),
                    "bin_curve_mean": float(np.mean(mcurve)),
                }
            )

        if matched_hi:
            mh = np.stack(matched_hi, axis=0).mean(axis=0)
            ml = np.stack(matched_lo, axis=0).mean(axis=0)
            matched_curve = np.sum((mh - ml) ** 2, axis=0) / B_SIZE
            matched_mean = float(np.mean(matched_curve))
        else:
            matched_mean = np.nan

        summary_rows.append(
            {
                "split": split,
                "full_prior_distance_mean": full_mean,
                "matched_bin_distance_mean": matched_mean,
                "matched_fraction_of_full": (
                    matched_mean / full_mean if full_mean > 0 and np.isfinite(matched_mean) else np.nan
                ),
                "n_matched_bins": len(split_bin_rows),
                "n_trials_in_split": int(b.shape[0]),
            }
        )
        all_bin_rows.extend(split_bin_rows)

    return pd.DataFrame(summary_rows), pd.DataFrame(all_bin_rows)


def _covariate_match_key(row, split, rt_bin_steps=10, tib_bin_width=10):
    """Match key: session, contrast, trial_side, choice, feedback, RT bin, trial_in_block bin."""
    contrast = _trial_contrast_value(row, split)
    if np.isnan(contrast):
        return None
    rt = row.get("reaction_time", np.nan)
    if not np.isfinite(rt):
        return None
    tib = row.get("trial_in_block", np.nan)
    if not np.isfinite(tib):
        return None
    return (
        int(row.get("session_id", 0)),
        float(contrast),
        int(row["trial_side"]),
        int(row["choice"]),
        int(row["feedbackType"]),
        int(rt // rt_bin_steps),
        int(tib // tib_bin_width),
    )


def _matched_covariate_valid_indices(
    sub, split, prior_column, min_per_arm=1, rt_bin_steps=10, tib_bin_width=10
):
    """Trial indices in bins with both prior arms on the full covariate key."""
    from collections import defaultdict

    high = _prior_high_mask(sub[prior_column].values, prior_column)
    arms_by_key = defaultdict(lambda: {"high": [], "low": []})
    for i in range(len(sub)):
        key = _covariate_match_key(
            sub.iloc[i], split, rt_bin_steps=rt_bin_steps, tib_bin_width=tib_bin_width
        )
        if key is None:
            continue
        if high[i]:
            arms_by_key[key]["high"].append(i)
        else:
            arms_by_key[key]["low"].append(i)

    valid = []
    bin_info = []
    for key in sorted(arms_by_key.keys(), key=str):
        arms = arms_by_key[key]
        nh, nl = len(arms["high"]), len(arms["low"])
        if nh >= min_per_arm and nl >= min_per_arm:
            valid.extend(arms["high"] + arms["low"])
            bin_info.append(
                {
                    "session_id": key[0],
                    "contrast": key[1],
                    "trial_side": key[2],
                    "choice": key[3],
                    "feedbackType": key[4],
                    "rt_bin": key[5],
                    "trial_in_block_bin": key[6],
                    "n_high": nh,
                    "n_low": nl,
                }
            )
    return np.asarray(sorted(valid), dtype=int), bin_info


def compute_matched_covariate_prior_distances(
    session_dfs,
    steps_before_obs,
    splits=None,
    prior_column=None,
    nrand=NRAND_DEFAULT,
    rng_seed=42,
    min_per_arm=1,
    n_jobs=1,
    contrast_matched_null=True,
    rt_bin_steps=10,
    tib_bin_width=10,
):
    """
    S prior-distance on full split vs covariate-matched trial pool.

    Matched bins: (session, contrast, trial_side, choice, feedbackType,
    reaction_time bin, trial_in_block bin) with >= min_per_arm high and low trials.
    """
    splits = splits or s_prior_splits()
    prior_column = prior_column or PRIOR_COLUMN
    rng = np.random.RandomState(rng_seed)
    all_df = pd.concat(
        [df.assign(session_id=s) for s, df in enumerate(session_dfs)],
        ignore_index=True,
    )
    summary_rows = []
    all_bin_rows = []
    for split in splits:
        m = split_fixed_condition_mask(all_df, split)
        sub = all_df.loc[m].copy().reset_index(drop=True)
        if len(sub) < 8:
            continue

        built = build_population_b_for_split(sub, split, "S", steps_before_obs)
        if built is None:
            continue
        b, n0, contrasts = built
        full_deucs, _ = compute_population_distances(
            b,
            n0,
            nrand,
            rng,
            n_jobs=n_jobs,
            contrasts=contrasts,
            contrast_matched_null=contrast_matched_null,
        )
        if full_deucs is None:
            continue
        full_mean = float(np.mean(full_deucs[0] / B_SIZE))
        full_null_med = float(np.median([np.mean(c / B_SIZE) for c in full_deucs[1:]]))

        valid_idx, bin_info = _matched_covariate_valid_indices(
            sub,
            split,
            prior_column,
            min_per_arm=min_per_arm,
            rt_bin_steps=rt_bin_steps,
            tib_bin_width=tib_bin_width,
        )
        for bi in bin_info:
            bi["split"] = split
        all_bin_rows.extend(bin_info)

        if len(valid_idx) < 8:
            matched_mean = np.nan
            matched_null_med = np.nan
            matched_frac = 0.0
            n_matched_bins = len(bin_info)
        else:
            sub_m = sub.iloc[valid_idx].reset_index(drop=True)
            built_m = build_population_b_for_split(sub_m, split, "S", steps_before_obs)
            if built_m is None:
                matched_mean = np.nan
                matched_null_med = np.nan
                matched_frac = len(valid_idx) / len(sub)
                n_matched_bins = len(bin_info)
            else:
                bm, n0m, contrasts_m = built_m
                matched_deucs, _ = compute_population_distances(
                    bm,
                    n0m,
                    nrand,
                    rng,
                    n_jobs=n_jobs,
                    contrasts=contrasts_m,
                    contrast_matched_null=contrast_matched_null,
                )
                if matched_deucs is None:
                    matched_mean = np.nan
                    matched_null_med = np.nan
                else:
                    matched_mean = float(np.mean(matched_deucs[0] / B_SIZE))
                    matched_null_med = float(
                        np.median([np.mean(c / B_SIZE) for c in matched_deucs[1:]])
                    )
                matched_frac = len(valid_idx) / len(sub)
                n_matched_bins = len(bin_info)

        summary_rows.append(
            {
                "split": split,
                "full_curve_mean": full_mean,
                "full_null_median": full_null_med,
                "matched_curve_mean": matched_mean,
                "matched_null_median": matched_null_med,
                "matched_fraction_of_full": (
                    matched_mean / full_mean
                    if full_mean > 0 and np.isfinite(matched_mean)
                    else np.nan
                ),
                "matched_over_null_ratio": (
                    matched_mean / matched_null_med
                    if matched_null_med > 0 and np.isfinite(matched_mean)
                    else np.nan
                ),
                "trial_fraction_retained": matched_frac,
                "n_trials_in_split": int(len(sub)),
                "n_trials_matched": int(len(valid_idx)),
                "n_matched_bins": n_matched_bins,
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(all_bin_rows)


def run_matched_covariate_prior_test(
    base_dir,
    condition="absence",
    g_s=0.0,
    d_s=0.0,
    weights_json=None,
    n_sessions=N_SESSIONS_DEFAULT,
    blocks_per_session=BLOCKS_PER_SESSION_DEFAULT,
    max_obs_per_trial=400,
    rng_seed=42,
    nrand=NRAND_DEFAULT,
    n_jobs=1,
    prior_column=None,
    contrast_matched_null=True,
    min_per_arm=1,
    rt_bin_steps=10,
    tib_bin_width=10,
):
    """
    Matched trial-set prior distance: restrict to bins matched on covariates,
    then recompute S prior distance vs shuffle null.
    """
    base_dir = Path(base_dir)
    prior_column = prior_column or PRIOR_COLUMN
    mp, _ = load_fitted_model(g_s=g_s, d_s=d_s, json_path=weights_json)
    print(
        f"\n=== Matched covariate prior test: {condition} (g_s={g_s}, d_s={d_s}, "
        f"seed={rng_seed}) ==="
    )
    session_dfs, steps_before_obs, _ = simulate_condition_sessions(
        mp, n_sessions, blocks_per_session, max_obs_per_trial, rng_seed
    )
    out_dir = base_dir / condition / "figs" / "matched_covariate_prior"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df, bins_df = compute_matched_covariate_prior_distances(
        session_dfs,
        steps_before_obs,
        prior_column=prior_column,
        nrand=nrand,
        rng_seed=rng_seed,
        min_per_arm=min_per_arm,
        n_jobs=n_jobs,
        contrast_matched_null=contrast_matched_null,
        rt_bin_steps=rt_bin_steps,
        tib_bin_width=tib_bin_width,
    )
    summary_csv = out_dir / "matched_covariate_summary.csv"
    bins_csv = out_dir / "matched_covariate_bins.csv"
    summary_df.to_csv(summary_csv, index=False)
    bins_df.to_csv(bins_csv, index=False)
    print(f"Saved {summary_csv}")
    print(f"Saved {bins_csv}")

    for _, row in summary_df.iterrows():
        print(
            f"  {row['split']}: full={row['full_curve_mean']:.4f} "
            f"(null {row['full_null_median']:.4f}) | "
            f"matched={row['matched_curve_mean']:.4f} "
            f"(null {row['matched_null_median']:.4f}) | "
            f"retain={row['trial_fraction_retained']:.2%}"
        )

    summary = {
        "condition": condition,
        "g_s": g_s,
        "d_s": d_s,
        "rng_seed": rng_seed,
        "prior_column": prior_column,
        "match_keys": [
            "session_id",
            "contrast",
            "trial_side",
            "choice",
            "feedbackType",
            "rt_bin",
            "trial_in_block_bin",
        ],
        "rt_bin_steps": rt_bin_steps,
        "tib_bin_width": tib_bin_width,
        "min_per_arm": min_per_arm,
        "output_dir": str(out_dir),
        "per_split": summary_df.to_dict(orient="records"),
    }
    summary_json = out_dir / "matched_covariate_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def plot_phase2_matched_s_trajectories(
    session_dfs,
    steps_before_obs,
    out_dir,
    splits=None,
    prior_column=None,
    condition_name="absence",
    rng_seed=42,
):
    """
    Per split × contrast: S_l/S_r trajectories for high vs low prior when block_side is fixed.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = splits or s_prior_splits()
    prior_column = prior_column or PRIOR_COLUMN
    all_df = pd.concat(session_dfs, ignore_index=True)

    for split in splits:
        m = split_fixed_condition_mask(all_df, split)
        sub = all_df.loc[m]
        for block_val, block_title in ((-1, "block_left"), (1, "block_right")):
            block_sub = sub.loc[sub["block_side"] == block_val]
            if len(block_sub) < 20:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
            left_mask = _prior_high_mask(block_sub[prior_column].values, prior_column)
            t_axis = time_axis_for_split(split, split_n_bins(split))
            for ax, is_left, title in zip(
                axes,
                (True, False),
                ("P block L (ITI P≥0.5)", "P block R (ITI P<0.5)"),
            ):
                rows = block_sub.loc[left_mask if is_left else ~left_mask]
                by_contrast = _collect_s_traces_by_contrast(rows, split, steps_before_obs)
                for contrast in CONTRAST_LEVELS:
                    s_l_traces = by_contrast[contrast]["s_l"]
                    if len(s_l_traces) < 2:
                        continue
                    color = CONTRAST_COLORS[contrast]
                    s_l_mean = np.mean(np.stack(s_l_traces, axis=0), axis=0)
                    s_r_mean = np.mean(np.stack(by_contrast[contrast]["s_r"], axis=0), axis=0)
                    ax.plot(t_axis, s_l_mean, color=color, lw=1.6, ls="-", label=f"c={contrast:g} S_l")
                    ax.plot(t_axis, s_r_mean, color=color, lw=1.6, ls="--", label=f"c={contrast:g} S_r")
                _shade_offset_window(ax, t_axis)
                _mark_align_event(ax, split)
                ax.set_title(f"{title}\nn={len(rows)}")
                ax.legend(fontsize=5, loc="best")
            axes[0].set_ylabel("S (solid S_l, dashed S_r)")
            axes[0].set_xlabel(_time_xlabel(split))
            axes[1].set_xlabel(_time_xlabel(split))
            fig.suptitle(
                f"{condition_name} — {_split_short_label(split)} — {block_title}\n"
                f"matched block_side; seed={rng_seed}",
                fontsize=11,
            )
            fig.tight_layout()
            out_path = out_dir / f"phase2_s_traj_{_split_short_label(split)}_{block_title}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out_path}")


def run_phase2_adaptation_analysis(
    base_dir,
    condition="absence",
    g_s=0.0,
    d_s=0.0,
    weights_json=None,
    n_sessions=N_SESSIONS_DEFAULT,
    blocks_per_session=BLOCKS_PER_SESSION_DEFAULT,
    max_obs_per_trial=400,
    rng_seed=42,
    prior_column=None,
):
    """Phase 2: adaptation / history confounds for absence S-prior shuffle artifact."""
    base_dir = Path(base_dir)
    prior_column = prior_column or PRIOR_COLUMN
    mp, _ = load_fitted_model(g_s=g_s, d_s=d_s, json_path=weights_json)
    print(
        f"\n=== Phase 2 adaptation analysis: {condition} (g_s={g_s}, d_s={d_s}, "
        f"seed={rng_seed}) ==="
    )
    session_dfs, steps_before_obs, _ = simulate_condition_sessions(
        mp, n_sessions, blocks_per_session, max_obs_per_trial, rng_seed
    )
    out_dir = base_dir / condition / "figs" / "phase2_adaptation"
    out_dir.mkdir(parents=True, exist_ok=True)

    cov_df = compute_phase2_covariate_table(session_dfs, prior_column=prior_column)
    cov_csv = out_dir / "phase2_covariate_mannwhitney.csv"
    cov_df.to_csv(cov_csv, index=False)
    print(f"Saved {cov_csv}")

    plot_phase2_adaptation_confounds(
        session_dfs,
        steps_before_obs,
        out_dir,
        condition_name=f"{condition} (g_s={g_s})",
        rng_seed=rng_seed,
        prior_column=prior_column,
    )
    plot_phase2_matched_s_trajectories(
        session_dfs,
        steps_before_obs,
        out_dir,
        prior_column=prior_column,
        condition_name=f"{condition} (g_s={g_s})",
        rng_seed=rng_seed,
    )

    match_summary, match_bins = compute_matched_history_s_distances(
        session_dfs, steps_before_obs, prior_column=prior_column
    )
    match_summary.to_csv(out_dir / "phase2_matched_history_summary.csv", index=False)
    match_bins.to_csv(out_dir / "phase2_matched_history_bins.csv", index=False)
    print(f"Saved matched-history CSVs to {out_dir}")

    # Correlation: a_at_stim vs high prior label (pooled splits)
    all_df = pd.concat(session_dfs, ignore_index=True)
    splits = s_prior_splits()
    corr_rows = []
    for split in splits:
        m = split_fixed_condition_mask(all_df, split)
        sub = all_df.loc[m]
        if len(sub) < 20:
            continue
        hi = _prior_high_mask(sub[prior_column].values, prior_column).astype(float)
        for col in ("a_at_stim_mean", "s_norm_iti_mean", "trial_in_block"):
            x = sub[col].astype(float).values
            ok = np.isfinite(x)
            if ok.sum() < 20:
                continue
            r = float(np.corrcoef(x[ok], hi[ok])[0, 1])
            corr_rows.append({"split": split, "metric": col, "pearson_r_with_high_prior": r})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(out_dir / "phase2_prior_correlations.csv", index=False)

    summary = {
        "condition": condition,
        "g_s": g_s,
        "d_s": d_s,
        "rng_seed": rng_seed,
        "prior_column": prior_column,
        "output_dir": str(out_dir),
        "matched_history": match_summary.to_dict(orient="records"),
        "covariate_table": cov_df.to_dict(orient="records"),
        "prior_correlations": corr_df.to_dict(orient="records"),
    }
    with open(out_dir / "phase2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def _unpack_regde_curves(curves):
    """Normalize split- or combined-level regde entry to (real, nulls)."""
    if isinstance(curves, (list, tuple)) and len(curves) == 2:
        real, nulls = curves
    else:
        arr = np.asarray(curves, dtype=float)
        real = arr[0]
        nulls = arr[1:]
    nulls = np.atleast_2d(np.asarray(nulls, dtype=float))
    return np.asarray(real, dtype=float), nulls


def per_split_population_prior_metrics(res_dir, populations):
    """Per-split true vs shuffle prior-distance metrics for each population."""
    populations = tuple(populations)
    rows = []
    for split in s_prior_splits():
        regde_path = Path(res_dir) / f"{split}_regde.npy"
        if not regde_path.exists():
            continue
        regde = np.load(regde_path, allow_pickle=True).item()
        for pop in populations:
            if pop not in regde:
                continue
            real, nulls = _unpack_regde_curves(regde[pop])
            real_mean = float(np.mean(real))
            null_means = np.mean(nulls, axis=1)
            rows.append(
                {
                    "split": split,
                    "population": pop,
                    "curve_mean": real_mean,
                    "curve_amp": float(np.max(real) - np.min(real)),
                    "null_curve_mean_median": float(np.median(null_means)),
                    "null_curve_mean_min": float(np.min(null_means)),
                    "null_curve_mean_max": float(np.max(null_means)),
                    "p_mean": float(np.mean(null_means >= real_mean)),
                    "n_shuffles": int(nulls.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def _population_prior_summary_row(prior):
    """Scalar summary fields for one population_prior_test result."""
    if prior is None:
        return None
    return {
        k: prior[k]
        for k in (
            "population",
            "curve_mean",
            "curve_amp",
            "null_curve_mean_median",
            "null_curve_mean_min",
            "null_curve_mean_max",
            "p_mean",
            "p_amp",
            "p_offset",
            "p_gain",
            "early_mean_direct",
            "gain_late_mean_direct",
            "amp_euc",
            "significant_p_mean",
            "significant_p_amp",
            "significant_p_offset",
            "significant_p_gain",
        )
    }


def plot_population_prior_figures(
    condition, priors_by_pop, fig_dir, title, rng_seed=0, file_prefix="population_prior"
):
    """Bar chart + per-population shuffle controls for S/I/M prior distance."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    pops = [p for p in SIM_POPULATIONS if priors_by_pop.get(p) is not None]
    if not pops:
        return

    colors = {"S": "#4C72B0", "I": "#55A868", "M": "#C44E52"}
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(pops))
    width = 0.35
    true_vals = [priors_by_pop[p]["curve_mean"] for p in pops]
    null_med = [priors_by_pop[p]["null_curve_mean_median"] for p in pops]
    ax.bar(x - width / 2, true_vals, width, label="true labels", color=[colors[p] for p in pops])
    ax.bar(x + width / 2, null_med, width, label="null median", color="0.75", edgecolor="0.4")
    for i, p in enumerate(pops):
        ax.text(
            i - width / 2,
            true_vals[i],
            f"p={priors_by_pop[p]['p_mean']:.3g}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(pops)
    ax.set_ylabel("Combined curve_mean")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_bar = fig_dir / f"{file_prefix}_curve_mean_comparison.png"
    fig.savefig(out_bar, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_bar}")

    n_pops = len(pops)
    fig, axes = plt.subplots(1, n_pops, figsize=(4.5 * n_pops, 4.2), sharey=True)
    if n_pops == 1:
        axes = [axes]
    for ax, pop in zip(axes, pops):
        prior = priors_by_pop[pop]
        _draw_shuffle_control_panel(
            ax,
            prior,
            condition,
            rng_seed=rng_seed + hash(pop) % 1000,
            true_color=colors.get(pop, "#C44E52"),
            null_color=colors.get(pop, "0.78"),
            ylabel=f"{pop} prior distance (raw)",
        )
    fig.suptitle(f"{condition}: true vs contrast-matched nulls by population", y=1.02)
    fig.tight_layout()
    out_shuf = fig_dir / f"{file_prefix}_shuffle_controls.png"
    fig.savefig(out_shuf, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_shuf}")


def _write_population_prior_outputs(
    condition,
    priors_by_pop,
    res_dir,
    populations,
    out_dir,
    file_prefix,
    plot_title,
    rng_seed,
    model_params,
    prior_column,
    nrand,
    contrast_matched_null,
):
    """Write CSV/JSON/figures for multi-population prior-distance analysis."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    null_rows = []
    for pop in populations:
        prior = priors_by_pop.get(pop)
        row = _population_prior_summary_row(prior)
        if row is not None:
            summary_rows.append(row)
        if prior is None:
            print(f"  {pop}: no combined prior metrics")
            continue
        print(
            f"  {pop}: curve_mean={prior['curve_mean']:.4f}, "
            f"null med={prior['null_curve_mean_median']:.4f}, p_mean={prior['p_mean']:.4f}"
        )
        for i, (m, a) in enumerate(
            zip(prior["null_curve_means"], prior["null_curve_amps"])
        ):
            null_rows.append(
                {
                    "population": pop,
                    "shuffle_idx": i,
                    "curve_mean": float(m),
                    "curve_amp": float(a),
                }
            )

    summary_csv = out_dir / f"{file_prefix}_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"Saved {summary_csv}")

    split_df = per_split_population_prior_metrics(res_dir, populations)
    split_csv = out_dir / f"{file_prefix}_by_split.csv"
    split_df.to_csv(split_csv, index=False)
    print(f"Saved {split_csv}")

    if null_rows:
        null_csv = out_dir / f"{file_prefix}_shuffle_nulls.csv"
        pd.DataFrame(null_rows).to_csv(null_csv, index=False)
        print(f"Saved {null_csv}")

    plot_population_prior_figures(
        condition,
        priors_by_pop,
        out_dir,
        title=plot_title,
        rng_seed=rng_seed,
        file_prefix=file_prefix,
    )

    summary = {
        "condition": condition,
        "model_params": model_params,
        "rng_seed": rng_seed,
        "prior_column": prior_column,
        "populations": list(populations),
        "nrand": nrand,
        "contrast_matched_null": contrast_matched_null,
        "output_dir": str(out_dir),
        "res_dir": str(res_dir),
        "combined": summary_rows,
        "by_split": split_df.to_dict(orient="records"),
    }
    summary_json = out_dir / f"{file_prefix}_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_phase4_no_prior_mod_analysis(
    base_dir,
    weights_json=None,
    n_sessions=N_SESSIONS_DEFAULT,
    blocks_per_session=BLOCKS_PER_SESSION_DEFAULT,
    max_obs_per_trial=400,
    rng_seed=42,
    nrand=NRAND_DEFAULT,
    n_jobs=1,
    contrast_matched_null=True,
    populations=None,
    prior_column=None,
    constant_s0=False,
):
    """
    Phase 4 control: absence with all P→population modulations off (g_*=d_*=0).

    Tests whether S/I/M still show ITI-P prior distance ≫ shuffle when the generative
    model has no prior coupling to S, I, or M — i.e. label↔trial-context confound only.
    """
    base_dir = Path(base_dir)
    populations = tuple(populations or SIM_POPULATIONS)
    prior_column = prior_column or PRIOR_COLUMN
    mp, _ = load_fitted_model(zero_all_prior_mod=True, json_path=weights_json)
    model_params = {
        "g_s": mp["g_s"],
        "d_s": mp["d_s"],
        "g_i": mp["g_i"],
        "d_i": mp["d_i"],
        "g_m": mp["g_m"],
        "d_m": mp["d_m"],
    }
    s0_label = "constant S0=contrast" if constant_s0 else "stochastic S0"
    log_canonical_analysis_banner()
    print(
        f"\n=== Phase 4 no prior modulation: absence "
        f"(g_s=d_s=g_i=d_i=g_m=d_m=0, {s0_label}, populations={populations}, "
        f"seed={rng_seed}) ==="
    )
    session_dfs, steps_before_obs, _ = simulate_condition_sessions(
        mp,
        n_sessions,
        blocks_per_session,
        max_obs_per_trial,
        rng_seed,
        constant_s0=constant_s0,
    )

    out_name = "phase4_no_prior_mod_constant_s0" if constant_s0 else "phase4_no_prior_mod"
    file_prefix = out_name
    out_dir = base_dir / "absence" / "figs" / out_name
    res_dir = out_dir / "res"
    priors_by_pop, res_dir = _population_prior_from_sessions(
        session_dfs,
        steps_before_obs,
        nrand,
        rng_seed,
        prior_column,
        populations,
        n_jobs=n_jobs,
        contrast_matched_null=contrast_matched_null,
        res_dir=res_dir,
    )

    splits = s_prior_splits()
    contrast_df = write_multi_population_split_contrast_diagnostics(
        session_dfs,
        steps_before_obs,
        splits,
        nrand,
        rng_seed,
        out_dir,
        populations,
        out_csv=f"{file_prefix}_split_contrast.csv",
        contrast_matched_null=contrast_matched_null,
        prior_column=prior_column,
    )
    for pop in ("S", "I"):
        plot_p_block_s_trajectories(
            session_dfs,
            steps_before_obs,
            out_dir,
            splits=splits,
            condition_name=f"absence (no P mod, {s0_label})",
            rng_seed=rng_seed,
            prior_column=prior_column,
            population=pop,
        )

    summary = _write_population_prior_outputs(
        f"absence (no P mod, {s0_label})",
        priors_by_pop,
        res_dir,
        populations,
        out_dir,
        file_prefix=file_prefix,
        plot_title=f"Phase 4: S/I/M prior distance (all g_*=d_*=0, {s0_label})",
        rng_seed=rng_seed,
        model_params=model_params,
        prior_column=prior_column,
        nrand=nrand,
        contrast_matched_null=contrast_matched_null,
    )
    summary["constant_s0"] = constant_s0
    summary["by_split_contrast"] = contrast_df.to_dict(orient="records")
    with open(out_dir / f"{file_prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_unsplit_prior_distance_analysis(
    base_dir,
    case,
    weights_json=None,
    n_sessions=N_SESSIONS_DEFAULT,
    blocks_per_session=BLOCKS_PER_SESSION_DEFAULT,
    max_obs_per_trial=400,
    rng_seed=42,
    nrand=NRAND_DEFAULT,
    n_jobs=1,
    contrast_matched_null=True,
    populations=None,
    prior_column=None,
    unsplit_mode="stim_side",
):
    """
    Experiment A: prior distance without f1/f2 choice×feedback splits.

    unsplit_mode:
      - ``stim_side`` (default): stim_l + stim_r unsplit splits, stacked.
      - ``fully``: single pool of all duringstim trials (L+R mixed; S artefact risk).

    case: ``phase4`` (all g_*=d_*=0) or ``absence`` (fitted I/M, g_s=d_s=0).
    """
    base_dir = Path(base_dir)
    populations = tuple(populations or SIM_POPULATIONS)
    prior_column = prior_column or PRIOR_COLUMN
    case = case.lower()
    unsplit_mode = unsplit_mode.lower().replace("-", "_")
    if unsplit_mode not in ("stim_side", "fully"):
        raise ValueError(f"Unknown unsplit_mode: {unsplit_mode!r}")
    fully = unsplit_mode == "fully"
    log_canonical_analysis_banner()
    if case == "phase4":
        mp, _ = load_fitted_model(zero_all_prior_mod=True, json_path=weights_json)
        suffix = "fully_unsplit" if fully else "unsplit"
        condition_name = f"absence (no P mod, {suffix})"
        out_name = f"phase4_no_prior_mod_{suffix}"
        plot_title = (
            "Phase 4 fully unsplit: S/I/M prior distance (all g_*=d_*=0)"
            if fully
            else "Phase 4 unsplit: S/I/M prior distance (all g_*=d_*=0)"
        )
    elif case == "absence":
        mp, _ = load_fitted_model(g_s=0.0, d_s=0.0, json_path=weights_json)
        suffix = "fully_unsplit" if fully else "unsplit"
        condition_name = f"absence ({suffix})"
        out_name = f"absence_{suffix}"
        plot_title = (
            "Absence fully unsplit: S/I/M prior distance (g_s=d_s=0, fitted I/M)"
            if fully
            else "Absence unsplit: S/I/M prior distance (g_s=d_s=0, fitted I/M)"
        )
    else:
        raise ValueError(f"Unknown unsplit case: {case!r} (expected 'phase4' or 'absence')")

    model_params = {
        "g_s": mp["g_s"],
        "d_s": mp["d_s"],
        "g_i": mp["g_i"],
        "d_i": mp["d_i"],
        "g_m": mp["g_m"],
        "d_m": mp["d_m"],
    }
    print(
        f"\n=== Unsplit prior distance: {case} "
        f"(mode={unsplit_mode}, populations={populations}, seed={rng_seed}) ==="
    )
    session_dfs, steps_before_obs, _ = simulate_condition_sessions(
        mp,
        n_sessions,
        blocks_per_session,
        max_obs_per_trial,
        rng_seed,
    )

    out_root = base_dir / "unsplit_prior" / f"seed_{rng_seed}" / out_name
    out_dir = out_root / "figs"
    res_dir = out_root / "res"
    file_prefix = out_name
    if fully:
        splits = s_prior_splits_fully_unsplit()
        timeframe = FULLY_UNSPLIT_PRIOR_TIMEFRAME
    else:
        splits = s_prior_splits_unsplit()
        timeframe = UNSPLIT_PRIOR_TIMEFRAME

    priors_by_pop, res_dir = _population_prior_from_sessions(
        session_dfs,
        steps_before_obs,
        nrand,
        rng_seed,
        prior_column,
        populations,
        n_jobs=n_jobs,
        contrast_matched_null=contrast_matched_null,
        res_dir=res_dir,
        splits=splits,
        timeframe=timeframe,
    )

    contrast_df = write_multi_population_split_contrast_diagnostics(
        session_dfs,
        steps_before_obs,
        splits,
        nrand,
        rng_seed,
        out_dir,
        populations,
        out_csv=f"{file_prefix}_split_contrast.csv",
        contrast_matched_null=contrast_matched_null,
        prior_column=prior_column,
    )

    summary = _write_population_prior_outputs(
        condition_name,
        priors_by_pop,
        res_dir,
        populations,
        out_dir,
        file_prefix=file_prefix,
        plot_title=plot_title,
        rng_seed=rng_seed,
        model_params=model_params,
        prior_column=prior_column,
        nrand=nrand,
        contrast_matched_null=contrast_matched_null,
    )
    summary["unsplit"] = True
    summary["unsplit_mode"] = unsplit_mode
    summary["case"] = case
    summary["splits"] = list(splits)
    summary["canonical_analysis"] = CANONICAL_PRIOR_DISTANCE_ANALYSIS
    summary["by_split_contrast"] = contrast_df.to_dict(orient="records")
    with open(out_root / f"{file_prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved unsplit summary to {out_root / f'{file_prefix}_summary.json'}")
    return summary


def run_s_only_presence_analysis(
    base_dir,
    weights_json=None,
    n_sessions=N_SESSIONS_DEFAULT,
    blocks_per_session=BLOCKS_PER_SESSION_DEFAULT,
    max_obs_per_trial=400,
    rng_seed=42,
    nrand=NRAND_DEFAULT,
    n_jobs=1,
    contrast_matched_null=True,
    g_s_presence=None,
    d_s_presence=None,
):
    """
    S-only sensory prior: presence g_s/d_s with I/M prior modulation zeroed.

    Isolates direct P→S coupling (g_s, d_s) without I/M choice/threshold bias.
    """
    base_dir = Path(base_dir)
    weights_json = resolve_weights_json(weights_json)
    g_i_fit, d_i_fit = fitted_integrator_scales(weights_json)
    g_s = g_s_presence if g_s_presence is not None else g_i_fit
    d_s = d_s_presence if d_s_presence is not None else d_i_fit
    condition = "s_presence_only"
    print(
        f"\n=== S-only presence: {condition} "
        f"(g_s={g_s}, d_s={d_s}, g_i=d_i=g_m=d_m=0, seed={rng_seed}) ==="
    )
    process_condition(
        condition_name=condition,
        g_s=g_s,
        d_s=d_s,
        n_sessions=n_sessions,
        nrand=nrand,
        blocks_per_session=blocks_per_session,
        max_obs_per_trial=max_obs_per_trial,
        base_dir=base_dir,
        rng_seed=rng_seed,
        weights_json=weights_json,
        s_prior_only=True,
        n_jobs=n_jobs,
        contrast_matched_null=contrast_matched_null,
        zero_im_prior_mod=True,
    )
    run_block_confound_plots(
        base_dir,
        condition=condition,
        g_s=g_s,
        d_s=d_s,
        weights_json=weights_json,
        n_sessions=n_sessions,
        blocks_per_session=blocks_per_session,
        max_obs_per_trial=max_obs_per_trial,
        rng_seed=rng_seed,
        zero_im_prior_mod=True,
    )
    return {
        "condition": condition,
        "g_s": g_s,
        "d_s": d_s,
        "g_i": 0.0,
        "g_m": 0.0,
        "d_i": 0.0,
        "d_m": 0.0,
        "g_i_fitted": g_i_fit,
        "d_i_fitted": d_i_fit,
        "rng_seed": rng_seed,
        "output_dir": str(base_dir / condition),
    }


def run_s_presence_tuned_plots(
    base_dir,
    g_s=1.0,
    d_s=48.0,
    weights_json=None,
    n_sessions=N_SESSIONS_DEFAULT,
    blocks_per_session=BLOCKS_PER_SESSION_DEFAULT,
    max_obs_per_trial=400,
    rng_seed=42,
    nrand=NRAND_DEFAULT,
    n_jobs=1,
    contrast_matched_null=True,
    tag_suffix="",
    gs_outside_adaptation=False,
):
    """
    Full plot suite for tuned s_presence_only (g_s/d_s, I/M prior mod off).

    Writes to ``s_presence_tune/g_s{g_s}_d_s{d_s}/`` under base_dir:
    S/I prior curves, shuffle controls, population comparison, block confounds.
    """
    base_dir = Path(base_dir)
    weights_json = resolve_weights_json(weights_json)
    g_s, d_s = float(g_s), float(d_s)
    tag = f"g_s{g_s:g}_d_s{d_s:g}".replace(".", "p")
    if tag_suffix:
        tag = f"{tag}_{tag_suffix}"
    out_root = base_dir / "s_presence_tune" / tag
    res_dir = out_root / "res"
    fig_dir = out_root / "figs"
    if res_dir.exists():
        shutil.rmtree(res_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    condition = f"s_presence_only (g_s={g_s}, d_s={d_s}, g_i=d_i=g_m=d_m=0)"
    print(f"\n=== Tuned presence plots: {condition}, seed={rng_seed} ===")
    print(f"  output: {out_root}")

    mp, _ = load_fitted_model(
        g_s=g_s, d_s=d_s, zero_im_prior_mod=True, json_path=weights_json,
        gs_outside_adaptation=gs_outside_adaptation,
    )
    session_dfs, steps_before_obs, _ = simulate_condition_sessions(
        mp, n_sessions, blocks_per_session, max_obs_per_trial, rng_seed
    )

    priors_by_pop, res_dir = _population_prior_from_sessions(
        session_dfs,
        steps_before_obs,
        nrand,
        rng_seed,
        PRIOR_COLUMN,
        populations=("S", "I"),
        n_jobs=n_jobs,
        contrast_matched_null=contrast_matched_null,
        res_dir=res_dir,
    )

    s_prior = priors_by_pop.get("S")
    if s_prior is not None:
        plot_s_prior_figures(condition, s_prior, fig_dir)
    i_prior = priors_by_pop.get("I")
    if i_prior is not None:
        plot_s_prior_curve(f"{condition} — I", i_prior, fig_dir / "I")
        plot_s_shuffle_control(f"{condition} — I", i_prior, fig_dir / "I")
        plot_s_shuffle_control(
            f"{condition} — I", i_prior, fig_dir, filename="i_shuffle_control.png"
        )
        pd.DataFrame(
            [{k: v for k, v in i_prior.items() if not isinstance(v, (np.ndarray, list))}]
        ).to_csv(fig_dir / "I" / "i_prior_stats.csv", index=False)

    plot_population_prior_figures(
        condition,
        priors_by_pop,
        fig_dir,
        title=f"S/I prior distance — {condition}",
        rng_seed=rng_seed,
        file_prefix="si_prior",
    )

    bc_dir = fig_dir / "block_confounds"
    plot_block_confound_distributions(
        session_dfs,
        steps_before_obs,
        bc_dir,
        condition_name=condition,
        rng_seed=rng_seed,
    )
    for pop in ("S", "I"):
        plot_p_block_s_trajectories(
            session_dfs,
            steps_before_obs,
            bc_dir,
            condition_name=condition,
            rng_seed=rng_seed,
            population=pop,
        )

    summary = {
        "condition": condition,
        "g_s": g_s,
        "d_s": d_s,
        "rng_seed": rng_seed,
        "output_dir": str(out_root),
        "S": _population_prior_summary_row(s_prior),
        "I": _population_prior_summary_row(i_prior),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Saved tuned-case plots to {out_root}")
    return summary


def run_s_presence_i_scaled_plots(
    base_dir,
    weights_json=None,
    n_sessions=N_SESSIONS_DEFAULT,
    blocks_per_session=BLOCKS_PER_SESSION_DEFAULT,
    max_obs_per_trial=400,
    rng_seed=42,
    nrand=NRAND_DEFAULT,
    n_jobs=1,
    contrast_matched_null=True,
    scale_n_sessions=10,
    gs_outside_adaptation=False,
):
    """
    Plot suite with g_s/d_s scaled to integrator-comparable units:

    g_s = g_i_fitted * (median |S| / median |S0|), d_s = d_i_fitted.
    """
    weights_json = resolve_weights_json(weights_json)
    scale = integrator_comparable_s_params(
        weights_json=weights_json,
        rng_seed=rng_seed,
        n_sessions=scale_n_sessions,
        blocks_per_session=blocks_per_session,
        max_obs_per_trial=max_obs_per_trial,
    )
    g_s, d_s = scale["g_s"], scale["d_s"]
    print(
        f"\n=== Integrator-comparable S params (seed={rng_seed}) ==="
        f"\n  g_i={scale['g_i_fitted']:.4g}, d_i={scale['d_i_fitted']:.4g}"
        f"\n  |S|/|S0|={scale['s_over_s0']:.4g} "
        f"(S_med={scale['s_median_abs']:.4g}, S0_med={scale['s0_median_abs']:.4g}, "
        f"n={scale['n_samples']})"
        f"\n  → g_s={g_s:.4g}, d_s={d_s:.4g}"
    )
    tag = "i_scaled_gs_free" if gs_outside_adaptation else "i_scaled"
    summary = run_s_presence_tuned_plots(
        base_dir,
        g_s=g_s,
        d_s=d_s,
        weights_json=weights_json,
        n_sessions=n_sessions,
        blocks_per_session=blocks_per_session,
        max_obs_per_trial=max_obs_per_trial,
        rng_seed=rng_seed,
        nrand=nrand,
        n_jobs=n_jobs,
        contrast_matched_null=contrast_matched_null,
        tag_suffix=tag,
        gs_outside_adaptation=gs_outside_adaptation,
    )
    summary["scaling"] = scale
    (Path(summary["output_dir"]) / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    return summary


def default_gs_ds_tune_grid(g_i_fit, d_i_fit):
    """Default (g_s, d_s) grid: log-ish spacing around fitted integrator scales."""
    g_s_vals = sorted(
        {
            10.0,
            50.0,
            100.0,
            float(g_i_fit),
            float(g_i_fit) * 2,
            float(g_i_fit) * 5,
            float(g_i_fit) * 10,
            float(g_i_fit) * 25,
            float(g_i_fit) * 50,
        }
    )
    d_s_vals = sorted(
        {
            0.0,
            float(d_i_fit) * 0.25,
            float(d_i_fit) * 0.5,
            float(d_i_fit),
            float(d_i_fit) * 2,
            float(d_i_fit) * 5,
            float(d_i_fit) * 10,
            float(d_i_fit) * 25,
        }
    )
    return g_s_vals, d_s_vals


def _parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _prior_tune_row(g_s, d_s, priors_by_pop, alpha=0.01):
    """One sweep row with S and I prior metrics."""
    row = {"g_s": g_s, "d_s": d_s}
    for pop in ("S", "I"):
        p = priors_by_pop.get(pop)
        prefix = pop.lower()
        if p is None:
            row.update(
                {
                    f"{prefix}_curve_mean": np.nan,
                    f"{prefix}_null_median": np.nan,
                    f"{prefix}_p_mean": np.nan,
                    f"{prefix}_p_offset": np.nan,
                    f"{prefix}_p_gain": np.nan,
                    f"{prefix}_significant": False,
                }
            )
            continue
        row.update(
            {
                f"{prefix}_curve_mean": p["curve_mean"],
                f"{prefix}_null_median": p["null_curve_mean_median"],
                f"{prefix}_p_mean": p["p_mean"],
                f"{prefix}_p_offset": p["p_offset"],
                f"{prefix}_p_gain": p["p_gain"],
                f"{prefix}_gain_effect": p.get("gain_effect", np.nan),
                f"{prefix}_significant": bool(p["significant_p_mean"]),
                f"{prefix}_p_gain_significant": bool(p["significant_p_gain"]),
            }
        )
    row["s_significant_alpha"] = (
        row.get("s_p_mean", np.nan) < alpha if np.isfinite(row.get("s_p_mean", np.nan)) else False
    )
    row["s_p_gain_significant_alpha"] = (
        row.get("s_p_gain", np.nan) < alpha
        if np.isfinite(row.get("s_p_gain", np.nan))
        else False
    )
    row["s_mean_and_gain_significant"] = (
        row["s_significant_alpha"] and row["s_p_gain_significant_alpha"]
    )
    row["i_significant_alpha"] = (
        row.get("i_p_mean", np.nan) < alpha if np.isfinite(row.get("i_p_mean", np.nan)) else False
    )
    row["i_p_gain_significant_alpha"] = (
        row.get("i_p_gain", np.nan) < alpha
        if np.isfinite(row.get("i_p_gain", np.nan))
        else False
    )
    return row


def run_gs_ds_tune_sweep(
    base_dir,
    weights_json=None,
    n_sessions=N_SESSIONS_DEFAULT,
    blocks_per_session=BLOCKS_PER_SESSION_DEFAULT,
    max_obs_per_trial=400,
    rng_seed=42,
    nrand=NRAND_DEFAULT,
    n_jobs=1,
    contrast_matched_null=True,
    g_s_values=None,
    d_s_values=None,
    alpha=0.01,
    stop_on_s_significant=False,
    stop_on_s_p_gain=False,
    stop_on_s_mean_and_gain=False,
    gs_outside_adaptation=False,
):
    """
    Grid search over g_s/d_s with g_i=d_i=g_m=d_m=0.

    Simulates each (g_s, d_s) pair and reports S and I block-prior significance.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    weights_json = resolve_weights_json(weights_json)
    g_i_fit, d_i_fit = fitted_integrator_scales(weights_json)
    if g_s_values is None or d_s_values is None:
        default_g, default_d = default_gs_ds_tune_grid(g_i_fit, d_i_fit)
        g_s_values = default_g if g_s_values is None else g_s_values
        d_s_values = default_d if d_s_values is None else d_s_values

    out_csv = base_dir / "gs_ds_tune_sweep.csv"
    rows = []
    n_total = len(g_s_values) * len(d_s_values)
    print(
        f"\n=== g_s/d_s tune sweep (g_i=d_i=g_m=d_m=0, seed={rng_seed}, "
        f"n_sessions={n_sessions}, nrand={nrand}, alpha={alpha}) ==="
    )
    print(f"  g_i_fitted={g_i_fit:.4g}, d_i_fitted={d_i_fit:.4g}")
    print(f"  grid: {len(g_s_values)} g_s x {len(d_s_values)} d_s = {n_total} pairs")
    print(f"  g_s values: {g_s_values}")
    print(f"  d_s values: {d_s_values}")

    for i, g_s in enumerate(g_s_values):
        for j, d_s in enumerate(d_s_values):
            idx = i * len(d_s_values) + j + 1
            print(f"\n--- [{idx}/{n_total}] g_s={g_s}, d_s={d_s} ---")
            mp, _ = load_fitted_model(
                g_s=g_s,
                d_s=d_s,
                zero_im_prior_mod=True,
                json_path=weights_json,
                gs_outside_adaptation=gs_outside_adaptation,
            )
            session_dfs, steps_before_obs, _ = simulate_condition_sessions(
                mp,
                n_sessions,
                blocks_per_session,
                max_obs_per_trial,
                rng_seed,
            )
            priors_by_pop, _ = _population_prior_from_sessions(
                session_dfs,
                steps_before_obs,
                nrand,
                rng_seed,
                PRIOR_COLUMN,
                populations=("S", "I"),
                n_jobs=n_jobs,
                contrast_matched_null=contrast_matched_null,
            )
            row = _prior_tune_row(g_s, d_s, priors_by_pop, alpha=alpha)
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(
                f"  S: curve_mean={row['s_curve_mean']:.4g}, "
                f"null={row['s_null_median']:.4g}, p_mean={row['s_p_mean']:.4g}, "
                f"p_gain={row['s_p_gain']:.4g}, "
                f"mean_sig={row['s_significant_alpha']}, gain_sig={row['s_p_gain_significant_alpha']}"
            )
            print(
                f"  I: curve_mean={row['i_curve_mean']:.4g}, "
                f"null={row['i_null_median']:.4g}, p_mean={row['i_p_mean']:.4g}, "
                f"p_gain={row['i_p_gain']:.4g}, "
                f"mean_sig={row['i_significant_alpha']}, gain_sig={row['i_p_gain_significant_alpha']}"
            )
            if stop_on_s_mean_and_gain and row["s_mean_and_gain_significant"]:
                print(
                    f"  >> Stopping: S p_mean and p_gain significant at g_s={g_s}, d_s={d_s}"
                )
                break
            if stop_on_s_p_gain and row["s_p_gain_significant_alpha"]:
                print(f"  >> Stopping: S p_gain significant at g_s={g_s}, d_s={d_s}")
                break
            if stop_on_s_significant and row["s_significant_alpha"]:
                print(f"  >> Stopping: S significant at g_s={g_s}, d_s={d_s}")
                break
        else:
            continue
        break

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    s_hits = df[df["s_significant_alpha"]].sort_values("s_p_mean")
    s_gain_hits = df[df["s_p_gain_significant_alpha"]].sort_values("s_p_gain")
    s_both_hits = df[df["s_mean_and_gain_significant"]].sort_values(["s_p_mean", "s_p_gain"])
    summary = {
        "g_i_fitted": g_i_fit,
        "d_i_fitted": d_i_fit,
        "rng_seed": rng_seed,
        "n_sessions": n_sessions,
        "nrand": nrand,
        "alpha": alpha,
        "n_grid_pairs": n_total,
        "n_evaluated": len(df),
        "n_s_significant": int(s_hits.shape[0]),
        "n_s_p_gain_significant": int(s_gain_hits.shape[0]),
        "n_s_mean_and_gain_significant": int(s_both_hits.shape[0]),
        "best_s_row": s_hits.iloc[0].to_dict() if len(s_hits) else None,
        "best_s_p_gain_row": s_gain_hits.iloc[0].to_dict() if len(s_gain_hits) else None,
        "best_s_mean_and_gain_row": s_both_hits.iloc[0].to_dict() if len(s_both_hits) else None,
        "s_significant_rows": s_hits.to_dict(orient="records"),
        "s_p_gain_significant_rows": s_gain_hits.to_dict(orient="records"),
        "s_mean_and_gain_significant_rows": s_both_hits.to_dict(orient="records"),
        "csv": str(out_csv),
    }
    summary_path = base_dir / "gs_ds_tune_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved sweep to {out_csv}")
    print(f"Saved summary to {summary_path}")
    if len(s_both_hits):
        best = s_both_hits.iloc[0]
        print(
            f"\nBest S (p_mean + p_gain): g_s={best['g_s']}, d_s={best['d_s']}, "
            f"p_mean={best['s_p_mean']:.4g}, p_gain={best['s_p_gain']:.4g}, "
            f"I p_mean={best['i_p_mean']:.4g}, I p_gain={best['i_p_gain']:.4g}"
        )
    elif len(s_gain_hits):
        best = s_gain_hits.iloc[0]
        print(
            f"\nBest S p_gain: g_s={best['g_s']}, d_s={best['d_s']}, "
            f"p_gain={best['s_p_gain']:.4g}, p_mean={best['s_p_mean']:.4g}"
        )
    elif len(s_hits):
        best = s_hits.iloc[0]
        print(
            f"\nBest S-significant: g_s={best['g_s']}, d_s={best['d_s']}, "
            f"p_mean={best['s_p_mean']:.4g}, I p_mean={best['i_p_mean']:.4g}"
        )
    else:
        print("\nNo S-significant (g_s, d_s) pair found in grid.")
    return df, summary


def default_gs_tune_p_gain_grid(g_i_fit):
    """g_s values for p_gain tuning at fixed d_s."""
    return sorted(
        {
            1.0,
            2.0,
            3.0,
            5.0,
            8.0,
            10.0,
            12.0,
            15.0,
            20.0,
            25.0,
            30.0,
            40.0,
            50.0,
            float(g_i_fit) * 0.05,
            float(g_i_fit) * 0.1,
        }
    )


def run_gs_tune_p_gain(
    base_dir,
    d_s_fixed=50.0,
    weights_json=None,
    n_sessions=N_SESSIONS_DEFAULT,
    blocks_per_session=BLOCKS_PER_SESSION_DEFAULT,
    max_obs_per_trial=400,
    rng_seed=42,
    nrand=NRAND_DEFAULT,
    n_jobs=1,
    contrast_matched_null=True,
    g_s_values=None,
    alpha=0.01,
    require_p_mean=True,
    gs_outside_adaptation=False,
):
    """
    Tune g_s at fixed d_s until S p_gain is significant (optionally p_mean too).
    """
    weights_json = resolve_weights_json(weights_json)
    g_i_fit, d_i_fit = fitted_integrator_scales(weights_json)
    if g_s_values is None:
        g_s_values = default_gs_tune_p_gain_grid(g_i_fit)
    return run_gs_ds_tune_sweep(
        base_dir,
        weights_json=weights_json,
        n_sessions=n_sessions,
        blocks_per_session=blocks_per_session,
        max_obs_per_trial=max_obs_per_trial,
        rng_seed=rng_seed,
        nrand=nrand,
        n_jobs=n_jobs,
        contrast_matched_null=contrast_matched_null,
        g_s_values=g_s_values,
        d_s_values=[float(d_s_fixed)],
        alpha=alpha,
        stop_on_s_mean_and_gain=require_p_mean,
        stop_on_s_p_gain=not require_p_mean,
        gs_outside_adaptation=gs_outside_adaptation,
    )


def apply_random_prior_labels(session_dfs, rng, match_marginal=True):
    """Random 0.8/0.2 prior labels independent of P trace (same S trajectories)."""
    out = []
    for df in session_dfs:
        d = df.copy()
        n = len(d)
        if match_marginal:
            n_hi = int((d[PRIOR_COLUMN].values >= 0.5).sum())
            hi = np.zeros(n, dtype=bool)
            hi[rng.choice(n, size=n_hi, replace=False)] = True
        else:
            hi = rng.rand(n) >= 0.5
        d[RANDOM_PRIOR_COLUMN] = np.where(hi, 0.8, 0.2)
        out.append(d)
    return out


def _population_prior_from_sessions(
    session_dfs,
    steps_before_obs,
    nrand,
    rng_seed,
    prior_column,
    populations,
    n_jobs=1,
    contrast_matched_null=True,
    res_dir=None,
    splits=None,
    timeframe=None,
):
    """Run distance pipeline on fixed trajectories; return prior metrics per population."""
    populations = tuple(populations)
    splits = splits or s_prior_splits()
    timeframe = timeframe or S_PRIOR_TIMEFRAME
    tmp = res_dir is None
    if tmp:
        cm = tempfile.TemporaryDirectory()
        res_dir = Path(cm.name)
    else:
        res_dir = Path(res_dir)
        res_dir.mkdir(parents=True, exist_ok=True)
        cm = None
    try:
        rng = np.random.RandomState(rng_seed)
        with use_prior_column(prior_column):
            build_res_from_trajectories(
                session_dfs,
                splits,
                steps_before_obs,
                nrand,
                rng,
                res_dir,
                populations=populations,
                n_jobs=n_jobs,
                contrast_matched_null=contrast_matched_null,
            )
            stack_combined_timeframes(res_dir, [timeframe])
        return {
            pop: population_prior_test(res_dir, pop, timeframe=timeframe)
            for pop in populations
        }, res_dir
    finally:
        if cm is not None:
            cm.cleanup()


def _s_prior_from_sessions(
    session_dfs,
    steps_before_obs,
    nrand,
    rng_seed,
    prior_column,
    n_jobs=1,
    contrast_matched_null=True,
):
    """Run distance pipeline on fixed trajectories with a given prior column."""
    results, _ = _population_prior_from_sessions(
        session_dfs,
        steps_before_obs,
        nrand,
        rng_seed,
        prior_column,
        populations=("S",),
        n_jobs=n_jobs,
        contrast_matched_null=contrast_matched_null,
    )
    return results.get("S")


def run_random_prior_label_test(
    base_dir,
    n_random_replicates=50,
    n_sessions=N_SESSIONS_DEFAULT,
    blocks_per_session=BLOCKS_PER_SESSION_DEFAULT,
    max_obs_per_trial=400,
    rng_seed=42,
    nrand=100,
    n_jobs=1,
    weights_json=None,
    match_marginal=True,
    contrast_matched_null=True,
):
    """
    Absence: true ITI-P labels vs random 0.8/0.2 labels on the same S trajectories.
    If random labels land near shuffle-null range, the true effect is label coupling only.
    """
    base_dir = Path(base_dir)
    weights_json = resolve_weights_json(weights_json)
    n_jobs = max(1, int(n_jobs))
    mp, _ = load_fitted_model(g_s=0.0, d_s=0.0, json_path=weights_json)
    print(f"\n=== Random ITI prior labels (absence, seed={rng_seed}) ===")
    session_dfs, steps_before_obs, _ = simulate_condition_sessions(
        mp, n_sessions, blocks_per_session, max_obs_per_trial, rng_seed
    )

    true_s = _s_prior_from_sessions(
        session_dfs,
        steps_before_obs,
        nrand,
        rng_seed,
        PRIOR_COLUMN,
        n_jobs=n_jobs,
        contrast_matched_null=contrast_matched_null,
    )
    if true_s is None:
        raise RuntimeError("No S prior metrics for true ITI labels")

    null_means = true_s.get("null_curve_means") or []
    random_rows = []
    for rep in range(n_random_replicates):
        rep_rng = np.random.RandomState(rng_seed + 10_000 + rep)
        rand_dfs = apply_random_prior_labels(session_dfs, rep_rng, match_marginal=match_marginal)
        rs = _s_prior_from_sessions(
            rand_dfs,
            steps_before_obs,
            nrand,
            rng_seed + 20_000 + rep,
            RANDOM_PRIOR_COLUMN,
            n_jobs=n_jobs,
            contrast_matched_null=contrast_matched_null,
        )
        if rs is None:
            continue
        random_rows.append(
            {
                "replicate": rep,
                "curve_mean": rs["curve_mean"],
                "curve_amp": rs["curve_amp"],
                "p_mean": rs["p_mean"],
            }
        )
        if (rep + 1) % 10 == 0 or rep + 1 == n_random_replicates:
            print(f"  random replicate {rep + 1}/{n_random_replicates}")

    rand_df = pd.DataFrame(random_rows)
    rand_arr = rand_df["curve_mean"].to_numpy(dtype=float)
    true_mean = float(true_s["curve_mean"])
    null_arr = np.asarray(null_means, dtype=float)

    summary = {
        "true_iti_curve_mean": true_mean,
        "true_iti_curve_amp": float(true_s["curve_amp"]),
        "true_shuffle_null_median": float(np.median(null_arr)) if null_arr.size else np.nan,
        "true_shuffle_null_max": float(np.max(null_arr)) if null_arr.size else np.nan,
        "true_p_mean": float(true_s["p_mean"]),
        "n_random_replicates": len(rand_df),
        "random_curve_mean_median": float(np.median(rand_arr)),
        "random_curve_mean_min": float(np.min(rand_arr)),
        "random_curve_mean_max": float(np.max(rand_arr)),
        "random_curve_mean_q05": float(np.percentile(rand_arr, 5)),
        "random_curve_mean_q95": float(np.percentile(rand_arr, 95)),
        "n_random_gte_true": int(np.sum(rand_arr >= true_mean)),
        "p_random_gte_true": float(np.mean(rand_arr >= true_mean)),
        "match_marginal_high_count": match_marginal,
        "interpretation": (
            "random≈shuffle and true≫random => label coupling artifact; "
            "random≈true => effect not from ITI-P label alone"
        ),
    }

    out_dir = base_dir / "absence" / "figs" / "random_prior_labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    rand_df.to_csv(out_dir / "random_prior_replicates.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "random_prior_summary.csv", index=False)
    with open(out_dir / "random_prior_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def build_population_b_for_split(df, split, population, steps_before_obs):
    """Per-trial binned model trajectories for one population: (trials, 2, bins).

    For stimOn-aligned splits with a post-stim window, trials that end before
    the window closes are extended by borrowing the start of the **next trial's
    ITI** — matching the fill-from-next logic in model_functions.py
    ``prior_distance_I_M_by_choice_and_prior``.  If the next trial is in a
    different session or its ITI is too short to fill the gap, the trial is
    skipped (no zero-padding).
    """
    align_kind = ALIGN.get(split, "stimOn_times")
    pre, post = PRE_POST[split]
    n_bins = split_n_bins(split)
    cond0, cond1 = trial_masks_for_split(df, split)
    idx0 = np.where(cond0)[0]
    idx1 = np.where(cond1)[0]
    if len(idx0) < 2 or len(idx1) < 2:
        return None

    # Number of post-stim steps that the analysis window spans.
    # S uses a shorter cap (S_DURINGSTIM_WINDOW_S) so that we only capture
    # genuine stim-driven activity without borrowing large chunks of next-trial
    # ITI at high contrast (where fast RTs leave most of the 150ms window empty).
    use_fill = align_kind == "stimOn_times" and post > 0
    if use_fill:
        eff_post = min(post, S_DURINGSTIM_WINDOW_S) if population == "S" else post
        win_post_steps = int(round(eff_post * 1000.0 / DT_MS))
        n_bins = int(round(eff_post / B_SIZE)) * max(1, int(B_SIZE // STS))
    else:
        win_post_steps = 0

    def trials_to_stack(idxs):
        chunks = []
        contrasts = []
        n_df = len(df)
        for ti in idxs:
            row = df.iloc[ti]
            trace = row["traces"][population]

            if use_fill:
                # Fill-from-next-ITI logic (mirrors model_functions.py).
                post_avail = max(0, row["length"] - steps_before_obs)
                take_post = min(win_post_steps, post_avail)

                if take_post < win_post_steps:
                    need = win_post_steps - take_post
                    ti_next = ti + 1
                    can_fill = (
                        ti_next < n_df
                        and df.iloc[ti_next]["trial_idx"] == row["trial_idx"] + 1
                        and min(steps_before_obs, df.iloc[ti_next]["length"]) >= need
                    )
                    if not can_fill:
                        continue  # skip rather than zero-pad
                    next_trace = df.iloc[ti_next]["traces"][population]
                    parts = []
                    if take_post > 0:
                        parts.append(trace[steps_before_obs : steps_before_obs + take_post])
                    parts.append(next_trace[:need])
                    combined = np.vstack(parts)  # (win_post_steps, 2)
                    seg = bin_trace_segment(combined, 0, win_post_steps, n_bins)
                else:
                    seg = bin_trace_segment(trace, steps_before_obs, steps_before_obs + win_post_steps, n_bins)
            else:
                # Non-stim-aligned or pre-action window: clipped logic.
                bounds = window_step_bounds(
                    align_kind, row["length"], row["reaction_time"], steps_before_obs, pre, post
                )
                if bounds is None:
                    continue
                s, e = bounds
                seg = bin_trace_segment(trace, s, e, n_bins)

            chunks.append(seg.T)
            contrasts.append(_trial_contrast_value(row, split))
        if not chunks:
            return None, None
        return np.stack(chunks, axis=0), np.asarray(contrasts, dtype=float)

    b0, c0 = trials_to_stack(idx0)
    b1, c1 = trials_to_stack(idx1)
    if b0 is None or b1 is None:
        return None
    return np.concatenate([b0, b1], axis=0), len(b0), np.concatenate([c0, c1])


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


def compute_population_distances(
    b, n0, nrand, rng, n_jobs=1, contrasts=None, contrast_matched_null=True
):
    """
    Euclidean prior-distance curves for one population.

    Nulls: by default, contrast-matched label shuffle — within each contrast bin,
    pseudo high/low groups preserve the real per-contrast trial counts (fair control
    vs unrestricted shuffle that mixes contrast across groups).
    b shape: (n_trials, n_channels, n_bins); first n0 trials are high-prior group.
    """
    ntr = b.shape[0]
    if ntr < 4 or n0 < 2 or (ntr - n0) < 2:
        return None

    ys_true = np.zeros(ntr, dtype=bool)
    ys_true[:n0] = True
    means = [b[:n0].mean(axis=0), b[n0:].mean(axis=0)]

    use_contrast_null = contrast_matched_null and contrasts is not None
    if use_contrast_null:
        contrasts = np.asarray(contrasts, dtype=float)
        if contrasts.shape[0] != ntr or np.any(np.isnan(contrasts)):
            use_contrast_null = False

    n_jobs = max(1, int(n_jobs))
    if n_jobs > 1 and nrand >= n_jobs:
        from null_shuffle_worker import null_shuffle_chunk, null_shuffle_contrast_chunk

        chunk = int(np.ceil(nrand / n_jobs))
        tasks = []
        start = 0
        while start < nrand:
            n_chunk = min(chunk, nrand - start)
            tasks.append((n_chunk, int(rng.randint(0, 2**31 - 1))))
            start += n_chunk
        with _null_shuffle_executor(min(n_jobs, len(tasks))) as pool:
            if use_contrast_null:
                futures = [
                    pool.submit(null_shuffle_contrast_chunk, b, n0, contrasts, nc, seed)
                    for nc, seed in tasks
                ]
            else:
                futures = [
                    pool.submit(null_shuffle_chunk, b, n0, ys_true, nc, seed)
                    for nc, seed in tasks
                ]
            for fut in futures:
                means.extend(fut.result())
    else:
        if use_contrast_null:
            from null_shuffle_worker import contrast_matched_shuffle_labels

            for _ in range(nrand):
                pseudo = contrast_matched_shuffle_labels(contrasts, n0, rng)
                means.append(b[pseudo].mean(axis=0))
                means.append(b[~pseudo].mean(axis=0))
        else:
            for _ in range(nrand):
                perm = ys_true.copy()
                rng.shuffle(perm)
                means.append(b[perm].mean(axis=0))
                means.append(b[~perm].mean(axis=0))

    means_arr = np.stack(means, axis=0)
    diff = means_arr[0::2] - means_arr[1::2]
    d_eucs = np.sum(diff**2, axis=1)
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


def build_split_results(
    df, split, steps_before_obs, nrand, rng, populations=None, n_jobs=1, contrast_matched_null=True,
):
    """Build {split}.npy / {split}_regde.npy outputs from pooled model trajectories."""
    populations = populations or MODEL_POPULATIONS
    regde = {}
    regxn = {}
    r = {}
    for pop in populations:
        built = build_population_b_for_split(df, split, pop, steps_before_obs)
        if built is None:
            continue
        b, n0, contrasts = built
        dist = compute_population_distances(
            b,
            n0,
            nrand,
            rng,
            n_jobs=n_jobs,
            contrasts=contrasts,
            contrast_matched_null=contrast_matched_null,
        )
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


def s_prior_splits_unsplit():
    """Duringstim trials pooled per stim side (no choice×feedback f1/f2 splits)."""
    return list(UNSPLIT_PRIOR_SPLITS)


def s_prior_splits_fully_unsplit():
    """All duringstim trials in one pool (no stim side or choice×feedback splits)."""
    return [FULLY_UNSPLIT_PRIOR_SPLIT]


def _splits_for_timeframe(timeframe):
    if timeframe in SIM_TIMEFRAME_SPLITS:
        return SIM_TIMEFRAME_SPLITS[timeframe]
    import analysis_functions as af

    return list(af.run_align[timeframe])


def build_res_from_trajectories(
    session_dfs,
    splits,
    steps_before_obs,
    nrand,
    rng,
    pth_res,
    populations=None,
    n_jobs=1,
    contrast_matched_null=True,
):
    """Write per-split res files by pooling trials across simulated sessions."""
    pth_res.mkdir(parents=True, exist_ok=True)
    all_df = pd.concat(session_dfs, ignore_index=True)
    n_saved = 0
    t0 = time.perf_counter()
    for i, split in enumerate(splits):
        out = build_split_results(
            all_df,
            split,
            steps_before_obs,
            nrand,
            rng,
            populations=populations,
            n_jobs=n_jobs,
            contrast_matched_null=contrast_matched_null,
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
    for timeframe in timeframes:
        splits = _splits_for_timeframe(timeframe)
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
    splits = _splits_for_timeframe(timeframe)
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


def _direct_early_mean(curve, n_early=PRIOR_OFFSET_BINS):
    """Mean of the first n_early bins on the real curve (no label shuffle)."""
    curve = np.asarray(curve, dtype=float)
    n_early = min(n_early, len(curve))
    if n_early < 1:
        return np.nan
    return float(np.mean(curve[:n_early]))


def _direct_gain_late_mean(curve, n_early=PRIOR_OFFSET_BINS):
    """
    Mean of bins 4+ after subtracting this curve's early-bin mean.

    Same bin window as p_gain, but offset removal uses only the real curve —
  no label-shuffle nulls.
    """
    curve = np.asarray(curve, dtype=float)
    if len(curve) <= 4:
        return np.nan
    n_early = min(n_early, len(curve))
    offset = float(np.mean(curve[:n_early]))
    shifted = curve - offset
    return float(np.mean(shifted[4:]))


def _one_sided_sign_p(diffs):
    """One-sided p-value: P(diff > 0) under split-level sign-flip null (no label shuffle)."""
    diffs = [float(d) for d in diffs if np.isfinite(d)]
    if not diffs:
        return np.nan
    k = sum(d > 0 for d in diffs)
    n = len(diffs)
    try:
        from scipy.stats import binomtest

        return float(binomtest(k, n, 0.5, alternative="greater").pvalue)
    except ImportError:
        from math import comb

        return sum(comb(n, i) for i in range(k, n + 1)) / (2**n)


def s_prior_metrics_per_split(pth_res, alpha=0.01, timeframe="act_block_duringstim"):
    """Per-split S prior metrics on real curves (for paired presence vs absence tests)."""
    _, _, splits = _combined_names(timeframe)
    out = {}
    for split in splits:
        regde_path = Path(pth_res) / f"{split}_regde.npy"
        if not regde_path.exists():
            continue
        regde = np.load(regde_path, allow_pickle=True).item()
        if "S" not in regde:
            continue
        curves = np.asarray(regde["S"], dtype=float)
        if curves.ndim == 1:
            real = curves
            nulls = np.empty((0, len(real)))
        else:
            real = curves[0]
            nulls = curves[1:]
        nulls = np.atleast_2d(nulls)
        offset = _prior_offset_stats(real, nulls)
        gain = _prior_gain_stats(real, nulls, alpha=alpha)
        out[split] = {
            "curve_mean": float(np.mean(real)),
            "early_mean_direct": _direct_early_mean(real),
            "gain_late_mean_direct": _direct_gain_late_mean(real),
            "p_mean": float(np.mean(np.mean(nulls, axis=1) >= np.mean(real))),
            "p_offset": offset["p_offset"],
            "p_gain": gain["p_gain"],
        }
    return out


def _one_sided_pres_gt_abs_p(null_values, presence_value):
    """One-sided p: P(absence replicate >= presence) under absence-only null."""
    null = np.asarray(null_values, dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(presence_value):
        return np.nan
    return float(np.mean(null >= float(presence_value)))


def _null_summary(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "median": np.nan, "q05": np.nan, "q95": np.nan}
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "q05": float(np.percentile(arr, 5)),
        "q95": float(np.percentile(arr, 95)),
    }


def pres_abs_p_values_absence_null(presence_s, absence_null_rows):
    """
    Presence vs absence p-values from repeated absence simulations.

    Fixed presence metrics are compared to an absence-only null (different trial
    seeds each replicate). One-sided p: P(absence_null >= presence).
    """
    if presence_s is None or not absence_null_rows:
        return {
            "p_pres_abs_mean": np.nan,
            "p_pres_abs_early": np.nan,
            "p_pres_abs_gain": np.nan,
            "n_absence_null_replicates": 0,
            "absence_null_curve_mean": [],
            "absence_null_early_mean_direct": [],
            "absence_null_gain_late_mean_direct": [],
        }

    null_mean = [r["curve_mean"] for r in absence_null_rows]
    null_early = [r["early_mean_direct"] for r in absence_null_rows]
    null_gain = [r["gain_late_mean_direct"] for r in absence_null_rows]

    return {
        "p_pres_abs_mean": _one_sided_pres_gt_abs_p(null_mean, presence_s["curve_mean"]),
        "p_pres_abs_early": _one_sided_pres_gt_abs_p(
            null_early, presence_s["early_mean_direct"]
        ),
        "p_pres_abs_gain": _one_sided_pres_gt_abs_p(
            null_gain, presence_s["gain_late_mean_direct"]
        ),
        "n_absence_null_replicates": len(absence_null_rows),
        "absence_null_curve_mean": null_mean,
        "absence_null_early_mean_direct": null_early,
        "absence_null_gain_late_mean_direct": null_gain,
        "absence_null_summary_mean": _null_summary(null_mean),
        "absence_null_summary_early": _null_summary(null_early),
        "absence_null_summary_gain": _null_summary(null_gain),
    }


def load_absence_pres_abs_null_metrics(cache_csv):
    cache_csv = Path(cache_csv)
    if not cache_csv.exists():
        return None
    df = pd.read_csv(cache_csv)
    return df.to_dict("records")


def run_absence_pres_abs_null(
    base_dir,
    n_replicates=ABSENCE_PRES_NULL_REPLICATES_DEFAULT,
    n_sessions=N_SESSIONS_DEFAULT,
    nrand=NRAND_PRES_ABS_NULL,
    rng_seed_start=1000,
    weights_json=None,
    n_jobs=1,
    max_obs_per_trial=400,
    blocks_per_session=BLOCKS_PER_SESSION_DEFAULT,
    contrast_matched_null=True,
    force=False,
    cache_name="absence_replicates",
):
    """
    Simulate absence n_replicates times (different trial seeds) for pres vs abs null.

    Caches scalar S metrics to pres_abs_null/{cache_name}.csv. Label-shuffle null
    settings do not affect the real curves used here; nrand defaults to 1 for speed.
    """
    base_dir = Path(base_dir)
    cache_dir = base_dir / "pres_abs_null"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_csv = cache_dir / f"{cache_name}.csv"
    if cache_csv.exists() and not force:
        rows = load_absence_pres_abs_null_metrics(cache_csv)
        print(f"Loaded {len(rows)} absence pres-abs null replicates from {cache_csv}")
        return rows

    weights_json = resolve_weights_json(weights_json)
    mp, _ = load_fitted_model(g_s=0.0, d_s=0.0, json_path=weights_json)
    splits = s_prior_splits()
    rows = []
    print(
        f"\n=== Absence pres-abs null ({n_replicates} replicates, "
        f"n_sessions={n_sessions}, seed_start={rng_seed_start}) ==="
    )
    for rep in range(n_replicates):
        rep_seed = rng_seed_start + rep
        t0 = time.perf_counter()
        session_dfs, steps_before_obs, _ = simulate_condition_sessions(
            mp,
            n_sessions,
            blocks_per_session,
            max_obs_per_trial,
            rep_seed,
        )
        with tempfile.TemporaryDirectory() as td:
            res_dir = Path(td)
            rng = np.random.RandomState(rep_seed + 999_999)
            build_res_from_trajectories(
                session_dfs,
                splits,
                steps_before_obs,
                nrand,
                rng,
                res_dir,
                populations=("S",),
                n_jobs=n_jobs,
                contrast_matched_null=contrast_matched_null,
            )
            stack_combined_timeframes(res_dir, [S_PRIOR_TIMEFRAME])
            s_prior = s_only_prior_test(res_dir)
        if s_prior is None:
            print(f"  replicate {rep + 1}/{n_replicates}: no S prior metrics (skipped)")
            continue
        rows.append(
            {
                "replicate": rep,
                "rng_seed": rep_seed,
                "curve_mean": s_prior["curve_mean"],
                "early_mean_direct": s_prior["early_mean_direct"],
                "gain_late_mean_direct": s_prior["gain_late_mean_direct"],
            }
        )
        if (rep + 1) % 10 == 0 or rep + 1 == n_replicates:
            print(
                f"  replicate {rep + 1}/{n_replicates} (seed={rep_seed}) "
                f"— {time.perf_counter() - t0:.1f}s"
            )
    pd.DataFrame(rows).to_csv(cache_csv, index=False)
    print(f"Saved absence pres-abs null metrics to {cache_csv}")
    return rows


def pres_abs_p_values_direct(abs_splits, pres_splits):
    """
    Presence vs absence p-values from paired per-split differences.

    One-sided sign test across act_block_duringstim splits (no label-shuffle nulls).
    """
    splits = sorted(set(abs_splits) & set(pres_splits))
    if not splits:
        return {
            "p_pres_abs_mean": np.nan,
            "p_pres_abs_early": np.nan,
            "p_pres_abs_gain": np.nan,
            "n_splits": 0,
        }

    def diffs(key):
        return [pres_splits[s][key] - abs_splits[s][key] for s in splits]

    return {
        "p_pres_abs_mean": _one_sided_sign_p(diffs("curve_mean")),
        "p_pres_abs_early": _one_sided_sign_p(diffs("early_mean_direct")),
        "p_pres_abs_gain": _one_sided_sign_p(diffs("gain_late_mean_direct")),
        "n_splits": len(splits),
        "split_diffs_mean": diffs("curve_mean"),
        "split_diffs_early": diffs("early_mean_direct"),
        "split_diffs_gain": diffs("gain_late_mean_direct"),
    }


def population_prior_test(pth_res, population, alpha=0.01, timeframe="act_block_duringstim"):
    """
    Block prior test for one model population using act_block_duringstim curves.

    No FDR across populations.
    """
    combined, regde_name, splits = _combined_names(timeframe)
    regde_path = pth_res / f"{regde_name}.npy"
    d = _load_combined_results(pth_res, combined)
    if not regde_path.exists() or d is None:
        return None

    regde = np.load(regde_path, allow_pickle=True).item()
    if population not in regde or population not in d:
        return None

    real, nulls = _unpack_regde_curves(regde[population])
    real_mean = float(np.mean(real))
    null_means = np.mean(nulls, axis=1)
    p_mean = float(np.mean(null_means >= real_mean))
    amp_real = float(np.max(real) - np.min(real))
    amp_null = np.array([np.max(n) - np.min(n) for n in nulls])
    p_amp = float(np.mean(amp_null >= amp_real))

    ref_split = splits[0]
    t_axis = time_axis_for_split(ref_split, len(real))
    offset = _prior_offset_stats(real, nulls)
    gain = _prior_gain_stats(real, nulls, alpha=alpha)
    early_mean_direct = _direct_early_mean(real)
    gain_late_mean_direct = _direct_gain_late_mean(real)
    null_curve_means = null_means.astype(float).tolist()
    null_curve_amps = amp_null.astype(float).tolist()

    return {
        "population": population,
        "timeframe": timeframe,
        "p_mean": p_mean,
        "p_amp": p_amp,
        "p_offset": offset["p_offset"],
        "p_gain": gain["p_gain"],
        "offset_effect": offset["offset_effect"],
        "gain_effect": gain["gain_effect"],
        "offset_mean": offset["offset_mean"],
        "early_mean_direct": early_mean_direct,
        "gain_late_mean": gain["gain_late_mean"],
        "gain_late_mean_direct": gain_late_mean_direct,
        "amp_euc": float(d[population]["amp_euc"]),
        "curve_mean": real_mean,
        "curve_amp": amp_real,
        "null_curve_mean_median": float(np.median(null_means)),
        "null_curve_mean_min": float(np.min(null_means)),
        "null_curve_mean_max": float(np.max(null_means)),
        "significant_p_mean": p_mean < alpha,
        "significant_p_amp": p_amp < alpha,
        "significant_p_offset": offset["p_offset"] < alpha,
        "significant_p_gain": gain["p_gain"] < alpha if not np.isnan(gain["p_gain"]) else False,
        "curve_real": real,
        "null_curves": nulls,
        "null_curve_means": null_curve_means,
        "null_curve_amps": null_curve_amps,
        "curve_null_mean": np.mean(nulls, axis=0),
        "t_axis": t_axis,
        "ref_split": ref_split,
    }


def s_only_prior_test(pth_res, alpha=0.01, timeframe="act_block_duringstim"):
    """S-only block prior test (wrapper around population_prior_test)."""
    return population_prior_test(pth_res, "S", alpha=alpha, timeframe=timeframe)


def _contrast_distance_curve(b, labels, contrasts, contrast):
    """
    Prior-distance curve for one contrast bin within a split.

    labels: bool (n_trials), True = high-prior group.
    Returns (curve / B_SIZE, n_high, n_low) or (None, n_high, n_low).
    """
    contrasts = np.asarray(contrasts, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    idx = np.where(contrasts == contrast)[0]
    if idx.size == 0:
        return None, 0, 0
    high = idx[labels[idx]]
    low = idx[~labels[idx]]
    if len(high) < 1 or len(low) < 1:
        return None, len(high), len(low)
    m0 = b[high].mean(axis=0)
    m1 = b[low].mean(axis=0)
    curve = np.sum((m0 - m1) ** 2, axis=0) / B_SIZE
    return curve, len(high), len(low)


def split_contrast_shuffle_diagnostics(
    b,
    n0,
    contrasts,
    nrand,
    rng,
    contrast_matched_null=True,
):
    """Per-contrast true vs contrast-matched shuffle distance means for one split."""
    contrasts = np.asarray(contrasts, dtype=float)
    ntr = b.shape[0]
    ys_true = np.zeros(ntr, dtype=bool)
    ys_true[:n0] = True
    rows = []
    for contrast in sorted(np.unique(contrasts)):
        if np.isnan(contrast):
            continue
        true_curve, n_high, n_low = _contrast_distance_curve(b, ys_true, contrasts, contrast)
        if true_curve is None:
            rows.append(
                {
                    "contrast": float(contrast),
                    "n_high": n_high,
                    "n_low": n_low,
                    "true_curve_mean": np.nan,
                    "shuffle_curve_mean_median": np.nan,
                    "shuffle_curve_mean_min": np.nan,
                    "shuffle_curve_mean_max": np.nan,
                    "shuffle_curve_mean_mean": np.nan,
                    "p_shuffle_gte_true": np.nan,
                    "n_shuffles": 0,
                }
            )
            continue
        true_mean = float(np.mean(true_curve))
        null_means = []
        if contrast_matched_null:
            from null_shuffle_worker import contrast_matched_shuffle_labels

            for _ in range(nrand):
                pseudo = contrast_matched_shuffle_labels(contrasts, n0, rng)
                null_curve, _, _ = _contrast_distance_curve(b, pseudo, contrasts, contrast)
                if null_curve is not None:
                    null_means.append(float(np.mean(null_curve)))
        null_arr = np.asarray(null_means, dtype=float)
        rows.append(
            {
                "contrast": float(contrast),
                "n_high": n_high,
                "n_low": n_low,
                "true_curve_mean": true_mean,
                "shuffle_curve_mean_median": float(np.median(null_arr)) if null_arr.size else np.nan,
                "shuffle_curve_mean_min": float(np.min(null_arr)) if null_arr.size else np.nan,
                "shuffle_curve_mean_max": float(np.max(null_arr)) if null_arr.size else np.nan,
                "shuffle_curve_mean_mean": float(np.mean(null_arr)) if null_arr.size else np.nan,
                "p_shuffle_gte_true": float(np.mean(null_arr >= true_mean)) if null_arr.size else np.nan,
                "n_shuffles": int(null_arr.size),
            }
        )
    return rows


def write_split_contrast_shuffle_diagnostics(
    session_dfs,
    steps_before_obs,
    splits,
    nrand,
    rng_seed,
    fig_dir,
    population="S",
    contrast_matched_null=True,
    prior_column=None,
    out_csv=None,
):
    """CSV of per-split, per-contrast true vs shuffle distance means."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    prior_column = prior_column or PRIOR_COLUMN
    all_df = pd.concat(session_dfs, ignore_index=True)
    rng = np.random.RandomState(rng_seed + 12345)
    rows = []
    with use_prior_column(prior_column):
        for split in splits:
            built = build_population_b_for_split(all_df, split, population, steps_before_obs)
            if built is None:
                continue
            b, n0, contrasts = built
            for entry in split_contrast_shuffle_diagnostics(
                b, n0, contrasts, nrand, rng, contrast_matched_null=contrast_matched_null
            ):
                rows.append({"split": split, "population": population, **entry})
    out_df = pd.DataFrame(rows)
    if out_csv is not False:
        path = Path(out_csv) if out_csv is not None else fig_dir / "s_prior_split_contrast_shuffle.csv"
        out_df.to_csv(path, index=False)
        print(f"Saved per-split/contrast shuffle diagnostics to {path}")
    return out_df


def write_multi_population_split_contrast_diagnostics(
    session_dfs,
    steps_before_obs,
    splits,
    nrand,
    rng_seed,
    fig_dir,
    populations,
    out_csv,
    contrast_matched_null=True,
    prior_column=None,
):
    """Per split × contrast × population contrast-matched shuffle table."""
    parts = []
    for i, pop in enumerate(populations):
        parts.append(
            write_split_contrast_shuffle_diagnostics(
                session_dfs,
                steps_before_obs,
                splits,
                nrand,
                rng_seed + 12345 + i * 9973,
                fig_dir,
                population=pop,
                contrast_matched_null=contrast_matched_null,
                prior_column=prior_column,
                out_csv=False,
            )
        )
    out_df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    out_path = Path(fig_dir) / out_csv
    out_df.to_csv(out_path, index=False)
    print(f"Saved per-split/contrast diagnostics ({len(populations)} pops) to {out_path}")
    return out_df


def write_s_prior_shuffle_diagnostics(s_prior, fig_dir):
    """Per-shuffle curve_mean/amp CSV plus true-vs-null summary for inspection."""
    if s_prior is None:
        return None

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    null_means = s_prior.get("null_curve_means")
    null_amps = s_prior.get("null_curve_amps")
    if null_means is None:
        nulls = np.atleast_2d(np.asarray(s_prior["null_curves"], dtype=float))
        null_means = np.mean(nulls, axis=1).astype(float).tolist()
        null_amps = [float(np.max(n) - np.min(n)) for n in nulls]
    true_mean = float(s_prior["curve_mean"])
    true_amp = float(s_prior["curve_amp"])

    rows = []
    for i, (m, a) in enumerate(zip(null_means, null_amps)):
        rows.append(
            {
                "shuffle_idx": i,
                "curve_mean": float(m),
                "curve_amp": float(a),
                "mean_gt_true": float(m) > true_mean,
                "amp_gt_true": float(a) > true_amp,
            }
        )
    null_df = pd.DataFrame(rows)
    null_csv = fig_dir / "s_prior_shuffle_nulls.csv"
    null_df.to_csv(null_csv, index=False)

    null_arr = null_df["curve_mean"].to_numpy(dtype=float)
    summary = {
        "true_curve_mean": true_mean,
        "true_curve_amp": true_amp,
        "n_null_shuffles": len(null_arr),
        "null_mean_min": float(np.min(null_arr)),
        "null_mean_median": float(np.median(null_arr)),
        "null_mean_max": float(np.max(null_arr)),
        "null_mean_q05": float(np.percentile(null_arr, 5)),
        "null_mean_q95": float(np.percentile(null_arr, 95)),
        "n_null_mean_gt_true": int(np.sum(null_arr > true_mean)),
        "n_null_mean_gte_true": int(np.sum(null_arr >= true_mean)),
        "p_mean": float(s_prior["p_mean"]),
        "p_amp": float(s_prior["p_amp"]),
        "true_minus_null_median": true_mean - float(np.median(null_arr)),
    }
    summary_csv = fig_dir / "s_prior_shuffle_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    print(f"Saved shuffle-null diagnostics to {null_csv} and {summary_csv}")
    print(json.dumps(summary, indent=2))
    return summary


def s_prior_presence_vs_absence_direct(abs_s, pres_s):
    """
    Direct presence vs absence comparison on real S prior curves.

    All metrics are scalar differences on the raw combined regde real curves.
    Label-shuffle nulls are not used (those test within-condition grouping only).
    """
    if abs_s is None or pres_s is None:
        return None

    diff_mean = float(pres_s["curve_mean"] - abs_s["curve_mean"])
    diff_amp = float(pres_s["curve_amp"] - abs_s["curve_amp"])
    diff_amp_euc = float(pres_s["amp_euc"] - abs_s["amp_euc"])
    early_abs = abs_s.get("early_mean_direct", _direct_early_mean(abs_s["curve_real"]))
    early_pres = pres_s.get("early_mean_direct", _direct_early_mean(pres_s["curve_real"]))
    diff_offset = float(early_pres - early_abs)
    gain_abs = abs_s.get(
        "gain_late_mean_direct", _direct_gain_late_mean(abs_s["curve_real"])
    )
    gain_pres = pres_s.get(
        "gain_late_mean_direct", _direct_gain_late_mean(pres_s["curve_real"])
    )
    diff_gain = float(gain_pres - gain_abs) if not (
        np.isnan(gain_pres) or np.isnan(gain_abs)
    ) else np.nan

    return {
        "diff_mean": diff_mean,
        "diff_amp": diff_amp,
        "diff_amp_euc": diff_amp_euc,
        "diff_offset": diff_offset,
        "diff_gain": diff_gain,
        "presence_gt_absence_mean": diff_mean > 0,
        "presence_gt_absence_amp": diff_amp > 0,
        "presence_gt_absence_amp_euc": diff_amp_euc > 0,
        "presence_gt_absence_offset": diff_offset > 0,
        "presence_gt_absence_gain": (
            diff_gain > 0 if not np.isnan(diff_gain) else False
        ),
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


def _draw_shuffle_control_panel(
    ax,
    s_prior,
    condition,
    n_sample=SHUFFLE_PLOT_N_SAMPLE,
    rng_seed=0,
    show_null_mean=True,
    true_color="#C44E52",
    null_color="0.78",
    prior_label="",
    ylabel=None,
):
    """Draw one shuffle-control panel on ax."""
    pop = s_prior.get("population", "S")
    ylabel = ylabel or f"{pop} prior distance (raw)"
    real = np.asarray(s_prior["curve_real"], dtype=float)
    nulls = np.atleast_2d(np.asarray(s_prior["null_curves"], dtype=float))
    n_plot = min(n_sample, nulls.shape[0])
    rng = np.random.RandomState(rng_seed)
    idx = rng.choice(nulls.shape[0], size=n_plot, replace=False)

    t = s_prior["t_axis"]
    for j in idx:
        ax.plot(t, nulls[j], color=null_color, lw=0.7, alpha=0.28, zorder=1)
    if show_null_mean:
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
    ax.plot(t, real, color=true_color, lw=2.5, label="true labels", zorder=10)
    ref_split = s_prior.get("ref_split", S_PRIOR_TIMEFRAME)
    _shade_offset_window(ax, t)
    _mark_align_event(ax, ref_split)
    ax.set_xlabel(_time_xlabel(ref_split))
    ax.set_ylabel(ylabel)
    label_txt = _prior_label_title(prior_label)
    title_suffix = f" [{label_txt}]" if label_txt else ""
    ax.set_title(
        f"{condition}{title_suffix}\n"
        f"true curve vs {n_plot} sampled nulls (of {nulls.shape[0]} shuffles)"
    )
    p_gain = s_prior.get("p_gain", np.nan)
    ax.text(
        0.98,
        0.98,
        f"p_mean={s_prior['p_mean']:.4f}\n"
        f"p_offset={s_prior['p_offset']:.4f}\n"
        f"p_gain={p_gain:.4f}\n"
        "(vs contrast-matched null)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )
    ax.legend(fontsize=7, loc="upper left")


def plot_s_shuffle_control(
    condition, s_prior, fig_dir, n_sample=SHUFFLE_PLOT_N_SAMPLE, rng_seed=0, prior_label="",
    filename="s_shuffle_control.png",
):
    """True S prior-distance curve vs sampled label-shuffle null curves."""
    if s_prior is None:
        return
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    _draw_shuffle_control_panel(ax, s_prior, condition, n_sample, rng_seed, prior_label=prior_label)
    fig.tight_layout()
    out = fig_dir / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved shuffle control figure to {out}")


def _plot_shuffle_trajectories(
    ax,
    s_prior,
    true_color,
    null_color,
    n_sample=SHUFFLE_PLOT_N_SAMPLE,
    rng_seed=0,
    true_label=None,
):
    """Plot time-resolved prior-distance trajectories: real + sampled null shuffles."""
    real = np.asarray(s_prior["curve_real"], dtype=float)
    nulls = np.atleast_2d(np.asarray(s_prior["null_curves"], dtype=float))
    t = s_prior["t_axis"]
    n_plot = min(n_sample, nulls.shape[0])
    rng = np.random.RandomState(rng_seed)
    idx = rng.choice(nulls.shape[0], size=n_plot, replace=False)
    for j in idx:
        ax.plot(t, nulls[j], color=null_color, lw=0.7, alpha=0.22, zorder=1)
    ax.plot(t, real, color=true_color, lw=2.5, label=true_label, zorder=10)


def plot_combined_shuffle_controls(base_dir, abs_s, pres_s, rng_seed=0, out_path=None, prior_label=""):
    """Side-by-side absence/presence panels: true trajectories + sampled null shuffles."""
    if abs_s is None and pres_s is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    label_txt = _prior_label_title(prior_label)
    panels = [
        ("absence", abs_s, rng_seed, "#4C72B0"),
        ("presence", pres_s, rng_seed + 1, "#DD8452"),
    ]
    for ax, (cond, s_prior, seed, color) in zip(axes, panels):
        if s_prior is None:
            ax.set_title(f"{cond} (no data)")
            ax.axis("off")
            continue
        t = s_prior["t_axis"]
        _plot_shuffle_trajectories(
            ax,
            s_prior,
            true_color=color,
            null_color=color,
            rng_seed=seed,
            true_label="true labels",
        )
        ref_split = s_prior.get("ref_split", S_PRIOR_TIMEFRAME)
        _shade_offset_window(ax, t, alpha=0.1)
        _mark_align_event(ax, ref_split)
        n_nulls = np.atleast_2d(s_prior["null_curves"]).shape[0]
        n_plot = min(SHUFFLE_PLOT_N_SAMPLE, n_nulls)
        p_gain = s_prior.get("p_gain", np.nan)
        ax.set_title(
            f"{cond}: true curve vs {n_plot} sampled nulls (of {n_nulls})"
        )
        ax.set_xlabel(_time_xlabel(ref_split))
        ax.set_ylabel("S prior distance (raw)")
        p_mean = s_prior.get("p_mean", np.nan)
        p_offset = s_prior.get("p_offset", np.nan)
        ax.text(
            0.98,
            0.98,
            f"p_mean={p_mean:.4f}\n"
            f"p_offset={p_offset:.4f}\n"
            f"p_gain={p_gain:.4f}\n"
            "(vs contrast-matched null)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            family="monospace",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
        )
        ax.legend(fontsize=7, loc="upper left")

    if label_txt:
        fig.suptitle(
            f"{label_txt}\n"
            "Within-condition label-shuffle controls (p_mean / p_offset / p_gain per panel)",
            y=1.04,
            fontsize=10,
        )
    else:
        fig.suptitle(
            "Within-condition label-shuffle controls (p_mean / p_offset / p_gain per panel)",
            y=1.02,
        )
    fig.tight_layout()
    out = out_path or (base_dir / "s_shuffle_control_combined.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined shuffle control figure to {out}")


def plot_s_prior_curve(condition, s_prior, fig_dir, prior_label=""):
    """S-only act_block_duringstim prior curve (raw, baseline not removed)."""
    if s_prior is None:
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))

    t = s_prior["t_axis"]
    real = s_prior["curve_real"]
    null_mean = s_prior["curve_null_mean"]
    ax.plot(t, real, "C0", lw=2, label="S real")
    ax.plot(t, null_mean, "C0", ls="--", alpha=0.6, label="null mean")
    ref_split = s_prior.get("ref_split", S_PRIOR_TIMEFRAME)
    _shade_offset_window(ax, t, label="offset window (bins 0–4)")
    _mark_align_event(ax, ref_split)
    ax.set_xlabel(_time_xlabel(ref_split))
    ax.set_ylabel("Prior distance (raw)")
    label_txt = _prior_label_title(prior_label)
    title_suffix = f" [{label_txt}]" if label_txt else ""
    ax.set_title(
        f"S prior curve ({condition}){title_suffix} — "
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
        ref_split = s_prior_splits()[0] if s_prior is None else s_prior.get("ref_split", s_prior_splits()[0])
        t_stim = time_axis_for_split(ref_split, len(next(iter(regde_stim.values()))[0]))
        fig, ax = plt.subplots(figsize=(7, 4))
        for gtype, color in zip(["S", "I", "M"], ["C0", "C1", "C2"]):
            regs = [r for r, t in POPULATION_TYPE.items() if t == gtype and r in regde_stim]
            if not regs:
                continue
            curves = [regde_stim[r][0] for r in regs if len(regde_stim[r])]
            if curves:
                mean_c = np.mean(curves, axis=0)
                ax.plot(t_stim, mean_c, label=f"{gtype} (n={len(curves)})", color=color)
        _mark_align_event(ax, ref_split)
        ax.set_xlabel(_time_xlabel(ref_split))
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


def plot_s_prior_figures(condition, s_prior, fig_dir, prior_label=""):
    """S-prior-only figure outputs (curve + shuffle control)."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    if s_prior is not None:
        plot_s_prior_curve(condition, s_prior, fig_dir, prior_label=prior_label)
        plot_s_shuffle_control(condition, s_prior, fig_dir, prior_label=prior_label)
        write_s_prior_shuffle_diagnostics(s_prior, fig_dir)
        pd.DataFrame(
            [{k: v for k, v in s_prior.items() if not isinstance(v, (np.ndarray, list))}]
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
    contrast_matched_null=True,
    zero_im_prior_mod=False,
):
    im_label = " (g_i=d_i=g_m=d_m=0)" if zero_im_prior_mod else ""
    print(f"\n=== Condition: {condition_name} (g_s={g_s}, d_s={d_s}){im_label} ===")
    null_scheme = (
        NULL_SCHEME_CONTRAST_MATCHED if contrast_matched_null else NULL_SCHEME_LABEL_SHUFFLE
    )
    print(f"  prior column: {PRIOR_COLUMN} | null: {null_scheme}")
    log_canonical_analysis_banner()
    t0 = time.time()
    mp, meta = load_fitted_model(
        g_s=g_s,
        d_s=d_s,
        zero_im_prior_mod=zero_im_prior_mod,
        json_path=weights_json,
    )

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
        contrast_matched_null=contrast_matched_null,
    )
    print(f"  wrote {n_splits} split files in {time.perf_counter() - t_dist:.1f}s")
    stack_combined_timeframes(res_dir, timeframes)

    s_prior = s_only_prior_test(res_dir)

    if s_prior_only:
        plot_s_prior_figures(condition_name, s_prior, fig_dir)
        write_split_contrast_shuffle_diagnostics(
            session_dfs,
            steps_before_obs,
            splits,
            nrand,
            rng_seed,
            fig_dir,
            population="S",
            contrast_matched_null=contrast_matched_null,
            prior_column=PRIOR_COLUMN,
        )
        n_trials_total = int(sum(m["n_trials"] for m in session_meta))
        summary = {
            "condition": condition_name,
            "g_s": g_s,
            "d_s": d_s,
            "rng_seed": rng_seed,
            "mode": "s_prior_only",
            "prior_conditioning": {
                "prior_column": PRIOR_COLUMN,
            "prior_window_ms": [
                -mf.ITI_START_BEFORE_MS,
                -mf.ITI_END_BEFORE_MS,
            ],
            "prior_window_note": "pre-stimulus intertrial P (not trial-averaged)",
                "timeframe": S_PRIOR_TIMEFRAME,
                "splits": splits,
                "populations": ["S"],
                "null_scheme": f"{null_scheme}, nrand={nrand}, n_jobs={n_jobs}",
            },
            "sessions": session_meta,
            "n_trials_total": n_trials_total,
            "g_i_fitted": float(meta["g"]["g_i"]),
            "d_i_fitted": float(meta["d"]["d_i"]),
            "g_i_used": float(mp["g_i"]),
            "g_m_used": float(mp["g_m"]),
            "d_i_used": float(mp["d_i"]),
            "d_m_used": float(mp["d_m"]),
            "zero_im_prior_mod": bool(zero_im_prior_mod),
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
        "rng_seed": rng_seed,
        "prior_conditioning": {
            "prior_column": PRIOR_COLUMN,
            "prior_window_ms": [
                -mf.ITI_START_BEFORE_MS,
                -mf.ITI_END_BEFORE_MS,
            ],
            "prior_window_note": "pre-stimulus intertrial P (not trial-averaged)",
            "description": "binarized model P subjective prior (mean P_L-P_R per trial)",
            "populations": "S, I, M, P share the same prior column per split",
            "null_scheme": f"{null_scheme}, nrand={nrand}",
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


def plot_s_prior_comparison(
    base_dir, abs_s, pres_s, direct_cmp, sensory_prior_recovery, out_path=None, prior_label=""
):
    """Dedicated S prior comparison figure (raw curves, baseline not removed)."""
    if abs_s is None or pres_s is None:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    t = abs_s["t_axis"]

    ref_split = abs_s.get("ref_split", S_PRIOR_TIMEFRAME)
    ax.plot(t, abs_s["curve_real"], color="#4C72B0", lw=2, label="absence")
    ax.plot(t, pres_s["curve_real"], color="#DD8452", lw=2, label="presence")
    _shade_offset_window(ax, t, label="offset window (bins 0–4)")
    _mark_align_event(ax, ref_split)
    ax.set_xlabel(_time_xlabel(ref_split))
    ax.set_ylabel("S prior distance (raw)")
    label_txt = _prior_label_title(prior_label)
    title_suffix = f" — {label_txt}" if label_txt else ""
    ax.set_title(f"S prior curves (raw, baseline retained){title_suffix}")
    ax.legend(fontsize=8)

    if direct_cmp:
        g_s = sensory_prior_recovery.get("g_s_presence", "?")
        d_s = sensory_prior_recovery.get("d_s_presence", "?")
        g_i = sensory_prior_recovery.get("g_i_fitted", "?")
        d_i = sensory_prior_recovery.get("d_i_fitted", "?")
        txt = (
            f"fitted integrator: g_i={g_i}, d_i={d_i} | presence sensory: g_s={g_s}, d_s={d_s}\n"
            f"direct pres-abs: mean={direct_cmp['diff_mean']:+.2f}  "
            f"amp(raw)={direct_cmp['diff_amp']:+.2f}  "
            f"early={direct_cmp['diff_offset']:+.2f}  "
            f"late={direct_cmp.get('diff_gain', np.nan):+.2f}"
        )
        fig.text(0.5, -0.02, txt, ha="center", va="top", fontsize=9, family="monospace")

    fig.tight_layout()
    out = out_path or (base_dir / "s_prior_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved S prior comparison figure to {out}")


def _plot_pres_abs_bar_comparison(
    ax,
    abs_val,
    pres_val,
    ylabel,
    metric_name,
    diff,
    p_value=None,
    absence_null_values=None,
):
    """Grouped absence vs presence bar chart for one scalar metric."""
    x = [0, 1]
    ax.bar(x, [abs_val, pres_val], color=["#4C72B0", "#DD8452"], width=0.65, zorder=2)
    if absence_null_values:
        null_arr = np.asarray(absence_null_values, dtype=float)
        null_arr = null_arr[np.isfinite(null_arr)]
        if null_arr.size:
            jitter = np.linspace(-0.15, 0.15, null_arr.size)
            ax.scatter(
                jitter,
                null_arr,
                alpha=0.35,
                s=14,
                color="#4C72B0",
                edgecolors="none",
                zorder=3,
                label="absence null" if metric_name == "S curve overall mean" else None,
            )
            q05, q95 = np.percentile(null_arr, [5, 95])
            ax.hlines(q05, -0.22, 0.22, colors="#2a4a7a", linewidth=1.4, alpha=0.75, zorder=3)
            ax.hlines(q95, -0.22, 0.22, colors="#2a4a7a", linewidth=1.4, alpha=0.75, zorder=3)
            ax.hlines(
                np.median(null_arr),
                -0.22,
                0.22,
                colors="#2a4a7a",
                linewidth=1.0,
                linestyles="--",
                alpha=0.75,
                zorder=3,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(["absence", "presence"])
    ax.set_ylabel(ylabel)
    diff_txt = f"pres - abs = {diff:+.3g}" if not np.isnan(diff) else "pres - abs = n/a"
    if p_value is not None and not np.isnan(p_value):
        p_txt = f"p(pres>abs) = {p_value:.4f}"
    else:
        p_txt = "p(pres>abs) = n/a"
    ax.set_title(f"{metric_name}\n{diff_txt}  |  {p_txt}")


def plot_comparison_metrics_summary(
    sensory_prior_recovery,
    presence,
    out_path,
):
    """Three-panel direct presence vs absence with absence-replicate null p-values."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    n_null = sensory_prior_recovery.get("n_absence_null_replicates", "?")
    null_mean = sensory_prior_recovery.get("absence_null_curve_mean") or []
    null_early = sensory_prior_recovery.get("absence_null_early_mean_direct") or []
    null_gain = sensory_prior_recovery.get("absence_null_gain_late_mean_direct") or []

    _plot_pres_abs_bar_comparison(
        axes[0],
        sensory_prior_recovery["s_curve_mean_absence"],
        sensory_prior_recovery["s_curve_mean_presence"],
        "mean prior distance (raw curve)",
        "S curve overall mean",
        sensory_prior_recovery["diff_mean"],
        p_value=sensory_prior_recovery.get("p_pres_abs_mean"),
        absence_null_values=null_mean,
    )
    _plot_pres_abs_bar_comparison(
        axes[1],
        sensory_prior_recovery["s_early_mean_absence"],
        sensory_prior_recovery["s_early_mean_presence"],
        f"mean first {PRIOR_OFFSET_BINS} bins (raw)",
        "S early-bin mean",
        sensory_prior_recovery["diff_offset"],
        p_value=sensory_prior_recovery.get("p_pres_abs_early"),
        absence_null_values=null_early,
    )
    _plot_pres_abs_bar_comparison(
        axes[2],
        sensory_prior_recovery["s_gain_late_mean_absence"],
        sensory_prior_recovery["s_gain_late_mean_presence"],
        "mean bins 4+ after own early mean removed",
        "S late gain (direct)",
        sensory_prior_recovery["diff_gain"],
        p_value=sensory_prior_recovery.get("p_pres_abs_gain"),
        absence_null_values=null_gain,
    )

    gs = presence.get("g_s", "?")
    ds = presence.get("d_s", "?")
    ref_seed = sensory_prior_recovery.get("rng_seed_absence", "?")
    fig.suptitle(
        f"Direct presence − absence (absence null: n={n_null} replicates; "
        f"reference absence seed={ref_seed})\n"
        f"absence (g_s=0,d_s=0) vs presence (g_s={gs}, d_s={ds})",
        y=1.08,
        fontsize=11,
    )
    fig.text(
        0.5,
        -0.02,
        "Blue dots: absence-only null (different trial seeds). p(pres>abs): one-sided "
        "P(absence_null >= presence). Reference absence bar: paired seed. "
        "Within-condition shuffle p-values: s_shuffle_control_combined.png.",
        ha="center",
        va="top",
        fontsize=8,
        wrap=True,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison metrics figure to {out_path}")


def compare_conditions(
    base_dir,
    alpha=0.01,
    s_amp_ratio_thresh=1.1,
    abs_res_dir=None,
    pres_res_dir=None,
    out_json=None,
    comparison_metrics_path=None,
    s_prior_comparison_path=None,
    shuffle_combined_path=None,
    prior_label="",
    prior_column=None,
    absence_null_metrics=None,
):
    base_dir = Path(base_dir)
    absence_path = base_dir / "absence" / "summary.json"
    presence_path = base_dir / "presence" / "summary.json"
    absence = json.loads(absence_path.read_text()) if absence_path.exists() else {}
    presence = json.loads(presence_path.read_text()) if presence_path.exists() else {}

    abs_res = Path(abs_res_dir) if abs_res_dir else base_dir / "absence" / "res"
    pres_res = Path(pres_res_dir) if pres_res_dir else base_dir / "presence" / "res"
    abs_s = s_only_prior_test(abs_res, alpha=alpha)
    pres_s = s_only_prior_test(pres_res, alpha=alpha)
    abs_splits = s_prior_metrics_per_split(abs_res, alpha=alpha)
    pres_splits = s_prior_metrics_per_split(pres_res, alpha=alpha)
    pres_abs_sign = pres_abs_p_values_direct(abs_splits, pres_splits)
    if absence_null_metrics is None:
        cache_csv = base_dir / "pres_abs_null" / "absence_replicates.csv"
        absence_null_metrics = load_absence_pres_abs_null_metrics(cache_csv)
    pres_abs_null = pres_abs_p_values_absence_null(pres_s, absence_null_metrics or [])
    use_absence_null = bool(absence_null_metrics)
    pres_abs_p = pres_abs_null if use_absence_null else pres_abs_sign
    direct_cmp = s_prior_presence_vs_absence_direct(abs_s, pres_s)
    s_amp_ratio = (
        pres_s["amp_euc"] / (abs_s["amp_euc"] + 1e-12) if abs_s and pres_s else np.nan
    )
    s_curve_amp_ratio = (
        pres_s["curve_amp"] / (abs_s["curve_amp"] + 1e-12) if abs_s and pres_s else np.nan
    )
    g_i_fit = presence.get("g_i_fitted") or absence.get("g_i_fitted")
    d_i_fit = presence.get("d_i_fitted") or absence.get("d_i_fitted")
    if g_i_fit is None or d_i_fit is None:
        g_i_fit, d_i_fit = fitted_integrator_scales()
    sensory_prior_recovery = {
        "prior_label": prior_label or None,
        "prior_column": prior_column,
        "comparison_method": (
            "direct_pres_abs_absence_replicate_null"
            if use_absence_null
            else "direct_pres_abs_split_sign_test"
        ),
        "pres_abs_p_method": (
            "one-sided P(absence_replicate >= presence) on combined S metrics "
            f"(curve_mean, early_mean_direct, gain_late_mean_direct); "
            f"n_absence_null={pres_abs_null.get('n_absence_null_replicates', 0)}"
            if use_absence_null
            else (
                "one-sided sign test on per-split paired differences "
                "(curve_mean, early_mean_direct, gain_late_mean_direct); no label shuffle"
            )
        ),
        "pres_abs_p_method_sign_test": (
            "one-sided sign test on per-split paired differences "
            "(curve_mean, early_mean_direct, gain_late_mean_direct); no label shuffle"
        ),
        "within_condition_p_method": (
            NULL_SCHEME_CONTRAST_MATCHED
            if prior_label == "contrast_null"
            else NULL_SCHEME_LABEL_SHUFFLE
            if prior_label == "label_null"
            else (
                "contrast-matched label-shuffle nulls per condition: p_mean, p_offset, p_gain "
                "(preserves per-contrast trial counts in high/low groups)"
            )
        ),
        "prior_window_ms": [-mf.ITI_START_BEFORE_MS, -mf.ITI_END_BEFORE_MS],
        "prior_window_note": "pre-stimulus intertrial P (not trial-averaged)",
        "g_i_fitted": g_i_fit,
        "d_i_fitted": d_i_fit,
        "g_s_presence": presence.get("g_s"),
        "d_s_presence": presence.get("d_s"),
        "paired_seed": presence.get("rng_seed") == absence.get("rng_seed"),
        "rng_seed_absence": absence.get("rng_seed"),
        "rng_seed_presence": presence.get("rng_seed"),
        "n_splits": pres_abs_sign.get("n_splits", 0),
        "n_absence_null_replicates": pres_abs_null.get("n_absence_null_replicates", 0),
        "p_pres_abs_mean": pres_abs_p.get("p_pres_abs_mean", np.nan),
        "p_pres_abs_early": pres_abs_p.get("p_pres_abs_early", np.nan),
        "p_pres_abs_gain": pres_abs_p.get("p_pres_abs_gain", np.nan),
        "p_pres_abs_mean_sign_test": pres_abs_sign.get("p_pres_abs_mean", np.nan),
        "p_pres_abs_early_sign_test": pres_abs_sign.get("p_pres_abs_early", np.nan),
        "p_pres_abs_gain_sign_test": pres_abs_sign.get("p_pres_abs_gain", np.nan),
        "absence_null_summary_mean": pres_abs_null.get("absence_null_summary_mean"),
        "absence_null_summary_early": pres_abs_null.get("absence_null_summary_early"),
        "absence_null_summary_gain": pres_abs_null.get("absence_null_summary_gain"),
        "absence_null_cache": str(base_dir / "pres_abs_null" / "absence_replicates.csv"),
        "s_early_mean_absence": abs_s["early_mean_direct"] if abs_s else np.nan,
        "s_early_mean_presence": pres_s["early_mean_direct"] if pres_s else np.nan,
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
        "s_curve_amp_ratio": float(s_curve_amp_ratio),
        "amp_euc_note": (
            "amp_euc = max(curve - min(curve)) on combined split sums; "
            "min-subtracted baseline — can saturate vs raw curve_amp"
        ),
        "s_p_mean_absence": abs_s["p_mean"] if abs_s else np.nan,
        "s_p_mean_presence": pres_s["p_mean"] if pres_s else np.nan,
        "s_p_gain_absence": abs_s["p_gain"] if abs_s else np.nan,
        "s_p_gain_presence": pres_s["p_gain"] if pres_s else np.nan,
        "s_gain_late_mean_absence": abs_s["gain_late_mean_direct"] if abs_s else np.nan,
        "s_gain_late_mean_presence": pres_s["gain_late_mean_direct"] if pres_s else np.nan,
        "s_gain_late_mean_shuffle_absence": abs_s["gain_late_mean"] if abs_s else np.nan,
        "s_gain_late_mean_shuffle_presence": pres_s["gain_late_mean"] if pres_s else np.nan,
        "diff_mean": direct_cmp["diff_mean"] if direct_cmp else np.nan,
        "diff_amp": direct_cmp["diff_amp"] if direct_cmp else np.nan,
        "diff_amp_euc": direct_cmp["diff_amp_euc"] if direct_cmp else np.nan,
        "diff_offset": direct_cmp["diff_offset"] if direct_cmp else np.nan,
        "diff_gain": direct_cmp["diff_gain"] if direct_cmp else np.nan,
        "split_diffs_mean": pres_abs_sign.get("split_diffs_mean", []),
        "split_diffs_early": pres_abs_sign.get("split_diffs_early", []),
        "split_diffs_gain": pres_abs_sign.get("split_diffs_gain", []),
    }
    if use_absence_null:
        sensory_prior_recovery["absence_null_curve_mean"] = pres_abs_null.get(
            "absence_null_curve_mean", []
        )
        sensory_prior_recovery["absence_null_early_mean_direct"] = pres_abs_null.get(
            "absence_null_early_mean_direct", []
        )
        sensory_prior_recovery["absence_null_gain_late_mean_direct"] = pres_abs_null.get(
            "absence_null_gain_late_mean_direct", []
        )
    sensory_prior_recovery.update(
        {
            "s_significant_absence": bool(abs_s["significant_p_mean"]) if abs_s else False,
            "s_significant_presence": bool(pres_s["significant_p_mean"]) if pres_s else False,
            "recovered_sensory_gain": bool(
                direct_cmp and direct_cmp.get("presence_gt_absence_gain", False)
            ),
            "recovered_sensory_offset": bool(
                direct_cmp and direct_cmp.get("presence_gt_absence_offset", False)
            ),
            "recovered_sensory_amp": bool(
                direct_cmp and direct_cmp.get("presence_gt_absence_amp", False)
            ),
        }
    )
    out_json = out_json or (base_dir / "sensory_prior_recovery.json")
    out_json = Path(out_json)
    out_json.write_text(json.dumps(sensory_prior_recovery, indent=2))

    plot_comparison_metrics_summary(
        sensory_prior_recovery,
        presence,
        out_path=comparison_metrics_path or (base_dir / "comparison_summary_metrics.png"),
    )
    plot_s_prior_comparison(
        base_dir,
        abs_s,
        pres_s,
        direct_cmp,
        sensory_prior_recovery,
        out_path=s_prior_comparison_path or (base_dir / "s_prior_comparison.png"),
        prior_label=prior_label,
    )
    plot_combined_shuffle_controls(
        base_dir,
        abs_s,
        pres_s,
        out_path=shuffle_combined_path or (base_dir / "s_shuffle_control_combined.png"),
        prior_label=prior_label,
    )
    print(json.dumps(sensory_prior_recovery, indent=2))
    return sensory_prior_recovery


def simulate_condition_sessions(
    mp,
    n_sessions,
    blocks_per_session,
    max_obs_per_trial,
    rng_seed,
    min_trials_per_session=MIN_TRIALS_PER_SESSION_DEFAULT,
    constant_s0=False,
):
    """Run trial simulation only; return pooled session DataFrames."""
    session_dfs = []
    session_meta = []
    steps_before_obs = int(mf.STEPS_BEFORE_OBS_DURATION_MS / DT_MS)
    for sess in range(n_sessions):
        sess_rng = np.random.RandomState(rng_seed + sess)
        if blocks_per_session is None:
            n_blocks, planned_trials = blocks_for_min_trials(min_trials_per_session, sess_rng)
        else:
            n_blocks = blocks_per_session
            planned_trials = None
        results, sbo = simulate_session(
            mp, n_blocks, sess_rng, max_obs_per_trial, constant_s0=constant_s0
        )
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
    return session_dfs, steps_before_obs, session_meta


def write_res_for_prior_column(
    session_dfs,
    steps_before_obs,
    nrand,
    rng_seed,
    res_dir,
    prior_column,
    n_jobs=1,
):
    """Build split distance outputs using a specific prior grouping column."""
    res_dir = Path(res_dir)
    if res_dir.exists():
        shutil.rmtree(res_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    splits = s_prior_splits()
    rng = np.random.RandomState(rng_seed)
    with use_prior_column(prior_column):
        build_res_from_trajectories(
            session_dfs,
            splits,
            steps_before_obs,
            nrand,
            rng,
            res_dir,
            populations=("S",),
            n_jobs=n_jobs,
            contrast_matched_null=True,
        )
        stack_combined_timeframes(res_dir, [S_PRIOR_TIMEFRAME])
    return res_dir


def _load_run_config(base_dir):
    """Read simulation config from existing condition summaries."""
    base_dir = Path(base_dir)
    absence = json.loads((base_dir / "absence" / "summary.json").read_text())
    presence = json.loads((base_dir / "presence" / "summary.json").read_text())
    return {
        "rng_seed": absence.get("rng_seed", 42),
        "n_sessions": absence.get("n_sessions", N_SESSIONS_DEFAULT),
        "nrand": _parse_nrand_from_summary({"nrand": absence.get("nrand"), "absence": absence}),
        "blocks_per_session": BLOCKS_PER_SESSION_DEFAULT,
        "g_s_absence": absence.get("g_s", 0.0),
        "d_s_absence": absence.get("d_s", 0.0),
        "g_s_presence": presence.get("g_s"),  # loaded from saved summary
        "d_s_presence": presence.get("d_s"),
    }


def _parse_nrand_from_summary(cfg):
    nrand = cfg.get("nrand")
    if isinstance(nrand, int):
        return nrand
    text = cfg.get("absence", {}).get("prior_conditioning", {}).get("null_scheme", "")
    if "nrand=" in text:
        try:
            return int(text.split("nrand=")[1].split(",")[0].split()[0])
        except ValueError:
            pass
    return NRAND_DEFAULT


NULL_VARIANTS = {
    "contrast_null": True,
    "label_null": False,
}


def _write_res_and_figures(
    condition_name,
    session_dfs,
    steps_before_obs,
    nrand,
    rng_seed,
    base_dir,
    variant_suffix,
    contrast_matched_null,
    n_jobs,
    prior_label="",
):
    """Build res + S-prior figures for one null variant."""
    base_dir = Path(base_dir)
    res_dir = base_dir / condition_name / f"res_{variant_suffix}"
    if res_dir.exists():
        shutil.rmtree(res_dir)
    rng = np.random.RandomState(rng_seed)
    build_res_from_trajectories(
        session_dfs,
        s_prior_splits(),
        steps_before_obs,
        nrand,
        rng,
        res_dir,
        populations=("S",),
        n_jobs=n_jobs,
        contrast_matched_null=contrast_matched_null,
    )
    stack_combined_timeframes(res_dir, [S_PRIOR_TIMEFRAME])
    fig_dir = base_dir / condition_name / f"figs_{variant_suffix}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    s_prior = s_only_prior_test(res_dir)
    plot_s_prior_figures(condition_name, s_prior, fig_dir, prior_label=prior_label)
    if s_prior:
        pd.DataFrame(
            [{k: v for k, v in s_prior.items() if not isinstance(v, np.ndarray)}]
        ).to_csv(fig_dir / "s_prior_stats.csv", index=False)
    return res_dir, s_prior


def run_null_scheme_comparison(
    base_dir,
    weights_json=None,
    n_jobs=None,
    max_obs_per_trial=400,
    n_sessions=N_SESSIONS_DEFAULT,
    nrand=NRAND_DEFAULT,
    rng_seed=42,
    blocks_per_session=BLOCKS_PER_SESSION_DEFAULT,
    n_absence_pres_null_replicates=ABSENCE_PRES_NULL_REPLICATES_DEFAULT,
    skip_absence_pres_null=False,
    force_absence_pres_null=False,
):
    """
    One simulation per condition; plot S-prior results under contrast-matched vs
    unrestricted label-shuffle nulls (pre-stim ITI subjective prior grouping).
    """
    base_dir = Path(base_dir)
    n_jobs = n_jobs if n_jobs is not None else _default_n_jobs()
    weights_json = resolve_weights_json(weights_json)

    print(
        f"\n=== Null scheme comparison (ITI P prior, n_sessions={n_sessions}, "
        f"nrand={nrand}, seed={rng_seed}) ==="
    )
    print(
        f"  prior: {PRIOR_COLUMN} from [{-mf.ITI_START_BEFORE_MS}, "
        f"{-mf.ITI_END_BEFORE_MS}) ms before stimOn"
    )

    g_i_fit, d_s_presence = fitted_integrator_scales(weights_json)
    res_by_variant = {"absence": {}, "presence": {}}
    for cond, g_s, d_s in (
        ("absence", 0.0, 0.0),
        ("presence", g_i_fit, d_s_presence),
    ):
        mp, meta = load_fitted_model(g_s=g_s, d_s=d_s, json_path=weights_json)
        print(f"\n--- Simulating {cond} (g_s={g_s}, d_s={d_s}) ---")
        session_dfs, steps_before_obs, _ = simulate_condition_sessions(
            mp, n_sessions, blocks_per_session, max_obs_per_trial, rng_seed
        )
        for variant_suffix, contrast_matched in NULL_VARIANTS.items():
            null_label = (
                NULL_SCHEME_CONTRAST_MATCHED if contrast_matched else NULL_SCHEME_LABEL_SHUFFLE
            )
            print(f"  building {variant_suffix} ({null_label})...")
            res_dir, _ = _write_res_and_figures(
                cond,
                session_dfs,
                steps_before_obs,
                nrand,
                rng_seed,
                base_dir,
                variant_suffix,
                contrast_matched,
                n_jobs,
                prior_label=variant_suffix,
            )
            res_by_variant[cond][variant_suffix] = res_dir

    absence_null_metrics = None
    if not skip_absence_pres_null:
        absence_null_metrics = run_absence_pres_abs_null(
            base_dir,
            n_replicates=n_absence_pres_null_replicates,
            n_sessions=n_sessions,
            weights_json=weights_json,
            n_jobs=n_jobs,
            max_obs_per_trial=max_obs_per_trial,
            blocks_per_session=blocks_per_session,
            force=force_absence_pres_null,
        )

    for variant_suffix, contrast_matched in NULL_VARIANTS.items():
        suffix = f"_{variant_suffix}"
        compare_conditions(
            base_dir,
            abs_res_dir=res_by_variant["absence"][variant_suffix],
            pres_res_dir=res_by_variant["presence"][variant_suffix],
            out_json=base_dir / f"sensory_prior_recovery{suffix}.json",
            comparison_metrics_path=base_dir / f"comparison_summary_metrics{suffix}.png",
            s_prior_comparison_path=base_dir / f"s_prior_comparison{suffix}.png",
            shuffle_combined_path=base_dir / f"s_shuffle_control_combined{suffix}.png",
            prior_label=variant_suffix,
            prior_column=PRIOR_COLUMN,
            absence_null_metrics=absence_null_metrics,
        )

    print(f"\nNull comparison figures under {base_dir}")
    print("  per-condition: figs_contrast_null/, figs_label_null/")
    print("  top-level: *_contrast_null.* and *_label_null.*")


def run_prior_label_comparison(base_dir, weights_json=None, n_jobs=None, max_obs_per_trial=400):
    """
    Compare subjective vs true block prior grouping on the same simulated trajectories.

    Subjective: re-plots from existing res/ (p_subjective_probabilityLeft).
    Block: re-simulates both conditions (paired seed), writes res_prior_block/ + figs.
    """
    base_dir = Path(base_dir)
    cfg = _load_run_config(base_dir)
    nrand = _parse_nrand_from_summary(cfg)
    n_jobs = n_jobs if n_jobs is not None else _default_n_jobs()
    weights_json = resolve_weights_json(weights_json)
    rng_seed = cfg["rng_seed"]
    n_sessions = cfg["n_sessions"]
    blocks_per_session = cfg["blocks_per_session"]

    print(f"\n=== Prior label comparison (n_sessions={n_sessions}, nrand={nrand}, seed={rng_seed}) ===")

    for variant_name, prior_column in PRIOR_LABEL_VARIANTS.items():
        print(f"\n--- Prior variant: {variant_name} ({prior_column}) ---")
        abs_res_dir = None
        pres_res_dir = None

        if variant_name == "subjective":
            abs_res_dir = base_dir / "absence" / "res"
            pres_res_dir = base_dir / "presence" / "res"
            for cond in ("absence", "presence"):
                fig_dir = base_dir / cond / f"figs_prior_{variant_name}"
                fig_dir.mkdir(parents=True, exist_ok=True)
                res_dir = base_dir / cond / "res"
                s_prior = s_only_prior_test(res_dir)
                if s_prior:
                    s_prior["prior_column"] = prior_column
                plot_s_prior_figures(cond, s_prior, fig_dir, prior_label=variant_name)
                if s_prior:
                    pd.DataFrame(
                        [{k: v for k, v in s_prior.items() if not isinstance(v, np.ndarray)}]
                    ).to_csv(fig_dir / "s_prior_stats.csv", index=False)
        else:
            for cond, g_s, d_s in (
                ("absence", cfg["g_s_absence"], cfg["d_s_absence"]),
                ("presence", cfg["g_s_presence"], cfg["d_s_presence"]),
            ):
                print(f"  re-simulating {cond} for block prior...")
                mp, _ = load_fitted_model(g_s=g_s, d_s=d_s, json_path=weights_json)
                session_dfs, steps_before_obs, _ = simulate_condition_sessions(
                    mp, n_sessions, blocks_per_session, max_obs_per_trial, rng_seed
                )
                res_dir = base_dir / cond / f"res_prior_{variant_name}"
                write_res_for_prior_column(
                    session_dfs,
                    steps_before_obs,
                    nrand,
                    rng_seed,
                    res_dir,
                    prior_column,
                    n_jobs=n_jobs,
                )
                fig_dir = base_dir / cond / f"figs_prior_{variant_name}"
                fig_dir.mkdir(parents=True, exist_ok=True)
                s_prior = s_only_prior_test(res_dir)
                if s_prior:
                    s_prior["prior_column"] = prior_column
                plot_s_prior_figures(cond, s_prior, fig_dir, prior_label=variant_name)
                if s_prior:
                    pd.DataFrame(
                        [{k: v for k, v in s_prior.items() if not isinstance(v, np.ndarray)}]
                    ).to_csv(fig_dir / "s_prior_stats.csv", index=False)
                if cond == "absence":
                    abs_res_dir = res_dir
                else:
                    pres_res_dir = res_dir

        suffix = f"_prior_{variant_name}"
        compare_conditions(
            base_dir,
            abs_res_dir=abs_res_dir,
            pres_res_dir=pres_res_dir,
            out_json=base_dir / f"sensory_prior_recovery{suffix}.json",
            comparison_metrics_path=base_dir / f"comparison_summary_metrics{suffix}.png",
            s_prior_comparison_path=base_dir / f"s_prior_comparison{suffix}.png",
            shuffle_combined_path=base_dir / f"s_shuffle_control_combined{suffix}.png",
            prior_label=variant_name,
            prior_column=prior_column,
        )

    print(f"\nPrior comparison figures written under {base_dir}")
    print("  per-condition: absence/figs_prior_{subjective,block}/, presence/figs_prior_{subjective,block}/")
    print("  top-level: *_prior_subjective.png/json and *_prior_block.png/json")


def run_pres_abs_null_update(
    base_dir,
    weights_json=None,
    n_jobs=None,
    max_obs_per_trial=400,
    n_sessions=N_SESSIONS_DEFAULT,
    n_absence_pres_null_replicates=ABSENCE_PRES_NULL_REPLICATES_DEFAULT,
    force_absence_pres_null=False,
    blocks_per_session=BLOCKS_PER_SESSION_DEFAULT,
):
    """Run absence replicate null and refresh comparison summary plots from existing res/."""
    base_dir = Path(base_dir)
    n_jobs = n_jobs if n_jobs is not None else _default_n_jobs()
    absence_null_metrics = run_absence_pres_abs_null(
        base_dir,
        n_replicates=n_absence_pres_null_replicates,
        n_sessions=n_sessions,
        weights_json=weights_json,
        n_jobs=n_jobs,
        max_obs_per_trial=max_obs_per_trial,
        blocks_per_session=blocks_per_session,
        force=force_absence_pres_null,
    )

    compare_targets = [("", base_dir / "absence" / "res", base_dir / "presence" / "res")]
    for variant_suffix in NULL_VARIANTS:
        abs_res = base_dir / "absence" / f"res_{variant_suffix}"
        pres_res = base_dir / "presence" / f"res_{variant_suffix}"
        if abs_res.exists() and pres_res.exists():
            compare_targets.append((f"_{variant_suffix}", abs_res, pres_res))

    for suffix, abs_res, pres_res in compare_targets:
        prior_label = suffix.lstrip("_") if suffix else ""
        compare_conditions(
            base_dir,
            abs_res_dir=abs_res,
            pres_res_dir=pres_res,
            out_json=base_dir / f"sensory_prior_recovery{suffix}.json",
            comparison_metrics_path=base_dir / f"comparison_summary_metrics{suffix}.png",
            s_prior_comparison_path=base_dir / f"s_prior_comparison{suffix}.png",
            shuffle_combined_path=base_dir / f"s_shuffle_control_combined{suffix}.png",
            prior_label=prior_label,
            prior_column=PRIOR_COLUMN,
            absence_null_metrics=absence_null_metrics,
        )

    print(f"\nUpdated comparison metrics with absence pres-abs null under {base_dir}")


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
    parser.add_argument(
        "--g-s-presence",
        type=float,
        default=None,
        help="Sensory prior gain in presence (default: g_i from fitted weights)",
    )
    parser.add_argument(
        "--d-s-presence",
        type=float,
        default=None,
        help="Sensory prior offset in presence (default: d_i from weights JSON)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output root (default: <ONE cache>/manifold_sim). "
            "Repo paths like output/... are auto-redirected to manifold_sim."
        ),
    )
    parser.add_argument(
        "--allow-repo-output",
        action="store_true",
        help="Allow --output-dir inside the git repo (default: redirect to manifold_sim)",
    )
    parser.add_argument(
        "--weights-json",
        type=str,
        default=None,
        help="Fitted weights JSON (default: search ONE cache / ONE_CACHE_DIR)",
    )
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
    parser.add_argument(
        "--label-shuffle-null",
        action="store_true",
        help="Use unrestricted label shuffle nulls (default: contrast-matched nulls)",
    )
    parser.add_argument(
        "--null-compare",
        action="store_true",
        help=(
            "Simulate once per condition; plot contrast-matched vs unrestricted nulls "
            "(figs_contrast_null/, figs_label_null/)"
        ),
    )
    parser.add_argument(
        "--prior-compare",
        action="store_true",
        help=(
            "Compare subjective vs true block prior grouping; "
            "writes figs_prior_{subjective,block}/ and top-level *_prior_*.png"
        ),
    )
    parser.add_argument(
        "--absence-pres-null-replicates",
        type=int,
        default=ABSENCE_PRES_NULL_REPLICATES_DEFAULT,
        help=(
            "Number of absence simulations for direct pres vs abs null "
            f"(default: {ABSENCE_PRES_NULL_REPLICATES_DEFAULT})"
        ),
    )
    parser.add_argument(
        "--skip-absence-pres-null",
        action="store_true",
        help="Skip absence replicate null for direct presence vs absence p-values",
    )
    parser.add_argument(
        "--force-absence-pres-null",
        action="store_true",
        help="Re-run absence replicate null even if cached CSV exists",
    )
    parser.add_argument(
        "--pres-abs-null-only",
        action="store_true",
        help=(
            "Run absence replicate null and update comparison_summary_metrics "
            "from existing res/ outputs"
        ),
    )
    parser.add_argument(
        "--block-confound-plots",
        action="store_true",
        help=(
            "Plot RT, contrast, and S peak-time distributions (left vs right block) "
            "per act_block_duringstim split"
        ),
    )
    parser.add_argument(
        "--phase2-adaptation",
        action="store_true",
        help=(
            "Phase 2: adaptation/history confounds (a at stim, ITI ||S||, matched-block "
            "S trajectories, trial-history matched distances)"
        ),
    )
    parser.add_argument(
        "--phase4-no-prior-mod",
        action="store_true",
        help=(
            "Phase 4: absence with g_s=d_s=g_i=d_i=g_m=d_m=0; S/I/M prior distance "
            "vs shuffle (label confound with no generative P coupling)"
        ),
    )
    parser.add_argument(
        "--phase4-constant-s0",
        action="store_true",
        help=(
            "With --phase4-no-prior-mod (or alone): set S0 to deterministic contrast "
            "on signal side (zero noise) instead of stochastic stimuli"
        ),
    )
    parser.add_argument(
        "--unsplit-prior",
        nargs="+",
        choices=["phase4", "absence", "all"],
        metavar="CASE",
        help=(
            "Experiment A: unsplit prior distance (no f1/f2 splits). "
            "Cases: phase4 (all g/d=0), absence (fitted I/M), or all. "
            "Use --unsplit-mode fully for L+R pooled (diagnostic only). "
            "E.g. --unsplit-prior phase4 absence"
        ),
    )
    parser.add_argument(
        "--unsplit-mode",
        choices=["stim_side", "fully"],
        default="stim_side",
        help=(
            "Unsplit trial pooling: stim_side (default; stim_l + stim_r stacked) or "
            "fully (all duringstim trials, L+R mixed — S artefact risk)"
        ),
    )
    parser.add_argument(
        "--s-only-presence",
        action="store_true",
        help=(
            "S-only sensory prior: presence g_s/d_s (default: g_i/d_i from weights) "
            "with g_i=d_i=g_m=d_m=0; runs S-prior analysis + block-confound plots"
        ),
    )
    parser.add_argument(
        "--s-presence-i-scaled-plots",
        action="store_true",
        help=(
            "Full plot suite with g_s = g_i * (|S|/|S0|), d_s = d_i "
            "(integrator-comparable scaling) → s_presence_tune/*_i_scaled/"
        ),
    )
    parser.add_argument(
        "--gs-outside-adaptation",
        action="store_true",
        help=(
            "Move g_s outside the adaptation gate: a*(J@S0) + g_s*P_gain@S0 "
            "instead of a*((J + g_s*P_gain)@S0). Tag: *_gs_free."
        ),
    )
    parser.add_argument(
        "--s-presence-tuned-plots",
        action="store_true",
        help=(
            "Full plot suite for tuned s_presence_only (default g_s=1, d_s=48): "
            "S/I prior curves, shuffle controls, block confounds → s_presence_tune/"
        ),
    )
    parser.add_argument(
        "--gs-ds-tune",
        action="store_true",
        help=(
            "Grid search g_s/d_s (g_i=d_i=g_m=d_m=0); report S and I prior significance "
            "per pair; writes gs_ds_tune_sweep.csv"
        ),
    )
    parser.add_argument(
        "--g-s-grid",
        type=str,
        default=None,
        help="Comma-separated g_s values for --gs-ds-tune (default: spaced around g_i_fitted)",
    )
    parser.add_argument(
        "--d-s-grid",
        type=str,
        default=None,
        help="Comma-separated d_s values for --gs-ds-tune (default: spaced around d_i_fitted)",
    )
    parser.add_argument(
        "--stop-on-s-significant",
        action="store_true",
        help="With --gs-ds-tune: stop sweep at first S-significant pair",
    )
    parser.add_argument(
        "--stop-on-s-p-gain",
        action="store_true",
        help="With --gs-ds-tune: stop at first S p_gain-significant pair",
    )
    parser.add_argument(
        "--stop-on-s-mean-and-gain",
        action="store_true",
        help="With --gs-ds-tune: stop at first pair with S p_mean and p_gain both significant",
    )
    parser.add_argument(
        "--gs-tune-p-gain",
        action="store_true",
        help=(
            "Tune g_s at fixed d_s (g_i=d_i=g_m=d_m=0) for S p_gain significance; "
            "requires --d-s-fixed; also requires p_mean sig unless --p-gain-only"
        ),
    )
    parser.add_argument(
        "--d-s-fixed",
        type=float,
        default=50.0,
        help="Fixed d_s for --gs-tune-p-gain (default: 50)",
    )
    parser.add_argument(
        "--p-gain-only",
        action="store_true",
        help="With --gs-tune-p-gain: stop at p_gain sig without requiring p_mean",
    )
    parser.add_argument(
        "--tune-alpha",
        type=float,
        default=0.01,
        help="Significance threshold for --gs-ds-tune (default: 0.01)",
    )
    parser.add_argument(
        "--matched-covariate-prior",
        action="store_true",
        help=(
            "Matched trial-set S prior distance: match on contrast, trial_side, choice, "
            "feedback, RT bin, trial_in_block bin, session; recompute vs shuffle null"
        ),
    )
    parser.add_argument(
        "--random-prior-labels",
        action="store_true",
        help=(
            "Absence: compare true ITI-P labels vs random 0.8/0.2 labels on same trajectories"
        ),
    )
    parser.add_argument(
        "--random-prior-replicates",
        type=int,
        default=50,
        help="Number of random prior label draws (default: 50)",
    )
    parser.add_argument(
        "--duringstim-window-ms",
        type=float,
        default=None,
        help=(
            "Override the post-stim PRE_POST window (ms) for duringstim splits "
            f"(default: {IM_DURINGSTIM_WINDOW_S*1000:.0f} ms for I/M). "
            f"S population always uses {S_DURINGSTIM_WINDOW_S*1000:.0f} ms via "
            "S_DURINGSTIM_WINDOW_S in build_population_b_for_split."
        ),
    )
    args = parser.parse_args()

    # Apply optional window override before any analysis functions reference PRE_POST.
    if args.duringstim_window_ms is not None:
        win_s = args.duringstim_window_ms / 1000.0
        for key in list(PRE_POST.keys()):
            if "duringstim" in key:
                PRE_POST[key] = [0, win_s]

    base_dir = resolve_output_dir(args.output_dir, allow_repo_output=args.allow_repo_output)
    base_dir.mkdir(parents=True, exist_ok=True)
    n_jobs = args.n_jobs if args.n_jobs is not None else _default_n_jobs()
    s_prior_only = not args.full_analysis
    contrast_matched_null = not args.label_shuffle_null

    if args.null_compare:
        run_null_scheme_comparison(
            base_dir,
            weights_json=args.weights_json,
            n_jobs=n_jobs,
            max_obs_per_trial=args.max_obs_per_trial,
            n_sessions=args.n_sessions,
            nrand=args.nrand,
            rng_seed=args.seed,
            blocks_per_session=args.blocks_per_session,
            n_absence_pres_null_replicates=args.absence_pres_null_replicates,
            skip_absence_pres_null=args.skip_absence_pres_null,
            force_absence_pres_null=args.force_absence_pres_null,
        )
        return

    if args.pres_abs_null_only:
        run_pres_abs_null_update(
            base_dir,
            weights_json=args.weights_json,
            n_jobs=n_jobs,
            max_obs_per_trial=args.max_obs_per_trial,
            n_sessions=args.n_sessions,
            n_absence_pres_null_replicates=args.absence_pres_null_replicates,
            force_absence_pres_null=args.force_absence_pres_null,
            blocks_per_session=args.blocks_per_session,
        )
        return

    if args.block_confound_plots:
        weights_json = resolve_weights_json(args.weights_json)
        g_i_fit, d_s_presence = fitted_integrator_scales(weights_json)
        g_s_presence = args.g_s_presence if args.g_s_presence is not None else g_i_fit
        d_s_presence = args.d_s_presence if args.d_s_presence is not None else d_s_presence
        for cond, g_s, d_s in (
            ("absence", 0.0, 0.0),
            ("presence", g_s_presence, d_s_presence),
        ):
            run_block_confound_plots(
                base_dir,
                condition=cond,
                g_s=g_s,
                d_s=d_s,
                weights_json=weights_json,
                n_sessions=args.n_sessions,
                blocks_per_session=args.blocks_per_session,
                max_obs_per_trial=args.max_obs_per_trial,
                rng_seed=args.seed,
            )
        return

    if args.phase2_adaptation:
        weights_json = resolve_weights_json(args.weights_json)
        run_phase2_adaptation_analysis(
            base_dir,
            condition="absence",
            g_s=0.0,
            d_s=0.0,
            weights_json=weights_json,
            n_sessions=args.n_sessions,
            blocks_per_session=args.blocks_per_session,
            max_obs_per_trial=args.max_obs_per_trial,
            rng_seed=args.seed,
        )
        return

    if args.phase4_no_prior_mod or args.phase4_constant_s0:
        weights_json = resolve_weights_json(args.weights_json)
        run_phase4_no_prior_mod_analysis(
            base_dir,
            weights_json=weights_json,
            n_sessions=args.n_sessions,
            blocks_per_session=args.blocks_per_session,
            max_obs_per_trial=args.max_obs_per_trial,
            rng_seed=args.seed,
            nrand=args.nrand,
            n_jobs=n_jobs,
            contrast_matched_null=contrast_matched_null,
            constant_s0=args.phase4_constant_s0,
        )
        return

    if args.unsplit_prior:
        weights_json = resolve_weights_json(args.weights_json)
        cases = list(args.unsplit_prior)
        if "all" in cases:
            cases = ["phase4", "absence"]
        print(f"ONE cache: {resolve_one_cache_dir()}")
        print(f"Output: {base_dir}")
        for case in cases:
            run_unsplit_prior_distance_analysis(
                base_dir,
                case=case,
                weights_json=weights_json,
                n_sessions=args.n_sessions,
                blocks_per_session=args.blocks_per_session,
                max_obs_per_trial=args.max_obs_per_trial,
                rng_seed=args.seed,
                nrand=args.nrand,
                n_jobs=n_jobs,
                contrast_matched_null=contrast_matched_null,
                unsplit_mode=args.unsplit_mode,
            )
        return

    if args.s_presence_i_scaled_plots:
        weights_json = resolve_weights_json(args.weights_json)
        run_s_presence_i_scaled_plots(
            base_dir,
            weights_json=weights_json,
            n_sessions=args.n_sessions,
            blocks_per_session=args.blocks_per_session,
            max_obs_per_trial=args.max_obs_per_trial,
            rng_seed=args.seed,
            nrand=args.nrand,
            n_jobs=n_jobs,
            contrast_matched_null=contrast_matched_null,
            gs_outside_adaptation=args.gs_outside_adaptation,
        )
        return

    if args.s_presence_tuned_plots:
        weights_json = resolve_weights_json(args.weights_json)
        g_s = args.g_s_presence if args.g_s_presence is not None else 1.0
        d_s = args.d_s_presence if args.d_s_presence is not None else 48.0
        tag_suffix = "gs_free" if args.gs_outside_adaptation else ""
        run_s_presence_tuned_plots(
            base_dir,
            g_s=g_s,
            d_s=d_s,
            weights_json=weights_json,
            n_sessions=args.n_sessions,
            blocks_per_session=args.blocks_per_session,
            max_obs_per_trial=args.max_obs_per_trial,
            rng_seed=args.seed,
            nrand=args.nrand,
            n_jobs=n_jobs,
            contrast_matched_null=contrast_matched_null,
            tag_suffix=tag_suffix,
            gs_outside_adaptation=args.gs_outside_adaptation,
        )
        return

    if args.s_only_presence:
        weights_json = resolve_weights_json(args.weights_json)
        g_i_fit, d_i_fit = fitted_integrator_scales(weights_json)
        g_s_presence = args.g_s_presence if args.g_s_presence is not None else g_i_fit
        d_s_presence = args.d_s_presence if args.d_s_presence is not None else d_i_fit
        run_s_only_presence_analysis(
            base_dir,
            weights_json=weights_json,
            n_sessions=args.n_sessions,
            blocks_per_session=args.blocks_per_session,
            max_obs_per_trial=args.max_obs_per_trial,
            rng_seed=args.seed,
            nrand=args.nrand,
            n_jobs=n_jobs,
            contrast_matched_null=contrast_matched_null,
            g_s_presence=g_s_presence,
            d_s_presence=d_s_presence,
        )
        return

    if args.gs_ds_tune:
        weights_json = resolve_weights_json(args.weights_json)
        g_s_grid = _parse_float_list(args.g_s_grid) if args.g_s_grid else None
        d_s_grid = _parse_float_list(args.d_s_grid) if args.d_s_grid else None
        run_gs_ds_tune_sweep(
            base_dir,
            weights_json=weights_json,
            n_sessions=args.n_sessions,
            blocks_per_session=args.blocks_per_session,
            max_obs_per_trial=args.max_obs_per_trial,
            rng_seed=args.seed,
            nrand=args.nrand,
            n_jobs=n_jobs,
            contrast_matched_null=contrast_matched_null,
            g_s_values=g_s_grid,
            d_s_values=d_s_grid,
            alpha=args.tune_alpha,
            stop_on_s_significant=args.stop_on_s_significant,
            stop_on_s_p_gain=args.stop_on_s_p_gain,
            stop_on_s_mean_and_gain=args.stop_on_s_mean_and_gain,
            gs_outside_adaptation=args.gs_outside_adaptation,
        )
        return

    if args.gs_tune_p_gain:
        weights_json = resolve_weights_json(args.weights_json)
        g_s_grid = _parse_float_list(args.g_s_grid) if args.g_s_grid else None
        run_gs_tune_p_gain(
            base_dir / "gs_tune_p_gain",
            d_s_fixed=args.d_s_fixed,
            weights_json=weights_json,
            n_sessions=args.n_sessions,
            blocks_per_session=args.blocks_per_session,
            max_obs_per_trial=args.max_obs_per_trial,
            rng_seed=args.seed,
            nrand=args.nrand,
            n_jobs=n_jobs,
            contrast_matched_null=contrast_matched_null,
            g_s_values=g_s_grid,
            alpha=args.tune_alpha,
            require_p_mean=not args.p_gain_only,
            gs_outside_adaptation=args.gs_outside_adaptation,
        )
        return

    if args.matched_covariate_prior:
        weights_json = resolve_weights_json(args.weights_json)
        run_matched_covariate_prior_test(
            base_dir,
            condition="absence",
            g_s=0.0,
            d_s=0.0,
            weights_json=weights_json,
            n_sessions=args.n_sessions,
            blocks_per_session=args.blocks_per_session,
            max_obs_per_trial=args.max_obs_per_trial,
            rng_seed=args.seed,
            nrand=args.nrand,
            n_jobs=n_jobs,
            contrast_matched_null=contrast_matched_null,
        )
        return

    if args.random_prior_labels:
        run_random_prior_label_test(
            base_dir,
            n_random_replicates=args.random_prior_replicates,
            n_sessions=args.n_sessions,
            blocks_per_session=args.blocks_per_session,
            max_obs_per_trial=args.max_obs_per_trial,
            rng_seed=args.seed,
            nrand=args.nrand,
            n_jobs=n_jobs,
            weights_json=args.weights_json,
            contrast_matched_null=contrast_matched_null,
        )
        return

    if args.prior_compare:
        run_prior_label_comparison(
            base_dir,
            weights_json=args.weights_json,
            n_jobs=n_jobs,
            max_obs_per_trial=args.max_obs_per_trial,
        )
        return

    if args.recovery_only:
        run_recovery_only(base_dir, s_prior_only=s_prior_only)
        return

    weights_json = resolve_weights_json(args.weights_json)
    g_i_fit, d_i_fit = fitted_integrator_scales(weights_json)
    g_s_presence = args.g_s_presence if args.g_s_presence is not None else g_i_fit
    d_s_presence = args.d_s_presence if args.d_s_presence is not None else d_i_fit
    print(f"ONE cache: {resolve_one_cache_dir()}")
    print(f"Output: {base_dir}")
    print(f"Weights: {weights_json}")
    print(
        f"Presence sensory prior: g_s={g_s_presence}, d_s={d_s_presence} "
        f"(paired seed={args.seed} for both conditions)"
    )

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
        contrast_matched_null=contrast_matched_null,
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
        g_s=g_s_presence,
        d_s=d_s_presence,
        rng_seed=args.seed,
        **common_kw,
    )
    absence_null_metrics = None
    if not args.skip_absence_pres_null:
        absence_null_metrics = run_absence_pres_abs_null(
            base_dir,
            n_replicates=args.absence_pres_null_replicates,
            n_sessions=args.n_sessions,
            weights_json=weights_json,
            n_jobs=n_jobs,
            max_obs_per_trial=args.max_obs_per_trial,
            blocks_per_session=args.blocks_per_session,
            force=args.force_absence_pres_null,
        )
    compare_conditions(base_dir, absence_null_metrics=absence_null_metrics)


if __name__ == "__main__":
    main()
