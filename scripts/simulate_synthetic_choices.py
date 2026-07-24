"""
Run the IBL 'action-kernel' behavioural model on a trials object to simulate
choices -- i.e. build the "synthetic sessions" used as the choice/feedback null
distribution in the Brainwide Map paper.

This is a stand-alone re-implementation of the call chain in
  prior_localization/functions/nulldistributions.py::generate_null_distribution_session
      -> generate_choices(...)            # fit/load ActionKernel, then .simulate()
using the upstream `behavior_models` package directly.

--------------------------------------------------------------------------------
WHAT YOU NEED TO SIMULATE CHOICES
--------------------------------------------------------------------------------
1. Fitted model parameters for the session: for the action-kernel with
   single_zeta=True these are 4 numbers  [alpha, zeta, lapse_pos, lapse_neg].
   They are obtained by MCMC-fitting the model to the *real* session's
   (actions, stimuli, stim_side), then taking the posterior mean. In the
   paper's pipeline these are loaded from a pre-fit pickle.

2. A stimulus sequence to simulate over: per trial a
       signed contrast  (stim, in [-1, 1])  and
       stimulus side    (side, -1 / +1).
   In the paper this sequence comes from a *pseudo-session* (freshly drawn
   block structure + contrasts), NOT the real one -- that is what makes the
   target behaviour-independent of the recorded neural activity.

   NOTE: the action kernel feeds its *own* previous (simulated) choice back in
   to update the prior, so you do NOT pass actions to .simulate(); only the
   stimulus stream is needed once the parameters are known.

To FIT the parameters you need the real session's choice / contrastLeft /
contrastRight / probabilityLeft / feedbackType columns. To only SIMULATE with
already-known parameters you just need the stimulus stream.

--------------------------------------------------------------------------------
USAGE IN THIS REPO
--------------------------------------------------------------------------------
BWM / paper null (default for ``--actkernel-choice-null``):
  ``synthetic_sessions_from_trials`` / ``make_synthetic_session`` — regenerate
  stim+block via ``generate_pseudo_session``, then simulate choices under fitted θ.

Fixed-stim helper (still available; not the wired null):
  ``synthetic_choices_fixed_stim`` / ``simulate_choices`` on
  ``stim_side_from_trials(trials)`` — artificial choices on the *real*
  stim/side stream.

``behavior_models`` is vendored as git submodule ``third_party/behavior_models``
(path-prepended below). Remote jobs only need this checkout + ``torch`` in the
conda env — no ``pip install behavior_models`` required. Init with::

    git submodule update --init --recursive
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Prefer in-repo submodule so Slurm / remote iblenv need not pip-install it.
# Layout: third_party/behavior_models/behavior_models/...
_REPO_ROOT = Path(__file__).resolve().parents[1]
_BM_ROOT = _REPO_ROOT / 'third_party' / 'behavior_models'
if _BM_ROOT.is_dir() and str(_BM_ROOT) not in sys.path:
    sys.path.insert(0, str(_BM_ROOT))

from behavior_models.models import ActionKernel
from behavior_models.utils import format_data, format_input as mut_format_input


# ActionKernel parameter order (single_zeta=True): see models.py::ActionKernel.simulate
PARAM_NAMES = ["alpha", "zeta", "lapse_pos", "lapse_neg"]


def _as_trials_df(trials):
    """Coerce a trials object to a DataFrame.

    Accepts a pandas DataFrame, an ALF `AlfBunch` (e.g. from one.load_object(eid, 'trials')),
    or a plain dict of equal-length arrays.
    """
    if isinstance(trials, pd.DataFrame):
        return trials
    if hasattr(trials, "to_df"):               # AlfBunch / Bunch
        return trials.to_df()
    return pd.DataFrame({k: np.asarray(v) for k, v in dict(trials).items()})


def fit_action_kernel(trials_df, eid="0000face-0000-0000-0000-000000000000",
                      subject="synthetic_mouse", model_dir=None,
                      nb_steps=None, nb_chains=4):
    """Fit the action-kernel model to a real session and return the model + posterior-mean params.

    trials_df must contain: choice, contrastLeft, contrastRight, feedbackType, probabilityLeft.
    nb_steps=None uses the package default (up to 5000 MCMC steps with early stopping);
    pass a small int (e.g. 300) for a quick approximate fit.
    """
    model_dir = Path(model_dir) if model_dir else Path(tempfile.mkdtemp(prefix="actkernel_"))
    trials_df = _as_trials_df(trials_df)

    # Real session regressors, exactly as the package expects them.
    stim_side, stimuli, actions, _ = format_data(trials_df)

    model = ActionKernel(
        path_to_results=model_dir,
        session_uuids=[eid],
        mouse_name=subject,
        actions=np.asarray(actions, dtype=float),
        stimuli=np.asarray(stimuli, dtype=float),
        stim_side=np.asarray(stim_side, dtype=float),
        single_zeta=True,
    )
    # Trains via MCMC and caches the pickle (or loads it if already there).
    model.load_or_train(sessions_id=np.array([0]), nb_steps=nb_steps, nb_chains=nb_chains)
    params = model.get_parameters(parameter_type="posterior_mean")
    return model, params


def _sim_model():
    """A reusable ActionKernel in pseudo-session mode (only .simulate is ever called)."""
    return ActionKernel(
        path_to_results=tempfile.mkdtemp(prefix="actkernel_sim_"),
        session_uuids=["sim0"], mouse_name="sim", single_zeta=True,
    )


def simulate_choices(stim, side, params, n_sim=1, seed=None, model=None):
    """Core simulator: run the action kernel forward over a stimulus stream.

    stim  : 1d array of signed contrasts in [-1, 1]
    side  : 1d array of stimulus sides (-1 / +1)
    params: length-4 array [alpha, zeta, lapse_pos, lapse_neg]
    n_sim : number of independent simulated sessions to draw
    model : optional pre-built ActionKernel to reuse (avoids rebuilding it on every call)
    returns act_sim with shape (n_trials,) if n_sim==1 else (n_sim, n_trials), values in {-1, +1}
    """
    import torch
    if seed is not None:
        torch.manual_seed(seed)

    if model is None:
        model = _sim_model()  # pseudo-session mode: no actions needed, only .simulate is used

    arr_params = np.asarray(params, dtype=float)[None]          # shape (1, 4)
    stim_f, _, side_f = mut_format_input([np.asarray(stim)], [np.zeros_like(stim)], [np.asarray(side)])
    act_sim, _, _ = model.simulate(arr_params, stim_f.squeeze(), side_f.squeeze(),
                                   nb_simul=n_sim, only_perf=False)
    act_sim = np.array(act_sim.squeeze().T, dtype=np.int64)
    return act_sim


def make_synthetic_session(trials_df, params, seed=None, n_trials=None):
    """Reproduce generate_null_distribution_session: a pseudo-session whose choice &
    feedbackType are simulated by the action-kernel model. This is one draw of the
    paper's 'synthetic session' choice/feedback null.

    ``n_trials``: if set, draw a pseudo of that length (contrast set still from
    ``trials_df``). Default None → same length as ``trials_df`` (BWM default).

    Requires brainbox for the pseudo-session generator.
    """
    from brainbox.task.closed_loop import generate_pseudo_session
    if seed is not None:
        np.random.seed(seed)

    trials_df = _as_trials_df(trials_df)
    if n_trials is None or int(n_trials) == trials_df.shape[0]:
        pseudosess = generate_pseudo_session(trials_df, generate_choices=False)
    else:
        # Longer (or shorter) world: same generative process, independent length.
        signed, side, pleft = _pseudo_sessions_vectorized(
            trials_df, n=1, seed=seed, n_trials=int(n_trials))
        pseudosess = pd.DataFrame({
            'signed_contrast': signed[0],
            'stim_side': side[0],
            'probabilityLeft': pleft[0],
        })
    choice = simulate_choices(pseudosess.signed_contrast.values,
                              pseudosess.stim_side.values, params, n_sim=1, seed=seed)
    pseudosess["choice"] = choice
    pseudosess["feedbackType"] = np.where(pseudosess["choice"] == pseudosess["stim_side"], 1, -1)
    return pseudosess


def generate_synthetic_sessions(eid, n=2000, one=None, nb_steps=None, seed=0,
                                model_dir=None, return_dataframes=False, fast=True):
    """Standalone: from a session eid, build `n` synthetic sessions (the choice/feedback null).

    Pipeline (matches the Brainwide Map paper):
      1. load the real session's trials for `eid` via ONE,
      2. MCMC-fit the action-kernel model once for this session,
      3. draw `n` pseudo-sessions and simulate choice + feedbackType on each with the fitted model.

    Parameters
    ----------
    eid : str               IBL experiment id.
    n : int                 number of synthetic sessions to generate (default 2000).
    one : ONE or None       an ONE instance; created with defaults if None.
    nb_steps : int or None  MCMC steps for the fit. None = package default (slow, up to 5000,
                            with early stopping) -- this is the paper-faithful choice.
    seed : int              base RNG seed; session i uses seed+i (reproducible).
    model_dir : str or None where to cache the fitted model pickle.
    return_dataframes : bool also return the list of `n` full pseudo-session DataFrames.

    Returns
    -------
    dict with rectangular arrays of shape (n, n_trials) -- every synthetic session has the
    same trial count as the real one:
        choice           int   {-1, +1}     <- the decoding target for 'choice'
        feedbackType     int   {-1, +1}     <- the decoding target for 'feedback'
        signed_contrast  float [-1, 1]      (regenerated stimulus stream)
        stim_side        float {-1, +1}
        probabilityLeft  float
    plus  eid, subject, params (the fitted [alpha, zeta, lapse_pos, lapse_neg]), n_trials,
    and (if requested) sessions = list of DataFrames.
    """
    from brainbox.io.one import SessionLoader
    from one.api import ONE

    one = one or ONE()
    subject = one.get_details(eid)["subject"]

    sl = SessionLoader(one=one, eid=eid)
    sl.load_trials()

    out = synthetic_sessions_from_trials(sl.trials, n=n, eid=eid, subject=subject,
                                         nb_steps=nb_steps, seed=seed, model_dir=model_dir,
                                         return_dataframes=return_dataframes, fast=fast)
    return out


def _pseudo_sessions_vectorized(trials, n, seed=0, n_trials=None):
    """Vectorized equivalent of n x generate_pseudo_session(..., generate_choices=False).

    Returns (signed_contrast, stim_side, probabilityLeft), each shape (n, n_trials), drawn from
    the SAME generative process as the upstream loop (biased blocks + non-uniform contrasts),
    but produced as arrays instead of per-trial DataFrame writes. The exact RNG sequence differs
    from the slow loop (so it is not seed-for-seed identical), but the distribution is the same.

    ``n_trials`` overrides session length (default = ``trials.shape[0]``). Use a larger
    value for stratified AK nulls when same-length pseudos undersample stim×prior strata.
    """
    from brainbox.task.closed_loop import generate_pseudo_blocks, _get_biased_probs

    np.random.seed(seed)
    n_trials = int(trials.shape[0] if n_trials is None else n_trials)
    if n_trials < 1:
        raise ValueError(f'n_trials must be ≥1, got {n_trials}')
    contrast_set = np.unique(trials["contrastLeft"][~np.isnan(trials["contrastLeft"])])
    idx0 = int(np.where(contrast_set == 0)[0][0])              # zero-contrast index
    p = np.array(_get_biased_probs(len(contrast_set), idx=idx0, prob=0.5))  # zero half as likely

    # block structure: probabilityLeft per trial, one independent draw per session
    pleft = np.stack([generate_pseudo_blocks(n_trials) for _ in range(n)])  # (n, n_trials)

    # stimulus side: position == -1 (left) with prob == probabilityLeft, else +1  (cf. _draw_position)
    position = np.where(np.random.random((n, n_trials)) < pleft, -1, 1)
    # absolute contrast drawn from the non-uniform contrast distribution
    contrast = np.random.choice(contrast_set, size=(n, n_trials), p=p)

    signed_contrast = contrast * position                      # = contrast * sign(position)
    stim_side = position.astype(float)
    return signed_contrast.astype(float), stim_side, pleft


def _synthetic_fast(trials, params, n, seed=0, n_trials=None):
    """Fast path: vectorized pseudo-sessions + a single batched simulate_parallel torch call."""
    signed_contrast, stim_side, pleft = _pseudo_sessions_vectorized(
        trials, n, seed=seed, n_trials=n_trials)

    import torch
    torch.manual_seed(seed)
    model = _sim_model()
    arr_params = np.asarray(params, dtype=float)[None]         # (1, 4): one parameter "chain"
    # simulate_parallel runs all n sessions through the trial loop at once (nb_simul=1 each).
    act_sim, _, _ = model.simulate_parallel(arr_params, signed_contrast, stim_side, nb_simul=1)
    choice = np.asarray(act_sim.squeeze(-1), dtype=np.int64)   # (n, n_trials), {-1, +1}
    feedback = np.where(choice == stim_side, 1, -1).astype(np.int64)
    return dict(choice=choice, feedbackType=feedback, signed_contrast=signed_contrast,
                stim_side=stim_side, probabilityLeft=pleft)


def synthetic_sessions_from_trials(trials, n=2000, eid="0000face-0000-0000-0000-000000000000",
                                   subject="synthetic_mouse", nb_steps=None, seed=0,
                                   model_dir=None, return_dataframes=False, params=None,
                                   fast=True, n_trials=None):
    """Standalone: from a trials DataFrame, build `n` synthetic sessions (choice/feedback null).

    Same as generate_synthetic_sessions but takes the trials object directly (no ONE lookup).

    Parameters
    ----------
    trials : DataFrame      real session trials with columns choice, contrastLeft, contrastRight,
                            feedbackType, probabilityLeft.
    n : int                 number of synthetic sessions (default 2000).
    eid, subject : str      only used to name/cache the fitted model.
    nb_steps : int or None  MCMC steps for the fit. None = package default (up to 5000, with
                            early stopping) -- the paper-faithful choice.
    seed : int              base RNG seed; session i uses seed+i (reproducible).
    params : array or None  pre-fitted [alpha, zeta, lapse_pos, lapse_neg]; if given, skips fitting.
    return_dataframes : bool also return the list of `n` full pseudo-session DataFrames.
    fast : bool             use the vectorized path (vectorized pseudo-sessions + one batched
                            simulate_parallel call). ~100x faster and distributionally identical.
                            Set fast=False for the exact upstream per-session loop (slow), e.g.
                            when you need return_dataframes or seed-for-seed parity with the paper.
    n_trials : int or None  length of each synthetic session. None → ``trials.shape[0]``.

    Returns
    -------
    dict of rectangular arrays of shape (n, n_trials) -- choice, feedbackType, signed_contrast,
    stim_side, probabilityLeft -- plus eid, subject, params, n_trials, and (optionally) sessions.
    """
    trials = _as_trials_df(trials)

    # Fit (or load cached) action-kernel for this session, once -- unless params were supplied.
    if params is None:
        _, params = fit_action_kernel(trials, eid=eid, subject=subject,
                                      model_dir=model_dir, nb_steps=nb_steps)

    n_real = trials.shape[0]
    n_out = int(n_real if n_trials is None else n_trials)

    if fast and not return_dataframes:
        arrs = _synthetic_fast(trials, params, n, seed=seed, n_trials=n_out)
        return dict(eid=eid, subject=subject, params=np.asarray(params), n_trials=n_out, **arrs)

    # ---- slow loop (per-session DataFrame; needed for return_dataframes) ----
    sim_model = _sim_model()           # build the torch model once, reuse for all n draws
    arrs = {k: np.empty((n, n_out), dtype=(np.int64 if k in ("choice", "feedbackType") else float))
            for k in ("choice", "feedbackType", "signed_contrast", "stim_side", "probabilityLeft")}
    sessions = [] if return_dataframes else None

    for i in range(n):
        ps = make_synthetic_session(
            trials, params, seed=seed + i, n_trials=n_out)
        for k in arrs:
            arrs[k][i] = ps[k].values
        if return_dataframes:
            sessions.append(ps)

    out = dict(eid=eid, subject=subject, params=np.asarray(params), n_trials=n_out, **arrs)
    if return_dataframes:
        out["sessions"] = sessions
    return out


def stim_side_from_trials(trials):
    """Extract the *real* session's signed contrast and stimulus side.

    Unlike ``synthetic_sessions_from_trials`` / ``make_synthetic_session``, this
    does **not** regenerate blocks or contrasts. Use with ``simulate_choices``
    when you need artificial choices on the same stim/block sequence as the
    recorded session (e.g. neural distance nulls with fixed eligibility).

    Returns
    -------
    stim : (n_trials,) float   signed contrast in [-1, 1]
    side : (n_trials,) float   stimulus side in {-1, +1}
    pLeft : (n_trials,) float  task ``probabilityLeft`` (block schedule)
    """
    trials = _as_trials_df(trials)
    stim_side, stimuli, _actions, pLeft = format_data(trials)
    return (
        np.asarray(stimuli, dtype=float),
        np.asarray(stim_side, dtype=float),
        np.asarray(pLeft, dtype=float),
    )


def synthetic_choices_fixed_stim(
        trials, params=None, n=1, eid="0000face-0000-0000-0000-000000000000",
        subject="synthetic_mouse", nb_steps=None, seed=None, model_dir=None,
        model=None):
    """Generate ``n`` choice sequences with the real stim / block schedule fixed.

    Fits ActionKernel on ``trials`` (unless ``params`` given), then calls
    ``simulate_choices`` on the **same** signed-contrast and stim-side stream
    as the recorded session. Kept as a helper; the wired
    ``--actkernel-choice-null`` path uses ``synthetic_sessions_from_trials``
    (BWM paper: regenerated pseudo stim/blocks) instead.

    Returns
    -------
    dict with
        choice : (n_trials,) if n==1 else (n, n_trials)  values in {-1, +1}
        params : fitted or provided [alpha, zeta, lapse_pos, lapse_neg]
        stim, side, probabilityLeft : fixed covariates from ``trials``
    """
    trials = _as_trials_df(trials)
    stim, side, pLeft = stim_side_from_trials(trials)

    if params is None:
        _, params = fit_action_kernel(
            trials, eid=eid, subject=subject, model_dir=model_dir,
            nb_steps=nb_steps)

    choice = simulate_choices(
        stim, side, params, n_sim=n, seed=seed, model=model)
    return dict(
        choice=np.asarray(choice, dtype=np.int64),
        params=np.asarray(params, dtype=float),
        stim=stim,
        side=side,
        probabilityLeft=pLeft,
    )


# # ----------------------------------------------------------------------------- demo
# if __name__ == "__main__":
#     rng = np.random.default_rng(0)
#     n = 400
#     contrasts = np.array([-1, -.25, -.125, -.0625, 0, .0625, .125, .25, 1.])
#     signed = rng.choice(contrasts, size=n)
#     # zero-contrast trials are assigned to one side with contrast 0 (as in real ALF data),
#     # so that the contrast set seen by generate_pseudo_session contains 0.
#     zero_side = rng.choice([-1, 1], size=n)
#     trials = pd.DataFrame({
#         "contrastLeft":  np.where(signed < 0, -signed, np.where((signed == 0) & (zero_side < 0), 0.0, np.nan)),
#         "contrastRight": np.where(signed > 0,  signed, np.where((signed == 0) & (zero_side > 0), 0.0, np.nan)),
#         "choice":        rng.choice([-1, 1], size=n),
#         "feedbackType":  rng.choice([-1, 1], size=n),
#         "probabilityLeft": rng.choice([0.2, 0.5, 0.8], size=n),
#     })

#     # 1) Demonstrate the simulator with hand-set parameters (fast, no MCMC).
#     demo_params = [0.3, 0.5, 0.05, 0.05]   # alpha, zeta, lapse_pos, lapse_neg
#     side = np.sign(np.nan_to_num(trials.contrastRight) - np.nan_to_num(trials.contrastLeft))
#     side[side == 0] = rng.choice([-1, 1], size=int((side == 0).sum()))
#     stim = np.nan_to_num(trials.contrastRight) - np.nan_to_num(trials.contrastLeft)
#     sim = simulate_choices(stim, side, demo_params, n_sim=3, seed=42)
#     print("simulate_choices output shape:", sim.shape, "unique vals:", np.unique(sim))
#     print("mean simulated p(right) per sim:", (sim == 1).mean(axis=1).round(3))

#     # 2) Quick end-to-end fit (short MCMC) then synthetic session.
#     print("\nFitting action-kernel (short MCMC, demo only)...")
#     model, params = fit_action_kernel(trials, nb_steps=200, nb_chains=4)
#     print("posterior-mean params:", dict(zip(PARAM_NAMES, np.round(params, 4))))

#     synth = make_synthetic_session(trials, params, seed=7)
#     print("\nsynthetic session columns:", list(synth.columns))
#     print(synth[["signed_contrast", "stim_side", "choice", "feedbackType"]].head())
#     print("synthetic feedback (frac correct):", (synth.feedbackType == 1).mean().round(3))
