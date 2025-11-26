import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from itertools import product
from scipy.signal import savgol_filter

from one.api import ONE
from reproducible_ephys_processing import bin_spikes2D
from brainwidemap import bwm_query, load_good_units, bwm_units
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions
from brainbox.io.one import SpikeSortingLoader, SessionLoader
from scipy import stats
from scipy.optimize import differential_evolution, minimize
from scipy.stats.qmc import Sobol
import time
from IPython.display import display
import json, os, time
from pathlib import Path

from collections import defaultdict
import re

# Commented out to avoid pickling issues with joblib parallel execution.
# The functions we need (load_group, load_combined_data, run_align, _debias_selected_vector)
# have been copied directly into this file above.
# from analysis_functions import *
# from dmn_ari import *


# set default plot style
import matplotlib.ticker as mticker
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['svg.fonttype'] = 'none'  # keep text as text

def set_default_plot_style(nbins_x=4, nbins_y=4, labelsize=12):
    """Set default style for all matplotlib plots, with transparent backgrounds."""

    def _apply_style(ax):
        # locators
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=nbins_x))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=nbins_y))
        # remove top/right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # label fontsize
        ax.xaxis.label.set_size(labelsize)
        ax.yaxis.label.set_size(labelsize)
        # transparent axis background
        ax.set_facecolor("none")
        return ax

    # patch plt.subplots to apply style automatically
    old_subplots = plt.subplots
    def subplots_with_style(*args, **kwargs):
        fig, axs = old_subplots(*args, **kwargs)
        fig.patch.set_alpha(0.0)   # transparent figure background
        if isinstance(axs, (list, np.ndarray)):
            for ax in axs.flat:
                _apply_style(ax)
        else:
            _apply_style(axs)
        return fig, axs
    plt.subplots = subplots_with_style

    # patch plt.subplot as well
    old_subplot = plt.subplot
    def subplot_with_style(*args, **kwargs):
        ax = old_subplot(*args, **kwargs)
        ax.figure.patch.set_alpha(0.0)
        return _apply_style(ax)
    plt.subplot = subplot_with_style

    # also patch plt.gca so ad-hoc plotting gets styled
    old_gca = plt.gca
    def gca_with_style(*args, **kwargs):
        ax = old_gca(*args, **kwargs)
        ax.figure.patch.set_alpha(0.0)
        return _apply_style(ax)
    plt.gca = gca_with_style

    # optionally set default font sizes globally
    mpl.rcParams['axes.labelsize'] = labelsize
    mpl.rcParams['xtick.labelsize'] = labelsize
    mpl.rcParams['ytick.labelsize'] = labelsize
    mpl.rcParams['figure.facecolor'] = "none"   # transparent globally
    mpl.rcParams['axes.facecolor'] = "none"

    # print(f"[matplotlib] Default style set: xticks={nbins_x}, yticks={nbins_y}, label fontsize={labelsize}")

set_default_plot_style()


one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international', silent=True)  # (mode='local')
# one = ONE(base_url='https://alyx.internationalbrainlab.org')
ba = AllenAtlas()
br = BrainRegions()

pth_res = Path(one.cache_dir, 'manifold', 'res') 
pth_res.mkdir(parents=True, exist_ok=True)
pth_dmn = Path(one.cache_dir, 'dmn', 'res')
pth_dmn.mkdir(parents=True, exist_ok=True)

save_dir = '/Users/ariliu/Desktop/ibl-figures'
# save_dir = Path(one.cache_dir, 'model')
# save_dir.mkdir(parents=True, exist_ok=True)

meta_splits={
    'duringstim': ['block_duringstim_r_choice_r_f1', 'block_duringstim_l_choice_l_f1',
                   'block_duringstim_l_choice_r_f2', 'block_duringstim_r_choice_l_f2'],
    'duringchoice': ['block_stim_r_duringchoice_r_f1', 'block_stim_l_duringchoice_l_f1',
                     'block_stim_l_duringchoice_r_f2', 'block_stim_r_duringchoice_l_f2']
}

'''# O: observations (as inputs), [Ol,Or]
S, I, P, M: stim, integrator, block, choice neurons; a: adaptation for stim neuron
alpha_i, alpha_p, alpha_m: exponential decay rate (inverse of time constant) for stim / block / choice neuron
W_pi: weights from stim integrator (I) to block prior (P); similarly for other W_** parameters
g_i/m/s, d_i/m/s: presence of prior gain effect, or prior initial offset; binary variables
    (for integrator I, choice M, & stim difference S)
'''

'''
equations:

S0 = np.matmul(V_s, O) [stim difference]
tau_s * dS/dt = -alpha_s * S + tanh(a * (v + g_s * P) * S0 + d_s * P)   [stim response]
tau_a * da/dt = -alpha_a * (a - 1) - W_as * abs(S) [stim adaptation]
tau_i * dI/dt = -alpha_i * I + tanh((W_PI + g_i * W_PM * P) * S + d_i * P)  [integrator]
tau_p * dP/dt = -alpha_p * P + tanh(W_MP * I)  [prior]
tau_m * dM/dt = -alpha_m * M + tanh((W_PM + g_m * W_MM * M) * I + d_m * P)  [choice]
action = Heavyside(|tanh(M)| - threshold) * Sign(M)

type 0 (gain on stim): d_s=0, g_s>0, d_i=0, g_i=0, d_m=0, g_M=0
type 1 (offset on stim): d_s>0, g_s=0, d_i=0, g_i=0, d_m=0, g_M=0
type 2 (gain on integrator): d_s=0, g_s=0, d_i=0, g_i>0, d_m=0, g_M=0
type 3 (offset on integrator): d_s=0, g_s=0, d_i>0, g_i=0, d_m=0, g_M=0
type 4 (gain on choice): d_s=0, g_s=0, d_i=0, g_i=0, d_m=0, g_M>0
type 5 (offset on choice): d_s=0, g_s=0, d_i=0, g_i=0, d_m>0, g_M=0

generic neuron time constant = 20ms = 10dt for simulation
'''


# global env variables
num_stimulus_strength=5
min_stimulus_strength=0.
max_stimulus_strength=1

block_side_probs=((0.8, 0.2),
                  (0.2, 0.8))
blocks_per_session=2
min_trials_per_block=20
max_trials_per_block=100
trials_per_block_param=1 / 60
# Note: dt is no longer a global variable. It must be set in model_params['dt'].
# Default dt value for initial model_params setup (can be overridden)
_DEFAULT_DT = 2.0  # ms

# Time duration constants (in milliseconds) for scaling with dt
# These can be modified if different durations are needed
MIN_TRIAL_DURATION_MS = 40  # Minimum trial length after stimulus onset
ITI_START_BEFORE_MS = 400   # ITI window start (ms before stimulus onset)
ITI_END_BEFORE_MS = 100     # ITI window end (ms before stimulus onset)
MAX_OBS_DURATION_MS = 2000  # Maximum observation duration per trial (ms)
STEPS_BEFORE_OBS_DURATION_MS = 1000  # Duration before observation starts (ms)


def _min_trial_steps(dt_value=None):
    """Calculate minimum trial steps based on dt."""
    if dt_value is None:
        dt_value = _DEFAULT_DT
    return int(MIN_TRIAL_DURATION_MS / dt_value)

def _iti_start_before_steps(dt_value=None):
    """Calculate ITI start_before steps based on dt."""
    if dt_value is None:
        dt_value = _DEFAULT_DT
    return int(ITI_START_BEFORE_MS / dt_value)

def _iti_end_before_steps(dt_value=None):
    """Calculate ITI end_before steps based on dt."""
    if dt_value is None:
        dt_value = _DEFAULT_DT
    return int(ITI_END_BEFORE_MS / dt_value)

# fixed model params
# NOTE: These are DEFAULT values computed for _DEFAULT_DT.
# When dt changes (e.g., in model_params['dt']), use _update_model_params_for_dt()
# to recompute all dt-dependent parameters. The run_model() function does this automatically.
tau = 20.0  # ms, fixed real-time constant
model_params = {
    'tau_s': tau,
    'tau_i': tau,
    'tau_p': tau,
    'tau_m': tau,
    'post_action_steps': int(40/_DEFAULT_DT),
    'stim_adap': True,
    'nonlin_type': 'linear',
    'baseline': 0.0,
    'internal_noise': [1.0, 1.0, 1.0, 1.0, 1.0],
    'prestim_offset_start': int(100/_DEFAULT_DT),
    'direct_offset': False,
    'dt': _DEFAULT_DT,  # Set default dt in model_params
}

model_params['alpha_w'] = 1.565
model_params['beta_w'] = 0.164
model_params['alpha_d'] = 35.277
model_params['beta_d'] = 2.0515
model_params['tau_a'] = 222.68  # ms, fixed real-time constant
model_params['W_as'] = 28.106
model_params['W_ss'] = 7.652e-05
model_params['g_i'] = 1.2976
model_params['g_m'] = 0.536
model_params['d_i'] = 0.102762
model_params['d_m'] = 0.00005362
model_params['g_s'] = 0
model_params['d_s'] = 0

theta = [0.91, 0.54]
model_params['action_thresholds']={
        'concordant': {
            1.0: theta[0],
            0.25: theta[0],
            0.125: theta[0],
            0.0625: theta[0],
            0: theta[0]
        },
        'discordant': {
            1.0: theta[1],
            0.25: theta[1],
            0.125: theta[1],
            0.0625: theta[1],
            0: theta[1]
        }
    }


model_params['W_mm'] = 0.316
model_params['W_ii'] = 0.451
model_params['W_pp'] = 0.496
model_params['W_is'] = 0.1209
model_params['W_mi'] = 0.521317
model_params['W_pi'] = 0.00164
# dt is already set in model_params above (from _DEFAULT_DT)


def _compute_dt_dependent_params(dt_value):
    """
    Compute all parameters that depend on dt.
    
    Returns a dictionary of parameters that should be updated in model_params
    when dt changes.
    """
    return {
        'tau_s': 20.0,
        'tau_i': 20.0,
        'tau_p': 20.0,
        'tau_m': 20.0,
        'tau_a': 222.68,
        'post_action_steps': int(40 / dt_value),
        'prestim_offset_start': int(100 / dt_value),
    }


def _get_dt_from_model_params(model_params, default_dt=None):
    """
    Get dt from model_params or return default.
    
    Args:
        model_params: Dictionary that should contain 'dt'
        default_dt: Default dt value if not in model_params. If None, uses _DEFAULT_DT.
    
    Returns:
        dt value to use
    
    Raises:
        ValueError: If dt is not in model_params and default_dt is None
    """
    if isinstance(model_params, dict) and 'dt' in model_params:
        return model_params['dt']
    if default_dt is not None:
        return default_dt
    # Fallback to default (for backward compatibility with old code)
    return _DEFAULT_DT


def _update_model_params_for_dt(model_params, dt_value):
    """
    Update model_params with dt-dependent values.
    Modifies model_params in-place.
    """
    dt_dependent = _compute_dt_dependent_params(dt_value)
    model_params.update(dt_dependent)
    model_params['dt'] = dt_value
    return model_params


def set_model_parameters(model_type, **model_params):
    d_s, d_i, d_m, g_s, g_i, g_m = 0, 0, 0, 0, 0, 0
    if model_type==1:
        d_s=model_params['d_s']
    elif model_type==0:
        g_s=model_params['g_s']
    elif model_type==2:
        g_i=model_params['g_i']
    elif model_type==3:
        d_i=model_params['d_i']
    elif model_type==4:
        g_m=model_params['g_m']
    elif model_type==5:
        d_m=model_params['d_m']
    elif model_type=='gain':
        # both gain on x & z present
        g_i, g_m=model_params['g_i'], model_params['g_m']
    elif model_type=='data':
        # possible gain and offset on x & z & s
        g_i, g_m, d_i, d_m=model_params['g_i'], model_params['g_m'], model_params['d_i'], model_params['d_m']
        g_s, d_s=model_params['g_s'], model_params['d_s']
    else:
        return 'what model type?'
        
    return d_s, d_i, d_m, g_s, g_i, g_m


def _create_block_stimuli_numpy(num_trials, noise,
                         block_side_bias_probabilities,
                         possible_trial_strengths,
                         possible_trial_strengths_probs,
                         max_steps_per_trial,
                         alpha_w,
                         beta_w,
                         retinal_w=None,
                         fcn='sigmoid',
                         rng=None):
    """
    Create block stimuli with strength-dependent noise.

    Args:
        num_trials: number of trials
        noise: list/array of noise std values, same length as possible_trial_strengths
        block_side_bias_probabilities: probability of block being Left vs Right
        possible_trial_strengths: list/array of possible stimulus strengths
        possible_trial_strengths_probs: probability distribution over strengths
        max_steps_per_trial: number of time steps per trial
    """

    if rng is None:
        rng = np.random

    # map each strength → its own noise
    strengths_arr = np.asarray(possible_trial_strengths)
    noise_arr = np.asarray(noise)
    assert strengths_arr.shape == noise_arr.shape, \
        "noise and possible_trial_strengths must match in shape"
    noise_map = {float(s): float(sigma) for s, sigma in zip(strengths_arr, noise_arr)}

    # choose which side has the signal
    signal_sides_indices = rng.choice(
        [0, 1],
        p=block_side_bias_probabilities,
        size=(num_trials, 1))
    signal_sides_indices = np.repeat(
        signal_sides_indices,
        axis=-1,
        repeats=max_steps_per_trial)

    trial_sides = 2 * signal_sides_indices - 1

    # choose trial strengths
    trial_strengths_one = rng.choice(
        possible_trial_strengths,
        p=possible_trial_strengths_probs,
        size=(num_trials, 1))
    # convert trial strengths (contrast) to perceived trial strengths
    retinal_w_vals = retinal_w if retinal_w is not None else [1, 1, 1, 1, 1]
    f_c = []
    for c in trial_strengths_one.flatten():
        f_c.append(perceived_contrast(c, alpha_w, beta_w, retinal_w=retinal_w_vals, fcn=fcn))
    perceived_trial_strengths_one = np.array(f_c).reshape(num_trials, 1)

    # hold trial strength constant for the duration
    trial_strengths = np.repeat(
        a=trial_strengths_one,
        repeats=max_steps_per_trial,
        axis=1)
    perceived_trial_strengths = np.repeat(
        a=perceived_trial_strengths_one,
        repeats=max_steps_per_trial,
        axis=1)

    # per-trial noise σ from the sampled strength
    trial_sigma = np.array([noise_map[float(s)] for s in trial_strengths_one.flatten()]) \
                    .reshape(num_trials, 1, 1)

    # sample base stimuli noise for both sides
    sampled_stimuli = rng.normal(
        loc=0.0,
        scale=trial_sigma,  # broadcasts over (steps, 2)
        size=(num_trials, max_steps_per_trial, 2))

    # signal with same σ as its strength level
    signal_sigma = np.repeat(trial_sigma.squeeze(-1), max_steps_per_trial, axis=1)
    signal = rng.normal(loc=perceived_trial_strengths, scale=signal_sigma)

    # # add signal on the chosen side
    # sampled_stimuli[np.eye(2)[signal_sides_indices].astype(bool)] = signal.flatten()
    # add signal on the chosen side (robust, vectorized advanced indexing)
    # Ensure side indices are 1D (num_trials,)
    # Safe per-channel assignment using 2D masks (num_trials x steps)
    # signal_sides_indices is already (num_trials, max_steps_per_trial)
    left_mask  = (signal_sides_indices == 0)             # (T, steps) bool
    right_mask = ~left_mask                               # (T, steps) bool

    # Assign per channel with same-shape masks
    s0 = sampled_stimuli[:, :, 0]
    s1 = sampled_stimuli[:, :, 1]
    s0[left_mask]  = signal[left_mask]
    s1[right_mask] = signal[right_mask]
    sampled_stimuli[:, :, 0] = s0
    sampled_stimuli[:, :, 1] = s1

    # quick sanity
    assert sampled_stimuli.shape == (num_trials, max_steps_per_trial, 2)
    assert signal.shape == (num_trials, max_steps_per_trial)
    assert left_mask.shape == signal.shape

    output = dict(
        stimuli=sampled_stimuli,
        stimuli_strengths=trial_strengths,
        perceived_stimuli_strengths=perceived_trial_strengths,
        trial_sides=trial_sides)

    return output


def _create_block_stimuli_torch(num_trials, noise,
                                block_side_bias_probabilities,
                                possible_trial_strengths,
                                possible_trial_strengths_probs,
                                max_steps_per_trial,
                                alpha_w,
                                beta_w,
                                *,
                                grad_options=None,
                                retinal_w=None,
                                fcn='sigmoid'):
    grad_options = grad_options or {}
    device = torch.device(grad_options.get('device', 'cpu'))
    dtype = grad_options.get('dtype', torch.float32)
    trainable = set(grad_options.get('trainable', []))

    def to_tensor(value, name=None):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=device)
            if tensor.dtype != dtype:
                tensor = tensor.to(dtype=dtype)
            if name in trainable and not tensor.requires_grad:
                tensor.requires_grad_(True)
            return tensor
        if isinstance(value, (float, int, np.floating, np.integer)):
            return torch.tensor(float(value), dtype=dtype, device=device,
                                requires_grad=(name in trainable))
        return torch.as_tensor(value, dtype=dtype, device=device)

    alpha_w_tensor = to_tensor(alpha_w, name='alpha_w')
    beta_w_tensor = to_tensor(beta_w, name='beta_w')

    if retinal_w is None:
        retinal_w_tensor = torch.ones(len(possible_trial_strengths), dtype=dtype, device=device)
    else:
        retinal_w_tensor = to_tensor(retinal_w, name='retinal_w')
        if retinal_w_tensor.numel() != len(possible_trial_strengths):
            retinal_w_tensor = torch.ones(len(possible_trial_strengths), dtype=dtype, device=device)

    possible_strengths_tensor = to_tensor(possible_trial_strengths, name='possible_trial_strengths')
    possible_probs_tensor = to_tensor(possible_trial_strengths_probs, name='possible_trial_strengths_probs')
    noise_tensor = to_tensor(noise, name='internal_noise')
    side_probs_tensor = to_tensor(block_side_bias_probabilities, name='block_side_bias_probabilities')

    categorical_strength = torch.distributions.Categorical(probs=possible_probs_tensor)
    strength_indices = categorical_strength.sample((num_trials,))
    trial_strengths_one = possible_strengths_tensor[strength_indices].unsqueeze(1)

    if alpha_w_tensor is None or beta_w_tensor is None or fcn == 'None':
        perceived_one = retinal_w_tensor[strength_indices].unsqueeze(1)
    elif fcn == 'sigmoid':
        w_lo = torch.tensor(0.0, dtype=dtype, device=device)
        w_hi = torch.tensor(2.0, dtype=dtype, device=device)
        c_vals = trial_strengths_one.squeeze(1)
        p = torch.sigmoid(alpha_w_tensor * (c_vals - beta_w_tensor))
        p1 = torch.sigmoid(-alpha_w_tensor * beta_w_tensor)
        perceived_one = (w_lo + (w_hi - w_lo) * (p - p1)).unsqueeze(1)
    elif fcn == 'power':
        c_vals = trial_strengths_one.squeeze(1)
        perceived_one = (alpha_w_tensor * torch.pow(c_vals, beta_w_tensor)).unsqueeze(1)
    else:
        raise ValueError(f"Nonlinear type {fcn} not recognized for perceived contrast.")

    trial_strengths = trial_strengths_one.repeat(1, max_steps_per_trial)
    perceived_strengths = perceived_one.repeat(1, max_steps_per_trial)

    noise_vals = noise_tensor[strength_indices].unsqueeze(1).unsqueeze(2)

    base_noise = torch.randn((num_trials, max_steps_per_trial, 2), dtype=dtype, device=device)
    sampled_stimuli = base_noise * noise_vals

    signal_sigma = noise_vals.squeeze(-1).repeat(1, max_steps_per_trial)
    epsilon = torch.randn((num_trials, max_steps_per_trial), dtype=dtype, device=device)
    signal = perceived_strengths + signal_sigma * epsilon

    categorical_side = torch.distributions.Categorical(probs=side_probs_tensor)
    signal_side_indices = categorical_side.sample((num_trials,)).unsqueeze(1)
    signal_side_indices = signal_side_indices.repeat(1, max_steps_per_trial)

    left_mask = signal_side_indices == 0
    right_mask = ~left_mask

    s0 = torch.where(left_mask, signal, sampled_stimuli[:, :, 0])
    s1 = torch.where(right_mask, signal, sampled_stimuli[:, :, 1])
    sampled_stimuli = torch.stack((s0, s1), dim=-1)

    trial_sides = signal_side_indices * 2 - 1

    output = dict(
        stimuli=sampled_stimuli,
        stimuli_strengths=trial_strengths,
        perceived_stimuli_strengths=perceived_strengths,
        trial_sides=trial_sides,
    )

    return output


def create_block_stimuli(num_trials, noise,
                         block_side_bias_probabilities,
                         possible_trial_strengths,
                         possible_trial_strengths_probs,
                         max_steps_per_trial,
                         alpha_w,
                         beta_w,
                         gradient_mode=False,
                         grad_options=None,
                         retinal_w=None,
                         fcn='sigmoid',
                         rng=None):

    grad_options = grad_options or {}
    use_torch = gradient_mode or grad_options.get('force_torch', False)

    if not use_torch:
        if torch.is_tensor(alpha_w) or torch.is_tensor(beta_w):
            use_torch = True
        if torch.is_tensor(noise) or torch.is_tensor(possible_trial_strengths) \
           or torch.is_tensor(possible_trial_strengths_probs):
            use_torch = True

    if use_torch:
        return _create_block_stimuli_torch(
            num_trials, noise, block_side_bias_probabilities,
            possible_trial_strengths, possible_trial_strengths_probs,
            max_steps_per_trial, alpha_w, beta_w,
            grad_options=grad_options,
            retinal_w=retinal_w,
            fcn=fcn)

    return _create_block_stimuli_numpy(
        num_trials, noise, block_side_bias_probabilities,
        possible_trial_strengths, possible_trial_strengths_probs,
        max_steps_per_trial, alpha_w, beta_w,
        retinal_w=retinal_w, fcn=fcn, rng=rng)


def create_stimuli(blocks_per_session, trials_per_block_param, 
                   block_side_probs, num_stimulus_strength,
                   min_stimulus_strength, max_stimulus_strength, 
                   min_trials_per_block, max_trials_per_block,
                   max_obs_per_trial=None, steps_before_obs=None,
                   gradient_mode=False, grad_options=None,
                   rng=None,
                   **model_params):
    # create stimulus for the whole session
    noise = model_params['internal_noise']
    alpha_w = model_params['alpha_w']
    beta_w = model_params['beta_w']
    retinal_w = model_params.get('retinal_w', None)
    perceived_fcn = model_params.get('perceived_contrast_fcn', 'sigmoid')

    # choose RNG
    if rng is None:
        rng = np.random

    dt_for_stim = _get_dt_from_model_params(model_params, _DEFAULT_DT)
    if max_obs_per_trial is None:
        max_obs_per_trial = int(MAX_OBS_DURATION_MS / dt_for_stim)
    if steps_before_obs is None:
        steps_before_obs = int(STEPS_BEFORE_OBS_DURATION_MS / dt_for_stim)

    grad_options = grad_options or {}
    use_torch = gradient_mode or grad_options.get('force_torch', False)
    if not use_torch:
        if torch.is_tensor(alpha_w) or torch.is_tensor(beta_w):
            use_torch = True
        if torch.is_tensor(noise):
            use_torch = True
    
    # first create num of trials for each block
    num_trials_per_block = []
    while len(num_trials_per_block) < blocks_per_session:
        sample = rng.geometric(p=trials_per_block_param)
        if min_trials_per_block <= sample <= max_trials_per_block:
            num_trials_per_block.append(sample)
                
    current_block_side = rng.choice([0, 1])  # choose first block bias with 50-50 probability
    # each of stimuli, trial_stimulus_side, block_side will have
    # the following structure:
    #       list of length blocks_per_session
    #       each list element will be a tensor with shape
    #           (trials_per_session, max_steps_per_trial, 2)
    stimuli, trial_strengths, perceived_trial_strengths = [], [], []
    trial_sides, block_sides = [], []
    if num_stimulus_strength == 5: # 5 contrast levels same as in experiments
        possible_trial_strengths = [max_stimulus_strength, max_stimulus_strength/4, 
                                    max_stimulus_strength/8, max_stimulus_strength/16, min_stimulus_strength]
        possible_trial_strengths_probs = tuple((2/(num_stimulus_strength*2-1), 2/(num_stimulus_strength*2-1), 
            2/(num_stimulus_strength*2-1), 2/(num_stimulus_strength*2-1), 1/(num_stimulus_strength*2-1)))
    else: # equally spaced contrast levels
        possible_trial_strengths = tuple(np.linspace(
            min_stimulus_strength, max_stimulus_strength, num_stimulus_strength))
        possible_trial_strengths_probs = tuple(np.ones(
            num_stimulus_strength) / num_stimulus_strength)
    max_steps_per_trial = steps_before_obs + max_obs_per_trial

    for num_trials in num_trials_per_block:
        #create stimulus for each block
        stimulus_creator_output = create_block_stimuli(
            num_trials=num_trials,
            noise=noise,
            block_side_bias_probabilities=block_side_probs[current_block_side],
            possible_trial_strengths=possible_trial_strengths,
            possible_trial_strengths_probs=possible_trial_strengths_probs,
            max_steps_per_trial=max_steps_per_trial,
            alpha_w=alpha_w,
            beta_w=beta_w,
            gradient_mode=use_torch,
            grad_options=grad_options,
            retinal_w=retinal_w,
            fcn=perceived_fcn,
            rng=rng)
        stimuli.append(stimulus_creator_output['stimuli'])
        trial_strengths.append(stimulus_creator_output['stimuli_strengths'])
        perceived_trial_strengths.append(stimulus_creator_output['perceived_stimuli_strengths'])
        trial_sides.append(stimulus_creator_output['trial_sides'])
        block_side = np.full(
            shape=(num_trials, max_steps_per_trial),
            fill_value=-1 if current_block_side == 0 else 1)
        if use_torch:
            block_sides.append(torch.as_tensor(block_side, dtype=grad_options.get('dtype', torch.float32),
                                               device=grad_options.get('device', 'cpu')))
        else:
            block_sides.append(block_side)
        current_block_side = 1 if current_block_side == 0 else 0

    #zero out stimuli for first steps_before_stimulus
    for block_stimuli in stimuli:
            block_stimuli[:, :steps_before_obs, :] = 0
            
    return stimuli, trial_strengths, perceived_trial_strengths, trial_sides, block_sides


contrast_to_index = {1.0: 0, 0.25: 1, 0.125: 2, 0.0625: 3, 0.0: 4}
def perceived_contrast(c, alpha_w, beta_w, retinal_w=[1, 1, 1, 1, 1], fcn='sigmoid'):
    """
    Parametric lookup of perceived contrast (SNR) enforcing monotonic relations
    c: contrast
    fcn: functional form, 'sigmoid', 'power', or None

    internal_noise is still indexed per-contrast.
    """
    idx = contrast_to_index.get(c, None)
    if idx is None:
        raise ValueError(f"Contrast {c} not recognized")
    c = float(c)

    if alpha_w == None or beta_w == None or fcn == 'None':
        w_val = retinal_w[idx]
    elif fcn == 'sigmoid':
        # logistic weight with fixed output range [w_lo, w_hi]
        w_lo, w_hi = 0, 2
        p = 1.0 / (1.0 + np.exp(-alpha_w * (c - beta_w)))  # a>0, b∈[0,1]
        p1 = 1.0 / (1.0 + np.exp(alpha_w * beta_w))
        w_val = w_lo + (w_hi - w_lo) * (p - p1)
    elif fcn == 'power':
        w_val = alpha_w * (c ** beta_w)

    return w_val


def _run_model_numpy(model_type, stimuli, trial_strengths, trial_sides, block_sides, 
                blocks_per_session, steps_before_obs, 
                punishment=-0.1, wait_penalty=0, only_initial=False, 
                debug=False, verbose=True,
                **model_params
                ):

    if debug:
        dbg_mdiff_sign, dbg_idiff_sign, dbg_sdiff_sign = [], [], []
        dbg_true_side, dbg_contrast, dbg_threshold = [], [], []
        dbg_margin = []
        # Our convention: Left choice label = -1 when (M0 - M1) > 0
        CHOICE_SIGN = -1  # = sign(choice) / sign(M0-M1). If you change policy, update this.
        dbg_expected_choice_sign = []
        margin_used = []           # |M0-M1| - action_threshold at first decision
        trial_threshold = []       # action_threshold actually used at first decision


    # unpack model params
    # retinal_w = model_params['retinal_w']
    # retinal_delay = model_params['retinal_delay']
    # alpha_w = model_params['alpha_w']
    # beta_w = model_params['beta_w']
    alpha_d = model_params['alpha_d']
    beta_d = model_params['beta_d']
    action_thresholds = model_params['action_thresholds']
    stim_adap = model_params['stim_adap']
    nonlin_type = model_params['nonlin_type']
    # alpha_a = model_params['alpha_a']
    W_ss = model_params['W_ss']
    W_ii = model_params['W_ii']
    W_pp = model_params['W_pp']
    W_mm = model_params['W_mm']
    tau_a = model_params['tau_a']
    tau_s = model_params['tau_s']
    tau_i = model_params['tau_i']
    tau_p = model_params['tau_p']
    tau_m = model_params['tau_m']
    W_is = model_params['W_is']
    W_mi = model_params['W_mi']
    W_pi = model_params['W_pi']
    W_as = model_params['W_as']
    # retinal_w = model_params['retinal_w']
    post_action_steps = model_params['post_action_steps']
    baseline = model_params['baseline']
    direct_offset = model_params['direct_offset']
    prestim_offset_start = model_params['prestim_offset_start']
    # print(prestim_offset_start)

    def retinal_delay(c, alpha_d, beta_d, actual_dt):
        """
        Parametric lookup enforcing monotonic relations:
        delay(c) = alpha_d / (1 + beta_d * c)   (base dt = _DEFAULT_DT)
        Returned as INTEGER steps for the current dt.
        """

        c = float(c)
        # integer delay in steps (non-negative)
        delay_cont_base = alpha_d / (1.0 + beta_d * c)
        delay_ms = delay_cont_base * _DEFAULT_DT
        delay_steps = max(0, int(round(delay_ms / actual_dt)))

        return delay_steps
    
    def nonlin(x, nonlin_type):
        if nonlin_type == 'tanh':
            return np.tanh(x)
        elif nonlin_type == 'sigmoid':
            return 1/(1 + np.exp(-x))
        elif nonlin_type == 'linear':
            return x
        else:
            raise ValueError(f"Nonlinear type {nonlin_type} not recognized")

    def _return_all_nan():
        keys = ["S","I","P","M","a","choices","reward","correct_action_taken","reaction_time",
                "trial_sides","block_sides","choice_sides","trial_strengths","perceived_stim",
                "sub_prior","action_time","action_signal"]
        return {k: np.nan for k in keys}

    dt = float(_get_dt_from_model_params(model_params))

    d_s, d_i, d_m, g_s, g_i, g_m = set_model_parameters(model_type, **model_params)
    if verbose:
        print('model', model_type, 
              'offset_s, offset_i, offset_m, gain_s, gain_i, gain_m:', 
              d_s, d_i, d_m, g_s, g_i, g_m,
              'nonlin_type:', nonlin_type)
        print('direct_offset:', direct_offset)
    J = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
    a = np.array([1.0, 1.0], dtype=float)
    S = np.full(2, float(baseline))
    S_ = np.full(2, float(baseline))
    I = np.full(2, float(baseline))
    I_ = np.full(2, float(baseline))
    P = np.full(2, float(baseline))
    M = np.full(2, float(baseline))
    M_ = np.full(2, float(baseline))
    stim_neuron, adaptation, integrator_belief, choice_belief, block_belief, reaction_time = [],[],[],[],[],[]
    perceived_stim, action_time, sub_prior = [], [], []
    action_signal = []
    choices, reward, correct_action_taken = [], [], []
    trial_sides_for_plot, block_sides_for_plot, trial_strengths_for_plot, choice_sides_for_plot = [], [], [], []
    if debug:
        S_ff_hist = []

    t = 0 # global time counter
    for i in range(blocks_per_session):
        for j in range(len(stimuli[i])):
            # loop over all trials in block
            trial_complete = 0
            k = 0
            contrast_mag = np.abs(trial_strengths[i][j][0])  # assumed fixed per trial
            delay = retinal_delay(contrast_mag, alpha_d, beta_d, dt)
            trial_rt = 0
            trial_sub_prior = []

            while trial_complete==0 and k<len(stimuli[i][j]):
                # loop over all steps within a trial

                if only_initial:
                    if k<=steps_before_obs:
                        d_s, d_i, d_m, g_s, g_i, g_m = set_model_parameters(model_type, **model_params)
                    else:
                        d_s, d_i, d_m, g_s, g_i, g_m = 0, 0, 0, 0, 0, 0

                # if k < (steps_before_obs-50):
                #     d_s, d_i, d_m = 0, 0, 0
                # else:
                #     d_s, d_i, d_m, _, _, _ = set_model_parameters(model_type, **model_params)

                S0 = stimuli[i][j][k]

                if stim_adap:
                    # Update adaptation based on previous S
                    a = a + dt/tau_a * nonlin(-(a - 1) - a * W_as * abs(S_), nonlin_type)

                # set concordant/discordant prior gain
                if (S[0]-S[1]) * (P[0]-P[1]) >= 0:
                    # positive prior gain for perceived concordant trials
                    conc=True
                    del_P = np.abs(P[0]-P[1])
                else:
                    # negative prior gain for perceived discordant trials
                    conc=False
                    del_P = -np.abs(P[0]-P[1])
                P_gain = np.array([[del_P, 0], [0, del_P]])
                if k >= (steps_before_obs-prestim_offset_start):
                    P_offset = J @ P
                else:
                    P_offset = np.array([0, 0])
                # P_offset = J @ P

                S0_delayed = perceived_stim[t - delay] if delay > 0 and k >= (delay+steps_before_obs) else np.array([0, 0]) if delay > 0 else S0
                # print(g_s, g_s * P_gain)
                if direct_offset:
                    S_ = S_ + dt/tau_s * nonlin(-S_ + W_ss * J @ S_
                                            # + d_s * P_offset
                                                + a * ((J + g_s * P_gain) @ S0_delayed),
                                                nonlin_type)
                    S = S_ + d_s * P_offset
                else:
                    S = S + dt/tau_s * nonlin(-S + W_ss * J @ S
                                                + d_s * P_offset
                                                + a * ((J + g_s * P_gain) @ S0_delayed),
                                                nonlin_type)
                    S_ = S

                if k == 0:
                    S_ff = S_.copy()  # pure feedforward tracker
                S_ff = S_ff + dt/tau_s * (-S_ff + W_ss * J @ S_ff + a * (J @ S0_delayed))

                if isinstance(action_thresholds, dict):
                    if conc:
                        # set diff action thresholds for concordant/discordant trials
                        action_threshold=action_thresholds['concordant'][contrast_mag]
                    else:
                        action_threshold=action_thresholds['discordant'][contrast_mag]
                else: # single action threshold for all trials
                    action_threshold=action_thresholds

                if direct_offset:
                    I_ = I_ + dt/tau_i * nonlin(-I_ + W_ii * J @ I_
                                                # + d_i * P_offset
                                                + (W_is * J + g_i * P_gain) @ S,
                                                nonlin_type)
                    I = I_ + d_i * P_offset
                    M_ = M_ + dt/tau_m * nonlin(-M_ + W_mm * J @ M_
                                            # + d_m * P_offset
                                            + (W_mi * J + g_m * P_gain) @ I,
                                            nonlin_type)
                    M = M_ + d_m * P_offset
                else:
                    I = I + dt/tau_i * nonlin(-I + W_ii * J @ I
                                                + d_i * P_offset
                                                + (W_is * J + g_i * P_gain) @ S,
                                                nonlin_type)
                    I_ = I
                    M = M + dt/tau_m * nonlin(-M + W_mm * J @ M
                                            + d_m * P_offset
                                            + (W_mi * J + g_m * P_gain) @ I,
                                            nonlin_type)
                    M_ = M
                # print(d_i, d_i * P_offset)
                P = P + dt/tau_p * nonlin(-P + W_pp * J @ P + W_pi * J @ I, nonlin_type)

                stim_neuron.append(S.copy())
                choice_belief.append(M.copy())
                integrator_belief.append(I.copy())
                block_belief.append(P.copy())
                adaptation.append(a.copy())
                perceived_stim.append(S0.copy())
                trial_sub_prior.append((P[0]-P[1]).copy())
                if debug:
                    S_ff_hist.append(S_ff.copy())

                action = nonlin(M[0]-M[1], 'tanh')
                action_signal.append(action)
                # Enforce post-action cutoff even if current |action| dips below threshold
                if (trial_rt > 0) and (k > (trial_rt + steps_before_obs + post_action_steps - 1)):
                    trial_complete = 1
                    
                elif k > steps_before_obs:
                    # action = M[0]-M[1]
                    if np.abs(action) >= (action_threshold+1e-6): # action taken
                        if trial_rt==0:
                            if debug:
                                m_diff = action
                                i_diff = I[0] - I[1]
                                s_diff = S[0] - S[1]
                                dbg_mdiff_sign.append(np.sign(m_diff))
                                dbg_idiff_sign.append(np.sign(i_diff))
                                dbg_sdiff_sign.append(np.sign(s_diff))
                                dbg_true_side.append(int(np.sign(trial_sides[i][j][k])))   # -1=L, +1=R
                                dbg_contrast.append(float(contrast_mag))
                                dbg_threshold.append(float(action_threshold))
                                dbg_margin.append(float(abs(m_diff) - action_threshold))
                                dbg_expected_choice_sign.append(int(CHOICE_SIGN * np.sign(m_diff)))
                                # log diagnostics once per trial
                                trial_threshold.append(float(action_threshold))
                                margin_used.append(float(abs(m_diff) - float(action_threshold)))

                            if action < 0: # right action                         
                                choices.append(1)
                                reward.append(1 if (trial_sides[i][j][k] == 1) else punishment)
                                correct_action_taken.append(1 if (trial_sides[i][j][k] == 1) else 0)
                                trial_rt = k+1-steps_before_obs
                                reaction_time.append(trial_rt)
                                action_time.append(t)                            
                            else: # left action
                                choices.append(-1)
                                reward.append(1 if (trial_sides[i][j][k] == -1) else punishment)
                                correct_action_taken.append(1 if (trial_sides[i][j][k] == -1) else 0)
                                trial_rt = k+1-steps_before_obs
                                reaction_time.append(trial_rt)
                                action_time.append(t)
                        # elif k > (trial_rt + steps_before_obs + post_action_steps - 1):
                        #     trial_complete = 1

                    else: # action not taken
                        reward.append(wait_penalty)
                        if k==(len(stimuli[i][j])-1) and trial_rt==0:
                            # time out
                            choices.append(0)
                            correct_action_taken.append(0)
                            reaction_time.append(k+1-steps_before_obs)

                k += 1
                t += 1
                
            trial_sides_for_plot.append(np.tile(trial_sides[i][j][0], k))
            block_sides_for_plot.append(np.tile(block_sides[i][j][0], k))
            choice_sides_for_plot.append(np.tile(choices[-1], k))
            trial_strengths_for_plot.append(np.tile(trial_strengths[i][j][0], k))
            sub_prior.append(np.tile(np.mean(trial_sub_prior), k))

            # break if any core state becomes non-finite
            if (not np.isfinite(S).all()) or (not np.isfinite(I).all()) \
            or (not np.isfinite(P).all()) or (not np.isfinite(M).all()):
                if verbose:
                    print('core state became non-finite')
                return _return_all_nan()

    # --- Diagnostic: trial count imbalance across conditions ---
    if debug:
        trial_data = []

        for i_side, t_side, b_side, corr in zip(
                trial_strengths_for_plot,
                trial_sides_for_plot,
                block_sides_for_plot,
                correct_action_taken):
            # each entry has shape (k,), we only need one label per trial
            trial_data.append({
                'contrast': float(np.abs(i_side[0])),
                'trial_side': int(np.sign(t_side[0])),
                'block_side': int(np.sign(b_side[0])),
                'correct': int(corr)
            })

        df_trials = pd.DataFrame(trial_data)
        df_trials['congruency'] = np.sign(df_trials['trial_side'] * df_trials['block_side'])
        summary = (
            df_trials.groupby(['contrast', 'congruency', 'correct'])
            .size()
            .unstack(fill_value=0)
        )
        print("\nTrial count summary by contrast, congruency, and correctness:")
        print(summary)

    if debug:
        return {
            "S": stim_neuron,
            "I": integrator_belief,
            "P": block_belief,
            "M": choice_belief,
            "a": adaptation,
            "choices": choices,
            "reward": reward,
            "correct_action_taken": correct_action_taken,
            "reaction_time": reaction_time,
            "trial_sides": trial_sides_for_plot,
            "block_sides": block_sides_for_plot,
            'choice_sides': choice_sides_for_plot,
            "trial_strengths": trial_strengths_for_plot,
            "perceived_stim": perceived_stim,
            'sub_prior': sub_prior,
            'action_time': action_time,
            'dbg_mdiff_sign': dbg_mdiff_sign,
            'dbg_idiff_sign': dbg_idiff_sign,
            'dbg_sdiff_sign': dbg_sdiff_sign,
            'dbg_true_side': dbg_true_side,
            'dbg_contrast': dbg_contrast,
            'dbg_threshold': dbg_threshold,
            'dbg_margin': dbg_margin,
            'trial_threshold': trial_threshold,
            'margin_used': margin_used,
            'S_ff': S_ff_hist,
            'action_signal': action_signal,
            }
    else:
        return {
            "S": stim_neuron,
            "I": integrator_belief,
            "P": block_belief,
            "M": choice_belief,
            "a": adaptation,
            "choices": choices,
            "reward": reward,
            "correct_action_taken": correct_action_taken,
            "reaction_time": reaction_time,
            "trial_sides": trial_sides_for_plot,
            "block_sides": block_sides_for_plot,
            'choice_sides': choice_sides_for_plot,
            "trial_strengths": trial_strengths_for_plot,
            "perceived_stim": perceived_stim,
            'sub_prior': sub_prior,
            'action_time': action_time,
            'action_signal': action_signal
        }


def _contains_torch(obj):
    if isinstance(obj, torch.Tensor):
        return True
    if isinstance(obj, (list, tuple)):
        return any(_contains_torch(x) for x in obj)
    if isinstance(obj, dict):
        return any(_contains_torch(v) for v in obj.values())
    return False


def _resolve_grad_options(model_params, override=None):
    """
    Extract gradient-related options with sensible defaults.
    """
    if override is not None:
        base = dict(model_params.get('grad_options', {}))
        base.update({k: v for k, v in override.items() if v is not None})
        grad_opts = base
    else:
        grad_opts = model_params.get('grad_options', {})
    dtype = grad_opts.get('dtype', torch.float32)
    device = torch.device(grad_opts.get('device', 'cpu'))
    return grad_opts, dtype, device


def _ensure_tensor(value, *, dtype, device, requires_grad=False):
    """
    Best-effort conversion to a torch.Tensor without breaking autograd when value is already a tensor.
    """
    if value is None:
        return None
    if torch.is_tensor(value):
        tensor = value.to(device=device, dtype=dtype, non_blocking=False)
        if requires_grad and not tensor.requires_grad:
            tensor.requires_grad_(True)
        return tensor
    arr = np.asarray(value, dtype=float)
    tensor = torch.tensor(arr, dtype=dtype, device=device)
    if requires_grad:
        tensor.requires_grad_(True)
    return tensor


def _tensor_nanmean(x, dim=None):
    return torch.nanmean(x, dim=dim)


def _tensor_nan(value, *, dtype, device):
    return torch.tensor(float('nan'), dtype=dtype, device=device)


def _detach_to_numpy(obj):
    """
    Recursively convert torch Tensors (possibly requiring grad) into numpy arrays or floats.
    Leaves other objects unchanged.
    """
    if torch.is_tensor(obj):
        data = obj.detach().cpu()
        if data.dim() == 0:
            return float(data.item())
        return data.numpy()
    if isinstance(obj, dict):
        return {k: _detach_to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_detach_to_numpy(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_detach_to_numpy(v) for v in obj)
    return obj


def run_model(model_type, stimuli, trial_strengths, trial_sides, block_sides,
              blocks_per_session, steps_before_obs=None,
              punishment=-0.1, wait_penalty=0, only_initial=False,
              debug=False, verbose=True,
              gradient_mode=False, grad_options=None,
              **model_params):
    """
    Run the model simulation.
    
    Args:
        steps_before_obs: Steps before observation. If None, computed from dt.
        **model_params: Model parameters. Must include 'dt' (time step in ms) or 
                       it will use the global default dt.
    
    Note:
        To change dt for a run, set model_params['dt'] = your_value before calling.
        All dt-dependent parameters (tau, post_action_steps, etc.) will be automatically
        updated based on the dt value in model_params.
    """
    # Get dt from model_params (or global default)
    # This is the single source of truth for dt
    dt = _get_dt_from_model_params(model_params)
    
    # Compute steps_before_obs if not provided
    if steps_before_obs is None:
        steps_before_obs = int(STEPS_BEFORE_OBS_DURATION_MS / dt)
    
    # Update all dt-dependent parameters in model_params
    # This ensures tau, post_action_steps, etc. are correct for the current dt
    # _update_model_params_for_dt(model_params, dt)
    
    """
    Run the brain-wide map dynamical model.

    Parameters
    ----------
    gradient_mode : bool, default False
        If True, execute the simulation using Torch tensors so that gradients
        can be propagated to differentiable model parameters. When False, the
        legacy NumPy implementation is used for maximum compatibility.
    grad_options : dict | None
        Extra options for the Torch backend. Recognized keys:
          - device: torch.device or string (default 'cpu')
          - dtype: torch.dtype (default torch.float32)
          - trainable: iterable of parameter names that should be created with
                       requires_grad=True when passed as Python scalars.
          - detach_outputs: if True, detach Torch outputs before returning and
                            convert them to NumPy arrays.
          - force_torch: if True, run Torch backend even if gradient_mode=False.
          - threshold_temperature: positive float controlling softness of the
                                   action threshold (default 0.01). Set ≤0 to
                                   recover the hard threshold.
    """
    grad_options = grad_options or {}
    use_torch = gradient_mode or grad_options.get('force_torch', False)

    if not use_torch:
        if _contains_torch(stimuli) or _contains_torch(trial_strengths) \
           or _contains_torch(trial_sides) or _contains_torch(block_sides) \
           or _contains_torch(model_params):
            use_torch = True

    if use_torch:
        return _run_model_torch(
            model_type, stimuli, trial_strengths, trial_sides, block_sides,
            blocks_per_session, steps_before_obs,
            punishment=punishment, wait_penalty=wait_penalty,
            only_initial=only_initial, debug=debug,
            grad_options=grad_options, **model_params)

    return _run_model_numpy(
        model_type, stimuli, trial_strengths, trial_sides, block_sides,
        blocks_per_session, steps_before_obs,
        punishment=punishment, wait_penalty=wait_penalty,
        only_initial=only_initial, debug=debug, verbose=verbose, **model_params)


def _run_model_torch(model_type, stimuli, trial_strengths, trial_sides, block_sides,
                     blocks_per_session, steps_before_obs,
                     punishment=-0.1, wait_penalty=0, only_initial=False,
                     debug=False, grad_options=None,
                     **model_params):
    """
    Torch implementation of `run_model` that keeps the computation graph so
    gradients can flow back to differentiable parameters.
    """
    _update_model_params_for_dt(model_params, dt)

    if debug:
        raise NotImplementedError("gradient_mode currently does not support debug=True.")

    grad_options = grad_options or {}
    dt = float(_get_dt_from_model_params(model_params))
    device = torch.device(grad_options.get('device', 'cpu'))
    dtype = grad_options.get('dtype', torch.float32)
    trainable = set(grad_options.get('trainable', []))
    detach_outputs = grad_options.get('detach_outputs', False)
    threshold_temperature = float(grad_options.get('threshold_temperature', 0.01))
    use_soft_threshold = threshold_temperature > 0.0
    if use_soft_threshold:
        threshold_temperature_tensor = torch.tensor(
            max(threshold_temperature, 1e-6), dtype=dtype, device=device)
    else:
        threshold_temperature_tensor = None

    def to_tensor(value, name=None):
        if isinstance(value, torch.Tensor):
            t = value.to(device=device)
            if dtype is not None and t.dtype != dtype:
                t = t.to(dtype=dtype)
            return t
        if isinstance(value, (list, tuple)):
            return [to_tensor(v, name=name) for v in value]
        if isinstance(value, np.ndarray):
            return torch.as_tensor(value, dtype=dtype, device=device)
        if isinstance(value, (float, int, np.floating, np.integer)):
            requires_grad = name in trainable
            return torch.tensor(float(value), dtype=dtype, device=device,
                                requires_grad=requires_grad)
        return value

    def ensure_tensor_like(structure, names=None):
        if isinstance(structure, (list, tuple)):
            return [ensure_tensor_like(v, names=names) for v in structure]
        return to_tensor(structure, name=names)

    # unpack model params (convert scalars to tensors where needed)
    alpha_d = to_tensor(model_params['alpha_d'], name='alpha_d')
    beta_d = to_tensor(model_params['beta_d'], name='beta_d')
    action_thresholds = model_params['action_thresholds']
    stim_adap = model_params['stim_adap']
    nonlin_type = model_params['nonlin_type']
    W_ss = to_tensor(model_params['W_ss'], name='W_ss')
    W_ii = to_tensor(model_params['W_ii'], name='W_ii')
    W_pp = to_tensor(model_params['W_pp'], name='W_pp')
    W_mm = to_tensor(model_params['W_mm'], name='W_mm')
    tau_a = to_tensor(model_params['tau_a'], name='tau_a')
    tau_s = to_tensor(model_params['tau_s'], name='tau_s')
    tau_i = to_tensor(model_params['tau_i'], name='tau_i')
    tau_p = to_tensor(model_params['tau_p'], name='tau_p')
    tau_m = to_tensor(model_params['tau_m'], name='tau_m')
    W_is = to_tensor(model_params['W_is'], name='W_is')
    W_mi = to_tensor(model_params['W_mi'], name='W_mi')
    W_pi = to_tensor(model_params['W_pi'], name='W_pi')
    W_as = to_tensor(model_params['W_as'], name='W_as')
    post_action_steps = int(model_params['post_action_steps'])
    baseline = float(model_params['baseline'])
    direct_offset = bool(model_params.get('direct_offset', False))
    prestim_offset_start = int(model_params.get('prestim_offset_start', 0))

    dt_tensor = torch.tensor(float(dt), dtype=dtype, device=device)

    def nonlin(x, mode):
        if mode == 'tanh':
            return torch.tanh(x)
        if mode == 'sigmoid':
            return torch.sigmoid(x)
        if mode == 'linear':
            return x
        raise ValueError(f"Nonlinear type {mode} not recognized")

    one_tensor = torch.tensor(1.0, dtype=dtype, device=device)
    zero_tensor = torch.tensor(0.0, dtype=dtype, device=device)
    eps_tensor = torch.tensor(1e-6, dtype=dtype, device=device)

    def compute_delay_steps(contrast_value):
        contrast_tensor = torch.as_tensor(contrast_value, dtype=dtype, device=device)
        denom = one_tensor + beta_d * contrast_tensor
        denom = torch.clamp(denom, min=eps_tensor)
        delay_base = alpha_d / denom
        delay_ms = delay_base * _DEFAULT_DT
        delay_steps = delay_ms / dt_tensor
        return torch.clamp(delay_steps, min=eps_tensor)

    d_s, d_i, d_m, g_s, g_i, g_m = set_model_parameters(model_type, **model_params)
    d_s = to_tensor(d_s, name='d_s')
    d_i = to_tensor(d_i, name='d_i')
    d_m = to_tensor(d_m, name='d_m')
    g_s = to_tensor(g_s, name='g_s')
    g_i = to_tensor(g_i, name='g_i')
    g_m = to_tensor(g_m, name='g_m')

    J = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=dtype, device=device)
    a = torch.ones(2, dtype=dtype, device=device)
    S = torch.full((2,), baseline, dtype=dtype, device=device)
    S_latent = S.clone()
    I = torch.full((2,), baseline, dtype=dtype, device=device)
    I_latent = I.clone()
    P = torch.full((2,), baseline, dtype=dtype, device=device)
    M = torch.full((2,), baseline, dtype=dtype, device=device)
    M_latent = M.clone()

    stimuli_t = ensure_tensor_like(stimuli)
    trial_strengths_t = ensure_tensor_like(trial_strengths)
    trial_sides_t = ensure_tensor_like(trial_sides)
    block_sides_t = ensure_tensor_like(block_sides)

    if isinstance(action_thresholds, dict):
        contrast_levels_tensor = None  # no tensor lookup; handled per trial
    else:
        contrast_levels_tensor = None

    stim_neuron = []
    adaptation = []
    integrator_belief = []
    choice_belief = []
    block_belief = []
    reaction_time = []
    perceived_stim = []
    action_time = []
    sub_prior = []
    choices = []
    reward = []
    correct_action_taken = []
    trial_sides_for_plot = []
    block_sides_for_plot = []
    choice_sides_for_plot = []
    trial_strengths_for_plot = []
    action_signal = []

    t_global = 0

    def sample_retinal_input(current_signal, delay_steps, step_index, trial_history):
        # Fractional delay with linear interpolation using Python lists (cheaper than
        # maintaining tensor buffers for every trial).
        if step_index < steps_before_obs:
            return torch.zeros_like(current_signal)

        history_len = len(trial_history)
        if history_len == 0:
            return torch.zeros_like(current_signal)

        history_tensor = torch.stack(trial_history, dim=0)
        max_valid = history_len - 1
        max_valid_f = float(max_valid)

        t_cur = torch.tensor(float(history_len), dtype=dtype, device=device)
        idx_cont = torch.clamp(t_cur - delay_steps, 0.0, max_valid_f)

        idx0 = torch.floor(idx_cont)
        idx1 = torch.clamp(idx0 + 1.0, max=max_valid_f)
        w1 = idx_cont - idx0
        w0 = 1.0 - w1

        idx0_int = int(idx0.detach().clamp(0.0, max_valid_f).cpu().item())
        idx1_int = int(idx1.detach().clamp(0.0, max_valid_f).cpu().item())
        v0 = history_tensor[idx0_int]
        v1 = history_tensor[idx1_int]
        out = w0 * v0 + w1 * v1

        return torch.where((delay_steps <= eps_tensor).expand_as(out), current_signal, out)
        
    for i_block in range(blocks_per_session):
        block_stimuli = stimuli_t[i_block]
        block_trial_strengths = trial_strengths_t[i_block]
        block_trial_sides = trial_sides_t[i_block]
        block_block_sides = block_sides_t[i_block]

        num_trials = len(block_stimuli)
        for j_trial in range(num_trials):
            # print('j_trial', j_trial)
            stim_trial = block_stimuli[j_trial]
            trial_strength = block_trial_strengths[j_trial]
            trial_side = block_trial_sides[j_trial]
            block_side = block_block_sides[j_trial]
            perceived_stim_trial = []

            trial_complete = False
            step_idx = 0
            trial_rt = 0
            trial_sub_prior = []
            continue_prob = torch.tensor(1.0, dtype=dtype, device=device)

            contrast_tensor = torch.as_tensor(trial_strength[0], dtype=dtype, device=device)
            contrast_abs = torch.abs(contrast_tensor)
            delay_steps = compute_delay_steps(contrast_abs)

            if isinstance(action_thresholds, dict):
                contrast_val = float(contrast_abs.detach().cpu().item())
                theta_conc_tensor = to_tensor(action_thresholds['concordant'][contrast_val])
                theta_disc_tensor = to_tensor(action_thresholds['discordant'][contrast_val])
            else:
                theta_conc_tensor = to_tensor(action_thresholds)
                theta_disc_tensor = theta_conc_tensor

            while (not trial_complete) and (step_idx < stim_trial.shape[0]):
                # print('step_idx', step_idx)
                if only_initial:
                    if step_idx <= steps_before_obs:
                        d_s, d_i, d_m, g_s, g_i, g_m = set_model_parameters(model_type, **model_params)
                        d_s = to_tensor(d_s, name='d_s')
                        d_i = to_tensor(d_i, name='d_i')
                        d_m = to_tensor(d_m, name='d_m')
                        g_s = to_tensor(g_s, name='g_s')
                        g_i = to_tensor(g_i, name='g_i')
                        g_m = to_tensor(g_m, name='g_m')
                    else:
                        d_s = torch.zeros_like(d_s)
                        d_i = torch.zeros_like(d_i)
                        d_m = torch.zeros_like(d_m)
                        g_s = torch.zeros_like(g_s)
                        g_i = torch.zeros_like(g_i)
                        g_m = torch.zeros_like(g_m)

                S0 = stim_trial[step_idx]
                cont = continue_prob

                if stim_adap:
                    delta_a = dt_tensor / tau_a * nonlin(
                        -(a - 1.0) - a * W_as * torch.abs(S_latent),
                        nonlin_type
                    )
                    a = a + cont * delta_a

                p_diff = P[0] - P[1]
                s_diff = S[0] - S[1]
                same_sign = torch.sign(p_diff * s_diff + 1e-12)
                del_P = torch.abs(p_diff)
                del_P_signed = torch.where(same_sign >= 0, del_P, -del_P)
                P_gain = torch.stack([
                    torch.stack([del_P_signed, torch.tensor(0.0, dtype=dtype, device=device)]),
                    torch.stack([torch.tensor(0.0, dtype=dtype, device=device), del_P_signed])
                ])

                conc_indicator = (p_diff * s_diff) >= 0

                if step_idx >= (steps_before_obs - prestim_offset_start):
                    P_offset = J @ P
                else:
                    P_offset = torch.zeros(2, dtype=dtype, device=device)

                S0_delayed = sample_retinal_input(S0, delay_steps, step_idx, perceived_stim_trial)

                action_threshold_tensor = torch.where(conc_indicator, theta_conc_tensor, theta_disc_tensor)

                if direct_offset:
                    delta_S_latent = dt_tensor / tau_s * nonlin(
                        -S_latent + W_ss * (J @ S_latent)
                        + a * ((J + g_s * P_gain) @ S0_delayed),
                        nonlin_type
                    )
                    S_latent = S_latent + cont * delta_S_latent
                    S = S_latent + d_s * P_offset
                else:
                    delta_S = dt_tensor / tau_s * nonlin(
                        -S + W_ss * (J @ S)
                        + d_s * P_offset
                        + a * ((J + g_s * P_gain) @ S0_delayed),
                        nonlin_type
                    )
                    S = S + cont * delta_S
                    S_latent = S

                if direct_offset:
                    delta_I_latent = dt_tensor / tau_i * nonlin(
                        -I_latent + W_ii * (J @ I_latent)
                        + (W_is * J + g_i * P_gain) @ S,
                        nonlin_type
                    )
                    I_latent = I_latent + cont * delta_I_latent
                    I = I_latent + d_i * P_offset
                    delta_M_latent = dt_tensor / tau_m * nonlin(
                        -M_latent + W_mm * (J @ M_latent)
                        + (W_mi * J + g_m * P_gain) @ I,
                        nonlin_type
                    )
                    M_latent = M_latent + cont * delta_M_latent
                    M = M_latent + d_m * P_offset
                else:
                    delta_I = dt_tensor / tau_i * nonlin(
                        -I + W_ii * (J @ I)
                        + d_i * P_offset
                        + (W_is * J + g_i * P_gain) @ S,
                        nonlin_type
                    )
                    I = I + cont * delta_I
                    I_latent = I
                    delta_M = dt_tensor / tau_m * nonlin(
                        -M + W_mm * (J @ M)
                        + d_m * P_offset
                        + (W_mi * J + g_m * P_gain) @ I,
                        nonlin_type
                    )
                    M = M + cont * delta_M
                    M_latent = M

                delta_P = dt_tensor / tau_p * nonlin(
                    -P + W_pp * (J @ P) + W_pi * (J @ I),
                    nonlin_type
                )
                P = P + cont * delta_P

                stim_neuron.append(S.clone())
                choice_belief.append(M.clone())
                integrator_belief.append(I.clone())
                block_belief.append(P.clone())
                adaptation.append(a.clone())
                history_entry = S0.clone()
                perceived_stim.append(history_entry)
                perceived_stim_trial.append(history_entry)
                trial_sub_prior.append((P[0] - P[1]).clone())

                action_value = nonlin(M[0] - M[1], 'tanh')
                action_signal.append(action_value.clone())

                continue_prob_next = cont

                if (trial_rt > 0) and (step_idx > (trial_rt + steps_before_obs + post_action_steps - 1)):
                    trial_complete = True

                elif step_idx > steps_before_obs:
                    action_mag_tensor = torch.abs(action_value)
                    if use_soft_threshold:
                        gate_soft = torch.sigmoid(
                            (action_mag_tensor - action_threshold_tensor) / threshold_temperature_tensor)
                    else:
                        gate_soft = (action_mag_tensor >= action_threshold_tensor).to(dtype)
                    gate_soft = torch.clamp(gate_soft, 0.0, 1.0)
                    continue_prob_next = cont * (one_tensor - gate_soft)

                    if (action_mag_tensor >= (action_threshold_tensor + eps_tensor)).item():
                        if trial_rt == 0:
                            action_dir = float(action_value.detach().cpu().item())
                            choice = -1 if action_dir >= 0 else 1
                            choices.append(choice)

                            true_side = float(trial_side[step_idx].detach().cpu().item())
                            reward_val = 1.0 if (choice == 1 and true_side == 1.0) or (choice == -1 and true_side == -1.0) else punishment
                            reward.append(reward_val)
                            correct_action_taken.append(1 if reward_val == 1.0 else 0)
                            trial_rt = step_idx + 1 - steps_before_obs
                            reaction_time.append(torch.tensor(trial_rt, dtype=dtype, device=device))
                            action_time.append(torch.tensor(t_global, dtype=dtype, device=device))
                        # allow continued accumulation until post_action window expires
                    else:
                        reward.append(wait_penalty)
                        if (step_idx == (stim_trial.shape[0] - 1)) and (trial_rt == 0):
                            choices.append(0)
                            correct_action_taken.append(0)
                            reaction_time.append(torch.tensor(step_idx + 1 - steps_before_obs, dtype=dtype, device=device))

                else:
                    reward.append(wait_penalty)
                    if (step_idx == (stim_trial.shape[0] - 1)) and (trial_rt == 0):
                        choices.append(0)
                        correct_action_taken.append(0)
                        reaction_time.append(torch.tensor(step_idx + 1 - steps_before_obs, dtype=dtype, device=device))

                continue_prob = torch.clamp(continue_prob_next, 0.0, 1.0)

                step_idx += 1
                t_global += 1

                if (not torch.isfinite(S).all()) or (not torch.isfinite(I).all()) \
                        or (not torch.isfinite(P).all()) or (not torch.isfinite(M).all()):
                    return _return_all_nan_torch(dtype=dtype, device=device,
                                                 detach_outputs=detach_outputs)

            # per-trial replicated signals for plotting compatibility
            if step_idx == 0:
                continue

            trial_sides_for_plot.append(torch.full(
                (step_idx,), float(trial_side[0].detach().cpu().item()),
                dtype=dtype, device=device))
            block_sides_for_plot.append(torch.full(
                (step_idx,), float(block_side[0].detach().cpu().item()),
                dtype=dtype, device=device))
            if choices:
                last_choice = float(choices[-1])
            else:
                last_choice = 0.0
            choice_sides_for_plot.append(torch.full(
                (step_idx,), last_choice, dtype=dtype, device=device))
            trial_strengths_for_plot.append(torch.full(
                (step_idx,), float(trial_strength[0].detach().cpu().item()),
                dtype=dtype, device=device))
            if trial_sub_prior:
                mean_prior = torch.stack(trial_sub_prior).mean()
            else:
                mean_prior = torch.tensor(0.0, dtype=dtype, device=device)
            sub_prior.append(torch.ones((step_idx,), dtype=dtype, device=device) * mean_prior)

    def maybe_detach(item):
        if isinstance(item, torch.Tensor):
            return item.detach().cpu().numpy() if detach_outputs else item
        if isinstance(item, list):
            return [maybe_detach(v) for v in item]
        return item

    results = {
        "S": torch.stack(stim_neuron) if stim_neuron else torch.empty((0, 2), dtype=dtype, device=device),
        "I": torch.stack(integrator_belief) if integrator_belief else torch.empty((0, 2), dtype=dtype, device=device),
        "P": torch.stack(block_belief) if block_belief else torch.empty((0, 2), dtype=dtype, device=device),
        "M": torch.stack(choice_belief) if choice_belief else torch.empty((0, 2), dtype=dtype, device=device),
        "a": torch.stack(adaptation) if adaptation else torch.empty((0, 2), dtype=dtype, device=device),
        "choices": choices,
        "reward": reward,
        "correct_action_taken": correct_action_taken,
        "reaction_time": torch.stack(reaction_time) if reaction_time else torch.empty((0,), dtype=dtype, device=device),
        "trial_sides": [maybe_detach(tensor) for tensor in trial_sides_for_plot],
        "block_sides": [maybe_detach(tensor) for tensor in block_sides_for_plot],
        'choice_sides': [maybe_detach(tensor) for tensor in choice_sides_for_plot],
        "trial_strengths": [maybe_detach(tensor) for tensor in trial_strengths_for_plot],
        "perceived_stim": torch.stack(perceived_stim) if perceived_stim else torch.empty((0, 2), dtype=dtype, device=device),
        'sub_prior': [maybe_detach(tensor) for tensor in sub_prior],
        'action_time': torch.stack(action_time) if action_time else torch.empty((0,), dtype=dtype, device=device),
        'action_signal': torch.stack(action_signal) if action_signal else torch.empty((0,), dtype=dtype, device=device),
    }

    if detach_outputs:
        for key in ("S", "I", "P", "M", "a", "perceived_stim", "reaction_time", "action_time", "action_signal"):
            if isinstance(results[key], torch.Tensor):
                results[key] = results[key].detach().cpu().numpy()

    return results


def _return_all_nan_torch(*, dtype=torch.float32, device='cpu', detach_outputs=False):
    nan_pair = torch.full((1, 2), float('nan'), dtype=dtype, device=device)
    nan_scalar = torch.full((1,), float('nan'), dtype=dtype, device=device)
    result = {
        "S": nan_pair.clone(),
        "I": nan_pair.clone(),
        "P": nan_pair.clone(),
        "M": nan_pair.clone(),
        "a": nan_pair.clone(),
        "choices": np.nan,
        "reward": np.nan,
        "correct_action_taken": np.nan,
        "reaction_time": nan_scalar.clone(),
        "trial_sides": np.nan,
        "block_sides": np.nan,
        'choice_sides': np.nan,
        "trial_strengths": np.nan,
        "perceived_stim": nan_pair.clone(),
        'sub_prior': np.nan,
        'action_time': nan_scalar.clone(),
        'action_signal': nan_scalar.clone(),
    }
    if detach_outputs:
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.detach().cpu().numpy()
    return result


def combine_run_results(run_results_list, *, reindex_action_time=True, skip_invalid=True):
    """
    Combine outputs from multiple `run_model` calls into a single 'continuous' run.

    Works if you pass:
      - a list/tuple of run dicts, OR
      - a dict mapping run_id → run dict (e.g. results_all).

    Args:
        run_results_list : dict | list | tuple
            Collection of run_model outputs.
        reindex_action_time : bool
            If True, shift action_time of later runs by cumulative step counts.
        skip_invalid : bool
            If True, skip any run that came from `_return_all_nan`.

    Returns:
        dict with the same schema as `run_model`.
    """
    # Normalize input
    if isinstance(run_results_list, (list, tuple)):
        runs = list(run_results_list)
    elif isinstance(run_results_list, dict):
        # Use sorted keys if possible, else insertion order
        try:
            runs = [run_results_list[k] for k in sorted(run_results_list.keys())]
        except Exception:
            runs = list(run_results_list.values())
    else:
        raise TypeError("run_results_list must be dict, list, or tuple of run dicts.")

    # Expected schema from run_model
    step_keys = ["S", "I", "P", "M", "a", "perceived_stim"]
    trial_scalar_keys = ["choices", "reward", "correct_action_taken", "reaction_time"]
    trial_array_keys = ["trial_sides", "block_sides", "choice_sides", "trial_strengths", "sub_prior"]
    all_keys = step_keys + trial_scalar_keys + trial_array_keys + ["action_time"]

    out = {k: [] for k in all_keys}

    def _is_nan_run(run):
        # `_return_all_nan()` returns np.nan for all keys
        for k in all_keys:
            v = run.get(k, None)
            if isinstance(v, float) and np.isnan(v):
                return True
        return False

    def _to_list(x):
        if x is None:
            return []
        if isinstance(x, list):
            return x
        if isinstance(x, np.ndarray):
            return x.tolist()
        # Scalars should normally not appear, but wrap safely
        return [x] if not (isinstance(x, float) and np.isnan(x)) else []

    cum_steps = 0

    for run in runs:
        if not isinstance(run, dict):
            continue
        if skip_invalid and _is_nan_run(run):
            continue

        # Per-step streams
        for k in step_keys:
            out[k].extend(_to_list(run.get(k, [])))

        # Per-trial scalar series
        for k in trial_scalar_keys:
            out[k].extend(_to_list(run.get(k, [])))

        # Per-trial arrays (each trial has its own per-step array)
        for k in trial_array_keys:
            out[k].extend(_to_list(run.get(k, [])))

        # action_time
        at = _to_list(run.get("action_time", []))
        if reindex_action_time and at:
            out["action_time"].extend([int(a) + int(cum_steps) for a in at])
        else:
            out["action_time"].extend(at)

        # Step count proxy = length of perceived_stim
        steps_this_run = len(_to_list(run.get("perceived_stim", [])))
        cum_steps += steps_this_run

    return out


def plot_perf(results, metric="correct", dt=2):
    """
    Plot performance by signed trial strength, split by congruency.
    
    metric: "correct" → percent correct
            "rt"      → reaction time (raw units, or seconds if dt given)
    """

    # extract
    trial_strengths = results['trial_strengths']
    trial_sides = results['trial_sides']
    block_sides = results['block_sides']
    values = results['correct_action_taken'] if metric == "correct" else results['reaction_time']

    # per-trial values
    strength_per_trial = [ts[0] for ts in trial_strengths]
    side_per_trial = [ts[0] for ts in trial_sides]
    block_per_trial = [bs[0] for bs in block_sides]

    # signed strengths
    strength_signed = [s * side for s, side in zip(strength_per_trial, side_per_trial)]

    # congruency labels
    is_congruent = [int(side == block) for side, block in zip(side_per_trial, block_per_trial)]
    groups = {'congruent': [], 'incongruent': []}
    for i, s in enumerate(strength_signed):
        key = 'congruent' if is_congruent[i] else 'incongruent'
        groups[key].append((s, values[i]))

    plt.figure(figsize=(6,4))
    for key, color in zip(groups.keys(), ['C0', 'C1']):
        signed = [x[0] for x in groups[key]]
        vals   = [x[1] for x in groups[key]]
        unique_strengths = sorted(set(signed))
        mean_vals = []
        for s in unique_strengths:
            idx = [i for i, st in enumerate(signed) if st == s]
            v = [vals[i] for i in idx]
            if metric == "correct":
                mean_vals.append(100 * np.mean(v))
            else:  # reaction time
                mean_vals.append(np.mean(v) * dt)
        # line plot with averages
        plt.plot(unique_strengths, mean_vals, '-o', label=key, color=color)
        # scatter raw trial values
        for s in unique_strengths:
            idx = [i for i, st in enumerate(signed) if st == s]
            v = [vals[i] for i in idx]
            # if metric == "correct":
            #     plt.scatter([s]*len(idx), [100*x for x in v],
            #                 alpha=0.3, color=color, s=10)
            # else:
            #     plt.scatter([s]*len(idx), [x*dt for x in v],
            #                 alpha=0.3, color=color, s=10)

    plt.xlabel("Signed trial strength")
    if metric == "correct":
        plt.ylabel("% correct")
        # plt.axhline(50, color='k', linestyle='--', linewidth=1)
        # plt.title("Performance by congruency (block vs trial side)")
    else:
        plt.ylabel("Reaction time (s)" if dt != 1.0 else "Reaction time (steps)")
        # plt.title("Reaction time by congruency (block vs trial side)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_overall_traj(results, model_type, start=0, length=20000,
                      steps_before_obs=None, shade_color='0.9', shade_alpha=0.3):
    fig = plt.figure(figsize=(10, 2), dpi=100)
    ax = plt.gca()

    # Resolve shading width
    if steps_before_obs is None:
        steps_before_obs = globals().get('steps_before_obs', None)

    # Extract windowed series
    stim_neuron = np.array(results['S'][start:start+length])
    integrator_belief = np.array(results['I'][start:start+length])
    choice_belief = np.array(results['M'][start:start+length])
    block_belief = np.array(results['P'][start:start+length])
    trial_sides_for_plot = np.concatenate(results['trial_sides'])[start:start+length]
    trial_strengths_for_plot = np.concatenate(results['trial_strengths'])[start:start+length]
    trial_side_weighted = trial_sides_for_plot * trial_strengths_for_plot + 0.05
    block_sides_for_plot = np.concatenate(results['block_sides'])[start:start+length]

    # Mark trial observation-onset (end of previous shaded box) with blue dashed lines
    if steps_before_obs is not None and steps_before_obs > 0:
        # Compute absolute trial start indices by cumulative lengths
        trial_lengths = [len(ts) for ts in results['trial_sides']]
        trial_starts_abs = np.cumsum([0] + trial_lengths[:-1])
        # Draw only those within our [start, start+length) window
        win_lo, win_hi = start, start + length
        for ts in trial_starts_abs:
            t_edge = ts + steps_before_obs
            if t_edge >= win_hi:
                if ts >= win_hi:
                    break
            if t_edge < win_lo:
                continue
            x = t_edge - start
            ax.axvline(x=x, color='#4A90E2', linestyle='--', alpha=0.8, linewidth=0.8, zorder=1)

    # Plot signals
    ax.plot(stim_neuron[:, 1], linewidth=0.5, label='stim_neuronR', color='#4A90E2', alpha=1)
    ax.plot(integrator_belief[:, 1], linewidth=0.5, color='#ffc400', label='int_beliefR')
    ax.plot(choice_belief[:, 1], linewidth=0.5, color='red', label='choice_beliefR')
    ax.plot(block_belief[:, 1]*1e1, linewidth=0.5, color='#8b2be2', label='block_beliefR')
    ax.plot(stim_neuron[:, 0], linewidth=0.5, label='stim_neuronL', color='#4A90E2', alpha=0.5)
    ax.plot(integrator_belief[:, 0], linewidth=0.5, color='#ffc400', label='int_beliefL', alpha=0.5)
    ax.plot(choice_belief[:, 0], linewidth=0.5, color='red', label='choice_beliefL', alpha=0.5)
    ax.plot(block_belief[:, 0]*1e1, linewidth=0.5, color='#8b2be2', label='block_beliefL', alpha=0.5)

    # Plot trial_side_weighted only between blue (obs onset) and black (action) lines
    # Build a boolean mask over the window [start, start+length)
    masked_trial_side = np.zeros_like(trial_side_weighted)

    # Absolute trial starts/ends
    trial_lengths = [len(ts) for ts in results['trial_sides']]
    trial_starts_abs = np.cumsum([0] + trial_lengths[:-1])
    trial_ends_abs = np.cumsum(trial_lengths)

    # Action times as a sorted 1D numpy array of absolute indices
    action_times = np.array(results.get('action_time', []), dtype=float)
    action_times = action_times[np.isfinite(action_times)] if action_times.size else np.array([], dtype=float)

    win_lo, win_hi = start, start + length

    # For each trial, find the first action within that trial; if none, skip
    for ts, te in zip(trial_starts_abs, trial_ends_abs):
        # observation onset (blue) for this trial
        t_obs = ts + (steps_before_obs or 0)
        if t_obs >= te:  # guard (obs after trial end)
            continue

        # locate action within [ts, te)
        if action_times.size:
            # pick the earliest action inside this trial window
            in_trial = action_times[(action_times >= ts) & (action_times < te)]
            if in_trial.size == 0:
                continue
            t_act = float(in_trial[0])
        else:
            continue

        # intersect with our plotting window
        lo = max(int(t_obs - start), 0)
        hi = min(int(t_act - start), length)
        if hi > lo:
            masked_trial_side[lo:hi] = trial_side_weighted[lo:hi]

    ax.plot(masked_trial_side, linewidth=0.5, color='black', label='true_trialside')
    
    ax.plot(block_sides_for_plot, linewidth=2, color='silver', label='true_block', alpha=0.7)

    # Mark action times (dashed verticals)
    for t_act in results.get('action_time', []):
        if start <= t_act < start + length:
            ax.axvline(x=t_act - start, color='black', linestyle='--', alpha=0.7, linewidth=0.8)

    # Dedup legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # ax.legend(by_label.values(), by_label.keys(), fontsize=4)

    plt.savefig(f'{save_dir}/type{model_type}.pdf', transparent=True)
    plt.close()


# ----- get real data for fitting ----- 
stim_regs = ['VISpm', 'VISam', 'FRP', 'VISp', 'VISli', 'LGd', 'LP', 'NOT']
int_regs = ['SSp-ul',
 'ICB',
 'NPC',
 'RSPagl',
 'MDRN',
 'EPd',
 'CP',
 'CLI',
 'SIM',
 'PRM',
 'SUV',
 'SNr',
 'SNc',
 'PRP',
 'LDT',
 'PRNc',
 'FOTU',
 'SAG',
 'ACB',
 'TRN',
 'NTS',
 'GPe',
 'PYR',
 'VM',
 'SSp-bfd',
 'PAG',
 'VeCB',
 'PCG',
 'PC5',
 'SPF',
 'SPIV',
 'AIv',
 'OP',
 'SUT',
 'NOD',
 'SSp-un',
 'PGRN',
 'APN',
 'PB',
 'BMA',
 'P5',
 'MV',
 'LA',
 'LSc',
 'LHA',
 'IC',
 'VCO',
 'PO',
 'MOs',
 'NOT',
 'SOC',
 'NLL',
 'FRP',
 'ANcr2',
 'COPY',
 'CA3',
 'POST',
 'MOp',
 'SSs',
 'SCs',
 'DTN',
 'AId',
 'ILA',
 'DEC',
 'DCO',
 'LING',
 'Eth',
 'LAV',
 'SMT',
 'PRNr',
 'UVU',
 'BLA',
 'DN',
 'PPN',
 'AIp',
 'MD',
 'CENT3',
 'PAR',
 'PF',
 'PL',
 'POL']
move_regs = ['IRN',
 'RN',
 'CUN',
 'VAL',
 'VII',
 'MRN',
 'CENT2',
 'MARN',
 'RT',
 'COAp',
 'PARN',
 'CUL4 5',
 'ZI',
 'RPF',
 'LRN',
 'IP',
 'VPM',
 'PoT',
 'GPi',
 'VPL',
 'GRN',
 'SCm',
 'LPO',
 'V',
 'AUDd',
 'FN']
prior_regs_int = ['VISa', 'AIp', 'SSp-n', 'LSr', 'CLA']



# ----- functions for stim response analysis & fitting ----- 

def mean_S_by_contrast(results, steps_before_obs, T=65, round_contrast=None):
    """
    Average S (2-dim) over EXACTLY T steps after trial start for each (side, contrast) bucket.

    Rules:
      1) Include a trial IFF its length >= steps_before_obs + _min_trial_steps().
      2) If a valid trial has < T post-start steps, fill the remainder from the
         *next trial's pre-start* segment (indices [0:steps_before_obs) of next trial)
         to reach exactly T. If next trial unavailable or insufficient to reach T, skip.
      3) If >50% of all trials are skipped, return NaNs (2, T) for all observed buckets.
      4) If any bucket has <10 valid trials, return NaNs (2, T) for that bucket.

    Returns:
      dict: { (side, contrast): np.ndarray shape (2, T) or all-NaNs (2, T) }
    """
    S = np.asarray(results['S'], dtype=float)    # flattened over trials: (sum m_i, 2)
    # Check for NaN values in S - if any found, return NaN for all buckets
    if np.isnan(S).any():
        return np.full((2, T), np.nan)
    trial_strengths = results['trial_strengths'] # list of arrays
    trial_sides     = results['trial_sides']     # list of arrays
    n = len(trial_strengths)

    lens    = [len(trial_strengths[i]) for i in range(n)]
    offsets = np.cumsum([0] + lens[:-1])
    if S.shape[0] != sum(lens):
        print("[mean_S_by_contrast] S rows:", S.shape[0], "sum(lens):", sum(lens))

    def ckey(v):
        v = abs(float(v))
        return round(v, round_contrast) if round_contrast is not None else v

    side_per_trial     = [int(np.sign(trial_sides[i][0])) or 1 for i in range(n)]  # treat 0 as +1
    contrast_per_trial = [ckey(trial_strengths[i][0]) for i in range(n)]

    seen_keys = set()
    buckets   = defaultdict(list)   # key -> list of (T,2) segments
    counts    = defaultdict(int)
    skipped_count = 0

    for i in range(n):
        m_i = lens[i]
        key = (side_per_trial[i], contrast_per_trial[i])
        seen_keys.add(key)

        # Hard inclusion requirement
        if m_i < steps_before_obs + _min_trial_steps():
            skipped_count += 1
            continue

        # Post-start part from current trial
        post_avail = max(0, m_i - steps_before_obs)
        take_post  = min(T, post_avail)
        start_i    = offsets[i] + steps_before_obs
        seg_parts  = []
        if take_post > 0:
            seg_parts.append(S[start_i:start_i + take_post, :])

        # Need to fill from NEXT trial's pre-start?
        if take_post < T:
            if i + 1 >= n:
                skipped_count += 1
                continue
            m_next = lens[i+1]
            pre_avail_next = min(steps_before_obs, m_next)  # safe cap
            need = T - take_post
            if pre_avail_next < need:
                # even though design says "long enough", keep robust behavior
                skipped_count += 1
                continue
            start_next = offsets[i+1]  # beginning of next trial
            seg_parts.append(S[start_next:start_next + need, :])

        seg = np.vstack(seg_parts) if seg_parts else None
        if seg is None or seg.shape[0] != T:
            skipped_count += 1
            continue

        buckets[key].append(seg)
        counts[key] += 1

    # Majority skipped → NaNs for all observed buckets
    if n > 0 and skipped_count > n / 2:
        print(f"More than 50% of trials skipped")
        return {k: np.full((2, T), np.nan) for k in seen_keys}

    # Aggregate per bucket; enforce >=10 valid trials else NaNs
    out = {}
    for k in seen_keys:
        if counts.get(k, 0) < 10:
            out[k] = np.full((2, T), np.nan)
            print(f"Less than 10 valid trials for bucket {k}")
        else:
            segs = buckets.get(k, [])
            if len(segs) == 0:
                out[k] = np.full((2, T), np.nan)
                print(f"No valid trials for bucket {k}")
            else:
                mean_T2 = np.mean(np.stack(segs, axis=0), axis=0)  # (T,2)
                out[k] = mean_T2.T                                  # (2,T)

    return out



def plot_S_diff_by_contrast_side(avg_dict, dt=2):
    """
    avg_dict: output of mean_S_by_contrast_and_side_varlen_vec2
              { (side, contrast): array (2, T_eff) }

    Plots channel0 − channel1 over time, separate plot for each trial side.
    """
    for side in [-1, +1]:
        plt.figure(figsize=(7,4))
        contrasts = sorted({c for (s,c) in avg_dict.keys() if s == side})
        for c in contrasts:
            arr = avg_dict.get((side, c))
            if arr is None or arr.size == 0:
                continue
            diff_t = arr[1] - arr[0]       # (T_eff,)
            T_eff = diff_t.shape[0]
            t = np.arange(T_eff) * dt
            plt.plot(t, diff_t, '-o', markersize=3, linewidth=1, label=f"contrast={c}")
        plt.axhline(0, color='k', linestyle='--', linewidth=1)
        plt.xlabel(f"time ({'ms' if dt!=1.0 else 'steps'})")
        plt.ylabel("S[0] − S[1]")
        plt.title(f"S difference by contrast (trial side {side})")
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_S_diff_by_contrast_side_with_data(avg_dict, avg_data_L, avg_data_R, baseline=0, dt=2, 
                                           save_dir=save_dir, ylim=None, yticks=None):
    """
    avg_dict: output of mean_S_by_contrast_and_side_varlen_vec2
              { (side, contrast): array (2, T_eff) }
    avg_data_L: dict { contrast: array (2, T) } to overlay on side = -1
    avg_data_R: dict { contrast: array (2, T) } to overlay on side = +1

    Model and ref curves share the same color per contrast.
    """
    for side, ref_dict, title_side in [
        # (-1, avg_data_L, "Left (-1)"),
        (+1, avg_data_R, "Right (+1)")]:
        plt.figure(figsize=(4,3.2))
        contrasts = sorted({c for (s,c) in avg_dict.keys() if s == side})
        colors = cm.Blues(np.linspace(0.4, 1.0, len(contrasts)))[::-1]  # assign different shades of blue per contrast

        for c, color in zip(contrasts, colors):
            arr = avg_dict.get((side, c))
            if arr is None or arr.size == 0:
                continue

            # model diff
            model_diff = arr[0] - arr[1] if side == -1 else arr[1] - arr[0]
            T_model = model_diff.shape[0]

            if c in ref_dict and ref_dict[c].size > 0:
                ref_diff = ref_dict[c]
                # ref_diff = ref[0] - ref[1]
                ref_diff = (ref_diff - baseline)
                T_ref = ref_diff.shape[0]
                T_aligned = min(T_model, T_ref)
                if T_aligned <= 0:
                    continue
                model_diff = model_diff[:T_aligned]
                ref_diff   = ref_diff[:T_aligned]
                t = np.arange(T_aligned) * dt
                plt.plot(t, model_diff, '--', color=color, markersize=3, linewidth=2,
                         label=f"{c} (model)")
                plt.plot(t, ref_diff, '-', color=color, linewidth=2,
                         label=f"{c} (data)")
                if ylim is not None:
                    plt.ylim(ylim)
                if yticks is not None:
                    plt.yticks(yticks)
            else:
                t = np.arange(T_model) * dt
                plt.plot(t, model_diff, '-o', color=color, markersize=3, linewidth=1,
                         label=f"{c} (model)")

        plt.axhline(0, color='k', linestyle='--', linewidth=1)
        plt.xlabel(f"time ({'ms' if dt!=1.0 else 'steps'})")
        # plt.ylabel("S[0] − S[1]") if side == -1 else plt.ylabel("S[1] − S[0]")
        plt.title(f"S difference by contrast (trial side {title_side})")
        # plt.legend(fontsize=8, frameon=False)
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(f'{save_dir}/model_vs_data_stim_{title_side}.svg', transparent=True)


def compute_sse_stim_right(avg_dict, avg_data_R, baseline_R=0,
                           snr_weight=1.0, snr_window=11, snr_poly=3, snr_mode='db'):
    """
    SSE+SNR loss between model (+1 trial side) and data (avg_data_R) per contrast.
    - Normalizes SSE by data energy to make it amplitude-invariant.
    - Normalizes SNR difference by data SNR magnitude for balance.
    - For contrast == 1.0, only first 45 steps are used (else first 60).
    Also computes goodness-of-fit R² per contrast and overall.

    Returns
    -------
    dict
        {
          'per_contrast': {
              contrast: {
                  'T': int,
                  'sse': float,
                  'snr_model': float,
                  'snr_data': float,
                  'snr_loss': float,
                  'loss': float,
                  'gof_r2': float
              }
          },
          'total_sse': float,
          'total_snr_loss': float,
          'total_loss': float,
          'total_T': int,
          'total_gof_r2': float
        }
    """
    def _to_diff_curve(arr):
        if arr is None:
            return None
        a = np.asarray(arr)
        if a.size == 0:
            return None
        if a.ndim == 1:
            return a.astype(float)
        if a.ndim == 2:
            if a.shape[0] == 2:
                return (a[1] - a[0]).astype(float)
            if a.shape[1] == 2:
                return (a[:, 1] - a[:, 0]).astype(float)
        return None

    def _get_by_contrast(dct, key):
        if not isinstance(dct, dict):
            return dct
        if key in dct:
            return dct[key]
        for k in dct.keys():
            try:
                if float(k) == float(key):
                    return dct[k]
            except Exception:
                pass
        return None

    def _snr_val(y):
        """Compute SNR of curve y via Savitzky–Golay smoothing."""
        y = np.asarray(y, float)
        T = y.size
        if T < 3:
            return np.nan
        w = min(snr_window, T if T % 2 == 1 else T - 1)
        if w < 3:
            w = 3
        if w <= snr_poly:
            w = snr_poly + 1 if (snr_poly + 1) % 2 == 1 else snr_poly + 2
            if w > T:
                w = T if T % 2 == 1 else T - 1
        try:
            smooth = savgol_filter(y, window_length=w, polyorder=snr_poly, mode='interp')
        except Exception:
            k = max(3, min(T - (1 - T % 2), 5))
            smooth = np.convolve(y, np.ones(k) / k, mode='same')
        resid = y - smooth
        eps = 1e-12
        signal_var = float(np.nanvar(smooth))
        noise_var  = float(np.nanvar(resid))
        ratio = (signal_var + eps) / (noise_var + eps)
        if snr_mode == 'db':
            return 10.0 * np.log10(ratio)
        return ratio

    if any(np.isnan(v).any() if hasattr(v, 'any') else np.isnan(v) for v in avg_dict.values()):
        return {
            'per_contrast': np.nan,
            'total_sse': np.nan,
            'total_snr_loss': np.nan,
            'total_loss': np.nan,
            'total_T': np.nan,
            'total_gof_r2': np.nan
        }

    contrasts = sorted({c for (side, c) in avg_dict.keys() if side == 1})
    per = {}
    total_sse, total_snr_loss, total_loss, total_T = 0.0, 0.0, 0.0, 0
    total_sse_raw, total_sst = 0.0, 0.0
    eps = 1e-12

    for c in contrasts:
        model_arr = avg_dict.get((1, c), None)
        data_arr  = _get_by_contrast(avg_data_R, c)
        m = _to_diff_curve(model_arr)
        d = _to_diff_curve(data_arr)
        if m is None or d is None or m.size == 0 or d.size == 0:
            continue

        d = d - baseline_R
        T = min(m.size, d.size)
        if T <= 0:
            continue
        T = min(T, 45 if abs(float(c) - 1.0) < 1e-9 else 60)

        m_seg = m[:T]
        d_seg = d[:T]

        # --- Normalized SSE ---
        err = m_seg - d_seg
        denom_e = float(np.sum(d_seg * d_seg) + eps)
        sse = float(np.sum(err * err) / denom_e)
        sse = sse * c * 2  # scale by contrast

        # --- Balanced SNR loss ---
        snr_m = _snr_val(m_seg)
        snr_d = _snr_val(d_seg)
        if np.isfinite(snr_m) and np.isfinite(snr_d):
            diff = snr_m - snr_d
            if snr_mode == 'db':
                denom_s = (abs(snr_d) + eps) ** 2
            else:
                denom_s = (snr_d + eps) ** 2
            snr_loss = float((diff ** 2) / denom_s)
        else:
            snr_loss = 0.0

        loss = sse + snr_weight * snr_loss

        # --- Goodness of fit R² ---
        sse_raw = float(np.sum(err * err))
        sst = float(np.sum((d_seg - float(np.mean(d_seg)))**2) + eps)
        gof_r2 = 1.0 - (sse_raw / sst)

        total_sse_raw += sse_raw
        total_sst     += sst

        per[c] = {
            'T': int(T),
            'sse': sse,
            'snr_model': float(snr_m),
            'snr_data': float(snr_d),
            'snr_loss': snr_loss,
            'loss': loss,
            'gof_r2': float(gof_r2)
        }

        total_sse      += sse
        total_snr_loss += snr_loss
        total_loss     += loss
        total_T        += T

    total_gof_r2 = float(1.0 - (total_sse_raw / (total_sst + eps))) if total_sst > 0 else np.nan

    return {
        'per_contrast': per,
        'total_sse': float(total_sse),
        'total_snr_loss': float(total_snr_loss),
        'total_loss': float(total_loss),
        'total_T': int(total_T),
        'total_gof_r2': float(total_gof_r2)
    }


# ----- weight fitting for I, M, P units ----- 


def _mean_by_condition_torch(results, steps_before_obs, T=72, var_names=("S","I","P","M"),
                             grad_options=None):
    grad_opts, dtype, device = _resolve_grad_options(
        {}, override=grad_options or {})

    choices = list(results['choices'])
    trial_sides_raw = results['trial_sides']
    sub_prior_raw = results['sub_prior']
    block_sides_raw = results['block_sides']
    reaction_time_raw = results.get('reaction_time', None)

    trial_sides_np = [
        np.asarray(ts.detach().cpu()) if torch.is_tensor(ts) else np.asarray(ts)
        for ts in trial_sides_raw
    ]
    sub_prior_np = [
        np.asarray(sp.detach().cpu()) if torch.is_tensor(sp) else np.asarray(sp)
        for sp in sub_prior_raw
    ]
    block_sides_np = [
        np.asarray(bs.detach().cpu()) if torch.is_tensor(bs) else np.asarray(bs)
        for bs in block_sides_raw
    ]

    reaction_time_list = None
    if reaction_time_raw is not None:
        reaction_time_list = [
            float(rt.detach().cpu()) if torch.is_tensor(rt) else float(rt)
            for rt in reaction_time_raw
        ]

    n = len(choices)
    lens = [int(ts.shape[0]) for ts in trial_sides_np]
    offsets = [0]
    for ln in lens[:-1]:
        offsets.append(offsets[-1] + ln)

    hard_need = steps_before_obs + _min_trial_steps()
    fail_cnt = sum(1 for m in lens if m < hard_need)

    def _nan_tensor(shape):
        return torch.full(shape, float('nan'), dtype=dtype, device=device)

    if n > 0 and fail_cnt > n / 2:
        out = {vn: {} for vn in var_names}
        for vn in var_names:
            if vn == "P":
                for sp in (-1, 1):
                    out[vn][sp] = _nan_tensor((2, 150))
            else:
                for ch in (-1, 1):
                    if vn in ("I", "M"):
                        out[vn][('post', ('ch', ch))] = _nan_tensor((2, T))
                        out[vn][('pre', ('ch', ch))] = _nan_tensor((2, T))
                    else:
                        out[vn][('ch', ch)] = _nan_tensor((2, T))
                if vn in ("I", "M"):
                    for prev_ch in (-1, 1):
                        out[vn][('iti_prev', ('prev_ch', prev_ch))] = _nan_tensor((2, 150))
        return out

    var_tensors = {}
    for vn in var_names:
        raw = results[vn]
        if torch.is_tensor(raw):
            tensor = raw.to(device=device, dtype=dtype)
        else:
            tensor = torch.tensor(np.asarray(raw, dtype=float), dtype=dtype, device=device)
        if tensor.ndim != 2 or tensor.shape[1] != 2:
            raise ValueError(f"{vn}: expected (TotalSteps,2), got {tensor.shape}")
        var_tensors[vn] = tensor.contiguous()

    def _sp_sign(value):
        return 1 if value < 0 else -1

    def _collect_post_start(i, var_tensor, apply_fill, target_T):
        m_i = lens[i]
        if apply_fill:
            post_avail = max(0, m_i - steps_before_obs)
            take_post = min(target_T, post_avail)
            parts = []
            if take_post > 0:
                start = offsets[i] + steps_before_obs
                parts.append(var_tensor[start:start + take_post, :])
            if take_post < target_T:
                if i + 1 >= n:
                    return None, None
                m_next = lens[i + 1]
                if m_next < hard_need:
                    return None, None
                pre_avail_next = min(steps_before_obs, m_next)
                need = target_T - take_post
                if pre_avail_next < need:
                    return None, None
                start_next = offsets[i + 1]
                parts.append(var_tensor[start_next:start_next + need, :])
            if not parts:
                return None, None
            seg = torch.cat(parts, dim=0)
            if seg.shape[0] != target_T:
                return None, None
            return seg, target_T
        else:
            if steps_before_obs >= m_i:
                return None, None
            start = offsets[i] + steps_before_obs
            this_T = min(target_T, m_i - steps_before_obs)
            seg = var_tensor[start:start + this_T, :]
            return seg, this_T

    def _avg_segments(indices, var_tensor, *, mode="post_start",
                      apply_fill=False, target_T=T):
        segs = []
        T_eff = None
        count = 0
        for idx in indices:
            m_i = lens[idx]
            if m_i < hard_need:
                continue
            if mode == "pre_action":
                if reaction_time_list is None:
                    continue
                act_start = steps_before_obs + int(reaction_time_list[idx])
                if act_start < target_T or act_start > m_i:
                    continue
                start = offsets[idx] + act_start - target_T
                seg = var_tensor[start:start + target_T, :]
                this_T = target_T
            else:
                seg, this_T = _collect_post_start(idx, var_tensor, apply_fill, target_T)
                if seg is None:
                    continue
            if seg.numel() == 0:
                continue
            segs.append(seg)
            T_eff = this_T if T_eff is None else min(T_eff, this_T)
            count += 1
        if not segs or T_eff is None:
            return torch.empty((2, 0), dtype=dtype, device=device), 0
        segs = [seg[:T_eff, :] for seg in segs]
        stacked = torch.stack(segs, dim=0)
        mean_T2 = torch.mean(stacked, dim=0)
        return mean_T2.transpose(0, 1), count

    def avg_by_ch_torch(var_tensor, ch_sign, apply_fill=False):
        indices = [i for i in range(n) if choices[i] == ch_sign]
        return _avg_segments(indices, var_tensor, mode="post_start",
                             apply_fill=apply_fill, target_T=T)

    def avg_by_ch_pre_torch(var_tensor, ch_sign):
        indices = [i for i in range(n) if choices[i] == ch_sign]
        return _avg_segments(indices, var_tensor, mode="pre_action",
                             apply_fill=False, target_T=T)

    def avg_for_P_torch(var_tensor, sp_sign, pre_T=150):
        indices = []
        for i in range(n):
            block_sign = int(np.sign(block_sides_np[i][0]))
            if block_sign == sp_sign:
                indices.append(i)
        segs = []
        T_eff = None
        count = 0
        if steps_before_obs < pre_T:
            return torch.empty((2, 0), dtype=dtype, device=device), 0
        for idx in indices:
            m_i = lens[idx]
            if m_i < hard_need:
                continue
            start = offsets[idx] + steps_before_obs - pre_T
            if start < offsets[idx]:
                continue
            seg = var_tensor[start:start + pre_T, :]
            if seg.shape[0] != pre_T:
                continue
            segs.append(seg)
            T_eff = pre_T if T_eff is None else min(T_eff, pre_T)
            count += 1
        if not segs or T_eff is None:
            return torch.empty((2, 0), dtype=dtype, device=device), 0
        stacked = torch.stack([seg[:T_eff, :] for seg in segs], dim=0)
        mean_T2 = torch.mean(stacked, dim=0)
        return mean_T2.transpose(0, 1), count

    def avg_intertrial_by_prev_ch_torch(var_tensor, prev_ch_sign,
                                        start_before=None, end_before=None):
        if start_before is None:
            start_before = _iti_start_before_steps()
        if end_before is None:
            end_before = _iti_end_before_steps()
        if steps_before_obs < start_before or steps_before_obs <= end_before:
            return torch.empty((2, 0), dtype=dtype, device=device), 0
        length = int(start_before - end_before)
        segs = []
        count = 0
        for i in range(1, n):
            prev_choice = int(choices[i - 1])
            if prev_choice == 0 or prev_choice != prev_ch_sign:
                continue
            m_i = lens[i]
            if m_i < hard_need:
                continue
            start = offsets[i] + steps_before_obs - start_before
            stop = offsets[i] + steps_before_obs - end_before
            if start < offsets[i] or stop > offsets[i] + steps_before_obs:
                continue
            seg = var_tensor[start:stop, :]
            if seg.shape[0] != length:
                continue
            segs.append(seg)
            count += 1
        if not segs:
            return torch.empty((2, 0), dtype=dtype, device=device), 0
        stacked = torch.stack(segs, dim=0)
        mean_T2 = torch.mean(stacked, dim=0)
        return mean_T2.transpose(0, 1), count

    def _nan_if_few_t(arr, cnt):
        if cnt < 10 and arr.numel() != 0:
            return torch.full_like(arr, float('nan'))
        return arr

    out = {vn: {} for vn in var_names}

    for vn in var_names:
        tensor = var_tensors[vn]
        if vn == "P":
            for sp in (-1, 1):
                arr, cnt = avg_for_P_torch(tensor, sp, pre_T=150)
                out[vn][sp] = _nan_if_few_t(arr, cnt)
            continue

        both_windows = (vn in ("I", "M"))

        for ch in (-1, 1):
            if both_windows:
                arr_post, cnt_post = avg_by_ch_torch(tensor, ch, apply_fill=True)
                arr_pre, cnt_pre = avg_by_ch_pre_torch(tensor, ch)
                out[vn][('post', ('ch', ch))] = _nan_if_few_t(arr_post, cnt_post)
                out[vn][('pre', ('ch', ch))] = _nan_if_few_t(arr_pre, cnt_pre)
            else:
                arr, cnt = avg_by_ch_torch(tensor, ch, apply_fill=True)
                out[vn][('ch', ch)] = _nan_if_few_t(arr, cnt)

        if both_windows:
            for prev_ch in (-1, 1):
                arr_iti, cnt_iti = avg_intertrial_by_prev_ch_torch(
                    tensor, prev_ch)
                out[vn][('iti_prev', ('prev_ch', prev_ch))] = _nan_if_few_t(arr_iti, cnt_iti)

    return out

def mean_by_condition(results, steps_before_obs, T=72, var_names=("S","I","P","M"),
                      gradient_mode=False, grad_options=None):
    """
    For each var in var_names (S, I, P, M), compute mean traces by condition.

    Rules:
      • Hard trial-length requirement: m_i >= steps_before_obs + _min_trial_steps() for ALL variables.
        Trials shorter than this are skipped.
      • If >50% of trials fail this rule, return all NaNs for every variable/bucket.
      • For S and I (post-start windows):
          - If the post-start segment < T, fill the remainder from the next trial's pre-start
            segment [0:steps_before_obs). If still < T, skip that trial.
      • For P: average over the 150 steps before trial start, grouped only by true block side.
      • For M: average over the T steps before action start, grouped by (ts, ch, sp).
      • If any condition/bucket has <10 valid trials, return NaNs for that bucket.

    Extended:
      • Also include (choice, prior) buckets → stored as keys ('chsp', ch, sp).
      • Also include (choice only) buckets → stored as keys ('ch', ch).
      • Also include (trial side only) buckets → stored as keys ('ts', ts), for S and I only.
      • Also include (prior only) buckets → stored as keys ('sp', sp), for S, I, M.
      • Also include intertrial (−200 to −50) previous-choice buckets for I and M.
    """

    if gradient_mode:
        return _mean_by_condition_torch(
            results, steps_before_obs, T=T, var_names=var_names,
            grad_options=grad_options or {})

    choices       = results['choices']
    trial_sides   = results['trial_sides']
    sub_prior     = results['sub_prior']
    block_sides   = results['block_sides']  # true block side (±1)
    reaction_time = results.get('reaction_time', None)
    n             = len(choices)

    lens    = [len(trial_sides[i]) for i in range(n)]
    offsets = np.cumsum([0] + lens[:-1])

    def _sp_sign(x):  # analog prior to ±1
        return 1 if x < 0 else -1

    # --- Global check ---
    hard_need = steps_before_obs + _min_trial_steps()
    fail_cnt = sum(1 for m in lens if m < hard_need)
    if n > 0 and fail_cnt > n/2:
        out = {vn: {} for vn in var_names}
        for vn in var_names:
            if vn == "P":
                for sp in [-1, 1]:
                    out[vn][sp] = np.full((2, 150), np.nan)
            else:
                for ts, ch, sp in product([-1, 1], repeat=3):
                    out[vn][(ts, ch, sp)] = np.full((2, T), np.nan)
                for ch, sp in product([-1, 1], [-1, 1]):
                    out[vn][('chsp', ch, sp)] = np.full((2, T), np.nan)
                for ch in [-1, 1]:
                    out[vn][('ch', ch)] = np.full((2, T), np.nan)
                if vn in ("S", "I", "S_ff"):
                    for ts in [-1, 1]:
                        out[vn][('ts', ts)] = np.full((2, T), np.nan)
                if vn in ("S", "I", "M"):
                    for sp in [-1, 1]:
                        out[vn][('sp', sp)] = np.full((2, T), np.nan)
        return out

    # --- helpers ---
    def avg_for_var(var_list, ts_sign, ch_sign, sp_sign, *, mode="post_start", pre_T=150, apply_fill=False):
        var_array = np.asarray(var_list, dtype=float)
        if var_array.ndim != 2 or var_array.shape[1] != 2:
            raise ValueError(f"Expected var entries to be 2D vectors; got shape {var_array.shape}")

        sel = []
        for i in range(n):
            ts0 = int(np.sign(trial_sides[i][0]))
            sp0 = _sp_sign(sub_prior[i][0])
            if (choices[i] == ch_sign) and (ts0 == ts_sign) and (sp0 == sp_sign):
                sel.append(i)

        segs, T_eff, n_valid = [], None, 0
        for i in sel:
            m_i = lens[i]
            if m_i < hard_need:
                continue

            if mode == "pre_action":  # M
                if reaction_time is None:
                    raise ValueError("reaction_time must be provided")
                act_start = steps_before_obs + int(reaction_time[i])
                if act_start < T or act_start > m_i:
                    continue
                start, this_T = offsets[i] + act_start - T, T
                seg = var_array[start:start+this_T, :]

            else:  # post_start
                if apply_fill:
                    post_avail = max(0, m_i - steps_before_obs)
                    take_post  = min(T, post_avail)
                    parts = []
                    if take_post > 0:
                        start = offsets[i] + steps_before_obs
                        parts.append(var_array[start:start+take_post, :])
                    if take_post < T:
                        if i + 1 >= n:
                            continue
                        m_next = lens[i+1]
                        if m_next < hard_need:
                            continue
                        pre_avail_next = min(steps_before_obs, m_next)
                        need = T - take_post
                        if pre_avail_next < need:
                            continue
                        start_next = offsets[i+1]
                        parts.append(var_array[start_next:start_next + need, :])
                    seg = np.vstack(parts) if parts else None
                    if seg is None or seg.shape[0] != T:
                        continue
                    this_T = T
                else:
                    if steps_before_obs >= m_i:
                        continue
                    start, this_T = offsets[i] + steps_before_obs, min(T, m_i - steps_before_obs)
                    seg = var_array[start:start+this_T, :]

            if seg is not None and seg.size:
                segs.append(seg)
                T_eff = this_T if T_eff is None else min(T_eff, this_T)
                n_valid += 1

        if not segs or T_eff is None:
            return np.empty((2, 0)), 0
        segs = [seg[:T_eff, :] for seg in segs]
        mean_T2 = np.mean(np.stack(segs, axis=0), axis=0)
        return mean_T2.T, n_valid

    def avg_for_P(var_list, sp_sign, *, pre_T=150):
        """Average P over the 150 steps before trial start, grouped by TRUE block side (±1)."""
        var_array = np.asarray(var_list, dtype=float)

        def _block_sign(i):
            return int(np.sign(block_sides[i][0]))

        sel = [i for i in range(n) if _block_sign(i) == sp_sign]

        segs, T_eff, n_valid = [], None, 0
        for i in sel:
            m_i = lens[i]
            if m_i < hard_need or steps_before_obs < pre_T:
                continue
            start, this_T = offsets[i] + steps_before_obs - pre_T, pre_T
            seg = var_array[start:start+this_T, :]
            if seg.size:
                segs.append(seg)
                T_eff = this_T if T_eff is None else min(T_eff, this_T)
                n_valid += 1
        if not segs or T_eff is None:
            return np.empty((2, 0)), 0
        segs = [seg[:T_eff, :] for seg in segs]
        mean_T2 = np.mean(np.stack(segs, axis=0), axis=0)
        return mean_T2.T, n_valid

    def avg_by_ch_sp(var_list, ch_sign, sp_sign, *, mode="post_start", pre_T=150, apply_fill=False):
        sel = [i for i in range(n) if (choices[i] == ch_sign) and (_sp_sign(sub_prior[i][0]) == sp_sign)]
        segs, T_eff, n_valid = [], None, 0
        var_array = np.asarray(var_list, dtype=float)
        for i in sel:
            m_i = lens[i]
            if m_i < hard_need:
                continue
            if mode == "pre_action":
                if reaction_time is None:
                    continue
                act_start = steps_before_obs + int(reaction_time[i])
                if act_start < T or act_start > m_i:
                    continue
                start, this_T = offsets[i] + act_start - T, T
                seg = var_array[start:start+this_T, :]
            else:
                if apply_fill:
                    post_avail = max(0, m_i - steps_before_obs)
                    take_post  = min(T, post_avail)
                    parts = []
                    if take_post > 0:
                        start = offsets[i] + steps_before_obs
                        parts.append(var_array[start:start+take_post, :])
                    if take_post < T:
                        if i + 1 >= n:
                            continue
                        m_next = lens[i+1]
                        if m_next < hard_need:
                            continue
                        pre_avail_next = min(steps_before_obs, m_next)
                        need = T - take_post
                        if pre_avail_next < need:
                            continue
                        start_next = offsets[i+1]
                        parts.append(var_array[start_next:start_next + need, :])
                    seg = np.vstack(parts) if parts else None
                    if seg is None or seg.shape[0] != T:
                        continue
                    this_T = T
                else:
                    if steps_before_obs >= m_i:
                        continue
                    start, this_T = offsets[i] + steps_before_obs, min(T, m_i - steps_before_obs)
                    seg = var_array[start:start+this_T, :]
            if seg is not None and seg.size:
                segs.append(seg)
                T_eff = this_T if T_eff is None else min(T_eff, this_T)
                n_valid += 1
        if not segs or T_eff is None:
            return np.empty((2, 0)), 0
        segs = [seg[:T_eff, :] for seg in segs]
        mean_T2 = np.mean(np.stack(segs, axis=0), axis=0)
        return mean_T2.T, n_valid

    def avg_by_ch(var_list, ch_sign, *, mode="post_start", pre_T=150, apply_fill=False):
        sel = [i for i in range(n) if choices[i] == ch_sign]
        segs, T_eff, n_valid = [], None, 0
        var_array = np.asarray(var_list, dtype=float)
        for i in sel:
            m_i = lens[i]
            if m_i < hard_need:
                continue
            if mode == "pre_action":
                if reaction_time is None:
                    continue
                act_start = steps_before_obs + int(reaction_time[i])
                if act_start < T or act_start > m_i:
                    continue
                start, this_T = offsets[i] + act_start - T, T
                seg = var_array[start:start+this_T, :]
            else:
                if apply_fill:
                    post_avail = max(0, m_i - steps_before_obs)
                    take_post  = min(T, post_avail)
                    parts = []
                    if take_post > 0:
                        start = offsets[i] + steps_before_obs
                        parts.append(var_array[start:start+take_post, :])
                    if take_post < T:
                        if i + 1 >= n:
                            continue
                        m_next = lens[i+1]
                        if m_next < hard_need:
                            continue
                        pre_avail_next = min(steps_before_obs, m_next)
                        need = T - take_post
                        if pre_avail_next < need:
                            continue
                        start_next = offsets[i+1]
                        parts.append(var_array[start_next:start_next + need, :])
                    seg = np.vstack(parts) if parts else None
                    if seg is None or seg.shape[0] != T:
                        continue
                    this_T = T
                else:
                    if steps_before_obs >= m_i:
                        continue
                    start, this_T = offsets[i] + steps_before_obs, min(T, m_i - steps_before_obs)
                    seg = var_array[start:start+this_T, :]
            if seg is not None and seg.size:
                segs.append(seg)
                T_eff = this_T if T_eff is None else min(T_eff, this_T)
                n_valid += 1
        if not segs or T_eff is None:
            return np.empty((2, 0)), 0
        segs = [seg[:T_eff, :] for seg in segs]
        mean_T2 = np.mean(np.stack(segs, axis=0), axis=0)
        return mean_T2.T, n_valid

    def avg_intertrial_by_prev_ch(var_list, prev_ch_sign, start_before=None, end_before=None):
        """
        Average over the intertrial window, defined as [−start_before, −end_before)
        relative to stimulus onset (trial start), grouping by the previous trial's choice.
        """
        if start_before is None:
            start_before = _iti_start_before_steps()
        if end_before is None:
            end_before = _iti_end_before_steps()
        if steps_before_obs < start_before or steps_before_obs <= end_before:
            return np.empty((2, 0)), 0

        var_array = np.asarray(var_list, dtype=float)
        if var_array.ndim != 2 or var_array.shape[1] != 2:
            raise ValueError(f"Expected var entries to be 2D vectors; got shape {var_array.shape}")

        length = int(start_before - end_before)
        segs, n_valid = [], 0

        for i in range(1, n):  # skip first trial (no previous choice)
            prev_choice = int(choices[i-1])
            if prev_choice == 0:
                continue
            if prev_choice != prev_ch_sign:
                continue

            m_i = lens[i]
            if m_i < hard_need:
                continue

            start = offsets[i] + steps_before_obs - start_before
            stop  = offsets[i] + steps_before_obs - end_before
            if start < offsets[i] or stop > offsets[i] + steps_before_obs:
                continue

            seg = var_array[start:stop, :]
            if seg.shape[0] != length:
                continue
            segs.append(seg)
            n_valid += 1

        if not segs:
            return np.empty((2, 0)), 0

        mean_T2 = np.mean(np.stack(segs, axis=0), axis=0)
        return mean_T2.T, n_valid

    def avg_by_ts(var_list, ts_sign, *, mode="post_start", pre_T=150, apply_fill=False):
        """Average grouped by trial side only (stim buckets)."""
        var_array = np.asarray(var_list, dtype=float)
        sel = [i for i in range(n) if int(np.sign(trial_sides[i][0])) == ts_sign]
        segs, T_eff, n_valid = [], None, 0
        for i in sel:
            m_i = lens[i]
            if m_i < hard_need:
                continue
            if mode == "pre_action":
                if reaction_time is None:
                    continue
                act_start = steps_before_obs + int(reaction_time[i])
                if act_start < T or act_start > m_i:
                    continue
                start, this_T = offsets[i] + act_start - T, T
                seg = var_array[start:start+this_T, :]
            else:
                if apply_fill:
                    post_avail = max(0, m_i - steps_before_obs)
                    take_post  = min(T, post_avail)
                    parts = []
                    if take_post > 0:
                        start = offsets[i] + steps_before_obs
                        parts.append(var_array[start:start+take_post, :])
                    if take_post < T:
                        if i + 1 >= n:
                            continue
                        m_next = lens[i+1]
                        if m_next < hard_need:
                            continue
                        pre_avail_next = min(steps_before_obs, m_next)
                        need = T - take_post
                        if pre_avail_next < need:
                            continue
                        start_next = offsets[i+1]
                        parts.append(var_array[start_next:start_next + need, :])
                    seg = np.vstack(parts) if parts else None
                    if seg is None or seg.shape[0] != T:
                        continue
                    this_T = T
                else:
                    if steps_before_obs >= m_i:
                        continue
                    start, this_T = offsets[i] + steps_before_obs, min(T, m_i - steps_before_obs)
                    seg = var_array[start:start+this_T, :]
            if seg is not None and seg.size:
                segs.append(seg)
                T_eff = this_T if T_eff is None else min(T_eff, this_T)
                n_valid += 1
        if not segs or T_eff is None:
            return np.empty((2, 0)), 0
        segs = [seg[:T_eff, :] for seg in segs]
        mean_T2 = np.mean(np.stack(segs, axis=0), axis=0)
        return mean_T2.T, n_valid

    def avg_by_sp(var_list, sp_sign, *, mode="post_start", pre_T=150, apply_fill=False):
        """Average grouped by prior side only (sp buckets)."""
        var_array = np.asarray(var_list, dtype=float)
        sel = [i for i in range(n) if _sp_sign(sub_prior[i][0]) == sp_sign]
        segs, T_eff, n_valid = [], None, 0
        for i in sel:
            m_i = lens[i]
            if m_i < hard_need:
                continue
            if mode == "pre_action":
                if reaction_time is None:
                    continue
                act_start = steps_before_obs + int(reaction_time[i])
                if act_start < T or act_start > m_i:
                    continue
                start, this_T = offsets[i] + act_start - T, T
                seg = var_array[start:start+this_T, :]
            else:
                if apply_fill:
                    post_avail = max(0, m_i - steps_before_obs)
                    take_post  = min(T, post_avail)
                    parts = []
                    if take_post > 0:
                        start = offsets[i] + steps_before_obs
                        parts.append(var_array[start:start+take_post, :])
                    if take_post < T:
                        if i + 1 >= n:
                            continue
                        m_next = lens[i+1]
                        if m_next < hard_need:
                            continue
                        pre_avail_next = min(steps_before_obs, m_next)
                        need = T - take_post
                        if pre_avail_next < need:
                            continue
                        start_next = offsets[i+1]
                        parts.append(var_array[start_next:start_next + need, :])
                    seg = np.vstack(parts) if parts else None
                    if seg is None or seg.shape[0] != T:
                        continue
                    this_T = T
                else:
                    if steps_before_obs >= m_i:
                        continue
                    start, this_T = offsets[i] + steps_before_obs, min(T, m_i - steps_before_obs)
                    seg = var_array[start:start+this_T, :]
            if seg is not None and seg.size:
                segs.append(seg)
                T_eff = this_T if T_eff is None else min(T_eff, this_T)
                n_valid += 1
        if not segs or T_eff is None:
            return np.empty((2, 0)), 0
        segs = [seg[:T_eff, :] for seg in segs]
        mean_T2 = np.mean(np.stack(segs, axis=0), axis=0)
        return mean_T2.T, n_valid

    # --- main loop ---
    out = {vn: {} for vn in var_names}

    def _nan_if_few(arr, cnt):
        return np.full((2, arr.shape[1]), np.nan) if (cnt < 10 and arr.size != 0) else arr

    for vn in var_names:
        vlist = results[vn]

        if vn == "P":
            for sp in [-1, 1]:
                arr, cnt = avg_for_P(vlist, sp, pre_T=150)
                out[vn][sp] = _nan_if_few(arr, cnt)
            continue

        # For S: keep post-start only (original behavior).
        # For I and M: compute BOTH post-start ("post") and pre-movement ("pre") windows.
        both_windows = (vn in ("I", "M"))

        # (ts, ch, sp) buckets
        # for ts, ch, sp in product([-1, 1], repeat=3):
        #     if both_windows:
        #         arr_post, cnt_post = avg_for_var(vlist, ts, ch, sp, mode="post_start", apply_fill=True)
        #         arr_pre,  cnt_pre  = avg_for_var(vlist, ts, ch, sp, mode="pre_action")
        #         out[vn][('post', (ts, ch, sp))] = _nan_if_few(arr_post, cnt_post)
        #         out[vn][('pre',  (ts, ch, sp))] = _nan_if_few(arr_pre,  cnt_pre)
        #     else:
        #         arr, cnt = avg_for_var(vlist, ts, ch, sp, mode="post_start")
        #         out[vn][(ts, ch, sp)] = _nan_if_few(arr, cnt)

        # choice × prior buckets
        # for ch, sp in product([-1, 1], [-1, 1]):
        #     if both_windows:
        #         arr_post, cnt_post = avg_by_ch_sp(vlist, ch, sp, mode="post_start", apply_fill=True)
        #         arr_pre,  cnt_pre  = avg_by_ch_sp(vlist, ch, sp, mode="pre_action")
        #         out[vn][('post', ('chsp', ch, sp))] = _nan_if_few(arr_post, cnt_post)
        #         out[vn][('pre',  ('chsp', ch, sp))] = _nan_if_few(arr_pre,  cnt_pre)
        #     else:
        #         arr, cnt = avg_by_ch_sp(vlist, ch, sp, mode="post_start")
        #         out[vn][('chsp', ch, sp)] = _nan_if_few(arr, cnt)

        # choice-only buckets
        for ch in [-1, 1]:
            if both_windows:
                arr_post, cnt_post = avg_by_ch(vlist, ch, mode="post_start", apply_fill=True)
                arr_pre,  cnt_pre  = avg_by_ch(vlist, ch, mode="pre_action")
                out[vn][('post', ('ch', ch))] = _nan_if_few(arr_post, cnt_post)
                out[vn][('pre',  ('ch', ch))] = _nan_if_few(arr_pre,  cnt_pre)
            else:
                arr, cnt = avg_by_ch(vlist, ch, mode="post_start")
                out[vn][('ch', ch)] = _nan_if_few(arr, cnt)

        if both_windows:
            for prev_ch in [-1, 1]:
                arr_iti, cnt_iti = avg_intertrial_by_prev_ch(vlist, prev_ch)
                out[vn][('iti_prev', ('prev_ch', prev_ch))] = _nan_if_few(arr_iti, cnt_iti)

        # trial side–only (stim buckets): only for S and I (include both windows for I)
        # if vn in ("S", "I", "S_ff"):
        #     for ts in [-1, 1]:
        #         if vn == "I":
        #             arr_post, cnt_post = avg_by_ts(vlist, ts, mode="post_start", apply_fill=True)
        #             arr_pre,  cnt_pre  = avg_by_ts(vlist, ts, mode="pre_action")
        #             out[vn][('post', ('ts', ts))] = _nan_if_few(arr_post, cnt_post)
        #             out[vn][('pre',  ('ts', ts))] = _nan_if_few(arr_pre,  cnt_pre)
        #         else:
        #             arr, cnt = avg_by_ts(vlist, ts, mode="post_start", apply_fill=True)
        #             out[vn][('ts', ts)] = _nan_if_few(arr, cnt)

        # prior-only (sp buckets): S keeps post-start; I and M get both
        # if vn in ("S", "I", "M"):
        #     for sp in [-1, 1]:
        #         if both_windows:
        #             arr_post, cnt_post = avg_by_sp(vlist, sp, mode="post_start", apply_fill=True)
        #             arr_pre,  cnt_pre  = avg_by_sp(vlist, sp, mode="pre_action")
        #             out[vn][('post', ('sp', sp))] = _nan_if_few(arr_post, cnt_post)
        #             out[vn][('pre',  ('sp', sp))] = _nan_if_few(arr_pre,  cnt_pre)
        #         else:
        #             arr, cnt = avg_by_sp(vlist, sp, mode="post_start", apply_fill=True)
        #             out[vn][('sp', sp)] = _nan_if_few(arr, cnt)

    return out


def plot_diff_by_condition(avg_dict, var_names=("S","I","P","M"), dt=2, reaction_time=None):
    """
    avg_dict: output of mean_by_condition(...)
              { var_name: { (ts,ch,sp): array(2, T_eff) } }
    reaction_time: list/array of per-trial reaction times (steps), optional.
                   Used only for M to mark trial start relative to action onset.
    Plots (channel1 - channel0) over time for each non-empty condition.
      - For 'P': x is negative time, aligned so 0 = trial start.
      - For 'M': x is negative time, aligned so 0 = action start.
                 Also draws a second dashed line at trial start if reaction_time is given.
      - Others: x is nonnegative time after trial start.
    """
    for vn in var_names:
        cond_map = avg_dict.get(vn, {})
        plt.figure(figsize=(7, 4))
        any_plotted = False

        for ts, ch, sp in product([-1, 1], repeat=3):
            arr = cond_map.get((ts, ch, sp))
            if arr is None or arr.size == 0:
                continue

            diff_t = arr[1] - arr[0]  # (T_eff,)
            T_eff = diff_t.shape[0]

            if vn in ("P", "M"):
                # indices: [-T_eff+1, ..., -1, 0]
                t = np.arange(-T_eff + 1, 1) * dt
            else:
                t = np.arange(T_eff) * dt

            label = f"ts={ts}, ch={ch}, sp={sp}"
            plt.plot(t, diff_t, '-o', markersize=3, linewidth=1, label=label)
            any_plotted = True

        # Reference lines
        plt.axhline(0, color='k', linestyle='--', linewidth=1)
        plt.axvline(0, color='k', linestyle='--', linewidth=1)

        # For M, mark trial start relative to action
        if vn == "M" and reaction_time is not None:
            rt_vals = np.asarray(reaction_time, dtype=float)
            if rt_vals.size > 0:
                mean_rt = np.mean(rt_vals) * dt
                plt.axvline(-mean_rt, color='r', linestyle='--', linewidth=1,
                            label="trial start")

        plt.xlabel(f"time ({'ms' if dt != 1.0 else 'steps'})")
        plt.ylabel(f"{vn}[1] − {vn}[0]")
        plt.title(f"{vn}: channel difference by condition")
        if any_plotted:
            plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.show()


def prior_distance_I_M_both_alignments(
    results, steps_before_obs, T=75, metric="l2",
    include_all_trials=True, lump_all=False
):
    """
    Prior-distance (sp=+1 vs sp=-1) under:
      • start alignment ("post_start" with fill)
      • action alignment ("pre_action")
    Variables: I, M (both alignments) and S (start-only).

    If include_all_trials is False (default): use CORRECT trials only (ts == ch). [Original behavior]
    If include_all_trials is True: include ALL trials with a realized choice (ch ∈ {±1}).
    If lump_all is True: IGNORE trial side and choice side; pool all qualifying trials by prior side only.
      Otherwise (default), compute within each (trial_side, choice_side) combo, average equally across combos
      for each sp, then take the distance between the two balanced means (sp=+1 vs sp=-1).

    Rules (unchanged):
      • Require m_i >= steps_before_obs + _min_trial_steps(); shorter trials skipped.
      • If >50% of trials fail this rule globally, return all-NaNs.
      • For start-aligned segments shorter than T, fill from next trial’s pre-start
        [0:steps_before_obs). If still < T, skip that trial.

    metric ∈ {'l2','side'}:
      - 'l2'  : Euclidean over the two channels at each time
      - 'side': |(R−L)_sp=+1 − (R−L)_sp=−1|
    """

    choices       = results['choices']
    trial_sides   = results['trial_sides']
    sub_prior     = results['sub_prior']
    reaction_time = results.get('reaction_time', None)
    n             = len(choices)

    lens    = [len(trial_sides[i]) for i in range(n)]
    offsets = np.cumsum([0] + lens[:-1])
    hard_need = steps_before_obs + _min_trial_steps()
    fail_cnt = sum(1 for m in lens if m < hard_need)

    # Global >50% fail → all NaNs
    if n > 0 and fail_cnt > n/2:
        nanT = np.full(T, np.nan)
        return {
            'I': {'start': nanT.copy(), 'action': nanT.copy()},
            'M': {'start': nanT.copy(), 'action': nanT.copy()},
            'S': {'start': nanT.copy(), 'action': np.array([])}
        }

    def _segments_by_bucket(var_name, mode):
        """
        Collect aligned segments.
        Returns dict[(ts, ch, sp)] -> list of (T,2) arrays, or (sp,) if lump_all=True.
        """
        var_array = np.asarray(results[var_name], dtype=float)
        if var_array.ndim != 2 or var_array.shape[1] != 2:
            raise ValueError(f"{var_name}: expected (TotalSteps,2), got {var_array.shape}")

        buckets = {}
        for i in range(n):
            m_i = lens[i]
            if m_i < hard_need:
                continue

            ts = int(np.sign(trial_sides[i][0]))     # -1 or +1
            ch = int(choices[i])                     # -1, 0, +1
            if ch == 0:
                continue  # skip no-choice trials

            # correct-only unless include_all_trials=True
            if not include_all_trials and (ch != ts):
                continue

            if mode == "pre_action":
                if reaction_time is None:
                    raise ValueError("reaction_time required for pre_action alignment")
                act_start = steps_before_obs + int(reaction_time[i])
                if act_start < T or act_start > m_i:
                    continue
                start = offsets[i] + act_start - T
                seg = var_array[start:start+T, :]
            else:  # "post_start" with fill rule
                post_avail = max(0, m_i - steps_before_obs)
                take_post  = min(T, post_avail)
                parts = []
                if take_post > 0:
                    start = offsets[i] + steps_before_obs
                    parts.append(var_array[start:start+take_post, :])
                if take_post < T:
                    if i + 1 >= n:
                        continue
                    m_next = lens[i+1]
                    if m_next < hard_need:
                        continue
                    pre_avail_next = min(steps_before_obs, m_next)
                    need = T - take_post
                    if pre_avail_next < need:
                        continue
                    start_next = offsets[i+1]
                    parts.append(var_array[start_next:start_next+need, :])
                if not parts:
                    continue
                seg = np.vstack(parts)
                if seg.shape[0] != T:
                    continue

            # subjective prior sign
            sp = 1 if sub_prior[i][0] < 0 else -1
            key = (sp,) if lump_all else (ts, ch, sp)
            buckets.setdefault(key, []).append(seg)

        return buckets

    def _mean_for_sp(buckets, sp_sign):
        """
        If lump_all: mean across ALL qualifying trials with given sp (key = (sp,))
        Else: balanced mean across present (ts, ch) ∈ {±1}×{±1} for given sp.
        Returns (T,2) or None.
        """
        if lump_all:
            key = (sp_sign,)
            if key not in buckets or len(buckets[key]) == 0:
                return None
            return np.mean(np.stack(buckets[key], axis=0), axis=0)
        bucket_means = []
        for ts in (+1, -1):
            for ch in (+1, -1):
                key = (ts, ch, sp_sign)
                if key in buckets and len(buckets[key]) > 0:
                    B = np.mean(np.stack(buckets[key], axis=0), axis=0)
                    bucket_means.append(B)
        if not bucket_means:
            return None
        return np.mean(np.stack(bucket_means, axis=0), axis=0)

    def _balanced_prior_distance(var_name, mode):
        buckets = _segments_by_bucket(var_name, mode)
        A_pos = _mean_for_sp(buckets, +1)
        A_neg = _mean_for_sp(buckets, -1)
        if A_pos is None or A_neg is None:
            return np.array([])
        if metric == "l2":
            return np.linalg.norm(A_pos - A_neg, axis=1)
        elif metric == "side":
            return np.abs((A_pos[:, 1] - A_pos[:, 0]) - (A_neg[:, 1] - A_neg[:, 0]))
        else:
            raise ValueError("metric must be 'l2' or 'side'")

    out = {'I': {}, 'M': {}, 'S': {}}
    # I, M: both alignments
    for vn in ('I', 'M'):
        out[vn]['start']  = _balanced_prior_distance(vn, "post_start")
        out[vn]['action'] = _balanced_prior_distance(vn, "pre_action")

    # S: start-only
    out['S']['start']  = _balanced_prior_distance('S', "post_start")
    out['S']['action'] = np.array([])

    return out


def _prior_distance_I_M_both_alignments_torch(
    results, steps_before_obs, T=75, metric="l2",
    include_all_trials=True, lump_all=False,
    dtype=torch.float32, device=torch.device('cpu')
):
    def _tensor(value):
        return _ensure_tensor(value, dtype=dtype, device=device, requires_grad=False)

    choices = [int(c) for c in results['choices']]
    trial_sides = [ _tensor(ts) for ts in results['trial_sides'] ]
    sub_prior = [ _tensor(sp) for sp in results['sub_prior'] ]
    reaction_time = results.get('reaction_time', None)
    if reaction_time is not None:
        reaction_time = [int(float(rt)) for rt in reaction_time]

    var_tensors = {
        'S': _tensor(results['S']),
        'I': _tensor(results['I']),
        'P': _tensor(results['P']),
        'M': _tensor(results['M'])
    }

    n = len(choices)
    lens = [len(ts) for ts in trial_sides]
    offsets = [0]
    for ln in lens[:-1]:
        offsets.append(offsets[-1] + ln)
    offsets = torch.tensor(offsets, dtype=torch.long, device=device)
    hard_need = steps_before_obs + _min_trial_steps()
    fail_cnt = sum(1 for m in lens if m < hard_need)
    if n > 0 and fail_cnt > n/2:
        nan_vec = torch.full((T,), float('nan'), dtype=dtype, device=device)
        return {
            'I': {'start': nan_vec.clone(), 'action': nan_vec.clone()},
            'M': {'start': nan_vec.clone(), 'action': nan_vec.clone()},
            'S': {'start': nan_vec.clone(), 'action': torch.empty(0, dtype=dtype, device=device)}
        }

    def _segments_by_bucket(var_name, mode):
        var_array = var_tensors[var_name]
        buckets = {}
        for i in range(n):
            m_i = lens[i]
            if m_i < hard_need:
                continue
            ts_val = int(torch.sign(trial_sides[i][0]).item())
            ch_val = int(choices[i])
            if ch_val == 0:
                continue
            if not include_all_trials and (ch_val != ts_val):
                continue

            if mode == "pre_action":
                if reaction_time is None:
                    raise ValueError("reaction_time required for pre_action alignment")
                act_start = steps_before_obs + int(reaction_time[i])
                if act_start < T or act_start > m_i:
                    continue
                start = int(offsets[i].item() + act_start - T)
                seg = var_array[start:start+T, :]
            else:
                post_avail = max(0, m_i - steps_before_obs)
                take_post = min(T, post_avail)
                parts = []
                if take_post > 0:
                    start = int(offsets[i].item() + steps_before_obs)
                    parts.append(var_array[start:start+take_post, :])
                if take_post < T:
                    if i + 1 >= n:
                        continue
                    m_next = lens[i+1]
                    if m_next < hard_need:
                        continue
                    pre_avail_next = min(steps_before_obs, m_next)
                    need = T - take_post
                    if pre_avail_next < need:
                        continue
                    start_next = int(offsets[i+1].item())
                    parts.append(var_array[start_next:start_next+need, :])
                if not parts:
                    continue
                seg = torch.cat(parts, dim=0)
                if seg.shape[0] != T:
                    continue

            sp_val = 1 if sub_prior[i][0] < 0 else -1
            key = (sp_val,) if lump_all else (ts_val, ch_val, sp_val)
            buckets.setdefault(key, []).append(seg)
        return buckets

    def _mean_for_sp(buckets, sp_sign):
        if lump_all:
            key = (sp_sign,)
            if key not in buckets or not buckets[key]:
                return None
            return torch.mean(torch.stack(buckets[key], dim=0), dim=0)
        bucket_means = []
        for ts in (+1, -1):
            for ch in (+1, -1):
                key = (ts, ch, sp_sign)
                if key in buckets and buckets[key]:
                    bucket_means.append(torch.mean(torch.stack(buckets[key], dim=0), dim=0))
        if not bucket_means:
            return None
        return torch.mean(torch.stack(bucket_means, dim=0), dim=0)

    def _balanced_prior_distance(var_name, mode):
        buckets = _segments_by_bucket(var_name, mode)
        A_pos = _mean_for_sp(buckets, +1)
        A_neg = _mean_for_sp(buckets, -1)
        if A_pos is None or A_neg is None:
            return torch.full((0,), float('nan'), dtype=dtype, device=device)
        if metric == "l2":
            return torch.linalg.norm(A_pos - A_neg, dim=1)
        elif metric == "side":
            diff = torch.abs((A_pos[:, 1] - A_pos[:, 0]) - (A_neg[:, 1] - A_neg[:, 0]))
            return diff
        else:
            raise ValueError("metric must be 'l2' or 'side'")

    out = {'I': {}, 'M': {}, 'S': {}}
    for vn in ('I', 'M'):
        out[vn]['start'] = _balanced_prior_distance(vn, "post_start")
        out[vn]['action'] = _balanced_prior_distance(vn, "pre_action")
    out['S']['start'] = _balanced_prior_distance('S', "post_start")
    out['S']['action'] = torch.empty(0, dtype=dtype, device=device)
    return out


def prior_distance_I_M_by_choice_and_prior(results, steps_before_obs, T=75, metric="side", min_valid_trials=10):
    """
    Prior-distance comparing subjective prior groups (sp=+1 vs sp=-1) for each CHOICE side,
    under both alignments (trial start, action start), for variables I and M.

    Selection: trials are included solely by (choice side, subjective prior), ignoring trial side and correctness.

    Rules:
      • Hard trial-length requirement: m_i >= steps_before_obs + _min_trial_steps() for ALL variables (skip otherwise).
      • If >50% of trials fail this rule, return all-NaNs for every variable/bucket.
      • For post-start windows: if length < T, fill the remainder from the NEXT trial's pre-start [0:steps_before_obs).
        If still < T, skip that trial.
      • If any (choice, sp) bucket has < min_valid_trials valid trials, its mean is set to NaNs.

    Returns:
      {
        'I': {'start':  (Tmin,), 'action': (Tmin,)},
        'M': {'start':  (Tmin,), 'action': (Tmin,)}
      }

    metric ∈ {'side','l2'}:
      - 'side': within each choice side, form channel diff as
                Right choice:  ch1 - ch0
                Left  choice:  ch0 - ch1
                then take |Δprior| between sp=+1 and sp=-1, and average over the two choice sides.
      - 'l2'  : Euclidean over the two channels, then |Δprior| between sp=+1 and sp=-1, averaged over choice sides.
    """

    choices       = results['choices']         # list/array of ±1
    trial_sides   = results['trial_sides']     # used only for lengths/offsets
    sub_prior     = results['sub_prior']       # real-valued; sign determines sp
    reaction_time = results.get('reaction_time', None)
    n             = len(choices)

    lens    = [len(trial_sides[i]) for i in range(n)]
    offsets = np.cumsum([0] + lens[:-1])
    hard_need = steps_before_obs + _min_trial_steps()
    fail_cnt = sum(1 for m in lens if m < hard_need)

    # Global >50% fail → all NaNs
    if n > 0 and fail_cnt > n/2:
        nanT = np.full(T, np.nan)
        return {'I': {'start': nanT.copy(), 'action': nanT.copy()},
                'M': {'start': nanT.copy(), 'action': nanT.copy()}}

    def _sp_sign(x):  # analog prior → ±1
        return 1 if x < 0 else -1

    def _avg_var_align_by_choice_and_sp(var_name, ch_sign, sp_sign, mode):
        """
        Average (2, T_eff) segment for var_name given choice side and subjective prior,
        applying the hard-length rule and (for post_start) fill-from-next.
        Returns (arr, count_valid), where arr.shape == (2, T_eff) or empty (2,0).
        """
        var_list  = results[var_name]
        var_array = np.asarray(var_list, dtype=float)
        if var_array.ndim != 2 or var_array.shape[1] != 2:
            raise ValueError(f"Expected var entries to be 2D vectors; got shape {var_array.shape}")

        # Select by (choice == ch_sign) AND (sp == sp_sign). Ignore trial side/correctness.
        sel = []
        for i in range(n):
            sp0 = _sp_sign(sub_prior[i][0])
            if (choices[i] == ch_sign) and (sp0 == sp_sign):
                sel.append(i)

        segs, T_eff, n_valid = [], None, 0
        for i in sel:
            m_i = lens[i]
            if m_i < hard_need:
                continue

            if mode == "pre_action":
                if reaction_time is None:
                    raise ValueError("reaction_time required for pre_action alignment")
                act_start = steps_before_obs + int(reaction_time[i])
                if act_start < T or act_start > m_i:
                    continue
                start, this_T = offsets[i] + act_start - T, T
                seg = var_array[start:start+this_T, :]

            else:  # "post_start" with fill-from-next rule
                post_avail = max(0, m_i - steps_before_obs)
                take_post  = min(T, post_avail)
                parts = []
                if take_post > 0:
                    start = offsets[i] + steps_before_obs
                    parts.append(var_array[start:start+take_post, :])
                if take_post < T:
                    if i + 1 >= n:
                        continue
                    m_next = lens[i+1]
                    if m_next < hard_need:
                        continue
                    pre_avail_next = min(steps_before_obs, m_next)
                    need = T - take_post
                    if pre_avail_next < need:
                        continue
                    start_next = offsets[i+1]
                    parts.append(var_array[start_next:start_next + need, :])
                seg = np.vstack(parts) if parts else None
                if seg is None or seg.shape[0] != T:
                    continue
                this_T = T

            if seg is not None and seg.size:
                segs.append(seg)
                T_eff = this_T if T_eff is None else min(T_eff, this_T)
                n_valid += 1

        if not segs or T_eff is None:
            return np.empty((2, 0)), 0

        segs = [seg[:T_eff, :] for seg in segs]
        mean_T2 = np.mean(np.stack(segs, axis=0), axis=0)  # (T_eff, 2)
        return mean_T2.T, n_valid                           # (2, T_eff), count

    out = {'I': {}, 'M': {}}
    for vn in ('I', 'M'):
        # Buckets: by CHOICE side (+1 Right, -1 Left) and PRIOR sign (+1, -1)
        R_start_sp1, cnt_Rs1  = _avg_var_align_by_choice_and_sp(vn, +1, +1, "post_start")
        R_start_spm, cnt_Rsm  = _avg_var_align_by_choice_and_sp(vn, +1, -1, "post_start")
        L_start_sp1, cnt_Ls1  = _avg_var_align_by_choice_and_sp(vn, -1, +1, "post_start")
        L_start_spm, cnt_Lsm  = _avg_var_align_by_choice_and_sp(vn, -1, -1, "post_start")

        R_action_sp1, cnt_Ra1 = _avg_var_align_by_choice_and_sp(vn, +1, +1, "pre_action")
        R_action_spm, cnt_Ram = _avg_var_align_by_choice_and_sp(vn, +1, -1, "pre_action")
        L_action_sp1, cnt_La1 = _avg_var_align_by_choice_and_sp(vn, -1, +1, "pre_action")
        L_action_spm, cnt_Lam = _avg_var_align_by_choice_and_sp(vn, -1, -1, "pre_action")

        # Apply min_valid_trials gate per bucket (mirror mean_by_condition behavior)
        def _gate(arr, cnt):
            if cnt < min_valid_trials and arr.size != 0:
                return np.full((2, arr.shape[1]), np.nan)
            return arr

        R_start_sp1  = _gate(R_start_sp1,  cnt_Rs1)
        R_start_spm  = _gate(R_start_spm,  cnt_Rsm)
        L_start_sp1  = _gate(L_start_sp1,  cnt_Ls1)
        L_start_spm  = _gate(L_start_spm,  cnt_Lsm)

        R_action_sp1 = _gate(R_action_sp1, cnt_Ra1)
        R_action_spm = _gate(R_action_spm, cnt_Ram)
        L_action_sp1 = _gate(L_action_sp1, cnt_La1)
        L_action_spm = _gate(L_action_spm, cnt_Lam)

        # Compute distances for 'start'
        arrs_s = [R_start_sp1, R_start_spm, L_start_sp1, L_start_spm]
        if any(a.size == 0 for a in arrs_s):
            dist_start = np.array([])
        else:
            Ts = min(a.shape[1] for a in arrs_s)
            if metric == "l2":
                dR = np.linalg.norm(R_start_sp1[:, :Ts] - R_start_spm[:, :Ts], axis=0)
                dL = np.linalg.norm(L_start_sp1[:, :Ts] - L_start_spm[:, :Ts], axis=0)
            elif metric == "side":
                # Choice-specific channel diff convention
                r1, r2 = R_start_sp1[1,:Ts]-R_start_sp1[0,:Ts], R_start_spm[1,:Ts]-R_start_spm[0,:Ts]
                l1, l2 = L_start_sp1[0,:Ts]-L_start_sp1[1,:Ts], L_start_spm[0,:Ts]-L_start_spm[1,:Ts]
                dR, dL = np.abs(r1 - r2), np.abs(l1 - l2)
            else:
                raise ValueError("metric must be 'l2' or 'side'")
            dist_start = 0.5 * (dR + dL)

        # Compute distances for 'action'
        arrs_a = [R_action_sp1, R_action_spm, L_action_sp1, L_action_spm]
        if any(a.size == 0 for a in arrs_a):
            dist_action = np.array([])
        else:
            Ta = min(a.shape[1] for a in arrs_a)
            if metric == "l2":
                dR = np.linalg.norm(R_action_sp1[:, :Ta] - R_action_spm[:, :Ta], axis=0)
                dL = np.linalg.norm(L_action_sp1[:, :Ta] - L_action_spm[:, :Ta], axis=0)
            elif metric == "side":
                r1, r2 = R_action_sp1[1,:Ta]-R_action_sp1[0,:Ta], R_action_spm[1,:Ta]-R_action_spm[0,:Ta]
                l1, l2 = L_action_sp1[0,:Ta]-L_action_sp1[1,:Ta], L_action_spm[0,:Ta]-L_action_spm[1,:Ta]
                dR, dL = np.abs(r1 - r2), np.abs(l1 - l2)
            else:
                raise ValueError("metric must be 'l2' or 'side'")
            dist_action = 0.5 * (dR + dL)

        out[vn]['start']  = dist_start
        out[vn]['action'] = dist_action

    return out


def loss_prior_effect(
    regions, results, model_params, steps_before_obs, T=72, model_metric="l2",
    timeframes=('act_block_duringstim','act_block_duringchoice'),
    alpha=1.0, ptype='p_mean_c', plot_window=80, reload=False,
    label_A='integrator', label_B='move', label_S='stim', plot_stim=False,
    do_plot=False, save_dir=save_dir, shift_baseline=False, plot_shifted=False, ylim=None,
    scale_factors=[1, 1, 1], include_all_trials=True, lump_all=False, correction='simple',
    gradient_mode=False, grad_options=None
):
    """
    Overlay real-data region-group curves with model prior-distance curves (I, M, and S for duringstim),
    and compute SSEs (normalized). For choice frames, sse_stim is fixed to 0.
    Plotting uses exactly the same data slices used for SSE calculation.

    Added: GoF (energy-normalized) per channel and per timeframe:
        GoF = 1 - SSE_norm , where SSE_norm = sum((y_m - y_d)^2) / sum(y_d^2)
    The returned dict includes per-timeframe GoFs and an overall mean GoF across timeframes.
    """

    if isinstance(timeframes, str):
        timeframes = (timeframes,)

    eps = 1e-12  # for normalization stability

    # --- helpers ---
    def time_axis_for(timeframe, time_window, duration_ms=None):
        """
        Create time axis for plotting.
        
        Args:
            timeframe: 'duringchoice' or 'duringstim'
            time_window: Number of bins
            duration_ms: Optional duration in milliseconds. If provided, the axis spans from 
                        -duration_ms to 0 (choice) or 0 to duration_ms (stim)
        """
        if duration_ms is not None:
            # time_window is number of bins, duration_ms is the duration in ms
            if 'duringchoice' in timeframe:
                return np.linspace(-duration_ms, 0, time_window)
            elif 'duringstim' in timeframe:
                return np.linspace(0, duration_ms, time_window)
        else:
            # Legacy: time_window is number of bins, assume 2ms per bin
            if 'duringchoice' in timeframe:
                return np.linspace(-2*time_window, 0, time_window)
            elif 'duringstim' in timeframe:
                return np.linspace(0, 2*time_window, time_window)

    def _align_baseline(y_model, y_data, baseline_mode="min"):
        if not shift_baseline:
            return y_data
        if y_model is None or y_data is None or len(y_model) == 0 or len(y_data) == 0:
            return y_data
        if np.all(np.isnan(y_model)) or np.all(np.isnan(y_data)):
            return y_data
        m_min = np.nanmin(y_model)
        d_base = np.nanmean(y_data[:5]) if baseline_mode == "min" else np.nanmean(y_data)
        return y_data - d_base + m_min

    def _amp(arr):
        if arr is None or (isinstance(arr, np.ndarray) and arr.size == 0):
            return np.nan
        amax, amin = np.nanmax(arr), np.nanmin(arr)
        return float(amax - amin) if np.isfinite(amax) and np.isfinite(amin) else np.nan

    def _safe_ratio(num, den):
        return float(num / den) if (np.isfinite(num) and np.isfinite(den) and den != 0) else np.nan

    if gradient_mode:
        out = _loss_prior_effect_torch(
            regions, results, model_params, steps_before_obs, T=T,
            model_metric=model_metric, timeframes=timeframes, alpha=alpha, ptype=ptype,
            plot_window=plot_window, shift_baseline=shift_baseline,
            scale_factors=scale_factors, include_all_trials=include_all_trials,
            lump_all=lump_all, correction=correction,
            grad_options=grad_options or model_params.get('grad_options', {}))
        if do_plot:
            results_np = _detach_to_numpy(results)
            model_params_np = _detach_to_numpy(model_params)
            loss_prior_effect(
                regions=regions,
                results=results_np,
                model_params=model_params_np,
                steps_before_obs=steps_before_obs,
                T=T,
                model_metric=model_metric,
                timeframes=timeframes,
                alpha=alpha,
                ptype=ptype,
                plot_window=plot_window,
                reload=reload,
                label_A=label_A,
                label_B=label_B,
                label_S=label_S,
                plot_stim=plot_stim,
                do_plot=True,
                save_dir=save_dir,
                shift_baseline=shift_baseline,
                plot_shifted=plot_shifted,
                ylim=ylim,
                scale_factors=scale_factors,
                include_all_trials=include_all_trials,
                lump_all=lump_all,
                correction=correction,
                gradient_mode=False,
                grad_options=grad_options,
            )
        return out

    # model distances
    model_dists = prior_distance_I_M_both_alignments(
        results, steps_before_obs, T=T, metric=model_metric,
        include_all_trials=include_all_trials, lump_all=lump_all
    )

    # parse scaling
    if scale_factors is None:
        s_fac, i_fac, m_fac = 1.0, 1.0, 1.0
    else:
        if len(scale_factors) != 3:
            raise ValueError("scale_factors must be [S_factor, I_factor, M_factor]")
        s_fac, i_fac, m_fac = (float(scale_factors[0]), float(scale_factors[1]), float(scale_factors[2]))

    # setup figure
    if do_plot:
        fig, axs = plt.subplots(1, len(timeframes), sharey=True, figsize=(5, 2), dpi=150)
        if len(timeframes) == 1:
            axs = [axs]
    else:
        axs = [None] * len(timeframes)

    sse = {'total': 0.0}
    had_any_tf = False
    any_tf_nan = False
    gof_over_timeframes = []  # mean GoF across channels per timeframe

    for ax, timeframe in zip(axs, timeframes):
        is_choice = ('duringchoice' in timeframe)
        regs_move = regions['move_regs_choice'] if is_choice else regions['move_regs_stim']
        regs_int  = regions['int_regs_choice']  if is_choice else regions['int_regs_stim']
        if not is_choice:
            regs_stim = regions['stim_regs']

        align_key = 'action' if is_choice else 'start'

        # Load data
        if reload:
            save_data = {'regs_int': regs_int, 'regs_move': regs_move, 'regs_stim': regs_stim}

            r_int  = load_group(regs_int, timeframe, ptype=ptype, alpha=alpha, correction=correction)
            r_move = load_group(regs_move, timeframe, ptype=ptype, alpha=alpha, correction=correction)

            if not is_choice:
                r_stim = load_group(regs_stim, timeframe, ptype=ptype, alpha=alpha, is_stim=True, correction=correction)
            else:
                r_stim = None

            save_data['r_int'] = r_int
            save_data['r_move'] = r_move
            save_data['r_stim'] = r_stim
            np.save(f'data_{timeframe}.npy', save_data, allow_pickle=True)
        else: 
            save_data = np.load(f'data_{timeframe}.npy', allow_pickle=True).flat[0]
            r_int = save_data['r_int']
            r_move = save_data['r_move']
            r_stim = save_data['r_stim']
        
        # plot_window is now in milliseconds. Use data's actual length for time axis
        # Data bin size is ~2.08ms, so calculate expected bins, but use actual data length
        # data_length = len(r_int) if r_int is not None and len(r_int) > 0 else int(round(plot_window / 2.08))
        data_length = int(round(plot_window / 2))
        # time_axis_for: time_window is number of bins, duration_ms is the duration in ms
        times_full = time_axis_for(timeframe, time_window=data_length, duration_ms=plot_window)
        # print(len(times_full))

        # Model
        m_I = i_fac * model_dists['I'][align_key]
        m_M = m_fac * model_dists['M'][align_key]
        m_S = (s_fac * model_dists['S'][align_key]) if (not is_choice and 'S' in model_dists) else None

        sse_int = sse_move = sse_stim = np.nan
        amp_I_data = amp_M_data = amp_I_model = amp_M_model = np.nan

        # unified processor for I/M/S with normalized SSE
        def _proc(name, r_data, m_model, color, label, baseline_mode, plot_flag=True):
            nonlocal amp_I_data, amp_M_data, amp_I_model, amp_M_model, sse_int, sse_move, sse_stim
            if (r_data is None) or (m_model is None) or (not getattr(m_model, "size", 0)):
                return np.nan
            L = min(len(m_model), len(r_data), len(times_full))
            if L <= 0:
                return np.nan

            if is_choice:
                t = times_full[-L:]
                y_m = m_model[-L:]
                y_d_src = r_data[-L:]
            else:
                t = times_full[:L]
                y_m = m_model[:L]
                y_d_src = r_data[:L]

            y_d = _align_baseline(y_m, y_d_src, baseline_mode=baseline_mode)

            if np.any(np.isnan(y_m)) or np.any(np.isnan(y_d)):
                sse_norm = np.nan
            else:
                denom = float(np.sum(y_d * y_d) + eps)  # energy-normalized
                sse_norm = float(np.sum((y_m - y_d) ** 2) / denom)

            # amplitude tracking
            if name == 'I':
                amp_I_model = _amp(y_m); amp_I_data = _amp(y_d)
            elif name == 'M':
                amp_M_model = _amp(y_m); amp_M_data = _amp(y_d)

            # plot
            if do_plot and plot_flag:
                d_plot = _align_baseline(y_m, y_d_src, baseline_mode=baseline_mode) if plot_shifted else y_d_src
                ax.plot(t, d_plot, color=color, linewidth=2, label=label)
                ax.plot(t, y_m, '--', color=color, alpha=0.9, linewidth=2)

            return sse_norm

        # compute/plot I, M, and S (S only during stim)
        sse_int  = _proc('I', r_int,  m_I, color='gold',   label=label_A, baseline_mode="min",  plot_flag=True)
        sse_move = _proc('M', r_move, m_M, color='tomato', label=label_B, baseline_mode="min",  plot_flag=True)
        # sse_stim = 0.0 if is_choice else _proc('S', r_stim, m_S, color='C0', label=label_S,
        #                                        baseline_mode="mean", plot_flag=plot_stim)
        sse_stim = 0.0

        # update totals (unchanged)
        rel_sses = [sse_int, sse_move, sse_stim]
        if any(np.isnan(x) for x in rel_sses):
            total_tf = np.nan
            any_tf_nan = True
        else:
            total_tf = float(np.sum(rel_sses))
            sse['total'] += total_tf
            had_any_tf = True
        sse[timeframe] = {'integrator': sse_int, 'move': sse_move, 'stim': sse_stim, 'total': total_tf}

        # --- GoF (energy-normalized): GoF = 1 - SSE_norm ---
        gof_I = (1.0 - sse_int)  if np.isfinite(sse_int)  else np.nan
        gof_M = (1.0 - sse_move) if np.isfinite(sse_move) else np.nan
        # gof_S = (1.0 - sse_stim) if np.isfinite(sse_stim) else np.nan
        # gof_tf_mean = float(np.nanmean([gof_I, gof_M, gof_S])) if not (np.isnan(gof_I) and np.isnan(gof_M) and np.isnan(gof_S)) else np.nan
        gof_tf_mean = float(np.nanmean([gof_I, gof_M])) if not (np.isnan(gof_I) and np.isnan(gof_M)) else np.nan
        sse[timeframe]['gof'] = {'integrator': gof_I, 'move': gof_M, 'mean': gof_tf_mean}
        if np.isfinite(gof_tf_mean):
            gof_over_timeframes.append(gof_tf_mean)

        # axis formatting
        if do_plot:
            if ylim is not None:
                ax.set_ylim(ylim[0], ylim[1])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_facecolor('none')
            ax.tick_params(labelsize=12)

    # overall GoF summary across timeframes (mean of per-timeframe means)
    sse['gof'] = float(np.nanmean(gof_over_timeframes)) if gof_over_timeframes else np.nan

    # finalize plot (unchanged text except we still display SSE)
    if do_plot and (axs is not None) and (axs[0] is not None):
        mode = 'prior' if 'block' in timeframes[0] else 'choice'
        ylabel_left = f'd$^{{\\mathrm{{{mode}}}}}_{{\\mathrm{{\\{{I, M, S\\}}}}}}$(t)'
        axs[0].set_ylabel(ylabel_left)
        axs[0].figure.tight_layout()
        sse_total_str = f"SSE: {sse['total']:.3g}, R$^{{2}}$: {sse['gof']:.3g}" if np.isfinite(sse['total']) else "SSE: NaN, R$^2$: NaN"
        axs[0].text(
            0.98, 0.98, sse_total_str,
            transform=axs[0].transAxes,
            ha='right', va='top',
            fontsize=12, color='k',
            bbox=dict(facecolor='none', edgecolor='none', alpha=0.7, pad=2)
        )
        if save_dir:
            param_name = (
                f"gi{model_params['g_i']}_gm{model_params['g_m']}_gs{model_params['g_s']}_"
                f"di{model_params['d_i']}_dm{model_params['d_m']}_ds{model_params['d_s']}"
            )
            if scale_factors is not None:
                param_name += f"_sfS{scale_factors[0]}_sfI{scale_factors[1]}_sfM{scale_factors[2]}"
            param_name += f"thr{model_params['action_thresholds']['concordant'][0]}_{model_params['action_thresholds']['discordant'][0]}"
            fname = f"{save_dir}/prior_effects_{param_name}.svg"
            axs[0].figure.savefig(fname, transparent=True)

    if any_tf_nan or not had_any_tf:
        sse['total'] = np.nan

    return sse


def loss_perf_with_data(results, behavior, model_params, metric="correct", dt=2,
                        do_plot=False, ax=None, save_dir=None, log_xaxis=True,
                        rt_mode="combined_all"):
    """
    Compute (and optionally plot) performance or reaction time vs signed contrast,
    split by congruency, and overlay behavior. Returns SSE and GoF per curve.

    metric="correct" : fraction correct (×100 for plotting).
    metric="rt"      : reaction time; rt_mode controls RT definition:

      rt_mode = "correct_split" : RT for congruent / incongruent using correct trials only.
      rt_mode = "combined_all"  : RT combined across all trials (one curve: cc/cw/dc/dw).
      rt_mode = "split_all"     : RT for congruent / incongruent using all trials.
    """

    stim = np.array([-1, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 1])

    # extract
    trial_strengths = results['trial_strengths']
    trial_sides     = results['trial_sides']
    block_sides     = results['block_sides']
    values = results['correct_action_taken'] if metric == "correct" else results['reaction_time']
    correct_flags = np.asarray(results['correct_action_taken'], dtype=bool)

    # per-trial values
    strength_per_trial = [ts[0] for ts in trial_strengths]
    side_per_trial     = [ts[0] for ts in trial_sides]
    block_per_trial    = [bs[0] for bs in block_sides]

    strength_signed = [s * side for s, side in zip(strength_per_trial, side_per_trial)]
    is_congruent = [int(side == block) for side, block in zip(side_per_trial, block_per_trial)]
    groups = {'congruent': [], 'incongruent': []}

    # RT-only correct filtering for correct_split mode
    for i, s in enumerate(strength_signed):
        if metric == "rt" and rt_mode == "correct_split" and not correct_flags[i]:
            continue
        key = 'congruent' if is_congruent[i] else 'incongruent'
        groups[key].append((s, values[i]))

    # plotting setup
    if do_plot and ax is None:
        fig_width_in = 240 / 72
        fig_height_in = 188 / 72
        fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))

    model_curves = {}
    colors = {'congruent': 'black', 'incongruent': 'gray'}

    # Compute model curves
    for key in ['congruent', 'incongruent']:
        signed = [x[0] for x in groups[key]]
        vals   = [x[1] for x in groups[key]]
        unique_strengths = sorted(set(signed))
        mean_vals = []

        for s in unique_strengths:
            idx = [i for i, st in enumerate(signed) if st == s]
            v = [vals[i] for i in idx]
            if metric == "correct":
                mean_vals.append(float(np.mean(v)))
            else:
                mean_vals.append(float(np.mean(v) * dt))

        model_curves[key] = (unique_strengths, mean_vals)

        if do_plot and metric == "correct":
            ax.plot(unique_strengths, [100*x for x in mean_vals], '--o',
                    label=f'model {key}', color=colors[key])

        elif do_plot and metric == "rt" and rt_mode != "combined_all":
            ax.plot(unique_strengths, mean_vals, '--o',
                    label=f'model {key}', color=colors[key])

    # axis labels
    if do_plot:
        ax.set_xlabel("stimulus contrast")
        if metric == "correct":
            ax.set_ylabel("% correct responses")
            ax.set_title("performance")
        else:
            ax.set_ylabel("reaction time")
            ax.set_title("reaction time")

    # SSE bookkeeping
    sse = {
        'congruent':   np.nan,
        'incongruent': np.nan,
        'total':       np.nan,
        'gof': {
            'congruent':   np.nan,
            'incongruent': np.nan,
            'total':       np.nan,
        }
    }

    def align_to_stim(model_xy, stim_vec):
        xs, ys = model_xy
        md = {float(x): float(y) for x, y in zip(xs, ys)}
        return np.array([md.get(float(s), np.nan) for s in stim_vec], dtype=float)

    def _sse_and_gof(beh, model):
        if np.any(np.isnan(model)) or beh.shape[0] != model.shape[0]:
            return np.nan, np.nan, 0.0
        sse_val = float(np.sum((model - beh)**2))
        sst_val = float(np.sum((beh - np.nanmean(beh))**2))
        gof = (1 - sse_val/sst_val) if sst_val > 0 else np.nan
        return sse_val, gof, max(sst_val, 0.0)

    # ============================
    #       PERFORMANCE MODE
    # ============================
    if metric == "correct" and (behavior is not None):

        beh_cong = np.asarray(behavior['pct_correct']['congruent'], dtype=float)
        beh_disc = np.asarray(behavior['pct_correct']['discordant'], dtype=float)

        model_cong = align_to_stim(model_curves['congruent'], stim)
        model_disc = align_to_stim(model_curves['incongruent'], stim)

        sse_c, gof_c, sst_c = _sse_and_gof(beh_cong, model_cong)
        sse_d, gof_d, sst_d = _sse_and_gof(beh_disc, model_disc)

        sse['congruent']   = sse_c
        sse['incongruent'] = sse_d
        sse['total']       = sse_c + sse_d

        sse['gof']['congruent']   = gof_c
        sse['gof']['incongruent'] = gof_d
        sse['gof']['total']       = 1 - sse['total']/(sst_c+sst_d)

        if do_plot:
            ax.plot(stim, 100*beh_cong, '-o', label='data congruent',
                    color=colors['congruent'], alpha=0.85)
            ax.plot(stim, 100*beh_disc, '-o', label='data incongruent',
                    color=colors['incongruent'], alpha=0.85)
            ax.text(0.3, 0.85, f"R$^2$ con: {sse['gof']['congruent']:.4f}",
                    transform=ax.transAxes, fontsize=10, color=colors['congruent'])
            ax.text(0.3, 0.75, f"R$^2$ incon: {sse['gof']['incongruent']:.4f}",
                    transform=ax.transAxes, fontsize=10, color=colors['incongruent'])
    # ============================
    #         RT MODE
    # ============================
    elif metric == "rt" and (behavior is not None):

        res   = behavior['reaction_times']
        total = behavior['trial_counts']

        # Build model curves
        model_cong = align_to_stim(model_curves['congruent'], stim)
        model_disc = align_to_stim(model_curves['incongruent'], stim)

        # ---- Mode A: correct_split (cc, dc only) ----
        if rt_mode == "correct_split":
            beh_cong = np.asarray(res['cc']) / np.asarray(total['cc'])
            beh_disc = np.asarray(res['dc']) / np.asarray(total['dc'])

            sse_c, gof_c, sst_c = _sse_and_gof(beh_cong, model_cong)
            sse_d, gof_d, sst_d = _sse_and_gof(beh_disc, model_disc)

            sse['congruent'], sse['incongruent'] = sse_c, sse_d
            sse['total'] = sse_c + sse_d

            sse['gof']['congruent'], sse['gof']['incongruent'] = gof_c, gof_d
            sse['gof']['total'] = 1 - sse['total']/(sst_c+sst_d)

            if do_plot:
                ax.plot(stim, beh_cong, '-o', label='data concordant', color=colors['congruent'], alpha=0.85)
                ax.plot(stim, beh_disc, '-o', label='data discordant', color=colors['incongruent'], alpha=0.85)
                ax.text(0.3, 0.20, f"R$^2$ con: {sse['gof']['congruent']:.4f}",
                        transform=ax.transAxes, fontsize=10, color=colors['congruent'])
                ax.text(0.3, 0.10, f"R$^2$ incon: {sse['gof']['incongruent']:.4f}",
                        transform=ax.transAxes, fontsize=10, color=colors['incongruent'])

        # ---- Mode B: combined_all (one RT curve across all trials) ----
        elif rt_mode == "combined_all":
            beh_all = (
                np.asarray(res['cc']) + np.asarray(res['cw']) +
                np.asarray(res['dc']) + np.asarray(res['dw'])
            ) / (
                np.asarray(total['cc']) + np.asarray(total['cw']) +
                np.asarray(total['dc']) + np.asarray(total['dw'])
            )

            model_all = np.nanmean(np.vstack([model_cong, model_disc]), axis=0)

            sse_all, gof_all, _ = _sse_and_gof(beh_all, model_all)

            sse['total'] = sse_all
            sse['gof']['total'] = gof_all

            if do_plot:
                ax.plot(stim, model_all, '--o', label='model all (rt)', alpha=0.7, color='black')
                ax.plot(stim, beh_all,  '-o', label='data all (rt)', alpha=0.7, color='black')
                ax.text(0.3, 0.15, f"R$^2$: {sse['gof']['total']:.4f}",
                        transform=ax.transAxes, fontsize=10, color=colors['congruent'])
                                                
        # ---- Mode C: split_all (cc+cw vs dc+dw) ----
        elif rt_mode == "split_all":
            beh_cong = (
                np.asarray(res['cc']) + np.asarray(res['cw'])
            ) / (
                np.asarray(total['cc']) + np.asarray(total['cw'])
            )
            beh_disc = (
                np.asarray(res['dc']) + np.asarray(res['dw'])
            ) / (
                np.asarray(total['dc']) + np.asarray(total['dw'])
            )

            sse_c, gof_c, sst_c = _sse_and_gof(beh_cong, model_cong)
            sse_d, gof_d, sst_d = _sse_and_gof(beh_disc, model_disc)

            sse['congruent'], sse['incongruent'] = sse_c, sse_d
            sse['total'] = sse_c + sse_d

            sse['gof']['congruent'], sse['gof']['incongruent'] = gof_c, gof_d
            sse['gof']['total'] = 1 - sse['total']/(sst_c+sst_d)

            if do_plot:
                ax.plot(stim, beh_cong, '-o', label='data concordant', color=colors['congruent'], alpha=0.85)
                ax.plot(stim, beh_disc, '-o', label='data discordant', color=colors['incongruent'], alpha=0.85)
                ax.text(0.3, 0.20, f"R$^2$ con: {sse['gof']['congruent']:.4f}",
                        transform=ax.transAxes, fontsize=10, color=colors['congruent'])
                ax.text(0.3, 0.10, f"R$^2$ incon: {sse['gof']['incongruent']:.4f}",
                        transform=ax.transAxes, fontsize=10, color=colors['incongruent'])
                        
        else:
            raise ValueError(f"Unknown rt_mode: {rt_mode}")

    # -------- Axis + save --------
    if do_plot:
        if log_xaxis:
            all_x = np.array(
                list(model_curves['congruent'][0])
                + list(model_curves['incongruent'][0])
                + list(stim),
                dtype=float
            )
            nonzero = np.abs(all_x[all_x != 0])
            linthresh = float(max(1e-4, 0.5 * np.min(nonzero))) if nonzero.size else 1e-3
            ax.set_xscale('symlog', linthresh=linthresh, linscale=1.0)

            xticks = np.array([-1, -0.1, 0, 0.1, 1])
            ax.set_xticks(xticks)
            ax.set_xlim(-2, 2)

            def _fmt_tick(x):
                if x == 0: return "0"
                elif abs(x) >= 0.1: return f"{x:.1f}"
                else: return f"{x:.2f}"
            ax.set_xticklabels([_fmt_tick(x) for x in xticks])
            ax.set_xlabel("stimulus contrast (symlog)")

        plt.tight_layout()

        if save_dir:
            param_name = (
                f"gi{model_params['g_i']}_gm{model_params['g_m']}_gs{model_params['g_s']}"
                f"_di{model_params['d_i']}_dm{model_params['d_m']}_ds{model_params['d_s']}"
            )
            param_name += (
                f"_thr{model_params['action_thresholds']['concordant'][0]}_"
                f"{model_params['action_thresholds']['discordant'][0]}"
            )
            plt.savefig(f"{save_dir}/{metric}_{param_name}.svg", transparent=True)

    return sse


def loss_perf_with_data_reweighted(results, behavior, metric="correct", dt=2,
                                   do_plot=False, ax=None, save_dir=save_dir,
                                   mode: str = "lr_sym", sse_use_reweighted: bool = True):
    """
    Reweighted/symmetrized performance plot to shrink L/R differences.
    - mode="lr_sym": for each |s|, set y(+s)=y(-s)=0.5*(y(+s)+y(-s)) (if both exist).
    - sse_use_reweighted: if True, compute SSE against behavior using the reweighted curves.

    Returns:
        dict with SSE for congruent, incongruent, and total.
    """

    stim = np.array([-1, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 1])

    # --- helpers ---
    def _symmetrize_signed_curve(xs, ys):
        md = {float(x): float(y) for x, y in zip(xs, ys)}
        levels = sorted({abs(float(x)) for x in xs})
        out = {}
        for a in levels:
            yp = md.get(+a, np.nan); yn = md.get(-a, np.nan)
            if np.isfinite(yp) and np.isfinite(yn):
                ysym = 0.5 * (yp + yn)
                out[+a] = ysym; out[-a] = ysym
            elif np.isfinite(yp):
                out[+a] = yp; out[-a] = yp
            elif np.isfinite(yn):
                out[+a] = yn; out[-a] = yn
        xs_out = sorted(out.keys())
        ys_out = [out[x] for x in xs_out]
        return xs_out, ys_out

    def _align_to_stim(model_xy, stim_vec):
        xs, ys = model_xy
        md = {float(x): float(y) for x, y in zip(xs, ys)}
        return np.array([md.get(float(s), np.nan) for s in stim_vec], dtype=float)

    # --- extract per-trial values ---
    trial_strengths = results['trial_strengths']
    trial_sides     = results['trial_sides']
    block_sides     = results['block_sides']
    values = results['correct_action_taken'] if metric == "correct" else results['reaction_time']

    strength_per_trial = [ts[0] for ts in trial_strengths]
    side_per_trial     = [ts[0] for ts in trial_sides]
    block_per_trial    = [bs[0] for bs in block_sides]

    strength_signed = [s * side for s, side in zip(strength_per_trial, side_per_trial)]
    is_congruent = [int(side == block) for side, block in zip(side_per_trial, block_per_trial)]

    groups = {'congruent': [], 'incongruent': []}
    for i, s in enumerate(strength_signed):
        key = 'congruent' if is_congruent[i] else 'incongruent'
        groups[key].append((s, values[i]))

    if do_plot and ax is None:
        fig, ax = plt.subplots(figsize=(4, 2.8))

    # --- build model curves (optionally reweighted) ---
    model_curves_raw = {}
    model_curves     = {}
    colors = {'congruent': 'black', 'incongruent': 'gray'}

    for key in ['congruent', 'incongruent']:
        signed = [x[0] for x in groups[key]]
        vals   = [x[1] for x in groups[key]]
        xs = sorted(set(signed))
        ys = []
        for s in xs:
            idx = [i for i, st in enumerate(signed) if st == s]
            v = [vals[i] for i in idx]
            ys.append(float(np.mean(v) if metric == "correct" else np.mean(v) * dt))
        model_curves_raw[key] = (xs, ys)

        if mode == "lr_sym":
            xs_rw, ys_rw = _symmetrize_signed_curve(xs, ys)
            model_curves[key] = (xs_rw, ys_rw)
        else:
            model_curves[key] = (xs, ys)

        if do_plot:
            plot_x, plot_y = model_curves[key]
            lbl = f"model {key}" + ("" if mode == "none" else " (sym)")
            if metric == "correct":
                ax.plot(plot_x, [100*y for y in plot_y], '--o', label=lbl, color=colors[key])
            else:
                ax.plot(plot_x, plot_y, '--o', label=lbl, color=colors[key])

    # --- axes & titles ---
    if do_plot:
        ax.set_xlabel("stimulus contrast")
        if metric == "correct":
            ax.set_ylabel("% correct responses")
            ax.set_title("Performance by congruency (model reweighted vs data)")
        else:
            ax.set_ylabel("reaction time (s)" if dt != 1.0 else "reaction time (steps)")
            ax.set_title("reaction time by congruency (model reweighted vs data)")

    # --- SSE vs behavior (optional) ---
    sse = {'congruent': np.nan, 'incongruent': np.nan, 'total': np.nan}
    if (metric == "correct") and (behavior is not None) and (stim is not None):
        beh_cong = np.asarray(behavior['pct_correct']['congruent'], dtype=float)   # [0,1]
        beh_disc = np.asarray(behavior['pct_correct']['discordant'], dtype=float)  # [0,1]

        curves_for_sse = model_curves if sse_use_reweighted else model_curves_raw
        model_cong = _align_to_stim(curves_for_sse['congruent'],   stim)
        model_disc = _align_to_stim(curves_for_sse['incongruent'], stim)

        sse_cong = (np.nan if (np.any(np.isnan(model_cong)) or beh_cong.shape[0] != model_cong.shape[0])
                    else float(np.sum((model_cong - beh_cong) ** 2)))
        sse_disc = (np.nan if (np.any(np.isnan(model_disc)) or beh_disc.shape[0] != model_disc.shape[0])
                    else float(np.sum((model_disc - beh_disc) ** 2)))

        if not np.isnan(sse_cong): sse_cong *= 100.0
        if not np.isnan(sse_disc): sse_disc *= 100.0
        sse['congruent']   = sse_cong
        sse['incongruent'] = sse_disc
        sse['total'] = np.nan if (np.isnan(sse_cong) or np.isnan(sse_disc)) else (sse_cong + sse_disc)

        if do_plot:
            ax.plot(stim, 100*beh_cong, '-o', label='data congruent',  color=colors['congruent'], alpha=0.85)
            ax.plot(stim, 100*beh_disc, '-o', label='data incongruent', color=colors['incongruent'], alpha=0.85)
            plt.tight_layout()
            if save_dir:
                plt.savefig(f'{save_dir}/performance_data_model_sym.svg', transparent=True)
    elif do_plot:
        plt.tight_layout()
        if save_dir:
            plt.savefig(f'{save_dir}/rt_data_model_sym.svg', transparent=True)

    return sse


def _loss_plot_diff_by_condition_with_data_torch(
    avg_dict, model_params, *, var_names, dt, reaction_time,
    mean_data_results, grad_options, scale_plot=False, save_dir=None
):
    grad_opts, dtype, device = _resolve_grad_options(model_params, override=grad_options)
    eps = torch.tensor(1e-12, dtype=dtype, device=device)

    def _tensor(value):
        return _ensure_tensor(value, dtype=dtype, device=device, requires_grad=False)

    def cond_key_stim(ts, ch, sp):
        stim = "stimL" if ts == -1 else "stimR"
        c = "cL" if ch == -1 else "cR"
        b = "bL" if sp == -1 else "bR"
        return f"{stim}{b}{c}"

    def cond_key_choice(ts, ch, sp):
        s = "sL" if ts == -1 else "sR"
        c = "choiceL" if ch == -1 else "choiceR"
        b = "bL" if sp == -1 else "bR"
        return f"{s}{b}{c}"

    total_sse = torch.zeros((), dtype=dtype, device=device)
    sse_I = torch.zeros((), dtype=dtype, device=device)
    sse_M = torch.zeros((), dtype=dtype, device=device)
    sse_P_neg = torch.zeros((), dtype=dtype, device=device)
    sse_P_diff = torch.zeros((), dtype=dtype, device=device)
    sse_raw = {
        'I': {'post': torch.zeros((), dtype=dtype, device=device),
              'pre': torch.zeros((), dtype=dtype, device=device)},
        'M': {'post': torch.zeros((), dtype=dtype, device=device),
              'pre': torch.zeros((), dtype=dtype, device=device)}
    }
    data_energy = {
        'I': {'post': torch.zeros((), dtype=dtype, device=device),
              'pre': torch.zeros((), dtype=dtype, device=device)},
        'M': {'post': torch.zeros((), dtype=dtype, device=device),
              'pre': torch.zeros((), dtype=dtype, device=device)}
    }
    iti_penalties = {
        'I': {'prev_ch_-1': _tensor(float('nan')), 'prev_ch_1': _tensor(float('nan'))},
        'M': {'prev_ch_-1': _tensor(float('nan')), 'prev_ch_1': _tensor(float('nan'))},
    }

    def _tensor_nan():
        return _tensor(float('nan'))

    def _nanmean(values):
        valid = [v for v in values if not torch.isnan(v).any().item()]
        if not valid:
            return _tensor_nan()
        return torch.mean(torch.stack(valid))
        
    def _interpolate_to_data_length(model_traj, data_length):
        """
        Interpolate model trajectory to match data length.
        Uses linear interpolation to upsample model output to data time grid.
        """
        if model_traj.shape[0] == data_length or model_traj.shape[0] == 0 or data_length == 0:
            return model_traj
        
        # Create target indices for interpolation
        target_indices = torch.linspace(0, model_traj.shape[0] - 1, data_length, dtype=dtype, device=device)
        interpolated = torch.zeros(data_length, dtype=dtype, device=device)
        
        for i, idx in enumerate(target_indices):
            idx_low = torch.clamp(torch.floor(idx).long(), 0, model_traj.shape[0] - 1)
            idx_high = torch.clamp(torch.ceil(idx).long(), 0, model_traj.shape[0] - 1)
            if idx_low == idx_high:
                interpolated[i] = model_traj[idx_low]
            else:
                w = idx - idx_low.float()
                interpolated[i] = (1 - w) * model_traj[idx_low] + w * model_traj[idx_high]
        
        return interpolated

    def _data_mean_and_baseline_t(
            vn,
            ch,
            window,
            baseline_len_I: int = 1,
            baseline_len_M: int = 1,
        ):
        if not mean_data_results or vn not in mean_data_results:
            return None, None
        mt_sets = mean_data_results[vn]["mean_traj"]
        mt_win = mt_sets.get(window, mt_sets) if isinstance(mt_sets, dict) else mt_sets
        mt_stim = mt_sets.get('stim', mt_sets) if isinstance(mt_sets, dict) else mt_sets

        diffs_win = []
        for ts0, sp0 in product((-1, 1), (-1, 1)):
            key = cond_key_stim(ts0, ch, sp0) if window == 'stim' else cond_key_choice(ts0, ch, sp0)
            arr = mt_win.get(key, None) if isinstance(mt_win, dict) else None
            if arr is None:
                continue
            arr_t = _tensor(arr)
            diff = arr_t[1] - arr_t[0] if (arr_t.ndim == 2 and arr_t.shape[0] == 2) else arr_t.reshape(-1)
            if diff.numel():
                diffs_win.append(diff)
        if not diffs_win:
            return None, None
        Tw = min(int(d.shape[0]) for d in diffs_win)
        data_mean = torch.mean(torch.stack([d[:Tw] for d in diffs_win], dim=0), dim=0)

        diffs_st = []
        for ts0, sp0 in product((-1, 1), (-1, 1)):
            key = cond_key_stim(ts0, ch, sp0)
            arr = mt_stim.get(key, None) if isinstance(mt_stim, dict) else None
            if arr is None:
                continue
            arr_t = _tensor(arr)
            diff = arr_t[1] - arr_t[0] if (arr_t.ndim == 2 and arr_t.shape[0] == 2) else arr_t.reshape(-1)
            if diff.numel():
                diffs_st.append(diff)
        if not diffs_st:
            return None, None
        Ts = min(int(d.shape[0]) for d in diffs_st)
        stim_mean = torch.mean(torch.stack([d[:Ts] for d in diffs_st], dim=0), dim=0)
        required_T = baseline_len_I if vn == 'I' else baseline_len_M
        if Ts < required_T:
            return None, None
        stim_mean = torch.mean(torch.stack([d[:Ts] for d in diffs_st], dim=0), dim=0)
        post_baseline = torch.mean(stim_mean[:required_T])
        return (data_mean - post_baseline), post_baseline

    for vn in var_names:
        cond_map = avg_dict.get(vn, {})
        if vn in ("I", "M"):
            for ch in (-1, 1):
                arr_post = cond_map.get(('post', ('ch', ch)), None)
                if arr_post is not None:
                    arr_post_t = _tensor(arr_post)
                    if arr_post_t.ndim >= 2:
                        model_diff_post = arr_post_t[1] - arr_post_t[0] if ch == 1 else arr_post_t[0] - arr_post_t[1]
                        data_post_norm, _ = _data_mean_and_baseline_t(vn, ch, 'stim')
                        if data_post_norm is not None:
                            # Interpolate model to match data length if needed
                            if model_diff_post.shape[0] < data_post_norm.shape[0]:
                                m_seg = _interpolate_to_data_length(model_diff_post, data_post_norm.shape[0])
                                d_seg = data_post_norm
                            else:
                                T_post = min(model_diff_post.shape[0], data_post_norm.shape[0])
                                m_seg = model_diff_post[:T_post]
                                d_seg = data_post_norm[:T_post]
                            if m_seg.shape[0] > 0:
                                skip_mask = torch.isnan(m_seg) | torch.isnan(d_seg)
                                if torch.any(skip_mask).item():
                                    nan_val = _tensor_nan()
                                    if vn == "I":
                                        sse_I = sse_I + nan_val
                                        sse_raw['I']['post'] = sse_raw['I']['post'] + nan_val
                                        data_energy['I']['post'] = data_energy['I']['post'] + nan_val
                                    else:
                                        sse_M = sse_M + nan_val
                                        sse_raw['M']['post'] = sse_raw['M']['post'] + nan_val
                                        data_energy['M']['post'] = data_energy['M']['post'] + nan_val
                                    total_sse = total_sse + nan_val
                                else:
                                    if m_seg.shape[0] > 15:
                                        m_eval = m_seg[15:]
                                        d_eval = d_seg[15:]
                                    else:
                                        m_eval, d_eval = m_seg, d_seg
                                    denom = torch.sum(d_eval * d_eval) + eps
                                    raw_sse = torch.sum((m_eval - d_eval) ** 2)
                                    sse_val = raw_sse / denom
                                    if vn == "I":
                                        sse_I = sse_I + sse_val
                                        sse_raw['I']['post'] = sse_raw['I']['post'] + raw_sse
                                        data_energy['I']['post'] = data_energy['I']['post'] + (denom - eps)
                                    else:
                                        sse_M = sse_M + sse_val
                                        sse_raw['M']['post'] = sse_raw['M']['post'] + raw_sse
                                        data_energy['M']['post'] = data_energy['M']['post'] + (denom - eps)
                                    total_sse = total_sse + sse_val

                arr_pre = cond_map.get(('pre', ('ch', ch)), None)
                if arr_pre is not None:
                    arr_pre_t = _tensor(arr_pre)
                    if arr_pre_t.ndim >= 2:
                        model_diff_pre = arr_pre_t[1] - arr_pre_t[0] if ch == 1 else arr_pre_t[0] - arr_pre_t[1]
                        data_pre_norm, _ = _data_mean_and_baseline_t(vn, ch, 'choice')
                        if data_pre_norm is not None:
                            # Interpolate model to match data length if needed
                            if model_diff_pre.shape[0] < data_pre_norm.shape[0]:
                                m_seg = _interpolate_to_data_length(model_diff_pre, data_pre_norm.shape[0])
                                d_seg = data_pre_norm
                            else:
                                T_pre = min(model_diff_pre.shape[0], data_pre_norm.shape[0])
                                m_seg = model_diff_pre[-T_pre:]
                                d_seg = data_pre_norm[-T_pre:]
                            if m_seg.shape[0] > 0:
                                skip_mask = torch.isnan(m_seg) | torch.isnan(d_seg)
                                if torch.any(skip_mask).item():
                                    nan_val = _tensor_nan()
                                    if vn == "I":
                                        sse_I = sse_I + nan_val
                                        sse_raw['I']['pre'] = sse_raw['I']['pre'] + nan_val
                                        data_energy['I']['pre'] = data_energy['I']['pre'] + nan_val
                                    else:
                                        sse_M = sse_M + nan_val
                                        sse_raw['M']['pre'] = sse_raw['M']['pre'] + nan_val
                                        data_energy['M']['pre'] = data_energy['M']['pre'] + nan_val
                                    total_sse = total_sse + nan_val
                                else:
                                    denom = torch.sum(d_seg * d_seg) + eps
                                    raw_sse = torch.sum((m_seg - d_seg) ** 2)
                                    sse_val = raw_sse / denom
                                    if vn == "I":
                                        sse_I = sse_I + sse_val
                                        sse_raw['I']['pre'] = sse_raw['I']['pre'] + raw_sse
                                        data_energy['I']['pre'] = data_energy['I']['pre'] + (denom - eps)
                                    else:
                                        sse_M = sse_M + sse_val
                                        sse_raw['M']['pre'] = sse_raw['M']['pre'] + raw_sse
                                        data_energy['M']['pre'] = data_energy['M']['pre'] + (denom - eps)
                                    total_sse = total_sse + sse_val

            arr_iti_prev = {
                -1: cond_map.get(('iti_prev', ('prev_ch', -1)), None),
                 1: cond_map.get(('iti_prev', ('prev_ch', 1)), None)
            }
            pen_sum = torch.zeros((), dtype=dtype, device=device)
            any_pen = False
            for prev_ch_sign, arr in arr_iti_prev.items():
                if arr is None:
                    continue
                arr_t = _tensor(arr)
                if arr_t.ndim != 2 or arr_t.shape[1] == 0:
                    continue
                abs_vals = torch.abs(arr_t)
                if torch.isfinite(abs_vals).any().item():
                    penalty = 4 * torch.nanmean(abs_vals)
                    iti_penalties[vn][f'prev_ch_{prev_ch_sign}'] = penalty
                    if torch.isfinite(penalty):
                        pen_sum = pen_sum + penalty
                        any_pen = True
            if any_pen:
                if vn == "I":
                    sse_I = sse_I + pen_sum
                else:
                    sse_M = sse_M + pen_sum
                total_sse = total_sse + pen_sum

        elif vn == "P":
            p_means = {}
            for sp in (-1, 1):
                arr = cond_map.get(sp, None)
                if arr is None:
                    continue
                arr_t = _tensor(arr)
                if arr_t.ndim >= 2 and arr_t.shape[1] >= 2:
                    model_diff = arr_t[1] - arr_t[0] if sp == 1 else arr_t[0] - arr_t[1]
                    p_means[sp] = torch.nanmean(model_diff)
            neg_pen = torch.zeros((), dtype=dtype, device=device)
            for mean_val in p_means.values():
                if torch.isfinite(mean_val) and (mean_val < 0):
                    neg_pen = neg_pen + mean_val ** 2
            diff_pen = torch.zeros((), dtype=dtype, device=device)
            if 1 in p_means and -1 in p_means:
                if torch.isfinite(p_means[1]) and torch.isfinite(p_means[-1]):
                    mean_diff = torch.abs(p_means[1] - p_means[-1])
                    denom = (torch.abs(p_means[1]) + torch.abs(p_means[-1])) / 2
                    denom = torch.where(denom > 0, denom, torch.ones_like(denom))
                    diff_pen = mean_diff / denom
            sse_P_neg = sse_P_neg + neg_pen
            sse_P_diff = sse_P_diff + diff_pen
            total_sse = total_sse + neg_pen + diff_pen

    def _gof(vn, win):
        num = sse_raw[vn][win]
        den = data_energy[vn][win]
        good = den > 0
        return torch.where(good, 1.0 - (num / den), _tensor_nan())

    gof_I_post = _gof('I', 'post')
    gof_I_pre = _gof('I', 'pre')
    gof_M_post = _gof('M', 'post')
    gof_M_pre = _gof('M', 'pre')

    return {
        'I': sse_I,
        'M': sse_M,
        'P_neg': sse_P_neg,
        'P_diff': sse_P_diff,
        'total': total_sse,
        'gof': {
            'I': {'post': gof_I_post, 'pre': gof_I_pre,
                  'total': _nanmean([gof_I_post, gof_I_pre])},
            'M': {'post': gof_M_post, 'pre': gof_M_pre,
                  'total': _nanmean([gof_M_post, gof_M_pre])}
        },
        'debug': {
            'energy': data_energy,
            'sse_raw': sse_raw,
            'iti_penalty': iti_penalties
        }
    }


def loss_plot_diff_by_condition_with_data(
    avg_dict, model_params, var_names=("S","I","P","M"), dt=None,
    reaction_time=None, mean_data_results=None, plot=True, save_dir=None,
    gradient_mode=False, grad_options=None
):
    """
    I = dark golds, M = dark oranges, P = purple.
    Data = solid, Model = dashed (same color).

    Returns:
        dict with SSE components and GoF values:
          {'I': nSSE_I, 'M': nSSE_M, 'P_neg': nPen_neg, 'P_diff': nPen_diff, 
           'total': sum, 'gof': {'I': {'post': ..., 'pre': ..., 'total': ...},
                                 'M': {'post': ..., 'pre': ..., 'total': ...}},
           'debug': {'energy': {...}, 'sse_raw': {...}} }
    Notes:
        Adds an intertrial penalty for I and M based on the mean absolute magnitude
        of each previous-choice-conditioned ITI trace (pushing ITI activity toward zero).
    """
    # Get dt from parameter, model_params, or global default
    if dt is None:
        dt = _get_dt_from_model_params(model_params)

    if gradient_mode:
        result = _loss_plot_diff_by_condition_with_data_torch(
            avg_dict, model_params, var_names=var_names, dt=dt,
            reaction_time=reaction_time, mean_data_results=mean_data_results,
            grad_options=grad_options or model_params.get('grad_options', {}),
            scale_plot=plot, save_dir=save_dir)
        if plot:
            avg_dict_np = _detach_to_numpy(avg_dict)
            model_params_np = _detach_to_numpy(model_params)
            # Produce plots using the non-gradient path (no effect on autograd graph)
            loss_plot_diff_by_condition_with_data(
                avg_dict_np,
                model_params_np,
                var_names=var_names,
                dt=dt,
                reaction_time=reaction_time,
                mean_data_results=mean_data_results,
                plot=True,
                save_dir=save_dir,
                gradient_mode=False,
                grad_options=grad_options,
            )
        return result

    # ---- colors ----
    I_COLORS = {-1: "#DAA520", 1: "#FFD700"}   # dark golds
    M_COLORS = {-1: "#CC5500", 1: "#FF7F0E"}   # dark oranges
    P_COLORS = {-1: "#6A3D9A", 1: "#8B5FBF"}   # purple shades

    # ---- helpers for data keys ----
    def cond_key_stim(ts, ch, sp):
        """Key for post-start (stim) window."""
        stim = "stimL" if ts == -1 else "stimR"
        c    = "cL"    if ch == -1 else "cR"
        b    = "bL"    if sp == -1 else "bR"
        return f"{stim}{b}{c}"

    def cond_key_choice(ts, ch, sp):
        """Key for pre-action (choice) window."""
        s = "sL" if ts == -1 else "sR"
        c = "choiceL" if ch == -1 else "choiceR"
        b = "bL" if sp == -1 else "bR"
        return f"{s}{b}{c}"

    eps = 1e-12
    total_sse = 0.0
    sse_terms = {'I': 0.0, 'M': 0.0, 'P_neg': 0.0, 'P_diff': 0.0}
    sse_raw   = {'I': {'post': 0.0, 'pre': 0.0}, 'M': {'post': 0.0, 'pre': 0.0}}
    data_energy = {'I': {'post': 0.0, 'pre': 0.0}, 'M': {'post': 0.0, 'pre': 0.0}}
    iti_penalties = {'I': {'prev_ch_-1': np.nan, 'prev_ch_1': np.nan},
                     'M': {'prev_ch_-1': np.nan, 'prev_ch_1': np.nan}}

    if plot:
        fig_post, ax_post = plt.subplots(figsize=(184/72, 155/72))
        fig_pre,  ax_pre  = plt.subplots(figsize=(184/72, 155/72))
        fig_p, ax_p = None, None

    # ---- helper to compute normalized data and baseline ----
    def _data_mean_and_baseline(
        vn,
        ch,
        window: str,
        baseline_len_I: int = 1,
        baseline_len_M: int = 1,
        ):
        """
        window ∈ {'stim','choice'}
        Returns (data_mean_norm, post_baseline) where:
        • post_baseline = mean of the first baseline_len_I samples from the STIM window
            when vn == 'I', or the first baseline_len_M samples otherwise
            (averaged over ts, sp at fixed ch)
        • data_mean_norm = <mean over ts,sp for the requested window> minus post_baseline
        Raises:
        ValueError if the STIM window has fewer samples than required to define the baseline.
        """
        if not (mean_data_results and vn in mean_data_results):
            return None, None

        mt_sets = mean_data_results[vn]["mean_traj"]
        mt_win  = mt_sets.get(window, mt_sets) if isinstance(mt_sets, dict) else mt_sets
        mt_stim = mt_sets.get('stim', mt_sets) if isinstance(mt_sets, dict) else mt_sets

        # collect diffs for requested window
        diffs_win = []
        for ts0, sp0 in product((-1, 1), (-1, 1)):
            key_w = cond_key_stim(ts0, ch, sp0) if window == 'stim' else cond_key_choice(ts0, ch, sp0)
            arr_w = mt_win.get(key_w, None) if isinstance(mt_win, dict) else None
            if arr_w is None:
                continue
            aw = np.asarray(arr_w)
            d  = (aw[1] - aw[0]) if (aw.ndim == 2 and aw.shape[0] == 2) else aw.reshape(-1)
            if d.size:
                diffs_win.append(d.astype(float))
        if not diffs_win:
            return None, None
        Tw = min(len(d) for d in diffs_win)
        data_mean = np.mean([d[:Tw] for d in diffs_win], axis=0)

        # compute baseline from STIM window (independent of window)
        diffs_st = []
        for ts0, sp0 in product((-1, 1), (-1, 1)):
            key_s = cond_key_stim(ts0, ch, sp0)
            arr_s = mt_stim.get(key_s, None) if isinstance(mt_stim, dict) else None
            if arr_s is None:
                continue
            as_ = np.asarray(arr_s)
            ds  = (as_[1] - as_[0]) if (as_.ndim == 2 and as_.shape[0] == 2) else as_.reshape(-1)
            if ds.size:
                diffs_st.append(ds.astype(float))
        if not diffs_st:
            return None, None
        Ts = min(len(d) for d in diffs_st)
        required_T = baseline_len_I if vn == 'I' else baseline_len_M
        if Ts < required_T:
            raise ValueError(
                f"Post-start baseline undefined for {vn}, ch={ch}: "
                f"need ≥{required_T} samples in 'stim' window."
            )
        stim_mean = np.mean([d[:Ts] for d in diffs_st], axis=0)
        post_baseline = float(np.mean(stim_mean[:required_T]))
        return (data_mean - post_baseline), post_baseline

    # --- main per-variable loop ---
    for vn in var_names:
        cond_map = avg_dict.get(vn, {})

        if vn in ("I", "M"):
            col_map = I_COLORS if vn == "I" else M_COLORS

            for ch in (-1, 1):
                # --- POST-START window (stim) ---
                arr_post = cond_map.get(('post', ('ch', ch)), None)
                if arr_post is not None and arr_post.size:
                    model_diff_post = (arr_post[1] - arr_post[0]) if ch == 1 else (arr_post[0] - arr_post[1])
                    Tm_post = model_diff_post.shape[0]

                    data_post_norm, _ = _data_mean_and_baseline(vn, ch, 'stim')
                    if data_post_norm is not None:
                        T_post = min(Tm_post, data_post_norm.shape[0])
                        if T_post > 0:
                            # first T_post bins after start
                            m_seg = model_diff_post[:T_post]
                            d_seg = data_post_norm[:T_post]
                            # exclude first 15 bins for SSE
                            if not (np.any(np.isnan(m_seg)) or np.any(np.isnan(d_seg))):
                                if T_post > 15:
                                    m_eval = m_seg[15:]
                                    d_eval = d_seg[15:]
                                else:
                                    m_eval, d_eval = m_seg, d_seg
                                denom   = float(np.sum(d_eval**2) + eps)
                                raw_sse = float(np.sum((m_eval - d_eval)**2))
                                sse_val = raw_sse / denom
                                sse_terms[vn] += sse_val
                                total_sse     += sse_val
                                sse_raw[vn]['post']   += raw_sse
                                data_energy[vn]['post'] += (denom - eps)
                            if plot:
                                t_post = np.arange(T_post) * dt
                                ax_post.plot(t_post, m_seg, '--', linewidth=2, color=col_map[ch])
                                ax_post.plot(t_post, d_seg, '-', linewidth=2, alpha=0.95, color=col_map[ch])
                    elif plot:
                        t_post = np.arange(Tm_post) * dt
                        ax_post.plot(t_post, model_diff_post, '--', linewidth=2, color=col_map[ch])

                # --- PRE-ACTION window (choice) ---
                arr_pre = cond_map.get(('pre', ('ch', ch)), None)
                if arr_pre is not None and arr_pre.size:
                    model_diff_pre = (arr_pre[1] - arr_pre[0]) if ch == 1 else (arr_pre[0] - arr_pre[1])
                    Tm_pre = model_diff_pre.shape[0]

                    data_pre_norm, _ = _data_mean_and_baseline(vn, ch, 'choice')
                    if data_pre_norm is not None:
                        T_pre = min(Tm_pre, data_pre_norm.shape[0])
                        if T_pre > 0:
                            m_seg = model_diff_pre[-T_pre:]
                            d_seg = data_pre_norm[-T_pre:]
                            has_nan = np.any(np.isnan(m_seg)) or np.any(np.isnan(d_seg))
                            if has_nan:
                                sse_terms[vn] += np.nan
                                total_sse     += np.nan
                                sse_raw[vn]['pre']   += np.nan
                                data_energy[vn]['pre'] += np.nan
                            else:
                                denom   = float(np.sum(d_seg**2) + eps)
                                raw_sse = float(np.sum((m_seg - d_seg)**2))
                                sse_val = raw_sse / denom
                                sse_terms[vn] += sse_val
                                total_sse     += sse_val
                                sse_raw[vn]['pre']   += raw_sse
                                data_energy[vn]['pre'] += (denom - eps)
                            if plot:
                                t_full = np.arange(-Tm_pre + 1, 1) * dt
                                t_pre  = t_full[-T_pre:]
                                ax_pre.plot(t_pre, m_seg, '--', linewidth=2, color=col_map[ch])
                                ax_pre.plot(t_pre, d_seg, '-', linewidth=2, alpha=0.95, color=col_map[ch])
                    elif plot:
                        t_full = np.arange(-Tm_pre + 1, 1) * dt
                        ax_pre.plot(t_full, model_diff_pre, '--', linewidth=2, color=col_map[ch])

            # --- Intertrial window penalty (conditioned on previous choice) ---
            arr_iti_prev = {
                -1: cond_map.get(('iti_prev', ('prev_ch', -1)), None),
                 1: cond_map.get(('iti_prev', ('prev_ch', 1)),  None)
            }
            pen_sum = 0.0
            any_pen = False
            for prev_ch_sign, arr in arr_iti_prev.items():
                if arr is None or arr.size == 0:
                    continue
                arr = np.asarray(arr, dtype=float)
                if arr.ndim != 2 or arr.shape[1] == 0:
                    continue
                abs_vals = np.abs(arr[:, :arr.shape[1]])
                if not np.isfinite(abs_vals).any():
                    continue
                penalty = 4*float(np.nanmean(abs_vals))
                iti_penalties[vn][f'prev_ch_{prev_ch_sign}'] = penalty
                if np.isfinite(penalty):
                    pen_sum += penalty
                    any_pen = True
            if any_pen:
                sse_terms[vn] += pen_sum
                total_sse += pen_sum

        # --- P (prior) ---
        elif vn == "P":
            p_means = {}
            if plot and ax_p is None:
                fig_p, ax_p = plt.subplots(figsize=(3.8, 3.0))
            for sp in (-1, 1):
                arr = cond_map.get(sp, None)
                if arr is None or arr.size == 0:
                    continue
                model_diff = (arr[1] - arr[0]) if sp == 1 else (arr[0] - arr[1])
                T_eff = model_diff.shape[0]
                t = np.arange(-T_eff + 1, 1) * dt
                if plot:
                    ax_p.plot(t, model_diff, '--', linewidth=2, color=P_COLORS[sp])
                p_means[sp] = float(np.nanmean(model_diff)) if model_diff.size else np.nan

            neg_pen = 0.0
            for sp, m in p_means.items():
                if np.isfinite(m) and m < 0:
                    neg_pen += (m**2)
            diff_pen = 0.0
            if (1 in p_means) and (-1 in p_means) and np.isfinite(p_means[1]) and np.isfinite(p_means[-1]):
                mean_diff = abs(p_means[1] - p_means[-1])
                denom = (abs(p_means[1]) + abs(p_means[-1]))/2
                diff_pen = mean_diff / (denom if denom > 0 else 1.0)
            sse_terms['P_neg'] += neg_pen
            sse_terms['P_diff'] += diff_pen
            total_sse += (neg_pen + diff_pen)

        # --- S (stimulus) ---
        elif vn == "S":
            for ts, ch, sp in product((-1, 1), repeat=3):
                if ch != ts:
                    continue
                arr = cond_map.get((ts, ch, sp), None)
                if arr is None or arr.size == 0:
                    continue
                diff_t = (arr[1] - arr[0]) if ts == 1 else (arr[0] - arr[1])
                diff_t = diff_t - diff_t[0]
                T_eff = diff_t.shape[0]
                t = np.arange(T_eff) * dt
                if plot:
                    ax_post.plot(t, diff_t, '-o', markersize=1.5, linewidth=1)

    # ---- compute R² ----
    def _r2(vn, win):
        num, den = sse_raw[vn][win], data_energy[vn][win]
        return np.nan if den <= 0 else 1.0 - (num / den)

    gof_I_post, gof_I_pre = _r2('I', 'post'), _r2('I', 'pre')
    gof_M_post, gof_M_pre = _r2('M', 'post'), _r2('M', 'pre')
    sse_terms['total'] = total_sse
    sse_terms['gof'] = {
        'I': {'post': gof_I_post, 'pre': gof_I_pre,
              'total': np.nanmean([gof_I_post, gof_I_pre]) if not (np.isnan(gof_I_post) and np.isnan(gof_I_pre)) else np.nan},
        'M': {'post': gof_M_post, 'pre': gof_M_pre,
              'total': np.nanmean([gof_M_post, gof_M_pre]) if not (np.isnan(gof_M_post) and np.isnan(gof_M_pre)) else np.nan}
    }
    sse_terms['debug'] = {'energy': data_energy, 'sse_raw': sse_raw, 'iti_penalty': iti_penalties}

    # ---- plotting cleanup ----
    if plot:
        if ax_post is not None:
            ax_post.axvline(0, color='k', linewidth=1)
            ax_post.spines['top'].set_visible(False)
            ax_post.spines['right'].set_visible(False)
            ax_post.set_xlabel(f"time ({'ms' if dt != 1.0 else 'steps'})")
            # ax_post.set_title("Post-start (I & M)")
            if np.isfinite(gof_I_post):
                ax_post.text(0.10, 0.90, f"R² {gof_I_post:.3f}", transform=ax_post.transAxes,
                             fontsize=10, color="#DAA520", ha="left", va="top")
            if np.isfinite(gof_M_post):
                ax_post.text(0.10, 0.82, f"R² {gof_M_post:.3f}", transform=ax_post.transAxes,
                             fontsize=10, color="#CC5500", ha="left", va="top")

        if ax_pre is not None:
            ax_pre.axvline(0, color='k', linewidth=1)
            if reaction_time is not None:
                rt_vals = np.asarray(reaction_time, dtype=float)
                if rt_vals.size > 0:
                    mean_rt = np.mean(rt_vals) * dt
                    ax_pre.axvline(-mean_rt, color='r', linewidth=1, label="trial start")
            ax_pre.spines['top'].set_visible(False)
            ax_pre.spines['right'].set_visible(False)
            ax_pre.set_xlabel(f"time ({'ms' if dt != 1.0 else 'steps'})")
            # ax_pre.set_title("Pre-action (I & M)")
            if np.isfinite(gof_I_pre):
                ax_pre.text(0.10, 0.90, f"R² {gof_I_pre:.3f}", transform=ax_pre.transAxes,
                            fontsize=10, color="#DAA520", ha="left", va="top")
            if np.isfinite(gof_M_pre):
                ax_pre.text(0.10, 0.82, f"R² {gof_M_pre:.3f}", transform=ax_pre.transAxes,
                            fontsize=10, color="#CC5500", ha="left", va="top")

        if ax_p is not None:
            ax_p.axvline(0, color='k', linewidth=1)
            ax_p.spines['top'].set_visible(False)
            ax_p.spines['right'].set_visible(False)
            ax_p.set_xlabel(f"time ({'ms' if dt != 1.0 else 'steps'})")
            ax_p.set_title("Prior (P)")

        plt.tight_layout()
        if save_dir:
            param_name = (
                f"gi{model_params['g_i']}_gm{model_params['g_m']}_gs{model_params['g_s']}"
                f"_di{model_params['d_i']}_dm{model_params['d_m']}_ds{model_params['d_s']}"
            )
            param_name += f"thr{model_params['action_thresholds']['concordant'][0]}_" \
                          f"{model_params['action_thresholds']['discordant'][0]}"
            fig_post.savefig(f'{save_dir}/IM_post_fit_{param_name}.svg', transparent=True)
            fig_pre.savefig(f'{save_dir}/IM_pre_fit_{param_name}.svg', transparent=True)
            if fig_p is not None:
                fig_p.savefig(f'{save_dir}/P_fit_{param_name}.svg', transparent=True)

    return sse_terms


def _loss_prior_effect_torch(
    regions, results, model_params, steps_before_obs, T, model_metric,
    timeframes, alpha, ptype, plot_window, shift_baseline,
    scale_factors, include_all_trials, lump_all, correction, grad_options
):
    grad_opts, dtype, device = _resolve_grad_options(model_params, override=grad_options)
    eps = torch.tensor(1e-12, dtype=dtype, device=device)

    model_dists = _prior_distance_I_M_both_alignments_torch(
        results, steps_before_obs, T=T, metric=model_metric,
        include_all_trials=include_all_trials, lump_all=lump_all,
        dtype=dtype, device=device
    )

    if scale_factors is None:
        s_fac, i_fac, m_fac = (1.0, 1.0, 1.0)
    else:
        if len(scale_factors) != 3:
            raise ValueError("scale_factors must be [S_factor, I_factor, M_factor]")
        s_fac, i_fac, m_fac = scale_factors
    s_fac = torch.tensor(float(s_fac), dtype=dtype, device=device)
    i_fac = torch.tensor(float(i_fac), dtype=dtype, device=device)
    m_fac = torch.tensor(float(m_fac), dtype=dtype, device=device)

    def _align_baseline_torch(y_model, y_data, baseline_mode="min"):
        if not shift_baseline:
            return y_data
        if y_model is None or y_data is None or y_model.numel() == 0 or y_data.numel() == 0:
            return y_data
        if torch.isnan(y_model).all().item() or torch.isnan(y_data).all().item():
            return y_data
        m_min = torch.nanmin(y_model)
        if baseline_mode == "min":
            d_base = torch.nanmean(y_data[:5]) if y_data.numel() >= 5 else torch.nanmean(y_data)
        else:
            d_base = torch.nanmean(y_data)
        return y_data - d_base + m_min

    total_loss = torch.zeros((), dtype=dtype, device=device)
    out = {'total': total_loss}

    # Get dt from model_params (no default needed - should always be in model_params)
    dt_value = _get_dt_from_model_params(model_params)
    
    for timeframe in timeframes:
        is_choice = ('duringchoice' in timeframe)
        align_key = 'action' if is_choice else 'start'

        if not Path(f'data_{timeframe}.npy').exists():
            raise FileNotFoundError(f"Missing cached data for timeframe {timeframe}. Run with reload=True once.")
        save_data = np.load(f'data_{timeframe}.npy', allow_pickle=True).flat[0]
        r_int = torch.tensor(np.asarray(save_data['r_int'], dtype=float), dtype=dtype, device=device)
        r_move = torch.tensor(np.asarray(save_data['r_move'], dtype=float), dtype=dtype, device=device)
        if not is_choice:
            r_stim_np = save_data['r_stim']
            r_stim = None if r_stim_np is None else torch.tensor(np.asarray(r_stim_np, dtype=float), dtype=dtype, device=device)
        else:
            r_stim = None

        m_I = i_fac * _ensure_tensor(model_dists['I'][align_key], dtype=dtype, device=device)
        m_M = m_fac * _ensure_tensor(model_dists['M'][align_key], dtype=dtype, device=device)
        m_S = None
        if not is_choice and 'S' in model_dists:
            m_S = s_fac * _ensure_tensor(model_dists['S'][align_key], dtype=dtype, device=device)

        # plot_window is now in milliseconds. Calculate number of bins for model based on dt
        # Use data's actual length for comparison (data bin size is ~2.08ms)
        data_length = int(r_int.shape[0]) if r_int.numel() > 0 else 0
        # Calculate expected model bins for the plot_window duration
        model_bins = int(round(plot_window / dt_value))
        # Use data length for times_full (since we compare with data)
        # times_full is mainly used for truncation, so use data length
        n_bins = max(data_length, model_bins) if data_length > 0 else model_bins
        times_full = torch.linspace(-2*plot_window if is_choice else 0,
                                    0 if is_choice else 2*plot_window,
                                    steps=n_bins, dtype=dtype, device=device)

        def _interpolate_torch(model_traj, target_length):
            """Interpolate model trajectory to target length using linear interpolation."""
            if model_traj.shape[0] == target_length or model_traj.shape[0] == 0 or target_length == 0:
                return model_traj
            model_indices = torch.linspace(0, model_traj.shape[0] - 1, model_traj.shape[0], dtype=dtype, device=device)
            target_indices = torch.linspace(0, model_traj.shape[0] - 1, target_length, dtype=dtype, device=device)
            interpolated = torch.zeros(target_length, dtype=dtype, device=device)
            for i, idx in enumerate(target_indices):
                idx_low = torch.clamp(torch.floor(idx).long(), 0, model_traj.shape[0] - 1)
                idx_high = torch.clamp(torch.ceil(idx).long(), 0, model_traj.shape[0] - 1)
                if idx_low == idx_high:
                    interpolated[i] = model_traj[idx_low]
                else:
                    w = idx - idx_low.float()
                    interpolated[i] = (1 - w) * model_traj[idx_low] + w * model_traj[idx_high]
            return interpolated

        def _proc(m_model, r_data, baseline_mode):
            if m_model is None or r_data is None or m_model.numel() == 0 or r_data.numel() == 0:
                return torch.tensor(float('nan'), dtype=dtype, device=device)
            # Interpolate model to match data length if needed
            if m_model.shape[0] < r_data.shape[0]:
                m_model = _interpolate_torch(m_model, r_data.shape[0])
            L = min(int(m_model.shape[0]), int(r_data.shape[0]), int(times_full.shape[0]))
            if L <= 0:
                return torch.tensor(float('nan'), dtype=dtype, device=device)
            if is_choice:
                y_m = m_model[-L:]
                y_d_src = r_data[-L:]
            else:
                y_m = m_model[:L]
                y_d_src = r_data[:L]
            y_d = _align_baseline_torch(y_m, y_d_src, baseline_mode=baseline_mode)
            if torch.isnan(y_m).any().item() or torch.isnan(y_d).any().item():
                return torch.tensor(float('nan'), dtype=dtype, device=device)
            denom = torch.sum(y_d * y_d) + eps
            return torch.sum((y_m - y_d) ** 2) / denom

        sse_int = _proc(m_I, r_int, baseline_mode="min")
        sse_move = _proc(m_M, r_move, baseline_mode="min")
        sse_stim = torch.tensor(0.0, dtype=dtype, device=device) if is_choice else (
            _proc(m_S, r_stim, baseline_mode="mean") if m_S is not None and r_stim is not None else torch.tensor(0.0, dtype=dtype, device=device)
        )

        losses = [v for v in (sse_int, sse_move, sse_stim) if not torch.isnan(v)]
        if len(losses) > 0:
            frame_loss = torch.sum(torch.stack(losses))
            total_loss = total_loss + frame_loss
        out[timeframe] = {
            'integrator': sse_int,
            'move': sse_move,
            'stim': sse_stim,
            'total': torch.sum(torch.stack(losses)) if len(losses) > 0 else torch.tensor(float('nan'), dtype=dtype, device=device)
        }

    out['total'] = total_loss
    return out


def corrected_aicc_from_total_loss(prior_out, im_out, k, n_prior=160, n_im=(57+72)*4, include_im=True):
    """
    Compute AICc using total normalized losses (monotonic with SSE totals).
    Ensures that smaller total loss -> smaller AIC.
    """
    eps = 1e-12
    total_loss = float(prior_out.get('total', np.nan))
    if include_im:
        total_loss += float(im_out.get('I', 0.0)) + float(im_out.get('M', 0.0))

    N = int(n_prior + (n_im if include_im else 0))
    if not (np.isfinite(total_loss) and N > 0):
        return {'aic': np.nan, 'aicc': np.nan}

    # Gaussian AIC: 2k + N * ln(SSE/N)
    aic = 2 * int(k) + N * np.log(max(total_loss, eps) / N)
    denom = N - int(k) - 1
    aicc = aic + (2 * int(k) * (int(k) + 1)) / denom if denom > 0 else np.nan

    return {'aic': aic, 'aicc': aicc, 'components': {'total_loss': total_loss, 'k': k, 'N': N}}


# ---- plotting functions, distance curves ----

def load_and_plot_stim_dist_data(regs, splits, plot_individual=False):
    """
    Regional data
    Load and plot stimulus distance curves for each contrast.
    """

    times = np.linspace(0, 0.15, 72)

    # Plot average across regions
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)

    avg_data = {}
    for split in splits:
        contrast = float(split.split('_')[-1])
        dists = None
        all_regs_cell_num = 0

        try:
            data = np.load(Path(pth_res, f'{split}.npy'), allow_pickle=True).flatten()[0]
        except Exception:
            print("error loading:", split)
            continue

        for reg in regs:
            if reg not in data:
                print(f"missing region {reg} in {split}")
                continue
            nclus = data[reg]['nclus']
            if dists is None:
                dists = (data[reg]['d_euc']**2) * nclus
            else:   
                dists += (data[reg]['d_euc']**2) * nclus
            all_regs_cell_num += nclus

        mean_d = (dists / all_regs_cell_num)**0.5
        avg_data[contrast] = mean_d
        ax.plot(times, mean_d, label=contrast)

    ax.set_title('Avg R-L Distance Across Regions')
    ax.set(xlabel='time (ms)', ylabel='Euclidean Distance')
    ax.legend(title='Contrast')
    fig.tight_layout()
    fig.savefig(f'{save_dir}/avg_d_stim_by_contrast.pdf', transparent=True)
    plt.close()

    # Plot individual regions
    if plot_individual:
        for reg in regs:
            fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
            all_contrasts_data = []
            for split in splits:
                contrast = float(split.split('_')[-1])
                try:
                    data = np.load(Path(pth_res, f'{split}.npy'), allow_pickle=True).flatten()[0]
                    d = data[reg]['d_euc']
                    ax.plot(times, d, label=contrast)
                    all_contrasts_data.append(d)
                except Exception:
                    print(f"error: {split}, {reg}")
                    continue

            mean_all_contrasts = np.mean(all_contrasts_data, axis=0)
            ax.plot(times, mean_all_contrasts, label='Avg over contrasts', linestyle='--', color='black')

            ax.set_title(reg)
            ax.set(xlabel='time (ms)', ylabel='Euclidean Distance')
            ax.legend(title='Contrast')
            fig.tight_layout()
            fig.savefig(f'{save_dir}/{reg}_d_stim_by_contrast.pdf', transparent=True)
            plt.close()
    
    return avg_data



def plot_S_dist_by_contrast(avg_dict, avg_data, dt=2, scale_factors=None, time_shifts_steps=None, 
                            save_dir=save_dir, ylim=None, yticks=None):
    """
    avg_dict: {(side, contrast): array (2, T_eff)}  # model stim vectors
    avg_data: {contrast: array (T_eff,)}            # data distance curves (already ||·||2)
    scale_factors: {contrast: float}                # scale model distance
    time_shifts_steps: {contrast: int}              # shift model curve by K steps (K>0 => delay)

    Plots:
      1) Per-contrast overlay: data (solid, baseline-shifted) vs model (dashed, scaled, step-shifted),
         with contrasts listed in text (bottom-to-top) in matching colors at top-left.
      2) Average across contrasts: applies the same step shifts and trims to the shortest common window.
      Adds R² goodness-of-fit (per contrast and average).
    """
    contrasts = sorted({c for (_, c) in avg_dict.keys()})
    colors = cm.Blues(np.linspace(0.4, 1.0, len(contrasts)))[::-1]  # shades of blue

    # ---------- Per-contrast overlay ----------
    plt.figure(figsize=(4, 3))
    per_contrast = []
    all_model_concat = []
    all_data_concat = []

    for idx, c in enumerate(contrasts):
        arr_L = avg_dict.get((-1, c))
        arr_R = avg_dict.get((+1, c))
        data_dist = avg_data.get(c)
        if arr_L is None or arr_R is None or data_dist is None:
            continue

        T_eff = min(arr_L.shape[-1], arr_R.shape[-1], data_dist.shape[-1])
        dist_model = np.linalg.norm(arr_R[:, :T_eff] - arr_L[:, :T_eff], axis=0)
        if scale_factors and c in scale_factors:
            dist_model = dist_model * float(scale_factors[c])

        k = int(time_shifts_steps.get(c, 0)) if time_shifts_steps else 0
        data_dist = data_dist[:T_eff] - np.min(data_dist[:T_eff])

        t_data  = np.arange(T_eff) * dt
        t_model = (np.arange(T_eff) + k) * dt
        col = colors[idx]

        plt.plot(t_data,  data_dist,  '-',  linewidth=2.0, color=col)
        plt.plot(t_model, dist_model, '--', linewidth=2.0, color=col)

        # --- compute per-contrast R² with alignment for shift k ---
        if k >= 0:
            start_m, start_d = 0, k
            L = T_eff - k
        else:
            start_m, start_d = -k, 0
            L = T_eff + k
        if L > 0:
            m_seg = dist_model[start_m:start_m + L]
            d_seg = data_dist[start_d:start_d + L]
            eps = 1e-12
            sse = float(np.sum((m_seg - d_seg) ** 2))
            sst = float(np.sum((d_seg - np.nanmean(d_seg)) ** 2) + eps)
            r2 = (1.0 - sse / sst) if sst > 0 else np.nan
            # collect aligned segments for total R²
            all_model_concat.append(m_seg)
            all_data_concat.append(d_seg)
        else:
            r2 = np.nan

        per_contrast.append({
            "contrast": c,
            "k": k,
            "T": T_eff,
            "model": dist_model,
            "data": data_dist,
            "color": col,
            "r2": r2
        })

    if per_contrast:
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim_ = ax.get_ylim()
        x0, y0 = xlim[0], ylim_[1]
        spacing = 0.08 * (ylim_[1] - ylim_[0])
        for idx, pc in enumerate(reversed(per_contrast)):
            label = (
                f"c={pc['contrast']}  R²="
                + ("nan" if np.isnan(pc["r2"]) else f"{pc['r2']:.3f}")
            )
            ax.text(
                x0 + 0.02 * (xlim[1] - xlim[0]),
                y0 - 0.1 - idx * spacing,
                label,
                color=pc["color"],
                fontsize=12,
                ha="left",
                va="top",
            )

        # --- Compute total R² across all contrasts ---
        if all_model_concat and all_data_concat:
            all_model = np.concatenate(all_model_concat)
            all_data = np.concatenate(all_data_concat)
            eps = 1e-12
            sse_total = float(np.sum((all_model - all_data) ** 2))
            sst_total = float(np.sum((all_data - np.nanmean(all_data)) ** 2) + eps)
            total_r2 = (1.0 - sse_total / sst_total) if sst_total > 0 else np.nan
            ax.text(0.98, 0.92, f"Total R² = {total_r2:.3f}",
                    transform=ax.transAxes, fontsize=10,
                    color='black', ha='right')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel(f"time ({'ms' if dt != 1.0 else 'steps'})", fontsize=12)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        if ylim is not None:
            plt.ylim(ylim)
        if yticks is not None:
            plt.yticks(yticks)
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(f'{save_dir}/S_dist_model_data_by_contrast.svg', transparent=True)

    # ---------- Average across contrasts ----------
    if not per_contrast:
        return

    max_k = max(pc["k"] for pc in per_contrast)
    global_end = min((pc["T"] - 1 if pc["k"] >= 0 else pc["k"] + pc["T"] - 1)
                     for pc in per_contrast)
    start_idx = max(0, max_k)
    end_idx = global_end
    if end_idx < start_idx:
        return

    L = end_idx - start_idx + 1
    stack_model, stack_data = [], []
    for pc in per_contrast:
        k, T = pc["k"], pc["T"]
        model, data = pc["model"], pc["data"]
        m_start = start_idx - k
        m_end   = m_start + L
        d_start = start_idx
        d_end   = d_start + L
        if m_start < 0 or m_end > T or d_end > T:
            continue
        stack_model.append(model[m_start:m_end])
        stack_data.append(data[d_start:d_end])

    if not stack_model or not stack_data:
        return

    stack_model = np.vstack(stack_model)
    stack_data  = np.vstack(stack_data)

    avg_model = np.mean(stack_model, axis=0)
    avg_data_curve = np.mean(stack_data, axis=0)
    avg_data_curve = avg_data_curve - np.min(avg_data_curve)

    t = (start_idx + np.arange(L)) * dt
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(t, avg_data_curve, '-',  linewidth=1, color='C0', label="data avg (baseline 0)")
    ax.plot(t, avg_model,      '--', linewidth=1, color='C0', label="model avg (shifted)")
    ax.set_xlabel(f"time ({'ms' if dt != 1.0 else 'steps'})", fontsize=12)
    ax.set_ylabel(r"Avg $\|\mathrm{S}_{+1}(t)-\mathrm{S}_{-1}(t)\|_2$")
    ax.legend()

    # Compute R² for averaged curves
    eps = 1e-12
    sse_avg = float(np.sum((avg_model - avg_data_curve) ** 2))
    sst_avg = float(np.sum((avg_data_curve - np.nanmean(avg_data_curve)) ** 2) + eps)
    r2_avg = (1.0 - sse_avg / sst_avg) if sst_avg > 0 else np.nan
    ax.text(0.02, 0.92, f"R² = {r2_avg:.3f}", transform=ax.transAxes, fontsize=10, color='C0')

    # Hide top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(f'{save_dir}/S_dist_model_data_avg.svg', transparent=True)


def sse_S_dist_by_contrast(avg_dict, avg_data):
    """
    Compute SSE between model and data distance curves for S, per contrast and for
    the average curve. No scale factors or time shifts are applied.

    avg_dict: {(side, contrast): array (2, T_eff)}   # model stim vectors
    avg_data: {contrast: array (T_eff,)}             # data distance curves (||·||2), baseline-shifted here
    """

    contrasts = sorted({c for (_, c) in avg_dict.keys()})
    per_sse = {}
    model_list, data_list = [], []   # <-- fix

    for c in contrasts:
        arr_L = avg_dict.get((-1, c))
        arr_R = avg_dict.get((+1, c))
        data_dist = avg_data.get(c)
        if arr_L is None or arr_R is None or data_dist is None:
            per_sse[c] = np.nan
            continue

        T_eff = min(arr_L.shape[-1], arr_R.shape[-1], data_dist.shape[-1])
        if T_eff <= 0:
            per_sse[c] = np.nan
            continue

        model = np.linalg.norm(arr_R[:, :T_eff] - arr_L[:, :T_eff], axis=0)  # no scaling/shifts
        data  = data_dist[:T_eff] - np.nanmin(data_dist[:T_eff])              # baseline shift data to its min

        if np.any(np.isnan(model)) or np.any(np.isnan(data)):
            per_sse[c] = np.nan
            continue

        per_sse[c] = float(np.sum((model - data)**2))
        model_list.append(model)
        data_list.append(data)

    # Average-curve SSE (truncate to shortest)
    if model_list and data_list:
        L = min(min(len(m) for m in model_list), min(len(d) for d in data_list))
        if L > 0:
            M = np.vstack([m[:L] for m in model_list])
            D = np.vstack([d[:L] for d in data_list])
            avg_model = np.mean(M, axis=0)
            avg_data  = np.mean(D, axis=0)
            avg_sse = float(np.sum((avg_model - avg_data)**2))
        else:
            avg_sse = np.nan
    else:
        avg_sse = np.nan

    valid = [v for v in per_sse.values() if not np.isnan(v)]
    total_sse = float(np.sum(valid)) if valid else np.nan

    return {'per_contrast': per_sse, 'avg_sse': avg_sse, 'total_loss': total_sse}


def plot_data_region_group_distance(plot_regs, timewindow, variable, reg_type):

    if 'intertrial' in timewindow:
        timeframe = timewindow
        split = run_align[timeframe][0]
        dist = []
        for reg in plot_regs:
            d_split = np.load(Path(pth_res, f'{split}.npy'), allow_pickle=True).flatten()[0][reg]
            dist.append(d_split['d_euc'])
        mean_d = {'mean': np.mean(dist, axis=0)}
        plt.plot(np.mean(dist, axis=0), label='mean')
    else:
        timeframe = variable + '_' + timewindow + '_act'
        print(timeframe)
        splits = run_align[timeframe]
        combined_name = 'combined_'+"_".join(splits)
        d = np.load(Path(pth_res, f'{combined_name}.npy'), allow_pickle=True).flatten()[0]
        
        dist = {}
        dist['mean'] = []
        for split in splits:
            dist[split] = []

        for reg in plot_regs:
            dist['mean'].append(d[reg]['d_euc']/len(splits))
            for split in splits:
                try:
                    d_split = np.load(Path(pth_res, f'{split}.npy'), allow_pickle=True).flatten()[0][reg]
                    dist[split].append(d_split['d_euc'])
                except:
                    pass

        mean_d = {}
        for k, v in dist.items():
            mean_d[k] = np.mean(v, axis=0)
            if k == 'mean':
                plt.plot(mean_d[k], label=k, color='black')
            else:
                plt.plot(mean_d[k], label=k, alpha=0.3)
    plt.legend()
    plt.title(f'{reg_type}')
    plt.ylabel(f'distance_{variable}')
    plt.savefig(f'{save_dir}/{reg_type}_{variable}_dist_curve.pdf', transparent=True)
    plt.show()
    return mean_d


# Copied from analysis_functions.py to avoid import and pickling issues with joblib
# These functions are needed for loss_prior_effect when reload=True
run_align = {
    'intertrial': ['block_stim_r_choice_r_f1', 'block_stim_l_choice_l_f1', 
                   'block_stim_l_choice_r_f2', 'block_stim_r_choice_l_f2'
                   ],
    'intertrial0': ['block_only'],
    'block_duringstim': ['block_duringstim_r_choice_r_f1', 'block_duringstim_l_choice_l_f1', 
                     'block_duringstim_l_choice_r_f2', 'block_duringstim_r_choice_l_f2'
                     ],
    'block_duringchoice': ['block_stim_r_duringchoice_r_f1', 'block_stim_l_duringchoice_l_f1', 
                            'block_stim_l_duringchoice_r_f2', 'block_stim_r_duringchoice_l_f2'
                            ],
    'intertrial1': ['block_stim_r_choice_r_f1', 'block_stim_l_choice_l_f1', 
                   ],
    'block_duringstim1': ['block_duringstim_r_choice_r_f1', 'block_duringstim_l_choice_l_f1', 
                     ],
    'block_duringchoice1': ['block_stim_r_duringchoice_r_f1', 'block_stim_l_duringchoice_l_f1', 
                            ],
    'act_intertrial': ['act_block_stim_r_choice_r_f1', 'act_block_stim_l_choice_l_f1', 
                   'act_block_stim_l_choice_r_f2', 'act_block_stim_r_choice_l_f2'
                   ],
    'act_intertrial0': ['act_block_only'],
    'act_intertrial0_old': ['act_block_only_old'],
    'act_block_duringstim': ['act_block_duringstim_r_choice_r_f1', 'act_block_duringstim_l_choice_l_f1', 
                     'act_block_duringstim_l_choice_r_f2', 'act_block_duringstim_r_choice_l_f2'
                     ],
    'act_block_duringchoice': ['act_block_stim_r_duringchoice_r_f1', 'act_block_stim_l_duringchoice_l_f1', 
                            'act_block_stim_l_duringchoice_r_f2', 'act_block_stim_r_duringchoice_l_f2'
                            ],
    'stim_duringstim': ['stim_choice_r_block_r', 'stim_choice_l_block_l', 
             'stim_choice_r_block_l', 'stim_choice_l_block_r'],
    'choice_duringchoice': ['choice_stim_r_block_r', 'choice_stim_l_block_l', 
               'choice_stim_r_block_l', 'choice_stim_l_block_r'],
    'stim_duringchoice': ['stim_duringchoice_r_block_r', 
                          'stim_duringchoice_l_block_l', 
                          'stim_duringchoice_r_block_l', 
                          'stim_duringchoice_l_block_r'],
    'choice_duringstim': ['choice_duringstim_r_block_r', 
                          'choice_duringstim_l_block_l', 
                          'choice_duringstim_r_block_l', 
                          'choice_duringstim_l_block_r'],
    'stim_duringstim_act': ['stim_choice_r_block_r_act', 'stim_choice_l_block_l_act', 
             'stim_choice_r_block_l_act', 'stim_choice_l_block_r_act'],
    'choice_duringchoice_act': ['choice_stim_r_block_r_act', 'choice_stim_l_block_l_act', 
               'choice_stim_r_block_l_act', 'choice_stim_l_block_r_act'],
    'stim_duringchoice_act': ['stim_duringchoice_r_block_r_act', 
                          'stim_duringchoice_l_block_l_act', 
                          'stim_duringchoice_r_block_l_act', 
                          'stim_duringchoice_l_block_r_act'],
    'choice_duringstim_act': ['choice_duringstim_r_block_r_act', 
                          'choice_duringstim_l_block_l_act', 
                          'choice_duringstim_r_block_l_act', 
                          'choice_duringstim_l_block_r_act'],
    'stim_duringstim1': ['stim_block_l', 'stim_block_r'],
    'stim_duringstim1_act': ['stim_block_l_act', 'stim_block_r_act'],
    'stim_duringstim_short': ['stim_choice_r_block_r_short', 'stim_choice_l_block_l_short', 
                         'stim_choice_r_block_l_short', 'stim_choice_l_block_r_short'], 
    'stim_duringstim_short_act': ['stim_choice_r_block_r_short_act', 'stim_choice_l_block_l_short_act', 
                         'stim_choice_r_block_l_short_act', 'stim_choice_l_block_r_short_act'], 
}


def _debias_selected_vector(
    d_vec, controls_mat, alpha=0.05, max_iter=50, tol=1e-8,
    apply_only_when_selected=True, selection_mode="mean_over_time"
):
    """
    Debias a selected/thresholded statistic vector using empirical or parametric
    truncated-normal correction.
    Copied from analysis_functions.py to avoid import issues.
    """
    from scipy import stats
    
    d = np.asarray(d_vec, dtype=float)
    C = np.asarray(controls_mat, dtype=float)
    if C.ndim != 2 or d.ndim != 1 or C.shape[1] != d.shape[0]:
        raise ValueError("shapes must be (T,) and (K,T)")

    mu0 = C.mean(axis=0)
    sigma0 = C.std(axis=0, ddof=1)
    small = sigma0 <= 0
    denom = np.where(sigma0 > 0, sigma0, 1.0)
    Tlen = d.shape[0]

    q = max(min(1.0 - alpha, 1.0), 0.0)

    if selection_mode == "max_over_time":
        max_controls = C.max(axis=1)
        t_global = float(np.quantile(max_controls, q))
        t_thresh = np.full_like(d, t_global, dtype=float)
        empirical_bias_vec = None

    elif selection_mode == "mean_over_time":
        mean_controls = C.mean(axis=1)
        u = float(np.quantile(mean_controls, q))
        sel_rows = mean_controls > u
        if sel_rows.sum() >= 5:
            empirical_bias_vec = C[sel_rows].mean(axis=0)
        else:
            empirical_bias_vec = None
        t_thresh = np.full_like(d, u, dtype=float)
        mu_m = float(mu0.mean())
        sigma_m2 = float((sigma0**2).sum() / (Tlen**2))
        sigma_m = np.sqrt(max(sigma_m2, 1e-32))

    else:  # per_time
        t_thresh = np.quantile(C, q, axis=0)
        empirical_bias_vec = None

    theta = d - mu0

    if apply_only_when_selected and alpha < 1.0 - 1e-12:
        if selection_mode == "mean_over_time":
            if d.mean() <= t_thresh[0]:
                return theta
        else:
            if not np.any(d > t_thresh):
                return theta

    if selection_mode == "mean_over_time" and empirical_bias_vec is not None:
        return d - empirical_bias_vec

    if selection_mode != "mean_over_time":
        sel = np.ones_like(d, dtype=bool)
        theta_sel = theta[sel]
        for _ in range(max_iter):
            a = (t_thresh[sel] - theta_sel - mu0[sel]) / denom[sel]
            tail = 1.0 - stats.norm.cdf(a)
            tail = np.maximum(tail, 1e-16)
            bias = mu0[sel] + denom[sel] * (stats.norm.pdf(a) / tail)
            new_theta_sel = d[sel] - bias
            new_theta_sel = np.where(small[sel], d[sel] - mu0[sel], new_theta_sel)
            if np.max(np.abs(new_theta_sel - theta_sel)) < tol:
                theta_sel = new_theta_sel
                break
            theta_sel = new_theta_sel
        theta[sel] = theta_sel
        return theta

    sel = np.ones_like(d, dtype=bool)
    theta_sel = theta[sel]
    for _ in range(max_iter):
        thetabar = float(theta.mean())
        alpha_std = (t_thresh[0] - thetabar - mu_m) / sigma_m
        tail = 1.0 - stats.norm.cdf(alpha_std)
        tail = max(tail, 1e-16)
        lambda_star = stats.norm.pdf(alpha_std) / tail
        scale = (sigma0[sel]**2) / (Tlen * sigma_m)
        bias_sel = mu0[sel] + scale * lambda_star
        new_theta_sel = d[sel] - bias_sel
        new_theta_sel = np.where(small[sel], d[sel] - mu0[sel], new_theta_sel)
        if np.max(np.abs(new_theta_sel - theta_sel)) < tol:
            theta_sel = new_theta_sel
            break
        theta_sel = new_theta_sel

    theta[sel] = theta_sel
    return theta


def load_combined_data(timeframe, dist='de'):
    """
    Load combined data for a timeframe.
    Copied from analysis_functions.py to avoid import issues.
    """
    splits = run_align[timeframe]
    if len(splits) == 1:
        combined_name = splits[0]
        combined_regd_name = f'{combined_name}_reg{dist}'
    else:
        combined_name = 'combined_'+"_".join(splits)
        combined_regd_name = f'combined_reg{dist}_'+"_".join(splits)
    d = np.load(Path(pth_res, f'{combined_name}.npy'), 
                    allow_pickle=True).flat[0]
    r = np.load(Path(pth_res, f'{combined_regd_name}.npy'), allow_pickle=True).flatten()[0]
    return d, r, combined_name, combined_regd_name


def load_group(regs, timeframe, ptype='p_mean_c', alpha=0.05, is_stim=False, correction='simple', dist='de'):
    """
    Load group data for regions.
    Copied from analysis_functions.py to avoid import issues.
    """
    if is_stim:
        alpha = 1.0
    d_all, r_all, _, _ = load_combined_data(timeframe, dist=dist)
    if correction=='intertrial':
        _, r_all_intertrial, _, _ = load_combined_data('act_intertrial0', dist=dist)
    splits = run_align[timeframe]
    split = splits[0]
    d_split = np.load(Path(pth_res, f'{split}.npy'), allow_pickle=True).flatten()[0]

    all_regs_r = None
    all_regs_cell_num = 0
    for reg in regs:
        d = d_all[reg]
        if d[ptype] <= alpha:
            if correction=='intertrial':
                baseline = np.mean(r_all_intertrial[reg][1:])        
            else:        
                baseline = 0
            if len(splits) > 1:
                r = np.concatenate([r_all[reg][0].reshape(1, -1), r_all[reg][1]], axis=0) / len(splits)
            else:
                r = r_all[reg]
            if dist=='xn':
                term = r - baseline
            elif dist=='de':
                cell_num = d_split[reg]['nclus']
                all_regs_cell_num += cell_num
                term = (r - baseline) * cell_num
            all_regs_r = term if all_regs_r is None else (all_regs_r + term)

    if dist=='xn':
        avg_r = all_regs_r
    elif dist=='de':
        if all_regs_cell_num == 0 or all_regs_r is None:
            return None
        avg_r = all_regs_r / all_regs_cell_num

    if correction=="debias":
        corrected = _debias_selected_vector(avg_r[0], avg_r[1:], alpha=alpha)
    elif correction=="simple":
        corrected = avg_r[0] - np.mean(avg_r[1:], axis=0)
    else: # no correction or intertrial baseline
        corrected = avg_r[0]
    return corrected


def load_normalized_r(regs, timeframe):
    _, r_all, _, _ = load_combined_data(timeframe)
    splits = run_align[timeframe]
    split = splits[0]
    d_split = np.load(Path(pth_res, f'{split}.npy'), allow_pickle=True).flatten()[0]

    all_regs_r = None
    all_regs_cell_num = 0
    for reg in regs:
        # d = d_all[reg]
        r = np.concatenate([r_all[reg][0].reshape(1, -1), r_all[reg][1]], axis=0) / len(splits)
        cell_num = d_split[reg]['nclus']
        all_regs_cell_num += cell_num
        term = (r ** 2) * cell_num
        all_regs_r = term if all_regs_r is None else (all_regs_r + term)

    avg_r = (all_regs_r / all_regs_cell_num) ** 0.5
    if all_regs_cell_num == 0 or all_regs_r is None:
        return None
    return (avg_r[0] - np.mean(avg_r[1:], axis=0))


def plot_dist_I_M_P(out, real_data, dt_I=2, dt_M=2, dt_P=2, scale_factors=None, ylim_P=None, save_dir=save_dir):
    """
    out: result of mean_by_condition(...)
         out['I'][('ts', ts)], out['I'][('ch', ch)], out['M'][('ch', ch)], out['P'][sp]
         each array shaped (2, T_eff) or all-NaNs when unavailable.

    Creates four separate figures:
      1) I: distance between trial sides (+1 vs -1)       [uses ('ts', ts)]
      2) I: distance between choice sides (+1 vs -1)      [uses ('ch', ch)]
      3) M: distance between choice sides (+1 vs -1)      [uses ('ch', ch)]
      4) P: distance between prior sides (+1 vs -1)       [uses sp = ±1]

    real_data: dict with keys like real_data['int_stim']['mean'], etc.
    scale_factors: optional {'int_stim': float, 'int': float, 'move': float}
                   (applies to model AFTER baseline shift; NOT applied to 'prior')
    ylim_P: tuple (ymin, ymax) to override y-axis limits for the P plot
    """

    def _valid(arr):
        return (
            isinstance(arr, np.ndarray)
            and arr.ndim == 2
            and arr.shape[0] == 2
            and arr.size
            and not np.isnan(arr).all()
        )

    def _dist(a_pos, a_neg):
        T = min(a_pos.shape[1], a_neg.shape[1])
        return np.linalg.norm(a_pos[:, :T] - a_neg[:, :T], axis=0)

    def _prep_real(curve, shift_mode="min", ref_mean=None):
        """Convert real curve to distance and shift baseline."""
        if curve is None:
            return None
        curve = np.asarray(curve)
        if curve.ndim == 1:
            d = curve
        elif curve.ndim == 2 and curve.shape[0] == 2:
            d = np.linalg.norm(curve[1] - curve[0], axis=0)
        else:
            return None

        if shift_mode == "min":
            return d - np.nanmean(d[:5])
        elif shift_mode == "ref" and ref_mean is not None:
            return d - np.nanmean(d) + ref_mean
        return d

    def _prep_model(d, key):
        if d is None:
            return None
        d = d - np.nanmin(d)  # baseline shift first
        # Scale for everything EXCEPT 'prior'
        if key != "prior" and scale_factors and key in scale_factors:
            d = d * float(scale_factors[key])
        return d

    def _plot(model_curve, real_curve, dt, color, ylabel, *, ylim=None, xlim=None):
        if real_curve is not None:
            T = min(len(model_curve), len(real_curve))
            model_curve, real_curve = model_curve[:T], real_curve[:T]
        else:
            T = len(model_curve)
        t = np.arange(T) * dt
        fig, ax = plt.subplots(figsize=(187/72, 160/72))
        r2_text = None
        if real_curve is not None:
            ax.plot(t, real_curve, "-", lw=2.0, color=color, label="data")
        ax.plot(t, model_curve, "--", lw=2.0, color=color, label="model")
        ax.set_xlabel(f"time ({'ms' if dt != 1.0 else 'steps'})")
        # ax.set_ylabel(ylabel)
        # ax.set_title(title)
        # ax.legend(frameon=False)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ylim is not None:
            ax.set_ylim(ylim)
        if xlim is not None:
            ax.set_xlim(xlim)

        # --- R^2 (GoF) printed on the plot ---
        if real_curve is not None and len(real_curve) == len(model_curve):
            rc = np.asarray(real_curve, float)
            mc = np.asarray(model_curve, float)
            if np.all(np.isfinite(rc)) and np.all(np.isfinite(mc)):
                eps = 1e-12
                sse_raw = float(np.sum((mc - rc) ** 2))
                sst = float(np.sum((rc - float(np.nanmean(rc))) ** 2) + eps)
                r2 = 1.0 - (sse_raw / sst) if sst > 0 else np.nan
                r2_text = f"R² = {r2:.3f}"
                ax.text(0.1, 0.62, r2_text, transform=ax.transAxes, fontsize=12, color=color)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/{ylabel}.svg")

    # --- 1) I: trial sides ---
    I_ts_pos = out.get("I", {}).get(("post", ("ts", +1)))
    I_ts_neg = out.get("I", {}).get(("post", ("ts", -1)))
    real_I_ts = _prep_real(real_data.get("int_stim", {}), shift_mode="min")
    if _valid(I_ts_pos) and _valid(I_ts_neg):
        d_model = _prep_model(_dist(I_ts_pos, I_ts_neg), "int_stim")
        _plot(
            d_model,
            real_I_ts,
            dt_I,
            "gold",
            "ds_I",
            ylim=None,
            xlim=(-5, 120),
        )

    # --- 2) I: choice sides ---
    I_ch_pos = out.get("I", {}).get(("post", ("ch", +1)))
    I_ch_neg = out.get("I", {}).get(("post", ("ch", -1)))
    real_I_ch = _prep_real(real_data.get("int", {}), shift_mode="min")
    if _valid(I_ch_pos) and _valid(I_ch_neg):
        d_model = _prep_model(_dist(I_ch_pos, I_ch_neg), "int")
        _plot(
            d_model,
            real_I_ch,
            dt_I,
            "gold",
            "dc_I",
            ylim=None,
            xlim=(-5, 120),
        )

    # --- 3) M: choice sides ---
    M_ch_pos = out.get("M", {}).get(("pre", ("ch", +1)))
    M_ch_neg = out.get("M", {}).get(("pre", ("ch", -1)))
    real_M_ch = _prep_real(real_data.get("move", {}), shift_mode="min")
    if _valid(M_ch_pos) and _valid(M_ch_neg):
        d_model = _prep_model(_dist(M_ch_pos, M_ch_neg), "move")
        _plot(
            d_model,
            real_M_ch,
            dt_M,
            "tomato",
            "dc_M",
        )

    # --- 4) P: prior sides ---
    P_pos, P_neg = out.get("P", {}).get(+1), out.get("P", {}).get(-1)
    if _valid(P_pos) and _valid(P_neg):
        d_model = _prep_model(_dist(P_pos, P_neg), "prior")
        real_P = _prep_real(
            real_data.get("prior", {}),
            shift_mode="ref",
            ref_mean=np.nanmean(d_model),
        )
        _plot(
            d_model,
            real_P,
            dt_P,
            "purple",
            "dp_P",
            ylim=ylim_P,
        )


def plot_choice_effect(
    regions, results, model_params,
    steps_before_obs, T=72, model_metric="l2",
    timeframes=('choice_duringstim_act','choice_duringchoice_act'),
    alpha=1.0, ptype='p_mean_c', plot_window=80,
    label_A='integrator', label_B='move',
    do_plot=True, save_dir=save_dir, shift_baseline=True, plot_shifted=True
):
    """
    Plot real-data region-group curves against model *choice-distance* curves (I, M, and S for duringstim).
    Identical visuals to loss_prior_effect but for choice difference and without SSE calculations.
    Additionally prints amplitude ratios (M/I) for model and data per timeframe.
    """

    if isinstance(timeframes, str):
        timeframes = (timeframes,)

    def time_axis_for(timeframe, time_window, duration_ms=None):
        """
        Create time axis for plotting.
        
        Args:
            timeframe: 'duringchoice' or 'duringstim'
            time_window: Number of bins
            duration_ms: Optional duration in milliseconds. If provided, the axis spans from 
                        -duration_ms to 0 (choice) or 0 to duration_ms (stim)
        """
        if duration_ms is not None:
            # time_window is number of bins, duration_ms is the duration in ms
            if 'duringchoice' in timeframe:
                return np.linspace(-duration_ms, 0, time_window)
            elif 'duringstim' in timeframe:
                return np.linspace(0, duration_ms, time_window)
        else:
            # Legacy: time_window is number of bins, assume 2ms per bin
            if 'duringchoice' in timeframe:
                return np.linspace(-2*time_window, 0, time_window)
            elif 'duringstim' in timeframe:
                return np.linspace(0, 2*time_window, time_window)

    def _align_baseline(y_model, y_data, baseline_mode="min"):
        if not shift_baseline:
            return y_data
        if y_model is None or y_data is None:
            return y_data
        if len(y_model) == 0 or len(y_data) == 0:
            return y_data
        if np.all(np.isnan(y_model)) or np.all(np.isnan(y_data)):
            return y_data
        m_min = np.nanmin(y_model)
        if baseline_mode == "mean":
            d_base = np.nanmean(y_data)
        else:
            d_base = np.nanmean(y_data[:5])
        return y_data - d_base + m_min

    def _amp(arr):
        if arr is None or len(arr) == 0:
            return np.nan
        amax = np.nanmax(arr)
        amin = np.nanmin(arr)
        return float(amax - amin) if np.isfinite(amax) and np.isfinite(amin) else np.nan

    def _safe_ratio(num, den):
        return float(num/den) if (den is not None and np.isfinite(den) and den != 0) else np.nan

    # model choice distances
    model_dists = choice_distance_I_M_both_alignments(results, steps_before_obs, T=T, metric=model_metric)

    # plotting setup
    if do_plot:
        fig, axs = plt.subplots(1, len(timeframes), sharey=True, figsize=(5, 2), dpi=150)
        if len(timeframes) == 1:
            axs = [axs]
    else:
        axs = [None] * len(timeframes)

    for ax, timeframe in zip(axs, timeframes):
        is_choice = ('duringchoice' in timeframe)
        # regs_move = regions['move_regs_choice'] if is_choice else regions['move_regs_stim']
        regs_move = list(set(move_regs_stim) | set(move_regs_choice))
        # regs_int  = regions['int_regs_choice']  if is_choice else regions['int_regs_stim']
        regs_int = list(set(int_regs_stim) | set(int_regs_choice))
        if not is_choice:
            regs_stim = regions['stim_regs']
        
        r_int  = load_group(regs_int, timeframe, ptype=ptype, alpha=alpha)
        r_move = load_group(regs_move, timeframe, ptype=ptype, alpha=alpha)
        
        # plot_window is now in milliseconds. Calculate number of bins from data length
        # Data bin size is ~2.08ms
        data_length = len(r_int) if r_int is not None and len(r_int) > 0 else int(round(plot_window / 2.08))
        times = time_axis_for(timeframe, time_window=data_length, duration_ms=plot_window)
        # r_stim = load_group(regs_stim, timeframe) if (not is_choice) else None

        align_key = 'action' if is_choice else 'start'
        m_I = model_dists['I'][align_key]
        m_M = model_dists['M'][align_key]
        # m_S = model_dists['S'][align_key] if (not is_choice and 'S' in model_dists) else None

        # Prepare plotted series (so amplitudes match visuals)
        plot_I_data = plot_M_data = None
        m_I_plot = m_M_plot = None

        if do_plot:
            # I
            if (r_int is not None) and m_I.size:
                L_I_plot = min(len(m_I), len(r_int), len(times))
                if L_I_plot > 0:
                    if is_choice:
                        times_plot = times[-L_I_plot:]
                        m_I_plot   = m_I[-L_I_plot:]
                        plot_I_data = _align_baseline(m_I_plot, r_int[-L_I_plot:], baseline_mode="min") if plot_shifted else r_int[-L_I_plot:]
                    else:
                        times_plot = times[:L_I_plot]
                        m_I_plot   = m_I[:L_I_plot]
                        plot_I_data = _align_baseline(m_I_plot, r_int[:L_I_plot], baseline_mode="min") if plot_shifted else r_int[:L_I_plot]
                    ax.plot(times_plot, plot_I_data, color='gold', linewidth=1, label=label_A)
                    ax.plot(times_plot, m_I_plot, '--', color='gold', alpha=0.9, linewidth=1)

            # M
            if (r_move is not None) and m_M.size:
                L_M_plot = min(len(m_M), len(r_move), len(times))
                if L_M_plot > 0:
                    if is_choice:
                        times_plot = times[-L_M_plot:]
                        m_M_plot   = m_M[-L_M_plot:]
                        plot_M_data = _align_baseline(m_M_plot, r_move[-L_M_plot:], baseline_mode="min") if plot_shifted else r_move[-L_M_plot:]
                    else:
                        times_plot = times[:L_M_plot]
                        m_M_plot   = m_M[:L_M_plot]
                        plot_M_data = _align_baseline(m_M_plot, r_move[:L_M_plot], baseline_mode="min") if plot_shifted else r_move[:L_M_plot]
                    ax.plot(times_plot, plot_M_data, color='tomato', linewidth=1, label=label_B)
                    ax.plot(times_plot, m_M_plot, '--', color='tomato', alpha=0.9, linewidth=1)

            # ax.set_xlim(times[0], times[-1])
            # ax.set_ylim(-0.05, ylim)
            ax.set_xticks(np.linspace(times[0], times[-1], 4))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_facecolor('none')
            ax.tick_params(labelsize=12)

        # --- Amplitude ratios (use plotted series; fall back if not plotting) ---
        if plot_I_data is None and (r_int is not None) and m_I.size:
            # prepare minimal series for amplitude even if do_plot=False
            L_I_plot = min(len(m_I), len(r_int), len(times))
            if L_I_plot > 0:
                if is_choice:
                    m_I_plot   = m_I[-L_I_plot:]
                    plot_I_data = _align_baseline(m_I_plot, r_int[-L_I_plot:], baseline_mode="min") if plot_shifted else r_int[-L_I_plot:]
                else:
                    m_I_plot   = m_I[:L_I_plot]
                    plot_I_data = _align_baseline(m_I_plot, r_int[:L_I_plot], baseline_mode="min") if plot_shifted else r_int[:L_I_plot]

        if plot_M_data is None and (r_move is not None) and m_M.size:
            L_M_plot = min(len(m_M), len(r_move), len(times))
            if L_M_plot > 0:
                if is_choice:
                    m_M_plot   = m_M[-L_M_plot:]
                    plot_M_data = _align_baseline(m_M_plot, r_move[-L_M_plot:], baseline_mode="min") if plot_shifted else r_move[-L_M_plot:]
                else:
                    m_M_plot   = m_M[:L_M_plot]
                    plot_M_data = _align_baseline(m_M_plot, r_move[:L_M_plot], baseline_mode="min") if plot_shifted else r_move[:L_M_plot]

        amp_I_data = _amp(plot_I_data)
        amp_M_data = _amp(plot_M_data)
        amp_I_model = _amp(m_I_plot)
        amp_M_model = _amp(m_M_plot)

        ratio_data  = _safe_ratio(amp_M_data, amp_I_data)
        ratio_model = _safe_ratio(amp_M_model, amp_I_model)

        print(f"[{timeframe}] amplitude data: I={amp_I_data:.4g}, M={amp_M_data:.4g}, M/I={ratio_data:.4g}; "
              f"model: I={amp_I_model:.4g}, M={amp_M_model:.4g}, M/I={ratio_model:.4g}")

    # finalize plot
    if do_plot and (axs is not None) and (axs[0] is not None):
        ylabel_left = 'd_choice(t)'
        axs[0].set_ylabel(ylabel_left)
        axs[0].figure.tight_layout()
        if save_dir:
            param_name = f"gi{model_params['g_i']}_gm{model_params['g_m']}_gs{model_params['g_s']}_di{model_params['d_i']}_dm{model_params['d_m']}_ds{model_params['d_s']}"
            fname = f"{save_dir}/choice_effect_{param_name}.svg"
            axs[0].figure.savefig(fname, transparent=True)

    return None


def choice_distance_I_M_both_alignments(results, steps_before_obs, T=75, metric="l2"):
    """
    Choice-distance (ch=+1 vs ch=-1) with BOTH alignments.
    IMPORTANT: balance across (ts, sp) buckets by averaging bucket MEANS first
    (equal weight across available buckets), then compute distance between the two
    balanced means. This matches the magnitude in plot_dist_I_M_P.
    """
    choices       = results['choices']
    trial_sides   = results['trial_sides']
    reaction_time = results.get('reaction_time', None)
    n             = len(choices)

    lens    = [len(trial_sides[i]) for i in range(n)]
    offsets = np.cumsum([0] + lens[:-1])
    hard_need = steps_before_obs + _min_trial_steps()
    fail_cnt = sum(1 for m in lens if m < hard_need)

    if n > 0 and fail_cnt > n/2:
        nanT = np.full(T, np.nan)
        return {
            'I': {'start': nanT.copy(), 'action': nanT.copy()},
            'M': {'start': nanT.copy(), 'action': nanT.copy()},
            'S': {'start': nanT.copy(), 'action': np.array([])}
        }

    def _segments_by_bucket(var_name, mode):
        """
        Return dict[(ts, sp, ch)] -> list of (T,2) segments for var_name.
        """
        var_array = np.asarray(results[var_name], dtype=float)
        if var_array.ndim != 2 or var_array.shape[1] != 2:
            raise ValueError(f"{var_name}: expected (TotalSteps,2), got {var_array.shape}")

        buckets = {}
        for i in range(n):
            m_i = lens[i]
            if m_i < hard_need:
                continue

            if mode == "pre_action":
                if reaction_time is None:
                    raise ValueError("reaction_time required for pre_action alignment")
                act_start = steps_before_obs + int(reaction_time[i])
                if act_start < T or act_start > m_i:
                    continue
                start = offsets[i] + act_start - T
                seg = var_array[start:start+T, :]
            else:  # post_start with fill
                post_avail = max(0, m_i - steps_before_obs)
                take_post  = min(T, post_avail)
                parts = []
                if take_post > 0:
                    start = offsets[i] + steps_before_obs
                    parts.append(var_array[start:start+take_post, :])
                if take_post < T:
                    if i + 1 >= n:
                        continue
                    m_next = lens[i+1]
                    if m_next < hard_need:
                        continue
                    pre_avail_next = min(steps_before_obs, m_next)
                    need = T - take_post
                    if pre_avail_next < need:
                        continue
                    start_next = offsets[i+1]
                    parts.append(var_array[start_next:start_next+need, :])
                if not parts:
                    continue
                seg = np.vstack(parts)
                if seg.shape[0] != T:
                    continue

            ts = int(np.sign(trial_sides[i][0]))
            ch = int(choices[i])
            sp = 1 if results['sub_prior'][i][0] < 0 else -1

            key = (ts, sp, ch)
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(seg)  # (T,2)

        return buckets

    def _balanced_mean(buckets, ch_sign):
        """
        Build balanced mean (T,2) for a given ch_sign by:
          1) For each available (ts,sp), average trials in that bucket -> (T,2)
          2) Average these bucket-means equally across buckets present
        Returns None if no buckets for this ch_sign.
        """
        bucket_means = []
        for ts in (+1, -1):
            for sp in (+1, -1):
                key = (ts, sp, ch_sign)
                if key in buckets and len(buckets[key]) > 0:
                    B = np.mean(np.stack(buckets[key], axis=0), axis=0)  # (T,2)
                    bucket_means.append(B)
        if not bucket_means:
            return None
        return np.mean(np.stack(bucket_means, axis=0), axis=0)  # (T,2)

    def _balanced_choice_distance(var_name, mode):
        buckets = _segments_by_bucket(var_name, mode)
        A_pos = _balanced_mean(buckets, +1)
        A_neg = _balanced_mean(buckets, -1)
        if A_pos is None or A_neg is None:
            return np.array([])
        if metric == "l2":
            return np.linalg.norm(A_pos - A_neg, axis=1)  # (T,)
        elif metric == "side":
            return np.abs((A_pos[:,1]-A_pos[:,0]) - (A_neg[:,1]-A_neg[:,0]))
        else:
            raise ValueError("metric must be 'l2' or 'side'")

    out = {'I': {}, 'M': {}, 'S': {}}
    for vn in ('I', 'M'):
        out[vn]['start']  = _balanced_choice_distance(vn, "post_start")
        out[vn]['action'] = _balanced_choice_distance(vn, "pre_action")

    # S: start-only, same balancing
    out['S']['start']  = _balanced_choice_distance('S', "post_start")
    out['S']['action'] = np.array([])

    return out