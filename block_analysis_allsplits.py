'''
This script is used to analyze prior sensitivity of all splits (e.g. different alignment times, choice/stim conditions).
'''

from one.api import ONE
from brainbox.singlecell import bin_spikes2D
from brainwidemap import (bwm_query, load_good_units, 
                          load_trials_and_mask, bwm_units,
                          download_aggregate_tables)
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions
import iblatlas
from iblatlas.plots import plot_swanson_vector 
from brainbox.io.one import SessionLoader

# from ibllib.atlas import FlatMap
# from ibllib.atlas.flatmaps import plot_swanson

from scipy import optimize, signal, stats
import pandas as pd
import numpy as np
from collections import Counter, ChainMap
from sklearn.decomposition import PCA
import gc
import os
from pathlib import Path
import glob
from dateutil import parser

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from adjustText import adjust_text
# import matplotlib.image as mpimg
# from matplotlib.gridspec import GridSpec
# from matplotlib import colors
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d import proj3d
# from PIL import Image
# import io
# import matplotlib.patches as mpatches
# import matplotlib.ticker as ticker

import random
from random import shuffle
from copy import deepcopy
import time
import sys
import re

import math
import string

import cProfile
import pstats

import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=sys.maxsize)

f_size = 15
def set_sizes(f_s = f_size):
    plt.rc('font', size=f_s)
    plt.rc('axes', titlesize=f_s)
    plt.rc('axes', labelsize=f_s)
    plt.rc('xtick', labelsize=f_s)
    plt.rc('ytick', labelsize=f_s)
    plt.rc('legend', fontsize=f_s)
    plt.rc('figure', titlesize=f_s)

blue_left = [0.13850039, 0.41331206, 0.74052025]
red_right = [0.66080672, 0.21526712, 0.23069468]

b_size = 0.0125  # 0.005 sec for a static bin size, or None for single bin
sts = 0.002  # stride size in s for overlapping bins 
ntravis = 30  # number of trajectories for visualisation, first 2 real
nrand = 2000  # number of random trial splits for null_d
min_reg = 20  # 100, minimum number of neurons in pooled region
min_trials_per_side = 5  # skip insertion if either split side has fewer trials
alpha = 0.2 # inverse of time constant for action kernel calculation
# IBL Bayes-optimal prior (Findling et al. Nature 2025 SI §1.1.1)
BAYES_TAU = 60.0       # truncated-exp block-length scale
BAYES_GAMMA = 0.8      # P(stim matches biased-block side)
BAYES_MIN_LEN = 20
BAYES_MAX_LEN = 100


class InsufficientTrials(ValueError):
    '''Raised when a split side has fewer than ``min_trials_per_side`` trials.'''

# trial split types, see get_d_vars for details
align_old = {
         'block_duringstim_r_choice_r_f1':'stimOn_times', #'srcrbl_srcrbr'
         'block_duringstim_l_choice_l_f1':'stimOn_times', #'slclbl_slclbr'
         'block_duringstim_l_choice_r_f2':'stimOn_times', #'slcrbl_slcrbr'
         'block_duringstim_r_choice_l_f2':'stimOn_times', #'srclbl_srclbr'
         'block_stim_r_duringchoice_r_f1':'firstMovement_times',
         'block_stim_l_duringchoice_l_f1':'firstMovement_times',    
         'block_stim_l_duringchoice_r_f2':'firstMovement_times',
         'block_stim_r_duringchoice_l_f2':'firstMovement_times',
         'block_stim_r_choice_r_f1':'stimOn_times',
         'block_stim_l_choice_l_f1':'stimOn_times',
         'block_stim_l_choice_r_f2':'stimOn_times',
         'block_stim_r_choice_l_f2':'stimOn_times',
         'block_only':'stimOn_times',
         'act_block_stim_r_choice_r_f1':'stimOn_times',
         'act_block_stim_l_choice_l_f1':'stimOn_times',
         'act_block_stim_l_choice_r_f2':'stimOn_times',
         'act_block_stim_r_choice_l_f2':'stimOn_times',
         'act_block_duringstim_r_choice_r_f1':'stimOn_times',
         'act_block_duringstim_l_choice_l_f1':'stimOn_times',
         'act_block_duringstim_l_choice_r_f2':'stimOn_times',
         'act_block_duringstim_r_choice_l_f2':'stimOn_times',
         'act_block_stim_r_duringchoice_r_f1':'firstMovement_times',
         'act_block_stim_l_duringchoice_l_f1':'firstMovement_times',
         'act_block_stim_l_duringchoice_r_f2':'firstMovement_times',
         'act_block_stim_r_duringchoice_l_f2':'firstMovement_times',
         'act_block_only':'stimOn_times',
         # Stim-side prior (no choice×feedback): post-stim window via 'durings' in name
         'block_duringstim_l':'stimOn_times',
         'block_duringstim_r':'stimOn_times',
         'act_block_duringstim_l':'stimOn_times',
         'act_block_duringstim_r':'stimOn_times',
         # Bayes-optimal prior (stimulus-history inference); mirrors act_block_*
         'bayes_block_stim_r_choice_r_f1':'stimOn_times',
         'bayes_block_stim_l_choice_l_f1':'stimOn_times',
         'bayes_block_stim_l_choice_r_f2':'stimOn_times',
         'bayes_block_stim_r_choice_l_f2':'stimOn_times',
         'bayes_block_duringstim_r_choice_r_f1':'stimOn_times',
         'bayes_block_duringstim_l_choice_l_f1':'stimOn_times',
         'bayes_block_duringstim_l_choice_r_f2':'stimOn_times',
         'bayes_block_duringstim_r_choice_l_f2':'stimOn_times',
         'bayes_block_stim_r_duringchoice_r_f1':'firstMovement_times',
         'bayes_block_stim_l_duringchoice_l_f1':'firstMovement_times',
         'bayes_block_stim_l_duringchoice_r_f2':'firstMovement_times',
         'bayes_block_stim_r_duringchoice_l_f2':'firstMovement_times',
         'bayes_block_only':'stimOn_times',
         'bayes_block_duringstim_l':'stimOn_times',
         'bayes_block_duringstim_r':'stimOn_times',
        }

# Choicestim family (stim L–R or choice L–R contrasts). Windows set below —
# do not rely on align_old's default ITI [0.4, -0.1].
CHOICESTIM_ALIGN = {
    # bare (no block)
    'stim_choice_r': 'stimOn_times',
    'stim_choice_l': 'stimOn_times',
    'stim_block_r': 'stimOn_times',
    'stim_block_l': 'stimOn_times',
    'choice_stim_l': 'firstMovement_times',
    'choice_stim_r': 'firstMovement_times',
    'stim_duringchoice_r': 'firstMovement_times',
    'stim_duringchoice_l': 'firstMovement_times',
    'choice_duringstim_r': 'stimOn_times',
    'choice_duringstim_l': 'stimOn_times',
    # stim_block ± act/bayes (short 80 ms)
    'stim_block_l_act': 'stimOn_times',
    'stim_block_r_act': 'stimOn_times',
    'stim_block_l_bayes': 'stimOn_times',
    'stim_block_r_bayes': 'stimOn_times',
    # stim_choice × block ± act/bayes/short
    'stim_choice_r_block_r': 'stimOn_times',
    'stim_choice_r_block_l': 'stimOn_times',
    'stim_choice_l_block_r': 'stimOn_times',
    'stim_choice_l_block_l': 'stimOn_times',
    'stim_choice_r_block_r_act': 'stimOn_times',
    'stim_choice_r_block_l_act': 'stimOn_times',
    'stim_choice_l_block_r_act': 'stimOn_times',
    'stim_choice_l_block_l_act': 'stimOn_times',
    'stim_choice_r_block_r_bayes': 'stimOn_times',
    'stim_choice_r_block_l_bayes': 'stimOn_times',
    'stim_choice_l_block_r_bayes': 'stimOn_times',
    'stim_choice_l_block_l_bayes': 'stimOn_times',
    'stim_choice_r_block_r_short': 'stimOn_times',
    'stim_choice_r_block_l_short': 'stimOn_times',
    'stim_choice_l_block_r_short': 'stimOn_times',
    'stim_choice_l_block_l_short': 'stimOn_times',
    'stim_choice_r_block_r_short_act': 'stimOn_times',
    'stim_choice_r_block_l_short_act': 'stimOn_times',
    'stim_choice_l_block_r_short_act': 'stimOn_times',
    'stim_choice_l_block_l_short_act': 'stimOn_times',
    'stim_choice_r_block_r_short_bayes': 'stimOn_times',
    'stim_choice_r_block_l_short_bayes': 'stimOn_times',
    'stim_choice_l_block_r_short_bayes': 'stimOn_times',
    'stim_choice_l_block_l_short_bayes': 'stimOn_times',
    # choice_stim × block ± act/bayes
    'choice_stim_l_block_r': 'firstMovement_times',
    'choice_stim_l_block_l': 'firstMovement_times',
    'choice_stim_r_block_r': 'firstMovement_times',
    'choice_stim_r_block_l': 'firstMovement_times',
    'choice_stim_l_block_r_act': 'firstMovement_times',
    'choice_stim_l_block_l_act': 'firstMovement_times',
    'choice_stim_r_block_r_act': 'firstMovement_times',
    'choice_stim_r_block_l_act': 'firstMovement_times',
    'choice_stim_l_block_r_bayes': 'firstMovement_times',
    'choice_stim_l_block_l_bayes': 'firstMovement_times',
    'choice_stim_r_block_r_bayes': 'firstMovement_times',
    'choice_stim_r_block_l_bayes': 'firstMovement_times',
    # stim_duringchoice × block ± act/bayes
    'stim_duringchoice_r_block_r': 'firstMovement_times',
    'stim_duringchoice_r_block_l': 'firstMovement_times',
    'stim_duringchoice_l_block_r': 'firstMovement_times',
    'stim_duringchoice_l_block_l': 'firstMovement_times',
    'stim_duringchoice_r_block_r_act': 'firstMovement_times',
    'stim_duringchoice_r_block_l_act': 'firstMovement_times',
    'stim_duringchoice_l_block_r_act': 'firstMovement_times',
    'stim_duringchoice_l_block_l_act': 'firstMovement_times',
    'stim_duringchoice_r_block_r_bayes': 'firstMovement_times',
    'stim_duringchoice_r_block_l_bayes': 'firstMovement_times',
    'stim_duringchoice_l_block_r_bayes': 'firstMovement_times',
    'stim_duringchoice_l_block_l_bayes': 'firstMovement_times',
    # choice_duringstim × block ± act/bayes
    'choice_duringstim_r_block_r': 'stimOn_times',
    'choice_duringstim_r_block_l': 'stimOn_times',
    'choice_duringstim_l_block_r': 'stimOn_times',
    'choice_duringstim_l_block_l': 'stimOn_times',
    'choice_duringstim_r_block_r_act': 'stimOn_times',
    'choice_duringstim_r_block_l_act': 'stimOn_times',
    'choice_duringstim_l_block_r_act': 'stimOn_times',
    'choice_duringstim_l_block_l_act': 'stimOn_times',
    'choice_duringstim_r_block_r_bayes': 'stimOn_times',
    'choice_duringstim_r_block_l_bayes': 'stimOn_times',
    'choice_duringstim_l_block_r_bayes': 'stimOn_times',
    'choice_duringstim_l_block_l_bayes': 'stimOn_times',
}


# align_act = {
#          'act_block_stim_r_choice_r_f1':'stimOn_times',
#          'act_block_stim_l_choice_l_f1':'stimOn_times',
#          'act_block_stim_l_choice_r_f2':'stimOn_times',
#          'act_block_stim_r_choice_l_f2':'stimOn_times',
#          'act_block_duringstim_r_choice_r_f1':'stimOn_times',
#          'act_block_duringstim_l_choice_l_f1':'stimOn_times',
#          'act_block_duringstim_l_choice_r_f2':'stimOn_times',
#          'act_block_duringstim_r_choice_l_f2':'stimOn_times',
#          'act_block_stim_r_duringchoice_r_f1':'firstMovement_times',
#          'act_block_stim_l_duringchoice_l_f1':'firstMovement_times',
#          'act_block_stim_l_duringchoice_r_f2':'firstMovement_times',
#          'act_block_stim_r_duringchoice_l_f2':'firstMovement_times',
#          'act_block_only':'stimOn_times',
#          'block_only':'stimOn_times',
#         }

splits= [
         'block_duringstim_r_choice_r_f1',
         'block_duringstim_l_choice_l_f1',
         'block_stim_r_duringchoice_r_f1',
         'block_stim_l_duringchoice_l_f1',
         'block_stim_r_choice_r_f1',
         'block_stim_l_choice_l_f1',
         'block_duringstim_l_choice_r_f2',
         'block_duringstim_r_choice_l_f2',
         'block_stim_l_duringchoice_r_f2',
         'block_stim_r_duringchoice_l_f2',
         'block_stim_l_choice_r_f2',
         'block_stim_r_choice_l_f2',
         'stim'
]
values = [1., 0.25, 0.125, 0.0625, 0.]
splits_new = [f'{split}_{value}' for split in splits for value in values]
align = {}
pre_post = {}
for split in align_old:
    if 'durings' in split:
        align[split] = 'stimOn_times'
        pre_post[split] = [0,0.15]        
    elif 'duringc' in split:
        align[split] = 'firstMovement_times'
        pre_post[split] = [0.15,0]
    else:
        align[split] = 'stimOn_times'
        pre_post[split] = [0.4,-0.1]

for split in splits_new:
    if 'durings' in split:
        align[split] = 'stimOn_times'
        pre_post[split] = [0,0.15]
    elif 'block' not in split:
        align[split] = 'stimOn_times'
        pre_post[split] = [0,0.15]        
    elif 'duringc' in split:
        align[split] = 'firstMovement_times'
        pre_post[split] = [0.15,0]
    else:
        align[split] = 'stimOn_times'
        pre_post[split] = [0.4,-0.1]

# Choicestim family: register align + windows (override any align_old ITI default).
# Rule matches choicestim_analysis: stim_block/short → 80 ms; else stimOn → 150 ms;
# firstMovement → [-0.15, 0] (pre/post).
SHORT_DURINGSTIM_WINDOW_S = 0.08


def is_choicestim_split(split):
    '''Stim/choice L–R contrasts from choicestim_analysis (not prior-distance).'''
    return split.startswith((
        'stim_choice', 'stim_block', 'choice_stim',
        'stim_duringchoice', 'choice_duringstim',
    ))


def is_choice_lr_split(split):
    '''Choice L vs R under fixed stim (±block): choice_stim* / choice_duringstim*.'''
    return split.startswith(('choice_stim', 'choice_duringstim'))


# Back-compat alias (eligibility only; session-shuffle is opt-in via flag).
def uses_session_shuffle_null(split):
    return is_choice_lr_split(split)


for _s, _ev in CHOICESTIM_ALIGN.items():
    align[_s] = _ev
    if 'stim_block' in _s or 'short' in _s:
        pre_post[_s] = [0, SHORT_DURINGSTIM_WINDOW_S]
    elif _ev == 'stimOn_times':
        pre_post[_s] = [0, 0.15]
    elif _ev == 'firstMovement_times':
        pre_post[_s] = [0.15, 0]
    else:
        raise ValueError(f'Unexpected choicestim align event for {_s}: {_ev}')

# Goal-3 contrast conditioning. CONTRASTS are the IBL stimulus contrasts; 0.0 is
# the fully prior-driven (0% contrast) stratum.
CONTRASTS = [1.0, 0.25, 0.125, 0.0625, 0.0]

# During-trial prior-modulation bases (stim window + choice window), act and
# non-act. Contrast-conditioned names are '{base}_{contrast}' (e.g. ..._f1_0.125).
GOAL3_DURINGSTIM_BASES = [
    'block_duringstim_r_choice_r_f1',
    'block_duringstim_l_choice_l_f1',
    'block_duringstim_l_choice_r_f2',
    'block_duringstim_r_choice_l_f2',
    'act_block_duringstim_r_choice_r_f1',
    'act_block_duringstim_l_choice_l_f1',
    'act_block_duringstim_l_choice_r_f2',
    'act_block_duringstim_r_choice_l_f2',
]
GOAL3_DURINGCHOICE_BASES = [
    'block_stim_r_duringchoice_r_f1',
    'block_stim_l_duringchoice_l_f1',
    'block_stim_l_duringchoice_r_f2',
    'block_stim_r_duringchoice_l_f2',
    'act_block_stim_r_duringchoice_r_f1',
    'act_block_stim_l_duringchoice_l_f1',
    'act_block_stim_l_duringchoice_r_f2',
    'act_block_stim_r_duringchoice_l_f2',
]
GOAL3_BASE_SPLITS = GOAL3_DURINGSTIM_BASES + GOAL3_DURINGCHOICE_BASES

# Bayes-optimal prior variants (same windows as act_block_*); kept separate so
# existing goal3_* presets (act vs true-block) are unchanged.
GOAL3_BAYES_DURINGSTIM_BASES = [
    'bayes_block_duringstim_r_choice_r_f1',
    'bayes_block_duringstim_l_choice_l_f1',
    'bayes_block_duringstim_l_choice_r_f2',
    'bayes_block_duringstim_r_choice_l_f2',
]
GOAL3_BAYES_DURINGCHOICE_BASES = [
    'bayes_block_stim_r_duringchoice_r_f1',
    'bayes_block_stim_l_duringchoice_l_f1',
    'bayes_block_stim_l_duringchoice_r_f2',
    'bayes_block_stim_r_duringchoice_l_f2',
]
GOAL3_BAYES_BASE_SPLITS = (
    GOAL3_BAYES_DURINGSTIM_BASES + GOAL3_BAYES_DURINGCHOICE_BASES
)

# Revised Goal 3 (2026-07-17): at 0% contrast, nominal stimulus side and
# correct/incorrect feedback subdivisions are not conditioning variables. Compare
# true block L vs R separately within each fixed choice side.
GOAL3_C0_CHOICE_BASES = [
    'block_duringstim_choice_l',
    'block_duringstim_choice_r',
]
GOAL3_C0_CHOICE_SPLITS = [
    f'{base}_0.0' for base in GOAL3_C0_CHOICE_BASES
]


def contrast_from_split(split):
    '''
    Parse optional trailing contrast from a split name.

    Supported suffixes: ``_{float}`` (e.g. ``..._f1_0.125``) or ``_c{float}``
    (e.g. ``..._c0.125``). Returns None if the split is not contrast-stratified.
    '''
    # Anchor at end-of-string so '_choice' etc. never match as '_c...'.
    m = re.search(r'_c([0-9]*\.?[0-9]+)$', split)
    if m:
        return float(m.group(1))
    m = re.search(r'_([0-9]*\.?[0-9]+)$', split)
    if m:
        return float(m.group(1))
    return None


def contrast_split_name(base, contrast):
    '''Canonical contrast-conditioned split name: ``{base}_{contrast}``.'''
    return f'{base}_{float(contrast)}'


def expand_contrast_splits(bases=None, contrasts=None):
    '''Cartesian product of base splits × contrasts (default: all Goal-3 bases).'''
    if bases is None:
        bases = GOAL3_BASE_SPLITS
    if contrasts is None:
        contrasts = CONTRASTS
    return [contrast_split_name(b, c) for b in bases for c in contrasts]


def _register_split_window(split, align_event, window):
    align[split] = align_event
    pre_post[split] = list(window)


def is_goal3_c0_choice_split(split):
    '''True-block L-vs-R split at 0% contrast within one fixed choice side.'''
    return split in GOAL3_C0_CHOICE_SPLITS


def goal3_c0_choice_mask(trials, split):
    '''Eligible trials for revised Goal 3, without stim-side/feedback filtering.'''
    if not is_goal3_c0_choice_split(split):
        raise ValueError(f'Not a revised Goal-3 split: {split}')
    choice = 1 if '_choice_l_' in split else -1
    zero_contrast = np.bitwise_or(
        np.isclose(trials['contrastLeft'].astype(float), 0.0),
        np.isclose(trials['contrastRight'].astype(float), 0.0),
    )
    return zero_contrast & (trials['choice'] == choice)


def register_contrast_splits(bases=None, contrasts=None):
    '''
    Register align/pre_post for contrast-conditioned during-trial splits,
    copying the base split's alignment event and analysis window.
    '''
    if bases is None:
        bases = GOAL3_BASE_SPLITS
    if contrasts is None:
        contrasts = CONTRASTS
    out = []
    for base in bases:
        if base not in align or base not in pre_post:
            raise KeyError(f'Base split not in align/pre_post: {base}')
        for c in contrasts:
            name = contrast_split_name(base, c)
            _register_split_window(name, align[base], pre_post[base])
            out.append(name)
    return out


# Register Goal-3 contrast splits (duringstim + duringchoice, act + non-act).
goal3_contrast_splits = register_contrast_splits()
# Bayes-optimal prior contrast splits (same windows as act_block_* bases).
goal3_bayes_contrast_splits = register_contrast_splits(
    bases=GOAL3_BAYES_BASE_SPLITS)
# Revised Goal-3 splits have no old base split to copy because their names
# intentionally omit stimulus side and f1/f2.
for _split in GOAL3_C0_CHOICE_SPLITS:
    _register_split_window(_split, 'stimOn_times', [0, 0.15])


# one = ONE(cache_dir='/om2/user/arily/int-brain-lab/ONE',
#           base_url='https://openalyx.internationalbrainlab.org',
#           password='international', silent=True)  # (mode='local')
# one = ONE(base_url='https://alyx.internationalbrainlab.org')
one = ONE()
ba = AllenAtlas()
br = BrainRegions()

# save results for plotting here
pth_res = Path(one.cache_dir, 'manifold', 'res')
pth_res.mkdir(parents=True, exist_ok=True)
pth_stream_acc = pth_res / '_stream_acc'
pth_stream_acc.mkdir(parents=True, exist_ok=True)

# Appended to on-disk basenames for structured choice nulls ('' = label shuffle).
# Logical split names used in analysis are unchanged.
RES_FILE_SUFFIX = ''


def configure_null_file_suffix(actkernel_choice_null=False, session_shuffle_null=False):
    '''Tag pooled / stream_acc filenames by null scheme; shuffle keeps plain names.

    - ``--actkernel-choice-null`` → ``{split}_actkernel*.npy``
    - ``--session-shuffle-null`` (Harris) → ``{split}_harris*.npy``
    - default label shuffle → ``{split}*.npy`` (unchanged)
    '''
    global RES_FILE_SUFFIX
    if actkernel_choice_null:
        RES_FILE_SUFFIX = '_actkernel'
    elif session_shuffle_null:
        RES_FILE_SUFFIX = '_harris'
    else:
        RES_FILE_SUFFIX = ''
    if RES_FILE_SUFFIX:
        print(f'null-tagged file suffix: *{{split}}{RES_FILE_SUFFIX}*.npy '
              f'(stream_acc + res pooled)')
    return RES_FILE_SUFFIX


def output_split_name(split):
    '''Logical split → on-disk basename (adds RES_FILE_SUFFIX if set).'''
    if not RES_FILE_SUFFIX:
        return split
    if str(split).endswith(RES_FILE_SUFFIX):
        return split
    return f'{split}{RES_FILE_SUFFIX}'

# null shuffles processed in batches inside get_d_vars (control=True) to cap RAM
NULL_BATCH_SIZE = 100
# Retries per null draw when indexing donor session choices (empty side / short donor).
HARRIS_MAX_TRIES = 50
SESSION_SHUFFLE_MAX_TRIES = HARRIS_MAX_TRIES  # back-compat alias
# Trial exclusion for drift×stickiness sensitivity (choice L–R).
STICKY_LATE_FRAC = 0.20
STICKY_MIN_RUN = 10
# Per-process cache: eid → ActionKernel fit (params + reusable sim model).
_actkernel_choice_fit_cache = {}

# scripts/simulate_synthetic_choices.py (user-added ActionKernel wrapper).
def _load_syn_choices():
    import importlib.util
    from pathlib import Path as _Path
    path = _Path(__file__).resolve().parent / 'scripts' / 'simulate_synthetic_choices.py'
    spec = importlib.util.spec_from_file_location(
        'simulate_synthetic_choices', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_syn_choices = None


def _syn():
    global _syn_choices
    if _syn_choices is None:
        _syn_choices = _load_syn_choices()
    return _syn_choices


def _actkernel_model_dir():
    '''Cache fitted ActionKernel pickles under the ONE manifold root.'''
    root = Path(getattr(one, 'cache_dir', Path.home() / 'Downloads' / 'ONE'))
    d = root / 'manifold' / 'actkernel_fits'
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_actkernel_choice_fit(eid, trials, nb_steps=None):
    '''Fit ActionKernel once per eid (MCMC); cache under manifold/actkernel_fits/.

    ``nb_steps`` overrides package default. If unset, ``ACTKERNEL_NB_STEPS`` env
    is honored (for short smoke fits). Short-step fits use a separate pickle
    directory so they do not overwrite paper-length MCMC caches.
    '''
    import os
    if nb_steps is None and os.environ.get('ACTKERNEL_NB_STEPS'):
        nb_steps = int(os.environ['ACTKERNEL_NB_STEPS'])
    key = str(eid)
    cache_key = f'{key}:nb={nb_steps}' if nb_steps is not None else key
    if cache_key in _actkernel_choice_fit_cache:
        return _actkernel_choice_fit_cache[cache_key]
    syn = _syn()
    tag = key.replace('-', '')[:8]
    if nb_steps is not None:
        tag = f'{tag}_nb{int(nb_steps)}'
    model, params = syn.fit_action_kernel(
        trials, eid=key, subject='bwm',
        model_dir=_actkernel_model_dir() / tag,
        nb_steps=nb_steps)
    fit = {
        'eid': key,
        'params': np.asarray(params, dtype=float),
        'mode': 'mcmc',
        'nb_steps': nb_steps,
        'model': model,
        'sim_model': syn._sim_model(),
    }
    _actkernel_choice_fit_cache[cache_key] = fit
    return fit


def _sample_actkernel_choice_ys(elig_idx, trials, fit, rng=None,
                                max_tries=HARRIS_MAX_TRIES):
    '''
    BWM-style ActionKernel null labels: draw a pseudo-session (new stim/blocks
    via ``generate_pseudo_session``), simulate choices under fitted θ, then
    read choices at the recipient's stratified trial indices ``elig_idx``.
    '''
    def _ok(ys):
        return (int(ys.sum()) >= min_trials_per_side
                and int((~ys).sum()) >= min_trials_per_side)

    syn = _syn()
    theta = fit['params'] if isinstance(fit, dict) else fit
    elig_idx = np.asarray(elig_idx, dtype=int)
    for _ in range(max_tries):
        seed = None if rng is None else int(rng.integers(0, 2**31 - 1))
        ps = syn.make_synthetic_session(trials, theta, seed=seed)
        ch = np.asarray(ps['choice'], dtype=float).reshape(-1)
        ys = ch[elig_idx] == 1
        if _ok(ys):
            return ys
    seed = None if rng is None else int(rng.integers(0, 2**31 - 1))
    ps = syn.make_synthetic_session(trials, theta, seed=seed)
    ch = np.asarray(ps['choice'], dtype=float).reshape(-1)
    return ch[elig_idx] == 1


def _stream_acc_path(split, shard=None):
    '''Checkpoint path. shard=None → {split}.npy; else {split}.shard{k}.npy.

    ``split`` may be logical; ``RES_FILE_SUFFIX`` is applied for on-disk names.
    '''
    name = output_split_name(split)
    if shard is None:
        return pth_stream_acc / f'{name}.npy'
    return pth_stream_acc / f'{name}.shard{int(shard)}.npy'


def _stream_acc_shard_paths(split):
    '''All existing shard checkpoint files for a split (sorted by shard index).'''
    name = output_split_name(split)
    paths = sorted(pth_stream_acc.glob(f'{name}.shard*.npy'))
    return paths


def perseveration_run_mask(choice, stim_is_left, contrast_mag, min_run=STICKY_MIN_RUN):
    '''
    Boolean mask: True on the **tail** of same-choice runs of length ≥ ``min_run``
    that are poorly explained by non-zero-contrast stimuli.

    For a poorly explained run of length L ≥ min_run, keep the first
    ``min_run - 1`` trials and drop only trials from position ``min_run``
    onward (1-based within the run). Short same-side streaks stay intact.

    A run is poorly explained if, among trials with contrast_mag > 0, at least
    one stim side disagrees with the perseverated choice — or if the run has
    no non-zero-contrast trials at all. Block identity is ignored.
    '''
    choice = np.asarray(choice, dtype=float)
    stim_is_left = np.asarray(stim_is_left, dtype=bool)
    contrast_mag = np.asarray(contrast_mag, dtype=float)
    n = len(choice)
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    keep_head = max(int(min_run) - 1, 0)

    def _stim_matches_choice(i, ch):
        left = bool(stim_is_left[i])
        return (ch == 1 and left) or (ch == -1 and (not left))

    t = 0
    while t < n:
        ch = choice[t]
        if ch not in (-1.0, 1.0):
            t += 1
            continue
        t1 = t + 1
        while t1 < n and choice[t1] == ch:
            t1 += 1
        run_len = t1 - t
        if run_len >= min_run:
            nz = [i for i in range(t, t1) if contrast_mag[i] > 0]
            poorly = (len(nz) == 0) or any(
                not _stim_matches_choice(i, ch) for i in nz)
            if poorly:
                # Tail only: drop from 0-based offset keep_head within the run.
                out[t + keep_head:t1] = True
        t = t1
    return out


def late_session_mask(n_trials, late_frac=STICKY_LATE_FRAC):
    '''True on the last ``late_frac`` of trials (session temporal order).'''
    n = int(n_trials)
    keep_n = int(np.floor((1.0 - float(late_frac)) * n))
    mask = np.zeros(n, dtype=bool)
    if keep_n < n:
        mask[keep_n:] = True
    return mask


def sticky_trial_exclusion_mask(trials, late_frac=STICKY_LATE_FRAC,
                                min_run=STICKY_MIN_RUN):
    '''
    Trials to drop: last ``late_frac`` of the session OR the **tail** of
    perseveration runs (≥ ``min_run`` same choice, poorly explained by non-0
    contrast stim; first ``min_run - 1`` trials of each such run are kept).

    Returns (drop_mask, info_dict). drop_mask True → exclude.
    '''
    choice = np.asarray(
        trials['choice'].to_numpy() if hasattr(trials['choice'], 'to_numpy')
        else trials['choice'], dtype=float)
    cl = np.asarray(
        trials['contrastLeft'].to_numpy()
        if hasattr(trials['contrastLeft'], 'to_numpy')
        else trials['contrastLeft'], dtype=float)
    cr = np.asarray(
        trials['contrastRight'].to_numpy()
        if hasattr(trials['contrastRight'], 'to_numpy')
        else trials['contrastRight'], dtype=float)
    stim_is_left = np.isnan(cr)
    contrast_mag = np.zeros(len(choice), dtype=float)
    contrast_mag[stim_is_left] = np.nan_to_num(cl[stim_is_left], nan=0.0)
    contrast_mag[~stim_is_left] = np.nan_to_num(cr[~stim_is_left], nan=0.0)

    late = late_session_mask(len(choice), late_frac=late_frac)
    pers = perseveration_run_mask(
        choice, stim_is_left, contrast_mag, min_run=min_run)
    drop = late | pers
    info = {
        'late_frac': float(late_frac),
        'min_run': int(min_run),
        'pers_mode': 'tail',
        'n_trials': int(len(choice)),
        'n_late': int(late.sum()),
        'n_perseveration': int(pers.sum()),
        'n_drop': int(drop.sum()),
        'n_keep': int((~drop).sum()),
        'n_drop_late_only': int((late & ~pers).sum()),
        'n_drop_pers_only': int((pers & ~late).sum()),
        'n_drop_both': int((late & pers).sum()),
    }
    return drop, info


def apply_sticky_trial_exclusion(trials, late_frac=STICKY_LATE_FRAC,
                                 min_run=STICKY_MIN_RUN):
    '''Return filtered trials table + exclusion info (keeps temporal order).'''
    drop, info = sticky_trial_exclusion_mask(
        trials, late_frac=late_frac, min_run=min_run)
    keep = ~drop
    if hasattr(trials, 'iloc'):
        out = trials.iloc[np.where(keep)[0]].copy()
    else:
        out = {k: np.asarray(v)[keep] for k, v in trials.items()}
    print(
        f'sticky exclusion: keep {info["n_keep"]}/{info["n_trials"]} '
        f'(late={info["n_late"]}, pers={info["n_perseveration"]}, '
        f'drop={info["n_drop"]})')
    return out, info


def configure_excl_sticky_output_dirs(cache_dir=None):
    '''Point pth_res / stream_acc at manifold/res_excl_sticky (avoid overwrite).'''
    global pth_res, pth_stream_acc
    root = Path(cache_dir or one.cache_dir)
    pth_res = root / 'manifold' / 'res_excl_sticky'
    pth_res.mkdir(parents=True, exist_ok=True)
    pth_stream_acc = pth_res / '_stream_acc'
    pth_stream_acc.mkdir(parents=True, exist_ok=True)
    print(f'excl-sticky outputs -> {pth_res}')
    return pth_res


def _null_labels(split, ntr, dx, choices=None):
    '''One boolean trial label vector for a null draw.

    ``choices``: choice (±1) in temporal trial order (same as sorted dx[:,1]).
    Used for stim_block nulls (shuffle stim-side labels within choice class).
    '''
    if 'block_only' in split:
        return generate_pseudo_blocks(ntr, first5050=0) == 0.8
    order = np.argsort(dx[:, 1])
    tr_c = dx[order][:, 0]
    if 'stim_block' in split and choices is not None:
        # Port of choicestim_analysis: shuffle stim sides within choice classes.
        tr_c2 = np.array(tr_c, copy=True)
        y_ = np.asarray(choices)
        for cc in [y_ == 1, y_ != 1]:
            r = tr_c[cc]
            tr_c2[cc] = np.array(random.sample(list(r), len(r)))
        return tr_c2 == 1
    tr_c2 = np.array(random.sample(list(tr_c), len(tr_c)))
    return tr_c2 == 1


def _region_perm_metrics(m0, m1, v0, v1, b, half1, half2, ys, reg_mask):
    '''
    Regional distance curves for one (true or null) label split.
    Returns region-summed (d_var_m, d_euc_m, d_xnobis) each shape (nbins,).
    '''
    m0_r = m0[reg_mask]
    m1_r = m1[reg_mask]
    v0_r = v0[reg_mask]
    v1_r = v1[reg_mask]
    d_var = (((m0_r - m1_r) / ((v0_r + v1_r) ** 0.5)) ** 2)
    d_euc = (m0_r - m1_r) ** 2

    m1_h1 = b[half1 & ys][:, reg_mask, :].mean(axis=0)
    m0_h1 = b[half1 & ~ys][:, reg_mask, :].mean(axis=0)
    m1_h2 = b[half2 & ys][:, reg_mask, :].mean(axis=0)
    m0_h2 = b[half2 & ~ys][:, reg_mask, :].mean(axis=0)
    dmu_h1 = m1_h1 - m0_h1
    dmu_h2 = m1_h2 - m0_h2
    var_pooled = 0.5 * (v0_r + v1_r)
    inv_var = 1.0 / (var_pooled + 1e-12)
    d_xcv_bins = np.nansum(dmu_h1 * inv_var * dmu_h2, axis=0)

    return np.nansum(d_var, axis=0), np.sum(d_euc, axis=0), d_xcv_bins


def _compute_control_D(b, bins, acs, acs1, dx, half1, half2, ntr, nrand, split,
                       null_batch_size=NULL_BATCH_SIZE, choices=None):
    '''
    Batched null loop: one full-tensor mean/var per null, then regional metrics.
    Matches original loop order (perm outer, region inner) without materialising
    ws of shape (2*(nrand+1), n_neurons, nbins).
    '''
    m0_true = bins[0].mean(axis=0)
    m1_true = bins[1].mean(axis=0)
    v0_true = bins[0].var(axis=0)
    v1_true = bins[1].var(axis=0)
    ys_true = dx[np.argsort(dx[:, 1])][:, 0].astype(bool)

    regs = list(Counter(acs).keys())
    reg_masks = {reg: (acs == reg) for reg in regs}
    D = {
        reg: {
            'nclus': int(np.sum(acs1 == reg)),
            'd_vars': [],
            'd_eucs': [],
            'd_xnobis': [],
        }
        for reg in regs
    }
    label_perms = [ys_true]

    def _append_perm(m0, m1, v0, v1, ys):
        for reg in regs:
            dv, de, dxn = _region_perm_metrics(
                m0, m1, v0, v1, b, half1, half2, ys, reg_masks[reg])
            D[reg]['d_vars'].append(dv)
            D[reg]['d_eucs'].append(de)
            D[reg]['d_xnobis'].append(dxn)

    # j=0: true condition means from bins (not label-shuffled on b)
    _append_perm(m0_true, m1_true, v0_true, v1_true, ys_true)

    for batch_start in range(0, nrand, null_batch_size):
        batch_end = min(batch_start + null_batch_size, nrand)
        for _ in range(batch_start, batch_end):
            ys = _null_labels(split, ntr, dx, choices=choices)
            label_perms.append(ys)
            _append_perm(
                b[ys].mean(axis=0), b[~ys].mean(axis=0),
                b[ys].var(axis=0), b[~ys].var(axis=0),
                ys,
            )

    d_var = (((m0_true - m1_true) / ((v0_true + v1_true) ** 0.5)) ** 2)
    d_euc = (m0_true - m1_true) ** 2
    return {
        'acs': acs,
        'acs1': acs1,
        'd_vars': d_var,
        'd_eucs': d_euc,
        'ws': np.array([m0_true, m1_true])[:ntravis],
        'uperms': len(np.unique([str(x.astype(int)) for x in label_perms])),
        'D': D,
    }


def _choice_donors_path():
    return Path(one.cache_dir, 'manifold', 'choice_donors.npy')


def _normalize_donor_rec(rec):
    '''
    Donor bank entry → dict with choice, stim_is_left, pleft_true.

    Backward compatible with legacy ``{eid: choice_array}``.
    '''
    if isinstance(rec, dict) and 'choice' in rec:
        return {
            'choice': np.asarray(rec['choice'], dtype=float),
            'stim_is_left': np.asarray(rec['stim_is_left'], dtype=bool),
            'pleft_true': np.asarray(rec['pleft_true'], dtype=float),
        }
    ch = np.asarray(rec, dtype=float)
    return {
        'choice': ch,
        'stim_is_left': np.zeros(len(ch), dtype=bool),
        'pleft_true': np.full(len(ch), np.nan),
        '_legacy': True,
    }


def build_choice_donor_bank(restart=True):
    '''
    Scan manifold/insertion_cache for unique-eid trial metadata.

    Saved to manifold/choice_donors.npy as
    ``{eid: {choice, stim_is_left, pleft_true}}``. Harris nulls use the full
    ``choice`` sequence indexed by the recipient's stratified trial numbers;
    stim/pleft fields remain for diagnostics / legacy tools.
    '''
    path = _choice_donors_path()
    if restart and path.exists():
        bank = np.load(path, allow_pickle=True).item()
        print(f'choice donor bank (cached): {len(bank)} eids -> {path}')
        return bank

    bank = {}
    cache_dir = Path(one.cache_dir, 'manifold', 'insertion_cache')
    if not cache_dir.exists():
        print(f'WARNING: no insertion_cache at {cache_dir}; donor bank empty')
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, bank, allow_pickle=True)
        return bank

    for fpath in sorted(cache_dir.glob('*.npy')):
        try:
            cache = np.load(fpath, allow_pickle=True).item()
        except Exception as exc:
            print(f'WARNING: skip corrupt cache {fpath.name}: {exc}')
            continue
        eid = str(cache.get('eid') or '')
        if not eid or eid in bank:
            continue
        trials_by = cache.get('trials') or {}
        if not trials_by:
            continue
        tdf = next(iter(trials_by.values()))
        choice = np.asarray(
            tdf['choice'].to_numpy() if hasattr(tdf['choice'], 'to_numpy')
            else tdf['choice'], dtype=float)
        # Left stim present ↔ contrastRight is NaN (same as session-shuffle path).
        cr = tdf['contrastRight']
        cr = cr.to_numpy() if hasattr(cr, 'to_numpy') else np.asarray(cr)
        stim_is_left = np.isnan(cr.astype(float))
        pleft = tdf['probabilityLeft']
        pleft = np.asarray(
            pleft.to_numpy() if hasattr(pleft, 'to_numpy') else pleft,
            dtype=float)
        if not (len(choice) == len(stim_is_left) == len(pleft)):
            print(f'WARNING: skip {eid}: length mismatch in donor fields')
            continue
        bank[eid] = {
            'choice': choice,
            'stim_is_left': stim_is_left,
            'pleft_true': pleft,
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, bank, allow_pickle=True)
    print(f'choice donor bank: {len(bank)} eids -> {path}')
    return bank


def load_choice_donor_bank():
    '''Load donor bank; build from insertion_cache if missing.'''
    path = _choice_donors_path()
    if path.exists():
        return np.load(path, allow_pickle=True).item()
    return build_choice_donor_bank(restart=False)



def _sample_harris_ys(elig_idx, donor_bank, eid, choices_true=None,
                      max_tries=HARRIS_MAX_TRIES, rng=None):
    """
    Harris-style null labels: take another session's **full** choice sequence
    and read choices at the recipient's stratified trial indices ``elig_idx``.

    Stratification (stim×prior) is applied only on the real session to define
    which trial numbers enter the neural tensor; the donor is not re-stratified.
    """
    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, np.random.RandomState):
        rng = np.random.default_rng(rng.randint(0, 2**31 - 1))

    eid = str(eid)
    elig_idx = np.asarray(elig_idx, dtype=int)
    n_elig = len(elig_idx)
    need = int(elig_idx.max()) + 1 if n_elig else 0

    def _ok(ys):
        return (int(ys.sum()) >= min_trials_per_side
                and int((~ys).sum()) >= min_trials_per_side)

    candidates = []
    for e, rec in (donor_bank or {}).items():
        if str(e) == eid:
            continue
        ch = _normalize_donor_rec(rec)['choice']
        if len(ch) >= need:
            candidates.append(np.asarray(ch, dtype=float))

    warned = False
    for _ in range(max_tries):
        if candidates:
            ch = candidates[int(rng.integers(0, len(candidates)))]
            cand = ch[elig_idx]
        else:
            if not warned:
                print('harris-null: no donor with len≥'
                      f'{need}; circular-shift own eligible choices')
                warned = True
            if choices_true is None or n_elig <= 1:
                cand = np.asarray(choices_true, dtype=float) if choices_true is not None else np.ones(n_elig)
            else:
                cand = np.roll(np.asarray(choices_true, dtype=float),
                               int(rng.integers(1, n_elig)))
        ys = cand == 1
        if _ok(ys):
            return ys

    if choices_true is not None and n_elig > 1:
        for _ in range(max_tries):
            ys = np.roll(np.asarray(choices_true, dtype=float),
                         int(rng.integers(1, n_elig))) == 1
            if _ok(ys):
                return ys
    if choices_true is not None:
        return np.asarray(choices_true, dtype=float) == 1
    return np.zeros(n_elig, dtype=bool)


def _compute_control_D_harris(
        b, acs, acs1, choices_true, half1, half2, ntr, nrand, split,
        donor_bank, eid, elig_idx, null_batch_size=NULL_BATCH_SIZE):
    """
    Literal Harris session-permutation null (behavior of another eid onto this
    session's neural tensor).

    Observed distance uses real choices on stim×prior–eligible trials.
    Each null draw indexes another session's full choice sequence at the same
    ``elig_idx`` trial numbers (no donor-side stratification).
    """
    choices_true = np.asarray(choices_true, dtype=float)
    ys_true = choices_true == 1
    m0_true = b[ys_true].mean(axis=0)
    m1_true = b[~ys_true].mean(axis=0)
    v0_true = b[ys_true].var(axis=0)
    v1_true = b[~ys_true].var(axis=0)

    regs = list(Counter(acs).keys())
    reg_masks = {reg: (acs == reg) for reg in regs}
    D = {
        reg: {
            'nclus': int(np.sum(acs1 == reg)),
            'd_vars': [],
            'd_eucs': [],
            'd_xnobis': [],
        }
        for reg in regs
    }
    label_perms = [ys_true]

    def _append_perm(m0, m1, v0, v1, ys):
        for reg in regs:
            dv, de, dxn = _region_perm_metrics(
                m0, m1, v0, v1, b, half1, half2, ys, reg_masks[reg])
            D[reg]['d_vars'].append(dv)
            D[reg]['d_eucs'].append(de)
            D[reg]['d_xnobis'].append(dxn)

    _append_perm(m0_true, m1_true, v0_true, v1_true, ys_true)

    eid = str(eid)
    elig_idx = np.asarray(elig_idx, dtype=int)
    need = int(elig_idx.max()) + 1 if len(elig_idx) else 0
    n_ok = sum(
        1 for e, rec in (donor_bank or {}).items()
        if str(e) != eid and len(_normalize_donor_rec(rec)['choice']) >= need)
    print(f'harris-null [{split}]: {n_ok} donors with len≥{need} '
          f'(elig={len(elig_idx)})')

    rng = np.random.default_rng()
    for batch_start in range(0, nrand, null_batch_size):
        batch_end = min(batch_start + null_batch_size, nrand)
        for _ in range(batch_start, batch_end):
            ys = _sample_harris_ys(
                elig_idx, donor_bank, eid, choices_true=choices_true, rng=rng)
            label_perms.append(ys)
            _append_perm(
                b[ys].mean(axis=0), b[~ys].mean(axis=0),
                b[ys].var(axis=0), b[~ys].var(axis=0),
                ys,
            )

    d_var = (((m0_true - m1_true) / ((v0_true + v1_true) ** 0.5)) ** 2)
    d_euc = (m0_true - m1_true) ** 2
    return {
        'acs': acs,
        'acs1': acs1,
        'd_vars': d_var,
        'd_eucs': d_euc,
        'ws': np.array([m0_true, m1_true])[:ntravis],
        'uperms': len(np.unique([str(x.astype(int)) for x in label_perms])),
        'D': D,
        'null_scheme': 'harris_session_permutation',
    }

def _compute_control_D_actkernel_choice(
        b, acs, acs1, choices_true, half1, half2, ntr, nrand, split,
        trials, elig_idx, eid, null_batch_size=NULL_BATCH_SIZE):
    '''
    BWM-style ActionKernel synthetic-session nulls.

    Fit ActionKernel once on the real session. Each null draw regenerates a
    pseudo stim/block stream (``generate_pseudo_session``) and simulates
    choices under the fitted θ (``synthetic_sessions_from_trials`` /
    ``make_synthetic_session``). Null labels for neural ``b`` are those
    synthetic choices at the real session's stratified ``elig_idx`` (same
    indexing pattern as Harris).
    '''
    choices_true = np.asarray(choices_true, dtype=float)
    ys_true = choices_true == 1
    m0_true = b[ys_true].mean(axis=0)
    m1_true = b[~ys_true].mean(axis=0)
    v0_true = b[ys_true].var(axis=0)
    v1_true = b[~ys_true].var(axis=0)

    regs = list(Counter(acs).keys())
    reg_masks = {reg: (acs == reg) for reg in regs}
    D = {
        reg: {
            'nclus': int(np.sum(acs1 == reg)),
            'd_vars': [],
            'd_eucs': [],
            'd_xnobis': [],
        }
        for reg in regs
    }
    label_perms = [ys_true]

    def _append_perm(m0, m1, v0, v1, ys):
        for reg in regs:
            dv, de, dxn = _region_perm_metrics(
                m0, m1, v0, v1, b, half1, half2, ys, reg_masks[reg])
            D[reg]['d_vars'].append(dv)
            D[reg]['d_eucs'].append(de)
            D[reg]['d_xnobis'].append(dxn)

    _append_perm(m0_true, m1_true, v0_true, v1_true, ys_true)

    fit = get_actkernel_choice_fit(eid, trials)
    elig_idx = np.asarray(elig_idx, dtype=int)
    print(f'actkernel-choice [{split}]: BWM pseudo-session null; '
          f'fit mode={fit.get("mode")} '
          f'params={np.array2string(np.asarray(fit["params"]), precision=3)}')

    def _ok(ys):
        return (int(ys.sum()) >= min_trials_per_side
                and int((~ys).sum()) >= min_trials_per_side)

    syn = _syn()
    rng = np.random.default_rng()
    seed_base = int(rng.integers(0, 2**31 - 1))
    n_done = 0
    gen_offset = 0
    # Batch-generate BWM synthetic sessions; reject draws with too few L/R
    # on elig_idx (same min_trials_per_side gate as other structured nulls).
    while n_done < nrand:
        need = nrand - n_done
        n_gen = max(need, min(null_batch_size, need + max(need // 5, 5)))
        out = syn.synthetic_sessions_from_trials(
            trials, n=n_gen, eid=str(eid), subject='bwm',
            params=fit['params'], seed=seed_base + gen_offset, fast=True)
        gen_offset += n_gen
        ch_mat = np.asarray(out['choice'], dtype=float)
        for i in range(n_gen):
            ys = ch_mat[i, elig_idx] == 1
            if not _ok(ys):
                continue
            label_perms.append(ys)
            _append_perm(
                b[ys].mean(axis=0), b[~ys].mean(axis=0),
                b[ys].var(axis=0), b[~ys].var(axis=0),
                ys,
            )
            n_done += 1
            if n_done >= nrand:
                break
        if n_gen > 0 and n_done < nrand and gen_offset > nrand * 20:
            # Pathological: fall back to single-draw sampler with circular shift.
            print('WARNING: actkernel BWM null: too many rejected draws; '
                  'falling back to per-draw make_synthetic_session')
            while n_done < nrand:
                ys = _sample_actkernel_choice_ys(
                    elig_idx, trials, fit, rng=rng)
                label_perms.append(ys)
                _append_perm(
                    b[ys].mean(axis=0), b[~ys].mean(axis=0),
                    b[ys].var(axis=0), b[~ys].var(axis=0),
                    ys,
                )
                n_done += 1
            break

    d_var = (((m0_true - m1_true) / ((v0_true + v1_true) ** 0.5)) ** 2)
    d_euc = (m0_true - m1_true) ** 2
    return {
        'acs': acs,
        'acs1': acs1,
        'd_vars': d_var,
        'd_eucs': d_euc,
        'ws': np.array([m0_true, m1_true])[:ntravis],
        'uperms': len(np.unique([str(x.astype(int)) for x in label_perms])),
        'D': D,
        'null_scheme': 'synthetic_choice_actkernel',
        'actkernel_fit_mode': fit.get('mode'),
        'actkernel_params': np.asarray(fit['params'], dtype=float),
    }


def _bin_spike_events(spikes, clusters, events, split):
    '''Overlapping bins for one list of event times → (n_trials, n_clus, n_bins).'''
    bis = []
    st = int(T_BIN(split) // sts)
    for ts in range(st):
        bi, _ = bin_spikes2D(
            spikes['times'],
            clusters['cluster_id'][spikes['clusters']],
            clusters['cluster_id'],
            np.array(events) + ts * sts,
            pre_post[split][0], pre_post[split][1],
            T_BIN(split))
        bis.append(bi)
    ntr, nn, nbin = bi.shape
    ar = np.zeros((ntr, nn, st * nbin))
    for ts in range(st):
        ar[:, :, ts::st] = bis[ts]
    return ar


def _get_d_vars_session_shuffle(
        split, trials, spikes, clusters, mapping, control, nrand,
        null_batch_size, donor_bank, eid, actkernel_choice_null=False):
    '''
    Choice L vs R under fixed stim (± prior/block), with structured nulls.

    Stratification (stim×prior) is applied only on this session to define
    eligible trial indices and the neural tensor ``b``. Observed distance uses
    real choices on those trials. Nulls:

    - ``actkernel_choice_null``: BWM-style ActionKernel pseudo-sessions
      (new stim/blocks + choices under fitted θ); labels at ``elig_idx``.
    - else (Harris): index another eid's full choice sequence at the same
      ``elig_idx`` trial numbers (no donor-side stratification).
    '''
    alignment = align[split]
    # choice_stim_* / choice_duringstim_* route via substring stim_l / stim_r.
    if 'stim_l' in split:
        stim_nan_col = 'contrastRight'  # left stimulus present
    elif 'stim_r' in split:
        stim_nan_col = 'contrastLeft'
    else:
        print('what is the split?', split)
        return None

    if 'block_l' in split:
        pleft = 0.8
    elif 'block_r' in split:
        pleft = 0.2
    else:
        pleft = None

    elig = np.isnan(trials[stim_nan_col].to_numpy()
                    if hasattr(trials[stim_nan_col], 'to_numpy')
                    else trials[stim_nan_col])
    if pleft is not None:
        elig = np.asarray(elig) & (trials['probabilityLeft'].to_numpy() == pleft)
    else:
        elig = np.asarray(elig)

    elig_idx = np.arange(len(trials))[elig]
    if len(elig_idx) == 0:
        raise InsufficientTrials('no eligible trials for structured-choice null')

    choices_true = trials['choice'].to_numpy()[elig_idx].astype(float)
    events_all = trials[alignment].to_numpy()[elig_idx]
    n_left = int(np.sum(choices_true == 1))
    n_right = int(np.sum(choices_true == -1))
    null_tag = ('actkernel-choice null' if actkernel_choice_null
                else 'harris session-permutation null')
    print('#trials per condition: ', n_left, n_right,
          f'(eligible={len(elig_idx)}, {null_tag})')
    if n_left < min_trials_per_side or n_right < min_trials_per_side:
        raise InsufficientTrials(
            f'need ≥{min_trials_per_side} trials/side, got {n_left}, {n_right}')

    assert len(spikes['times']) == len(spikes['clusters']), 'spikes != clusters'

    b = _bin_spike_events(spikes, clusters, events_all, split)
    half1 = (np.arange(b.shape[0]) % 2 == 0)
    half2 = ~half1
    ntr, nclus, nbins = b.shape

    acs = np.array(br.id2acronym(clusters['atlas_id'], mapping=mapping))
    wsc = np.concatenate(b, axis=1)
    goodcells_count = [
        k for k in range(wsc.shape[0])
        if (not np.isnan(wsc[k]).any() and wsc[k].any())
    ]
    acs1 = acs[goodcells_count]
    goodcells = ~np.bitwise_or.reduce([acs == reg for reg in ['void', 'root']])
    goodcells1 = ~np.bitwise_or.reduce([acs1 == reg for reg in ['void', 'root']])
    acs = acs[goodcells]
    acs1 = acs1[goodcells1]
    b = b[:, goodcells, :]

    if control:
        if actkernel_choice_null:
            return _compute_control_D_actkernel_choice(
                b, acs, acs1, choices_true, half1, half2, ntr, nrand, split,
                trials=trials, elig_idx=elig_idx, eid=eid,
                null_batch_size=null_batch_size,
            )
        if not donor_bank:
            print('WARNING: empty donor_bank; harris-null falls back to '
                  'circular shifts of own eligible choices')
        return _compute_control_D_harris(
            b, acs, acs1, choices_true, half1, half2, ntr, nrand, split,
            donor_bank=donor_bank or {}, eid=eid, elig_idx=elig_idx,
            null_batch_size=null_batch_size,
        )

    print('all trials')
    ys = choices_true == 1
    bins = [b[ys], b[~ys]]
    w0 = [bi.mean(axis=0) for bi in bins]
    s0 = [bi.var(axis=0) for bi in bins]
    ws = np.array(w0)
    ss = np.array(s0)
    d_var = (((ws[0] - ws[1]) / ((ss[0] + ss[1]) ** 0.5)) ** 2)
    d_euc = (ws[0] - ws[1]) ** 2
    return {
        'acs': acs,
        'acs1': acs1,
        'd_vars': d_var,
        'd_eucs': d_euc,
        'ws': ws[:ntravis],
        'null_scheme': (
            'synthetic_choice_actkernel' if actkernel_choice_null
            else 'harris_session_permutation'),
    }


# Function to calculate action kernel
def action_kernel_priors(alpha, actions):
    
    # initialization
    prior = 0.5
    priors = [prior]
    
    # calculate action kernel for each trial
    for t in range(len(actions)-1):
        action = actions[t]
        prior = alpha * int(action>0) + (1-alpha) * prior
        priors.append(prior)
    
    binary_priors = np.double(list(np.double(priors)>=0.5))
    binary_priors = binary_priors*0.6+0.2
    
    return priors, binary_priors


def bayesian_priors(
        stim_is_left,
        tau=BAYES_TAU,
        gamma=BAYES_GAMMA,
        min_len=BAYES_MIN_LEN,
        max_len=BAYES_MAX_LEN):
    '''
    IBL Bayes-optimal prior from stimulus history (Findling et al. Nature 2025
    SI §1.1.1 / behavior_models.OptimalBayesian).

    Infers P(stim left on trial t | sides on trials 1..t-1) under the task
    generative model: truncated-exponential block lengths, biased blocks at
    gamma, and uncued switches. Mirrors ``action_kernel_priors`` return
    convention: continuous P(left) plus binarized 0.8 / 0.2 labels.

    Parameters
    ----------
    stim_is_left : array-like of bool
        True if the stimulus appeared on the left on that trial (contrastLeft
        finite). Length T.
    tau, gamma, min_len, max_len : float / int
        Generative-model hyperparameters (IBL defaults).

    Returns
    -------
    priors : ndarray, shape (T,)
        Continuous P(left) before each trial.
    binary_priors : ndarray, shape (T,)
        0.8 if priors >= 0.5 else 0.2 (same encoding as action kernel).
    '''
    stim_is_left = np.asarray(stim_is_left, dtype=bool)
    T = len(stim_is_left)
    if T == 0:
        return np.array([]), np.array([])

    # Hazard H(n) for current block length n in {1..max_len}.
    ns = np.arange(1, max_len + 1)
    h = np.exp(-ns / float(tau))
    h[(ns < min_len) | (ns > max_len)] = 0.0
    H = np.zeros(max_len)
    for i in range(max_len):
        denom = h[i:].sum()
        H[i] = (h[i] / denom) if denom > 0 else 0.0

    # State: length index 0..max_len-1 (= length 1..max_len), block b in {-1,0,1}.
    b_vals = np.array([-1, 0, 1], dtype=int)
    n_b = 3
    # p(stim left | b)
    p_left_given_b = np.array([gamma, 0.5, 1.0 - gamma], dtype=float)

    # g[l, bi] = p(length, block, s_1:(t-1)); h_post after observing s_t.
    g = np.zeros((max_len, n_b), dtype=float)
    g[0, 1] = 1.0  # length=1, unbiased block

    priors = np.empty(T, dtype=float)
    for t in range(T):
        if t >= 1:
            g_new = np.zeros_like(g)
            for lp in range(max_len):
                Hp = H[lp]
                for bip, bp in enumerate(b_vals):
                    mass = g[lp, bip]
                    if mass == 0.0:
                        continue
                    # Switch: length → 1
                    if Hp > 0.0:
                        if bp == 0:
                            for bi_new, b_new in enumerate(b_vals):
                                if b_new != 0:
                                    g_new[0, bi_new] += mass * Hp * 0.5
                        else:
                            b_new = -bp
                            bi_new = int(b_new + 1)  # -1→0, 1→2
                            g_new[0, bi_new] += mass * Hp
                    # Continue: length → length+1
                    if Hp < 1.0 and lp + 1 < max_len:
                        g_new[lp + 1, bip] += mass * (1.0 - Hp)
            g = g_new

        z = g.sum()
        if z <= 0:
            priors[t] = 0.5
        else:
            p_b = g.sum(axis=0) / z
            priors[t] = float(np.dot(p_b, p_left_given_b))

        # Incorporate observation s_t for next trial's g.
        p_obs = np.where(stim_is_left[t], p_left_given_b, 1.0 - p_left_given_b)
        g = g * p_obs[np.newaxis, :]
        z = g.sum()
        if z > 0:
            g /= z
        else:
            g[:] = 0.0
            g[0, 1] = 1.0

    binary_priors = (np.asarray(priors, dtype=float) >= 0.5).astype(float)
    binary_priors = binary_priors * 0.6 + 0.2
    return priors, binary_priors


def T_BIN(split, b_size=b_size):

    # c = 0.005 # time bin size in seconds (5 ms)
    if b_size is None:
        return pre_post[split][0] + pre_post[split][1]
    else:
        return b_size  


def grad(c,nobs):
    cmap = mpl.cm.get_cmap(c)
    
    return [cmap(0.5*(nobs - p)/nobs) for p in range(nobs)]

def get_name(brainregion):
    regid = br.id[np.argwhere(br.acronym == brainregion)][0, 0]
    return br.name[np.argwhere(br.id == regid)[0, 0]]


def generate_pseudo_blocks(
        n_trials,
        factor=60,
        min_=20,
        max_=100,
        first5050=90):
    """
    Generate a pseudo block structure
    Parameters
    ----------
    n_trials : int
        how many trials to generate
    factor : int
        factor of the exponential
    min_ : int
        minimum number of trials per block
    max_ : int
        maximum number of trials per block
    first5050 : int
        amount of trials with 50/50 left right probability at the beginning
    Returns
    ---------
    probabilityLeft : 1D array
        array with probability left per trial
    """

    block_ids = []
    while len(block_ids) < n_trials:
        x = np.random.exponential(factor)
        while (x <= min_) | (x >= max_):
            x = np.random.exponential(factor)
        if (len(block_ids) == 0) & (np.random.randint(2) == 0):
            block_ids += [0.2] * int(x)
        elif (len(block_ids) == 0):
            block_ids += [0.8] * int(x)
        elif block_ids[-1] == 0.2:
            block_ids += [0.8] * int(x)
        elif block_ids[-1] == 0.8:
            block_ids += [0.2] * int(x)
    return np.array([0.5] * first5050 + block_ids[:n_trials - first5050])


def saturation_for_split(split):
    '''Saturation-interval key used to load/mask trials for a split's alignment.'''
    # Match choicestim_analysis: stim_choice / stim_block use stim sat;
    # choice_stim / duringchoice use move sat.
    if ('duringstim' in split or 'stim_choice' in split
            or 'stim_block' in split):
        return 'saturation_stim_plus04'
    if 'duringchoice' in split or 'choice_stim' in split:
        return 'saturation_move_minus02'
    if 'fback' in split:
        return 'saturation_feedback_plus04'
    return 'saturation_stim_plus04'


SATURATION_TYPES = (
    'saturation_stim_plus04',
    'saturation_move_minus02',
    'saturation_feedback_plus04',
)


def _apply_saturation_mask(trials, base_mask, one, eid, saturation_intervals):
    '''
    Saturation exclusion on top of load_trials_and_mask base mask.

    load_trials_and_mask(..., saturation_intervals=st) can assert when
    truncate_to_pass shortens trials vs the aggregate table; we apply saturation
    here with row alignment instead.
    '''
    all_trials = pd.read_parquet(download_aggregate_tables(one, type='trials'))
    sess_trials = all_trials[all_trials['eid'] == str(eid)].copy()
    sess_trials.reset_index(drop=True, inplace=True)
    n_sess = trials.shape[0]
    if len(sess_trials) > n_sess:
        sess_trials = sess_trials.iloc[:n_sess]
    elif len(sess_trials) < n_sess:
        raise AssertionError(
            f'Trials table ({len(sess_trials)}) shorter than session ({n_sess}) for {eid}.'
        )
    intervals = (
        [saturation_intervals]
        if isinstance(saturation_intervals, str)
        else list(saturation_intervals)
    )
    mask = base_mask.to_numpy().copy()
    for interval in intervals:
        mask[sess_trials[interval].to_numpy() == True] = False
    return trials, mask


def load_trials_for_saturation(one, eid, saturation_intervals):
    '''Trials + mask for one saturation key; safe when truncate_to_pass applies.'''
    trials, base_mask = load_trials_and_mask(one, eid, saturation_intervals=None)
    return _apply_saturation_mask(trials, base_mask, one, eid, saturation_intervals)


def build_insertion_cache(pid, satur_types=SATURATION_TYPES, save=True, restart=True):
    '''
    Load an insertion's raw data ONCE (the expensive step) and cache it so every
    split reuses it instead of re-loading per split.

    Caches: spikes (times, clusters), clusters (cluster_id, atlas_id), and the
    bad-trial-masked trials table for each saturation type (one per alignment
    event: stim / move / feedback). Saved to manifold/insertion_cache/{eid_probe}.npy.
    '''
    eid, probe = one.pid2eid(pid)
    eid_probe = f'{eid}_{probe}'
    cpath = Path(one.cache_dir, 'manifold', 'insertion_cache', f'{eid_probe}.npy')
    if restart and cpath.exists():
        return np.load(cpath, allow_pickle=True).item()

    spikes, clusters = load_good_units(one, pid)
    trials, base_mask = load_trials_and_mask(one, eid, saturation_intervals=None)
    trials_by_satur = {}
    for st in satur_types:
        _, mask = _apply_saturation_mask(trials, base_mask, one, eid, st)
        trials_by_satur[st] = trials[mask]

    cache = {
        'pid': pid,
        'eid': eid,
        'probe': probe,
        'spikes': {'times': spikes['times'], 'clusters': spikes['clusters']},
        'clusters': {'cluster_id': clusters['cluster_id'], 'atlas_id': clusters['atlas_id']},
        'trials': trials_by_satur,
    }
    if save:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        np.save(cpath, cache, allow_pickle=True)
    return cache


def get_d_vars(split, pid, mapping='Beryl', lowcontrast=False,
               control=True, nrand=2000, bycontrast=False, cached=None,
               null_batch_size=NULL_BATCH_SIZE, donor_bank=None,
               session_shuffle_null=False,
               actkernel_choice_null=False,
               exclude_sticky_trials=False,
               sticky_late_frac=STICKY_LATE_FRAC,
               sticky_min_run=STICKY_MIN_RUN):

    '''
    for a given session, probe, bin neural activity
    cut into trials, compute d_var per region

    ``cached``: optional per-insertion cache from build_insertion_cache. When
    provided, spikes/clusters/trials are reused (no per-split reload), which is
    the time-efficient path. When None, loads from ONE as before (identical result).

    ``actkernel_choice_null``: if True and split is choice_stim* /
    choice_duringstim*, use BWM-style ActionKernel synthetic-session nulls
    (``synthetic_sessions_from_trials`` / ``make_synthetic_session``: new
    pseudo stim/blocks + choices under fitted θ; null labels at real
    ``elig_idx``). Takes precedence over session_shuffle_null.

    ``session_shuffle_null``: if True and split is choice_stim* / choice_duringstim*,
    use literal Harris session-permutation nulls: stratify stim×prior only on
    this session to define ``elig_idx`` / neural ``b``; null labels are another
    eid's full choice sequence indexed at those trial numbers (requires
    ``donor_bank``). Default False → label shuffle.

    ``exclude_sticky_trials``: drop last ``sticky_late_frac`` of the session and
    the **tail** of perseveration runs (≥ ``sticky_min_run`` same choice poorly
    explained by non-0 contrast stim; keep first ``sticky_min_run - 1`` trials
    of each such run). Intended for choice L–R sensitivity analyses with
    label-shuffle nulls within stim×block.
    '''
    
    
    saturation_intervals = saturation_for_split(split)

    if cached is not None:
        # Reuse the once-loaded insertion data (spikes/clusters/masked trials).
        spikes = cached['spikes']
        clusters = cached['clusters']
        trials = cached['trials'][saturation_intervals].copy()
        eid = cached.get('eid')
        probe = cached.get('probe')
        if eid is None or probe is None:
            eid, probe = one.pid2eid(pid)
    else:
        eid, probe = one.pid2eid(pid)
        # load in spikes
        spikes, clusters = load_good_units(one, pid)

        # Load in trials data and mask bad trials (False if bad)
        trials, mask = load_trials_for_saturation(one, eid, saturation_intervals)
        # remove certain trials
        trials = trials[mask]
    # Bayes-optimal prior needs the full stimulus history (incl. 0.5 blocks).
    # Compute before dropping unbiased trials; labels are applied after.
    if 'bayes' in split:
        stim_is_left = ~np.isnan(trials['contrastLeft'].astype(float).values)
        bayes_cont, bayes_bin = bayesian_priors(stim_is_left)
        trials = trials.copy()
        trials['bayes_priors'] = bayes_cont
        trials['_bayes_binary'] = bayes_bin

    # Drop true 0.5 only for prior-distance *block_* splits. Choicestim family
    # keeps 0.5 trials (priors overwritten later for act/bayes).
    if 'block' in split and not is_choicestim_split(split):
        trials = trials[trials['probabilityLeft']!=0.5] # remove trials without block bias
    # rs_range = [0.08, 2]  # discard [long/short] reaction time trials
    # stim_diff = trials['firstMovement_times'] - trials['stimOn_times']
    if lowcontrast: # restrict to low contrast trials only (for block comparisons)
        rm_trials = np.bitwise_or.reduce([trials['contrastRight']>0.1,
                               trials['contrastLeft']>0.1])
        trials  = trials[~rm_trials]
        
    if 'act' in split:
        # calculate action kernel prior to use for analysis
        trials['true_priors'] = trials['probabilityLeft']
        actions = list(trials['choice'])
        trials['act_priors'], trials['probabilityLeft'] = action_kernel_priors(alpha, actions)
    elif 'bayes' in split:
        # Bayes-optimal prior (stimulus-history inference); same 0.8/0.2 labels
        trials['true_priors'] = trials['probabilityLeft']
        trials['probabilityLeft'] = trials['_bayes_binary'].values

    excl_info = None
    if exclude_sticky_trials:
        trials, excl_info = apply_sticky_trial_exclusion(
            trials, late_frac=sticky_late_frac, min_run=sticky_min_run)
        if len(trials) == 0:
            raise InsufficientTrials('no trials left after sticky exclusion')

    # Structured nulls for choice L–R: actkernel / Harris session-permutation.
    if ((actkernel_choice_null or session_shuffle_null)
            and is_choice_lr_split(split)):
        D = _get_d_vars_session_shuffle(
            split, trials, spikes, clusters, mapping, control, nrand,
            null_batch_size, donor_bank, eid,
            actkernel_choice_null=bool(actkernel_choice_null))
        if excl_info is not None and isinstance(D, dict):
            D = dict(D)
            D['trial_exclusion'] = excl_info
        return D

    # Contrast stratification from split name (..._0.125 / ..._c0.125) or legacy
    # bycontrast=True (same trailing-float convention).
    contrast = contrast_from_split(split)
    if bycontrast and contrast is None:
        try:
            contrast = float(split.split('_')[-1])
        except ValueError:
            contrast = None

    events = []
    trn = []
    if contrast is not None and split == f'stim_{contrast}':
        for side in ['Left', 'Right']:
            events.append(trials['stimOn_times'][
                np.isclose(trials[f'contrast{side}'].astype(float), contrast)])
            trn.append(np.arange(len(trials['stimOn_times']))[
                np.isclose(trials[f'contrast{side}'].astype(float), contrast)])

    def _filter_stim_side(trials_df, side):
        '''Keep stim-side trials; if contrast set, restrict to that |contrast|.'''
        col = f'contrast{side}'
        if contrast is not None:
            return trials_df[np.isclose(trials_df[col].astype(float), contrast)]
        return trials_df[~np.isnan(trials_df[col])]

    # Choicestim family (stim L–R or choice L–R). Port of choicestim_analysis
    # get_d_vars substring dispatch — must run before prior-distance stim_l/stim_r.
    if is_choicestim_split(split):
        alignment = align[split]
        if 'stim_block' in split:
            # Fixed prior; stim L vs R; no choice filter (80 ms via pre_post).
            if 'block_l' in split:
                pleft = 0.8
            elif 'block_r' in split:
                pleft = 0.2
            else:
                print('what is the split?', split)
                return
            for side in ['Right', 'Left']:
                sel = np.bitwise_and.reduce([
                    trials['probabilityLeft'] == pleft,
                    np.isnan(trials[f'contrast{side}']),
                ])
                events.append(trials[alignment][sel])
                trn.append(np.arange(len(trials['choice']))[sel])
        elif 'stim_l' in split:
            # Fixed stim Left; choice L vs R (± fixed prior).
            # Matches choice_stim_l*, choice_duringstim_l* (substring stim_l).
            if 'block_l' in split:
                pleft = 0.8
            elif 'block_r' in split:
                pleft = 0.2
            else:
                pleft = None
            for choice in [1, -1]:
                conds = [
                    trials['choice'] == choice,
                    np.isnan(trials['contrastRight']),
                ]
                if pleft is not None:
                    conds.insert(0, trials['probabilityLeft'] == pleft)
                sel = np.bitwise_and.reduce(conds)
                events.append(trials[alignment][sel])
                trn.append(np.arange(len(trials['choice']))[sel])
        elif 'stim_r' in split:
            # Fixed stim Right; choice L vs R (± fixed prior).
            if 'block_l' in split:
                pleft = 0.8
            elif 'block_r' in split:
                pleft = 0.2
            else:
                pleft = None
            for choice in [1, -1]:
                conds = [
                    trials['choice'] == choice,
                    np.isnan(trials['contrastLeft']),
                ]
                if pleft is not None:
                    conds.insert(0, trials['probabilityLeft'] == pleft)
                sel = np.bitwise_and.reduce(conds)
                events.append(trials[alignment][sel])
                trn.append(np.arange(len(trials['choice']))[sel])
        elif 'choice_l' in split:
            # Fixed choice Left; stim L vs R (± fixed prior).
            # Matches stim_choice_l*, stim_duringchoice_l*.
            if 'block_l' in split:
                pleft = 0.8
            elif 'block_r' in split:
                pleft = 0.2
            else:
                pleft = None
            for side in ['Right', 'Left']:
                conds = [
                    np.isnan(trials[f'contrast{side}']),
                    trials['choice'] == 1,
                ]
                if pleft is not None:
                    conds.insert(0, trials['probabilityLeft'] == pleft)
                sel = np.bitwise_and.reduce(conds)
                events.append(trials[alignment][sel])
                trn.append(np.arange(len(trials['choice']))[sel])
        elif 'choice_r' in split:
            # Fixed choice Right; stim L vs R (± fixed prior).
            if 'block_l' in split:
                pleft = 0.8
            elif 'block_r' in split:
                pleft = 0.2
            else:
                pleft = None
            for side in ['Right', 'Left']:
                conds = [
                    np.isnan(trials[f'contrast{side}']),
                    trials['choice'] == -1,
                ]
                if pleft is not None:
                    conds.insert(0, trials['probabilityLeft'] == pleft)
                sel = np.bitwise_and.reduce(conds)
                events.append(trials[alignment][sel])
                trn.append(np.arange(len(trials['choice']))[sel])
        else:
            print('what is the split?', split)
            return

    elif is_goal3_c0_choice_split(split):
        # Revised Goal 3: block L vs R at 0% contrast, within a fixed choice.
        # Include both nominal stimulus sides and both feedback outcomes.
        trials = trials[goal3_c0_choice_mask(trials, split)]
        for pleft in [0.8, 0.2]:
            sel = trials['probabilityLeft'] == pleft
            events.append(trials[align[split]][sel])
            trn.append(np.arange(len(trials['choice']))[sel])

    elif 'stim_l' in split:
        if split in ('act_block_duringstim_l', 'block_duringstim_l',
                     'bayes_block_duringstim_l'):
            # L vs R prior within left-stimulus trials (no choice filter);
            # post-stim window [0, 0.15] via 'durings' in name.
            trials = _filter_stim_side(trials, 'Left')
            for pleft in [0.8, 0.2]:
                events.append(trials[align[split]][trials['probabilityLeft'] == pleft])
                trn.append(np.arange(len(trials['choice']))[trials['probabilityLeft'] == pleft])
        elif 'choice_l' in split and 'f1' in split: # correct trials, f1
            trials = _filter_stim_side(trials, 'Left')
            trials = trials[trials['choice'] == 1]
            for pleft in [0.8, 0.2]:
                    events.append(trials[align[split]][np.bitwise_and.reduce([
                        trials['feedbackType'] == 1, trials['probabilityLeft'] == pleft])])
                    trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
                        trials['feedbackType'] == 1, trials['probabilityLeft'] == pleft])])
        elif 'choice_r' in split and 'f2' in split: 
            # choice_r trials, stim_l so these are incorrect trials, f2
            trials = _filter_stim_side(trials, 'Left')
            trials = trials[trials['choice'] == -1]
            for pleft in [0.8, 0.2]:
                    events.append(trials[align[split]][np.bitwise_and.reduce([
                        trials['feedbackType'] == -1,trials['probabilityLeft'] == pleft])])
                    trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
                        trials['feedbackType'] == -1,trials['probabilityLeft'] == pleft])])
        else:
            print('what is the split?', split)
            return
        
    elif 'stim_r' in split:
        if split in ('act_block_duringstim_r', 'block_duringstim_r',
                     'bayes_block_duringstim_r'):
            # L vs R prior within right-stimulus trials (no choice filter);
            # post-stim window [0, 0.15] via 'durings' in name.
            trials = _filter_stim_side(trials, 'Right')
            for pleft in [0.8, 0.2]:
                events.append(trials[align[split]][trials['probabilityLeft'] == pleft])
                trn.append(np.arange(len(trials['choice']))[trials['probabilityLeft'] == pleft])
        elif 'choice_l' in split and 'f2' in split: # incorrect trials, f2
            trials = _filter_stim_side(trials, 'Right')
            trials = trials[trials['choice'] == 1]
            for pleft in [0.8, 0.2]:
                    events.append(trials[align[split]][np.bitwise_and.reduce([
                        trials['feedbackType'] == -1,trials['probabilityLeft'] == pleft])])
                    trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
                        trials['feedbackType'] == -1,trials['probabilityLeft'] == pleft])])
        elif 'choice_r' in split and 'f1' in split: # choice_r trials, correct, f1
            trials = _filter_stim_side(trials, 'Right')
            trials = trials[trials['choice'] == -1]
            for pleft in [0.8, 0.2]:
                    events.append(trials[align[split]][np.bitwise_and.reduce([
                        trials['feedbackType'] == 1,trials['probabilityLeft'] == pleft])])
                    trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
                        trials['feedbackType'] == 1,trials['probabilityLeft'] == pleft])])
        else:
            print('what is the split?', split)
            return
   
    # elif 'srcrbl_slclbl' in split:
    #     trials = trials[trials['probabilityLeft'] == 0.8]
    #     events.append(trials[align[split]][np.bitwise_and.reduce([
    #                     trials['choice'] == -1,np.isnan(trials['contrastLeft'])])])
    #     trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
    #                     trials['choice'] == -1,np.isnan(trials['contrastLeft'])])])
    #     events.append(trials[align[split]][np.bitwise_and.reduce([
    #                     trials['choice'] == 1,np.isnan(trials['contrastRight'])])])
    #     trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
    #                     trials['choice'] == 1,np.isnan(trials['contrastRight'])])])
        
    # elif 'srcrbr_slclbr' in split:
    #     trials = trials[trials['probabilityLeft'] == 0.2]
    #     events.append(trials[align[split]][np.bitwise_and.reduce([
    #                     trials['choice'] == -1,np.isnan(trials['contrastLeft'])])])
    #     trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
    #                     trials['choice'] == -1,np.isnan(trials['contrastLeft'])])])
    #     events.append(trials[align[split]][np.bitwise_and.reduce([
    #                     trials['choice'] == 1,np.isnan(trials['contrastRight'])])])
    #     trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
    #                     trials['choice'] == 1,np.isnan(trials['contrastRight'])])])
        
    # elif 'srcrbl_slclbr' in split:
    #     #trials = trials[~rm_trials]
    #     events.append(trials[align[split]][np.bitwise_and.reduce([
    #                   trials['probabilityLeft'] == 0.8, 
    #                   trials['choice'] == -1,np.isnan(trials['contrastLeft'])])])
    #     trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
    #                trials['probabilityLeft'] == 0.8, 
    #                trials['choice'] == -1,np.isnan(trials['contrastLeft'])])])
    #     events.append(trials[align[split]][np.bitwise_and.reduce([
    #                   trials['probabilityLeft'] == 0.2, 
    #                   trials['choice'] == 1,np.isnan(trials['contrastRight'])])])
    #     trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
    #                trials['probabilityLeft'] == 0.2, 
    #                trials['choice'] == 1,np.isnan(trials['contrastRight'])])])
        
    # elif 'slclbl_srcrbr' in split:
    #     #trials = trials[~rm_trials]
    #     events.append(trials[align[split]][np.bitwise_and.reduce([
    #                   trials['probabilityLeft'] == 0.8, 
    #                   trials['choice'] == 1,np.isnan(trials['contrastRight'])])])
    #     trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
    #                trials['probabilityLeft'] == 0.8, 
    #                trials['choice'] == 1,np.isnan(trials['contrastRight'])])])
    #     events.append(trials[align[split]][np.bitwise_and.reduce([
    #                   trials['probabilityLeft'] == 0.2, 
    #                   trials['choice'] == -1,np.isnan(trials['contrastLeft'])])])
    #     trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
    #                trials['probabilityLeft'] == 0.2, 
    #                trials['choice'] == -1,np.isnan(trials['contrastLeft'])])])
        
    # elif 'slcrbl_srclbl' in split:
    #     trials = trials[trials['probabilityLeft'] == 0.8]
    #     events.append(trials[align[split]][np.bitwise_and.reduce([
    #                     trials['choice'] == -1,np.isnan(trials['contrastRight'])])])
    #     trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
    #                     trials['choice'] == -1,np.isnan(trials['contrastRight'])])])
    #     events.append(trials[align[split]][np.bitwise_and.reduce([
    #                     trials['choice'] == 1,np.isnan(trials['contrastLeft'])])])
    #     trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
    #                     trials['choice'] == 1,np.isnan(trials['contrastLeft'])])])
        
    # elif 'slcrbr_srclbr' in split:
    #     trials = trials[trials['probabilityLeft'] == 0.2]
    #     events.append(trials[align[split]][np.bitwise_and.reduce([
    #                     trials['choice'] == -1,np.isnan(trials['contrastRight'])])])
    #     trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
    #                     trials['choice'] == -1,np.isnan(trials['contrastRight'])])])
    #     events.append(trials[align[split]][np.bitwise_and.reduce([
    #                     trials['choice'] == 1,np.isnan(trials['contrastLeft'])])])
    #     trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
    #                     trials['choice'] == 1,np.isnan(trials['contrastLeft'])])])
        
    # elif 'slcrbr_srclbl' in split:
    #     #trials = trials[~rm_trials]
    #     events.append(trials[align[split]][np.bitwise_and.reduce([
    #                   trials['probabilityLeft'] == 0.2, 
    #                   trials['choice'] == -1,np.isnan(trials['contrastRight'])])])
    #     trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
    #                trials['probabilityLeft'] == 0.2, 
    #                trials['choice'] == -1,np.isnan(trials['contrastRight'])])])
    #     events.append(trials[align[split]][np.bitwise_and.reduce([
    #                   trials['probabilityLeft'] == 0.8, 
    #                   trials['choice'] == 1,np.isnan(trials['contrastLeft'])])])
    #     trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
    #                trials['probabilityLeft'] == 0.8, 
    #                trials['choice'] == 1,np.isnan(trials['contrastLeft'])])])
        
    # elif 'slcrbl_srclbr' in split:
    #     #trials = trials[~rm_trials]
    #     events.append(trials[align[split]][np.bitwise_and.reduce([
    #                   trials['probabilityLeft'] == 0.8, 
    #                   trials['choice'] == -1,np.isnan(trials['contrastRight'])])])
    #     trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
    #                trials['probabilityLeft'] == 0.8, 
    #                trials['choice'] == -1,np.isnan(trials['contrastRight'])])])
    #     events.append(trials[align[split]][np.bitwise_and.reduce([
    #                   trials['probabilityLeft'] == 0.2, 
    #                   trials['choice'] == 1,np.isnan(trials['contrastLeft'])])])
    #     trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
    #                trials['probabilityLeft'] == 0.2, 
    #                trials['choice'] == 1,np.isnan(trials['contrastLeft'])])])
    
    elif 'block_only' in split:
        # Optional contrast stratification via trailing _{c} / _c{c} on the name.
        cval = contrast_from_split(split)
        if cval is not None:
            cmag = np.nanmax(
                np.c_[trials['contrastLeft'].values.astype(float),
                      trials['contrastRight'].values.astype(float)], axis=1)
            trials = trials[np.isclose(cmag, cval)]
        for pleft in [0.8, 0.2]:
            events.append(trials[align[split]][trials['probabilityLeft'] == pleft])
            trn.append(np.arange(len(trials['choice']))[trials['probabilityLeft'] == pleft])

    else:
        print('what is the split?', split)
        return
    
    print('#trials per condition: ', len(trn[0]), len(trn[1]))
    if len(trn[0]) < min_trials_per_side or len(trn[1]) < min_trials_per_side:
        raise InsufficientTrials(
            f'need ≥{min_trials_per_side} trials/side, got {len(trn[0])}, {len(trn[1])}')

    assert len(spikes['times']) == len(spikes['clusters']), 'spikes != clusters'   
            
    # bin and cut into trials    
    bins = []

    for event in events:
    
        #  overlapping time bins, bin size = T_BIN, stride = sts 
        bis = []
        st = int(T_BIN(split)//sts) 
        
        for ts in range(st):
    
            bi, _ = bin_spikes2D(spikes['times'],
                               clusters['cluster_id'][spikes['clusters']],
                               clusters['cluster_id'],
                               np.array(event) + ts*sts, 
                               pre_post[split][0], pre_post[split][1], 
                               T_BIN(split))
            bis.append(bi)
            
        ntr, nn, nbin = bi.shape
        ar = np.zeros((ntr, nn, st*nbin))
        
        for ts in range(st):
            ar[:,:,ts::st] = bis[ts]
                           
        bins.append(ar)                   
                                              
    b = np.concatenate(bins)
    
    # recreate temporal trial order              
    dx = np.concatenate([list(zip([True]*len(trn[0]),trn[0])),
                    list(zip([False]*len(trn[1]),trn[1]))])

    b = b[np.argsort(dx[:, 1])]    
    # --- split-halves mask for crossnobis (independent halves) ---
    half1 = (np.arange(b.shape[0]) % 2 == 0)   # trials in split 1
    half2 = ~half1                              # trials in split 2
           
    ntr, nclus, nbins = b.shape 
    
    acs = br.id2acronym(clusters['atlas_id'],mapping=mapping)
               
    acs = np.array(acs)
    wsc = np.concatenate(b,axis=1)

    # Discard cells with any nan or 0 for all bins
    ## currently not doing this - keeping all cells in order to make sure
    ## same cells are included in both manifold & decoding analysis
    ## so that we can project traj onto decoded variable's direction
    goodcells = [k for k in range(wsc.shape[0]) if 
                 (not np.isnan(wsc[k]).any()
                 and wsc[k].any())]
    
    # only count the good cells, later used for normalization
    acs1 = acs[goodcells]
    #b = b[:,goodcells,:]
    #bins2 = [x[:,goodcells,:] for x in bins]
    #bins = bins2    

    # Discard cells in ill-defined regions
    goodcells = ~np.bitwise_or.reduce([acs == reg for 
                     reg in ['void','root']])
    goodcells1 = ~np.bitwise_or.reduce([acs1 == reg for 
                     reg in ['void','root']])
    
    acs = acs[goodcells]
    acs1 = acs1[goodcells1]
    b = b[:,goodcells,:]
    bins2 = [x[:,goodcells,:] for x in bins]
    bins = bins2

    if control:
        # Temporal-order choices for stim_block within-choice null shuffle.
        order = np.argsort(dx[:, 1])
        choices_ord = trials['choice'].to_numpy()[dx[order, 1].astype(int)]
        D = _compute_control_D(
            b, bins, acs, acs1, dx, half1, half2, ntr, nrand, split,
            null_batch_size=null_batch_size,
            choices=choices_ord if 'stim_block' in split else None,
        )
        if excl_info is not None:
            D = dict(D)
            D['trial_exclusion'] = excl_info
            D['null_scheme'] = 'label_shuffle_excl_sticky'
        return D

    # average trials per condition (no null)
    print('all trials')
    w0 = [bi.mean(axis=0) for bi in bins]
    s0 = [bi.var(axis=0) for bi in bins]

    ws = np.array(w0)
    ss = np.array(s0)

    # strictly standardized mean difference (single-cell, true split only)
    d_var = (((ws[0] - ws[1]) / ((ss[0] + ss[1]) ** 0.5)) ** 2)
    d_euc = (ws[0] - ws[1]) ** 2

    return {
        'acs': acs,
        'acs1': acs1,
        'd_vars': d_var,
        'd_eucs': d_euc,
        'ws': ws[:ntravis],
    }


def _pool_scale():
    return 1.0 / b_size if b_size else 1.0


class SplitPoolAccumulator:
    '''
    Streaming pool across insertions for one split (replaces per-insertion
    manifold/{split}/*.npy + separate d_var_stacked pass).

    shard: optional int. When set, checkpoints go to {split}.shard{k}.npy so
    multiple Slurm jobs can process disjoint insertion subsets in parallel,
    then merge_shards() + finalize().
    '''

    def __init__(self, split, min_reg=min_reg, shard=None):
        self.split = split
        self.min_reg = min_reg
        self.shard = shard
        self.pooled_keys = set()
        self.key_order = []
        self.acs = []
        self.acs1 = []
        self.ws = []
        self.regdv0 = {}
        self.regde0 = {}
        self.uperms = {}

    def _path(self):
        return _stream_acc_path(self.split, shard=self.shard)

    @classmethod
    def load(cls, split, min_reg=min_reg, shard=None):
        path = _stream_acc_path(split, shard=shard)
        if not path.exists():
            return cls(split, min_reg=min_reg, shard=shard)
        try:
            state = np.load(path, allow_pickle=True).item()
        except Exception as exc:
            # Truncated/corrupt checkpoint (e.g. job killed mid-np.save before
            # atomic-write fix). Quarantine and start fresh rather than crash.
            bad = path.with_suffix(path.suffix + f'.corrupt.{os.getpid()}')
            try:
                path.rename(bad)
                print(f'WARNING: corrupt stream_acc for {split}: {exc}')
                print(f'  quarantined -> {bad}; restarting accumulator from empty')
            except OSError:
                print(f'WARNING: corrupt stream_acc for {split}: {exc}; starting empty')
            return cls(split, min_reg=min_reg, shard=shard)
        acc = cls(split, min_reg=min_reg, shard=shard)
        acc.pooled_keys = set(state['pooled_keys'])
        acc.key_order = list(state.get('key_order') or [])
        if len(acc.key_order) != len(state['acs']):
            if state.get('uperms') and len(state['uperms']) == len(state['acs']):
                acc.key_order = list(state['uperms'].keys())
            else:
                acc.key_order = [f'ins_{i}' for i in range(len(state['acs']))]
        acc.acs = state['acs']
        acc.acs1 = state['acs1']
        acc.ws = state['ws']
        acc.regdv0 = state['regdv0']
        acc.regde0 = state['regde0']
        acc.uperms = state['uperms']
        return acc

    def add(self, eid_probe, D_):
        if eid_probe in self.pooled_keys:
            return False
        if not hasattr(self, 'key_order') or self.key_order is None:
            self.key_order = []
        if 'uperms' in D_:
            self.uperms[eid_probe] = D_['uperms']
        self.acs.append(D_['acs'])
        self.acs1.append(D_['acs1'])
        self.ws.append(D_['ws'])
        self.key_order.append(eid_probe)
        scale = _pool_scale()
        if 'D' not in D_:
            raise ValueError('stream_pool requires control=True (regional null curves in D_)')
        for reg in D_['D']:
            self.regdv0.setdefault(reg, []).append(
                np.array(D_['D'][reg]['d_vars']) * scale)
            self.regde0.setdefault(reg, []).append(
                np.array(D_['D'][reg]['d_eucs']) * scale)
        self.pooled_keys.add(eid_probe)
        return True

    def save(self):
        '''Atomic checkpoint: write to a temp *.npy then os.replace so a kill
        mid-write cannot leave a truncated file at the real path.'''
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Must end in .npy so np.save does not append another .npy suffix.
        tmp = path.parent / f'.{path.stem}.tmp.{os.getpid()}.npy'
        key_order = getattr(self, 'key_order', None)
        if key_order is None or len(key_order) != len(self.acs):
            if self.uperms and len(self.uperms) == len(self.acs):
                key_order = list(self.uperms.keys())
            else:
                key_order = [f'ins_{i}' for i in range(len(self.acs))]
            self.key_order = key_order
        payload = {
            'pooled_keys': list(self.pooled_keys),
            'key_order': list(self.key_order),
            'acs': self.acs,
            'acs1': self.acs1,
            'ws': self.ws,
            'regdv0': self.regdv0,
            'regde0': self.regde0,
            'uperms': self.uperms,
        }
        try:
            np.save(tmp, payload, allow_pickle=True)
            os.replace(tmp, path)
        except Exception:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise

    def finalize(self, save=True, cleanup_checkpoint=True):
        '''Write manifold/res/{split}*.npy (same layout as d_var_stacked).

        After a successful write, remove stream_acc checkpoints for this split
        (unsharded + shards).
        '''
        out = _finalize_pooled_split(
            self.split, self.acs, self.acs1, self.ws,
            self.regdv0, self.regde0, self.min_reg, save=save,
        )
        if save and cleanup_checkpoint:
            final = Path(pth_res, f'{output_split_name(self.split)}.npy')
            if final.exists():
                for path in [_stream_acc_path(self.split)] + _stream_acc_shard_paths(self.split):
                    if path.exists():
                        path.unlink()
                        print(f'removed stream_acc checkpoint {path}')
        return out


def merge_stream_acc_shards(split, min_reg=min_reg, include_unsharded=True):
    '''
    Merge {split}.shard*.npy (and optional {split}.npy) into one accumulator.
    Shards must be insertion-disjoint. Returns the merged SplitPoolAccumulator.
    '''
    paths = []
    if include_unsharded:
        p0 = _stream_acc_path(split)
        if p0.exists():
            paths.append(p0)
    paths.extend(_stream_acc_shard_paths(split))
    if not paths:
        return SplitPoolAccumulator(split, min_reg=min_reg)

    merged = SplitPoolAccumulator(split, min_reg=min_reg)
    merged.key_order = []
    for path in paths:
        try:
            state = np.load(path, allow_pickle=True).item()
        except Exception as exc:
            raise RuntimeError(f'Cannot load shard {path}: {exc}') from exc
        key_order = state.get('key_order')
        if not key_order or len(key_order) != len(state['acs']):
            if state.get('uperms') and len(state['uperms']) == len(state['acs']):
                key_order = list(state['uperms'].keys())
            else:
                key_order = [f'{path.stem}_{i}' for i in range(len(state['acs']))]
        n_before = len(merged.acs)
        skipped = 0
        for i, key in enumerate(key_order):
            if key in merged.pooled_keys:
                skipped += 1
                continue
            merged.acs.append(state['acs'][i])
            merged.acs1.append(state['acs1'][i])
            merged.ws.append(state['ws'][i])
            merged.pooled_keys.add(key)
            merged.key_order.append(key)
            if key in state.get('uperms', {}):
                merged.uperms[key] = state['uperms'][key]
        n_added = len(merged.acs) - n_before
        if skipped:
            raise RuntimeError(
                f'Overlap merging {path}: {skipped} duplicate keys. '
                f'Shards must be insertion-disjoint.'
            )
        if n_added != len(state['acs']):
            raise RuntimeError(f'Unexpected merge size for {path}')
        for reg, arrs in state.get('regdv0', {}).items():
            merged.regdv0.setdefault(reg, []).extend(arrs)
        for reg, arrs in state.get('regde0', {}).items():
            merged.regde0.setdefault(reg, []).extend(arrs)
        print(f'merged {path.name}: +{n_added} insertions '
              f'(total {len(merged.pooled_keys)})')
    return merged


def finalize_stream_shards(split, min_reg=min_reg, cleanup=True):
    '''Merge all shards for ``split`` and write manifold/res/{split}*.npy.'''
    acc = merge_stream_acc_shards(split, min_reg=min_reg)
    print(f'finalize {split}: {len(acc.pooled_keys)} insertions pooled')
    return acc.finalize(save=True, cleanup_checkpoint=cleanup)


def _accumulate_from_D(D_, regdv0, regde0, acs, acs1, ws, uperms, key):
    '''Add one insertion result into pool lists (used by d_var_stacked).'''
    if 'uperms' in D_:
        uperms[key] = D_['uperms']
    acs.append(D_['acs'])
    acs1.append(D_['acs1'])
    ws.append(D_['ws'])
    scale = _pool_scale()
    for reg in D_.get('D', {}):
        regdv0.setdefault(reg, []).append(np.array(D_['D'][reg]['d_vars']) * scale)
        regde0.setdefault(reg, []).append(np.array(D_['D'][reg]['d_eucs']) * scale)


def _euc_curve_summary(curves, split):
    '''Summary statistics for one true curve followed by its null curves.'''
    curves = np.asarray(curves)
    amps = np.ptp(curves, axis=1)
    d_euc = curves[0] - np.mean(curves[1:], axis=0)
    d_euc = d_euc - np.min(d_euc)
    amp_euc = float(np.max(d_euc))
    loc = np.where(d_euc > 0.7 * amp_euc)[0]
    lat_euc = (
        float(np.linspace(
            -pre_post[split][0], pre_post[split][1], len(d_euc))[loc[0]])
        if len(loc) else np.nan
    )
    return {
        'p_euc': float(np.mean(amps >= amps[0])),
        'd_euc': d_euc,
        'amp_euc': amp_euc,
        'lat_euc': lat_euc,
    }


def _finalize_pooled_split(split, acs, acs1, ws, regdv0, regde0, min_reg=min_reg,
                         save=True):
    if not acs:
        return {}, {}
    acs_cat = np.concatenate(acs)
    acs1_cat = np.concatenate(acs1)
    ws_cat = np.concatenate(ws, axis=1)
    regs0 = Counter(acs1_cat)
    regs = {reg: regs0[reg] for reg in regs0 if regs0[reg] > min_reg}

    regdv = {reg: (np.nansum(regdv0[reg], axis=0) / regs[reg]) ** 0.5 for reg in regs}
    regde = {reg: (np.nansum(regde0[reg], axis=0) / regs[reg]) ** 0.5 for reg in regs}

    r = {}
    for reg in regs:
        res = _euc_curve_summary(regde[reg], split)
        dat = ws_cat[:, acs_cat == reg, :]
        res['nclus'] = regs[reg]
        res['ws'] = dat[:2]
        r[reg] = res

    # Pool raw squared distances over every valid neuron before normalization.
    # This is a literal all-region population result, not an average of regional
    # RMS curves. It also includes regions below the per-region min_reg cutoff.
    all_nclus = len(acs1_cat)
    all_regde = None
    all_result = {}
    if all_nclus and regde0:
        all_raw = np.nansum(
            [np.nansum(arrays, axis=0) for arrays in regde0.values()],
            axis=0,
        )
        all_regde = np.sqrt(all_raw / all_nclus)
        all_result = _euc_curve_summary(all_regde, split)
        all_result['nclus'] = all_nclus
        all_result['n_regions'] = len(regde0)

    if save:
        out_base = output_split_name(split)
        np.save(Path(pth_res, f'{out_base}.npy'), r, allow_pickle=True)
        np.save(Path(pth_res, f'{out_base}_regde.npy'), regde, allow_pickle=True)
        np.save(Path(pth_res, f'{out_base}_all.npy'), all_result, allow_pickle=True)
        if all_regde is not None:
            np.save(
                Path(pth_res, f'{out_base}_all_regde.npy'),
                all_regde,
                allow_pickle=True,
            )
    return r, regde


def identify_good_session(eid):
    #sess_loader = SessionLoader(one, eid)
    #sess_loader.load_trials()
    #trials = sess_loader.trials
    trials, mask = load_trials_and_mask(one, eid)
    
    #remove sessions with less than 400 trials
    if len(trials) < 400:
        return 0
    
    else:        
        #identify overall performance
        performance = len(trials[trials['feedbackType']==1.0])/len(trials)
        #identify performance in zero contrast trials
        zeroleft_trials = trials[trials['contrastLeft']==0]
        zeroright_trials = trials[trials['contrastRight']==0]
        zero_trials = pd.concat([zeroleft_trials, zeroright_trials])
        #remove trials that had no bias at the beginning of a session
        zero_trials = zero_trials[zero_trials['probabilityLeft']!=0.5]
        nocontrast_performance = len(zero_trials[zero_trials['feedbackType']==1.0])/len(zero_trials)
        if performance > 0.8 and nocontrast_performance > 0.6:
            return 1
        else:
            return 0

'''    
###
### bulk processing 
###    
''' 

def get_all_d_vars(split, eids_plus = None, control = True,
                   mapping='Beryl', bycontrast=True, restart=True):

    '''
    for all BWM insertions, get the PSTHs and acronyms
    '''
    
    time00 = time.perf_counter()
    
    print('split', split, 'control', control)
    
    if eids_plus is None:
        df = bwm_query(one)
        eids_plus = df[['eid','probe_name', 'pid']].values

    def pid__(eid,probe_name):
        return df[np.bitwise_and(df['eid'] == eid, 
                          df['probe_name'] == probe_name
                          )]['pid'].values[0]

    # save results per insertion (eid_probe) in FlatIron folder
    pth = Path(one.cache_dir, 'manifold', split) 
    pth.mkdir(parents=True, exist_ok=True)

    if restart: # only go through all the insertions not already stored in pth
        ss = [f for f in os.listdir(pth) if f.endswith(".npy")]  # get insertions
        current_pids = [pid__(s.split('_')[0], s.split('_')[1].split('.')[0]) for s in ss]
        eids_plus = np.array([entry for entry in eids_plus if entry[2] not in current_pids])
 
    Fs = []   
    k=0
    print(f'Processing {len(eids_plus)} insertions')
    for i in eids_plus:
        eid, probe, pid = i
        k+=1
          
        time0 = time.perf_counter()
        try:
            D_ = get_d_vars(split, pid, control=control, mapping=mapping, bycontrast=bycontrast)
            eid_probe = eid+'_'+probe
            
            np.save(Path(pth,f'{eid_probe}.npy'), D_, allow_pickle=True)
                                         
            gc.collect() 
            print(k, 'of', len(eids_plus), 'ok') 
        except:
            Fs.append(pid)
            gc.collect()
            print(k, 'of', len(eids_plus), 'fail', pid)
            
        time1 = time.perf_counter()
        print(time1 - time0, 'sec')
                                
               
    time11 = time.perf_counter()
    print((time11 - time00)/60, f'min for the complete bwm set, {split}')
    print(f'{len(Fs)}, load failures:')
    print(Fs)

    
def get_all_d_vars_goodsession(split, eids_plus = None, control = True,
                   mapping='Beryl', bycontrast=True):

    '''
    for all BWM insertions, get the PSTHs and acronyms
    '''
    
    time00 = time.perf_counter()
    
    print('split', split, 'control', control)
    
    if eids_plus is None:
        df = bwm_query(one)
        eids_plus = df[['eid','probe_name', 'pid']].values

    # save results per insertion (eid_probe) in FlatIron folder
    pth = Path(one.cache_dir, 'manifold', split) 
    pth.mkdir(parents=True, exist_ok=True)
 
    Fs = []   
    k=0
    print(f'Processing {len(eids_plus)} insertions')
    for i in eids_plus:
        eid, probe, pid = i
        k+=1
          
        time0 = time.perf_counter()
        try:
            goodsession = identify_good_session(eid)
            if goodsession==0:
                print(k, 'of', len(eids_plus), 'notgoodperformance') 
                continue
            D_ = get_d_vars(split, pid, control=control, mapping=mapping, bycontrast=bycontrast)
            eid_probe = eid+'_'+probe
            
            np.save(Path(pth,f'{eid_probe}.npy'), D_, allow_pickle=True)
                                         
            gc.collect() 
            print(k, 'of', len(eids_plus), 'ok') 
        except:
            Fs.append(pid)
            gc.collect()
            print(k, 'of', len(eids_plus), 'fail', pid)
            
        time1 = time.perf_counter()
        print(time1 - time0, 'sec')
                                
               
    time11 = time.perf_counter()
    print((time11 - time00)/60, f'min for the complete bwm set, {split}')
    print(f'{len(Fs)}, load failures:')
    print(Fs)

    
def get_all_d_vars_badsession(split, eids_plus = None, control = True,
                   mapping='Beryl', bycontrast=True):

    '''
    for all BWM insertions, get the PSTHs and acronyms
    '''
    
    time00 = time.perf_counter()
    
    print('split', split, 'control', control)
    
    if eids_plus is None:
        df = bwm_query(one)
        eids_plus = df[['eid','probe_name', 'pid']].values

    # save results per insertion (eid_probe) in FlatIron folder
    pth = Path(one.cache_dir, 'manifold', split) 
    pth.mkdir(parents=True, exist_ok=True)
 
    Fs = []   
    k=0
    print(f'Processing {len(eids_plus)} insertions')
    for i in eids_plus:
        eid, probe, pid = i
        k+=1
          
        time0 = time.perf_counter()
        try:
            goodsession = identify_good_session(eid)
            if goodsession==1:
                print(k, 'of', len(eids_plus), 'skip,goodperformance') 
                continue
            D_ = get_d_vars(split, pid, control=control, mapping=mapping, bycontrast=bycontrast)
            eid_probe = eid+'_'+probe
            
            np.save(Path(pth,f'{eid_probe}.npy'), D_, allow_pickle=True)
                                         
            gc.collect() 
            print(k, 'of', len(eids_plus), 'ok') 
        except:
            Fs.append(pid)
            gc.collect()
            print(k, 'of', len(eids_plus), 'fail', pid)
            
        time1 = time.perf_counter()
        print(time1 - time0, 'sec')
                                
               
    time11 = time.perf_counter()
    print((time11 - time00)/60, f'min for the complete bwm set, {split}')
    print(f'{len(Fs)}, load failures:')
    print(Fs)

    
def get_all_d_vars_allsplits(splits_list, eids_plus=None, control=True,
                             mapping='Beryl', bycontrast=False, restart=True,
                             use_cache=True, save_cache=True,
                             stream_pool=False, save_per_insertion=None,
                             null_batch_size=NULL_BATCH_SIZE, nrand=nrand,
                             shard_idx=None, n_shards=None, finalize=True,
                             session_shuffle_null=False,
                             actkernel_choice_null=False,
                             exclude_sticky_trials=False,
                             sticky_late_frac=STICKY_LATE_FRAC,
                             sticky_min_run=STICKY_MIN_RUN):
    '''
    Time-efficient driver: iterate insertions in the OUTER loop, load each
    insertion's raw data ONCE (build_insertion_cache), then compute ALL splits
    for that insertion from the cached data. Replaces calling get_all_d_vars per
    split (which reloaded spikes/trials for every split).

    When ``stream_pool=True``, results are accumulated into
    ``manifold/res/_stream_acc/{split}.npy`` and finalized to
    ``manifold/res/{split}*.npy`` without writing per-insertion
    ``manifold/{split}/{eid_probe}.npy`` (major disk savings).

    Sharding (parallel Slurm jobs for one split):
      shard_idx, n_shards — process eids_plus[shard_idx::n_shards], write
      ``{split}.shard{k}.npy``. Set finalize=False on shard workers; run
      finalize_stream_shards(split) after all shards complete.

  ``save_per_insertion``: write manifold/{split}/{eid_probe}.npy (default False
    when stream_pool else True). ``restart``: skip (split, insertion) already done.

    ``actkernel_choice_null``: if True, choice_stim* / choice_duringstim* use
    BWM-style ActionKernel synthetic-session nulls
    (``scripts/simulate_synthetic_choices.synthetic_sessions_from_trials``).
    Takes precedence over session_shuffle_null.

    ``session_shuffle_null``: if True, choice_stim* / choice_duringstim* use
    literal Harris session-permutation nulls (loads donor bank; indexes donor
    choice sequences at recipient ``elig_idx``). Default False.

    ``exclude_sticky_trials``: drop late-session + perseveration trials before
    distance/null (see ``apply_sticky_trial_exclusion``). Prefer directing
    outputs to ``manifold/res_excl_sticky`` via ``configure_excl_sticky_output_dirs``.
    '''
    if exclude_sticky_trials:
        configure_excl_sticky_output_dirs()
    configure_null_file_suffix(
        actkernel_choice_null=actkernel_choice_null,
        session_shuffle_null=session_shuffle_null and not actkernel_choice_null,
    )
    if save_per_insertion is None:
        save_per_insertion = not stream_pool
    if (shard_idx is None) ^ (n_shards is None):
        raise ValueError('Provide both shard_idx and n_shards, or neither')
    if n_shards is not None:
        n_shards = int(n_shards)
        shard_idx = int(shard_idx)
        if not (0 <= shard_idx < n_shards):
            raise ValueError(f'shard_idx must be in [0, {n_shards})')
        if not stream_pool:
            raise ValueError('Sharding requires stream_pool=True')
        if finalize and n_shards > 1:
            print('NOTE: shard worker with finalize=True will only finalize '
                  'this shard\'s insertions; prefer finalize=False + '
                  'finalize_stream_shards after all shards finish')

    time00 = time.perf_counter()
    print('splits', splits_list, 'control', control, 'use_cache', use_cache,
          'stream_pool', stream_pool, 'null_batch_size', null_batch_size,
          'shard', shard_idx, '/', n_shards, 'finalize', finalize,
          'session_shuffle_null', session_shuffle_null,
          'actkernel_choice_null', actkernel_choice_null,
          'exclude_sticky_trials', exclude_sticky_trials)

    donor_bank = None
    use_donor = (session_shuffle_null
                 and not actkernel_choice_null
                 and any(is_choice_lr_split(sp) for sp in splits_list))
    if use_donor:
        donor_bank = load_choice_donor_bank()
        print(f'harris donor bank: {len(donor_bank)} eids')
    elif session_shuffle_null and not actkernel_choice_null:
        print('WARNING: --session-shuffle-null set but no choice_stim*/'
              'choice_duringstim* splits in list')
    elif actkernel_choice_null and any(
            is_choice_lr_split(sp) for sp in splits_list):
        print('actkernel BWM synthetic-session null enabled for choice L–R splits')
    elif actkernel_choice_null:
        print('WARNING: --actkernel-choice-null set but no choice_stim*/'
              'choice_duringstim* splits in list')

    if eids_plus is None:
        df = bwm_query(one)
        eids_plus = df[['eid', 'probe_name', 'pid']].values
    eids_plus = np.asarray(eids_plus)
    if n_shards is not None:
        eids_plus = eids_plus[shard_idx::n_shards]
        print(f'Shard {shard_idx}/{n_shards}: {len(eids_plus)} insertions')

    accumulators = {}
    if stream_pool:
        for split in splits_list:
            accumulators[split] = SplitPoolAccumulator.load(
                split, shard=shard_idx)
    elif save_per_insertion:
        for split in splits_list:
            Path(one.cache_dir, 'manifold', split).mkdir(parents=True, exist_ok=True)

    Fs = []
    k = 0
    print(f'Processing {len(eids_plus)} insertions x {len(splits_list)} splits')
    for eid, probe, pid in eids_plus:
        k += 1
        eid_probe = f'{eid}_{probe}'
        time0 = time.perf_counter()

        pending = []
        for split in splits_list:
            if stream_pool:
                if restart and eid_probe in accumulators[split].pooled_keys:
                    continue
            elif restart and save_per_insertion:
                outp = Path(one.cache_dir, 'manifold', split, f'{eid_probe}.npy')
                if outp.exists():
                    continue
            pending.append(split)
        if not pending:
            print(k, 'of', len(eids_plus), 'all splits done, skip')
            continue

        try:
            cache = build_insertion_cache(pid, save=save_cache, restart=restart) if use_cache else None
        except Exception as exc:
            Fs.append(pid)
            gc.collect()
            print(k, 'of', len(eids_plus), 'fail (load)', pid, exc)
            continue

        n_ok = 0
        n_skip = 0
        for split in pending:
            try:
                D_ = get_d_vars(split, pid, control=control, mapping=mapping,
                                bycontrast=bycontrast, cached=cache,
                                null_batch_size=null_batch_size, nrand=nrand,
                                donor_bank=donor_bank,
                                session_shuffle_null=session_shuffle_null,
                                actkernel_choice_null=actkernel_choice_null,
                                exclude_sticky_trials=exclude_sticky_trials,
                                sticky_late_frac=sticky_late_frac,
                                sticky_min_run=sticky_min_run)
                if stream_pool:
                    accumulators[split].add(eid_probe, D_)
                    accumulators[split].save()
                if save_per_insertion:
                    outp = Path(one.cache_dir, 'manifold', split, f'{eid_probe}.npy')
                    np.save(outp, D_, allow_pickle=True)
                n_ok += 1
            except InsufficientTrials as exc:
                n_skip += 1
                print('   split skip', split, pid, exc)
            except Exception as exc:
                print('   split fail', split, pid, exc)
        del cache
        gc.collect()
        time1 = time.perf_counter()
        print(k, 'of', len(eids_plus),
              f'ok {n_ok}/{len(pending)} splits',
              f'skip {n_skip}',
              round(time1 - time0, 1), 'sec')

    if stream_pool and finalize:
        for split in splits_list:
            if n_shards is not None and n_shards > 1:
                print('shard done — skipping finalize (run finalize_stream_shards later)')
                print(f'  {split} shard{shard_idx}: {len(accumulators[split].pooled_keys)} insertions')
            else:
                print('finalize stream pool', split, f'({len(accumulators[split].pooled_keys)} insertions)')
                accumulators[split].finalize()

    time11 = time.perf_counter()
    print((time11 - time00) / 60, f'min for {len(eids_plus)} insertions x {len(splits_list)} splits')
    print(f'{len(Fs)} load failures:')
    print(Fs)


def cache_all_insertions(eids_plus=None, restart=True):
    '''Build (and persist) the per-insertion raw-data cache for all BWM insertions.'''
    if eids_plus is None:
        df = bwm_query(one)
        eids_plus = df[['eid', 'probe_name', 'pid']].values
    Fs = []
    for k, (eid, probe, pid) in enumerate(eids_plus, 1):
        t0 = time.perf_counter()
        try:
            build_insertion_cache(pid, restart=restart)
            print(k, 'of', len(eids_plus), 'cached', round(time.perf_counter() - t0, 1), 'sec')
        except Exception as exc:
            Fs.append(pid)
            print(k, 'of', len(eids_plus), 'fail', pid, exc)
        gc.collect()
    print(f'{len(Fs)} cache failures:', Fs)


def get_crf_slope(pid, cached=None, mapping='Beryl', window=(0.0, 0.15),
                  contrasts=CONTRASTS, nrand=1000, min_reg=min_reg):
    '''
    Goal 3: contrast-response function (CRF) per region, split by block prior,
    and a test of whether the prior modulates the CRF slope (gain).

    For each stimulus side (L/R) and |contrast| in ``contrasts`` we compute the
    mean post-stim population response per neuron over ``window`` (single time
    bin from stimOn), then average within region. The CRF is response vs contrast.

    "concordant" prior = block favors the stimulus side (high prior for that side),
    "discordant" = block favors the opposite side. At 0% contrast behavior is fully
    prior-driven, so it anchors the low end of the CRF.

    Prior modulation of gain = slope(concordant) - slope(discordant), averaged over
    sides. Significance via a null that shuffles block (concordant/discordant)
    labels *within* each (side, contrast) cell, preserving side/contrast structure.

    Returns per region: {'nclus', 'contrasts', 'crf_conc', 'crf_disc',
    'slope_conc', 'slope_disc', 'slope_mod', 'p_slope_mod'}.
    '''
    satur = 'saturation_stim_plus04'
    eid = None
    if cached is not None:
        spikes = cached['spikes']
        clusters = cached['clusters']
        trials = cached['trials'][satur].copy()
        eid = cached.get('eid')
    if eid is None:
        eid, probe = one.pid2eid(pid)
    else:
        probe = cached.get('probe')
    if cached is None:
        spikes, clusters = load_good_units(one, pid)
        trials, mask = load_trials_for_saturation(one, eid, satur)
        trials = trials[mask]
    trials = trials[trials['probabilityLeft'] != 0.5]  # block-biased trials only

    acs = np.array(br.id2acronym(clusters['atlas_id'], mapping=mapping))
    good = ~np.bitwise_or.reduce([acs == r for r in ['void', 'root']])
    acs = acs[good]

    contrasts = sorted(set(float(c) for c in contrasts))
    pre_t = 0.0
    post_t = float(window[1] - window[0])

    # Per (side, contrast): single-bin response (ntr, n_good_neurons) + conc mask.
    cond = {}
    stim_on = trials['stimOn_times'].values
    pl = trials['probabilityLeft'].values
    for side, cside in (('L', 'contrastLeft'), ('R', 'contrastRight')):
        conc_pleft = 0.8 if side == 'L' else 0.2
        cvals = trials[cside].values
        for c in contrasts:
            sel = np.isclose(cvals, c)
            if sel.sum() == 0:
                continue
            ev = stim_on[sel]
            bi, _ = bin_spikes2D(
                spikes['times'],
                clusters['cluster_id'][spikes['clusters']],
                clusters['cluster_id'],
                np.array(ev), pre_t, post_t, post_t)
            R = bi[:, :, 0][:, good]  # (ntr, n_good)
            cond[(side, c)] = {'R': R, 'conc': pl[sel] == conc_pleft}

    def _slope_mod(reg_mask, perm=None):
        side_mods = []
        for side in ('L', 'R'):
            xs, rc, rd = [], [], []
            for c in contrasts:
                key = (side, c)
                if key not in cond:
                    continue
                R = cond[key]['R'][:, reg_mask]
                conc = cond[key]['conc'] if perm is None else perm[key]
                if conc.sum() == 0 or (~conc).sum() == 0:
                    continue
                xs.append(c)
                rc.append(float(R[conc].mean()))
                rd.append(float(R[~conc].mean()))
            if len(xs) >= 2:
                xs = np.array(xs)
                side_mods.append(np.polyfit(xs, rc, 1)[0] - np.polyfit(xs, rd, 1)[0])
        return float(np.mean(side_mods)) if side_mods else np.nan

    def _crf(reg_mask, which):
        out = []
        for c in contrasts:
            vals = []
            for side in ('L', 'R'):
                key = (side, c)
                if key not in cond:
                    continue
                R = cond[key]['R'][:, reg_mask]
                m = cond[key]['conc'] if which == 'conc' else ~cond[key]['conc']
                if m.sum():
                    vals.append(float(R[m].mean()))
            out.append(np.mean(vals) if vals else np.nan)
        return out

    regs = Counter(acs)
    D = {}
    for reg, n in regs.items():
        if n < min_reg:
            continue
        rm = (acs == reg)
        true_mod = _slope_mod(rm)
        if np.isnan(true_mod):
            continue
        null = np.array([
            _slope_mod(rm, perm={k: np.random.permutation(v['conc']) for k, v in cond.items()})
            for _ in range(nrand)
        ])
        null = null[~np.isnan(null)]
        p = float(np.mean(np.abs(null) >= abs(true_mod))) if null.size else np.nan
        crf_c = _crf(rm, 'conc')
        crf_d = _crf(rm, 'disc')
        D[reg] = {
            'nclus': int(n),
            'contrasts': list(contrasts),
            'crf_conc': crf_c,
            'crf_disc': crf_d,
            'slope_conc': (np.polyfit(contrasts, crf_c, 1)[0]
                           if not np.any(np.isnan(crf_c)) else np.nan),
            'slope_disc': (np.polyfit(contrasts, crf_d, 1)[0]
                           if not np.any(np.isnan(crf_d)) else np.nan),
            'slope_mod': true_mod,
            'p_slope_mod': p,
        }
    return {'pid': pid, 'eid': eid, 'D': D, 'contrasts': list(contrasts)}


def get_all_crf_slope(eids_plus=None, control=True, mapping='Beryl',
                      nrand=1000, restart=True, use_cache=True):
    '''Driver: per-insertion CRF slope + prior-modulation test; save per insertion.'''
    if eids_plus is None:
        df = bwm_query(one)
        eids_plus = df[['eid', 'probe_name', 'pid']].values
    pth = Path(one.cache_dir, 'manifold', 'crf_slope')
    pth.mkdir(parents=True, exist_ok=True)
    Fs = []
    for k, (eid, probe, pid) in enumerate(eids_plus, 1):
        eid_probe = f'{eid}_{probe}'
        outp = Path(pth, f'{eid_probe}.npy')
        if restart and outp.exists():
            continue
        t0 = time.perf_counter()
        try:
            cache = build_insertion_cache(pid, restart=restart) if use_cache else None
            D_ = get_crf_slope(pid, cached=cache, mapping=mapping, nrand=nrand)
            np.save(outp, D_, allow_pickle=True)
            del cache
            gc.collect()
            print(k, 'of', len(eids_plus), 'ok', round(time.perf_counter() - t0, 1), 'sec')
        except Exception as exc:
            Fs.append(pid)
            gc.collect()
            print(k, 'of', len(eids_plus), 'fail', pid, exc)
    print(f'{len(Fs)} failures:', Fs)


def crf_slope_stacked(min_reg=min_reg, alpha_sig=0.05):
    '''
    Pool CRF-slope prior modulation across insertions per region.

    Aggregates slope_mod (concordant-discordant CRF slope) by nanmean across
    insertions, averages the per-insertion p-values, and reports the mean CRF
    curves. Writes manifold/res/crf_slope_stacked.npy.
    '''
    pth = Path(one.cache_dir, 'manifold', 'crf_slope')
    files = [f for f in os.listdir(pth) if f.endswith('.npy')]
    agg = {}
    for f in files:
        D_ = np.load(Path(pth, f), allow_pickle=True).item()
        contrasts = D_.get('contrasts')
        for reg, r in D_['D'].items():
            a = agg.setdefault(reg, {'slope_mod': [], 'p': [], 'nclus': [],
                                     'crf_conc': [], 'crf_disc': [], 'contrasts': contrasts})
            a['slope_mod'].append(r['slope_mod'])
            a['p'].append(r['p_slope_mod'])
            a['nclus'].append(r['nclus'])
            a['crf_conc'].append(r['crf_conc'])
            a['crf_disc'].append(r['crf_disc'])
    res = {}
    for reg, a in agg.items():
        if np.nansum(a['nclus']) < min_reg or len(a['slope_mod']) == 0:
            continue
        res[reg] = {
            'n_insertions': len(a['slope_mod']),
            'nclus_total': int(np.nansum(a['nclus'])),
            'slope_mod_mean': float(np.nanmean(a['slope_mod'])),
            'p_slope_mod_mean': float(np.nanmean(a['p'])),
            'frac_sig': float(np.nanmean(np.array(a['p']) < alpha_sig)),
            'crf_conc_mean': np.nanmean(np.array(a['crf_conc'], dtype=float), axis=0).tolist(),
            'crf_disc_mean': np.nanmean(np.array(a['crf_disc'], dtype=float), axis=0).tolist(),
            'contrasts': a['contrasts'],
        }
    outp = Path(one.cache_dir, 'manifold', 'res', 'crf_slope_stacked.npy')
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.save(outp, res, allow_pickle=True)
    print(f'crf_slope_stacked: {len(res)} regions -> {outp}')
    return res


def _ols_r2_multi(X, Y):
    '''Vectorized OLS R² for shared design ``X`` (n×p) and responses ``Y`` (n×m).'''
    n = X.shape[0]
    if n < X.shape[1] + 1:
        return np.full(Y.shape[1], np.nan)
    beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    ss_res = np.sum(resid * resid, axis=0)
    y_c = Y - Y.mean(axis=0, keepdims=True)
    ss_tot = np.sum(y_c * y_c, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        r2 = 1.0 - ss_res / ss_tot
    r2 = np.where(ss_tot > 1e-12, r2, np.nan)
    return r2


def _signed_contrast(trials):
    '''Signed stimulus contrast: +c for left, −c for right.'''
    cl = trials['contrastLeft'].to_numpy(dtype=float)
    cr = trials['contrastRight'].to_numpy(dtype=float)
    return np.nan_to_num(cl, nan=0.0) - np.nan_to_num(cr, nan=0.0)


def get_var_partition(pid, cached=None, mapping='Beryl',
                      window=(0.0, SHORT_DURINGSTIM_WINDOW_S),
                      regions=None, min_trials=30, prior_type='act',
                      act_alpha=None):
    '''
    Per-neuron OLS variance partition in the early duringstim window
    (default 0–80 ms post-stimOn, ``SHORT_DURINGSTIM_WINDOW_S``).

    Model: y ~ 1 + stim + choice + prior + stim×prior, where
      stim  = signed contrast (+left / −right),
      choice = ±1,
      prior  = action-kernel prior − 0.5 (default ``prior_type='act'``),
               via ``action_kernel_priors`` (same as act SC splits).

    Unique main-effect R² are Type-II relative to the additive model
    (stim+choice+prior); interaction ΔR² = R²(full) − R²(additive).

    ``regions``: optional set/list of Beryl acronyms to keep; if None, all
    non-void/root regions with ≥1 neuron are fit.

    Returns dict with per-neuron arrays under key ``neurons`` plus metadata.
    '''
    satur = 'saturation_stim_plus04'
    eid = None
    probe = None
    if cached is not None:
        spikes = cached['spikes']
        clusters = cached['clusters']
        trials = cached['trials'][satur].copy()
        eid = cached.get('eid')
        probe = cached.get('probe')
    if eid is None:
        eid, probe = one.pid2eid(pid)
    if cached is None:
        spikes, clusters = load_good_units(one, pid)
        trials, mask = load_trials_for_saturation(one, eid, satur)
        trials = trials[mask]

    # Keep full trial order (incl. true-block 0.5) so the action kernel matches
    # act SC splits; drop only invalid choices.
    choice = trials['choice'].to_numpy(dtype=float)
    valid = np.isin(choice, [-1.0, 1.0])
    trials = trials.loc[valid].reset_index(drop=True)
    choice = trials['choice'].to_numpy(dtype=float)

    if len(trials) < min_trials:
        raise InsufficientTrials(
            f'pid {pid}: only {len(trials)} trials (need ≥{min_trials})')

    prior_type = str(prior_type).lower()
    if prior_type == 'act':
        # Same transform as get_d_vars for '*_act' splits.
        a = float(alpha if act_alpha is None else act_alpha)
        trials = trials.copy()
        trials['true_priors'] = trials['probabilityLeft']
        act_cont, act_bin = action_kernel_priors(a, list(choice))
        trials['act_priors'] = act_cont
        trials['probabilityLeft'] = act_bin
        # Continuous EMA for encoding (graded action-kernel belief).
        prior = np.asarray(act_cont, dtype=float) - 0.5
    elif prior_type in ('block', 'true'):
        trials = trials[trials['probabilityLeft'] != 0.5].reset_index(drop=True)
        choice = trials['choice'].to_numpy(dtype=float)
        if len(trials) < min_trials:
            raise InsufficientTrials(
                f'pid {pid}: only {len(trials)} block-biased trials '
                f'(need ≥{min_trials})')
        prior = trials['probabilityLeft'].to_numpy(dtype=float) - 0.5
    else:
        raise ValueError(
            f"prior_type must be 'act' or 'block', got {prior_type!r}")

    stim = _signed_contrast(trials)
    choice = trials['choice'].to_numpy(dtype=float)
    inter = stim * prior
    ones = np.ones(len(trials), dtype=float)

    X_full = np.column_stack([ones, stim, choice, prior, inter])
    X_add = np.column_stack([ones, stim, choice, prior])
    X_no_stim = np.column_stack([ones, choice, prior])
    X_no_choice = np.column_stack([ones, stim, prior])
    X_no_prior = np.column_stack([ones, stim, choice])

    acs = np.array(br.id2acronym(clusters['atlas_id'], mapping=mapping))
    good = ~np.bitwise_or.reduce([acs == r for r in ['void', 'root']])
    acs = acs[good]
    cluster_ids_good = np.asarray(clusters['cluster_id'])[good]
    if regions is not None:
        region_set = set(regions)
        keep_reg = np.array([a in region_set for a in acs], dtype=bool)
    else:
        keep_reg = np.ones(len(acs), dtype=bool)
    if keep_reg.sum() == 0:
        return {
            'pid': pid, 'eid': eid, 'probe': probe,
            'n_trials': int(len(trials)), 'neurons': {},
            'skipped': 'no_target_regions',
            'prior_type': prior_type,
        }

    acs_fit = acs[keep_reg]
    cluster_ids = cluster_ids_good[keep_reg]

    pre_t = 0.0
    post_t = float(window[1] - window[0])
    stim_on = trials['stimOn_times'].to_numpy(dtype=float)
    # Bin all clusters then slice to targets (same pattern as get_crf_slope /
    # get_d_vars). Passing a subset of cluster_ids into bin_spikes2D causes
    # shape-mismatch failures on some insertions.
    bi, _ = bin_spikes2D(
        spikes['times'],
        clusters['cluster_id'][spikes['clusters']],
        clusters['cluster_id'],
        stim_on, pre_t, post_t, post_t)
    R = (bi[:, :, 0][:, good][:, keep_reg] / post_t).astype(float)

    r2_full = _ols_r2_multi(X_full, R)
    r2_add = _ols_r2_multi(X_add, R)
    r2_no_stim = _ols_r2_multi(X_no_stim, R)
    r2_no_choice = _ols_r2_multi(X_no_choice, R)
    r2_no_prior = _ols_r2_multi(X_no_prior, R)

    r2_unique_stim = r2_add - r2_no_stim
    r2_unique_choice = r2_add - r2_no_choice
    r2_unique_prior = r2_add - r2_no_prior
    r2_stim_x_prior = r2_full - r2_add

    beta_full, _, _, _ = np.linalg.lstsq(X_full, R, rcond=None)
    # beta rows: intercept, stim, choice, prior, stim×prior

    neurons = {
        'cluster_id': cluster_ids,
        'region': acs_fit,
        'r2_full': r2_full,
        'r2_additive': r2_add,
        'r2_unique_stim': r2_unique_stim,
        'r2_unique_choice': r2_unique_choice,
        'r2_unique_prior': r2_unique_prior,
        'r2_stim_x_prior': r2_stim_x_prior,
        'beta_stim': beta_full[1],
        'beta_choice': beta_full[2],
        'beta_prior': beta_full[3],
        'beta_stim_x_prior': beta_full[4],
    }
    return {
        'pid': pid,
        'eid': eid,
        'probe': probe,
        'n_trials': int(len(trials)),
        'window': list(window),
        'n_neurons': int(R.shape[1]),
        'prior_type': prior_type,
        'neurons': neurons,
    }


def get_all_var_partition(eids_plus=None, regions=None, mapping='Beryl',
                          window=(0.0, SHORT_DURINGSTIM_WINDOW_S),
                          restart=True, use_cache=True,
                          min_trials=30, prior_type='act'):
    '''Driver: per-insertion variance partition (default early 0–80 ms,
    action-kernel prior); save under manifold/var_partition.'''
    if eids_plus is None:
        df = bwm_query(one)
        eids_plus = df[['eid', 'probe_name', 'pid']].values
    pth = Path(one.cache_dir, 'manifold', 'var_partition')
    pth.mkdir(parents=True, exist_ok=True)
    Fs = []
    for k, (eid, probe, pid) in enumerate(eids_plus, 1):
        eid_probe = f'{eid}_{probe}'
        outp = Path(pth, f'{eid_probe}.npy')
        if restart and outp.exists():
            continue
        t0 = time.perf_counter()
        try:
            cache = None
            if use_cache:
                cpath = Path(one.cache_dir, 'manifold', 'insertion_cache',
                             f'{eid_probe}.npy')
                # Always reuse an existing insertion cache (avoids pid2eid offline).
                if cpath.exists():
                    cache = np.load(cpath, allow_pickle=True).item()
                else:
                    cache = build_insertion_cache(pid, restart=False)
            # Skip early if cache has no overlap with target regions.
            if cache is not None and regions is not None:
                acs = np.array(br.id2acronym(cache['clusters']['atlas_id'],
                                             mapping=mapping))
                if not any(a in set(regions) for a in acs):
                    print(k, 'of', len(eids_plus), 'skip (no target regions)',
                          eid_probe)
                    continue
            D_ = get_var_partition(
                pid, cached=cache, mapping=mapping, window=window,
                regions=regions, min_trials=min_trials, prior_type=prior_type)
            if D_.get('skipped') or not D_.get('neurons'):
                print(k, 'of', len(eids_plus), 'skip (empty)',
                      eid_probe, D_.get('skipped', ''))
                del cache
                gc.collect()
                continue
            np.save(outp, D_, allow_pickle=True)
            del cache
            gc.collect()
            print(k, 'of', len(eids_plus), 'ok',
                  D_.get('n_neurons', 0), 'neu',
                  round(time.perf_counter() - t0, 1), 'sec')
        except InsufficientTrials as exc:
            Fs.append(pid)
            gc.collect()
            print(k, 'of', len(eids_plus), 'skip', pid, exc)
        except Exception as exc:
            Fs.append(pid)
            gc.collect()
            print(k, 'of', len(eids_plus), 'fail', pid, exc)
    print(f'{len(Fs)} failures/skips:', Fs)
    return Fs


def var_partition_stacked(regtype_csv=None, regtypes=(0.0, 0.5, 1.0),
                          min_neurons=5, alpha_sig=None, mixed_only=False):
    '''
    Pool per-neuron variance-partition stats across insertions by region.

    If ``regtype_csv`` is given (from export_stimchoice_regtypes), join
    ``sc_duringstim_regtype`` / ``mixed_stim_choice`` and optionally restrict.

    Writes manifold/res/var_partition_stacked.npy and meta CSV under cache.
    '''
    pth = Path(one.cache_dir, 'manifold', 'var_partition')
    if not pth.exists():
        raise FileNotFoundError(pth)
    files = [f for f in os.listdir(pth) if f.endswith('.npy')]
    rows = []
    for f in files:
        D_ = np.load(Path(pth, f), allow_pickle=True).item()
        neu = D_.get('neurons') or {}
        if not neu or 'region' not in neu:
            continue
        regs = neu['region']
        for i in range(len(regs)):
            rows.append({
                'eid': D_.get('eid'),
                'probe': D_.get('probe'),
                'pid': D_.get('pid'),
                'cluster_id': int(neu['cluster_id'][i]),
                'region': regs[i],
                'r2_full': float(neu['r2_full'][i]),
                'r2_additive': float(neu['r2_additive'][i]),
                'r2_unique_stim': float(neu['r2_unique_stim'][i]),
                'r2_unique_choice': float(neu['r2_unique_choice'][i]),
                'r2_unique_prior': float(neu['r2_unique_prior'][i]),
                'r2_stim_x_prior': float(neu['r2_stim_x_prior'][i]),
                'beta_stim': float(neu['beta_stim'][i]),
                'beta_choice': float(neu['beta_choice'][i]),
                'beta_prior': float(neu['beta_prior'][i]),
                'beta_stim_x_prior': float(neu['beta_stim_x_prior'][i]),
                'n_trials': D_.get('n_trials'),
            })
    if not rows:
        print('var_partition_stacked: no neurons found')
        return {}
    df = pd.DataFrame(rows)

    regtype_col = None
    if regtype_csv is not None:
        rt = pd.read_csv(regtype_csv)
        keep_cols = ['region', 'sc_duringstim_regtype', 'sc_duringchoice_regtype']
        for c in ('mixed_stim_choice', 'has_stim', 'has_choice',
                  'stim_processor', 'stim_processor_loose',
                  'sigma_stim_s', 'sigma_stim_s_prime'):
            if c in rt.columns:
                keep_cols.append(c)
        rt = rt[keep_cols].drop_duplicates('region')
        df = df.merge(rt, on='region', how='left')
        regtype_col = 'sc_duringstim_regtype'
        if mixed_only and 'mixed_stim_choice' in df.columns:
            mixed = df['mixed_stim_choice']
            if mixed.dtype == object:
                mixed = mixed.astype(str).str.lower().isin(('true', '1', '1.0'))
            else:
                mixed = mixed.fillna(False).astype(bool)
            df = df[mixed]
        elif regtypes is not None:
            allowed = set(float(x) for x in regtypes)
            df = df[df[regtype_col].isin(allowed)]

    agg = {}
    for reg, g in df.groupby('region'):
        if len(g) < min_neurons:
            continue
        entry = {
            'n_neurons': int(len(g)),
            'n_insertions': int(g[['eid', 'probe']].drop_duplicates().shape[0]),
            'r2_full_mean': float(np.nanmean(g['r2_full'])),
            'r2_unique_stim_mean': float(np.nanmean(g['r2_unique_stim'])),
            'r2_unique_choice_mean': float(np.nanmean(g['r2_unique_choice'])),
            'r2_unique_prior_mean': float(np.nanmean(g['r2_unique_prior'])),
            'r2_stim_x_prior_mean': float(np.nanmean(g['r2_stim_x_prior'])),
            'r2_unique_stim_median': float(np.nanmedian(g['r2_unique_stim'])),
            'r2_unique_choice_median': float(np.nanmedian(g['r2_unique_choice'])),
            'r2_stim_x_prior_median': float(np.nanmedian(g['r2_stim_x_prior'])),
        }
        if regtype_col and regtype_col in g.columns:
            entry['sc_duringstim_regtype'] = float(g[regtype_col].iloc[0])
            if 'sc_duringchoice_regtype' in g.columns:
                entry['sc_duringchoice_regtype'] = float(
                    g['sc_duringchoice_regtype'].iloc[0])
        agg[reg] = entry

    outp = Path(one.cache_dir, 'manifold', 'res', 'var_partition_stacked.npy')
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.save(outp, agg, allow_pickle=True)

    meta = Path(one.cache_dir, 'meta')
    meta.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([
        {'region': reg, **vals} for reg, vals in agg.items()
    ]).sort_values('region')
    csv_path = meta / 'var_partition_by_region.csv'
    summary.to_csv(csv_path, index=False)
    print(f'var_partition_stacked: {len(agg)} regions -> {outp}\n  CSV {csv_path}')
    return agg


def d_var_stacked(split, min_reg = min_reg, uperms_ = False):
                  
    time0 = time.perf_counter()

    '''
    average d_var_m via nanmean across insertions,
    remove regions with too few neurons (min_reg)
    compute maxes, latencies and p-values
  '''
    
    print(split)
    pth = Path(one.cache_dir, 'manifold', split) 
    ss = os.listdir(pth)  # get insertions
    
    acs = []
    acs1 = []
    ws = [] 
    regdv0 = {}
    regde0 = {}
    uperms = {}
    
    for s in ss:
        D_ = np.load(Path(pth, s), allow_pickle=True).flat[0]
        key = s.split('.')[0]
        if 'uperms' in D_:
            uperms[key] = D_['uperms']
        if uperms_:
            continue
        _accumulate_from_D(D_, regdv0, regde0, acs, acs1, ws, uperms, key)
    
    if uperms_:
        return uperms

    _finalize_pooled_split(split, acs, acs1, ws, regdv0, regde0, min_reg=min_reg, save=True)
           
    time1 = time.perf_counter()    
    print('total time:', time1 - time0, 'sec')



def curves_params_all(split, control=True, goodsession=None, bycontrast=True):

    if goodsession==1:
        get_all_d_vars_goodsession(split, control=control, bycontrast=bycontrast)
    elif goodsession==0:
        get_all_d_vars_badsession(split, control=control, bycontrast=bycontrast)
    else:
        get_all_d_vars(split, control=control, bycontrast=bycontrast)
        
    d_var_stacked(split)        

    
def get_average_peth():

    res = {}
    
    for split in align:
        r = {}
    
        pth = Path(one.cache_dir, 'manifold', split) 
        ss = os.listdir(pth)  # get insertions
        
        ws = [] 
        
        # group results across insertions
        for s in ss:
        
            D_ = np.load(Path(pth,s), 
                        allow_pickle=True).flat[0]
                           
            ws.append(D_['ws'])        
            

        ws = np.concatenate(ws, axis=1)
        ntr, ncells, nt = ws.shape

        r['m0'] = np.mean(ws[0], axis=0)
        r['m1'] = np.mean(ws[1], axis=0)
        r['ms'] = np.mean(ws[2:], axis=(0,1))

        r['v0'] = np.std(ws[0], axis=0)/(ncells**0.5)
        r['v1'] = np.std(ws[1], axis=0)/(ncells**0.5)
        r['vs'] = np.std(ws[2:], axis=(0,1))/(ncells**0.5)
        
        r['euc'] = np.mean((ws[0] - ws[1])**2, axis=0)**0.5
        r['nclus'] = ncells
        
        pca = PCA(n_components = 3)
        wsc = pca.fit_transform(np.concatenate(ws,axis=1).T).T        
        r['pcs'] = wsc


        res[split] = r

    np.save(Path(pth_res,'grand_averages.npy'), res,
                allow_pickle=True)
    
    
def d_var_stacked_single_session(eid, split, min_reg = 10):
    time0 = time.perf_counter()

    '''
    average d_var_m via nanmean across insertions in the same session/animal,
    remove regions with too few neurons (min_reg)
    compute maxes, latencies and p-values
    '''
    
    print(split)
    pth = Path(one.cache_dir, 'manifold', split) 
    ss = os.listdir(pth)  # get insertions
    

    # pool data for illustrative PCA
    acs = []
    ws = [] 
    regdv0 = {}
    regde0 = {}
    
    # group results across insertions
    for s in ss:
        if s.split('_')[0]==eid:
            D_ = np.load(Path(pth,s), allow_pickle=True).flat[0]
            acs.append(D_['acs'])
            ws.append(D_['ws'])        
        
            for reg in D_['D']:
                if reg not in regdv0:
                    regdv0[reg] = []
                regdv0[reg].append(np.array(D_['D'][reg]['d_vars'])/b_size)
                if reg not in regde0:
                    regde0[reg] = []            
                regde0[reg].append(np.array(D_['D'][reg]['d_eucs'])/b_size)

    acs = np.concatenate(acs)
    ws = np.concatenate(ws, axis=1)
    regs0 = Counter(acs)
    regs = {reg: regs0[reg] for reg in regs0 if regs0[reg] > min_reg}
        
    # nansum across insertions and take sqrt
    regdv = {reg: (np.nansum(regdv0[reg],axis=0)/regs[reg])**0.5 
                  for reg in regs }         
    regde = {reg: (np.nansum(regde0[reg],axis=0)/regs[reg])**0.5
                  for reg in regs}
        
    r = {}
    for reg in regs:         
        res = {}        
    
        # get PCA for 3d trajectories
        dat = ws[:,acs == reg,:]
        pca = PCA(n_components = 3)
        wsc = pca.fit_transform(np.concatenate(dat,axis=1).T).T

        res['pcs'] = wsc
        res['nclus'] = regs[reg]

        '''
        var
        '''        
        # amplitudes
        ampsv = [np.max(x) - np.min(x) for x in regdv[reg]]                

        # p value
        res['p_var'] = np.mean(np.array(ampsv) >= ampsv[0])      

        # full curve, subtract null-d mean
        d_var = regdv[reg][0] - np.mean(regdv[reg][1:], axis=0)
        res['d_var'] = d_var - min(d_var)
        res['amp_var'] = max(res['d_var'])
    
        # latency  
        if np.max(res['d_var']) == np.inf:
            loc = np.where(res['d_var'] == np.inf)[0]  
        else:
            loc = np.where(res['d_var'] > 0.7*(np.max(res['d_var'])))[0]
        
        res['lat_var'] = np.linspace(-pre_post[split][0], 
                        pre_post[split][1], len(res['d_var']))[loc[0]]
                      
        '''
        euc
        '''          
        # amplitudes
        ampse = [np.max(x) - np.min(x) for x in regde[reg]]                

        # p value
        res['p_euc'] = np.mean(np.array(ampse) >= ampse[0])      

        # full curve, subtract null-d mean
        d_euc = regde[reg][0] - np.mean(regde[reg][1:], axis=0)
        res['d_euc'] = d_euc - min(d_euc)
        res['amp_euc'] = max(res['d_euc'])
    
        # latency  
        loc = np.where(res['d_euc'] > 0.7*(np.max(res['d_euc'])))[0]
        
        res['lat_euc'] = np.linspace(-pre_post[split][0], 
                        pre_post[split][1], len(res['d_euc']))[loc[0]]   

        r[reg] = res        
    
    np.save(Path(pth_res,f'{split}_{eid}.npy'), r, allow_pickle=True)
           
    time1 = time.perf_counter()    
    print('total time:', time1 - time0, 'sec')
              

def get_average_peth_by_reg(reg):
    
    for split in align:
        pth = Path(one.cache_dir, 'manifold', split) 
        ss = os.listdir(pth)  # get insertions
        
        ws = [] 
        
        # group results across insertions
        for s in ss:
            D_ = np.load(Path(pth,s), 
                    allow_pickle=True).flat[0]        
            if reg not in D_['acs']:
                continue
        
            w = []
            for i in range(len(D_['acs'])):
                if D_['acs'][i]==reg:
                    w.append(D_['ws'][:2, i])
            w = np.array(w)                
            ws.append(w)    
    
        try:
            ws = np.concatenate(ws)
            np.save(Path(pth_res,f'traj_{reg}_{split}.npy'), ws,
                    allow_pickle=True)
        except:
            print(split, 'does not have', reg)

            
def d_with_controls(split, min_reg = min_reg):
    
    print(split)
    pth = Path(one.cache_dir, 'manifold', split) 
    ss = os.listdir(pth)
    
    # pool data for illustrative PCA
    acs = []
    # ws = [] 
    regde0 = {}
    
    # group results across insertions
    for s in ss:
    
        D_ = np.load(Path(pth,s), 
                    allow_pickle=True).flat[0]
                       
        acs.append(D_['acs'])
        # ws.append(D_['ws'])        
        
        for reg in D_['D']:
            if reg not in regde0:
                regde0[reg] = []            
            regde0[reg].append(np.array(D_['D'][reg]['d_eucs'])/b_size)
            
    acs = np.concatenate(acs)
    # ws = np.concatenate(ws, axis=1)
    regs0 = Counter(acs)
    regs = {reg: regs0[reg] for reg in regs0 if regs0[reg] > min_reg}
        
    # nansum across insertions and take sqrt
    regde = {reg: (np.nansum(regde0[reg],axis=0)/regs[reg])**0.5
                  for reg in regs}
        
    r = {}
    for reg in regs:
        # full curves, subtract null-d mean, including curves for control
        d_euc = {}
        for i in range(len(regde[reg])):
            d_euc[i] = regde[reg][i] - np.mean(regde[reg][1:], axis=0)
            d_euc[i] = d_euc[i] - min(d_euc[i])
        
        r[reg] = d_euc
        
    np.save(Path(pth_res,f'd_with_controls_{split}.npy'), r, allow_pickle=True)


def d_var_stacked_multi(splits, min_reg=min_reg, uperms_=False):
    """
    Like d_var_stacked, but takes a list of splits, processes each, and also computes
    combined metrics across all splits. Only uses Euclidean distance (regde) and crossnobis (regxn).
    """
    import time
    time0 = time.perf_counter()
    print("Processing splits:", splits)
    all_uperms = {}

    # Process each split as before, and pool for combined
    for split in splits:
        out = output_split_name(split)
        if (Path(pth_res, f'{out}.npy').exists()
                and Path(pth_res, f'{out}_regde.npy').exists()):
            print(f'Skipping {split}: already finalized in res/ as {out}')
            continue
        print(f"Processing split: {split}")
        pth = Path(one.cache_dir, 'manifold', split)
        ss = os.listdir(pth)
        acs = []
        acs1 = []
        ws = []
        regde0 = {}
        regxn0 = {}
        uperms = {}
        
        for s in ss:
            D_ = np.load(Path(pth, s), allow_pickle=True).flat[0]
            uperms[s.split('.')[0]] = D_['uperms']
            if uperms_:
                continue
            acs.append(D_['acs'])
            acs1.append(D_['acs1'])
            ws.append(D_['ws'])
            
            for reg in D_['D']:
                if reg not in regde0:
                    regde0[reg] = []
                if reg not in regxn0:
                    regxn0[reg] = []
                regde0[reg].append(np.array(D_['D'][reg]['d_eucs']) / b_size)
                # crossnobis across insertions (no sqrt later; already linear)
                if 'd_xnobis' in D_['D'][reg]:
                    regxn0[reg].append(np.array(D_['D'][reg]['d_xnobis']) / b_size)
        
        if uperms_:
            all_uperms[split] = uperms
            continue
            
        # Process individual split results
        acs = np.concatenate(acs)
        acs1 = np.concatenate(acs1)
        ws = np.concatenate(ws, axis=1)
        regs0 = Counter(acs1)
        regs = {reg: regs0[reg] for reg in regs0 if regs0[reg] > min_reg}
        regde = {reg: (np.nansum(regde0[reg], axis=0) / regs[reg]) ** 0.5 for reg in regs}
        regxn = {reg:  (np.nansum(regxn0.get(reg, []), axis=0) / regs[reg]) if reg in regxn0 else None
                 for reg in regs}
        
        r = {}
        for reg in regs:
            res = {}
            dat = ws[:, acs == reg, :]
            pca = PCA(n_components=3)
            wsc = pca.fit_transform(np.concatenate(dat, axis=1).T).T
            res['pcs'] = wsc
            res['nclus'] = regs[reg]
            res['ws'] = dat[:2]
            ampse = [np.max(x) - np.min(x) for x in regde[reg]]
            res['p_euc'] = np.mean(np.array(ampse) >= ampse[0])
            d_euc = regde[reg][0] - np.mean(regde[reg][1:], axis=0)
            res['d_euc'] = d_euc - min(d_euc)
            res['amp_euc'] = max(res['d_euc'])
            loc = np.where(res['d_euc'] > 0.7 * (np.max(res['d_euc'])))[0]
            res['lat_euc'] = np.linspace(-pre_post[split][0], pre_post[split][1], len(res['d_euc']))[loc[0]]

            # --- crossnobis metrics (mirror Euclidean pipeline) ---
            if (regxn[reg] is not None) and (len(regxn[reg]) >= 1):
                amp_x = [np.max(x) - np.min(x) for x in regxn[reg]]
                res['p_xnobis'] = np.mean(np.array(amp_x) >= amp_x[0])
                d_x = regxn[reg][0] - np.mean(regxn[reg][1:], axis=0) if len(regxn[reg]) > 1 else regxn[reg][0]
                res['d_xnobis'] = d_x - np.min(d_x)
                res['amp_xnobis'] = np.max(res['d_xnobis'])
                locx = np.where(res['d_xnobis'] > 0.7 * (np.max(res['d_xnobis'])))[0]
                res['lat_xnobis'] = (np.linspace(-pre_post[split][0], pre_post[split][1], len(res['d_xnobis']))[locx[0]]
                                     if len(locx) else np.nan)
            else:
                res['p_xnobis'] = np.nan
                res['d_xnobis'] = None
                res['amp_xnobis'] = np.nan
                res['lat_xnobis'] = np.nan

            r[reg] = res
            
        # Save regde for this split
        out = output_split_name(split)
        np.save(Path(pth_res, f'{out}_regde.npy'), regde, allow_pickle=True)
        np.save(Path(pth_res, f'{out}_regxn.npy'), regxn, allow_pickle=True)
        np.save(Path(pth_res, f'{out}.npy'), r, allow_pickle=True)

    if uperms_:
        return all_uperms

    # Combine all splits for a grand analysis
    combined_regde = {}
    combined_regxn = {}
    for split in splits:
        # Euclidean aggregation
        out = output_split_name(split)
        split_regde_file = Path(pth_res, f"{out}_regde.npy")
        if split_regde_file.exists():
            split_regde = np.load(split_regde_file, allow_pickle=True).item()
            for reg, curves in split_regde.items():
                if reg not in combined_regde:
                    combined_regde[reg] = [curves[0], np.array(curves[1:])]
                else:
                    combined_regde[reg][0] += curves[0]
                    combined_regde[reg][1] += np.array(curves[1:])
        # Crossnobis aggregation
        split_regxn_file = Path(pth_res, f"{out}_regxn.npy")
        if split_regxn_file.exists():
            split_regxn = np.load(split_regxn_file, allow_pickle=True).item()
            for reg, curves in split_regxn.items():
                if curves is None:
                    continue
                if reg not in combined_regxn:
                    combined_regxn[reg] = [curves[0], np.array(curves[1:]) if len(curves) > 1 else np.empty((0, len(curves[0])))]
                else:
                    combined_regxn[reg][0] += curves[0]
                    if len(curves) > 1:
                        if combined_regxn[reg][1].size == 0:
                            combined_regxn[reg][1] = np.array(curves[1:])
                        else:
                            combined_regxn[reg][1] += np.array(curves[1:])

    # Calculate combined metrics
    r = {}
    for reg, (sum_real_curve, control_curves) in combined_regde.items():
        # Calculate amplitude for real
        amp_real = np.max(sum_real_curve) - np.min(sum_real_curve)
        # Calculate amplitude for controls
        amp_controls = [np.max(c) - np.min(c) for c in control_curves]
        # Calculate p-value
        p_euc = np.mean(np.array(amp_controls) >= amp_real)
        # Latency: where does the real curve cross 0.7*max?
        d_euc = sum_real_curve - np.min(sum_real_curve)
        amp_euc = np.max(d_euc)
        loc = np.where(d_euc > 0.7 * amp_euc)[0]
        if len(loc) == 0:
            lat_euc = np.nan
        else:
            lat_euc = np.linspace(-pre_post[splits[0]][0], pre_post[splits[0]][1], len(d_euc))[loc[0]]
        res = {}
        res['d_euc'] = d_euc
        res['amp_euc'] = amp_euc
        res['p_euc'] = p_euc
        res['lat_euc'] = lat_euc

        # Crossnobis combined metrics (if available)
        if reg in combined_regxn:
            xn_real, xn_ctrl = combined_regxn[reg]
            amp_x_real = np.max(xn_real) - np.min(xn_real)
            amp_x_ctrl = [np.max(c) - np.min(c) for c in xn_ctrl] if xn_ctrl.size else []
            p_x = np.mean(np.array(amp_x_ctrl) >= amp_x_real) if len(amp_x_ctrl) else np.nan
            d_x = xn_real - np.min(xn_real)
            amp_x = np.max(d_x)
            locx = np.where(d_x > 0.7 * amp_x)[0]
            lat_x = (np.linspace(-pre_post[splits[0]][0], pre_post[splits[0]][1], len(d_x))[locx[0]]
                     if len(locx) else np.nan)

            res['d_xnobis'] = d_x
            res['amp_xnobis'] = amp_x
            res['p_xnobis'] = p_x
            res['lat_xnobis'] = lat_x
        r[reg] = res

    combined_name = "_".join(splits)
    np.save(Path(pth_res, f'combined_{combined_name}.npy'), r, allow_pickle=True)
    np.save(Path(pth_res, f'combined_regde_{combined_name}.npy'), combined_regde, allow_pickle=True)
    np.save(Path(pth_res, f'combined_regxn_{combined_name}.npy'), combined_regxn, allow_pickle=True)    
    time1 = time.perf_counter()
    print('total time:', time1 - time0, 'sec')


"""
Usage
"""

# run_align = {
#      'block_duringstim_r_choice_r_f1_1.0': 'stimOn_times',
#      'block_duringstim_r_choice_r_f1_0.25': 'stimOn_times',
#      'block_duringstim_r_choice_r_f1_0.125': 'stimOn_times',
#      'block_duringstim_r_choice_r_f1_0.0625': 'stimOn_times',
#      'block_duringstim_r_choice_r_f1_0.0': 'stimOn_times',
#      'block_duringstim_l_choice_l_f1_1.0': 'stimOn_times',
#      'block_duringstim_l_choice_l_f1_0.25': 'stimOn_times',
#      'block_duringstim_l_choice_l_f1_0.125': 'stimOn_times',
#      'block_duringstim_l_choice_l_f1_0.0625': 'stimOn_times',
#      'block_duringstim_l_choice_l_f1_0.0': 'stimOn_times',
#      'block_duringstim_l_choice_r_f2_1.0': 'stimOn_times',
#      'block_duringstim_l_choice_r_f2_0.25': 'stimOn_times',
#      'block_duringstim_l_choice_r_f2_0.125': 'stimOn_times',
#      'block_duringstim_l_choice_r_f2_0.0625': 'stimOn_times',
#      'block_duringstim_l_choice_r_f2_0.0': 'stimOn_times',
#      'block_duringstim_r_choice_l_f2_1.0': 'stimOn_times',
#      'block_duringstim_r_choice_l_f2_0.25': 'stimOn_times',
#      'block_duringstim_r_choice_l_f2_0.125': 'stimOn_times',
#      'block_duringstim_r_choice_l_f2_0.0625': 'stimOn_times',
#      'block_duringstim_r_choice_l_f2_0.0': 'stimOn_times',

# }

# run_align = {
#      'stim_1.0': 'stimOn_times',
#      # 'stim_0.25': 'stimOn_times',
#      'stim_0.125': 'stimOn_times',
#      # 'stim_0.0625': 'stimOn_times',
#      # 'stim_0.0': 'stimOn_times'       
# }

run_align = {
    # 'intertrial': ['act_block_stim_r_choice_r_f1', 'act_block_stim_l_choice_l_f1', 
    #                'act_block_stim_l_choice_r_f2', 'act_block_stim_r_choice_l_f2'
    #                ],
    'stimOn_times': ['act_block_duringstim_r_choice_r_f1', 'act_block_duringstim_l_choice_l_f1', 
                     'act_block_duringstim_l_choice_r_f2', 'act_block_duringstim_r_choice_l_f2'
                     ],
    'firstMovement_times': ['act_block_stim_r_duringchoice_r_f1', 'act_block_stim_l_duringchoice_l_f1', 
                            'act_block_stim_l_duringchoice_r_f2', 'act_block_stim_r_duringchoice_l_f2'
                            ],
    'intertrial1': ['block_stim_r_choice_r_f1', 'block_stim_l_choice_l_f1', 
                   ],
    'stimOn_times1': ['block_duringstim_r_choice_r_f1', 'block_duringstim_l_choice_l_f1', 
                     'block_duringstim_l_choice_r_f2', 'block_duringstim_r_choice_l_f2'
                     ],
    'firstMovement_times1': ['block_stim_r_duringchoice_r_f1', 'block_stim_l_duringchoice_l_f1', 
                            'block_stim_l_duringchoice_r_f2', 'block_stim_r_duringchoice_l_f2'
                            ],
}

if __name__ == '__main__':
    restart = True

    # All active splits. Loaded ONCE per insertion via the reordered driver
    # (outer loop = insertions), instead of reloading spikes/trials per split.
    intertrial_splits = ['block_only', 'act_block_only']
    duringtrial_splits = [s for splits in run_align.values() for s in splits]
    all_splits = intertrial_splits + duringtrial_splits

    get_all_d_vars_allsplits(all_splits, bycontrast=False, restart=restart,
                             stream_pool=True)

    # Pool across insertions per split / split-group (skip splits already finalized).
    for split in intertrial_splits:
        out = output_split_name(split)
        if not Path(pth_res, f'{out}.npy').exists():
            d_var_stacked(split)
    for timeframe, splits in run_align.items():
        d_var_stacked_multi(splits)

