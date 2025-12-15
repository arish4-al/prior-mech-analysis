## not discarding cells with 0 or nan for all bins; 
## but only included goodcells for counting/normalization (acs1)

from one.api import ONE
from brainbox.singlecell import bin_spikes2D
from brainwidemap import (bwm_query, load_good_units, 
                          load_trials_and_mask, bwm_units)
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
from adjustText import adjust_text
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from PIL import Image
import io
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

import random
from random import shuffle
from copy import deepcopy
import time
import sys

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
alpha = 0.2 # inverse of time constant for action kernel calculation

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
         'durings_srcrbl_slclbl':'stimOn_times',
         'durings_srcrbr_slclbr':'stimOn_times',
         'durings_srcrbl_slclbr':'stimOn_times',
         'durings_slclbl_srcrbr':'stimOn_times',
         'durings_slcrbl_srclbl':'stimOn_times',
         'durings_slcrbl_srclbr':'stimOn_times',
         'durings_slcrbr_srclbl':'stimOn_times',
         'durings_slcrbr_srclbr':'stimOn_times',
         'duringc_srcrbl_slclbl':'firstMovement_times',
         'duringc_srcrbr_slclbr':'firstMovement_times',
         'duringc_srcrbl_slclbr':'firstMovement_times',
         'duringc_slclbl_srcrbr':'firstMovement_times',
         'duringc_slcrbl_srclbl':'firstMovement_times',
         'duringc_slcrbl_srclbr':'firstMovement_times',
         'duringc_slcrbr_srclbl':'firstMovement_times',
         'duringc_slcrbr_srclbr':'firstMovement_times',
        #  'srcrbl_slclbl':'stimOn_times',
        #  'srcrbr_slclbr':'stimOn_times',
        #  'srcrbl_slclbr':'stimOn_times',
        #  'slclbl_srcrbr':'stimOn_times',
        #  'slcrbl_srclbl':'stimOn_times',
        #  'slcrbl_srclbr':'stimOn_times',
        #  'slcrbr_srclbl':'stimOn_times',
        #  'slcrbr_srclbr':'stimOn_times',
        #  'act_srcrbl_slclbl':'stimOn_times',
        #  'act_srcrbr_slclbr':'stimOn_times',
        #  'act_srcrbl_slclbr':'stimOn_times',
        #  'act_slclbl_srcrbr':'stimOn_times',
        #  'act_slcrbl_srclbl':'stimOn_times',
        #  'act_slcrbl_srclbr':'stimOn_times',
        #  'act_slcrbr_srclbl':'stimOn_times',
        #  'act_slcrbr_srclbr':'stimOn_times',
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


# one = ONE(cache_dir='/om2/user/arily/int-brain-lab/ONE',
#           base_url='https://openalyx.internationalbrainlab.org',
#           password='international', silent=True)  # (mode='local')
one = ONE(base_url='https://alyx.internationalbrainlab.org')
ba = AllenAtlas()
br = BrainRegions()

# save results for plotting here
pth_res = Path(one.cache_dir, 'manifold', 'res') 
pth_res.mkdir(parents=True, exist_ok=True)


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


def get_d_vars(split, pid, mapping='Beryl', lowcontrast=False,
               control=True, nrand = 2000, bycontrast=False):

    '''
    for a given session, probe, bin neural activity
    cut into trials, compute d_var per region
    '''
    
    
    eid,probe = one.pid2eid(pid)
    
    # load in spikes
    spikes, clusters = load_good_units(one, pid)    

    if 'duringstim' in split:
        saturation_intervals = 'saturation_stim_plus04'
            
    elif 'duringchoice' in split:
         saturation_intervals = 'saturation_move_minus02'
            
    elif 'fback' in split:
        saturation_intervals = 'saturation_feedback_plus04'
    
    else:
        saturation_intervals='saturation_stim_plus04'
                              # 'saturation_feedback_plus04',
                              # 'saturation_move_minus02',
                              # 'saturation_stim_minus04_minus01',
                              # 'saturation_stim_plus06',
                              # 'saturation_stim_minus06_plus06']


    # Load in trials data and mask bad trials (False if bad)
    trials, mask = load_trials_and_mask(one, eid, 
                   saturation_intervals = saturation_intervals)        
    # remove certain trials
    trials = trials[mask]
    if 'block' in split:
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

        
    events = []
    trn = []
    if bycontrast:
        contrast = float(split.split('_')[-1])
        if split == f'stim_{contrast}':
            for side in ['Left', 'Right']:
                events.append(trials['stimOn_times'][trials[f'contrast{side}']==contrast])
                trn.append(
                    np.arange(len(trials['stimOn_times']))[trials[f'contrast{side}']==contrast])

    if 'stim_l' in split:
        if bycontrast:
            trials = trials[trials[f'contrastLeft']==contrast]
        else:
            trials = trials[~np.isnan(trials[f'contrastLeft'])]
        if 'choice_l' in split and 'f1' in split: # correct trials, f1
            trials = trials[trials['choice'] == 1]
            for pleft in [0.8, 0.2]:
                    events.append(trials[align[split]][np.bitwise_and.reduce([
                        trials['feedbackType'] == 1, trials['probabilityLeft'] == pleft])])
                    trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
                        trials['feedbackType'] == 1, trials['probabilityLeft'] == pleft])])
        elif 'choice_r' in split and 'f2' in split: 
            # choice_r trials, stim_l so these are incorrect trials, f2
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
        if bycontrast:  
            trials = trials[trials[f'contrastRight']==contrast]
        else:
            trials = trials[~np.isnan(trials[f'contrastRight'])]
        if 'choice_l' in split and 'f2' in split: # incorrect trials, f2
            trials = trials[trials['choice'] == 1]
            for pleft in [0.8, 0.2]:
                    events.append(trials[align[split]][np.bitwise_and.reduce([
                        trials['feedbackType'] == -1,trials['probabilityLeft'] == pleft])])
                    trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
                        trials['feedbackType'] == -1,trials['probabilityLeft'] == pleft])])
        elif 'choice_r' in split and 'f1' in split: # choice_r trials, correct, f1
            trials = trials[trials['choice'] == -1]
            for pleft in [0.8, 0.2]:
                    events.append(trials[align[split]][np.bitwise_and.reduce([
                        trials['feedbackType'] == 1,trials['probabilityLeft'] == pleft])])
                    trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
                        trials['feedbackType'] == 1,trials['probabilityLeft'] == pleft])])
        else:
            print('what is the split?', split)
            return
   
    elif 'srcrbl_slclbl' in split:
        trials = trials[trials['probabilityLeft'] == 0.8]
        events.append(trials[align[split]][np.bitwise_and.reduce([
                        trials['choice'] == -1,np.isnan(trials['contrastLeft'])])])
        trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
                        trials['choice'] == -1,np.isnan(trials['contrastLeft'])])])
        events.append(trials[align[split]][np.bitwise_and.reduce([
                        trials['choice'] == 1,np.isnan(trials['contrastRight'])])])
        trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
                        trials['choice'] == 1,np.isnan(trials['contrastRight'])])])
        
    elif 'srcrbr_slclbr' in split:
        trials = trials[trials['probabilityLeft'] == 0.2]
        events.append(trials[align[split]][np.bitwise_and.reduce([
                        trials['choice'] == -1,np.isnan(trials['contrastLeft'])])])
        trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
                        trials['choice'] == -1,np.isnan(trials['contrastLeft'])])])
        events.append(trials[align[split]][np.bitwise_and.reduce([
                        trials['choice'] == 1,np.isnan(trials['contrastRight'])])])
        trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
                        trials['choice'] == 1,np.isnan(trials['contrastRight'])])])
        
    elif 'srcrbl_slclbr' in split:
        #trials = trials[~rm_trials]
        events.append(trials[align[split]][np.bitwise_and.reduce([
                      trials['probabilityLeft'] == 0.8, 
                      trials['choice'] == -1,np.isnan(trials['contrastLeft'])])])
        trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
                   trials['probabilityLeft'] == 0.8, 
                   trials['choice'] == -1,np.isnan(trials['contrastLeft'])])])
        events.append(trials[align[split]][np.bitwise_and.reduce([
                      trials['probabilityLeft'] == 0.2, 
                      trials['choice'] == 1,np.isnan(trials['contrastRight'])])])
        trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
                   trials['probabilityLeft'] == 0.2, 
                   trials['choice'] == 1,np.isnan(trials['contrastRight'])])])
        
    elif 'slclbl_srcrbr' in split:
        #trials = trials[~rm_trials]
        events.append(trials[align[split]][np.bitwise_and.reduce([
                      trials['probabilityLeft'] == 0.8, 
                      trials['choice'] == 1,np.isnan(trials['contrastRight'])])])
        trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
                   trials['probabilityLeft'] == 0.8, 
                   trials['choice'] == 1,np.isnan(trials['contrastRight'])])])
        events.append(trials[align[split]][np.bitwise_and.reduce([
                      trials['probabilityLeft'] == 0.2, 
                      trials['choice'] == -1,np.isnan(trials['contrastLeft'])])])
        trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
                   trials['probabilityLeft'] == 0.2, 
                   trials['choice'] == -1,np.isnan(trials['contrastLeft'])])])
        
    elif 'slcrbl_srclbl' in split:
        trials = trials[trials['probabilityLeft'] == 0.8]
        events.append(trials[align[split]][np.bitwise_and.reduce([
                        trials['choice'] == -1,np.isnan(trials['contrastRight'])])])
        trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
                        trials['choice'] == -1,np.isnan(trials['contrastRight'])])])
        events.append(trials[align[split]][np.bitwise_and.reduce([
                        trials['choice'] == 1,np.isnan(trials['contrastLeft'])])])
        trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
                        trials['choice'] == 1,np.isnan(trials['contrastLeft'])])])
        
    elif 'slcrbr_srclbr' in split:
        trials = trials[trials['probabilityLeft'] == 0.2]
        events.append(trials[align[split]][np.bitwise_and.reduce([
                        trials['choice'] == -1,np.isnan(trials['contrastRight'])])])
        trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
                        trials['choice'] == -1,np.isnan(trials['contrastRight'])])])
        events.append(trials[align[split]][np.bitwise_and.reduce([
                        trials['choice'] == 1,np.isnan(trials['contrastLeft'])])])
        trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
                        trials['choice'] == 1,np.isnan(trials['contrastLeft'])])])
        
    elif 'slcrbr_srclbl' in split:
        #trials = trials[~rm_trials]
        events.append(trials[align[split]][np.bitwise_and.reduce([
                      trials['probabilityLeft'] == 0.2, 
                      trials['choice'] == -1,np.isnan(trials['contrastRight'])])])
        trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
                   trials['probabilityLeft'] == 0.2, 
                   trials['choice'] == -1,np.isnan(trials['contrastRight'])])])
        events.append(trials[align[split]][np.bitwise_and.reduce([
                      trials['probabilityLeft'] == 0.8, 
                      trials['choice'] == 1,np.isnan(trials['contrastLeft'])])])
        trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
                   trials['probabilityLeft'] == 0.8, 
                   trials['choice'] == 1,np.isnan(trials['contrastLeft'])])])
        
    elif 'slcrbl_srclbr' in split:
        #trials = trials[~rm_trials]
        events.append(trials[align[split]][np.bitwise_and.reduce([
                      trials['probabilityLeft'] == 0.8, 
                      trials['choice'] == -1,np.isnan(trials['contrastRight'])])])
        trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
                   trials['probabilityLeft'] == 0.8, 
                   trials['choice'] == -1,np.isnan(trials['contrastRight'])])])
        events.append(trials[align[split]][np.bitwise_and.reduce([
                      trials['probabilityLeft'] == 0.2, 
                      trials['choice'] == 1,np.isnan(trials['contrastLeft'])])])
        trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
                   trials['probabilityLeft'] == 0.2, 
                   trials['choice'] == 1,np.isnan(trials['contrastLeft'])])])
    
    elif 'block_only' in split:
        for pleft in [0.8, 0.2]:
            events.append(trials[align[split]][trials['probabilityLeft'] == pleft])
            trn.append(np.arange(len(trials['choice']))[trials['probabilityLeft'] == pleft])

    else:
        print('what is the split?', split)
        return
    
    print('#trials per condition: ',len(trn[0]), len(trn[1]))
    assert (len(trn[0]) != 0) and (len(trn[1]) != 0), 'zero trials to average'
           
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
        # get mean and var across trials
        w0 = [bi.mean(axis=0) for bi in bins]  
        s0 = [bi.var(axis=0) for bi in bins]
        
        perms = []  # keep track of random trial splits to test sig
        
        # nrand times random impostor/pseudo split of trials 
        for i in range(nrand):
            
            if 'block_only' in split: # 'block' pseudo sessions
                ys = generate_pseudo_blocks(ntr, first5050=0) == 0.8
        
            else:
                # simply shuffle the two labels (block l vs r)                
                tr_c = dx[np.argsort(dx[:,1])][:,0]  # true labels
                tr_c2 = deepcopy(tr_c)
                
                tr_c2 = np.array(random.sample(list(tr_c), len(tr_c)))
                ys = tr_c2 == 1 # boolean shuffled labels
                
            w0.append(b[ys].mean(axis=0))
            s0.append(b[ys].var(axis=0))
            
            w0.append(b[~ys].mean(axis=0))
            s0.append(b[~ys].var(axis=0))                      

            perms.append(ys)

    else: # average trials per condition
        print('all trials')
        w0 = [bi.mean(axis=0) for bi in bins] 
        s0 = [bi.var(axis=0) for bi in bins]

    ws = np.array(w0)
    ss = np.array(s0)

    # --- align permutations list with a "true labels" entry first (for crossnobis) ---
    ys_true = dx[np.argsort(dx[:, 1])][:, 0].astype(bool)
    perms = [ys_true] + perms  # now perms[j] matches ws_[2*j], ws_[2*j+1]
    
    regs = Counter(acs)

    # Keep single cell d_var in extra file for computation of mean
    # Can't be done with control data as files become too large 
    # strictly standardized mean difference
    d_var = (((ws[0] - ws[1])/
              ((ss[0] + ss[1])**0.5))**2)
              
    d_euc = (ws[0] - ws[1])**2          
              
    D_ = {}
    D_['acs'] = acs
    D_['acs1'] = acs1
    D_['d_vars'] = d_var
    D_['d_eucs'] = d_euc
    D_['ws'] = ws[:ntravis]

    if not control:
        return D_

    #  Sum together cells in same region to save memory
    D = {}
    
    for reg in regs:
    
        res = {}

        ws_ = [y[acs == reg] for y in ws]
        ss_ = [y[acs == reg] for y in ss]
     
        res['nclus'] = sum(acs1 == reg)
        d_vars = []
        d_eucs = []
        d_xnobis = []
        # precompute region mask once for xnobis
        reg_mask = (acs == reg)
                
        for j in range(len(ws_)//2):

            # strictly standardized mean difference
            d_var = (((ws_[2*j] - ws_[2*j + 1])/
                      ((ss_[2*j] + ss_[2*j + 1])**0.5))**2)
            
            # Euclidean distance          
            d_euc = (ws_[2*j] - ws_[2*j + 1])**2

            # --- crossnobis (diagonal Σ): (Δμ_1)^T Σ^{-1} (Δμ_2) ---
            ys = perms[j]  # boolean labels for condition 1 in this (true/permuted) split

            # means per half & condition, restricted to region
            # half 1
            m1_h1 = b[half1 & ys][:, reg_mask, :].mean(axis=0)
            m0_h1 = b[half1 & ~ys][:, reg_mask, :].mean(axis=0)
            # half 2
            m1_h2 = b[half2 & ys][:, reg_mask, :].mean(axis=0)
            m0_h2 = b[half2 & ~ys][:, reg_mask, :].mean(axis=0)

            # Δμ per half
            dmu_h1 = (m1_h1 - m0_h1)           # shape: (n_reg_neurons, nbins)
            dmu_h2 = (m1_h2 - m0_h2)

            # diagonal Σ estimate: pooled variance across conditions for this pair
            # (use ss_ entries, already per-condition variances)
            var_pooled = 0.5 * (ss_[2*j] + ss_[2*j + 1])
            inv_var = 1.0 / (var_pooled + 1e-12)

            # crossnobis per time bin: sum over neurons of Δμ_h1 * Σ^{-1} * Δμ_h2
            d_xcv_bins = np.nansum(dmu_h1 * inv_var * dmu_h2, axis=0)  # shape (nbins,)

            # sum over cells, divide by #neu later
            d_var_m = np.nansum(d_var,axis=0)
            d_euc_m = np.sum(d_euc,axis=0)
            
            d_vars.append(d_var_m)
            d_eucs.append(d_euc_m)
            d_xnobis.append(d_xcv_bins)
            
        res['d_vars'] = d_vars
        res['d_eucs'] = d_eucs
        res['d_xnobis'] = d_xnobis
        
        D[reg] = res
        
    D_['uperms'] = len(np.unique([str(x.astype(int)) for x in perms]))
    D_['D'] = D    
    return D_    


def identify_good_session(eid):
    #sess_loader = SessionLoader(one, eid)
    #sess_loader.load_trials()
    #trials = sess_loader.trials
    trials, mask = bwm_loading.load_trials_and_mask(one, eid)
    
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
    

    # pool data for illustrative PCA
    acs = []
    acs1 = []
    ws = [] 
    regdv0 = {}
    regde0 = {}
    uperms = {}
    
    # group results across insertions
    for s in ss:
    
        D_ = np.load(Path(pth,s), 
                    allow_pickle=True).flat[0]
                       
        uperms[s.split('.')[0]] = D_['uperms']
        if uperms_:
            continue
        acs.append(D_['acs'])
        acs1.append(D_['acs1'])
        ws.append(D_['ws'])        
        
        for reg in D_['D']:
            if reg not in regdv0:
                regdv0[reg] = []
            regdv0[reg].append(np.array(D_['D'][reg]['d_vars'])/b_size)
            if reg not in regde0:
                regde0[reg] = []            
            regde0[reg].append(np.array(D_['D'][reg]['d_eucs'])/b_size)
    
    if uperms_:
        return uperms

    acs = np.concatenate(acs)
    acs1 = np.concatenate(acs1)
    ws = np.concatenate(ws, axis=1)
    regs0 = Counter(acs1)
    regs = {reg: regs0[reg] for reg in regs0 if regs0[reg] > min_reg}
        
    # nansum across insertions and take sqrt
    regdv = {reg: (np.nansum(regdv0[reg],axis=0)/regs[reg])**0.5 
                  for reg in regs }         
    regde = {reg: (np.nansum(regde0[reg],axis=0)/regs[reg])**0.5
                  for reg in regs}
        
    r = {}
    for reg in regs:         
        res = {}        
    
        # # get PCA for 3d trajectories
        dat = ws[:,acs == reg,:]
        # pca = PCA(n_components = 3)
        # wsc = pca.fit_transform(np.concatenate(dat,axis=1).T).T

        # res['pcs'] = wsc
        res['nclus'] = regs[reg]
        res['ws'] = dat[:2] #first two are real trajectories

        '''
        var
        '''        
        # amplitudes
#        ampsv = [np.max(x) - np.min(x) for x in regdv[reg]]                

        # p value
#        res['p_var'] = np.mean(np.array(ampsv) >= ampsv[0])      

        # full curve, subtract null-d mean
#        d_var = regdv[reg][0] - np.mean(regdv[reg][1:], axis=0)
#        res['d_var'] = d_var - min(d_var)
#        res['amp_var'] = max(res['d_var'])
    
        # latency  
#        if np.max(res['d_var']) == np.inf:
#            loc = np.where(res['d_var'] == np.inf)[0]  
#        else:
#            loc = np.where(res['d_var'] > 0.7*(np.max(res['d_var'])))[0]
        
#        res['lat_var'] = np.linspace(-pre_post[split][0], 
#                        pre_post[split][1], len(res['d_var']))[loc[0]]
                      
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

        res['nclus'] = regs[reg]
        r[reg] = res        
    
    np.save(Path(pth_res,f'{split}.npy'), r, allow_pickle=True)
    np.save(Path(pth_res, f'{split}_regde.npy'), regde, allow_pickle=True)
           
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
              
'''    
#####################################################
### plotting 
#####################################################    
'''

def get_allen_info():
    dfa = pd.read_csv(Path(pth_res,'allen_structure_tree.csv'))
    
    # get colors per acronym and transfomr into RGB
    dfa['color_hex_triplet'] = dfa['color_hex_triplet'].fillna('FFFFFF')
    dfa['color_hex_triplet']  = dfa['color_hex_triplet'
                                    ].replace('19399','19399a')
    dfa['color_hex_triplet']  = dfa['color_hex_triplet'] .replace('0','FFFFFF')
    dfa['color_hex_triplet'] = '#' + dfa['color_hex_triplet'].astype(str)
    dfa['color_hex_triplet'] = dfa['color_hex_triplet'
                                ].apply(lambda x: 
                                mpl.colors.to_rgba(x))
                                
    palette = dict(zip(dfa.acronym, dfa.color_hex_triplet))

    return dfa, palette           


def put_panel_label(ax, k):                    
        ax.annotate(string.ascii_lowercase[k],(-0.05, 1.15),
                        xycoords='axes fraction',
                        fontsize=f_size*1.5, va='top', 
                        ha='right', weight='bold')


def plot_all(exs_reg, curve='euc', amp_number=False, intro = False, 
             sigl=0.01, only3d = False, onlyScat = False, 
             single_scat=False, auto_label_adjust = False,
             ga_pcs=True, lowcontrast=False):

    '''
    main figure: show example trajectories,
    d_var_m for select regions and scatter all sigificant regions
    curve in [var, euc]
    sigl: significance level, default 0.05
    intro: if False, don't plot schematics of method and contrast
    only3d: only 3d plots 
    '''

    
    nrows = 12
        
    fig = plt.figure(figsize=(10, 10*(2**.5)))  #10, 15
    gs = fig.add_gridspec(nrows, len(align))
    
    dfa, palette = get_allen_info()
                   

    axs = []
    k = 0  # panel counter
    row = 0  # row counter
    

    '''
    get significant regions
    '''           
    tops = {}
    regsa = []

    for split in align: 
        
        if lowcontrast:
            d = np.load(Path(pth_res,f'{split}_lowcontrast.npy'),
                    allow_pickle=True).flat[0]
        else:
            d = np.load(Path(pth_res,f'{split}.npy'),
                    allow_pickle=True).flat[0]              
             
        maxs = np.array([d[x][f'amp_{curve}'] for x in d])
        acronyms = np.array(list(d.keys()))
        order = list(reversed(np.argsort(maxs)))
        maxs = maxs[order]
        acronyms = acronyms[order] 
             
        tops[split] = [acronyms, 
                      [d[reg][f'p_{curve}'] for reg in acronyms], maxs]

            
        maxs = np.array([d[reg][f'amp_{curve}'] for reg in acronyms
                         if d[reg][f'p_{curve}'] < sigl])
                         
        maxsf = [v for v in maxs if not (math.isinf(v) or math.isnan(v))] 

        print(split, curve)
        print(f'{len(maxsf)}/{len(d)} are significant')
        tops[split+'_s'] = f'{len(maxsf)}/{len(d)}'
        regs_a = [tops[split][0][j] for j in range(len(tops[split][0]))
                if tops[split][1][j] < sigl]
        
        
        regsa.append(regs_a)
        print(regs_a)
        print(' ')


    #  get Cosmos parent region for yellow color adjustment
    regsa.append([exs_reg])
    regsa = np.unique(np.concatenate(regsa))
    cosregs_ = [dfa[dfa['id'] == int(dfa[dfa['acronym']==reg]['structure_id_path']
           .values[0].split('/')[4])]['acronym']
           #.values[0] for reg in regsa]
           .values[0] for reg in list(d.keys())]
    
    cosregs = dict(zip(regsa,cosregs_)) 

    v = 0 if intro else 3  # indexing rows
    vv = 1 if v == 0 else 0  # indexing columns

    
    if not onlyScat:

        
        '''
        load schematic intro and contrast trajectory plot (convert from svg)
        '''

        if intro:
    
            axs.append(fig.add_subplot(gs[:3,:]))
      
            pdf = fitz.open(Path(pth_res,'intro2.pdf'))
            rgb = pdf[0].get_pixmap(dpi=600)
            pil_image = Image.open(io.BytesIO(rgb.tobytes()))
                    
            imgplot = axs[k].imshow(pil_image.convert('RGB'))
            axs[k].axis('off')
            put_panel_label(axs[k], k)
            k += 1
            row += 1

           
        '''
        Trajectories for example regions in PCA embedded 3d space
        '''
                         
        c = 0
        for split in align:
        
            # pick example region
            reg = exs_reg
            
            if ga_pcs:
                d = np.load(Path(pth_res,'grand_averages.npy'),
                  allow_pickle=True).flat[0][split]
                #d = np.load(Path(pth_res, f'{split}_grand_averages.npy'),
                #        allow_pickle=True).flat[0]
            else:
                if lowcontrast:
                    d = np.load(Path(pth_res,f'{split}_lowcontrast.npy'),
                            allow_pickle=True).flat[0][reg] 
                else:
                    d = np.load(Path(pth_res,f'{split}.npy'),
                            allow_pickle=True).flat[0][reg] 

            if only3d:
                axs.append(fig.add_subplot(gs[:,c],
                                           projection='3d'))             
            else:
                axs.append(fig.add_subplot(gs[3-v:6-v,c],
                                           projection='3d'))           

            npcs, allnobs = d['pcs'].shape
            nobs = allnobs // ntravis

            for j in range(2): 
            #for j in range(ntravis):
            
                # 3d trajectory
                cs = d['pcs'][:,nobs * j: nobs * (j+1)].T
        

                if j == 0:
                    col = grad('Blues_r', nobs)
                elif j == 1:
                    col = grad('Reds_r', nobs)
                else:
                    col = grad('Greys_r', nobs)
                        
                axs[k].plot(cs[:,0],cs[:,1],cs[:,2], color=col[len(col)//2],
                        linewidth = 5 if j in [0,1] else 1,
                        label=trial_split[split][j] if j in range(3)
                                  else '_nolegend_', alpha = 0.5)
                              
                p = axs[k].scatter(cs[:,0],cs[:,1],cs[:,2], color=col,
                               edgecolors = col, 
                               s=20 if j in [0,1] else 1,
                               depthshade=False)

            
            axs[k].set_title(f"{split}, {reg} {d['nclus']}" if not ga_pcs
                             else split)   
            axs[k].grid(False)
            axs[k].axis('off')

            if c == 0:
                put_panel_label(axs[k], row)

            c += 1
            k += 1
            
        row += 1
        
                   
        if only3d:
            fig.savefig(Path(pth_res,f'trajectories_{curve}.pdf'), dpi=400)
            plt.show()
            return
                 
        '''
        line plot per 5 example regions per split
        '''
        c = 0  # column index

        for split in align:
            if c == 0:
                axs.append(fig.add_subplot(gs[6-v:8-v,c]))
                #axs[-1].set_ylim(0, 4.5/b_size)
            else:  # to share y axis
                axs.append(fig.add_subplot(gs[6-v:8-v,c]))
                           #,sharey=axs[len(axs)-1]))
            
            if lowcontrast:
                d = np.load(Path(pth_res,f'{split}_lowcontrast.npy'),
                            allow_pickle=True).flat[0] 
            else:
                d = np.load(Path(pth_res,f'{split}.npy'),
                            allow_pickle=True).flat[0] 
                       
            # example regions to illustrate line plots              
            #regs = exs[split]
            regs = regsa # use significant regions only + the exs region for 3d plot
            
            texts = []          
            for reg in regs:
                if reg not in list(d.keys()):
                    print(f'{reg} not in {split}')
                    continue
                if any(np.isinf(d[reg][f'd_{curve}'])):
                    print(f'inf in {curve} of {reg}')
                    continue


                xx = np.linspace(-pre_post[split][0], 
                                  pre_post[split][1], 
                                  len(d[reg][f'd_{curve}']))       
      
                # get units in Hz          
                yy = d[reg][f'd_{curve}']
                
                
                axs[k].plot(xx,yy, linewidth = 2,
                              color=palette[reg], 
                              label=f"{reg} {d[reg]['nclus']}")
                

#                # plot stderror bars on lines
#                axs[k].fill_between(xx,    
#                                 yy + d[reg][f'stde_{curve}'],
#                                 yy - d[reg][f'stde_{curve}'],
#                                 color=palette[reg],
#                                 alpha = 0.2)
                                 
                # put region labels                 
                y = yy[-1]
                x = xx[-1]
                ss = f"{reg} {d[reg]['nclus']}" 

                if cosregs[reg] in ['CBX', 'CBN']:  # darken yellow
                    texts.append(axs[k].text(x, y, ss, 
                                             color = 'k',
                                             fontsize=8))             
                                               
                texts.append(axs[k].text(x, y, ss, 
                                         color = palette[reg],
                                         fontsize=8))                 




            axs[k].axvline(x=0, lw=0.5, linestyle='--', c='k')
            
            if split in ['block', 'choice']:
                ha = 'left'
            else:
                ha = 'right'              

            axs[k].spines['top'].set_visible(False)
            axs[k].spines['right'].set_visible(False)
            if c == 0:        
                axs[k].set_ylabel(f'distance [Hz]')
            axs[k].set_xlabel('time [sec]')

            if c == 0:
                put_panel_label(axs[k], row)
            
            c += 1        
            k += 1
            
        row += 1
                   
    if onlyScat:
        v = 8

    '''
    scatter latency versus max amplitude for significant regions
    '''   
 
    fsize = 20 if single_scat else 9 if onlyScat else 7
    dsize = 120 if single_scat else 9 if onlyScat else 4 # was 1
    

    if amp_number: 
        fig2 = plt.figure()
        axss = []
        fig2.suptitle(f'distance metric: {curve}')
    
    if single_scat:
        figs = [plt.subplots(figsize=(10,10)) 
                for split in align]
           
            
    c = 0  
    for split in align:

        if not single_scat:
            
            if c == 0:
                axs.append(fig.add_subplot(gs[8-v:,c]))
            else:
                axs.append(fig.add_subplot(gs[8-v:,c]))
                           #,sharey=axs[len(axs)-1]
                                  
            axs[-1].set_ylim(0, 4.5)
        
        else:   
            axs.append(figs[c][1])
            axs[-1].set_ylim(0, 7.5/b_size if split == 'fback' else 4.5/b_size)
            axs[-1].xaxis.set_major_locator(ticker.MultipleLocator(0.05))


        if lowcontrast:
            d = np.load(Path(pth_res,f'{split}_lowcontrast.npy'),
                        allow_pickle=True).flat[0] 
        else:
            d = np.load(Path(pth_res,f'{split}.npy'),
                        allow_pickle=True).flat[0] 

        acronyms = [tops[split][0][j] for j in range(len(tops[split][0]))] 
        ac_sig = np.array([True if tops[split][1][j] < sigl 
                        else False for j in range(len(tops[split][0]))])

        maxes = np.array([d[x][f'amp_{curve}'] for x in acronyms])
        lats = np.array([d[x][f'lat_{curve}'] for x in acronyms])
        #stdes = np.array([d[x][f'stde_{curve}'] for x in acronyms])
        cols = [palette[reg] for reg in acronyms]


        if amp_number:  # supp figure for correlation of nclus and maxes
            axss.append(fig2.add_subplot(int(f'1{len(align)}{c+1}')))
            nums = [1/d[reg]['nclus'] for reg in np.array(acronyms)[ac_sig]]

            l = list(zip(nums, np.array(maxes)[ac_sig], 
                               np.array(cols)[ac_sig]))
            df = pd.DataFrame(l, columns=['1/nclus', 'maxes', 'cols'])
                  
            sns.regplot(ax=axss[c], x="1/nclus", y="maxes", data=df)
            axss[c].set_title(split)
        
        # yerr = 100*maxes/d[reg]['nclus']
        axs[k].errorbar(lats, maxes, yerr=None, fmt='None', 
                        ecolor=cols, ls = 'None', elinewidth = 0.5)
        
        # plot significant regions               
        axs[k].scatter(np.array(lats)[ac_sig], 
                       np.array(maxes)[ac_sig], 
                       color = np.array(cols)[ac_sig], 
                       marker='D',s=dsize)
        
        # plot insignificant regions               
        axs[k].scatter(np.array(lats)[~ac_sig], 
                       np.array(maxes)[~ac_sig], 
                       color = np.array(cols)[~ac_sig], 
                       marker='o',s=dsize/10)                       
                
        # put extra marker for highlighted regions
        #exs_i = [i for i in range(len(acronyms)) 
        #         if acronyms[i] in exs[split]]
        exs_i = [i for i in range(len(acronyms)) 
                 if acronyms[i] in regsa]
                 
#        axs[k].scatter(np.array(lats)[exs_i], np.array(maxes)[exs_i], 
#                       color=np.array(cols)[exs_i], marker='x',s=10*dsize)
         
        texts = []
        alw = 0.1
        for i in range(len(acronyms)):
            if ac_sig[i]:  # only decorate marker with label if sig
                reg = acronyms[i]
                
                if cosregs[reg] in ['CBX', 'CBN']:
                        texts.append(axs[k].annotate('  ' + reg, 
                                    (lats[i], maxes[i]),
                                    fontsize=fsize,color='k',
                                    arrowprops=dict(arrowstyle="-", 
                                    color='k', 
                                    lw=alw) if auto_label_adjust else None))

#                        axs[k].annotate('  ' + reg, 
#                                    (lats[i], maxes[i]),
#                        fontsize=fsize,color=palette[acronyms[i]],
#                        arrowprops=dict(arrowstyle=" ", 
#                                    color=palette[acronyms[i]], 
#                                    lw=alw) if auto_label_adjust else None))
                else:
                    texts.append(axs[k].annotate('  ' + reg, 
                    # f"{reg} {f[reg]['nclus']}" ,
                        (lats[i], maxes[i]),
                        fontsize=fsize,color=palette[acronyms[i]],
                        arrowprops=dict(arrowstyle=" ", 
                                    color=palette[acronyms[i]], 
                                    lw=alw) if auto_label_adjust else None))
                                    
        if auto_label_adjust:        
            adjust_text(texts,force_text=(3, 6),ax=axs[k])

        axs[k].axvline(x=0, lw=0.5, linestyle='--', c='k')
        
        ha = 'left'   
        
        axs[k].text(0, 0.95, align[split],
                      transform=axs[k].get_xaxis_transform(),
                      horizontalalignment = ha, rotation=90,
                      fontsize = f_size*0.8)
                                
        axs[k].spines['top'].set_visible(False)
        axs[k].spines['right'].set_visible(False)        
        if c == 0:   
            axs[k].set_ylabel(f'max dist. [Hz]')
        axs[k].set_xlabel('latency [sec]')
        axs[k].set_title(f"{tops[split+'_s']} sig")
        
        if c == 0:
            put_panel_label(axs[k], row)        

        
        if single_scat:
            figs[c][0].tight_layout()
            figs[c][0].savefig(Path(pth_res, 
                               f'scat_{split}.pdf'), dpi=300)         
            plt.close()
        
        c += 1     
        k += 1


    fig = plt.gcf()
#    fig.tight_layout()
#    
    fig.subplots_adjust(top=0.98,
                        bottom=0.051,
                        left=0.06,
                        right=0.98,
                        hspace=5.1,
                        wspace=0.62)

    #fig.savefig(Path(pth_res,f'manifold_{curve}.pdf'), dpi=400)              
    plt.show()

    
def plot_3d_all_splits(exs_reg, curve='euc', amp_number=False,
             sigl=0.01, auto_label_adjust = False,
             ga_pcs=False, lowcontrast=False):

    '''
    plot example 3d trajectories for all splits in the same space
    d_var_m for select regions and scatter all sigificant regions
    curve in [var, euc]
    sigl: significance level, default 0.05
    '''

    
    nrows = 12
        
    fig = plt.figure(figsize=(10, 10*(2**.5)))  #10, 15
    gs = fig.add_gridspec(nrows, len(align))
    
    dfa, palette = get_allen_info()
                   

    axs = []
    k = 0  # panel counter
           
    '''
    Trajectories for example regions in PCA embedded 3d space
    '''
                         
    c = 0
    axs.append(fig.add_subplot(gs[:,c], projection='3d'))
    idx = 0
    for split in align:
        
            # pick example region
            reg = exs_reg
            if lowcontrast:
                d = np.load(Path(pth_res,f'{split}_lowcontrast.npy'),
                            allow_pickle=True).flat[0][reg]
            else:
                d = np.load(Path(pth_res,f'{split}.npy'),
                            allow_pickle=True).flat[0][reg]

            npcs, allnobs = d['pcs'].shape
            nobs = allnobs // ntravis

            for j in range(2):
            
                # 3d trajectory
                cs = d['pcs'][:,nobs * j: nobs * (j+1)].T
        
                if idx == 0:
                    if j == 0:
                        col = grad('Blues_r', nobs)
                    elif j == 1:
                        col = grad('Reds_r', nobs)
                    else:
                        col = grad('Greys_r', nobs)
                else:
                    if j == 0:
                        col = grad('Greens_r', nobs)
                    elif j == 1:
                        col = grad('Oranges_r', nobs)
                    else:
                        col = grad('Greys_r', nobs)
                        
                axs[k].plot(cs[:,0],cs[:,1],cs[:,2], color=col[len(col)//2],
                        linewidth = 5 if j in [0,1] else 1,
                        label=trial_split[split][j] if j in range(3)
                                  else '_nolegend_', alpha = 0.5)
                              
                p = axs[k].scatter(cs[:,0],cs[:,1],cs[:,2], color=col,
                               edgecolors = col, 
                               s=20 if j in [0,1] else 1,
                               depthshade=False)

            
            axs[k].set_title(f"{reg} {d['nclus']}")   
            axs[k].grid(False)
            axs[k].axis('off')
            axs[k].legend()

            put_panel_label(axs[k], 0)

            #c += 1
            #k += 1
            idx += 1
                    
                   
    plt.show()
    return

    
def plot_grand_average():

    fig, axs = plt.subplots(nrows=3,
                            ncols=len(align))
    
    res = np.load(Path(pth_res,'grand_averages.npy'),
                  allow_pickle=True).flat[0]
    
    cols = {'0':blue_left,
            '1':red_right,
            's': 'gray'}
    
    k = 0
    for split in res:
        r = 0
        for t in ['0', '1', 's']:
            yy = res[split][f'm{t}']
            xx = np.linspace(-pre_post[split][0], 
                      pre_post[split][1], 
                      len(yy))
                                          
            axs[r,k].plot(xx,yy, c = cols[t])   
            axs[r,k].fill_between(xx,    
                         yy + res[split][f'v{t}']/2,
                         yy - res[split][f'v{t}']/2,
                         color=cols[t],
                         alpha = 0.2)
                                
        axs[r,k].axvline(x=0, lw=0.5, linestyle='--', c='k')
        axs[r,k].spines['top'].set_visible(False)
        axs[r,k].spines['right'].set_visible(False)
        if k == 0:        
            axs[r,k].set_ylabel(f'average firing rate [Hz]')
        axs[r,k].set_xlabel('time [sec]')
        axs[r,k].set_title(split)    
        
        r = 1
        
        #yy = np.abs(res[split][f'm0'] - res[split][f'm1'])
        yy = res[split]['euc']
        xx = np.linspace(-pre_post[split][0], 
                  pre_post[split][1], 
                  len(yy))
                                      
        axs[r,k].plot(xx,yy, c = 'k')   

                                
        axs[r,k].axvline(x=0, lw=0.5, linestyle='--', c='k')
        axs[r,k].spines['top'].set_visible(False)
        axs[r,k].spines['right'].set_visible(False)
        if k == 0:        
            axs[r,k].set_ylabel(f'distance [Hz]')
        axs[r,k].set_xlabel('time [sec]')
        axs[r,k].set_title(split) 
        
        axs[0,k].sharex(axs[1,k])
        
        if k > 0:
            axs[r,k].sharey(axs[r,k-1])
  
          
        k +=1 
          
    axs[1,0].get_shared_y_axes().join(axs[1,0],axs[1,1])
    axs[1,0].autoscale()
    
    fig.tight_layout()

    
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
        np.save(Path(pth_res, f'{split}_regde.npy'), regde, allow_pickle=True)
        np.save(Path(pth_res, f'{split}_regxn.npy'), regxn, allow_pickle=True)
        np.save(Path(pth_res, f'{split}.npy'), r, allow_pickle=True)

    if uperms_:
        return all_uperms

    # Combine all splits for a grand analysis
    combined_regde = {}
    combined_regxn = {}
    for split in splits:
        # Euclidean aggregation
        split_regde_file = Path(pth_res, f"{split}_regde.npy")
        if split_regde_file.exists():
            split_regde = np.load(split_regde_file, allow_pickle=True).item()
            for reg, curves in split_regde.items():
                if reg not in combined_regde:
                    combined_regde[reg] = [curves[0], np.array(curves[1:])]
                else:
                    combined_regde[reg][0] += curves[0]
                    combined_regde[reg][1] += np.array(curves[1:])
        # Crossnobis aggregation
        split_regxn_file = Path(pth_res, f"{split}_regxn.npy")
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
    # 'stimOn_times': ['act_block_duringstim_r_choice_r_f1', 'act_block_duringstim_l_choice_l_f1', 
    #                  'act_block_duringstim_l_choice_r_f2', 'act_block_duringstim_r_choice_l_f2'
    #                  ],
    # 'firstMovement_times': ['act_block_stim_r_duringchoice_r_f1', 'act_block_stim_l_duringchoice_l_f1', 
    #                         'act_block_stim_l_duringchoice_r_f2', 'act_block_stim_r_duringchoice_l_f2'
    #                         ],
    # 'intertrial1': ['block_stim_r_choice_r_f1', 'block_stim_l_choice_l_f1', 
    #                ],
    'stimOn_times1': ['block_duringstim_r_choice_r_f1', 'block_duringstim_l_choice_l_f1', 
                     'block_duringstim_l_choice_r_f2', 'block_duringstim_r_choice_l_f2'
                     ],
    'firstMovement_times1': ['block_stim_r_duringchoice_r_f1', 'block_stim_l_duringchoice_l_f1', 
                            'block_stim_l_duringchoice_r_f2', 'block_stim_r_duringchoice_l_f2'
                            ],
}

# run_align = {
#     'block_duringstim_r_choice_r_f1': ['block_duringstim_r_choice_r_f1_0.0',
#     'block_duringstim_r_choice_r_f1_0.0625',
#     'block_duringstim_r_choice_r_f1_0.125',
#     'block_duringstim_r_choice_r_f1_0.25',
#     'block_duringstim_r_choice_r_f1_1.0'],
#     'block_duringstim_l_choice_l_f1': ['block_duringstim_l_choice_l_f1_0.0',
#     'block_duringstim_l_choice_l_f1_0.0625',
#     'block_duringstim_l_choice_l_f1_0.125',
#     'block_duringstim_l_choice_l_f1_0.25',
#     'block_duringstim_l_choice_l_f1_1.0'],
#     'block_duringstim_l_choice_r_f2': ['block_duringstim_l_choice_r_f2_0.0',
#     'block_duringstim_l_choice_r_f2_0.0625',
#     'block_duringstim_l_choice_r_f2_0.125',
#     'block_duringstim_l_choice_r_f2_0.25',
#     'block_duringstim_l_choice_r_f2_1.0'],
#     'block_duringstim_r_choice_l_f2': ['block_duringstim_r_choice_l_f2_0.0',
#     'block_duringstim_r_choice_l_f2_0.0625',
#     'block_duringstim_r_choice_l_f2_0.125',
#     'block_duringstim_r_choice_l_f2_0.25',
#     'block_duringstim_r_choice_l_f2_1.0']
# }


restart = True

# for timeframe in ['intertrial', 'stimOn_times', 'firstMovement_times']:
#     splits = run_align[timeframe]
#     for split in splits:
#         get_all_d_vars(split, bycontrast=False)

for split in ['block_only']:
    get_all_d_vars(split, bycontrast=False, restart=restart)
    d_var_stacked(split)

# split = 'act_block_stim_r_duringchoice_l_f2'
# get_all_d_vars(split, bycontrast=False, restart=restart)

for timeframe, splits in run_align.items():
# for timeframe in ['stimOn_times']:
#     splits = run_align[timeframe]
    for split in splits:
        get_all_d_vars(split, bycontrast=False, restart=restart)
    d_var_stacked_multi(splits)

