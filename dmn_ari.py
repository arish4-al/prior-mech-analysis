from one.api import ONE
from brainbox.singlecell import bin_spikes2D
from brainwidemap import (bwm_query, load_good_units, 
                          load_trials_and_mask, bwm_units)
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions
import iblatlas
from iblatlas.plots import plot_swanson_vector 
from brainbox.io.one import SessionLoader

import sys
sys.path.append('Dropbox/scripts/IBL/')
#from granger import get_volume, get_centroids, get_res, get_structural, get_ari
#from state_space_bwm import get_cmap_bwm, pre_post
#from bwm_figs import variverb

from scipy import signal
import pandas as pd
import numpy as np

from collections import Counter
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import confusion_matrix
from sklearn.cluster import SpectralCoclustering, SpectralBiclustering
from sklearn.cluster import DBSCAN, OPTICS, Birch, MiniBatchKMeans
from numpy.linalg import norm
from scipy.stats import gaussian_kde, f_oneway, pearsonr, spearmanr, kruskal, linregress
from scipy.spatial import distance
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform, cdist
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import SpectralEmbedding
from random import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sknetwork.clustering import Louvain, Leiden, KCenters
from sknetwork.visualization import visualize_graph
from IPython.display import SVG
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from scipy import sparse
from scipy.stats import wasserstein_distance, wasserstein_distance_nd
import gc
from pathlib import Path
import random
from copy import deepcopy
import time, sys, math, string, os
from scipy.stats import spearmanr, zscore, combine_pvalues
import umap
from itertools import combinations, chain
from datetime import datetime
import scipy.ndimage as ndi
from rastermap import Rastermap
import scipy.cluster.hierarchy as sch
import plotly.graph_objects as go
import copy
from collections import defaultdict
#import hdbscan

from matplotlib.axis import Axis
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap, LinearSegmentedColormap   
from matplotlib.gridspec import GridSpec   
import mpldatacursor
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import cm
from venny4py.venny4py import *
import networkx as nx
from termcolor import colored
import dataframe_image as dfi
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import chi2
# from scipy.stats import norm
from scipy.cluster.hierarchy import fcluster
from collections import defaultdict

from PIL import Image

import warnings
warnings.filterwarnings("ignore")
#mpl.use('QtAgg')

# for vari plot
#_, b, lab_cols = labs()
plt.ion() 
 
np.set_printoptions(threshold=sys.maxsize)

plt.rcParams.update(plt.rcParamsDefault)
plt.ion()

f_size = 15  # font size

# canonical colors for left and right trial types
blue_left = [0.13850039, 0.41331206, 0.74052025]
red_right = [0.66080672, 0.21526712, 0.23069468]

T_BIN = 0.0125  # bin size [sec] for neural binning
sts = 0.002  # stride size in [sec] for overlapping bins
ntravis = 30  # #trajectories for vis, first 2 real, rest pseudo

# conversion divident to get bins in seconds 
# (taking strinding into account)
c_sec =  int(T_BIN // sts) / T_BIN


one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international', silent=True)

#base_url='https://openalyx.internationalbrainlab.org',
#          password='international', silent=True 
                   
br = BrainRegions()
#units_df = bwm_units(one)  # canonical set of cells


# save results here
pth_dmn = Path(one.cache_dir, 'dmn', 'res')
pth_dmn.mkdir(parents=True, exist_ok=True)

sigl = 0.05  # significance level (for stacking, plotting, fdr)


# order sensitive: must be tts__ = concat_PETHs(pid, get_tts=True).keys()
tts__ = ['inter_trial', 'blockL', 'blockR', 'block50', 'quiescence', 
         'stimLbLcL', 'stimLbRcL', 'stimLbRcR', 'stimLbLcR', 'stimRbLcL', 
         'stimRbRcL', 'stimRbRcR', 'stimRbLcR', 'motor_init', 'sLbLchoiceL', 
         'sLbRchoiceL', 'sLbRchoiceR', 'sLbLchoiceR', 'sRbLchoiceL', 'sRbRchoiceL', 
         'sRbRchoiceR', 'sRbLchoiceR', 'choiceL', 'choiceR',  'fback1', 'fback0']
#'fback0sRbL', 'fback0sLbR',


PETH_types_dict = {
    'concat': [item for item in tts__],
    'resting': ['inter_trial'],
    'quiescence': ['quiescence'],
    'pre-stim-prior': ['blockL', 'blockR'],
    'block50': ['block50'],
    'stim_surp_incon': ['stimLbRcL','stimRbLcR'],
    'stim_surp_con': ['stimLbLcL', 'stimRbRcR'],
    'stim_all': ['stimLbRcL','stimRbLcR','stimLbLcL', 'stimRbRcR'],
    'mistake': ['stimLbRcR', 'stimLbLcR', 'stimRbLcL', 'stimRbRcL',
                'sLbRchoiceR', 'sLbLchoiceR', 'sRbLchoiceL', 'sRbRchoiceL'],
    'motor_init': ['motor_init'],
    'movement': ['choiceL', 'choiceR'],
    'fback1': ['fback1'],
    'fback0': ['fback0']}      

phases_dict = {
    'concat': 'concat',
    'resting': 'resting',
    'quiescence': 'quiescence',
    'pre-stim-prior': 'pre-stim prior',
    'stim_surp_incon': 'stim surprise',
    'stim_surp_con': 'stim congruent',
    'mistake': 'mistake',
    'motor_init': 'motor init',
    'movement': 'movement',
    'fback1': 'fback correct',
    'fback0': 'fback incorrect'}      


data_lengths={'inter_trial': 72,
              'blockL': 144,
              'blockR': 144,
              'block50': 144,
              'quiescence': 144,
              'stimLbLcL': 96,
              'stimLbRcL': 96,
              'stimLbRcR': 96,
              'stimLbLcR': 96,
              'stimRbLcL': 96,
              'stimRbRcL': 96,
              'stimRbRcR': 96,
              'stimRbLcR': 96,
              'motor_init': 72,
              'sLbLchoiceL': 72,
              'sLbRchoiceL': 72,
              'sLbRchoiceR': 72,
              'sLbLchoiceR': 72,
              'sRbLchoiceL': 72,
              'sRbRchoiceL': 72,
              'sRbRchoiceR': 72,
              'sRbLchoiceR': 72,
              'choiceL': 72,
              'choiceR': 72,
              'fback1': 144,
              'fback0': 144
             }
     
peth_ila = [
    r"$\mathrm{rest}$",
    r"$\mathrm{L_b}$",
    r"$\mathrm{R_b}$",
    r"$\mathrm{50_b}$",
    r"$\mathrm{quies}$",
    r"$\mathrm{L_sL_cL_b, s}$",
    r"$\mathrm{L_sL_cR_b, s}$",
    r"$\mathrm{L_sR_cR_b, s}$",
    r"$\mathrm{L_sR_cL_b, s}$",
    r"$\mathrm{R_sL_cL_b, s}$",
    r"$\mathrm{R_sL_cR_b, s}$",
    r"$\mathrm{R_sR_cR_b, s}$",
    r"$\mathrm{R_sR_cL_b, s}$",
    r"$\mathrm{m}$",
    r"$\mathrm{L_sL_cL_b, m}$",
    r"$\mathrm{L_sL_cR_b, m}$",
    r"$\mathrm{L_sR_cR_b, m}$",
    r"$\mathrm{L_sR_cL_b, m}$",
    r"$\mathrm{R_sL_cL_b, m}$",
    r"$\mathrm{R_sL_cR_b, m}$",
    r"$\mathrm{R_sR_cR_b, m}$",
    r"$\mathrm{R_sR_cL_b, m}$",
    r"$\mathrm{L_{move}}$",
    r"$\mathrm{R_{move}}$",
    r"$\mathrm{feedbk1}$",
    r"$\mathrm{feedbk0}$"
]


peth_dict = dict(zip(tts__, peth_ila))


def get_name(brainregion):
    '''
    get verbose name for brain region acronym
    '''
    regid = br.id[np.argwhere(br.acronym == brainregion)][0, 0]
    return br.name[np.argwhere(br.id == regid)[0, 0]]

dmn_regs = ['ACAd', 'ACAv', 'PL', 'ILA', 'ORBl', 'ORBm', 
            'ORBvl', 'VISa', 'VISam', 'RSPagl','RSPd', 
            'RSPv', 'SSp-tr', 'SSp-ll', 'MOs']

cortical_regions = {
    "Prefrontal": [
        "FRP", "ACAd", "ACAv", "PL", "ILA",
        "ORBl", "ORBm", "ORBvl"
    ],
    "Lateral": [
        "AId", "AIv", "AIp", "GU", "VISc",
        "TEa", "PERI", "ECT"
    ],
    "Somatomotor": [
        "SSs", "SSp-bfd", "SSp-tr", "SSp-ll",
        "SSp-ul", "SSp-un", "SSp-n", "SSp-m",
        "MOp", "MOs"
    ],
    "Visual": [
        "VISal", "VISl", "VISp", "VISpl",
        "VISli", "VISpor", "VISrl"
    ],
    "Medial": [
        "VISa", "VISam", "VISpm",
        "RSPagl", "RSPd", "RSPv"
    ],
    "Auditory": [
        "AUDd", "AUDp", "AUDpo", "AUDv"
    ]
}

cortical_colors = {"Prefrontal": 'r', 
                   "Lateral": 'yellow',
                   "Somatomotor": 'orange',
                   "Visual": 'g',
                   "Medial": 'blue',
                   "Auditory": 'purple'
                  }


networks = {
    # 'concat_1': 'VISp',
    # 'concat_2': 'PRM',
    # 'quiescence_1': 'DP',
    'quiescence_2': 'GRN',
    # 'pre-stim-prior_1': 'VISp',
    # 'pre-stim-prior_2': 'MRN',
    # 'movement_1': 'MOs',
    'movement_2': 'VISp',
    'movement_3': 'CA1',
    # 'mistake_1': 'PRNr',
    # 'mistake_2': 'ORBvl',
    # 'mistake_3': 'SUB',
    'stim_surp_incon_1': 'VISp',
    'stim_surp_incon_2': 'GRN',
    'stim_surp_con_1': 'VAL',
    'stim_surp_con_2': 'IP',
    'stim_surp_con_3': 'VISp',
    'motor_init_1': 'VISp',
    'motor_init_2': 'CP',
    'motor_init_3': 'IP',
    'fback1_1': 'VISp',
    'fback1_2': 'GRN',
    'fback1_3': 'CA1',
    'fback0_1': 'CA1',
    'fback0_2': 'PRNr'
}

networks_full = {
    'concat_1': 'VISp',
    'concat_2': 'PRM',
    'quiescence_1': 'DP',
    'quiescence_2': 'GRN',
    'pre-stim-prior_1': 'VISp',
    'pre-stim-prior_2': 'MRN',
    'movement_1': 'MOs',
    'movement_2': 'VISp',
    'movement_3': 'CA1',
    'mistake_1': 'PRNr',
    'mistake_2': 'ORBvl',
    'mistake_3': 'SUB',
    'stim_surp_incon_1': 'VISp',
    'stim_surp_incon_2': 'GRN',
    'stim_surp_con_1': 'VAL',
    'stim_surp_con_2': 'IP',
    'stim_surp_con_3': 'VISp',
    'motor_init_1': 'VISp',
    'motor_init_2': 'CP',
    'motor_init_3': 'IP',
    'fback1_1': 'VISp',
    'fback1_2': 'GRN',
    'fback1_3': 'CA1',
    'fback0_1': 'CA1',
    'fback0_2': 'PRNr'
}

network_colors = {
    'concat_1': 'green',
    'concat_2': 'pink',
    'quiescence_1': 'green',
    'quiescence_2': 'pink',
    'pre-stim prior_1': 'green',
    'pre-stim prior_2': 'pink',
    'movement_1': 'pink',
    'movement_2': 'green',
    'movement_3': 'green',
    'mistake_1': 'pink',
    'mistake_2': 'green',
    'mistake_3': 'green',
    'stim surprise_1': 'green',
    'stim surprise_2': 'pink',
    'stim congruent_1': 'pink',
    'stim congruent_2': 'pink',
    'stim congruent_3': 'green',
    'motor init_1': 'green',
    'motor init_2': 'pink',
    'motor init_3': 'pink',
    'fback correct_1': 'green',
    'fback correct_2': 'pink',
    'fback correct_3': 'green',
    'fback incorrect_1': 'green',
    'fback incorrect_2': 'pink'
}

raster_types = {
    'stim_response': {'start': [6000], 'end': [6734]},
    'error': {'start': [7200, 10000, 20750, 34600], 
                       'end': [8400, 11000, 21250, 36000]},
    'integrator': {'start':[3700], 'end':[6000]},
    'stim_n_int': {'start':[3700], 'end': [6734]},
    'sequence': {'start': [13450, 21250], 'end': [20750, 34600]}, 
    'move_init': {'start': [840], 'end':[3700]},
    'movement': {'start':[43740], 'end':[44890]}
    
}

# for block analysis
block_pairs_by_contrast = [['sLbLchoiceL_1.0', 'sLbRchoiceL_1.0'],
 ['sLbLchoiceL_0.25', 'sLbRchoiceL_0.25'],
 ['sLbLchoiceL_0.125', 'sLbRchoiceL_0.125'],
 ['sLbLchoiceL_0.0625', 'sLbRchoiceL_0.0625'],
 ['sLbLchoiceL_0.0', 'sLbRchoiceL_0.0'],
 ['sRbLchoiceR_1.0', 'sRbRchoiceR_1.0'],
 ['sRbLchoiceR_0.25', 'sRbRchoiceR_0.25'],
 ['sRbLchoiceR_0.125', 'sRbRchoiceR_0.125'],
 ['sRbLchoiceR_0.0625', 'sRbRchoiceR_0.0625'],
 ['sRbLchoiceR_0.0', 'sRbRchoiceR_0.0'],
 ['stimLbLcL_1.0', 'stimLbRcL_1.0'],
 ['stimLbLcL_0.25', 'stimLbRcL_0.25'],
 ['stimLbLcL_0.125', 'stimLbRcL_0.125'],
 ['stimLbLcL_0.0625', 'stimLbRcL_0.0625'],
 ['stimLbLcL_0.0', 'stimLbRcL_0.0'],
 ['stimRbLcR_1.0', 'stimRbRcR_1.0'],
 ['stimRbLcR_0.25', 'stimRbRcR_0.25'],
 ['stimRbLcR_0.125', 'stimRbRcR_0.125'],
 ['stimRbLcR_0.0625', 'stimRbRcR_0.0625'],
 ['stimRbLcR_0.0', 'stimRbRcR_0.0']]

block_pairs = [
 ['sLbLchoiceL', 'sLbRchoiceL'],
 ['sLbLchoiceR', 'sLbRchoiceR'],
 ['sRbLchoiceL', 'sRbRchoiceL'],
 ['sRbLchoiceR', 'sRbRchoiceR'],
 ['stimLbLcL', 'stimLbRcL'],
 ['stimLbLcR', 'stimLbRcR'],
 ['stimRbLcL', 'stimRbRcL'],
 ['stimRbLcR', 'stimRbRcR'],
]

stim_pairs = [
 ['stimLbLcL', 'stimRbLcL'],
 ['stimLbRcL', 'stimRbRcL'],
 ['stimLbLcR', 'stimRbLcR'],
 ['stimLbRcR', 'stimRbRcR'],
 ['sLbLchoiceL', 'sRbLchoiceL'],
 ['sLbRchoiceL', 'sRbRchoiceL'],
 ['sLbRchoiceR', 'sRbRchoiceR'],
 ['sLbLchoiceR', 'sRbLchoiceR'],
]

choice_pairs = [
 ['stimLbLcL', 'stimLbLcR'],
 ['stimLbRcL', 'stimLbRcR'],
 ['stimRbLcL', 'stimRbLcR'],
 ['stimRbRcL', 'stimRbRcR'],
 ['sLbLchoiceL', 'sLbLchoiceR'],
 ['sLbRchoiceL', 'sLbRchoiceR'],
 ['sRbRchoiceL', 'sRbRchoiceR'],
 ['sRbLchoiceL', 'sRbLchoiceR']
]

ttypes_short_ = ['stimLbLcL', 'stimLbRcL', 'stimLbRcR', 'stimLbLcR', 'stimRbLcL', 'stimRbRcL', 
                'stimRbRcR', 'stimRbLcR', 'sLbLchoiceL', 'sLbRchoiceL', 'sLbRchoiceR', 'sLbLchoiceR', 
                'sRbLchoiceL', 'sRbRchoiceL', 'sRbRchoiceR', 'sRbLchoiceR']

data_lengths_short_ = {'stimLbLcL':96, 
                       'stimLbRcL':96, 
                       'stimLbRcR':96, 
                       'stimLbLcR':96, 
                       'stimRbLcL':96, 
                       'stimRbRcL':96, 
                       'stimRbRcR':96, 
                       'stimRbLcR':96, 
                       'sLbLchoiceL':72, 
                       'sLbRchoiceL':72, 
                       'sLbRchoiceR':72, 
                       'sLbLchoiceR':72, 
                       'sRbLchoiceL':72, 
                       'sRbRchoiceL':72, 
                       'sRbRchoiceR':72, 
                       'sRbLchoiceR':72
                       }


def put_panel_label(ax, k):
    ax.annotate(string.ascii_lowercase[k], (-0.05, 1.15),
                xycoords='axes fraction',
                fontsize=f_size * 1.5, va='top',
                ha='right', weight='bold')


def grad(c, nobs, fr=1):
    '''
    color gradient for plotting trajectories
    c: color map type
    nobs: number of observations
    '''

    cmap = mpl.cm.get_cmap(c)

    return [cmap(fr * (nobs - p) / nobs) for p in range(nobs)]


def get_name(brainregion):
    '''
    get verbose name for brain region acronym
    '''
    regid = br.id[np.argwhere(br.acronym == brainregion)][0, 0]
    return br.name[np.argwhere(br.id == regid)[0, 0]]


def get_allen_info(rerun=False):
    '''
    Function to load Allen atlas info, like region colors
    '''
    
    pth_dmna = Path(one.cache_dir, 'dmn', 'alleninfo.npy')
    
    if (not pth_dmna.is_file() or rerun):
        p = (Path(ibllib.__file__).parent /
             'atlas/allen_structure_tree.csv')

        dfa = pd.read_csv(p)

        # replace yellow by brown #767a3a    
        cosmos = []
        cht = []
        
        for i in range(len(dfa)):
            try:
                ind = dfa.iloc[i]['structure_id_path'].split('/')[4]
                cr = br.id2acronym(ind, mapping='Cosmos')[0]
                cosmos.append(cr)
                if cr == 'CB':
                    cht.append('767A3A')
                else:
                    cht.append(dfa.iloc[i]['color_hex_triplet'])    
                        
            except:
                cosmos.append('void')
                cht.append('FFFFFF')
                

        dfa['Cosmos'] = cosmos
        dfa['color_hex_triplet2'] = cht
        
        # get colors per acronym and transfomr into RGB
        dfa['color_hex_triplet2'] = dfa['color_hex_triplet2'].fillna('FFFFFF')
        dfa['color_hex_triplet2'] = dfa['color_hex_triplet2'
                                       ].replace('19399', '19399a')
        dfa['color_hex_triplet2'] = dfa['color_hex_triplet2'].replace(
                                                         '0', 'FFFFFF')
        dfa['color_hex_triplet2'] = '#' + dfa['color_hex_triplet2'].astype(str)
        dfa['color_hex_triplet2'] = dfa['color_hex_triplet2'
                                       ].apply(lambda x:
                                               mpl.colors.to_rgba(x))

        palette = dict(zip(dfa.acronym, dfa.color_hex_triplet2))

        #add layer colors
        bc = ['b', 'g', 'r', 'c', 'm', 'y', 'brown', 'pink']
        for i in range(7):
            palette[str(i)] = bc[i]
        
        palette['thal'] = 'k'    
        r = {}
        r['dfa'] = dfa
        r['palette'] = palette    
        np.save(pth_dmna, r, allow_pickle=True)   

    r = np.load(pth_dmna, allow_pickle=True).flat[0]
    return r['dfa'], r['palette']  



def cosine_sim(v0, v1):
    # cosine similarity 
    return np.inner(v0,v1)/ (norm(v0) * norm(v1))


def get_uuids_raster_types(regs=None):
    r1 = np.load(Path(pth_dmn,'cross_val_test.npy'), allow_pickle=True).flat[0]
    uuids_included, pids_included, cell_raster_type = [], [], {}

    if regs!=None:
        inclusion = [reg in regs for reg in r1['acs']]
        uuids_included.append(r1['uuids'][inclusion])
        pids_included.append(r1['pid'][inclusion])
        cell_raster_type['vis'] = r1['uuids'][inclusion]
        uuids_included=np.concatenate(uuids_included)
        pids_included=np.concatenate(pids_included)
        pids_included=list(set(pids_included))
    else:
        raster_types = {
            'stim_res': {'start': [6000], 'end': [6700]},
            'integrator': {'start':[4500], 'end':[6000]},
            'move_init': {'start': [750], 'end':[4500]},
            'movement': {'start':[43720], 'end':[44880]}
            
        }
        
        for raster_type in raster_types:
            start = raster_types[raster_type]['start'][0]
            end = raster_types[raster_type]['end'][0]
            isort = r1['isort']
            raster_type_ids=r1['uuids'][isort][start:end]
            pids = r1['pid'][isort][start:end]
            uuids_included.append(raster_type_ids)
            pids_included.append(pids)
            # cell_raster_type.append(np.repeat(raster_type, len(raster_type_ids)))
            cell_raster_type[raster_type]=raster_type_ids
            
        uuids_included=np.concatenate(uuids_included)
        pids_included=np.concatenate(pids_included)
        pids_included=list(set(pids_included))

        cell_raster_type['vis_subset0'] = np.array(['d7c0cc75-2e0a-4795-962b-3d23be419827',
            '435f743f-746e-47cb-917c-87afe117b784',
            'fbe5e813-c293-44f2-b02e-45c571f1efb4',
            '08e0c991-beb3-40d4-a1d9-3e13bd247857',
            '1e3686b3-9451-4e4e-a158-2a3d8243cd0f',
            'eba0cd14-69fd-48c1-a01e-769efd5d6b5f',
            '519986ee-f9d4-4d86-83c4-62ec9283a38e',
            'dd405576-00d7-456e-9b59-600e36ec3ac0',
            '00519868-02cd-4b59-ae85-01a55d8cc007',
            '4f13563f-f8a3-46d7-9077-573b1844f95b',
            '75584b6d-0ee9-49fb-b9b9-38391bf9ebe8',
            '1cf86729-6cd0-43c8-a69a-445db285056d',
            '33ac959e-ebc5-4a5d-a50c-05fec22520b8',
            '53a02b44-bfc2-409e-a362-9868d55c195e',
            'd518040d-34fd-4732-9b34-b711962b5564',
            '9030d12a-60ae-4e3f-865d-d2437df9f6f2',
            '1e9e67e0-41d1-44f6-922c-d749ec0cc335',
            '20cd97c8-5e20-42a8-9983-d5a21a82cafd',
            '0178d3d7-580e-4e0e-8904-0dd2ec84693e',
            '66522c3a-5dde-4471-ba85-4444a5274acf',
            'e37280d8-9fd2-444c-a81f-77fc668db189',
            '2b40e9df-de96-4110-bfcc-3b2472a8fa1c',
            'f22be33d-9642-417b-be65-df194cde8b94',
            '4bc0f255-6b82-41f8-b368-2c04f06cd875',
            '89db33b1-733a-4838-b2dc-38a5002bfbd5',
            '38cf1307-cec9-404b-a673-4719f3edab6e',
            '1698743c-d272-49a4-a125-3fd652bd8f4e',
            '448cdc83-51bc-4c56-8268-04d1435823e1',
            'acd9bb5e-208a-403a-830b-6b145799d80a',
            '7a614328-1953-487b-8e99-77b66f62fd70',
            '463037b6-dbdc-49b1-a8c7-0bfabc5b1323',
            'a3cc624d-12e5-4514-b5f0-cd7c17330fef',
            '35d5891d-0d32-479a-9606-a2eb8b698002',
            '7c49b3d8-7bb2-4aad-96d0-73e6dcd6e0df',
            'd1ae60e4-07d3-4758-89bd-5234f89bf70d',
            '0da23641-ecda-4fc3-81f7-fd98a0f41d04',
            '2c31329e-069a-4721-981f-0d281f9c7763',
            '05949c00-64cb-459b-b42a-d678a7c6e8e9',
            'c874ba37-a478-4af6-b46b-9a211df0b2c8',
            '5f826d20-24a7-4763-8f67-ea68cf1a4ab0',
            'c96c3866-e76f-4139-9123-8bd141c186a7',
            '5f6faf5b-df35-4577-b4f5-088b1ba1dfe3',
            '4f5580b0-f60f-48cd-87cf-5bd7bca2c63f',
            'd9775331-4c55-4353-811d-bc7037357e72',
            '630912af-383c-4060-9812-49fe18944d0b',
            'e31f3b0c-1241-4aba-a91c-227e9bdf276f',
            'e9615565-63b9-428a-a786-602fec05b1a5',
            'fc62258a-89a0-475e-810a-43f4cc27d597',
            'bf1ff5ee-8f45-4c64-aeb4-4884213bcad9',
            '8096bdb8-31de-4575-9d83-78c29d7d0ad5',
            '7f154dd7-a754-4c0d-809a-743a7debf3fb',
            '29f2b650-f52b-49d4-8b5c-88a019e735b5',
            'eb898932-eb64-432b-9273-6598b2b62a0f',
            'bc895acf-14f9-4c2c-b4b6-7276b9d0f310',
            '599ac6d0-c2aa-4edd-b5cc-4d8158da1905',
            '009fb36b-d18a-4cf0-b8d8-cbb6e305f900',
            'e1e1d982-4d55-406e-86fe-d9dcf0a5a9ca',
            '772d5b7b-c32e-4cd1-9bb4-cbe477422733',
            '9caa1217-1b04-4917-ae55-dc1feb13e752',
            '12508ea8-af21-450a-9c54-0cfc8941d075',
            'c7e613ca-c058-442d-8766-6a0495390330',
            '0c102670-aff5-44d6-aa74-241c42972e1a',
            '824d63ca-a3a5-49c2-8d48-a0624d37c3a0',
            '89d709ba-1e52-4c69-96d8-5e48041adc97',
            'f803611d-0f78-45eb-a9d3-7af6f4f755c5',
            '1105d2ec-91d0-4d65-9607-916024de60bd',
            'd769b55d-fcdc-4163-ae60-5b00b1f1343f',
            '7106022b-a9bc-4ff5-ab9a-f6329d5ddf91',
            '499b9d06-ce85-43b6-bb16-508835af8888',
            '3cf66edf-d8b3-451d-bbb0-43e8dbe74976',
            '2d8d2a7c-0bf3-440c-a4cf-bf7cbfdffbf2',
            '40a9378e-352c-4ff5-8132-12fb741e24bd',
            '6d722520-0085-4e76-acf3-65cf53365b0d',
            '36045c15-2a0e-4406-b259-33fa412a7624',
            '2a90e5c0-d84a-42ff-b3c4-1f4d554e4a0b',
            'fd1f4789-4e6b-4e8b-ad25-b43aa96f5e7c',
            '33ee4083-94d0-4f9d-bc6f-931685f28d95',
            'ea1e6629-a977-4c62-bd87-e91b8409203f',
            'abe3c3fd-e3c2-43ac-b898-4a8dbcf83d5b'], dtype=object)
        cell_raster_type['vis_subset1'] = np.array(['c96c3866-e76f-4139-9123-8bd141c186a7',
            '3468ac20-4418-4766-ac58-5b16f50007f3',
            '435f743f-746e-47cb-917c-87afe117b784',
            '5f6faf5b-df35-4577-b4f5-088b1ba1dfe3',
            '47743328-5519-4966-a851-46ce19e4ed98',
            'b3a949e0-1fbe-43cd-8907-d90b61677a81',
            '4f5580b0-f60f-48cd-87cf-5bd7bca2c63f',
            'd9775331-4c55-4353-811d-bc7037357e72',
            '08e0c991-beb3-40d4-a1d9-3e13bd247857',
            'e31f3b0c-1241-4aba-a91c-227e9bdf276f',
            '1e3686b3-9451-4e4e-a158-2a3d8243cd0f',
            '519986ee-f9d4-4d86-83c4-62ec9283a38e',
            'd937857f-2997-495f-b168-b07e9c3ffeaa',
            '07d4ddcb-4a6f-4dd2-8485-52857fb2f7e3',
            '33f67715-c790-4dd7-b23a-792163bd63c2',
            '29f2b650-f52b-49d4-8b5c-88a019e735b5',
            '33ac959e-ebc5-4a5d-a50c-05fec22520b8',
            'dd8c158a-e835-4ed1-ac01-9b5ce00dee24',
            '009fb36b-d18a-4cf0-b8d8-cbb6e305f900',
            '5c3c3104-3abb-40e0-a55d-39323ca36d66',
            'd518040d-34fd-4732-9b34-b711962b5564',
            '9030d12a-60ae-4e3f-865d-d2437df9f6f2',
            'ece2a2e4-bae6-4725-96b8-22e0e5d06f04',
            'e5895cc5-1b2e-4e2c-b53a-4bca44607464',
            '20cd97c8-5e20-42a8-9983-d5a21a82cafd',
            '0178d3d7-580e-4e0e-8904-0dd2ec84693e',
            '12508ea8-af21-450a-9c54-0cfc8941d075',
            'c7e613ca-c058-442d-8766-6a0495390330',
            '6fea8068-f573-4aef-b577-13a03010b9a6',
            '96a2f983-2c2c-44d8-b78a-41ba19610583',
            '4bc0f255-6b82-41f8-b368-2c04f06cd875',
            '824d63ca-a3a5-49c2-8d48-a0624d37c3a0',
            'f22be33d-9642-417b-be65-df194cde8b94',
            '38cf1307-cec9-404b-a673-4719f3edab6e',
            '89d709ba-1e52-4c69-96d8-5e48041adc97',
            'c8f6bdf8-1f67-4bab-b4da-fd0f5a8fcb59',
            '1105d2ec-91d0-4d65-9607-916024de60bd',
            '11fce5e3-f031-4516-9eef-88a0e2409b53',
            'd769b55d-fcdc-4163-ae60-5b00b1f1343f',
            'f3ce1a14-11e9-4096-a847-4f3e2704990f',
            'cc01d3a9-4edd-460b-98da-f6edcb1a7eca',
            '4684fea7-487e-4e05-b1f8-c7e3e2f47198',
            '78ec6dfe-9039-4ce2-a109-5a5e1f8e56ca',
            '2d8d2a7c-0bf3-440c-a4cf-bf7cbfdffbf2',
            'c74ea639-0eac-49c0-bfd8-1b14c6c6b95a',
            '0bfa2da6-ea58-40d7-9191-565c91277727',
            '35d5891d-0d32-479a-9606-a2eb8b698002',
            '7c49b3d8-7bb2-4aad-96d0-73e6dcd6e0df',
            '1686de96-e776-44a8-9438-7408173559f0',
            '6d722520-0085-4e76-acf3-65cf53365b0d',
            'd1ae60e4-07d3-4758-89bd-5234f89bf70d',
            '4cc1ffe8-7ef1-40ad-b0e5-f2ca9cec1b08',
            '36045c15-2a0e-4406-b259-33fa412a7624',
            '05949c00-64cb-459b-b42a-d678a7c6e8e9',
            'f8a7afcf-6c89-4792-bc85-d0fca399da77',
            '2a90e5c0-d84a-42ff-b3c4-1f4d554e4a0b',
            'fd1f4789-4e6b-4e8b-ad25-b43aa96f5e7c',
            '2c31329e-069a-4721-981f-0d281f9c7763',
            'fc0bbe17-dedc-470e-84b3-d76342b0f377',
            '5f826d20-24a7-4763-8f67-ea68cf1a4ab0',
            'ea1e6629-a977-4c62-bd87-e91b8409203f',
            '37c1e062-f7e9-4c50-9210-f35ed9452cef'], dtype=object)
        cell_raster_type['topstim_subset'] = np.array(['35c03054-f8bd-47a7-9dbf-7119ab5992ba',
            '54873530-6f5f-4609-a4ef-e8b5a10187e0',
            '4e54a65d-53d2-4df5-b202-a857c5926f6f',
            '49de0cb6-eee5-4eab-8c3c-a591b1f7456d',
            '5c3c3104-3abb-40e0-a55d-39323ca36d66',
            'ba7e282a-5459-416e-abbe-a0655b30e19e',
            'f22be33d-9642-417b-be65-df194cde8b94',
            '7518f0f5-a8d1-4998-b8a5-dfe3c59664e0',
            'bd67bcd9-e17e-41cf-831a-5e86ed0b8af4',
            'c9045bc4-9610-41fc-98c6-4464431a4eb9',
            '10a8342f-15f9-4c88-a3e7-f68b6488296a',
            'f6958a9e-7b4e-4476-8603-4b89b39dfc59',
            'a0fc29d7-5e2b-4fe2-854c-d33924c12d9c',
            '2954ca22-6763-4bee-b9e6-6a57e2371df9',
            '35d5891d-0d32-479a-9606-a2eb8b698002',
            '899d1971-8493-4f34-8abc-6e18807f9fa7',
            '698f1298-6896-4837-ab66-1682e12bd280',
            '79a45714-c418-4c27-9ecf-c86733908a97',
            '0ce07ff6-d479-4bac-85a7-0b8d9e15d1e2',
            '419f387d-e08b-41ff-b187-bdbfa961ec7b',
            '98f9fc2f-d9b8-43e8-9a8f-17006234e445',
            '2c5df4a7-6a28-4fdf-9f61-776bdbb278c7',
            '9749198e-7f8c-423e-bfcf-7001226ec88b',
            'a7c3680a-5bf5-4c02-ae84-c808185122d7',
            '0cff8603-60d9-4a4c-90cf-eef1ec5b016e',
            '2b94a188-af0f-4dd9-88a4-e0be6cc67c00',
            '314ce3a1-6590-4630-a315-4aa3dd318ea0',
            '21a41b80-3ee5-456b-bf87-5ed5e7b0525a',
            '8ca37749-5b5f-4a9a-9d32-030f15f0a536',
            '7796a6fd-fb12-42c0-beb4-c1ec4cdbea5a',
            'ab55b09d-1c7a-46bb-9dae-f290fd5cbb75',
            'd1ae60e4-07d3-4758-89bd-5234f89bf70d',
            '0293361a-c8db-4e55-a5c7-949e596bebee',
            '2c31329e-069a-4721-981f-0d281f9c7763',
            '825e9d30-6d23-4445-b19f-92110f6a8016',
            '77c27d40-d778-4012-9877-c54a779b5f95',
            '66301612-9509-49a6-9b7e-4c7240504c6f',
            '9d9d2895-b2f5-4bfd-8496-ddd11b7e9b28',
            '8cd4f58a-8690-4aed-897a-3a36df71be43',
            'bceb6264-b9da-4864-b665-2c4f4e0ff559',
            '7aab640f-125f-4e70-b6d7-da75fff6231c',
            '1e861e98-2e7b-4b1b-b643-8f5c1c229c64',
            '633af22f-a58c-4b62-a90a-3264953310b1',
            '692f62d7-fa0b-481f-85e8-82a19706ade7',
            '3c98334a-741e-4d0f-aeab-137a7a0f251c',
            '77e2daa1-d5de-4c29-86b5-4f250b833c9a',
            '221825e5-aa14-489a-a8af-6a119455fb8c',
            'e45dbbcc-7bab-4533-8bb8-ae83b928b2dc',
            'ec82f8e0-d28d-40ed-9152-35e9742031ba',
            '70dd2e95-5b55-4d02-90f9-ae497afdc2e0',
            '77e27e06-2d44-4543-a965-4c006ec057aa',
            'dc2926e5-df36-449b-8f53-ae48ba3c17f4',
            'c39264cd-c4b4-4b07-9a20-e1bf90ae690b',
            'ebaaed73-d993-4fae-89db-8cc209f1ab69',
            '4028de29-27d9-4a88-8878-a3e971e4a3ec',
            '148b69fa-354d-4a98-968f-3aac0c4f32fb'], dtype=object)
        
    return uuids_included, pids_included, cell_raster_type



def get_reg_dist(rerun=False, algo='umap_z', control=False,
                  mapping='Beryl', vers='concat', shuffling=False):

    if control!=False:
        if shuffling:
            pth_ = Path(one.cache_dir, 'dmn', 
                f'{algo}_{mapping}_{vers}_control{control}_smooth_shuffled.npy')
        else:
            pth_ = Path(one.cache_dir, 'dmn', 
                f'{algo}_{mapping}_{vers}_control{control}_smooth.npy')
    else:
        if shuffling:
            pth_ = Path(one.cache_dir, 'dmn', 
                f'{algo}_{mapping}_{vers}_smooth_shuffled.npy')
        else:
            pth_ = Path(one.cache_dir, 'dmn', 
                f'{algo}_{mapping}_{vers}_smooth.npy')
        
    if (not pth_.is_file() or rerun):
        res, regs = smooth_dist(algo=algo, mapping=mapping, vers=vers, 
                                shuffling=shuffling, control=control, dendro=False)    
        d = {'res': res, 'regs' : regs}
        np.save(pth_, d, allow_pickle=True)
    else:
        d = np.load(pth_, allow_pickle=True).flat[0]        
        
    return d     



def clustering_on_peth_data(r, algo='concat_z', k=2, clustering='kmeans', min_s=10, eps=0.5, random_state=0):

    res = r[algo]
    
    if clustering=='hierarchy': # Order the matrix using hierarchical clustering            
        linkage_matrix = hierarchy.linkage(res)
        #ordered_indices = hierarchy.leaves_list(linkage_matrix)            
        clusters = hierarchy.fcluster(linkage_matrix, k, criterion='maxclust')

    elif clustering == 'spectralco': # Order the matrix using spectral co-clustering
        clustering_result = SpectralCoclustering(n_clusters=k, random_state=random_state).fit(res)
        clusters = clustering_result.row_labels_

    elif clustering == 'spectralbi': #spectral bi-clustering
        clustering_result = SpectralBiclustering(n_clusters=k, random_state=random_state).fit(res)
        clusters = clustering_result.row_labels_

    elif clustering == 'dbscan':
        clustering_result = DBSCAN(eps=eps, min_samples=min_s, metric='cosine').fit(res)
        clusters = clustering_result.labels_
        
    elif clustering == 'birch': #birch clustering
        clustering_result = Birch(n_clusters=k).fit(res)
        clusters = clustering_result.labels_
        
    elif clustering == 'mbkmeans': 
        clustering_result = MiniBatchKMeans(n_clusters=k, batch_size=20, random_state=random_state).fit(res)
        clusters = clustering_result.labels_

    elif clustering == 'kmeans': 
        clustering_result = KMeans(n_clusters=k, random_state=random_state).fit(res)
        clusters = clustering_result.labels_
        r['centers'] = clustering_result.cluster_centers_
        print(clustering_result.inertia_)

    else:
        print('what clustering method?')
        return
    
    return r, clusters


def regional_group(mapping, algo, vers='concat', norm_=False, min_s=10, eps=0.5,
                   run_umap=False, run_pca=False, n_neighbors=10, d=0.2, ncomp=2,
                   nclus = 13, random_seed = 0):

    '''
    find group labels for all cells
    mapping: [Allen, Beryl, Cosmos, layers, clusters, clusters_xyz] or some clustering algorithm name
    algo: concat_z(original high-dim data) or umap_z(2d dim-reduced data)
    '''

    if vers=='concat0':
        r = np.load('/Users/ariliu/Downloads/concat_ephysTrue.npy',
                 allow_pickle=True).flat[0]
    else:
        r = np.load(Path(pth_dmn, f'{vers}_norm{norm_}.npy'),
                 allow_pickle=True).flat[0]
                 
                              
    if run_umap==True:
        r['umap_z'] = umap.UMAP(random_state=random_seed, n_components=ncomp, min_dist=d, 
                             n_neighbors=n_neighbors).fit_transform(r['concat_z'])

    if run_pca==True:
        r['pca_z'] = PCA(n_components=ncomp).fit_transform(r['concat_z'])
                   

    # add point names to dict
    r['nums'] = range(len(r[algo][:,0]))

    if mapping in ['kmeans', 'mbkmeans', 'dbscan', 'hierarchy', 'birch', 'spectralbi', 'spectralco']:
        # use the corresponding clustering method on full data or dim-reduced 2d data
        r, clusters = clustering_on_peth_data(r, algo=algo, k=nclus, 
                                              clustering=mapping, 
                                              min_s=min_s, eps=eps, 
                                              random_state=random_seed) 
               
        cmap = mpl.cm.get_cmap('Spectral')
        cols = cmap(clusters/nclus)
        acs = clusters
        regs = np.unique(clusters)
            
        color_map = dict(zip(list(acs), list(cols)))
        r['els'] = [Line2D([0], [0], color=color_map[reg], 
                    lw=4, label=f'{reg + 1}')
                    for reg in regs]
        
        # get average point and color per region
        av = {clus: [np.mean(r[algo][clusters == clus], axis=0), 
                    cmap(clus/nclus)] 
              for clus in range(1,nclus+1)}
              

    elif mapping == 'hdbscan':
        mcs = 10
        clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs)    
        clusterer.fit(r[algo])
        labels = clusterer.labels_
        unique_labels = np.unique(labels)
        mapping = {old_label: new_label 
                      for new_label, old_label in 
                      enumerate(unique_labels)}
        clusters = np.array([mapping[label] for label in labels])

        cmap = mpl.cm.get_cmap('Spectral')
        cols = cmap(clusters/len(unique_labels))
        acs = clusters
        # get average point and color per region
        av = {clus: [np.mean(r[algo][clusters == clus], axis=0), 
                    cols] 
              for clus in range(1,len(unique_labels)+1)} 
        


    elif mapping == 'layers':       
    
        acs = np.array(br.id2acronym(r['ids'], 
                                     mapping='Allen'))
        
        regs0 = Counter(acs)
                                     
        # get regs with number at and of acronym
        regs = [reg for reg in regs0 
                if reg[-1].isdigit()]
        
        for reg in regs:        
            acs[acs == reg] = reg[-1]       
        
        # extra class of thalamic (and hypothalamic) regions 
        names = dict(zip(regs0,[get_name(reg) for reg in regs0]))
        thal = {x:names[x] for x in names if 'thala' in names[x]}
                                          
        for reg in thal: 
            acs[acs == reg] = 'thal'       
        
        mask = np.array([(x.isdigit() or x == 'thal') for x in acs])
        acs[~mask] = '0'
        
        remove_0 = True
        
        if remove_0:
            # also remove layer 6, as there are only 20 neurons 
            zeros = np.arange(len(acs))[
                        np.bitwise_or(acs == '0', acs == '6')]
            for key in r:
                if len(r[key]) == len(acs):
                    r[key] = np.delete(r[key], zeros, axis=0)
                       
            acs = np.delete(acs, zeros)        
        
        _,pa = get_allen_info()
        cols = [pa[reg] for reg in acs]
        regs = Counter(acs)      
        r['els'] = [Line2D([0], [0], color=pa[reg], 
               lw=4, label=f'{reg} {regs[reg]}')
               for reg in regs]
               
        # get average points and color per region
        av = {reg: [np.mean(r[algo][acs == reg], axis=0), pa[reg]] 
              for reg in regs}
               

    elif mapping == 'clusters_xyz':
   
        # use clusters from hierarchical clustering to color
        nclus = 1000
        clusters = fcluster(r['linked_xyz'], t=nclus, 
                            criterion='maxclust')
        cmap = mpl.cm.get_cmap('Spectral')
        cols = cmap(clusters/nclus)
        acs = clusters   
        # get average points per region
        av = {reg: [np.mean(r[algo][clusters == clus], axis=0), 
                    cmap(clus/nclus)] 
              for clus in range(1,nclus+1)}      

    else:
        acs = np.array(br.id2acronym(r['ids'], 
                                     mapping=mapping))
                                     
#        # remove void and root
#        zeros = np.arange(len(acs))[np.bitwise_or(acs == 'root',
#                                                  acs == 'void')]
#        for key in r:
#            if len(r[key]) == len(acs):
#                r[key] = np.delete(r[key], zeros, axis=0)
#                   
#        acs = np.delete(acs, zeros)          
        
                                                              
        _,pa = get_allen_info()
        cols = [pa[reg] for reg in acs]
        
        # get average points and color per region
        regs = Counter(acs)  
        av = {reg: [np.mean(r[algo][acs == reg], axis=0), pa[reg]] 
              for reg in regs}
              

    if 'end' in r['len']:
        del r['len']['end']
              
    r['acs'] = acs
    r['cols'] = cols
    r['av'] = av

    return r



def smooth_dist(algo='umap_z', mapping='Beryl', show_imgs=False, shuffling=False,
                norm_=True, dendro=False, nmin=30, vers='concat', control_list=[False]):

    '''
    Smooth 2D pointclouds, show per class.
    norm_: normalize smoothed image by max brightness
    control_list: list of control versions (e.g. [False, 'ver0_0', ...])
    '''

    feat = 'concat_z' if algo[-1] == 'z' else 'concat'
    fontsize = 12

    if not shuffling:
        if len(control_list) != 1:
            raise ValueError("When shuffling=False, control_list must contain exactly one version.")
        control = control_list[0]
        if control is not False:
            r = regional_group(mapping, algo, vers=vers, norm_='False_control' + control)
        else:
            r = regional_group(mapping, algo, vers=vers)

        return _process_single_r(r, algo, mapping, norm_, dendro, nmin, feat, show_imgs)

    # shuffling == True: align & shuffle across all control versions
    r_dict = {}
    uuid_sets = {}
    uuid_to_index = {}

    # Load all control versions
    for control in control_list:
        if control is not False:
            r = regional_group(mapping, algo, vers=vers, norm_='False_control' + control)
        else:
            r = regional_group(mapping, algo, vers=vers)
        r_dict[control] = r
        uuid_sets[control] = set(r['uuids'])
        uuid_to_index[control] = {uuid: i for i, uuid in enumerate(r['uuids'])}

    # Find shared uuids across all versions
    shared_uuids = sorted(set.intersection(*uuid_sets.values()))

    # Build shuffled acs assignment from base control
    base_control = control_list[0]
    base_r = r_dict[base_control]
    base_idx = uuid_to_index[base_control]
    original_acs = [base_r['acs'][base_idx[uuid]] for uuid in shared_uuids]
    shuffled_acs = original_acs[:]
    random.shuffle(shuffled_acs)
    uuid_to_shuffled_acs = dict(zip(shared_uuids, shuffled_acs))

    # Process each aligned version
    results = {}
    for control in control_list:
        r = r_dict[control]
        idx_map = uuid_to_index[control]
        shared_indices = [idx_map[uuid] for uuid in shared_uuids]

        # Restrict all per-cell fields
        n_cells = len(r['uuids'])
        for key in r:
            val = r[key]
            if isinstance(val, list) and len(val) == n_cells:
                r[key] = [val[i] for i in shared_indices]
            elif isinstance(val, np.ndarray) and val.shape[0] == n_cells:
                r[key] = val[shared_indices]

        r['acs'] = [uuid_to_shuffled_acs[uuid] for uuid in shared_uuids]
        # r['uuids'] = shared_uuids

        # Process and collect results, skip plotting
        res, regs = _process_single_r(r, algo, mapping, norm_, dendro, nmin, feat, show_imgs=False)
        results[control] = (res, regs)

    return results

def _process_single_r(r, algo, mapping, norm_, dendro, nmin, feat, show_imgs):
    x_min = np.floor(np.min(r[algo][:,0]))
    x_max = np.ceil(np.max(r[algo][:,0]))
    y_min = np.floor(np.min(r[algo][:,1]))
    y_max = np.ceil(np.max(r[algo][:,1]))

    imgs = {}
    xys = {}

    regs00 = Counter(r['acs'])
    regcol = {reg: np.array(r['cols'])[np.array(r['acs']) == reg][0] for reg in regs00}    

    if mapping == 'Beryl':
        p = (Path(iblatlas.__file__).parent / 'beryl.npy')
        regsord = dict(zip(br.id2acronym(np.load(p), mapping='Beryl'),
                           br.id2acronym(np.load(p), mapping='Cosmos')))
        regs = [reg for reg in regsord if reg in regs00 and regs00[reg] > nmin]
    else:
        regs = [reg for reg in regs00 if regs00[reg] > nmin]

    for reg in regs:
        x = (r[algo][np.array(r['acs'])==reg,0] - x_min)/ (x_max - x_min)    
        y = (r[algo][np.array(r['acs'])==reg,1] - y_min)/ (y_max - y_min)

        data = np.array([x,y]).T         
        inds = (data * 255).astype('uint')

        img = np.zeros((256,256))
        for i in np.arange(data.shape[0]):
            img[inds[i,0], inds[i,1]] += 1

        imsm = ndi.gaussian_filter(img.T, (10,10))
        imgs[reg] = imsm/np.max(imsm) if norm_ else imsm
        xys[reg] = [x,y]

    res = np.zeros((len(regs),len(regs)))
    for i, reg_i in enumerate(imgs):
        for j, reg_j in enumerate(imgs):
            v0 = imgs[reg_i].flatten()
            v1 = imgs[reg_j].flatten()
            res[i,j] = cosine_sim(v0, v1)

    if dendro:
        res = np.round(res, decimals=8)
        cres = squareform(1 - res)
        linkage_matrix = hierarchy.linkage(cres)
        ordered_indices = hierarchy.leaves_list(linkage_matrix)
        res = res[:, ordered_indices][ordered_indices, :]
        regs = np.array(regs)[ordered_indices]

    return res, regs


def compare_distance_metrics_scatter(vers, nclus=7, nd=2, rerun=False):
    # load data
    wass = np.load(Path(pth_dmn, f'wasserstein_matrix_{nclus}_{vers}_nd{nd}.npy'), allow_pickle=True).flat[0]
    wass_d = wass['res']
    wass_regs = wass['regs']
    umap = get_reg_dist(algo='umap_z', vers=vers, rerun=rerun)
    umap_d = umap['res']
    umap_regs = umap['regs']

    # reorder umap similarity matrix w/ wasserstein matrix entries' ordering
    ordered_umap_indices = [list(umap_regs).index(reg) for reg in wass_regs]
    umap_regs = np.array(umap_regs)[ordered_umap_indices]
    umap_d = umap_d[:, ordered_umap_indices][ordered_umap_indices, :]

    umap_d_flat = umap_d.flatten()
    wass_d_flat = wass_d.flatten()

    corp,pp = pearsonr(umap_d_flat, wass_d_flat)
    cors,ps = spearmanr(umap_d_flat, wass_d_flat)

    print(corp, pp, cors, ps)

    plt.scatter(umap_d_flat, wass_d_flat, s=2)
    plt.xlabel('Umap Cosine Similarity')
    plt.ylabel('Wasserstein Distance')
    plt.title(f'{vers}\n pearsonr: {corp}, spearmanr: {cors}')
    plt.savefig(Path(pth_dmn.parent, 'figs', f'compare_metrics_{vers}.pdf'))
    plt.show()


def clustering_on_connectivity_matrix(res, regs, k=4, metric='umap_z', clustering='hierarchy', 
                                      random_state=0, resl=1.01, tau=0.01):
    if clustering=='hierarchy': # Order the matrix using hierarchical clustering
        
        if metric in ['umap_z', 'pca_z']: # convert similarity scores to distances
            res0 = np.copy(res)
            res = np.amax(res) - res
            
        # get condensed distance matrix (upper-triangular part of distance matrix) as input
        cres = squareform(res)
        linkage_matrix = hierarchy.linkage(cres)
        ordered_indices = hierarchy.leaves_list(linkage_matrix)            
        regs_r = np.array(regs)[ordered_indices]
        regs_c = regs_r
        
        if metric in ['umap_z', 'pca_z']: # convert distances back to similarity scores
            res = res0
        elif metric == 'wass': # convert wasserstein distance to similarity for plotting
            res = (np.amax(res) - res)/np.amax(res)
            
        res = res[:, ordered_indices][ordered_indices, :]
        cluster_info = hierarchy.fcluster(linkage_matrix, 10, criterion='maxclust')

    elif clustering == 'spectral_cutoff': # spectral clustering but with a cutoff tau
        embedding = SpectralEmbedding(n_components=k, affinity='precomputed')
        eigenvectors = embedding.fit_transform(res)
        projection_magnitudes = np.linalg.norm(eigenvectors, axis=1)
        strong_points = projection_magnitudes >= tau
        V_filtered = eigenvectors[strong_points]
        cluster_labels_filtered = KMeans(n_clusters=k, random_state=random_state).fit_predict(V_filtered)
        
        labels = np.full(res.shape[0], -1)  # Initialize all as unassigned (-1)
        labels[strong_points] = cluster_labels_filtered
        cluster_info = labels[strong_points]
        ordered_indices = np.argsort(labels[strong_points])
        regs_r = np.array(regs)[ordered_indices]
        regs_c = regs_r
        res = res[:, ordered_indices][ordered_indices, :]         

    elif clustering == 'ICA_cutoff': # ICA component projection with a cutoff tau
        # convert similarity to distance
        res0 = np.copy(res)
        res = np.amax(res) - res
        # apply ICA and project data onto components
        ica = FastICA(n_components=k, random_state=random_state)
        ica_features = ica.fit_transform(res)
        # get networks that can overlap (unlike in other cases which are disjoint clusters!)
        cluster_info = (np.abs(ica_features) >= tau).astype(int)        
        regs_r = regs
        regs_c = regs_r

    elif clustering == 'spectralco': # Order the matrix using spectral co-clustering
        clustering_result = SpectralCoclustering(n_clusters=k, random_state=random_state).fit(res)
        ordered_indices_r = np.argsort(clustering_result.row_labels_)
        ordered_indices_c = np.argsort(clustering_result.column_labels_)
        regs_r = np.array(regs)[ordered_indices_r]
        regs_c = np.array(regs)[ordered_indices_c]
        res = res[ordered_indices_r]
        res = res[:, ordered_indices_c]
        cluster_info = clustering_result.row_labels_

    elif clustering == 'spectralbi': #spectral bi-clustering
        clustering_result = SpectralBiclustering(n_clusters=k, random_state=random_state).fit(res)
        ordered_indices_r = np.argsort(clustering_result.row_labels_)
        ordered_indices_c = np.argsort(clustering_result.column_labels_)
        regs_r = np.array(regs)[ordered_indices_r]
        regs_c = np.array(regs)[ordered_indices_c]
        res = res[ordered_indices_r]
        res = res[:, ordered_indices_c]
        cluster_info = clustering_result.row_labels_
        
    elif clustering == 'birch': #birch clustering
        clustering_result = Birch(n_clusters=k).fit(res)
        cluster_info = clustering_result.labels_
        ordered_indices = np.argsort(cluster_info)
        regs_r = np.array(regs)[ordered_indices]
        regs_c = regs_r
        res = res[:, ordered_indices][ordered_indices, :] 
        
    elif clustering == 'mbkmeans': 
        clustering_result = MiniBatchKMeans(n_clusters=k, batch_size=20, random_state=random_state).fit(res)
        cluster_info = clustering_result.labels_
        ordered_indices = np.argsort(cluster_info)
        regs_r = np.array(regs)[ordered_indices]
        regs_c = regs_r
        res = res[:, ordered_indices][ordered_indices, :]

    elif clustering == 'kmeans': 
        clustering_result = KMeans(n_clusters=k, random_state=random_state).fit(res)
        cluster_info = clustering_result.labels_
        ordered_indices = np.argsort(cluster_info)
        regs_r = np.array(regs)[ordered_indices]
        regs_c = regs_r
        res = res[:, ordered_indices][ordered_indices, :]

    elif clustering == 'louvain':
        adjacency = sparse.csr_matrix(res)
        louvain = Louvain(random_state=random_state, resolution=resl)
        cluster_info = louvain.fit_predict(adjacency)
        ordered_indices=np.argsort(cluster_info)
        regs_r = np.array(regs)[ordered_indices]
        regs_c = regs_r
        res = res[:, ordered_indices][ordered_indices, :]

    elif clustering == 'leiden':
        adjacency = sparse.csr_matrix(res)
        leiden = Leiden(random_state=random_state, resolution=resl)
        cluster_info = leiden.fit_predict(adjacency)
        ordered_indices=np.argsort(cluster_info)
        regs_r = np.array(regs)[ordered_indices]
        regs_c = regs_r
        res = res[:, ordered_indices][ordered_indices, :]

    elif clustering == 'kcenters':
        adjacency = sparse.csr_matrix(res)
        kcenters = KCenters(n_clusters=k)
        cluster_info = kcenters.fit_predict(adjacency)
        ordered_indices=np.argsort(cluster_info)
        regs_r = np.array(regs)[ordered_indices]
        regs_c = regs_r
        res = res[:, ordered_indices][ordered_indices, :]

    else:
        print('what clustering method?')
        return

    return res, regs_r, regs_c, cluster_info



def get_reproducibility_score(vers, clustering, k=None, resl=None):
    d = get_reg_dist(algo='umap_z', vers=vers)
    res = d['res']
    regs = d['regs']

    ARI, AMI = [], []
    for i in range(20):
        _, _, _, cluster_info0 = clustering_on_connectivity_matrix(
            res, regs, k=k, clustering=clustering, random_state=i, resl=resl)
        _, _, _, cluster_info1 = clustering_on_connectivity_matrix(
            res, regs, k=k, clustering=clustering, random_state=457-i, resl=resl)
    
        ARI.append(adjusted_rand_score(cluster_info0, cluster_info1))
        AMI.append(adjusted_mutual_info_score(cluster_info0, cluster_info1))
    print('mean ARI', np.mean(ARI), 'mean AMI', np.mean(AMI))


def get_quiescence_resting_diff(metric='umap_z', diff='shifted'):
    dr = get_reg_dist(algo=metric, vers='resting')
    dq = get_reg_dist(algo=metric, vers='quiescence')
    d = {}

    # order regions by canonical list 
    p = (Path(iblatlas.__file__).parent / 'beryl.npy')
    regs_can = br.id2acronym(np.load(p), mapping='Beryl')
    regs = [reg for reg in regs_can if reg in dr['regs']]
    ordered_indices_r = [list(dr['regs']).index(reg) for reg in regs]
    ordered_indices_q = [list(dq['regs']).index(reg) for reg in regs]
    res_r = dr['res'][:, ordered_indices_r][ordered_indices_r, :]
    res_q = dq['res'][:, ordered_indices_q][ordered_indices_q, :]

    if diff == 'shifted':
        d['res'] = res_q-res_r - np.min(res_q-res_r)
    else:
        d['res'] = abs(res_q-res_r)
    d['regs'] = regs

    return d


def get_all_clustering_results(clustering, metric='umap_z', nclus=13, nd=2, 
                               k=None, resl=1.01, tau=0.01, rerun=False):

    '''   
    '''

    clusters, regs_ordered = {}, {}
    for vers in ['concat', 'resting', 'quiescence', 'quie-rest-diff-shifted', 'quie-rest-diff-abs', 
                 'pre-stim-prior', 'mistake', 'stim_all',
                 'stim_surp_con', 'stim_surp_incon', 'motor_init', 'fback1', 'fback0']:
        if metric=='wass':
            d = np.load(Path(pth_dmn, f'wasserstein_matrix_{nclus}_{vers}_nd{nd}.npy'), 
                        allow_pickle=True).flat[0]
        elif vers=='quie-rest-diff-shifted':
            d = get_quiescence_resting_diff(metric=metric, diff='shifted')
        elif vers=='quie-rest-diff-abs':
            d = get_quiescence_resting_diff(metric=metric, diff='abs')
        elif metric in ['umap_z', 'pca_z']:
            d = get_reg_dist(algo=metric, vers=vers, rerun=rerun)            
            
        res = d['res']
        regs = d['regs']
        _, _, regs, cluster_info = clustering_on_connectivity_matrix(
            res, regs, k=k, metric=metric, clustering=clustering, resl=resl, tau=tau)
        clusters[vers] = np.sort(cluster_info)
        regs_ordered[vers] = regs
        
    return clusters, regs_ordered


def find_cluster_overlap_between_paired_conditions(clusters, ids, vers0, vers1, threshold=2):
    overlaps=[]
    for i in set(clusters[vers0]):
        for j in set(clusters[vers1]):
            overlap = set(ids[vers0][clusters[vers0]==i]) & set(ids[vers1][clusters[vers1]==j])
            if len(overlap)>threshold:
                overlaps.append(overlap)

    return overlaps

def find_overlaps(x0, x1):
    overlaps=[]
    for i in range(len(x0)):
        for j in range(len(x1)):
            overlap = x0[i] & x1[j]
            if len(overlap) > 2:
                overlaps.append(overlap)
    return overlaps


def find_cluster_overlap_all_conditions(vers, clustering='louvain', k=5):
    '''
    vers includes all conditions to be considered
    '''

    clusters, regs_ordered = get_all_clustering_results(clustering, k=k)
    for n in range(int(len(vers)/2)+1):
        if n==0:
            x0 = find_cluster_overlap_between_paired_conditions(clusters, regs_ordered, 
                                                                vers[n], vers[len(vers)-n-1])
        else:
            x1 = find_cluster_overlap_between_paired_conditions(clusters, regs_ordered, 
                                                                vers[n], vers[len(vers)-n-1])
            x0 = find_overlaps(x0, x1)
    return x0


def get_all_highd_clustering_results(clustering, algo='concat_z', vers='concat', norm_=False, nclus=13,  
                                     min_s=10, eps=0.5, random_seed=0):

    '''
    get clustering results on high-d original feature vectors
    for all possible time windows/ conditions
    '''

    clusters, neurons_xyz, neurons_uuids, neurons_ids = {}, {}, {}, {}
    for vers in ['concat', 'resting', 'quiescence', 'pre-stim-prior', 'mistake', 'stim_all',
                 'stim_surp_con', 'stim_surp_incon', 'motor_init', 'fback1', 'fback0']:
        r = np.load(Path(pth_dmn, f'{vers}_norm{norm_}.npy'),
                 allow_pickle=True).flat[0]
        r, cluster_info = clustering_on_peth_data(r, algo=algo, k=nclus, 
                                              clustering=clustering, 
                                              min_s=min_s, eps=eps, 
                                              random_state=random_seed)
        
        clusters[vers] = cluster_info
        neurons_xyz[vers] = r['xyz']
        neurons_uuids[vers] = r['uuids']
        neurons_ids[vers] = r['ids']
        
    return clusters, neurons_xyz, neurons_uuids, neurons_ids


def find_highd_cluster_overlap_all_conditions(vers, clusters, neurons_uuids, threshold=10):
    '''
    vers includes all conditions to be considered
    '''

    for n in range(int(len(vers)/2)+1):
        if n==0:
            x0 = find_cluster_overlap_between_paired_conditions(clusters, neurons_uuids, 
                                                                vers[n], vers[len(vers)-n-1],
                                                                threshold=threshold)
        else:
            x1 = find_cluster_overlap_between_paired_conditions(clusters, neurons_uuids, 
                                                                vers[n], vers[len(vers)-n-1],
                                                                threshold=threshold)
            x0 = find_overlaps(x0, x1, threshold=threshold)
    return x0


def rgb_color_text(color, text):
    r=color[0]
    g=color[1]
    b=color[2]
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"

def print_conserved_region_groups_colored(conserved):
    _, pal = get_allen_info()
    for i in range(len(conserved)):
        print(f'group {i}')
        colors=[tuple(int(c * 255) for c in pal[reg][:3]) for reg in conserved[i]]
        for j in range(len(colors)):
            print(rgb_color_text(colors[j], list(conserved[i])[j]))



'''
plotting
'''

def plot_dim_reduction(algo_data='concat_z', algo='umap_z', mapping='Beryl', norm_=False,
                       run_umap=False, ncomp=2, n_neighbors=15, d=0.1, min_s=10, eps=0.5, 
                       means=False, exa=False, shuf=False, ds=0.5,
                       exa_squ=False, vers='concat', ax=None, control=False, cross_val=False,
                       axx=None, exa_clus=False, leg=False, restr=None,
                       nclus = 13, random_seed=0):
                       
    '''
    2 dims being pca on concat PETH; 
    colored by region
    algo_data in ['umap_z', 'concat_z']: chooses in which space to do clustering (method specified by mapping argument)
    algo in ['umap','tSNE','PCA','ICA']: in what space to plot data
    means: plot average dots per region
    exa: plot some example feature vectors
    exa_squ: highlight example squares in embedding space,
             make and extra plot for each with mean feature vector 
             and those of cells in square in color of mapping
    space: 'concat'  # can also be tSNE, PCA, umap, for distance space
    ds: marker size in main scatter
    restr: list of Beryl regions to restrict plot to
    '''
    
    feat = 'concat_z'
    if control!=False:
        norm_ = 'False_control'+control+'_0'
    
    r = regional_group(mapping, algo_data, vers=vers, norm_=norm_, min_s=min_s,
                       eps=eps, run_umap=run_umap, n_neighbors=n_neighbors, d=d,
                       ncomp=ncomp, nclus=nclus, random_seed=random_seed)

    if cross_val: # plot one set using results sorted from another set of trials
        norm_ = 'False_control'+control+'_1'
        r1 = regional_group(mapping, algo_data, vers=vers, norm_=norm_, min_s=min_s,
                       eps=eps, run_umap=run_umap, n_neighbors=n_neighbors, d=d,
                       ncomp=ncomp, nclus=nclus, random_seed=random_seed)
        r_cells = [cell in r1['uuids'] for cell in r['uuids']]
        r1_cells = [cell in r['uuids'] for cell in r1['uuids']]
        r1['cols'] = r['cols'][r_cells]
        r1['acs'] = r['acs'][r_cells]
        r1['umap_z'] = r1['umap_z'][r1_cells]
        r1['concat_z'] = r1['concat_z'][r1_cells]
        r = r1
        
    alone = False
    if not ax:
        alone = True
        if ncomp==3:
            fig = plt.figure(dpi=200)
            ax = fig.add_subplot(projection='3d')
        else:
            fig, ax = plt.subplots(label=f'{vers}_{mapping}', dpi=200)
        #ax.set_title(vers)
    
    if shuf:
        shuffle(r['cols'])
    
    if restr:
        # restrict to certain Beryl regions
        #r2 = regional_group('Beryl', algo, vers=vers)
        ff = np.bitwise_or.reduce([r['acs'] == reg for reg in restr]) 
    
    
        im = ax.scatter(r[algo][:,0][ff], r[algo][:,1][ff], 
                        marker='o', c=r['cols'][ff], s=ds, rasterized=True)
                        
    else: 
        if ncomp==3:
            im = ax.scatter(r[algo][:,0], r[algo][:,1], r[algo][:,2],
                            marker='o', c=r['cols'], s=ds, rasterized=True)
        else:
            im = ax.scatter(r[algo][:,0], r[algo][:,1], 
                            marker='o', c=r['cols'], s=ds, rasterized=True)                            
                        
    
    if means:
        # show means
        emb1 = [r['av'][reg][0][0] for reg in r['av']]
        emb2 = [r['av'][reg][0][1] for reg in r['av']]
        cs = [r['av'][reg][1] for reg in r['av']]
        ax.scatter(emb1, emb2, marker='o', facecolors='none', 
                   edgecolors=cs, s=600, linewidths=4, rasterized=True)
    
#    ax.set_xlabel(f'{algo} dim1')
#    ax.set_ylabel(f'{algo} dim2')
    zs = True if algo == 'umap_z' else False
    if alone:
        ax.set_title(f'norm: {norm_}, z-score: {zs}')
    ax.axis('off')
    ss = 'shuf' if shuf else ''
       
    
    if mapping in ['layers', 'kmeans']:
        if leg:
            ax.legend(handles=r['els'], ncols=1,
                      frameon=False).set_draggable(True)

    elif 'clusters' in mapping:
        nclus = len(Counter(r['acs']))
        cax = fig.add_axes([0.27, 0.2, 0.5, 0.01])
        norm = mpl.colors.Normalize(vmin=0, 
                                    vmax=nclus)
        cmap = mpl.cm.get_cmap('Spectral')                            
        fig.colorbar(mpl.cm.ScalarMappable(
                                norm=norm, 
                                cmap=cmap), 
                                cax=cax, orientation='horizontal')

    if alone:
        fig.tight_layout()
    fig.savefig(Path(one.cache_dir,'dmn', 'figs',
        f'{algo}_{vers}_norm{norm_}_{mapping}_{nclus}_{algo_data}_{ncomp}d_{d}_{n_neighbors}.pdf'), dpi=200, bbox_inches='tight')


    if exa:
        # plot a cells' feature vector
        # in extra panel when hovering over point
        fig_extra, ax_extra = plt.subplots()
        
        line, = ax_extra.plot(r[feat][0], 
                              label='Extra Line Plot')

        # Define a function to update the extra line plot 
        # based on the selected point
        
        def update_line(event):
            if event.mouseevent.inaxes == ax:
                x_clicked = event.mouseevent.xdata
                y_clicked = event.mouseevent.ydata
                
                selected_point = None
                for key, value in zip(r['nums'], r[algo]):
                    if (abs(value[0] - x_clicked) < 0.01 and 
                       abs(value[1] - y_clicked) < 0.01):
                        selected_point = key
                        break
                
                if selected_point:

                    line.set_data(T_BIN *np.arange(len(r[feat][key])),
                                  r[feat][key])
                    ax_extra.relim()
                    ax_extra.set_ylabel(feat)
                    ax_extra.set_xlabel('time [sec]')
                    ax_extra.autoscale_view()              
                    ax_extra.set_title(
                        f'Line Plot for x,y ='
                        f' {np.round(x_clicked,2), np.round(y_clicked,2)}')
                    fig_extra.canvas.draw()   
    
        # Connect the pick event to the scatter plot
        fig.canvas.mpl_connect('pick_event', update_line)
        im.set_picker(5)  # Set the picker radius for hover detection

    if exa_clus:
        # show for each cluter the mean PETH
        if axx is None:
            fg, axx = plt.subplots(nrows=len(np.unique(r['acs'])),
                                   sharex=True, sharey=False,
                                   figsize=(6,6))
                
        maxys = [np.max(np.mean(r[feat][
                 np.where(r['acs'] == clu)], axis=0)) 
                 for clu in np.unique(r['acs'])]
        
        kk = 0             
        for clu in np.unique(r['acs']):
                    
            #cluster mean
            xx = np.arange(len(r[feat][0])) /480
            yy = np.mean(r[feat][np.where(r['acs'] == clu)], axis=0)

            axx[kk].plot(xx, yy,
                     color=r['cols'][np.where(r['acs'] == clu)][0],
                     linewidth=2)
                     

            
            if kk != (len(np.unique(r['acs'])) - 1):
                axx[kk].axis('off')
            else:

                axx[kk].spines['top'].set_visible(False)
                axx[kk].spines['right'].set_visible(False)
                axx[kk].spines['left'].set_visible(False)      
                axx[kk].tick_params(left=False, labelleft=False)
                
            d2 = {}
            for sec in PETH_types_dict[vers]:
                d2[sec] = r['len'][sec]
                                
            # plot vertical boundaries for windows
            h = 0
            for i in d2:
            
                xv = d2[i] + h
                axx[kk].axvline(xv/480, linestyle='--', linewidth=1,
                            color='grey')
                
                if  kk == 0:            
                    axx[kk].text(xv/480 - d2[i]/(2*480), max(yy),
                             '   '+i, rotation=90, color='k', 
                             fontsize=10, ha='center')
            
                h += d2[i] 
            kk += 1                

#        #axx.set_title(f'{s} \n {len(pts)} points in square')
        axx[kk - 1].set_xlabel('time [sec]')
#        axx.set_ylabel(feat)
        if alone:
            fg.tight_layout()
        fg.savefig(Path(one.cache_dir,'dmn', 'figs',
            f'{vers}_norm{norm_}_{mapping}_clusters_{nclus}_{algo_data}.pdf'), dpi=150, bbox_inches='tight')


    if exa_squ:
    
        # get squares
        ns = 10  # number of random square regions of interest
        ss = 0.01  # square side length as a fraction of total area
        x_min = np.floor(np.min(r[algo][:,0]))
        x_max = np.ceil(np.max(r[algo][:,0]))
        y_min = np.floor(np.min(r[algo][:,1]))
        y_max = np.ceil(np.max(r[algo][:,1]))
        
        
        side_length = ss * (x_max - x_min)
        
        sqs = []
        for _ in range(ns):
            # Generate random x and y coordinates within the data range
            x = random.uniform(x_min, x_max - side_length)
            y = random.uniform(y_min, y_max - side_length)
            
            # Create a square represented as (x, y, side_length)
            square = (x, y, side_length)
            
            # Add the square to the list of selected squares
            sqs.append(square)
            

        
        r['nums'] = range(len(r[algo][:,0]))
        
        
        k = 0
        for s in sqs:
    
            
            # get points within square
            
            pts = []
            sq_x, sq_y, side_length = s
            
            for ke, value in zip(r['nums'], r[algo]):
                if ((sq_x <= value[0] <= sq_x + side_length) 
                    and (sq_y <= value[1] <= sq_y + side_length)):
                    pts.append(ke)            
          
            if len(pts) == 0:
                continue
          
            # plot squares in main figure
            rect = plt.Rectangle((s[0], s[1]), s[2], s[2], 
                    fill=False, color='r', linewidth=2)
            ax.add_patch(rect)
          
          
            # plot mean and individual feature line plots
            fg, axx = plt.subplots()          
          
            # each point individually
            maxys = []
            for pt in pts:
                axx.plot(T_BIN * np.arange(len(r[feat][pt])),
                         r[feat][pt],color=r['cols'][pt], linewidth=0.5)
                maxys.append(np.max(r[feat][pt]))         
                         
                
            #square mean
            axx.plot(T_BIN * np.arange(len(r[feat][pt])),
                     np.mean(r[feat][pts],axis=0),
                color='k', linewidth=2)    

            axx.set_title(f'{s} \n {len(pts)} points in square')
            axx.set_xlabel('time [sec]')
            axx.set_ylabel(feat)
            
            # plot vertical boundaries for windows
            h = 0
            for i in r['len']:
            
                xv = r['len'][i] + h
                axx.axvline(T_BIN * xv, linestyle='--',
                            color='grey')
                            
                axx.text(T_BIN * xv, 0.8 * np.max(maxys), 
                         i, rotation=90, 
                         fontsize=12, color='k')
            
                h += r['len'][i]



def clus_freqs(foc='reg', mapping='kmeans', algo_data='concat_z', nmin=50, overrep='std',
               nclus=13, vers='concat', nd=2, plot_wass=False, plot_hierarchy=False):

    '''
    For each k-means cluster, show an Allen region bar plot of frequencies,
    or vice versa
    foc: focus, either kmeans or Allen 
    '''
    
    r_a = regional_group('Beryl', algo_data, vers=vers, nclus=nclus)    
    r_k = regional_group(mapping, algo_data, vers=vers, nclus=nclus)

    if foc == 'cluster':
    
        # show frequency of regions for all clusters
        cluss = sorted(Counter(r_k['acs']))
        fig, axs = plt.subplots(nrows = len(cluss), ncols = 1,
                               figsize=(18.79,  15),
                               sharex=True, sharey=False)
        
        fig.canvas.manager.set_window_title(
            f'Frequency of Beryl region label per'
            f' kmeans cluster ({nclus}); vers ={vers}')                      
                               
        cols_dict = dict(list(Counter(zip(r_a['acs'], r_a['cols']))))
        
        # order regions by canonical list 
        p = (Path(iblatlas.__file__).parent / 'beryl.npy')
        regs_can = br.id2acronym(np.load(p), mapping='Beryl')
        regs_ = Counter(r_a['acs'])
        reg_ord = []
        for reg in regs_can:
            if reg in regs_:
                reg_ord.append(reg)        
        
        k = 0                       
        for clus in cluss:                       
            counts = Counter(r_a['acs'][r_k['acs'] == clus])
            reg_order = {reg: 0 for reg in reg_ord}
            for reg in reg_order:
                if reg in counts:
                    reg_order[reg] = counts[reg] 
                    
            # Preparing data for plotting
            labels = list(reg_order.keys())
            values = list(reg_order.values())        
            colors = [cols_dict[label] for label in labels]                
                               
            # Creating the bar chart
            bars = axs[k].bar(labels, values, color=colors)
            axs[k].set_ylabel(f'clus {clus}')
            axs[k].set_xticklabels(labels, rotation=90, 
                                   fontsize=6)
            
            for ticklabel, bar in zip(axs[k].get_xticklabels(), bars):
                ticklabel.set_color(bar.get_facecolor())        

            axs[k].set_xlim(-0.5, len(labels)-0.5)

            k += 1
        
        fig.tight_layout()        
        fig.subplots_adjust(top=0.951,
                            bottom=0.059,
                            left=0.037,
                            right=0.992,
                            hspace=0.225,
                            wspace=0.2)       

    elif foc=='reg':

        # show frequency of clusters for all regions

        # order regions by canonical list 
        p = (Path(iblatlas.__file__).parent / 'beryl.npy')
        regs_can = br.id2acronym(np.load(p), mapping='Beryl')
        regs_ = Counter(r_a['acs'])
        reg_ord = []
        for reg in regs_can:
            if reg in regs_ and regs_[reg] >= nmin:
                reg_ord.append(reg)        

        print(len(reg_ord), f'regions with at least {nmin} cells')
        ncols = int((len(reg_ord) ** 0.5) + 0.999)
        nrows = (len(reg_ord) + ncols - 1) // ncols
        
        fig, axs = plt.subplots(nrows = nrows, 
                                ncols = ncols,
                                figsize=(18.79,  15),
                                sharex=True)
        
        axs = axs.flatten()
                               
        cols_dict = dict(list(Counter(zip(r_k['acs'],
                    [tuple(color) for color in r_k['cols']]))))
                    
        cols_dictr = dict(list(Counter(zip(r_a['acs'],
                                          r_a['cols']))))
        
        cluss = sorted(list(Counter(r_k['acs'])))
        
        k = 0                         
        weights = []
        for reg in reg_ord:                       
            counts = Counter(r_k['acs'][r_a['acs'] == reg])
            clus_order = {clus: 0 for clus in cluss}
            for clus in clus_order:
                if clus in counts:
                    clus_order[clus] = counts[clus] 
                    
            # Preparing data for plotting
            labels = list(clus_order.keys())
            values = list(clus_order.values())
            #weights.append([x / sum(values) for x in values])
            weights.append(values)
            colors = [cols_dict[label] for label in labels]                
                               
            # Creating the bar chart
            bars = axs[k].bar(labels, values, color=colors)
            axs[k].set_ylabel(reg, color=cols_dictr[reg])
            axs[k].set_xticks(labels)
            axs[k].set_xticklabels(labels, fontsize=8)
            
            for ticklabel, bar in zip(axs[k].get_xticklabels(), bars):
                ticklabel.set_color(bar.get_facecolor())        

            axs[k].set_xlim(-0.5, len(labels)-0.5)

            k += 1
            
        fig.canvas.manager.set_window_title(
            f'Frequency of kmeans cluster ({nclus}) per'
            f' Beryl region label per; vers = {vers}')
                     
        fig.tight_layout()
        
        centers = r_k['centers']
        if plot_wass:
            plot_wasserstein_matrix(weights, centers, reg_ord, cols_dictr, vers=vers, nclus=nclus, nd=nd)
        if plot_hierarchy:
            get_difference_from_flat_dist(weights, centers, reg_ord, cols_dictr, vers=vers, nclus=nclus, nd=nd)

        list_overrep = get_clus_overrep(nclus, weights, reg_ord, overrep=overrep)

    fig.savefig(Path(pth_dmn.parent, 'figs',
                     f'{foc}_{algo_data}_{nclus}_{vers}.png')) 
    
    if foc=='reg':
    #    return weights, centers, reg_ord, list_overrep
        return list_overrep

    
def plot_wasserstein_matrix(weights, centers, reg_ord, cols_dictr, vers='concat', nclus=7, nd=2):
    # Calculate and plot wasserstein matrix of k-means clusters distributions over regions
    
    u = np.linspace(0, nclus-1, nclus)
    wass = np.zeros([len(weights),len(weights)])
    if nd==1:
        for i in range(len(weights)):
            for j in range(i):
                wass[i][j] = wasserstein_distance_nd(u,u,weights[i],weights[j])
                wass[j][i] = wass[i][j]
    elif nd>1:
        for i in range(len(weights)):
            for j in range(i):
                wass[i][j] = wasserstein_distance_nd(centers,centers,weights[i],weights[j])
                wass[j][i] = wass[i][j]
    else:
        return('what is nd')
            
    fig, ax0 = plt.subplots(figsize=(6, 6), dpi=200)
    ims = ax0.imshow(wass, origin='lower', interpolation=None)
    ax0.set_xticks(np.arange(len(reg_ord)), reg_ord, rotation=90, fontsize=4)
    ax0.set_yticks(np.arange(len(reg_ord)), reg_ord, fontsize=4)       
                   
    [t.set_color(i) for (i,t) in
        zip([cols_dictr[reg] for reg in reg_ord],
        ax0.xaxis.get_ticklabels())] 
         
    [t.set_color(i) for (i,t) in    
        zip([cols_dictr[reg] for reg in reg_ord],
        ax0.yaxis.get_ticklabels())]
    
    ax0.set_title(f'{vers}')
    cbar = plt.colorbar(ims,fraction=0.046, pad=0.04, 
                        extend='neither', ticks=[0, 0.5, 1, 1.5, 2, 2.5, 3])
    

    fig.savefig(Path(pth_dmn.parent, 'figs',
                     f'wasserstein_matrix_{nclus}_{vers}_nd{nd}.png'), dpi=200)
    
    save_wass = {}
    save_wass['res'] = wass
    save_wass['regs'] = reg_ord
    np.save(Path(pth_dmn,f'wasserstein_matrix_{nclus}_{vers}_nd{nd}.npy'), save_wass, allow_pickle=True)


def get_clus_overrep(nclus, weights, reg_ord, overrep='std'):
    # report if any cluster(s) overrepresented in a region
    list_overrep={f'cluster{k}': [] for k in range(nclus)}
    
    for i in range(len(weights)):
        if max(weights[i])<20:
            continue
            
        if overrep=='std': 
            #define overrep: at least one std over the mean
            idx=0
            for x in weights[i]:
                if x > np.mean(weights[i])+np.std(weights[i]):
                    list_overrep[f'cluster{idx}'].append(reg_ord[i])
                    print(reg_ord[i], 'overrep cluster', idx)
                idx = idx+1
        else: 
            #define overrep: at least 2x as large as cluster K's count for K in at least half of all the clusters
            if sum([2*x <= max(weights[i]) for x in weights[i]]) > nclus/2: 
                #first check if the max cluster is overrep
                idx=weights[i].index(max(weights[i]))
                list_overrep[f'cluster{idx}'].append(reg_ord[i])
                print(reg_ord[i], 'overrep cluster', idx)
                
                if sum([2*x <= np.sort(weights[i])[::-1][1] for x in weights[i]]) > nclus/2: 
                    #then check 2nd highest cluster
                    idx=weights[i].index(np.sort(weights[i])[::-1][1])
                    list_overrep[f'cluster{idx}'].append(reg_ord[i])
                    print(reg_ord[i], 'overrep cluster', idx)
    return list_overrep



def plot_wasserstein_matrix_from_data(vers='concat', nd=2, nclus=7):
    d = np.load(Path(pth_dmn, f'wasserstein_matrix_{nclus}_{vers}_nd{nd}.npy'), allow_pickle=True).flat[0]
    wass = d['res']
    reg_ord = d['regs']
    r_a = regional_group('Beryl', 'umap_z', vers=vers, nclus=nclus)
    cols_dictr = dict(list(Counter(zip(r_a['acs'], r_a['cols']))))
    
    fig, ax0 = plt.subplots(figsize=(6, 6), dpi=200)
    ims = ax0.imshow(wass, origin='lower', interpolation=None)
    ax0.set_xticks(np.arange(len(reg_ord)), reg_ord, rotation=90, fontsize=4)
    ax0.set_yticks(np.arange(len(reg_ord)), reg_ord, fontsize=4)       
                   
    [t.set_color(i) for (i,t) in
        zip([cols_dictr[reg] for reg in reg_ord],
        ax0.xaxis.get_ticklabels())] 
         
    [t.set_color(i) for (i,t) in    
        zip([cols_dictr[reg] for reg in reg_ord],
        ax0.yaxis.get_ticklabels())]
    
    ax0.set_title(f'{vers}')
    cbar = plt.colorbar(ims,fraction=0.046, pad=0.04, 
                        extend='neither', ticks=[0, 0.5, 1, 1.5, 2, 2.5, 3])
    

    fig.savefig(Path(pth_dmn.parent, 'figs',
                     f'wasserstein_matrix_{nclus}_{vers}_nd{nd}.png'), dpi=200)



def plot_xyz(mapping='Beryl', algo='concat_z', vers='concat', add_cents=False, nclus=13, save_fig=True,
             restr=False, restr_cell=False, name='all', print_name=False, ax=None, exa=False):

    '''
    3d plot of feature per cell
    add_cents: superimpose stars for region volumes and centroids
    exa: show example average feature vectors
    restr: a list of regions to be displayed
    restr_cell: a list of cells to be displayed
    name: identifier of the network being displayed, if restricted to a specific network of regions (restr!=False)
    '''

    r = regional_group(mapping, algo=algo, vers=vers, nclus=nclus)

    if ((mapping in tts__) or (mapping in PETH_types_dict)):
        cmap = mpl.cm.get_cmap('Spectral')
        norm = mpl.colors.Normalize(vmin=min(r['rankings']), 
                                    vmax=max(r['rankings']))
        cols = cmap(norm(r['rankings']))
        r['cols'] = cols

    xyz = r['xyz']*1000  #convert to mm
    
    alone = False
    if not ax:
        alone = True
        fig = plt.figure(figsize=(8.43,7.26), label=mapping)
        ax = fig.add_subplot(111,projection='3d')

    scalef = 1.2                  
    ax.view_init(elev=45.78, azim=-33.4)
    ax.set_xlim(min(xyz[:,0])/scalef, max(xyz[:,0])/scalef)
    ax.set_ylim(min(xyz[:,1])/scalef, max(xyz[:,1])/scalef)
    ax.set_zlim(min(xyz[:,2])/scalef, max(xyz[:,2])/scalef)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    if isinstance(restr, list):
        idcs = np.bitwise_or.reduce([r['acs'] == reg for reg in restr])   
        xyz = xyz[idcs]
        r['cols'] = np.array(r['cols'])[idcs]
        r['acs'] = np.array(r['acs'])[idcs]        
    elif isinstance(restr_cell, list):
        idcs = np.bitwise_or.reduce([r['uuids'] == uuid for uuid in restr_cell])   
        xyz = xyz[idcs]
        r['cols'] = np.array(r['cols'])[idcs]
        r['acs'] = np.array(r['acs'])[idcs]
    
    ax.scatter(xyz[:,0], xyz[:,1],xyz[:,2], 
               marker='o', s = 1 if alone else 0.5, c=r['cols'])
               
    if add_cents:
        # add centroids with size relative to volume
        if mapping !='Beryl':
            print('add cents only for Beryl')
            
        else:    
            regs = list(Counter(r['acs']))
            centsd = get_centroids()
            cents = np.array([centsd[x] for x in regs])          
            volsd = get_volume()
            vols = [volsd[x] for x in regs]

            if restr_cell!=False:
                scale = 10000
            else:
                scale = 5000
            vols = scale * np.array(vols)/np.max(vols)
            
            _,pa = get_allen_info()
            cols = [pa[reg] for reg in regs]
            ax.scatter(cents[:,0], cents[:,1], cents[:,2], 
                       marker='*', s = vols, color=cols)
                       
    fontsize = 14
    ax.set_xlabel('x [mm]', fontsize = fontsize)
    ax.set_ylabel('y [mm]', fontsize = fontsize)
    ax.set_zlabel('z [mm]', fontsize = fontsize)
    ax.tick_params(axis='both', labelsize=12)
    #ax.set_title(f'Mapping: {mapping}')
    ax.grid(False)
    nbins = 3
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=nbins))

    if ((mapping in tts__) or (mapping in PETH_types_dict)):
        # Create a colorbar based on the colormap and normalization
        mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array(r['rankings'])  # Set the data array for the colorbar
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label(f'mean {mapping} rankings')

    if alone:
        ax.set_title(f'{mapping}_{vers}_{name}')
        if print_name==True:
            plt.figtext(0.5, 0.9, restr, ha='center', fontsize=10)
        fig.tight_layout()
        if save_fig:
            fig.savefig(Path(one.cache_dir,'dmn', 'figs',
                f'{mapping}_{vers}_{name}_3d.png'),dpi=150)

    if exa:
        # add example time series; 20 from max to min equally spaced 
        if (mapping not in tts__) and (mapping not in PETH_types_dict):
            print('not implemented for other mappings')
            return

        
        feat = 'concat_bd'
        nrows = 10  # show 5 top cells in the ranking and 5 last
        rankings_s = sorted(r['rankings'])
        indices = [list(r['rankings']).index(x) for x in
                    np.concatenate([rankings_s[:nrows//2], 
                                    rankings_s[-nrows//2:]])]

        fg, axx = plt.subplots(nrows=nrows,
                                   sharex=True, sharey=False,
                                   figsize=(7,7))                   

        xx = np.arange(len(r[feat][0]))/c_sec 

        kk = 0             
        for ind in indices:
                                
            yy = r[feat][ind]

            axx[kk].plot(xx, yy,
                     color=r['cols'][ind],
                     linewidth=2)

            sss = (r['acs'][ind] + '\n' + str(r['pid'][ind][:3]))


            axx[kk].set_ylabel(sss)

            if kk != (len(indices) -1):
                axx[kk].spines['top'].set_visible(False)
                axx[kk].spines['right'].set_visible(False)
                axx[kk].spines['bottom'].set_visible(False)                
            else:

                axx[kk].spines['top'].set_visible(False)
                axx[kk].spines['right'].set_visible(False)
                #axx[kk].spines['left'].set_visible(False)      
                #axx[kk].tick_params(left=False, labelleft=False)
                       
            # plot vertical boundaries for windows
            h = 0
            for i in r['len']:
            
                xv = r['len'][i] + h
                axx[kk].axvline(xv/c_sec, linestyle='--', linewidth=1,
                            color='grey')
                
                ccc = ('r' if ((i == mapping) or 
                        (i in PETH_types_dict[mapping])) else 'k')

                if  kk == 0:            
                    axx[kk].text(xv/c_sec - r['len'][i]/(2*c_sec), max(yy),
                             '   '+i, rotation=90, 
                             color=ccc, 
                             fontsize=10, ha='center')
            
                h += r['len'][i] 
            kk += 1                

#        #axx.set_title(f'{s} \n {len(pts)} points in square')
        axx[kk - 1].set_xlabel('time [sec]')

        fg.suptitle(f'mapping: {mapping}, feat: {feat}')
        fg.tight_layout()



def get_difference_from_flat_dist(weights, centers, reg_ord, cols_dictr=None, vers='concat', nclus=13, nd=2):
    
    '''
    Calculate wasserstein distance between the k-means clusters distributions and a flat distribution, 
    then plot the correlation with cortical hierarchy
    centers: the centers of the k clusters
    weights: count for each of the k clusters in each region; 
             len(weights)=number of brain regions=len(reg_ord)
    reg_ord: ordered list of brain regions
    '''
    
    wass = np.zeros(len(weights))
    flat_dist = np.ones(nclus)
    if nd==1:
        u = np.linspace(0, nclus-1, nclus)
        for i in range(len(weights)):
            wass[i] = wasserstein_distance_nd(u,u,weights[i],flat_dist)
    elif nd>1:
        for i in range(len(weights)):
            wass[i]= wasserstein_distance_nd(centers,centers,weights[i],flat_dist)
    else:
        return('what is nd')
        
    save_wass = {}
    save_wass['res'] = wass
    save_wass['regs'] = reg_ord
    np.save(Path(pth_dmn,f'wasserstein_fromflatdist_{nclus}_{vers}_nd{nd}.npy'), save_wass, allow_pickle=True)

    # plot dist in rising order for all regions
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6,7), label=f'{vers}_{nclus}', dpi=150)
    ax[0].plot(np.sort(wass))
    order = np.argsort(wass)
    ax[0].set_xticks(np.arange(len(reg_ord)), np.array(reg_ord)[order], rotation=90, fontsize=4)
    ax[0].set_title(f'{vers}')
    ax[0].set_ylabel('wass_d from flat dist')

    if cols_dictr==None:
        r_a = regional_group('Beryl', 'umap_z', vers='concat', nclus=13)
        cols_dictr = dict(list(Counter(zip(r_a['acs'], r_a['cols']))))
        
    [t.set_color(i) for (i,t) in
        zip([cols_dictr[reg] for reg in np.array(reg_ord)[order]],
        ax[0].xaxis.get_ticklabels())]

    # plot dist w/ cortical hierarchy list
    area_list = np.loadtxt(Path(pth_dmn,'area_list.csv'), dtype=str)
    plot_list = set(area_list) & set(reg_ord)
    hierarchy = [list(area_list).index(reg) for reg in plot_list]
    order = [reg_ord.index(reg) for reg in plot_list]
    color_list=[cols_dictr[reg] for reg in plot_list]
    ax[1].scatter(hierarchy, wass[order], color=color_list)
    for i, txt in enumerate(plot_list):
        ax[1].annotate(txt, (hierarchy[i], wass[order][i]))

    spearman_corr, spearman_p = spearmanr(hierarchy, wass[order])
    slope, intercept, r_value, p_value, std_err = linregress(hierarchy, wass[order])
    x=np.sort(hierarchy)
    line_fit = slope * x + intercept
    ax[1].plot(x, line_fit, color="red", label=f"Linear fit: y = {slope:.2f}x + {intercept:.2f}")
    ax[1].legend()
    ax[1].set_ylabel('wass_d from flat dist')
    ax[1].set_xlabel('position in hierarchy')
    ax[1].set_title(f'{vers}, spearman R: {spearman_corr:.2f}, p_val: {spearman_p:.4f}')
    
    fig.tight_layout
    fig.savefig(Path(one.cache_dir,'dmn', 'figs', 
                     f'{vers}_{nclus}_correlate_hierarchy.pdf'), dpi=150)



def plot_connectivity_matrix(metric='umap_z', mapping='Beryl', nclus=7, nd=2, k=2, resl=1.01, 
                             ticktype='rectangles', vers='concat', ax0=None, clustering='louvain', 
                             diff='shifted', rerun=False, shuffling=False):

    '''
    all-to-all matrix for some measures
    '''


    if metric == 'cartesian':
        d = get_centroids(dist_=True)
    elif metric == 'pw':
        d = get_pw_dist(mapping=mapping, vers=vers)
    elif metric == 'wass':
        d = np.load(Path(pth_dmn, f'wasserstein_matrix_{nclus}_{vers}_nd{nd}.npy'), 
                    allow_pickle=True).flat[0]
    elif vers == 'quie-rest-diff-shifted':
        d = get_quiescence_resting_diff(metric=metric, diff='shifted')
    elif vers == 'quie-rest-diff-abs':
        d = get_quiescence_resting_diff(metric=metric, diff='abs')
    else:     
        d = get_reg_dist(algo=metric, vers=vers, rerun=rerun, shuffling=shuffling)
                
    res = d['res']
    regs = d['regs']
    
    _,pal = get_allen_info()
    
    alone = False
    if not ax0:
        alone=True
        if clustering=='hierarchy':
            fig, (ax_dendro, ax0) = plt.subplots(1, 2, 
                figsize=(8, 6), 
                gridspec_kw={'width_ratios': [1, 5]})
        else:
            fig, ax0 = plt.subplots(figsize=(6, 6), dpi=200)

        
    if clustering=='ari':
        rs = get_ari()
    
        ints = []
        for reg in rs:
            if reg in regs:
                ints.append(reg)
        
        rems = [reg for reg in regs if reg not in ints] 
        print(list(ints)[0], rems[0])
        node_order = list(ints) + rems
        
        ordered_indices = [list(regs).index(reg) for reg in node_order]
        regs_c = np.array(regs)[ordered_indices]
        regs_r = regs_c
        res = res[:, ordered_indices][ordered_indices, :]

    elif clustering=='dmn':
        dmn_idx = [list(regs).index(reg) for reg in dmn_regs]
        cortical_list = np.concatenate(list(cortical_regions.values()))
        cortical_list = set(cortical_list) & set(regs)
        ndmn_cortical_idx = [list(regs).index(reg) for reg in cortical_list
                    if reg not in dmn_regs]
        ndmn_idx = [list(regs).index(reg) for reg in regs
                    if reg not in cortical_list]
        ordered_indices = dmn_idx + ndmn_cortical_idx + ndmn_idx
        regs_c = np.array(regs)[ordered_indices]
        regs_r = regs_c
        res = res[:, ordered_indices][ordered_indices, :]
        

    else: 
        res, regs_r, regs_c, cluster_info = clustering_on_connectivity_matrix(
            res, regs, k=k, metric=metric, clustering=clustering, resl=resl)
        
    
    ims = ax0.imshow(res, origin='lower', interpolation=None)
    if ticktype == 'acronyms':
        ax0.set_xticks(np.arange(len(regs_c)), regs_c,
                       rotation=90, fontsize=2.5)
        ax0.set_yticks(np.arange(len(regs_r)), regs_r, fontsize=2.5)       
                       
        [t.set_color(i) for (i,t) in
            zip([pal[reg] for reg in regs_c],
            ax0.xaxis.get_ticklabels())] 
             
        [t.set_color(i) for (i,t) in    
            zip([pal[reg] for reg in regs_r],
            ax0.yaxis.get_ticklabels())]
                
    else:
        # plot region rectangles
        rect_height = 15 
        data_height = len(regs_c)
        
        x_tick_colors = [to_rgba(pal[reg]) for reg in regs_c]
        ax0.axis('off')

        for i, color in enumerate(x_tick_colors):
            rect = Rectangle((i - 0.5, -rect_height - 0.5), 1, 
                       rect_height, color=color, clip_on=False,
                       transform=ax0.transData)
            ax0.add_patch(rect)

        # Create colored rectangles for y-axis ticks
        y_tick_colors = [to_rgba(pal[reg]) for reg in regs_c]


        for i, color in enumerate(y_tick_colors):
            rect = Rectangle((-rect_height - 0.5, i - 0.5),
                              rect_height, 
                              1, color=color, clip_on=False,
                              transform=ax0.transData)
            ax0.add_patch(rect)         
                    
    
    if metric[-1] == 'e':
        vers = '30 ephysAtlas'
        
    #ax0.set_title(f'{metric}, {vers}')
    #ax0.set_ylabel(mapping)
    #cbar = plt.colorbar(ims,fraction=0.046, pad=0.04, 
    #                    extend='neither')#, ticks=[0, 0.5, 1]
    ax0.set_facecolor('none')
    #cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

    # Compute and plot cluster boundaries
    cluster_ids, cluster_sizes = np.unique(cluster_info, return_counts=True)
    start_idx = 0
    for size in cluster_sizes:
        # Draw a square around the cluster
        rect = Rectangle((start_idx, start_idx), size, size, 
                                 linewidth=2, edgecolor='black', facecolor='none')
        ax0.add_patch(rect)
        start_idx += size

    # Plot dendrogram if hierarchical clustering
    if clustering=='hierarchy':
        with plt.rc_context({'lines.linewidth': 0.5}):
            hierarchy.dendrogram(linkage_matrix, ax=ax_dendro, 
                orientation='left', labels=regs)

            
        ax_dendro.set_axis_off()
    
#    ax_dendro.set_yticklabels(regs)
#    [ax_dendro.spines[s].set_visible(False) for s in
#        ['left', 'right', 'top', 'bottom']]
#    ax_dendro.get_xaxis().set_visible(False)
#    [t.set_color(i) for (i,t) in    
#        zip([pal[reg] for reg in regs],
#        ax_dendro.yaxis.get_ticklabels())]    
    
    plt.subplots_adjust(wspace=0.05)
    fig.tight_layout()
    if clustering in ['spectralco', 'spectralbi', 'kmeans', 'kcenters', 'birch']:
        name = f'connectivity_matrix_{metric}_{clustering}_{vers}_k{k}'
        #np.save(Path(pth_dmn.parent, 'res', 
        #                 f'cluster_info_{metric}_{clustering}_{vers}_k{k}.npy'),
        #       cluster_info, allow_pickle=True)
    elif clustering in ['louvain', 'leiden']:
        name = f'connectivity_matrix_{metric}_{clustering}_{vers}_{resl}'
    else:
        name = f'connectivity_matrix_{metric}_{clustering}_{vers}'

    if shuffling==True:
        name = name + '_shuffled'
    fig.savefig(Path(pth_dmn.parent, 'figs', f'{name}.pdf'), dpi=200, transparent=True)
        
        #np.save(Path(pth_dmn.parent, 'res', 
        #                 f'cluster_info_{metric}_{clustering}_{vers}.npy'),
        #       cluster_info, allow_pickle=True)
    
    #if alone:
    #    fig.tight_layout()

    #fig0.suptitle(f'{algo}, {mapping}')
    #else:
    if not alone:
        return cluster_info



def plot_all_connectivity_matrices(mapping='Beryl', algo='umap_z', nclus=13, nd=2, k_range=[2,3,4,5,6,7],
                                   resl_range=[1,1.01,1.02,1.03,1.04,1.05,1.06,1.07],
                                   vers='concat', ax0=None, rerun=False):

    '''
    all-to-all matrix for all clustering measures
    '''

    fig, axs = plt.subplots(nrows=7, ncols=len(k_range),
                            figsize=(10,10), dpi=400)
    axs = axs.flatten()
    _,pal = get_allen_info()

    n = 0
    listr = {}
    for clustering in ['hierarchy', 'hierarchy', 'louvain', 'leiden', 'birch',
                       'spectralco', 'spectralbi', 'kmeans']:
        if n==0:
            metric='wass'
            d = np.load(Path(pth_dmn, f'wasserstein_matrix_{nclus}_{vers}_nd{nd}.npy'), 
                        allow_pickle=True).flat[0]
        else:
            metric = algo
            d = get_reg_dist(algo=metric, vers=vers, rerun=rerun)            
            
        res = d['res']
        regs = d['regs']

        if n<2:
            res, regs_r, regs_c, cluster_info = clustering_on_connectivity_matrix(
                res=res, regs=regs, k=None, metric=metric, clustering=clustering)
            
            ims = axs[n].imshow(res, origin='lower', interpolation=None)
            axs[n].set_xticks(np.arange(len(regs_c)), regs_c,
                   rotation=90, fontsize=1)
            axs[n].set_yticks(np.arange(len(regs_r)), regs_r, fontsize=1)       
                   
            [t.set_color(i) for (i,t) in
                zip([pal[reg] for reg in regs_c],
                axs[n].xaxis.get_ticklabels())] 
         
            [t.set_color(i) for (i,t) in    
                zip([pal[reg] for reg in regs_r],
                axs[n].yaxis.get_ticklabels())]

            if n==0:
                axs[n].set_title(f'wass_{clustering}', fontsize=10)
            else:
                axs[n].set_title(f'{clustering}', fontsize=10)
            cbar = plt.colorbar(ims,fraction=0.046, pad=0.04, 
                        extend='neither')#, ticks=[0, 0.5, 1]
            cbar.ax.tick_params(labelsize=5)
            
            n=n+1
            
        elif clustering in ['louvain', 'leiden']:
            for resl in resl_range:
                res, regs_r, regs_c, cluster_info = clustering_on_connectivity_matrix(
                    res=res, regs=regs, resl=resl, metric=metric, clustering=clustering)
                
                ims = axs[n].imshow(res, origin='lower', interpolation=None)
                axs[n].set_xticks(np.arange(len(regs_c)), regs_c,
                   rotation=90, fontsize=1)
                axs[n].set_yticks(np.arange(len(regs_r)), regs_r, fontsize=1)       
                   
                [t.set_color(i) for (i,t) in
                    zip([pal[reg] for reg in regs_c],
                    axs[n].xaxis.get_ticklabels())] 
         
                [t.set_color(i) for (i,t) in    
                    zip([pal[reg] for reg in regs_r],
                    axs[n].yaxis.get_ticklabels())]
            
                axs[n].set_title(f'{clustering}, resl{resl}', fontsize=10)
                cbar = plt.colorbar(ims,fraction=0.046, pad=0.04, 
                            extend='neither')#, ticks=[0, 0.5, 1]
                cbar.ax.tick_params(labelsize=5)
                                
                n=n+1

        else:
            for k in k_range:
                res, regs_r, regs_c, cluster_info = clustering_on_connectivity_matrix(
                    res=res, regs=regs, k=k, metric=metric, clustering=clustering)
                
                ims = axs[n].imshow(res, origin='lower', interpolation=None)
                axs[n].set_xticks(np.arange(len(regs_c)), regs_c,
                   rotation=90, fontsize=1)
                axs[n].set_yticks(np.arange(len(regs_r)), regs_r, fontsize=1)       
                   
                [t.set_color(i) for (i,t) in
                    zip([pal[reg] for reg in regs_c],
                    axs[n].xaxis.get_ticklabels())] 
         
                [t.set_color(i) for (i,t) in    
                    zip([pal[reg] for reg in regs_r],
                    axs[n].yaxis.get_ticklabels())]
            
                axs[n].set_title(f'{clustering}, k{k}', fontsize=10)
                cbar = plt.colorbar(ims,fraction=0.046, pad=0.04, 
                            extend='neither')#, ticks=[0, 0.5, 1]
                cbar.ax.tick_params(labelsize=5)
                                
                n=n+1
                
    
    fig.suptitle(vers)
    fig.tight_layout()
    fig.savefig(Path(pth_dmn.parent, 'figs', 
                    f'all_connectivity_matrices_{vers}_{algo}.pdf'), 
                dpi=400)



def plot_avg_peth_from_vers(vers, metric='umap_z', clustering='louvain', 
                                  nclus=7, nd=2, k=2, resl=1.01, rerun=False,
                                  same_plot=True):
    d = get_reg_dist(algo=metric, vers=vers, rerun=rerun)
    _, regs_r, _, info0 = clustering_on_connectivity_matrix(
            d['res'], d['regs'], k=k, metric=metric, clustering=clustering, resl=resl)
    info0 = np.sort(info0) # sort cluster labels for regions
    r = regional_group(mapping='Beryl', algo=metric, vers=vers) # get peth data

    feat = 'concat_z'
    if same_plot:
        fg, axx = plt.subplots(figsize=(10,2))
    else:
        fg, axx = plt.subplots(nrows=len(np.unique(info0)),
                               sharex=True, sharey=True,
                               figsize=(6,6))
                        
    #for clu in np.unique(info0):
    for clu in range(3):
            
            #cluster mean
            listr = regs_r[info0==clu] #list of regions in a cluster
            #print('regs in the cluster:', listr)
            xx = np.arange(len(r[feat][0])) /480
            yy = np.mean(r[feat][np.where(np.isin(r['acs'], listr))], axis=0)
            print('cluster:', clu, 'mean', np.mean(yy))

            if same_plot:
                axx.plot(xx, yy, linewidth=1)
                axx.spines['top'].set_visible(False)
                axx.spines['right'].set_visible(False)
            else:
                axx[clu].plot(xx, yy, linewidth=1)
                                 
                if clu != (len(np.unique(r['acs'])) - 1):
                    axx[clu].axis('off')
                else:
    
                    axx[clu].spines['top'].set_visible(False)
                    axx[clu].spines['right'].set_visible(False)
                    axx[clu].spines['left'].set_visible(False)      
                    axx[clu].tick_params(left=False, labelleft=False)
                
            d2 = {}
            for sec in PETH_types_dict[vers]:
                d2[sec] = r['len'][sec]
                                
            # plot vertical boundaries for windows
            h = 0
            for i in d2:
            
                xv = d2[i] + h
                if same_plot:
                    axx.axvline(xv/480, linestyle='--', linewidth=0.75,
                                color='grey')
                    if clu==0:
                        axx.text(xv/480 - d2[i]/(2*480), max(yy),
                                 '   '+i, rotation=90, color='k', 
                                 fontsize=10, ha='center')
                else:
                    axx[clu].axvline(xv/480, linestyle='--', linewidth=0.75,
                                    color='grey')
                
                    if  clu==0:   
                        axx[clu].text(xv/480 - d2[i]/(2*480), max(yy),
                                 '   '+i, rotation=90, color='k', 
                                 fontsize=10, ha='center')
            
                h += d2[i] 

    #axx[kk - 1].set_xlabel('time [sec]')
    #axx.set_ylabel(feat)
    fg.tight_layout()
    fg.savefig(Path(one.cache_dir,'dmn', 'figs',
       f'{vers}_avg_peth_from_{clustering}_k{k}_resl{resl}.png'), dpi=150, bbox_inches='tight')


def plot_avg_peth_networks(vers, networks, metric='umap_z', clustering='louvain', 
                           resl=1.01, rerun=False, rename=True, same_plot=False,
                           n_permutations = 1000, multi_test='stouffer'):
    '''
    plot avg peth from networks defined from networks dict (reliable networks from split trial controls)
    also plot randomly shuffled peths as controls
    '''

    participation_networks = get_regional_participation_all_networks(networks=networks,
                                                                     clustering=clustering,
                                                                     rename=rename)
    d = get_reg_dist(algo='umap_z', vers='concat', rerun=rerun)
    regs = d['regs']
    
    r = regional_group(mapping='Beryl', algo=metric, vers=vers) # get peth data
    feat = 'concat_z'
                        
    phase=phases_dict[vers]
    clusters = [network for network in list(participation_networks.keys()) if network.startswith(phase)]
    if same_plot:
        fg, axx = plt.subplots(figsize=(10,2), dpi=200)
    else:
        fg, axx = plt.subplots(nrows=len(clusters),
                               sharex=True, sharey=True,
                               figsize=(10,6), dpi=100)

    # colors_rgb = [
    #     [65 / 255, 105 / 255, 225 / 255],   # Royal Blue
    #     [220 / 255, 20 / 255, 60 / 255],    # Crimson
    #     [46 / 255, 139 / 255, 87 / 255],    # Sea Green
    #     [255 / 255, 140 / 255, 0 / 255]     # Dark Orange
    # ]
    rgba_map = {
        'pink': [255/255, 182/255, 193/255],
        'green': [60/255, 179/255, 113/255]
    }
    # colors_rgb = {k: rgba_map.get(network_colors.get(k)) for k in networks}

    kk=0
    for clu in clusters:

        weights = participation_networks[clu]
        # filter out cells not in regs (void or undefined)
        mask = np.isin(r['acs'], regs)
        filtered_feat = r[feat][mask]
        # calculated weighted average based on regional participation in the cluster
        mapped_weights = np.array([weights[list(regs).index(reg)] for reg in r['acs'][mask]])
        yy = np.average(filtered_feat, axis=0, weights=mapped_weights)                
        xx = np.arange(len(r[feat][0])) /480
        #print('cluster:', clu, 'mean', np.mean(yy))

        # create randomly shuffled controls
        controls = []
        for _ in range(n_permutations):
            shuffled_weights = np.random.permutation(mapped_weights)
            control_yy = np.average(filtered_feat, axis=0, weights=shuffled_weights)
            controls.append(control_yy)
        controls = np.array(controls)
        p_values = np.mean(np.abs(controls - np.mean(controls, axis=0)) >= np.abs(yy - np.mean(controls, axis=0)), 
                           axis=0)
        alpha_values = np.where(p_values < 0.01, 1.0, 0.2)
        # Create an RGBA color array (blue color with variable alpha)
        colors = np.zeros((len(xx), 4))  # 4 for RGBA
        colors[:, 3] = alpha_values  # Alpha channel
        colors[:, 0] = rgba_map[network_colors[clu]][0]
        colors[:, 1] = rgba_map[network_colors[clu]][1]
        colors[:, 2] = rgba_map[network_colors[clu]][2]           

        # print(p_values, np.mean(p_values))
        
        # get a combined p value
        # if multi_test == 'fisher':
            # chi2_stat = -2 * np.sum(np.log(p_values))
            # combined_p_value = chi2.sf(chi2_stat, df=2 * len(p_values))  # Degrees of freedom = 2 * number of p-values
            # print(chi2_stat)
        # elif multi_test == 'stouffer':
        if multi_test == 'stouffer':
            p_values = np.clip(p_values, 1e-10, 1-1e-10) # correct for zero/one values to avoid inf when z-scoring
            # z_scores = norm.ppf(1 - np.array(p_values))
            # weights = np.ones_like(z_scores)
            # weighted_z_sum = np.sum(weights * z_scores)
            # combined_z_score = weighted_z_sum / np.sqrt(np.sum(weights**2))
            # combined_p_value = norm.cdf(-abs(combined_z_score))
            # print(z_scores)
            # print(weighted_z_sum, combined_z_score)
        combined_res = combine_pvalues(p_values, method=multi_test)
        combined_p_value = combined_res.pvalue
            
        print(f'{clu}: p_val {combined_p_value:.10f}')
    
        if same_plot:
            for control in controls[:100]:
                axx.plot(xx, control, color='gray', alpha=0.1, linewidth=0.1)
            axx.plot(xx, yy, linewidth=0.5)
            axx.scatter(xx, yy, 
                        facecolors=colors, 
                        # edgecolors='black', 
                        s=10)
            axx.spines['top'].set_visible(False)
            axx.spines['right'].set_visible(False)
            axx.set_facecolor('none')
        else:
            for control in controls[:100]:
                axx[kk].plot(xx, control, color='gray', alpha=0.1, linewidth=0.1)                             
            axx[kk].spines['top'].set_visible(False)
            axx[kk].spines['right'].set_visible(False)
            axx[kk].plot(xx, yy, linewidth=0.5)
            axx[kk].scatter(xx, yy, facecolors=colors, 
                            # edgecolors='black', 
                            s=20)
            #axx[kk].spines['left'].set_visible(False)      
            #axx[kk].tick_params(left=False, labelleft=False)
            axx[kk].text(0.1, max(yy) * 0.9, f'p={combined_p_value:.4f}', fontsize=7, color='black')
            axx[kk].set_facecolor('none')
                
            d2 = {}
            for sec in PETH_types_dict[vers]:
                d2[sec] = r['len'][sec]
                                
            # plot vertical boundaries for windows
            h = 0
            for i in d2:
            
                xv = d2[i] + h
                if same_plot:
                    axx.axvline(xv/480, linestyle='--', linewidth=0.75,
                                color='grey')
                    if kk==0:
                        axx.text(xv/480 - d2[i]/(2*480), max(yy),
                                 '   '+i, rotation=90, color='k', 
                                 fontsize=10, ha='center')
                else:
                    axx[kk].axvline(xv/480, linestyle='--', linewidth=0.75,
                                    color='grey')
                
                    if  kk==0:   
                        axx[kk].text(xv/480 - d2[i]/(2*480), max(yy),
                                 '   '+i, rotation=90, color='k', 
                                 fontsize=10, ha='center')
            
                h += d2[i] 

        kk+=1

    #axx[kk - 1].set_xlabel('time [sec]')
    #axx.set_ylabel(feat)
    fg.tight_layout()
    fg.savefig(Path(one.cache_dir,'dmn', 'figs',
       f'{vers}_avg_peth_wcontrols_{clustering}_resl{resl}.pdf'), dpi=100, 
               bbox_inches='tight', transparent=True)



def plot_avg_peth_all_vers(vers_list, metric='umap_z', clustering='louvain', 
                                  nclus=7, nd=2, k=2, resl=1.01, rerun=False,
                                  same_plot=True):
    '''
    for each vers in vers_list, plot avg peth for first three clusters from clustering results
    '''
    
    fg, axx = plt.subplots(nrows=len(vers_list),
                           figsize=(12, 1.5*len(vers_list)))

    kk=0
    for vers in vers_list:
        d = get_reg_dist(algo=metric, vers=vers, rerun=rerun)
        _, regs_r, _, info0 = clustering_on_connectivity_matrix(
                d['res'], d['regs'], k=k, metric=metric, clustering=clustering, resl=resl)
        info0 = np.sort(info0) # sort cluster labels for regions
        r = regional_group(mapping='Beryl', algo=metric, vers=vers) # get peth data
    
        feat = 'concat_z'
                        
        #for clu in np.unique(info0):
        for clu in range(3):
                
                #cluster mean
                listr = regs_r[info0==clu] #list of regions in a cluster
                #print('regs in the cluster:', listr)
                xx = np.arange(len(r[feat][0])) /480
                yy = np.mean(r[feat][np.where(np.isin(r['acs'], listr))], axis=0)
                #print('cluster:', clu, 'mean', np.mean(yy))
    
                axx[kk].plot(xx, yy, linewidth=1)
            
                axx[kk].spines['top'].set_visible(False)
                axx[kk].spines['right'].set_visible(False)
                axx[kk].tick_params(left=False, labelleft=False)
                    
                d2 = {}
                for sec in PETH_types_dict[vers]:
                    d2[sec] = r['len'][sec]
                                    
                # plot vertical boundaries for windows
                h = 0
                for i in d2:
                
                    xv = d2[i] + h
                    axx[kk].axvline(xv/480, linestyle='--', linewidth=0.75,
                                    color='grey')
                    if clu==0:
                        axx[kk].text(xv/480 - d2[i]/(2*480), max(yy),
                                     peth_dict[i],
                                     rotation=90, color='k', 
                                     fontsize=10, ha='center')
                
                    h += d2[i]
        kk+=1
        
    #axx[kk - 1].set_xlabel('time [sec]')
    #axx.set_ylabel(feat)
    fg.tight_layout()
    fg.savefig(Path(one.cache_dir,'dmn', 'figs',
       f'avg_peth_all_clusters_{clustering}_resl{resl}.pdf'), dpi=150, bbox_inches='tight')


def plot_avg_peth_all_networks(networks, clustering='louvain', resl=1.01, rename=False,
                               show_name=False):

    participation_networks = get_regional_participation_all_networks(networks=networks,
                                                                     clustering=clustering,
                                                                     rename=rename)
    d = get_reg_dist(algo='umap_z', vers='concat')
    regs = d['regs']
    vers_list = [network.rsplit('_', 1)[0] for network in networks]
    vers_list = list(dict.fromkeys(vers_list))
    fg, axx = plt.subplots(nrows=len(vers_list),
                           figsize=(12, 1.5*len(vers_list)))

    kk=0
    for vers in vers_list:
        r = regional_group(mapping='Beryl', algo='umap_z', vers=vers) # get peth data    
        feat = 'concat_z'

        clusters = [network for network in networks if network.startswith(vers)]                        
        for clu in clusters:

            weights = participation_networks[clu]
            # filter out cells not in regs (void or undefined)
            mask = np.isin(r['acs'], regs)
            filtered_feat = r[feat][mask]
            # calculated weighted average based on regional participation in the cluster
            mapped_weights = np.array([weights[list(regs).index(reg)] for reg in r['acs'][mask]])
            yy = np.average(filtered_feat, axis=0, weights=mapped_weights)                
            xx = np.arange(len(r[feat][0])) /480
            #print('cluster:', clu, 'mean', np.mean(yy))
            axx[kk].plot(xx, yy, linewidth=0.5)

            d2 = {}
            for sec in PETH_types_dict[vers]:
                d2[sec] = r['len'][sec]
                                
            # plot vertical boundaries for windows
            h = 0
            for i in d2:
            
                xv = d2[i] + h
                axx[kk].axvline(xv/480, linestyle='--', linewidth=0.25,
                                color='grey')
                if show_name:
                    if clu.rsplit('_', 1)[1]=='1':
                        axx[kk].text(xv/480 - d2[i]/(2*480), max(yy),
                                     peth_dict[i],
                                     rotation=90, color='k', 
                                     fontsize=10, ha='center')
            
                h += d2[i]
                    
        axx[kk].spines['top'].set_visible(False)
        axx[kk].spines['right'].set_visible(False)
        axx[kk].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)                    

        axx[kk].set_facecolor('none')
        kk+=1
        
    #axx[kk - 1].set_xlabel('time [sec]')
    #axx.set_ylabel(feat)
    fg.tight_layout()
    fg.savefig(Path(one.cache_dir,'dmn', 'figs',
       f'avg_peth_weighted_networks_{clustering}_resl{resl}.pdf'), dpi=150, 
               bbox_inches='tight', transparent=True)



def plot_avg_peth_from_all_clustering(vers, nd=2, rerun=False, k_range=[2,3,4,5,6]):

    fig = plt.figure(figsize=(10,10), dpi=400)
    outer_grid = gridspec.GridSpec(4, len(k_range), wspace=0.4, hspace=0.4)
    
    n=0
    for clustering in ['hierarchy', 'hierarchy', 'louvain', 'leiden', 'birch', 
                       'spectralco', 'spectralbi', 'kmeans']:
        print(clustering)

        # load data from connectivity matrix
        if n==0:
            metric='wass'
            d = np.load(Path(pth_dmn, f'wasserstein_matrix_7_{vers}_nd{nd}.npy'), 
                        allow_pickle=True).flat[0]
        else:
            metric = 'umap_z'
            d = get_reg_dist(algo=metric, vers=vers, rerun=rerun)

        # perform clustering and plot results
        if n<5: # first five methods without a k value
            _, regs_r, _, info0 = clustering_on_connectivity_matrix(
                d['res'], d['regs'], k=None, metric=metric, clustering=clustering)
            
            info0 = np.sort(info0)
            # load peth data
            r = regional_group(mapping='Beryl', algo='umap_z', vers=vers, nclus=7)
            feat = 'concat_z'

            # plot nth subplot
            inner_grid = gridspec.GridSpecFromSubplotSpec(len(np.unique(info0)), 1, 
                            subplot_spec=outer_grid[n])
            outer_ax = fig.add_subplot(outer_grid[n])
            outer_ax.set_title(clustering)  # Title for outer subplot
            outer_ax.set_frame_on(False)  # Hide the frame of the outer axis
            outer_ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

                        
            kk = 0
            for clu in np.unique(info0):
                print('cluster:', clu)
            
                #cluster mean
                listr = regs_r[info0==clu] #list of regions in a cluster
                print('regs in the cluster:', listr)
                xx = np.arange(len(r[feat][0])) /480
                yy = np.mean(r[feat][np.where(np.isin(r['acs'], listr))], axis=0)

                axx = fig.add_subplot(inner_grid[kk])
                axx.plot(xx, yy, linewidth=2)
                
                if kk != (len(np.unique(r['acs'])) - 1):
                    axx.axis('off')
                else:

                    axx.spines['top'].set_visible(False)
                    axx.spines['right'].set_visible(False)
                    axx.spines['left'].set_visible(False)      
                    axx.tick_params(left=False, labelleft=False)
                
                d2 = {}
                for sec in PETH_types_dict[vers]:
                    d2[sec] = r['len'][sec]
                                
                # plot vertical boundaries for windows
                h = 0
                for i in d2:
            
                    xv = d2[i] + h
                    axx.axvline(xv/480, linestyle='--', linewidth=1,
                                    color='grey')
                
                    #if  kk == 0:            
                    #    axx.text(xv/480 - d2[i]/(2*480), max(yy),
                    #         '   '+i, rotation=90, color='k', 
                    #         fontsize=10, ha='center')
            
                    h += d2[i] 
                kk += 1
        
            axx.set_xlabel('time [sec]')

            n = n+1                     
            
        else: # clustering methods with specified k
            for k in k_range:
                print('k =', k)
                _, regs_r, _, info0 = clustering_on_connectivity_matrix(
                    d['res'], d['regs'], k=k, metric=metric, clustering=clustering)
                
                # plot nth subplot
                inner_grid = gridspec.GridSpecFromSubplotSpec(len(np.unique(info0)), 1, 
                            subplot_spec=outer_grid[n])
                outer_ax = fig.add_subplot(outer_grid[n])
                outer_ax.set_title(f'{clustering}_{k}')  # Title for outer subplot
                outer_ax.set_frame_on(False)  # Hide the frame of the outer axis
                outer_ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        
                kk = 0
                for clu in np.unique(info0):
                    print('cluster:', clu)
            
                    #cluster mean
                    listr = regs_r[info0==clu] #list of regions in a cluster
                    print('regs in the cluster:', listr)
                    xx = np.arange(len(r[feat][0])) /480
                    yy = np.mean(r[feat][np.where(np.isin(r['acs'], listr))], axis=0)

                    axx = fig.add_subplot(inner_grid[kk])
                    axx.plot(xx, yy, linewidth=2)
                
                    if kk != (len(np.unique(r['acs'])) - 1):
                        axx.axis('off')
                    else:

                        axx.spines['top'].set_visible(False)
                        axx.spines['right'].set_visible(False)
                        axx.spines['left'].set_visible(False)      
                        axx.tick_params(left=False, labelleft=False)
                
                    d2 = {}
                    for sec in PETH_types_dict[vers]:
                        d2[sec] = r['len'][sec]
                                
                    # plot vertical boundaries for windows
                    h = 0
                    for i in d2:
            
                        xv = d2[i] + h
                        axx.axvline(xv/480, linestyle='--', linewidth=1,
                                    color='grey')
                
                        #if  kk == 0:            
                        #    axx.text(xv/480 - d2[i]/(2*480), max(yy),
                        #         '   '+i, rotation=90, color='k', 
                        #         fontsize=10, ha='center')
            
                        h += d2[i] 
                    kk += 1

                n = n+1
            axx.set_xlabel('time [sec]')

    fig.suptitle(vers, fontsize=20)
    fig.tight_layout()    
    fig.savefig(Path(one.cache_dir,'dmn', 'figs',
        f'{vers}_avg_peth_from_all_clustering.png'), dpi=400, bbox_inches='tight')


def get_regional_participation_all_networks(networks, clustering='louvain', rename=True):
    participation_networks={}
    d = get_reg_dist(algo='umap_z', vers='concat')
    regs = d['regs']
    control_list=[False, 'ver0_0', 'ver0_1', 'ver1_0', 'ver1_1',
                  'ver2_0', 'ver2_1', 'ver3_0', 'ver3_1']
    
    for network in networks:
        vers=network.rsplit('_', 1)[0]
        net_id=network.rsplit('_', 1)[1]
        reg=networks[network]
        
        regs_ordered, clusters = get_cluster_membership_across_controls(control_list, 
                                                                        vers, 
                                                                        clustering)    
        
        # Get network participation ratio
        participation=np.zeros(len(regs))
        for control in control_list:
            clus_id=int(clusters[control][regs_ordered[control]==reg]) # get clus id from representative reg in clus
            regs_in_clus=regs_ordered[control][clusters[control]==clus_id] # find all other regs in this cluster
            participation=participation + [reg in regs_in_clus for reg in regs]
        
        participation=participation/np.max(participation)
        phase=phases_dict[vers]
        if rename:
            participation_networks[f'{phase}_{net_id}']=participation
        else:
            participation_networks[network]=participation

    return participation_networks

def get_reproducible_clusters_with_controls(control_list, vers='concat', clustering='louvain'):
    
    regs_ordered, clusters = get_cluster_membership_across_controls(control_list,
                                                                    vers=vers,
                                                                    clustering=clustering)   
    for n in range(int(len(control_list)/2)+1):
        if n==0:
            x0 = find_cluster_overlap_between_paired_conditions(clusters, regs_ordered, 
                                                                control_list[n], control_list[len(control_list)-n-1])
        else:
            x1 = find_cluster_overlap_between_paired_conditions(clusters, regs_ordered, 
                                                                control_list[n], control_list[len(control_list)-n-1])
            x0 = find_overlaps(x0, x1)
    return x0


def get_cluster_membership_across_controls(control_list, vers='concat', clustering='louvain', shuffling=False):
    clusters, regs_ordered = {}, {}
    for control in control_list:
        d = get_reg_dist(algo='umap_z', vers=vers, control=control, shuffling=shuffling)
        res = d['res']
        regs = d['regs']
        _, _, regs, cluster_info = clustering_on_connectivity_matrix(res, regs, 
                                                                     clustering=clustering)
        clusters[control] = np.sort(cluster_info)
        regs_ordered[control] = regs

    return regs_ordered, clusters


def plot_connectivity_network_style(vers, clustering, control_list=None, metric='umap_z', tau=0.01, 
                                    k=2, resl=1.01, layout='manual', coloring='Beryl', labels=False,
                                    diff='shifted', threshold=0, edge_display=0.5, 
                                    simplify_clusters=True, amplify=1):

    # get data from clustering
    if vers == 'quie-rest-diff-shifted':
        d = get_quiescence_resting_diff(metric=metric, diff='shifted')
    elif vers == 'quie-rest-diff-abs':
        d = get_quiescence_resting_diff(metric=metric, diff='abs')
    else:
        d = get_reg_dist(algo=metric, vers=vers)
    res = d['res']
    regs = d['regs']    
    res, regs_r, regs_c, cluster_info = clustering_on_connectivity_matrix(
            res, regs, k=k, metric=metric, clustering=clustering, resl=resl, tau=tau)

    if control_list!=None:
        x0=get_reproducible_clusters_with_controls(control_list, vers=vers, clustering=clustering)
        cluster_info=np.full(len(regs_r), -1, dtype=int)
        included_regs=[]
        for i in range(len(x0)):
            for reg in x0[i]:
                idx=np.where(regs_r==reg)[0][0]
                cluster_info[idx]=i
                included_regs.append(idx)
        cluster_info = cluster_info[included_regs]

    # Add graph nodes and edges with weights (connectivity strengths)
    G = nx.Graph()    
    if control_list==None:
        G.add_nodes_from(regs_r)
        for i in range(len(regs_r)):
            for j in range(i+1, len(regs_r)):  # Use only upper triangle for undirected graph
                if res[i, j]!=0:  # Only add edges for nonzero connectivity
                    G.add_edge(regs_r[i], regs_r[j], weight=res[i, j])
    else:
        G.add_nodes_from(regs_r[included_regs])
        for i in range(len(included_regs)):
            for j in range(i+1, len(included_regs)):  # Use only upper triangle for undirected graph
                if res[included_regs[i], included_regs[j]]!=0:  # Only add edges for nonzero connectivity
                    G.add_edge(regs_r[included_regs[i]], regs_r[included_regs[j]], 
                               weight=res[included_regs[i], included_regs[j]])

    # Define a layout for the nodes
    if layout=='spring':
        pos = nx.spring_layout(G)
    elif layout=='shell':
        pos = nx.shell_layout(G)
    elif layout=='kamada':
        pos = nx.kamada_kawai_layout(G)
    elif layout=='manual': #manually put nodes in the same cluster near each other
        pos = nx.kamada_kawai_layout(G)
        clusters = np.sort(cluster_info)
        for cluster in set(clusters):
            if control_list==None:
                 cluster_nodes = regs_r[clusters==cluster]
            else:
                 cluster_nodes = regs_r[included_regs][clusters==cluster]
            cluster_center = np.mean([pos[node] for node in cluster_nodes], axis=0)
            for node in cluster_nodes:
                pos[node] = 0.1 * pos[node] + 0.9 * cluster_center
    
    # Draw nodes
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 11), dpi=200)
    if coloring=='dmn':
        cmap0 = cm.get_cmap('tab10')
        cols = [cmap0(int(reg in dmn_regs)) for reg in G.nodes()]
        transparency = [(int(reg in dmn_regs)+1)/2 for reg in G.nodes()]
    elif coloring=='cortical':
        pal={}
        for reg in G.nodes():
            pal[reg] = 'white'
            for cortical in cortical_regions.keys():
                        if reg in cortical_regions[cortical]:
                            pal[reg] = cortical_colors[cortical]
        cols = [pal[reg] for reg in G.nodes()]
        transparency = [(int(color!='white')+1)/2 for color in cols]
    else:
        _, pal = get_allen_info()
        cols = [pal[reg] for reg in G.nodes()]
        transparency = 1
    cmap = cm.get_cmap('tab20')
    #if control_list!=None:
    #    cluster_info = cluster_info[included_regs]
    node_colors = [cmap(i) for i in np.sort(cluster_info)]
    nx.draw_networkx_nodes(G, pos, ax=axs[0], node_color=cols, alpha=transparency, node_size=40)
    nx.draw_networkx_nodes(G, pos, ax=axs[1], node_color=node_colors, node_size=40)

    
    if simplify_clusters:
        # Only show one edge of avg connectivity between cluster pairs

        if amplify==None:
            amplify=1/len(regs_r)
        node_to_cluster = {node: cluster for node, cluster in zip(regs_r, np.sort(cluster_info))}
        cluster_edges = {}
        within_cluster_weights = {}
        
        # Compute inter-cluster and within-cluster connectivity        
        for i in range(len(regs_r)):
            for j in range(i+1, len(regs_r)):
                #if res[i, j] != 0:  # Only consider nonzero connectivity
                    cluster_i = node_to_cluster[regs_r[i]]
                    cluster_j = node_to_cluster[regs_r[j]]
        
                    if cluster_i == cluster_j:  # Within-cluster connections
                        if cluster_i not in within_cluster_weights:
                            within_cluster_weights[cluster_i] = []
                        within_cluster_weights[cluster_i].append(res[i, j])
                    else:  # Between-cluster connections
                        key = tuple(sorted((cluster_i, cluster_j)))  # Ensure consistent ordering
                        if key not in cluster_edges:
                            cluster_edges[key] = []
                        cluster_edges[key].append(res[i, j])
        
        # Add only one edge between clusters, with average weight
        for (c1, c2), weights in cluster_edges.items():
            avg_weight = np.sum(weights)*amplify
            
            # Find a representative node pair between clusters
            node_i, node_j = None, None
            for i in range(len(regs_r)):
                for j in range(i+1, len(regs_r)):
                    if res[i, j] != 0:
                        if node_to_cluster[regs_r[i]] == c1 and node_to_cluster[regs_r[j]] == c2:
                            node_i, node_j = regs_r[i], regs_r[j]
                            break
                if node_i is not None:
                    break
            
            if node_i and node_j:
                G.add_edge(node_i, node_j, weight=avg_weight)

        # Add self-loop edges for within-cluster connectivity
        for cluster, weights in within_cluster_weights.items():
            avg_weight = np.sum(weights)*amplify
            
            # Choose a representative node from the cluster to draw self-loop
            cluster_nodes = [node for node, c in node_to_cluster.items() if c == cluster]
            if cluster_nodes:
                center_node = cluster_nodes[0]  # Pick the first node in the cluster
                G.add_edge(center_node, center_node, weight=avg_weight)  # Self-loop
                
        threshold=1 # Set high threshold so that all other edges don't display

    
    # Draw edges with varying thickness based on connectivity strength
    edges = G.edges(data=True)
    strong_edges = [(i, j) for i, j, w in G.edges(data=True) if w['weight'] > threshold]
    nx.draw_networkx_edges(G, pos, edgelist=strong_edges, ax=axs[0],
        width=[d['weight']*edge_display for (u, v, d) in 
               G.edges(data=True) if d['weight']> threshold])  # Adjust thickness based on weight
    nx.draw_networkx_edges(G, pos, edgelist=strong_edges, ax=axs[1],
        width=[d['weight']*edge_display for (u, v, d) in 
               G.edges(data=True) if d['weight']> threshold])  # Adjust thickness based on weight

    # Draw labels for the nodes
    if labels:
        nx.draw_networkx_labels(G, pos, ax=axs[0], font_size=4)
    nx.draw_networkx_labels(G, pos, ax=axs[1], font_size=4)

    axs[0].set_title(f'colored by {coloring}', size=20)
    axs[1].set_title(f'colored by {clustering} clustering', size=20)
    fig.suptitle(f'{vers}, {clustering}', fontsize=40)

    axs[0].axis('off')
    axs[1].axis('off')
    axs[0].set_facecolor('none')
    axs[1].set_facecolor('none')
    fig.tight_layout()

    if control_list!=None:
        name=f'network_{metric}_{clustering}_{vers}_controlled'
    else:
        name=f'network_{metric}_{clustering}_{vers}'
        
    if clustering in ['spectralco', 'spectralbi', 'kmeans', 'kcenters', 'birch']:
        fig.savefig(Path(pth_dmn.parent, 'figs', 
                         f'{name}_k{k}_thr{threshold}_{layout}.pdf'), 
                    dpi=200, transparent=True)
        #np.save(Path(pth_dmn.parent, 'res', 
        #                 f'cluster_info_{metric}_{clustering}_{vers}_k{k}.npy'),
        #       cluster_info, allow_pickle=True)
    elif clustering in ['louvain', 'leiden']:
        fig.savefig(Path(pth_dmn.parent, 'figs', 
                    f'{name}_{resl}_thr{threshold}_{layout}_{coloring}.pdf'), 
                    dpi=200, transparent=True)
    else:
        fig.savefig(Path(pth_dmn.parent, 'figs', 
                    f'{name}_thr{threshold}_{layout}_{coloring}.pdf'), 
                    dpi=200, transparent=True)




def plot_all_connectivity_networks(mapping='Beryl', algo='umap_z', nclus=13, nd=2, edge_display=0.1, k_range=[2,3,4,5,6,7],
                                   resl_range=[1,1.01,1.02,1.03,1.04,1.05,1.06,1.07], threshold=0.9,
                                   vers='concat', layout='shell', coloring='Beryl', rerun=False):

    '''
    network style connectivity plots for all clustering measures & parameters
    '''

    fig, axs = plt.subplots(nrows=7, ncols=len(k_range),
                            figsize=(10,12), dpi=400)
    axs = axs.flatten()
    _,pal = get_allen_info()

    n = 0
    for clustering in ['hierarchy', 'hierarchy', 'louvain', 'leiden', 'birch',
                       'spectralco', 'spectralbi', 'kmeans']:
        if n==0:
            metric='wass'
            d = np.load(Path(pth_dmn, f'wasserstein_matrix_{nclus}_{vers}_nd{nd}.npy'), 
                        allow_pickle=True).flat[0]
        else:
            metric = algo
            d = get_reg_dist(algo=metric, vers=vers, rerun=rerun)            
            
        res0 = d['res']
        regs = d['regs']

        
        if n<2:
            res, regs_r, regs_c, cluster_info = clustering_on_connectivity_matrix(
                res0, regs, k=None, metric=metric, clustering=clustering)
            
            # Add graph nodes and edges with weights (connectivity strengths)
            G = nx.Graph()
            G.add_nodes_from(regs_r)
            for i in range(len(regs_r)):
                for j in range(i+1, len(regs_r)):  # Use only upper triangle for undirected graph
                    if res[i, j]!=0:  # Only add edges for nonzero connectivity
                        G.add_edge(regs_r[i], regs_r[j], weight=res[i, j])
                        
            # Define a layout for the nodes
            if layout=='spring':
                pos = nx.spring_layout(G)
            elif layout=='shell':
                pos = nx.shell_layout(G)
            elif layout=='kamada':
                pos = nx.kamada_kawai_layout(G)
            elif layout=='manual': #manually put nodes in the same cluster near each other
                pos = nx.kamada_kawai_layout(G)
                clusters = np.sort(cluster_info)
                for cluster in set(clusters):
                    cluster_nodes = regs_r[clusters==cluster]
                    cluster_center = np.mean([pos[node] for node in cluster_nodes], axis=0)
                    for node in cluster_nodes:
                        pos[node] = 0.1 * pos[node] + 0.9 * cluster_center
                
            # Plot graph
            if coloring=='Beryl':
                _, pal = get_allen_info()
                cols = [pal[reg] for reg in G.nodes()]
                nx.draw_networkx_nodes(G, pos, ax=axs[n], node_color=cols, node_size=2)
            else:
                cmap = cm.get_cmap('tab20')
                node_colors = [cmap(i) for i in np.sort(cluster_info)]
                nx.draw_networkx_nodes(G, pos, ax=axs[n], node_color=node_colors, node_size=4)

            edges = G.edges(data=True)
            strong_edges = [(i, j) for i, j, w in G.edges(data=True) if w['weight'] > threshold]
            # Adjust thickness based on weight
            nx.draw_networkx_edges(G, pos, edgelist=strong_edges, ax=axs[n], 
                                   width=[d['weight']*edge_display for (u, v, d) in 
                                          G.edges(data=True) if d['weight']> threshold]) 
            if n==0:
                axs[n].set_title(f'wass_{clustering}', fontsize=10)
            else:
                axs[n].set_title(f'{clustering}', fontsize=10)

            #nx.draw_networkx_labels(G, pos, ax=axs[n], font_size=1)
            axs[n].axis('off')
            n=n+1

        
        elif clustering in ['louvain', 'leiden']:
            for resl in resl_range:
                res, regs_r, regs_c, cluster_info = clustering_on_connectivity_matrix(
                    res0, regs, resl=resl, metric=metric, clustering=clustering)
                
                # Add graph nodes and edges with weights (connectivity strengths)
                G = nx.Graph()
                G.add_nodes_from(regs_r)
                for i in range(len(regs_r)):
                    for j in range(i+1, len(regs_r)):  # Use only upper triangle for undirected graph
                        if res[i, j]!=0:  # Only add edges for nonzero connectivity
                            G.add_edge(regs_r[i], regs_r[j], weight=res[i, j])
                            
                # Define a layout for the nodes
                if layout=='spring':
                    pos = nx.spring_layout(G)
                elif layout=='shell':
                    pos = nx.shell_layout(G)
                elif layout=='kamada':
                    pos = nx.kamada_kawai_layout(G)
                elif layout=='manual': #manually put nodes in the same cluster near each other
                    pos = nx.kamada_kawai_layout(G)
                    clusters = np.sort(cluster_info)
                    for cluster in set(clusters):
                        cluster_nodes = regs_r[clusters==cluster]
                        cluster_center = np.mean([pos[node] for node in cluster_nodes], axis=0)
                        for node in cluster_nodes:
                            pos[node] = 0.1 * pos[node] + 0.9 * cluster_center
                
                # Plot graph
                if coloring=='Beryl':
                    _, pal = get_allen_info()
                    cols = [pal[reg] for reg in G.nodes()]
                    nx.draw_networkx_nodes(G, pos, ax=axs[n], node_color=cols, node_size=2)
                else:
                    cmap = cm.get_cmap('tab20')
                    node_colors = [cmap(i) for i in np.sort(cluster_info)]
                    nx.draw_networkx_nodes(G, pos, ax=axs[n], node_color=node_colors, node_size=4)
                edges = G.edges(data=True)
                strong_edges = [(i, j) for i, j, w in G.edges(data=True) if w['weight'] > threshold]
            
                # Adjust thickness based on weight
                nx.draw_networkx_edges(G, pos, edgelist=strong_edges, ax=axs[n], 
                                   width=[d['weight']*edge_display for (u, v, d) in 
                                          G.edges(data=True) if d['weight']> threshold]) 
            
                axs[n].set_title(f'{clustering}, resl{resl}', fontsize=10)
                #nx.draw_networkx_labels(G, pos, ax=axs[n], font_size=1)
                axs[n].axis('off')
                n=n+1

        else:
            for k in k_range:
                res, regs_r, regs_c, cluster_info = clustering_on_connectivity_matrix(
                    res0, regs, k=k, metric=metric, clustering=clustering)
                
                # Add graph nodes and edges with weights (connectivity strengths)
                G = nx.Graph()
                G.add_nodes_from(regs_r)
                for i in range(len(regs_r)):
                    for j in range(i+1, len(regs_r)):  # Use only upper triangle for undirected graph
                        if res[i, j]!=0:  # Only add edges for nonzero connectivity
                            G.add_edge(regs_r[i], regs_r[j], weight=res[i, j])
                            
                # Define a layout for the nodes
                if layout=='spring':
                    pos = nx.spring_layout(G)
                elif layout=='shell':
                    pos = nx.shell_layout(G)
                elif layout=='kamada':
                    pos = nx.kamada_kawai_layout(G)
                elif layout=='manual': #manually put nodes in the same cluster near each other
                    pos = nx.kamada_kawai_layout(G)
                    clusters = np.sort(cluster_info)
                    for cluster in set(clusters):
                        cluster_nodes = regs_r[clusters==cluster]
                        cluster_center = np.mean([pos[node] for node in cluster_nodes], axis=0)
                        for node in cluster_nodes:
                            pos[node] = 0.1 * pos[node] + 0.9 * cluster_center
                
                # Plot graph
                if coloring=='Beryl':
                    _, pal = get_allen_info()
                    cols = [pal[reg] for reg in G.nodes()]
                    nx.draw_networkx_nodes(G, pos, ax=axs[n], node_color=cols, node_size=2)
                else:
                    cmap = cm.get_cmap('tab20')
                    node_colors = [cmap(i) for i in np.sort(cluster_info)]
                    nx.draw_networkx_nodes(G, pos, ax=axs[n], node_color=node_colors, node_size=4)
                edges = G.edges(data=True)
                strong_edges = [(i, j) for i, j, w in G.edges(data=True) if w['weight'] > threshold]
                
                # Adjust thickness based on weight
                nx.draw_networkx_edges(G, pos, edgelist=strong_edges, ax=axs[n], 
                                   width=[d['weight']*edge_display for (u, v, d) in 
                                          G.edges(data=True) if d['weight']> threshold]) 
            
            
                axs[n].set_title(f'{clustering}, k{k}', fontsize=10)
                #nx.draw_networkx_labels(G, pos, ax=axs[n], font_size=1)
                axs[n].axis('off')
                n=n+1
                
    
    fig.suptitle(vers)
    fig.tight_layout()
    fig.savefig(Path(pth_dmn.parent, 'figs', 
                    f'all_connectivity_networks_{coloring}_{vers}_{algo}_{threshold}_{layout}.pdf'), 
                dpi=400)



def plot_connec_networks_over_time(clustering, layout='manual', nclus=13, nd=2, edge_display=0.1,
                                   k=None, resl=None, tau=0.01, threshold=0.9, top_n=100, metric='umap_z', 
                                   coloring='Beryl', rerun=False):

    '''
    network style connectivity plots for all clustering measures & parameters
    '''

    fig, axs = plt.subplots(nrows=4, ncols=3,
                            figsize=(9,11), dpi=400)
    axs = axs.flatten()
    _,pal = get_allen_info()

    n = 0
    for vers in ['concat', 'resting', 'quiescence', 'quie-rest-diff-shifted', 'pre-stim-prior', 'stim_all', 'mistake', 
                 'stim_surp_con', 'stim_surp_incon', 'motor_init', 'fback1', 'fback0']:
        if metric=='wass':
            d = np.load(Path(pth_dmn, f'wasserstein_matrix_{nclus}_{vers}_nd{nd}.npy'), 
                        allow_pickle=True).flat[0]
        elif vers=='quie-rest-diff-shifted':
            d = get_quiescence_resting_diff(metric=metric, diff='shifted')
        else:
            d = get_reg_dist(algo=metric, vers=vers, rerun=rerun)            
            
        res0 = d['res']
        regs = d['regs']

        if layout=='shell':
            # order regions by canonical list 
            p = (Path(iblatlas.__file__).parent / 'beryl.npy')
            regs_can = br.id2acronym(np.load(p), mapping='Beryl')
            regs_r,reg_ord = [],[]
            for reg in regs_can:
                if reg in regs:
                    regs_r.append(reg)
                    reg_ord.append(np.where(regs==reg)[0][0])

            res=res0[reg_ord]
            #coloring='Beryl'
        else:
            res, regs_r, regs_c, cluster_info = clustering_on_connectivity_matrix(
                res0, regs, k=k, resl=resl, metric=metric, tau=tau, clustering=clustering)
            
        # Add graph nodes and edges with weights (connectivity strengths)
        G = nx.Graph()
        G.add_nodes_from(regs_r)
        for i in range(len(regs_r)):
            for j in range(i+1, len(regs_r)):  # Use only upper triangle for undirected graph
                if res[i, j]!=0:  # Only add edges for nonzero connectivity
                        G.add_edge(regs_r[i], regs_r[j], weight=res[i, j])
                        
        # Define a layout for the nodes
        if layout=='spring':
                pos = nx.spring_layout(G)
        elif layout=='shell':
                pos = nx.shell_layout(G)
        elif layout=='kamada':
                pos = nx.kamada_kawai_layout(G)
        elif layout=='manual': #manually put nodes in the same cluster near each other
                pos = nx.kamada_kawai_layout(G)
                clusters = np.sort(cluster_info)
                for cluster in set(clusters):
                    cluster_nodes = regs_r[clusters==cluster]
                    cluster_center = np.mean([pos[node] for node in cluster_nodes], axis=0)
                    for node in cluster_nodes:
                        pos[node] = 0.1 * pos[node] + 0.9 * cluster_center
                
        # Plot graph
        if coloring=='Beryl':
                _, pal = get_allen_info()
                cols = [pal[reg] for reg in G.nodes()]
                nx.draw_networkx_nodes(G, pos, ax=axs[n], node_color=cols, node_size=2)
        elif coloring=='dmn':
                cmap = cm.get_cmap('tab10')
                node_colors = [cmap(int(reg in dmn_regs)) for reg in G.nodes()]
                nx.draw_networkx_nodes(G, pos, ax=axs[n], node_color=node_colors, node_size=4)
        elif coloring=='cortical':
                pal={}
                for reg in G.nodes():
                    pal[reg] = 'white'
                    for cortical in cortical_regions.keys():
                        if reg in cortical_regions[cortical]:
                            pal[reg] = cortical_colors[cortical]
                node_colors = [pal[reg] for reg in G.nodes()]
                transparency = [(int(color!='white')+1)/2 for color in node_colors]
                nx.draw_networkx_nodes(G, pos, ax=axs[n], node_color=node_colors, 
                                       alpha=transparency, node_size=4)
        else:
                cmap = cm.get_cmap('tab20')
                node_colors = [cmap(i) for i in np.sort(cluster_info)]
                nx.draw_networkx_nodes(G, pos, ax=axs[n], node_color=node_colors, node_size=4)

        edges = G.edges(data=True)
        strong_edges = [(i, j) for i, j, w in G.edges(data=True) if w['weight'] > threshold]
        # Adjust thickness based on weight
        nx.draw_networkx_edges(G, pos, edgelist=strong_edges, ax=axs[n], 
                                   width=[d['weight']*edge_display for (u, v, d) in 
                                          G.edges(data=True) if d['weight']> threshold]) 
        axs[n].set_title(f'{vers}', fontsize=10)

        #nx.draw_networkx_labels(G, pos, ax=axs[n], font_size=1)
        axs[n].axis('off')
        n=n+1
                
    if layout=='shell':
        fig.suptitle(f'canonical ordering')
    else:
        fig.suptitle(f'{metric}_{clustering}_k{k}_resl{resl}')
    fig.tight_layout()
    if layout=='shell':
        fig.savefig(Path(pth_dmn.parent, 'figs', 
                    f'conn_networks_over_time_{metric}_{coloring}_{threshold}_{layout}.pdf'), 
                    dpi=400)
    else:
        fig.savefig(Path(pth_dmn.parent, 'figs', 
                    f'conn_networks_over_time_{metric}_{coloring}_{clustering}_k{k}resl{resl}_{threshold}_{layout}.pdf'), 
                    dpi=400)



def plot_avg_corr_with_dmn_regions(vers, metric='umap_z', rerun=False, cols_dictr=None,
                                   only_cortical=False):

    if cols_dictr==None:
        r_a = regional_group('Beryl', 'umap_z', vers='concat', nclus=13)
        cols_dictr = dict(list(Counter(zip(r_a['acs'], r_a['cols']))))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4), label=f'{vers}', dpi=150)
    cmap = cm.get_cmap('tab10')

    d = get_reg_dist(algo=metric, vers=vers, rerun=rerun)
    dmn_idx = [list(d['regs']).index(reg) for reg in dmn_regs]
    if only_cortical:
        cortical_list = np.concatenate(list(cortical_regions.values()))
        cortical_list = set(cortical_list) & set(d['regs'])
        cortical_idx = [list(d['regs']).index(reg) for reg in cortical_list]
        d['res'] = d['res'][:,cortical_idx]
        ndmn_idx = [list(d['regs']).index(reg) for reg in cortical_list
                    if reg not in dmn_regs]
        d['regs'] = d['regs'][cortical_idx]
    else:
        ndmn_idx = [list(d['regs']).index(reg) for reg in d['regs']
                    if reg not in dmn_regs]
    avg_corr_dmn = np.mean(d['res'][dmn_idx,:], axis=0)
    avg_corr_ndmn = np.mean(d['res'][ndmn_idx,:], axis=0)

    order = np.argsort(avg_corr_dmn)
    colors = [cmap(int(reg in dmn_regs)) for reg in d['regs'][order]]
    ax.scatter(d['regs'][order], avg_corr_dmn[order], color=colors, s=7, label='with dmn regs')
    ax.scatter(d['regs'][order], avg_corr_ndmn[order], color=colors, s=5, marker='v', label='with non-dmn regs')
    ax.set_xticks(np.arange(len(d['regs'])), d['regs'][order], rotation=90, fontsize=4)
    [t.set_color(i) for (i,t) in
        zip([cols_dictr[reg] for reg in d['regs'][order]],
        ax.xaxis.get_ticklabels())]
    ax.set_title(f'{vers}, average correlation per region', size=10)
    ax.legend()

    fig.tight_layout
    if only_cortical:
        fig.savefig(Path(one.cache_dir,'dmn', 'figs', 
                     f'{vers}_avg_corr_dmn_cortical.pdf'), dpi=150)
    else:
        fig.savefig(Path(one.cache_dir,'dmn', 'figs', 
                     f'{vers}_avg_corr_dmn.pdf'), dpi=150)



def plot_rastermap(feat='concat_z', exa = False, vers='concat', r=None, control=False,
                   norm_=False, mapping='Beryl', nclus=13, alpha=0.5, bg=False):
    """
    Function to run rastermap algorithm and plot a rastermap 
    with vertical segment boundaries 
    and labels positioned above the segments.

    Extra panel with colors of mapping.

    """

    if r==None:
        if control!=False:
            norm_ = 'False_control'+control
        r = regional_group(mapping, algo='concat_z', vers=vers, nclus=nclus, norm_=norm_)

    if feat == 'concat_z_no_mistake':

        # remove all mistake PETHs

        to_remove = ['stimLbRcR',
                    'stimLbLcR',
                    'stimRbLcL',
                    'stimRbRcL',
                    'sLbRchoiceR',
                    'sLbLchoiceR',
                    'sRbLchoiceL',
                    'sRbRchoiceL']
    
        # Extract relevant information from the data
        # Initialize an empty list to store the indices to keep
        keep_indices = []

        # Get segment names and lengths
        segment_names = list(r['len'].keys())
        segment_lengths = list(r['len'].values())

        # Track the current start index
        current_idx = 0

        # Identify indices to keep
        for i, segment in enumerate(segment_names):
            segment_length = segment_lengths[i]
            if segment not in to_remove:
                # Add indices of this segment to the keep list
                keep_indices.extend(range(current_idx, 
                                    current_idx + segment_length))
            # Update the current index for the next segment
            current_idx += segment_length

        # Filter the data to keep only the desired indices
        

        data = r['concat_z'][:, keep_indices]
        r[feat] = data
        # Update r['len'] to reflect the new structure
        r['len'] = {k: v for k, v in r['len'].items() if k not in to_remove}
        print('embedding rastermap ...')

        # Fit the Rastermap model
        model = Rastermap(bin_size=1).fit(data)

        r['isort'] = model.isort

    if feat == 'only_cortical':
        reg ='Isocortex'
        assert mapping == 'Cosmos'
        data = r['concat_z'][r['acs'] == reg, :]
        r[feat] = data
        print(f'embedding rastermap for {reg} cells only')
        # Fit the Rastermap model
        model = Rastermap(n_PCs=200, n_clusters=100,
                      locality=0.75, time_lag_window=5, bin_size=1).fit(data)
        r['isort'] = model.isort

    if exa:
        plot_cluster_mean_PETHs(r,mapping, feat)


    spks = r[feat]
    isort = r['isort']
    data = spks[isort]
    row_colors = np.array(r['cols'])[isort]  # Reorder colors by sorted index

    n_rows, n_cols = data.shape
    # Create a figure for the rastermap
    fig, ax = plt.subplots(figsize=(10, 8))

    # plot in greys, then overlay color (good for big picture)
    im = ax.imshow(data, vmin=0, vmax=1.5, cmap="gray_r",
                aspect="auto")

    if bg:   
        for i, color in enumerate(row_colors):
            ax.hlines(i, xmin=0, xmax=n_cols, colors=color, 
            lw=.01, alpha=alpha)

    ax.set_xlabel('time [bins]')    
    ax.set_ylabel('cells')

    ylim = ax.get_ylim()  

    # Plot vertical boundaries and add text labels
    h = 0
    #for segment in PETH_types_dict[vers]:
    for segment in r['len']:
        xv = h + r['len'][segment]  # Cumulative position of the vertical line
        ax.axvline(xv, linestyle='--', linewidth=1, color='grey')  # Draw vertical line
        
        # Add text label above the segment boundary
        midpoint = h + r['len'][segment] / 2  # Midpoint of the segment
        ax.text(midpoint, ylim[1] + 0.05 * (ylim[1] - ylim[0]), 
                peth_dict[segment], rotation=90, color='k', 
                fontsize=10, ha='center')  # Label positioned above the plot
        
        h += r['len'][segment]  # Update cumulative sum for the next segment


    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_facecolor('none')

    plt.tight_layout()  # Adjust the layout to prevent clipping

    plt.savefig(Path(pth_dmn.parent, 'figs',
                     f'rastermap_{mapping}_{vers}_norm{norm_}.png'), 
                dpi=200, transparent=True)


def get_cross_val_rastermap(control, mapping='Beryl', vers='concat', nclus=13, zscore=True):
    
        norm_ = 'False_control'+control+'_0'
        r = regional_group(mapping, algo='concat_z', vers=vers, nclus=nclus, norm_=norm_)
        norm1_ = 'False_control'+control+'_1'
        r1 = regional_group(mapping, algo='concat_z', vers=vers, nclus=nclus, norm_=norm1_)
    
        # Restrict to the overlapping set of cells to run sorting algorithm on 
        r_cells = [cell in r1['uuids'] for cell in r['uuids']]
        r['concat_z'] = r['concat_z'][r_cells]
        r['concat'] = r['concat'][r_cells]
        r['cols'] = np.array(r['cols'])[r_cells]
        r['pid'] = np.array(r['pid'])[r_cells]
        r['acs'] = np.array(r['acs'])[r_cells]
        # Run sorting algorithm on r
        if zscore:
            model = Rastermap(n_PCs=200, n_clusters=100, locality=0.75, 
                              time_lag_window=5, bin_size=1).fit(r['concat_z'])
        else:
            model = Rastermap(n_PCs=200, n_clusters=100, locality=0.75, 
                              time_lag_window=5, bin_size=1).fit(r['concat'])
            
        r['isort'] = model.isort

        # Cross validation: order r1 from the sorting result on r
        r1_cells = [cell in r['uuids'] for cell in r1['uuids']]
        r1['concat_z'] = r1['concat_z'][r1_cells]
        r1['concat'] = r1['concat'][r1_cells]
        r1['pid'] = np.array(r1['pid'])[r1_cells]
        r1['cols'] = np.array(r1['cols'])[r1_cells]
        r1['acs'] = np.array(r1['acs'])[r1_cells]
        # r1['cols'] = r['cols']
        r1['isort'] = model.isort

        # Unify the cell ids in both sets
        r['uuids'] = r['uuids'][r_cells]
        r1['uuids'] = r1['uuids'][r1_cells]

        return r, r1, norm_, norm1_


def plot_regional_network_participation(reg, control_list, vers='concat', 
                                        cmap_name='Blues', clustering='louvain',
                                        plot_raster=True, save=True, annotate=False,
                                        shuffling=False):
    '''
    Get regional network participation for a network, and plot swanson & rastermap
    reg: the focus region for network participation (=1 always for this region)
    control_list: list of control versions to include for participation ratio calculation
    '''

    regs_ordered, clusters = get_cluster_membership_across_controls(control_list, 
                                                                    vers=vers, 
                                                                    shuffling=shuffling)    
    regs=regs_ordered[False]
    
    # Get network participation ratio
    participation=np.zeros(len(regs))
    for control in control_list:
        clus_id=int(clusters[control][regs_ordered[control]==reg])
        regs_in_clus=regs_ordered[control][clusters[control]==clus_id]
        participation=participation + [reg in regs_in_clus for reg in regs]

    participation=participation/np.max(participation)

    # Plot swanson
    cmap = cm.get_cmap(cmap_name)#.reversed()
    fig,ax = plt.subplots(figsize=(8,3.34))
    acronyms = regs
    values = participation
    plot_swanson_vector(acronyms, values, annotate=annotate, ax=ax,
                        annotate_list=regs, empty_color='silver',
                        cmap=cmap, fontsize=5)
    
    ax.set_axis_off()
    ax.set_facecolor('none')
    if save:
        fig.savefig(Path(pth_dmn.parent, 'figs',
                    f'{vers}_{clustering}_{reg}network_participation.pdf'),
                    transparent=True, format="pdf", dpi=150, bbox_inches="tight")

    if plot_raster:
        # Get data for rastermap
        feat='concat_z'
        r = regional_group(mapping='Beryl', algo=feat, vers='concat0')
        spks = r[feat]
        isort = r['isort']
        data = spks[isort]
    
        # Set color based on network participation colormap
        acs = r['acs']
        r['cols'] = [cmap(participation)[regs==reg] for reg in acs]
        r['cols'] = [entry if entry.size > 0 else np.array([cmap(0)]) for entry in r['cols']]
        row_colors = np.array(r['cols'])[isort]

        # Plot rastermap
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(data, vmin=0, vmax=1.5, cmap="gray_r", aspect="auto")
        n_rows, n_cols = data.shape
        ylim = ax.get_ylim()  
        h=0
        for segment in r['len']:
            xv = h + r['len'][segment]  # Cumulative position of the vertical line
            ax.axvline(xv, linestyle='--', linewidth=1, color='grey')  # Draw vertical line
        
            # Add text label above the segment boundary
            midpoint = h + r['len'][segment] / 2  # Midpoint of the segment
            ax.text(midpoint, ylim[1] + 0.05 * (ylim[1] - ylim[0]), 
                    peth_dict[segment], rotation=90, color='k', 
                    fontsize=10, ha='center')  # Label positioned above the plot
        
            h += r['len'][segment]  # Update cumulative sum for the next segment


        for i, color in enumerate(row_colors):
            ax.hlines(i, xmin=0, xmax=n_cols, colors=color, 
            lw=.01, alpha=0.5)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()  # Adjust the layout to prevent clipping

        plt.savefig(Path(pth_dmn.parent, 'figs',
                     f'rastermap_{reg}network_shuffle{shuffling}.png'), dpi=200)


def plot_swansons_clusters(control, algo='umap_z', vers='concat', k=4, tau=0.01, clustering='louvain'):
    d = get_reg_dist(algo=algo, vers=vers, control=control)
    res = d['res']
    regs = d['regs']
    _, _, regs, cluster_info = clustering_on_connectivity_matrix(res, regs, 
                                                                 clustering=clustering, 
                                                                 k=k, tau=tau)
    if clustering=='ICA_cutoff':
        clusters = np.arange(0, k, 1)
    else: 
        clusters = np.sort(cluster_info)
        
    for n in set(clusters):
        fig,ax = plt.subplots(figsize=(8,3.34), dpi=150)
        if clustering=='ICA_cutoff':
            acronyms = regs[cluster_info[:,n]==1]
        else:
            acronyms = regs[clusters==n]
        if len(acronyms)<4:
            continue
        values = np.ones(len(acronyms))
        plot_swanson_vector(acronyms, values, annotate=True, ax=ax,
                            annotate_list=regs, empty_color='silver',
                            cmap='coolwarm', fontsize=5)
        ax.set_axis_off()
        fig.savefig(Path(pth_dmn.parent, 'figs',
                     f'{vers}_{algo}_{clustering}_ctr{control}_{n}.png'), dpi=200)


def rgb_to_hex(rgb):
    return '#{:02X}{:02X}{:02X}'.format(
        int(rgb[0] * 255),
        int(rgb[1] * 255),
        int(rgb[2] * 255)
    )

def get_hierarchy_label_and_color(reg):
    '''
    Returns a (label, color) tuple for a region. Label is plain text, color is hex.
    '''
    a, _ = get_allen_info()
    a['Beryl'] = br.id2acronym(a['id'].values, mapping='Beryl')
    a['Cosmos'] = br.id2acronym(a['id'].values, mapping='Cosmos')

    cdict = Counter(a['Cosmos'])
    cdict.pop('void', None)
    cdict.pop('root', None)
    cosmos_ids = br.acronym2id(list(cdict.keys()))

    idp = a['structure_id_path'][a['Beryl'] == reg].values[0]
    idp = idp.split('/')[-6:-1]
    cos_i = next(i for i, x in enumerate(idp) if int(x) in cosmos_ids)
    idp = idp[cos_i:]
    idp_int = list(map(int, idp))

    label = ' / '.join([
        f"{get_name(br.id2acronym(x))} ({br.id2acronym(x)[0]})"
        for x in idp_int
    ])
    _, pal = get_allen_info()
    col = rgb_to_hex(pal[reg])
    return label, col


def plot_regional_hist(plot_data, name, raster_type=None, start=None, end=None, normalized=True, full_name=False):
    '''
    Plot histogram of selected group of cells' beryl regions
    start & end: two arrays of start/end times for lists of cells to be included
    '''
    isort = plot_data['isort']
    cell_regs = plot_data['acs'][isort]

    if start is None or end is None:
        start=raster_types[raster_type]['start']
        end=raster_types[raster_type]['end']

    regs=[]
    for i in range(len(start)):
        regs.append(cell_regs[start[i]:end[i]])
    regs=np.concatenate(regs)
    
    filtered_data = [item for item in regs if item not in ['void', 'root']]
    counts = Counter(filtered_data)
    #normalize by full regional counts
    if normalized:
        full_counts = Counter(cell_regs)
        normalized_counts = {
            key: counts[key] / full_counts[key]
            for key in counts if full_counts[key] >= 50  # Only keep if more than 50 cells in full counts
        }
    else:
        normalized_counts=counts

    #sort categories by frequency in descending order
    sorted_items = sorted(normalized_counts.items(), key=lambda x: x[1], reverse=True)
    categories, frequencies = zip(*sorted_items)  #unpack sorted items
    _, pal = get_allen_info()
    colors=[pal[reg] for reg in categories]

    if full_name:
        fig, ax = plt.subplots(figsize=(18, 14), dpi=150)
        ax.barh(categories, frequencies, color=colors)
        ax.set_yticks([])
        for i, reg in enumerate(categories):
            label, color = get_hierarchy_label_and_color(reg)
            ax.text(
                -0.02, i,  # just below x-axis (tune as needed)
                label,
                color=color,
                # rotation=90,
                ha='right',
                va='center',
                fontsize=5,
                transform=ax.get_yaxis_transform()
            )
        ax.set_title(raster_type, fontsize=20)
        # ax.text(1.02, 1.0, raster_type, transform=ax.transAxes, 
        #         rotation=270, ha='left', va='top', fontsize=20)
        ax.set_xlabel('normalized regional participation', fontsize=15)
    else:
        fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
        ax.bar(categories, frequencies, color=colors)
        ax.set_xticklabels(categories, rotation=90, fontsize=4)
        #ax.set_title(name, fontsize=20)
        ax.set_xlim(left=-4)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_facecolor('none')
    #ax.set_ylim(0, 0.25)

    if full_name:
        name = 'SI_' + raster_type
        plt.subplots_adjust(left=0.55, right=0.98)
    if normalized:
        fig.savefig(Path(pth_dmn.parent, 'figs', f'{name}_regional_hist.pdf'), 
                    dpi=200, transparent=True)
    else:
        fig.savefig(Path(pth_dmn.parent, 'figs', f'{name}_regional_hist_count.pdf'), 
                    dpi=200, transparent=True)

    plt.close()
    return frequencies


def plot_subset_raster(raster_data, raster_type=None, name=None, 
                       start=None, end=None, ex_bin='stimLbLcL'):
    '''
    Plot raster and avg peth for interested subset(s) of cells
    start & end: two arrays of start/end times for lists of cells to be included
    '''
    if start is None or end is None:
        start=raster_types[raster_type]['start']
        end=raster_types[raster_type]['end']

    spks = raster_data['concat_z']
    isort = raster_data['isort']
    data = spks[isort]

    plot_data=[]
    for i in range(len(start)):
        plot_data.append(data[start[i]:end[i]])
    plot_data=np.concatenate(plot_data)

    fig, axs = plt.subplots(3,1, figsize=(30, 8))
    axs[0].imshow(plot_data, 
                  vmin=0, vmax=1.5, 
                  cmap="gray_r",
                  aspect="auto") # Plot raster

    y = np.mean(plot_data, axis=0) # Plot avg peth
    axs[1].plot(y, linewidth=0.75)
    #colors=['black', 'blue', 'cyan', 'green', 'yellow']
    #for i in range(len(plot_data)//5):
    #    ploty = np.mean(plot_data[i*5:i*5+5], axis=0)
    #    axs[1].plot(ploty[648:(648+96)], linewidth=0.5, color=colors[i])
        #axs[1].plot(plot_data[i*5][648:(648+96)], linewidth=0.5, color=colors[i])
        #axs[1].plot(plot_data[i*5+1][648:(648+96)], linewidth=0.5, color=colors[i])
        #axs[1].plot(plot_data[i*5+2][648:(648+96)], linewidth=0.5, color=colors[i])
        #axs[1].plot(plot_data[i*5+2+1][648:(648+96)], linewidth=0.5, color=colors[i])
        #axs[1].plot(plot_data[i*5+4][648:(648+96)], linewidth=0.5, color=colors[i])
    
    ylim = axs[0].get_ylim()
    h=0
    data_lengths = raster_data['len']
    start = sum_for_key(data_lengths, ex_bin)
    x = np.linspace(0, 0.2, data_lengths[ex_bin])
    axs[2].plot(x, y[start:(start+data_lengths[ex_bin])], linewidth=2)
    for segment in data_lengths:
        xv = h + data_lengths[segment]  # Cumulative position of the vertical line
        axs[0].axvline(xv, linestyle='--', linewidth=0.2, color='grey')  # Draw vertical line
        axs[1].axvline(xv, linestyle='--', linewidth=0.2, color='grey')
        
        # Add text label above the segment boundary
        midpoint = h + data_lengths[segment] / 2  # Midpoint of the segment
        axs[0].text(midpoint, ylim[1] + 0.05 * (ylim[1] - ylim[0]),  
                peth_dict[segment], rotation=90, color='k', 
                fontsize=10, ha='center')  # Label positioned above the plot
        
        h += data_lengths[segment]  # Update cumulative sum for the next segment

    # Remove top and right spines
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)

    axs[0].set_facecolor('none')
    axs[1].set_facecolor('none')

    for spine in axs[0].spines.values():
        spine.set_linewidth(0.2)

    axs[0].yaxis.set_ticks([0, len(plot_data)])
    axs[0].xaxis.set_ticks([])
    axs[0].set_yticklabels(axs[0].get_yticks(), fontsize=25)
    axs[1].yaxis.set_ticks([])
    axs[1].xaxis.set_ticks([])
    axs[1].set_xticklabels([])

    fig.savefig(Path(pth_dmn.parent, 'figs', f'{name}_subset_raster.pdf'), 
                dpi=200, transparent=True)



def plot_regional_hist_comparison(cell_regs, indices0, indices1, name, full_name=False,
                                  top_cells=2000, normalized=True, full_cell_regs=None):
    '''
    Plot histogram of most differential cells for two conditions in comparison=[condition0, condition1]
    indices0, indices1: sorted most differential cells of the two conditions
    to-do: plot this on original data? or on other sets of controls?
    '''
    indices = set(indices0[:top_cells]) | set(indices1[:top_cells])
    regs = cell_regs[list(indices)]
    filtered_data = [item for item in regs if item not in ['void', 'root']]
    counts = Counter(filtered_data)
    #normalize by full regional counts
    if full_cell_regs is None:
        full_cell_regs = cell_regs
    full_counts = Counter(full_cell_regs)
    if normalized:
        normalized_counts = {
            key: counts[key] / full_counts[key]
            for key in counts if full_counts[key] >= 50  # Only keep if more than 50 cells in full counts
        }
    else:
        normalized_counts = counts

    #sort categories by frequency in descending order
    sorted_items = sorted(normalized_counts.items(), key=lambda x: x[1], reverse=True)
    categories, frequencies = zip(*sorted_items)  #unpack sorted items
    _, pal = get_allen_info()
    colors=[pal[reg] for reg in categories]

    if full_name:
        fig, ax = plt.subplots(figsize=(18, 14), dpi=150)
        ax.barh(categories, frequencies, color=colors)
        ax.set_yticks([])
        for i, reg in enumerate(categories):
            label, color = get_hierarchy_label_and_color(reg)
            ax.text(
                -0.02, i,  # just below x-axis (tune as needed)
                label,
                color=color,
                # rotation=90,
                ha='right',
                va='center',
                fontsize=5,
                transform=ax.get_yaxis_transform()
            )
        if 'integrator' in name:
            title = 'top 50% prior diff cells in integrators'
        elif 'move_init_movement' in name:
            title = 'top 50% prior diff cells in movement cells'
        else:
            title = name
        ax.set_title(title, fontsize=20)
        # ax.text(1.02, 1.0, name, transform=ax.transAxes, 
        #         rotation=270, ha='left', va='top', fontsize=20)
        ax.set_xlabel('normalized regional participation', fontsize=15)
    else:
        fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
        ax.bar(categories, frequencies, color=colors)
        ax.set_xticklabels(categories, rotation=90, fontsize=4)
        #ax.set_title(name, fontsize=20)
        ax.set_xlim(left=-4)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_facecolor('none')
    #ax.set_ylim(0, 0.25)

    if full_name:
        name = 'SI_' + name
        plt.subplots_adjust(left=0.55, right=0.98)
            
    fig.savefig(Path(pth_dmn.parent, 'figs', 
                     f'{name}_diff_cells_hist.pdf'), 
                dpi=200, transparent=True)


def plot_two_subsets_raster(plot_data, indices0, indices1, labels, save_name, top_cells=None,
                            data_lengths=data_lengths):
    '''
    Plot rastermap sorted separately for two conditions,
    also plot the two avg peths, plus a zoomed-in panel (first 90 points).

    Parameters
    ----------
    plot_data : np.ndarray
        Cell × time matrix.
    indices0, indices1 : list or np.ndarray
        Cell indices for the two subsets.
    labels : list
        Names for the two conditions [name0, name1].
    save_name : str
        File prefix for saving.
    top_cells : int, optional
        Limit to top N cells (sorting and plotting restricted to these).
    data_lengths : dict
        Segment lengths for drawing vertical lines.
    '''
    name0, name1 = labels[0], labels[1]

    # convert indices
    indices0 = np.asarray(indices0)
    indices1 = np.asarray(indices1)

    # if top_cells, restrict before fitting
    if top_cells is not None:
        idx0_for_fit = indices0[:top_cells]
        idx1_for_fit = indices1[:top_cells]
    else:
        idx0_for_fit = indices0
        idx1_for_fit = indices1

    # Rastermap fits on restricted subsets
    model0 = Rastermap(bin_size=1).fit(plot_data[idx0_for_fit])
    model1 = Rastermap(bin_size=1).fit(plot_data[idx1_for_fit])

    ordered0 = idx0_for_fit[model0.isort]
    ordered1 = idx1_for_fit[model1.isort]

    # fixed figure size
    fig, axs = plt.subplots(4, 1, figsize=(20, 10), sharex=False)

    # compute truncated columns: 45 per segment
    start = 0
    cols_to_show = []
    for seg_len in data_lengths.values():
        cols_to_show.extend(range(start, start + min(45, seg_len)))
        start += seg_len
    cols_to_show = np.array(cols_to_show)

    # raster plots (fixed vmin/vmax)
    axs[0].imshow(plot_data[ordered0][:, cols_to_show], vmin=0, vmax=1.5, cmap="gray_r", aspect="auto")
    axs[1].imshow(plot_data[ordered1][:, cols_to_show], vmin=0, vmax=1.5, cmap="gray_r", aspect="auto")
    axs[0].set_xlim(0, len(cols_to_show)+5)
    axs[1].set_xlim(0, len(cols_to_show)+5)

    # Average PETHs (restricted to 45 pts per segment)
    y  = plot_data[ordered0][:, cols_to_show].mean(axis=0)
    y1 = plot_data[ordered1][:, cols_to_show].mean(axis=0)
    axs[2].plot(np.arange(len(cols_to_show)), y, linewidth=0.75, label=name0)
    axs[2].plot(np.arange(len(cols_to_show)), y1, linewidth=0.75, label=name1)
    axs[2].legend(frameon=False)
    axs[2].set_xlim(0, len(cols_to_show)+5)

    # Zoomed-in panel: still show first 96 datapoints overall
    n_short = min(96, y.shape[0], y1.shape[0])
    axs[3].plot(np.arange(n_short), y[:n_short], linewidth=0.75, label=name0)
    axs[3].plot(np.arange(n_short), y1[:n_short], linewidth=0.75, label=name1)
    axs[3].set_xlim(0, n_short)
    axs[3].legend(frameon=False)

    # Draw vertical lines and labels (adjusted for truncated 45-col segments)
    h = 0
    ylim = axs[0].get_ylim()
    for segment, seg_len in data_lengths.items():
        xv = h + min(45, seg_len)
        axs[0].axvline(xv, linestyle='--', linewidth=0.5, color='grey')
        axs[1].axvline(xv, linestyle='--', linewidth=0.5, color='grey')
        axs[2].axvline(xv, linestyle='--', linewidth=0.5, color='grey')

        midpoint = h + min(45, seg_len) / 2
        axs[0].text(midpoint, ylim[1] + 0.05 * (ylim[1] - ylim[0]),
                    peth_dict[segment], rotation=90, color='k',
                    fontsize=10, ha='center')
        h += min(45, seg_len)

    # Aesthetics
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('none')
    for spine in axs[0].spines.values():
        spine.set_linewidth(0.2)

    axs[0].yaxis.set_ticks([0, len(ordered0)])
    axs[0].xaxis.set_ticks([])
    axs[0].set_yticklabels(axs[0].get_yticks(), fontsize=25)

    axs[1].yaxis.set_ticks([0, len(ordered1)])
    axs[1].xaxis.set_ticks([])
    axs[1].set_yticklabels(axs[1].get_yticks(), fontsize=25)

    axs[3].yaxis.set_ticks([])  # keep x-axis for zoom panel

    # Save figure
    fig.savefig(Path(pth_dmn.parent, 'figs',
                     f'{save_name}_diff_raster.pdf'),
                dpi=200, transparent=True)


def sum_for_key(data, key, after=False):
    if key not in data.keys():
        return 'key not in data'
        
    total = 0
    for k, v in data.items():
        if k == key:
            if after:
                total += v
            break  # Stop when we reach the given key
        total += v
    return total


def sort_by_diff_two_conditions(r, comparison, start_times=None, end_times=None, data_lengths=data_lengths,
                                feat='concat_z', mean_baseline=False, window_len=None):
    '''
    sort data by how much the cells respond differently to two conditions
    e.g. [blockL, blockR] in comparison
    return two sets of indices: 
        if mean_baseline:
            indices0 sort from largest blockL-avg(blockL,blockR) to smallest; indices1 for blockR
        else:
            indices0 sort from largest blockL to smaller; indices1 for blockR
    '''
    data = r[feat]
    if start_times==None:
        start_times={key: sum_for_key(data_lengths, key) for key in comparison}
    if end_times==None and window_len is not None:
        start_times={key: sum_for_key(data_lengths, key) + 10 for key in comparison}
        end_times={key: sum_for_key(data_lengths, key) + window_len for key in comparison}
    elif end_times==None:
        end_times={key: sum_for_key(data_lengths, key, after=True) for key in comparison}
    
    cond0 = np.mean(data[:,start_times[comparison[0]]:end_times[comparison[0]]], axis=1)
    cond1 = np.mean(data[:,start_times[comparison[1]]:end_times[comparison[1]]], axis=1)
    if mean_baseline:
        mean_res=(cond0+cond1)/2
        cond0=cond0-mean_res
        cond1=cond1-mean_res
    indices0=sorted(range(len(cond0)), key=lambda i: cond0[i], reverse=True)
    indices1=sorted(range(len(cond1)), key=lambda i: cond1[i], reverse=True)

    diff=np.abs(cond0-cond1)
    indices_diff=sorted(range(len(diff)), key=lambda i: diff[i], reverse=True)
    return indices0, indices1, diff, indices_diff


def float_array_to_rgba(img_float):
    cmap = cm._colormaps['gray_r']
    rgba = cmap(img_float)  # Returns float32 RGBA
    return (rgba * 255).astype(np.uint8)  # Convert to uint8
    
def save_rastermap_pdf(start=[0], end=None, name=None, 
                       feat='concat_z', mapping='kmeans', bg=False, cv=False):

    if cv:
        # load Ari's files
        r_test = np.load(Path(pth_dmn, 'cross_val_test.npy'),
                         allow_pickle=True).flat[0]
        r_train = np.load(Path(pth_dmn, 'cross_val_train.npy'), 
                          allow_pickle=True).flat[0]
        
        spks = r_test[feat]
        isort = r_train['isort'] 
        data = spks[isort]

        bg = False

        plot_data=[]
        if end==None:
            end = [len(data)]
        for i in range(len(start)):
            plot_data.append(data[start[i]:end[i]])
        data=np.concatenate(plot_data)


    else:
        r = regional_group(mapping)
        spks = r[feat]
        isort = r['isort'] 
        data = spks[isort]
    
    # Normalize and convert to RGBA image
    norm_data = data - data.min()
    norm_data = norm_data / norm_data.max()
    image_rgba = float_array_to_rgba(norm_data)
    image_rgba[..., 3] = 255

    if bg:
        # RGBA rows
        row_colors = np.array(r['cols'])[isort]  

        # Blend row colors onto grayscale image
        alpha_overlay = 0.2  # adjust intensity of color overlay

        for i in range(image_rgba.shape[0]):
            r_c, g_c, b_c, _ = row_colors[i] * 255  # get color per row
            overlay = np.array([r_c, g_c, b_c], dtype=np.uint8)

            # Blend overlay with original grayscale values (RGB only)
            image_rgba[i, :, :3] = (
                (1 - alpha_overlay) * image_rgba[i, :, :3] +
                alpha_overlay * overlay[None, :]
            ).astype(np.uint8)

    # Convert to RGB (drop alpha) for PDF compatibility
    img = Image.fromarray(image_rgba[..., :3], mode='RGB')
    if name==None:
        name = f'_{mapping}' if bg else ''
        if cv:
            name = 'cross_val'
    img.save(Path(pth_dmn.parent, 'figs', f"rastermap{name}.pdf"), "PDF")


def plot_cluster_similarity_comparison(control_list, vers, min_cluster_size=3,
                                       rerun_shuffle=True, algo='umap_z', mapping='Beryl',
                                       show_colorbar=False): # Default changed to False
    """
    Computes and compares cosine similarity matrices between clusters
    with and without shuffling. Each matrix is ordered via separate hierarchical clustering.
    Dendrograms are removed. X and Y tick labels are removed for matrices, and X and Y labels are added.
    Optionally displays a color bar.

    Parameters:
    - control_list: list of control inputs to be passed to get_cluster_membership_across_controls
    - vers: network identifier, e.g. 'concat', 'quiescence'
    - min_cluster_size: minimum number of regions required in a cluster to include it in the analysis
    - show_colorbar: If True, displays a color bar for the similarity matrices. Defaults to False.
    """

    if rerun_shuffle:
        # For the shuffling case, apply same shuffling of regional labels
        # across controls before computing reg similarity
        results = smooth_dist(algo=algo, mapping=mapping, vers=vers,
                              shuffling=True, control_list=control_list)
        for control in control_list:
            pth_ = Path(one.cache_dir, 'dmn',
                        f'{algo}_{mapping}_{vers}_control{control}_smooth_shuffled.npy')
            res, regs = results[control]
            d = {'res': res, 'regs': regs}
            np.save(pth_, d, allow_pickle=True)

    def build_cluster_matrix(regs_ordered, clusters):
        all_regions = sorted(set(np.concatenate(list(regs_ordered.values()))))
        region_to_index = {region: idx for idx, region in enumerate(all_regions)}

        cluster_vectors = []
        cluster_labels = []

        for version in regs_ordered:
            version_regions = regs_ordered[version]
            version_clusters = clusters[version]
            n_clusters = np.max(version_clusters) + 1

            for cluster_id in range(n_clusters):
                members = version_regions[version_clusters == cluster_id]
                if len(members) < min_cluster_size:
                    continue

                vec = np.zeros(len(all_regions))
                for region in members:
                    vec[region_to_index[region]] = 1
                cluster_vectors.append(vec)
                cluster_labels.append(f"{version}_c{cluster_id}")

        # Ensure cluster_vectors is not empty before stacking
        if not cluster_vectors:
            return np.array([]).reshape(len(all_regions), 0), []
        return np.stack(cluster_vectors, axis=1), cluster_labels

    # Get cluster memberships for shuffled and non-shuffled regional labels
    regs_no_shuffle, clusters_no_shuffle = get_cluster_membership_across_controls(
        control_list, vers=vers, shuffling=False
    )
    regs_shuffle, clusters_shuffle = get_cluster_membership_across_controls(
        control_list, vers=vers, shuffling=True
    )

    # Build matrices
    X_no_shuffle, labels_no_shuffle = build_cluster_matrix(regs_no_shuffle, clusters_no_shuffle)
    X_shuffle, labels_shuffle = build_cluster_matrix(regs_shuffle, clusters_shuffle)

    if X_no_shuffle.shape[1] == 0 or X_shuffle.shape[1] == 0:
        print("No clusters met the minimum size requirement in one of the conditions.")
        return

    # Compute similarity matrices
    sim_no_shuffle = cosine_similarity(X_no_shuffle.T)
    sim_shuffle = cosine_similarity(X_shuffle.T)

    # Hierarchical clustering for ordering
    linkage_no_shuffle = linkage(1 - sim_no_shuffle, method='average')
    linkage_shuffle = linkage(1 - sim_shuffle, method='average')
    order_no_shuffle = leaves_list(linkage_no_shuffle)
    order_shuffle = leaves_list(linkage_shuffle)

    sim_no_shuffle_ordered = sim_no_shuffle[order_no_shuffle][:, order_no_shuffle]
    sim_shuffle_ordered = sim_shuffle[order_shuffle][:, order_shuffle]
    labels_no_shuffle_ordered = [labels_no_shuffle[i].replace("False_c", "full data_c") for i in order_no_shuffle]
    labels_shuffle_ordered = [labels_shuffle[i].replace("False_c", "full data_c") for i in order_shuffle]

    # Plot similarity matrices without dendrograms
    fig_width = 10
    if show_colorbar:
        fig_width += 2
    fig = plt.figure(figsize=(fig_width, 7))
    gs = GridSpec(1, 2, width_ratios=[1, 1])

    # Similarity matrices
    ax_matrix1 = fig.add_subplot(gs[0, 0])
    im1 = ax_matrix1.imshow(sim_no_shuffle_ordered, cmap='viridis', vmin=0, vmax=1)
    ax_matrix1.set_xticks([])
    ax_matrix1.set_yticks([])
    ax_matrix1.set_xlabel('All clusters from different trial splits', fontsize=12) # Updated x-label
    ax_matrix1.set_ylabel('All clusters from different trial splits', fontsize=12) # Updated y-label
    ax_matrix1.set_title("True Region Labels", fontsize=16)


    ax_matrix2 = fig.add_subplot(gs[0, 1])
    im2 = ax_matrix2.imshow(sim_shuffle_ordered, cmap='viridis', vmin=0, vmax=1)
    ax_matrix2.set_xticks([])
    ax_matrix2.set_yticks([])
    ax_matrix2.set_xlabel('All clusters from different trial splits', fontsize=12) # Updated x-label
    ax_matrix2.set_ylabel('All clusters from different trial splits', fontsize=12) # Updated y-label
    ax_matrix2.set_title("Shuffled Region Labels", fontsize=16)


    ax_matrix1.set_facecolor('none')
    ax_matrix2.set_facecolor('none')

    # Colorbar - conditionally added
    if show_colorbar:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(im1, cax=cbar_ax)
        cbar_ax.set_facecolor('none')
        plt.tight_layout(rect=[0, 0, 0.91, 0.95])
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.suptitle(f"{vers}", fontsize=20)
    plt.savefig(Path(pth_dmn.parent, 'figs',
                     f'cluster_sim_comparison_{vers}.pdf'),
                transparent=True, dpi=200)
    plt.close()


def plot_log_hist_freqs(frequencies, name):
    x = np.arange(1, len(frequencies) + 1)
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5)) # 1 row, 2 columns
    
    # Plot 1: Log-Log plot
    ax1.loglog(x, frequencies, marker='o', markersize=2)
    ax1.set_xlabel('X-axis (Log Scale)')
    ax1.set_ylabel('Frequencies (Log Scale)')
    ax1.set_title('Log-Log Plot of Frequencies')
    # ax1.grid(True, which="both", ls="-", color='0.7')
    
    # Plot 2: Log-Y (Semi-Log) plot
    ax2.semilogy(x, frequencies, marker='o', markersize=2)
    ax2.set_xlabel('X-axis (Linear Scale)')
    ax2.set_ylabel('Frequencies (Log Scale)')
    ax2.set_title('Semi-Log (Log-Y) Plot of Frequencies')
    # ax2.grid(True, which="both", ls="-", color='0.7')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.suptitle(name)

    plt.savefig(Path(pth_dmn.parent, 'figs', f'{name}_regional_hist_logscale.pdf'), 
                    dpi=100, transparent=True)
    
