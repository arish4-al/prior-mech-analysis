import pandas as pd
import numpy as np
from pathlib import Path

from one.api import ONE
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions
from ibllib.atlas.plots import plot_swanson_vector
#from brainwidemap.manifold.state_space_bwm import (plot_traj_and_dist,
#                                                   plot_all)

from statsmodels.stats.multitest import multipletests

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_hex
# from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
# import dataframe_image as dfi
# from matplotlib.gridspec import GridSpec

from PIL import Image


#sns.set(font_scale=1.5)
#sns.set_style('ticks')

ba = AllenAtlas()
br = BrainRegions()
one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international', silent=True)
          
# get pooled results here
meta_pth = Path(one.cache_dir, 'meta')
meta_pth.mkdir(parents=True, exist_ok=True)          

pth_res = Path(one.cache_dir, 'manifold', 'res') 
pth_res.mkdir(parents=True, exist_ok=True)
pth_avg = Path(one.cache_dir, 'manifold', 'avgs') 
pth_avg.mkdir(parents=True, exist_ok=True)

save_dir = '/Users/ariliu/Desktop/ibl-figures'


# set default plot style
import matplotlib.ticker as mticker
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['svg.fonttype'] = 'none'  # keep text as text

def set_default_plot_style(nbins_x=4, nbins_y=5, labelsize=15):
    """Set default style for all matplotlib plots, with transparent backgrounds."""

    def _apply_style(ax):
        # locators
        # ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=nbins_x))
        # ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=nbins_y))
        # remove top/right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # label and tick fontsize
        ax.xaxis.label.set_size(labelsize)
        ax.yaxis.label.set_size(labelsize)
        ax.tick_params(axis='x', labelsize=labelsize)
        ax.tick_params(axis='y', labelsize=labelsize)
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


def swanson_to_beryl_hex(beryl_acronym,br):
    beryl_id = br.id[br.acronym==beryl_acronym]
    rgb = br.get(ids=beryl_id)['rgb'][0].astype(int)
    return '#' + rgb_to_hex((rgb[0],rgb[1],rgb[2]))

def beryl_to_cosmos(beryl_acronym,br):
    beryl_id = br.id[br.acronym==beryl_acronym]
    return br.get(ids=br.remap(beryl_id, source_map='Beryl', 
                  target_map='Cosmos'))['acronym'][0]

def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb


def get_name(brainregion):
    regid = br.id[np.argwhere(br.acronym == brainregion)][0, 0]
    return br.name[np.argwhere(br.id == regid)[0, 0]]


def get_cmap_(meta_split):
    '''
    for each split, get a colormap defined by Yanliang,
    updated by Chris
    '''
    dc = {}
    if 'regtype' in meta_split or 'move_shape' in meta_split:
        base_colors = ['#ffffb3', '#ffed6f', 
                          '#feda7e', '#feb23f', '#d55607']
        base_cmap = LinearSegmentedColormap.from_list("orange_yellow", base_colors)
        dc[meta_split] = base_cmap(np.linspace(0, 1, 256))
        dc[meta_split][0] = [int("57", 16)/255, int("C1", 16)/255, int("EB", 16)/255, 1.0]  # base blue (#57C1EB)
        n = dc[meta_split].shape[0]
        idx_0 = 0
        idx_015 = int(round(0.15 * (n - 1)))
        # smooth gradient from solid blue → very light transparent blue
        for i, frac in enumerate(np.linspace(0, 1, idx_015 - idx_0 + 1)):
            r = (1 - frac) * (87/255) + frac * (240/255)
            g = (1 - frac) * (193/255) + frac * (250/255)
            b = (1 - frac) * (235/255) + frac * (255/255)
            a = (1 - frac) * 1.0 + frac * 0.01  # more transparent and lighter blue
            dc[meta_split][idx_0 + i] = [r, g, b, a]
    elif 'stim_d' in meta_split:
        # dc[meta_split] = ["#EAF4B3","#D5E1A0", "#A3C968",
        #                   "#86AF40", "#517146","#33492E"]
        dc[meta_split] = ["#B3D4F4", "#A0C3E1", "#68A3C9", 
                          "#4079AF", "#2E5171", "#1E3449"]
    elif 'choice_d' in meta_split:
        dc[meta_split] = ["#F8E4AA","#F9D766","#E8AC22",
                          "#DA4727","#96371D"]
    elif 'sc' in meta_split:
        base_colors = ['#A7E3FA', '#ffffb3', '#ffed6f', 
                          '#feda7e', '#feb23f']
        base_cmap = LinearSegmentedColormap.from_list("orange_yellow", base_colors)
        dc[meta_split] = base_cmap(np.linspace(0, 1, 256))
        dc[meta_split][0] = [int("57", 16)/255, int("C1", 16)/255, int("EB", 16)/255, 1.0]  # base blue (#57C1EB)
        n = dc[meta_split].shape[0]
        idx_0 = 0
        idx_015 = int(round(0.15 * (n - 1)))
        # smooth gradient from solid blue → very light transparent blue
        for i, frac in enumerate(np.linspace(0, 1, idx_015 - idx_0 + 1)):
            r = (1 - frac) * (87/255) + frac * (240/255)
            g = (1 - frac) * (193/255) + frac * (250/255)
            b = (1 - frac) * (235/255) + frac * (255/255)
            a = (1 - frac) * 1.0 + frac * 0.01  # more transparent and lighter blue
            dc[meta_split][idx_0 + i] = [r, g, b, a]
    else:
        dc[meta_split] = ["#D0CDE4","#998DC3","#6159A6",
                          "#42328E", "#262054"]


    return LinearSegmentedColormap.from_list("mycmap", dc[meta_split])


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
    # 'stim_duringstim': ['stim_choice_l', 'stim_choice_r'], 
    # 'choice_duringchoice': ['choice_stim_l', 'choice_stim_r'],
    # 'choice_duringstim': ['choice_duringstim_l', 'choice_duringstim_r'],
    # 'stim_duringchoice': ['stim_duringchoice_l', 'stim_duringchoice_r'],
    'stim_duringstim1': ['stim_block_l', 'stim_block_r'],
    'stim_duringstim1_act': ['stim_block_l_act', 'stim_block_r_act'],
    'stim_duringstim_short': ['stim_choice_r_block_r_short', 'stim_choice_l_block_l_short', 
                         'stim_choice_r_block_l_short', 'stim_choice_l_block_r_short'], 
    'stim_duringstim_short_act': ['stim_choice_r_block_r_short_act', 'stim_choice_l_block_l_short_act', 
                         'stim_choice_r_block_l_short_act', 'stim_choice_l_block_r_short_act'], 
}


def compute_amp_slope(timeframe, n=20):
    '''
    for stim/choice splits, locate the peak of the amplitude and fit the slope of the last n points
    used later for region type classification
    n: number of last points to fit the slope
    '''

    splits = run_align[timeframe]
    if len(splits) == 1:
        combined_name = splits[0]
        combined_regde_name = f'{combined_name}_regde'
    else:
        combined_name = 'combined_'+"_".join(splits)
        combined_regde_name = 'combined_regde_'+"_".join(splits)

    # run for combined results
    d = np.load(Path(pth_res, f'{combined_name}.npy'), 
                    allow_pickle=True).flat[0]  
    regs = [x for x in d]
    for reg in regs:
        r = np.load(Path(pth_res, f'{combined_regde_name}.npy'), allow_pickle=True).flatten()[0][reg][0]
        slope = np.polyfit(np.linspace(0, 0.15, len(r)), r, 1)[0]
        d[reg]['amp_slope'] = slope
        
        slope_last = np.polyfit(np.arange(n), r[-n:], 1)[0]
        d[reg]['slope_last'] = slope_last

        amp_loc = np.argmax(r)
        d[reg]['amp_loc'] = amp_loc

        slope_last_5 = np.polyfit(np.arange(5), r[-5:], 1)[0]
        d[reg]['slope_last_5'] = slope_last_5

        slope_last_10 = np.polyfit(np.arange(10), r[-10:], 1)[0]
        d[reg]['slope_last_10'] = slope_last_10

        # --- moving 5-bin amplitude check ---
        if len(r) >= 5:
            # mean amplitude of each 5-bin window (stride 1)
            win5_means = np.convolve(r, np.ones(5)/5.0, mode='valid')
            last5_mean = win5_means[-1]
            prev5_means = win5_means[:-1]  # all windows before the last
            if prev5_means.size > 0:
                prev5_max = np.max(prev5_means)
                is_global_max = int(last5_mean > prev5_max)
            else:
                # only one window exists; treat as global max
                prev5_max = np.nan
                is_global_max = 1
        else:
            last5_mean = np.nan
            prev5_max = np.nan
            is_global_max = 0

        d[reg]['amp_last5_mean'] = float(last5_mean)
        d[reg]['amp_prev5_max_mean'] = float(prev5_max) if not np.isnan(prev5_max) else np.nan
        d[reg]['amp_last5_is_global_max'] = int(is_global_max)
        
    np.save(Path(pth_res, f'{combined_name}.npy'), d, allow_pickle=True)


def fdr_combined(timeframe, ptype='p_euc', sigl=0.05):
    
    '''
    FDR correction, based on regions (same as in bwm analysis)
    results saved as '{ptype}_c'
    '''
    splits = run_align[timeframe]
    if len(splits) == 1:
        combined_name = splits[0]
    else:
        combined_name = 'combined_'+"_".join(splits)

    # run correction for combined results
    d = np.load(Path(pth_res, f'{combined_name}.npy'), 
                    allow_pickle=True).flat[0]
    regs = [x for x in d]
    pvals = [d[x][ptype] for x in d]
    _, pvals_c, _, _ = multipletests(pvals, sigl, method='fdr_bh')

    for i in range(len(regs)):
        d[regs[i]][f'{ptype}_c'] = pvals_c[i]

    np.save(Path(pth_res, f'{combined_name}.npy'), d, allow_pickle=True)

    # run for each split
    if len(splits) > 1:
        for split in splits:
            d = np.load(Path(pth_res, f'{split}.npy'), 
                        allow_pickle=True).flat[0]
            regs = [x for x in d]
            pvals = [d[x][ptype] for x in d]
            _, pvals_c, _, _ = multipletests(pvals, sigl, method='fdr_bh')
        
            for i in range(len(regs)):
                d[regs[i]][f'{ptype}_c'] = pvals_c[i]
        
            np.save(Path(pth_res, f'{split}.npy'), d, allow_pickle=True)


def load_combined_data(timeframe, dist='de'):
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


def compute_p_value(timeframe, ptype='p_mean', alpha=0.05, dist='de'):

    # load data
    splits = run_align[timeframe]
    d, r_all, combined_name, combined_regd_name = load_combined_data(timeframe, dist=dist)
    if 'duringchoice' in timeframe and ptype == 'p_gain':
        # load corresponding stim aligned data to identify offset
        timeframe_s = timeframe.replace('duringchoice', 'duringstim')
        _, r_all_s, _, _ = load_combined_data(timeframe_s, dist=dist)

    # run for combined results
    regs = [x for x in d]
    for reg in regs:
        r = r_all[reg]
        if len(splits) > 1: # for combined splits the control curves are all stored in r[1]
            r = np.concatenate([r[0].reshape(1, -1), r[1]], axis=0)
        if ptype == 'p_amp':
            amplitude = np.max(r, axis=1) - np.min(r, axis=1)
            p_val = np.mean(amplitude >= amplitude[0])
        elif ptype == 'p_mean':
            p_val = np.mean(np.mean(r, axis=1) >= np.mean(r[0]))
        elif ptype == 'p_max':
            p_val = np.mean(np.max(r, axis=1) >= np.max(r[0]))
        elif ptype == 'p_offset':
            mean_first5 = np.mean(r[:, :5], axis=1)
            p_val = np.mean(mean_first5 >= mean_first5[0])
            effect = mean_first5[0] - np.mean(mean_first5[1:])
            d[reg][f'{ptype}_effect'] = effect
        elif ptype == 'p_gain':
            if 'duringchoice' in timeframe:
                # load corresponding stim aligned data to identify offset
                r_s = r_all_s[reg]
                if len(splits) > 1: # for combined splits the control curves are all stored in r[1]
                    r_s = np.concatenate([r_s[0].reshape(1, -1), r_s[1]], axis=0)
                mean_first5 = np.mean(r_s[:, :5], axis=1)
            else:
                mean_first5 = np.mean(r[:, :5], axis=1)
            p_val_offset = np.mean(mean_first5 >= mean_first5[0])
            if p_val_offset < alpha:
                offset = mean_first5[0] - np.mean(mean_first5[1:])
            else:
                offset = 0
            r_shifted = r[0] - offset
            r_new = r[:, 4:]
            r_new[0] = r_shifted[4:]
            max_idx = np.argmax(r_new[0])
            p_val = np.mean(np.mean(r_new, axis=1) >= np.mean(r_new[0]))
            effect = np.max(r_new[0]) - np.mean(r_new[:, max_idx])
            d[reg][f'{ptype}_effect'] = effect
        else:
            raise ValueError(f"Invalid ptype: {ptype}")
        d[reg][f'{ptype}'] = p_val
        
    np.save(Path(pth_res, f'{combined_name}.npy'), d, allow_pickle=True)

    # run for each split
    if len(splits) > 1: 
        for split in splits:
            d = np.load(Path(pth_res, f'{split}.npy'), 
                        allow_pickle=True).flat[0]
            regs = [x for x in d]
            for reg in regs:
                r = np.load(Path(pth_res, f'{split}_reg{dist}.npy'), allow_pickle=True).flat[0][reg]
                if ptype == 'p_amp':
                    amplitude = np.max(r, axis=1) - np.min(r, axis=1)
                    p_val = np.mean(amplitude >= amplitude[0])
                elif ptype == 'p_mean':
                    p_val = np.mean(np.mean(r, axis=1) >= np.mean(r[0]))
                elif ptype == 'p_max':
                    p_val = np.mean(np.max(r, axis=1) >= np.max(r[0]))
                elif ptype == 'p_offset':
                    mean_first5 = np.mean(r[:, :5], axis=1)
                    p_val = np.mean(mean_first5 >= mean_first5[0])
                    effect = mean_first5[0] - np.mean(mean_first5[1:])
                    d[reg][f'{ptype}_effect'] = effect
                elif ptype == 'p_gain':
                    if 'duringchoice' in timeframe:
                        split_s = split.replace('duringchoice', 'choice')
                        split_s = split_s.replace('stim', 'duringstim')
                        try:
                            r_s = np.load(Path(pth_res, f'{split_s}_reg{dist}.npy'), allow_pickle=True).flat[0][reg]
                            mean_first5 = np.mean(r_s[:, :5], axis=1)
                        except:
                            mean_first5 = np.mean(r[:, :5], axis=1)
                    else:
                        mean_first5 = np.mean(r[:, :5], axis=1)
                    p_val_offset = np.mean(mean_first5 >= mean_first5[0])
                    if p_val_offset < alpha:
                        offset = mean_first5[0] - np.mean(mean_first5[1:])
                    else:
                        offset = 0
                    r_shifted = r[0] - offset
                    r_new = r[:, 4:]
                    r_new[0] = r_shifted[4:]
                    p_val = np.mean(np.mean(r_new, axis=1) >= np.mean(r_new[0]))
                    effect = np.max(r_new[0]) - np.mean(r_new[:, max_idx])
                    d[reg][f'{ptype}_effect'] = effect
                else:
                    raise ValueError(f"Invalid ptype: {ptype}")
                d[reg][f'{ptype}'] = p_val
            
            np.save(Path(pth_res, f'{split}.npy'), d, allow_pickle=True)
            
            
def manifold_to_csv(split, sigl, p_type, sample=True):

    '''
    reformat results for table
    '''
    
    # mapping = 'Beryl'

    columns = ['region', #'name', 
               p_type, 'p_gain', 'p_offset', 'amp_euc', 'lat_euc', 'p_gain_effect', 'p_offset_effect',
               'amp_slope', 'slope_last', 'amp_loc', 'slope_last_5', 'slope_last_10', 'amp_last5_is_global_max'
               #'amp_euc_can','lat_euc_can', 'amp_eucn_can', 'lat_eucn_can'
              ]
               
    d = np.load(Path(pth_res, f'{split}.npy'), 
                    allow_pickle=True).flat[0]
    
    # use a sample to align the regions if sample file exists (easier to align here than later when plotting tables!!)
    if sample:
        sample_path = Path(pth_res, f'act_block_only.csv')
        if not sample_path.exists():
            regs = [x for x in d]
        else:
            sample = pd.read_csv(sample_path)
            regs = sample.region
    else:
        regs = [x for x in d]

    r = []   
    for reg in regs:
        if reg not in d:
            r.append([reg, None, None, None])
            continue
        r.append([reg, d[reg][p_type], d[reg]['p_gain'], d[reg]['p_offset'],
                    d[reg]['amp_euc'], d[reg]['lat_euc'], d[reg]['p_gain_effect'], d[reg]['p_offset_effect'],
                    d[reg]['amp_slope'], d[reg]['slope_last'], 
                    d[reg]['amp_loc'], d[reg]['slope_last_5'],
                    d[reg]['slope_last_10'], d[reg]['amp_last5_is_global_max']
                    ])
    
    df  = pd.DataFrame(data=r, columns=columns)
    
    df['significant'] = (df[p_type] <= sigl).astype(int)
    df.to_csv(Path(pth_res, f'{split}.csv'), index=False) 
    return df

def manifold_to_csv_old(meta_split, sigl, p_type):

    '''
    reformat results for table
    '''
                   
    splits = meta_splits[meta_split]
    # sample = pd.read_pickle('~/Downloads/stim.pkl')
    sample = pd.read_csv(Path(pth_res, f'intertrial.csv'))

    for split in splits:
        r = []
        d = np.load(Path(pth_res,f'{split}.npy'),
                    allow_pickle=True).flat[0] 
        
        for reg in sample.region:
            if reg not in d:
                r.append([reg, None, None, None])
                continue        
        #for reg in d:
            r.append([reg, d[reg][p_type],
                      d[reg]['amp_euc'], d[reg]['lat_euc'],
                     ])
        
        df  = pd.DataFrame(data=r,
                           columns=['region',
                                    f'p_{split}', f'amp_{split}',
                                    f'lat_{split}'])
        
        df[f'{split}_significant'] = df[f'p_{split}']<=sigl
        df.to_csv(Path(pth_res, f'{split}.csv'), index=False)



# Plotting functions

def plot_regional_distance(reg, time, combined=True, ptype='p_mean_c', dist='de', alpha=0.05, plot_p_per_time=True, 
                           plot_gain=True, plot_offset=True, show_y=False, ylim=None, yticks=None, p_mean_early=False):
    """
    Plot regional distance for a given region over time.
    Args:
        reg: str, region name
        time: str, timeframe name for combined splits or split name for single split
        combined: bool, whether to use combined data
        ptype: str, p-value type, e.g. p_mean, p_amp, p_max
        alpha: float, significance level
        plot_p_per_time: bool, whether to plot p-values per time
        plot_gain: bool, whether to plot gain effect as a second curve and histogram
    """

    # Get times for x axis
    if 'duringchoice' in time:
        times = np.linspace(-150, 0, 72)
    elif 'duringstim1' in time or 'short' in time:
        times = np.linspace(0, 80, 42)
    elif 'duringstim' in time:
        times = np.linspace(0, 150, 72)
    else:
        times = np.linspace(-400, -100, 144)

    # Load data and plot trajectories
    if combined:
        d_all, r_all, _, _ = load_combined_data(time, dist=dist)
        r = r_all[reg]
        if 'intertrial0' not in time:
            r = np.concatenate([r[0].reshape(1, -1), r[1]], axis=0)
        splits = run_align[time]
        r = r / len(splits)
        d = d_all[reg]
    else:
        r = np.load(Path(pth_res, f'{time}_reg{dist}.npy'), allow_pickle=True).flatten()[0][reg]
        d = np.load(Path(pth_res, f'{time}.npy'), allow_pickle=True).flatten()[0][reg]

    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(6, 4), dpi=250,
                            gridspec_kw={'width_ratios': [6, 1]})
    for i in range(1, 40):
        if 'duringstim' in time and plot_offset:
            axs[0].plot(times[:5], r[i][:5], c='#5f7ea3', alpha=0.5, linewidth=0.5)
            axs[0].plot(times[4:], r[i][4:], c='gray', alpha=0.2, linewidth=0.5)
        else:
            axs[0].plot(times, r[i], c='gray', alpha=0.2, linewidth=0.5)

    # if 'duringstim' in time and plot_offset:
    #     axs[0].plot(times[:5], r[0][:5], c='blue', linewidth=1, linestyle='--')
    #     axs[0].plot(times[4:], r[0][4:], c='black', linewidth=1, linestyle='--')
    # else:
    if plot_gain:
        axs[0].plot(times, r[0], c='black', linewidth=1, linestyle='--')
    else:
        axs[0].plot(times, r[0], c='black', linewidth=1, linestyle='-')
        # axs[0].scatter(times, r[0], c='black', s=15, zorder=3)

    if 'duringstim' in time and p_mean_early:
        mean_early = np.mean(r[:, :42], axis=1)
        p_val_mean_early = np.mean(mean_early >= mean_early[0])
        axs[0].text(0.55, 0.85, f'p_mean_early_stim: {p_val_mean_early:.4f}', transform=axs[0].transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right', color='tab:blue')

    # Subtract offset to examine gain effect
    if plot_gain:
        # for choice, use stim aligned data to identify offset
        if 'duringchoice' in time:
            if combined:
                time_s = time.replace('duringchoice', 'duringstim')
                d_all_s, r_all_s, _, _ = load_combined_data(time_s, dist=dist)
                d_s = d_all_s[reg]
                r_s = r_all_s[reg]
                r_s = np.concatenate([r_s[0].reshape(1, -1), r_s[1]], axis=0)
                splits_s = run_align[time_s]
                r_s = r_s / len(splits_s)
            else:
                time_s = time.replace('duringchoice', 'choice')
                time_s = time_s.replace('stim', 'duringstim')
                d_s = np.load(Path(pth_res, f'{time_s}.npy'), allow_pickle=True).flatten()[0][reg]
                r_s = np.load(Path(pth_res, f'{time_s}_regde.npy'), allow_pickle=True).flatten()[0][reg]
            d['p_offset'] = d_s['p_offset']
            mean_first5 = np.mean(r_s[:, :5], axis=1)
            # print(mean_first5[0])
        else:
            mean_first5 = np.mean(r[:, :5], axis=1)
            # print(mean_first5[0])
        # if offset is significant, subtract offset to examine gain effect
        if d['p_offset'] < alpha:
            offset = mean_first5[0] - np.mean(mean_first5[1:])
            r_shifted = r[0] - offset
            # print(offset, r_shifted[0], r[0])
            axs[0].plot(times, r_shifted, c='black', linewidth=1)
        else: 
            r_shifted = r[0]
        p_val_gain = d['p_gain']

    # Plot p-values per time
    if plot_p_per_time:
        p_per_time = np.mean(r >= r[0], axis=0)
        # # Perform Bonferroni FDR correction
        # corrected_p_values = multipletests(p_per_time, alpha=0.05, method='fdr_bh')[1]
        # p_per_time = corrected_p_values

        ax2 = axs[0].twinx()
        ax2.plot(times, p_per_time, color='blue', linestyle='--', linewidth=1, label='p per time')
        ax2.set_ylim([0, 1])
        ax2.set_ylabel('p', fontsize=10, color='blue')
        ax2.tick_params(axis='y', labelcolor='blue', labelsize=8)

        sig_mask = p_per_time <= alpha
        axs[0].scatter(times[sig_mask], np.full(np.sum(sig_mask), axs[0].get_ylim()[0]),
                       marker='v', color='blue', s=20, zorder=5)

    # Plot histograms for all times, initial offset, and gain
    ls_main = '--' if plot_gain else '-'
    if 'p_mean' in ptype:
        axs[1].hist(np.mean(r[1:], axis=1), density=True, bins=20,
                    color='silver', orientation='horizontal')
        axs[1].axhline(y=np.mean(r[0]), c='black', linestyle=ls_main)
    elif 'p_amp' in ptype:
        amplitude = np.max(r, axis=1) - np.min(r, axis=1)
        axs[1].hist(amplitude[1:], density=True, bins=20,
                    color='silver', orientation='horizontal')
        axs[1].axhline(y=amplitude[0], c='black', linestyle=ls_main)
    elif 'p_max' in ptype:
        axs[1].hist(np.max(r[1:], axis=1), density=True, bins=20,
                    color='silver', orientation='horizontal')
        axs[1].axhline(y=np.max(r[0]), c='black', linestyle=ls_main)
    # Offset
    if 'duringstim' in time and plot_offset:
        axs[1].hist(np.mean(r[1:, :5], axis=1), density=True, bins=20,
                    color='#5f7ea3', orientation='horizontal', alpha=0.5)
        axs[1].axhline(y=np.mean(r[0, :5]), c='blue', linestyle='--')
    # Gain
    if plot_gain and d['p_offset'] < alpha:
        # axs[1].hist(np.mean(r[1:, 4:], axis=1), density=True, bins=20,
        #             color='#c8a2d6', orientation='horizontal', alpha=0.5)
        axs[1].axhline(y=np.mean(r_shifted[4:]), c='black', linestyle='-')

    # Print p-values
    p_val = d[ptype]
    axs[0].text(0.1, 0.97, f'p_val {p_val:.4f}', transform=axs[0].transAxes,
                color='red' if p_val <= alpha else 'black', fontsize=20, ha='left', va='top')
    if 'duringstim' in time and plot_offset:
        p_val_first5 = d['p_offset']
        axs[0].text(0.45, 0.15, f'p_val_offset {p_val_first5:.4f}', transform=axs[0].transAxes,
                    color='red' if p_val_first5 <= alpha else 'blue', fontsize=16, ha='left', va='top')
    if plot_gain:
        axs[0].text(0.1, 0.85, f'p_val_gain {p_val_gain:.4f}', transform=axs[0].transAxes,
                    color='red' if p_val_gain <= alpha else 'purple', fontsize=16, ha='left', va='top')

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.set_xticklabels([])
        if not show_y:
            ax.set_yticklabels([])
            axs[0].spines['left'].set_visible(False)
            axs[0].tick_params(axis='y', left=False)        
        ax.set_facecolor('none')
        ax.tick_params(labelsize=15)
    if 'duringstim1' in time or 'short' in time:
        axs[0].set_xticks(np.linspace(times[0], times[-1], 3))
    else:
        axs[0].set_xticks(np.linspace(times[0], times[-1], 4))
    axs[1].spines['bottom'].set_visible(False)
    axs[1].tick_params(axis='y', left=False, labelleft=False)
    axs[1].tick_params(axis='x', bottom=False, labelbottom=False)

    # axs[0].set_ylim(1.02, 1.9)
    # if p_mean_early:
    #     axs[0].set_xlim(-5, 82)
    #     axs[0].set_xticks([0, 40, 80])
    if ylim is not None:
        axs[0].set_ylim(ylim)
    if yticks is not None:
        axs[0].set_yticks(yticks)
    fig.tight_layout()
    fig.savefig(f'{save_dir}/{reg}_{time}_{ptype}_dist.svg', 
                transparent=True)


def plot_average_distance_over_regions(regs, timewindow, name, alpha=0.05, 
                                       ptype='p_mean_c', plot_p_per_time=True, plot_gain=True, show_y=False,
                                       ylim=None, yticks=None, return_mean=False):
    
    # get times for x axis
    if 'duringchoice' in timewindow:
        times = np.linspace(-150, 0, 72)
    elif 'short' in timewindow:
        times = np.linspace(0, 80, 42)
    elif 'duringstim' in timewindow:
        times = np.linspace(0, 150, 72)
    else:
        times = np.linspace(-400, -100, 144)
       
    # color for the mean curve
    if 'move' in name:
        c = 'tomato'
    elif 'int' in name:
        c = 'gold'
    else:
        c = 'blue'

    # load data
    d_all, r_all, _, _ = load_combined_data(timewindow)
    # get cell count from the first split
    splits = run_align[timewindow]
    split = splits[0] 
    d_split = np.load(Path(pth_res, f'{split}.npy'), allow_pickle=True).flatten()[0]

    all_regs_r = 0
    all_regs_d = []
    all_regs_cell_num = 0

    for reg in regs:
        try:
            r = r_all[reg]
            if len(splits) > 1:
                r = np.concatenate([r[0].reshape(1, -1), r[1]], axis=0)
            r = r / len(splits)
            d = d_all[reg]
            cell_num = d_split[reg]['nclus']
            all_regs_cell_num += cell_num
            all_regs_r += r * cell_num
            # all_regs_r += (r ** 2 * cell_num) # scale by cell count
            all_regs_d.append(d)
        except:
            print(f"Error loading data for region: {reg}")
            continue

    # average across regions (shape: [n_samples, time])
    r_avg = all_regs_r / all_regs_cell_num
    # r_avg = (all_regs_r / all_regs_cell_num) ** 0.5

    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(6, 4), dpi=250,
                            gridspec_kw={'width_ratios': [6, 1]})
    for i in range(1, min(40, r_avg.shape[0])):
        axs[0].plot(times, r_avg[i], c='gray', alpha=0.2, linewidth=0.5)

    axs[0].plot(times, r_avg[0], c=c, linewidth=2, linestyle='--')

    # --- inset: r_avg[0] - mean(r_avg[1:], axis=0) with shuffle band ---
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    diff_curve = r_avg[0] - np.mean(r_avg[1:], axis=0)

    # compute shuffle-centered distribution per time bin
    shuf_centered = r_avg[1:] - np.mean(r_avg[1:], axis=0, keepdims=True)
    low_band = np.percentile(shuf_centered, 100 * (alpha / 2), axis=0)
    high_band = np.percentile(shuf_centered, 100 * (1 - alpha / 2), axis=0)

    ax_ins = inset_axes(axs[0], width="20%", height="20%", loc='upper left', borderpad=2.8)
    ax_ins.fill_between(times, low_band, high_band, color='gray', alpha=0.3, linewidth=0)
    ax_ins.plot(times, diff_curve, linewidth=1, c=c)

    # ax_ins.axhline(0, linestyle='--', linewidth=0.8, alpha=0.6)
    ax_ins.set_xticks([])
    ax_ins.set_ylim(-0.05, 0.2)
    ax_ins.set_yticks([0, 0.1, 0.2])
    ax_ins.set_facecolor('none')
    for spine in ("top","right"):
        ax_ins.spines[spine].set_visible(False)

    # p-value curve (per time) and aggregate
    p_per_time = np.mean(r_avg >= r_avg[0], axis=0)

    # Calculate p_val for the mean of the first 5 datapoints
    mean_first5 = np.mean(r_avg[:, :5], axis=1)
    p_val_first5 = np.mean(mean_first5 >= mean_first5[0])
    
    # Subtract the mean of the first 5 datapoints to examine gain effect
    if plot_gain:
        if p_val_first5 < alpha:
            offset = mean_first5[0] - np.mean(mean_first5[1:])
            r_shifted = r_avg[0] - offset
            axs[0].plot(times, r_shifted, c=c, linewidth=2, linestyle='-')
        else: 
            r_shifted = r_avg[0]
        p_val_gain = np.mean(np.mean(r_avg[:, 4:], axis=1) >= np.mean(r_shifted[4:]))

    if 'p_mean' in ptype:
        p_val = np.mean(np.mean(r_avg, axis=1) >= np.mean(r_avg[0]))
    elif 'p_amp' in ptype:
        amp = np.max(r_avg, axis=1) - np.min(r_avg, axis=1)
        p_val = np.mean(amp >= amp[0])
    elif 'p_max' in ptype:
        p_val = np.mean(np.max(r_avg, axis=1) >= np.max(r_avg[0]))
    else:
        raise ValueError(f"Unsupported ptype: {ptype}")

    if plot_p_per_time:
        ax2 = axs[0].twinx()
        ax2.plot(times, p_per_time, color='blue', linestyle='--', linewidth=1, label='p per time')
        ax2.set_ylim([0, 1])
        ax2.set_ylabel('p', fontsize=10, color='blue')
        ax2.tick_params(axis='y', labelcolor='blue', labelsize=8)

        sig_mask = p_per_time <= alpha
        axs[0].scatter(times[sig_mask], np.full(np.sum(sig_mask), axs[0].get_ylim()[0]),
                       marker='v', color='blue', s=20, zorder=5)

    if 'p_mean' in ptype:
        axs[1].hist(np.mean(r_avg[1:], axis=1), density=True, bins=20,
                    color='silver', orientation='horizontal')
        axs[1].axhline(y=np.mean(r_avg[0]), c='black')
    elif 'p_amp' in ptype:
        amp = np.max(r_avg, axis=1) - np.min(r_avg, axis=1)
        axs[1].hist(amp[1:], density=True, bins=20,
                    color='silver', orientation='horizontal')
        axs[1].axhline(y=amp[0], c='black')
    elif 'p_max' in ptype:
        axs[1].hist(np.max(r_avg[1:], axis=1), density=True, bins=20,
                    color='silver', orientation='horizontal')
        axs[1].axhline(y=np.max(r_avg[0]), c='black')
    # # Offset histogram
    # if 'duringstim' in timewindow:
    #     axs[1].hist(np.mean(r[1:, :5], axis=1), density=True, bins=20,
    #                 color='#5f7ea3', orientation='horizontal', alpha=0.5)
    #     axs[1].axhline(y=np.mean(r[0, :5]), c='blue')
    # # Gain histogram
    # axs[1].hist(np.max(r_sub[1:], axis=1), density=True, bins=20,
    #             color='#c8a2d6', orientation='horizontal', alpha=0.5)
    # axs[1].axhline(y=np.max(r_sub[0]), c='purple')

    # Print p-values
    axs[0].text(0.2, 0.97, f'p_val {p_val:.4f}', transform=axs[0].transAxes,
                color='red' if p_val <= alpha else 'black', fontsize=8, ha='left', va='top')
    axs[0].text(0.7, 0.1, f'p_val_offset {p_val_first5:.4f}', transform=axs[0].transAxes,
                color='red' if p_val_first5 <= alpha else 'black', fontsize=8, ha='left', va='top')
    if plot_gain:
        axs[0].text(0.5, 0.97, f'p_val_gain {p_val_gain:.3f}', transform=axs[0].transAxes,
                    color='red' if p_val_gain <= alpha else 'purple', fontsize=8, ha='left', va='top')

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if not show_y:
            ax.set_yticklabels([])
            axs[0].spines['left'].set_visible(False)
            axs[0].tick_params(axis='y', left=False)
        ax.set_facecolor('none')
        ax.tick_params(labelsize=20)

    axs[0].set_xticks(np.linspace(times[0], times[-1], 4))
    axs[1].spines['bottom'].set_visible(False)
    axs[1].tick_params(axis='y', left=False, labelleft=False)
    axs[1].tick_params(axis='x', bottom=False, labelbottom=False)
    if ylim is not None:
        axs[0].set_ylim(ylim)
    if yticks is not None:
        axs[0].set_yticks(yticks)

    fig.tight_layout()
    fig.savefig(f'{save_dir}/{name}_{split}_{ptype}_dist_avg.svg', transparent=True)
    if return_mean:
        return r_avg[0]

# helper: selection-aware debiasing for a vector d_t against per-timepoint controls
def _debias_selected_vector(
    d_vec, controls_mat, alpha=0.05, max_iter=50, tol=1e-8,
    apply_only_when_selected=True, selection_mode="mean_over_time"
):
    """
    Debias a selected/thresholded statistic vector using empirical or parametric
    truncated-normal correction.

    Parameters
    ----------
    d_vec : array, shape (T,)
        Observed real data statistic over time.
    controls_mat : array, shape (K, T)
        Shuffle/null samples for the same statistic.
    alpha : float
        Significance level for selection threshold.
    max_iter : int
        Max iterations for parametric fixed-point solver (fallback mode).
    tol : float
        Convergence tolerance for the fixed-point solver.
    apply_only_when_selected : bool
        Whether to apply correction only if the data passed the selection criterion.
    selection_mode : {"per_time", "max_over_time", "mean_over_time"}
        Defines how the selection threshold is computed:
          - "per_time":       per-timepoint truncation
          - "max_over_time":  truncation on max-over-time
          - "mean_over_time": empirical correction on mean-over-time (recommended here)

    Returns
    -------
    theta : array, shape (T,)
        Bias-corrected estimate (same shape as d_vec).
    """

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

    # --------------------------------------------------------------------------
    # Compute threshold depending on selection mode
    # --------------------------------------------------------------------------
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
            # Empirical mean-over-time bias vector
            empirical_bias_vec = C[sel_rows].mean(axis=0)
        else:
            empirical_bias_vec = None
        t_thresh = np.full_like(d, u, dtype=float)

        # Parameters for parametric fallback (linear constraint)
        mu_m = float(mu0.mean())
        sigma_m2 = float((sigma0**2).sum() / (Tlen**2))
        sigma_m = np.sqrt(max(sigma_m2, 1e-32))

    else:  # per_time
        t_thresh = np.quantile(C, q, axis=0)
        empirical_bias_vec = None

    # --------------------------------------------------------------------------
    # Start from simple centering
    # --------------------------------------------------------------------------
    theta = d - mu0

    # --------------------------------------------------------------------------
    # Apply correction only if selected
    # --------------------------------------------------------------------------
    if apply_only_when_selected and alpha < 1.0 - 1e-12:
        if selection_mode == "mean_over_time":
            if d.mean() <= t_thresh[0]:
                return theta  # not selected -> return simple centering
        else:
            if not np.any(d > t_thresh):
                return theta

    # --------------------------------------------------------------------------
    # Case 1: empirical mean-over-time correction (nonparametric)
    # --------------------------------------------------------------------------
    if selection_mode == "mean_over_time" and empirical_bias_vec is not None:
        # Subtract empirical conditional mean
        return d - empirical_bias_vec

    # --------------------------------------------------------------------------
    # Case 2: per-time or max-over-time (parametric truncated normal)
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # Case 3: parametric fallback for mean-over-time (linear constraint)
    # --------------------------------------------------------------------------
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


def load_group(regs, timeframe, ptype='p_mean_c', alpha=0.05, is_stim=False, correction='simple', dist='de'):
        # print(timeframe)
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
                # print(reg)
                if len(splits) > 1:
                    r = np.concatenate([r_all[reg][0].reshape(1, -1), r_all[reg][1]], axis=0) / len(splits)
                else:
                    r = r_all[reg]
                if dist=='xn':
                    term = r - baseline
                elif dist=='de':
                    cell_num = d_split[reg]['nclus']
                    all_regs_cell_num += cell_num
                    # term = (r ** 2) * cell_num
                    term = (r - baseline) * cell_num
                all_regs_r = term if all_regs_r is None else (all_regs_r + term)

        if dist=='xn':
            avg_r = all_regs_r
        elif dist=='de':
            if all_regs_cell_num == 0 or all_regs_r is None:
                return None
            # avg_r = (all_regs_r / all_regs_cell_num) ** 0.5
            avg_r = all_regs_r / all_regs_cell_num

        if correction=="debias":
            # selection-aware correction per timepoint:
            # d_t = avg_r[0], controls_t = avg_r[1:]
            corrected = _debias_selected_vector(avg_r[0], avg_r[1:], alpha=alpha)
        elif correction=="simple":
            corrected = avg_r[0] - np.mean(avg_r[1:], axis=0)
        else: # no correction or intertrial baseline
            corrected = avg_r[0]
        return corrected

        
def plot_group_comparison_over_regions(regions, 
                                       timeframes=('act_block_duringstim', 'act_block_duringstim'),
                                       alpha=1.0, ptype='p_mean_c',
                                       label_A='integrator', label_B='move',
                                       correction=None, dist='de'):

    """
    regions: dictionary of regions to plot for each timeframe and group
    timeframes: list of timeframes to plot
    alpha: significance level
    ptype: type of p-value to plot
    label_A: label for the first group of regions
    label_B: label for the second group of regions
    correction: type of correction to apply
    dist: distance metric to use, 'de' or 'xn'
    """

    if isinstance(timeframes, str):
        timeframes = (timeframes,)

    fig, axs = plt.subplots(1, len(timeframes), sharey=True, figsize=(6, 1.7), dpi=250)
    if len(timeframes) == 1:
        axs = [axs]

    amp_ratio = {}
    offset_ratio = {}
    for ax, timeframe in zip(axs, timeframes):

        # pick region sets depending on alignment
        regs_move = regions['move_regs_choice'] if 'duringchoice' in timeframe else regions['move_regs_stim']
        regs_int = regions['int_regs_choice'] if 'duringchoice' in timeframe else regions['int_regs_stim']

        if 'duringchoice' in timeframe:
            times = np.linspace(-150, 0, 72)
        elif 'duringstim' in timeframe:
            times = np.linspace(0, 150, 72)
        else:
            times = np.linspace(-400, -100, 144)

        r_int = load_group(regs_int, timeframe, ptype=ptype, alpha=alpha, correction=correction, dist=dist)
        amp_int = np.max(r_int)-np.min(r_int)
        print('amp_int', amp_int)
        r_move = load_group(regs_move, timeframe, ptype=ptype, alpha=alpha, correction=correction, dist=dist)
        amp_move = np.max(r_move)-np.min(r_move)
        print('amp_move', amp_move)

        print('amp_move/amp_int', amp_move/amp_int)
        amp_ratio[timeframe] = amp_move/amp_int
        amp_diff = amp_move - amp_int
        offset_ratio[timeframe] = np.mean(r_move[:5])/np.mean(r_int[:5])
        offset_diff = np.mean(r_move[:5]) - np.mean(r_int[:5])

        ax.plot(times, r_int, color='gold', linewidth=2, label=label_A)
        ax.plot(times, r_move, color='tomato', linewidth=2, label=label_B)

        # Add text annotation for both amp_ratio and offset_ratio in same text box
        # if 'duringstim' in timeframe:
        #     text_content = f'amp ratio: {amp_ratio[timeframe]:.3f}'
        #     # text_content = f'amp ratio: {amp_ratio[timeframe]:.3f}\nOffset ratio: {offset_ratio[timeframe]:.3f}'
        # else:
        #     text_content = f'amp ratio: {amp_ratio[timeframe]:.3f}'       
        # ax.text(0.05, 0.95, text_content, 
        #         transform=ax.transAxes, fontsize=10, verticalalignment='top',
        #         # bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        #         )

        ax.set_xticks(np.linspace(times[0], times[-1], 4))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        ax.set_facecolor('none')
        ax.tick_params(labelsize=12)
        # ax.set_yticks([])
        # ax.set_xticks([])
        # ax.set_title(timeframe.replace('during', 'during '), fontsize=12)

    # shared legend
    # handles, labels = axs[0].get_legend_handles_labels()
    # fig.legend(handles, labels, frameon=False, loc='upper left',
    #            bbox_to_anchor=(0.02, 1.02), fontsize=10)

    mode = 'prior' if 'block' in timeframes[0] else 'choice'
    axs[0].set_ylabel(r'd$^{\mathrm{%s, }\mathit{s}}_{\mathit{\{I, M\}}}(t)$' % mode,
                  fontsize=14, fontfamily='Times New Roman')
    axs[1].set_ylabel(r'd$^{\mathrm{%s, }\mathit{m}}_{\mathit{\{I, M\}}}(t)$' % mode,
                  fontsize=14, fontfamily='Times New Roman')

    # fig.tight_layout()
    fig.savefig(f"{save_dir}/compare_int_choice_{dist}_{'_'.join(timeframes)}.svg", transparent=True)

    return amp_ratio, offset_ratio


# ------- functions to plot regional tables ---------
def get_sc_table(times, ptype, stim_time=None, alpha=0.05, n=72, combined_p=True, slope_threshold=0, 
                 sc_threshold=0.0, amp_loc_threshold=67, stim_restr=True):
    
    # # Plot comparison table
    sc_splits = {'sc_duringchoice': [time for time in times if 'duringchoice' in time and time.startswith('stim')] + 
                                  [time for time in times if 'duringchoice' in time and time.startswith('choice')],
                 'sc_duringstim': [time for time in times if 'duringstim' in time and time.startswith('stim')] + 
                                  [time for time in times if 'duringstim' in time and time.startswith('choice')]}

    tables, res = {}, {}

    if combined_p:
        for time in times:
            splits = run_align[time]
            split_name = 'combined_'+"_".join(splits)

            compute_amp_slope(time, n)
            results = manifold_to_csv(split_name, alpha, ptype)
            # min_val = results['amp_euc'].min()
            # max_val = results['amp_euc'].max()
            # results['amp_euc'] = (results['amp_euc'] - min_val) / (max_val - min_val) + 1e-4

            results['amp_euc'] *= results['significant']
            results = results.fillna(0)
            tables[time] = results['amp_euc']
            tables[f'{time}_amp_slope'] = results['amp_slope']
            tables[f'{time}_slope_last'] = results['slope_last']
            tables[f'{time}_amp_loc'] = results['amp_loc']
            tables[f'{time}_slope_last_5'] = results['slope_last_5']
            tables[f'{time}_slope_last_10'] = results['slope_last_10']
            tables[f'{time}_amp_last5_is_global_max'] = results['amp_last5_is_global_max']

        # add in short splits for stim
        if stim_time is None:
            if 'act' in times[0]:
                stim_time1 = 'stim_duringstim1_act' # stim w/ control for prior only
                stim_time = 'stim_duringstim_short_act' # stim w/ control for prior and choice
            else:
                stim_time1 = 'stim_duringstim1'
                stim_time = 'stim_duringstim_short'
        for s in [stim_time, stim_time1]:
            splits = run_align[s]
            split_name = 'combined_'+"_".join(splits)
            results = manifold_to_csv(split_name, alpha, ptype)
            results['amp_euc'] *= results['significant']
            tables[s] = results['amp_euc']

    else:
        for time in times:
            splits = run_align[time]
            for split in splits:
                results = manifold_to_csv(split, alpha, ptype)
                results['amp_euc'] *= results['significant']
                # min_val = results['amp_euc'].min()
                # max_val = results['amp_euc'].max()
                # results['amp_euc'] = (results['amp_euc'] - min_val) / (max_val - min_val) + 1e-4
                results = results.fillna(0)
                if time not in tables:
                    tables[time] = results['amp_euc']
                else:
                    tables[time] += results['amp_euc']
        
    tables['region'] = results['region']

    # identify regions with move_init ramp shape based on choice diff curve shape, movement aligned
    meta_split = 'sc_duringchoice'
    splits = sc_splits[meta_split]
    res[f'{meta_split}_amp_loc'] = (tables[f'{splits[1]}_amp_loc'] > amp_loc_threshold)
    res[f'{meta_split}_slope_last'] = (tables[f'{splits[1]}_slope_last'] > slope_threshold)
    res[f'{meta_split}_amp_slope'] = (tables[f'{splits[1]}_amp_slope'] > 0)
    res[f'{meta_split}_slope_last_n'] = (tables[f'{splits[1]}_slope_last_5'] > 0)
    # res[f'{meta_split}_slope_last_n'] = (tables[f'{splits[1]}_slope_last_5'] > 0) & (tables[f'{splits[1]}_slope_last_10'] > 0)

    res[f'{meta_split}_move_shape'] = np.full(len(res[f'{meta_split}_amp_loc']), np.nan)  # Initialize with NaN
    # move_shape = (res[f'{meta_split}_slope_last']
    #                 & res[f'{meta_split}_slope_last_n']
    #                 & res[f'{meta_split}_amp_loc'])
    move_shape = (tables[f'{splits[1]}_amp_last5_is_global_max'] & res[f'{meta_split}_slope_last'])
    res[f'{meta_split}_move_shape'][move_shape] = 1
    # res[f'{meta_split}_move_shape1'] = (tables[f'{splits[1]}_amp_last5_is_global_max'] & res[f'{meta_split}_slope_last'])


    for meta_split in sc_splits:
        # Calculate choice-stim metric, within [0,1] to be plotted
        splits = sc_splits[meta_split]
        res[f'{meta_split}_amp_slope'] = (tables[f'{splits[1]}_amp_slope'] > 0)
        if 'stim' in meta_split:
            res[meta_split] = tables[splits[1]]/(tables[splits[0]] + tables[splits[1]] + tables[stim_time] + tables[stim_time1])
            res[f'{meta_split}0'] = tables[splits[1]]/(tables[splits[0]] + tables[splits[1]] + tables[stim_time])
        else:
            res[meta_split] = tables[splits[1]]/(tables[splits[0]] + tables[splits[1]])

        # Assign movement areas
        res[f'{meta_split}_move_init'] = (move_shape & (res['sc_duringchoice'] > sc_threshold)).astype(int)

        # Assign integrator areas
        # if 'choice' in meta_split:
        res[f'{meta_split}_integrator'] = (((res[meta_split] > 0)
                                            & res[f'{meta_split}_amp_slope']).astype(int) - res[f'{meta_split}_move_init'])
        # else:
        #     res[f'{meta_split}_integrator'] = ((res[meta_split] > 0).astype(int) - res[f'{meta_split}_move_init'])
        
        res[f'{meta_split}_regtype'] = np.full(len(res[meta_split]), np.nan)  # Initialize with NaN
        res[f'{meta_split}_regtype'][res[f'{meta_split}_move_init'] == 1] = 1
        res[f'{meta_split}_regtype'][res[f'{meta_split}_integrator'] == 1] = 0.5

    res['sc_stim_regtype'] = np.full(len(res['sc_duringstim']), np.nan)  # Initialize with NaN
    # res['sc_stim_regtype'][res['sc_duringstim'] ==0] = 0
    res['sc_stim_regtype'][tables[stim_time1] >0] = 0.1

    # add in early stim (short) results for stim
    if stim_restr:
        # restrict to stim regions without any significant choice coding
        # mask = (res['sc_duringstim']<0.5) & (~res['sc_duringstim_amp_slope']) | (res['sc_duringstim']<0.2)
        mask = (res['sc_duringstim']<0.2)
        res['sc_duringstim_regtype'][mask] = 0
        mask1 = np.isnan(res['sc_duringstim0']) & (tables[stim_time1] > 0)
        res['sc_duringstim0'][mask1] = 0.1
        res['sc_duringstim_regtype'][mask1] = 0.1
    else:
        mask = (tables[stim_time] > 0)
        res['sc_stim_regtype'][mask] = 0
        mask1 = np.isnan(res['sc_duringstim0']) & (tables[stim_time1] > 0)
        res['sc_duringstim'][mask1] = 0.1
        res['sc_duringstim_regtype'][res['sc_duringstim']==0.1] = 0.1
    
    res['region'] = results['region']
    res = pd.DataFrame(data=res)

    return res

def plot_table_with_styles(df, beryl_palette, colormap_lookup, out_path):
    fig, ax = plt.subplots()
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=None,           # no headers
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.2, 1.3)

    nrows, ncols = df.shape

    def _is_num(v):
        return isinstance(v, (int, float, np.integer, np.floating)) or (
            hasattr(v, "dtype") and np.issubdtype(v.dtype, np.number)
        )

    for r in range(nrows):
        for c in range(ncols):
            cell = table[r, c]
            col_name = df.columns[c]
            val = df.iat[r, c]

            # Region column
            if col_name == 'region':
                cell.set_facecolor(beryl_palette.get(val, '#ffffff'))
                cell.get_text().set_fontsize(10)
                cell.get_text().set_weight('bold')
                cell.set_width(0.18)

            # Numeric / boolean-ish columns
            elif _is_num(val):
                v = float(val)
                cell.set_width(0.12)

                if pd.isna(v):
                    cell.set_facecolor('#f2f2f2')
                elif 'sc' in col_name:
                    # assume v in [0,1]; clamp just in case
                    v = min(max(v, 0.0), 1.0)
                    rgb = colormap_lookup.get(col_name, lambda x: (1, 1, 1))(v)
                    cell.set_facecolor(to_hex(rgb))
                else:
                    if v == 0.0:
                        cell.set_facecolor('#f2f2f2')
                    else:
                        v = min(max(v, 0.0), 1.0)
                        rgb = colormap_lookup.get(col_name, lambda x: (1, 1, 1))(v)
                        cell.set_facecolor(to_hex(rgb))
                cell.get_text().set_text('')  # hide numbers

            # Non-numeric fallback
            else:
                cell.set_facecolor('#ffffff')
                cell.get_text().set_fontsize(6)

            cell.set_linewidth(0.5)
            cell.set_edgecolor('white')

    plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=350, transparent=True)
    plt.close()



def plot_table(times, alpha=0.05, ptype='p_euc_c', datatype='true_block'):
    table = {}
    for timeframe in times:
        splits = run_align[timeframe]
        if len(splits) == 1:
            split_name = splits[0]
        else:
            split_name = 'combined_'+"_".join(splits)
        if timeframe=='act_intertrial0':
            sample = False
        else:
            sample = True
        res = manifold_to_csv(split_name, alpha, ptype, sample=sample)
        min_val = res['amp_euc'].min()
        max_val = res['amp_euc'].max()
        res['amp_euc'] = (res['amp_euc'] - min_val) / (max_val - min_val) + 1e-4
        res['amp_euc'] *= res['significant']
        res = res.fillna(0)
        table[timeframe] = res['amp_euc']
        
    table = pd.DataFrame(data=table)
    table['region'] = res.region
    table['beryl_hex'] = res.region.apply(swanson_to_beryl_hex, args=[br])
    beryl_palette = dict(zip(table['region'], table['beryl_hex']))
    table['sum'] = table[times].sum(axis=1)
    table['cosmos'] = table['region'].apply(lambda r: beryl_to_cosmos(r, br))

    # Load or compute region order
    ordering_path = Path(meta_pth, 'region_order.txt')
    if ordering_path.exists():
        with open(ordering_path) as f:
            region_order = [line.strip() for line in f]
    else:
        table = table.sort_values(['cosmos', 'sum'], ascending=[True, False])
        region_order = table['region'].tolist()
        with open(ordering_path, 'w') as f:
            f.writelines(r + '\n' for r in region_order)

    table['region'] = pd.Categorical(table['region'], categories=region_order, ordered=True)
    table = table.sort_values('region')

    # Drop non-display columns
    df_to_plot = table.drop(columns=['beryl_hex', 'sum', 'cosmos']).reset_index(drop=True)
    cols = df_to_plot.columns.tolist()
    cols = ['region'] + [c for c in cols if c != 'region']
    df_to_plot = df_to_plot[cols]

    colormap_lookup = {timeframe: get_cmap_(timeframe) for timeframe in times}
    plot_table_with_styles(
        df=df_to_plot,
        colormap_lookup=colormap_lookup,
        beryl_palette=beryl_palette,
        out_path=Path(meta_pth, f'table_{datatype}_alltimes_{ptype}_{alpha}.png')
    )
    return df_to_plot


def plot_combined_table_summary(sc_times, timing_splits, ptype='p_mean_c', alpha=0.05, alpha_sc=0.05, combined_p=True, 
                                sc_threshold=0.6, slope_threshold=0, amp_loc_threshold=67, n=72, stim_restr=True,
                                display='overall'):

    if stim_restr: # if stim regions are restricted to those without any significant choice coding
        sc_splits = ['sc_duringchoice_regtype', 'sc_duringstim_regtype']
    else:
        sc_splits = ['sc_duringchoice_regtype', 'sc_duringstim_regtype', 'sc_stim_regtype']

    # Handle SC splits with combined L/R
    table = get_sc_table(sc_times, ptype, alpha=alpha_sc, combined_p=combined_p,
                         sc_threshold=sc_threshold, slope_threshold=slope_threshold, 
                         amp_loc_threshold=amp_loc_threshold, n=n)

    # Handle timing splits
    for timing_split in timing_splits:
        if combined_p:
            splits = run_align[timing_split]
            split_name = 'combined_'+"_".join(splits)
            res = manifold_to_csv(split_name, alpha, ptype)
            min_val = res['amp_euc'].min()
            max_val = res['amp_euc'].max()
            res['amp_euc'] = (res['amp_euc'] - min_val) / (max_val - min_val) + 1e-4
            res['amp_euc'] *= res['significant']
            res = res.fillna(0)
            table[timing_split] = res['amp_euc']
            # add gain and offset results
            res['gain_sig'] = res['p_gain'] < alpha
            table[timing_split + '_gain_sig'] = res['gain_sig'] * res['significant'] * res['p_gain_effect']
            if 'duringstim' in timing_split:
                res['offset_sig'] = res['p_offset'] < alpha
                table[timing_split + '_offset_sig'] = res['offset_sig'] * res['significant'] * res['p_offset_effect']
        else:
            for split in run_align[timing_split]:
                res = manifold_to_csv(split, alpha, ptype)
                min_val = res['amp_euc'].min()
                max_val = res['amp_euc'].max()
                res['amp_euc'] = (res['amp_euc'] - min_val) / (max_val - min_val) + 1e-4
                res['amp_euc'] *= res['significant']
                res = res.fillna(0)
                if timing_split not in table:
                    table[timing_split] = res['amp_euc']
                else:
                    table[timing_split] += res['amp_euc']

    # Create DataFrame
    df = pd.DataFrame(table)
    df['beryl_hex'] = df['region'].apply(swanson_to_beryl_hex, args=[br])
    beryl_palette = dict(zip(df['region'], df['beryl_hex']))
    df['cosmos'] = df['region'].apply(lambda r: beryl_to_cosmos(r, br))
    df['sum'] = df[sc_splits + timing_splits].sum(axis=1, skipna=True)

    # Region ordering
    ordering_path = Path(meta_pth, 'region_order.txt')
    if ordering_path.exists():
        with open(ordering_path) as f:
            region_order = [line.strip() for line in f]
    else:
        df_sorted = df.sort_values(['cosmos', 'sum'], ascending=[True, False])
        region_order = df_sorted['region'].tolist()
        with open(ordering_path, 'w') as f:
            f.writelines(r + '\n' for r in region_order)

    df['region'] = pd.Categorical(df['region'], categories=region_order, ordered=True)
    df = df.sort_values('region')
    column_names = df.columns.difference(['region']).tolist()

    # Prepare and plot
    # for display in ['overall', 'gain_offset']:
    if display == 'overall':
        choice_time = [time for time in timing_splits if 'duringchoice' in time]
        stim_time = [time for time in timing_splits if 'duringstim' in time]
        if stim_restr:
            display_cols = ['region'] + choice_time + ['sc_duringchoice_regtype'] + stim_time + ['sc_duringstim_regtype']
        else:
            display_cols = ['region'] + choice_time + ['sc_duringchoice_regtype'] + stim_time + ['sc_duringstim_regtype'] + ['sc_stim_regtype']
    else: 
        # display only gain and offset
        choice_time = [time+'_gain_sig' for time in timing_splits if 'duringchoice' in time]
        stim_time = [item for time in timing_splits if 'duringstim' in time 
                    for item in (time+'_gain_sig', time+'_offset_sig')]
        display_cols = ['region'] + choice_time + stim_time
    df_to_plot = df[display_cols].reset_index(drop=True)

    colormap_lookup = {name: get_cmap_(name) for name in column_names}

    if 'act' in timing_splits[0]:
        block_type = 'act_block'
    else:
        block_type = 'true_block'

    if 'act' in sc_times[0]:
        out_path = Path(meta_pth, f'table_{block_type}_combined_summary_act_{ptype}_combinedp{combined_p}_{alpha}_{display}.png')
    else: 
        out_path = Path(meta_pth, f'table_{block_type}_combined_summary_{ptype}_combinedp{combined_p}_{alpha}_{display}.png')
    plot_table_with_styles(
        df=df_to_plot,
        beryl_palette=beryl_palette,
        colormap_lookup=colormap_lookup,
        out_path=out_path
    )

    if display == 'gain_offset':
        return df.reset_index(drop=True)


def plot_sc_table(times, ptype, metric='regtype', alpha=0.05, slope_threshold=0, 
                 sc_threshold=0.7, amp_loc_threshold=67, n=72, stim_restr=True):
    '''
    metric: 'regtype' (region category: integrator, movement, stim) or 'move_shape' or 'sc'
    '''

    if metric == 'regtype': 
        if stim_restr:
            sc_splits = ['sc_duringchoice_regtype', 'sc_duringstim_regtype']
        else:
            sc_splits = ['sc_duringchoice_regtype', 'sc_duringstim_regtype', 'sc_stim_regtype']
    elif metric == 'move_shape':
        sc_splits = ['sc_duringchoice_move_shape']
    elif metric == 'sc':
        sc_splits = ['sc_duringchoice', 'sc_duringstim0']
    else: 
        raise ValueError(f"Invalid metric: {metric}")
    
    if 'act' in times[0]:
        datatype = 'stimchoice_act'
    else:
        datatype = 'stimchoice'
    datatype = f'{datatype}_{metric}'

    res = get_sc_table(times, ptype, alpha=alpha, sc_threshold=sc_threshold, 
                       slope_threshold=slope_threshold, amp_loc_threshold=amp_loc_threshold, n=n)
    
    # Add hex values for Beryl regions
    res['beryl_hex'] = res.region.apply(swanson_to_beryl_hex,args=[br])    
    beryl_palette = dict(zip(res.region, res.beryl_hex))
    
    # Combine sc_duringstim and sc_duringchoice to get one unified region categorization
    # if metric == 'int_mov':
    #     res['sc_int_mov'] = res['sc_duringstim_int_mov']
    #     # Set sc_int_mov to 0.5 where sc_duringchoice_int_mov is 0.5
    #     res.loc[res['sc_duringchoice_int_mov'] == 0.5, 'sc_int_mov'] = 0.5

    # Order columns according to panels in Figure
    names = ['region'] #, 'region_color']
    for split in sc_splits:
        names.append(split)
    res = res[names]

    # Sum values in each row to use for sorting
    res['sum']  = res[names[2:]].apply(np.sum,axis=1)
    res['cosmos'] = res['region'].apply(lambda r: beryl_to_cosmos(r, br))
    
    # Load or compute region order
    ordering_path = Path(meta_pth, 'region_order.txt')
    if ordering_path.exists():
        with open(ordering_path) as f:
            region_order = [line.strip() for line in f]
    else:
        res = res.sort_values(['cosmos', 'sum'], ascending=[True, False])
        region_order = res['region'].tolist()
        with open(ordering_path, 'w') as f:
            f.writelines(r + '\n' for r in region_order)

    res['region'] = pd.Categorical(res['region'], categories=region_order, ordered=True)
    res = res.sort_values('region')
    
    df_to_plot = res.drop(columns=['cosmos', 'sum']).reset_index(drop=True)

    # Ensure region is first column
    cols = df_to_plot.columns.tolist()
    cols = ['region'] + [c for c in cols if c != 'region']
    df_to_plot = df_to_plot[cols]

    # Build column-specific colormap dictionary
    colormap_lookup = {col: get_cmap_(col) for col in df_to_plot.columns if col != 'region'}

    # Export using correct filename
    outname = f'table_{datatype}_{ptype}_{alpha}.png'
    plot_table_with_styles(
        df=df_to_plot,
        beryl_palette=beryl_palette,
        colormap_lookup=colormap_lookup,
        out_path=Path(meta_pth, outname)
    )


# def plot_combined_sc_table_summary(sc_times, ptype='p_euc_c1', alpha=0.05, combined_p=True):
#     timing_splits = ['block_duringchoice', 'block_duringstim']
#     sc_splits = ['sc_duringchoice', 'sc_duringstim']
#     # region_sets = []
#     # region_map = {}

#     # Handle SC splits with combined L/R
#     table = get_sc_table(sc_times, ptype, alpha=alpha, combined_p=combined_p)
#     # tables = {}
#     # for meta_split in sc_splits:
#     #     r = load_meta_results(f'{meta_split}1')
#     #     splits = meta_splits[f'{meta_split}1']
#     #     newsplits = meta_splits[f'{meta_split}']
#     #     r = r.fillna(0)

#     #     # Combine L/R
#     #     r[f'amp_{newsplits[0]}'] = (
#     #         r[f'amp_{splits[0]}'] * r[f'{splits[0]}_significant'] +
#     #         r[f'amp_{splits[1]}'] * r[f'{splits[1]}_significant']
#     #     ) / 2
#     #     r[f'amp_{newsplits[1]}'] = (
#     #         r[f'amp_{splits[2]}'] * r[f'{splits[2]}_significant'] +
#     #         r[f'amp_{splits[3]}'] * r[f'{splits[3]}_significant']
#     #     ) / 2

#     #     tables[meta_split] = r
#     #     splits = meta_splits[meta_split]
#     #     amp0 = f'amp_{splits[0]}'
#     #     amp1 = f'amp_{splits[1]}'
#     #     choice_stim = r[amp1] / (r[amp0] + r[amp1])
#     #     # region_map[meta_split] = r['region']        
#     #     table[meta_split] = choice_stim
#     #     # region_sets.append(r['region'])

#     # Handle timing splits
#     for timing_split in timing_splits:
#         if combined_p:
#             splits = run_align[timing_split]
#             split_name = 'combined_'+"_".join(splits)
#             res = manifold_to_csv(split_name, alpha, ptype)
#             min_val = res['amp_euc'].min()
#             max_val = res['amp_euc'].max()
#             res['amp_euc'] = (res['amp_euc'] - min_val) / (max_val - min_val) + 1e-4
#             res['amp_euc'] *= res['significant']
#             res = res.fillna(0)
#             # region_map[timing_split] = res['region']
#             table[timing_split] = res['amp_euc']
#             # region_sets.append(res['region'])
#         else:
#             for split in run_align[timing_split]:
#                 res = manifold_to_csv(split, alpha, ptype)
#                 min_val = res['amp_euc'].min()
#                 max_val = res['amp_euc'].max()
#                 res['amp_euc'] = (res['amp_euc'] - min_val) / (max_val - min_val) + 1e-4
#                 res['amp_euc'] *= res['significant']
#                 res = res.fillna(0)
#                 if timing_split not in table:
#                     table[timing_split] = res['amp_euc']
#                 else:
#                     table[timing_split] += res['amp_euc']

#     # # Union of all regions
#     # all_regions = pd.Index(sorted(set().union(*region_sets)))

#     # # Align each column to full region list, filling with NaN
#     # for k in table:
#     #     col = table[k]
#     #     col = pd.Series(col.values, index=region_map[k])
#     #     table[k] = col.reindex(all_regions)

#     # Create DataFrame
#     df = pd.DataFrame(table)
#     # df['region'] = all_regions
#     df['beryl_hex'] = df['region'].apply(swanson_to_beryl_hex, args=[br])
#     beryl_palette = dict(zip(df['region'], df['beryl_hex']))
#     df['cosmos'] = df['region'].apply(lambda r: beryl_to_cosmos(r, br))
#     df['sum'] = df[sc_splits + timing_splits].sum(axis=1, skipna=True)

#     # Region ordering
#     ordering_path = Path(meta_pth, 'region_order.txt')
#     if ordering_path.exists():
#         with open(ordering_path) as f:
#             region_order = [line.strip() for line in f]
#     else:
#         df_sorted = df.sort_values(['cosmos', 'sum'], ascending=[True, False])
#         region_order = df_sorted['region'].tolist()
#         with open(ordering_path, 'w') as f:
#             f.writelines(r + '\n' for r in region_order)

#     df['region'] = pd.Categorical(df['region'], categories=region_order, ordered=True)
#     df = df.sort_values('region')
#     column_names = df.columns.difference(['region']).tolist()

#     # Prepare and plot
#     display_cols = ['region'] + ['sc_duringchoice', 'block_duringchoice',
#                                  'sc_duringstim', 'block_duringstim']
#     df_to_plot = df[display_cols].reset_index(drop=True)

#     colormap_lookup = {name: get_cmap_(name) for name in column_names}

#     if 'stim_duringstim0' in sc_times:
#         out_path = Path(meta_pth, f'table_combined_sc_summary0_{ptype}_combinedp{combined_p}.png')
#     else: 
#         out_path = Path(meta_pth, f'table_combined_sc_summary_{ptype}_combinedp{combined_p}.png')
#     plot_table_with_styles(
#         df=df_to_plot,
#         beryl_palette=beryl_palette,
#         colormap_lookup=colormap_lookup,
#         out_path=out_path
#     )


def plot_combined_onetype(sc_times, sc_type, timing_split, ptype='p_euc', alpha=0.05, combined_p=True,
                          sc_threshold=0.6, slope_threshold=0, amp_loc_threshold=67, n=72, 
                          add_intertrial=False, stim_criteria='strict'):

    '''
    sc_type: 'stim' or 'move' or 'integrator'
    timing_split: list of block splits during specific time windows, 
    e.g. ['block_duringstim'], ['act_block_duringstim', 'act_block_duringchoice']
    '''

    table = {}
    # region_sets = []
    # region_map = {}

    # Handle SC splits with combined L/R
    sc_res = get_sc_table(sc_times, ptype, alpha=alpha, combined_p=combined_p, 
                       sc_threshold=sc_threshold, slope_threshold=slope_threshold, 
                       amp_loc_threshold=amp_loc_threshold, n=n)
    table['region'] = sc_res['region']

    for t_split in timing_split:
        # Handle timing split
        if 'choice' in t_split:
            sc_split = 'sc_duringchoice_regtype'
        # elif sc_type == 'stim':
        #     sc_split = 'sc_stim_regtype'
        else:
            sc_split = 'sc_duringstim_regtype'
        table[sc_split] = sc_res[sc_split]
        print(sc_split, t_split)

        if combined_p:
            splits = run_align[t_split]
            combined_name = 'combined_'+"_".join(splits)
            res = manifold_to_csv(combined_name, alpha, ptype)
            min_val = res['amp_euc'].min()
            max_val = res['amp_euc'].max()
            res['amp_euc'] = (res['amp_euc'] - min_val) / (max_val - min_val) + 1e-4
            res['amp_euc'] *= res['significant']
            res = res.fillna(0)
            table[t_split] = res['amp_euc']
            if add_intertrial:
                int_split = 'act_block_only' if 'act' in t_split else 'block_only'
                res = manifold_to_csv(int_split, alpha, ptype)
                min_val = res['amp_euc'].min()
                max_val = res['amp_euc'].max()
                res['amp_euc'] = (res['amp_euc'] - min_val) / (max_val - min_val) + 1e-4
                res['amp_euc'] *= res['significant']
                res = res.fillna(0)
                table[int_split] = res['amp_euc']

        else:
            for split in run_align[t_split]:
                res = manifold_to_csv(split, alpha, ptype)
                min_val = res['amp_euc'].min()
                max_val = res['amp_euc'].max()
                res['amp_euc'] = (res['amp_euc'] - min_val) / (max_val - min_val) + 1e-4
                res['amp_euc'] *= res['significant']
                res = res.fillna(0)
                if t_split not in table:
                    table[t_split] = res['amp_euc']
                else:
                    table[t_split] += res['amp_euc']

        # Binarize and combine timing split to sc_split
        table[t_split] = table[t_split].apply(lambda x: 1 if x > 0 else np.nan)
        if sc_type == 'stim':
            if stim_criteria=='strict':
                table[sc_split] = table[sc_split].apply(lambda x: 1 if x == 0 else np.nan)
            else:
                table[sc_split] = table[sc_split].apply(lambda x: 1 if x <= 0.1 else np.nan)
            table[f'combined_{t_split}'] = table[sc_split] * table[t_split]
        elif sc_type == 'move':
            table[sc_split] = table[sc_split].apply(lambda x: 1 if x == 1 else np.nan)
            table[f'combined_{t_split}'] = table[sc_split] * table[t_split]
        elif sc_type == 'integrator':
            table[sc_split] = table[sc_split].apply(lambda x: 1 if 0<x<1 else np.nan)
            table[f'combined_{t_split}'] = table[sc_split] * table[t_split]
        else:
            raise ValueError(f"Invalid sc_type: {sc_type}")
        
    # Combine all timing splits - entry is 1 if any combined_{t_split} column has 1
    combined_cols = [f'combined_{t_split}' for t_split in timing_split]
    table['combined'] = pd.DataFrame(table)[combined_cols].max(axis=1, skipna=True)

    # Create DataFrame
    df = pd.DataFrame(table)
    # df['region'] = all_regions
    df['beryl_hex'] = df['region'].apply(swanson_to_beryl_hex, args=[br])
    beryl_palette = dict(zip(df['region'], df['beryl_hex']))

    # Region ordering
    ordering_path = Path(meta_pth, 'region_order.txt')
    with open(ordering_path) as f:
        region_order = [line.strip() for line in f]

    df['region'] = pd.Categorical(df['region'], categories=region_order, ordered=True)
    df = df.sort_values('region')

    # Prepare and plot
    display_cols = ['region'] + ['combined'] + [int_split] if add_intertrial else ['region'] + ['combined']
    df_to_plot = df[display_cols].reset_index(drop=True)

    colormap_lookup = {
        'combined': get_cmap_('intertrial')
    }
    if add_intertrial:
        colormap_lookup[int_split] = get_cmap_('intertrial')
    
    regions_with_1 = df_to_plot.loc[df_to_plot['combined'] == 1, 'region'].tolist()

    out_path = Path(meta_pth, f'table_combined_{sc_type}_{timing_split}_{ptype}_combinedp{combined_p}_{alpha}.png')
    plot_table_with_styles(
        df=df_to_plot,
        beryl_palette=beryl_palette,
        colormap_lookup=colormap_lookup,
        out_path=out_path
    )

    return regions_with_1


# ----- plotting functions for cellular analysis --------

T_BIN = 0.0125  # bin size [sec] for neural binning
sts = 0.002  # stride size in [sec] for overlapping bins
ntravis = 30  # #trajectories for vis, first 2 real, rest pseudo

# save results here
pth_dmn = Path(one.cache_dir, 'dmn', 'res')
pth_dmn.mkdir(parents=True, exist_ok=True)


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


def plot_averaged_distances_comparison(raster_type1='integrator', raster_type2='move_init_movement',
                                       vers='concat_by_contrast_act', topprior=False, alpha=0.05, correction="simple"):
    """
    Plot averaged distance traces for two raster types overlaid in each panel.
    Left: stim-aligned trial types. Right: choice-aligned.
    """

    if 'choice_pairs' in vers:
        comparison_pairs = choice_pairs
    elif 'stim_pairs' in vers:
        comparison_pairs = stim_pairs
    elif 'block_pairs' in vers:
        comparison_pairs = block_pairs
    else:
        comparison_pairs = block_pairs_by_contrast
        
    if topprior:
        raster_types = [raster_type1+'_topprior', raster_type2+'_topprior']
    else:
        raster_types = [raster_type1, raster_type2]
    colors = {raster_types[0]: 'gold', raster_types[1]: 'tomato'}

    grouped_by_type = {'stim': {}, 'choice': {}}

    for rt in raster_types:
        d = np.load(Path(pth_dmn, f'{vers}_normFalse_shuffleTrue_{rt}.npy'),
                    allow_pickle=True).flat[0]

        for comparison in comparison_pairs:
            if 'contrast' in vers:
                tt1 = '_'.join(comparison[0].split('_')[:-1])
                tt2 = '_'.join(comparison[1].split('_')[:-1])
            else: 
                tt1, tt2 = comparison
            if 'stim' in tt1:
                group = 'stim'
            elif 'choice' in tt1:
                group = 'choice'
            else:
                continue
            dist = d['distance'][f'{comparison[0]}_{comparison[1]}']/T_BIN
            dist = (dist / d['ids'].shape) ** 0.5
            distance_controls = d['distance_controls'][f'{comparison[0]}_{comparison[1]}']/T_BIN
            distance_controls = (distance_controls / d['ids'].shape) ** 0.5
            if correction == "debias":
                corrected = _debias_selected_vector(dist, distance_controls, alpha=alpha)
            else:
                corrected = dist - np.mean(distance_controls, axis=0)
            grouped_by_type[group].setdefault(rt, []).append(corrected)

    fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.5), dpi=150, sharey=True)
    time_configs = {
        'stim': np.linspace(0, 200, grouped_by_type['stim'][raster_types[0]][0].shape[-1]),
        'choice': np.linspace(-150, 0, grouped_by_type['choice'][raster_types[0]][0].shape[-1])
    }
    tick_configs = {
        'stim': [0, 40, 80, 120],
        'choice': [0, -40, -80, -120]
    }

    for i, trial_type in enumerate(['stim', 'choice']):
        amp = {}
        ax = axs[i]
        for rt in raster_types:
            if rt not in grouped_by_type[trial_type]:
                continue
            traces = np.stack(grouped_by_type[trial_type][rt])
            mean_trace = np.mean(traces, axis=0)
            t = time_configs[trial_type]
            if trial_type == 'stim':
                valid = t <= 150
                t = t[valid]
                mean_trace = mean_trace[:len(t)]
            amp[rt] = np.max(mean_trace)-np.min(mean_trace)
            ax.plot(t, mean_trace, label=rt, linewidth=2, color=colors[rt])

        amp_ratio = amp[raster_types[1]]/amp[raster_types[0]]
        ax.text(0.05, 0.95, f'amp ratio: {amp_ratio:.2f}', transform=ax.transAxes, 
                fontsize=10, verticalalignment='top')

        # ax.set_title(f'{trial_type.title()}-aligned', fontsize=11)
        # ax.set_xlabel('Time (ms)')
        ax.set_xticks(tick_configs[trial_type])
        ax.set_xlim([t[0] - 10, t[-1]])
        ax.set_facecolor('none')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)

    # axs[0].set_ylabel('Distance')
    # axs[1].legend(frameon=False, fontsize=9)
    # fig.suptitle(f'Overlayed Avg Distance: {raster_type1} vs {raster_type2}', fontsize=12)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.5)
    fig.savefig(Path(one.cache_dir, 'dmn', 'figs', f'{raster_types[0]}_vs_{raster_types[1]}_{vers}_overlay.svg'),
                dpi=100, transparent=True, bbox_inches='tight')


def plot_averaged_distances(block_pairs, raster_type, vers='concat_by_contrast_act', ptype='mean'):
    d = np.load(Path(pth_dmn, f'{vers}_normFalse_shuffleTrue_{raster_type}.npy'),
                allow_pickle=True).flat[0]

    grouped_distances = defaultdict(list)
    grouped_distance_controls = defaultdict(list)

    for comparison in block_pairs:
        trial_type1 = '_'.join(comparison[0].split('_')[:-1])
        trial_type2 = '_'.join(comparison[1].split('_')[:-1])
        trial_type_pair = (trial_type1, trial_type2)

        distance = d['distance'][f'{comparison[0]}_{comparison[1]}']/T_BIN
        distance = (distance / d['ids'].shape) ** 0.5
        distance_controls = d['distance_controls'][f'{comparison[0]}_{comparison[1]}']/T_BIN
        distance_controls = (distance_controls / d['ids'].shape) ** 0.5

        grouped_distances[trial_type_pair].append(distance)
        grouped_distance_controls[trial_type_pair].append(distance_controls)

    for trial_type_pair, distances_list in grouped_distances.items():
        mean_distance = np.mean(np.stack(distances_list), axis=0)
        stacked_controls = np.mean(np.stack(grouped_distance_controls[trial_type_pair]), 
                                   axis=0)  # shape: [n_controls, time]

        distances = np.concatenate([mean_distance[None, :], stacked_controls], axis=0)
        p_per_time = np.mean(distances >= distances[0], axis=0)

        if ptype == 'amp':
            amp = np.max(distances, axis=1) - np.min(distances, axis=1)
            p_value = np.mean(amp >= amp[0])
        elif ptype == 'mean':
            mean_vals = np.mean(distances, axis=1)
            p_value = np.mean(mean_vals >= mean_vals[0])
        else:
            print('Invalid ptype')
            return

        x_full = (np.linspace(0, 0.20, len(mean_distance)) if 'stim' in trial_type_pair[0]
                  else np.linspace(-0.15, 0, len(mean_distance)))

        x_plot = x_full
        mean_distance_plot = mean_distance
        averaged_controls_across_contrasts_plot = stacked_controls
        p_per_time_plot = p_per_time

        if 'stim' in trial_type_pair[0]:
            valid_idx = np.where(x_full <= 0.15)[0]
            cut_end_idx = valid_idx[-1] + 1 if len(valid_idx) > 0 else 0
            x_plot = x_full[:cut_end_idx]
            mean_distance_plot = mean_distance[:cut_end_idx]
            averaged_controls_across_contrasts_plot = stacked_controls[:, :cut_end_idx]
            p_per_time_plot = p_per_time[:cut_end_idx]

        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[4, 0.7], wspace=0.05)
        ax_main = fig.add_subplot(gs[0, 0])
        ax_hist = fig.add_subplot(gs[0, 1], sharey=ax_main)

        for i in range(min(40, averaged_controls_across_contrasts_plot.shape[0])):
            ax_main.plot(x_plot, averaged_controls_across_contrasts_plot[i], color='silver', linewidth=0.2, alpha=0.5)

        ax_main.plot(x_plot, mean_distance_plot, linewidth=2, label='Distance')
        ax_main.set_facecolor('none')
        ax_main.spines['top'].set_visible(False)
        ax_main.spines['right'].set_visible(False)
        ax_main.tick_params(axis='both', labelsize=15)

        if 'stim' in trial_type_pair[0]:
            ax_main.set_xticks([0, 0.04, 0.08])
        elif 'choice' in trial_type_pair[0]:
            ax_main.set_xticks([0, -0.04, -0.08, -0.12])

        if raster_type == 'stim_res':
            p_pos = 0.90
        else:
            p_pos = 0.2
        ax_main.text(0.90, p_pos, f'p_val {p_value:.4f}', transform=ax_main.transAxes,
                     fontsize=20, verticalalignment='bottom', horizontalalignment='right')

        # p-value curve and markers
        ax_p = ax_main.twinx()
        ax_p.plot(x_plot, p_per_time_plot, linestyle='--', color='blue', linewidth=1, label='p per time')
        ax_p.set_ylabel('p', fontsize=10, color='blue')
        ax_p.tick_params(axis='y', labelcolor='blue', labelsize=8)
        ax_p.set_ylim([0, 1])

        sig_mask = p_per_time_plot < 0.05
        ax_main.scatter(x_plot[sig_mask], np.full(np.sum(sig_mask), ax_main.get_ylim()[0]),
                        marker='v', color='blue', s=20, zorder=5, label='p < 0.05')

        # Histogram panel
        avg_val_per_control = np.mean(stacked_controls, axis=1)
        ax_hist.hist(avg_val_per_control, bins=20, orientation='horizontal', color='gray', alpha=0.6)
        ax_hist.axhline(np.mean(mean_distance), linestyle='-', linewidth=2)
        ax_hist.set_facecolor('none')
        ax_hist.spines['top'].set_visible(False)
        ax_hist.spines['right'].set_visible(False)
        ax_hist.spines['bottom'].set_visible(False)
        ax_hist.tick_params(axis='y', left=False, labelleft=False)
        ax_hist.tick_params(axis='x', bottom=False, labelbottom=False)

        # Save
        output_dir = Path(one.cache_dir, 'dmn', 'figs')
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f'{"_".join(trial_type_pair)}_{raster_type}_dist_averaged_{ptype}.pdf',
                    dpi=100, transparent=True, bbox_inches='tight')
        plt.close()



def plot_average_distance_across_all_pairs(comparison_pairs, raster_type, vers='concat_by_contrast_act', 
                                           ptype='mean', suffix='', plot_offset=False, offset_window=5,
                                           alpha=0.05, plot_p_per_time=False, plot_gain=False, short_window=False, 
                                           ylim=None, yticks=None):
    """ 
    Plot averaged distance traces for a raster type across all contrasts and comparison (block, stim, choice) 
    pairs within a timewindow.
    comparison_pairs: list of tuples, each tuple contains two trial types.
    suffix: string, indicating timewindow (stim or choice)
    """

    d = np.load(Path(pth_dmn, f'{vers}_normFalse_shuffleTrue_{raster_type}.npy'),
                allow_pickle=True).flat[0]

    all_distances = []
    all_controls = []

    if 'move' in raster_type:
        dist_color = 'tomato'
    elif 'stim' in raster_type or 'vis' in raster_type:
        dist_color = '#57C1EB'
    elif raster_type == 'integrator':
        dist_color = 'gold'
    # else:
    #     raise ValueError(f'Invalid raster type: {raster_type}')

    for comparison in comparison_pairs:
        distance = d['distance'][f'{comparison[0]}_{comparison[1]}']/T_BIN
        distance = (distance / d['uuids'].shape) ** 0.5
        distance_controls = d['distance_controls'][f'{comparison[0]}_{comparison[1]}']/T_BIN
        distance_controls = (distance_controls / d['uuids'].shape) ** 0.5

        all_distances.append(distance)
        all_controls.append(distance_controls)

    mean_distance = np.mean(np.stack(all_distances), axis=0)
    stacked_controls = np.mean(np.stack(all_controls), axis=0)  # shape: [n_controls, time]

    # infer alignment
    if short_window:
        window_len = 80
    else:
        window_len = 150
    if 'stim' in comparison_pairs[0][0]:
        x_full = np.linspace(0, 200, len(mean_distance))
        valid_idx = np.where(x_full <= window_len)[0]
        cut_end_idx = valid_idx[-1] + 1 if len(valid_idx) > 0 else 0
        x_plot = x_full[:cut_end_idx]
        mean_distance = mean_distance[:cut_end_idx]
        stacked_controls = stacked_controls[:, :cut_end_idx]
        # p_per_time_plot = p_per_time[:cut_end_idx]
    elif 'choice' in comparison_pairs[0][0]:
        x_plot = np.linspace(-150, 0, len(mean_distance))
        # p_per_time_plot = p_per_time
    else:
        raise ValueError("Can't determine alignment from comparison pair naming.")

    distances = np.concatenate([mean_distance[None, :], stacked_controls], axis=0)
    p_per_time = np.mean(distances >= distances[0], axis=0)
    # Calculate p_val for the mean of the first 5 datapoints
    mean_first_offset = np.mean(distances[:, :offset_window], axis=1)
    p_val_first_offset = np.mean(mean_first_offset >= mean_first_offset[0])
    amp = np.max(distances, axis=1) - np.min(distances, axis=1)

    if ptype == 'amp':
        p_value = np.mean(amp >= amp[0])
    elif ptype == 'mean':
        mean_vals = np.mean(distances[:], axis=1)
        p_value = np.mean(mean_vals >= mean_vals[0])
    else:
        raise ValueError(f'Invalid ptype: {ptype}')


    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 0.7], wspace=0.05)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1], sharey=ax_main)

    for i in range(min(100, stacked_controls.shape[0])):
        if 'stim' in suffix and plot_offset:
            ax_main.plot(x_plot[:offset_window], stacked_controls[i][:offset_window], color='#5f7ea3', linewidth=0.2, alpha=0.5)
            ax_main.plot(x_plot[offset_window-1:], stacked_controls[i][offset_window-1:], color='silver', linewidth=0.2, alpha=0.5)
        else:
            ax_main.plot(x_plot, stacked_controls[i], color='silver', linewidth=0.2, alpha=0.5)

    if 'stim' in suffix and plot_offset:
        ax_main.plot(x_plot[:offset_window], mean_distance[:offset_window], color='blue', 
            linestyle='--', linewidth=2, label='Distance')
        ax_main.plot(x_plot[offset_window-1:], mean_distance[offset_window-1:], color=dist_color, 
            linestyle='--', linewidth=2, label='Distance')
    else:
        ax_main.plot(x_plot, mean_distance, color=dist_color, linestyle='--', linewidth=2, label='Distance')

    # Subtract the mean of the first 5 datapoints to examine gain effect
    if plot_gain:
        # if 'stim' in suffix:
        if p_val_first_offset < alpha:
            offset = mean_first_offset[0] - np.mean(mean_first_offset[1:])
            r_shifted = mean_distance - offset
            ax_main.plot(x_plot, r_shifted, c=dist_color, linewidth=2)
        else: 
            r_shifted = mean_distance
        r_gain = np.concatenate([r_shifted[4:][np.newaxis, :], stacked_controls[:, 4:]], axis=0)
        p_val_gain = np.mean(np.mean(r_gain, axis=1) >= np.mean(r_gain[0]))

    ax_main.set_facecolor('none')
    # ax_main.set_xticklabels([])
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    ax_main.tick_params(axis='both', labelsize=20)

    if 'stim' in comparison_pairs[0][0]:
        ax_main.set_xticks([0, 40, 80, 120])
    elif 'choice' in comparison_pairs[0][0]:
        ax_main.set_xticks([0, -40, -80, -120])

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # Inset: mean_distance - mean(stacked_controls) with shuffle band
    ctrl_mean = np.mean(stacked_controls, axis=0)
    diff_curve = mean_distance - ctrl_mean

    # shuffle-centered distribution per time bin
    shuf_centered = stacked_controls - ctrl_mean[None, :]
    low_band = np.percentile(shuf_centered, 100 * (alpha / 2), axis=0)
    high_band = np.percentile(shuf_centered, 100 * (1 - alpha / 2), axis=0)

    ax_ins = inset_axes(ax_main, width="20%", height="20%", loc='upper left', borderpad=2.8)
    ax_ins.fill_between(x_plot, low_band, high_band, alpha=0.3, color='gray', linewidth=0)
    ax_ins.plot(x_plot, diff_curve, linewidth=1, color=dist_color)

    # tidy inset
    ax_ins.set_xticks([])
    # ymin, ymax = np.min(low_band), np.max(high_band)
    # if ymin == ymax:
    #     ymax = ymin + 1e-6
    # pad = 0.05 * (ymax - ymin)
    ax_ins.set_ylim(-0.05, 0.22)
    # ax_ins.set_yticks([0, 0.15, 0.3])
    ax_ins.set_facecolor('none')
    for spine in ("top", "right"):
        ax_ins.spines[spine].set_visible(False)

    # print p_value
    color_main = 'red' if p_value < alpha else 'black'
    color_offset = 'red' if p_val_first_offset < alpha else 'blue'
    ax_main.text(0.40, 0.9, f'p_val {p_value:.4f}', transform=ax_main.transAxes,
                 fontsize=20, verticalalignment='bottom', horizontalalignment='right', color=color_main)
    if 'stim' in suffix and plot_offset:
        ax_main.text(0.80, 0.10, f'p_val_offset {p_val_first_offset:.4f}', transform=ax_main.transAxes,
                    fontsize=20, verticalalignment='bottom', horizontalalignment='right', color=color_offset)
    if plot_gain:
        ax_main.text(0.10, 0.85, f'p_val_gain {p_val_gain:.4f}', transform=ax_main.transAxes,
                    color='red' if p_val_gain <= alpha else 'purple', fontsize=20, ha='left', va='top')

    # p-value per time (optional)
    if plot_p_per_time:
        # ax_p = ax_main.twinx()
        # ax_p.plot(x_plot, p_per_time_plot, linestyle='--', color='blue', linewidth=1, label='p per time')
        # ax_p.set_ylabel('p', fontsize=10, color='blue')
        # ax_p.tick_params(axis='y', labelcolor='blue', labelsize=8)
        # ax_p.set_ylim([0, 1])

        sig_mask = p_per_time < 0.05
        ax_main.scatter(x_plot[sig_mask], np.full(np.sum(sig_mask), ax_main.get_ylim()[0]),
                        marker='v', color='blue', s=20, zorder=5, label='p < 0.05')

    # Histogram panel
    avg_val_per_control = np.mean(stacked_controls, axis=1)
    ax_hist.hist(avg_val_per_control, bins=20, orientation='horizontal', color='gray', alpha=0.6)
    ax_hist.axhline(np.mean(mean_distance), linestyle='--', linewidth=2, color=dist_color)
    if 'stim' in suffix and plot_offset:
        ax_hist.hist(np.mean(stacked_controls[:,:offset_window], axis=1), bins=20, 
                     orientation='horizontal', color='#5f7ea3', alpha=0.5)
        ax_hist.axhline(np.mean(mean_distance[:offset_window]), color='blue', linestyle='--', linewidth=2)
    if plot_gain:
        ax_hist.hist(np.mean(stacked_controls[:, 4:], axis=1), density=True, bins=20,
                    color='#c8a2d6', orientation='horizontal', alpha=0.5)
        ax_hist.axhline(y=np.mean(r_shifted[4:]), c=dist_color, linewidth=2)

    ax_hist.set_facecolor('none')
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)
    ax_hist.spines['bottom'].set_visible(False)
    ax_hist.tick_params(axis='y', left=False, labelleft=False)
    ax_hist.tick_params(axis='x', bottom=False, labelbottom=False)

    if ylim is not None:
        ax_main.set_ylim(ylim)

    if yticks is not None:
        ax_main.set_yticks(yticks)
    ax_main.tick_params(axis='both', labelsize=30)
    

    # Save
    output_dir = Path(one.cache_dir, 'dmn', 'figs')
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f'all_{vers}_{suffix}_{raster_type}_dist_averaged_{ptype}.svg',
                dpi=100, transparent=True, bbox_inches='tight')
    plt.close()

    if plot_gain:
        return mean_first_offset[0], amp[0]
