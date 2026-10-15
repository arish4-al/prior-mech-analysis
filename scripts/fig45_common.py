"""Shared constants / helpers for the Figure 4.5 rerun (block effects on
rastermap-defined cell types).

Source of truth for the original analysis (paper-brain-wide-map):
  - ``dmn_avgbystim.concat_PETHs`` / ``get_all_PETHs`` / ``stack_concat``
  - ``analysis_functions.plot_average_distance_across_all_pairs`` /
    ``plot_averaged_distances_comparison``
  - ``bwm_dmn.ipynb`` sections "examine block effects on cell types from
    rastermap" and "find stim res cells from rastermap".

Journal: journals/fig45_block_effects_cell_types.md
"""
from pathlib import Path

import numpy as np

DMN = Path.home() / 'Downloads/ONE/openalyx.internationalbrainlab.org/dmn'
DMN_RES = DMN / 'res'
NEW_STACK = Path.home() / 'Downloads/concat_cvTrue.npy'
OUT_ROOT = DMN_RES / 'fig45'
PAPER_FIGS = DMN.parent / 'manifold' / 'paper_figs'
CLUSTERS_TABLE = DMN.parent / 'bwm_tables' / 'clusters.pqt'
# Built locally from NEW_STACK (not on ORCD); git-tracked so jobs read it from the repo.
TYPES_DIR = Path(__file__).resolve().parents[1] / 'data' / 'fig45'
CELL_TYPES = TYPES_DIR / 'cell_types.npy'

T_BIN = 0.0125   # bin size [s]
STS = 0.002      # stride [s] for overlapping bins -> int(T_BIN // STS) = 6 sub-bins
NRAND = 2000     # block-label shuffles per pair
ALPHA_AK = 0.2   # action-kernel inverse time constant ('act' prior)

STIM_WIN = (0.0, 0.2)    # stimOn-aligned [pre, post]; plotted up to 150 ms
MOVE_WIN = (0.15, 0.0)   # firstMovement-aligned [pre, post]

SATURATION_INTERVALS = [
    'saturation_stim_plus04',
    'saturation_feedback_plus04',
    'saturation_move_minus02',
    'saturation_stim_minus04_minus01',
    'saturation_stim_plus06',
    'saturation_stim_minus06_plus06',
]

# Block-L vs block-R pairs at fixed stim side x choice (vers='concat_block_pairs_act').
BLOCK_PAIRS = [
    ['sLbLchoiceL', 'sLbRchoiceL'],
    ['sLbLchoiceR', 'sLbRchoiceR'],
    ['sRbLchoiceL', 'sRbRchoiceL'],
    ['sRbLchoiceR', 'sRbRchoiceR'],
    ['stimLbLcL', 'stimLbRcL'],
    ['stimLbLcR', 'stimLbRcR'],
    ['stimRbLcL', 'stimRbRcL'],
    ['stimRbLcR', 'stimRbRcR'],
]
STIM_PAIRS = [
    ['stimLbLcL', 'stimRbLcL'],
    ['stimLbRcL', 'stimRbRcL'],
    ['stimLbLcR', 'stimRbLcR'],
    ['stimLbRcR', 'stimRbRcR'],
    ['sLbLchoiceL', 'sRbLchoiceL'],
    ['sLbRchoiceL', 'sRbRchoiceL'],
    ['sLbRchoiceR', 'sRbRchoiceR'],
    ['sLbLchoiceR', 'sRbLchoiceR'],
]
CHOICE_PAIRS = [
    ['stimLbLcL', 'stimLbLcR'],
    ['stimLbRcL', 'stimLbRcR'],
    ['stimRbLcL', 'stimRbLcR'],
    ['stimRbRcL', 'stimRbRcR'],
    ['sLbLchoiceL', 'sLbLchoiceR'],
    ['sLbRchoiceL', 'sLbRchoiceR'],
    ['sRbRchoiceL', 'sRbRchoiceR'],
    ['sRbLchoiceL', 'sRbLchoiceR'],
]
def pairs_for_vers(vers):
    """Same substring rule as ``stack_concat`` / the plotting helpers."""
    if 'stim_pairs' in vers:
        return STIM_PAIRS
    if 'choice_pairs' in vers:
        return CHOICE_PAIRS
    return BLOCK_PAIRS

# Stacked column order (PETH_types_dict['concat_block_pairs_act']).
TTYPES = ['stimLbLcL', 'stimLbRcL', 'stimLbRcR', 'stimLbLcR', 'stimRbLcL',
          'stimRbRcL', 'stimRbRcR', 'stimRbLcR', 'sLbLchoiceL', 'sLbRchoiceL',
          'sLbRchoiceR', 'sLbLchoiceR', 'sRbLchoiceL', 'sRbRchoiceL',
          'sRbRchoiceR', 'sRbLchoiceR']

# Figure colours (analysis_functions.plot_average_distance_across_all_pairs).
COLORS = {'integrator': 'gold', 'move_init_movement': 'tomato', 'move_all': 'tomato',
          'move_init': 'tomato', 'move': 'darkred', 'move_fback': 'mediumpurple',
          'stim_res': '#57C1EB', 'vis_subset0': '#57C1EB'}


def sum_for_key(lens, key, after=False):
    """Start column of ``key`` in an ordered length dict (end if ``after``)."""
    total = 0
    for k, v in lens.items():
        if k == key:
            if after:
                total += v
            break
        total += v
    return total


def stacked_path(vers, raster_type, root=DMN_RES):
    return Path(root) / f'{vers}_normFalse_shuffleTrue_{raster_type}.npy'


def load_stacked(vers, raster_type, root=DMN_RES):
    return np.load(stacked_path(vers, raster_type, root), allow_pickle=True).flat[0]


def normalized_distance(d, pair):
    """Population distance as plotted in the paper:
    sqrt(sum_cells (dPETH)^2 / T_BIN / n_cells); same for the shuffle rows."""
    key = f'{pair[0]}_{pair[1]}'
    n = len(d['uuids'])
    dist = np.sqrt(d['distance'][key] / T_BIN / n)
    ctrl = np.sqrt(d['distance_controls'][key] / T_BIN / n)
    return dist, ctrl


def savefig_svg_png(fig, path_no_ext, dpi=150):
    path_no_ext = Path(path_no_ext)
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_no_ext.with_suffix('.svg'), bbox_inches='tight')
    fig.savefig(path_no_ext.with_suffix('.png'), dpi=dpi, bbox_inches='tight',
                facecolor='white')
