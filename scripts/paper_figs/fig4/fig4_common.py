"""Shared paths, colours and page geometry for the Figure 4 rebuild.

Original page: ``~/Downloads/figure4.pdf`` (759.276 × 632.395 pt).
Panel a is the alyx overall earlystim table; b's one-row strip is
``prior_80ms`` FDR ∩ ``type_stim`` sensory; c is the already-analysed
LGd + cell-weighted sensory combined, replotted in the paper style.
d–g, h, i are redrawn from the same SC types (Figure 2 openalyx
``sc_table.csv``) and the current openalyx 4-split act-block maps.

Journal: journals/paper_main_figures.md
"""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OPENALYX = Path.home() / 'Downloads/ONE/openalyx.internationalbrainlab.org'
ALYX = Path.home() / 'Downloads/ONE/alyx.internationalbrainlab.org'

FIG_DIR = OPENALYX / 'manifold' / 'paper_figs' / 'fig4'
ORIGINAL_PDF = Path.home() / 'Downloads' / 'figure4.pdf'
PAGE_WH = (759.276, 632.395)  # points, PDF MediaBox

OA_RES = OPENALYX / 'manifold' / 'res'
ALYX_RES = ALYX / 'manifold' / 'res' / 'new'
ALYX_META = ALYX / 'meta'
SC_TABLE = OPENALYX / 'manifold' / 'paper_figs' / 'fig2' / 'openalyx' / 'sc_table.csv'
REGION_ORDER = ALYX_META / 'region_order.txt'
TABLE_PNG = (
    ALYX_META
    / 'table_act_block_combined_summary_act_p_mean_c_combinedpTrue_0.01_overall_earlystim80.png'
)
EARLYSTIM_NPY = (
    ALYX_RES / 'combined_act_block_duringstim_r_act_block_duringstim_l_earlystim80.npy'
)
UNSPLIT_REGDE = (
    ALYX_RES / 'combined_regde_act_block_duringstim_l_act_block_duringstim_r.npy'
)
NCLUS_SPLIT = 'act_block_duringstim_l'

# get_sc_table settings shared with Figure 2 / the overall table
PTYPE = 'p_mean_c'
ALPHA = 0.01
SENSORY_REGTYPES = (0.0, 0.1)
EXAMPLE_S = 'LGd'
EXAMPLE_C = ('PAG', 'PRNc')   # published C examples; both still prior_stim FDR ≤ 0.01
EXAMPLE_M = 'IRN'

WINDOW_MS = 150.0
T_MAX_MS = 80.0
N_SHOW = 40

C_STIM = '#57C1EB'
C_INTEG = '#E8AC22'
C_MOVE = '#E24A33'
C_STIM_ON = '#2BA9E0'
C_MOVE_ON = '#E8806A'
C_OFFSET = '#5f7ea3'
C_PRIOR = '#6159A6'
C_GREY = '#f2f2f2'

# Content rows of the overall-table PNG, top to bottom, after dropping the
# black bars (y on the 1736-tall PNG). Order is the reverse of
# plot_combined_table_summary display columns.
TABLE_Y0, TABLE_Y1 = 189, 1547
# Short right-hand labels (original a only had italic s / m).
A_ROW_LABELS = ['s', '80', 'stim', 'm', 'choice']


def load_af(res_dir):
    """Import analysis_functions with ONE in local mode."""
    import one.api as one_api
    real = one_api.ONE

    def _local_one(*args, **kwargs):
        return real(cache_dir=str(OPENALYX), mode='local', silent=True)

    one_api.ONE = _local_one
    import analysis_functions as af
    af.pth_res = Path(res_dir)
    af.meta_pth = Path(res_dir)
    return af


def combined_stems(af, timeframe):
    joined = '_'.join(af.run_align[timeframe])
    return f'combined_{joined}', f'combined_regde_{joined}'


def n_bins_le_tmax(n_bins, window_ms=WINDOW_MS, t_max_ms=T_MAX_MS):
    import numpy as np
    t = np.linspace(0.0, float(window_ms), int(n_bins))
    return int(np.sum(t <= float(t_max_ms)))


def fig_box(x0, y0, x1, y1):
    """Points from the top-left → matplotlib [left, bottom, w, h]."""
    w, h = PAGE_WH
    return [x0 / w, 1 - y1 / h, (x1 - x0) / w, (y1 - y0) / h]


def savefig_svg_png_pdf(fig, path_no_ext, dpi=200):
    path_no_ext = Path(path_no_ext)
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(facecolor='white', bbox_inches=None)
    fig.savefig(path_no_ext.with_suffix('.pdf'), **kw)
    fig.savefig(path_no_ext.with_suffix('.svg'), **kw)
    fig.savefig(path_no_ext.with_suffix('.png'), dpi=dpi, **kw)
