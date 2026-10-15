"""Shared paths for the Figure 4 rebuild.

Original page: openalyx ``manifold/figs/figure4.pdf``.
Panel a is replaced by the early-stim overall table, panel b's header by the
sensory ∩ prior_80ms strip, and panel c's two averages by the already-plotted
sensory example and combined trajectories.

Journal: journals/paper_main_figures.md
"""
from pathlib import Path

OPENALYX = Path.home() / 'Downloads/ONE/openalyx.internationalbrainlab.org'
ALYX = Path.home() / 'Downloads/ONE/alyx.internationalbrainlab.org'

FIG_DIR = OPENALYX / 'manifold' / 'paper_figs' / 'fig4'
ORIGINAL_PDF = OPENALYX / 'manifold' / 'figs' / 'figure4.pdf'
PAGE_WH = (990.29, 1127.96)   # points, from the PDF MediaBox

TABLE_PNG = (
    ALYX / 'meta'
    / 'table_act_block_combined_summary_act_p_mean_c_combinedpTrue_0.01_overall_earlystim80.png'
)
REGION_ORDER = ALYX / 'meta' / 'region_order.txt'
SC_TABLE = OPENALYX / 'manifold' / 'paper_figs' / 'fig2' / 'openalyx' / 'sc_table.csv'
EARLYSTIM_NPY = (
    ALYX / 'manifold' / 'res' / 'new'
    / 'combined_act_block_duringstim_r_act_block_duringstim_l_earlystim80.npy'
)
TRAJ_DIR = ALYX / 'manifold' / 'figs' / 'earlystim80_sensory_prior'

# stim / stim_early on type_stim (sc_duringstim_regtype 0 and 0.1)
SENSORY_REGTYPES = (0.0, 0.1)
ALPHA = 0.01
EXAMPLE = 'LGd'

# Content rows of the overall table, top to bottom (black bars cropped off).
# Order is the reverse of plot_combined_table_summary's display columns, with
# the region swatch at the bottom. y pixel ranges are on the 15849×1736 png.
TABLE_ROWS = [
    ('type_stim', 192, 398),
    ('prior_80ms', 400, 607),
    ('prior_stim', 609, 815),
    ('type_choice', 817, 1023),
    ('prior_choice', 1025, 1232),
    ('region', 1234, 1544),
]
TABLE_Y0, TABLE_Y1 = 190, 1546
