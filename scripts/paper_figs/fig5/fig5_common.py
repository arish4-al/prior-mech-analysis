"""Shared geometry, colours and paths for the Figure 5 rebuild.

Original page: ``~/Downloads/figure5.pdf`` (1105.75 × 666.286 pt).
Axes boxes are the black spines measured from that PDF (top-left origin,
y down). ``xlim`` / ``ylim`` are the data limits of those spines, fit from
the tick marks so labelled ticks land on the published positions.

Panels a (circuit) and j (P–S–C–M flow) are left empty.

Journal: journals/paper_main_figures.md (2026-10-15 Figure 5).
"""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OPENALYX = Path.home() / 'Downloads/ONE/openalyx.internationalbrainlab.org'
FIG_DIR = OPENALYX / 'manifold' / 'paper_figs' / 'fig5'
ORIGINAL_PDF = Path.home() / 'Downloads' / 'figure5.pdf'
PAGE_WH = (1105.75, 666.286)  # points, PDF MediaBox

# 80 ms full family, ranked on tot80. See journals/stage_b_fit_variants.md.
RUN_DIR = (
    OPENALYX / 'models'
    / 'weights_run_fj_stageB_hold_s89_full_mpre3_meancell_full_masknone_s89'
)
WEIGHTS_JSON = RUN_DIR / 'weights_final_loss1p125_20260915-224209.json'
BEHAVIOR_ACT = OPENALYX / 'manifold' / 'res' / 'behavior_actprior.npy'
PSYCH_CACHE = RUN_DIR / 'psychometric_model_vs_data_actprior' / 'combined_results.npy'

# Spines (x0, y0, x1, y1), top-left origin. Event line sits at t = 0.
AX_B = (238.9, 11.0, 383.4, 133.5)
AX_C_STIM = (445.1, 9.4, 587.7, 133.0)
AX_C_MOVE = (615.9, 9.4, 758.5, 133.1)
AX_D_STIM = (822.5, 11.9, 957.7, 132.9)
AX_D_MOVE = (965.0, 13.1, 1100.1, 133.0)
AX_E = (85.1, 192.1, 354.9, 316.7)
AX_F = (451.2, 189.5, 700.6, 315.8)
AX_G = (820.9, 187.1, 1066.0, 315.7)
AX_I_PERF = (72.2, 499.2, 327.5, 629.3)
AX_I_RT = (442.4, 502.8, 690.2, 629.0)
# h has no spines; the trace strip between the grey block-side levels.
AX_H = (19.4, 355.0, 1073.2, 478.0)

# Data limits of the spines (tick-mark fit).
LIM_B = ((-6.6, 132.6), (-0.140, 0.75))
LIM_C_STIM = ((-7.1, 149.2), (-0.026, 0.548))
LIM_C_MOVE = ((-149.2, 7.1), (-0.020, 0.843))
LIM_D_STIM = ((-4.0, 84.1), (0.033, 0.151))
LIM_D_MOVE = ((-81.1, 3.6), (0.033, 0.151))
LIM_E = ((-6.4, 134.5), (-0.051, 1.215))
LIM_F = ((-4.5, 120.0), (-0.030, 0.304))
LIM_G = ((-151.0, 5.0), (-0.043, 0.801))
LIM_I_PERF = ((-2.0, 2.0), (23.5, 103.6))
LIM_I_RT = ((-2.0, 2.0), (0.086, 0.470))

TICK_B_X = (0, 40, 80, 120)
TICK_B_Y = (0.0, 0.25, 0.50, 0.75)
TICK_C_STIM_Y = (0.0, 0.15, 0.30, 0.45)
TICK_C_MOVE_X = (-120, -80, -40, 0)
TICK_C_MOVE_Y = (0.0, 0.25, 0.50, 0.75)
TICK_D_X = (0, 25, 50, 75)
TICK_D_X_MOVE = (-75, -50, -25, 0)
TICK_D_Y = (0.06, 0.09, 0.12, 0.15)
TICK_E_X = (0, 40, 80, 120)
TICK_E_Y = (0.0, 0.4, 0.8, 1.2)
TICK_F_X = (0, 40, 80, 120)
TICK_F_Y = (0.0, 0.1, 0.2, 0.3)
TICK_G_X = (-120, -80, -40, 0)
TICK_G_Y = (0.0, 0.25, 0.50, 0.75)
TICK_CONTRAST = (-1, -0.1, 0, 0.1, 1)
TICK_PERF_Y = (25, 50, 75, 100)
TICK_RT_Y = (0.15, 0.30, 0.45)

# PDF stroke / fill colours.
C_INK = '#231F20'
C_STIM_ON = '#27AAE1'       # (0.152, 0.667, 0.884), 2 pt
C_MOVE_ON = '#DC7F6B'       # (0.863, 0.498, 0.421), 2 pt
C_S_BADGE = '#A2DDF8'       # (0.636, 0.865, 0.973)
C_C_BADGE = '#FEE499'       # (0.996, 0.896, 0.601)
C_M_BADGE = '#DC7F6B'
C_R2_S = '#4B87C6'
C_R2_I = '#DAA628'
C_R2_M = '#CC5827'
C_R2_F = '#FCD704'          # gold, panel f
C_R2_G = '#F2644B'          # tomato, panel g
C_I_DARK = '#DAA520'
C_I_LIGHT = '#FFD700'
C_M_DARK = '#CC5500'
C_M_LIGHT = '#FF7F0E'
C_PRIOR_I = 'gold'
C_PRIOR_M = 'tomato'
C_H_STIM_LINE = '#F08F21'   # obs-onset dashes in h
C_H_MOVE_LINE = '#1B9CD8'   # action dashes in h
C_S_TRACE = '#4A90E2'
C_I_TRACE = '#FFC400'
C_M_TRACE = '#FF0000'
C_P_TRACE = '#8B2BE2'
C_BLOCK = '#C0C0C0'

# Badge centres (x, y) in PDF points, top-left origin, and letter.
BADGES = (
    (258.8, 21.9, 'S', C_S_BADGE),   # b
    (516.0, 21.9, 'C', C_C_BADGE),   # c stim
    (546.5, 21.5, 'M', C_M_BADGE),
    (688.3, 21.9, 'C', C_C_BADGE),   # c move
    (718.8, 21.5, 'M', C_M_BADGE),
    (877.1, 42.0, 'C', C_C_BADGE),   # d
    (907.6, 41.6, 'M', C_M_BADGE),
    (111.6, 195.2, 'S', C_S_BADGE),  # e
    (476.4, 195.1, 'C', C_C_BADGE),  # f
    (841.5, 195.2, 'M', C_M_BADGE),  # g
)

# Panel letters (x, y of the glyph top-left-ish; we centre on the PDF bbox).
LETTERS = (
    (5.4, 9.2, 'a'),
    (192.0, 9.8, 'b'),
    (393.4, 9.8, 'c'),
    (766.9, 9.8, 'd'),
    (35.0, 185.2, 'e'),
    (402.1, 185.2, 'f'),
    (762.8, 185.3, 'g'),
    (4.9, 355.9, 'h'),
    (8.3, 498.2, 'i'),
    (744.2, 497.2, 'j'),
)


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
