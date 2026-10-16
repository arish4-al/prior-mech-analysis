"""Paths and colours for the model-recovery SI figure.

The analysis itself is ``simulate_recovery.py`` (``--full-analysis`` and
``--unsplit-prior``). This figure reads those outputs. It does not
re-simulate.

Journal: journals/paper_si_figures.md
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

OPENALYX = Path.home() / 'Downloads/ONE/openalyx.internationalbrainlab.org'
STAGEB = OPENALYX / 'manifold_sim' / 'stageB_bwm'
FULL = OPENALYX / 'manifold_sim' / 'full_mpre3_meancell_s89'
FIG_DIR = OPENALYX / 'manifold' / 'paper_figs' / 'si_recovery'

ALPHA = 0.01
SIGMA_STIM = 0.8

# Same population colours as the Figure 4 rebuild.
C_S = '#57C1EB'
C_I = '#E8AC22'
C_M = '#E24A33'
C_INK = '#231F20'
C_SPLIT = '#6159A6'    # stim × choice (f1/f2)
C_STIMONLY = '#2BA9E0'  # stim side only

POP_COLOR = {'S': C_S, 'I': C_I, 'M': C_M}
PRED_NAME = {'S': 'stimulus', 'I': 'integrator', 'M': 'movement', 'none': 'unassigned'}

# Left: regular s101 (2026-08-13). Right: full mpre3 s89 (this run).
# Both: seed 123, 40 sessions, nrand 100, contrast-matched null.
CONDITIONS = (
    {
        'key': 'no_sensory_prior',
        'title': 'no sensory prior',
        'model': 'regular s101',
        'ground_truth': 'g_s = d_s = 0, fitted I/M prior',
        'class_csv': (
            STAGEB / 'goal1' / 'absence' / 'cm_full' / 'figs'
            / 'bwm_classification.csv'
        ),
        'split_s_csv': (
            STAGEB / 'goal1' / 'absence' / 'cm_full' / 'figs' / 's_prior_stats.csv'
        ),
        'unsplit_csv': (
            STAGEB / 'unsplit_prior' / 'seed_123' / 'absence_unsplit' / 'figs'
            / 'absence_unsplit_summary.csv'
        ),
    },
    {
        'key': 'full_model',
        'title': 'full model',
        'model': 'full mpre3 s89',
        'ground_truth': 'd_s = 49.7, g_s ≈ 0, g_i = 138',
        'class_csv': (
            FULL / 'goal1' / 'full_mpre3_s89' / 'cm_full' / 'figs'
            / 'bwm_classification.csv'
        ),
        'split_s_csv': (
            FULL / 'goal1' / 'full_mpre3_s89' / 'cm_full' / 'figs'
            / 's_prior_stats.csv'
        ),
        'unsplit_csv': None,  # presence_*_unsplit summary, filled in by the plot
    },
)
