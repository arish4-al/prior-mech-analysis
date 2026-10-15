"""Shared paths / constants for the Figure 2 rebuild (stim and choice
sensitivity, function assignments).

Original pipeline (analysis_figs.ipynb, paper-brain-wide-map/bwm_figs.ipynb):
  a-c  analysis_functions.plot_regional_distance(plot_gain=False, plot_offset=False)
  e    analysis_figs.ipynb cell 6 on get_sc_table(...) (cell 5)
  f    analysis_functions.plot_sc_table(metric='regtype') region-type table

All of them read openalyx manifold/res. ``VERSIONS`` adds the alyx res/new
min5 choice L-R shuffle (2026-08-14) as a second choice source; stim is from
openalyx in both.

Journal: journals/paper_main_figures.md
"""
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OPENALYX = Path.home() / 'Downloads/ONE/openalyx.internationalbrainlab.org'
ALYX = Path.home() / 'Downloads/ONE/alyx.internationalbrainlab.org'
OA_RES = OPENALYX / 'manifold' / 'res'
PAPER_FIGS = OPENALYX / 'manifold' / 'paper_figs'
FIG_DIR = PAPER_FIGS / 'fig2'
ORIGINAL_PDF = FIG_DIR / 'original' / 'figure2.pdf'

SC_TIMES = ['stim_duringstim_act', 'choice_duringstim_act',
            'stim_duringchoice_act', 'choice_duringchoice_act']
STIM_EXTRA = ['stim_duringstim_short_act', 'stim_duringstim1_act']
CHOICE_TIMES = ['choice_duringstim_act', 'choice_duringchoice_act']

# get_sc_table parameters of analysis_figs.ipynb cells 3 / 5
PTYPE = 'p_mean_c'
ALPHA = 0.01
SC_THRESHOLD = 0.0
SLOPE_THRESHOLD = 0.05
AMP_LOC_THRESHOLD = 67
N_SLOPE = 20

VERSIONS = {
    'openalyx': {'choice_res': OA_RES,
                 'desc': 'stim + choice from openalyx manifold/res (published fig2 files)'},
    'min5': {'choice_res': ALYX / 'manifold' / 'res' / 'new',
             'desc': 'stim from openalyx; choice L-R from alyx res/new min5 shuffle (2026-08-14)'},
}

C_STIM = '#57C1EB'
C_STIM_EARLY = '#A6DBF0'
C_INTEG = '#feda7e'
C_INTEG_HIST = '#fec65f'
C_MOVE = '#d55607'
C_STIM_ON = '#2BA9E0'
C_MOVE_ON = '#E8806A'


def version_dir(version):
    return FIG_DIR / version


def staging_dir(version):
    return version_dir(version) / 'staging'


def load_af(res_dir):
    """Import analysis_functions with ONE in local mode and point it at res_dir."""
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


def build_staging(af, version):
    """Copy the small combined dicts (get_sc_table rewrites them) and symlink the
    read-only regde arrays into the version's staging dir."""
    stage = staging_dir(version)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    shutil.copy2(OA_RES / 'act_block_only.csv', stage / 'act_block_only.csv')
    choice_res = VERSIONS[version]['choice_res']
    for tf in SC_TIMES + STIM_EXTRA:
        src_root = choice_res if tf in CHOICE_TIMES else OA_RES
        name, regde = combined_stems(af, tf)
        for stem, link in ((name, False), (regde, True)):
            src = src_root / f'{stem}.npy'
            if not src.exists():
                raise FileNotFoundError(src)
            dst = stage / f'{stem}.npy'
            if link:
                dst.symlink_to(src)
            else:
                shutil.copy2(src, dst)
    return stage


def savefig_svg_png(fig, path_no_ext, dpi=150, tight=False):
    path_no_ext = Path(path_no_ext)
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(facecolor='white', bbox_inches='tight' if tight else None)
    fig.savefig(path_no_ext.with_suffix('.svg'), **kw)
    fig.savefig(path_no_ext.with_suffix('.png'), dpi=dpi, **kw)
