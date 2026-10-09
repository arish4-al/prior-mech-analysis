"""S / I/M / prior / ITI + act-prior RT for s89ft d_i punch (di / wiigi).

Same shared stim as the eval (bps=20, seed 12345, from regular s101).
Plots go in each run dir. Also writes prior_effects_80ms.svg/png.
"""
from __future__ import annotations

import os
from pathlib import Path

from _tmp_s89ft_imshape_plots import main as _plots_main  # noqa: F401

# Re-exec with this module's ARMS by patching the plot driver namespace.
import _tmp_s89ft_imshape_plots as drv  # noqa: E402

drv.ARMS = {
    "di": (
        "weights_run_fj_stageB_hold_s89_full_s89ft_di_choicei_meancell_full_mask0-1-2-3-4-5-6-7-9-10-11-12-13-14-15-16-17-18-19-20",
        True,
    ),
    "wiigi": (
        "weights_run_fj_stageB_hold_s89_full_s89ft_wiigi_choicei_meancell_full_mask1-3-4-10-11-12-13-14-15-16-17-18-19-20",
        True,
    ),
}
drv.SEEDS = (89, 7, 12, 45)
drv.OUT = Path(os.environ.get(
    "PLOT_OUT",
    str(drv.BASE / "stageB_hold_s89_s89ft_di_choicei_plot_summary.json"),
))


if __name__ == "__main__":
    drv.main()
