"""Model vs data performance and RT psychometrics for baseline s333.

Matches paper-brain-wide-map/model_test.ipynb (loss_perf_with_data overlay of
model dashed vs behavior.npy solid). Performance is split by true-block
congruency. RT is plotted both as the notebook default (combined_all) and as
the split concordant/discordant curves from the same notebook’s data cell.

More sessions than the bps=20 eval stim, then combine_run_results, to cut
model-curve noise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from plot_best_fit_results import (  # noqa: E402
    load_plot_model,
    make_shared_stimuli,
)
from model_functions import combine_run_results, loss_perf_with_data, run_model  # noqa: E402

SEEDS_BEST = 333
N_SESSIONS = 10
BPS = 20
STIM_SEED0 = 12345
BASE = Path.home() / (
    "Downloads/ONE/openalyx.internationalbrainlab.org/models"
)
BEHAVIOR = Path.home() / (
    "Downloads/ONE/openalyx.internationalbrainlab.org/manifold/res/behavior.npy"
)


def latest_final(d: Path) -> Path:
    finals = sorted(d.glob("weights_final_*.json"))
    if not finals:
        raise FileNotFoundError(d)
    return finals[-1]


def run_many_sessions(mp, n_sessions: int, bps: int, seed0: int):
    runs = []
    n_trials = 0
    for i in range(n_sessions):
        (
            stimuli, trial_strengths, trial_sides, block_sides,
            steps_before_obs, bps_out,
        ) = make_shared_stimuli(mp, bps=bps, seed=seed0 + i)
        results = run_model(
            "data",
            stimuli,
            trial_strengths,
            trial_sides,
            block_sides,
            bps_out,
            steps_before_obs=steps_before_obs,
            verbose=False,
            backend="numba",
            **mp,
        )
        n_ok = int(np.sum(np.asarray(results["choices"]) != 0))
        n_trials += n_ok
        print(f"  session {i + 1}/{n_sessions}  trials={n_ok}")
        runs.append(results)
    return combine_run_results(runs), n_trials


def save_current(out_stem: Path):
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.gcf()
    fig.set_size_inches(6.2, 4.2)
    ax = fig.axes[0]
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".svg"), bbox_inches="tight", transparent=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=160, bbox_inches="tight",
                transparent=False, facecolor="white")
    plt.close(fig)


def main():
    run = BASE / f"weights_run_fj_stageB_hold_s89_regular_mask12-13_s{SEEDS_BEST}"
    jp = latest_final(run)
    mp, meta = load_plot_model(jp)
    dt_ms = float(mp["dt"])
    dt_s = dt_ms / 1000.0
    out_dir = run / "psychometric_model_vs_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"model {jp.parent.name}  rec={meta.get('loss')}  dt={dt_ms} ms")
    cache = out_dir / "combined_results.npy"
    if cache.is_file():
        results = np.load(cache, allow_pickle=True).item()
        n_trials = int(np.sum(np.asarray(results["choices"]) != 0))
        print(f"loaded {cache.name}  committed trials={n_trials}")
    else:
        print(f"sim {N_SESSIONS} sessions × {BPS} blocks  seed0={STIM_SEED0}")
        results, n_trials = run_many_sessions(mp, N_SESSIONS, BPS, STIM_SEED0)
        print(f"combined committed trials={n_trials}")
        slim = {
            k: results[k]
            for k in (
                "trial_strengths", "trial_sides", "block_sides",
                "correct_action_taken", "reaction_time", "choices",
            )
        }
        np.save(cache, slim, allow_pickle=True)

    behavior = np.load(BEHAVIOR, allow_pickle=True).flat[0]

    # Notebook cell 58: performance, true-block congruent vs incongruent.
    sse_correct = loss_perf_with_data(
        results, behavior, mp,
        metric="correct", dt=1.0, do_plot=True, save_dir=None, log_xaxis=True,
    )
    save_current(out_dir / "performance_model_vs_data")
    print("performance SSE", sse_correct)

    # Notebook cell 58 default RT (one combined curve).
    sse_rt = loss_perf_with_data(
        results, behavior, mp,
        metric="rt", dt=dt_s, do_plot=True, save_dir=None, log_xaxis=True,
        rt_mode="combined_all",
    )
    save_current(out_dir / "rt_model_vs_data")
    print("rt combined SSE", sse_rt)

    # Same function, split like the notebook’s data psychometric (cell 64).
    sse_rt_split = loss_perf_with_data(
        results, behavior, mp,
        metric="rt", dt=dt_s, do_plot=True, save_dir=None, log_xaxis=True,
        rt_mode="split_all",
    )
    save_current(out_dir / "rt_split_model_vs_data")
    print("rt split SSE", sse_rt_split)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
