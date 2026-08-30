"""Model vs data performance and RT psychometrics for well-fit baselines.

Matches paper-brain-wide-map/model_test.ipynb (loss_perf_with_data overlay).
Concordant / discordant:

  * data — action-kernel prior (α=0.2) × stim side
  * model — binarized trial-average P (mean P_L−P_R over the trial)

True-block overlays stay in psychometric_model_vs_data/. These go in
psychometric_model_vs_data_actprior/.

More sessions than the bps=20 eval stim, then combine_run_results, to cut
model-curve noise. Default seeds: s333 plus the next-best eval totals
(s101, s34, s12).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_choice_epochs import (  # noqa: E402
    ACT_ALPHA,
    action_kernel_binary,
    load_sessions_from_aggregate,
    stim_side_labels,
)
from plot_best_fit_results import (  # noqa: E402
    load_plot_model,
    make_shared_stimuli,
)
from model_functions import combine_run_results, loss_perf_with_data, run_model  # noqa: E402

# s333 best eval tot; 101 / 34 / 12 are the next cluster (~1.017–1.026).
SEEDS = (333, 101, 34, 12)
N_SESSIONS = 10
BPS = 20
STIM_SEED0 = 12345
BASE = Path.home() / (
    "Downloads/ONE/openalyx.internationalbrainlab.org/models"
)
ALYX = Path.home() / "Downloads/ONE/alyx.internationalbrainlab.org"
BEHAVIOR_ACT = Path.home() / (
    "Downloads/ONE/openalyx.internationalbrainlab.org/manifold/res/"
    "behavior_actprior.npy"
)
WEIGHTS_REL = (
    BASE
    / "weights_run_20251125_182058"
    / "weights_2stagelocalrefine_loss0p4044_20251125-195255.json"
)
OUT_SUBDIR = "psychometric_model_vs_data_actprior"
# Notebook / behavior.npy contrast order: L→center→R.
CONTRASTS_DESC = (1.0, 0.25, 0.125, 0.0625, 0.0)
TITLE_SUFFIX = "data: act prior; model: binarized trial-avg P"


def latest_final(d: Path) -> Path:
    finals = sorted(d.glob("weights_final_*.json"))
    if not finals:
        raise FileNotFoundError(d)
    return finals[-1]


def snap_mag(mag: float) -> float:
    arr = np.asarray(CONTRASTS_DESC, dtype=float)
    return float(arr[np.argmin(np.abs(arr - abs(float(mag))))])


def stim_bin(mag: float, stim_left: bool) -> int:
    mag = snap_mag(mag)
    if mag == 0.0:
        return 4
    i = CONTRASTS_DESC.index(mag)
    return i if stim_left else 8 - i


def _sp_sign(x: float) -> float:
    """Match model_functions._sp_sign: P_L−P_R < 0 → +1 (right), else −1 (left)."""
    return 1.0 if float(x) < 0.0 else -1.0


def results_with_subjective_prior(results: dict) -> dict:
    """Replace true block_sides with binarized trial-average P."""
    out = dict(results)
    new_blocks = []
    for i, sp in enumerate(results["sub_prior"]):
        arr = np.asarray(sp, dtype=float).reshape(-1)
        sign = _sp_sign(arr[0])
        n = len(results["block_sides"][i])
        new_blocks.append(np.full(n, sign, dtype=float))
    out["block_sides"] = new_blocks
    return out


def build_actprior_behavior(sessions: dict[str, pd.DataFrame]) -> dict:
    """Same 9-bin cc/cw/dc/dw tables as behavior.npy, but act-prior congruency.

    Per session: drop unbiased (probabilityLeft==0.5), then α=0.2 action kernel
    on remaining choices (IBL +1 = left). Keep committed choices and
    feedbackType ±1. RT = firstMovement − stimOn.
    """
    total = {k: np.zeros(9, dtype=float) for k in ("cc", "dc", "cw", "dw")}
    res = {k: np.zeros(9, dtype=float) for k in ("cc", "dc", "cw", "dw")}
    n_sess = n_tr = 0
    for _eid, df in sessions.items():
        d = df.sort_values("stimOn_times").reset_index(drop=True)
        pleft = np.asarray(d["probabilityLeft"].to_numpy(), dtype=float)
        keep = ~np.isclose(pleft, 0.5)
        if int(keep.sum()) < 2:
            continue
        d = d.loc[keep].reset_index(drop=True)
        choice = np.asarray(d["choice"].to_numpy(), dtype=float)
        act = action_kernel_binary(np.nan_to_num(choice, nan=0.0), alpha=ACT_ALPHA)
        stim = stim_side_labels(d["contrastLeft"], d["contrastRight"])
        cl = np.asarray(d["contrastLeft"].to_numpy(), dtype=float)
        cr = np.asarray(d["contrastRight"].to_numpy(), dtype=float)
        fb = np.asarray(d["feedbackType"].to_numpy(), dtype=float)
        rt = (
            np.asarray(d["firstMovement_times"].to_numpy(), dtype=float)
            - np.asarray(d["stimOn_times"].to_numpy(), dtype=float)
        )
        used = False
        for i in range(len(d)):
            if stim[i] not in ("L", "R"):
                continue
            if not np.isfinite(choice[i]) or abs(choice[i]) != 1:
                continue
            if fb[i] not in (1.0, -1.0):
                continue
            if not np.isfinite(rt[i]):
                continue
            stim_left = stim[i] == "L"
            mag = cl[i] if stim_left else cr[i]
            if not np.isfinite(mag):
                continue
            prior_left = bool(np.isclose(act[i], 0.8))
            cong = "c" if stim_left == prior_left else "d"
            corr = "c" if fb[i] == 1.0 else "w"
            key = f"{cong}{corr}"
            j = stim_bin(float(mag), stim_left)
            total[key][j] += 1.0
            res[key][j] += float(rt[i])
            n_tr += 1
            used = True
        if used:
            n_sess += 1

    acc_cong = np.divide(
        total["cc"], total["cc"] + total["cw"],
        out=np.zeros(9), where=(total["cc"] + total["cw"]) > 0,
    )
    acc_disc = np.divide(
        total["dc"], total["dc"] + total["dw"],
        out=np.zeros(9), where=(total["dc"] + total["dw"]) > 0,
    )
    acc_all = np.divide(
        total["cc"] + total["dc"],
        total["cc"] + total["dc"] + total["cw"] + total["dw"],
        out=np.zeros(9),
        where=(total["cc"] + total["dc"] + total["cw"] + total["dw"]) > 0,
    )
    print(f"act-prior data: sessions={n_sess}  trials={n_tr}")
    return {
        "trial_counts": total,
        "reaction_times": res,
        "pct_correct": {
            "congruent": acc_cong,
            "discordant": acc_disc,
            "overall": acc_all,
        },
        "n_sessions": n_sess,
        "n_trials": n_tr,
        "prior": "act",
        "alpha": ACT_ALPHA,
    }


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


def save_current(out_stem: Path, title: str | None = None):
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.gcf()
    fig.set_size_inches(6.2, 4.2)
    ax = fig.axes[0]
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".svg"), bbox_inches="tight", transparent=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=160, bbox_inches="tight",
                transparent=False, facecolor="white")
    plt.close(fig)


def gof(sse, *keys):
    cur = sse.get("gof", sse)
    for k in keys:
        cur = cur[k]
    v = float(cur)
    return v if np.isfinite(v) else np.nan


def plot_one_json(label: str, jp: Path, run: Path, behavior) -> dict:
    mp, meta = load_plot_model(jp)
    dt_ms = float(mp["dt"])
    dt_s = dt_ms / 1000.0
    out_dir = run / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    rec = meta.get("loss")
    print(f"\n=== {label}  {jp.name}  rec={rec}  dt={dt_ms} ms ===")
    print(f"  g_i={mp['g_i']:.3g}  d_i={mp['d_i']:.3g}  "
          f"theta_c={mp['action_thresholds']['concordant'][0.0]:.3f}  "
          f"theta_d={mp['action_thresholds']['discordant'][0.0]:.3f}")
    cache = out_dir / "combined_results.npy"
    results = None
    if cache.is_file():
        loaded = np.load(cache, allow_pickle=True).item()
        if "sub_prior" in loaded:
            results = loaded
            n_trials = int(np.sum(np.asarray(results["choices"]) != 0))
            print(f"loaded {cache.name}  committed trials={n_trials}")
        else:
            print(f"{cache.name} missing sub_prior; re-sim")
    if results is None:
        print(f"sim {N_SESSIONS} sessions × {BPS} blocks  seed0={STIM_SEED0}")
        results, n_trials = run_many_sessions(mp, N_SESSIONS, BPS, STIM_SEED0)
        print(f"combined committed trials={n_trials}")
        slim = {
            k: results[k]
            for k in (
                "trial_strengths", "trial_sides", "block_sides",
                "correct_action_taken", "reaction_time", "choices",
                "sub_prior",
            )
        }
        np.save(cache, slim, allow_pickle=True)

    results = results_with_subjective_prior(results)

    sse_correct = loss_perf_with_data(
        results, behavior, mp,
        metric="correct", dt=1.0, do_plot=True, save_dir=None, log_xaxis=True,
    )
    save_current(out_dir / "performance_model_vs_data",
                 f"performance ({TITLE_SUFFIX})")

    sse_rt = loss_perf_with_data(
        results, behavior, mp,
        metric="rt", dt=dt_s, do_plot=True, save_dir=None, log_xaxis=True,
        rt_mode="combined_all",
    )
    save_current(out_dir / "rt_model_vs_data",
                 f"reaction time ({TITLE_SUFFIX})")

    sse_rt_split = loss_perf_with_data(
        results, behavior, mp,
        metric="rt", dt=dt_s, do_plot=True, save_dir=None, log_xaxis=True,
        rt_mode="split_all",
    )
    save_current(out_dir / "rt_split_model_vs_data",
                 f"reaction time ({TITLE_SUFFIX})")
    print(f"wrote {out_dir}")
    return {
        "label": label,
        "recorded": rec,
        "n_trials": n_trials,
        "g_i": float(mp["g_i"]),
        "theta_c": float(mp["action_thresholds"]["concordant"][0.0]),
        "theta_d": float(mp["action_thresholds"]["discordant"][0.0]),
        "perf_r2": gof(sse_correct, "total"),
        "rt_r2": gof(sse_rt, "total"),
        "rt_split_r2": gof(sse_rt_split, "total"),
        "rt_split_r2_con": gof(sse_rt_split, "congruent"),
        "rt_split_r2_inc": gof(sse_rt_split, "incongruent"),
        "out_dir": str(out_dir),
    }


def plot_one_seed(seed: int, behavior) -> dict:
    run = BASE / f"weights_run_fj_stageB_hold_s89_regular_mask12-13_s{seed}"
    return plot_one_json(f"s{seed}", latest_final(run), run, behavior)


def main():
    print("loading BWM trials.pqt for act-prior data curves …")
    sessions = load_sessions_from_aggregate(ALYX)
    behavior = build_actprior_behavior(sessions)
    BEHAVIOR_ACT.parent.mkdir(parents=True, exist_ok=True)
    np.save(BEHAVIOR_ACT, behavior, allow_pickle=True)
    print(f"wrote {BEHAVIOR_ACT}")

    rows = [plot_one_seed(seed, behavior) for seed in SEEDS]
    rows.append(plot_one_json(
        "WEIGHTS_REL", WEIGHTS_REL, WEIGHTS_REL.parent, behavior,
    ))
    print("\nact-prior / subj-P  "
          "model        rec     n_tr  g_i    θc/θd       "
          "perf R²  RT comb R²  RT split R²  (con / inc)")
    for r in rows:
        print(
            f"{r['label']:<12} {r['recorded']:6.3f} {r['n_trials']:5d}  "
            f"{r['g_i']:5.1f}  {r['theta_c']:.3f}/{r['theta_d']:.3f}  "
            f"{r['perf_r2']:7.3f}  {r['rt_r2']:10.3f}  {r['rt_split_r2']:11.3f}  "
            f"({r['rt_split_r2_con']:.3f} / {r['rt_split_r2_inc']:.3f})"
        )


if __name__ == "__main__":
    main()
