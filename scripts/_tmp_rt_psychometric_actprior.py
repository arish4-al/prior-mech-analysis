"""RT vs signed contrast, concordant vs discordant by act-prior × stim side.

Model: best this-batch baseline (s333). Data: BWM trials, first vs last 50%
of each session (pooled) overlaid on one axes.
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
from model_functions import run_model  # noqa: E402

SEEDS_BEST = 333
BASE = Path.home() / (
    "Downloads/ONE/openalyx.internationalbrainlab.org/models"
)
ALYX = Path.home() / "Downloads/ONE/alyx.internationalbrainlab.org"
CONTRASTS = np.array([0.0, 0.0625, 0.125, 0.25, 1.0])
SIGNED = np.array([-1.0, -0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25, 1.0])


def latest_final(d: Path) -> Path:
    finals = sorted(d.glob("weights_final_*.json"))
    if not finals:
        raise FileNotFoundError(d)
    return finals[-1]


def snap_signed(mag: float, side: float) -> float:
    mag = abs(float(mag))
    snapped = float(CONTRASTS[np.argmin(np.abs(CONTRASTS - mag))])
    if snapped == 0.0:
        return 0.0
    return snapped * (1.0 if side > 0 else -1.0)


def mean_sem(groups: dict[str, dict[float, list[float]]]):
    """groups[key][signed] -> list of RT (s)."""
    out = {}
    for key in ("concordant", "discordant"):
        ys, es, ns = [], [], []
        for s in SIGNED:
            v = np.asarray(groups[key].get(float(s), []), dtype=float)
            v = v[np.isfinite(v)]
            ns.append(int(v.size))
            if v.size == 0:
                ys.append(np.nan)
                es.append(np.nan)
            else:
                ys.append(float(np.mean(v)))
                es.append(float(np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0)
        out[key] = (np.array(ys), np.array(es), np.array(ns))
    return out


def empty_groups():
    return {
        "concordant": {float(s): [] for s in SIGNED},
        "discordant": {float(s): [] for s in SIGNED},
    }


def add_trial(groups, signed, concordant, rt):
    if not np.isfinite(rt):
        return
    key = "concordant" if concordant else "discordant"
    groups[key][float(signed)].append(float(rt))


def style_ax(ax, title: str, ylabel: str = "reaction time (s)"):
    ax.set_xscale("symlog", linthresh=0.05, linscale=0.6)
    ax.set_xticks(SIGNED)
    ax.set_xticklabels(
        ["−1", "−0.25", "−0.125", "−0.0625", "0", "0.0625", "0.125", "0.25", "1"],
        rotation=45,
        ha="right",
    )
    ax.set_xlim(-1.6, 1.6)
    ax.set_xlabel("signed contrast (left − / right +)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_fig(fig, out_stem: Path):
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_two_curves(stats, title, out_stem: Path, prior_label: str):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    colors = {"concordant": "#2166ac", "discordant": "#b2182b"}
    labels = {
        "concordant": f"concordant (stim = {prior_label})",
        "discordant": f"discordant (stim ≠ {prior_label})",
    }
    for key in ("concordant", "discordant"):
        y, e, n = stats[key]
        ax.errorbar(
            SIGNED, y, yerr=e, fmt="-o", color=colors[key],
            label=labels[key], capsize=3, lw=1.6, ms=5,
        )
    style_ax(ax, title)
    fig.tight_layout()
    _save_fig(fig, out_stem)


def plot_four_curves(st_first, st_last, title, out_stem: Path, prior_label: str):
    """Overlay first/last 50% × concordant/discordant on one axes."""
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    series = (
        ("first 50%", st_first, "concordant", "#2166ac", "-o", True),
        ("first 50%", st_first, "discordant", "#b2182b", "-o", True),
        ("last 50%", st_last, "concordant", "#67a9cf", "--s", False),
        ("last 50%", st_last, "discordant", "#ef8a62", "--s", False),
    )
    for half, stats, key, color, fmt, filled in series:
        y, e, _n = stats[key]
        rel = "=" if key == "concordant" else "≠"
        ax.errorbar(
            SIGNED, y, yerr=e, fmt=fmt, color=color,
            label=f"{half} {key} (stim {rel} {prior_label})",
            capsize=3, lw=1.6, ms=5,
            mfc=color if filled else "white",
        )
    style_ax(ax, title)
    fig.tight_layout()
    _save_fig(fig, out_stem)


def model_groups(results, dt_ms: float, prior: str):
    """prior: 'act' (action kernel on model choices) or 'block' (true block)."""
    # Model stores +1 = right, −1 = left (see run_model comments).
    ch_model = np.asarray(results["choices"], dtype=float)
    ch_bwm = np.where(ch_model == 0, 0.0, -ch_model)
    sides = np.array([ts[0] for ts in results["trial_sides"]], dtype=float)
    blocks = np.array([bs[0] for bs in results["block_sides"]], dtype=float)
    mags = np.array([ts[0] for ts in results["trial_strengths"]], dtype=float)
    rt_steps = np.asarray(results["reaction_time"], dtype=float)
    rt_s = rt_steps * float(dt_ms) / 1000.0
    if prior == "act":
        act = action_kernel_binary(ch_bwm, alpha=ACT_ALPHA)
        prior_left = np.isclose(act, 0.8)
    elif prior == "block":
        prior_left = blocks < 0  # −1 left, +1 right
    else:
        raise ValueError(prior)
    groups = empty_groups()
    n_ok = 0
    for i in range(len(ch_bwm)):
        if ch_bwm[i] == 0:
            continue
        side = sides[i]  # −1 left, +1 right
        stim_left = side < 0
        signed = snap_signed(mags[i], side)
        add_trial(groups, signed, stim_left == bool(prior_left[i]), rt_s[i])
        n_ok += 1
    return groups, n_ok


def data_groups(sessions: dict[str, pd.DataFrame], prior: str):
    first = empty_groups()
    last = empty_groups()
    n_sess = 0
    n_first = n_last = 0
    for eid, df in sessions.items():
        d = df.sort_values("stimOn_times").reset_index(drop=True)
        n = len(d)
        if n < 20:
            continue
        choice = np.asarray(d["choice"].to_numpy(), dtype=float)
        choice_ak = np.nan_to_num(choice, nan=0.0)
        act = action_kernel_binary(choice_ak, alpha=ACT_ALPHA)
        pleft = np.asarray(d["probabilityLeft"].to_numpy(), dtype=float)
        stim = stim_side_labels(d["contrastLeft"], d["contrastRight"])
        cl = np.asarray(d["contrastLeft"].to_numpy(), dtype=float)
        cr = np.asarray(d["contrastRight"].to_numpy(), dtype=float)
        mag = np.where(stim == "L", cl, np.where(stim == "R", cr, np.nan))
        rt = (
            np.asarray(d["firstMovement_times"].to_numpy(), dtype=float)
            - np.asarray(d["stimOn_times"].to_numpy(), dtype=float)
        )
        n_half = n // 2
        if n_half < 5:
            continue
        n_sess += 1
        for i in range(n):
            if stim[i] not in ("L", "R") or not np.isfinite(mag[i]):
                continue
            if not np.isfinite(choice[i]) or abs(choice[i]) != 1:
                continue
            stim_left = stim[i] == "L"
            if prior == "act":
                prior_left = np.isclose(act[i], 0.8)
            elif prior == "block":
                if np.isclose(pleft[i], 0.8):
                    prior_left = True
                elif np.isclose(pleft[i], 0.2):
                    prior_left = False
                else:
                    continue  # 0.5: no left/right block
            else:
                raise ValueError(prior)
            side = -1.0 if stim_left else 1.0
            signed = snap_signed(float(mag[i]), side)
            bucket = first if i < n_half else last
            add_trial(bucket, signed, stim_left == prior_left, rt[i])
            if i < n_half:
                n_first += 1
            else:
                n_last += 1
    return first, last, n_sess, n_first, n_last


def dump_table(name, st):
    print(f"\n{name} mean RT (s)")
    print(f"{'signed':>8} {'conc':>8} {'n_c':>6} {'disc':>8} {'n_d':>6}")
    yc, ec, nc = st["concordant"]
    yd, ed, nd = st["discordant"]
    for s, a, na, b, nb in zip(SIGNED, yc, nc, yd, nd):
        print(f"{s:8.4f} {a:8.3f} {na:6d} {b:8.3f} {nb:6d}")


def main():
    run = BASE / f"weights_run_fj_stageB_hold_s89_regular_mask12-13_s{SEEDS_BEST}"
    jp = latest_final(run)
    mp, meta = load_plot_model(jp)
    dt_ms = float(mp["dt"])
    stim_bundle = make_shared_stimuli(mp, bps=20, seed=12345)
    (
        stimuli, trial_strengths, trial_sides, block_sides,
        steps_before_obs, bps,
    ) = stim_bundle
    print(f"model {jp.parent.name}  rec={meta.get('loss')}  dt={dt_ms} ms")
    results = run_model(
        "data",
        stimuli,
        trial_strengths,
        trial_sides,
        block_sides,
        bps,
        steps_before_obs=steps_before_obs,
        verbose=False,
        backend="numba",
        **mp,
    )
    print("loading BWM trials.pqt …")
    sessions = load_sessions_from_aggregate(ALYX)
    print(f"sessions: {len(sessions)}")

    s333 = BASE / f"weights_run_fj_stageB_hold_s89_regular_mask12-13_s{SEEDS_BEST}"
    specs = (
        (
            "act",
            "act prior",
            s333 / "rt_psychometric_actprior",
            "act-prior",
        ),
        (
            "block",
            "true block",
            s333 / "rt_psychometric_trueblock",
            "true-block",
        ),
    )
    for prior, prior_label, out_dir, tag in specs:
        g_model, n_model = model_groups(results, dt_ms, prior)
        st_model = mean_sem(g_model)
        print(f"\n=== {tag}  model trials={n_model} ===")
        plot_two_curves(
            st_model,
            f"Model RT  ·  baseline s{SEEDS_BEST}  ·  {tag} conc/disc",
            out_dir / "model_rt_baseline_s333",
            prior_label,
        )
        g0, g1, n_sess, n0, n1 = data_groups(sessions, prior)
        print(f"sessions used={n_sess}  first-half={n0}  last-half={n1}")
        plot_four_curves(
            mean_sem(g0),
            mean_sem(g1),
            f"Data RT  ·  first vs last 50%  ·  {tag} conc/disc",
            out_dir / "data_rt_first_last50",
            prior_label,
        )
        for stem in ("data_rt_first50", "data_rt_last50"):
            for suf in (".png", ".svg"):
                old = out_dir / f"{stem}{suf}"
                if old.exists():
                    old.unlink()
        dump_table(f"{tag} model", st_model)
        dump_table(f"{tag} first", mean_sem(g0))
        dump_table(f"{tag} last", mean_sem(g1))
        print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
