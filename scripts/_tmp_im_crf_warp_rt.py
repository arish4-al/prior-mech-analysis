"""Artificial test: match per-contrast M |Δ| to data, recompute RT.

Commit is first-passage of tanh(M0−M1) to θ_c / θ_d (perceived S×P).
I is not the bound variable. Two post-hoc warps of M_diff, same sims:

  scale — multiply M0−M1 by data/model late-25 M stim residual
  add   — add sign(M0−M1) × (data − model) residual

Then score act-prior RT (10×20, seed 12345) vs the unwarped baseline.
Plots go in the s101 run dir.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from plot_best_fit_results import load_plot_model, make_shared_stimuli  # noqa: E402
from _tmp_perf_rt_model_vs_data import (  # noqa: E402
    BEHAVIOR_ACT,
    BPS,
    N_SESSIONS,
    STIM_SEED0,
    gof,
    results_with_subjective_prior,
    snap_mag,
)
from model_functions import combine_run_results, loss_perf_with_data, run_model  # noqa: E402
import model_functions as mf  # noqa: E402

CONTRASTS = (1.0, 0.25, 0.125, 0.0625, 0.0)
JSON = Path.home() / (
    "Downloads/ONE/openalyx.internationalbrainlab.org/models/"
    "weights_run_fj_stageB_hold_s89_regular_mask12-13_s101/"
    "weights_final_loss1p131_20260812-232651.json"
)
DATA = Path.home() / (
    "Downloads/ONE/alyx.internationalbrainlab.org/manifold/"
    "mean_data_im_from_cache/mean_data_results_by_contrast.npy"
)
OUT = JSON.parent / "im_crf_warp_rt"
T_STIM = 72
LATE = 12


def data_m_stim_resid():
    payload = np.load(DATA, allow_pickle=True).flat[0]
    out = {}
    for c in CONTRASTS:
        ys = []
        for ch in ("L", "R"):
            raw = np.asarray(payload["M"]["mean_traj"][c]["stim"][ch], float)
            t0 = float(np.asarray(payload["M"]["mean_traj"][c]["stim"][ch], float)[0])
            ys.append(float(np.mean(raw[-LATE:] - t0)))
        out[c] = float(np.mean(ys))
    return out


def trial_c(results, i):
    return snap_mag(np.asarray(results["trial_strengths"][i]).reshape(-1)[0])


def trial_slices(results, sbo):
    n = len(results["choices"])
    lens = [len(results["trial_sides"][i]) for i in range(n)]
    offsets = np.cumsum([0] + lens[:-1])
    return n, lens, offsets


def model_m_stim_resid(results, sbo):
    """Mean |M0−M1| in the last 25 ms of the first 72 post-stim steps."""
    M = np.asarray(results["M"], float)
    n, lens, offsets = trial_slices(results, sbo)
    acc = {c: [] for c in CONTRASTS}
    for i in range(n):
        post = lens[i] - sbo
        if post < T_STIM:
            continue
        start = offsets[i] + sbo
        seg = M[start:start + T_STIM]
        acc[trial_c(results, i)].append(float(np.mean(np.abs(seg[-LATE:, 0] - seg[-LATE:, 1]))))
    return {c: float(np.mean(v)) if v else np.nan for c, v in acc.items()}


def replay_rt(results, sbo, mp, warp, gain, shift):
    """First-passage of tanh(warped M0−M1) with the sim's S×P θ rule."""
    M = np.asarray(results["M"], float)
    S = np.asarray(results["S"], float)
    P = np.asarray(results["P"], float)
    n, lens, offsets = trial_slices(results, sbo)
    th = mp["action_thresholds"]
    rt = np.zeros(n, dtype=float)
    choices = np.asarray(results["choices"], float).copy()
    correct = np.asarray(results["correct_action_taken"], float).copy()
    n_hit = 0
    for i in range(n):
        c = trial_c(results, i)
        th_c = float(th["concordant"][c])
        th_d = float(th["discordant"][c])
        sl = slice(offsets[i], offsets[i] + lens[i])
        md = M[sl, 0] - M[sl, 1]
        if warp == "scale":
            md = gain[c] * md
        elif warp == "add":
            md = md + np.sign(md) * shift[c]
        action = np.tanh(md)
        hit = None
        for k in range(sbo + 1, lens[i]):
            dS = S[sl][k, 0] - S[sl][k, 1]
            dP = P[sl][k, 0] - P[sl][k, 1]
            theta = th_c if (dS * dP >= 0.0) else th_d
            if abs(action[k]) >= (theta + 1e-6):
                hit = k + 1 - sbo
                break
        if hit is None:
            rt[i] = lens[i] - sbo
            choices[i] = 0.0
            correct[i] = 0.0
        else:
            rt[i] = hit
            n_hit += 1
    out = dict(results)
    out["reaction_time"] = rt
    out["choices"] = choices
    out["correct_action_taken"] = correct
    return out, n_hit


def slim(res):
    return {k: res[k] for k in (
        "trial_strengths", "trial_sides", "block_sides",
        "correct_action_taken", "reaction_time", "choices", "sub_prior",
    )}


def score(res, behavior, mp, dt_s, tag, out_dir):
    res = results_with_subjective_prior(res)
    sse_rt = loss_perf_with_data(
        res, behavior, mp, metric="rt", dt=dt_s, do_plot=True,
        save_dir=None, log_xaxis=True, rt_mode="combined_all",
    )
    fig = plt.gcf()
    fig.savefig(out_dir / f"rt_combined_{tag}.png", dpi=160, bbox_inches="tight",
                facecolor="white", transparent=False)
    plt.close(fig)
    sse_sp = loss_perf_with_data(
        res, behavior, mp, metric="rt", dt=dt_s, do_plot=True,
        save_dir=None, log_xaxis=True, rt_mode="split_all",
    )
    fig = plt.gcf()
    fig.savefig(out_dir / f"rt_split_{tag}.png", dpi=160, bbox_inches="tight",
                facecolor="white", transparent=False)
    plt.close(fig)
    sse_pf = loss_perf_with_data(
        res, behavior, mp, metric="correct", dt=1.0, do_plot=False,
        log_xaxis=True,
    )
    return {
        "tag": tag,
        "perf_r2": gof(sse_pf, "total"),
        "rt_r2": gof(sse_rt, "total"),
        "rt_split_r2": gof(sse_sp, "total"),
        "rt_split_r2_con": gof(sse_sp, "congruent"),
        "rt_split_r2_inc": gof(sse_sp, "incongruent"),
        "gof_rt": sse_rt.get("gof"),
        "gof_split": sse_sp.get("gof"),
    }


def plot_split_overlay(rows_curves, behavior, dt_s, out_dir):
    """Data vs baseline / scale / add incongruent + concordant RT."""
    stim = np.array([-1, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 1])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
    styles = {
        "baseline": dict(ls="-", color="0.2", lw=1.8),
        "replay": dict(ls=":", color="0.45", lw=1.4),
        "scale": dict(ls="--", color="C3", lw=1.8),
        "add": dict(ls="-.", color="C0", lw=1.8),
    }
    for ax, key, title in (
        (axes[0], "congruent", "concordant"),
        (axes[1], "incongruent", "incongruent"),
    ):
        if key == "congruent":
            n1 = np.asarray(behavior["trial_counts"]["cc"], float)
            n2 = np.asarray(behavior["trial_counts"]["cw"], float)
            r1 = np.asarray(behavior["reaction_times"]["cc"], float)
            r2 = np.asarray(behavior["reaction_times"]["cw"], float)
        else:
            n1 = np.asarray(behavior["trial_counts"]["dc"], float)
            n2 = np.asarray(behavior["trial_counts"]["dw"], float)
            r1 = np.asarray(behavior["reaction_times"]["dc"], float)
            r2 = np.asarray(behavior["reaction_times"]["dw"], float)
        den = n1 + n2
        # behavior reaction_times are sums; same as loss_perf split_all
        data = np.divide(r1 + r2, den, out=np.full_like(den, np.nan), where=den > 0)
        ax.plot(stim, data, "o", color="0.3", ms=4, label="data")
        for tag, (xs, ys) in rows_curves[key].items():
            ax.plot(xs, ys, label=tag, **styles[tag])
        ax.set_xscale("symlog", linthresh=0.05)
        ax.set_title(title)
        ax.set_xlabel("signed contrast")
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("RT (s)")
    fig.tight_layout()
    for ext in (".svg", ".png"):
        fig.savefig(out_dir / f"rt_split_warp_overlay{ext}", bbox_inches="tight",
                    facecolor="white", transparent=False)
    plt.close(fig)


def model_split_curves(res, dt_s):
    """Mean RT vs signed contrast, split_all, same mask as loss_perf."""
    res = results_with_subjective_prior(res)
    ts = [float(a[0]) for a in res["trial_strengths"]]
    side = [float(a[0]) for a in res["trial_sides"]]
    block = [float(a[0]) for a in res["block_sides"]]
    rt = np.asarray(res["reaction_time"], float)
    ch = np.asarray(res["choices"], float)
    signed = np.asarray(ts) * np.asarray(side)
    cong = np.asarray(side) == np.asarray(block)
    out = {}
    for key, mask_c in (("congruent", cong), ("incongruent", ~cong)):
        xs, ys = [], []
        for s in (-1, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 1):
            m = (np.abs(ch) == 1) & mask_c & np.isclose(signed, s)
            if int(m.sum()) < 1:
                continue
            rt_s = rt[m] * dt_s
            keep = (rt_s >= 0.08) & (rt_s <= 2.0)
            if int(keep.sum()) < 1:
                continue
            xs.append(float(s))
            ys.append(float(np.mean(rt_s[keep])))
        out[key] = (xs, ys)
    return out


def main():
    mp, meta = load_plot_model(JSON)
    dt_ms = float(mp["dt"])
    dt_s = dt_ms / 1000.0
    sbo = int(mf.STEPS_BEFORE_OBS_DURATION_MS / mp["dt"])
    behavior = np.load(BEHAVIOR_ACT, allow_pickle=True).item()
    data_resid = data_m_stim_resid()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"s101 rec={meta.get('loss')}  θc={mp['action_thresholds']['concordant'][0.0]:.3f}  "
          f"θd={mp['action_thresholds']['discordant'][0.0]:.3f}  "
          f"{N_SESSIONS}×{BPS} seed0={STIM_SEED0}", flush=True)

    raw_runs = []
    for i in range(N_SESSIONS):
        stim = make_shared_stimuli(mp, bps=BPS, seed=STIM_SEED0 + i)
        stimuli, trial_strengths, trial_sides, block_sides, sbo_i, bps = stim
        assert sbo_i == sbo
        res = run_model(
            "data", stimuli, trial_strengths, trial_sides, block_sides, bps,
            steps_before_obs=sbo, verbose=False, backend="numba", **mp,
        )
        print(f"  session {i + 1}/{N_SESSIONS}  n={len(res['choices'])}", flush=True)
        raw_runs.append(res)

    combined = combine_run_results(raw_runs)
    model_resid = model_m_stim_resid(combined, sbo)
    gain, shift = {}, {}
    print("\nM stim late-25 |Δ|  (data residual vs model):")
    print(f"{'c':>8} {'data':>8} {'model':>8} {'scale':>8} {'add':>8}")
    for c in CONTRASTS:
        d, m = data_resid[c], model_resid[c]
        gain[c] = float(d / m) if (np.isfinite(m) and m > 1e-6) else 1.0
        shift[c] = float(d - m) if np.isfinite(m) else 0.0
        print(f"{c:8.4f} {d:8.3f} {m:8.3f} {gain[c]:8.3f} {shift[c]:+8.3f}")

    # replay unwarped M to check first-passage recovery
    replay_runs = []
    scale_runs = []
    add_runs = []
    for res in raw_runs:
        r0, n0 = replay_rt(res, sbo, mp, "none", gain, shift)
        rs, ns = replay_rt(res, sbo, mp, "scale", gain, shift)
        ra, na = replay_rt(res, sbo, mp, "add", gain, shift)
        replay_runs.append(slim(r0))
        scale_runs.append(slim(rs))
        add_runs.append(slim(ra))
    base = slim(combined)
    replay = combine_run_results(replay_runs)
    scaled = combine_run_results(scale_runs)
    added = combine_run_results(add_runs)

    orig_rt = np.asarray(base["reaction_time"], float)
    rep_rt = np.asarray(replay["reaction_time"], float)
    ok = np.asarray(base["choices"], float) != 0
    dt_steps = np.abs(orig_rt[ok] - rep_rt[ok])
    print(f"\nreplay vs original RT: median |Δ|={np.median(dt_steps):.2f} steps  "
          f"p(exact)={np.mean(dt_steps < 0.5):.3f}  n={int(ok.sum())}")

    rows = []
    curves = {"congruent": {}, "incongruent": {}}
    for tag, res in (
        ("baseline", base),
        ("replay", replay),
        ("scale", scaled),
        ("add", added),
    ):
        row = score(res, behavior, mp, dt_s, tag, OUT)
        rows.append(row)
        split = model_split_curves(res, dt_s)
        for k in curves:
            curves[k][tag] = split[k]
        print(
            f"{tag:<10} perf={row['perf_r2']:.3f}  RTcomb={row['rt_r2']:.3f}  "
            f"split={row['rt_split_r2']:.3f}  "
            f"(con {row['rt_split_r2_con']:.3f} / inc {row['rt_split_r2_inc']:.3f})",
            flush=True,
        )

    plot_split_overlay(curves, behavior, dt_s, OUT)
    payload = {
        "json": str(JSON),
        "n_sessions": N_SESSIONS,
        "bps": BPS,
        "stim_seed0": STIM_SEED0,
        "data_m_stim_resid": data_resid,
        "model_m_stim_resid": model_resid,
        "gain": gain,
        "shift": shift,
        "replay_median_abs_steps": float(np.median(dt_steps)),
        "rows": rows,
        "out": str(OUT),
    }
    (OUT / "meta.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
