"""Overlay per-contrast I/M in traj-loss units + per-contrast nSSE.

Data: alyx manifold/mean_data_im_from_cache (150 ms, choice×contrast).
Model: regular s101, shared-stim bps=20 seed 12345 (same as fair eval).

Comparable traces: data RMS minus first stim bin vs model channel |Δ|
(same transform as ``loss_plot_diff_by_condition_with_data``). Stim nSSE
skips the first 15 bins; choice nSSE is the full window (M × m_pre_weight).

Plots go in the s101 run dir (plots-live-with-data).
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
from model_functions import run_model  # noqa: E402
import model_functions as mf  # noqa: E402

CONTRASTS = (1.0, 0.25, 0.125, 0.0625, 0.0)
DATA = Path.home() / (
    "Downloads/ONE/alyx.internationalbrainlab.org/manifold/"
    "mean_data_im_from_cache/mean_data_results_by_contrast.npy"
)
JSON = Path.home() / (
    "Downloads/ONE/openalyx.internationalbrainlab.org/models/"
    "weights_run_fj_stageB_hold_s89_regular_mask12-13_s101/"
    "weights_final_loss1p131_20260812-232651.json"
)
OUT = JSON.parent / "im_percontrast_vs_data"
T = 72
SKIP_STIM = 15
EPS = 1e-12
CH_L = 1
CH_R = -1
CHOICES = (("L", CH_L), ("R", CH_R))


def snap_c(c):
    c = float(abs(c))
    return min(CONTRASTS, key=lambda x: abs(x - c))


def trial_c(results, i):
    return snap_c(np.asarray(results["trial_strengths"][i]).reshape(-1)[0])


def _min_steps():
    return int(getattr(mf, "_min_trial_steps")())


def extract_mean(results, vn, steps_before_obs, contrast, mode, T=T, ch=CH_L):
    """Mean (2, T) for one choice × contrast. Same windows as mean_by_condition."""
    choices = results["choices"]
    sides = results["trial_sides"]
    rt = results.get("reaction_time")
    n = len(choices)
    lens = [len(sides[i]) for i in range(n)]
    offsets = np.cumsum([0] + lens[:-1])
    hard = steps_before_obs + _min_steps()
    var = np.asarray(results[vn], float)
    segs = []
    for i in range(n):
        if int(choices[i]) != ch:
            continue
        if contrast is not None and trial_c(results, i) != contrast:
            continue
        m_i = lens[i]
        if m_i < hard:
            continue
        if mode == "pre":
            if rt is None:
                continue
            act = steps_before_obs + int(rt[i])
            if act < T or act > m_i:
                continue
            start = offsets[i] + act - T
            seg = var[start:start + T, :]
        else:
            post_avail = max(0, m_i - steps_before_obs)
            take = min(T, post_avail)
            parts = []
            if take > 0:
                start = offsets[i] + steps_before_obs
                parts.append(var[start:start + take, :])
            if take < T:
                if i + 1 >= n or lens[i + 1] < hard:
                    continue
                need = T - take
                pre_next = min(steps_before_obs, lens[i + 1])
                if pre_next < need:
                    continue
                parts.append(var[offsets[i + 1]:offsets[i + 1] + need, :])
            seg = np.vstack(parts) if parts else None
            if seg is None or seg.shape[0] != T:
                continue
        if seg is not None and seg.shape[0] == T:
            segs.append(seg)
    if len(segs) < 10:
        return None, len(segs)
    mean = np.mean(np.stack(segs, 0), 0).T  # (2, T)
    return mean, len(segs)


def model_amp(arr, ch):
    if arr is None:
        return None
    d = (arr[1] - arr[0]) if ch == 1 else (arr[0] - arr[1])
    return np.asarray(d, float)


def data_residual(payload, vn, c, win, ch_name):
    """RMS minus first stim bin (traj-loss baseline)."""
    raw = np.asarray(payload[vn]["mean_traj"][c][win][ch_name], float)
    t0 = float(np.asarray(payload[vn]["mean_traj"][c]["stim"][ch_name], float)[0])
    return raw - t0, t0


def nsse(model, data, win, w=1.0):
    """Energy-normalized SSE. Stim skips first 15 bins; choice uses the full window."""
    if model is None or data is None:
        return {"nsse": np.nan, "raw": np.nan, "energy": np.nan, "nbin": 0}
    n = min(len(model), len(data))
    m = np.asarray(model[:n] if win == "stim" else model[-n:], float)
    d = np.asarray(data[:n] if win == "stim" else data[-n:], float)
    if win == "stim" and n > SKIP_STIM:
        m, d = m[SKIP_STIM:], d[SKIP_STIM:]
    if m.size == 0 or np.any(~np.isfinite(m)) or np.any(~np.isfinite(d)):
        return {"nsse": np.nan, "raw": np.nan, "energy": np.nan, "nbin": 0}
    energy = float(np.sum(d ** 2))
    raw = float(np.sum((m - d) ** 2))
    return {
        "nsse": w * raw / (energy + EPS),
        "raw": raw,
        "energy": energy,
        "nbin": int(m.size),
    }


def main():
    payload = np.load(DATA, allow_pickle=True).flat[0]
    mp, meta = load_plot_model(JSON)
    m_pre_w = float(mf.m_pre_weight_of(mp))
    stim = make_shared_stimuli(mp, bps=20, seed=12345)
    stimuli, trial_strengths, trial_sides, block_sides, sbo, bps = stim
    print(f"model {JSON.parent.name}  recorded={meta.get('loss')}  "
          f"m_pre_weight={m_pre_w}", flush=True)
    results = run_model(
        "data", stimuli, trial_strengths, trial_sides, block_sides, bps,
        steps_before_obs=sbo, verbose=False, backend="numba", **mp,
    )
    OUT.mkdir(parents=True, exist_ok=True)

    model_n = {}
    model_curves = {}
    data_curves = {}
    losses = {}
    for vn in ("I", "M"):
        for win, mode in (("stim", "post"), ("choice", "pre")):
            w = m_pre_w if (vn == "M" and win == "choice") else 1.0
            for ch_name, ch in CHOICES:
                for c in CONTRASTS:
                    arr, n = extract_mean(results, vn, sbo, c, mode, ch=ch)
                    m = model_amp(arr, ch)
                    d, t0 = data_residual(payload, vn, c, win, ch_name)
                    key = (vn, win, ch_name, c)
                    model_curves[key] = m
                    data_curves[key] = d
                    model_n[key] = n
                    losses[key] = {**nsse(m, d, win, w=w), "t0": t0, "weight": w}
                    print(f"  {vn} {win} {ch_name} c={c} n={n}  "
                          f"nSSE={losses[key]['nsse']:.4f}", flush=True)

    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(CONTRASTS)))

    # traces: same axis, choice L (same as the earlier overlay)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), sharey=False)
    for row, vn in enumerate(("I", "M")):
        for col, win in enumerate(("stim", "choice")):
            ax = axes[row, col]
            t = (np.linspace(0.0, 150.0, T) if win == "stim"
                 else np.linspace(-150.0, 0.0, T))
            for i, c in enumerate(CONTRASTS):
                d = data_curves[(vn, win, "L", c)]
                m = model_curves[(vn, win, "L", c)]
                td = (np.linspace(t[0], t[-1], d.size) if d is not None else t)
                ax.plot(td, d, color=cmap[i], lw=1.5, label=f"c={c}")
                if m is not None:
                    ax.plot(t[:len(m)], m, color=cmap[i], lw=1.5, ls="--")
            nsum = sum(losses[(vn, win, "L", c)]["nsse"] for c in CONTRASTS)
            ax.axhline(0.0, color="0.75", lw=0.7)
            ax.set_title(f"{vn} {win}  L  solid=data−t0  dashed=model Δ")
            ax.set_xlabel("time (ms)")
            if col == 0:
                ax.set_ylabel("amplitude")
            ax.legend(frameon=False, fontsize=7, ncol=2,
                      title=f"Σ nSSE={nsum:.3f}")
    fig.tight_layout()
    for ext in (".svg", ".png"):
        fig.savefig(OUT / f"im_percontrast_overlay{ext}", bbox_inches="tight",
                    facecolor="white", transparent=False)
    plt.close(fig)

    # late-25 residual CRF, same units
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2), sharex=True)
    late = {}
    print("\nlast 25 ms residual (choice L, data−t0 vs model):")
    print(f"{'':16} {'c=1':>7} {'c=.25':>7} {'c=.125':>7} {'c=.0625':>7} {'c=0':>7}  1/.0625")
    for row, vn in enumerate(("I", "M")):
        for col, win in enumerate(("stim", "choice")):
            ax = axes[row, col]
            ys_d, ys_m = [], []
            for c in CONTRASTS:
                d = data_curves[(vn, win, "L", c)]
                m = model_curves[(vn, win, "L", c)]
                sl = slice(-12, None)
                ys_d.append(float(np.mean(d[sl])))
                ys_m.append(float(np.mean(m[sl])) if m is not None else np.nan)
            late[(vn, win)] = {"data": ys_d, "model": ys_m}
            ax.plot(CONTRASTS, ys_d, "-o", color="0.25", label="data − t0")
            ax.plot(CONTRASTS, ys_m, "--s", color="C3", label="model Δ")
            ax.axhline(0.0, color="0.75", lw=0.7)
            ax.set_xscale("symlog", linthresh=0.05)
            ax.set_title(f"{vn} {win}  last ~25 ms")
            ax.legend(frameon=False, fontsize=8)
            if row == 1:
                ax.set_xlabel("|contrast|")
            if col == 0:
                ax.set_ylabel("amplitude")
            r = ys_d[0] / ys_d[3] if ys_d[3] else np.nan
            rm = ys_m[0] / ys_m[3] if ys_m[3] else np.nan
            print(f"data {vn} {win:<6} "
                  + " ".join(f"{y:7.3f}" for y in ys_d)
                  + f"  {r:5.3f}")
            print(f"mod  {vn} {win:<6} "
                  + " ".join(f"{y:7.3f}" for y in ys_m)
                  + f"  {rm:5.3f}")
    fig.tight_layout()
    for ext in (".svg", ".png"):
        fig.savefig(OUT / f"im_percontrast_late25{ext}", bbox_inches="tight",
                    facecolor="white", transparent=False)
    plt.close(fig)

    # nSSE bars: L, R, L+R
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.4), sharex=True)
    print("\nnSSE (traj-loss units; stim skip 15; M choice × "
          f"m_pre_weight={m_pre_w}):")
    hdr = f"{'term':<16} " + " ".join(f"{c:>8}" for c in CONTRASTS) + f" {'sum':>8}"
    print(hdr)
    loss_table = {}
    x = np.arange(len(CONTRASTS))
    width = 0.36
    for row, vn in enumerate(("I", "M")):
        for col, win in enumerate(("stim", "choice")):
            ax = axes[row, col]
            yL = [losses[(vn, win, "L", c)]["nsse"] for c in CONTRASTS]
            yR = [losses[(vn, win, "R", c)]["nsse"] for c in CONTRASTS]
            yB = [a + b for a, b in zip(yL, yR)]
            loss_table[f"{vn}_{win}_L"] = yL
            loss_table[f"{vn}_{win}_R"] = yR
            loss_table[f"{vn}_{win}_LR"] = yB
            ax.bar(x - width / 2, yL, width, color="0.35", label="choice L")
            ax.bar(x + width / 2, yR, width, color="0.7", label="choice R")
            ax.set_title(f"{vn} {win} nSSE")
            ax.legend(frameon=False, fontsize=8)
            if row == 1:
                ax.set_xticks(x, [str(c) for c in CONTRASTS])
                ax.set_xlabel("|contrast|")
            if col == 0:
                ax.set_ylabel("nSSE")
            print(f"{vn} {win} L      "
                  + " ".join(f"{v:8.4f}" for v in yL)
                  + f" {sum(yL):8.4f}")
            print(f"{vn} {win} R      "
                  + " ".join(f"{v:8.4f}" for v in yR)
                  + f" {sum(yR):8.4f}")
            print(f"{vn} {win} L+R    "
                  + " ".join(f"{v:8.4f}" for v in yB)
                  + f" {sum(yB):8.4f}")

    # per-contrast total (I+M, stim+choice, L+R)
    tot = []
    tot_L = []
    for i, c in enumerate(CONTRASTS):
        s = 0.0
        sL = 0.0
        for vn in ("I", "M"):
            for win in ("stim", "choice"):
                s += losses[(vn, win, "L", c)]["nsse"]
                s += losses[(vn, win, "R", c)]["nsse"]
                sL += losses[(vn, win, "L", c)]["nsse"]
        tot.append(s)
        tot_L.append(sL)
    loss_table["I+M_stim+choice_LR"] = tot
    loss_table["I+M_stim+choice_L"] = tot_L
    print(f"{'I+M all L':<16} "
          + " ".join(f"{v:8.4f}" for v in tot_L)
          + f" {sum(tot_L):8.4f}")
    print(f"{'I+M all L+R':<16} "
          + " ".join(f"{v:8.4f}" for v in tot)
          + f" {sum(tot):8.4f}")
    fig.tight_layout()
    for ext in (".svg", ".png"):
        fig.savefig(OUT / f"im_percontrast_nsse{ext}", bbox_inches="tight",
                    facecolor="white", transparent=False)
    plt.close(fig)

    slim = {
        "json": str(JSON),
        "data": str(DATA),
        "bps": 20,
        "stim_seed": 12345,
        "compare": "data RMS − first stim bin vs model |Δ|; "
                   "stim nSSE skips 15 bins; choice full window",
        "m_pre_weight": m_pre_w,
        "skip_stim_bins": SKIP_STIM,
        "n_model": {f"{k[0]}_{k[1]}_{k[2]}_{k[3]}": v for k, v in model_n.items()},
        "late25_residual_L": {f"{a}_{b}": late[(a, b)] for a, b in late},
        "nsse": loss_table,
        "nsse_detail": {
            f"{k[0]}_{k[1]}_{k[2]}_{k[3]}": v for k, v in losses.items()
        },
        "out": str(OUT),
    }
    (OUT / "meta.json").write_text(json.dumps(slim, indent=2, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
