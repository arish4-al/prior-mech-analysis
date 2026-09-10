"""Diagnose why incongruent RT fails (no refit).

On regular s101 / s12 and WEIGHTS_REL:

  * split inc RT into correct vs error (dc / dw) vs data
  * timeout and RT<80 ms rates by (signed contrast, conc/inc)
  * ``rt_mode=correct_split`` R² (dc / cc only)
  * |action| at stimOn vs θ (3 sessions; caches have no M)

Plots + JSON live with the models (each run's actprior dir, plus a
comparison dump under openalyx ``models/stageB_hold_s89_inc_rt_diag/``).
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

from _tmp_perf_rt_model_vs_data import (  # noqa: E402
    BASE,
    BEHAVIOR_ACT,
    CONTRASTS_DESC,
    OUT_SUBDIR,
    STIM_SEED0,
    WEIGHTS_REL,
    gof,
    latest_final,
    results_with_subjective_prior,
    run_many_sessions,
    save_current,
    snap_mag,
)
from model_functions import loss_perf_with_data, STEPS_BEFORE_OBS_DURATION_MS  # noqa: E402
from plot_best_fit_results import load_plot_model, make_shared_stimuli  # noqa: E402
from model_functions import run_model  # noqa: E402

STIM = np.array([-1.0, -0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25, 1.0])
N_M_SESSIONS = 3
DIAG_NAME = "inc_rt_diag"
CMP_DIR = BASE / "stageB_hold_s89_inc_rt_diag"

MODELS = (
    (
        "regular_s101",
        BASE / "weights_run_fj_stageB_hold_s89_regular_mask12-13_s101",
        None,
    ),
    (
        "regular_s12",
        BASE / "weights_run_fj_stageB_hold_s89_regular_mask12-13_s12",
        None,
    ),
    ("WEIGHTS_REL", WEIGHTS_REL.parent, WEIGHTS_REL),
)


def _theta(mp):
    th = mp["action_thresholds"]
    conc = th["concordant"]
    disc = th["discordant"]
    key = 0.0 if 0.0 in conc else "0.0" if "0.0" in conc else next(iter(conc))
    return float(conc[key]), float(disc[key])


def _signed_c(results, i):
    mag = snap_mag(abs(float(np.asarray(results["trial_strengths"][i]).reshape(-1)[0])))
    side = float(np.sign(np.asarray(results["trial_sides"][i]).reshape(-1)[0]))
    return mag * side


def _is_cong(results, i):
    side = float(np.sign(np.asarray(results["trial_sides"][i]).reshape(-1)[0]))
    block = float(np.sign(np.asarray(results["block_sides"][i]).reshape(-1)[0]))
    return int(side == block)


def _rt_s(results, i, dt_s):
    return float(results["reaction_time"][i]) * float(dt_s)


def trial_table(results, dt_s):
    n = len(results["choices"])
    rows = []
    for i in range(n):
        ch = float(results["choices"][i])
        rt = _rt_s(results, i, dt_s)
        committed = abs(ch) == 1.0
        in_win = committed and (0.08 <= rt <= 2.0)
        rows.append(
            {
                "signed_c": _signed_c(results, i),
                "cong": _is_cong(results, i),
                "correct": int(results["correct_action_taken"][i]),
                "committed": int(committed),
                "timeout": int(not committed),
                "short": int(committed and rt < 0.08),
                "long": int(committed and rt > 2.0),
                "in_win": int(in_win),
                "rt": rt if in_win else np.nan,
            }
        )
    return rows


def _bin_means(rows, pred, signed=True):
    xs = STIM if signed else np.array(CONTRASTS_DESC, dtype=float)
    out = np.full(len(xs), np.nan)
    n = np.zeros(len(xs), dtype=int)
    for j, x in enumerate(xs):
        vals = [pred(r) for r in rows if np.isclose(r["signed_c"] if signed else abs(r["signed_c"]), x)]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        n[j] = len(vals)
        if vals:
            out[j] = float(np.mean(vals))
    return out, n


def data_rt_curves(behavior):
    res = behavior["reaction_times"]
    tot = behavior["trial_counts"]

    def _mean(num, den):
        num = np.asarray(num, float)
        den = np.asarray(den, float)
        out = np.full(len(STIM), np.nan)
        ok = den > 0
        out[ok] = num[ok] / den[ok]
        return out, den

    return {
        "cc": _mean(res["cc"], tot["cc"]),
        "cw": _mean(res["cw"], tot["cw"]),
        "dc": _mean(res["dc"], tot["dc"]),
        "dw": _mean(res["dw"], tot["dw"]),
        "con": _mean(np.asarray(res["cc"]) + np.asarray(res["cw"]),
                     np.asarray(tot["cc"]) + np.asarray(tot["cw"])),
        "inc": _mean(np.asarray(res["dc"]) + np.asarray(res["dw"]),
                     np.asarray(tot["dc"]) + np.asarray(tot["dw"])),
        "n": {k: np.asarray(tot[k], float) for k in ("cc", "cw", "dc", "dw")},
    }


def model_cell_rt(rows, cong, correct):
    def pred(r):
        if r["cong"] != cong or r["correct"] != correct or not r["in_win"]:
            return None
        return r["rt"]
    return _bin_means(rows, pred)


def model_rate(rows, flag, cong=None):
    def pred(r):
        if cong is not None and r["cong"] != cong:
            return None
        return float(r[flag])
    return _bin_means(rows, pred)


def summarize_cells(rows, data):
    cells = {}
    for name, cong, corr in (("cc", 1, 1), ("cw", 1, 0), ("dc", 0, 1), ("dw", 0, 0)):
        mu, n = model_cell_rt(rows, cong, corr)
        cells[name] = {"rt": mu.tolist(), "n": n.tolist(), "mean_rt": _nanmean(mu)}
    for name, cong in (("con", 1), ("inc", 0)):
        def pred(r, c=cong):
            if r["cong"] != c or not r["in_win"]:
                return None
            return r["rt"]
        mu, n = _bin_means(rows, pred)
        cells[name] = {"rt": mu.tolist(), "n": n.tolist(), "mean_rt": _nanmean(mu)}
    rates = {}
    for flag in ("timeout", "short", "in_win"):
        rates[flag] = {}
        for tag, cong in (("all", None), ("con", 1), ("inc", 0)):
            mu, n = model_rate(rows, flag, cong)
            rates[flag][tag] = {"rate": mu.tolist(), "n": n.tolist(), "mean": _nanmean(mu)}
    # pooled scalars
    inc = [r for r in rows if r["cong"] == 0]
    con = [r for r in rows if r["cong"] == 1]
    inc_win = [r for r in inc if r["in_win"]]
    n_inc_err = sum(1 for r in inc_win if r["correct"] == 0)
    out = {
        "cells": cells,
        "rates": rates,
        "n_trials": len(rows),
        "n_inc": len(inc),
        "n_con": len(con),
        "inc_error_rate_in_win": (n_inc_err / len(inc_win)) if inc_win else np.nan,
        "inc_timeout": _mean_flag(inc, "timeout"),
        "con_timeout": _mean_flag(con, "timeout"),
        "inc_short": _mean_flag(inc, "short"),
        "con_short": _mean_flag(con, "short"),
        "inc_mean_rt_dc": _mean_rt(inc_win, correct=1),
        "inc_mean_rt_dw": _mean_rt(inc_win, correct=0),
        "con_mean_rt_cc": _mean_rt([r for r in con if r["in_win"]], correct=1),
        "con_mean_rt_cw": _mean_rt([r for r in con if r["in_win"]], correct=0),
        "data_mean_rt_dc": _nanmean(data["dc"][0]),
        "data_mean_rt_dw": _nanmean(data["dw"][0]),
        "data_mean_rt_cc": _nanmean(data["cc"][0]),
        "data_inc_error_rate": float(
            np.nansum(data["n"]["dw"]) / max(np.nansum(data["n"]["dc"]) + np.nansum(data["n"]["dw"]), 1.0)
        ),
    }
    return out


def _nanmean(x):
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if len(a) else float("nan")


def _mean_flag(rows, flag):
    if not rows:
        return float("nan")
    return float(np.mean([r[flag] for r in rows]))


def _mean_rt(rows, correct=None):
    vals = [r["rt"] for r in rows if np.isfinite(r["rt"]) and (correct is None or r["correct"] == correct)]
    return float(np.mean(vals)) if vals else float("nan")


def plot_dc_dw(out_dir, rows, data, label):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
    specs = (
        (axes[0], "discordant", 0, "dc", "dw", "gray"),
        (axes[1], "concordant", 1, "cc", "cw", "black"),
    )
    for ax, title, cong, ck, wk, color in specs:
        rtc, nc = model_cell_rt(rows, cong, 1)
        rtw, nw = model_cell_rt(rows, cong, 0)
        ax.plot(STIM, data[ck][0], "-o", color=color, alpha=0.9, label=f"data {ck}")
        ax.plot(STIM, data[wk][0], "-s", color=color, alpha=0.45, label=f"data {wk}")
        ax.plot(STIM, rtc, "--o", color=color, alpha=0.9, label=f"model {ck}")
        ax.plot(STIM, rtw, "--s", color=color, alpha=0.45, label=f"model {wk}")
        ax.set_title(title)
        ax.set_xlabel("signed contrast")
        ax.set_xticks(STIM)
        ax.tick_params(axis="x", labelrotation=45)
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("RT (s)")
    fig.suptitle(f"{label}: correct vs error RT", fontsize=11)
    fig.tight_layout()
    stem = out_dir / f"{DIAG_NAME}_dc_dw"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", transparent=True)
    fig.savefig(stem.with_suffix(".png"), dpi=160, bbox_inches="tight",
                facecolor="white", transparent=False)
    plt.close(fig)


def plot_rates(out_dir, rows, label):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
    for ax, flag, title in (
        (axes[0], "timeout", "timeout (choice=0)"),
        (axes[1], "short", "committed RT < 80 ms"),
    ):
        for cong, name, color in ((1, "con", "black"), (0, "inc", "gray")):
            mu, _ = model_rate(rows, flag, cong)
            ax.plot(STIM, 100 * mu, "-o", color=color, label=name)
        ax.set_title(title)
        ax.set_xlabel("signed contrast")
        ax.set_xticks(STIM)
        ax.tick_params(axis="x", labelrotation=45)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("% of trials")
    fig.suptitle(f"{label}: selection", fontsize=11)
    fig.tight_layout()
    stem = out_dir / f"{DIAG_NAME}_timeout_short"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", transparent=True)
    fig.savefig(stem.with_suffix(".png"), dpi=160, bbox_inches="tight",
                facecolor="white", transparent=False)
    plt.close(fig)


def extract_m_stimon(results, steps_before_obs, theta_c, theta_d):
    M = np.asarray(results["M"], float)
    if M.ndim != 2 or M.shape[1] != 2:
        raise ValueError(f"unexpected M shape {M.shape}")
    lens = [len(np.asarray(ts)) for ts in results["trial_sides"]]
    offsets = np.cumsum([0] + lens[:-1])
    asig = results.get("action_signal")
    rows = []
    for i, (off, L) in enumerate(zip(offsets, lens)):
        idx = int(off + steps_before_obs - 1)  # last prestim step
        if idx < off or idx >= off + L:
            continue
        if asig is not None:
            action = float(np.asarray(asig).reshape(-1)[idx])
        else:
            action = float(np.tanh(M[idx, 0] - M[idx, 1]))
        side = float(np.sign(np.asarray(results["trial_sides"][i]).reshape(-1)[0]))
        block = float(np.sign(np.asarray(results["block_sides"][i]).reshape(-1)[0]))
        cong = int(side == block)
        theta = theta_c if cong else theta_d
        toward_prior = -block * action
        toward_stim = -side * action
        rows.append(
            {
                "signed_c": _signed_c(results, i),
                "cong": cong,
                "correct": int(results["correct_action_taken"][i]),
                "committed": int(abs(float(results["choices"][i])) == 1),
                "action": action,
                "abs_action": abs(action),
                "toward_prior": toward_prior,
                "toward_stim": toward_stim,
                "theta": theta,
                "past_prior": int(toward_prior >= theta),
                "rem_prior": theta - toward_prior,
                "rem_stim": theta - toward_stim,
            }
        )
    return rows


def plot_m_stimon(out_dir, mrows, theta_c, theta_d, label):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
    for ax, cong, name, theta, color in (
        (axes[0], 0, "incongruent", theta_d, "gray"),
        (axes[1], 1, "concordant", theta_c, "black"),
    ):
        vals = [r["toward_prior"] for r in mrows if r["cong"] == cong]
        ax.hist(vals, bins=40, color=color, alpha=0.7, density=True)
        ax.axvline(theta, color="crimson", ls="--", label=f"θ={theta:.3f}")
        ax.axvline(0.0, color="0.5", ls=":", lw=0.8)
        past = np.mean([r["past_prior"] for r in mrows if r["cong"] == cong]) if vals else np.nan
        ax.set_title(f"{name}  P(past prior bound)={past:.3f}")
        ax.set_xlabel("action toward prior at stimOn")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("density")
    fig.suptitle(f"{label}: start vs bound", fontsize=11)
    fig.tight_layout()
    stem = out_dir / f"{DIAG_NAME}_m_stimon"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", transparent=True)
    fig.savefig(stem.with_suffix(".png"), dpi=160, bbox_inches="tight",
                facecolor="white", transparent=False)
    plt.close(fig)


def m_summary(mrows):
    out = {}
    for tag, cong in (("inc", 0), ("con", 1)):
        sub = [r for r in mrows if r["cong"] == cong]
        out[tag] = {
            "n": len(sub),
            "mean_toward_prior": _nanmean([r["toward_prior"] for r in sub]),
            "mean_abs_action": _nanmean([r["abs_action"] for r in sub]),
            "frac_past_prior_bound": _nanmean([r["past_prior"] for r in sub]),
            "mean_rem_prior": _nanmean([r["rem_prior"] for r in sub]),
            "mean_rem_stim": _nanmean([r["rem_stim"] for r in sub]),
            "theta": float(sub[0]["theta"]) if sub else float("nan"),
        }
    return out


def run_m_sessions(mp, n_sessions, steps_before_obs):
    runs = []
    for i in range(n_sessions):
        (
            stimuli, trial_strengths, trial_sides, block_sides,
            sbo, bps_out,
        ) = make_shared_stimuli(mp, bps=20, seed=STIM_SEED0 + i)
        results = run_model(
            "data", stimuli, trial_strengths, trial_sides, block_sides, bps_out,
            steps_before_obs=sbo, verbose=False, backend="numba", **mp,
        )
        print(f"  M-session {i + 1}/{n_sessions}  trials={len(results['choices'])}")
        runs.append(results)
    # concat trial-level only (keep M aligned)
    from model_functions import combine_run_results
    return combine_run_results(runs), int(steps_before_obs)


def load_cached_or_sim(label, run, jp, mp):
    out_dir = run / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "combined_results.npy"
    if cache.is_file():
        loaded = np.load(cache, allow_pickle=True).item()
        if "sub_prior" in loaded:
            print(f"loaded {cache}  n={len(loaded['choices'])}")
            return loaded, out_dir
        print(f"{cache.name} missing sub_prior; re-sim")
    print(f"sim 10×20 for {label}")
    results, _ = run_many_sessions(mp, 10, 20, STIM_SEED0)
    slim = {
        k: results[k]
        for k in (
            "trial_strengths", "trial_sides", "block_sides",
            "correct_action_taken", "reaction_time", "choices", "sub_prior",
        )
    }
    np.save(cache, slim, allow_pickle=True)
    return slim, out_dir


def main():
    if not BEHAVIOR_ACT.is_file():
        raise FileNotFoundError(BEHAVIOR_ACT)
    behavior = np.load(BEHAVIOR_ACT, allow_pickle=True).item()
    data = data_rt_curves(behavior)
    CMP_DIR.mkdir(parents=True, exist_ok=True)

    dump = []
    print(
        f"{'model':14} {'θc/θd':>11} {'incR²all':>8} {'incR²corr':>9} "
        f"{'inc_to':>7} {'inc_<80':>7} {'inc_err':>7} "
        f"{'RTdc':>6} {'RTdw':>6} {'pastθ':>6}",
        flush=True,
    )
    for label, run, jp in MODELS:
        jp = jp or latest_final(run)
        mp, meta = load_plot_model(jp)
        dt_ms = float(mp["dt"])
        dt_s = dt_ms / 1000.0
        theta_c, theta_d = _theta(mp)
        sbo = int(STEPS_BEFORE_OBS_DURATION_MS / dt_ms)

        raw, out_dir = load_cached_or_sim(label, run, jp, mp)
        results = results_with_subjective_prior(raw)
        rows = trial_table(results, dt_s)

        sse_all = loss_perf_with_data(
            results, behavior, mp, metric="rt", dt=dt_s, do_plot=False,
            rt_mode="split_all",
        )
        sse_corr = loss_perf_with_data(
            results, behavior, mp, metric="rt", dt=dt_s, do_plot=True,
            rt_mode="correct_split",
        )
        save_current(out_dir / f"{DIAG_NAME}_correct_split",
                     f"{label} RT correct-only (act prior)")

        plot_dc_dw(out_dir, rows, data, label)
        plot_rates(out_dir, rows, label)
        cells = summarize_cells(rows, data)

        print(f"  sim {N_M_SESSIONS} sessions for |M| at stimOn …", flush=True)
        mres, sbo_used = run_m_sessions(mp, N_M_SESSIONS, sbo)
        mres = results_with_subjective_prior(mres)
        mrows = extract_m_stimon(mres, sbo_used, theta_c, theta_d)
        plot_m_stimon(out_dir, mrows, theta_c, theta_d, label)
        msum = m_summary(mrows)

        rec = {
            "label": label,
            "json": str(jp),
            "out_dir": str(out_dir),
            "recorded_loss": meta.get("loss"),
            "g_i": float(mp["g_i"]),
            "g_m": float(mp.get("g_m", np.nan)),
            "W_mi": float(mp.get("W_mi", np.nan)),
            "theta_c": theta_c,
            "theta_d": theta_d,
            "rt_split_r2_inc": gof(sse_all, "incongruent"),
            "rt_split_r2_con": gof(sse_all, "congruent"),
            "rt_correct_r2_inc": gof(sse_corr, "incongruent"),
            "rt_correct_r2_con": gof(sse_corr, "congruent"),
            "cells": cells,
            "m_stimon": msum,
            "n_m_sessions": N_M_SESSIONS,
        }
        dump.append(rec)
        print(
            f"{label:<14} {theta_c:.2f}/{theta_d:.2f}  "
            f"{rec['rt_split_r2_inc']:8.3f} {rec['rt_correct_r2_inc']:9.3f} "
            f"{cells['inc_timeout']:7.3f} {cells['inc_short']:7.3f} "
            f"{cells['inc_error_rate_in_win']:7.3f} "
            f"{cells['inc_mean_rt_dc']:6.3f} {cells['inc_mean_rt_dw']:6.3f} "
            f"{msum['inc']['frac_past_prior_bound']:6.3f}",
            flush=True,
        )

    out_json = CMP_DIR / "inc_rt_diag.json"
    out_json.write_text(json.dumps(dump, indent=2, default=str))
    print(f"\nwrote {out_json}")
    print(f"per-run plots: {DIAG_NAME}_*.png in each actprior dir")


if __name__ == "__main__":
    main()
