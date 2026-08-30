"""Why incongruent RT degraded in Stage B vs WEIGHTS_REL.

Uses cached 10×20-block sims from _tmp_perf_rt_model_vs_data.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from plot_best_fit_results import load_plot_model  # noqa: E402
from model_functions import tau_delta_ms  # noqa: E402

BASE = Path.home() / "Downloads/ONE/openalyx.internationalbrainlab.org/models"
BEHAVIOR = Path.home() / (
    "Downloads/ONE/openalyx.internationalbrainlab.org/manifold/res/behavior.npy"
)
SIGNED = np.array([-1.0, -0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25, 1.0])
DT_S = 0.002
MAX_OBS_S = 2.0  # MAX_OBS_DURATION_MS / 1000

MODELS = [
    ("WEIGHTS_REL", BASE / "weights_run_20251125_182058"
     / "weights_2stagelocalrefine_loss0p4044_20251125-195255.json"),
    ("s101", None),
    ("s12", None),
    ("s34", None),
    ("s333", None),
]


def latest_stage_b(seed: int) -> Path:
    d = BASE / f"weights_run_fj_stageB_hold_s89_regular_mask12-13_s{seed}"
    finals = sorted(d.glob("weights_final_*.json"))
    return finals[-1]


def json_for(label: str, jp):
    if jp is not None:
        return jp
    return latest_stage_b(int(label[1:]))


def snap(s):
    return float(SIGNED[np.argmin(np.abs(SIGNED - float(s)))])


def trial_table(cache: Path):
    r = np.load(cache, allow_pickle=True).item()
    mag = np.array([ts[0] for ts in r["trial_strengths"]], float)
    side = np.array([ts[0] for ts in r["trial_sides"]], float)
    block = np.array([bs[0] for bs in r["block_sides"]], float)
    ch = np.asarray(r["choices"], float)
    rt = np.asarray(r["reaction_time"], float) * DT_S
    signed = np.array([snap(m * si) for m, si in zip(mag, side)])
    conc = side == block
    timeout = ch == 0
    return signed, conc, timeout, rt, ch


def mean_at(signed, mask, rt):
    out = np.full(len(SIGNED), np.nan)
    n = np.zeros(len(SIGNED), dtype=int)
    for i, s in enumerate(SIGNED):
        v = rt[(signed == s) & mask]
        n[i] = v.size
        if v.size:
            out[i] = float(np.mean(v))
    return out, n


def main():
    beh = np.load(BEHAVIOR, allow_pickle=True).flat[0]
    res, tot = beh["reaction_times"], beh["trial_counts"]
    data_con = (np.asarray(res["cc"]) + np.asarray(res["cw"])) / (
        np.asarray(tot["cc"]) + np.asarray(tot["cw"])
    )
    data_inc = (np.asarray(res["dc"]) + np.asarray(res["dw"])) / (
        np.asarray(tot["dc"]) + np.asarray(tot["dw"])
    )

    print("=== params ===")
    hdr = (
        f"{'model':<12} {'g_i':>6} {'d_i':>5} {'θc':>5} {'θd':>5} {'Δθ':>5} "
        f"{'W_mm':>5} {'τM':>5} {'W_ii':>5} {'τI':>5} {'W_mi':>5} {'W_is':>5} "
        f"{'τa':>6} {'αd':>5} {'βd':>5}"
    )
    print(hdr)
    rows = []
    for label, jp in MODELS:
        jp = json_for(label, jp)
        mp, meta = load_plot_model(jp)
        thc = float(mp["action_thresholds"]["concordant"][0.0])
        thd = float(mp["action_thresholds"]["discordant"][0.0])
        rec = {
            "label": label,
            "jp": jp,
            "run": jp.parent,
            "mp": mp,
            "g_i": float(mp["g_i"]),
            "d_i": float(mp["d_i"]),
            "thc": thc,
            "thd": thd,
            "W_mm": float(mp["W_mm"]),
            "W_ii": float(mp["W_ii"]),
            "W_mi": float(mp["W_mi"]),
            "W_is": float(mp["W_is"]),
            "tau_a": float(mp["tau_a"]),
            "alpha_d": float(mp["alpha_d"]),
            "beta_d": float(mp["beta_d"]),
            "tau_M": tau_delta_ms(mp["W_mm"]),
            "tau_I": tau_delta_ms(mp["W_ii"]),
        }
        rows.append(rec)
        print(
            f"{label:<12} {rec['g_i']:6.1f} {rec['d_i']:5.1f} {thc:5.3f} {thd:5.3f} "
            f"{thd-thc:5.3f} {rec['W_mm']:5.3f} {rec['tau_M']:5.0f} "
            f"{rec['W_ii']:5.3f} {rec['tau_I']:5.0f} {rec['W_mi']:5.3f} "
            f"{rec['W_is']:5.3f} {rec['tau_a']:6.1f} {rec['alpha_d']:5.1f} "
            f"{rec['beta_d']:5.2f}"
        )

    print("\n=== timeouts (choice=0; RT still entered as max trial) ===")
    print(f"{'model':<12} {'n':>6} {'timeout%':>9} {'inc t/o%':>9} {'con t/o%':>9} "
          f"{'inc t/o @0':>11} {'mean inc RT all':>16} {'mean inc RT commit':>18}")
    for rec in rows:
        cache = rec["run"] / "psychometric_model_vs_data" / "combined_results.npy"
        signed, conc, timeout, rt, ch = trial_table(cache)
        inc = ~conc
        n = rt.size
        t_all = 100 * timeout.mean()
        t_inc = 100 * timeout[inc].mean()
        t_con = 100 * timeout[conc].mean()
        z = (signed == 0) & inc
        t_z = 100 * timeout[z].mean() if z.any() else np.nan
        m_all = np.mean(rt[inc])
        m_ok = np.mean(rt[inc & ~timeout])
        print(
            f"{rec['label']:<12} {n:6d} {t_all:8.2f}% {t_inc:8.2f}% {t_con:8.2f}% "
            f"{t_z:10.1f}% {m_all:16.3f} {m_ok:18.3f}"
        )

    print("\n=== incongruent mean RT (s) by signed contrast  [all trials] ===")
    print(f"{'signed':>8}", *[f"{r['label']:>12}" for r in rows], f"{'data':>12}")
    for i, s in enumerate(SIGNED):
        vals = []
        for rec in rows:
            cache = rec["run"] / "psychometric_model_vs_data" / "combined_results.npy"
            signed, conc, timeout, rt, ch = trial_table(cache)
            y, n = mean_at(signed, ~conc, rt)
            vals.append(y[i])
        print(f"{s:8.4f}", *[f"{v:12.3f}" for v in vals], f"{data_inc[i]:12.3f}")

    print("\n=== incongruent mean RT (s)  [committed only] ===")
    print(f"{'signed':>8}", *[f"{r['label']:>12}" for r in rows], f"{'data':>12}")
    for i, s in enumerate(SIGNED):
        vals = []
        for rec in rows:
            cache = rec["run"] / "psychometric_model_vs_data" / "combined_results.npy"
            signed, conc, timeout, rt, ch = trial_table(cache)
            y, n = mean_at(signed, (~conc) & (~timeout), rt)
            vals.append(y[i])
        print(f"{s:8.4f}", *[f"{v:12.3f}" for v in vals], f"{data_inc[i]:12.3f}")

    print("\n=== model − data incongruent RT (s), committed only ===")
    print(f"{'signed':>8}", *[f"{r['label']:>12}" for r in rows])
    for i, s in enumerate(SIGNED):
        diffs = []
        for rec in rows:
            cache = rec["run"] / "psychometric_model_vs_data" / "combined_results.npy"
            signed, conc, timeout, rt, ch = trial_table(cache)
            y, n = mean_at(signed, (~conc) & (~timeout), rt)
            diffs.append(y[i] - data_inc[i])
        print(f"{s:8.4f}", *[f"{v:+12.3f}" for v in diffs])

    print("\n=== same for congruent, committed only, model − data ===")
    print(f"{'signed':>8}", *[f"{r['label']:>12}" for r in rows])
    for i, s in enumerate(SIGNED):
        diffs = []
        for rec in rows:
            cache = rec["run"] / "psychometric_model_vs_data" / "combined_results.npy"
            signed, conc, timeout, rt, ch = trial_table(cache)
            y, n = mean_at(signed, conc & (~timeout), rt)
            diffs.append(y[i] - data_con[i])
        print(f"{s:8.4f}", *[f"{v:+12.3f}" for v in diffs])

    print("\n=== easy vs zero: inc RT (committed) and θd / g_i ===")
    print(f"{'model':<12} {'inc@±1':>8} {'inc@0':>8} {'data@0':>8} {'excess@0':>9} "
          f"{'θd':>6} {'g_i':>6} {'τa':>6}")
    for rec in rows:
        cache = rec["run"] / "psychometric_model_vs_data" / "combined_results.npy"
        signed, conc, timeout, rt, ch = trial_table(cache)
        ok = (~conc) & (~timeout)
        easy = np.mean(rt[ok & (np.abs(signed) == 1.0)])
        zero = np.mean(rt[ok & (signed == 0)])
        print(
            f"{rec['label']:<12} {easy:8.3f} {zero:8.3f} {data_inc[4]:8.3f} "
            f"{zero-data_inc[4]:+9.3f} {rec['thd']:6.3f} {rec['g_i']:6.1f} "
            f"{rec['tau_a']:6.1f}"
        )


if __name__ == "__main__":
    # plot_best_fit_results.latest_final may not exist; we don't import it.
    main()
