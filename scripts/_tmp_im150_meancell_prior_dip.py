"""Characterize the mid-window dip in BWM vs meancell / regular prior curves."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from plot_best_fit_results import (  # noqa: E402
    ensure_fit_data_links_paper,
    load_plot_model,
    make_shared_stimuli,
)
from model_functions import (  # noqa: E402
    prior_distance_I_M_both_alignments,
    resolve_prior_distance_window,
    run_model,
    _resample_to_len,
)

BASE = Path.home() / (
    "Downloads/ONE/openalyx.internationalbrainlab.org/models"
)
OUT = BASE / "stageB_hold_s89_im150_meancell_prior_dip.json"


def latest_final(d: Path) -> Path:
    finals = sorted(d.glob("weights_final_*.json"))
    if not finals:
        raise FileNotFoundError(d)
    return finals[-1]


def local_min_after_peak(y, t, t_lo=30.0, t_hi=130.0):
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    m = (t >= t_lo) & (t <= t_hi) & np.isfinite(y)
    if int(m.sum()) < 5:
        return None
    yy, tt = y[m], t[m]
    i_peak = int(np.argmax(yy))
    # dip = min after first local peak in window, else min in window
    after = yy[i_peak:]
    if after.size < 3:
        i = int(np.argmin(yy))
    else:
        i = i_peak + int(np.argmin(after))
    return {
        "t_peak": float(tt[i_peak]),
        "y_peak": float(yy[i_peak]),
        "t_min": float(tt[i]),
        "y_min": float(yy[i]),
        "depth": float(yy[i_peak] - yy[i]),
    }


def at(y, t, tq):
    y = np.asarray(y, float)
    t = np.asarray(t, float)
    if y.size == 0:
        return None
    return float(np.interp(tq, t, y))


def load_data_curves():
    stim = np.load("data_act_block_duringstim.npy", allow_pickle=True).flat[0]
    ch = np.load("data_act_block_duringchoice.npy", allow_pickle=True).flat[0]
    return {
        "I_stim": np.asarray(stim["r_int"], float),
        "M_stim": np.asarray(stim["r_move"], float),
        "I_choice": np.asarray(ch["r_int"], float),
        "M_choice": np.asarray(ch["r_move"], float),
    }


def sim_curves(jp, stim_bundle, prior_window_ms):
    mp, _ = load_plot_model(jp)
    mp = dict(mp)
    mp["prior_window_ms"] = prior_window_ms
    (
        stimuli, trial_strengths, trial_sides, block_sides,
        steps_before_obs, bps,
    ) = stim_bundle
    results = run_model(
        "data",
        stimuli, trial_strengths, trial_sides, block_sides, bps,
        steps_before_obs=steps_before_obs, verbose=False, backend="numba",
        **mp,
    )
    T, plot_win, _ = resolve_prior_distance_window(mp, T=72, plot_window=80)
    out = prior_distance_I_M_both_alignments(
        results, steps_before_obs, T=T, metric="l2",
        include_all_trials=True, lump_all=False,
    )
    rt = np.asarray(results["reaction_time"], float)
    ch = np.asarray(results["choices"])
    ok = ch != 0
    rt_ms = rt[ok] * float(mp.get("dt", 2.0))
    return mp, out, {
        "n_ok": int(ok.sum()),
        "rt_p10": float(np.percentile(rt_ms, 10)),
        "rt_p50": float(np.percentile(rt_ms, 50)),
        "rt_p90": float(np.percentile(rt_ms, 90)),
        "frac_lt80": float(np.mean(rt_ms < 80)),
        "frac_lt100": float(np.mean(rt_ms < 100)),
        "frac_lt120": float(np.mean(rt_ms < 120)),
        "frac_lt150": float(np.mean(rt_ms < 150)),
    }


def pack_curve(y, t, name):
    dip = local_min_after_peak(y, t)
    return {
        "name": name,
        "y0": at(y, t, t[0]),
        "y40": at(y, t, 40 if t[-1] > 0 else t[0] + 40),
        "y80": at(y, t, 80 if t[-1] > 0 else -80),
        "y110": at(y, t, 110 if t[-1] > 0 else -40),
        "yend": at(y, t, t[-1]),
        "dip": dip,
        "n": int(np.asarray(y).size),
    }


def main():
    ensure_fit_data_links_paper()
    data = load_data_curves()
    t_stim = np.linspace(0.0, 150.0, len(data["I_stim"]))
    t_ch = np.linspace(-150.0, 0.0, len(data["I_choice"]))

    stim_ref = latest_final(
        BASE / "weights_run_fj_stageB_hold_s89_regular_mask12-13_s101"
    )
    mp0, _ = load_plot_model(stim_ref)
    stim_bundle = make_shared_stimuli(mp0, bps=20, seed=12345)

    models = {
        "meancell_s12": (
            BASE / "weights_run_fj_stageB_hold_s89_im150_meancell_regular_mask12-13_s12",
            150.0,
        ),
        "meancell_s34": (
            BASE / "weights_run_fj_stageB_hold_s89_im150_meancell_regular_mask12-13_s34",
            150.0,
        ),
        "regular_s101": (
            BASE / "weights_run_fj_stageB_hold_s89_regular_mask12-13_s101",
            150.0,
        ),
    }

    report = {
        "data": {
            "I_stim": pack_curve(data["I_stim"], t_stim, "data I stim"),
            "M_stim": pack_curve(data["M_stim"], t_stim, "data M stim"),
            "I_choice": pack_curve(data["I_choice"], t_ch, "data I choice"),
            "M_choice": pack_curve(data["M_choice"], t_ch, "data M choice"),
        },
        "models": {},
    }
    print("DATA stim I:", json.dumps(report["data"]["I_stim"], indent=2))
    print("DATA stim M:", json.dumps(report["data"]["M_stim"], indent=2))
    print("DATA choice I:", json.dumps(report["data"]["I_choice"], indent=2))

    for label, (run, win) in models.items():
        jp = latest_final(run)
        mp, out, rt = sim_curves(jp, stim_bundle, win)
        I = _resample_to_len(out["I"]["start"], len(data["I_stim"]))
        M = _resample_to_len(out["M"]["start"], len(data["M_stim"]))
        Ia = _resample_to_len(out["I"]["action"], len(data["I_choice"]))
        Ma = _resample_to_len(out["M"]["action"], len(data["M_choice"]))
        rec = {
            "g_i": float(mp["g_i"]),
            "d_i": float(mp["d_i"]),
            "W_is": float(mp.get("W_is", np.nan)),
            "rt": rt,
            "I_stim": pack_curve(I, t_stim, f"{label} I stim"),
            "M_stim": pack_curve(M, t_stim, f"{label} M stim"),
            "I_choice": pack_curve(Ia, t_ch, f"{label} I choice"),
            "M_choice": pack_curve(Ma, t_ch, f"{label} M choice"),
        }
        report["models"][label] = rec
        print(
            f"\n{label} gi={rec['g_i']:.1f} Wis={rec['W_is']:.3f} "
            f"rt50={rt['rt_p50']:.0f} <80={rt['frac_lt80']:.3f} "
            f"<150={rt['frac_lt150']:.3f}"
        )
        for k in ("I_stim", "M_stim", "I_choice"):
            print(k, json.dumps(rec[k]["dip"]), rec[k]["y80"], rec[k]["yend"])

    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
