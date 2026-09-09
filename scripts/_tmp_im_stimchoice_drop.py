"""Why stim×choice I/M prior-distance drops after ~80 ms post-stim.

Regular s101, shared stim (bps=20, seed 12345), 150 ms start-aligned window.
Plots + JSON go in that run dir.
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

from plot_best_fit_results import (  # noqa: E402
    ensure_fit_data_links_paper,
    load_plot_model,
    make_shared_stimuli,
)
from model_functions import (  # noqa: E402
    _resample_to_len,
    prior_distance_I_M_both_alignments,
    run_model,
)

BASE = Path.home() / "Downloads/ONE/openalyx.internationalbrainlab.org/models"
RUN = BASE / "weights_run_fj_stageB_hold_s89_regular_mask12-13_s101"
T_MS = 150.0
DT = 2.0
T = int(round(T_MS / DT))
POST_ACTION_MS = 40.0


def latest_final(d: Path) -> Path:
    finals = sorted(d.glob("weights_final_*.json"))
    if not finals:
        raise FileNotFoundError(d)
    return finals[-1]


def load_data():
    stim = np.load(ROOT / "fit_targets/data_act_block_duringstim.npy", allow_pickle=True).flat[0]
    ch = np.load(ROOT / "fit_targets/data_act_block_duringchoice.npy", allow_pickle=True).flat[0]
    return {k: np.asarray(stim[k], float) for k in ("r_int", "r_move")}, {
        k: np.asarray(ch[k], float) for k in ("r_int", "r_move")
    }


def _l2(a, b):
    return np.linalg.norm(a - b, axis=-1)


def collect_start_segments(results, sbo, var_name, T):
    """Start-aligned segments plus epoch tags and trial meta.

    epoch[t]: 0 pre-action, 1 post-action, 2 next-ITI fill, -1 skipped
    """
    choices = results["choices"]
    trial_sides = results["trial_sides"]
    sub_prior = results["sub_prior"]
    rts = results["reaction_time"]
    var = np.asarray(results[var_name], float)
    n = len(choices)
    lens = [len(trial_sides[i]) for i in range(n)]
    offsets = np.cumsum([0] + lens[:-1])
    hard_need = sbo + 1
    post_action = int(results.get("post_action_steps", POST_ACTION_MS / DT))
    if "post_action_steps" not in results:
        post_action = int(round(POST_ACTION_MS / DT))

    rows = []
    for i in range(n):
        m_i = lens[i]
        if m_i < hard_need:
            continue
        ch = int(choices[i])
        if ch == 0:
            continue
        ts = int(np.sign(trial_sides[i][0]))
        rt = int(rts[i])
        post_avail = max(0, m_i - sbo)
        take_post = min(T, post_avail)
        parts = []
        epoch = np.full(T, -1, dtype=np.int8)
        if take_post > 0:
            start = offsets[i] + sbo
            parts.append(var[start : start + take_post, :])
            for t in range(take_post):
                epoch[t] = 0 if t < rt else 1
        if take_post < T:
            if i + 1 >= n:
                continue
            m_next = lens[i + 1]
            if m_next < hard_need:
                continue
            need = T - take_post
            pre_avail_next = min(sbo, m_next)
            if pre_avail_next < need:
                continue
            start_next = offsets[i + 1]
            parts.append(var[start_next : start_next + need, :])
            epoch[take_post:] = 2
        seg = np.vstack(parts)
        if seg.shape[0] != T:
            continue
        sp = 1 if sub_prior[i][0] < 0 else -1
        # trial-level concordance: stim side vs prior-favored side
        # channel 1 = right = +1 choice; prior sp coding matches production distance
        conc = int(ts == -sp)  # see note in summary; also store ts*sp
        p0 = float(np.asarray(sub_prior[i])[0])
        rows.append(
            {
                "ts": ts,
                "ch": ch,
                "sp": sp,
                "rt": rt,
                "p0": p0,
                "conc_ts_neg_sp": conc,
                "conc_ts_sp": int(ts == sp),
                "seg": seg,
                "epoch": epoch,
            }
        )
    return rows


def collect_action_segments(results, sbo, var_name, T):
    """Movement-aligned T-step snippets (same skip rules as production)."""
    choices = results["choices"]
    trial_sides = results["trial_sides"]
    sub_prior = results["sub_prior"]
    rts = results["reaction_time"]
    var = np.asarray(results[var_name], float)
    n = len(choices)
    lens = [len(trial_sides[i]) for i in range(n)]
    offsets = np.cumsum([0] + lens[:-1])
    hard_need = sbo + 1
    rows = []
    for i in range(n):
        if lens[i] < hard_need:
            continue
        ch = int(choices[i])
        if ch == 0:
            continue
        rt = int(rts[i])
        act_start = sbo + rt
        if act_start < T or act_start > lens[i]:
            continue
        start = offsets[i] + act_start - T
        seg = var[start : start + T, :]
        if seg.shape[0] != T:
            continue
        ts = int(np.sign(trial_sides[i][0]))
        sp = 1 if sub_prior[i][0] < 0 else -1
        rows.append({"ts": ts, "ch": ch, "sp": sp, "seg": seg})
    return rows


def cell_l2_pair(rows, cells):
    """Return ‖mean_c Δ‖ and mean_c ‖Δ‖ for the given (ts, ch) cells."""
    deltas = []
    per = []
    for ts, ch in cells:
        means = {}
        for sp in (+1, -1):
            segs = [r["seg"] for r in rows if r["ts"] == ts and r["ch"] == ch and r["sp"] == sp]
            means[sp] = np.mean(np.stack(segs, 0), 0) if segs else np.full((T, 2), np.nan)
        dlt = means[+1] - means[-1]
        deltas.append(dlt)
        per.append(_l2(means[+1], means[-1]))
    pooled = np.linalg.norm(np.nanmean(np.stack(deltas, 0), 0), axis=1)
    mean_cell = np.nanmean(np.stack(per, 0), 0)
    return pooled, mean_cell


def balanced_l2(rows, key_fn, mask_fn=None):
    """Equal-weight L2 over keys returned by key_fn(row) -> (cell..., sp)."""
    buckets = {}
    for r in rows:
        if mask_fn is not None:
            keep = mask_fn(r)
            if keep is None:
                continue
            seg = np.where(keep[:, None], r["seg"], np.nan)
        else:
            seg = r["seg"]
        buckets.setdefault(key_fn(r), []).append(seg)

    def mean_sp(sp):
        cells = {}
        for key, segs in buckets.items():
            if key[-1] != sp:
                continue
            cell = key[:-1]
            stack = np.stack(segs, 0)
            cells[cell] = np.nanmean(stack, 0)
        if not cells:
            return None
        return np.nanmean(np.stack(list(cells.values()), 0), 0)

    a, b = mean_sp(+1), mean_sp(-1)
    if a is None or b is None:
        return np.full(T, np.nan)
    return _l2(a, b)


def cell_signed(rows, ts, ch):
    """Mean I/M side-diff (ch1-ch0) for one (stim, choice) cell, by prior."""
    out = {}
    for sp in (+1, -1):
        segs = [r["seg"] for r in rows if r["ts"] == ts and r["ch"] == ch and r["sp"] == sp]
        if not segs:
            out[sp] = np.full(T, np.nan)
        else:
            m = np.mean(np.stack(segs, 0), 0)
            out[sp] = m[:, 1] - m[:, 0]
    return out


def summarize_data(d_stim, d_ch):
    n = len(d_stim["r_int"])
    t = np.linspace(0.0, 150.0, n)
    i80 = int(np.argmin(np.abs(t - 80.0)))
    return {
        "n": n,
        "I_0": float(d_stim["r_int"][0]),
        "I_peak_pre80": float(np.nanmax(d_stim["r_int"][: i80 + 1])),
        "I_80": float(d_stim["r_int"][i80]),
        "I_end": float(d_stim["r_int"][-1]),
        "I_argmax_ms": float(t[int(np.nanargmax(d_stim["r_int"]))]),
        "M_0": float(d_stim["r_move"][0]),
        "M_80": float(d_stim["r_move"][i80]),
        "M_end": float(d_stim["r_move"][-1]),
        "I_ch_0": float(d_ch["r_int"][0]),
        "I_ch_end": float(d_ch["r_int"][-1]),
        "M_ch_0": float(d_ch["r_move"][0]),
        "M_ch_end": float(d_ch["r_move"][-1]),
        "t80_bin": i80,
    }


def main():
    ensure_fit_data_links_paper()
    d_stim, d_ch = load_data()
    data_sum = summarize_data(d_stim, d_ch)

    mp, meta = load_plot_model(latest_final(RUN))
    dt = float(mp.get("dt", DT))
    post_action = int(mp.get("post_action_steps", round(POST_ACTION_MS / dt)))
    stim_bundle = make_shared_stimuli(mp, bps=20, seed=12345)
    stimuli, tstr, tsides, bsides, sbo, bps = stim_bundle
    results = run_model(
        "data", stimuli, tstr, tsides, bsides, bps,
        steps_before_obs=sbo, verbose=False, backend="numba", **mp,
    )

    kw = dict(results=results, steps_before_obs=sbo, T=T, metric="l2", include_all_trials=True)
    d_sc = prior_distance_I_M_both_alignments(**kw, stratum="stim_choice")
    d_st = prior_distance_I_M_both_alignments(**kw, stratum="stim")
    d_sc_corr = prior_distance_I_M_both_alignments(
        **{**kw, "include_all_trials": False}, stratum="stim_choice"
    )

    rows_I = collect_start_segments(results, sbo, "I", T)
    rows_M = collect_start_segments(results, sbo, "M", T)
    rows_S = collect_start_segments(results, sbo, "S", T)

    # which concordance coding splits RT the way θ_c > θ_d predicts
    rts = np.array([r["rt"] for r in rows_I], float) * dt
    conc_a = np.array([r["conc_ts_neg_sp"] for r in rows_I], bool)
    conc_b = np.array([r["conc_ts_sp"] for r in rows_I], bool)

    def rt_stats(mask):
        x = rts[mask]
        return {
            "n": int(mask.sum()),
            "median": float(np.median(x)) if x.size else None,
            "p10": float(np.percentile(x, 10)) if x.size else None,
            "p25": float(np.percentile(x, 25)) if x.size else None,
            "p75": float(np.percentile(x, 75)) if x.size else None,
            "frac_lt_80": float(np.mean(x < 80)) if x.size else None,
            "frac_lt_150": float(np.mean(x < 150)) if x.size else None,
        }

    # pick the coding where the "concordant" group is slower (θ_c > θ_d)
    med_a = np.median(rts[conc_a]) if conc_a.any() else 0
    med_b = np.median(rts[conc_b]) if conc_b.any() else 0
    use_neg = med_a > med_b
    conc_mask = conc_a if use_neg else conc_b
    conc_name = "ts == -sp" if use_neg else "ts == sp"

    t_ms = np.arange(T) * dt
    i80 = int(np.argmin(np.abs(t_ms - 80.0)))

    def key_sc(r):
        return (r["ts"], r["ch"], r["sp"])

    def key_st(r):
        return (r["ts"], r["sp"])

    I_sc = balanced_l2(rows_I, key_sc)
    I_st = balanced_l2(rows_I, key_st)
    M_sc = balanced_l2(rows_M, key_sc)
    M_st = balanced_l2(rows_M, key_st)

    def uncommitted(r):
        return r["epoch"] == 0

    def long_rt(r):
        return None if r["rt"] * dt < T_MS else np.ones(T, dtype=bool)

    def no_fill(r):
        return r["epoch"] != 2

    I_sc_unc = balanced_l2(rows_I, key_sc, uncommitted)
    I_sc_long = balanced_l2(rows_I, key_sc, long_rt)
    I_sc_nofill = balanced_l2(rows_I, key_sc, no_fill)
    M_sc_unc = balanced_l2(rows_M, key_sc, uncommitted)
    I_st_unc = balanced_l2(rows_I, key_st, uncommitted)
    I_st_long = balanced_l2(rows_I, key_st, long_rt)

    # epoch mix
    epochs = np.stack([r["epoch"] for r in rows_I], 0)
    frac = {lab: np.mean(epochs == k, 0) for lab, k in (("pre", 0), ("post", 1), ("iti", 2))}

    # within-cell signed I (right minus left) for all 4 cells
    cells = [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]
    signed = {c: cell_signed(rows_I, *c) for c in cells}
    signed_M = {c: cell_signed(rows_M, *c) for c in cells}

    def cell_means(rows, ts, ch):
        out = {}
        for sp in (+1, -1):
            segs = [r["seg"] for r in rows if r["ts"] == ts and r["ch"] == ch and r["sp"] == sp]
            out[sp] = np.mean(np.stack(segs, 0), 0) if segs else np.full((T, 2), np.nan)
        return out

    means_I = {c: cell_means(rows_I, *c) for c in cells}
    cell_l2 = {}
    cell_delta = {}
    cell_n = {}
    for c in cells:
        cell_l2[c] = _l2(means_I[c][+1], means_I[c][-1])
        cell_delta[c] = means_I[c][+1] - means_I[c][-1]
        cell_n[c] = {
            "sp+": int(sum(1 for r in rows_I if r["ts"] == c[0] and r["ch"] == c[1] and r["sp"] == +1)),
            "sp-": int(sum(1 for r in rows_I if r["ts"] == c[0] and r["ch"] == c[1] and r["sp"] == -1)),
        }
    deltas = np.stack([cell_delta[c] for c in cells], 0)  # (4, T, 2)
    mean_delta = np.nanmean(deltas, 0)
    l2_mean_delta = np.linalg.norm(mean_delta, axis=1)
    mean_l2 = np.nanmean(np.stack([cell_l2[c] for c in cells], 0), 0)
    cancel = l2_mean_delta / np.maximum(mean_l2, 1e-12)
    correct_cells = [(+1, +1), (-1, -1)]
    error_cells = [(+1, -1), (-1, +1)]
    l2_correct = np.linalg.norm(np.nanmean(np.stack([cell_delta[c] for c in correct_cells], 0), 0), axis=1)
    l2_error = np.linalg.norm(np.nanmean(np.stack([cell_delta[c] for c in error_cells], 0), 0), axis=1)

    means_M = {c: cell_means(rows_M, *c) for c in cells}
    cell_l2_M = {c: _l2(means_M[c][+1], means_M[c][-1]) for c in cells}
    cell_delta_M = {c: means_M[c][+1] - means_M[c][-1] for c in cells}
    l2_mean_delta_M = np.linalg.norm(
        np.nanmean(np.stack([cell_delta_M[c] for c in cells], 0), 0), axis=1
    )
    mean_l2_M = np.nanmean(np.stack([cell_l2_M[c] for c in cells], 0), 0)
    cancel_M = l2_mean_delta_M / np.maximum(mean_l2_M, 1e-12)

    # concordance at stimOn from P vs stim side (S is ~0 at t=0)
    # channel 1 = right = trial_sides +1; P[1]>P[0] favors right
    P_all = np.asarray(results["P"], float)
    trial_sides_raw = results["trial_sides"]
    lens_raw = [len(trial_sides_raw[i]) for i in range(len(results["choices"]))]
    offsets_raw = np.cumsum([0] + lens_raw[:-1])
    conc_true = []
    rts_true = []
    for i in range(len(results["choices"])):
        if int(results["choices"][i]) == 0 or lens_raw[i] < sbo + 1:
            continue
        ts = int(np.sign(trial_sides_raw[i][0]))
        pdiff = float(P_all[offsets_raw[i] + sbo, 0] - P_all[offsets_raw[i] + sbo, 1])
        prior_right = pdiff < 0
        stim_right = ts > 0
        conc_true.append(stim_right == prior_right)
        rts_true.append(int(results["reaction_time"][i]) * dt)
    conc_true = np.asarray(conc_true, bool)
    rts_true = np.asarray(rts_true, float)
    # S magnitude
    S_abs = []
    for r in rows_S:
        S_abs.append(np.abs(r["seg"][:, 1] - r["seg"][:, 0]))
    S_abs = np.mean(np.stack(S_abs, 0), 0)

    # production function check
    I_sc_prod = np.asarray(d_sc["I"]["start"], float)
    I_st_prod = np.asarray(d_st["I"]["start"], float)
    I_sc_act = np.asarray(d_sc["I"]["action"], float)
    M_sc_act = np.asarray(d_sc["M"]["action"], float)
    I_st_act = np.asarray(d_st["I"]["action"], float)
    M_st_act = np.asarray(d_st["M"]["action"], float)
    M_sc_prod = np.asarray(d_sc["M"]["start"], float)
    I_sc_corr = np.asarray(d_sc_corr["I"]["start"], float)
    M_sc_corr = np.asarray(d_sc_corr["M"]["start"], float)

    def at(y, i):
        return float(y[i]) if y.size > i else float("nan")

    summary = {
        "run": str(RUN),
        "g_i": float(mp.get("g_i", meta.get("g", {}).get("g_i", np.nan))),
        "d_i": float(mp.get("d_i", meta.get("d", {}).get("d_i", np.nan))),
        "W_ii": float((meta.get("W") or {}).get("W_ii", mp.get("W_ii", np.nan))),
        "theta_c": float((meta.get("theta") or {}).get("theta_c", np.nan)),
        "theta_d": float((meta.get("theta") or {}).get("theta_d", np.nan)),
        "tau_i": float(mp.get("tau_i", np.nan)),
        "tau_s": float(mp.get("tau_s", np.nan)),
        "dt": dt,
        "T": T,
        "post_action_steps": post_action,
        "n_trials": len(rows_I),
        "concordance_coding": "stim_side == prior_side at stimOn",
        "rt_all": rt_stats(np.ones(len(rts), dtype=bool)),
        "rt_concordant_proxy": rt_stats(conc_mask),
        "rt_discordant_proxy": rt_stats(~conc_mask),
        "rt_concordant": {
            "n": int(conc_true.sum()) if conc_true.size else 0,
            "median": float(np.median(rts_true[conc_true])) if conc_true.any() else None,
            "p10": float(np.percentile(rts_true[conc_true], 10)) if conc_true.any() else None,
            "p25": float(np.percentile(rts_true[conc_true], 25)) if conc_true.any() else None,
            "p75": float(np.percentile(rts_true[conc_true], 75)) if conc_true.any() else None,
            "frac_lt_80": float(np.mean(rts_true[conc_true] < 80)) if conc_true.any() else None,
            "frac_lt_150": float(np.mean(rts_true[conc_true] < 150)) if conc_true.any() else None,
        },
        "rt_discordant": {
            "n": int((~conc_true).sum()) if conc_true.size else 0,
            "median": float(np.median(rts_true[~conc_true])) if (~conc_true).any() else None,
            "p10": float(np.percentile(rts_true[~conc_true], 10)) if (~conc_true).any() else None,
            "p25": float(np.percentile(rts_true[~conc_true], 25)) if (~conc_true).any() else None,
            "p75": float(np.percentile(rts_true[~conc_true], 75)) if (~conc_true).any() else None,
            "frac_lt_80": float(np.mean(rts_true[~conc_true] < 80)) if (~conc_true).any() else None,
            "frac_lt_150": float(np.mean(rts_true[~conc_true] < 150)) if (~conc_true).any() else None,
        },
        "cell_n": {f"ts{c[0]}_ch{c[1]}": cell_n[c] for c in cells},
        "data": data_sum,
        "model": {
            "I_sc_0": at(I_sc_prod, 0),
            "I_sc_80": at(I_sc_prod, i80),
            "I_sc_end": at(I_sc_prod, -1),
            "I_st_0": at(I_st_prod, 0),
            "I_st_80": at(I_st_prod, i80),
            "I_st_end": at(I_st_prod, -1),
            "I_sc_unc_80": at(I_sc_unc, i80),
            "I_sc_unc_end": at(I_sc_unc, -1),
            "I_sc_long_80": at(I_sc_long, i80),
            "I_sc_long_end": at(I_sc_long, -1),
            "I_sc_nofill_80": at(I_sc_nofill, i80),
            "I_sc_nofill_end": at(I_sc_nofill, -1),
            "I_st_unc_end": at(I_st_unc, -1),
            "I_st_long_end": at(I_st_long, -1),
            "M_sc_0": at(M_sc_prod, 0),
            "M_sc_80": at(M_sc_prod, i80),
            "M_sc_end": at(M_sc_prod, -1),
            "M_st_end": at(np.asarray(d_st["M"]["start"], float), -1),
            "I_sc_act_start": at(I_sc_act, 0),
            "I_sc_act_end": at(I_sc_act, -1),
            "M_sc_act_start": at(M_sc_act, 0),
            "M_sc_act_end": at(M_sc_act, -1),
            "rebuild_max_abs_err": float(np.nanmax(np.abs(I_sc - I_sc_prod))),
            "I_sc_corr_0": at(np.asarray(d_sc_corr["I"]["start"], float), 0),
            "I_sc_corr_80": at(np.asarray(d_sc_corr["I"]["start"], float), i80),
            "I_sc_corr_end": at(np.asarray(d_sc_corr["I"]["start"], float), -1),
            "I_mean_cell_l2_80": float(mean_l2[i80]),
            "I_mean_cell_l2_end": float(mean_l2[-1]),
            "I_l2_mean_delta_80": float(l2_mean_delta[i80]),
            "I_l2_mean_delta_end": float(l2_mean_delta[-1]),
            "cancel_80": float(cancel[i80]),
            "cancel_end": float(cancel[-1]),
            "I_correct_cells_80": float(l2_correct[i80]),
            "I_correct_cells_end": float(l2_correct[-1]),
            "I_error_cells_80": float(l2_error[i80]),
            "I_error_cells_end": float(l2_error[-1]),
            "I_cell_l2_80": {f"ts{c[0]}_ch{c[1]}": float(cell_l2[c][i80]) for c in cells},
            "I_cell_l2_end": {f"ts{c[0]}_ch{c[1]}": float(cell_l2[c][-1]) for c in cells},
        },
        "frac_pre_80": float(frac["pre"][i80]),
        "frac_post_80": float(frac["post"][i80]),
        "frac_iti_80": float(frac["iti"][i80]),
        "frac_pre_end": float(frac["pre"][-1]),
        "frac_post_end": float(frac["post"][-1]),
        "frac_iti_end": float(frac["iti"][-1]),
        "S_abs_0": float(S_abs[0]),
        "S_abs_80": float(S_abs[i80]),
        "S_abs_end": float(S_abs[-1]),
        "S_abs_argmax_ms": float(t_ms[int(np.argmax(S_abs))]),
    }

    # within-cell gap ( |signed_P+ - signed_P-| ) averaged over 4 cells
    gaps = []
    for c in cells:
        gaps.append(np.abs(signed[c][+1] - signed[c][-1]))
    gap_mean = np.nanmean(np.stack(gaps, 0), 0)
    summary["I_cellgap_0"] = float(gap_mean[0])
    summary["I_cellgap_80"] = float(gap_mean[i80])
    summary["I_cellgap_end"] = float(gap_mean[-1])

    out = RUN / "stimchoice_drop_diag"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    t_data = np.linspace(0.0, 150.0, len(d_stim["r_int"]))
    t_ch = np.linspace(-150.0, 0.0, len(d_ch["r_int"]))
    t_act = np.linspace(-T_MS, 0.0, len(I_sc_act)) if I_sc_act.size else t_ms - T_MS

    fig, axs = plt.subplots(2, 2, figsize=(9.2, 6.4), dpi=140)

    ax = axs[0, 0]
    ax.plot(t_data, d_stim["r_int"], color="gold", lw=2.2, label="I data")
    ax.plot(t_data, d_stim["r_move"], color="tomato", lw=2.2, label="M data")
    ax.plot(t_ms, I_sc_prod, "--", color="gold", lw=1.8, label="I stim×choice")
    ax.plot(t_ms, M_sc_prod, "--", color="tomato", lw=1.8, label="M stim×choice")
    ax.plot(t_ms, I_st_prod, ":", color="0.25", lw=2.0, label="I stim only")
    ax.plot(t_ms, np.asarray(d_st["M"]["start"], float), ":", color="0.5", lw=2.0, label="M stim only")
    ax.axvline(80, color="0.6", ls="-.", lw=0.8)
    ax.set_xlim(0, 150)
    ax.set_xlabel("ms after stimOn")
    ax.set_ylabel("prior distance")
    ax.set_title("Start-aligned: stratum vs data")
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    ax = axs[0, 1]
    ax.plot(t_ms, I_sc_prod, color="gold", lw=2.0, label="I 4-cell equal-weight")
    ax.plot(t_ms, I_sc_corr, color="C0", lw=1.8, label="I correct-only (2 cells)")
    ax.plot(t_ms, mean_l2, color="C2", lw=1.8, label="mean of 4 per-cell L2")
    ax.plot(t_ms, l2_mean_delta, ":", color="0.2", lw=2.0, label="L2 of mean Δ (cancels)")
    ax.plot(t_ms, l2_correct, "--", color="C0", lw=1.2, label="correct-cell mean Δ")
    ax.plot(t_ms, l2_error, "--", color="C3", lw=1.2, label="error-cell mean Δ")
    ax.axvline(80, color="0.6", ls="-.", lw=0.8)
    ax.set_xlim(0, 150)
    ax.set_xlabel("ms after stimOn")
    ax.set_ylabel("I prior distance")
    ax.set_title("4-cell average vs per-cell distance")
    ax.legend(frameon=False, fontsize=6.5)

    ax = axs[1, 0]
    labels = {
        (+1, +1): "stim R, choice R",
        (+1, -1): "stim R, choice L",
        (-1, +1): "stim L, choice R",
        (-1, -1): "stim L, choice L",
    }
    for c, lab in labels.items():
        ax.plot(t_ms, signed[c][+1], lw=1.3, label=f"{lab} sp+")
        ax.plot(t_ms, signed[c][-1], lw=1.3, ls="--", label=f"{lab} sp−")
    ax.axvline(80, color="0.6", ls="-.", lw=0.8)
    ax.axhline(0, color="0.85", lw=0.6)
    ax.set_xlim(0, 150)
    ax.set_xlabel("ms after stimOn")
    ax.set_ylabel("I right − I left")
    ax.set_title("Within stim×choice cells, both priors")
    ax.legend(frameon=False, fontsize=5.5, ncol=2)

    ax = axs[1, 1]
    ax.plot(t_act, I_sc_act, color="gold", lw=2.0, label="I stim×choice")
    ax.plot(t_act, M_sc_act, color="tomato", lw=2.0, label="M stim×choice")
    ax.plot(t_act, I_st_act, ":", color="0.25", lw=2.0, label="I stim only")
    ax.plot(t_act, M_st_act, ":", color="0.5", lw=2.0, label="M stim only")
    ax.plot(t_ch, d_ch["r_int"], color="gold", lw=1.2, alpha=0.7, label="I data")
    ax.plot(t_ch, d_ch["r_move"], color="tomato", lw=1.2, alpha=0.7, label="M data")
    ax.set_xlim(-150, 0)
    ax.set_xlabel("ms before movement")
    ax.set_ylabel("prior distance")
    ax.set_title("Movement-aligned (same stratum)")
    ax.legend(frameon=False, fontsize=7)

    for ax in axs.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(
        f"regular s101  θc={summary['theta_c']:.2f} θd={summary['theta_d']:.2f}  "
        f"g_i={summary['g_i']:.0f}  W_ii={summary['W_ii']:.2f}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(9.4, 2.8), dpi=140)
    ax = axs[0]
    bins = np.linspace(0, 400, 41)
    ax.hist(rts_true, bins=bins, color="0.75", label="all")
    if conc_true.size:
        ax.hist(rts_true[conc_true], bins=bins, histtype="step", color="C0", lw=1.6, label="concordant")
        ax.hist(rts_true[~conc_true], bins=bins, histtype="step", color="C3", lw=1.6, label="discordant")
    ax.axvline(80, color="0.4", ls="-.", lw=0.8)
    ax.axvline(150, color="0.4", ls=":", lw=0.8)
    ax.set_xlabel("RT (ms)")
    ax.set_ylabel("trials")
    ax.set_title("RT by stim×prior concordance")
    ax.legend(frameon=False, fontsize=7)

    ax = axs[1]
    ax.plot(t_ms, frac["pre"], color="C0", lw=1.8, label="pre-action")
    ax.plot(t_ms, frac["post"], color="C1", lw=1.8, label="post-action")
    ax.plot(t_ms, frac["iti"], color="C3", lw=1.8, label="next-ITI fill")
    ax.axvline(80, color="0.6", ls="-.", lw=0.8)
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 1)
    ax.set_xlabel("ms after stimOn")
    ax.set_ylabel("fraction of trials")
    ax.set_title("What occupies each start-aligned bin")
    ax.legend(frameon=False, fontsize=7)

    ax = axs[2]
    ax.plot(t_ms, S_abs, color="C2", lw=2.0, label="mean |S_R−S_L|")
    ax.plot(t_ms, gap_mean, color="gold", lw=2.0, label="mean |ΔI| within cells")
    ax.plot(t_ms, cancel, color="0.3", lw=1.6, label="alignment ||mean Δ|| / mean||Δ||")
    ax.axvline(80, color="0.6", ls="-.", lw=0.8)
    ax.set_xlim(0, 150)
    ax.set_xlabel("ms after stimOn")
    ax.set_ylabel("amplitude / alignment")
    ax.set_title("S transient, within-cell gap, 4-cell alignment")
    ax.legend(frameon=False, fontsize=6.5)
    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "composition.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(7.6, 5.4), dpi=150, sharex="col")
    series = (
        ("I", l2_mean_delta, mean_l2, cancel, "gold"),
        ("M", l2_mean_delta_M, mean_l2_M, cancel_M, "tomato"),
    )
    for col, (name, pooled, per_cell, aln, color) in enumerate(series):
        ax = axs[0, col]
        ax.plot(t_ms, per_cell, color=color, lw=2.2, label=r"mean$_c\,\|\mu_+-\mu_-\|$")
        ax.plot(
            t_ms, pooled, color="0.15", lw=2.0, ls="--",
            label=r"$\|\mathrm{mean}_c(\mu_+-\mu_-)\|$",
        )
        ax.axvline(80, color="0.65", ls="-.", lw=0.8)
        ax.set_xlim(0, 150)
        ax.set_ylabel("prior distance")
        ax.set_title(f"{name}  stim×choice 4-cell")
        ax.legend(frameon=False, fontsize=8, loc="upper left")

        ax = axs[1, col]
        ax.plot(t_ms, aln, color="0.2", lw=2.0)
        ax.axvline(80, color="0.65", ls="-.", lw=0.8)
        ax.axhline(1.0, color="0.85", lw=0.6)
        ax.set_xlim(0, 150)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("ms after stimOn")
        ax.set_ylabel(r"$\|\mathrm{mean}\,\Delta\|$ / mean$\|\Delta\|$")
    for ax in axs.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(
        "Equal-weight L2 vs mean of per-cell L2  ·  regular s101",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out / "mean_vs_cell_l2.png", dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(out / "mean_vs_cell_l2.svg", bbox_inches="tight", transparent=True)
    plt.close(fig)

    np.savez(
        out / "mean_vs_cell_l2.npz",
        t_ms=t_ms,
        I_l2_mean_delta=l2_mean_delta,
        I_mean_cell_l2=mean_l2,
        I_alignment=cancel,
        M_l2_mean_delta=l2_mean_delta_M,
        M_mean_cell_l2=mean_l2_M,
        M_alignment=cancel_M,
    )

    rows_I_act = collect_action_segments(results, sbo, "I", T)
    rows_M_act = collect_action_segments(results, sbo, "M", T)
    I_act_pooled, I_act_cell = cell_l2_pair(rows_I_act, cells)
    M_act_pooled, M_act_cell = cell_l2_pair(rows_M_act, cells)

    n_stim = len(d_stim["r_int"])
    n_ch = len(d_ch["r_int"])
    t_stim = np.linspace(0.0, T_MS, n_stim)
    t_choice = np.linspace(-T_MS, 0.0, n_ch)

    def onto_data(y, n):
        return _resample_to_len(np.asarray(y, float), n)

    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.7), dpi=150, sharey=True)
    ax = axs[0]
    ax.plot(t_stim, d_stim["r_int"], color="gold", lw=2.2, label="I data")
    ax.plot(t_stim, d_stim["r_move"], color="tomato", lw=2.2, label="M data")
    ax.plot(
        t_stim, onto_data(I_sc_prod, n_stim), "--", color="gold", lw=1.8,
        label=r"I model $\|\mathrm{mean}_c\Delta\|$",
    )
    ax.plot(
        t_stim, onto_data(M_sc_prod, n_stim), "--", color="tomato", lw=1.8,
        label=r"M model $\|\mathrm{mean}_c\Delta\|$",
    )
    ax.plot(
        t_stim, onto_data(mean_l2, n_stim), ":", color="gold", lw=2.2,
        label=r"I model mean$_c\|\Delta\|$",
    )
    ax.plot(
        t_stim, onto_data(mean_l2_M, n_stim), ":", color="tomato", lw=2.2,
        label=r"M model mean$_c\|\Delta\|$",
    )
    ax.axvline(80, color="0.7", ls="-.", lw=0.7)
    ax.set_xlim(0, T_MS)
    ax.set_xlabel("ms after stimOn")
    ax.set_ylabel(r"$d^{\mathrm{prior}}_{\{I,M\}}(t)$")

    ax = axs[1]
    ax.plot(t_choice, d_ch["r_int"], color="gold", lw=2.2)
    ax.plot(t_choice, d_ch["r_move"], color="tomato", lw=2.2)
    ax.plot(t_choice, onto_data(I_sc_act, n_ch), "--", color="gold", lw=1.8)
    ax.plot(
        t_choice, onto_data(np.asarray(d_sc["M"]["action"], float), n_ch),
        "--", color="tomato", lw=1.8,
    )
    ax.plot(t_choice, onto_data(I_act_cell, n_ch), ":", color="gold", lw=2.2)
    ax.plot(t_choice, onto_data(M_act_cell, n_ch), ":", color="tomato", lw=2.2)
    ax.set_xlim(-T_MS, 0.0)
    ax.set_xlabel("ms before movement")
    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10)
    axs[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.suptitle(
        "regular s101  150 ms  solid=data  dashed=‖mean Δ‖  dotted=mean ‖Δ‖",
        fontsize=10,
    )
    fig.tight_layout()
    pe = out / "prior_effects_150ms_cellmean.png"
    fig.savefig(pe, dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(out / "prior_effects_150ms_cellmean.svg", bbox_inches="tight", transparent=True)
    fig.savefig(RUN / "prior_effects_150ms_cellmean.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"wrote {out / 'curves.png'}")
    print(f"wrote {out / 'composition.png'}")
    print(f"wrote {out / 'mean_vs_cell_l2.png'}")
    print(f"wrote {pe}")


if __name__ == "__main__":
    main()
