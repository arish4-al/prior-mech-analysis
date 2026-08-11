"""
Fit retinal front-end params (Stage A of staged retinal→joint).

7-d optimizer vector (asinh β_w, matching fit_joint retinal block):

    0 alpha_w (log)
    1 beta_w  (asinh; z = asinh(β/BETA_W_SCALE))
    2–6 alpha_d, beta_d, tau_a, W_as, W_ss (log)

Prior gains g_*/d_* are held at 0; network W/θ fixed at script defaults.
Loss = L_S only (rms / compute_sse_stim_right vs avg_mean_R).

Optimizer scaffolding is fit_weights.fit_weights_two_stage_v2 with vector_api hooks
(same DE→CMA→local pipeline as weights / joint).
"""
from __future__ import annotations

import datetime
import json
import time
import traceback
from pathlib import Path

import numpy as np

from model_functions import *  # noqa: F401,F403
import fit_weights as fw
from fit_weights import (
    fit_weights_two_stage_v2,
    _ensure_run_dirs,
    _now_iso,
    max_obs_per_trial,
    steps_before_obs,
    blocks_per_session,
    trials_per_block_param,
    block_side_probs,
    num_stimulus_strength,
    min_stimulus_strength,
    max_stimulus_strength,
    min_trials_per_block,
    max_trials_per_block,
)

# ---- Stage-A defaults (match original fit_retinal script anchors) ----
# Non-retinal: W / θ / all prior gains. Retinal: script starting values.
STAGE_A_NETWORK = {
    "direct_offset": False,
    "W_pp": 0.45, "W_ii": 0.375, "W_mm": 0.139,
    "W_is": 0.119, "W_pi": 0.00107, "W_mi": 1.471,
    "g_i": 0.0, "d_i": 0.0, "g_m": 0.0, "d_m": 0.0,
    "g_s": 0.0, "d_s": 0.0,
}
STAGE_A_RETINAL = {
    "alpha_w": 1.569, "beta_w": -0.0815,
    "alpha_d": 35.277, "beta_d": 2.0515,
    "tau_a": 320.6594, "W_as": 29.3573, "W_ss": 0.00069,
}
STAGE_A_THETA = (0.78, 0.54)

model_params.update(STAGE_A_NETWORK)
model_params.update(STAGE_A_RETINAL)
model_params["action_thresholds"] = {
    "concordant": {c: STAGE_A_THETA[0] for c in [1.0, 0.25, 0.125, 0.0625, 0.0]},
    "discordant": {c: STAGE_A_THETA[1] for c in [1.0, 0.25, 0.125, 0.0625, 0.0]},
}

# Re-export plotting / counters from fit_weights so drivers can share state.
loss_history = fw.loss_history
_eval_counter = fw._eval_counter
enable_realtime_plot = fw.enable_realtime_plot
disable_realtime_plot = fw.disable_realtime_plot

LOG_ZERO = -30.0
D_RETINAL = 7
BETA_W_IDX = 1
BETA_W_NATIVE = (-0.2, 0.2)
BETA_W_SCALE = 0.05

PARAMS = ["alpha_w", "beta_w", "alpha_d", "beta_d", "tau_a", "W_as", "W_ss"]
IDX = {n: i for i, n in enumerate(PARAMS)}
PARAM_NAMES = list(PARAMS)

NATIVE_BOUNDS = {
    "alpha_w": (1.0, 2.6),
    "beta_w": BETA_W_NATIVE,
    "alpha_d": (20.0, 40.0),
    "beta_d": (1e-2, 3.0),
    "tau_a": (100.0, 400.0),
    "W_as": (1.0, 50.0),
    "W_ss": (1e-6, 2e-1),
}

# CMA σ in optimizer space (β_w in asinh).
CMA_STDS = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=float)

# Default polish: all retinal dims (intersected with train_mask).
DEFAULT_REFINE_IDX = list(range(D_RETINAL))

diag = {
    "evals": 0, "sim_calls": 0, "sim_ok": 0, "sim_nan": 0, "sim_exc": 0,
    "t_loss": 0.0, "t_sim": 0.0,
}
_exc_counters = {"unpack": 0, "stimuli": 0, "sim": 0, "avg": 0, "sse": 0, "other": 0}
_MAX_SHOW = 3


def beta_w_to_opt(beta_w):
    return float(np.arcsinh(float(beta_w) / BETA_W_SCALE))


def beta_w_from_opt(z):
    return float(BETA_W_SCALE * np.sinh(float(z)))


def apply_retinal_stage_a_defaults(mp=None):
    """Zero all prior gains; set Stage-A network W/θ anchors on model_params (or mp)."""
    target = model_params if mp is None else mp
    target.update(STAGE_A_NETWORK)
    th = STAGE_A_THETA
    target["action_thresholds"] = {
        "concordant": {c: th[0] for c in [1.0, 0.25, 0.125, 0.0625, 0.0]},
        "discordant": {c: th[1] for c in [1.0, 0.25, 0.125, 0.0625, 0.0]},
    }
    return target


def freeze_fill_retinal():
    fill = np.full(D_RETINAL, LOG_ZERO, dtype=float)
    fill[BETA_W_IDX] = beta_w_to_opt(0.0)
    return fill


def _log_bounds_retinal():
    b_alpha_w = NATIVE_BOUNDS["alpha_w"]
    b_beta_w = NATIVE_BOUNDS["beta_w"]
    b_alpha_d = NATIVE_BOUNDS["alpha_d"]
    b_beta_d = NATIVE_BOUNDS["beta_d"]
    btau_a = NATIVE_BOUNDS["tau_a"]
    bAs = NATIVE_BOUNDS["W_as"]
    bSS = NATIVE_BOUNDS["W_ss"]
    L = [
        np.log(b_alpha_w[0]), beta_w_to_opt(b_beta_w[0]),
        np.log(b_alpha_d[0]), np.log(b_beta_d[0]),
        np.log(btau_a[0]), np.log(bAs[0]), np.log(bSS[0]),
    ]
    U = [
        np.log(b_alpha_w[1]), beta_w_to_opt(b_beta_w[1]),
        np.log(b_alpha_d[1]), np.log(b_beta_d[1]),
        np.log(btau_a[1]), np.log(bAs[1]), np.log(bSS[1]),
    ]
    return list(zip(L, U))


def unpack_retinal(theta):
    t = np.asarray(theta, float)
    if t.size != D_RETINAL:
        raise ValueError(f"unpack_retinal expected {D_RETINAL}; got {t.size}")
    return {
        "alpha_w": float(np.exp(t[0])),
        "beta_w": float(beta_w_from_opt(t[1])),
        "alpha_d": float(np.exp(t[2])),
        "beta_d": float(np.exp(t[3])),
        "tau_a": float(np.exp(t[4])),
        "W_as": float(np.exp(t[5])),
        "W_ss": float(np.exp(t[6])),
    }


def pack_retinal(init_params):
    p = init_params
    lo_b, hi_b = BETA_W_NATIVE
    beta = float(np.clip(float(p["beta_w"]), lo_b, hi_b))
    return np.asarray([
        np.log(float(p["alpha_w"])),
        beta_w_to_opt(beta),
        np.log(float(p["alpha_d"])),
        np.log(float(p["beta_d"])),
        np.log(float(p["tau_a"])),
        np.log(float(p["W_as"])),
        np.log(float(p["W_ss"])),
    ], dtype=float)


def apply_retinal_to_model_params(theta):
    u = unpack_retinal(theta)
    model_params.update({
        "alpha_w": u["alpha_w"], "beta_w": u["beta_w"],
        "alpha_d": u["alpha_d"], "beta_d": u["beta_d"],
        "tau_a": u["tau_a"], "W_as": u["W_as"], "W_ss": u["W_ss"],
    })
    # Keep Stage-A gains / W anchors even if a resume JSON overwrote them.
    apply_retinal_stage_a_defaults(model_params)
    model_params.update({
        "alpha_w": u["alpha_w"], "beta_w": u["beta_w"],
        "alpha_d": u["alpha_d"], "beta_d": u["beta_d"],
        "tau_a": u["tau_a"], "W_as": u["W_as"], "W_ss": u["W_ss"],
    })
    return u


def unpack_result_retinal(theta):
    u = unpack_retinal(theta)
    return {
        "retinal": dict(u),
        "alpha_w": u["alpha_w"], "beta_w": u["beta_w"],
        "alpha_d": u["alpha_d"], "beta_d": u["beta_d"],
        "tau_a": u["tau_a"], "W_as": u["W_as"], "W_ss": u["W_ss"],
        # Placeholder keys so generic report code that expects W/g/d/θ does not KeyError.
        "W": None, "g": (0.0, 0.0), "d": (0.0, 0.0), "theta": STAGE_A_THETA,
    }


def reconstruct_theta_retinal_from_json(meta):
    """Rebuild 7-d asinh/log vector from retinal (or joint/weights) JSON.

    β_w coordinate:
      - layout=='retinal7' or beta_w_coord=='asinh' → θ[1] already asinh
      - layout in {mixed, legacy_mixed} or beta_w_coord=='native' → θ[1] native
      - otherwise treat 7-d θ as asinh (do **not** guess from |θ[1]|≤0.25)
    Missing groups fall back to Stage-A retinal anchors (not WEIGHTS_REL).
    """
    if "theta_log" in meta and len(meta["theta_log"]) == D_RETINAL:
        th = np.asarray(meta["theta_log"], float)
        layout = meta.get("layout")
        beta_coord = meta.get("beta_w_coord")
        if layout == "retinal7" or beta_coord == "asinh":
            return th
        if layout in ("mixed", "legacy_mixed") or beta_coord == "native":
            return pack_retinal({
                "alpha_w": float(np.exp(th[0])),
                "beta_w": float(th[1]),
                "alpha_d": float(np.exp(th[2])),
                "beta_d": float(np.exp(th[3])),
                "tau_a": float(np.exp(th[4])),
                "W_as": float(np.exp(th[5])),
                "W_ss": float(np.exp(th[6])),
            })
        # Ambiguous 7-d dump: assume current optimizer space (asinh).
        return th
    mp = meta.get("model_params") or {}
    ret = meta.get("retinal") or {}

    def _f(key, default):
        if key in ret:
            return float(ret[key])
        if key in mp:
            return float(mp[key])
        if key in meta:
            return float(meta[key])
        return float(default)

    return pack_retinal({
        "alpha_w": _f("alpha_w", STAGE_A_RETINAL["alpha_w"]),
        "beta_w": _f("beta_w", STAGE_A_RETINAL["beta_w"]),
        "alpha_d": _f("alpha_d", STAGE_A_RETINAL["alpha_d"]),
        "beta_d": _f("beta_d", STAGE_A_RETINAL["beta_d"]),
        "tau_a": _f("tau_a", STAGE_A_RETINAL["tau_a"]),
        "W_as": _f("W_as", STAGE_A_RETINAL["W_as"]),
        "W_ss": _f("W_ss", STAGE_A_RETINAL["W_ss"]),
    })


# ---- Legacy names (asinh primary; pack_theta_mixed / _unpack_params_mixed updated) ----
def _unpack_params_mixed(theta):
    u = unpack_retinal(theta)
    return (u["alpha_w"], u["beta_w"], u["alpha_d"], u["beta_d"],
            u["tau_a"], u["W_as"], u["W_ss"])


def pack_theta_mixed(init_params):
    return pack_retinal(init_params)


def _bounds_mixed():
    return _log_bounds_retinal()


def _nan_or_exploded(x):
    if x is None:
        return True
    if isinstance(x, dict):
        return any(_nan_or_exploded(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return any(_nan_or_exploded(v) for v in x)
    a = np.asarray(x)
    if a.dtype == object:
        try:
            a = a.astype(float)
        except Exception:
            return True
    if not np.all(np.isfinite(a)):
        return True
    if np.any(np.abs(a) > 1e6):
        return True
    return False


def _payload_retinal(theta, loss, train_mask=None, random_state=None, **extra):
    u = unpack_retinal(theta)
    theta_arr = np.asarray(theta, float)
    if train_mask is not None:
        frozen_idx = np.where(~np.asarray(train_mask, bool))[0].tolist()
    else:
        frozen_idx = []
    payload = {
        "ts": _now_iso(),
        "loss": float(loss),
        "random_state": int(random_state) if random_state is not None else None,
        "train_mask": (np.asarray(train_mask, bool).tolist() if train_mask is not None else None),
        "frozen_idx": frozen_idx,
        "theta_log": theta_arr.tolist(),
        "layout": "retinal7",
        "beta_w_coord": "asinh",
        "retinal": dict(u),
        "g": {"g_i": 0.0, "g_m": 0.0, "g_s": 0.0},
        "d": {"d_i": 0.0, "d_m": 0.0, "d_s": 0.0},
        "model_params": {
            k: float(v) if isinstance(v, (int, float, np.floating)) else v
            for k, v in model_params.items()
        },
    }
    payload.update(extra)
    return payload


def _save_params_retinal(theta_log, loss, tag="v2", random_state=None, train_mask=None, grad=None):
    if fw._RUN_DIR is None:
        _ensure_run_dirs()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    loss_str = f"{float(loss):.4g}".replace(".", "p")
    base = Path(fw._RUN_DIR) / f"retinal_{tag}_loss{loss_str}_{stamp}"
    payload = _payload_retinal(theta_log, loss, train_mask=train_mask, random_state=random_state)
    if grad is not None:
        payload["gradient"] = np.asarray(grad, float).tolist()
    with open(base.with_suffix(".json"), "w") as f:
        json.dump(payload, f, indent=2)
    np.save(base.with_suffix(".npy"), np.asarray(theta_log, float))
    print(f"[save] base={base}")


def _save_rolling_retinal(theta_log_full, loss, stage="stage2", gen=None,
                          train_mask=None, random_state=None,
                          val_loss=None, selection=None):
    if fw._RUN_DIR is None:
        _ensure_run_dirs()
    # Same rolling filename as weights/joint so resume helpers stay uniform.
    base = Path(fw._RUN_DIR) / f"weights_{stage}_last"
    payload = _payload_retinal(
        theta_log_full, loss, train_mask=train_mask, random_state=random_state,
        stage=stage,
        gen=int(gen) if gen is not None else None,
        val_loss=(float(val_loss) if val_loss is not None and np.isfinite(val_loss) else None),
        selection=selection,
    )
    tmp = base.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(base.with_suffix(".json"))
    np.save(base.with_suffix(".npy"), np.asarray(theta_log_full, float))


def loss_retinal_core(theta, mean_data_results=None, prior_regions=None, behavior=None,
                      model_type="data", plot=False, debug=False, return_details=False,
                      blocks_per_session_override=None, verbose=True,
                      stim_rng=None, stimuli_bundle=None, avg_data_R=None,
                      s_baseline=0.0, fit_mode="rms"):
    """
    L_S-only loss. Signature matches fit_weights hooks; mean_data / prior / behavior unused.
    """
    t0 = time.perf_counter()
    try:
        bps = (blocks_per_session_override
               if blocks_per_session_override is not None else blocks_per_session)
        try:
            apply_retinal_to_model_params(theta)
        except Exception:
            _exc_counters["unpack"] += 1
            if debug or _exc_counters["unpack"] <= _MAX_SHOW:
                print("EXC@unpack(retinal):", traceback.format_exc().splitlines()[-1])
            diag["sim_exc"] += 1
            diag["evals"] += 1
            diag["t_loss"] += time.perf_counter() - t0
            return 1e12

        if avg_data_R is None:
            if debug:
                print("penalty@retinal: avg_data_R is None")
            return 1e12

        try:
            bundle = stimuli_bundle if stimuli_bundle is not None else fw._STIMULI_BUNDLE_CACHE
            if bundle is not None:
                stimuli, trial_strengths, trial_sides, block_sides = bundle
            else:
                stimuli, trial_strengths, _, trial_sides, block_sides = create_stimuli(
                    bps, trials_per_block_param,
                    block_side_probs, num_stimulus_strength,
                    min_stimulus_strength, max_stimulus_strength,
                    min_trials_per_block, max_trials_per_block,
                    max_obs_per_trial, steps_before_obs,
                    rng=stim_rng, **model_params)
        except Exception:
            _exc_counters["stimuli"] += 1
            if debug or _exc_counters["stimuli"] <= _MAX_SHOW:
                print("EXC@stimuli(retinal):", traceback.format_exc().splitlines()[-1])
            diag["sim_exc"] += 1
            diag["evals"] += 1
            diag["t_loss"] += time.perf_counter() - t0
            return 1e12

        diag["sim_calls"] += 1
        try:
            t_sim0 = time.perf_counter()
            results = run_model(
                model_type, stimuli, trial_strengths, trial_sides, block_sides, bps,
                steps_before_obs=steps_before_obs, only_initial=False,
                verbose=verbose, backend="numba", **model_params,
            )
            diag["t_sim"] += time.perf_counter() - t_sim0
        except Exception:
            _exc_counters["sim"] += 1
            if debug or _exc_counters["sim"] <= _MAX_SHOW:
                print("EXC@sim(retinal):", traceback.format_exc().splitlines()[-1])
            diag["sim_exc"] += 1
            diag["evals"] += 1
            diag["t_loss"] += time.perf_counter() - t0
            return 1e12

        try:
            S_avg = mean_S_by_contrast(results, steps_before_obs)
            if _nan_or_exploded(S_avg):
                diag["sim_nan"] += 1
                diag["evals"] += 1
                diag["t_loss"] += time.perf_counter() - t0
                return 1e12
        except Exception:
            _exc_counters["avg"] += 1
            if debug or _exc_counters["avg"] <= _MAX_SHOW:
                print("EXC@avg(retinal):", traceback.format_exc().splitlines()[-1])
            diag["sim_exc"] += 1
            diag["evals"] += 1
            diag["t_loss"] += time.perf_counter() - t0
            return 1e12

        try:
            if fit_mode == "dist":
                loss_results = sse_S_dist_by_contrast(S_avg, avg_data_R)
            else:
                loss_results = compute_sse_stim_right(S_avg, avg_data_R, s_baseline)
            loss = float(loss_results["total_loss"])
        except Exception:
            _exc_counters["sse"] += 1
            if debug or _exc_counters["sse"] <= _MAX_SHOW:
                print("EXC@sse(retinal):", traceback.format_exc().splitlines()[-1])
            diag["sim_exc"] += 1
            diag["evals"] += 1
            diag["t_loss"] += time.perf_counter() - t0
            return 1e12

        diag["sim_ok"] += 1
        diag["evals"] += 1
        diag["t_loss"] += time.perf_counter() - t0
        if not np.isfinite(loss) or loss >= 1e11:
            return 1e12
        if return_details:
            return results, loss, S_avg
        return loss
    except Exception:
        _exc_counters["other"] += 1
        if debug or _exc_counters["other"] <= _MAX_SHOW:
            print("EXC@loss_retinal_core:", traceback.format_exc().splitlines()[-1])
        diag["sim_exc"] += 1
        diag["evals"] += 1
        diag["t_loss"] += time.perf_counter() - t0
        return 1e12


def loss_retinal_weight(theta_vec, avg_data_R, model_type="data", baseline=0, fit_mode="rms",
                        verbose=False, stim_rng=None):
    """Back-compat: returns (loss, S_avg)."""
    out = loss_retinal_core(
        theta_vec, avg_data_R=avg_data_R, model_type=model_type,
        s_baseline=baseline, fit_mode=fit_mode, verbose=verbose,
        stim_rng=stim_rng, return_details=True, debug=False,
    )
    if isinstance(out, tuple):
        _, loss, S_avg = out
        return loss, S_avg
    return float(out), None


def _safe_loss_retinal(theta_log, *args, **kwargs):
    kwargs.pop("verbose", None)
    v = loss_retinal_core(theta_log, *args, verbose=False, **kwargs)
    if not np.isfinite(v):
        return 1e12
    if v >= 1e11:
        return 1e11
    return float(v)


def _tracked_loss_retinal(theta_log, mean_data_results, prior_regions, behavior, debug=False,
                          model_type="data", plot=False, verbose=True, SAVE_THRESH_V2=0.85,
                          random_state=None, train_mask=None, blocks_per_session_override=None,
                          stim_rng=None, stimuli_bundle=None, avg_data_R=None, s_baseline=0.0,
                          fit_mode="rms"):
    loss = _safe_loss_retinal(
        theta_log, mean_data_results, prior_regions, behavior,
        model_type=model_type, plot=False, debug=debug,
        blocks_per_session_override=blocks_per_session_override,
        verbose=False, stim_rng=stim_rng, stimuli_bundle=stimuli_bundle,
        avg_data_R=avg_data_R, s_baseline=s_baseline, fit_mode=fit_mode,
    )
    fw._eval_counter["n"] += 1
    step = fw._eval_counter["n"]
    fw.loss_history.append(float(loss))
    if verbose:
        try:
            u = unpack_retinal(theta_log)
            _msg = (
                f"[step {step:05d}] "
                f"αw={u['alpha_w']:.4f} βw={u['beta_w']:.4f} "
                f"αd={u['alpha_d']:.4f} βd={u['beta_d']:.4f} "
                f"τa={u['tau_a']:.4f} Was={u['W_as']:.4f} Wss={u['W_ss']:.6g} "
                f"-> L_S={loss:.6f}"
            )
        except Exception:
            _msg = f"[step {step:05d}] L_S={loss:.6f}"
        print(_msg)
        if step % 10 == 0 and fw._LOG_PATH is not None:
            try:
                with open(fw._LOG_PATH, "a") as _f:
                    _f.write(_msg + "\n")
            except Exception:
                pass
        if step % 50 == 0:
            calls = max(1, diag["sim_calls"])
            print(
                f"   diag: evals={diag['evals']} sim_calls={diag['sim_calls']} "
                f"ok={diag['sim_ok']} nan={diag['sim_nan']} exc={diag['sim_exc']}  "
                f"⟨t_sim⟩={diag['t_sim']/calls:.4f}s  "
                f"⟨t_loss⟩={diag['t_loss']/max(1, diag['evals']):.4f}s"
            )
    if (np.isfinite(loss) and loss < SAVE_THRESH_V2) or (step % 1000 == 0):
        try:
            _save_params_retinal(theta_log, loss, tag="v2",
                                 random_state=random_state, train_mask=train_mask)
        except Exception as e:
            if verbose:
                print(f"[warn] save failed: {e}")
    return float(loss)


def fit_retinal_two_stage(avg_data_R, mean_data_results=None, prior_regions=None, behavior=None,
                          **kwargs):
    """
    Retinal DE→CMA→polish via fit_weights_two_stage_v2 hooks.
    mean_data_results / prior_regions / behavior are unused placeholders for the shared API.
    """
    if avg_data_R is None:
        raise ValueError("fit_retinal_two_stage requires avg_data_R")
    apply_retinal_stage_a_defaults(model_params)
    return fit_weights_two_stage_v2(
        mean_data_results, prior_regions, behavior,
        safe_loss_fn=_safe_loss_retinal,
        tracked_loss_fn=_tracked_loss_retinal,
        bounds_fn=_log_bounds_retinal,
        unpack_result_fn=unpack_result_retinal,
        save_params_fn=_save_params_retinal,
        save_rolling_fn=_save_rolling_retinal,
        freeze_fill=freeze_fill_retinal(),
        loss_extra_kwargs={
            "avg_data_R": avg_data_R,
            "s_baseline": float(kwargs.pop("s_baseline", 0.0)),
            "fit_mode": kwargs.pop("fit_mode", "rms"),
        },
        default_refine_idx=DEFAULT_REFINE_IDX,
        **kwargs,
    )


# Back-compat name used by older notes / notebooks.
fit_retinal_params_two_stage = fit_retinal_two_stage


if __name__ == "__main__":
    print(
        "fit_retinal.py: use scripts/run_fit_retinal.py "
        "(importing this module no longer starts a fit)."
    )
