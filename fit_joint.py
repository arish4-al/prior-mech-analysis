"""
Joint fit: retinal front-end + g_s/d_s + network weights/θ.

21-d optimizer vector (indices 0–11 match fit_weights so freeze masks stay familiar):

    0–5   W_ii, W_pp, W_mm, W_is, W_pi, W_mi          (log native)
    6–7   g_i, g_m                                   (log native)
    8–9   d_i, d_m                                   (log native)
    10–11 theta_c, theta_d                           (log native)
    12–13 g_s, d_s                                   (log native)
    14–20 alpha_w (log), beta_w (asinh), alpha_d,
          beta_d, tau_a, W_as, W_ss (log)            (mixed)

beta_w is signed and can be near zero, so the optimizer coordinate is
    z = asinh(beta_w / BETA_W_SCALE),  beta_w = BETA_W_SCALE * sinh(z)
Uniform / CMA sampling in z is denser near beta_w=0 — qualitatively like
log-uniform magnitude sampling on the positive params.

Loss = L_weights(I/P/M traj + I/M prior) + L_S(S rms vs avg_mean_R), one run_model.

Optimizer scaffolding is fit_weights.fit_weights_two_stage_v2 with vector_api hooks.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np

# Star-import like fit_weights so stimulus globals / helpers match that module.
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

LOG_ZERO = -30.0

D_JOINT = 21
BETA_W_IDX = 15
# Native retinal beta_w bounds (match fit_retinal); optimizer uses asinh(beta/scale).
BETA_W_NATIVE = (-0.2, 0.2)
BETA_W_SCALE = 0.05  # characteristic |beta|; denser sampling near 0 in opt space

PARAM_NAMES = [
    "W_ii", "W_pp", "W_mm", "W_is", "W_pi", "W_mi",
    "g_i", "g_m", "d_i", "d_m", "theta_c", "theta_d",
    "g_s", "d_s",
    "alpha_w", "beta_w", "alpha_d", "beta_d", "tau_a", "W_as", "W_ss",
]

# Native (positive) bounds for documentation / validation. Optimizer uses log/asinh.
NATIVE_BOUNDS = {
    "W_ii": (2e-1, 0.49), "W_pp": (0.496, 0.49999), "W_mm": (1e-1, 0.40),
    "W_is": (1e-4, 5.0), "W_pi": (1e-7, 1e-1), "W_mi": (1e-3, 10.0),
    "g_i": (1e-1, 2e2), "g_m": (1e-12, 2e2),
    "d_i": (1e-5, 1e2), "d_m": (1e-12, 1e2),
    "theta_c": (0.1, 0.99999), "theta_d": (0.1, 0.99999),
    "g_s": (1e-1, 2e2), "d_s": (1e-5, 1e2),
    "alpha_w": (1.0, 2.6), "beta_w": BETA_W_NATIVE,
    "alpha_d": (20.0, 40.0), "beta_d": (1e-2, 3.0),
    "tau_a": (100.0, 400.0), "W_as": (1.0, 50.0), "W_ss": (1e-6, 2e-1),
}

# Larger σ on prior gains (I/M/S) like fit_weights CMA_STDS for g_i/g_m.
# beta_w σ is in asinh-space (same order as other retinal log dims).
CMA_STDS = np.array([
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1,   # W
    1.0, 1.0, 0.1, 0.1, 0.1, 0.1,   # g_i,g_m,d_i,d_m,θ
    1.0, 0.1,                         # g_s, d_s
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  # retinal (beta_w in asinh space)
], dtype=float)


def beta_w_to_opt(beta_w):
    """Native beta_w → asinh optimizer coordinate."""
    return float(np.arcsinh(float(beta_w) / BETA_W_SCALE))


def beta_w_from_opt(z):
    """Asinh optimizer coordinate → native beta_w."""
    return float(BETA_W_SCALE * np.sinh(float(z)))

# Default polish: I prior + θ + sensory prior (intersected with train_mask).
DEFAULT_REFINE_IDX = [6, 8, 10, 11, 12, 13]
SENSORY_REFINE_IDX = [10, 11, 12, 13]   # θ + g_s,d_s
RETINAL_REFINE_IDX = list(range(14, 21))


def freeze_fill_joint():
    """Frozen-dim fill: LOG_ZERO for log dims; asinh(0)=0 for beta_w."""
    fill = np.full(D_JOINT, LOG_ZERO, dtype=float)
    fill[BETA_W_IDX] = beta_w_to_opt(0.0)  # 0 in asinh space
    return fill


def _bounds_retinal_opt():
    """Retinal block bounds in optimizer space (log / asinh), from NATIVE_BOUNDS."""
    b_alpha_w = NATIVE_BOUNDS["alpha_w"]
    b_beta_w = NATIVE_BOUNDS["beta_w"]
    b_alpha_d = NATIVE_BOUNDS["alpha_d"]
    b_beta_d = NATIVE_BOUNDS["beta_d"]
    btau_a = NATIVE_BOUNDS["tau_a"]
    bAs = NATIVE_BOUNDS["W_as"]
    bSS = NATIVE_BOUNDS["W_ss"]
    L = [np.log(b_alpha_w[0]), beta_w_to_opt(b_beta_w[0]),
         np.log(b_alpha_d[0]), np.log(b_beta_d[0]),
         np.log(btau_a[0]), np.log(bAs[0]), np.log(bSS[0])]
    U = [np.log(b_alpha_w[1]), beta_w_to_opt(b_beta_w[1]),
         np.log(b_alpha_d[1]), np.log(b_beta_d[1]),
         np.log(btau_a[1]), np.log(bAs[1]), np.log(bSS[1])]
    return list(zip(L, U))


def _log_bounds_joint():
    """Full 21-d optimizer bounds: weight block + g_s/d_s + retinal."""
    b_w = fw._log_bounds_weights_v2()
    # g_s like g_i (away from 0); d_s like d_i — matches NATIVE_BOUNDS
    b_gs = (np.log(NATIVE_BOUNDS["g_s"][0]), np.log(NATIVE_BOUNDS["g_s"][1]))
    b_ds = (np.log(NATIVE_BOUNDS["d_s"][0]), np.log(NATIVE_BOUNDS["d_s"][1]))
    b_ret = _bounds_retinal_opt()
    return list(b_w) + [b_gs, b_ds] + list(b_ret)


def unpack_joint(theta):
    """Return native params as a flat named tuple-like dict of scalars."""
    t = np.asarray(theta, float)
    if t.size != D_JOINT:
        raise ValueError(f"unpack_joint expected {D_JOINT} params; got {t.size}")
    W_ii, W_pp, W_mm, W_is, W_pi, W_mi = np.exp(t[0:6])
    g_i, g_m = np.exp(t[6:8])
    d_i, d_m = np.exp(t[8:10])
    theta_c, theta_d = np.exp(t[10:12])
    g_s, d_s = np.exp(t[12:14])
    alpha_w = np.exp(t[14])
    beta_w = beta_w_from_opt(t[15])
    alpha_d, beta_d, tau_a, W_as, W_ss = np.exp(t[16:21])
    return {
        "W_ii": float(W_ii), "W_pp": float(W_pp), "W_mm": float(W_mm),
        "W_is": float(W_is), "W_pi": float(W_pi), "W_mi": float(W_mi),
        "g_i": float(g_i), "g_m": float(g_m), "d_i": float(d_i), "d_m": float(d_m),
        "theta_c": float(theta_c), "theta_d": float(theta_d),
        "g_s": float(g_s), "d_s": float(d_s),
        "alpha_w": float(alpha_w), "beta_w": float(beta_w),
        "alpha_d": float(alpha_d), "beta_d": float(beta_d),
        "tau_a": float(tau_a), "W_as": float(W_as), "W_ss": float(W_ss),
    }


def pack_joint(init_params):
    """Pack native dict → 21-d optimizer vector (log / asinh)."""
    p = init_params
    return np.asarray([
        np.log(float(p["W_ii"])), np.log(float(p["W_pp"])), np.log(float(p["W_mm"])),
        np.log(float(p["W_is"])), np.log(float(p["W_pi"])), np.log(float(p["W_mi"])),
        np.log(float(p["g_i"])), np.log(float(p["g_m"])),
        np.log(float(p["d_i"])), np.log(float(p["d_m"])),
        np.log(float(p["theta_c"])), np.log(float(p["theta_d"])),
        np.log(float(p["g_s"])), np.log(float(p["d_s"])),
        np.log(float(p["alpha_w"])), beta_w_to_opt(p["beta_w"]),
        np.log(float(p["alpha_d"])), np.log(float(p["beta_d"])),
        np.log(float(p["tau_a"])), np.log(float(p["W_as"])), np.log(float(p["W_ss"])),
    ], dtype=float)


def apply_joint_to_model_params(theta):
    """Write unpacked joint params into global model_params."""
    u = unpack_joint(theta)
    model_params.update({
        "W_ii": u["W_ii"], "W_pp": u["W_pp"], "W_mm": u["W_mm"],
        "W_is": u["W_is"], "W_pi": u["W_pi"], "W_mi": u["W_mi"],
        "g_i": u["g_i"], "g_m": u["g_m"], "d_i": u["d_i"], "d_m": u["d_m"],
        "g_s": u["g_s"], "d_s": u["d_s"],
        "alpha_w": u["alpha_w"], "beta_w": u["beta_w"],
        "alpha_d": u["alpha_d"], "beta_d": u["beta_d"],
        "tau_a": u["tau_a"], "W_as": u["W_as"], "W_ss": u["W_ss"],
        "action_thresholds": {
            "concordant": {c: u["theta_c"] for c in [1.0, 0.25, 0.125, 0.0625, 0.0]},
            "discordant": {c: u["theta_d"] for c in [1.0, 0.25, 0.125, 0.0625, 0.0]},
        },
    })
    return u


def unpack_result_joint(theta):
    """Shape expected by fit_weights return sites + joint extras."""
    u = unpack_joint(theta)
    return {
        "W": (u["W_ii"], u["W_pp"], u["W_mm"], u["W_is"], u["W_pi"], u["W_mi"]),
        "g": (u["g_i"], u["g_m"]),
        "d": (u["d_i"], u["d_m"]),
        "theta": (u["theta_c"], u["theta_d"]),
        "g_s": u["g_s"],
        "d_s": u["d_s"],
        "retinal": {
            "alpha_w": u["alpha_w"], "beta_w": u["beta_w"],
            "alpha_d": u["alpha_d"], "beta_d": u["beta_d"],
            "tau_a": u["tau_a"], "W_as": u["W_as"], "W_ss": u["W_ss"],
        },
    }


def reconstruct_theta_joint_from_json(meta):
    """Rebuild 21-d vector from joint (or padded weights) JSON."""
    if "theta_log" in meta and len(meta["theta_log"]) == D_JOINT:
        return np.asarray(meta["theta_log"], float)
    # Build from groups; pad missing g_s/d_s/retinal from model_params or defaults.
    mp = meta.get("model_params") or {}
    W = meta["W"]
    g = meta.get("g") or {}
    d = meta.get("d") or {}
    th = meta.get("theta") or {}
    ret = meta.get("retinal") or {}

    def _f(group, key, mp_key=None, default=None):
        if key in group:
            return float(group[key])
        mk = mp_key or key
        if mk in mp:
            return float(mp[mk])
        if default is not None:
            return float(default)
        raise ValueError(f"Missing {key}")

    init = {
        "W_ii": _f(W, "W_ii"), "W_pp": _f(W, "W_pp"), "W_mm": _f(W, "W_mm"),
        "W_is": _f(W, "W_is"), "W_pi": _f(W, "W_pi"), "W_mi": _f(W, "W_mi"),
        "g_i": max(_f(g, "g_i", default=1e-12), 1e-12),
        "g_m": max(_f(g, "g_m", default=1e-12), 1e-12),
        "d_i": max(_f(d, "d_i", default=1e-12), 1e-12),
        "d_m": max(_f(d, "d_m", default=1e-12), 1e-12),
        "theta_c": _f(th, "theta_c", default=0.78),
        "theta_d": _f(th, "theta_d", default=0.54),
        "g_s": max(float(meta.get("g_s", g.get("g_s", mp.get("g_s", 1e-12)))), 1e-12),
        "d_s": max(float(meta.get("d_s", d.get("d_s", mp.get("d_s", 1e-12)))), 1e-12),
        "alpha_w": float(ret.get("alpha_w", mp.get("alpha_w", 1.565))),
        "beta_w": float(ret.get("beta_w", mp.get("beta_w", 0.164))),
        "alpha_d": float(ret.get("alpha_d", mp.get("alpha_d", 35.277))),
        "beta_d": float(ret.get("beta_d", mp.get("beta_d", 2.0515))),
        "tau_a": float(ret.get("tau_a", mp.get("tau_a", 222.68))),
        "W_as": float(ret.get("W_as", mp.get("W_as", 28.106))),
        "W_ss": float(ret.get("W_ss", mp.get("W_ss", 7.652e-5))),
    }
    # Clamp beta_w into native bounds before asinh packing.
    lo_b, hi_b = BETA_W_NATIVE
    init["beta_w"] = float(np.clip(init["beta_w"], lo_b, hi_b))
    return pack_joint(init)


def build_stage_b_hybrid_payload(
    weights_meta,
    retinal_meta,
    *,
    g_s=1e-12,
    d_s=1e-12,
    source_weights=None,
    source_retinal=None,
):
    """
    Stage A→B handoff: WEIGHTS_REL (or weights JSON) W/g/d/θ ∪ Stage-A retinal.

    Returns a joint21 payload loadable by ``reconstruct_theta_joint_from_json`` /
    ``--resume-json``. Variant freeze masks do **not** include retinal 14–20.
    Stage B DE holds those dims at this Stage-A retinal (``--stage1-hold-retinal``);
    Stage-2 CMA / polish unfreeze them. ``g_s``/``d_s`` default near zero; regular
    freezes them, sensory trains them from this warm start.
    """
    W = weights_meta["W"]
    g = weights_meta.get("g") or {}
    d = weights_meta.get("d") or {}
    th = weights_meta.get("theta") or {}
    ret = retinal_meta.get("retinal") or {}
    if not ret and "theta_log" in retinal_meta:
        # Stage-A retinal7 JSON without nested retinal (rare).
        import fit_retinal as fr
        ret = fr.unpack_retinal(fr.reconstruct_theta_retinal_from_json(retinal_meta))
    if not ret:
        raise ValueError("retinal_meta missing 'retinal' (Stage-A final required)")

    def _pos(x, floor=1e-12):
        return max(float(x), floor)

    init = {
        "W_ii": float(W["W_ii"]), "W_pp": float(W["W_pp"]), "W_mm": float(W["W_mm"]),
        "W_is": float(W["W_is"]), "W_pi": float(W["W_pi"]), "W_mi": float(W["W_mi"]),
        "g_i": _pos(g.get("g_i", 1e-12)),
        "g_m": _pos(g.get("g_m", 1e-12)),
        "d_i": _pos(d.get("d_i", 1e-12)),
        "d_m": _pos(d.get("d_m", 1e-12)),
        "theta_c": float(th.get("theta_c", 0.78)),
        "theta_d": float(th.get("theta_d", 0.54)),
        "g_s": _pos(g_s),
        "d_s": _pos(d_s),
        "alpha_w": float(ret["alpha_w"]),
        "beta_w": float(ret["beta_w"]),
        "alpha_d": float(ret["alpha_d"]),
        "beta_d": float(ret["beta_d"]),
        "tau_a": float(ret["tau_a"]),
        "W_as": float(ret["W_as"]),
        "W_ss": _pos(ret["W_ss"], floor=1e-12),
    }
    lo_b, hi_b = BETA_W_NATIVE
    init["beta_w"] = float(np.clip(init["beta_w"], lo_b, hi_b))
    # Keep W_ss inside native bounds for packing.
    lo_ss, hi_ss = NATIVE_BOUNDS["W_ss"]
    init["W_ss"] = float(np.clip(init["W_ss"], lo_ss, hi_ss))

    theta = pack_joint(init)
    u = unpack_joint(theta)
    mp = dict(weights_meta.get("model_params") or {})
    mp.update({k: u[k] for k in (
        "W_ii", "W_pp", "W_mm", "W_is", "W_pi", "W_mi",
        "g_i", "g_m", "d_i", "d_m", "g_s", "d_s",
        "alpha_w", "beta_w", "alpha_d", "beta_d", "tau_a", "W_as", "W_ss",
    )})
    mp["action_thresholds"] = {
        "concordant": {str(c): u["theta_c"] for c in [1.0, 0.25, 0.125, 0.0625, 0.0]},
        "discordant": {str(c): u["theta_d"] for c in [1.0, 0.25, 0.125, 0.0625, 0.0]},
    }
    # Placeholder only — not a measured L_w+L_S. Must be a finite float so
    # run_fit_joint's float(meta["loss"]) does not raise on JSON null.
    w_loss = weights_meta.get("loss")
    r_loss = retinal_meta.get("loss")
    try:
        placeholder_loss = float(w_loss) + float(r_loss)
    except (TypeError, ValueError):
        placeholder_loss = 1.0
    if not np.isfinite(placeholder_loss):
        placeholder_loss = 1.0
    return {
        "ts": _now_iso(),
        "loss": float(placeholder_loss),
        "layout": "joint21",
        "beta_w_coord": "asinh",
        "handoff": "stageA_retinal_plus_WEIGHTS_REL",
        "source_weights": source_weights,
        "source_retinal": source_retinal,
        "recorded_weights_loss": w_loss,
        "recorded_retinal_loss": r_loss,
        "train_mask": [True] * D_JOINT,
        "frozen_idx": [],
        "theta_log": np.asarray(theta, float).tolist(),
        "W": {k: u[k] for k in ("W_ii", "W_pp", "W_mm", "W_is", "W_pi", "W_mi")},
        "g": {"g_i": u["g_i"], "g_m": u["g_m"], "g_s": u["g_s"]},
        "d": {"d_i": u["d_i"], "d_m": u["d_m"], "d_s": u["d_s"]},
        "theta": {"theta_c": u["theta_c"], "theta_d": u["theta_d"]},
        "g_s": u["g_s"],
        "d_s": u["d_s"],
        "retinal": {k: u[k] for k in (
            "alpha_w", "beta_w", "alpha_d", "beta_d", "tau_a", "W_as", "W_ss")},
        "model_params": jsonify_model_params(mp),
        "note": (
            "Warm-start for Stage B joint: retinal free under regular:12|13 / "
            "sensory:6|7|8|9 (do not freeze 14–20). "
            "'loss' is a placeholder (Lw_rec + LS_rec), not joint eval."
        ),
    }


def write_stage_b_hybrid_json(weights_json, retinal_json, out_json):
    """Load WEIGHTS + Stage-A retinal JSONs, write hybrid joint21 JSON."""
    weights_path = Path(weights_json)
    retinal_path = Path(retinal_json)
    out_path = Path(out_json)
    weights_meta = json.loads(weights_path.read_text())
    retinal_meta = json.loads(retinal_path.read_text())
    payload = build_stage_b_hybrid_payload(
        weights_meta, retinal_meta,
        source_weights=str(weights_path),
        source_retinal=str(retinal_path),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    np.save(out_path.with_suffix(".npy"), np.asarray(payload["theta_log"], float))
    return out_path, payload


def _payload_joint(theta, loss, train_mask=None, random_state=None, **extra):
    u = unpack_joint(theta)
    theta_arr = np.asarray(theta, float)
    if train_mask is not None:
        frozen_idx = np.where(~np.asarray(train_mask, bool))[0].tolist()
    else:
        frozen_idx = np.where(np.isclose(theta_arr, LOG_ZERO, atol=1e-8))[0].tolist()
    payload = {
        "ts": _now_iso(),
        "loss": float(loss),
        "random_state": int(random_state) if random_state is not None else None,
        "train_mask": (np.asarray(train_mask, bool).tolist() if train_mask is not None else None),
        "frozen_idx": frozen_idx,
        "theta_log": theta_arr.tolist(),
        "layout": "joint21",
        "W": {k: u[k] for k in ("W_ii", "W_pp", "W_mm", "W_is", "W_pi", "W_mi")},
        "g": {"g_i": u["g_i"], "g_m": u["g_m"], "g_s": u["g_s"]},
        "d": {"d_i": u["d_i"], "d_m": u["d_m"], "d_s": u["d_s"]},
        "theta": {"theta_c": u["theta_c"], "theta_d": u["theta_d"]},
        "g_s": u["g_s"],
        "d_s": u["d_s"],
        "retinal": {k: u[k] for k in (
            "alpha_w", "beta_w", "alpha_d", "beta_d", "tau_a", "W_as", "W_ss")},
        "model_params": jsonify_model_params(model_params),
    }
    payload.update(extra)
    return payload


def _save_params_joint(theta_log, loss, tag="v2", random_state=None, train_mask=None, grad=None):
    if fw._RUN_DIR is None:
        _ensure_run_dirs()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    loss_str = f"{float(loss):.4g}".replace(".", "p")
    base = Path(fw._RUN_DIR) / f"weights_{tag}_loss{loss_str}_{stamp}"
    payload = _payload_joint(theta_log, loss, train_mask=train_mask, random_state=random_state)
    if grad is not None:
        payload["gradient"] = np.asarray(grad, float).tolist()
    with open(base.with_suffix(".json"), "w") as f:
        json.dump(payload, f, indent=2)
    np.save(base.with_suffix(".npy"), np.asarray(theta_log, float))
    print(f"[save] base={base}")


def _save_rolling_joint(theta_log_full, loss, stage="stage2", gen=None,
                        train_mask=None, random_state=None,
                        val_loss=None, selection=None):
    if fw._RUN_DIR is None:
        _ensure_run_dirs()
    base = Path(fw._RUN_DIR) / f"weights_{stage}_last"
    payload = _payload_joint(
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


def loss_joint_core(theta, mean_data_results, prior_regions, behavior,
                    model_type="data", plot=False, debug=False, return_details=False,
                    blocks_per_session_override=None, verbose=True,
                    stim_rng=None, stimuli_bundle=None, avg_data_R=None,
                    s_baseline=0.0, p_offset_always_on=None, iti_penalty=None):
    """
    Joint loss: one sim → L_w (I/P/M + prior) + L_S (S rms).
    avg_data_R required (passed explicitly or via loss_extra_kwargs).
    """
    try:
        # Use fw.blocks_per_session (live), not the import-time snapshot.
        # `from fit_weights import blocks_per_session` stays at 5 even after
        # run_fit_joint sets fw.blocks_per_session = --bps-stage1 (ORCD 2026-08-12f).
        bps = (blocks_per_session_override
               if blocks_per_session_override is not None else fw.blocks_per_session)
        try:
            apply_joint_to_model_params(theta)
        except Exception:
            if debug:
                import traceback
                print("EXC@unpack(joint):", traceback.format_exc().splitlines()[-1])
            return 1e12

        apply_model_ablation_flags(
            model_params,
            p_offset_always_on=p_offset_always_on,
            iti_penalty=iti_penalty,
        )

        if avg_data_R is None:
            if debug:
                print("penalty@joint: avg_data_R is None")
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
            if debug:
                import traceback
                print("EXC@stimuli(joint):", traceback.format_exc().splitlines()[-1])
            return 1e12

        try:
            results = run_model(
                model_type, stimuli, trial_strengths, trial_sides, block_sides, bps,
                steps_before_obs=steps_before_obs, gradient_mode=False, grad_options=None,
                verbose=verbose, backend="numba", **model_params,
            )
        except Exception:
            if debug:
                import traceback
                print("EXC@sim(joint):", traceback.format_exc().splitlines()[-1])
            return 1e12

        try:
            sim_out = mean_by_condition(results, steps_before_obs, T=72,
                                        var_names=("I", "P", "M"))
            S_avg = mean_S_by_contrast(results, steps_before_obs)
        except Exception:
            if debug:
                import traceback
                print("EXC@avg(joint):", traceback.format_exc().splitlines()[-1])
            return 1e12

        try:
            loss_traj = loss_plot_diff_by_condition_with_data(
                sim_out, model_params, var_names=("I", "P", "M"),
                mean_data_results=mean_data_results, plot=plot)
            loss_prior = loss_prior_effect(
                regions=prior_regions, results=results, model_params=model_params,
                steps_before_obs=steps_before_obs, T=72,
                timeframes=("act_block_duringstim", "act_block_duringchoice"),
                alpha=0.05, ptype="p_mean_c",
                label_A="integrator", label_B="move", do_plot=plot,
                scale_factors=[1, 1, 1], include_all_trials=True)
            L_w = float(loss_traj["total"] + loss_prior["total"])
        except Exception:
            if debug:
                import traceback
                print("EXC@loss_w(joint):", traceback.format_exc().splitlines()[-1])
            return 1e12

        try:
            L_S = float(compute_sse_stim_right(
                S_avg, avg_data_R, s_baseline)["total_loss"])
        except Exception:
            if debug:
                import traceback
                print("EXC@loss_S(joint):", traceback.format_exc().splitlines()[-1])
            return 1e12

        total = float(L_w + L_S)
        if not np.isfinite(total) or total >= 1e11:
            return 1e12
        if return_details:
            return results, total, L_w, L_S, loss_traj, loss_prior
        return total
    except Exception:
        if debug:
            import traceback
            print("EXC@loss_joint_core:", traceback.format_exc().splitlines()[-1])
        return 1e12


def _safe_loss_joint(theta_log, *args, **kwargs):
    verbose = kwargs.pop("verbose", True) if "verbose" in kwargs else True
    v = loss_joint_core(theta_log, *args, verbose=verbose, **kwargs)
    if not np.isfinite(v):
        return 1e12
    if v >= 1e11:
        return 1e11
    return float(v)


def _tracked_loss_joint(theta_log, mean_data_results, prior_regions, behavior, debug=False,
                        model_type="data", plot=False, verbose=True, SAVE_THRESH_V2=0.8,
                        random_state=None, train_mask=None, blocks_per_session_override=None,
                        stim_rng=None, stimuli_bundle=None, avg_data_R=None, s_baseline=0.0,
                        **loss_kw):
    loss = _safe_loss_joint(
        theta_log, mean_data_results, prior_regions, behavior,
        model_type=model_type, plot=False, debug=debug,
        blocks_per_session_override=blocks_per_session_override,
        verbose=verbose, stim_rng=stim_rng, stimuli_bundle=stimuli_bundle,
        avg_data_R=avg_data_R, s_baseline=s_baseline, **loss_kw,
    )
    fw._eval_counter["n"] += 1
    step = fw._eval_counter["n"]
    fw.loss_history.append(float(loss))
    if verbose:
        try:
            u = unpack_joint(theta_log)
            _msg = (
                f"[step {step:05d}] "
                f"W=({u['W_ii']:.3f},{u['W_pp']:.3f},{u['W_mm']:.3f},"
                f"{u['W_is']:.3f},{u['W_pi']:.3f},{u['W_mi']:.3f}) "
                f"g=({u['g_i']:.3f},{u['g_m']:.3f},{u['g_s']:.3f}) "
                f"d=({u['d_i']:.3f},{u['d_m']:.3f},{u['d_s']:.3f}) "
                f"theta=({u['theta_c']:.3f},{u['theta_d']:.3f}) "
                f"ret(αw={u['alpha_w']:.3f},βw={u['beta_w']:.3f}) "
                f"-> loss={loss:.6f}"
            )
        except Exception:
            _msg = f"[step {step:05d}] loss={loss:.6f}"
        print(_msg)
        if step % 10 == 0 and fw._LOG_PATH is not None:
            try:
                with open(fw._LOG_PATH, "a") as _f:
                    _f.write(_msg + "\n")
            except Exception:
                pass
    if (np.isfinite(loss) and loss < SAVE_THRESH_V2) or (step % 1000 == 0):
        try:
            _save_params_joint(theta_log, loss, tag="v2",
                               random_state=random_state, train_mask=train_mask)
        except Exception as e:
            if verbose:
                print(f"[warn] save failed: {e}")
    return float(loss)


def fit_joint_two_stage(mean_data_results, prior_regions, behavior, avg_data_R,
                        p_offset_always_on=False, iti_penalty=True, **kwargs):
    """
    Joint DE→CMA→polish via fit_weights_two_stage_v2 hooks.
    Requires avg_data_R (S target curves from avg_mean_R.npy).
    """
    if avg_data_R is None:
        raise ValueError("fit_joint_two_stage requires avg_data_R")
    # stage2_restim: rebuild stim from fixed seeds each eval so free α_w/β_w
    # (baked into create_stimuli) actually move during CMA/polish. Callers may
    # override via kwargs.
    kwargs.setdefault("stage2_restim", True)
    extra = dict(kwargs.pop("loss_extra_kwargs", None) or {})
    extra.setdefault("avg_data_R", avg_data_R)
    extra.setdefault("s_baseline", 0.0)
    extra["p_offset_always_on"] = bool(p_offset_always_on)
    extra["iti_penalty"] = bool(iti_penalty)
    return fit_weights_two_stage_v2(
        mean_data_results, prior_regions, behavior,
        safe_loss_fn=_safe_loss_joint,
        tracked_loss_fn=_tracked_loss_joint,
        bounds_fn=_log_bounds_joint,
        unpack_result_fn=unpack_result_joint,
        save_params_fn=_save_params_joint,
        save_rolling_fn=_save_rolling_joint,
        freeze_fill=freeze_fill_joint(),
        loss_extra_kwargs=extra,
        default_refine_idx=DEFAULT_REFINE_IDX,
        **kwargs,
    )
