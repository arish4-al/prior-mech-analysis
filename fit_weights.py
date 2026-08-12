'''
This script is used to fit the weights of the model, 2nd stage of the fitting process.
'''

import os
import time
import types
import cma
from model_functions import *

model_params['alpha_w'] = 1.565
model_params['beta_w'] = 0.164
model_params['alpha_d'] = 35.277
model_params['beta_d'] = 2.0515
model_params['tau_a'] = 222.68
model_params['W_as'] = 28.106
model_params['W_ss'] = 7.652e-05

model_params['W_ii'] = 0.375
model_params['W_mm'] = 0.139
model_params['W_pp'] = 0.49
model_params['W_is'] = 0.119
model_params['W_pi'] = 0.00107
model_params['W_mi'] = 1.471

model_params['g_i'] = 0
model_params['g_m'] = 0
model_params['g_s'] = 0
model_params['d_i'] = 0
model_params['d_m'] = 0
model_params['d_s'] = 0

theta = [0.78, 0.54]
model_params['action_thresholds']={
        'concordant': {
            1.0: theta[0],
            0.25: theta[0],
            0.125: theta[0],
            0.0625: theta[0],
            0: theta[0]
        },
        'discordant': {
            1.0: theta[1],
            0.25: theta[1],
            0.125: theta[1],
            0.0625: theta[1],
            0: theta[1]
        }
    }

# Default dt-dependent counts for legacy numpy pipeline
model_params['dt'] = float(model_params.get('dt', 2.0))
_steps_default = int(STEPS_BEFORE_OBS_DURATION_MS / model_params['dt'])
_max_obs_default = int(MAX_OBS_DURATION_MS / model_params['dt'])

steps_before_obs = globals().get('steps_before_obs', _steps_default)
max_obs_per_trial = globals().get('max_obs_per_trial', _max_obs_default)
max_steps_per_trial = globals().get('max_steps_per_trial', steps_before_obs + max_obs_per_trial)


# globals for logging + realtime plotting
loss_history = []                # stores loss trajectory
_eval_counter = {'n': 0}         # step counter
_rt_plot = {                     # live plot state
    'enabled': False,
    'every': 10,
    'fig': None, 'ax': None, 'line': None,
    'inline': True, 'handle': None,
}
diag = {                         # diagnostics counters
    'evals':0,'sim_calls':0,'sim_ok':0,'sim_nan':0,'sim_exc':0,
    't_loss':0.0,'t_sim':0.0
}

# context for Stage 1 DE CPU-parallel evaluation
_LOSS_ACTIVE_DE_CONTEXT = None

# Optional cached stimulus batch (tuple passed to run_model) for stage-2 refine
_STIMULI_BUNDLE_CACHE = None

# plotting helper functions
def enable_realtime_plot(every=10, title="Loss vs evaluation steps", inline=True):
    plt.ion()
    _rt_plot['enabled'] = True
    _rt_plot['every'] = max(1, int(every))
    _rt_plot['inline'] = bool(inline)
    _rt_plot['fig'], _rt_plot['ax'] = plt.subplots(figsize=(6, 4))
    (_rt_plot['line'],) = _rt_plot['ax'].plot([], [], lw=1.5)
    _rt_plot['ax'].set_xlabel("Evaluation step")
    _rt_plot['ax'].set_ylabel("Loss")
    _rt_plot['ax'].set_title(title)
    _rt_plot['ax'].grid(True, alpha=0.3)
    if _rt_plot['inline']:
        _rt_plot['handle'] = display(_rt_plot['fig'], display_id=True)
    else:
        plt.show(block=False)

def disable_realtime_plot():
    _rt_plot['enabled'] = False
    if not _rt_plot['inline']:
        plt.ioff(); plt.show()


# save checkpoints and restart
import json, datetime
_RUN_DIR = None
_CKPT_DIR = None
_LOG_PATH = None

def _now_iso():
    return datetime.datetime.now().isoformat(timespec='seconds')

def _ensure_run_dirs(run_dir=None):
    """
    Create a new timestamped run folder under `save_dir` OR reuse a specific run folder.

    If `run_dir` is provided, it is used directly (without touching global `save_dir`).
    Otherwise, a fresh `weights_run_*` directory is created under `save_dir`.
    """
    global _RUN_DIR, _CKPT_DIR, _LOG_PATH
    assert 'save_dir' in globals(), "Define global `save_dir` (Path or str) before running."

    if run_dir is not None:
        _RUN_DIR = Path(run_dir)
    else:
        root = Path(save_dir)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        _RUN_DIR = root / f"weights_run_{ts}"

    _CKPT_DIR = _RUN_DIR / "ckpts"
    _RUN_DIR.mkdir(parents=True, exist_ok=True)
    _CKPT_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_PATH = _RUN_DIR / "fit_log.jsonl"
    return _RUN_DIR, _CKPT_DIR, _LOG_PATH

# --- saving config ---
def _save_params_v2(theta_log, loss, tag="v2", random_state=None, train_mask=None, grad=None):
    """Save current params (both log and unpacked) into this run’s folder when loss is small.
    Optionally records gradient vector (if available) for diagnostic checks.
    """
    (W_ii, W_pp, W_mm, W_is, W_pi, W_mi,
     g_i, g_m, d_i, d_m, theta_c, theta_d) = _unpack_log_params_weights_v2(theta_log)

    if _RUN_DIR is None:
        _ensure_run_dirs()

    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    loss_str = f"{float(loss):.4g}".replace('.', 'p')
    base = Path(_RUN_DIR) / f"weights_{tag}_loss{loss_str}_{stamp}"

    # mask/frozen info
    LOG_ZERO, tol = -30.0, 1e-8
    theta_log_arr = np.asarray(theta_log, float)
    frozen_idx = np.where(np.isclose(theta_log_arr, LOG_ZERO, atol=tol))[0].tolist()
    train_mask_list = (np.asarray(train_mask, bool).tolist() if train_mask is not None else None)
    grad_list = np.asarray(grad, float).tolist() if grad is not None else None

    payload = {
        "ts": _now_iso(),
        "loss": float(loss),
        "random_state": int(random_state) if random_state is not None else None,
        "train_mask": train_mask_list,
        "frozen_idx": frozen_idx,
        "gradient": grad_list,
        "theta_log": theta_log_arr.tolist(),
        "W": {"W_ii": float(W_ii), "W_pp": float(W_pp), "W_mm": float(W_mm),
              "W_is": float(W_is), "W_pi": float(W_pi), "W_mi": float(W_mi)},
        "g": {"g_i": float(g_i), "g_m": float(g_m)},
        "d": {"d_i": float(d_i), "d_m": float(d_m)},
        "theta": {"theta_c": float(theta_c), "theta_d": float(theta_d)},
        "model_params": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in model_params.items()},
    }
    with open(base.with_suffix(".json"), "w") as f:
        json.dump(payload, f, indent=2)

    np.save(base.with_suffix(".npy"), theta_log_arr)
    print(f"[save] base={base}")


def _save_rolling_checkpoint(theta_log_full, loss, stage="stage2", gen=None,
                             train_mask=None, random_state=None,
                             val_loss=None, selection=None):
    """Overwrite a *stable* full-vector checkpoint (no timestamp) for restartability.

    Writes `weights_{stage}_last.{npy,json}` in the current run dir. Unlike
    `_save_params_v2` (timestamped, gated on small loss), this always overwrites so
    `run_fit_weights.py --resume auto` can pick up the most recent best after a crash
    or SLURM timeout mid-Stage-2.

    When held-out selection is active, callers should pass the held-out incumbent
    (selection='held_out', val_loss=...) so restart continues from the generalizing
    point rather than a train-only overfit.
    """
    if _RUN_DIR is None:
        _ensure_run_dirs()
    theta_log_arr = np.asarray(theta_log_full, float)
    (W_ii, W_pp, W_mm, W_is, W_pi, W_mi,
     g_i, g_m, d_i, d_m, theta_c, theta_d) = _unpack_log_params_weights_v2(theta_log_arr)
    LOG_ZERO, tol = -30.0, 1e-8
    frozen_idx = np.where(np.isclose(theta_log_arr, LOG_ZERO, atol=tol))[0].tolist()
    base = Path(_RUN_DIR) / f"weights_{stage}_last"
    payload = {
        "ts": _now_iso(),
        "stage": stage,
        "gen": int(gen) if gen is not None else None,
        "loss": float(loss),
        "val_loss": (float(val_loss) if val_loss is not None and np.isfinite(val_loss) else None),
        "selection": selection,
        "random_state": int(random_state) if random_state is not None else None,
        "train_mask": (np.asarray(train_mask, bool).tolist() if train_mask is not None else None),
        "frozen_idx": frozen_idx,
        "theta_log": theta_log_arr.tolist(),
        "W": {"W_ii": float(W_ii), "W_pp": float(W_pp), "W_mm": float(W_mm),
              "W_is": float(W_is), "W_pi": float(W_pi), "W_mi": float(W_mi)},
        "g": {"g_i": float(g_i), "g_m": float(g_m)},
        "d": {"d_i": float(d_i), "d_m": float(d_m)},
        "theta": {"theta_c": float(theta_c), "theta_d": float(theta_d)},
        "model_params": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in model_params.items()},
    }
    # Atomic-ish: write then replace, so a reader never sees a truncated file.
    tmp = base.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(base.with_suffix(".json"))
    np.save(base.with_suffix(".npy"), theta_log_arr)


def _save_de_result(de_result, stage="de1", tag="v2", fit_idx=None, random_state=None, algo="de"):
    """
    Save global-search stage result (x, fun, metadata) to this run’s ckpt folder; also append to JSONL log.
    Records active indices (train mask proxy) and random_state for reproducibility.
    `algo` identifies which global optimizer generated the result ("de" or "cma").
    """
    if _CKPT_DIR is None or _LOG_PATH is None:
        _ensure_run_dirs()

    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    loss_str = f"{de_result.fun:.4g}".replace('.', 'p')
    base = Path(_CKPT_DIR) / f"{tag}_{stage}_loss{loss_str}_{stamp}"

    payload = {
        "ts": _now_iso(),
        "stage": stage,
        "algo": algo,
        "loss": float(de_result.fun),
        "theta_log": np.asarray(de_result.x, float).tolist(),  # active coords only
        "fit_idx": (list(map(int, fit_idx)) if fit_idx is not None else None),
        "random_state": int(random_state) if random_state is not None else None,
        "nit": getattr(de_result, "nit", None),
        "nfev": getattr(de_result, "nfev", None),
        "message": getattr(de_result, "message", None),
        "model_params": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in model_params.items()},
    }
    with open(base.with_suffix(".json"), "w") as f:
        json.dump(payload, f, indent=2)
    np.save(base.with_suffix(".npy"), np.asarray(de_result.x, float))

    with open(_LOG_PATH, "a") as f:
        f.write(json.dumps(payload) + "\n")

    print(f"[saved] {stage} ({algo.upper()}) result: loss={de_result.fun:.6g} → {base}")


def _log_info(message, metadata=None):
    """Log an info message to fit_log.jsonl for real-time monitoring."""
    if _LOG_PATH is None:
        _ensure_run_dirs()
    
    payload = {
        "ts": _now_iso(),
        "type": "info",
        "message": message,
    }
    if metadata:
        payload.update(metadata)
    
    with open(_LOG_PATH, "a") as f:
        f.write(json.dumps(payload) + "\n")


# --- helpers to resume from saved checkpoints ---
import re, glob

def list_weight_runs(save_dir):
    """
    Return timestamped run directories created by fit_weights_two_stage_v2,
    sorted newest-first.
    """
    root = Path(save_dir)
    runs = sorted(root.glob("weights_run_*"), key=lambda p: p.name, reverse=True)
    return runs

def read_run_log(run_dir):
    """
    Read the JSONL log file for a run and return a list of dicts (in file order).
    """
    log_path = Path(run_dir) / "fit_log.jsonl"
    entries = []
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
    return entries

def _stage_ckpt_paths(run_dir, stage):
    """
    List checkpoint files (NPY/JSON) for a given stage ('de1' or 'de2'), newest-first.
    """
    ckdir = Path(run_dir) / "ckpts"
    pats = [
        str(ckdir / f"v2_{stage}_loss*_*.*"),   # our saver naming
    ]
    files = []
    for pat in pats:
        files.extend(glob.glob(pat))
    # sort by timestamp suffix in name if present; fallback to mtime
    def _key(p):
        p = Path(p)
        m = re.search(r"_(\d{8}-?\d{6})", p.name)  # matches 20251013-123456 style
        if m:
            try:
                return datetime.datetime.strptime(m.group(1), "%Y%m%d-%H%M%S").timestamp()
            except Exception:
                pass
        return p.stat().st_mtime
    return sorted(map(Path, files), key=_key, reverse=True)
    

def load_theta_from_ckpt(path):
    """
    Load a theta_log vector and the corresponding train_mask from a checkpoint.

    Returns
    -------
    theta_log_full_or_active : np.ndarray
        - If metadata contains enough information (train_mask / fit_idx / frozen_idx)
          and its size matches the number of parameters, this is a full-length
          log-parameter vector, with frozen entries set to LOG_ZERO (≈ 0 actual),
          matching the fitter's masking rule.
        - Otherwise, returns the raw stored vector (active coords only).
    train_mask : np.ndarray or None
        Boolean mask of length D_full where True indicates a trainable parameter
        in the *original* fit. None if it cannot be inferred from metadata.
    """
    p = Path(path)
    LOG_ZERO = -30.0
    meta = None
    train_mask = None

    # --- load theta and metadata ---
    if p.suffix == ".npy":
        theta_active = np.load(p)
        meta_path = p.with_suffix(".json")
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception:
                meta = None
    elif p.suffix == ".json":
        with open(p) as f:
            meta = json.load(f)
        if "theta_log" not in meta:
            raise ValueError("JSON checkpoint missing 'theta_log'.")
        theta_active = np.array(meta["theta_log"], dtype=float)
    else:
        raise ValueError("Unsupported checkpoint type (use .npy or .json).")

    # Infer D_full (12-d weights / 21-d joint / 7-d retinal). Do not assume weights-only.
    D_full = len(_log_bounds_weights_v2())
    if meta is not None:
        if meta.get("layout") == "joint21":
            D_full = 21
        elif meta.get("layout") == "retinal7":
            D_full = 7
        elif meta.get("train_mask") is not None:
            D_full = int(len(meta["train_mask"]))
        else:
            fit_idx_meta = meta.get("fit_idx", meta.get("fit_id", None))
            frozen_meta = meta.get("frozen_idx", None)
            idxs = []
            if fit_idx_meta is not None:
                idxs.extend(int(i) for i in fit_idx_meta)
            if frozen_meta is not None:
                idxs.extend(int(i) for i in frozen_meta)
            if idxs:
                D_full = max(D_full, max(idxs) + 1)
            elif theta_active.size in (7, 12, 21):
                # Full-vector dumps (no mask) — trust length.
                D_full = int(theta_active.size)

    # --- infer train_mask from metadata, if available ---
    if meta is not None:
        # 1) direct train_mask field (preferred)
        tm = meta.get("train_mask", None)
        if tm is not None:
            tm_arr = np.asarray(tm, dtype=bool)
            if tm_arr.size == D_full:
                train_mask = tm_arr

        # 2) reconstruct from fit_idx / fit_id or frozen_idx
        if train_mask is None:
            fit_idx = meta.get("fit_idx", meta.get("fit_id", None))
            frozen_idx = meta.get("frozen_idx", None)

            if fit_idx is not None:
                mask = np.zeros(D_full, dtype=bool)
                mask[np.asarray(fit_idx, dtype=int)] = True
                train_mask = mask
            elif frozen_idx is not None:
                mask = np.ones(D_full, dtype=bool)
                mask[np.asarray(frozen_idx, dtype=int)] = False
                train_mask = mask

    # --- reconstruct full vector if we know which coords were active ---
    if (train_mask is not None) and (theta_active.size == int(np.count_nonzero(train_mask))):
        theta_full = np.full(D_full, LOG_ZERO, dtype=float)
        # β_w uses asinh coords; LOG_ZERO would be wrong if frozen.
        if D_full == 21 and (not train_mask[15]):
            theta_full[15] = 0.0  # asinh(0)
        elif D_full == 7 and (not train_mask[1]):
            theta_full[1] = 0.0  # asinh(0)
        theta_full[train_mask] = theta_active
        return theta_full, train_mask

    # fallback: return as-is with whatever train_mask we could infer
    return theta_active, train_mask


def latest_theta_from_stage(run_dir, stage="de2"):
    """
    Return (theta_log, ckpt_path, train_mask) for the newest checkpoint of a stage.
    Stage ∈ {'de1','de2'}.

    theta_log is either full-length (if reconstruction was possible) or the
    stored vector; train_mask is the boolean mask inferred from the checkpoint
    (or None if it cannot be inferred).
    """
    paths = _stage_ckpt_paths(run_dir, stage)
    if not paths:
        raise FileNotFoundError(f"No checkpoints found for stage '{stage}' in {run_dir}.")
    th, train_mask = load_theta_from_ckpt(paths[0])
    return th, paths[0], train_mask

def prepare_resume_args_from_run(run_dir, stage="de2"):
    """
    Build kwargs to resume fit_weights_two_stage_v2 from a saved run.
    Example:
        kwargs = prepare_resume_args_from_run(best['run_dir'], 'de2')
        best2 = fit_weights_two_stage_v2(..., **kwargs)
    """
    stage = stage.lower()
    if stage not in {"de1","de2","local"}:
        raise ValueError("stage must be 'de1', 'de2', or 'local'")
    # for 'local', reuse the best de2 vector (common pattern)
    stage_for_ckpt = "de2" if stage == "local" else stage
    theta_log, ckpt_path = latest_theta_from_stage(run_dir, stage_for_ckpt)
    return {
        "resume_from": stage,
        "resume_theta_log": np.asarray(theta_log, float),
        "resume_path": str(ckpt_path),
    }

def resume_two_stage_from_run(mean_data_results, prior_regions, behavior,
                              run_dir, stage="de2", **fit_kwargs):
    """
    Convenience wrapper: resume fit_weights_two_stage_v2 from a prior run & stage.
    Any extra fit_kwargs override defaults (e.g., de2_maxiter, local_maxiter).
    """
    resume_kwargs = prepare_resume_args_from_run(run_dir, stage=stage)
    return fit_weights_two_stage_v2(
        mean_data_results, prior_regions, behavior,
        resume_from=resume_kwargs["resume_from"],
        resume_theta_log=resume_kwargs["resume_theta_log"],
        resume_path=resume_kwargs["resume_path"],
        **fit_kwargs
    )


# Reconstruct theta_log (12-dim) from a JSON dict that has unpacked fields
def reconstruct_theta_log_from_json(meta):
    # Expect structure like the saver we used:
    # {
    #   "loss": ...,
    #   "taus": {"tau_i":..., "tau_p":..., "tau_m":...},
    #   "W":    {"W_ii":..., "W_pp":..., "W_mm":..., "W_is":..., "W_pi":..., "W_mi":...},
    #   "g":    {"g_i":..., "g_m":...},
    #   "d":    {"d_i":..., "d_m":...},
    #   "theta":{"theta_c":..., "theta_d":...},
    #   "internal_noise": [n0, n1, n2, n3, n4]
    # }
    try:
        # tau_i = float(meta["taus"]["tau_i"])
        # tau_p = float(meta["taus"]["tau_p"])
        # tau_m = float(meta["taus"]["tau_m"])
        W_ii  = float(meta["W"]["W_ii"])
        W_pp  = float(meta["W"]["W_pp"])
        W_mm  = float(meta["W"]["W_mm"])
        W_is  = float(meta["W"]["W_is"])
        W_pi  = float(meta["W"]["W_pi"])
        W_mi  = float(meta["W"]["W_mi"])
        g_i   = float(meta["g"]["g_i"])
        g_m   = float(meta["g"]["g_m"])
        d_i   = float(meta["d"]["d_i"])
        d_m   = float(meta["d"]["d_m"])
        theta_c = float(meta["theta"]["theta_c"])
        theta_d = float(meta["theta"]["theta_d"])
        # noise = np.array(meta["internal_noise"], dtype=float).ravel()
    except KeyError as e:
        raise ValueError(f"Missing field in JSON: {e}")

    # if noise.size != 5:
    #     raise ValueError(f"internal_noise must have length 5; got {noise.size}")

    vec = np.array([
        # tau_i, tau_p, tau_m,
        W_ii, W_pp, W_mm, W_is, W_pi, W_mi,
        g_i, g_m, d_i, d_m,
        theta_c, theta_d,
        # *noise.tolist()
    ], dtype=float)

    if np.any(vec <= 0):
        bad = np.where(vec <= 0)[0]
        raise ValueError(f"All params must be > 0 for log; nonpositive at indices {bad.tolist()}")

    return np.log(vec)  # theta_log (12,)


# log space helper functions & bounds
# --- Pack/unpack & bounds for 12-dim parameter set ---
# layout: [W_ii, W_pp, W_mm, W_is, W_pi, W_mi,
#          g_i, g_m, d_i, d_m, theta_c, theta_d]

def _unpack_log_params_weights_v2(theta_log):
    t = np.asarray(theta_log, float)
    if t.size != 12:
        raise ValueError(f"_unpack_log_params_weights_v2 expected 12 log-params; got {t.size}")

    # All 12 parameters are strictly positive and stored in log-space
    W_ii, W_pp, W_mm, W_is, W_pi, W_mi, \
    g_i, g_m, d_i, d_m, theta_c, theta_d = np.exp(t)

    return (W_ii, W_pp, W_mm, W_is, W_pi, W_mi,
            g_i, g_m, d_i, d_m, theta_c, theta_d)


def pack_theta_log_weights_v2(init_params):
    v = np.array([
        # init_params['tau_i'], init_params['tau_p'], init_params['tau_m'],
        init_params['W_ii'],  init_params['W_pp'],  init_params['W_mm'],
        init_params['W_is'],  init_params['W_pi'],  init_params['W_mi'],
        init_params['g_i'],   init_params['g_m'],   
        init_params['d_i'],   init_params['d_m'],
        init_params['theta_c'], init_params['theta_d'],
        # *init_params['int_noise'],  # len 5
    ], float)
    return np.log(v)


def _log_bounds_weights_v2():
    # btau_i = (40.0,   200.0)
    # btau_p = (1000.0, 2000.0)
    # btau_m = (20.0,   200.0)

    # individual weight bounds
    bW_ii = (2e-1, 0.49)
    bW_pp = (0.496, 0.49999)
    bW_mm = (1e-1, 0.40)
    bW_is = (1e-4, 5)
    bW_pi = (1e-7, 1e-1)
    bW_mi = (1e-3, 10)

    # gains: g_i kept away from zero; g_m may be negligible (paper / ckpt ≈ 0)
    bG_i = (1e-1, 2e2)
    bG_m = (1e-12, 2e2)
    # offsets: d_i kept; d_m may be negligible
    bD_i = (1e-5, 1e2)
    bD_m = (1e-12, 1e2)

    # thresholds (set around amplitude of M neurons; allow discordant a bit higher)
    bTh_c  = (0.1, 0.99999)     # theta_c (concordant)
    bTh_d  = (0.1, 0.99999) # theta_d (discordant)

    # bN     = (1e-1, 1.0)   # noise (5 params)

    L = [
        # np.log(btau_i[0]), np.log(btau_p[0]), np.log(btau_m[0]),
        np.log(bW_ii[0]), np.log(bW_pp[0]), np.log(bW_mm[0]),
        np.log(bW_is[0]), np.log(bW_pi[0]), np.log(bW_mi[0]),
        np.log(bG_i[0]), np.log(bG_m[0]),  # g_i, g_m
        np.log(bD_i[0]), np.log(bD_m[0]),  # d_i, d_m
        np.log(bTh_c[0]), np.log(bTh_d[0]),
        # *([np.log(bN[0])] * 5)
    ]

    U = [
        # np.log(btau_i[1]), np.log(btau_p[1]), np.log(btau_m[1]),
        np.log(bW_ii[1]), np.log(bW_pp[1]), np.log(bW_mm[1]),
        np.log(bW_is[1]), np.log(bW_pi[1]), np.log(bW_mi[1]),
        np.log(bG_i[1]), np.log(bG_m[1]),   # g_i, g_m
        np.log(bD_i[1]), np.log(bD_m[1]),   # d_i, d_m
        np.log(bTh_c[1]), np.log(bTh_d[1]),
        # *([np.log(bN[1])] * 5)
    ]

    return list(zip(L, U))


def _nan_or_exploded(x):
    """
    Recursively check if input contains NaNs, infs, or excessively large values.
    Works for nested dict/list/tuple/ndarray structures.
    """
    
    if x is None:
        return True

    # Recurse through containers
    if isinstance(x, dict):
        return any(_nan_or_exploded(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return any(_nan_or_exploded(v) for v in x)

    # Base case: array-like
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


# track losses
def build_stimuli_bundle(bps, stim_rng=None, **create_kw):
    """Create and return a reusable stimulus tuple for repeated loss evaluations."""
    stimuli, trial_strengths, perceived_trial_strengths, trial_sides, block_sides = create_stimuli(
        bps, trials_per_block_param,
        block_side_probs, num_stimulus_strength,
        min_stimulus_strength, max_stimulus_strength,
        min_trials_per_block, max_trials_per_block,
        max_obs_per_trial, steps_before_obs,
        rng=stim_rng,
        **create_kw,
    )
    return stimuli, trial_strengths, trial_sides, block_sides


def loss_weights_core_v2(theta_log, mean_data_results, prior_regions, behavior,
                         model_type='data', plot=False, debug=False, return_details=False,
                         blocks_per_session_override=None, verbose=True,
                         stim_rng=None, stimuli_bundle=None):
    """
    Core loss in log-space for the v2 (12-param, taus fixed in model_params) model.
    Combines trajectory, prior-effect, and behavioral losses.
    Debug version: prints exact failure sites when returning 1e12.
    
    Args:
        blocks_per_session_override: If provided, use this instead of global blocks_per_session.
        stim_rng: Optional numpy RandomState to use for deterministic stimulus generation.
        stimuli_bundle: Optional pre-built (stimuli, trial_strengths, trial_sides, block_sides).
    """
    try:
        # Use override if provided, otherwise use global
        bps = blocks_per_session_override if blocks_per_session_override is not None else blocks_per_session
        
        # ---------- UNPACK ----------
        try:
            (W_ii, W_pp, W_mm, W_is, W_pi, W_mi,
             g_i, g_m, d_i, d_m, theta_c, theta_d) = _unpack_log_params_weights_v2(theta_log)
        except Exception:
            if debug:
                import traceback
                print("EXC@unpack(weights_v2):", traceback.format_exc().splitlines()[-1])
            return 1e12

        model_params.update({
            # taus remain fixed in model_params
            'W_ii': W_ii, 'W_pp': W_pp, 'W_mm': W_mm,
            'W_is': W_is, 'W_pi': W_pi, 'W_mi': W_mi,
            'g_i': g_i, 'g_m': g_m, 
            'd_i': d_i, 'd_m': d_m, 
            'action_thresholds': {
                'concordant': {c: theta_c for c in [1.0, 0.25, 0.125, 0.0625, 0.0]},
                'discordant': {c: theta_d for c in [1.0, 0.25, 0.125, 0.0625, 0.0]},
            },
        })

        # ---------- STIMULI ----------
        try:
            bundle = stimuli_bundle if stimuli_bundle is not None else _STIMULI_BUNDLE_CACHE
            if bundle is not None:
                stimuli, trial_strengths, trial_sides, block_sides = bundle
            else:
                stimuli, trial_strengths, perceived_trial_strengths, trial_sides, block_sides = create_stimuli(
                    bps, trials_per_block_param,
                    block_side_probs, num_stimulus_strength,
                    min_stimulus_strength, max_stimulus_strength,
                    min_trials_per_block, max_trials_per_block,
                    max_obs_per_trial, steps_before_obs,
                    rng=stim_rng,
                    **model_params)
        except Exception:
            if debug:
                import traceback
                print("EXC@stimuli(weights_v2):", traceback.format_exc().splitlines()[-1])
            return 1e12

        # ---------- SIM ----------
        try:
            results = run_model(
                model_type,
                stimuli,
                trial_strengths,
                trial_sides,
                block_sides,
                bps,
                steps_before_obs=steps_before_obs,
                gradient_mode=False,
                grad_options=None,
                verbose=verbose,
                backend='numba',  # hard-require; no silent numpy fallback (fit speedups)
                **model_params,
            )
        except Exception:
            if debug:
                import traceback
                print("EXC@sim(weights_v2):", traceback.format_exc().splitlines()[-1])
            return 1e12

        # ---------- AVG ----------
        try:
            sim_out = mean_by_condition(results, steps_before_obs, T=72,
                                        var_names=("I", "P", "M"))
        except Exception:
            if debug:
                import traceback
                print("EXC@avg(weights_v2):", traceback.format_exc().splitlines()[-1])
            return 1e12

        # ---------- LOSSES ----------
        try:
            loss_traj = loss_plot_diff_by_condition_with_data(
                sim_out, model_params, var_names=("I", "P", "M"),
                mean_data_results=mean_data_results, plot=plot)
        except Exception:
            if debug:
                import traceback
                print("EXC@loss_traj(weights_v2):", traceback.format_exc().splitlines()[-1])
            return 1e12

        try:
            loss_prior = loss_prior_effect(
                regions=prior_regions, results=results, model_params=model_params,
                steps_before_obs=steps_before_obs, T=72,
                timeframes=('act_block_duringstim', 'act_block_duringchoice'),
                alpha=0.05, ptype='p_mean_c',
                label_A='integrator', label_B='move', do_plot=plot,
                scale_factors=[1, 1, 1], include_all_trials=True)
        except Exception:
            if debug:
                import traceback
                print("EXC@loss_prior(weights_v2):", traceback.format_exc().splitlines()[-1])
            return 1e12

        # try:
        #     loss_beh = loss_perf_with_data(
        #         results, behavior,
        #         metric="correct", dt=1.0, do_plot=plot)
        # except Exception:
        #     if debug:
        #         import traceback
        #         print("EXC@loss_beh(weights_v2):", traceback.format_exc().splitlines()[-1])
        #     return 1e12

        total = float(loss_traj['total'] + loss_prior['total'])
        if not np.isfinite(total) or total >= 1e11:
            if debug:
                print(f"penalty@total(weights_v2): loss={total}")
            return 1e12

        if return_details:
            return results, total, loss_traj, loss_prior
        else:
            return total

    except Exception:
        if debug:
            import traceback
            print("EXC@loss_weights_core_v2:", traceback.format_exc().splitlines()[-1])
        return 1e12



def _safe_loss_weights_v2(theta_log, *args, **kwargs):
    verbose = kwargs.pop('verbose', True) if 'verbose' in kwargs else True
    v = loss_weights_core_v2(theta_log, *args, verbose=verbose, **kwargs)
    if not np.isfinite(v):
        return 1e12
    if v >= 1e11:
        return 1e11
    return float(v)


def _init_loss_active_de_context(ctx):
    """Pool initializer: install Stage-1 DE context in each worker process."""
    global _LOSS_ACTIVE_DE_CONTEXT
    _LOSS_ACTIVE_DE_CONTEXT = ctx


def _make_de_workers(n_jobs, ctx):
    """Build a scipy-DE ``workers`` argument that actually sees the DE context.

    ``differential_evolution(..., workers=N)`` uses ``multiprocessing.Pool`` with
    the *default* start method. On Python ≥3.8 macOS and often 3.13/ORCD that is
    ``spawn``, which re-imports the module and leaves ``_LOSS_ACTIVE_DE_CONTEXT``
    as None → ``RuntimeError: Stage 1 DE context not initialized`` (seen in ORCD
    smoke 2026-08-05). Fix: explicit ``fork`` Pool + initializer (Linux/ORCD).
    Returns ``(workers, cleanup_fn)``; always call ``cleanup_fn()`` when done.
    """
    if n_jobs is None or int(n_jobs) == 1:
        return 1, (lambda: None)
    import multiprocessing as mp
    try:
        fork_ctx = mp.get_context("fork")
    except ValueError:
        print("[Stage1 DE] fork start method unavailable; falling back to workers=1")
        return 1, (lambda: None)
    n = max(1, int(n_jobs))
    pool = fork_ctx.Pool(
        processes=n,
        initializer=_init_loss_active_de_context,
        initargs=(ctx,),
    )
    print(f"[Stage1 DE] parallel via fork Pool (n_jobs={n})")

    def _cleanup():
        try:
            pool.close()
            pool.join()
        except Exception:
            try:
                pool.terminate()
            except Exception:
                pass

    return pool.map, _cleanup


# Quiet DE worker for Stage 1 (CPU-parallel, no print/logging)
def _loss_active_de_worker(x_act):
    """
    Top-level DE worker for Stage 1 (CPU-parallel, quiet).
    Uses a module-level context set inside fit_weights_two_stage_v2.
    Supports joint fits via ctx['safe_loss_fn'] / freeze_fill / loss_extra_kwargs.
    """
    if _LOSS_ACTIVE_DE_CONTEXT is None:
        raise RuntimeError("Stage 1 DE context not initialized")
    ctx = _LOSS_ACTIVE_DE_CONTEXT

    x_act = np.asarray(x_act, float)

    theta_log0 = ctx["theta_log0"]
    idx = ctx["idx"]
    train_mask = ctx["train_mask"]
    LOG_ZERO = ctx["LOG_ZERO"]
    full_bounds = ctx["full_bounds"]
    freeze_fill = ctx.get("freeze_fill")
    safe_loss_fn = ctx.get("safe_loss_fn") or _safe_loss_weights_v2
    loss_extra = ctx.get("loss_extra_kwargs") or {}

    mean_data_results = ctx["mean_data_results"]
    prior_regions = ctx["prior_regions"]
    behavior = ctx["behavior"]
    model_type = ctx["model_type"]
    random_state = ctx["random_state"]
    blocks_per_session_override = ctx.get("blocks_per_session_override", None)

    # Reconstruct full log-parameter vector from active coordinates
    th_full = theta_log0.copy()
    th_full[idx] = x_act
    if freeze_fill is not None:
        th_full[~train_mask] = np.asarray(freeze_fill, float)[~train_mask]
    else:
        th_full[~train_mask] = LOG_ZERO

    # Clamp FREE dims only. Clamping frozen dims would pull LOG_ZERO up to the
    # lower bound (e.g. g_i → 0.1 instead of ~0), breaking --freeze semantics.
    Lb_full = np.array([L for (L, U) in full_bounds], float)
    Ub_full = np.array([U for (L, U) in full_bounds], float)
    th_full[train_mask] = np.minimum(
        Ub_full[train_mask], np.maximum(Lb_full[train_mask], th_full[train_mask]))

    # Quiet evaluation: no printing/logging, no tracked counters
    return safe_loss_fn(
        th_full,
        mean_data_results,
        prior_regions,
        behavior,
        model_type=model_type,
        plot=False,
        debug=False,
        blocks_per_session_override=blocks_per_session_override,
        verbose=False,
        stim_rng=None,
        **loss_extra,
    )


def _tracked_loss_weights_v2(theta_log, mean_data_results, prior_regions, behavior, debug=False,
                             model_type='data', plot=False, verbose=True, SAVE_THRESH_V2=0.4,
                             random_state=None, train_mask=None, blocks_per_session_override=None,
                             stim_rng=None, stimuli_bundle=None):
    """
    Logs params + loss each evaluation, updates live loss trace,
    and plots model vs data (I,P,M) when loss < SAVE_THRESH_V2 or every 100 steps.
    
    Args:
        blocks_per_session_override: If provided, use this instead of global blocks_per_session.
        verbose: If True, print loss and parameters. Set to False to disable printing
                 during parallel evaluation to avoid I/O contention.
        stim_rng: Optional numpy RandomState to use for deterministic stimulus generation.
        stimuli_bundle: Optional pre-built stimulus tuple (see build_stimuli_bundle).
    """
    loss = _safe_loss_weights_v2(theta_log, mean_data_results, prior_regions, behavior,
                                 model_type=model_type, plot=False, debug=debug,
                                 blocks_per_session_override=blocks_per_session_override,
                                 verbose=verbose,
                                 stim_rng=stim_rng,
                                 stimuli_bundle=stimuli_bundle)
    _eval_counter['n'] += 1
    step = _eval_counter['n']
    loss_history.append(float(loss))

    # --- print & log params + loss ---
    if verbose:
        try:
            (W_ii, W_pp, W_mm, W_is, W_pi, W_mi,
             g_i, g_m, d_i, d_m, theta_c, theta_d) = _unpack_log_params_weights_v2(theta_log)
            _msg = (f"[step {step:05d}] "
                    f"W=({W_ii:.3f},{W_pp:.3f},{W_mm:.3f},{W_is:.3f},{W_pi:.3f},{W_mi:.3f}) "
                    f"g=({g_i:.3f},{g_m:.3f}) "
                    f"d=({d_i:.3f},{d_m:.3f}) "
                    f"theta=({theta_c:.3f},{theta_d:.3f}) "
                    f"-> loss={loss:.6f}")
        except Exception:
            _msg = f"[step {step:05d}] loss={loss:.6f}"
        print(_msg)
        # Defer file I/O to reduce blocking in parallel execution
        # Only write to file every N steps to reduce I/O contention
        if step % 10 == 0:  # Write every 10 steps instead of every step
            try:
                with open(_LOG_PATH, "a") as _f:
                    _f.write(_msg + "\n")
            except Exception:
                pass
        
    # --- live loss plot (every _rt_plot['every']) ---
    if _rt_plot['enabled'] and (step % _rt_plot['every'] == 0):
        xs = np.arange(len(loss_history))
        _rt_plot['line'].set_data(xs, loss_history)
        _rt_plot['ax'].relim(); _rt_plot['ax'].autoscale_view()
        if _rt_plot['inline'] and _rt_plot['handle'] is not None:
            _rt_plot['handle'].update(_rt_plot['fig'])
        else:
            _rt_plot['fig'].canvas.draw()
            _rt_plot['fig'].canvas.flush_events()
            plt.pause(0.001)

    # --- save checkpoint when loss is small ---
    if (np.isfinite(loss) and loss < SAVE_THRESH_V2) or (step % 1000 == 0):
        try:
            _save_params_v2(theta_log, loss, tag="v2",
                    random_state=random_state, train_mask=train_mask)
            if verbose:
                print(f"[saved] loss={loss:.6g} → {_RUN_DIR}/weights_v2_*.json/.npy")
        except Exception as e:
            if verbose:
                print(f"[warn] save failed: {e}")

    # --- plot sim vs data when loss small or every 100 steps ---
    # if (np.isfinite(loss) and loss < SAVE_THRESH_V2) or (step % 100 == 0):
    #     try:
    #         (W_ii, W_pp, W_mm, W_is, W_pi, W_mi,
    #          g_i, g_m, d_i, d_m, theta_c, theta_d) = _unpack_log_params_weights_v2(theta_log)
    #         model_params.update({
    #             # taus remain as preset
    #             'W_ii': W_ii, 'W_pp': W_pp, 'W_mm': W_mm,
    #             'W_is': W_is, 'W_pi': W_pi, 'W_mi': W_mi,
    #             'g_i': g_i, 'g_m': g_m, 
    #             'd_i': d_i, 'd_m': d_m,
    #             'action_thresholds': {
    #                 'concordant': {c: theta_c for c in [1.0, 0.25, 0.125, 0.0625, 0.0]},
    #                 'discordant': {c: theta_d for c in [1.0, 0.25, 0.125, 0.0625, 0.0]},
    #             },
    #         })
    #         stimuli, trial_strengths, perceived_trial_strengths, trial_sides, block_sides = create_stimuli(
    #             blocks_per_session, trials_per_block_param, 
    #             block_side_probs, num_stimulus_strength,
    #             min_stimulus_strength, max_stimulus_strength, 
    #             min_trials_per_block, max_trials_per_block,
    #             max_obs_per_trial, steps_before_obs, **model_params)
    #         results = run_model(
    #             model_type,
    #             stimuli,
    #             trial_strengths,
    #             trial_sides,
    #             block_sides,
    #             blocks_per_session,
    #             steps_before_obs=steps_before_obs,
    #             gradient_mode=False,
    #             grad_options=None,
    #             **model_params,
    #         )
    #         sim_out = mean_by_condition(results, steps_before_obs, T=72, var_names=("S", "I", "P", "M"))
    #         if not _nan_or_exploded(sim_out):
    #             _ = loss_plot_diff_by_condition_with_data(
    #                 sim_out, model_params, var_names=("I", "P", "M"),
    #                 mean_data_results=mean_data_results, plot=True)
    #         _ = loss_prior_effect(
    #             regions=prior_regions, model_params=model_params,
    #             results=results, steps_before_obs=steps_before_obs, T=72,
    #             timeframes=('act_block_duringstim','act_block_duringchoice'),
    #             alpha=0.05, ptype='p_mean_c',
    #             label_A='integrator', label_B='move', do_plot=True, 
    #             scale_factors=[1, 1, 1], include_all_trials=True)
    #     except Exception:
    #         pass

    return float(loss)


# two-stage global fitting

def fit_weights_two_stage_v2(mean_data_results, prior_regions, behavior,
                             model_type='data', plot=False, random_state=0,
                             de1_maxiter=120, elite_frac=0.10, de2_maxiter=150,
                             top_k=8, local_maxiter=400, de_popsize=15, jitter_scale=0.05,
                             global_method_stage1='de', global_method_stage2=None, L_threshold=None,
                             cma_sigma_scale=0.25, cma_sigma_scale_stage2=None,
                             cma_opts_stage1=None, cma_opts_stage2=None,
                             theta_log0=None, init_params=None, sobol_count=64,
                             resume_from="none", resume_theta_log=None, resume_path=None,
                             train_mask=None, blocks_per_session_stage2=None, n_jobs=1,
                             parallel_backend='loky', deterministic_stage2=False,
                             cma_early_stop_patience=8,
                             cma_early_stop_beat_loss=0.4044,
                             local_refine_after_cma=True,
                             local_refine_idx=None,
                             local_refine_use_powell=False,
                             local_refine_maxls=100,
                             local_refine_patience=8,
                             local_refine_max_wall_s=None,
                             local_refine_method="powell",
                             local_refine_cma_sigma=0.05,
                             stage2_n_stim_seeds=3,
                             stage2_stim_aggregate="sample",
                             val_stim_seed=None,
                             stage2_restim=False,
                             # Optional vector API hooks (joint fit / alternate layouts)
                             safe_loss_fn=None,
                             tracked_loss_fn=None,
                             bounds_fn=None,
                             unpack_result_fn=None,
                             save_params_fn=None,
                             save_rolling_fn=None,
                             freeze_fill=None,
                             loss_extra_kwargs=None,
                             default_refine_idx=None,
                             de1_inf_restarts=2,
                             train_mask_stage1=None):
    """
    Two-stage optimizer with configurable global search (DE or CMA-ES) + local polish.
    Supports freezing a subset of parameters via `train_mask` (bool array or index list).
    When `train_mask` is None, behavior is identical to the unfrozen version.

    Optional hooks (all default to the 12-d weights path):
      safe_loss_fn / tracked_loss_fn / bounds_fn / unpack_result_fn /
      save_params_fn / save_rolling_fn / freeze_fill (per-dim frozen values) /
      loss_extra_kwargs (merged into every loss call) / default_refine_idx.
    Masked parameters are fixed to zero (log-space value LOG_ZERO ≈ -30).

    Args:
        de1_inf_restarts: If Stage-1 DE ends at penalty loss (>=1e11; often NaN S
                          buckets / invalid sims), re-run DE this many times with a
                          fresh uniform population (basin jump). Default 2; 0 disables.
        train_mask_stage1: Optional narrower mask for Stage-1 DE only. Dims free in
                          `train_mask` but frozen here are **held at theta_log0**
                          (not zero-filled), then unfrozen for Stage-2 CMA / polish.
                          Joint Stage B uses this to keep Stage-A retinal fixed in DE.
        global_method_stage1: 'de' (default) or 'cma'/'cmaes' to select Stage 1 global optimizer.
        global_method_stage2: Override for Stage 2; defaults to `global_method_stage1`.
        cma_sigma_scale: Fraction of (hi-lo) span used as default sigma for CMA-ES Stage 1.
        cma_sigma_scale_stage2: Fraction of (hi-lo) span for CMA-ES Stage 2. If None, uses cma_sigma_scale.
                                Typically smaller than Stage 1 for focused refinement.
        cma_opts_stage1 / cma_opts_stage2: Optional dicts merged into CMA options per stage.
        blocks_per_session_stage2: If provided, use this for Stage 2 evaluations instead of global blocks_per_session.
                                   Stage 1 always uses the global blocks_per_session.
        n_jobs: Workers for CMA candidate evals (Stage 2 and optional CMA polish).
                Default 1 (sequential). Use -1 for all cores.
        parallel_backend: Backend for joblib parallel execution. Options:
            - 'loky': Multiprocessing (default). Used for CMA Stage-2 candidate evals —
                      true multicore, ~5x faster than threading (bench: 89 s vs 369 s)
                      because the numba kernel is @njit WITHOUT nogil and holds the GIL.
            - 'threading': Threads in one process (used for CMA-polish candidate evals,
                      which close over non-picklable nested closures).
        deterministic_stage2: If True and blocks_per_session_stage2 is not None, Stage 2
                              uses fixed stim seed(s) (see stage2_stim_aggregate).
        stage2_restim: If True with deterministic_stage2, do **not** cache stimulus
                       arrays — each eval rebuilds stim via RandomState(seed) under the
                       *current* model_params. Required when α_w/β_w are free (they bake
                       into create_stimuli); joint Stage B should set this. Weights-only
                       (retinal frozen) can keep False and reuse cached bundles.
        cma_early_stop_patience: After early-stop is armed, stop CMA when best_overall has not
                                 improved for this many gens. Default 8. Set 0 to disable.
        cma_early_stop_beat_loss: Quality gate for early-stop. Plateau stopping is armed only once
                                  the beat metric < this value (default 0.4044 = known WEIGHTS_REL
                                  baseline). With held-out selection the beat metric is the *train*
                                  loss of the held-out incumbent (same scale as 0.4044); plateau
                                  itself is timed on held-out. Set None to disable the gate.
        local_refine_after_cma: If True (default), run a bounded local polish after CMA Stage 2
                                on `local_refine_idx`. Polish uses the same Stage-2 train-loss
                                protocol (`stage2_stim_aggregate` over `stage2_n_stim_seeds`
                                bundles) and the same held-out gate (`val_stim_seed`) as Stage 2.
        local_refine_idx: Full-vector indices to polish after CMA. Default None = the
                          "prior" set [6, 8, 10, 11] (g_i, d_i, theta_c, theta_d) — the
                          focused polish that closes the CMA integrator-gain overshoot
                          WITHOUT touching W. Full-12 "active" refine (pass list(range(12))
                          or the active idx) reaches slightly lower in-sample loss but
                          overfits the training bundle (held-out worse; 2026-08-04g) because
                          the extra W freedom fits noise, so it is opt-in. Always intersected
                          with train_mask.
        local_refine_use_powell: If True and method='cma', fall back to Powell when CMA polish
                                 fails to beat the start loss. Default False (Powell is already
                                 the default method).
        local_refine_maxls: Unused (kept for API compat; was L-BFGS-B maxls, now removed).
        local_refine_patience: Stop polish early after this many iters/gens with no best-loss
                               improvement (default 8; used by Powell callback + CMA polish).
                               Set 0 to disable.
        local_refine_max_wall_s: Safety cap (seconds) on total local-refine wall time,
                                 checked each iteration/generation; on exceed, stop and keep
                                 the best-so-far vector. Default None = no cap. Useful when
                                 refining many dims under a Slurm --time limit.
        local_refine_method: Polish optimizer: 'powell' (default — bounded Powell; best
                             held-out 2026-08-04f/g) or 'cma' (small-sigma CMA-ES restart
                             on the refine dims). L-BFGS removed (stalled on this surface).
        local_refine_cma_sigma: Initial sigma for method='cma', as a fraction of the median
                                refine-dim bound span (default 0.05 = a local restart).
        stage2_n_stim_seeds: Number of fixed Stage-2 (and polish) stim bundles (default 3).
                             Bundles are built once from RandomState(seed+100003+k). How they
                             enter the train loss is controlled by stage2_stim_aggregate.
        stage2_stim_aggregate: How multi-bundle train loss is formed (default 'sample'):
            - 'sample': each eval draws ONE of the K fixed bundles at random (~1× wall vs
              single-bundle; over many evals the optimizer sees all K — anti-overfit without
              the K× cost of averaging).
            - 'mean': each eval averages loss over all K bundles (~K× wall; lower variance
              per eval; kept as an opt-in for diagnostics).
        val_stim_seed: If set (int) AND deterministic Stage-2 with a bps override is active,
                       hold out a separate stimulus bundle (built like the bench eval:
                       RandomState(val_stim_seed + 100003)) that the optimizer NEVER trains
                       on. Stage-2 CMA then SELECTS the incumbent with the lowest held-out
                       loss (not lowest train loss) and EARLY-STOPS on held-out plateau, and
                       the post-CMA polish is kept only if it does not worsen held-out loss.
                       Rolling checkpoints store this held-out incumbent. Must differ from
                       random_state..random_state+stage2_n_stim_seeds-1 (a collision is
                       warned + ignored). Default None = no held-out selection.
    """
    
    if '_RUN_DIR' in globals() and (_RUN_DIR is not None):
        run_dir, ckpt_dir, log_path = _RUN_DIR, _CKPT_DIR, _LOG_PATH
    else:
        run_dir, ckpt_dir, log_path = _ensure_run_dirs()

    # Declared once for Stage-1 DE workers (must precede any assignment in this fn).
    global _LOSS_ACTIVE_DE_CONTEXT

    rng = np.random.RandomState(random_state)
    # Optional fixed seed(s) for Stage 2 stimulus generation so Stage-2 evals are stable.
    stage2_n_stim_seeds = max(1, int(stage2_n_stim_seeds or 1))
    agg = str(stage2_stim_aggregate or "sample").strip().lower()
    if agg not in {"sample", "mean"}:
        raise ValueError("stage2_stim_aggregate must be 'sample' or 'mean'")
    stage2_stim_aggregate = agg
    stage2_restim = bool(stage2_restim)
    if deterministic_stage2 and (blocks_per_session_stage2 is not None):
        stage2_stim_seed = int(random_state) + 100003  # any deterministic offset is fine
    else:
        stage2_stim_seed = None

    # Fixed Stage-2 stim identities: either cached bundles (retinal frozen) or
    # integer seeds rebuilt every eval (α_w/β_w free — joint / retinal Stage B).
    stage2_stim_seeds_list = []
    stage2_stimuli_bundles = []
    if stage2_stim_seed is not None and blocks_per_session_stage2 is not None:
        stage2_stim_seeds_list = [
            int(stage2_stim_seed + k) for k in range(stage2_n_stim_seeds)
        ]
        if not stage2_restim:
            for seed_k in stage2_stim_seeds_list:
                stage2_stimuli_bundles.append(
                    build_stimuli_bundle(
                        blocks_per_session_stage2,
                        stim_rng=np.random.RandomState(seed_k),
                        **model_params,
                    )
                )
    # Back-compat single-bundle name used by older call sites / workers
    stage2_stimuli_bundle = stage2_stimuli_bundles[0] if stage2_stimuli_bundles else None
    # Dedicated RNG for per-eval bundle/seed sampling (independent of optimizer rng draws).
    _stage2_bundle_rng = np.random.RandomState(int(random_state) + 900001)
    _stage2_has_fixed_stim = bool(stage2_stim_seeds_list)
    if stage2_n_stim_seeds > 1 and _stage2_has_fixed_stim:
        # Logical seeds match CLI --seed / --val-seed: RandomState(seed + 100003 + k).
        log_lo, log_hi = int(random_state), int(random_state) + stage2_n_stim_seeds - 1
        mode = "seed-restim" if stage2_restim else "cached bundles"
        if stage2_stim_aggregate == "sample":
            print(
                f"[Stage2] train loss samples 1 of {stage2_n_stim_seeds} fixed stim "
                f"({mode}) per eval (~1× wall; logical seeds {log_lo}..{log_hi})"
            )
        else:
            print(
                f"[Stage2] averaging loss over {stage2_n_stim_seeds} stim "
                f"({mode}; ~{stage2_n_stim_seeds}× wall; logical seeds {log_lo}..{log_hi})"
            )
    elif _stage2_has_fixed_stim and stage2_restim:
        print(
            f"[Stage2] seed-restim ON: rebuild stim each eval from "
            f"RandomState({stage2_stim_seeds_list[0]}) under current model_params "
            f"(α_w/β_w free-safe)"
        )

    # Held-out validation (never trained on). Requires deterministic Stage-2 so
    # "held-out" is a fixed seed identity. With stage2_restim, rebuild from that
    # seed each val eval (so α_w/β_w updates apply); else cache one bundle.
    val_stimuli_bundle = None
    val_stim_rng_seed = None  # RandomState seed for restim held-out
    if (
        val_stim_seed is not None
        and blocks_per_session_stage2 is not None
        and deterministic_stage2
    ):
        val_seed_int = int(val_stim_seed)
        train_seed_range = set(range(int(random_state), int(random_state) + stage2_n_stim_seeds))
        if val_seed_int in train_seed_range:
            print(
                f"[Stage2/val] WARNING: val_stim_seed={val_seed_int} collides with the "
                f"training seed range {sorted(train_seed_range)}; disabling held-out selection."
            )
        else:
            val_stim_rng_seed = int(val_seed_int + 100003)
            if stage2_restim:
                print(
                    f"[Stage2/val] held-out selection ON (seed-restim): val seed "
                    f"{val_seed_int} (RandomState {val_stim_rng_seed}); CMA selects + "
                    f"early-stops on held-out loss, polish kept only if held-out not worsened."
                )
            else:
                val_stimuli_bundle = build_stimuli_bundle(
                    blocks_per_session_stage2,
                    stim_rng=np.random.RandomState(val_stim_rng_seed),
                    **model_params,
                )
                print(
                    f"[Stage2/val] held-out selection ON: val bundle from seed "
                    f"{val_seed_int} (RandomState {val_stim_rng_seed}); CMA selects + "
                    f"early-stops on held-out loss, polish kept only if held-out not worsened."
                )
    elif val_stim_seed is not None:
        print(
            "[Stage2/val] WARNING: val_stim_seed set but deterministic_stage2 + "
            "blocks_per_session_stage2 required; disabling held-out selection."
        )
    val_held_out_on = (val_stimuli_bundle is not None) or (
        stage2_restim and val_stim_rng_seed is not None
    )

    # --- vector API (defaults = 12-d weights; joint fit passes hooks) ---
    LOG_ZERO = -30.0  # log-space value corresponding to ~0 actual
    _safe = safe_loss_fn or _safe_loss_weights_v2
    _tracked = tracked_loss_fn or _tracked_loss_weights_v2
    _bounds = bounds_fn or _log_bounds_weights_v2
    _save_p = save_params_fn or _save_params_v2
    _save_r = save_rolling_fn or _save_rolling_checkpoint
    _loss_extra = dict(loss_extra_kwargs or {})
    _freeze_fill = (None if freeze_fill is None else np.asarray(freeze_fill, float))

    def _call_safe(th, *a, **kw):
        merged = {**_loss_extra, **kw}
        return _safe(th, *a, **merged)

    def _call_tracked(th, *a, **kw):
        merged = {**_loss_extra, **kw}
        return _tracked(th, *a, **merged)

    hold_mask = None  # dims held at theta_log0 in Stage 1; set below

    def _apply_frozen(th, mask):
        th = np.asarray(th, float).copy()
        frozen = ~np.asarray(mask, bool)
        if hold_mask is not None:
            frozen = frozen & ~hold_mask
        if not np.any(frozen):
            return th
        if _freeze_fill is not None:
            th[frozen] = _freeze_fill[frozen]
        else:
            th[frozen] = LOG_ZERO
        return th

    def _unpack_result(th):
        if unpack_result_fn is not None:
            return unpack_result_fn(th)
        (W_ii, W_pp, W_mm, W_is, W_pi, W_mi,
         g_i, g_m, d_i, d_m, theta_c, theta_d) = _unpack_log_params_weights_v2(th)
        return {
            'W': (W_ii, W_pp, W_mm, W_is, W_pi, W_mi),
            'g': (g_i, g_m), 'd': (d_i, d_m),
            'theta': (theta_c, theta_d),
        }

    full_bounds = _bounds()
    D_full = len(full_bounds)
    if _freeze_fill is not None and len(_freeze_fill) != D_full:
        raise ValueError(f"freeze_fill length {len(_freeze_fill)} != D_full={D_full}")

    def _normalize_global_method(name, default="de"):
        nm = (name or default)
        if isinstance(nm, str):
            nm = nm.strip().lower()
        if nm in {None, "", "auto"}:
            nm = default
        if nm in {"cmaes"}:
            nm = "cma"
        return nm

    def _method_label(name):
        return "CMA-ES" if name == "cma" else str(name).upper()

    method_stage1 = _normalize_global_method(global_method_stage1, "de")
    method_stage2 = _normalize_global_method(global_method_stage2, method_stage1)

    supported_global = {"de", "cma"}
    if method_stage1 not in supported_global or method_stage2 not in supported_global:
        raise ValueError(f"Global methods must be in {supported_global}; "
                         f"got stage1={method_stage1!r}, stage2={method_stage2!r}.")
    if ("cma" in {method_stage1, method_stage2}) and (cma is None):
        raise ImportError("pycma is required for CMA-ES global search. Install via `pip install cma`.")

    if cma_opts_stage1 is None:
        cma_opts_stage1 = {}
    if cma_opts_stage2 is None:
        cma_opts_stage2 = {}
    if not isinstance(cma_opts_stage1, dict) or not isinstance(cma_opts_stage2, dict):
        raise TypeError("cma_opts_stage1 and cma_opts_stage2 must be dicts or None.")
    cma_opts_stage1 = dict(cma_opts_stage1)
    cma_opts_stage2 = dict(cma_opts_stage2)

    # will hold loss loaded from a resume checkpoint, if available
    resume_loss = None
    # will hold frozen indices inferred from the checkpoint (if any)
    checkpoint_frozen_idx = None
    
    # --- build a provisional mask for initialization ---
    if train_mask is None:
        init_mask = np.ones(D_full, dtype=bool)
    elif np.issubdtype(np.asarray(train_mask).dtype, np.integer):
        init_mask = np.zeros(D_full, dtype=bool)
        init_mask[np.asarray(train_mask, int)] = True
    else:
        tm_arr = np.asarray(train_mask, bool)
        if tm_arr.shape[0] != D_full:
            raise ValueError(f"train_mask has length {tm_arr.shape[0]} but expected {D_full}.")
        init_mask = tm_arr

    # --- initial vector (full) ---
    if theta_log0 is None and init_params is not None:
        # Prefer pack via bounds length: joint callers pass theta_log0 directly.
        if D_full == 12:
            theta_log0 = pack_theta_log_weights_v2(init_params)
        else:
            raise ValueError("init_params packing for non-12-d layouts requires theta_log0")
    elif theta_log0 is None and init_params is None:
        # only print the random-init message when not resuming from a checkpoint
        if (resume_from is None) or (str(resume_from).lower() == "none"):
            print("[Init] No init_params or theta_log0 provided — initializing free params within bounds; "
                  "masked params fixed to ~0 actual.")
        if _freeze_fill is not None:
            theta_log0 = _freeze_fill.copy()
        else:
            theta_log0 = np.full(D_full, LOG_ZERO, dtype=float)
        if init_mask.any():
            Lb_free = np.array([full_bounds[i][0] for i in range(D_full) if init_mask[i]], float)
            Ub_free = np.array([full_bounds[i][1] for i in range(D_full) if init_mask[i]], float)
            theta_log0[init_mask] = rng.uniform(Lb_free, Ub_free)
    else:
        theta_log0 = np.asarray(theta_log0, float)

    # clamp to bounds for free params only; keep masked fixed at freeze fill / LOG_ZERO
    Lb_full = np.array([L for (L, U) in full_bounds], float)
    Ub_full = np.array([U for (L, U) in full_bounds], float)
    theta_log0[init_mask] = np.minimum(Ub_full[init_mask], np.maximum(Lb_full[init_mask], theta_log0[init_mask]))
    theta_log0 = _apply_frozen(theta_log0, init_mask)

    # --- build train mask (active parameters) ---
    if train_mask is None:
        train_mask = np.ones(D_full, dtype=bool)
    elif np.issubdtype(np.asarray(train_mask).dtype, np.integer):
        m = np.zeros(D_full, dtype=bool)
        m[np.asarray(train_mask, int)] = True
        train_mask = m
    else:
        train_mask = np.asarray(train_mask, bool)
        if train_mask.shape[0] != D_full:
            raise ValueError(f"train_mask has length {train_mask.shape[0]} but expected {D_full}.")

    idx = np.where(train_mask)[0]
    if idx.size == 0:
        loss = _call_tracked(theta_log0, mean_data_results, prior_regions, behavior,
                             model_type=model_type, plot=plot, stim_rng=None)
        out = _unpack_result(theta_log0)
        out.update({
            'theta_log': theta_log0.copy(), 'loss': float(loss),
            'bounds_stage1': full_bounds, 'bounds_stage2': [full_bounds[i] for i in idx],
            'fit_idx': idx,
        })
        return out

    bnds_act = [full_bounds[i] for i in idx]
    Lb_act = np.array([L for (L, U) in bnds_act], float)
    Ub_act = np.array([U for (L, U) in bnds_act], float)

    # Optional Stage-1-only hold: keep theta_log0 on extra frozen dims (e.g. retinal),
    # then unfreeze them for Stage-2 CMA / polish. Ablation freezes in `train_mask`
    # stay zero-filled for both stages.
    mask_s2 = train_mask.copy()
    idx_s2 = idx.copy()
    bnds_act_s2 = list(bnds_act)
    _use_s1_hold = False
    if train_mask_stage1 is not None and method_stage1 == "de" and str(resume_from).lower() not in {"de2", "local"}:
        tm1 = np.asarray(train_mask_stage1)
        if tm1.dtype == bool or tm1.dtype == np.bool_:
            if tm1.shape[0] != D_full:
                raise ValueError(
                    f"train_mask_stage1 has length {tm1.shape[0]} but expected {D_full}.")
            mask_s1 = tm1.astype(bool)
        elif np.issubdtype(tm1.dtype, np.integer):
            mask_s1 = np.zeros(D_full, dtype=bool)
            mask_s1[np.asarray(tm1, int)] = True
        else:
            mask_s1 = np.asarray(train_mask_stage1, bool)
            if mask_s1.shape[0] != D_full:
                raise ValueError(
                    f"train_mask_stage1 has length {mask_s1.shape[0]} but expected {D_full}.")
        if np.any(mask_s1 & ~mask_s2):
            raise ValueError("train_mask_stage1 cannot unfreeze dims that train_mask freezes")
        hold_mask = mask_s2 & ~mask_s1
        if hold_mask.any():
            _use_s1_hold = True
            train_mask = mask_s1
            idx = np.where(train_mask)[0]
            bnds_act = [full_bounds[i] for i in idx]
            Lb_act = np.array([L for (L, U) in bnds_act], float)
            Ub_act = np.array([U for (L, U) in bnds_act], float)
            print(
                f"[Stage1] holding {int(hold_mask.sum())} dims at theta_log0 "
                f"(unfreeze for Stage 2): {np.where(hold_mask)[0].tolist()}"
            )

    def _de_context_freeze_fill():
        """Ablation fill, with Stage-1 hold dims taken from current theta_log0."""
        if _freeze_fill is None and not _use_s1_hold:
            return None
        fill = (_freeze_fill.copy() if _freeze_fill is not None
                else np.full(D_full, LOG_ZERO, dtype=float))
        if _use_s1_hold and hold_mask is not None:
            fill[hold_mask] = np.asarray(theta_log0, float)[hold_mask]
        return fill

    def full_from_active(x_act):
        th = theta_log0.copy()
        th[idx] = x_act
        return _apply_frozen(th, train_mask)

    def loss_active(x_act, verbose=None):
        """
        Loss function for active parameters.
        
        Args:
            verbose: If None, uses default (True). Set to False to disable printing
                     during parallel evaluation to avoid I/O contention.
        """
        th_full = full_from_active(x_act)
        # Use verbose parameter if provided, otherwise default to True for backward compatibility
        verbose_val = verbose if verbose is not None else True
        return _call_tracked(th_full, mean_data_results, prior_regions, behavior,
                             model_type=model_type, plot=False, verbose=verbose_val,
                             random_state=random_state, train_mask=train_mask,
                             stim_rng=None)
    
    # Stage 2 loss wrapper with blocks_per_session override and optional deterministic seeding
    def loss_active_stage2(x_act, verbose=None, bundle_idx=None):
        """
        Loss function for Stage 2 with blocks_per_session override.
        When deterministic_stage2 is True:
          - stage2_restim=False: reuse prebuilt stim bundle(s)
          - stage2_restim=True: rebuild stim each eval from fixed seed(s) (α_w/β_w free-safe)
          - aggregate='sample' (default): one randomly chosen identity per eval (~1× wall)
          - aggregate='mean': mean over all identities (~K× wall)
        Pass bundle_idx to force a specific identity (used by parallel CMA workers).
        
        Args:
            verbose: If None, uses default (True). Set to False to disable printing
                     during parallel evaluation to avoid I/O contention.
        """
        th_full = full_from_active(x_act)
        verbose_val = verbose if verbose is not None else True

        def _one_bundle(bundle):
            return float(_call_tracked(
                th_full, mean_data_results, prior_regions, behavior,
                model_type=model_type, plot=False, verbose=verbose_val,
                random_state=random_state, train_mask=train_mask,
                blocks_per_session_override=blocks_per_session_stage2,
                stim_rng=None,
                stimuli_bundle=bundle,
            ))

        def _one_seed(seed):
            return float(_call_tracked(
                th_full, mean_data_results, prior_regions, behavior,
                model_type=model_type, plot=False, verbose=verbose_val,
                random_state=random_state, train_mask=train_mask,
                blocks_per_session_override=blocks_per_session_stage2,
                stim_rng=np.random.RandomState(int(seed)),
                stimuli_bundle=None,
            ))

        if stage2_restim and stage2_stim_seeds_list:
            n_b = len(stage2_stim_seeds_list)
            if n_b == 1 or stage2_stim_aggregate == "mean":
                return float(np.mean([_one_seed(s) for s in stage2_stim_seeds_list]))
            if bundle_idx is not None:
                return _one_seed(stage2_stim_seeds_list[int(bundle_idx) % n_b])
            return _one_seed(
                stage2_stim_seeds_list[int(_stage2_bundle_rng.randint(0, n_b))]
            )
        if not stage2_stimuli_bundles:
            return _one_bundle(None)
        n_b = len(stage2_stimuli_bundles)
        if n_b == 1 or stage2_stim_aggregate == "mean":
            return float(np.mean([_one_bundle(b) for b in stage2_stimuli_bundles]))
        if bundle_idx is not None:
            return _one_bundle(stage2_stimuli_bundles[int(bundle_idx) % n_b])
        return _one_bundle(stage2_stimuli_bundles[int(_stage2_bundle_rng.randint(0, n_b))])

    def _val_loss_active(x_act):
        """Held-out loss for active-space vector `x_act` (never trained on).
        Runs in the main process only (called on CMA improvements + around polish)."""
        if not val_held_out_on:
            return np.inf
        th_full = full_from_active(x_act)
        if stage2_restim and val_stim_rng_seed is not None:
            return float(_call_safe(
                th_full, mean_data_results, prior_regions, behavior,
                model_type=model_type, plot=False, debug=False,
                blocks_per_session_override=blocks_per_session_stage2,
                verbose=False,
                stim_rng=np.random.RandomState(int(val_stim_rng_seed)),
                stimuli_bundle=None,
            ))
        return float(_call_safe(
            th_full, mean_data_results, prior_regions, behavior,
            model_type=model_type, plot=False, debug=False,
            blocks_per_session_override=blocks_per_session_stage2,
            verbose=False, stimuli_bundle=val_stimuli_bundle))

    def _make_init_population(bounds, popsize, rng, x0=None, jitter=0.05):
        L = np.array([lo for lo, hi in bounds])
        U = np.array([hi for lo, hi in bounds])
        n = popsize * len(bounds)
        pop = rng.uniform(L, U, size=(n, len(bounds)))
        if x0 is not None:
            k = min(len(bounds)*4, n//2)
            inj = np.clip(x0 + rng.normal(scale=jitter, size=(k, len(bounds))), L, U)
            pop[:k, :] = inj
            pop[0] = np.clip(np.asarray(x0, float), L, U)
        return pop

    def _shrink_bounds(elites, bounds, pad=0.20):
        E = np.vstack(elites)
        lo = E.min(axis=0)
        hi = E.max(axis=0)
        span = np.maximum(hi-lo, 1e-6)
        Lb = np.array([L for (L, U) in bounds])
        Ub = np.array([U for (L, U) in bounds])
        newL = np.maximum(Lb, lo - pad*span)
        newU = np.minimum(Ub, hi + pad*span)
        return list(zip(newL.tolist(), newU.tolist()))

    def _box_around(vec, bounds, pad=0.10):
        v = np.asarray(vec, float)
        Lb = np.array([L for (L, U) in bounds])
        Ub = np.array([U for (L, U) in bounds])
        span = Ub - Lb
        lo = np.maximum(Lb, v - pad*span)
        hi = np.minimum(Ub, v + pad*span)
        return list(zip(lo.tolist(), hi.tolist()))

    def _run_cma_es(bounds, x0=None, maxiter=200, opts_extra=None, sigma_scale=None, loss_func=None, n_jobs=1, parallel_backend='loky',
                    early_stop_patience=None, early_stop_beat_loss=None, checkpoint_stage=None,
                    val_eval=None):
        if cma is None:
            raise ImportError("pycma is required for CMA-ES global search.")

        # Use provided loss function or default to loss_active
        eval_func = loss_func if loss_func is not None else loss_active
        
        # Capture variables needed for parallel-safe evaluation
        # These will be pickled and sent to worker processes
        eval_mean_data = mean_data_results
        eval_prior_regions = prior_regions
        eval_behavior = behavior
        eval_model_type = model_type
        eval_blocks_override = blocks_per_session_stage2 if loss_func == loss_active_stage2 else None
        # Capture Stage-2 stim identities for parallel workers (bundles or restim seeds).
        _stage2_eval = (loss_func == loss_active_stage2)
        eval_stim_restim = bool(stage2_restim and _stage2_eval and stage2_stim_seeds_list)
        eval_stim_seeds = (
            list(stage2_stim_seeds_list) if eval_stim_restim else None
        )
        eval_stim_bundles = (
            list(stage2_stimuli_bundles)
            if (_stage2_eval and stage2_stimuli_bundles and not eval_stim_restim)
            else None
        )
        eval_stim_aggregate = (
            stage2_stim_aggregate
            if (eval_stim_bundles or eval_stim_seeds)
            else "mean"
        )
        
        # Capture variables needed for full_from_active function
        # These are needed because full_from_active is a closure that references local variables
        eval_theta_log0 = theta_log0.copy()  # Make a copy to avoid sharing mutable state
        eval_idx = idx.copy() if hasattr(idx, 'copy') else idx
        eval_train_mask = train_mask.copy() if hasattr(train_mask, 'copy') else train_mask
        eval_log_zero = LOG_ZERO  # Capture LOG_ZERO constant
        eval_freeze_fill = (None if _freeze_fill is None else _freeze_fill.copy())
        eval_safe = _safe
        eval_loss_extra = dict(_loss_extra)

        # Patience / quality gate: prefer explicit args, else opts_extra, else outer defaults.
        opts_extra = dict(opts_extra) if opts_extra else {}
        if early_stop_patience is None and 'early_stop_patience' in opts_extra:
            early_stop_patience = opts_extra.pop('early_stop_patience')
        elif 'early_stop_patience' in opts_extra:
            opts_extra.pop('early_stop_patience')
        if early_stop_beat_loss is None and 'early_stop_beat_loss' in opts_extra:
            early_stop_beat_loss = opts_extra.pop('early_stop_beat_loss')
        elif 'early_stop_beat_loss' in opts_extra:
            opts_extra.pop('early_stop_beat_loss')
        if early_stop_patience is None:
            early_stop_patience = cma_early_stop_patience
        if early_stop_beat_loss is None:
            early_stop_beat_loss = cma_early_stop_beat_loss
        if early_stop_patience is not None:
            early_stop_patience = int(early_stop_patience)
            if early_stop_patience <= 0:
                early_stop_patience = None
        if early_stop_beat_loss is not None:
            early_stop_beat_loss = float(early_stop_beat_loss)
            if not np.isfinite(early_stop_beat_loss):
                early_stop_beat_loss = None
        
        # Set up parallelization if requested
        use_parallel = (n_jobs != 1) and (n_jobs is not None)
        if use_parallel:
            try:
                from joblib import Parallel, delayed
                import os
                if n_jobs == -1:
                    # Check environment variable first (set by shell script)
                    joblib_n_jobs = os.environ.get('JOBLIB_N_JOBS')
                    if joblib_n_jobs:
                        n_jobs_actual = int(joblib_n_jobs)
                        msg = f"[CMA-ES] Using JOBLIB_N_JOBS from environment: {n_jobs_actual} cores"
                        print(msg)
                        _log_info(msg, {"n_jobs": n_jobs_actual, "source": "JOBLIB_N_JOBS"})
                    else:
                        cpu_count = os.cpu_count() or 1
                        n_jobs_actual = cpu_count  # Use all cores explicitly
                        msg = f"[CMA-ES] n_jobs=-1: Using all available cores ({cpu_count})"
                        print(msg)
                        _log_info(msg, {"n_jobs": cpu_count, "source": "auto_detect"})
                else:
                    n_jobs_actual = max(1, int(n_jobs))
                    msg = f"[CMA-ES] Using specified n_jobs: {n_jobs_actual} cores"
                    print(msg)
                    _log_info(msg, {"n_jobs": n_jobs_actual, "source": "parameter"})
            except ImportError:
                print("[Warning] joblib not available, falling back to sequential evaluation")
                use_parallel = False
                n_jobs_actual = 1
        else:
            n_jobs_actual = 1
            msg = f"[CMA-ES] Sequential evaluation (n_jobs={n_jobs})"
            print(msg)
            _log_info(msg, {"n_jobs": 1, "source": "sequential"})

        sigma_scale = cma_sigma_scale if sigma_scale is None else sigma_scale
        bounds = list(bounds)
        if not bounds:
            raise ValueError("CMA-ES cannot run with zero active dimensions.")

        Lb = np.array([lo for lo, _ in bounds], float)
        Ub = np.array([hi for _, hi in bounds], float)
        span = np.maximum(Ub - Lb, 1e-12)

        if x0 is None:
            x0 = rng.uniform(Lb, Ub)
        x0 = np.asarray(x0, float)
        x0 = np.minimum(Ub, np.maximum(Lb, x0))

        sigma0 = float(sigma_scale) * float(np.median(span))
        if not np.isfinite(sigma0) or sigma0 <= 0:
            sigma0 = max(1e-3, float(np.mean(span)))
        if not np.isfinite(sigma0) or sigma0 <= 0:
            sigma0 = 1e-3

        opts = {
            'bounds': [Lb.tolist(), Ub.tolist()],
            'maxiter': int(maxiter),
            'seed': int(random_state),
            'verb_disp': 0,
        }
        if opts_extra:
            for k, v in opts_extra.items():
                if v is None or k == 'bounds':
                    continue
                opts[k] = v
        opts.setdefault('popsize', max(4, 4 + int(3 * np.log(len(bounds) + 1))))

        # Check initial point - if it gives infinite loss, try to find a better starting point
        best_x = np.array(x0, float)
        best_eval = float(eval_func(best_x))
        if not np.isfinite(best_eval) or best_eval >= 1e11:
            print(f"[CMA-ES] Initial point has penalty loss ({best_eval:.2e}), searching for valid starting point...")
            # Try random points within bounds to find a valid starting point
            max_attempts = 20
            for attempt in range(max_attempts):
                x_try = rng.uniform(Lb, Ub)
                val_try = float(eval_func(x_try))
                if np.isfinite(val_try) and val_try < 1e11:
                    best_x = x_try
                    best_eval = val_try
                    print(f"[CMA-ES] Found valid starting point after {attempt+1} attempts, loss={val_try:.6g}")
                    break
            if not np.isfinite(best_eval) or best_eval >= 1e11:
                print(f"[CMA-ES] Warning: Could not find valid starting point, proceeding with initial guess")
                best_eval = np.inf
                best_x = x0.copy()
        
        if np.isfinite(best_eval):
            best_f = best_eval
        else:
            best_f = np.inf
            best_x = None
        
        # Use the best starting point found
        x0_final = best_x if best_x is not None else x0
        es = cma.CMAEvolutionStrategy(x0_final.tolist(), sigma0, opts)
        
        # Held-out validation selection/early-stop trackers. When val_eval is provided we
        # select the incumbent with the lowest HELD-OUT loss (not train) and time the
        # plateau early-stop off held-out, so the returned point is the one that generalizes.
        use_val = val_eval is not None
        best_val_f = np.inf
        best_val_x = None
        best_val_train = np.inf
        val_plateau = 0
        if use_val and best_x is not None and np.isfinite(best_f):
            try:
                v0 = float(val_eval(best_x))
            except Exception:
                v0 = np.inf
            if np.isfinite(v0):
                best_val_f, best_val_x, best_val_train = v0, best_x.copy(), float(best_f)
                print(f"[CMA-ES/val] start: train={float(best_f):.6f} held-out={v0:.6f}")

        def _ckpt_incumbent(gen_tag, x_act, train_loss, vloss=None, sel=None):
            """Write rolling Stage-2 checkpoint for the current selection incumbent."""
            if checkpoint_stage is None or x_act is None or not np.isfinite(train_loss):
                return
            try:
                _save_r(
                    full_from_active(x_act), float(train_loss),
                    stage=checkpoint_stage, gen=int(gen_tag) if gen_tag is not None else None,
                    train_mask=train_mask, random_state=random_state,
                    val_loss=vloss, selection=sel,
                )
            except Exception:
                pass

        # Restart hole fix: persist the Stage-2 entry point immediately (before any
        # generation improves), so a kill during early CMA can warm-restart.
        if use_val and best_val_x is not None:
            _ckpt_incumbent(-1, best_val_x, best_val_train, best_val_f, "held_out")
        elif best_x is not None and np.isfinite(best_f):
            _ckpt_incumbent(-1, best_x, best_f, None, "train")

        # Track consecutive generations with all infinite losses
        consecutive_inf_gens = 0
        max_inf_gens = 5  # If 5 consecutive generations all give infinite loss, increase sigma
        
        # Track convergence: no improvement for N generations (sigma adaptation)
        no_improvement_count = 0
        no_improvement_threshold = 20  # If no improvement for 20 generations, reduce sigma for finer search
        last_improvement_gen = 0
        gen_count = 0
        # Separate plateau counter for early-stop (not reset by sigma adaptation)
        plateau_count = 0
        early_stopped = False
        early_stop_armed = False
        if early_stop_patience is not None:
            if early_stop_beat_loss is not None:
                beat_note = (
                    "train-of-held-out-incumbent" if use_val else "best_overall"
                )
                msg_pat = (
                    f"[CMA-ES] early_stop_patience={early_stop_patience} "
                    f"(armed only after {beat_note} < {early_stop_beat_loss:.6f})"
                )
            else:
                msg_pat = (
                    f"[CMA-ES] early_stop_patience={early_stop_patience} "
                    f"(no beat-loss gate — can freeze a worse plateau)"
                )
            print(msg_pat)
            try:
                _log_info(msg_pat, {
                    "early_stop_patience": int(early_stop_patience),
                    "early_stop_beat_loss": (
                        float(early_stop_beat_loss)
                        if early_stop_beat_loss is not None else None
                    ),
                })
            except Exception:
                pass

        while not es.stop():
            samples = es.ask()
            xs = []
            for cand in samples:
                cand_arr = np.asarray(cand, float)
                cand_arr = np.minimum(Ub, np.maximum(Lb, cand_arr))
                xs.append(cand_arr)

            # Per-candidate stim-identity indices for sample aggregate (assigned in
            # main so loky workers stay deterministic / picklable).
            n_stim_ids = (
                len(eval_stim_seeds) if eval_stim_seeds
                else (len(eval_stim_bundles) if eval_stim_bundles else 0)
            )
            if n_stim_ids > 1 and eval_stim_aggregate == "sample":
                bundle_idxs = [
                    int(_stage2_bundle_rng.randint(0, n_stim_ids)) for _ in xs
                ]
            else:
                bundle_idxs = [None] * len(xs)

            # Evaluate candidates in parallel or sequentially
            if use_parallel:
                def eval_single(x, bidx=None):
                    try:
                        th_full = eval_theta_log0.copy()
                        th_full[eval_idx] = x
                        if eval_freeze_fill is not None:
                            th_full[~eval_train_mask] = eval_freeze_fill[~eval_train_mask]
                        else:
                            th_full[~eval_train_mask] = eval_log_zero
                        _kw = dict(
                            model_type=eval_model_type, plot=False, debug=False,
                            blocks_per_session_override=eval_blocks_override,
                            verbose=False, **eval_loss_extra,
                        )
                        if eval_stim_seeds:
                            if eval_stim_aggregate == "mean" or bidx is None:
                                vals_b = [
                                    float(eval_safe(
                                        th_full, eval_mean_data, eval_prior_regions, eval_behavior,
                                        stim_rng=np.random.RandomState(int(seed)),
                                        stimuli_bundle=None, **_kw,
                                    ))
                                    for seed in eval_stim_seeds
                                ]
                                val = float(np.mean(vals_b))
                            else:
                                seed = eval_stim_seeds[int(bidx) % len(eval_stim_seeds)]
                                val = float(eval_safe(
                                    th_full, eval_mean_data, eval_prior_regions, eval_behavior,
                                    stim_rng=np.random.RandomState(int(seed)),
                                    stimuli_bundle=None, **_kw,
                                ))
                        elif eval_stim_bundles:
                            if eval_stim_aggregate == "mean" or bidx is None:
                                vals_b = [
                                    float(eval_safe(
                                        th_full, eval_mean_data, eval_prior_regions, eval_behavior,
                                        stimuli_bundle=bundle, **_kw,
                                    ))
                                    for bundle in eval_stim_bundles
                                ]
                                val = float(np.mean(vals_b))
                            else:
                                bundle = eval_stim_bundles[int(bidx) % len(eval_stim_bundles)]
                                val = float(eval_safe(
                                    th_full, eval_mean_data, eval_prior_regions, eval_behavior,
                                    stimuli_bundle=bundle, **_kw,
                                ))
                        else:
                            val = float(eval_safe(
                                th_full, eval_mean_data, eval_prior_regions, eval_behavior,
                                stimuli_bundle=(stage2_stimuli_bundle
                                                if eval_blocks_override is not None else None),
                                **_kw,
                            ))
                    except Exception as e:
                        try:
                            val = float(eval_func(x, verbose=False))
                        except TypeError:
                            val = float(eval_func(x))
                    return val if np.isfinite(val) else np.inf
                # Backend note (bench 2026-08-03l): loky beats threading ~5x here
                # (7 vs 36 s/gen at bps=20, n_jobs=8) because the numba kernel does
                # not release the GIL, so threads serialize CMA candidate evals.
                # Keep parallel_backend='loky' (default) for Stage-2 CMA.
                vals = Parallel(n_jobs=n_jobs_actual, backend=parallel_backend, batch_size='auto', verbose=0)(
                    delayed(eval_single)(x, bidx) for x, bidx in zip(xs, bundle_idxs)
                )
            else:
                vals = []
                for cand_arr, bidx in zip(xs, bundle_idxs):
                    try:
                        val = float(eval_func(cand_arr, bundle_idx=bidx))
                    except TypeError:
                        val = float(eval_func(cand_arr))
                    if not np.isfinite(val):
                        val = np.inf
                    vals.append(val)

            # Check if all values are infinite
            all_inf = all(not np.isfinite(v) or v >= 1e11 for v in vals)
            if all_inf:
                consecutive_inf_gens += 1
                if consecutive_inf_gens >= max_inf_gens:
                    current_sigma = es.sigma
                    new_sigma = min(current_sigma * 2.0, float(np.median(span)))
                    print(f"[CMA-ES] All samples infinite for {consecutive_inf_gens} generations, increasing sigma from {current_sigma:.4f} to {new_sigma:.4f}")
                    es.sigma = new_sigma
                    consecutive_inf_gens = 0
            else:
                consecutive_inf_gens = 0

            # Update best and track improvements
            improved = False
            for i, val in enumerate(vals):
                if val < best_f:
                    best_f = val
                    best_x = xs[i].copy()
                    improved = True
                    last_improvement_gen = gen_count

            if improved:
                no_improvement_count = 0
                plateau_count = 0
                # Score the new train-incumbent on the held-out bundle; keep the best-val one.
                # Checkpoint the *selection* incumbent (held-out when active, else train).
                if use_val and best_x is not None and np.isfinite(best_f):
                    try:
                        vloss = float(val_eval(best_x))
                    except Exception:
                        vloss = np.inf
                    if np.isfinite(vloss) and vloss < best_val_f - 1e-9:
                        best_val_f, best_val_x, best_val_train = vloss, best_x.copy(), float(best_f)
                        val_plateau = 0
                        print(f"[CMA-ES/val] gen {gen_count:03d}: NEW held-out best="
                              f"{vloss:.6f} (train={float(best_f):.6f})")
                        _ckpt_incumbent(gen_count, best_val_x, best_val_train, best_val_f, "held_out")
                    else:
                        val_plateau += 1
                        print(f"[CMA-ES/val] gen {gen_count:03d}: train improved to "
                              f"{float(best_f):.6f} but held-out={vloss:.6f} "
                              f"(best held-out={best_val_f:.6f}, plateau={val_plateau})")
                else:
                    _ckpt_incumbent(gen_count, best_x, best_f, None, "train")
            else:
                no_improvement_count += 1
                plateau_count += 1
                if use_val:
                    val_plateau += 1

            # Adaptive sigma reduction: if no improvement for many generations, reduce sigma for finer search
            if no_improvement_count >= no_improvement_threshold and gen_count - last_improvement_gen >= no_improvement_threshold:
                current_sigma = es.sigma
                min_sigma = sigma0 * 0.01
                if current_sigma > min_sigma * 2:
                    new_sigma = max(current_sigma * 0.7, min_sigma)
                    print(f"[CMA-ES] No improvement for {no_improvement_count} generations, reducing sigma from {current_sigma:.4f} to {new_sigma:.4f} for finer search")
                    es.sigma = new_sigma
                    no_improvement_count = 0  # Reset sigma-adaptation counter only

            es.tell(xs, vals)
            # Lightweight per-generation logging (no per-candidate spam)
            try:
                gen_best = float(np.min(vals))
            except Exception:
                gen_best = float('inf')
            # Log every generation; change to `(gen_count % 5) == 0` if you want sparser logs
            msg_gen = (
                f"[CMA-ES] gen {gen_count:03d}: "
                f"best_in_gen={gen_best:.6f}, "
                f"best_overall={float(best_f):.6f}"
            )
            print(msg_gen)
            # Also append to fit_log.jsonl for offline inspection
            try:
                _log_info(msg_gen, {
                    "stage": "cma",
                    "gen": int(gen_count),
                    "best_in_gen": float(gen_best),
                    "best_overall": float(best_f),
                })
            except Exception:
                # Never let logging crash the optimizer
                pass
            gen_count += 1

            # Arm early-stop only after beating the known-best gate (if set).
            # With held-out selection, arm on the *train* loss of the held-out incumbent
            # (same scale as beat_loss=0.4044); plateau itself is timed on held-out.
            gate_metric = (
                best_val_train if (use_val and np.isfinite(best_val_train)) else best_f
            )
            if (
                early_stop_patience is not None
                and early_stop_beat_loss is not None
                and not early_stop_armed
                and np.isfinite(gate_metric)
                and gate_metric < early_stop_beat_loss
            ):
                early_stop_armed = True
                msg_arm = (
                    f"[CMA-ES] early-stop armed at gen {gen_count}: "
                    f"{'train(held-out-incumbent)' if use_val else 'best_overall'}="
                    f"{float(gate_metric):.6f} < beat_loss={early_stop_beat_loss:.6f}"
                )
                print(msg_arm)
                try:
                    _log_info(msg_arm, {
                        "stage": "cma",
                        "early_stop_armed": True,
                        "gen": int(gen_count),
                        "gate_metric": float(gate_metric),
                        "best_overall": float(best_f),
                        "best_val": float(best_val_f) if np.isfinite(best_val_f) else None,
                        "beat_loss": float(early_stop_beat_loss),
                        "use_val": bool(use_val),
                    })
                except Exception:
                    pass

            # Ungated mode (beat_loss=None): arm immediately so patience alone can stop.
            if (
                early_stop_patience is not None
                and early_stop_beat_loss is None
                and not early_stop_armed
            ):
                early_stop_armed = True

            # Plateau early-stop: only after armed. With held-out selection the plateau
            # is timed on held-out (val_plateau); otherwise on train best_overall.
            plateau_for_stop = val_plateau if use_val else plateau_count
            if (
                early_stop_patience is not None
                and early_stop_armed
                and plateau_for_stop >= early_stop_patience
            ):
                early_stopped = True
                msg_stop = (
                    f"[CMA-ES] early stop at gen {gen_count}: no "
                    f"{'held-out' if use_val else 'best_overall'} improvement "
                    f"for {plateau_for_stop} gens (patience={early_stop_patience}"
                    + (
                        f", beat_loss={early_stop_beat_loss:.6f}"
                        if early_stop_beat_loss is not None else ""
                    )
                    + f"); best_overall={float(best_f):.6f}"
                    + (f", best_heldout={float(best_val_f):.6f}" if use_val else "")
                )
                print(msg_stop)
                try:
                    _log_info(msg_stop, {
                        "stage": "cma",
                        "early_stop": True,
                        "gen": int(gen_count),
                        "plateau_count": int(plateau_for_stop),
                        "patience": int(early_stop_patience),
                        "beat_loss": (
                            float(early_stop_beat_loss)
                            if early_stop_beat_loss is not None else None
                        ),
                        "best_overall": float(best_f),
                        "best_val": float(best_val_f) if use_val and np.isfinite(best_val_f) else None,
                        "use_val": bool(use_val),
                    })
                except Exception:
                    pass
                break

        result = es.result
        # Prefer the held-out-selected incumbent when val_eval is active.
        if use_val and best_val_x is not None and np.isfinite(best_val_f):
            final_x = best_val_x
            final_f = float(best_val_train)  # report train loss of the selected point
            print(
                f"[CMA-ES/val] selecting held-out best: held-out={float(best_val_f):.6f} "
                f"(its train={float(best_val_train):.6f}; train-only best was {float(best_f):.6f})"
            )
        elif best_x is None:
            final_x = np.asarray(result.xbest, float)
            final_f = float(result.fbest)
        else:
            final_x = best_x
            final_f = best_f if np.isfinite(best_f) else float(result.fbest)
        final_x = np.minimum(Ub, np.maximum(Lb, final_x))
        if not np.isfinite(final_f):
            final_f = np.inf

        stop_msg = "early_stop_patience" if early_stopped else str(es.stop())
        return types.SimpleNamespace(
            x=np.asarray(final_x, float),
            fun=float(final_f),
            nit=es.countiter,
            nfev=es.countevals,
            message=stop_msg,
            val_fun=(float(best_val_f) if use_val and np.isfinite(best_val_f) else None),
        )

    def _effective_cma_popsize(bounds, opts_extra=None):
        dim = len(bounds)
        base = max(4, 4 + int(3 * np.log(dim + 1)))
        if opts_extra and ('popsize' in opts_extra) and (opts_extra['popsize'] is not None):
            try:
                return int(opts_extra['popsize'])
            except Exception:
                return base
        return base

    resume_from_norm = (resume_from or "none").lower()
    if resume_from_norm not in {"none", "de1", "de2", "local"}:
        raise ValueError("resume_from must be one of: 'none','de1','de2','local'")
    # store normalized value back
    resume_from = resume_from_norm

    if resume_path is not None:
        p = Path(resume_path)

        # try to load metadata to get the recorded loss and frozen indices
        meta = None
        try:
            if p.suffix == ".json":
                with open(p) as f:
                    meta = json.load(f)
            else:
                meta_path = p.with_suffix(".json")
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)

            if meta is not None:
                if "loss" in meta:
                    resume_loss = float(meta["loss"])

                # infer frozen indices either directly or from fit_idx/fit_id
                frozen_idx = meta.get("frozen_idx", None)
                if frozen_idx is None:
                    fit_idx = meta.get("fit_idx", meta.get("fit_id", None))
                    if fit_idx is not None:
                        # Use fitter D_full (12 or 21); never reset to weights-only 12.
                        fit_set = {int(i) for i in fit_idx}
                        frozen_idx = [i for i in range(D_full) if i not in fit_set]
                if frozen_idx is not None:
                    checkpoint_frozen_idx = [int(i) for i in frozen_idx]
        except Exception:
            # if anything goes wrong, fall back to recomputing the loss later
            resume_loss = None
            checkpoint_frozen_idx = None

        # only load theta_log from file if caller did not already supply it
        if resume_theta_log is None:
            if p.suffix == ".npy":
                resume_theta_log = np.load(p)
            elif p.suffix == ".json" and meta is not None and "theta_log" in meta:
                resume_theta_log = np.array(meta["theta_log"], dtype=float)

    if resume_from in {"de1", "de2", "local"}:
        if resume_theta_log is None:
            raise ValueError("resume_from specified but no resume_theta_log/resume_path provided.")
        resume_theta_log = np.asarray(resume_theta_log, float)
        resume_theta_log[train_mask] = np.minimum(Ub_full[train_mask],
                                                  np.maximum(Lb_full[train_mask], resume_theta_log[train_mask]))
        resume_theta_log = _apply_frozen(resume_theta_log, train_mask)
        # Do NOT clamp frozen dims (see DE worker comment on LOG_ZERO vs Lb).
        # use resumed full vector as the starting point so full_from_active really
        # reflects the checkpoint instead of any earlier random initialization
        theta_log0 = resume_theta_log.copy()
        resume_x_act = resume_theta_log[idx]
    else:
        resume_x_act = None

    x0_act = theta_log0[idx]

    # --- Stage 1 global search ---
    if resume_from in {"de2", "local"}:
        de1_x = resume_x_act
        print(f"\n>>> Resume: skipping Stage 1 {_method_label(method_stage1)} <<<")
        de1 = None
    elif resume_from == "de1":
        de1_x = resume_x_act
        print(f"\n>>> Resume: treating resume vector as Stage 1 {_method_label(method_stage1)} best <<<")
        de1 = None
    else:
        if method_stage1 == "de":
            # Initialize global context for CPU-parallel DE worker
            _LOSS_ACTIVE_DE_CONTEXT = {
                "theta_log0": theta_log0,
                "idx": idx,
                "train_mask": train_mask,
                "LOG_ZERO": LOG_ZERO,
                "full_bounds": full_bounds,
                "mean_data_results": mean_data_results,
                "prior_regions": prior_regions,
                "behavior": behavior,
                "model_type": model_type,
                "random_state": random_state,
                "blocks_per_session_override": int(blocks_per_session),
                "safe_loss_fn": _safe,
                "freeze_fill": _de_context_freeze_fill(),
                "loss_extra_kwargs": _loss_extra,
            }
            # Flat all-penalty landscape (NaN S buckets → 1e11) makes scipy DE
            # converge after ~1 generation. Restart with a fresh uniform pop.
            n_de_attempts = 1 + max(0, int(de1_inf_restarts))
            de1 = None
            for _de_attempt in range(n_de_attempts):
                _seed_att = int(random_state) + 10007 * int(_de_attempt)
                _rng_att = np.random.RandomState(_seed_att)
                _x0_att = x0_act if _de_attempt == 0 else None  # restarts: no x0 inject
                _tag = (f" basin-jump restart {_de_attempt}/{n_de_attempts - 1}"
                        if _de_attempt else "")
                print(
                    f"[Stage1 DE] (active dims={len(idx)}) pop={de_popsize*len(idx)}, "
                    f"iters={de1_maxiter}{_tag}"
                )
                init_pop1 = _make_init_population(
                    bnds_act, de_popsize, _rng_att, _x0_att, jitter_scale)
                _de_iter = {'i': 0}

                def _de_callback(xk, convergence):
                    _de_iter['i'] += 1
                    try:
                        _log_info(f"[DE1] iter={_de_iter['i']}")
                    except Exception:
                        pass
                    return False

                de_workers, de_cleanup = _make_de_workers(n_jobs, _LOSS_ACTIVE_DE_CONTEXT)
                try:
                    de1 = differential_evolution(
                        func=_loss_active_de_worker, bounds=bnds_act, strategy='best1bin',
                        maxiter=de1_maxiter, popsize=de_popsize, init=init_pop1,
                        polish=False, updating='deferred', workers=de_workers,
                        seed=_seed_att, callback=_de_callback,
                    )
                finally:
                    de_cleanup()
                _best = float(de1.fun) if de1 is not None else 1e11
                if np.isfinite(_best) and _best < 1e11:
                    if _de_attempt:
                        print(
                            f"[Stage1 DE] basin-jump recovered finite loss={_best:.6f} "
                            f"(attempt {_de_attempt + 1}/{n_de_attempts})"
                        )
                    break
                if _de_attempt + 1 < n_de_attempts:
                    print(
                        f"[Stage1 DE] best still penalty ({_best:.3g}); "
                        f"jumping to a new random basin "
                        f"({_de_attempt + 1}/{n_de_attempts - 1} restarts left)"
                    )
        else:
            lam = _effective_cma_popsize(bnds_act, cma_opts_stage1)
            print(f"[Stage1 CMA-ES] (active dims={len(idx)}) lambda={lam}, iters={de1_maxiter}, n_jobs={n_jobs}")
            de1 = _run_cma_es(
                bounds=bnds_act,
                x0=x0_act,
                maxiter=de1_maxiter,
                opts_extra=cma_opts_stage1,
                n_jobs=n_jobs,
                parallel_backend=parallel_backend,
            )
        de1_x = de1.x
        _save_de_result(
            de1,
            stage="de1",
            tag="v2",
            fit_idx=idx,
            random_state=random_state,
            algo=("de" if method_stage1 == "de" else "cma"),
        )

    # --- Post-Stage1 scoring / shrink step ---
    # For DE-based Stage 1, score jittered/Sobol candidates and shrink bounds.
    # For CMA-ES Stage 1, skip extra scoring and just use the Stage 1 best.
    if method_stage1 == "de":
        cand1 = [de1_x] + ([x0_act] if x0_act is not None else [])
        sob = Sobol(d=len(idx), scramble=True, seed=random_state)
        sob_pts = sob.random_base2(int(np.ceil(np.log2(sobol_count)))) if sobol_count > 0 else []
        for z in sob_pts:
            th = np.array([L + z[i] * (U - L) for i, (L, U) in enumerate(bnds_act)], float)
            cand1.append(th)
        print(f"[Post-Stage1] scoring candidates (active): {len(cand1)}")

        from joblib import Parallel, delayed
        _post1_fulls = [full_from_active(th) for th in cand1]
        vals1 = Parallel(n_jobs=n_jobs, backend=parallel_backend)(
            delayed(_safe)(
                th, mean_data_results, prior_regions, behavior,
                model_type=model_type, plot=False, debug=False, verbose=False, stim_rng=None,
                **_loss_extra,
            ) for th in _post1_fulls
        )
        k_elite = max(5, int(np.ceil(elite_frac * len(cand1))))
        elite = [cand1[i] for i in np.argsort(vals1)[:k_elite]]
        bnds_shrunk = _shrink_bounds(elite, bnds_act, pad=0.20)
        bnds_local = bnds_shrunk  # unified name used by helpers below
    else:
        # CMA-ES already did its own internal selection; avoid extra forward passes.
        elite = [de1_x]
        bnds_shrunk = bnds_act
        bnds_local = bnds_act
        print("[Post-Stage1] Skipped scoring for CMA-ES Stage 1 (using Stage 1 best only)")

    # Stage 2 global search with Stage 1 loss-based gating ---
    # Compute Stage 1 loss (using saved result if available)
    if resume_from in {"de2", "local", "de1"} and de1 is None:
        # prefer loss recorded in the checkpoint, fall back to recomputing
        if resume_loss is not None:
            stage1_loss = float(resume_loss)
        else:
            stage1_loss = float(loss_active(de1_x))
    else:
        stage1_loss = float(de1.fun) if de1 is not None else float(loss_active(de1_x))
    _log_info("[Stage1] final loss", {"stage1_loss": float(stage1_loss)})

    # threshold for deciding whether to enter Stage 2
    if L_threshold is None:
        # Auto threshold from the number of *active* (unfrozen) params:
        # near-full fits (>=10 active) start Stage 2 only from a good basin (0.8);
        # heavily frozen fits get a looser gate (3.0).
        n_active = int(np.sum(train_mask))
        if n_active >= 10:
            L_threshold = 0.8
        else:
            L_threshold = 3.0

    borderline_hi = L_threshold + 0.4

    # --- print + log threshold + frozen info ---
    print(f"[Resume] frozen_idx={checkpoint_frozen_idx}, L_threshold={L_threshold}")
    try:
        _log_info(
            "[resume thresholds]",
            {
                "frozen_idx": checkpoint_frozen_idx,
                "L_threshold": float(L_threshold),
                "borderline_hi": float(borderline_hi),
            }
        )
    except Exception:
        pass

    # Borderline regime: extend Stage 1 when loss ∈ [L_threshold, borderline_hi).
    # CMA: more gens from current best. DE: another DE run seeded around current best.
    # Also allow resume_from='de1' (Stage-1 ckpt restart) even when de1_maxiter was
    # zeroed by the driver — otherwise a borderline DE ckpt would FIT_FAILED with no extend.
    if (
        L_threshold <= stage1_loss < borderline_hi
        and resume_from in ("none", "de1")
    ):
        _base_iters = int(de1_maxiter) if int(de1_maxiter or 0) > 0 else 40
        extra_iters = max(10, _base_iters // 2)
        if method_stage1 == "cma":
            print(f"[Stage1 CMA-ES] Borderline loss {stage1_loss:.3f} ∈ [{L_threshold}, {borderline_hi}); "
                  f"extending Stage 1 by {extra_iters} iterations")
            _log_info("[Stage1] extending CMA for borderline loss", {
                "stage1_loss": float(stage1_loss),
                "extra_iters": int(extra_iters)
            })
            de1_ext = _run_cma_es(
                bounds=bnds_act,
                x0=de1_x,
                maxiter=extra_iters,
                opts_extra=cma_opts_stage1,
                n_jobs=n_jobs,
                parallel_backend=parallel_backend,
            )
            stage1_loss_ext = float(de1_ext.fun)
            print(f"[Stage1 CMA-ES] extended run loss={stage1_loss_ext:.6f} (prev={stage1_loss:.6f})")
            _log_info("[Stage1] extended CMA result", {
                "stage1_loss_prev": float(stage1_loss),
                "stage1_loss_ext": float(stage1_loss_ext)
            })
            if stage1_loss_ext < stage1_loss:
                de1 = de1_ext
                de1_x = de1_ext.x
                stage1_loss = stage1_loss_ext
                _save_de_result(
                    de1,
                    stage="de1_ext",
                    tag="v2",
                    fit_idx=idx,
                    random_state=random_state,
                    algo="cma",
                )
        elif method_stage1 == "de":
            print(f"[Stage1 DE] Borderline loss {stage1_loss:.3f} ∈ [{L_threshold}, {borderline_hi}); "
                  f"extending Stage 1 by {extra_iters} iterations around current best")
            _log_info("[Stage1] extending DE for borderline loss", {
                "stage1_loss": float(stage1_loss),
                "extra_iters": int(extra_iters)
            })
            _LOSS_ACTIVE_DE_CONTEXT = {
                "theta_log0": theta_log0,
                "idx": idx,
                "train_mask": train_mask,
                "LOG_ZERO": LOG_ZERO,
                "full_bounds": full_bounds,
                "mean_data_results": mean_data_results,
                "prior_regions": prior_regions,
                "behavior": behavior,
                "model_type": model_type,
                "random_state": random_state,
                "blocks_per_session_override": int(blocks_per_session),
                "safe_loss_fn": _safe,
                "freeze_fill": _de_context_freeze_fill(),
                "loss_extra_kwargs": _loss_extra,
            }
            init_pop_ext = _make_init_population(bnds_act, de_popsize, rng, de1_x, jitter_scale)
            de_workers, de_cleanup = _make_de_workers(n_jobs, _LOSS_ACTIVE_DE_CONTEXT)
            try:
                de1_ext = differential_evolution(
                    func=_loss_active_de_worker, bounds=bnds_act, strategy='best1bin',
                    maxiter=extra_iters, popsize=de_popsize, init=init_pop_ext,
                    polish=False, updating='deferred', workers=de_workers,
                    seed=int(random_state) + 17,
                )
            finally:
                de_cleanup()
            stage1_loss_ext = float(de1_ext.fun)
            print(f"[Stage1 DE] extended run loss={stage1_loss_ext:.6f} (prev={stage1_loss:.6f})")
            _log_info("[Stage1] extended DE result", {
                "stage1_loss_prev": float(stage1_loss),
                "stage1_loss_ext": float(stage1_loss_ext)
            })
            if stage1_loss_ext < stage1_loss:
                de1 = de1_ext
                de1_x = de1_ext.x
                stage1_loss = stage1_loss_ext
                _save_de_result(
                    de1,
                    stage="de1_ext",
                    tag="v2",
                    fit_idx=idx,
                    random_state=random_state,
                    algo="de",
                )

    def _failed_stage1_return(reason):
        """Stage 1 did not reach L_threshold — do not enter Stage 2; mark fit failed."""
        print(f"\n>>> FIT FAILED ({reason}): Stage 1 loss={stage1_loss:.3f} "
              f"(L_threshold={L_threshold}, borderline_hi={borderline_hi}); skipping Stage 2 <<<")
        _log_info("[Stage2] skipped — fit failed", {
            "stage1_loss": float(stage1_loss),
            "L_threshold": float(L_threshold),
            "borderline_hi": float(borderline_hi),
            "reason": reason,
        })
        theta_best_full = full_from_active(de1_x)
        out = _unpack_result(theta_best_full)
        out.update({
            'theta_log': theta_best_full, 'loss': float(stage1_loss),
            'bounds_stage1': full_bounds, 'bounds_stage2': bnds_act,
            'fit_idx': idx,
            'run_dir': str(run_dir),
            'log_path': str(log_path),
            'fit_status': 'failed_stage1',
            'fail_reason': reason,
        })
        return out

    # Decide whether to proceed to Stage 2
    if stage1_loss >= borderline_hi:
        return _failed_stage1_return("stage1_loss_ge_borderline_hi")
    elif stage1_loss >= L_threshold:
        return _failed_stage1_return("stage1_loss_ge_L_threshold_after_extend")

    # Unfreeze Stage-1 hold dims (e.g. retinal) before CMA / polish.
    if _use_s1_hold:
        th_s1 = full_from_active(de1_x)
        train_mask = mask_s2
        idx = idx_s2
        bnds_act = bnds_act_s2
        Lb_act = np.array([L for (L, U) in bnds_act], float)
        Ub_act = np.array([U for (L, U) in bnds_act], float)
        hold_mask = None
        _use_s1_hold = False
        theta_log0 = np.asarray(th_s1, float)
        de1_x = theta_log0[idx]
        x0_act = theta_log0[idx]
        elite = [de1_x]
        bnds_shrunk = _box_around(de1_x, bnds_act, pad=0.10)
        bnds_local = bnds_shrunk
        print(f"[Stage2] unfroze held dims; active={int(idx.size)} {idx.tolist()}")

    # At this point, stage1_loss < L_threshold → proceed with Stage 2 as before
    # Skip Stage 2 entirely if de2_maxiter <= 0
    if de2_maxiter <= 0:
        print(f"\n>>> Skipping Stage 2 (de2_maxiter={de2_maxiter}) <<<")
        de2_x = de1_x  # Use Stage 1 result as Stage 2 result
        de2 = type('obj', (object,), {'x': de2_x, 'fun': float(loss_active(de2_x))})()
        _save_de_result(
            de2,
            stage="de2",
            tag="v2",
            fit_idx=idx,
            random_state=random_state,
            algo=("de" if method_stage1 == "de" else "cma"),  # Use Stage 1 algo label
        )
        cand2 = [de2_x] + elite + ([x0_act] if x0_act is not None else [])
    elif resume_from == "local":
        # Skip Stage-2 CMA; re-enter polish / finalize from the resumed vector.
        # Must define `de2` — polish reject / no-refine paths read de2.fun.
        de2_x = resume_x_act
        _loss_s2 = (
            loss_active_stage2 if blocks_per_session_stage2 is not None else loss_active
        )
        fun0 = (
            float(resume_loss)
            if resume_loss is not None and np.isfinite(resume_loss)
            else float(_loss_s2(de2_x))
        )
        de2 = types.SimpleNamespace(x=np.asarray(de2_x, float), fun=fun0)
        cand2 = [de2_x] + elite + ([x0_act] if x0_act is not None else [])
        print(f"\n>>> Resume: skipping Stage 2 CMA (resume_from=local); "
              f"start loss≈{fun0:.6f} <<<")
    else:
        # Use Stage 2 loss wrapper if override is provided, otherwise use regular loss
        loss_func_stage2 = loss_active_stage2 if blocks_per_session_stage2 is not None else loss_active
        # Use Stage 1 result (de1_x) as starting point when not resuming, not the initial guess
        if resume_from == "de2":
            focus_vec = resume_x_act
            bnds_shrunk = _box_around(resume_x_act, bnds_act, pad=0.10)
            print(f"\n>>> Resume: entering Stage 2 {_method_label(method_stage2)} around resume vector <<<")
        else:
            focus_vec = de1_x  # Use Stage 1 best result, not initial guess
            print(f"\n>>> Entering Stage 2 (focused {_method_label(method_stage2)}) starting from Stage 1 best <<<")
        
        if method_stage2 == "de":
            init_pop2 = _make_init_population(bnds_shrunk, de_popsize, rng, focus_vec, jitter_scale)
            # Stage-2 DE candidate evals run in parallel like Stage 1 (was workers=1,
            # leaving cores idle for DE->DE schedules). loss_func_stage2 is a nested
            # closure (not picklable for scipy's multiprocessing Pool), so parallelize
            # via a threading map instead: no pickling, and the numba kernel releases
            # the GIL. Falls back to serial when n_jobs == 1.
            if n_jobs is not None and n_jobs != 1:
                from joblib import Parallel, delayed

                def _de2_worker_map(fn, it):
                    return Parallel(n_jobs=n_jobs, backend="threading")(
                        delayed(fn)(x) for x in it
                    )

                de2_workers = _de2_worker_map
            else:
                de2_workers = 1
            de2 = differential_evolution(
                func=loss_func_stage2, bounds=bnds_shrunk, strategy='best1bin',
                maxiter=de2_maxiter, popsize=de_popsize, init=init_pop2,
                polish=False, updating='deferred', workers=de2_workers, seed=random_state
            )
        else:
            lam2 = _effective_cma_popsize(bnds_shrunk, cma_opts_stage2)
            # Use Stage 2 sigma scale if provided, otherwise use Stage 1 value
            sigma_scale_stage2 = cma_sigma_scale_stage2 if cma_sigma_scale_stage2 is not None else cma_sigma_scale
            print(f"[Stage2 CMA-ES] (active dims={len(idx)}) lambda={lam2}, iters={de2_maxiter}, sigma_scale={sigma_scale_stage2:.3f}, n_jobs={n_jobs}")
            de2 = _run_cma_es(
                bounds=bnds_shrunk,
                x0=focus_vec,
                maxiter=de2_maxiter,
                opts_extra=cma_opts_stage2,
                sigma_scale=sigma_scale_stage2,
                loss_func=loss_func_stage2,
                n_jobs=n_jobs,
                parallel_backend=parallel_backend,
                early_stop_patience=cma_early_stop_patience,
                early_stop_beat_loss=cma_early_stop_beat_loss,
                checkpoint_stage="stage2",  # rolling restart checkpoint per improved gen
                val_eval=(_val_loss_active if val_held_out_on else None),
            )
        de2_x = de2.x
        _save_de_result(
            de2,
            stage="de2",
            tag="v2",
            fit_idx=idx,
            random_state=random_state,
            algo=("de" if method_stage2 == "de" else "cma"),
        )
        cand2 = [de2_x] + elite + ([x0_act] if x0_act is not None else [])

    # --- Robust local refinement with penalty-detection + Powell fallback ---
    # Skip local refinement if CMA-ES was used in either stage (CMA-ES already provides good convergence)
    use_cma = (method_stage1 == "cma") or (method_stage2 == "cma")
    
    # Post-Stage2 candidate scoring (only needed if local refinement will run)
    # Skip scoring if Stage 2 was skipped OR if local refinement will be skipped
    if de2_maxiter > 0:
        if method_stage2 == "de" and not use_cma:
            for _ in range(top_k):
                cand2.append(de2_x + rng.normal(scale=jitter_scale, size=len(idx)))
            print(f"[Post-Stage2] scoring candidates (active): {len(cand2)}")

            # Use Stage 2 loss function for candidate scoring if override is provided
            eval_func_stage2 = (
                loss_func_stage2 if blocks_per_session_stage2 is not None else loss_active
            )
            vals2 = [eval_func_stage2(th) for th in cand2]
            seeds = [cand2[i] for i in np.argsort(vals2)[:top_k]]
        elif method_stage2 == "cma":
            # Stage 2 used CMA-ES: rely on CMA's internal selection, skip extra scoring
            print("[Post-Stage2] Skipped scoring for CMA-ES Stage 2 (local refinement / CMA selection sufficient)")
            seeds = [de2_x]  # Best from Stage 2 CMA-ES
        else:
            # Stage 2 ran but local refinement will be skipped (CMA-ES used in some stage)
            print("[Post-Stage2] Skipped (local refinement will be skipped, scoring not needed)")
            seeds = [de2_x]  # Still need seeds variable, but won't be used
    else:
        # Stage 2 was skipped
        print("[Post-Stage2] Skipped (Stage 2 was skipped)")
        seeds = [de2_x]  # Use Stage 2 result (which is Stage 1 result) as the seed
        
    # Loss used for local polish: Stage-2 protocol when available (bps override + stim bundles)
    loss_func_local = (
        loss_active_stage2 if blocks_per_session_stage2 is not None else loss_active
    )

    if use_cma and not local_refine_after_cma:
        print(
            f"\n>>> Skipping local polish - CMA-ES used in "
            f"{'Stage 1' if method_stage1 == 'cma' else ''} "
            f"{'Stage 2' if method_stage2 == 'cma' else ''} "
            f"(local_refine_after_cma=False) <<<"
        )
        best_xa = de2_x
        best_fun = float(de2.fun)
        best_loc = types.SimpleNamespace(x=de2_x, fun=best_fun)
    elif use_cma and local_refine_after_cma:
        # Post-CMA polish. Default: the focused "prior" set [6,8,10,11]
        # (g_i, d_i, theta_c, theta_d) — closes CMA integrator-gain overshoot without
        # touching W. Full-12 "active" refine (pass list(range(12))) reaches lower
        # in-sample loss but overfits the training bundle (held-out worse; 2026-08-04g),
        # so it is opt-in. Always intersected with train_mask.
        default_refine = (list(default_refine_idx)
                          if default_refine_idx is not None else [6, 8, 10, 11])
        refine_full = list(local_refine_idx) if local_refine_idx is not None else default_refine
        refine_full = [int(i) for i in refine_full if 0 <= int(i) < D_full and train_mask[int(i)]]
        if not refine_full:
            print("[Local-after-CMA] no refine indices in train_mask; using CMA best")
            best_xa = de2_x
            best_fun = float(de2.fun)
            best_loc = types.SimpleNamespace(x=de2_x, fun=best_fun)
        else:
            theta_cma = full_from_active(de2_x)
            # Pre-polish held-out baseline (used to reject polish that overfits train).
            val_before_polish = None
            if val_held_out_on:
                try:
                    val_before_polish = float(_val_loss_active(de2_x))
                    print(f"[Local-after-CMA/val] pre-polish held-out={val_before_polish:.6f}")
                except Exception:
                    val_before_polish = None
            bnds_ref = [full_bounds[i] for i in refine_full]
            L_ref = np.array([lo for lo, _ in bnds_ref], float)
            U_ref = np.array([hi for _, hi in bnds_ref], float)
            x0_ref = np.minimum(U_ref, np.maximum(L_ref, theta_cma[refine_full].astype(float)))

            def _full_from_refine(x_ref):
                th = theta_cma.copy()
                th[refine_full] = x_ref
                return _apply_frozen(th, train_mask)

            import threading
            _refine_lock = threading.Lock()
            _refine_nfev = [0]
            _refine_best = [np.inf]
            _refine_best_x = [x0_ref.copy()]
            _refine_t0 = time.perf_counter()

            # Worker count for optional CMA-polish candidate evals (threading: closures
            # holding a Lock are not loky-picklable; GIL-bound but quality > speed here).
            if n_jobs == -1:
                n_jobs_ref = int(os.environ.get("JOBLIB_N_JOBS") or (os.cpu_count() or 1))
            else:
                n_jobs_ref = max(1, int(n_jobs or 1))

            def loss_refine(x_ref, verbose=False):
                """Polish train loss = Stage-2 protocol (sample/mean over stage2 bundles)."""
                th = _full_from_refine(x_ref)
                val = float(loss_func_local(th[idx], verbose=verbose))
                with _refine_lock:
                    _refine_nfev[0] += 1
                    nfev = _refine_nfev[0]
                    if val < _refine_best[0]:
                        _refine_best[0] = val
                        _refine_best_x[0] = np.asarray(x_ref, float).copy()
                    best = _refine_best[0]
                if (nfev == 1) or (nfev % 5 == 0):
                    print(
                        f"[Local-after-CMA] nfev={nfev}  "
                        f"last={val:.6f}  best={best:.6f}  "
                        f"wall={time.perf_counter() - _refine_t0:.0f}s"
                    )
                return val

            def loss_refine_bounded(x_ref):
                xb = np.minimum(U_ref, np.maximum(L_ref, x_ref))
                return loss_refine(xb, verbose=False)

            class _LocalRefinePlateau(Exception):
                pass

            _stall = {"iters": 0, "best_at_cb": np.inf}
            patience = int(local_refine_patience or 0)
            max_wall = (float(local_refine_max_wall_s)
                        if local_refine_max_wall_s is not None else None)

            def _plateau_callback(xk):
                """Powell/CMA shared: stop on train-loss plateau or wall cap."""
                best = _refine_best[0]
                if best < _stall["best_at_cb"] - 1e-5:
                    _stall["best_at_cb"] = best
                    _stall["iters"] = 0
                else:
                    _stall["iters"] += 1
                elapsed = time.perf_counter() - _refine_t0
                print(
                    f"[Local-after-CMA] iter callback: best={best:.6f}  "
                    f"stall_iters={_stall['iters']}/{patience or 'off'}  "
                    f"wall={elapsed:.0f}s"
                    + (f"/{max_wall:.0f}s" if max_wall is not None else "")
                )
                if patience > 0 and _stall["iters"] >= patience:
                    raise _LocalRefinePlateau(
                        f"no best-loss improve for {patience} polish iters"
                    )
                if max_wall is not None and elapsed >= max_wall:
                    raise _LocalRefinePlateau(
                        f"wall-clock cap reached ({elapsed:.0f}s >= {max_wall:.0f}s)"
                    )

            def _cma_polish_refine():
                """Small-sigma CMA-ES restart on the refine dims (optional polish method)."""
                import cma as _cma
                span_ref = np.maximum(U_ref - L_ref, 1e-12)
                sigma0 = float(local_refine_cma_sigma) * float(np.median(span_ref))
                if not np.isfinite(sigma0) or sigma0 <= 0:
                    sigma0 = 1e-2
                popsize = max(6, 4 + int(3 * np.log(len(refine_full) + 1)))
                es = _cma.CMAEvolutionStrategy(
                    x0_ref.tolist(), sigma0,
                    {"bounds": [L_ref.tolist(), U_ref.tolist()],
                     "maxiter": int(local_maxiter), "seed": int(random_state),
                     "verb_disp": 0, "popsize": popsize},
                )
                cma_stall, cma_best_at, gen = 0, np.inf, 0
                while not es.stop():
                    cands = [np.minimum(U_ref, np.maximum(L_ref, np.asarray(c, float)))
                             for c in es.ask()]
                    if n_jobs_ref > 1:
                        from joblib import Parallel, delayed
                        vals = Parallel(n_jobs=n_jobs_ref, backend="threading")(
                            delayed(loss_refine)(c, False) for c in cands)
                    else:
                        vals = [loss_refine(c, False) for c in cands]
                    vals = [float(v) if np.isfinite(v) else 1e11 for v in vals]
                    es.tell([c.tolist() for c in cands], vals)
                    gen += 1
                    best = _refine_best[0]
                    if best < cma_best_at - 1e-5:
                        cma_best_at, cma_stall = best, 0
                    else:
                        cma_stall += 1
                    elapsed = time.perf_counter() - _refine_t0
                    print(f"[Local-after-CMA/cma] gen={gen} best={best:.6f} "
                          f"stall={cma_stall}/{patience or 'off'} wall={elapsed:.0f}s")
                    if patience > 0 and cma_stall >= patience:
                        print("[Local-after-CMA/cma] early stop (plateau)")
                        break
                    if max_wall is not None and elapsed >= max_wall:
                        print("[Local-after-CMA/cma] early stop (wall cap)")
                        break
                return types.SimpleNamespace(
                    x=np.asarray(_refine_best_x[0], float), fun=float(_refine_best[0]),
                    nit=gen, nfev=_refine_nfev[0], success=True, message="cma", jac=None)

            # g_i / d_i live at 6 / 8 only in the 12-d weights / 21-d joint layouts.
            if len(theta_cma) > 8:
                g_i0, d_i0 = float(np.exp(theta_cma[6])), float(np.exp(theta_cma[8]))
                gain_note = f"g_i,d_i,θ start: g_i={g_i0:.4g}, d_i={d_i0:.4g}"
            else:
                g_i0 = d_i0 = float("nan")
                gain_note = f"D={len(theta_cma)} (no g_i/d_i slots)"
            method = str(local_refine_method or "powell").lower()
            if method == "lbfgs":
                print("[Local-after-CMA] WARNING: method='lbfgs' removed; using 'powell'")
                method = "powell"
            if _stage2_has_fixed_stim and stage2_n_stim_seeds > 1:
                n_stim_note = (
                    f"train={stage2_stim_aggregate}({stage2_n_stim_seeds} stim"
                    f"{', restim' if stage2_restim else ''})"
                )
            elif _stage2_has_fixed_stim:
                n_stim_note = "train=single" + (", restim" if stage2_restim else "")
            else:
                n_stim_note = "train=single"
            val_note = (
                f", held-out gate ON (seed={int(val_stim_seed)})"
                if val_held_out_on else ", held-out gate off"
            )
            wall_note = (
                f"max_wall={max_wall:.0f}s"
                if max_wall is not None else "max_wall=off"
            )
            print(
                f"\n>>> Local refine after CMA on idx={refine_full} "
                f"(method={method}, {gain_note}; "
                f"maxiter={int(local_maxiter)}, n_jobs={n_jobs_ref}, "
                f"patience={patience or 'off'}, {wall_note}, "
                f"{n_stim_note}{val_note}, "
                f"powell_fallback={'on' if local_refine_use_powell else 'off'}) <<<"
            )
            f0 = float(loss_refine(x0_ref, verbose=False))
            _stall["best_at_cb"] = f0
            print(f"[Local-after-CMA] start loss={f0:.6f}")

            def _powell_fallback(base_x, cur_fun):
                if (not local_refine_use_powell) or (float(cur_fun) < f0 - 1e-9):
                    return None
                print("[Local-after-CMA] no improvement — Powell fallback")
                try:
                    pt = minimize(loss_refine_bounded, np.asarray(base_x, float),
                                  method="Powell", callback=_plateau_callback,
                                  options={"maxiter": max(5, int(local_maxiter))})
                except _LocalRefinePlateau as e:
                    print(f"[Local-after-CMA] early stop (plateau): {e}")
                    pt = types.SimpleNamespace(
                        x=np.asarray(_refine_best_x[0], float), fun=float(_refine_best[0]),
                        nit=None, nfev=_refine_nfev[0], success=False, message=str(e))
                print(f"[Local-after-CMA] Powell: loss={pt.fun:.6g}, nit≈{getattr(pt,'nit',None)}")
                return pt

            if method == "powell":
                try:
                    loc_try = minimize(
                        loss_refine_bounded, x0_ref, method="Powell",
                        callback=_plateau_callback,
                        options={"maxiter": int(local_maxiter)},
                    )
                except _LocalRefinePlateau as e:
                    print(f"[Local-after-CMA] early stop (plateau): {e}")
                    loc_try = types.SimpleNamespace(
                        x=np.asarray(_refine_best_x[0], float), fun=float(_refine_best[0]),
                        nit=None, nfev=_refine_nfev[0], success=False, message=str(e))
                print(f"[Local-after-CMA] Powell: loss={loc_try.fun:.6g}, "
                      f"nit≈{getattr(loc_try,'nit',None)}, nfev≈{_refine_nfev[0]}")
            elif method == "cma":
                loc_try = _cma_polish_refine()
                print(f"[Local-after-CMA] CMA: loss={loc_try.fun:.6g}, "
                      f"gens≈{getattr(loc_try,'nit',None)}, nfev≈{_refine_nfev[0]}")
                pt = _powell_fallback(loc_try.x, loc_try.fun)
                if pt is not None and float(pt.fun) <= float(loc_try.fun):
                    loc_try = pt
            else:
                raise ValueError(
                    f"unknown local_refine_method {method!r} (use 'powell' or 'cma')"
                )

            # Prefer tracked best (plateau stop can leave result.x suboptimal)
            if float(_refine_best[0]) < float(loc_try.fun) - 1e-12:
                loc_try = types.SimpleNamespace(
                    x=np.asarray(_refine_best_x[0], float), fun=float(_refine_best[0]),
                    nit=getattr(loc_try, "nit", None), nfev=getattr(loc_try, "nfev", None),
                    success=getattr(loc_try, "success", None),
                    message=getattr(loc_try, "message", None), jac=getattr(loc_try, "jac", None))

            # The saved polish vector MUST live inside the box. Powell can return an
            # out-of-bounds result.x whose reported .fun was actually measured at the *clipped*
            # point (loss_refine_bounded clips before every eval), so saving the raw result.x
            # gave params that re-score to the 1e11 penalty on reload — e.g. Powell/active
            # escaped to W_mm=3.59, W_pp=0.0075 (2026-08-04g). Clip, then recompute the loss AT
            # the clipped vector so reported == re-scored.
            loc_x_clipped = np.minimum(U_ref, np.maximum(L_ref, np.asarray(loc_try.x, float)))
            theta_best_loc = _full_from_refine(loc_x_clipped)
            best_xa = theta_best_loc[idx]
            best_fun = float(loss_func_local(theta_best_loc[idx]))
            # Held-out gate: keep polish only if it does not worsen the held-out loss.
            # Prevents the active-refine overfit mode (lower train, higher held-out).
            if val_before_polish is not None:
                try:
                    val_after = float(_val_loss_active(best_xa))
                except Exception:
                    val_after = np.inf
                if (not np.isfinite(val_after)) or (val_after > val_before_polish + 1e-6):
                    print(
                        f"[Local-after-CMA/val] REJECT polish: held-out "
                        f"{val_before_polish:.6f} → {val_after:.6f} (worse); "
                        f"keeping pre-polish CMA params (train {f0:.6f})"
                    )
                    best_xa = de2_x
                    best_fun = float(de2.fun)
                    theta_best_loc = full_from_active(de2_x)
                else:
                    print(
                        f"[Local-after-CMA/val] KEEP polish: held-out "
                        f"{val_before_polish:.6f} → {val_after:.6f}"
                    )
            best_loc = types.SimpleNamespace(x=best_xa, fun=best_fun, jac=getattr(loc_try, "jac", None))
            if len(theta_best_loc) > 8:
                g_i1, d_i1 = float(np.exp(theta_best_loc[6])), float(np.exp(theta_best_loc[8]))
                gain_delta = f"g_i {g_i0:.4g} → {g_i1:.4g}; d_i {d_i0:.4g} → {d_i1:.4g}; "
            else:
                g_i1 = d_i1 = float("nan")
                gain_delta = ""
            print(
                f"[Local-after-CMA] done: loss {f0:.6f} → {best_fun:.6f}; "
                f"{gain_delta}"
                f"nfev_total≈{_refine_nfev[0]} wall={time.perf_counter() - _refine_t0:.0f}s"
            )
            try:
                _log_info(
                    f"[Local-after-CMA] g_i {g_i0:.6g}->{g_i1:.6g} loss {f0:.6f}->{best_fun:.6f}",
                    {
                        "stage": "local_after_cma",
                        "method": method,
                        "refine_idx": refine_full,
                        "g_i_before": g_i0,
                        "g_i_after": g_i1,
                        "d_i_before": d_i0,
                        "d_i_after": d_i1,
                        "loss_before": f0,
                        "loss_after": best_fun,
                        "nfev": int(_refine_nfev[0]),
                        "n_jobs": int(n_jobs_ref),
                        "patience": int(patience),
                        "stage2_n_stim_seeds": int(stage2_n_stim_seeds),
                        "val_before": val_before_polish,
                    },
                )
            except Exception:
                pass
    else:
        from scipy.optimize import approx_fprime

        def _project_in_bounds(x, bounds):
            lo = np.array([lo for lo, _ in bounds], float)
            hi = np.array([hi for _, hi in bounds], float)
            margin = 1e-12
            return np.minimum(hi - margin, np.maximum(lo + margin, x))

        def _grad_norm(x):
            L = np.array([lo for lo,_ in bnds_local], float)
            U = np.array([hi for _,hi in bnds_local], float)
            def cdiff(i, rel=1e-3, amin=1e-6):
                h = max(amin, rel*max(1.0, abs(x[i])))
                xm = x.copy(); xp = x.copy()
                xm[i] = np.clip(x[i]-h, L[i]+1e-12, U[i]-1e-12)
                xp[i] = np.clip(x[i]+h, L[i]+1e-12, U[i]-1e-12)
                fm = loss_func_local(xm); fp = loss_func_local(xp)
                return (fp - fm) / (xp[i] - xm[i])
            try:
                g = np.array([cdiff(i) for i in range(x.size)], float)
                return float(np.max(np.abs(g)))
            except Exception:
                return np.inf
            
        def _penalty_coords_active(x, step=1e-6):
            """Indices in active space where f(x + step*e_i) triggers penalty."""
            hits = []
            f_pen = lambda v: _call_tracked(full_from_active(v), mean_data_results, prior_regions, behavior,
                                            model_type=model_type, plot=False, debug=False)
            for i in range(x.size):
                xv = x.copy(); xv[i] = xv[i] + step
                fv = f_pen(xv)
                if (not np.isfinite(fv)) or (fv >= 1e11):
                    hits.append(i)
            return hits

        max_restarts = 3
        grad_tol = 1e-6

        best_loc_local = None
        best_fun_local = np.inf
        best_xa_local = None

        # ensure final local bounds alias is up to date after Stage 2 setup
        bnds_local = bnds_shrunk

        L_act = np.array([lo for lo,_ in bnds_local], float)
        U_act = np.array([hi for _,hi in bnds_local], float)

        def loss_active_bounded(x):
            # project into [L_act, U_act] before evaluating
            xb = np.minimum(U_act, np.maximum(L_act, x))
            return loss_func_local(xb)

        for si, xa0 in enumerate(seeds, 1):
            x_curr = _project_in_bounds(np.asarray(xa0, float), bnds_local)
            local_best = None

            for attempt in range(max_restarts + 1):
                pen_hits = _penalty_coords_active(x_curr, step=1e-6)
                if pen_hits:
                    print(f"[Local] seed {si}/{len(seeds)} attempt {attempt}: penalty at coords {pen_hits}; Powell fallback")

                    # use bounded objective for Powell
                    loc_try = minimize(loss_active_bounded, x_curr, method='Powell',
                                       options={'maxiter': int(local_maxiter)})
                else:
                    loc_try = minimize(
                        fun=loss_func_local, x0=x_curr, method='L-BFGS-B', bounds=bnds_local,
                        options={'maxiter': int(local_maxiter), 'ftol': 1e-14, 'gtol': 1e-10, 'eps': 1e-6, 'maxls': 100}
                    )

                gnorm = _grad_norm(loc_try.x)
                print(f"[Local] seed {si}/{len(seeds)} attempt {attempt}: loss={loc_try.fun:.6g}, "
                      f"iters≈{getattr(loc_try, 'nit', None)}, grad_inf={gnorm:.3e}, "
                      f"success={getattr(loc_try,'success',None)}")

                if (local_best is None) or (loc_try.fun <= local_best.fun):
                    local_best = loc_try
                if bool(getattr(loc_try, 'success', False)) and gnorm <= grad_tol:
                    break

                jitter = np.random.normal(scale=1e-3, size=loc_try.x.shape)
                x_curr = _project_in_bounds(loc_try.x + jitter, bnds_local)

            # final Powell safeguard if still not satisfactory
            if (not bool(getattr(local_best, 'success', False))) or _grad_norm(local_best.x) > grad_tol:
                pow_try = minimize(loss_active_bounded, x_curr, method='Powell',
                       options={'maxiter': int(local_maxiter)})
                print(f"[Local] seed {si}/{len(seeds)} Powell: loss={pow_try.fun:.6g}, "
                      f"iters≈{getattr(pow_try,'nit',None)}")
                if pow_try.fun <= local_best.fun:
                    local_best = pow_try

            if (best_loc_local is None) or (local_best.fun < best_fun_local):
                best_loc_local = local_best
                best_fun_local = float(local_best.fun)
                best_xa_local = np.asarray(local_best.x, float)
        
        # Use local refinement results
        best_loc = best_loc_local
        best_fun = best_fun_local
        best_xa = best_xa_local

    # optional: save the best local active result + metadata if helpers are available
    try:
        # save full-vector snapshot with gradient if available
        grad_vec = getattr(best_loc, "jac", None)
        theta_best_full_tmp = full_from_active(best_xa)
        _save_p(theta_best_full_tmp, best_fun, tag="2stagelocalrefine",
                random_state=random_state, train_mask=train_mask, grad=grad_vec)
    except Exception:
        pass
    
    theta_best_full = full_from_active(best_xa)
    out = _unpack_result(theta_best_full)

    # Final rolling snapshot so --resume auto sees the polished / selected vector as latest.
    try:
        _save_r(
            theta_best_full, float(best_fun),
            stage="stage2", gen=None,
            train_mask=train_mask, random_state=random_state,
            selection="final",
        )
    except Exception:
        pass

    out.update({
        'theta_log': theta_best_full, 'loss': best_fun,
        'bounds_stage1': full_bounds, 'bounds_stage2': bnds_shrunk,
        'fit_idx': idx,
        'run_dir': str(run_dir),
        'log_path': str(log_path),
        'fit_status': 'ok',
    })
    return out



def fit_weights_local_refine(mean_data_results, prior_regions, behavior,
                             theta_log0=None, init_params=None, train_mask=None,
                             bounds=None, model_type='data', plot=False,
                             local_maxiter=400, random_state=0):
    """
    Pure local refinement (L-BFGS-B) from a provided parameter vector. 
    Supports freezing parameters via `train_mask`.
    Mirrors masking and return structure of fit_weights_two_stage_v2.
    
    Args:
        mean_data_results, prior_regions, behavior: same as two-stage version.
        theta_log0: 1D array of initial log-parameters (preferred).
        init_params: dict/struct to be packed if theta_log0 is None.
        train_mask: bool array (len D) or list of indices to fit. Others are frozen.
        bounds: list of (lo, hi) for all params; defaults to _log_bounds_weights_v2().
        model_type: forwarded to loss.
        plot: forwarded to loss (kept for API parity; typically False for speed).
        local_maxiter: max iterations for L-BFGS-B.
        random_state: unused here; kept for API parity.

    Returns:
        dict with fitted parameters mirroring the two-stage return structure.
    """
    # create run dirs/log file under save_dir
    if '_RUN_DIR' in globals() and (_RUN_DIR is not None):
        run_dir, ckpt_dir, log_path = _RUN_DIR, _CKPT_DIR, _LOG_PATH
    else:
        run_dir, ckpt_dir, log_path = _ensure_run_dirs()
    
    if bounds is None:
        bounds = _log_bounds_weights_v2()
    D = len(bounds)
    LOG_ZERO = -30.0

    # Resolve initial vector
    if theta_log0 is None and init_params is not None:
        theta_log0 = pack_theta_log_weights_v2(init_params)
    if theta_log0 is None:
        raise ValueError("Provide theta_log0 or init_params.")
    theta_log0 = np.asarray(theta_log0, float)
    if theta_log0.shape[0] != D:
        raise ValueError(f"theta_log0 has length {theta_log0.shape[0]} but expected {D}.")

    # Clamp to bounds
    Lb_full = np.array([L for (L, U) in bounds], float)
    Ub_full = np.array([U for (L, U) in bounds], float)
    theta_log0 = np.minimum(Ub_full, np.maximum(Lb_full, theta_log0))

    # Build train mask (freeze others to LOG_ZERO, matching two-stage behavior)
    if train_mask is None:
        train_mask = np.ones(D, dtype=bool)
    elif np.issubdtype(np.asarray(train_mask).dtype, np.integer):
        m = np.zeros(D, dtype=bool)
        m[np.asarray(train_mask, int)] = True
        train_mask = m
    else:
        train_mask = np.asarray(train_mask, bool)
        if train_mask.shape[0] != D:
            raise ValueError(f"train_mask has length {train_mask.shape[0]} but expected {D}.")

    theta_log0 = theta_log0.copy()
    theta_log0[~train_mask] = LOG_ZERO  # freeze non-trained to ~0 actual

    fit_idx = np.where(train_mask)[0]
    if fit_idx.size == 0:
        loss = _tracked_loss_weights_v2(theta_log0, mean_data_results, prior_regions, behavior,
                                        model_type=model_type, plot=plot,
                                        stim_rng=None)
        (W_ii, W_pp, W_mm, W_is, W_pi, W_mi,
         g_i, g_m, d_i, d_m, theta_c, theta_d) = _unpack_log_params_weights_v2(theta_log0)
        return {
            'W': (W_ii, W_pp, W_mm, W_is, W_pi, W_mi),
            'g': (g_i, g_m), 'd': (d_i, d_m),
            'theta': (theta_c, theta_d),
            'theta_log': theta_log0.copy(), 'loss': float(loss),
            'bounds_stage1': bounds, 'bounds_stage2': [bounds[i] for i in fit_idx],
            'fit_idx': fit_idx, 'run_dir': str(run_dir), 'log_path': str(log_path),
        }

    x0 = theta_log0[fit_idx]
    bnds_active = [bounds[i] for i in fit_idx]
    bnds_local = bnds_active  # unified name used by helpers below

    L_act = np.array([lo for lo,_ in bnds_local], float)
    U_act = np.array([hi for _,hi in bnds_local], float)

    def loss_active_bounded(x):
        # project into [L_act, U_act] before evaluating
        xb = np.minimum(U_act, np.maximum(L_act, x))
        return loss_active(x)
    
    def _full_from_active(x_active):
        th = theta_log0.copy()
        th[fit_idx] = x_active
        th[~train_mask] = LOG_ZERO
        return th

    def loss_active(x_active):
        th = _full_from_active(x_active)
        return _tracked_loss_weights_v2(th, mean_data_results, prior_regions, behavior,
                                        model_type=model_type, plot=False,
                                        random_state=random_state, train_mask=train_mask,
                                        stim_rng=None)
    
    # --- robust local refinement with penalty-detection + Powell fallback ---
    from scipy.optimize import approx_fprime

    def _project_in_bounds(x):
        lo = np.array([lo for lo, _ in bnds_local], float)
        hi = np.array([hi for _, hi in bnds_local], float)
        margin = 1e-12
        return np.minimum(hi - margin, np.maximum(lo + margin, x))

    def _grad_norm(x):
        L = np.array([lo for lo,_ in bnds_local], float)
        U = np.array([hi for _,hi in bnds_local], float)
        def cdiff(i, rel=1e-3, amin=1e-6):
            h = max(amin, rel*max(1.0, abs(x[i])))
            xm = x.copy(); xp = x.copy()
            xm[i] = np.clip(x[i]-h, L[i]+1e-12, U[i]-1e-12)
            xp[i] = np.clip(x[i]+h, L[i]+1e-12, U[i]-1e-12)
            fm = loss_active(xm); fp = loss_active(xp)
            return (fp - fm) / (xp[i] - xm[i])
        try:
            g = np.array([cdiff(i) for i in range(x.size)], float)
            return float(np.max(np.abs(g)))
        except Exception:
            return np.inf
        
    def _penalty_coords(x, step=1e-6):
        """Return indices i where f(x + step*e_i) triggers penalty (>=1e11 or non-finite)."""
        hits = []
        f_pen = lambda v: loss_weights_core_v2(v, mean_data_results, prior_regions, behavior,
                                               model_type=model_type, plot=False, debug=False)
        for i in range(x.size):
            xh = x.copy(); xh[i] = xh[i] + step
            fh = f_pen(_full_from_active(xh))
            if (not np.isfinite(fh)) or (fh >= 1e11):
                hits.append(i)
        return hits

    max_restarts = 3
    grad_tol = 1e-6

    x_curr = _project_in_bounds(x0)
    best = None

    for attempt in range(max_restarts + 1):
        # if any coordinate step hits penalty, skip L-BFGS-B and go Powell immediately
        pen_hits = _penalty_coords(x_curr, step=1e-6)
        if pen_hits:
            print(f"[local] penalty detected at coords {pen_hits}; using Powell fallback")
            res_try = minimize(loss_active_bounded, x_curr, method='Powell',
                               options={'maxiter': int(local_maxiter)})
        else:
            res_try = minimize(
                fun=loss_active, x0=x_curr, method='L-BFGS-B', bounds=bnds_local,
                options={'maxiter': int(local_maxiter), 'ftol': 1e-14, 'gtol': 1e-10, 'eps': 1e-6, 'maxls': 100}
            )

        gnorm = _grad_norm(res_try.x)
        if best is None or res_try.fun <= best.fun:
            best = res_try

        # accept if success and gradient is small
        if bool(getattr(res_try, 'success', False)) and gnorm <= grad_tol:
            break

        # prepare jittered restart
        jitter = np.random.normal(scale=1e-3, size=res_try.x.shape)
        x_curr = _project_in_bounds(res_try.x + jitter)

    # final safety: if still poor gradient, try Powell once more
    if (not bool(getattr(best, 'success', False))) or _grad_norm(best.x) > grad_tol:
        pen_hits = _penalty_coords(best.x, step=1e-6)
        if pen_hits:
            print(f"[local] post-check penalty at coords {pen_hits}; Powell retry")
        res_pow = minimize(loss_active_bounded, best.x, method='Powell',
                           options={'maxiter': int(local_maxiter)})
        if res_pow.fun <= best.fun:
            best = res_pow

    res = best
        
    theta_best = _full_from_active(res.x)
    best_loss = float(res.fun)
    
    (W_ii, W_pp, W_mm, W_is, W_pi, W_mi,
     g_i, g_m, d_i, d_m, theta_c, theta_d) = _unpack_log_params_weights_v2(theta_best)


    if plot:
        _ = _tracked_loss_weights_v2(theta_best, mean_data_results, prior_regions, behavior,
                                     model_type=model_type, plot=True,
                                     stim_rng=None)

    # --- save and metadata ---
    _save_params_v2(theta_best, best_loss, tag="localrefine",
                    random_state=random_state, train_mask=train_mask)

    meta = {
        'loss': best_loss,
        'nit': getattr(res, 'nit', None),
        'nfev': getattr(res, 'nfev', None),
        'status': getattr(res, 'status', None),
        'success': getattr(res, 'success', None),
        'message': getattr(res, 'message', None),
    }
    with open(Path(run_dir) / "local_refine_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {
        'W': (W_ii, W_pp, W_mm, W_is, W_pi, W_mi),
        'g': (g_i, g_m), 'd': (d_i, d_m),
        'theta': (theta_c, theta_d),
        'theta_log': theta_best, 'loss': best_loss,
        'bounds_stage1': bounds, 'bounds_stage2': [bounds[i] for i in fit_idx],
        'fit_idx': fit_idx, 'nit': getattr(res, 'nit', None),
        'run_dir': str(run_dir), 'log_path': str(log_path),
        'success': getattr(res, 'success', None),
        'message': getattr(res, 'message', None),
        'status': getattr(res, 'status', None),
    }





# --------- USAGE (guarded — importing this module must not start a fit) -----------
if __name__ == '__main__':
    try:
        from _fit_data import ensure_fit_data_links, load_validated_mean_data
    except ImportError:
        from scripts._fit_data import ensure_fit_data_links, load_validated_mean_data
    ensure_fit_data_links(pth_res=pth_res, require_avg_mean_r=False)
    _mean_path, mean_data_results = load_validated_mean_data()
    print(f"[fit-data] mean_data_results={_mean_path}")
    behavior = np.load(Path(pth_res, 'behavior.npy'), allow_pickle=True).flat[0]
    prior_regions = {'int_regs_choice': int_regs, 'int_regs_stim': int_regs,
            'move_regs_choice': move_regs, 'move_regs_stim': move_regs}
    prior_regions['stim_regs'] = ['VISpm', 'FRP', 'VISal']

    model_type = 'data'
    model_params['direct_offset'] = False
    blocks_per_session=5

    dt = 2.0
    steps_before_obs = 500
    max_obs_per_trial = 1000
    max_steps_per_trial = steps_before_obs + max_obs_per_trial
    # Ensure model_params carries the updated dt-dependent values
    model_params['dt'] = dt
    from model_functions import _update_model_params_for_dt
    _update_model_params_for_dt(model_params, dt)


    loss_history.clear(); _eval_counter['n'] = 0

    frozen_idx = []           # indices to freeze
    train_mask = np.ones(12, dtype=bool)
    train_mask[frozen_idx] = False     # fit all except frozen ones
    disable_realtime_plot()


    # ---run the complete two-stage fitting process---
    CMA_stds2 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1]

    best_v2 = fit_weights_two_stage_v2(
        mean_data_results, prior_regions, behavior, model_type=model_type,
        random_state=56,
        top_k=0,
        # Stage 1: DE as global explorer (now parallelizable via n_jobs)
        global_method_stage1='de',
        de_popsize=8,
        de1_maxiter=40,   # DE Stage 1 iterations
        sobol_count=8,
        # Stage 2: CMA-ES for focused refinement
        global_method_stage2='cma',
        cma_sigma_scale=0.25,  # Base sigma scale (used by CMA-ES)
        cma_sigma_scale_stage2=0.02,  # Smaller sigma for Stage 2 refinement
        cma_opts_stage2={
            'popsize': 16,
            'tolfun': 5e-4,  # Tighter tolerance for better convergence
            'tolx': 5e-5,  # Tighter tolerance for parameter convergence
            'CMA_stds': list(np.array(CMA_stds2)[train_mask]),
            'CMA_diagonal': False,  # Use full covariance matrix for better adaptation
        },
        de2_maxiter=40,   # CMA-ES Stage 2 iterations
        train_mask=train_mask,
    #     resume_from='de2',
    #     resume_path=str(resume_path),
    #     resume_theta_log=theta_log_de1,
        blocks_per_session_stage2=20,  # Increased to reduce loss noise
        n_jobs=16,  # Parallel evaluation for DE and CMA-ES
        parallel_backend='loky',
        deterministic_stage2=True,
        L_threshold=2,
        local_refine_after_cma=True,  # polish g_i/d_i/θ after CMA (Phase 2c)
        local_refine_patience=8,      # stop polish only on plateau
        # stage2_n_stim_seeds=3, stage2_stim_aggregate='sample',  # 1-of-3 per eval (~1× wall)
    )

    print(best_v2)


    # ---restart from checkpoint for local refine run---
    ### comment out if running complete two-stage fit process
    # fname = "weights_v2_loss0p7153_20251029-115424.npy"
    # resume_path = Path(save_dir) / "weights_run_20251028_180629" / fname
    # assert resume_path.exists(), f"Checkpoint not found: {resume_path}"
    # theta_log0 = np.load(resume_path, allow_pickle=True)
    # _ensure_run_dirs(run_dir=resume_path.parent)

    # res = fit_weights_local_refine(mean_data_results, prior_regions, behavior,
    #                                theta_log0=theta_log0,
    #                                model_type=model_type,
    #                                train_mask=train_mask)

    # print("Optimized gains:", res["g"])
    # print("Other parameters (frozen):", res["W"], res["theta"])
    # print(res)

