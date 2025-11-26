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
    D_full = len(_log_bounds_weights_v2())
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

    # gains: g_i, g_m
    bG_i_m = (1e-1, 2e2)
    # bG_s   = (1e-12, 1e-11)

    # offsets: d_i, d_m
    bD_i_m = (1e-5, 1e2)
    # bD_s   = (1e-12, 1e-11)

    # thresholds (set around amplitude of M neurons; allow discordant a bit higher)
    bTh_c  = (0.1, 0.99999)     # theta_c (concordant)
    bTh_d  = (0.1, 0.99999) # theta_d (discordant)

    # bN     = (1e-1, 1.0)   # noise (5 params)

    L = [
        # np.log(btau_i[0]), np.log(btau_p[0]), np.log(btau_m[0]),
        np.log(bW_ii[0]), np.log(bW_pp[0]), np.log(bW_mm[0]),
        np.log(bW_is[0]), np.log(bW_pi[0]), np.log(bW_mi[0]),
        np.log(bG_i_m[0]), np.log(bG_i_m[0]), # g_i, g_m
        np.log(bD_i_m[0]), np.log(bD_i_m[0]), # d_i, d_m
        np.log(bTh_c[0]), np.log(bTh_d[0]),
        # *([np.log(bN[0])] * 5)
    ]

    U = [
        # np.log(btau_i[1]), np.log(btau_p[1]), np.log(btau_m[1]),
        np.log(bW_ii[1]), np.log(bW_pp[1]), np.log(bW_mm[1]),
        np.log(bW_is[1]), np.log(bW_pi[1]), np.log(bW_mi[1]),
        np.log(bG_i_m[1]), np.log(bG_i_m[1]),   # g_i, g_m
        np.log(bD_i_m[1]), np.log(bD_i_m[1]),   # d_i, d_m
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
def loss_weights_core_v2(theta_log, mean_data_results, prior_regions, behavior,
                         model_type='data', plot=False, debug=False, return_details=False,
                         blocks_per_session_override=None, verbose=True,
                         stim_rng=None):
    """
    Core loss in log-space for the v2 (12-param, taus fixed in model_params) model.
    Combines trajectory, prior-effect, and behavioral losses.
    Debug version: prints exact failure sites when returning 1e12.
    
    Args:
        blocks_per_session_override: If provided, use this instead of global blocks_per_session.
        stim_rng: Optional numpy RandomState to use for deterministic stimulus generation.
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
    # Extract verbose from kwargs if present, pass it through to loss_weights_core_v2
    verbose = kwargs.pop('verbose', True) if 'verbose' in kwargs else True
    v = loss_weights_core_v2(theta_log, *args, verbose=verbose, **kwargs)
    if not np.isfinite(v):
        return 1e12
    if v >= 1e11:
        return 1e11
    return float(v)


# Quiet DE worker for Stage 1 (CPU-parallel, no print/logging)
def _loss_active_de_worker(x_act):
    """
    Top-level DE worker for Stage 1 (CPU-parallel, quiet).
    Uses a module-level context set inside fit_weights_two_stage_v2.
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

    mean_data_results = ctx["mean_data_results"]
    prior_regions = ctx["prior_regions"]
    behavior = ctx["behavior"]
    model_type = ctx["model_type"]
    random_state = ctx["random_state"]
    blocks_per_session_override = ctx.get("blocks_per_session_override", None)

    # Reconstruct full log-parameter vector from active coordinates
    th_full = theta_log0.copy()
    th_full[idx] = x_act
    th_full[~train_mask] = LOG_ZERO

    # Clamp to full bounds for safety
    Lb_full = np.array([L for (L, U) in full_bounds], float)
    Ub_full = np.array([U for (L, U) in full_bounds], float)
    th_full = np.minimum(Ub_full, np.maximum(Lb_full, th_full))

    # Quiet evaluation: no printing/logging, no tracked counters
    return _safe_loss_weights_v2(
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
    )


def _tracked_loss_weights_v2(theta_log, mean_data_results, prior_regions, behavior, debug=False,
                             model_type='data', plot=False, verbose=True, SAVE_THRESH_V2=0.4,
                             random_state=None, train_mask=None, blocks_per_session_override=None,
                             stim_rng=None):
    """
    Logs params + loss each evaluation, updates live loss trace,
    and plots model vs data (I,P,M) when loss < SAVE_THRESH_V2 or every 100 steps.
    
    Args:
        blocks_per_session_override: If provided, use this instead of global blocks_per_session.
        verbose: If True, print loss and parameters. Set to False to disable printing
                 during parallel evaluation to avoid I/O contention.
        stim_rng: Optional numpy RandomState to use for deterministic stimulus generation.
    """
    # Pass verbose to disable printing in run_model and loss_weights_core_v2
    loss = _safe_loss_weights_v2(theta_log, mean_data_results, prior_regions, behavior,
                                 model_type=model_type, plot=False, debug=debug,
                                 blocks_per_session_override=blocks_per_session_override,
                                 verbose=verbose,
                                 stim_rng=stim_rng)  # keep core eval cheap
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
                             parallel_backend='loky', deterministic_stage2=False):
    """
    Two-stage optimizer with configurable global search (DE or CMA-ES) + L-BFGS-B.
    Supports freezing a subset of parameters via `train_mask` (bool array or index list).
    When `train_mask` is None, behavior is identical to the unfrozen version.
    Masked parameters are fixed to zero (log-space value LOG_ZERO ≈ -30).

    Args:
        global_method_stage1: 'de' (default) or 'cma'/'cmaes' to select Stage 1 global optimizer.
        global_method_stage2: Override for Stage 2; defaults to `global_method_stage1`.
        cma_sigma_scale: Fraction of (hi-lo) span used as default sigma for CMA-ES Stage 1.
        cma_sigma_scale_stage2: Fraction of (hi-lo) span for CMA-ES Stage 2. If None, uses cma_sigma_scale.
                                Typically smaller than Stage 1 for focused refinement.
        cma_opts_stage1 / cma_opts_stage2: Optional dicts merged into CMA options per stage.
        blocks_per_session_stage2: If provided, use this for Stage 2 evaluations instead of global blocks_per_session.
                                   Stage 1 always uses the global blocks_per_session.
        n_jobs: Number of parallel workers for CMA-ES candidate evaluation. Default 1 (sequential).
                Use -1 to use all available CPU cores, or specify a positive integer.
        parallel_backend: Backend for joblib parallel execution. Options:
            - 'loky': Multiprocessing (default, true parallelism, better for CPU-bound tasks)
            - 'threading': Threads (lower overhead, but limited by GIL for CPU-bound tasks)
            Test both to see which is faster for your specific workload.
        deterministic_stage2: If True and blocks_per_session_stage2 is not None, Stage 2 evaluations
                              reseed NumPy so that the same stimulus batch is reused for every loss
                              evaluation (useful to reduce Monte Carlo noise during CMA refinement).
    """
    
    if '_RUN_DIR' in globals() and (_RUN_DIR is not None):
        run_dir, ckpt_dir, log_path = _RUN_DIR, _CKPT_DIR, _LOG_PATH
    else:
        run_dir, ckpt_dir, log_path = _ensure_run_dirs()

    rng = np.random.RandomState(random_state)
    # Optional fixed seed for Stage 2 stimulus generation so all Stage 2 evals see the same stimuli.
    if deterministic_stage2 and (blocks_per_session_stage2 is not None):
        stage2_stim_seed = int(random_state) + 100003  # any deterministic offset is fine
    else:
        stage2_stim_seed = None
    full_bounds = _log_bounds_weights_v2()
    D_full = len(full_bounds)

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

    LOG_ZERO = -30.0  # log-space value corresponding to ~0 actual

    # --- initial vector (full) ---
    if theta_log0 is None and init_params is not None:
        theta_log0 = pack_theta_log_weights_v2(init_params)
    elif theta_log0 is None and init_params is None:
        # only print the random-init message when not resuming from a checkpoint
        if (resume_from is None) or (str(resume_from).lower() == "none"):
            print("[Init] No init_params or theta_log0 provided — initializing free params within bounds; "
                  "masked params fixed to ~0 actual.")
        theta_log0 = np.full(D_full, LOG_ZERO, dtype=float)
        if init_mask.any():
            Lb_free = np.array([full_bounds[i][0] for i in range(D_full) if init_mask[i]], float)
            Ub_free = np.array([full_bounds[i][1] for i in range(D_full) if init_mask[i]], float)
            theta_log0[init_mask] = rng.uniform(Lb_free, Ub_free)
    else:
        theta_log0 = np.asarray(theta_log0, float)

    # clamp to bounds for free params only; keep masked fixed at LOG_ZERO
    Lb_full = np.array([L for (L, U) in full_bounds], float)
    Ub_full = np.array([U for (L, U) in full_bounds], float)
    theta_log0[init_mask] = np.minimum(Ub_full[init_mask], np.maximum(Lb_full[init_mask], theta_log0[init_mask]))
    theta_log0[~init_mask] = LOG_ZERO

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
            'bounds_stage1': full_bounds, 'bounds_stage2': [full_bounds[i] for i in idx],
            'fit_idx': idx
        }

    bnds_act = [full_bounds[i] for i in idx]
    Lb_act = np.array([L for (L, U) in bnds_act], float)
    Ub_act = np.array([U for (L, U) in bnds_act], float)

    def full_from_active(x_act):
        th = theta_log0.copy()
        th[idx] = x_act
        th[~train_mask] = LOG_ZERO
        return th

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
        return _tracked_loss_weights_v2(th_full, mean_data_results, prior_regions, behavior,
                                    model_type=model_type, plot=False, verbose=verbose_val,
                                    random_state=random_state, train_mask=train_mask,
                                    stim_rng=None)
    
    # Stage 2 loss wrapper with blocks_per_session override and optional deterministic seeding
    def loss_active_stage2(x_act, verbose=None):
        """
        Loss function for Stage 2 with blocks_per_session override.
        When deterministic_stage2 is True and a stage2_stim_seed is set, a local
        NumPy RNG is constructed with that seed and passed into loss_weights_core_v2,
        so that create_stimuli(...) generates the same stimuli batch on every
        Stage-2 evaluation. Otherwise behaves like loss_active with an override.
        
        Args:
            verbose: If None, uses default (True). Set to False to disable printing
                     during parallel evaluation to avoid I/O contention.
        """
        th_full = full_from_active(x_act)
        verbose_val = verbose if verbose is not None else True

        if deterministic_stage2 and (stage2_stim_seed is not None):
            stim_rng = np.random.RandomState(stage2_stim_seed)
        else:
            stim_rng = None

        return _tracked_loss_weights_v2(
            th_full, mean_data_results, prior_regions, behavior,
            model_type=model_type, plot=False, verbose=verbose_val,
            random_state=random_state, train_mask=train_mask,
            blocks_per_session_override=blocks_per_session_stage2,
            stim_rng=stim_rng,
        )

    def _make_init_population(bounds, popsize, rng, x0=None, jitter=0.05):
        L = np.array([lo for lo, hi in bounds])
        U = np.array([hi for lo, hi in bounds])
        n = popsize * len(bounds)
        pop = rng.uniform(L, U, size=(n, len(bounds)))
        if x0 is not None:
            k = min(len(bounds)*4, n//2)
            inj = np.clip(x0 + rng.normal(scale=jitter, size=(k, len(bounds))), L, U)
            pop[:k, :] = inj
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

    def _run_cma_es(bounds, x0=None, maxiter=200, opts_extra=None, sigma_scale=None, loss_func=None, n_jobs=1, parallel_backend='loky'):
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
        
        # Capture variables needed for full_from_active function
        # These are needed because full_from_active is a closure that references local variables
        eval_theta_log0 = theta_log0.copy()  # Make a copy to avoid sharing mutable state
        eval_idx = idx.copy() if hasattr(idx, 'copy') else idx
        eval_train_mask = train_mask.copy() if hasattr(train_mask, 'copy') else train_mask
        eval_log_zero = LOG_ZERO  # Capture LOG_ZERO constant
        
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
        
        # Track consecutive generations with all infinite losses
        consecutive_inf_gens = 0
        max_inf_gens = 5  # If 5 consecutive generations all give infinite loss, increase sigma
        
        # Track convergence: no improvement for N generations
        no_improvement_count = 0
        no_improvement_threshold = 20  # If no improvement for 20 generations, reduce sigma for finer search
        last_improvement_gen = 0
        gen_count = 0

        while not es.stop():
            samples = es.ask()
            xs = []
            for cand in samples:
                cand_arr = np.asarray(cand, float)
                cand_arr = np.minimum(Ub, np.maximum(Lb, cand_arr))
                xs.append(cand_arr)

            # Evaluate candidates in parallel or sequentially
            if use_parallel:
                def eval_single(x):
                    try:
                        th_full = eval_theta_log0.copy()
                        th_full[eval_idx] = x
                        th_full[~eval_train_mask] = eval_log_zero
                        val = float(_safe_loss_weights_v2(
                            th_full, eval_mean_data, eval_prior_regions, eval_behavior,
                            model_type=eval_model_type, plot=False, debug=False,
                            blocks_per_session_override=eval_blocks_override,
                            verbose=False
                        ))
                    except Exception as e:
                        try:
                            val = float(eval_func(x, verbose=False))
                        except TypeError:
                            val = float(eval_func(x))
                    return val if np.isfinite(val) else np.inf
                vals = Parallel(n_jobs=n_jobs_actual, backend=parallel_backend, batch_size='auto', verbose=0)(
                    delayed(eval_single)(x) for x in xs
                )
            else:
                vals = []
                for cand_arr in xs:
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
            else:
                no_improvement_count += 1

            # Adaptive sigma reduction: if no improvement for many generations, reduce sigma for finer search
            if no_improvement_count >= no_improvement_threshold and gen_count - last_improvement_gen >= no_improvement_threshold:
                current_sigma = es.sigma
                min_sigma = sigma0 * 0.01
                if current_sigma > min_sigma * 2:
                    new_sigma = max(current_sigma * 0.7, min_sigma)
                    print(f"[CMA-ES] No improvement for {no_improvement_count} generations, reducing sigma from {current_sigma:.4f} to {new_sigma:.4f} for finer search")
                    es.sigma = new_sigma
                    no_improvement_count = 0  # Reset counter after adjustment

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
        result = es.result
        if best_x is None:
            final_x = np.asarray(result.xbest, float)
        else:
            final_x = best_x
        final_x = np.minimum(Ub, np.maximum(Lb, final_x))
        alt_f = float(result.fbest)
        final_f = best_f if np.isfinite(best_f) else alt_f
        if not np.isfinite(final_f):
            final_f = np.inf

        return types.SimpleNamespace(
            x=np.asarray(final_x, float),
            fun=float(final_f),
            nit=es.countiter,
            nfev=es.countevals,
            message=str(es.stop())
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
                        D_full = len(_log_bounds_weights_v2())
                        frozen_idx = [i for i in range(D_full) if i not in fit_idx]
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
        resume_theta_log[~train_mask] = LOG_ZERO
        resume_theta_log = np.minimum(Ub_full, np.maximum(Lb_full, resume_theta_log))
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
            global _LOSS_ACTIVE_DE_CONTEXT
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
                "blocks_per_session_override": None,
            }
            print(f"[Stage1 DE] (active dims={len(idx)}) pop={de_popsize*len(idx)}, iters={de1_maxiter}")
            init_pop1 = _make_init_population(bnds_act, de_popsize, rng, x0_act, jitter_scale)
            # Insert DE iteration counter and callback
            _de_iter = {'i': 0}
            def _de_callback(xk, convergence):
                _de_iter['i'] += 1
                try:
                    _log_info(f"[DE1] iter={_de_iter['i']}")
                except Exception:
                    pass
                return False
            de1 = differential_evolution(
                func=_loss_active_de_worker, bounds=bnds_act, strategy='best1bin',
                maxiter=de1_maxiter, popsize=de_popsize, init=init_pop1,
                polish=False, updating='deferred', workers=n_jobs, seed=random_state,
                callback=_de_callback
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
        vals1 = Parallel(n_jobs=n_jobs, backend=parallel_backend)(
            delayed(_safe_loss_weights_v2)(
                full_from_active(th), mean_data_results, prior_regions, behavior,
                model_type=model_type, plot=False, debug=False, verbose=False, stim_rng=None
            ) for th in cand1
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
    # if (resume_from in {"de1", "de2", "local"}) and (checkpoint_frozen_idx is not None):
        # fi = list(checkpoint_frozen_idx)
        ti = sum(train_mask)
        # set L_threshold based on how many ids are frozen / their pattern
        if len(ti) >= 10:
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

    # Borderline regime: run a bit more Stage 1 CMA (only if CMA was used)
    if (L_threshold < stage1_loss < borderline_hi) and (method_stage1 == "cma") and (de1_maxiter > 0):
        extra_iters = max(10, int(de1_maxiter // 2))
        print(f"[Stage1 CMA-ES] Borderline loss {stage1_loss:.3f} ∈ ({L_threshold}, {borderline_hi}); "
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

    # Decide whether to proceed to Stage 2
    if stage1_loss >= borderline_hi:
        # Clearly bad basin: discard Stage 2 and return Stage 1 result only
        print(f"\n>>> Discarding Stage 2: Stage 1 loss={stage1_loss:.3f} ≥ {borderline_hi} <<<")
        _log_info("[Stage2] skipped due to high Stage 1 loss", {
            "stage1_loss": float(stage1_loss),
            "borderline_hi": float(borderline_hi)
        })
        theta_best_full = full_from_active(de1_x)
        (W_ii, W_pp, W_mm, W_is, W_pi, W_mi,
         g_i, g_m, d_i, d_m, theta_c, theta_d) = _unpack_log_params_weights_v2(theta_best_full)
        return {
            'W': (W_ii, W_pp, W_mm, W_is, W_pi, W_mi),
            'g': (g_i, g_m), 'd': (d_i, d_m),
            'theta': (theta_c, theta_d),
            'theta_log': theta_best_full, 'loss': float(stage1_loss),
            'bounds_stage1': full_bounds, 'bounds_stage2': bnds_act,
            'fit_idx': idx,
            'run_dir': str(run_dir),
            'log_path': str(log_path),
        }
    elif stage1_loss >= L_threshold:
        # Borderline but failed to improve enough with extra Stage 1
        print(f"\n>>> Skipping Stage 2: Stage 1 loss={stage1_loss:.3f} ≥ {L_threshold} after extension <<<")
        _log_info("[Stage2] skipped after borderline Stage 1", {
            "stage1_loss": float(stage1_loss),
            "L_threshold": float(L_threshold),
            "borderline_hi": float(borderline_hi)
        })
        theta_best_full = full_from_active(de1_x)
        (W_ii, W_pp, W_mm, W_is, W_pi, W_mi,
         g_i, g_m, d_i, d_m, theta_c, theta_d) = _unpack_log_params_weights_v2(theta_best_full)
        return {
            'W': (W_ii, W_pp, W_mm, W_is, W_pi, W_mi),
            'g': (g_i, g_m), 'd': (d_i, d_m),
            'theta': (theta_c, theta_d),
            'theta_log': theta_best_full, 'loss': float(stage1_loss),
            'bounds_stage1': full_bounds, 'bounds_stage2': bnds_act,
            'fit_idx': idx,
            'run_dir': str(run_dir),
            'log_path': str(log_path),
        }

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
        de2_x = resume_x_act
        cand2 = [de2_x] + elite + ([x0_act] if x0_act is not None else [])
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
            de2 = differential_evolution(
                func=loss_func_stage2, bounds=bnds_shrunk, strategy='best1bin',
                maxiter=de2_maxiter, popsize=de_popsize, init=init_pop2,
                polish=False, updating='deferred', workers=1, seed=random_state
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
        
    if use_cma:
        print(f"\n>>> Skipping local refinement (L-BFGS-B) - CMA-ES used in {'Stage 1' if method_stage1 == 'cma' else ''} {'Stage 2' if method_stage2 == 'cma' else ''} <<<")
        # Use best from Stage 2 directly
        best_xa = de2_x
        best_fun = float(de2.fun)
        best_loc = types.SimpleNamespace(x=de2_x, fun=best_fun)
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
                fm = loss_active(xm); fp = loss_active(xp)
                return (fp - fm) / (xp[i] - xm[i])
            try:
                g = np.array([cdiff(i) for i in range(x.size)], float)
                return float(np.max(np.abs(g)))
            except Exception:
                return np.inf
            
        def _penalty_coords_active(x, step=1e-6):
            """Indices in active space where f(x + step*e_i) triggers penalty."""
            hits = []
            f_pen = lambda v: _tracked_loss_weights_v2(full_from_active(v), mean_data_results, prior_regions, behavior,
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
            return loss_active(xb)

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
                        fun=loss_active, x0=x_curr, method='L-BFGS-B', bounds=bnds_local,
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
        _save_params_v2(theta_best_full_tmp, best_fun, tag="2stagelocalrefine",
                        random_state=random_state, train_mask=train_mask, grad=grad_vec)
    except Exception:
        pass
    
    theta_best_full = full_from_active(best_xa)
    (W_ii, W_pp, W_mm, W_is, W_pi, W_mi,
     g_i, g_m, d_i, d_m, theta_c, theta_d) = _unpack_log_params_weights_v2(theta_best_full)

    return {
        'W': (W_ii, W_pp, W_mm, W_is, W_pi, W_mi),
        'g': (g_i, g_m), 'd': (d_i, d_m),
        'theta': (theta_c, theta_d),
        'theta_log': theta_best_full, 'loss': best_fun,
        'bounds_stage1': full_bounds, 'bounds_stage2': bnds_shrunk,
        'fit_idx': idx,
        'run_dir': str(run_dir),
        'log_path': str(log_path),
    }



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





# --------- USAGE -----------
mean_data_results = np.load('mean_data_results.npy', allow_pickle=True).flat[0]
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

