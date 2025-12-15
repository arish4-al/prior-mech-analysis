import time, traceback
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.optimize import differential_evolution, minimize
from scipy.stats.qmc import Sobol
from model_functions import *

# params set for stim fitting (no prior effects)
model_params['direct_offset'] = False
model_params['W_pp'] = 0.45
model_params['W_ii'] = 0.375
model_params['W_mm'] = 0.139
model_params['W_is'] = 0.119
model_params['W_pi'] = 0.00107
model_params['W_mi'] = 1.471
model_params['g_i'] = 0
model_params['d_i'] = 0
model_params['g_m'] = 0
model_params['d_m'] = 0
model_params['g_s'] = 0
model_params['d_s'] = 0

model_params['alpha_w'] = 1.569
model_params['beta_w'] = -0.0815
model_params['alpha_d'] = 35.277
model_params['beta_d'] = 2.0515
model_params['tau_a'] = 320.6594
model_params['W_as'] = 29.3573
model_params['W_ss'] = 0.00069

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


# ---------------------------------------------------------------------
# Utilities: realtime plot + logs
# ---------------------------------------------------------------------
loss_history = []
_eval_counter = {'n': 0}
_rt_plot = {
    'enabled': False, 'every': 10,
    'fig': None, 'ax': None, 'line': None,
    'inline': True, 'handle': None,
}

# Diagnostics
diag = {'evals':0,'sim_calls':0,'sim_ok':0,'sim_nan':0,'sim_exc':0,'t_loss':0.0,'t_sim':0.0}

# Exception counters for phase-split tracing
_exc_counters = {'unpack':0,'stimuli':0,'sim':0,'avg':0,'sse':0,'other':0}
_MAX_SHOW = 3  # print only first few tracebacks per site

# ---------------------------------------------------------------------
# Parameter layout (MIXED space): beta_w is native (can be negative).
# All other parameters are positive (optimized in log-space).
# ---------------------------------------------------------------------
PARAMS = ['alpha_w','beta_w','alpha_d','beta_d','tau_a','W_as','W_ss']
IDX = {n:i for i,n in enumerate(PARAMS)}

def _unpack_params_mixed(theta):
    """
    theta layout (7 params, mixed):
      [log alpha_w, beta_w (native), log alpha_d, log beta_d, log tau_a, log W_as, log W_ss]
    """
    t = np.asarray(theta, dtype=float)
    assert t.size == 7, "theta length must be 7"
    alpha_w = np.exp(t[0])
    beta_w  = t[1]                 # native (may be negative)
    alpha_d = np.exp(t[2])
    beta_d  = np.exp(t[3])
    tau_a   = np.exp(t[4])
    W_as    = np.exp(t[5])
    W_ss    = np.exp(t[6])
    return alpha_w, beta_w, alpha_d, beta_d, tau_a, W_as, W_ss

def _bounds_mixed():
    """
    Bounds for mixed parameterization:
      indices 0,2,3,4,5,6 are in log-space of a positive native param
      index 1 (beta_w) is bounded in native space (allows negatives)
    """
    b_alpha_w = (1.0, 2.6)
    b_beta_w  = (-0.2, 0.2)      # native
    b_alpha_d = (20.0, 40.0)
    b_beta_d  = (1e-2, 3.0)
    btau_a    = (100.0, 400.0)
    bAs       = (1.0, 50.0)
    bSS       = (1e-6, 2e-1)

    L = [np.log(b_alpha_w[0]), b_beta_w[0],
         np.log(b_alpha_d[0]), np.log(b_beta_d[0]),
         np.log(btau_a[0]),    np.log(bAs[0]),     np.log(bSS[0])]
    U = [np.log(b_alpha_w[1]), b_beta_w[1],
         np.log(b_alpha_d[1]), np.log(b_beta_d[1]),
         np.log(btau_a[1]),    np.log(bAs[1]),     np.log(bSS[1])]
    return list(zip(L, U))

def _bounds_arrays(bnds):
    Lb = np.array([lo for lo, hi in bnds]); Ub = np.array([hi for lo, hi in bnds])
    return Lb, Ub

def pack_theta_mixed(init_params):
    """
    Pack native dict into 7-param mixed vector:
      [log alpha_w, beta_w (native), log alpha_d, log beta_d, log tau_a, log W_as, log W_ss]
    """
    return np.asarray([
        np.log(float(init_params['alpha_w'])),
        float(init_params['beta_w']),                 # native
        np.log(float(init_params['alpha_d'])),
        np.log(float(init_params['beta_d'])),
        np.log(float(init_params['tau_a'])),
        np.log(float(init_params['W_as'])),
        np.log(float(init_params['W_ss'])),
    ], dtype=float)

# ---------------------------------------------------------------------
# NaN/overflow guard
# ---------------------------------------------------------------------
def _nan_or_exploded(x):
    """
    Recursively check if input contains NaNs, infs, or excessively large values.
    Works for nested dict/list/tuple/ndarray structures.
    """
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

# ---------------------------------------------------------------------
# Realtime plotting helpers
# ---------------------------------------------------------------------
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
        plt.ioff()
        plt.show()

# ---------------------------------------------------------------------
# Loss (uses mixed params; beta_w can be negative)
# NOTE: relies on project-level globals & functions:
#   - model_params (dict to be mutated)
#   - create_stimuli(...), run_model(...), mean_S_by_contrast(...)
#   - sse_S_dist_by_contrast(...), compute_sse_stim_right(...)
#   - plot_S_dist_by_contrast(...), plot_S_diff_by_contrast_side_with_data(...)
#   - blocks_per_session, trials_per_block_param, block_side_probs, ...
#   - dt, steps_before_obs, etc.
# ---------------------------------------------------------------------
def loss_retinal_weight(theta_vec, avg_data_R, model_type='data', baseline=0, fit_mode='rms'):
    """
    Returns (loss, S_avg). Phase-split diagnostics to reveal where exceptions occur.
    Mixed parameterization: beta_w is native; others are positive via log-space.
    """
    t0 = time.perf_counter()
    try:
        # ---- UNPACK ----
        try:
            alpha_w, beta_w, alpha_d, beta_d, tau_a, W_as, W_ss = _unpack_params_mixed(theta_vec)

            # Pass parametric retinal front-end (no tau_s, no W_s0 array)
            model_params['alpha_w'] = alpha_w
            model_params['beta_w']  = beta_w
            model_params['alpha_d'] = alpha_d
            model_params['beta_d']  = beta_d
            model_params['tau_a']   = tau_a
            model_params['W_as']    = W_as
            model_params['W_ss']    = W_ss
        except Exception:
            _exc_counters['unpack'] += 1
            if _exc_counters['unpack'] <= _MAX_SHOW:
                print("EXC@unpack:", traceback.format_exc().splitlines()[-1])
            diag['sim_exc'] += 1; diag['evals'] += 1; diag['t_loss'] += time.perf_counter()-t0
            return 1e12, None

        # ---- STIMULI ----
        try:
            stimuli, trial_strengths, perceived_trial_strengths, trial_sides, block_sides = create_stimuli(
                blocks_per_session, trials_per_block_param,
                block_side_probs, num_stimulus_strength,
                min_stimulus_strength, max_stimulus_strength,
                min_trials_per_block, max_trials_per_block,
                max_obs_per_trial, steps_before_obs, **model_params
            )
        except Exception:
            _exc_counters['stimuli'] += 1
            if _exc_counters['stimuli'] <= _MAX_SHOW:
                print("EXC@stimuli:", traceback.format_exc().splitlines()[-1])
            diag['sim_exc'] += 1; diag['evals'] += 1; diag['t_loss'] += time.perf_counter()-t0
            return 1e12, None

        # ---- SIM ----
        diag['sim_calls'] += 1
        try:
            t_sim0 = time.perf_counter()
            results = run_model(
                model_type, stimuli, trial_strengths, trial_sides, block_sides, blocks_per_session,
                dt, steps_before_obs, only_initial=False, **model_params
            )
            diag['t_sim'] += time.perf_counter() - t_sim0
        except Exception:
            _exc_counters['sim'] += 1
            if _exc_counters['sim'] <= _MAX_SHOW:
                print("EXC@sim:", traceback.format_exc().splitlines()[-1])
            diag['sim_exc'] += 1; diag['evals'] += 1; diag['t_loss'] += time.perf_counter()-t0
            return 1e12, None

        # ---- AVG ----
        try:
            S_avg = mean_S_by_contrast(results, steps_before_obs)
            if _nan_or_exploded(S_avg):
                diag['sim_nan'] += 1
                diag['evals'] += 1; diag['t_loss'] += time.perf_counter()-t0
                return 1e12, S_avg
        except Exception:
            _exc_counters['avg'] += 1
            if _exc_counters['avg'] <= _MAX_SHOW:
                print("EXC@avg:", traceback.format_exc().splitlines()[-1])
            diag['sim_exc'] += 1; diag['evals'] += 1; diag['t_loss'] += time.perf_counter()-t0
            return 1e12, None

        # ---- SSE ----
        try:
            if fit_mode == 'dist':
                loss_results = sse_S_dist_by_contrast(S_avg, avg_data_R)
            else:
                loss_results = compute_sse_stim_right(S_avg, avg_data_R, baseline)
            loss = float(loss_results['total_loss'])
        except Exception:
            _exc_counters['sse'] += 1
            if _exc_counters['sse'] <= _MAX_SHOW:
                print("EXC@sse:", traceback.format_exc().splitlines()[-1])
            diag['sim_exc'] += 1; diag['evals'] += 1; diag['t_loss'] += time.perf_counter()-t0
            return 1e12, None

        # ---- OK ----
        diag['sim_ok'] += 1
        diag['evals'] += 1
        diag['t_loss'] += time.perf_counter() - t0
        return loss, S_avg

    except Exception:
        _exc_counters['other'] += 1
        if _exc_counters['other'] <= _MAX_SHOW:
            print("EXC@other:", traceback.format_exc().splitlines()[-1])
        diag['sim_exc'] += 1; diag['evals'] += 1; diag['t_loss'] += time.perf_counter()-t0
        return 1e12, None

def loss_retinal_weight_verbose(theta_vec, avg_data_R, baseline=0, fit_mode='rms'):
    """
    Scalar loss for optimizer; prints params and shows curves.
    """
    loss, S_avg = loss_retinal_weight(theta_vec, avg_data_R, baseline=baseline, fit_mode=fit_mode)
    print(f"theta(mixed): {theta_vec}, loss: {loss:.6f}")
    if S_avg is not None:
        if fit_mode == 'dist':
            plot_S_dist_by_contrast(S_avg, avg_data_R)
        else:
            plot_S_diff_by_contrast_side_with_data(S_avg, None, avg_data_R, baseline)
    return float(loss)

def _tracked_loss(theta_vec, avg_data_R, baseline=0, fit_mode='rms', verbose=True):
    """
    Print params + loss each eval, update live loss plot every N steps,
    and plot curves periodically or when loss is small.
    """
    loss, S_avg = loss_retinal_weight(theta_vec, avg_data_R, baseline=baseline, fit_mode=fit_mode)

    _eval_counter['n'] += 1
    step = _eval_counter['n']
    loss_history.append(float(loss))

    if verbose:
        try:
            alpha_w, beta_w, alpha_d, beta_d, tau_a, W_as, W_ss = _unpack_params_mixed(theta_vec)
            print(
                f"[step {step:05d}] "
                f"alpha_w={alpha_w:.4f}, beta_w={beta_w:.4f}, "
                f"alpha_d={alpha_d:.4f}, beta_d={beta_d:.4f}, "
                f"tau_a={tau_a:.4f}, W_as={W_as:.4f}, W_ss={W_ss:.4f} "
                f"-> loss={loss:.6f}"
            )
        except Exception:
            print(f"[step {step:05d}] (could not unpack params) -> loss={loss:.6f}")

        if step % 50 == 0:
            calls = max(1, diag['sim_calls'])
            print(
                f"   diag: evals={diag['evals']} sim_calls={diag['sim_calls']} "
                f"ok={diag['sim_ok']} nan={diag['sim_nan']} exc={diag['sim_exc']}  "
                f"⟨t_sim⟩={diag['t_sim']/calls:.4f}s  "
                f"⟨t_loss⟩={diag['t_loss']/max(1,diag['evals']):.4f}s"
            )

    # live loss plot
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

    # plot curves if loss is reasonable or periodically
    should_plot_curves = (np.isfinite(loss) and loss < 2.5) or (step % 100 == 0)
    if should_plot_curves and (S_avg is not None):
        try:
            if fit_mode == 'dist':
                plot_S_dist_by_contrast(S_avg, avg_data_R)
            else:
                plot_S_diff_by_contrast_side_with_data(S_avg, None, avg_data_R, baseline)
        except Exception:
            pass  # keep optimization robust even if plotting errors

    return float(loss)

# ---------------------------------------------------------------------
# DE utilities and bounds shrinking
# ---------------------------------------------------------------------
def _shrink_bounds_around_elites(elite_thetas, base_bounds, pad=0.15):
    elite = np.vstack(elite_thetas)  # (k, D)
    lo = elite.min(axis=0); hi = elite.max(axis=0)
    span = hi - lo
    Lb = np.array([L for (L, U) in base_bounds])
    Ub = np.array([U for (L, U) in base_bounds])
    newL = np.maximum(Lb, lo - pad*span)
    newU = np.minimum(Ub, hi + pad*span)
    return list(zip(newL.tolist(), newU.tolist()))

def _make_init_population(bounds, popsize, rng, theta_log0=None, jitter_scale=0.05):
    """
    Create an initial DE population (mixed space).
    If theta_log0 is provided, inject several jittered copies clipped to bounds.
    """
    D = len(bounds)
    L = np.array([lo for lo, hi in bounds]); U = np.array([hi for lo, hi in bounds])
    n_pop = popsize * D
    pop = rng.uniform(L, U, size=(n_pop, D))
    if theta_log0 is not None:
        k = min(D * 4, n_pop // 2)  # inject up to half population as jittered initials
        base = np.tile(theta_log0, (k, 1))
        jit = rng.normal(scale=jitter_scale, size=(k, D))
        inj = np.clip(base + jit, L, U)
        pop[:k, :] = inj
    return pop

# ---------------------------------------------------------------------
# Two-stage global + local fitting
# ---------------------------------------------------------------------
def fit_retinal_params_two_stage(
    avg_data_R, baseline=0, random_state=0,
    de1_maxiter=120, elite_frac=0.10,
    de2_maxiter=150, top_k=8, local_maxiter=400,
    theta_log0=None, init_params=None,
    de_popsize=15, jitter_scale=0.05, fit_mode='rms'
):
    """
    Two-stage optimization:
      1) coarse DE (stage 1)
      2) focused DE (stage 2)
      3) local L-BFGS-B refine
    Automatically saves checkpoints and logs inside a timestamped subfolder under save_dir.
    """
    import os, json, datetime
    rng = np.random.RandomState(random_state)
    bnds_log = _bounds_mixed()
    D = len(bnds_log)

    # ------------------------------------------------------------------
    # Setup save folder and paths
    # ------------------------------------------------------------------
    assert 'save_dir' in globals(), "Global variable save_dir must be defined before calling."
    run_name = f"fit_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    ckpt_dir = os.path.join(run_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "fit_log.jsonl")

    def _now_iso():
        return datetime.datetime.now().isoformat(timespec='seconds')

    # ----- resolve user initial guess -----
    if theta_log0 is None and init_params is not None:
        theta_log0 = pack_theta_mixed(init_params)
    if theta_log0 is not None:
        Lb, Ub = _bounds_arrays(bnds_log)
        theta_log0 = np.minimum(Ub, np.maximum(Lb, np.asarray(theta_log0, float)))

    # ===== Stage 1: coarse DE =====
    init_pop1 = _make_init_population(bnds_log, de_popsize, rng, theta_log0, jitter_scale)
    de1 = differential_evolution(
        func=lambda th: _tracked_loss(th, avg_data_R, baseline, fit_mode),
        bounds=bnds_log, strategy='best1bin', maxiter=de1_maxiter,
        popsize=de_popsize, init=init_pop1, polish=False,
        updating='deferred', workers=1, seed=random_state
    )
    de1_best = np.asarray(de1.x, float)

    # candidate + elites
    cand1 = [de1_best]
    if theta_log0 is not None:
        cand1.append(theta_log0)
    sob = Sobol(d=D, scramble=True, seed=random_state)
    mstarts = 64
    sob_pts = sob.random_base2(int(np.ceil(np.log2(mstarts))))
    for z in sob_pts[:mstarts]:
        th = np.array([L + z[i]*(U - L) for i, (L, U) in enumerate(bnds_log)], dtype=float)
        cand1.append(th)

    vals1 = [_tracked_loss(th, avg_data_R, baseline, fit_mode) for th in cand1]
    k_elite = max(5, int(np.ceil(elite_frac * len(cand1))))
    elite_idx = np.argsort(vals1)[:k_elite]
    elite_thetas = [cand1[i] for i in elite_idx]

    # --- checkpoint/log DE1 ---
    de1_ckpt = os.path.join(ckpt_dir, f"de1_ckpt_{_now_iso().replace(':','-')}.npz")
    np.savez(
        de1_ckpt,
        de1_best=de1_best,
        bnds_stage1=np.array(bnds_log, dtype=float),
        elite_thetas=np.vstack(elite_thetas) if elite_thetas else np.empty((0, D)),
        cand1=np.vstack(cand1)
    )
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "ts": _now_iso(),
            "stage": "DE1",
            "de1_maxiter": de1_maxiter,
            "popsize": de_popsize,
            "best_loss": float(min(vals1)),
            "elite_k": int(k_elite)
        }) + "\n")

    # ===== shrink bounds around elites =====
    bnds_shrunk = _shrink_bounds_around_elites(elite_thetas, bnds_log, pad=0.20)

    # ===== Stage 2: focused DE =====
    init_pop2 = _make_init_population(bnds_shrunk, de_popsize, rng, theta_log0, jitter_scale)
    de2 = differential_evolution(
        func=lambda th: _tracked_loss(th, avg_data_R, baseline, fit_mode),
        bounds=bnds_shrunk, strategy='best1bin', maxiter=de2_maxiter,
        popsize=de_popsize, init=init_pop2, polish=False,
        updating='deferred', workers=1, seed=random_state
    )
    de2_best = np.asarray(de2.x, float)

    # pool for local refine
    cand2 = [de2_best] + elite_thetas
    if theta_log0 is not None:
        cand2.append(theta_log0)
    for _ in range(top_k):
        cand2.append(de2_best + rng.normal(scale=jitter_scale, size=D))

    vals2 = [_tracked_loss(th, avg_data_R, baseline, fit_mode) for th in cand2]
    seed_idx = np.argsort(vals2)[:top_k]
    seeds = [np.asarray(cand2[i], float) for i in seed_idx]

    # --- checkpoint/log DE2 ---
    de2_ckpt = os.path.join(ckpt_dir, f"de2_ckpt_{_now_iso().replace(':','-')}.npz")
    np.savez(
        de2_ckpt,
        de2_best=de2_best,
        bnds_stage2=np.array(bnds_shrunk, dtype=float),
        seeds=np.vstack(seeds) if seeds else np.empty((0, D))
    )
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "ts": _now_iso(),
            "stage": "DE2",
            "de2_maxiter": de2_maxiter,
            "top_k": int(top_k),
            "best_loss": float(min(vals2))
        }) + "\n")

    # ===== Stage 3: local L-BFGS-B refine =====
    best_x, best_fun, best_loc = None, np.inf, None
    for th0 in seeds:
        loc = minimize(
            fun=lambda th: _tracked_loss(th, avg_data_R, baseline, fit_mode),
            x0=th0, method='L-BFGS-B', bounds=bnds_shrunk,
            options={'maxiter': local_maxiter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 100, 'eps': 1e-4}
        )
        if best_loc is None or loc.fun < best_fun:
            best_x, best_fun, best_loc = loc.x, float(loc.fun), loc

    # --- log local stage ---
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "ts": _now_iso(),
            "stage": "LOCAL",
            "local_maxiter": local_maxiter,
            "best_loss": float(best_fun),
            "success": bool(best_loc.success),
            "niter": int(best_loc.nit),
            "message": str(best_loc.message)
        }) + "\n")

    # ===== Return best result + restart info =====
    alpha_w, beta_w, alpha_d, beta_d, tau_a, W_as, W_ss = _unpack_params_mixed(best_x)
    return {
        'alpha_w': alpha_w, 'beta_w': beta_w,
        'alpha_d': alpha_d, 'beta_d': beta_d,
        'tau_a': tau_a, 'W_as': W_as, 'W_ss': W_ss,
        'theta_log': best_x,
        'loss': best_fun,
        'bounds_stage1': bnds_log,
        'bounds_stage2': bnds_shrunk,
        # restart + logging artifacts
        'de1_best': de1_best,
        'de1_elites': np.vstack(elite_thetas) if elite_thetas else np.empty((0, D)),
        'de2_best': de2_best,
        'local_seeds': np.vstack(seeds) if seeds else np.empty((0, D)),
        'run_dir': run_dir,
        'log_path': log_path,
        'ckpt_de1': de1_ckpt,
        'ckpt_de2': de2_ckpt,
        # local convergence diagnostics
        'local_success': bool(best_loc.success),
        'local_niter': int(best_loc.nit),
        'local_message': str(best_loc.message)
    }


# ---------------------------------------------------------------------
# Local refine with optional parameter freezing (mixed space)
# ---------------------------------------------------------------------
def fit_retinal_params_from_init(
    avg_data_R, baseline=0,
    theta_log0=None, init_params=None,
    maxiter=600, fit_mode="rms",
    free_names=None, fixed_params=None,
):
    """
    Local L-BFGS-B refine with optional parameter freezing (mixed space).
    Param names: {'alpha_w','beta_w','alpha_d','beta_d','tau_a','W_as','W_ss'}
    fixed_params accepts native-space values, with logs applied internally
      for positive-only params; beta_w is used natively.
    """
    bnds_log = _bounds_mixed()

    if theta_log0 is None and init_params is not None:
        theta_log0 = pack_theta_mixed(init_params)
    if theta_log0 is None:
        raise ValueError("Provide theta_log0 or init_params.")

    theta_log0 = np.asarray(theta_log0, float)
    Lb, Ub = _bounds_arrays(bnds_log)
    theta_log0 = np.minimum(Ub, np.maximum(Lb, theta_log0))

    name2idx = IDX

    # apply fixed params (native→mixed) into seed, then clamp
    if fixed_params:
        theta_log0 = theta_log0.copy()
        for n, val in fixed_params.items():
            if n == 'beta_w':
                theta_log0[name2idx[n]] = float(val)          # native
            else:
                theta_log0[name2idx[n]] = np.log(float(val))  # log-space
        theta_log0 = np.minimum(Ub, np.maximum(Lb, theta_log0))

    # choose free indices
    D = theta_log0.size
    if free_names is None:
        free_idx = np.arange(D)
    else:
        free_idx = np.array([name2idx[n] for n in free_names], dtype=int)

    x0_free   = theta_log0[free_idx]
    bnds_free = [bnds_log[i] for i in free_idx]

    def _assemble_full(x_free):
        full = theta_log0.copy()
        full[free_idx] = x_free
        return np.minimum(Ub, np.maximum(Lb, full))

    def _obj_free(x_free):
        th_full = _assemble_full(x_free)
        return _tracked_loss(th_full, avg_data_R, baseline, fit_mode)

    res = minimize(
        fun=_obj_free, x0=x0_free, method='L-BFGS-B', bounds=bnds_free,
        options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
    )

    theta_opt = _assemble_full(res.x)
    alpha_w, beta_w, alpha_d, beta_d, tau_a, W_as, W_ss = _unpack_params_mixed(theta_opt)
    return {
        'alpha_w': alpha_w, 'beta_w': beta_w,
        'alpha_d': alpha_d, 'beta_d': beta_d,
        'tau_a': tau_a, 'W_as': W_as, 'W_ss': W_ss,
        'theta_log': theta_opt, 'loss': float(res.fun),
        'n_iter': res.nit, 'success': res.success,
        'free_names': PARAMS if free_names is None else list(free_names),
    }

avg_mean_R = np.load('avg_mean_R.npy', allow_pickle=True).flat[0]
blocks_per_session = 10
# init_params = {
#     'alpha_w': model_params['alpha_w'],
#     'beta_w': model_params['beta_w'],
#     'alpha_d': model_params['alpha_d'],
#     'beta_d': model_params['beta_d'],
#     'tau_a': model_params['tau_a'],
#     'W_as' : model_params['W_as'],
#     'W_ss' : model_params['W_ss'],
# }
# print(init_params)

loss_history.clear(); _eval_counter['n'] = 0
enable_realtime_plot(every=10)

# Run two-stage fitting (creates a timestamped folder under save_dir)
best = fit_retinal_params_two_stage(
    avg_data_R=avg_mean_R,
    baseline=0,
    random_state=42,
    de1_maxiter=2,          # low for test; raise for real runs (≥100)
    de2_maxiter=2,
    local_maxiter=8,
    de_popsize=4,
    fit_mode='rms'
)

print("\n=== Fit complete ===")
print(f"Run directory: {best['run_dir']}")
print(f"Best loss:     {best['loss']:.6f}")
print(f"Local success: {best['local_success']}")
print(f"Local iters:   {best['local_niter']}")
print(f"Local message: {best['local_message']}")

# Inspect saved files
print("\nSaved files:")
print(f"  Log file: {best['log_path']}")
print(f"  DE1 ckpt: {best['ckpt_de1']}")
print(f"  DE2 ckpt: {best['ckpt_de2']}")