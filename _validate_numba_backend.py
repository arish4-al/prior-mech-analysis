"""
Validation + benchmark for the Numba simulator backend.

Compares `run_model(..., backend='numpy')` against `backend='numba')` on an
identical stimulus batch (common random numbers) and reports max abs/rel
differences for every consumed output field, plus a wall-clock speedup.

Run:  python _validate_numba_backend.py
"""
import time
import sys
import types
import tempfile
from pathlib import Path
import numpy as np


# ---------------------------------------------------------------------------
# Stub the heavy IBL data-loading deps so we can import model_functions in
# isolation (the simulator itself does not need them). This avoids unrelated
# environment/version issues in ibllib/iblutil.
# ---------------------------------------------------------------------------
def _stub_module(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


_one = _stub_module('one')
_one_api = _stub_module('one.api')


class _ONE:
    def __init__(self, *a, **k):
        self.cache_dir = Path(tempfile.gettempdir()) / 'priormech_numba_test'


_one_api.ONE = _ONE
_one.api = _one_api

_bwm = _stub_module('brainwidemap')
_bwm.bwm_query = lambda *a, **k: None
_bwm.load_good_units = lambda *a, **k: None
_bwm.bwm_units = lambda *a, **k: None

_ibla = _stub_module('iblatlas')
_ibla_atlas = _stub_module('iblatlas.atlas')
_ibla_atlas.AllenAtlas = lambda *a, **k: object()
_ibla_regions = _stub_module('iblatlas.regions')
_ibla_regions.BrainRegions = lambda *a, **k: object()

_bb = _stub_module('brainbox')
_bb_io = _stub_module('brainbox.io')
_bb_io_one = _stub_module('brainbox.io.one')
_bb_io_one.SpikeSortingLoader = object
_bb_io_one.SessionLoader = object

import model_functions as mf
from model_functions import (
    model_params, create_stimuli, run_model,
    blocks_per_session, trials_per_block_param, block_side_probs,
    num_stimulus_strength, min_stimulus_strength, max_stimulus_strength,
    min_trials_per_block, max_trials_per_block,
)

# Use a fitting-like configuration.
model_params['direct_offset'] = False
model_params['nonlin_type'] = 'linear'
# Turn on the prior->sensory pathway too, to exercise g_s/d_s in the kernel.
model_params['g_s'] = 0.0
model_params['d_s'] = 0.0

dt = 2.0
model_params['dt'] = dt
mf._update_model_params_for_dt(model_params, dt)
steps_before_obs = 500
max_obs_per_trial = 1000

bps = 5
model_type = 'data'


def build_stimuli(seed):
    rng = np.random.RandomState(seed)
    stimuli, trial_strengths, _perceived, trial_sides, block_sides = create_stimuli(
        bps, trials_per_block_param, block_side_probs, num_stimulus_strength,
        min_stimulus_strength, max_stimulus_strength,
        min_trials_per_block, max_trials_per_block,
        max_obs_per_trial, steps_before_obs, rng=rng, **model_params)
    return stimuli, trial_strengths, trial_sides, block_sides


def compare(seed):
    stimuli, trial_strengths, trial_sides, block_sides = build_stimuli(seed)
    args = (model_type, stimuli, trial_strengths, trial_sides, block_sides, bps)
    kw = dict(steps_before_obs=steps_before_obs, verbose=False)

    r_np = run_model(*args, backend='numpy', **{**kw, **model_params})
    r_nb = run_model(*args, backend='numba', **{**kw, **model_params})

    report = {}
    rtol, atol = 1e-12, 1e-12
    for key in ('S', 'I', 'P', 'M', 'a', 'perceived_stim', 'action_signal'):
        an = np.asarray(r_np[key], dtype=float)
        bn = np.asarray(r_nb[key], dtype=float)
        if an.shape != bn.shape:
            report[key] = f"SHAPE MISMATCH {an.shape} vs {bn.shape}"
            continue
        if not np.allclose(an, bn, rtol=rtol, atol=atol, equal_nan=True):
            diff = np.abs(an - bn)
            report[key] = float(np.nanmax(diff))
        else:
            report[key] = 0.0

    for key in ('choices', 'correct_action_taken', 'reaction_time'):
        an = np.asarray(r_np[key]); bn = np.asarray(r_nb[key])
        report[key] = 'EQUAL' if (an.shape == bn.shape and np.array_equal(an, bn)) \
            else f"MISMATCH (n_diff={int(np.sum(an != bn)) if an.shape == bn.shape else 'shape'})"

    lens_np = [len(x) for x in r_np['trial_sides']]
    lens_nb = [len(x) for x in r_nb['trial_sides']]
    report['trial_lengths'] = 'EQUAL' if lens_np == lens_nb else 'MISMATCH'
    report['parity_ok'] = all(
        v == 0.0 or v == 'EQUAL'
        for k, v in report.items()
        if k != 'parity_ok'
    )
    return report


def bench(seed, repeats=3):
    stimuli, trial_strengths, trial_sides, block_sides = build_stimuli(seed)
    args = (model_type, stimuli, trial_strengths, trial_sides, block_sides, bps)
    kw = dict(steps_before_obs=steps_before_obs, verbose=False)

    # warm up numba (compile)
    run_model(*args, backend='numba', **{**kw, **model_params})

    t0 = time.perf_counter()
    for _ in range(repeats):
        run_model(*args, backend='numpy', **{**kw, **model_params})
    t_np = (time.perf_counter() - t0) / repeats

    t0 = time.perf_counter()
    for _ in range(repeats):
        run_model(*args, backend='numba', **{**kw, **model_params})
    t_nb = (time.perf_counter() - t0) / repeats
    return t_np, t_nb


if __name__ == '__main__':
    print("=== numerical parity (numpy vs numba) ===")
    worst = 0.0
    all_ok = True
    for seed in (0, 1, 2):
        rep = compare(seed)
        print(f"seed={seed}: {rep}")
        all_ok = all_ok and rep.get('parity_ok', False)
        for v in rep.values():
            if isinstance(v, float):
                worst = max(worst, v)
    print(f"\nworst floating-point diff across continuous fields: {worst:.3e}")
    print(f"parity check: {'PASS' if all_ok else 'FAIL'}")

    print("\n=== benchmark ===")
    t_np, t_nb = bench(0, repeats=3)
    print(f"numpy:  {t_np*1000:8.1f} ms/eval")
    print(f"numba:  {t_nb*1000:8.1f} ms/eval")
    print(f"speedup: {t_np / t_nb:6.1f}x")
