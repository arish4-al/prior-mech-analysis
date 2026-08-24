# Testing / revising generative-model details

**Scope:** four modeling assumptions that shape the fitted I/M/P dynamics
and the form of prior influence. Each is currently baked into
`run_model` and/or the weights/joint loss; this journal is the place to
ablate them and decide what to keep.

**Not in scope:** retinal Stage A; prior-distance recovery on BWM data;
fit *speed* ([simulation_fit_speedups.md](simulation_fit_speedups.md)).
Fit *quality* after each ablation is in scope.

**Status:** tests 1–2 wired (2026-08-24b). Flags default **off** (gated
`P_offset` + ITI penalty still on). Stage B refits with the 15
`stageB_hold_s89` seeds are ready to submit — not run yet.

**Code:** dynamics in [`model_functions.py`](../model_functions.py)
(`run_model`, `prestim_offset_start`, `p_offset_always_on`,
`iti_penalty`, `action_thresholds`, ITI window constants). Loss:
`loss_plot_diff_by_condition_with_data` (I/M ITI penalty) plus
`loss_prior_effect`. Optimizers: [`fit_weights.py`](../fit_weights.py),
[`fit_joint.py`](../fit_joint.py). Drivers:
[`scripts/run_fit_joint.py`](../scripts/run_fit_joint.py)
(`--p-offset-always-on`, `--no-iti-penalty`);
[`scripts/submit_fit_stage_b_model_ablations.sh`](../scripts/submit_fit_stage_b_model_ablations.sh).

---

## Current implementation (what we would change)

### 1. Prior → other-node feedback is gated off for most of the ITI

Additive P offset into S/I/M is **off** until 100 ms before stimOn, then
**on** through the trial (including the 40 ms post-action tail). Gate:

```python
if k >= (steps_before_obs - prestim_offset_start):  # prestim_offset_start = 100 ms
    P_offset = J @ P
else:
    P_offset = 0
```

(`model_functions.py`; same in numba / torch.) There is no separate
“turn off at movement” flag: the trial ends `post_action_steps` after
commit, and the next trial’s ITI starts with `P_offset = 0`.

Gain modulation (`g_* * P_gain`) is **not** gated this way. During the
ITI, S is near zero so `g_i P_gain @ S` is small anyway.

`dt = 2` ms → `prestim_offset_start = 50` steps.

**Hypothesis:** this off-window is crucial for the *shape* of prior
influence in I/M (and S if `d_s`/`g_s` are free). Leaving P→I/M on
through the ITI should smear prior modulation into the pre-stim epoch
and change fitted `d_i`/`d_m` / trajectories.

**To try:** always `P_offset = J @ P` via `model_params['p_offset_always_on']=True`
 / `--p-offset-always-on` (numpy, numba, and torch). Compare I/M (and P)
condition diffs and `L_w` vs the gated baseline. Optionally also gate
`P_gain` (not implemented).

### 2. I/M ITI zero-activity penalty (−400 → −100 ms)

Still in the fitting loss (confirmed 2026-08-24). For I and M,
`mean_by_condition` averages `[−400, −100)` ms before stimOn, split by
previous choice (`iti_prev`). The loss adds

`4 × nanmean(|trace|)`

per previous-choice stratum to I, M, and `total`
(`loss_plot_diff_by_condition_with_data`). Weights and joint both call
this with `var_names=("I","P","M")`. There is no disable flag. Retinal
fitting does not use it.

Effective recurrent timescale (difference mode, linear, `J = [[1,−1],[−1,1]]`,
`τ = 20` ms):

`τ_Δ ≈ τ / (1 − 2 W)`

| Weight | Fit bounds | `τ_Δ` at bounds |
|--------|------------|-----------------|
| `W_ii` | `[0.20, 0.49]` | ~33 ms → **1.0 s** |
| `W_mm` | `[0.10, 0.40]` | ~25 ms → **100 ms** |
| `W_pp` | `[0.496, 0.49999]` | **2.5 s → ~10³ s** |

**Hypothesis:** the ITI penalty is a major reason the optimizer keeps
`W_ii` / `W_mm` from sitting at the slow end of those boxes (activity
must die before −100 ms). Removing it should slow fitted I and/or M
(`W_ii`, `W_mm` up) and change `L_w`.

**To try:** `model_params['iti_penalty']=False` / `--no-iti-penalty`
skips the `iti_prev` term (data-matching pre-action SSE and P penalties
stay). Refit Stage B (same protocol / seeds as
[retinal then joint](retinal_then_joint_fitting.md) `stageB_hold_s89`)
and compare `W_ii`, `W_mm`, I/M traces, and loss.

### 3. Prior internal feedback is boxed to a slow integrator

`τ_p` is **frozen** at 20 ms. Slow P comes entirely from `W_pp` near
½ (`τ_P,Δ ≈ τ_p / (1 − 2 W_pp)`). Bound `W_pp ∈ [0.496, 0.49999]` ⇒
`τ_P,Δ ∈ [2.5 s, ~1000 s]`. (The intended floor is often quoted as
**≥ 4 s**; the actual lower edge of the box is **2.5 s** at
`W_pp = 0.496`. `W_pp = 0.4975` is 4 s.)

Older code fitted `tau_p ∈ [1000, 2000]` ms; that is commented out.

**Hypothesis:** the box guarantees a long prior integration constant.
Starting at a *small* `τ_P` (if the bound is opened) vs a *large* one
should differ in speed and final `L_w` — one init should win.

**To try:** two inits, otherwise identical weights (or joint) protocol.

| Arm | Init | Bound |
|-----|------|--------|
| large (current-like) | `W_pp ≈ 0.499` (`τ_P ~ 10 s`) | keep `[0.496, 0.49999]` **and** a run with the box opened |
| small | `W_pp` well below 0.496 (e.g. `τ_P ~ 100–500 ms`) | must loosen/remove the `W_pp` floor |

Report wall-clock / evals to a loss threshold, final `L_w`, and fitted
`W_pp` / `τ_P,Δ`. Do **not** change the I/M ITI penalty in this arm
unless combining with test 2.

### 4. Two action thresholds (concordant vs discordant)

`action_thresholds` is a dict: one value for perceived-concordant trials
(`theta_c`, stim-difference sign matches P), one for discordant
(`theta_d`). Same value is reused across contrasts. `run_model` already
accepts a **scalar** threshold for all trials.

Defaults / typical fit inits: `theta_c ≈ 0.78–0.91`, `theta_d ≈ 0.54`.
Bounds: both `(0.1, 0.99999)`. Packed as two log-space params (indices
10–11 in the 12-d weights vector).

**Hypothesis:** two thresholds may be unnecessary for a good `L_w`.

**To try:** tie `theta_c = theta_d` (one free param), or pass a scalar
`action_thresholds`. Refit; compare `L_w`, psychometric / RT, and I/M
trajectories to the two-threshold baseline. The scalar branch in
`run_model` is already there.

---

## Shared protocol notes

- Compare against the current weights (and, if relevant, joint) baseline
  at the same `bps`, seeds, and freeze mask. Do not mix retinal-only
  `L_S` into these ablations unless the change is in S dynamics (test 1
  with `d_s`/`g_s` free).
- Sticky without an explicit ask: S-bucket `<10` → NaN, fail-closed SSE,
  canonical prior-distance settings. These tests are about **model /
  loss terms**, not the BWM distance pipeline.
- `τ_Δ` above assumes linear nonlinearity (the fit default).
- Confirm journal/CLI option names before submitting ORCD jobs.

---

## Open questions

- Test 1: gate only `P_offset`, or also `P_gain`? Always-on from trial
  start vs always-on including the previous trial’s post-action tail.
- Test 2: does the penalty mainly hit `W_ii`/`W_mm`, or also `d_i`/`d_m`
  (offset that would otherwise leak into the ITI)?
- Test 3: is the scientifically intended floor 4 s (`W_pp ≥ 0.4975`) or
  the code’s 2.5 s (`W_pp ≥ 0.496`)? Should `tau_p` be unfrozen instead
  of boxing `W_pp`?
- Test 4: one threshold vs two, vs contrast-dependent thresholds (the
  dict is already per-contrast but fits share one value per
  conc/disc).

---

## Dated entries

### 2026-08-24 — journal started

Four tests listed from modeling review. Confirmed the −400→−100 ms I/M
ITI penalty is still in `loss_plot_diff_by_condition_with_data` and is
used by `fit_weights` / `fit_joint`. No code or jobs yet.

### 2026-08-24b — tests 1–2 implemented; Stage B seeds ready

Defaults unchanged: `p_offset_always_on=False`, `iti_penalty=True`.

| Flag | CLI | What it does |
|------|-----|----------------|
| `p_offset_always_on` | `--p-offset-always-on` | `P_offset = J @ P` every step (numpy / numba / torch) |
| `iti_penalty` | `--no-iti-penalty` sets False | skip `4 × nanmean(\|ITI\|)` on I/M |

Loky CMA workers re-import `model_params` at defaults, so the flags ride
`loss_extra_kwargs` into every eval (`apply_model_ablation_flags` inside
`loss_joint_core` / `loss_weights_core_v2`). Saved JSON `model_params`
records both.

**Compare to** [retinal then joint](retinal_then_joint_fitting.md)
2026-08-12g + 2026-08-13: hybrid WEIGHTS_REL ∪ retinal s89,
`--stage1-hold-retinal`, regular `12\|13` + sensory `6\|7\|8\|9`,
`bps=20/20`, `OUT_TAG=stageB_hold_s89`. Winners regular **s101**,
sensory **s23**.

**Seeds (15):** `7 12 23 34 42 45 56 67 78 89 101 111 202 303 333`
(batch-1 `56 34 78 89 202` ∪ batch-2 `7 12 23 42 45 67 101 111 303 333`).

**Run dirs** (vs `weights_run_fj_stageB_hold_s89_<mtype>_mask<slug>_s<seed>/`):

- test 1: `…_fj_stageB_hold_s89_poffset_<mtype>_mask<slug>_s<seed>/`
- test 2: `…_fj_stageB_hold_s89_noiti_<mtype>_mask<slug>_s<seed>/`

30 jobs per ablation (15 seeds × 2 variants). Paste on ORCD (do not
sbatch from the laptop):

```bash
# smoke one seed / one ablation
SEEDS=999 ABLATIONS=poffset OUT_TAG=stageB_ablate_poffset_smoke \
  DE1_MAXITER=2 DE2_MAXITER=3 POPSIZE=8 SOBOL_COUNT=4 \
  PATIENCE=0 LOCAL_REFINE_MAX_WALL_S=60 FORCE=1 TIME=1:00:00 \
  bash scripts/submit_fit_stage_b_model_ablations.sh

# production — test 1 then test 2 (60 jobs)
bash scripts/submit_fit_stage_b_model_ablations.sh

# or one arm
ABLATIONS=poffset bash scripts/submit_fit_stage_b_model_ablations.sh
ABLATIONS=noiti bash scripts/submit_fit_stage_b_model_ablations.sh
```

Fair compare later: same `bps=20` stim seed **12345** protocol as 12g/13
(`plot_best_fit_results.py` / shared-stim `L_w+L_S`). Tests 3–4 not wired.

