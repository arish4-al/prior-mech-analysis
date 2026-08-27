# Testing / revising generative-model details

**Scope:** four modeling assumptions that shape the fitted I/M/P dynamics
and the form of prior influence. Each is currently baked into
`run_model` and/or the weights/joint loss; this journal is the place to
ablate them and decide what to keep.

**Not in scope:** retinal Stage A; prior-distance recovery on BWM data;
fit *speed* ([simulation_fit_speedups.md](simulation_fit_speedups.md)).
Fit *quality* after each ablation is in scope.

**Status:** tests 1–2 run (2026-08-27). Tests 3–4 **wired** (2026-08-27b),
not yet fit. **Regular only** (P→I/M; `g_s`/`d_s` frozen). Flags default
**off** (keep `W_pp` box, two thresholds). Test 1: keep the ITI gate.
Test 2: `W_ii`/`W_mm` do **not** slow as hypothesized; best noiti s34
fair **0.985** vs this-batch baseline **1.015**.

**Code:** dynamics in [`model_functions.py`](../model_functions.py)
(`run_model`, `prestim_offset_start`, `p_offset_always_on`,
`iti_penalty`, `action_thresholds`, ITI window constants). Loss:
`loss_plot_diff_by_condition_with_data` (I/M ITI penalty) plus
`loss_prior_effect`. Optimizers: [`fit_weights.py`](../fit_weights.py),
[`fit_joint.py`](../fit_joint.py). Drivers:
[`scripts/run_fit_joint.py`](../scripts/run_fit_joint.py)
(`--p-offset-always-on`, `--no-iti-penalty`, `--w-pp-lo`/`--w-pp-hi`/
`--set-w-pp`, `--tied-thresholds`);
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
this with `var_names=("I","P","M")`. Disable with
`iti_penalty=False` / `--no-iti-penalty`. Retinal fitting does not use
it.

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

- **Variant: regular only** (`--mtype regular --freeze 12,13`). Prior
  into I/M; `g_s`/`d_s` frozen ≈0. Do **not** run sensory unless
  explicitly asked. Baseline: Stage B `stageB_hold_s89` regular
  (winner **s101**).
- **Seeds (default):** `7 12 34 45 89 101 303 333` — 8 best regular
  shared-stim fair \(L_w+L_S\) from 2026-08-13.
- Compare at the same `bps` / freeze mask. Do not mix retinal-only
  `L_S` into these ablations unless the change is in S dynamics.
- Sticky without an explicit ask: S-bucket `<10` → NaN, fail-closed SSE,
  canonical prior-distance settings. These tests are about **model /
  loss terms**, not the BWM distance pipeline.
- `τ_Δ` above assumes linear nonlinearity (the fit default).
- Confirm journal/CLI option names before submitting ORCD jobs.

---

## Open questions

- Test 1: gate only `P_offset`, or also `P_gain`? Always-on from trial
  start vs always-on including the previous trial’s post-action tail.
  (2026-08-27: leaving `P_offset` on through the ITI is clearly worse;
  no reason to try `P_gain` until the offset gate is kept.)
- Test 2: 2026-08-27 — penalty does **not** pin `W_ii`/`W_mm` near the
  slow bound. Most seeds already have near-zero ITI activity without it.
  Remaining question: drop it from the default loss (best seed improves;
  median slightly worse; s7 much worse), or keep as a regularizer?
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
2026-08-12g + 2026-08-13 **regular** only (`12|13`, `g_s`/`d_s` frozen).
Winner **s101**. Sensory is not in this journal’s default.

**Seeds (8 best regular fair, 2026-08-13):**
`7 12 34 45 89 101 303 333`.

**Run dirs** (vs `weights_run_fj_stageB_hold_s89_regular_mask12-13_s<seed>/`):

- test 1: `…_fj_stageB_hold_s89_poffset_regular_mask12-13_s<seed>/`
- test 2: `…_fj_stageB_hold_s89_noiti_regular_mask12-13_s<seed>/`

8 jobs per ablation. Both tests = **16 jobs**.
Paste on ORCD (do not sbatch from the laptop):

```bash
# smoke one seed / one ablation
SEEDS=999 ABLATIONS=poffset OUT_TAG=stageB_ablate_poffset_smoke \
  DE1_MAXITER=2 DE2_MAXITER=3 POPSIZE=8 SOBOL_COUNT=4 \
  PATIENCE=0 LOCAL_REFINE_MAX_WALL_S=60 FORCE=1 TIME=1:00:00 \
  bash scripts/submit_fit_stage_b_model_ablations.sh

# production — test 1 then test 2 (16 jobs, regular only)
bash scripts/submit_fit_stage_b_model_ablations.sh

# or one arm
ABLATIONS=poffset bash scripts/submit_fit_stage_b_model_ablations.sh
ABLATIONS=noiti bash scripts/submit_fit_stage_b_model_ablations.sh
```

Fair compare later: same `bps=20` stim seed **12345** protocol as 12g/13
(`plot_best_fit_results.py` / shared-stim `L_w+L_S`). Tests 3–4 not wired.

### 2026-08-24c — regular only; recut seeds

Sensory dropped from this journal’s default (tests 1–2 and later).
Seeds recut to the 8 best **regular** fairs (dropped s23 / s56, which
were in the list for sensory; added s7 / s45). Driver default
`VARIANTS=regular:12|13`.

### 2026-08-27 — tests 1–2 results (regular, 8 seeds)

Local copies: alyx `models/new/`
`weights_run_fj_stageB_hold_s89_{poffset,noiti}_regular_mask12-13_s{7,12,34,45,89,101,303,333}/`.
All **16/16 `FIT_DONE`**. Flags in `run_fit_joint_report.json` /
`weights_final_*.json` match the arm (`poffset`: `p_offset_always_on`
true, `iti_penalty` true; `noiti`: the reverse). Baseline:
openalyx `weights_run_fj_stageB_hold_s89_regular_mask12-13_s<seed>/`.

**Eval:** same protocol as [retinal then joint](retinal_then_joint_fitting.md)
2026-08-12g/13 — `bps=20`, stim seed **12345**, nested `fit_targets/`,
`L_w` = traj+prior, **fair** = `L_w+L_S`. Stimuli built from baseline
**s101** (retinal ≈ Stage-A s89), not the hybrid JSON, so this-batch
baseline fairs sit ~0.01–0.04 above the 08-13 table (s101 **1.017** vs
journal **1.001**). Ablation vs baseline below is on **this** shared
batch. `HAVE_NUMBA=True`.

#### Column key

**Fit arm:** **base** = Stage B regular (gate on, ITI penalty on);
**poff** = test 1, fitted with `P_offset` always on; **noiti** = test 2,
fitted with the I/M ITI penalty off.

**Score:** **rec** = fitter’s recorded loss (own objective, own stim —
**not** comparable across arms, especially noiti rec vs base rec).
Everything else is shared-stim **fair** \(L_w+L_S\) on one `bps=20`
session. Compare **poff own** / **poff gated** to **base fair**, and
**noiti +iti** to **base fair**. Do not rank on rec, and do not treat
**noiti own** as beating baseline (penalty left out of the sum).

| Column | Fit | What is reported |
|--------|-----|------------------|
| **base rec** | baseline | fitter’s recorded loss |
| **base fair** | baseline | shared-stim \(L_w+L_S\) under production flags |
| **poff rec** | always-on `P_offset` | fitter’s recorded loss |
| **poff own** | always-on `P_offset` | shared-stim fair **as fitted** (offset still always on) |
| **poff gated** | always-on `P_offset` | same weights, eval with the **production gate** restored |
| **noiti rec** | no ITI penalty | fitter’s recorded loss (objective without the ITI term) |
| **noiti own** | no ITI penalty | shared-stim fair **as fitted** (penalty still off; optimistic vs base) |
| **noiti +iti** | no ITI penalty | same weights, eval with the **ITI penalty restored** (apples-to-apples vs base fair) |

#### Shared-stim fair

| seed | base rec | base fair | poff rec | poff own | poff gated | noiti rec | noiti own | noiti +iti |
|-----:|---------:|----------:|---------:|---------:|-----------:|----------:|----------:|-----------:|
| 7 | 0.996 | 1.076 | 1.428 | 1.566 | 1.809 | 1.096 | 1.228 | 1.323 |
| 12 | 0.960 | 1.026 | 2.040 | 1.608 | 1.842 | 1.016 | 1.139 | 1.159 |
| **34** | 0.952 | 1.026 | 1.695 | 1.855 | 1.858 | 0.927 | **0.977** | **0.985** |
| 45 | 1.524 | 1.114 | 2.512 | 1.667 | 3.587 | 1.520 | 1.006 | 1.017 |
| 89 | 1.149 | 1.095 | 1.605 | 1.660 | 1.949 | 1.083 | 1.114 | 1.118 |
| 101 | 1.131 | 1.017 | 1.905 | 1.818 | 2.395 | 1.206 | 1.125 | 1.128 |
| 303 | 1.030 | 1.083 | 1.613 | 1.880 | 2.022 | 1.134 | 0.990 | 1.049 |
| **333** | 0.962 | **1.015** | 1.846 | 1.970 | 2.519 | 0.905 | 0.994 | 1.047 |

| arm | best fair | median | mean |
|-----|----------:|-------:|-----:|
| baseline own | **1.015** (s333) | 1.051 | 1.056 |
| poffset own | 1.566 (s7) | 1.743 | 1.753 |
| poffset → gated | 1.809 (s7) | 1.986 | 2.248 |
| noiti own (no pen) | 0.977 (s34) | 1.060 | 1.072 |
| noiti +iti (canon) | **0.985** (s34) | 1.083 | 1.103 |

`L_S` stays ~0.496–0.498 on baseline and noiti (regular, `g_s`/`d_s`
frozen). Poffset s7 drifted to **0.616** and s12 to 0.514 (CMA/polish
unfreezes retinal).

#### Test 1 — always-on `P_offset`

Hypothesis: smearing P→I/M through the ITI should change `d_i`/`d_m`
and I/M shape.

**Yes on params, and the fit gets much worse.** Mean `g_i` 157 → 65;
mean `d_i` 26 → 11. Six of eight seeds collapse `d_i` from ~20–44 down
to 0.5–6; s7 is the exception (`d_i` 29 → 68) and is still the least-bad
poffset (fair 1.566 vs its baseline 1.076). `W_ii`/`W_mm` also drop on
average (0.418 → 0.373, 0.258 → 0.229). Evaluating the same weights
with the **gate restored** is worse still (median fair 1.99) — these
are not transferable to the production model.

Best poffset **s7** vs its baseline (native):

| | W_ii | W_pp | W_mm | W_is | W_pi | W_mi | g_i | d_i | g_m | θ_c | θ_d |
|--|-----:|-----:|-----:|-----:|-----:|-----:|----:|----:|----:|----:|----:|
| base s7 | 0.404 | 0.496 | 0.260 | 0.188 | 1.4e-5 | 0.622 | 200 | 28.6 | ≈0 | 0.745 | 0.473 |
| poff s7 | 0.413 | 0.498 | 0.264 | 0.176 | 1.3e-6 | 0.574 | 164 | **68.2** | ≈0 | 0.709 | 0.504 |

**Keep the ITI gate.** Always-on is not a better generative assumption
on this protocol.

#### Test 2 — drop I/M ITI zero-activity penalty

Hypothesis: without `4 × nanmean(|ITI|)` on I/M, `W_ii` / `W_mm` should
move toward the slow end of the box (`W_ii → 0.49`, `τ_I,Δ → 1 s`;
`W_mm → 0.40`, `τ_M,Δ → 100 ms`).

**Not supported.** Mean `W_ii` 0.418 vs 0.416; mean `W_mm` 0.258 vs
0.261. No seed hits the slow bound. `τ_I,Δ` stays ~90–170 ms (not 1 s);
`τ_M,Δ` stays ~30–53 ms. Restoring the penalty at eval adds almost
nothing to traj for s12/s34/s45/s89/s101 (Δ 0.003–0.021) — those fits
already sit near zero in `[−400, −100)` ms. s7/s303/s333 leak more
(Δ 0.05–0.10).

What *did* change: several seeds improved **prior** match. noiti s34
prior 0.106 vs baseline s34 0.211; fair **0.985** (traj 0.383 + prior
0.106 + `L_S` 0.497) beats this-batch baseline best **1.015**. noiti
s45 canon fair 1.017 matches baseline s101. Median canon fair is
slightly *worse* (1.083 vs 1.051) because s7 blows up (1.323) and
s12/s89/s101 do not beat their baselines. Side effect: noiti s89
`g_m` exploded to **187** (baseline ~0); s45 `g_m` to 2.93.

Best noiti **s34** vs its baseline:

| | W_ii | W_pp | W_mm | W_is | W_pi | W_mi | g_i | d_i | g_m | θ_c | θ_d |
|--|-----:|-----:|-----:|-----:|-----:|-----:|----:|----:|----:|----:|----:|
| base s34 | 0.402 | 0.496 | 0.300 | 0.178 | 2.3e-5 | 0.492 | 166 | 22.4 | ≈0 | 0.768 | 0.408 |
| noiti s34 | 0.403 | 0.497 | 0.287 | 0.199 | 1.7e-5 | 0.483 | 190 | 21.3 | ≈0 | 0.750 | 0.400 |

Weights are almost the same; the win is `L_w` (prior), not a slower I/M.

**Do not drop the penalty by default yet** (median not better; s7
fails). It is also **not** the reason I/M stay fast — something else
in the trial SSE already keeps ITI activity down. Next: decide whether
to keep it as a cheap regularizer, or drop it and add a different
constraint if s7-style leaks show up in trajectories.

#### Takeaways

1. Test 1 is a clear keep-the-gate. Always-on `P_offset` collapses
   `g_i`/`d_i` on most seeds and costs ~0.7 fair.
2. Test 2’s slow-`W_ii`/`W_mm` story is wrong. The ITI penalty is not
   what holds those weights in the middle of the box.
3. Dropping the penalty can find a **better** shared-stim regular fit
   (s34 fair 0.985). That is a loss-function change, not a dynamics
   change — only adopt it if we accept the mixed seed reliability.
4. Tests 3–4 wired 2026-08-27b (`W_pp` box / one vs two thresholds);
   not yet fit.

Eval dump: `scripts/_tmp_ablation_fair_eval.py` /
`scripts/_tmp_ablation_fair_eval.json` (untracked).

### 2026-08-27b — tests 3–4 wired (regular, same 8 seeds)

Same Stage B protocol as tests 1–2 (`stageB_hold_s89` hybrid, regular
`12|13`, `bps=20`, seeds `7 12 34 45 89 101 303 333`). ITI gate and
ITI penalty stay at production defaults (do **not** combine with tests
1–2). Driver default `ABLATIONS` is now tests 3–4.

**Test 3 — `W_pp` box / init** (`τ_P,Δ = 20/(1−2W_pp)` ms)

| Arm | `ABLATIONS=` | Init `W_pp` | Bounds | `τ_P,Δ` at init |
|-----|----------------|------------:|--------|----------------:|
| large + keep box | `wpplarge` | **0.499** | `[0.496, 0.49999]` (current) | **10 s** |
| large + open floor | `wppopen` | **0.499** | `[0.20, 0.49999]` | **10 s** |
| small + open floor | `wppsmall` | **0.45** | `[0.20, 0.49999]` | **200 ms** |

CLI: `--set-w-pp`, `--w-pp-lo`, `--w-pp-hi`. Open floor matches the
`W_ii` box (0.20); upper stays `< 0.5` (difference mode diverges at
½). Small init is the journal’s 100–500 ms example at 200 ms. `wpplarge`
is the control: same box as baseline, start at the slow end instead of
hybrid `W_pp≈0.496` (2.5 s). Compare fitted `W_pp` / `τ_P,Δ`, wall, and
shared-stim fair to baseline.

**Test 4 — one action threshold** (`ABLATIONS=onethr`)

`--tied-thresholds`: `theta_c = theta_d` every eval (loky via
`loss_extra_kwargs`). Freezes index **11** (`theta_d`); run dir slug
`mask11-12-13`. Init both to the **mean** of hybrid `theta_c` and
`theta_d`. Scalar `run_model` branch is unused — the dict is kept, both
sides equal. Compare `L_w`, psychometric / RT, I/M traj to two-θ
baseline.

**Run dirs**

- `weights_run_fj_stageB_hold_s89_wpplarge_regular_mask12-13_s<seed>/`
- `…_wppopen_regular_mask12-13_s<seed>/`
- `…_wppsmall_regular_mask12-13_s<seed>/`
- `…_onethr_regular_mask11-12-13_s<seed>/`

**32 jobs** (4 arms × 8 seeds). Paste on ORCD (do not sbatch from the
laptop). Code for this must be on **`main`** first.

```bash
# smoke one seed / one arm
SEEDS=999 ABLATIONS=wppsmall OUT_TAG=stageB_ablate_wppsmall_smoke \
  DE1_MAXITER=2 DE2_MAXITER=3 POPSIZE=8 SOBOL_COUNT=4 \
  PATIENCE=0 LOCAL_REFINE_MAX_WALL_S=60 FORCE=1 TIME=1:00:00 \
  bash scripts/submit_fit_stage_b_model_ablations.sh

# production — tests 3–4 (32 jobs, regular only)
bash scripts/submit_fit_stage_b_model_ablations.sh

# or one arm
ABLATIONS=wpplarge bash scripts/submit_fit_stage_b_model_ablations.sh
ABLATIONS=wppopen bash scripts/submit_fit_stage_b_model_ablations.sh
ABLATIONS=wppsmall bash scripts/submit_fit_stage_b_model_ablations.sh
ABLATIONS=onethr bash scripts/submit_fit_stage_b_model_ablations.sh
```

Fair compare later: same `bps=20` stim seed **12345**. Tests 1–2:
`ABLATIONS="poffset noiti"`.

**Local smoke (2026-08-27, seed 999, tiny DE/CMA):** `--set-w-pp` /
`--tied-thresholds` apply only on **external** warm start (in-folder
resume would average `theta_c` with freeze-fill `LOG_ZERO` ≈ 0 and
halve θ). One-eval: hybrid `L≈1.028`; `W_pp=0.45` → `L≈4.95`; tied
mean θ `0.581` → `L≈1.53`. Tiny fits `FIT_DONE`: wppsmall final
`W_pp=0.482` (below the 0.496 floor); onethr `theta_c=theta_d=0.568`,
`frozen_idx=[11,12,13]`.

