# Testing / revising generative-model details

**Scope:** four modeling assumptions that shape the fitted I/M/P dynamics
and the form of prior influence. Each is currently baked into
`run_model` and/or the weights/joint loss; this journal is the place to
ablate them and decide what to keep.

**Not in scope:** retinal Stage A; prior-distance recovery on BWM data;
fit *speed* ([simulation_fit_speedups.md](simulation_fit_speedups.md)).
Fit *quality* after each ablation is in scope.

**Status:** tests 1–6 run (2026-08-27 / 08-27c / 08-31 / 09-07 / 09-08).
**Regular only** (P→I/M; `g_s`/`d_s` frozen). Defaults stay:
ITI gate on, ITI penalty on, `W_pp` box `[0.496, 0.49999]`, two action
thresholds, `m_pre_weight=1`, `prior_window_ms` unset (legacy T=72 /
plot_window=80), `prior_stratum` unset (stim×choice). **Restore `g_i`
floor `0.1`** — opening it to `1e-12` (09-08d/e) made stim-only collapse
worse and did not beat regular. Test 1: keep the ITI gate. Test 2:
`W_ii`/`W_mm` do **not** slow as hypothesized. Test 3: keep the 2.5 s
`W_pp` floor. Test 4: keep two thresholds. Test 5: 3× pre-action M does
**not** beat regular’s best pooled act-prior RT R² (mpre3 s303 **0.772**
vs regular s101 **0.787**). Test 6 / `im150stim` / `stimonly`: none beat
regular on production eval tot (baseline median **1.051**).

**Code:** dynamics in [`model_functions.py`](../model_functions.py)
(`run_model`, `prestim_offset_start`, `p_offset_always_on`,
`iti_penalty`, `action_thresholds`, ITI window constants). Loss:
`loss_plot_diff_by_condition_with_data` (I/M ITI penalty) plus
`loss_prior_effect`. Optimizers: [`fit_weights.py`](../fit_weights.py),
[`fit_joint.py`](../fit_joint.py). Drivers:
[`scripts/run_fit_joint.py`](../scripts/run_fit_joint.py)
(`--p-offset-always-on`, `--no-iti-penalty`, `--w-pp-lo`/`--w-pp-hi`/
`--set-w-pp`, `--tied-thresholds`, `--m-pre-weight`, `--prior-window-ms`,
`--prior-stratum`);
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
  eval total loss from 2026-08-13.
- **How we score:** rank on **eval total loss** (re-run on one shared
  `bps=20` session, stim seed 12345). The JSON `final_loss` is the same
  objective on that job’s own stim — not comparable across seeds or
  arms.
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
- Test 3: 2026-08-27c — loss sits on the **2.5 s** edge (`W_pp=0.496`),
  not 10 s and not a 4 s (`W_pp=0.4975`) floor. Opening the floor lets
  some seeds go sub-second but **hurts** eval total loss. Remaining: unfreeze
  `tau_p` instead of boxing `W_pp`? Not needed to keep P slow if the
  0.496 floor stays.
- Test 4: 2026-08-27c — one shared θ is **not** enough (`L_w` prior
  collapses). Contrast-dependent thresholds still open (the dict is
  already per-contrast but fits share one value per conc/disc).

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

**Seeds (8 best regular eval total loss, 2026-08-13):**
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

Score later on eval total loss: same `bps=20` stim seed **12345**
protocol as 12g/13 (`plot_best_fit_results.py`). Tests 3–4 not wired.

### 2026-08-24c — regular only; recut seeds

Sensory dropped from this journal’s default (tests 1–2 and later).
Seeds recut to the 8 best **regular** eval total loss (dropped s23 / s56, which
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
2026-08-12g/13 — `bps=20`, stim seed **12345**, nested `fit_targets/`.
Rank on **eval total loss** (traj + prior + S, one shared session).
Stimuli built from baseline **s101** (retinal ≈ Stage-A s89), not the
hybrid JSON, so this-batch baseline eval total sits ~0.01–0.04 above
the 08-13 table (s101 **1.017** vs journal **1.001**). Ablation vs
baseline below is on **this** shared batch. `HAVE_NUMBA=True`.

#### Column key

**Fit arm:** **base** = Stage B regular (gate on, ITI penalty on);
**poff** = test 1, fitted with `P_offset` always on; **noiti** = test 2,
fitted with the I/M ITI penalty off.

**Score:** **fit** = JSON `final_loss` (own stim — **not** comparable
across arms). **eval tot** = eval total loss on the shared session.
Compare **poff own** / **poff gated** to **base eval**, and
**noiti +iti** to **base eval**. Do not rank on fit loss, and do not
treat **noiti own** as beating baseline (penalty left out of the sum).

| Column | Fit | What is reported |
|--------|-----|------------------|
| **base fit** | baseline | JSON `final_loss` (own stim) |
| **base eval** | baseline | eval total loss under production flags |
| **poff fit** | always-on `P_offset` | JSON `final_loss` |
| **poff own** | always-on `P_offset` | eval total **as fitted** (offset still always on) |
| **poff gated** | always-on `P_offset` | same weights, eval with the **production gate** restored |
| **noiti fit** | no ITI penalty | JSON `final_loss` (objective without the ITI term) |
| **noiti own** | no ITI penalty | eval total **as fitted** (penalty still off; optimistic vs base) |
| **noiti +iti** | no ITI penalty | same weights, eval with the **ITI penalty restored** (same objective as base eval) |

#### Eval total loss

| seed | base fit | base eval | poff fit | poff own | poff gated | noiti fit | noiti own | noiti +iti |
|-----:|---------:|----------:|---------:|---------:|-----------:|----------:|----------:|-----------:|
| 7 | 0.996 | 1.076 | 1.428 | 1.566 | 1.809 | 1.096 | 1.228 | 1.323 |
| 12 | 0.960 | 1.026 | 2.040 | 1.608 | 1.842 | 1.016 | 1.139 | 1.159 |
| **34** | 0.952 | 1.026 | 1.695 | 1.855 | 1.858 | 0.927 | **0.977** | **0.985** |
| 45 | 1.524 | 1.114 | 2.512 | 1.667 | 3.587 | 1.520 | 1.006 | 1.017 |
| 89 | 1.149 | 1.095 | 1.605 | 1.660 | 1.949 | 1.083 | 1.114 | 1.118 |
| 101 | 1.131 | 1.017 | 1.905 | 1.818 | 2.395 | 1.206 | 1.125 | 1.128 |
| 303 | 1.030 | 1.083 | 1.613 | 1.880 | 2.022 | 1.134 | 0.990 | 1.049 |
| **333** | 0.962 | **1.015** | 1.846 | 1.970 | 2.519 | 0.905 | 0.994 | 1.047 |

| arm | best eval tot | median | mean |
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
poffset (eval tot 1.566 vs its baseline 1.076). `W_ii`/`W_mm` also drop on
average (0.418 → 0.373, 0.258 → 0.229). Evaluating the same weights
with the **gate restored** is worse still (median eval tot 1.99) — these
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
prior 0.106 vs baseline s34 0.211; eval tot **0.985** (traj 0.383 + prior
0.106 + S 0.497) beats this-batch baseline best **1.015**. noiti
s45 canon eval tot 1.017 matches baseline s101. Median canon eval tot is
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
   `g_i`/`d_i` on most seeds and costs ~0.7 eval tot.
2. Test 2’s slow-`W_ii`/`W_mm` story is wrong. The ITI penalty is not
   what holds those weights in the middle of the box.
3. Dropping the penalty can find a **better** eval total loss
   (s34 **0.985**). That is a loss-function change, not a dynamics
   change — only adopt it if we accept the mixed seed reliability.
4. Tests 3–4 results: 2026-08-27c.

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
eval total loss to baseline.

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

Score later on eval total loss: same `bps=20` stim seed **12345**. Tests 1–2:
`ABLATIONS="poffset noiti"`.

**Local smoke (2026-08-27, seed 999, tiny DE/CMA):** `--set-w-pp` /
`--tied-thresholds` apply only on **external** warm start (in-folder
resume would average `theta_c` with freeze-fill `LOG_ZERO` ≈ 0 and
halve θ). One-eval: hybrid `L≈1.028`; `W_pp=0.45` → `L≈4.95`; tied
mean θ `0.581` → `L≈1.53`. Tiny fits `FIT_DONE`: wppsmall final
`W_pp=0.482` (below the 0.496 floor); onethr `theta_c=theta_d=0.568`,
`frozen_idx=[11,12,13]`.

### 2026-08-27c — tests 3–4 results (regular, 8 seeds)

Local copies: alyx `models/new/`
`weights_run_fj_stageB_hold_s89_{wpplarge,wppopen,wppsmall}_regular_mask12-13_s{7,12,34,45,89,101,303,333}/`
and `…_onethr_regular_mask11-12-13_s{…}/`. All **32/32 `FIT_DONE`**.
Flags in `run_fit_joint_report.json` match the arm (`set_w_pp` /
`w_pp_bounds` / `tied_thresholds`; onethr freezes `11,12,13`). Same
this-batch baseline and shared-stim protocol as 2026-08-27 (`bps=20`,
seed **12345**, stim from baseline s101). Eval is **as fitted** (test 3
same dynamics as baseline; test 4 tied θ vs two-θ baseline).
`HAVE_NUMBA=True`.

#### Column key

**Fit arm:** **base** = Stage B regular (box `[0.496, 0.49999]`, two
θ); **large** = init `W_pp=0.499` (10 s), same box; **open** = init
0.499, floor **0.20**; **small** = init **0.45** (200 ms), floor 0.20;
**1thr** = `theta_c = theta_d`, freeze idx 11.

**Score:** **fit** = JSON `final_loss` (own stim — not comparable
across arms). **eval tot** = eval total loss on the shared session.
Rank on eval tot.

| Column | Fit | What is reported |
|--------|-----|------------------|
| **base fit / eval** | baseline | JSON `final_loss`; eval total loss |
| **large fit / eval** | `wpplarge` | same box, slow init |
| **open fit / eval** | `wppopen` | opened floor, slow init |
| **small fit / eval** | `wppsmall` | opened floor, fast init |
| **1thr fit / eval** | `onethr` | tied thresholds, production `W_pp` box |

#### Eval total loss

| seed | base fit | base eval | large fit | large eval | open fit | open eval | small fit | small eval | 1thr fit | 1thr eval |
|-----:|---------:|----------:|----------:|-----------:|---------:|----------:|----------:|-----------:|---------:|----------:|
| 7 | 0.996 | 1.076 | 1.061 | 1.056 | 1.116 | 1.055 | 1.173 | 1.133 | 1.436 | 1.505 |
| 12 | 0.960 | 1.026 | 1.322 | 1.058 | 1.361 | 1.208 | 1.460 | 1.089 | 3.789 | 1.189 |
| 34 | 0.952 | 1.026 | 0.916 | 1.314 | 1.796 | 1.189 | 1.485 | 1.112 | 1.961 | 2.871 |
| 45 | 1.524 | 1.114 | 1.557 | 1.097 | 1.731 | 1.286 | 1.647 | 1.128 | 2.869 | 1.642 |
| 89 | 1.149 | 1.095 | 1.778 | 1.073 | 1.097 | 1.128 | 0.875 | **1.011** | 2.719 | 2.673 |
| **101** | 1.131 | 1.017 | 1.660 | **0.947** | 1.317 | 1.096 | 1.342 | 1.118 | 1.448 | 2.601 |
| 303 | 1.030 | 1.083 | 1.037 | 1.003 | 1.020 | 1.276 | 1.019 | 1.149 | 1.490 | 3.051 |
| **333** | 0.962 | **1.015** | 1.252 | 1.034 | 1.057 | 1.129 | 1.050 | 1.196 | 2.072 | 2.747 |

| arm | best eval tot | median | mean |
|-----|----------:|-------:|-----:|
| baseline | **1.015** (s333) | 1.051 | 1.056 |
| wpplarge | **0.947** (s101) | 1.057 | 1.073 |
| wppopen | 1.055 (s7) | 1.159 | 1.171 |
| wppsmall | 1.011 (s89) | 1.123 | 1.117 |
| onethr | 1.189 (s12) | 2.637 | 2.285 |

`L_S` stays ~0.496–0.498 except wppopen s12 **0.574** and s303 **0.556**
(CMA/polish unfreezes retinal). onethr traj is only mildly worse
(~0.40–0.59 vs baseline ~0.32–0.41); the blow-up is **prior** (median
1.70 vs 0.21).

#### Fitted `W_pp` / `τ_P,Δ`

`τ_P,Δ = 20/(1−2W_pp)` ms. Production floor `W_pp=0.496` is **2.50 s**.

| seed | base `W_pp` | tP_s | large | tP_s | open | tP_s | small | tP_s | 1thr | tP_s |
|-----:|------------:|-----:|------:|-----:|-----:|-----:|------:|-----:|-----:|-----:|
| 7 | 0.49616 | 2.60 | 0.49840 | **6.24** | 0.49568 | 2.31 | 0.49714 | 3.50 | 0.49689 | 3.22 |
| 12 | 0.49600 | 2.50 | 0.49635 | 2.74 | 0.49846 | 6.48 | 0.49754 | 4.06 | 0.49638 | 2.76 |
| 34 | 0.49619 | 2.62 | 0.49681 | 3.14 | 0.49602 | 2.51 | **0.48917** | **0.92** | 0.49629 | 2.69 |
| 45 | 0.49820 | 5.55 | 0.49683 | 3.16 | **0.48809** | **0.84** | 0.49568 | 2.31 | 0.49650 | 2.85 |
| 89 | 0.49633 | 2.73 | 0.49849 | **6.60** | 0.49656 | 2.91 | 0.49326 | 1.48 | 0.49655 | 2.90 |
| 101 | 0.49619 | 2.63 | 0.49602 | 2.51 | 0.49438 | 1.78 | 0.49602 | 2.52 | 0.49637 | 2.75 |
| 303 | 0.49733 | 3.74 | 0.49600 | 2.50 | 0.49190 | 1.23 | 0.49659 | 2.93 | 0.49800 | 5.01 |
| 333 | 0.49613 | 2.59 | 0.49652 | 2.87 | 0.49551 | 2.23 | **0.47813** | **0.46** | 0.49665 | 2.98 |

| arm | `W_pp` median | `τ_P,Δ` median | below 0.496 |
|-----|-------------:|---------------:|------------:|
| baseline | 0.49619 | 2.62 s | 0/8 |
| wpplarge | 0.49666 | 3.00 s | 0/8 (box) |
| wppopen | 0.49559 | 2.27 s | **5/8** |
| wppsmall | 0.49585 | 2.42 s | **4/8** |
| onethr | 0.49653 | 2.88 s | 0/8 (box) |

No open-floor seed went below `W_pp=0.45` (200 ms). Fastest is wppsmall
s333 at **460 ms**.

#### Test 3 — `W_pp` box / init

Hypothesis: large vs small init, and keeping vs opening the 2.5 s
floor, should change fitted `W_pp` / `τ_P,Δ` and `L_w`.

**Init does not pin the timescale; the floor does.** Starting at 10 s
with the box kept (`wpplarge`), 6/8 seeds fall back to ~2.5–3 s.
Only s7 / s89 stay slow (~6 s). Median eval tot **1.057** matches baseline
**1.051**. Best is wpplarge **s101** eval tot **0.947** (traj 0.336 + prior
0.114 + S 0.497) vs its baseline 1.017 — still at the **floor**
(`W_pp=0.49602`, 2.51 s), so this is a better optimizer start on the
same model, not a slower prior. s34 is a miss (+0.289 eval tot).

Opening the floor **hurts** median eval tot (open 1.159, small 1.123).
From a slow init, 5/8 go below 0.496 (down to 0.84 s). From a 200 ms
init, two stay fast (s34 0.92 s, s333 0.46 s) and the rest **climb**
to ~1.5–4 s — they do not stay at 200 ms, and they do not all meet
the boxed runs at 2.5 s. Fast P is allowed and used, but it does not
win `L_w`.

A 4 s scientific floor (`W_pp ≥ 0.4975`) would be **tighter** than
where the loss sits (baseline / large medians 2.6–3.0 s). Keep the
code’s **2.5 s** box. Do not open the floor by default.

Best wpplarge **s101** vs its baseline:

| | W_ii | W_pp | W_mm | W_is | W_pi | W_mi | g_i | d_i | θ_c | θ_d |
|--|-----:|-----:|-----:|-----:|-----:|-----:|----:|----:|----:|----:|
| base s101 | 0.425 | 0.496 | 0.255 | 0.157 | 1.7e-5 | 0.576 | 196 | 20.0 | 0.768 | 0.389 |
| large s101 | 0.414 | 0.496 | 0.266 | 0.188 | 1.6e-5 | 0.515 | 189 | 26.3 | 0.729 | 0.443 |

Weights are close; the win is traj (0.336 vs 0.414), not a
different `W_pp`.

#### Test 4 — one action threshold

Hypothesis: `theta_c = theta_d` may be enough for a good `L_w`.

**Not supported.** Median eval tot **2.637** vs baseline **1.051**. Best
onethr s12 **1.189** is still worse than the *worst* baseline (s45
1.114). Tied θ sits at **0.52–0.62** (median 0.563) — the midpoint of
baseline’s two values (`θ_c` ~0.74, `θ_d` ~0.41, gap 0.27–0.39).
`g_i` collapses (median **5.2** vs 180). Traj is only ~0.1 worse;
**prior** is 0.25–2.14 (six of eight > 0.7). `L_S` stays ~0.497.

Best onethr **s12** vs its baseline:

| | W_ii | W_pp | W_mm | g_i | d_i | θ_c | θ_d | traj | prior | eval tot |
|--|-----:|-----:|-----:|----:|----:|----:|----:|-----:|------:|-----:|
| base s12 | 0.426 | 0.496 | 0.261 | 188 | 22.9 | 0.742 | 0.427 | 0.401 | 0.129 | 1.026 |
| 1thr s12 | 0.377 | 0.496 | 0.211 | **1.35** | 50.3 | **0.524** | **0.524** | 0.437 | 0.255 | 1.189 |

**Keep two thresholds.** Concordant vs discordant commit levels are
not interchangeable on this protocol.

#### Takeaways

1. Test 3: keep `W_pp ∈ [0.496, 0.49999]`. The loss wants the **2.5 s**
   edge, not 10 s. Opening the floor allows 0.5–2 s P and costs
   ~0.07–0.11 median eval tot.
2. Slow vs fast *init* inside an open box does not pick one winner
   timescale — seeds split. The box is what makes P uniformly slow.
3. wpplarge s101 eval tot **0.947** is the best regular number
   in this whole four-test batch (beats noiti s34 0.985), but it is
   still the boxed model. Treat as optimizer luck, not a modeling
   change, unless it replicates.
4. Test 4: one threshold is not enough. Prior match and `g_i` collapse.
   Do not tie `theta_c`/`theta_d`.

Eval dump: `scripts/_tmp_ablation_t34_eval.py` /
`scripts/_tmp_ablation_t34_eval.json` (untracked).

### 2026-08-27d — I/M + prior overlays for all eval’d models

Same shared stim as the eval (`bps=20`, seed **12345**, from baseline
s101). Driver: `scripts/plot_best_fit_results.py` `plot_one` (traj
`loss_plot_diff_by_condition_with_data` + `loss_prior_effect`, as fitted).
**56/56** runs: baseline + tests 1–4, 8 seeds each.

Plots live **in each run dir** (openalyx `models/` for baseline; alyx
`models/` for poffset/noiti; alyx `models/new/` for tests 3–4):
`IM_pre.svg`, `IM_post.svg`, `P_fit.svg`, `prior_effects.svg`
(plus long param-name originals and `S_fit.png`).

Tests 1–2 JSON still live under alyx `models/` (not `models/new`).

### 2026-08-27e — RT vs signed contrast (act-prior conc/disc)

Concordant = stim side matches action-kernel prior (α=0.2), not true
block. x = signed contrast (left − / right +), **symlog**. RT = model
commit time, or data `firstMovement − stimOn`. Baseline **s333**. Data:
BWM `trials.pqt`, BWM RT/NaN mask, first vs last 50% of each session
then pooled (459 sessions).

In baseline s333 (openalyx `models/…_s333/`):
`rt_psychometric_actprior/` — `model_rt_baseline_s333.png`,
`data_rt_first50.png`, `data_rt_last50.png`.

Same three plots with **true block** (stim vs `probabilityLeft`; 0.5 dropped)
in `rt_psychometric_trueblock/` next to that.

### 2026-08-30 — Test 5: 3× pre-action M (regular sweep)

Traj loss matches I/M over **T=72 steps = 144 ms** post-stim plus a
pre-action window. Those two nSSE terms were equal. Mean data RT is
0.3–0.6 s, so the optimizer never sees |M| near the bound on hard /
incongruent trials — `θ_d` and `g_i` wander to help early I/M and
wreck time-to-bound (s333).

**Change:** `model_params['m_pre_weight']` multiplies **only** pre-action
**M** nSSE. Post-start M, both I windows, P, prior, and the ITI
penalty stay at 1. CLI `--m-pre-weight` / env `M_PRE_WEIGHT`. Default
remains 1 (old fits unchanged). R² in the loss debug is still
unweighted.

**Sweep:** same 8 regular seeds as tests 1–4, `stageB_hold_s89`
protocol, weight **3**. Driver (do **not** run from the laptop):

```bash
bash scripts/submit_fit_stage_b_mpre.sh
```

Override: `M_PRE_WEIGHT=2` (tag becomes `stageB_hold_s89_mpre2`).
Score after: shared-stim eval tot **and** act-prior RT R² vs baseline
s101 / s333 / WEIGHTS_REL. Expect `θ_d` closer to `θ_c` and fewer
incongruent timeouts if the extra M-pre term is doing its job.

### 2026-08-31 — Test 5 landed (mpre3 vs regular RT)

All 8 seeds `FIT_DONE` under openalyx
`models/weights_run_fj_stageB_hold_s89_mpre3_regular_mask12-13_s<seed>/`.
Recorded JSON loss includes 3× M-pre — **not** comparable to baseline.
Fair eval (shared stim, `bps=20`, seed 12345, from baseline s101;
`scripts/_tmp_mpre3_eval.py`): score both arms at `m_pre_weight=1`.

Dump: `models/stageB_hold_s89_mpre3_eval.json`.

| seed | mpre3 fair | base fair | mpre3 `g_i` | base `g_i` | mpre3 `θ_d` | base `θ_d` |
|-----:|----------:|----------:|----------:|----------:|----------:|----------:|
| 7 | 1.095 | 1.076 | 159 | 200 | 0.462 | 0.473 |
| 12 | 1.120 | 1.026 | **76.5** | 188 | 0.501 | 0.427 |
| 34 | 1.216 | 1.026 | 183 | 166 | 0.413 | 0.408 |
| 45 | 1.156 | 1.114 | 200 | 178 | 0.405 | 0.373 |
| 89 | 1.030 | 1.095 | **180** | **0.3** | 0.457 | 0.386 |
| 101 | 1.025 | 1.017 | 186 | 196 | 0.405 | 0.389 |
| **303** | **0.981** | 1.083 | 170 | 182 | 0.399 | 0.409 |
| 333 | 1.066 | **1.015** | 187 | 143 | 0.513 | 0.458 |

| arm | best fair | median |
|-----|----------:|-------:|
| baseline | **1.015** (s333) | **1.051** |
| mpre3 | **0.981** (s303) | 1.081 |

Best single seed is mpre3 s303; **median still favors regular**. s89’s
`g_i` collapse is rescued (0.3 → 180). s12 collapses instead (188 → 76).
`θ_d` does **not** move toward `θ_c` (s333 mpre3 still 0.51 vs `θ_c`
0.81).

#### S / I/M / prior R² (shared stim, 1 session)

Driver: `scripts/_tmp_mpre3_plots.py` → `plot_one` + `alias_svgs`.
Plots in each mpre3 run dir: `IM_pre.svg`, `IM_post.svg`, `P_fit.svg`,
`prior_effects.svg`, `S_fit.png`.

| seed | I pre / post | M pre / post | S | prior |
|-----:|--------------|--------------|--:|------:|
| 7 | 0.932 / 0.964 | 0.992 / 0.948 | 0.466 | 0.950 |
| 12 | 0.975 / 0.954 | 0.995 / 0.925 | 0.464 | 0.944 |
| 34 | 0.947 / 0.967 | 0.982 / 0.937 | 0.464 | 0.928 |
| 45 | 0.953 / 0.972 | 0.984 / 0.956 | 0.429 | 0.937 |
| 89 | 0.963 / 0.963 | 0.993 / 0.945 | 0.464 | 0.950 |
| 101 | 0.978 / 0.963 | 0.972 / 0.936 | 0.466 | 0.950 |
| 303 | 0.977 / 0.963 | 0.972 / 0.933 | 0.466 | **0.969** |
| 333 | 0.974 / 0.958 | 0.997 / 0.923 | 0.464 | 0.946 |

I/M and prior stay high (M-pre 0.97–1.00 as intended). S is still
~0.43–0.47.

#### Act-prior RT R² vs regular (same protocol)

10 sessions × 20 blocks, stim seed 12345. Data: action-kernel α=0.2
(`behavior_actprior.npy`). Model: binarized trial-average P. Driver:
`scripts/_tmp_perf_rt_model_vs_data.py` (`plot_one_json`). Regular
rescored the same way (`scripts/_tmp_regular_actprior_rt.py`).

**comb** = pooled RT R²: mean RT vs signed contrast over **all**
committed trials (0.08–2 s), no con/inc split. (Until 08-31 this was
wrongly the unweighted mean of the two split model curves; that is
fixed in `loss_perf_with_data` `combined_all`.) **con / inc** = the
same R² after splitting by whether stim side matches the prior
(concordant vs incongruent): data uses the action kernel; model uses
binarized trial-avg P. Choice (perf) R² is a separate overlay, not in
this table.

Plots: each run’s `psychometric_model_vs_data_actprior/`. Summaries:
`models/stageB_hold_s89_mpre3_plot_summary.json`,
`models/stageB_hold_s89_regular_actprior_rt.json`.

| seed | mpre3 comb | mpre3 con | mpre3 inc | regular comb | regular con | regular inc |
|-----:|-----------:|----------:|----------:|-------------:|------------:|------------:|
| 7 | 0.208 | 0.710 | −4.76 | 0.475 | 0.589 | −0.15 |
| 12 | 0.518 | 0.625 | 0.000 | 0.566 | 0.750 | −1.07 |
| 34 | 0.325 | 0.790 | −4.01 | 0.713 | 0.846 | −1.22 |
| 45 | 0.630 | 0.928 | −4.32 | 0.748 | 0.825 | −1.65 |
| 89 | 0.374 | 0.723 | −2.53 | 0.672† | 0.773 | 0.11 |
| 101 | 0.636 | 0.818 | −1.32 | **0.787** | **0.848** | −0.21 |
| **303** | **0.772** | 0.843 | −0.43 | 0.576 | 0.811 | −2.54 |
| 333 | 0.312 | 0.608 | −1.66 | −0.084 | 0.315 | −2.25 |
| WEIGHTS_REL | — | — | — | 0.730 | 0.809 | **0.18** |

† regular s89 has collapsed `g_i=0.3` — not a usable fit.

**Best pooled RT R²:** mpre3 s303 **0.772** vs this-batch regular s101
**0.787** (Δ **−0.015**) vs last year’s WEIGHTS_REL **0.730**. 3× M-pre
does **not** beat the best regular pooled RT, but s303 does beat
WEIGHTS_REL on this metric. Same-seed s303: regular 0.576 → mpre3
0.772. Concordant RT is already decent on both arms (mpre3 s303
**0.84**, regular s101 **0.85**); incongruent is still negative (mpre3
s303 **−0.43** vs regular s101 **−0.21** vs WEIGHTS_REL **+0.18**).

**Keep `m_pre_weight=1` as the default.** The extra M-pre term can
rescue a bad RT seed without pulling `θ_d` up. Pooled RT is close
(s303 0.772 vs regular s101 0.787) but incongruent is still the hole.

### 2026-09-07 — Test 6: I/M prior at full 150 ms (wired)

`loss_prior_effect` currently builds model I/M prior-distance at **T=72**
(= **144 ms** at `dt=2`) and labels the axis with `plot_window=80`. The
BWM I/M targets are the full **150 ms** after stimOn (`act_block_duringstim`)
and **150 ms** before movement (`act_block_duringchoice`) — 72 bins at
~2.08 ms. Traj loss stays T=72; only the prior-distance term changes.

**Change:** `model_params['prior_window_ms']` sets the I/M prior window.
When **150**: model T = `round(150/dt)` = **75** steps; `plot_window=150`;
the 75-step curve is resampled onto the 72 data bins so the full 150 ms
is scored on both alignments. S sidecar (unsplit 80 ms) is unchanged.
Default remains unset (legacy T=72 / plot_window=80). Loky via
`loss_extra_kwargs` → `apply_model_ablation_flags`. CLI
`--prior-window-ms` / env `PRIOR_WINDOW_MS`.

**Sweep:** same 8 regular seeds as tests 1–5, `stageB_hold_s89`
protocol. Driver (do **not** run from the laptop):

```bash
ABLATIONS=im150 bash scripts/submit_fit_stage_b_model_ablations.sh
```

Run dirs: `weights_run_fj_stageB_hold_s89_im150_regular_mask12-13_s<seed>/`.
Score after on shared-stim eval tot **as fitted** (150 ms prior) and
with `prior_window_ms` unset (legacy T=72) so the new loss is not
compared raw to baseline.

Smoke:

```bash
SEEDS=999 ABLATIONS=im150 OUT_TAG=stageB_ablate_im150_smoke \
  DE1_MAXITER=2 DE2_MAXITER=3 POPSIZE=8 SOBOL_COUNT=4 \
  PATIENCE=0 LOCAL_REFINE_MAX_WALL_S=60 FORCE=1 TIME=1:00:00 \
  bash scripts/submit_fit_stage_b_model_ablations.sh
```

**Local smoke (2026-09-07):** hybrid s89, bps=10, stim seed 12345, 605 trials,
`HAVE_NUMBA=True`. Model curves are 72 bins (legacy) vs **75** (im150), both
finite. Prior nSSE **0.785 → 0.821**; joint **1.839 → 1.875** (Δ **+0.036**
is entirely the prior term). Flag rides `loss_joint_core` /
`apply_model_ablation_flags`.

### 2026-09-07b — Test 6 results (openalyx, 8/8)

Local copies: openalyx `models/`
`weights_run_fj_stageB_hold_s89_im150_regular_mask12-13_s{7,12,34,45,89,101,303,333}/`.
All **8/8 `FIT_DONE`**, `fit_status=ok`, `prior_window_ms=150` in
`run_fit_joint_report.json` and `weights_final_*.json`.

**Eval:** same protocol as tests 1–5 — `bps=20`, stim seed **12345**,
stim from baseline **s101**, nested `fit_targets/`. Driver:
`scripts/_tmp_im150_eval.py`. Dump:
`models/stageB_hold_s89_im150_eval.json`. Rank on **eval tot**
(traj + prior + S). JSON `final_loss` is own-stim — not comparable.

Score each arm **as fitted** and on the other window:

| Column | Fit | What is reported |
|--------|-----|------------------|
| **im150 150** | `prior_window_ms=150` | eval tot as fitted (75 steps resampled onto 72 BWM bins) |
| **im150 → leg** | `prior_window_ms=150` | same weights, production T=72 / 40-bin `plot_window=80` |
| **base leg** | unset | production eval tot (tests 1–5 column) |
| **base → 150** | unset | same weights, scored on the 150 ms objective |

`loss_prior_effect` briefly used `len(r_int)` (72 bins) even when the
flag was unset, which inflated baseline prior ~4–6× vs the 08-31
table. Restored: unset → `data_length = round(plot_window/2)` = **40**;
`prior_window_ms=150` still uses the data’s 72 bins. After the restore,
baseline **s333 = 1.015** / median **1.051** match tests 1–5.

#### Eval total loss

| seed | im150 fit | im150 150 | im150 → leg | base fit | base leg | base → 150 | im150 `g_i` | base `g_i` |
|-----:|----------:|----------:|------------:|---------:|---------:|-----------:|----------:|----------:|
| 7 | 1.436 | 1.780 | 1.144 | 0.996 | 1.076 | 1.717 | 200 | 200 |
| 12 | 1.195 | 1.414 | 1.284 | 0.960 | 1.026 | 1.458 | 161 | 188 |
| 34 | 1.224 | 1.504 | **1.080** | 0.952 | 1.026 | 1.593 | 188 | 166 |
| **45** | 2.035 | **1.313** | 1.253 | 1.524 | 1.114 | 1.450 | 200 | 178 |
| 89 | 1.356 | 1.384 | 1.122 | 1.149 | 1.095 | **1.364** | 193 | **0.3** |
| 101 | 1.393 | 1.429 | 1.280 | 1.131 | 1.017 | 1.461 | 160 | 196 |
| 303 | 1.975 | 1.335 | 1.196 | 1.030 | 1.083 | 1.633 | 200 | 182 |
| 333 | 1.044 | 1.517 | 1.375 | 0.962 | **1.015** | 1.914 | 194 | 143 |

| arm | best eval tot | median | mean |
|-----|----------:|-------:|-----:|
| baseline (legacy) | **1.015** (s333) | **1.051** | 1.056 |
| im150 → legacy | 1.080 (s34) | 1.225 | 1.217 |
| im150 as fitted (150) | **1.313** (s45) | 1.422 | 1.460 |
| baseline → 150 | 1.364 (s89†) | 1.527 | 1.574 |

† baseline s89 has collapsed `g_i=0.3` — not a usable fit. Next-best
baseline-on-150 is s45 **1.450**.

`L_S` stays ~0.490–0.510 except im150 s45 **0.461** (CMA/polish can
unfreeze retinal). Traj on im150 is in the same band as regular
(~0.28–0.44). The extra loss is **prior**: as-fitted 150 nSSE
0.49–1.01 vs baseline legacy 0.11–0.25.

Best im150 **s45** as fitted: traj 0.302 + prior 0.550 + S 0.461 =
**1.313**. Same weights on the production window: 1.253. Best
transfer to production is s34 (**1.080** = 0.383 + 0.200 + 0.497) —
still above baseline s333 **1.015**.

#### Test 6 — full 150 ms I/M prior

Hypothesis: scoring model I/M prior-distance on the same 150 ms
post-stimOn / pre-movement window as the BWM targets (instead of the
legacy T=72 / `plot_window=80` = 40 bins) should lower `L_w`.

**Not on the production metric.** Median eval tot on the legacy
window is **1.225** vs baseline **1.051**. Best single seed is also
worse (1.080 vs 1.015). On the new 150 ms objective, im150 beats
baseline-at-150 (median 1.422 vs 1.527; best 1.313 vs 1.364) — that
is the loss they were trained on, and even there the prior term stays
large (best prior nSSE **0.493**, GoF 0.88). `g_i` stays healthy
(160–200); no s89-style collapse. `θ_d` 0.33–0.43 vs baseline
0.37–0.47. `W_pp` still sits on the 2.5 s floor.

Plots (as fitted, 150 ms axis) in each im150 run dir:
`prior_effects.svg` / `.png`, `IM_pre.svg`, `IM_post.svg`, `P_fit.svg`,
`S_fit.png`.

**Why movement-aligned model curves start low (09-07b):** not a
panel-swap / axis bug. Stim-aligned t=0 is stimOn for every trial
(I/M still hold the last 100 ms prestim `P_offset`). Movement-aligned
t=−150 ms is `RT−150` — the same clock only when RT=150 ms. On s45
(bps=20, stim 12345) median RT is **170 ms** and **39%** of committed
trials have RT < 150 ms; those snippets start in the ITI, but
restricting to RT≥150 ms makes `action[0]` *lower* (I 0.016 → 0.005),
so the dip is the long-RT majority, sampled well after stimOn, where
model I prior-distance has faded. Data I also drops a little
(duringstim 0.050 → duringchoice 0.038); the model overdoes it
(0.051 → 0.016). Legacy scoring only kept the last **80 ms** before
movement, where model I/M have risen again (`action[-1]` I 0.063,
M 0.159) — that is why this hole was hidden in tests 1–5.

#### Act-prior RT (same protocol as test 5)

10 sessions × 20 blocks, stim seed 12345. Data: action-kernel α=0.2.
Model: binarized trial-average P. Driver: `scripts/_tmp_im150_actprior_rt.py`.
Plots: each run’s `psychometric_model_vs_data_actprior/`. Dump:
`models/stageB_hold_s89_im150_actprior_rt.json`.

| seed | im150 comb | im150 con | im150 inc | im150 perf |
|-----:|-----------:|----------:|----------:|-----------:|
| 7 | 0.597 | 0.733 | −1.79 | 0.798 |
| 12 | 0.768 | 0.696 | −0.52 | 0.722 |
| 34 | 0.686 | 0.887 | −3.84 | 0.611 |
| 45 | 0.771 | 0.855 | −1.52 | 0.755 |
| 89 | 0.820 | 0.824 | −0.07 | 0.848 |
| **101** | **0.874** | **0.951** | −0.76 | 0.829 |
| 303 | 0.771 | 0.354 | −2.83 | 0.028 |
| 333 | 0.762 | 0.641 | −0.45 | 0.684 |

Best pooled RT R² is im150 s101 **0.874** vs this-batch regular s101
**0.787** vs mpre3 s303 **0.772** vs WEIGHTS_REL **0.730**. Incongruent
is still negative (s101 **−0.76** vs regular s101 **−0.21** vs
WEIGHTS_REL **+0.18**). s303’s choice R² collapsed (0.028).

**Keep `prior_window_ms` unset** for the stim×choice 150 ms arm. The
BWM I/M curves are 150 ms, but fitting to that full window does not
beat regular on the loss used for tests 1–5. Leave
`--prior-window-ms 150` as an opt-in.

### 2026-09-07c — Test 6 revised: 150 ms + stim-only stratum

The stim×choice im150 model I/M prior **drops after ~80 ms** because I
becomes choice-locked and the loss conditions on choice. Dropping the
choice cell (`stratum='stim'`) removes that drop (s45 I 0.065 → 0.158
at 150 ms). Data targets stay the 4-split `act_block_duringstim` /
`duringchoice` curves.

**Change:** `model_params['prior_stratum']`: unset / `stim_choice`
(production) vs `stim` (stim side only; equal-weight L/R means, then
one L2) vs `all` (`lump_all`). Rides `loss_extra_kwargs` →
`apply_model_ablation_flags`. CLI `--prior-stratum` / env
`PRIOR_STRATUM`. `None` does **not** clear a previous `stim` (same
footgun as `prior_window_ms`).

**Sweep:** same 8 regular seeds, `stageB_hold_s89` protocol. New run
dirs so the stim×choice im150 jobs are not overwritten:

`weights_run_fj_stageB_hold_s89_im150stim_regular_mask12-13_s<seed>/`

```bash
ABLATIONS=im150stim bash scripts/submit_fit_stage_b_model_ablations.sh
```

`ABLATIONS=im150` is still the old stim×choice 150 ms arm.

Scored 2026-09-08 on shared-stim eval tot **as fitted** (150 + stim) and
with `prior_stratum` / `prior_window_ms` cleared (production
stim×choice T=72 / 40-bin). See below.

Smoke:

```bash
SEEDS=999 ABLATIONS=im150stim OUT_TAG=stageB_ablate_im150stim_smoke \
  DE1_MAXITER=2 DE2_MAXITER=3 POPSIZE=8 SOBOL_COUNT=4 \
  PATIENCE=0 LOCAL_REFINE_MAX_WALL_S=60 FORCE=1 TIME=1:00:00 \
  bash scripts/submit_fit_stage_b_model_ablations.sh
```

### 2026-09-08 — Test 6 revised results (`im150stim`)

Local copies: openalyx `models/`
`weights_run_fj_stageB_hold_s89_im150stim_regular_mask12-13_s{7,12,34,45,89,101,303,333}/`.
All **8/8 `FIT_DONE`**, `fit_status=ok`, `prior_window_ms=150` and
`prior_stratum=stim` in `run_fit_joint_report.json` and
`weights_final_*.json`.

**Eval:** same protocol as tests 1–6 — `bps=20`, stim seed **12345**,
stim from baseline **s101**, nested `fit_targets/`. Driver:
`scripts/_tmp_im150stim_eval.py`. Dump:
`models/stageB_hold_s89_im150stim_eval.json`. Rank on **eval tot**
(traj + prior + S). JSON `final_loss` is own-stim — not comparable.

Score each arm **as fitted** and on the other setting:

| Column | Fit | What is reported |
|--------|-----|------------------|
| **im150stim 150stim** | `prior_window_ms=150`, `prior_stratum=stim` | eval tot as fitted (75 steps, stim-only I/M, resampled onto 72 BWM bins) |
| **im150stim → prod** | same weights | production stim×choice T=72 / 40-bin `plot_window=80` |
| **base prod** | unset / unset | production eval tot (tests 1–5 column) |
| **base → 150stim** | unset | same weights, scored on the 150 ms + stim-only objective |

Data targets stay the cached 4-split `act_block_duringstim` /
`duringchoice` curves. Only the **model** distance changes stratum.

#### Eval total loss

| seed | im150stim fit | im150stim 150stim | im150stim → prod | base fit | base prod | base → 150stim | im150stim `g_i` | base `g_i` |
|-----:|--------------:|------------------:|-----------------:|---------:|----------:|---------------:|----------------:|----------:|
| 7 | 1.114 | 1.215 | **1.515** | 0.996 | 1.076 | 2.953 | **38.8** | 200 |
| **12** | 1.675 | **1.194** | 2.247 | 0.960 | 1.026 | 4.091 | 0.1 | 188 |
| 34 | 1.916 | 1.243 | 2.314 | 0.952 | 1.026 | 5.035 | 2.1 | 166 |
| 45 | 2.035 | 1.538 | 2.272 | 1.524 | 1.114 | 6.812 | 1.1 | 178 |
| 89 | 1.276 | 1.584 | 2.686 | 1.149 | 1.095 | 2.760 | 0.4 | **0.3** |
| 101 | 1.989 | 1.260 | 2.268 | 1.131 | 1.017 | 4.655 | 0.1 | 196 |
| 303 | 1.210 | 1.276 | 2.122 | 1.030 | 1.083 | 5.773 | 0.9 | 182 |
| 333 | 1.175 | 1.517 | 1.855 | 0.962 | **1.015** | **2.684** | 4.1 | 143 |

| arm | best eval tot | median | mean |
|-----|----------:|-------:|-----:|
| baseline (production) | **1.015** (s333) | **1.051** | 1.056 |
| im150 → production (09-07b) | 1.080 (s34) | 1.225 | 1.217 |
| im150stim as fitted (150+stim) | **1.194** (s12) | 1.268 | 1.353 |
| im150stim → production | 1.515 (s7) | 2.257 | 2.160 |
| baseline → 150+stim | 2.684 (s333) | 4.373 | 4.345 |

`L_S` stays ~0.496–0.497. Traj is in the same band as regular
(~0.34–0.56). The extra production loss is **prior**: stim×choice 40-bin
nSSE 0.67–1.63 vs baseline 0.11–0.25.

Best as-fitted **s12**: traj 0.392 + prior 0.305 + S 0.497 = **1.194**
(GoF 0.92). Same weights on production: **2.247**. Best transfer is s7
(**1.515** = 0.351 + 0.668 + 0.497) — still 0.50 above baseline s333
**1.015**, and 0.43 above the original im150→legacy best (s34 **1.080**).

On the training objective, im150stim beats baseline-at-150stim (median
1.268 vs 4.373) because regular I/M **drop** after ~80 ms once choice
is conditioned, and stim-only scoring of those weights is a large
mismatch (prior nSSE 1.82–5.92). That is not the loss used for tests
1–5.

#### `g_i` collapsed; thresholds nearly tied

Seven of eight seeds have `g_i` ≤ 4.1 (s12 / s101 **0.1**). Only s7
keeps a usable gain (**38.8**), still far below regular 143–200.
`θ_c` 0.54–0.64 and `θ_d` 0.48–0.56 sit on top of each other (regular
0.73–0.77 / 0.37–0.47). `W_pp` still sits on the 2.5 s floor.
`g_m` is ~0 except s7 / s12 (~0.01). This is the same failure mode as
test 4 (tied θ blows prior / `g_i`) and baseline s89.

#### Test 6 revised — 150 ms + stim-only I/M prior

Hypothesis: drop the choice cell so model I/M prior-distance can keep
rising after 80 ms, matching the full 150 ms BWM window without the
choice-lock collapse.

**Not on the production metric.** Median eval tot on the legacy
stim×choice window is **2.257** vs baseline **1.051** vs original
im150→legacy **1.225**. Best single seed is also worse (1.515 vs 1.015
vs 1.080). On the new 150+stim objective the arm wins — that is the
loss they were trained on. Data targets are still the 4-split curves,
so the model is matching a stim-only distance to a stim×choice target;
winning that objective did not recover a usable `g_i`.

Plots (as fitted, 150 ms + stim) in each im150stim run dir:
`prior_effects.svg` / `.png`, `IM_pre.svg`, `IM_post.svg`, `P_fit.svg`,
`S_fit.png`.

#### Act-prior RT (same protocol as tests 5–6)

10 sessions × 20 blocks, stim seed 12345. Data: action-kernel α=0.2.
Model: binarized trial-average P. Driver:
`scripts/_tmp_im150stim_actprior_rt.py`. Plots: each run’s
`psychometric_model_vs_data_actprior/`. Dump:
`models/stageB_hold_s89_im150stim_actprior_rt.json`.

| seed | im150stim comb | im150stim con | im150stim inc | im150stim perf |
|-----:|---------------:|--------------:|--------------:|---------------:|
| 7 | −7.61 | −5.98 | −24.9 | 0.537 |
| 12 | −10.3 | −8.60 | −25.7 | 0.536 |
| 34 | −7.11 | −5.07 | −23.1 | 0.605 |
| 45 | −8.91 | −7.60 | −23.0 | 0.502 |
| 89 | −1.97 | −1.26 | −5.20 | 0.751 |
| 101 | −7.86 | −5.99 | −18.3 | 0.611 |
| 303 | −5.36 | −3.86 | −17.9 | 0.678 |
| **333** | **−1.46** | −0.65 | −5.95 | **0.804** |

Every combined RT R² is negative. Best is s333 **−1.46** vs im150 s101
**0.874** vs regular s101 **0.787** vs mpre3 s303 **0.772**. Choice R²
is also down (best 0.804 vs im150 0.848 / regular ~0.9). Collapsed
`g_i` plus nearly-tied thresholds wreck the RT curves.

**Keep `prior_window_ms` and `prior_stratum` unset.** Fitting I/M to
the full 150 ms window, with or without the choice cell, does not beat
regular on the loss used for tests 1–5. Both flags stay opt-in.

### 2026-09-08d — `g_i` floor = `g_m`; rerun stim-only arms

`im150stim` sat on the old `g_i` floor (`0.1`) on s12/s101. Native
box is now `[1e-12, 200]` for both `g_i` and `g_m` (`fit_weights.py`
`_log_bounds_weights_v2`, `fit_joint.NATIVE_BOUNDS`). Log-uniform
support is wider (geometric midpoint `√(1e-12·200) ≈ 4.5e-5`). `g_s`
stays `[0.1, 200]`. Frozen-dim clamp is unchanged (still free dims
only).

Two arms, same 8 seeds, `FORCE=1` so `im150stim` overwrites the 09-08
run dirs:

| Ablation | Window | Stratum | Run dir tag |
|----------|--------|---------|-------------|
| `im150stim` | 150 ms | `stim` | `stageB_hold_s89_im150stim` (replace) |
| `stimonly` | legacy T=72 / `plot_window=80` | `stim` | `stageB_hold_s89_stimonly` (new) |

```bash
PARTITION=mit_preemptable FORCE=1 ABLATIONS="im150stim stimonly" \
  bash scripts/submit_fit_stage_b_model_ablations.sh
```

**Local smoke (2026-09-08):** hybrid s89, bps=10, stim seed 12345, 601 trials,
`HAVE_NUMBA=True`. Bounds `g_i=g_m=[1e-12, 200]`, `g_s` still `[0.1, 200]`.
Frozen `g_i` at `LOG_ZERO` stays `9.4e-14` (not pulled to the floor).
`stimonly` after `im150stim` does not leak `PRIOR_WINDOW_MS`. One-eval
prior nSSE: regular **0.083** (T=72, stim×choice) → stimonly **5.998**
(T=72, stim) → im150stim **4.360** (T=75, stim). Joint totals 0.734 /
6.650 / 5.011, all finite.

### 2026-09-08e — `g_i` floor `1e-12` results (`im150stim` + `stimonly`)

Local copies: openalyx `models/`
`weights_run_fj_stageB_hold_s89_{im150stim,stimonly}_regular_mask12-13_s{7,12,34,45,89,101,303,333}/`.
All **16/16 `FIT_DONE`**, `fit_status=ok`, `g_i_bounds=[1e-12, 200]`.
`im150stim`: `prior_window_ms=150`, `prior_stratum=stim`. `stimonly`:
window unset, `prior_stratum=stim`. JSON timestamps 20260908-17xx/18xx
(replaced the 09-08 morning `im150stim` dirs).

**Eval:** same protocol as tests 1–6 — `bps=20`, stim seed **12345**,
stim from baseline **s101**. Driver: `scripts/_tmp_gilo_stim_eval.py`.
Dump: `models/stageB_hold_s89_gilo_stim_eval.json`. Rank on **eval tot**
(traj + prior + S).

| Column | Fit | What is reported |
|--------|-----|------------------|
| **im150stim 150stim** | 150 + `stim` | as fitted |
| **im150stim → prod** | same weights | production stim×choice T=72 / 40-bin |
| **stimonly fit** | legacy + `stim` | as fitted |
| **stimonly → prod** | same weights | production |
| **base prod** | unset / unset | tests 1–5 column |

#### Eval total loss

| seed | im150stim 150 | im150stim → prod | stimonly fit | stimonly → prod | base prod | im150stim `g_i` | stimonly `g_i` |
|-----:|--------------:|-----------------:|-------------:|----------------:|----------:|----------------:|---------------:|
| 7 | **1.188** | 1.736 | 1.120 | 1.717 | 1.076 | 1.6e-5 | 6e-6 |
| 12 | 1.397 | 2.359 | 1.148 | 1.922 | 1.026 | 1.4e-6 | 0.070 |
| 34 | 1.384 | **1.252** | 1.469 | **1.443** | 1.026 | 0.006 | 2.6e-11 |
| 45 | 1.479 | 2.002 | 1.122 | 1.866 | 1.114 | 0.024 | 3.9e-10 |
| 89 | 1.509 | 2.218 | 1.086 | 2.101 | 1.095 | 0.002 | 6.8e-5 |
| 101 | 1.235 | 2.498 | 1.430 | 2.132 | 1.017 | 2.3e-5 | **115** |
| 303 | 1.286 | 2.013 | **1.060** | 1.652 | 1.083 | 1.6e-7 | 2.5e-4 |
| 333 | 1.309 | 2.564 | 1.336 | 1.874 | **1.015** | **76** | 7.2e-6 |

| arm | best eval tot | median | mean |
|-----|----------:|-------:|-----:|
| baseline (production) | **1.015** (s333) | **1.051** | 1.056 |
| im150 → production (09-07b) | 1.080 (s34) | 1.225 | 1.217 |
| im150stim → prod (0.1 floor, 09-08) | 1.515 (s7) | 2.257 | 2.160 |
| im150stim → prod (`1e-12` floor) | 1.252 (s34) | 2.115 | 2.080 |
| stimonly → prod | 1.443 (s34) | 1.870 | 1.838 |
| im150stim as fitted (150+stim) | 1.188 (s7) | 1.347 | 1.349 |
| stimonly as fitted (legacy+stim) | 1.060 (s303) | 1.135 | 1.221 |
| baseline → 150+stim | 2.684 | 4.373 | 4.345 |
| baseline → legacy+stim | 3.358 | 5.452 | 5.386 |

Opening the floor did **not** recover a usable prior gain. 14/16 seeds
have `g_i` ≤ 0.07; most sit at `10⁻¹¹`–`10⁻⁵`. Only im150stim s333
(**76**) and stimonly s101 (**115**) stay in the regular band. Thresholds
are still nearly tied (`θ_c` 0.51–0.67, `θ_d` 0.42–0.63 vs regular
0.73–0.77 / 0.37–0.47). `L_S` ~0.496 except im150stim s12 **0.574**.

On production, neither arm beats baseline. Best transfer is im150stim
s34 **1.252** (traj 0.450 + prior 0.306 + S 0.496) — still 0.24 above
s333 **1.015**, and that seed has `g_i=0.006`. stimonly transfer is
better than the first `im150stim` batch (median 1.870 vs 2.257) but
worse than original stim×choice im150→legacy (1.225).

They win their own objectives vs baseline scored the same way (as
fitted 1.06–1.19 vs baseline-at-150stim 2.68 / baseline-at-stimonly
3.36). That is the mismatch they were trained on, not tests 1–5.

Plots (as fitted) in each run dir: `prior_effects.svg` / `.png`,
`IM_pre`, `IM_post`, `P_fit`, `S_fit`.

**Keep `prior_window_ms` and `prior_stratum` unset. Put `g_i` floor
back to `0.1`.** The open box only let the stim-only loss drive `g_i`
closer to zero.

#### Act-prior RT (same protocol as tests 5–6)

10 sessions × 20 blocks, stim seed 12345. Driver:
`scripts/_tmp_gilo_stim_actprior_rt.py`. Dump:
`models/stageB_hold_s89_gilo_stim_actprior_rt.json`.

| seed | im150stim comb | im150stim perf | stimonly comb | stimonly perf |
|-----:|---------------:|---------------:|--------------:|--------------:|
| 7 | −5.24 | 0.677 | −1.40 | 0.721 |
| 12 | −2.75 | 0.772 | −4.89 | 0.641 |
| **34** | **0.253** | **0.858** | −4.24 | 0.661 |
| 45 | −2.55 | 0.706 | −5.22 | 0.628 |
| 89 | −9.21 | 0.538 | −5.15 | 0.682 |
| 101 | −8.70 | 0.598 | −5.36 | 0.628 |
| 303 | −10.9 | 0.728 | −7.25 | 0.629 |
| 333 | −7.78 | 0.584 | −0.16 | **0.810** |

Only im150stim s34 has a positive combined RT R² (**0.253** vs regular
s101 **0.787** vs im150 s101 **0.874**). Every stimonly seed is
negative (least-bad s333 **−0.16**).

### 2026-09-08f — model prior-distance = mean of per-cell L2s

BWM I/M targets (`load_group` on the 4-split `act_block_duringstim` /
`duringchoice` combine) are **`mean_c ‖μ₊−μ₋‖`**: each split is a
within-cell Euclidean curve, then sum / 4. The model was doing the
other order — equal-weight the four `(stim, choice)` means, then one
L2 (`‖mean_c Δ‖`). After S peaks (~68 ms) I/M rotate onto choice;
error-cell `Δ` anti-aligns with correct-cell `Δ` and the pooled L2
drops even though each cell’s own prior gap is still growing. That is
why stim×choice im150 fell after ~80 ms while movement-aligned (and
`θ_c > θ_d`) still rose. Not same-threshold collapse and not
concordant-faster RT (median RT 164 vs 174 ms; 0.9% of trials
committed by 80 ms).

`prior_distance_I_M_both_alignments` (numpy + torch) now averages
within-cell distances for every multi-cell stratum:

| stratum | cells | was | now (matches data) |
|---------|-------|-----|--------------------|
| `stim_choice` (I/M default) | 4 `(ts, ch)` | `‖mean_c Δ‖` | `mean_c ‖Δ‖` |
| `stim` (S default; im150stim) | 2 stim sides | same bug | mean of 2 within-stim L2s |
| `all` | 1 pool | unchanged | the two orders coincide |

`prior_distance_I_M_by_choice_and_prior` is unused in the fit (fill-from-ITI
was copied from it). It now does the same 4-cell `mean_c ‖Δ‖` as the
production prior function (it previously ignored stim).
`choice_distance_I_M_both_alignments` is plot-only
(`plot_choice_effect`, no SSE). It now averages the four (stim, prior)
choice L–R distances to match `choice_duringstim_act` /
`choice_duringchoice_act`.

Fill-from-next-ITI, `include_all_trials`, `side` metric, and both
alignments are unchanged. `sub_prior[i][0]` is still P at trial start
(slow `W_pp`; sign is stable).

**Rerun:** regular + `im150` (150 ms, stim×choice) on the 8 seeds,
**new** run dirs so the 09-07 `‖mean_c Δ‖` campaign stays:

`weights_run_fj_stageB_hold_s89_im150_meancell_regular_mask12-13_s<seed>/`

```bash
PARTITION=mit_preemptable ABLATIONS=im150 \
  bash scripts/submit_fit_stage_b_model_ablations.sh
```

Default `OUT_TAG` for `im150` is now `stageB_hold_s89_im150_meancell`.
Do not `FORCE` the old `stageB_hold_s89_im150` dirs.

**Continued** (09-09 score + remaining shape aims):
[prior-curve dips and discordant RT](modeling_details_prior_rt_gaps.md).

### 2026-09-15 — mpre3 × 150 ms meancell + 80 ms full (queued, not scored)

Test 5 (`m_pre_weight=3`) was 80 ms **regular** only. New dirs, current
`mean_c ‖Δ‖`, stim×choice. Fit at 3× M-pre; fair eval later at weight 1
(same as 08-31). `g_i` floor is still the checkout **1e-12**. Do not
FORCE old `mpre3` / `full` / `im150_meancell` dirs.

| ARM | window | freeze | S sidecar | `OUT_TAG` |
|-----|--------|--------|-----------|-----------|
| `full` | unset (~80 ms) | none | unsplit-80 | `stageB_hold_s89_full_mpre3_meancell` |
| `im150` | 150 ms | regular `12\|13` | no | `stageB_hold_s89_mpre3_im150_meancell` |
| `full_im150` | 150 ms | none | unsplit-80 | `stageB_hold_s89_full_mpre3_im150_meancell` |

```bash
PARTITION=mit_preemptable ARMS="full im150 full_im150" FORCE=0 \
  bash scripts/submit_fit_stage_b_mpre.sh
```



