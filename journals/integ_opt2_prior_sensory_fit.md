# Integrator-rule opt2: improving the prior and sensory fits

**Scope:** after the 2026-10-29 warm fine-tunes, improve the **opt2**
(shuffle-slope) integrator-rule Stage B fits. The focus is the **S
trajectory**, the **S prior** curve, and the **C/M prior** curves.
Behaviour (act-prior psychometric and RT) stays held out of the loss and
is used only as a check (see memory: no behaviour in the fit loss).

**Not in scope:** opt1 except as a reference; new ORCD campaigns that
the user has not confirmed (agents print `submit_*.sh` commands only);
Harris / long-session nulls; changing the canonical prior-distance
settings.

**Starting point:** [Stage B fit variants](stage_b_fit_variants.md),
section "Integrator-rule targets" through "Report (2026-10-29)". Report:
`alyx/models/stageB_hold_newS34_integ_ft_gm0_report/report.pdf`.

**Code:** eval `scripts/_tmp_integ_ft_eval_plots.py`; re-plot
`scripts/_tmp_integ_ft_replot_s_r2.py`; report builder
`alyx/models/stageB_hold_newS34_integ_ft_gm0_report/make_ft_report.py`;
fine-tune submit `scripts/submit_fit_integ_finetune.sh`.

---

## Where opt2 stands (2026-10-29)

Anchors: **s303→s7** (best opt2 fair, 1.412) and **s303→s45** (best
balanced fit, fair 1.636, psychometric 0.94, RT 0.77). Same shared stim
as the catalog (bps 20, seed 12345, regular s101), `m_pre_weight` 1.

| | opt2 s12→s7 | opt2 s303→s7 | opt2 s303→s45 † |
|---|---:|---:|---:|
| fair | 1.282 | 1.412 | 1.636 |
| `L_S` | 0.556 | 0.593 | 0.653 |
| prior S | 0.088 | 0.203 | 0.403 |
| prior C stim / move | 0.041 / 0.062 | 0.027 / 0.043 | 0.021 / 0.040 |
| prior M stim / move | 0.090 / 0.102 | 0.143 / 0.087 | 0.086 / 0.054 |
| g_s / d_s | 3.2 / 11.3 | 0.098 / 17.1 | ≈0 / 3.9 |
| psychometric / RT R² | 0.43 / 0.08 | 0.85 / 0.23 | 0.94 / 0.77 |

### What is wrong, by curve

1. **S trajectory (Stage A, arm-independent).** opt1 and opt2 use the
   same `avg_mean_R` (identical md5) and the same S prior sidecar. Stage
   B holds the retinal parameters, so the S trajectory is fixed by Stage
   A s34. At c=1 the data rises gradually from about 20 ms and peaks at
   about 0.52. The model is flat until about 30 ms, then steps up and
   overshoots to about 0.65. Low contrasts come in late. The pooled
   10-region target probably mixes early (LGd/LP/VIS) and late
   (PO/ZI/PF) latencies, and one retinal delay cannot smear the onset.
2. **S prior (0–80 ms).** The target (5 regions: VISal, VISam, LP, PO,
   ZI; `r_obs − r_null`) is already about 0.018 at 0 ms and rises to
   about 0.03 by 80 ms. The model S prior is flat at about 0.01 until the
   S response arrives (~35 ms), then steps. A gain on the stimulus input
   (g_s) cannot make a prior signal before the response, which matches
   [direct sensory prior coupling](direct_sensory_prior_coupling.md):
   the d_s offset is necessary. opt2 never found the d_s ≈ 45, g_s ≈ 0
   basin that opt1 uses. Its S-prior gains came through g_s instead, and
   g_s is the term that costs behaviour (s12→s7: psychometric 0.43).
3. **M prior.** In s303→s7 the model M is too high in the stim window
   (about 0.10–0.12 against data 0.06–0.08), and the move-aligned M
   overshoots from −60 ms. With g_m = d_m = 0, model M prior comes only
   through I→M (W_mi) and the thresholds.
4. **Move-aligned C prior.** opt2 fits it about 3× better than opt1, but
   in s303→s7 the model C falls in the last 20 ms while the data rises.
5. **Eval noise.** A single 20-block stim draw moves `L_S` by ±0.16
   (catalog, "S fit noise"). That is larger than most arm and restart
   differences, so one draw cannot rank fits.

---

## Plan (2026-10-29)

Steps are ordered so each one's result decides the next. Nothing below
step 0 runs without the user's go-ahead. ORCD steps are printed, not
submitted.

**Step 0: measurement (laptop, no fitting).**
- Overall R² on every S trajectory plot. Done the same day; see below.
- Score each opt2 source and fine-tune restart on the mean of 8 stim
  draws (bps 20, `make_shared_stimuli` seeds 1–8), as the catalog did
  for Stage A. Report mean ± sd for each term, plus S trajectory R²
  (pooled and per contrast) and behaviour. From here on,
  rank on that mean.

**Step 1: S prior through the offset basin (Stage B warm fine-tune, opt2).**
- Start from s303 and s303→s45 (they have the best behaviour).
- Hold g_s at 0 and free d_s, starting it at the opt1 value (≈ 45) and
  at a middle value (≈ 20). Free W_mm, W_mi, g_i and d_i as in the gm0
  fine-tune. Keep g_m = d_m = 0, the retinal parameters and the rest at
  the source values. Use the same loss and window as gm0 (window unset
  ≈ 80 ms, stim×choice, S prior on, `m_pre_weight` 3 as fitted, rank at 1).
- Question: does the offset-only basin lift the S prior near 0 ms and
  bring prior S below about 0.15, without the behaviour loss that g_s
  causes?
- Needs a `--freeze-hold` mask with g_s frozen at 0, plus a d_s init in
  the warm start. Before writing the submit script, state aim, window,
  stratum, freeze and what the loss sees.

**Step 2: S trajectory (Stage A; changes both arms).** Aim (user,
same day): the **amplitude of the early bump**, above all at the higher
contrasts. c=0 is mostly noise, and the late rise after about 100 ms is
not a target. Current fits overshoot the c=1 bump by 20–25% (model
0.62–0.65 against data 0.52, peak at 48 ms against 52 ms). See the
2026-10-29b breakdown below for how the current loss treats contrast and
time. Changes to decide on (each one changes the loss, so each needs the
user's go-ahead):
- (i) **Window:** cap every contrast at 100 ms. Today c=1 stops at 90 ms
  and the others at 120 ms, so the late rise is scored at c ≤ 0.25.
- (ii) **c=0:** drop it from `L_S`. Its SSE weight is already 0 (×2c),
  but its SNR term is unweighted and adds 0.11–0.14, about 20% of `L_S`.
- (iii) **Contrast weight:** the SSE is already ×2c on an
  energy-normalized SSE. Options are to keep that and weight the SNR term
  the same way, or to make it steeper (e.g. c²) so c=1 dominates.
- (iv) **Amplitude term:** an explicit penalty on bump peak height (e.g.
  max over 30–70 ms, model vs data) at c ≥ 0.25, if the SSE alone keeps
  letting the c=1 overshoot through.
- Then refit Stage A (retinal only) under the agreed loss and rescore.
  The older target and model options (an early-latency region subset,
  onset dispersion in the retinal front end) stay in reserve in case the
  bump amplitude still cannot be matched.
- Report an S R² that matches the aim (contrast-weighted, 0–100 ms, no
  c=0) next to the current pooled R². This needs the user's OK too.

**Step 3: M prior overshoot and late move-aligned C (after step 1).**
- First find out which parameter sets model M prior level: W_mi, W_mm
  and the two thresholds. Do this with local sweeps from the step 1 best
  fit, without fitting.
- If the thresholds drive it, a fine-tune that also frees them is the
  next arm. If W_mi drives it, the C/M trade-off is structural with
  g_m = 0, and freeing g_m/d_m again is the question to ask the user.

**Selection rule for every step.** Rank on the 8-draw mean fair. Accept
a fit only if the model C prior rises in both windows and behaviour is
no worse than its source (held-out check only).

---

## 2026-10-29 — overall R² on S plots

**Code.**
- `plot_S_diff_by_contrast_side_with_data(..., r2=None)` prints the
  overall R² on the S trajectory figure (`S_fit`). `plot_one` passes the
  value from `compute_sse_stim_right` (`total_gof_r2`). That R² pools
  over contrasts: `1 − Σ_c SSE_c / Σ_c SST_c`, with each SST around that
  contrast's own mean. It covers the scored window (90 ms at c=1,
  120 ms otherwise), not the full 128 ms drawn.
- The report builder adds an S traj R² column to the table and to the S
  caption. The previous builder is kept as `make_ft_report.py.bak_pre_s_r2`.
- An S-prior-curve R² was added to the prior panels the same day, then
  taken out at the user's request (2026-10-29b). The prior panels and the
  report show no S R² now.

**Re-plot.** `scripts/_tmp_integ_ft_replot_s_r2.py` redrew `S_fit` and
`prior_effects` for the four report fits. It used the same shared stim
as the eval and updated `traj` in each arm's `*_ft_gm0_plot_summary.json`.
`L_S` and prior S match the eval JSON to three decimals. The report PDF
is rebuilt (5 pages, no overfull boxes; the table uses `tabcolsep` 3 pt).

| fit | `L_S` | S traj R² (pooled) |
|---|---:|---:|
| opt1 s45→s45 | 0.523 | 0.34 |
| opt1 s7→s7 | 0.528 | 0.30 |
| opt2 s12→s7 | 0.556 | 0.31 |
| opt2 s303→s7 | 0.593 | 0.22 |

## 2026-10-29b — how the S loss treats contrast and time

Same shared stim (bps 20, seed 12345), the four report fits, scratch
script (per-contrast output of `compute_sse_stim_right`). Contrasts are
0, 0.0625, 0.125, 0.25 and 1.

**How `L_S` is built today** (`compute_sse_stim_right`):
- Per contrast: SSE normalized by the data energy, × 2c, plus an SNR
  mismatch term (Savitzky–Golay, dB) that is **not** weighted by
  contrast.
- Window: c=1 the first 90 ms; every other contrast the first 120 ms.

**opt2 s303→s7** (the other three fits follow the same pattern):

| c | window | weighted SSE | SNR | R² | R² 0–100 ms | data peak (0–70 ms) | model peak |
|---|---:|---:|---:|---:|---:|---|---|
| 0 | 120 | 0.000 | 0.112 | −3.75 | −2.73 | 0.127 @ 68 ms | 0.005 |
| 0.0625 | 120 | 0.062 | 0.004 | −0.12 | −0.23 | 0.111 @ 46 ms | 0.004 |
| 0.125 | 120 | 0.047 | 0.022 | 0.50 | 0.47 | 0.091 @ 68 ms | 0.143 |
| 0.25 | 120 | 0.072 | 0.028 | 0.50 | 0.31 | 0.217 @ 68 ms | 0.250 |
| 1 | 90 | 0.219 | 0.026 | 0.61 | 0.61 | 0.517 @ 52 ms | 0.649 @ 48 ms |

`L_S` 0.593 = SSE 0.400 + SNR 0.193.

- **c=0 enters only through SNR,** but it is the largest SNR term
  (0.11–0.14 across the four fits). It also pulls the pooled R² down:
  its own R² is about −3.5.
- **c=1 is the largest SSE term** (0.17–0.22). The model bump is 20–25%
  too high in every fit: 0.62–0.65 against 0.52.
- **c=0.0625:** the data shows a bump of 0.11 at 46 ms. The model has
  none.
- **c=0.125 and 0.25:** the data is still rising at 70 ms, so there is
  no separate early bump to match. Cutting at 100 ms lowers R² at
  c=0.25 (0.50 → 0.31), because the model matches the late part better
  than the early part.
- **Overall:** the S trajectory loss does not miss because of contrast
  weighting on the SSE. It misses because c=0 noise enters through the
  SNR term, and because the c=1 amplitude is too high.

## 2026-10-29c — S trajectory loss: 0–100 ms, c=0 dropped

**Decision (user, same day):** options (i) and (ii) from step 2. Cap every
contrast at 100 ms and drop c=0 from `L_S`. The ×2c SSE weight stays.

**Code.** `compute_sse_stim_right(..., max_steps=50, drop_zero_contrast=True)`
in `model_functions.py`. These are the new defaults, so `fit_retinal`,
`fit_joint` and every eval and plot that calls it pick them up. Passing
`max_steps=None, drop_zero_contrast=False` gives the legacy loss (45 steps
at c=1, 60 otherwise, c=0 in). It is the only implementation: there is no
torch or numba copy. The pooled S R² follows the same window and contrasts.

**Parity.** On the shared stim, the legacy arguments reproduce the old
`L_S` exactly: 0.523 / 0.528 / 0.556 / 0.593.

| fit | legacy `L_S` (R²) | new `L_S` = SSE + SNR | new R² |
|---|---|---|---:|
| opt1 s45→s45 | 0.523 (0.34) | 0.455 = 0.354 + 0.101 | 0.61 |
| opt1 s7→s7 | 0.528 (0.30) | 0.470 = 0.374 + 0.096 | 0.58 |
| opt2 s12→s7 | 0.556 (0.31) | 0.477 = 0.371 + 0.106 | 0.59 |
| opt2 s303→s7 | 0.593 (0.22) | 0.541 = 0.434 + 0.107 | 0.51 |

New per-contrast values for opt2 s303→s7 (weighted SSE / SNR / R²):
c=0.0625 0.072 / 0.015 / −0.23; c=0.125 0.052 / 0.004 / 0.47;
c=0.25 0.109 / 0.063 / 0.31; c=1 0.201 / 0.025 / 0.61. c=1 is now
about 37% of the new `L_S`. Cutting at 100 ms raises the SNR term at
c=0.25 (0.028 → 0.063).

**Not comparable across the change.** `L_S`, fair and the Stage A seed
ranking (the selector's shared-stim `L_S`) all change meaning. Old runs
must be re-scored with the new defaults (or with the legacy arguments to
reproduce old tables). The report keeps the as-fitted (legacy) `L_S` and
adds the new `L_S` and R² beside it.

**Data normalization (written down the same day, from
`build_fit_traj_targets_from_cache.py` / `build_mean_data_im_from_cache.py`).**
- **Data curve per contrast:** stim-R correct trials in each act-prior
  block. Spike counts in 12.5 ms bins sliding by about 2.08 ms
  (0–200 ms, 96 bins), averaged over trials per cell. Then the RMS over
  all cells of the 10 regions, `sqrt(Σ mean² / n / 0.0125)`. There is no
  per-cell baseline, z-score or amplitude normalization. Average the two
  blocks, then subtract the mean of the first 7 bins (spikes from about
  0–25 ms after stim onset). That is the data's 0 line: an early
  post-onset level, not a pre-stim baseline.
- **Model curve:** the signed S[1]−S[0] on stim-R trials, in model units,
  with `baseline_R=0`. The loss normalizes the SSE per contrast by `Σd²`.
- **Possible confounds for the early-bump aim (inferred from the code,
  not yet checked):**
  - **Time axis.** The plot and the loss put data bin i at 2i ms. The
    bin actually starts at about 2.08i ms and averages the next 12.5 ms,
    so the data sit roughly 6–10 ms later than drawn: the "52 ms" peak
    is centred near 60 ms.
  - **Smoothing.** The data are 12.5 ms boxcar-smoothed and the model is
    not, so a sharp model bump reads taller and steeper than the data.
  - Either one could account for part of the c=1 overshoot and of the
    onset mismatch. Check both before the Stage A refit: box-filter the
    model by 12.5 ms and put both on the true bin-centre time axis.

## 2026-10-29d — binning / time-axis check (S trajectory)

**Script:** `scripts/_tmp_s_binning_check.py` (one process per arm, same
shared stim as the report).
- **Bin windows.** Confirmed in brainbox `bin_spikes2D` (pre=0): data
  index i = 6j + ts covers [12.5j + 2ts, +12.5) ms after stimOn, so its
  true centre is about 2.08i + 6.25 ms. The fit puts it at 2i ms.
- **"Binned" model.** Model step k ([2k, 2k+2) ms) is box-averaged over
  each data bin's window, then zeroed on its first 7 bins like the data.
- **Outputs:** report folder `figs/s_binning_check/S_binning_*.{svg,png}`
  (true time axis; data solid, raw model dotted, binned model dashed) and
  `s_binning_check_<arm>.json`.

| fit | c=1 data peak (true t / as fitted) | raw model peak | binned model peak | R² raw → binned (0–100 ms, c>0) |
|---|---|---|---|---|
| opt1 s45→s45 | 0.517 @ 60 / 52 ms | 0.622 @ 49 | 0.599 @ 48 | 0.61 → 0.47 |
| opt1 s7→s7 | 0.517 @ 60 / 52 | 0.628 @ 49 | 0.607 @ 48 | 0.58 → 0.45 |
| opt2 s12→s7 | 0.517 @ 60 / 52 | 0.627 @ 49 | 0.606 @ 48 | 0.59 → 0.45 |
| opt2 s303→s7 | 0.517 @ 60 / 52 | 0.649 @ 49 | 0.633 @ 48 | 0.51 → 0.36 |

**Reading.**
- **Smoothing is not the overshoot.** The 12.5 ms box lowers the model
  c=1 peak by only about 3% (0.63 → 0.61). The bump is still 16–22% too
  high.
- **Timing is.** On the true axis the data c=1 peak is at about 60 ms,
  and the model peaks about 11–12 ms earlier. The fit compares index to
  index (data bin i against model step i), so the half-bin (6.25 ms)
  plus the 4% stretch is never seen. Stage A has been fitting a
  latency that is about 10 ms too early. With the model put on the
  data's real windows, R² drops at c=1 (0.61–0.70 → 0.42–0.54) and at
  c=0.0625.
- **Contrast-independent early transient.** The data rise at 6–25 ms is
  the same at every contrast, and it sits inside the 7-bin zeroing
  window. That is too early and too contrast-blind to be the visual
  response the retina model makes. It sets where the data's 0 line
  lands.
- **The SNR term blows up when the model is binned.** `L_S` goes from
  about 0.5 to about 3.1 although R² falls only moderately. This is
  probably the SNR term (a smooth binned model against noisy data), but
  SSE and SNR were not split for the binned case.

**Options (not decided):**
- (A) Score the S loss on the data's real bin windows: bin and zero the
  model inside `compute_sse_stim_right`, or before it. This would let
  Stage A fit the right latency. The SNR term then needs a rethink:
  drop it, or compute it on the raw model.
- (B) Rebuild the data target on a 2 ms grid aligned to bin centres. The
  12.5 ms smoothing stays, but the shift goes away.
- In both cases, refit Stage A after.

## 2026-10-29e — the binning offset is in every neural loss

**Targets.** Every neural target uses the same overlapping bins: 12.5 ms
boxcar, 2 ms sub-offsets, 6 per bin, so one index step is about 2.08 ms.
Each bin is labelled at its start (`bin_spikes2D`).
- **S trajectory** (`avg_mean_R`) and **I/M trajectories**
  (`mean_data_results`): `build_fit_traj_targets_from_cache.py`. Stim
  window 0–200 ms (96 bins); move window −150–0 ms (72 bins).
- **I/M prior distance** (`data_act_block_during*.npy`) and the **S prior
  sidecar**: BWM `block_analysis_allsplits.py`, with `b_size` 0.0125 and
  `sts` 0.002 (the same construction).

**Losses.** Every loss compares data index i to model step i (2 ms):
- I/M trajectory (`loss_plot_diff_by_condition_with_data`): the stim
  window from the start (`[:T]`), the move window from the end
  (`[-T:]`).
- Prior distance (`loss_prior_effect`, default window): 40 model steps
  against the first 40 data bins.
- S prior sidecar: drawn on `linspace(0, 80, L)`.
- S trajectory (`compute_sse_stim_right`): `[:T]`.

**Size of the offset** (data true bin centre minus model time):
- **Stim-aligned:** about +6 ms at 0 ms, +8 ms at 50 steps (100 ms), and
  +12 ms at 75 steps (150 ms). The data are later than the model thinks.
- **Move-aligned:** the last data bin is [−2.5, +10) ms around movement
  onset (centre +3.75), so the offset is about +5 ms at movement onset,
  falling to about −1 ms at −140 ms.

**Changing the model dt to about 2.08 ms is not the fix.**
- It removes only the 4% stretch. The half-bin shift (6.25 ms) and the
  12.5 ms smoothing stay.
- The data grid is not uniform anyway (sub-offsets 0, 2, …, 10 then
  12.5).
- A new dt changes the Euler step and every quantity counted in steps
  (delays, `steps_before_obs`, ITI, the prestim offset gate). Every stage
  would need recalibration and a refit.

**Proposed fix (not implemented).** One "observe like the data" operator
for all neural losses. It box-averages the 2 ms model trace over each
data bin's real window (a fixed bins × steps matrix), applies the same
zeroing as each target, and then compares index to index as now.
- It can be checked on the existing fits, as 10-29d did for S.
- The SNR term in `L_S` would need a rethink (see 10-29d).
- For prior distance, it is strictly more faithful to bin the model
  trajectories before taking distances. Binning the final distance curve
  is close when the curves are smooth.
- Every loss changes, so Stage A and Stage B both need refits, and old
  scores are not comparable.

## 2026-10-29f — model dt = 12.5/6 ms (user's suggestion), and corrections to 10-29e

**Two corrections to 10-29e.**
- **The data grid is effectively uniform.** The sub-offsets 0, 2, …, 10
  and then 12.5 ms are within 0.4 ms of an even 12.5/6 ≈ 2.083 ms grid.
- **dt is already plumbed.** dt is `model_params['dt']`. The τ's and the
  retinal delay are in ms and converted with dt, and
  `_compute_dt_dependent_params` derives `post_action_steps` and
  `prestim_offset_start` from it. A new dt is not a recalibration of
  everything.

**Proposal: model dt = 12.5/6 ms and compare bin by bin.**
- **What it removes.** The 4% stretch disappears: model step i and data
  point i then start at the same time. Windows already counted in data
  bins (the 50-step S cap, 40-bin prior, 72/96-bin trajectories) become
  correct as they stand.
- **What it leaves.** Data point i averages [t_i, t_i + 12.5) ms, so it
  is centred 6.25 ms after t_i. At dt = 12.5/6 that is exactly 3 model
  steps. Compare data i with the mean of model steps i…i+5 (the exact
  box), or simply with model step i+3. The box adds only about 3% at
  the c=1 peak (10-29d), so the 3-step shift does almost all of it.
- **Move-aligned.** The last data bin runs to +10 ms after movement
  onset, so the model's move window needs about 5 steps after movement
  onset to match. Not yet checked whether the model trace has them.
- **To check before switching.**
  - The places that hard-code 2 ms per step: `mean_S_by_contrast`
    T=65, `plot_window/2`, the legacy 45/60, plot time axes.
  - Numba parity at the new dt.
  - Whether stimulus generation and the RT conversion are in steps or ms.
  - Old fits must be re-scored or refit at the new dt (the Euler step
    changes slightly).

## 2026-10-29g — bin-by-bin comparison everywhere; Stage A refit setup

**Convention (user, same day).** One model step = one data bin. Model and
data are compared bin by bin over equal lengths, with no conversion
through real time and no smoothing on the model side. The 12.5/6 ≈ 2.08 ms
per bin is only for reading model parameters and windows in ms. The
10-29e/f proposals (binning operator, dt change, 3-step shift) are
dropped.

**Audit.** Each loss and whether it is bin by bin:
- **Production** (window unset): yes, with equal lengths. S trajectory
  50 bins (0–100 ms label); I/M trajectory T=72 against the data
  windows; prior distance 40 bins.
- **S prior sidecar:** 38 data bins against a 40-step model, truncated
  to 38. Left as is, because the sidecar is the canonical 80 ms data
  product.
- **Custom windows** (`prior_window_ms` / `im_window_*_ms` ≠ 80): no.
  They used `round(w/dt)` model steps (150 → 75), linearly resampled
  onto the data's 72 bins. **Fixed:**
  - `window_bins(w)` gives 80 → 40, otherwise `round(w / DATA_BIN_STEP_MS)`
    (150 → 72).
  - `resolve_im_window` returns that bin count with `resample=False`.
  - The numpy `loss_prior_effect` takes its compare length from
    `window_bins(plot_window)`.
  - `DATA_BIN_STEP_MS = 12.5/6` is defined in `model_functions.py` and
    used only for this conversion.
- **Check (opt2 s303→s7, shared stim).** Production prior and traj match
  the eval JSON exactly (0.502258 / 0.316456). At 150 ms: T=72, no
  resample; stim M and choice I are 72 model steps against 72 data bins;
  I/M traj T_post/T_pre 72/72.
- **Not touched, flagged:**
  - **RT model vs data** (`_tmp_perf_rt_model_vs_data.py`) uses model
    steps × dt (2 ms) against real RT. RT has no bins, so under this
    convention it should use 2.08 ms per step: model RTs are about 4%
    short. Behaviour is held out, so the fit is unaffected.
  - **Torch `_loss_prior_effect_torch`** (gradient_mode, not used by the
    production `de_cma_local` fits) scores min(model, data) = 72 bins with
    the window unset, where numpy scores 40. This discrepancy predates
    this change.
  - **Plot time axes** still draw step × 2 ms.

**Stage A refit setup** (`scripts/submit_fit_targets_integ.sh`,
`scripts/run_fit_integ_select_slurm.sh`):
- **Naming.** Hybrid and Stage B names now follow `RETINAL_TAG`:
  `retinal_<X>` → `stage_b_hybrid_WEIGHTS_REL_retinal_<X>_s<seed>.json`
  and `stageB_hold_<X><seed>_…`. `retinal_newS` keeps its old names.
  Before this, a new Stage A tag that picked s34 would have written into
  the existing `stageB_hold_newS34_*` directories.
- **`STEP=retinal SELECT=1`** chains the selector after the Stage A jobs
  (afterany).
- **`SUBMIT_FITS=0`** makes the selector stop after ranking. That is the
  default for `STEP=retinal`; `STEP=all` is unchanged and still submits
  Stage B.
- **Dry run** with a fake `sbatch`:
  - `STEP=retinal`: 5 Stage A jobs (s56 34 78 89 202, `OUT_TAG
    retinal_newSbin`, opt1 targets dir, whose `avg_mean_R` is the same as
    opt2's), then the selector after them with SUBMIT_FITS=0.
  - `STEP=fits`: hybrid `…_retinal_newSbin_s34.json` and `OUT_TAG
    stageB_hold_newSbin34_full_mpre3_meancell_integ_opt2_shuffleslope`.
- **Gates.** `BEAT_LOSS` 0.85 and `L_THRESHOLD` 2.0 were set for a frozen
  front-end `L_S` of about 0.8. The new `L_S` of the current fits is
  0.45–0.54, so both stay permissive.

**Aim / window / stratum / freeze / what the loss sees (Stage A refit).**
- **Aim:** fit the S trajectory early bump (higher contrasts) under the
  new S loss.
- **Window:** S trajectory 0–100 ms label = 50 bins, bin by bin. No I/M
  or prior windows: Stage A is `L_S` only.
- **Stratum:** stim-R correct trials per contrast (the `avg_mean_R`
  recipe); c=0 is dropped.
- **Freeze:** retinal mask none, prior gains 0, network W/θ at script
  defaults.
- **The loss sees:** `L_S` only = per contrast (c>0) energy-normalized
  SSE ×2c + SNR term, over the first 50 bins.
- **Submit command (print only; the user submits):**

      STEP=retinal RETINAL_TAG=retinal_newSbin SELECT=1 SUBMIT_FITS=0 \
        bash scripts/submit_fit_targets_integ.sh

  The code has to be on ORCD first (develop → main copy as usual).

**Local smoke (same day).** Stage A through `scripts/run_fit_retinal.py`,
set up as the Slurm runner does: `FIT_TARGETS_DIR=fit_targets_integ/opt1_rawslope`,
work dir `fit_workdirs/retinal_retinal_newSbin_smoke_retinal_s34`.
- **Settings:** seed 34, `de_cma_local`, DE/CMA 1 iteration, popsize
  2/4, bps 10/20, 1 stage-2 stim seed, local polish ≤ 60 s.
- **A first try at bps 2/3 failed closed,** as designed: S buckets with
  fewer than 10 trials → NaN → 1e11 → `failed_stage1`. Smokes need the
  production bps.
- **Rerun:** 191 s, `FIT_DONE`, no `EXC@`. DE1 0.761 → CMA 1.065 →
  final 0.3955 (own stim). Run dir:
  `openalyx/models/retinal_run_fr_retinal_newSbin_smoke_retinal_masknone_s34/`.
- **Selector ranking** (`plot_retinal_fit_s.py`, bps 20, seed 12345, new
  loss) on that final: shared `L_S` 0.536 (SSE 0.425 + SNR 0.111), R²
  0.60. `summary.json` has the `rows[0].json` field the selector reads.
  Output is in `selector_rank_smoke/` inside the smoke run dir.
- The smoke parameters sit at their bounds after one iteration
  (W_as 50, alpha_d 20, tau_a 100). It is a plumbing check, not a fit.
- **Not smoked:** a Stage B `fit_joint` run (the selector does not submit
  it with SUBMIT_FITS=0). Its `L_S` call is the same
  `compute_sse_stim_right`, which the eval scripts above already ran.

## 2026-10-29h — RT at 12.5/6 ms per step; report re-plotted

**Code.**
- `model_functions.model_step_s()` returns `DATA_BIN_STEP_MS / 1000`. The
  report RT plot (`scripts/_tmp_perf_rt_model_vs_data.py`,
  `plot_one_json`) and paper Figure 5 (`fig5_plot.py`) now convert model
  RT steps with it instead of `mp['dt']` (2 ms).
- Old diagnostic `_tmp_*` RT scripts are unchanged. Figure 5 has not been
  regenerated.
- `scripts/_tmp_integ_ft_replot_s_r2.py` now also redraws the act-prior
  behaviour and updates `rt` in the plot summaries.

**Report.** All figures for the four fits are redrawn: S (R², 0–100 ms,
c>0), I/M traj, prior, psychometric and RT. The PDF is rebuilt (5 pages,
no overfull boxes).

| fit | psychometric R² | RT R² 2 ms → 2.083 ms | RT split (con / inc) |
|---|---:|---|---|
| opt1 s45→s45 | 0.87 | 0.41 → 0.26 | −0.12 (0.49 / −0.77) |
| opt1 s7→s7 | 0.93 | 0.52 → 0.39 | 0.23 (0.52 / −0.09) |
| opt2 s12→s7 | 0.43 | 0.08 → −0.17 | −4.77 (0.80 / −10.72) |
| opt2 s303→s7 | 0.85 | 0.23 → 0.05 | −1.17 (0.59 / −3.05) |

- The model RT curve against contrast is too steep: too slow at low
  contrast (about 0.47 against 0.42 s at c=0) and too fast at c=1
  (about 0.12 against 0.18–0.19 s).
- A 4% longer step stretches the slow end, so RT R² falls in every fit.
  The psychometric is unchanged.
- The RT numbers in the 10-29 catalog tables (stage_b_fit_variants) were
  at 2 ms per step.

## 2026-10-29i — Stage A `retinal_newSbin` results (new S loss)

**Runs.** Five Stage A seeds (`retinal_run_fr_retinal_newSbin_retinal_masknone_s{34,56,78,89,202}`),
all `FIT_DONE`, run with the new loss: 0–100 ms, c>0. The selector rank is
`retinal_s_fit_plots_retinal_newSbin_bps20_seed12345/summary.json`. It ranks
on a single stimulus draw and puts s34 first (shared `L_S` 0.633).

**Multi-seed re-eval** (`scripts/_tmp_stageA_newSbin_eval.py`). Each
final was run on 5 stimulus seeds (bps 20) and scored with the current
loss. The old `retinal_newS` finals were run on the same stimuli. The
output is in `multiseed/` inside the rank folder (`multiseed.json`, and
an S overlay at seed 12345 for each final).

| run | recorded | `L_S` | SSE | SNR | R² |
|---|---:|---|---|---|---|
| newSbin s56 | 0.367 | 0.99 ± 0.76 | 0.331 ± 0.016 | 0.66 ± 0.77 | **0.680 ± 0.008** |
| newSbin s202 | 0.331 | 1.12 ± 0.84 | 0.343 ± 0.022 | 0.78 ± 0.85 | 0.670 ± 0.024 |
| newSbin s78 | 0.446 | 1.03 ± 0.79 | 0.349 ± 0.024 | 0.69 ± 0.80 | 0.653 ± 0.034 |
| newSbin s34 | 0.396 | 0.69 ± 0.39 | 0.358 ± 0.055 | 0.33 ± 0.39 | 0.644 ± 0.053 |
| newSbin s89 | 1.236 | 0.83 ± 0.55 | 0.371 ± 0.040 | 0.46 ± 0.55 | 0.618 ± 0.057 |
| newS s34 (current Stage B source) | 0.909 | 0.65 ± 0.33 | 0.348 ± 0.051 | 0.30 ± 0.31 | 0.657 ± 0.052 |
| newS s202 | 0.368 | 0.91 ± 0.65 | 0.362 ± 0.016 | 0.55 ± 0.66 | 0.650 ± 0.027 |
| newS s78 | 0.445 | 0.69 ± 0.39 | 0.353 ± 0.049 | 0.34 ± 0.38 | 0.637 ± 0.052 |
| newS s56 | 0.905 | 0.68 ± 0.36 | 0.363 ± 0.065 | 0.32 ± 0.35 | 0.628 ± 0.069 |
| newS s89 | 0.570 | 0.70 ± 0.29 | 0.495 ± 0.131 | 0.20 ± 0.21 | 0.549 ± 0.101 |

**Findings.**
- **The SNR term is noise, and it decides the ranking.**
  - Its standard deviation across stimulus draws is about equal to its
    mean (±0.2–0.85).
  - Almost all of it comes from c=0.0625. The data SNR there is 12.5 dB.
    The model's SNR is 5–11 dB, and it moves ±4–7 dB from draw to draw
    because the model's 6.25% curve is near flat and noisy at bps 20.
    The other contrasts add ≤0.04 each.
  - SSE and R² are stable (SD ≤0.06 for most runs). The single-draw
    selector order (s34 > s89 > s56 > s78 > s202) is close to the
    reverse of the R² order.
- **newSbin vs newS: a small, real gain at the top.**
  - Best R² is 0.680 ± 0.008 (newSbin s56) against 0.657 ± 0.052 (newS
    s34). newSbin seeds are also more consistent from draw to draw.
  - c=1 bump: data peak 0.517, newSbin s56 0.53, newS s34 0.57.
  - c=0.25: newSbin matches the about 0.22 bump. newS overshoots it
    (about 0.27).
  - Onset is still late in every fit: the model's c=1 rise comes about 7 ms
    after the data's.
  - R² at c=0.0625 is negative in every fit (−0.5 to −0.75). The model
    stays flat there while the data shows a small early rise.
- **The s89 local refine kept a worse loss.** Powell after CMA went
  0.513 → 1.236 and was accepted, so its recorded loss is the refine's.
  It looks like a re-draw of the stimulus seeds rather than a real
  regression, but `run_fit_retinal` should keep the better of the two.
  Not fixed.
- **Shared figures:** the selector's own S plots
  (`plot_retinal_fit_s.py`) do not show R².

**Open.**
- Pick the Stage B source. By R², newSbin s56; by single-draw `L_S`,
  s34.
- Decide whether the selector should rank on a multi-seed mean, or on
  SSE / R² without the c=0.0625 SNR term.

**Decision (same day).** The user chose **newSbin s56** as the Stage B
source. The command was printed for ORCD, not submitted:
`STEP=fits RETINAL_TAG=retinal_newSbin RETINAL_JSON=<models>/retinal_run_fr_retinal_newSbin_retinal_masknone_s56/retinal_final_loss0p3673_20261029-163804.json bash scripts/submit_fit_targets_integ.sh`
→ hybrid `stage_b_hybrid_WEIGHTS_REL_retinal_newSbin_s56.json`, OUT_TAG
`stageB_hold_newSbin56_full_mpre3_meancell_integ_{opt1_rawslope,opt2_shuffleslope}`.
