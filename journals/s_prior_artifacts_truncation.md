# Spurious S prior distance: truncation artefacts and the Phase 4b residual

**Scope:** why the simulated sensory population (S) showed a significant prior-distance effect versus contrast-matched shuffle controls when the generative model had **no** P→S coupling, and how that was traced to an analysis-window truncation artefact.

**Status:** resolved. The dominant cause was zero-padding of truncated trials in `bin_trace_segment`; with fill-from-next-ITI and an 80 ms S window, Phase 4b (all `g_*=d_*=0`) is null everywhere. The remaining absence-condition S signal is real but **I/M-mediated**, and is itself largely a split-composition effect — see [split conditioning vs unsplit](split_conditioning_vs_unsplit.md).

Sources: dated entries 2026-06-17 (Phases 0–5), 2026-06-18 (Q1/Q2 + truncation), 2026-06-19, 2026-06-19b.

---

## Goal

**Investigate, debug, and identify why there is a significant S prior-distance effect vs contrast-matched shuffle controls in the absence case (`g_s=0`, `d_s=0`), when the generative equations have no direct P→S coupling.**

The question was never whether absence shows an effect (it does). The question was **what mechanism produces it**: analysis artefact, unintended code path, or something else.

**Success criteria**

1. Name the dominant cause(s) with quantitative evidence.
2. Show whether the effect survives controls that should remove each candidate.
3. Decide whether absence shuffle significance can be used as any kind of null, or must be discarded for recovery inference.

Two follow-up questions framed the 06-18 work:

- **Q1** — Why does I/M prior modulation (standard absence) increase S prior distance ~12× relative to Phase 4b?
- **Q2** — Why is there any S prior difference when all `g/d = 0` (Phase 4b)?

---

## Answer (final)

1. **`bin_trace_segment` zero-padded trailing bins** when a trial ended before the 150 ms analysis window. At high contrast most correct trials end well before 110 ms, so the mean S trajectory was a mixture of real signal and structural zeros. The two prior groups have different RT distributions → asymmetric zero-padding → spurious distance. This explains essentially the whole Phase 4b residual and inflated Q1.
2. With **fill-from-next-ITI** and an **80 ms S window**, Phase 4b S/I/M prior distance is null across all contrasts and splits (S p=0.78).
3. **Standard absence** (fitted I/M, `g_s=d_s=0`) retains a large, genuine S prior distance (curve_mean 0.798, p=0) that is mediated by I/M modulation changing choice composition within the f1/f2 splits — not by any hidden P→S path.
4. There is **no code bug** coupling P into S when `g_s=d_s=0` (Phase 4a audit).
5. **Contrast-matched absence shuffle is not a valid causal null** for recovery: true ≫ shuffle even when all generative prior coupling is zero (pre-fix), and post-fix the surviving absence signal is a composition effect.

---

## Model setup (absence)

| Parameter | Value |
| --------- | ----- |
| Condition | absence |
| `g_s`, `d_s` | 0.0, 0.0 (no sensory prior pathway) |
| `g_i`, `d_i` | 189.68, 21.56 (fitted; I still modulated by P) |
| Sessions | 40 × 6 blocks |
| Trials | 12,036 total |
| RNG seed | 42 (later 123) |
| Prior column | `p_subjective_probabilityLeft` (ITI subjective P ≥ 0.5 vs < 0.5) |
| Null scheme | Contrast-matched label shuffle, `nrand=100` |
| Splits | 4 × `act_block_duringstim` |
| Output | `output/absence_shuffle_debug/` |

**Causal structure:** with `g_s=d_s=0` the S equation contains no P and no `block_side`. S at stim depends on the external stimulus `S0`, a negligible `W_ss`, and the adaptation state `a`.

### Excluded from consideration — trial structure / history as mechanism

Do **not** invoke block epoch, trial index, stimulus-sequence position, or cross-trial carryover to explain the Phase 4b residual (or S trajectory differences under splits that already fix stim and choice side):

1. **S is almost completely feedforward.** Fitted `W_ss ≈ 7.6×10⁻⁵`, `tau_s = 20` ms, long ITI → negligible cross-trial S memory. Within a trial, S is driven by the current `S0`.
2. **`block_side` does not enter the dynamics** when all prior modulations are off — no `g_*`/`d_*`, no block term in the S/I/M ODEs.
3. **Splits fix stim and choice side.** At a given contrast bin, groups are compared on the same stimulated side and outcome class; `block_side` / `trial_in_block` imbalance across label groups is not a lawful driver of different S trajectories under Phase 4b.
4. **Adaptation `a` at stim ≈ 1** for both prior groups (Phase 2 covariate check).

---

## Investigation plan (Phases 0–5)

Ordered from fastest / most diagnostic to heavier model changes.

### Phase 0 — Baseline characterization

| Task | Tool | What it tests |
| ---- | ---- | ------------- |
| Combined + per-shuffle `curve_mean` | `s_prior_shuffle_*.csv` | Confirm effect size, not p-value direction |
| Per-split × contrast true vs shuffle | `s_prior_split_contrast_shuffle.csv` | Rule out contrast imbalance; localize to splits |
| Block-confound distributions | `--block-confound-plots` | RT, contrast, S peak time: P-block-L vs R per split |
| Subjective vs block prior grouping | `--prior-compare` | Is the effect specific to ITI P vs `probabilityLeft`? |

Read of the per-contrast data: true > shuffle at every contrast with adequate n; `*_f2` > `*_f1`; strong high/low count asymmetry per split.

### Phase 1 — Label / covariate checks (no model changes)

1a. Block prior vs subjective P (`--prior-compare`); 1b. covariate balance (RT, contrast marginals); 1c. error-rate asymmetry (`*_f1` vs `*_f2` counts by prior group).

### Phase 2 — Covariate / adaptation diagnostics — DONE

2a. Record `a` at stim onset (`extract_trial_table`); 2b. S trajectories by contrast with prior panels at fixed `block_side`; 2c. trial-history matched control (match on block_side, contrast, trial decile, session).

```bash
python simulate_recovery.py --phase2-adaptation --seed 42 --output-dir output/absence_phase2
```

### Phase 3 — Analysis / null-scheme checks

3a. Unrestricted vs contrast-matched shuffle (`--null-compare`). 3b. Split-sum artefact: per-split shuffle significance vs the combined sum (`stack_combined_timeframes`).

### Phase 4 — Simulation controls (ground truth)

| Control | Implementation | Notes |
| ------- | -------------- | ----- |
| **Random ITI labels** | Assign 0.8/0.2 at random, independent of the P trace | Collapses the standard-absence effect below the shuffle null |
| **Absence + no I/M prior** | `g_i=d_i=g_m=d_m=0` with `g_s=d_s=0` | `--phase4-no-prior-mod` — residual still ≫ shuffle (pre-fix) |

4a code audit and 4b no-prior-modulation runs are both below.

### Phase 5 — Decision for the recovery pipeline

| Outcome | Action |
| ------- | ------ |
| Absence shuffle unsuitable as a causal null | Use presence − absence + absence replicate null for injection only; do not interpret shuffle p as a generative test |
| Minor analysis fix (e.g. split sum) | Patch `simulate_recovery.py`, re-run Slurm |
| Residual effect only on `*_f2` | Consider restricting S-prior splits to `*_f1` for a cleaner sensory window |
| Effect persists after all controls | Trace the analysis pipeline / dynamics step by step |

Diagnostic commands:

```bash
# 1. Block confounds (RT, contrast, S peak)
python simulate_recovery.py --block-confound-plots --seed 42 --output-dir output/absence_confound
# 2. Subjective vs block prior
python simulate_recovery.py --prior-compare --seed 42 --output-dir output/absence_prior_compare
# 3. Null scheme comparison
python simulate_recovery.py --null-compare --seed 42 --output-dir output/absence_null_compare
```

---

## Pre-fix results

### Combined results, standard absence (4 splits summed)

From `combined_regde` over all `act_block_duringstim` splits.

| Metric | True labels | Shuffle nulls (n=100) |
| ------ | ----------- | --------------------- |
| `curve_mean` (time-avg distance) | **1.298** | min 0.118, **median 0.174**, max 0.251 |
| `curve_amp` (max − min) | **4.010** | min 0.460, med 0.644, max 0.951 |
| `early_mean_direct` (bins 0–4) | ~0 | ~0 |
| `gain_late_mean_direct` (bins 4+) | **1.375** | — |
| Shuffles with mean ≥ true | — | **0 / 100** |
| `p_mean`, `p_amp`, `p_offset`, `p_gain` | — | **0.0** |

True combined distance is ~**7.5×** the shuffle median on `curve_mean`.

### Per-split, per-contrast (standard absence, pre-fix)

Distance per row: time-mean of the bin-wise squared S trajectory difference (high vs low prior) within that contrast only, with contrast-matched shuffles preserving per-contrast high/low counts.

**`act_block_duringstim_r_choice_r_f1`**

| contrast | n_high | n_low | true mean | shuffle med | shuffle min | shuffle max | p(shuf≥true) |
| -------- | ------ | ----- | --------- | ----------- | ----------- | ----------- | ------------ |
| 0.0 | 53 | 360 | 0.457 | 0.048 | 0.009 | 0.164 | 0.0 |
| 0.0625 | 137 | 702 | 0.240 | 0.020 | 0.006 | 0.070 | 0.0 |
| 0.125 | 222 | 856 | 0.267 | 0.012 | 0.003 | 0.060 | 0.0 |
| 0.25 | 325 | 947 | 0.054 | 0.009 | 0.002 | 0.032 | 0.0 |
| 1.0 | 400 | 950 | 0.096 | 0.006 | 0.002 | 0.017 | 0.0 |

**`act_block_duringstim_l_choice_l_f1`**

| contrast | n_high | n_low | true mean | shuffle med | shuffle min | shuffle max | p(shuf≥true) |
| -------- | ------ | ----- | --------- | ----------- | ----------- | ----------- | ------------ |
| 0.0 | 315 | 47 | 0.448 | 0.050 | 0.012 | 0.219 | 0.0 |
| 0.0625 | 793 | 144 | 0.467 | 0.018 | 0.006 | 0.107 | 0.0 |
| 0.125 | 754 | 206 | 0.197 | 0.014 | 0.004 | 0.085 | 0.0 |
| 0.25 | 913 | 320 | 0.062 | 0.009 | 0.002 | 0.032 | 0.0 |
| 1.0 | 922 | 420 | 0.101 | 0.006 | 0.002 | 0.018 | 0.0 |

**`act_block_duringstim_l_choice_r_f2`**

| contrast | n_high | n_low | true mean | shuffle med | shuffle min | shuffle max | p(shuf≥true) |
| -------- | ------ | ----- | --------- | ----------- | ----------- | ----------- | ------------ |
| 0.0 | 80 | 107 | 0.355 | 0.042 | 0.012 | 0.214 | 0.0 |
| 0.0625 | 116 | 217 | 0.630 | 0.030 | 0.008 | 0.143 | 0.0 |
| 0.125 | 58 | 134 | 0.977 | 0.052 | 0.015 | 0.191 | 0.0 |
| 0.25 | 23 | 52 | 0.769 | 0.134 | 0.037 | 0.401 | 0.0 |
| 1.0 | 0 | 1 | — | — | — | — | — |

**`act_block_duringstim_r_choice_l_f2`**

| contrast | n_high | n_low | true mean | shuffle med | shuffle min | shuffle max | p(shuf≥true) |
| -------- | ------ | ----- | --------- | ----------- | ----------- | ----------- | ------------ |
| 0.0 | 124 | 96 | 0.311 | 0.037 | 0.010 | 0.176 | 0.0 |
| 0.0625 | 190 | 113 | 0.285 | 0.033 | 0.008 | 0.129 | 0.0 |
| 0.125 | 132 | 56 | 0.474 | 0.056 | 0.016 | 0.259 | 0.0 |
| 0.25 | 57 | 22 | 0.703 | 0.136 | 0.051 | 0.444 | 0.0 |
| 1.0 | 1 | 0 | — | — | — | — | — |

**Pattern:** true > shuffle at every contrast with adequate trials (18/20 rows); largest on error splits (`*_f2`), especially mid contrasts. `l_choice_l_f1` is block-left heavy and `r_choice_r_f1` its mirror; true distance is elevated in both directions.

### Random ITI labels (Phase 4 control)

Same absence trajectories (seed 42); only the grouping changes.

| | curve_mean |
| --- | ---------- |
| True ITI P | **1.30** |
| Contrast-matched shuffle null (median) | 0.17 |
| Random 0.8/0.2 labels (median, n=50) | **0.034** |

Random labels collapse the effect *below* the shuffle null → the label assignment matters, not the simulated dynamics. Output: `output/random_prior_test/absence/figs/random_prior_labels/`.

---

## Phase 2 results: covariate / adaptation diagnostics

Run: `output/absence_phase2/absence/figs/phase2_adaptation/` (absence, seed 42, 40 sessions, ITI subjective P).

| Candidate | Evidence | Conclusion |
| --------- | -------- | ---------- |
| **Adaptation `a` at stim** | Medians ~0.9913 both groups; diff ~7×10⁻⁵ | **Not a driver** |
| **ITI ‖S‖** | Medians ~10⁻¹⁶ | **Irrelevant** |

### 2a. Covariate table (high vs low ITI subjective P)

| split | metric | med(high) | med(low) | p |
| ----- | ------ | --------- | -------- | - |
| r_choice_r_f1 | a_at_stim_mean | 0.99142 | 0.99135 | 0.021 |
| r_choice_r_f1 | trial_in_block | 20 | 28 | ~10⁻²⁴ |
| r_choice_r_f1 | reaction_time | 73 | 80 | ~10⁻¹¹ |
| l_choice_l_f1 | trial_in_block | 25 | 21 | ~10⁻⁹ |
| l_choice_l_f1 | reaction_time | 80 | 72 | ~10⁻¹³ |
| l_choice_r_f2 | (all metrics) | — | — | n.s. |
| r_choice_l_f2 | trial_in_block | 20 | 27 | ~10⁻⁶ |

Full table: `phase2_covariate_mannwhitney.csv`. Covariate descriptives are **not** used as an explanatory mechanism (see "Excluded from consideration" above).

### 2c. Matched-bin distances

Match bins: `(block_side, contrast, trial_in_block decile, session)`.

| split | full distance | matched-bin distance | fraction | n bins |
| ----- | ------------- | -------------------- | -------- | ------ |
| r_choice_r_f1 | 0.203 | 0.043 | **0.21** | 100 |
| l_choice_l_f1 | 0.234 | 0.077 | **0.33** | 84 |
| l_choice_r_f2 | 0.537 | 1.067 | — | 1 (unreliable) |
| r_choice_l_f2 | 0.324 | — | — | 0 |

**Artifacts:**

```
output/absence_phase2/absence/figs/phase2_adaptation/
  phase2_covariate_mannwhitney.csv
  phase2_prior_correlations.csv
  phase2_matched_history_summary.csv
  phase2_matched_history_bins.csv
  phase2_a_at_stim_by_prior_split.png
  phase2_s_norm_iti_by_prior_split.png
  phase2_trial_in_block_by_prior_split.png
  phase2_s_traj_{split}_{block_left|block_right}.png
  phase2_summary.json
```

---

## Phase 4a results: code audit — no bug

**Question:** is there an unintended code path that couples P (or block) into S when `g_s=d_s=0`? **Verdict: no.** P terms are present in the S update but multiplied by zero.

### `load_fitted_model` — `g_s`/`d_s` override

```383:405:simulate_recovery.py
def load_fitted_model(g_s=0.0, d_s=0.0, json_path=None):
    ...
    mp.update(meta.get("model_params", {}))
    mp.update(meta["W"])
    mp["g_i"] = meta["g"]["g_i"]
    ...
    mp["g_s"] = float(g_s)
    mp["d_s"] = float(d_s)
    ...
    mf._update_model_params_for_dt(mp, DT_MS)
```

- `g_s`/`d_s` are assigned **after** the JSON `model_params` and `W` merge — fitted JSON values cannot leak through.
- `_update_model_params_for_dt` only updates `tau_*`, `post_action_steps`, `prestim_offset_start`.
- All absence entry points call `load_fitted_model(g_s=0.0, d_s=0.0)` or pass explicit zeros via `process_condition`.

### S ODE — all backends (NumPy, Numba, Torch)

```891:901:model_functions.py
                if direct_offset:
                    S_ = S_ + dt/tau_s * nonlin(-S_ + W_ss * J @ S_
                                                + a * ((J + g_s * P_gain) @ S0_delayed), ...)
                    S = S_ + d_s * P_offset
                else:
                    S = S + dt/tau_s * nonlin(-S + W_ss * J @ S
                                                + d_s * P_offset
                                                + a * ((J + g_s * P_gain) @ S0_delayed), ...)
```

With `g_s=d_s=0`: `d_s * P_offset` is zero; `(J + g_s * P_gain) @ S0_delayed` reduces to `J @ S0_delayed` (so `del_P` never enters S); `block_side` is not used in the dynamics loop at all (only in `create_stimuli` and output metadata). `direct_offset=False` for the fitted model.

`only_initial` is never passed by `simulate_recovery` (defaults `False`), and when `True` it would *remove* I/M prior rather than add S coupling.

**Indirect P involvement** (not S ODE bugs): concordant/discordant thresholds select `theta_c` vs `theta_d` for the M action only; I/M prior coupling changes choices and RT, hence selection into `*_f1`/`*_f2`.

**Backend parity** (NumPy vs Numba, `g_s=d_s=0`, seed 42): max |numpy − numba| = 0 (S), 1.4×10⁻¹⁶ (I), 3.1×10⁻¹⁷ (P), 4.4×10⁻¹⁶ (M).

---

## Phase 4b results: no prior modulation (pre-fix)

Run: `.../manifold_sim/absence/figs/phase4_no_prior_mod/` (seed 42, 40 sessions, `g_s=d_s=g_i=d_i=g_m=d_m=0`, nrand=100).

### Combined (4 splits summed)

| Pop | true `curve_mean` | null median | p_mean |
| --- | ----------------- | ----------- | ------ |
| **S** | **0.11** | 0.026 | 0.0 |
| I | 0.07 | 0.004 | 0.0 |
| M | 0.25 | 0.007 | 0.0 |

### Per-split (pooled over contrast)

| split | S p_mean | I p_mean | M p_mean |
| ----- | -------- | -------- | -------- |
| `*_f1` (correct) | **0.0** | **0.0** | **0.0** |
| `*_f2` (error) | 0.56–0.59 | 0.39–0.74 | 0.31–0.83 |

### Per-split × contrast

| split | pop | contrast | true mean | shuffle med | p(shuf≥true) |
| ----- | --- | -------- | --------- | ----------- | ------------ |
| r_choice_r_f1 | S | 1.0 | **0.51** | 0.006 | **0.0** |
| l_choice_l_f1 | S | 1.0 | **0.51** | 0.006 | **0.0** |
| r_choice_r_f1 | S | 0.125 | 0.011 | 0.010 | 0.37 |
| l_choice_l_f1 | S | 0.0 | 0.020 | 0.033 | 0.74 |
| *_f2 | S | all | — | — | **0.22–0.74** |

The same c=1.0 spike appears for I and M on `*_f1` (true ~0.34, p=0). Combined significance is dominated by **full-contrast trials**, not a flat across-contrast confound.

**Verdict (pre-fix):** zeroing all prior modulation collapses S combined **1.30 → 0.11**, so the bulk of the standard-absence signal is I/M prior mod → choices/selection, not hidden P→S. The residual is mostly c=1.0 on `*_f1`; error splits are n.s.

```bash
conda activate iblenv
python simulate_recovery.py --phase4-no-prior-mod --seed 42 --output-dir $ONE_CACHE_DIR/manifold_sim
```

### Follow-up: constant S0 = contrast

`apply_constant_s0_stimuli()` overwrites the stochastic draws after `create_stimuli`: signal channel = |nominal contrast|, other channel = 0, pre-stim steps zero. CLI `--phase4-constant-s0`.

| Pop | Stochastic S0 true | null med | Constant S0 true | null med |
| --- | ------------------ | -------- | ---------------- | -------- |
| **S** | 0.11 | 0.026 | **7.61** | 4.31 |
| **I** | 0.07 | 0.004 | **6.12** | 3.00 |
| **M** | 0.25 | 0.007 | **16.05** | 6.65 |

All p_mean = 0/100 in both conditions. Constant S0 **amplifies** the residual (~70× for S combined) rather than removing it → **S0 stochasticity is not the explanation**. With deterministic contrast the f2 splits are empty (no errors), so only correct-feedback splits contribute; only c=1.0 on `*_f1` has both prior groups populated (S true 2.90–2.93 vs shuffle median 0.001, p=0).

### Multiseed robustness

`scripts/run_phase4b_multiseed.py`, seeds 42, 7, 123, 999, 2024 (40 sessions each, nrand=100).

| Seed | S true | S null med | S/I/M p_mean |
| ---- | ------ | ---------- | ------------ |
| 42 | 0.11 | 0.026 | 0 / 0 / 0 |
| 7 | 0.13 | 0.024 | 0 / 0 / 0 |
| 123 | 0.12 | 0.025 | 0 / 0 / 0 |
| 999 | 0.13 | 0.024 | 0 / 0 / 0 |
| 2024 | 0.13 | 0.026 | 0 / 0 / 0 |

Per split: both `*_f1` splits p=0 in **5/5** seeds for S, I and M; `l_choice_r_f2` 0/5 significant; `r_choice_l_f2` 0/5 for S, 1/5 for I and M. At c=1.0 on both f1 splits, 5/5 seeds p=0 (S true medians 0.52 / 0.58 vs null 0.006). All other contrast bins: at most 1/5 significant.

```bash
python scripts/run_phase4b_multiseed.py --output-dir output/phase4_multiseed \
  --seeds 42 7 123 999 2024 --nrand 100
```

---

## Q1 hypotheses (before the truncation fix)

The 4 splits are defined by `(block_during_stim, stim_side, choice_side, feedback)`. With I/M mod on, P biases choices toward the prior-concordant side, which changes **which trials** land in each split.

| Hypothesis | Mechanism | Key test |
| ---------- | --------- | -------- |
| S0 filtering / collider selection | I/M selection creates an S0 group imbalance within a split | Direct S0 mean per group cell |
| RT / S-trajectory timing | Faster RT in the high-prior group → earlier S peak → inflated time-averaged distance | RT-aligned S trajectory comparison |
| `trial_in_block` × adaptation | Early-in-block trials have a different cumulative stimulation history | `a_at_stim` per (split, contrast, group) cell |
| Standard absence + constant S0 | Isolate the stochastic-S0 contribution | `--phase4-constant-s0` **with** fitted I/M |

The selection story: in `r_choice_r_f1`, the low-prior group (right block, concordant) gets I/M help, so even weak-S0 trials reach the correct-right bucket; the high-prior group (near block transitions) needs stronger S0 to pass. Different S0 distributions → different S trajectories with no P→S coupling at all.

The RT story turned out to be the operative one, but through the analysis window rather than through dynamics — see next section.

---

## The truncation bug (2026-06-18, high confidence)

> **TL;DR:** `bin_trace_segment` zero-pads trailing bins when a trial ends before the 150 ms window. At c=1.0 most trials have RT < 110 ms, so the mean S trajectory mixes real signal with structural zeros — and the two prior groups have different RT distributions, hence different zero-padding, hence spurious distance.

The analysis window for `act_block_duringstim` is `[0, 150 ms]` = 75 steps. `window_step_bounds` clips to `min(trial_len, sbo+75)`. Trials end at `sbo + RT + post_action_steps` with `post_action_steps = 20` steps (40 ms), so a trial is truncated whenever `RT < 55 steps = 110 ms`.

| RT (ms) | trial ends post-stim | zero bins / 75 | fraction zeros |
| ------- | -------------------- | -------------- | -------------- |
| 20 ms | 60 ms | 45 / 75 | 60 % |
| 40 ms | 80 ms | 35 / 75 | 47 % |
| 60 ms | 100 ms | 25 / 75 | 33 % |
| 80 ms | 120 ms | 15 / 75 | 20 % |
| 100 ms | 140 ms | 5 / 75 | 7 % |
| ≥110 ms | ≥150 ms | 0 | 0 % |

Quantified against the actual binning (`linspace(0, seg_len, n_bins+1).astype(int)` with 72 bins; `seg_len = RT_steps + 20`):

| RT (ms) | seg_len (steps) | empty zero-bins / 72 | % zeros |
| ------- | --------------- | -------------------- | ------- |
| 20 ms | 40 | **32/72** | 44 % |
| 40 ms | 60 | **12/72** | 17 % |
| 60 ms | 80 | 0/72 (no clip) | 0 % |
| ≥90 ms | ≥110 | 0 | 0 % |

**Evidence from the Phase 4b trajectory figure** (`p_block_s_trajectory_r_choice_r_f1.png`, seed 123): in the high-prior panel, c=1 S_r peaks at ~40 ms then sharply returns to zero with a brief oscillation — the signature of zero-padded trailing bins. In the low-prior panel it is sustained to ~70–80 ms. Real S driven by a constant c=1.0 stimulus does not abruptly return to zero mid-trajectory.

**Why RT differs between groups at c=1.0 under Phase 4b:** with all g/d=0 the only cross-trial carry-over is the slow decay of `I[0]−I[1]` (~204 ms time constant for the difference mode; 0.74 % residual after a 1000 ms ITI). Left-block (high-prior) trials carry a left-biased I offset which interacts nonlinearly with a strong right stimulus. The exact direction needs a dedicated diagnostic, but the directional asymmetry in zero-padding is visible in the figure and sufficient to explain the observed distance.

**Implication for Q1:** in standard absence the split's selection filter also generates an RT asymmetry — high-prior (anti-concordant correct) trials need strong S → faster M firing; low-prior (concordant correct) trials are aided by I/M → mixed RTs. Phase 2 confirms high-prior RT median 73 ms vs low-prior 80 ms, both below the 110 ms truncation threshold (~18 vs ~15 zero bins). With constant S0 = 1.0 the drive is stronger, RT even shorter, and the distance larger — consistent with the constant-S0 amplification above.

### Fix

- **fill-from-next-ITI** becomes the default in `build_population_b_for_split`, mirroring `prior_distance_I_M_by_choice_and_prior` in `model_functions.py`: borrow the leading `need` steps of the next trial's ITI; skip the trial if the next trial is in a different session or its ITI is too short.
- `--require-full-window` retained as a strict-exclusion diagnostic.
- `--duringstim-window-ms` added to override the post-stim window.

---

## Post-fix retests

### Phase 4b retest (seed 123, all g/d=0, nrand=100)

| Condition | S curve_mean | S p | I curve_mean | I p | M curve_mean | M p |
|-----------|-------------|-----|-------------|-----|-------------|-----|
| Old (zero-pad, 150 ms) | 0.117 | 0.00 | 0.054 | 0.00 | 0.203 | 0.00 |
| Fix (fill-next, 150 ms) | 0.041 | 0.04 | 0.004 | 0.60 ✓ | 0.004 | 0.81 ✓ |
| Fix (fill-next, 80 ms) | 0.012 | 0.78 ✓ | 0.001 | 0.25 ✓ | 0.001 | 0.08 ✓ |

**I and M** are completely non-significant with fill-from-next at 150 ms — the artefact was the sole cause of their apparent prior distance. **S at 150 ms** drops 0.117 → 0.041 (marginal p=0.04), still driven by c=1.0 f1; that residual is the fill-from-next tail (80–150 ms post-RT) where borrowed ITI data still differs between prior groups. **S at 80 ms** is fully null (p=0.78); c=1.0 f1 collapses to 0.008 vs null 0.007.

**2026-06-19b final (canonical S=80 ms / I/M=150 ms):**

| Population | curve_mean | null_median | p |
|------------|-----------|-------------|---|
| S | 0.0124 | 0.0172 | 0.78 ✓ |
| I | 0.0037 | 0.0046 | 0.60 ✓ |
| M | 0.0042 | 0.0077 | 0.81 ✓ |

Phase 4b is a pure null under the corrected analysis. This is now the [regression check](canonical_analysis_conventions.md#phase-4b-sanity-check).

### Standard absence retest (fitted I/M, seed 123)

At **150 ms, fill-next**:

| Condition | curve_mean | null_median | true−null |
|-----------|-----------|-------------|-----------|
| OLD absence (zero-pad, 150 ms) | 1.298 | 0.174 | 1.124 |
| NEW absence (fill-next, 150 ms) | 1.450 | 0.164 | 1.286 |

Both p=0.0. The overall `curve_mean` is slightly *larger* with fill-from-next — zero-padding had been suppressing the genuine low-contrast signal.

Per-contrast, `r_choice_r_f1` (S):

| contrast | OLD true | NEW true | OLD null | NEW null |
|----------|----------|----------|----------|----------|
| c=0.00 | 0.457 | **0.558** | 0.048 | 0.049 |
| c=0.0625 | 0.240 | **0.249** | 0.020 | 0.017 |
| c=0.125 | 0.267 | **0.269** | 0.012 | 0.013 |
| c=0.25 | 0.054 | **0.102** | 0.009 | 0.010 |
| c=1.00 | **0.096** | **0.028** | 0.006 | 0.005 |

The c=1.0 distance drops 3× — confirming the zero-padding inflation there — while all contrasts remain highly significant.

By split at 150 ms (mean across contrasts): r_choice_r_f1 0.241 / null 0.019; l_choice_l_f1 0.164 / 0.018; l_choice_r_f2 0.566 / 0.070; r_choice_l_f2 0.690 / 0.068. Error splits show the largest true distances.

At **canonical S=80 ms**: S overall `curve_mean=0.798`, null median 0.240, **p=0.0** (~3.3× above null).

| split | true | null |
|-------|------|------|
| r_choice_r_f1 | 0.102 | 0.008 |
| l_choice_l_f1 | 0.060 | 0.008 |
| l_choice_r_f2 | 0.426 | 0.048 |
| r_choice_l_f2 | 0.436 | 0.052 |

#### Per-contrast breakdown — all four splits (standard absence, S=80 ms)

**r_choice_r_f1** (right stim, right choice, correct)

| c | n_high | n_low | true | null | p |
|---|--------|-------|------|------|---|
| 0.0 | 51 | 323 | 0.0279 | 0.0074 | 0.13 |
| 0.0625 | 150 | 790 | 0.1505 | 0.0098 | 0.00 |
| 0.125 | 226 | 893 | 0.1977 | 0.0064 | 0.00 |
| 0.25 | 334 | 993 | 0.1270 | 0.0080 | 0.00 |
| 1.0 | 465 | 1005 | 0.0052 | 0.0075 | 0.66 |

**l_choice_l_f1** (left stim, left choice, correct)

| c | n_high | n_low | true | null | p |
|---|--------|-------|------|------|---|
| 0.0 | 324 | 50 | 0.0084 | 0.0099 | 0.54 |
| 0.0625 | 761 | 187 | 0.1032 | 0.0071 | 0.00 |
| 0.125 | 860 | 226 | 0.1105 | 0.0075 | 0.00 |
| 0.25 | 1028 | 328 | 0.0590 | 0.0086 | 0.00 |
| 1.0 | 947 | 412 | 0.0180 | 0.0083 | 0.20 |

**l_choice_r_f2** (left stim, right choice, incorrect)

| c | n_high | n_low | true | null | p |
|---|--------|-------|------|------|---|
| 0.0 | 81 | 155 | 0.1160 | 0.0073 | 0.00 |
| 0.0625 | 116 | 204 | 0.3212 | 0.0134 | 0.00 |
| 0.125 | 71 | 131 | 0.0928 | 0.0283 | 0.14 |
| 0.25 | 18 | 57 | 1.1719 | 0.1426 | 0.01 |

**r_choice_l_f2** (right stim, left choice, incorrect)

| c | n_high | n_low | true | null | p |
|---|--------|-------|------|------|---|
| 0.0 | 125 | 105 | 0.0717 | 0.0076 | 0.00 |
| 0.0625 | 191 | 124 | 0.1906 | 0.0113 | 0.00 |
| 0.125 | 127 | 70 | 0.4459 | 0.0291 | 0.00 |
| 0.25 | 47 | 18 | 1.0365 | 0.1599 | 0.00 |
| 1.0 | 1 | 0 | nan | nan | nan |

**Patterns:**

- f1 (correct) splits: signal peaks at c=0.125, near-null at c=1.0 (accuracy ≈ 100 % regardless of prior → no selection bias).
- f2 (incorrect) splits: signal grows with contrast to c=0.25, dominated by prior-driven wrong choices; almost no c=1.0 f2 trials.
- The large, consistent n_high/n_low imbalance reflects block-concordance asymmetry in trial counts, but count imbalance alone does not create distance — the distance comes from differential S trajectories.
- The pattern is symmetric across both f1 and both f2 splits → structural, not split-specific.
- c=0.0 (catch): near-null for f1 (no sensory drive → S≈0 for all trials); significant for f2 (wrong choices on catch trials are rare and highly prior-driven).

**Revised interpretation of Q1:** the genuine I/M→S prior effect exists and is large at mid-contrast (c=0.0625–0.25), mediated by the split's choice conditioning (f1/f2 selection bias interacting with I/M-driven choice probability). The c=1.0 and c=0.0 effects are near-null, consistent with saturation and with catch trials having no sensory drive.

---

## Trajectory-plot artefact (same bug, visualization path)

**Observation (2026-06-19):** in the **presence** condition (`g_s ≈ g_i`, `d_s ≈ d_i`), `p_block_s_trajectory_{split}.png` looked nearly identical to the absence plots and did not resemble the clear block separation in `p_block_i_trajectory_{split}`.

**Root cause:** `trial_s_binned_signed` (used by `_collect_s_traces_by_contrast` → `plot_p_block_s_trajectories`) used the full **150 ms** `PRE_POST[split]` window for *all* populations including S, so `bin_trace_segment` zero-padded there too. The distance path had been fixed; the visualization path had not.

With g_s/d_s active, concordant trials get a stronger feedforward S drive → M fires sooner → **shorter RT** → **more zero-padded bins**; anti-concordant trials get a suppressed drive → longer RT → fewer zero bins but a weaker signal. The two effects partially cancel in the 150 ms time-average:

| Group | S amplitude | RT | Zero-padded bins (of 75) | Mean S[1] over 150 ms |
|-------|-------------|----|--------------------------|-----------------------|
| P-block-R (concordant, presence) | ~0.4 (boosted) | ~25 ms | ~55 | ~0.4 × 20/75 ≈ 0.11 |
| P-block-L (anti-conc., presence) | ~0.2 (suppressed) | ~55 ms | ~35 | ~0.2 × 40/75 ≈ 0.11 |
| P-block-R (absence) | ~0.3 | ~40 ms | ~45 | ~0.3 × 30/75 ≈ 0.12 |
| P-block-L (absence) | ~0.3 | ~42 ms | ~43 | ~0.3 × 32/75 ≈ 0.13 |

(Schematic, `r_choice_r_f1`, high contrast.) I trajectories suffer less because I integrates over a longer window and also receives the boosted S as input.

**Secondary issue:** `plot_p_block_s_trajectories` computed its own `t_axis` from the full 150 ms split binning independently of population, so even after capping S at 80 ms the axis would be mislabeled.

**Code fix**

```python
if population == "S" and align_kind == "stimOn_times" and post > 0:
    post = min(post, S_DURINGSTIM_WINDOW_S)
    n_coarse = max(1, int(round(post / B_SIZE)))
    n_bins = n_coarse * max(1, int(B_SIZE // STS))
else:
    n_bins = split_n_bins(split)
```

```python
post_eff = min(PRE_POST[split][1], S_DURINGSTIM_WINDOW_S) if population == "S" else PRE_POST[split][1]
t_end_ms = post_eff * 1000.0
n_bins_pop = _population_n_bins(split, population)
t_axis = np.linspace(0, t_end_ms, n_bins_pop)
```

**Status:** both fixes landed; presence plots regenerated (`output/absence_80ms/seed_123/{absence,presence}/figs/block_confounds/p_block_s_trajectory_stim_*.png`; note `_split_short_label` now prefixes the split name with `stim_`).

**Caveat carried forward:** the 80 ms cap was necessary but **not sufficient** to make g_s/d_s block modulation visible. Even at S-significant parameters, P-block-L vs P-block-R S trajectories still largely overlap — that open issue lives in [direct sensory prior coupling](direct_sensory_prior_coupling.md).

---

## Code / outputs added during this investigation

`simulate_recovery.py`:

- `write_s_prior_shuffle_diagnostics()` → per-shuffle `curve_mean` / `curve_amp` CSV
- `write_split_contrast_shuffle_diagnostics()` → per-split, per-contrast true vs shuffle means
- Both run automatically in `process_condition()` (S-prior-only mode)
- `--phase2-adaptation`, `--random-prior-labels`, `--phase4-no-prior-mod`, `--phase4-constant-s0`, `--require-full-window`, `--duringstim-window-ms`
- `a_at_stim` / ITI ‖S‖ in the trial table + confound plots

```
output/absence_shuffle_debug/absence/
  figs/s_prior_shuffle_nulls.csv          # 100 shuffles, combined curve stats
  figs/s_prior_shuffle_summary.csv        # true vs shuffle summary
  figs/s_prior_split_contrast_shuffle.csv # per-split × contrast table
  figs/s_shuffle_control.png
  summary.json
  res/                                    # split-level regde outputs
```

```bash
export ONE_CACHE_DIR=/path/to/ONE/cache
export MPLBACKEND=Agg
python -c "
import simulate_recovery as sr
from pathlib import Path
sr.process_condition(
    'absence', g_s=0.0, d_s=0.0,
    n_sessions=40, nrand=100, blocks_per_session=6,
    base_dir=Path('output/absence_shuffle_debug'),
    rng_seed=42, weights_json=sr.resolve_weights_json(),
    s_prior_only=True, n_jobs=8, contrast_matched_null=True,
)
"
```
