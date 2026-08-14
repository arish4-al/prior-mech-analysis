# BWM functional classification recovery on simulated experiments

**Scope:** the `--full-analysis` path — recovering S/I/M population identity from simulated data with the BWM Σ classifier — plus the decorrelation-window and plotting bugs that had to be fixed before it produced deliverables.

**Status:** complete for all four canonical experiments. The classifier recovers **3/3** populations in the absence condition (fitted I/M prior modulation) and **1/3** in the three experiments without I/M prior modulation, which act as negative controls.

Sources: dated entries 2026-07-06e, 07-06f, 07-06g, 07-06i, 07-07, 07-07b.

---

## Goal

Analysis 1.3 of the Goal-1 matrix: run BWM functional classification (`classify_regions`) on each of the four experiments (phase4, absence, s1800, s2025 — see [simulation infrastructure](simulation_infrastructure.md)) and ask whether the simulated S/I/M populations are recovered as stimulus / integrator / movement regions.

---

## Bugs that blocked the deliverables

### 1. `compute_amp_slope` on short (80 ms) S splits

`analysis_functions.compute_amp_slope` did `np.polyfit(np.arange(n), r[-n:], 1)` with fixed `n` (20, and 5/10), but the short 80 ms `duringstim` region curves (`act_block_duringstim`, `stim_duringstim_short_act`) have fewer than 20 bins → "expected x and y to have same length" → `amp_slope` never set → `manifold_to_csv` raised `KeyError: 'amp_slope'`. This failed at real params, not just tiny tests.

**Fix:** clamp each fit window to `min(k, len(r))` (helper `_last_slope`); return NaN with <2 points. Backward-compatible (identical when `len(r) >= k`).

### 2. Pooled prior-distance plot vs mixed population windows

After the `amp_slope` fix, classification passed but `*_cm_full` crashed in `plot_recovery_figures`: `ax.plot(t_stim, mean_c)` → `x and y must have same first dimension, (36,) vs (72,)`. Under the canonical analysis each split's combined regde holds curves of **two lengths** — S = 80 ms (36 bins), I/M/P = 150 ms (72 bins) — but the plot built one `t_stim` from an arbitrary region and reused it. Confirmed empirically: per-split regde length distribution `{36: 1, 72: 3}`.

**Fix:** build the time axis **per group** (S uses the 0–80 ms axis, I/M the 0–150 ms axis) and pool only same-length curves.

No CSVs had been saved (classification outputs are written *after* this plot), so the crash blocked the actual deliverables. The four `*_cm_full` matrix runs used pre-fix code and failed; `run_goal1_full_refix.sh` re-ran only those four (full path wipes `res_dir` and recomputes, ~8 min each; the session simulation is still a cache hit).

**2026-07-06g:** with both fixes, all four `*_cm_full` runs completed and wrote full deliverables — `figs/bwm_classification.csv`, `classification_details.csv`, `population_prior_tests.csv`, `prior_modulation.csv`, `s_prior_stats.csv`, `summary.json`.

### 3. Decorrelation `PRE_POST` window (2026-07-06i)

**Bug:** `stim_choice_*_act` / `choice_duringstim_*_act` matched `block` in the name → `PRE_POST=[0.4,−0.1]` (ITI window) while `split_n_bins` used 0.5 s → 240 bins on a ~150-step segment → alternating empty/filled bins → oscillating `d_euc`.

**Fix:** in `build_align_pre_post()`, decorrelation splits (`stim_choice_*`, `choice_duringstim_*`, `stim_block_*_act`) get `[0, 0.15]` (72 bins, during-stim window).

Rebuilt `goal1/absence/cm_full` (session cache HIT, ~5.4 min); curves smooth (len=72, no oscillation). Classifier then gave **3/3**: S→stimulus (Σ=0.99), I→integrator (Σ=0.42), M→movement (monotonicity=1).

### 4. Short-window + combine fixes (2026-07-07)

- **`_short` splits:** genuine 80 ms `PRE_POST` (not just a plot axis) → `stim_duringstim_short_act` now differs from `stim_duringstim_act` (36 vs 72 bins for I/M).
- **`stim_block_*` (d^{stim,se′}):** 80 ms early window (was 150 ms); used only in `stim_duringstim1_act`.
- **`stack_combined_timeframes`:** average across L/R splits (`/ n_stacked`), matching `analysis_functions.plot_regional_distance` — fixes a spurious 0.5× amplitude for 2-split vs 4-split timeframes.
- **80 ms S cap in `build_population_b_for_split`** limited to prior-distance splits (`act_block_duringstim_*`) only, **not** decorrelation splits — fixes the d^{stim,s} vs d^{stim,se} identity for S (2026-07-07b).
- **`plot_bwm_decorrelation_curves`** wired into `plot_recovery_figures`; default populations S/I/M; independent axes; saves `sim_duringstim_stim_choice_d_euc_SIM.png`.

Rebuilt `goal1/absence/cm_full` (~4.7 min, cache HIT). Panels 1 ≠ 3; panel 4 now 80 ms; se′ still correlates with choice (same trial pools per stim side) but no longer shows the half-amplitude artefact.

---

## Classifier sanity check

`classify_regions` rules match hand-computed Σ thresholds on absence: `σ_stim,s > 0.8 → S`; `σ_stim,m ≤ 0.8 ∧ monotonicity → M`; else integrator. It uses raw `amp_euc` without significance masking for assignment; the prior population P is excluded. Implementation confirmed correct.

---

## Results

### Absence, post-fix rebuild (seed 123)

Accuracy **3/3** — S→stimulus ✓, I→integrator ✓, M→movement ✓.

**BWM metrics** (`figs/bwm_classification.csv`):

| pop | true | pred | Σ^stim,s | Σ^stim,m | monotonicity | sc_duringstim | sc_duringchoice |
|-----|------|------|----------|----------|--------------|---------------|-----------------|
| S | S | S | **0.993** | 0.551 | 0 | 0.007 | 0.449 |
| I | I | I | 0.351 | 0.143 | 0 | 0.649 | 0.857 |
| M | M | M | 0.331 | 0.154 | **1** | 0.669 | 0.845 |

Compared to the 07-06i state (PRE_POST fix only, before the short/combine fixes): S Σ unchanged (0.99); I/M Σ dropped 0.42 → 0.35/0.33 because denominators now include a distinct d^{stim,se} (80 ms) and averaged (not summed) multi-split amplitudes. S/I/M assignments unchanged.

**Prior modulation** (act_block_duringstim, all significant at p≈0.01): S amp 0.950, I 0.106, M 0.379.

### All four experiments (2026-07-07b, after all decorrelation fixes)

Rebuilt via `run_goal1_cm_full_rebuild.sh` (phase4, s1800, s2025; absence already done). All **session cache HIT**; ~5 min each for the `res/` recompute, no re-simulation.

| exp | acc | S true→pred | I | M | why failures |
|-----|-----|-------------|---|---|--------------|
| phase4 | **1/3** | S→**I** (Σ^stim,s=0.59) | I→I ✓ | M→**I** (mono=0) | No prior mod: stim/choice decorrelation amps tiny and similar; M lacks a pre-movement ramp |
| absence | **3/3** | S→S ✓ (Σ=0.99) | I→I ✓ | M→M ✓ (mono=1) | Fitted I/M prior mod gives separable signatures |
| s1800 | **1/3** | S→**I** (Σ=0.64) | I→I ✓ | M→**I** (mono=0) | P→S only, I/M prior mod off: S below the σ threshold |
| s2025 | **1/3** | S→**I** (Σ=0.66) | I→I ✓ | M→**I** (mono=0) | same as s1800 |

**Interpretation:** the classifier works as designed. Perfect recovery requires absence-like fitted I/M prior modulation to sculpt distinct stim/choice/movement decorrelation. Phase4 and the s_presence experiments (zero I/M prior mod) are negative controls for recovery — only the I integrator label is stable across them.

**Plots** (all under `goal1/<exp>/cm_full/figs/`): `sim_duringstim_stim_choice_d_euc_SIM.png`, `bwm_classification.csv`, `classification_confusion.png`, `prior_distance_by_group.png`.

---

### 2026-08-13b — Stage B fitted θ (not WEIGHTS_REL)

Full write-up: [retinal_then_joint_fitting.md](retinal_then_joint_fitting.md) 2026-08-13b.
`--full-analysis` on Stage B shared-stim winners (seed 123, nrand 100, canonical windows).

| θ | acc | S | I | M |
|---|-----|---|---|---|
| regular s101 (fitted I/M, `g_s=d_s=0`, `g_m≈0.20`, Stage-A retinal) | **3/3** | S→S (Σ=0.993) | I→I | M→M (mono=1) |
| sensory s23 (P→S only, `g_s≈38`, `d_s≈33`, Stage-A retinal) | **2/3** | S→S (Σ=0.801) | I→I | M→**I** (mono=0) |

Regular matches canonical absence. Sensory is **not** the 1/3 s1800/s2025 pattern: S just clears σ=0.8. M still fails without I/M prior mod. S/I/M prior distance significant in both (sensory S amp 0.065 vs regular 1.076).
