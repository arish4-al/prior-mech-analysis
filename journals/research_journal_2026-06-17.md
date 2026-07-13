# Research journal — 2026-06-17

## Circuit generative recovery: absence S-prior shuffle diagnostics

### Goal

**Investigate, debug, and identify why there is a significant S prior-distance effect vs contrast-matched shuffle controls in the absence case (`g_s=0`, `d_s=0`), when the generative equations have no direct P→S coupling.**

We are *not* asking whether absence shows an effect (it does — see results below). We want to know **what mechanism produces it**: analysis artifact, unintended code path, or something else not yet identified.

**Success criteria**

1. Name dominant cause(s) with quantitative evidence.
2. Show whether the effect survives controls that should remove each candidate.
3. Decide whether absence shuffle significance can be used as any kind of null, or must be discarded for recovery inference.

**What we already know**

**Generative model / code (Phase 4a)**

- With `g_s=d_s=0`, the S ODE has no P terms; P still modulates I/M (`g_i`, `d_i`, …) in **standard absence**.
- `load_fitted_model` forces `g_s`/`d_s` after JSON merge; NumPy/Numba parity when zero. **No unintended P→S coupling bug.**

**Standard absence (`g_s=d_s=0`, fitted I/M prior)**

- True S prior distance ≫ contrast-matched shuffles (combined and per-contrast; all `p=0` in initial runs).
- Effect largest on error-trial splits (`*_f2`) and mid contrasts in those runs.
- Random ITI labels on **same trajectories** collapse `curve_mean` to **0.034** (below shuffle null median **0.17**) → label **assignment** matters; not a change in simulated dynamics.
- Bulk of standard-absence signal vs Phase 4b: zeroing all `g_*`/`d_*` drops S combined **1.30 → 0.11** → most of the large effect is **I/M prior mod changing choices/selection**, not hidden P→S.

**Phase 4b (all `g_*=d_*=0`)**

- S/I/M prior distance still ≫ contrast-matched shuffle (combined S ~0.11–0.13, null ~0.024–0.026; **p=0/100** across seeds 42, 7, 123, 999, 2024).
- Residual significance **mostly c=1.0 on `*_f1`**; `*_f2` n.s. under no-modulation.
- Constant S0 (= contrast, no noise) **amplifies** residual (S combined ~7.6) — **S0 stochasticity is not the explanation.**
- **Causal mechanism for Phase 4b residual: open** (not generative P coupling, not S0 noise).

**Excluded from consideration — trial structure / history as mechanism**

Do **not** invoke block epoch, trial index, stimulus-sequence position, or cross-trial carryover to explain the Phase 4b residual (or S trajectory differences under splits that already fix stim and choice sides):

1. **S is almost completely feedforward.** Fitted `W_ss ≈ 7.6×10⁻⁵`, `tau_s = 20` ms, long ITI → negligible cross-trial S memory. Within trial, S is driven by current `S0`.
2. `**block_side` does not enter dynamics** when all prior modulations are off — no `g_`*/`d_`*, no block term in the S/I/M ODEs.
3. **Splits fix stim and choice sides** (`stim_l`/`stim_r`, `choice_l`/`choice_r`, feedback). At a given contrast bin, groups are compared on the same stimulated side and outcome class; `block_side` / `trial_in_block` imbalance across label groups is **not a lawful driver of different S trajectories** under Phase 4b.
4. **Adaptation `a` at stim ≈ 1** for both prior groups (Phase 2 covariate check).

**Inference**

- **Contrast-matched absence shuffle is not a valid causal null** for recovery (true ≫ shuffle even when all generative prior coupling is zero), regardless of the open mechanism question.

### Model setup (absence)


| Parameter    | Value                                                            |
| ------------ | ---------------------------------------------------------------- |
| Condition    | absence                                                          |
| `g_s`, `d_s` | 0.0, 0.0 (no sensory prior pathway)                              |
| `g_i`, `d_i` | 189.68, 21.56 (fitted; I still modulated by P)                   |
| Sessions     | 40 × 6 blocks                                                    |
| Trials       | 12,036 total                                                     |
| RNG seed     | 42                                                               |
| Prior column | `p_subjective_probabilityLeft` (ITI subjective P ≥ 0.5 vs < 0.5) |
| Null scheme  | Contrast-matched label shuffle, `nrand=100`                      |
| Splits       | 4 × `act_block_duringstim`                                       |
| Output       | `output/absence_shuffle_debug/`                                  |


### Causal structure (what can affect S in absence)

With `g_s=d_s=0`, the S equation has **no P and no block_side**. S at stim depends on external stimulus `S0`, negligible `W_ss`, and adaptation `a`.

**Contrast-matched label shuffle** preserves per-contrast high/low counts; used as the analysis null throughout.

## Investigation plan

Ordered from fastest / most diagnostic to heavier model changes.

### Phase 0 — Baseline characterization (mostly done)


| Task                                 | Tool                                 | What it tests                                       |
| ------------------------------------ | ------------------------------------ | --------------------------------------------------- |
| Combined + per-shuffle `curve_mean`  | `s_prior_shuffle_*.csv`              | Confirm effect size, not p-value direction          |
| Per-split × contrast true vs shuffle | `s_prior_split_contrast_shuffle.csv` | Rule out contrast imbalance; localize to splits     |
| Block-confound distributions         | `--block-confound-plots`             | RT, contrast, S peak time: P-block-L vs R per split |
| Subjective vs block prior grouping   | `--prior-compare`                    | Is effect specific to ITI P vs `probabilityLeft`?   |


**Read of existing per-contrast data:** true > shuffle at every contrast with adequate n; `*_f2` > `*_f1`; strong high/low count asymmetry per split.

---

### Phase 1 — Label / covariate checks (no model changes)

**1a. Block prior vs subjective P** — `--prior-compare`

**1b. Covariate balance** — RT, contrast marginals (Phase 2 CSVs)

**1c. Error-rate asymmetry** — `*_f1` vs `*_f2` trial counts by prior group

---

### Phase 2 — Covariate / adaptation diagnostics — **DONE (2026-06-17)**

See [Phase 2 results](#phase-2-results-covariate--adaptation-diagnostics) below.

**2a. Record `a` at stim onset** — implemented in `extract_trial_table`; histograms in `phase2_a_at_stim_by_prior_split.png`.

**2b. S trajectories by contrast with prior panels** — `phase2_s_traj_*_{block_left,block_right}.png` (fixed `block_side`).

**2c. Trial-history matched control** — `phase2_matched_history_*.csv` (match on block_side, contrast, trial decile, session).

Run: `python simulate_recovery.py --phase2-adaptation --seed 42 --output-dir output/absence_phase2`

---

### Phase 3 — Analysis / null-scheme checks

**3a. Unrestricted vs contrast-matched shuffle**

Run `--null-compare` on absence. Compare unrestricted vs contrast-matched shuffle null distances.

**3b. Split-sum artifact**

Per-split shuffle significance vs combined sum (`stack_combined_timeframes`). Check whether summing splits amplifies a small per-split effect.

---

### Phase 4 — Simulation controls (ground truth)

These establish whether the generative code can produce a flat S-prior null when labels are uninformative.


| Control                    | Implementation                                   | Notes                                                |
| -------------------------- | ------------------------------------------------ | ---------------------------------------------------- |
| **Random ITI labels**      | Assign 0.8/0.2 at random, independent of P trace | Collapses standard-absence effect below shuffle null |
| **Absence + no I/M prior** | `g_i=d_i=g_m=d_m=0` (with `g_s=d_s=0`)           | `--phase4-no-prior-mod` — residual still ≫ shuffle   |


**4a. Code audit** — **DONE (2026-06-17)**

See [Phase 4a results](#phase-4a-results-code-audit) below.

**4b. Absence + no P modulation (`--phase4-no-prior-mod`)** — **DONE (2026-06-17)**

See [Phase 4b results](#phase-4b-results-no-prior-modulation) below.

---

### Phase 5 — Decision for recovery pipeline


| Outcome                                   | Action                                                                                                                |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Absence shuffle unsuitable as causal null | Use **presence − absence** + absence replicate null for injection only; do not interpret shuffle p as generative test |
| Minor analysis fix (e.g. split sum)       | Patch `simulate_recovery.py`, re-run Slurm                                                                            |
| Residual effect only on `*_f2`            | Consider restricting S-prior splits to `*_f1` for cleaner sensory window                                              |
| Effect persists after all controls        | Trace analysis pipeline / dynamics step-by-step; mechanism still open                                                 |


**Suggested immediate next commands**

```bash
# 1. Block confounds (RT, contrast, S peak)
python simulate_recovery.py --block-confound-plots --seed 42 \
  --output-dir output/absence_confound

# 2. Subjective vs block prior
python simulate_recovery.py --prior-compare --seed 42 \
  --output-dir output/absence_prior_compare

# 3. Null scheme comparison
python simulate_recovery.py --null-compare --seed 42 \
  --output-dir output/absence_null_compare
```

**Implementation backlog** (add to `simulate_recovery.py` as needed)

- `a_at_stim` / ITI ‖S‖ in trial table + confound plots (`--phase2-adaptation`)
- Covariate balance table per split (`phase2_covariate_mannwhitney.csv`)
- Matched-block S trajectories + trial-history matched distances
- `--random-prior-labels` simulation control
- `--freeze-adaptation-at-stim` model flag (in `model_functions.py`)
- `--phase4-no-prior-mod` (all `g_*=d_*=0`, S/I/M prior distance vs shuffle)

---

## Phase 2 results: covariate / adaptation diagnostics

**Run:** `output/absence_phase2/absence/figs/phase2_adaptation/` (absence, seed 42, 40 sessions, ITI subjective P).

### Summary


| Candidate                  | Evidence                                  | Conclusion                          |
| -------------------------- | ----------------------------------------- | ----------------------------------- |
| **Adaptation `a` at stim** | Medians ~0.9913 both groups; diff ~7×10⁻⁵ | **Not a driver** (tiny effect size) |
| **ITI ‖S‖**                | Medians ~10⁻¹⁶                            | **Irrelevant**                      |


Covariate descriptives (block_side, `trial_in_block`, RT) are in `phase2_covariate_mannwhitney.csv`. **Not used as explanatory mechanism** — see [What we already know](#what-we-already-know).

### 2a. Covariate table (high vs low ITI subjective P)


| split         | metric         | med(high) | med(low) | p      |
| ------------- | -------------- | --------- | -------- | ------ |
| r_choice_r_f1 | a_at_stim_mean | 0.99142   | 0.99135  | 0.021  |
| r_choice_r_f1 | trial_in_block | 20        | 28       | ~10⁻²⁴ |
| r_choice_r_f1 | reaction_time  | 73        | 80       | ~10⁻¹¹ |
| l_choice_l_f1 | trial_in_block | 25        | 21       | ~10⁻⁹  |
| l_choice_l_f1 | reaction_time  | 80        | 72       | ~10⁻¹³ |
| l_choice_r_f2 | (all metrics)  | —         | —        | n.s.   |
| r_choice_l_f2 | trial_in_block | 20        | 27       | ~10⁻⁶  |


Full table: `phase2_covariate_mannwhitney.csv`

### 2b. Matched-block S trajectories

`phase2_s_traj_{split}_{block_left|block_right}.png` — S_l/S_r by contrast, high vs low prior within fixed `block_side`.

### 2c. Matched-bin distances

Match bins: `(block_side, contrast, trial_in_block decile, session)`.


| split         | full distance | matched-bin distance | fraction | n bins         |
| ------------- | ------------- | -------------------- | -------- | -------------- |
| r_choice_r_f1 | 0.203         | 0.043                | **0.21** | 100            |
| l_choice_l_f1 | 0.234         | 0.077                | **0.33** | 84             |
| l_choice_r_f2 | 0.537         | 1.067                | —        | 1 (unreliable) |
| r_choice_l_f2 | 0.324         | —                    | —        | 0              |


### Figures & CSVs

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

## Phase 4a results: code audit

**Question:** Is there an unintended code path that couples P (or block) into S when `g_s=d_s=0`?

**Verdict: No code bug.** P terms are present in the S update but **multiplied by zero** when `g_s=d_s=0`.

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

- `g_s`/`d_s` are assigned **after** JSON `model_params` and `W` merge — fitted JSON values cannot leak through.
- `_update_model_params_for_dt` only updates `tau_`*, `post_action_steps`, `prestim_offset_start` — does **not** touch `g_s`/`d_s`.
- All absence entry points call `load_fitted_model(g_s=0.0, d_s=0.0)` or pass explicit zeros via `process_condition`.
- `simulate_session` → `run_model("data", ..., **model_params)` — `mp` dict carries the overridden values into `set_model_parameters`.

### S ODE — all backends (NumPy, Numba, Torch)

`simulate_session` uses `backend="auto"` (Numba when available). For `model_type='data'`, `set_model_parameters` reads `g_s`, `d_s` from `model_params`.

S update (NumPy reference; Numba kernel `_run_model_kernel` and Torch `_run_model_torch` are structurally identical):

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

With `g_s=d_s=0`:


| Term                              | Effect                                                                                           |
| --------------------------------- | ------------------------------------------------------------------------------------------------ |
| `d_s * P_offset`                  | Zero                                                                                             |
| `(J + g_s * P_gain) @ S0_delayed` | Reduces to `J @ S0_delayed` — `del_P` (from P–S concordance) does not enter S                    |
| `block_side`                      | **Not used** in dynamics loop — only in `create_stimuli` (stimulus schedule) and output metadata |


`direct_offset=False` for fitted model (default in `model_functions.model_params`).

### `only_initial` flag

- `simulate_recovery` never passes `only_initial` → defaults `False`.
- When `only_initial=True`, gains are zeroed **after** `steps_before_obs` (would remove I/M prior too, not add S coupling). Numba backend raises `_NumbaUnsupported` and auto-falls back to NumPy.
- **Not active** in recovery pipeline.

### Indirect P involvement (not S ODE bugs)

These can change **which trials** land in splits or **when** choices occur, but do not add P→S terms:

1. **Concordant/discordant thresholds** — `del_P` from P trace selects `theta_c` vs `theta_d` for M action only.
2. **I/M prior coupling** (`g_i`, `d_i` nonzero) — affects choices and RT → selection into `*_f1`/`*_f2` splits.

### Backend parity check

Ran NumPy vs Numba with `g_s=d_s=0`, `model_type='data'`, `only_initial=False` (stub import, seed 42):

| Field | max |numpy − numba| |
| ----- | ------------------- |
| S     | 0                   |
| I     | 1.4×10⁻¹⁶           |
| P     | 3.1×10⁻¹⁷           |
| M     | 4.4×10⁻¹⁶           |

### Implication for Phase 5

Absence shuffle significance remains **unsuitable as a causal null** — not because of a code defect. See [What we already know](#what-we-already-know).

---

## Phase 4b results: no prior modulation

**Run:** `output/.../manifold_sim/absence/figs/phase4_no_prior_mod/` (seed 42, 40 sessions, `g_s=d_s=g_i=d_i=g_m=d_m=0`, nrand=100).

**Question:** With **no generative P→S/I/M coupling**, do S/I/M still show ITI-P prior distance ≫ contrast-matched shuffle?

### Combined (4 splits summed)

Generative model: `**g_s=d_s=g_i=d_i=g_m=d_m=0`**. `curve_mean` = analysis prior-distance (ITI-P high vs low groups), not a model parameter.


| Pop   | true `curve_mean` | null median | p_mean |
| ----- | ----------------- | ----------- | ------ |
| **S** | **0.11**          | 0.026       | 0.0    |
| I     | 0.07              | 0.004       | 0.0    |
| M     | 0.25              | 0.007       | 0.0    |


### Per-split (pooled over contrast)


| split            | S p_mean  | I p_mean  | M p_mean  |
| ---------------- | --------- | --------- | --------- |
| `*_f1` (correct) | **0.0**   | **0.0**   | **0.0**   |
| `*_f2` (error)   | 0.56–0.59 | 0.39–0.74 | 0.31–0.83 |


### Per-split × contrast (`phase4_no_prior_mod_split_contrast.csv`)

Contrast-matched shuffle **within each contrast bin**. Residual significance is **not uniform across contrast**:


| split         | pop | contrast | true mean | shuffle med | p(shuf≥true)  |
| ------------- | --- | -------- | --------- | ----------- | ------------- |
| r_choice_r_f1 | S   | 1.0      | **0.51**  | 0.006       | **0.0**       |
| l_choice_l_f1 | S   | 1.0      | **0.51**  | 0.006       | **0.0**       |
| r_choice_r_f1 | S   | 0.125    | 0.011     | 0.010       | 0.37          |
| l_choice_l_f1 | S   | 0.0      | 0.020     | 0.033       | 0.74          |
| *_f2          | S   | all      | —         | —           | **0.22–0.74** |


Same **c=1.0 spike** for I and M on `*_f1` (true ~0.34, p=0). Mid/low contrasts mostly n.s. → combined significance is dominated by **full-contrast trials**, not a flat across-contrast confound.

### S trajectories by ITI P × contrast

`p_block_s_trajectory_{split}.png` — left panel ITI P≥0.5, right panel P<0.5; colored curves per contrast (solid S_l, dashed S_r). Visual check of whether gaps are contrast-specific.

### Outputs

```
absence/figs/phase4_no_prior_mod/
  phase4_no_prior_mod_split_contrast.csv
  p_block_s_trajectory_{r_choice_r_f1,l_choice_l_f1,l_choice_r_f2,r_choice_l_f2}.png
  phase4_no_prior_mod_summary.csv
  phase4_no_prior_mod_by_split.csv
  phase4_no_prior_mod_shuffle_controls.png
  ...
```

### Verdict

1. Zeroing all prior modulations collapses S combined **1.30 → 0.11** — bulk of standard-absence signal is **I/M prior mod → choices/selection**, not hidden P→S.
2. Residual significance is **mostly c=1.0 on `*_f1`**; mid/low contrasts mostly n.s.
3. **Error splits n.s.** under no-modulation (p ≈ 0.3–0.8).
4. Random ITI labels give **0.034** vs residual **0.11** — label assignment matters; Phase 4b mechanism **open** (see [What we already know](#what-we-already-know)).

**Reproduce:**

```bash
conda activate iblenv
python simulate_recovery.py --phase4-no-prior-mod --seed 42 \
  --output-dir $ONE_CACHE_DIR/manifold_sim
```

---

## Phase 4b follow-up: constant S0 = contrast

**Run:** `manifold_sim/absence/figs/phase4_no_prior_mod_constant_s0/` (seed 42, 40 sessions, `g_s=d_s=g_i=d_i=g_m=d_m=0`, **deterministic S0**, nrand=100).

**Question:** If S0 is replaced with **noise-free constant contrast** on the signal side (other channel 0, from stim onset), does residual S/I/M prior distance vs contrast-matched shuffle disappear?

**Implementation:** `apply_constant_s0_stimuli()` in `simulate_recovery.py` — after `create_stimuli`, overwrites stochastic draws: signal channel = |nominal contrast|, other channel = 0; pre-stim steps zero. CLI: `--phase4-constant-s0` (runs Phase 4b with this flag).

### Combined (4 splits summed)


| Pop   | Stochastic S0 (Phase 4b) |          | Constant S0 |          |
| ----- | ------------------------ | -------- | ----------- | -------- |
|       | true                     | null med | true        | null med |
| **S** | 0.11                     | 0.026    | **7.61**    | 4.31     |
| **I** | 0.07                     | 0.004    | **6.12**    | 3.00     |
| **M** | 0.25                     | 0.007    | **16.05**   | 6.65     |


All populations **p_mean = 0/100** in both conditions. Constant S0 **amplifies** the residual (~70× for S combined), not removes it.

### Per-split


| split           | S p (const) | I p     | M p     | notes                                                |
| --------------- | ----------- | ------- | ------- | ---------------------------------------------------- |
| `r_choice_r_f1` | **0.0**     | **0.0** | **0.0** | correct feedback                                     |
| `l_choice_l_f1` | **0.0**     | **0.0** | **0.0** | correct feedback                                     |
| `*_f2`          | —           | —       | —       | **0 trials** (no errors with deterministic contrast) |


Stochastic Phase 4b: f2 splits had ~~100–130 trials/session-pool and were n.s. (p ≈ 0.3–0.8). With constant S0, error trials vanish (~~0 f2 trials in 5-session check vs ~100+ stochastic) — only correct-feedback splits contribute.

### Per-split × contrast (constant S0)

Only **c=1.0** on `*_f1` has both prior groups populated; all S/I/M **p=0/100**:


| split         | pop | c=1.0 true | shuffle med | p   |
| ------------- | --- | ---------- | ----------- | --- |
| r_choice_r_f1 | S   | 2.93       | 0.001       | 0   |
| l_choice_l_f1 | S   | 2.90       | 0.001       | 0   |
| r_choice_r_f1 | I   | 1.73       | 0.0006      | 0   |
| l_choice_l_f1 | I   | 1.71       | 0.0005      | 0   |
| r_choice_r_f1 | M   | 4.54       | 0.0016      | 0   |
| l_choice_l_f1 | M   | 4.49       | 0.002       | 0   |


Lower contrasts mostly empty (trials in one prior group only). Same **c=1.0 spike** pattern as stochastic Phase 4b, but much larger absolute distances.

### Verdict

1. **S0 stochasticity is not the explanation** — constant S0 amplifies prior-distance.
2. Significance persists on **c=1.0, `*_f1`** for S/I/M with all `g_*=d_*=0`.
3. Deterministic contrast → f2 splits empty; effect on `*_f1` only.

**Reproduce:**

```bash
conda activate iblenv
python simulate_recovery.py --phase4-constant-s0 --seed 42 --nrand 100 \
  --output-dir $ONE_CACHE_DIR/manifold_sim
```

**Outputs:**

```
absence/figs/phase4_no_prior_mod_constant_s0/
  phase4_no_prior_mod_constant_s0_split_contrast.csv
  phase4_no_prior_mod_constant_s0_summary.csv
  phase4_no_prior_mod_constant_s0_by_split.csv
  p_block_s_trajectory_{split}.png
  ...
```

---

## Phase 4b multiseed robustness

**Run:** `output/phase4_multiseed/` — seeds **42, 7, 123, 999, 2024** (40 sessions each, `g_s=d_s=g_i=d_i=g_m=d_m=0`, nrand=100).

**Question:** Is the Phase 4b residual (f1 significant, f2 n.s., c=1.0 spike) a seed-42 artifact?

**Script:** `scripts/run_phase4b_multiseed.py` — runs Phase 4b per seed, writes aggregate CSVs to `output/phase4_multiseed/aggregate/`.

### Combined (4 splits summed)


| Seed | S true | S null med | S/I/M p_mean |
| ---- | ------ | ---------- | ------------ |
| 42   | 0.11   | 0.026      | 0 / 0 / 0    |
| 7    | 0.13   | 0.024      | 0 / 0 / 0    |
| 123  | 0.12   | 0.025      | 0 / 0 / 0    |
| 999  | 0.13   | 0.024      | 0 / 0 / 0    |
| 2024 | 0.13   | 0.026      | 0 / 0 / 0    |


All seeds: **S/I/M p=0/100** combined. True distance stable (~0.11–0.13); null median ~0.024–0.026.

### Per-split (p_mean across seeds)


| Split           | S           | I                | M                |
| --------------- | ----------- | ---------------- | ---------------- |
| `r_choice_r_f1` | **5/5** p=0 | **5/5** p=0      | **5/5** p=0      |
| `l_choice_l_f1` | **5/5** p=0 | **5/5** p=0      | **5/5** p=0      |
| `l_choice_r_f2` | 0/5 sig     | 0/5 sig          | 0/5 sig          |
| `r_choice_l_f2` | 0/5 sig     | 1/5 sig (p=0.04) | 1/5 sig (p=0.02) |


Correct-feedback f1 splits significant in **every seed** for all populations. Error f2 splits remain mostly n.s.

### Per-split × contrast

**c=1.0 on both f1 splits — 5/5 seeds p=0 for S, I, M:**


| Split         | pop | true (median) | null med (median) |
| ------------- | --- | ------------- | ----------------- |
| r_choice_r_f1 | S   | 0.52          | 0.006             |
| l_choice_l_f1 | S   | 0.58          | 0.006             |
| r_choice_r_f1 | I   | 0.34          | 0.001             |
| l_choice_l_f1 | I   | 0.37          | 0.001             |
| r_choice_r_f1 | M   | 1.25          | 0.002             |
| l_choice_l_f1 | M   | 1.36          | 0.003             |


**All other contrast bins:** 0/5 or at most 1/5 significant (no consistent signal below c=1.0).

### Verdict

Phase 4b pattern is **seed-robust** (combined + f1 significance for S/I/M; c=1.0 on f1). Mechanism **open**.

**Reproduce:**

```bash
conda activate iblenv
python scripts/run_phase4b_multiseed.py \
  --output-dir output/phase4_multiseed \
  --seeds 42 7 123 999 2024 --nrand 100
```

**Outputs:**

```
output/phase4_multiseed/
  seed_{N}/absence/figs/phase4_no_prior_mod/...
  aggregate/phase4_multiseed_split_contrast.csv
  aggregate/phase4_multiseed_summary.csv
  aggregate/phase4_multiseed_by_split_all_seeds.csv
```

---

## Combined results (4 splits summed)

From `combined_regde` over all `act_block_duringstim` splits.


| Metric                                  | True labels | Shuffle nulls (n=100)                     |
| --------------------------------------- | ----------- | ----------------------------------------- |
| `curve_mean` (time-avg distance)        | **1.298**   | min 0.118, **median 0.174**, max 0.251    |
| `curve_amp` (max − min)                 | **4.010**   | min 0.460, med 0.644, max 0.951           |
| `early_mean_direct` (bins 0–4)          | ~0          | ~0                                        |
| `gain_late_mean_direct` (bins 4+)       | **1.375**   | —                                         |
| Shuffles with mean ≥ true               | —           | **0 / 100**                               |
| `p_mean`, `p_amp`, `p_offset`, `p_gain` | —           | **0.0** (true beats all shuffles on each) |


True combined distance is ~**7.5×** the shuffle median on `curve_mean`. This matches `s_shuffle_control.png`: red curve well above gray nulls.

---

## Per-split, per-contrast results

Distance metric per row: time-mean of the bin-wise squared S trajectory difference (high vs low prior) **within that contrast only**, using contrast-matched shuffles that preserve per-contrast high/low counts.

`shuffle_curve_mean_`* = statistics over 100 contrast-matched shuffles for that (split, contrast) bin.

### `act_block_duringstim_r_choice_r_f1`


| contrast | n_high | n_low | true mean | shuffle med | shuffle min | shuffle max | p(shuf≥true) |
| -------- | ------ | ----- | --------- | ----------- | ----------- | ----------- | ------------ |
| 0.0      | 53     | 360   | 0.457     | 0.048       | 0.009       | 0.164       | 0.0          |
| 0.0625   | 137    | 702   | 0.240     | 0.020       | 0.006       | 0.070       | 0.0          |
| 0.125    | 222    | 856   | 0.267     | 0.012       | 0.003       | 0.060       | 0.0          |
| 0.25     | 325    | 947   | 0.054     | 0.009       | 0.002       | 0.032       | 0.0          |
| 1.0      | 400    | 950   | 0.096     | 0.006       | 0.002       | 0.017       | 0.0          |


### `act_block_duringstim_l_choice_l_f1`


| contrast | n_high | n_low | true mean | shuffle med | shuffle min | shuffle max | p(shuf≥true) |
| -------- | ------ | ----- | --------- | ----------- | ----------- | ----------- | ------------ |
| 0.0      | 315    | 47    | 0.448     | 0.050       | 0.012       | 0.219       | 0.0          |
| 0.0625   | 793    | 144   | 0.467     | 0.018       | 0.006       | 0.107       | 0.0          |
| 0.125    | 754    | 206   | 0.197     | 0.014       | 0.004       | 0.085       | 0.0          |
| 0.25     | 913    | 320   | 0.062     | 0.009       | 0.002       | 0.032       | 0.0          |
| 1.0      | 922    | 420   | 0.101     | 0.006       | 0.002       | 0.018       | 0.0          |


### `act_block_duringstim_l_choice_r_f2`


| contrast | n_high | n_low | true mean | shuffle med | shuffle min | shuffle max | p(shuf≥true) |
| -------- | ------ | ----- | --------- | ----------- | ----------- | ----------- | ------------ |
| 0.0      | 80     | 107   | 0.355     | 0.042       | 0.012       | 0.214       | 0.0          |
| 0.0625   | 116    | 217   | 0.630     | 0.030       | 0.008       | 0.143       | 0.0          |
| 0.125    | 58     | 134   | 0.977     | 0.052       | 0.015       | 0.191       | 0.0          |
| 0.25     | 23     | 52    | 0.769     | 0.134       | 0.037       | 0.401       | 0.0          |
| 1.0      | 0      | 1     | —         | —           | —           | —           | —            |


### `act_block_duringstim_r_choice_l_f2`


| contrast | n_high | n_low | true mean | shuffle med | shuffle min | shuffle max | p(shuf≥true) |
| -------- | ------ | ----- | --------- | ----------- | ----------- | ----------- | ------------ |
| 0.0      | 124    | 96    | 0.311     | 0.037       | 0.010       | 0.176       | 0.0          |
| 0.0625   | 190    | 113   | 0.285     | 0.033       | 0.008       | 0.129       | 0.0          |
| 0.125    | 132    | 56    | 0.474     | 0.056       | 0.016       | 0.259       | 0.0          |
| 0.25     | 57     | 22    | 0.703     | 0.136       | 0.051       | 0.444       | 0.0          |
| 1.0      | 1      | 0     | —         | —           | —           | —           | —            |


**Per-contrast pattern:** True > shuffle at **every** contrast level with adequate trials (18/20 rows). Effect is largest on error-trial splits (`*_f2`), especially mid contrasts (0.125, 0.25). Two `c=1.0` bins have only one trial on one side → skipped.

**Asymmetry note:** `l_choice_l_f1` has many more high- than low-prior trials (block-left heavy); `r_choice_r_f1` is the mirror. True distance remains elevated vs shuffle in both directions.

---

## Code / outputs added today

`**simulate_recovery.py`**

- `write_s_prior_shuffle_diagnostics()` → per-shuffle `curve_mean` / `curve_amp` CSV
- `write_split_contrast_shuffle_diagnostics()` → per-split, per-contrast true vs shuffle means
- Both run automatically in `process_condition()` (S-prior-only mode)

**Artifact paths**

```
output/absence_shuffle_debug/absence/
  figs/s_prior_shuffle_nulls.csv          # 100 shuffles, combined curve stats
  figs/s_prior_shuffle_summary.csv        # true vs shuffle summary
  figs/s_prior_split_contrast_shuffle.csv # per-split × contrast table above
  figs/s_shuffle_control.png
  summary.json
  res/                                   # split-level regde outputs
```

**Reproduce**

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

---

## Log / changelog

- Initial absence rerun (seed 42, 100 shuffles); combined + per-contrast diagnostics; shuffle CSV writers added.
- Goal reframed — debug *why* absence shows S prior distance ≫ shuffle, not whether `g_s=d_s=0` is set correctly. Investigation plan added (phases 0–5).
- Random ITI labels test (`--random-prior-labels`, 50 reps): true curve_mean **1.30**, random median **0.034**, shuffle null median **0.17**. Same S trajectories — effect is **label assignment only**, not S dynamics.

## Random ITI labels (Phase 4 control)

Same absence trajectories (seed 42); only grouping changes.


|                                        | curve_mean |
| -------------------------------------- | ---------- |
| True ITI P                             | **1.30**   |
| Contrast-matched shuffle null (median) | 0.17       |
| Random 0.8/0.2 labels (median, n=50)   | **0.034**  |


Random labels collapse the effect below shuffle null (same trajectories; only grouping changes).

Output: `output/random_prior_test/absence/figs/random_prior_labels/`

- Phase 4a code audit complete. No P→S coupling bug when `g_s=d_s=0`.
- Phase 4b `--phase4-no-prior-mod`: S combined 1.30→0.11; residual sig at c=1.0 `*_f1`; f2 n.s.
- Phase 4b constant-S0 and multiseed runs completed. Trial-structure/history excluded as mechanism; see **What we already know**.

