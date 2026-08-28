# Prior definitions and label conventions (true block, action kernel, Bayes-optimal)

**Scope:** how the prior label is computed for each split family in the real-data pipeline, the split-naming conventions that depend on it, and the two unresolved definition mismatches.

**Status:** action-kernel and Bayes-optimal priors are implemented and wired; a prior-type routing bug for `act_block_*` was found and fixed on 2026-07-27. Two conventions remain open (fixed α vs per-session fit; drop-0.5 timing). Bayes shuffle tables: [Bayesian prior](bayesian_vs_act_prior.md).

Sources: dated entries 2026-07-12 (Goal 1), 07-12e, 07-12f, 07-12h, 07-20h, 07-27e.

---

## The three prior types

| Type | Name trigger | Definition |
|------|--------------|------------|
| **True block** | default (no `act` / `bayes` in the split name) | `probabilityLeft` from the task (0.8 / 0.2 after dropping 0.5 blocks) |
| **Action kernel** | `'act' in split` | EMA over the animal's own choices, binarized |
| **Bayes-optimal** | `'bayes' in split` | Inferred P(stim left | past stimulus sides) under the IBL generative model, binarized |

### Action kernel (`action_kernel_priors`, α=0.2)

Already in `get_d_vars` when `'act' in split`:

1. Drop true-block 0.5 trials (`probabilityLeft != 0.5`).
2. EMA over **choices**: `prior ← α·[choice>0] + (1−α)·prior` (choice > 0 = left).
3. Binarize to **0.8 / 0.2** (`≥0.5 → 0.8`) and **overwrite** `probabilityLeft` (the true block is kept in `true_priors`).
4. Downstream L-vs-R-prior splits use that overwritten column unchanged.

### Bayes-optimal (`bayesian_priors`)

Findling et al., Nature 2025, SI §1.1.1. Wired the same way when `'bayes' in split`:

1. Infer the continuous **P(stim left | past stimulus sides)** under the IBL generative model (τ=60, γ=0.8, block length 20–100).
2. Computed on the **full** trial list **before** dropping 0.5 blocks (it needs the full stimulus history); then the same 0.5 filter, then overwrite `probabilityLeft` with 0.8/0.2. The continuous trace is kept in `bayes_priors`, the true block in `true_priors`.
3. Split names mirror act: `bayes_block_*` (duringstim / duringchoice / stim_l / stim_r / only) plus contrast registration (`GOAL3_BAYES_*`).
4. Presets in `scripts/run_goal2_splits.py`: `stimOn_times_bayes`, `goal3_duringstim_bayes`, `goal3_duringchoice_bayes`, `goal3_bayes_all`.

**Smoke (priors only):** 80 left stimuli → P(left) ≈ 0.77 (→ γ); 80 right → ≈ 0.23; a mid-session switch flips the binary label 0.8 → 0.2.

**Null choices (2026-08-23):** `*bayes*` structured nulls sample choices from this same prior via the IBL OptimalBayesian policy (fixed ζ=0.1, lapse=0.05) + copy-last — not from fitted ActionKernel. See [structured nulls](structured_nulls_choice_lr.md) 2026-08-23.

### Encoding-model prior (variance partition)

`get_var_partition` defaults to `prior_type='act'`: `action_kernel_priors` computed on the full choice sequence, using the **continuous** EMA − 0.5 as the regressor. True-block 0.5 trials are **kept** (the kernel needs them). Override with `--prior-type block` for comparison. See [variance partition](variance_partition_mixed_regions.md).

---

## Split naming

### Stim-side splits renamed: were ITI, now during-stim (2026-07-12f)

**Bug:** bare `*block_stim_l` / `*block_stim_r` (no choice × feedback) lacked `durings`/`duringc` in the name, so the align setup assigned the **intertrial** window `[0.4, −0.1]` despite stimOn alignment.

**Fix:** renamed to `*block_duringstim_l` / `*block_duringstim_r` for `block_`, `act_block_`, `bayes_block_`. The substring `durings` now selects the post-stim window `[0, 0.15]`. Scripts and presets updated (`stimOn_times_act`, `stimOn_times_bayes`, submitters).

Any cached outputs under the old names are **invalid for during-stim claims** and must be re-run.

### Bayes stim L–R splits in the cached pipeline (2026-07-12h)

Ported choicestim's **stim L vs R** contrasts into `block_analysis_allsplits.py` with Bayes-optimal prior labels. Do **not** revive `choicestim_analysis.py` for new runs (legacy docstring only).

| Timeframe | Splits | Window |
|-----------|--------|--------|
| `stim_duringstim_bayes` | `stim_choice_{r,l}_block_{r,l}_bayes` | `[0, 0.15]` — stim L/R, fixed choice + Bayes prior |
| `stim_duringstim1_bayes` | `stim_block_{l,r}_bayes` | `[0, 0.08]` — stim L/R, fixed Bayes prior only |

Parity with choicestim act: **no** true-block 0.5 drop before the prior overwrite for these names.

```bash
python scripts/run_goal2_splits.py --preset stim_duringstim_bayes --list-splits
python scripts/run_goal2_splits.py --preset stim_duringstim1_bayes
python scripts/run_goal2_splits.py --preset stim_lr_bayes_all
```

---

## Prior-type routing audit and fix (2026-07-27e)

`_split_uses_act_prior` had been narrowed to `*_act` / `_act_` and **missed `act_block_*`**, so both the recipient observation and the donor nulls for `act_block_*` splits silently fell back to **true block**. The historical `'act' in split` test was restored, so `act_block_*` uses the action-kernel binary on both recipients and donors (the same helper as choice `*_act` and AK stratified pseudo-sessions). Bayes still routes via `bayes` in the name; everything else uses true block.

For the choice null schemes (Harris / AK strat / fixed), donor or pseudo-session stratum priors are recomputed via `_stratum_prior_for_stream` → `action_kernel_priors` on that stream's own choices, which is correct. Non-`act` choice splits use true `pleft`; `*bayes*` use Bayes.

**Donor / recipient 0.5 parity:** donors now drop true `probabilityLeft == 0.5` **before** `action_kernel_priors` (the same order as recipient `get_d_vars`), then intersect the conditioning with that keep mask.

---

## Open questions

### 1. Fixed α vs per-session action-kernel fit

The analysis currently uses a **fixed** `α = 0.2` in `action_kernel_priors` on each session's choice sequence (the same α everywhere), then pools into the supersession.

Should we instead run `fit_action_kernel` **per session** (MCMC → a session-specific α, and optionally the full `[α, ζ, lapse±]`), recompute that session's continuous/binary act priors from the fitted kernel, and only then pool into the supersession for all act-conditioned analyses?

*(Raised 2026-07-23 in the choice actkernel null work; see [structured nulls](structured_nulls_choice_lr.md).)*

### 2. Unify drop-0.5 timing for act labels across split families

Prior-distance `*block*` splits (including all `act_block_*`) drop true `probabilityLeft == 0.5` **then** run the action kernel. Choice L–R splits (`choice_*`, choicestim) **keep** 0.5 and run the kernel on the full sequence. The same biased trials can therefore get different binary act sides across families.

Quick check on 2 alyx sessions (biased trials only):

| eid (short) | n_0.5 / biased | binary flips | cont. MAE |
| ----------- | -------------- | ------------ | --------- |
| `4364a246…` | 37 / 124 | 1/124 (0.8 %) | 0.006 |
| `56956777…` | 45 / 200 | 2/200 (1.0 %) | 0.006 |

Flips occur early in the biased sequence (leftover influence of the initial unbiased block). The effect is small, but it is a real definition mismatch for the "same" trials. Decide whether to standardize one convention, and if so whether existing act-conditioned results need a re-run.

*(Raised 2026-07-27.)*

---

## Open goal — Bayesian prior on sensory responses; conflict vs alignment with recent reward

On **real data**, test whether sensory-population prior effects hold when the prior is framed as a **Bayesian / sensory prior**, and stratify trials by whether the **current block prior conflicts** with or **aligns** with **recent reward history**.

Motivation: block identity and recent outcomes can pull in opposite directions; pooling them may mask or invent a sensory prior effect.

**Questions:**

1. Is there a significant prior effect in sensory responses under a Bayesian prior framing?
2. Does any effect concentrate on conflict trials, alignment trials, or both?
3. Does the conclusion change relative to the usual block-prior split (pooled across recent-reward history)?

**Status:** the Bayes prior machinery is implemented and the `bayes_block_*` splits are registered. Shuffle tables exist for duringstim prior L–R (4-split 57 FDR @0.01; stim-side 116) — see [Bayesian prior](bayesian_vs_act_prior.md). Harris unique and conflict-vs-alignment have not been scored.
