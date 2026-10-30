# Alternative-null fixes (Harris, sticky)

**Goal.** Fix the problems found in the alternative null schemes for the standard splits (Harris unique and the ActionKernel pseudo-session with copy-last, `_pseudo_strat_sticky`), then rerun their standard splits. The split catalog, labels, strata and windows are in [standard splits](standard_splits.md). That file stays the reference for what each split is. This file tracks what is wrong with the nulls and what has been fixed.

Code: `block_analysis_allsplits.py`. Choice nulls go through `_get_d_vars_session_shuffle`. Prior nulls go through `_get_d_vars_block_harris`. Harris donors come from `manifold/choice_donors_standard.npy` (`build_standard_choice_donor_bank`).

## 2026-10-30 — problems found

A review against the standard-splits spec found that these parts match:

- The donor bank: kernel on the full session, then the mask, then the true-0.5 drop, with separate stim and move saturation views.
- The strata for each arm, and the windows.
- The recipient side for the prior arms: observed curves are identical to the plain-shuffle files, except where Harris skipped an insertion (problem 2).

These are the problems.

### 1. No-choice trials counted as right choices (fixed 2026-10-30)

The structured choice nulls set the label to `ys = choice == 1` on every stratum trial. So a `choice == 0` trial went into the right-choice class, both in the observed split and in the Harris donor labels. The plain within-stratum shuffle keeps only `choice == ±1` (explicit filters in `get_d_vars`). The two arms therefore scored different trials.

On the standard Harris choice files, about a third of regions (43–66 per split) had observed curves different from the shuffle files with the same neuron count (median 1.4%, up to about 16%). Every one of those regions was recorded in a session that has a kept no-choice trial in that stratum (9–21 sessions per split, from the BWM trials table).

**Scope.** This is not Harris-only. The observed labels in `_get_d_vars_session_shuffle` are shared by every structured choice null:

- Harris: the old `choice_donors.npy` bank and the standard bank.
- ActionKernel nulls in every mode: `unconstrained`, `strat`, `fixedstim`, and sticky / copy-last.

Simulated pseudo-session choices are only ±1 (`scripts/simulate_synthetic_choices.py`), so the pseudo labels were fine. What was wrong in the ActionKernel arms is the observed split. The coding dates from `4f29bc1` (2026-07-13, session shuffle for choice). `exclude_nochoice` has been off in `load_trials_and_mask` since the loader was added, so no-choice trials were always in the tables.

Not affected:

- **Plain shuffles.** Choice L vs R, stim L vs R with a choice stratum, and the act-block f1/f2 splits all filter `choice == ±1`.
- **Prior Harris / ActionKernel arms.** Their labels are the prior. Choice enters only through the f1/f2 conditioning mask, and that mask uses `isclose(choice, ±1)`.
- **The kernel.** A 0 holds the prior on purpose (see [standard splits](standard_splits.md) 2026-10-15).

Every structured choice-null file made since July is affected, not only `res_standard`. That includes the `res/new` choice Harris / ActionKernel runs, `res_sc_choice_harris_unique`, and the sticky-exclusion choice runs.

**Fix.**
- `_get_d_vars_session_shuffle`: the eligible trials are the stratum ∩ `choice ∈ {1, −1}`.
- `_donor_stratum_mask`: the donor stratum is also restricted to ±1, for both the standard and the legacy bank.

An offline check (fake trials with 3 no-choice trials in a 30-trial stratum, spike binning stubbed) gives 27 observed trials and no zeros in the donor stratum. Not yet copied to `main`, not committed, not rerun.

### 2. Harris skips insertions with no long-enough donor

`_compute_control_D_harris` and `_compute_control_D_harris_block` raise `InsufficientTrials` when no donor has at least `n_elig` trials in the stratum. The insertion then drops out of the pooled region. In the standard files this costs about 6 regions per prior split, and leaves 177 regions in the combined choice groups (the shuffle has 202). The insertions that drop are the long sessions, so the sample is not random. Kept, and now reported (10-30b).

### 3. The pooled null shrinks to the smallest per-insertion unique count

Long recipients get few unique contiguous donor windows. When any insertion saturates, `_pool_insertion_curve_arrays` pools by product Monte Carlo with `n_mc = min(U_i)`. So one short-pool insertion caps the whole region. In the standard choice files some regions pool only 5 nulls. In the combined choice groups, 51 of 177 regions have fewer than 99, so they cannot reach raw p = 0.01. Fixed 10-30b.

### 4. Donor windows do not match the recipient's label balance

The plain shuffle keeps the recipient's exact L/R (or 0.8/0.2) counts. A donor window of length `n_elig` does not. The squared-mean distance has a noise floor proportional to 1/n0 + 1/n1. So a donor window with a worse balance raises the null, and draw-to-draw balance changes widen it.

Estimate from the BWM trials table, `act_block_duringstim_l`:
- Donor floor / recipient floor: median 1.15, higher in 72% of sessions.
- Relative SD across donor draws: about 0.29.

This applies to choice Harris as well, where the balance varies even more across strata. It also applies in principle to the ActionKernel `strat` mode, whose window comes from a pseudo stratum. Fixed 10-30b with a noise-corrected distance. Balance-matched donor windows were not chosen, because they shrink the donor pool and make problems 2 and 3 worse. The cross-validated distance (`d_xnobis`) was not chosen either: it is noisier and is not the figures' statistic.

### 5. Sticky θ fit provenance (from standard splits, 10-19 / 10-26)

- The 8 sticky choice files are from Oct 16, fit on the masked-table θ, and must be rerun on `{eid8}_fullsess`.
- For the 4 sticky prior files it is not known which θ was used.
- Problem 1 also applies to the sticky choice arm.

## 2026-10-30b — fixes for problems 2–4

Chosen after weighing which problems come from the Harris method itself and which come from our implementation:

- **Problem 4** comes from the distance, not from Harris. Harris is meant to keep the label's temporal statistics, and session-to-session differences in class balance are part of that. They only matter because the squared mean difference carries a noise floor that depends on class counts.
- **Problem 3** comes from our pooling rule.
- **Problem 2** is partly built into session permutation: a donor must be at least as long as the recipient.

### Problem 4: noise-corrected distance (`--debias-euc` / `DEBIAS_EUC=1`)

`_region_perm_metrics`, per neuron and bin: `(m0 − m1)² − s²·(1/n0 + 1/n1)`.

- `s²` is the variance of all trials in the split (`ddof=1`). It is computed exactly from `m0, m1, v0, v1` and the counts (within-class plus between-class), so it is the same for every labeling. The result has null expectation 0 for any class counts.
- Under a shuffle with fixed counts the correction is a constant per neuron and bin, so the shuffle ranking barely changes. Under Harris, where the counts change from draw to draw, it removes the class-count effect.
- The first version subtracted per-class variances, `v0/(n0 − 1) + v1/(n1 − 1)`. That is also unbiased, but with minority classes of about 5 trials those variances are noisy, and the noise entered the statistic even under a fixed-count shuffle. On the 7 local insertions (shuffle, `nrand=200`, 115 region × insertion × split) it moved per-insertion p by a median of 0.065, up to 0.77 on the choice splits. Rank correlation was 0.87, and the p ≤ 0.01 hits were unchanged (3/3). Replaced.
- Pooled version, same 7 insertions and shuffle seed: rank correlation 0.986. Median |Δp| is 0.020 (95th percentile 0.10, max 0.21); the residual comes from the square root and the time mean, which are not linear. The p ≤ 0.01 hits are the same (3/3). At p ≤ 0.05 there were 10 before and 9 after, 8 in both. The largest moves are the 80 ms stim split and the congruent during-stim choice split. This is a per-insertion check; pooled region results have not been compared yet.
- `m0` and `v0` are the `ys` class in every caller, including the shuffle's `bins[0]`.
- The pooled sum can now be negative, so the region curve is a signed square root (`_euc_root`).
- It is applied to every arm (shuffle, Harris, ActionKernel) so the arms stay comparable.
- Off by default. When on, `_debias` is appended to every output and stream-checkpoint name: `{split}_debias`, `{split}_harris_unique_debias`, `{split}_pseudo_strat_sticky_debias`.

`d_var` and `d_xnobis` are unchanged. The legacy `d_var_stacked` paths (not used by the standard runs) still take a plain square root and would give NaN with the switch on.

Both standard submit scripts (`submit_standard_fig2_orcd.sh` and `submit_standard_alt_nulls_orcd.sh`) take `DEBIAS_EUC=1`. They export it, and with it on, `CLEAR_STREAM=1` clears only the `_debias` files, so the uncorrected catalogs are kept.

Offline check, 300 trials × 40 Poisson neurons with no label effect:

| class split | uncorrected null mean | corrected null mean |
|---|---:|---:|
| 50 / 50 | 1.59 | −0.001 ± 0.010 |
| 20 / 80 | 2.47 | +0.009 ± 0.013 |
| 6 / 94 | 7.12 | +0.066 ± 0.039 |

### Problem 3: pool from the product of unique nulls

`_pool_insertion_curve_arrays`, when any insertion's null stack is short:
- Draw `nrand` combined nulls (each the sum of one null per insertion) instead of `min(U_i)`.
- When ∏U_i ≤ `nrand`, list every combination, so the p is exact.

The all-region pool keeps Monte Carlo draws (`exact=False`), so its row counts stay equal across regions. Complete equal-length stacks still use the aligned sum.

Each region record now carries `n_insertions` and `min_insertion_nulls`. A region dominated by one insertion with few unique draws really cannot go below 1/(1 + U); the field makes that visible.

Offline check:
- 3 × 4 unique nulls give the 12 exact combinations.
- 3 × 4 × 40 gives 2,000 Monte Carlo draws.
- Complete stacks give the aligned sum.

### Problem 2: report the dropped insertions (not removed)

Session permutation needs a donor at least as long as the recipient, so this is kept as a property of the method:
- Every `InsufficientTrials` skip in a stream-pooled run is appended to `_stream_acc/{name}[.shardN].skipped.jsonl`.
- Finalize writes `{name}_skipped.json` next to the results (`n_pooled`, `n_skipped`, insertion → reason), leaving out insertions that a restart later pooled.

Options not taken, in case coverage matters for a figure:
- **Per-insertion fallback** to the ActionKernel pseudo-session null, as a separately labeled arm.
- **Linear-shift test** (shift the recipient's own labels). Too few shifts for block-scale prior labels.
- **Concatenated or wrapped donors.** Breaks the temporal structure at each seam.

Offline check: a fake standard bank with no-choice trials runs the Harris choice control (30 unique nulls from 6 donors) with the correction off and on. The suffixes are `_harris_unique_debias`, `_pseudo_strat_sticky_debias` and `_debias`. The skip summary drops a pooled insertion and ignores another split's log.

## Rerun plan (not started)

All standard arms get rerun with `DEBIAS_EUC=1`, so they are scored on the same statistic. The uncorrected files stay as they are.

| arm | splits | why rerun |
|---|---|---|
| Shuffle | 28 standard splits (`standard` preset, `act_block_only` included) | corrected distance |
| Harris choice | 8 `choice_*_act` (`standard_choice_harris`) | problem 1, corrected distance, pooling |
| Sticky choice | 8 `choice_*_act` | problem 1, full-session θ, corrected distance |
| Harris prior | stim-side pair, fully unsplit pair | corrected distance, pooling |
| Sticky prior | stim-side pair, fully unsplit pair | θ provenance, corrected distance |

Before printing a submit command, follow the standard-splits rule: write down aim, splits, stratum, window, and the rows the prior sees.

## Open

1. Copy the fixes to `main` for ORCD (after a commit is asked for).
2. Whether to flag or rerun the older non-standard structured choice nulls (`res/new`, `res_sc_choice_harris_unique`, sticky exclusion) that problem 1 affects.
3. Whether to add the per-insertion ActionKernel fallback for Harris coverage (problem 2).
