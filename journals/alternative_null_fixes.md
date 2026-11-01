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

## 2026-10-30c — Harris reruns submitted; `--list-splits` bug

The three Harris arms were submitted on ORCD with `DEBIAS_EUC=1` (main `a61af57`). `set_debias_euc` printed its suffix message to stdout. `--list-splits` emitted that line, so each arm queued one extra "split" named after the message. Its shards died at once ("Splits missing from align/pre_post"), and its finalize job stays pending (DependencyNeverSatisfied) until cancelled. The real splits were unaffected. Fix: the message goes to stderr. Not committed or on `main` yet as of this entry.

## 2026-11-01 — full-session θ prefit for the sticky arm

The sticky arm had no θ prefit. `--prefit-actkernel` fits the `bwm_include` rows of the aggregate trials table and writes the masked `{eid8}` pickles, not `{eid8}_fullsess`. Without a prefit, each sticky shard fits the MCMC for every session it touches the first time it needs it. That causes three problems:
- The same session gets fit about 12 times, once per split.
- Two jobs can write the same pickle at once. `load_or_train` only checks whether the file exists, so another job can read a half-written pickle.
- MCMC differs from run to run, so different splits can get a different θ for the same session.

Changes:
- `run_goal2_splits.py --prefit-actkernel --full-session [--shard-idx k --n-shards N]`. It covers `bwm_query`'s 459 eids, split across shards (`eids[k::N]`). Each session is fit on `_full_session_trials_for_ak_fit(eid)` with `full_session=True`, so pickles land in `{eid8}_fullsess`. Sessions that already have a `train_*.pkl` are skipped. The old mode is unchanged.
- `run_goal2_ak_prefit_slurm.sh` takes `PREFIT_FULL_SESSION=1`, `SHARD_IDX` and `N_SHARDS`.
- `submit_standard_alt_nulls_orcd.sh`: `ARM=sticky` (or `all`) first submits `N_PREFIT_SHARDS=24` prefit jobs (8G, 12 h). Every sticky shard job then waits on `afterok` of all 24. `PREFIT=0` skips the prefit.
- Timing: a 5,000-step fit runs about 5 steps/s on a 692-trial session, so up to about 17 min per session. That is about 19 sessions per shard, under 6 h.
- If a prefit shard times out, the sticky jobs never start. Cancel them and resubmit the same command: done sessions are skipped, and `CLEAR_STREAM=1` clears the `_debias` sticky outputs.

Checks:
- One-session smoke test (`ACTKERNEL_NB_STEPS=20`, shard 0/500): 692 full-session trials, pickle in `004d8fd5_fullsess_nb20`. Rerunning with a pickle in `004d8fd5_fullsess` skipped the fit. Both test dirs were removed.
- Dry run of `DEBIAS_EUC=1 ARM=sticky` with `sbatch` stubbed: 100 jobs, made up of 24 prefit jobs plus 12 splits (8 choice, 2 stim-side prior, 2 fully unsplit prior). Every one of the 72 shard jobs depends on all 24 prefit jobs. The jobs export `STANDARD_ACT_PRIOR=1`, `ACTKERNEL_NULL_MODE=strat`, `ACTKERNEL_LATE_STICKY=1` and `DEBIAS_EUC=1`; the suffix is `_pseudo_strat_sticky_debias`. There is no junk split.

## 2026-11-01b — Harris rerun results (noise-corrected, no-choice fix, product pooling)

All 12 `{split}_harris_unique_debias` files are back from ORCD (main `a61af57`). Splits are grouped as in the figures, with each group's regde stacks summed over its splits; when null counts differ across splits, 2,000 random combinations are drawn. The score is the time-mean p, then BH over the 202 regions in the figure sample (`fig2/res_standard/staging/act_block_only.csv`). `harris_old` is the Oct 19 `_harris_unique` set. `shuffle` is the plain shuffle catalog, which is still uncorrected: its `_debias` rerun has not been done, but the 10-30b check shows the correction barely moves shuffle ranks. Per-region p and q: `manifold/res_standard/harris_unique_debias_group_fdr.csv`.

| group | splits | Harris new FDR .01 / .05 | Harris old | shuffle |
|---|---|---:|---:|---:|
| choice during stim | 4 `choice_duringstim_*_act` | 65 / 94 | 21 / 33 | 88 / 113 |
| choice around movement | 4 `choice_stim_*_act` | 102 / 128 | 53 / 71 | 115 / 137 |
| prior, stim-side | `act_block_duringstim_{l,r}` | 14 / 21 | 4 / 13 | 123 / 158 |
| prior, fully unsplit, stim | `act_block_duringstim_fully_unsplit` | 21 / 50 | 13 / 27 | not run |
| prior, fully unsplit, move | `act_block_duringchoice_fully_unsplit` | 35 / 71 | 14 / 53 | not run |

- **Choice.**
  - New Harris is close to the shuffle (Spearman 0.96 during stim, 0.94 around movement).
  - At FDR 0.01 its hits are a subset of the shuffle's (65 of 88, 102 of 115). At 0.05 the only Harris-only regions are 5 around movement: CLA, ENTm, ORBm, PPT, VISam.
  - The old Harris null was much wider, from the no-choice coding, the class-balance noise term, and min-U pooling (as low as 202 nulls per region). With those fixed, the choice result holds up under session permutation.
  - The corrected null mean is about −0.035, against an observed mean of 0.54 (during stim) and 0.89 (around movement).
- **Prior, stim-side.**
  - Only 14 regions pass at 0.01, against 123 under the shuffle, and all 14 are also shuffle hits: CLA, CP, GRN, ILA, IRN, MOp, MRN, PL, RN, SCm, SOC, VPL, FOTU, LPO.
  - The corrected Harris null still has a mean of 0.21, against an observed 0.29. The noise floor is removed, so what remains is temporal structure: block-length label runs lined up with slow drift in the recording.
  - The within-stratum shuffle breaks that structure, so most of its prior hits are not separable from slow drift under this null. This is the drift confound that session permutation exists to test.
- **Prior, fully unsplit.** 21 regions pass during stim and 35 around movement. There is no shuffle file for these splits.
- **Coverage (problem 2).**
  - All 202 figure regions are present in every group.
  - The donor gap now costs 2 insertions per split, both probes of one long session.
  - Most other skips are the usual min-5-per-side rule (116–206 per choice split, 2–6 per prior split), which the shuffle also applies. A few are other errors (2 each in two choice splits).
- **Pooling (problem 3).**
  - Every group region now has 2,000 nulls.
  - In the choice groups, 51 regions still contain an insertion with fewer than 99 unique donor draws (15–17 with fewer than 20). Product pooling keeps those regions resolvable, since each one combines several insertions.
  - The prior groups have no such insertions.

## 2026-11-01c — Harris arm for the Figure 4 f1/f2 prior splits (set up, not run)

Purpose: compare the Figure 4 prior results with the shuffle under session permutation. The stim-side prior lost most of its shuffle hits under Harris (11-01b).

- **Splits:** the 8 `act_block_*_f1/f2` splits from the shuffle catalog.
- **Contrast:** act prior L vs R.
- **Stratum:** stim side × choice. That is the same cells as the journal's choice × feedback (stimOn splits) and stim × feedback (movement splits); choice is ±1 only.
- **Windows:** stimOn 0–150 ms for `act_block_duringstim_*`; firstMovement 150 ms before for `act_block_stim_*_duringchoice_*`.
- **Prior rows:**
  - Recipient: α=0.2 kernel on the full session, then the mask, then the true-0.5 drop, min 5 per side.
  - Donor: `act_binary` from `choice_donors_standard.npy`, the same pipeline, in the donor's own stim × choice cell. The donor view follows the split's alignment: stim view for stimOn splits, move view for movement splits.
- **Null and statistic:** `nrand=2000`, `DEBIAS_EUC=1`, product pooling. Suffix `_harris_unique_debias`.

No new analysis code was needed. `_act_block_conditioning_spec` already gives stim × choice for these names, and `_donor_block_conditioning_mask` and `_harris_block_donor_candidates` match it on the donor. Added:
- Preset `standard_prior_harris_f12` in `run_goal2_splits.py`.
- `ARM=harris_prior_f12` in `submit_standard_alt_nulls_orcd.sh`: 4 shards, 8G, 12 h, waiting on the donor job. It is not part of `ARM=all`.

Checks:
- `--list-splits` with `DEBIAS_EUC=1` gives the 8 splits.
- A stubbed-`sbatch` dry run gives 41 jobs (the donor job plus 8 × (4 shards + finalize)). All 32 shards wait on the donor job, and the exports are `SESSION_SHUFFLE_NULL=1`, `STANDARD_ACT_PRIOR=1`, `DEBIAS_EUC=1`.
- On a fake standard bank with no-choice trials, the null runs for one stimOn split and one movement split. Every donor-stratum trial has the split's stim side and choice.
- The shuffle `_regde` files for all 8 splits are local, for the comparison.

## Rerun plan (Harris done 11-01; shuffle and sticky not started)

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

1. Done 10-30 (`a61af57`). Still to copy: the stderr fix (10-30c) and the sticky prefit (11-01).
2. Whether to flag or rerun the older non-standard structured choice nulls (`res/new`, `res_sc_choice_harris_unique`, sticky exclusion) that problem 1 affects.
3. Whether to add the per-insertion ActionKernel fallback for Harris coverage (problem 2).
