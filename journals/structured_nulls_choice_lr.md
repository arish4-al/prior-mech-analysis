# Structured nulls for choice L–R (and prior L–R) neural distance

**Scope:** the problem that an unrestricted label shuffle is too narrow a null when choice/prior labels are temporally autocorrelated and neural responses drift, and every null scheme built to address it — Harris session permutation and ActionKernel synthetic sessions (stratified pseudo and fixed-stim). Late-session / perseveration **trial exclusion** is [sticky / end-of-session trial exclusion](sticky_end_of_session_exclusion.md), not a structured null.

**Status:** four valid arms exist and have been compared at α=0.01 over the full BWM. Liberality order: **Harris (unique) ≈ Harris (with replacement) < label shuffle < AK stratified pseudo (×3) < AK fixed-stim**. Harris is the only structured null **stricter** than shuffle; `_harris_unique` is the preferred file going forward. Two earlier arms (calendar-indexed Harris, unconstrained BWM pseudo-session) are **invalid** and must not be interpreted.

Sources: dated entries 2026-07-12 (Goal 2), 07-13 (Harris donor-window), 07-18, 07-21 (AK audit), 07-23, 07-23b, 07-24, 07-24b–f, 07-27, 07-27b–e, 08-14, 08-14b–d, 08-17, 08-17b, 08-18. **2026-08-21:** `--actkernel-late-sticky`. **2026-08-22:** fitted+copy-last matches BWM quintiles; wired as within-stratum null for choice L–R **and** act_block (stim + move).

---



## Goal / problem statement

**Revised problem (2026-07-12):** trial averaging does **not** by itself remove the issue. At each peri-event bin the choice/prior distance is a difference between condition-averaged trial responses. If the labels occur in long epochs and neural responses also drift or remain autocorrelated across trials, the two averages can sample different parts of the session. An unrestricted label shuffle mixes those epochs and can produce a null that is too narrow.

Label autocorrelation **alone** is not sufficient: if trial responses were independent and stationary, label ordering would not matter once condition counts were fixed. The relevant failure mode is **autocorrelated choice/prior labels interacting with across-trial neural nonstationarity**.

**Affected contrasts:**

- **Choice:** animals show runs of one-sided responses, so choice L–R tests under fixed stimulus/block need a temporally structured null.
- **Prior:** true block labels are explicitly block-autocorrelated. `block_only` already generates pseudo-blocks, but other prior-distance splits used unrestricted permutations. Action-kernel and Bayes-derived priors are also temporally structured.
- **Stimulus:** stimulus identities are randomized by the task, so a task-matched pseudo-session/shuffle remains appropriate; stimulus significance is not the target of this correction.

**Terminology note:** earlier donor-window code (2026-07-13) transplanted stim×block-matched choice subsequences and was a **matched donor surrogate**, not literal Harris. That path was replaced by the Harris implementation below.

---



## The null schemes


| Scheme                                       | On-disk basename              | Status                                |
| -------------------------------------------- | ----------------------------- | ------------------------------------- |
| label shuffle (within stim×block)            | `{split}.npy`                 | baseline                              |
| option 1 — AK stratified pseudo-session      | `{split}_pseudo_strat.npy`    | valid (needs `pseudo_len_factor ≥ 3`) |
| option 2 — AK fixed real stim×block          | `{split}_pseudo_fixed.npy`    | valid, most liberal                   |
| option 3 — Harris (legacy, with replacement) | `{split}_harris.npy`          | valid after donor re-stratification   |
| option 3 — Harris **unique-null**            | `{split}_harris_unique.npy`   | **preferred**                         |
| legacy unconstrained BWM index               | `{split}_pseudosession.npy`   | **invalid** — do not interpret        |




### Option 3 — Literal Harris session permutation

1. On the **real** session only: stratify by stim (± prior/block) to get eligible trial indices `elig_idx` and the neural tensor `b` on those trials. The observed distance uses this session's real choices on `elig_idx`.
2. For each null draw: sample another `eid`'s choice sequence from the donor bank; null labels = donor choices **restricted to the same stim×prior stratum**, length-matched to `n_elig`.
3. Distance under null = re-split `b` by those labels. Tag: `null_scheme: harris_session_permutation`.
4. CLI: `--session-shuffle-null` in `scripts/run_goal2_splits.py`.

Stratifying only on the real data is enough to argue that the measured choice neural distance is not driven by stim/prior composition; the null is session-permuted behaviour applied to the same neural trials.

### Options 1 & 2 — ActionKernel synthetic choices

**Source:** `scripts/simulate_synthetic_choices.py` wraps the IBL `behavior_models` `ActionKernel`. The package is a **git submodule** at `third_party/behavior_models` (path-prepended); remote Slurm needs the repo checkout plus `torch` and `sobol_seq` in conda (MCMC init) — not a cluster `pip install behavior_models`. Init: `git submodule update --init --recursive`.


| API                                                          | Keeps real stim/block?                                                                         | Use for our choice L–R null?         |
| ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- | ------------------------------------ |
| `synthetic_sessions_from_trials` / `make_synthetic_session`  | **No** — draws pseudo blocks + contrasts via `generate_pseudo_session`, then simulates choices | **Yes** — BWM paper null (wired)     |
| `fit_action_kernel` + `simulate_choices(stim, side, params)` | **Yes** — if you pass the real session's stim/side                                             | Available helper; not the wired null |
| `synthetic_choices_fixed_stim` (added)                       | **Yes** — thin wrapper of the above                                                            | Available helper; not the wired null |


**Steps per insertion × choice L–R split** (`synthetic_choice_pseudosession`):

1. **Load / prepare trials** (as for other choice L–R paths): insertion cache → trials; apply the act/bayes prior overwrite if the split name asks for it.
2. **Stratify on the real session only** (stim side ± block/prior from the split name) → `elig_idx`. Bin spikes on those trials → neural tensor `b`. Observed choices = `trials.choice[elig_idx]`.
3. **Observed distance:** split `b` by real L vs R choices; region distances as usual (true permutation index 0 in `D[reg]['d_*']`).
4. **Fit ActionKernel once per** `eid` (`get_actkernel_choice_fit`): MCMC on the **real** session's choice / stim / side → posterior-mean `[α, ζ, lapse+, lapse−]`. Pickled under `manifold/actkernel_fits/` and shared across probes of the same eid.
5. **For each of** `nrand` **null draws:** draw a new pseudo stim/block schedule; simulate a full-session choice sequence under the fitted parameters (the prior updates from the *simulated* choices on that fake stim stream); null labels = synthetic choices at `elig_idx` (option 2 uses the real stim/block stream instead of a pseudo one; option 1 takes the pseudo's own stim×prior stratum, length-matched). Reject draws with fewer than `min_trials_per_side` on either side; keep sampling until `nrand` valid draws.
6. **Pool / p-values** as for other control runs (`uperms` counts unique label patterns).

**What is frozen vs resampled:**


| Piece               | Real / observed                                   | Under null                                           |
| ------------------- | ------------------------------------------------- | ---------------------------------------------------- |
| Neural `b`          | fixed (real spikes at real `elig_idx`)            | fixed                                                |
| Which trial indices | stim×prior stratification on the **real** session | same `elig_idx`                                      |
| Stim / block stream | real (only to define `elig_idx` + fit)            | **new** pseudo each draw (option 1); real (option 2) |
| Choices             | real                                              | simulated under fitted θ                             |
| θ                   | fitted once on the real session                   | held fixed                                           |


Rationale for regenerating the stimulus stream (option 1): the paper's choice null is behaviour under a **fresh** task schedule with the animal's fitted policy. Holding the recorded stimulus would couple null choices to the same sensory sequence that drove `b`. Indexing at the real `elig_idx` keeps the neural tensor aligned while allowing unlimited Monte Carlo draws (unlike Harris's finite donor bank).

```bash
conda activate iblenv   # needs torch; behavior_models from third_party/ submodule
python scripts/run_goal2_splits.py --preset choice_lr_session_null_all \
  --actkernel-choice-null --nrand 200
# shards: bash scripts/submit_goal2_choice_actkernel_null_sharded.sh
# smoke:  python scripts/smoke_choice_actkernel_null.py
```

**Cluster:** `NULL_SCHEME=pseudo_strat|pseudo_fixed|harris` via `scripts/submit_goal2_choice_null_sharded.sh`, or all three with `scripts/submit_goal2_choice_null_all_schemes_sharded.sh`. AK schemes need `sobol_seq` + `torch`; Harris needs the donor bank job (auto-submitted). Finalize must export the same `ACTKERNEL_NULL_MODE` / `SESSION_SHUFFLE_NULL` as the shards.

### Removed: sticky psychometric synthetic-choice null

Implemented briefly (2026-07-18) as `--synthetic-choice-null` / `null_scheme: synthetic_choice_sticky`, then **removed** (2026-07-23) in favour of literal Harris session permutation.

---



## AK pseudo-session run history and the under-dispersion bug



### 2026-07-23 — first ORCD run invalid (missing `sobol_seq`)

The default submitter preset `choice_lr_session_null_all` does include all 8 act splits (duringchoice `choice_stim_*` + duringstim `choice_duringstim_*`).

**Failure:** every insertion failed with `No module named 'sobol_seq'` (ActionKernel MCMC init). Shard logs showed `ok 0/1 splits` and `MISSING shard …`. Pooled `*_pseudosession`* files from that attempt are **not** a successful null run.

**Deps:** `torch` + `pip install sobol_seq` in the ibl conda env. Clear `manifold/actkernel_fits/` and failed outputs before resubmitting.

### 2026-07-23b — re-run after `sobol_seq`: statistical bug found

The re-run landed in alyx `manifold/res/new/` with all 8 `*_pseudosession` act splits (~197–200 regions, ~49–54 k cells per split, ~79–89 % of the openalyx sibling coverage).


| α    | epoch        | shuffle | pseudosession | lost | gained | kept |
| ---- | ------------ | ------- | ------------- | ---- | ------ | ---- |
| 0.05 | duringstim   | 71      | **204**       | 0    | 133    | 71   |
| 0.05 | duringchoice | 107     | **202**       | 1    | 96     | 106  |
| 0.01 | duringstim   | 46      | **200**       | 1    | 155    | 45   |
| 0.01 | duringchoice | 84      | **201**       | 1    | 118    | 83   |


Pseudo-session calls almost every region significant — **not** a tighter null.

**Bug (under-dispersed null):** for the same insertion and the same observed euc amplitude, AK pseudo-session null amplitudes are far below label-shuffle null amplitudes (e.g. LD: shuffle null median ≈ 2.9 vs observed 3.6; AK null median ≈ 1.2, max 2.2 → false p ≪ 0.05). Pooled regde (CP): shuffle null median ≈ observed; pseudo null median ≈ **½** observed and never exceeds it.

**Mechanism:** eligibility is stim×prior (act-binary 0.8/0.2). Real choices on those trials are often **highly imbalanced** (e.g. 57 L vs 3 R). Label shuffle **preserves** that n_L/n_R → large noise-floor distances. BWM pseudo-sessions regenerate full stim/blocks, and reading choices at the real `elig_idx` gives ~**50/50** labels on the same neural trials → much smaller null distances → inflated significance. Confirmed: reshuffling those balanced AK labels still yields the low null floor; fixed-stim AK choices partially restore imbalance but stay below shuffle.

This is a null-construction bug, not a `sobol_seq` / fit crash.

### Null options considered (2026-07-23b)

Goal: restore realistic n_L/n_R (and temporal structure) on stim×block-stratified `elig_idx` without defeating the structured null.


| #     | Null                                                     | Stim×block                                                                                                       | Choice process                    | Late-session stickiness                                                        | `nrand`≈2000?                                      |
| ----- | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | --------------------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------- |
| **1** | Pseudo-session + **stim×block stratification**           | New pseudo schedule, stratified so eligible slots match the real stim×block bias context                         | AK simulate under fitted θ        | **Optional** `--actkernel-late-sticky`: copy-last on post-0.5 quintiles matching BWM Δ mean_run (08-21). Default off = stationary α | **Yes** — unlimited synthetic draws                |
| **2** | Pseudo-session on the **exact real stim×block sequence** | Pin recorded `(stim_side, pLeft)`; only choices are synthetic                                                    | AK simulate under fitted θ        | Same optional `--actkernel-late-sticky` as option 1; default off               | **Yes**                                            |
| **3** | **Harris / session transplant**                          | Recipient stim×block defines neural `elig_idx`; the **donor** is also restricted to that stratum, length-matched | Empirical choices from other eids | **Yes** — real mice carry late-session / sticky structure within stratum       | **No** — donor pool ≪ 2000 unique usable sequences |


Notes:

- The unconstrained full-BWM pseudo (calendar index into a mismatched stim×block; the former `_pseudosession` run) is **retired as a default** — ~50/50 labels → under-dispersed null. Kept only as a legacy `unconstrained` mode.
- **1 vs 2:** both fit θ once per eid, both can draw `nrand ≈ 2000`, both use a stationary AK α **unless** `--actkernel-late-sticky` (2026-08-21), and both aim to restore stim×block-appropriate choice imbalance. Option 1 draws a **new** pseudo stim/block stream per null and takes labels from the pseudo's own stim×prior stratum in temporal order, length-matched to `n_elig` — a stronger "new world" confound break, but stratum size varies per draw so short strata are rejected. Option 2 uses the **exact** recorded `(stim, side)` sequence with labels at the real `elig_idx` — the strongest match to the session's bias timeline and usually closest to the observed n_L/n_R, but a weaker confound break because neurons and the null policy share the same stim stream.
- Option 3 gives empirical sticky structure within the matched stratum, with a finite donor pool.

---



## Fixes to the structured nulls



### 2026-07-24b — stratified pseudo needs longer worlds (`pseudo_len_factor = 3`)

**Problem:** BWM same-length pseudo-sessions undersize congruent act strata (probe: need 61 trials, median pseudo stratum 36 → 0–1 % accept), so congruent insertions were skipped wholesale.

**Fix:** draw longer worlds — `n_pseudo = ceil(n_real × factor)` via `generate_pseudo_blocks(n_pseudo)` + AK simulate (the fit is still on the real session). `--actkernel-pseudo-len-factor` / env `ACTKERNEL_PSEUDO_LEN_FACTOR` (default **3**); on a low accept rate the control loop **doubles up to 16**. Always writes `_pseudo_strat` (submit with `CLEAR_STREAM=1` to clear prior stream_acc and pooled res). Probe: factor 1 → ~1 % accept; **factor 3 → 100 %** (median stratum 97).

```bash
bash scripts/submit_goal2_choice_strat_x3_sharded.sh
# or: NULL_SCHEME=pseudo_strat bash scripts/submit_goal2_choice_null_sharded.sh
```



### 2026-07-24c/d — Harris donor re-stratification

**Bug in the prior** `_harris` **run:** the recipient stim×prior defined `elig_idx` and the neural tensor `b`, but null labels were `donor_choice[elig_idx]` — a **calendar index** with no donor-side stratum match. Same pathology as the unconstrained pseudo-session: near-balanced labels on imbalanced strata → under-dispersed null → almost every region FDR-significant. **Those** `_harris` **results are invalid.**

**Fix:** for each null draw, restrict the donor to the **same** stim×prior stratum as the split (true-block `pleft`, act-binary, or bayes), then length-match choices to `n_elig` via `_ys_from_stratum_choices` (contiguous window if longer). Skip the insertion if no donor has ≥ `n_elig` stratum trials or if `nrand` balanced draws cannot be filled — no calendar or circular-shift fallback.

Real eligibility, pseudo_strat, and Harris donors now all use the same act/bayes/true stratum definition (pseudo_strat was already act-matched; only Harris was wrong).

```bash
NULL_SCHEME=harris bash scripts/submit_goal2_choice_null_sharded.sh
# or: bash scripts/submit_goal2_choice_session_null_sharded.sh
```



### 2026-07-24f — Harris unique-null sampling

**Issue:** Harris was drawing `nrand=2000` labels **with replacement** from ≪500 stratum-matched donors (random contiguous windows). Many of the 2000 were duplicate label patterns, so the p-value resolution looked like 1/2000 while `uperms` was much smaller.

**Code change** (`block_analysis_allsplits.py`):

- Keep only **distinct** label patterns in `_compute_control_D_harris`.
- Stop at `nrand` uniques, or earlier when the unique pool saturates (`HARRIS_UNIQUE_STALE_LIMIT` consecutive duplicate/reject draws).
- Finalize: equal unique counts → index-aligned sum; unequal → product-MC over each insertion's unique set with `n_mc = min(U_i)` (not padded to 2000). `_pool_insertion_curve_arrays`.
- Harris finalize **keeps** `stream_acc` by default (also `KEEP_STREAM_ACC=1`) so unique counts remain inspectable.
- Smoke: 3 donors → saturated at **5** unique nulls for `nrand=20`.
- Helper: `scripts/recompute_harris_unique_from_stream.py` (needs stream_acc). Local alyx has **no** Harris stream_acc (cleaned after the 24e finalize), so unique-null p cannot be re-derived from pooled `*_harris_regde.npy` alone.

**Overwrite safety:** unique-null runs write `_harris_unique` only. `CLEAR_STREAM=1` removes prior `_harris_unique` stream/res and refuses `SUFFIX=_harris`. Legacy `_harris` is never deleted by the submitter. Donor loading prefers the larger of `manifold/choice_donors.npy` vs `manifold/res/choice_donors.npy` (local: 457 eids under `res/`).

```bash
bash scripts/submit_goal2_choice_session_null_sharded.sh   # = NULL_SCHEME=harris_unique
python scripts/plot_choice_null_comparison_table.py \
  --arm-res ~/Downloads/ONE/alyx.internationalbrainlab.org/manifold/res/new \
  --arm-tag harris_unique --force-combine --alpha 0.01
```

**Combine fix (2026-07-27):** ragged unique counts `U` across splits broke the aligned sum; then `n_mc = min(U)` capped the minimum p at ≈0.04 → artefactual 0 FDR hits. `plot_choice_null_comparison_table._combine_split_curve_stacks` now does product-MC from each split's unique set with `n_mc = max(min(U), 2000)`.

---



## Results



### 2026-07-24 — first cross-arm comparison (α=0.01)

Data: alyx `manifold/res/new/`, all 8 act splits × `_pseudo_strat` / `_pseudo_fixed` / `_harris` / legacy `_pseudosession` (nrand=2000). Baseline = openalyx label shuffle. Combined 4-split tables; BH-FDR on `p_mean`.

**Policy:** failed null draws (cannot fill `nrand` balanced labels; no long-enough Harris donors) **skip the insertion** (logged `split skip`) — no circular-shift, unconstrained, or observed-label fallbacks.


| arm                                   | duringstim | duringchoice | median p (stim / choice) |
| ------------------------------------- | ---------- | ------------ | ------------------------ |
| shuffle (openalyx)                    | 46         | 84           | 0.071 / 0.019            |
| **pseudo_strat** (pre-×3)             | **0**      | **0**        | **0.995 / 0.987**        |
| pseudo_fixed                          | 95         | 124          | 0.009 / 0.002            |
| harris (**INVALID** — calendar index) | 201        | 202          | ~0.0005                  |
| pseudosession (legacy)                | 201        | 201          | ~0.0005                  |


**Region / cell retention vs shuffle.** Congruent = stim side matches the act-prior side (`*_stim_l_block_l_`*, `*_stim_r_block_r_*`); incongruent = crossed.


|                    | Congruent                      | Incongruent                |
| ------------------ | ------------------------------ | -------------------------- |
| **strat** (pre-×3) | ~37 % regions, **~10 % cells** | ~96 % regions, ~87 % cells |
| **fixed / harris** | ~96 % regions, ~83 % cells     | ~96 % regions, ~87 % cells |


**Why strat skipped congruent insertions:** strat length-matches labels from the **pseudo's own** stim×act-prior stratum to the real `n_elig`, rejecting if the stratum is too short or has <5 trials per side; after `nrand × 20` synthetic sessions without `nrand` accepts the insertion is skipped. Probe (local eid, short MCMC): congruent `stim_l × block_l_act` had `n_elig=61` but a median pseudo stratum of **36** → **100 % short rejects**. The same eligibility under fixed-stim: 97 % accept. Incongruent on the same eid: ~48 % accept. So congruent loss was driven by **short act-prior strata on new pseudo-sessions** (real sticky act-priors align with stim → large congruent pools; a remade pseudo + AK history does not), not primarily by the ≥5/side gate. Fixed in 24b.

### 2026-07-27 — final cross-arm comparison (α=0.01, includes `_harris_unique`)

New data: alyx `manifold/res/new/` with all 8 act splits `*_harris_unique*.npy`; legacy `_harris` untouched; coverage in the same ballpark as 24e (~96 % regions / ~83–87 % cells). Per-split unique null counts (pooled regde `n_null`) vary widely across regions (e.g. duringstim_l_block_l: min 99, median ~1520, max 2000).


| arm                                               | duringstim | duringchoice | median p (stim / choice) |
| ------------------------------------------------- | ---------- | ------------ | ------------------------ |
| shuffle (openalyx)                                | 46         | 84           | —                        |
| **harris unique** (`_harris_unique`)              | **21**     | **58**       | **0.263 / 0.092**        |
| harris with-replacement (`_harris`)               | 21         | 57           | 0.283 / 0.118            |
| pseudo_strat (×3)                                 | 74         | 105          | 0.042 / 0.009            |
| pseudo_fixed                                      | 95         | 124          | 0.017 / 0.004            |
| ~~harris calendar-index~~ (INVALID)               | 201        | 202          | ~0.0005                  |
| ~~pseudo_strat thin~~ (pre-×3; coverage artefact) | 0          | 0            | 0.995 / 0.987            |


Versus shuffle (lost / gained / kept; n=207 regions present in both):


| epoch        | arm                | shuffle | arm n  | lost | gained | kept |
| ------------ | ------------------ | ------- | ------ | ---- | ------ | ---- |
| duringstim   | **harris_unique**  | 46      | **21** | 31   | 6      | 15   |
| duringchoice | **harris_unique**  | 84      | **58** | 37   | 11     | 47   |
| duringstim   | harris (`_harris`) | 46      | 21     | 31   | 6      | 15   |
| duringchoice | harris (`_harris`) | 84      | 57     | 37   | 10     | 47   |
| duringstim   | pseudo_strat       | 46      | 74     | 12   | 40     | 34   |
| duringchoice | pseudo_strat       | 84      | 105    | 14   | 35     | 70   |
| duringstim   | pseudo_fixed       | 46      | 95     | 8    | 57     | 38   |
| duringchoice | pseudo_fixed       | 84      | 124    | 9    | 49     | 75   |


**Retention after the 24b/24c fixes** (congruent and incongruent both restored):


|                               | Congruent                      | Incongruent                |
| ----------------------------- | ------------------------------ | -------------------------- |
| **strat ×3 / harris / fixed** | ~96 % regions, **~83 % cells** | ~96 % regions, ~87 % cells |


Congruent example (`choice_duringstim_l_block_l_act`): strat 200 regions / 53.7 k cells vs shuffle 207 / 62.0 k (was 71 / 4.7 k pre-×3).

**Interpretation**

- Liberality order at α=0.01: **harris unique ≈ harris with-replacement < shuffle < strat ×3 < fixed** ≪ invalid calendar / unconstrained.
- **Harris** (either file) is the only valid structured null **stricter than shuffle** (stim 21 vs 46; choice ~58 vs 84). Prefer `_harris_unique` going forward; archive `_harris` as the with-replacement version.
- Unique vs with-replacement Harris are nearly identical (21/57 → 21/58) once combined distances use adequate MC from the unique sets — deduping per insertion does not change the supersession FDR map here.
- **pseudo_strat (×3):** full-map coverage restored, but still more liberal than shuffle (74/105). The pre-×3 "0 FDR" was congruent dropout, not calibration.
- **pseudo_fixed:** the most liberal of the valid arms (95/124).
- CP sanity check (congruent duringstim, 24e): strat null amplitude median ≈ 0.11 vs harris ≈ 0.18 vs shuffle ≈ 0.33 — strat still under-disperses relative to Harris/shuffle.
- Caveat: per-region unique pools still vary widely at the per-split level.

**Plots / CSV:** `meta/table_choice_{harris_unique,harris,pseudo_strat,pseudo_fixed,pseudosession}_vs_shuffle_*_p_mean_c_0.01.png`; `meta/choice_null_harris_unique_summary_a0.01.csv`; `meta/choice_null_res_new_rerun_harris_strat_a0.01.csv`; `meta/choice_null_res_new_summary_a0.01.csv`.

**Open decision:** pick the primary structured null for choice L–R claims (Harris = empirical sticky structure within stratum; strat = BWM-like new world with matched bias context). Optionally probe why strat null amplitudes remain lower than Harris at matched coverage.

---



## Openalyx vs alyx: `min_trials_per_side` gap

**Note (2026-07-27b):** raw observed curves match across alyx null schemes (harris / strat / fixed) when the same insertions are kept. Openalyx label-shuffle observations can differ slightly because openalyx was finalized **before** `min_trials_per_side = 5` (see [pipeline efficiency](realdata_pipeline_efficiency.md)): insertions with <5 trials on either split side were still included. All other analysis aspects are intended to be identical; residual `nclus` / amplitude differences are largely that gate plus null RNG.

**Audit (local** `~/Downloads/ONE`**, 2026-07-27):** every openalyx `manifold/res/*.npy` is still pre-2026-07-12. Criterion for "has re-run" = the same split basename exists under alyx `manifold/res{,/new,/neww}` with mtime ≥ 2026-07-12 (plain **or** tagged `_harris`* / `_pseudo_*` / `_pseudosession`).

**Openalyx splits with an alyx post-min5 re-run (12):**

- plain: `act_block_duringstim_{l,r}_choice_{l_f1,r_f2}` (4; ~2026-07-14)
- tagged only (structured nulls; no plain shuffle on alyx at the time): `choice_duringstim_{l,r}_block_{l,r}_act`, `choice_stim_{l,r}_block_{l,r}_act` (8)

**Openalyx analytic splits with no alyx post-min5 counterpart (145;** excluding legacy / `combined_`* / `d_with_controls_*` / `_old` / junk**):**

- `act_block_only`; `act_block_stim_{l,r}_duringchoice_{l_f1,r_f2}` (4)
- `block_choice_{l,r}` (+ `_intertrial`); `block_{concordant,discordant}` (+ `_duringchoice`)
- `block_duringstim_{l,r}_choice_{l_f1,r_f2}` (+ contrasts 0/0.0625/0.125/0.25/1.0); `block_stim_{l,r}_{all,choice_*,duringchoice_*}` (+ contrasts)
- `choice_block_{l,r}`; `choice_duringfback`; `choice_duringstim` (+ `_l`/`_r`); `choice_duringstim_{l,r}_block_{l,r}` (**no** `_act`); `choice_intertrial`; `choice_stim_{l,r}` (+ `_block_{l,r}`, **no** `_act`)
- `concordant_duringchoice`
- `stim_{0,0.0625,0.125,0.25,1.0}`; `stim_block_{l,r}` (+ `_act`); `stim_choice_{l,r}` (+ `_block_{l,r}` / `_act` / `_short` / `_short_act`); `stim_duringchoice` (+ `_l`/`_r` / `_block_{l,r}` / `_act`); `stim_duringfback`; `stim_intertrial`

Alyx also has post-min5 files **not** on openalyx (e.g. `act_block_duringstim_{l,r}`, contrast `act_block_*_0.`*, bayes variants, `block_duringstim_choice_*_0.0`) — those are alyx-only, not openalyx gaps.

**2026-07-27c — plain label-shuffle re-run for the 8 choice L–R act splits:** `bash scripts/submit_goal2_choice_shuffle_sharded.sh` → `$ONE_CACHE_DIR/manifold/res/{split}.npy` (plain names, min5). This removes the caveat that those 8 splits were compared against a pre-min5 openalyx baseline. Local copy is alyx `manifold/res/new/` (mtime 2026-08-14); analysis in **2026-08-14** below.

---



## Harris unique-null for `act_block` prior L–R (2026-07-27e)

Same `--session-shuffle-null` / `{split}_harris_unique` scheme as choice L–R, but for `act_block_*` the transplanted labels are **priors**, not choices. Recipient conditioning is **stim × choice** (± contrast); the f1/f2 in the name is implied by that pair (no separate feedback filter). The donor is re-filtered to the same stratum, with unique-null sampling.


|                           | choice L–R Harris | act_block Harris              |
| ------------------------- | ----------------- | ----------------------------- |
| Distance                  | choice L vs R     | prior L vs R                  |
| Conditioning / `elig_idx` | stim × prior      | stim × choice (± c)           |
| Null labels               | donor choices     | donor priors (act/bayes/true) |


The prior-type routing fix that this exposed (`act_block_*` was silently falling back to true block) is documented in [prior definitions](prior_definitions.md).

**Donor bank:** the rebuild includes `contrast_left` / `contrast_right` (needed for contrast splits). Old banks still work for non-contrast `act_block`.

**Presets** (`run_goal2_splits.py`):

- `act_block_harris_all` — **9** splits: 4 duringstim choice×f + 4 duringchoice + `act_block_only` (no stim/choice stratum). **Not** the unconditioned `act_block_duringstim_{l,r}` (those are `act_block_harris_unsplit`; see **2026-08-14c**).
- `act_block_only` — ITI only (`PRESET=act_block_only`; does not CLEAR the other 8)
- `act_block_duringstim` / `act_block_duringchoice`
- `act_block_harris_unsplit` / `act_block_unsplit_duringstim` / `act_block_unsplit_duringchoice`
- also `goal3_duringstim_act` / `goal3_duringchoice_act` with `--session-shuffle-null`

Disk names: `{split}_harris_unique.npy`; plain shuffle `{split}.npy` is untouched.

```bash
bash scripts/submit_goal2_act_block_harris_sharded.sh
# PRESET=act_block_duringstim …   # subset
# PRESET=goal3_duringstim_act …   # contrast-expanded
python scripts/smoke_act_block_harris_null.py
```

`act_block_only` smoke: ITI window `[0.4, −0.1]`, act prior, no stim/choice stratum, parity OK.

---



## 2026-08-14 — min5 shuffle + act_block Harris unique (local `res/new`)

Same combine / BH-FDR / amp×sig tables as 07-27 (`plot_choice_null_comparison_table.py`, α=0.01, `p_mean`). Data: alyx `manifold/res/new/`. Script now takes `--family act_block` and `--force-combine-shuffle`.

```bash
python scripts/plot_choice_null_comparison_table.py \
  --openalyx-res ~/Downloads/ONE/alyx.internationalbrainlab.org/manifold/res/new \
  --arm-res ~/Downloads/ONE/alyx.internationalbrainlab.org/manifold/res/new \
  --arm-tag harris_unique --force-combine-shuffle --alpha 0.01 \
  --out-prefix table_choice_harris_unique_vs_min5shuffle

python scripts/plot_choice_null_comparison_table.py --family act_block \
  --openalyx-res ~/Downloads/ONE/alyx.internationalbrainlab.org/manifold/res/new \
  --shuffle-res-duringchoice ~/Downloads/ONE/openalyx.internationalbrainlab.org/manifold/res \
  --arm-res ~/Downloads/ONE/alyx.internationalbrainlab.org/manifold/res/new \
  --arm-tag harris_unique --force-combine --alpha 0.01
```



### Choice L–R: Harris unique vs min5 shuffle

8 act splits present (plain `{split}.npy` + `{split}_harris_unique.npy`). Shuffle coverage ~198–200 regions / 49–55 k cells, `n_null=2000`. Harris unique coverage matches (~197–200 regions / 49–55 k cells); per-split unique pools still ragged (incongruent medians ~700–800, congruent ~1300–1520).


| arm                                 | duringstim | duringchoice | median p (stim / choice) |
| ----------------------------------- | ---------- | ------------ | ------------------------ |
| shuffle (openalyx, pre-min5; 07-27) | 46         | 84           | 0.071 / 0.019            |
| **shuffle (alyx min5)**             | **88**     | **122**      | **0.011 / 0.0015**       |
| harris unique (`_harris_unique`)    | **21**     | **58**       | **0.263 / 0.092**        |


Versus min5 shuffle (lost / gained / kept; n=207 regions present in both):


| epoch        | shuffle | harris | lost | gained | kept |
| ------------ | ------- | ------ | ---- | ------ | ---- |
| duringstim   | 88      | **21** | 67   | 0      | 21   |
| duringchoice | 122     | **58** | 65   | 1      | 57   |


Harris unique FDR counts are **unchanged** vs 07-27 (21 / 58). The min5 shuffle baseline is **more liberal** than the old openalyx shuffle (88 / 122 vs 46 / 84), so Harris looks even stricter relative to shuffle. All 21 duringstim Harris hits were already shuffle hits (gained=0).

Plots / CSV: `meta/table_choice_harris_unique_vs_min5shuffle_{,duringchoice_}p_mean_c_0.01.png`; `meta/table_choice_harris_unique_vs_min5shuffle_p_mean_c_0.01.csv`. Does not overwrite the 07-27 openalyx-baseline table.

### act_block prior L–R: Harris unique vs shuffle

8 of 9 `act_block_harris_all` splits were in `res/new` as `*_harris_unique` on 08-14; `act_block_only` Harris unique arrived **2026-08-18**. Shuffle `act_block_only` arrived 2026-08-14d (plain `{split}.npy`, n_null=2000). Shuffle for the 8: duringstim = alyx min5 (07-14); duringchoice = openalyx pre-min5 (no alyx counterpart; same caveat as 07-27b).

**f2 Harris skip** (error-trial / short-stratum donors), analogous to the simulation split collapse:


| split                           | shuffle nreg / cells  | Harris nreg / cells   | cells kept |
| ------------------------------- | --------------------- | --------------------- | ---------- |
| duringstim f1 (`*_choice_*_f1`) | 207–208 / 62.5–62.6 k | 207–208 / 62.0–62.3 k | ~99 %      |
| duringstim f2 (`*_choice_*_f2`) | 200–201 / 55.2–56.0 k | 181–188 / 37.4–42.3 k | 68–75 %    |
| duringchoice f1                 | 207–208 / 62.5–62.7 k | 207–208 / 62.1–62.4 k | ~99 %      |
| duringchoice f2                 | 200–202 / 55.5–56.3 k | 180–188 / 37.7–42.5 k | 68–76 %    |


f2 unique pools: min 9, median ~425 (f1 median ~1800–1910). Combined product-MC still has `n_null≈2000`.


| arm               | duringstim | duringchoice | median uncorr p (stim / choice)              |
| ----------------- | ---------- | ------------ | -------------------------------------------- |
| shuffle           | **42**     | **42**       | 0.119 / 0.089 (`p_mean_c` med 0.236 / 0.176) |
| **harris unique** | **0**      | **0**        | **0.725 / 0.695**                            |


Versus shuffle (n=208 regions in both): duringstim 42→0 (lost 42, gained 0, kept 0); duringchoice 42→0 (lost 42, gained 0, kept 0). Shuffle duringstim 42 matches the 07-14b alyx all-contrast table.

Uncorrected: Harris duringstim 2 regions at `p_mean≤0.01` (RN, FOTU) and 3 at ≤0.05; duringchoice 1 at ≤0.01 (FOTU) and 5 at ≤0.05. Not enough for BH at α=0.01 **or** 0.05 (need ~30 at the p-floor). Observed amps are not gone (harris/shuffle `amp_euc` median ratio 0.71; e.g. GRN 2.28 vs 3.10) — the Harris null is wider, so the same curves are no longer significant.

Plots / CSV: `meta/table_act_block_harris_unique_vs_shuffle_{,duringchoice_}p_mean_c_0.01.png`; `meta/table_act_block_harris_unique_vs_shuffle_p_mean_c_0.01.csv`.

**Interpretation:** for prior L–R, Harris unique is far stricter than shuffle — it removes the entire FDR map. That is a larger correction than for choice L–R (21/58 still survive). Caveats: f2 insertions are thinned by donor-stratum length; duringchoice shuffle is still pre-min5 openalyx; `act_block_only` Harris unique was not in this copy (shuffle only: **2026-08-14d**).

**2026-08-14b — 0 FDR is not the f2-stratum skip.** Recombine in memory: f1-only / f2-only / all-4, same product-MC + [BH@0.01](mailto:BH@0.01).

Shuffle prior signal is almost entirely **f1** (correct choice). f2 shuffle alone is already null (0 FDR; 13 / 4 uncorr duringstim / duringchoice). Including f2 in the four-split **lowers** shuffle FDR (duringstim 89 → 42; duringchoice 112 → 42) — f2 dilutes, it does not carry the hits.


| combine            | duringstim shuffle FDR | duringstim Harris FDR | duringchoice shuffle FDR | duringchoice Harris FDR |
| ------------------ | ---------------------- | --------------------- | ------------------------ | ----------------------- |
| f1 only (2 splits) | **89**                 | **0** (4 uncorr)      | **112**                  | **0** (4 uncorr)        |
| f2 only            | 0                      | 0                     | 0                        | 0                       |
| all 4              | 42                     | 0 (2 uncorr)          | 42                       | 0 (1 uncorr)            |


On f1, Harris coverage is ~99 % and unique pools are large (median U ~1800–1910). Observed `amp_euc` matches shuffle (median ratio **1.00**); Harris null amplitudes are only modestly wider (ratio ~1.18–1.19). That SNR drop (stim 0.41 → 0.20; choice 0.36 → 0.14) is enough for 0 FDR. Among the 42 four-split shuffle hits, 35/42 still have all four Harris splits — whole-region f2 dropout is rare; 2–3/42 have Harris f1 uncorr p ≤ 0.01.

So the blank Harris map is the **structured prior null on f1**, not missing error-trial insertions. f2 skip is real (68–75 % cells) and further shrinks four-split observed amps (ratio ~0.75), but dropping f2 does not restore significance.

---



## Downstream: SC regtype tables

The stim/choice region-type table depends on which choice null feeds the choice L–R amplitudes, so it was regenerated per arm. Region-type definitions and the `mixed` target set live in [variance partition](variance_partition_mixed_regions.md). Excl-sticky choice regtype lives in [sticky / end-of-session trial exclusion](sticky_end_of_session_exclusion.md).

**2026-07-27d — Harris unique choice.** Stim from openalyx; choice L–R from alyx `res/new` `*_harris_unique` combined fours.

```bash
python scripts/plot_stimchoice_regtype_excl_sticky.py --alpha 0.01 \
  --choice-res ~/Downloads/ONE/alyx.internationalbrainlab.org/manifold/res/new \
  --choice-suffix _harris_unique --tag harris_unique
```

Output: `alyx.../meta/table_stimchoice_act_regtype_harris_unique_p_mean_c_0.01.png` (+ `.csv`). Counts @α=0.01: duringchoice — integrator 42, move 16; duringstim — stim 7, early 25, integrator 14, move 14.

---



## Simulation analog (Stage B prior distance, 2026-08-13d)

`simulate_recovery.py --harris-unique-null` is the act_block Harris unique-null
for **model** S/I/M prior distance: recipient neural `b` frozen per simulated
session; null labels are other-session subjective priors in the same
stim×choice (split) or stim-side (unsplit) stratum; unique patterns only;
40 extra donor sessions at `seed+10007`. Details and numbers:
[retinal then joint](retinal_then_joint_fitting.md) 2026-08-13d.

Unsplit (U=100): regular S stays null; sensory S stays significant. Split
f1/f2 unique pool collapses on incongruent f2 cells — same short-stratum
failure mode as real-data Harris skip.

**2026-08-14 — longer sessions fix f2.** Same analog at `--blocks-per-session 40`,
`--nrand 2000`, `--harris-n-extra-donors 80`. All splits unique=2000, 40/40 kept
(including f2). Combined p is now at real-data null resolution; the unsplit
conclusion is unchanged (regular S p=0.63, sensory S p=0). Numbers:
[retinal then joint](retinal_then_joint_fitting.md) 2026-08-14.

**Do not re-run this on the laptop.** The 40-block × 80-donor session cache was
~12 GB; the whole `session_cache/` hit 42 GB and was wiped after the campaign.
Future Harris unique-null / long-session / nrand=2000 / extra-donor jobs go on
**ORCD**. See [simulation infrastructure](simulation_infrastructure.md) 2026-08-14.

---



## 2026-08-14c — Harris unique unsplit prior tests (real data)

Real-data analog of the model `--unsplit-prior` tests in
[retinal then joint](retinal_then_joint_fitting.md) 2026-08-13c/d: **no f1/f2**.
Stim-aligned prior L–R is stratified by **stim side only**; movement-aligned
prior L–R is stratified by **choice side only**.


| split                             | align                 | window      | stratum         | distance     |
| --------------------------------- | --------------------- | ----------- | --------------- | ------------ |
| `act_block_duringstim_l` / `_r`   | `stimOn_times`        | `[0, 0.15]` | stim L or R     | prior L vs R |
| `act_block_duringchoice_l` / `_r` | `firstMovement_times` | `[0.15, 0]` | choice +1 or −1 | prior L vs R |


Harris path (`_get_d_vars_block_harris`) uses `_act_block_conditioning_spec`
(explicit name cases **before** the `'choice_l' in name` substring parse —
`'choice_l'` is a substring of `duringchoice_l`). Shuffle of the new
duringchoice splits has a matching `get_d_vars` branch. Combine is 2-split
(`--family act_block_unsplit`).

**Presets** (`run_goal2_splits.py`): `act_block_harris_unsplit` (4),
`act_block_unsplit_duringstim` (2), `act_block_unsplit_duringchoice` (2).

**ORCD**

```bash
bash scripts/submit_act_block_unsplit_orcd.sh
# Harris only: bash scripts/submit_goal2_act_block_harris_unsplit_sharded.sh
# PRESET=act_block_unsplit_duringstim …
# PRESET=act_block_unsplit_duringchoice …
python scripts/smoke_act_block_harris_null.py
```

Outputs: `$ONE_CACHE_DIR/manifold/res/{split}_harris_unique.npy`. The ORCD
wrapper `submit_act_block_unsplit_orcd.sh` exports `NRAND=1000`. The sharded
script default remains 2000 if called directly without `NRAND`.
`SESSION_SHUFFLE_NULL=1`, rebuild donors, `CLEAR_STREAM` only `*_harris_unique`
for these four. Wrapper job prefix `g2ahu`.

Duringstim unsplit **shuffle** (`act_block_duringstim_{l,r}.npy`) already
exists on alyx `res/new` (Jul 14). Duringchoice unsplit shuffle does **not**.
If a Harris-vs-shuffle table is needed, submit **only** the new pair (do not
`CLEAR_STREAM` the existing duringstim shuffle):

```bash
PRESET=act_block_unsplit_duringchoice bash scripts/submit_goal2_choice_shuffle_sharded.sh
```

Plot after both arms exist:

```bash
python scripts/plot_choice_null_comparison_table.py --family act_block_unsplit \
  --openalyx-res $ONE_CACHE_DIR/manifold/res/new \
  --shuffle-res-duringchoice $ONE_CACHE_DIR/manifold/res \
  --arm-res $ONE_CACHE_DIR/manifold/res \
  --arm-tag harris_unique --force-combine --alpha 0.01
```

Not added: true-block / bayes `duringchoice_l/r`.

---



## 2026-08-17 — unsplit Harris unique in local `res/new`

Copied into alyx `manifold/res/new/` (mtime 2026-08-16 10:53–10:54): the four
`act_block_harris_unsplit` Harris unique files.


| split                                    | regions / cells | n_null min/med/max    |
| ---------------------------------------- | --------------- | --------------------- |
| `act_block_duringstim_l_harris_unique`   | 209 / 62,580    | 141 / 1947 / **2000** |
| `act_block_duringstim_r_harris_unique`   | 207 / 62,619    | 114 / 2000 / 2000     |
| `act_block_duringchoice_l_harris_unique` | 207 / 62,376    | 119 / 2000 / 2000     |
| `act_block_duringchoice_r_harris_unique` | 208 / 62,556    | 98 / 2000 / 2000      |


Unique pools top out at 2000 (p-floor ≈ 1/2001), so this copy was **nrand=2000**,
not the wrapper default 1000. Coverage matches shuffle duringstim (~99 % cells).
**Duringchoice unsplit shuffle is still missing** (`act_block_duringchoice_{l,r}.npy`).

2-split combine (`p_mean`, product-MC when U is ragged; combined `n_null` med 2000):


| arm                   | window               | uncorr ≤0.01 | uncorr ≤0.05 | BH-FDR @0.01 | BH-FDR @0.05 | median p | median amp |
| --------------------- | -------------------- | ------------ | ------------ | ------------ | ------------ | -------- | ---------- |
| shuffle (Jul 14 min5) | duringstim unsplit   | 134          | 153          | **126**      | 146          | 0.001    | 0.492      |
| **harris unique**     | duringstim unsplit   | 16           | 28           | **0**        | 9            | 0.388    | 0.485      |
| **harris unique**     | duringchoice unsplit | 2            | 6            | **0**        | **0**        | 0.585    | 0.429      |


Duringstim Harris vs shuffle (209 shared): [FDR@0.01](mailto:FDR@0.01) **126 → 0** (lost 126, gained 0).
Observed `amp_euc` median ratio **1.00** — same curves, wider Harris null.
[FDR@0.05](mailto:FDR@0.05) Harris duringstim hits (p-floor 0.0005 unless noted): IRN, FOTU, GRN,
CLA, RN, MRN, CP; VISa (0.001); SCm (0.0015). Duringchoice uncorr ≤0.01: FOTU, GRN.

Same qualitative result as split-conditioned act_block Harris unique (08-14 /
08-14b): **0 FDR @0.01**. Unsplit duringstim shuffle is *more* liberal than the
four-split (126 vs 42) because there is no f1/f2 thinning; Harris still wipes
the map at α=0.01. Duringchoice Harris is even quieter (0 FDR @0.05).

Plot vs shuffle duringchoice still blocked on the missing shuffle pair.

---



## 2026-08-17b — early-stim (80 ms) slice of duringstim prior L–R

Same combined `*_regde` as 08-14 (split f1/f2 four) and 08-17 (unsplit stim-side two).
Recompute `p_mean` / BH-FDR on the prefix `t ≤ 80 ms` of the 150 ms curves
(38 of 72 bins; last kept bin 78.2 ms on `linspace(0, 150, 72)`). This is a
curve slice, not a re-bin with `PRE_POST = [0, 0.08]`. Per-bin Euclidean
distance is independent of later bins. Script:
`python scripts/summarize_prior_earlystim.py --alpha 0.01`.

CSV: alyx `meta/table_act_block_earlystim_80ms_{summary,p_mean}.csv`.


| conditioning | null          | window               | uncorr ≤0.01 | uncorr ≤0.05 | FDR @0.01 | FDR @0.05 | median p | median amp |
| ------------ | ------------- | -------------------- | ------------ | ------------ | --------- | --------- | -------- | ---------- |
| split        | shuffle       | 150 ms               | 54           | 75           | **42**    | 56        | 0.119    | 1.751      |
| split        | shuffle       | **early-stim 80 ms** | 42           | 58           | **16**    | 42        | 0.217    | 1.321      |
| split        | Harris unique | 150 ms               | 2            | 3            | **0**     | 0         | 0.725    | 1.105      |
| split        | Harris unique | **early-stim 80 ms** | 1            | 3            | **0**     | 0         | 0.759    | 0.856      |
| unsplit      | shuffle       | 150 ms               | 134          | 153          | **126**   | 146       | 0.001    | 0.492      |
| unsplit      | shuffle       | **early-stim 80 ms** | 96           | 129          | **84**    | 120       | 0.014    | 0.351      |
| unsplit      | Harris unique | 150 ms               | 16           | 28           | **0**     | 9         | 0.388    | 0.485      |
| unsplit      | Harris unique | **early-stim 80 ms** | 6            | 21           | **0**     | **0**     | 0.504    | 0.351      |


150 ms rows reproduce 08-14 / 08-17. Restricting to 80 ms shrinks shuffle maps
(split 42→16, unsplit 126→84) but **does not flip the ordering**: unsplit
shuffle stays the liberal arm. Harris unique is still 0 FDR @0.01 for both;
the 9 unsplit Harris [FDR@0.05](mailto:FDR@0.05) hits at 150 ms (IRN, FOTU, GRN, CLA, RN, MRN,
CP, VISa, SCm) all drop out at 80 ms.

Early-stim split-shuffle [FDR@0.01](mailto:FDR@0.01) hits (16; all also unsplit hits): DCO, DTN,
FL, FOTU, GRN, IRN, LPO, MRN, MV, PGRN, PPN, RN, SCm, SPIV, SUV, VCO —
cerebellar / medullary / midbrain, not visual cortex. Unsplit shuffle still
has 84 [FDR@0.01](mailto:FDR@0.01) hits. Harris uncorr ≤0.01 at 80 ms: split FOTU only; unsplit
CP, FOTU, IRN, MRN, PRM, VISa — none survive BH.

So the sim-style “unsplit S goes null at 80 ms” does **not** happen on the
all-region BWM prior map. Early-stim makes the shuffle claim smaller but not
identified: split remains stricter than unsplit, and Harris still wipes both.

---



## 2026-08-14d — `act_block_only` shuffle in local `res/new`

Copied into alyx `manifold/res/new/` (mtime 2026-08-14 16:23):

- `act_block_only.npy` / `act_block_only_regde.npy` — **label shuffle** (plain names; `n_null=2000` every region; 208 regions)
- Harris unique: **2026-08-18**

ITI window `[0.4, −0.1]`, act prior, no stim/choice stratum. Not part of the 4-split duringstim / duringchoice combine.


| metric                | uncorr ≤0.01 | uncorr ≤0.05 | BH-FDR @0.01 | BH-FDR @0.05                   | median p |
| --------------------- | ------------ | ------------ | ------------ | ------------------------------ | -------- |
| `p_mean` (from regde) | 9            | 13           | **0**        | 5 (VISa, AIp, SSp-n, LSr, CLA) | 0.689    |
| `p_euc`               | 11           | 25           | **0**        | 10                             | 0.571    |


`amp_euc` median 0.172. Even unrestricted shuffle is already null at α=0.01 FDR on this ITI split — unlike the 42-hit four-split during-trial shuffle. Harris unique comparison: **2026-08-18**.

---

## 2026-08-18 — `act_block_only` Harris unique in local `res/new`

Copied into alyx `manifold/res/new/` (mtime 2026-08-18 15:50), after ORCD shards 1/2 OOM-restarted at 24G (`submit_goal2_act_block_only_harris_restart.sh`) and finalize:

- `act_block_only_harris_unique.npy` / `_regde.npy` (208 regions, 63,346 cells)
- `n_null` min/median/max **352 / 2000 / 2000** (nrand=2000 unique-null; some regions below the cap)
- shuffle pair from 08-14d unchanged

ITI `[0.4, −0.1]`, act prior, no stim/choice stratum. Single-split (not a 4- or 2-split combine). `p_mean` from `*_regde`; BH-FDR over 208 regions.

| arm | uncorr ≤0.01 | uncorr ≤0.05 | BH-FDR @0.01 | BH-FDR @0.05 | median p | median amp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| shuffle (08-14d) | 9 | 13 | **0** | 5 (AIp, CLA, LSr, SSp-n, VISa) | 0.689 | 0.172 |
| **harris unique** | 8 | 30 | **0** | **0** | 0.316 | 0.167 |

Paired `amp_euc` median ratio harris/shuffle **0.98** (208 shared). Shuffle `p_mean≤0.01` regions still have small Harris p (AIp 0, CLA/FOTU 0.001, SSp-n 0.002, VISa 0.006, PL 0.007) — none survive BH at 0.01 or 0.05.

Unlike during-trial act_block (Harris median p *larger* / stricter), ITI Harris median p is *smaller* than shuffle, but the 0.05 FDR map still goes 5 → 0. Same 0 FDR @0.01 as the other act_block Harris unique arms. The 9-split `act_block_harris_all` set is now complete in `res/new`.

---

## 2026-08-21 — explicit late stickiness on AK option 1

Option 1 (stratified pseudo + stationary AK θ) had **no** time-varying perseveration: late clumpiness only if the fake stim/block stream happened to be blocky. Real sessions are mildly clumpier mid-to-late on the **post-0.5 calendar** sequence ([sticky §8](sticky_end_of_session_exclusion.md)): mean_run 2.91 → 2.92 → 3.06 → **3.19** → 3.18, peak in Q4 not Q5, without a rate shift in accuracy or true-block alignment.

**Mechanism (copy-last mixture, not a new MCMC parameter).** After AK `.simulate` under fitted `[α, ζ, lapse±]`:

1. Drop pLeft=0.5, split the rest into five equal-count quintiles (same masks as `analyze_perseveration_counts._quintile_masks`).
2. Measure this draw's Q1 mean_run `μ0`. Target `μ_q = μ0 + (μ_BWM[q] − μ_BWM[0])` so Q1 stays at the kernel baseline and later quintiles pick up the empirical Δ.
3. Convert with the geometric relation `p_repeat = 1 − 1/μ`. Copy-last probability `ρ_q = 1.27 × (p_q − p_Q1) / (1 − p_Q1)`, `ρ_0 = 0`. The 1.27 gain offsets copy-last no-ops (already-repeating trials). Peak `ρ_Q4 ≈ 0.09–0.11`.
4. Walk the session: with probability `ρ[quintile(t)]`, replace choice `t` by the last valid ±1 choice. Recompute `feedbackType`. Act-prior strata for option 1 are then taken from these sticky choices.

Does **not** change fitted θ, the pseudo stim/block stream, or neural `b`. Default **off** so existing `_pseudo_strat` files stay stationary-α.

| Piece | Detail |
| ----- | ------ |
| Helper | `scripts/late_choice_stickiness.py` |
| Hook | `late_sticky=` on `synthetic_sessions_from_trials` / `make_synthetic_session` / `synthetic_choices_fixed_stim` |
| CLI | `--actkernel-late-sticky` (option 1 or 2) |
| Disk | `{split}_pseudo_strat_sticky.npy` (does not overwrite `_pseudo_strat`) |
| Tag | `null_scheme: synthetic_choice_pseudo_strat_sticky` |
| Smoke | `python scripts/test_late_stickiness.py` |

Markov check (p_repeat from μ=2.91, 80 sessions): Q1 mean_run unchanged; Q4−Q1 rises by ~+0.22–0.25 (target +0.28); block-align shift < 0.03. That is the intended “clumpiness not a rate shift.”

```bash
python scripts/test_late_stickiness.py
# BWM arm (not yet run):
python scripts/run_goal2_splits.py --preset choice_lr_session_null_all \
  --actkernel-choice-null --actkernel-late-sticky --nrand 200
```

**Not done (08-21):** a full BWM choice L–R null with this flag. **2026-08-22** wired the fitted+copy-last arm (below); FDR not yet run.

---

## 2026-08-22 — fitted θ + copy-last vs BWM quintiles; within-stratum null

Default demo θ `[0.2, 0.5, 0.05, 0.05]` is much stickier than the mouse (mean_run ~8 vs ~2.9) and near-chance. **Per-session MCMC** (what option 1 actually uses) is not: median fitted α=0.16, ζ=0.07.

80 BWM sessions, fixed real stim, MCMC 200 steps, copy-last targeting that session's real μ_q (`scripts/compare_ak_late_stickiness.py --fit-ak --n-sess 80`):

| | Q1 | Q2 | Q3 | Q4 | Q5 | Q4−Q1 | acc | block | lag-1 |
|---|---|---|---|---|---|---|---|---|---|
| mouse | 2.88 | 2.72 | 2.95 | 2.91 | 3.10 | +0.23 | 0.87 | 0.76 | 0.30 |
| fitted AK | 2.56 | 2.66 | 2.64 | 2.77 | 2.67 | +0.13 | 0.86 | 0.74 | 0.26 |
| **fitted + copy-last** | **2.93** | **2.71** | **3.12** | **2.85** | **3.17** | −0.11 | 0.82 | 0.73 | 0.30 |

Per-quintile (sticky − real) median is 0. Quintile MAE 0.58 → 0.38; closer in 69/80. Fitted-only matches rates; copy-last pins the quintile *levels*. Q4−Q1 is median of per-session deltas (this 80-session slice is a weaker late bump than the 459-session [sticky §8](sticky_end_of_session_exclusion.md) 2.91 → 3.19). CSV: `manifold/choice_epoch_diag/ak_late_sticky/fitted/`.

**Fit once per eid, not per split.** θ is a property of the session's behaviour, not of the neural split. `get_actkernel_choice_fit` already pickles under `manifold/actkernel_fits/{eid8}/`. Prefit all BWM sessions, then every choice / act_block split and every probe of that eid reuses the pickle:

```bash
python scripts/run_goal2_splits.py --prefit-actkernel
# ORCD: sbatch scripts/run_goal2_ak_prefit_slurm.sh
```

Refitting every split would redo the same ~20 s MCMC × 16 splits × n insertions.

**Within-stratum null (fixedstim + late-sticky).** Real `elig_idx` is the shuffle stratum; neural `b` stays put; only labels are AK+copy-last.

| Family | Shuffle stratum | Null labels |
|---|---|---|
| choice L–R | stim × prior | synthetic **choices** at real `elig_idx` |
| act_block prior L–R | stim × choice | synthetic **priors** (choices → analysis α=0.2 binary) at real `elig_idx` |

Disk: `{split}_pseudo_fixed_sticky.npy`. Stim-aligned + move-aligned presets: `choice_lr_ak_sticky` (8), `act_block_ak_sticky` (8).

```bash
# laptop smoke
python scripts/run_goal2_splits.py --preset choice_lr_ak_sticky_duringstim \
  --actkernel-null-mode fixedstim --actkernel-late-sticky --nrand 20
# ORCD both families:
bash scripts/submit_goal2_ak_sticky_orcd.sh
# or one family:
PRESET=choice_lr_ak_sticky NULL_SCHEME=pseudo_fixed_sticky \
  bash scripts/submit_goal2_choice_null_sharded.sh
PRESET=act_block_ak_sticky NULL_SCHEME=pseudo_fixed_sticky \
  bash scripts/submit_goal2_choice_null_sharded.sh
```

**Not done:** full-BWM FDR vs Harris unique / min5 shuffle.

---

## Open questions

1. **Primary null choice for choice L–R claims** — Harris (empirical sticky structure within stratum) vs AK stratified pseudo (BWM-like new world with matched bias context). Min5 shuffle (08-14) is more liberal than openalyx, which widens the Harris–shuffle gap but does not change Harris FDR counts.
2. **Why strat null amplitudes stay below Harris** at matched coverage.
3. **Fixed α vs per-session action-kernel fit** for act labels — see [prior definitions](prior_definitions.md).
4. **Drop-0.5 timing mismatch** between prior-distance and choice L–R families — see [prior definitions](prior_definitions.md).
5. **act_block Harris unique is a near-total null** (0 FDR @0.01). 08-14b: **not** the f2 donor-stratum skip. **2026-08-17 unsplit:** same 0 FDR @0.01. **2026-08-17b early-stim 80 ms:** shuffle FDR shrinks but unsplit stays liberal; Harris still 0 FDR @0.01. **2026-08-18 `act_block_only` ITI:** shuffle already 0 FDR @0.01 (5 at 0.05); Harris unique 0 FDR @0.01 **and** @0.05 (208 regions, 63k cells). Remaining question: is that the intended correction for block-autocorrelated priors, or is the Harris prior-transplant null too wide? Duringchoice unsplit **shuffle** still missing.
6. **AK + late stickiness** — 08-21 copy-last implemented. **08-22:** per-session MCMC + copy-last matches BWM quintile mean_run on 80 sessions (fitted-only slightly under-sticky). Wired as within-stratum (`fixedstim`) null for choice L–R **and** act_block (stim + move). Prefit θ once per eid. Full-BWM FDR not yet run.

