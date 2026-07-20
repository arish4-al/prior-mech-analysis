# Research journal — 2026-07-12

## Standing context (carry-forward from 2026-07-06)

Real-data pipeline only — see [prior journal](research_journal_2026-07-06.md) for Goals 2–3 (insertion cache, stream_pool, contrast-stratified during-trial splits) and ORCD sharding notes (07-12a–d). Do not carry forward generative-model / simulated fitting results.

### Real-data analysis stack

| Piece | Status / location |
|-------|-------------------|
| Driver | `block_analysis_allsplits.py` |
| Insertion cache | `manifold/insertion_cache/{eid_probe}.npy` (spikes + trials once per pid) |
| Multi-split | outer loop = insertions, inner = splits; `stream_pool` → `manifold/res/` |
| Contrast splits | `'{base}_{contrast}'` on duringstim / duringchoice (act + non-act), incl. 0% |
| Min trials | ≥5 per split side (`InsufficientTrials` → skip) |
| Sharding | `scripts/submit_goal2_*` / `submit_goal3_sharded.sh` (ORCD `mit_normal`) |
| Null (current) | label shuffle within condition (contrast-matched where applicable); choice L–R: `--synthetic-choice-null` or `--exclude-sticky-trials` (opt-in) |

---

## Today's goals

**Scope:** BWM / real recordings only (`block_analysis_allsplits.py` and related real-data scripts). No simulated sessions.

### Goal 1 — Bayesian (sensory) prior on sensory responses; conflict vs alignment with recent reward

On **real data**, test whether sensory-population prior effects hold when the prior is framed as a **Bayesian / sensory prior**, and stratify trials by whether the **current block prior conflicts** vs **aligns** with **recent reward history**.

Motivation: block identity and recent outcomes can pull in opposite directions; pooling them may mask or invent a sensory prior effect.

**Questions:**
1. Is there a significant prior effect in sensory responses under a Bayesian prior framing?
2. Does any effect concentrate on conflict trials, alignment trials, or both?
3. Does the conclusion change relative to the usual block-prior split (pooled across recent-reward history)?

### Goal 2 — Null that respects choice / prior autocorrelation and neural drift

**Revised problem:** Trial averaging does **not** by itself remove the issue. At each peri-event bin the choice/prior distance is a difference between condition-averaged trial responses. If the labels occur in long epochs and neural responses also drift or remain autocorrelated across trials, the two averages can sample different parts of the session. An unrestricted label shuffle mixes those epochs and can produce a null that is too narrow. Label autocorrelation **alone** is not sufficient: if trial responses were independent and stationary, label ordering would not matter once condition counts were fixed. The relevant failure mode is therefore **autocorrelated choice/prior labels interacting with across-trial neural nonstationarity**.

**Affected contrasts:**
- **Choice:** animals show runs of one-sided responses, so choice L–R tests under fixed stimulus/block need a temporally structured null.
- **Prior:** true block labels are explicitly block-autocorrelated. `block_only` already generates pseudo-blocks, but other prior-distance splits currently use unrestricted permutations and can destroy this structure. Action-kernel and Bayes-derived priors are also temporally structured.
- **Stimulus:** stimulus identities are randomized by the task, so a task-matched pseudo-session/shuffle remains appropriate; stimulus significance is not the main target of this correction.

**Important terminology / implementation correction:** the opt-in donor-window code added on 2026-07-13 is a **stim×block-matched donor surrogate**, not a literal Harris (2020) session-permutation test. It transplants a donor choice subsequence independently for each insertion/null draw. A Harris test instead permutes complete behavioral sessions against complete neural sessions, uses one coherent session mapping for every probe from the same `eid`, and evaluates an aggregate session-level statistic. Keep the current method labeled as a matched donor surrogate unless it is redesigned accordingly.

**Literal Harris session-permutation setup:**
1. Treat each unique `eid` as the exchangeable unit; all probes from that `eid` must share the same donor assignment.
2. Predefine a scalar association statistic for each region/session from the trial-resolved neural tensor and the complete behavioral history. Apply deterministic, permutation-independent trial trimming so paired sessions have compatible lengths.
3. The observed statistic is the aggregate over matched sessions, \(T_{\mathrm{obs}}=\sum_s A(X_s,B_s)\), with a fixed weighting rule so sessions with multiple probes are not implicitly over-weighted.
4. For each null draw, sample a bijection \(\pi\) of compatible sessions (preferably a derangement), pair \(X_s\) with the **complete** behavioral history \(B_{\pi(s)}\), and compute \(T_\pi=\sum_s A(X_s,B_{\pi(s)})\). Use the same \(\pi\) for all probes/regions in that draw.
5. Compute \(p=(1+\#\{T_\pi\ge T_{\mathrm{obs}}\})/(1+N_{\mathrm{perm}})\), then apply the planned across-region correction.

**Conditional-choice caveat:** transplanting only donor choices after filtering each donor to a recipient-like stim×block stratum is not literal session permutation, and skipping intervening trials changes the relevant lag structure. Moreover, a Harris permutation of complete sessions directly tests neural–behavioral independence, whereas “choice beyond stimulus and block” is a partial-association question.

**Implemented primary structured null (2026-07-18):** sticky psychometric **synthetic-choice** null for `choice_stim*` / `choice_duringstim*`:

- Keep neural tensor `b` and stim×block eligibility fixed (real stim / contrast / prior covariates frozen — including act/bayes overwrites; do **not** recompute act from synthetic choices).
- Per `eid`, fit  
  \(\operatorname{logit} P(\mathrm{choice}_t=\mathrm{L})=\beta_0+\beta_s s_t+\beta_c c_t+\beta_p(p_t-0.5)+\beta_{\mathrm{lag}} a_{t-1}\)  
  (MLE; empirical cell-rate × lag-1 stickiness fallback).
- Null draws: sequentially sample full-session synthetic ±1 choices; split eligible trials by those labels.
- CLI: `--synthetic-choice-null` in `scripts/run_goal2_splits.py` → `null_scheme: synthetic_choice_sticky`.
- Donor-window `--session-shuffle-null` remains a comparison surrogate; label shuffle stays the default until width/p-value validation.

Not the same as `brainbox.generate_pseudo_session` (i.i.d. within contrast×side×block cells — destroys stickiness) and not a literal Harris session permutation.

**Validation:** compare unrestricted label shuffle, the matched donor surrogate, and synthetic-choice using both null width and region-level p-values. The choice-run analysis establishes that label structure exists; it does not by itself show that neural-distance significance changes.

---

## Notes / results

### 2026-07-12e — Action-kernel prior (existing) + Bayes-optimal prior (implemented)

**Action kernel** (`action_kernel_priors`, α=0.2) — already in `get_d_vars` when `'act' in split`:

1. Drop true-block 0.5 trials (`probabilityLeft != 0.5`).
2. EMA over **choices**: `prior ← α·[choice>0] + (1−α)·prior` (choice>0 = left).
3. Binarize to **0.8 / 0.2** (`≥0.5 → 0.8`) and **overwrite** `probabilityLeft` (true block kept in `true_priors`).
4. Downstream L-vs-R-prior splits use that overwritten column unchanged.

**Bayes-optimal prior** (Findling et al. Nature 2025 SI §1.1.1) — new `bayesian_priors`, wired the same way when `'bayes' in split`:

1. Infer continuous **P(stim left | past stimulus sides)** under the IBL generative model (τ=60, γ=0.8, block len 20–100).
2. Computed on the **full** trial list **before** dropping 0.5 blocks (needs full stim history); then same 0.5 filter + overwrite `probabilityLeft` with 0.8/0.2; continuous trace in `bayes_priors`, true block in `true_priors`.
3. Split names mirror act: `bayes_block_*` (duringstim / duringchoice / stim_l / stim_r / only) + Goal-3 contrast registration (`GOAL3_BAYES_*`).
4. Presets in `scripts/run_goal2_splits.py`: `stimOn_times_bayes`, `goal3_duringstim_bayes`, `goal3_duringchoice_bayes`, `goal3_bayes_all`.

**Smoke (priors only):** 80 left stims → P(left)≈0.77 (→γ); 80 right → ≈0.23; mid-session switch flips binary 0.8→0.2.

**Next:** run real-data `bayes_block_*` vs `act_block_*` / true-block; then conflict vs alignment (true block vs recent reward / vs Bayes).

### 2026-07-12f — Rename stim-side splits: were ITI, now during-stim

**Bug:** bare `*block_stim_l` / `*block_stim_r` (no choice×feedback) lacked `durings`/`duringc` in the name, so align setup assigned the **intertrial** window `[0.4, −0.1]` despite stimOn alignment.

**Fix:** rename → `*block_duringstim_l` / `*block_duringstim_r` (`block_`, `act_block_`, `bayes_block_`). Name substring `durings` now selects post-stim `[0, 0.15]`. Scripts/presets updated (`stimOn_times_act`, `stimOn_times_bayes`, submitters).

Any cached outputs under the old names are **invalid for during-stim claims** and must be re-run.

### 2026-07-12g — Lower mem for Goal 2 sharded submit (8 splits)

**Estimate:** stream_pool peak RSS ≈ **1.5–2.5 GB** / shard (nrand=2000; 07-10b). Finalize merges 4 shard accs → modestly above one shard.

**Defaults** (`submit_goal2_stimOn_act_sharded.sh` + worker `#SBATCH`):

| | was | now |
|--|-----|-----|
| MEM_SHARD | 12G | **6G** |
| MEM_FIN | 16G | **10G** |
| Concurrent (8×4 shards) | 384G | **192G** |

Override if OOM: `MEM_SHARD=8G MEM_FIN=12G bash scripts/submit_goal2_stimOn_act_sharded.sh`.

### 2026-07-12h — Bayes stim L–R splits in cached pipeline (not choicestim)

Ported choicestim’s **stim L vs R** contrasts into `block_analysis_allsplits.py` with Bayes-optimal prior labels. Do **not** revive `choicestim_analysis.py` for new runs (legacy docstring only).

| Timeframe | Splits | Window |
|-----------|--------|--------|
| `stim_duringstim_bayes` | `stim_choice_{r,l}_block_{r,l}_bayes` | `[0, 0.15]` — stim L/R, fixed choice + Bayes prior |
| `stim_duringstim1_bayes` | `stim_block_{l,r}_bayes` | `[0, 0.08]` — stim L/R, fixed Bayes prior only |

Parity with choicestim act: **no** true-block 0.5 drop before prior overwrite for these names.

```bash
python scripts/run_goal2_splits.py --preset stim_duringstim_bayes --list-splits
python scripts/run_goal2_splits.py --preset stim_duringstim1_bayes
python scripts/run_goal2_splits.py --preset stim_lr_bayes_all
```

### 2026-07-13 — Goal 2 choice L–R: Harris session-permutation null (opt-in)

**Problem (journal Goal 2):** unrestricted label shuffle on choice contrasts destroys choice autocorrelation → inflated significance.

**Implemented** in `block_analysis_allsplits.py` for `choice_stim*` / `choice_duringstim*` — **opt-in only** (`session_shuffle_null=False` by default → label shuffle):

1. Filter stim (± block/prior) only; bin eligible trials in time order.
2. True distance: split by this session’s choices (L vs R).
3. Null (when `--session-shuffle-null`): transplant a contiguous choice window from another BWM **eid** (`manifold/choice_donors.npy`); resample if a side has <5 trials; circular-shift own choices if no long-enough donor.
4. Stim contrasts unchanged.

**Assumption check** (`scripts/analyze_choice_epochs.py`; n=459; `bwm_include` + RT/NaN mask; drop pLeft=0.5):

**2026-07-13b — true matched to stim×block.** Stickiness for **true and null** is scored within stim×block cells. Null shuffles choices within strata then uses the same metric. Fairer vs ephys cell conditioning.

**How strata / tertiles are pooled**

1. After dropping pLeft=0.5 and non-±1 choices, label each trial by stim side (L/R from contrasts) × block side (L if pLeft=0.8, R if 0.2) — either **true** block or **act**-kernel (α=0.2 → 0.8/0.2).
2. **Within a stratum:** take that cell’s choices in session temporal order (intervening other-stratum trials skipped). Compute run lengths on that subsequence.
3. **Across strata (session score):** concatenate all within-stratum run lengths and take their mean (`mean_run`). Same idea for lag1 (pool consecutive pairs within each stratum, then correlate). Not a mean of per-stratum means — longer strata contribute more runs/pairs.
4. **Tertiles:** split the post-0.5 trial sequence into early/mid/late thirds **first**, then apply the same within-stratum pooling **inside each tertile slice** (strata labels restricted to that slice). Null: shuffle within those slice-local strata.

**Overall (all tertiles combined — full post-0.5 session; median mean_run):**

| | true | null | p&lt;0.01 | p&lt;0.05 |
|--|------|------|---------|---------|
| stim×true-block | 4.84 | 4.44 | 24.0% | 39.0% |
| stim×act | 4.80 | 4.48 | 12.0% | 26.1% |

(lag1 session-level p&lt;0.01: stim×true-block 25%, stim×act 14%.)

**Tertiles — mean_run only** (median true/null; frac sessions with p&lt;0.01 / p&lt;0.05):

stim×true-block:

| tertile | true | null | p&lt;0.01 | p&lt;0.05 |
|---------|------|------|---------|---------|
| early | 4.43 | 4.20 | 5.9% | 17.2% |
| mid | 4.59 | 4.30 | 4.8% | 13.1% |
| late | 4.61 | 4.20 | 9.8% | 22.0% |

stim×act:

| tertile | true | null | p&lt;0.01 | p&lt;0.05 |
|---------|------|------|---------|---------|
| early | 4.29 | 4.19 | 3.9% | 10.2% |
| mid | 4.47 | 4.31 | 3.7% | 10.0% |
| late | 4.50 | 4.20 | 5.2% | 13.9% |

Late>early (stim×true-block mean_run) in **56%** of sessions (Δ +0.20). Excess stickiness vs stratified null is mild; late has the highest sig rates but still modest.

```bash
python scripts/analyze_choice_epochs.py --cache-dir $ONE_CACHE_DIR --nrand 200
```

Goal 3 contrast retention + gain/offset tables: see [07-06 journal](research_journal_2026-07-06.md) §2026-07-14.

### 2026-07-18 — Synthetic-choice sticky null (choice L–R)

Implemented preferred Goal-2 structured null for stratified choice contrasts:

| Piece | Detail |
|-------|--------|
| Model | \(\operatorname{logit}P(L)=\beta_0+\beta_s s+\beta_c c+\beta_p(p-0.5)+\beta_{\mathrm{lag}}a_{t-1}\) |
| Fit | per `eid` (cached across probes); MLE with empirical+stickiness fallback |
| Null | sample full-session synthetic choices; apply to fixed eligible `b` |
| Tag | `null_scheme: synthetic_choice_sticky` |
| CLI | `--synthetic-choice-null` (`scripts/run_goal2_splits.py`) |
| Smoke | `python scripts/smoke_synthetic_choice_null.py` |

Donor `--session-shuffle-null` kept for comparison; default still label shuffle until null-width validation.

### 2026-07-20 — Late + perseveration trial exclusion (choice L–R sensitivity)

Alternative to structured nulls: remove likely drift×stickiness trials, then use **label shuffle within stim×block** (existing default null on block-conditioned choice splits).

**Drop** (union):
1. Last 20% of the session (temporal order)
2. **Tail** of same-choice runs of length ≥10 that are poorly explained by non-0 contrast stim (block ignored): keep the first 9 trials of the run; drop from trial 10 onward within the run. A run is poorly explained if among |contrast|>0 trials, any stim side ≠ choice — or the run has no non-0 contrast trials

Well-explained long runs (all non-0 stims match the perseverated choice) are kept in full.

| Piece | Detail |
|-------|--------|
| API | `apply_sticky_trial_exclusion` / `--exclude-sticky-trials` |
| Outputs | `manifold/res_excl_sticky/` (avoids overwriting main `res/`) |
| Tag | `null_scheme: label_shuffle_excl_sticky` + `trial_exclusion` stats |
| Presets | `choice_lr_excl_sticky_{act,true,bayes}` |
| Submit | `bash scripts/submit_goal2_choice_excl_sticky_sharded.sh` |
| Smoke | `python scripts/smoke_excl_sticky_trials.py` |

```bash
# Local / single split
python scripts/run_goal2_splits.py --preset choice_lr_excl_sticky_act \
  --exclude-sticky-trials --nrand 2000

# ORCD sharded
bash scripts/submit_goal2_choice_excl_sticky_sharded.sh
PRESET=choice_lr_excl_sticky_true bash scripts/submit_goal2_choice_excl_sticky_sharded.sh
```

Use as a robustness arm next to synthetic-choice nulls; not a complete replacement for a temporally structured null.

### 2026-07-20b — Perseveration counts over all BWM sessions

**Tail-of-run exclusion** (keep first `min_run-1=9` of each poorly explained run).

Ran `scripts/analyze_perseveration_counts.py` on `bwm_tables/trials.pqt` (459 sessions, `bwm_include=True`, min_run=10, late_frac=0.2).

| Metric | median | mean | IQR |
|--------|-------:|-----:|-----|
| # perseveration **tail** trials | 26 | 33.9 | [11, 49] |
| frac perseveration | 0.066 | 0.076 | [0.031, 0.107] |
| # dropped (late ∪ pers) | 100 | 111 | [74, 136] |
| frac dropped | 0.248 | 0.257 | [0.221, 0.283] |
| # kept | 291 | 317 | [230, 376] |

23/459 sessions (5%) have zero perseveration-tail trials. vs whole-run (earlier): median pers 66→26, frac dropped 0.33→0.25.

Plot + CSV: `manifold/choice_epoch_diag/perseveration_exclusion_distributions.png`, `…_by_session.csv`.

```bash
python scripts/analyze_perseveration_counts.py \
  --cache-dir $HOME/Downloads/ONE/openalyx.internationalbrainlab.org
```


