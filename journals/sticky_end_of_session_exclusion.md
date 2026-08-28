# Sticky / end-of-session trial exclusion

**Scope:** dropping late-session and perseveration-tail trials before label-shuffle neural distances — originally as a robustness arm against drift × sticky-label false positives, now also as a possible SNR filter if the end of session is zoned-out noise. Choice L–R and act-prior L–R (excl-sticky) have both been run.

**Status:** choice L–R FDR **widens**. Act-prior L–R under the same trim **does not** go Harris-null. Canonical **4-split duringstim** (08-25, both f2 files in) matches duringchoice 4-split: **expands @0.05 / shrinks @0.01**. Split-conditioned **f1** and **unsplit** duringstim still **shrink** at both α. Late behaviour: not zoned-out, not more block-aligned. Post-0.5 quintiles (08-21): stickiness peaks in **Q4**, not the last 20 % (Q5); only RT is a Q5 outlier.

Sources: dated entries in [structured nulls](structured_nulls_choice_lr.md) 07-20 (exclusion API), 07-20b (BWM counts), 07-21 (choice FDR comparison), 07-21b (SC regtype); choice-epoch stickiness 07-13 / 07-13b **in this file**; 2026-08-18 / 08-18b–g / **08-21** (quintiles) / **08-21 AK late-sticky**; **08-23** (prior L–R submitter); **08-24** (prior L–R FDR, incomplete 4-split); **08-25** (canonical 4-split duringstim).

---



## Goal / two competing stories

**Original hypothesis (2026-07-20):** choice (and prior) labels are temporally autocorrelated, and neural responses drift across the session. Condition averages can then sample different parts of the session, so an unrestricted label shuffle is too narrow and can produce **false-positive** L–R distances. Dropping the trials that most couple stickiness to late-session drift, then shuffling as usual, should **shrink** the significant set if those hits were confounds.

**Drop rule (union):**

1. Last **20 %** of the session (temporal order).
2. **Tail** of same-choice runs of length ≥10 that are poorly explained by non-zero-contrast stimuli (block ignored): keep the first 9 trials of the run, drop from trial 10 onward. A run is poorly explained if, among |contrast| > 0 trials, any stim side ≠ choice — or if the run has no non-zero-contrast trials.

Well-explained long runs (all non-zero stimuli match the perseverated choice) are kept in full.

**What actually happened on choice L–R (2026-07-21):** exclusion does **not** shrink the map. At FDR 0.05, duringstim 71 → 95 and duringchoice 106 → 123 (net +17 / +17). At FDR 0.01, 46 → 69 and 84 → 100. Most shuffle hits are kept; many new regions become significant.

**Reinterpretation to test (2026-08-18):** if later trials are mostly zoned-out noisy activity (worse performance, lapses, unstructured firing), dropping them raises SNR of a **real** choice signal. That would explain a *wider* significant set, and would not be a reason to distrust the remaining hits. The original drift-confound story is not ruled out — both could be true in different regions — but the net FDR change goes the wrong way for “exclusion removes false positives.”

If the zoned-out story is right, the same exclusion should be tried on **prior L–R** splits (split-conditioned `act_block_`* and unsplit `act_block_duringstim_{l,r}` / `duringchoice_{l,r}`). Those maps are currently shuffle-liberal and Harris-null ([structured nulls](structured_nulls_choice_lr.md) 08-14 / 08-17 / 08-18); a late-trial SNR filter is a different intervention from Harris.

This arm is **not** a complete replacement for a temporally structured null (Harris / AK). It is a data-trim then the usual shuffle.

---



## Implementation


| Piece     | Detail                                                                                                                                 |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| API       | `apply_sticky_trial_exclusion` / `--exclude-sticky-trials`                                                                             |
| Mask      | `sticky_trial_exclusion_mask` → `late                                                                                                  |
| Constants | `STICKY_LATE_FRAC = 0.20`, `STICKY_MIN_RUN = 10`                                                                                       |
| Outputs   | `manifold/res_excl_sticky/` (avoids overwriting the main `res/`)                                                                       |
| Tag       | `null_scheme: label_shuffle_excl_sticky` + `trial_exclusion` stats                                                                     |
| Presets   | choice: `choice_lr_excl_sticky_{act,true,bayes}`; prior: `act_block_excl_sticky` (8 split-cond + 2 unsplit duringstim) and subsets     |
| Submit    | choice: `bash scripts/submit_goal2_choice_excl_sticky_sharded.sh`; prior: `bash scripts/submit_goal2_act_block_excl_sticky_sharded.sh` |
| Smoke     | `python scripts/smoke_excl_sticky_trials.py`                                                                                           |
| Counts    | `python scripts/analyze_perseveration_counts.py`                                                                                       |


The flag is applied in `get_d_vars` **before** stim×prior (or stim×choice) stratification, so it is not choice-L–R-specific. Prior L–R reuses it (`act_block_excl_sticky`*) and writes to the same `res_excl_sticky/` folder; the prior submitter CLEARs only the listed `act_block_`* names.

```bash
python scripts/run_goal2_splits.py --preset choice_lr_excl_sticky_act \
  --exclude-sticky-trials --nrand 2000
bash scripts/submit_goal2_choice_excl_sticky_sharded.sh
PRESET=choice_lr_excl_sticky_true bash scripts/submit_goal2_choice_excl_sticky_sharded.sh
bash scripts/submit_goal2_act_block_excl_sticky_sharded.sh
```

`info` already records the overlap the next count analysis needs:

- `n_late`, `n_perseveration`, `n_drop`, `n_keep`
- `n_drop_late_only`, `n_drop_pers_only`, `n_drop_both`

`analyze_perseveration_counts.py` writes those columns to CSV but the 07-20b summary did **not** report the overlap or “what fraction of sticky tails sit in the last 20 %.”

---



## What is already measured



### Choice-epoch stickiness (2026-07-13 / 07-13b)

Moved here from [structured nulls](structured_nulls_choice_lr.md). `scripts/analyze_choice_epochs.py`; n=459 sessions; `bwm_include` + RT/NaN mask; drop pLeft=0.5.

**2026-07-13b — true matched to stim×block.** Stickiness for **both true and null** is scored within stim×block cells; the null shuffles choices within strata and uses the same metric. This is fairer relative to how the ephys distances condition trials.

**How strata / tertiles are pooled**

1. After dropping pLeft=0.5 and non-±1 choices, label each trial by stim side (L/R from contrasts) × block side (L if pLeft=0.8, R if 0.2) — either **true** block or **act**-kernel (α=0.2 → 0.8/0.2).
2. **Within a stratum:** take that cell's choices in session temporal order (intervening other-stratum trials skipped) and compute run lengths on that subsequence.
3. **Across strata (session score):** concatenate all within-stratum run lengths and take their mean (`mean_run`). Same for lag-1 (pool consecutive pairs within each stratum, then correlate). This is *not* a mean of per-stratum means — longer strata contribute more runs/pairs.
4. **Tertiles:** split the post-0.5 trial sequence into early/mid/late thirds **first**, then apply the same within-stratum pooling inside each tertile slice. The null shuffles within those slice-local strata.

**Overall (all tertiles combined; median** `mean_run`**):**


|                 | true | null | p<0.01 | p<0.05 |
| --------------- | ---- | ---- | ------ | ------ |
| stim×true-block | 4.84 | 4.44 | 24.0 % | 39.0 % |
| stim×act        | 4.80 | 4.48 | 12.0 % | 26.1 % |


(lag-1 session-level p<0.01: stim×true-block 25 %, stim×act 14 %.)

**Tertiles —** `mean_run` (median true/null; fraction of sessions with p<0.01 / p<0.05):

stim×true-block:


| tertile | true | null | p<0.01 | p<0.05 |
| ------- | ---- | ---- | ------ | ------ |
| early   | 4.43 | 4.20 | 5.9 %  | 17.2 % |
| mid     | 4.59 | 4.30 | 4.8 %  | 13.1 % |
| late    | 4.61 | 4.20 | 9.8 %  | 22.0 % |


stim×act:


| tertile | true | null | p<0.01 | p<0.05 |
| ------- | ---- | ---- | ------ | ------ |
| early   | 4.29 | 4.19 | 3.9 %  | 10.2 % |
| mid     | 4.47 | 4.31 | 3.7 %  | 10.0 % |
| late    | 4.50 | 4.20 | 5.2 %  | 13.9 % |


Late > early (stim×true-block `mean_run`) in **56 %** of sessions (Δ +0.20). Excess stickiness versus the stratified null is mild; the late tertile has the highest significance rates but they remain modest.

```bash
python scripts/analyze_choice_epochs.py --cache-dir $ONE_CACHE_DIR --nrand 200
```

This shows late-session **choice structure** exists. It is serial correlation within stim×block, not a change in P(choice = block) or in accuracy — see **2026-08-18d**.

### Perseveration counts over all BWM sessions (2026-07-20b)

`scripts/analyze_perseveration_counts.py` on `bwm_tables/trials.pqt` (459 sessions, `bwm_include=True`, min_run=10, late_frac=0.2), tail-of-run exclusion:


| Metric                          | median | mean  | IQR            |
| ------------------------------- | ------ | ----- | -------------- |
| # perseveration **tail** trials | 26     | 33.9  | [11, 49]       |
| frac perseveration              | 0.066  | 0.076 | [0.031, 0.107] |
| # dropped (late ∪ pers)         | 100    | 111   | [74, 136]      |
| frac dropped                    | 0.248  | 0.257 | [0.221, 0.283] |
| # kept                          | 291    | 317   | [230, 376]     |


23/459 sessions (5 %) have zero perseveration-tail trials. Versus whole-run exclusion (earlier version): median perseveration 66 → 26, fraction dropped 0.33 → 0.25.

Plot + CSV: `manifold/choice_epoch_diag/perseveration_exclusion_distributions.png`, `…_by_session.csv`.

Overlap of the two drop rules (late vs pers tails): **2026-08-18b**.

```bash
python scripts/analyze_perseveration_counts.py --cache-dir $HOME/Downloads/ONE/openalyx.internationalbrainlab.org
```



### Choice L–R: excl-sticky vs shuffle (2026-07-21)

Four-split combined choice sensitivity (sum `*_regde` → `p_mean` → BH-FDR → amp × sig table):


| Arm     | Cache                            | Null                                                       |
| ------- | -------------------------------- | ---------------------------------------------------------- |
| shuffle | openalyx `manifold/res/`         | label shuffle within stim×block                            |
| excl    | alyx `manifold/res_excl_sticky/` | late 20 % ∪ perseveration-tail drop, then the same shuffle |


```bash
python scripts/plot_choice_excl_sticky_comparison_table.py --alpha 0.05
```

[FDR@0.05](mailto:FDR@0.05) among 205 regions present in both caches:


| epoch        | shuffle | excl | lost (sig→ns) | gained (ns→sig) | kept |
| ------------ | ------- | ---- | ------------- | --------------- | ---- |
| duringstim   | 71      | 95   | 9             | 33              | 62   |
| duringchoice | 106     | 123  | 16            | 33              | 90   |


[FDR@0.01](mailto:FDR@0.01) among the same 205 regions (2026-08-13; existing `p_mean_c` CSV, BH-FDR threshold only):


| epoch        | shuffle | excl | lost (sig→ns) | gained (ns→sig) | kept |
| ------------ | ------- | ---- | ------------- | --------------- | ---- |
| duringstim   | 46      | 69   | 8             | 31              | 38   |
| duringchoice | 84      | 100  | 13            | 29              | 71   |


Exclusion does **not** shrink the significant set (net +17 during-choice at 0.05; net +16 at 0.01). The regional pattern is largely preserved (90/106 shuffle hits kept at 0.05; 71/84 at 0.01).

Plots (alyx `meta/`; **no column headers** on PNGs):

- `table_choice_excl_sticky_vs_shuffle_duringchoice_p_mean_c_0.05.png` (2 columns): shuffle vs excl, duringchoice.
- `table_choice_excl_sticky_vs_shuffle_p_mean_c_0.05.png` (4 columns): duringstim shuffle / excl, duringchoice shuffle / excl.

Cell colour = normalized `amp_euc` if FDR `p_mean_c` ≤ 0.05, else blank. CSV companion: `table_choice_excl_sticky_vs_shuffle_p_mean_c_0.05.csv`.

**Caveat:** this comparison used the **pre-min5 openalyx** shuffle baseline. The 2026-08-14 min5 shuffle for the same 8 act splits is more liberal (duringstim 88, duringchoice 122). Re-plotting excl-sticky against min5 shuffle has not been done; the *direction* (excl ≥ shuffle) is unlikely to reverse, but the lost/gained counts would change.

### SC regtype with excl-sticky choice (2026-07-21b)

Same layout as openalyx `table_stimchoice_act_regtype_p_mean_c_0.01.png`, but choice L–R amplitudes from `res_excl_sticky` (stim / short / stim1 still openalyx).

```bash
python scripts/plot_stimchoice_regtype_excl_sticky.py --alpha 0.01
```

Counts @α=0.01 (excl-sticky choice): duringchoice — integrator 75, move 22; duringstim — stim 3, early 14, integrator 48, move 22. (Harris unique choice, for comparison: integrator 42 / 14 and move 16 / 14.)

---



## 2026-08-18b — overlap of the two drop rules, and late-session behaviour

Same 459 BWM sessions as 07-20b (`bwm_tables/trials.pqt`, `bwm_include=True`, min_run=10, late_frac=0.2). Extended `scripts/analyze_perseveration_counts.py` (overlap summary + early/late/pers accuracy and RT). Exclusion counts match 07-20b (median 26 pers-tail trials, 100 dropped, 0.248 frac dropped).

```bash
python scripts/analyze_perseveration_counts.py \
  --cache-dir $HOME/Downloads/ONE/openalyx.internationalbrainlab.org
```

CSV + plots: `manifold/choice_epoch_diag/perseveration_exclusion_by_session.csv`, `sticky_late_overlap.png`, `sticky_late_performance.png`, `sticky_late_performance_deltas.png`, `sticky_late_overlap_performance_summary.txt`.

### 1. Sticky tails are not the last 20 %


| Metric                                   | median    | mean  | IQR            |
| ---------------------------------------- | --------- | ----- | -------------- |
| # late trials                            | 79        | 86.1  | [63, 104]      |
| # pers tail                              | 26        | 33.9  | [11, 49]       |
| # late-only (not pers)                   | 72        | 77.4  | [55, 92.5]     |
| # pers-only (not late)                   | 18        | 25.3  | [7, 37.5]      |
| # both (pers ∩ late)                     | 5         | 8.63  | [0, 12.5]      |
| # dropped (union)                        | 100       | 111   | [74, 136]      |
| frac of pers tails that sit in last 20 % | **0.194** | 0.256 | [0, 0.385]     |
| extra drop (pers-only / n_trials)        | 0.048     | 0.056 | [0.020, 0.082] |


Among 436 sessions with any pers tail: **2.5 %** entirely inside the last 20 %, **72.2 %** mixed, **25.2 %** entirely *outside* the last 20 % (110 sessions). 23/459 have zero pers tails.

The median 19 % of sticky tails falling in the last 20 % is what you would get if pers tails were spread uniformly over the session — not a late-session pile-up. Late trials are 81 % of the union (median); the extra pers-only cut is 19 % of the union (median 18 trials). The two rules do **different jobs**. Dropping “late” is not a proxy for dropping sticky, and vice versa.

### 2. Last 20 % is slower, not zoned-out inaccurate

Paired per session vs the early 80 %. Wilcoxon two-sided on late−early (or pers−early). “Frac worse” = lower accuracy / higher RT / higher no-choice.


| Metric                  | early median | late median | Δ median   | frac worse | p         | pers median | pers Δ | pers p  |
| ----------------------- | ------------ | ----------- | ---------- | ---------- | --------- | ----------- | ------ | ------- |
| accuracy (all c)        | 0.857        | 0.868       | **+0.005** | 44 %       | 0.046     | 0.838       | −0.012 | 0.0024  |
| accuracy (0 % contrast) | 0.585        | 0.615       | **+0.040** | 40 %       | 6.5×10⁻⁴  | 0.80        | +0.17  | 5×10⁻¹⁸ |
| accuracy (              | c            | ≥0.25)      | 0.975      | 0.970      | −0.0001   | 51 %        | 0.0027 | 1.00    |
| accuracy (c=1)          | 1.00         | 1.00        | 0          | 27 %       | 0.029     | 1.00        | 0      | 0.041   |
| median RT (s)           | 0.163        | 0.190       | **+0.021** | **81 %**   | 4.6×10⁻⁴⁸ | 0.164       | −0.003 | 0.54    |
| no-choice fraction      | 0            | 0           | 0          | 8 %        | —         | 0           | 0      | —       |


No-choice is essentially absent after `bwm_include` (mean 0.02 % early, 0.13 % late). Not a usable lapse metric here.

**Late window (including pers):** overall and 0-contrast accuracy slightly *up*, not down. High-contrast accuracy dips by a tiny amount (mean 0.965 → 0.955). The high-contrast dip **vanishes** once pers tails are taken out of the late window (late-not-pers vs early, c≥0.25 p=0.25; c=1 p=0.27). RT slowing does **not** vanish: late-not-pers median RT +22 ms, 83 % of sessions slower, p=3.5×10⁻⁵¹. Late RT IQR also widens (median 0.078 → 0.143 s).

**Pers tails:** not slower than early. 0-contrast accuracy is *higher* (median 0.80). High-contrast accuracy is a biased readout — the mask *defines* poorly explained runs as those with a stim≠choice among c>0 (or no non-zero trials), so mean high-c acc is pulled down by construction (0.933 vs 0.965) even though the median sits at 1.0. Elevated 0-contrast acc is also expected if the animal is repeating the block-majority choice (0 % sides still follow 80/20).

**Against the zoned-out story:** the last 20 % is not a pile of lapses or chance-level choices. Mice still hit ~97 % on easy trials and slightly *better* on 0-contrast. What changes is speed and RT variability. That can still be a different neural state (fatigue, caution, drift) without being “noise to drop for SNR.” The choice-L–R FDR expansion after exclusion is therefore **not** explained by removing failing behavioural trials. Remaining candidates: slower/more variable late neural responses, n_L/n_R or trial-count effects, or a real choice signal that late trials were diluting for some other reason than inaccurate choices.

### 3. Last 20 % is not more block-aligned (2026-08-18c)

Same 459 sessions. Among valid ±1 choices on **biased** true blocks (`probabilityLeft` 0.8/0.2; drop 0.5), P(choice = block side). High = |c|≥0.25; low = |c|<0.25 (includes 0). Plot: `sticky_late_block_match.png`.


| Slice                     | all c                      | high                       | low                        |
| ------------------------- | -------------------------- | -------------------------- | -------------------------- |
| early 80 % median         | 0.767                      | 0.802                      | 0.735                      |
| last 20 % median          | 0.770                      | 0.800                      | 0.745                      |
| Δ median (late−early)     | +0.003                     | −0.003                     | +0.007                     |
| frac sessions late higher | 52 %                       | 47 %                       | 53 %                       |
| Wilcoxon p                | **0.74**                   | 0.099                      | 0.37                       |
| late not pers Δ / p       | −0.009 / **1.3×10⁻⁴**      | −0.017 / **1.6×10⁻⁸**      | −0.009 / 0.015             |
| pers tail median / Δ / p  | **0.97** / +0.19 / 3×10⁻⁵⁸ | **1.00** / +0.17 / 7×10⁻⁵⁷ | **0.97** / +0.21 / 2×10⁻⁶⁰ |


The last 20 % as a whole does **not** choose the true-block side more than the early 80 %, at high contrast, low contrast, or overall. The small 0-contrast accuracy bump in 08-18b is not a general increase in block following.

Sticky tails *are* strongly block-aligned (median ~0.97, 93–95 % of sessions higher than early). That is expected: a long same-choice run in a biased block usually sits on the majority side, including on incongruent stim (the “poorly explained” definition). Removing those tails from the late window leaves late **less** block-aligned than early (high-c p=1.6×10⁻⁸). So late-session behaviour is not “more prior-driven”; the pers rule is what captures block-side perseveration, and it is not concentrated in the last 20 % (08-18b).

### 4. Late stickiness without a rate shift (2026-08-18d)

The 07-13b late bump and the 08-18b/c rate results are not in conflict. They measure different things.

`mean_run` **is clumpiness, not a choice rate.** After dropping 0.5, each trial is in a stim×block cell. Run length is computed on that cell's choices in time, **skipping other cells**, then pooled. The stratified null shuffles choices *within* the cell (and within the tertile slice), so it already matches that cell's n_L/n_R. Excess `mean_run` is extra serial correlation: the next time the same stim×block appears, the choice is more likely to match the last visit. Block-align and accuracy are **marginals** — P(choice = block side) and P(correct) — which do not have to move when the same mix is delivered in longer bursts (LRLRLR vs LLLLLRR, same 3L/2R).

Late vs early (stim×true-block): median `mean_run` 4.43 → 4.61 (Δ +0.20, only **56 %** of sessions). The slice-local null stays 4.20, so the excess over shuffle grows 0.23 → 0.41 and p<0.01 rates go 5.9 % → 9.8 %. That is a mild autocorrelation bump in a minority tail of sessions, not a new late-session policy.

**That also matches 08-18b/c:**

- P(choice = block) is unchanged in the last 20 % (overall / high / low). The mix is the same; the order is slightly clumpier.
- Accuracy is not down. Clumping the same correct/error mix does not change % correct. (If late stickiness were “always take the block side,” block-align and 0-contrast accuracy would both jump; only pers *tails* do that, and they are not concentrated late.)
- Poorly-explained runs ≥10 (the exclusion mask) are a different, rarer object. A +0.20 shift around mean_run ≈ 4.5 is extra length on ordinary (often stim-congruent) within-cell runs, not a pile-up of mismatch tails in the last 20 %. Median 19 % of those tails sit in the last 20 % — uniform, not late-loaded.

**What is happening late:** slightly more stereotyped *repeats of the same (stim, block) → choice mapping when that cell recurs*, plus slower/more variable RT, without shifting how often that mapping is the block side or the correct side. Calendar-adjacent perseveration is not even what 07-13b scores. For neural distances this still matters — clumped labels can sample a drifting session unevenly even when P(L) is stable — which is why a structured null is still the right tool, but it is not evidence that late trials are zoned out or more prior-driven.

### 5. Calendar-order sequences (no stim×block skip) (2026-08-18e)

Same 459 sessions, last 20 % vs early 80 % as in 08-18b/c. Run lengths on valid ±1 choices in **session temporal order** (no skipping other stim×block cells). Plot: `sticky_late_calendar_stickiness.png`.


| Metric                        | early | late  | Δ median  | frac late higher | p         |
| ----------------------------- | ----- | ----- | --------- | ---------------- | --------- |
| mean run (all valid)          | 2.86  | 3.21  | **+0.33** | 65 %             | 7.8×10⁻²¹ |
| lag-1 corr (all valid)        | 0.283 | 0.327 | +0.029    | 56 %             | 7.9×10⁻⁵  |
| frac in runs ≥5               | 0.52  | 0.59  | +0.064    | 67 %             | 5.2×10⁻¹⁴ |
| frac in runs ≥10              | 0.247 | 0.289 | +0.036    | 56 %             | 5.9×10⁻⁶  |
| mean run (**drop pLeft=0.5**) | 3.03  | 3.21  | **+0.14** | 56 %             | 3.0×10⁻⁶  |
| lag-1 (drop 0.5)              | 0.319 | 0.327 | −0.007    | 49 %             | **0.44**  |
| frac ≥5 (drop 0.5)            | 0.565 | 0.59  | +0.018    | 55 %             | 0.058     |
| frac ≥10 (drop 0.5)           | 0.295 | 0.289 | −0.007    | 49 %             | 0.77      |


`p` = paired Wilcoxon signed-rank on the 459 session-level (late − early) values (one cohort test, not a per-session p). `frac late higher` = fraction of sessions with late > early (a sign count; **not** “% of sessions with a significant late increase”). Ties are neither higher nor lower, so 56 % higher does not imply 44 % lower.

Raw calendar sequences **are** stickier in the last 20 %, but a large part of that is the unbiased 0.5 block sitting in the early window (shorter runs at session start). After dropping 0.5, **mean run is still significantly longer late** (3.03 → 3.21, Δ +0.14, Wilcoxon p=3.0×10⁻⁶, n=459). The p-value is small because the cohort is large and the shift is consistent in rank, not because the effect is large: +0.14 on a baseline of ~3 is ~5 %, and only 56 % of sessions go late>early (same sign count as the 07-13b tertile comparison). Lag-1 and the fraction of trials in runs ≥10 do **not** move (p=0.44 / 0.77). So the late window is detectably clumpier in mean run length, without more trial-to-trial autocorrelation or more mass in long runs. Calendar mean runs are shorter than within-stratum ones (∼3 vs ∼4.5) because stim/block switches break calendar runs; 07-13b skips those switches.

Dropping 0.5 removes the *dramatic* calendar effect (Δ +0.33 → +0.14; lag-1 and run≥10 go from significant to n.s.). It does not make mean run itself null.

### 6. Overall early 80 % vs last 20 %, drop pLeft=0.5 (2026-08-18f)

Same windows as 08-18e: last 20 % of the full `bwm_include` session vs the complementary early window, then drop the unbiased 0.5 block inside each window (it sits in early). Session medians; p = paired Wilcoxon on late−early; frac late>early = sign count, not per-session significance.

**n trials (459 sessions):** session length median 395 (IQR 312–515). Last 20 % median **79** trials (mean 86, IQR 63–104, range 26–283). Drop-0.5 does not change the late count (0.5 is at the start); early 80 % is 316 trials before drop-0.5 and **256** after.


| Metric                                  | early 80 % | last 20 % | Δ median   | frac late > early | p             |
| --------------------------------------- | ---------- | --------- | ---------- | ----------------- | ------------- |
| block alignment P(choice=true block)    | 0.767      | 0.770     | +0.003     | 52 %              | 0.74          |
| act-kernel alignment P(choice=AK prior) | 0.706      | 0.722     | **+0.016** | 58 %              | **1.7×10⁻⁴**  |
| median RT (s)                           | 0.166      | 0.190     | **+0.020** | **80 %**          | **1.3×10⁻⁴⁴** |
| P(correct)                              | 0.863      | 0.868     | +0.003     | 52 %              | 0.92          |
| mean run length                         | 3.03       | 3.21      | **+0.14**  | 56 %              | **3.0×10⁻⁶**  |
| lag-1 choice corr                       | 0.319      | 0.327     | −0.007     | 49 %              | 0.44          |
| frac in runs ≥5                         | 0.565      | 0.590     | +0.018     | 55 %              | 0.058         |
| frac in runs ≥10                        | 0.295      | 0.289     | −0.007     | 49 %              | 0.77          |


AK is `action_kernel_priors` α=0.2 on the **full** session choice sequence (0.5 trials kept for the EMA), then alignment scored on the same drop-0.5 early/late windows as the true-block row.

On biased-block trials, late differs from early in **speed**, **mean run length**, and **following the animal's own action kernel** — not in accuracy or true-block alignment. Higher AK alignment with longer mean runs is partly mechanical (the kernel is an EMA of recent choices). True-block alignment staying flat means they are not locking onto the task block more; they are slightly more consistent with their own recent choice history.

### 7. Same table, last 10 % vs early 90 % (2026-08-18g)

Same drop-0.5 rule; late window is the last **10 %** of the full `bwm_include` session. CSV: `manifold/choice_epoch_diag/late10/`. Δ median is median(late−early), which can differ from median(late)−median(early).

**n trials:** last 10 % median **40** (mean 43, IQR 32–52, range 13–142). Drop-0.5 again leaves the late count unchanged; early 90 % is 355 before drop-0.5 and **294** after.


| Metric                                  | early 90 % | last 10 % | Δ median   | frac late > early | p             |
| --------------------------------------- | ---------- | --------- | ---------- | ----------------- | ------------- |
| block alignment P(choice=true block)    | 0.770      | 0.762     | −0.009     | 47 %              | **0.047**     |
| act-kernel alignment P(choice=AK prior) | 0.710      | 0.714     | +0.002     | 52 %              | 0.36          |
| median RT (s)                           | 0.168      | 0.203     | **+0.027** | **81 %**          | **8.6×10⁻⁴⁷** |
| P(correct)                              | 0.864      | 0.864     | −0.004     | 47 %              | 0.098         |
| mean run length                         | 3.08       | 3.08      | −0.066     | 45 %              | 0.31          |
| lag-1 choice corr                       | 0.330      | 0.238     | **−0.091** | 35 %              | **1.5×10⁻¹⁶** |
| frac in runs ≥5                         | 0.568      | 0.558     | −0.005     | 49 %              | 0.16          |
| frac in runs ≥10                        | 0.298      | 0.289     | −0.054     | 44 %              | **0.0067**    |


The last 10 % is **slower**, not stickier. Mean run does not increase; lag-1 and long-run mass go **down**. True-block alignment is slightly lower (p=0.047); AK alignment is n.s. The 20 % mean-run / AK bump (08-18f) is therefore not coming from the very end of the session — if anything the final 10 % is less autocorrelated (and still slower).

### 8. Five equal 20 % windows of the post-0.5 sequence (2026-08-21)

Drop pLeft=0.5 first, then split the remaining trials into five equal-count quintiles (Q1 = earliest, **Q5 = last 20 % of biased-block trials**). This is not last 20 % of the full session: the old last-20 % window mixed the end of Q4 with Q5. 459/459 sessions had ≥10 trials per quintile. Median **67** trials per quintile (IQR ~50–90).

Q5 is compared to the **distribution of Q1–Q4**, not to a pooled 80 %. Δ = median(Q5 − median(Q1–Q4) within session). `frac Q5>max` = fraction of sessions where Q5 exceeds every earlier quintile. Plot: `sticky_post05_quintiles.png`.


| Metric               | Q1    | Q2    | Q3        | Q4        | Q5        | Δ vs med(Q1–4) | frac Q5>med | frac Q5>max | p             |
| -------------------- | ----- | ----- | --------- | --------- | --------- | -------------- | ----------- | ----------- | ------------- |
| block alignment      | 0.754 | 0.767 | 0.776     | **0.785** | 0.768     | 0              | 49 %        | 19 %        | 0.31          |
| act-kernel alignment | 0.700 | 0.695 | 0.713     | **0.727** | 0.714     | +0.010         | 54 %        | 23 %        | 0.052         |
| median RT (s)        | 0.160 | 0.163 | 0.167     | 0.176     | **0.191** | **+0.022**     | **81 %**    | **58 %**    | **4.4×10⁻⁴⁵** |
| P(correct)           | 0.861 | 0.868 | 0.875     | 0.866     | 0.864     | −0.005         | 47 %        | 20 %        | 0.076         |
| mean run length      | 2.91  | 2.92  | 3.06      | **3.19**  | 3.18      | **+0.11**      | 56 %        | 25 %        | **4.5×10⁻⁵**  |
| lag-1 corr           | 0.267 | 0.267 | **0.320** | 0.304     | 0.304     | +0.018         | 54 %        | 22 %        | 0.15          |
| frac in runs ≥5      | 0.534 | 0.538 | 0.568     | **0.585** | 0.583     | +0.025         | 54 %        | 22 %        | 0.026         |
| frac in runs ≥10     | 0.259 | 0.238 | 0.280     | **0.289** | 0.275     | 0              | 50 %        | 22 %        | 0.082         |


Paired Q5 vs Q4 (the previous 20 % window): block alignment **lower** (p=0.004); mean run **tied** (p=0.91); RT still slower (p=4.6×10⁻²³). Mean run and AK alignment **peak in Q4**, not Q5. RT is the only metric where Q5 is an outlier relative to all earlier quintiles (58 % of sessions Q5 > max(Q1–Q4)).

That is why last 20 % of the *full* session looked stickier (it mixed Q4+Q5) while last 10 % looked less sticky (it sits inside Q5). Stickiness, if anything, is a **mid-to-late** effect (Q3–Q4), not an end-of-session pile-up. Slowing continues through Q5.

### 9. Explicit late stickiness on AK synthetic choices (2026-08-21 / 08-22)

Stationary AK has no time-varying perseveration. Copy-last after simulate matches each draw's post-0.5 quintile mean_run to **that session's** real μ_q. Default **off**. Details: [structured nulls](structured_nulls_choice_lr.md) 2026-08-21 / **08-22**.

**2026-08-22 — fitted θ + copy-last vs this table's metric (80 sessions).** Demo ζ=0.5 is far stickier than the mouse. Per-session MCMC (median α=0.16, ζ=0.07) matches accuracy / block-align and is slightly *under*-sticky. Adding copy-last pins quintile mean_run (MAE 0.58 → 0.38; per-quintile Δ median 0). Small acc cost (0.87 → 0.82). This 80-session slice has a weaker Q4−Q1 than the 459-session row above. Intended neural arm is option 1 (stratified pseudo), not fixedstim. A fixedstim BWM job was run by mistake; those FDR numbers are deleted. See [structured nulls](structured_nulls_choice_lr.md) 2026-08-22.

---



## 2026-08-23 — prior L–R excl-sticky wired (not run)

Local alyx `manifold/res/new` and `res_excl_sticky` checked before adding jobs. **No** `act_block_`* excl-sticky files exist. `res_excl_sticky` is still choice L–R only (`choice_{duringstim,stim}_*_act`).

Plain shuffle already in `res/new` (baselines, not this arm):


| Split set                   | In `res/new` `{split}.npy`                                                          |
| --------------------------- | ----------------------------------------------------------------------------------- |
| split-cond duringstim (4)   | yes (~70–74 MB)                                                                     |
| split-cond duringchoice (4) | **no** (Harris unique + `_pseudo_`* only; shuffle still the openalyx pre-min5 copy) |
| unsplit duringstim (2)      | yes (~74 MB)                                                                        |


Added presets in `scripts/run_goal2_splits.py` and submitter `scripts/submit_goal2_act_block_excl_sticky_sharded.sh`. Default = 10 splits (8 split-cond + unsplit duringstim). Writes `manifold/res_excl_sticky/{split}.npy`. `CLEAR_STREAM=1` removes only those `act_block_*` names (refuses any non-`act_block_` split). Not submitted.

```bash
bash scripts/submit_goal2_act_block_excl_sticky_sharded.sh
PRESET=act_block_excl_sticky_duringstim bash scripts/submit_goal2_act_block_excl_sticky_sharded.sh
PRESET=act_block_excl_sticky_duringchoice bash scripts/submit_goal2_act_block_excl_sticky_sharded.sh
PRESET=act_block_excl_sticky_unsplit_duringstim bash scripts/submit_goal2_act_block_excl_sticky_sharded.sh
```

---



## 2026-08-24 — act-prior L–R excl-sticky FDR

Local alyx `manifold/res_excl_sticky/` (mtime 2026-08-24 21:55–21:58). Label shuffle after late 20 % ∪ pers-tail drop; `n_null=2000` (p-floor 0.0005). Same combine as 08-14: sum `*_regde` → `p_mean` → BH-FDR. Shuffle duringstim / unsplit = alyx `res/new` min5; duringchoice 4-split shuffle = **openalyx pre-min5** (no alyx counterpart, same caveat as 08-14). Harris unique from `res/new`.

**Missing file:** `act_block_duringstim_l_choice_r_f2` did not land, so the canonical **4-split duringstim** excl combine is blocked. Duringchoice 4, duringstim f1 (both files), one duringstim f2, and unsplit duringstim are complete.

### Coverage (pooled `{split}.npy`)


| split                           | shuffle nreg / cells  | excl nreg / cells     | cells kept  |
| ------------------------------- | --------------------- | --------------------- | ----------- |
| duringstim f1 (`*_choice_*_f1`) | 207–208 / 62.5–62.6 k | 205–207 / 60.5–60.7 k | ~97 %       |
| duringstim f2 `r_choice_l_f2`   | 200 / 55.2 k          | 172 / 31.2 k          | **57 %**    |
| duringstim f2 `l_choice_r_f2`   | 201 / 56.0 k          | **MISSING**           | —           |
| duringchoice f1                 | 207–208 / 62.5–62.7 k | 206–207 / 60.6–60.9 k | ~97 %       |
| duringchoice f2                 | 200–202 / 55.5–56.3 k | 176–184 / 31.6–34.2 k | **57–61 %** |
| unsplit duringstim `{l,r}`      | 207–209 / 62.8 k      | 205–207 / 61.1 k      | ~97 %       |


f1 / unsplit lose a few percent of cells (late∪pers trim + `min_trials_per_side`). f2 is already sparse; exclusion cuts it almost in half. `trial_exclusion` stats are not stored on the pooled region dicts.

### FDR (`p_mean`, BH)

Shared-region lost/gained vs the shuffle combine of the **same split list**. Harris unique of that list is 0 FDR @0.01 in every split-conditioned row (matches 08-14 / 08-14b / 08-17).


| combine                               | α    | shuffle FDR | excl FDR | lost | gained | kept | median p sh / excl |
| ------------------------------------- | ---- | ----------- | -------- | ---- | ------ | ---- | ------------------ |
| duringchoice **4**                    | 0.05 | 52          | **71**   | 18   | 37     | 34   | 0.089 / 0.078      |
| duringchoice **4**                    | 0.01 | 42          | **33**   | 22   | 13     | 20   | 0.089 / 0.078      |
| duringstim **f1 only**                | 0.05 | 129         | **106**  | 30   | 7      | 99   | 0.009 / 0.021      |
| duringstim **f1 only**                | 0.01 | 89          | **72**   | 29   | 12     | 60   | 0.009 / 0.021      |
| duringstim **3** (no `l_choice_r_f2`) | 0.05 | 69          | 67       | 29   | 27     | 40   | 0.085 / 0.071      |
| duringstim **3** (no `l_choice_r_f2`) | 0.01 | 45          | 37       | 25   | 17     | 20   | 0.085 / 0.071      |
| unsplit duringstim                    | 0.05 | 146         | **127**  | 27   | 9      | 118  | 0.001 / 0.008      |
| unsplit duringstim                    | 0.01 | 126         | **95**   | 38   | 8      | 87   | 0.001 / 0.008      |


Shuffle duringstim **4-split** still FDR **42** @0.01 (08-14 reproduced). Shuffle f1-only 89 and unsplit 126 also match 08-14b / 08-17.

Median `amp_euc` excl/shuffle: duringchoice 4 **0.76**; duringstim 3 0.86; f1 **1.13**; unsplit **1.09**. f1/unsplit distances are not smaller; p-values get worse (fewer p at the 0.0005 floor: f1 64→45, unsplit 99→67). Duringchoice 4-split @0.01 FDR drops 42→33 even though uncorr ≤0.01 rises 46→61 — extra hits sit off the p-floor and fail BH.

Harris unique remains 0 FDR @0.01 on every split-conditioned combine (duringchoice 4, duringstim 3, f1). Unsplit Harris is still 0 @0.01 and 9 @0.05 (08-17), vs excl 95 / 127.

Plots / CSV (alyx `meta/`; 2 columns, shuffle vs excl):

- `table_act_block_excl_sticky_vs_shuffle_duringchoice_p_mean_c_{0.05,0.01}.png`
- `table_act_block_excl_sticky_vs_shuffle_duringstim_f1_p_mean_c_{0.05,0.01}.png`
- `table_act_block_excl_sticky_vs_shuffle_unsplit_duringstim_p_mean_c_{0.05,0.01}.png`

**Interpretation:** the trim is **not** Harris. Choice L–R expanded; prior f1 and unsplit duringstim **shrink** and stay far above Harris-null. That is closer to the original “drop late/sticky trials, lose some shuffle hits” story than to “drop zoned-out noise, raise SNR,” and it does not explain why Harris wipes prior maps (label autocorrelation under a donor-prior null). Duringchoice 4-split is mixed by α (wider @0.05, narrower @0.01) and still uses the pre-min5 openalyx shuffle. Do not treat the missing duringstim f2 as a 4-split result; f1-only is the fair duringstim comparison (08-14b: shuffle prior is almost entirely f1). Canonical 4-split duringstim completed **2026-08-25**.

---



## 2026-08-25 — act-prior duringstim 4-split excl-sticky FDR (canonical)

`act_block_duringstim_l_choice_r_f2` landed locally (mtime 2026-08-25 09:51; 184 regions / 33.9 k cells, `n_null=2000`). Shuffle counterpart: 201 / 56.0 k (~**60 %** cells kept, same as the other duringstim f2). All 10 requested splits are now in `res_excl_sticky/`. Same combine as 08-24 / 08-14: sum `*_regde` → `p_mean` → BH-FDR. Shuffle duringstim 4-split from alyx `res/new` min5 (FDR **42** @0.01 reproduced). Duringchoice 4 / f1 / unsplit numbers unchanged from 08-24.

### Coverage (new file)


| split                                 | shuffle nreg / cells | excl nreg / cells | cells kept |
| ------------------------------------- | -------------------- | ----------------- | ---------- |
| duringstim f2 `l_choice_r_f2`         | 201 / 56.0 k         | 184 / 33.9 k      | **60 %**   |
| duringstim f2 `r_choice_l_f2` (08-24) | 200 / 55.2 k         | 172 / 31.2 k      | **57 %**   |




### FDR (`p_mean`, BH)


| combine            | α    | shuffle FDR | excl FDR | lost | gained | kept | median p sh / excl |
| ------------------ | ---- | ----------- | -------- | ---- | ------ | ---- | ------------------ |
| duringstim **4**   | 0.05 | 56          | **69**   | 22   | 35     | 34   | 0.119 / 0.080      |
| duringstim **4**   | 0.01 | 42          | **28**   | 24   | 10     | 18   | 0.119 / 0.080      |
| duringchoice **4** | 0.05 | 52          | **71**   | 18   | 37     | 34   | 0.089 / 0.078      |
| duringchoice **4** | 0.01 | 42          | **33**   | 22   | 13     | 20   | 0.089 / 0.078      |
| unsplit duringstim | 0.05 | 146         | **127**  | 27   | 9      | 118  | 0.001 / 0.008      |
| unsplit duringstim | 0.01 | 126         | **95**   | 38   | 8      | 87   | 0.001 / 0.008      |


Harris unique of the duringstim **4-split** remains **0 FDR** at both 0.05 and 0.01 (uncorr 3 / 2). f2-only shuffle is already 0 FDR at both α; excl adds 4 hits only at 0.05.

Uncorr vs FDR on the 4-split: @0.05 uncorr 75→92 and FDR 56→**69** (both up); @0.01 uncorr 54→59 but FDR 42→**28** — extra hits sit off the p-floor (shuffle 25 → excl 22 regions at p≤0.0005) and fail BH. Same mixed-α pattern as duringchoice 4-split.

Median `amp_euc` excl/shuffle: duringstim 4 **0.77**; f2-only 0.73; duringchoice 4 0.76; f1 **1.13**; unsplit **1.09**. The 4-split amplitude is pulled by f2 (smaller distances), not by f1.

Plots / CSV (alyx `meta/`):

- `table_act_block_excl_sticky_vs_shuffle_duringstim_p_mean_c_{0.05,0.01}.*` (2-col, canonical 4-split)
- `table_act_block_excl_sticky_vs_shuffle_p_mean_c_{0.05,0.01}.png` (4-col: duringstim 4 + duringchoice 4, shuffle vs excl)

The incomplete 08-24 3-split duringstim row is superseded.

**Interpretation:** with both f2 files in, canonical **4-split duringstim** is no longer a shrink-at-both-α map. It tracks duringchoice 4-split (wider @0.05, narrower @0.01). The f1-only and unsplit maps still **shrink** at both α and still have tens of FDR hits (72 / 95 @0.01). Harris unique is still 0 @0.01 on every split-conditioned combine. The trim is still **not** Harris: it does not wipe prior maps. The 4-split expansion @0.05 is f2-driven (shuffle f2 FDR = 0; excl f2 FDR = 4; cell retention ~57–60 %). f1 remains the fair “where shuffle prior lives” comparison (08-14b).

---



## Next investigations



### 3. Same exclusion on prior L–R splits

**Done 2026-08-25** for the full 8 split-conditioned splits + unsplit duringstim (canonical 4-split duringstim complete). Still not run: `act_block_only`, unsplit duringchoice, and a duringchoice 4-split vs **min5** shuffle if that baseline is ever copied to alyx.

Do **not** treat excl-sticky as a replacement for Harris. 4-split maps are mixed by α; f1 / unsplit shrink but leave tens of FDR hits; Harris unique is still 0 @0.01.

---



## Open questions

1. **Overlap (answered 08-18b):** sticky tails are spread through the session (median 19 % in the last 20 %). The pers rule is an extra ~5 % cut, not redundant with late.
2. **Behaviour in the last 20 % (answered 08-18b–g / 08-21):** not zoned-out inaccurate; not more true-block-aligned. Calendar mean run vs pooled early 80 % is longer (08-18f), but quintiles of the post-0.5 sequence show that bump **peaks in Q4**, not Q5. RT is the only Q5 outlier vs Q1–Q4. Last 10 % is slower and *less* autocorrelated (08-18g).
3. **Why choice FDR expands:** not because late trials are behavioural failures. Still open: late neural variability (RT slowing), n_L/n_R / trial-count, or a real choice signal diluted by late trials for another reason. Canonical prior **4-split** now expands at FDR 0.05 too (08-25); f1/unsplit still shrink.
4. **Prior L–R under the same trim (answered 08-25):** canonical 4-split duringstim **expands @0.05 (56→69) / shrinks @0.01 (42→28)**, same mixed-α pattern as duringchoice 4. f1 and unsplit still **shrink**, not Harris-null. f2-only shuffle FDR is 0; excl adds 4 hits only @0.05.
5. Re-plot choice excl-sticky against the **min5** shuffle baseline (08-14), not only pre-min5 openalyx.
6. This remains a robustness arm next to Harris, not the primary structured null ([structured nulls](structured_nulls_choice_lr.md) open question 1).

