# Bayesian prior (real data)

**Scope:** Bayes-optimal prior on BWM (`bayesian_priors` / `*bayes*` / `bayes_block_*`). Label definition: [prior definitions](prior_definitions.md). Null machinery: [structured nulls](structured_nulls_choice_lr.md).

**Status:** duringstim prior L–R shuffle tables (4-split **57** FDR @0.01, stim-side **116**) do **not** survive Harris unique — both maps are **0 FDR @0.01** (09-01; same qualitative result as act Harris). Stim-side Harris has **3** FDR @0.05 (IRN, RN, CLA). Bayes-stratum duringstim shuffles: choice L–R **100**, stim L–R **47** @0.01. Duringchoice prior, contrast, conflict-vs-alignment, Bayes-agent sticky, and choice Harris have not been run.

Sources: 2026-07-12e/f/h (implementation); 07-14 shuffle files on alyx `res/new`; gain/offset tables in alyx `meta/`; 08-23 Bayes-agent null; 08-27b Harris submitter; 08-27c donor-history fix; 08-27d local-6 + stratum shuffle submitters; **2026-09-01** Harris unique + stratum-shuffle FDR.

---

## Label

Findling et al. 2025 SI §1.1.1: **P(stim left | past stimulus sides)** under the IBL generative model (τ=60, γ=0.8, block length 20–100). Binarized 0.8/0.2. Computed on the **full** trial list, then 0.5 blocks dropped. Trigger: `'bayes' in split`. Smoke: 80 left stims → P(left)≈0.77; 80 right → ≈0.23. Does not see choices or rewards.

---

## Split families

| Family | Names | Window | Shuffle | Harris unique | Tables |
|--------|-------|--------|---------|---------------|--------|
| Prior L–R, stim×choice duringstim | `bayes_block_duringstim_{l,r}_choice_*_f{1,2}` | stimOn `[0, 0.15]` | **yes** (07-14) | **yes** (08-30) | shuffle + Harris (09-01) |
| Prior L–R, stim-side only | `bayes_block_duringstim_{l,r}` | `[0, 0.15]` | **yes** (07-14) | **yes** (08-30) | shuffle + Harris (09-01) |
| Prior L–R, stim×choice duringchoice | `bayes_block_stim_{l,r}_duringchoice_*_f{1,2}` | firstMovement `[0.15, 0]` | no | no | — |
| Prior L–R, choice-side only | `bayes_block_duringchoice_{l,r}` | firstMovement | no | no | — |
| Choice L–R, Bayes×stim duringstim | `choice_duringstim_{l,r}_block_{l,r}_bayes` | `[0, 0.15]` | **yes** (08-28) | no | **yes** (09-01) |
| Stim L–R, fixed choice + Bayes | `stim_choice_{l,r}_block_{l,r}_bayes` | `[0, 0.15]` | **yes** (08-28; replaced 07-14) | no | **yes** (09-01) |
| Stim L–R, Bayes only | `stim_block_{l,r}_bayes` | `[0, 0.08]` | **yes** (07-14) | no | no |
| Contrast (Goal 3) | `GOAL3_BAYES_*` | same bases × 5 contrasts | no | no | — |

ITI `bayes_block_only` is scored in [ITI prior](iti_prior.md) (08-31: 0 FDR @0.01 default and Harris).

---

## Shuffle results (duringstim prior L–R)

Alyx `manifold/res/new/`. Combine + BH-FDR + tables via `scripts/plot_goal3_c0_summary_table.py --bayes-choice` / `--bayes-stim-side`. Counts below are from the combined CSVs (`p_mean_c` ≤ α; gain/offset ∩ = `p_*` < α among those). 09-01 re-read of the same 07-14 combines reproduces these numbers.

### Stim × choice (4-split)

f1: 207–208 regions / 62.5 k cells. f2: 184–188 / 39–43 k. Combined **208** regions. **42** at the p-floor (0.0005).

| α | FDR `p_mean_c` | gain ∩ | offset ∩ |
|---|----------------|--------|----------|
| 0.05 | **100** / 208 | 71 | 48 |
| 0.01 | **57** / 208 | 33 | 27 |

`alyx.../meta/table_bayes_block_combined_summary_bayes_p_mean_c_combinedpTrue_{0.05,0.01}_gain_offset.png`

Lowest-p examples: CA3, CA1, PPN, MRN, SCm, CP, MOp, DCO, IP, ANcr1.

### Stim-side only (no f1/f2)

207–209 regions / 62.8 k cells. Combined **209**. **81** at the p-floor (0.0005).

| α | FDR `p_mean_c` | gain ∩ | offset ∩ |
|---|----------------|--------|----------|
| 0.05 | **147** / 209 | 107 | 81 |
| 0.01 | **116** / 209 | 75 | 58 |

`…_gain_offset_stim_lr.png`

**Early-stim 80 ms (09-01):** same stim-side shuffle, prefix `t ≤ 80 ms` of the 150 ms curves. FDR **84** @0.01 / **113** @0.05 (gain ∩ 35, offset ∩ 53 at 0.01). Table: `…_gain_offset_stim_lr_earlystim80.png` — columns L→R: **region**, **gain**, **offset** (see [structured nulls](structured_nulls_choice_lr.md) 09-01).

Dropping choice/f1/f2 **widens** the shuffle map (57 → 116 FDR @0.01). 80 ms shrinks the stim-side shuffle (116 → 84) but does not empty it.

### Reading

These are shuffle-liberal prior maps with many regions at the p-floor. They are **not** a structured-null result. Harris unique on the same splits (below) wipes both maps at α=0.01.

---

## 2026-09-01 — Harris unique (6 local duringstim prior maps)

Local alyx `res/new/`, mtime 28–30 Aug. `FAMILY=local` from `submit_goal2_bayes_harris_orcd.sh`. Donor Bayes labels: full stim history then drop 0.5 (08-27c). Scoring: `scripts/plot_bayes_harris_stratum_shuffle.py` — product-MC four-split combine (ragged unique counts; same as [act Harris](structured_nulls_choice_lr.md) 07-27 / 08-14). Does not rebuild the 07-14 shuffle combines.

| split | regions / cells | n_null min/med/max |
|-------|-----------------|--------------------|
| `…_r_choice_r_f1_harris_unique` | 207 / 62.4 k | 102 / 1819 / 2000 |
| `…_l_choice_l_f1_harris_unique` | 208 / 62.4 k | 152 / 1911 / 2000 |
| `…_l_choice_r_f2_harris_unique` | 188 / 42.6 k | 23 / 465 / 2000 |
| `…_r_choice_l_f2_harris_unique` | 184 / 39.4 k | 9 / 456 / 2000 |
| `…_l_harris_unique` (stim-side) | 209 / 62.7 k | 141 / 1956 / 2000 |
| `…_r_harris_unique` (stim-side) | 207 / 62.6 k | 114 / 2000 / 2000 |

Coverage vs shuffle is ~99 % cells on f1 / stim-side; f2 is the usual short-stratum thin (cells already 39–43 k on shuffle). Combined 4-split **208**, stim-side **209**.

| arm | combine | uncorr ≤0.01 | uncorr ≤0.05 | FDR @0.01 | FDR @0.05 | median p | median amp |
|-----|---------|-------------:|-------------:|----------:|----------:|---------:|-----------:|
| shuffle (07-14) | 4-split | 79 | 120 | **57** | 100 | 0.031 | 1.12 |
| **Harris unique** | 4-split | 0 | 2 | **0** | **0** | 0.653 | 1.12 |
| shuffle (07-14) | stim-side | 127 | 154 | **116** | 147 | 0.002 | 0.466 |
| **Harris unique** | stim-side | 8 | 11 | **0** | **3** | 0.626 | 0.466 |

4-split vs shuffle (208 shared): **57 → 0** @0.01 (lost 57, gained 0); **100 → 0** @0.05. Stim-side (209): **116 → 0** @0.01; @0.05 **147 → 3** (kept IRN, RN, CLA). Observed `amp_euc` median ratio **1.00** on both combines — same curves, wider Harris null.

4-split Harris uncorr @0.05: IRN, GRN (lowest p 0.030 / 0.043). Stim-side uncorr @0.01: CP, IRN, MOs, GRN, RN, SOC, CLA, ECU.

**f1 / f2 only** (in-memory product-MC; same 08-14b check as act):

| combine | shuffle FDR @0.01 | Harris FDR @0.01 | Harris uncorr @0.01 / @0.05 |
|---------|------------------:|-----------------:|-----------------------------|
| f1 only (2 splits) | **90** / 208 | **0** | 0 / 5 |
| f2 only | 0 / 195 | **0** | 1 / 7 |
| all 4 | 57 | **0** | 0 / 2 |

Shuffle prior signal is almost entirely **f1**. Including f2 dilutes 90 → 57, same as act (89 → 42). Harris is 0 on f1 alone (coverage ~99 %, unique pools median ~1800–1910) — not the f2 donor skip.

**vs act Harris** ([structured nulls](structured_nulls_choice_lr.md) 08-14 / 08-17): same qualitative result. Bayes shuffle is a bit more liberal than act (4-split 57 vs 42; stim-side 116 vs 126). Harris wipes both at α=0.01. Stim-side Harris @0.05 is quieter than act (3 vs 9); the three Bayes hits (IRN, RN, CLA) were also in the act unsplit @0.05 list.

Plots / CSV: `alyx.../meta/table_bayes_block_{,unsplit_}harris_unique_vs_shuffle_p_mean_c_0.01.{png,csv}`.

**Reading:** do not treat shuffle 57 / 116 as a sensory-prior claim. Harris unique is a structured total null on these duringstim Bayes prior maps at α=0.01, matching act-prior Harris.

---

## 2026-09-01 — label shuffle inside Bayes stratum (duringstim 4+4)

`submit_goal2_bayes_shuffle_orcd.sh` (08-27d). mtime 28 Aug. `n_null=2000` everywhere. Combined via the same script as Harris (aligned sum; all U=2000). Gain/offset tables: `plot_goal3` path.

### Choice L–R (stratum = stim × Bayes)

`choice_duringstim_{l,r}_block_{l,r}_bayes`. Per-split 195–201 regions / 47–55 k. Combined **209**. **77** at the p-floor (0.0005).

| α | FDR `p_mean_c` | gain ∩ | offset ∩ |
|---|----------------|--------|----------|
| 0.05 | **133** / 209 | 108 | 51 |
| 0.01 | **100** / 209 | 85 | 26 |

`…_gain_offset_choice_lr.png`

Liberal, in the same ballpark as act choice L–R min5 duringstim (88 FDR @0.01; [structured nulls](structured_nulls_choice_lr.md) 08-14). Not a structured null — Harris unique for `*_bayes` choice has not been run.

### Stim L–R (stratum = choice × Bayes)

`stim_choice_{l,r}_block_{l,r}_bayes` (08-28 re-run; journal previously listed these as 07-14). Per-split 195–203 / 47–56 k. Combined **208**. **31** at the p-floor.

| α | FDR `p_mean_c` | gain ∩ | offset ∩ |
|---|----------------|--------|----------|
| 0.05 | **86** / 208 | 58 | 33 |
| 0.01 | **47** / 208 | 38 | 11 |

`…_gain_offset_stim_choice_lr.png` (not the prior stim-side `…_stim_lr.png`).

The 07-14 stim-only pair `stim_block_{l,r}_bayes` is unchanged and still untabled.

---

## Structured nulls still not scored

**Bayes-agent option-1 + copy-last (08-23).** OptimalBayesian choices (fixed ζ=0.1, lapse=0.05). Disk `{split}_pseudo_strat_sticky.npy`. Files not in local `res/new`. FDR not run.

**Choice Harris unique** (`FAMILY=choice`) and **duringchoice prior Harris** (`FAMILY=prior`) were not in the local-6 copy.

```bash
bash scripts/submit_goal2_bayes_harris_orcd.sh          # local 6 — scored 09-01
FAMILY=choice bash scripts/submit_goal2_bayes_harris_orcd.sh
FAMILY=prior bash scripts/submit_goal2_bayes_harris_orcd.sh
```

---

## Open

1. Duringchoice prior L–R (shuffle + Harris).
2. Choice L–R Harris unique inside the Bayes stratum — does shuffle 100 survive?
3. Goal 1: conflict vs alignment of the Bayes prior with recent reward.
4. Goal 3 contrast `goal3_*_bayes`.
5. Bayes-agent option-1 + copy-last FDR on `*_bayes` / `bayes_block_*`.
