# Bayesian prior (real data)

**Scope:** Bayes-optimal prior on BWM (`bayesian_priors` / `*bayes*` / `bayes_block_*`). Label definition: [prior definitions](prior_definitions.md). Null machinery: [structured nulls](structured_nulls_choice_lr.md).

**Status:** shuffle tables exist for duringstim prior L–R (4-split and stim-side). Harris unique submitter defaults to those **6 local splits** (08-27d). Stim L–R / choice L–R duringstim (4+4) label shuffle inside Bayes stratum is wired (`submit_goal2_bayes_shuffle_orcd.sh`). Duringchoice prior, contrast, and conflict-vs-alignment have not been run.

Sources: 2026-07-12e/f/h (implementation); 07-14 shuffle files on alyx `res/new`; gain/offset tables in alyx `meta/`; 08-23 Bayes-agent null; 08-27b Harris submitter; 08-27c donor-history fix.

---

## Label

Findling et al. 2025 SI §1.1.1: **P(stim left | past stimulus sides)** under the IBL generative model (τ=60, γ=0.8, block length 20–100). Binarized 0.8/0.2. Computed on the **full** trial list, then 0.5 blocks dropped. Trigger: `'bayes' in split`. Smoke: 80 left stims → P(left)≈0.77; 80 right → ≈0.23. Does not see choices or rewards.

---

## Split families

| Family | Names | Window | Shuffle npy | Tables |
|--------|-------|--------|-------------|--------|
| Prior L–R, stim×choice duringstim | `bayes_block_duringstim_{l,r}_choice_*_f{1,2}` | stimOn `[0, 0.15]` | **yes** (07-14) | **yes** |
| Prior L–R, stim-side only | `bayes_block_duringstim_{l,r}` | `[0, 0.15]` | **yes** (07-14) | **yes** |
| Prior L–R, stim×choice duringchoice | `bayes_block_stim_{l,r}_duringchoice_*_f{1,2}` | firstMovement `[0.15, 0]` | no | — |
| Prior L–R, choice-side only | `bayes_block_duringchoice_{l,r}` | firstMovement | no | — |
| Choice L–R, Bayes×stim | `choice_*_bayes` | choicestim windows | no | — |
| Stim L–R, fixed choice + Bayes | `stim_choice_*_block_*_bayes` | `[0, 0.15]` | **yes** (07-14) | no |
| Stim L–R, Bayes only | `stim_block_{l,r}_bayes` | `[0, 0.08]` | **yes** (07-14) | no |
| Contrast (Goal 3) | `GOAL3_BAYES_*` | same bases × 5 contrasts | no | — |

All on-disk neural distances are **plain label shuffle** (`{split}.npy`), not Harris / sticky.

---

## Shuffle results (duringstim prior L–R)

Alyx `manifold/res/new/`. Combine + BH-FDR + tables via `scripts/plot_goal3_c0_summary_table.py --bayes-choice` / `--bayes-stim-side`. Counts below are from the combined CSVs (`p_mean_c` ≤ α; gain/offset ∩ = `p_*` < α among those).

### Stim × choice (4-split)

f1: 207–208 regions / 62.5 k cells. f2: 184–188 / 39–43 k. Combined **208** regions. **42** at the p-floor (0.0025).

| α | FDR `p_mean_c` | gain ∩ | offset ∩ |
|---|----------------|--------|----------|
| 0.05 | **100** / 208 | 71 | 48 |
| 0.01 | **57** / 208 | 33 | 27 |

`alyx.../meta/table_bayes_block_combined_summary_bayes_p_mean_c_combinedpTrue_{0.05,0.01}_gain_offset.png`

Lowest-p examples: CA3, CA1, PPN, MRN, SCm, CP, MOp, DCO, IP, ANcr1.

### Stim-side only (no f1/f2)

207–209 regions / 62.8 k cells. Combined **209**. **81** at the p-floor (0.0013).

| α | FDR `p_mean_c` | gain ∩ | offset ∩ |
|---|----------------|--------|----------|
| 0.05 | **147** / 209 | 107 | 81 |
| 0.01 | **116** / 209 | 75 | 58 |

`…_gain_offset_stim_lr.png`

Dropping choice/f1/f2 **widens** the shuffle map (57 → 116 FDR @0.01).

### Reading

These are shuffle-liberal prior maps with many regions at the p-floor. They are **not** a structured-null result. Do not treat 57 / 116 as a sensory-prior claim until Harris (or Bayes-agent sticky) is scored on the same splits.

### Stim L–R under Bayes (not prior distance)

Shuffle files on `res/new` (07-14); coverage similar (stim-only ~208 reg / 63 k; stim×choice ~195–203 / 47–56 k). No combine/FDR tables.

---

## Structured nulls (not scored)

**Harris unique (08-27d).** Default is the 6 splits with local shuffle maps (not duringchoice / not choice Harris):

```bash
bash scripts/submit_goal2_bayes_harris_orcd.sh
```

4-split duringstim + stim-side unsplit. Donor Bayes labels: full stim history then drop 0.5 (08-27c).

**Label shuffle inside Bayes stratum (08-27d).** Duringstim only, 4+4. Stim L–R: choice × Bayes; choice L–R: stim × Bayes. Writes `{split}.npy`.

```bash
bash scripts/submit_goal2_bayes_shuffle_orcd.sh
```

**Bayes-agent option-1 + copy-last (08-23).** OptimalBayesian choices (fixed ζ=0.1, lapse=0.05). Disk `{split}_pseudo_strat_sticky.npy`. FDR not run.

---

## Open

1. Harris unique FDR on the two duringstim prior maps — does shuffle 57 / 116 survive?
2. Choice L–R / stim L–R duringstim shuffle inside Bayes stratum (08-27d submitter; not scored).
3. Duringchoice prior L–R.
4. Goal 1: conflict vs alignment of the Bayes prior with recent reward.
5. Goal 3 contrast `goal3_*_bayes`.
