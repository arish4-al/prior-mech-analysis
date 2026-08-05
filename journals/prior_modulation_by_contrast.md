# Prior modulation by contrast, and the 0 %-contrast choice-conditioned analysis

**Scope:** the real-data question of whether block/action-kernel prior modulation of during-trial population responses depends on stimulus contrast, including the special case of 0 % contrast where the prior is the only information available.

**Status:** the contrast-stratified sweep is complete and retained as a diagnostic; the **revised** primary analysis (0 % contrast, conditioned only on choice side) is finalized over the full BWM. Result: prior distance at 0 % contrast is weak — only **MRN** and **SCm** clear BH-FDR α=0.05, both offset- rather than gain-driven, and nothing survives α=0.01.

Sources: dated entries 2026-07-06 (Goal 3), 07-06d, 07-12, 07-14, 07-14b, 07-14c, 07-14d, 07-17, 07-20.

---

## Goal (original)

Redo the real-data prior-modulation analysis **separately per contrast** on **during-trial** splits (`*block_duringstim*`, `*block_stim*duringchoice*`), for both `act_*` and non-act, including **0 % contrast**. Then optionally compute the contrast-response function (CRF) slope and test whether the prior modulates that slope.

Names: `'{base}_{contrast}'` (e.g. `act_block_duringstim_l_choice_l_f1_0.125`). The contrast is parsed from the split name in the cached pipeline — no `bycontrast` flag needed.

## Goal (revised, 2026-07-17)

Focus on **0 %-contrast trials only**, and at zero contrast **remove the old same-stim-side restriction**: stimulus side is not a meaningful conditioning variable there and unnecessarily fragments the trials. For each choice side separately, compare block L versus block R using all 0 %-contrast trials with that choice:

- choice L: block L versus block R
- choice R: block L versus block R

So the new analysis conditions only on a common **choice side**; it does not retain the stim-side or f1/f2 subdivisions. Report results both **per individual region** and **aggregated across all regions**.

This supersedes the four-way `stim side × choice/feedback` split for the main result. The contrast-conditioned tables below remain useful diagnostics and historical results.

---

## Implementation

### First attempt (2026-07-06d) — ITI `block_only_c*` (wrong target)

`CONTRASTS = [1.0, 0.25, 0.125, 0.0625, 0.0]`; new split family `block_only_c{contrast}` (and `act_block_only_c{contrast}`) registered in `align`/`pre_post` with the ITI window as the base `block_only`. In `get_d_vars`, the `block_only` branch parses a trailing `_c{val}` and filters trials to `|contrast| == val` before the block-L-vs-R split.

This was the wrong scaffolding: Goal 3 is about **during-trial** splits, not the ITI `block_only` family.

### Contrast-response function (CRF) slope (2026-07-06d, built, not run)

- `get_crf_slope(pid, cached=...)`: per region, single-bin post-stim response (default window `[0, 0.15]`) as a function of contrast, computed separately for **concordant** (block favours the stim side) vs **discordant** priors. Fits an OLS slope of response vs contrast for each; prior modulation of gain = `slope_conc − slope_disc` averaged over L/R sides. Significance via a null that **shuffles concordant/discordant block labels within each (side, contrast) cell** (preserving side/contrast structure). Returns per-region CRF curves, slopes, `slope_mod`, `p_slope_mod`.
- `get_all_crf_slope(...)`: per-insertion driver (uses the cache), saves `manifold/crf_slope/{eid_probe}.npy`.
- `crf_slope_stacked(...)`: pools across insertions per region (nanmean `slope_mod`, mean p, `frac_sig`, mean CRF curves) → `manifold/res/crf_slope_stacked.npy`.

Design choices to confirm: concordant = high prior for the stimulus side; the slope fit is linear in raw contrast (not log); the response is a total spike count in a single `[0, 0.15]` bin; 0 % contrast anchors the CRF low end (fully prior-driven). **Never run.**

### Corrected during-trial contrast splits (2026-07-12)

- Bases: `GOAL3_DURINGSTIM_BASES` (8) + `GOAL3_DURINGCHOICE_BASES` (8) = 16 bases × 5 contrasts → **80** registered splits.
- `contrast_from_split(name)` auto-parses a trailing `_{float}` / `_c{float}` (regex anchored at end-of-string so `_choice` never false-matches).
- `get_d_vars`: the contrast filter is applied via `_filter_stim_side` whenever the name carries a contrast — **no** `bycontrast=True` flag. Act vs non-act is still toggled by `'act' in split`. Windows are copied from the base split (`[0,0.15]` duringstim, `[0.15,0]` duringchoice).
- The cached path skips the remote `pid2eid` when the insertion cache already has eid/probe.
- CLI presets in `scripts/run_goal2_splits.py`: `goal3_duringstim`, `goal3_duringchoice`, `goal3_duringstim_act`, `goal3_duringstim_block`, `goal3_duringchoice_act`, `goal3_duringchoice_block`, `goal3_all`; optional `--contrasts 0.0 0.125 1.0`; `--list-splits`.
- **Sharding:** contrast splits are ordinary split names, so `run_goal2_shard_slurm.sh` + `run_goal2_finalize_slurm.sh` work unchanged (`{split}.shard{k}.npy` tolerates dots in `0.125`). Submitter: `scripts/submit_goal3_sharded.sh` (default `goal3_duringstim_act`, N_SHARDS=4). Unsharded smoke: `scripts/run_goal3_contrast_slurm.sh`.

```bash
# Full BWM on ORCD (recommended):
bash scripts/submit_goal3_sharded.sh
PRESET=goal3_duringstim_act CONTRASTS="0.0 0.125 1.0" N_SHARDS=4 bash scripts/submit_goal3_sharded.sh

# Local / smoke (unsharded):
conda activate iblenv
python scripts/run_goal2_splits.py --preset goal3_duringstim_act --contrasts 0.0 0.125 1.0
```

**Smoke (alyx insertion cache, 3 pids × 6 splits, nrand=10):** 18/18 OK — act + non-act, duringstim + duringchoice, contrasts 0/0.125/0.25/1.0; duringstim curves len=72.

### Revised 0 %-contrast implementation (2026-07-17)

- New true-block splits: `block_duringstim_choice_l_0.0` and `block_duringstim_choice_r_0.0`.
- The trial mask is `(contrastLeft == 0 OR contrastRight == 0) AND fixed choice`; neither nominal stimulus side nor `feedbackType` is used.
- Existing finalized `{split}.npy` / `{split}_regde.npy` provide regional results. Finalization now also writes `{split}_all.npy` and `{split}_all_regde.npy`, pooling raw squared distances over all valid neurons **before** normalization (not averaging normalized regional curves).
- `scripts/run_goal2_splits.py --preset goal3_c0_choice` and the Goal-3 Slurm scripts run the two revised splits by default.
- `scripts/plot_goal3_c0_summary_table.py` defaults to the revised analysis and writes one BH-FDR regional CSV per choice plus `goal3_c0_choice_all_regions.csv`. The historical analysis remains available via `--legacy-contrast`.

**Validation:** split registration/window, synthetic zero-contrast masks (both nominal sides and feedback outcomes), raw all-region pooling, summary CSV output, Python compilation, and shell syntax all pass.

---

## Results — contrast-stratified sweep (diagnostic)

### Cell retention (2026-07-14)

Downloaded finalized contrast splits: `alyx.../manifold/res/new/`.

An earlier draft summed L+R `nclus` (~125 k) and double-counted neurons across stim sides. The correct baseline is **per stim side** (~62.5 k f1; ~55.6 k f2), matching unsplit `act_block_duringstim_{l,r}` (~62.8 k).

**Metric:** pooled `nclus` from finalized splits; **% kept** = mean(L,R) / mean(all-contrast L,R). Region count = union of L∪R with ≥1 cell in that split.

**f1 (correct; L = `*_l_choice_l_f1`, R = `*_r_choice_r_f1`)**

| contrast | L cells | R cells | mean | % kept | nreg |
|----------|--------:|--------:|-----:|-------:|-----:|
| all | 62,520 | 62,575 | 62,548 | 100 % | 208 |
| 1.0 | 53,136 | 55,601 | 54,368 | 86.9 % | 206 |
| 0.25 | 56,013 | 53,996 | 55,004 | 87.9 % | 203 |
| 0.125 | 46,269 | 45,364 | 45,816 | 73.3 % | 199 |
| 0.0625 | 35,290 | 35,823 | 35,556 | 56.8 % | 191 |
| 0.0 | 2,788 | 6,816 | 4,802 | **7.7 %** | 92 |

**f2 (incorrect; L = `*_l_choice_r_f2`, R = `*_r_choice_l_f2`)**

| contrast | L cells | R cells | mean | % kept | nreg |
|----------|--------:|--------:|-----:|-------:|-----:|
| all | 56,045 | 55,216 | 55,630 | 100 % | 206 |
| 1.0 | — | 51 | 51 | **0.1 %** | 1 |
| 0.25 | 408 | 280 | 344 | 0.6 % | 17 |
| 0.125 | 3,458 | 2,664 | 3,061 | 5.5 % | 86 |
| 0.0625 | 19,244 | 15,906 | 17,575 | 31.6 % | 174 |
| 0.0 | 11,151 | 9,197 | 10,174 | 18.3 % | 139 |

f1 thins at low contrast (0 % nearly empty under the ≥5 trials/side gate). f2 is nearly empty at high contrast (errors are rare) and only modest at 0 %.

### Gain/offset tables at α=0.01 — nothing survives

Combine the available four `act_block_duringstim_*_{c}` splits → `p_mean`/`p_gain`/`p_offset` → BH-FDR → `plot_table_with_styles` gain/offset, α=0.01:

`alyx.../meta/table_act_block_combined_summary_act_p_mean_c_combinedpTrue_0.01_gain_offset_{c1,c025,c0125,c00625,c0}.png`

Script: `scripts/plot_goal3_c0_summary_table.py` (all contrasts; `--retention-only`).

At α=0.01 FDR, **0 regions** pass `p_mean_c` for any contrast-conditioned combined table. c=1 uses 3/4 splits (`l_choice_r_f2_1.0` missing).

### Pipeline sanity: all-contrast recoverability (2026-07-14b)

Same combine → `p_mean`/`p_gain`/`p_offset` → BH-FDR (α=0.01) → gain/offset table, on the **unconditioned** four `act_block_duringstim_*` splits (no contrast suffix):

| source | path | FDR `p_mean_c` ≤ 0.01 | gain ∩ sig |
|--------|------|--------------------:|---------:|
| openalyx copies (isolated) | `res/new/openalyx_allcontrast_ref/` | **37**/208 | 20 |
| alyx `res/new` all-contrast | `res/new/act_block_duringstim_*` | **42**/208 | 26 |

Plots (alyx meta only; do not overwrite openalyx): `…_gain_offset_openalyx_ref.png`, `…_gain_offset_alyx_new_all.png`.

alyx and openalyx files are **not bitwise identical** (null shuffle seeds + ~15/207 regions with small `nclus` differences, e.g. CP 2759 vs 2655), but the true curves match (VISp corr = 1.0) and both recover tens of FDR hits — so the combine/FDR path is fine. The per-contrast zeros are not a plotting bug.

### Why per-contrast looks dead at α=0.01 even at c=1 (2026-07-14c)

**Not mainly cell loss.** f1 at c=1 keeps ~87 % of cells and 206 regions. Two compounding issues:

**(1) Discrete p-floor vs BH-FDR at α=0.01 (dominant).** With `nrand=2000` the minimum attainable p ≈ **1/2001 ≈ 0.0005**. For BH at α=0.01 with m ≈ 206 tests you need **≥11 regions pinned at that floor** before *any* rejection is possible (`k ≥ ceil(0.0005·m/α)`).

| set | n at p-floor | `p<0.01` | FDR@0.01 | FDR@0.05 |
|-----|-------------:|---------:|---------:|---------:|
| all-contrast | 25 | 54 | **42** | 56 |
| c=1.0 | **10** | 36 | **0** | **32** |
| c=0.25 | 3 | 24 | 0 | 9 |

So c=1 is one region short of clearing the α=0.01 floor barrier; at **α=0.05 FDR there are 32 hits**. "Nothing significant at the highest contrast" was an α=0.01 × nrand interaction, not an absence of signal.

**(2) Smaller per-contrast effect / SNR.** Pooling all contrasts accumulates prior distance across conditions; restricting to c=1 uses fewer trials per cell:

| set | median amp | median effect (true − null mean) | median SNR |
|-----|-----------:|-------------------------------:|-----------:|
| all-contrast | 1.75 | 0.114 | 1.23 |
| c=1.0 | 0.85 | 0.056 | 1.12 |
| c=0.25 | 0.98 | 0.043 | 0.82 |

Of the 42 all-contrast FDR@0.01 regions, **18** still have uncorrected `p ≤ 0.01` at c=1 (27 at `p ≤ 0.05`), but only 10 hit the floor — not enough for BH@0.01.

**Also:** f2 is empty at high contrast, so the "4-split" c=1 table is effectively **f1-only** (plus a tiny f2 R file).

**Takeaways:** (a) re-plot / re-threshold per-contrast at **FDR α=0.05**, or raise `nrand` (e.g. 10k) to insist on α=0.01; (b) high-contrast prior distance is weaker than pooled — consistent with the prior mattering more when the stimulus is ambiguous.

### Per-contrast gain/offset tables at FDR α=0.05 (2026-07-14d)

```bash
python scripts/plot_goal3_c0_summary_table.py --alpha 0.05 --skip-retention
```

| contrast | nreg | FDR `p_mean_c` ≤ 0.05 | gain ∩ sig | offset ∩ sig | notes |
|----------|-----:|--------------------:|---------:|-----------:|-------|
| 1.0 | 206 | **32** | 23 | 11 | 3/4 splits |
| 0.25 | 203 | **9** | 6 | 4 | |
| 0.125 | 200 | **3** | 1 | 2 | |
| 0.0625 | 198 | **4** | 2 | 2 | |
| 0.0 | 151 | **0** | 0 | 0 | |

Plots: `alyx.../meta/table_act_block_combined_summary_act_p_mean_c_combinedpTrue_0.05_gain_offset_{c1,c025,c0125,c00625,c0}.png`

Signal recovers strongly at c=1 under α=0.05; mid/low contrasts are weak; 0 % is still null.

---

## Results — revised 0 %-contrast, choice-conditioned (primary, 2026-07-20)

Finalized outputs in `alyx.../manifold/res/new/`: `block_duringstim_choice_{l,r}_0.0{,_regde,_all,_all_regde}.npy`.

**Cell retention** (pooled regions with ≥ min_reg):

| split | nreg | nclus |
|-------|-----:|------:|
| `block_duringstim_choice_l_0.0` | 162 | 25,360 |
| `block_duringstim_choice_r_0.0` | 174 | 34,271 |

**All-region population** (`*_all.npy`; raw-pooled before RMS and min_reg):

| choice | nclus | n_regions | p_euc | amp_euc |
|--------|------:|----------:|------:|--------:|
| L | 25,946 | 238 | 0.074 | 0.059 |
| R | 34,833 | 249 | 0.020 | 0.062 |

Choice-R all-region is uncorrected p ≈ 0.02; neither side supports a strong claim alone. Per-split regional BH-FDR on `p_euc` gives **0** hits at α=0.05 and α=0.01.

**Combined L+R choice** (same path as the old per-contrast tables: sum regde → `p_mean`/`p_gain`/`p_offset` → BH-FDR → gain/offset table):

```bash
python scripts/plot_goal3_c0_summary_table.py --alphas 0.05 0.01
```

| α | nreg | FDR `p_mean_c` | gain ∩ sig | offset ∩ sig | regions |
|---|-----:|---------------:|---------:|-----------:|---------|
| 0.05 | 185 | **2** | 0 | 2 | **MRN**, **SCm** |
| 0.01 | 185 | **0** | 0 | 0 | — |

Both FDR@0.05 hits sit at the nrand=2000 p-floor (`p_mean ≈ 0.0005`); after offset subtraction at α=0.05 neither retains a significant gain (MRN's `p_gain` rises to ~0.15 once the significant offset is removed; SCm was gain-null already).

Plots: `alyx.../meta/table_block_block_combined_summary_block_p_mean_c_combinedpTrue_{0.05,0.01}_gain_offset_c0_choice.png`. Also per-split regional CSVs `…_regions_a{0.05,0.01}.csv` and `goal3_c0_choice_all_regions.csv` in the same meta directory.

**Takeaway:** relaxing the stim-side and f1/f2 conditioning at 0 % recovers far more cells than the old c=0 four-way split (~5 k f1 cells → ~25–35 k per choice). Combined prior distance is nonetheless weak: only midbrain MRN and SCm clear FDR@0.05, both offset- rather than gain-driven, and nothing survives FDR@0.01.

> MRN and SCm also appear as strong early-stim regions in the [variance partition](variance_partition_mixed_regions.md) analysis.

---

## Follow-ups

- Optional: run the CRF slope test (`get_crf_slope`) now that contrast splits exist.
- Optional: raise `nrand` (e.g. 10 k) if per-contrast claims at α=0.01 are needed.
- Optional: finish the ORCD `stimOn_times_act` BWM sweep with the sharded submit (or an unsharded restart where checkpoints are valid).
