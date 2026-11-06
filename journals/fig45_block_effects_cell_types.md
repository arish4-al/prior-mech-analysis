# Figure 4.5: block (prior) effects on rastermap cell types

**Goal.** Recreate / rerun the analysis behind Figure 4.5: whole-population
rastermap (a), block-L vs block-R population distance for the C
(choice/integrator), M (movement-init + movement) and S (stimulus) cell
types against block-label shuffles (b, c), and C vs M shuffle-subtracted
curves (e). Panel d is a schematic. The rerun uses the new rastermap stack
`~/Downloads/concat_cvTrue.npy`.

Source code (paper-brain-wide-map, read-only from here):
`bwm_dmn.ipynb` sections "examine block effects on cell types from
rastermap" (cells 73–91) and "find stim res cells from rastermap" (92–111);
`dmn_avgbystim.py` (`get_uuids_raster_types`, `action_kernel_priors`,
`concat_PETHs`, `get_all_PETHs`, `stack_concat`); `analysis_functions.py`
(`plot_average_distance_across_all_pairs`,
`plot_averaged_distances_comparison`); `dmn_ari.py` (old 26-type concat,
`vis_subset0` uuid list).

Scripts (this repo): `scripts/fig45_common.py`, `fig45_define_cell_types.py`,
`fig45_panel_a.py`, `fig45_block_peths.py`, `fig45_plot_distances.py`.
Outputs live under `<dmn>/res/fig45/` with
`<dmn> = ~/Downloads/ONE/openalyx.internationalbrainlab.org/dmn`.

---


<!-- plot stim responding cells in raster -->
<!-- check percentage of prior modulations?? -->
<!-- quantification of contextualization in sequence/tornado cells -->

## What each panel is

| Panel | Analysis | Data / function |
|---|---|---|
| a top | rastermap of all cells, held-out half sorted by train-half `isort`; columns quiescence → R_move | old: `cross_val_test.npy` sorted by `cross_val_train` isort (26-type concat, `dmn_ari.tts__`) |
| a bottom | zoom on the M (L,R mvt init) + C (choice/integrator) rows | same |
| b | `d_prior(t)` for C (gold) and M (tomato), stim-aligned 0–150 ms and move-aligned −150–0 ms | `plot_average_distance_across_all_pairs(vers='concat_block_pairs_act', plot_offset=True, plot_gain=True, plot_inset=True)` on `*_shuffleTrue_integrator.npy` / `*_move_init_movement.npy` |
| c | same for S, stim-aligned | same function on `concat_block_pairs_act_vis_normFalse_shuffleTrue_vis_subset0.npy` (**not** `stim_res`, see below) |
| e | mean over pairs of (distance − shuffle mean), C vs M | `plot_averaged_distances_comparison(vers='concat_block_pairs_act')` |

### Pipeline (original)

1. **Cell types** from index ranges along the old rastermap `isort`
   (`dmn_avgbystim.get_uuids_raster_types`): stim_res [6000, 6700),
   integrator [4500, 6000), move_init [750, 4500), movement [43720, 44880).
   The notebook cell 72 copy (integrator 3700–6000, move_init 840–3700) is
   stale; the cell counts in the 2025 files (1,457 integrator, 4,856
   move_init + movement) match the `dmn_avgbystim` ranges.
2. **Per-insertion PETHs** (`concat_PETHs`, vers `concat_block_pairs_act`):
   - trials from `load_trials_and_mask` with the six saturation intervals;
   - `act` prior: action kernel, α = 0.2, over all choices
     (prior_t from choices < t), binarised (≥ 0.5 → 0.8, else 0.2), replacing
     `probabilityLeft`;
   - 16 trial types = stim side × choice × block; stimOn [0, 200] ms and
     firstMovement [−150, 0] ms; 12.5 ms bins at 2 ms stride (96 / 72 points);
   - real PETH = trial-mean spike count per bin, typed cells only;
   - null: per block pair, pool both conditions' trials and permute the
     block label 2,000 times (group sizes kept).
3. **Stack** (`stack_concat(shuffling=True)`): drop insertions with > 2
   empty trial types and cells with NaN / all-zero PETHs; drop AUDp from
   stim_res; `distance[pair] = Σ_cells (PETH_a − PETH_b)²` per bin; the same
   on each shuffle is `distance_controls` (n_rand × T).
4. **Plot**: `d = sqrt(distance / T_BIN / n_cells)`, averaged over the 4
   stim-aligned or 4 move-aligned block pairs. The null is the same average
   over the shuffle rows. Stim side is cut at 150 ms.
   - `p_mean`: fraction of shuffles whose time-mean ≥ the real one.
   - `p_offset`: same on the first 5 bins.
   - Solid line: if `p_offset < 0.05`, the real curve minus the offset
     excess (dashed = raw).
   - `p_gain`: time-mean of the offset-removed curve from bin 4 onward vs
     shuffles.
   - Inset: real − shuffle mean, with the 2.5–97.5 % shuffle band.

Units quirk: `/T_BIN` is applied to *squared* count differences, so `d` is
in counts/√s, not Hz. This is kept as-is to match the paper's y-axes.

---

## 2026-10-13 — reproduction from the 2025 intermediates; new-stack mapping

### b, c, e reproduce from the existing stacked files

`fig45_plot_distances.py --tag old_crossval` on
`<dmn>/res/concat_block_pairs_act_normFalse_shuffleTrue_{integrator,move_init_movement}.npy`
reproduces panels b and e exactly (curves, y-ranges, the gold / tomato
peaks at ~0.15 / ~0.25 in e). Plots are in `<dmn>/res/fig45/old_crossval/`.

| type | n | stim p_mean | stim p_offset | stim p_gain | choice p_mean | M/C amp ratio (e) |
|---|---|---|---|---|---|---|
| C integrator | 1457 | 0.0005 | 0.0025 | 0.0005 | 0.0005 | — |
| M move_init_movement | 4856 | 0.0005 | 0.0005 | 0.0005 | 0.0005 | stim 1.45, choice 1.22 |
| S `stim_res` | 680 | 0.0005 | 0.005 | 0.71 | 0.0005 | — |
| S `vis_subset0` | 67 | 0.39 | 0.43 | 0.40 | — | — |

(0.0005 = 1/2001, the floor.)

**Panel c is `vis_subset0`, not `stim_res`.** `stim_res` shows a large
stim-locked transient (peak ~1.85) with a significant mean distance. The
published S panel (flat ~0.8, null spanning 0.6–1.2, n.s.) matches only
`concat_block_pairs_act_vis_normFalse_shuffleTrue_vis_subset0.npy`.
Plots are in `<dmn>/res/fig45/old_crossval_S/`. Out of 40 candidate files
scanned, this was the only match.

`vis_subset0` is a hard-coded 67-uuid list in `dmn_ari.get_uuids_raster_types`:
LP 31, LGd 13, VISp 12, VISam 4, NOT 4, VISpm 3, across 40 insertions. It
appears to be hand-picked from the visual-region rastermap (notebook cells
97–103: vis regions → top stim-L/R-difference cells with small choice
difference → `vis_subset` 1,645 cells → `vis_subset0`). The exact
selection step is not in code.

### New stack `concat_cvTrue.npy`

54,569 cells × 1,872 bins, 21 trial types (`dmn_new.tts__`: rest, L_b, R_b,
quies, change_b,s, 4 correct stim×block types, mistake_s, m, change_b,m,
4 move types, mistake_m, L/R move, fback1/0). Odd/even CV split
(`concat_z_train` = odd = rastermap fit; `concat_z` = even). Stored `isort`
and `rm_labels` (100 clusters, monotonic along `isort`). It is the same stack
`bwm_dmn/fig5.py --data new` uses; there, stim/integ = clusters 22–28
(`bwm_dmn/FIG5_CLUSTERS.md`).

Panel-a difference: the new concat has only 4 correct stim×block columns
per alignment and pools mistakes, so it cannot show the 8 + 8 s×c×b
columns of the published raster. `fig45_panel_a.py` draws quies → R_move
from the new stack (`<dmn>/res/fig45/cvTrue/plots/panel_a_raster_test.png`).

**Old types → new clusters** (by uuid; 82–83 % of each old set is in the
new stack). `fig45_define_cell_types.py overlap` writes
`old_type_overlap_per_cluster.csv`, an enrichment heatmap, and train/test
cluster rasters (`cluster_id/`):

| old type | old cells in new stack | clusters dominated by it | notes |
|---|---|---|---|
| stim_res | 558 | 26, 27, 28, 29 (30 mixed) | FIG5 called 28 "stim" |
| integrator | 1228 | 23, 24, 25 (+ 20, 22, 30, 31 mixed with move_init) | FIG5 used 22–27 as integ |
| move_init | 3081 | 16–22, 31, 32, 38–42, 56 | |
| movement | 964 | 81–86 (+ 37–39 mixed) | |
| vis_subset0 | 52 | none; ≤ 2 cells per cluster | no cluster equivalent |

Candidate picks (integrator 23–25, stim_res 26–29, move_init 16–22, 31–32,
38–42, 56, movement 81–86) give 1,837 / 1,457 / 8,817 / 3,806 cells. They
touch 512 of the 515 insertions. **Not saved yet; picks need sign-off.**

### PETH recompute parity

`fig45_block_peths.py compute_insertion` on pid
`f083fcd5-e456-433d-8aa5-6ace0a7ac170` (8 old integrator cells), act prior:
the 16 trial-type lengths match, and the real PETHs match the 2025 stacked
rows to **3e-8** max abs (float32). That run took 56 s including a 231 MB
spike download into the laptop ONE cache.

Implementation differences from `stack_concat`, all equivalent in
expectation:

- Shuffles are computed with an assignment-matrix matmul.
- Only per-type Σ shuffle² is kept.
- AUDp is dropped from both the real and the control sums for stim_res.
  The original dropped it from the real sum only, but divided by the
  post-drop n.
- Per-cell `p_vals` are not computed.

Full run size: ~512 insertions × ~1 min, plus about 100 GB of spike
downloads. That suggests ORCD, or a sharded laptop run with
`--cache-dir` on a big disk. Not run.

### Commands

```bash
conda activate iblenv   # outside sandbox
python scripts/fig45_plot_distances.py --tag old_crossval                 # 2025 b/e
python scripts/fig45_plot_distances.py --vers concat_block_pairs_act_vis \
    --types vis_subset0 --tag old_crossval_S                              # 2025 c
python scripts/fig45_define_cell_types.py overlap
python scripts/fig45_define_cell_types.py save --pick integrator=23-25 \
    stim_res=26-29 move_init=16-22,31-32,38-42,56 movement=81-86          # after sign-off
python scripts/fig45_panel_a.py
python scripts/fig45_block_peths.py compute --shard 0 --n-shards 1
python scripts/fig45_block_peths.py stack
python scripts/fig45_plot_distances.py \
    --root ~/Downloads/ONE/openalyx.internationalbrainlab.org/dmn/res/fig45/cvTrue --tag act
```

### Open decisions

1. Cluster picks for C / M on the new sort (table above). Should C include
   the mixed clusters 22 and 30? FIG5 used 22–27.
2. What to use for S on the new data:
   - the 52 surviving `vis_subset0` uuids (same cells as published);
   - the stim_res clusters 26–29 (which, in the old data, give a
     significant transient, unlike the published panel); or
   - redoing the vis-region top-stim-difference selection on the new
     stack.
3. Where to run the 512-insertion recompute: ORCD or the laptop.
4. Prior: the figure is `act` (α = 0.2). `--prior true` is wired in but
   not requested.

## 2026-10-13b — notebook C/M picks; S options; ORCD path on the insertion cache

### What the notebook's "id sequence (and other) cells" actually used

`bwm_dmn.ipynb` cells 13–30 (both the `bwm_dmn` and paper-brain-wide-map
copies) run on the **old** 53,021-cell stack
(`rm_concat_cvTrue_ephysFalse_nclusrm100_zsc0.npy`), not on
`concat_cvTrue.npy`:

| group | old clusters (cell 27) |
|---|---|
| stim | 4 |
| integrator | 38, 1–3 |
| move | 72–75 |
| move init | 34–37, 41, 42, 10, 32, 39, 61 |

The only by-eye picks on the new file are from the 2026-10-08 Figure 5
rebuild in `bwm_dmn` (`FIG5_CLUSTERS.md`): sequences, and stim/integ
**22–28** (28 ≈ old stim, 60% of old stim cells; 22–27 ≈ old integrator,
68% of old integ cells). **No move / move-init pick was made on the new
file.** `fig5.py --data new` skips the composition panels for that reason.

So for Fig 4.5 on the new stack:

- **C (integrator)** = clusters 22–27 (by eye, FIG5).
- **M (move_init + movement)** = not picked. The uuid-overlap candidates
  (move_init 17–21, 31–32, 38–40, 42, 56; movement 81–83; cluster 22 is
  shared with C) are a starting point only. Train/test rasters for picking
  are in `dmn/res/fig45/cvTrue/cluster_id/`.

### S: how panel c's cells were chosen, and what that means on the new stack

Panel c's `vis_subset0` (67 cells) matches notebook cell 96. The pool was
old `stim_res ∪ integrator` restricted to `VISpm, VISam, FRP, VISp, VISli,
LGd, LP, NOT`. Ranking used `sort_by_diff_two_conditions` on `concat_z`
with `mean_baseline=True` and `window_len=45`, keeping the top 200 per
side for each of the 4 stimulus pairs `stim_pairs[:4]`. That selection is
not cluster-based.

On the new stack:

- `concat_cvTrue.npy` has only the 4 correct-trial stim types
  (`stimLbLcL`, `stimLbRcL`, `stimRbRcR`, `stimRbLcR`, 72 bins). Two of the
  four comparisons in `stim_pairs[:4]` need error-trial types it lacks, so
  the cell-96 ranking cannot be rerun on the stack itself. It needs the
  16-type block PETHs that the ORCD run produces.
- Visual regions ∩ clusters 22–28 = **127 cells** (LP 55, VISp 29, LGd 17,
  VISam 13, VISli 7, …). Cluster 28 has only **5** visual cells of 353, so
  "stim cluster" and "visual stim cells" are different populations on the
  new sort.
- Of the 67 `vis_subset0` uuids, 52 are in the new stack, scattered across
  clusters.

S is **not decided** (to discuss). Because extra types share the same
shuffles, any set of S candidates can go in one ORCD pass. A two-stage run
(stage 1 includes the 127-cell pool; rank locally; stage 2 runs S-only
insertions) costs minutes.

### ORCD path: reuse `manifold/insertion_cache`

`fig45_block_peths.py compute` now reads
`<ONE_CACHE_DIR>/manifold/insertion_cache/{eid}_{probe}.npy` (built by
`block_analysis_allsplits.build_insertion_cache`, already full-BWM on ORCD)
plus `bwm_tables/trials.pqt`. ONE is used only on a cache miss.

- **Trials.** The cache keeps the masked trials for the 3
  `SATURATION_TYPES`, with the original trial index. Their index
  intersection is base & ~(those 3). The other 3 Fig 4.5 saturation keys
  are applied by row from `trials.pqt`, the same way `load_trials_and_mask`
  does it. The action-kernel prior is run over the full session choice
  sequence from `trials.pqt`.
- **Cells.** The cache has `cluster_id` but no uuids.
  `fig45_define_cell_types.py save` now stores `cluster_id` per typed cell
  (from openalyx `bwm_tables/clusters.pqt`) and `insertions: pid →
  (eid, probe)` (from the `bwm_query` freeze). ORCD needs no Alyx or
  `clusters.pqt`.
- **Binning.** Spikes are filtered to the typed clusters before
  `bin_spikes2D`, giving the same counts as binning all cells.
- **Writes.** Per-insertion files are written atomically (tmp +
  `replace`). The ONE fallback now uses the same row-aligned saturation
  mask, so it no longer fails `load_trials_and_mask`'s length assert on
  truncated sessions.

**Parity (pid `8dfb86c8…`, 14 old-type cells, nrand=2000, alyx cache):**

- Cache vs ONE: uuids, trial counts, PETHs and all control sums are
  **bit-identical**.
- Cache vs 2025 stacked rows: ≤ 6e-8 max abs, which is float32 precision.
- This session is truncated relative to `trials.pqt`. The old
  `load_trials_and_mask(saturation_intervals=…)` asserts on it, yet the 2025
  run included it, so the row-aligned mask is what the 2025 run did.
- Load time: 0 s from the cache vs 9 s from ONE (spikes already local).
  Compute time was 1 s for 14 cells.

Scripts: `scripts/run_fig45_peths_slurm.sh` (`STEP=compute|stack`) and
`scripts/submit_fig45_peths_sharded.sh`. Defaults: 8 shards, 6G, then one
stack + plot job (afterok). Output goes to
`$ONE_CACHE_DIR/dmn/res/fig45/cvTrue`. Not submitted: `cell_types.npy`
needs the M and S picks first.

## 2026-10-13c — M = move + move_init + move&fback, mapped from the notebook groups

Notebook cell 27 defines three movement groups on the old 53,021-cell
stack:

| group | old clusters |
|---|---|
| move | 72–75 |
| move_init | 34–37, 41, 42, 10, 32, 39, 61 |
| move&fback (`mfb`) | 0, 62, 63, 67, 68, 70, 71, 76, 77, 83–88, 90, 96 |

Each old cell was assigned its cell-27 group. The old stack rows line up
with `rm_concat_cvTrue_ephysFalse_nclusrm100_zsc0.npy`. Groups were then
counted by uuid in each new `concat_cvTrue` cluster.

**Rule:** a new cluster goes to M if its plurality old group is move,
move_init or mfb, or if those three together make up ≥ 40% of its matched
cells. C clusters 22–28 are excluded. Each cluster is labelled by its
dominant M subgroup.

| type | new clusters | cells | insertions |
|---|---|---|---|
| C `integrator` | 22–27 (by eye, FIG5) | 3,279 | 432 |
| `move_init` | 17–21, 31–32, 40–41 | 5,449 | 486 |
| `move` | 38–39, 81 | 1,663 | 403 |
| `move_fback` | 14–15, 37, 56–57, 78–80, 82–83, 85–86, 89–92, 96 | 11,379 | 508 |
| M `move_all` (union) | 29 clusters | 18,491 | 512 |

Coverage of the old groups: move_init 59%, move 74%, mfb 69%.

Borderline clusters:

- **20:** plurality "other" (0.45 vs move_init 0.44). It looks like 17–21
  in the raster.
- **41:** move_init 0.25.
- **78, 91, 92:** fback is almost as common as mfb.
- **22:** move_init 0.33 vs int 0.32. Kept in C (by eye), not in M.

Written with `fig45_define_cell_types.py save --pick … --union
move_all=move_init+move+move_fback`. The output, `cell_types.npy` with
cluster_id and insertions, is provisional until reviewed.

Review figure (train | held-out, all 21 trial types, colour band per
type):
`dmn/res/fig45/cvTrue/plots/review_C_M_picks.{png,svg}`.

The ORCD defaults now run
`TYPES="integrator move_all move_init move move_fback"`.
`fig45_plot_distances` panel e uses `move_all` as M when it is present.

**Prior check.** PETHs and shuffles use the action-kernel prior
(α = 0.2, binarised 0.8/0.2): `--prior act` is the default and is now
passed explicitly as `PRIOR=act` in `run_fig45_peths_slurm.sh`. The
16 trial types are built from the AK block labels, and each shuffle
permutes those labels within a pair. On cached pid `c893c0a3…`, the AK
label differs from the true block on 39% of kept trials, and the per-type
trial counts change to match (for example `stimLbRcL` 37 vs 18). Unbiased
trials (true pL = 0.5) were not dropped. `exclude_unbiased` is off, so a
true-0.5 trial that passed the base mask stayed in and took an act label.
Corrected 2026-10-19.

### S — open (suggestions in chat, not implemented)

The aim is no longer the old "early visual cells have no block signal".
Instead, follow the regional pipeline: 0–80 ms after stimOn, stratified
by stimulus only (no choice split), act prior, and find cells with
prior modulation.

Forks to confirm:

1. Strata = stim side only, or stim side × contrast.
2. Per-cell, per-cluster, or regression (`get_var_partition` with choice
   dropped) statistic.
3. Raw, or pre-stim-baseline-subtracted.
4. A held-out selection split (odd/even trials) so that the S panel is not
   circular.

## 2026-10-13d — redesign: stim / choice / act-prior sensitivity, stratified shuffles

### How S was defined before

- **Fig 4.5:** every type is a population filter, followed by a population
  block-distance test against block-label shuffles within pairs.
  - stim_res / C / M are index ranges along the old isort.
  - `vis_subset0` (cell 96) is the top-N cells ranked by mean stim L−R
    difference, with no significance test.
- **Per-cell tests existed** (`dmn_avgbystim.calculate_p_value`) but did not
  feed the figure.
  - Statistic: per-cell mean |Δ| over the 4 stim pairs (stim L vs R at
    fixed block × choice), with per-cell label shuffles.
  - Cell 104 applied BH to these and kept stim-sig ∧ choice-non-sig cells
    as `topstim3` (124 cells).

### New design (user, 10-13)

All analyses:

- act-kernel block labels (α 0.2, binarised);
- the six Fig 4.5 saturation keys;
- the same trials throughout (no odd/even split);
- pairs gated at ≥ 5 trials per side, the regional `min_trials_per_side`.

| analysis | window | contrast | strata (one pair each) |
|---|---|---|---|
| `S_stim` | stimOn 0–80 ms | stim L vs R | act block |
| `S_prior` | stimOn 0–80 ms | act block L vs R | stim side (contrast pooled) |
| `choice_stim` | stimOn 0–150 ms | choice L vs R | stim × act block |
| `choice_move` | move −150–0 ms | choice L vs R | stim × act block |
| `prior_stim` | stimOn 0–150 ms | act block L vs R | stim × choice |
| `prior_move` | move −150–0 ms | act block L vs R | stim × choice |

Windows use the regional binning: 12.5 ms bins, 2 ms stride,
`ceil(post / T_BIN)` bins. So "0–80 ms" has 7 bins, with bin centres up to
about 91 ms; this is the same as the regional `[0, 0.08]` `pre_post`.

**Shuffles and statistics:**

- Null: the contrast label is permuted inside each pair, so the shuffle is
  stratified by the other factors. nrand = 2000.
- Population statistic: Fig 4.5 normalisation, sqrt(Σ Δ² / T_BIN / n) per
  pair, averaged over pairs, with `distance_stats` p_mean / p_gain. Here n
  is per pair, because the trial gate can drop a pair in some insertions.
- Per-cell statistic: mean over bins of Δ², averaged over valid pairs;
  p = (1 + #null ≥ obs) / (1 + nrand).

**Workflow** (`scripts/fig45_sensitivity.py`):

- **Stage A:**
  - `S_*` on the stim-res + C pool `stimint` (clusters 22–28; no AUDp
    exclusion, see below): per cluster `rm22`–`rm28`, the whole pool, and per-cell
    `S_stim`.
  - `choice_*` / `prior_*` on `integrator`, `move_all`, `move_init`,
    `move`, `move_fback`: population (the "population" mode), plus
    per-cell `choice_*`.
- **select:**
  - `S_cell` = BH(0.05) over pool cells on `S_stim`.
  - `S_clus` = clusters with BH(0.05) over the 7 cluster `S_stim` p_mean.
  - `C_choice` / `M_choice` = BH(0.05) over the choice p-values of both
    windows jointly; a cell is kept if significant in either window
    (the "filter" mode).
- **Stage B:** `S_prior` on `S_cell` / `S_clus`; `prior_*` and `choice_*`
  on `C_choice` / `M_choice`.
- **Comparison:** C/M filter mode (stage B) vs population mode (stage A).

**Implementation notes:**

- Population sums are additive, so each shard keeps one accumulator,
  checkpointed every 5 insertions (requeue-safe). Per-insertion files hold
  only per-cell statistics.
- ORCD scripts: `run_fig45_sensitivity_slurm.sh` and
  `submit_fig45_sensitivity.sh`, which chain A (8 shards) → stack A →
  select → B (4 shards) → stack B. Output:
  `$ONE_CACHE_DIR/dmn/res/fig45/cvTrue/sensitivity/`.
- `cell_types.npy` re-saved with `stimint` and `rm22`–`rm28` added.

**Smoke test** (6 cached alyx insertions, nrand 200, `/tmp`):

- Stage A ran at 1–2 s per insertion.
- With 6 insertions nothing passes BH, which is expected.
- With select `--q 1`, stage B's `S_clus` real `S_prior` distances equal
  stage A `stimint` exactly; only the shuffle draws differ (the seed
  includes the stage).

**Caveat:** with nrand = 2000 the minimum per-cell p is 5e-4. BH over the
3,632-cell pool can therefore select only if at least about 36 cells reach
that minimum. Raise nrand for the per-cell stage if `S_cell` comes back
empty.

**10-13d follow-up — no AUDp exclusion.** `stack_concat` dropped AUDp from
stim_res (`dmn_avgbystim.py`, `raster_type == 'stim_res'`) without a
recorded reason. The likely reason is the go-cue tone at stimOn.
That tone is identical on L and R trials, so it cannot drive the
stim L−R or prior contrasts here. `fig45_sensitivity.py` keeps all regions.
The Fig 4.5 replication script `fig45_block_peths.py` still drops AUDp from
`stim_res`, to match the published S panel.

## 2026-10-13e — slimmer stage B; population-only comparison figure

**Change (user, 10-13):** no prior tests on individual cells or clusters, and no
M-subgroup populations. Prior is tested only on the selected populations.

**Stage A** (`fig45_sensitivity.py`, `STAGES['A']`):

- `S_stim` on the `stimint` pool (22–28), on each of `rm22`–`rm28`, and per cell.
  This is the input to select.
- `S_prior` on the whole `stimint` pool only.
- `choice_*` (population and per cell) and `prior_*` (population only), on
  `integrator` (C) and `move_all` (M) only. `move_init` / `move` /
  `move_fback` are dropped.

**select:** unchanged (`S_cell`, `S_clus`, `C_choice`, `M_choice`).

**Stage B** (`STAGES['B']`): population tests only.

- `S_stim` / `S_prior` on `S_cell` and `S_clus`.
- `choice_*` / `prior_*` on `C_choice` and `M_choice`.

**`compare` step:** writes `<out>/compare/compare_{S,CM}.{svg,png}` plus `summary.json`.
It runs automatically after stack B on ORCD.

- Each trace is d − shuffle mean, with the shuffle 2.5–97.5 % band.
  The legend gives p_mean and the median number of cells per pair.
- **S figure:** stim and act-prior sensitivity, 0–80 ms. Traces: S pool,
  S per cell, S per cluster.
- **C/M figure:** choice and act-prior sensitivity, stim 0–150 ms and move
  −150–0 ms. Traces: C all vs C choice-selected, M all vs M choice-selected.

**Smoke test** (6 insertions, nrand 200, select `--q 1`): the full chain ran.
At q = 1 the selected sets are almost the whole pools, so their traces overlap
those of the pools. M choice has p = 0.005 (the floor) in both windows; nothing
else is significant on 6 sessions.

**10-13e follow-up — cell types in the repo.** `cell_types.npy` (8.9 MB) and
`cell_types_picks.json` moved from `dmn/res/fig45/cvTrue/` to git-tracked
`data/fig45/`. They are built from the local-only `concat_cvTrue.npy`.
All fig45 scripts and the ORCD job scripts default to
`$REPO_DIR/data/fig45/cell_types.npy`, so no scp is needed.

## 2026-10-13f — prior offset and gain on the compare populations

`compare` now scores offset and gain on the act-prior curves (S prior, C/M prior stim, C/M prior move). The distance traces are averaged onto the 12.5 ms bins (mean of the 6 sub-bins) so the windows match the regional test.

- **Offset:** mean of the first 5 stim-aligned bins (0–62.5 ms) minus the shuffle mean. Significant if that early mean beats the shuffles at p < 0.05.
- **Gain:** if the offset is significant it is subtracted, then the mean of the later bins (from 50 ms stim-aligned, or from −100 ms move-aligned) is tested against the shuffles. The move test uses the stim-aligned offset of the same population.
- **Gain size:** the peak of that later window minus the shuffle mean at that time. The p-value is the late-window mean test, not a test of the peak itself.

Numbers are in `manifold/res/fig45/cvTrue/sensitivity/compare/summary.json` under `offset_gain`. Every prior offset is significant at the 2000-shuffle floor (p = 0.0005). Gain fails for S per cell (p = 0.098) and for choice-selected C before movement (p = 0.112). S pool and S per cluster gain are p ≈ 0.004, significant at 0.05 and not at 0.01. S per cluster kept all 7 clusters, so it is the same population as the S pool.

## 2026-10-13g — Figure 4.5 panels b, c, e with S per cell, C all, M all

`fig45_sensitivity.py fig45` draws the block-distance panels from the sensitivity
accumulators into `manifold/res/fig45/cvTrue/sensitivity/fig45/`.

- **b.** C all and M all. Act-block L vs R within stim × choice, stim 0–150 ms and move −150–0 ms.
- **c.** S per cell (54 cells). Act-block L vs R within stim side, 0–80 ms. This is the S prior test, not the 150 ms stim × choice protocol used for C and M.
- **e.** C vs M, distance minus shuffle mean. Amplitude ratio M/C is 0.80 (stim) and 0.89 (move).

Dashed curve is the raw distance. The blue segment is the first 62.5 ms. The solid curve is the raw distance minus the offset excess, drawn because every offset is significant. The inset is the distance minus the shuffle mean. p_mean is at the 2000-shuffle floor for all five panels. S per cell p_gain is 0.098; the other gains are at or under 0.001.

## 2026-10-14 — Figure 4.5 redrawn in the published layout

The composite `sensitivity/fig45/fig45_Spercell_Call_Mall.{svg,png}` follows the original panel arrangement: **a** full raster plus the M/C zoom, **b** C and M in a 2×2 (stim, then move), **c** S, **d** the P–S–C–M schematic, **e** C vs M with the shuffle mean removed.

Each distance panel has the shuffle cloud, a dashed raw curve, a solid offset-removed curve, an inset of the shuffle-subtracted trace, and a side histogram. A star marks a test with p < 0.05. On the schematic the left mark of each pair is the offset and the right mark is the gain. S is a check then a cross (offset significant, gain not). C and M are two checks.

**10-14 follow-up — figure details.** Panel proportions follow the original (a about as wide as b; the top block is two rows tall). The blue offset mark, its histogram, and its average line use the first 5 samples. Histogram reference lines end in a short vertical cap, including the lower one, and the left spine of the histogram is kept. Inset frames are off. Panel d is left blank. Panels b, c, and e use at most five ticks, at the original label size.

**10-14b — figure folder moved.** The figure folder moved from `manifold/res/fig45/cvTrue/sensitivity/fig45/` to openalyx `manifold/paper_figs/fig45/`, next to the other main figures ([paper main figures](paper_main_figures.md)). `fig45_sensitivity.py fig45` now writes there by default (`--fig-dir`). The accumulators (`stageA`, `stageB`, `compare/`) stay in `sensitivity/`. Rerun to regenerate the composite in the new folder:

```bash
python scripts/fig45_sensitivity.py fig45 \
    --out ~/Downloads/ONE/openalyx.internationalbrainlab.org/manifold/res/fig45/cvTrue/sensitivity
```

## 2026-10-19 — trial table matches the standard splits

The loader in `fig45_block_peths.py` (shared with `fig45_sensitivity.py`) now follows the standard order.

1. α=0.2 kernel on the full choice sequence. Bad trials and true `probabilityLeft == 0.5` stay in the EMA. On the cache path that sequence is `trials.pqt`; on a ONE miss it is the `load_trials_and_mask` table with `truncate_to_pass` left on.
2. Bad-trial mask, then one saturation key for the alignment. stimOn uses `saturation_stim_plus04`. firstMovement uses `saturation_move_minus02`. The other four Figure 4.5 keys are not applied.
3. Drop trials whose true `probabilityLeft` is 0.5, using the saved true block. The act label is written on after that, so the drop is not a filter on the overwritten column.

Finished `dmn/res/fig45` and `manifold/res/fig45` files were built with all six saturation keys and with true-0.5 trials left in the pools. They are not this trial table.

## 2026-11-02 — standard trial table rerun (Oct 20) and S cells at uncorrected p < 0.01

**Oct 20 run, not journaled until now.** The full sensitivity chain was rerun on ORCD on the 10-19 trial table (commit `587c3d4` or later) with nrand 2000. A local copy is in alyx `dmn/res/sensitivity_standard/` (stageA, stageB, compare, fig45). Of 513 insertions, Stage A finished all of them. BH at q 0.05 now keeps **0** S cells (54 on the old table). 34 cells sit at the 5e-4 floor, and BH would need about 36. So panel c of that figure is the whole S pool (clusters 22–28, 3,624 cells): offset 0.10 (p 0.0005), gain 0.042 (p 0.008). C_choice is 129 and M_choice 1,461 (160 / 1,688 before). C and M prior stay significant in both windows. M/C amplitude is 0.75 (stim) and 0.95 (move).

**S cells without BH (user, 11-02).** Uncorrected S_stim p < 0.01 keeps **119** of 3,632 pool cells across 71 insertions; chance alone predicts about 36. Other thresholds: p < 0.05 → 343 (182 expected), p < 0.005 → 82 (18), p < 0.001 → 49 (4).

- **Code** (`ebadd93` develop, `bcddf95` main):
  - `select --s-cell-p P` sets S_cell to cells with uncorrected p < P. BH is still the default, and the rule is written to `stageB_selection.json`.
  - New stage `BS` runs S_stim / S_prior on S_cell only, so the other stage-B populations are not recomputed.
  - `fig45` takes S from BS when it is present.
  - Submit with `S_ONLY=1 CELL_TYPES_B=...`.
  - The selection is in the repo as `data/fig45/stageB_cell_types_Sp01.npy`.
- **Run:** ORCD, 4 shards. All 71 insertions done, 0 failures. Locally run with `SLURM_JOB_ID=test BLOCK_ALYX=1`, it made 0 HTTP requests. Output: alyx `dmn/res/sensitivity_standard_Sp01/stageBS/`. stageA and stageB there are symlinks to `sensitivity_standard/`.

S prior, stimOn 0–80 ms, act block L vs R within stim side:

| S population | cells | offset (p) | gain (p) |
|---|---|---|---|
| old table, BH | 53 | 0.52 (0.0005) | 0.082 (0.098) |
| standard table, whole pool | 3,624 | 0.10 (0.0005) | 0.042 (0.008) |
| standard table, p < 0.01 | 119 | 0.33 (0.0005) | 0.117 (0.0035) |

On the p < 0.01 set, both offset and gain are significant, so the S mark on the schematic would be ✓ / ✓. Panels b and e are the same as the Oct 20 run.

**Caveats.**
- About 30% of the 119 are expected false positives.
- Selection uses the same trials as the prior test; the contrasts differ (stim side vs block within stim side).

**Figures:**
- res_standard (p < 0.01 S): `<oa>/manifold/paper_figs/fig45/res_standard/fig45_Spercell_Call_Mall.{svg,png}` and `summary.json`. The same figure is also in `sensitivity_standard_Sp01/fig45/`.
- Whole-pool S version: `sensitivity_standard/fig45/`.
- `paper_figs/fig45/` itself is still the Oct 14 old-table figure.

```bash
python scripts/fig45_sensitivity.py fig45 \
    --out ~/Downloads/ONE/alyx.internationalbrainlab.org/dmn/res/sensitivity_standard_Sp01 \
    --fig-dir ~/Downloads/ONE/openalyx.internationalbrainlab.org/manifold/paper_figs/fig45/res_standard
```

## 2026-11-02b — Figure 4 styling; choice sensitivity of the C / M rastermap clusters

**Restyle (develop `38d5f08`, main `3220f4b`).** `fig45_sensitivity.py fig45` now matches Figure 4.

- **Layout and fonts:** the 759 pt page width with a point grid. Helvetica throughout: 11 pt ticks, 14 pt "time (ms)" labels, 26 pt panel letters. y labels are Times and are placed clear of the tick labels.
- **Circles and ticks:** S/C/M are drawn as Figure 4's filled circles. Time ticks are 0/50/100/150 and −150/−100/−50/0.
- **Panel a:**
  - Drawn from `dmn/res/fig45/cvTrue/plots/panel_a_test.npz`, which `fig45_panel_a.py` writes (rows binned to 2,000).
  - It used to show 1 px tick marks, with no S and M in three colours. It now has solid S (clusters 22–28), C and M strips.
- **No overlaps:** each distance panel keeps its top 30% clear for the circle and the inset. Each draw checks the full axis extents and writes the result to `summary.json` as `layout_overlaps` (empty).
- Redrawn: `paper_figs/fig45/res_standard/` and `sensitivity_standard_Sp01/fig45/`.

**Choice sensitivity of the C / M clusters (user, 11-02).**

- **Contrast:** choice L vs R within stim × act block. stimOn 0–150 ms and move −150–0 ms, standard trial table, nrand 2000.
- **Trace:** distance minus the shuffle mean, with the 2.5–97.5 % shuffle band.
- **Code:**
  - `cluster-types` builds one type per rastermap cluster from the concat stack: 35 clusters (C 22–27; M = the 29 `move_all` clusters), 296–1,243 cells each, saved in `data/fig45/cell_types_cm_clusters.npy`.
  - Stage `CL` runs choice_* on each cluster.
  - `choice-clusters` draws the plots.
  - Submit with `ONLY_STAGE=CL`.
- **Groups — plotted.** Stage A of the Oct 20 run, C (`integrator`) vs M (`move_all`) overlaid. Output: alyx `dmn/res/sensitivity_standard_clusters/choice_clusters/choice_C_vs_M.{svg,png}`.
  - Both are at p_mean 0.0005 in both windows.
  - The two are about the same up to roughly 60 ms after stimOn. After that M rises above C: about 0.43 vs 0.19 at 150 ms, and about 0.65 vs 0.25 at movement onset.
- **Per cluster — code ready, not yet run.**
  - Each cluster will get its own figure: stim and move panels, with its C or M circle. Output: `choice_clusters/clusters/choice_cluster<k>_<C|M>`.
  - Needs stage CL on ORCD over the same 513 insertions as Stage A. A local check (3 cached insertions plus 1 uncached, Alyx blocked) made 0 HTTP requests.
  - Then copy `stageCL/` into `sensitivity_standard_clusters/` and rerun `choice-clusters`.

```bash
ONLY_STAGE=CL OUT=$ONE_CACHE_DIR/dmn/res/fig45/cvTrue/sensitivity_standard_clusters \
  CELL_TYPES_B=$PWD/data/fig45/cell_types_cm_clusters.npy bash scripts/submit_fig45_sensitivity.sh
```

**Fix.** `compare` loads only the stages that exist. After stage BS was added, stack B in a full chain would have crashed on the missing `stageBS`.

## 2026-11-03 — per-cluster choice (stage CL); choice-selected C / M cells (stage CS); panel e on the selected cells

**Runs.** Both ran on ORCD with nrand 2000, on the standard trial table (Oct 20 Stage A). No failures in either.

| stage | insertions | output |
|---|---|---|
| CL | 513 / 513 | alyx `dmn/res/sensitivity_standard_clusters/stageCL/` |
| CS | 378 / 378 | alyx `dmn/res/sensitivity_standard_cmsel/stageCS/` |

**Per-cluster choice (CL).** One figure per cluster in `sensitivity_standard_clusters/choice_clusters/clusters/choice_cluster<k>_<C|M>`.
- 34 of the 35 C / M clusters are choice-significant in both windows. Most are at the 0.0005 floor; M clusters 17, 81 and 90 have stim-window p of 0.004–0.013.
- The exception is **M cluster 80** (about 131 cells/pair): p 0.99 stim, 0.89 move. It has no choice signal.
- End peak and slope were not computed per cluster (user asked for groups only).

**Choice-selected cells (user, 11-03; options confirmed).**
- **Selection:** per-cell choice tests from Stage A (choice L vs R within stim × act block), uncorrected p < 0.01.
  - Figure 2's BH 0.01 keeps no cells: the per-cell p floor at nrand 2000 is 5e-4.
  - **C_sel** = C clusters 22–27 with p < 0.01 in the stim (0–150 ms) or move (−150–0 ms) window: 325 of 3,271 cells, about 65 expected by chance.
  - **M_sel** = the 29 M clusters with p < 0.01 in the move window only: 1,885 of 18,418, about 184 expected by chance.
  - Selection file: `data/fig45/cell_types_cm_choice_p01.npy` (`select-cm --p 0.01`).
- **Stage CS:** choice_* and prior_* on both sets.
- **Choice shape:** Figure 2's `compute_amp_slope` definitions on the raw choice distance, values only (user). These are per-cell-normalised units, so Figure 2's regional slope cuts do not apply.

| set | window | cells/pair | end peak | late slope (20 bins) | overall slope /s | peak |
|---|---|---|---|---|---|---|
| C_sel | stim | 280 | no | 0.0006 | 7.7 | 152 ms |
| C_sel | move | 281 | no | −0.0041 | 7.9 | −15 ms |
| M_sel | stim | 1,547 | yes | 0.0383 | 13.5 | 154 ms |
| M_sel | move | 1,556 | yes | 0.0603 | 19.6 | 4 ms |
| C all | stim / move | 2,765 / 2,771 | no / no | −0.0018 / 0.0014 | 2.7 / 2.8 | 119 / −9 ms |
| M all | stim / move | 14,265 / 14,334 | yes / yes | 0.0154 / 0.0224 | 4.1 / 5.6 | 154 / 4 ms |

- **C_sel** has the integrator shape: an early rise, then a plateau; around movement it peaks about 15 ms before onset and dips.
- **M_sel** has the movement shape: it ramps into movement with an end peak in both windows.
- In the move window C_sel is above M_sel until about −70 ms.

**Prior on the selected cells.** Offset and gain as in panel b, with the move offset taken from stim. Every p is at or near the floor.

| set | offset (p) | gain stim (p) | gain move (p) |
|---|---|---|---|
| C_sel | 0.22 (0.0005) | 0.12 (0.001) | 0.14 (0.0005) |
| M_sel | 0.13 (0.0005) | 0.18 (0.0005) | 0.23 (0.0005) |
| C all | 0.06 (0.0005) | 0.036 (0.006) | 0.039 (0.001) |
| M all | 0.05 (0.0005) | 0.069 (0.0005) | 0.068 (0.0005) |

- The selected cells carry about 3–4× the prior signal of their clusters.
- C_sel has the larger offset (early prior, by about 40 ms after stimOn).
- M_sel has the larger gain (builds toward movement).

Figures: `sensitivity_standard_cmsel/choice_selected/{choice,prior}_Csel_vs_Msel.{svg,png}` and `summary.json` (`choice-selected`).

**Panel e on the selected cells.**
- `panel-e-sel` draws Figure 4.5 e with C_sel / M_sel in place of C all / M all. S is the 119-cell p < 0.01 S set from stage BS: `sensitivity_standard_cmsel/stageBS` is a symlink to `sensitivity_standard_Sp01/stageBS`.
- Output: `sensitivity_standard_cmsel/fig45/panel_e_Csel_Msel.{svg,png,json}`.
- Amplitudes (max − min of distance − shuffle mean):
  - stim: S 0.38, C 0.28, M 0.25 (M/C 0.89);
  - move: C 0.32, M 0.22 (M/C 0.69).
  - With all cells: C 0.115 / M 0.086 (stim) and C 0.084 / M 0.080 (move).
- With the selected cells, C ≥ M on both sides. On the move side, M rises steadily to movement, while C is higher and noisier through the window.
- Panel e code is now `_draw_panel_e`, shared with `fig45`. Redrawing res_standard reproduced its amplitudes exactly.

**Note.** S on the stim side of panel e (`grp = 'S,C,M'`) was a working-tree edit that went into `d5edeab` / `c50d435` with the stage CS commit.

## 2026-11-05 — rastermap clusters as regions (stage RM): standard splits per cluster, S / C / M selection

**Aim (user, 11-04).** Treat each of the 100 rastermap clusters of `concat_cvTrue` (test set, 54,569 cells, 515 insertions) as a region. Run the six standard analyses on each and select S / C / M clusters with the Figure 2 region rules.

**Setup.**
- `rm-types` writes `data/fig45/cell_types_rm100.npy` (types `rm0`–`rm99`).
- Stage RM runs the six ANALYSES (S_stim, S_prior, choice_stim, choice_move, prior_stim, prior_move) on the standard trial table, nrand 2000. It shards over clusters (20 shards × 5 clusters), each shard running every insertion of its clusters.
- `select-rm` applies the Figure 2 rules to the per-cluster population curves, with BH over the 100 clusters at 0.01:
  - S = 80 ms stim significant, minus an onset confound (offset only) and early choice (run of ≥ 5 bins with p ≤ 0.01 at t ≤ 80 ms);
  - C = choice significant and rising, minus movement;
  - M = end peak, late (20-bin) slope above the cut, and choice significant around movement. Default cut is `late_mean` (mean over choice-significant clusters, 0.0114).
- `rm-report` draws the overview and the S / C / M group curves (per-cluster accumulators summed).
- Run on ORCD `mit_preemptable`, 11-04. Output: alyx `dmn/res/sensitivity_standard_rm100/stageRM/`, selection and figures in `select_rm/`.

**Save race (fixed 11-05).**
- All RM shards wrote `cells/{pid}.npy` through the same temporary name. Concurrent saves failed, and the failure came after the insertion had been added to the sums, so the done lists hold only 6,483 of 9,389 shard × insertion entries.
- **The sums are complete and not double-counted:**
  - no shard's done list is out of order, so no requeue reprocessed an insertion;
  - stage CL counts and distances equal RM exactly on all 35 shared clusters, and S_stim rm22–28 equals stage A;
  - all 515 insertions and 54,569 cells are present. Against the trial counts, no count exceeds the expected number (2,000 cluster × pair entries); the median is 0.985 of it. The shortfall is cells silent in that stratum, which stage CL shows identically.
  - The clobbered cells files hold only trial counts (RM has no per-cell stats), so nothing was lost. **No rerun.**
- Fix: unique temp names in `save_atomic`, per-shard cells files in RM, and the cells save before `add_to_acc`.

**Selection (late_mean).**

| set | clusters | cells |
|---|---|---|
| S | 5, 8, 15, 20, 24–27, 30, 32, 37, 40, 58, 86, 89, 95 (16) | 9,709 |
| C | 41 | 24,842 |
| M | 18, 21, 22, 37–42, 56, 57, 81–83, 85, 86 (16) | 9,235 |
| none | 38 | 18,133 |

- Overlaps: S∩C 8, S∩M 3 (37, 40, 86).
- 30 clusters pass the 80 ms stim test; 9 fail the onset confound and 9 the early-choice cut. The early-choice cut removes the three strongest stim clusters (4, 33, 42; early choice runs of 11–15 bins). Choice is stratified by stim side × act block, not contrast, so part of that early choice may be contrast.
- Movement rule matters: C / M = 41 / 16 (late_mean), 53 / 4 (late_fixed 0.05), 27 / 30 (end_peak).
- Against the old picks: of the 29 old M clusters, 14 are M and 14 C; of old S pool 22–28, 4 are S.
- Prior with no S / C / M: clusters 9 and 10 have the largest S_prior excess (0.48, 0.37). Clusters 28, 70, 71, 97 are choice-significant but not rising.

**Prior significance (BH q ≤ 0.01).**

| group | any window | S_prior (0–80) | prior_stim (150) | prior_move (−150–0) | all three |
|---|---|---|---|---|---|
| S (16) | 15 | 15 | 14 | 14 | 14 |
| C (41) | 34 | 33 | 26 | 29 | 25 |
| M (16) | 16 | 16 | 13 | 16 | 13 |

- None: S rm5 (q 0.012–0.022); C rm44, 46, 54, 62, 92, 94, 98 (most q 0.012–0.06; rm54 0.6–0.7).

**Group curves (all S / C / M clusters, no prior gate).** Every p at the floor (0.0005).
- Choice excess, M vs C: 0.19 vs 0.12 (stim), 0.30 vs 0.17 (move).
- Prior offset (stim) is similar: S 0.055, C 0.057, M 0.047.
- Prior gain is about 2.5× larger in M: 0–80 ms S 0.042 / C 0.037 / M 0.083; stim S 0.041 / C 0.036 / M 0.107; move S 0.060 / C 0.045 / M 0.091.

**Prior gate in the regional Figure 4 (checked 11-05).** `fig4_plot.py` gates only S: panel c and the S curve of i average sensory ∩ prior 0–80 ms (5 regions). Panels e, g and the C / M curves of i average **all** C (85) and M (30) regions with no prior test. Only the strips (b, d, f, h) are prior-gated. The RM group curves above gate nothing, so they match the regional C / M but not S. Open: whether to gate (user asked 11-05).

**Figure 4.5 on the RM selection (user, 11-05: keep all groups ungated).**
- `fig45_panel_a.py --rm-select <stageRM>/select_rm` outlines the RM C / M clusters in panel a; `fig45-rm` draws the published layout with S / C / M = all clusters of each group (late_mean). The page code is now `_fig45_page`, shared with `fig45`; `fig45` on `sensitivity_standard_Sp01` reproduces its summary.json exactly.
- Output: `sensitivity_standard_rm100/stageRM/select_rm/fig45/fig45_rm_late_mean.{svg,png,json}`.
- Cells/pair: S 9,515; C 20,918 / 21,076; M 7,900 / 7,958. Every p_mean, p_offset and p_gain at the floor.
- Panel e amplitudes: stim S 0.079, C 0.052, M 0.135 (M/C 2.6); move C 0.050, M 0.104 (M/C 2.1). With the old picks (all cells) C and M were close (0.115 / 0.086 stim). On the cluster selection M is above C from about 90 ms after stim and in the last 50 ms before movement.
- **Revision (user, 11-05).** Panel e S is now the 150 ms prior_stim curve (stim × choice strata), the same analysis as C / M (amp 0.073; panel c keeps S_prior 0–80 ms). Panel a also outlines the S clusters in blue.
- **Panel a box rule (general, `TYPE_COLS`):** S and C boxes span every stim-aligned segment (change_b,s → mistake,s), M boxes every move-aligned segment (m → R move). `sensitivity_standard_Sp01/fig45` was redrawn with it (curves unchanged).
- **Layout (user, 11-05), `fig45-rm` only (`compact`):** panel a has no C + M zoom, a taller raster and the S / C / M circles stacked beside it; rows are tighter and the two halves of e are equal width (110 pt). The 8 S∩C clusters (15, 20, 24, 25, 30, 32, 89, 95) are boxed in green (`#7FC8A9`, Figure 2 `C_SENS_INTEG`); S-only blue, C-only yellow (`BOX_COLORS`). The published `fig45` layout is unchanged (summary.json identical).
- **Panel c (user, 11-05):** S in panel c is now prior_stim too (0–150 ms, stim × choice strata), so c and e show the same S curve (8,391 cells/pair; mean, offset and gain p at the floor). `_fig45_page(s_name=...)` sets c's window; the published `fig45` keeps S_prior 0–80 ms.
- **Sensitivity panel (user, 11-05), layout `sens` (`FIG45_LAYOUTS`):**
  - a is unchanged.
  - New b, stacked to the right of a: S_stim (0–80 ms) for S, then choice_stim and choice_move for C and M, as distance minus shuffle mean.
  - c is the old b and c together: S prior in the left column, then the C and M rows.
  - The d slot is left empty under S; e is stacked to the right of c.
  - The published layout is unchanged (summary.json identical).
- **Layout `sens` reworked (user, 11-05):**
  - a: the full-height raster down the left.
  - b: the three sensitivity panels in one row on the top right (S_stim narrower, as its window is 80 ms).
  - c: stim and move columns, with rows S, C, M. The move cell beside S is the empty d schematic slot.
  - e: stim and move halves, equal width (162 pt), below c.
  - No axes or labels overlap.

**S from stim sensitivity with choice held fixed (user, 11-05; options confirmed).**
- Every Figure 4.5 RM analysis so far used the act prior (α=0.2 kernel on the full session, binarised onto `probabilityLeft`, true-0.5 trials dropped after).
- New analysis `S_stim_cb`: stim-aligned, 0–150 ms, stim L vs R within choice × act prior (4 pairs). With choice fixed, one side of each pair is error trials, mostly low contrast, so contrast is not matched.
- Stage RS runs it on the 100 clusters (sharded over clusters like RM, same OUT). Local check, one insertion: 4 pairs of 72 bins; the error side has 16–35 trials per pair.
- Selection: `select-rm --s-test S_stim_cb` → `select_rm_S150cb/`. S = BH q ≤ 0.01 on S_stim_cb minus the onset confound; no early-choice cut, since choice is held fixed. C / M unchanged; default `select-rm` output reproduced (lists identical).
- Figure: `fig45-rm --s-test S_stim_cb` draws a separate variant in `select_rm_S150cb/fig45/`, with the S_stim_cb curve in panel b. Panel a needs `fig45_panel_a.py --rm-select <stageRM>/select_rm_S150cb` first.
- Submit (not yet sent): `ONLY_STAGE=RS N_SHARDS_B=20`, `mit_preemptable`, same OUT as RM. Dry run with stubbed sbatch: 20 shards and 1 stack job.
