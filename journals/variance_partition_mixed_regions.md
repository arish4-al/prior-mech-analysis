# Single-neuron variance partition in mixed stim×choice regions

**Scope:** the real-data encoding analysis that asks whether the **stimulus** component of early during-stim activity is **prior-modulated** in regions that carry both stimulus and choice signals — the target-set definition, the OLS variance partition, the full BWM results, and the neuron- and region-level nulls.

**Status:** descriptive results complete over the full BWM (13,394 neurons, 19 mixed regions); the neuron-level prior-shuffle null is complete (`nrand=2000`); the region-level mean-R² null is coded but needs an ORCD re-run to save per-draw arrays.

Sources: dated entries 2026-07-20 (Goal 1), 07-20c–i, 07-28, 07-28b, 07-28c.

---

## Goal

Re-examine regions with **both stimulus and choice** sensitivity in the **during-stim** window (act / action-kernel splits), including the early-stim block-only control (d^{stim,se′} = `stim_block_{l,r}_act`).

**Approach:** single-neuron OLS variance partition (stim / choice / prior / stim×prior) to test whether the **stimulus component** in mixed-selectivity regions is **prior-modulated**.

**Target set:** `has_stim ∧ has_choice`, where stim = a significant amplitude on `stim_duringstim_act` ∨ short (choice+prior) ∨ `stim_duringstim1_act` (block-only), and choice = a significant `choice_duringstim_act`.

---

## Target-set definition (SC region types)

### Region-type export (2026-07-20c)

Region list source: openalyx `get_sc_table` → alyx CSV (does not overwrite openalyx `meta/table_*.png`).

```bash
conda activate iblenv
python scripts/export_stimchoice_regtypes.py \
  --out-cache-dir ~/Downloads/ONE/alyx.internationalbrainlab.org --copy-table-png
```

**duringstim regtype @ α=0.01:** integrator (`0.5`) = **28**, stim (`0`) = **1** (VISpm), move (`1`) = **22**. Integrators include CP, GPe, MOs, VM, SNr, …

### SC table recreate + mixed target (2026-07-20d)

Recreated `alyx.../meta/table_stimchoice_act_regtype_p_mean_c_0.01.png` from the openalyx act combined splits (action-kernel prior). The openalyx original was copied as `…_openalyx_copy.png` for comparison.

All SC inputs are **act** splits:

- `stim_duringstim_act` / `choice_duringstim_act` and the duringchoice act counterparts
- early stim **choice+prior** control: `stim_duringstim_short_act`
- early stim **block/prior only** (the paper's d^{stim,se′}): `stim_duringstim1_act` = `stim_block_{l,r}_act` (not `act_block_duringstim_*`, which is prior distance)

Paper checks (α=0.01): **Σ>0.8** → 1 stim processor (VISpm); **Σ′>0.8** → **23** (1 + 22 early), matching the manuscript counts. Regtype: stim=1, early=22, integrator=28, move=22; stim|choice coding defined for **69/208** regions.

**Var-partition target (default `--target mixed`):** significant stim (`stim_s` ∨ `stim_se` ∨ `stim_se′`) **and** significant `choice_s` → **19** regions: BMA, CENT2, CP, CUL4 5, FN, GRN, IP, IRN, LING, MRN, PF, PGRN, PRNc, SCm, SIM, SNr, VCO, VPL, VeCB.

Note: only **9/28** duringstim "integrators" (regtype 0.5) have significant stim amplitudes — the rest are choice-significant without stim (e.g. GPe, VM, MOs). Early stim processors (VISp, LGd, …) have stim without choice and are excluded from `mixed`. CSV columns: `sigma_stim_s`, `sigma_stim_s_prime`, `stim_processor(_loose)`, `has_stim`, `has_choice`, `mixed_stim_choice`.

Alternative SC tables built on different choice nulls (excl-sticky, Harris unique) are in [structured nulls](structured_nulls_choice_lr.md#downstream-sc-regtype-tables).

### Repo-tracked region lists (2026-07-20e)

Region types live in git under `data/` so remote runs do not need an openalyx → alyx copy:

- `data/stimchoice_act_regtype_regions_p_mean_c_0.01.csv` — default `--regtype-csv`
- `data/var_partition_mixed_stim_choice_regions.csv` — the 19 mixed regions

```bash
python scripts/run_var_partition.py --target mixed
```

---

## Model and implementation

**Per neuron, 0–80 ms post-stimOn:**

`y ~ 1 + stim + choice + prior + stim×prior`

with stim = signed contrast, choice = ±1, prior = `probabilityLeft − 0.5`. Unique R² is additive Type-II; prior-modulated stim = ΔR²(full − additive).

**Window (2026-07-20f):** the default analysis window is **0–80 ms** post-`stimOn` (`SHORT_DURINGSTIM_WINDOW_S`), matching the early-stim / short decorrelation splits. Override with `--window 0.15`.

**Prior (2026-07-20h):** `get_var_partition` defaults to `prior_type='act'` — compute `action_kernel_priors` on the full choice sequence (same as the act SC splits / `get_d_vars`) and use the continuous EMA − 0.5 as the prior regressor. True-block 0.5 trials are **kept** (the kernel needs them). Override with `--prior-type block` for comparison.

**Code:** `get_var_partition` / `get_all_var_partition` / `var_partition_stacked` in `block_analysis_allsplits.py`; CLI `scripts/run_var_partition.py` (writes under **alyx** only).

```bash
# smoke on the existing insertion_cache
python scripts/run_var_partition.py \
  --one-cache-dir ~/Downloads/ONE/alyx.internationalbrainlab.org \
  --cached-only --n-insertions 3 --no-restart

# full BWM (needs network / more caches)
python scripts/run_var_partition.py \
  --one-cache-dir ~/Downloads/ONE/alyx.internationalbrainlab.org
```

**Smoke (3 cached insertions):** 64 neurons → 4 regions in `alyx.../meta/var_partition_by_region.csv`. Example means:

| region | regtype | R²_stim | R²_choice | R²_stim×prior |
| ------ | ------- | ------- | --------- | ------------- |
| GPe | 0.5 | 0.007 | 0.005 | 0.005 |
| VM | 0.5 | 0.033 | 0.008 | 0.003 |
| VAL | 1.0 | 0.008 | 0.007 | 0.011 |

**Outputs:** `alyx.../manifold/var_partition/`, `.../res/var_partition_stacked.npy`, `.../meta/var_partition_by_region.csv`.

**Slurm (2026-07-20g):** with the insertion cache, ~0.5–10 s per insertion (compute vs ORCD filesystem load); ~700 BWM probes → ~0.5–2 h in a single job. Default **4 shards** (2 h walltime each) + a finalize stack.

```bash
bash scripts/submit_var_partition_sharded.sh
# N_SHARDS=2 TARGET=mixed WINDOW=0.08 bash scripts/submit_var_partition_sharded.sh
```

**Sync note (2026-07-20i):** a binning fix (`bin_spikes2D` over all clusters, then slice) and a 1 h Slurm walltime were brought from `main` → `develop`; the earlier subset-`cluster_ids` path mostly failed.

---

## Full BWM results (descriptive, 2026-07-20i)

Outputs (local alyx copy): `meta/var_partition_by_region.csv`, `manifold/res/new/var_partition_stacked.npy` (19 regions; matches the CSV).

Design reminder: 0–80 ms post-stimOn; act prior; unique Type-II R²; `R²_stim×prior` = ΔR²(full − additive) = the prior-modulated stim readout. Means are neuron-averaged within region.

**Global:** 13,394 neurons · 19/19 mixed regions. Mean unique R²: prior **0.0083** ≫ stim **0.0044** ≈ stim×prior **0.0035** ≈ choice **0.0031**. So in the early during-stim window the **main prior term dominates** the partition, and the stim×prior interaction is on the **same order as unique stim**, not a negligible leftover. stim×prior > unique stim in **8/19** regions; > unique choice in **12/19**.

| region | n | ins | R²_full | stim | choice | prior | stim×prior | s×p/stim | regtype | read |
| ------ | ---- | --- | --------- | ---------- | ------ | ---------- | ---------- | -------- | ------- | ---- |
| BMA | 175 | 13 | 0.012 | 0.0027 | 0.0028 | 0.0039 | 0.0029 | 1.08 | — | Small balanced partition; stim×prior ≈ stim ≈ choice. Amygdala mixed unit with weak but even prior modulation of stim. |
| CENT2 | 613 | 16 | 0.021 | 0.0031 | 0.0037 | 0.0087 | 0.0038 | 1.23 | 1.0 | Cerebellar vermis (move-typed). Prior-led; stim×prior exceeds unique stim — early rate already carries prior-gated stim variance. |
| CP | 2892 | 76 | 0.015 | 0.0034 | 0.0030 | 0.0050 | 0.0028 | 0.83 | 0.5 | Largest sample (striatum integrator). Modest everything; stim slightly > choice; stim×prior a bit under unique stim. Stable "yes mixed, mild modulation" baseline. |
| CUL4 5 | 1183 | 33 | 0.020 | 0.0039 | 0.0035 | 0.0074 | 0.0034 | 0.87 | 1.0 | Culmen (move). Prior main largest; stim / choice / stim×prior nearly tied — classic mixed early encoding with moderate prior gating of stim. |
| FN | 46 | 5 | 0.013 | 0.0032 | 0.0019 | 0.0034 | 0.0031 | 0.96 | 1.0 | Fastigial; **low n**. stim ≈ stim×prior ≫ choice — stim variance is almost fully prior-contingent; treat as suggestive. |
| GRN | 560 | 17 | 0.021 | 0.0034 | 0.0030 | 0.0076 | 0.0035 | 1.02 | 1.0 | Gigantocellular RF. stim×prior ≈ unique stim; prior dominates. Brainstem mixed site with clear prior-modulated stim. |
| IP | 604 | 28 | 0.025 | 0.0044 | 0.0032 | 0.0097 | 0.0049 | 1.11 | 1.0 | Interpositus: **2nd-highest stim×prior**. Strong prior + interaction > unique stim — cerebellar deep nucleus with prior-gated early stim. |
| IRN | 735 | 30 | 0.020 | 0.0029 | 0.0030 | 0.0085 | 0.0026 | 0.90 | 1.0 | Intermediate RF. Prior-heavy; stim ≈ choice; stim×prior slightly below stim. Mixed but modulation not oversized. |
| LING | 45 | 5 | 0.019 | 0.0035 | 0.0034 | 0.0082 | 0.0030 | 0.84 | 0.5 | Lingula; **low n**. Balanced stim/choice under a large prior; stim×prior close to stim. |
| MRN | 2678 | 128 | 0.024 | 0.0054 | 0.0037 | 0.0080 | 0.0036 | 0.67 | 1.0 | Huge midbrain RF sample. Clear stim > choice; stim×prior solid but **below** unique stim. Overlaps the 0 %-contrast hit regions — early stim variance is real and partly prior-gated. |
| PF | 108 | 7 | 0.038 | **0.0160** | 0.0030 | 0.0058 | 0.0026 | **0.16** | 0.5 | **Outlier: stim-dominated.** Highest unique stim, lowest s×p/stim. Parafascicular / thalamic mixed label, but the early window looks like a **near-additive stim encoder**. |
| PGRN | 120 | 12 | 0.022 | 0.0023 | 0.0028 | 0.0094 | 0.0030 | 1.33 | 0.5 | Paragigantocellular. Prior ≫ rest; stim×prior > unique stim (and ≈ choice). Prior-led mixed with strong gating relative to weak stim main. |
| PRNc | 280 | 11 | 0.024 | 0.0054 | 0.0032 | 0.0081 | 0.0032 | 0.59 | 0.5 | Pontine RF central. Stim ≈ MRN-level; lower stim×prior ratio — more additive stim than cerebellar peers. |
| SCm | 1666 | 72 | 0.022 | 0.0051 | 0.0032 | 0.0074 | 0.0035 | 0.68 | 1.0 | Superior colliculus (medial). Parallel to MRN: stim > choice, mid stim×prior. Also a 0 %-contrast offset region — consistent midbrain early stim + partial prior gate. |
| SIM | 893 | 21 | 0.021 | 0.0032 | 0.0026 | 0.0083 | 0.0046 | **1.43** | 0.5 | Simplex cerebellum. High stim×prior / stim — among the strongest **relative** prior modulation of stim in the set. |
| SNr | 149 | 19 | 0.036 | 0.0066 | 0.0042 | 0.0141 | 0.0048 | 0.72 | 0.5 | Substantia nigra reticulata. High full R²; **largest prior** among well-sampled regions after VCO; strong stim and top-tier stim×prior. BG output with concurrent prior + prior-gated stim. |
| VCO | 141 | 5 | **0.041** | 0.0049 | 0.0037 | **0.0192** | **0.0057** | 1.17 | 0.5 | Ventral cochlear / related; **highest stim×prior and prior**, highest full R² — but only 5 insertions. Flag as the **strongest modulation candidate** pending more coverage. |
| VPL | 436 | 24 | 0.019 | 0.0035 | 0.0032 | 0.0071 | 0.0030 | 0.86 | 1.0 | Ventral posterolateral thalamus. Textbook mixed: stim ≈ choice under prior; stim×prior ≈ stim. Sensory-adjacent thalamic prior gating of early stim. |
| VeCB | 70 | 6 | 0.018 | 0.0017 | 0.0019 | 0.0084 | 0.0027 | **1.65** | 0.5 | Vestibulocerebellum; **low n**. Weak stim/choice mains but stim×prior > both (highest ratio) — almost all early "stim" variance looks interaction-shaped; interpret cautiously. |

**Takeaways**

1. **Prior main ≫ stim×prior ≈ unique stim** across mixed regions in 0–80 ms — early activity is prior-rich, so asking whether stim is prior-modulated is well posed and generally **yes at a modest R² scale**.
2. **Cerebellar / reticular-formation cluster** (SIM, IP, CENT2, GRN, PGRN, VeCB, VCO): stim×prior often **≥** unique stim → early stim encoding looks contingent on the act prior.
3. **Midbrain motor (MRN, SCm)** and **PRNc**: clearer additive stim with mid-strength interaction (s×p/stim ~0.6–0.7).
4. **PF** is the clear **stim-additive** exception (interaction tiny versus stim).
5. **CP / VPL / CUL4 5**: balanced mixed without extreme gating.
6. Caveats: FN, LING, VeCB and VCO are thin on insertions; regtype 1.0 vs 0.5 does not cleanly separate modulation strength.

---

## Swanson / meta tables (2026-07-28)

`scripts/plot_var_partition_table.py` → alyx `meta/`:

| PNG | Columns (L→R) |
| --- | ------------- |
| `table_var_partition_sxp_regtype.png` | region · `sxp_stim` (green) · regtype |
| `table_var_partition_mixed.png` | region · regtype · unique prior (purple) · stim (blue) · choice (orange) · `sxp_stim` (green) |

Only the 19 mixed regions are coloured; others are grey. `sxp_stim` = R²_stim×prior / R²_stim clipped at `--sxp-vmax 2`; unique R² share via `--r2-vmax 0.02`.

---

## Neuron-level prior-shuffle null

**Design** (primary readout `R²_stim×prior`, also unique prior):

- Keep y, stim and choice fixed; shuffle **prior** within insertion.
- Default `--null-mode contrast`: shuffle within |stim| bins.
- Refit full vs additive on each draw; p = (1 + #{null ≥ obs}) / (1 + nrand).
- Wired into `get_var_partition` / `get_all_var_partition` / `var_partition_stacked`; CLI `--nrand` / `--null-mode` / `--alpha-sig` on `scripts/run_var_partition.py`.
- The stacked CSV gains `p_*_mean` and `frac_sig_*` when nulls are present.
- `RESTART=1` now **re-runs** caches whose `nrand` / null mode is below the request (it no longer silently skips old `nrand=0` files).

**Smoke:** one BMA insertion, `nrand=50`, 46 neurons → ~0.2 s; p ∈ [1/51, 1].

**Timing (`nrand=2000`, mixed 19 regions, cache present):**

| | Estimate |
| --- | --- |
| Local null compute | ~1.4 ms/null @ 46 neurons → ~3 s per hit insertion |
| BWM null lstsq | ~15–30 min total |
| + ORCD I/O | dominates (~2–10 s × ~700 probes) |
| 4 shards | ~20–45 min/shard typical; the submitter uses `--time=3:00:00` when `NRAND ≥ 1000` |

```bash
NRAND=2000 bash scripts/submit_var_partition_sharded.sh
```

**Port to `main`:** `block_analysis_allsplits.py`, `scripts/run_var_partition.py`, `scripts/run_var_partition_slurm.sh`, `scripts/submit_var_partition_sharded.sh`.

### Full BWM null stack (2026-07-28b)

Output: `alyx.../manifold/res/new/var_partition_stacked.npy` — same 19 mixed regions / 13,394 neurons; R² means **identical** to the descriptive stack above. Null: contrast-stratified prior shuffle, `nrand=2000`, α=0.05 for `frac_sig_*`.

**Global (neuron-weighted):**

| | mean R² | mean p | frac p<0.05 |
| ------------ | -------- | -------- | ----------- |
| stim×prior | 0.0035 | **0.45** | **0.114** |
| unique prior | 0.0083 | 0.35 | **0.239** |

So the region-mean R²_stim×prior is real but small: under this null only ~**11 %** of mixed neurons individually clear α=0.05 for the interaction (~2× the 5 % false-positive floor). Unique prior is clearer (~**24 %** significant). A mean p near 0.45 for stim×prior implies most cells are null-consistent.

| region | n | frac_sig s×p | frac_sig prior | mean p s×p | s×p/stim |
| ------ | ---- | ------------ | -------------- | ---------- | -------- |
| SNr | 149 | **0.148** | 0.349 | 0.467 | 0.72 |
| VeCB | 70 | 0.143 | 0.243 | 0.436 | 1.65 |
| GRN | 560 | 0.143 | 0.279 | 0.430 | 1.02 |
| SIM | 893 | 0.140 | 0.256 | 0.445 | 1.43 |
| PGRN | 120 | 0.133 | **0.392** | 0.468 | 1.33 |
| IRN | 735 | 0.128 | 0.331 | 0.440 | 0.90 |
| VCO | 141 | 0.128 | 0.355 | 0.443 | 1.17 |
| IP | 604 | 0.123 | 0.298 | 0.432 | 1.11 |
| PRNc | 280 | 0.118 | 0.300 | 0.436 | 0.59 |
| CP | 2892 | 0.113 | 0.171 | 0.455 | 0.83 |
| CENT2 | 613 | 0.113 | 0.294 | 0.419 | 1.23 |
| MRN | 2678 | 0.111 | 0.245 | 0.451 | 0.67 |
| CUL4 5 | 1183 | 0.110 | 0.197 | 0.441 | 0.87 |
| FN | 46 | 0.109 | 0.217 | 0.423 | 0.96 |
| SCm | 1666 | 0.103 | 0.247 | 0.451 | 0.68 |
| BMA | 175 | 0.086 | 0.103 | 0.491 | 1.08 |
| PF | 108 | 0.083 | 0.213 | 0.494 | **0.16** |
| VPL | 436 | 0.073 | 0.222 | 0.466 | 0.86 |
| LING | 45 | 0.067 | 0.333 | 0.495 | 0.84 |

**Takeaways**

1. The descriptive region means still stand; the null does **not** invent a large neuron-level stim×prior population — enrichment is modest (~11 % vs 5 %).
2. **Unique prior** is the stronger single-neuron signal in this window (higher R² and ~2× the significant fraction of stim×prior).
3. Regions with high s×p/stim ratios (SIM, VeCB, PGRN, GRN, …) also tend to sit at the top of `frac_sig_stim_x_prior`, but absolute rates stay below 15 %.
4. **PF** remains the stim-additive outlier (lowest s×p/stim and low `frac_sig`).
5. Caveat: these are **uncorrected** per-neuron α=0.05 rates (no FDR across cells); insertion-nested tests are still open.

---

## Region-level mean-R² null (2026-07-28c, code only)

Per-neuron p-values do **not** yield a region p for the mean R². The region p needs the null distribution of the **region mean**:

`p_region = (1 + #{ mean_null,k ≥ mean_obs }) / (1 + nrand)`

with `mean_null,k = mean_i R²_{i,k}` over neurons in the region (the product of within-insertion prior shuffles).

**Code:** `get_var_partition` now stores `r2_stim_x_prior_null` / `r2_unique_prior_null` as `(nrand, n_neurons)` float32; `var_partition_stacked` writes `p_region_stim_x_prior`, `p_region_unique_prior`, and `r2_*_null_mean` (the mean of the null mean-R² draws). Restart skips only if those draw arrays are present.

**The current `res/new` stack lacks the draw arrays** → re-run on ORCD (it will auto-refit because the arrays are missing):

```bash
NRAND=2000 bash scripts/submit_var_partition_sharded.sh
```

Then copy the updated `var_partition_stacked.npy` → `res/new/`.

---

## Follow-ups

- Region-level mean-R² p-values (needs the ORCD re-run above).
- Optional BH-FDR across neurons.
- Optional `--prior-type block` comparison.
- Insertion-level aggregation of p / `frac_sig`.
- Checks for the full run: stim-only regions (VISpm) should have R²_stim ≫ R²_choice; integrators should show both stim and choice unique R².
