# Research journal — 2026-07-20

## Standing context (carry-forward)

- **Real-data stack:** [2026-07-06](research_journal_2026-07-06.md) / [2026-07-12](research_journal_2026-07-12.md) — `block_analysis_allsplits.py`, insertion cache, contrast-stratified duringstim splits, ORCD sharding, structured choice nulls (synthetic-choice / excl-sticky).
- **Canonical prior-distance defaults:** 80 ms S / 150 ms I–M, fill-from-next-ITI, contrast-matched null — see `AGENTS.md` / `simulate_recovery.py`.
- **Earlier today (logged under 07-12):** perseveration-tail exclusion counts and excl-sticky submit notes (`2026-07-20a–b` in that file).

---

## Today's goals

### Goal 1 — Single-neuron variance partition in mixed stim×choice duringstim regions

Re-examine regions with **both stimulus and choice** sensitivity in the **duringstim**
window (act / action-kernel splits), including early-stim block-only control
(\(d^{\mathrm{stim},se'}\) = `stim_block_{l,r}_act`).

**Approach:** single-neuron OLS variance partition (stim / choice / prior /
stim×prior) to test whether the **stimulus component** in mixed-selectivity
regions is **prior-modulated**.

**Target set:** `has_stim ∧ has_choice` where stim = significant amp on
`stim_duringstim_act` ∨ short (choice+prior) ∨ `stim_duringstim1_act` (block-only);
choice = significant `choice_duringstim_act`. See 2026-07-20d.

### Goal 2 — Speed up simulation + model fitting for refits

Go through the model-fitting codebase and optimize for wall-clock speed, targeting both:

1. **Per-session simulation** (generative / recovery path), and
2. **Model fitting** itself,

so that planned **refits** are feasible at useful `n_sessions` / `nrand` / grid resolution.

**Success criteria (provisional):** profile hotspots; land concrete speedups (vectorization, caching, fewer redundant builds, parallel fit steps) without changing canonical analysis defaults or statistical conclusions of the Phase 4b sanity check.

---

## Notes / results

### 2026-07-20 — Revised Goal 3 (0% choice-conditioned) tables

BWM finalize landed in `alyx.../manifold/res/new/`. Combined choice-L + choice-R
gain/offset tables at FDR α=0.05 / 0.01: see
[07-06 §2026-07-20](research_journal_2026-07-06.md). Brief: **2**/185 regions
(MRN, SCm) at α=0.05 (offset-driven); **0** at α=0.01.

### 2026-07-20c — Goal 1: single-neuron variance partition (implemented)

**Region list source:** openalyx `get_sc_table` → alyx CSV (does not overwrite openalyx
`meta/table_*.png`).

```bash
conda activate iblenv
python scripts/export_stimchoice_regtypes.py \
  --out-cache-dir ~/Downloads/ONE/alyx.internationalbrainlab.org --copy-table-png
```

**duringstim regtype @ α=0.01:** integrator (`0.5`) = **28**, stim (`0`) = **1**
(VISpm), move (`1`) = **22**. Integrators include CP, GPe, MOs, VM, SNr, …

**Model (per neuron, 0–80 ms post-stimOn):**  
`y ~ 1 + stim + choice + prior + stim×prior`  
(stim = signed contrast; choice = ±1; prior = `probabilityLeft−0.5`).  
Unique R² = additive Type-II; prior-modulated stim = ΔR²(full − additive).

**Code:** `get_var_partition` / `get_all_var_partition` / `var_partition_stacked` in
`block_analysis_allsplits.py`; CLI `scripts/run_var_partition.py` (writes under
**alyx** only).

```bash
# smoke on existing insertion_cache
python scripts/run_var_partition.py \
  --one-cache-dir ~/Downloads/ONE/alyx.internationalbrainlab.org \
  --cached-only --n-insertions 3 --no-restart

# full BWM (needs network / more caches)
python scripts/run_var_partition.py \
  --one-cache-dir ~/Downloads/ONE/alyx.internationalbrainlab.org
```

**Smoke (3 cached insertions):** 64 neurons → 4 regions in
`alyx.../meta/var_partition_by_region.csv`. Example means:

| region | regtype | R²_stim | R²_choice | R²_stim×prior |
|--------|--------:|--------:|----------:|--------------:|
| GPe | 0.5 | 0.007 | 0.005 | 0.005 |
| VM | 0.5 | 0.033 | 0.008 | 0.003 |
| VAL | 1.0 | 0.008 | 0.007 | 0.011 |

**Checks (full run):** stim-only (VISpm) should have R²_stim ≫ R²_choice;
integrators should show both stim and choice unique R²; R²_stim×prior is the
prior-modulation readout. Outputs: `alyx.../manifold/var_partition/`,
`.../res/var_partition_stacked.npy`, `.../meta/var_partition_by_region.csv`.

### 2026-07-20d — SC table recreate + mixed stim×choice target (act only)

**Recreated** `alyx.../meta/table_stimchoice_act_regtype_p_mean_c_0.01.png` from openalyx
act combined splits (action-kernel prior). Openalyx original copied as
`…_openalyx_copy.png` for comparison.

All SC inputs are **act** splits:
- `stim_duringstim_act` / `choice_duringstim_act` / duringchoice act counterparts
- early stim **choice+prior** control: `stim_duringstim_short_act`
- early stim **block/prior only** (paper \(d^{\mathrm{stim},se'}\)): `stim_duringstim1_act`
  = `stim_block_{l,r}_act` (not `act_block_duringstim_*`, which is prior distance)

Paper checks (α=0.01): **Σ>0.8** → 1 stim processor (VISpm); **Σ′>0.8** → **23**
(1+22 early), matching the manuscript counts. Regtype: stim=1, early=22,
integrator=28, move=22; stim|choice coding defined for **69/208**.

**Var-partition target (default `--target mixed`):** significant stim
(`stim_s` ∨ `stim_se` ∨ `stim_se'`) **and** significant `choice_s` → **19** regions:
BMA, CENT2, CP, CUL4 5, FN, GRN, IP, IRN, LING, MRN, PF, PGRN, PRNc, SCm, SIM,
SNr, VCO, VPL, VeCB.

Note: only **9/28** duringstim “integrators” (regtype 0.5) have significant stim
amps — the rest are choice-significant without stim (e.g. GPe, VM, MOs). Early
stim processors (VISp, LGd, …) have stim without choice → excluded from mixed.
CSV columns: `sigma_stim_s`, `sigma_stim_s_prime`, `stim_processor(_loose)`,
`has_stim`, `has_choice`, `mixed_stim_choice`.

### 2026-07-20e — Repo-tracked region list for remote runs

Region types live in git under `data/` (not openalyx→alyx copy on cluster):

- `data/stimchoice_act_regtype_regions_p_mean_c_0.01.csv` — default `--regtype-csv`
- `data/var_partition_mixed_stim_choice_regions.csv` — 19 mixed regions

Remote::

```bash
python scripts/run_var_partition.py --target mixed
```

### 2026-07-20f — Early stim window for variance partition

Default analysis window is now **0–80 ms** post-`stimOn`
(`SHORT_DURINGSTIM_WINDOW_S`), matching early-stim / short decorrelation splits.
Override with `--window 0.15` if needed.

### 2026-07-20g — Slurm submit (sharded)

Estimate with insertion cache: ~0.5–10 s/insertion (compute vs ORCD FS load);
~700 BWM probes → **~0.5–2 h** single job. Default **4 shards** (2 h walltime
each) + finalize stack.

```bash
bash scripts/submit_var_partition_sharded.sh
# N_SHARDS=2 TARGET=mixed WINDOW=0.08 bash scripts/submit_var_partition_sharded.sh
```

### 2026-07-20h — Encoding prior = action kernel

`get_var_partition` default `prior_type='act'`: compute `action_kernel_priors`
on the full choice sequence (same as act SC / `get_d_vars`), use continuous EMA
− 0.5 as the prior regressor in \(y\sim\mathrm{stim}+\mathrm{choice}+\mathrm{prior}+\mathrm{stim}\times\mathrm{prior}\).
True-block 0.5 trials are kept (needed for the kernel). Override with
`--prior-type block` only for comparison.
