# Research journal — 2026-07-20

## Standing context (carry-forward)

- **Real-data stack:** [2026-07-06](research_journal_2026-07-06.md) / [2026-07-12](research_journal_2026-07-12.md) — `block_analysis_allsplits.py`, insertion cache, contrast-stratified duringstim splits, ORCD sharding, structured choice nulls (Harris / excl-sticky / actkernel).
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

### Goal 3 — Behavior-model synthetic choices for structured choice nulls (cont. of 07-12 Goal 2)

Continuation of [2026-07-12 Goal 2](research_journal_2026-07-12.md). Primary structured nulls for choice L–R: Harris session-permutation (`--session-shuffle-null`) and BWM-style ActionKernel synthetic sessions (`--actkernel-choice-null`).

**Questions:**
1. Can `synthetic_sessions_from_trials` generate artificial choices while holding the real **stim** and **block** sequences fixed? → **No** (it regenerates both via `generate_pseudo_session`). That is the **intended** BWM paper null.
2. Wire that BWM path into choice L–R neural distance? → **Yes** (`--actkernel-choice-null`, updated 2026-07-23). Null labels = synthetic choices at the real session’s stratified `elig_idx` (same indexing as Harris).

**Approach:** audit `scripts/simulate_synthetic_choices.py`; use paper synthetic sessions for the AK null; compare vs Harris / label shuffle.

---

## Notes / results

### 2026-07-21 — Goal 3: `scripts/simulate_synthetic_choices.py` audit + wiring

**Source:** user-added [`scripts/simulate_synthetic_choices.py`](../scripts/simulate_synthetic_choices.py) (wraps IBL [`behavior_models`](https://github.com/int-brain-lab/behavior_models) `ActionKernel`). Package is a **git submodule** at [`third_party/behavior_models`](../third_party/behavior_models) (path-prepended); remote Slurm needs the repo checkout + **`torch`** and **`sobol_seq`** in conda (MCMC init) — not a cluster `pip install behavior_models`. Init: `git submodule update --init --recursive`.

| API | Keeps real stim/block? | Use for our choice L–R null? |
|-----|------------------------|------------------------------|
| `synthetic_sessions_from_trials` / `make_synthetic_session` | **No** — draws *pseudo* blocks+contrasts via `generate_pseudo_session`, then simulates choices | **Yes** — BWM paper null (wired) |
| `fit_action_kernel` + `simulate_choices(stim, side, params)` | **Yes** — if you pass the real session's stim/side | Available helper; **not** the wired null |
| `synthetic_choices_fixed_stim` (added) | **Yes** — thin wrapper of the above | Available helper; **not** the wired null |

**Null path wired (2026-07-23 update):** `--actkernel-choice-null` → tag
`null_scheme: synthetic_choice_pseudosession`. Ported to **`main`** for ORCD
(`152d0af`; submodule init required).

#### Steps for `synthetic_choice_pseudosession` (per insertion × choice L–R split)

1. **Load / prepare trials** (same as other choice L–R paths): insertion cache →
   trials; apply act/bayes prior overwrite if the split name asks for it; optional
   `--exclude-sticky-trials` trim.
2. **Stratify on the real session only** (stim side ± block/prior from the split
   name) → eligible trial indices `elig_idx`. Bin spikes on those trials → neural
   tensor `b` (rows = eligible trials). Observed choices =
   `trials.choice[elig_idx]`.
3. **Observed distance:** split `b` by real L vs R choices; region distances as usual
   (true permutation index 0 in `D[reg]['d_*']`).
4. **Fit ActionKernel once per `eid`** (`get_actkernel_choice_fit`): MCMC on the
   **real** session’s choice / stim / side → posterior-mean
   \(\hat{\bm\theta}=[\hat\alpha,\hat\zeta,\widehat{\mathrm{lapse}}_+,\widehat{\mathrm{lapse}}_-]\).
   Pickle under `manifold/actkernel_fits/` (shared across probes of the same eid).
5. **For each of `nrand` null draws** (`synthetic_sessions_from_trials`, BWM paper):
   - Draw a **new** pseudo stim/block schedule (`generate_pseudo_session` /
     vectorized equivalent) of the same length as the real session.
   - Simulate a full-session choice sequence under \(\hat{\bm\theta}\)
     (`ActionKernel.simulate` / `simulate_parallel`) — prior updates from the
     **simulated** choices on that fake stim stream.
   - Null labels for neural data: `ys = (synthetic_choice[elig_idx] == 1)` —
     same trial numbers as the real stratification; **do not** re-stratify by
     the pseudo stim.
   - Reject draws with &lt; `min_trials_per_side` L or R on `elig_idx`; keep
     sampling until `nrand` valid draws.
   - Recompute region distances on fixed `b` under those labels (null distribution).
6. **Pool / p-values** as for other control runs (`uperms` counts unique label
   patterns).

**What is frozen vs resampled**

| Piece | Real / observed | Under null |
|-------|-----------------|------------|
| Neural `b` | fixed (real spikes @ real `elig_idx`) | fixed |
| Which trial indices | stim×prior stratification on **real** session | same `elig_idx` |
| Stim / block stream | real (only for defining `elig_idx` + fit) | **new** pseudo each draw |
| Choices | real | simulated under \(\hat{\bm\theta}\) on pseudo stim |
| \(\bm\theta\) | fitted once on real session | held fixed |

Rationale for regenerating stim (vs fixing the real stream): the paper’s choice null
is behaviour under a **fresh** task schedule with the animal’s fitted policy.
Holding the recorded stim would couple null choices to the same sensory sequence
that drove `b`. Indexing at real `elig_idx` keeps the neural tensor aligned while
still allowing unlimited Monte Carlo draws (unlike Harris’s finite donor bank).

```bash
conda activate iblenv   # needs torch; behavior_models from third_party/ submodule
# fresh clone: git submodule update --init --recursive
python scripts/run_goal2_splits.py --preset choice_lr_session_null_all \
  --actkernel-choice-null --nrand 200
# shards:
#   bash scripts/submit_goal2_choice_actkernel_null_sharded.sh
# smoke:
#   python scripts/smoke_choice_actkernel_null.py
```

**Smoke (2026-07-23):** `scripts/smoke_choice_actkernel_null.py` on local
insertion_cache → `null_scheme=synthetic_choice_pseudosession` (`choice_stim_l`,
short MCMC `ACTKERNEL_NB_STEPS=40`). Same smoke also passed on `main` after the
ORCD port. Submitter: `scripts/submit_goal2_choice_actkernel_null_sharded.sh`
(`ACTKERNEL_CHOICE_NULL=1`, optional `SMOKE_FIRST=1`).

**Next:** compare null width / p-values vs `--session-shuffle-null` (Harris) and
label shuffle; first eid MCMC is slow (later probes reuse the pickle).

**To be resolved:** currently act-prior labels for analysis use a **fixed**
`α=0.2` via `action_kernel_priors` on each session’s choice sequence (same α
everywhere), then results are pooled into the supersession. Should we instead run
`fit_action_kernel` **per session** (MCMC → session-specific `α`, and optionally
the full `[α, ζ, lapse±]`), recompute that session’s continuous/binary act priors
from the fitted kernel, and **only then** pool into the supersession for all
act-conditioned analyses?

### 2026-07-23 — Choice L–R actkernel ORCD run: **invalid** (missing `sobol_seq`)

Default submitter preset `choice_lr_session_null_all` **does** include all 8 act
splits (duringchoice `choice_stim_*` + duringstim `choice_duringstim_*`).

**Failure:** every insertion failed with `No module named 'sobol_seq'` (ActionKernel
MCMC init in `behavior_models`). Shard logs e.g.
`goal2_shard_g2ak_choice_stim_r_block_l_act_s0_*.out`: `ok 0/1 splits`,
`MISSING shard …/choice_stim_…shard0.npy`. Duringstim shards failed the same way
(confirmed). Pooled `*_pseudosession*` files under local `res/new` from that
attempt are **not** a successful BWM null run.

**Deps:** `torch` + **`pip install sobol_seq`** in ibl conda (plus usual stack).
Clear `manifold/actkernel_fits/` and failed choice stream/pooled `_pseudosession`
outputs before resubmitting.

### 2026-07-23b — Re-run after `sobol_seq` (all 8 splits) + null bug

**Re-run landed** in alyx `manifold/res/new/` (mtimes ~15:25–15:28). All 8
`*_pseudosession` act splits present. Coverage still short of sibling
`act_block` / openalyx (~62k): ~197–200 regions, ~49–54k cells / split
(~79–89% vs oa; matched Δnclus vs `act_block_duringstim_l` ≈ −43 to −47).

**Tables** (`meta/table_choice_pseudosession_vs_shuffle_*`):

| α | epoch | shuffle | pseudosession | lost | gained | kept |
|---|-------|--------:|---------------:|-----:|-------:|-----:|
| 0.05 | duringstim | 71 | **204** | 0 | 133 | 71 |
| 0.05 | duringchoice | 107 | **202** | 1 | 96 | 106 |
| 0.01 | duringstim | 46 | **200** | 1 | 155 | 45 |
| 0.01 | duringchoice | 84 | **201** | 1 | 118 | 83 |

Pseudosession calls almost every region significant — **not** a tighter null.

**Bug (under-dispersed null):** same insertion, same observed euc amp; AK
pseudo-session null amps ≪ label-shuffle null amps (e.g. LD: shuffle null
med ≈ 2.9 vs obs 3.6; AK null med ≈ 1.2, max 2.2 → false p≪0.05). Pooled
regde (CP): shuffle null med ≈ obs; pseudo null med ≈ **½** obs and never
exceeds it.

**Mechanism:** eligibility is stim×prior (act-binary 0.8/0.2). Real choices on
those trials are often **highly imbalanced** (e.g. 57 L vs 3 R). Label shuffle
**preserves** that n_L/n_R → large noise-floor distances. BWM pseudo-sessions
regenerate full stim/blocks, then we read choices at real `elig_idx` → ~**50/50**
labels on the same neural trials → much smaller null euc distances → inflated
significance. Confirmed: reshuffling those balanced AK labels still yields the
low null floor; fixed-stim AK choices (real stim/block) partially restore
imbalance but still below shuffle.

**Not a `sobol_seq` / fit crash** on this re-run; statistical null construction
bug relative to stratified choice L–R.

#### Current null options (pending decision)

Goal: restore realistic **n_L/n_R** (and temporal structure) on stim×block–
stratified `elig_idx` without defeating the structured null.

| # | Null | Stim×block | Choice process | Late-session stickiness | `nrand`≈2000? |
|---|------|------------|----------------|-------------------------|---------------|
| **1** | Pseudosession + **stim×block stratification** | New pseudo schedule, but stratified / constrained so eligible slots match real stim×block (bias context) | AK simulate under fitted θ | **No** explicit late stickiness (stationary `α` only; blocky runs via stim+history) | **Yes** — unlimited synthetic draws |
| **2** | Pseudosession on **exact real stim×block sequence** | Pin recorded `(stim_side, pLeft)` (fixed-stim); only choices are synthetic | AK simulate under fitted θ | **No** (same as 1 — AK has no time-varying perseveration) | **Yes** — resample choices on the fixed stream |
| **3** | **Harris / session transplant** (original): other sessions’ choice sequences at recipient `elig_idx`, conditioned on stim×block stratification | Real recipient stim×block defines eligibility; donor choices indexed in | Empirical choices from other eids | **Yes** — real mice carry late-session / sticky structure | **No** — donor pool ≪ 2000 unique usable sequences |

**Notes:**
- Unconstrained full BWM pseudo (calendar-index into mismatched stim×block;
  former `_pseudosession` run) is **retired as default**: ~50/50 labels →
  under-dispersed null. Kept only as legacy `unconstrained` mode if needed.
- **1 vs 2 (AK synthetic choices):**
  - **Shared:** both fit θ once per eid; both can draw `nrand≈2000`; both use
    stationary AK `α` (no extra late-session perseveration beyond blocks +
    choice history); both aim to restore stim×block–appropriate choice
    imbalance on the labels applied to neural `elig` trials.
  - **1 (stratified pseudo):** each null draws a **new** pseudo stim/block
    stream + AK choices; labels = choices on the pseudo’s own stim×prior
    stratum (same definition as the split: true-block or act-binary), taken in
    temporal order and length-matched to `n_elig`. Breaks the recorded
    stim/block schedule (stronger “new world” / BWM-like confound break) while
    still evaluating choices that lived in the correct bias context. Stratum
    size varies per draw → reject if too few trials.
  - **2 (fixed real stim×block):** AK choices on the **exact** recorded
    `(stim, side)` sequence; labels at real `elig_idx`. Strongest match to the
    session’s bias timeline and usually closest n_L/n_R to observed; weaker as
    a confound break because neurons and the null policy share the same stim
    stream (choices remain stochastic under θ, not a copy of real choices).
  - **When to prefer which:** use **1** if the scientific goal is closest to
    BWM “behavior under an independent generative world”; use **2** if the
    priority is a fair stratified null that matches imbalance/temporal bias
    with minimal reject rate and maximum schedule fidelity.
- **3** (`--session-shuffle-null` / `_harris`): empirical sticky structure;
  donor pool ≪ 2000 unique sequences without replacement / circular shifts.
- **Cluster:** `NULL_SCHEME=pseudo_strat|pseudo_fixed|harris` via
  `scripts/submit_goal2_choice_null_sharded.sh`, or all three with
  `scripts/submit_goal2_choice_null_all_schemes_sharded.sh`.

**Filename tagging:**

| Null | On-disk basename |
|------|------------------|
| label shuffle | `{split}.npy` |
| option 1 AK stratified pseudo | `{split}_pseudo_strat.npy` |
| option 2 AK fixed stim×block | `{split}_pseudo_fixed.npy` |
| option 3 Harris | `{split}_harris.npy` |
| legacy unconstrained BWM index | `{split}_pseudosession.npy` |

```bash
python scripts/plot_choice_null_comparison_table.py \
  --arm-res ~/Downloads/ONE/alyx.internationalbrainlab.org/manifold/res/new \
  --arm-tag pseudosession --force-combine --alpha 0.05
```

**ORCD:**
```bash
# all three schemes (opt 1–3)
bash scripts/submit_goal2_choice_null_all_schemes_sharded.sh

# or one at a time:
NULL_SCHEME=pseudo_strat bash scripts/submit_goal2_choice_null_sharded.sh
NULL_SCHEME=pseudo_fixed bash scripts/submit_goal2_choice_null_sharded.sh
NULL_SCHEME=harris      bash scripts/submit_goal2_choice_null_sharded.sh
```
Needs `sobol_seq` + `torch` for AK schemes; Harris needs donor bank job
(auto-submitted). Finalize must export the same `ACTKERNEL_NULL_MODE` /
`SESSION_SHUFFLE_NULL` as shards. Do **not** interpret legacy unconstrained
`_pseudosession` sig tables.

### 2026-07-24 — res/new: all three nulls analyzed (α=0.01)

**Data:** alyx `manifold/res/new/` — full 8 act splits ×
`_pseudo_strat` / `_pseudo_fixed` / `_harris` / legacy `_pseudosession`
(nrand=2000). Baseline = openalyx label-shuffle. Combined 4-split tables;
BH-FDR on `p_mean` at **α=0.01**.

**Policy note:** failed null draws (cannot fill `nrand` balanced labels; no
long-enough Harris donors) **skip the insertion** (logged `split skip`); no
circular-shift / unconstrained / observed-label fallbacks.

#### FDR sig counts (α=0.01)

| arm | duringstim | duringchoice | median p (stim / choice) |
|-----|----------:|-------------:|--------------------------|
| shuffle (openalyx) | 46 | 84 | 0.071 / 0.019 |
| **pseudo_strat** | **0** | **0** | **0.995 / 0.987** |
| pseudo_fixed | 95 | 124 | 0.009 / 0.002 |
| harris | 201 | 202 | ~0.0005 |
| pseudosession (legacy) | 201 | 201 | ~0.0005 |

Vs shuffle (lost / gained / kept at α=0.01):

| epoch | arm | shuffle | arm n | lost | gained | kept |
|-------|-----|--------:|------:|-----:|-------:|-----:|
| duringstim | pseudo_strat | 46 | **0** | 46 | 0 | 0 |
| duringchoice | pseudo_strat | 84 | **0** | 84 | 0 | 0 |
| duringstim | pseudo_fixed | 46 | 95 | 8 | 57 | 38 |
| duringchoice | pseudo_fixed | 84 | 124 | 9 | 49 | 75 |
| duringstim | harris | 46 | 200 | 1 | 155 | 45 |
| duringchoice | harris | 84 | 202 | 1 | 119 | 83 |
| duringstim | pseudosession | 46 | 200 | 1 | 155 | 45 |
| duringchoice | pseudosession | 84 | 201 | 1 | 118 | 83 |

**Plots / CSV:**
`meta/table_choice_{pseudo_strat,pseudo_fixed,harris,pseudosession}_vs_shuffle_*_p_mean_c_0.01.png`
(+ duringchoice companions); `meta/choice_null_res_new_summary_a0.01.csv`.

#### Region / cell retention vs shuffle

Congruent = stim side matches act-prior side (`*_stim_l_block_l_*`,
`*_stim_r_block_r_*`); incongruent = crossed.

| | Congruent | Incongruent |
|--|-----------|-------------|
| **strat** | ~37% regions, **~10% cells** | ~96% regions, ~87% cells |
| **fixed / harris** | ~96% regions, ~83% cells | ~96% regions, ~87% cells |

Strat’s missing mass is almost entirely **congruent** splits (e.g.
`choice_stim_l_block_l_act`: 71 regs / 4.7k cells vs fixed 199 / 54k).

#### Why strat skips congruent insertions

Strat length-matches labels from the **pseudo’s own** stim×act-prior stratum
to real `n_elig`. Reject if stratum too short or &lt;5 trials/side; after
`nrand×20` synthetic sessions without `nrand` accepts → skip insertion.

Probe (local eid, short MCMC): congruent `stim_l×block_l_act` had
`n_elig=61` but pseudo stratum size med **36** (always &lt;61) → **100% short**
rejects (0/100 accept). Same elig under **fixedstim**: 97% accept. Incongruent
on the same eid: ~48% accept (mix of short + imbal). So congruent loss is
driven by **short act-prior strata on new pseudos** (real sticky act-priors
align with stim → large congruent pools; remade pseudo+AK history does not),
not primarily by the ≥5/side gate.

#### Interpretation

- **Opt 1 (strat):** only arm that looks **calibrated** at α=0.01 (0 FDR hits),
  but congruent coverage is too thin to trust as a full-map null — treat as
  provisional / incongruent-dominated.
- **Opt 2 (fixed):** still **more liberal than shuffle** (95/124 vs 46/84).
- **Opt 3 (harris) + legacy unconstrained:** still **broken** (almost all
  regions FDR-sig) — same under-dispersed / mismatch pathology as 2026-07-23b.

**Next:** fix strat length-match (e.g. allow shorter windows + pad? subsample
real `n_elig` down to available stratum size? or reject only on imbalance) so
congruent insertions are retained without reintroducing unconstrained indexing;
re-plot α=0.01 after coverage is restored.

### 2026-07-24b — Strat longer pseudos (`pseudo_len_factor=3`)

**Fix:** BWM same-length pseudos undersize congruent act strata (probe: need 61,
med stratum 36 → 0–1% accept). Draw longer worlds:
`n_pseudo = ceil(n_real × factor)` via `generate_pseudo_blocks(n_pseudo)` + AK
simulate (fit still on real session).

**Code:** `--actkernel-pseudo-len-factor` / env `ACTKERNEL_PSEUDO_LEN_FACTOR`
(default **3**); on low accept rate the control loop **doubles up to 16**.
Always writes **`_pseudo_strat`** (submit `CLEAR_STREAM=1` clears prior
stream_acc + pooled res). Probe: factor 1 → ~1% accept; **factor 3 → 100%**
(med stratum 97).

**ORCD rerun (strat only):**
```bash
# on main with updated code + submodule
bash scripts/submit_goal2_choice_strat_x3_sharded.sh
# or: NULL_SCHEME=pseudo_strat bash scripts/submit_goal2_choice_null_sharded.sh
# optional: SMOKE_FIRST=1 …
```
Outputs: `$ONE_CACHE_DIR/manifold/res/{split}_pseudo_strat*.npy`.
Plot with `--arm-split-suffix _pseudo_strat --arm-tag strat --alpha 0.01`.

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

### 2026-07-20i — Full BWM mixed var-partition (alyx results)

**Sync:** binning fix (`bin_spikes2D` all clusters → slice) + Slurm 1 h walltime
brought from `main` → `develop` (earlier subset-`cluster_ids` path left mostly fails).

**Outputs (local alyx copy):**
- `meta/var_partition_by_region.csv`
- `manifold/res/new/var_partition_stacked.npy` (19 regions; matches CSV)

**Design reminder:** 0–80 ms post-stimOn; act prior; unique Type-II R²;  
`R²_stim×prior` = ΔR²(full − additive) = prior-modulated stim readout.  
Means below are neuron-averaged within region.

**Global:** 13 394 neurons · 19/19 mixed regions. Mean unique R²:  
prior **0.0083** ≫ stim **0.0044** ≈ stim×prior **0.0035** ≈ choice **0.0031**.  
So in early duringstim, the **main prior term dominates** the partition; the
stim×prior interaction is on the **same order as unique stim**, not a
negligible leftover. stim×prior > unique stim in **8/19**; > unique choice in
**12/19**.

| region | n | ins | R²_full | stim | choice | prior | stim×prior | s×p/stim | regtype | read |
|--------|--:|---:|--------:|-----:|-------:|------:|-----------:|---------:|--------:|------|
| BMA | 175 | 13 | 0.012 | 0.0027 | 0.0028 | 0.0039 | 0.0029 | 1.08 | — | Small balanced partition; stim×prior ≈ stim ≈ choice. Amygdala mixed unit with weak but even prior modulation of stim. |
| CENT2 | 613 | 16 | 0.021 | 0.0031 | 0.0037 | 0.0087 | 0.0038 | 1.23 | 1.0 | Cerebellar vermis (move-typed). Prior-led; stim×prior exceeds unique stim — early rate already carries prior-gated stim variance. |
| CP | 2892 | 76 | 0.015 | 0.0034 | 0.0030 | 0.0050 | 0.0028 | 0.83 | 0.5 | Largest sample (striatum integrator). Modest everything; stim slightly > choice; stim×prior a bit under unique stim. Stable “yes mixed, mild modulation” baseline. |
| CUL4 5 | 1183 | 33 | 0.020 | 0.0039 | 0.0035 | 0.0074 | 0.0034 | 0.87 | 1.0 | Culmen (move). Prior main largest; stim / choice / stim×prior nearly tied — classic mixed early encoding with moderate prior gating of stim. |
| FN | 46 | 5 | 0.013 | 0.0032 | 0.0019 | 0.0034 | 0.0031 | 0.96 | 1.0 | Fastigial; **low n**. stim ≈ stim×prior ≫ choice — stim variance is almost fully prior-contingent; treat as suggestive. |
| GRN | 560 | 17 | 0.021 | 0.0034 | 0.0030 | 0.0076 | 0.0035 | 1.02 | 1.0 | Gigantocellular RF. stim×prior ≈ unique stim; prior dominates. Brainstem mixed site with clear prior-modulated stim. |
| IP | 604 | 28 | 0.025 | 0.0044 | 0.0032 | 0.0097 | 0.0049 | 1.11 | 1.0 | Interpositus: **2nd-highest stim×prior**. Strong prior + interaction > unique stim — cerebellar deep nucleus with prior-gated early stim. |
| IRN | 735 | 30 | 0.020 | 0.0029 | 0.0030 | 0.0085 | 0.0026 | 0.90 | 1.0 | Intermediate RF. Prior-heavy; stim≈choice; stim×prior slightly below stim. Mixed but modulation not oversized. |
| LING | 45 | 5 | 0.019 | 0.0035 | 0.0034 | 0.0082 | 0.0030 | 0.84 | 0.5 | Lingula; **low n**. Balanced stim/choice under a large prior; stim×prior close to stim. |
| MRN | 2678 | 128 | 0.024 | 0.0054 | 0.0037 | 0.0080 | 0.0036 | 0.67 | 1.0 | Huge midbrain RF sample. Clear stim > choice; stim×prior solid but **below** unique stim (ratio 0.67). Overlaps Goal‑3 0%-contrast hit regions — early stim variance is real and partly prior-gated. |
| PF | 108 | 7 | 0.038 | **0.0160** | 0.0030 | 0.0058 | 0.0026 | **0.16** | 0.5 | **Outlier: stim-dominated.** Highest unique stim, lowest s×p/stim. Parafascicular / thalamic mixed label but early window looks like a **near-additive stim encoder** (prior modulation small vs stim). |
| PGRN | 120 | 12 | 0.022 | 0.0023 | 0.0028 | 0.0094 | 0.0030 | 1.33 | 0.5 | Paragigantocellular. Prior ≫ rest; stim×prior > unique stim (and ≈ choice). Prior-led mixed with strong gating relative to weak stim main. |
| PRNc | 280 | 11 | 0.024 | 0.0054 | 0.0032 | 0.0081 | 0.0032 | 0.59 | 0.5 | Pontine RF central. Stim ≈ MRN-level; stim×prior lower ratio (0.59) — more additive stim than cerebellar peers. |
| SCm | 1666 | 72 | 0.022 | 0.0051 | 0.0032 | 0.0074 | 0.0035 | 0.68 | 1.0 | Superior colliculus (medial). Parallel to MRN: stim > choice, mid stim×prior. Also a Goal‑3 offset region — consistent midbrain early stim + partial prior gate. |
| SIM | 893 | 21 | 0.021 | 0.0032 | 0.0026 | 0.0083 | 0.0046 | **1.43** | 0.5 | Simplex cerebellum. High stim×prior / stim — among strongest **relative** prior modulation of stim in the set. |
| SNr | 149 | 19 | 0.036 | 0.0066 | 0.0042 | 0.0141 | 0.0048 | 0.72 | 0.5 | Substantia nigra reticulata. High full R²; **largest prior** among well-sampled regions after VCO; strong stim and top-tier stim×prior. BG output with concurrent prior + prior-gated stim. |
| VCO | 141 | 5 | **0.041** | 0.0049 | 0.0037 | **0.0192** | **0.0057** | 1.17 | 0.5 | Ventral cochlear / related; **highest stim×prior and prior**, highest full R² — but only 5 insertions. Flag as **strongest modulation candidate** pending more coverage. |
| VPL | 436 | 24 | 0.019 | 0.0035 | 0.0032 | 0.0071 | 0.0030 | 0.86 | 1.0 | Ventral posterolateral thalamus. Textbook mixed: stim≈choice under prior; stim×prior ≈ stim. Sensory-adjacent thalamic prior gating of early stim. |
| VeCB | 70 | 6 | 0.018 | 0.0017 | 0.0019 | 0.0084 | 0.0027 | **1.65** | 0.5 | Vestibulocerebellum; **low n**. Weak stim/choice mains but stim×prior > both (highest ratio) — almost all early “stim” variance looks interaction-shaped; interpret cautiously. |

**Takeaways for Goal 1**

1. **Prior main ≫ stim×prior ≈ stim unique** across mixed regions in 0–80 ms — early activity is prior-rich; asking whether stim is prior-modulated is well-posed and generally **yes at a modest R² scale**.
2. **Cerebellar / RF cluster** (SIM, IP, CENT2, GRN, PGRN, VeCB, VCO): stim×prior often **≥** unique stim → early stim encoding looks contingent on act prior.
3. **Midbrain motor (MRN, SCm)** and **PRNc**: clearer additive stim with mid-strength interaction (s×p/stim ~0.6–0.7).
4. **PF** is the clear **stim-additive** exception (interaction tiny vs stim).
5. **CP / VPL / CUL4 5**: balanced mixed without extreme gating.
6. Caveats: means are descriptive (no neuron-level null yet); FN/LING/VeCB/VCO are thin on insertions; regtype 1.0 vs 0.5 does not cleanly separate modulation strength.

**Next (optional):** insertion- or neuron-level null on `R²_stim×prior`; compare `--prior-type block`; plot region means ± SEM from stacked npy.
