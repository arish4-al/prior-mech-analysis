# Research journal — 2026-07-06

## Standing context (carry-forward from 2026-06-29)

See [prior journal](research_journal_2026-06-29.md) for Experiments A–D and the unsplit prior-distance results.

### Canonical analysis defaults (mandatory — see `AGENTS.md`)

| Setting | Value |
|---------|-------|
| S window | **80 ms** (`S_DURINGSTIM_WINDOW_S`) |
| I/M window | **150 ms** |
| Truncation | **fill-from-next-ITI** (never zero-pad) |
| Null | **contrast-matched shuffle** (default); `--label-shuffle-null` to override |
| Output root | `<ONE cache>/manifold_sim` |
| Phase 4b sanity (split, seed 123) | S curve_mean≈0.012, p≈0.78 |

ONE cache resolves to `/Users/ariliu/Downloads/ONE/openalyx.internationalbrainlab.org` → sims live under `.../manifold_sim/`.

### Established facts (seed 123, contrast-matched null, α=0.01)

| Result | Value |
|--------|-------|
| Absence S (f1/f2 split) | curve_mean≈0.80, sig — **I/M-mediated composition artefact** |
| Absence S (stim_side unsplit) | 0.011, n.s. — S signal collapses without f1/f2 splits |
| Gain-only I-first threshold (inside adap) | g_s ≈ **1800** (I sig, S n.s.) |
| Gain-only S-first threshold (inside adap) | g_s ≈ **2025** (S sig) |
| Gain-only I-first threshold (outside adap, `gs_free`) | g_s ≈ **700** |
| Gain-only S-first threshold (outside adap, `gs_free`) | g_s ≈ **900** |

---

## Today's goals

### Goal 1 — Complete the generative-model analysis suite

Run **three new analysis variants** on **four saved experiments**, reusing simulations wherever possible.

**The four experiments** (differ only in model params `mp`; seed 123, 40 sessions, nrand=100):

| # | Experiment | g_s | d_s | g_i / d_i / g_m / d_m | CLI today | Saved res? |
|---|-----------|-----|-----|-----------------------|-----------|-----------|
| E1 | No prior mod on S/I/M (Phase 4b) | 0 | 0 | 0 | `--phase4-no-prior-mod` | `absence/figs/phase4_no_prior_mod/res` |
| E2 | Canonical absence (no P→S; fitted P→I/M) | 0 | 0 | fitted | default `absence` | `absence/res`, `absence_fill_next/seed_123/absence/res` |
| E3 | P→S only, I-first threshold | 1800 | 0 | 0 | `s_presence_tune` g_s=1800 | `s_presence_tune/g_s1800_d_s0/res` |
| E4 | P→S only, S-first threshold | 2025 | 0 | 0 | `s_presence_tune` g_s=2025 | `s_presence_tune/g_s2025_d_s0/res` |

(E3/E4 use **inside-adaptation** gain placement — the pair with complete saved `res/`. Outside-adaptation analogues `g_s700_gs_free` / `g_s900` also exist if we prefer that placement.)

**The three new analyses:**

- **1.1 — No contrast matching.** Re-run each experiment with the **label-shuffle null** (`--label-shuffle-null`) instead of the contrast-matched null; compare significance/curve_mean to the contrast-matched results already in hand.
- **1.2 — No f1/f2 splits (stim-side unsplit).** Re-run each experiment with stim_l+stim_r unsplit (no choice×feedback conditioning), still preserving stim side. Currently `--unsplit-prior` supports only `phase4` and `absence` — **needs extension** to the S-presence (E3/E4) param sets.
- **1.3 — Full analysis.** BWM functional classification (S/I/M, `--full-analysis`) for all four experiments.

### Key architecture finding (blocks the "just re-run analyses" hope)

Every analysis entry point calls `simulate_condition_sessions(mp, …, rng_seed)` and then analyses the resulting `session_dfs` (per-trial full time-series). **`session_dfs` is never saved to disk** — only the *post-split, post-average* arrays are cached under each experiment's `res/` (e.g. `act_block_duringstim_l_choice_l_f1.npy`).

Consequence:
- `res/` **cannot** be re-split differently (1.2) or re-nulled (1.1) — those need the raw `session_dfs`.
- `--recovery-only` only re-runs analysis on the *existing split arrays*, so it does not help for 1.1/1.2.
- Simulation is deterministic in `(mp, rng_seed, n_sessions, blocks_per_session, max_obs_per_trial)`.

**Decision (matches user's "if not, run simulations once and save"):** add a `session_dfs` cache keyed by that tuple. Simulate each of E1–E4 **once**, persist `session_dfs`, then run all analysis variants (1.1/1.2/1.3 + future) off the cache. This is the generative-model twin of Goal 2's real-data refactor.

### Goal 2 — Real-data analysis: make `block_analysis_allsplits.py` time/storage-efficient

Current flow: `get_all_d_vars(split)` loops all insertions; `get_d_vars(split, pid)` per insertion **re-loads spikes (`load_good_units`) + trials (`load_trials_and_mask`) from scratch for every split**. With N splits × M insertions this is N× redundant loading of the expensive spike/trial data.

**Plan:** two-stage pipeline.
- **Stage 1 (once per insertion):** load spikes (`times`, `clusters`), `clusters` (`cluster_id`, `atlas_id` → acronyms), and the trials table + mask; save a compact per-insertion cache (`insertions_cache/{pid}.*`). Mask depends on alignment event (stim/choice/fback → different `saturation_intervals`); cache per event-type (≤3) or store raw + recompute mask cheaply.
- **Stage 2 (per split, fast):** load cached insertion, do event/trial selection + `bin_spikes2D` + trial averaging + `d_var`/`d_euc`/crossnobis + nrand shuffles. No spike/trial reload.

This makes adding/redoing splits cheap and directly enables Goal 3.

### Goal 3 — Prior modulation by contrast; contrast-response slope

Redo the real-data prior-modulation analysis **separately per contrast** on
**during-trial** splits (`*block_duringstim*`, `*block_stim*duringchoice*`), for
both **act_*** and non-act, including **0% contrast**. Then optionally compute
the CRF slope and test whether the prior modulates that slope.

Names: `'{base}_{contrast}'` (e.g. `act_block_duringstim_l_choice_l_f1_0.125`).
Contrast is parsed from the split name on the Goal-2 cached pipeline — no
`bycontrast` flag needed. See 07-12.

### Updated Goal 3 — Prior modulation at 0% contrast (added 07-17)

Focus the real-data prior-modulation analysis on **0% contrast only**. At zero
contrast, remove the old same-stim-side restriction: stimulus side is not a
meaningful conditioning variable and unnecessarily fragments the trials. For each
choice side separately, compare block L versus block R using all 0%-contrast trials
with that choice:

- choice L: block L versus block R
- choice R: block L versus block R

Thus the new analysis conditions only on a common **choice side**; it does not
retain the old stim-side or f1/f2 subdivisions. Report results both **per individual
region** and **aggregated across all regions**. The earlier per-contrast,
stim-side-conditioned implementation and results below are retained as experiment
history, but they do not answer this updated Goal 3.

### Goal 4 — Presence sweep on S prior-mod params (fitted I/M, stim-side unsplit)

**Question:** With **canonical fitted I/M prior modulation** left on (as in absence/E2), at what `(g_s, d_s)` does **direct P→S coupling** produce a detectable S prior signature under the **stim-side unsplit** pipeline — and specifically when do both **p_mean** (avg) and **p_gain** cross α=0.01?

This is distinct from E3/E4 (`s_presence` with `zero_im_prior_mod=True`, split-conditioned thresholds at g_s≈1800/2025) and from absence unsplit (S n.s. at g_s=d_s=0). Here I/M feedforward/selection effects stay in the model while we sweep only S coupling.

**Model params per run:**
- `g_i, d_i, g_m, d_m` = fitted (from weights JSON; same as absence).
- Sweep `g_s`, `d_s` (2D grid or staged 1D sweeps).
- Default: gain inside adaptation (`gs_outside_adaptation=False`); optionally repeat outside-adaptation arm if thresholds differ materially from June split runs.

**Analysis:** `--unsplit-prior` with `unsplit_mode=stim_side` (stim_l + stim_r, no f1/f2). Contrast-matched null (default); label-shuffle on borderline points if needed. Read out per population: `curve_mean`, `p_mean`, `p_gain` (and `p_offset` for offset-route diagnostics).

**Suggested sweep grid (seed 123, 40 sessions, nrand=100):**
- **Gain-only (`d_s=0`):** g_s ∈ {0, 68, 200, 500, 900, 1200, 1500, 1800, 2025, 2500} (span i-scaled → June split thresholds).
- **Offset route:** at moderate g_s (e.g. 1, 10), d_s ∈ {0, d_i, 2×d_i, 40, 48} (June offset minimums).
- **2D patch:** g_s × d_s around first (p_mean, p_gain) joint-significance hits.

**Deliverable:** threshold table — minimum g_s (gain-only) and minimum (g_s, d_s) pairs for S **p_mean** and **p_gain** significance under unsplit + fitted I/M; compare to E3/E4 zero-I/M split/unsplit thresholds in Table B.

**Implementation (2026-07-07):**
- `--unsplit-prior presence` — single (g_s, d_s) with fitted I/M.
- `--presence-unsplit-sweep` — 2D grid with stim-side unsplit; default **g_s ∈ [0, 2500]**, **d_s ∈ [0, d_i_fitted]** (`default_presence_unsplit_sweep_grid`).
- Outputs: `presence_unsplit_sweep/seed_<seed>/presence_unsplit_sweep.csv` + `_summary.json`.
- Runner: `run_presence_unsplit_sweep.sh` (seed 123, 40 sessions, nrand 100).

```bash
conda activate iblenv
./run_presence_unsplit_sweep.sh
# or custom grid:
python simulate_recovery.py --presence-unsplit-sweep --g-s-grid 0,500,1800,2500 --d-s-grid 0,10,21.56
```

---

## Plan / execution order

1. ~~**Goal 1**~~ — DONE (session cache, matrix, comparison tables).
2. **Goal 2 & 3 validation** in working env (BWM `SessionLoader` compat).
3. **Goal 4:** ~~wire presence unsplit case~~ DONE; ~~run full sweep~~ DONE (see 07-07c).

Runs: `conda activate iblenv`, outside sandbox, seed 123, 40 sessions, nrand=100, n-jobs 8.

---

## Progress log

### 2026-07-06a — Session cache refactor (Goal 1 infra) — DONE + validated

Implemented the "simulate once, reuse across analyses" layer in `simulate_recovery.py`:

- **`simulate_condition_sessions(...)` is now cache-aware.** Deterministic draws are persisted to `<manifold_sim>/session_cache/{key}.pkl.gz` (+ `{key}.json` sidecar), keyed by a sha1 of `(mp, seed, n_sessions, blocks_per_session, max_obs_per_trial, min_trials_per_session, constant_s0, dt)` and `SESSION_CACHE_VERSION`. Raw simulation moved to `_simulate_condition_sessions_raw`.
- **`process_condition` now routes through the cache** (its inlined per-session loop was removed), so absence/presence/s-presence share the same cache as phase4/unsplit.
- **Replicate-null loop uses `use_cache=False`** (high seed cardinality → avoid bloat).
- **CLI `--no-session-cache`** to disable (default: enabled).
- Bumped nothing else; contrast-matched null remains default.

**Validation (iblenv, outside sandbox):**
- Phase4 seed 999, 2 sessions: run 1 → `[session cache MISS] … saved`; run 2 → `[session cache HIT]`. ✓
- New `--unsplit-prior s_presence --g-s-presence 1800 --d-s-presence 0`: runs end-to-end, writes `s_presence_g_s1800_d_s0_unsplit/`, and reuses the same cache key as a future full s_presence run at matching (mp, seed, n_sessions). ✓
- Smoke artifacts cleaned up.

Cache size ≈ 15 MB (gzip) per 2-session draw → ≈ **300 MB per 40-session experiment**, ≈ 1.2 GB for E1–E4. Acceptable storage-for-time tradeoff.

> Note: the phase4 output path (`absence/figs/phase4_no_prior_mod/`) has no seed component, so the 2-session smoke run overwrote the old E1 baseline dir. E1 will be regenerated cleanly in the runs below.

### `--unsplit-prior` extended (Goal 1.2 for E3/E4) — DONE

Added a `s_presence` case to `run_unsplit_prior_distance_analysis` + CLI. It builds `mp` via `load_fitted_model(g_s, d_s, zero_im_prior_mod=True, gs_outside_adaptation=…)` and tags output `s_presence_g_s{…}_d_s{…}[_gs_free]_unsplit`. This is what lets 1.2 (stim-side unsplit) run on the gain-only threshold experiments, which the old code could not do.

### Wiring gaps discovered for the Goal-1 matrix (need resolution before full runs)

The three analyses do **not** all map cleanly onto existing subcommands for all four experiments:

| Analysis | E1 phase4 | E2 absence | E3/E4 s_presence |
|----------|-----------|-----------|------------------|
| baseline contrast-matched | `--phase4-no-prior-mod` ✓ | default path (runs presence too) ⚠ | `s_presence_tune` res exists ✓ |
| **1.1** label-shuffle null | `--phase4-no-prior-mod --label-shuffle-null` ✓ | needs **absence-only** entry ⚠ | `--s-presence-tuned-plots … --label-shuffle-null`? (verify) |
| **1.2** stim-side unsplit | `--unsplit-prior phase4` ✓ | `--unsplit-prior absence` ✓ | `--unsplit-prior s_presence …` ✓ (new) |
| **1.3** full classification | **no path** ✗ | `--full-analysis` default (also presence) ⚠ | **no path** ✗ |

- **1.3 (`--full-analysis`) is only wired into the default absence/presence path** (`process_condition` with `s_prior_only=False` → `classify_regions` BWM Σ). Phase4 (E1) and s_presence (E3/E4) have no classification path — needs a small wiring addition (e.g. `--full-analysis` support on `--phase4-no-prior-mod` and a full s_presence entry).
- **E2 canonical absence** currently only runs bundled with presence + replicate null. A cheap `absence`-only convenience (reuse the cache, skip presence) would make 1.1/1.3 clean for E2.

Proposed next step: add these small wirings (full-analysis for phase4 + s_presence; absence-only run), then execute the uniform 3×4 matrix (all cache-shared per experiment). See "Next steps".

### 2026-07-06b — Unified experiment runner + matrix launched

Resolved the wiring gaps with a single cache-backed entry point rather than per-experiment special cases:

- **`run_experiment_case` + `--run-experiment {phase4,absence,s_presence}`.**
  - **sprior mode** (default): S/I/M split-conditioned (f1/f2) prior distance via the shared `_run_split_population_prior_distance` helper (extracted from `run_phase4`), under contrast-matched (baseline) or `--label-shuffle-null` (1.1).
  - **full mode** (`--full-analysis`): BWM Σ classification via `classify_regions` (1.3), reusing `process_condition` (rebuilds identical `mp` → session-cache HIT).
  - Outputs: `goal1/<exp_tag>/<null>_<mode>/` (`null` ∈ {cm, ls}; `mode` ∈ {sprior, full}).
- `process_condition` generalized with `zero_all_prior_mod` + `gs_outside_adaptation`.
- All four experiments now share one simulation per `mp` (unsplit + sprior + full all HIT the same cache key).
- Validated: phase4 full-analysis path runs (real-params BWM classification via `bwm_classification.csv` / `classification_details.csv`; tiny-params `amp_slope` failure is only from too-few-trial splits — the standard 150 ms `duringstim` timeframe has exactly 72 bins = `n`). Unified sprior S/I/M + label-shuffle validated end-to-end. Smoke artifacts cleaned.

**Matrix launched** via `run_goal1_matrix.sh` (background, seed 123, 40 sessions, nrand 100, n-jobs 8). 20 runs = 4 experiments × {cm_sprior (baseline), ls_sprior (1.1), cm_unsplit + ls_unsplit (1.2), cm_full (1.3)}. Status log: `manifold_sim/goal1/_logs/matrix_status.log`.

## Results

_(matrix running — to be filled from `goal1/_logs/` and `goal1/<exp>/<variant>/` summaries)_

### 2026-07-06c — Goal 2: block_analysis_allsplits.py efficiency refactor

**Problem:** the old `__main__` looped ~20 splits, each calling `get_all_d_vars(split)` which loops all ~500 BWM insertions and re-runs `load_good_units` + `load_trials_and_mask` **per split** → ~20× redundant loading of the expensive spike/trial data.

**Refactor (implemented):**
- `saturation_for_split(split)` — the stim/move/feedback saturation key.
- `build_insertion_cache(pid)` — loads an insertion's raw data **once** (spikes `times`/`clusters`, clusters `cluster_id`/`atlas_id`, and bad-trial-masked trials for each of the 3 saturation types) and persists to `manifold/insertion_cache/{eid_probe}.npy`.
- `get_d_vars(..., cached=None)` — when a cache is passed, reuses spikes/clusters/trials (no reload); else loads as before. Downstream binning/averaging/d_var/xnobis code is unchanged → identical output.
- `get_all_d_vars_allsplits(splits_list)` — **loop reorder**: outer loop = insertions (load once), inner loop = all splits. Same per-split output layout `manifold/{split}/{eid_probe}.npy`. `restart` skips already-computed (split, insertion).
- `cache_all_insertions()` — pre-build all per-insertion caches (for adding future splits without reloading).
- `__main__` now guarded (`if __name__ == '__main__':`, so the module is importable) and switched to the reordered driver; pooling (`d_var_stacked` / `d_var_stacked_multi`) unchanged.

Expected speedup ≈ N_splits× fewer spike/trial loads (the dominant cost). Storage: small per-insertion `.npy` caches (spikes reused in-memory within a pass; persisted cache enables future splits).

**Validation status:** compiles; `build_insertion_cache` successfully runs `load_good_units` (spike load works). Full numeric cached-vs-original comparison is **blocked by a pre-existing env issue**: vendored `brainwidemap/bwm_loading.py:285` calls `SessionLoader(..., revision=MODIFIED_BEFORE)` but the installed `one-api` `SessionLoader` has no `revision` kwarg → `load_trials_and_mask` raises `TypeError`. The **original** `get_d_vars` hits this identically (unrelated to the refactor). Needs env/version fix (or a compat shim) before an end-to-end numeric check.

### 2026-07-06d — Goal 3: contrast-conditioned prior modulation + CRF slope test

Built on the Goal-2 insertion cache. Two additions to `block_analysis_allsplits.py`:

**(A) Prior modulation stratified by contrast (reuses existing pipeline).**
`CONTRASTS = [1.0, 0.25, 0.125, 0.0625, 0.0]`. New split family `block_only_c{contrast}` (and `act_block_only_c{contrast}`), registered in `align`/`pre_post` (ITI window, as base `block_only`). In `get_d_vars`, the `block_only` branch now parses a trailing `_c{val}` and filters trials to `|contrast| == val` before the block-L-vs-R split. So block-prior distance (d_var / d_euc / crossnobis + shuffle null, pooled by `d_var_stacked`) can be computed **separately per contrast, including 0%** — with zero changes to the tested distance machinery.

**(B) Contrast-response function (CRF) slope + prior-modulation-of-gain test (new).**
- `get_crf_slope(pid, cached=...)`: per region, single-bin post-stim response (default window `[0, 0.15]`) as a function of contrast, computed separately for **concordant** (block favors stim side) vs **discordant** priors. Fits OLS slope of response vs contrast for each; prior modulation of gain = `slope_conc − slope_disc` averaged over L/R sides. Significance via a null that **shuffles concordant/discordant block labels within each (side, contrast) cell** (preserves side/contrast structure). Returns per region CRF curves, slopes, `slope_mod`, `p_slope_mod`.
- `get_all_crf_slope(...)`: per-insertion driver (uses the cache), saves `manifold/crf_slope/{eid_probe}.npy`.
- `crf_slope_stacked(...)`: pools across insertions per region (nanmean `slope_mod`, mean p, `frac_sig`, mean CRF curves) → `manifold/res/crf_slope_stacked.npy`.

**Design notes / choices to confirm:** concordant = high prior for the stimulus side; slope fit is linear in raw contrast (not log); response is total spike count in a single `[0,0.15]` bin. 0% contrast anchors the CRF low end (fully prior-driven). Not run here (deferred to working env per Goal-2 decision). Compiles; no new lint errors.

### 2026-07-06e — Goal 1.3 bug fix: compute_amp_slope on short (80 ms) S splits

The `--full-analysis` classification path **failed at real params** (not just tiny tests): `analysis_functions.compute_amp_slope` did `np.polyfit(np.arange(n), r[-n:], 1)` with fixed `n` (20, and 5/10), but the short 80 ms `duringstim` region curves (`act_block_duringstim`, `stim_duringstim_short_act`) have fewer than 20 bins → "expected x and y to have same length" → `amp_slope` never set → `manifold_to_csv` `KeyError: 'amp_slope'`.

**Fix:** clamp each fit window to `min(k, len(r))` (helper `_last_slope`), return NaN if <2 points. Backward-compatible (identical when `len(r) >= k`). The 4 `*_cm_full` matrix runs used the pre-fix code and FAILED; queued `run_goal1_full_refix.sh` to re-run only those 4 after the matrix completes (full path wipes `res_dir` and recomputes, ~8 min each; session sim still cache-hit).

### 2026-07-06f — Goal 1.3 bug #2: pooled prior-distance plot vs mixed population windows

After the `amp_slope` fix, classification passed but `*_cm_full` still crashed later in `plot_recovery_figures`: `ax.plot(t_stim, mean_c)` → `x and y must have same first dimension, (36,) vs (72,)`. Cause: under the canonical analysis each split's combined regde holds curves of **two lengths** — S population = 80 ms window (36 bins), I/M/P = 150 ms (72 bins) — but the plot built a single `t_stim` from one arbitrary region and reused it for all S/I/M groups. Confirmed empirically: per-split regde length dist `{36: 1, 72: 3}`.

**Fix:** build the time axis **per group** (S uses the `_short` 0–80 ms axis, I/M the 0–150 ms axis) and pool only same-length curves. No CSVs had been saved (classification outputs are written *after* this plot), so the crash blocked the actual deliverables — hence the re-run is required. Re-launched `run_goal1_full_refix.sh` with both fixes.

### 2026-07-06g — Goal 1.3 COMPLETE (all 4 experiments, both fixes)

With the `amp_slope` clamp + per-group plot-axis fixes, all four `*_cm_full` runs completed OK (13:35–13:58) and wrote full deliverables (`figs/bwm_classification.csv`, `classification_details.csv`, `population_prior_tests.csv`, `prior_modulation.csv`, `s_prior_stats.csv`, `summary.json`).

Goal 1 is now fully done: 1.1 (contrast-matched vs label-shuffle), 1.2 (unsplit), 1.3 (BWM classification recovery) all computed for all 4 experiments; simulate-once session cache reused throughout. *(Initial matrix predates decorrelation-window fixes in 07-06i/07-07; absence re-validated there.)*

### 2026-07-06h — Goal-1 comparison tables (all 4 experiments, seed 123, n_sessions 40, nrand 100)

Data source: `goal1/{exp}/{cm,ls}_sprior/*_summary.json` (split) and `unsplit_prior/seed_123/{exp}_unsplit/*_summary.json` (unsplit; CM preserved as `*_CM_summary.json` after re-run since the ls run shares the path). **`curve_mean` is null-independent** (identical under CM/LS) — only p-values change with the null.

**Table A — SPLIT-conditioned (S/I/M split), contrast-matched (CM) vs label-shuffle (LS) null**

| exp | pop | curve_mean | p (CM) | p (LS) | sig CM | sig LS |
|-----|-----|-----------:|-------:|-------:|:------:|:------:|
| phase4 | S | 0.0124 | 0.78 | 0.88 | – | – |
| phase4 | I | 0.0037 | 0.60 | 0.69 | – | – |
| phase4 | M | 0.0042 | 0.81 | 0.86 | – | – |
| absence | S | 0.798 | 0.00 | 0.00 | ✓ | ✓ |
| absence | I | 0.492 | 0.00 | 0.00 | ✓ | ✓ |
| absence | M | 2.028 | 0.00 | 0.00 | ✓ | ✓ |
| s1800 | S | 0.0418 | 0.04 | 0.04 | – | – |
| s1800 | I | 0.0162 | 0.02 | 0.07 | – | – |
| s1800 | M | 0.0260 | 0.00 | 0.03 | ✓ | – |
| s2025 | S | 0.0542 | 0.00 | 0.00 | ✓ | ✓ |
| s2025 | I | 0.0194 | 0.00 | 0.02 | ✓ | – |
| s2025 | M | 0.0316 | 0.00 | 0.02 | ✓ | – |

**Table B — UNSPLIT (stim-side, no f1/f2), CM vs LS**

| exp | pop | curve_mean | p (CM) | p (LS) | sig CM | sig LS |
|-----|-----|-----------:|-------:|-------:|:------:|:------:|
| phase4 | S | 0.00284 | 0.64 | 0.75 | – | – |
| phase4 | I | 0.00270 | 0.17 | 0.24 | – | – |
| phase4 | M | 0.00390 | 0.20 | 0.44 | – | – |
| absence | S | 0.0111 | 0.13 | 0.05 | – | – |
| absence | I | 1.099 | 0.00 | 0.00 | ✓ | ✓ |
| absence | M | 3.078 | 0.00 | 0.00 | ✓ | ✓ |
| s1800 | S | 0.0197 | 0.00 | 0.01 | ✓ | – |
| s1800 | I | 0.00817 | 0.00 | 0.02 | ✓ | – |
| s1800 | M | 0.0125 | 0.00 | 0.04 | ✓ | – |
| s2025 | S | 0.0250 | 0.00 | 0.00 | ✓ | ✓ |
| s2025 | I | 0.00995 | 0.00 | 0.02 | ✓ | – |
| s2025 | M | 0.0158 | 0.01 | 0.02 | – | – |

**Table C — SPLIT vs UNSPLIT curve_mean (CM)** — effect of dropping f1/f2 splits

| exp | pop | split | unsplit | ratio |
|-----|-----|------:|--------:|------:|
| absence | S | 0.798 | 0.0111 | **0.014** |
| absence | I | 0.492 | 1.099 | 2.2 |
| absence | M | 2.028 | 3.078 | 1.5 |
| s1800 | S | 0.0418 | 0.0197 | 0.47 |
| s1800 | M | 0.0260 | 0.0125 | 0.48 |
| s2025 | S | 0.0542 | 0.0250 | 0.46 |
| s2025 | M | 0.0316 | 0.0158 | 0.50 |

**Findings**
1. **Null choice:** label-shuffle is systematically **more conservative** (higher p) than contrast-matched. Borderline S-presence effects (s1800 M split; s1800 I/M/S unsplit; s2025 I/M) are significant under CM but **not** under LS. phase4 (no prior mod) and absence are robust to the null choice.
2. **Split vs unsplit:** the **absence-S** signal is enormous when split (curve_mean 0.798) but **collapses to ~0.011 (non-sig) when unsplit** (ratio 0.014) — i.e. absence-S is driven by the f1/f2 (choice×feedback) splits, not stim-side alone. absence I/M instead grow unsplit. s-presence S/M roughly halve unsplit but S stays significant under CM.
3. **Classification (1.3):** BWM Σ functional recovery (S/I/M) in all 4 experiments (see 07-06g matrix; corrected decorrelation path in 07-07 for absence). Prior population P excluded.

**Caveat:** in the original matrix, `ls_unsplit` overwrote `cm_unsplit` (shared output dir, no null tag). Re-ran the 4 CM unsplit and preserved as `*_CM_summary.json`. Future matrices should write null-specific unsplit dirs.

### 2026-07-06i — Fix decorrelation PRE_POST + absence rebuild

**Bug:** `stim_choice_*_act` / `choice_duringstim_*_act` matched `block` in name → `PRE_POST=[0.4,-0.1]` (ITI) but `split_n_bins` used 0.5s → 240 bins on ~150-step segment → alternating empty/filled bins → oscillating d_euc.

**Fix:** in `build_align_pre_post()`, decorrelation splits (`stim_choice_*`, `choice_duringstim_*`, `stim_block_*_act`) now get `[0, 0.15]` (72 bins, during-stim window).

**Rebuilt:** `goal1/absence/cm_full` (session cache HIT, ~5.4 min). Curves smooth (len=72, no oscillation).

**BWM classifier (corrected, S/I/M):** **3/3 perfect** — S→stimulus (Σ=0.99), I→integrator (Σ=0.42), M→movement (monotonicity=1).

Plots: `.../absence/cm_full/figs/sim_duringstim_stim_choice_d_euc_SIM.png`, `bwm_classification.csv`.

### 2026-07-07 — Short-window + combine fixes for decorrelation panels

**`_short` splits:** genuine 80 ms `PRE_POST` (not plot-axis only) → `stim_duringstim_short_act` now differs from `stim_duringstim_act` (36 vs 72 bins for I/M).

**`stim_block_*` (d^stim,se'):** 80 ms early window (was 150 ms); only used in `stim_duringstim1_act`.

**`stack_combined_timeframes`:** average across L/R splits (`/ n_stacked`), matching `analysis_functions.plot_regional_distance` — fixes spurious 0.5× amplitude for 2-split vs 4-split timeframes.

**Rebuilt:** `goal1/absence/cm_full` (~4.7 min, session cache HIT). Panels 1≠3, panel 4 now 80 ms; se' still correlates with choice (same trial pools per stim side) but no longer half-amplitude artifact.

**Classification (absence, post-fix rebuild)** — BWM Σ classifier, S/I/M only (prior population P excluded):

| accuracy | S | I | M |
|----------|---|---|---|
| **3/3** | S→stimulus ✓ | I→integrator ✓ | M→movement ✓ |

**BWM metrics** (`figs/bwm_classification.csv`):

| pop | true | pred | Σ^stim,s | Σ^stim,m | monotonicity | sc_duringstim | sc_duringchoice |
|-----|------|------|----------|----------|--------------|---------------|-----------------|
| S | S | S | **0.993** | 0.551 | 0 | 0.007 | 0.449 |
| I | I | I | 0.351 | 0.143 | 0 | 0.649 | 0.857 |
| M | M | M | 0.331 | 0.154 | **1** | 0.669 | 0.845 |

Compared to **07-06i** (PRE_POST fix only, pre-short/combine): S Σ unchanged (0.99); I/M Σ dropped 0.42→0.35/0.33 because denominators now include distinct d^stim,se (80 ms) and averaged (not summed) multi-split amps. S/I/M assignments unchanged.

**Prior modulation** (act_block_duringstim, all sig at p≈0.01): S amp 0.950, I 0.106, M 0.379.

Plots: `.../absence/cm_full/figs/sim_duringstim_stim_choice_d_euc_SIM.png`, `bwm_classification.csv`.

**Note:** only **absence** `cm_full` had been rebuilt with all decorrelation fixes at this point; phase4 / s1800 / s2025 pending (done 07-07b below).

### 2026-07-07b — BWM classification audit + all four `cm_full` rebuilds (session cache)

**Classifier sanity check (`classify_regions`):** rules match hand-computed Σ thresholds on absence (σ_stim,s>0.8 → S; σ_stim,m≤0.8 ∧ monotonicity → M; else integrator). Uses raw `amp_euc` without significance masking for assignment; P excluded. Implementation OK.

**Additional decorrelation fix:** 80 ms S cap in `build_population_b_for_split` limited to prior-distance splits (`act_block_duringstim_*`) only — not decorrelation splits. Fixes d^stim,s vs d^stim,se identity for S.

**Plot:** `plot_bwm_decorrelation_curves` wired into `plot_recovery_figures`; default populations S/I/M; independent axes; saves `sim_duringstim_stim_choice_d_euc_SIM.png`.

**Rebuilds** via `run_goal1_cm_full_rebuild.sh` (phase4, s1800, s2025; absence already done). All **session cache HIT**; ~5 min each for `res/` recompute (no re-simulation).

**Table — BWM Σ classification (all 4 experiments, seed 123, decorrelation fixes)**

| exp | acc | S true→pred | I | M | why failures |
|-----|-----|-------------|---|---|--------------|
| phase4 | **1/3** | S→**I** (Σ^stim,s=0.59) | I→I ✓ | M→**I** (mono=0) | No prior mod: stim/choice decorrelation amps tiny & similar; M lacks pre-movement ramp |
| absence | **3/3** | S→S ✓ (Σ=0.99) | I→I ✓ | M→M ✓ (mono=1) | Fitted I/M prior mod gives separable signatures |
| s1800 | **1/3** | S→**I** (Σ=0.64) | I→I ✓ | M→**I** (mono=0) | P→S only, I/M prior mod off: S below σ threshold |
| s2025 | **1/3** | S→**I** (Σ=0.66) | I→I ✓ | M→**I** (mono=0) | same as s1800 |

**Interpretation:** classifier **works as designed**; perfect recovery requires absence-like fitted I/M prior modulation to sculpt distinct stim/choice/movement decorrelation. Phase4 and s_presence (zero I/M prior mod) are negative controls for recovery — only I integrator label is stable.

**Plots (all under `goal1/<exp>/cm_full/figs/`):** `sim_duringstim_stim_choice_d_euc_SIM.png`, `bwm_classification.csv`, `classification_confusion.png`, `prior_distance_by_group.png`.

### 2026-07-07c — Goal 4 presence unsplit sweep COMPLETE

**Run:** `./run_presence_unsplit_sweep.sh` — seed 123, 40 sessions, nrand 100, stim-side unsplit, fitted I/M, contrast-matched null. **80/80 pairs** in ~75 min (09:11–10:26). Outputs: `presence_unsplit_sweep/seed_123/presence_unsplit_sweep.csv`, `_summary.json`.

**Summary counts:** 46/80 pairs with S **p_mean** and **p_gain** both significant (α=0.01); always co-occur in this grid.

**Table D — S significance thresholds (fitted I/M, stim-side unsplit, CM null)**

| route | threshold | S curve_mean | p_mean | p_gain | notes |
|-------|-----------|-------------:|-------:|-------:|-------|
| baseline | g_s=0, d_s=0 | 0.0056 | 0.13 | 0.13 | n.s. — matches absence unsplit (Table B) |
| offset-only | g_s=0, d_s=d_i | 0.0105 | 0.00 | 0.00 | sig at **max d_s only**; intermediate d_s n.s. at g_s=0 |
| gain-only | d_s=0, g_s≥**1200** | 0.0094 | 0.00 | 0.00 | first both-sig; g_s=900 borderline (p≈0.06) |
| gain-only | d_s=0, g_s=1800 | 0.0223 | 0.00 | 0.00 | |
| gain-only | d_s=0, g_s=2025 | 0.0282 | 0.00 | 0.00 | |
| mixed | d_s=d_i, any g_s≥0 | 0.0105+ | 0.00 | 0.00 | offset at d_i dominates; low g_s sufficient |

**Min g_s for both-sig at each d_s:**

| d_s | min g_s (both sig) |
|-----|-------------------:|
| 0 | 1200 |
| 5.4 | 1200 |
| 10.8 | 900 |
| 16.2 | 379 |
| 21.6 (d_i) | **0** |

**Findings**
1. **Gain-only with fitted I/M:** S both-sig threshold drops to **g_s≈1200** (vs g_s≈1800 for zero-I/M `s_presence` unsplit in Table B) — I/M context makes moderate direct P→S gain easier to detect, but still requires **~6× g_i_fitted**.
2. **Offset route is narrow:** only **d_s=d_i** at g_s=0 reaches significance; sub-max offsets fail at g_s=0 despite I/M on. Likely needs near-full integrator-scale offset for a detectable S level shift.
3. **Baseline (g_s=d_s=0)** remains n.s. on S — fitted I/M alone does not create stim-side-unsplit S prior distance (consistent with absence unsplit).
4. **p_mean and p_gain always co-significant** in this grid — no case of gain-only significance without mean significance.

**Compare to zero-I/M unsplit (Table B):** s1800 S curve_mean=0.0197; presence+g_s=1800+d_s=0 curve_mean=0.0223 — similar magnitude, but presence hits sig at lower g_s (1200 vs 1800) when I/M is on.

**Example plots (2026-07-08):** `run_presence_unsplit_examples.sh` → `presence_unsplit_sweep/seed_123/examples/` (session cache HIT). Key figures per case (`presence_g_s{1200,1800}_d_s0_unsplit/figs/`): `s_prior_curve.png`, `s_shuffle_control.png`, `presence_*_curve_mean_comparison.png`, `presence_*_shuffle_controls.png`, `block_confounds/p_block_s_trajectory_*.png`.

### 2026-07-09 — Goal 2 VALIDATED locally (ibllib 4.0.1 / ONE-api 3.5.2)

**Env fix:** upgraded `ibllib` 2.38 → 4.0.1 (adds `SessionLoader(..., revision=)`). Also patched `str(eid)` when filtering aggregate trials table (ONE-api 3.x returns `uuid.UUID` from `pid2eid`).

**Parity (`scripts/validate_goal2_cache.py`):** 5 insertions × 4 splits × cached vs uncached, `nrand=20`, with null shuffles — **all passed** (~4.6 min). Cache trials table check OK on 3 insertions.

**Speed benchmark (`scripts/benchmark_goal2_10splits.py`, 5 insertions, 10 splits, `control=False`):**

| insertion | old (10 reloads) | new (1 load + 10 splits) | speedup |
|-----------|-----------------:|-------------------------:|--------:|
| 1 | 47s | 10.5s | 4.5× |
| 2 | 51s | 12.8s | 4.0× |
| 3 | 51s | 11.8s | 4.5× |
| 4 | 68s | 16.9s | 4.1× |
| 5 | 86s | 27.6s | 3.1× |
| **mean** | **61s** | **15.9s** | **3.8×** |

Extrapolation to full BWM (699 insertions, 10 splits): **~12 h → ~3 h** wall time. Load component drops from ~55 s to ~5 s per insertion (~90% redundant I/O removed).

**Storage:** one insertion cache ≈ **30 MB** (`manifold/insertion_cache/{eid_probe}.npy`); full BWM cache ≈ **22 GB** one-time. Per-split outputs (`manifold/{split}/*.npy`) unchanged. Tradeoff: +22 GB cache vs ~4× faster multi-split runs; adding splits 11–20 is nearly free once cache exists (`restart=True` skips existing outputs).

**Caveat:** benchmark used `control=False` (no 2000-shuffle null). Production `control=True` adds per-split compute that is identical in old/new paths — speedup on full runs will be **load-dominated** (still ~N_splits× on reload, but smaller fraction of total when null loop is on).

### 2026-07-10 — Goal 2: null-loop fix + nrand=2000 validation

**Bug:** batched-null refactor briefly used **region-outer × null-inner** loop order, recomputing `b[ys].mean()` per region per null (~60 regions × 2000 nulls). That inflated one `block_only` call to **~147 s** at nrand=2000.

**Fix:** restored original order — **null-outer × region-inner** in `_compute_control_D()` / `_append_perm()`. One full-tensor mean/var per null, then regional metrics.

**nrand=2000 parity (`validate_goal2_cache.py`, 5 pids × 4 splits):** **all 20 pairs passed** (~71 min wall). Cached vs uncached outputs match bit-for-bit (with seed reset).

**nrand=2000 timing (1 pid, `block_only`):**

| path | time |
|------|-----:|
| uncached | ~64 s |
| cached (after 8 s load) | ~61 s |

Scaling: nrand=100 → 3.8 s, nrand=500 → 15.7 s, nrand=2000 → 60 s (~linear in nrand). **No regression vs pre-refactor** — user's "<1 min per insertion" holds for a single split.

**3-split bench (`test_goal2_nrand2000.py`):** uncached 155 s, cached 149 s (load 8 s) → **1.04× speedup**. At nrand=2000 the null loop (~50 s/split) dominates; cache only saves reload I/O (~8 s once). Multi-split speedup from Goal 2 is **load-dominated when control=False**; with control=True the win is smaller per insertion but still ~8 s × (N_splits−1) saved across splits.

**Scripts:** `scripts/test_goal2_nrand2000.py` (parity + bench + optional storage); import fix for `validate_goal2_cache.compare_d`.

### 2026-07-10b — Goal 2 end-to-end comparison (alyx ONE cache, 5 insertions, nrand=2000)

**Test ONE root:** `https://alyx.internationalbrainlab.org` → `/Users/ariliu/Downloads/ONE/alyx.internationalbrainlab.org`. (Earlier openalyx runs used a different cache root; alyx chosen for apples-to-apples comparison with pre-Goal-2 baseline.)

**Scripts:** `scripts/_original_pipeline_worker.py` (isolated ee849e0 pipeline), `scripts/compare_alyx_pipeline.py` (new vs original + 2-split aggregate).

#### Storage test (stream_pool, openalyx reference, 5 insertions, 1 split)

| metric | value |
|--------|------:|
| Wall time | 404.5 s (~6.7 min) |
| Peak RSS | 2429.8 MB |
| Insertion cache | 322.4 MB (5 files) |
| Stream acc checkpoint | 135.0 MB |
| Final res | 24.1 MB |
| **Total disk** | **481.5 MB** |
| Per-split insertion files | **0** (stream_pool skips `manifold/{split}/*.npy`) |

#### Original baseline (`block_only`, ee849e0, uncached)

5 insertions → per-insertion `manifold/block_only/{eid_probe}.npy` + `d_var_stacked` → `manifold/res/block_only*.npy`.

| metric | value |
|--------|------:|
| Wall time | 415.1 s (~6.9 min) |
| Per-insertion mean | 82.8 s |
| `d_var_stacked` | 0.7 s |
| Peak RSS | 3735.4 MB |
| Per-ins files | 231.2 MB (5 × ~46 MB) |
| Final res | 24.1 MB |

#### Split 1: `block_only` — original vs new (stream_pool + insertion cache)

Final outputs (`res/` vs `res/_compare_stream/`): **OK — 10 regions match**.

| | Original | New |
|--|--:|--:|
| Wall time | 415.1 s | 488.8 s |
| Peak RSS | 3735 MB | 2423 MB |
| Disk (intermediates + final) | 255 MB | 482 MB |

Split 1 new path **slower** — pays one-time insertion-cache build (~322 MB). **~1.3 GB less peak RAM** (no materialised `ws` tensor for 2000 nulls).

#### Split 2: `block_duringstim_l_choice_l_f1` — same 5 insertions, cache reused (`restart=True`)

Final outputs: **OK — 10 regions match**.

| | Original | New |
|--|--:|--:|
| Wall time | 283.9 s | **75.8 s** |
| Per-insertion mean | 56.6 s | **15.1 s** |
| Cache load mean | (full reload) | **0.0 s** |
| Peak RSS | 2972 MB | 1457 MB |
| Per-ins files | 120.6 MB | 0 (stream_pool) |

Split 2 new path **3.7× faster** with cache hit.

#### Two-split aggregate (same 5 insertions)

| metric | Original | New (stream_pool) |
|--------|----------:|------------------:|
| **Total wall time** | **699 s (11.7 min)** | **565 s (9.4 min)** → **1.24× faster** |
| Peak RSS (max of runs) | 3735 MB | 2422 MB |
| Total disk | 388 MB | 561 MB |
| Per-split insertion files | 10 (5×2) | 5 (split1 orig only; new writes 0) |

Per-split timing: `block_only` orig 415 s / new 489 s; `duringstim_l_f1` orig 284 s / new **76 s**.

#### Stream acc checkpoints (`manifold/res/_stream_acc/{split}.npy`)

Incremental pool state saved after each insertion (replaces per-insertion `manifold/{split}/*.npy` + separate `d_var_stacked` pass). Contents: `pooled_keys`, `acs`/`acs1`, `ws`, `regdv0`/`regde0`, `uperms`. `finalize()` writes `manifold/res/{split}*.npy`.

**Size vs per-insertion files (5 insertions, excluding insertion cache):**

| split | per-ins files | stream acc | ratio |
|-------|-------------:|-----------:|------:|
| `block_only` | 231 MB | 135 MB | 0.58× |
| `block_duringstim_l_choice_l_f1` | 121 MB | 68 MB | 0.56× |

Stream acc **~55–60%** the size of per-split insertion files for the same insertions.

#### Storage break-even (full BWM, 699 insertions)

Formula (intermediates kept): `N × (P − S) = C` where `C` = cache bytes/insertion, `P` = per-ins file bytes/insertion/split, `S` = stream_acc bytes/insertion/split.

| cache estimate | avg P, S from test | **N_splits to break even** |
|----------------|-------------------|-----------------------------|
| ~30 MB/ins (~22 GB total) | 35 / 20 MB | **~2 splits** |
| ~64.5 MB/ins (alyx test avg) | 35 / 20 MB | **~4 splits** |
| 64.5 MB/ins, `block_only`-like only | 46 / 27 MB | **~3 splits** |
| 64.5 MB/ins, small splits only | 24 / 14 MB | **~6 splits** |

If stream acc deleted after `finalize()`, persistent new storage ≈ cache + final res only → break-even **~1–2 splits**. Insertion cache is one-time; splits beyond break-even add mainly stream_acc (during run) + small final res.

### 2026-07-12b — Goal 2: insertion sharding for ORCD (`mit_normal` 12 h)

**Problem:** Full BWM (~699 insertions) × nrand=2000 for one split is ~19 h wall (~100 s/ins) — exceeds `mit_normal` max **12 h**. Parallelizing across splits alone is not enough; jobs timed out mid-split (e.g. `act_block_stim_r` ~463/699). Kill mid-`np.save` also left truncated `_stream_acc/{split}.npy` (`pickle data was truncated` on restart).

**Fixes / features:**
1. **Atomic stream_acc save** — write `.{split}.tmp.{pid}.npy` then `os.replace`; corrupt load quarantines to `*.corrupt.{pid}` and starts empty.
2. **Delete stream_acc after successful finalize** — once `manifold/res/{split}.npy` exists.
3. **Insertion sharding** — `shard_idx` / `n_shards` on `get_all_d_vars_allsplits`: each job processes `eids_plus[k::N]`, writes `_stream_acc/{split}.shard{k}.npy` (no finalize). `finalize_stream_shards(split)` merges disjoint shards (+ optional leftover unsharded `{split}.npy`) → `res/{split}*.npy`, then cleans checkpoints. CLI: `--shard-idx` / `--n-shards` / `--finalize-only` / `--no-finalize`.

**Scripts (promote to `main` for cluster):**
- `scripts/run_goal2_shard_slurm.sh` — one shard
- `scripts/run_goal2_finalize_slurm.sh` — merge + finalize
- `scripts/submit_goal2_stimOn_act_sharded.sh` — default **N_SHARDS=4** × 6 stimOn_act splits + finalize deps
- `scripts/run_goal2_cache_slurm.sh` — cache-only (restart skips existing)

**Expected timing:** 4 shards → ~5 h/shard at ~100 s/ins (fits 12 h). Do **not** mix sharding with an existing good unsharded `{split}.npy` restart (duplicate keys); continue timed-out splits unsharded with `restart=True`, or delete corrupt checkpoints and shard from scratch.

```bash
# New / full redo (cache already built):
bash scripts/submit_goal2_stimOn_act_sharded.sh
# N_SHARDS=6 bash scripts/submit_goal2_stimOn_act_sharded.sh
```

## Next steps

1. ~~Goals 2 validation in working env~~ DONE (07-09).
2. ~~nrand=2000 parity + timing~~ DONE (07-10).
3. ~~End-to-end original vs new pipeline comparison (alyx)~~ DONE (07-10b).
4. ~~Insertion sharding + atomic stream_acc~~ DONE (07-12b).
5. ~~Old Goal 3: contrast-conditioned **during-trial** prior mod on cached + **sharded** pipeline~~ DONE (07-12 + 07-14 tables; retained as diagnostic).
6. Updated Goal 3: 0%-contrast block L/R within each choice side, without stim-side
   or f1/f2 conditioning; report per-region and all-region aggregate results.
7. Optional: CRF slope test (`get_crf_slope`) after contrast splits land.
8. Optional: finish ORCD `stimOn_times_act` BWM with sharded submit (or unsharded restart where checkpoints are valid).

### 2026-07-12 — Goal 3 corrected: during-trial contrast splits (not ITI block_only)

**Clarification:** Goal 3 prior-modulation-by-contrast is for **during-trial** splits —
`*block_duringstim*` and `*block_stim*duringchoice*` — both **act_*** (action-kernel
prior) and non-act. The 07-06d `block_only_c*` ITI scaffolding was the wrong target.

**Implementation (cached pipeline):**
- `CONTRASTS = [1.0, 0.25, 0.125, 0.0625, 0.0]`
- Bases: `GOAL3_DURINGSTIM_BASES` (8) + `GOAL3_DURINGCHOICE_BASES` (8) = 16 bases × 5
  contrasts → **80** registered splits `'{base}_{contrast}'` (e.g.
  `act_block_duringstim_l_choice_l_f1_0.125`).
- `contrast_from_split(name)` auto-parses trailing `_{float}` / `_c{float}` (regex
  anchored at EOS so `_choice` never false-matches).
- `get_d_vars`: contrast filter applied via `_filter_stim_side` whenever the name
  carries a contrast — **no** `bycontrast=True` flag required. Act vs non-act still
  toggled by `'act' in split`. Windows copied from the base split (`[0,0.15]`
  duringstim, `[0.15,0]` duringchoice).
- Cached path skips remote `pid2eid` when the insertion cache already has eid/probe.
- CLI presets in `scripts/run_goal2_splits.py`:
  `goal3_duringstim`, `goal3_duringchoice`, `goal3_duringstim_act`,
  `goal3_duringstim_block`, `goal3_duringchoice_act`, `goal3_duringchoice_block`,
  `goal3_all`; optional `--contrasts 0.0 0.125 1.0`; `--list-splits` to print names.
- **Sharding (same as Goal 2 / 07-12b):** contrast splits are ordinary split names, so
  `run_goal2_shard_slurm.sh` + `run_goal2_finalize_slurm.sh` work unchanged
  (`{split}.shard{k}.npy` tolerates dots in `0.125`). Submitter:
  `scripts/submit_goal3_sharded.sh` (default `goal3_duringstim_act`, N_SHARDS=4).
- Unsharded single-job smoke: `scripts/run_goal3_contrast_slurm.sh`.

```bash
# Full BWM on ORCD (recommended):
bash scripts/submit_goal3_sharded.sh
PRESET=goal3_duringstim_act CONTRASTS="0.0 0.125 1.0" N_SHARDS=4 \
  bash scripts/submit_goal3_sharded.sh

# Local / smoke (unsharded):
conda activate iblenv
python scripts/run_goal2_splits.py --preset goal3_duringstim_act --contrasts 0.0 0.125 1.0
```

**Smoke (alyx insertion cache, 3 pids × 6 splits, nrand=10):** 18/18 OK — act+non-act,
duringstim+duringchoice, contrasts 0/0.125/0.25/1.0; duringstim curves len=72.

### 2026-07-12c — Min 5 trials per split side

**Change:** `min_trials_per_side = 5` in `block_analysis_allsplits.py`. Both sides of a
split must have ≥5 trials; otherwise `get_d_vars` raises `InsufficientTrials` and the
cached driver logs `split skip` (no stream_acc / no per-insertion save). Replaces the
old assert that only rejected zero-trial sides.

**Smoke (alyx insertion cache, 2 pids × 5 contrast splits, nrand=5):** ok=2, skip=8,
fail=0. Examples: `…_f1_1.0` (13/7, 10/10) ran; `…_f1_0.125` (12/2), `…_f1_0.0` (4/1),
`…_f2_1.0` (0/0) skipped.

### 2026-07-12d — Lower Slurm mem for Goal 3 / Goal 2 shards

**Problem:** shard workers requested **48G** each; Goal 3 default submit = 20 splits × 4
shards = 80 jobs → **~3.8 TB** concurrent mem request → hit per-user mem limit → pend.

**Evidence (peak RSS, stream_pool, nrand=2000; journal 07-10b):** ~1.5–2.5 GB.
Goal 3 contrast + `min_trials_per_side=5` skips many insertions → smaller stream_acc.

**New defaults:**
| job | was | now |
|-----|-----|-----|
| shard (`run_goal2_shard_slurm.sh`) | 48G / 4 cpus | **12G** / 2 cpus |
| finalize | 32G | **16G** / 2 cpus |
| Goal 3 submit override | (inherit 48G) | **MEM_SHARD=8G**, **MEM_FIN=12G** |
| Goal 2 submit override | (inherit) | **MEM_SHARD=12G**, **MEM_FIN=16G** |
| unsharded `run_goal3_contrast_slurm.sh` | 48G | **16G** |

Concurrent Goal 3 default: 80 × 8G = **640G** (~6× less). Override:
`MEM_SHARD=6G MEM_FIN=10G bash scripts/submit_goal3_sharded.sh`.

### 2026-07-14 — Goal 3 results: cell retention + gain/offset tables

Downloaded finalized contrast splits: `alyx.../manifold/res/new/`.

**Cell retention (corrected).** An earlier draft summed L+R nclus (~125k) and
double-counted neurons across stim sides. Correct baseline is **per stim-side**
(~62.5k f1; ~55.6k f2), matching unsplit `act_block_duringstim_{l,r}` (~62.8k).

**Metric:** pooled `nclus` from finalized splits; **% kept** = mean(L,R) /
mean(all-contrast L,R). Region count = union of L∪R with ≥1 cell in that split.

#### f1 (correct; L = `*_l_choice_l_f1`, R = `*_r_choice_r_f1`)

| contrast | L cells | R cells | mean | % kept | nreg |
|----------|--------:|--------:|-----:|-------:|-----:|
| all | 62,520 | 62,575 | 62,548 | 100% | 208 |
| 1.0 | 53,136 | 55,601 | 54,368 | 86.9% | 206 |
| 0.25 | 56,013 | 53,996 | 55,004 | 87.9% | 203 |
| 0.125 | 46,269 | 45,364 | 45,816 | 73.3% | 199 |
| 0.0625 | 35,290 | 35,823 | 35,556 | 56.8% | 191 |
| 0.0 | 2,788 | 6,816 | 4,802 | **7.7%** | 92 |

#### f2 (incorrect; L = `*_l_choice_r_f2`, R = `*_r_choice_l_f2`)

| contrast | L cells | R cells | mean | % kept | nreg |
|----------|--------:|--------:|-----:|-------:|-----:|
| all | 56,045 | 55,216 | 55,630 | 100% | 206 |
| 1.0 | — | 51 | 51 | **0.1%** | 1 |
| 0.25 | 408 | 280 | 344 | 0.6% | 17 |
| 0.125 | 3,458 | 2,664 | 3,061 | 5.5% | 86 |
| 0.0625 | 19,244 | 15,906 | 17,575 | 31.6% | 174 |
| 0.0 | 11,151 | 9,197 | 10,174 | 18.3% | 139 |

f1 thins at low contrast (0% nearly empty under ≥5 trials/side). f2 is nearly empty
at high contrast (errors rare) and only modest at 0%.

**Gain/offset summary tables** (combine available of 4 `act_block_duringstim_*_{c}`
splits → `p_mean`/`p_gain`/`p_offset` → BH-FDR → `plot_table_with_styles` gain/offset;
α=0.01):

`alyx.../meta/table_act_block_combined_summary_act_p_mean_c_combinedpTrue_0.01_gain_offset_{c1,c025,c0125,c00625,c0}.png`

Script: `scripts/plot_goal3_c0_summary_table.py` (all contrasts; `--retention-only`).

**Note:** at α=0.01 FDR, **0 regions** pass `p_mean_c` for any contrast-conditioned
combined table. c=1 uses 3/4 splits (`l_choice_r_f2_1.0` missing).

### 2026-07-14b — Pipeline sanity: all-contrast recoverability

Same combine → `p_mean`/`p_gain`/`p_offset` → BH-FDR (α=0.01) → gain/offset table,
on the **unconditioned** four `act_block_duringstim_*` splits (no contrast suffix).

| source | path | FDR `p_mean_c`≤0.01 | gain∩sig |
|--------|------|--------------------:|---------:|
| openalyx copies (isolated) | `res/new/openalyx_allcontrast_ref/` | **37**/208 | 20 |
| alyx `res/new` all-contrast | `res/new/act_block_duringstim_*` | **42**/208 | 26 |

Plots (alyx meta only; do not overwrite openalyx):
`…_gain_offset_openalyx_ref.png`, `…_gain_offset_alyx_new_all.png`.

**alyx vs openalyx files are not bitwise identical** (null shuffle seeds + ~15/207
regions with small `nclus` diffs, e.g. CP 2759 vs 2655), but true curves match
(VISp corr=1.0) and both recover tens of FDR hits — so the combine/FDR path is fine.
Per-contrast zeros are not a plotting bug.

### 2026-07-14c — Why per-contrast looks dead at α=0.01 even at c=1 (~87% cells)

**Not mainly cell loss.** f1 at c=1 keeps ~87% cells / 206 regions. Two compounding
issues:

**(1) Discrete p-floor vs BH-FDR at α=0.01 (dominant).**
With `nrand=2000`, min attainable p ≈ **1/2001 ≈ 0.0005**. For BH at α=0.01 with
m≈206 tests you need **≥11 regions pinned at that floor** before *any* rejection is
possible (`k ≥ ceil(0.0005·m/α)`). Count at floor / FDR survivors:

| set | n at p-floor | `p<0.01` | FDR@0.01 | FDR@0.05 |
|-----|-------------:|---------:|---------:|---------:|
| all-contrast | 25 | 54 | **42** | 56 |
| c=1.0 | **10** | 36 | **0** | **32** |
| c=0.25 | 3 | 24 | 0 | 9 |

So c=1 is one region short of clearing the α=0.01 floor barrier; at **α=0.05 FDR
there are 32 hits**. The earlier “nothing significant at highest contrast” was an
α=0.01 + nrand interaction, not absence of signal.

**(2) Smaller per-contrast effect / SNR (real biology + fewer trials).**
Pooling all contrasts accumulates prior-distance across conditions. Restricting to
c=1 uses fewer trials per cell → smaller `amp_euc` / effect, fewer regions pushed
to the p-floor:

| set | median amp | median effect (true−null mean) | median SNR |
|-----|-----------:|-------------------------------:|-----------:|
| all-contrast | 1.75 | 0.114 | 1.23 |
| c=1.0 | 0.85 | 0.056 | 1.12 |
| c=0.25 | 0.98 | 0.043 | 0.82 |

Of 42 all-contrast FDR@0.01 regions, **18** still have uncorrected `p≤0.01` at c=1
(27 at `p≤0.05`), but only 10 hit the floor — not enough for BH@0.01.

**Also:** f2 is empty at high contrast, so “4-split” c=1 is effectively **f1-only**
(plus a tiny f2 R file). Cell % is high on f1, but you lose incorrect-trial splits
and all lower-contrast trials that carried prior signal in the pooled analysis.

**Takeaways:** (a) re-plot / re-threshold per-contrast at **FDR α=0.05**, or raise
`nrand` (e.g. 10k) if insisting on α=0.01; (b) high-contrast prior distance is
weaker than pooled — consistent with prior mattering more when stim is ambiguous.

### 2026-07-14d — Per-contrast gain/offset tables at FDR α=0.05

```bash
python scripts/plot_goal3_c0_summary_table.py --alpha 0.05 --skip-retention
```

| contrast | nreg | FDR `p_mean_c`≤0.05 | gain∩sig | offset∩sig | notes |
|----------|-----:|--------------------:|---------:|-----------:|-------|
| 1.0 | 206 | **32** | 23 | 11 | 3/4 splits |
| 0.25 | 203 | **9** | 6 | 4 | |
| 0.125 | 200 | **3** | 1 | 2 | |
| 0.0625 | 198 | **4** | 2 | 2 | |
| 0.0 | 151 | **0** | 0 | 0 | |

Plots: `alyx.../meta/table_act_block_combined_summary_act_p_mean_c_combinedpTrue_0.05_gain_offset_{c1,c025,c0125,c00625,c0}.png`

Signal recovers strongly at c=1 under α=0.05; mid/low contrasts weak; 0% still null.

### 2026-07-17 — Goal 3 scope update: choice-conditioned 0% contrast

The previous Goal-3 contrast sweep conditioned on nominal stimulus side and then
split correct/incorrect trials (`f1`/`f2`). This is too restrictive for the revised
question and leaves very few 0%-contrast trials.

**New primary analysis:**

1. Keep only 0%-contrast trials.
2. Do **not** condition on nominal stimulus side.
3. Do **not** subdivide by `f1`/`f2`; condition only on choice L or choice R.
4. Within each fixed choice side, measure the population difference between block
   L and block R.
5. Produce both (a) individual-region results and (b) one result aggregated across
   all regions.

This supersedes the old four-way `stim side × choice/feedback` split for the main
Goal-3 result. The 07-14 contrast-conditioned tables remain useful diagnostics and
historical results.

**Implementation (07-17):**

- New true-block splits: `block_duringstim_choice_l_0.0` and
  `block_duringstim_choice_r_0.0`.
- Trial mask is `(contrastLeft == 0 OR contrastRight == 0) AND fixed choice`;
  neither nominal stimulus side nor `feedbackType` is used.
- Existing finalized `{split}.npy` / `{split}_regde.npy` files provide regional
  results. Finalization now also writes `{split}_all.npy` and
  `{split}_all_regde.npy`, pooling raw squared distances over all valid neurons
  before normalization (not averaging normalized regional curves).
- `scripts/run_goal2_splits.py --preset goal3_c0_choice` and the Goal-3 Slurm
  scripts run the two revised splits by default.
- `scripts/plot_goal3_c0_summary_table.py` now defaults to revised Goal 3 and
  writes one BH-FDR regional CSV per choice plus
  `goal3_c0_choice_all_regions.csv`. Historical analysis remains available via
  `--legacy-contrast` and the prior explicit options.

**Validation:** split registration/window, synthetic zero-contrast masks (both
nominal sides and feedback outcomes), raw all-region pooling, summary CSV output,
Python compilation, and shell syntax all pass. Full BWM run remains to be submitted.
