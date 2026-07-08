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

Redo the real-data prior-modulation analysis **separately for low- vs high-contrast trials**, and specifically at **0% contrast** (behavior fully prior-driven). Then **compute the slope of the contrast-response function** and test whether the prior modulates that slope.

Scaffolding already present: `bycontrast`, `lowcontrast`, and a `stim_{contrast}` split in `get_d_vars`. Build the contrast-conditioned prior test + slope test on top of the Goal 2 cache.

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

## Next steps

1. Goals 2 & 3 validation in working env (BWM `SessionLoader` compat).
