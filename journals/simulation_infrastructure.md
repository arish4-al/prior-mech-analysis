# Simulation infrastructure: session cache, unified experiment runner, analysis matrix

**Scope:** the "simulate once, analyse many ways" layer in `simulate_recovery.py` — why it was needed, how the cache is keyed, the unified `--run-experiment` entry point, and the 4 × 5 analysis matrix built on top of it.

**Status:** done and validated. All four canonical experiments share one simulation per parameter set; every analysis variant (split/unsplit, contrast-matched/label-shuffle, full classification) hits the same cache key.

Sources: dated entries 2026-07-06 (Goal 1), 2026-07-06a, 2026-07-06b. Speedup follow-on: [simulation and model-fitting speedups](simulation_fit_speedups.md).

---

## Goal

Run **three new analysis variants** on **four saved experiments**, reusing simulations wherever possible.

**The four experiments** (differ only in model params `mp`; seed 123, 40 sessions, nrand=100):

| # | Experiment | g_s | d_s | g_i / d_i / g_m / d_m | CLI | Saved res? |
|---|-----------|-----|-----|-----------------------|-----|-----------|
| E1 | No prior mod on S/I/M (Phase 4b) | 0 | 0 | 0 | `--phase4-no-prior-mod` | `absence/figs/phase4_no_prior_mod/res` |
| E2 | Canonical absence (no P→S; fitted P→I/M) | 0 | 0 | fitted | default `absence` | `absence/res`, `absence_fill_next/seed_123/absence/res` |
| E3 | P→S only, I-first threshold | 1800 | 0 | 0 | `s_presence_tune` g_s=1800 | `s_presence_tune/g_s1800_d_s0/res` |
| E4 | P→S only, S-first threshold | 2025 | 0 | 0 | `s_presence_tune` g_s=2025 | `s_presence_tune/g_s2025_d_s0/res` |

E3/E4 use **inside-adaptation** gain placement — the pair with complete saved `res/`. Outside-adaptation analogues (`g_s700_gs_free` / `g_s900`) also exist.

**The three analyses:**

- **1.1 — No contrast matching.** Re-run each experiment with the label-shuffle null (`--label-shuffle-null`) instead of the contrast-matched null.
- **1.2 — No f1/f2 splits (stim-side unsplit).** Re-run each with stim_l+stim_r unsplit, preserving stim side.
- **1.3 — Full analysis.** BWM functional classification (S/I/M, `--full-analysis`) for all four experiments.

---

## The blocking architecture finding

Every analysis entry point calls `simulate_condition_sessions(mp, …, rng_seed)` and then analyses the resulting `session_dfs` (per-trial full time series). **`session_dfs` was never saved to disk** — only the *post-split, post-average* arrays were cached under each experiment's `res/` (e.g. `act_block_duringstim_l_choice_l_f1.npy`).

Consequences:

- `res/` **cannot** be re-split differently (1.2) or re-nulled (1.1) — both need the raw `session_dfs`.
- `--recovery-only` only re-runs analysis on existing split arrays, so it does not help for 1.1/1.2.
- Simulation is deterministic in `(mp, rng_seed, n_sessions, blocks_per_session, max_obs_per_trial)`.

**Decision:** add a `session_dfs` cache keyed by that tuple. Simulate each of E1–E4 once, persist `session_dfs`, then run all analysis variants off the cache. (This is the generative-model twin of the real-data insertion cache — see [real-data pipeline efficiency](realdata_pipeline_efficiency.md).)

---

## Session cache (2026-07-06a)

- **`simulate_condition_sessions(...)` is cache-aware.** Deterministic draws persist to `<manifold_sim>/session_cache/{key}.pkl.gz` (+ `{key}.json` sidecar), keyed by a sha1 of `(mp, seed, n_sessions, blocks_per_session, max_obs_per_trial, min_trials_per_session, constant_s0, dt)` and `SESSION_CACHE_VERSION`. Raw simulation moved to `_simulate_condition_sessions_raw`.
- **`process_condition` routes through the cache** (its inlined per-session loop was removed), so absence / presence / s-presence share the same cache as phase4 / unsplit.
- **Replicate-null loop uses `use_cache=False`** (high seed cardinality → avoid bloat).
- **CLI `--no-session-cache`** disables it (default: enabled).

**Validation (iblenv, outside sandbox):**

- Phase4 seed 999, 2 sessions: run 1 → `[session cache MISS] … saved`; run 2 → `[session cache HIT]`. ✓
- New `--unsplit-prior s_presence --g-s-presence 1800 --d-s-presence 0` runs end to end, writes `s_presence_g_s1800_d_s0_unsplit/`, and reuses the same cache key as a future full `s_presence` run at matching `(mp, seed, n_sessions)`. ✓

**Cost (6-block default):** ≈ 15 MB (gzip) per 2-session draw → ≈ **300 MB per 40-session experiment**, ≈ 1.2 GB for E1–E4. Acceptable for the default laptop path.

### 2026-08-14 — long-session Harris cache blew the laptop; wipe + ORCD

`--harris-unique-null` with `--blocks-per-session 40`, `--nrand 2000`, and
`--harris-n-extra-donors 80` wrote **~12 GB** of new gzipped traces (regular
observed 1.8 GB + extra-80 3.7 GB; sensory observed 2.1 GB + extra-80 4.4 GB).
The whole `session_cache/` hit **42 GB**. Analysis outputs under
`manifold_sim/stageB_bwm/harris_bps40/` were only **~31 MB**. Python RSS stayed
~167 MB — disk is the cost, not RAM.

After that campaign finished, the **entire**
`<ONE cache>/manifold_sim/session_cache/` was deleted (2026-08-14). Curve
results were kept.

**Policy:** do not refill that cache on the laptop. Harris unique-null with
long sessions / `nrand=2000` / extra donors goes on **ORCD** (`mit_normal`).
The default 6-block / `nrand=100` / contrast-matched path can still run locally
(Phase 4b check). See [retinal then joint](retinal_then_joint_fitting.md) 2026-08-14.

> Note: the phase4 output path (`absence/figs/phase4_no_prior_mod/`) has no seed component, so a 2-session smoke run overwrote the old E1 baseline directory. E1 was regenerated cleanly afterwards.

### `--unsplit-prior` extended (enables 1.2 for E3/E4)

Added an `s_presence` case to `run_unsplit_prior_distance_analysis` + CLI. It builds `mp` via `load_fitted_model(g_s, d_s, zero_im_prior_mod=True, gs_outside_adaptation=…)` and tags output `s_presence_g_s{…}_d_s{…}[_gs_free]_unsplit`. This is what lets stim-side unsplit run on the gain-only threshold experiments, which the old code could not do.

---

## Wiring gaps found before the full runs

The three analyses did not map cleanly onto existing subcommands for all four experiments:

| Analysis | E1 phase4 | E2 absence | E3/E4 s_presence |
|----------|-----------|-----------|------------------|
| baseline contrast-matched | `--phase4-no-prior-mod` ✓ | default path (runs presence too) ⚠ | `s_presence_tune` res exists ✓ |
| **1.1** label-shuffle null | `--phase4-no-prior-mod --label-shuffle-null` ✓ | needs an **absence-only** entry ⚠ | `--s-presence-tuned-plots … --label-shuffle-null`? (verify) |
| **1.2** stim-side unsplit | `--unsplit-prior phase4` ✓ | `--unsplit-prior absence` ✓ | `--unsplit-prior s_presence …` ✓ (new) |
| **1.3** full classification | **no path** ✗ | `--full-analysis` default (also presence) ⚠ | **no path** ✗ |

- `--full-analysis` was only wired into the default absence/presence path (`process_condition` with `s_prior_only=False` → `classify_regions`). Phase4 and s_presence had no classification path.
- E2 canonical absence only ran bundled with presence + replicate null.

---

## Unified experiment runner (2026-07-06b)

Resolved the gaps with a single cache-backed entry point rather than per-experiment special cases:

**`run_experiment_case` + `--run-experiment {phase4,absence,s_presence}`**

- **sprior mode** (default): S/I/M split-conditioned (f1/f2) prior distance via the shared `_run_split_population_prior_distance` helper (extracted from `run_phase4`), under contrast-matched (baseline) or `--label-shuffle-null`.
- **full mode** (`--full-analysis`): BWM Σ classification via `classify_regions`, reusing `process_condition` (rebuilds an identical `mp` → session-cache HIT).
- Outputs: `goal1/<exp_tag>/<null>_<mode>/` where `null ∈ {cm, ls}` and `mode ∈ {sprior, full}`.
- `process_condition` generalized with `zero_all_prior_mod` + `gs_outside_adaptation`.

All four experiments now share one simulation per `mp` (unsplit + sprior + full all hit the same cache key).

**Matrix:** `run_goal1_matrix.sh` (background, seed 123, 40 sessions, nrand 100, n-jobs 8). 20 runs = 4 experiments × {cm_sprior (baseline), ls_sprior, cm_unsplit + ls_unsplit, cm_full}. Status log: `manifold_sim/goal1/_logs/matrix_status.log`.

**Results** from the matrix live in [split conditioning vs unsplit](split_conditioning_vs_unsplit.md) (Tables A–C) and [BWM classification recovery](bwm_classification_recovery.md).

**Known issue in the first matrix:** `ls_unsplit` overwrote `cm_unsplit` (shared output dir, no null tag). The four CM unsplit runs were repeated and preserved as `*_CM_summary.json`. Future matrices should write null-specific unsplit directories.

---

## Related — faster fitting (not analysis)

The session cache above speeds up *re*-analysis. Making **fitting** reach the baseline loss in ≲1–2 h is separate — see [faster model fitting](simulation_fit_speedups.md).
