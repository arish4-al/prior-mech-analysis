# Canonical analysis conventions

**Scope:** the mandatory defaults for every simulation / prior-distance experiment in this repo, plus the environment rules for running them. Code is the source of truth (`simulate_recovery.py` module docstring, `CANONICAL_PRIOR_DISTANCE_ANALYSIS`, `build_population_b_for_split`); this file records *why* each convention exists and when it changed.

**Status:** stable since 2026-06-19. Any result produced before that date with zero-padding or a 150 ms S window is invalid for significance claims.

---

## Goal

Fix one analysis configuration so that prior-distance numbers are comparable across experiments and so that known artefacts (see [S prior artefacts](s_prior_artifacts_truncation.md)) cannot re-enter through a side door.

---

## The defaults

| Setting | Value | Notes |
|---------|-------|-------|
| **S analysis window** | **80 ms** post-stim | `S_DURINGSTIM_WINDOW_S = 0.08`, applied in `build_population_b_for_split` when `population == "S"`; 80 ms → 32/36 bins depending on binning |
| **I/M analysis window** | **150 ms** post-stim | `PRE_POST` duringstim splits |
| **Truncated trials** | **fill-from-next-ITI** | Never zero-pad. Borrow the leading `need` steps of the next trial's ITI; skip the trial if the next trial is in a different session or its ITI is too short |
| **Null** | **contrast-matched label shuffle** | Default CLI; `--label-shuffle-null` for the unrestricted alternative |
| **Output root** | `<ONE cache>/manifold_sim` | via `resolve_output_dir()` / `default_output_dir()`. Do not write to repo `output/` unless `--allow-repo-output` |
| **Environment** | `conda activate iblenv` | Run **outside** the Cursor sandbox on macOS |

`--duringstim-window-ms` overrides the post-stim analysis window for batch experiments.

On this machine the ONE cache resolves to `/Users/ariliu/Downloads/ONE/openalyx.internationalbrainlab.org`, so sims live under `.../manifold_sim/`. The alyx cache (`/Users/ariliu/Downloads/ONE/alyx.internationalbrainlab.org`) is used for the real-data pipeline.

Trajectory plots must use the same 80 ms S cap as the distance analysis (`trial_s_binned_signed`), otherwise the plot path reintroduces the zero-padding artefact the analysis path fixed.

---

## Phase 4b sanity check

Split-conditioned Phase 4b (all `g_*=d_*=0`) at seed 123 must reproduce:

```bash
conda activate iblenv
python simulate_recovery.py --phase4-no-prior-mod \
  --seed 123 --n-sessions 40 --nrand 100 --n-jobs 8
```

**Expected:** S `curve_mean` ≈ **0.012**, `p_mean` ≈ **0.78** (not significant); I and M also null.

If S comes out significant with a large `curve_mean`, the analysis path is wrong — most likely zero-padding, a 150 ms S window, or pooling L+R stim trials without stim-side splits.

---

## Unsplit experiments

- `--unsplit-prior` = no f1/f2 choice×feedback splits, but **still stim_l + stim_r** (see `UNSPLIT_PRIOR_SPLITS`). Do not use a single pool of all trials for the S distance.
- `--unsplit-mode fully` = diagnostic only (L+R mixed). It reproduces a spurious S signal even under Phase 4b and must never be used for inference. See [split conditioning vs unsplit](split_conditioning_vs_unsplit.md).

---

## Common pitfalls

1. **Do not pool left- and right-stim trials in one S distance** without stim-side splits — activity lives on different channels and this creates spurious S signal even with `g=d=0`.
2. **Unsplit** means no f1/f2 conditioning; it still uses stim_l + stim_r stacked.
3. **Old results** using zero-padding or a 150 ms S window are invalid for significance claims.
4. `summary.json` **does not record the null scheme** — infer it from CLI defaults or re-run with `--label-shuffle-null` to compare.
5. When re-running a variant into a shared output directory, tag the output by null scheme. In the original Goal-1 matrix `ls_unsplit` silently overwrote `cm_unsplit` (see [simulation infrastructure](simulation_infrastructure.md)).

---

## Sandbox warning (ONE)

Anything that touches ONE caches (`~/Downloads/ONE/...`), large `manifold/res/*.npy` combines/plots, or `iblenv` analysis scripts **hangs or sits at 0 % CPU inside the Cursor sandbox** — seen repeatedly on combine/plot/null jobs. Always:

1. Request `required_permissions: ["all"]` (disable the sandbox), and
2. `conda activate iblenv` before `python …`.

Do not retry the same ONE command under the default sandbox hoping it will finish.

**Harris unique-null / long sessions (since 2026-08-14):** `--harris-unique-null`
with `--blocks-per-session` ≫ 6, `nrand=2000`, or large `--harris-n-extra-donors`
belongs on **ORCD**, not the laptop. Those draws refill `session_cache/`
(40-block × 80-donor ≈ 12 GB; whole cache hit 42 GB and was wiped). Laptop is
for the default 6-block / `nrand=100` / contrast-matched path. See
[simulation infrastructure](simulation_infrastructure.md).

---

## Where these conventions live

- **Code (source of truth):** `simulate_recovery.py` — module docstring, `CANONICAL_PRIOR_DISTANCE_ANALYSIS`, `build_population_b_for_split`
- **Cursor rule:** `.cursor/rules/prior-distance-analysis.mdc` (auto-loaded for agents)
- **Agent guide:** `AGENTS.md`
- **Experiment history:** these topic journals

---

## Timeline

| Date | Change |
|------|--------|
| 2026-06-18 | Zero-padding replaced by fill-from-next-ITI as the default in `build_population_b_for_split`; `--require-full-window` added as a strict-exclusion diagnostic; `--duringstim-window-ms` CLI added |
| 2026-06-19 | S window set to 80 ms (`S_DURINGSTIM_WINDOW_S`); I/M stay at 150 ms |
| 2026-06-19 | Trajectory plot path (`trial_s_binned_signed`, `plot_p_block_s_trajectories`) given the same 80 ms cap and a population-specific time axis |
| 2026-06-19b | Final retest at S=80 ms / I/M=150 ms declared canonical going forward |
| 2026-06-20 | Canonical output root `<ONE cache>/manifold_sim/` enforced via `resolve_output_dir()` |
| 2026-07-06 | Session cache added; conventions unchanged (see [simulation infrastructure](simulation_infrastructure.md)) |
| 2026-07-12c | Real-data pipeline: `min_trials_per_side = 5` (both sides of a split need ≥5 trials) |
| 2026-08-14 | Harris unique-null long-session / nrand=2000 / extra-donor runs → ORCD; laptop `session_cache/` wiped |

Sources: dated entries 2026-06-18, 2026-06-19, 2026-06-19b, 2026-06-20, 2026-06-29, 2026-07-06.
