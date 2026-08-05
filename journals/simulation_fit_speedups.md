# Faster model fitting (simulation + optimizer)

**Scope:** make the **fitting process** (`fit_retinal.py` → `fit_weights.py`) reach a solution with errors similar to the current fitted baseline in **about 1–2 hours**. Two levers only:

1. **Per-session simulation** inside each loss eval (`create_stimuli` + `run_model`) is too slow.
2. **The fitting process** (optimizer schedule, eval budget, noise, parallelism, search vs refine) is too slow or ineffective.

**Out of scope:** prior-distance / recovery analysis, session-`session_dfs` cache, real-data insertion cache. Those speed up *analysis reuse*, not fitting.

**Status:** Cluster-ready after 2026-08-05a pre-submit fixes. Backend **loky** + numba; ONE bypassed on fit path. Defaults: **`stage2_n_stim_seeds=3`** with **`stage2_stim_aggregate='sample'`** (1-of-3 per eval ≈1× wall, not mean≈3×); **`--val-seed`** held-out select + early-stop + polish gate; rolling ckpt stores **held-out incumbent**; resume = **latest mtime**; polish = **Powell/`prior`** with **`LOCAL_REFINE_MAX_WALL_S=1800`**. Slurm default **6 h**. Full snapshot in **"Current fitting pipeline (reference)"**.

Sources: 2026-07-20 Goal 2 (reframed); baselines 2026-08-03b–c; parity 2026-08-03d; profile 2026-08-03e; short-opt 2026-08-03f; early-stop 2026-08-03g–i; effectiveness 2026-08-03j–k; pipeline review + cluster 2026-08-03l–m; pre-submit 2026-08-05a.

**Agent constraints** (also `.cursor/rules/fit-speedups.mdc` + `AGENTS.md`):

- **Do not** shorten trials or cut `blocks_per_session` to speed search.
- **Do not** reuse one stimulus batch across an optimizer generation for speed.
- Skipping **unused sim outputs** in the fit path is OK if the loss is unchanged.

---

## Goal

**Primary target (matches how the current best was produced):** with **retinal frozen**, get `fit_weights` to a traj+prior loss **near 0.40** in **≲ 1–2 hours**.

**Secondary:** characterize / speed up `fit_retinal` so a from-scratch front-end fit is also affordable if we unfreeze later.

Success = match baseline **loss scale**, not Phase 4b analysis.

---

## Quality baselines

### Weights — retinal frozen (canonical)

| Field | Value |
|-------|--------|
| File | `models/weights_run_20251125_182058/weights_2stagelocalrefine_loss0p4044_20251125-195255.json` (`WEIGHTS_REL`) |
| Recorded loss | **0.4044** |
| Retinal | **Frozen** at values embedded in JSON `model_params` (see below) — this run was weights-only / Stage-2 CMA refine |
| Free params | network W / g / d / θ (12-d log vector) |
| Key fitted | `g_i≈189.7`, `d_i≈21.56`, `g_m≈d_m≈0`, `θ_c≈0.76`, `θ_d≈0.40` |
| Historical wall | `fit_log.jsonl` 2025-11-25 **18:21 → 19:53** (≈ **1.5 h**) Stage-2 CMA only (`n_jobs=16`, 40 gens); Stage-1 loss at resume **0.805**. Best **0.404** at gen 12; gens 13–39 no improvement |

Frozen retinal in that JSON (also hardcoded at top of `fit_weights.py`):

`α_w=1.565`, `β_w=0.164`, `α_d=35.277`, `β_d=2.0515`, `τ_a=222.68`, `W_as=28.106`, `W_ss=7.652e-5`

### Retinal front-end

| Field | Value |
|-------|--------|
| Data | `avg_mean_R.npy` (resolved from `paper-brain-wide-map/avg_mean_R.npy` for the bench) |
| Loss | `fit_mode='rms'` → `compute_sse_stim_right` on `mean_S_by_contrast` |
| Script default | `blocks_per_session=10` |
| Frozen front-end loss (2026-08-03c) | **~0.78–0.82** per random eval at `bps=10` |
| Old DE2 ckpt | `fit_run_20251013_212918` DE2 `best_loss=2.076` in log (smoke: `de*_maxiter=2`); re-eval today **~0.84–1.02** — worse than frozen on current data. LOCAL stage aborted (`ABNORMAL_TERMINATION_IN_LNSRCH`). **Not** the production quality baseline; frozen JSON params are. |
| Historical smoke wall | DE1→DE2 on that run ≈ **46 min** for tiny DE budget; not a production retinal fit clock |

---

## Baseline timings

Machine: local macOS, `iblenv`, `HAVE_NUMBA=True`.

Bench scripts avoid importing `fit_weights.py` / `fit_retinal.py` (both have **unguarded** bottoms that start fits on import):

- Weights: `scripts/bench_fit_baseline_sim.py`
- Retinal: `scripts/bench_fit_retinal_baseline.py`
- Parity: `_validate_numba_backend.py` + fitted-params check (2026-08-03d)

### A — Weights path (retinal frozen) — 2026-08-03b

Fit-like: `dt=2`, `steps_before_obs=500`, `max_obs_per_trial=1000`, Stage-1 `bps=5`, Stage-2 `bps=20`. Params from `load_fitted_model(g_s=0, d_s=0)`.

| Setting | Mean wall / session or eval | Notes |
|---------|------------------------------|--------|
| Sim `bps=5` | **~0.43 s** | |
| Sim `bps=20` | **~1.84 s** | |
| numpy vs numba (`bps=5`) | **~15.6 s vs ~0.59 s** | **~26×** |
| Full weights loss `bps=5` | **~0.38–0.49 s** | ≈ sim; non-sim ~tens of ms |
| Full weights loss `bps=20` | **~2.1–2.2 s** | ≈ sim (~1.9–2.1 s); stim~0.1 + avg/prior~0.06 |

**Can the “weights loss” row get much faster without faster sim?** No. Section breakdown: traj + prior + `mean_by_condition` together are **≲ ~0.07 s** at `bps=20` vs **~2 s** of `run_model`. Speeding loss math alone saves at most a few percent per eval. The lever is still the simulator (and later the optimizer’s eval count / parallel efficiency).

### B — Retinal path — 2026-08-03c

`bps=10`, prior gains zeroed (as in `fit_retinal.py`), network W at retinal-script defaults.

| Config | Loss (3 random seeds) | Wall / eval | sim share |
|--------|----------------------|-------------|-----------|
| **Frozen JSON retinal** | 0.815 / 0.781 / 0.796 | **~0.92–1.44 s** | ~1.1 s mean sim |
| DE2 smoke ckpt | 0.838 / 0.853 / 1.018 | **~0.86–1.29 s** | same order |
| numpy vs numba (`bps=10`, frozen) | — | **~36.1 s vs ~1.34 s** | **~27×** |

SSE / `mean_S_by_contrast` are negligible vs sim (few ms).

**Implication:** both stages are sim-bound under numba. Retinal evals at `bps=10` cost ~1–1.4 s — comparable to weights Stage-2 `bps=20` (~2 s). A serious retinal DE (large pop × many iters × parallel workers) can dominate wall clock unless budgeted carefully; the **0.404** solution did **not** re-fit retinal.

### C — Numba ↔ numpy parity — 2026-08-03d

| Check | Result |
|-------|--------|
| `_validate_numba_backend.py` (default mp, `g_s=d_s=0`, seeds 0–2) | **PASS** — continuous fields max ‖Δ‖ = **0**; choices / RT / trial lengths EQUAL; speedup **~25.6×** |
| Fitted `WEIGHTS_REL` params (`g_i≈189.7`, `d_i≈21.56`), `bps=5`, seeds 0,1,2,42 | continuous max ‖Δ‖ **~1e-16** (fp noise); choices / RT **EQUAL** |
| Same fitted, `bps=20`, seed 7 | max ‖Δ‖ S=0, I/P/M ~1e-16 |

Numba is safe to hard-require on the fit path.

### D — Fit-path section profile — 2026-08-03e

Fitted weights, `backend='numba'`, `scripts/profile_fit_sim_sections.py` (mean of 4 seeds):

| bps | stim | flatten | **kernel** | reassembly (full) | reassembly (fit-minimal) | `run_model` total |
|-----|------|---------|------------|-------------------|--------------------------|-------------------|
| 5 | 0.026 s | 0.001 s | **0.46 s** | 0.007 s | 0.002 s | 0.48 s |
| 20 | 0.114 s | 0.007 s | **2.03 s** | 0.029 s | 0.007 s | 2.03 s |

- Kernel ≈ **97%** of `run_model`. Flatten negligible.
- Buffer fill `ntot/(Ntr·L) ≈ 0.44` (trials end before max length; still allocate full `L`).
- Fit-minimal reassembly (drop `a` / perceived / action_signal / extra tiles) saves **~0.005 s / ~0.022 s** — not actionable vs kernel.
- Code: `fit_weights.py` / `fit_retinal.py` now pass `backend='numba'` (raises if numba missing or unsupported).

---

## Plan

### Phase 0 — Baselines

- [x] Weights quality target **0.404** (retinal frozen) + sim/loss section timings.
- [x] Retinal frozen / DE2 ckpt loss + section timings at `bps=10`.

### Phase 1 — Faster per-eval simulation

Constraints: **no** shorter trials, **no** fewer blocks, **no** cross-generation stim reuse. Optional: skip unused outputs.

1. [x] Confirm numba ≡ numpy (2026-08-03d).
2. [x] Hard-require numba in `fit_weights.loss_weights_core_v2` and `fit_retinal.loss_retinal_weight` (`backend='numba'`).
3. [x] Profile stim / flatten / kernel / reassembly (`scripts/profile_fit_sim_sections.py`, 2026-08-03e).
4. ~~Skip unused outputs~~ — **deferred**: saves only ~0.02 s at `bps=20` (~1%); not worth API risk now.
5. Optional later: deepen `_run_model_kernel` (fill is ~44% of `Ntr×L` buffers — allocation / early-exit packing). Only if per-eval sim must drop further after Phase 2.

### Phase 2 — Faster / more effective fitting

**2a — `fit_weights.py` (primary for 1–2 h → ~0.40 with retinal frozen)**

1. [x] Guard module bottom with `if __name__ == '__main__':`.
2. [x] Short-opt baseline from perturbed fitted θ (`scripts/bench_fit_weights_shortopt.py`, 2026-08-03f).
3. [x] Early-stop after plateau, **armed only after beating known best** (`cma_early_stop_beat_loss=0.4044`, patience=8; 2026-08-03i).
4. [x] Parallel / schedule edits (2026-08-03l–m): variant bench (`scripts/bench_fit_pipeline_variants.py`); **CMA backend = loky** (5x faster than threading); Stage-2 DE now parallel via threading map; `L_threshold=None` auto path fixed.
5. Keep production `bps` / trial length; improve schedule and stopping, not task size.

**2d — Cluster submission (ORCD `mit_normal`)**

1. [x] CLI driver `scripts/run_fit_weights.py` (`--seed/--n-jobs/--pipeline/--out-tag/--resume-json`).
2. [x] Worker `scripts/run_fit_weights_slurm.sh` (16 cpu, 40G, 4h, `OMP=1`, `miniforge`/`~/conda_envs/ibl`).
3. [x] Seed-sweep / multi-start submitter `scripts/submit_fit_weights_sharded.sh` (`sbatch --parsable`, env `--export`).
4. [x] Model variants x seeds + deterministic restartable run dirs (rolling Stage-2 checkpoint; `--resume auto`; `FIT_DONE`); 2026-08-04a.

**2c — Fit effectiveness (paper hand-tweaks vs checkpoint)**

1. [x] Paper vs ckpt compare + `g_i` loss slice (`scripts/bench_fit_gi_slice.py`, 2026-08-03j).
2. [x] Post-CMA local refine on `{g_i,d_i,θ}` (`local_refine_after_cma=True`).
3. [x] Multi-seed Stage-2 loss API (`stage2_n_stim_seeds`); `g_m`/`d_m` bounds to ~1e-12.
4. [x] Cheaper local refine (2026-08-03k): progress logs; Powell off unless no improve; **plateau patience** (not looser tols); parallel FD jac via `n_jobs`.

**2b — `fit_retinal.py`**

1. Guard module bottom.
2. Treat frozen JSON front-end as quality reference (~0.80 rms on current `avg_mean_R`); old smoke DE logs are not.
3. Budget a real retinal DE (pop/iters/`n_jobs`) so a refit is minutes-to-~hour, not open-ended; or keep retinal frozen for weights refits unless S-fit residuals demand it.

### Phase 3 — Acceptance

1. **Weights, retinal frozen:** timed run ≤ 2 h, loss ≲ ~0.45 (stretch ≤ 0.41).
2. **Optional retinal:** timed DE reaching ≲ frozen loss (~0.80) under documented budget.
3. Update this note.

---

## Current fitting pipeline (reference)

Authoritative snapshot of how `fit_weights` runs today (retinal frozen). Driver
`scripts/run_fit_weights.py` → `fit_weights_two_stage_v2`; cluster via
`run_fit_weights_slurm.sh` + `submit_fit_weights_sharded.sh` on ORCD `mit_normal`
(16 cpu, 40G, **6h**). Parallelism = `joblib`, one process, `n_jobs=SLURM_CPUS_PER_TASK`.
No ONE / network in the fit path (`PRIOR_MECH_NO_ONE=1`; cache via `ONE_CACHE_DIR`).

**Parameter vector (12-d, log space):**
`[0 W_ii, 1 W_pp, 2 W_mm, 3 W_is, 4 W_pi, 5 W_mi, 6 g_i, 7 g_m, 8 d_i, 9 d_m, 10 θ_c, 11 θ_d]`.
A **variant** = (`--mtype` label, `--freeze` indices). Frozen dims are pinned at ~0
(`LOG_ZERO`) and dropped from the active dims/bounds. `train_mask` marks active dims;
`idx = np.where(train_mask)[0]`.

| Stage | Method | bps | Key budget | Parallel | Stopping | Goal |
|-------|--------|-----|-----------|----------|----------|------|
| **1 — global** | Differential Evolution (scipy) | 5 (cheap, noisy) | `de1-maxiter=40`, `de-popsize=8`, `sobol-count=8`; borderline `[L_th, L_th+0.4)` **extends DE** by `max(10, de1//2)` | Stage-1 module-level `_loss_active_de_worker` (picklable; fork/ORCD) | DE convergence / maxiter; if still ≥ `L_threshold` → **FIT_FAILED** (no `FIT_DONE`) | broad global coverage from random/Sobol init |
| **2 — refine** | CMA-ES | 20, **K=3 fixed bundles**, train = **sample 1-of-K** per eval (`stage2_stim_aggregate=sample`) | `de2-maxiter=40`, `popsize=16`, `cma_sigma_scale_stage2=0.02`, `L_threshold=2.0` | **loky** candidate evals | **held-out select** (`--val-seed`, default `seed+7777`); plateau patience 8 on held-out; arm when **train loss of held-out incumbent** `< beat_loss=0.4044` | generalize; rolling `weights_stage2_last` = **held-out incumbent** (written at Stage-2 entry + each held-out improve) |
| **3 — local polish** | **Powell** (default); optional `cma` | same sample-1-of-K train + val gate | **`prior`** `[6,8,10,11]`; `local_refine_patience=8`; **`local_refine_max_wall_s=1800`** | Powell serial; CMA-polish via threading | plateau / wall cap; keep polish only if held-out not worsened | close CMA integrator-gain overshoot without overfitting W |

**Schedules (`--pipeline`):** `de_cma_local` (cold-start default: 1→2→3), `de_cma`
(1→2, no polish), `cma_only` (2→3 from a warm `--resume-json`; the 0.404 baseline was
CMA-only from `stage1_loss≈0.805`). DE Stage 1 is unnecessary when starting from a good
warm vector; keep it for cold starts.

**Local-refine target (`--local-refine-idx`):** `prior` (**default** = `[6,8,10,11]` —
best held-out 2026-08-04g/h), `active` (opt-in; overfits), or a comma list.

**Anti-overfit (2026-08-04h + 2026-08-05a):** `stage2_n_stim_seeds=3` with
**`stage2_stim_aggregate='sample'`** — each train eval draws one of the three fixed
bundles (~**1×** wall vs single-bundle; over many evals all three are seen). Opt-in
`mean` restores the old K× average. `--val-seed` held-out drives selection, early-stop
arming (via train-of-incumbent vs `beat_loss`), and polish keep/reject.

**Rough wall (ORCD, numba + loky, `n_jobs=16`, sample aggregate):**

| Piece | Estimate | Basis |
|-------|----------|--------|
| Stage-1 DE (bps=5) | ~5–20 min | ~0.4 s/eval; pop≈80; 16-way; + optional borderline extend |
| Stage-2 CMA (bps=20, 40 gen) | ~10–30 min | ~7–15 s/gen (2026-08-03f/l short-opt ~7–8 s/gen @ n_jobs=8); early-stop can cut |
| Powell/`prior` polish | ~10–20 min (cap 30) | 2026-08-04f ~15 min @ 1 stim; sample ≈ same; `LOCAL_REFINE_MAX_WALL_S=1800` |
| **Cold `de_cma_local` total** | **~0.5–1.5 h** typical | Target ≲ 1–2 h; Slurm **6 h** margin |
| Warm `cma_only` | **~0.5–1 h** | Skip DE |

(Pre-numba historical Stage-2 alone was ~1.5 h @ 135 s/gen — not the current clock.)

**Variants × seeds & restartability:** deterministic run dir
`weights_run_fw[_<tag>]_<mtype>_mask<slug>_s<seed>/`. `--resume auto` loads the
**latest mtime** checkpoint (rolling Stage-2 / polish / final / de1 ckpt), sets
`resume_from` from kind (`de2` / `local` / `de1`), and warm-restarts (covariance not
persisted). Rolling file is written at **Stage-2 entry** (gen=-1) so a kill before the
first improve is still restartable. `FIT_DONE` = success; `FIT_FAILED` = Stage-1 never
reached `L_threshold` (no `FIT_DONE`; re-submit retries). `FORCE=1` redo.

---

## Dated notes

### 2026-08-03a — Topic opened (superseded framing)

Earlier draft mixed analysis / session-cache speedups.

### 2026-08-03b — Weights baselines; goal reframed

`scripts/bench_fit_baseline_sim.py`. Numba ~26× vs numpy; sim ≫ loss math. Stage-2 CMA refine ≈ 1.5 h for this weights file.

### 2026-08-03c — Retinal frozen confirmed; retinal bench

User: the 0.404 run was **retinal frozen**. Added `scripts/bench_fit_retinal_baseline.py`. Frozen front-end rms ~0.78–0.82 at `bps=10` (~1 s/eval, sim-dominated); smoke DE2 ckpt worse on current data. numpy ~27× slower at `bps=10`.

### 2026-08-03d — Phase 1: numba parity; plan constraints

Numba↔numpy PASS (exact on default mp; ~1e-16 under fitted weights). Dropped plan items: shorter trials / fewer blocks / stim reuse across generations. Agent rule `.cursor/rules/fit-speedups.mdc` + `AGENTS.md` note. Weights-loss wall at `bps=20` cannot move much without faster sim (non-sim ≲ ~0.07 s).

### 2026-08-03e — Hard-require numba; sim section profile

`backend='numba'` in both fit loss cores. Profile: kernel dominates; unused-output skip not worth it (~1%). Per-eval sim further gains need kernel work; **main remaining wall for the 1–2 h goal is Phase 2** (eval count / early-stop / parallel efficiency — historical Stage-2 ~2.2 min/gen vs ~2 s ideal).

### 2026-08-03f — Phase 2a short-opt baseline (perturbed fitted → 10 CMA gens)

**Setup** (`scripts/bench_fit_weights_shortopt.py`):

- Start from `WEIGHTS_REL` θ, freeze `g_m`/`d_m` (below log-bounds in the JSON), small log-space perturb (`--perturb-scale 0.02`, resampled if loss NaN).
- `resume_from='de2'`, Stage-2 CMA only, `de2_maxiter=10`, `popsize=16`, `n_jobs=8`, `blocks_per_session_stage2=20`, `deterministic_stage2=True` (production Stage-2 protocol).
- `fit_weights.py` bottom now guarded (`if __name__ == '__main__'`).

**Current pipeline (numba), tag `phase2a_baseline_numba`:**

| | Value |
|--|--|
| Run dir | `models/weights_run_phase2a_baseline_numba_20260803_132652/` |
| Ref loss (fitted, freeze only) | 0.655 |
| Start loss (perturbed) | **0.746** |
| Best after 10 gens | **0.422** (best hit gen 3; gens 4–9 no improve) |
| Wall total | **98.7 s** |
| Mean s/gen | **7.7 s** (median 8.0) |

**Historical** `weights_run_20251125_182058` first 10 CMA gens (`n_jobs=16`, different start ≈0.45→0.41):

| | Value |
|--|--|
| Best overall gen0 → gen9 | 0.452 → 0.406 |
| Wall to gen9 | **1217 s** |
| Mean s/gen | **135 s** |

**Compare:** current ≈ **18×** less wall per gen than that log on this machine (numba + laptop `n_jobs=8` vs historical 16). Loss paths are not matched starts — use **s/gen** and final-vs-start Δ for regression when editing the pipeline. Re-run:

```bash
conda activate iblenv
PYTHONPATH=. python scripts/bench_fit_weights_shortopt.py \
  --n-gens 10 --n-jobs 8 --popsize 16 --perturb-scale 0.02 \
  --seed 123 --tag phase2a_<edit_name>
```

Report JSON: `phase2a_shortopt_report.json` in the run dir.

### 2026-08-03g — CMA early-stop after plateau

**Code:** `fit_weights_two_stage_v2(..., cma_early_stop_patience=8)` → `_run_cma_es` stops when `best_overall` has not improved for that many gens. `0` / `None` disables. Plateau counter is separate from the sigma-adaptation counter (so sigma shrink at 20 gens does not reset patience). Short-opt: `--patience N`.

**Verify** (`--patience 3 --n-gens 40 --seed 123 --tag phase2a_earlystop`):

| | Value |
|--|--|
| Run dir | `models/weights_run_phase2a_earlystop_20260803_151449/` |
| Start loss | 0.746 (same perturb as 2026-08-03f) |
| Best | **0.437** (hit gen 2) |
| Stop | early-stop at **gen 6** (3 gens no improve); message `early_stop_patience` |
| Wall | **63.3 s** (6 gens, ~7.4 s/gen) |

Mechanism works; **quality risk** clarified in 2026-08-03h (do not use low patience for real fits).

### 2026-08-03h — Early-stop can miss later improvements → default off

Replay of historical Stage-2 CMA (`weights_run_20251125_182058`, improves at gens 0→2→6→12, then flat):

| patience | Would stop | Kept best | vs true min 0.404 |
|--|--|--|--|
| 3 | after gen 5 | **0.441** | **misses** gen-6 / gen-12 drops |
| 5 | after gen 11 | **0.406** | **misses** gen-12 (0.404) |
| 8 | after gen 20 | **0.404** | OK on this log; cuts gens 21–39 |
| 0 (off) | maxiter | 0.404 | full budget |

Short-opt 10-gen baseline best **0.422** vs patience-3 run **0.437** is mostly path noise, but the historical replay shows the real issue: CMA often plateaus for several gens then improves again. Aggressive stop freezes a worse loss.

**Policy (superseded by 2026-08-03i):** bare patience is unsafe before matching the known best.

### 2026-08-03i — Early-stop only after beating known best loss

**Rule:** plateau early-stop is **armed** only once `best_overall < cma_early_stop_beat_loss` (default **0.4044** = WEIGHTS_REL). Until then, run to maxiter / CMA tols. After arming, stop after `cma_early_stop_patience` (default **8**) gens with no further improve.

| Scenario | Behavior |
|--|--|
| Historical refine (min 0.404353) | Arms at gen 12; stops ~gen 20 — keeps 0.404, cuts flat tail |
| patience=3 before beating 0.404 | **Does not stop** — avoids freezing at 0.44 |
| Short-opt (~0.42 best) | Never arms → full `n_gens` budget |

API: `cma_early_stop_beat_loss=0.4044` (None = ungated, not recommended). Short-opt: `--beat-loss` (default = JSON recorded loss), `--no-beat-gate`, `--patience`.

### 2026-08-03j — Phase 2c: paper vs ckpt; `g_i` slice; local-after-CMA

**Paper hand-tweaks vs `WEIGHTS_REL` top-level ckpt** (paper `c` = code `i`):

| Param | Paper | Ckpt | Notes |
|--|--|--|--|
| `W_*` | ≈0.43/0.27/0.496/0.17/0.50 | within ~1% | CMA already matched |
| `d_i` | 21.4 | 21.56 | matched |
| **`g_i`** | **163** | **189.7** | **main gap (+16%)** |
| `g_m`/`d_m` | ~0 | ~0 | matched |
| nested `model_params.g_i` | — | 168.8 | partial hand-edit in JSON |

**`g_i` slice** (`scripts/bench_fit_gi_slice.py`, other params = ckpt, `bps=20`, seeds 0+1): multi-seed **mean** loss prefers `g_i≈210` over 163 (Δ paper−ckpt ≈ +0.01). Both traj and prior decrease as `g_i` rises — paper 163 is **not** the traj+prior minimum under that mean. Report: `…/weights_run_20251125_182058/gi_slice_report.json`.

**Local refine after CMA** (smoke: `scripts/bench_fit_local_after_cma.py`, single Stage-2 stim bundle, `maxiter=15`):

| | Value |
|--|--|
| Wall | **~66 min** (L-BFGS ~36 min + Powell) |
| Loss | 0.655 → **0.530** |
| `g_i` | 189.7 → **165.6** (near paper 163) |
| `d_i` | 21.56 → 26.1 |

So on the **production deterministic Stage-2 bundle**, local polish recovers the hand-tweak direction for `g_i`. Multi-seed mean landscape can disagree — validate finals on held-out seeds; use `stage2_n_stim_seeds=2–3` when wall allows.

**Code:** `local_refine_after_cma=True` (default), `local_refine_idx` default `[6,8,10,11]`, `stage2_n_stim_seeds`, `g_m`/`d_m` log-bounds down to `1e-12`.

### 2026-08-03k — Faster post-CMA local refine (revised)

Smoke wall ~66 min: Powell ran after L-BFGS already improved (`success=False` / hit maxiter).

**Keep quality tols** (`ftol=1e-14`, `gtol=1e-10`, `maxls=100`). Speed via:
- Progress every 5 evals
- `local_refine_use_powell=False` (default); Powell only if enabled **and** no L-BFGS improve
- `local_refine_patience=8`: stop L-BFGS only after that many **iters with no best-loss improve**
- Parallel FD gradients for L-BFGS using `n_jobs` (threading backend — numba releases GIL; 4 refine dims ⇒ up to 4-way parallel per jac)

Smoke: `PYTHONPATH=. python scripts/bench_fit_local_after_cma.py --n-jobs 8 --patience 8`

### 2026-08-03l — Pipeline review: DE→CMA→local, parallelization, backend

Reviewed whether all stages are needed and whether parallelism is efficient.
New bench: `scripts/bench_fit_pipeline_variants.py` (reuses short-opt helpers;
warm start = perturbed fitted θ, Stage-2 CMA only, 10 gens, `n_jobs=8`, seed 123).

**CMA backend — loky vs threading (decisive):**

| variant (warm, no local) | final loss | g_i | wall | s/gen |
|--|--|--|--|--|
| `cma_only_nolocal_loky` | 0.6658 | 186.8 | **89 s** | **7.0** |
| `cma_only_nolocal_threading` | 0.6658 | 186.8 | 369 s | 36.4 |

**loky ~5x faster.** The numba kernel does **not** release the GIL, so a
`threading` backend serializes CMA candidate evals. → keep `parallel_backend='loky'`
for Stage-2 CMA (already the function default; now documented in-code near the
`_run_cma_es` Parallel block). `threading` is retained only for the 4-dim
local-refine FD jac (avoids re-pickling large stim bundles for a tiny eval count).

**Is DE Stage 1 needed?** The 0.404 baseline was **CMA-only** (warm resume,
`stage1_loss≈0.805`), and warm `cma_only` reaches the Stage-2 basin without DE.
So DE Stage 1 is **not** needed when starting from a good warm vector — use
`--pipeline cma_only --resume-json …`. Keep **DE→CMA** as the cold-start default
(DE gives global coverage a single CMA from a random init lacks). Both are now
selectable (no separate optimizer library needed).

**Local refine value is basin-gated.** On this short 10-gen warm start (loss ~0.665,
CMA had **not** reached the good basin), 4-param L-BFGS moved loss only 0.6658→0.6653
and `g_i` barely (186.8→~186) while adding ~300 s. Contrast 2026-08-03j: from a
proper basin (fitted ckpt, loss 0.655) local refine moved `g_i` 189.7→165.6 (≈paper
163). ⇒ local refine pays off only after CMA converges near ~0.404 (production 40-gen
runs); it is wasted polish on an unconverged plateau. Kept **on by default** since
production budgets reach the basin.

**Parallelization fixes (`fit_weights.py`):**
- **Stage-2 DE** was hard-coded `workers=1`. `loss_active_stage2` is a nested closure
  (not picklable for scipy's multiprocessing Pool, which is why Stage-1 uses the
  module-level `_loss_active_de_worker`). Fixed by parallelizing Stage-2 DE through a
  **threading map** (`workers=<joblib threading map>`), gated on `n_jobs≠1`: no
  pickling, GIL released enough for the deferred population. Only affects DE→DE
  schedules (default DE→CMA leaves this path unused).
- **`L_threshold=None` auto path** used `ti = sum(train_mask)` then `len(ti)` (int has
  no `len`) — would crash whenever a caller left `L_threshold` unset. Replaced with
  `n_active = int(np.sum(train_mask))`; `>=10 active → 0.8 else 3.0`.

**Env note (loky + ONE):** one loky run crashed with `BrokenProcessPool` →
`JSONDecodeError` on `~/.one/.openalyx.internationalbrainlab.org`. Fresh loky workers
re-import `model_functions`, which constructs `ONE(...)` and calls `_clear_token`
(rewrites that params file); parallel workers can race and momentarily truncate it.
**Fixed 2026-08-04b — see below (ONE bypass).**

### 2026-08-04b — Bypass ONE in the fitting path (fixes the loky race)

The fit path uses ONE **only** to resolve `one.cache_dir` for `pth_res` / `save_dir`;
no ONE data-loading is called during a fit (grep of `fit_weights.py` for `one`: none).
Constructing ONE in each loky worker was the sole cause of the params-file race.

- `model_functions.py`: ONE construction is now gated. When `PRIOR_MECH_NO_ONE` is
  truthy **and** `ONE_CACHE_DIR` is set, `one = None` and the cache dir comes straight
  from the env — no ONE, no `~/.one` read/write, no network. Otherwise ONE is built as
  before (data-loading scripts that don't set the flag are unaffected).
- `scripts/_one_bypass.py`: imported **before** `model_functions` by the fit driver /
  benches. If `ONE_CACHE_DIR` is unset it constructs ONE **once in the main process**
  (no race) to read `cache_dir`, exports it, and sets `PRIOR_MECH_NO_ONE=1`. loky
  workers inherit the env and skip ONE entirely. If ONE can't be built it silently
  falls back to normal construction.
- `run_fit_weights_slurm.sh` exports `PRIOR_MECH_NO_ONE=1` (ORCD already sets
  `ONE_CACHE_DIR`) → full bypass, safe on internet-less compute nodes.

Smoke (laptop, **loky `n_jobs=4`** — the config that raced before): main + all 4
workers print `[model_functions] ONE bypassed`, CMA runs to completion, no
`BrokenProcessPool`/`JSONDecodeError`. cache_dir resolved to the laptop ONE dir via
the one-time main-process ONE call.

### 2026-08-03m — Phase 2d: ORCD cluster submission for weights fitting

No `fit_weights` SLURM existed in this repo (only in `paper-brain-wide-map`, partition
`fiete`, `generate_jobs.sh` sed style). Added an ORCD `mit_normal` set mirroring the
goal2 two-layer submitter+worker pattern (not the `fiete`/sed style):

- **`scripts/run_fit_weights.py`** — CLI driver refactored from `fit_weights.__main__`
  (`main(argv)`; `__main__` local behavior untouched). Flags: `--seed`, `--n-jobs`
  (default `SLURM_CPUS_PER_TASK`), `--pipeline {de_cma_local,de_cma,cma_only}`,
  `--out-tag`, `--backend` (default loky), `--resume-json`, plus budget knobs. Writes
  a final weights JSON + `run_fit_weights_report.json` per run dir.
- **`scripts/run_fit_weights_slurm.sh`** — worker: `-p mit_normal`, `--nodes=1
  --ntasks=1 --cpus-per-task=16 --mem=40G --time=4:00:00` (historical Stage-2 ≈1.5 h +
  margin), `OMP/MKL/OPENBLAS/NUMEXPR=1`, `MPLBACKEND=Agg`, `module load miniforge` +
  `conda activate ~/conda_envs/ibl`, `PYTHONPATH=$REPO_DIR`, `ONE_CACHE_DIR` default.
  Reads env `SEED/PIPELINE/OUT_TAG/RESUME_JSON/...`.
- **`scripts/submit_fit_weights_sharded.sh`** — login-node seed sweep / multi-start:
  loops `SEEDS` and `sbatch --parsable --mem/--cpus-per-task/--time --export=ALL,...`.
  The seed sweep is the mechanism for finding **alternative parameter sets** of similar
  quality (multi-start) and running fits in parallel. `cma_only` requires `RESUME_JSON`.

Usage: `SEEDS="56 34 78" PIPELINE=de_cma_local bash scripts/submit_fit_weights_sharded.sh`.
Benchmarks / fits can also be run on ORCD via these scripts if the laptop is slow.

### 2026-08-04a — Model variants + restartable per-variant/seed run dirs

Extended the cluster set to fit **several model variants x seeds**, each in its own
restartable folder.

- **Variant = (`mtype` label, `--freeze` indices)** on the 12-d vector
  `[0 W_ii,1 W_pp,2 W_mm,3 W_is,4 W_pi,5 W_mi,6 g_i,7 g_m,8 d_i,9 d_m,10 θ_c,11 θ_d]`.
  Examples: `--mtype none --freeze ""` (fit all 12); `--mtype gain --freeze 7,9`
  (freeze g_m,d_m ~0, per paper). Frozen params are held at ~0 and excluded from the
  active dims / bounds.
- **Deterministic run dir** per (variant, seed):
  `weights_run_fw[_<tag>]_<mtype>_mask<slug>_s<seed>/` (no timestamp) so a re-submit
  targets the same folder.
- **Restart (`run_fit_weights.py --resume auto`, default):** Stage-2 CMA now writes a
  rolling full-vector checkpoint `weights_stage2_last.{json,npy}` on every improved
  generation (`_save_rolling_checkpoint`, atomic replace). On restart the driver loads
  the newest in-folder checkpoint (rolling > final), skips DE Stage 1, and re-enters
  Stage 2 around it (then local). A completed fit writes `FIT_DONE`; re-running skips it
  unless `--force`. Because CMA internal covariance is not persisted, restart is a
  **warm restart from the best vector**, not exact CMA-state resume — fine for
  continuing to improve after a timeout/preemption.
- **Submitter** `submit_fit_weights_sharded.sh`: `VARIANTS="none: gain:7|9"` (indices
  joined by `|` since commas break `sbatch --export`; worker converts `|`->`,`),
  `SEEDS="..."`; loops variants x seeds, exports env, `sbatch --parsable --export=ALL`.

Smoke (laptop, `n_jobs=1` to avoid the loky/ONE race): warm `de_cma`, 2 gens →
folder `…_none_mask7-9_s999` with `weights_stage2_last` (0.738), `weights_final`,
`FIT_DONE`; re-run skipped; `--force` resumed from `weights_stage2_last.json`. All OK.

### 2026-08-04c — Local refine index API + pipeline reference (superseded default)

Recorded a pipeline reference and added `--local-refine-idx {active|prior|<list>}`.

**Superseded (see 2026-08-04g/h):** briefly defaulted refine to all-active; held-out
benches showed `active` overfits vs `prior`. **Current default = `prior` `[6,8,10,11]`**;
`active` remains opt-in.

### 2026-08-04d — loky can't parallelize the FD jac (pickling); local-refine cost model corrected

Tried to speed multi-dim local refine by running the FD-gradient perturbations under
**loky** (like CMA). **Infeasible:** the perturbation task closes over `loss_refine`→
`loss_func_local` (nested closures capturing stim bundles), which are **not picklable** →
`joblib` raises `PicklingError: Could not pickle the task`. Verified with
`bench_fit_local_after_cma.py --backend loky` (errors at nfev=1 for both `prior` and
`active`). This is the same non-picklable-closure wall that forces Stage-1 DE to use a
module-level worker. **Kept threading** for the FD jac (reverted the loky attempt);
comment/docstring corrected (previously wrongly claimed "numba releases GIL").

**Corrected cost model** (earlier linear-in-dims estimate was too pessimistic):
- Threading FD is GIL-serial (kernel `@njit` has no `nogil`), so the **gradient** portion
  is `1 + n_dims` serial sim evals per L-BFGS iter (`prior` n=4 → 5; `active` n=12 → 13).
- But the **line search** (scipy L-BFGS-B, inherently serial `fun` calls, up to
  `maxls=100`) often **dominates**: a `maxiter=3`, `patience=0`, 4-dim run logged
  **nfev≈310, wall≈660 s** (~2.1 s/eval) on a near-flat surface — i.e. ~295 of 310 evals
  were line search, only ~15 were gradient. So the 4→12 dim change grows only the
  gradient part; total is driven more by line-search + iteration count, both dim-independent.
  ⇒ all-active is slower than 4-dim, but **sub-linear in dims**, not ~3x.
- (The 660 s was inflated by `patience=0` for the microbench; production `patience=8`
  stops once best plateaus — here best hit 0.65436 almost immediately.)

**Levers that actually bound local-refine wall** (no kernel change, no loky):
`local_refine_patience=8` (default), the new **`local_refine_max_wall_s`** cap
(exposed as `--local-refine-max-wall-s` / `LOCAL_REFINE_MAX_WALL_S`), and `maxls`
(currently 100 "for quality" per 2026-08-03k; lowering to ~20–25 would cut line-search
thrashing on flat surfaces — left at 100 pending a quality check).

**Only way to truly parallelize the FD gradient** would be `@njit(nogil=True)` on the
sim kernel (then threading parallelizes; loky still blocked by pickling). Deferred: it
touches the hot kernel (needs numba↔numpy parity re-check per `fit-speedups.mdc`) and
its payoff is limited while the serial line search dominates. CMA already gets multicore
via loky, so nogil is not required there.

### 2026-08-04e — all-params vs 4-params refine bench + DE parallelization audit; L-BFGS is the wrong polish

**Refine comparison** (`bench_fit_local_after_cma.py`, from the 0.404 ckpt = 0.655275 on
the deterministic bps=20 bundle; L-BFGS-B, `powell=off`, `patience=8`, `maxiter=40`,
`n_jobs=8`, threading):

| refine set | dims | best loss | reached | behaviour |
|--|--|--|--|--|
| `active` | 12 (incl. W) | **0.6068** | ~458 s | descends further but thrashes on tight W bounds (penalty evals 1e11/100/183…), then stuck to 909 s |
| `prior`  | 4 (g_i,d_i,θ) | **0.6544** | ~312 s | ≈flat from start (0.6553→0.6544); ran to 1237 s with no real progress |

Both sit far above the **0.530** this polish reached in 2026-08-03j — and that drop came
from the **Powell** stage, not L-BFGS. Conclusion: **L-BFGS-B (FD) is the wrong local
optimizer here.** At the checkpoint the g_i/d_i/θ gradient is tiny/noisy (4-dim run can't
move) and near the W bounds the FD steps hit penalty cliffs (12-dim run thrashes).
Refining *more* params helps descent slightly but is slower and noisier; refining *fewer*
stalls immediately. Dim count / parallelism is not the lever — **the optimizer is**.

Also found: the plateau patience uses a `1e-9` improvement threshold, so microscopic noise
(…365→…364) keeps resetting the stall counter → it **never early-stops**, running to
maxiter while flat. Loosen to ~1e-5.

**DE parallelization audit** (`/tmp/de_workers_test.py`, reproduces the Stage-1 pattern):

- Default multiprocessing start method on this mac = **spawn** (verified).
- **Stage-1 DE** (`workers=n_jobs` → scipy `multiprocessing.Pool`; module-level
  `_loss_active_de_worker` + runtime module-global `_LOSS_ACTIVE_DE_CONTEXT`): real
  multicore **only under fork (Linux/ORCD)**. Under spawn the global isn't inherited →
  `workers=1` OK, **`workers=4` FAILS in 0.5 s** with `RuntimeError: CTX not initialized`.
  So cold DE→CMA at `n_jobs>1` **crashes on mac**; works on ORCD. (Fix if needed: force
  fork, or pass ctx via a Pool initializer / picklable args.)
- **Stage-2 DE** (non-default DE→DE only): `workers` = joblib **threading** map →
  GIL-bound → **not real parallelism** (effectively serial).
- **Stage-2 CMA** (the one that matters): **loky → genuinely parallel** (the 5x win).

**Decision:** do NOT pursue FD parallelization / nogil (parallelizing a stalling optimizer
is pointless). Instead change the polish algorithm to a **bounded, derivative-free,
noise-robust** method. Implemented `local_refine_method` ∈ {lbfgs, cma, powell} (+ Powell
fallback, patience threshold loosened 1e-9→1e-5); benched below.

### 2026-08-04f — Polish head-to-head: Powell wins; CMA ok on few dims; L-BFGS worst

Same 0.404 ckpt (start loss **0.655275** on the deterministic bps=20 bundle), patience 8,
maxiter 40, n_jobs 8. `bench_fit_local_after_cma.py --method {cma,powell} --local-refine-idx {prior,active}`:

| method | refine set | final loss | wall | g_i 189.7→ | note |
|--|--|--|--|--|--|
| L-BFGS | 4 `prior` | 0.6544 | ~660 s | 189.7 (stuck) | stalls immediately (2026-08-04e) |
| L-BFGS | 12 `active` | 0.6068 | ~909 s | — | thrashes on W bounds |
| CMA | 4 `prior` | **0.5174** | 682 s | **167.5** | ≈ paper g_i=163; d_i→26.3 |
| CMA | 12 `active` | 0.6553 | 222 s | none | σ=0.05 too small: 9 gens all in W-penalty region → plateau, no move |
| Powell | 4 `prior` | **0.4991** | 900 s | 105.4 | low loss but g_i overshoots |
| Powell | 12 `active` | **0.4903** | 1782 s | 189.6 | **best loss**; descends via W, keeps g_i |

Takeaways:
- **Powell dominates** (best loss on both sets), CMA is a solid #2 (and its `prior` run lands
  g_i≈167, closest to the paper 163), L-BFGS is worst. All of CMA/Powell >> L-BFGS,
  confirming 2026-08-04e (derivative-free is the right family here).
- **CMA-restart doesn't scale to 12-d at σ=0.05** — the tight W bounds make small steps land
  in the penalty region. Would need a larger σ or W-aware scaling; not worth it since Powell wins.
- **Lower loss ≠ paper params:** Powell/prior→g_i 105, Powell/active keeps g_i≈190 (moves W),
  CMA/prior→g_i 167 (≈paper). The loss surface has several comparable minima; the choice of
  refine set + method selects which one.
- Wall: Powell/active best but ~30 min; Powell/prior ~15 min for nearly the same loss.

**Recommendation:** default `local_refine_method='powell'`. Refine-set is a
science/speed tradeoff (open question below).

---

### 2026-08-04g — Same-session eval, held-out session, and paper params (W_pi=1.6e-5)

Three fixes to make the polish comparison trustworthy:

1. **Same-session eval.** All 2026-08-04f losses are on one deterministic bps=20 bundle
   (train seed 123). The ckpt's stored **0.4044** is on *its own* averaged stimulus set, so it
   isn't comparable — re-scored on the seed-123 bundle the same params give **0.6553**. Compare
   methods only within one seed.
2. **Held-out session.** Added `--eval-seed` to `bench_fit_local_after_cma.py`: train on seed
   123, re-score the final θ on seed **777** (same length, different stimuli). Guards against a
   method just memorizing the training bundle.
3. **Paper params.** Added `--paper` (+`--eval-only`). Paper W list
   (w_ss,w_cc,w_mm,w_pp,w_cs,w_mc)=(0,0.43,0.27,0.496,0.17,0.50), g_c=163, d_c=21.4,
   g_m,d_m negligible, θ=(0.76,0.40). **W_pi is load-bearing** — it is *not* in the paper's W
   list; zeroing it (1e-12) blows loss to ~2.7. Correct paper value **W_pi = 1.6e-5**.

**Anchor evals (train seed 123 / held-out seed 777):**

| params | train (123) | held-out (777) | g_i |
|--|--|--|--|
| ckpt (fitted 0.404 JSON) | 0.6553 | 0.5202 | 189.7 |
| **paper** (W_pi=1.6e-5, g_i=163, θ=0.76/0.40) | 0.7606 | **0.5171** | 163 |

→ On the held-out session **paper ≈ ckpt** (0.517 vs 0.520): the hand-tweak is *not* worse
out-of-sample even though its in-sample number looks worse. In-sample loss alone is misleading.

**Held-out = pure eval, not refit.** The fit is deterministic per seed, so the held-out score
is just the *already-fit* seed-123 params re-scored on seed 777 — no second optimization. Added
`--params-json` + `--eval-only` to `bench_fit_local_after_cma.py`; loads a saved weights JSON and
scores it on train+eval seeds via the canonical guarded `_safe_loss_weights_v2`.

**Held-out polish comparison** (fit on 123, score on 777; the four 2026-08-04f JSONs):

| method | refine set | train (123) | held-out (777) | g_i | valid? |
|--|--|--|--|--|--|
| ckpt (0.404 JSON) | — | 0.6553 | 0.5202 | 189.7 | ✓ |
| paper (W_pi=1.6e-5) | — | 0.7606 | 0.5171 | 163 | ✓ |
| CMA | 4 `prior` | 0.5174 | 0.4315 | 167.5 | ✓ |
| **Powell** | 4 `prior` | 0.4991 | **0.4253** | 105.4 | ✓ |
| CMA | 12 `active` | 0.6553 | 0.5202 | 189.7 | ✓ (no move) |
| Powell | 12 `active` | 0.4903\* | **1e11** | 189.6 | ✗ **out of box** |

**Powell/active's *saved vector* was out of the box — but its loss was legal.** The saved JSON
had `W_pp=0.0075` (bound [0.496,0.49999]), `W_mm=3.59` (bound [0.1,0.40]), `g_m=3.8e-13`
(idx 1,2,7 out); re-scored raw → 1e11 on both seeds. **Root cause (a scipy quirk, not a bad
minimum):** the `powell` path minimizes `loss_refine_bounded` (which clips x *before* every eval),
so scipy's reported `result.fun` is the loss at the *clipped* point — but `result.x` is returned
**unclipped**, and the old code saved that raw `result.x`. Clipping the saved vector back into the
box reproduces the reported loss exactly:

| Powell/active | train (123) | held-out (777) |
|--|--|--|
| saved raw (W_mm=3.59) | 1e11 | 1e11 |
| **clipped** (W_mm→0.40, W_pp→0.496) | **0.4903** | **0.4451** |

**Fix (2026-08-04g, `fit_weights.py` ~L2334):** after any polish method, clip
`loc_x_clipped = clip(loc_try.x, L_ref, U_ref)`, build the saved θ from that, and **recompute
`best_fun = loss_func_local(θ_clipped)`** so reported == re-scored for every method. Verified: a
fresh Powell/active run now converges to the same 0.4903 and saves an in-box vector.

**Corrected held-out ranking (all valid, clipped):**

| params | train (123) | held-out (777) | g_i |
|--|--|--|--|
| ckpt (0.404 JSON) | 0.6553 | 0.5202 | 189.7 |
| paper (W_pi=1.6e-5) | 0.7606 | 0.5171 | 163 |
| **Powell / prior** | 0.4991 | **0.4253** | 105.4 |
| CMA / prior | 0.5174 | 0.4315 | 167.5 |
| Powell / active (clipped) | 0.4903 | 0.4451 | 189.6 |
| CMA / active | 0.6553 | 0.5202 | 189.7 (no move) |

**Conclusions:**
- **Best held-out = Powell/prior (0.4253)**, then CMA/prior (0.4315), then Powell/active (0.4451).
  All three beat ckpt (0.5202) and paper (0.5171) out-of-sample. Held-out < train because seed 777
  is an easier bundle.
- **`active` mildly overfits vs `prior`:** Powell/active wins in-sample (0.4903 < 0.4991) but
  *loses* held-out (0.4451 > 0.4253) — the extra W freedom fits the training bundle's noise.
- CMA/prior lands g_i≈167 (≈ paper 163); Powell/prior→g_i 105. Similar loss, different basin.

**Recommendation:** default polish = **`prior` refine set** (g,d,θ), method Powell or CMA (Powell
marginally better held-out; CMA closer to paper g_i and ~25% faster). `active` is now *legal*
(box-clipped) but overfits, so keep it opt-in.

---

### 2026-08-04h — Defaults: n_stim=3, prior polish, in-training held-out (`--val-seed`)

Implemented the three anti-overfit / quality defaults from 2026-08-04g:

1. **`stage2_n_stim_seeds=3`** (was 1). Three fixed deterministic bundles
   (`RandomState(seed+100003+k)`). Originally train loss = **mean** (~3× wall);
   **2026-08-05a** changed default aggregate to **`sample`** (1-of-3 per eval, ~1× wall).
2. **`--val-seed` held-out selection** (CLI default `seed+7777`). Builds a never-trained-on
   bundle matching bench `--eval-seed`. Stage-2 CMA: (a) **selects** the incumbent with
   lowest held-out loss, (b) **early-stops** on held-out plateau (armed after
   train-of-incumbent `< beat_loss` as of 2026-08-05a), (c) post-CMA polish is
   **kept only if held-out is not worsened**.
3. **`local_refine_idx` default → `prior`** (`[6,8,10,11]`). `active` (full 12) remains
   opt-in via `--local-refine-idx active`.

**Wiring:** `fit_weights_two_stage_v2(val_stim_seed=…)` → `_val_loss_active` →
`_run_cma_es(..., val_eval=…)`; CLI/Slurm/submitter defaults updated
(`STAGE2_N_STIM_SEEDS=3`, `LOCAL_REFINE_IDX=prior`, `VAL_SEED` optional). Pipeline
reference table above refreshed.

---

### 2026-08-04i — Polish = Powell default; L-BFGS removed; same 3-seed + val gate

Post-CMA polish cleanup:

- **`local_refine_method` default → `powell`** (was `lbfgs`). L-BFGS path + FD jac
  removed from the polish stage (it stalled on this surface; 2026-08-04e/f). Optional
  `cma` small-sigma restart kept (`--local-refine-method cma`);
  `local_refine_use_powell` is now a CMA→Powell fallback only.
- Polish already used `loss_active_stage2` (= mean over `stage2_n_stim_seeds`) and the
  held-out keep/reject gate; made that explicit in the polish banner
  (`train=mean(3 stim), held-out gate ON`) and wired through CLI/Slurm
  (`LOCAL_REFINE_METHOD=powell`) + bench defaults (`--n-stim-seeds 3`,
  `--eval-seed` → `val_stim_seed`, refine idx `prior`).

---

### 2026-08-05a — Pre-submit fixes: sample multi-stim, held-out ckpt, resume, DE fail

Pre-batch review (efficiency + effectiveness). Changes:

1. **`stage2_stim_aggregate='sample'` (default).** Keep `K=3` fixed bundles but each
   train eval draws **one** at random (~1× wall). `mean` remains opt-in (~K×). Held-out
   selection still scores a never-trained bundle — primary anti-overfit lever.
2. **Yes — select on held-out; checkpoint that incumbent.** Rolling
   `weights_stage2_last` updates on held-out improve (`selection='held_out'`) and at
   Stage-2 entry (gen=-1) so a kill before the first improve is restartable. Early-stop
   **arms** on train loss of the held-out incumbent vs `beat_loss=0.4044` (same scale as
   the 0.404 baseline); plateau still timed on held-out.
3. **Resume = latest mtime** among rolling / polish / final / de1 ckpts (no longer
   hard-prefer `stage2_last` over a newer polish). `kind` → `resume_from`
   (`de2`/`local`/`de1`). `selection=final` on the rolling file counts as polish.
4. **Borderline DE extends** (like CMA) when loss ∈ `[L_threshold, borderline_hi)`.
   If still ≥ `L_threshold` → `fit_status='failed_stage1'`, write **`FIT_FAILED`**, exit 2,
   **no `FIT_DONE`**.
5. **`LOCAL_REFINE_MAX_WALL_S=1800`** default — **both Powell and CMA polish are hard-capped
   at 30 minutes** (keeps best-so-far on exceed; set `0` to disable). Typical
   Powell/`prior` ~10–20 min, CMA/`prior` ~11 min under sample, so the cap is a
   safety net not the usual stop. `_save_params_v2` now includes `theta_log`. Journal
   drift from 2026-08-04c (all-active default) corrected — default remains `prior`.
   Slurm/submit default **TIME=6:00:00**.

Wall estimates under current numba+loky+sample: cold `de_cma_local` **~0.5–1.5 h**
typical (see pipeline reference table).

### 2026-08-05b — Final pre-submit check

Confirmed polish wall cap is wired end-to-end (`run_fit_weights.py` default 1800 →
`fit_weights_two_stage_v2(local_refine_max_wall_s=…)` → Powell callback + CMA-polish
loop; Slurm/submitter export `LOCAL_REFINE_MAX_WALL_S=1800`). Banner prints `max_wall=1800s`.

Bugs found and fixed in this check:

1. **`resume_from='local'` left `de2` undefined** — polish reject / skip-polish paths
   read `de2.fun` → `NameError` on restart from a polished checkpoint. Now builds a
   `SimpleNamespace(x, fun)` from resume loss / Stage-2 loss.
2. **Borderline DE extend skipped on `resume_from='de1'`** — driver zeros `de1_maxiter`
   on resume, so a borderline Stage-1 ckpt went straight to `FIT_FAILED` with no
   extend. Extend now runs for `resume_from in {none, de1}` with a 40-iter fallback
   budget when `de1_maxiter=0`.
3. **Excluded `weights_v2_*.json` from `_find_resume`** — opportunistic mid-fit dumps
   (loss&lt;0.4) could outrank the held-out rolling checkpoint by mtime.

No further blockers found for batch submit under sample aggregate + 6 h Slurm.

### 2026-08-05c — ORCD smoke: Stage-1 DE spawn broke context (fixed)

ORCD smoke (`fw_smoke`, Python 3.13) failed at Stage-1 DE:

`RuntimeError: Stage 1 DE context not initialized`

Cause: `differential_evolution(..., workers=n_jobs)` uses the default mp start
method, which on this node is **spawn** (worker re-imported `model_functions` —
second `ONE bypassed` line). Spawn does not inherit `_LOSS_ACTIVE_DE_CONTEXT`.
(Journal 2026-08-04e had flagged this for mac; ORCD/3.13 hits it too.)

**Fix:** `_make_de_workers` — explicit `multiprocessing.get_context('fork').Pool`
with initializer that installs the context; pass `pool.map` as scipy `workers`.
Used for Stage-1 DE and borderline DE extend. Look for log line
`[Stage1 DE] parallel via fork Pool (n_jobs=16)`.

Also: `sbatch --export=...,FREEZE=7,9,...` truncates at the comma → only `g_m`
frozen. Worker now accepts `FREEZE=7|9`; smoke/submit should use `|` in export.

## Questions to be resolved

1. ~~End-to-end or weights-only?~~ **Answered:** 0.404 baseline = **weights with retinal frozen**. 1–2 h target applies first to that setting; retinal is a separate budgeted stage.
2. ~~Acceptable search approximations (shorter trials / fewer blocks)?~~ **Answered: no.** Keep production trial length and `bps`; do not reuse stim across generations. Unused-output skipping OK if loss-identical. Sampling 1-of-K *fixed* bundles per eval is allowed (not cross-generation stim reuse of one batch for a whole gen).
3. ~~Target machine for the 1–2 h claim?~~ **ORCD `mit_normal`**, 16 cpu; laptop benches for s/gen.
4. **When to unfreeze retinal?** Only if weights-only refits leave unacceptable S residuals vs `avg_mean_R`.
