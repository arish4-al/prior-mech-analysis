# Two-stage fitting: retinal first, then joint (retinal free to tweak)

**Scope:** abandon cold/warm **joint-from-scratch** as the primary path after the
2026-08-11 fair compare failed to beat the separate-fit baseline. Instead:

1. **Fit retinal** under all prior gains ≈0 (`g_i,g_m,d_i,d_m,g_s,d_s` frozen / zero).
2. **Joint Stage-2** (regular / sensory masks as in
   [joint_fitting_pipeline.md](joint_fitting_pipeline.md)): co-optimize W/θ +
   variant gains **and allow retinal dims to move** (not freeze the Stage-1 front-end).

**Not in scope:** weights-only ORCD batch ([simulation_fit_speedups.md](simulation_fit_speedups.md));
prior-distance recovery; re-running failed joint-cold campaigns as the main line.

**Status:** Local Stage B smoke (2026-08-12d) used repo `fit_targets/` nested
I/M `{stim,choice}` (I-pre / M-post solid data on plots). `cma_only` from hybrid
finished; DE Stage-1 at `bps=10` hit S-bucket `<10` → 1e12 because DE jittered
retinal away from Stage A (exact hybrid was never in the pop). **2026-08-12e:**
Stage-1 DE now **holds** Stage-A retinal; CMA/polish unfreeze it
(`--stage1-hold-retinal`, Stage B default on). **2026-08-12f:** ORCD hold-smoke
both masks still 1e12 — `--bps-stage1` never reached joint DE (stale import-time
bps=5). Fix in `loss_joint_core` / DE context; resubmit after it is on `main`.

**Code:** [`fit_retinal.py`](../fit_retinal.py) (hooks into
`fit_weights_two_stage_v2`); drivers
[`scripts/run_fit_retinal.py`](../scripts/run_fit_retinal.py),
`run_fit_retinal_slurm.sh`, `submit_fit_retinal_sharded.sh`. Stage B:
[`scripts/build_stage_b_hybrid.py`](../scripts/build_stage_b_hybrid.py),
[`scripts/submit_fit_stage_b_sharded.sh`](../scripts/submit_fit_stage_b_sharded.sh)
(+ existing `run_fit_joint*` / `fit_joint.py`). Diagnostics:
[`scripts/plot_retinal_fit_s.py`](../scripts/plot_retinal_fit_s.py) (Stage A `L_S`);
[`scripts/plot_best_fit_results.py`](../scripts/plot_best_fit_results.py)
(weights/joint `L_w` traj + prior overlays, notebook data).

---

## Why pivot

From [joint_fitting_pipeline.md](joint_fitting_pipeline.md) 2026-08-11 fair
shared-stim compare (`bps=20`, fixed 5 bundles):

| Source | Fair `L_w+L_S` | Notes |
|--------|---------------:|-------|
| WEIGHTS_REL + gs0 (separate-fit baseline) | **1.344** | Lw 0.561 + LS 0.783 |
| Best joint regular (warmde s101) | **2.083** | +0.74; gap mostly Lw |
| Best joint sensory (cold s101) | **4.347** | +3.00; sensory mostly Stage-1 fail |

Cold10 / warm-DE from cold winners do not reach baseline. Searching the full 21-d
joint space from cold (or from a cold joint winner) is not competitive with the
historical path that first locked a good S front-end, then fitted weights.

**Tension with the original joint rationale:** fitting retinal at `g_*=d_*=0` then
*freezing* it can misattribute prior-driven S structure when `g_s`/`d_s` are free.
This staged plan keeps that concern: Stage 2 **does not freeze retinal** — joint
(or joint-like) search may **tweak** retinal while fitting the regular / sensory
masks. Stage 1 only supplies a strong S-aligned initialization under zero priors.

---

## Target protocol

### Stage A — retinal only (`L_S`)

| Setting | Value |
|---------|--------|
| Loss | `L_S` only (`compute_sse_stim_right` / rms vs `avg_mean_R`) |
| Prior gains | **all zero** — I/M and S (`g_i,g_m,d_i,d_m,g_s,d_s`) |
| Network W / θ | fixed at retinal-script / paper defaults (as today’s `fit_retinal`) |
| Free params | 7-d retinal: `α_w`, `β_w`, `α_d`, `β_d`, `τ_a`, `W_as`, `W_ss` |
| Quality ref | frozen JSON front-end ≈ **0.78–0.82** rms at `bps=10` (see speedups journal) |

### Stage B — joint, retinal free to tweak (`L_w + L_S`)

Warm-start from Stage A retinal + WEIGHTS_REL (or Stage-A-compatible) W/θ; run
joint with the same freeze masks as the joint campaign:

| Label | `--mtype` | `--freeze` | Free (incl. retinal 14–20) |
|-------|-----------|------------|----------------------------|
| **regular** | `regular` | `12\|13` | W + `g_i,g_m,d_i,d_m` + θ + retinal; `g_s,d_s≈0` |
| **sensory** | `sensory` | `6\|7\|8\|9` | W + `g_s,d_s` + θ + retinal; I/M gains ≈0 |

Polish ∩ mask (unchanged intent):

- regular → `[6, 8, 10, 11]` (and decide whether retinal polish is opt-in)
- sensory → `[10, 11, 12, 13]` (same)

Fair compare must reuse the joint protocol (fixed stim bundles, report
`L_w`, `L_S`, total) against baseline **1.344**.

---

## Work plan

### 1. Optimize `fit_retinal` to match current `fit_weights` pipeline

**Done (2026-08-11d).** `fit_retinal` now hooks `fit_weights_two_stage_v2` with
`layout: retinal7`, asinh `β_w`, CLI/ORCD drivers, `FIT_DONE`, held-out CMA, etc.
See dated entry below. ORCD multi-seed + shared-stim ranking: 2026-08-12.

### 2. Wire Stage A → Stage B handoff

**Done (2026-08-12b).** `fit_joint.build_stage_b_hybrid_payload` /
`scripts/build_stage_b_hybrid.py` writes joint21 JSON (WEIGHTS_REL W/g/d/θ ∪
Stage-A retinal, `g_s`/`d_s`≈0). Load with `reconstruct_theta_joint_from_json`
/ `--resume-json`. Default hybrid uses shared-stim best Stage-A **s89**.

### 3. Implement regular + sensory campaigns

**Done (2026-08-12b).** `scripts/submit_fit_stage_b_sharded.sh` builds/refreshes
the hybrid then calls `submit_fit_joint_sharded.sh` with
`VARIANTS="regular:12|13 sensory:6|7|8|9"` (retinal 14–20 **not** frozen).
`LOCAL_REFINE_IDX=prior` ∩ mask → regular `[6,8,10,11]`, sensory `[10,11,12,13]`.
Smoke both masks before a 2×N production batch.

---

### 2026-08-11c — Opened: staged retinal→joint after joint-direct miss

**Decision:** stop treating joint-from-scratch (cold or warm-from-cold-joint) as
the main line. WEIGHTS_REL under joint loss is still the fair baseline (**1.344**);
best joint regular **2.083** did not close the gap.

**Next implementation order:**

1. ~~Modernize `fit_retinal`~~ → 2026-08-11d.
2. Handoff JSON → joint warm-start with retinal **unfrozen**.
3. ORCD smokes: regular `12|13`, sensory `6|7|8|9`.
4. Fair `bps=20` compare vs 1.344.

**Open design choices (resolve while implementing):**

1. Stage B start: cold DE on joint dims with retinal warm, vs `cma_only` from
   hybrid WEIGHTS_REL+Stage-A retinal.
2. Whether Stage-3 polish may touch retinal (`RETINAL_REFINE_IDX`) or stays
   prior/θ-only as today.
3. Stage A network W: keep retinal-script defaults vs start from WEIGHTS_REL W
   with gains zeroed (affects which S residual retinal absorbs).

---

### 2026-08-11d — Modernized `fit_retinal` to weights-era pipeline

Ported Stage A onto `fit_weights_two_stage_v2` hooks (same pattern as `fit_joint`).

**API / layout**

| Item | Choice |
|------|--------|
| Dims | 7 (`layout: retinal7`) |
| `β_w` | **asinh** (`BETA_W_SCALE=0.05`), matching joint |
| Loss | `L_S` only; all prior g/d forced 0 each eval |
| Network W/θ | retinal-script Stage-A anchors |
| Schedules | `de_cma_local` / `de_cma` / `cma_only` |
| Defaults | `bps1=10`, `bps2=20`, `L_threshold=2`, `beat_loss=0.85`, polish=all 7 |
| Run dir | `retinal_run_fr[_tag]_<mtype>_mask<slug>_s<seed>/` |
| Module bottom | **guarded** (import-safe) |

**Drivers:** `scripts/run_fit_retinal.py` + `run_fit_retinal_slurm.sh` +
`submit_fit_retinal_sharded.sh`.

**`fit_weights` fix:** Local-after-CMA logging no longer indexes `θ[6]/θ[8]` when
`D_full < 12` (was crashing 7-d retinal polish).

**Local smoke** (`--out-tag smoke --seed 999`, tiny DE/CMA budget, `n_jobs=1`):
`FIT_DONE`, final `L_S≈0.72` (held-out select rejected polish). Not a quality
claim — wiring only. Frozen front-end reference remains ~0.78–0.82.

### 2026-08-11e — Stim / defaults audit fixes

1. **Stage A / B:** both use `deterministic_stage2` + `stage2_restim=True` —
   fixed seeds, rebuild stim each eval (α_w/β_w free-safe; sample-K + held-out).
   Weights-only still caches bundles (`stage2_restim=False`).
2. **Non-retinal init:** confirmed identical to original script
   (`W_pp=0.45`, …, all g/d=0, θ=0.78/0.54) via `STAGE_A_NETWORK` / `STAGE_A_THETA`.
3. **P2:** reconstruct defaults → Stage-A retinal anchors; no more
   `|θ[1]|≤0.25` native guess — require `layout`/`beta_w_coord` markers.
4. **Retinal tracked loss:** no opportunistic `retinal_v2_*` dumps on
   `loss < SAVE_THRESH` / every 1000 steps — only stage rolling / DE-CMA ckpts / finals.

```bash
# ORCD smoke
SEEDS="999" OUT_TAG=smoke DE1_MAXITER=2 DE2_MAXITER=3 POPSIZE=8 SOBOL_COUNT=4 \
  PATIENCE=0 LOCAL_REFINE_MAX_WALL_S=60 FORCE=1 \
  bash scripts/submit_fit_retinal_sharded.sh

# Production multi-start
SEEDS="56 34 78 89 202" bash scripts/submit_fit_retinal_sharded.sh
```

**Still TODO (Stage B):** hybrid JSON (Stage-A retinal ∪ WEIGHTS_REL W/θ) →
`run_fit_joint` with retinal free; regular `12|13` + sensory `6|7|8|9`.

---

### 2026-08-12 — Stage A multi-seed ORCD results + shared-stim eval

**Runs** (local openalyx mirror): `retinal_run_fr_retinal_masknone_s{56,34,78,89,202}/`
(`de_cma_local`, default budgets, `bps1=10` / `bps2=20`, seed-restim Stage 2).

| seed | recorded fit `L_S` | notes |
|-----:|-------------------:|-------|
| 202 | **0.370** | lowest recorded |
| 89 | 0.371 | |
| 78 | 0.748 | |
| 34 | 1.200 | |
| 56 | 1.925 | highest recorded |

**Fair re-eval** (same protocol as notebook / Stage-A restim): `bps=20`, stim seed
**12345**, rebuild under each mp’s α_w/β_w; loss =
`compute_sse_stim_right(mean_S_by_contrast(…), avg_mean_R)`.
Script: [`scripts/plot_retinal_fit_s.py`](../scripts/plot_retinal_fit_s.py).
Plots + `summary.json`:
`~/Downloads/ONE/openalyx…/models/retinal_s_fit_plots_bps20_seed12345/`.

| seed | recorded | shared `L_S` | R² |
|-----:|---------:|-------------:|---:|
| **89** | 0.371 | **0.559** | 0.668 |
| 202 | 0.370 | 0.614 | 0.564 |
| 78 | 0.748 | 0.618 | 0.610 |
| 34 | 1.200 | 0.657 | 0.649 |
| 56 | 1.925 | 0.704 | 0.635 |

**Takeaways**

1. Ranking by recorded fit loss ≠ shared-stim ranking: **s89** wins fair `L_S`;
   s202 (best recorded) is 2nd–3rd on the held-out batch.
2. All five shared `L_S` ∈ **0.56–0.70**, beating the frozen-JSON front-end
   reference (~**0.78–0.82**). Stage A quality is usable for Stage B handoff.
3. Prefer **s89** (then s202 / s78) as Stage-A retinal init(s) for joint warm-start;
   do not pick solely by `retinal_final_loss*`.

**Eval note:** do not call `_update_model_params_for_dt` after applying fitted
retinal — it hard-resets `tau_a→222.68`.

**Next:** Stage B hybrid JSON (Stage-A retinal ∪ WEIGHTS_REL) → joint with retinal
free; start with s89.

---

### 2026-08-12b — Stage B handoff + campaign submit

**Hybrid builder** (`fit_joint.build_stage_b_hybrid_payload` /
[`scripts/build_stage_b_hybrid.py`](../scripts/build_stage_b_hybrid.py)):

| Field | Source |
|-------|--------|
| W, g_i/g_m, d_i/d_m, θ | WEIGHTS_REL (`loss0p4044`) |
| retinal 7-d | Stage-A final (default **s89**) |
| `g_s`,`d_s` | ≈0 (regular freezes; sensory trains from ~0) |
| `layout` | `joint21` + `theta_log` (asinh `β_w`) |

Local artifact:
`~/Downloads/ONE/…/models/stage_b_hybrid_WEIGHTS_REL_retinal_s89.json`

**Campaign driver** [`scripts/submit_fit_stage_b_sharded.sh`](../scripts/submit_fit_stage_b_sharded.sh):
refresh hybrid → `RESUME_JSON` → `submit_fit_joint_sharded.sh`.

```bash
# Smoke both masks
SEEDS="999" OUT_TAG=stageB_smoke \
  DE1_MAXITER=2 DE2_MAXITER=3 POPSIZE=8 SOBOL_COUNT=4 \
  PATIENCE=0 LOCAL_REFINE_MAX_WALL_S=60 FORCE=1 \
  bash scripts/submit_fit_stage_b_sharded.sh

# Production (2 variants × N seeds); retinal free under both masks
SEEDS="56 34 78 89 202" OUT_TAG=stageB_s89 \
  bash scripts/submit_fit_stage_b_sharded.sh
```

Run dirs: `weights_run_fj_stageB_s89_<mtype>_mask<slug>_s<seed>/`.
Polish: `LOCAL_REFINE_IDX=prior` ∩ mask (retinal polish opt-in via `retinal|active`).

**Still open:** ORCD smoke → production; fair `bps=20` `L_w+L_S` vs baseline **1.344**.
Also: sync nested `mean_data` / paper prior `.npy` to ORCD (2026-08-12c) before
comparing Stage B losses to notebook / **1.344**.

---

### 2026-08-12c — Flat `mean_data_results` understated `L_w` (fit vs notebook)

**Symptom** (plotting best weights-only ORCD finals with
`loss_plot_diff_by_condition_with_data` / `loss_prior_effect`, matching
`paper-brain-wide-map/model_test.ipynb`): IM post/pre missing solid data curves;
prior data overlays looked wrong vs the notebook.

**Root cause — not the loss functions.** `loss_weights_core_v2` /
`loss_plot_diff_by_condition_with_data` / `loss_prior_effect` are fine. Drivers
wired the wrong files:

| File | Wrong (ORCD / `run_fit_weights` default) | Correct (notebook) |
|------|------------------------------------------|--------------------|
| `mean_data_results.npy` | `manifold/res` **flat**: I = stim keys only, M = choice keys only | `paper-brain-wide-map` **nested** `{stim, choice}` for both I and M |
| `data_act_block_during{stim,choice}.npy` | `manifold/figs` (different curves) | paper-brain-wide-map copies |

`_data_mean_and_baseline` always needs the **stim** window for baseline. With the
flat file: I-post scores; I-pre / M-post / M-pre all drop (`gof` NaN). Traj loss ≈
**I-post + P penalties** only. `run_fit_*._ensure_data_links` only creates cwd
symlinks if missing — never refreshes a bad existing link/file.

**Re-score** (shared `bps=20` seed `12345`; gain finals from speedups ORCD batch):

| Model | Recorded (ORCD) | Flat + manif prior | Nested + paper prior |
|-------|----------------:|-------------------:|---------------------:|
| gain s89 | 0.217 | **0.345** | **3.54** |
| gain s78 | 0.249 | **0.303** | **0.77** |

So the “beat 0.404” weights-only winners were optimized under the incomplete
target. They are **not** notebook-comparable `L_w`. Stage B / joint fair compare
vs **1.344** must use nested paper targets (same as `model_test.ipynb`).

**Local fix**

1. Replaced ONE `manifold/res/mean_data_results.npy` and
   `manifold/figs/data_act_block_*.npy` with paper copies (backups:
   `*.flat_bak.npy`, `*.manif_bak.npy`).
2. Fail-closed nested `{stim,choice}` check after load in
   [`scripts/run_fit_weights.py`](../scripts/run_fit_weights.py),
   [`scripts/run_fit_joint.py`](../scripts/run_fit_joint.py),
   [`fit_weights.py`](../fit_weights.py) `__main__`.
3. Plot driver [`scripts/plot_best_fit_results.py`](../scripts/plot_best_fit_results.py)
   — notebook-matching traj + prior overlays for weight/joint finals (see below).
4. **Script fix (same day):** shared [`scripts/_fit_data.py`](../scripts/_fit_data.py)
   + repo [`fit_targets/`](../fit_targets/) (copied notebook nested mean_data, paper
   prior curves, `avg_mean_R`). Drivers load **only** from `fit_targets/` and
   refresh cwd symlinks — no ONE / sibling-paper fallback for these files.

**Plotting script** ([`scripts/plot_best_fit_results.py`](../scripts/plot_best_fit_results.py)):

- Runs `mean_by_condition` → `loss_plot_diff_by_condition_with_data` +
  `loss_prior_effect` (same call pattern as `model_test.ipynb`).
- Shared stim session (`--bps 20`, `--seed 12345` default) across models.
- Defaults to ORCD gain finals s89 / s78; override with `--weights-json`.
- Loads nested mean_data + prior curves from paper / `fit_targets` (not flat
  `manifold/res`).
- Writes SVG (+ PNG) under
  `…/models/remote/fit_result_plots_bps20/<run>/` and
  `figs/fit_result_plots_bps20/`.

```bash
PRIOR_MECH_NO_ONE=1 PYTHONPATH=. python scripts/plot_best_fit_results.py \
  --bps 20 --seed 12345
```

**ORCD:** pull `main` (includes `fit_targets/`). No separate ONE file swap needed
for these four targets; `behavior.npy` still comes from ONE `manifold/res`.

Cross-ref: weights-only campaign context in
[simulation_fit_speedups.md](simulation_fit_speedups.md); joint baseline **1.344**
in [joint_fitting_pipeline.md](joint_fitting_pipeline.md).

---

### 2026-08-12d — Local Stage B smoke + nested-target plot check

**Data files (confirmed):** drivers and
[`scripts/plot_best_fit_results.py`](../scripts/plot_best_fit_results.py) load
**only** repo [`fit_targets/`](../fit_targets/):

- `mean_data_results.npy` — I/M `mean_traj` nested `{stim, choice}` (stim 96-bin,
  choice 72-bin). Fail-closed if flat.
- cwd prior links → `fit_targets/data_act_block_during{stim,choice}.npy`
- `avg_mean_R.npy` for `L_S`

Not ONE `manifold/res` (flat incomplete).

**Stage B smokes** (regular freeze `12,13` = `g_s`/`d_s`, seed 999, hybrid resume,
`--force`):

| Run | Pipeline | Result |
|-----|----------|--------|
| `stageB_smoke` | `de_cma_local`, `bps=10` Stage 1 | **failed** `loss=1e12` (S buckets `<10` → NaN → fail-closed) |
| `stageB_cma` | `cma_only` 3 iters + 60s Powell | **ok** recorded **1.241**; `g_s`/`d_s` stayed ≈0 |

Run dir: `…/models/weights_run_fj_stageB_cma_regular_mask12-13_s999`

**Shared-stim plots** (`plot_best_fit_results.py`, `bps=20`, seed `12345`):

| Model | Recorded | eval traj+prior |
|-------|---------:|----------------:|
| hybrid (WEIGHTS_REL ∪ s89 retinal) | 0.776 | **0.533** (0.417+0.116) |
| WEIGHTS_REL | 0.404 | **0.581** (0.435+0.147) |
| Stage B CMA smoke | 1.241 | **0.542** (0.403+0.139) |

WEIGHTS_REL recorded 0.404 is the **flat-target** ORCD number; nested re-score
is 0.581 traj+prior (no `L_S`). CMA 1.241 includes `L_S` on a tiny search — not
a fair vs **1.344**.

**Visual check:** I-pre and IM-post have **solid data curves** (I-pre rises to
~0.8 at t=0; post has early bump + late rise). Prior-effect panels have noisy
solid data in both 0–80 and −80–0. That is the nested-target signature; the
flat file dropped I-pre / M-post.

Plots: `…/models/stageB_smoke_plots/` and
`figs/fit_result_plots_bps20/weights_run_fj_stageB_cma_regular_mask12-13_s999/`.

**Do not** use `de_cma_local` Stage 1 at `bps=10` for Stage B smoke. Use
`cma_only` from hybrid, or Stage 1 at `bps=20`.

---

### 2026-08-12e — Stage-1 DE holds Stage-A retinal; CMA unfreezes

The failed `stageB_smoke` DE (`nit=1`, `nfev=304`, loss `1e12`) never evaluated
the loaded hybrid. `_make_init_population` injected **jittered** copies of x0
(`N(0, 0.05)` on all free dims, including retinal 14–20) and did not keep the
exact Stage-A vector. Restarts dropped x0 entirely. Scipy DE then “converged”
on a flat 1e12 plateau after one generation — more `maxiter` cannot help.

**Change (optimizer only; loss / bucket rules unchanged):**

1. `--stage1-hold-retinal` → `train_mask_stage1` freezes 14–20 in Stage-1 DE
   **at `theta_log0`** (Stage-A values), not `freeze_fill` / LOG_ZERO.
2. After Stage 1 passes `L_threshold`, those dims are unfrozen for Stage-2 CMA
   and polish (refine set ∪ retinal).
3. DE init pop now includes the **exact** x0 as member 0 (plus jittered copies).
4. Stage B submit defaults `STAGE1_HOLD_RETINAL=1`.

Regular DE active set: W + I/M gains + θ (12-d) with retinal locked. Sensory:
W + θ + `g_s`/`d_s` (10-d). Ablation freezes (`12|13` / `6|7|8|9`) unchanged.

```bash
SEEDS="999" OUT_TAG=stageB_hold_smoke \
  DE1_MAXITER=2 DE2_MAXITER=3 POPSIZE=8 SOBOL_COUNT=4 \
  PATIENCE=0 LOCAL_REFINE_MAX_WALL_S=60 FORCE=1 \
  bash scripts/submit_fit_stage_b_sharded.sh
```

**Local smoke (2026-08-12e, same day):** regular `12|13`, seed 999, `--stage1-hold-retinal`,
`bps=10`, tiny budget (`de1=1`, `de-popsize=4`, `de2=1`, polish 30s cap). **FIT_DONE.**
DE held `[14–20]`, active 12, pop 48; Stage-1 loss **3.503** (not 1e12). Unfroze to
19-d for CMA; polish ∪ retinal. Recorded **1.534**, wall 193s. Random DE members
still print S-bucket `<10`; the held hybrid is enough for a finite best. ORCD
smoke can proceed (`STAGE1_HOLD_RETINAL=1` is the Stage B default).

---

### 2026-08-12f — ORCD hold-smoke both masks 1e12 (bps-stage1 never reached joint DE)

ORCD jobs `20307629` (regular) and `20307630` (sensory), `OUT_TAG=stageB_hold_smoke`,
seed 999: Stage 1 `loss=1e12`, `wall≈20s`, skip Stage 2. Same S-bucket `<10` prints
as the local no-hold DE failure.

**Cause:** `--bps-stage1` did not change joint Stage-1 session length. `loss_joint_core`
used `from fit_weights import blocks_per_session` (import-time **5**).
`run_fit_joint.py` set `fw.blocks_per_session = 20` but that does not update the
imported name. DE workers also passed `blocks_per_session_override=None`. Stage B’s
`BPS_STAGE1=20` default was a no-op; DE ran at **bps=5**. At that length, empty
S-buckets are common (16-way fork, unseeded `stim_rng=None`). Scipy then sees a
flat 1e12 landscape (`de1_inf_restarts=2` → three 1-gen attempts ≈ 20s) and
gates out.

Local hold-smoke succeeded at the same effective bps=5 because a single-machine
4-worker stream happened to draw a stim with enough trials (Stage-1 loss 3.503).
Not a hold-retinal regression.

**Fix:** `loss_joint_core` uses live `fw.blocks_per_session`; DE context passes
that as `blocks_per_session_override`; driver also sets `fj.blocks_per_session`.
Slurm echo now prints `BPS_STAGE1` / `STAGE1_HOLD_RETINAL`.

Resubmit after this is on ORCD `main` (`FORCE=1`, new tag or same):

```bash
SEEDS="999" OUT_TAG=stageB_hold_smoke TIME=1:00:00 \
  DE1_MAXITER=2 DE2_MAXITER=3 POPSIZE=8 SOBOL_COUNT=4 \
  PATIENCE=0 LOCAL_REFINE_MAX_WALL_S=60 FORCE=1 \
  bash scripts/submit_fit_stage_b_sharded.sh
```

Log must show `BPS_STAGE1=20`, `hold_retinal=True`, `[Stage1] holding 7 dims`,
and a **finite** Stage-1 loss (not 1e12).

---

## Questions to be resolved

1. Stage A W/θ anchor (script defaults vs WEIGHTS_REL with zero gains).
2. ~~Stage B polish: include retinal dims or not.~~ with `--stage1-hold-retinal`,
   polish ∪ retinal (CMA already searches retinal after DE).
3. ~~Retinal `L_threshold` / `beat_loss`~~ provisional: **2.0 / 0.85** (tune on ORCD).
4. Whether equal-sum `L_w+L_S` still needs retuning once Stage A gives a better S init.
5. ~~Sync nested `mean_data` + paper prior `.npy` to ORCD~~ → vendored in
   `fit_targets/` on `main`; re-baseline WEIGHTS_REL / Stage B fair `L_w` on the
   notebook target (flat-scale losses obsolete).
