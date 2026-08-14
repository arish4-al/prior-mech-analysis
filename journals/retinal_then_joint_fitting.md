# Two-stage fitting: retinal first, then joint (retinal free to tweak)

**Scope:** abandon cold/warm **joint-from-scratch** as the primary path after the
2026-08-11 fair compare failed to beat the separate-fit baseline. Instead:

1. **Fit retinal** under all prior gains ≈0 (`g_i,g_m,d_i,d_m,g_s,d_s` frozen / zero).
2. **Joint Stage-2** (regular / sensory masks as in
   [joint_fitting_pipeline.md](joint_fitting_pipeline.md)): co-optimize W/θ +
   variant gains **and allow retinal dims to move** (not freeze the Stage-1 front-end).

**Not in scope:** weights-only ORCD batch ([simulation_fit_speedups.md](simulation_fit_speedups.md));
prior-distance recovery; re-running failed joint-cold campaigns as the main line.

**Status:** Stage B winners regular **s101**, sensory **s23**. Split-conditioned
`--full-analysis`: regular **3/3**, sensory **2/3** with S prior sig. Stim-side
**unsplit:** regular S **collapses** (0.003, p=0.63); sensory S **survives**
(0.069, p=0). **Harris unique-null** agrees at both nrand=100/6-block (2026-08-13d)
and nrand=2000/40-block (2026-08-14): regular unsplit S null; sensory unsplit S
survives. Split-conditioned regular S remains the f1/f2 composition artefact.
I/M remain sig. both models. **Future Harris / long-session runs: ORCD**, not
the laptop (`session_cache/` wiped 2026-08-14). See 2026-08-13b / 13c / 13d / 14.

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

**ORCD hold-smoke retry (same day, after bps fix):** both masks `FIT_DONE` on a
tiny budget. Regular recorded **1.281** (DE 1.005; polish rejected). Sensory
recorded **2.183**. Wiring only — not a quality claim.

---

### 2026-08-12g — Stage B production `stageB_hold_s89` + shared-stim plots

**Runs:** `weights_run_fj_stageB_hold_s89_{regular_mask12-13,sensory_mask6-7-8-9}_s{56,34,78,89,202}/`
all `FIT_DONE`. Hybrid resume = WEIGHTS_REL ∪ Stage-A retinal **s89**. Stage-1 DE
held retinal; CMA/polish unfroze. `bps1=bps2=20`.

**Shared-stim eval** ([`scripts/plot_best_fit_results.py`](../scripts/plot_best_fit_results.py)):
one `bps=20` session, stim seed **12345**, stim built from the **hybrid** mp
(Stage-A α_w/β_w baked into the batch). Nested `fit_targets/` I/M + prior +
`avg_mean_R`. `L_w` = traj+prior; **fair** = `L_w+L_S`.

This is the notebook 1-bundle protocol, **not** the 2026-08-11 5-bundle mean that
gave WEIGHTS_REL **1.344**. On *this* batch WEIGHTS_REL is **1.260**.

| Model | recorded | L_w (traj+prior) | L_S | **fair** |
|-------|---------:|-----------------:|----:|---------:|
| hybrid (init) | 0.776 | 0.533 (0.417+0.116) | 0.496 | **1.028** |
| WEIGHTS_REL | 0.404 | 0.581 (0.435+0.147) | 0.679 | 1.260 |
| **regular s34** | 0.952 | **0.527** (0.320+0.207) | 0.495 | **1.023** |
| regular s89 | 1.149 | 0.598 (0.403+0.195) | 0.496 | 1.094 |
| regular s56 | 2.399 | 0.634 (0.347+0.287) | 0.496 | 1.129 |
| regular s78 | 1.045 | 0.663 (0.387+0.276) | 0.508 | 1.171 |
| regular s202 | 1.108 | 0.691 (0.420+0.271) | 0.496 | 1.187 |
| **sensory s89** | 1.125 | 0.585 (0.364+0.221) | 0.441 | **1.025** |
| sensory s56 | 1.304 | 0.631 (0.472+0.159) | 0.434 | 1.065 |
| sensory s202 | 1.357 | 0.661 (0.558+0.104) | **0.431** | 1.093 |
| sensory s34 | 1.658 | 0.698 (0.327+0.371) | 0.454 | 1.152 |
| sensory s78 | 1.399 | 0.711 (0.471+0.240) | 0.455 | 1.167 |

**Takeaways**

1. All 10 Stage B finals beat the old joint path (best regular **2.083**) and beat
   WEIGHTS_REL on this batch. Ranking by recorded fit loss ≠ shared-stim ranking
   (regular s56 recorded 2.40 but fair 1.13).
2. Most of the drop vs 1.344 / WEIGHTS_REL `L_S=0.679` is **Stage A**: hybrid
   `L_S=0.496`. Regular Stage B barely moves `L_S` (retinal stayed at s89).
   Sensory shaves `L_S` to **0.43–0.45**.
3. Best regular **s34** is a small `L_w` win vs hybrid (traj 0.320 vs 0.417;
   prior worse 0.207 vs 0.116). Net fair **1.023 vs 1.028**.
4. Best sensory **s89** matches hybrid fair via better `L_S`, worse `L_w`.
5. Regular s89 collapsed `g_i` to 0.28 (hybrid ~190) — still finite, not the
   shared-stim winner.

**Best-of parameters** (native; frozen gains are LOG_ZERO ≈ `9.36e-14`, written 0).
Shared-stim winners, not lowest recorded fit loss.

Regular **s34** — `weights_final_loss0p9523_20260812-161934.json`  
(`frozen_idx=[12,13]` = `g_s`,`d_s`)

| | W_ii | W_pp | W_mm | W_is | W_pi | W_mi |
|--|-----:|-----:|-----:|-----:|-----:|-----:|
| s34 | 0.402 | 0.496 | 0.299 | 0.178 | 2.31e-5 | 0.492 |
| hybrid | 0.426 | 0.496 | 0.270 | 0.168 | 1.63e-5 | 0.507 |

| | g_i | g_m | d_i | d_m | g_s | d_s | θ_c | θ_d |
|--|----:|----:|----:|----:|----:|----:|----:|----:|
| s34 | 166.2 | ≈0 | 22.36 | ≈0 | **0** | **0** | 0.768 | 0.408 |
| hybrid | 189.7 | 0 | 21.56 | 0 | 0 | 0 | 0.760 | 0.401 |

| | α_w | β_w | α_d | β_d | τ_a | W_as | W_ss |
|--|----:|----:|----:|----:|----:|-----:|-----:|
| s34 | 2.201 | −0.0138 | 32.37 | 1.193 | 117.7 | 36.95 | 0.00214 |
| hybrid | 2.209 | −0.0139 | 32.38 | 1.194 | 117.3 | 36.85 | 0.00215 |

Sensory **s89** — `weights_final_loss1p125_20260812-164336.json`  
(`frozen_idx=[6,7,8,9]` = `g_i`,`g_m`,`d_i`,`d_m`)

| | W_ii | W_pp | W_mm | W_is | W_pi | W_mi |
|--|-----:|-----:|-----:|-----:|-----:|-----:|
| s89 | 0.424 | 0.497 | 0.313 | 0.188 | 8.55e-5 | 0.505 |

| | g_i | g_m | d_i | d_m | g_s | d_s | θ_c | θ_d |
|--|----:|----:|----:|----:|----:|----:|----:|----:|
| s89 | **0** | **0** | **0** | **0** | 0.709 | 12.12 | 0.681 | 0.443 |

| | α_w | β_w | α_d | β_d | τ_a | W_as | W_ss |
|--|----:|----:|----:|----:|----:|-----:|-----:|
| s89 | 2.211 | +0.0052 | 25.00 | 0.811 | 156.2 | 37.22 | 0.00215 |
| hybrid | 2.209 | −0.0139 | 32.38 | 1.194 | 117.3 | 36.85 | 0.00215 |

Regular s34 stayed near Stage-A retinal; `g_i` dropped 190→166. Sensory s89 moved
the front-end (`β_w` sign flip, `α_d`/`β_d` down, `τ_a` 117→156) and learned
`g_s≈0.71`, `d_s≈12`.

**Plots** (S = `model_vs_data_stim_Right (+1).svg` + `S_fit.png`; I/M pre/post;
P; prior effects):

- `~/Downloads/ONE/…/models/stageB_hold_s89_plots_bps20/`
- copy: `figs/fit_result_plots_bps20/weights_run_fj_stageB_hold_s89_*/`
  (hybrid folder: `hybrid_WEIGHTS_REL_retinal_s89/`)

---

### 2026-08-13 — Stage B second seed batch, shared-stim re-rank

**Runs:** same hybrid / `OUT_TAG=stageB_hold_s89`, seeds
`7 12 23 42 45 67 101 111 303 333` × regular `12|13` + sensory `6|7|8|9`.
All 20 `FIT_DONE` (local openalyx `models/`). Combined with batch-1
(`56 34 78 89 202`) → **15 × 2**.

**Eval:** same protocol as 2026-08-12g (`bps=20`, stim seed **12345**, stim from
hybrid mp, nested `fit_targets/`). Hybrid re-eval matched **fair 1.0284**.

| Model | recorded | L_w (traj+prior) | L_S | **fair** |
|-------|---------:|-----------------:|----:|---------:|
| hybrid (init) | 0.776 | 0.533 (0.417+0.116) | 0.496 | 1.028 |
| WEIGHTS_REL | 0.404 | 0.581 (0.435+0.147) | 0.679 | 1.260 |
| **regular s101** | 1.131 | **0.504** (0.402+0.102) | 0.497 | **1.001** |
| regular s333 | 0.962 | 0.521 (0.360+0.162) | 0.496 | 1.017 |
| regular s34 | 0.952 | 0.527 (0.320+0.207) | 0.495 | 1.023 |
| regular s12 | 0.960 | 0.532 (0.406+0.126) | 0.496 | 1.027 |
| regular s303 | 1.030 | 0.549 (0.335+0.215) | 0.496 | 1.045 |
| regular s45 | 1.524 | 0.576 (0.399+0.177) | 0.496 | 1.072 |
| regular s7 | 0.996 | 0.550 (0.320+0.230) | 0.526 | 1.076 |
| regular s89 | 1.149 | 0.598 (0.403+0.195) | 0.496 | 1.094 |
| regular s42 | 1.169 | 0.630 (0.421+0.210) | 0.496 | 1.126 |
| regular s56 | 2.399 | 0.634 (0.347+0.287) | 0.496 | 1.129 |
| regular s78 | 1.045 | 0.663 (0.387+0.276) | 0.508 | 1.171 |
| regular s67 | 1.380 | 0.683 (0.521+0.163) | 0.496 | 1.179 |
| regular s202 | 1.108 | 0.691 (0.420+0.271) | 0.496 | 1.187 |
| regular s111 | 1.549 | 0.780 (0.391+0.389) | 0.496 | 1.276 |
| regular s23 | 1.168 | 0.983 (0.316+0.667) | 0.496 | 1.479 |
| **sensory s23** | 1.148 | 0.559 (0.381+0.178) | 0.446 | **1.005** |
| sensory s89 | 1.125 | 0.585 (0.364+0.221) | 0.441 | 1.025 |
| sensory s333 | 1.523 | 0.615 (0.435+0.180) | 0.439 | 1.054 |
| sensory s56 | 1.304 | 0.631 (0.472+0.159) | 0.434 | 1.065 |
| sensory s12 | 1.584 | 0.619 (0.477+0.142) | 0.448 | 1.068 |
| sensory s101 | 1.927 | 0.632 (0.447+0.185) | 0.436 | 1.068 |
| sensory s7 | 1.123 | 0.632 (0.452+0.180) | 0.450 | 1.083 |
| sensory s67 | 1.357 | 0.646 (0.434+0.212) | 0.441 | 1.087 |
| sensory s202 | 1.357 | 0.661 (0.558+0.104) | **0.431** | 1.093 |
| sensory s111 | 1.546 | 0.574 (0.504+0.070) | 0.536 | 1.111 |
| sensory s34 | 1.658 | 0.698 (0.327+0.371) | 0.454 | 1.152 |
| sensory s78 | 1.399 | 0.711 (0.471+0.240) | 0.455 | 1.167 |
| sensory s303 | 1.068 | 0.504 (0.329+0.175) | 0.672 | 1.176 |
| sensory s45 | 1.575 | 0.787 (0.681+0.106) | 0.449 | 1.236 |
| sensory s42 | 1.292 | 0.957 (0.509+0.449) | 0.424 | 1.381 |

Beat hybrid **1.028**: regular s101 / s333 / s34 / s12; sensory s23 / s89.
Worse than WEIGHTS_REL **1.260** on this batch: regular s111, regular s23,
sensory s42.

**Takeaways**

1. Second batch replaces the winners. Regular **s101 fair 1.001** and sensory
   **s23 fair 1.005** beat first-batch s34 / s89 and the hybrid init. Recorded
   fit loss still ≠ shared-stim rank (s101 rec 1.131 vs s12 rec 0.960).
2. Regular s101 is an `L_w` win (0.504 vs hybrid 0.533): prior **0.102** (hybrid
   0.116) and traj 0.402 (hybrid 0.417). `L_S` stays at Stage A (~0.497).
   Learned **`g_m≈0.20`** (hybrid ~0); `W_mi` 0.507→0.576. Retinal still s89.
3. Sensory s23 beats hybrid via `L_S` 0.446 (hybrid 0.496), with `L_w` a bit
   worse (0.559). Unlike first-batch sensory s89, **retinal stayed at Stage A**
   (no `β_w` sign flip / `τ_a` jump). Gains are much larger: **`g_s≈38`,
   `d_s≈33`** vs s89’s 0.71 / 12.
4. Regular `L_S` remains pinned ~0.496 except s7 (0.526). Sensory `L_S` is
   typically 0.42–0.45; s111 (0.536) and s303 (0.672) drifted the front-end
   (s303 recorded 1.068 looked good, fair did not).
5. Regular s23 exploded prior (0.667) → fair **1.479**, the worst regular.

**Best-of parameters** (native; frozen gains LOG_ZERO ≈ `9.36e-14`, written 0).
Shared-stim winners, not lowest recorded fit loss.

Regular **s101** — `weights_final_loss1p131_20260812-232651.json`  
(`frozen_idx=[12,13]` = `g_s`,`d_s`)

| | W_ii | W_pp | W_mm | W_is | W_pi | W_mi |
|--|-----:|-----:|-----:|-----:|-----:|-----:|
| s101 | 0.425 | 0.496 | 0.255 | 0.157 | 1.67e-5 | 0.576 |
| hybrid | 0.426 | 0.496 | 0.270 | 0.168 | 1.63e-5 | 0.507 |

| | g_i | g_m | d_i | d_m | g_s | d_s | θ_c | θ_d |
|--|----:|----:|----:|----:|----:|----:|----:|----:|
| s101 | 196.4 | **0.204** | 19.99 | ≈0 | **0** | **0** | 0.768 | 0.389 |
| hybrid | 189.7 | 0 | 21.56 | 0 | 0 | 0 | 0.760 | 0.401 |

| | α_w | β_w | α_d | β_d | τ_a | W_as | W_ss |
|--|----:|----:|----:|----:|----:|-----:|-----:|
| s101 | 2.214 | −0.0140 | 32.55 | 1.193 | 117.7 | 36.74 | 0.00215 |
| hybrid | 2.209 | −0.0139 | 32.38 | 1.194 | 117.3 | 36.85 | 0.00215 |

Sensory **s23** — `weights_final_loss1p148_20260812-234002.json`  
(`frozen_idx=[6,7,8,9]` = `g_i`,`g_m`,`d_i`,`d_m`)

| | W_ii | W_pp | W_mm | W_is | W_pi | W_mi |
|--|-----:|-----:|-----:|-----:|-----:|-----:|
| s23 | 0.403 | 0.499 | 0.241 | 0.239 | 8.74e-6 | 0.526 |
| hybrid | 0.426 | 0.496 | 0.270 | 0.168 | 1.63e-5 | 0.507 |

| | g_i | g_m | d_i | d_m | g_s | d_s | θ_c | θ_d |
|--|----:|----:|----:|----:|----:|----:|----:|----:|
| s23 | **0** | **0** | **0** | **0** | **38.28** | **32.89** | 0.663 | 0.499 |
| s89 (old) | 0 | 0 | 0 | 0 | 0.709 | 12.12 | 0.681 | 0.443 |

| | α_w | β_w | α_d | β_d | τ_a | W_as | W_ss |
|--|----:|----:|----:|----:|----:|-----:|-----:|
| s23 | 2.208 | −0.0138 | 32.40 | 1.194 | 117.1 | 36.80 | 0.00215 |
| hybrid | 2.209 | −0.0139 | 32.38 | 1.194 | 117.3 | 36.85 | 0.00215 |
| s89 (old) | 2.211 | +0.0052 | 25.00 | 0.811 | 156.2 | 37.22 | 0.00215 |

**Plots:** `~/Downloads/ONE/…/models/stageB_hold_s89_plots_bps20/` (incl.
`batch2_summary.json`); copy `figs/fit_result_plots_bps20/`.

---

### 2026-08-13b — BWM classifier + prior tests on Stage B winners

Goal-1 `--full-analysis` on shared-stim winners (regular **s101**, sensory **s23**).
Canonical analysis: 80 ms S / 150 ms I/M, fill-from-next-ITI, contrast-matched
null. Seed **123**, 40 sessions, nrand **100**, n-jobs 8. Output under
`manifold_sim/stageB_bwm/` (not the canonical `goal1/absence/` tree).

**Load:** `load_fitted_model` now re-applies JSON `retinal` after
`_update_model_params_for_dt` (that helper still hard-resets `tau_a→222.68`).
Without this, Stage B `τ_a≈117` would be discarded. WEIGHTS_REL has no `retinal`
key → unchanged. Verified: s101/s23 `tau_a≈117.7/117.1`; WEIGHTS_REL `222.68`.

```bash
# regular s101 — absence-style (fitted I/M, g_s=d_s=0)
python simulate_recovery.py --run-experiment absence --full-analysis \
  --seed 123 --n-sessions 40 --nrand 100 --n-jobs 8 \
  --weights-json …/weights_run_fj_stageB_hold_s89_regular_mask12-13_s101/weights_final_loss1p131_*.json \
  --output-dir …/manifold_sim/stageB_bwm

# sensory s23 — P→S only (I/M zeroed)
python simulate_recovery.py --run-experiment s_presence --full-analysis \
  --g-s-presence 38.28114411878634 --d-s-presence 32.885204811626714 \
  --seed 123 --n-sessions 40 --nrand 100 --n-jobs 8 \
  --weights-json …/weights_run_fj_stageB_hold_s89_sensory_mask6-7-8-9_s23/weights_final_loss1p148_*.json \
  --output-dir …/manifold_sim/stageB_bwm
```

Session cache **MISS** both (new mp). Wall ~5.6 min each.

#### Regular s101 — `goal1/absence/cm_full/`

`g_s=d_s=0`, `g_i≈196`, `d_i≈20`, **`g_m≈0.20`**, Stage-A retinal.

Classifier **3/3**. Σ metrics almost identical to canonical WEIGHTS_REL absence
(2026-07-07b):

| pop | true | pred | Σ^stim,s | Σ^stim,m | mono | sc_stim | sc_choice |
|-----|------|------|----------|----------|------|---------|-----------|
| S | S | S | **0.993** | 0.509 | 0 | 0.007 | 0.491 |
| I | I | I | 0.351 | 0.178 | 0 | 0.649 | 0.822 |
| M | M | M | 0.330 | 0.174 | **1** | 0.670 | 0.826 |

Absence reference: S Σ=0.993, I 0.351, M 0.331 / mono=1.

Prior (`population_prior_tests` / `prior_modulation`, all p_mean=0 / ≈0.01):

| pop | amp_euc | sig | absence amp (07-07) |
|-----|--------:|:---:|--------------------:|
| S | **1.076** | ✓ | 0.950 |
| I | 0.070 | ✓ | 0.106 |
| M | 0.411 | ✓ | 0.379 |

`s_prior_stats`: S curve_mean **0.199**, p_mean=p_gain=**0**. Nonzero `g_m` did
not break 3/3 recovery; I/M prior-mod signatures still sculpt the classifier.

#### Sensory s23 — `goal1/s_presence_g_s38p2811_d_s32p8852/cm_full/`

`g_i=d_i=g_m=d_m=0`, **`g_s≈38.28`, `d_s≈32.89`**, Stage-A retinal still in
place (no s89-style `β_w`/`τ_a` move). Inside-adaptation `g_s`.

Classifier **2/3** (S and I recovered; M→I). Historical P→S-only at
`g_s=1800/2025` on WEIGHTS_REL was **1/3** (S→I).

| pop | true | pred | Σ^stim,s | Σ^stim,m | mono | sc_stim | sc_choice |
|-----|------|------|----------|----------|------|---------|-----------|
| S | S | S | **0.801** | 0.429 | 0 | 0.199 | 0.571 |
| I | I | I | 0.715 | 0.240 | 0 | 0.285 | 0.760 |
| M | M | **I** | 0.641 | 0.212 | 0 | 0.359 | 0.788 |

S Σ=0.801 is just above the 0.8 stimulus threshold (fragile). M still lacks a
pre-movement ramp (`mono=0`) — no I/M prior mod.

Prior — **S, I, and M all significant** (p_mean=0):

| pop | amp_euc | p_mean | sig |
|-----|--------:|-------:|:---:|
| S | 0.065 | 0 | ✓ |
| I | 0.211 | 0 | ✓ |
| M | **0.597** | 0 | ✓ |

`s_prior_stats`: S curve_mean **0.055**, p_mean=**0**, p_gain=0.01 (not <0.01).
Amp order I/M ≫ S: S is detectable but small; downstream I (and M via `W_mi`)
are larger, matching the historical I-before-S pattern. Contrast Experiment 1
on WEIGHTS_REL (`g_s=189.7`, `d_s=21.6`): S curve_mean 0.037, p=0.15 ✗. Here
`d_s≈33` plus Stage-A retinal is enough for S `p_mean` significance, in the
same neighborhood as Experiment 2’s S-onset (`g_s=10`, `d_s=40`).

#### Takeaways

1. Regular Stage B θ behaves like canonical absence for the BWM classifier
   (**3/3**) and for S/I/M prior significance, despite `g_m≈0.20` and Stage-A
   retinal (`τ_a≈118`, `α_w≈2.21` vs WEIGHTS_REL 222.68 / 1.565).
2. Sensory Stage B θ is **not** a 1/3 negative control: S is classified as
   stimulus and S prior distance is significant, at joint-fitted
   (`g_s,d_s`)≈(38, 33) with the Stage-A front-end. M still fails classification.
3. Sensory S amp (0.065) is ~16× smaller than regular S (1.076). Detectable ≠
   absence-scale. Downstream I/M still dominate the prior-distance readout.

**CSVs:** `manifold_sim/stageB_bwm/goal1/{absence,s_presence_g_s38p2811_d_s32p8852}/cm_full/figs/`
(`bwm_classification.csv`, `population_prior_tests.csv`, `prior_modulation.csv`,
`s_prior_stats.csv`). Cross-ref:
[bwm_classification_recovery.md](bwm_classification_recovery.md),
[direct_sensory_prior_coupling.md](direct_sensory_prior_coupling.md).

---

### 2026-08-13c — Stim-side unsplit prior distance (s101 / s23)

Same θ, sessions (cache HIT), seed 123, nrand 100, contrast-matched null, 80 ms S /
150 ms I/M. `--unsplit-prior` default `stim_side`: `stim_l_unsplit` +
`stim_r_unsplit` stacked (no f1/f2). Not fully unsplit.

```bash
python simulate_recovery.py --unsplit-prior absence \
  --seed 123 --n-sessions 40 --nrand 100 --n-jobs 8 \
  --weights-json …/regular_mask12-13_s101/weights_final_loss1p131_*.json \
  --output-dir …/manifold_sim/stageB_bwm

python simulate_recovery.py --unsplit-prior s_presence \
  --g-s-presence 38.28114411878634 --d-s-presence 32.885204811626714 \
  --seed 123 --n-sessions 40 --nrand 100 --n-jobs 8 \
  --weights-json …/sensory_mask6-7-8-9_s23/weights_final_loss1p148_*.json \
  --output-dir …/manifold_sim/stageB_bwm
```

| Model | pooling | S curve_mean | S p | I curve_mean | I p | M curve_mean | M p |
|-------|---------|-------------:|----:|-------------:|----:|-------------:|----:|
| regular s101 | f1/f2 (13b) | 0.199 | **0** | — | sig | — | sig |
| **regular s101** | **stim_side unsplit** | **0.0027** | **0.63** | 0.498 | **0** | 1.638 | **0** |
| WEIGHTS_REL absence | stim_side unsplit | 0.011 | 0.13 | 1.099 | 0 | 3.078 | 0 |
| sensory s23 | f1/f2 (13b) | 0.055 | **0** | — | sig | — | sig |
| **sensory s23** | **stim_side unsplit** | **0.069** | **0** | 0.260 | **0** | 0.824 | **0** |

WEIGHTS_REL absence unsplit from [split_conditioning_vs_unsplit.md](split_conditioning_vs_unsplit.md).
Split I/M in 13b were `amp_euc` (different metric); unsplit numbers are `curve_mean`.

**Takeaways**

1. Regular S prior on the f1/f2 test was the known **composition artefact**.
   Unsplit S is null and below the WEIGHTS_REL absence unsplit residual. I and M
   stay large — genuine I/M prior mod, same pattern as canonical absence unsplit.
2. Sensory S **does not collapse**: unsplit 0.069, p=0 (p_mean/p_gain/p_amp all
   0). Slightly *larger* than the split-conditioned 0.055. Direct P→S at
   (`g_s,d_s`)≈(38, 33) with Stage-A retinal is detectable without f1/f2
   selection. I and M remain larger (0.26 / 0.82).
3. So the 13b regular S significance should not be read as sensory prior
   coupling; the 13b sensory S significance should.

**Outputs:** `manifold_sim/stageB_bwm/unsplit_prior/seed_123/{absence_unsplit,s_presence_g_s38p2811_d_s32p8852_unsplit}/`

---

### 2026-08-13d — Harris unique-null prior distance (s101 / s23)

Act_block analog of `_harris_unique` from
[structured_nulls_choice_lr.md](structured_nulls_choice_lr.md): freeze each
session's S/I/M trials on the split's stim×choice (or stim-side) stratum;
null labels are **other-session prior sequences** from the same stratum,
length-matched with a contiguous window, unique patterns only. Observed
sessions are leave-one-out donors; plus **40 extra** sessions at
`seed=123+10007` (cache keys `1592e616773630d4` regular /
`ee44046fa73b85fe` sensory). Canonical 80 ms S / 150 ms I/M, fill-from-next.
Does not overwrite contrast-matched outputs.

CLI: `--harris-unique-null` (incompatible with `--full-analysis` /
`--label-shuffle-null`). Default `--harris-n-extra-donors 40`.

```bash
python simulate_recovery.py --unsplit-prior absence --harris-unique-null \
  --seed 123 --n-sessions 40 --nrand 100 --n-jobs 8 \
  --weights-json …/regular_mask12-13_s101/weights_final_loss1p131_*.json \
  --output-dir …/manifold_sim/stageB_bwm

python simulate_recovery.py --unsplit-prior s_presence --harris-unique-null \
  --g-s-presence 38.28114411878634 --d-s-presence 32.885204811626714 \
  --seed 123 --n-sessions 40 --nrand 100 --n-jobs 8 \
  --weights-json …/sensory_mask6-7-8-9_s23/weights_final_loss1p148_*.json \
  --output-dir …/manifold_sim/stageB_bwm

# split-conditioned (f1/f2) analog — see caveat below
python simulate_recovery.py --run-experiment absence --harris-unique-null …
python simulate_recovery.py --run-experiment s_presence --harris-unique-null …
```

#### Stim-side unsplit (the 13c test under Harris)

Unique pool **100/100** on both stim_l and stim_r; 39/40 sessions kept
(1 skipped: no donor with ≥ n_elig stratum trials). Null median curve_mean
≈ 0.002, same ballpark as contrast-matched shuffle.

| Model | null | S curve_mean | S p | S p_gain | I curve_mean | I p | M curve_mean | M p |
|-------|------|-------------:|----:|---------:|-------------:|----:|-------------:|----:|
| regular s101 | contrast-matched (13c) | 0.0027 | **0.63** | — | 0.498 | **0** | 1.638 | **0** |
| **regular s101** | **Harris unique** | **0.0031** | **0.33** | 0.33 | 0.503 | **0** | 1.652 | **0** |
| sensory s23 | contrast-matched (13c) | 0.069 | **0** | — | 0.260 | **0** | 0.824 | **0** |
| **sensory s23** | **Harris unique** | **0.069** | **0** | **0** | 0.260 | **0** | 0.823 | **0** |

Harris does **not** overturn 13c. Regular unsplit S stays null (p=0.33;
observed still sits on the Harris floor). Sensory unsplit S still p=0 at
the same 0.069, now with p_gain=0 too. I/M stay far above the Harris null
in both models.

#### Split-conditioned f1/f2 — unique pool collapses on f2

Congruent f1 splits reached unique=100. **Incongruent f2** saturated at
1–38 unique nulls (regular) or skipped entirely / unique=1–9 (sensory):
small stim×choice cells are prior-imbalanced, so transplanted donor
labels often fail ≥2/side. Combined `p_mean=0` for S/I/M is **not
comparable** to contrast-matched nrand=100 (combined n_null is the min
unique across splits; sensory M had a single null curve). Do not use the
split-conditioned Harris combined p-values for claims.

| Model | pooling | Harris S curve_mean | Harris S p | note |
|-------|---------|--------------------:|-----------:|------|
| regular s101 | f1/f2 | 0.203 | 0† | composition artefact still present; p under-resolved |
| sensory s23 | f1/f2 | 0.037 | 0† | f2 S/I skipped; p under-resolved |

† under-resolved unique pool on f2.

**Takeaways**

1. The Harris analog is well-powered on **stim-side unsplit** (U=100).
   It is a stricter structured null than contrast-matched shuffle and
   still agrees with 13c: regular S is composition; sensory S is not.
2. Extra donor sessions were needed in the sense the user flagged: 40
   extra at a held-out seed plus leave-one-out among the observed 40.
   Unsplit still skipped 1/40 recipients (stratum longer than every donor).
3. Split-conditioned Harris unique-null is **not** a drop-in replacement
   for the 13b contrast-matched f1/f2 test — f2 unique-pool collapse is
   the same pathology the real-data journal warns about for short strata.

**Outputs:**
- unsplit: `manifold_sim/stageB_bwm/unsplit_prior/seed_123/{absence_unsplit_harris_unique,s_presence_g_s38p2811_d_s32p8852_unsplit_harris_unique}/`
- split: `manifold_sim/stageB_bwm/goal1/{absence,s_presence_g_s38p2811_d_s32p8852}/hu_sprior/`

---

### 2026-08-14 — long sessions, nrand=2000, extra 80 donors; then wipe cache

Re-ran the 13d analog at real-data null resolution:
`--n-sessions 40 --blocks-per-session 40 --nrand 2000 --harris-n-extra-donors 80`
(~1779–2300 trials/session). Output root:
`manifold_sim/stageB_bwm/harris_bps40/`. Combined p averages real/null curves
across splits first (product-MC when unique counts are ragged; here they were
not). All splits **unique=2000, kept=40, skipped=0** — the 13d f2 unique-pool
collapse is gone.

| Model | pooling | S | S p | I | I p | M | M p |
|-------|---------|--:|----:|--:|----:|--:|----:|
| regular s101 | f1/f2 | 0.195 | 0 | 0.131 | 0 | 0.591 | 0 |
| sensory s23 | f1/f2 | 0.066 | 0 | 0.105 | 0 | 0.394 | 0 |
| regular s101 | unsplit | **0.0003** | **0.63** | 0.541 | 0 | 1.773 | 0 |
| sensory s23 | unsplit | **0.072** | **0** | 0.269 | 0 | 0.849 | 0 |

Same scientific picture as 13c/13d: regular unsplit S null; sensory unsplit S
survives; split-conditioned regular S is still the f1/f2 composition artefact.

**Cache:** this campaign added **~12 GB** of 40-block pickles; whole
`session_cache/` reached **42 GB**. After the run finished, the **entire**
`<ONE cache>/manifold_sim/session_cache/` was deleted. `harris_bps40/` results
(~31 MB) were kept. See [simulation infrastructure](simulation_infrastructure.md).

**Policy:** do not re-run Harris unique-null with long sessions / `nrand=2000` /
extra donors on the laptop. Submit on **ORCD**. Laptop stays for 6-block /
`nrand=100` / contrast-matched checks.

### 2026-08-14b — real-data Harris unique unsplit analog

Wired the 13c/13d unsplit design on BWM `act_block` prior tests: stim-aligned
stratify by stim only (`act_block_duringstim_{l,r}`); movement-aligned
stratify by choice only (`act_block_duringchoice_{l,r}`); Harris unique-null.
Submit on ORCD:

```bash
bash scripts/submit_goal2_act_block_harris_unsplit_sharded.sh
```

Details: [structured nulls](structured_nulls_choice_lr.md) 2026-08-14c.

---

## To-do

1. ~~**More Stage B seeds, shared-stim eval.**~~ Done 2026-08-13. Combined
   15 × 2; winners **regular s101** / **sensory s23**.

2. ~~**BWM classification + prior tests on the best Stage B θ.**~~ Done
   2026-08-13b. Regular **3/3** (S/I/M prior sig.); sensory **2/3** with S
   prior sig. at `g_s≈38`, `d_s≈33`.

3. ~~**Harris unique-null on S/I/M prior distance.**~~ Done 2026-08-13d
   (nrand=100 / 6-block) and 2026-08-14 (nrand=2000 / 40-block). Unsplit agrees
   with 13c at both resolutions. Further Harris / long-session runs → **ORCD**.

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
