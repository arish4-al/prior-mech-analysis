# Two-stage fitting: retinal first, then joint (retinal free to tweak)

**Scope:** abandon cold/warm **joint-from-scratch** as the primary path after the
2026-08-11 fair compare failed to beat the separate-fit baseline. Instead:

1. **Fit retinal** under all prior gains ≈0 (`g_i,g_m,d_i,d_m,g_s,d_s` frozen / zero).
2. **Joint Stage-2** (regular / sensory masks as in
   [joint_fitting_pipeline.md](joint_fitting_pipeline.md)): co-optimize W/θ +
   variant gains **and allow retinal dims to move** (not freeze the Stage-1 front-end).

**Not in scope:** weights-only ORCD batch ([simulation_fit_speedups.md](simulation_fit_speedups.md));
prior-distance recovery; re-running failed joint-cold campaigns as the main line.

**Status:** Stage A modernized (2026-08-11d). Joint-direct fair best regular **2.083** vs
WEIGHTS_REL baseline **1.344** — pivot to staged retinal→joint. Next: Stage B
hand-off + regular/sensory campaigns.

**Code:** [`fit_retinal.py`](../fit_retinal.py) (hooks into
`fit_weights_two_stage_v2`); drivers
[`scripts/run_fit_retinal.py`](../scripts/run_fit_retinal.py),
`run_fit_retinal_slurm.sh`, `submit_fit_retinal_sharded.sh`.

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
See dated entry below. Remaining: ORCD multi-seed quality campaign vs frozen ~0.80.

### 2. Wire Stage A → Stage B handoff

- Save Stage A JSON with retinal native params (+ mixed θ) loadable by
  `reconstruct_theta_joint_from_json` / joint resume.
- Stage B default: `--pipeline de_cma_local` or warm `cma_only` from
  retinal+WEIGHTS_REL hybrid JSON; **retinal dims free** under the variant mask
  (do not add 14–20 to `--freeze`).
- Keep fair-compare harness (fixed bundles) as the ranking metric — not raw
  `final_loss` from a single fit seed.

### 3. Implement regular + sensory campaigns

Same variant table as joint_fitting_pipeline; submit shape analogous to
`submit_fit_joint_sharded.sh`, but each seed is **retinal-fit → joint-fit** (or
joint resume from staged JSON). Smoke both masks before a 2×N seed batch.

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

## Questions to be resolved

1. Stage A W/θ anchor (script defaults vs WEIGHTS_REL with zero gains).
2. Stage B polish: include retinal dims or not.
3. ~~Retinal `L_threshold` / `beat_loss`~~ provisional: **2.0 / 0.85** (tune on ORCD).
4. Whether equal-sum `L_w+L_S` still needs retuning once Stage A gives a better S init.
