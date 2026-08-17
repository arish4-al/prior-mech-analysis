# Split conditioning vs unsplit prior distance

**Scope:** how the f1/f2 (choice × feedback) conditioning of the `act_block_duringstim` splits creates S prior distance by trial composition, what happens when those splits are removed, and how the choice of null (contrast-matched vs label shuffle) interacts with all of it.

**Status:** decisive result in hand. Absence S prior distance collapses from **0.798 → 0.011 (n.s.)** when f1/f2 conditioning is dropped while stim side is preserved. I and M prior effects survive and grow. The split-conditioned S readout is therefore largely a composition artefact.

Sources: dated entries 2026-06-29 (Goals 1–2, Experiment A), 2026-07-06h (Tables A–C).

---

## Goal

**Hypothesis:** significant S prior distance can arise from **selection bias** and from **prior effects on I/M** (choice composition, threshold shifts, RT truncation) even when there is no meaningful direct P→S coupling — or when `g_s`/`d_s` are too weak to produce a detectable sensory signature.

Evidence already in hand before the experiment:

- Absence condition: large S prior distance with `g_s=0` (I/M-mediated).
- Phase 4b (pre-fix): residual significant S at c=1.0 on `*_f1` splits only, with no g/d at all.
- S-only presence at canonical params: I significant, S not — same split pipeline.

**Key test:** re-run the same analyses **without splitting into trial types** (no f1/f2 / stim×choice conditioning). This was the clean test flagged in the 2026-06-20 open questions.

**Success criteria**

- If S prior distance collapses to null when unsplit → the significant S in the split pipeline is largely a composition artefact.
- If S prior distance persists unsplit with `g_s=0` → a non-selection mechanism is needed (block epoch, constant S0, etc.).

**Working hypotheses going in**

1. Significant S in the split pipeline is often confounded — driven by I/M prior mod changing which trials enter each split, not by direct sensory prior coupling.
2. Detectable sensory gain on S requires unphysical `g_s` (~5–11× `g_i`) or implausible `d_s` offsets; biologically plausible coupling gives **I significant, S not**.
3. Weak S → strong I is the plausible biological regime for direct P→S coupling.
4. Unsplit analysis is the decisive test; most of the 2026-06-20 sweeps are inconclusive on biological plausibility until split conditioning is removed.

---

## Experiment A — Unsplit prior distance

**Implementation:** `--unsplit-prior {phase4,absence,all,s_presence,presence}` with `--unsplit-mode`:

| Mode | Splits | Meaning |
|------|--------|---------|
| `stim_side` (default) | S/I: `stim_l`+`stim_r` unsplit; M: `choice_l`+`choice_r` unsplit (since 2026-08-14e) | No f1/f2; stim side for stim-aligned, choice side for move-aligned |
| `fully` | `act_block_duringstim_fully_unsplit` | All duringstim trials, L+R mixed (**diagnostic only**) |

Canonical fill-next + S=80 ms / I/M=150 ms throughout.

```bash
conda activate iblenv
# stim-side unsplit (recommended)
python simulate_recovery.py --unsplit-prior phase4 absence \
  --seed 123 --n-sessions 40 --nrand 100 --n-jobs 8
# fully unsplit (L+R pooled — S artefact risk)
python simulate_recovery.py --unsplit-prior phase4 absence --unsplit-mode fully --seed 123 ...
```

### Results (seed 123, 40 sessions, nrand=100, α=0.01, contrast-matched null)

| Pooling | Case | S curve_mean | S p | I curve_mean | I p | M curve_mean | M p |
|---------|------|-------------|-----|-------------|-----|-------------|-----|
| **f1/f2 splits** (4) | Phase 4b | 0.012 | 0.78 ✗ | 0.004 | 0.60 ✗ | 0.004 | 0.81 ✗ |
| **f1/f2 splits** (4) | Absence | 0.798 | 0.00 ✓ | — | — | — | — |
| **stim_side unsplit** (2) | Phase 4b | 0.003 | 0.64 ✗ | 0.003 | 0.17 ✗ | 0.004 | 0.20 ✗ |
| **stim_side unsplit** (2) | Absence | 0.011 | 0.13 ✗ | 1.099 | 0.00 ✓ | 3.078 | 0.00 ✓ |
| **fully unsplit** (1) | Phase 4b | **0.297** | **0.00** ✓† | 0.264 | 0.00 ✓† | 0.504 | 0.00 ✓† |
| **fully unsplit** (1) | Absence | **0.404** | **0.00** ✓† | 1.911 | 0.00 ✓ | 4.526 | 0.00 ✓ |

† Spurious for S (and likely I/M at phase4): L+R stim pooled without channel alignment.

**Output paths** (base `manifold_sim/unsplit_prior/seed_123/`):

| Mode | Phase 4b | Absence |
|------|----------|---------|
| stim_side | `.../phase4_no_prior_mod_unsplit/` | `.../absence_unsplit/` |
| fully | `.../phase4_no_prior_mod_fully_unsplit/` | `.../absence_fully_unsplit/` |

### Interpretation

1. **Three-way comparison:** removing f1/f2 splits collapses absence **S** (0.80 → 0.011, n.s.) when stim side is preserved. **Fully unsplit** restores a large spurious S (0.40, p=0) — an artefact of mixing left- and right-stim trials in one S distance (activity on different channels).
2. **Phase 4b fully unsplit** (S=0.30, p=0) is **not** evidence of prior coupling; split and stim_side-unsplit Phase 4b are both null.
3. **Absence fully unsplit** (S=0.40) **overstates** S prior distance vs stim_side unsplit (0.011) — channel mixing adds ~0.39 to S `curve_mean`. I/M are significant in all unsplit modes (genuine prior effects in integrator/motor).
4. **Hypothesis 1 strongly supported** via stim_side unsplit: split-conditioned S (0.80) is almost entirely f1/f2 composition artefact; I/M effects are real without that filter.
5. **Do not use fully unsplit for S inference** — only as a diagnostic showing why stim-side conditioning is required.

---

## Null scheme: contrast-matched vs label shuffle

From the Goal-1 matrix (seed 123, 40 sessions, nrand=100; see [simulation infrastructure](simulation_infrastructure.md) for how the matrix was run). `curve_mean` is **null-independent** — identical under CM and LS; only p-values change.

The four experiments: **phase4** (all g/d=0), **absence** (g_s=d_s=0, fitted I/M), **s1800** (`g_s=1800, d_s=0`, I/M off), **s2025** (`g_s=2025, d_s=0`, I/M off).

**Table A — SPLIT-conditioned (S/I/M split), contrast-matched (CM) vs label-shuffle (LS)**

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

**Table C — SPLIT vs UNSPLIT `curve_mean` (CM)** — effect of dropping f1/f2 splits

| exp | pop | split | unsplit | ratio |
|-----|-----|------:|--------:|------:|
| absence | S | 0.798 | 0.0111 | **0.014** |
| absence | I | 0.492 | 1.099 | 2.2 |
| absence | M | 2.028 | 3.078 | 1.5 |
| s1800 | S | 0.0418 | 0.0197 | 0.47 |
| s1800 | M | 0.0260 | 0.0125 | 0.48 |
| s2025 | S | 0.0542 | 0.0250 | 0.46 |
| s2025 | M | 0.0316 | 0.0158 | 0.50 |

### Findings

1. **Null choice:** label shuffle is systematically **more conservative** (higher p) than contrast-matched. Borderline S-presence effects (s1800 M split; s1800 I/M/S unsplit; s2025 I/M) are significant under CM but **not** under LS. phase4 and absence are robust to the null choice.
2. **Split vs unsplit:** the absence-S signal is enormous when split (0.798) but **collapses to ~0.011 (n.s.) when unsplit** (ratio 0.014) — absence-S is driven by the f1/f2 splits, not stim side alone. Absence I/M instead *grow* when unsplit. S-presence S/M roughly halve unsplit, with S still significant under CM.
3. **Classification (BWM Σ recovery)** for all four experiments is in [BWM classification recovery](bwm_classification_recovery.md).

**Caveat:** in the original matrix, `ls_unsplit` overwrote `cm_unsplit` (shared output dir, no null tag). The four CM unsplit runs were repeated and preserved as `*_CM_summary.json`. Future matrices should write null-specific unsplit directories.

---

## Data sources

- Split results: `goal1/{exp}/{cm,ls}_sprior/*_summary.json`
- Unsplit results: `unsplit_prior/seed_123/{exp}_unsplit/*_summary.json` (CM preserved as `*_CM_summary.json`)

---

### 2026-08-13c — Stage B fitted θ (s101 / s23)

Full write-up: [retinal_then_joint_fitting.md](retinal_then_joint_fitting.md) 2026-08-13c.
Stim-side unsplit, same sessions as the 13b `--full-analysis` (cache HIT).

| θ | S unsplit | S p | vs f1/f2 S |
|---|----------:|----:|------------|
| regular s101 (fitted I/M, `g_s=0`) | 0.0027 | 0.63 ✗ | 0.199, p=0 → **collapses** |
| sensory s23 (P→S only, `g_s≈38`, `d_s≈33`) | 0.069 | 0 ✓ | 0.055, p=0 → **survives** |

Regular matches canonical absence unsplit (S n.s., I/M large). Sensory S is not a composition artefact.

### 2026-08-13d — Harris unique-null on the same unsplit test

Full write-up: [retinal_then_joint_fitting.md](retinal_then_joint_fitting.md) 2026-08-13d.
`--harris-unique-null` (other-session prior sequences, unique patterns, 40 extra donor sessions). Unsplit unique pool 100/100.

| θ | contrast-matched S | Harris unique S |
|---|-------------------:|----------------:|
| regular s101 | 0.0027, p=0.63 ✗ | 0.0031, p=0.33 ✗ |
| sensory s23 | 0.069, p=0 ✓ | 0.069, p=0 ✓ |

Harris does not change the unsplit conclusion. Split-conditioned Harris unique-null is under-resolved on f2 (do not use).

### 2026-08-14 — same unsplit test at nrand=2000 / 40 blocks

Full write-up: [retinal_then_joint_fitting.md](retinal_then_joint_fitting.md) 2026-08-14.
Longer sessions fill the f2 unique pool (U=2000, 40/40 kept). Unsplit S is still
regular **0.0003, p=0.63** vs sensory **0.072, p=0**. Split-conditioned regular S
(0.195, p=0) remains the composition artefact. Future Harris / long-session runs
→ **ORCD**; local `session_cache/` wiped after this campaign.

### 2026-08-14e — move-aligned unsplit uses choice strata

Default unsplit no longer analyses M on stim-aligned stim_l/r. **S/I** keep
stim strata at stimOn; **M** uses choice strata at firstMovement. Real-data
`act_block_duringchoice_{l,r}` already did this; the sim `--unsplit-prior`
path now matches. Submit on ORCD (`nrand=1000`, **S curves 0–150 ms**):
`bash scripts/submit_simulate_unsplit_harris_orcd.sh`.

**Do not compare this run’s unsplit M to 2026-08-13c/d / 2026-08-14 unsplit M**
(those were stim-aligned at stimOn; this M is choice strata at movement).

---

## Open / follow-up items

- ~~**S p-values at 150 ms and 80 ms**~~ Done 2026-08-17: regular unsplit S
  is null at t≤80 ms (p=0.64) and only “sig” on the full 150 ms curve;
  sensory S sig in both. See [retinal then joint](retinal_then_joint_fitting.md).
- **Do not compare new unsplit M (choice / movement) to old unsplit M
  (stim-aligned).** Different analysis, not a bug.
- Unsplit S-only canonical (`g_s=g_i_fitted`, `g_i=0`) — not run.
- Unsplit I-sig diagnostic (`g_s=10, d_s=d_i`) — not run.
- **Experiment B — concordance-grouped trajectories:** plot S (and I) trajectories grouped by (1) P-block-L vs P-block-R over all trials and (2) trial-level concordance `(S[0]−S[1])·(P[0]−P[1])` at stim onset, which matches the `g_s` boost logic. This addresses the persistent visual failure of `p_block_s_trajectory` under split conditioning. Not run.
- **Experiment C (optional):** at `d_s=0`, sweep `g_s ∈ {850, 900, 2025}` with and without `gs_outside_adaptation`, unsplit only, to confirm whether significance thresholds change once the selection reversal is removed. Partially superseded by the presence unsplit sweep in [direct sensory prior coupling](direct_sensory_prior_coupling.md).
