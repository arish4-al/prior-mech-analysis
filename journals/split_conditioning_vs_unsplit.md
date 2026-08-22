# Split conditioning vs unsplit prior distance

**Scope:** how the f1/f2 (choice × feedback) conditioning of the `act_block_duringstim` splits creates S prior distance by trial composition, what happens when those splits are removed, and how the choice of null (contrast-matched vs label shuffle) interacts with all of it.

**Status:** decisive result in hand. Absence S prior distance collapses from **0.798 → 0.011 (n.s.)** when f1/f2 conditioning is dropped while stim side is preserved. I and M prior effects survive and grow. The split-conditioned S readout is therefore largely a composition artefact.

Sources: dated entries 2026-06-29 (Goals 1–2, Experiment A), 2026-07-06h (Tables A–C), 2026-08-13c–14e, 2026-08-18 (choice lapse), 2026-08-18b (soft threshold), 2026-08-18c (T+ε), 2026-08-21 (ε sweep), 2026-08-21b (M-aligned lapse).

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

## 2026-08-18 — post-decision choice lapse (ε=0.1)

H3 test: if the split S artefact is S0 selection induced by a near-deterministic
choice map, independent lapse should shrink split S at `g_s=0` without creating
unsplit S at 80 ms.

**Implementation (Option A):** `--choice-lapse ε` in `simulate_recovery.py`. After
`run_model`, with probability ε replace a committed ±1 choice by an independent
fair L/R draw and recompute feedback. Neural S/I/M/RT unchanged. ε=0 is a no-op
(existing session-cache keys). Lapse-on writes to tagged dirs
(`absence_unsplit_lapse0.1`, `goal1/absence_lapse0.1/…`). Smoke:
`python scripts/test_choice_lapse.py`.

Do **not** interpret unsplit **M** under this intervention: M traces still follow
the race, labels do not.

Canonical analysis, seed 123, 40 sessions, nrand=100, contrast-matched, 80 ms S.
Weights `WEIGHTS_REL` (fitted I/M). Cache HIT on the split arms.

| Pooling | Case | lapse | S curve_mean | S p | I | I p | M | M p |
|---------|------|------:|-------------:|----:|--:|----:|--:|----:|
| f1/f2 splits | Phase 4b | 0 | 0.012 | 0.78 | 0.004 | 0.60 | 0.004 | 0.81 |
| f1/f2 splits | Phase 4b | **0.1** | 0.003 | 0.77 | 0.001 | 0.89 | 0.001 | 0.89 |
| f1/f2 splits | Absence | 0 | 0.798 | 0.00 | 0.492 | 0.00 | 2.028 | 0.00 |
| f1/f2 splits | Absence | **0.1** | **0.119** | **0.00** | 0.388 | 0.00 | 1.183 | 0.00 |
| stim_side unsplit | Phase 4b | 0 | 0.003 | 0.64 | 0.003 | 0.17 | 0.004 | 0.20 |
| stim_side unsplit | Phase 4b | **0.1** | 0.001 | 0.64 | 0.001 | 0.12 | 1.613† | 0.00 |
| stim_side unsplit | Absence | 0 | 0.011 | 0.13 | 1.099 | 0.00 | 3.078 | 0.00 |
| stim_side unsplit | Absence | **0.1** | 0.006 | 0.16 | 0.550 | 0.00 | 1.767† | 0.00 |

† Unsplit M is **not** a valid readout here (label ≠ M). Phase 4b unsplit M
blowing up is that mismatch, not a prior effect.

**Outputs** (under `manifold_sim/`):
- unsplit: `unsplit_prior/seed_123/{phase4_no_prior_mod,absence}_unsplit_lapse0.1/`
- split absence: `goal1/absence_lapse0.1/cm_sprior/`
- split Phase 4b: `absence/figs/phase4_no_prior_mod_lapse0.1/`

**Interpretation**

1. Split absence S **shrinks ~7×** (0.798 → 0.119) but **stays significant**.
   Independent 10% lapse weakens the collider; it does not remove it. A larger
   ε (0.2) was not run.
2. Unsplit absence S stays **n.s.** (0.006, p=0.16), same as ε=0. Unsplit I
   remains large (0.55, p=0).
3. Split Phase 4b stays null (S p=0.77), matching the ε=0 regression.
4. So H3 is partly right: making choice a weaker function of S0 reduces the
   split S artefact. Residual split S at ε=0.1 is still I/M-mediated composition
   (RT / remaining S0 selection on the 90% non-lapsed trials). Option B
   (in-kernel stochastic threshold) is 2026-08-18b.

---

## 2026-08-18b — stochastic/soft action threshold (T=0.05)

H3, in-kernel: if the split S artefact is S0 selection from a near-deterministic
race, a sigmoid first-passage should shrink split S at `g_s=0` without creating
unsplit S at 80 ms. Unlike Option A, **choice and M stay consistent** (commit
still follows `sign(action)`); RT can change.

**Implementation (Option B):** `--threshold-temperature T` in
`simulate_recovery.py`. Numpy/numba race: T=0 is the hard first-passage
(`|action| ≥ θ+1e-6`, bit-exact vs previous). T>0 commits with
`P = sigmoid((|action|−θ)/T)` using pre-drawn uniforms `u[tr,k]` from a
per-session seed taken after `create_stimuli` (not stored in the cache
payload). T=0 omits the key so existing session-cache hashes are unchanged.
T>0 writes to tagged dirs (`absence_unsplit_softthr0.05`,
`goal1/absence_softthr0.05/…`). Smoke: `python scripts/test_soft_threshold.py`
(T=0 numpy=numba; T>0 changes RT; backends match; choice follows action).

T=0.05 is a first experimental scale (torch fitting uses 0.01 only for
`continue_prob` gradients; discrete torch choice is still hard). At the
boundary, p=0.5; 0.1 below/above θ is p≈0.12/0.88. Did **not** refit θ.
`--choice-lapse` left at 0.

Canonical analysis, seed 123, 40 sessions, nrand=100, contrast-matched, 80 ms S.
Weights `WEIGHTS_REL` (fitted I/M). Session-cache HIT on the split arms.

| Pooling | Case | T | S curve_mean | S p | I | I p | M | M p |
|---------|------|--:|-------------:|----:|--:|----:|--:|----:|
| f1/f2 splits | Phase 4b | 0 | 0.012 | 0.78 | 0.004 | 0.60 | 0.004 | 0.81 |
| f1/f2 splits | Phase 4b | **0.05** | 0.003 | 0.53 | 0.002 | 0.04 | 0.003 | 0.02 |
| f1/f2 splits | Absence | 0 | 0.798 | 0.00 | 0.492 | 0.00 | 2.028 | 0.00 |
| f1/f2 splits | Absence | **0.05** | **0.143** | **0.00** | 0.090 | 0.00 | 0.368 | 0.00 |
| stim_side unsplit | Phase 4b | 0 | 0.003 | 0.64 | 0.003 | 0.17 | 0.004 | 0.20 |
| stim_side unsplit | Phase 4b | **0.05** | 0.002 | 0.32 | 0.003 | 0.01 | 0.706 | 0.00 |
| stim_side unsplit | Absence | 0 | 0.011 | 0.13 | 1.099 | 0.00 | 3.078 | 0.00 |
| stim_side unsplit | Absence | **0.05** | 0.005 | 0.11 | 0.383 | 0.00 | 0.812 | 0.00 |

**Outputs** (under `manifold_sim/`):
- unsplit: `unsplit_prior/seed_123/{phase4_no_prior_mod,absence}_unsplit_softthr0.05/`
- split absence: `goal1/absence_softthr0.05/cm_sprior/`
- split Phase 4b: `absence/figs/phase4_no_prior_mod_softthr0.05/`

**Interpretation**

1. Split absence S **shrinks ~5.6×** (0.798 → 0.143) but **stays significant**,
   similar to Option A ε=0.1 (0.119, p=0). Early/noisy commits weaken the
   collider; they do not remove it.
2. Unsplit absence S stays **n.s.** (0.005, p=0.11). Unsplit I remains large
   (0.383, p=0) but smaller than T=0 (1.099).
3. Split Phase 4b **S** stays null (p=0.53). I/M p-values dip to 0.04/0.02 with
   tiny curve_means (~0.002–0.003) — nrand=100 noise, not a recovered prior.
4. Unsplit **M is a valid readout** here (labels = M). Absence unsplit M
   shrinks 3.078 → 0.812 but stays p=0. Phase 4b unsplit M **blows up**
   (0.004 → 0.706, p=0): move-aligned choice strata plus a soft boundary
   let commit-margin / RT differ by 80/20 stim composition even with
   `g_*=d_*=0`. That is a B side-effect, not a label/trace mismatch.
5. H3 again partly right: a less deterministic mapping from S0 to choice
   shrinks split S. Residual split S is still I/M-mediated composition.
   T=0.02 was not run.

---

## 2026-08-18c — T=0.05 and ε=0.1 together (split absence only)

Same canonical analysis as 2026-08-18 / 18b, split absence only
(`--run-experiment absence --threshold-temperature 0.05 --choice-lapse 0.1`).
Soft threshold in the race, then post-decision lapse on committed ±1 choices.
Dir tag `absence_lapse0.1_softthr0.05`. Session-cache MISS (mp has both keys).

| Split absence | T | ε | S | S p | I | I p | M | M p |
|---------------|--:|--:|--:|----:|--:|----:|--:|----:|
| neither | 0 | 0 | 0.798 | 0 | 0.492 | 0 | 2.028 | 0 |
| A only | 0 | 0.1 | 0.119 | 0 | 0.388 | 0 | 1.183† | 0 |
| B only | 0.05 | 0 | 0.143 | 0 | 0.090 | 0 | 0.368 | 0 |
| **A+B** | **0.05** | **0.1** | **0.081** | **0** | 0.185 | 0 | 0.635† | 0 |

† M (and to a lesser extent I composition) is label≠trace wherever ε>0.

**Output:** `goal1/absence_lapse0.1_softthr0.05/cm_sprior/`

**Interpretation.** Combined S (0.081) is below either intervention alone, so
they are not fully redundant — extra label noise on top of a soft first-passage
mixes f1/f2 a bit more. They are also not independent
(0.798 × 0.119/0.798 × 0.143/0.798 ≈ 0.021, much smaller than 0.081). Residual
split S is still p=0. Combined M sits between A-only and B-only because lapse
re-inflates the M readout after the in-kernel shrink.

---

## 2026-08-21 — ε sweep at T=0 (split absence)

Goal: push split S to n.s. while keeping I (and, if valid, M) significant.
T=0; `--run-experiment absence --choice-lapse {0.2,0.3,0.5}`. Canonical
seed 123, 40 sessions, nrand=100, 80 ms S. Each ε is its own session-cache
key (MISS). ε=0 and ε=0.1 from 2026-08-18 included for the series.

| ε | S | S p | I | I p | M | M p | S vs null med |
|--:|--:|----:|--:|----:|--:|----:|---|
| 0 | 0.798 | 0 | 0.492 | 0 | 2.028 | 0 | — |
| 0.1 | 0.119 | 0 | 0.388 | 0 | 1.183† | 0 | — |
| 0.2 | 0.157 | 0 | 0.552 | 0 | 1.533† | 0 | 0.157 vs 0.073 |
| 0.3 | 0.111 | 0 | 0.581 | 0 | 1.572† | 0 | 0.111 vs 0.071 |
| **0.5** | **0.063** | **0.01** | **0.591** | **0** | 1.604† | 0 | 0.063 vs 0.034 |

† M invalid (label ≠ M). Do not read M p as a generative prior.

**Outputs:** `goal1/absence_lapse{0.2,0.3,0.5}/cm_sprior/`

**Interpretation**

1. S falls overall (0.798 → 0.063) but is **still significant at ε=0.5**
   (p=0.01). The 0.1→0.2 bump (0.119→0.157) is non-monotone split-membership
   noise; 0.2→0.5 is down.
2. **I stays large and even grows** (0.388 → 0.59). Mixing choice labels
   makes f1/f2 less of a collider on S0 and more like a looser pool, so
   fitted P→I is easier to see. That is the intended absence pattern on I.
3. The S-n.s. / I-sig window is **not reached yet**. Next ε would be ~0.6–0.8
   if we keep pushing this lever; at ε→1 split I should also collapse.
4. M curve_mean **rises** with ε (1.18 → 1.60). That is the mismatch
   artefact, not more prior in M. **Superseded** by 2026-08-21b (M swapped
   to match the lapsed label).

---

## 2026-08-21b — post-decision lapse with M aligned (split absence)

Same ε sweep as 08-21, but lapse now **swaps that trial's M channels** when
the fair L/R draw differs from the race, so `sign(M0−M1)` matches the new
choice. S/I/P and RT unchanged. `choice_lapse_align_m` is on the session-cache
key so old (unaligned) lapse pickles are not reused. Dir tags still
`absence_lapse{ε}` (overwrites 08-18 / 08-21 summaries). Smoke:
`python scripts/test_choice_lapse.py`.

Canonical split absence, T=0, seed 123, 40 sessions, nrand=100, 80 ms S.

| ε | S | S p | I | I p | M | M p |
|--:|--:|----:|--:|----:|--:|----:|
| 0 | 0.798 | 0 | 0.492 | 0 | 2.028 | 0 |
| 0.1 | 0.119 | 0 | 0.388 | 0 | **0.323** | 0 |
| 0.2 | 0.157 | 0 | 0.552 | 0 | **0.367** | 0 |
| 0.3 | 0.111 | 0 | 0.581 | 0 | **0.419** | 0 |
| **0.5** | **0.063** | **0.01** | **0.591** | **0** | **0.530** | **0** |

S/I match 08-21 (labels and S/I traces unchanged). M is now a valid
choice-conditioned readout: 2.028 → 0.32 at ε=0.1 (was 1.18 when misaligned),
then rises slowly with ε like I, all p=0.

**Output:** `goal1/absence_lapse{0.1,0.2,0.3,0.5}/cm_sprior/`

**Interpretation.** Split M is usable. I and M stay significant across the
sweep; S is still p=0.01 at ε=0.5. The intended S-n.s. / I/M-sig window is
not reached. M no longer inflates with ε.

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
- ~~**Choice lapse ε=0.2 / Option B (in-kernel M noise)**~~ Option B at T=0.05
  is 2026-08-18b; T+ε is 2026-08-18c; ε sweep is 2026-08-21; M-aligned lapse
  is 2026-08-21b (ε=0.5 split S p=0.01, I/M p=0, M now valid). T=0.02 not run.
  Option C (sticky ActionKernel) not run.
