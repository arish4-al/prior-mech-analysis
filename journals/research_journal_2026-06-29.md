# Research journal — 2026-06-29

## Standing context (carry-forward from 2026-06-20)

See [prior journal](research_journal_2026-06-20.md) for Experiments 1–7.

### Canonical analysis defaults (mandatory — see `AGENTS.md`)

Since the 2026-06-19 retest ([2026-06-18 journal](research_journal_2026-06-18.md)):

| Setting | Value |
|---------|-------|
| S window | **80 ms** (`S_DURINGSTIM_WINDOW_S`) |
| I/M window | **150 ms** |
| Truncation | **fill-from-next-ITI** (never zero-pad) |
| Phase 4b sanity (split, seed 123) | S curve_mean≈0.012, p≈0.78 |

Agent-facing docs: `AGENTS.md`, `.cursor/rules/prior-distance-analysis.mdc`, `CANONICAL_PRIOR_DISTANCE_ANALYSIS` in `simulate_recovery.py`.

**Established facts (seed 123, contrast-matched shuffle null, α=0.01):**

| Mechanism | Key result |
|-----------|------------|
| S prior distance in **absence** (I/M on, g_s=0) | Large (curve_mean≈0.80), significant — **I/M-mediated**, not direct P→S |
| S-only presence at canonical `g_s=g_i_fitted, d_s=d_i_fitted` | S n.s. (p≈0.15), **I significant** (p=0) via feedforward amplification |
| Offset route (`g_s=1, d_s=48`) | S+I significant, but p_gain likely **offset-decay artefact** (biologically implausible d_s≈2.2× d_i) |
| Gain-only route (`d_s=0`) | S p_gain significant only at extreme g_s: **2025** (inside adaptation) or **~850–900** (outside adaptation, `gs_outside_adaptation`) |
| Removing adaptation gate on g_s | ~2–4× reduction in crossover g_s, **not** the predicted 11× — reversal is RT/selection-structural |
| I-scaled params (`g_s = g_i × |S|/|S0| ≈ 68, d_s=d_i`) | S n.s.; I borderline/sig — **accumulation advantage** by architecture |

**Open problem (visual):** `p_block_s_trajectory` plots show no P-block-L vs P-block-R separation in any condition tested, even when S prior distance is significant.

**Structural insight from 2026-06-20:** f1/f2 split conditioning + RT-driven zero-padding can **invert** late-time S gain_effect (concordant trials zero-pad faster → discordant trials dominate late bins). This is not fixed by moving g_s outside the adaptation gate.

---

## Today's goals

Three related questions that reframes what "significant S prior distance" means in this pipeline — and whether the I/M signals we trust might reflect weak, undetectable S modulation amplified downstream.

### Goal 1 — Confounded significant S without direct prior modulation on S

**Hypothesis:** Significant S prior distance can arise from **selection bias** and **prior effects on I/M** (choice composition, threshold shifts, RT truncation) even when there is **no meaningful direct P→S coupling** — or when g_s/d_s are too weak to produce a detectable sensory signature.

**Evidence already in hand:**
- Absence condition: large S prior distance with g_s=0 (I/M-mediated).
- Phase 4b: residual significant S at c=1.0 on `*_f1` splits only (no g/d at all).
- S-only at canonical params: I significant, S not — same split pipeline.

**Key to-do:** Re-run the same analyses **without splitting into trial types** (no f1/f2 / stim×choice conditioning). This is the clean test flagged in the 2026-06-20 open questions: unconditional or concordance-grouped trajectories, and prior distance aggregated over all `duringstim` trials.

**Success criteria:**
- If S prior distance collapses to null when unsplit → current significant S in split pipeline is largely **composition artefact**.
- If S prior distance persists unsplit with g_s=0 → need a non-selection mechanism (block epoch, constant S0, etc.).

---

### Goal 2 — Is sensory gain modulation biologically plausible in this architecture?

**Hypothesis:** S is too **feedforward / transient** (80 ms window, tau_s=20 ms, adaptation-suppressed g_s) for gain modulation to produce a detectable prior signature at physically reasonable parameter scales. I, as an active integrator, amplifies weak S modulation over 150 ms — so I/M effects are **easier to see** and do not require extreme g_s.

**Parameter thresholds for S p_gain significance (gain-only, d_s=0, seed 123):**

| g_s placement | Min g_s for S p_gain sig | vs g_i_fitted (189.7) |
|---------------|--------------------------|------------------------|
| Inside adaptation (`a`) | **~2025** | ~10.7× |
| Outside adaptation (`gs_outside_adaptation`) | **~850–900** | ~4.5× |

**Revised g_s outside adaptation (Exp 6–7):** Helps offset detection (`p_offset`) and lowers gain crossover ~2–4×, but **does not solve the core problem** — still requires g_s ≫ g_i for a genuine gain profile on S, and S p_mean remains n.s. at i-scaled params.

**Interpretation to test today:** If the only routes to S significance are (a) extreme g_s, (b) implausible d_s offsets, or (c) I/M-mediated selection — then **direct sensory prior gain modulation may not be a viable biological explanation** for block-dependent structure in S under this model and metric. The brain might modulate sensory gain, but this pipeline cannot detect it at fitted scales.

**Key to-do (shared with Goal 1):** Same experiment without trial-type splits — removes the RT/zero-padding reversal that structurally suppresses late-bin S gain.

---

### Goal 3 — Weak S modulation → significant I prior distance (biological plausibility)

**Hypothesis:** **Yes, this is possible and may be the default regime.** Direct P→S coupling can be **too weak for the 80 ms S-distance metric** while still producing **significant I prior distance** via feedforward integration.

**Evidence already in hand (canonical S-only, g_s=189.7, d_s=21.56, g_i=d_i=g_m=d_m=0):**

| pop | curve_mean | p_mean | significant? |
|-----|-----------|--------|--------------|
| S | 0.037 | 0.15 | ✗ |
| I | 0.031 | 0.00 | ✓ |

Same pattern at `g_s=10, d_s=21.56` (I-sig diagnostic suite): S p_mean=0.18, p_gain=0.29; I p_mean=0.00, p_gain=0.00.

**Biological reading:** A small concordance-dependent boost on S0 at stim onset, integrated over the trial by I, can yield a robust prior signal downstream even when S itself looks null. This is **architecturally expected** (I accumulates; S is instantaneous) and **does not require** the large S prior distance seen in absence — that absence signal is likely I/M selection, not proof of strong sensory modulation.

**Question for today:** Is this weak-S / strong-I regime **reasonable** as an account of experimental data (prior effects in integrator-like populations but not in early sensory)? Or does it make the presence/absence comparison uninterpretable because any I/M prior effect could always be "explained away" as weak S feedforward?

---

## Planned experiments

### Experiment A — Unsplit prior distance (priority)

Run S (and I) prior distance on **all duringstim trials**, no f1/f2 / stim×choice×feedback splits.

| condition | g_s | d_s | g_i…g_m | split? |
|-----------|-----|-----|---------|--------|
| Phase 4b | 0 | 0 | 0 | **no** |
| Absence (standard) | 0 | 0 | fitted | **no** |
| S-only canonical | g_i_fitted | d_i_fitted | 0 | **no** |
| S-only I-sig diagnostic | 10 | d_i_fitted | 0 | **no** |

Compare to existing split-conditioned results. If absence S prior distance drops from ~0.80 → near-null unsplit, the standard S readout is largely confound.

### Experiment B — Unsplit / concordance-grouped trajectories

Plot S (and I) trajectories grouped by:
1. P-block-L vs P-block-R (all trials)
2. Trial-level concordance `(S[0]-S[1])*(P[0]-P[1])` at stim onset (matches g_s boost logic)

Addresses the persistent visual failure of `p_block_s_trajectory` under split conditioning.

### Experiment C — (Optional) Re-run gain threshold check unsplit

At `d_s=0`, sweep g_s ∈ {850, 900, 2025} with and without `gs_outside_adaptation`, unsplit only — confirm whether significance thresholds change when selection reversal is removed.

---

## Working hypotheses (to update with results)

1. **Significant S in the split pipeline is often confounded** — driven by I/M prior mod changing which trials enter each split, not by direct sensory prior coupling.
2. **Detectable sensory gain on S requires unphysical g_s** (~5–11× g_i) or implausible d_s offsets; modest biologically plausible coupling produces **I significant, S not**.
3. **Weak S → strong I is the plausible biological regime** for direct P→S coupling; effects seen in I/M but not S in data may reflect downstream amplification, not absence of sensory modulation.
4. **Unsplit analysis is the decisive test** for goals 1–3; most 2026-06-20 sweeps are inconclusive on biological plausibility until split conditioning is removed.

---

## Results

### Experiment A — Unsplit prior distance (seed 123, 40 sessions, nrand=100, α=0.01)

**CLI** (outputs to default `<ONE cache>/manifold_sim/`):

```bash
conda activate iblenv
python simulate_recovery.py --unsplit-prior phase4 absence \
  --seed 123 --n-sessions 40 --nrand 100 --n-jobs 8
```

**Implementation:** `--unsplit-prior {phase4,absence,all}` with `--unsplit-mode`:

| Mode | Splits | Meaning |
|------|--------|---------|
| `stim_side` (default) | `stim_l_unsplit` + `stim_r_unsplit` | No f1/f2; stim side preserved |
| `fully` | `act_block_duringstim_fully_unsplit` | All duringstim trials, L+R mixed (diagnostic) |

Uses canonical fill-next + S=80ms / I/M=150ms.

```bash
# stim-side unsplit (recommended)
python simulate_recovery.py --unsplit-prior phase4 absence --seed 123 ...

# fully unsplit (L+R pooled — S artefact risk)
python simulate_recovery.py --unsplit-prior phase4 absence --unsplit-mode fully --seed 123 ...
```

#### Results comparison (seed 123, fill-next, S=80ms)

| Pooling | Case | S curve_mean | S p | I curve_mean | I p | M curve_mean | M p |
|---------|------|-------------|-----|-------------|-----|-------------|-----|
| **f1/f2 splits** (4) | Phase 4b | 0.012 | 0.78 ✗ | 0.004 | 0.60 ✗ | 0.004 | 0.81 ✗ |
| **f1/f2 splits** (4) | Absence | 0.798 | 0.00 ✓ | — | — | — | — |
| **stim_side unsplit** (2) | Phase 4b | 0.003 | 0.64 ✗ | 0.003 | 0.17 ✗ | 0.004 | 0.20 ✗ |
| **stim_side unsplit** (2) | Absence | 0.011 | 0.13 ✗ | 1.099 | 0.00 ✓ | 3.078 | 0.00 ✓ |
| **fully unsplit** (1) | Phase 4b | **0.297** | **0.00** ✓† | 0.264 | 0.00 ✓† | 0.504 | 0.00 ✓† |
| **fully unsplit** (1) | Absence | **0.404** | **0.00** ✓† | 1.911 | 0.00 ✓ | 4.526 | 0.00 ✓ |

†Spurious for S (and likely I/M at phase4): L+R stim pooled without channel alignment. Same values as invalid first run.

**Output paths:**

| Mode | Phase 4b | Absence |
|------|----------|---------|
| stim_side | `.../phase4_no_prior_mod_unsplit/` | `.../absence_unsplit/` |
| fully | `.../phase4_no_prior_mod_fully_unsplit/` | `.../absence_fully_unsplit/` |

Base: `manifold_sim/unsplit_prior/seed_123/`

### Interpretation (Experiment A)

1. **Three-way comparison:** Removing f1/f2 splits collapses absence **S** (0.80 → 0.011, n.s.) when stim side is preserved. **Fully unsplit** restores large spurious S (0.40, p=0) — artefact from mixing left- and right-stim trials in one S distance (activity on different channels).

2. **Phase 4b fully unsplit** (S=0.30, p=0) is **not** evidence of prior coupling — same spurious signal as first invalid run; split and stim_side unsplit Phase 4b are both null.

3. **Absence fully unsplit** (S=0.40) **overstates** S prior distance vs stim_side unsplit (0.011) — channel-mixing artefact adds ~0.39 to S curve_mean. I/M significant in all unsplit modes (genuine prior effects in integrator/motor).

4. **Goal 1 strongly supported** via stim_side unsplit: split-conditioned S (0.80) is almost entirely f1/f2 composition artefact; I/M effects are real without that filter.

5. **Do not use fully unsplit for S inference** — only as a diagnostic showing why stim-side conditioning is required.

---

---

## Experiment D — Gain-only g_s sweep to find I-sig / S-not-sig threshold (2026-06-29)

### Goal

Find the minimum `g_s` at `d_s=0` (gain-only, all other g/d=0) where **I is significant** but **S is not**, for both g_s inside and outside the adaptation gate.

### Results (seed 123, 40 sessions, nrand=100, α=0.01, contrast-matched null)

#### Inside adaptation (default, `gs_outside_adaptation=False`)

| g_s | S p_mean | S p_gain | I p_mean | I p_gain | I sig? | S sig? |
|-----|----------|----------|----------|----------|--------|--------|
| 10 | 0.69 | 0.69 | 0.57 | 0.57 | ✗ | ✗ |
| 100 | 0.78 | 0.78 | 0.64 | 0.64 | ✗ | ✗ |
| 500 | 0.40 | 0.40 | 0.65 | 0.65 | ✗ | ✗ |
| 1000 | 0.47 | 0.47 | 0.40 | 0.40 | ✗ | ✗ |
| 1500 | 0.06 | 0.06 | 0.05 | 0.05 | ✗ | ✗ |
| **1800** | **0.04** | **0.04** | **0.00** | **0.00** | **✓** | **✗** |
| 2000 | 0.01 | 0.01 | 0.01 | 0.01 | borderline | borderline |
| **2025** | **0.00** | **0.00** | **0.00** | **0.00** | **✓** | **✓** |
| 2200 | 0.00 | 0.00 | 0.00 | 0.00 | ✓ | ✓ |

**I-sig without S-sig: g_s ≈ 1800** (~9.5× g_i_fitted)

#### Outside adaptation (`gs_outside_adaptation=True`)

| g_s | S p_mean | S p_gain | I p_mean | I p_gain | I sig? | S sig? |
|-----|----------|----------|----------|----------|--------|--------|
| 50–200 | 0.73–0.83 | same | 0.62–0.68 | same | ✗ | ✗ |
| 300 | 0.60 | 0.60 | 0.32 | 0.32 | ✗ | ✗ |
| 400 | 0.54 | 0.54 | 0.09 | 0.09 | ✗ | ✗ |
| 500 | 0.33 | 0.33 | 0.01 | 0.01 | borderline | ✗ |
| 600 | 0.07 | 0.07 | 0.01 | 0.01 | borderline | ✗ |
| **700** | **0.08** | **0.08** | **0.00** | **0.00** | **✓** | **✗** |
| **750** | **0.01** | **0.01** | **0.00** | **0.00** | **✓** | **✗** |
| **800** | **0.02** | **0.02** | **0.00** | **0.00** | **✓** | **✗** |
| **850** | **0.01** | **0.01** | **0.00** | **0.00** | **✓** | **✗** |
| **900** | **0.00** | **0.00** | **0.00** | **0.00** | **✓** | **✓** |

**I-sig without S-sig: g_s ≈ 700–850** (~3.7–4.5× g_i_fitted)

### Threshold summary

| g_s placement | I becomes sig | S becomes sig | I-only window | vs g_i_fitted (189.7) |
|---------------|--------------|--------------|---------------|----------------------|
| Inside adaptation | ~1800 | ~2025 | ~225 units | I: ~9.5×, S: ~10.7× |
| Outside adaptation | ~700 | ~900 | ~200 units | I: ~3.7×, S: ~4.7× |

Moving g_s outside adaptation lowers both thresholds by ~2.5–2.6×. The I-only window is narrow (~200 units) in both cases: I and S tend to become significant at nearly the same g_s.

**Key confirmation:** `g_s=10, d_s=0` (inside adaptation) does **NOT** make I significant — I p_mean=0.57, S p_mean=0.69. The I-sig result in Exp 3 (`g_s=10, d_s=21.56`) was **offset-driven** (d_s≠0), not gain-driven.

### Diagnostic plots (I-sig but S not sig)

Full plot suites for the two chosen diagnostic cases:

| case | g_s | d_s | gs_free | S p_mean | S p_gain | I p_mean | I p_gain | output dir |
|------|-----|-----|---------|----------|----------|----------|----------|------------|
| Inside adap, I-only | 1800 | 0 | ✗ | 0.04 ✗ | 0.04 ✗ | 0.00 ✓ | 0.00 ✓ | `s_presence_tune/g_s1800_d_s0/` |
| Outside adap, I-only (**primary**) | **700** | 0 | ✓ | **0.08 ✗** | **0.08 ✗** | **0.00 ✓** | **0.00 ✓** | `s_presence_tune/g_s700_d_s0_gs_free/` |
| Outside adap, I-only (alt) | 800 | 0 | ✓ | 0.02 ✗ | 0.02 ✗ | 0.00 ✓ | 0.00 ✓ | `s_presence_tune/g_s800_d_s0_gs_free/` |

Key figures: `figs/s_shuffle_control.png`, `figs/i_shuffle_control.png` in each output dir.

### Visual observations (g_s=700, outside adaptation — most striking case)

**S (p_mean=0.08, p_gain=0.08):** True curve rises slowly from zero, tracking just above the null band for most of the window but with heavy overlap and noisy late bins. 8 of 100 null shuffles exceed the true curve mean — visually indistinguishable from null for most of the trial.

**I (p_mean=0.00, p_gain=0.00):** True curve diverges from null with an exponential-looking rise starting at ~60 ms, reaching >10× the null mean and ~5× the highest null shuffle at 150 ms. All 100 nulls are clearly below the true curve from ~80 ms onward. The contrast between S (buried in null band) and I (far above all nulls) is visually unmistakable.

**Mechanistic reading:** The same weak `g_s × del_P × S0` gain drive that is undetectable in S's 80 ms window is converted by I's leaky integration into a large, monotonically growing separation over 150 ms. Direct sensory gain modulation is architecturally invisible in S but leaves a robust downstream signature in I.

---

## Implementation checklist

Carried forward from 2026-06-20:

- [x] S-only presence, g_s/d_s sweeps, gain-only tuning, `gs_outside_adaptation`
- [x] Plot suites through `g_s68p0941_d_s21p5585_i_scaled_gs_free`
- [x] **`--unsplit-prior` CLI + Experiment A phase4 + absence (seed 123)**
- [ ] Unsplit S-only canonical (`g_s=g_i_fitted, g_i=0`)
- [ ] Unsplit I-sig diagnostic (`g_s=10, d_s=d_i`)
- [ ] **Unconditional / concordance-grouped trajectory diagnostic** (Experiment B)
- [ ] Re-run full presence with `g_s=g_i_fitted` (separate g_s scale from I/M contribution)
- [ ] `g_s1_d_s48` + `--gs-outside-adaptation` (tuned pair sensitivity)
- [ ] Optionally compare contrast-matched vs unrestricted nulls at borderline cases

---

## Next steps (after today's runs)

1. ~~Run Experiment A phase4 + absence unsplit~~ ✓ (seed 123)
2. Run Experiment A remaining cases (S-only canonical, I-sig diagnostic) unsplit.
3. Experiment B: concordance-grouped trajectories.
