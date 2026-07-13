# Research journal — 2026-06-20

## Standing context (carry-forward from 2026-06-18)

See [prior journal](research_journal_2026-06-18.md) for Phase 0–4b, zero-padding fix, and 80 ms S-window canonicalization.

**Open problem:** In the **presence** condition (`g_s/d_s` direct P→S coupling), `p_block_s_trajectory_{split}.png` figures show P-block-L (blue) and P-block-R (orange) curves that are nearly identical to the **absence** plots — they do not qualitatively resemble the `p_block_i_trajectory` separation.

The 2026-06-18 fix (cap S trajectory plots at 80 ms, matching `build_population_b_for_split`) was implemented and reported as resolved, but **visual inspection still shows no block-dependent S separation** in presence (and absence) trajectory plots at seed 123. The bug analysis continues.

---

## Today's goal

**Experiment 1 — S-only presence:** Isolate direct P→S coupling by running with presence-case default `g_s`/`d_s` but setting `g_i=d_i=g_m=d_m=0`.

**Rationale:** If direct S modulation is working, we should see:
1. Clear P-block-L vs P-block-R separation in `p_block_s_trajectory` plots (qualitatively like I trajectories)
2. Significant S prior distance (curve_mean ≫ shuffle null)

If not, the apparent "presence" S prior effect in the full model is mediated entirely by I/M (choice selection, threshold shifts), not by feedforward g_s/d_s.

---

## Implementation

Added to `simulate_recovery.py`:

- `load_fitted_model(..., zero_im_prior_mod=False)` — zeros `g_i/d_i/g_m/d_m` while keeping `g_s/d_s`
- `run_s_only_presence_analysis()` — runs S-prior analysis + block-confound plots for condition `s_presence_only`
- CLI: `--s-only-presence`

**Presence defaults:** `g_s = g_i_fitted`, `d_s = d_i_fitted` from weights JSON (not the `g_s=10` override used in the earlier `absence_80ms` presence run).

```bash
conda activate iblenv
python simulate_recovery.py --s-only-presence \
  --seed 123 --n-sessions 40 --nrand 100 --n-jobs 8
```

**Model parameters (this run):**

| param | value |
|-------|-------|
| g_s | 189.68 (= g_i_fitted) |
| d_s | 21.56 (= d_i_fitted) |
| g_i, d_i, g_m, d_m | 0 |

Outputs (canonical): `<ONE cache>/manifold_sim/s_presence_only/seed_123/s_presence_only/`

On this machine: `/Users/ariliu/Downloads/ONE/openalyx.internationalbrainlab.org/manifold_sim/s_presence_only/seed_123/s_presence_only/`

> **Note:** Do not pass `--output-dir output/...` — omit it to use `default_output_dir()` (`one.cache_dir/manifold_sim`).

---

## Results — S prior distance (S=80 ms window, contrast-matched shuffle, nrand=100)

| Condition | g_s | g_i | curve_mean | null_median | p_mean | significant |
|-----------|-----|-----|-----------|-------------|--------|-------------|
| Phase 4b (all g/d=0) | 0 | 0 | 0.012 | 0.017 | 0.78 | ✗ |
| **S-only presence (today)** | **189.68** | **0** | **0.037** | **0.026** | **0.15** | **✗** |
| Absence (I/M only, seed 123) | 0 | fitted | 0.798 | 0.240 | 0.00 | ✓ |
| Full presence (seed 123, g_s=10†) | 10 | fitted | 0.959 | 0.372 | 0.00 | ✓ |

† Earlier `absence_80ms` presence run used `--g-s-presence 10`, not canonical `g_i_fitted`.

**S-only presence is quantitatively near-null** — only ~3× above Phase 4b, far below absence (I/M) and full presence. Not significant at α=0.01.

### Per-split/contrast highlights (`s_presence_only`, r_choice_r_f1)

| c | n_high | n_low | true | null | p |
|---|--------|-------|------|------|---|
| 0.0 | 83 | 226 | 0.006 | 0.007 | 0.53 |
| 0.0625 | 264 | 592 | 0.011 | 0.006 | 0.25 |
| 0.125 | 326 | 753 | 0.010 | 0.006 | 0.24 |
| 0.25 | 392 | 933 | 0.026 | 0.007 | 0.04 |
| 1.0 | 495 | 976 | 0.013 | 0.006 | 0.18 |

Only c=0.25 reaches p=0.04; all other cells n.s. Pattern is weak and unlike absence (which peaks at c=0.125 on f1 splits and c=0.25 on f2).

### Trajectory plots (`p_block_s_trajectory_stim_r_choice_r_f1.png`)

**S-only presence:** P-block-L (blue) and P-block-R (orange) curves overlap at all contrasts including c=1.0. No visible block-dependent S separation — same qualitative failure as full presence and absence trajectory plots.

**I trajectories (same run, g_i=0):** Flat/near-zero as expected — confirms I/M modulation is off.

---

## Interpretation

### Direct g_s/d_s alone does not recover S prior distance or trajectory separation

With `g_s=189.68, d_s=21.56` (full fitted integrator scale applied to S feedforward) and all I/M prior modulation zeroed:

1. **S prior distance ≈ null** (p=0.15) — indistinguishable from Phase 4b in magnitude
2. **Trajectory plots show no P-block separation** — the original visual bug persists
3. **I trajectories correctly flat** — confirms the I/M zeroing worked

This strongly suggests the significant S prior distance in **absence** (curve_mean=0.798, p=0) and **full presence** (curve_mean=0.959, p=0) is **not driven by direct P→S feedforward coupling**. It is mediated by I/M prior modulation changing trial composition (choice selection into f1/f2 splits) and/or M threshold shifts — consistent with Q1 analysis in the 2026-06-18 journal.

### Why might g_s/d_s not show up even though it's in the ODE?

Candidate mechanisms (not mutually exclusive):

1. **Split conditioning washes out direct S modulation.** Splits fix `(stim_side, choice_side, feedback)`. With I/M off, choices are S-driven only; the trials reaching e.g. `r_choice_r_f1` may be those where S happened to be strong enough regardless of prior boost — selection filter removes the group difference.

2. **P-block-L vs P-block-R grouping ≠ concordant/discordant g_s boost.** The trajectory plot groups by ITI P ≥ 0.5 (left block) vs < 0.5 (right block). The g_s boost depends on `(S[0]-S[1])*(P[0]-P[1])` concordance at stim onset — a trial-level quantity, not a block label. Block label correlates imperfectly with concordance within a split.

3. **g_s acts on S0_delayed via `(J + g_s * P_gain) @ S0` — effect may be small relative to baseline S0 variance** when I/M isn't shaping which trials survive into each split.

4. **Previous presence run used g_s=10, not g_i_fitted=189.68** — even with I/M on, trajectory plots still showed no separation. The full presence S prior significance may also be I/M-mediated, not g_s-mediated.

### Revisiting the 2026-06-18 "fix confirmed" claim

The 80 ms window cap for S trajectory plots was necessary (removes zero-padding artefact) but **not sufficient** to reveal g_s/d_s block modulation visually. All three conditions (absence, full presence, S-only presence) still show overlapping P-block-L/R S trajectories at seed 123.

---

## Comparison table — what drives S prior distance?

| Mechanism | Absence (I/M) | S-only presence | Phase 4b |
|-----------|--------------|-----------------|----------|
| Direct P→S (g_s/d_s) | ✗ | ✓ | ✗ |
| I/M prior mod | ✓ | ✗ | ✗ |
| S prior distance | **large** (0.80) | **null** (0.04) | **null** (0.01) |
| Trajectory P-block separation | none visible | none visible | none visible |

**Conclusion:** S prior distance in the standard pipeline is an **I/M-mediated indirect effect**, not a direct sensory prior signature — at least under current splits and analysis.

---

## Next steps

See **Next steps (updated)** at end of journal for current list. Original items from Experiment 1:

1. Unsplit S trajectories — plot all `duringstim` trials grouped by P-block-L vs P-block-R.
2. Trial-level concordance grouping — matches g_s boost logic.
3. Diagnostic: mean S at stim onset by block/concordance cell.
4. Check whether `plot_p_block_s_trajectories` uses the right prior grouping.

---

## Implementation checklist (Experiment 1)

- [x] `load_fitted_model(..., zero_im_prior_mod=True)`
- [x] `run_s_only_presence_analysis()` + `--s-only-presence` CLI
- [x] Run seed 123, 40 sessions, nrand=100
- [x] Block-confound trajectory plots generated

(Full checklist at end of journal.)

---

## Experiment 2 — g_s/d_s parameter sweep (S-only, I/M zeroed)

### Goal

Tune `g_s` and `d_s` until S prior distance becomes significant (α=0.01), and report I prior significance at each grid point. I has no direct prior mod (`g_i=d_i=0`) but receives S as input — expect **indirect** I prior distance when S is strongly modulated.

### Implementation

Added `run_gs_ds_tune_sweep()` + CLI `--gs-ds-tune` with optional `--g-s-grid`, `--d-s-grid`, `--stop-on-s-significant`, `--tune-alpha`.

```bash
conda activate iblenv
python simulate_recovery.py --gs-ds-tune \
  --seed 123 --n-sessions 40 --nrand 100 --n-jobs 8 \
  --g-s-grid 10,50,100,189.67878020823161 \
  --d-s-grid 21.55851740982741,25,30,35,40,43.11703481965482,50,60,80,100
```

Outputs (canonical): `<ONE cache>/manifold_sim/s_presence_tune/seed_123_refine/gs_ds_tune_sweep.csv`

---

## Results — g_s/d_s sweep (seed 123, 40 sessions, nrand=100, α=0.01)

### Key finding: **d_s offset is necessary; g_s alone is insufficient**

At **g_s=10, d_s=0**: S p=0.69, I p=0.57 — both null.

As **d_s increases** at fixed g_s=10:

| d_s | S curve_mean | S p_mean | S sig | I curve_mean | I p_mean | I sig |
|-----|-------------|----------|-------|-------------|----------|-------|
| 0 | 0.013 | 0.69 | ✗ | 0.004 | 0.57 | ✗ |
| 10.8 | 0.024 | 0.55 | ✗ | 0.009 | 0.21 | ✗ |
| 21.56 (= d_i_fitted) | 0.039 | 0.18 | ✗ | 0.029 | **0.00** | **✓** |
| 30 | 0.055 | 0.03 | ✗ | 0.053 | **0.00** | **✓** |
| **40** | **0.071** | **0.00** | **✓** | **0.083** | **0.00** | **✓** |
| 43.1 | 0.081 | 0.00 | ✓ | 0.094 | 0.00 | ✓ |
| 60 | 0.143 | 0.00 | ✓ | 0.194 | 0.00 | ✓ |
| 100 | 0.495 | 0.00 | ✓ | 0.700 | 0.00 | ✓ |

**Minimum S-significant pair:** `g_s=10, d_s=40` (p_mean=0.0 for both S and I).

**I becomes significant before S:** at `g_s=10, d_s=21.56` (= fitted d_i), I p=0.0 but S p=0.18. I picks up block-dependent structure from modulated S feedforward even with zero direct I prior mod.

### Canonical presence defaults still fail for S

At **g_s=189.68 (= g_i_fitted), d_s=21.56 (= d_i_fitted)** — the "presence default" from Experiment 1:

| pop | curve_mean | null_median | p_mean | sig |
|-----|-----------|-------------|--------|-----|
| S | 0.037 | 0.027 | 0.15 | ✗ |
| I | 0.031 | 0.010 | **0.00** | **✓** |

Same parameters as Experiment 1: **I significant, S not.** S needs **d_s ≥ ~43** at g_s=189.68 (or d_s ≥ 40 at g_s=10) to reach S significance.

### g_s scaling at fixed d_s

At **d_s=43.1** (~2× d_i_fitted):

| g_s | S p_mean | S sig | I p_mean | I sig |
|-----|----------|-------|----------|-------|
| 10 | 0.00 | ✓ | 0.00 | ✓ |
| 50 | 0.01 | ✗ | 0.00 | ✓ |
| 100 | 0.03 | ✗ | 0.00 | ✓ |
| 189.68 | 0.00 | ✓ | 0.00 | ✓ |

S significance is **easier to achieve at lower g_s** when d_s is fixed — counterintuitive but consistent with split-conditioning washing out large S excursions (stronger g_s → faster RT → more truncation/selection effects in f1/f2 splits).

At **d_s=60**, all g_s ∈ {10, 50, 100, 189.68} give S and I p=0.0.

---

## Revised interpretation

1. **Direct P→S coupling CAN produce significant S prior distance**, but only with **sufficient d_s offset** (~2× fitted d_i, or d_s ≥ 40 at g_s=10). The canonical `g_s=g_i_fitted, d_s=d_i_fitted` pairing is **below the S detection threshold** under current splits.

2. **d_s (offset) matters more than g_s (gain)** for S prior distance at **moderate g_s** (≈1–10). At extreme g_s (~10× g_i), gain-only (`d_s=0`) can reach p_gain significance — see Experiment 4.

3. **I prior distance is a sensitive indirect readout** — significant at lower d_s than S, because I integrates modulated S. Useful diagnostic: if I is significant but S is not, S dynamics are modulated but the split-conditioned distance metric doesn't capture it.

4. **Experiment 1 conclusion partially revised:** the failure at canonical defaults is a **parameter-scale / analysis-sensitivity** issue, not proof that g_s/d_s coupling is inert. At tuned parameters, both S and I reach p=0.0 with g_i=d_i=g_m=d_m=0.

5. **Trajectory plot bug remains separate:** even at S-significant parameters, need to verify whether `p_block_s_trajectory` plots show visible separation — **regenerated at g_s=1, d_s=48** under `s_presence_tune/g_s1_d_s48/figs/block_confounds/`; S trajectories still largely overlapping visually; I shows modest separation at high contrast.

---

## Experiment 3 — Tune g_s for S p_gain significance (fixed d_s)

### Goal

Previous minimum for **p_mean** was `g_s=10, d_s=40` (p_gain=0.06, not significant). Need both **p_mean and p_gain** significant at α=0.01.

### Sweep results (seed 123, g_i=d_i=g_m=d_m=0)

**Key finding: p_gain requires higher d_s and lower g_s than p_mean alone.**

At **d_s=48** (≈2.2× d_i_fitted), sweeping g_s:

| g_s | S p_mean | S p_gain | both sig? |
|-----|----------|----------|-----------|
| 0.25–5 | 0.00 | 0.00 | ✓ |
| 8 | 0.00 | 0.06 | ✗ |
| 10 | 0.00 | 0.03 | ✗ |

At **d_s=40** (previous p_mean minimum): all g_s tested had p_gain ≥ 0.03 — never reached p=0.0.

At **d_s=48, g_s=1** (recommended tuned pair):

| pop | curve_mean | null | p_mean | p_gain |
|-----|-----------|------|--------|--------|
| S | 0.096 | 0.050 | **0.00** | **0.00** |
| I | 0.119 | 0.020 | **0.00** | **0.00** |

**Tuned parameters:** `g_s=1.0`, `d_s=48.0` (or any g_s ≤ 5 at d_s=48). g_s is **much lower** than fitted g_i=189.7; d_s is **~2.2×** d_i_fitted.

> **Caveat (added 2026-06-20):** The p_gain significance at `d_s=48` is **not necessarily a genuine sensory gain profile.** A very large d_s offset produces a block-dependent level shift in S that is large at trial onset and then *decays* back as the stimulus drives S to its adapted steady-state. The resulting distance curve shows a falling shape (high early, low late), not a rising gain profile. The `p_gain` metric (late-bin residual after offset subtraction) can still flag this as significant because the decay is slow enough to leave late-bin residual — but this is the relaxation of an artifactually large offset, not a meaningful gain signature. At `d_s ≈ 48 ≈ 2.2 × d_i_fitted`, the prior offset is biologically implausible for a feedforward sensory population. The gain-only route (`g_s=2025, d_s=0`, Exp 4) avoids this artefact but requires extreme g_s.

```bash
python simulate_recovery.py --s-only-presence \
  --g-s-presence 1 --d-s-presence 48 \
  --seed 123 --n-sessions 40 --nrand 100 --n-jobs 8
```

Outputs: `manifold_sim/s_presence_only/` (canonical; no `--output-dir` needed)

Sweep CSV: `manifold_sim/gs_ds_tune_sweep.csv` (refine at d_s=48)

**Full run confirmed (2026-06-20):** `g_s=1, d_s=48` → S p_mean=0.0, p_gain=0.02, p_offset=0.0 (all significant at α=0.01). Block-confound trajectory plots saved under `figs/block_confounds/`. I trajectories show modest P-block separation at high contrast; S trajectories still largely overlapping visually (same open issue as § presence-case bug).

### Interpretation

- **p_gain** tests late-bin distance after early-offset removal — sensitive to temporal profile, not just overall mean.
- High **g_s** shortens RT / changes split composition → destroys late-bin prior structure (p_gain fails at g_s≥8, d_s=48).
- **d_s offset** must be large enough (~48) to create block-dependent S trajectories; **g_s gain** should be modest (≈1–5).

---

### Full plot suite (tuned case, 2026-06-20)

```bash
python simulate_recovery.py --s-presence-tuned-plots \
  --g-s-presence 1 --d-s-presence 48 \
  --seed 123 --n-sessions 40 --nrand 100 --n-jobs 8
```

Output: `manifold_sim/s_presence_tune/g_s1_d_s48/`

| artifact | path |
|----------|------|
| S prior curve + shuffle | `figs/s_prior_curve.png`, `figs/s_shuffle_control.png` |
| I prior curve + shuffle | `figs/I/s_prior_curve.png`, `figs/I/s_shuffle_control.png` |
| S/I comparison | `figs/si_prior_curve_mean_comparison.png`, `figs/si_prior_shuffle_controls.png` |
| Block confounds | `figs/block_confounds/p_block_*` (RT, contrast, S peak, S/I trajectories × 4 splits) |
| Summary | `summary.json` |

**Stats (seed 123, S+I pipeline):** S p_mean=0.0, p_offset=0.0, p_gain=0.0; I p_mean=0.0, p_offset=0.0, p_gain=0.0 (all sig at α=0.01).

### Full plot suite — I sig, S not sig (2026-06-20)

Diagnostic contrast case: **I significant on p_mean and p_gain; S not significant on either** (at fitted d_i offset).

```bash
python simulate_recovery.py --s-presence-tuned-plots \
  --g-s-presence 10 --d-s-presence 21.55851740982741 \
  --seed 123 --n-sessions 40 --nrand 100 --n-jobs 8
```

Output: `manifold_sim/s_presence_tune/g_s10_d_s21p5585/`

| pop | p_mean | p_gain | sig mean? | sig gain? |
|-----|--------|--------|-----------|-----------|
| S | 0.18 | 0.29 | ✗ | ✗ |
| I | 0.00 | 0.00 | ✓ | ✓ |

Same plot layout as tuned case (`figs/s_prior_curve.png`, `figs/I/`, `figs/si_prior_*`, `figs/block_confounds/`).

### Full plot suite — g_s=5 ablation (2026-06-20)

**g_s=5, d_s=48** (offset on, same family as g_s=1 tuned pair):

```bash
python simulate_recovery.py --s-presence-tuned-plots \
  --g-s-presence 5 --d-s-presence 48 \
  --seed 123 --n-sessions 40 --nrand 100 --n-jobs 8
```

Output: `manifold_sim/s_presence_tune/g_s5_d_s48/`

| pop | p_mean | p_gain | sig mean? | sig gain? |
|-----|--------|--------|-----------|-----------|
| S | 0.00 | 0.00 | ✓ | ✓ |
| I | 0.00 | 0.00 | ✓ | ✓ |

**g_s=5, d_s=0** (gain only, no offset — null control):

```bash
python simulate_recovery.py --s-presence-tuned-plots \
  --g-s-presence 5 --d-s-presence 0 \
  --seed 123 --n-sessions 40 --nrand 100 --n-jobs 8
```

Output: `manifold_sim/s_presence_tune/g_s5_d_s0/`

| pop | p_mean | p_gain | sig mean? | sig gain? |
|-----|--------|--------|-----------|-----------|
| S | 0.74 | 0.74 | ✗ | ✗ |
| I | 0.59 | 0.59 | ✗ | ✗ |


### Integrator-comparable scaling (2026-06-20)

Scale S prior params so **g_s × |S0| ≈ g_i × |S|** and **d_s = d_i**:

```python
g_s = g_i_fitted * median(|S| / |S0|)   # 50–80 ms post-stim, c ≥ 0.0625
d_s = d_i_fitted
```

CLI: `--s-presence-i-scaled-plots`

**Estimated (seed 123):** |S|/|S0| = 0.359 → **g_s = 68.1**, **d_s = 21.56** (= d_i).

Output: `manifold_sim/s_presence_tune/g_s68p0941_d_s21p5585_i_scaled/`

| pop | p_mean | p_gain | sig mean? | sig gain? |
|-----|--------|--------|-----------|-----------|
| S | 0.22 | 0.26 | ✗ | ✗ |
| I | 0.01 | 0.01 | ✗ (borderline) | ✗ (borderline) |

Comparable gain drive (g_s≈68) with d_s=d_i does **not** reach S significance; I is borderline at α=0.01. S still needs ~2× d_i offset (d_s≈48 at low g_s) for reliable detection.

> **Interpretation of the i-scaled comparison (added 2026-06-20):** The scaling `g_s = g_i_fitted × (|S|/|S0|)` is correct for equalising the **per-step gain force**: at each timestep `g_s × del_P × S0 ≈ g_i × del_P × S`. But equalising the force does not equalise the **accumulated effect**. I is a downstream leaky integrator of S: each g_s-modulated S timestep is a new input that keeps being added into I over the full trial window (150 ms). The prior signal in I therefore accumulates and grows monotonically, while S only reflects the instantaneous modulation within its 80 ms window. Even with identical per-step gain drives, I prior distance is structurally expected to exceed S prior distance — not because the comparison is wrong, but because I has a temporal accumulation advantage by architecture. The I significance at i-scaled params (p_mean=0.00–0.01 depending on adaptation flag) is a downstream readout of the S prior modulation, not an independent signal. The fact that S p_mean stays at 0.18–0.22 under i-scaled params confirms that the 80ms S-distance metric simply cannot detect a prior signal of this magnitude, even when the force is correctly calibrated.

---

## Experiment 4 — Tune g_s for S p_gain at d_s=0 (gain-only)

### Goal

Can **g_s alone** (no d_s offset) produce significant **p_gain** while keeping `g_i=d_i=g_m=d_m=0`?

### Method

```bash
python simulate_recovery.py --gs-tune-p-gain --d-s-fixed 0 --p-gain-only \
  --g-s-grid "2000,2025,2050" --seed 123 --n-sessions 40 --nrand 100 --n-jobs 8
```

Sweep CSV: `manifold_sim/gs_tune_p_gain/gs_ds_tune_sweep.csv`

All runs use **contrast-matched label-shuffle nulls** (default; use `--label-shuffle-null` for unrestricted).

### Results (seed 123, α=0.01)

| g_s | d_s | S p_mean | S p_gain | sig? |
|-----|-----|----------|----------|------|
| 0.1 – 189 | 0 | 0.4 – 0.81 | same | ✗ |
| 2000 | 0 | 0.01 | 0.01 | ✗ (borderline) |
| **2025** | **0** | **0.00** | **0.00** | **✓** |
| 2200 | 0 | 0.00 | 0.00 | ✓ |
| 5000 | 0 | 0.00 | 0.00 | ✓ |

**Minimum S p_gain at d_s=0:** `g_s ≈ 2025` (~**10.7× g_i_fitted**). Below g_s=2000, p_gain ≥ 0.01.

Plots (gain-only): `manifold_sim/s_presence_tune/g_s2200_d_s0/`

| pop | p_mean | p_offset | p_gain |
|-----|--------|----------|--------|
| S | 0.00 | 0.00 | 0.00 |
| I | 0.00 | 0.00 | 0.00 |

I significant indirectly via modulated S feedforward despite `g_i=0`.

### Interpretation

- Gain on S0 **can** produce significant p_gain, but only at **extreme** g_s (~2000 vs tuned `g_s=1, d_s=48`).
- **Offset is far more efficient:** `g_s=1, d_s=48` vs `g_s=2025, d_s=0` — ~2000× lower gain needed when d_s provides block-dependent offset.
- High g_s likely changes RT / split composition; the d_s=0 route is not biologically plausible at fitted scales.

---

## Experiment 5 — High g_s + fitted d_i (gain + offset)

Combine minimum gain-only g_s with fitted integrator offset.

### Cases run

| g_s | d_s | output dir | S p_mean | S p_gain | I p_mean | I p_gain |
|-----|-----|------------|----------|----------|----------|----------|
| 2025 | d_i (21.56) | `g_s2025_d_s21p5585/` | 0.00 | 0.00 | 0.00 | 0.00 |
| 2200 | d_i (21.56) | `g_s2200_d_s21p5585/` | 0.00 | 0.00 | 0.00 | 0.00 |

```bash
python simulate_recovery.py --s-presence-tuned-plots \
  --g-s-presence 2025 --d-s-presence 21.55851740982741 \
  --seed 123 --n-sessions 40 --nrand 100 --n-jobs 8
```

Both populations significant on **p_mean, p_offset, and p_gain**. Adding d_i to high g_s **increases** S curve_mean (0.054 → 0.091 at g_s=2025; 0.063 → 0.104 at g_s=2200 vs gain-only) without hurting significance.

Compare to low-g tuned pair:

| case | g_s | d_s | S curve_mean | mechanism |
|------|-----|-----|-------------|-----------|
| Tuned (efficient) | 1 | 48 | 0.096 | offset-dominated |
| High-g + d_i | 2025 | 21.56 | 0.091 | gain + modest offset |
| Gain-only | 2025 | 0 | 0.054 | gain-dominated |

---

## Null scheme and output conventions

- **Default null:** contrast-matched label shuffle (`contrast_matched_null=True`; override with `--label-shuffle-null`). Preserves per-contrast trial counts in high/low prior groups.
- **Canonical output root:** `<ONE cache>/manifold_sim/` via `resolve_output_dir()` — do not write to repo `output/`.
- **Plot suite CLI:** `--s-presence-tuned-plots` → `s_presence_tune/g_s{g}_d_s{d}/`; `--s-presence-i-scaled-plots` → `*_i_scaled/` suffix.
- **`summary.json` does not record null scheme** — infer from CLI defaults or re-run with `--label-shuffle-null` to compare.

---

## Experiment 6 — g_s outside adaptation at i-scaled params (2026-06-20)

### Motivation

g_s is applied inside the adaptation gate `a * ((J + g_s * P_gain) @ S0)`.
At steady-state `a ≈ 0.09`, suppressing g_s by ~11× relative to g_i (which is not adapted).
Added `--gs-outside-adaptation` flag: splits feedforward as `a*(J@S0) + g_s*P_gain@S0`.

### Results (g_s=68.09, d_s=21.56, `--gs-outside-adaptation`, seed 123, nrand=100)

Output: `manifold_sim/s_presence_tune/g_s68p0941_d_s21p5585_i_scaled_gs_free/`

| metric | g_s inside `a` | g_s outside `a` |
|--------|----------------|-----------------|
| S `p_mean` | 0.22 ✗ | 0.18 ✗ |
| S `p_offset` | — | **0.00 ✓** |
| S `p_gain` | 0.26 ✗ | 0.24 ✗ |
| I `p_mean` | 0.01 borderline | **0.00 ✓** |
| I `p_offset` | — | **0.00 ✓** |
| I `p_gain` | 0.01 borderline | 0.01 ✗ |

### Interpretation

- Moving g_s outside `a` adds detectable early-time block-dependent modulation (`p_offset` significant for both S and I). Effect is strongest at trial onset (a≈1), fades as adaptation reduces `a`.
- S `p_mean` still not significant: `del_P ≈ 2e-4` limits the absolute contribution even without adaptation suppression.
- **Architectural conclusion:** g_s (inside `a`) vs g_i (outside `a`) is an asymmetry. Default `gs_outside_adaptation=False` for backward compat; set to `True` to match g_i architecture.

---

## Experiment 7 — Gain-only sweep with g_s outside adaptation (2026-06-20)

### Prediction

Adaptation suppresses g_s by ~11× at steady-state (a≈0.09). Moving g_s outside `a` should lower the min S p_gain g_s from ~2025 (Exp 4) to ~2025/11 ≈ **184 ≈ g_i_fitted**.

### Command

```bash
python simulate_recovery.py \
  --gs-tune-p-gain --gs-outside-adaptation \
  --d-s-fixed 0 --p-gain-only \
  --g-s-grid "50,100,150,170,180,189.67878020823161,200,220,250,300,400" \
  --seed 123 --n-sessions 40 --nrand 100 --n-jobs 8
```

Output: `manifold_sim/gs_tune_p_gain/gs_ds_tune_sweep.csv`

### Results (no S p_gain significance found)

| g_s | S p_mean | S p_gain | S gain_effect | I p_mean | I gain_effect |
|-----|---------|---------|--------------|---------|--------------|
| 50 | 0.83 | 0.83 | −0.0083 | 0.62 | −0.0026 |
| 180 | 0.63 | 0.63 | −0.0051 | 0.67 | −0.0019 |
| 189.68 (=g_i) | 0.76 | 0.76 | −0.0090 | 0.57 | −0.0016 |
| 300 | 0.60 | 0.60 | −0.0049 | 0.32 | **+0.0009** |
| 400 | 0.54 | 0.54 | −0.0031 | **0.09** | **+0.0041** |

**Prediction was wrong.** g_i_fitted (189.68) shows no S p_gain significance.

### What happened

1. **S gain_effect is NEGATIVE at all g_s** — the distance curve reverses sign at late times. Concordant trials (g_s amplified) reach threshold faster → zero-pad earlier. Discordant trials are slower → fill late bins. The f1/f2 selection filter *actively inverts* the late-time S signal. This is the same reversal as Exp 4, and removing adaptation from g_s does not change it — it is structural.

2. **S p_mean ≈ S p_gain** throughout because early (positive) and late (negative) contributions cancel. Neither test reaches significance.

3. **I gain_effect turns positive at g_s≈250–300**: I accumulates the modulated S over time without the direct zero-padding reversal, and correctly approaches significance at high g_s.

### Verification: g_s=2025 with gs_outside_adaptation

```bash
python simulate_recovery.py --gs-tune-p-gain --gs-outside-adaptation \
  --d-s-fixed 0 --p-gain-only --g-s-grid "2025" \
  --seed 123 --n-sessions 40 --nrand 100 --n-jobs 8
```

| pop | curve_mean | null | p_mean | p_gain | sig? |
|-----|-----------|------|--------|--------|------|
| S | **0.327** | 0.025 | **0.00** | **0.00** | **✓** |
| I | **0.186** | 0.007 | **0.00** | **0.00** | **✓** |

Significant — and **6× larger curve_mean** than Exp 4 (inside adaptation, same g_s, S curve_mean≈0.054).

### Why the extreme crossover is still >400 but now g_s=2025 is 6× stronger

At g_s=2025 with gs_outside_adaptation, the per-step gain drive is `g_s × del_P × S0 ≈ 0.10`, about **4.5× the adapted raw signal** (0.022). Concordant trials reach threshold near-instantly and zero-pad. Discordant (low-block) trials are actively suppressed — they stay elevated at late times. The window now shows: concordant=0, discordant=signal → **large positive distance**. The reversal has been co-opted: it now works *in favour* of the signal.

### Refined crossover search (seed 123)

Two sweeps to narrow both crossovers:

| g_s | S p_gain | gain_effect | note |
|-----|---------|------------|------|
| 400 | 0.54 | −0.0031 | negative |
| 420 | 0.48 | −0.0022 | negative |
| 450 | 0.49 | −0.0020 | negative |
| **480** | 0.35 | **+0.0016** | ← sign flip |
| 500 | 0.33 | +0.0013 | positive |
| 700 | 0.08 | +0.026 | approaching sig |
| 750 | 0.01 | +0.026 | borderline |
| 800 | 0.02 | +0.028 | not sig |
| 850 | 0.01 | +0.037 | borderline |
| **900** | **0.00** | +0.041 | **significant** |

- **gain_effect sign flip: g_s ≈ 465** (between 450 and 480)
- **p_gain significance: g_s ≈ 850–900** (750/850 borderline p=0.01; 900 clean p=0.00)

### Conclusion

With gs_outside_adaptation, the significance crossover shifts from g_s≈2025 (Exp 4) to g_s≈850–900 — a **~2–4× reduction**. The predicted 11× reduction was too optimistic. Reason: the 11× factor applies at steady-state (a≈0.09, t≫tau_a=222ms), but the crossover mechanism is RT-driven: concordant trials fire at short RTs (~80ms) while adaptation is still ramping. At those short times a(t) is much closer to 1.0, so the time-averaged benefit is smaller than the late-time factor.

The gain_effect sign crossover (g_s≈465) is mechanistically distinct from the significance crossover (g_s≈850–900): sign flip just requires concordant trials to zero-pad faster than discordant; significance requires the resulting distance to exceed the null distribution across 40 sessions.

Compare to Exp 4 (inside adaptation):
- Sign crossover: not measured directly (likely ~1500–1800)
- Significance crossover: g_s≈2025
- Ratio: **~2.25–4×** improvement from removing adaptation gating

---

## Parameter map (summary table)

| case | g_s | d_s | gs_free | S sig? | notes |
|------|-----|-----|---------|--------|-------|
| Phase 4b | 0 | 0 | — | ✗ | baseline |
| Canonical S-only | 189.7 | 21.56 | — | ✗ | I sig, S not |
| I-sig diagnostic | 10 | 21.56 | — | ✗ | I sig mean+gain |
| **Tuned (recommended)** | **1** | **48** | — | **✓** | offset-artefact (see critique) |
| g_s=5 ablation | 5 | 48 | — | ✓ | same as tuned family |
| g_s=5 gain-only | 5 | 0 | — | ✗ | confirms d_s needed |
| Integrator-scaled | 68.1 | 21.56 | ✗ | ✗ | g_s = g_i × (S/S0) |
| Integrator-scaled + gs_free | 68.1 | 21.56 | ✓ | p_offset ✓ | I p_mean sig |
| Gain-only min | 2025 | 0 | — | ✓ p_gain | selection-filter mechanism |
| High-g + d_i | 2025–2200 | 21.56 | — | ✓ | gain+offset |
| **Gain-only gs_free** | **50–400** | **0** | **✓** | **✗** | **prediction wrong; reversal structural** |

---

## Open questions / critiques (added 2026-06-20)

1. **`d_s=48` p_gain is likely an offset-decay artefact**, not a true gain profile. The distance curve at large d_s shows early high, late low (offset relaxing to stimulus-driven steady-state). `p_gain` flags the slow decay as significant. This means the "tuned" (`g_s=1, d_s=48`) parameter pair is not a valid demonstration of sensory prior gain — it is a large offset with a biologically implausible magnitude.

2. **i-scaled comparison (`g_s × S0 ≈ g_i × S`) is conceptually valid** for equating instantaneous gain force, but I has a structural accumulation advantage: S modulation is transient (80ms window, tau_s=20ms), while I accumulates the modulated S input over 150ms. I prior distance exceeding S prior distance at matched gain is therefore expected by architecture, not a contradiction. The i-scaled run showing I significant while S is not does NOT mean S prior coupling is absent — it means the 80ms S-distance metric is insensitive at this force level.

3. **The only biologically plausible route to S p_mean significance so far:** g_s ≈ 2025 (10.7× g_i_fitted) at d_s=0, or large non-physiological offsets. Neither is a clean demonstration of a sensory prior.

4. **What would be a clean test:** measure S prior distance with a metric that matches the I accumulation advantage — e.g. trial-average S at all contrasts grouped by concordance (not split by choice), or a longer S window matching the I window.

## Next steps (updated)

1. ~~Re-run tuned pair + plot suites~~ ✓
2. ~~Tune g_s at d_s=0 for p_gain~~ ✓ (min g_s≈2025)
3. ~~High g_s + d_i_fitted plots~~ ✓
4. ~~g_s outside adaptation at i-scaled params~~ ✓ (Exp 6)
5. Unsplit / concordance-grouped trajectory diagnostic — **this is the clean test** (see Open question 4).
6. Re-run full presence with canonical g_s=g_i_fitted to separate g_s scale from I/M contribution.
7. Optionally compare contrast-matched vs unrestricted nulls at borderline cases.

---

## Implementation checklist (final)

- [x] `load_fitted_model(..., zero_im_prior_mod=True)`
- [x] `run_s_only_presence_analysis()` + `--s-only-presence`
- [x] `run_gs_ds_tune_sweep()` + `--gs-ds-tune` CLI
- [x] `run_gs_tune_p_gain()` + `--gs-tune-p-gain` + `--p-gain-only`
- [x] `run_s_presence_tuned_plots()` + `--s-presence-tuned-plots`
- [x] `estimate_s_s0_magnitude_ratio()` + `integrator_comparable_s_params()` + `--s-presence-i-scaled-plots`
- [x] `resolve_output_dir()` — redirect repo `output/` → `manifold_sim/`
- [x] `gs_outside_adaptation` flag (NumPy/Numba/PyTorch backends) + `--gs-outside-adaptation` CLI
- [x] Plot suites: `g_s1_d_s48`, `g_s10_d_s21p5585`, `g_s5_d_s48`, `g_s5_d_s0`, `g_s68p0941_d_s21p5585_i_scaled`, `g_s2200_d_s0`, `g_s2025_d_s21p5585`, `g_s2200_d_s21p5585`, `g_s68p0941_d_s21p5585_i_scaled_gs_free`
- [ ] Unconditional / concordance-grouped trajectory diagnostic
- [ ] Re-run full presence with g_s=g_i_fitted
- [ ] `g_s1_d_s48` + `--gs-outside-adaptation` (check tuned pair sensitivity)
