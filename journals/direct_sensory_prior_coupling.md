# Direct sensory prior coupling (g_s / d_s): detectability and biological plausibility

**Scope:** everything about the direct P→S pathway in the generative model — whether feedforward gain (`g_s`) and offset (`d_s`) produce a detectable S prior signature, at what parameter scale, inside vs outside the adaptation gate, with and without I/M prior modulation, and split vs unsplit.

**Status:** the question is answered negatively at biologically plausible scales. With I/M modulation off, S prior distance only becomes significant at `g_s` ≈ 10.7× `g_i_fitted` (inside adaptation) or ≈ 4.7× (outside), or via an implausibly large `d_s` offset (~2.2× `d_i_fitted`). Meanwhile **I** — which has no direct prior modulation in these runs — becomes significant first, because it integrates the modulated S. The plausible biological regime is therefore **weak/undetectable S modulation with a robust downstream I signature**.

Sources: dated entries 2026-06-20 (Experiments 1–7, parameter map, critiques), 2026-06-29 (Goals 2–3, Experiment D), 2026-07-06 Goal 4 / 2026-07-07c (presence unsplit sweep).

---

## Goals

**Experiment 1 goal:** isolate direct P→S coupling by running with presence-case default `g_s`/`d_s` but `g_i=d_i=g_m=d_m=0`. If direct S modulation works we should see (1) clear P-block-L vs P-block-R separation in `p_block_s_trajectory` plots, and (2) significant S prior distance.

**Broader question (2026-06-29 Goal 2):** is sensory gain modulation biologically plausible in this architecture? Hypothesis: S is too feedforward / transient (80 ms window, `tau_s=20` ms, adaptation-suppressed `g_s`) for gain modulation to produce a detectable prior signature at reasonable parameter scales, whereas I, as an active integrator, amplifies weak S modulation over 150 ms.

**Related question (2026-06-29 Goal 3):** can weak S modulation produce significant **I** prior distance? Hypothesis: yes, and this may be the default regime.

**Goal 4 (2026-07-06):** with canonical fitted I/M prior modulation left **on**, at what `(g_s, d_s)` does direct P→S coupling produce a detectable S prior signature under the **stim-side unsplit** pipeline, and when do both `p_mean` and `p_gain` cross α=0.01?

---

## Headline thresholds

| Regime | Condition | Min `g_s` for S significance | vs `g_i_fitted` (189.7) |
|--------|-----------|------------------------------|--------------------------|
| Gain-only, I/M off, split | inside adaptation | **~2025** | ~10.7× |
| Gain-only, I/M off, split | outside adaptation (`gs_outside_adaptation`) | **~850–900** | ~4.5–4.7× |
| Gain-only, I/M **on**, stim-side unsplit | inside adaptation | **~1200** | ~6.3× |
| Offset route, I/M off, split | `g_s ≤ 5` | `d_s ≈ 48` (~2.2× `d_i`) | — |

**I becomes significant before S in every regime tested.** Gain-only I-first thresholds: `g_s ≈ 1800` (inside adaptation), `g_s ≈ 700` (outside). The I-only window is narrow (~200 units of `g_s`) in both cases.

---

## Experiment 1 — S-only presence (canonical presence params, I/M zeroed)

**Implementation:** `load_fitted_model(..., zero_im_prior_mod=False→True)` zeros `g_i/d_i/g_m/d_m` while keeping `g_s/d_s`; `run_s_only_presence_analysis()`; CLI `--s-only-presence`.

Presence defaults: `g_s = g_i_fitted = 189.68`, `d_s = d_i_fitted = 21.56` (not the `g_s=10` override used in the earlier `absence_80ms` presence run).

```bash
conda activate iblenv
python simulate_recovery.py --s-only-presence --seed 123 --n-sessions 40 --nrand 100 --n-jobs 8
```

Outputs: `<ONE cache>/manifold_sim/s_presence_only/seed_123/s_presence_only/`. Do **not** pass `--output-dir output/...`.

### Results (S=80 ms window, contrast-matched shuffle, nrand=100)

| Condition | g_s | g_i | curve_mean | null_median | p_mean | significant |
|-----------|-----|-----|-----------|-------------|--------|-------------|
| Phase 4b (all g/d=0) | 0 | 0 | 0.012 | 0.017 | 0.78 | ✗ |
| **S-only presence** | **189.68** | **0** | **0.037** | **0.026** | **0.15** | **✗** |
| Absence (I/M only, seed 123) | 0 | fitted | 0.798 | 0.240 | 0.00 | ✓ |
| Full presence (seed 123, g_s=10†) | 10 | fitted | 0.959 | 0.372 | 0.00 | ✓ |

† earlier `absence_80ms` presence run used `--g-s-presence 10`, not canonical `g_i_fitted`.

Per-split/contrast (`s_presence_only`, r_choice_r_f1):

| c | n_high | n_low | true | null | p |
|---|--------|-------|------|------|---|
| 0.0 | 83 | 226 | 0.006 | 0.007 | 0.53 |
| 0.0625 | 264 | 592 | 0.011 | 0.006 | 0.25 |
| 0.125 | 326 | 753 | 0.010 | 0.006 | 0.24 |
| 0.25 | 392 | 933 | 0.026 | 0.007 | 0.04 |
| 1.0 | 495 | 976 | 0.013 | 0.006 | 0.18 |

Only c=0.25 reaches p=0.04. The pattern is weak and unlike absence (which peaks at c=0.125 on f1 and c=0.25 on f2). Trajectory plots show P-block-L and P-block-R overlapping at all contrasts; I trajectories are flat as expected with `g_i=0`.

### Interpretation

With the full fitted integrator scale applied to the S feedforward path and all I/M prior modulation zeroed, S prior distance is indistinguishable from Phase 4b in magnitude. The significant S prior distance in **absence** and **full presence** is therefore not driven by direct P→S feedforward coupling; it is mediated by I/M prior modulation changing trial composition and/or M threshold shifts.

**Comparison table — what drives S prior distance?**

| Mechanism | Absence (I/M) | S-only presence | Phase 4b |
|-----------|--------------|-----------------|----------|
| Direct P→S (g_s/d_s) | ✗ | ✓ | ✗ |
| I/M prior mod | ✓ | ✗ | ✗ |
| S prior distance | **large** (0.80) | **null** (0.04) | **null** (0.01) |
| Trajectory P-block separation | none visible | none visible | none visible |

**Why might `g_s`/`d_s` not show up even though they are in the ODE?** Candidates, not mutually exclusive:

1. **Split conditioning washes out direct S modulation.** With I/M off, choices are S-driven only; the trials reaching e.g. `r_choice_r_f1` may be those where S happened to be strong regardless of the prior boost.
2. **P-block-L vs P-block-R grouping ≠ concordant/discordant `g_s` boost.** The boost depends on `(S[0]−S[1])·(P[0]−P[1])` concordance at stim onset — a trial-level quantity that correlates imperfectly with the block label within a split.
3. **`g_s` acts on `S0_delayed` via `(J + g_s·P_gain) @ S0`** — small relative to baseline S0 variance when I/M is not shaping which trials survive.
4. The earlier presence run used `g_s=10`, not `g_i_fitted`.

This conclusion was **partially revised** by Experiment 2: the failure at canonical defaults is a parameter-scale / analysis-sensitivity issue, not proof that `g_s`/`d_s` coupling is inert.

---

## Experiment 2 — g_s/d_s sweep (S-only, I/M zeroed)

`run_gs_ds_tune_sweep()` + `--gs-ds-tune`, with `--g-s-grid`, `--d-s-grid`, `--stop-on-s-significant`, `--tune-alpha`.

```bash
python simulate_recovery.py --gs-ds-tune \
  --seed 123 --n-sessions 40 --nrand 100 --n-jobs 8 \
  --g-s-grid 10,50,100,189.67878020823161 \
  --d-s-grid 21.55851740982741,25,30,35,40,43.11703481965482,50,60,80,100
```

Output: `<ONE cache>/manifold_sim/s_presence_tune/seed_123_refine/gs_ds_tune_sweep.csv`.

### Key finding: the d_s offset is necessary; g_s alone is insufficient

At `g_s=10, d_s=0`: S p=0.69, I p=0.57 — both null. Increasing `d_s` at fixed `g_s=10`:

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

**Minimum S-significant pair:** `g_s=10, d_s=40`. **I becomes significant before S**, at `d_s=21.56` (= fitted `d_i`), where S is still p=0.18.

At canonical presence defaults (`g_s=189.68, d_s=21.56`): S curve_mean 0.037, p=0.15 ✗; I curve_mean 0.031, p=0.00 ✓.

`g_s` scaling at fixed `d_s=43.1` (~2× `d_i_fitted`):

| g_s | S p_mean | S sig | I p_mean | I sig |
|-----|----------|-------|----------|-------|
| 10 | 0.00 | ✓ | 0.00 | ✓ |
| 50 | 0.01 | ✗ | 0.00 | ✓ |
| 100 | 0.03 | ✗ | 0.00 | ✓ |
| 189.68 | 0.00 | ✓ | 0.00 | ✓ |

S significance is **easier at lower `g_s`** when `d_s` is fixed — counterintuitive but consistent with split conditioning washing out large S excursions (stronger `g_s` → faster RT → more truncation/selection effects). At `d_s=60`, all tested `g_s` give S and I p=0.0.

### Revised interpretation

1. Direct P→S coupling **can** produce significant S prior distance, but only with a sufficient `d_s` offset. The canonical `g_s=g_i_fitted, d_s=d_i_fitted` pairing is below the S detection threshold under current splits.
2. `d_s` (offset) matters more than `g_s` (gain) at moderate `g_s` (≈1–10). At extreme `g_s` (~10× `g_i`), gain-only can reach `p_gain` significance (Experiment 4).
3. **I prior distance is a sensitive indirect readout** — significant at lower `d_s` than S because I integrates modulated S. Useful diagnostic: I significant while S is not means S dynamics *are* modulated but the split-conditioned distance metric doesn't capture it.
4. The trajectory-plot problem is separate: even at S-significant parameters (regenerated at `g_s=1, d_s=48`), S trajectories still largely overlap visually while I shows modest separation at high contrast.

---

## Experiment 3 — Tune g_s for S `p_gain` significance at fixed d_s

The Experiment-2 minimum (`g_s=10, d_s=40`) had `p_gain=0.06`. Both `p_mean` and `p_gain` need to clear α=0.01.

At `d_s=48` (≈2.2× `d_i_fitted`), sweeping `g_s`:

| g_s | S p_mean | S p_gain | both sig? |
|-----|----------|----------|-----------|
| 0.25–5 | 0.00 | 0.00 | ✓ |
| 8 | 0.00 | 0.06 | ✗ |
| 10 | 0.00 | 0.03 | ✗ |

At `d_s=40` no `g_s` reached `p_gain=0.0`. Recommended tuned pair `d_s=48, g_s=1`:

| pop | curve_mean | null | p_mean | p_gain |
|-----|-----------|------|--------|--------|
| S | 0.096 | 0.050 | **0.00** | **0.00** |
| I | 0.119 | 0.020 | **0.00** | **0.00** |

`g_s` is much lower than fitted `g_i=189.7`; `d_s` is ~2.2× `d_i_fitted`. Full run confirmed: S p_mean=0.0, p_gain=0.02, p_offset=0.0.

> **Caveat:** the `p_gain` significance at `d_s=48` is **not necessarily a genuine sensory gain profile.** A very large `d_s` produces a block-dependent level shift in S that is large at trial onset and then *decays* as the stimulus drives S to its adapted steady state. The resulting distance curve falls (high early, low late) rather than rising. `p_gain` (late-bin residual after offset subtraction) can still flag this because the decay is slow enough to leave a late-bin residual — but this is the relaxation of an artefactually large offset, not a gain signature. At `d_s ≈ 48 ≈ 2.2 × d_i_fitted` the prior offset is biologically implausible for a feedforward sensory population. The gain-only route (Experiment 4) avoids the artefact but requires extreme `g_s`.

Mechanistically: `p_gain` tests late-bin distance after early-offset removal, so it is sensitive to temporal profile, not just the overall mean. High `g_s` shortens RT / changes split composition → destroys late-bin prior structure (`p_gain` fails at `g_s ≥ 8, d_s=48`). The offset must be large (~48) to create block-dependent S trajectories; the gain should be modest (≈1–5).

```bash
python simulate_recovery.py --s-only-presence --g-s-presence 1 --d-s-presence 48 \
  --seed 123 --n-sessions 40 --nrand 100 --n-jobs 8
```

### Plot suites (2026-06-20)

`--s-presence-tuned-plots` writes to `s_presence_tune/g_s{g}_d_s{d}/`:

| artifact | path |
|----------|------|
| S prior curve + shuffle | `figs/s_prior_curve.png`, `figs/s_shuffle_control.png` |
| I prior curve + shuffle | `figs/I/s_prior_curve.png`, `figs/I/s_shuffle_control.png` |
| S/I comparison | `figs/si_prior_curve_mean_comparison.png`, `figs/si_prior_shuffle_controls.png` |
| Block confounds | `figs/block_confounds/p_block_*` (RT, contrast, S peak, S/I trajectories × 4 splits) |
| Summary | `summary.json` |

| case | g_s | d_s | S p_mean | S p_gain | I p_mean | I p_gain | output |
|------|-----|-----|----------|----------|----------|----------|--------|
| Tuned | 1 | 48 | 0.00 ✓ | 0.00 ✓ | 0.00 ✓ | 0.00 ✓ | `g_s1_d_s48/` |
| I-sig diagnostic | 10 | 21.5585 | 0.18 ✗ | 0.29 ✗ | 0.00 ✓ | 0.00 ✓ | `g_s10_d_s21p5585/` |
| g_s=5 ablation | 5 | 48 | 0.00 ✓ | 0.00 ✓ | 0.00 ✓ | 0.00 ✓ | `g_s5_d_s48/` |
| gain-only null control | 5 | 0 | 0.74 ✗ | 0.74 ✗ | 0.59 ✗ | 0.59 ✗ | `g_s5_d_s0/` |

### Integrator-comparable scaling

Scale S prior params so `g_s × |S0| ≈ g_i × |S|` with `d_s = d_i`:

```python
g_s = g_i_fitted * median(|S| / |S0|)   # 50–80 ms post-stim, c ≥ 0.0625
d_s = d_i_fitted
```

CLI `--s-presence-i-scaled-plots`. Estimated (seed 123): |S|/|S0| = 0.359 → **g_s = 68.1**, **d_s = 21.56**. Output `s_presence_tune/g_s68p0941_d_s21p5585_i_scaled/`.

| pop | p_mean | p_gain | sig mean? | sig gain? |
|-----|--------|--------|-----------|-----------|
| S | 0.22 | 0.26 | ✗ | ✗ |
| I | 0.01 | 0.01 | ✗ (borderline) | ✗ (borderline) |

> **Interpretation of the i-scaled comparison:** the scaling equalises the **per-step gain force** (`g_s × del_P × S0 ≈ g_i × del_P × S`), but not the **accumulated effect**. I is a downstream leaky integrator of S: each `g_s`-modulated S timestep keeps being added into I over the full 150 ms window, so the prior signal in I accumulates and grows monotonically, while S only reflects the instantaneous modulation within its 80 ms window. Even with identical per-step drives, I prior distance is structurally expected to exceed S prior distance — an architectural property, not a flaw in the comparison. That S `p_mean` stays at 0.18–0.22 under i-scaled params confirms the 80 ms S-distance metric cannot detect a prior signal of this magnitude even when the force is correctly calibrated.

---

## Experiment 4 — Gain-only tuning at d_s = 0

```bash
python simulate_recovery.py --gs-tune-p-gain --d-s-fixed 0 --p-gain-only \
  --g-s-grid "2000,2025,2050" --seed 123 --n-sessions 40 --nrand 100 --n-jobs 8
```

Sweep CSV: `manifold_sim/gs_tune_p_gain/gs_ds_tune_sweep.csv`. Contrast-matched nulls (default).

| g_s | d_s | S p_mean | S p_gain | sig? |
|-----|-----|----------|----------|------|
| 0.1 – 189 | 0 | 0.4 – 0.81 | same | ✗ |
| 2000 | 0 | 0.01 | 0.01 | ✗ (borderline) |
| **2025** | **0** | **0.00** | **0.00** | **✓** |
| 2200 | 0 | 0.00 | 0.00 | ✓ |
| 5000 | 0 | 0.00 | 0.00 | ✓ |

**Minimum S `p_gain` at `d_s=0`:** `g_s ≈ 2025` (~10.7× `g_i_fitted`). Plots at `g_s2200_d_s0/` give S and I p_mean/p_offset/p_gain all 0.00 — I is significant indirectly via modulated S feedforward despite `g_i=0`.

**Interpretation:** gain on `S0` *can* produce significant `p_gain`, but only at extreme `g_s`. Offset is ~2000× more efficient (`g_s=1, d_s=48` vs `g_s=2025, d_s=0`). Neither route is biologically clean.

---

## Experiment 5 — High g_s + fitted d_i (gain + offset)

| g_s | d_s | output dir | S p_mean | S p_gain | I p_mean | I p_gain |
|-----|-----|------------|----------|----------|----------|----------|
| 2025 | 21.56 | `g_s2025_d_s21p5585/` | 0.00 | 0.00 | 0.00 | 0.00 |
| 2200 | 21.56 | `g_s2200_d_s21p5585/` | 0.00 | 0.00 | 0.00 | 0.00 |

Adding `d_i` to a high `g_s` **increases** S `curve_mean` (0.054 → 0.091 at `g_s=2025`; 0.063 → 0.104 at `g_s=2200`) without hurting significance.

| case | g_s | d_s | S curve_mean | mechanism |
|------|-----|-----|-------------|-----------|
| Tuned (efficient) | 1 | 48 | 0.096 | offset-dominated |
| High-g + d_i | 2025 | 21.56 | 0.091 | gain + modest offset |
| Gain-only | 2025 | 0 | 0.054 | gain-dominated |

---

## Experiment 6 — g_s outside the adaptation gate

**Motivation:** `g_s` is applied inside the adaptation gate, `a * ((J + g_s·P_gain) @ S0)`. At steady state `a ≈ 0.09`, suppressing `g_s` ~11× relative to `g_i` (which is not adapted). `--gs-outside-adaptation` splits the feedforward term as `a*(J@S0) + g_s*P_gain@S0`.

Results at `g_s=68.09, d_s=21.56` (i-scaled), seed 123, nrand=100 → `s_presence_tune/g_s68p0941_d_s21p5585_i_scaled_gs_free/`:

| metric | g_s inside `a` | g_s outside `a` |
|--------|----------------|-----------------|
| S `p_mean` | 0.22 ✗ | 0.18 ✗ |
| S `p_offset` | — | **0.00 ✓** |
| S `p_gain` | 0.26 ✗ | 0.24 ✗ |
| I `p_mean` | 0.01 borderline | **0.00 ✓** |
| I `p_offset` | — | **0.00 ✓** |
| I `p_gain` | 0.01 borderline | 0.01 ✗ |

Moving `g_s` outside `a` adds detectable early-time block-dependent modulation (`p_offset` significant for both S and I), strongest at trial onset (`a≈1`) and fading as adaptation reduces `a`. S `p_mean` is still n.s.: `del_P ≈ 2×10⁻⁴` limits the absolute contribution even without adaptation suppression.

**Architectural conclusion:** `g_s` inside `a` vs `g_i` outside `a` is a real asymmetry in the model. Default `gs_outside_adaptation=False` for backward compatibility; set `True` to match the `g_i` architecture. Implemented across NumPy / Numba / PyTorch backends.

---

## Experiment 7 — Gain-only sweep with g_s outside adaptation

**Prediction:** adaptation suppresses `g_s` ~11× at steady state, so the min `g_s` for S `p_gain` should drop from ~2025 to ~184 ≈ `g_i_fitted`.

```bash
python simulate_recovery.py --gs-tune-p-gain --gs-outside-adaptation \
  --d-s-fixed 0 --p-gain-only \
  --g-s-grid "50,100,150,170,180,189.67878020823161,200,220,250,300,400" \
  --seed 123 --n-sessions 40 --nrand 100 --n-jobs 8
```

| g_s | S p_mean | S p_gain | S gain_effect | I p_mean | I gain_effect |
|-----|---------|---------|--------------|---------|--------------|
| 50 | 0.83 | 0.83 | −0.0083 | 0.62 | −0.0026 |
| 180 | 0.63 | 0.63 | −0.0051 | 0.67 | −0.0019 |
| 189.68 (=g_i) | 0.76 | 0.76 | −0.0090 | 0.57 | −0.0016 |
| 300 | 0.60 | 0.60 | −0.0049 | 0.32 | **+0.0009** |
| 400 | 0.54 | 0.54 | −0.0031 | **0.09** | **+0.0041** |

**Prediction was wrong.** What happened:

1. **S `gain_effect` is negative at all `g_s`** — the distance curve reverses sign at late times. Concordant trials (amplified by `g_s`) reach threshold faster → truncate earlier; discordant trials are slower → fill late bins. The f1/f2 selection filter *actively inverts* the late-time S signal. Removing adaptation from `g_s` does not change this — it is structural.
2. **S `p_mean` ≈ S `p_gain`** throughout because early (positive) and late (negative) contributions cancel.
3. **I `gain_effect` turns positive at `g_s ≈ 250–300`**: I accumulates the modulated S without the truncation reversal.

**Verification at `g_s=2025` with `gs_outside_adaptation`:**

| pop | curve_mean | null | p_mean | p_gain | sig? |
|-----|-----------|------|--------|--------|------|
| S | **0.327** | 0.025 | **0.00** | **0.00** | **✓** |
| I | **0.186** | 0.007 | **0.00** | **0.00** | **✓** |

Six times larger `curve_mean` than Experiment 4 at the same `g_s` inside adaptation (≈0.054). At this drive the per-step gain is `g_s × del_P × S0 ≈ 0.10`, about 4.5× the adapted raw signal (0.022): concordant trials reach threshold near-instantly and truncate, discordant trials stay elevated late → the window shows concordant≈0 vs discordant=signal → **large positive distance**. The reversal has been co-opted in favour of the signal.

**Refined crossover search (seed 123):**

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

- `gain_effect` sign flip: `g_s ≈ 465`
- `p_gain` significance: `g_s ≈ 850–900`

**Conclusion:** moving `g_s` outside adaptation shifts the significance crossover from ~2025 to ~850–900, a **~2–4× reduction**, not the predicted 11×. The 11× applies at steady state (`a≈0.09`, `t ≫ tau_a = 222` ms), but the crossover mechanism is RT-driven: concordant trials fire at short RTs (~80 ms) while adaptation is still ramping and `a(t)` is much closer to 1. The sign crossover (~465) is mechanistically distinct from the significance crossover (~850–900): the sign flip only requires concordant trials to truncate faster than discordant ones; significance requires the resulting distance to exceed the null across 40 sessions.

---

## Experiment D — Gain-only threshold for "I significant, S not"

**Goal:** find the minimum `g_s` at `d_s=0` (all other g/d=0) where I is significant but S is not, inside and outside the adaptation gate. Seed 123, 40 sessions, nrand=100, α=0.01, contrast-matched null.

**Inside adaptation:**

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

**Outside adaptation:**

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

**Threshold summary**

| g_s placement | I becomes sig | S becomes sig | I-only window | vs g_i_fitted (189.7) |
|---------------|--------------|--------------|---------------|----------------------|
| Inside adaptation | ~1800 | ~2025 | ~225 units | I: ~9.5×, S: ~10.7× |
| Outside adaptation | ~700 | ~900 | ~200 units | I: ~3.7×, S: ~4.7× |

Moving `g_s` outside adaptation lowers both thresholds ~2.5–2.6×. The I-only window is narrow in both cases.

**Key confirmation:** `g_s=10, d_s=0` (inside adaptation) does **not** make I significant (I p=0.57, S p=0.69). The I-significance at `g_s=10, d_s=21.56` in Experiment 3 was **offset-driven**, not gain-driven.

### Diagnostic plot suites (I significant, S not)

| case | g_s | d_s | gs_free | S p_mean | S p_gain | I p_mean | I p_gain | output dir |
|------|-----|-----|---------|----------|----------|----------|----------|------------|
| Inside adap, I-only | 1800 | 0 | ✗ | 0.04 ✗ | 0.04 ✗ | 0.00 ✓ | 0.00 ✓ | `s_presence_tune/g_s1800_d_s0/` |
| Outside adap, I-only (**primary**) | **700** | 0 | ✓ | **0.08 ✗** | **0.08 ✗** | **0.00 ✓** | **0.00 ✓** | `s_presence_tune/g_s700_d_s0_gs_free/` |
| Outside adap, I-only (alt) | 800 | 0 | ✓ | 0.02 ✗ | 0.02 ✗ | 0.00 ✓ | 0.00 ✓ | `s_presence_tune/g_s800_d_s0_gs_free/` |

Key figures: `figs/s_shuffle_control.png`, `figs/i_shuffle_control.png`.

**Visual observations at `g_s=700`, outside adaptation (most striking case):**

- **S (p=0.08):** the true curve rises slowly from zero, tracking just above the null band but with heavy overlap and noisy late bins. 8 of 100 null shuffles exceed the true curve mean — visually indistinguishable from null for most of the trial.
- **I (p=0.00):** the true curve diverges from null with an exponential-looking rise starting at ~60 ms, reaching >10× the null mean and ~5× the highest null shuffle at 150 ms. All 100 nulls sit clearly below the true curve from ~80 ms onward.

**Mechanistic reading:** the same weak `g_s × del_P × S0` gain drive that is undetectable in S's 80 ms window is converted by I's leaky integration into a large, monotonically growing separation over 150 ms. Direct sensory gain modulation is architecturally invisible in S but leaves a robust downstream signature in I.

---

## Goal 4 — Presence unsplit sweep (fitted I/M left on)

Distinct from Experiments 1–7 (`zero_im_prior_mod=True`) and from absence unsplit (S n.s. at `g_s=d_s=0`): here I/M feedforward/selection effects stay in the model while only S coupling is swept, and the analysis is **stim-side unsplit**.

**Implementation (2026-07-07):**

- `--unsplit-prior presence` — single `(g_s, d_s)` with fitted I/M.
- `--presence-unsplit-sweep` — 2D grid with stim-side unsplit; default `g_s ∈ [0, 2500]`, `d_s ∈ [0, d_i_fitted]` (`default_presence_unsplit_sweep_grid`).
- Outputs: `presence_unsplit_sweep/seed_<seed>/presence_unsplit_sweep.csv` + `_summary.json`.
- Runner: `run_presence_unsplit_sweep.sh` (seed 123, 40 sessions, nrand 100).

```bash
conda activate iblenv
./run_presence_unsplit_sweep.sh
# or custom grid:
python simulate_recovery.py --presence-unsplit-sweep --g-s-grid 0,500,1800,2500 --d-s-grid 0,10,21.56
```

**Run (2026-07-07c):** 80/80 pairs in ~75 min. 46/80 pairs have S `p_mean` and `p_gain` both significant at α=0.01; they always co-occur in this grid.

**Table D — S significance thresholds (fitted I/M, stim-side unsplit, CM null)**

| route | threshold | S curve_mean | p_mean | p_gain | notes |
|-------|-----------|-------------:|-------:|-------:|-------|
| baseline | g_s=0, d_s=0 | 0.0056 | 0.13 | 0.13 | n.s. — matches absence unsplit |
| offset-only | g_s=0, d_s=d_i | 0.0105 | 0.00 | 0.00 | sig at **max d_s only**; intermediate d_s n.s. at g_s=0 |
| gain-only | d_s=0, g_s≥**1200** | 0.0094 | 0.00 | 0.00 | first both-sig; g_s=900 borderline (p≈0.06) |
| gain-only | d_s=0, g_s=1800 | 0.0223 | 0.00 | 0.00 | |
| gain-only | d_s=0, g_s=2025 | 0.0282 | 0.00 | 0.00 | |
| mixed | d_s=d_i, any g_s≥0 | 0.0105+ | 0.00 | 0.00 | offset at d_i dominates; low g_s sufficient |

**Min `g_s` for both-sig at each `d_s`:**

| d_s | min g_s (both sig) |
|-----|-------------------:|
| 0 | 1200 |
| 5.4 | 1200 |
| 10.8 | 900 |
| 16.2 | 379 |
| 21.6 (d_i) | **0** |

**Findings**

1. **Gain-only with fitted I/M:** the S both-sig threshold drops to `g_s ≈ 1200` (vs ~1800 for the zero-I/M unsplit case) — I/M context makes moderate direct P→S gain easier to detect, but it still requires ~6× `g_i_fitted`.
2. **The offset route is narrow:** only `d_s = d_i` at `g_s=0` reaches significance; sub-maximal offsets fail at `g_s=0` despite I/M being on.
3. **Baseline (`g_s=d_s=0`) remains n.s. on S** — fitted I/M alone does not create stim-side-unsplit S prior distance, consistent with absence unsplit.
4. `p_mean` and `p_gain` are always co-significant in this grid.

Compared to zero-I/M unsplit: s1800 S curve_mean 0.0197 vs presence + `g_s=1800, d_s=0` 0.0223 — similar magnitude, but presence hits significance at a lower `g_s`.

**Example plots (2026-07-08):** `run_presence_unsplit_examples.sh` → `presence_unsplit_sweep/seed_123/examples/`. Per case (`presence_g_s{1200,1800}_d_s0_unsplit/figs/`): `s_prior_curve.png`, `s_shuffle_control.png`, `presence_*_curve_mean_comparison.png`, `presence_*_shuffle_controls.png`, `block_confounds/p_block_s_trajectory_*.png`.

---

## Parameter map (summary)

| case | g_s | d_s | gs_free | S sig? | notes |
|------|-----|-----|---------|--------|-------|
| Phase 4b | 0 | 0 | — | ✗ | baseline |
| Canonical S-only | 189.7 | 21.56 | — | ✗ | I sig, S not |
| I-sig diagnostic | 10 | 21.56 | — | ✗ | I sig mean+gain |
| **Tuned (recommended)** | **1** | **48** | — | **✓** | offset-artefact (see critique) |
| g_s=5 ablation | 5 | 48 | — | ✓ | same family as tuned |
| g_s=5 gain-only | 5 | 0 | — | ✗ | confirms d_s needed |
| Integrator-scaled | 68.1 | 21.56 | ✗ | ✗ | g_s = g_i × (S/S0) |
| Integrator-scaled + gs_free | 68.1 | 21.56 | ✓ | p_offset ✓ | I p_mean sig |
| Gain-only min | 2025 | 0 | — | ✓ p_gain | selection-filter mechanism |
| High-g + d_i | 2025–2200 | 21.56 | — | ✓ | gain + offset |
| **Gain-only gs_free** | **50–400** | **0** | **✓** | **✗** | prediction wrong; reversal structural |
| Gain-only gs_free (crossover) | 850–900 | 0 | ✓ | ✓ | significance crossover |
| Presence unsplit gain-only | 1200 | 0 | — | ✓ | fitted I/M on, stim-side unsplit |

---

## Biological reading (2026-06-29 Goal 3)

Evidence at canonical S-only params (`g_s=189.7, d_s=21.56`, all I/M prior mod off):

| pop | curve_mean | p_mean | significant? |
|-----|-----------|--------|--------------|
| S | 0.037 | 0.15 | ✗ |
| I | 0.031 | 0.00 | ✓ |

Same pattern at `g_s=10, d_s=21.56`: S p_mean=0.18, p_gain=0.29; I p_mean=0.00, p_gain=0.00.

A small concordance-dependent boost on `S0` at stim onset, integrated over the trial by I, yields a robust prior signal downstream even when S itself looks null. This is **architecturally expected** (I accumulates; S is instantaneous) and does **not** require the large S prior distance seen in absence — that absence signal is I/M selection (see [split conditioning vs unsplit](split_conditioning_vs_unsplit.md)).

**Open interpretive question:** is this weak-S / strong-I regime a reasonable account of experimental data (prior effects in integrator-like populations but not early sensory)? Or does it make the presence/absence comparison uninterpretable, since any I/M prior effect could always be explained away as weak S feedforward?

---

## Open questions / critiques

1. **`d_s=48` `p_gain` is likely an offset-decay artefact**, not a true gain profile (early high, late low as the offset relaxes to the stimulus-driven steady state). The "tuned" `g_s=1, d_s=48` pair is therefore not a valid demonstration of sensory prior gain.
2. **The i-scaled comparison is conceptually valid** for equating instantaneous gain force, but I has a structural accumulation advantage. I prior distance exceeding S at matched gain is expected by architecture. The i-scaled run showing I significant while S is not does **not** mean S prior coupling is absent — it means the 80 ms S-distance metric is insensitive at that force level.
3. **The only routes to S `p_mean` significance so far** are `g_s ≈ 2025` (10.7× `g_i_fitted`) at `d_s=0`, or large non-physiological offsets. Neither is a clean demonstration of a sensory prior.
4. **What would be a clean test:** measure S prior distance with a metric that matches I's accumulation advantage — e.g. trial-average S at all contrasts grouped by concordance (not split by choice), or a longer S window matching the I window. The unsplit analysis is the partial answer; concordance-grouped trajectories remain unrun.
5. **The visual failure persists:** `p_block_s_trajectory` plots show no P-block-L vs P-block-R separation in any condition tested, even where S prior distance is significant. The zero-padding cause was fixed (see [S prior artefacts](s_prior_artifacts_truncation.md)); the residual visual null is unexplained.
6. **Structural insight:** f1/f2 split conditioning + RT-driven truncation can **invert** the late-time S `gain_effect` (concordant trials truncate faster → discordant trials dominate late bins). This is not fixed by moving `g_s` outside the adaptation gate.

### Not-yet-run items

- Unsplit S-only canonical (`g_s=g_i_fitted`, `g_i=0`) and unsplit I-sig diagnostic (`g_s=10, d_s=d_i`).
- Concordance-grouped trajectory diagnostic.
- Full presence re-run with canonical `g_s=g_i_fitted` to separate the `g_s` scale from the I/M contribution.
- `g_s1_d_s48` with `--gs-outside-adaptation` (tuned-pair sensitivity).
- Contrast-matched vs unrestricted nulls at borderline cases (partly covered by Tables A/B).

---

## Implementation checklist

- [x] `load_fitted_model(..., zero_im_prior_mod=True)`
- [x] `run_s_only_presence_analysis()` + `--s-only-presence`
- [x] `run_gs_ds_tune_sweep()` + `--gs-ds-tune`
- [x] `run_gs_tune_p_gain()` + `--gs-tune-p-gain` + `--p-gain-only`
- [x] `run_s_presence_tuned_plots()` + `--s-presence-tuned-plots`
- [x] `estimate_s_s0_magnitude_ratio()` + `integrator_comparable_s_params()` + `--s-presence-i-scaled-plots`
- [x] `resolve_output_dir()` — redirect repo `output/` → `manifold_sim/`
- [x] `gs_outside_adaptation` flag (NumPy/Numba/PyTorch) + `--gs-outside-adaptation`
- [x] `--unsplit-prior` extended to `s_presence` and `presence`; `--presence-unsplit-sweep`
- [x] Plot suites: `g_s1_d_s48`, `g_s10_d_s21p5585`, `g_s5_d_s48`, `g_s5_d_s0`, `g_s68p0941_d_s21p5585_i_scaled`, `g_s2200_d_s0`, `g_s2025_d_s21p5585`, `g_s2200_d_s21p5585`, `g_s68p0941_d_s21p5585_i_scaled_gs_free`, `g_s1800_d_s0`, `g_s700_d_s0_gs_free`, `g_s800_d_s0_gs_free`
- [ ] Unconditional / concordance-grouped trajectory diagnostic
- [ ] Re-run full presence with `g_s=g_i_fitted`
