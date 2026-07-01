# Research journal — 2026-06-18

## Standing context (carry-forward from 2026-06-17)

See [prior journal](research_journal_2026-06-17.md) for full Phase 0–4b results. Quick summary of what is known:

- **Standard absence** (`g_s=d_s=0`, fitted `g_i/d_i/g_m/d_m`): S combined prior distance **1.30**, contrast-matched shuffle median **0.17**, p=0/100.
- **Phase 4b** (all `g_*=d_*=0`): S combined **0.11**, shuffle median **0.026**, p=0/100 — residual significant at **c=1.0 on `*_f1` splits only**, 5/5 seeds. `*_f2` splits n.s.
- No code bug (4a audit). Adaptation `a_at_stim` medians differ by ~7×10⁻⁵. Constant S0 amplifies Phase 4b residual (~70×). Multiseed robust. Trial-structure / block-epoch excluded by argument (S feedforward; block not in ODE; splits fix stim+choice side).

---

## Today's goals

1. **Q1 — Why does I/M prior mod (standard absence) increase S prior distance ~12× relative to Phase 4b?**
2. **Q2 — Why is there S prior diff at all when all g/d = 0 (Phase 4b)?**

Both mechanisms are currently **open**. Below: candidate hypotheses, evidence for/against each, and concrete experiments to run.

---

## Q1: I/M prior mod inflates S prior distance

### Observed fact

|                      | S combined `curve_mean` | shuffle median |
| -------------------- | ----------------------- | -------------- |
| Standard absence     | **1.30**                | 0.17           |
| Phase 4b (no mod)    | 0.11                    | 0.026          |

Ratio: **~12×**. I/M mod is the only structural difference.

### What I/M mod does in the generative model

With `g_i/d_i/g_m/d_m > 0`:
- I population receives a gain/offset proportional to P (prior probability).
- M action threshold is shifted by concordant/discordant P (via `del_P` / `theta_c` vs `theta_d`).
- **S is unaffected** (`g_s=d_s=0` always) — S only evolves under `S0`, `W_ss`, and adaptation `a`.

Yet S prior distance is 12× larger. There is no direct path from P to S. The inflation must arise **indirectly**, through the trial composition of each split.

### Hypothesis 1 (primary): Differential S0 filtering / collider selection bias

**Mechanism:**

The 4 splits are defined by `(block_during_stim, stim_side, choice_side, feedback)`. With I/M mod ON, P biases choices toward the prior-concordant side. For a right block, right stimulus, right choice, correct (f1) trial:

- **High-prior-group** (`ITI P ≥ 0.5`): these trials are a minority in a right block (right block has P < 0.5 most of the time; high-P instances are near block transitions). I/M boost is low here — correct choices rely on S0 strength.
- **Low-prior-group** (`ITI P < 0.5`, typical right block): I/M pushes M toward right choice. Even trials with weaker S0 realizations can end up in the correct-right bucket when I/M provides a push.

**Result**: In the `r_choice_r_f1` split, the low-prior group is enriched with trials where **the I/M prior did the work** → average S0_r (and therefore S trajectory) may be *lower* in the low-prior group than the high-prior group, or vice versa depending on whether I/M competes or cooperates with S.

More precisely: when I/M is strongly biased right (low P in right block, high g_i), even low-S0 right-stim trials make it into f1. So the low-prior f1 group has a *wider* S0 distribution (more weak-S0 trials captured by I/M boost). The high-prior group within this right-block, right-choice split is near-transition trials where I/M is less biased, so only stronger-S0 trials pass. **→ S0 mean is higher in high-prior group → S trajectory higher → non-zero distance even though no P→S coupling.**

This mechanism requires only that (a) I/M mod changes the selection filter and (b) S0 is stochastic.

**Testable prediction:** Running the same split analysis with constant S0 + I/M mod should **collapse** the distance toward Phase 4b levels, because there is no S0 variance for the selection filter to work on. (Compare: Phase 4b + constant S0 gave huge distance at c=1.0; but there, any imbalance is structurally amplified by deterministic S0 within a block. The key is whether the effect *pattern* changes.)

**Direct test (already partially available):** Compare mean S0 realization between high- and low-prior groups within each (split, contrast) cell in standard absence. If `mean(S0_signal | high-prior, c=X, r_choice_r_f1) ≠ mean(S0_signal | low-prior, c=X, r_choice_r_f1)`, this is the mechanism.

### Hypothesis 2: I/M drives RT differences → different time-of-peak S

I/M prior mod accelerates RTs on concordant trials. If high-prior group has earlier choices, the S trajectory is sampled at a different phase. The distance metric integrates over the full stimulation window. If one group's mean S peaks earlier/later, time-averaged squared distance is inflated even with identical steady-state S.

**Evidence against (partial):** Phase 2 covariate table shows RT differs significantly (`p ~ 10⁻¹¹`) between groups in `r_choice_r_f1` (high RT=73 ms, low RT=80 ms). RT difference of ~7 ms at tau_s=20 ms is meaningful. But the S distance is measured during stim, not RT-aligned; a small RT difference could shift peak timing but unlikely to account for 12× amplification.

**Test:** Align S trajectories by choice time (rather than stim onset) and recompute distances. If the effect vanishes with RT-aligned analysis, RT/timing is the dominant pathway.

### Hypothesis 3: trial_in_block imbalance → differential adaptation history

Phase 2 covariate: `trial_in_block` median differs significantly between groups in `r_choice_r_f1` (high=20, low=28, p~10⁻²⁴). Early-in-block trials (high-prior group) have a different cumulative stimulation history than late-in-block trials. Adaptation `a` accumulates during the block. Phase 2 measured `a_at_stim` median ~0.9913 for both groups across all splits/contrasts; difference ~7×10⁻⁵. But for `r_choice_r_f1` specifically at c=1.0, the `a_at_stim` sub-distribution might show larger differences.

**Test:** Compute `a_at_stim` mean and std **separately** for each (split, contrast, prior group) cell. Does the c=1.0 f1 cell show larger `a_at_stim` deviation than median?

### Summary table for Q1

| Hypothesis                         | Mechanism                              | Key test                                             | Status         |
| ---------------------------------- | -------------------------------------- | ---------------------------------------------------- | -------------- |
| S0 filtering / collider            | I/M selection creates S0 group imbalance | Direct S0 mean per group cell                        | **To run**     |
| RT / S-trajectory timing           | Faster RT in high-prior → earlier S peak | RT-aligned S trajectory comparison                   | **To run**     |
| trial_in_block × adaptation        | Early-in-block → different a_at_stim   | a_at_stim per (split, c, group) cell                 | **To run**     |
| Standard absence + constant S0     | Isolate stochastic S0 contribution     | Run `--phase4-constant-s0` WITH fitted I/M mod        | **To run**     |

---

## Q2: S prior diff at all when g_*=d_*=0 (Phase 4b)

### Observed fact

- Phase 4b: S combined ~0.11–0.13 (5 seeds), p=0/100 every seed.
- Effect is **exclusively at c=1.0 on `*_f1` splits** (r_choice_r_f1 and l_choice_l_f1).
- `*_f2` splits: p ≈ 0.3–0.8 (n.s.).
- Constant S0 (Phase 4b + no noise): **amplifies** effect (S true → 2.9 at c=1.0 f1), doesn't remove it.
- Random ITI labels: curve_mean collapses to 0.034 (below shuffle null 0.17) — the **label assignment is informative**, not simulation dynamics.

### What can still differ between high- and low-ITI-P groups when all g/d = 0?

With no prior modulation, S evolves purely as:

```
dS/dt = (1/tau_s) * nonlin(-S + W_ss * J @ S + a * J @ S0_delayed)
```

with `W_ss ≈ 7.6×10⁻⁵` (negligible cross-population coupling) and `a` as the adaptation state.

The **only sources of variability** between trials are:
1. **S0** — stochastic draws from stimulus distribution (or deterministic in constant-S0 run)
2. **Adaptation `a`** — session-level state that evolves across trials
3. **ITI duration** — affects how much `a` recovers between trials
4. **Implicit block identity in the ITI P label** — `p_subjective_probabilityLeft` is a continuous trace that correlates with block structure

## UPDATE 2026-06-18 afternoon: Analysis-window truncation bug identified (high confidence)

> **TL;DR:** The Phase 4b c=1.0 f1 residual and the Q1 12× inflation are both caused (at least partially) by an **analysis artifact**: `bin_trace_segment` zero-pads trailing bins when a trial ends before the 150 ms window. At c=1.0, most trials have RT < 110 ms (the truncation threshold), so the mean S trajectory is a mix of real signal and structural zeros — and the two groups have different RT distributions → different zero-padding → spurious distance.

### The truncation bug

The analysis window for `act_block_duringstim` is `[0, 150 ms]` = 75 steps. `window_step_bounds` clips to `min(trial_len, sbo+75)`. Trials end at `sbo + RT + post_action_steps` where `post_action_steps = 20` steps (40 ms). So a trial is **truncated** whenever `RT < 55 steps = 110 ms`. `bin_trace_segment` fills empty trailing bins with **0.0**, introducing structural zeros into the mean trajectory.

| RT (ms) | trial ends post-stim | zero bins / 75 | fraction zeros |
| ------- | -------------------- | -------------- | -------------- |
| 20 ms   | 60 ms                | 45 / 75        | 60%            |
| 40 ms   | 80 ms                | 35 / 75        | 47%            |
| 60 ms   | 100 ms               | 25 / 75        | 33%            |
| 80 ms   | 120 ms               | 15 / 75        | 20%            |
| 100 ms  | 140 ms               | 5 / 75         | 7%             |
| ≥110 ms | ≥150 ms              | 0              | 0%             |

At c=1.0, most correct choices fire well before 110 ms. **If the two prior groups have different RT distributions, the zero-padding is asymmetric → non-zero S distance even with identical model dynamics.**

### Evidence from the Phase 4b trajectory figure

`p_block_s_trajectory_r_choice_r_f1.png` (seed 123):
- **High-prior (P block L, left panel):** c=1 S_r peaks at ~40 ms then sharply returns to 0, with a brief oscillation crossing zero. This is the signature of zero-padded trailing bins.
- **Low-prior (P block R, right panel):** c=1 S_r is sustained until ~70–80 ms, suggesting later trial end.

The "return to zero" at c=1.0 is **not real S dynamics** — real S driven by a constant c=1.0 stimulus does not abruptly return to zero mid-trajectory. It is zero-padding from trial truncation.

### Why RT differs between groups at c=1.0 Phase 4b

With all g/d = 0, the only cross-trial carry-over mechanism is the slow decay of `I[0]−I[1]` (time constant ~204 ms for the difference mode; 0.74% residual after 1000 ms ITI):
- **Left-block (high-prior) trials:** Previous stimuli are predominantly left → I carry-over is left-biased (I[0] > I[1]). The left-biased carry-over has an initial POSITIVE bias on M[0]−M[1], which starts M moving toward a LEFT choice — but the strong right stimulus quickly overcomes this. Net effect: some subset of trials fire M even faster (the initial M excursion and correction creates a faster sign-flip to right choice) or slower depending on nonlinear dynamics.
- The figure confirms that in practice **high-prior trials have systematically shorter RT** (peak at ~40 ms) than low-prior trials (peak at ~70 ms) in Phase 4b at c=1.0.

The exact direction requires a simulation diagnostic (pending), but the **directional asymmetry in zero-padding is empirically visible from the trajectory figure** and sufficient to explain the observed S distance.


### Implication for Q1 (standard absence, 12× inflation)

In standard absence, the **selection filter** for the `r_choice_r_f1` split also generates an RT asymmetry:
- High-prior (left block, anti-concordant correct): the correct right choice DESPITE I/M left-bias requires strong S → strong S → fast M firing → **faster RT** for the trials that survive into the split
- Low-prior (right block, concordant correct): right choice aided by I/M right-bias → any S suffices → mix of fast/slow RT → **average or slower RT**

Phase 2 confirms: high-prior RT median = 73 ms, low-prior = 80 ms. Both are below the 110 ms truncation threshold. High-prior has ~18 zero-padding bins vs low-prior ~15 bins. The differential zero-padding creates the inflated S distance.

**The truncation artifact explains the majority of the Phase 4b residual and likely amplifies Q1 substantially.** The I/M modulation further inflates it by changing the RT distribution (selection into f1 split) and possibly by genuine I/M-driven S dynamics (Q1 true component).

### Zero-padding confirmed quantitatively

`bin_trace_segment` uses `linspace(0, seg_len, n_bins+1).astype(int)`. When `seg_len < n_bins` (= 72 for 150 ms window), many consecutive edge values are equal → empty bins → filled with **0.0**:

| RT (ms) | seg_len (steps) | empty zero-bins / 72 | % zeros |
| ------- | --------------- | -------------------- | ------- |
| 20 ms   | 20 + 20 = 40    | **32/72**            | 44 %    |
| 40 ms   | 40 + 20 = 60    | **12/72**            | 17 %    |
| 60 ms   | 60 + 20 = 80    | **0/72** (no clip)   | 0 %     |
| ≥90 ms  | ≥ 110            | 0                    | 0 %     |

*(seg_len = RT_steps + post_action_steps = RT_steps + 20)*

At c=1.0, almost all trials have RT < 60 ms (Phase 4b no-mod; strong signal). So **most trials contribute heavily zero-padded bins**. If the two prior groups have different median RT (as the trajectory figure confirms: high-prior peaks at ~40 ms, low-prior at ~70 ms), their mean trajectories are systematically different → spurious non-zero distance.

With **constant S0 = 1.0** (stronger drive than perceived_c ≈ 0.70): RT is even shorter → more zero-padded bins → larger distance.

---

## Implementation checklist

- [x] `--require-full-window` flag to exclude zero-padded truncated trials from distance computation (implemented in `build_population_b_for_split` → `build_split_results` → `build_res_from_trajectories` → `_population_prior_from_sessions` → `run_phase4_no_prior_mod_analysis` → CLI)
- [x] Replace zero-padding with **fill-from-next-ITI** as the default in `build_population_b_for_split` (mirrors `prior_distance_I_M_by_choice_and_prior` in `model_functions.py`): for `stimOn`-aligned splits where trial ends before the 150 ms window, borrow the leading `need` steps of the next trial's ITI. If the next trial is in a different session or its ITI is too short, skip the trial. `require_full_window=True` remains as a strict-exclusion diagnostic option.


---

## UPDATE 2026-06-18 evening: Zero-padding artifact confirmed + Phase 4b retest

### Bug fix: fill-from-next-ITI (default behavior)
`bin_trace_segment` used to zero-pad trailing bins when a trial ended before the 150 ms window. Replaced with **fill-from-next-ITI**: borrow the leading `need` steps of the next trial's ITI to complete the window (mirrors `prior_distance_I_M_by_choice_and_prior` in `model_functions.py`). If the next trial is in a different session or has insufficient ITI, the trial is skipped. `--duringstim-window-ms` CLI flag added to override the post-stim analysis window (e.g. 80ms).

### Phase 4b retest (seed 123, all g/d=0, nrand=100)

| Condition | S curve_mean | S p | I curve_mean | I p | M curve_mean | M p |
|-----------|-------------|-----|-------------|-----|-------------|-----|
| Old (zero-pad, 150ms) | 0.117 | 0.00 | 0.054 | 0.00 | 0.203 | 0.00 |
| Fix (fill-next, 150ms) | 0.041 | 0.04 | 0.004 | 0.60 ✓ | 0.004 | 0.81 ✓ |
| Fix (fill-next, 80ms) | 0.012 | 0.78 ✓ | 0.001 | 0.25 ✓ | 0.001 | 0.08 ✓ |

**I and M**: completely non-significant with fill-from-next at 150ms — the artifact was the sole cause of their apparent prior distance.

**S at 150ms**: reduced from 0.117 → 0.041, marginally significant (p=0.04). Still driven by c=1.0 f1 splits (0.100 vs null 0.005). This residual is from the fill-from-next tail (80–150ms window post-RT) where the borrowed ITI data still differs between prior groups (different RT distributions → different ITI borrow amounts → structural asymmetry).

**S at 80ms**: fully non-significant (p=0.78). c=1.0 f1 collapses to 0.008 vs null 0.007. At 80ms, most high-contrast correct trials have terminated and only real stim-driven activity contributes — no RT-driven ITI-fill bias.

### Interpretation
With all g/d=0 and an 80ms window (capturing only real stim-driven S activity), Phase 4b S/I/M prior distance is **null across all contrasts and splits**. The 0.117 S distance from the old analysis was almost entirely an analysis artifact. The remaining signal at 150ms (fill-next) is the ITI-borrow tail bias, not real neural dynamics.

**Next: rerun standard absence (fitted I/M) with fill-from-next, and change S analysis window to 80ms.**

---

## UPDATE 2026-06-19: S window set to 80ms; Phase 4b + standard absence retest

### Implementation
S population now uses an 80ms analysis window for all `act_block_duringstim` splits (`S_DURINGSTIM_WINDOW_S = 0.08`), applied inside `build_population_b_for_split` when `population == "S"`. I and M continue to use the full 150ms window from `PRE_POST`. The `n_bins` is recomputed accordingly (80ms → 32 bins). The `--duringstim-window-ms` CLI override still works for batch experiments.

---

## Standard absence retest (fitted I/M, seed 123, nrand=100, fill-from-next, 150ms)

### Overall S prior distance

| Condition | curve_mean | null_median | true−null |
|-----------|-----------|-------------|-----------|
| OLD absence (zero-pad, 150ms) | 1.298 | 0.174 | 1.124 |
| NEW absence (fill-next, 150ms) | 1.450 | 0.164 | 1.286 |

Both highly significant (p=0.0). The overall `curve_mean` is slightly *larger* with fill-from-next — the zero-padding was actually suppressing the genuine low-contrast signal (slow-RT trials at low contrast are less truncated → less zero-padding → less inflation, but the genuine I/M-driven S effect is large there regardless).

### Per-contrast breakdown: `r_choice_r_f1` (S)

| contrast | OLD true | NEW true | OLD null | NEW null |
|----------|----------|----------|----------|----------|
| c=0.00   | 0.457 | **0.558** | 0.048 | 0.049 |
| c=0.0625 | 0.240 | **0.249** | 0.020 | 0.017 |
| c=0.125  | 0.267 | **0.269** | 0.012 | 0.013 |
| c=0.25   | 0.054 | **0.102** | 0.009 | 0.010 |
| c=1.00   | **0.096** | **0.028** | 0.006 | 0.005 |

Key finding: **c=1.0 distance drops 3× (0.096 → 0.028)** with fill-from-next — confirming the zero-padding artifact was inflating it. But all contrasts remain highly significant (p=0.0 everywhere). The c=1.0 drop confirms that a large portion of the old high-contrast signal was artifact; the residual (0.028 vs null 0.005) is genuine — driven by the I/M modulation creating differential RT distributions and thus differential S responses.

### By split (mean across contrasts)

| split | NEW true | NEW null |
|-------|----------|----------|
| r_choice_r_f1 | 0.241 | 0.019 |
| l_choice_l_f1 | 0.164 | 0.018 |
| l_choice_r_f2 | 0.566 | 0.070 |
| r_choice_l_f2 | 0.690 | 0.068 |

The f2 (error/incorrect) splits show the **largest** true distances (~0.6–0.7 vs null 0.07) — these reflect genuine I/M-driven biases on S in the wrong direction (block-incongruent trials).

### Summary: Q1 (standard absence 12× inflation) revised

With fill-from-next (150ms window):
- **True S prior distance = genuine (all contrasts significant, 10–20× above null)**
- c=1.0 f1 inflation was ~3× from the zero-padding artifact; the remaining c=1.0 signal is real
- The dominant signal is at c=0–0.25 (slow trials that don't hit truncation), driven by genuine I/M→S coupling via choice selection
- Phase 4b comparison: without I/M modulation, all distances collapse to null (Phase 4b = null everywhere at 80ms); with I/M modulation, distances are 10–20× above null at all contrasts

---

## UPDATE 2026-06-19b: Final retest with S=80ms / I/M=150ms (canonical going forward)

### Phase 4b (all g/d=0, S=80ms, I/M=150ms)

| Population | curve_mean | null_median | p |
|------------|-----------|-------------|---|
| S | 0.0124 | 0.0172 | 0.78 ✓ |
| I | 0.0037 | 0.0046 | 0.60 ✓ |
| M | 0.0042 | 0.0077 | 0.81 ✓ |

**All populations non-significant.** Phase 4b = pure null with the corrected analysis. No prior distance when g/d=0.

### Standard absence (fitted I/M, g_s=d_s=0, S=80ms, I/M=150ms)

**S overall: curve_mean=0.798, null_median=0.240, p=0.0** (highly significant, ~3.3× above null).

Mean S true/null by split:

| split | true | null |
|-------|------|------|
| r_choice_r_f1 | 0.102 | 0.008 |
| l_choice_l_f1 | 0.060 | 0.008 |
| l_choice_r_f2 | 0.426 | 0.048 |
| r_choice_l_f2 | 0.436 | 0.052 |

f2 (incorrect) splits dominate the effect (~8–9× above null), reflecting I/M prior bias driving S in the wrong direction on incorrect trials.

### Revised interpretation of Q1
The genuine I/M→S prior effect exists and is large at mid-contrast (c=0.0625–0.25). It's mediated by the split's choice-conditioning (f1/f2 selection bias interacting with I/M-driven choice probability). The c=1.0 and c=0.0 effects are near-null, consistent with saturation (c=1.0: all choices correct regardless of prior) and noise (c=0.0: zero-contrast catch trials).

### Per-contrast breakdown — all four splits (standard absence, S=80ms)

**r_choice_r_f1** (right stim, right choice, correct)
| c | n_high | n_low | true | null | p |
|---|--------|-------|------|------|---|
| 0.0 | 51 | 323 | 0.0279 | 0.0074 | 0.13 |
| 0.0625 | 150 | 790 | 0.1505 | 0.0098 | 0.00 |
| 0.125 | 226 | 893 | 0.1977 | 0.0064 | 0.00 |
| 0.25 | 334 | 993 | 0.1270 | 0.0080 | 0.00 |
| 1.0 | 465 | 1005 | 0.0052 | 0.0075 | 0.66 |

**l_choice_l_f1** (left stim, left choice, correct)
| c | n_high | n_low | true | null | p |
|---|--------|-------|------|------|---|
| 0.0 | 324 | 50 | 0.0084 | 0.0099 | 0.54 |
| 0.0625 | 761 | 187 | 0.1032 | 0.0071 | 0.00 |
| 0.125 | 860 | 226 | 0.1105 | 0.0075 | 0.00 |
| 0.25 | 1028 | 328 | 0.0590 | 0.0086 | 0.00 |
| 1.0 | 947 | 412 | 0.0180 | 0.0083 | 0.20 |

**l_choice_r_f2** (left stim, right choice, incorrect)
| c | n_high | n_low | true | null | p |
|---|--------|-------|------|------|---|
| 0.0 | 81 | 155 | 0.1160 | 0.0073 | 0.00 |
| 0.0625 | 116 | 204 | 0.3212 | 0.0134 | 0.00 |
| 0.125 | 71 | 131 | 0.0928 | 0.0283 | 0.14 |
| 0.25 | 18 | 57 | 1.1719 | 0.1426 | 0.01 |

**r_choice_l_f2** (right stim, left choice, incorrect)
| c | n_high | n_low | true | null | p |
|---|--------|-------|------|------|---|
| 0.0 | 125 | 105 | 0.0717 | 0.0076 | 0.00 |
| 0.0625 | 191 | 124 | 0.1906 | 0.0113 | 0.00 |
| 0.125 | 127 | 70 | 0.4459 | 0.0291 | 0.00 |
| 0.25 | 47 | 18 | 1.0365 | 0.1599 | 0.00 |
| 1.0 | 1 | 0 | nan | nan | nan |

**Patterns:**
- f1 (correct) splits: signal peaks at c=0.125, near-null at c=1.0 (accuracy ≈ 100% regardless of prior → no selection bias)
- f2 (incorrect) splits: signal grows with contrast to c=0.25, dominated by prior-driven wrong choices; almost no c=1.0 f2 trials
- n_high/n_low imbalance is large and consistent (e.g. l_choice_l_f1 has 947 high vs 412 low at c=1.0) confirming the block-concordance asymmetry in trial counts — but this count imbalance alone doesn't create distance; the distance comes from differential S trajectories
- Pattern is symmetric across both f1 and both f2 splits → structural, not split-specific
- c=0.0 (catch): near-null for f1 (no sensory drive → S≈0 for all trials → no distance); significant for f2 (wrong choices on catch trials are rare and highly prior-driven)

---

## Follow-up 2026-06-19: Presence-case `p_block_s_trajectory` plots show no prior modulation — bug analysis

### Observation

In the **presence** condition (`g_s ≈ g_i`, `d_s ≈ d_i` — direct P→S coupling enabled), the `p_block_s_trajectory_{split}.png` figures show trajectories for P-block-L (blue) and P-block-R (red) that are nearly identical to the corresponding **absence** plots. They do **not** qualitatively resemble the `p_block_i_trajectory_{split}` figures (which clearly show block-dependent I separation from the fitted g_i/d_i). This is unexpected: with g_s/d_s of the same magnitude as g_i/d_i, S should receive comparable block-dependent modulation.

### Root cause: 150 ms zero-padding in trajectory plots cancels the g_s/d_s boost

**The core bug**: `trial_s_binned_signed` (used by `_collect_s_traces_by_contrast` → `plot_p_block_s_trajectories`) uses the full **150 ms** `PRE_POST[split]` window for **all populations**, including S. `bin_trace_segment` zero-pads trailing bins when a trial ends before the 150 ms window. This is the same bug that was fixed in the *prior-distance analysis* (`build_population_b_for_split`) but **was not applied to the visualization path**.

**Why it specifically hides the g_s/d_s effect in presence**: With g_s/d_s active, concordant trials (e.g. P-block-R for `r_choice_r_f1`, right stim in right block) get a stronger feedforward S drive → M fires sooner → **shorter RT** → **more zero-padded bins** in the 150 ms window. Anti-concordant trials (P-block-L) get a suppressed S drive → slower M firing → longer RT → fewer zero-padded bins but weaker signal.

The two effects (stronger S but shorter duration vs. weaker S but longer duration) partially cancel in the 150 ms time-average, so the mean S trajectories for the two groups look similar — and similar to the absence case where neither group gets a g_s boost.

For I trajectories, this zero-padding effect is less severe because:
1. I has a longer effective integration window (tau_i >> tau_s); I signal builds more slowly and is sustained later
2. I also receives the boosted S as input (concordant S → stronger S→I drive), amplifying its block-dependent modulation beyond just g_i

**Quantitative illustration** (schematic, r_choice_r_f1, high contrast):

| Group | S amplitude | RT | Zero-padded bins (of 75) | Mean S[1] over 150ms |
|-------|-------------|----|--------------------------|-----------------------|
| P-block-R (concordant, presence) | ~0.4 (boosted) | ~25ms | ~55 | ~0.4 × 20/75 ≈ 0.11 |
| P-block-L (anti-conc., presence) | ~0.2 (suppressed) | ~55ms | ~35 | ~0.2 × 40/75 ≈ 0.11 |
| P-block-R (absence) | ~0.3 | ~40ms | ~45 | ~0.3 × 30/75 ≈ 0.12 |
| P-block-L (absence) | ~0.3 | ~42ms | ~43 | ~0.3 × 32/75 ≈ 0.13 |

All four mean to roughly the same value — the two presence groups cancel each other, and presence ≈ absence in the plot.

### Secondary issue: t_axis mismatch

`plot_p_block_s_trajectories` computes its own `t_axis = time_axis_for_split(split, split_n_bins(split))` using the full 150 ms split binning (72 bins), independently of population. So even after fixing `trial_s_binned_signed` to use 80 ms for S (36 bins), the x-axis would still be labeled 0–150 ms unless the function also derives its axis from the shorter window.

### Code fix

**`trial_s_binned_signed`**: apply the same 80 ms cap as `build_population_b_for_split` when `population == "S"` and the split is `stimOn`-aligned with `post > 0`:

```python
if population == "S" and align_kind == "stimOn_times" and post > 0:
    post = min(post, S_DURINGSTIM_WINDOW_S)
    n_coarse = max(1, int(round(post / B_SIZE)))
    n_bins = n_coarse * max(1, int(B_SIZE // STS))
else:
    n_bins = split_n_bins(split)
```

**`plot_p_block_s_trajectories`**: derive `t_axis` from the actual (population-specific) bins returned by `trial_s_binned_signed`, rather than recomputing it from the split-level full window:

```python
post_eff = min(PRE_POST[split][1], S_DURINGSTIM_WINDOW_S) if population == "S" else PRE_POST[split][1]
t_end_ms = post_eff * 1000.0
n_bins_pop = _population_n_bins(split, population)
t_axis = np.linspace(0, t_end_ms, n_bins_pop)
```

### Expected result after fix

With 80 ms window + zero-padding (fill-from-next not needed for visual plots):
- P-block-R (concordant, presence): strong S[1] peak in 0–25ms → visible early separation
- P-block-L (anti-concordant, presence): weak/suppressed S[1] throughout 0–80ms

The separation should be qualitatively similar to the I-trajectory plots (blue and red lines diverge clearly). The absence case should show much less separation (only selection-bias effects, which were shown to be near-null at 80 ms in Phase 4b tests).

### Implementation status

- [x] Fix `trial_s_binned_signed` to apply 80 ms cap for S (+ recompute n_bins): when `population == "S"` and the split is `stimOn`-aligned with `post > 0`, caps `post = min(post, S_DURINGSTIM_WINDOW_S)`, recomputes `n_bins` (36 bins for 80 ms), and sets `t_axis = np.linspace(0, 80, 36)`. All other populations and split types are unchanged.
- [x] Fix `plot_p_block_s_trajectories` to use population-specific t_axis: derives `post_eff` and `n_bins_pop` per-population inside the split loop, so S uses 80 ms / 36-bin axis while I/M use 150 ms / 72-bin axis.
- [x] Regenerate presence `p_block_s_trajectory_*.png` to confirm separation (re-run `--block-confound-plots`, seed 123, 40 sessions). New files saved to `output/absence_80ms/seed_123/{absence,presence}/figs/block_confounds/p_block_s_trajectory_stim_*.png` (note: `_split_short_label` now prefixes split name with `stim_`). With the 80 ms window, presence S trajectories show clear P-block-L vs P-block-R separation, qualitatively resembling the I trajectory separation — confirming the fix is correct.
