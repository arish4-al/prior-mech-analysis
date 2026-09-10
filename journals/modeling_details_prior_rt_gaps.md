# Prior-curve dips and discordant RT

**Scope:** remaining **shape** gaps after tests 1–6
([modeling_details_revisions.md](modeling_details_revisions.md)). Two
targets, both vs BWM act-prior data:

1. **M prior-distance vs time** — stim-aligned model M ramps through the
   S-peak window instead of pausing / notching with the data. I is close
   enough; leave it. Movement-aligned M already rises into commit.
2. **Discordant / incongruent RT** — concordant RT vs signed contrast is
   already decent; incongruent R² is negative on every Stage B regular
   seed, and the contrast-dependence looks wrong.

**Not in scope:** retinal Stage A / `avg_mean_R` (except as a frozen
`L_S` sidecar); Harris / Bayes priors; fit *speed*; changing canonical
prior-distance analysis defaults (80 ms S, fill-from-next-ITI,
contrast-matched null).

**Status:** 2026-09-09i — ORCD gather is a **single** job (no shards):
laptop 7 caches ~7.5 s; ~700 insertions → ~30–60 min on Lustre,
`--time=2:00:00`. Submit
`scripts/submit_mean_data_im_from_cache.sh`. 09-09d `gm0`/`mleak`
still pending. **Restore `g_i` floor `0.1`.**

**Code:** `prior_distance_I_M_both_alignments` / `loss_prior_effect` /
`loss_perf_with_data` in [`model_functions.py`](../model_functions.py);
Stage B driver [`scripts/run_fit_joint.py`](../scripts/run_fit_joint.py);
submit
[`scripts/submit_fit_stage_b_model_ablations.sh`](../scripts/submit_fit_stage_b_model_ablations.sh).
Eval / plots: `scripts/_tmp_im150_meancell_eval.py`,
`_tmp_im150_meancell_prior_dip.py`,
`_tmp_im150_meancell_actprior_rt.py`,
`_tmp_perf_rt_model_vs_data.py`.
Data I/M by contrast: `scripts/build_mean_data_im_from_cache.py`
(cache path; do not use `build_mean_data_im_by_contrast.py` /
outdated `dmn/res/concat_*`). ORCD:
`scripts/submit_mean_data_im_from_cache.sh`.

---

## Aims

Close the two shape residuals **without** undoing the test 1–6 keep
list or the `mean_c ‖Δ‖` metric. Rank on the usual shared-stim eval tot
(traj + prior + `L_S`; `bps=20`, stim seed **12345**, stim from regular
**s101**) **and** on act-prior RT (10×20, same seed; data =
`behavior_actprior.npy`). A change that buys prior-curve shape by
wrecking incongruent RT (or the reverse) is not a win.

Do **not** chase single noisy bins, and do **not** retune I to deepen
its 50–80 ms wiggle. Durable **M** pattern: stay low/flat through
~40–70 ms (notch at the S peak), then climb after 80 ms. Durable RT
pattern: incongruent should slow at low contrast the way the data do,
not invert or collapse.

---

## Problem 1 — M prior-distance shape (I is good enough)

### A. Stim-aligned M: model ramps through the S-peak pause

BWM (`fit_targets/data_act_block_duringstim.npy`, 72 bins / 150 ms),
act-prior 4-split `mean_c ‖Δ‖`. I shown only as context — leave it.

| t (ms) | data I | data M | s12 M | s34 M | regular s101 M |
|-------:|-------:|-------:|------:|------:|---------------:|
| 0 | 0.050 | 0.065 | 0.089 | 0.077 | 0.064 |
| 40 | 0.064 | 0.078 | **0.123** | 0.106 | 0.088 |
| 50 | 0.049 | **0.096** | — | — | — |
| 60 | 0.066 | **0.085** | — | — | — |
| 70 | 0.063 | **0.080** | — | — | — |
| 80 | 0.058 | 0.104 | **0.141** | 0.119 | 0.108 |
| 110 | 0.070 | 0.137 | 0.132 | 0.111 | 0.106 |
| 150 | 0.076 | 0.145 | 0.152 | 0.139 | 0.140 |

Data M: low/flat to ~30 ms, a brief 50 ms bump, **notch at the S peak
(60–70 ms: 0.096 → 0.080)**, then the late climb. The 130 ms wiggle
(0.107) is likely noise / early-commit mix — do not chase it.

Model M never takes that pause. s12 is already high at stimOn and
keeps climbing through 40–80 ms (0.09 → 0.14). Regular s101 matches
*amplitude* at 0/80/150 ms better but is still a smooth ramp (no
60–70 ms notch). s12’s late 110 ms dip is the wrong time.

**Not** short-RT fill: s12 **0.8%** RT &lt; 80 ms (median 168 ms).
**Not** “I trough too shallow”: data M’s 50→70 ms drop (**0.016**) is
*larger* than data I’s 40→80 change (0.006). M is not a lagged copy of
I. Movement-aligned M already matches the late rise (data 0.067 →
0.146; s12 0.064 → 0.157) — do not break that.

**Mechanism (M, not I):** there is no S→M synapse. M is

`leak + W_mm J M + (W_mi J + g_m P_gain) @ I + d_m P_offset`

(`tau_m` frozen at 20 ms). Prior gap in M is I’s gap, filtered and
held, plus whatever `g_m`/`d_m` add. `P_offset` is on for the last
100 ms of the ITI, so `W_mi~0.5` **charges M before stimOn**. Then
`W_mm` holds that charge while S common-drives I; a small I wiggle
cannot pull M down. `g_m` (s12 **0.79**) and `d_m` (s45 **2.8**) add
prior drive *through* the S peak and make early M worse. Most other
seeds have `g_m≈d_m≈0` and still ramp — prestim I→M leak + `W_mm` is
enough. The 150 ms energy-normalized M SSE rewards the late climb, so
the optimizer is not punished for being high at 40–80 ms.

### B. Movement-aligned start (secondary; I-side leftover)

`duringchoice` t=−150 ms is `RT−150`. Legacy 80 ms pre-movement hid
the fade. Data I drops 0.076 → 0.038; s12 I already matches that
start (0.037). **M** at t=−150 is fine (data 0.067, s12 0.064). Keep
production as last 80 ms if we re-score; do not slow M decay to fill
an I hole.

---

## Problem 2 — discordant / incongruent RT

Act-prior (α=0.2) × stim side. **con** = stim matches prior; **inc** =
discordant. Data: BWM `firstMovement − stimOn`. Model: commit time,
prior = binarized trial-average P.

Concordant RT R² is already in the **0.75–0.90** band on usable regular
/ meancell seeds. Incongruent is **negative** on every Stage B regular
seed we have scored. Last year’s `WEIGHTS_REL` is the only reference
with a slightly positive inc R² (**+0.18**).

The overlay (`rt_split_model_vs_data`) is the shape complaint: at low
signed contrast the model discordant curve does not slow the way the
data do (too fast, or the wrong contrast dependence). Two thresholds
(`θ_c > θ_d`) were kept in test 4 **because** they help this split;
they are not enough. Test 5 (3× pre-action M) was an attempt to make
the optimizer see |M| near bound on hard / incongruent trials — it
rescued some seeds’ *pooled* RT without pulling `θ_d` up, and
**did not** fix inc.

---

## What we already tried

Pointer: full tables live in
[modeling_details_revisions.md](modeling_details_revisions.md).
Keep / drop from that campaign:

| Test | Change | Outcome |
|------|--------|---------|
| 1 | `P_offset` always on through the ITI | **Keep the gate** (off until 100 ms pre-stim) |
| 2 | Drop I/M ITI zero-penalty | Did not help eval tot as hypothesized |
| 3 | Open / shrink `W_pp` box | **Keep 2.5 s floor** `[0.496, 0.49999]` |
| 4 | One action threshold | **Keep two** (`θ_c ≠ θ_d`) |
| 5 | `m_pre_weight=3` | Best pooled RT 0.772 (s303) vs regular s101 **0.787**. Inc still negative. **Keep `m_pre_weight=1`** |
| 6 | `prior_window_ms=150` (stim×choice) | Did not beat regular on production eval tot. Extra loss is prior. **Keep production window unset** |

Other arms on the same Stage B stack:

- **`im150stim` / `stimonly`** (09-08): stim-only I/M stratum. Do not
  beat regular on production tot. Opening `g_i` to `1e-12` made
  stim-only collapse worse. **Restore `g_i` floor 0.1.**
- **`full` + S sidecar** ([fit_gs_ds_with_im.md](fit_gs_ds_with_im.md)):
  unfreeze `g_s`/`d_s`, fit unsplit-80 S prior-distance (13 stim_early
  regions). Best full s34 fair **1.057** via `d_s≈26`, `g_s≈0`. Not a
  drop-in for the mid-trough / inc-RT aims; a small `g_s` is still a
  candidate for the *early I bump* (see next).

### Metric bug, not a dynamics bug (09-08f)

BWM I/M targets are **`mean_c ‖μ₊−μ₋‖`** (within-cell Euclidean, then
mean over 4 stim×choice splits). The model did the other order:
equal-weight the four cell means, then one L2 (`‖mean_c Δ‖`). After S
peaks (~68 ms) I/M rotate onto choice; error-cell `Δ` anti-aligns with
correct cells and the pooled L2 **drops** even though each cell’s own
prior gap is still growing. That was the post-80 ms *fall* on
stim×choice im150 — not same-threshold collapse, not concordant-faster
RT.

`prior_distance_I_M_both_alignments` (numpy + torch) now does
`mean_c ‖Δ‖` for `stim_choice` (I/M) and `stim` (S / im150stim).
`choice_distance_*` (plot-only) matches the 4-cell choice L–R combine.

This **fixed the post-80 ms fall**. It did **not** create the M pause
at 60–70 ms (problem 1A) or the incongruent RT hole (problem 2).

### `im150_meancell` rerun (2026-09-09)

8/8 `FIT_DONE`, `prior_window_ms=150`, regular mask, `include_stim=False`.
Dirs:
`weights_run_fj_stageB_hold_s89_im150_meancell_regular_mask12-13_s{7,12,34,45,89,101,303,333}/`.
`g_i` bounds were still **`[1e-12, 200]`** — s101 collapsed.

Shared-stim eval (current `mean_c` code; baseline re-scored the same
way). Dump: `models/stageB_hold_s89_im150_meancell_eval.json`.

| arm | best eval tot | median |
|-----|-------------:|-------:|
| baseline (production) | **0.999** (s34) | **1.030** |
| meancell → production | 1.029 (s34) | 1.213 |
| baseline → 150 | 1.086 (s303) | 1.162 |
| meancell as fitted (150) | 1.089 (s12) | 1.263 |

s12 as fitted: traj 0.332 + prior 0.260 + `L_S` 0.497 = **1.089**.
`L_S` is a flat **0.496–0.497** on every seed (frozen retinal). Stim-
aligned I/M now **rise** 80 → 150 ms on every usable seed. That was the
metric fix working. I shape is good enough. Model M still ramps through
the 40–80 ms window (table in problem 1A).

`L_S` is **not** a pooling-order bug (right-stim per-contrast SSE vs
`avg_mean_R`; `sse *= c * 2` zeros 0-contrast amplitude; reported
`gof_S~0.46` is unweighted and dragged by c=0 R² ≈ −2). Left-stim NaN
buckets would fail-close the whole `L_S`; not hit on this stim.

### Act-prior RT on meancell (2026-09-09)

Same 10×20 protocol. Plots in each run’s
`psychometric_model_vs_data_actprior/`. Dump:
`models/stageB_hold_s89_im150_meancell_actprior_rt.json`.

| seed | perf | RTcomb | RTspl | con / inc |
|-----:|-----:|-------:|------:|----------:|
| 7 | 0.633 | 0.639 | −1.54 | 0.90 / −4.15 |
| **12** | 0.713 | **0.834** | −0.07 | 0.84 / −1.04 |
| 34 | 0.739 | 0.357 | −1.80 | 0.76 / −4.55 |
| 45 | 0.467 | 0.607 | −2.42 | 0.83 / −5.90 |
| 89 | 0.661 | 0.669 | −1.07 | 0.82 / −3.09 |
| 101† | 0.888 | 0.024 | −0.56 | 0.31 / −1.49 |
| 303 | 0.719 | 0.005 | −2.67 | 0.70 / −6.27 |
| 333 | 0.641 | 0.596 | **0.213** | 0.41 / 0.01 |

† `g_i≈0`. Best pooled RT s12 **0.834** (regular s101 **0.787**, old
im150 s101 **0.874**, `WEIGHTS_REL` **0.730**). Only s333 has a
non-negative *split* RT. Incongruent is still the hole.

---

## What we have **not** tried (next)

For **stim-aligned M** — hold prior-distance down through ~40–70 ms,
then let it climb. Do not retune I. Do not chase the 130 ms bin.
Smoke on s12 (worst early-M: `g_m=0.79`) and s34 / regular s101
(`g_m≈0`, still no notch).

1. **Gate I→M until after the S peak** (most direct). `W_mi` / `g_m`
   off (or strongly attenuated) from prestim through ~80 ms post-stim,
   same family as the existing `P_offset` ITI gate. M stays near the
   ITI/prestim floor through the S transient, then tracks choice-
   aligned I. Must not wreck movement-aligned M or RT.
2. **Decay the prestim M charge at stimOn.** *(clamp 09-09c: no notch
   on frozen I/`W_mi`; `W_mm=0.10` kills late M. Refit `mleak` pending.)*
   Faster effective M leak over 0–80 ms: weaker `W_mm`. `tau_*` stay
   20 ms — W already sets the effective timescale.
3. **Kill extra prior drive through the S peak.** *(clamp 09-09c:
   zeroing `g_m` on s12 does not drop M40.)* Cap `g_m` near 0 and keep
   `d_m` from sitting at 2.8. Necessary, not sufficient.
4. **Prior SSE that sees 40–80 ms M.** Split M’s term (0–80 vs
   80–150) or up-weight 40–80 ms so a high-and-smooth M is expensive.
   I’s term stays as-is. Test 5 (`m_pre_weight=3`) does the *opposite*
   (cares about pre-movement |M|).
5. **Heavier (new dynamics):** a brief common-mode S→M or
   S-dependent `W_mi` shrink at the S peak, so M is washed more than
   I. Only if (1)–(4) cannot make a 60–70 ms notch.

Avoid: slowing M decay to fill the movement-aligned I fade; opening
`g_i` to `1e-12`; deepening I’s trough in the hope M follows (data M
notch is larger than I’s).

For **incongruent RT**: still open. Test 4 (two thresholds) and test 5
(3× M-pre) did not close it. A W_mi gate (1) will change time-to-
bound — score inc RT on the same smoke, not after an 8-seed campaign.

---

## 2026-09-09 — journal opened

Recorded the two shape aims, the `mean_c` metric fix, the
`im150_meancell` score (eval tot + prior-dip numbers + act-prior RT),
the `L_S` audit, and the untried trough / inc-RT levers. Tests 1–6 stay
in [modeling_details_revisions.md](modeling_details_revisions.md).

## 2026-09-09b — dip is M, not I

I stim-aligned is close enough (s12 already has the 40 → 80 → 110
wiggle). The residual is **M**: data pauses / notches at the S peak
(60–70 ms), model ramps through 40–80 ms. M is not a lagged I (data M
drop 0.016 vs I 0.006). Working mechanism: prestim `W_mi` charges M,
`W_mm` holds it, `g_m`/`d_m` when nonzero add drive through the S
peak; no S→M path.

## 2026-09-09c — try 2+3 (clamp; no τ change, no I→M gate)

`tau_s/i/p/m` are all **20 ms**. Leak is `W_mm` (and the other W’s).
Did **not** touch τ, the I→M gate (1), prior-SSE split (4), or S→M (5).

**Clamp** on existing finals, shared stim `bps=20` seed 12345:
`g_m=d_m=1e-12`, `W_mm` ∈ {fitted, 0.20, 0.15, 0.10}. Driver
`scripts/_tmp_mleak_clamp.py`. Plots:
`models/stageB_hold_s89_mleak_clamp/{meancell_s12,meancell_s34,regular_s101}_M_wmm_sweep.png`.

| model | `W_mm` | M0 | M40 | M70 | M80 | M150 |
|-------|-------:|---:|----:|----:|----:|-----:|
| data | — | 0.065 | 0.078 | **0.080** | 0.104 | 0.145 |
| s12 `g_m=0` fit W | 0.270 | 0.090 | 0.123 | 0.140 | 0.141 | 0.152 |
| s12 | 0.15 | 0.092 | 0.120 | 0.133 | 0.139 | 0.171 |
| s12 | 0.10 | 0.098 | 0.125 | 0.135 | 0.132 | **0.116** |
| s34 `g_m=0` fit W | 0.277 | 0.077 | 0.106 | 0.121 | 0.119 | 0.139 |
| s34 | 0.15 | 0.077 | 0.098 | 0.101 | 0.093 | 0.120 |
| s101 `g_m=0` fit W | 0.255 | 0.064 | 0.087 | 0.103 | 0.107 | 0.136 |
| s101 | 0.15 | 0.062 | 0.082 | 0.098 | 0.102 | 0.149 |

Zeroing `g_m` on s12 does **not** drop M40 (0.123 with `g_m=0.79` or
0). The early ramp is prestim `W_mi`, not `g_m`. Weakening `W_mm` on
frozen I/`W_mi` never makes the 60–70 ms data notch; `W_mm=0.10` on
s12 **kills the late rise**. s34 @ 0.15 is the least-bad clamp (tiny
70→80 dip, late M a bit low).

**Refit arms** (other weights free, so `W_mi` can drop): freeze
`g_m`/`d_m`/`g_s`/`d_s` (mask `7|9|12|13` → `LOG_ZERO` ≈ 0). Two
boxes so the clamp’s “`g_m`=0 helped, `W_mm` cut did not” can be
tested in a real search:

| arm | `W_mm` | tag |
|-----|--------|-----|
| `gm0` | `[0.10, 0.40]` (usual) | `stageB_hold_s89_gm0` |
| `mleak` | `[0.10, 0.15]`, warm-start 0.15 | `stageB_hold_s89_mleak` |

`set_w_mm_native_bounds` must sync `fit_weights.W_MM_NATIVE_BOUNDS`.
Do not sbatch from the agent:

```bash
PARTITION=pi_fiete ABLATIONS="gm0 mleak" \
  bash scripts/submit_fit_stage_b_model_ablations.sh
```

A delay-from-I-to-M (option 1, fitted lag) stays on the later list.

## 2026-09-09d — gm0 vs mleak

Clamp read: zeroing `g_m`/`d_m` did most of the visible work; cutting
`W_mm` did not make the 60–70 ms notch and can kill the late rise.
Split the refit: `gm0` = freeze only; `mleak` = freeze + `W_mm≤0.15`.

## 2026-09-09e — inc RT diagnostics (no refit)

Driver: `scripts/_tmp_inc_rt_diagnostics.py`. Regular **s101**, **s12**,
`WEIGHTS_REL`. RT/timeouts from the existing 10×20 act-prior caches
(stim seed 12345); `|action|` at stimOn from 3 fresh sessions. Plots in
each run’s `psychometric_model_vs_data_actprior/inc_rt_diag_*.png`.
Dump: `models/stageB_hold_s89_inc_rt_diag/inc_rt_diag.json`.

The overlay complaint was “too fast, or the wrong contrast dependence.”
The second is the one that is true — and the slope is the **opposite**
of “does not slow at low c.”

### Ruled out

| Hypothesis | Result |
|------------|--------|
| Already past `θ_d` toward prior at stimOn | **No.** `P(past)=0` on all three. Mean `action` toward prior **0.041–0.045** vs `θ_d` **0.39–0.43**. Remaining to prior bound **0.35–0.38**. |
| Timeouts dropping slow inc trials | **No.** Inc timeout **0.5–1.4%**. |
| Fast errors pulling `split_all` down | **No.** Inc errors are *slower* than corrects (s101 dw 0.377 vs dc 0.306). `correct_split` inc R² is **worse** (s101 **−1.23** vs split_all **−0.21**; WEIGHTS_REL **−0.57** vs **+0.18**). |

Short RT (<80 ms, dropped by the mask) is almost only **|c|=1 incongruent**
(s101 14–17%, s12 9–12%, WEIGHTS_REL **25–26%**). Mid/low c: 0%. That
trims the fastest easy-inc commits; the remaining |c|=1 dc is still
~110 ms.

Start-point bias is tiny on every seed (`toward_prior ≈ 0.04`). Almost
all of the con/inc remaining-distance gap is `θ_c ≈ 0.76` vs `θ_d ≈ 0.40`.

### The hole: too-steep discordant contrast–RT

Pooled inc RT (s) vs signed contrast. Data = act-prior `behavior_actprior.npy`.

| \|c\| | data | s101 | s12 | WEIGHTS_REL |
|------:|-----:|-----:|----:|------------:|
| 1 | 0.19–0.21 | **0.11** | **0.11** | **0.11** |
| 0.25 | 0.24–0.27 | 0.26–0.27 | 0.28 | 0.29–0.30 |
| 0.125 | 0.29–0.30 | **0.40–0.41** | **0.44** | 0.37–0.39 |
| 0.0625 | 0.35–0.37 | **0.43–0.47** | **0.47–0.51** | 0.40–0.42 |
| 0 | 0.43 | 0.43 | 0.47 | 0.40 |

- **|c|=1:** model inc is faster than data *and* faster than model con
  (s101 con 0.13). `θ_d < θ_c` plus a strong stim.
- **0.0625–0.125:** model overshoots (too slow). s12 is the worst peak;
  WEIGHTS_REL is the least-bad mid-c overshoot — that is why it is the
  only positive inc R², not because it slows low-c more.
- Correct-only (dc) is steeper still (s101 dc 0.11 at ±1, **0.50** at
  0.0625 vs data 0.19 / 0.34). Including slower dw *helps* split_all.

Concordant stays in the usual good band. Data inc error rate 0.27;
models 0.28–0.35.

### What this does to the next levers

- A **higher `θ_d` at low c** (the usual “caution when weak”) would
  worsen the mid-c overshoot. If anything, a contrast schedule would
  want the **opposite** slope: higher `θ_d` at |c|=1 (slow the 110 ms
  easy-inc) and/or lower `θ_d` at 0.0625–0.125.
- Raising `θ_d` globally slows both ends — helps |c|=1, hurts the
  already-too-slow mid.
- A global non-decision / motor delay would lift the 110 ms floor
  toward data ~190 ms and also lift the mid-c peak further above data.
- WEIGHTS_REL vs s101 still have the same `θ` and the same tiny
  prestim |M|; the remaining gap is **drift vs contrast** (retinal
  `τ_a`, CRF), not the bound schedule. Score that clamp next if we
  keep going, not a 10-param `θ(c)` fit.

## 2026-09-09f — data I/M by contrast (notebook RMS)

How pooled I/M targets are made (`model_test.ipynb` / cell that writes
`mean_data_results.npy`; older `get_data_for_fitting.py` is the flat
I-stim / M-choice version):

1. Load ONE `dmn/res/concat_act_normFalse.npy`.
2. Keep cells whose Beryl acronym is in `int_regs` (I, 81) or
   `move_regs` (M, 26) — same lists as `fit_targets/mean_data_results.npy`.
3. Slice `concat` with `sum_for_key` on `len` (dict order).
4. Per key, store **RMS across cells**
   `sqrt(nansum(x², 0) / n / T_BIN)`, `T_BIN=0.0125`. Notebook does
   **not** subtract `rms[0]` for I/M. Keys: 8 stim (96 bin) + 8 choice
   (72 bin), all contrasts pooled.

S already uses `concat_by_contrast_act_noshuffle` and the same RMS.
`concat_PETHs(..., vers='concat_by_contrast_act')` already builds
`{base}_{c}` for all 8+8 keys. The all-cell noshuffle file is **not**
on this laptop.

Driver: `scripts/build_mean_data_im_by_contrast.py`. Fallback (concat
is still the real PETH; shuffle is only `distance_controls`):

| node | file | cells → after `int_regs`/`move_regs` | keys |
|------|------|--------------------------------------|------|
| I | `concat_by_contrast_act_normFalse_shuffleTrue_integrator.npy` | 1446 → **555** | correct-only 4 stim + 4 choice × 5 c |
| M | `concat_by_contrast_act_normFalse_shuffleTrue_move_init.npy` | 3567 → **968** | same |

Missing vs production: error-cell keys (`stimLbLcR`, `sLbLchoiceR`, …).
Output (with the concat, not repo `figs/`):
`dmn/res/mean_data_im_by_contrast/mean_data_results_by_contrast.npy`
+ `im_crf_amp.png`, `{I,M}_traces_by_contrast.png`.

Choice-L collapse, mean RMS after bin 15:

| \|c\| | I stim | I choice | M stim | M choice |
|------:|-------:|---------:|-------:|---------:|
| 1 | 4.86 | 4.83 | **6.49** | 5.66 |
| 0.25 | 4.86 | 4.85 | 6.06 | 5.59 |
| 0.125 | 4.86 | 4.84 | 5.80 | 5.60 |
| 0.0625 | 4.92 | 4.88 | 5.65 | 5.64 |
| 0 | 5.19 | 5.23 | 5.85 | 5.77 |

**I stim is almost flat** (slightly *higher* at c=0). **M stim rises
with contrast** (5.65 → 6.49). That CRF is **not usable** — those
concat files are outdated (correct-only keys, restricted rasters).
Do not put this fallback npy into `fit_targets/`.

## 2026-09-09g — I/M RMS from the insertion cache

The 09-09f “I flat, M rises with contrast” pattern does not make
sense as a sensory CRF, and the concat sources are stale. Switched
to the current real-data path
([realdata_pipeline_efficiency.md](realdata_pipeline_efficiency.md)):

- `manifold/insertion_cache/{eid_probe}.npy` already holds spikes +
  saturation-masked trials.
- Driver `scripts/build_mean_data_im_from_cache.py`: filter
  `int_regs` / `move_regs`, apply act-prior (`α=0.2`), bin **once**
  per alignment (`bin_spikes2D`, `T_BIN=0.0125`, `sts=0.002`), then
  slice the 8 stim + 8 choice cells × 5 contrasts **and** a pooled
  all-contrast condition. No `nrand`, no `d_var`.
- Windows match current I/M analysis: stim `[0, 0.15]`, choice
  `[0.15, 0]` → 72 bins (not the old 200 ms / 96-bin concat).
  `--stim-post 0.2` if we need notebook-length stim.
- RMS is the notebook formula
  `sqrt(nansum(x²) / n / T_BIN)` streamed as `sum(x²)` / `n`.
- Output next to the cache:
  `manifold/mean_data_im_from_cache/`
  (`mean_data_results_by_contrast.npy`, `im_crf_amp.png`,
  `{I,M}_traces_by_contrast.png`, `allcontrast_vs_fit_targets.png`).

**Local run (this laptop):** only the 7-insertion Goal-2 smoke
cache exists (`alyx…/manifold/insertion_cache/`, ~365 MB). 6 of 7
had I/M cells → **197 I / 130 M**. Wall ~2.5 s after load. Choice-L
collapse, mean RMS after bin 15:

| \|c\| | I stim | I choice | M stim | M choice |
|------:|-------:|---------:|-------:|---------:|
| 1 | 1.36 | 1.36 | 2.47 | 2.44 |
| 0.25 | 1.67 | 1.68 | 3.06 | 3.03 |
| 0.125 | 2.46 | 2.43 | 2.88 | 2.73 |
| 0.0625 | 2.46 | 2.55 | 3.54 | 3.62 |
| 0 | 2.70 | 2.68 | 3.12 | 3.21 |
| all | 2.07 | 2.09 | 2.54 | 2.48 |

This is **not** a BWM CRF (and does not match
`fit_targets/mean_data_results.npy`: I stim r ≈ 0.04, M choice
r ≈ 0.84; scale ~0.7). It is only a pipeline check. Error keys at
`|c|=1` are almost empty (as expected). Do **not** put this npy in
`fit_targets/`. Do **not** add a Stage B by-contrast I/M term until
the full-cache CRF exists.

Full BWM cache is on ORCD (~22 GB). Command (user pastes; agent
does not sbatch / ssh):

```bash
conda activate iblenv
python scripts/build_mean_data_im_from_cache.py \
  --cache-dir "$ONE_CACHE/manifold/insertion_cache"
```

Plots land in `$ONE_CACHE/manifold/mean_data_im_from_cache/`.

## 2026-09-09h — I/M lists vs stim×choice; drop act-prior

`fit_targets` / notebook I/M (`int_regs` 81, `move_regs` 26) vs
current `data/stimchoice_act_regtype_regions_p_mean_c_0.01.csv`
(union of duringstim ∨ duringchoice labels):

| | current | in old list | missed | extras |
|--|--:|--:|--:|--:|
| I integrator | **60** | 60 | **0** | **21** |
| M move | **23** | 23 | **0** | **3** |

No I/M swap. Extras are not current I/M:

- I extras (21): 18 unlabeled (AIp, AIv, BLA, BMA, CA3, COPY, DTN,
  EPd, Eth, ICB, LA, LHA, LSc, PC5, PRP, RSPagl, SCs, SNc); 3
  stim-early only (OP, PO, SAG).
- M extras (3): AUDd, COAp, CUN (unlabeled).

A few old-I regions are stim-early in one window but still
integrator in the other (DCO, IC, NOT, PAG, PB, PRNr, TRN) — those
stay in the 60. ZI is stim-early duringstim and move duringchoice;
it stays in M.

Default gather now uses the **60 / 23** current lists
(`--regs sc`). `--regs fit_targets` restores 81 / 26.

Act-prior removed. The fit (`_data_mean_and_baseline`) already
collapses stim×prior at fixed choice, so we only store choice L/R
× contrast (and `all`). No 8-key notebook cells.

7-cache smoke with `--regs sc`: 92 I / 122 M cells. Choice-L RMS
after bin 15 is nearly flat in I (1.93–2.14) and M (2.15–2.51).
Still not a BWM CRF.

## 2026-09-09i — ORCD job, no shards

Laptop: 7 insertions in ~7.5 s (~1 s each) after filtering to I/M
cells and binning two alignments only. Full BWM ~700 caches → ~12 min
on SSD. Lustre I/O 2–5× → **~30–60 min**, worst ~90 min. Memory is
one insertion at a time (~30–130 MB cache). **One job**, 8G / 2 CPU /
`--time=2:00:00`. Shards would need a merge and are not worth it
(no `nrand`).

Default partition is `mit_preemptable` (`--requeue` via
`sbatch_defaults.sh`).

```bash
bash scripts/submit_mean_data_im_from_cache.sh
```

Output: `$ONE_CACHE/manifold/mean_data_im_from_cache/`. Agent does
not sbatch.
