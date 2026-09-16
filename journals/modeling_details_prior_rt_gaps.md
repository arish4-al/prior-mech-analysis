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

**Status:** 2026-09-14 — full act-prior RT table (s101) is in
the dated entry below. Concordant matches on average; incongruent
is the wrong shape (too fast at |c|=1, too slow at ±0.0625).
Open: whether further RT/lapse/threshold machinery is worth it
for this paper. **Restore `g_i` floor `0.1`.**

**Code:** `prior_distance_I_M_both_alignments` / `loss_prior_effect` /
`loss_perf_with_data` in [`model_functions.py`](../model_functions.py);
Stage B driver [`scripts/run_fit_joint.py`](../scripts/run_fit_joint.py);
submit
[`scripts/submit_fit_stage_b_model_ablations.sh`](../scripts/submit_fit_stage_b_model_ablations.sh).
Eval / plots: `scripts/_tmp_im150_meancell_eval.py`,
`_tmp_im150_meancell_prior_dip.py`,
`_tmp_im150_meancell_actprior_rt.py`,
`_tmp_perf_rt_model_vs_data.py`,
`_tmp_gm0_mleak_eval.py`, `_tmp_gm0_mleak_actprior_rt.py`,
`_tmp_gm0_im150_eval.py`, `_tmp_gm0_im150_actprior_rt.py`,
`_tmp_im_to_m_gate_clamp.py`. Optional I→M gate:
`w_mi_off_until_ms` / `w_mi_off_prestim` (default off).
Data I/M by contrast: `scripts/build_mean_data_im_from_cache.py`
(cache path; do not use `build_mean_data_im_by_contrast.py` /
outdated `dmn/res/concat_*`). ORCD:
`scripts/submit_mean_data_im_from_cache.sh`. Overlay:
`scripts/_tmp_im_percontrast_overlay.py` (plots in s101
`im_percontrast_vs_data/`). Artificial M-CRF→RT test:
`scripts/_tmp_im_crf_warp_rt.py` (s101 `im_crf_warp_rt/`).

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

1. **Gate I→M until after the S peak** (most direct). *(clamp 09-11d:
   prestim-only fixes M40; hard 60/80 ms cutoff kills early M and
   tot, no data-height notch. Soft / fitted gate still open.)*
   `W_mi` / `g_m` off (or strongly attenuated) from prestim through
   ~80 ms post-stim. Must not wreck movement-aligned M or RT.
2. **Decay the prestim M charge at stimOn.** *(done 09-09j `mleak`:
   no notch; `W_mi` rose; late M dies. Drop the cap.)*
3. **Kill extra prior drive through the S peak.** *(done 09-09j at
   80 ms; **09-11c `gm0_im150`**: still no notch, and it **loses**
   eval tot at the window that can see the residual. Drop the freeze.)*
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

## 2026-09-09j — gm0 / mleak scored

16/16 `FIT_DONE`, mask `7|9|12|13` (`g_m`/`d_m`/`g_s`/`d_s` ≈ 0).
Shared-stim eval `bps=20` seed **12345**, stim from regular s101,
production prior window. Dump + overlay:
`models/stageB_hold_s89_gm0_mleak_eval/`
(`eval.json`, `actprior_rt.json`, `IM_overlay_s12_s34_s101.png`).
Per-run prior/S plots in each run dir.

| arm | best eval tot | median | prior med | M70 / M80 / M150 med |
|-----|-------------:|-------:|----------:|---------------------:|
| **gm0** | **0.967** (s7) | **1.022** | 0.150 | 0.106 / 0.108 / 0.121 |
| regular | 0.999 (s34) | 1.030 | 0.193 | 0.116 / 0.117 / 0.135 |
| mleak | 0.980 (s34) | 1.117 | 0.189 | 0.106 / 0.104 / **0.094** |
| data M | — | — | — | **0.080 / 0.104 / 0.145** |

`gm0` `W_mm` stayed 0.24–0.29 (same basin as regular); `W_mi` 0.47–0.60.
`mleak` sat in `[0.11, 0.14]` and **raised `W_mi` to 0.63–0.85**. Four
`mleak` seeds collapsed `g_i` (`s7`, `s12`, `s45` ≈ 0; `s89` = 9.4) —
the open `1e-12` floor again.

No 60–70 ms notch on either arm. `gm0` still ramps 40→80 (best tot s7:
M 0.085 / 0.102 / 0.106 / 0.118). `mleak` often **kills the late rise**
(s89 M150 = 0.057; s333 = 0.067; s45 = 0.074).

Act-prior RT (10×20, same seed):

| arm | seed | perf | RTcomb | split (con / inc) |
|-----|-----:|-----:|-------:|------------------:|
| gm0 | 12 | 0.923 | 0.618 | **0.303** (0.76 / **−0.19**) |
| gm0 | 303 | 0.882 | 0.682 | 0.228 (0.72 / −0.30) |
| gm0 | 34 | 0.902 | 0.657 | 0.155 (0.78 / −0.51) |
| gm0 | 7 | 0.868 | 0.352 | −0.58 (0.66 / −1.91) |
| gm0 | 333 | 0.808 | −1.12 | −2.68 (−0.40 / −5.11) |
| mleak | 101 | 0.755 | **0.773** | −0.37 (0.86 / −1.68) |
| mleak | 89 | 0.919 | 0.331 | 0.057 (0.47 / −0.38) |
| mleak | 34 | 0.820 | 0.586 | −0.62 (0.80 / −2.13) |
| mleak | 7† | 0.866 | −0.56 | −1.67 (0.00 / −3.46) |

† `g_i≈0`. Inc R² still negative on every seed. `gm0` s12 is the
least-bad inc (−0.19 vs meancell s12 −1.04 / regular s101 −0.21).
Collapsed-`g_i` `mleak` seeds flatten concordant RT.

**Keep:** freeze `g_m`/`d_m` (`gm0`) as a cheap eval-tot win. **Drop:**
the `W_mm≤0.15` cap. The M pause still needs a delayed I→M drive
(option 1), not a leakier M. Restore `g_i` floor 0.1 before the next
campaign.

## 2026-09-09k — 09-09j was the 80 ms window

`gm0` / `mleak` left `prior_window_ms` unset (test-6 keep list). The
M notch / late climb are after 80 ms, so that campaign does not
answer the shape question. `gm0` arm now sets `PRIOR_WINDOW_MS=150`
and tag `stageB_hold_s89_gm0_im150`. No `mleak` rerun.

## 2026-09-10 — full-BWM I/M RMS; late-window contrast

ORCD job landed in alyx
`manifold/mean_data_im_from_cache/` (594 / 696 insertions with I/M
cells; 21,594 I / 12,487 M; 9.0 min; `--regs sc`). All-contrast
recovers `fit_targets` (r = 0.994–0.999, scale 0.96–1.03).

`all` sits below every per-contrast curve because it is RMS of the
**pooled-trial** cell mean (exactly the n-weighted average of the
five `μ_c`). RMS is convex, so that is lower than the mean of the
noisier per-contrast RMS — not a pooling bug.

Mean RMS after bin 15 looks almost flat (~3% I, ~5% M). That average
hides a **late** contrast ramp. Choice-L, vs c=0.0625:

| epoch | I stim | M stim | I choice | M choice |
|-------|-------:|-------:|---------:|---------:|
| 0–50 ms | −0.03 | −0.04 | −0.07 | −0.05 |
| 50–100 | +0.03 | +0.08 | −0.05 | −0.02 |
| 100–150 | **+0.21** | **+0.37** | +0.08 | +0.13 |
| last 25 | +0.23 | +0.43 | +0.13 | +0.19 |

c=1 / c=0.0625 in the last 25 ms: I stim **1.07**, M stim **1.13**,
I choice 1.04, M choice 1.05. Mid contrasts sit in between. **c=0
is the exception** — offset *up* in every epoch (~+0.09), not a
late-only ramp.

Early 0–80 ms is flat or slightly inverted. A scalar 5-point I/M
CRF on the whole 150 ms would pin the wrong thing. The late
stim-aligned rise is the same clock as easy RT (data |c|=1 ≈
0.19–0.21 s): the last 50 ms of the 150 ms stim window is
peri-move on easy trials and still pre-move on hard ones. That is
movement bleed / earlier commit, not I/M sensory gain. Choice-
aligned late rise is the last 25 ms before movement (commit),
larger on easy trials as expected.

Do **not** add a Stage B by-contrast I/M amplitude term. Discordant
RT is still an S-drift / bound problem.

## 2026-09-10b — overlay regular s101 vs per-contrast data

Choice L, shared stim `bps=20` seed 12345, regular s101. Data RMS
vs model channel `|Δ|` (twin axes; units do not match). Plots:
openalyx `weights_run_fj_stageB_hold_s89_regular_mask12-13_s101/im_percontrast_vs_data/`.

Last ~25 ms, each series / its own c=0.0625:

| series | I stim | M stim | I choice | M choice |
|--------|-------:|-------:|---------:|---------:|
| data c=1 / c=0.0625 | 1.065 | 1.106 | 1.024 | 1.035 |
| model c=1 / c=0.0625 | **4.19** | **4.79** | 1.66 | 1.49 |
| data c=0 vs 0.0625 | +0.09 up | +0.09 up | +0.08 up | +0.07 up |
| model c=0 vs 0.0625 | **below** | **below** | **below** | **below** |

Model n per contrast (left choice): 130 / 122 / 127 / 134 / 56
(c=1 … 0).

**Trend that matches:** late-window rank among c>0 is the same
(easy > mid > hard). Choice-aligned traces also share the late
climb into commit.

**Trend that does not:**
1. Stim-aligned model I/M stay near 0 until ~50 ms, then fan out
   by contrast. Data stay stacked (and c=0 sits *above* the stack)
   until a shallow late lift.
2. The model's late CRF is ~4–5× on stim I/M and ~1.5× on choice
   I/M; data are ~1.07 / 1.11 and ~1.02 / 1.04.
3. Zero contrast is inverted: data offset up in every epoch;
   model is the lowest curve.

So the fitted I/M already have *too much* contrast dependence, not
too little. Same conclusion as 2026-09-10: do not add a 5-point
I/M amplitude term. The remaining M-shape gap is still the
stim-aligned pause through the S peak, not a missing CRF.

## 2026-09-11 — high-c “best fit” is not missing S CRF

The twin-axis overlay of **raw** RMS (~3) vs model `|Δ|` (~0–1)
makes c=1 look like the match: both purple curves hit the top of
their own axis, and every other data curve still sits on the ~3
floor while low-c model stays near 0. That is not a scale the
loss uses. Traj SSE baseline-subtracts the first stim bin, then
compares the residual to model `|Δ|`.

Last-25 residual (data late − t0) vs model, choice L:

| | I stim | M stim | I choice | M choice |
|--|-------:|-------:|---------:|---------:|
| data resid c=1 / c=0.0625 | 2.26 | 2.03 | 1.37 | 1.25 |
| model c=1 / c=0.0625 | **4.19** | **4.79** | 1.66 | 1.49 |
| model / data resid at c=1 | 1.23 | 1.33 | 1.19 | 1.07 |
| model / data resid at c=0.0625 | 0.67 | 0.57 | 0.98 | 0.90 |
| model / data resid at c=0 | 0.45 | 0.44 | 0.87 | 0.87 |

Pin c=1 (the visual): every other contrast undershoots, worst at
c=0. That is the same fact as a **too-steep** CRF, not a too-shallow
one. More S contrast dependence would widen c=1 vs hard/zero
further; c=1 is already 7–33% high.

What low/zero contrast is missing is a **contrast-independent**
(or weakly contrast-dependent) I/M component — prior / prestim
I→M charge / recurrent floor — not more S. There is no S→M
synapse. At c=0, S≈0 and the model residual is the smallest
curve; data residual at c=0 is still ~0.22 I / 0.36 M.

## 2026-09-11b — comparable overlay + per-contrast nSSE

Replot in traj-loss units: data RMS − first stim bin vs model
`|Δ|`, same axis. Stim nSSE skips 15 bins; choice uses the full
window (`m_pre_weight=1`). Regular s101, shared stim 12345 / 20.
Plots: s101 `im_percontrast_vs_data/`
(`im_percontrast_overlay`, `im_percontrast_late25`,
`im_percontrast_nsse`).

nSSE by |contrast|:

| term | 1.0 | 0.25 | 0.125 | 0.0625 | 0.0 |
|------|----:|-----:|------:|-------:|----:|
| I stim L+R | 0.228 | 0.186 | 0.379 | 0.518 | **0.757** |
| M stim L+R | **0.135** | 0.272 | 0.441 | 0.547 | **0.771** |
| I choice L+R | 0.064 | 0.134 | 0.083 | 0.056 | 0.085 |
| M choice L+R | 0.077 | 0.080 | 0.129 | 0.124 | 0.186 |
| I+M all L | 0.253 | 0.366 | 0.488 | 0.713 | 1.003 |
| I+M all L+R | **0.505** | 0.671 | 1.032 | 1.245 | **1.799** |

Choice L last-25 residual (same units as the overlay):

| | I stim | M stim | I choice | M choice |
|--|-------:|-------:|---------:|---------:|
| data − t0 c=1 | 0.47 | 0.84 | 0.53 | 0.91 |
| model c=1 | 0.52 | 1.00 | 0.50 | 0.76 |
| data − t0 c=0 | 0.23 | 0.39 | 0.38 | 0.66 |
| model c=0 | 0.10 | 0.16 | 0.28 | 0.47 |

Stim-aligned: c=1 is the only contrast where the model rise is
at or above the data; hard/zero stay near 0 until late and
underestimate. Choice-aligned is much closer (nSSE ≲ 0.13) at
every contrast. The loss that is actually getting worse as
contrast drops is **stim I/M**, not choice I/M.

## 2026-09-11c — `gm0_im150` scored (as fitted, 150 ms)

8/8 `FIT_DONE`, `prior_window_ms=150`, mask `7|9|12|13`
(`g_m`/`d_m`/`g_s`/`d_s` ≈ 0). Tag
`stageB_hold_s89_gm0_im150` so the 80 ms `gm0` dirs are not
reused. `g_i` stayed high (136–200); no floor collapse. `W_mm`
0.255–0.286 (usual basin). `W_mi` 0.45–0.58.

Shared-stim eval `bps=20` seed **12345**, stim from regular s101,
**scored at 150 ms** (same window as the fit). Dump + overlay:
`models/stageB_hold_s89_gm0_im150_eval/`
(`eval.json`, `actprior_rt.json`, `IM_overlay_s12_s34_s101.png`).
Per-run prior/S plots in each `gm0_im150` run dir.

| arm | best eval tot | median | prior med | M70 / M80 / M150 med |
|-----|-------------:|-------:|----------:|---------------------:|
| regular @150 | **1.086** (s303) | **1.162** | **0.284** | 0.117 / 0.117 / 0.137 |
| meancell @150 | 1.089 (s12) | 1.263 | 0.355 | 0.135 / 0.134 / 0.156 |
| **gm0_im150** | 1.128 (s101) | 1.347 | 0.446 | **0.138 / 0.147 / 0.193** |
| data M | — | — | — | **0.080 / 0.104 / 0.145** |

Best `gm0_im150` s101: traj 0.337 + prior 0.293 + `L_S` 0.497 =
**1.128**. Prior is the term that lost vs regular@150 (0.243 on
regular s101). Fit losses (own-stim) are not comparable across
arms; s333 recorded 0.925 but eval-fair is 1.340.

M shape (shared stim). Data 40→70 is a **notch** (0.078 → 0.080
after a 50 ms bump at 0.096). Every `gm0_im150` seed still
**ramps** 40→70 by +0.018 to +0.028:

| t (ms) | data M | gm0 s101 | gm0 s12 | meancell s12 | regular s101 |
|-------:|-------:|---------:|--------:|-------------:|-------------:|
| 0 | 0.065 | 0.085 | 0.082 | 0.089 | 0.064 |
| 40 | 0.078 | 0.120 | 0.114 | 0.123 | 0.088 |
| 60 | 0.085 | 0.137 | 0.130 | 0.136 | 0.099 |
| 70 | **0.080** | **0.143** | **0.136** | 0.140 | 0.105 |
| 80 | 0.104 | 0.146 | 0.140 | 0.141 | 0.108 |
| 150 | 0.145 | 0.163 | 0.170 | 0.152 | 0.140 |

No 60–70 ms pause. Early M is **higher** than regular@150, not
lower. Late M **overshoots** (median 0.193 vs data 0.145 /
regular 0.137). s89 is the only seed with a tiny 70→80 dip
(0.129 → 0.127) and then a flat late rise to 0.147 — still no
S-peak notch, and eval tot is 1.354.

The 80 ms `gm0` “eval-tot win” (09-09j, 0.967) does not survive
when the prior SSE can see 80–150 ms. Freezing `g_m`/`d_m` and
letting the 150 ms loss pull the late climb just makes M taller
through the S-peak window.

Act-prior RT (10×20, same seed). Plots in each run’s
`psychometric_model_vs_data_actprior/`.

| seed | perf | RTcomb | split (con / inc) |
|-----:|-----:|-------:|------------------:|
| **101** | 0.758 | **0.791** | −0.17 (0.86 / **−1.26**) |
| 12 | 0.628 | 0.676 | −0.07 (0.47 / −0.65) |
| 89 | 0.754 | 0.727 | −0.99 (0.84 / −2.94) |
| 45 | 0.426 | 0.730 | −1.54 (0.73 / −3.96) |
| 7 | 0.554 | 0.623 | −2.11 (0.92 / −5.36) |
| 333 | 0.513 | 0.464 | −2.61 (0.90 / −6.37) |
| 34 | 0.423 | −0.10 | −5.27 (0.73 / −11.7) |
| 303 | 0.137 | 0.316 | −5.16 (0.95 / −11.7) |

Inc R² still negative on every seed. Best pooled RT is s101
0.791 (meancell s12 0.834; 80 ms `gm0` s12 split +0.303 / inc
−0.19). Concordant stays decent; incongruent is unchanged or
worse than the 80 ms freeze.

**Drop:** `g_m`/`d_m` freeze as an M-shape / 150 ms keep-list
item. It was a cheap 80 ms tot win on a window that cannot see
the residual. **Still drop:** `W_mm≤0.15`. **Next M lever:**
option 1 (gate I→M until after the S peak). Option 4 (split /
up-weight 40–80 ms M SSE) is the loss-side alternative. Restore
`g_i` floor 0.1 before the next campaign (this run did not
collapse, but the floor is still open).

## 2026-09-11d — matching I/M CRF does not fix inc RT

Artificial test on regular s101 (10×20, seed 12345, act-prior).
Commit is first-passage of `tanh(M0−M1)` to θ_c / θ_d (S×P). I
is not the bound variable. Per-contrast late-25 M stim residual
(data − t0 vs model `|Δ|`) gave:

| c | data | model | scale | add |
|--:|-----:|------:|------:|----:|
| 1.0 | 0.80 | 0.95 | 0.84 | −0.15 |
| 0.25 | 0.58 | 0.45 | 1.29 | +0.13 |
| 0.125 | 0.45 | 0.35 | 1.29 | +0.10 |
| 0.0625 | 0.40 | 0.31 | 1.28 | +0.09 |
| 0.0 | 0.40 | 0.28 | 1.42 | +0.12 |

Replay of unwarped M recovers baseline RT (median |Δ| = 1 step;
R² within ~0.02). Then:

| arm | perf | RTcomb | split (con / inc) |
|-----|-----:|-------:|------------------:|
| baseline | 0.875 | **0.787** | 0.337 (**0.848** / **−0.209**) |
| replay | 0.869 | 0.790 | 0.347 (0.842 / −0.184) |
| scale to data M | 0.875 | **−0.230** | −0.275 (−0.196 / **−0.360**) |
| add (data−model) | 0.875 | −0.200 | −0.268 (−0.137 / **−0.409**) |

Boosting low-c / shrinking high-c M to match the data CRF
**flattens** the RT vs contrast curve (hard trials hit θ
earlier). Concordant collapses; incongruent gets worse. The
inc hole is that discordant low-c does not *slow*; a larger
low-c `|M|` does the opposite.

So fitting the choice×contrast I/M traces better would not
reduce the RT mismatch. Those curves pool con+inc and are not
the bound-crossing clock. Plots: s101 `im_crf_warp_rt/`
(`rt_split_baseline`, `rt_split_scale`, `rt_split_add`).

## 2026-09-11d — I→M gate clamp (regular s303 @ 150 ms)

Not a refit. Hard-zero `W_mi`/`g_m` on the current best 150 ms
eval tot (regular s303, 1.086). Shared stim `bps=20` seed 12345.
`prior_window_ms=150`. Optional params (default off):
`w_mi_off_until_ms`, `w_mi_off_prestim`. Driver
`scripts/_tmp_im_to_m_gate_clamp.py`. Plots + dump in s303
`im_to_m_gate_clamp/`.

| tag | tot | M0 | M40 | M70 | M80 | M150 | inc RT |
|-----|----:|---:|----:|----:|----:|-----:|-------:|
| data | — | 0.065 | 0.078 | **0.080** | 0.104 | 0.145 | — |
| baseline | **1.086** | 0.077 | 0.107 | 0.127 | 0.128 | 0.146 | −2.54 |
| prestim | 1.095 | 0.005 | **0.080** | 0.116 | 0.121 | 0.149 | −2.63 |
| post60 | 1.401 | 0.074 | 0.029 | 0.049 | 0.068 | 0.143 | −2.80 |
| post80 | 1.983 | 0.071 | 0.027 | 0.013 | 0.014 | 0.151 | −3.30 |
| full60 | 1.609 | 0.000 | 0.000 | 0.035 | 0.056 | 0.152 | −2.74 |
| full80 | 2.169 | 0.000 | 0.000 | 0.000 | 0.003 | 0.149 | −3.27 |

Prestim-only is almost free on tot and puts M40 on the data.
Then it **ramps harder** (40→70 +0.036 vs baseline +0.020).
M0 collapses (0.005 vs data 0.065). Not a notch.

Post-stim cutoffs hold M down through the S peak, then the
late climb still recovers (M150 ≈ 0.15 on every arm). Amplitude
in 40–80 ms is **too low**, not notched-at-data. Eval tot
jumps +0.3 to +1.1 (traj + prior). `post80` has a 50→70 dip
(−0.009) from a floor of 0.022 — the wrong height.

RT (10×20, same seed): pooled RT is a bit better on the long
gates (post80 0.617 vs 0.576) because con improves; **inc gets
worse**. Gating does not close problem 2.

**Do not launch** a hard 80 ms I→M cutoff. The lever is real
(prestim `W_mi` is why early M is high). A campaign would need
a **soft / fitted** attenuation that keeps M near ~0.08 through
60–70 ms, not a binary off. Option 4 (40–80 M SSE) is still
the loss-side alternative.

## 2026-09-11e — inc RT dip at 0 is model, not a plot swap

Act-prior con/inc is **plotted correctly**. Data: stim side
(`contrastLeft` finite → L; IBL zeros are L=0/R=NaN or
R=0/L=NaN, never both) × action-kernel 0.8 = left. Model:
`trial_sides` +1 = R × binarized trial-mean P (`P_L−P_R < 0` →
+1 = R). Concordant R² **0.85** would not survive a sign flip.
True-block vs act-prior agree on 86% of model trials; the RT
shape is the same either way. n is fine (data inc ~5.5–6.6k /
bin; model inc ~270–330).

**Data** discordant is a smooth inverted-U, peak at 0, a bit
above concordant at every contrast. No dip at 0; ±0.0625 is
not high.

| signed c | data n con/inc | data RTcon | data RTinc | model RTcon | model RTinc |
|---------:|---------------:|-----------:|-----------:|------------:|------------:|
| −1 | 14415 / 6246 | 0.174 | 0.191 | 0.130 | **0.109** |
| −0.125 | 13241 / 5996 | 0.260 | 0.292 | 0.252 | **0.397** |
| −0.0625 | 12076 / 5768 | 0.325 | 0.353 | 0.324 | **0.431** |
| 0 | 12204 / 5677 | 0.413 | 0.426 | 0.403 | 0.429 |
| +0.0625 | 11850 / 5486 | 0.352 | 0.366 | 0.345 | **0.467** |
| +1 | 13285 / 6599 | 0.186 | 0.209 | 0.129 | **0.109** |

The “dip at 0 / huge low-c RT” is **model incongruent**:
peaks at ±0.0625 (esp. +0.0625 = 0.467), sits at 0.429 at
c=0 (≈ model con 0.403), and is **too fast** at ±1 (0.109 vs
data 0.19–0.21; 15% of |c|=1 inc have RT < 80 ms and are
dropped). Negative inc R² is this W-shape, not a label swap.

Why 0 dips: with S≈0, perceived S×P counts as concordant
(`0·ΔP ≥ 0`), so labeled-inc zeros are prior-only — same
clock as con. ±0.0625 inc has real weak S against
prior-charged M, so first-passage is late. θ_d < θ_c
(0.39 vs 0.77) makes **easy** inc even faster than con,
which is the wrong sign vs data (data inc is slower at
|c|=1 too). Correct-only inc looks worse (R² −1.23): at
c=0 only 19% of model inc are “correct” (they followed the
phantom stim side against the prior).

## 2026-09-14 — act-prior RT table (s101) and paper scope

Overlay (data con/inc solid, model dashed): openalyx
`weights_run_fj_stageB_hold_s89_regular_mask12-13_s101/psychometric_model_vs_data_actprior/rt_split_model_vs_data.{png,svg}`.
Not `psychometric_model_vs_data/` (true-block). Regular s101,
10×20, seed 12345, data = action-kernel α=0.2. RT in seconds
(0.08–2 s). Assignment check: 2026-09-11e.

| signed c | data inc | model inc | data con | model con |
|---------:|---------:|----------:|---------:|----------:|
| −1 | 0.191 | 0.109 | 0.174 | 0.130 |
| −0.25 | 0.243 | 0.273 | 0.217 | 0.198 |
| −0.125 | 0.292 | 0.397 | 0.260 | 0.252 |
| −0.0625 | 0.353 | 0.431 | 0.325 | 0.324 |
| 0 | 0.426 | 0.429 | 0.413 | 0.403 |
| +0.0625 | 0.366 | 0.467 | 0.352 | 0.345 |
| +0.125 | 0.301 | 0.413 | 0.285 | 0.261 |
| +0.25 | 0.265 | 0.259 | 0.235 | 0.197 |
| +1 | 0.209 | 0.109 | 0.186 | 0.129 |

Concordant is the average inverted-U. Incongruent is not: too
fast at |c|=1, too slow at ±0.0625 / ±0.125, and a dip at 0
that the data do not have.

**Scope for the paper.** This is not a complete account of
why the animal moves when it does. It does not include lapse
structure (impatience, zoning out, and other mistakes),
contrast dependence of the action threshold, or contrast
dependence of prior modulation — among other things that
clearly matter for single-trial behavior. What it can capture
is the **overall** psychometric / RT pattern on average
(concordant RT already does). One could take the model further
on those missing pieces (arguably many steps further). Whether
that is worth it for the **current** paper is the decision:
the remaining incongruent RT shape is real, but it sits
outside the mechanisms this draft is using to make the
prior / I / M claim.

## 2026-09-14b — S-prior 150 ms stim×choice, correct metric

Regular 150 ms meancell is done; 80 ms regular and all `full` + S
arms were still `‖mean_c Δ‖`. Wired a rerun of **one** arm: `full`
(`g_s`/`d_s` free) + unsplit-80 S + **150 ms stim×choice I/M**, tag
`stageB_hold_s89_full_im150_meancell`. **Scored 09-14c:** 8/8, best
fair s101 **1.269**, median **1.359** (old-metric im150 1.292 /
1.707). No M notch; several S-success seeds kill late M. Plots +
act-prior RT in each run dir (s12 inc R² **+0.21** but `g_i≈0`;
s101 inc still −3.21). Tables:
[fit_gs_ds_with_im.md](fit_gs_ds_with_im.md) 2026-09-14c.
