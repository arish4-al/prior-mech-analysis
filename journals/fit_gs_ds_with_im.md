# Fit g_s / d_s together with I/M prior-mod parameters

**Scope:** a Stage B / joint variant that leaves **all** prior-mod gains free
(`g_i, g_m, d_i, d_m, g_s, d_s`) instead of the current exclusive masks
(regular freezes `g_s`/`d_s`; sensory freezes I/M). The S target for that
variant is a **real-data prior-distance curve**, not the retinal `avg_mean_R`
mean-activity curves.

**Not in scope:** changing I/M traj targets, `avg_mean_R` / Stage A, Harris /
Bayes priors, or the default regular/sensory freeze masks.

**Status:** 1e-12-floor campaign scored 2026-09-09. Best fair tot **full
s34 = 1.057** (S **0.028** via `d_s≈26`, `g_s≈0`, `g_i=185`). All four
arms **32/32** local.

**Code:** curve builder
[`scripts/build_s_prior_curve_unsplit80.py`](../scripts/build_s_prior_curve_unsplit80.py);
S SSE gated by `include_stim` in
[`loss_prior_effect`](../model_functions.py); joint driver
[`scripts/run_fit_joint.py`](../scripts/run_fit_joint.py)
(`--mtype full` → freeze none). Sidecar
`fit_targets/data_act_block_duringstim_s_unsplit80.npy`.

---

## Why

Regular Stage B freezes `g_s`/`d_s` ≈ 0 (mask `12|13`). Sensory freezes
I/M gains (mask `6|7|8|9`). Neither arm asks whether a modest direct P→S
coupling can coexist with the fitted I/M prior mods.

To fit `g_s`/`d_s` we need an S prior-distance target. The notebook path
(`loss_prior_effect` + `reload=True` → `load_group` → `data_act_block_*.npy`)
already stored `r_stim`, but:

1. `stim_regs` were hardcoded visual areas (`VISpm`, `FRP`, `VISal`) with
   `is_stim=True` → `alpha=1.0` (included whether or not they were
   prior-significant).
2. The S SSE term in `loss_prior_effect` was later hardcoded to **0**
   (numpy path). Torch still scored S if `r_stim` was present.
3. Split-conditioned duringstim prior maps had **no** stim-classified
   (regtype 0) FDR hits, so there was no principled S-region pool.

Unsplit 80 ms act-prior **label shuffle within stim side**
(`act_block_duringstim_{l,r}`) has **84 FDR @ 0.01**
([structured nulls](structured_nulls_choice_lr.md) 08-17b / 09-01). The
S-curve pool is **not** those 84: it is their intersection with SC
**sensory** regions (`sc_duringstim_regtype` 0 = stim or 0.1 =
stim_early; 23 regions in
`data/stimchoice_act_regtype_regions_p_mean_c_0.01.csv`).

---

## Region set (2026-09-02; corrected same day)

Sources: alyx `meta/table_act_block_earlystim_80ms_p_mean.csv` (`unsplit`
/ `shuffle` / `80ms_earlystim`, `p_mean_c ≤ 0.01`) **∩** repo
`data/stimchoice_act_regtype_regions_p_mean_c_0.01.csv` sensory labels.

| Choice | Value |
|--------|--------|
| Window | first 38 of 72 bins (`t ≤ 80` ms of the 150 ms curve) |
| Prior | act (action-kernel) |
| Conditioning | stim-side unsplit (`act_block_duringstim_{l,r}`) |
| Null | simple label shuffle (not Harris, not sticky) |
| α | BH-FDR **0.01** on the unsplit-80 map |
| Sensory | duringstim regtype **0 or 0.1** (stim / stim_early) |

Strict stim (regtype 0) is only **VISpm**, which is **not** in the 84.
The intersection is the **13** stim_early regions that clear unsplit-80
FDR@0.01:

CA1, DCO, LGd, LP, PAG, PB, PO, PRNr, SAG, TEa, TRN, VISam, ZI.

**2026-09-07:** per-region and cell-weighted combined prior-distance
plots (shuffle overlay, 80 ms) are in alyx
`manifold/figs/earlystim80_sensory_prior/`
(`scripts/plot_earlystim_sensory_prior_distance.py`).

Not used: the other 71 FDR hits (integrators / move / unclassified),
`has_stim` mixed regions, or the old hardcoded `VISpm`/`FRP`/`VISal`.

---

## How the old S curve was built

Notebook (`paper-brain-wide-map/model_test.ipynb` / repo `model_test.ipynb`):

1. `plot_combined_onetype(..., 'stim', ['act_block_duringstim'])` for a
   stim ∩ prior-sig list (historically empty / unused).
2. Fallback `stim_regs = ['VISpm', 'FRP', 'VISal']`.
3. `loss_prior_effect(..., reload=True)` → `load_group(regs, timeframe,
   is_stim=True)` → cell-weighted `(obs − mean(nulls))` over the
   **4-split** `act_block_duringstim` combine →
   `data_act_block_duringstim.npy` (`r_int`, `r_move`, `r_stim`).

Canonical copies live in [`fit_targets/`](../fit_targets/). `r_stim`
there is 72 bins (150 ms) from those three visual regions. `L_S` in
joint/retinal is a **different** object (`avg_mean_R`: right-stim mean
S activity vs contrast), not prior distance.

New curve uses the same `load_group` weighting (`obs − mean(nulls)`,
then `/ n_splits`), but on the **unsplit** combine and the **80 ms**
prefix. The 2026-09-02 first write missed `/ n_splits` (fixed 2026-09-07).

---

## Variant

| Label | `--mtype` | `--freeze` | Meaning |
|-------|-----------|------------|---------|
| regular | `regular` | `12\|13` | I/M free; `g_s`/`d_s` ≈ 0 (unchanged) |
| sensory | `sensory` | `6\|7\|8\|9` | `g_s`/`d_s` free; I/M ≈ 0 (unchanged) |
| **full** | `full` | *(none)* | I/M **and** `g_s`/`d_s` free |

`include_stim=True` adds S prior nSSE to `loss_prior_effect` (duringstim
only; duringchoice S stays 0). Default remains `False` so regular /
sensory `L_w` is unchanged. Torch is gated the same way (it previously
scored leftover `r_stim`).

`L_S` vs `avg_mean_R` is **not** removed — retinal alignment stays.

Polish for `full` uses the existing `prior` refine set
`[6, 8, 10, 11, 12, 13]` (`g_i, d_i, θ, g_s, d_s`) ∩ train mask.

---

## 2026-09-02 — journal + S curve + full mask

First pass used all **84** FDR hits. Corrected the same day to
**FDR ∩ stim/stim_early = 13**. Rebuilt the cell-weighted unsplit-80
curve (simple correction = obs − mean nulls). **13 / 13** requested;
**9,504** cells (`nclus` from `act_block_duringstim_l`). 38 bins.

| curve | bins | min | max | end |
|-------|-----:|----:|----:|----:|
| S (unsplit 80 ∩ sensory, 13 regs) | 38 | **0.072** | **0.102** | 0.102 |
| discarded 84-region pool | 38 | 0.185 | 0.245 | 0.245 |
| old `r_stim` (VISpm/FRP/VISal, first 38) | 38 | −0.009 | 0.159 | — |
| I target first 38 (unchanged) | 38 | 0.038 | 0.075 | — |

The 13-region S curve is smaller than the 84-region mix (those extra
hits were mostly integrator/move) and stays strictly positive.

Wrote (overwrote the 84-region sidecar):

- alyx `meta/data_act_block_duringstim_s_unsplit80.npy` (+ PNG)
- repo `fit_targets/data_act_block_duringstim_s_unsplit80.npy` (fit copy)

Did **not** overwrite `fit_targets/data_act_block_duringstim.npy`
(`r_int` / `r_move` / old `r_stim`).

Submit (user pastes; agents do not run Slurm):

```bash
INCLUDE_STIM_PRIOR=1 VARIANTS="full:" OUT_TAG=stageB_hold_s89_full \
  SEEDS="7 12 34 45 89 101 303 333" \
  bash scripts/submit_fit_stage_b_sharded.sh
```

Empty freeze after `full:` is `masknone`. The slurm wrapper must not
rewrite an empty `FREEZE` back to `6|7|8|9`.

---

## 2026-09-07 — `/ n_splits` on the S sidecar

I/M fit targets come from `load_group(..., correction='simple')`: combined
`regde` is a **sum** of split Euclidean curves, then divide obs+nulls by
`len(splits)` (4 for split-conditioned duringstim), then cell-weighted
`obs − mean(nulls)`. The 09-02 S sidecar did the shuffle-mean subtract but
**not** `/ n_splits` (unsplit l+r → 2), so amplitude was ~2× the I/M plot
scale (min/max **0.072 / 0.102** vs I **0.038 / 0.075**).

Rebuilt with `/ 2`. Same 13 regions / 9,504 cells / 38 bins.

| curve | bins | min | max | end |
|-------|-----:|----:|----:|----:|
| S (unsplit 80 ∩ sensory, `/ 2`) | 38 | **0.036** | **0.051** | 0.051 |
| I target first 38 (unchanged) | 38 | 0.038 | 0.075 | — |
| M target first 38 (unchanged) | 38 | 0.047 | 0.103 | — |

Overwrote alyx `meta/` npy+PNG and `fit_targets/` copy. Regenerated regular
s101 `prior_effects` with the scaled S overlay (I/M 150 ms; S first 80 ms).

---

## 2026-09-08 — full campaign (openalyx, 8/8)

Local copies: openalyx `models/`
`weights_run_fj_stageB_hold_s89_full_full_masknone_s{7,12,34,45,89,101,303,333}/`.
All **8/8 `FIT_DONE`**, `fit_status=ok`, `include_stim=true`, `mask=none`,
`prior_stratum` unset (I/M stim×choice; model S forced `stratum_s=stim`).

**Eval:** `bps=20`, stim seed **12345**, stim from baseline regular
**s101**, nested `fit_targets/` + unsplit-80 S sidecar. Dump:
`models/stageB_hold_s89_full_eval.json`. Regular on the same seeds is
scored with `include_stim=True` so S nSSE is visible even though it was
not in that arm’s fit. JSON `final_loss` is own-stim — not comparable.

Fair tot = traj + I/M prior + S prior nSSE + retinal `L_S`.
Fair without S = production regular objective.

| seed | full rec | traj | I/M | S | L_S | fair | w/o S | g_s | d_s | g_i |
|-----:|---------:|-----:|----:|--:|----:|-----:|------:|----:|----:|----:|
| 7 | 1.866 | 0.393 | 0.255 | 0.676 | 0.496 | 1.820 | 1.144 | 1.87 | 0.32 | 184 |
| 12 | 1.496 | 0.479 | 0.106 | 0.241 | 0.447 | 1.274 | **1.032** | 27.7 | 65.4 | 190 |
| 34 | 1.641 | 0.410 | 0.301 | 0.474 | 0.476 | 1.661 | 1.187 | 0.60 | 12.9 | 166 |
| 45 | 1.378 | 0.489 | 0.315 | 0.045 | 0.422 | 1.272 | 1.227 | 0.32 | 2.60 | 37 |
| 89 | 1.664 | 0.410 | 0.153 | 0.706 | 0.498 | 1.766 | 1.060 | 3.47 | 0.06 | 116 |
| **101** | 1.360 | 0.378 | 0.250 | **0.018** | 0.429 | **1.076** | 1.057 | 37.3 | 45.7 | **0.15** |
| 303 | 2.252 | 0.356 | 0.460 | 0.079 | 0.482 | 1.377 | 1.298 | 50.5 | 27.2 | 187 |
| 333 | 1.882 | 0.357 | 0.367 | 0.034 | 0.434 | 1.191 | 1.157 | 0.17 | 53.5 | 96 |

Regular S nSSE (same stim, g_s≈0) is **0.56–0.75**. Production tot
without S: regular best **1.015** (s333), median **1.051**.

| arm | best fair | median fair | best w/o S | median S nSSE |
|-----|----------:|------------:|-----------:|--------------:|
| full | **1.076** (s101) | 1.326 | 1.032 (s12) | **0.160** |
| regular (S scored) | 1.642 (s303) | 1.726 | **1.015** (s333) | 0.680 |

S-success (nSSE &lt; 0.1): **s101, s45, s333, s303**. Two routes —
large g_s+d_s (s101 37/46, s303 51/27) or g_s near the 0.1 floor plus
offset (s45 0.32/2.6, s333 0.17/54). Failed S (still ~0.68): s7, s89.
Partial: s12 (0.241), s34 (0.474).

Tradeoff: best S fit **s101** collapsed `g_i` to 0.15 (regular 196).
s45 dropped `g_i` to 37. s303 kept `g_i=187` but I/M prior rose to
0.46. `g_m`/`d_m` stayed ~0 (free in DE/CMA, not in default polish).
`L_S` on S-success seeds is intact or slightly better (0.42–0.43 vs
regular ~0.50).

---

## 2026-09-08b — `g_*`/`d_*` floor 1e-12 + I/M-window arms

Native floors were mixed: `g_s` **0.1**, `d_s`/`d_i` **1e-5**,
`g_i`/`g_m`/`d_m` **1e-12**. That clipped a hybrid `g_s≈0` warm start
to 0.1 and blocked a true near-zero S coupling. All six now share
**1e-12**:

| param | old lo | new lo | hi |
|-------|-------:|-------:|---:|
| `g_i`, `g_m`, `g_s` | 1e-12 / 1e-12 / **0.1** | **1e-12** | 200 |
| `d_i`, `d_m`, `d_s` | **1e-5** / 1e-12 / **1e-5** | **1e-12** | 100 |

`d_i` is the weights-v2 box (`fit_weights._log_bounds_weights_v2`);
`g_s`/`d_s` are joint-only (`NATIVE_BOUNDS`).

Submit wrapper
[`scripts/submit_fit_stage_b_full_s_prior.sh`](../scripts/submit_fit_stage_b_full_s_prior.sh):
`VARIANTS=full:` + `INCLUDE_STIM_PRIOR=1`. `FORCE=1` replaces
`stageB_hold_s89_full_*`. New arms use new OUT_TAGs so they do not
overwrite regular im150 / im150stim / stimonly.

| arm | I/M window | I/M stratum | OUT_TAG |
|-----|------------|-------------|---------|
| `full` | legacy T=72 / `plot_window=80` | stim×choice | `stageB_hold_s89_full` (replace) |
| `im150` | 150 ms | stim×choice | `stageB_hold_s89_full_im150` |
| `im150stim` | 150 ms | `stim` | `stageB_hold_s89_full_im150stim` |
| `stimonly` | legacy | `stim` | `stageB_hold_s89_full_stimonly` |

Model S stays `stratum_s=stim` (2-split sidecar) on every arm. Same 8
seeds. User pastes (agents do not run Slurm):

```bash
PARTITION=mit_preemptable FORCE=1 \
  bash scripts/submit_fit_stage_b_full_s_prior.sh
```

---

## 2026-09-08c — 1e-12 floor + I/M-window arms (32/32)

Local copies: openalyx `models/`
`weights_run_fj_stageB_hold_s89_full_{full,im150,im150stim,stimonly}_full_masknone_s{7,12,34,45,89,101,303,333}/`.
**32/32 `FIT_DONE`**, `fit_status=ok`, `include_stim=true`. Flags match
the wrapper: `full` both unset; `im150` `prior_window_ms=150`;
`im150stim` 150 + `prior_stratum=stim`; `stimonly` `prior_stratum=stim`.
(`im150stim` landed first under `models/new/`, then in `models/`.)

**Eval:** same protocol as the morning campaign — `bps=20`, stim seed
**12345**, stim from regular **s101**, nested `fit_targets/` + unsplit-80
S sidecar, `include_stim=True`, model S `stratum_s=stim`. Driver
`scripts/_tmp_full_s_prior_eval.py`. Dump:
`models/stageB_hold_s89_full_s_prior_1e12_eval.json`. Regular on the
same seeds is re-scored so S nSSE is visible. JSON `final_loss` is
own-stim — not comparable. Rank on **as-fitted** fair tot
(traj + I/M + S + `L_S`). im150 / stimonly I/M terms are different
objectives; production-window rescores are in the dump.

`nfin=1` in every dir (FORCE replace; no leftover 0.1-floor finals).

### As-fitted summary

| arm | best fair | median fair | best S | S-success | best w/o S |
|-----|----------:|------------:|-------:|----------:|-----------:|
| **full** | **1.057** (s34) | 1.697 | **0.027** (s45) | **4/8** | 1.022 (s7) |
| im150 | 1.292 (s303) | 1.707 | 0.041 (s12) | 4/8 | 1.239 (s303) |
| im150stim | 1.445 (s303) | 1.848 | **0.021** (s303) | 2/8 | 1.126 (s7) |
| stimonly | 1.237 (s89) | 1.668 | 0.095 (s89) | 1/8 | 1.142 (s89) |
| regular (S scored) | 1.642 (s303) | 1.726 | 0.559 | 0/8 | **1.015** (s333) |

S-success = nSSE &lt; 0.1. Regular S stays **0.56–0.75**.

### full (legacy stim×choice)

| seed | rec | traj | I/M | S | L_S | fair | w/o S | g_s | d_s | g_i |
|-----:|----:|-----:|----:|--:|----:|-----:|------:|----:|----:|----:|
| 7 | 1.702 | 0.296 | 0.229 | 0.704 | 0.497 | 1.726 | **1.022** | ~0 | ~0 | 182 |
| 12 | 1.693 | 0.376 | 0.181 | 0.622 | 0.496 | 1.675 | 1.053 | ~0 | 0.027 | **0** |
| **34** | 1.250 | 0.354 | 0.231 | **0.028** | 0.444 | **1.057** | 1.029 | ~0 | **25.7** | **185** |
| 45 | 1.578 | 0.441 | 0.257 | **0.027** | 0.457 | 1.182 | 1.155 | ~0 | 45.7 | 83 |
| 89 | 1.367 | 0.431 | 0.485 | **0.075** | 0.499 | 1.491 | 1.416 | ~0 | 82.4 | **0** |
| 101 | 2.172 | 0.585 | 0.345 | **0.096** | 0.692 | 1.719 | 1.623 | 0.025 | 100 | **0** |
| 303 | 2.103 | 0.560 | 0.256 | 0.576 | 0.497 | 1.889 | 1.313 | 0.050 | ~0 | 61 |
| 333 | 1.987 | 0.389 | 0.662 | 0.569 | 0.496 | 2.117 | 1.547 | ~0 | ~0 | 153 |

S-success: **s34, s45, s89, s101**. All four have `g_s` at/near the
1e-12 floor; the working knob is **`d_s`**. Failed S (still ~0.57–0.70):
s7, s12, s303, s333 — parked `g_s≈d_s≈0` like regular.

Best **s34** keeps `g_i=185` (regular s34 166). Fair **1.057** beats the
morning 0.1-floor best (s101 **1.076**, `g_s=37`, `g_i=0.15`). Production
tot without S: s34 **1.029** vs regular s333 **1.015**. `L_S` 0.444 is
slightly better than regular ~0.50.

### vs morning 0.1-floor full (same seeds)

| seed | old S | old g_s | old g_i | new S | new d_s | new g_i | fair |
|-----:|------:|--------:|--------:|------:|--------:|--------:|-----:|
| 7 | 0.676 | 1.87 | 184 | 0.704 | ~0 | 182 | 1.820 → 1.726 |
| 12 | 0.241 | 27.7 | 190 | 0.622 | 0.03 | 0 | 1.274 → 1.675 |
| **34** | 0.474 | 0.60 | 166 | **0.028** | 26 | **185** | 1.661 → **1.057** |
| 45 | 0.045 | 0.32 | 37 | 0.027 | 46 | 83 | 1.272 → 1.182 |
| 89 | 0.706 | 3.47 | 116 | 0.075 | 82 | 0 | 1.766 → 1.491 |
| 101 | **0.018** | 37.3 | 0.15 | 0.096 | 100 | 0 | **1.076** → 1.719 |
| 303 | 0.079 | 50.5 | 187 | 0.576 | ~0 | 61 | 1.377 → 1.889 |
| 333 | 0.034 | 0.17 | 96 | 0.569 | ~0 | 153 | 1.191 → 2.117 |

Best-of improved; **median fair got worse** (1.326 → 1.697). Hybrid x0
can now sit at `g_s=1e-12`, so half the seeds stay in the “do nothing
on S” basin. The seeds that find `d_s` no longer need a large `g_s`
and do not have to collapse `g_i`. The morning large-`g_s` winners
(s101 / s303 / s333) do not come back.

### im150 (150 ms, stim×choice)

| seed | rec | traj | I/M | S | L_S | fair | g_s | d_s | g_i |
|-----:|----:|-----:|----:|--:|----:|-----:|----:|----:|----:|
| 7 | 1.598 | 0.654 | 0.423 | **0.074** | 0.451 | 1.602 | ~0 | 13.5 | 1.3 |
| 12 | 1.497 | 0.344 | 0.864 | **0.041** | 0.459 | 1.708 | ~0 | 36.1 | 182 |
| 34 | 2.082 | 0.485 | 0.713 | **0.087** | 0.460 | 1.745 | ~0 | 24.4 | 167 |
| 45 | 1.817 | 0.468 | 0.338 | 0.264 | 0.520 | 1.589 | **119** | 78.1 | 200 |
| 89 | 1.844 | 0.506 | 0.562 | 0.229 | 0.461 | 1.759 | ~0 | 29.4 | 0 |
| 101 | 3.004 | 0.614 | 0.525 | 0.757 | 0.499 | 2.396 | 62.8 | ~0 | 0 |
| **303** | 1.302 | 0.465 | 0.321 | **0.053** | 0.453 | **1.292** | 0.001 | 42.3 | **200** |
| 333 | 2.288 | 0.306 | 0.823 | 0.130 | 0.446 | 1.706 | ~0 | 34.8 | 0.93 |

S-success: s7, s12, s34, s303 — again `d_s` not `g_s`. Large `g_s`
(s45 / s101) does not win. s303 on the production window is **1.179**
(still behind regular 1.015). Traj is worse than full (best traj here
0.306 vs full s7 0.296, but the 150 ms I/M term is a different scale).

### stimonly (legacy window, stim I/M)

| seed | rec | traj | I/M | S | L_S | fair | g_s | d_s | g_i |
|-----:|----:|-----:|----:|--:|----:|-----:|----:|----:|----:|
| 7 | 1.892 | 0.560 | 0.272 | 0.760 | 0.497 | 2.089 | ~0 | ~0 | **0** |
| 12 | 1.443 | 0.496 | 0.202 | 0.216 | 0.446 | 1.360 | ~0 | 89.8 | **0** |
| 34 | 1.890 | 0.396 | 0.386 | 0.600 | 0.497 | 1.879 | ~0 | ~0 | 0.05 |
| 45 | 1.962 | 0.663 | 0.235 | 0.441 | 0.444 | 1.783 | ~0 | 20.6 | **0** |
| **89** | 1.255 | 0.463 | 0.231 | **0.095** | 0.449 | **1.237** | 0.003 | 92.3 | 4.3 |
| 101 | 2.170 | 0.509 | 0.208 | 0.315 | 0.521 | 1.553 | 2.74 | 100 | 14 |
| 303 | 1.403 | 0.688 | 0.142 | 0.223 | 0.431 | 1.484 | ~0 | 7.0 | **0** |
| 333 | 1.673 | 0.505 | 0.482 | 0.645 | 0.479 | 2.111 | ~0 | 7.1 | **0** |

Only **s89** clears S &lt; 0.1. **7/8** seeds collapse `g_i`. Scored
back on production stim×choice, I/M blows up (s7 prod I/M **1.599**,
fair 3.42) — the stim-stratum fit does not transfer.

### im150stim (150 ms, stim I/M) — scored 2026-09-09

| seed | rec | traj | I/M | S | L_S | fair | g_s | d_s | g_i |
|-----:|----:|-----:|----:|--:|----:|-----:|----:|----:|----:|
| 7 | 1.327 | 0.392 | 0.265 | 0.216 | 0.470 | 1.342 | **4.27** | 11.3 | **0** |
| 12 | 1.661 | 0.362 | 0.529 | 0.680 | 0.497 | 2.068 | ~0 | 0.013 | 0.02 |
| 34 | 2.904 | 0.536 | 0.378 | 0.460 | 0.497 | 1.871 | ~0 | ~0 | **0** |
| 45 | 2.330 | 0.452 | 0.401 | 0.584 | 0.491 | 1.928 | 0.074 | 5.29 | **0** |
| 89 | 1.585 | 0.401 | 0.369 | 0.557 | 0.497 | 1.824 | 0.007 | 0.002 | **0** |
| 101 | 2.182 | 0.590 | 0.503 | **0.034** | 0.442 | 1.569 | 2.22 | 49.2 | **0** |
| **303** | 1.501 | 0.595 | 0.378 | **0.021** | 0.452 | **1.445** | ~0 | 16.2 | 0.72 |
| 333 | 2.723 | 0.637 | 0.312 | 0.707 | 0.497 | 2.152 | ~0 | 0.001 | **0** |

Worst of the four full arms on fair tot (best **1.445**, median
**1.848**). S-success only **s303** (d_s=16) and **s101** (g_s=2.22,
d_s=49). **7/8** seeds collapse `g_i` (s303 keeps 0.72). Production
rescore is bad (s303 prod fair **2.113**, I/M **1.045**) — same
non-transfer as stimonly.

Act-prior RT is mostly broken (s7/s12/s34/s89/s303/s333 RT comb
**−4.6 to −10**). Exception: s101 perf **0.908** / RT **0.495** (g_i
still 0). s45 is the next-least-bad (perf 0.816, RT −0.50) but failed S.

### Read

1. **S coupling is an offset.** Once `g_s` can be ~0, the optimizer
   prefers `d_s` and almost never keeps a large P→S gain. The morning
   `g_s=37` solution was a basin created by clipping hybrid x0 to 0.1.
2. **Best full is now usable with I.** s34 fair 1.057 / S 0.028 / `g_i`
   intact / w/o S 1.029. That is the first full seed that is close to
   production regular **and** fits the S curve.
3. **Median got worse.** Four full seeds stay at `g_s≈d_s≈0` and look
   like regular on S. More seeds or a `d_s`-aware warm start would be
   the next lever, not another `g_s` floor.
4. **im150** can also fit S via `d_s` (4/8) but the 150 ms I/M term
   stays expensive (best 1.292 as fitted, 1.179 on production).
5. **stimonly and im150stim + free S** collapse `g_i` (7/8 each) and
   do not transfer to production stim×choice. im150stim has the lowest
   single-seed S nSSE (0.021) and the worst median fair (1.848).
6. Campaign is complete: **32/32** in openalyx `models/`.

### Plots (2026-09-08c / 09-09, in each run dir)

Shared stim `bps=20` seed 12345 from regular s101. Driver
`scripts/_tmp_full_s_prior_plots.py` (`im150stim` finished 2026-09-09).
Summary: `models/stageB_hold_s89_full_s_prior_1e12_plot_summary.json`.

| file | what |
|------|------|
| `IM_pre.svg` / `IM_post.svg` | I/M traj vs data |
| `P_fit.svg` | P traj |
| `S_fit.png` | retinal mean S vs contrast (`avg_mean_R` / `L_S`) |
| `prior_effects.svg` / `.png` | I/M + unsplit-80 S prior-distance |
| `psychometric_model_vs_data_actprior/` | perf + RT combined + RT split (act-prior data, subj-P model) |

