# s89ft I/M shape fine-tune

**Scope:** warm local from 80 ms `full_mpre3_meancell` **s89** to fix
two residuals on that seed’s 80 ms prior overlay: stim-window **M
overshoot** after ~40 ms, and **choice-window I/M**. Hold the S-success
basin (`d_s≈50`, `g_i` intact). Two freeze sets. Rank later at extras=0,
`m_pre_weight=1`, current `mean_c ‖Δ‖`.

**Not in scope:** cold 8-seed `full` reruns; 150 / split windows;
`gm0` / `mleak`; incongruent RT machinery; regular-mask Stage B.

**Status:** 2026-10-05 — scored (8/8 `FIT_DONE`). Fair tot at extras=0:
`wii` **s7 1.014** beats source s89 **1.054**. Stim-M overshoot drops
on `wii` (0.133 → 0.070) but the curve still ramps through 40–80 ms.
Choice I still collapses in the last ~20 ms before move (`d_i≈0`).
**Queued:** `d_i` punch from `wii` s7 (`s89ft_di_choicei`).

**Code:** `--freeze-hold`, `--choice-im-extra-weight`,
`--m-stim-overshoot-weight` in
[`scripts/run_fit_joint.py`](../scripts/run_fit_joint.py) /
[`fit_joint.py`](../fit_joint.py). Submit
[`scripts/submit_fit_stage_b_s89ft_imshape.sh`](../scripts/submit_fit_stage_b_s89ft_imshape.sh).
Catalog: [Stage B fit variants](stage_b_fit_variants.md). Parent shape
aims: [prior-curve dips](modeling_details_prior_rt_gaps.md).

---

## Why this, not a longer cold search

s89 already used **`mean_c ‖Δ‖`** (`_meancell`). Old-metric `full` s34
was never re-fit; this campaign does not start from s34 (`d_i≈0`).
Median `full` tot is ~1.66 vs s89 **1.054** — most seeds never get S.
Same tot, more gens, will not punish M-overshoot or choice I/M.

Default joint freeze **zeros** frozen log-dims (`LOG_ZERO`). Holding
`d_s` therefore needs `--freeze-hold` (freeze-fill = resume θ).

---

## Setup (2026-10-05)

Warm `PIPELINE=cma_only` from
`weights_run_fj_stageB_hold_s89_full_mpre3_meancell_full_masknone_s89`
newest `weights_final_*.json`. `include_stim=1`, window **unset** (~80
ms), stim×choice, `m_pre_weight=1`, extras weight **1** each, overshoot
from **40 ms**. CMA seeds `89 7 12 45` (restarts from the same JSON).
`BEAT_LOSS=-1` (new loss scale). `FORCE=0`.

Fair eval later: extras off, same 80 ms tot as the catalog.

### Freeze sets

s89 W / prior-mod stay at the fitted θ except the free indices.

| Index | Param | Role | nowii (1) | wii (2) |
|------:|-------|------|-----------|---------|
| 0 | `W_ii` | I recurrence | freeze-hold | **free** |
| 1 | `W_pp` | P recurrence (2.5 s floor) | freeze-hold | freeze-hold |
| 2 | `W_mm` | M recurrence | **free** | **free** |
| 3 | `W_is` | S→I | freeze-hold | freeze-hold |
| 4 | `W_pi` | I→P | freeze-hold | freeze-hold |
| 5 | `W_mi` | I→M | **free** | **free** |
| 6 | `g_i` | S×P → I | freeze-hold | freeze-hold |
| 7 | `g_m` | I×P → M | **free** | **free** |
| 8 | `d_i` | P→I offset | **free** | **free** |
| 9 | `d_m` | P→M offset | **free** | **free** |
| 10–11 | `θ_c`, `θ_d` | action thresholds | freeze-hold | freeze-hold |
| 12–13 | `g_s`, `d_s` | P→S | freeze-hold | freeze-hold |
| 14–20 | retinal | Stage A | freeze-hold | freeze-hold |

`nowii` is option 1 as first wired: M-side + `d_i` only. Choice I can
shift (DC `d_i`) but not change hold time. `wii` adds `W_ii` so
post-S I can keep that offset into the choice window.

Polish `LOCAL_REFINE_IDX` matches the free set (intersected with
`train_mask`).

| ARM | `OUT_TAG` | freeze mask | polish |
|-----|-----------|-------------|--------|
| `nowii` | `stageB_hold_s89_full_s89ft_imshape_meancell` | `0\|1\|3\|4\|6\|10–20` | `2,5,7,8,9` |
| `wii` | `stageB_hold_s89_full_s89ft_imshape_wii_meancell` | `1\|3\|4\|6\|10–20` | `0,2,5,7,8,9` |

### Extra loss (fit only)

On top of traj + I/M prior + S nSSE + `L_S`:

1. `choice_im_extra_weight × (duringchoice I nSSE + M nSSE)` — doubles
   the choice panel relative to the pooled prior term when weight=1.
2. `m_stim_overshoot_weight ×` energy-normalized hinge nSSE of stim
   M where model > data for `t ≥ 40` ms.

JSON `final_loss` includes these. Rank without them.

### Submit (user pastes; agents do not `sbatch`)

```bash
PARTITION=mit_preemptable FORCE=0 \
  bash scripts/submit_fit_stage_b_s89ft_imshape.sh
```

One arm: `ARMS=nowii` or `ARMS=wii`.

---

## 2026-10-05 — local smoke (laptop)

`iblenv`, numba, one `loss_joint_core` pair on the s89 final
(`bps=20`, `include_stim`, shared stim seed 12345). No CMA.

| Check | Result |
|-------|--------|
| Missing stim-M curves → overshoot | `nan` (not a crash; `np.asarray(None)` would have been size-1) |
| s89 native | `g_i=138`, `d_s=49.7`, `d_i=0.0026`, `W_ii=0.422`, `W_mm=0.283`, `W_mi=0.517` |
| Default freeze-fill (`LOG_ZERO`) | `g_i` and `d_s` → `~1e-13` (would kill the S basin) |
| `--freeze-hold` | keeps `d_s=49.7` |
| extras=0 | tot **1.225** (`L_w=0.704`, `L_S=0.521`) |
| choice I+M nSSE | 0.015 + 0.054 = **0.124** |
| M hinge nSSE t≥40 ms | **0.055** (curve 0–80 ms, 40 bins) |
| extras=1 | tot **1.349** = extras=0 + 0.124 + 0.055 (bit-match); `L_S` unchanged |

`--freeze-hold` is required. Extras are additive and finite. Ready to
commit; not scored; do not FORCE old dirs.

---

## 2026-10-05 — scored (openalyx, extras=0)

Eight `cma_only` warm restarts, all `FIT_DONE`. Shared-stim eval
(`bps=20`, seed **12345**, stim from regular **s101**, `m_pre_weight=1`,
`include_stim`, current `mean_c ‖Δ‖`, window unset). JSON `loss`
includes extras and a per-gen stim draw — **do not rank on it**.
Source s89 on this protocol is tot **1.054** (matches the catalog).

Freeze-hold held: every seed `g_i=138`, `d_s=49.7`, `g_s≈0`, θ
unchanged. Free dims barely moved (`W_ii` 0.422 → 0.417–0.422;
`d_i` stayed `~0.003` except `nowii` s7 `0.0006`; `d_m` still `~0`).
`wii` s12 is a no-op clone of source θ (fair tot identical).

### Fair tot (extras off)

| arm | seed | rec (extras=1, fit stim) | fair | extras=1 eval | traj | IM | S | `L_S` | Ich | Mch | Mov≥40 |
|-----|-----:|-------------------------:|-----:|--------------:|-----:|---:|--:|------:|----:|----:|-------:|
| src | 89 | 1.125 | 1.054 | 1.236 | 0.375 | 0.201 | 0.022 | 0.456 | 0.021 | 0.028 | 0.133 |
| **wii** | **7** | 1.135 | **1.014** | **1.146** | 0.385 | **0.151** | **0.017** | 0.460 | 0.026 | 0.036 | **0.070** |
| nowii | 7 | 1.546 | 1.018 | 1.179 | 0.364 | 0.176 | 0.023 | 0.455 | 0.020 | 0.024 | 0.118 |
| wii | 89 | 1.048 | 1.025 | 1.175 | 0.377 | 0.166 | 0.022 | 0.460 | 0.027 | 0.022 | 0.101 |
| wii | 45 | 2.038 | 1.029 | 1.160 | 0.380 | 0.147 | 0.046 | 0.456 | 0.021 | 0.046 | **0.064** |
| wii | 12 | 1.685 | 1.054 | 1.236 | 0.375 | 0.201 | 0.022 | 0.456 | 0.021 | 0.028 | 0.133 |
| nowii | 12 | 1.123 | 1.071 | 1.249 | 0.396 | 0.194 | 0.034 | 0.448 | 0.019 | 0.039 | 0.120 |
| nowii | 89 | 1.256 | 1.088 | 1.268 | 0.406 | 0.200 | 0.029 | 0.453 | 0.021 | 0.035 | 0.126 |
| nowii | 45 | 1.573 | 1.122 | 1.317 | 0.402 | 0.220 | 0.044 | 0.456 | 0.018 | 0.038 | 0.138 |

### Stim M (the overshoot aim)

Data-space `mean_c ‖Δ‖` at 40 / 70 / 80 ms (same T=72 / 80 ms axis as
the source overlay). Source ramps **0.109 / 0.131 / 0.132**.

| arm | seed | M40 | M70 | M80 | hinge nSSE |
|-----|-----:|----:|----:|----:|-----------:|
| src | 89 | 0.109 | 0.131 | 0.132 | 0.133 |
| nowii | 7 | 0.107 | 0.130 | 0.130 | 0.118 |
| wii | 89 | 0.105 | 0.127 | 0.129 | 0.101 |
| wii | 7 | 0.101 | 0.121 | 0.119 | 0.070 |
| wii | 45 | 0.099 | 0.120 | 0.120 | 0.064 |

`nowii` cannot cut the ramp with frozen `W_ii`. s7 “won” tot by
collapsing `g_m` 0.012 → 0.0007 — not a shape fix. `wii` lowers the
whole M trace ~0.01–0.02 and halves hinge nSSE; there is still no
notch after the S peak.

Choice-window I/M: source 0.021 / 0.028. No seed moved `d_i` off the
floor. Choice I is flat or slightly worse (`wii` s89 0.027).

Act-prior behavior (10×20, seed 12345; inc RT was **not** in the
loss). Source-like: perf R² 0.81–0.85, pooled RT 0.65–0.76, inc RT
still negative (−0.37 to −1.30). `wii` s7 inc **−0.95**.

### Overlays (in each run dir)

`plot_one` + act-prior RT + extra `prior_effects_80ms.svg/png` (fit
window; does not replace the 150 ms display `prior_effects.*`).
SVG+PNG for I/M, P, S, prior, ITI, 80 ms prior, and
`psychometric_model_vs_data_actprior/`. Drivers:
[`scripts/_tmp_s89ft_imshape_eval.py`](../scripts/_tmp_s89ft_imshape_eval.py),
[`scripts/_tmp_s89ft_imshape_plots.py`](../scripts/_tmp_s89ft_imshape_plots.py).

openalyx `models/`:

- `nowii`: `weights_run_fj_stageB_hold_s89_full_s89ft_imshape_meancell_full_mask0-1-3-4-6-10-11-12-13-14-15-16-17-18-19-20_s{89,7,12,45}/`
- `wii`: `weights_run_fj_stageB_hold_s89_full_s89ft_imshape_wii_meancell_full_mask1-3-4-6-10-11-12-13-14-15-16-17-18-19-20_s{89,7,12,45}/`

Best seed to inspect: **`wii` s7**.

Short CMA from s89 with extras **does** beat source tot (`wii` s7
**1.014** vs **1.054**) and cuts stim-M hinge nSSE, but it does not
produce a post-S M pause or a live choice I (`d_i` still ~0). The
choice-I drop is at commit, with stim still on.

---

## 2026-10-05b — `d_i` punch + late choice-I extra (queued)

Choice I decays in the last ~20 ms before move **while stim is still
on**. That is the commit constraint: with `d_i≈0`, I at t=0 is
whatever trips `|M|=θ`, so stim×choice prior distance collapses. Data
stays ~0.08. Regular holds this panel with `d_i ~ 20`.

Warm `cma_only` from `wii` **s7**. `--set-d-i 5`. Extra: undershoot
hinge on duringchoice I in the last **40 ms**. Pooled choice I+M
extra and stim-M overshoot **off**. Window unset, stim×choice,
`m_pre_weight=1`, `--freeze-hold`. Seeds `89 7 12 45`. `BEAT_LOSS=-1`.
`FORCE=0`. Two freeze sets.

Rank later at extras=0. JSON loss includes the hinge.

| ARM | `OUT_TAG` | freeze mask | polish | init |
|-----|-----------|-------------|--------|------|
| `di` | `stageB_hold_s89_full_s89ft_di_choicei_meancell` | `0–7,9–20` | `8` | `d_i=5` |
| `wiigi` | `stageB_hold_s89_full_s89ft_wiigi_choicei_meancell` | `1,3,4,10–20` | `0,2,5,6,7,8,9` | `d_i=5`; `g_i` free |

```bash
PARTITION=mit_preemptable FORCE=0 \
  bash scripts/submit_fit_stage_b_s89ft_di_choicei.sh
```
