# s89ft I/M shape fine-tune

**Scope:** warm local from 80 ms `full_mpre3_meancell` **s89** to fix
two residuals on that seed’s 80 ms prior overlay: stim-window **M
overshoot** after ~40 ms, and **choice-window I/M**. Hold the S-success
basin (`d_s≈50`, `g_i` intact). Two freeze sets. Rank later at extras=0,
`m_pre_weight=1`, current `mean_c ‖Δ‖`.

**Not in scope:** cold 8-seed `full` reruns; 150 / split windows;
`gm0` / `mleak`; incongruent RT machinery; regular-mask Stage B.

**Status:** 2026-10-05 — wired, freeze-hold bug-checked, one local
loss eval from the s89 JSON. **Not scored.** Do not FORCE old `full`
/ `full_mpre3_meancell` dirs.

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
