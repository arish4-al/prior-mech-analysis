# Joint fitting pipeline (retinal + g_s/d_s + weights)

**Scope:** co-optimize retinal front-end, sensory prior gains (`g_s`/`d_s`), and
network W/θ under a combined loss
`L = L_w(I/P/M traj + I/M prior) + L_S(S rms vs avg_mean_R)`.

**Not in scope:** weights-only speedups / ORCD `fit_weights` batch (see
[simulation_fit_speedups.md](simulation_fit_speedups.md)); prior-distance recovery
analysis.

**Status:** Fair shared-stim compare (2026-08-11b): baseline joint **1.344**; best
fit warmde regular s101 **2.083**. Cold10 / warmde from cold winners do not reach
baseline; sensory warmde 0/10 Stage-1 fail. **Main line pivoted** (2026-08-11c) to
staged retinal→joint — see
[retinal_then_joint_fitting.md](retinal_then_joint_fitting.md).

**Code:** [`fit_joint.py`](../fit_joint.py);
hooks in [`fit_weights.fit_weights_two_stage_v2`](../fit_weights.py);
drivers [`scripts/run_fit_joint.py`](../scripts/run_fit_joint.py),
`run_fit_joint_slurm.sh`, `submit_fit_joint_sharded.sh`.

---

## Why joint

`g_s`/`d_s` enter S via `tanh(a·(v + g_s·P)·S0 + d_s·P)` and change adaptation
through `W_as·|S|`. Fitting retinal at `g_s=d_s=0` then freezing it misattributes
prior-driven S structure. `g_s·P` also couples to weight params that shape P. So
sensory-prior models need retinal ∪ `{g_s,d_s}` ∪ W/θ co-optimized.

For a fair **no-sensory-prior** baseline under the same `L_w+L_S` objective, use
the joint pipeline with `g_s`/`d_s` frozen ≈0 (do not mix weights-only `L_w` with
joint `L_w+L_S` when comparing variants).

---

## Layout (21-d mixed; indices 0–11 match `fit_weights`)

| idx | params |
|-----|--------|
| 0–5 | `W_*` (log) |
| 6–9 | `g_i,g_m,d_i,d_m` (log) |
| 10–11 | `θ_c,θ_d` (log) |
| 12–13 | **`g_s`,`d_s`** (log) |
| 14–20 | retinal: `α_w` (log), **`β_w` asinh**, `α_d`,`β_d`,`τ_a`,`W_as`,`W_ss` (log) |

Freeze fill uses `LOG_ZERO` except **`β_w→asinh(0)=0`** when frozen.

**Loss:** one `run_model` → `L = L_w + L_S`. Equal-sum v1; components visible in
verbose steps / checkpoints (`layout: joint21`).

**Defaults:** `L_threshold=10`, `bps_stage1=10` (match `fit_retinal`),
`beat_loss=1.2` (joint scale); polish `prior` → `[6,8,10,11,12,13]` ∩ train_mask.
Needs `avg_mean_R.npy` (symlink from `paper-brain-wide-map/` or ONE figs).

---

## Planned ORCD variants (2026-08-10)

| Label | `--mtype` | `--freeze` | Meaning |
|-------|-----------|------------|---------|
| **regular** | `regular` | `12\|13` | No prior mod on sensory (`g_s`,`d_s`≈0); I/M gains + W + θ + retinal free |
| **sensory** | `sensory` | `6\|7\|8\|9` | Prior mod only on sensory; freeze `g_i,g_m,d_i,d_m`≈0; `g_s`/`d_s` + W + θ + retinal free |

Optional paper-like regular (also freeze move gains): `gain:7|9|12|13`.

Polish after CMA (`LOCAL_REFINE_IDX=prior`):

- regular → `[6, 8, 10, 11]` (g_i, d_i, θ)
- sensory → `[10, 11, 12, 13]` (θ + g_s, d_s)

```bash
# Full campaign (both variants × 5 seeds)
VARIANTS="regular:12|13 sensory:6|7|8|9" SEEDS="56 34 78 89 202" \
  bash scripts/submit_fit_joint_sharded.sh

# Sensory only (legacy default)
VARIANTS="sensory:6|7|8|9" SEEDS="56 34 78 89 202" \
  bash scripts/submit_fit_joint_sharded.sh
```

Run dirs: `weights_run_fj[_tag]_<mtype>_mask<slug>_s<seed>/`.

---

### 2026-08-06b — Joint fit pipeline added

Moved from [simulation_fit_speedups.md](simulation_fit_speedups.md).

**Code:** [`fit_joint.py`](../fit_joint.py) (pack/unpack/loss/save);
[`fit_weights.fit_weights_two_stage_v2`](../fit_weights.py) accepts optional hooks
(`safe_loss_fn`, `bounds_fn`, `freeze_fill`, `loss_extra_kwargs`, …) so the 12-d
weights path is unchanged. Drivers: `scripts/run_fit_joint.py`,
`run_fit_joint_slurm.sh`, `submit_fit_joint_sharded.sh`.

**Defaults (then):** variant `sensory:6|7|8|9`; see layout/loss above.

---

### 2026-08-06c — Joint bounds audit; β_w asinh; freeze-clamp fix

Reviewed native bounds and optimizer coords for all 21 dims (same bounds at
Stage-1 DE, Stage-2 CMA, and polish — only active mask changes).

**Native bounds** (`fit_joint.NATIVE_BOUNDS`; weights block = `fit_weights._log_bounds_weights_v2`):

| idx | name | native | opt coord |
|-----|------|--------|-----------|
| 0–5 | W_* | as weights | log |
| 6 | g_i | (0.1, 200) | log |
| 7 | g_m | (1e-12, 200) | log |
| 8 | d_i | (1e-5, 100) | log |
| 9 | d_m | (1e-12, 100) | log |
| 10–11 | θ | (0.1, 0.99999) | log |
| 12 | g_s | (0.1, 200) like g_i | log |
| 13 | d_s | (1e-5, 100) like d_i | log |
| 14 | α_w | (1, 2.6) | log |
| 15 | **β_w** | (−0.2, 0.2) | **asinh(β/0.05)** |
| 16–20 | α_d, β_d, τ_a, W_as, W_ss | fit_retinal | log |

**Bug fixed:** after `--freeze`, several paths clamped the *full* vector to bounds,
so frozen `g_i` was pulled from `LOG_ZERO` up to Lb=`log(0.1)` → native 0.1 (not ~0).
DE worker + resume now clamp **free dims only**. Freeze fill for β_w is asinh(0)=0.

**β_w sampling:** was uniform in native [−0.2, 0.2] (unlike log-uniform on positives).
Now optimizer stores `z=asinh(β/0.05)`; DE/Sobol/CMA sample uniformly in z → denser
near β≈0, qualitatively matching log-magnitude exploration. CMA_stds[15]=0.1 (asinh).

**Later todos (do not block smoke):**

1. **Review loss functions and how real data is gathered for the losses** — joint
   `L_w` (I/P/M traj + I/M prior vs `mean_data_results` / act_block) + `L_S`
   (`compute_sse_stim_right` vs `avg_mean_R`); check alignment with paper targets,
   windows, region lists, and whether S should also enter prior-distance terms.
2. Joint λ (Lw vs LS) if one term dominates on ORCD smoke.

---

### 2026-08-10 — Joint Stage-1 gate / bps defaults

Cold joint DE often finished with finite best ~5–6 or all-`1e11` under the
weights-era `L_threshold=3.5` (borderline_hi=3.9), so Stage 1 aborted even when
`fit_retinal`-scale S averaging at `bps=10` was fine. Raised defaults to
`BPS_STAGE1=10`, `L_THRESHOLD=10` in `run_fit_joint.py` / slurm / submit scripts.

**ORCD smoke (tiny budget):**

```bash
cd ~/int-brain-lab/prior-mech-analysis
# ensure avg_mean_R.npy + act_block + mean_data_results available
sbatch --parsable --mem=40G --cpus-per-task=16 --time=1:00:00 \
  --job-name=fj_smoke \
  --export=ALL,MTYPE=sensory,FREEZE=6\|7\|8\|9,SEED=999,PIPELINE=de_cma_local,OUT_TAG=smoke,\
DE1_MAXITER=2,DE2_MAXITER=3,POPSIZE=8,SOBOL_COUNT=4,PATIENCE=0,\
LOCAL_REFINE_MAX_WALL_S=60,FORCE=1 \
  scripts/run_fit_joint_slurm.sh
```

---

### 2026-08-10b — Pre-ORCD code review (regular + sensory)

Reviewed `fit_joint.py`, `scripts/run_fit_joint*.sh`, and joint hooks in
`fit_weights.py` before submitting both variants.

**Sanity (iblenv):**

- Bounds / `CMA_STDS` / pack↔unpack: 21-d consistent; β_w freeze fill = asinh(0).
- Freeze → native ~0: `12|13` zeros `g_s`/`d_s`; `6|7|8|9` zeros I/M gains
  (not clamped up to Lb).
- Polish ∩ mask: regular → `[6,8,10,11]`; sensory → `[10,11,12,13]`.

**Bugs fixed (this entry):**

1. **`load_theta_from_ckpt` assumed D=12** — joint active-only DE ckpts with
   `fit_idx` up to 20 could not expand; now infers 21 from `layout: joint21`,
   `train_mask` length, or max index. Frozen β_w fill uses asinh(0), not `LOG_ZERO`.
2. **Resume `frozen_idx` inference reset `D_full` to 12** via
   `_log_bounds_weights_v2()` — now uses the fitter’s existing `D_full`.
3. **`run_fit_joint._find_resume`** could not expand active-only DE/CMA ckpts
   (reconstruct required `W` groups). Now expands via `fit_idx` + `freeze_fill_joint`.

**Not bugs / watchouts (do not block submit):**

- Free `g_s` cannot go below native 0.1 (same as `g_i`); “no sensory prior” **must**
  use freeze `12|13`, not an unconstrained fit hoping for ~0.
- Joint `SAVE_THRESH_V2=0.8` is low vs typical `L_w+L_S` (~1+); mid-run JSON saves
  are sparse (every 1000 steps) — rely on `weights_stage2_last` / finals.
- Equal-sum `L_w+L_S` and whether S enters prior-distance terms still open
  (2026-08-06c todos).
- Working tree may also contain Stage-1 DE **basin-jump** on all-penalty
  (`de1_inf_restarts`, default 2) — useful for joint NaN-S flats; confirm it is
  on the branch you push to ORCD.
- Slurm: use `FREEZE=6|7|8|9` (pipes) in `--export=VAR=...` lists; commas truncate.
  `submit_fit_joint_sharded.sh` uses `--export=ALL` after setting env (safe).
- ORCD needs `avg_mean_R.npy` + `mean_data_results` + act_block figs under
  `ONE_CACHE_DIR` (or symlink targets in the slurm script).

**Suggested smoke before full 2×5:**

```bash
# regular
sbatch --parsable --mem=40G --cpus-per-task=16 --time=1:00:00 \
  --job-name=fj_smoke_reg \
  --export=ALL,MTYPE=regular,FREEZE=12\|13,SEED=999,PIPELINE=de_cma_local,OUT_TAG=smoke,\
DE1_MAXITER=2,DE2_MAXITER=3,POPSIZE=8,SOBOL_COUNT=4,PATIENCE=0,\
LOCAL_REFINE_MAX_WALL_S=60,FORCE=1 \
  scripts/run_fit_joint_slurm.sh

# sensory
sbatch --parsable --mem=40G --cpus-per-task=16 --time=1:00:00 \
  --job-name=fj_smoke_sen \
  --export=ALL,MTYPE=sensory,FREEZE=6\|7\|8\|9,SEED=999,PIPELINE=de_cma_local,OUT_TAG=smoke,\
DE1_MAXITER=2,DE2_MAXITER=3,POPSIZE=8,SOBOL_COUNT=4,PATIENCE=0,\
LOCAL_REFINE_MAX_WALL_S=60,FORCE=1 \
  scripts/run_fit_joint_slurm.sh
```

### 2026-08-11 — Cold10 + warmde vs separate-fit baseline (fair compare)

Local: `~/Downloads/ONE/openalyx.internationalbrainlab.org/models/weights_run_fj_{cold10,warmde_*}_*`.
Seeds for both campaigns: `56 34 78 89 202 12 45 67 101 303`.

**Campaigns**

| Campaign | Variant (freeze) | How started | FIT_DONE |
|----------|------------------|-------------|---------|
| `cold10` | regular `12\|13` | cold DE | **5 / 10** |
| `cold10` | sensory `6\|7\|8\|9` | cold DE | **1 / 10** |
| `warmde_r12` | regular `12\|13` | warm DE from cold regular s12 | **3 / 10** (s12, s101, s202) |
| `warmde_s101` | sensory `6\|7\|8\|9` | warm DE from cold sensory s101 | **0 / 10** |

Failures: Stage-1 `failed_stage1` / penalty `1e11` (or loss above `L_threshold=10`).

**Comparison protocol (use these numbers only):** build **5 stim bundles once**
at `bps=20` with WEIGHTS_REL session hyperparams and RNG seeds
`RandomState(s+100003)` for `s ∈ {0,1,2,56,89}`; score every θ with **mean**
joint `L_w+L_S` on that fixed set. Do **not** rank by fit `final_loss` (`sample`
aggregate is seed-/draw-dependent). Same seed alone is not enough if
`model_params` (e.g. retinal) differ when rebuilding stim.

**Separate-fit baseline** (WEIGHTS_REL / paper weights + frozen retinal + `g_s=d_s≈0`
under the joint loss; weights-only recorded **0.404** is `L_w` only):

| Params | Fair total | L_w | L_S |
|--------|-----------:|----:|----:|
| WEIGHTS_REL (`g_i≈190`) + gs0 | **1.344** | 0.561 | 0.783 |
| Paper `g_i=163` + gs0 | 1.375 | 0.592 | 0.783 |

**Best by variant (fair total vs baseline 1.344)**

| Variant | Best run | Fair | L_w | L_S | Δ vs baseline | Succeeded |
|---------|----------|-----:|----:|----:|--------------:|-----------|
| **regular** | warmde_r12 **s101** | **2.083** | 1.159 | 0.924 | +0.74 | cold 5/10 + warmde 3/10 scored |
| **sensory** | cold10 **s101** | **4.347** | 2.938 | 1.410 | +3.00 | cold 1/10; warmde 0/10 |

No joint fit reached the separate-fit baseline. Gap is mostly **L_w**.

**All scored finals** (fair ascending)

| Campaign | Variant | Seed | Fair | L_w | L_S | Δ vs 1.344 |
|----------|---------|-----:|-----:|----:|----:|-----------:|
| warmde_r12 | regular | 101 | **2.083** | 1.159 | 0.924 | +0.74 |
| warmde_r12 | regular | 202 | 2.102 | 1.210 | 0.893 | +0.76 |
| cold10 | regular | 12 | 2.243 | 1.265 | 0.978 | +0.90 |
| cold10 | regular | 56 | 2.591 | 1.772 | 0.819 | +1.25 |
| warmde_r12 | regular | 12 | 3.569 | 2.541 | 1.028 | +2.23 |
| cold10 | regular | 89 | 4.053 | 2.828 | 1.225 | +2.71 |
| cold10 | sensory | 101 | **4.347** | 2.938 | 1.410 | +3.00 |
| cold10 | regular | 34 | 4.761 | 3.735 | 1.026 | +3.42 |
| cold10 | regular | 303 | 5.751 | 2.985 | 2.766 | +4.41 |

**Warm-DE note:** `--resume-json` + `de_cma*` seeds Stage-1 DE (`theta_log0` + x0
inject); `cma_only` still skips DE. Warm from cold winners often failed Stage-1;
next try should warm from WEIGHTS_REL JSON.

Canvas: `joint-fair-compare-bps20.canvas.tsx`.

---

## Questions to be resolved

1. **Joint λ (Lw vs LS):** equal-sum v1 — retune if one term dominates on ORCD.
2. **Loss / data audit:** alignment of Lw targets, LS (`avg_mean_R`), windows,
   regions; whether S should enter prior-distance terms (see 2026-08-06c).
3. **Regular mask:** `12\|13` only vs also freeze `g_m`/`d_m` (`7\|9\|12\|13`)?
4. ~~Compare joint to 0.404~~ **Answered:** use joint fair `L_w+L_S`; WEIGHTS_REL
   baseline **1.344**; best regular **2.083**; best sensory **4.347**.
5. ~~Primary path = joint-from-scratch?~~ **No (2026-08-11c):** staged
   retinal→joint with retinal free in Stage B — see
   [retinal_then_joint_fitting.md](retinal_then_joint_fitting.md).
