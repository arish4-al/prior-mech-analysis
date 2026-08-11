# Joint fitting pipeline (retinal + g_s/d_s + weights)

**Scope:** co-optimize retinal front-end, sensory prior gains (`g_s`/`d_s`), and
network W/θ under a combined loss
`L = L_w(I/P/M traj + I/M prior) + L_S(S rms vs avg_mean_R)`.

**Not in scope:** weights-only speedups / ORCD `fit_weights` batch (see
[simulation_fit_speedups.md](simulation_fit_speedups.md)); prior-distance recovery
analysis.

**Status:** Pipeline code ready (`fit_joint.py` + ORCD drivers). Pre-ORCD code
review 2026-08-10; next = submit **regular** (no sensory prior) and **sensory**
(I/M prior frozen) campaigns.

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

---

## Questions to be resolved

1. **Joint λ (Lw vs LS):** equal-sum v1 — retune if one term dominates on ORCD.
2. **Loss / data audit:** alignment of Lw targets, LS (`avg_mean_R`), windows,
   regions; whether S should enter prior-distance terms (see 2026-08-06c).
3. **Regular mask:** `12|13` only vs also freeze `g_m`/`d_m` (`7|9|12|13`)?
4. Compare joint-regular winners to weights-only 0.404 baseline only after
   separating `L_w` vs `L_S` (different objectives).
