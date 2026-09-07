# Fit g_s / d_s together with I/M prior-mod parameters

**Scope:** a Stage B / joint variant that leaves **all** prior-mod gains free
(`g_i, g_m, d_i, d_m, g_s, d_s`) instead of the current exclusive masks
(regular freezes `g_s`/`d_s`; sensory freezes I/M). The S target for that
variant is a **real-data prior-distance curve**, not the retinal `avg_mean_R`
mean-activity curves.

**Not in scope:** changing I/M traj targets, `avg_mean_R` / Stage A, Harris /
Bayes priors, or the default regular/sensory freeze masks.

**Status:** region set + S curve built 2026-09-02. Fit variant wired
(`--mtype full`, `--include-stim-prior`). No ORCD campaign submitted.

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
