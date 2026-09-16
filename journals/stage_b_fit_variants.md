# Stage B fit variants (catalog)

**Scope:** every Stage B / joint arm we have actually fit and scored.
One row per **run tag** (the `OUT_TAG` in `weights_run_fj_<tag>_…`).
Details, keep/drop, and plots live in the topic journals; this file is
the index of **what was fit**.

**Not in scope:** Stage A retinal-only (except as the hybrid source);
smokes (`*_smoke`); clamp-only diagnostics that did not refit
(`mleak_clamp`, `im_to_m_gate_clamp`, CRF-warp RT).

**How to use:** append a row when a new arm is scored. Do not overwrite
old tags (metric-bug and FORCE-replaced dirs stay as separate rows).

**Shared protocol** (unless a row says otherwise): hybrid WEIGHTS_REL ∪
Stage-A retinal **s89**, `--stage1-hold-retinal`, `bps1=bps2=20`,
pipeline `de_cma_local`. Tests 1–6 / `full` seeds after 2026-08-24c:
`7 12 34 45 89 101 303 333`. Eval: one shared stim, `bps=20`, seed
**12345**, stim from regular **s101**. Rank on that eval (JSON
`final_loss` is own-stim — not comparable).

**Freeze masks**

| Label | `--freeze` | Frozen ≈ 0 |
|-------|------------|------------|
| regular | `12\|13` | `g_s`, `d_s` |
| sensory | `6\|7\|8\|9` | `g_i`, `g_m`, `d_i`, `d_m` |
| full | *(none)* / `masknone` | — |
| onethr | `11\|12\|13` | `θ_d` tied to `θ_c`, plus `g_s`/`d_s` |
| gm0 | `7\|9\|12\|13` | `g_m`, `d_m`, `g_s`, `d_s` |

I/M prior window **unset** = production (legacy T=72 / `plot_window=80`).
Stratum **unset** = stim×choice. `include_stim` default **false** (no
unsplit-80 S nSSE). I/M metric: `‖mean_c Δ‖` until 2026-09-08f, then
`mean_c ‖Δ‖`. Later evals re-score old weights with current code; the
**fit** still used whatever the code was that day.

Dir pattern:
`weights_run_fj_<OUT_TAG>_{regular_mask12-13|sensory_mask6-7-8-9|full_masknone|onethr_mask11-12-13}_s<seed>/`.

**Plot names:** `loss_plot_diff_by_condition_with_data` writes
`IM_pre_fit_{gi…thr…}.svg` (filename encodes the fitted gains / θ).
`alias_svgs` copies the newest of those to the stable `IM_pre.svg` —
**same figure**, short name for journals. Same for `IM_post`, `P_fit`,
`prior_effects`. Each SVG should have a PNG sibling. ITI mean
trajectories (`plot_iti_mean_trajectories`): `S_iti` (prev-stim L/R),
`I_iti` / `M_iti` (prev-choice L/R); two signed pop-diff traces each.

---

## Production baselines

| Tag | Restrictions | Best seed (eval) | Notes |
|-----|--------------|------------------|-------|
| `stageB_hold_s89` regular | mask `12\|13`; window / stratum unset; ITI gate on; ITI penalty on; `W_pp` `[0.496, 0.49999]`; two θ; `m_pre_weight=1` | **s101** fair **1.001** (08-13 hybrid-stim table). Later s101-stim batch: **s333 1.015** / median 1.051. mean_c re-score: **s34 0.999** / median 1.030 | Production keep-list. Paper RT overlay uses **s101**. [retinal then joint](retinal_then_joint_fitting.md) 08-13; re-scores in [revisions](modeling_details_revisions.md) / [gaps](modeling_details_prior_rt_gaps.md) |
| `stageB_hold_s89` sensory | mask `6\|7\|8\|9`; else production | **s23** fair **1.005** | Not in the modeling-details default seed list. Same 08-13 table |
| WEIGHTS_REL | notebook weights (not a Stage B tag) | fair **1.260** on the 08-13 batch | RT reference (only slightly +inc R²) |

Stage-A source for the hybrid: `retinal_run_fr_retinal_masknone_s89`.

---

## Tests 1–6 (regular mask; old `‖mean_c Δ‖` unless noted)

All vs production regular on the s101-stim batch unless the row says
as-fitted. Topic: [modeling_details_revisions.md](modeling_details_revisions.md).

| Test | Tag | Restrictions (vs regular) | Best seed | Keep / drop |
|------|-----|---------------------------|-----------|-------------|
| 1 | `stageB_hold_s89_poffset` | `P_offset` always on through the ITI | s7 **1.566** as fitted (gated restore 1.809) | **Drop** (keep the 100 ms pre-stim gate) |
| 2 | `stageB_hold_s89_noiti` | I/M ITI zero-penalty off | s34 **0.985** with penalty restored | **Drop** as default (median worse) |
| 3 | `stageB_hold_s89_wpplarge` | init `W_pp=0.499` (10 s); **same** box `[0.496, 0.49999]` | **s101 0.947** | Init only; **keep the 2.5 s floor** |
| 3 | `stageB_hold_s89_wppopen` | init 0.499; box floor **0.20** | s7 **1.055**; median 1.159 | **Drop** open floor |
| 3 | `stageB_hold_s89_wppsmall` | init `W_pp=0.45` (200 ms); floor 0.20 | s89 **1.011**; median 1.123 | **Drop** |
| 4 | `stageB_hold_s89_onethr` | `θ_c=θ_d`; freeze `11\|12\|13` | s12 **1.189**; median 2.637 | **Keep two θ** |
| 5 | `stageB_hold_s89_mpre3` | `m_pre_weight=3` (eval re-scored at 1) | s303 fair **0.981**; median 1.081 | **Keep `m_pre_weight=1`** (RT did not beat regular s101 0.787) |
| 6 | `stageB_hold_s89_im150` | `prior_window_ms=150`; stim×choice; **old metric** | as fitted s45 **1.313**; → production s34 **1.080** | **Keep production window unset** on that ranking. Dirs kept; do not FORCE |
| 6 | `stageB_hold_s89_im150stim` | 150 ms + `prior_stratum=stim`. Morning 0.1-floor **FORCE-replaced** by 09-08e `g_i` floor `1e-12` | 1e-12 as fitted s7 **1.188**; → prod s34 **1.252** | **Drop** stim-only I/M. Restore `g_i` floor **0.1** |
| 6 | `stageB_hold_s89_stimonly` | window unset + `prior_stratum=stim`; `g_i` floor `1e-12` | as fitted s303 **1.060**; → prod s34 **1.443** | **Drop** |

`mpre2` was an optional tag; not a scored 8-seed campaign.

---

## Metric fix (`mean_c ‖Δ‖`, 09-08f on)

Topic: [modeling_details_revisions.md](modeling_details_revisions.md) 09-08f;
[modeling_details_prior_rt_gaps.md](modeling_details_prior_rt_gaps.md).

| Tag | Restrictions | Best seed | Notes |
|-----|--------------|-----------|-------|
| `stageB_hold_s89_im150_meancell` | regular mask; 150 ms stim×choice; **new** I/M metric; `include_stim=false`; `g_i` floor still `1e-12` (s101 collapsed) | as fitted **s12 1.089**; → production s34 **1.029** | Only regular-mask 150 ms **fit** with the corrected metric. I/M now rise 80→150. No M notch. Best pooled RT s12 **0.834** |
| regular 80 ms | *(no meancell rerun)* | — | Production `stageB_hold_s89` regular was **not** re-fit after 09-08f; later tables re-**score** it with current code |

---

## M-shape (freeze `g_m`/`d_m`)

Topic: [modeling_details_prior_rt_gaps.md](modeling_details_prior_rt_gaps.md).
Mask `7\|9\|12\|13`. No 60–70 ms M notch on either arm.

| Tag | Restrictions | Best seed | Notes |
|-----|--------------|-----------|-------|
| `stageB_hold_s89_gm0` | freeze `g_m`/`d_m`; **80 ms** window (mis-wired vs M-shape aim); usual `W_mm` box | **s7 0.967**; median 1.022 | Cheap 80 ms tot win. Window cannot see the post-80 climb. **Do not reuse this tag** |
| `stageB_hold_s89_mleak` | same freeze + `W_mm∈[0.10,0.15]`, init 0.15; 80 ms | s34 **0.980**; median 1.117 | Late M dies; several `g_i` collapses. **Drop** the `W_mm` cap |
| `stageB_hold_s89_gm0_im150` | same freeze; **150 ms** stim×choice; new metric | as fitted **s101 1.128**; median 1.347 | Loses vs regular@150 (s303 **1.086** / median 1.162). **Drop** the freeze as an M-shape keep |

---

## `full` + unsplit-80 S prior

Topic: [fit_gs_ds_with_im.md](fit_gs_ds_with_im.md).
`VARIANTS=full:`, `include_stim=1`, model S `stratum_s=stim`, sidecar
`fit_targets/data_act_block_duringstim_s_unsplit80.npy`. Rank **fair** =
traj + I/M + S nSSE + `L_S`. Morning **0.1-floor `full`** dirs were
FORCE-replaced by the 1e-12 campaign.

| Tag | I/M window / stratum | Best seed (as fitted fair) | Notes |
|-----|----------------------|----------------------------|-------|
| `stageB_hold_s89_full` (1e-12; current dirs) | unset / stim×choice | **s34 1.057** (S 0.028, `d_s≈26`, `g_s≈0`, `g_i=185`) | Best joint+S so far. Median 1.697 (half the seeds do nothing on S). Old metric |
| `stageB_hold_s89_full_im150` | 150 ms / stim×choice | s303 **1.292**; median 1.707 | Old metric. **Keep**; do not FORCE |
| `stageB_hold_s89_full_im150stim` | 150 ms / stim | s303 **1.445**; median 1.848 | Old metric; 7/8 `g_i` collapse |
| `stageB_hold_s89_full_stimonly` | unset / stim | s89 **1.237**; median 1.668 | Old metric; 7/8 `g_i` collapse |
| `stageB_hold_s89_full_im150_meancell` | 150 ms / stim×choice; **new** I/M metric | **s101 1.269** (S 0.098, `d_s≈59`, `g_i=86`); median 1.359 | Does not beat 80 ms `full` s34 or regular `im150_meancell`. No M notch. s12 inc RT +0.21 but `g_i` collapsed |

The 09-08 morning 0.1-floor `full` (best s101 fair **1.076**, `g_s=37`,
`g_i=0.15`) is **not** on disk as a separate tag.

---

## Ranking cheat-sheet (best seed only)

Numbers are **as the campaign ranked them** (as-fitted if the window /
stratum / S term differs from production).

The three **production regular** rows are the **same**
`stageB_hold_s89` regular finals. What changes is the **eval**, not the
fit: 08-13 shared stim from the **hybrid** JSON (winner **s101 1.001**);
later tests 1–6 rebuilt that session from **regular s101** (same seed
12345 / `bps=20`), which sits ~0.01–0.04 higher so **s333 1.015** /
median 1.051; mean_c re-score is current `mean_c ‖Δ‖` on that s101-stim
session (**s34 0.999**).

| Family | Best arm × seed | Number |
|--------|-----------------|--------|
| Production regular (08-13, hybrid stim) | regular **s101** | fair 1.001 |
| Production regular (s101-stim batch) | regular **s333** | 1.015 |
| Production regular (mean_c re-score) | regular **s34** | 0.999 |
| Test 3 tot (`wpplarge`) | `wpplarge` **s101** | 0.947 |
| Test 2 tot (`noiti`, penalty restored) | `noiti` **s34** | 0.985 |
| 80 ms freeze `g_m`/`d_m` (`gm0`) | `gm0` **s7** | 0.967 |
| Regular 150 ms, new metric | `im150_meancell` **s12** | 1.089 |
| `full` + S, 80 ms | `full` **s34** | 1.057 |
| `full` + S, 150 ms, new metric | `full_im150_meancell` **s101** | 1.269 |

---

## Queued (not scored)

| Tag | Restrictions | Notes |
|-----|--------------|-------|
| `stageB_hold_s89_full_mpre3` | full none; `m_pre_weight=3`; window unset (~80 ms); unsplit-80 S; `mean_c ‖Δ‖` | Test 5 × `full`. Do not FORCE old `full` dirs. |
| `stageB_hold_s89_mpre3_im150_meancell` | regular `12\|13`; `m_pre_weight=3`; 150 ms stim×choice; `mean_c ‖Δ‖` | Test 5 × 150. Fair eval at weight 1. [revisions](modeling_details_revisions.md) 09-15 |
| `stageB_hold_s89_full_mpre3_im150_meancell` | full none; `m_pre_weight=3`; 150 ms stim×choice; unsplit-80 S; `mean_c ‖Δ‖` | Same + `g_s`/`d_s` free. [g_s/d_s with I/M](fit_gs_ds_with_im.md) 09-15 |

---

## 2026-09-15 — catalog opened

Collected tags, freeze / window / stratum, and best seeds from
[retinal then joint](retinal_then_joint_fitting.md),
[modeling_details_revisions.md](modeling_details_revisions.md),
[modeling_details_prior_rt_gaps.md](modeling_details_prior_rt_gaps.md),
and [fit_gs_ds_with_im.md](fit_gs_ds_with_im.md).
