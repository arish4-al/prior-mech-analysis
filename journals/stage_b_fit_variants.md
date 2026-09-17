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
**fit** still used whatever the code was that day. **New** I/M `OUT_TAG`s
include `_meancell` so they do not collide with pre-09-08f dirs.

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
| `stageB_hold_s89_mpre3_im150_meancell` | same + `m_pre_weight=3` (eval @ 1) | as fitted **s303 1.154**; median 1.225 | Loses to `im150_meancell` on best tot. [revisions](modeling_details_revisions.md) 09-16 |
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
| `stageB_hold_s89_full` (1e-12; current dirs) | unset / stim×choice | **s34 1.057** (S 0.028, `d_s≈26`, `g_s≈0`, `g_i=185`) | Old metric. Median 1.697 (half the seeds do nothing on S). Current-metric 80 ms `full` is `full_mpre3_meancell` s89 **1.054** |
| `stageB_hold_s89_full_im150` | 150 ms / stim×choice | s303 **1.292**; median 1.707 | Old metric. **Keep**; do not FORCE |
| `stageB_hold_s89_full_im150stim` | 150 ms / stim | s303 **1.445**; median 1.848 | Old metric; 7/8 `g_i` collapse |
| `stageB_hold_s89_full_stimonly` | unset / stim | s89 **1.237**; median 1.668 | Old metric; 7/8 `g_i` collapse |
| `stageB_hold_s89_full_im150_meancell` | 150 ms / stim×choice; **new** I/M metric | **s101 1.269** (S 0.098, `d_s≈59`, `g_i=86`); median 1.359 | Does not beat 80 ms `full` s34 or regular `im150_meancell`. No M notch. s12 inc RT +0.21 but `g_i` collapsed |
| `stageB_hold_s89_full_mpre3_meancell` | unset / stim×choice; `m_pre_weight=3` (eval @ 1); **new** metric | **s89 1.054** (S 0.022, `d_s≈50`, `g_i=138`); median 1.665 | First 80 ms `full` with current metric. S-success 2/8. [revisions](modeling_details_revisions.md) 09-16 |
| `stageB_hold_s89_full_mpre3_im150_meancell` | 150 ms / stim×choice; mpre3; **new** metric | **s7 1.186** (S 0.023, `d_s≈41`, `g_i=180`); median 1.338 | Beats `full_im150_meancell` but not 80 ms `full` mpre3 or regular 150 meancell |

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
| 80 ms `full` + mpre3, new metric | `full_mpre3_meancell` **s89** | 1.054 |
| Regular 150 + mpre3 | `mpre3_im150_meancell` **s303** | 1.154 |
| `full` + S + 150 + mpre3 | `full_mpre3_im150_meancell` **s7** | 1.186 |

Cross-window meancell (every seed scored at **both** 80 and 150;
`m_pre_weight=1`; `full` tot includes unsplit-80 S). 80-fit mpre3
regular **s101** wins both (**0.927** / **1.069**). 80-fit `full`
mpre3 **s89** wins both `full` rankings (**1.054** / **1.080**).
150-fit tags do not win the 150 ms ranking. Dump:
`models/stageB_hold_s89_mpre3_crosswindow_meancell_eval.json`.
Topic: [revisions](modeling_details_revisions.md) 09-16b.

Ranked at **80 ms** tot (same seed’s `tot80` / `tot150`; frozen
regular `g_s`/`d_s` print as 0):

| tag | seed | tot80 | tot150 | g_i | g_m | g_s | d_i | d_m | d_s |
|-----|-----:|------:|-------:|----:|----:|----:|----:|----:|----:|
| regular 80 | **34** | **0.999** | 1.107 | 166 | 0 | 0 | 22.4 | 0 | 0 |
| regular 80 mpre3 | **101** | **0.927** | 1.069 | 186 | 0 | 0 | 21.5 | 0 | 0 |
| regular 150 meancell | 34 | 1.029 | 1.210 | 185 | 0 | 0 | 22.8 | 0 | 0 |
| regular 150 mpre3 | 303 | 1.085 | 1.154 | 196 | 0 | 0 | 19.2 | 0 | 0 |
| `full` 80 | **34** | **1.056** | 1.483 | 185 | 0 | 0 | 0 | 0 | 25.7 |
| `full` 80 mpre3 | **89** | **1.054** | 1.080 | 138 | 0.012 | 0 | 0.003 | 0 | 49.7 |
| `full` 150 meancell | 333 | 1.389 | 1.638 | 183 | 0 | 124 | 5.26 | 0 | 36.6 |
| `full` 150 mpre3 | 101 | 1.198 | 1.235 | 200 | 5.78 | 0 | 7.16 | 0 | 48.8 |

Ranked at **150 ms** tot:

| tag | seed | tot80 | tot150 | g_i | g_m | g_s | d_i | d_m | d_s |
|-----|-----:|------:|-------:|----:|----:|----:|----:|----:|----:|
| regular 80 | 303 | 1.035 | **1.086** | 182 | 0 | 0 | 21.0 | 0 | 0 |
| regular 80 mpre3 | **101** | 0.927 | **1.069** | 186 | 0 | 0 | 21.5 | 0 | 0 |
| regular 150 meancell | **12** | 1.213 | **1.089** | 200 | 0.79 | 0 | 26.5 | 0 | 0 |
| regular 150 mpre3 | 303 | 1.085 | 1.154 | 196 | 0 | 0 | 19.2 | 0 | 0 |
| `full` 80 | 45 | 1.182 | **1.180** | 82.7 | 0.030 | 0 | 2.16 | 0.008 | 45.7 |
| `full` 80 mpre3 | **89** | 1.054 | **1.080** | 138 | 0.012 | 0 | 0.003 | 0 | 49.7 |
| `full` 150 meancell | **101** | 1.463 | **1.269** | 85.6 | 0 | 0.013 | 22.9 | 0 | 58.7 |
| `full` 150 mpre3 | **7** | 1.364 | **1.186** | 180 | 0 | 0 | 0 | 0.80 | 41.1 |

Pooled family (80-fit ∪ 150-fit, 16 seeds): 80-fit mpre3 regular
s101 is cheapest at both windows; 80-fit `full` mpre3 s89 is the
only `full` seed cheap at both. 80-fit `full` s34 (1.056) does not
transfer (1.483 at 150); s45 does (1.182 / 1.180) via `d_s≈46`.

---

## mpre3 × 150 / full (`mean_c ‖Δ‖`, 09-16)

Topic: [modeling_details_revisions.md](modeling_details_revisions.md) 09-16;
full arms also [g_s/d_s with I/M](fit_gs_ds_with_im.md). Fit at
`m_pre_weight=3`; **rank at 1** (same as test 5). `g_i` floor still
`1e-12`. Do not FORCE old `mpre3` / `full` / `im150_meancell` dirs.
**Keep `m_pre_weight=1`.**

| Tag | Restrictions | Best seed (eval @ w=1) | Notes |
|-----|--------------|------------------------|-------|
| `stageB_hold_s89_full_mpre3_meancell` | full none; window unset (~80 ms); unsplit-80 S | **s89 1.054** (S 0.022, `d_s≈50`, `g_i=138`); median 1.665 | First 80 ms `full`+S scored with the new metric. S-success 2/8 (s89, s45). Median still a S-fail majority. No M notch |
| `stageB_hold_s89_mpre3_im150_meancell` | regular `12\|13`; 150 ms stim×choice; no S sidecar | **s303 1.154**; median 1.225 | Loses to `im150_meancell` s12 **1.089**. 2/8 `g_i` collapse. Pooled RT s333 **0.806** / inc **+0.379** (not the tot winner) |
| `stageB_hold_s89_full_mpre3_im150_meancell` | full none; 150 ms; unsplit-80 S | **s7 1.186** (S 0.023, `d_s≈41`, `g_i=180`); median 1.338 | Beats `full_im150_meancell` s101 1.269 (S-success 6/8 vs 4/8). Still loses to 80 ms `full` mpre3 and regular 150 meancell. s7 inc RT **+0.207** with `g_i` intact |

---

## 2026-09-16c — split window queued (150 post-stim / 80 pre-move)

Not scored. Unify I/M **traj + prior** windows. Post-stim 150 ms
to get past the stim-onset auditory transient; pre-move **explicit
80 ms** (T=40 extract+score — not production unset, which extracts
T=72 and scores the last 40). Stratum unset = stim×choice. S
sidecar unsplit-80 on `full`. `mean_c ‖Δ‖`. `g_i` floor `1e-12`.
`FORCE=0`. Driver:
[`scripts/submit_fit_stage_b_splitwin.sh`](../scripts/submit_fit_stage_b_splitwin.sh).
Topic: [revisions](modeling_details_revisions.md) 09-16c.

| Tag | Restrictions |
|-----|--------------|
| `stageB_hold_s89_stim150_choice80_meancell` | regular `12\|13`; `m_pre_weight=1`; no S |
| `stageB_hold_s89_mpre3_stim150_choice80_meancell` | regular; `m_pre_weight=3` (eval @ 1); no S |
| `stageB_hold_s89_full_stim150_choice80_meancell` | full none; `m_pre_weight=1`; unsplit-80 S |
| `stageB_hold_s89_full_mpre3_stim150_choice80_meancell` | full; `m_pre_weight=3` (eval @ 1); unsplit-80 S |

---

## 2026-09-15 — catalog opened

Collected tags, freeze / window / stratum, and best seeds from
[retinal then joint](retinal_then_joint_fitting.md),
[modeling_details_revisions.md](modeling_details_revisions.md),
[modeling_details_prior_rt_gaps.md](modeling_details_prior_rt_gaps.md),
and [fit_gs_ds_with_im.md](fit_gs_ds_with_im.md).
