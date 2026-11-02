# Stage B fit variants (catalog)

**Scope:** every Stage B / joint arm we have actually fit and scored.
One row per **run tag** (the `OUT_TAG` in `weights_run_fj_<tag>_…`).
Details, keep/drop, and plots live in the topic journals; this file is
the index of **what was fit**.

**Current focus (2026-11-01, user decision).** From now on, fitting uses the
**original-rule C/M** (`get_sc_table` published parameters: C 85, M 30) with
the **new 10-region S** (S prior VISal VISam LP PO ZI), i.e. target set
`fit_targets_integ/oldrule/`, Stage A newSbin s56. The opt1 (raw slope) and
opt2 (shuffle slope) target sets are no longer the default for new fits. See
"Original-rule target set" (10-30f setup, 11-01 results) below.

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


| Label   | `--freeze`            | Frozen ≈ 0 |
| ------- | --------------------- | ---------- |
| regular | `12                   | 13`        |
| sensory | `6                    | 7          |
| full    | *(none)* / `masknone` | —          |
| onethr  | `11                   | 12         |
| gm0     | `7                    | 9          |


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


| Tag                       | Restrictions                         | Best seed (eval)                                                                                             | Notes                                                                                                                                                  |
| ------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `stageB_hold_s89` regular | mask `12                             | 13`; window / stratum unset; ITI gate on; ITI penalty on;` W_pp ``[0.496, 0.49999]`; two θ;` m_pre_weight=1` | **s101** fair **1.001** (08-13 hybrid-stim table). Later s101-stim batch: **s333 1.015** / median 1.051. mean_c re-score: **s34 0.999** / median 1.030 |
| `stageB_hold_s89` sensory | mask `6                              | 7                                                                                                            | 8                                                                                                                                                      |
| WEIGHTS_REL               | notebook weights (not a Stage B tag) | fair **1.260** on the 08-13 batch                                                                            | RT reference (only slightly +inc R²)                                                                                                                   |


Stage-A source for the hybrid: `retinal_run_fr_retinal_masknone_s89`.

---



## Tests 1–6 (regular mask; old `‖mean_c Δ‖` unless noted)

All vs production regular on the s101-stim batch unless the row says
as-fitted. Topic: [modeling_details_revisions.md](modeling_details_revisions.md).


| Test | Tag                         | Restrictions (vs regular)                                                                         | Best seed                                           | Keep / drop                                                               |
| ---- | --------------------------- | ------------------------------------------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------- |
| 1    | `stageB_hold_s89_poffset`   | `P_offset` always on through the ITI                                                              | s7 **1.566** as fitted (gated restore 1.809)        | **Drop** (keep the 100 ms pre-stim gate)                                  |
| 2    | `stageB_hold_s89_noiti`     | I/M ITI zero-penalty off                                                                          | s34 **0.985** with penalty restored                 | **Drop** as default (median worse)                                        |
| 3    | `stageB_hold_s89_wpplarge`  | init `W_pp=0.499` (10 s); **same** box `[0.496, 0.49999]`                                         | **s101 0.947**                                      | Init only; **keep the 2.5 s floor**                                       |
| 3    | `stageB_hold_s89_wppopen`   | init 0.499; box floor **0.20**                                                                    | s7 **1.055**; median 1.159                          | **Drop** open floor                                                       |
| 3    | `stageB_hold_s89_wppsmall`  | init `W_pp=0.45` (200 ms); floor 0.20                                                             | s89 **1.011**; median 1.123                         | **Drop**                                                                  |
| 4    | `stageB_hold_s89_onethr`    | `θ_c=θ_d`; freeze `11                                                                             | 12                                                  | 13`                                                                       |
| 5    | `stageB_hold_s89_mpre3`     | `m_pre_weight=3` (eval re-scored at 1)                                                            | s303 fair **0.981**; median 1.081                   | **Keep** `m_pre_weight=1` (RT did not beat regular s101 0.787)            |
| 6    | `stageB_hold_s89_im150`     | `prior_window_ms=150`; stim×choice; **old metric**                                                | as fitted s45 **1.313**; → production s34 **1.080** | **Keep production window unset** on that ranking. Dirs kept; do not FORCE |
| 6    | `stageB_hold_s89_im150stim` | 150 ms + `prior_stratum=stim`. Morning 0.1-floor **FORCE-replaced** by 09-08e `g_i` floor `1e-12` | 1e-12 as fitted s7 **1.188**; → prod s34 **1.252**  | **Drop** stim-only I/M. Restore `g_i` floor **0.1**                       |
| 6    | `stageB_hold_s89_stimonly`  | window unset + `prior_stratum=stim`; `g_i` floor `1e-12`                                          | as fitted s303 **1.060**; → prod s34 **1.443**      | **Drop**                                                                  |


`mpre2` was an optional tag; not a scored 8-seed campaign.

---



## Metric fix (`mean_c ‖Δ‖`, 09-08f on)

Topic: [modeling_details_revisions.md](modeling_details_revisions.md) 09-08f;
[modeling_details_prior_rt_gaps.md](modeling_details_prior_rt_gaps.md).


| Tag                                    | Restrictions                                                                                                           | Best seed                                           | Notes                                                                                                                     |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `stageB_hold_s89_im150_meancell`       | regular mask; 150 ms stim×choice; **new** I/M metric; `include_stim=false`; `g_i` floor still `1e-12` (s101 collapsed) | as fitted **s12 1.089**; → production s34 **1.029** | Only regular-mask 150 ms **fit** with the corrected metric. I/M now rise 80→150. No M notch. Best pooled RT s12 **0.834** |
| `stageB_hold_s89_mpre3_im150_meancell` | same + `m_pre_weight=3` (eval @ 1)                                                                                     | as fitted **s303 1.154**; median 1.225              | Loses to `im150_meancell` on best tot. [revisions](modeling_details_revisions.md) 09-16                                   |
| regular 80 ms                          | *(no meancell rerun)*                                                                                                  | —                                                   | Production `stageB_hold_s89` regular was **not** re-fit after 09-08f; later tables re-**score** it with current code      |


---



## M-shape (freeze `g_m`/`d_m`)

Topic: [modeling_details_prior_rt_gaps.md](modeling_details_prior_rt_gaps.md).
Mask `7\|9\|12\|13`. No 60–70 ms M notch on either arm.


| Tag                         | Restrictions                                                                      | Best seed                              | Notes                                                                                        |
| --------------------------- | --------------------------------------------------------------------------------- | -------------------------------------- | -------------------------------------------------------------------------------------------- |
| `stageB_hold_s89_gm0`       | freeze `g_m`/`d_m`; **80 ms** window (mis-wired vs M-shape aim); usual `W_mm` box | **s7 0.967**; median 1.022             | Cheap 80 ms tot win. Window cannot see the post-80 climb. **Do not reuse this tag**          |
| `stageB_hold_s89_mleak`     | same freeze + `W_mm∈[0.10,0.15]`, init 0.15; 80 ms                                | s34 **0.980**; median 1.117            | Late M dies; several `g_i` collapses. **Drop** the `W_mm` cap                                |
| `stageB_hold_s89_gm0_im150` | same freeze; **150 ms** stim×choice; new metric                                   | as fitted **s101 1.128**; median 1.347 | Loses vs regular@150 (s303 **1.086** / median 1.162). **Drop** the freeze as an M-shape keep |


---



## `full` + unsplit-80 S prior

Topic: [fit_gs_ds_with_im.md](fit_gs_ds_with_im.md).
`VARIANTS=full:`, `include_stim=1`, model S `stratum_s=stim`, sidecar
`fit_targets/data_act_block_duringstim_s_unsplit80.npy`. Rank **fair** =
traj + I/M + S nSSE + `L_S`. Morning **0.1-floor** `full` dirs were
FORCE-replaced by the 1e-12 campaign.


| Tag                                          | I/M window / stratum                                             | Best seed (as fitted fair)                                 | Notes                                                                                                                         |
| -------------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `stageB_hold_s89_full` (1e-12; current dirs) | unset / stim×choice                                              | **s34 1.057** (S 0.028, `d_s≈26`, `g_s≈0`, `g_i=185`)      | Old metric. Median 1.697 (half the seeds do nothing on S). Current-metric 80 ms `full` is `full_mpre3_meancell` s89 **1.054** |
| `stageB_hold_s89_full_im150`                 | 150 ms / stim×choice                                             | s303 **1.292**; median 1.707                               | Old metric. **Keep**; do not FORCE                                                                                            |
| `stageB_hold_s89_full_im150stim`             | 150 ms / stim                                                    | s303 **1.445**; median 1.848                               | Old metric; 7/8 `g_i` collapse                                                                                                |
| `stageB_hold_s89_full_stimonly`              | unset / stim                                                     | s89 **1.237**; median 1.668                                | Old metric; 7/8 `g_i` collapse                                                                                                |
| `stageB_hold_s89_full_im150_meancell`        | 150 ms / stim×choice; **new** I/M metric                         | **s101 1.269** (S 0.098, `d_s≈59`, `g_i=86`); median 1.359 | Does not beat 80 ms `full` s34 or regular `im150_meancell`. No M notch. s12 inc RT +0.21 but `g_i` collapsed                  |
| `stageB_hold_s89_full_mpre3_meancell`        | unset / stim×choice; `m_pre_weight=3` (eval @ 1); **new** metric | **s89 1.054** (S 0.022, `d_s≈50`, `g_i=138`); median 1.665 | First 80 ms `full` with current metric. S-success 2/8. [revisions](modeling_details_revisions.md) 09-16                       |
| `stageB_hold_s89_full_mpre3_im150_meancell`  | 150 ms / stim×choice; mpre3; **new** metric                      | **s7 1.186** (S 0.023, `d_s≈41`, `g_i=180`); median 1.338  | Beats `full_im150_meancell` but not 80 ms `full` mpre3 or regular 150 meancell                                                |


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


| Family                                  | Best arm × seed                        | Number                    |
| --------------------------------------- | -------------------------------------- | ------------------------- |
| Production regular (08-13, hybrid stim) | regular **s101**                       | fair 1.001                |
| Production regular (s101-stim batch)    | regular **s333**                       | 1.015                     |
| Production regular (mean_c re-score)    | regular **s34**                        | 0.999                     |
| Test 3 tot (`wpplarge`)                 | `wpplarge` **s101**                    | 0.947                     |
| Test 2 tot (`noiti`, penalty restored)  | `noiti` **s34**                        | 0.985                     |
| 80 ms freeze `g_m`/`d_m` (`gm0`)        | `gm0` **s7**                           | 0.967                     |
| Regular 150 ms, new metric              | `im150_meancell` **s12**               | 1.089                     |
| `full` + S, 80 ms                       | `full` **s34**                         | 1.057                     |
| `full` + S, 150 ms, new metric          | `full_im150_meancell` **s101**         | 1.269                     |
| 80 ms `full` + mpre3, new metric        | `full_mpre3_meancell` **s89**          | 1.054                     |
| 80 ms `full` s89ft `wii` (extras=0)     | `s89ft_imshape_wii_meancell` **s7**    | **1.014**                 |
| Regular 150 + mpre3                     | `mpre3_im150_meancell` **s303**        | 1.154                     |
| `full` + S + 150 + mpre3                | `full_mpre3_im150_meancell` **s7**     | 1.186                     |
| Regular split 150/80 (fair @80)         | `stim150_choice80_meancell` **s7**     | 1.050                     |
| `full` split mpre3 (fair @80)           | `full_mpre3_stim150_choice80` **s303** | 1.138                     |
| Regular ranked @ stim150/choice80       | 80-fit mpre3 **s101**                  | **0.905**                 |
| `full` ranked @ stim150/choice80        | 80-fit mpre3 **s89**                   | **0.969** (tot_noS 0.947) |


Cross-window meancell (every seed scored at **both** 80 and 150;
`m_pre_weight=1`; `full` tot includes unsplit-80 S). 80-fit mpre3
regular **s101** still wins both (**0.927** / **1.069**). 80-fit
`full` mpre3 **s89** still wins both `full` rankings (**1.054** /
**1.080**). Split 150/80 and 150-fit tags do not win either
ranking. Dumps:
`models/stageB_hold_s89_mpre3_crosswindow_meancell_eval.json`,
`models/stageB_hold_s89_splitwin_crosswindow_meancell_eval.json`.
Topic: [revisions](modeling_details_revisions.md) 09-16b / 09-17.

Ranked at **80 ms** tot (same seed’s `tot80` / `tot150`; frozen
regular `g_s`/`d_s` print as 0). Regular and `full` scored
**separately** — `full` tot includes unsplit-80 S nSSE.
`tot_noS` = tot minus that S term (S is the same at 80 and 150).

**Regular** (6 tags, best seed each, ordered by tot80):


| tag                  | seed    | tot80     | tot150 | g_i | g_m | g_s | d_i  | d_m   | d_s |
| -------------------- | ------- | --------- | ------ | --- | --- | --- | ---- | ----- | --- |
| regular 80 mpre3     | **101** | **0.927** | 1.069  | 186 | 0   | 0   | 21.5 | 0     | 0   |
| regular 80           | 34      | 0.999     | 1.107  | 166 | 0   | 0   | 22.4 | 0     | 0   |
| regular 150 meancell | 34      | 1.029     | 1.210  | 185 | 0   | 0   | 22.8 | 0     | 0   |
| regular split 150/80 | 7       | 1.050     | 1.223  | 182 | 0   | 0   | 22.4 | 0     | 0   |
| regular split mpre3  | 101     | 1.060     | 1.361  | 200 | 0   | 0   | 18.9 | 0.399 | 0   |
| regular 150 mpre3    | 303     | 1.085     | 1.154  | 196 | 0   | 0   | 19.2 | 0     | 0   |


`full` (`tot_noS` sits next to regular tot):


| tag                 | seed   | tot80     | tot_noS80 | tot150 | tot_noS150 | g_i | g_m   | g_s | d_i   | d_m | d_s  |
| ------------------- | ------ | --------- | --------- | ------ | ---------- | --- | ----- | --- | ----- | --- | ---- |
| `full` 80 mpre3     | **89** | **1.054** | 1.032     | 1.080  | 1.058      | 138 | 0.012 | 0   | 0.003 | 0   | 49.7 |
| `full` 80           | 34     | 1.056     | 1.028     | 1.483  | 1.455      | 185 | 0     | 0   | 0     | 0   | 25.7 |
| `full` split mpre3  | 303    | 1.138     | 1.110     | 1.612  | 1.585      | 183 | 0     | 0   | 0     | 0   | 21.9 |
| `full` 150 mpre3    | 101    | 1.198     | 1.140     | 1.235  | 1.176      | 200 | 5.78  | 0   | 7.16  | 0   | 48.8 |
| `full` 150 meancell | 333    | 1.389     | 1.372     | 1.638  | 1.620      | 183 | 0     | 124 | 5.26  | 0   | 36.6 |
| `full` split 150/80 | 89     | 1.466     | 1.356     | 1.876  | 1.767      | 181 | 0     | 0   | 0     | 0   | 24.6 |


Ranked at **150 ms** tot:

**Regular:**


| tag                  | seed    | tot80 | tot150    | g_i | g_m   | g_s | d_i  | d_m | d_s |
| -------------------- | ------- | ----- | --------- | --- | ----- | --- | ---- | --- | --- |
| regular 80 mpre3     | **101** | 0.927 | **1.069** | 186 | 0     | 0   | 21.5 | 0   | 0   |
| regular 80           | 303     | 1.035 | 1.086     | 182 | 0     | 0   | 21.0 | 0   | 0   |
| regular 150 meancell | 12      | 1.213 | 1.089     | 200 | 0.79  | 0   | 26.5 | 0   | 0   |
| regular split mpre3  | 12      | 1.129 | 1.141     | 162 | 0     | 0   | 14.9 | 0   | 0   |
| regular 150 mpre3    | 303     | 1.085 | 1.154     | 196 | 0     | 0   | 19.2 | 0   | 0   |
| regular split 150/80 | 45      | 1.295 | 1.222     | 181 | 0.001 | 0   | 22.8 | 0   | 0   |


`full`**:**


| tag                 | seed   | tot80 | tot_noS80 | tot150    | tot_noS150 | g_i  | g_m   | g_s   | d_i   | d_m   | d_s  |
| ------------------- | ------ | ----- | --------- | --------- | ---------- | ---- | ----- | ----- | ----- | ----- | ---- |
| `full` 80 mpre3     | **89** | 1.054 | 1.032     | **1.080** | 1.058      | 138  | 0.012 | 0     | 0.003 | 0     | 49.7 |
| `full` 80           | 45     | 1.182 | 1.155     | 1.180     | 1.153      | 82.7 | 0.030 | 0     | 2.16  | 0.008 | 45.7 |
| `full` 150 mpre3    | 7      | 1.364 | 1.341     | 1.186     | 1.162      | 180  | 0     | 0     | 0     | 0.80  | 41.1 |
| `full` split mpre3  | 333    | 1.506 | 1.470     | 1.199     | 1.162      | 141  | 0.014 | 0.80  | 9.16  | 0     | 49.2 |
| `full` 150 meancell | 101    | 1.463 | 1.364     | 1.269     | 1.170      | 85.6 | 0     | 0.013 | 22.9  | 0     | 58.7 |
| `full` split 150/80 | 12     | 1.605 | 1.342     | 1.390     | 1.127      | 0    | 0     | 0     | 22.3  | 0     | 35.4 |


Pooled family (80-fit ∪ 150-fit ∪ split, 24 seeds): 80-fit mpre3
regular s101 is still cheapest at both windows; 80-fit `full`
mpre3 s89 is still the only `full` seed cheap at both. Split
150/80 does not take any family win. 80-fit `full` s34 (1.056)
does not transfer (1.483 at 150); s45 does (1.182 / 1.180) via
`d_s≈46`. Cheapest `full` tot_noS at 80 is s34 **1.028** (still
above regular 0.927); at 150, s89 **1.058** sits just under
regular s101 **1.069**.

Ranked at **stim150 / choice80** (traj T=75/40 + split prior;
`m_pre_weight=1`; same shared stim). Regular and `full` scored
**separately** — `full` tot includes unsplit-80 S nSSE, so it is
not comparable to regular tot. `tot_noS` = traj + I/M + `L_S`
(drop S nSSE). Dump:
`models/stageB_hold_s89_stim150_choice80_rank_eval.json`.
Driver:
`[scripts/_tmp_stim150_choice80_rank_eval.py](../scripts/_tmp_stim150_choice80_rank_eval.py)`.
Topic: [revisions](modeling_details_revisions.md) 09-18.

**Regular** (6 tags, best seed each, ordered by tot):


| tag                  | seed    | tot150/80 | g_i | g_m | g_s | d_i  | d_m | d_s |
| -------------------- | ------- | --------- | --- | --- | --- | ---- | --- | --- |
| regular 80 mpre3     | **101** | **0.905** | 186 | 0   | 0   | 21.5 | 0   | 0   |
| regular 80           | 34      | 0.926     | 166 | 0   | 0   | 22.4 | 0   | 0   |
| regular 150 meancell | 34      | 0.952     | 185 | 0   | 0   | 22.8 | 0   | 0   |
| regular split 150/80 | 7       | 0.971     | 182 | 0   | 0   | 22.4 | 0   | 0   |
| regular 150 mpre3    | 303     | 0.999     | 196 | 0   | 0   | 19.2 | 0   | 0   |
| regular split mpre3  | 12      | 1.028     | 162 | 0   | 0   | 14.9 | 0   | 0   |


`full` (6 tags; `tot_noS` sits next to regular tot):


| tag                 | seed   | tot150/80 | tot_noS | g_i  | g_m   | g_s  | d_i   | d_m   | d_s  |
| ------------------- | ------ | --------- | ------- | ---- | ----- | ---- | ----- | ----- | ---- |
| `full` 80 mpre3     | **89** | **0.969** | 0.947   | 138  | 0.012 | 0    | 0.003 | 0     | 49.7 |
| `full` 150 mpre3    | 101    | 1.070     | 1.011   | 200  | 5.78  | 0    | 7.16  | 0     | 48.8 |
| `full` 80           | 45     | 1.072     | 1.045   | 82.7 | 0.030 | 0    | 2.16  | 0.008 | 45.7 |
| `full` split mpre3  | 333    | 1.116     | 1.079   | 141  | 0.014 | 0.80 | 9.16  | 0     | 49.2 |
| `full` 150 meancell | 303    | 1.152     | 1.048   | 200  | 0     | 0    | 23.2  | 0     | 63.8 |
| `full` split 150/80 | 7      | 1.292     | 1.222   | 190  | 0.067 | 0    | 19.6  | 0     | 48.7 |


Even on this window, 80-fit mpre3 still wins both families.
Cheapest `full` tot_noS (**0.947**, s89) is still above cheapest
regular (**0.905**, s101). Split as-fitted does not take either
win (regular s7 0.971; `full` s7 1.292 / tot_noS 1.222).

---



## mpre3 × 150 / full (`mean_c ‖Δ‖`, 09-16)

Topic: [modeling_details_revisions.md](modeling_details_revisions.md) 09-16;
full arms also [g_s/d_s with I/M](fit_gs_ds_with_im.md). Fit at
`m_pre_weight=3`; **rank at 1** (same as test 5). `g_i` floor still
`1e-12`. Do not FORCE old `mpre3` / `full` / `im150_meancell` dirs.
**Keep** `m_pre_weight=1`**.**


| Tag                                         | Restrictions                                   | Best seed (eval @ w=1)                                     | Notes                                                                                                                                                             |
| ------------------------------------------- | ---------------------------------------------- | ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `stageB_hold_s89_full_mpre3_meancell`       | full none; window unset (~80 ms); unsplit-80 S | **s89 1.054** (S 0.022, `d_s≈50`, `g_i=138`); median 1.665 | First 80 ms `full`+S scored with the new metric. S-success 2/8 (s89, s45). Median still a S-fail majority. No M notch                                             |
| `stageB_hold_s89_mpre3_im150_meancell`      | regular `12                                    | 13`; 150 ms stim×choice; no S sidecar                      | **s303 1.154**; median 1.225                                                                                                                                      |
| `stageB_hold_s89_full_mpre3_im150_meancell` | full none; 150 ms; unsplit-80 S                | **s7 1.186** (S 0.023, `d_s≈41`, `g_i=180`); median 1.338  | Beats `full_im150_meancell` s101 1.269 (S-success 6/8 vs 4/8). Still loses to 80 ms `full` mpre3 and regular 150 meancell. s7 inc RT **+0.207** with `g_i` intact |


---



## 2026-10-28 — integrator rule, option 1 vs option 2 (set up, not submitted)

Topic: [paper main figures](paper_main_figures.md) 2026-10-28. Arm: `full` mpre3 (`ARMS=full`, `m_pre_weight` 3, no freeze, I/M window unset = 80 ms, stim×choice stratum, unsplit-80 S prior on, seeds `7 12 34 45 89 101 303 333`, hybrid s89 warm start). Only the targets differ.

| Tag | I (regions) | M | S RMS (`avg_mean_R`) | S prior |
|---|---:|---:|---|---|
| `stageB_hold_s89_full_mpre3_meancell_integ_opt1_rawslope` | 84 (choice significant & slope > 0) | 31 | the 10 sensory regions | VISal VISam LP PO ZI |
| `stageB_hold_s89_full_mpre3_meancell_integ_opt2_shuffleslope` | 57 (& whole-curve slope beats shuffles, FDR ≤ 0.01) | 31 | same | same |

**Targets** live in `fit_targets_integ/<opt>/`; `regions.json` lists I, M, S, S prior and P. The fitter takes them through `FIT_TARGETS_DIR`.
- **Prior curves:** built on the laptop by `scripts/build_fit_targets_integ.py` from the res-standard staging (act-kernel prior). They use the old recipes: `load_group(alpha=1, simple)` for I/M, and the unsplit-80 recipe for S. On the old 81/26 lists this source reproduces the old I prior target (mean 0.0635 vs 0.0632, r = 0.85 stim / 0.56 choice).
- **I/M traj and S RMS:** built on ORCD by `scripts/build_fit_traj_targets_from_cache.py`, a single pass over `insertion_cache`. It uses the notebook keys, stim 0–200 ms, move −150–0 ms, and the standard α=0.2 act prior with true 0.5 dropped. It also rebuilds the old lists (`legacy`) and checks them against `fit_targets` in `traj_targets_check.json`.
- **P:** copied from `fit_targets`.

**How the new targets differ from production.** All targets are new, so tot is not comparable to production rows; compare opt1 and opt2 with each other.
- The new S prior (5 regions) sits at about 0.02–0.03 and rises over 0–80 ms. The old one (13 regions) was flat at about 0.045.
- The M prior is lower than the old target (`manifold/paper_figs/fit_targets_integ/prior_targets.png`).

**Run isolation.** A `FIT_TARGETS_DIR` job runs in `fit_workdirs/<OUT_TAG>_<mtype>_s<seed>/` with its own target links. Before this, every job linked targets into the repo root, and the loss re-reads `data_*.npy` from cwd on each evaluation. Two target sets at once would have read each other's files.

Submit: `STEP=targets`, check the legacy match, then `STEP=fits` (`scripts/submit_fit_targets_integ.sh`).

**Review (same day).**
- **Loss eval.** One eval of `full_mpre3_meancell` s89 (shared stim 12345, bps 20) runs on the new target files without errors. Prior terms with the old weights, old → new targets:

  | prior term | old targets | new targets |
  |---|---:|---:|
  | I stim | 0.020 | 0.119 |
  | M stim | 0.131 | 0.507 |
  | S | 0.023 | 0.801 |

  The S jump comes from the regions, not the source: the old 13 regions from the res-standard staging give 0.041 (old target 0.044), while the 5 current regions give 0.025.
- **Not yet checkable.** Traj and S RMS can only be judged after the ORCD build; 3–7 laptop insertions are noise.
- **Warm start (superseded below).** Both arms started from the s89 hybrid. Its Stage-A retinal parameters were fit to the old `avg_mean_R`, and Stage-1 DE holds them. The S fit therefore starts mismatched in both arms. That is fair between the arms, but a Stage-A refit on the new S RMS may be needed for good absolute S.
- **Shared cells.** As in Figure 4 i, I shares LGd, LP, PF, PO and VISp with S, and M shares ZI. Those cells enter both data targets. The old lists shared FRP and NOT between I and `stim_regs`.
- **Fail closed.** The traj builder now writes nothing if more than 5% of insertions fail, or if any I/M key or S contrast has no cells.

**Stage A first (same day).** The retinal stage is now refit on the new S RMS curves before the two arms (`scripts/submit_fit_targets_integ.sh`, four steps).
1. Laptop: region lists and prior targets (done).
2. `STEP=targets`: traj and S RMS targets, one ORCD job.
3. `STEP=retinal`: Stage A on the new `avg_mean_R`, production protocol (retinal, mask none, `de_cma_local`, seeds 56 34 78 89 202), tag `retinal_newS`. `avg_mean_R` must be identical in both target dirs, so one Stage A serves both arms.
4. `STEP=fits RETINAL_JSON=<chosen final>`: builds the hybrid `stage_b_hybrid_WEIGHTS_REL_retinal_newS_s<seed>.json` (WEIGHTS_REL ∪ the new retinal) and runs both arms from it. The tags become `stageB_hold_newS<seed>_full_mpre3_meancell_integ_<opt>`, replacing the `stageB_hold_s89_…` tags in the table above.

Pick the Stage A seed by shared-stim `L_S` (`plot_retinal_fit_s.py`, bps 20, seed 12345), not by recorded loss, as for s89. That ranking runs on a new `avg_mean_R`; on the old target it reproduces the s89 score of 0.559. Stage A jobs with `FIT_TARGETS_DIR` also run in their own `fit_workdirs/` folder.

**One submission (same day).** `STEP=all bash scripts/submit_fit_targets_integ.sh` queues the whole thing as a Slurm chain, all on `pi_fiete`:
1. The targets job.
2. Five Stage A jobs (`afterok` on the targets job).
3. A selector job, `run_fit_integ_select_slurm.sh` (`afterany` on Stage A). It ranks the finished Stage A finals by shared-stim `L_S` on the new `avg_mean_R` and submits `STEP=fits` with the best one. It is not requeued, so Stage B cannot be submitted twice.

Every dependent job uses `--kill-on-invalid-dep`. A dry run with a fake `sbatch` produced exactly this chain, and the fits step built the hybrid and 2 × 8 jobs with the right `FIT_TARGETS_DIR` and tags. The recipe check (`traj_targets_check.json`) does not stop the chain: read it when the targets job ends, and `scancel` if the legacy rebuild does not match `fit_targets`. A Stage A seed that times out is left out of the ranking.

**Results (same day, scored).**
- **Targets check.** The I/M legacy rebuild matches `fit_targets`: r 0.97–1.00, scale 1.00–1.04. Coverage: 683/696 insertions, none skipped. S matches less well at low contrast: r 0.79 at c=0 and 0.90 at c=0.0625, scale 1.31–1.43 at c=0.0625–0.125. The new 10-region S curve at c=1 peaks at about 0.52 near 50 ms; the old one peaked at about 0.70–0.82.
- **Stage A.** The selector picked s34, with shared `L_S` 0.459 and R² 0.53. The other seeds scored s56 0.483, s78 0.502, s89 0.564 and s202 0.833. s202 had the best recorded loss (0.368), so recorded loss would have picked the worst seed.
- **Eval method.** Script: `scripts/_tmp_integ_opt_eval_plots.py`, one process per arm. Each process sets `FIT_TARGETS_DIR` to its arm and runs in its own `fit_workdirs/eval_<opt>` folder. Shared stim: bps 20, seed 12345, from regular s101. Ranking uses fair (traj + prior + `L_S`) at `m_pre_weight` 1. As a reference, the production `stageB_hold_s89_full_mpre3_meancell` seeds are scored on the same targets.
- **Output files.**
  - Eval: `alyx/models/stageB_hold_newS34_integ_<opt>_eval.json`.
  - Plots, in each `weights_run_fj_stageB_hold_newS34_full_mpre3_meancell_integ_<opt>_full_masknone_s*/`: IM_pre/post, P_fit, S_fit, prior_effects, S/I/M_iti and `psychometric_model_vs_data_actprior/`. `prior_effects` is drawn on the scored windows only: 0–80 ms after stim and −80–0 ms before the move. `plot_one(prior_plot_window=80)`; full arms otherwise default to drawing 150 ms. The SSE in each panel title equals the eval prior term.
  - Plot summary: `…_<opt>_plot_summary.json`.

| targets | run | best fair (seed) | median fair | traj | I/M prior | S prior | `L_S` |
|---|---|---|---:|---:|---:|---:|---:|
| opt1 | opt1 refit | 1.314 (s45) | 1.682 | 0.398 | 0.237 | 0.424 | 0.656 |
| opt1 | production s89 | 1.818 (s7) | 2.400 | 0.319 | 0.613 | 0.451 | 0.861 |
| opt2 | opt2 refit | 1.476 (s12) | 1.795 | 0.375 | 0.258 | 0.499 | 0.670 |
| opt2 | production s89 | 1.858 (s7) | 2.329 | 0.299 | 0.635 | 0.451 | 0.861 |

The traj through `L_S` columns are medians over seeds.

- **What the refit changes.** On both target sets the refit beats production mainly on the I/M prior (about 0.24 vs 0.62) and on `L_S` (0.66 vs 0.86). Traj gets slightly worse (0.38–0.40 vs 0.30–0.32).
- **`L_S` in Stage B: not drift (corrected below).** The Stage B `L_S` of 0.49–0.68 sits above Stage A's 0.459, but that difference is stimulus-draw noise. See "S fit noise".
- **The two prior targets are almost the same.** The 84-region and 57-region I lists give nearly the same I prior curve: during stim, `r_int` ends at 0.073 vs 0.079; during choice, 0.078 vs 0.086. Most of the opt1/opt2 difference comes from the I traj target and from optimizer variance.
- **Integrator gain.** opt1 turns off the I prior in 3/8 seeds (g_i about 0: s7, s12, s333). opt2 does so in 2/8 (s333, plus s45 at g_i ≈ 2). The best opt1 seed, s45 (g_i 42), gives a model I prior that is flat to falling: 0.036 at 0 ms, 0.032 at 80 ms, and falling before the move. The data I prior rises. Its lower loss comes with the wrong shape.
- **opt2 shape and behaviour.** The best opt2 seed, s12 (g_i 171, d_i 7.1), has a rising I prior in both windows. Its behaviour is poor: perf R² 0.48, RT split −3.2, with incongruent trials too wrong and too slow.
- **Production behaviour reference.** For production full80, perf R² is 0.75–0.91 and RT split ranges from −1.37 to 0.48.

| arm, seed | fair | I prior 0→80 ms | perf R² | RT R² | RT split |
|---|---:|---|---:|---:|---:|
| opt1 s45 (best fair) | 1.314 | 0.036→0.032 | 0.847 | 0.269 | −0.03 |
| opt1 s7 | 1.368 | 0.039→0.041 | 0.929 | 0.545 | 0.41 |
| opt2 s12 (best fair) | 1.476 | 0.044→0.059 | 0.481 | 0.303 | −3.22 |
| opt2 s101 | 1.602 | 0.037→0.043 | 0.884 | 0.008 | −0.30 |
| opt2 s303 | 1.732 | 0.037→0.050 | 0.974 | 0.775 | 0.67 |

**Per-term breakdown, four seeds.** Same shared stim, `m_pre_weight` 1, each arm scored on its own targets. C = integrator (I). Rows sum to the totals above: traj = C + M + P penalties; prior = S + C + M. g ≈ 0 means below 1e-4, and its d is then irrelevant.

| | opt1 s45 | opt1 s7 | opt2 s12 | opt2 s303 |
|---|---:|---:|---:|---:|
| `L_S` (S RMS) | 0.527 | 0.526 | 0.606 | 0.652 |
| traj C | 0.125 | 0.101 | 0.165 | 0.111 |
| traj M | 0.255 | 0.224 | 0.132 | 0.234 |
| traj P penalties | 0.026 | 0.010 | 0.000 | 0.020 |
| prior S (0–80 ms) | 0.062 | 0.146 | 0.213 | 0.460 |
| prior C stim / move | 0.027 / 0.183 | 0.014 / 0.136 | 0.033 / 0.068 | 0.027 / 0.048 |
| prior M stim / move | 0.085 / 0.022 | 0.116 / 0.094 | 0.149 / 0.110 | 0.096 / 0.083 |
| fair | 1.314 | 1.368 | 1.476 | 1.732 |
| g_s / d_s | ≈0 / 45.2 | ≈0 / 43.8 | 0.34 / 6.2 | ≈0 / 3.9 |
| g_i / d_i | 41.8 / 11.0 | 0.009 / 14.3 | 171 / 7.1 | 195 / 22.0 |
| g_m / d_m | ≈0 / ≈0 | ≈0 / ≈0 | 3.3 / ≈0 | ≈0 / ≈0 |
| psychometric R² | 0.847 | 0.929 | 0.481 | 0.974 |
| RT R² | 0.269 | 0.545 | 0.303 | 0.775 |
| RT split R² (con / inc) | −0.03 (0.45 / −0.55) | 0.41 (0.65 / 0.16) | −3.22 (0.86 / −7.59) | 0.67 (0.83 / 0.51) |

- **Move-aligned C prior.** opt1 misses it (0.14–0.18); opt2 fits it about 3× better (0.05–0.07).
- **opt1 s45.** It wins fair through a low S prior and a low move-aligned M prior, with g_s and g_m off.
- **opt1 s7.** Fits with the C prior off.
- **g_m.** About 0 in all four seeds except opt2 s12 (3.3).

**S fit noise (same day).** The question was whether the S fit is worse than before.

1. **S depends only on the retinal parameters.** opt1 s101 has the same retinal values as Stage A s34. Their S curves, and so their `L_S`, are identical whether run with the Stage A network and thresholds or the Stage B ones (0.669 vs 0.669). Swapping only the thresholds changes nothing either. Stage B barely moves the retinal parameters: 14/16 seeds stay within about 1% of s34. The exceptions are opt1 s7 (τ_a 400) and opt2 s303 (τ_a 219).
2. **One 20-block draw is very noisy.** For identical parameters, `L_S` was 0.459 on the selector's stim draw and 0.669 on the eval draw.
3. **Mean over 8 stim draws** (bps 20, `make_shared_stimuli`, seeds 1–8):

| Stage A | target | `L_S` mean ± sd (range) | c=1 R² |
|---|---|---|---:|
| new s34 | new S (10 regions) | **0.582 ± 0.161** (0.35–0.81) | 0.83 |
| new s56 | new | 0.644 ± 0.164 | 0.78 |
| new s78 | new | 0.654 ± 0.165 | 0.78 |
| new s89 | new | 0.700 ± 0.176 | 0.85 |
| new s202 | new | 0.746 ± 0.205 | 0.83 |
| old s89 (production) | old S (`fit_targets`) | 0.676 ± 0.158 | 0.87 |

- **The new S fit is not worse.** s34 on the new target scores 0.58, against 0.68 for production s89 on the old target. The c=1 R² is about the same.
- **s34 stays the right Stage A pick on the mean.** The single-draw `L_S` and Stage B `L_S` values above are inside the ±0.16 noise. They cannot rank seeds or arms.
- **Shape mismatch that remains at c=1.** The new data rises gradually from about 20 ms, peaks at about 0.52, then ramps after 80 ms. The model stays flat until about 30 ms, steps up and overshoots to about 0.67. The pooled 10-region curve probably mixes early (LGd/LP/VIS) and later (PO/ZI/PF) latencies, and one retinal latency cannot smear its onset like that. The old 8-region curve had a sharper onset. Even so, the c=1 R² averaged over draws is only slightly lower on the new target (0.83 vs 0.87).

**Open.**
- Neither arm has a seed that is best on fair, has a rising I prior and has good behaviour at once.
- opt2 s303 is the best balanced seed: third on fair, rising I and the best behaviour of all 16 fits.
- The comparison is 8 seeds per arm; the seed-to-seed spread (fair 1.3–2.1) is larger than the arm difference.

**Code changes (same day).**
- **g_i floor back to 0.1.** Set in `fit_joint.NATIVE_BOUNDS` and in `fit_weights._log_bounds_weights_v2`. It had sat at 1e-12 since commit 6bfa9bc, although the revisions journal said twice to restore it. Every integ fit above ran with 1e-12; opt1 s7, s12 and s333 collapsed g_i.
- **Behaviour stays out of the loss on purpose.** The commented-out `loss_perf_with_data` block in `fit_weights.loss_weights_core_v2` is deleted. Behaviour is a held-out check only. The fit drivers no longer load `behavior.npy`: `run_fit_joint.py`, `run_fit_weights.py`, the `fit_weights.py` main and the `bench_fit_*` scripts now pass `behavior = None`, so the fitter never sees behaviour data.
- **`--zero-g-m-d-m` added to `run_fit_joint.py`.** It sets g_m = d_m = LOG_ZERO in an external warm start, and requires 7|9 to be frozen, so `--freeze-hold` keeps them at 0. The env var is `ZERO_G_M_D_M=1`.

**Warm fine-tune, set up (same day).** Script: `scripts/submit_fit_integ_finetune.sh`.
- **Starts:** opt1 s45, opt1 s7, opt2 s12 and opt2 s303, each on its own arm's targets.
- **Method:** `cma_only`, restart seeds `89 7 12 45`, 4 × 4 = 16 jobs on `pi_fiete`.
- **Free:** W_mm, W_mi, g_i, d_i, g_s, d_s; the local polish uses the same set (2,5,6,8,12,13).
- **Fixed:** g_m and d_m at 0. All other parameters are held at the source fit, including W_ii, W_pp, W_is, W_pi, both thresholds and the retinal parameters.
- **Loss and window:** window unset (~80 ms), stim×choice, S prior on, `m_pre_weight` 3 as fitted, no extra terms. Rank later at `m_pre_weight` 1.
- **Tags:** `stageB_hold_newS34_integ_<opt>_ft<src>_gm0`. A dry run with a fake `sbatch` gave 16 jobs with the right targets, freeze mask and resume JSONs.

**Fine-tune scored (2026-10-29).** All 16 runs have `FIT_DONE`, taking 2–13 min each.
- **Method.** Script: `scripts/_tmp_integ_ft_eval_plots.py`. Same shared stim as above (bps 20, seed 12345, regular s101), `m_pre_weight` 1, each arm on its own targets.
- **Outputs.** Eval: `alyx/models/stageB_hold_newS34_integ_<opt>_ft_gm0_eval.json`. Overlays and act-prior behaviour go in each `weights_run_fj_stageB_hold_newS34_integ_<opt>_ft<src>_gm0_full_mask0-1-3-4-7-9-10-11-14-15-16-17-18-19-20_s<seed>/`, with prior_effects drawn on the 80 ms window.
- **Constraints held.** g_m and d_m stayed at about 1e-13 in every run. The retinal parameters are frozen, but `L_S` still moves because g_s/d_s (P→S) are free.

Fair, best restart per source vs the source (all restarts in the eval JSON):

| source | source fair | restarts (89 / 7 / 12 / 45) | best |
|---|---:|---|---|
| opt1 s45 | 1.314 | 1.321 / 1.326 / 1.398 / **1.295** | s45, −0.02 |
| opt1 s7 | 1.368 | 1.372 / **1.363** / 1.395 / 1.368 | s7, −0.005 |
| opt2 s12 | 1.476 | 1.443 / **1.282** / 1.525 / 1.472 | s7, −0.19 |
| opt2 s303 | 1.732 | 1.582 / **1.412** / 1.610 / 1.636 | s7, −0.32 |

Per term for the best restart, plus opt2 s303 ft s45, which keeps the best behaviour:

| | opt1 ft45 s45 | opt1 ft7 s7 | opt2 ft12 s7 | opt2 ft303 s7 | opt2 ft303 s45 |
|---|---:|---:|---:|---:|---:|
| fair | 1.295 | 1.363 | 1.282 | 1.412 | 1.636 |
| `L_S` | 0.523 | 0.528 | 0.556 | 0.593 | 0.653 |
| traj C / M / P | 0.128 / 0.251 / 0.004 | 0.100 / 0.221 / 0.021 | 0.188 / 0.141 / 0.014 | 0.126 / 0.183 / 0.008 | 0.112 / 0.218 / 0.049 |
| prior S | 0.086 | 0.128 | 0.088 | 0.203 | 0.403 |
| prior C stim / move | 0.024 / 0.165 | 0.014 / 0.144 | 0.041 / 0.062 | 0.027 / 0.043 | 0.021 / 0.040 |
| prior M stim / move | 0.093 / 0.019 | 0.118 / 0.089 | 0.090 / 0.102 | 0.143 / 0.087 | 0.086 / 0.054 |
| g_i / d_i | 77 / 9.4 | **0.10 (floor)** / 14.4 | 200 / 3.4 | 200 / 19.8 | 198 / 22.1 |
| g_s / d_s | ≈0 / 48.6 | ≈0 / 43.5 | 3.2 / 11.3 | 0.098 / 17.1 | ≈0 / 3.9 |
| psychometric R² (source) | 0.87 (0.85) | 0.93 (0.93) | 0.43 (0.48) | 0.85 (0.97) | 0.94 (0.97) |
| RT R² (source) | 0.41 (0.27) | 0.52 (0.55) | 0.08 (0.30) | 0.23 (0.78) | 0.77 (0.78) |
| RT split R² con / inc | 0.59 / −0.49 | 0.63 / 0.11 | 0.84 / −9.76 | 0.70 / −2.78 | 0.86 / 0.25 |

- **opt1 barely moves.** The s7 restarts push g_i onto the 0.1 floor: that basin wants no C prior, and the floor only clips it. The move-aligned C prior is unchanged in opt1 (0.14–0.17).
- **opt2 gains come from the S prior via P→S coupling.** ft12 s7 sets g_s 3.2 / d_s 11 and brings S from 0.21 to 0.09. ft303 s7 sets g_s 0.1 / d_s 17 and brings S from 0.46 to 0.20. Neither found the d_s ≈ 45 basin that opt1 uses.
- **The fair gain in opt2 costs behaviour.** ft303 s7 drops RT R² from 0.78 to 0.23 and incongruent RT from 0.51 to −2.78. ft12 keeps s12's overweighted prior (psychometric R² about 0.4). The two ft303 restarts that keep g_s ≈ 0 (s12, s45) hold behaviour at the source level (psychometric 0.94, RT split 0.50–0.57), but they gain less fair (1.61–1.64).
- **Shape.** In ft303 s7 (prior_effects), the move-aligned M overshoots from −60 ms on and C falls in the last 20 ms. The low move-aligned C SSE comes with that shape.

**All 16 restarts.**
- **Overall** is fair (traj + prior + `L_S`) at `m_pre_weight` 1. "Old" is the source fit on the same protocol.
- **Traj columns:** S traj is `L_S` (S RMS); C and M are the I/M traj terms. The P penalties (≤ 0.05) are in overall only.
- **Prior columns:** C and M are stim + move-aligned summed; S is the 0–80 ms sidecar.
- **Behaviour:** act-prior R², combined only.
- **Parameters:** ≈0 means below 1e-4.
- **Highlighting:** bold restart = best overall within its source. † = best balanced fit (rising C prior, behaviour at source level), with its behaviour R² in bold.

| arm | source → restart | overall old → new | S traj (`L_S`) | C traj | M traj | S prior | C prior | M prior | psychometric R² | RT R² | g_i / d_i | g_s / d_s | g_m / d_m |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| opt1 | s45 → s89 | 1.314 → 1.321 | 0.527 | 0.127 | 0.257 | 0.093 | 0.183 | 0.124 | 0.86 | 0.34 | 44.6 / 11 | ≈0 / 44.9 | ≈0 / ≈0 |
| opt1 | s45 → s7 | 1.314 → 1.326 | 0.503 | 0.102 | 0.223 | 0.164 | 0.170 | 0.109 | 0.80 | -0.32 | 200 / 5.89 | ≈0 / 50.9 | ≈0 / ≈0 |
| opt1 | s45 → s12 | 1.314 → 1.398 | 0.528 | 0.128 | 0.260 | 0.097 | 0.197 | 0.106 | 0.83 | 0.15 | 42.5 / 11 | ≈0 / 45.3 | ≈0 / ≈0 |
| opt1 | **s45 → s45** | 1.314 → **1.295** | 0.523 | 0.128 | 0.251 | 0.086 | 0.189 | 0.113 | 0.87 | 0.41 | 77.2 / 9.37 | ≈0 / 48.6 | ≈0 / ≈0 |
| opt1 | s7 → s89 | 1.368 → 1.372 | 0.528 | 0.101 | 0.225 | 0.158 | 0.157 | 0.189 | 0.94 | 0.60 | 0.218 / 14.5 | ≈0 / 43 | ≈0 / ≈0 |
| opt1 | **s7 → s7** | 1.368 → **1.363** | 0.528 | 0.100 | 0.221 | 0.128 | 0.158 | 0.206 | 0.93 | 0.52 | 0.102 / 14.4 | ≈0 / 43.5 | ≈0 / ≈0 |
| opt1 | s7 → s12 | 1.368 → 1.395 | 0.530 | 0.102 | 0.229 | 0.190 | 0.162 | 0.178 | 0.94 | 0.61 | 0.1 / 14.3 | ≈0 / 43.9 | ≈0 / ≈0 |
| opt1 | s7 → s45 | 1.368 → 1.368 | 0.526 | 0.101 | 0.223 | 0.146 | 0.150 | 0.211 | 0.93 | 0.54 | 0.1 / 14.3 | ≈0 / 43.8 | ≈0 / ≈0 |
| opt2 | s12 → s89 | 1.476 → 1.443 | 0.607 | 0.177 | 0.134 | 0.161 | 0.085 | 0.241 | 0.39 | 0.34 | 182 / 7.05 | 0.355 / 6.17 | ≈0 / ≈0 |
| opt2 | **s12 → s7** | 1.476 → **1.282** | 0.556 | 0.188 | 0.141 | 0.088 | 0.102 | 0.192 | 0.43 | 0.08 | 200 / 3.37 | 3.23 / 11.3 | ≈0 / ≈0 |
| opt2 | s12 → s12 | 1.476 → 1.525 | 0.597 | 0.150 | 0.151 | 0.154 | 0.121 | 0.324 | 0.35 | 0.60 | 200 / 6.22 | 0.9 / 7.36 | ≈0 / ≈0 |
| opt2 | s12 → s45 | 1.476 → 1.472 | 0.607 | 0.172 | 0.133 | 0.158 | 0.085 | 0.256 | 0.40 | 0.38 | 180 / 7.15 | 0.291 / 6.19 | ≈0 / ≈0 |
| opt2 | s303 → s89 | 1.732 → 1.582 | 0.634 | 0.146 | 0.201 | 0.345 | 0.070 | 0.111 | 0.88 | -0.08 | 200 / 20.7 | 0.000376 / 7.22 | ≈0 / ≈0 |
| opt2 | **s303 → s7** | 1.732 → **1.412** | 0.593 | 0.126 | 0.183 | 0.203 | 0.070 | 0.230 | 0.85 | 0.23 | 200 / 19.8 | 0.0979 / 17.1 | ≈0 / ≈0 |
| opt2 | s303 → s12 | 1.732 → 1.610 | 0.652 | 0.110 | 0.216 | 0.380 | 0.065 | 0.138 | 0.94 | 0.75 | 194 / 22.3 | ≈0 / 3.97 | ≈0 / ≈0 |
| opt2 | s303 → s45 † | 1.732 → 1.636 | 0.653 | 0.112 | 0.218 | 0.403 | 0.061 | 0.140 | **0.94** | **0.77** | 198 / 22.1 | ≈0 / 3.92 | ≈0 / ≈0 |

**Regular arm (2026-10-30e).** `stageB_hold_newSbin56_mpre3_meancell_integ_<opt>` (`STAGEB_ARMS=regular`): mask `12|13` (g_s = d_s ≈ 0), no S prior in the loss, otherwise as the newSbin56 `full` runs (WEIGHTS_REL ∪ newSbin s56, mpre3, window unset, stim×choice, 8 seeds). Own-stim 4-draw overall: best **opt1 s303 1.542** (median 1.620; best full 1.165), best **opt2 s303 1.625** (median 1.689; best opt2 fit 1.190). S prior fixed at about 0.5; without it, regular ties full. Details: [integ opt2 prior/sensory](integ_opt2_prior_sensory_fit.md) 2026-10-30e.

**Original-rule target set, set up (2026-10-30f).** Aim: fit the standard split data with the 10-region S, but I/M defined the original way. Until now every 10-region fit also used the new I/M lists (opt1 84 / opt2 57, M 31).
- **Rule.** `get_sc_table` with the published parameters (alpha 0.01, sc 0, fixed late-slope cut 0.05, amp_loc 67, n 20) on the res-standard staging, without `apply_move_rule`. I = choice significant & rising; M = end peak & late slope > 0.05 & choice significant; union of the two rows. This gives I 85, M 30, with no I/M overlap.
- **Difference from opt1.** Only 7 regions change sides. MOs, PL, CA1 and MD are I here (M in opt1); MARN, PSV and SSp-ul are M here (I in opt1). S (10 regions), S prior (VISal VISam LP PO ZI) and P are identical to opt1 (checked).
- **Figure 4 i check.** `scripts/paper_figs/fig4/panel_i_old_rules.py` → `<oa>/manifold/paper_figs/fig4/res_standard/panel_i_old_rules.{png,svg,json}` (+ `_sc_table.csv`). Rows: original rules, opt1 and opt2.

  | rules | C stim / move mean | M stim / move mean |
  |---|---|---|
  | original (85 / 30) | 0.053 / 0.057 | 0.092 / 0.082 |
  | opt1 (84 / 31) | 0.056 / 0.059 | 0.080 / 0.075 |
  | opt2 (57 / 31) | 0.062 / 0.064 | 0.080 / 0.075 |

  The original rules separate M from C most: the 4 regions that move to M under the shuffle rule (MOs, PL, CA1, MD) dilute M.
- **Targets.** `python scripts/build_fit_targets_integ.py --opts oldrule` → `fit_targets_integ/oldrule/` (prior curves and `regions.json`). The prior targets equal the panel i means above. The traj targets (`mean_data_results.npy`, `avg_mean_R.npy`) come from the ORCD targets job (`STEP=targets OPTS=oldrule`). Plots and checks go to `manifold/fit_targets_integ/oldrule/`, so the 10-28 check is not overwritten.
- **Stage A reused.** newSbin s56 serves as Stage A: its S target is the same 10-region `avg_mean_R`. The fits step now refuses to submit if a set's `avg_mean_R` differs from `opt1_rawslope`'s (`S_REF`). If the rebuilt file differs (for example, because the insertion cache changed), stop and compare before copying.
- **Stage B.** Both arms, same protocol as newSbin56: WEIGHTS_REL ∪ newSbin s56, mpre3, window unset (~80 ms), stim×choice, 8 seeds.
  - `full`: no freeze, S prior on. Tag `stageB_hold_newSbin56_full_mpre3_meancell_integ_oldrule`.
  - `regular`: mask 12|13, no S prior. Tag `stageB_hold_newSbin56_mpre3_meancell_integ_oldrule`.
  - A dry run with a stubbed `sbatch` gave 16 jobs with the right tags and `FIT_TARGETS_DIR`.
- **Partition (same day).** `submit_fit_targets_integ.sh` now defaults to `mit_preemptable` and sources `sbatch_defaults.sh` at the top, so every job gets `--requeue`: targets, Stage A, selector and Stage B. `STEP=all` keeps `--requeue` next to its dependency flags. The selector is now requeue-safe. Before submitting Stage B it atomically creates `RANK_DIR/stageB_submitted_<opts>`, and a requeued run that finds the marker exits without submitting. `PARTITION=pi_fiete` restores the old behaviour.

**Original-rule target set, results (2026-11-01).** All 16 fits (`full` and `regular`, 8 seeds each) `FIT_DONE`. Eval: the 10-30b protocol (own α_w/β_w stimuli, 4 draws 12345/1/2/3, m_pre_weight 1, overall = traj + prior + `L_S`), via `ARM=oldrule OWN_STIM=1 BEHAV=1 BEHAV_CACHE=0 GROUPS="newSbin56 newSbin56_regular" python scripts/_tmp_integ_newSbin56_eval_plots.py`. Both `_tmp_integ_*_eval_plots.py` modules now fall back to no fine-tune picks for an arm other than opt1/opt2. Outputs: `alyx/models/stageB_hold_newSbin56_integ_oldrule_eval_ownstim.json`, `…_plot_summary_ownstim.json`, behaviour in `…_oldrule_{full,regular}_plot_summary.json`; S / I/M / P / prior overlays and act-prior psychometric + RT in every run dir.
- **Comparability.** S prior, `L_S` and the P traj share targets with opt1/opt2. C/M traj and C/M prior are scored against oldrule's own curves, so overall is not a like-for-like comparison of the I/M fit.
- **Per seed (oldrule):**

  | arm | seed | overall | traj (C / M / P) | S prior | C prior | M prior | `L_S` | S R² | psych / RT R² | g_i / d_i | g_s / d_s | g_m |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | full | **s7** | **1.133±0.121** | 0.454 (0.135 / 0.226 / 0.093) | 0.145 | 0.109 | 0.077 | 0.348 | 0.72 | 0.91 / 0.15 | 99 / 3.8 | ≈0 / 35.6 | 0.27 |
  | full | s101 | 1.269±0.115 | 0.476 (0.188 / 0.208 / 0.081) | 0.208 | 0.059 | 0.133 | 0.393 | 0.70 | 0.35 / 0.46 | 195 / 8.0 | ≈0 / 16.0 | 6.7 |
  | full | s45 | 1.380±0.187 | 0.592 (0.161 / 0.291 / 0.141) | 0.194 | 0.131 | 0.074 | 0.390 | 0.70 | 0.99 / 0.33 | 65 / 20.3 | 135 / 90.7 | 0.89 |
  | full | s34 | 1.491±0.210 | 0.422 | 0.346 | 0.138 | 0.176 | 0.408 | 0.68 | 0.96 / 0.13 | 61 / 20.8 | ≈0 / 10.5 | ≈0 |
  | full | s303 | 1.524±0.068 | 0.391 | 0.515 | 0.068 | 0.090 | 0.460 | 0.61 | 0.64 / 0.30 | 200 / 15.5 | 0.19 / ≈0 | ≈0 |
  | full | s12 | 1.574±0.179 | 0.507 | 0.513 | 0.050 | 0.081 | 0.424 | 0.67 | 0.91 / 0.09 | 197 / 21.0 | ≈0 / ≈0 | ≈0 |
  | full | s89 | 1.659±0.182 | 0.511 | 0.536 | 0.133 | 0.059 | 0.420 | 0.67 | 0.97 / −0.29 | 11 / 20.6 | ≈0 / ≈0 | ≈0 |
  | full | s333 | 1.693±0.117 | 0.361 | 0.525 | 0.115 | 0.175 | 0.517 | 0.60 | 0.67 / 0.18 | 200 / 9.4 | ≈0 / 0.12 | ≈0 |
  | regular | **s303** | **1.494±0.167** | 0.425 (0.118 / 0.187 / 0.120) | 0.519 | 0.082 | 0.070 | 0.398 | 0.68 | 0.95 / −0.09 | 190 / 21.5 | 0 / 0 | ≈0 |
  | regular | s12 | 1.536±0.157 | 0.459 | 0.526 | 0.066 | 0.062 | 0.422 | 0.67 | 0.55 / −1.07 | 184 / 14.3 | 0 / 0 | ≈0 |
  | regular | s34 | 1.559±0.141 | 0.456 | 0.499 | 0.086 | 0.094 | 0.423 | 0.67 | 0.81 / −0.12 | 177 / 18.2 | 0 / 0 | ≈0 |
  | regular | s101 | 1.561±0.191 | 0.450 | 0.513 | 0.115 | 0.060 | 0.423 | 0.67 | 0.95 / −0.53 | 200 / 24.1 | 0 / 0 | 57 |
  | regular | s333 | 1.581±0.197 | 0.490 | 0.510 | 0.118 | 0.040 | 0.422 | 0.67 | 0.98 / −0.34 | 103 / 20.0 | 0 / 0 | 0.08 |
  | regular | s45 | 1.618±0.176 | 0.487 | 0.544 | 0.049 | 0.115 | 0.422 | 0.67 | 0.88 / 0.38 | 191 / 23.3 | 0 / 0 | ≈0 |
  | regular | s7 | 1.623±0.178 | 0.563 | 0.528 | 0.066 | 0.044 | 0.422 | 0.67 | 0.98 / −0.03 | 186 / 20.4 | 0 / 0 | ≈0 |
  | regular | s89 | 1.648±0.175 | 0.483 | 0.554 | 0.099 | 0.090 | 0.422 | 0.67 | 0.99 / 0.26 | 75 / 16.8 | 0 / 0 | 0.58 |

  C and M prior = stim + move-aligned summed.
- **Medians over 8 seeds** (opt1 / opt2 from 10-30e):

  | target set | arm | overall | overall − S prior | traj | S prior | `L_S` | psych R² | RT R² |
  |---|---|---|---|---|---|---|---|---|
  | oldrule | full | **1.508** | 1.092 | 0.465 | 0.430 | 0.414 | 0.91 | **0.17** |
  | opt1 | full | 1.654 | 1.159 | 0.484 | 0.457 | 0.423 | 0.88 | −0.19 |
  | opt2 | full | 1.653 | 1.143 | 0.428 | 0.465 | 0.424 | 0.85 | −0.44 |
  | oldrule | regular | 1.571 | 1.065 | 0.471 | 0.523 | 0.422 | 0.95 | −0.11 |
  | opt1 | regular | 1.620 | 1.105 | 0.442 | 0.513 | 0.422 | 0.87 | −0.69 |
  | opt2 | regular | 1.689 | 1.153 | 0.498 | 0.528 | 0.422 | 0.90 | −0.64 |

**Findings.**
- **Best overall is oldrule full s7 (1.133 ± 0.121)**, against opt1 s45 1.165 ± 0.073 and the opt2 gm0 ft s12→s7 1.190. The gap is inside one draw-sd, so this is a tie, not a win. s7 gets there differently from opt1 s45: the lowest S prior (0.145) and `L_S` (0.348, S R² 0.72) of any fit, through the d_s offset basin (d_s 35.6), but a worse trajectory (0.454 vs 0.366; C 0.135 vs 0.093).
- **Medians favour oldrule in both arms** (full 1.51 vs 1.65; regular 1.57 vs 1.62 / 1.69). In the full arm most of it is the S prior and `L_S`, which have the same targets as opt1, so it reflects more seeds finding the d_s basin (s7, s101, s45, s34: 4 of 8, against 3 of 8 for opt1) rather than an easier I/M target. In the regular arm the gain is in C/M prior (traj is similar).
- **The d_s basin again.** Every full fit with S prior < 0.35 has d_s 10–91. s45 is the first fit in any set to use g_s (135, with d_s 90.7); it pays in trajectory (M 0.291).
- **Same move-aligned miss.** In s7 the model C prior falls over the last ~40 ms before the move while the data rises, and the model M prior overshoots stim-aligned (≈0.10 against 0.06–0.09 data). The move-aligned C trajectory flattens about 20 ms early (R² 0.97, M 0.99).
- **Behaviour (held out, check only).** RT R² is better than for opt1/opt2: median 0.17 full (opt1 −0.19, opt2 −0.44) and −0.11 regular (−0.69 / −0.64); 6 of 8 full seeds positive. Psychometric is about the same (0.91 / 0.95), except full s101 (0.35), s303 (0.64), s333 (0.67).
- **Regular arm without the S prior (same day, user ask).** The overall in the tables above includes the S prior, which a regular fit cannot move (g_s = d_s = 0, not in its fit loss). From now on regular fits are scored as overall − S prior (traj + C/M prior + `L_S`, per draw; `prior` = the sum of its five parts, checked). Same-seed full fits are compared on the same quantity. Ranking hardly changes:

  | set | best regular (no S) | 2nd | median regular / full (no S) | best full (no S) |
  |---|---|---|---|---|
  | oldrule | s303 0.975 ± 0.137 | s12 1.009 | 1.065 / 1.092 | s7 0.988 |
  | opt1 | s303 1.043 ± 0.097 | s34 1.043 | 1.105 / 1.159 | s45 0.936 |
  | opt2 | s303 1.095 ± 0.144 | s101 1.117 (was s12) | 1.153 / 1.143 | s45 1.070 |

  Without the S prior, oldrule regular s303 matches the best oldrule full fit (0.975 vs 0.988). The report (`stageB_hold_newS34_integ_ft_gm0_report`, regular table and pages) now uses this score; opt2's second page moves from s12 to s101.
- **Regular arm** behaves as on opt1/opt2: retinal parameters stay at s56 (`L_S` 0.422, S R² 0.67) except s303 (`L_S` 0.398), S prior fixed at about 0.5–0.55, g_i 75–200.

**Open.**
- Score the oldrule fits on the opt1 targets (and the opt1 fits on oldrule's) to separate "better model" from "easier target".
- s7 is a candidate for a gm0-style fine-tune aimed at the C move-aligned fall.

**Prior-gain variants on oldrule, set up (2026-11-01).** Aim: which prior modulations the oldrule C/M + 10-region S data need. Five Stage B arms. Each freezes a set of g_/d_ at ≈0: frozen dims are filled with LOG_ZERO, not held at the warm start, since `--freeze-hold` is off. Everything else is as the oldrule `full`/`regular` runs:
- **Fixed protocol:** target set `fit_targets_integ/oldrule`, warm start WEIGHTS_REL ∪ newSbin s56, `--stage1-hold-retinal`, `de_cma_local`, bps 20/20, m_pre_weight 3, window unset (~80 ms), stim×choice, seeds 7 12 34 45 89 101 303 333, `mit_preemptable` + `--requeue`.
- **What the loss sees:** C/M/P trajectory, C/M prior and `L_S`. The S prior curve is in the loss unless both g_s and d_s are frozen (user rule, same day).

| arm | `VARIANTS` | frozen ≈ 0 | free prior params | S prior in loss | tag suffix |
|---|---|---|---|---|---|
| `gs0gm0` | `full:7\|9\|12` | g_s, g_m, d_m | g_i, d_i, d_s | yes | `_oldrule_gs0gm0` |
| `gm0` | `full:7\|9` | g_m, d_m | g_i, d_i, g_s, d_s | yes | `_oldrule_gm0` |
| `gm0gs0` | `regular:7\|9\|12\|13` | g_m, d_m, g_s, d_s | g_i, d_i | **no** | `_oldrule_gm0gs0` |
| `donly` | `full:6\|7\|12` | g_i, g_m, g_s | d_i, d_m, d_s | yes | `_oldrule_donly` |
| `gonly` | `full:8\|9\|13` | d_i, d_m, d_s | g_i, g_m, g_s | yes | `_oldrule_gonly` |

Indices: g_i 6, g_m 7, d_i 8, d_m 9, g_s 12, d_s 13. Full tag: `stageB_hold_newSbin56_mpre3_meancell_integ_oldrule_<arm>`; dirs `weights_run_fj_<tag>_<mtype>_mask<slug>_s<seed>`. Wired as `STAGEB_ARMS=<arm>` in `scripts/submit_fit_targets_integ.sh`. A dry run with a stubbed `sbatch` gave 40 jobs: 5 arms × 8 seeds, the masks and S-prior flags above, and `FIT_TARGETS_DIR=fit_targets_integ/oldrule`. When scoring, leave the S prior out of overall for `gm0gs0`, as for regular.

**Options 3 and 4: movement cut at the mean slope, set up (2026-11-02).** Aim: full Stage B fits on the C/M sets from the Fig 4 e/g/i mean-cut comparison ([paper main figures](paper_main_figures.md) 11-02). Both use oldrule's labelling (fixed rule, `get_sc_table` style, `panel_i_slope_mean_cut.label_sets`) and change only the movement cut. Each mean is taken over the 118 choice-significant regions.
- `opt3_latemean`: late slope > 0.0437 (the mean; oldrule uses 0.05). I 84, M 31. The only change from oldrule is that PL moves from I to M.
- `opt4_overallmean`: end peak & overall slope (`amp_slope`, 0–150 ms) > 12.54/s (the mean). I 91, M 24. CENT2, COAp, CP, GPe, GPi, PoT, SPVI and VPM move from M to I; CLI and PAR move from I to M.
- **Targets:** `build_fit_targets_integ.py --opts opt3_latemean opt4_overallmean` → `fit_targets_integ/<opt>/`. `regions.json` records `move_cut`. Prior target means (stim / choice):
  - opt3: I 0.0526 / 0.0573, M 0.0909 / 0.0813.
  - opt4: I 0.0516 / 0.0551, M 0.1150 / 0.1019.
  - oldrule, for comparison: I 0.0527 / 0.0570, M 0.0916 / 0.0824.
  - S, S prior, P and the S prior curve are identical to oldrule, so the newSbin s56 Stage A serves both options (the fits step checks `avg_mean_R`).
- **Traj targets** (`mean_data_results`, `avg_mean_R`) come from one ORCD job: `STEP=targets OPTS="opt3_latemean opt4_overallmean"`.
- **Fits:** the `full` arm only: no freeze, S prior on, m_pre_weight 3, window unset (~80 ms), stim×choice, 8 seeds, the oldrule protocol otherwise. Tag `stageB_hold_newSbin56_full_mpre3_meancell_integ_<opt>`.
- **No Alyx in this chain:**
  - `submit_fit_targets_integ.sh` exports `PRIOR_MECH_NO_ONE=1` and `ONE_CACHE_DIR`, so neither the fit jobs nor the login-node hybrid build create a ONE object.
  - The targets job opens ONE in local mode with Alyx blocked (`one_offline.py`) on `ONE_CACHE_DIR`. An insertion with missing local data now stops the job instead of being skipped.
  - `STEP=targets/all` first runs `check_offline_inputs.py`, the same preflight as the alt-null submit; `PREFLIGHT=0` skips it.
- **Dry run:**
  - A stubbed `sbatch` gave 1 targets job plus 16 fits (2 options × 8 seeds, `VARIANTS=full:`, S prior on, `PRIOR_MECH_NO_ONE=1`, `BLOCK_ALYX=1`).
  - The traj builder ran 2–3 insertions with `SLURM_JOB_ID` set: ONE was in local mode and no Alyx request was made. It then failed closed on purpose, because M is too sparse with that few insertions.
  - On the laptop the preflight correctly flagged sessions with no local trials.

**Report (2026-10-29).** A LaTeX summary of the four bold restarts (opt1 s45→s45, opt1 s7→s7, opt2 s12→s7, opt2 s303→s7) is at `alyx/models/stageB_hold_newS34_integ_ft_gm0_report/`. The folder holds `report.tex`, `report.pdf` (5 landscape pages), copies of the figures in `figs/<fit>/`, and `make_ft_report.py`, which rebuilds the `.tex` from the eval and plot-summary JSONs.
- **Page 1:** the setup and this table's columns for the four fits.
- **One page per fit:**
  - S trajectory by contrast (`S_fit`).
  - C/M trajectories, stim-aligned (`IM_post`) and move-aligned (`IM_pre`).
  - `prior_effects` on the 80 ms windows.
  - Act-prior psychometric split congruent / incongruent, and RT over all trials.
  - Losses, R² and g/d values in the captions.

---

## 2026-09-17 — split window scored (150 post-stim / 80 pre-move)

Topic: [revisions](modeling_details_revisions.md) 09-17. Fit at
JSON windows 150/80; **rank fair at shared 80 and 150** (split
keys cleared so they cannot win). `m_pre_weight=1` at eval. Plots
in each run dir. Driver
`[scripts/_tmp_splitwin_meancell_eval.py](../scripts/_tmp_splitwin_meancell_eval.py)`.


| Tag                                                    | Restrictions                      | Best seed                                                                                                       | Notes                                                                                                  |
| ------------------------------------------------------ | --------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `stageB_hold_s89_stim150_choice80_meancell`            | regular `12                       | 13`; 150/80; no S                                                                                               | fair @80 **s7 1.050** / @150 s45 **1.222**; as-fitted s7 **0.971**                                     |
| `stageB_hold_s89_mpre3_stim150_choice80_meancell`      | regular; mpre3 (eval @ 1); 150/80 | fair @80 **s101 1.060** / @150 s12 **1.141**; as-fitted s12 **1.028**                                           | Loses to 80-fit mpre3 s101 **0.927 / 1.069**. As-fitted RT s12 pooled −0.69 / inc −9.79                |
| `stageB_hold_s89_full_stim150_choice80_meancell`       | full none; 150/80; unsplit-80 S   | fair @80 **s89 1.466** / @150 s12 **1.390** (`g_i` collapsed); as-fitted s7 **1.292** (S 0.069, `d_s≈49`)       | S-success 0/8. Median tot80 1.702. Collapsed s12 has the good inc RT (+0.325)                          |
| `stageB_hold_s89_full_mpre3_stim150_choice80_meancell` | full; mpre3; 150/80; unsplit-80 S | fair @80 **s303 1.138** (S 0.028, `d_s≈22`) / @150 s333 **1.199** (S 0.036, `d_s≈49`); as-fitted **s333 1.116** | S-success 2/8. Loses to 80-fit `full` mpre3 s89 **1.054 / 1.080**. s333 inc RT −0.68 with `g_i` intact |


**Keep production unset +** `m_pre_weight=1`**.** Split 150/80 does
not beat 80-fit on either fair tot.

---



## 2026-10-05 — s89ft imshape (scored)

Warm CMA from `full_mpre3_meancell` **s89**. Two freeze sets.
Rank at extras=0, `m_pre_weight=1`, 80 ms, shared stim 12345.
Topic: [s89ft imshape](s89ft_imshape.md).

| Tag                                               | Restrictions                                                                                         | Best seed | Notes |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | --------- | ----- |
| `stageB_hold_s89_full_s89ft_imshape_meancell`     | full; freeze-hold except `2,5,7,8,9`; 80 ms stim×choice; extras on choice I/M + M overshoot ≥40 ms | **s7 1.018** | option 1 (`nowii`); `g_m` collapsed; M ramp not fixed |
| `stageB_hold_s89_full_s89ft_imshape_wii_meancell` | same + free `W_ii` (`0,2,5,7,8,9`)                                                                   | **s7 1.014** | option 2 (`wii`); tot beats source s89 **1.054**; stim-M hinge 0.133→0.070; still no notch; `d_i≈0` |
| `stageB_hold_s89_full_s89ft_di_choicei_meancell`  | full; freeze-hold except `d_i`; from `wii` s7; init `d_i=5`; late choice-I undershoot ≥−40 ms     | **s89 1.033** | loses to `wii` s7 **1.014**; `d_i≈1.1`; commit I still ~0.043 |
| `stageB_hold_s89_full_s89ft_wiigi_choicei_meancell` | same extras; wii free set + `g_i` (`0,2,5,6,7,8,9`)                                            | s7 1.095 | `g_i` did not back off; tot worse |


---



## 2026-09-18 — ranked at stim150 / choice80

All 12 tags (6 regular + 6 `full`) re-scored at the split
window. Regular and `full` ranked separately (`full` tot
includes S nSSE; `tot_noS` drops it). Tables are in the
ranking cheat-sheet above. Topic:
[revisions](modeling_details_revisions.md) 09-18.

Winners still 80-fit mpre3: regular **s101 0.905**; `full`
**s89 0.969** (tot_noS 0.947). Split as-fitted does not take
either family. **Keep production unset +** `m_pre_weight=1`**.**

---



## 2026-09-15 — catalog opened

Collected tags, freeze / window / stratum, and best seeds from
[retinal then joint](retinal_then_joint_fitting.md),
[modeling_details_revisions.md](modeling_details_revisions.md),
[modeling_details_prior_rt_gaps.md](modeling_details_prior_rt_gaps.md),
and [fit_gs_ds_with_im.md](fit_gs_ds_with_im.md).