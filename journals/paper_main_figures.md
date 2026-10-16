# Paper main figures: rebuild on current results

**Goal.** Rebuild the main figures of the paper on the current analyses: act
prior, and label shuffles within strata. Each figure keeps its published
layout. Panels change only where the newer results differ from the published
ones.

**Where things live**

- Figures go in openalyx `manifold/paper_figs/<fig>/`. Each figure folder holds
  the full figure, `panels/`, and the derived tables the plots read.
  `<oa> = ~/Downloads/ONE/openalyx.internationalbrainlab.org`.
- Scripts go in `scripts/paper_figs/<fig>/`. Figure 4.5 is the exception: its
  composite is still drawn by `scripts/fig45_sensitivity.py fig45`, because that
  script also holds the ORCD compute pipeline.

| figure | folder | scripts | status |
|---|---|---|---|
| 2 (stim / choice sensitivity, function assignments) | `<oa>/manifold/paper_figs/fig2/` | `scripts/paper_figs/fig2/` | rebuilt, two choice versions (10-14) |
| 4 (prior on region types) | `<oa>/manifold/paper_figs/fig4/` | `scripts/paper_figs/fig4/` | redrawn 10-15; a = overall earlystim table, b = sensory ∩ prior_80ms onetype, c = LGd + combined 80 ms, d–i replotted |
| 4.5 (block effects on cell types) | `<oa>/manifold/paper_figs/fig45/` | `scripts/fig45_sensitivity.py fig45` | rough draft; see [Figure 4.5](fig45_block_effects_cell_types.md) |
| 5 (model vs data) | `<oa>/manifold/paper_figs/fig5/` | `scripts/paper_figs/fig5/` | b–i redrawn 10-15 from full mpre3 s89; a and j empty |

---

## 2026-10-14 — Figure 2

### Original pipeline

All panels were plotted from openalyx `manifold/res` (the `one.cache_dir` of
`analysis_functions`).

| panel | contrast / strata | function |
|---|---|---|
| a | `stim_duringstim1_act` = `stim_block_{l,r}_act`: stim L vs R, 0–80 ms, act-block strata only | `plot_regional_distance(plot_gain=False, plot_offset=False)` (`bwm_figs.ipynb` cell 30) |
| b | `stim_duringstim_act` / `stim_duringchoice_act`: stim L vs R, choice × act-block strata | same |
| c | `choice_duringstim_act` / `choice_duringchoice_act`: choice L vs R, stim × act-block strata | same (cell 32) |
| d | schematic | — |
| e | Σ^{stim,s} = 1 − `sc_duringstim` histogram; Σ^{stim,m} = 1 − `sc_duringchoice` vs monotonicity | `analysis_figs.ipynb` cell 6 on `get_sc_table` (cell 5) |
| f | `sc_duringstim_regtype` (s) and `sc_duringchoice_regtype` (m) per Beryl region | `plot_sc_table(metric='regtype')` → openalyx `meta/table_stimchoice_act_regtype_p_mean_c_0.01.png` |

`get_sc_table` parameters (notebook cells 3 and 5): `p_mean_c`, α = 0.01,
`slope_threshold` 0.05, `n` 20, `amp_loc_threshold` 67, `sc_threshold` 0,
`stim_restr`. Σ^{stim,s} also includes the 80 ms short and `stim1` amplitudes in
its denominator. Monotonicity is defined on the move-aligned choice curve: the
last 5-bin window is the global maximum and the last-20 slope is above 0.05.

### Two choice versions

Stim L vs R act splits exist only on openalyx. Choice L vs R has a newer re-run
of the same stratified shuffle, the alyx `res/new` min5 (2026-08-14; see
[structured nulls](structured_nulls_choice_lr.md) 07-27c / 08-14). The figure
is built for both:

- `openalyx` uses the published-figure files for both stim and choice.
- `min5` uses stim from openalyx and choice from alyx `res/new`.

`fig2_data.py` stages the inputs and runs `get_sc_table` on the staged copies,
because it rewrites the combined dicts in its results dir. Small combined
dicts are copied; the regde arrays are symlinked and only read. Nothing in
`manifold/res` is modified.

**Check.** On `openalyx`, the during-stim region types are stim 1 (VISpm),
integrator 28 and move 22. This matches the 07 entry in
[variance partition](variance_partition_mixed_regions.md) from
`export_stimchoice_regtypes.py`.

### a–c examples (FDR-corrected `p_mean_c`)

| panel | region | published | `openalyx` now | `min5` now |
|---|---|---|---|---|
| a sig | VISp | 0.0038 | 0.0038 | 0.0038 |
| a n.s. | MOs | 0.3414 | 0.3414 | 0.3414 |
| b sig (stim / move) | VeCB | 0.0080 / 0.0069 | 0.0080 / 0.0069 | same (stim is openalyx) |
| b n.s. (stim / move) | MOs | 0.5284 / 0.6330 | **0.0567 / 0.0099** | same |
| c sig (stim / move) | GRN | 0.0031 / 0.0018 | 0.0031 / 0.0018 | 0.0016 / 0.0011 |
| c n.s. (stim / move) | RSPd | 0.3283 / 0.3210 | 0.3283 / 0.3210 | 0.0638 / 0.0670 |

- **b non-significant MOs no longer holds.** The openalyx stim combined files
  were rewritten on 2026-09-07 (the split files are from 2025-10). MOs
  move-aligned stim now passes FDR 0.01. Why it changed since the published
  figure is not traced.
- **Replacement: ORBl** (0.72 / 0.62). Frontal cortex like MOs, flat against
  shuffles. `fig2_data.py` writes other candidates (p > 0.3 in both windows) to
  `examples_check.json`: PFL, DG, PIR, ENTl, MG, …, ACAv, ORBm, AIv, ORBl.
  Provisional.
- RSPd stays non-significant under min5, but only at about 0.065.

### Region-type counts

| | `openalyx` | `min5` |
|---|---|---|
| s row: stim / stim early / integ / move | 1 / 22 / 28 / 22 | 1 / 10 / 65 / 25 |
| m row: integ / move | 58 / 23 | 94 / 25 |
| e left: n regions, Σ^{stim,s} > 0.8 | 69, 23 | 99, 11 |
| e right: n regions | 81 | 119 |

min5 makes choice more liberal: FDR 0.01 counts go from 46 to 88 stim-aligned
and from 84 to 122 move-aligned. Choice significance then shows up in many
early-stim regions, which lowers their Σ^{stim,s} below 0.8. The sensory class
roughly halves (23 → 11), and the integrator class grows on both rows.

### Outputs

All under `<oa>/manifold/paper_figs/fig2/`:

- `original/`: `figure2.pdf` and `panel_d_original.png`, a 600 dpi raster crop
  of d that is embedded in the composite. Replace it by hand for vector d.
- `<version>/fig2_<version>.{svg,png}`: composite on the original 858.7 × 554.8
  pt page, with axes boxes measured from the PDF.
- `<version>/panels/`: `panel_{a,b,c}_{sig,nonsig}_<reg>`, `panel_e`, `panel_f`.
- `<version>/`: `sc_table.csv` (region types, Σ, monotonicity), `pvals.csv`
  (per-region `p_mean_c` and `amp_euc` for all six timeframes),
  `examples_check.json`, `summary.json`, and `staging/`.

```bash
conda activate iblenv   # outside sandbox
cd scripts/paper_figs/fig2
python fig2_data.py                      # both versions, ~1 min
python fig2_plot.py --version openalyx   # --examples b=VeCB,ORBl to change regions
python fig2_plot.py --version min5
```

### Open

1. Pick the choice version for the paper (`openalyx` or `min5`).
2. Confirm the b non-significant region (ORBl is provisional).
3. Under min5, consider a clearer c non-significant region than RSPd (0.064).
4. There is no min5 re-run of the act stim splits, so stim stays openalyx in
   both versions.
5. Panel d is a raster crop of the original.

---

## 2026-10-14 — Figure 4

First attempt pasted diagnostic rasters onto the published PDF. Replaced 10-15.

---

## 2026-10-15 — Figure 4

Redrawn (same approach as Figure 2): matplotlib on the published 759.276 × 632.395 pt
page, axes boxes measured from `~/Downloads/figure4.pdf`. Diagnostic LGd / combined
PNGs are **not** pasted — they have black backgrounds and p-value annotations. The
gain/offset cartoons and P–S–C–M flow diagrams in c, e, g are left empty.

### Style (from the original PDF)

| element | original |
|---|---|
| page | 759.3 × 632.4 pt |
| panel letters | Myriad Pro 28 pt → Helvetica 26 |
| tick labels | Helvetica 11 |
| `time (ms)` | Helvetica 14 |
| y-labels | Times $d^{\mathrm{prior},s/m}_{\mathrm{reg}}(t)$, ~10–11 pt |
| event line | 3 pt, stim `#2BA9E0`, move `#E8806A` |
| shuffles | grey, 40 traces, lw 0.45; first 5 bins `#5f7ea3` when offset is drawn |
| observed | dashed = raw, solid = offset-subtracted (only if $p_{\mathrm{offset}}\le 0.01$) |
| S / C / M | `#57C1EB` / gold / tomato |
| one-row strip | `plot_combined_onetype`: 208 Beryl columns + a combined hit row |

### Data

The overall table (panel a PNG) was made by
`scripts/plot_goal3_c0_summary_table.py --combined-regtype-prior --earlystim-ms 80 --alpha 0.01`
(`plot_combined_table_summary`, `display='overall'`).

| column | source |
|---|---|
| `type_stim` / `type_choice` | Figure 2 openalyx `sc_table.csv` (`get_sc_table`, $p_{\mathrm{mean},c}$, α=0.01, slope 0.05, n=20) |
| `prior_stim` / `prior_choice` | openalyx `manifold/res` 4-split act-block, 150 ms |
| `prior_80ms` | alyx `res/new` stim-side unsplit, $t\le 80$ ms (`…_earlystim80.npy`) |

Nothing under `manifold/res` is rewritten. The Aug-24 duringchoice combine has
`p_mean` but no `p_mean_c`; FDR is filled **in memory** (42 hits @0.01).
`p_gain` on that file is nan; panel h's gain_m row is computed from the curves.

| panel | what | source |
|---|---|---|
| a | 6-row overall table | cropped alyx `meta/table_act_block_combined_summary_act_p_mean_c_combinedpTrue_0.01_overall_earlystim80.png`; right labels **s / 80 / stim / m / choice** |
| b strip | one-row onetype | `prior_80ms` FDR ≤ 0.01 ∩ `type_stim` ∈ {0, 0.1} |
| b example | LGd, 0–80 ms | alyx unsplit `regde`, paper-style (not the diagnostic PNG) |
| c | cell-weighted sensory combined, 0–80 ms | same 13 regions |
| d | PAG, PRNc stim-aligned | openalyx 4-split; both still `prior_stim` FDR ≤ 0.01 |
| e | C average (all integrator types) | union of `type_stim`/`type_choice` = 0.5 |
| f | IRN stim- and move-aligned | same; still type 1.0 and prior-significant |
| g | M average, move-aligned | union of move types |
| h | offset / gain s / gain m | offset and gain s are the larger of the 150 ms 4-split effect and the prior_80ms effect (FDR and the component both ≤ 0.01); gain m from duringchoice curves |
| i | shuffle-subtracted C vs M | `load_group`-style cell-weighted mean − shuffle mean |

Overlap, table order: VISam, TEa, CA1, LGd, LP, PO, ZI, SAG, PAG, PB, TRN, PRNr, DCO
(13). C onetype strip (integrator ∩ `prior_stim`): ILA, PF, PRNc, P5, VCO, PGRN, LAV,
MV, SPIV, SUV, DN. PAG is `type_stim` = 0.1 (sensory early), so it is in the S
overlap and is **not** in the C strip; it is kept as the published C example because
`prior_stim` still passes FDR.

d–g curves match the published shapes (PRNc ramp, IRN stim+move, C/M averages with
inset). They were replotted from current files rather than cropped from the PDF.

```bash
conda activate iblenv   # outside sandbox
cd scripts/paper_figs/fig4
python fig4_plot.py
```

Outputs under `<oa>/manifold/paper_figs/fig4/`: `figure4.{pdf,svg,png}`, `panels/`,
`summary.json`, `original/figure4.pdf`.

---

## 2026-10-15 — Figure 5

Redrawn on the published page (`~/Downloads/figure5.pdf`, MediaBox
**1105.75 × 666.286 pt**). Matplotlib axes sit on the spine boxes measured
from that PDF (top-left origin, y down; `fig5_common.fig_box`). a (circuit)
and j (P–S–C–M flow) are left empty. Panel letters are still drawn.

Model is the **full** family winner on the “Ranked at 80 ms tot” table in
[stage B fit variants](stage_b_fit_variants.md): `stageB_hold_s89_full_mpre3_meancell`
seed **89**, tot80 **1.054**. JSON
`weights_final_loss1p125_20260915-224209.json` in
`models/weights_run_fj_stageB_hold_s89_full_mpre3_meancell_full_masknone_s89`.
`prior_window_ms` is unset (legacy traj T = 72 steps = 144 ms at dt = 2;
prior `plot_window` = 80). The later s89ft warm-start is not this row.

### Style (from the original PDF)

Myriad Pro is not installed. Letters use Helvetica at the measured sizes
(a 22 pt, g 19 pt, the rest 21 pt). Everything else is Helvetica.

| element | original |
|---|---|
| page | 1105.75 × 666.286 pt |
| panel letters | Myriad Pro 21 pt (a 22, g 19) → Helvetica, same sizes, `#231F20` |
| tick labels | Helvetica 12 (`#030505`); b/e ticks in the PDF are 11.6–11.8 |
| `time (ms)` | Helvetica 14 |
| `stimulus contrast`, `% correct`, `reaction time (s)` | Helvetica 13.8 |
| R² on b–g | Helvetica 10 |
| R² on i | Helvetica 8.4 |
| ΔAIC | Helvetica 13, `#231F20`, above the d pair |
| spines | 0.8 pt black; top and right off |
| stim event (t = 0) | 2 pt `#27AAE1` |
| move event (t = 0) | 2 pt `#DC7F6B` |
| badges | diameter 19 pt, stroke 0.5 `#231F20`, letter 11.5 pt `#231F20` |
| badge fills | S `#A2DDF8`, C `#FEE499`, M `#DC7F6B` |

R² colours, measured from the PDF (not the diagnostic-plot defaults):

| panel | colour |
|---|---|
| b, e | `#4B87C6` |
| c, I | `#DAA628` |
| c, M | `#CC5827` |
| d | `#030505`, written `R²: …` |
| f | `#FCD704` |
| g | `#F2644B` |
| i con / RT | `#030505` |
| i incon | `#808181` |

I/M trajectory colours are the loss-function pair: I `#DAA520` / `#FFD700`,
M `#CC5500` / `#FF7F0E`, solid = data, dashed = model, lw 2. Prior-distance
I is `gold`, M is `tomato`. S-by-contrast uses `Blues` from 0.4 to 1,
reversed (low contrast darkest), same as `plot_S_diff_by_contrast_side_with_data`.

`xlim` / `ylim` are the data limits of the measured spines, so the labelled
ticks land on the published tick marks. If a curve exceeds a ylim the axis
is widened just enough to keep it on the page; the tick list stays the
published one.

| panel | box (x0, y0, x1, y1) | xlim | ylim | ticks |
|---|---|---|---|---|
| b | 238.9, 11.0, 383.4, 133.5 | −6.6, 132.6 | −0.140, 0.75 | x 0/40/80/120; y 0/0.25/0.50/0.75 |
| c stim | 445.1, 9.4, 587.7, 133.0 | −7.1, 149.2 | −0.026, 0.548 | x 0/40/80/120; y 0/0.15/0.30/0.45 |
| c move | 615.9, 9.4, 758.5, 133.1 | −149.2, 7.1 | −0.020, 0.843 | x −120/−80/−40/0; y 0/0.25/0.50/0.75 |
| d stim | 822.5, 11.9, 957.7, 132.9 | −4.0, 84.1 | 0.033, 0.151 | x 0/25/50/75; y 0.06/0.09/0.12/0.15 |
| d move | 965.0, 13.1, 1100.1, 133.0 | −81.1, 3.6 | 0.033, 0.151 | x −75/−50/−25/0; y ticks unlabeled |
| e | 85.1, 192.1, 354.9, 316.7 | −6.4, 134.5 | −0.051, 1.215 | x 0/40/80/120; y 0/0.4/0.8/1.2 |
| f | 451.2, 189.5, 700.6, 315.8 | −4.5, 120 | −0.030, 0.304 | x 0/40/80/120; y 0/0.1/0.2/0.3 |
| g | 820.9, 187.1, 1066.0, 315.7 | −151, 5 | −0.043, 0.801 | x −120/−80/−40/0 (unlabeled mark at −144); y 0/0.25/0.50/0.75 |
| h | 19.4, 355.0, 1073.2, 478.0 | no spines | | |
| i perf | 72.2, 499.2, 327.5, 629.3 | symlog −2, 2 | 23.5, 103.6 | x −1/−0.1/0/0.1/1; y 25/50/75/100 |
| i RT | 442.4, 502.8, 690.2, 629.0 | symlog −2, 2 | 0.086, 0.470 | x same; y 0.15/0.30/0.45 |

h has no frame. Trace colours match `plot_overall_traj`: S `#4A90E2` (L at
α 0.5), I `#FFC400`, M red, P `#8B2BE2` ×10, trial side black between obs
onset and action, block side silver lw 2. The PDF restyles the verticals:
obs onset `#F08F21` dashed, action `#1B9CD8` dashed, both lw 0.8. Trace lw
0.5. Labels (11.3 pt) use the PDF colours: s_L `#558dca`, s_R `#1d78b5`,
c_L `#ffc50f`, c_R `#fbab1a`, m `#ed2124`, p `#7850a1`. “stim” and “block
side” are 13 pt (`#231F20` / `#737477`). A “0” sits on the right, 12 pt.

Y labels (11 pt):

| panel | label |
|---|---|
| b | $\|r_S^{\mathrm{stim},s}(t;c)\|$ |
| c (left only) | $r_{\{C,M\}}^{\mathrm{choice},\{s,m\}}(t)$ |
| d (left only) | $d_{\{C,M\}}^{\mathrm{prior},\{s,m\}}(t)$ |
| e | $d_S^{\mathrm{stim},s}(t)$ |
| f | $d_C^{\mathrm{choice},s}(t)$ |
| g | $d_M^{\mathrm{choice},m}(t)$ |

b also has the word “contrast” (11.6 pt, `#231F20`) and a downward arrow.
The R² next to it is blue.

### Which function feeds which panel

Same calls as `paper-brain-wide-map/model_test.ipynb`, on one shared
bps=20, seed **12345** session (`plot_best_fit_results.make_shared_stimuli`).
Fit targets come from repo `fit_targets/` via `ensure_fit_data_links_paper`.
Distance files are read from openalyx `manifold/res`. Nothing there is rewritten.
Helper svgs go to `paper_figs/fig5/_scratch/`.

| panel | function | window |
|---|---|---|
| b | `mean_S_by_contrast` (T=65) + `plot_S_diff_by_contrast_side_with_data` (right side only) + `compute_sse_stim_right` for the one R² | ~130 ms, ticks through 120 |
| c | `mean_by_condition` at `im_traj_T_of` (72, 72) + `loss_plot_diff_by_condition_with_data` (`I`, `M`) | stim 0–144 ms; move −144–0 |
| d | `loss_prior_effect`, `plot_window=80`, `plot_stim=False`, `plot_shifted=False`, `scale_factors=[1,1,1]` | 80 ms; I and M only |
| e | `load_and_plot_stim_dist_data(stim_regs, stim_{1.0,0.25,0.125,0.0625,0.0})` then `plot_S_dist_by_contrast` | per-contrast overlay, not the average panel. ylim notebook `[-0.05, 1.2]` |
| f | `plot_dist_I_M_P` **dc_I** (`choice_duringstim_act` on `int_regs`) | xlim (−5, 120) inside the function; paper xlim (−4.5, 120) |
| g | `plot_dist_I_M_P` **dc_M** (`choice_duringchoice_act` on `move_regs`) | 72×2 ms. The function’s time axis is 0→144; the paper runs it backward so the last sample (the action) is at 0 |
| h | same series as `plot_overall_traj` | first block switch that has 4 trials before it and 5 after (~9 trials) |
| i | cached `psychometric_model_vs_data_actprior/combined_results.npy`, then `results_with_subjective_prior`, then `loss_perf_with_data` | correct, split con/incon; RT `combined_all`. Behavior is openalyx `manifold/res/behavior_actprior.npy`. No legend, no title |

Real choice-distance curves are `analysis_functions.plot_average_distance_over_regions`
(`return_mean=True`, `p_mean_c`, α=0.01), notebook cell 73.

**Scales.** Passed into the same plot functions the notebook used, then
harvested for the page. Each factor is the least-squares match of this
model’s curve to the data on the plotted window. Time shifts stay 0.
Notebook cell 72 used 1.05 on every S contrast; cell 76 used
`{'int': 0.67, 'move': 0.88}` for the published weights.

| panel | function argument | factor |
|---|---|---|
| e | `plot_S_dist_by_contrast` `scale_factors` | **0.906** on every contrast |
| f | `plot_dist_I_M_P` `int` | 0.590 on 0–120 ms (full-curve least squares is 0.663) |
| g | `plot_dist_I_M_P` `move` | 0.795 on the whole move-aligned curve |

The annotated R² uses that same window. Panel e is one factor for every
contrast. The 0 and 0.0625 curves still disagree with the data in shape.

**ΔAIC.** Not the published −1181.72. Recomputed as
AIC(fitted) − AIC(g_i = d_i = g_m = d_m = g_s = d_s = 0) with
`corrected_aicc_from_total_loss` on traj I+M plus prior I+M (no S nSSE).
N = 160 + (57+72)×4 = 676. k_fit = 21 (`train_mask` is 21 True), k_null = 15.
Both losses use `m_pre_weight=1` (the JSON stores 3; trajectories and R²
do not depend on it). The null is a second `run_model` on the same stimuli.

```bash
conda activate iblenv   # outside sandbox
python scripts/paper_figs/fig5/fig5_plot.py
```

`--recompute` ignores `cache/series.pkl`. Outputs:
`figure5.{pdf,svg,png}`, `summary.json`, `original/figure5.pdf`.

### This seed (s89, shared bps=20 / seed 12345)

| panel | R² |
|---|---|
| b S contrast | 0.63 |
| c stim I / M | 0.965 / 0.930 |
| c move I / M | 0.966 / 0.981 |
| d prior I+M | 0.950 |
| e S distance (scale 0.906) | 0.914 |
| f C choice, stim (scale 0.590) | 0.919 |
| g M choice, move (scale 0.795) | 0.956 |
| i con / incon / RT | 0.338 / 0.906 / 0.751 |

ΔAIC = **−1081.05** (AIC fitted −4779, null −3698). g’s R² sits upper-left because the curve fills the upper-right corner where the PDF places that number. h is the first block switch with four trials before it and five after (trials 35–44, 6146 steps). SNc is missing from the integrator region list in openalyx `res` and is skipped by the distance average.
