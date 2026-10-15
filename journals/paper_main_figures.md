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
| 4 (prior on region types) | `<oa>/manifold/paper_figs/fig4/` | `scripts/paper_figs/fig4/` | first rebuild (10-14); a = earlystim table, b header = sensory ∩ prior_80ms, c = LGd + combined |
| 4.5 (block effects on cell types) | `<oa>/manifold/paper_figs/fig45/` | `scripts/fig45_sensitivity.py fig45` | rough draft; see [Figure 4.5](fig45_block_effects_cell_types.md) |

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

First rebuild on the published page (`manifold/figs/figure4.pdf`, 990.3 × 1128.0 pt).
The integrator / choice-motor example grid and the panel-c schematic stay crops
of that PDF. Three pieces are new.

| panel | what changed | source |
|---|---|---|
| a | the region strip | alyx `meta/table_act_block_combined_summary_act_p_mean_c_combinedpTrue_0.01_overall_earlystim80.png` |
| b header | the check / cross cartoons | the 13 regions where `prior_80ms` is FDR ≤ 0.01 and `type_stim` is sensory (stim or stim early) |
| c curves | the two class averages | already-plotted LGd example and the cell-weighted sensory combined, alyx `manifold/figs/earlystim80_sensory_prior/` |

The overall table mixes caches. `prior_choice`, `type_choice`, `prior_stim`, and `type_stim` are the openalyx 4-split 150 ms maps (`manifold/res`). `prior_80ms` is the alyx `res/new` stim-side unsplit shuffle sliced to 80 ms (`combined_act_block_duringstim_r_act_block_duringstim_l_earlystim80.npy`). Sensory type is `sc_duringstim_regtype` ∈ {0, 0.1} from the Figure 2 openalyx `sc_table.csv`, the same `get_sc_table` settings as that table. Nothing in `manifold/res` is rewritten.

Overlap, in table order: VISam, TEa, CA1, LGd, LP, PO, ZI, SAG, PAG, PB, TRN, PRNr, DCO. The example in c is **LGd** (`--example` changes it). Both c curves are the 0–80 ms stim-aligned plots; the old right-hand axis was move-aligned.

```bash
conda activate iblenv   # outside sandbox
python scripts/paper_figs/fig4/fig4_plot.py
python scripts/paper_figs/fig4/fig4_plot.py --example VISam
```

Outputs under `<oa>/manifold/paper_figs/fig4/`: `figure4.pdf`, `figure4.png`, `panels/`, `summary.json`, and `original/figure4.pdf`.
