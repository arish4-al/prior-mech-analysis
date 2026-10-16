# Paper SI figures

**Goal.** Supplementary figures for the paper, drawn from analyses that are
already finished. Same place as the main figures: openalyx
`manifold/paper_figs/<fig>/` holds the figure, `panels/`, and the tables the
plot reads. Scripts live in `scripts/paper_figs/<fig>/`.
`<oa> = ~/Downloads/ONE/openalyx.internationalbrainlab.org`.

| figure | folder | scripts | status |
|---|---|---|---|
| Recovery (classification and sensory prior) | `<oa>/manifold/paper_figs/si_recovery/` | `scripts/paper_figs/si_recovery/` | redrawn 10-15; right arm is full mpre3 s89 |

---

## 2026-10-15 — Model recovery

### What the pipeline is

`simulate_recovery.py` simulates S/I/M/P trajectories, then runs the same
tests as the data.

| step | CLI | what it scores |
|---|---|---|
| Classification | `--run-experiment {phase4,absence,s_presence} --full-analysis` | BWM Σ classifier (`classify_regions`). Stimulus if $\Sigma^{\mathrm{stim},s} > 0.8$; else movement if the choice curve is monotonic; else integrator. |
| Prior, stim × choice | same `--full-analysis` run | 4-split `act_block_duringstim` (stim side × choice × feedback). This is the f1/f2 composition test. |
| Prior, stim only | `--unsplit-prior {absence,s_presence}` (default `stim_side`) | S/I on stim L vs R only. No choice split. |

Canonical settings on every run: 80 ms S window, 150 ms I/M window,
fill-from-next-ITI, contrast-matched label shuffle, seed 123, 40 sessions,
nrand 100. Details and the decorrelation-window fixes are in
[BWM classification recovery](bwm_classification_recovery.md) and
[split conditioning vs unsplit](split_conditioning_vs_unsplit.md).

### Which models have been through it

**November 2025 weights** (`WEIGHTS_REL`, loss 0.404). Four experiments, full
classification plus both prior strata. Write-ups:
[classification](bwm_classification_recovery.md),
[split vs unsplit](split_conditioning_vs_unsplit.md).

| experiment | ground truth | classification | S prior, stim × choice | S prior, stim only |
|---|---|---|---|---|
| phase4 | all $g = d = 0$ | 1/3 | null (0.012, p = 0.78) | null (0.003, p = 0.64) |
| absence | $g_s = d_s = 0$, fitted I/M | **3/3** | significant (0.80) | null (0.011, p = 0.13) |
| s1800 | $g_s = 1800$, I/M off | 1/3 | 0.042, p = 0.04 | 0.020, p = 0 (CM) |
| s2025 | $g_s = 2025$, I/M off | 1/3 | 0.054, p = 0 | 0.025, p = 0 (CM) |

Absence is the composition result: stim × choice reports a sensory prior that
is not in the model; stim only does not. s1800 and s2025 use a direct P→S
gain far above the later fits, and the classifier misses S and M.

**Stage B regular s101** (2026-08-13) is the left arm of the figure: no
sensory prior, fitted I/M. Write-up:
[retinal then joint](retinal_then_joint_fitting.md) 08-13b and 08-13c.
Outputs under `<oa>/manifold_sim/stageB_bwm/`.

| arm | weights | ground truth |
|---|---|---|
| regular s101 | `weights_run_fj_stageB_hold_s89_regular_mask12-13_s101`, loss 1.131 | $g_s = d_s = 0$, $g_i \approx 196$, $d_i \approx 20$, $g_m \approx 0.20$ |

An earlier right arm, sensory s23 (P→S only, $g_s \approx 38$, $d_s \approx 33$,
I/M prior off), is in the same 08-13 write-up. The figure uses the full model
below instead.

**Full mpre3 s89** (2026-10-15 run). Figure 5 weights
`weights_final_loss1p125_20260915-224209.json` in
`weights_run_fj_stageB_hold_s89_full_mpre3_meancell_full_masknone_s89`.
Nothing is frozen. Fitted sensory prior is an offset: $g_s \approx 0$,
$d_s \approx 49.7$, with $g_i \approx 138$. Same canonical settings (seed 123,
40 sessions, nrand 100). Session cache miss on the classifier, hit on the
stim-only pass. Outputs under `<oa>/manifold_sim/full_mpre3_meancell_s89/`.

### Figure

Left: regular s101, no sensory prior. Right: full mpre3 s89, sensory offset
on and I/M prior on.

**a.** Each population on the classifier plane ($\Sigma^{\mathrm{stim},s}$
vs monotonicity). Shaded regions are the assignment rule. Both arms are
**3/3**.

| | S | I | M |
|---|---|---|---|
| no sensory prior | stimulus (Σ = 0.993) | integrator (0.351) | movement (mono = 1) |
| full model | stimulus (Σ = 0.928) | integrator (0.427) | movement (mono = 1) |

**b.** S prior distance (curve mean). A star is $p_{\mathrm{mean}} \le 0.01$.

| | stim × choice | stim only |
|---|---|---|
| no sensory prior | **0.199, significant** | **0.0027, p = 0.63** |
| full model | 0.134, significant | 0.096, significant |

Stim × choice reports a sensory prior for regular s101, where $g_s = d_s = 0$.
Stim only does not. On the full model the sensory offset stays significant
after the choice split is dropped (0.134 → 0.096). Stim-only I and M stay
significant in both arms (regular 0.50 / 1.64; full 0.58 / 1.58).

```bash
python scripts/paper_figs/si_recovery/si_recovery_plot.py
```

Outputs: `<oa>/manifold/paper_figs/si_recovery/si_recovery.{pdf,svg,png}`,
`panels/panel_{a,b}.{pdf,svg,png}`, `classification.csv`,
`prior_by_stratum.csv`, `summary.json`.
