# Research journals

Experiment notes organized **by topic**, not by date. Each file covers one line of investigation end to end: the goal, the implementation, every dated update, the results, and the open questions. **Develop branch only** — not committed to `main`. See `AGENTS.md`.

Dated entries are preserved inside each topic file (with their original date tags) so the chronology is still readable; the pre-2026-08 dated files (`research_journal_YYYY-MM-DD.md`) were folded into these and removed — see git history if you need the original layout.

## Generative model / simulation (`simulate_recovery.py`)

| Topic | What's in it |
|-------|--------------|
| [Canonical analysis conventions](canonical_analysis_conventions.md) | The mandatory defaults (80 ms S window, fill-from-next-ITI, contrast-matched null, output roots), the Phase 4b regression check, sandbox/env rules |
| [S prior artefacts: truncation and the Phase 4b residual](s_prior_artifacts_truncation.md) | Why absence/Phase 4b showed spurious S prior distance; Phases 0–5; code audit; the zero-padding bug and its fix; post-fix retests; the trajectory-plot version of the same bug |
| [Split conditioning vs unsplit prior distance](split_conditioning_vs_unsplit.md) | f1/f2 composition artefact; stim-side vs fully unsplit; contrast-matched vs label-shuffle null comparison (Tables A–C) |
| [Direct sensory prior coupling (g_s / d_s)](direct_sensory_prior_coupling.md) | Whether direct P→S coupling is detectable; all g_s/d_s sweeps; adaptation-gate placement; I-first vs S-first thresholds; presence unsplit sweep |
| [Simulation infrastructure](simulation_infrastructure.md) | Session cache, unified `--run-experiment` entry point, the 4 × 5 analysis matrix; 2026-08-14 cache wipe + Harris/long-session → ORCD |
| [Faster model fitting](simulation_fit_speedups.md) | Fit to baseline loss (~0.40) in ≲1–2 h; weights-only ORCD batch / optimizer speedups |
| [Joint fitting pipeline](joint_fitting_pipeline.md) | Retinal + `g_s`/`d_s` + weights (`L_w+L_S`); regular vs sensory freeze masks; ORCD drivers; joint-direct fair compare |
| [Retinal then joint](retinal_then_joint_fitting.md) | Pivot: fit retinal @ all prior g/d≈0, then joint with retinal free to tweak; regular/sensory; modernize `fit_retinal` |
| [Testing / revising model details](modeling_details_revisions.md) | Ablations: P→I/M ITI gate, I/M −400→−100 ms zero penalty, `W_pp` slow-prior box, one vs two action thresholds; 3× pre-action M (08-31); I/M prior at full 150 ms (09-07); `im150stim` / `stimonly` (09-08; open `g_i` floor worse; restore `0.1`); model I/M distance = `mean_c ‖Δ‖` (09-08f; im150 meancell rerun) |
| [Prior-curve dips and discordant RT](modeling_details_prior_rt_gaps.md) | Remaining shape gaps after tests 1–6: **M** stim-aligned ramps through the S-peak pause (I is good enough); incongruent RT R² negative / wrong contrast shape. `im150_meancell` scored 09-09; neither gap closed |
| [Fit g_s/d_s with I/M prior mods](fit_gs_ds_with_im.md) | Joint `full` mask (all prior gains free); S prior-distance target from unsplit 80 ms FDR@0.01 ∩ stim/stim_early (13 regions) |
| [BWM classification recovery](bwm_classification_recovery.md) | The `--full-analysis` Σ classifier on simulated experiments; decorrelation-window and plotting fixes |

## Real data (`block_analysis_allsplits.py`, BWM)

| Topic | What's in it |
|-------|--------------|
| [Real-data pipeline efficiency](realdata_pipeline_efficiency.md) | Insertion cache, loop reorder, stream pooling, ORCD sharding, memory settings, `min_trials_per_side` |
| [Prior definitions and label conventions](prior_definitions.md) | True block vs action kernel vs Bayes-optimal; ITI true-block uses trial t−1 (08-28); split naming; the prior-type routing fix; drop-0.5 and fixed-α open questions |
| [ITI prior (stimOn −400 to −100 ms)](iti_prior.md) | Three `*_block_only` labels (08-28 lag); default `{split}.npy` is pseudo-blocks not a shuffle (fair only for true block); Harris 08-31; matched `_pseudosession` 09-01 (act 7-hit FDR wiped) |
| [Bayesian prior (real data)](bayesian_vs_act_prior.md) | Shuffle 4-split **57** / stim-side **116** FDR @0.01 do **not** survive Harris (both **0**; 09-01). Bayes-stratum shuffles: choice L–R **100**, stim L–R **47** |
| [Prior modulation by contrast](prior_modulation_by_contrast.md) | Contrast-stratified during-trial splits, cell retention, FDR p-floor analysis, and the revised 0 %-contrast choice-conditioned result |
| [Structured nulls for choice L–R](structured_nulls_choice_lr.md) | Why label shuffle is too narrow; Harris; AK; option-1 + copy-last (`_pseudo_strat_sticky`); Bayes-agent sampler (08-23) and Bayes Harris unique submitter (08-27b), FDR not yet run; fully unsplit prior FDR (08-24e) |
| [Sticky / end-of-session trial exclusion](sticky_end_of_session_exclusion.md) | Late 20 % ∪ perseveration-tail drop; choice FDR *expanded*; prior 4-split duringstim expands @0.05 / shrinks @0.01 (08-25); f1/unsplit *shrink* and stay Harris-nonzero; sticky tails not concentrated late; last 20 % slower not inaccurate / not more block-aligned |
| [Single-neuron variance partition](variance_partition_mixed_regions.md) | Mixed stim×choice target set, OLS variance partition, full BWM results, neuron- and region-level nulls |

## Other files

- `action_kernel_model.tex` / `.pdf` — action-kernel model write-up.
