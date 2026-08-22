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
| [BWM classification recovery](bwm_classification_recovery.md) | The `--full-analysis` Σ classifier on simulated experiments; decorrelation-window and plotting fixes |

## Real data (`block_analysis_allsplits.py`, BWM)

| Topic | What's in it |
|-------|--------------|
| [Real-data pipeline efficiency](realdata_pipeline_efficiency.md) | Insertion cache, loop reorder, stream pooling, ORCD sharding, memory settings, `min_trials_per_side` |
| [Prior definitions and label conventions](prior_definitions.md) | True block vs action kernel vs Bayes-optimal; split naming; the prior-type routing fix; drop-0.5 and fixed-α open questions |
| [Prior modulation by contrast](prior_modulation_by_contrast.md) | Contrast-stratified during-trial splits, cell retention, FDR p-floor analysis, and the revised 0 %-contrast choice-conditioned result |
| [Structured nulls for choice L–R](structured_nulls_choice_lr.md) | Why label shuffle is too narrow; Harris; AK; option-1 + copy-last (`_pseudo_strat_sticky`) wired, FDR not yet run |
| [Sticky / end-of-session trial exclusion](sticky_end_of_session_exclusion.md) | Late 20 % ∪ perseveration-tail drop; choice FDR *expanded*; sticky tails not concentrated late; last 20 % slower not inaccurate / not more block-aligned; late `mean_run` is clumpiness not a rate shift; next: same trim on prior splits |
| [Single-neuron variance partition](variance_partition_mixed_regions.md) | Mixed stim×choice target set, OLS variance partition, full BWM results, neuron- and region-level nulls |

## Other files

- `action_kernel_model.tex` / `.pdf` — action-kernel model write-up.
