# Research journal — 2026-07-12

## Standing context (carry-forward from 2026-07-06)

Real-data pipeline only — see [prior journal](research_journal_2026-07-06.md) for Goals 2–3 (insertion cache, stream_pool, contrast-stratified during-trial splits) and ORCD sharding notes (07-12a–d). Do not carry forward generative-model / simulated fitting results.

### Real-data analysis stack

| Piece | Status / location |
|-------|-------------------|
| Driver | `block_analysis_allsplits.py` |
| Insertion cache | `manifold/insertion_cache/{eid_probe}.npy` (spikes + trials once per pid) |
| Multi-split | outer loop = insertions, inner = splits; `stream_pool` → `manifold/res/` |
| Contrast splits | `'{base}_{contrast}'` on duringstim / duringchoice (act + non-act), incl. 0% |
| Min trials | ≥5 per split side (`InsufficientTrials` → skip) |
| Sharding | `scripts/submit_goal2_*` / `submit_goal3_sharded.sh` (ORCD `mit_normal`) |
| Null (current) | label shuffle within condition (contrast-matched where applicable) |

---

## Today's goals

**Scope:** BWM / real recordings only (`block_analysis_allsplits.py` and related real-data scripts). No simulated sessions.

### Goal 1 — Bayesian (sensory) prior on sensory responses; conflict vs alignment with recent reward

On **real data**, test whether sensory-population prior effects hold when the prior is framed as a **Bayesian / sensory prior**, and stratify trials by whether the **current block prior conflicts** vs **aligns** with **recent reward history**.

Motivation: block identity and recent outcomes can pull in opposite directions; pooling them may mask or invent a sensory prior effect.

**Questions:**
1. Is there a significant prior effect in sensory responses under a Bayesian prior framing?
2. Does any effect concentrate on conflict trials, alignment trials, or both?
3. Does the conclusion change relative to the usual block-prior split (pooled across recent-reward history)?

### Goal 2 — Null that respects choice / prior autocorrelation (Harris-style or matched-spectrum surrogates)

**Problem:** Label-shuffling **within stim × block** destroys **across-trial choice autocorrelation**. Animals show one-side response epochs (runs of L or R choices). Destroying that structure **narrows the null** and **inflates false positives** for **choice** and **prior** effects — not for stimulus (stim labels lack that temporal structure in the same way).

**Plan:** redo significance for choice/prior contrasts on real data using a null that preserves temporal structure, e.g.:
- **Session-shuffled** null (Harris 2020), or
- **Surrogate data** with matched autocorrelation / power spectrum

**Scope:** affects **choice** and **prior** analyses; **stimulus** significance claims are not the target of this correction.

---

## Notes / results

### 2026-07-12e — Action-kernel prior (existing) + Bayes-optimal prior (implemented)

**Action kernel** (`action_kernel_priors`, α=0.2) — already in `get_d_vars` when `'act' in split`:

1. Drop true-block 0.5 trials (`probabilityLeft != 0.5`).
2. EMA over **choices**: `prior ← α·[choice>0] + (1−α)·prior` (choice>0 = left).
3. Binarize to **0.8 / 0.2** (`≥0.5 → 0.8`) and **overwrite** `probabilityLeft` (true block kept in `true_priors`).
4. Downstream L-vs-R-prior splits use that overwritten column unchanged.

**Bayes-optimal prior** (Findling et al. Nature 2025 SI §1.1.1) — new `bayesian_priors`, wired the same way when `'bayes' in split`:

1. Infer continuous **P(stim left | past stimulus sides)** under the IBL generative model (τ=60, γ=0.8, block len 20–100).
2. Computed on the **full** trial list **before** dropping 0.5 blocks (needs full stim history); then same 0.5 filter + overwrite `probabilityLeft` with 0.8/0.2; continuous trace in `bayes_priors`, true block in `true_priors`.
3. Split names mirror act: `bayes_block_*` (duringstim / duringchoice / stim_l / stim_r / only) + Goal-3 contrast registration (`GOAL3_BAYES_*`).
4. Presets in `scripts/run_goal2_splits.py`: `stimOn_times_bayes`, `goal3_duringstim_bayes`, `goal3_duringchoice_bayes`, `goal3_bayes_all`.

**Smoke (priors only):** 80 left stims → P(left)≈0.77 (→γ); 80 right → ≈0.23; mid-session switch flips binary 0.8→0.2.

**Next:** run real-data `bayes_block_*` vs `act_block_*` / true-block; then conflict vs alignment (true block vs recent reward / vs Bayes).

### 2026-07-12f — Rename stim-side splits: were ITI, now during-stim

**Bug:** bare `*block_stim_l` / `*block_stim_r` (no choice×feedback) lacked `durings`/`duringc` in the name, so align setup assigned the **intertrial** window `[0.4, −0.1]` despite stimOn alignment.

**Fix:** rename → `*block_duringstim_l` / `*block_duringstim_r` (`block_`, `act_block_`, `bayes_block_`). Name substring `durings` now selects post-stim `[0, 0.15]`. Scripts/presets updated (`stimOn_times_act`, `stimOn_times_bayes`, submitters).

Any cached outputs under the old names are **invalid for during-stim claims** and must be re-run.

### 2026-07-12g — Lower mem for Goal 2 sharded submit (8 splits)

**Estimate:** stream_pool peak RSS ≈ **1.5–2.5 GB** / shard (nrand=2000; 07-10b). Finalize merges 4 shard accs → modestly above one shard.

**Defaults** (`submit_goal2_stimOn_act_sharded.sh` + worker `#SBATCH`):

| | was | now |
|--|-----|-----|
| MEM_SHARD | 12G | **6G** |
| MEM_FIN | 16G | **10G** |
| Concurrent (8×4 shards) | 384G | **192G** |

Override if OOM: `MEM_SHARD=8G MEM_FIN=12G bash scripts/submit_goal2_stimOn_act_sharded.sh`.

### 2026-07-12h — Bayes stim L–R splits in cached pipeline (not choicestim)

Ported choicestim’s **stim L vs R** contrasts into `block_analysis_allsplits.py` with Bayes-optimal prior labels. Do **not** revive `choicestim_analysis.py` for new runs (legacy docstring only).

| Timeframe | Splits | Window |
|-----------|--------|--------|
| `stim_duringstim_bayes` | `stim_choice_{r,l}_block_{r,l}_bayes` | `[0, 0.15]` — stim L/R, fixed choice + Bayes prior |
| `stim_duringstim1_bayes` | `stim_block_{l,r}_bayes` | `[0, 0.08]` — stim L/R, fixed Bayes prior only |

Parity with choicestim act: **no** true-block 0.5 drop before prior overwrite for these names.

```bash
python scripts/run_goal2_splits.py --preset stim_duringstim_bayes --list-splits
python scripts/run_goal2_splits.py --preset stim_duringstim1_bayes
python scripts/run_goal2_splits.py --preset stim_lr_bayes_all
```
