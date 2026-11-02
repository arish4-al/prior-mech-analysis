# Agent guide — prior-mech-analysis

## Canonical prior-distance analysis (mandatory since 2026-06-19)

All simulation / prior-distance experiments in this repo **must** use these defaults. They are implemented in `simulate_recovery.py` (`build_population_b_for_split`, `CANONICAL_PRIOR_DISTANCE_ANALYSIS`).

| Setting | Value | Notes |
|---------|-------|-------|
| **S analysis window** | **80 ms** post-stim | `S_DURINGSTIM_WINDOW_S = 0.08` |
| **I/M analysis window** | **150 ms** post-stim | `PRE_POST` duringstim splits |
| **Truncated trials** | **fill-from-next-ITI** | Never zero-pad; skip if next ITI too short |
| **Null** | **contrast-matched shuffle** | Default CLI; `--label-shuffle-null` to override |
| **Output root** | `<ONE cache>/manifold_sim` | Do not use repo `output/` unless `--allow-repo-output` |
| **Environment** | `conda activate iblenv` | Run outside sandbox on this machine |

### Cluster jobs never contact Alyx (since 2026-11-02)

On 11-01/02 the standard-split Harris/sticky shards sent ~120k Alyx requests
(remote-mode `load_trials_and_mask` per insertion in ~200 concurrent jobs) and
throttled the database for other IBL users. Now:

- Create ONE only through `make_one()` (`one_offline.py`). In Slurm it is
  `mode='local'` and HTTP to Alyx raises `AlyxBlockedError`; submit wrappers
  export `BLOCK_ALYX=1` (`scripts/sbatch_defaults.sh`).
- Per-insertion / per-split code reads disk only. Missing local data stops
  the job (`is_missing_local_data`), it does not skip the insertion.
- `scripts/check_offline_inputs.py` runs before any standard alt-null submit.
- Before pushing cluster code: one real insertion with `SLURM_JOB_ID=test`
  through the changed path, and report Alyx requests per job × jobs (must be 0).

Cursor rule: `.cursor/rules/no-alyx-in-cluster-jobs.mdc` (always applied).

### Phase 4b sanity check

Before trusting new analysis paths, verify split-conditioned Phase 4b matches the retest:

```bash
conda activate iblenv
python simulate_recovery.py --phase4-no-prior-mod \
  --seed 123 --n-sessions 40 --nrand 100 --n-jobs 8
```

**Expected (seed 123):** S curve_mean ≈ **0.012**, p_mean ≈ **0.78** (not significant). I/M also null.

Source: [research_journal_2026-06-18.md](research_journal_2026-06-18.md) (2026-06-19b retest).

### Common pitfalls

1. **Do not pool left- and right-stim trials in one S distance** without stim-side splits — creates spurious S signal even with g/d=0.
2. **Unsplit** (`--unsplit-prior`) means no f1/f2 choice×feedback splits; it still uses **stim_l + stim_r** unsplit splits, stacked.
3. **Old results** using zero-padding or 150 ms S window are invalid for significance claims.
4. Trajectory plots must use the same 80 ms S cap as distance analysis (`trial_s_binned_signed`).

### Where conventions live

- **Code (source of truth):** `simulate_recovery.py` — module docstring, `CANONICAL_PRIOR_DISTANCE_ANALYSIS`, `build_population_b_for_split`
- **Cursor rule:** `.cursor/rules/prior-distance-analysis.mdc` (auto-loaded for agents)
- **Experiment history:** `research_journal_*.md` — dated results, not agent defaults

- **ORCD submit scripts:** source `scripts/sbatch_defaults.sh` after `PARTITION` is set and pass `$SBATCH_EXTRA` to every `sbatch`. `PARTITION=mit_preemptable` (or `mit_preem`) defaults `--requeue`.

### Research journals (`research_journal_*.md`) — develop only

- Journals live **only on the `develop` branch**. Do **not** create, edit, or commit them on `main`.
- When journaling: check out / work on `develop` (or open the journal from that branch).
- `main` is for code/scripts needed to run analyses (e.g. ORCD); keep journals off it.
- If asked to start a journal while on `main`, switch to `develop` first (or warn the user).
