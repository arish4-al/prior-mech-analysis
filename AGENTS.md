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
