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
| **Environment** | `conda activate iblenv` | Run **outside** sandbox on this machine (see below) |

### Do not run ONE / analysis jobs in the Cursor sandbox

Anything that touches ONE caches (`~/Downloads/ONE/...`), large `manifold/res/*.npy`,
or `iblenv` analysis scripts **hangs or sits at 0% CPU inside the sandbox** (seen
repeatedly on combine/plot/null jobs). Always:

1. Request `required_permissions: ["all"]` (disable sandbox), and
2. `conda activate iblenv` before `python …`.

Do not retry the same ONE command under the default sandbox hoping it will finish.

### Phase 4b sanity check

Before trusting new analysis paths, verify split-conditioned Phase 4b matches the retest:

```bash
conda activate iblenv
python simulate_recovery.py --phase4-no-prior-mod \
  --seed 123 --n-sessions 40 --nrand 100 --n-jobs 8
```

**Expected (seed 123):** S curve_mean ≈ **0.012**, p_mean ≈ **0.78** (not significant). I/M also null.

Source: [canonical_analysis_conventions.md](journals/canonical_analysis_conventions.md) (2026-06-19b retest).

### Common pitfalls

1. **Do not pool left- and right-stim trials in one S distance** without stim-side splits — creates spurious S signal even with g/d=0.
2. **Unsplit** (`--unsplit-prior`) means no f1/f2 choice×feedback splits; it still uses **stim_l + stim_r** unsplit splits, stacked.
3. **Old results** using zero-padding or 150 ms S window are invalid for significance claims.
4. Trajectory plots must use the same 80 ms S cap as distance analysis (`trial_s_binned_signed`).

### Where conventions live

- **Code (source of truth):** `simulate_recovery.py` — module docstring, `CANONICAL_PRIOR_DISTANCE_ANALYSIS`, `build_population_b_for_split`
- **Cursor rule:** `.cursor/rules/prior-distance-analysis.mdc` (auto-loaded for agents)
- **Experiment history:** `journals/*.md` — topic notes with dated results inside, not agent defaults

## Scope — latest ask wins (avoid over-fixing)

Long fitting/diagnosis threads accumulate menus of possible fixes. Agents must
**not** treat earlier proposals as approved work.

- Implement only what the **latest user message** asks for.
- Optimizer/search changes ≠ rewriting loss / NaN / bucket semantics unless asked.
- Diagnosis-only turns: no code changes unless requested.
- Sticky without explicit ask: S-bucket `<10` → NaN, fail-closed SSE, canonical
  prior-distance settings.

Cursor rule: `.cursor/rules/scope-latest-request.mdc` (always applied).

### Research journals (`journals/`) — develop only

- Journals are organized **by topic, not by date**: one file per line of investigation (e.g. `journals/s_prior_artifacts_truncation.md`), each holding the goal, the implementation, all dated updates, results, and open questions. `journals/README.md` is the index.
- When journaling: append a dated entry to the matching topic file, or create a new topic file and add it to the index. Do **not** start `research_journal_YYYY-MM-DD.md` files.
- Journals live **only on the `develop` branch**. Do **not** create, edit, or commit them on `main`.
- When journaling: check out / work on `develop`; put new files in `journals/`.
- `main` is for code/scripts needed to run analyses (e.g. ORCD); keep journals off it.
- If asked to start a journal while on `main`, switch to `develop` first (or warn the user).

### Fit speedups (`fit_retinal.py` / `fit_weights.py`)

- Topic journal: [journals/simulation_fit_speedups.md](journals/simulation_fit_speedups.md).
- Cursor rule: `.cursor/rules/fit-speedups.mdc` — **do not** shorten trials, cut
  `blocks_per_session`, or reuse stim batches across optimizer generations for speed.
  Numba must stay parity-checked vs numpy.

### Git — no merges; no commits/pushes without approval

- **Never** `git merge` / `git pull` / `gh pr merge`. Copy files with
  `git checkout <branch> -- <paths>` when asked.
- **Do not** `git commit` or `git push` unless the **latest message** says
  **commit** or **push**. `add` / `sync to main` / `put on both branches` is
  **not** a commit.
- Cursor rule: `.cursor/rules/git-commits-require-approval.mdc` (always applied).
- When porting code to `main` for ORCD, copy/`git add` as asked; wait for
  “commit” before running `git commit`.
