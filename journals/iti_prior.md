# ITI prior (stimOn [−400, −100] ms)

**Scope:** prior L–R in the pre-stimulus window for the three `*_block_only` splits — no stim/choice/f1/f2 stratum. Label conventions, default vs Harris results after the 08-28 true-block lag, and the unstratified `_pseudosession` null (one generative process per prior type).

**Status:** labels and default/Harris reruns are scored (08-31). Unstratified `_pseudosession` is wired and smokes locally; **not yet run on ORCD**.

Sources: [prior definitions](prior_definitions.md) 08-28; [structured nulls](structured_nulls_choice_lr.md) 08-14d / 08-18 (`act_block_only` only).

---

## Window and labels

Aligned to **trial t's** `stimOn`, window `[0.4, −0.1]` (after t−1's action, before t's stim). Alignment does not pick the prior.

| Split | Observed label | Causal? |
|-------|----------------|---------|
| `block_only` | `probabilityLeft[t−1]` | **fixed 08-28** (was trial t) |
| `act_block_only` | `priors[t]` = AK of `action[0..t−1]` | already |
| `bayes_block_only` | `priors[t]` = P(left on t \| stims 1..t−1) | already |

True-block lag: drop 0.5, drop the first remaining trial, assign each later row the previous remaining trial's `probabilityLeft`. During-stim / during-choice true-block still uses trial t.

---

## Nulls (no stratum)

| Null | Suffix | What the labels are |
|------|--------|---------------------|
| default | `{split}.npy` | `generate_pseudo_blocks` 0.8/0.2 for **all three** names (`first5050=0`; true-block uses `ntr+1` then `[1:]`). **Not** AK-prior or Bayes-from-stim. |
| Harris unique | `{split}_harris_unique.npy` | Other eids' observed priors (true-block lagged; act/Bayes recomputed on the donor) |
| **pseudosession** (new) | `{split}_pseudosession.npy` | Full IBL-like pseudo-session, then the **same** prior definition as the observation: lagged pLeft / AK of synthetic choices / Bayes of the stim sequence. No remade stratum, not `fixedstim`, not Harris. Only act generates choices through the fitted model. |

Default `{split}.npy` for act/Bayes ITI is therefore a **true-block-like** control. The `_pseudosession` arm is the matched generative null for each prior type.

---

## 2026-08-28 — true-block ITI lag

See [prior definitions](prior_definitions.md) 08-28. Unit tests: `python scripts/test_iti_true_block_lag.py`.

ORCD rerun of default + Harris (lag-corrected true-block; act/Bayes labels unchanged):

```bash
FAMILY=all NULL=both bash scripts/submit_goal2_iti_prior_orcd.sh
```

---

## 2026-08-31 — default + Harris FDR (lag-corrected)

Local alyx `manifold/res/new/`, mtime 30 Aug. All six files present. Single-split BH-FDR on `p_mean` from `*_regde` (same scoring as 08-14d). 208 regions, ~63.5k cells. Default `n_null=2000`; Harris min/med/max **352 / 2000 / 2000**.

| Prior | Null | raw ≤0.01 | FDR @0.01 | raw ≤0.05 | FDR @0.05 | median p |
|-------|------|----------:|----------:|----------:|----------:|---------:|
| true block | default | 0 | **0** | 1 | **0** | 0.88 |
| true block | Harris | 2 | **0** | 12 | **0** | 0.53 |
| act | default | 9 | **0** | 14 | **7** | 0.67 |
| act | Harris | 7 | **0** | 27 | **0** | 0.32 |
| Bayes | default | 0 | **0** | 0 | **0** | 0.91 |
| Bayes | Harris | 2 | **0** | 9 | **0** | 0.47 |

The only FDR map is **act + default @0.05**: AIp, CLA, FOTU, LSr, PIR, SSp-n, VISa. Harris wipes it. True-block and Bayes are complete nulls on both arms.

### vs 08-14d `act_block_only` default

The 08-14d file was overwritten. Comparison is to that journal table (same `p_mean`-from-regde scoring). Act labels were already `priors[t]`; `FAMILY=all` recomputed the 2000-draw (not the t−1 lag).

| | 08-14d | 08-30 rerun |
|---|---:|---:|
| raw `p_mean` ≤0.01 | 9 | 9 |
| raw ≤0.05 | 13 | 14 |
| FDR @0.01 | **0** | **0** |
| FDR @0.05 | **5** (AIp, CLA, LSr, SSp-n, VISa) | **7** (those five + **FOTU, PIR**) |
| median p | 0.689 | 0.672 |
| median amp | 0.172 | 0.170 |

Harris unique is essentially unchanged (0 FDR @0.01 and @0.05; median p 0.316 → 0.315).

---

## 2026-08-31 — unstratified `_pseudosession` (set up, not scored)

One control per ITI prior, **no stratum**:

| Split | Pseudo-session content | Null labels |
|-------|------------------------|-------------|
| `block_only` | IBL blocks (`generate_pseudo_blocks`, ~90-trial 0.5 warm-up then drop) | lagged `probabilityLeft` (same as observation). **No** choice model. |
| `act_block_only` | same blocks + stims + **fitted AK** choices | `action_kernel_priors` on the biased choices |
| `bayes_block_only` | same blocks + stims | `bayesian_priors(stim_side)` on the full sequence, then drop 0.5. **No** choices. |

Length: biased leftover ≈ real `n_elig` (`_strat_pseudo_n_trials`, +1 for true-block lag); then a contiguous window if the draw is longer. Disk: `{split}_pseudosession.npy`. Does **not** overwrite `{split}.npy` or `_harris_unique`. During-trial `act_block_*` unconstrained still raises (use `strat` / `fixedstim`).

Smoke: `python scripts/test_iti_true_block_lag.py` (includes the unconstrained path; act uses a fake choice model so it does not need MCMC).

### ORCD (do not submit from the laptop)

```bash
# all three priors; writes _pseudosession only
FAMILY=all NULL=pseudo bash scripts/submit_goal2_iti_prior_orcd.sh

# one family
FAMILY=block NULL=pseudo bash scripts/submit_goal2_iti_prior_orcd.sh
FAMILY=act NULL=pseudo PREFIT=1 bash scripts/submit_goal2_iti_prior_orcd.sh
FAMILY=bayes NULL=pseudo bash scripts/submit_goal2_iti_prior_orcd.sh
```

`PREFIT=1` only if AK pickles are missing (`manifold/actkernel_fits/`). Default **24 shards** (~29 insertions/shard; 12 missed 5 h on default act), `TIME_SHARD=5:00:00`, `TIME_FIN=5:00:00`, `MEM_SHARD=24G`, job prefix `g2itp`. `PARTITION=mit_preemptable` defaults `--requeue` (`sbatch_defaults.sh`). `CLEAR_STREAM=1` removes prior `_pseudosession` shards/pooled files only.

---

## Open

1. Score `_pseudosession` FDR against the 08-31 default/Harris table. Expect true-block ≈ default (same generative family). Act and Bayes should differ from default because default was true-block-like 0.8/0.2, not AK/Bayes labels.
2. Fixed α vs per-session AK fit for the **observed** act labels — still open ([prior definitions](prior_definitions.md)).
