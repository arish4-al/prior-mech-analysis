# ITI prior (stimOn [−400, −100] ms)

**Scope:** prior L–R in the pre-stimulus window for the three `*_block_only` splits — no stim/choice/f1/f2 stratum. Label conventions, default vs Harris results after the 08-28 true-block lag, and the unstratified `_pseudosession` null (one generative process per prior type).

**Status:** labels, default/Harris (08-31), and matched `_pseudosession` (09-01) are scored. Default `{split}.npy` is **not** a label shuffle — it is `generate_pseudo_blocks` 0.8/0.2 (true-block-like); fair only for `block_only`.

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

**“Shuffle” in earlier notes / `{split}.npy` is not a label shuffle.** It is IBL `generate_pseudo_blocks` (0.8/0.2 runs, `first5050=0`; true-block uses `ntr+1` then `[1:]`). Same null labels are written for true / act / Bayes. That is a **true-block pseudo-session**, so it is fair **only** for `block_only`. For act and Bayes it is an unmatched 0.8/0.2 control (too blocky vs AK / Bayes-from-stim). Call it **pseudo-blocks** below.

| Null | Suffix | Fair for | What the labels are |
|------|--------|----------|---------------------|
| pseudo-blocks (default; often called shuffle) | `{split}.npy` | **true block only** | `generate_pseudo_blocks` 0.8/0.2 for **all three** names. **Not** AK-prior or Bayes-from-stim. |
| Harris unique | `{split}_harris_unique.npy` | all three | Other eids' observed priors (true-block lagged; act/Bayes recomputed on the donor) |
| **pseudosession** | `{split}_pseudosession.npy` | **all three** | Full IBL-like pseudo-session, then the **same** prior definition as the observation: lagged pLeft / AK of synthetic choices / Bayes of the stim sequence. No remade stratum, not `fixedstim`, not Harris. Only act generates choices through the fitted model. |

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

**Default column is pseudo-blocks, not a shuffle** (fair only for true block). Matched `_pseudosession` is 09-01 below.

| Prior | Null | raw ≤0.01 | FDR @0.01 | raw ≤0.05 | FDR @0.05 | median p |
|-------|------|----------:|----------:|----------:|----------:|---------:|
| true block | pseudo-blocks | 0 | **0** | 1 | **0** | 0.88 |
| true block | Harris | 2 | **0** | 12 | **0** | 0.53 |
| act | pseudo-blocks | 9 | **0** | 14 | **7** | 0.67 |
| act | Harris | 7 | **0** | 27 | **0** | 0.32 |
| Bayes | pseudo-blocks | 0 | **0** | 0 | **0** | 0.91 |
| Bayes | Harris | 2 | **0** | 9 | **0** | 0.47 |

The only FDR map on this **unmatched** act null is **act + pseudo-blocks @0.05**: AIp, CLA, FOTU, LSr, PIR, SSp-n, VISa. Harris wipes it. True-block and Bayes are complete nulls on both arms. The act 7-hit map does **not** survive the fair AK `_pseudosession` (09-01).

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

### 2026-09-01 — `act_block_only` table @ FDR 0.05 (pseudo-blocks, unmatched)

Single-split BH-FDR on `p_mean` from `act_block_only_regde.npy` (**pseudo-blocks**, not a shuffle). Cell = min–max `amp_euc` if `p_mean_c` ≤ 0.05, else blank.

```bash
python scripts/plot_iti_prior_table.py --split act_block_only --alpha 0.05 --tag shuffle
```

Alyx `meta/table_act_block_only_shuffle_p_mean_c_0.05.{png,csv}`. **7** / 208: AIp, CLA, FOTU, LSr, PIR, SSp-n, VISa. **Unfair for act** — 0.8/0.2 pseudo-blocks, not AK labels. Fair `_pseudosession` wipes this map (next section).

### 2026-09-01 — all-region population (`*_all.npy`)

Files in `res/new/` (`{split}_all.npy` / `_all_regde.npy`). Literal neuron-weighted pool **before** RMS and min_reg (266 regions, ~64k cells; regional table keeps 208). Single test — no BH-FDR.

`p_mean` / `p_amp` from `_all_regde` (`mean(mean(stack,1) ≥ mean(obs))` and `mean(ptp(stack,1) ≥ ptp(obs))`). `p_euc` / `amp_euc` from `_all.npy`. Here `p_amp` equals stored `p_euc`.

**Pseudo-blocks ≠ shuffle.** Fair matched null is `_pseudosession` (mtime 1 Sep 10:14).

| Prior | Null | nclus | n_null | p_mean | p_amp | p_euc | amp |
|-------|------|------:|-------:|-------:|------:|------:|----:|
| true | pseudo-blocks | 64,045 | 2000 | **1.00** | 0.81 | 0.81 | 0.012 |
| true | Harris | 63,848 | 352 | 0.98 | 0.60 | 0.60 | 0.011 |
| true | **pseudosession** | 64,045 | 2000 | **1.00** | 0.82 | 0.82 | 0.012 |
| act | pseudo-blocks (unfair) | 64,050 | 2000 | **1.00** | 0.53 | 0.53 | 0.011 |
| act | Harris | 63,853 | 352 | 0.11 | 0.39 | 0.39 | 0.011 |
| **act** | **pseudosession** | 64,050 | 2000 | **1.00** | 0.46 | 0.46 | 0.011 |
| Bayes | pseudo-blocks (unfair) | 64,050 | 2000 | **1.00** | 0.59 | 0.59 | 0.011 |
| Bayes | Harris | 63,853 | 352 | 0.99 | 0.29 | 0.29 | 0.010 |
| Bayes | **pseudosession** | 64,050 | 2000 | **1.00** | 0.50 | 0.50 | 0.011 |

All three fair `_pseudosession` pools are a complete mean-null (`p_mean=1`). Amplitude sits near the middle of the matched null (`p_amp` 0.46–0.82), unlike the inflated pseudo-block null for Bayes (`p_amp` 0.59 vs fair 0.50). CSV: `meta/iti_block_only_all_region_summary.csv`. Curve: `meta/act_block_only_shuffle_all_region_distance.png`.

### 2026-09-01 — matched `_pseudosession` FDR (fair for act / Bayes)

Alyx `res/new/`, mtime 1 Sep 10:14. Same scoring as 08-31 (BH on `p_mean` from `*_regde`). 208 regions, `n_null=2000`. True-block `_pseudosession` ≈ pseudo-blocks (same generative family). Act/Bayes now get AK / Bayes-from-stim labels on the pseudo-session.

| Prior | Null | raw ≤0.01 | FDR @0.01 | raw ≤0.05 | FDR @0.05 | median p |
|-------|------|----------:|----------:|----------:|----------:|---------:|
| true block | pseudo-blocks | 0 | **0** | 1 | **0** | 0.88 |
| true block | Harris | 2 | 0 | 12 | **0** | 0.53 |
| act | pseudo-blocks (unfair) | 9 | **0** | 14 | **7** | 0.67 |
| act | Harris | 7 | 0 | 27 | **0** | 0.32 |
| act | pseudosession-act | 2 | **0** | 9 | **0** | 0.65 |
| Bayes | pseudo-blocks (unfair) | 0 | **0** | 0 | **0** | 0.91 |
| Bayes | Harris | 2 | **0** | 9 | **0** | 0.47 |
| Bayes | pseudosession-bayes | 0 | **0** | 1 | **0** | 0.83 |

**Act @0.05: 7 → 0.** The 7-region map (AIp, CLA, FOTU, LSr, PIR, SSp-n, VISa) was an unmatched 0.8/0.2 null. Fair AK `_pseudosession` leaves **0** FDR @0.01 and @0.05. Raw ≤0.01: LSr (p=0.0025, p_c=0.52), PL (p=0.0055, p_c=0.57). Raw ≤0.05 also FOTU, VPMpc, SSp-n, CP, CLA, PIR, MOp — none survive BH. AIp and VISa drop out of even the raw-0.05 list.

True-block `_pseudosession` matches default (0 FDR; median p 0.884 vs 0.878; raw ≤0.05 is MOs). Bayes stays 0 FDR; one raw ≤0.05 (VPMpc). Harris remains the most liberal median p and still 0 FDR.

**Why only act + pseudo-blocks survive FDR.** Same BH as BWM (`statsmodels.multipletests`, `method='fdr_bh'`), one test per region, m=208, **not** combined across splits (ITI is a single split). `p_mean` = fraction of curves (observed + nulls) whose mean ≥ observed mean, so the discrete floor is `1/(1+n_null)` = **1/2001 ≈ 0.0005** when `n_null=2000`. BH @α rejects the largest k with `p_(k) ≤ α·k/m`. With m=208 that threshold at k=1 is 0.00024, **below the floor**, so a singleton at 0.0005 cannot reject. Need **≥3 regions at the floor** before anything can pass @0.05, and **≥11** @0.01. Only act + pseudo-blocks has that cluster (**5** at 0.0005, then VISa 0.0010, FOTU 0.0015 → k*=7, `p_c` 0.021–0.045). Fair act min p is 0.0025 (LSr; five nulls beat observed) → `p_c` = 0.0025·208 = **0.52**. Harris act has 27 raw ≤0.05 but **0** at the floor (min p 0.0010) so `p_c` ≥ 0.10. Every arm is 0 FDR @0.01 because none have 11 floor hits.

Regional CSV: `meta/iti_block_only_regional_fdr_summary.csv`.

---

## 2026-08-31 — unstratified `_pseudosession` (wired; scored 09-01)

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

1. Fixed α vs per-session AK fit for the **observed** act labels — still open ([prior definitions](prior_definitions.md)).
