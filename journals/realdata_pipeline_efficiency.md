# Real-data pipeline: insertion cache, stream pooling, and ORCD sharding

**Scope:** making `block_analysis_allsplits.py` time- and storage-efficient over the full BWM dataset — the per-insertion cache, the loop reorder, streaming pooled accumulation, insertion sharding for Slurm, and the memory/trial-count settings that go with them.

**Status:** implemented, validated numerically (including `nrand=2000` parity), and in production on ORCD. Multi-split runs are ~4× faster on the load-dominated path; adding new splits once the cache exists is nearly free.

Sources: dated entries 2026-07-06 (Goal 2), 07-06c, 07-09, 07-10, 07-10b, 07-12b, 07-12c, 07-12d, 07-12g.

---

## Goal

The old flow: `get_all_d_vars(split)` loops all insertions; `get_d_vars(split, pid)` per insertion **re-loads spikes (`load_good_units`) and trials (`load_trials_and_mask`) from scratch for every split**. With N splits × M insertions this is N× redundant loading of the expensive spike/trial data.

**Plan — two-stage pipeline:**

- **Stage 1 (once per insertion):** load spikes (`times`, `clusters`), clusters (`cluster_id`, `atlas_id` → acronyms), and the trials table + mask; save a compact per-insertion cache (`insertion_cache/{pid}.*`). The mask depends on the alignment event (stim/choice/fback → different `saturation_intervals`), so cache per event type (≤3).
- **Stage 2 (per split, fast):** load the cached insertion, do event/trial selection + `bin_spikes2D` + trial averaging + `d_var`/`d_euc`/crossnobis + nrand shuffles. No spike/trial reload.

This makes adding or redoing splits cheap and directly enables the contrast-stratified analyses in [prior modulation by contrast](prior_modulation_by_contrast.md).

---

## Current stack

| Piece | Status / location |
|-------|-------------------|
| Driver | `block_analysis_allsplits.py` |
| Insertion cache | `manifold/insertion_cache/{eid_probe}.npy` (spikes + trials once per pid) |
| Multi-split | outer loop = insertions, inner = splits; `stream_pool` → `manifold/res/` |
| Contrast splits | `'{base}_{contrast}'` on duringstim / duringchoice (act + non-act), incl. 0 % |
| Min trials | ≥5 per split side (`InsufficientTrials` → skip) |
| Sharding | `scripts/submit_goal2_*` / `submit_goal3_sharded.sh` (ORCD `mit_normal`) |
| Nulls | label shuffle within condition (contrast-matched where applicable); choice L–R has structured options — see [structured nulls](structured_nulls_choice_lr.md) |

---

## Refactor (2026-07-06c)

- `saturation_for_split(split)` — the stim/move/feedback saturation key.
- `build_insertion_cache(pid)` — loads an insertion's raw data **once** (spikes `times`/`clusters`, clusters `cluster_id`/`atlas_id`, and bad-trial-masked trials for each of the 3 saturation types) and persists to `manifold/insertion_cache/{eid_probe}.npy`.
- `get_d_vars(..., cached=None)` — reuses spikes/clusters/trials when a cache is passed; else loads as before. Downstream binning/averaging/`d_var`/xnobis code unchanged → identical output.
- `get_all_d_vars_allsplits(splits_list)` — **loop reorder**: outer loop = insertions (load once), inner loop = all splits. Same per-split output layout `manifold/{split}/{eid_probe}.npy`. `restart` skips already-computed (split, insertion).
- `cache_all_insertions()` — pre-build all per-insertion caches.
- `__main__` guarded (`if __name__ == '__main__':`) so the module is importable; pooling (`d_var_stacked` / `d_var_stacked_multi`) unchanged.

**Initial validation was blocked** by a pre-existing environment issue: vendored `brainwidemap/bwm_loading.py:285` calls `SessionLoader(..., revision=MODIFIED_BEFORE)` but the installed `one-api` `SessionLoader` had no `revision` kwarg → `load_trials_and_mask` raised `TypeError`. The original `get_d_vars` hit this identically, so it was unrelated to the refactor.

---

## Validation and benchmarks

### 2026-07-09 — validated locally (ibllib 4.0.1 / ONE-api 3.5.2)

**Env fix:** upgraded `ibllib` 2.38 → 4.0.1 (adds `SessionLoader(..., revision=)`). Also patched `str(eid)` when filtering the aggregate trials table (ONE-api 3.x returns `uuid.UUID` from `pid2eid`).

**Parity** (`scripts/validate_goal2_cache.py`): 5 insertions × 4 splits, cached vs uncached, `nrand=20`, with null shuffles — **all passed** (~4.6 min). Cache trials-table check OK on 3 insertions.

**Speed** (`scripts/benchmark_goal2_10splits.py`, 5 insertions, 10 splits, `control=False`):

| insertion | old (10 reloads) | new (1 load + 10 splits) | speedup |
|-----------|-----------------:|-------------------------:|--------:|
| 1 | 47 s | 10.5 s | 4.5× |
| 2 | 51 s | 12.8 s | 4.0× |
| 3 | 51 s | 11.8 s | 4.5× |
| 4 | 68 s | 16.9 s | 4.1× |
| 5 | 86 s | 27.6 s | 3.1× |
| **mean** | **61 s** | **15.9 s** | **3.8×** |

Extrapolated to full BWM (699 insertions, 10 splits): **~12 h → ~3 h** wall time. The load component drops from ~55 s to ~5 s per insertion (~90 % of redundant I/O removed).

**Storage:** one insertion cache ≈ **30 MB**; full BWM cache ≈ **22 GB** one-time. Per-split outputs unchanged. Adding splits 11–20 is nearly free once the cache exists (`restart=True` skips existing outputs).

**Caveat:** the benchmark used `control=False` (no 2000-shuffle null). With `control=True` the per-split compute is identical in old and new paths, so the speedup on production runs is **load-dominated**.

### 2026-07-10 — null-loop fix + nrand=2000 parity

**Bug:** the batched-null refactor briefly used a **region-outer × null-inner** loop order, recomputing `b[ys].mean()` per region per null (~60 regions × 2000 nulls). That inflated one `block_only` call to **~147 s** at nrand=2000.

**Fix:** restored **null-outer × region-inner** in `_compute_control_D()` / `_append_perm()`. One full-tensor mean/var per null, then regional metrics.

**Parity at nrand=2000** (5 pids × 4 splits): all 20 pairs passed (~71 min wall); cached vs uncached match bit-for-bit with seed reset.

**Timing (1 pid, `block_only`):** uncached ~64 s; cached ~61 s after an 8 s load. Scaling is ~linear in nrand: 100 → 3.8 s, 500 → 15.7 s, 2000 → 60 s. No regression versus pre-refactor.

**3-split bench** (`scripts/test_goal2_nrand2000.py`): uncached 155 s, cached 149 s (load 8 s) → 1.04× speedup. At nrand=2000 the null loop (~50 s/split) dominates and the cache only saves reload I/O (~8 s once). The multi-split win is load-dominated when `control=False`; with `control=True` it is smaller per insertion but still ~8 s × (N_splits − 1).

### 2026-07-10b — end-to-end comparison on the alyx cache

**Test ONE root:** `https://alyx.internationalbrainlab.org` → `/Users/ariliu/Downloads/ONE/alyx.internationalbrainlab.org` (chosen for an apples-to-apples comparison with the pre-refactor baseline).

**Scripts:** `scripts/_original_pipeline_worker.py` (isolated ee849e0 pipeline), `scripts/compare_alyx_pipeline.py` (new vs original + 2-split aggregate).

**Storage test** (stream_pool, openalyx reference, 5 insertions, 1 split):

| metric | value |
|--------|------:|
| Wall time | 404.5 s (~6.7 min) |
| Peak RSS | 2429.8 MB |
| Insertion cache | 322.4 MB (5 files) |
| Stream acc checkpoint | 135.0 MB |
| Final res | 24.1 MB |
| **Total disk** | **481.5 MB** |
| Per-split insertion files | **0** (stream_pool skips `manifold/{split}/*.npy`) |

**Original baseline** (`block_only`, ee849e0, uncached): wall 415.1 s; per-insertion mean 82.8 s; `d_var_stacked` 0.7 s; peak RSS 3735.4 MB; per-insertion files 231.2 MB; final res 24.1 MB.

**Split 1 — `block_only`, original vs new** (final outputs match: 10 regions OK):

| | Original | New |
|--|--:|--:|
| Wall time | 415.1 s | 488.8 s |
| Peak RSS | 3735 MB | 2423 MB |
| Disk (intermediates + final) | 255 MB | 482 MB |

The new path is **slower on split 1** because it pays the one-time insertion-cache build, but uses **~1.3 GB less peak RAM** (no materialised `ws` tensor for 2000 nulls).

**Split 2 — `block_duringstim_l_choice_l_f1`, cache reused** (outputs match: 10 regions OK):

| | Original | New |
|--|--:|--:|
| Wall time | 283.9 s | **75.8 s** |
| Per-insertion mean | 56.6 s | **15.1 s** |
| Cache load mean | (full reload) | **0.0 s** |
| Peak RSS | 2972 MB | 1457 MB |
| Per-insertion files | 120.6 MB | 0 (stream_pool) |

**Two-split aggregate:**

| metric | Original | New (stream_pool) |
|--------|----------:|------------------:|
| **Total wall time** | **699 s (11.7 min)** | **565 s (9.4 min)** → **1.24× faster** |
| Peak RSS (max of runs) | 3735 MB | 2422 MB |
| Total disk | 388 MB | 561 MB |
| Per-split insertion files | 10 (5×2) | 5 (split 1 original only) |

**Stream acc checkpoints** (`manifold/res/_stream_acc/{split}.npy`): incremental pool state saved after each insertion, replacing per-insertion `manifold/{split}/*.npy` plus a separate `d_var_stacked` pass. Contents: `pooled_keys`, `acs`/`acs1`, `ws`, `regdv0`/`regde0`, `uperms`. `finalize()` writes `manifold/res/{split}*.npy`.

| split | per-ins files | stream acc | ratio |
|-------|-------------:|-----------:|------:|
| `block_only` | 231 MB | 135 MB | 0.58× |
| `block_duringstim_l_choice_l_f1` | 121 MB | 68 MB | 0.56× |

**Storage break-even (full BWM, 699 insertions)** with intermediates kept: `N × (P − S) = C`, where `C` = cache bytes/insertion, `P` = per-insertion file bytes/insertion/split, `S` = stream_acc bytes/insertion/split.

| cache estimate | avg P, S from test | **N_splits to break even** |
|----------------|-------------------|-----------------------------|
| ~30 MB/ins (~22 GB total) | 35 / 20 MB | **~2 splits** |
| ~64.5 MB/ins (alyx test avg) | 35 / 20 MB | **~4 splits** |
| 64.5 MB/ins, `block_only`-like only | 46 / 27 MB | **~3 splits** |
| 64.5 MB/ins, small splits only | 24 / 14 MB | **~6 splits** |

If stream acc is deleted after `finalize()`, persistent new storage ≈ cache + final res → break-even **~1–2 splits**.

---

## Insertion sharding for ORCD (2026-07-12b)

**Problem:** full BWM (~699 insertions) × nrand=2000 for one split is ~19 h wall (~100 s/insertion) — beyond the `mit_normal` 12 h limit. Parallelizing across splits alone was not enough; jobs timed out mid-split (e.g. `act_block_stim_r` at ~463/699). A kill mid-`np.save` also left truncated `_stream_acc/{split}.npy` (`pickle data was truncated` on restart).

**Fixes / features:**

1. **Atomic stream_acc save** — write `.{split}.tmp.{pid}.npy` then `os.replace`; a corrupt load quarantines to `*.corrupt.{pid}` and starts empty.
2. **Delete stream_acc after successful finalize** — once `manifold/res/{split}.npy` exists.
3. **Insertion sharding** — `shard_idx` / `n_shards` on `get_all_d_vars_allsplits`: each job processes `eids_plus[k::N]` and writes `_stream_acc/{split}.shard{k}.npy` (no finalize). `finalize_stream_shards(split)` merges disjoint shards (plus any leftover unsharded `{split}.npy`) → `res/{split}*.npy`, then cleans checkpoints. CLI: `--shard-idx` / `--n-shards` / `--finalize-only` / `--no-finalize`.

**Scripts (promoted to `main` for the cluster):**

- `scripts/run_goal2_shard_slurm.sh` — one shard
- `scripts/run_goal2_finalize_slurm.sh` — merge + finalize
- `scripts/submit_goal2_stimOn_act_sharded.sh` — default **N_SHARDS=4** × 6 stimOn_act splits + finalize dependencies
- `scripts/run_goal2_cache_slurm.sh` — cache-only (restart skips existing)

**Expected timing:** 4 shards → ~5 h/shard at ~100 s/insertion (fits 12 h).

Do **not** mix sharding with an existing good unsharded `{split}.npy` restart (duplicate keys). Continue timed-out splits unsharded with `restart=True`, or delete corrupt checkpoints and shard from scratch.

```bash
# New / full redo (cache already built):
bash scripts/submit_goal2_stimOn_act_sharded.sh
# N_SHARDS=6 bash scripts/submit_goal2_stimOn_act_sharded.sh
```

---

## Trial-count gate: `min_trials_per_side = 5` (2026-07-12c)

Both sides of a split must have ≥5 trials; otherwise `get_d_vars` raises `InsufficientTrials` and the cached driver logs `split skip` (no stream_acc, no per-insertion save). This replaced an older assert that only rejected zero-trial sides.

**Smoke (alyx insertion cache, 2 pids × 5 contrast splits, nrand=5):** ok=2, skip=8, fail=0. Examples: `…_f1_1.0` (13/7, 10/10) ran; `…_f1_0.125` (12/2), `…_f1_0.0` (4/1), `…_f2_1.0` (0/0) skipped.

**Consequence for cross-cache comparisons:** openalyx `manifold/res/*.npy` were finalized **before** this gate, so openalyx label-shuffle baselines can differ slightly from alyx runs (insertions with <5 trials on a side were still included). See the audit note in [structured nulls](structured_nulls_choice_lr.md#openalyx-vs-alyx-min_trials_per_side-gap).

---

## Slurm memory settings

**Evidence (peak RSS, stream_pool, nrand=2000):** ~1.5–2.5 GB per shard. Contrast splits with `min_trials_per_side=5` skip many insertions → smaller stream_acc still.

**2026-07-12d — Goal 3 / Goal 2 shard defaults.** Problem: shard workers requested 48 G each; the Goal-3 default submit (20 splits × 4 shards = 80 jobs) requested ~3.8 TB concurrently → hit the per-user memory limit → pending.

| job | was | now |
|-----|-----|-----|
| shard (`run_goal2_shard_slurm.sh`) | 48G / 4 cpus | **12G** / 2 cpus |
| finalize | 32G | **16G** / 2 cpus |
| Goal 3 submit override | (inherit 48G) | **MEM_SHARD=8G**, **MEM_FIN=12G** |
| Goal 2 submit override | (inherit) | **MEM_SHARD=12G**, **MEM_FIN=16G** |
| unsharded `run_goal3_contrast_slurm.sh` | 48G | **16G** |

Concurrent Goal-3 default: 80 × 8G = **640 G** (~6× less). Override: `MEM_SHARD=6G MEM_FIN=10G bash scripts/submit_goal3_sharded.sh`.

**2026-07-12g — Goal 2 sharded submit (8 splits).** Finalize merges 4 shard accumulators, so it needs modestly more than one shard.

| | was | now |
|--|-----|-----|
| MEM_SHARD | 12G | **6G** |
| MEM_FIN | 16G | **10G** |
| Concurrent (8 × 4 shards) | 384 G | **192 G** |

Override if OOM: `MEM_SHARD=8G MEM_FIN=12G bash scripts/submit_goal2_stimOn_act_sharded.sh`.
