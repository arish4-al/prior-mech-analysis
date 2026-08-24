#!/bin/bash
# Fully unsplit act-prior L–R: one split per timeframe, no stim/choice/f1/f2
# stratum in the real data or the null.
#
# Splits:
#   act_block_duringstim_fully_unsplit   — stimOn, [0, 150] ms, all biased trials
#   act_block_duringchoice_fully_unsplit — firstMovement, [150, 0] ms
#
# Nulls (both, default):
#   Harris unique-null  → {split}_harris_unique.npy
#   AK option-1 + copy-last → {split}_pseudo_strat_sticky.npy
#
# More shards than the f1/f2 campaign: every insertion uses the full biased
# session (like act_block_only), so per-shard work is heavier. Default
# N_SHARDS=12 (vs 4). Bump further with N_SHARDS=16 if 8 h is tight.
#
#   bash scripts/submit_goal2_act_block_fully_unsplit_orcd.sh
#   FAMILY=harris|ak|both  (default both)
#   PREFIT=1  — also submit ActionKernel prefit (default 0; fits already cached)
#   REBUILD_DONORS=1  — rebuild Harris donor bank (default 0; bank exists)
#   N_SHARDS=16 TIME_SHARD=8:00:00 PARTITION=mit_preemptable \
#     bash scripts/submit_goal2_act_block_fully_unsplit_orcd.sh
#
# Journal 2026-08-24d. Not stim-side unsplit (act_block_duringstim_{l,r}).
#
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

FAMILY="${FAMILY:-both}"
NRAND="${NRAND:-2000}"
# 12 shards: full-session n_elig (no stratum). Override up if jobs time out.
N_SHARDS="${N_SHARDS:-12}"
TIME_SHARD="${TIME_SHARD:-8:00:00}"
PARTITION="${PARTITION:-pi_fiete}"
CLEAR_STREAM="${CLEAR_STREAM:-1}"
RESTART="${RESTART:-0}"
PREFIT="${PREFIT:-0}"
REBUILD_DONORS="${REBUILD_DONORS:-0}"
PSEUDO_LEN_FACTOR="${PSEUDO_LEN_FACTOR:-}"
# Harris unique on the full biased session was 24G for ITI act_block_only;
# 150 ms windows are lighter, 16G is the starting point (bump if OOM).
MEM_SHARD="${MEM_SHARD:-16G}"
MEM_FIN="${MEM_FIN:-16G}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"
export NRAND N_SHARDS TIME_SHARD PARTITION CLEAR_STREAM RESTART
export PSEUDO_LEN_FACTOR MEM_SHARD MEM_FIN REBUILD_DONORS

if [[ "$FAMILY" != "harris" && "$FAMILY" != "ak" && "$FAMILY" != "both" ]]; then
  echo "ERROR: FAMILY must be harris|ak|both (got $FAMILY)" >&2
  exit 1
fi

echo "=== Fully unsplit act-prior (no stim/choice/f1/f2 stratum) ==="
echo "FAMILY=$FAMILY  N_SHARDS=$N_SHARDS  TIME_SHARD=$TIME_SHARD  nrand=$NRAND"
echo "CLEAR_STREAM=$CLEAR_STREAM  RESTART=$RESTART  PARTITION=$PARTITION"
echo "MEM_SHARD=$MEM_SHARD  MEM_FIN=$MEM_FIN"
echo "Splits: act_block_duringstim_fully_unsplit"
echo "        act_block_duringchoice_fully_unsplit"

PREFIT_JID="${PREFIT_JID:-}"
if [[ "$FAMILY" == "ak" || "$FAMILY" == "both" ]]; then
  if [[ "$PREFIT" == "1" ]]; then
    echo "=== Prefit ActionKernel (once per eid) ==="
    PREFIT_JID=$(sbatch --parsable \
      --partition="$PARTITION" \
      --job-name=g2_ak_prefit \
      --mem="${MEM_PREFIT:-8G}" --cpus-per-task=2 --time="${TIME_PREFIT:-8:00:00}" \
      --export=ALL \
      scripts/run_goal2_ak_prefit_slurm.sh)
    echo "Prefit -> $PREFIT_JID"
  fi
  export PREFIT_JID
  export NULL_SCHEME=pseudo_strat_sticky
  echo "=== AK option-1 + copy-last ==="
  PRESET=act_block_ak_sticky_fully_unsplit \
    JOB_PREFIX="${JOB_PREFIX_AK:-g2afus}" \
    bash scripts/submit_goal2_choice_null_sharded.sh
fi

if [[ "$FAMILY" == "harris" || "$FAMILY" == "both" ]]; then
  echo "=== Harris unique-null ==="
  PRESET=act_block_harris_fully_unsplit \
    JOB_PREFIX="${JOB_PREFIX_HU:-g2afuh}" \
    bash scripts/submit_goal2_act_block_harris_sharded.sh
fi

echo "Outputs:"
echo "  \$ONE_CACHE_DIR/manifold/res/{split}_harris_unique.npy"
echo "  \$ONE_CACHE_DIR/manifold/res/{split}_pseudo_strat_sticky.npy"
echo "Monitor: squeue -u \$USER"
