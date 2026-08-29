#!/bin/bash
# ITI prior L–R (stimOn [−400, −100] ms) for true-block, act, and Bayes.
#
# Labels:
#   block_only       — probabilityLeft[t-1] (previous trial's true block)
#   act_block_only   — action kernel through action[t-1]
#   bayes_block_only — Bayes P(left on t | stims 1..t-1)
#
#   bash scripts/submit_goal2_iti_prior_orcd.sh
#   FAMILY=block|act|bayes|all
#   NULL=both|default|harris
#
# default = plain {split}.npy (generate_pseudo_blocks for *block_only*)
# harris  = {split}_harris_unique.npy  (MEM_SHARD=24G; ITI Harris OOMs at 6G)
#
# Sharding: full-session ITI (no stratum), ~700 probes, nrand=2000.
# Harris was ~1–3 min/insertion (08-18 restart). 12 shards → ~58
# insertions/shard → ~3 h worst-case, TIME_SHARD=5:00:00.
# Bump N_SHARDS=16 if a shard still times out.
#
# Run on ORCD. Do not submit from the laptop.
#
#   FAMILY=all NULL=both bash scripts/submit_goal2_iti_prior_orcd.sh
#   FAMILY=block NULL=default bash scripts/submit_goal2_iti_prior_orcd.sh
#   CLEAR_STREAM=0 if re-running this wrapper after a crash
#   PARTITION=mit_preemptable bash scripts/submit_goal2_iti_prior_orcd.sh
#     --requeue is the default on mit_preemptable / mit_preem (see sbatch_defaults.sh)
#
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

FAMILY="${FAMILY:-all}"
NULL="${NULL:-both}"
NRAND="${NRAND:-2000}"
# 12 shards: same as fully-unsplit full-session jobs; 4 shards was ~6–9 h
# for ITI Harris and missed a 5 h cap.
N_SHARDS="${N_SHARDS:-12}"
TIME_SHARD="${TIME_SHARD:-5:00:00}"
PARTITION="${PARTITION:-pi_fiete}"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/sbatch_defaults.sh"

CLEAR_STREAM="${CLEAR_STREAM:-1}"
RESTART="${RESTART:-1}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"
export NRAND N_SHARDS TIME_SHARD PARTITION CLEAR_STREAM RESTART

if [[ "$FAMILY" != "block" && "$FAMILY" != "act" && "$FAMILY" != "bayes" && "$FAMILY" != "all" ]]; then
  echo "ERROR: FAMILY must be block|act|bayes|all (got $FAMILY)" >&2
  exit 1
fi
if [[ "$NULL" != "both" && "$NULL" != "default" && "$NULL" != "harris" ]]; then
  echo "ERROR: NULL must be both|default|harris (got $NULL)" >&2
  exit 1
fi

PRESETS=()
if [[ "$FAMILY" == "block" || "$FAMILY" == "all" ]]; then
  PRESETS+=("block_only")
fi
if [[ "$FAMILY" == "act" || "$FAMILY" == "all" ]]; then
  PRESETS+=("act_block_only")
fi
if [[ "$FAMILY" == "bayes" || "$FAMILY" == "all" ]]; then
  PRESETS+=("bayes_block_only")
fi

echo "ITI prior  FAMILY=$FAMILY  NULL=$NULL  presets: ${PRESETS[*]}"
echo "CLEAR_STREAM=$CLEAR_STREAM  nrand=$NRAND  N_SHARDS=$N_SHARDS  TIME_SHARD=$TIME_SHARD  PARTITION=$PARTITION  SBATCH_EXTRA=${SBATCH_EXTRA:-}"
if [[ "$FAMILY" == "all" || "$FAMILY" == "act" ]]; then
  echo "NOTE: act_block_only CLEAR will replace existing alyx {split}.npy / _harris_unique.npy"
fi

_default() {
  local preset="$1"
  echo "=== default (pseudo-blocks)  PRESET=$preset ==="
  # Full-session ITI is heavier than choice L–R; 16G default.
  PRESET="$preset" \
    SESSION_SHUFFLE_NULL=0 ACTKERNEL_CHOICE_NULL=0 \
    N_SHARDS="$N_SHARDS" TIME_SHARD="$TIME_SHARD" \
    MEM_SHARD="${MEM_SHARD_DEFAULT:-16G}" MEM_FIN="${MEM_FIN_DEFAULT:-16G}" \
    bash scripts/submit_goal2_choice_shuffle_sharded.sh
}

_harris() {
  local preset="$1"
  echo "=== harris unique  PRESET=$preset ==="
  PRESET="$preset" \
    N_SHARDS="$N_SHARDS" TIME_SHARD="$TIME_SHARD" \
    MEM_SHARD="${MEM_SHARD_HARRIS:-24G}" MEM_FIN="${MEM_FIN_HARRIS:-32G}" \
    JOB_PREFIX="${JOB_PREFIX:-g2iti}" \
    bash scripts/submit_goal2_act_block_harris_sharded.sh
}

for preset in "${PRESETS[@]}"; do
  if [[ "$NULL" == "default" || "$NULL" == "both" ]]; then
    _default "$preset"
  fi
  if [[ "$NULL" == "harris" || "$NULL" == "both" ]]; then
    _harris "$preset"
  fi
done

echo "Outputs:"
echo "  default: \$ONE_CACHE_DIR/manifold/res/{split}.npy"
echo "  harris:  \$ONE_CACHE_DIR/manifold/res/{split}_harris_unique.npy"
echo "Monitor: squeue -u \$USER"
