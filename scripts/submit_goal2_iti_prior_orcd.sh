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
#   NULL=both|default|harris|pseudo|all
#
# default = plain {split}.npy (generate_pseudo_blocks for *block_only*)
# harris  = {split}_harris_unique.npy  (MEM_SHARD=24G; ITI Harris OOMs at 6G)
# pseudo  = {split}_pseudosession.npy  (unstratified; no remade stratum)
#           block_only       — lagged pseudo-session probabilityLeft
#           act_block_only   — fitted AK choices → action-kernel prior
#           bayes_block_only — Bayes prior from pseudo stim sequence
#                              (no choice model)
#
# Sharding: full-session ITI (no stratum), ~700 probes, nrand=2000.
# 12 shards (~58/shard) missed 5 h on default act (worst 37/58) and
# nearly on Harris act/Bayes. 24 shards → ~29 insertions/shard.
# At the 08-31 worst cheap rate (~8 min/ins) that is ~4 h, with headroom
# for AK simulate on act_block_only. TIME_SHARD=5:00:00.
#
# Run on ORCD. Do not submit from the laptop.
#
#   FAMILY=all NULL=pseudo bash scripts/submit_goal2_iti_prior_orcd.sh
#   FAMILY=block NULL=default bash scripts/submit_goal2_iti_prior_orcd.sh
#   PREFIT=1 if act_block_only AK pickles are missing
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
# 24 shards: 12 missed the 5 h cap on default act (37–46/58).
N_SHARDS="${N_SHARDS:-24}"
TIME_SHARD="${TIME_SHARD:-5:00:00}"
PARTITION="${PARTITION:-pi_fiete}"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/sbatch_defaults.sh"

CLEAR_STREAM="${CLEAR_STREAM:-1}"
RESTART="${RESTART:-1}"
PREFIT="${PREFIT:-0}"
# Finalize of ~700 insertions died at the worker's 2 h cap (08-31).
TIME_FIN="${TIME_FIN:-5:00:00}"
TIME_SHARD_PSEUDO="${TIME_SHARD_PSEUDO:-$TIME_SHARD}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"
export NRAND N_SHARDS TIME_SHARD PARTITION CLEAR_STREAM RESTART TIME_FIN

if [[ "$FAMILY" != "block" && "$FAMILY" != "act" && "$FAMILY" != "bayes" && "$FAMILY" != "all" ]]; then
  echo "ERROR: FAMILY must be block|act|bayes|all (got $FAMILY)" >&2
  exit 1
fi
if [[ "$NULL" != "both" && "$NULL" != "default" && "$NULL" != "harris" && "$NULL" != "pseudo" && "$NULL" != "all" ]]; then
  echo "ERROR: NULL must be both|default|harris|pseudo|all (got $NULL)" >&2
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
if [[ "$NULL" == "default" || "$NULL" == "harris" || "$NULL" == "both" || "$NULL" == "all" ]]; then
  if [[ "$FAMILY" == "all" || "$FAMILY" == "act" ]]; then
    echo "NOTE: act_block_only CLEAR will replace existing alyx {split}.npy / _harris_unique.npy"
  fi
fi
if [[ "$NULL" == "pseudo" || "$NULL" == "all" ]]; then
  echo "NOTE: NULL=pseudo writes {split}_pseudosession.npy only (does not touch plain / harris)"
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
    JOB_PREFIX=g2iti \
    bash scripts/submit_goal2_act_block_harris_sharded.sh
}

_pseudo() {
  local preset="$1"
  echo "=== unstratified pseudosession  PRESET=$preset ==="
  PRESET="$preset" \
    NULL_SCHEME=pseudosession \
    N_SHARDS="$N_SHARDS" TIME_SHARD="${TIME_SHARD_PSEUDO}" \
    TIME_FIN="$TIME_FIN" \
    MEM_SHARD="${MEM_SHARD_PSEUDO:-24G}" MEM_FIN="${MEM_FIN_PSEUDO:-16G}" \
    JOB_PREFIX=g2itp \
    PREFIT_JID="${PREFIT_JID:-}" \
    bash scripts/submit_goal2_choice_null_sharded.sh
}

if [[ "$NULL" == "pseudo" || "$NULL" == "all" ]]; then
  if [[ "$PREFIT" == "1" && ( "$FAMILY" == "act" || "$FAMILY" == "all" ) ]]; then
    echo "=== Prefit ActionKernel (once per eid; act_block_only only) ==="
    # shellcheck disable=SC2086
    PREFIT_JID=$(sbatch --parsable $SBATCH_EXTRA \
      --partition="$PARTITION" \
      --job-name=g2_ak_prefit \
      --mem="${MEM_PREFIT:-8G}" --cpus-per-task=2 --time="${TIME_PREFIT:-8:00:00}" \
      --export=ALL \
      scripts/run_goal2_ak_prefit_slurm.sh)
    echo "Prefit -> $PREFIT_JID"
    export PREFIT_JID
  fi
fi

for preset in "${PRESETS[@]}"; do
  if [[ "$NULL" == "default" || "$NULL" == "both" || "$NULL" == "all" ]]; then
    _default "$preset"
  fi
  if [[ "$NULL" == "harris" || "$NULL" == "both" || "$NULL" == "all" ]]; then
    _harris "$preset"
  fi
  if [[ "$NULL" == "pseudo" || "$NULL" == "all" ]]; then
    # Prefit only gates act shards; block/bayes do not use AK.
    if [[ "$preset" != "act_block_only" ]]; then
      PREFIT_JID="" _pseudo "$preset"
    else
      _pseudo "$preset"
    fi
  fi
done

echo "Outputs:"
echo "  default: \$ONE_CACHE_DIR/manifold/res/{split}.npy"
echo "  harris:  \$ONE_CACHE_DIR/manifold/res/{split}_harris_unique.npy"
echo "  pseudo:  \$ONE_CACHE_DIR/manifold/res/{split}_pseudosession.npy"
echo "Monitor: squeue -u \$USER"
