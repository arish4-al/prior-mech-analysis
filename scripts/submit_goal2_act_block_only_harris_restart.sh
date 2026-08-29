#!/bin/bash
# Restart Harris unique-null for act_block_only after OOM'd shards.
#
# Does NOT CLEAR stream_acc. Resubmits only the listed shards with more RAM;
# RESTART=1 skips insertions already in {split}_harris_unique.shard{k}.npy.
# Finalize waits on those jobs, then merges all existing shards (including
# finished 0 and 3).
#
# Keep N_SHARDS=4: shards 0/3 are done; changing N would re-slice insertions
# and invalidate those checkpoints. Remaining work (shard 1 ~31 left, shard 2
# ~13 left, ~1–3 min/insertion) fits in 6 h on two parallel jobs — extra
# shards are not needed.
#
#   bash scripts/submit_goal2_act_block_only_harris_restart.sh
#
# Override:
#   SHARDS="1 2" MEM_SHARD=24G MEM_FIN=32G \
#     bash scripts/submit_goal2_act_block_only_harris_restart.sh
#   SHARDS="0 1 2 3" …   # redo every shard (still no CLEAR)
#   PARTITION=mit_normal bash scripts/submit_goal2_act_block_only_harris_restart.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

SPLIT="${SPLIT:-act_block_only}"
N_SHARDS="${N_SHARDS:-4}"
# Default: the two Aug-14 OOM shards. Finished 0 and 3 stay on disk.
SHARDS="${SHARDS:-1 2}"
NRAND="${NRAND:-2000}"
RESTART="${RESTART:-1}"
MEM_SHARD="${MEM_SHARD:-24G}"
MEM_FIN="${MEM_FIN:-32G}"
CPUS_SHARD="${CPUS_SHARD:-2}"
CPUS_FIN="${CPUS_FIN:-2}"
TIME_SHARD="${TIME_SHARD:-6:00:00}"
TIME_FIN="${TIME_FIN:-6:00:00}"
JOB_PREFIX="${JOB_PREFIX:-g2abh}"
PARTITION="${PARTITION:-pi_fiete}"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/sbatch_defaults.sh"

ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"
export SESSION_SHUFFLE_NULL=1 ACTKERNEL_CHOICE_NULL=0
export ACTKERNEL_NULL_MODE="" ACTKERNEL_PSEUDO_LEN_FACTOR=""

read -r -a SHARD_ARR <<< "$SHARDS"
if [[ ${#SHARD_ARR[@]} -eq 0 ]]; then
  echo "ERROR: SHARDS is empty" >&2
  exit 1
fi
for k in "${SHARD_ARR[@]}"; do
  if [[ "$k" -lt 0 || "$k" -ge "$N_SHARDS" ]]; then
    echo "ERROR: shard $k not in 0..$((N_SHARDS - 1))" >&2
    exit 1
  fi
done

ACC="$ONE_CACHE_DIR/manifold/res/_stream_acc/${SPLIT}_harris_unique"
echo "NULL_SCHEME=harris_unique  SPLIT=$SPLIT  restart shards: ${SHARD_ARR[*]}/$N_SHARDS"
echo "RESTART=$RESTART  MEM_SHARD=$MEM_SHARD  MEM_FIN=$MEM_FIN  nrand=$NRAND"
echo "TIME_SHARD=$TIME_SHARD  TIME_FIN=$TIME_FIN  N_SHARDS=$N_SHARDS (keep 4; do not re-slice)"
echo "Will NOT clear stream_acc. Existing checkpoints:"
ls -lh "${ACC}".shard*.npy 2>/dev/null || echo "  (none yet)"

job_tag() {
  local s="$1"
  s="${s//./p}"
  echo "${s:0:40}"
}
TAG=$(job_tag "$SPLIT")

SHARD_JOBS=()
for k in "${SHARD_ARR[@]}"; do
  # shellcheck disable=SC2086
  JID=$(sbatch --parsable $SBATCH_EXTRA \
    --partition="$PARTITION" \
    --mem="$MEM_SHARD" --cpus-per-task="$CPUS_SHARD" --time="$TIME_SHARD" \
    --job-name="${JOB_PREFIX}_${TAG}_s${k}" \
    --export=ALL,SPLIT="$SPLIT",SHARD_IDX="$k",N_SHARDS="$N_SHARDS",NRAND="$NRAND",RESTART="$RESTART",SESSION_SHUFFLE_NULL=1,ACTKERNEL_CHOICE_NULL=0,ACTKERNEL_NULL_MODE=,ACTKERNEL_PSEUDO_LEN_FACTOR= \
    scripts/run_goal2_shard_slurm.sh)
  SHARD_JOBS+=("$JID")
  echo "  $SPLIT shard $k/$N_SHARDS -> $JID"
done

DEP=$(IFS=:; echo "${SHARD_JOBS[*]}")
# shellcheck disable=SC2086
FID=$(sbatch --parsable $SBATCH_EXTRA \
  --partition="$PARTITION" \
  --mem="$MEM_FIN" --cpus-per-task="$CPUS_FIN" --time="$TIME_FIN" \
  --dependency=afterok:"$DEP" \
  --job-name="${JOB_PREFIX}_fin_${TAG}" \
  --export=ALL,SPLIT="$SPLIT",SESSION_SHUFFLE_NULL=1,ACTKERNEL_CHOICE_NULL=0,ACTKERNEL_NULL_MODE=,ACTKERNEL_PSEUDO_LEN_FACTOR= \
  scripts/run_goal2_finalize_slurm.sh)
echo "  $SPLIT finalize -> $FID  (merges all existing .shard*.npy)"
echo "Monitor: squeue -u \$USER"
echo "Final: \$ONE_CACHE_DIR/manifold/res/${SPLIT}_harris_unique.npy"
