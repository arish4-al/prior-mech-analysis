#!/bin/bash
# Resume timed-out option-1 + copy-last f1 shards (keep N_SHARDS=4).
#
# Does NOT CLEAR stream_acc. RESTART=1 skips insertions already in
# {split}_pseudo_strat_sticky.shard{k}.npy. Finalize merges all four shard
# files (including the two already-finished
# act_block_duringstim_r_choice_r_f1 s2/s3).
#
# Keep N_SHARDS=4: leftover work is the same insertion slice. Changing N
# re-slices and invalidates those checkpoints (~699 insertions/split).
# Start at factor 6 so leftover insertions skip the failed 42k-draw factor-3
# pass. Already-pooled insertions stay as they were (factor 3 or 3→6 bump).
#
#   bash scripts/submit_goal2_ak_sticky_f1_restart.sh
#
# Override one split:
#   SPLIT=act_block_stim_r_duringchoice_r_f1 SHARDS="0 1 2 3" \
#     bash scripts/submit_goal2_ak_sticky_f1_restart.sh
#
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

N_SHARDS="${N_SHARDS:-4}"
if [[ "$N_SHARDS" != "4" ]]; then
  echo "ERROR: N_SHARDS must stay 4 (got $N_SHARDS). Re-slicing drops checkpoints." >&2
  exit 1
fi

NRAND="${NRAND:-2000}"
RESTART="${RESTART:-1}"
MEM_SHARD="${MEM_SHARD:-12G}"
MEM_FIN="${MEM_FIN:-10G}"
CPUS_SHARD="${CPUS_SHARD:-2}"
CPUS_FIN="${CPUS_FIN:-2}"
TIME_SHARD="${TIME_SHARD:-2:00:00}"
TIME_FIN="${TIME_FIN:-2:00:00}"
JOB_PREFIX="${JOB_PREFIX:-g2pss}"
PSEUDO_LEN_FACTOR="${PSEUDO_LEN_FACTOR:-6}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"
export ACTKERNEL_CHOICE_NULL=1 ACTKERNEL_NULL_MODE=strat ACTKERNEL_LATE_STICKY=1
export ACTKERNEL_PSEUDO_LEN_FACTOR="$PSEUDO_LEN_FACTOR"
export SESSION_SHUFFLE_NULL=0

# Unfinished shards from 2026-08-22/23 logs (s2/s3 of duringstim_r_choice_r_f1 done).
# Override with SPLIT=... SHARDS="..." to run one split only.
if [[ -n "${SPLIT:-}" ]]; then
  SPLITS=("$SPLIT")
  SHARDS_FOR=("${SHARDS:-0 1 2 3}")
else
  SPLITS=(
    act_block_duringstim_r_choice_r_f1
    act_block_duringstim_l_choice_l_f1
    act_block_stim_r_duringchoice_r_f1
    act_block_stim_l_duringchoice_l_f1
  )
  SHARDS_FOR=(
    "0 1"
    "0 1 2 3"
    "0 1 2 3"
    "0 1 2 3"
  )
fi

job_tag() {
  local s="$1"
  s="${s//./p}"
  echo "${s:0:40}"
}

ACC_ROOT="$ONE_CACHE_DIR/manifold/res/_stream_acc"
echo "NULL_SCHEME=pseudo_strat_sticky  mode=strat  late_sticky=1"
echo "N_SHARDS=$N_SHARDS (keep 4)  factor=$PSEUDO_LEN_FACTOR  RESTART=$RESTART  nrand=$NRAND"
echo "MEM_SHARD=$MEM_SHARD  TIME_SHARD=$TIME_SHARD  TIME_FIN=$TIME_FIN"
echo "Will NOT clear stream_acc."

n_shard_jobs=0
for i in "${!SPLITS[@]}"; do
  sp="${SPLITS[$i]}"
  read -r -a SHARD_ARR <<< "${SHARDS_FOR[$i]}"
  if [[ ${#SHARD_ARR[@]} -eq 0 ]]; then
    echo "ERROR: empty SHARDS for $sp" >&2
    exit 1
  fi
  for k in "${SHARD_ARR[@]}"; do
    if [[ "$k" -lt 0 || "$k" -ge "$N_SHARDS" ]]; then
      echo "ERROR: $sp shard $k not in 0..$((N_SHARDS - 1))" >&2
      exit 1
    fi
  done
  echo
  echo "=== $sp  restart shards: ${SHARD_ARR[*]}/$N_SHARDS ==="
  ls -lh "$ACC_ROOT/${sp}_pseudo_strat_sticky.shard"*.npy 2>/dev/null || echo "  (no shard files yet)"

  TAG=$(job_tag "$sp")
  SHARD_JOBS=()
  for k in "${SHARD_ARR[@]}"; do
    JID=$(sbatch --parsable \
      --mem="$MEM_SHARD" --cpus-per-task="$CPUS_SHARD" --time="$TIME_SHARD" \
      --job-name="${JOB_PREFIX}_${TAG}_s${k}" \
      --export=ALL,SPLIT="$sp",SHARD_IDX="$k",N_SHARDS="$N_SHARDS",NRAND="$NRAND",RESTART="$RESTART",ACTKERNEL_CHOICE_NULL=1,ACTKERNEL_NULL_MODE=strat,ACTKERNEL_PSEUDO_LEN_FACTOR="$PSEUDO_LEN_FACTOR",ACTKERNEL_LATE_STICKY=1,SESSION_SHUFFLE_NULL=0 \
      scripts/run_goal2_shard_slurm.sh)
    SHARD_JOBS+=("$JID")
    n_shard_jobs=$((n_shard_jobs + 1))
    echo "  $sp shard $k/$N_SHARDS -> $JID"
  done
  DEP=$(IFS=:; echo "${SHARD_JOBS[*]}")
  FID=$(sbatch --parsable \
    --mem="$MEM_FIN" --cpus-per-task="$CPUS_FIN" --time="$TIME_FIN" \
    --dependency=afterok:"$DEP" \
    --job-name="${JOB_PREFIX}_fin_${TAG}" \
    --export=ALL,SPLIT="$sp",ACTKERNEL_CHOICE_NULL=1,ACTKERNEL_NULL_MODE=strat,ACTKERNEL_PSEUDO_LEN_FACTOR="$PSEUDO_LEN_FACTOR",ACTKERNEL_LATE_STICKY=1,SESSION_SHUFFLE_NULL=0 \
    scripts/run_goal2_finalize_slurm.sh)
  echo "  $sp finalize -> $FID  (merges all existing .shard*.npy)"
done

echo
echo "Submitted $n_shard_jobs shard jobs + ${#SPLITS[@]} finalize."
echo "Monitor: squeue -u \$USER"
echo "Final: \$ONE_CACHE_DIR/manifold/res/{split}_pseudo_strat_sticky.npy"
