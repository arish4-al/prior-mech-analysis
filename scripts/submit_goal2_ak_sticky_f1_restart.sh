#!/bin/bash
# Resume timed-out option-1 + copy-last f1 shards (keep N_SHARDS=4).
#
# Does NOT CLEAR stream_acc. RESTART=1 skips insertions already in
# {split}_pseudo_strat_sticky.shard{k}.npy. Finalize merges all four shard
# files (including shards that already finished).
#
# Keep N_SHARDS=4: leftover work is the same insertion slice. Changing N
# re-slices and invalidates those checkpoints.
# Start at factor 6 so leftover insertions skip the failed 42k-draw factor-3
# pass.
#
# 2026-08-23 restart #2 leftovers (2h TIME_SHARD timed out again):
#   duringstim_r_choice_r_f1  s0=15  s1=43   (s2/s3 done first wave)
#   duringstim_l_choice_l_f1  s0=3   s1=26  s2=38  s3=6
#   stim_r_duringchoice_r_f1  s0=26  s1=10  (s2/s3 done this restart)
#   stim_l_duringchoice_l_f1  DONE + finalized (635 ins) — not in default list
# Default TIME_SHARD=12h (mit_normal max). Worst leftover shard is 43
# insertions at factor 6.
#
#   bash scripts/submit_goal2_ak_sticky_f1_restart.sh
#
# Override one split:
#   SPLIT=act_block_stim_r_duringchoice_r_f1 SHARDS="0 1" \
#     bash scripts/submit_goal2_ak_sticky_f1_restart.sh
#   PARTITION=mit_normal bash scripts/submit_goal2_ak_sticky_f1_restart.sh
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
TIME_SHARD="${TIME_SHARD:-12:00:00}"
TIME_FIN="${TIME_FIN:-2:00:00}"
JOB_PREFIX="${JOB_PREFIX:-g2pss}"
PSEUDO_LEN_FACTOR="${PSEUDO_LEN_FACTOR:-6}"
PARTITION="${PARTITION:-pi_fiete}"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/sbatch_defaults.sh"

ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"
export ACTKERNEL_CHOICE_NULL=1 ACTKERNEL_NULL_MODE=strat ACTKERNEL_LATE_STICKY=1
export ACTKERNEL_PSEUDO_LEN_FACTOR="$PSEUDO_LEN_FACTOR"
export SESSION_SHUFFLE_NULL=0

# Unfinished shards after the 2h restart (2026-08-23 16:33–16:39 TIME LIMIT).
# Override with SPLIT=... SHARDS="..." to run one split only.
if [[ -n "${SPLIT:-}" ]]; then
  SPLITS=("$SPLIT")
  SHARDS_FOR=("${SHARDS:-0 1 2 3}")
else
  SPLITS=(
    act_block_duringstim_r_choice_r_f1
    act_block_duringstim_l_choice_l_f1
    act_block_stim_r_duringchoice_r_f1
  )
  SHARDS_FOR=(
    "0 1"
    "0 1 2 3"
    "0 1"
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
    # shellcheck disable=SC2086
    JID=$(sbatch --parsable $SBATCH_EXTRA \
      --partition="$PARTITION" \
      --mem="$MEM_SHARD" --cpus-per-task="$CPUS_SHARD" --time="$TIME_SHARD" \
      --job-name="${JOB_PREFIX}_${TAG}_s${k}" \
      --export=ALL,SPLIT="$sp",SHARD_IDX="$k",N_SHARDS="$N_SHARDS",NRAND="$NRAND",RESTART="$RESTART",ACTKERNEL_CHOICE_NULL=1,ACTKERNEL_NULL_MODE=strat,ACTKERNEL_PSEUDO_LEN_FACTOR="$PSEUDO_LEN_FACTOR",ACTKERNEL_LATE_STICKY=1,SESSION_SHUFFLE_NULL=0 \
      scripts/run_goal2_shard_slurm.sh)
    SHARD_JOBS+=("$JID")
    n_shard_jobs=$((n_shard_jobs + 1))
    echo "  $sp shard $k/$N_SHARDS -> $JID"
  done
  DEP=$(IFS=:; echo "${SHARD_JOBS[*]}")
  # shellcheck disable=SC2086
  FID=$(sbatch --parsable $SBATCH_EXTRA \
    --partition="$PARTITION" \
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
