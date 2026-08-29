#!/bin/bash
# Submit sharded Goal 2 act-prior L–R runs with late+perseveration trial
# exclusion, then label-shuffle within stim×block (default null; no donor bank).
#
# Default PRESET=act_block_excl_sticky (10 splits):
#   duringstim (4):  act_block_duringstim_{l,r}_choice_{l_f1,r_f2}
#   duringchoice (4): act_block_stim_{l,r}_duringchoice_{l_f1,r_f2}
#   unsplit stim (2): act_block_duringstim_{l,r}
# No unsplit duringchoice / act_block_only.
#
#   bash scripts/submit_goal2_act_block_excl_sticky_sharded.sh
#   PRESET=act_block_excl_sticky_duringstim \
#     bash scripts/submit_goal2_act_block_excl_sticky_sharded.sh
#   PRESET=act_block_excl_sticky_duringchoice …
#   PRESET=act_block_excl_sticky_unsplit_duringstim …
#   PARTITION=mit_normal bash scripts/submit_goal2_act_block_excl_sticky_sharded.sh
#
# Outputs → manifold/res_excl_sticky/{split}.npy (same folder as choice
# excl-sticky). CLEAR_STREAM=1 (default) removes only the listed act_block
# basenames there — never choice_* files.
#
# Assumes insertion cache already exists.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

PRESET="${PRESET:-act_block_excl_sticky}"
N_SHARDS="${N_SHARDS:-4}"
NRAND="${NRAND:-2000}"
RESTART="${RESTART:-1}"
CLEAR_STREAM="${CLEAR_STREAM:-1}"
EXCLUDE_STICKY_TRIALS="${EXCLUDE_STICKY_TRIALS:-1}"
STICKY_LATE_FRAC="${STICKY_LATE_FRAC:-0.2}"
STICKY_MIN_RUN="${STICKY_MIN_RUN:-10}"
MEM_SHARD="${MEM_SHARD:-6G}"
MEM_FIN="${MEM_FIN:-10G}"
CPUS_SHARD="${CPUS_SHARD:-2}"
CPUS_FIN="${CPUS_FIN:-2}"
TIME_SHARD="${TIME_SHARD:-12:00:00}"
PARTITION="${PARTITION:-pi_fiete}"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/sbatch_defaults.sh"

ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"
export EXCLUDE_STICKY_TRIALS STICKY_LATE_FRAC STICKY_MIN_RUN
# Pin shuffle-after-trim. --export=ALL would otherwise leak AK/Harris flags
# from a prior submit in this shell and write _pseudo_* / _harris into
# res_excl_sticky.
export SESSION_SHUFFLE_NULL=0
export ACTKERNEL_CHOICE_NULL=0
export ACTKERNEL_NULL_MODE=""
export ACTKERNEL_PSEUDO_LEN_FACTOR=""
export ACTKERNEL_LATE_STICKY=0

JOB_PREFIX="${JOB_PREFIX:-g2abx}"

module load miniforge 2>/dev/null || true
if [[ -f "$HOME/conda_envs/ibl/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/conda_envs/ibl/bin/activate"
elif command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate ibl 2>/dev/null || conda activate iblenv 2>/dev/null || true
fi

SPLITS=()
while IFS= read -r line; do
  [[ -n "$line" ]] && SPLITS+=("$line")
done < <(python3 -u scripts/run_goal2_splits.py --preset "$PRESET" --list-splits)

if [[ ${#SPLITS[@]} -eq 0 ]]; then
  echo "ERROR: no splits for PRESET=$PRESET" >&2
  exit 1
fi

for sp in "${SPLITS[@]}"; do
  if [[ "$sp" != act_block_* ]]; then
    echo "ERROR: refusing non-act_block split '$sp' (would risk choice excl-sticky)" >&2
    exit 1
  fi
done

RES_ROOT="$ONE_CACHE_DIR/manifold/res_excl_sticky"
if [[ "$CLEAR_STREAM" == "1" ]]; then
  ACC="$RES_ROOT/_stream_acc"
  echo "CLEAR_STREAM=1: removing listed act_block stream_acc + res under $RES_ROOT"
  echo "(choice_* files are not touched)"
  for sp in "${SPLITS[@]}"; do
    rm -f "$ACC/${sp}.npy" "$ACC/${sp}.shard"*.npy 2>/dev/null || true
    rm -f "$RES_ROOT/${sp}.npy" "$RES_ROOT/${sp}_regde.npy" \
      "$RES_ROOT/${sp}_all.npy" "$RES_ROOT/${sp}_all_regde.npy" \
      2>/dev/null || true
  done
fi

n_shard_jobs=$(( ${#SPLITS[@]} * N_SHARDS ))
echo "PRESET=$PRESET  N_SHARDS=$N_SHARDS  nrand=$NRAND  splits=${#SPLITS[@]}"
echo "EXCLUDE_STICKY_TRIALS=$EXCLUDE_STICKY_TRIALS  late_frac=$STICKY_LATE_FRAC  min_run=$STICKY_MIN_RUN"
echo "SESSION_SHUFFLE_NULL=$SESSION_SHUFFLE_NULL  CLEAR_STREAM=$CLEAR_STREAM"
echo "PARTITION=$PARTITION  MEM_SHARD=$MEM_SHARD  TIME_SHARD=$TIME_SHARD  MEM_FIN=$MEM_FIN  shard_jobs=$n_shard_jobs"
echo "Outputs: \$ONE_CACHE_DIR/manifold/res_excl_sticky/{split}.npy"
echo "Splits:"
printf '  %s\n' "${SPLITS[@]}"

job_tag() {
  local s="$1"
  s="${s//./p}"
  echo "${s:0:40}"
}

for sp in "${SPLITS[@]}"; do
  TAG=$(job_tag "$sp")
  SHARD_JOBS=()
  for ((k=0; k<N_SHARDS; k++)); do
    # shellcheck disable=SC2086
    JID=$(sbatch --parsable $SBATCH_EXTRA \
      --partition="$PARTITION" \
      --mem="$MEM_SHARD" --cpus-per-task="$CPUS_SHARD" --time="$TIME_SHARD" \
      --job-name="${JOB_PREFIX}_${TAG}_s${k}" \
      --export=ALL,SPLIT="$sp",SHARD_IDX="$k",N_SHARDS="$N_SHARDS",NRAND="$NRAND",RESTART="$RESTART",EXCLUDE_STICKY_TRIALS="$EXCLUDE_STICKY_TRIALS",STICKY_LATE_FRAC="$STICKY_LATE_FRAC",STICKY_MIN_RUN="$STICKY_MIN_RUN",SESSION_SHUFFLE_NULL=0,ACTKERNEL_CHOICE_NULL=0,ACTKERNEL_NULL_MODE=,ACTKERNEL_PSEUDO_LEN_FACTOR=,ACTKERNEL_LATE_STICKY=0 \
      scripts/run_goal2_shard_slurm.sh)
    SHARD_JOBS+=("$JID")
    echo "  $sp shard $k/$N_SHARDS -> $JID"
  done
  DEP=$(IFS=:; echo "${SHARD_JOBS[*]}")
  # shellcheck disable=SC2086
  FID=$(sbatch --parsable $SBATCH_EXTRA \
    --partition="$PARTITION" \
    --mem="$MEM_FIN" --cpus-per-task="$CPUS_FIN" \
    --dependency=afterok:"$DEP" \
    --job-name="${JOB_PREFIX}_fin_${TAG}" \
    --export=ALL,SPLIT="$sp",EXCLUDE_STICKY_TRIALS="$EXCLUDE_STICKY_TRIALS",SESSION_SHUFFLE_NULL=0,ACTKERNEL_CHOICE_NULL=0,ACTKERNEL_NULL_MODE=,ACTKERNEL_PSEUDO_LEN_FACTOR=,ACTKERNEL_LATE_STICKY=0 \
    scripts/run_goal2_finalize_slurm.sh)
  echo "  $sp finalize -> $FID (after $DEP)"
done

echo "Done. Monitor: squeue -u \$USER"
echo "Null: label shuffle within stim×block on excluded-trial set (default)"
echo "Final outputs: \$ONE_CACHE_DIR/manifold/res_excl_sticky/{split}.npy"
