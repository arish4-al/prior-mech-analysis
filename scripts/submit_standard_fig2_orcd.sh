#!/bin/bash
# Submit the standard-split pipeline (Figure 2, Figure 4 prior, ITI act_block_only).
#
# Labels: action kernel (α=0.2) on the full session, including bad trials and
# the unbiased block, then the bad-trial / saturation mask, then drop true
# probabilityLeft==0.5, then min_trials_per_side=5. Null is the plain
# within-stratum label shuffle (nrand=2000), except act_block_only, which
# uses generate_pseudo_blocks. No Harris / actkernel suffix.
#
# Writes only:
#   $ONE_CACHE_DIR/manifold/res_standard/{split}.npy
#   $ONE_CACHE_DIR/manifold/res_standard/{split}_regde.npy
# Does not touch manifold/res or manifold/res/new.
#
#   bash scripts/submit_standard_fig2_orcd.sh
#
# Defaults: PRESET=standard (29 splits: 18 Figure 2 + 10 Figure 4 prior
# + act_block_only), N_SHARDS=4, NRAND=2000, CLEAR_STREAM=1 (clears only
# those basenames under res_standard). 29 × 4 shards at 6G is ~696G if
# every shard runs at once.
#
# Override: N_SHARDS=2 CLEAR_STREAM=0 MEM_SHARD=8G \
#   bash scripts/submit_standard_fig2_orcd.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

PRESET="${PRESET:-standard}"
NRAND="${NRAND:-2000}"
RESTART="${RESTART:-1}"
CLEAR_STREAM="${CLEAR_STREAM:-1}"
# DEBIAS_EUC=1: noise-corrected d_euc, files {split}_debias*; CLEAR_STREAM then
# clears only those, so the uncorrected shuffle catalog is kept.
DEBIAS_EUC="${DEBIAS_EUC:-0}"
SFX=""
if [[ "$DEBIAS_EUC" == "1" ]]; then SFX="_debias"; fi
MEM_SHARD="${MEM_SHARD:-6G}"
MEM_FIN="${MEM_FIN:-10G}"
CPUS_SHARD="${CPUS_SHARD:-2}"
CPUS_FIN="${CPUS_FIN:-2}"
N_SHARDS="${N_SHARDS:-4}"
TIME_SHARD="${TIME_SHARD:-12:00:00}"
PARTITION="${PARTITION:-pi_fiete}"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/sbatch_defaults.sh"

ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"

# Plain shuffle. Standard labels are a separate flag, not a null suffix.
export SESSION_SHUFFLE_NULL=0
export ACTKERNEL_CHOICE_NULL=0
export ACTKERNEL_NULL_MODE=""
export ACTKERNEL_PSEUDO_LEN_FACTOR=""
export STANDARD_ACT_PRIOR=1

# List splits with the ibl env. `module load miniforge` puts a python3
# without `one` first on PATH, and --list-splits imports block_analysis.
PY="${PY:-$HOME/conda_envs/ibl/bin/python}"
if [[ ! -x "$PY" ]]; then
  echo "ERROR: ibl python not found at $PY" >&2
  exit 1
fi

SPLITS=()
while IFS= read -r line; do
  [[ -n "$line" ]] && SPLITS+=("$line")
done < <("$PY" -u scripts/run_goal2_splits.py --preset "$PRESET" --list-splits)

if [[ ${#SPLITS[@]} -eq 0 ]]; then
  echo "ERROR: no splits for PRESET=$PRESET" >&2
  exit 1
fi

RES_ROOT="$ONE_CACHE_DIR/manifold/res_standard"
if [[ "$CLEAR_STREAM" == "1" ]]; then
  ACC="$RES_ROOT/_stream_acc"
  echo "CLEAR_STREAM=1: removing prior plain {split}${SFX} stream_acc + res under $RES_ROOT"
  for sp in "${SPLITS[@]}"; do
    base="${sp}${SFX}"
    rm -f "$ACC/${base}.npy" "$ACC/${base}.shard"*.npy 2>/dev/null || true
    rm -f "$ACC/${base}.skipped.jsonl" "$ACC/${base}.shard"*.skipped.jsonl 2>/dev/null || true
    rm -f "$RES_ROOT/${base}.npy" "$RES_ROOT/${base}_regde.npy" \
      "$RES_ROOT/${base}_all.npy" "$RES_ROOT/${base}_all_regde.npy" \
      "$RES_ROOT/${base}_skipped.json" 2>/dev/null || true
  done
fi

n_shard_jobs=$(( ${#SPLITS[@]} * N_SHARDS ))
echo "NULL_SCHEME=label_shuffle (plain); act_block_only=pseudo-blocks  PRESET=$PRESET  STANDARD_ACT_PRIOR=1"
echo "CLEAR_STREAM=$CLEAR_STREAM  RESTART=$RESTART  DEBIAS_EUC=$DEBIAS_EUC"
echo "N_SHARDS=$N_SHARDS  nrand=$NRAND  splits=${#SPLITS[@]}  shard_jobs=$n_shard_jobs"
echo "If all shards run at once: about $(( ${#SPLITS[@]} * N_SHARDS )) x $MEM_SHARD"
echo "PARTITION=$PARTITION  SBATCH_EXTRA=${SBATCH_EXTRA:-}  MEM_SHARD=$MEM_SHARD  TIME_SHARD=$TIME_SHARD"
echo "Outputs: \$ONE_CACHE_DIR/manifold/res_standard/{split}.npy  (no suffix)"
printf '  %s\n' "${SPLITS[@]}"

job_tag() {
  local s="$1"
  s="${s//./p}"
  echo "${s:0:40}"
}

EXPORT_COMMON="SESSION_SHUFFLE_NULL=0,ACTKERNEL_CHOICE_NULL=0,ACTKERNEL_NULL_MODE=,ACTKERNEL_PSEUDO_LEN_FACTOR=,STANDARD_ACT_PRIOR=1,EXCLUDE_STICKY_TRIALS=0,DEBIAS_EUC=${DEBIAS_EUC}"

for sp in "${SPLITS[@]}"; do
  TAG=$(job_tag "$sp")
  SHARD_JOBS=()
  for ((k=0; k<N_SHARDS; k++)); do
    # shellcheck disable=SC2086
    JID=$(sbatch --parsable $SBATCH_EXTRA \
      --partition="$PARTITION" \
      --mem="$MEM_SHARD" --cpus-per-task="$CPUS_SHARD" --time="$TIME_SHARD" \
      --job-name="std_${TAG}_s${k}" \
      --export=ALL,SPLIT="$sp",SHARD_IDX="$k",N_SHARDS="$N_SHARDS",NRAND="$NRAND",RESTART="$RESTART",$EXPORT_COMMON \
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
    --job-name="std_fin_${TAG}" \
    --export=ALL,SPLIT="$sp",$EXPORT_COMMON \
    scripts/run_goal2_finalize_slurm.sh)
  echo "  $sp finalize -> $FID"
done

echo "Done (standard splits). Monitor: squeue -u \$USER"
echo "Final outputs: \$ONE_CACHE_DIR/manifold/res_standard/{split}.npy (+ _regde / _all)"
