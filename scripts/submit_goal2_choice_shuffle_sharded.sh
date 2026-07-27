#!/bin/bash
# Submit sharded Goal 2 jobs: plain label-shuffle null for the 8 choice L–R act
# splits (min_trials_per_side=5). No Harris / actkernel suffix — writes
#   $ONE_CACHE_DIR/manifold/res/{split}.npy
#   $ONE_CACHE_DIR/manifold/res/{split}_regde.npy
# etc. Structured-null files ({split}_harris*, _pseudo_*) are left alone.
#
#   bash scripts/submit_goal2_choice_shuffle_sharded.sh
#
# Defaults: PRESET=choice_lr_session_null_all (8 act splits), N_SHARDS=4,
# NRAND=2000, CLEAR_STREAM=1 (clears only plain {split} stream_acc + res).
#
# After finalize, copy/symlink into res/new if that is your analysis folder, then:
#   python scripts/plot_choice_null_comparison_table.py \
#     --openalyx-res $ONE_CACHE_DIR/manifold/res \
#     --arm-res $ONE_CACHE_DIR/manifold/res/new \
#     --arm-tag harris_unique --arm-suffix _harris_unique --force-combine --alpha 0.01
#
# Override: N_SHARDS=3 CLEAR_STREAM=0 MEM_SHARD=8G \
#   bash scripts/submit_goal2_choice_shuffle_sharded.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

PRESET="${PRESET:-choice_lr_session_null_all}"
N_SHARDS="${N_SHARDS:-4}"
NRAND="${NRAND:-2000}"
RESTART="${RESTART:-1}"
CLEAR_STREAM="${CLEAR_STREAM:-1}"
MEM_SHARD="${MEM_SHARD:-6G}"
MEM_FIN="${MEM_FIN:-10G}"
CPUS_SHARD="${CPUS_SHARD:-2}"
CPUS_FIN="${CPUS_FIN:-2}"
TIME_SHARD="${TIME_SHARD:-12:00:00}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"

# Plain shuffle: no structured-null flags
export SESSION_SHUFFLE_NULL=0
export ACTKERNEL_CHOICE_NULL=0
export ACTKERNEL_NULL_MODE=""
export ACTKERNEL_PSEUDO_LEN_FACTOR=""

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

# Clear only exact plain basenames (never *_harris* / *_pseudo*).
if [[ "$CLEAR_STREAM" == "1" ]]; then
  RES_ROOT="$ONE_CACHE_DIR/manifold/res"
  ACC="$RES_ROOT/_stream_acc"
  echo "CLEAR_STREAM=1: removing prior plain {split} stream_acc + res under $RES_ROOT"
  for sp in "${SPLITS[@]}"; do
    rm -f "$ACC/${sp}.npy" "$ACC/${sp}.shard"*.npy 2>/dev/null || true
    rm -f "$RES_ROOT/${sp}.npy" "$RES_ROOT/${sp}_regde.npy" \
      "$RES_ROOT/${sp}_all.npy" "$RES_ROOT/${sp}_all_regde.npy" \
      2>/dev/null || true
  done
fi

n_shard_jobs=$(( ${#SPLITS[@]} * N_SHARDS ))
echo "NULL_SCHEME=label_shuffle (plain)  PRESET=$PRESET"
echo "CLEAR_STREAM=$CLEAR_STREAM  RESTART=$RESTART"
echo "N_SHARDS=$N_SHARDS  nrand=$NRAND  splits=${#SPLITS[@]}  shard_jobs=$n_shard_jobs"
echo "MEM_SHARD=$MEM_SHARD  TIME_SHARD=$TIME_SHARD"
echo "Outputs: \$ONE_CACHE_DIR/manifold/res/{split}.npy  (no suffix)"
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
    JID=$(sbatch --parsable \
      --mem="$MEM_SHARD" --cpus-per-task="$CPUS_SHARD" --time="$TIME_SHARD" \
      --job-name="g2sh_${TAG}_s${k}" \
      --export=ALL,SPLIT="$sp",SHARD_IDX="$k",N_SHARDS="$N_SHARDS",NRAND="$NRAND",RESTART="$RESTART",SESSION_SHUFFLE_NULL=0,ACTKERNEL_CHOICE_NULL=0,ACTKERNEL_NULL_MODE=,ACTKERNEL_PSEUDO_LEN_FACTOR= \
      scripts/run_goal2_shard_slurm.sh)
    SHARD_JOBS+=("$JID")
    echo "  $sp shard $k/$N_SHARDS -> $JID"
  done
  DEP=$(IFS=:; echo "${SHARD_JOBS[*]}")
  FID=$(sbatch --parsable \
    --mem="$MEM_FIN" --cpus-per-task="$CPUS_FIN" \
    --dependency=afterok:"$DEP" \
    --job-name="g2sh_fin_${TAG}" \
    --export=ALL,SPLIT="$sp",SESSION_SHUFFLE_NULL=0,ACTKERNEL_CHOICE_NULL=0,ACTKERNEL_NULL_MODE=,ACTKERNEL_PSEUDO_LEN_FACTOR= \
    scripts/run_goal2_finalize_slurm.sh)
  echo "  $sp finalize -> $FID"
done

echo "Done (plain label shuffle). Monitor: squeue -u \$USER"
echo "Final outputs: \$ONE_CACHE_DIR/manifold/res/{split}.npy (+ _regde / _all)"
