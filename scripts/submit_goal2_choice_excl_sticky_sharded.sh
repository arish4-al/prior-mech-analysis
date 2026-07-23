#!/bin/bash
# Submit sharded Goal 2 choice L–R runs with late+perseveration trial exclusion
# and label-shuffle nulls within stim×block (default null; no donor bank).
#
# Exclusion: drop last 20% of each session OR the **tail** of same-choice
# runs ≥10 that are poorly explained by non-0 contrast stim (keep first 9
# trials of each such run). Outputs → manifold/res_excl_sticky/.
#
#   bash scripts/submit_goal2_choice_excl_sticky_sharded.sh
#   PRESET=choice_lr_excl_sticky_true N_SHARDS=4 \
#     bash scripts/submit_goal2_choice_excl_sticky_sharded.sh
#   PRESET=choice_lr_excl_sticky_bayes \
#     bash scripts/submit_goal2_choice_excl_sticky_sharded.sh
#
# Assumes insertion cache already exists.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

PRESET="${PRESET:-choice_lr_excl_sticky_act}"
N_SHARDS="${N_SHARDS:-4}"
NRAND="${NRAND:-2000}"
RESTART="${RESTART:-1}"
EXCLUDE_STICKY_TRIALS="${EXCLUDE_STICKY_TRIALS:-1}"
STICKY_LATE_FRAC="${STICKY_LATE_FRAC:-0.2}"
STICKY_MIN_RUN="${STICKY_MIN_RUN:-10}"
SESSION_SHUFFLE_NULL="${SESSION_SHUFFLE_NULL:-0}"
MEM_SHARD="${MEM_SHARD:-6G}"
MEM_FIN="${MEM_FIN:-10G}"
CPUS_SHARD="${CPUS_SHARD:-2}"
CPUS_FIN="${CPUS_FIN:-2}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"
export EXCLUDE_STICKY_TRIALS STICKY_LATE_FRAC STICKY_MIN_RUN
export SESSION_SHUFFLE_NULL

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

n_shard_jobs=$(( ${#SPLITS[@]} * N_SHARDS ))
echo "PRESET=$PRESET  N_SHARDS=$N_SHARDS  nrand=$NRAND  splits=${#SPLITS[@]}"
echo "EXCLUDE_STICKY_TRIALS=$EXCLUDE_STICKY_TRIALS  late_frac=$STICKY_LATE_FRAC  min_run=$STICKY_MIN_RUN"
echo "SESSION_SHUFFLE_NULL=$SESSION_SHUFFLE_NULL"
echo "MEM_SHARD=$MEM_SHARD  MEM_FIN=$MEM_FIN  shard_jobs=$n_shard_jobs"
echo "Outputs: \$ONE_CACHE_DIR/manifold/res_excl_sticky/"
echo "Splits:"
printf '  %s\n' "${SPLITS[@]}"

job_tag() {
  local s="$1"
  s="${s//./p}"
  echo "${s:0:48}"
}

for sp in "${SPLITS[@]}"; do
  TAG=$(job_tag "$sp")
  SHARD_JOBS=()
  for ((k=0; k<N_SHARDS; k++)); do
    JID=$(sbatch --parsable \
      --mem="$MEM_SHARD" --cpus-per-task="$CPUS_SHARD" \
      --job-name="g2x_${TAG}_s${k}" \
      --export=ALL,SPLIT="$sp",SHARD_IDX="$k",N_SHARDS="$N_SHARDS",NRAND="$NRAND",RESTART="$RESTART",EXCLUDE_STICKY_TRIALS="$EXCLUDE_STICKY_TRIALS",STICKY_LATE_FRAC="$STICKY_LATE_FRAC",STICKY_MIN_RUN="$STICKY_MIN_RUN",SESSION_SHUFFLE_NULL="$SESSION_SHUFFLE_NULL" \
      scripts/run_goal2_shard_slurm.sh)
    SHARD_JOBS+=("$JID")
    echo "  $sp shard $k/$N_SHARDS -> $JID"
  done
  DEP=$(IFS=:; echo "${SHARD_JOBS[*]}")
  FID=$(sbatch --parsable \
    --mem="$MEM_FIN" --cpus-per-task="$CPUS_FIN" \
    --dependency=afterok:"$DEP" \
    --job-name="g2x_fin_${TAG}" \
    --export=ALL,SPLIT="$sp",EXCLUDE_STICKY_TRIALS="$EXCLUDE_STICKY_TRIALS" \
    scripts/run_goal2_finalize_slurm.sh)
  echo "  $sp finalize -> $FID (after $DEP)"
done

echo "Done. Monitor: squeue -u \$USER"
echo "Null: label shuffle within stim×block on excluded-trial set (default)"
echo "Final outputs: \$ONE_CACHE_DIR/manifold/res_excl_sticky/{split}.npy"
