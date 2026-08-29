#!/bin/bash
# Submit sharded jobs for stim L vs R under Bayes prior (plain label shuffle).
# Default: duringstim 4-split (choice × Bayes stratum). Not stim_block_{l,r}_bayes.
#
#   bash scripts/submit_goal2_stim_lr_bayes_sharded.sh
#   PRESET=stim_lr_bayes_all bash scripts/submit_goal2_stim_lr_bayes_sharded.sh  # + 80 ms prior-only
#
# Prefer the Bayes shuffle wrapper (also submits choice L–R duringstim):
#   bash scripts/submit_goal2_bayes_shuffle_orcd.sh
#
# Assumes insertion cache already exists (run_goal2_cache_slurm.sh done).
#
# Default PRESET=stim_duringstim_bayes (4 splits, 150 ms, choice × Bayes).
# PRESET=stim_lr_bayes_all also includes stim_block_{l,r}_bayes (80 ms).
#
# Memory (override with MEM_SHARD / MEM_FIN):
#   Peak RSS stream_pool nrand=2000 ≈ 1.5–2.5 GB (journal 07-10b).
#   Defaults: MEM_SHARD=6G, MEM_FIN=10G.
# Override: N_SHARDS=3 MEM_SHARD=8G PARTITION=mit_normal \
#   bash scripts/submit_goal2_stim_lr_bayes_sharded.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

PRESET="${PRESET:-stim_duringstim_bayes}"
N_SHARDS="${N_SHARDS:-4}"
NRAND="${NRAND:-2000}"
RESTART="${RESTART:-1}"
MEM_SHARD="${MEM_SHARD:-6G}"
MEM_FIN="${MEM_FIN:-10G}"
CPUS_SHARD="${CPUS_SHARD:-2}"
CPUS_FIN="${CPUS_FIN:-2}"
PARTITION="${PARTITION:-pi_fiete}"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/sbatch_defaults.sh"

ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"
export SESSION_SHUFFLE_NULL=0 ACTKERNEL_CHOICE_NULL=0

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
echo "NULL_SCHEME=label_shuffle (plain)  PRESET=$PRESET"
echo "N_SHARDS=$N_SHARDS  nrand=$NRAND  splits=${#SPLITS[@]}  shard_jobs=$n_shard_jobs"
echo "PARTITION=$PARTITION  MEM_SHARD=$MEM_SHARD  MEM_FIN=$MEM_FIN"
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
    # shellcheck disable=SC2086
    JID=$(sbatch --parsable $SBATCH_EXTRA \
      --partition="$PARTITION" \
      --mem="$MEM_SHARD" --cpus-per-task="$CPUS_SHARD" \
      --job-name="g2_${TAG}_s${k}" \
      --export=ALL,SPLIT="$sp",SHARD_IDX="$k",N_SHARDS="$N_SHARDS",NRAND="$NRAND",RESTART="$RESTART",SESSION_SHUFFLE_NULL=0,ACTKERNEL_CHOICE_NULL=0 \
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
    --job-name="g2_fin_${TAG}" \
    --export=ALL,SPLIT="$sp",SESSION_SHUFFLE_NULL=0,ACTKERNEL_CHOICE_NULL=0 \
    scripts/run_goal2_finalize_slurm.sh)
  echo "  $sp finalize -> $FID (after $DEP)"
done

echo "Done. Monitor: squeue -u \$USER"
echo "Shard outputs: \$ONE_CACHE_DIR/manifold/res/_stream_acc/{split}.shard{k}.npy"
echo "Final outputs: \$ONE_CACHE_DIR/manifold/res/{split}.npy"
