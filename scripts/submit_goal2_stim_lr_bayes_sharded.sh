#!/bin/bash
# Submit sharded jobs for preset stim_lr_bayes_all (stim L vs R under Bayes prior).
# Default: 4 shards/split × 6 splits = 24 shard jobs + 6 finalize jobs.
#
#   bash scripts/submit_goal2_stim_lr_bayes_sharded.sh
#
# Assumes insertion cache already exists (run_goal2_cache_slurm.sh done).
#
# Splits (same as run_goal2_splits.py --preset stim_lr_bayes_all):
#   stim_choice_{r,l}_block_{r,l}_bayes  — 150 ms; fixed choice + Bayes prior
#   stim_block_{l,r}_bayes               — 80 ms; fixed Bayes prior only
#
# Memory (override with MEM_SHARD / MEM_FIN):
#   Peak RSS stream_pool nrand=2000 ≈ 1.5–2.5 GB (journal 07-10b).
#   Defaults: MEM_SHARD=6G, MEM_FIN=10G.
# Override: N_SHARDS=3 MEM_SHARD=8G bash scripts/submit_goal2_stim_lr_bayes_sharded.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

N_SHARDS="${N_SHARDS:-4}"
NRAND="${NRAND:-2000}"
RESTART="${RESTART:-1}"
MEM_SHARD="${MEM_SHARD:-6G}"
MEM_FIN="${MEM_FIN:-10G}"
CPUS_SHARD="${CPUS_SHARD:-2}"
CPUS_FIN="${CPUS_FIN:-2}"

SPLITS=(
  # stim L vs R | fixed choice + Bayes prior | [0, 0.15]
  stim_choice_r_block_r_bayes
  stim_choice_l_block_l_bayes
  stim_choice_r_block_l_bayes
  stim_choice_l_block_r_bayes
  # stim L vs R | fixed Bayes prior only | [0, 0.08]
  stim_block_l_bayes
  stim_block_r_bayes
)

n_shard_jobs=$(( ${#SPLITS[@]} * N_SHARDS ))
echo "preset=stim_lr_bayes_all  N_SHARDS=$N_SHARDS  nrand=$NRAND  splits=${#SPLITS[@]}"
echo "MEM_SHARD=$MEM_SHARD  MEM_FIN=$MEM_FIN  CPUS_SHARD=$CPUS_SHARD  CPUS_FIN=$CPUS_FIN"
echo "shard_jobs=$n_shard_jobs  Concurrent mem if all shards run: ${n_shard_jobs} × $MEM_SHARD"

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
      --job-name="g2_${TAG}_s${k}" \
      --export=ALL,SPLIT="$sp",SHARD_IDX="$k",N_SHARDS="$N_SHARDS",NRAND="$NRAND",RESTART="$RESTART" \
      scripts/run_goal2_shard_slurm.sh)
    SHARD_JOBS+=("$JID")
    echo "  $sp shard $k/$N_SHARDS -> $JID"
  done
  DEP=$(IFS=:; echo "${SHARD_JOBS[*]}")
  FID=$(sbatch --parsable \
    --mem="$MEM_FIN" --cpus-per-task="$CPUS_FIN" \
    --dependency=afterok:"$DEP" \
    --job-name="g2_fin_${TAG}" \
    --export=ALL,SPLIT="$sp" \
    scripts/run_goal2_finalize_slurm.sh)
  echo "  $sp finalize -> $FID (after $DEP)"
done

echo "Done. Monitor: squeue -u \$USER"
echo "Shard outputs: \$ONE_CACHE_DIR/manifold/res/_stream_acc/{split}.shard{k}.npy"
echo "Final outputs: \$ONE_CACHE_DIR/manifold/res/{split}.npy"
