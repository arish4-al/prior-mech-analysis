#!/bin/bash
# Submit sharded Goal 2 jobs for duringstim stim-side + bayes duringstim splits.
# Default: 4 shards/split × 8 splits = 32 shard jobs + 8 finalize jobs.
#
#   bash scripts/submit_goal2_stimOn_act_sharded.sh
#
# Assumes insertion cache already exists (run_goal2_cache_slurm.sh done).
#
# Memory (override with MEM_SHARD / MEM_FIN):
#   Peak RSS stream_pool nrand=2000 ≈ 1.5–2.5 GB (journal 07-10b).
#   Defaults: MEM_SHARD=6G, MEM_FIN=10G (~2–4× headroom).
#   Concurrent if all shards pend/run: 8×4×6G = 192G (was 12G → 384G).
# Override: N_SHARDS=3 MEM_SHARD=8G bash scripts/submit_goal2_stimOn_act_sharded.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

N_SHARDS="${N_SHARDS:-4}"
MEM_SHARD="${MEM_SHARD:-6G}"
MEM_FIN="${MEM_FIN:-10G}"
CPUS_SHARD="${CPUS_SHARD:-2}"
CPUS_FIN="${CPUS_FIN:-2}"
SPLITS=(
  # act: stim-side only (no choice×feedback), post-stim [0, 0.15]
  act_block_duringstim_l
  act_block_duringstim_r
  # bayes: choice×feedback duringstim
  bayes_block_duringstim_r_choice_r_f1
  bayes_block_duringstim_l_choice_l_f1
  bayes_block_duringstim_l_choice_r_f2
  bayes_block_duringstim_r_choice_l_f2
  # bayes: stim-side only
  bayes_block_duringstim_l
  bayes_block_duringstim_r
)

n_shard_jobs=$(( ${#SPLITS[@]} * N_SHARDS ))
echo "N_SHARDS=$N_SHARDS  splits=${#SPLITS[@]}  shard_jobs=$n_shard_jobs"
echo "MEM_SHARD=$MEM_SHARD  MEM_FIN=$MEM_FIN  CPUS_SHARD=$CPUS_SHARD  CPUS_FIN=$CPUS_FIN"
echo "Concurrent mem if all shards run: ${n_shard_jobs} × $MEM_SHARD"

for sp in "${SPLITS[@]}"; do
  SHARD_JOBS=()
  for ((k=0; k<N_SHARDS; k++)); do
    JID=$(sbatch --parsable \
      --mem="$MEM_SHARD" --cpus-per-task="$CPUS_SHARD" \
      --job-name="g2_${sp}_s${k}" \
      --export=ALL,SPLIT="$sp",SHARD_IDX="$k",N_SHARDS="$N_SHARDS" \
      scripts/run_goal2_shard_slurm.sh)
    SHARD_JOBS+=("$JID")
    echo "  $sp shard $k/$N_SHARDS -> $JID"
  done
  # finalize after all shards for this split
  DEP=$(IFS=:; echo "${SHARD_JOBS[*]}")
  FID=$(sbatch --parsable \
    --mem="$MEM_FIN" --cpus-per-task="$CPUS_FIN" \
    --dependency=afterok:"$DEP" \
    --job-name="g2_fin_${sp}" \
    --export=ALL,SPLIT="$sp" \
    scripts/run_goal2_finalize_slurm.sh)
  echo "  $sp finalize -> $FID (after $DEP)"
done

echo "Done. Monitor: squeue -u \$USER"
