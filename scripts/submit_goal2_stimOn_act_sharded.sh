#!/bin/bash
# Submit sharded Goal 2 stimOn_times_act jobs.
# Default: 4 shards/split × 6 splits = 24 shard jobs + 6 finalize jobs.
#
#   bash scripts/submit_goal2_stimOn_act_sharded.sh
#
# Assumes insertion cache already exists (run_goal2_cache_slurm.sh done).
# Override: N_SHARDS=3 bash scripts/submit_goal2_stimOn_act_sharded.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

N_SHARDS="${N_SHARDS:-4}"
SPLITS=(
  act_block_duringstim_r_choice_r_f1
  act_block_duringstim_l_choice_l_f1
  act_block_duringstim_l_choice_r_f2
  act_block_duringstim_r_choice_l_f2
  act_block_stim_l
  act_block_stim_r
)

echo "N_SHARDS=$N_SHARDS  splits=${#SPLITS[@]}"

for sp in "${SPLITS[@]}"; do
  SHARD_JOBS=()
  for ((k=0; k<N_SHARDS; k++)); do
    JID=$(sbatch --parsable \
      --job-name="g2_${sp}_s${k}" \
      --export=ALL,SPLIT="$sp",SHARD_IDX="$k",N_SHARDS="$N_SHARDS" \
      scripts/run_goal2_shard_slurm.sh)
    SHARD_JOBS+=("$JID")
    echo "  $sp shard $k/$N_SHARDS -> $JID"
  done
  # finalize after all shards for this split
  DEP=$(IFS=:; echo "${SHARD_JOBS[*]}")
  FID=$(sbatch --parsable \
    --dependency=afterok:"$DEP" \
    --job-name="g2_fin_${sp}" \
    --export=ALL,SPLIT="$sp" \
    scripts/run_goal2_finalize_slurm.sh)
  echo "  $sp finalize -> $FID (after $DEP)"
done

echo "Done. Monitor: squeue -u \$USER"
