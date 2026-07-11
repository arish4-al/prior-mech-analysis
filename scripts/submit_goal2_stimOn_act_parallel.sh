#!/bin/bash
# Submit Goal 2 stimOn_times_act: cache job, then 6 parallel split jobs.
#
#   bash scripts/submit_goal2_stimOn_act_parallel.sh
#
# Requires insertion cache before split jobs (sbatch --dependency).

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

SPLITS=(
  act_block_duringstim_r_choice_r_f1
  act_block_duringstim_l_choice_l_f1
  act_block_duringstim_l_choice_r_f2
  act_block_duringstim_r_choice_l_f2
  act_block_stim_l
  act_block_stim_r
)

echo "Submitting insertion cache job..."
CACHE_JOB=$(sbatch --parsable scripts/run_goal2_cache_slurm.sh)
echo "  cache job: $CACHE_JOB"

for sp in "${SPLITS[@]}"; do
  JOB=$(sbatch --parsable --dependency=afterok:"$CACHE_JOB" \
    --job-name="g2_${sp}" \
    --export=ALL,SPLIT="$sp" \
    scripts/run_goal2_one_split_slurm.sh)
  echo "  split $sp: $JOB (after cache $CACHE_JOB)"
done

echo "Done. Monitor: squeue -u \$USER"
