#!/bin/bash
# Submit Stage B unsplit Harris unique-null jobs on ORCD.
#
# S/I: stim-aligned stim strata. M: move-aligned choice strata.
# Do not run this on the laptop (session_cache is multi-GB).
#
#   bash scripts/submit_simulate_unsplit_harris_orcd.sh
#
# Two jobs: regular s101 (absence) and sensory s23 (s_presence).
# Override: N_SESSIONS=40 BLOCKS_PER_SESSION=40 NRAND=1000 HARRIS_N_EXTRA_DONORS=80

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

export ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export OUTPUT_DIR="${OUTPUT_DIR:-$ONE_CACHE_DIR/manifold_sim/stageB_bwm/unsplit_stim_choice}"
export SEED="${SEED:-123}"
export N_SESSIONS="${N_SESSIONS:-40}"
export BLOCKS_PER_SESSION="${BLOCKS_PER_SESSION:-40}"
export NRAND="${NRAND:-1000}"
export HARRIS_N_EXTRA_DONORS="${HARRIS_N_EXTRA_DONORS:-80}"
export S_WINDOW_MS="${S_WINDOW_MS:-150}"
export PRIOR_MECH_NO_ONE="${PRIOR_MECH_NO_ONE:-1}"

echo "ORCD unsplit Harris unique-null (stim strata @ stimOn; choice strata @ movement)"
echo "OUTPUT_DIR=$OUTPUT_DIR  blocks=$BLOCKS_PER_SESSION nrand=$NRAND extra=$HARRIS_N_EXTRA_DONORS s_window_ms=$S_WINDOW_MS"

J_REG=$(sbatch --parsable \
  --job-name=sim_uhu_reg \
  --export=ALL,VARIANT=regular,CASE=absence \
  scripts/run_simulate_unsplit_harris_slurm.sh)
echo "  regular absence -> $J_REG"

J_SENS=$(sbatch --parsable \
  --job-name=sim_uhu_sens \
  --export=ALL,VARIANT=sensory,CASE=s_presence \
  scripts/run_simulate_unsplit_harris_slurm.sh)
echo "  sensory s_presence -> $J_SENS"

echo "Monitor: squeue -u \$USER"
echo "When done: $OUTPUT_DIR/unsplit_prior/seed_${SEED}/"
