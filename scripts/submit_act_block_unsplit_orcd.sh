#!/bin/bash
# Real-data unsplit act_block Harris unique-null (+ optional shuffle) on ORCD.
#
# Stim-aligned: act_block_duringstim_{l,r}  — stim strata, no f1/f2
# Move-aligned: act_block_duringchoice_{l,r} — choice strata, no f1/f2
#
#   bash scripts/submit_act_block_unsplit_orcd.sh
#
# SHUFFLE_DURINGCHOICE=1 (default) also submits plain shuffle for the new
# duringchoice pair only (duringstim unsplit shuffle already exists on alyx).

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

PARTITION="${PARTITION:-pi_fiete}"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/sbatch_defaults.sh"

SHUFFLE_DURINGCHOICE="${SHUFFLE_DURINGCHOICE:-1}"
export NRAND="${NRAND:-1000}"

echo "=== Harris unique-null: act_block_harris_unsplit (nrand=$NRAND) ==="
bash scripts/submit_goal2_act_block_harris_unsplit_sharded.sh

if [[ "$SHUFFLE_DURINGCHOICE" == "1" ]]; then
  echo "=== Shuffle baseline: act_block_unsplit_duringchoice only ==="
  PRESET=act_block_unsplit_duringchoice \
    bash scripts/submit_goal2_choice_shuffle_sharded.sh
fi

echo "Monitor: squeue -u \$USER"
echo "Harris: \$ONE_CACHE_DIR/manifold/res/{split}_harris_unique.npy"
echo "Shuffle duringchoice: \$ONE_CACHE_DIR/manifold/res/act_block_duringchoice_{l,r}.npy"
