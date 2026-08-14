#!/bin/bash
# Submit sharded Goal 2 jobs: Harris unique-null for unsplit act_block prior L–R.
#
#   bash scripts/submit_goal2_act_block_harris_unsplit_sharded.sh
#
# Default PRESET=act_block_harris_unsplit (4 splits; model analog of stim-side /
# choice-side unsplit — no f1/f2):
#   stimOn (2):  act_block_duringstim_{l,r}     — stratify by stim only
#   movement (2): act_block_duringchoice_{l,r}  — stratify by choice only
#
# Subsets:
#   PRESET=act_block_unsplit_duringstim bash scripts/submit_goal2_act_block_harris_unsplit_sharded.sh
#   PRESET=act_block_unsplit_duringchoice …
#
# Outputs: $ONE_CACHE_DIR/manifold/res/{split}_harris_unique.npy
#   (plain shuffle stays at {split}.npy — never overwritten)
# Rebuilds donor bank first (includes contrast_left/right).
#
# CLEAR_STREAM=1 (default) clears only *_harris_unique for these splits.
#
# Optional shuffle baseline for the *new* duringchoice unsplit splits only
# (duringstim unsplit shuffle already exists on alyx — do not CLEAR those):
#   PRESET=act_block_unsplit_duringchoice bash scripts/submit_goal2_choice_shuffle_sharded.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

PRESET="${PRESET:-act_block_harris_unsplit}" \
  JOB_PREFIX="${JOB_PREFIX:-g2ahu}" \
  bash scripts/submit_goal2_act_block_harris_sharded.sh
