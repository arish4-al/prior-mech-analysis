#!/bin/bash
# Thin wrapper: Harris unique-null, Bayes prior L–R stim-side only (2 splits).
#   bayes_block_duringstim_{l,r}  — no choice / f1/f2
# Prefer:
#   FAMILY=unsplit bash scripts/submit_goal2_bayes_harris_orcd.sh
set -euo pipefail
PRESET="${PRESET:-bayes_block_unsplit_duringstim}" \
  JOB_PREFIX="${JOB_PREFIX:-g2bbu}" \
  bash scripts/submit_goal2_act_block_harris_sharded.sh
