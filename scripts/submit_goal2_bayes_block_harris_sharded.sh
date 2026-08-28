#!/bin/bash
# Thin wrapper: Harris unique-null, Bayes prior L–R stim×choice duringstim (4 splits).
# Matches local shuffle maps. Duringchoice is FAMILY=prior on the analog script.
# Prefer:
#   bash scripts/submit_goal2_bayes_harris_orcd.sh
set -euo pipefail
PRESET="${PRESET:-bayes_block_duringstim}" \
  JOB_PREFIX="${JOB_PREFIX:-g2bbh}" \
  bash scripts/submit_goal2_act_block_harris_sharded.sh
