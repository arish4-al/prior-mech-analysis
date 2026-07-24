#!/bin/bash
# Rerun stratified AK choice null with longer BWM pseudos (default 3×).
# Always writes {split}_pseudo_strat*.npy (clears prior stream_acc + pooled res).
# Control loop doubles factor up to 16 if accept rate is still too low.
#
#   bash scripts/submit_goal2_choice_strat_x3_sharded.sh
#   SMOKE_FIRST=1 bash scripts/submit_goal2_choice_strat_x3_sharded.sh
#
# Override: PSEUDO_LEN_FACTOR=4 bash scripts/submit_goal2_choice_strat_x3_sharded.sh

set -euo pipefail

export NULL_SCHEME=pseudo_strat
export PSEUDO_LEN_FACTOR="${PSEUDO_LEN_FACTOR:-3}"
export ACTKERNEL_PSEUDO_LEN_FACTOR="$PSEUDO_LEN_FACTOR"
export CLEAR_STREAM="${CLEAR_STREAM:-1}"

exec bash scripts/submit_goal2_choice_null_sharded.sh
