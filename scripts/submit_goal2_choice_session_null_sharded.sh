#!/bin/bash
# Harris session-permutation nulls (donor re-filtered to same stim×prior stratum).
# Clears prior _harris stream_acc + pooled res, then submits shards.
#
#   bash scripts/submit_goal2_choice_session_null_sharded.sh
#   PRESET=choice_lr_session_null_true bash scripts/submit_goal2_choice_session_null_sharded.sh
#   SMOKE_FIRST=1 bash scripts/submit_goal2_choice_session_null_sharded.sh

set -euo pipefail

export NULL_SCHEME=harris
export CLEAR_STREAM="${CLEAR_STREAM:-1}"
export PRESET="${PRESET:-choice_lr_session_null_all}"

exec bash scripts/submit_goal2_choice_null_sharded.sh
