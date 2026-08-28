#!/bin/bash
# Harris unique-null session-permutation (donor re-filtered to same stim×prior).
# Writes {split}_harris_unique*.npy — does NOT overwrite legacy _harris.
# Clears prior _harris_unique stream_acc + pooled res, then submits shards.
#
#   bash scripts/submit_goal2_choice_session_null_sharded.sh
#   PRESET=choice_lr_session_null_true bash scripts/submit_goal2_choice_session_null_sharded.sh
#   PRESET=choice_lr_session_null_bayes bash scripts/submit_goal2_choice_session_null_sharded.sh
#   SMOKE_FIRST=1 bash scripts/submit_goal2_choice_session_null_sharded.sh

set -euo pipefail

export NULL_SCHEME=harris_unique
export CLEAR_STREAM="${CLEAR_STREAM:-1}"
export PRESET="${PRESET:-choice_lr_session_null_all}"

exec bash scripts/submit_goal2_choice_null_sharded.sh
