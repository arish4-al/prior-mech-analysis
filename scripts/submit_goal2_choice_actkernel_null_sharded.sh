#!/bin/bash
# Backward-compatible wrapper → stratified AK null (journal opt 1).
# Prefer: NULL_SCHEME=pseudo_strat bash scripts/submit_goal2_choice_null_sharded.sh
# Or all three: bash scripts/submit_goal2_choice_null_all_schemes_sharded.sh

set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"
NULL_SCHEME="${NULL_SCHEME:-pseudo_strat}" \
  bash scripts/submit_goal2_choice_null_sharded.sh
