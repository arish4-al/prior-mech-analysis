#!/bin/bash
# Plain label shuffle inside Bayes-prior stratum, duringstim only (4+4).
#
#   stim   — stim L vs R, 4 splits: stim_choice_{r,l}_block_{r,l}_bayes
#            stratum = choice × Bayes-prior; shuffle stim labels in that mask
#   choice — choice L vs R, 4 splits: choice_duringstim_{r,l}_block_{r,l}_bayes
#            stratum = stim × Bayes-prior; shuffle choice labels in that mask
#
# Not Harris / sticky. Writes $ONE_CACHE_DIR/manifold/res/{split}.npy
# (CLEAR_STREAM=1 default: wipes only those plain basenames).
#
#   bash scripts/submit_goal2_bayes_shuffle_orcd.sh
#   FAMILY=stim|choice|all
#   PARTITION=mit_normal bash scripts/submit_goal2_bayes_shuffle_orcd.sh
#
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

FAMILY="${FAMILY:-all}"
NRAND="${NRAND:-2000}"
N_SHARDS="${N_SHARDS:-4}"
PARTITION="${PARTITION:-pi_fiete}"
CLEAR_STREAM="${CLEAR_STREAM:-1}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"
export NRAND N_SHARDS PARTITION CLEAR_STREAM

if [[ "$FAMILY" != "stim" && "$FAMILY" != "choice" && "$FAMILY" != "all" ]]; then
  echo "ERROR: FAMILY must be stim|choice|all (got $FAMILY)" >&2
  exit 1
fi

_shuffle() {
  local preset="$1"
  echo "=== label shuffle  PRESET=$preset ==="
  PRESET="$preset" \
    SESSION_SHUFFLE_NULL=0 ACTKERNEL_CHOICE_NULL=0 \
    bash scripts/submit_goal2_choice_shuffle_sharded.sh
}

if [[ "$FAMILY" == "stim" || "$FAMILY" == "all" ]]; then
  _shuffle stim_duringstim_bayes
fi
if [[ "$FAMILY" == "choice" || "$FAMILY" == "all" ]]; then
  _shuffle choice_lr_session_null_bayes_duringstim
fi

echo "Outputs: \$ONE_CACHE_DIR/manifold/res/{split}.npy  (no suffix)"
echo "Monitor: squeue -u \$USER"
