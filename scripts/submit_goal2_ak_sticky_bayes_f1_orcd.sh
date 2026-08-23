#!/bin/bash
# New campaign: same-side f1 prior L–R with Bayes labels, option 1 + copy-last.
#
# Stratum is still stim × generated sticky choices (same as act_block f1).
# Null labels are Bayes-binary from that draw's stim history.
# Disk: {bayes_block_*}_pseudo_strat_sticky.npy  (does not touch act_block_*).
#
# Full 4 shards × 4 splits, start at factor 6 (same n_elig / 2-D remake as
# the act f1 jobs that died at factor 3). Prefit pickles reused (PREFIT=0).
#
#   bash scripts/submit_goal2_ak_sticky_bayes_f1_orcd.sh
#
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

export NULL_SCHEME=pseudo_strat_sticky
export PRESET=bayes_block_ak_sticky_f1
export N_SHARDS="${N_SHARDS:-4}"
export NRAND="${NRAND:-2000}"
export PSEUDO_LEN_FACTOR="${PSEUDO_LEN_FACTOR:-6}"
export TIME_SHARD="${TIME_SHARD:-8:00:00}"
export PREFIT_JID=""
export ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"

echo "=== Bayes f1 option-1 + copy-last (new; not a resume of act_block) ==="
bash scripts/submit_goal2_choice_null_sharded.sh
echo "Outputs: \$ONE_CACHE_DIR/manifold/res/bayes_block_*_pseudo_strat_sticky.npy"
