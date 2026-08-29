#!/bin/bash
# Option 1 + copy-last, Bayes *mouse* (not AK + Bayes analysis labels).
#
# New BWM stim/blocks; choices from IBL OptimalBayesian (fixed τ,γ,ζ,lapse)
# then copy-last. Remake:
#   Choice L–R *_bayes: stim × Bayes-prior; labels = sticky Bayes-agent choices.
#   Bayes_block: stim × those choices; labels = Bayes-binary from stim.
# Disk: {split}_pseudo_strat_sticky.npy  (does not touch act_block_* / *_act).
# Does not load / fit ActionKernel.
#
# Factor: f1 starts at 6 (same 2-D stim×choice + large n_elig that killed
# act f1 at factor 3). Choice / f2 / unsplit stay at 3 (0 bumps on the act run).
#
#   bash scripts/submit_goal2_ak_sticky_bayes_orcd.sh
#   FAMILY=choice|bayes_block|f1|f2|unsplit|both   (default both)
#   PARTITION=mit_normal bash scripts/submit_goal2_ak_sticky_bayes_orcd.sh
#
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

FAMILY="${FAMILY:-both}"
NRAND="${NRAND:-2000}"
N_SHARDS="${N_SHARDS:-4}"
PARTITION="${PARTITION:-pi_fiete}"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/sbatch_defaults.sh"

ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"
export NRAND N_SHARDS PARTITION PREFIT_JID="${PREFIT_JID:-}"
export NULL_SCHEME=pseudo_strat_sticky

if [[ "$FAMILY" != "choice" && "$FAMILY" != "bayes_block" \
      && "$FAMILY" != "f1" && "$FAMILY" != "f2" \
      && "$FAMILY" != "unsplit" && "$FAMILY" != "both" ]]; then
  echo "ERROR: FAMILY must be choice|bayes_block|f1|f2|unsplit|both (got $FAMILY)" >&2
  exit 1
fi

_run() {
  local preset="$1" factor="$2" time_shard="$3"
  echo "=== PRESET=$preset  factor=$factor  TIME_SHARD=$time_shard ==="
  PRESET="$preset" PSEUDO_LEN_FACTOR="$factor" TIME_SHARD="$time_shard" \
    bash scripts/submit_goal2_choice_null_sharded.sh
}

# Choice / f2 / unsplit: act campaign finished at factor 3 with no bumps.
TIME_EASY="${TIME_EASY:-6:00:00}"
# f1: act campaign needed 3→6 (often 0/2000 at factor 3). Start at 6.
TIME_F1="${TIME_F1:-8:00:00}"

if [[ "$FAMILY" == "choice" || "$FAMILY" == "both" ]]; then
  _run choice_lr_ak_sticky_bayes 3 "$TIME_EASY"
fi
if [[ "$FAMILY" == "f1" || "$FAMILY" == "bayes_block" || "$FAMILY" == "both" ]]; then
  _run bayes_block_ak_sticky_f1 6 "$TIME_F1"
fi
if [[ "$FAMILY" == "f2" || "$FAMILY" == "bayes_block" || "$FAMILY" == "both" ]]; then
  _run bayes_block_ak_sticky_f2 3 "$TIME_EASY"
fi
if [[ "$FAMILY" == "unsplit" || "$FAMILY" == "bayes_block" || "$FAMILY" == "both" ]]; then
  _run bayes_block_ak_sticky_unsplit 3 "$TIME_EASY"
fi

echo "Outputs: \$ONE_CACHE_DIR/manifold/res/{split}_pseudo_strat_sticky.npy"
