#!/bin/bash
# Harris unique-null for Bayes prior (not Bayes-agent sticky).
#
#   local   — 6 splits that have laptop shuffle maps (default):
#             4 duringstim stim×choice + 2 stim-side unsplit
#   prior   — 8 stim×choice (4 duringstim + 4 duringchoice)
#   unsplit — 2 stim-side only: bayes_block_duringstim_{l,r}
#   choice  — 8 choice_*_bayes
#
# Disk: {split}_harris_unique.npy  (does not touch act_block_* / *_act / sticky).
# Donor labels: Bayes-binary from stim history (same as analysis overwrite).
#
#   bash scripts/submit_goal2_bayes_harris_orcd.sh
#   FAMILY=local|prior|unsplit|choice|all
#   PARTITION=mit_preemptable bash scripts/submit_goal2_bayes_harris_orcd.sh
#     --requeue is the default on mit_preemptable / mit_preem (sbatch_defaults.sh)
#   If you re-run this wrapper after a crash: CLEAR_STREAM=0
#
# Label shuffle (stim L–R / choice L–R duringstim, Bayes stratum):
#   bash scripts/submit_goal2_bayes_shuffle_orcd.sh
#
# Not included in default local: duringchoice prior, choice Harris,
# choice-side unsplit, fully unsplit, Bayes-agent sticky.
#
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

FAMILY="${FAMILY:-local}"
NRAND="${NRAND:-2000}"
N_SHARDS="${N_SHARDS:-4}"
PARTITION="${PARTITION:-pi_fiete}"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/sbatch_defaults.sh"

REBUILD_DONORS="${REBUILD_DONORS:-1}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"
export NRAND N_SHARDS PARTITION
RESTART="${RESTART:-1}"
CLEAR_STREAM="${CLEAR_STREAM:-1}"
export RESTART CLEAR_STREAM

if [[ "$FAMILY" != "local" && "$FAMILY" != "choice" && "$FAMILY" != "prior" \
      && "$FAMILY" != "unsplit" && "$FAMILY" != "all" ]]; then
  echo "ERROR: FAMILY must be local|choice|prior|unsplit|all (got $FAMILY)" >&2
  exit 1
fi

TIME_SHARD="${TIME_SHARD:-12:00:00}"
MEM_DONORS="${MEM_DONORS:-8G}"
CPUS_DONORS="${CPUS_DONORS:-2}"

DONOR_JID="${DONOR_JID:-}"
if [[ -z "$DONOR_JID" && "$REBUILD_DONORS" == "1" ]]; then
  # shellcheck disable=SC2086
  DONOR_JID=$(sbatch --parsable $SBATCH_EXTRA \
    --partition="$PARTITION" \
    --mem="$MEM_DONORS" --cpus-per-task="$CPUS_DONORS" \
    --job-name="g2_choice_donors" \
    --export=ALL,FORCE_REBUILD=1 \
    scripts/run_goal2_choice_donors_slurm.sh)
  echo "choice donors job -> $DONOR_JID"
fi
export DONOR_JID
export REBUILD_DONORS=0

_prior() {
  local preset="$1" prefix="$2"
  echo "=== Harris unique  PRESET=$preset  JOB_PREFIX=$prefix ==="
  PRESET="$preset" JOB_PREFIX="$prefix" TIME_SHARD="$TIME_SHARD" \
    REBUILD_DONORS=0 DONOR_JID="$DONOR_JID" \
    bash scripts/submit_goal2_act_block_harris_sharded.sh
}

_choice() {
  local preset="$1" prefix="$2"
  echo "=== Harris unique  PRESET=$preset  JOB_PREFIX=$prefix ==="
  PRESET="$preset" JOB_PREFIX="$prefix" TIME_SHARD="$TIME_SHARD" \
    DONOR_JID="$DONOR_JID" \
    bash scripts/submit_goal2_choice_session_null_sharded.sh
}

if [[ "$FAMILY" == "local" ]]; then
  _prior bayes_block_duringstim g2bbh
  _prior bayes_block_unsplit_duringstim g2bbu
fi
if [[ "$FAMILY" == "prior" || "$FAMILY" == "all" ]]; then
  _prior bayes_block_harris g2bbh
fi
if [[ "$FAMILY" == "unsplit" || "$FAMILY" == "all" ]]; then
  _prior bayes_block_unsplit_duringstim g2bbu
fi
if [[ "$FAMILY" == "choice" || "$FAMILY" == "all" ]]; then
  _choice choice_lr_session_null_bayes g2hub
fi

echo "Outputs: \$ONE_CACHE_DIR/manifold/res/{split}_harris_unique.npy"
echo "Monitor: squeue -u \$USER"
