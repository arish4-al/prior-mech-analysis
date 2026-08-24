#!/bin/bash
# Option 1 + copy-last: AK pseudo-sessions, remade stratum, sticky choices.
#
# Choice L–R: pseudo stim × that session's act-prior; labels = sticky choices.
# Act_block:  pseudo stim × generated choices; labels = that session's act-prior.
# Disk: {split}_pseudo_strat_sticky.npy  (does not overwrite _pseudo_strat).
#
#   bash scripts/submit_goal2_ak_sticky_orcd.sh
#   PREFIT=0 bash scripts/submit_goal2_ak_sticky_orcd.sh   # skip prefit
#   FAMILY=choice|act_block|act_block_unsplit|both  (default both)
#   PREFIT=0 FAMILY=act_block_unsplit bash scripts/submit_goal2_ak_sticky_orcd.sh
#   PARTITION=mit_normal bash scripts/submit_goal2_ak_sticky_orcd.sh
#
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

PREFIT="${PREFIT:-1}"
FAMILY="${FAMILY:-both}"
NRAND="${NRAND:-2000}"
N_SHARDS="${N_SHARDS:-4}"
PARTITION="${PARTITION:-pi_fiete}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"
export NRAND N_SHARDS PARTITION

PREFIT_JID=""
if [[ "$PREFIT" == "1" ]]; then
  echo "=== Prefit ActionKernel (once per eid) ==="
  PREFIT_JID=$(sbatch --parsable \
    --partition="$PARTITION" \
    --job-name=g2_ak_prefit \
    --mem="${MEM_PREFIT:-8G}" --cpus-per-task=2 --time="${TIME_PREFIT:-8:00:00}" \
    --export=ALL \
    scripts/run_goal2_ak_prefit_slurm.sh)
  echo "Prefit -> $PREFIT_JID (shard jobs afterok this JID)"
fi
export PREFIT_JID

export NULL_SCHEME=pseudo_strat_sticky

if [[ "$FAMILY" != "choice" && "$FAMILY" != "act_block" \
      && "$FAMILY" != "act_block_unsplit" && "$FAMILY" != "both" ]]; then
  echo "ERROR: FAMILY must be choice|act_block|act_block_unsplit|both (got $FAMILY)" >&2
  exit 1
fi

if [[ "$FAMILY" == "choice" || "$FAMILY" == "both" ]]; then
  echo "=== Choice L–R (stim + move) option-1 + copy-last ==="
  PRESET=choice_lr_ak_sticky bash scripts/submit_goal2_choice_null_sharded.sh
fi
if [[ "$FAMILY" == "act_block" || "$FAMILY" == "both" ]]; then
  echo "=== act_block prior L–R f1/f2 + unsplit (stim + move) option-1 + copy-last ==="
  PRESET=act_block_ak_sticky bash scripts/submit_goal2_choice_null_sharded.sh
fi
if [[ "$FAMILY" == "act_block_unsplit" ]]; then
  echo "=== act_block unsplit only (stim-side + choice-side) option-1 + copy-last ==="
  PRESET=act_block_ak_sticky_unsplit bash scripts/submit_goal2_choice_null_sharded.sh
fi

echo "Outputs: \$ONE_CACHE_DIR/manifold/res/{split}_pseudo_strat_sticky.npy"
