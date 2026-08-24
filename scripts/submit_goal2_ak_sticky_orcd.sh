#!/bin/bash
# CLEAN option-1 + copy-last run: AK pseudo-sessions, remade stratum, sticky choices.
# Covers ALL choice L–R + act_block prior L–R splits (f1/f2 + unsplit).
#
# Choice L–R: pseudo stim × that session's act-prior; labels = sticky choices.
# Act_block:  pseudo stim × generated choices; labels = that session's act-prior,
#             with the pseudo's pLeft=0.5 block dropped (matches real path).
# Disk: {split}_pseudo_strat_sticky.npy  (does not overwrite _pseudo_strat).
#
# Post-2026-08-24 fixes baked in:
#   - AK choice sign convention corrected (was L↔R flipped).
#   - pseudo drops its own pLeft=0.5 warm-up; additive sizing, default factor 1
#     (biased pseudo length ≈ real session; +~90 pad in code; auto-grows if short).
# All pre-08-24 {split}_pseudo_strat_sticky.npy are INVALID → this run CLEARS them
# (CLEAR_STREAM=1) and starts from scratch (RESTART=0).
#
#   bash scripts/submit_goal2_ak_sticky_orcd.sh            # clean full run (both families)
#   PREFIT=0 bash scripts/submit_goal2_ak_sticky_orcd.sh   # skip prefit (fits cached)
#   FAMILY=choice|act_block|act_block_unsplit|both  (default both)
#   CLEAR_STREAM=0 RESTART=1 bash scripts/submit_goal2_ak_sticky_orcd.sh  # resume instead
#   PARTITION=mit_normal bash scripts/submit_goal2_ak_sticky_orcd.sh
#
# Bayes prior/choice splits are a SEPARATE campaign — they also need a clean rerun
# (the 0.5-drop / sizing fixes touch bayes_block too): scripts/submit_goal2_ak_sticky_bayes_orcd.sh
#
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

PREFIT="${PREFIT:-1}"
FAMILY="${FAMILY:-both}"
NRAND="${NRAND:-2000}"
N_SHARDS="${N_SHARDS:-4}"
PARTITION="${PARTITION:-pi_fiete}"
# Clean run by default: clear prior _pseudo_strat_sticky stream_acc + res, no resume.
CLEAR_STREAM="${CLEAR_STREAM:-1}"
RESTART="${RESTART:-0}"
# Factor unset → code default 1 (additive sizing). Override to pin a multiple.
PSEUDO_LEN_FACTOR="${PSEUDO_LEN_FACTOR:-}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"
export NRAND N_SHARDS PARTITION CLEAR_STREAM RESTART PSEUDO_LEN_FACTOR

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

echo "=== CLEAN option-1 + copy-last run (post-08-24 sign/0.5/sizing fixes) ==="
echo "FAMILY=$FAMILY  CLEAR_STREAM=$CLEAR_STREAM  RESTART=$RESTART  nrand=$NRAND"
echo "PSEUDO_LEN_FACTOR=${PSEUDO_LEN_FACTOR:-1 (code default; additive +~90 pad)}"
echo "Suffix: {split}_pseudo_strat_sticky.npy (pre-08-24 files cleared)"

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
