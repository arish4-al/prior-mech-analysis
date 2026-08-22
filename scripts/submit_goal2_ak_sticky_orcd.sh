#!/bin/bash
# Fitted ActionKernel + copy-last, within each shuffle stratum (fixedstim).
#
# Choice L–R (stim-aligned duringstim + move-aligned duringchoice, 8 act splits)
# and act_block prior L–R (same two alignments, 8 splits). Prefit θ once per
# eid so shards reuse manifold/actkernel_fits/.
#
#   bash scripts/submit_goal2_ak_sticky_orcd.sh
#   PREFIT=0 bash scripts/submit_goal2_ak_sticky_orcd.sh   # skip prefit
#   FAMILY=choice|act_block|act_block_unsplit|both  (default both)
#   PREFIT=0 FAMILY=act_block_unsplit bash scripts/submit_goal2_ak_sticky_orcd.sh
#
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

PREFIT="${PREFIT:-1}"
FAMILY="${FAMILY:-both}"
NRAND="${NRAND:-2000}"
N_SHARDS="${N_SHARDS:-4}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"
export NRAND N_SHARDS

PREFIT_JID=""
if [[ "$PREFIT" == "1" ]]; then
  echo "=== Prefit ActionKernel (once per eid) ==="
  PREFIT_JID=$(sbatch --parsable \
    --job-name=g2_ak_prefit \
    --mem="${MEM_PREFIT:-8G}" --cpus-per-task=2 --time="${TIME_PREFIT:-8:00:00}" \
    --export=ALL \
    scripts/run_goal2_ak_prefit_slurm.sh)
  echo "Prefit -> $PREFIT_JID (shard jobs afterok this JID)"
fi
export PREFIT_JID

export NULL_SCHEME=pseudo_fixed_sticky

if [[ "$FAMILY" != "choice" && "$FAMILY" != "act_block" \
      && "$FAMILY" != "act_block_unsplit" && "$FAMILY" != "both" ]]; then
  echo "ERROR: FAMILY must be choice|act_block|act_block_unsplit|both (got $FAMILY)" >&2
  exit 1
fi

if [[ "$FAMILY" == "choice" || "$FAMILY" == "both" ]]; then
  echo "=== Choice L–R (stim + move) fitted+copy-last ==="
  PRESET=choice_lr_ak_sticky bash scripts/submit_goal2_choice_null_sharded.sh
fi
if [[ "$FAMILY" == "act_block" || "$FAMILY" == "both" ]]; then
  echo "=== act_block prior L–R f1/f2 (stim + move) fitted+copy-last ==="
  PRESET=act_block_ak_sticky bash scripts/submit_goal2_choice_null_sharded.sh
fi
if [[ "$FAMILY" == "act_block_unsplit" ]]; then
  echo "=== act_block unsplit prior L–R (stim-side + choice-side) fitted+copy-last ==="
  PRESET=act_block_ak_sticky_unsplit bash scripts/submit_goal2_choice_null_sharded.sh
fi

echo "Outputs: \$ONE_CACHE_DIR/manifold/res/{split}_pseudo_fixed_sticky.npy"
