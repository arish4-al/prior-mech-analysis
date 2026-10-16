#!/bin/bash
# Standard-split alternative nulls. Same labels as the shuffle catalog:
# action kernel (α=0.2) on the full session, then the bad-trial / saturation
# mask, then drop true probabilityLeft==0.5, then min_trials_per_side=5.
#
# Aim: those labels, scored under the same splits for both nulls:
#   sticky — option-1 action-kernel pseudo-session plus copy-last.
#            Writes res_standard/{split}_pseudo_strat_sticky.npy
#   harris — unique session transplant.
#            Writes res_standard/{split}_harris_unique.npy
#   choice — choice L vs R (8), stratum = stim × act prior.
#   prior_stim — prior L vs R inside stim side (2).
#   prior_none — prior L vs R with no stratum (2 fully unsplit).
#
# Harris reads manifold/choice_donors_standard.npy (full-session kernel on
# the donor, then the same mask and true-0.5 drop). The shuffle catalog in
# res_standard/{split}.npy is left in place.
#
#   bash scripts/submit_standard_alt_nulls_orcd.sh
#   ARM=sticky|harris_choice|harris_prior_stim|harris_prior_none|all
#   REBUILD_DONORS=1   rebuild the standard donor bank (default 0)
#   N_SHARDS=  MEM_SHARD=  TIME_SHARD=  override that arm's defaults
#     (ARM=all applies one override to every arm)
#
# Defaults: choice 4 shards (sticky 12G, Harris 6G), stim-stratum prior
# 4 shards (sticky 12G, Harris 8G), no-stratum prior 12×16G. nrand=2000.
# CLEAR_STREAM=1 clears only that arm's suffix under res_standard.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

ARM="${ARM:-all}"
NRAND="${NRAND:-2000}"
RESTART="${RESTART:-1}"
CLEAR_STREAM="${CLEAR_STREAM:-1}"
REBUILD_DONORS="${REBUILD_DONORS:-0}"
N_SHARDS_USER="${N_SHARDS:-}"
MEM_SHARD_USER="${MEM_SHARD:-}"
TIME_SHARD_USER="${TIME_SHARD:-}"
MEM_FIN="${MEM_FIN:-16G}"
CPUS_SHARD="${CPUS_SHARD:-2}"
CPUS_FIN="${CPUS_FIN:-2}"
MEM_DONORS="${MEM_DONORS:-16G}"
CPUS_DONORS="${CPUS_DONORS:-2}"
TIME_DONORS="${TIME_DONORS:-8:00:00}"
PARTITION="${PARTITION:-pi_fiete}"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/sbatch_defaults.sh"

ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"

PY="${PY:-$HOME/conda_envs/ibl/bin/python}"
if [[ ! -x "$PY" ]]; then
  echo "ERROR: ibl python not found at $PY" >&2
  exit 1
fi
if [[ ! -d third_party/behavior_models/behavior_models ]]; then
  echo "ERROR: missing submodule third_party/behavior_models" >&2
  echo "  git submodule update --init --recursive" >&2
  exit 1
fi

case "$ARM" in
  sticky|harris_choice|harris_prior_stim|harris_prior_none|all) ;;
  *)
    echo "ERROR: ARM must be sticky|harris_choice|harris_prior_stim|harris_prior_none|all (got $ARM)" >&2
    exit 1
    ;;
esac

echo "Aim: standard act labels (full-session kernel, mask, drop true 0.5, min5)."
echo "ARM=$ARM  nrand=$NRAND  CLEAR_STREAM=$CLEAR_STREAM  REBUILD_DONORS=$REBUILD_DONORS"
echo "PARTITION=$PARTITION  SBATCH_EXTRA=${SBATCH_EXTRA:-}"
echo "Outputs: \$ONE_CACHE_DIR/manifold/res_standard/{split}{suffix}.npy"

job_tag() {
  local s="$1"
  s="${s//./p}"
  echo "${s:0:40}"
}

list_splits() {
  local preset="$1"
  SPLITS=()
  while IFS= read -r line; do
    [[ -n "$line" ]] && SPLITS+=("$line")
  done < <("$PY" -u scripts/run_goal2_splits.py --preset "$preset" --list-splits)
  if [[ ${#SPLITS[@]} -eq 0 ]]; then
    echo "ERROR: no splits for PRESET=$preset" >&2
    exit 1
  fi
}

submit_arm() {
  local preset="$1" scheme="$2" shards="$3" mem="$4" tshard="$5" prefix="$6"
  if [[ -n "$N_SHARDS_USER" ]]; then shards="$N_SHARDS_USER"; fi
  if [[ -n "$MEM_SHARD_USER" ]]; then mem="$MEM_SHARD_USER"; fi
  if [[ -n "$TIME_SHARD_USER" ]]; then tshard="$TIME_SHARD_USER"; fi

  local mode="" sticky=0 shuffle=0 ak=0 suffix=""
  if [[ "$scheme" == "sticky" ]]; then
    ak=1
    mode=strat
    sticky=1
    suffix=_pseudo_strat_sticky
  else
    shuffle=1
    suffix=_harris_unique
  fi

  list_splits "$preset"
  local RES_ROOT="$ONE_CACHE_DIR/manifold/res_standard"
  if [[ "$CLEAR_STREAM" == "1" ]]; then
    local ACC="$RES_ROOT/_stream_acc"
    echo "CLEAR_STREAM=1: removing prior ${suffix} under $RES_ROOT"
    local sp
    for sp in "${SPLITS[@]}"; do
      local base="${sp}${suffix}"
      rm -f "$ACC/${base}.npy" "$ACC/${base}.shard"*.npy 2>/dev/null || true
      rm -f "$RES_ROOT/${base}.npy" "$RES_ROOT/${base}_regde.npy" \
        "$RES_ROOT/${base}_all.npy" "$RES_ROOT/${base}_all_regde.npy" \
        2>/dev/null || true
    done
  fi

  echo "PRESET=$preset  scheme=$scheme  suffix=$suffix  shards=$shards  mem=$mem  time=$tshard"
  printf '  %s\n' "${SPLITS[@]}"

  local EXPORT="STANDARD_ACT_PRIOR=1,EXCLUDE_STICKY_TRIALS=0,SESSION_SHUFFLE_NULL=${shuffle},ACTKERNEL_CHOICE_NULL=${ak},ACTKERNEL_NULL_MODE=${mode},ACTKERNEL_LATE_STICKY=${sticky},ACTKERNEL_PSEUDO_LEN_FACTOR="
  local dep_flag=()
  if [[ -n "${DONOR_JID:-}" && "$scheme" == "harris" ]]; then
    dep_flag=(--dependency="afterok:${DONOR_JID}")
  fi

  local sp k TAG JID FID DEP
  for sp in "${SPLITS[@]}"; do
    TAG=$(job_tag "$sp")
    local SHARD_JOBS=()
    for ((k=0; k<shards; k++)); do
      # shellcheck disable=SC2086
      JID=$(sbatch --parsable $SBATCH_EXTRA \
        --partition="$PARTITION" \
        --mem="$mem" --cpus-per-task="$CPUS_SHARD" --time="$tshard" \
        --job-name="${prefix}_${TAG}_s${k}" \
        "${dep_flag[@]}" \
        --export=ALL,SPLIT="$sp",SHARD_IDX="$k",N_SHARDS="$shards",NRAND="$NRAND",RESTART="$RESTART",$EXPORT \
        scripts/run_goal2_shard_slurm.sh)
      SHARD_JOBS+=("$JID")
      echo "  $sp shard $k/$shards -> $JID"
    done
    DEP=$(IFS=:; echo "${SHARD_JOBS[*]}")
    # shellcheck disable=SC2086
    FID=$(sbatch --parsable $SBATCH_EXTRA \
      --partition="$PARTITION" \
      --mem="$MEM_FIN" --cpus-per-task="$CPUS_FIN" --time=2:00:00 \
      --dependency=afterok:"$DEP" \
      --job-name="${prefix}_fin_${TAG}" \
      --export=ALL,SPLIT="$sp",$EXPORT \
      scripts/run_goal2_finalize_slurm.sh)
    echo "  $sp finalize -> $FID"
  done
}

need_donors=0
case "$ARM" in
  harris_choice|harris_prior_stim|harris_prior_none|all) need_donors=1 ;;
esac

DONOR_JID=""
if [[ "$need_donors" == "1" ]]; then
  # shellcheck disable=SC2086
  DONOR_JID=$(sbatch --parsable $SBATCH_EXTRA \
    --partition="$PARTITION" \
    --mem="$MEM_DONORS" --cpus-per-task="$CPUS_DONORS" --time="$TIME_DONORS" \
    --job-name="std_donors" \
    --export=ALL,REBUILD_DONORS="$REBUILD_DONORS" \
    scripts/run_goal2_standard_donors_slurm.sh)
  echo "standard donor bank -> $DONOR_JID"
fi

if [[ "$ARM" == "sticky" || "$ARM" == "all" ]]; then
  submit_arm standard_choice_harris sticky 4 12G 12:00:00 stdssc
  submit_arm standard_prior_harris_stim sticky 4 12G 12:00:00 stdsss
  submit_arm standard_prior_harris_none sticky 12 16G 12:00:00 stdssn
fi
if [[ "$ARM" == "harris_choice" || "$ARM" == "all" ]]; then
  submit_arm standard_choice_harris harris 4 6G 12:00:00 stdhc
fi
if [[ "$ARM" == "harris_prior_stim" || "$ARM" == "all" ]]; then
  submit_arm standard_prior_harris_stim harris 4 8G 12:00:00 stdhs
fi
if [[ "$ARM" == "harris_prior_none" || "$ARM" == "all" ]]; then
  submit_arm standard_prior_harris_none harris 12 16G 12:00:00 stdhn
fi

echo "Done (standard alternative nulls). Monitor: squeue -u \$USER"
echo "Final outputs: \$ONE_CACHE_DIR/manifold/res_standard/{split}_pseudo_strat_sticky.npy and {split}_harris_unique.npy"
