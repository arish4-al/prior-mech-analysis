#!/bin/bash
# Submit sharded Goal 2 choice L–R jobs for one structured null scheme.
#
#   NULL_SCHEME=pseudo_strat|pseudo_fixed|harris \
#     bash scripts/submit_goal2_choice_null_sharded.sh
#
# Schemes (journal 2026-07-23b / 2026-07-24):
#   pseudo_strat  — opt 1: AK + stim×block–stratified pseudo
#                   default PSEUDO_LEN_FACTOR=3; adaptive bump to 16 if needed
#                   always writes {split}_pseudo_strat*.npy (overwrites prior strat)
#   pseudo_fixed  — opt 2: AK on exact real stim×block sequence
#   harris        — opt 3: session-transplant; donor re-filtered to same
#                   stim×prior stratum (overwrites _harris; CLEAR_STREAM=1)
#
# Strat rerun (recommended):
#   bash scripts/submit_goal2_choice_strat_x3_sharded.sh
#   # or:
#   NULL_SCHEME=pseudo_strat bash scripts/submit_goal2_choice_null_sharded.sh
#
# Harris rerun (donor re-strat; clears prior _harris):
#   NULL_SCHEME=harris bash scripts/submit_goal2_choice_null_sharded.sh
#
#   CLEAR_STREAM=0  — keep existing stream_acc / res (default clears for strat+harris)
#   SMOKE_FIRST=1 NULL_SCHEME=pseudo_strat \
#     bash scripts/submit_goal2_choice_null_sharded.sh
#
# Submit all three: bash scripts/submit_goal2_choice_null_all_schemes_sharded.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

NULL_SCHEME="${NULL_SCHEME:-pseudo_strat}"
PRESET="${PRESET:-choice_lr_session_null_all}"
N_SHARDS="${N_SHARDS:-4}"
NRAND="${NRAND:-2000}"
RESTART="${RESTART:-1}"
SMOKE_FIRST="${SMOKE_FIRST:-0}"
PSEUDO_LEN_FACTOR="${PSEUDO_LEN_FACTOR:-}"
CLEAR_STREAM="${CLEAR_STREAM:-}"
MEM_SHARD="${MEM_SHARD:-12G}"
MEM_FIN="${MEM_FIN:-10G}"
MEM_SMOKE="${MEM_SMOKE:-16G}"
MEM_DONORS="${MEM_DONORS:-8G}"
CPUS_SHARD="${CPUS_SHARD:-2}"
CPUS_FIN="${CPUS_FIN:-2}"
CPUS_SMOKE="${CPUS_SMOKE:-2}"
CPUS_DONORS="${CPUS_DONORS:-2}"
TIME_SHARD="${TIME_SHARD:-12:00:00}"
TIME_SMOKE="${TIME_SMOKE:-2:00:00}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"

SESSION_SHUFFLE_NULL=0
ACTKERNEL_CHOICE_NULL=0
ACTKERNEL_NULL_MODE=""
CASE_TAG=""
SUFFIX=""
JOB_PREFIX=""

_ORIG_SCHEME="$NULL_SCHEME"
case "$_ORIG_SCHEME" in
  pseudo_strat_x3|strat_x3|pseudo_strat|strat)
    NULL_SCHEME=pseudo_strat
    ACTKERNEL_CHOICE_NULL=1
    ACTKERNEL_NULL_MODE=strat
    PSEUDO_LEN_FACTOR="${PSEUDO_LEN_FACTOR:-3}"
    CASE_TAG=strat
    SUFFIX=_pseudo_strat
    JOB_PREFIX=g2ps
    CLEAR_STREAM="${CLEAR_STREAM:-1}"
    ;;
  pseudo_fixed|fixedstim|fixed)
    NULL_SCHEME=pseudo_fixed
    ACTKERNEL_CHOICE_NULL=1
    ACTKERNEL_NULL_MODE=fixedstim
    CASE_TAG=fixed
    SUFFIX=_pseudo_fixed
    JOB_PREFIX=g2pf
    PSEUDO_LEN_FACTOR=""
    CLEAR_STREAM="${CLEAR_STREAM:-0}"
    ;;
  harris|session)
    NULL_SCHEME=harris
    SESSION_SHUFFLE_NULL=1
    CASE_TAG=harris
    SUFFIX=_harris
    JOB_PREFIX=g2h
    PSEUDO_LEN_FACTOR=""
    CLEAR_STREAM="${CLEAR_STREAM:-1}"
    if [[ "${MEM_SHARD}" == "12G" ]]; then
      MEM_SHARD=6G
    fi
    ;;
  *)
    echo "ERROR: NULL_SCHEME must be pseudo_strat|pseudo_fixed|harris (got $_ORIG_SCHEME)" >&2
    exit 1
    ;;
esac

export SESSION_SHUFFLE_NULL ACTKERNEL_CHOICE_NULL ACTKERNEL_NULL_MODE NULL_SCHEME
export ACTKERNEL_PSEUDO_LEN_FACTOR="${PSEUDO_LEN_FACTOR:-}"
export PSEUDO_LEN_FACTOR

module load miniforge 2>/dev/null || true
if [[ -f "$HOME/conda_envs/ibl/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/conda_envs/ibl/bin/activate"
elif command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate ibl 2>/dev/null || conda activate iblenv 2>/dev/null || true
fi

if [[ "$ACTKERNEL_CHOICE_NULL" == "1" && ! -d third_party/behavior_models/behavior_models ]]; then
  echo "ERROR: missing submodule third_party/behavior_models" >&2
  echo "  git submodule update --init --recursive" >&2
  exit 1
fi

SPLITS=()
while IFS= read -r line; do
  [[ -n "$line" ]] && SPLITS+=("$line")
done < <(python3 -u scripts/run_goal2_splits.py --preset "$PRESET" --list-splits)

if [[ ${#SPLITS[@]} -eq 0 ]]; then
  echo "ERROR: no splits for PRESET=$PRESET" >&2
  exit 1
fi

# Overwrite prior run for this SUFFIX (CLEAR_STREAM=1): drop stream_acc + pooled res.
# Exact basename match so e.g. _pseudo_strat_x3 leftovers are left alone.
if [[ "$CLEAR_STREAM" == "1" && -n "$SUFFIX" ]]; then
  RES_ROOT="$ONE_CACHE_DIR/manifold/res"
  ACC="$RES_ROOT/_stream_acc"
  echo "CLEAR_STREAM=1: removing prior ${SUFFIX} stream_acc + res under $RES_ROOT"
  for sp in "${SPLITS[@]}"; do
    base="${sp}${SUFFIX}"
    rm -f "$ACC/${base}.npy" "$ACC/${base}.shard"*.npy 2>/dev/null || true
    rm -f "$RES_ROOT/${base}.npy" "$RES_ROOT/${base}_regde.npy" "$RES_ROOT/${base}_all.npy" \
      2>/dev/null || true
  done
fi

n_shard_jobs=$(( ${#SPLITS[@]} * N_SHARDS ))
echo "NULL_SCHEME=$NULL_SCHEME  mode=$ACTKERNEL_NULL_MODE  harris=$SESSION_SHUFFLE_NULL"
echo "PSEUDO_LEN_FACTOR=${PSEUDO_LEN_FACTOR:-1}  ACTKERNEL_PSEUDO_LEN_FACTOR=${ACTKERNEL_PSEUDO_LEN_FACTOR:-}"
echo "CLEAR_STREAM=$CLEAR_STREAM  RESTART=$RESTART"
echo "PRESET=$PRESET  N_SHARDS=$N_SHARDS  nrand=$NRAND  splits=${#SPLITS[@]}"
echo "MEM_SHARD=$MEM_SHARD  TIME_SHARD=$TIME_SHARD  shard_jobs=$n_shard_jobs"
echo "Suffix: {split}${SUFFIX}.npy"
printf '  %s\n' "${SPLITS[@]}"

job_tag() {
  local s="$1"
  s="${s//./p}"
  echo "${s:0:40}"
}

DEP_AFTER=""
if [[ "$SESSION_SHUFFLE_NULL" == "1" ]]; then
  DONOR_JID=$(sbatch --parsable \
    --mem="$MEM_DONORS" --cpus-per-task="$CPUS_DONORS" \
    --job-name="g2_choice_donors" \
    --export=ALL,SMOKE_FIRST="$SMOKE_FIRST" \
    scripts/run_goal2_choice_donors_slurm.sh)
  echo "choice donors job -> $DONOR_JID"
  DEP_AFTER="--dependency=afterok:${DONOR_JID}"
elif [[ "$SMOKE_FIRST" == "1" && "$ACTKERNEL_CHOICE_NULL" == "1" ]]; then
  SMOKE_JID=$(sbatch --parsable \
    --mem="$MEM_SMOKE" --cpus-per-task="$CPUS_SMOKE" --time="$TIME_SMOKE" \
    --job-name="g2_ak_smoke_${CASE_TAG}" \
    --export=ALL,ACTKERNEL_NULL_MODE="$ACTKERNEL_NULL_MODE",ACTKERNEL_PSEUDO_LEN_FACTOR="${ACTKERNEL_PSEUDO_LEN_FACTOR:-}" \
    scripts/run_goal2_actkernel_smoke_slurm.sh)
  echo "actkernel smoke ($ACTKERNEL_NULL_MODE factor=${ACTKERNEL_PSEUDO_LEN_FACTOR:-1}) -> $SMOKE_JID"
  DEP_AFTER="--dependency=afterok:${SMOKE_JID}"
fi

for sp in "${SPLITS[@]}"; do
  TAG=$(job_tag "$sp")
  SHARD_JOBS=()
  for ((k=0; k<N_SHARDS; k++)); do
    # shellcheck disable=SC2086
    JID=$(sbatch --parsable \
      --mem="$MEM_SHARD" --cpus-per-task="$CPUS_SHARD" --time="$TIME_SHARD" \
      --job-name="${JOB_PREFIX}_${TAG}_s${k}" \
      $DEP_AFTER \
      --export=ALL,SPLIT="$sp",SHARD_IDX="$k",N_SHARDS="$N_SHARDS",NRAND="$NRAND",RESTART="$RESTART",ACTKERNEL_CHOICE_NULL="$ACTKERNEL_CHOICE_NULL",ACTKERNEL_NULL_MODE="$ACTKERNEL_NULL_MODE",ACTKERNEL_PSEUDO_LEN_FACTOR="${ACTKERNEL_PSEUDO_LEN_FACTOR:-}",SESSION_SHUFFLE_NULL="$SESSION_SHUFFLE_NULL" \
      scripts/run_goal2_shard_slurm.sh)
    SHARD_JOBS+=("$JID")
    echo "  $sp shard $k/$N_SHARDS -> $JID"
  done
  DEP=$(IFS=:; echo "${SHARD_JOBS[*]}")
  FID=$(sbatch --parsable \
    --mem="$MEM_FIN" --cpus-per-task="$CPUS_FIN" \
    --dependency=afterok:"$DEP" \
    --job-name="${JOB_PREFIX}_fin_${TAG}" \
    --export=ALL,SPLIT="$sp",ACTKERNEL_CHOICE_NULL="$ACTKERNEL_CHOICE_NULL",ACTKERNEL_NULL_MODE="$ACTKERNEL_NULL_MODE",ACTKERNEL_PSEUDO_LEN_FACTOR="${ACTKERNEL_PSEUDO_LEN_FACTOR:-}",SESSION_SHUFFLE_NULL="$SESSION_SHUFFLE_NULL" \
    scripts/run_goal2_finalize_slurm.sh)
  echo "  $sp finalize -> $FID"
done

echo "Done ($NULL_SCHEME suffix=$SUFFIX). Monitor: squeue -u \$USER"
echo "Outputs: \$ONE_CACHE_DIR/manifold/res/{split}${SUFFIX}.npy"
