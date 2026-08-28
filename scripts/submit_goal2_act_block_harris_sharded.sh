#!/bin/bash
# Submit sharded Goal 2 jobs: Harris unique-null for all act_block prior L–R splits.
#
#   bash scripts/submit_goal2_act_block_harris_sharded.sh
#
# Default PRESET=act_block_harris_all (9 splits):
#   duringstim (4): act_block_duringstim_{l,r}_choice_{l_f1,r_f2}
#   duringchoice (4): act_block_stim_{l,r}_duringchoice_{l_f1,r_f2}
#   act_block_only (1): prior L vs R, no stim/choice stratum
#
# Other presets:
#   PRESET=act_block_only bash scripts/submit_goal2_act_block_harris_sharded.sh
#   PRESET=act_block_duringstim …
#   PRESET=act_block_duringchoice …
#   PRESET=act_block_harris_unsplit …   # stim-side + choice-side, no f1/f2
#   PRESET=act_block_harris_fully_unsplit …  # one pool per window, no stratum
#   PRESET=goal3_duringstim_act …   # contrast-expanded (needs donor contrasts)
# Bayes (Harris unique; donor labels = Bayes-binary from stim):
#   PRESET=bayes_block_duringstim …          # 4 duringstim (local maps)
#   PRESET=bayes_block_unsplit_duringstim …  # stim-side only
#   PRESET=bayes_block_harris …              # 8 stim×choice (incl. duringchoice)
#   bash scripts/submit_goal2_bayes_harris_orcd.sh
#
# Fully-unsplit wrapper (Harris unique + AK copy-last, more shards):
#   bash scripts/submit_goal2_act_block_fully_unsplit_orcd.sh
#
# Unsplit-only wrapper:
#   bash scripts/submit_goal2_act_block_harris_unsplit_sharded.sh
#
# Restart OOM'd act_block_only shards (does not CLEAR finished 0/3):
#   bash scripts/submit_goal2_act_block_only_harris_restart.sh
#
# Outputs: $ONE_CACHE_DIR/manifold/res/{split}_harris_unique.npy
#   (plain shuffle stays at {split}.npy — never overwritten)
# Rebuilds donor bank first (includes contrast_left/right for Goal-3).
#
# CLEAR_STREAM=1 (default) clears only *_harris_unique for these splits.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

PRESET="${PRESET:-act_block_harris_all}"
N_SHARDS="${N_SHARDS:-4}"
NRAND="${NRAND:-2000}"
RESTART="${RESTART:-1}"
CLEAR_STREAM="${CLEAR_STREAM:-1}"
REBUILD_DONORS="${REBUILD_DONORS:-1}"
MEM_SHARD="${MEM_SHARD:-6G}"
MEM_FIN="${MEM_FIN:-10G}"
MEM_DONORS="${MEM_DONORS:-8G}"
CPUS_SHARD="${CPUS_SHARD:-2}"
CPUS_FIN="${CPUS_FIN:-2}"
CPUS_DONORS="${CPUS_DONORS:-2}"
TIME_SHARD="${TIME_SHARD:-12:00:00}"
PARTITION="${PARTITION:-pi_fiete}"
# --requeue is the default on preemptable partitions. Explicit SBATCH_EXTRA wins
# (including SBATCH_EXTRA="" to disable). Requeued shards keep stream_acc;
# RESTART=1 skips insertions already in the shard file. CLEAR_STREAM only
# runs in this wrapper, not on Slurm requeue.
if [[ -z "${SBATCH_EXTRA+x}" ]]; then
  case "$PARTITION" in
    mit_preemptable|mit_preem) SBATCH_EXTRA="--requeue" ;;
    *) SBATCH_EXTRA="" ;;
  esac
fi
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"
export SBATCH_EXTRA

export SESSION_SHUFFLE_NULL=1
export ACTKERNEL_CHOICE_NULL=0
export ACTKERNEL_NULL_MODE=""
export ACTKERNEL_PSEUDO_LEN_FACTOR=""

SUFFIX=_harris_unique
JOB_PREFIX="${JOB_PREFIX:-g2abh}"

module load miniforge 2>/dev/null || true
if [[ -f "$HOME/conda_envs/ibl/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/conda_envs/ibl/bin/activate"
elif command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate ibl 2>/dev/null || conda activate iblenv 2>/dev/null || true
fi

SPLITS=()
while IFS= read -r line; do
  [[ -n "$line" ]] && SPLITS+=("$line")
done < <(python3 -u scripts/run_goal2_splits.py --preset "$PRESET" --list-splits)

if [[ ${#SPLITS[@]} -eq 0 ]]; then
  echo "ERROR: no splits for PRESET=$PRESET" >&2
  exit 1
fi

if [[ "$CLEAR_STREAM" == "1" ]]; then
  RES_ROOT="$ONE_CACHE_DIR/manifold/res"
  ACC="$RES_ROOT/_stream_acc"
  echo "CLEAR_STREAM=1: removing prior ${SUFFIX} stream_acc + res under $RES_ROOT"
  for sp in "${SPLITS[@]}"; do
    base="${sp}${SUFFIX}"
    rm -f "$ACC/${base}.npy" "$ACC/${base}.shard"*.npy 2>/dev/null || true
    rm -f "$RES_ROOT/${base}.npy" "$RES_ROOT/${base}_regde.npy" \
      "$RES_ROOT/${base}_all.npy" "$RES_ROOT/${base}_all_regde.npy" \
      2>/dev/null || true
  done
fi

n_shard_jobs=$(( ${#SPLITS[@]} * N_SHARDS ))
echo "NULL_SCHEME=harris_unique (act_block prior L–R)"
echo "PRESET=$PRESET  N_SHARDS=$N_SHARDS  nrand=$NRAND  splits=${#SPLITS[@]}"
echo "CLEAR_STREAM=$CLEAR_STREAM  REBUILD_DONORS=$REBUILD_DONORS  RESTART=$RESTART"
echo "PARTITION=$PARTITION  SBATCH_EXTRA=${SBATCH_EXTRA:-}  MEM_SHARD=$MEM_SHARD  TIME_SHARD=$TIME_SHARD  shard_jobs=$n_shard_jobs"
echo "Outputs: \$ONE_CACHE_DIR/manifold/res/{split}${SUFFIX}.npy"
printf '  %s\n' "${SPLITS[@]}"

job_tag() {
  local s="$1"
  s="${s//./p}"
  echo "${s:0:40}"
}

DEP_AFTER=""
if [[ -n "${DONOR_JID:-}" ]]; then
  echo "reusing donors job $DONOR_JID"
  DEP_AFTER="--dependency=afterok:${DONOR_JID}"
elif [[ "$REBUILD_DONORS" == "1" ]]; then
  # Force rebuild so contrast_left/right are present for Goal-3 / contrast splits.
  # shellcheck disable=SC2086
  DONOR_JID=$(sbatch --parsable $SBATCH_EXTRA \
    --partition="$PARTITION" \
    --mem="$MEM_DONORS" --cpus-per-task="$CPUS_DONORS" \
    --job-name="g2_choice_donors" \
    --export=ALL,FORCE_REBUILD=1 \
    scripts/run_goal2_choice_donors_slurm.sh)
  echo "choice donors job -> $DONOR_JID"
  DEP_AFTER="--dependency=afterok:${DONOR_JID}"
fi

for sp in "${SPLITS[@]}"; do
  TAG=$(job_tag "$sp")
  SHARD_JOBS=()
  for ((k=0; k<N_SHARDS; k++)); do
    # shellcheck disable=SC2086
    JID=$(sbatch --parsable $SBATCH_EXTRA \
      --partition="$PARTITION" \
      --mem="$MEM_SHARD" --cpus-per-task="$CPUS_SHARD" --time="$TIME_SHARD" \
      --job-name="${JOB_PREFIX}_${TAG}_s${k}" \
      $DEP_AFTER \
      --export=ALL,SPLIT="$sp",SHARD_IDX="$k",N_SHARDS="$N_SHARDS",NRAND="$NRAND",RESTART="$RESTART",SESSION_SHUFFLE_NULL=1,ACTKERNEL_CHOICE_NULL=0,ACTKERNEL_NULL_MODE=,ACTKERNEL_PSEUDO_LEN_FACTOR= \
      scripts/run_goal2_shard_slurm.sh)
    SHARD_JOBS+=("$JID")
    echo "  $sp shard $k/$N_SHARDS -> $JID"
  done
  DEP=$(IFS=:; echo "${SHARD_JOBS[*]}")
  # shellcheck disable=SC2086
  FID=$(sbatch --parsable $SBATCH_EXTRA \
    --partition="$PARTITION" \
    --mem="$MEM_FIN" --cpus-per-task="$CPUS_FIN" \
    --dependency=afterok:"$DEP" \
    --job-name="${JOB_PREFIX}_fin_${TAG}" \
    --export=ALL,SPLIT="$sp",SESSION_SHUFFLE_NULL=1,ACTKERNEL_CHOICE_NULL=0,ACTKERNEL_NULL_MODE=,ACTKERNEL_PSEUDO_LEN_FACTOR= \
    scripts/run_goal2_finalize_slurm.sh)
  echo "  $sp finalize -> $FID"
done

echo "Done (act_block harris_unique). Monitor: squeue -u \$USER"
echo "Final outputs: \$ONE_CACHE_DIR/manifold/res/{split}${SUFFIX}.npy"
