#!/bin/bash
# Submit sharded Goal 2 jobs for choice L–R with BWM-style ActionKernel
# synthetic-session nulls (--actkernel-choice-null).
#
# Default preset: choice_lr_session_null_all (8 act splits: duringchoice + duringstim).
#
#   bash scripts/submit_goal2_choice_actkernel_null_sharded.sh
#   PRESET=choice_lr_session_null_true N_SHARDS=4 \
#     bash scripts/submit_goal2_choice_actkernel_null_sharded.sh
#   PRESET=choice_lr_session_null_bayes \
#     bash scripts/submit_goal2_choice_actkernel_null_sharded.sh
#
# Assumes insertion cache already exists. No choice-donor bank needed.
#
# Optional smoke on a compute node first:
#   SMOKE_FIRST=1 bash scripts/submit_goal2_choice_actkernel_null_sharded.sh
#
# Needs torch in the conda env (ibl / iblenv). behavior_models comes from the
# repo submodule third_party/behavior_models (no pip install on the cluster).
# On a fresh clone: git submodule update --init --recursive
# First eid does MCMC fit (cached under manifold/actkernel_fits/); later
# probes/shards reuse the pickle. Prefer more mem/time than Harris shards.
#
# Defaults: MEM_SHARD=12G, MEM_FIN=10G, TIME_SHARD=12:00:00
# Override: N_SHARDS=3 MEM_SHARD=16G bash scripts/submit_goal2_choice_actkernel_null_sharded.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

PRESET="${PRESET:-choice_lr_session_null_all}"
N_SHARDS="${N_SHARDS:-4}"
NRAND="${NRAND:-2000}"
RESTART="${RESTART:-1}"
ACTKERNEL_CHOICE_NULL="${ACTKERNEL_CHOICE_NULL:-1}"
SESSION_SHUFFLE_NULL="${SESSION_SHUFFLE_NULL:-0}"
SMOKE_FIRST="${SMOKE_FIRST:-0}"
MEM_SHARD="${MEM_SHARD:-12G}"
MEM_FIN="${MEM_FIN:-10G}"
MEM_SMOKE="${MEM_SMOKE:-16G}"
CPUS_SHARD="${CPUS_SHARD:-2}"
CPUS_FIN="${CPUS_FIN:-2}"
CPUS_SMOKE="${CPUS_SMOKE:-2}"
TIME_SHARD="${TIME_SHARD:-12:00:00}"
TIME_SMOKE="${TIME_SMOKE:-2:00:00}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"

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

n_shard_jobs=$(( ${#SPLITS[@]} * N_SHARDS ))
echo "PRESET=$PRESET  N_SHARDS=$N_SHARDS  nrand=$NRAND  splits=${#SPLITS[@]}"
echo "ACTKERNEL_CHOICE_NULL=$ACTKERNEL_CHOICE_NULL  SMOKE_FIRST=$SMOKE_FIRST"
echo "MEM_SHARD=$MEM_SHARD  MEM_FIN=$MEM_FIN  TIME_SHARD=$TIME_SHARD"
echo "shard_jobs=$n_shard_jobs"
echo "Splits:"
printf '  %s\n' "${SPLITS[@]}"

job_tag() {
  local s="$1"
  s="${s//./p}"
  echo "${s:0:48}"
}

DEP_AFTER=""
if [[ "$SMOKE_FIRST" == "1" ]]; then
  SMOKE_JID=$(sbatch --parsable \
    --mem="$MEM_SMOKE" --cpus-per-task="$CPUS_SMOKE" --time="$TIME_SMOKE" \
    --job-name="g2_actkernel_smoke" \
    --export=ALL \
    scripts/run_goal2_actkernel_smoke_slurm.sh)
  echo "actkernel smoke job -> $SMOKE_JID (shards wait afterok)"
  DEP_AFTER="--dependency=afterok:${SMOKE_JID}"
fi

for sp in "${SPLITS[@]}"; do
  TAG=$(job_tag "$sp")
  SHARD_JOBS=()
  for ((k=0; k<N_SHARDS; k++)); do
    # shellcheck disable=SC2086
    JID=$(sbatch --parsable \
      --mem="$MEM_SHARD" --cpus-per-task="$CPUS_SHARD" --time="$TIME_SHARD" \
      --job-name="g2ak_${TAG}_s${k}" \
      $DEP_AFTER \
      --export=ALL,SPLIT="$sp",SHARD_IDX="$k",N_SHARDS="$N_SHARDS",NRAND="$NRAND",RESTART="$RESTART",ACTKERNEL_CHOICE_NULL="$ACTKERNEL_CHOICE_NULL",SESSION_SHUFFLE_NULL="$SESSION_SHUFFLE_NULL" \
      scripts/run_goal2_shard_slurm.sh)
    SHARD_JOBS+=("$JID")
    echo "  $sp shard $k/$N_SHARDS -> $JID"
  done
  DEP=$(IFS=:; echo "${SHARD_JOBS[*]}")
  FID=$(sbatch --parsable \
    --mem="$MEM_FIN" --cpus-per-task="$CPUS_FIN" \
    --dependency=afterok:"$DEP" \
    --job-name="g2ak_fin_${TAG}" \
    --export=ALL,SPLIT="$sp",ACTKERNEL_CHOICE_NULL="$ACTKERNEL_CHOICE_NULL",SESSION_SHUFFLE_NULL="$SESSION_SHUFFLE_NULL" \
    scripts/run_goal2_finalize_slurm.sh)
  echo "  $sp finalize -> $FID (after $DEP)"
done

echo "Done. Monitor: squeue -u \$USER"
echo "Null scheme: BWM ActionKernel synthetic sessions when ACTKERNEL_CHOICE_NULL=1"
echo "Pooled filenames: {split}_pseudosession*.npy (label shuffle stays {split}*.npy)"
echo "Smoke (optional): scripts/smoke_choice_actkernel_null.py"
echo "Shard outputs: \$ONE_CACHE_DIR/manifold/res/_stream_acc/{split}_pseudosession.shard{k}.npy"
echo "Final outputs: \$ONE_CACHE_DIR/manifold/res/{split}_pseudosession.npy"
echo "Fits cache: \$ONE_CACHE_DIR/manifold/actkernel_fits/"
