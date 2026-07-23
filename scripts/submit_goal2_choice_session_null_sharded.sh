#!/bin/bash
# Submit sharded Goal 2 jobs for choice L–R splits with Harris session-permutation nulls.
# Default preset: choice_lr_session_null_all (8 act splits: duringchoice + duringstim).
#
#   bash scripts/submit_goal2_choice_session_null_sharded.sh
#   PRESET=choice_lr_session_null_true N_SHARDS=4 \   # true-block (8)
#     bash scripts/submit_goal2_choice_session_null_sharded.sh
#   PRESET=choice_lr_session_null_bayes N_SHARDS=4 \
#     bash scripts/submit_goal2_choice_session_null_sharded.sh
#
# Assumes insertion cache already exists (run_goal2_cache_slurm.sh done).
#
# Donor bank: submitted as a Slurm job (NOT on the login node). Shard jobs
# depend on it (afterok). Bank stores full-session choice sequences; Harris
# nulls index donor choices at the recipient's stim×prior–eligible trial
# numbers. SESSION_SHUFFLE_NULL=1 by default.
#
# Optional smoke inside the donor job (compute node):
#   SMOKE_FIRST=1 bash scripts/submit_goal2_choice_session_null_sharded.sh
#
# Memory (override with MEM_SHARD / MEM_FIN / MEM_DONORS):
#   Peak RSS stream_pool nrand=2000 ≈ 1.5–2.5 GB (journal 07-10b).
#   Defaults: MEM_SHARD=6G, MEM_FIN=10G, MEM_DONORS=8G.
# Override: N_SHARDS=3 MEM_SHARD=8G bash scripts/submit_goal2_choice_session_null_sharded.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

PRESET="${PRESET:-choice_lr_session_null_all}"
N_SHARDS="${N_SHARDS:-4}"
NRAND="${NRAND:-2000}"
RESTART="${RESTART:-1}"
SESSION_SHUFFLE_NULL="${SESSION_SHUFFLE_NULL:-1}"
SMOKE_FIRST="${SMOKE_FIRST:-0}"
MEM_SHARD="${MEM_SHARD:-6G}"
MEM_FIN="${MEM_FIN:-10G}"
MEM_DONORS="${MEM_DONORS:-8G}"
CPUS_SHARD="${CPUS_SHARD:-2}"
CPUS_FIN="${CPUS_FIN:-2}"
CPUS_DONORS="${CPUS_DONORS:-2}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"

module load miniforge 2>/dev/null || true
# Prefer cluster env if present; fall back to whatever python3 is on PATH.
if [[ -f "$HOME/conda_envs/ibl/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/conda_envs/ibl/bin/activate"
elif command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate ibl 2>/dev/null || conda activate iblenv 2>/dev/null || true
fi

# Resolve split list only (lightweight; no cache I/O).
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
echo "SESSION_SHUFFLE_NULL=$SESSION_SHUFFLE_NULL  SMOKE_FIRST=$SMOKE_FIRST"
echo "MEM_DONORS=$MEM_DONORS  MEM_SHARD=$MEM_SHARD  MEM_FIN=$MEM_FIN"
echo "shard_jobs=$n_shard_jobs  Concurrent mem if all shards run: ${n_shard_jobs} × $MEM_SHARD"
echo "Splits:"
printf '  %s\n' "${SPLITS[@]}"

job_tag() {
  local s="$1"
  s="${s//./p}"
  echo "${s:0:48}"
}

DEP_AFTER=""
if [[ "$SESSION_SHUFFLE_NULL" == "1" ]]; then
  DONOR_JID=$(sbatch --parsable \
    --mem="$MEM_DONORS" --cpus-per-task="$CPUS_DONORS" \
    --job-name="g2_choice_donors" \
    --export=ALL,SMOKE_FIRST="$SMOKE_FIRST" \
    scripts/run_goal2_choice_donors_slurm.sh)
  echo "choice donors job -> $DONOR_JID (shards wait afterok)"
  DEP_AFTER="--dependency=afterok:${DONOR_JID}"
fi

for sp in "${SPLITS[@]}"; do
  TAG=$(job_tag "$sp")
  SHARD_JOBS=()
  for ((k=0; k<N_SHARDS; k++)); do
    # shellcheck disable=SC2086
    JID=$(sbatch --parsable \
      --mem="$MEM_SHARD" --cpus-per-task="$CPUS_SHARD" \
      --job-name="g2_${TAG}_s${k}" \
      $DEP_AFTER \
      --export=ALL,SPLIT="$sp",SHARD_IDX="$k",N_SHARDS="$N_SHARDS",NRAND="$NRAND",RESTART="$RESTART",SESSION_SHUFFLE_NULL="$SESSION_SHUFFLE_NULL" \
      scripts/run_goal2_shard_slurm.sh)
    SHARD_JOBS+=("$JID")
    echo "  $sp shard $k/$N_SHARDS -> $JID"
  done
  DEP=$(IFS=:; echo "${SHARD_JOBS[*]}")
  FID=$(sbatch --parsable \
    --mem="$MEM_FIN" --cpus-per-task="$CPUS_FIN" \
    --dependency=afterok:"$DEP" \
    --job-name="g2_fin_${TAG}" \
    --export=ALL,SPLIT="$sp",SESSION_SHUFFLE_NULL="$SESSION_SHUFFLE_NULL" \
    scripts/run_goal2_finalize_slurm.sh)
  echo "  $sp finalize -> $FID (after $DEP)"
done

echo "Done. Monitor: squeue -u \$USER"
echo "Null scheme: Harris session-permutation when SESSION_SHUFFLE_NULL=1"
echo "Pooled filenames: {split}_harris*.npy (label shuffle stays {split}*.npy)"
echo "Donor bank job (compute node): scripts/run_goal2_choice_donors_slurm.sh"
echo "Shard outputs: \$ONE_CACHE_DIR/manifold/res/_stream_acc/{split}_harris.shard{k}.npy"
echo "Final outputs: \$ONE_CACHE_DIR/manifold/res/{split}_harris.npy"
