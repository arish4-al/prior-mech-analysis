#!/bin/bash
#SBATCH --job-name=goal2_shard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH -p mit_normal
#SBATCH --time=6:00:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o goal2_shard_%x_%j.out

# One insertion shard for one split (no finalize).
# Env: SPLIT, SHARD_IDX, N_SHARDS, NRAND, RESTART,
#      SESSION_SHUFFLE_NULL, ACTKERNEL_CHOICE_NULL, ACTKERNEL_NULL_MODE,
#      EXCLUDE_STICKY_TRIALS, ...

set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"

SPLIT="${SPLIT:?Set SPLIT=}"
SHARD_IDX="${SHARD_IDX:?Set SHARD_IDX=0..N-1}"
N_SHARDS="${N_SHARDS:-4}"
NRAND="${NRAND:-2000}"
RESTART="${RESTART:-1}"
SESSION_SHUFFLE_NULL="${SESSION_SHUFFLE_NULL:-0}"
EXCLUDE_STICKY_TRIALS="${EXCLUDE_STICKY_TRIALS:-0}"
STICKY_LATE_FRAC="${STICKY_LATE_FRAC:-0.2}"
STICKY_MIN_RUN="${STICKY_MIN_RUN:-10}"
ACTKERNEL_CHOICE_NULL="${ACTKERNEL_CHOICE_NULL:-0}"
ACTKERNEL_NULL_MODE="${ACTKERNEL_NULL_MODE:-}"

module load miniforge
conda activate ~/conda_envs/ibl
cd "$REPO_DIR"

if [[ "$ACTKERNEL_CHOICE_NULL" == "1" || -n "$ACTKERNEL_NULL_MODE" ]]; then
  if [[ ! -d third_party/behavior_models/behavior_models ]]; then
    echo "ERROR: missing submodule third_party/behavior_models" >&2
    echo "  git submodule update --init --recursive" >&2
    exit 1
  fi
fi

echo "Host: $(hostname) Date: $(date)"
git log -1 --oneline
echo "SPLIT=$SPLIT shard=$SHARD_IDX/$N_SHARDS nrand=$NRAND"
echo "session_shuffle=$SESSION_SHUFFLE_NULL actkernel=$ACTKERNEL_CHOICE_NULL mode=${ACTKERNEL_NULL_MODE:-default}"
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-?} SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}"

ARGS=(--splits "$SPLIT" --nrand "$NRAND" --no-save-cache
      --shard-idx "$SHARD_IDX" --n-shards "$N_SHARDS" --no-finalize)
[[ "$RESTART" == "1" ]] && ARGS+=(--restart) || ARGS+=(--no-restart)
ARGS+=(--stream-pool)
[[ "$SESSION_SHUFFLE_NULL" == "1" ]] && ARGS+=(--session-shuffle-null)
[[ "$ACTKERNEL_CHOICE_NULL" == "1" ]] && ARGS+=(--actkernel-choice-null)
[[ -n "$ACTKERNEL_NULL_MODE" ]] && ARGS+=(--actkernel-null-mode "$ACTKERNEL_NULL_MODE")
[[ "$EXCLUDE_STICKY_TRIALS" == "1" ]] && ARGS+=(
  --exclude-sticky-trials
  --sticky-late-frac "$STICKY_LATE_FRAC"
  --sticky-min-run "$STICKY_MIN_RUN"
)

python3 -u scripts/run_goal2_splits.py "${ARGS[@]}"
echo "Shard done: $(date)"
RES_ROOT="$ONE_CACHE_DIR/manifold/res"
[[ "$EXCLUDE_STICKY_TRIALS" == "1" ]] && RES_ROOT="$ONE_CACHE_DIR/manifold/res_excl_sticky"
SUFFIX=""
if [[ -n "$ACTKERNEL_NULL_MODE" ]]; then
  case "$ACTKERNEL_NULL_MODE" in
    strat) SUFFIX=_pseudo_strat ;;
    fixedstim) SUFFIX=_pseudo_fixed ;;
    unconstrained) SUFFIX=_pseudosession ;;
  esac
elif [[ "$ACTKERNEL_CHOICE_NULL" == "1" ]]; then
  SUFFIX=_pseudo_strat
elif [[ "$SESSION_SHUFFLE_NULL" == "1" ]]; then
  SUFFIX=_harris
fi
ls -lh "$RES_ROOT/_stream_acc/${SPLIT}${SUFFIX}.shard${SHARD_IDX}.npy" 2>/dev/null || true
