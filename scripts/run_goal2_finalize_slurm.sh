#!/bin/bash
#SBATCH --job-name=goal2_finalize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH -p mit_normal
#SBATCH --time=2:00:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o goal2_finalize_%x_%j.out

# Merge stream_acc shards → manifold/res/{split}*.npy
# Env: SPLIT, SESSION_SHUFFLE_NULL, ACTKERNEL_CHOICE_NULL, ACTKERNEL_NULL_MODE

set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"

SPLIT="${SPLIT:?Set SPLIT=}"
EXCLUDE_STICKY_TRIALS="${EXCLUDE_STICKY_TRIALS:-0}"
SESSION_SHUFFLE_NULL="${SESSION_SHUFFLE_NULL:-0}"
ACTKERNEL_CHOICE_NULL="${ACTKERNEL_CHOICE_NULL:-0}"
ACTKERNEL_NULL_MODE="${ACTKERNEL_NULL_MODE:-}"
ACTKERNEL_PSEUDO_LEN_FACTOR="${ACTKERNEL_PSEUDO_LEN_FACTOR:-}"

module load miniforge
conda activate ~/conda_envs/ibl
cd "$REPO_DIR"

echo "Host: $(hostname) Date: $(date) SPLIT=$SPLIT"
echo "exclude_sticky=$EXCLUDE_STICKY_TRIALS session_shuffle=$SESSION_SHUFFLE_NULL actkernel=$ACTKERNEL_CHOICE_NULL mode=${ACTKERNEL_NULL_MODE:-} pseudo_len_factor=${ACTKERNEL_PSEUDO_LEN_FACTOR:-1}"
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-?}"
ARGS=(--finalize-only --splits "$SPLIT")
[[ "$EXCLUDE_STICKY_TRIALS" == "1" ]] && ARGS+=(--exclude-sticky-trials)
[[ "$SESSION_SHUFFLE_NULL" == "1" ]] && ARGS+=(--session-shuffle-null)
[[ "$ACTKERNEL_CHOICE_NULL" == "1" ]] && ARGS+=(--actkernel-choice-null)
[[ -n "$ACTKERNEL_NULL_MODE" ]] && ARGS+=(--actkernel-null-mode "$ACTKERNEL_NULL_MODE")
[[ -n "$ACTKERNEL_PSEUDO_LEN_FACTOR" ]] && ARGS+=(--actkernel-pseudo-len-factor "$ACTKERNEL_PSEUDO_LEN_FACTOR")
python3 -u scripts/run_goal2_splits.py "${ARGS[@]}"
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
ls -lh "$RES_ROOT/${SPLIT}${SUFFIX}"*.npy 2>/dev/null || true
