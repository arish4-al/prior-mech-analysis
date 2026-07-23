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
# Peak RSS (stream_pool, nrand=2000): ~1.5–2.5 GB (journal 07-10b); stream_acc
# grows with insertions in the shard. Default 6G ≈ 2–4× headroom (was 12G).
# Submitters override via --mem= (see submit_goal2_stimOn_act_sharded.sh).
#
#   SPLIT=act_block_duringstim_l SHARD_IDX=0 N_SHARDS=4 sbatch \
#     --export=ALL,SPLIT,SHARD_IDX,N_SHARDS \
#     --job-name=g2_stim_l_s0 scripts/run_goal2_shard_slurm.sh

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

module load miniforge
conda activate ~/conda_envs/ibl
cd "$REPO_DIR"

if [[ "$ACTKERNEL_CHOICE_NULL" == "1" && ! -d third_party/behavior_models/behavior_models ]]; then
  echo "ERROR: ACTKERNEL_CHOICE_NULL=1 but missing submodule third_party/behavior_models" >&2
  echo "  git submodule update --init --recursive" >&2
  exit 1
fi

echo "Host: $(hostname) Date: $(date)"
git log -1 --oneline
echo "SPLIT=$SPLIT shard=$SHARD_IDX/$N_SHARDS nrand=$NRAND session_shuffle_null=$SESSION_SHUFFLE_NULL exclude_sticky=$EXCLUDE_STICKY_TRIALS actkernel_choice=$ACTKERNEL_CHOICE_NULL"
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-?} SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}"

ARGS=(--splits "$SPLIT" --nrand "$NRAND" --no-save-cache
      --shard-idx "$SHARD_IDX" --n-shards "$N_SHARDS" --no-finalize)
[[ "$RESTART" == "1" ]] && ARGS+=(--restart) || ARGS+=(--no-restart)
ARGS+=(--stream-pool)
[[ "$SESSION_SHUFFLE_NULL" == "1" ]] && ARGS+=(--session-shuffle-null)
[[ "$ACTKERNEL_CHOICE_NULL" == "1" ]] && ARGS+=(--actkernel-choice-null)
[[ "$EXCLUDE_STICKY_TRIALS" == "1" ]] && ARGS+=(
  --exclude-sticky-trials
  --sticky-late-frac "$STICKY_LATE_FRAC"
  --sticky-min-run "$STICKY_MIN_RUN"
)

python3 -u scripts/run_goal2_splits.py "${ARGS[@]}"
echo "Shard done: $(date)"
RES_ROOT="$ONE_CACHE_DIR/manifold/res"
[[ "$EXCLUDE_STICKY_TRIALS" == "1" ]] && RES_ROOT="$ONE_CACHE_DIR/manifold/res_excl_sticky"
ls -lh "$RES_ROOT/_stream_acc/${SPLIT}.shard${SHARD_IDX}.npy" 2>/dev/null || true
