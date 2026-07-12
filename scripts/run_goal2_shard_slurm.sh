#!/bin/bash
#SBATCH --job-name=goal2_shard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -p mit_normal
#SBATCH --time=12:00:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o goal2_shard_%x_%j.out

# One insertion shard for one split (no finalize).
#   SPLIT=act_block_stim_l SHARD_IDX=0 N_SHARDS=4 sbatch \
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

module load miniforge
conda activate ~/conda_envs/ibl
cd "$REPO_DIR"

echo "Host: $(hostname) Date: $(date)"
git log -1 --oneline
echo "SPLIT=$SPLIT shard=$SHARD_IDX/$N_SHARDS nrand=$NRAND"

ARGS=(--splits "$SPLIT" --nrand "$NRAND" --no-save-cache
      --shard-idx "$SHARD_IDX" --n-shards "$N_SHARDS" --no-finalize)
[[ "$RESTART" == "1" ]] && ARGS+=(--restart) || ARGS+=(--no-restart)
ARGS+=(--stream-pool)

python3 -u scripts/run_goal2_splits.py "${ARGS[@]}"
echo "Shard done: $(date)"
ls -lh "$ONE_CACHE_DIR/manifold/res/_stream_acc/${SPLIT}.shard${SHARD_IDX}.npy" 2>/dev/null || true
