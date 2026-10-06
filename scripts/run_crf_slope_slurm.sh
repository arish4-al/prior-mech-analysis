#!/bin/bash
#SBATCH --job-name=crf_slope
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -p pi_fiete
#SBATCH --time=4:00:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o crf_slope_%x_%j.out

# One insertion shard of the contrast-response slope test.
# Submit via scripts/submit_crf_slope.sh

set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"

SHARD_IDX="${SHARD_IDX:?Set SHARD_IDX=0..N-1}"
N_SHARDS="${N_SHARDS:-4}"
NRAND="${NRAND:-10000}"
RESTART="${RESTART:-1}"

module load miniforge
conda activate ~/conda_envs/ibl
cd "$REPO_DIR"

echo "Host: $(hostname) Date: $(date)"
git log -1 --oneline
echo "crf_slope shard=$SHARD_IDX/$N_SHARDS nrand=$NRAND"
echo "ONE_CACHE_DIR=$ONE_CACHE_DIR"
echo "true-block prior | window 0-0.15 s | raw contrast including 0%"

ARGS=(--cached-only --local --nrand "$NRAND"
      --one-cache-dir "$ONE_CACHE_DIR"
      --shard-idx "$SHARD_IDX" --n-shards "$N_SHARDS" --no-stack)
if [[ "$RESTART" == "1" ]]; then
  ARGS+=(--restart)
else
  ARGS+=(--no-restart)
fi

python3 -u scripts/run_crf_slope.py "${ARGS[@]}"
echo "Shard done: $(date)"
