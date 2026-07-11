#!/bin/bash
#SBATCH --job-name=goal2_cache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -p mit_normal
#SBATCH --time=24:00:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o goal2_cache_%j.out

# Phase 1: build manifold/insertion_cache for all BWM insertions (once).
# Submit before parallel split jobs:
#   sbatch scripts/run_goal2_cache_slurm.sh

set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"
RESTART="${RESTART:-1}"

module load miniforge
conda activate ~/conda_envs/ibl
cd "$REPO_DIR"

echo "Host: $(hostname)  Date: $(date)"
git log -1 --oneline
echo "ONE_CACHE_DIR: $ONE_CACHE_DIR"

ARGS=(--cache-only)
[[ "$RESTART" == "1" ]] && ARGS+=(--restart) || ARGS+=(--no-restart)

python3 -u scripts/run_goal2_splits.py "${ARGS[@]}"

echo "Cache done: $(date)"
ls -lh "$ONE_CACHE_DIR/manifold/insertion_cache/" 2>/dev/null | tail -5 || true
