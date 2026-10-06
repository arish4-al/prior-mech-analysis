#!/bin/bash
#SBATCH --job-name=crf_fin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p pi_fiete
#SBATCH --time=0:30:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o crf_fin_%x_%j.out

# Stack manifold/crf_slope/*.npy into a region-level permutation test.

set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"

module load miniforge
conda activate ~/conda_envs/ibl
cd "$REPO_DIR"

echo "Host: $(hostname) Date: $(date)"
echo "ONE_CACHE_DIR=$ONE_CACHE_DIR"
python3 -u scripts/run_crf_slope.py --stack-only --local \
  --one-cache-dir "$ONE_CACHE_DIR"
echo "Finalize done: $(date)"
