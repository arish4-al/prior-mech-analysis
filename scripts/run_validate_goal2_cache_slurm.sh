#!/bin/bash
#SBATCH --job-name=goal2_validate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -p mit_normal
#SBATCH --time=2:00:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o validate_goal2_cache_%j.out

# Goal 2 cached-vs-uncached parity check on BWM.
# Submit from repo root:
#   sbatch scripts/run_validate_goal2_cache_slurm.sh
#
# Optional overrides:
#   N_PIDS=20 N_RAND=50 sbatch scripts/run_validate_goal2_cache_slurm.sh

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
N_PIDS="${N_PIDS:-5}"
N_CACHE_PIDS="${N_CACHE_PIDS:-2}"
N_RAND="${N_RAND:-50}"
SEED="${SEED:-123}"

# module load miniforge
conda activate ~/conda_envs/ibl

cd "$REPO_DIR"
echo "Host: $(hostname)"
echo "Repo: $REPO_DIR"
echo "ONE_CACHE_DIR: $ONE_CACHE_DIR"
git log -1 --oneline

python scripts/validate_goal2_cache.py \
  --n-pids "$N_PIDS" \
  --n-cache-pids "$N_CACHE_PIDS" \
  --nrand "$N_RAND" \
  --seed "$SEED"

echo "validate_goal2_cache finished OK"
