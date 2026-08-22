#!/bin/bash
#SBATCH --job-name=g2_ak_prefit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p mit_normal
#SBATCH --time=8:00:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o goal2_ak_prefit_%j.out

# MCMC-fit ActionKernel once per BWM eid → manifold/actkernel_fits/
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"

module load miniforge
conda activate ~/conda_envs/ibl
cd "$REPO_DIR"

if [[ ! -d third_party/behavior_models/behavior_models ]]; then
  echo "ERROR: missing submodule third_party/behavior_models" >&2
  exit 1
fi

echo "Host: $(hostname) Date: $(date)"
python3 -u scripts/run_goal2_splits.py --prefit-actkernel
echo "Prefit done: $(date)"
ls "$ONE_CACHE_DIR/manifold/actkernel_fits" | wc -l
