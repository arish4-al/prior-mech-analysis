#!/bin/bash
#SBATCH --job-name=goal2_actkernel_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -p mit_normal
#SBATCH --time=2:00:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o goal2_actkernel_smoke_%j.out

# Short smoke for BWM ActionKernel choice nulls (compute node).
# Prefer: SMOKE_FIRST=1 bash scripts/submit_goal2_choice_actkernel_null_sharded.sh

set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"
export SMOKE_NRAND="${SMOKE_NRAND:-8}"
export ACTKERNEL_NB_STEPS="${ACTKERNEL_NB_STEPS:-80}"
export ACTKERNEL_NULL_MODE="${ACTKERNEL_NULL_MODE:-strat}"
export ACTKERNEL_PSEUDO_LEN_FACTOR="${ACTKERNEL_PSEUDO_LEN_FACTOR:-3}"

module load miniforge
conda activate ~/conda_envs/ibl
cd "$REPO_DIR"

if [[ ! -d third_party/behavior_models/behavior_models ]]; then
  echo "ERROR: missing submodule third_party/behavior_models" >&2
  echo "  git submodule update --init --recursive" >&2
  exit 1
fi

echo "Host: $(hostname) Date: $(date)"
git log -1 --oneline
echo "ONE_CACHE_DIR=$ONE_CACHE_DIR SMOKE_NRAND=$SMOKE_NRAND ACTKERNEL_NB_STEPS=$ACTKERNEL_NB_STEPS MODE=$ACTKERNEL_NULL_MODE FACTOR=$ACTKERNEL_PSEUDO_LEN_FACTOR"

python3 -u scripts/smoke_choice_actkernel_null.py
echo "Smoke done: $(date)"
