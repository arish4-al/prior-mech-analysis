#!/bin/bash
#SBATCH --job-name=fit_traj_targets
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -p mit_preemptable
#SBATCH --time=6:00:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o fit_traj_targets_%j.out

# I/M traj + S RMS fit targets for every TARGET_DIRS entry, one pass over
# manifold/insertion_cache. Submit via scripts/submit_fit_targets_integ.sh.

set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"
CACHE_DIR="${CACHE_DIR:-$ONE_CACHE_DIR/manifold/insertion_cache}"
PLOT_DIR="${PLOT_DIR:-$ONE_CACHE_DIR/manifold/fit_targets_integ}"
TARGET_DIRS="${TARGET_DIRS:-fit_targets_integ/opt1_rawslope fit_targets_integ/opt2_shuffleslope}"

module load miniforge
conda activate ~/conda_envs/ibl
cd "$REPO_DIR"

echo "Host: $(hostname) Date: $(date)"
git log -1 --oneline
echo "CACHE_DIR=$CACHE_DIR  n=$(ls -1 "$CACHE_DIR"/*.npy 2>/dev/null | wc -l)"
echo "TARGET_DIRS=$TARGET_DIRS  PLOT_DIR=$PLOT_DIR"

ARGS=()
for d in $TARGET_DIRS; do ARGS+=(--target-dir "$d"); done
python3 -u scripts/build_fit_traj_targets_from_cache.py \
  --cache-dir "$CACHE_DIR" --plot-dir "$PLOT_DIR" "${ARGS[@]}"

echo "Done: $(date)"
for d in $TARGET_DIRS; do ls -lh "$d"; done
