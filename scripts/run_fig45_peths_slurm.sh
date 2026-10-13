#!/bin/bash
#SBATCH --job-name=fig45_peths
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH -p pi_fiete
#SBATCH --time=4:00:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o fig45_peths_%x_%j.out

# Figure 4.5 block-pair PETHs + block-label shuffles for one insertion shard
# (STEP=compute), or stack + distance plots after all shards (STEP=stack).
# Reads manifold/insertion_cache + bwm_tables/trials.pqt under ONE_CACHE_DIR;
# falls back to ONE only for insertions missing from the cache.
# Env: STEP, SHARD_IDX, N_SHARDS, NRAND, SEED, TYPES, OUT, CELL_TYPES, SOURCE, PRIOR

set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"

STEP="${STEP:?Set STEP=compute|stack}"
OUT="${OUT:-$ONE_CACHE_DIR/dmn/res/fig45/cvTrue}"
CELL_TYPES="${CELL_TYPES:-$REPO_DIR/data/fig45/cell_types.npy}"
TYPES="${TYPES:-integrator move_all move_init move move_fback}"
NRAND="${NRAND:-2000}"
SEED="${SEED:-0}"
SOURCE="${SOURCE:-auto}"
TAG="${TAG:-act}"
# act = action-kernel (alpha 0.2) subjective block labels for PETHs and shuffles
PRIOR="${PRIOR:-act}"

module load miniforge
conda activate ~/conda_envs/ibl
cd "$REPO_DIR/scripts"
echo "Host: $(hostname) Date: $(date) STEP=$STEP OUT=$OUT TYPES=$TYPES"
git log -1 --oneline

# shellcheck disable=SC2086
if [[ "$STEP" == "compute" ]]; then
  python3 -u fig45_block_peths.py compute \
    --shard "${SHARD_IDX:?Set SHARD_IDX}" --n-shards "${N_SHARDS:?Set N_SHARDS}" \
    --nrand "$NRAND" --seed "$SEED" --source "$SOURCE" --prior "$PRIOR" \
    --cell-types "$CELL_TYPES" --out "$OUT" --one-cache-dir "$ONE_CACHE_DIR" \
    --types $TYPES
elif [[ "$STEP" == "stack" ]]; then
  python3 -u fig45_block_peths.py stack --out "$OUT" --prior "$PRIOR" --types $TYPES
  python3 -u fig45_plot_distances.py --root "$OUT" --tag "$TAG" --types $TYPES
else
  echo "unknown STEP=$STEP" >&2; exit 2
fi
echo "Done: $(date)"
