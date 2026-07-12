#!/bin/bash
#SBATCH --job-name=goal3_contrast
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -p mit_normal
#SBATCH --time=12:00:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o goal3_%j.out

# Goal 3: single-job (unsharded) contrast-conditioned run.
# For full BWM on ORCD (12 h wall), use insertion sharding instead:
#   bash scripts/submit_goal3_sharded.sh
#   PRESET=goal3_duringstim_act CONTRASTS="0.0 0.125 1.0" N_SHARDS=4 \
#     bash scripts/submit_goal3_sharded.sh
#
# This script is fine for smoke / small contrast subsets / few insertions.
# Default preset: goal3_duringstim_act.
#   sbatch scripts/run_goal3_contrast_slurm.sh
#   PRESET=goal3_duringchoice_block CONTRASTS="0.0 1.0" sbatch ...

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR
export ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"

PRESET="${PRESET:-goal3_duringstim_act}"
NRAND="${NRAND:-2000}"
RESTART="${RESTART:-1}"
STREAM_POOL="${STREAM_POOL:-1}"

module load miniforge
conda activate ~/conda_envs/ibl

cd "$REPO_DIR"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Repo: $REPO_DIR"
git log -1 --oneline
echo "ONE_CACHE_DIR: $ONE_CACHE_DIR"
echo "Preset: $PRESET  nrand: $NRAND  contrasts: ${CONTRASTS:-all}"
echo "NOTE: for full BWM prefer bash scripts/submit_goal3_sharded.sh"

ARGS=(--preset "$PRESET" --nrand "$NRAND")
if [[ "$RESTART" == "1" ]]; then
  ARGS+=(--restart)
else
  ARGS+=(--no-restart)
fi
if [[ "$STREAM_POOL" == "1" ]]; then
  ARGS+=(--stream-pool)
else
  ARGS+=(--no-stream-pool)
fi
if [[ -n "${CONTRASTS:-}" ]]; then
  ARGS+=(--contrasts $CONTRASTS)
fi

python3 -u scripts/run_goal2_splits.py "${ARGS[@]}"

echo "Finished: $(date)"
ls -lh "$ONE_CACHE_DIR/manifold/res/"*duringstim*_*.npy 2>/dev/null | tail -30 || true
