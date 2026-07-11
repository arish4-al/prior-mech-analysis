#!/bin/bash
#SBATCH --job-name=goal2_split
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -p mit_normal
#SBATCH --time=24:00:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o goal2_split_%x_%j.out

# Phase 2: one split (requires insertion cache from run_goal2_cache_slurm.sh).
#
#   SPLIT=act_block_stim_l sbatch --job-name=g2_stim_l scripts/run_goal2_one_split_slurm.sh
#
# Or submit all 6 after cache via submit_goal2_stimOn_act_parallel.sh

set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"

SPLIT="${SPLIT:?Set SPLIT=split_name}"
NRAND="${NRAND:-2000}"
RESTART="${RESTART:-1}"
STREAM_POOL="${STREAM_POOL:-1}"

module load miniforge
conda activate ~/conda_envs/ibl
cd "$REPO_DIR"

echo "Host: $(hostname)  Date: $(date)"
git log -1 --oneline
echo "SPLIT=$SPLIT  nrand=$NRAND  restart=$RESTART"

ARGS=(--splits "$SPLIT" --nrand "$NRAND" --no-save-cache)
[[ "$RESTART" == "1" ]] && ARGS+=(--restart) || ARGS+=(--no-restart)
[[ "$STREAM_POOL" == "1" ]] && ARGS+=(--stream-pool) || ARGS+=(--no-stream-pool)

python3 -u scripts/run_goal2_splits.py "${ARGS[@]}"

echo "Done $SPLIT: $(date)"
ls -lh "$ONE_CACHE_DIR/manifold/res/${SPLIT}"*.npy 2>/dev/null || true
