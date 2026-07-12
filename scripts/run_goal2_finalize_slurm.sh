#!/bin/bash
#SBATCH --job-name=goal2_finalize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -p mit_normal
#SBATCH --time=2:00:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o goal2_finalize_%x_%j.out

# Merge stream_acc shards → manifold/res/{split}*.npy
#   SPLIT=act_block_stim_l sbatch --export=ALL,SPLIT --job-name=g2_fin_stim_l \
#     scripts/run_goal2_finalize_slurm.sh

set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"

SPLIT="${SPLIT:?Set SPLIT=}"

module load miniforge
conda activate ~/conda_envs/ibl
cd "$REPO_DIR"

echo "Host: $(hostname) Date: $(date) SPLIT=$SPLIT"
python3 -u scripts/run_goal2_splits.py --finalize-only --splits "$SPLIT"
ls -lh "$ONE_CACHE_DIR/manifold/res/${SPLIT}"*.npy 2>/dev/null || true
