#!/bin/bash
#SBATCH --job-name=goal2_finalize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH -p mit_normal
#SBATCH --time=2:00:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o goal2_finalize_%x_%j.out

# Merge stream_acc shards → manifold/res/{split}*.npy
# Loads all shard checkpoints into memory; default 10G (was 16G/32G).
# Submitters override via --mem= (see submit_goal2_stimOn_act_sharded.sh).
#
#   SPLIT=act_block_duringstim_l sbatch --export=ALL,SPLIT --job-name=g2_fin_dstim_l \
#     scripts/run_goal2_finalize_slurm.sh

set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"

SPLIT="${SPLIT:?Set SPLIT=}"
EXCLUDE_STICKY_TRIALS="${EXCLUDE_STICKY_TRIALS:-0}"

module load miniforge
conda activate ~/conda_envs/ibl
cd "$REPO_DIR"

echo "Host: $(hostname) Date: $(date) SPLIT=$SPLIT exclude_sticky=$EXCLUDE_STICKY_TRIALS"
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-?}"
ARGS=(--finalize-only --splits "$SPLIT")
[[ "$EXCLUDE_STICKY_TRIALS" == "1" ]] && ARGS+=(--exclude-sticky-trials)
python3 -u scripts/run_goal2_splits.py "${ARGS[@]}"
RES_ROOT="$ONE_CACHE_DIR/manifold/res"
[[ "$EXCLUDE_STICKY_TRIALS" == "1" ]] && RES_ROOT="$ONE_CACHE_DIR/manifold/res_excl_sticky"
ls -lh "$RES_ROOT/${SPLIT}"*.npy 2>/dev/null || true
