#!/bin/bash
#SBATCH --job-name=goal2_stimOn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -p mit_normal
#SBATCH --time=72:00:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o goal2_stimOn_act_%j.out

# Goal 2 BWM run: run_align['stimOn_times'] action-kernel during-stim splits
# plus new act_block_stim_l / act_block_stim_r (L vs R trajectories within
# left- or right-stim trials, action-kernel prior).
#
# Pipeline: insertion cache (once per insertion) + stream_pool accumulators
# → manifold/res/{split}.npy + {split}_regde.npy
#
# Submit from repo root:
#   sbatch scripts/run_goal2_bwm_stimOn_slurm.sh
#
# Overrides:
#   PRESET=stimOn_times_act NRAND=2000 RESTART=1 sbatch scripts/run_goal2_bwm_stimOn_slurm.sh
#   SPLITS="act_block_stim_l act_block_stim_r" sbatch ...   # explicit list instead of preset

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

PRESET="${PRESET:-stimOn_times_act}"
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
echo "Python: $(which python3)"
echo "Preset: $PRESET  nrand: $NRAND  restart: $RESTART  stream_pool: $STREAM_POOL"

ARGS=(--nrand "$NRAND")
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
if [[ -n "${SPLITS:-}" ]]; then
  ARGS+=(--splits $SPLITS)
else
  ARGS+=(--preset "$PRESET")
fi

python3 -u scripts/run_goal2_splits.py "${ARGS[@]}"

echo "Finished: $(date)"
echo "Outputs:"
ls -lh "$ONE_CACHE_DIR/manifold/res/"*.npy 2>/dev/null | tail -20 || true
echo "Stream acc checkpoints:"
ls -lh "$ONE_CACHE_DIR/manifold/res/_stream_acc/" 2>/dev/null || true
