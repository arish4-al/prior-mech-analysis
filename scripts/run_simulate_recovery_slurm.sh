#!/bin/bash
#SBATCH --job-name=sim_recovery
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p mit_normal
#SBATCH --time=24:00:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o simulate_recovery_%j.out

# Circuit generative recovery: 40 sessions x 6 blocks, nrand=2000, absence + presence.
# Default: S-prior distance only (4 splits, population S), null shuffles parallelized.
# Submit from prior-mech-analysis repo root:
#   sbatch scripts/run_simulate_recovery_slurm.sh
#
# Override paths if needed:
#   REPO_DIR=/path/to/prior-mech-analysis WEIGHTS_JSON=/path/to/weights.json sbatch ...
# Full slow pipeline (all splits + classifiers):
#   FULL_ANALYSIS=1 sbatch scripts/run_simulate_recovery_slurm.sh

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-/home/arily/int-brain-lab/prior-mech-analysis}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
WEIGHTS_JSON="${WEIGHTS_JSON:-$ONE_CACHE_DIR/models/weights_run_20251125_182058/weights_2stagelocalrefine_loss0p4044_20251125-195255.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$ONE_CACHE_DIR/manifold_sim}"

N_SESSIONS="${N_SESSIONS:-40}"
NRAND="${NRAND:-2000}"
BLOCKS_PER_SESSION="${BLOCKS_PER_SESSION:-6}"
SEED="${SEED:-42}"
N_JOBS="${N_JOBS:-8}"
FULL_ANALYSIS="${FULL_ANALYSIS:-0}"

module load miniforge
conda activate ~/conda_envs/ibl

cd "$REPO_DIR"

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Repo: $REPO_DIR"
echo "ONE cache: $ONE_CACHE_DIR"
echo "Python: $(which python3)"
echo "Weights: $WEIGHTS_JSON"
echo "Output: $OUTPUT_DIR"
echo "Config: n_sessions=$N_SESSIONS nrand=$NRAND blocks=$BLOCKS_PER_SESSION seed=$SEED n_jobs=$N_JOBS full_analysis=$FULL_ANALYSIS"

if [[ ! -f "$WEIGHTS_JSON" ]]; then
  echo "ERROR: weights JSON not found: $WEIGHTS_JSON" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ "$FULL_ANALYSIS" == "1" ]]; then
  EXTRA_ARGS+=(--full-analysis)
fi

python3 -u simulate_recovery.py \
  --n-sessions "$N_SESSIONS" \
  --nrand "$NRAND" \
  --blocks-per-session "$BLOCKS_PER_SESSION" \
  --output-dir "$OUTPUT_DIR" \
  --weights-json "$WEIGHTS_JSON" \
  --seed "$SEED" \
  --n-jobs "$N_JOBS" \
  "${EXTRA_ARGS[@]}"

echo "Done: $(date)"
echo "Top-level figures:"
ls -la "$OUTPUT_DIR"/*.png 2>/dev/null || true
echo "Per-condition shuffle controls:"
ls -la "$OUTPUT_DIR"/absence/figs/s_shuffle_control.png "$OUTPUT_DIR"/presence/figs/s_shuffle_control.png 2>/dev/null || true
