#!/bin/bash
#SBATCH --job-name=sim_recovery
#SBATCH --output=logs/simulate_recovery_%j.out
#SBATCH --error=logs/simulate_recovery_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
# NOTE: set partition/account for your ORCD allocation, e.g.:
#SBATCH --partition=mit_normal
# #SBATCH --account=YOUR_ORCD_ACCOUNT

set -euo pipefail

# ---------------------------------------------------------------------------
# Edit these for your ORCD paths (or export before sbatch)
# ---------------------------------------------------------------------------
REPO_DIR="${REPO_DIR:-$HOME/prior-mech-analysis}"
CONDA_ENV="${CONDA_ENV:-iblenv}"
WEIGHTS_JSON="${WEIGHTS_JSON:-$HOME/ONE/openalyx.internationalbrainlab.org/models/weights_run_20251125_182058/weights_2stagelocalrefine_loss0p4044_20251125-195255.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/ONE/openalyx.internationalbrainlab.org/manifold_sim}"

N_SESSIONS="${N_SESSIONS:-40}"
Nrand="${NRAND:-2000}"
BLOCKS_PER_SESSION="${BLOCKS_PER_SESSION:-6}"
SEED="${SEED:-42}"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
cd "$REPO_DIR"
mkdir -p logs "$(dirname "$OUTPUT_DIR")"

# ORCD: load conda — adjust if you use miniconda/module differently
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/opt/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/opt/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Repo: $REPO_DIR"
echo "Python: $(which python)"
echo "Weights: $WEIGHTS_JSON"
echo "Output: $OUTPUT_DIR"
echo "Config: n_sessions=$N_SESSIONS nrand=$NRAND blocks=$BLOCKS_PER_SESSION"

if [[ ! -f "$WEIGHTS_JSON" ]]; then
  echo "ERROR: weights JSON not found: $WEIGHTS_JSON" >&2
  exit 1
fi

python -u simulate_recovery.py \
  --n-sessions "$N_SESSIONS" \
  --nrand "$NRAND" \
  --blocks-per-session "$BLOCKS_PER_SESSION" \
  --output-dir "$OUTPUT_DIR" \
  --weights-json "$WEIGHTS_JSON" \
  --seed "$SEED"

echo "Done: $(date)"
echo "Figures:"
ls -la "$OUTPUT_DIR"/*.png 2>/dev/null || true
ls -la "$OUTPUT_DIR"/absence/figs/*.png 2>/dev/null || true
ls -la "$OUTPUT_DIR"/presence/figs/*.png 2>/dev/null || true
