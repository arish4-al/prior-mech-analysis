#!/bin/bash
#SBATCH --job-name=fig45_sens
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p pi_fiete
#SBATCH --time=6:00:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o fig45_sens_%x_%j.out

# fig45_sensitivity.py on ORCD (insertion cache + bwm_tables/trials.pqt).
#   STEP=compute STAGE=A|B SHARD_IDX N_SHARDS   one shard (checkpointed; requeue-safe)
#   STEP=stack   STAGE=A|B                      merge shards, plots, summary.json
#   STEP=select                                 stage A -> stageB_cell_types.npy
#   STEP=compare                                A + B -> compare/ (also run after stack B)
# Env: NRAND, SEED, PRIOR (act), Q, OUT, CELL_TYPES, SOURCE

set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"

STEP="${STEP:?Set STEP=compute|stack|select}"
OUT="${OUT:-$ONE_CACHE_DIR/dmn/res/fig45/cvTrue/sensitivity}"
CELL_TYPES="${CELL_TYPES:-$ONE_CACHE_DIR/dmn/res/fig45/cvTrue/cell_types.npy}"
NRAND="${NRAND:-2000}"
SEED="${SEED:-0}"
PRIOR="${PRIOR:-act}"
Q="${Q:-0.05}"
SOURCE="${SOURCE:-auto}"

module load miniforge
conda activate ~/conda_envs/ibl
cd "$REPO_DIR/scripts"
echo "Host: $(hostname) Date: $(date) STEP=$STEP STAGE=${STAGE:-} OUT=$OUT"
git log -1 --oneline

case "$STEP" in
  compute)
    CT_ARGS=()
    [[ "${STAGE:?}" == "A" ]] && CT_ARGS=(--cell-types "$CELL_TYPES")
    python3 -u fig45_sensitivity.py compute --stage "$STAGE" \
      --shard "${SHARD_IDX:?}" --n-shards "${N_SHARDS:?}" \
      --nrand "$NRAND" --seed "$SEED" --prior "$PRIOR" --source "$SOURCE" \
      --out "$OUT" --one-cache-dir "$ONE_CACHE_DIR" "${CT_ARGS[@]}" ;;
  stack)
    python3 -u fig45_sensitivity.py stack --stage "${STAGE:?}" --out "$OUT"
    [[ "$STAGE" == "B" ]] && python3 -u fig45_sensitivity.py compare --out "$OUT" ;;
  compare)
    python3 -u fig45_sensitivity.py compare --out "$OUT" ;;
  select)
    python3 -u fig45_sensitivity.py select --out "$OUT" --cell-types "$CELL_TYPES" --q "$Q" ;;
  *)
    echo "unknown STEP=$STEP" >&2; exit 2 ;;
esac
echo "Done: $(date)"
