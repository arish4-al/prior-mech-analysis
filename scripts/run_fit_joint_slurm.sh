#!/bin/bash
#SBATCH --job-name=fit_joint
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH -p mit_normal
#SBATCH --time=6:00:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o fit_joint_%x_%j.out

# Joint fit (retinal + g_s/d_s + weights) on ORCD — one variant x seed.
# RESUME_JSON + PIPELINE=de_cma_local|de_cma → warm-start DE (x0 inject).
# RESUME_JSON + PIPELINE=cma_only → skip DE, Stage-2 CMA only.
# Env: MTYPE, FREEZE (e.g. "6|7|8|9"), SEED, PIPELINE, OUT_TAG, RESUME_JSON, FORCE,
#      DE1_MAXITER, DE2_MAXITER, DE_POPSIZE, POPSIZE, SOBOL_COUNT, PATIENCE,
#      BEAT_LOSS, L_THRESHOLD, BPS_STAGE1, BPS_STAGE2, STAGE2_N_STIM_SEEDS,
#      STAGE2_STIM_AGGREGATE, BACKEND, VAL_SEED,
#      LOCAL_REFINE_IDX (prior|sensory|retinal|active), LOCAL_REFINE_METHOD,
#      LOCAL_REFINE_MAX_WALL_S.

set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"
export PRIOR_MECH_NO_ONE="${PRIOR_MECH_NO_ONE:-1}"

MTYPE="${MTYPE:-sensory}"
FREEZE="${FREEZE:-6|7|8|9}"
FREEZE="${FREEZE//|/,}"
SEED="${SEED:-0}"
PIPELINE="${PIPELINE:-de_cma_local}"
OUT_TAG="${OUT_TAG:-}"
RESUME_JSON="${RESUME_JSON:-}"
FORCE="${FORCE:-0}"
DE1_MAXITER="${DE1_MAXITER:-40}"
DE2_MAXITER="${DE2_MAXITER:-40}"
DE_POPSIZE="${DE_POPSIZE:-8}"
POPSIZE="${POPSIZE:-16}"
SOBOL_COUNT="${SOBOL_COUNT:-8}"
PATIENCE="${PATIENCE:-8}"
BEAT_LOSS="${BEAT_LOSS:-1.2}"
L_THRESHOLD="${L_THRESHOLD:-10}"
# Stage 1: match fit_retinal session length (weights-only still uses 5).
BPS_STAGE1="${BPS_STAGE1:-10}"
BPS_STAGE2="${BPS_STAGE2:-20}"
STAGE2_N_STIM_SEEDS="${STAGE2_N_STIM_SEEDS:-3}"
STAGE2_STIM_AGGREGATE="${STAGE2_STIM_AGGREGATE:-sample}"
VAL_SEED="${VAL_SEED:-}"
BACKEND="${BACKEND:-loky}"
LOCAL_REFINE_IDX="${LOCAL_REFINE_IDX:-prior}"
LOCAL_REFINE_METHOD="${LOCAL_REFINE_METHOD:-powell}"
LOCAL_REFINE_MAX_WALL_S="${LOCAL_REFINE_MAX_WALL_S:-1800}"

module load miniforge
conda activate ~/conda_envs/ibl
cd "$REPO_DIR"

# Ensure avg_mean_R is available (paper-brain-wide-map sibling or ONE figs).
if [[ ! -f avg_mean_R.npy ]]; then
  for cand in \
    "$HOME/int-brain-lab/paper-brain-wide-map/avg_mean_R.npy" \
    "$ONE_CACHE_DIR/manifold/figs/avg_mean_R.npy" \
    "$ONE_CACHE_DIR/manifold/res/avg_mean_R.npy"; do
    if [[ -f "$cand" ]]; then
      ln -sfn "$cand" avg_mean_R.npy
      break
    fi
  done
fi

echo "Host: $(hostname) Date: $(date)"
git log -1 --oneline 2>/dev/null || true
echo "MTYPE=$MTYPE FREEZE='${FREEZE}' SEED=$SEED PIPELINE=$PIPELINE OUT_TAG=${OUT_TAG:-none}"
echo "RESUME_JSON=${RESUME_JSON:-none} FORCE=$FORCE L_THRESHOLD=$L_THRESHOLD"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?} SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-?}"

ARGS=(--mtype "$MTYPE" --freeze "$FREEZE" --seed "$SEED"
      --pipeline "$PIPELINE" --n-jobs "${SLURM_CPUS_PER_TASK:-16}" --backend "$BACKEND"
      --de1-maxiter "$DE1_MAXITER" --de2-maxiter "$DE2_MAXITER"
      --de-popsize "$DE_POPSIZE" --popsize "$POPSIZE" --sobol-count "$SOBOL_COUNT"
      --patience "$PATIENCE" --beat-loss "$BEAT_LOSS" --l-threshold "$L_THRESHOLD"
      --bps-stage1 "$BPS_STAGE1" --bps-stage2 "$BPS_STAGE2"
      --stage2-n-stim-seeds "$STAGE2_N_STIM_SEEDS"
      --stage2-stim-aggregate "$STAGE2_STIM_AGGREGATE"
      --local-refine-idx "$LOCAL_REFINE_IDX"
      --local-refine-method "$LOCAL_REFINE_METHOD"
      --local-refine-max-wall-s "$LOCAL_REFINE_MAX_WALL_S")
[[ -n "$VAL_SEED" ]] && ARGS+=(--val-seed "$VAL_SEED")
[[ -n "$OUT_TAG" ]] && ARGS+=(--out-tag "$OUT_TAG")
[[ -n "$RESUME_JSON" ]] && ARGS+=(--resume-json "$RESUME_JSON")
[[ "$FORCE" == "1" ]] && ARGS+=(--force)

python3 -u scripts/run_fit_joint.py "${ARGS[@]}"
echo "Joint fit done: $(date)"
