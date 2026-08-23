#!/bin/bash
#SBATCH --job-name=fit_weights
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH -p pi_fiete
#SBATCH --time=6:00:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o fit_weights_%x_%j.out

# One weights fit (retinal frozen) on ORCD, for one model variant x seed.
# Variant = (MTYPE label, FREEZE indices). Restartable: the run dir is
# deterministic per (MTYPE, mask, SEED) and --resume auto continues from the
# rolling Stage-2 checkpoint.
#
# Env: MTYPE, FREEZE (e.g. "7,9" or ""), SEED, PIPELINE (de_cma_local|de_cma|cma_only),
#      OUT_TAG (optional campaign tag), RESUME_JSON (opt external warm start), FORCE,
#      DE1_MAXITER, DE2_MAXITER, DE_POPSIZE, POPSIZE, SOBOL_COUNT, PATIENCE,
#      BEAT_LOSS, BPS_STAGE1, BPS_STAGE2, STAGE2_N_STIM_SEEDS (default 3),
#      STAGE2_STIM_AGGREGATE (sample|mean; default sample ≈1× wall), BACKEND,
#      VAL_SEED (held-out stim seed; default SEED+7777),
#      LOCAL_REFINE_IDX (prior|active|"6,8,10,11"; default prior = g_i,d_i,θ),
#      LOCAL_REFINE_METHOD (powell|cma; default powell),
#      LOCAL_REFINE_MAX_WALL_S (default 1800s; set 0 to disable).
# Parallel workers come from SLURM_CPUS_PER_TASK.

set -euo pipefail

# Prevent BLAS oversubscription — joblib does the parallelism.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"
# Fitting doesn't need ONE (only cache_dir). Bypass ONE construction so parallel
# workers don't race on ~/.one params and no network is needed on compute nodes.
export PRIOR_MECH_NO_ONE="${PRIOR_MECH_NO_ONE:-1}"

MTYPE="${MTYPE:-none}"
# Accept '|' as index separator too — commas break `sbatch --export=ALL,FREEZE=7,9,...`
FREEZE="${FREEZE:-}"
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
BEAT_LOSS="${BEAT_LOSS:-0.4044}"
BPS_STAGE1="${BPS_STAGE1:-5}"
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

# Ensure fit targets from repo fit_targets/ (Python drivers also refresh these).
if [[ -d "$REPO_DIR/fit_targets" ]]; then
  for name in avg_mean_R.npy mean_data_results.npy \
              data_act_block_duringstim.npy data_act_block_duringchoice.npy; do
    if [[ -f "$REPO_DIR/fit_targets/$name" ]]; then
      ln -sfn "$REPO_DIR/fit_targets/$name" "$name"
    fi
  done
fi

echo "Host: $(hostname) Date: $(date)"
git log -1 --oneline 2>/dev/null || true
echo "MTYPE=$MTYPE FREEZE='${FREEZE}' SEED=$SEED PIPELINE=$PIPELINE OUT_TAG=${OUT_TAG:-none}"
echo "RESUME_JSON=${RESUME_JSON:-none} FORCE=$FORCE"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?} SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-?}"

ARGS=(--mtype "$MTYPE" --freeze "$FREEZE" --seed "$SEED"
      --pipeline "$PIPELINE" --n-jobs "${SLURM_CPUS_PER_TASK:-16}" --backend "$BACKEND"
      --de1-maxiter "$DE1_MAXITER" --de2-maxiter "$DE2_MAXITER"
      --de-popsize "$DE_POPSIZE" --popsize "$POPSIZE" --sobol-count "$SOBOL_COUNT"
      --patience "$PATIENCE" --beat-loss "$BEAT_LOSS"
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

python3 -u scripts/run_fit_weights.py "${ARGS[@]}"
echo "Fit done: $(date)"
