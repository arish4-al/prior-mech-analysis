#!/bin/bash
# Submit retinal Stage-A fits for several seeds (optional freeze variants) on ORCD.
#
# Usage:
#   SEEDS="56 34 78 89 202" bash scripts/submit_fit_retinal_sharded.sh
#
#   # tiny smoke:
#   SEEDS="999" OUT_TAG=smoke DE1_MAXITER=2 DE2_MAXITER=3 POPSIZE=8 SOBOL_COUNT=4 \
#     PATIENCE=0 LOCAL_REFINE_MAX_WALL_S=60 FORCE=1 \
#     bash scripts/submit_fit_retinal_sharded.sh
#
# Env knobs:
#   VARIANTS  space list of mtype:freeze (default "retinal:")
#   SEEDS     space-separated seeds (default "56 34 78 89 202")
#   PIPELINE  de_cma_local|de_cma|cma_only
#   OUT_TAG RESUME_JSON FORCE MEM CPUS TIME
#   DE1_MAXITER DE2_MAXITER DE_POPSIZE POPSIZE SOBOL_COUNT PATIENCE
#   BEAT_LOSS L_THRESHOLD BPS_STAGE1 BPS_STAGE2 STAGE2_N_STIM_SEEDS
#   STAGE2_STIM_AGGREGATE VAL_SEED LOCAL_REFINE_* BACKEND

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

VARIANTS="${VARIANTS:-retinal:}"
SEEDS="${SEEDS:-56 34 78 89 202}"
PIPELINE="${PIPELINE:-de_cma_local}"
OUT_TAG="${OUT_TAG:-}"
RESUME_JSON="${RESUME_JSON:-}"
FORCE="${FORCE:-0}"
MEM="${MEM:-40G}"
CPUS="${CPUS:-16}"
TIME="${TIME:-6:00:00}"

export DE1_MAXITER="${DE1_MAXITER:-40}"
export DE2_MAXITER="${DE2_MAXITER:-40}"
export DE_POPSIZE="${DE_POPSIZE:-8}"
export POPSIZE="${POPSIZE:-16}"
export SOBOL_COUNT="${SOBOL_COUNT:-8}"
export PATIENCE="${PATIENCE:-8}"
export BEAT_LOSS="${BEAT_LOSS:-0.85}"
export L_THRESHOLD="${L_THRESHOLD:-2}"
export BPS_STAGE1="${BPS_STAGE1:-10}"
export BPS_STAGE2="${BPS_STAGE2:-20}"
export STAGE2_N_STIM_SEEDS="${STAGE2_N_STIM_SEEDS:-3}"
export STAGE2_STIM_AGGREGATE="${STAGE2_STIM_AGGREGATE:-sample}"
export VAL_SEED="${VAL_SEED:-}"
export LOCAL_REFINE_IDX="${LOCAL_REFINE_IDX:-retinal}"
export LOCAL_REFINE_METHOD="${LOCAL_REFINE_METHOD:-powell}"
export LOCAL_REFINE_MAX_WALL_S="${LOCAL_REFINE_MAX_WALL_S:-1800}"
export BACKEND="${BACKEND:-loky}"
export PIPELINE OUT_TAG RESUME_JSON FORCE REPO_DIR

ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"

if [[ "$PIPELINE" == "cma_only" && -z "$RESUME_JSON" ]]; then
  echo "ERROR: PIPELINE=cma_only requires RESUME_JSON=/path/to/retinal.json" >&2
  exit 1
fi

module load miniforge 2>/dev/null || true

read -r -a VAR_ARR <<< "$VARIANTS"
read -r -a SEED_ARR <<< "$SEEDS"
n_jobs=$(( ${#VAR_ARR[@]} * ${#SEED_ARR[@]} ))
echo "PIPELINE=$PIPELINE  variants=(${VARIANTS})  seeds=(${SEEDS})  jobs=$n_jobs"
echo "MEM=$MEM CPUS=$CPUS TIME=$TIME  RESUME_JSON=${RESUME_JSON:-none}  FORCE=$FORCE"

for VAR in "${VAR_ARR[@]}"; do
  MTYPE="${VAR%%:*}"
  FREEZE_RAW="${VAR#*:}"
  FREEZE="${FREEZE_RAW//|/,}"
  SLUG="${FREEZE_RAW//|/-}"; [[ -z "$SLUG" ]] && SLUG="none"
  export MTYPE FREEZE
  for SEED in "${SEED_ARR[@]}"; do
    export SEED
    JID=$(sbatch --parsable \
      --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" \
      --job-name="fr_${MTYPE}_m${SLUG}_s${SEED}" \
      --export=ALL \
      scripts/run_fit_retinal_slurm.sh)
    echo "  ${MTYPE} mask=${SLUG} seed=${SEED} -> $JID"
  done
done

echo "Done. Monitor: squeue -u \$USER"
echo "Run dirs: \$save_dir/retinal_run_fr[_${OUT_TAG}]_<mtype>_mask<slug>_s<seed>/"
echo "Restart a killed job: re-run the same command (resumes from weights_stage2_last.json)."
