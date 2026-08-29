#!/bin/bash
# Submit weights fits (retinal frozen) for several model variants x seeds on ORCD.
#
# A model VARIANT is "mtype:freeze" where freeze is a |-separated index list (empty
# = nothing frozen). Each (variant, seed) is one independent, restartable fit job
# with its own deterministic run dir. Sweeping seeds is multi-start (looks for
# alternative parameter sets of similar quality and runs fits in parallel).
# Mirrors the goal2 two-layer submitter+worker pattern.
#
# Usage:
#   # two variants (none, and gain with g_m,d_m frozen) x 3 seeds:
#   VARIANTS="none: gain:7|9" SEEDS="56 34 78" bash scripts/submit_fit_weights_sharded.sh
#
#   # single variant, custom resources:
#   VARIANTS="none:" SEEDS="1 2 3 4 5" MEM=48G TIME=6:00:00 \
#     bash scripts/submit_fit_weights_sharded.sh
#
#   # warm-refine (cma_only) from an existing weights JSON:
#   PIPELINE=cma_only RESUME_JSON=/path/weights.json VARIANTS="none:" SEEDS="1 2" \
#     bash scripts/submit_fit_weights_sharded.sh
#
# Env knobs:
#   VARIANTS  space list of mtype:freeze (freeze uses '|' between indices)  default "none: gain:7|9"
#   SEEDS     space-separated seeds                                          default "56 34 78 89 202"
#   PIPELINE  de_cma_local|de_cma|cma_only                                   default de_cma_local
#   OUT_TAG   optional campaign tag in the run-dir name
#   RESUME_JSON  external warm start (required for cma_only)
#   FORCE=1   re-run even if a variant/seed already finished (FIT_DONE)
#   MEM CPUS TIME  sbatch resource overrides (default 40G / 16 / 6:00:00)
#   DE1_MAXITER DE2_MAXITER DE_POPSIZE POPSIZE SOBOL_COUNT PATIENCE BEAT_LOSS
#   BPS_STAGE1 BPS_STAGE2 STAGE2_N_STIM_SEEDS (default 3)
#   STAGE2_STIM_AGGREGATE (sample|mean; default sample)
#   VAL_SEED LOCAL_REFINE_IDX (prior) LOCAL_REFINE_METHOD (powell)
#   LOCAL_REFINE_MAX_WALL_S (default 1800) BACKEND

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

PARTITION="${PARTITION:-pi_fiete}"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/sbatch_defaults.sh"

VARIANTS="${VARIANTS:-none: gain:7|9}"
SEEDS="${SEEDS:-56 34 78 89 202}"
PIPELINE="${PIPELINE:-de_cma_local}"
OUT_TAG="${OUT_TAG:-}"
RESUME_JSON="${RESUME_JSON:-}"
FORCE="${FORCE:-0}"
MEM="${MEM:-40G}"
CPUS="${CPUS:-16}"
TIME="${TIME:-6:00:00}"

# fit knobs forwarded to the worker (exported below, picked up via --export=ALL)
export DE1_MAXITER="${DE1_MAXITER:-40}"
export DE2_MAXITER="${DE2_MAXITER:-40}"
export DE_POPSIZE="${DE_POPSIZE:-8}"
export POPSIZE="${POPSIZE:-16}"
export SOBOL_COUNT="${SOBOL_COUNT:-8}"
export PATIENCE="${PATIENCE:-8}"
export BEAT_LOSS="${BEAT_LOSS:-0.4044}"
export BPS_STAGE1="${BPS_STAGE1:-5}"
export BPS_STAGE2="${BPS_STAGE2:-20}"
export STAGE2_N_STIM_SEEDS="${STAGE2_N_STIM_SEEDS:-3}"
export STAGE2_STIM_AGGREGATE="${STAGE2_STIM_AGGREGATE:-sample}"
export VAL_SEED="${VAL_SEED:-}"
export LOCAL_REFINE_IDX="${LOCAL_REFINE_IDX:-prior}"
export LOCAL_REFINE_METHOD="${LOCAL_REFINE_METHOD:-powell}"
export LOCAL_REFINE_MAX_WALL_S="${LOCAL_REFINE_MAX_WALL_S:-1800}"
export BACKEND="${BACKEND:-loky}"
export PIPELINE OUT_TAG RESUME_JSON FORCE REPO_DIR

ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"

if [[ "$PIPELINE" == "cma_only" && -z "$RESUME_JSON" ]]; then
  echo "ERROR: PIPELINE=cma_only requires RESUME_JSON=/path/to/weights.json" >&2
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
  # variant token uses '|' between indices (commas break sbatch --export); worker
  # wants commas.
  FREEZE="${FREEZE_RAW//|/,}"
  SLUG="${FREEZE_RAW//|/-}"; [[ -z "$SLUG" ]] && SLUG="none"
  export MTYPE FREEZE
  for SEED in "${SEED_ARR[@]}"; do
    export SEED
    # shellcheck disable=SC2086
    JID=$(sbatch --parsable $SBATCH_EXTRA \
      --partition="$PARTITION" \
      --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" \
      --job-name="fw_${MTYPE}_m${SLUG}_s${SEED}" \
      --export=ALL \
      scripts/run_fit_weights_slurm.sh)
    echo "  ${MTYPE} mask=${SLUG} seed=${SEED} -> $JID"
  done
done

echo "Done. Monitor: squeue -u \$USER"
echo "Run dirs: \$save_dir/weights_run_fw[_${OUT_TAG}]_<mtype>_mask<slug>_s<seed>/"
echo "Restart a killed job: re-run the same command (resumes from weights_stage2_last.json)."
