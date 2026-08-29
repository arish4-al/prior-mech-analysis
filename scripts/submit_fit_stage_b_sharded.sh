#!/bin/bash
# Stage B campaign: warm joint from Stage-A retinal ∪ WEIGHTS_REL hybrid.
#
# Builds (or reuses) the hybrid JSON, then submits regular + sensory variants
# with retinal held at Stage-A values during Stage-1 DE, then unfrozen for
# Stage-2 CMA / polish (--stage1-hold-retinal). Variant freeze masks still do
# NOT include 14–20.
#
# Usage:
#   # Smoke both masks (tiny budget):
#   SEEDS="999" OUT_TAG=stageB_smoke \
#     DE1_MAXITER=2 DE2_MAXITER=3 POPSIZE=8 SOBOL_COUNT=4 \
#     PATIENCE=0 LOCAL_REFINE_MAX_WALL_S=60 FORCE=1 \
#     bash scripts/submit_fit_stage_b_sharded.sh
#
#   # Production (shared-stim best Stage-A = s89 hybrid by default):
#   SEEDS="56 34 78 89 202" OUT_TAG=stageB_s89 \
#     bash scripts/submit_fit_stage_b_sharded.sh
#
# Env:
#   RETINAL_JSON  Stage-A final (default: openalyx s89 retinal_final)
#   WEIGHTS_JSON  WEIGHTS_REL (default: openalyx 0p4044)
#   HYBRID_JSON   output / resume path (default: models/stage_b_hybrid_*.json)
#   VARIANTS      default "regular:12|13 sensory:6|7|8|9"
#   LOCAL_REFINE_IDX  default prior → ∩ mask ∪ retinal when STAGE1_HOLD_RETINAL=1
#   STAGE1_HOLD_RETINAL  default 1 (DE holds Stage-A retinal; CMA unfreezes)
#   BPS_STAGE1        default 20 (DE); BPS_STAGE2 default 20 (CMA)
#   plus all submit_fit_joint_sharded.sh knobs (SEEDS PIPELINE OUT_TAG FORCE …)
#   P_OFFSET_ALWAYS_ON=1 / NO_ITI_PENALTY=1 — modeling-detail ablations
#   W_PP_LO / W_PP_HI / SET_W_PP / TIED_THRESHOLDS — tests 3–4
#     (prefer scripts/submit_fit_stage_b_model_ablations.sh)

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

PARTITION="${PARTITION:-pi_fiete}"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/sbatch_defaults.sh"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

ONE_MODELS="${ONE_MODELS:-$HOME/Downloads/ONE/openalyx.internationalbrainlab.org/models}"
# On ORCD, prefer the alyx ONE mirror if present.
if [[ -d /orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx ]]; then
  ORCD_MODELS="/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx/models"
  if [[ -d "$ORCD_MODELS" ]]; then
    ONE_MODELS="$ORCD_MODELS"
  fi
fi

WEIGHTS_JSON="${WEIGHTS_JSON:-$ONE_MODELS/weights_run_20251125_182058/weights_2stagelocalrefine_loss0p4044_20251125-195255.json}"
RETINAL_JSON="${RETINAL_JSON:-$ONE_MODELS/retinal_run_fr_retinal_masknone_s89/retinal_final_loss0p3712_20260811-171342.json}"
HYBRID_JSON="${HYBRID_JSON:-$ONE_MODELS/stage_b_hybrid_WEIGHTS_REL_retinal_s89.json}"

VARIANTS="${VARIANTS:-regular:12|13 sensory:6|7|8|9}"
SEEDS="${SEEDS:-56 34 78 89 202}"
PIPELINE="${PIPELINE:-de_cma_local}"
OUT_TAG="${OUT_TAG:-stageB_s89}"
LOCAL_REFINE_IDX="${LOCAL_REFINE_IDX:-prior}"
FORCE="${FORCE:-0}"
STAGE1_HOLD_RETINAL="${STAGE1_HOLD_RETINAL:-1}"
# Stage-1 DE at bps=10 hits S-bucket <10 → NaN → 1e12. Default both stages to 20.
BPS_STAGE1="${BPS_STAGE1:-20}"
BPS_STAGE2="${BPS_STAGE2:-20}"

module load miniforge 2>/dev/null || true
# Prefer ORCD ibl env if present; else whatever python3 is on PATH.
if [[ -f "$HOME/conda_envs/ibl/bin/python" ]]; then
  PY="$HOME/conda_envs/ibl/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "ERROR: no python3" >&2
  exit 1
fi

# Rebuild hybrid when both sources exist; otherwise reuse an existing HYBRID_JSON
# (ORCD often has the built hybrid without the full retinal_run tree).
if [[ -f "$WEIGHTS_JSON" && -f "$RETINAL_JSON" ]]; then
  echo "Building / refreshing Stage-B hybrid:"
  echo "  weights: $WEIGHTS_JSON"
  echo "  retinal: $RETINAL_JSON"
  echo "  out:     $HYBRID_JSON"
  "$PY" -u scripts/build_stage_b_hybrid.py \
    --weights-json "$WEIGHTS_JSON" \
    --retinal-json "$RETINAL_JSON" \
    --out "$HYBRID_JSON"
elif [[ -f "$HYBRID_JSON" ]]; then
  echo "WARN: sources missing; reusing existing HYBRID_JSON=$HYBRID_JSON"
  echo "  WEIGHTS_JSON exists=$([ -f "$WEIGHTS_JSON" ] && echo yes || echo no)"
  echo "  RETINAL_JSON exists=$([ -f "$RETINAL_JSON" ] && echo yes || echo no)"
else
  echo "ERROR: need WEIGHTS_JSON+RETINAL_JSON to build, or an existing HYBRID_JSON" >&2
  echo "  WEIGHTS_JSON=$WEIGHTS_JSON" >&2
  echo "  RETINAL_JSON=$RETINAL_JSON" >&2
  echo "  HYBRID_JSON=$HYBRID_JSON" >&2
  exit 1
fi

export RESUME_JSON="$HYBRID_JSON"
export VARIANTS SEEDS PIPELINE OUT_TAG LOCAL_REFINE_IDX FORCE REPO_DIR
export BPS_STAGE1 BPS_STAGE2 STAGE1_HOLD_RETINAL
# Forward optional fit knobs only if set (set -u safe).
for _k in DE1_MAXITER DE2_MAXITER DE_POPSIZE POPSIZE SOBOL_COUNT PATIENCE \
          BEAT_LOSS L_THRESHOLD BPS_STAGE1 BPS_STAGE2 STAGE1_HOLD_RETINAL \
          STAGE2_N_STIM_SEEDS STAGE2_STIM_AGGREGATE VAL_SEED \
          LOCAL_REFINE_METHOD LOCAL_REFINE_MAX_WALL_S BACKEND MEM CPUS TIME \
          P_OFFSET_ALWAYS_ON NO_ITI_PENALTY \
          W_PP_LO W_PP_HI SET_W_PP TIED_THRESHOLDS; do
  if [[ -n "${!_k:-}" ]]; then
    export "$_k"
  fi
done

echo "Submitting Stage-B joint (retinal held in DE, free in CMA/polish):"
echo "  VARIANTS=$VARIANTS"
echo "  SEEDS=$SEEDS  OUT_TAG=$OUT_TAG  PIPELINE=$PIPELINE"
echo "  BPS_STAGE1=$BPS_STAGE1  BPS_STAGE2=$BPS_STAGE2"
echo "  RESUME_JSON=$RESUME_JSON  LOCAL_REFINE_IDX=$LOCAL_REFINE_IDX"
echo "  STAGE1_HOLD_RETINAL=$STAGE1_HOLD_RETINAL"
bash scripts/submit_fit_joint_sharded.sh
