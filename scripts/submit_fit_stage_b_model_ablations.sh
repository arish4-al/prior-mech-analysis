#!/bin/bash
# Stage B modeling-detail ablations (tests 1–2), same protocol/seeds as
# journals/retinal_then_joint_fitting.md 2026-08-12g + 2026-08-13:
#   hybrid WEIGHTS_REL ∪ retinal s89, --stage1-hold-retinal,
#   regular 12|13 + sensory 6|7|8|9, bps1=bps2=20.
#
# Compare to OUT_TAG=stageB_hold_s89 (15 seeds below).
#
# Usage (print-only for the user to paste on ORCD; agents must not sbatch):
#   # Both ablations (30 jobs × 2):
#   bash scripts/submit_fit_stage_b_model_ablations.sh
#
#   # Test 1 only (P offset always on):
#   ABLATIONS=poffset bash scripts/submit_fit_stage_b_model_ablations.sh
#
#   # Test 2 only (no I/M ITI penalty):
#   ABLATIONS=noiti bash scripts/submit_fit_stage_b_model_ablations.sh
#
#   # Smoke:
#   SEEDS=999 ABLATIONS=poffset OUT_TAG=stageB_ablate_poffset_smoke \
#     DE1_MAXITER=2 DE2_MAXITER=3 POPSIZE=8 SOBOL_COUNT=4 \
#     PATIENCE=0 LOCAL_REFINE_MAX_WALL_S=60 FORCE=1 TIME=1:00:00 \
#     bash scripts/submit_fit_stage_b_model_ablations.sh
#
# Env: ABLATIONS (poffset / noiti), plus all submit_fit_stage_b_sharded.sh knobs.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

# Combined Stage B production seeds (batch-1 2026-08-12g + batch-2 2026-08-13).
SEEDS="${SEEDS:-7 12 23 34 42 45 56 67 78 89 101 111 202 303 333}"
ABLATIONS="${ABLATIONS:-poffset noiti}"
VARIANTS="${VARIANTS:-regular:12|13 sensory:6|7|8|9}"

export REPO_DIR SEEDS VARIANTS
export STAGE1_HOLD_RETINAL="${STAGE1_HOLD_RETINAL:-1}"
export BPS_STAGE1="${BPS_STAGE1:-20}"
export BPS_STAGE2="${BPS_STAGE2:-20}"
export PIPELINE="${PIPELINE:-de_cma_local}"
export LOCAL_REFINE_IDX="${LOCAL_REFINE_IDX:-prior}"

read -r -a ABL_ARR <<< "$ABLATIONS"
if [[ -n "${OUT_TAG:-}" && ${#ABL_ARR[@]} -gt 1 ]]; then
  echo "ERROR: OUT_TAG cannot be set when running multiple ABLATIONS;" >&2
  echo "  use ABLATIONS=poffset or OUT_TAG_POFFSET / OUT_TAG_NOITI" >&2
  exit 1
fi
for ABLATION in "${ABL_ARR[@]}"; do
  case "$ABLATION" in
    poffset)
      export P_OFFSET_ALWAYS_ON=1
      export NO_ITI_PENALTY=0
      TAG="${OUT_TAG_POFFSET:-stageB_hold_s89_poffset}"
      ;;
    noiti)
      export P_OFFSET_ALWAYS_ON=0
      export NO_ITI_PENALTY=1
      TAG="${OUT_TAG_NOITI:-stageB_hold_s89_noiti}"
      ;;
    *)
      echo "ERROR: unknown ABLATION='$ABLATION' (use poffset or noiti)" >&2
      exit 1
      ;;
  esac
  # Per-ablation OUT_TAG; caller OUT_TAG overrides both if set.
  export OUT_TAG="${OUT_TAG:-$TAG}"
  echo "=== ablation=$ABLATION  OUT_TAG=$OUT_TAG  SEEDS=$SEEDS ==="
  echo "    P_OFFSET_ALWAYS_ON=$P_OFFSET_ALWAYS_ON  NO_ITI_PENALTY=$NO_ITI_PENALTY"
  bash scripts/submit_fit_stage_b_sharded.sh
  unset OUT_TAG
done
