#!/bin/bash
# Stage B modeling-detail ablations. Regular only (P→I/M; g_s/d_s frozen).
# Same protocol as journals/retinal_then_joint_fitting.md stageB_hold_s89:
#   hybrid WEIGHTS_REL ∪ retinal s89, --stage1-hold-retinal, bps1=bps2=20.
# Sensory is not a default for this journal.
#
# Default seeds: 8 best regular shared-stim fair L_w+L_S (2026-08-13).
#
# Usage:
#   # Tests 1–2 (8 seeds × 1 variant × 2 = 16 jobs):
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

# Top-8 regular Stage B seeds by shared-stim fair L_w+L_S (2026-08-13):
#   101 1.001, 333 1.017, 34 1.023, 12 1.027,
#   303 1.045, 45 1.072, 7 1.076, 89 1.094.
SEEDS="${SEEDS:-7 12 34 45 89 101 303 333}"
ABLATIONS="${ABLATIONS:-poffset noiti}"
VARIANTS="${VARIANTS:-regular:12|13}"

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
