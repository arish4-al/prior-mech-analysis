#!/bin/bash
# Stage B modeling-detail ablations. Regular only (P→I/M; g_s/d_s frozen).
# Same protocol as journals/retinal_then_joint_fitting.md stageB_hold_s89:
#   hybrid WEIGHTS_REL ∪ retinal s89, --stage1-hold-retinal, bps1=bps2=20.
# Sensory is not a default for this journal.
#
# Default seeds: 8 best regular shared-stim fair L_w+L_S (2026-08-13).
#
# Usage:
#   # Tests 3–4 (8 seeds × 1 variant × 4 arms = 32 jobs):
#   bash scripts/submit_fit_stage_b_model_ablations.sh
#
#   # Tests 1–2 (already run 2026-08-27):
#   ABLATIONS="poffset noiti" bash scripts/submit_fit_stage_b_model_ablations.sh
#
#   # One arm:
#   ABLATIONS=wppsmall bash scripts/submit_fit_stage_b_model_ablations.sh
#   ABLATIONS=onethr bash scripts/submit_fit_stage_b_model_ablations.sh
#
#   # Smoke:
#   SEEDS=999 ABLATIONS=wppsmall OUT_TAG=stageB_ablate_wppsmall_smoke \
#     DE1_MAXITER=2 DE2_MAXITER=3 POPSIZE=8 SOBOL_COUNT=4 \
#     PATIENCE=0 LOCAL_REFINE_MAX_WALL_S=60 FORCE=1 TIME=1:00:00 \
#     bash scripts/submit_fit_stage_b_model_ablations.sh
#
# Env: ABLATIONS (poffset / noiti / wpplarge / wppopen / wppsmall / onethr),
#      plus all submit_fit_stage_b_sharded.sh knobs.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

PARTITION="${PARTITION:-pi_fiete}"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/sbatch_defaults.sh"

# Top-8 regular Stage B seeds by shared-stim fair L_w+L_S (2026-08-13):
#   101 1.001, 333 1.017, 34 1.023, 12 1.027,
#   303 1.045, 45 1.072, 7 1.076, 89 1.094.
SEEDS="${SEEDS:-7 12 34 45 89 101 303 333}"
# Tests 3–4 are the current default (1–2 already FIT_DONE).
ABLATIONS="${ABLATIONS:-wpplarge wppopen wppsmall onethr}"
VARIANTS_DEFAULT="${VARIANTS:-regular:12|13}"
VARIANTS="$VARIANTS_DEFAULT"

export REPO_DIR SEEDS VARIANTS
export STAGE1_HOLD_RETINAL="${STAGE1_HOLD_RETINAL:-1}"
export BPS_STAGE1="${BPS_STAGE1:-20}"
export BPS_STAGE2="${BPS_STAGE2:-20}"
export PIPELINE="${PIPELINE:-de_cma_local}"
export LOCAL_REFINE_IDX="${LOCAL_REFINE_IDX:-prior}"

# W_pp τ_Δ = 20 / (1−2W) ms. Open floor = W_ii box (0.20).
W_PP_OPEN_LO=0.20
W_PP_OPEN_HI=0.49999
W_PP_LARGE=0.499    # τ_Δ = 10 s
W_PP_SMALL=0.45     # τ_Δ = 200 ms

_reset_ablation_env() {
  unset W_PP_LO W_PP_HI SET_W_PP
  export P_OFFSET_ALWAYS_ON=0
  export NO_ITI_PENALTY=0
  export TIED_THRESHOLDS=0
  export VARIANTS="$VARIANTS_DEFAULT"
}

read -r -a ABL_ARR <<< "$ABLATIONS"
if [[ -n "${OUT_TAG:-}" && ${#ABL_ARR[@]} -gt 1 ]]; then
  echo "ERROR: OUT_TAG cannot be set when running multiple ABLATIONS;" >&2
  echo "  use ABLATIONS=<one arm> or OUT_TAG_<ARM>" >&2
  exit 1
fi
for ABLATION in "${ABL_ARR[@]}"; do
  _reset_ablation_env
  case "$ABLATION" in
    poffset)
      export P_OFFSET_ALWAYS_ON=1
      TAG="${OUT_TAG_POFFSET:-stageB_hold_s89_poffset}"
      ;;
    noiti)
      export NO_ITI_PENALTY=1
      TAG="${OUT_TAG_NOITI:-stageB_hold_s89_noiti}"
      ;;
    wpplarge)
      # Test 3: large init, keep current box [0.496, 0.49999].
      export SET_W_PP="$W_PP_LARGE"
      TAG="${OUT_TAG_WPPLARGE:-stageB_hold_s89_wpplarge}"
      ;;
    wppopen)
      # Test 3: large init, floor loosened to 0.20.
      export SET_W_PP="$W_PP_LARGE"
      export W_PP_LO="$W_PP_OPEN_LO"
      export W_PP_HI="$W_PP_OPEN_HI"
      TAG="${OUT_TAG_WPPOPEN:-stageB_hold_s89_wppopen}"
      ;;
    wppsmall)
      # Test 3: small init (200 ms), floor loosened to 0.20.
      export SET_W_PP="$W_PP_SMALL"
      export W_PP_LO="$W_PP_OPEN_LO"
      export W_PP_HI="$W_PP_OPEN_HI"
      TAG="${OUT_TAG_WPPSMALL:-stageB_hold_s89_wppsmall}"
      ;;
    onethr)
      # Test 4: theta_c = theta_d (freeze idx 11 in the Python driver).
      export TIED_THRESHOLDS=1
      export VARIANTS="regular:11|12|13"
      TAG="${OUT_TAG_ONETHR:-stageB_hold_s89_onethr}"
      ;;
    *)
      echo "ERROR: unknown ABLATION='$ABLATION'" >&2
      echo "  use poffset | noiti | wpplarge | wppopen | wppsmall | onethr" >&2
      exit 1
      ;;
  esac
  # Per-ablation OUT_TAG; caller OUT_TAG overrides both if set.
  export OUT_TAG="${OUT_TAG:-$TAG}"
  echo "=== ablation=$ABLATION  OUT_TAG=$OUT_TAG  SEEDS=$SEEDS ==="
  echo "    P_OFFSET_ALWAYS_ON=$P_OFFSET_ALWAYS_ON  NO_ITI_PENALTY=$NO_ITI_PENALTY"
  echo "    SET_W_PP=${SET_W_PP:-} W_PP_LO=${W_PP_LO:-} W_PP_HI=${W_PP_HI:-}"
  echo "    TIED_THRESHOLDS=$TIED_THRESHOLDS"
  bash scripts/submit_fit_stage_b_sharded.sh
  unset OUT_TAG
done
