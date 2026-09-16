#!/bin/bash
# Stage B with extra weight on pre-action M in the traj loss.
#
# Same protocol as journals/retinal_then_joint_fitting.md stageB_hold_s89:
#   hybrid WEIGHTS_REL ∪ retinal s89, --stage1-hold-retinal, bps1=bps2=20.
# Pre-action M nSSE is multiplied by M_PRE_WEIGHT (default 3). Post-start M,
# I (both windows), P, prior, and the ITI penalty stay at weight 1.
# I/M prior metric is whatever the checkout has (mean_c‖Δ‖ since 09-08f).
#
# Usage:
#   # Test 5 (80 ms, regular mask) — already FIT_DONE:
#   bash scripts/submit_fit_stage_b_mpre.sh
#
#   # 150 ms stim×choice + m_pre=3, regular mask (g_s/d_s frozen):
#   ARMS=im150 FORCE=0 bash scripts/submit_fit_stage_b_mpre.sh
#     → OUT_TAG stageB_hold_s89_mpre3_im150_meancell
#
#   # 80 ms (window unset) + m_pre=3, full (g_s/d_s free) + unsplit-80 S:
#   ARMS=full FORCE=0 bash scripts/submit_fit_stage_b_mpre.sh
#     → OUT_TAG stageB_hold_s89_full_mpre3_meancell
#
#   # 150 ms stim×choice + m_pre=3, full + unsplit-80 S:
#   ARMS=full_im150 FORCE=0 bash scripts/submit_fit_stage_b_mpre.sh
#     → OUT_TAG stageB_hold_s89_full_mpre3_im150_meancell
#
#   ARMS="full im150 full_im150" FORCE=0 bash scripts/submit_fit_stage_b_mpre.sh
#
#   M_PRE_WEIGHT=2 OUT_TAG=stageB_hold_s89_mpre2 \
#     bash scripts/submit_fit_stage_b_mpre.sh
#
#   # Smoke:
#   SEEDS=999 M_PRE_WEIGHT=3 OUT_TAG=stageB_hold_s89_mpre3_smoke \
#     DE1_MAXITER=2 DE2_MAXITER=3 POPSIZE=8 SOBOL_COUNT=4 \
#     PATIENCE=0 LOCAL_REFINE_MAX_WALL_S=60 FORCE=1 TIME=1:00:00 \
#     bash scripts/submit_fit_stage_b_mpre.sh
#
# Env: ARMS (full / im150 / full_im150), M_PRE_WEIGHT (default 3), plus
# all submit_fit_stage_b_sharded.sh knobs. New I/M tags include _meancell
# (mean_c‖Δ‖). Do not FORCE existing mpre3 / full / full_im150_meancell dirs.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

PARTITION="${PARTITION:-pi_fiete}"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/sbatch_defaults.sh"

# Top-8 regular Stage B seeds by shared-stim fair L_w+L_S (2026-08-13).
SEEDS="${SEEDS:-7 12 34 45 89 101 303 333}"
M_PRE_WEIGHT="${M_PRE_WEIGHT:-3}"
_wtag="${M_PRE_WEIGHT}"
_wtag="${_wtag%.0}"
_wtag="${_wtag//./p}"

export REPO_DIR SEEDS M_PRE_WEIGHT
export STAGE1_HOLD_RETINAL="${STAGE1_HOLD_RETINAL:-1}"
export BPS_STAGE1="${BPS_STAGE1:-20}"
export BPS_STAGE2="${BPS_STAGE2:-20}"
export PIPELINE="${PIPELINE:-de_cma_local}"
export LOCAL_REFINE_IDX="${LOCAL_REFINE_IDX:-prior}"
export P_OFFSET_ALWAYS_ON="${P_OFFSET_ALWAYS_ON:-0}"
export NO_ITI_PENALTY="${NO_ITI_PENALTY:-0}"
export TIED_THRESHOLDS="${TIED_THRESHOLDS:-0}"
export FORCE="${FORCE:-0}"

_reset_arm_env() {
  unset PRIOR_WINDOW_MS PRIOR_STRATUM INCLUDE_STIM_PRIOR
  export VARIANTS="regular:12|13"
}

_submit_one() {
  local tag="$1"
  export OUT_TAG="${OUT_TAG:-$tag}"
  echo "=== m_pre_weight=$M_PRE_WEIGHT  OUT_TAG=$OUT_TAG  SEEDS=$SEEDS ==="
  echo "    VARIANTS=$VARIANTS  FORCE=$FORCE"
  echo "    INCLUDE_STIM_PRIOR=${INCLUDE_STIM_PRIOR:-} PRIOR_WINDOW_MS=${PRIOR_WINDOW_MS:-} PRIOR_STRATUM=${PRIOR_STRATUM:-}"
  bash scripts/submit_fit_stage_b_sharded.sh
  unset OUT_TAG
}

if [[ -z "${ARMS:-}" ]]; then
  # Legacy test 5: production window unset (~80 ms), regular mask.
  export VARIANTS="${VARIANTS:-regular:12|13}"
  _submit_one "stageB_hold_s89_mpre${_wtag}"
  exit 0
fi

read -r -a ARM_ARR <<< "$ARMS"
if [[ -n "${OUT_TAG:-}" && ${#ARM_ARR[@]} -gt 1 ]]; then
  echo "ERROR: OUT_TAG cannot be set when running multiple ARMS;" >&2
  echo "  use ARMS=<one arm> or the per-arm defaults" >&2
  exit 1
fi
for ARM in "${ARM_ARR[@]}"; do
  _reset_arm_env
  case "$ARM" in
    full)
      # Production I/M window unset (~80 ms); full + unsplit-80 S.
      export VARIANTS="full:"
      export INCLUDE_STIM_PRIOR=1
      TAG="${OUT_TAG_FULL:-stageB_hold_s89_full_mpre${_wtag}_meancell}"
      ;;
    im150)
      # Regular mask; 150 ms stim×choice I/M; mean_c‖Δ‖ (current code).
      export PRIOR_WINDOW_MS=150
      TAG="${OUT_TAG_IM150:-stageB_hold_s89_mpre${_wtag}_im150_meancell}"
      ;;
    full_im150)
      # full + unsplit-80 S; 150 ms stim×choice I/M; mean_c‖Δ‖.
      export VARIANTS="full:"
      export INCLUDE_STIM_PRIOR=1
      export PRIOR_WINDOW_MS=150
      TAG="${OUT_TAG_FULL_IM150:-stageB_hold_s89_full_mpre${_wtag}_im150_meancell}"
      ;;
    *)
      echo "ERROR: unknown ARM='$ARM'" >&2
      echo "  use full | im150 | full_im150  (or omit ARMS for 80 ms regular mpre)" >&2
      exit 1
      ;;
  esac
  _submit_one "$TAG"
done
