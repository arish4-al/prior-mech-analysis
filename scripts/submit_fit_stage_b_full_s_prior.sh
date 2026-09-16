#!/bin/bash
# Stage B ``full`` (all prior-mod gains free) + S unsplit-80 prior nSSE.
#
# Same hybrid / hold-retinal / bps=20 protocol as
# ``submit_fit_stage_b_sharded.sh``. I/M window and stratum flags match
# ``submit_fit_stage_b_model_ablations.sh`` (im150 / im150stim / stimonly).
# Model S stays ``stratum_s=stim`` (2-split sidecar) regardless of PRIOR_STRATUM.
#
# Default: replace the baseline full dirs, then submit the three new arms.
#
#   PARTITION=mit_preemptable FORCE=1 \
#     bash scripts/submit_fit_stage_b_full_s_prior.sh
#
#   ARMS=full FORCE=0 bash scripts/submit_fit_stage_b_full_s_prior.sh
#     → OUT_TAG stageB_hold_s89_full_meancell
#   ARMS="im150 im150stim stimonly" bash scripts/submit_fit_stage_b_full_s_prior.sh
#
#   # 150 ms stim×choice I/M + unsplit-80 S, mean_c‖Δ‖ (do not overwrite 09-08c):
#   PARTITION=mit_preemptable ARMS=im150 FORCE=0 \
#     bash scripts/submit_fit_stage_b_full_s_prior.sh
#     → OUT_TAG stageB_hold_s89_full_im150_meancell
#       (09-08c tag stageB_hold_s89_full_im150 used ‖mean_c Δ‖)
#
# Env: ARMS (full / im150 / im150stim / stimonly), plus all
# submit_fit_stage_b_sharded.sh knobs (SEEDS PARTITION FORCE …).

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

PARTITION="${PARTITION:-pi_fiete}"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/sbatch_defaults.sh"

SEEDS="${SEEDS:-7 12 34 45 89 101 303 333}"
ARMS="${ARMS:-full im150 im150stim stimonly}"
VARIANTS="full:"
export INCLUDE_STIM_PRIOR="${INCLUDE_STIM_PRIOR:-1}"
export VARIANTS SEEDS REPO_DIR
export STAGE1_HOLD_RETINAL="${STAGE1_HOLD_RETINAL:-1}"
export BPS_STAGE1="${BPS_STAGE1:-20}"
export BPS_STAGE2="${BPS_STAGE2:-20}"
export PIPELINE="${PIPELINE:-de_cma_local}"
export LOCAL_REFINE_IDX="${LOCAL_REFINE_IDX:-prior}"
# Replace existing full dirs by default; new OUT_TAGs do not collide.
export FORCE="${FORCE:-1}"

_reset_arm_env() {
  unset PRIOR_WINDOW_MS PRIOR_STRATUM
}

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
      # 80 ms stim×choice + unsplit-80 S. Default tag is mean_c‖Δ‖.
      # The 1e-12 campaign used stageB_hold_s89_full (‖mean_c Δ‖) — do
      # not overwrite that dir.
      TAG="${OUT_TAG_FULL:-stageB_hold_s89_full_meancell}"
      ;;
    im150)
      # 150 ms stim×choice I/M + unsplit-80 S. Default tag is mean_c‖Δ‖
      # (2026-09-08f). The 09-08c run used ‖mean_c Δ‖ under
      # stageB_hold_s89_full_im150 — do not overwrite that dir.
      export PRIOR_WINDOW_MS=150
      TAG="${OUT_TAG_IM150:-stageB_hold_s89_full_im150_meancell}"
      ;;
    im150stim)
      export PRIOR_WINDOW_MS=150
      export PRIOR_STRATUM=stim
      TAG="${OUT_TAG_IM150STIM:-stageB_hold_s89_full_im150stim}"
      ;;
    stimonly)
      export PRIOR_STRATUM=stim
      TAG="${OUT_TAG_STIMONLY:-stageB_hold_s89_full_stimonly}"
      ;;
    *)
      echo "ERROR: unknown ARM='$ARM'" >&2
      echo "  use full | im150 | im150stim | stimonly" >&2
      exit 1
      ;;
  esac
  export OUT_TAG="${OUT_TAG:-$TAG}"
  echo "=== full+S arm=$ARM  OUT_TAG=$OUT_TAG  SEEDS=$SEEDS ==="
  echo "    INCLUDE_STIM_PRIOR=$INCLUDE_STIM_PRIOR  FORCE=$FORCE"
  echo "    PRIOR_WINDOW_MS=${PRIOR_WINDOW_MS:-} PRIOR_STRATUM=${PRIOR_STRATUM:-}"
  bash scripts/submit_fit_stage_b_sharded.sh
  unset OUT_TAG
done
