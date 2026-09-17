#!/bin/bash
# Stage B split I/M windows: 150 ms post-stim, 80 ms pre-move.
#
# Same hybrid / hold-retinal / bps=20 / mean_c‖Δ‖ protocol as
# submit_fit_stage_b_sharded.sh. Traj + prior use the same per-alignment
# window (unify). S sidecar stays unsplit-80. Stratum unset = stim×choice.
# g_i floor is whatever the checkout has (1e-12). Do not FORCE old dirs.
#
# Motivation: post-stim I/M needs 150 ms to get past the stim-onset
# auditory transient; pre-move I/M is most informative near commit, and
# earlier bins are noisy from trial-length alignment. Explicit 80 ms
# choice is T=40 extract+score, not production unset (T=72 extract, last
# 40 scored).
#
# Usage:
#   PARTITION=mit_preemptable FORCE=0 \
#     bash scripts/submit_fit_stage_b_splitwin.sh
#     → four tags:
#        stageB_hold_s89_stim150_choice80_meancell
#        stageB_hold_s89_mpre3_stim150_choice80_meancell
#        stageB_hold_s89_full_stim150_choice80_meancell
#        stageB_hold_s89_full_mpre3_stim150_choice80_meancell
#
#   ARMS=regular FORCE=0 bash scripts/submit_fit_stage_b_splitwin.sh
#   ARMS="regular_mpre3 full" FORCE=0 bash scripts/submit_fit_stage_b_splitwin.sh
#
# Env: ARMS (regular / regular_mpre3 / full / full_mpre3), plus all
# submit_fit_stage_b_sharded.sh knobs. New tags include _meancell.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

PARTITION="${PARTITION:-pi_fiete}"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/sbatch_defaults.sh"

SEEDS="${SEEDS:-7 12 34 45 89 101 303 333}"
ARMS="${ARMS:-regular regular_mpre3 full full_mpre3}"
export REPO_DIR SEEDS
export STAGE1_HOLD_RETINAL="${STAGE1_HOLD_RETINAL:-1}"
export BPS_STAGE1="${BPS_STAGE1:-20}"
export BPS_STAGE2="${BPS_STAGE2:-20}"
export PIPELINE="${PIPELINE:-de_cma_local}"
export LOCAL_REFINE_IDX="${LOCAL_REFINE_IDX:-prior}"
export P_OFFSET_ALWAYS_ON="${P_OFFSET_ALWAYS_ON:-0}"
export NO_ITI_PENALTY="${NO_ITI_PENALTY:-0}"
export TIED_THRESHOLDS="${TIED_THRESHOLDS:-0}"
export FORCE="${FORCE:-0}"
export IM_WINDOW_STIM_MS="${IM_WINDOW_STIM_MS:-150}"
export IM_WINDOW_CHOICE_MS="${IM_WINDOW_CHOICE_MS:-80}"

_reset_arm_env() {
  unset PRIOR_WINDOW_MS PRIOR_STRATUM INCLUDE_STIM_PRIOR M_PRE_WEIGHT
  export VARIANTS="regular:12|13"
}

_submit_one() {
  local tag="$1"
  export OUT_TAG="${OUT_TAG:-$tag}"
  echo "=== splitwin OUT_TAG=$OUT_TAG  SEEDS=$SEEDS  FORCE=$FORCE ==="
  echo "    VARIANTS=$VARIANTS  M_PRE_WEIGHT=${M_PRE_WEIGHT:-1}"
  echo "    INCLUDE_STIM_PRIOR=${INCLUDE_STIM_PRIOR:-}"
  echo "    IM_WINDOW_STIM_MS=$IM_WINDOW_STIM_MS IM_WINDOW_CHOICE_MS=$IM_WINDOW_CHOICE_MS"
  echo "    PRIOR_WINDOW_MS=${PRIOR_WINDOW_MS:-} PRIOR_STRATUM=${PRIOR_STRATUM:-}"
  bash scripts/submit_fit_stage_b_sharded.sh
  unset OUT_TAG
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
    regular)
      TAG="${OUT_TAG_REGULAR:-stageB_hold_s89_stim150_choice80_meancell}"
      ;;
    regular_mpre3)
      export M_PRE_WEIGHT=3
      TAG="${OUT_TAG_REGULAR_MPRE3:-stageB_hold_s89_mpre3_stim150_choice80_meancell}"
      ;;
    full)
      export VARIANTS="full:"
      export INCLUDE_STIM_PRIOR=1
      TAG="${OUT_TAG_FULL:-stageB_hold_s89_full_stim150_choice80_meancell}"
      ;;
    full_mpre3)
      export VARIANTS="full:"
      export INCLUDE_STIM_PRIOR=1
      export M_PRE_WEIGHT=3
      TAG="${OUT_TAG_FULL_MPRE3:-stageB_hold_s89_full_mpre3_stim150_choice80_meancell}"
      ;;
    *)
      echo "ERROR: unknown ARM='$ARM'" >&2
      echo "  use regular | regular_mpre3 | full | full_mpre3" >&2
      exit 1
      ;;
  esac
  _submit_one "$TAG"
done
