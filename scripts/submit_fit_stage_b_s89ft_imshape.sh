#!/bin/bash
# Warm local from full 80 mpre3 s89. Two freeze sets (journals/s89ft_imshape.md):
#   nowii  free {W_mm, W_mi, g_m, d_i, d_m}           (option 1)
#   wii    also free W_ii                             (option 2)
# Hold S / g_i / W_pp / W_is / W_pi / θ / retinal at s89.
# Extra fit-only loss: duringchoice I/M nSSE + stim M hinge after 40 ms.
#
# Aim: choice-window I/M + stim M overshoot ≥40 ms, without leaving the
# s89 S-success basin.
# Window / stratum: production unset (~80 ms), stim×choice.
# What the extra SSE can see: duringchoice I+M plus hinge nSSE on stim M
# for t>=40 ms. Rank later at extras=0, m_pre_weight=1, mean_c ‖Δ‖.
#
# Do not FORCE old full / full_mpre3_meancell dirs.
#
#   PARTITION=mit_preemptable FORCE=0 \
#     bash scripts/submit_fit_stage_b_s89ft_imshape.sh
#
#   ARMS=nowii FORCE=0 bash scripts/submit_fit_stage_b_s89ft_imshape.sh
#   ARMS=wii   FORCE=0 bash scripts/submit_fit_stage_b_s89ft_imshape.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

PARTITION="${PARTITION:-pi_fiete}"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/sbatch_defaults.sh"

ONE_MODELS="${ONE_MODELS:-$HOME/Downloads/ONE/openalyx.internationalbrainlab.org/models}"
if [[ -d /orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx ]]; then
  ORCD_MODELS="/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx/models"
  if [[ -d "$ORCD_MODELS" ]]; then
    ONE_MODELS="$ORCD_MODELS"
  fi
fi

S89_PREFIX="weights_run_fj_stageB_hold_s89_full_mpre3_meancell_full_masknone_s89"
S89_DIR=""
for root in "$ONE_MODELS" "$ONE_MODELS/new"; do
  if [[ -d "$root/$S89_PREFIX" ]]; then
    S89_DIR="$root/$S89_PREFIX"
    break
  fi
done
if [[ -z "$S89_DIR" ]]; then
  echo "ERROR: missing $S89_PREFIX under $ONE_MODELS" >&2
  exit 1
fi
if [[ -z "${RESUME_JSON:-}" ]]; then
  RESUME_JSON="$(ls -1t "$S89_DIR"/weights_final_*.json 2>/dev/null | head -1 || true)"
fi
if [[ -z "$RESUME_JSON" || ! -f "$RESUME_JSON" ]]; then
  echo "ERROR: no weights_final_*.json in $S89_DIR" >&2
  exit 1
fi

export SEEDS="${SEEDS:-89 7 12 45}"
export PIPELINE="${PIPELINE:-cma_only}"
export RESUME_JSON
export INCLUDE_STIM_PRIOR="${INCLUDE_STIM_PRIOR:-1}"
export STAGE1_HOLD_RETINAL="${STAGE1_HOLD_RETINAL:-0}"
export FREEZE_HOLD="${FREEZE_HOLD:-1}"
export M_PRE_WEIGHT="${M_PRE_WEIGHT:-1}"
export CHOICE_IM_EXTRA_WEIGHT="${CHOICE_IM_EXTRA_WEIGHT:-1}"
export M_STIM_OVERSHOOT_WEIGHT="${M_STIM_OVERSHOOT_WEIGHT:-1}"
export M_STIM_OVERSHOOT_FROM_MS="${M_STIM_OVERSHOOT_FROM_MS:-40}"
export BPS_STAGE1="${BPS_STAGE1:-20}"
export BPS_STAGE2="${BPS_STAGE2:-20}"
export BEAT_LOSS="${BEAT_LOSS:--1}"
export FORCE="${FORCE:-0}"
export REPO_DIR

ARMS="${ARMS:-nowii wii}"
read -r -a ARM_ARR <<< "$ARMS"
if [[ -n "${OUT_TAG:-}" && ${#ARM_ARR[@]} -gt 1 ]]; then
  echo "ERROR: OUT_TAG cannot be set when running multiple ARMS;" >&2
  echo "  use ARMS=<one arm> or the per-arm defaults" >&2
  exit 1
fi

_submit_one() {
  local tag="$1"
  export OUT_TAG="${OUT_TAG:-$tag}"
  echo "=== s89ft imshape  ARM=$ARM  OUT_TAG=$OUT_TAG  SEEDS=$SEEDS ==="
  echo "  Aim: choice I/M + stim M overshoot after 40 ms, from s89 full mpre3"
  echo "  Window: unset (~80 ms)  stratum: stim×choice  metric: mean_c ‖Δ‖"
  echo "  Freeze-hold=1  VARIANTS=$VARIANTS  LOCAL_REFINE_IDX=$LOCAL_REFINE_IDX"
  echo "  Extra: choice_im=$CHOICE_IM_EXTRA_WEIGHT  M_overshoot=$M_STIM_OVERSHOOT_WEIGHT from ${M_STIM_OVERSHOOT_FROM_MS} ms"
  echo "  RESUME_JSON=$RESUME_JSON  PIPELINE=$PIPELINE  FORCE=$FORCE"
  bash scripts/submit_fit_joint_sharded.sh
  unset OUT_TAG
}

for ARM in "${ARM_ARR[@]}"; do
  case "$ARM" in
    nowii)
      # Option 1: free W_mm(2), W_mi(5), g_m(7), d_i(8), d_m(9).
      export VARIANTS="full:0|1|3|4|6|10|11|12|13|14|15|16|17|18|19|20"
      export LOCAL_REFINE_IDX="2,5,7,8,9"
      TAG="${OUT_TAG_NOWII:-stageB_hold_s89_full_s89ft_imshape_meancell}"
      ;;
    wii)
      # Option 2: also free W_ii(0) so choice I can change hold time.
      export VARIANTS="full:1|3|4|6|10|11|12|13|14|15|16|17|18|19|20"
      export LOCAL_REFINE_IDX="0,2,5,7,8,9"
      TAG="${OUT_TAG_WII:-stageB_hold_s89_full_s89ft_imshape_wii_meancell}"
      ;;
    *)
      echo "ERROR: unknown ARM='$ARM'  (use nowii | wii)" >&2
      exit 1
      ;;
  esac
  _submit_one "$TAG"
done
