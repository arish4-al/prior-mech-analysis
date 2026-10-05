#!/bin/bash
# Punch d_i from wii s7. Freeze-hold S / g_i / M-side / W_ii / retinal.
# Init d_i=5 (modest regular-scale). Extra: late during-choice I undershoot
# hinge (last 40 ms). No pooled choice I+M extra, no stim-M overshoot.
#
# Aim: keep choice-window I prior from collapsing in the last ~20–40 ms
# before move onset (commit constraint), without leaving the wii-s7 S/M
# basin.
# Window / stratum: production unset (~80 ms), stim×choice.
# What the extra SSE can see: undershoot hinge on duringchoice I for
# t>=-40 ms. Rank later at extras=0, m_pre_weight=1, mean_c ‖Δ‖.
#
# Do not FORCE old s89ft / full_mpre3 dirs.
#
#   PARTITION=mit_preemptable FORCE=0 \
#     bash scripts/submit_fit_stage_b_s89ft_di_choicei.sh

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

WII7_PREFIX="weights_run_fj_stageB_hold_s89_full_s89ft_imshape_wii_meancell_full_mask1-3-4-6-10-11-12-13-14-15-16-17-18-19-20_s7"
WII7_DIR=""
for root in "$ONE_MODELS" "$ONE_MODELS/new"; do
  if [[ -d "$root/$WII7_PREFIX" ]]; then
    WII7_DIR="$root/$WII7_PREFIX"
    break
  fi
done
if [[ -z "$WII7_DIR" ]]; then
  echo "ERROR: missing $WII7_PREFIX under $ONE_MODELS" >&2
  exit 1
fi
if [[ -z "${RESUME_JSON:-}" ]]; then
  RESUME_JSON="$(ls -1t "$WII7_DIR"/weights_final_*.json 2>/dev/null | head -1 || true)"
fi
if [[ -z "$RESUME_JSON" || ! -f "$RESUME_JSON" ]]; then
  echo "ERROR: no weights_final_*.json in $WII7_DIR" >&2
  exit 1
fi

export SEEDS="${SEEDS:-89 7 12 45}"
export PIPELINE="${PIPELINE:-cma_only}"
export RESUME_JSON
export INCLUDE_STIM_PRIOR="${INCLUDE_STIM_PRIOR:-1}"
export STAGE1_HOLD_RETINAL="${STAGE1_HOLD_RETINAL:-0}"
export FREEZE_HOLD="${FREEZE_HOLD:-1}"
export M_PRE_WEIGHT="${M_PRE_WEIGHT:-1}"
export SET_D_I="${SET_D_I:-5}"
export CHOICE_I_LATE_WEIGHT="${CHOICE_I_LATE_WEIGHT:-1}"
export CHOICE_I_LATE_MS="${CHOICE_I_LATE_MS:-40}"
export CHOICE_IM_EXTRA_WEIGHT="${CHOICE_IM_EXTRA_WEIGHT:-0}"
export M_STIM_OVERSHOOT_WEIGHT="${M_STIM_OVERSHOOT_WEIGHT:-0}"
export BPS_STAGE1="${BPS_STAGE1:-20}"
export BPS_STAGE2="${BPS_STAGE2:-20}"
export BEAT_LOSS="${BEAT_LOSS:--1}"
export FORCE="${FORCE:-0}"
export REPO_DIR
# Free d_i(8) only. Hold W / g / d_m / θ / S / retinal at wii s7.
export VARIANTS="${VARIANTS:-full:0|1|2|3|4|5|6|7|9|10|11|12|13|14|15|16|17|18|19|20}"
export LOCAL_REFINE_IDX="${LOCAL_REFINE_IDX:-8}"
export OUT_TAG="${OUT_TAG:-stageB_hold_s89_full_s89ft_di_choicei_meancell}"

echo "=== s89ft d_i choice-I  OUT_TAG=$OUT_TAG  SEEDS=$SEEDS ==="
echo "  Aim: late during-choice I floor via d_i, from wii s7"
echo "  Window: unset (~80 ms)  stratum: stim×choice  metric: mean_c ‖Δ‖"
echo "  Freeze-hold=1  free d_i only  VARIANTS=$VARIANTS  LOCAL_REFINE_IDX=$LOCAL_REFINE_IDX"
echo "  Init d_i=$SET_D_I  extra: choice_I_late=$CHOICE_I_LATE_WEIGHT last ${CHOICE_I_LATE_MS} ms"
echo "  choice_im=$CHOICE_IM_EXTRA_WEIGHT  M_overshoot=$M_STIM_OVERSHOOT_WEIGHT"
echo "  RESUME_JSON=$RESUME_JSON  PIPELINE=$PIPELINE  FORCE=$FORCE"
echo "  What the extra can see: undershoot hinge on duringchoice I, t>=-${CHOICE_I_LATE_MS} ms"
bash scripts/submit_fit_joint_sharded.sh
