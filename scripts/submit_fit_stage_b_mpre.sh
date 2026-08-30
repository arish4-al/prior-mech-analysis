#!/bin/bash
# Regular Stage B with extra weight on pre-action M in the traj loss.
#
# Same protocol as journals/retinal_then_joint_fitting.md stageB_hold_s89:
#   hybrid WEIGHTS_REL ∪ retinal s89, --stage1-hold-retinal, bps1=bps2=20,
#   regular:12|13 (g_s/d_s frozen). Default 8 seeds (2026-08-13 fair ranking).
#
# Pre-action M nSSE is multiplied by M_PRE_WEIGHT (default 3). Post-start M,
# I (both windows), P, prior, and the ITI penalty stay at weight 1.
#
# Usage:
#   bash scripts/submit_fit_stage_b_mpre.sh
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
# Env: M_PRE_WEIGHT (default 3), plus all submit_fit_stage_b_sharded.sh knobs.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

PARTITION="${PARTITION:-pi_fiete}"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/sbatch_defaults.sh"

# Top-8 regular Stage B seeds by shared-stim fair L_w+L_S (2026-08-13).
SEEDS="${SEEDS:-7 12 34 45 89 101 303 333}"
VARIANTS="${VARIANTS:-regular:12|13}"
M_PRE_WEIGHT="${M_PRE_WEIGHT:-3}"
# Tag encodes the weight (3 → mpre3, 2.5 → mpre2p5).
_wtag="${M_PRE_WEIGHT}"
_wtag="${_wtag%.0}"
_wtag="${_wtag//./p}"
OUT_TAG="${OUT_TAG:-stageB_hold_s89_mpre${_wtag}}"

export REPO_DIR SEEDS VARIANTS M_PRE_WEIGHT OUT_TAG
export STAGE1_HOLD_RETINAL="${STAGE1_HOLD_RETINAL:-1}"
export BPS_STAGE1="${BPS_STAGE1:-20}"
export BPS_STAGE2="${BPS_STAGE2:-20}"
export PIPELINE="${PIPELINE:-de_cma_local}"
export LOCAL_REFINE_IDX="${LOCAL_REFINE_IDX:-prior}"
export P_OFFSET_ALWAYS_ON="${P_OFFSET_ALWAYS_ON:-0}"
export NO_ITI_PENALTY="${NO_ITI_PENALTY:-0}"
export TIED_THRESHOLDS="${TIED_THRESHOLDS:-0}"

echo "=== stage B regular  m_pre_weight=$M_PRE_WEIGHT  OUT_TAG=$OUT_TAG ==="
echo "    SEEDS=$SEEDS  VARIANTS=$VARIANTS"
bash scripts/submit_fit_stage_b_sharded.sh
