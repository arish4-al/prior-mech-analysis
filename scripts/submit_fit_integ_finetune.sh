#!/bin/bash
# Warm fine-tune of the integrator-rule Stage B arms (journals/stage_b_fit_variants.md,
# 2026-10-28). One CMA-only restart set per source seed, each on its own arm's targets.
#
# Aim: lower the prior curves (S, C, M) from the best opt1 / opt2 fits with g_m = d_m = 0.
# Sources (stageB_hold_newS34_full_mpre3_meancell_integ_<opt>, newest weights_final):
#   opt1_rawslope s45, s7   opt2_shuffleslope s12, s303
# Free: W_mm(2) W_mi(5) g_i(6) d_i(8) g_s(12) d_s(13).
# Zero + frozen: g_m(7) d_m(9) (--zero-g-m-d-m). g_i floor 0.1 (code bound).
# Freeze-hold at the source θ: W_ii W_pp W_is W_pi θ_c θ_d, retinal (14–20).
# Window unset (~80 ms), stim×choice stratum, unsplit-80 S prior on, m_pre_weight 3
# (as the arms were fit). Loss sees I/M traj, I/M prior, S prior 0–80 ms, S RMS.
# No extra loss terms. Rank later at m_pre_weight 1 on the shared-stim eval.
#
#   STARTS="opt1_rawslope:45 opt1_rawslope:7 opt2_shuffleslope:12 opt2_shuffleslope:303" \
#     bash scripts/submit_fit_integ_finetune.sh
#
# Do not submit from the laptop.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

export PARTITION="${PARTITION:-pi_fiete}"
ONE_MODELS="${ONE_MODELS:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx/models}"
STARTS="${STARTS:-opt1_rawslope:45 opt1_rawslope:7 opt2_shuffleslope:12 opt2_shuffleslope:303}"

export SEEDS="${SEEDS:-89 7 12 45}"
export PIPELINE=cma_only
export INCLUDE_STIM_PRIOR=1
export STAGE1_HOLD_RETINAL=0
export FREEZE_HOLD=1
export ZERO_G_M_D_M=1
export M_PRE_WEIGHT="${M_PRE_WEIGHT:-3}"
export CHOICE_IM_EXTRA_WEIGHT=0 M_STIM_OVERSHOOT_WEIGHT=0 CHOICE_I_LATE_WEIGHT=0
export BPS_STAGE1="${BPS_STAGE1:-20}"
export BPS_STAGE2="${BPS_STAGE2:-20}"
export BEAT_LOSS="${BEAT_LOSS:--1}"
export FORCE="${FORCE:-0}"
export VARIANTS="full:0|1|3|4|7|9|10|11|14|15|16|17|18|19|20"
export LOCAL_REFINE_IDX="2,5,6,8,12,13"
export REPO_DIR

for start in $STARTS; do
  opt="${start%%:*}"
  src="${start##*:}"
  [[ -f "fit_targets_integ/$opt/mean_data_results.npy" ]] || {
    echo "ERROR: fit_targets_integ/$opt targets missing" >&2; exit 1; }
  run="$ONE_MODELS/weights_run_fj_stageB_hold_newS34_full_mpre3_meancell_integ_${opt}_full_masknone_s${src}"
  RESUME_JSON="$(ls -1t "$run"/weights_final_*.json 2>/dev/null | head -1 || true)"
  [[ -n "$RESUME_JSON" ]] || { echo "ERROR: no weights_final in $run" >&2; exit 1; }
  export RESUME_JSON
  export FIT_TARGETS_DIR="fit_targets_integ/$opt"
  export OUT_TAG="stageB_hold_newS34_integ_${opt}_ft${src}_gm0"
  echo "=== fine-tune $opt s$src  OUT_TAG=$OUT_TAG  CMA seeds=$SEEDS ==="
  echo "  free W_mm W_mi g_i d_i g_s d_s; g_m=d_m=0; rest held at source"
  echo "  window unset (~80 ms), stim×choice, S prior on, m_pre=$M_PRE_WEIGHT"
  echo "  RESUME_JSON=$RESUME_JSON  FIT_TARGETS_DIR=$FIT_TARGETS_DIR  PARTITION=$PARTITION"
  bash scripts/submit_fit_joint_sharded.sh
done
