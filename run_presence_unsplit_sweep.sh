#!/usr/bin/env bash
# Goal 4: presence sweep with fitted I/M, stim-side unsplit.
# Default grid: g_s in [0, 2500], d_s in [0, fitted d_i].
set -euo pipefail

SEED="${SEED:-123}"
N_SESSIONS="${N_SESSIONS:-40}"
NRAND="${NRAND:-100}"
N_JOBS="${N_JOBS:-8}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate iblenv

python simulate_recovery.py \
  --presence-unsplit-sweep \
  --seed "$SEED" \
  --n-sessions "$N_SESSIONS" \
  --nrand "$NRAND" \
  --n-jobs "$N_JOBS" \
  --g-s-max 2500 \
  "$@"
