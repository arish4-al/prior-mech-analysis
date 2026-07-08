#!/usr/bin/env bash
# Example presence unsplit plots (Goal 4 gain-only thresholds), seed 123.
set -euo pipefail

SEED="${SEED:-123}"
N_SESSIONS="${N_SESSIONS:-40}"
NRAND="${NRAND:-100}"
N_JOBS="${N_JOBS:-8}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate iblenv

COMMON=(--presence-unsplit-plots --seed "$SEED" --n-sessions "$N_SESSIONS" --nrand "$NRAND" --n-jobs "$N_JOBS" --d-s-presence 0)

for GS in 1200 1800; do
  echo "=== presence unsplit plots g_s=$GS d_s=0 ==="
  python simulate_recovery.py "${COMMON[@]}" --g-s-presence "$GS"
done
