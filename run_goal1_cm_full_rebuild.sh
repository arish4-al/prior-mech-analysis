#!/usr/bin/env bash
# Rebuild Goal-1.3 cm_full for phase4 / s1800 / s2025 with decorrelation fixes.
# Session cache HIT expected for all (same mp+seed as matrix).
set -u
PY=/Users/ariliu/opt/anaconda3/envs/iblenv/bin/python
SCRIPT="$(cd "$(dirname "$0")" && pwd)/simulate_recovery.py"
COMMON="--seed 123 --n-sessions 40 --nrand 100 --n-jobs 8 --full-analysis"
LOGDIR="/Users/ariliu/Downloads/ONE/openalyx.internationalbrainlab.org/manifold_sim/goal1/_logs"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/cm_full_rebuild_status.log"
echo "=== cm_full rebuild started $(date) ===" | tee "$SUMMARY"

run() {
  local label="$1"; shift
  local log="$LOGDIR/${label}.log"
  echo "[$(date +%H:%M:%S)] START $label" | tee -a "$SUMMARY"
  if $PY "$SCRIPT" $COMMON "$@" >"$log" 2>&1; then
    echo "[$(date +%H:%M:%S)] OK    $label" | tee -a "$SUMMARY"
  else
    echo "[$(date +%H:%M:%S)] FAIL  $label (see $log)" | tee -a "$SUMMARY"
    exit 1
  fi
}

run phase4_cm_full --run-experiment phase4
run s1800_cm_full --run-experiment s_presence --g-s-presence 1800 --d-s-presence 0
run s2025_cm_full --run-experiment s_presence --g-s-presence 2025 --d-s-presence 0

echo "=== cm_full rebuild finished $(date) ===" | tee -a "$SUMMARY"
