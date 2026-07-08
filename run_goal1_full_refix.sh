#!/usr/bin/env bash
# Re-run ONLY the 4 full-classification (Goal 1.3) experiments after the
# compute_amp_slope fix (analysis_functions.py). Waits for the main matrix to
# finish first (shared condition dirs), then runs the 4 *_cm_full experiments.
#
# Usage: bash run_goal1_full_refix.sh   (run outside sandbox, iblenv)
set -u

PY=/Users/ariliu/opt/anaconda3/envs/iblenv/bin/python
SCRIPT="$(cd "$(dirname "$0")" && pwd)/simulate_recovery.py"
COMMON="--seed 123 --n-sessions 40 --nrand 100 --n-jobs 8"
BASE=/Users/ariliu/Downloads/ONE/openalyx.internationalbrainlab.org/manifold_sim
LOGDIR="$BASE/goal1/_logs"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/full_refix_status.log"

# Wait for the main matrix to finish (up to ~2h) to avoid clobbering cond dirs.
echo "[$(date +%H:%M:%S)] waiting for main matrix to finish..." | tee -a "$SUMMARY"
for _ in $(seq 1 240); do
  if grep -q "matrix finished" "$LOGDIR/matrix_status.log" 2>/dev/null; then
    break
  fi
  sleep 30
done
echo "[$(date +%H:%M:%S)] proceeding with full-classification re-runs" | tee -a "$SUMMARY"

run() {  # $1 = label, rest = args
  local label="$1"; shift
  local log="$LOGDIR/${label}.log"
  echo "[$(date +%H:%M:%S)] START $label" | tee -a "$SUMMARY"
  if $PY "$SCRIPT" "$@" >"$log" 2>&1; then
    echo "[$(date +%H:%M:%S)] OK    $label" | tee -a "$SUMMARY"
  else
    echo "[$(date +%H:%M:%S)] FAIL  $label (see $log)" | tee -a "$SUMMARY"
  fi
}

declare -a NAMES=(phase4 absence s1800 s2025)
declare -a EXPARGS=(
  "--run-experiment phase4"
  "--run-experiment absence"
  "--run-experiment s_presence --g-s-presence 1800 --d-s-presence 0"
  "--run-experiment s_presence --g-s-presence 2025 --d-s-presence 0"
)

for i in "${!NAMES[@]}"; do
  run "${NAMES[$i]}_cm_full" $COMMON ${EXPARGS[$i]} --full-analysis
done

echo "=== Goal-1 full-classification re-runs finished $(date) ===" | tee -a "$SUMMARY"
