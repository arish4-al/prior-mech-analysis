#!/usr/bin/env bash
# Goal-1 matrix driver (2026-07-06): run the three new analyses on the four
# experiments, reusing the session cache (simulate once per experiment).
#
#   Analyses:  cm_sprior (baseline S/I/M), ls_sprior (1.1 no contrast match),
#              unsplit cm+ls (1.2 no f1/f2 splits), full (1.3 classification)
#   Experiments: E1 phase4, E2 absence, E3 s_presence g_s=1800, E4 s_presence g_s=2025
#
# Usage: bash run_goal1_matrix.sh   (run outside sandbox, iblenv)
set -u

PY=/Users/ariliu/opt/anaconda3/envs/iblenv/bin/python
SCRIPT="$(cd "$(dirname "$0")" && pwd)/simulate_recovery.py"
COMMON="--seed 123 --n-sessions 40 --nrand 100 --n-jobs 8"
BASE=/Users/ariliu/Downloads/ONE/openalyx.internationalbrainlab.org/manifold_sim
LOGDIR="$BASE/goal1/_logs"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/matrix_status.log"
echo "=== Goal-1 matrix started $(date) ===" | tee -a "$SUMMARY"

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

# Experiment specs: name | run-experiment args | unsplit args
declare -a NAMES=(phase4 absence s1800 s2025)
declare -a EXPARGS=(
  "--run-experiment phase4"
  "--run-experiment absence"
  "--run-experiment s_presence --g-s-presence 1800 --d-s-presence 0"
  "--run-experiment s_presence --g-s-presence 2025 --d-s-presence 0"
)
declare -a UNSPLITARGS=(
  "--unsplit-prior phase4"
  "--unsplit-prior absence"
  "--unsplit-prior s_presence --g-s-presence 1800 --d-s-presence 0"
  "--unsplit-prior s_presence --g-s-presence 2025 --d-s-presence 0"
)

for i in "${!NAMES[@]}"; do
  n="${NAMES[$i]}"
  exp="${EXPARGS[$i]}"
  uns="${UNSPLITARGS[$i]}"
  # baseline contrast-matched S/I/M split (first run per experiment => cache MISS)
  run "${n}_cm_sprior"      $COMMON $exp
  # 1.1 no contrast matching
  run "${n}_ls_sprior"      $COMMON $exp --label-shuffle-null
  # 1.2 no f1/f2 splits (stim-side unsplit), both nulls
  run "${n}_cm_unsplit"     $COMMON $uns
  run "${n}_ls_unsplit"     $COMMON $uns --label-shuffle-null
  # 1.3 full classification (contrast-matched)
  run "${n}_cm_full"        $COMMON $exp --full-analysis
done

echo "=== Goal-1 matrix finished $(date) ===" | tee -a "$SUMMARY"
