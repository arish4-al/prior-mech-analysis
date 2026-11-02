#!/bin/bash
# Standard-split alternative nulls. Same labels as the shuffle catalog:
# action kernel (α=0.2) on the full session, then the bad-trial / saturation
# mask, then drop true probabilityLeft==0.5, then min_trials_per_side=5.
#
# Aim: those labels, scored under the same splits for both nulls:
#   sticky — option-1 action-kernel pseudo-session plus copy-last.
#            Writes res_standard/{split}_pseudo_strat_sticky.npy
#   harris — unique session transplant.
#            Writes res_standard/{split}_harris_unique.npy
#   choice — choice L vs R (8), stratum = stim × act prior.
#   prior_stim — prior L vs R inside stim side (2).
#   prior_none — prior L vs R with no stratum (2 fully unsplit).
#   prior_f12 — prior L vs R inside stim × choice (8 Figure 4 f1/f2;
#            ARM=harris_prior_f12 / sticky_prior_f12, not part of ARM=all).
#
# Harris reads manifold/choice_donors_standard.npy (full-session kernel on
# the donor, then the same mask and true-0.5 drop). The shuffle catalog in
# res_standard/{split}.npy is left in place.
#
#   bash scripts/submit_standard_alt_nulls_orcd.sh
#   ARM=sticky|sticky_prior_f12|harris_choice|harris_prior_stim|harris_prior_none|harris_prior_f12|all
#   REBUILD_DONORS=1   rebuild the standard donor bank (default 0)
#   PREFIT=0           skip the sticky θ prefit (default 1). The prefit fits
#                      each session once on the full session-loader table
#                      ({eid8}_fullsess) before any sticky shard starts, so
#                      every split uses the same θ; sessions already fit
#                      are skipped. N_PREFIT_SHARDS (24; 459 eids, up to
#                      ~17 min each) MEM_PREFIT (8G)
#                      TIME_PREFIT (12:00:00)
#   PREFLIGHT=1        run the offline input check first (default 0: it
#                      passed on ORCD 2026-11-02 at 39cd06c, 459 sessions,
#                      0 errors, ~13 min). Rerun it after the session
#                      tables, θ pickles or donor bank change. Every job
#                      runs with ONE offline and Alyx blocked (one_offline.py);
#                      the check loads each session's inputs the same way and
#                      stops the submit if any would need Alyx.
#   PREFIT_DEP=j1:j2:… with PREFIT=0: sticky shards wait on prefit jobs
#                      already queued by another submit (e.g. ARM=sticky,
#                      then ARM=sticky_prior_f12), so two prefits never
#                      fit the same session at once. Only for prefit jobs
#                      still pending/running; once they are done, PREFIT=0
#                      alone is enough
#   DEBIAS_EUC=1       noise-corrected d_euc; outputs and CLEAR_STREAM use
#                      {suffix}_debias, so uncorrected files are kept
#   SKIP_SPLITS="a b"  space-separated split names not to submit (finished
#                      runs); applies to every arm in this submit, and
#                      CLEAR_STREAM leaves their files alone
#   N_SHARDS=  MEM_SHARD=  TIME_SHARD=  override that arm's defaults
#     (ARM=all applies one override to every arm)
#
# Defaults: choice 4 shards (sticky 12G, Harris 6G), f1/f2 prior sticky
# 8 shards × 12G (Harris 4 × 8G), stim-stratum prior
# 4 shards (sticky 12G, Harris 8G), no-stratum prior 12×16G. nrand=2000.
# CLEAR_STREAM=1 clears only that arm's suffix under res_standard.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

ARM="${ARM:-all}"
NRAND="${NRAND:-2000}"
RESTART="${RESTART:-1}"
CLEAR_STREAM="${CLEAR_STREAM:-1}"
REBUILD_DONORS="${REBUILD_DONORS:-0}"
PREFIT="${PREFIT:-1}"
PREFLIGHT="${PREFLIGHT:-0}"  # checked 2026-11-02 (39cd06c)
N_PREFIT_SHARDS="${N_PREFIT_SHARDS:-24}"
MEM_PREFIT="${MEM_PREFIT:-8G}"
TIME_PREFIT="${TIME_PREFIT:-12:00:00}"
DEBIAS_EUC="${DEBIAS_EUC:-0}"
SKIP_SPLITS="${SKIP_SPLITS:-}"
N_SHARDS_USER="${N_SHARDS:-}"
MEM_SHARD_USER="${MEM_SHARD:-}"
TIME_SHARD_USER="${TIME_SHARD:-}"
MEM_FIN="${MEM_FIN:-16G}"
CPUS_SHARD="${CPUS_SHARD:-2}"
CPUS_FIN="${CPUS_FIN:-2}"
MEM_DONORS="${MEM_DONORS:-16G}"
CPUS_DONORS="${CPUS_DONORS:-2}"
TIME_DONORS="${TIME_DONORS:-8:00:00}"
# mit_preemptable: sbatch_defaults adds --requeue to every job. Shards resume
# from their atomic stream checkpoint; the prefit refits any session whose
# pickle is missing or unreadable.
PARTITION="${PARTITION:-mit_preemptable}"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/sbatch_defaults.sh"

ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"

PY="${PY:-$HOME/conda_envs/ibl/bin/python}"
if [[ ! -x "$PY" ]]; then
  echo "ERROR: ibl python not found at $PY" >&2
  exit 1
fi
if [[ ! -d third_party/behavior_models/behavior_models ]]; then
  echo "ERROR: missing submodule third_party/behavior_models" >&2
  echo "  git submodule update --init --recursive" >&2
  exit 1
fi

case "$ARM" in
  sticky|sticky_prior_f12|harris_choice|harris_prior_stim|harris_prior_none|harris_prior_f12|all) ;;
  *)
    echo "ERROR: ARM must be sticky|sticky_prior_f12|harris_choice|harris_prior_stim|harris_prior_none|harris_prior_f12|all (got $ARM)" >&2
    exit 1
    ;;
esac

echo "Aim: standard act labels (full-session kernel, mask, drop true 0.5, min5)."
echo "ARM=$ARM  nrand=$NRAND  CLEAR_STREAM=$CLEAR_STREAM  REBUILD_DONORS=$REBUILD_DONORS  PREFIT=$PREFIT  DEBIAS_EUC=$DEBIAS_EUC"
echo "PARTITION=$PARTITION  SBATCH_EXTRA=${SBATCH_EXTRA:-}"
echo "Outputs: \$ONE_CACHE_DIR/manifold/res_standard/{split}{suffix}.npy"

job_tag() {
  local s="$1"
  s="${s//./p}"
  echo "${s:0:40}"
}

list_splits() {
  local preset="$1"
  SPLITS=()
  while IFS= read -r line; do
    [[ -n "$line" ]] && SPLITS+=("$line")
  done < <("$PY" -u scripts/run_goal2_splits.py --preset "$preset" --list-splits)
  if [[ ${#SPLITS[@]} -eq 0 ]]; then
    echo "ERROR: no splits for PRESET=$preset" >&2
    exit 1
  fi
}

submit_arm() {
  local preset="$1" scheme="$2" shards="$3" mem="$4" tshard="$5" prefix="$6"
  if [[ -n "$N_SHARDS_USER" ]]; then shards="$N_SHARDS_USER"; fi
  if [[ -n "$MEM_SHARD_USER" ]]; then mem="$MEM_SHARD_USER"; fi
  if [[ -n "$TIME_SHARD_USER" ]]; then tshard="$TIME_SHARD_USER"; fi

  local mode="" sticky=0 shuffle=0 ak=0 suffix=""
  if [[ "$scheme" == "sticky" ]]; then
    ak=1
    mode=strat
    sticky=1
    suffix=_pseudo_strat_sticky
  else
    shuffle=1
    suffix=_harris_unique
  fi
  if [[ "$DEBIAS_EUC" == "1" ]]; then suffix="${suffix}_debias"; fi

  list_splits "$preset"
  if [[ -n "$SKIP_SPLITS" ]]; then
    local keep=() sp2
    for sp2 in "${SPLITS[@]}"; do
      if [[ " $SKIP_SPLITS " == *" $sp2 "* ]]; then
        echo "SKIP_SPLITS: not submitting $sp2${suffix}"
      else
        keep+=("$sp2")
      fi
    done
    if [[ ${#keep[@]} -eq 0 ]]; then
      echo "PRESET=$preset  suffix=$suffix: every split skipped"
      return 0
    fi
    SPLITS=("${keep[@]}")
  fi
  local RES_ROOT="$ONE_CACHE_DIR/manifold/res_standard"
  if [[ "$CLEAR_STREAM" == "1" ]]; then
    local ACC="$RES_ROOT/_stream_acc"
    echo "CLEAR_STREAM=1: removing prior ${suffix} under $RES_ROOT"
    local sp
    for sp in "${SPLITS[@]}"; do
      local base="${sp}${suffix}"
      rm -f "$ACC/${base}.npy" "$ACC/${base}.shard"*.npy 2>/dev/null || true
      rm -f "$ACC/${base}.skipped.jsonl" "$ACC/${base}.shard"*.skipped.jsonl 2>/dev/null || true
      rm -f "$RES_ROOT/${base}.npy" "$RES_ROOT/${base}_regde.npy" \
        "$RES_ROOT/${base}_all.npy" "$RES_ROOT/${base}_all_regde.npy" \
        "$RES_ROOT/${base}_skipped.json" 2>/dev/null || true
    done
  fi

  echo "PRESET=$preset  scheme=$scheme  suffix=$suffix  shards=$shards  mem=$mem  time=$tshard"
  printf '  %s\n' "${SPLITS[@]}"

  local EXPORT="STANDARD_ACT_PRIOR=1,EXCLUDE_STICKY_TRIALS=0,SESSION_SHUFFLE_NULL=${shuffle},ACTKERNEL_CHOICE_NULL=${ak},ACTKERNEL_NULL_MODE=${mode},ACTKERNEL_LATE_STICKY=${sticky},ACTKERNEL_PSEUDO_LEN_FACTOR=,DEBIAS_EUC=${DEBIAS_EUC}"
  local dep_flag=()
  if [[ -n "${DONOR_JID:-}" && "$scheme" == "harris" ]]; then
    dep_flag=(--dependency="afterok:${DONOR_JID}")
  fi
  if [[ -n "${PREFIT_DEP:-}" && "$scheme" == "sticky" ]]; then
    dep_flag=(--dependency="afterok:${PREFIT_DEP}")
  fi

  local sp k TAG JID FID DEP
  for sp in "${SPLITS[@]}"; do
    TAG=$(job_tag "$sp")
    local SHARD_JOBS=()
    for ((k=0; k<shards; k++)); do
      # shellcheck disable=SC2086
      JID=$(sbatch --parsable $SBATCH_EXTRA \
        --partition="$PARTITION" \
        --mem="$mem" --cpus-per-task="$CPUS_SHARD" --time="$tshard" \
        --job-name="${prefix}_${TAG}_s${k}" \
        ${dep_flag[@]+"${dep_flag[@]}"} \
        --export=ALL,SPLIT="$sp",SHARD_IDX="$k",N_SHARDS="$shards",NRAND="$NRAND",RESTART="$RESTART",$EXPORT \
        scripts/run_goal2_shard_slurm.sh)
      SHARD_JOBS+=("$JID")
      echo "  $sp shard $k/$shards -> $JID"
    done
    DEP=$(IFS=:; echo "${SHARD_JOBS[*]}")
    # shellcheck disable=SC2086
    FID=$(sbatch --parsable $SBATCH_EXTRA \
      --partition="$PARTITION" \
      --mem="$MEM_FIN" --cpus-per-task="$CPUS_FIN" --time=2:00:00 \
      --dependency=afterok:"$DEP" \
      --job-name="${prefix}_fin_${TAG}" \
      --export=ALL,SPLIT="$sp",$EXPORT \
      scripts/run_goal2_finalize_slurm.sh)
    echo "  $sp finalize -> $FID"
  done
}

need_donors=0
case "$ARM" in
  harris_choice|harris_prior_stim|harris_prior_none|harris_prior_f12|all) need_donors=1 ;;
esac

if [[ "$PREFLIGHT" == "1" ]]; then
  PF_ARGS=()
  case "$ARM" in
    sticky|sticky_prior_f12|all)
      PF_ARGS+=(--sticky)
      if [[ "$PREFIT" == "1" || -n "${PREFIT_DEP:-}" ]]; then
        PF_ARGS+=(--no-pickle-check)
      fi ;;
  esac
  [[ "$need_donors" == "1" ]] && PF_ARGS+=(--harris)
  echo "Offline preflight: check_offline_inputs.py ${PF_ARGS[*]-}"
  if ! "$PY" -u scripts/check_offline_inputs.py ${PF_ARGS[@]+"${PF_ARGS[@]}"} \
      --one-cache-dir "$ONE_CACHE_DIR"; then
    echo "ERROR: offline preflight failed; nothing submitted" >&2
    exit 1
  fi
fi

DONOR_JID=""
DONOR_BANK="$ONE_CACHE_DIR/manifold/choice_donors_standard.npy"
if [[ "$need_donors" == "1" && "$REBUILD_DONORS" == "0" && -s "$DONOR_BANK" ]]; then
  # The job would only reload the bank; shards read it directly.
  echo "Donor bank present, REBUILD_DONORS=0: no donor job ($DONOR_BANK)"
  need_donors=0
fi
if [[ "$need_donors" == "1" ]]; then
  # shellcheck disable=SC2086
  DONOR_JID=$(sbatch --parsable $SBATCH_EXTRA \
    --partition="$PARTITION" \
    --mem="$MEM_DONORS" --cpus-per-task="$CPUS_DONORS" --time="$TIME_DONORS" \
    --job-name="std_donors" \
    --export=ALL,REBUILD_DONORS="$REBUILD_DONORS" \
    scripts/run_goal2_standard_donors_slurm.sh)
  echo "standard donor bank -> $DONOR_JID"
fi

PREFIT_DEP="${PREFIT_DEP:-}"
if [[ "$PREFIT" == "1" && -n "$PREFIT_DEP" ]]; then
  echo "ERROR: PREFIT_DEP is set; pass PREFIT=0 to wait on those jobs instead" >&2
  exit 1
fi
if [[ ( "$ARM" == "sticky" || "$ARM" == "sticky_prior_f12" || "$ARM" == "all" ) && "$PREFIT" == "1" ]]; then
  PREFIT_JOBS=()
  for ((k=0; k<N_PREFIT_SHARDS; k++)); do
    # shellcheck disable=SC2086
    JID=$(sbatch --parsable $SBATCH_EXTRA \
      --partition="$PARTITION" \
      --mem="$MEM_PREFIT" --cpus-per-task=2 --time="$TIME_PREFIT" \
      --job-name="std_akfit_s${k}" \
      --export=ALL,PREFIT_FULL_SESSION=1,SHARD_IDX="$k",N_SHARDS="$N_PREFIT_SHARDS" \
      scripts/run_goal2_ak_prefit_slurm.sh)
    PREFIT_JOBS+=("$JID")
    echo "sticky θ prefit (full session) shard $k/$N_PREFIT_SHARDS -> $JID"
  done
  PREFIT_DEP=$(IFS=:; echo "${PREFIT_JOBS[*]}")
  echo "PREFIT_DEP=$PREFIT_DEP  (pass with PREFIT=0 to another sticky submit)"
fi

if [[ "$ARM" == "sticky" || "$ARM" == "all" ]]; then
  submit_arm standard_choice_harris sticky 4 12G 12:00:00 stdssc
  submit_arm standard_prior_harris_stim sticky 4 12G 12:00:00 stdsss
  submit_arm standard_prior_harris_none sticky 12 16G 12:00:00 stdssn
fi
if [[ "$ARM" == "harris_choice" || "$ARM" == "all" ]]; then
  submit_arm standard_choice_harris harris 4 6G 12:00:00 stdhc
fi
if [[ "$ARM" == "harris_prior_stim" || "$ARM" == "all" ]]; then
  submit_arm standard_prior_harris_stim harris 4 8G 12:00:00 stdhs
fi
if [[ "$ARM" == "harris_prior_none" || "$ARM" == "all" ]]; then
  submit_arm standard_prior_harris_none harris 12 16G 12:00:00 stdhn
fi
if [[ "$ARM" == "sticky_prior_f12" ]]; then
  submit_arm standard_prior_harris_f12 sticky 8 12G 12:00:00 stdssf
fi
if [[ "$ARM" == "harris_prior_f12" ]]; then
  submit_arm standard_prior_harris_f12 harris 4 8G 12:00:00 stdhf
fi

echo "Done (standard alternative nulls). Monitor: squeue -u \$USER"
echo "Final outputs: \$ONE_CACHE_DIR/manifold/res_standard/{split}_pseudo_strat_sticky[_debias].npy and {split}_harris_unique[_debias].npy"
