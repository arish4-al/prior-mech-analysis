#!/bin/bash
# Stage A + Stage B on two integrator rules (journals/paper_main_figures.md
# and journals/stage_b_fit_variants.md, 2026-10-28).
#
# Aim: compare Stage B fits whose I targets come from
#   opt1_rawslope      integrator = choice significant & overall slope > 0 (I 84)
#   opt2_shuffleslope  … & whole-curve slope beats the shuffles, FDR ≤ 0.01 (I 57)
# M (31, late-slope shuffle rule) and S (the 10 sensory regions) are the same
# in both. Every target is rebuilt on the new lists, so neither arm is
# directly comparable to production fit_targets (old 81/26 lists, old S).
#
# Stage A (retinal, L_S only, prior gains 0) is refit on the new S RMS curves
# (avg_mean_R, the 10 sensory regions), because the s89 retinal was fit to
# the old ones. Production Stage A protocol: retinal mask none,
# de_cma_local, seeds 56 34 78 89 202. avg_mean_R is the same file in both
# target dirs, so one Stage A serves both arms.
#
# Stage B arm: `full` mpre3 (submit_fit_stage_b_mpre.sh ARMS=full): no freeze,
# m_pre_weight 3, I/M prior window unset (80 ms), stim×choice stratum,
# unsplit-80 S prior on, 8 seeds, warm start = WEIGHTS_REL ∪ the new Stage A
# retinal. Loss sees the I/M traj (mean_data_results), I/M prior curves, the
# S prior 0–80 ms curve (S ∩ prior 0–80 ms: VISal VISam LP PO ZI) and the
# S RMS curves (avg_mean_R). Prior curves are from res_standard.
#
# All jobs run on pi_fiete (PARTITION to override).
#
# Step 1 — laptop (done 2026-10-28): regions + prior targets
#   python scripts/build_fit_targets_integ.py
#   then get fit_targets_integ/ onto ORCD with the repo.
#
# One submission (recommended): queues steps 2–4 as a Slurm dependency chain
#   STEP=all bash scripts/submit_fit_targets_integ.sh
#   targets job → 5 Stage A jobs (afterok) → selector job (afterany) that
#   ranks the finished Stage A finals and submits Stage B (both arms).
#   Dependents use --kill-on-invalid-dep, so a failed targets job cancels the
#   rest. The recipe check (traj_targets_check.json) is not a gate: read it
#   when the targets job ends, and scancel the chain if it is off.
#
# Or step by step:
# Step 2 — traj + S RMS targets from the insertion cache (one job):
#   STEP=targets bash scripts/submit_fit_targets_integ.sh
#   check $ONE_CACHE_DIR/manifold/fit_targets_integ/traj_targets_check.json:
#   the legacy rebuild should match fit_targets (r ≈ 1, scale ≈ 1).
# Step 3 — Stage A on the new S RMS (5 seeds):
#   STEP=retinal bash scripts/submit_fit_targets_integ.sh
#   → models/retinal_run_fr_retinal_newS_retinal_masknone_s<seed>/
#   Pick the seed by shared-stim L_S (bps 20, seed 12345), not recorded loss
#   (scripts/run_fit_integ_select_slurm.sh does this and then step 4):
#   PYTHONPATH=.:scripts python scripts/plot_retinal_fit_s.py --avg-mean-r \
#     fit_targets_integ/opt1_rawslope/avg_mean_R.npy --json <the 5 finals> \
#     --out-root <models>/retinal_s_fit_plots_retinal_newS_bps20_seed12345
# Step 4 — Stage B, both arms, from the chosen Stage A final:
#   STEP=fits RETINAL_JSON=<models>/retinal_run_fr_retinal_newS_…/retinal_final_*.json \
#     bash scripts/submit_fit_targets_integ.sh
#   → hybrid  <models>/stage_b_hybrid_WEIGHTS_REL_retinal_newS_s<seed>.json
#   → OUT_TAG stageB_hold_newS<seed>_full_mpre3_meancell_integ_{opt1_rawslope,opt2_shuffleslope}
#
# S loss rebuild (2026-10-29, journals/integ_opt2_prior_sensory_fit.md):
# L_S now scores 0–100 ms (50 bins, bin by bin) for every contrast and drops
# c=0. Stage A fits on the old loss are not comparable; refit under a new tag:
#   STEP=retinal RETINAL_TAG=retinal_newSbin SELECT=1 SUBMIT_FITS=0 \
#     bash scripts/submit_fit_targets_integ.sh
#   → 5 Stage A jobs + selector (afterany) that ranks them by shared-stim L_S
#     and stops (SUBMIT_FITS=0); SUBMIT_FITS=1 also submits Stage B.
# Hybrid / Stage B names follow RETINAL_TAG (retinal_<X> → stageB_hold_<X><seed>),
# so retinal_newS keeps the old names and a new tag cannot overwrite them.
#
# Do not submit from the laptop.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

STEP="${STEP:?set STEP=all, targets, retinal or fits}"
OPTS="${OPTS:-opt1_rawslope opt2_shuffleslope}"
RETINAL_TAG="${RETINAL_TAG:-retinal_newS}"
ONE_MODELS="${ONE_MODELS:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx/models}"
export PARTITION="${PARTITION:-pi_fiete}"
STAGEB_TAG="${RETINAL_TAG#retinal_}"
export REPO_DIR OPTS RETINAL_TAG ONE_MODELS

_need_targets() {
  for o in $OPTS; do
    for f in mean_data_results.npy avg_mean_R.npy data_act_block_duringstim.npy \
             data_act_block_duringchoice.npy data_act_block_duringstim_s_unsplit80.npy; do
      [[ -f "fit_targets_integ/$o/$f" ]] || {
        echo "ERROR: fit_targets_integ/$o/$f missing (steps 1–2)" >&2; exit 1; }
    done
  done
}

_submit_targets() {
  # shellcheck disable=SC1091
  source "$REPO_DIR/scripts/sbatch_defaults.sh"
  TARGET_DIRS=""
  for o in $OPTS; do
    [[ -f "fit_targets_integ/$o/regions.json" ]] || {
      echo "ERROR: fit_targets_integ/$o/regions.json missing (run step 1)" >&2; exit 1; }
    TARGET_DIRS="$TARGET_DIRS fit_targets_integ/$o"
  done
  export TARGET_DIRS="${TARGET_DIRS# }"
  # shellcheck disable=SC2086
  TJID=$(sbatch --parsable $SBATCH_EXTRA --partition="$PARTITION" \
    --export=ALL scripts/run_fit_traj_targets_slurm.sh)
  echo "targets job -> $TJID  (TARGET_DIRS=$TARGET_DIRS  PARTITION=$PARTITION)"
  echo "  tail -f fit_traj_targets_${TJID}.out"
}

_submit_retinal() {
  read -r -a O <<< "$OPTS"
  FIT_TARGETS_DIR="fit_targets_integ/${O[0]}" OUT_TAG="$RETINAL_TAG" \
    SEEDS="${SEEDS:-56 34 78 89 202}" VARIANTS="retinal:" \
    bash scripts/submit_fit_retinal_sharded.sh
}

case "$STEP" in
  all)
    _submit_targets
    DEP="--dependency=afterok:$TJID --kill-on-invalid-dep=yes"
    ROUT=$(SBATCH_EXTRA="$DEP" _submit_retinal)
    echo "$ROUT"
    RJIDS=$(grep -o -- '-> [0-9]*' <<< "$ROUT" | awk '{print $2}' | paste -sd: -)
    [[ -n "$RJIDS" ]] || { echo "ERROR: no Stage A job ids; scancel $TJID" >&2; exit 1; }
    echo "Stage A jobs (after $TJID) -> $RJIDS"
    SJID=$(sbatch --parsable --partition="$PARTITION" \
      --dependency="afterany:$RJIDS" --kill-on-invalid-dep=yes \
      --export=ALL scripts/run_fit_integ_select_slurm.sh)
    echo "selector job (after Stage A) -> $SJID  — submits Stage B both arms"
    echo "  tail -f integ_select_${SJID}.out"
    echo "Cancel the chain: scancel $TJID ${RJIDS//:/ } $SJID"
    ;;
  targets)
    _submit_targets
    ;;
  retinal)
    _need_targets
    read -r -a O <<< "$OPTS"
    for o in "${O[@]:1}"; do
      cmp -s "fit_targets_integ/${O[0]}/avg_mean_R.npy" "fit_targets_integ/$o/avg_mean_R.npy" || {
        echo "ERROR: avg_mean_R differs between ${O[0]} and $o; one Stage A cannot serve both" >&2
        exit 1; }
    done
    ROUT=$(_submit_retinal)
    echo "$ROUT"
    if [[ "${SELECT:-0}" == "1" ]]; then
      RJIDS=$(grep -o -- '-> [0-9]*' <<< "$ROUT" | awk '{print $2}' | paste -sd: -)
      [[ -n "$RJIDS" ]] || { echo "ERROR: no Stage A job ids" >&2; exit 1; }
      export SUBMIT_FITS="${SUBMIT_FITS:-0}"
      SJID=$(sbatch --parsable --partition="$PARTITION" \
        --dependency="afterany:$RJIDS" --kill-on-invalid-dep=yes \
        --export=ALL scripts/run_fit_integ_select_slurm.sh)
      echo "selector job (after $RJIDS) -> $SJID  SUBMIT_FITS=$SUBMIT_FITS"
      echo "  tail -f integ_select_${SJID}.out"
      echo "Cancel the chain: scancel ${RJIDS//:/ } $SJID"
    fi
    ;;
  fits)
    _need_targets
    RETINAL_JSON="${RETINAL_JSON:?set RETINAL_JSON to the chosen Stage A retinal_final_*.json}"
    [[ -f "$RETINAL_JSON" ]] || { echo "ERROR: missing $RETINAL_JSON" >&2; exit 1; }
    [[ "$RETINAL_JSON" == *"$RETINAL_TAG"* ]] || {
      echo "ERROR: RETINAL_JSON is not a $RETINAL_TAG Stage A run" >&2; exit 1; }
    rseed="$(basename "$(dirname "$RETINAL_JSON")")"
    rseed="${rseed##*_s}"
    export RETINAL_JSON
    export HYBRID_JSON="$ONE_MODELS/stage_b_hybrid_WEIGHTS_REL_${RETINAL_TAG}_s${rseed}.json"
    for o in $OPTS; do
      FIT_TARGETS_DIR="fit_targets_integ/$o" ARMS=full M_PRE_WEIGHT=3 \
        OUT_TAG_FULL="stageB_hold_${STAGEB_TAG}${rseed}_full_mpre3_meancell_integ_$o" \
        bash scripts/submit_fit_stage_b_mpre.sh
    done
    ;;
  *)
    echo "ERROR: STEP=$STEP (use all, targets, retinal or fits)" >&2; exit 1 ;;
esac
