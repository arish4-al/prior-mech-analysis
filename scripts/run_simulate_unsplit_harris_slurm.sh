#!/bin/bash
#SBATCH --job-name=sim_unsplit_hu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH -p mit_normal
#SBATCH --time=12:00:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o sim_unsplit_hu_%x_%j.out

# Stage B unsplit Harris unique-null on ORCD (not the laptop).
# S/I: stim-aligned stim strata. M: move-aligned choice strata.
# S curves: full 0–150 ms (--s-window-ms 150); 80 ms p-values sliced later.
#
#   CASE=absence VARIANT=regular sbatch scripts/run_simulate_unsplit_harris_slurm.sh
#   CASE=s_presence VARIANT=sensory sbatch scripts/run_simulate_unsplit_harris_slurm.sh
#
# Or: bash scripts/submit_simulate_unsplit_harris_orcd.sh
#
# Long sessions / extra donors refill session_cache (multi-GB). Default nrand=1000.

set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export PRIOR_MECH_NO_ONE="${PRIOR_MECH_NO_ONE:-1}"

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"

CASE="${CASE:-}"
VARIANT="${VARIANT:-regular}"
SEED="${SEED:-123}"
N_SESSIONS="${N_SESSIONS:-40}"
BLOCKS_PER_SESSION="${BLOCKS_PER_SESSION:-40}"
NRAND="${NRAND:-1000}"
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-16}}"
HARRIS_N_EXTRA_DONORS="${HARRIS_N_EXTRA_DONORS:-80}"
S_WINDOW_MS="${S_WINDOW_MS:-150}"
OUTPUT_DIR="${OUTPUT_DIR:-$ONE_CACHE_DIR/manifold_sim/stageB_bwm/unsplit_stim_choice}"

REG_JSON="${REG_JSON:-$ONE_CACHE_DIR/models/weights_run_fj_stageB_hold_s89_regular_mask12-13_s101/weights_final_loss1p131_20260812-232651.json}"
SENS_JSON="${SENS_JSON:-$ONE_CACHE_DIR/models/weights_run_fj_stageB_hold_s89_sensory_mask6-7-8-9_s23/weights_final_loss1p148_20260812-234002.json}"
GS="${GS:-38.28114411878634}"
DS="${DS:-32.885204811626714}"

if [[ "$VARIANT" == "sensory" ]]; then
  WEIGHTS_JSON="${WEIGHTS_JSON:-$SENS_JSON}"
  CASE="${CASE:-s_presence}"
else
  WEIGHTS_JSON="${WEIGHTS_JSON:-$REG_JSON}"
  CASE="${CASE:-absence}"
fi

module load miniforge
conda activate ~/conda_envs/ibl
cd "$REPO_DIR"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

mkdir -p "$OUTPUT_DIR"

echo "Host: $(hostname) Date: $(date)"
git log -1 --oneline 2>/dev/null || true
echo "CASE=$CASE VARIANT=$VARIANT seed=$SEED n_sessions=$N_SESSIONS blocks=$BLOCKS_PER_SESSION"
echo "nrand=$NRAND extra_donors=$HARRIS_N_EXTRA_DONORS n_jobs=$N_JOBS s_window_ms=$S_WINDOW_MS"
echo "WEIGHTS_JSON=$WEIGHTS_JSON"
echo "OUTPUT_DIR=$OUTPUT_DIR"

if [[ ! -f "$WEIGHTS_JSON" ]]; then
  echo "ERROR: weights JSON not found: $WEIGHTS_JSON" >&2
  exit 1
fi

EXTRA=()
if [[ "$CASE" == "s_presence" ]]; then
  EXTRA+=(--g-s-presence "$GS" --d-s-presence "$DS")
fi

python3 -u simulate_recovery.py \
  --unsplit-prior "$CASE" \
  --unsplit-mode stim_side \
  --harris-unique-null \
  --harris-n-extra-donors "$HARRIS_N_EXTRA_DONORS" \
  --seed "$SEED" \
  --n-sessions "$N_SESSIONS" \
  --blocks-per-session "$BLOCKS_PER_SESSION" \
  --nrand "$NRAND" \
  --n-jobs "$N_JOBS" \
  --s-window-ms "$S_WINDOW_MS" \
  --weights-json "$WEIGHTS_JSON" \
  --output-dir "$OUTPUT_DIR" \
  "${EXTRA[@]}"

echo "Done: $(date)"
echo "Summary:"
ls -la "$OUTPUT_DIR"/unsplit_prior/seed_"$SEED"/*/ 2>/dev/null | head || true
