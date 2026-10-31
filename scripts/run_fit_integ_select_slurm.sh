#!/bin/bash
#SBATCH --job-name=integ_select
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p pi_fiete
#SBATCH --time=1:00:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o integ_select_%j.out

# Pick the Stage A retinal seed for the integrator-rule arms, then submit them.
# Runs after the Stage A jobs (STEP=all in scripts/submit_fit_targets_integ.sh).
# Ranks every finished retinal_run_fr_<RETINAL_TAG>_retinal_masknone_s* final by
# shared-stim L_S on the new avg_mean_R (plot_retinal_fit_s.py, bps 20,
# seed 12345), as for production s89, and submits STEP=fits with the best.
# Requeue-safe (2026-10-30, mit_preemptable --requeue): a marker in RANK_DIR,
# created atomically before submitting, stops a requeued run from submitting
# Stage B twice. Delete it to submit again by hand.
# SUBMIT_FITS=0 stops after the ranking (summary.json in RANK_DIR).

set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"
export PRIOR_MECH_NO_ONE="${PRIOR_MECH_NO_ONE:-1}"
ONE_MODELS="${ONE_MODELS:-$ONE_CACHE_DIR/models}"
RETINAL_TAG="${RETINAL_TAG:-retinal_newS}"
OPTS="${OPTS:-opt1_rawslope opt2_shuffleslope}"

module load miniforge
conda activate ~/conda_envs/ibl
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:$REPO_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"

echo "Host: $(hostname) Date: $(date)"
git log -1 --oneline

read -r -a O <<< "$OPTS"
for o in "${O[@]:1}"; do
  cmp -s "fit_targets_integ/${O[0]}/avg_mean_R.npy" "fit_targets_integ/$o/avg_mean_R.npy" || {
    echo "ERROR: avg_mean_R differs between ${O[0]} and $o" >&2; exit 1; }
done

FINALS=()
for d in "$ONE_MODELS"/retinal_run_fr_"$RETINAL_TAG"_retinal_masknone_s*/; do
  f=$(ls -t "$d"retinal_final_*.json 2>/dev/null | head -1 || true)
  [[ -n "$f" ]] && FINALS+=("$f")
done
echo "Stage A finals: ${#FINALS[@]}"
printf '  %s\n' "${FINALS[@]}"
[[ ${#FINALS[@]} -gt 0 ]] || { echo "ERROR: no Stage A finals for $RETINAL_TAG" >&2; exit 1; }

RANK_DIR="$ONE_MODELS/retinal_s_fit_plots_${RETINAL_TAG}_bps20_seed12345"
python3 -u scripts/plot_retinal_fit_s.py \
  --json "${FINALS[@]}" \
  --avg-mean-r "fit_targets_integ/${O[0]}/avg_mean_R.npy" \
  --bps 20 --seed 12345 --out-root "$RANK_DIR"

BEST=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['rows'][0]['json'])" \
  "$RANK_DIR/summary.json")
echo "Best Stage A (shared L_S): $BEST"
if [[ "${SUBMIT_FITS:-1}" == "0" ]]; then
  echo "SUBMIT_FITS=0: ranking only, Stage B not submitted. Done: $(date)"
  exit 0
fi

MARK="$RANK_DIR/stageB_submitted_${OPTS// /_}"
if ! ( set -o noclobber; echo "$SLURM_JOB_ID $(date)" > "$MARK" ) 2>/dev/null; then
  echo "Stage B already submitted ($MARK: $(cat "$MARK")); not submitting again."
  exit 0
fi
STEP=fits RETINAL_JSON="$BEST" RETINAL_TAG="$RETINAL_TAG" OPTS="$OPTS" \
  ONE_MODELS="$ONE_MODELS" bash scripts/submit_fit_targets_integ.sh
echo "Done: $(date)"
