#!/bin/bash
# Full-BWM I/M mean RMS from manifold/insertion_cache (no nulls, no act-prior).
#
# Timing (no shards):
#   Laptop: 7 caches in ~7.5 s (~1 s/insertion, I/M cells only, 2 alignments).
#   ~700 BWM insertions × 1 s ≈ 12 min on local SSD.
#   ORCD Lustre I/O is slower (2–5×); expect ~30–60 min, worst ~90 min.
#   One job, --time=2:00:00, 8G, 2 CPUs. Shards are not worth it (no
#   nrand, streaming RMS, one insertion at a time).
#
# Assumes insertion_cache already exists on the alyx ONE tree.
#
#   bash scripts/submit_mean_data_im_from_cache.sh
#   REGS=fit_targets bash scripts/submit_mean_data_im_from_cache.sh
#   PARTITION=mit_preemptable bash scripts/submit_mean_data_im_from_cache.sh
#
# Output: $ONE_CACHE_DIR/manifold/mean_data_im_from_cache/
# Do not submit from the laptop.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

PARTITION="${PARTITION:-mit_preemptable}"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/sbatch_defaults.sh"

ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"
export REGS="${REGS:-sc}"
export STIM_POST="${STIM_POST:-0.15}"
export REGTYPE_CSV="${REGTYPE_CSV:-data/stimchoice_act_regtype_regions_p_mean_c_0.01.csv}"
export CACHE_DIR="${CACHE_DIR:-$ONE_CACHE_DIR/manifold/insertion_cache}"
export OUT_DIR="${OUT_DIR:-$ONE_CACHE_DIR/manifold/mean_data_im_from_cache}"

TIME="${TIME:-2:00:00}"
MEM="${MEM:-8G}"
CPUS="${CPUS:-2}"

echo "I/M RMS from cache  REGS=$REGS  STIM_POST=$STIM_POST"
echo "ONE_CACHE_DIR=$ONE_CACHE_DIR"
echo "CACHE_DIR=$CACHE_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "TIME=$TIME  MEM=$MEM  CPUS=$CPUS  PARTITION=$PARTITION"

# shellcheck disable=SC2086
JID=$(sbatch --parsable $SBATCH_EXTRA \
  --partition="$PARTITION" \
  --mem="$MEM" --cpus-per-task="$CPUS" \
  --time="$TIME" \
  --job-name="im_rms_cache" \
  --export=ALL \
  scripts/run_mean_data_im_from_cache_slurm.sh)
echo "  job -> $JID"
echo "Done. Monitor: squeue -u \$USER"
echo "  tail -f im_rms_cache_${JID}.out"
