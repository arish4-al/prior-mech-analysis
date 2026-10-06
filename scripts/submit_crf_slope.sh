#!/bin/bash
# Contrast-response slope vs true-block prior (full cached BWM).
#
# Aim: does the slope of region-mean spike count vs raw contrast differ
# when the true block favors the stimulated side?
# Window: one bin, 0–150 ms from stimOn. Contrasts: 0, 0.0625, 0.125, 0.25, 1.
# Prior: true block (probabilityLeft 0.8/0.2). 0.5-blocks dropped.
# Null: label shuffle inside each (stim side, contrast) cell. nrand=10000.
#
# Assumes manifold/insertion_cache already exists. Writes
#   $ONE_CACHE/manifold/crf_slope/{eid_probe}.npy
#   $ONE_CACHE/manifold/res/crf_slope_stacked.npy
#   $ONE_CACHE/manifold/res/crf_slope_by_region.csv
#
#   bash scripts/submit_crf_slope.sh
#   N_SHARDS=4 NRAND=10000 bash scripts/submit_crf_slope.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

PARTITION="${PARTITION:-pi_fiete}"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/sbatch_defaults.sh"

N_SHARDS="${N_SHARDS:-4}"
NRAND="${NRAND:-10000}"
RESTART="${RESTART:-1}"
MEM_SHARD="${MEM_SHARD:-16G}"
MEM_FIN="${MEM_FIN:-8G}"
CPUS_SHARD="${CPUS_SHARD:-2}"
CPUS_FIN="${CPUS_FIN:-2}"
TIME_SHARD="${TIME_SHARD:-4:00:00}"
TIME_FIN="${TIME_FIN:-0:30:00}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR
export ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"

echo "ONE_CACHE_DIR=$ONE_CACHE_DIR"
echo "N_SHARDS=$N_SHARDS NRAND=$NRAND RESTART=$RESTART"
echo "TIME_SHARD=$TIME_SHARD MEM_SHARD=$MEM_SHARD"
echo "Aim: true-block CRF slope difference, window 0-0.15 s, contrasts incl. 0%"

SHARD_JOBS=()
for ((k=0; k<N_SHARDS; k++)); do
  # shellcheck disable=SC2086
  JID=$(sbatch --parsable $SBATCH_EXTRA \
    --partition="$PARTITION" \
    --mem="$MEM_SHARD" --cpus-per-task="$CPUS_SHARD" \
    --time="$TIME_SHARD" \
    --job-name="crf_s${k}" \
    --export=ALL,SHARD_IDX="$k",N_SHARDS="$N_SHARDS",NRAND="$NRAND",RESTART="$RESTART" \
    scripts/run_crf_slope_slurm.sh)
  SHARD_JOBS+=("$JID")
  echo "  shard $k/$N_SHARDS -> $JID"
done

DEP=$(IFS=:; echo "${SHARD_JOBS[*]}")
# shellcheck disable=SC2086
FID=$(sbatch --parsable $SBATCH_EXTRA \
  --partition="$PARTITION" \
  --mem="$MEM_FIN" --cpus-per-task="$CPUS_FIN" \
  --time="$TIME_FIN" \
  --dependency=afterok:"$DEP" \
  --job-name="crf_fin" \
  --export=ALL \
  scripts/run_crf_slope_finalize_slurm.sh)
echo "  finalize -> $FID (after $DEP)"
echo "Done. Monitor: squeue -u \$USER"
