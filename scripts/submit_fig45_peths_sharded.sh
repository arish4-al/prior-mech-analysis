#!/bin/bash
# Figure 4.5 rerun on the concat_cvTrue cell types: N_SHARDS compute jobs over
# the typed insertions, then one stack + plot job (afterok).
#
#   bash scripts/submit_fig45_peths_sharded.sh
#
# Needs on ORCD: $OUT/cell_types.npy (from fig45_define_cell_types.py save,
# copied from the laptop), manifold/insertion_cache and bwm_tables/trials.pqt
# under ONE_CACHE_DIR. Cached insertions take ~1-30 s each at nrand=2000.
#
# Override: N_SHARDS=4 MEM_SHARD=8G PARTITION=mit_preemptable \
#   TYPES="integrator move_all" \
#   bash scripts/submit_fig45_peths_sharded.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
OUT="${OUT:-$ONE_CACHE_DIR/dmn/res/fig45/cvTrue}"
CELL_TYPES="${CELL_TYPES:-$OUT/cell_types.npy}"
TYPES="${TYPES:-integrator move_all move_init move move_fback}"
N_SHARDS="${N_SHARDS:-8}"
MEM_SHARD="${MEM_SHARD:-6G}"
MEM_STACK="${MEM_STACK:-8G}"
CPUS_SHARD="${CPUS_SHARD:-2}"
PARTITION="${PARTITION:-pi_fiete}"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/sbatch_defaults.sh"

[[ -f "$CELL_TYPES" ]] || { echo "missing $CELL_TYPES" >&2; exit 1; }
[[ -f "$ONE_CACHE_DIR/bwm_tables/trials.pqt" ]] || {
  echo "missing $ONE_CACHE_DIR/bwm_tables/trials.pqt" >&2; exit 1; }

COMMON="ALL,ONE_CACHE_DIR=$ONE_CACHE_DIR,OUT=$OUT,CELL_TYPES=$CELL_TYPES,TYPES=$TYPES"
echo "PARTITION=$PARTITION N_SHARDS=$N_SHARDS MEM_SHARD=$MEM_SHARD OUT=$OUT"
echo "TYPES=$TYPES"

SHARD_JOBS=()
for ((k=0; k<N_SHARDS; k++)); do
  # shellcheck disable=SC2086
  JID=$(sbatch --parsable $SBATCH_EXTRA \
    --partition="$PARTITION" \
    --mem="$MEM_SHARD" --cpus-per-task="$CPUS_SHARD" \
    --job-name="f45_s${k}" \
    --export="$COMMON,STEP=compute,SHARD_IDX=$k,N_SHARDS=$N_SHARDS" \
    scripts/run_fig45_peths_slurm.sh)
  SHARD_JOBS+=("$JID")
  echo "  shard $k/$N_SHARDS -> $JID"
done
DEP=$(IFS=:; echo "${SHARD_JOBS[*]}")
# shellcheck disable=SC2086
SID=$(sbatch --parsable $SBATCH_EXTRA \
  --partition="$PARTITION" \
  --mem="$MEM_STACK" --cpus-per-task=2 \
  --dependency=afterok:"$DEP" \
  --job-name="f45_stack" \
  --export="$COMMON,STEP=stack" \
  scripts/run_fig45_peths_slurm.sh)
echo "  stack + plots -> $SID (after $DEP)"
