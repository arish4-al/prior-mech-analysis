#!/bin/bash
# Submit sharded single-neuron variance-partition jobs (early 0–80 ms window).
#
# Assumes manifold/insertion_cache already exists.
#
# Runtime estimate (cache present, ~700 BWM insertions, mixed 19 regions):
#   Local smoke ≈ 0.5–1 s/insertion compute; ORCD FS load often 2–10 s.
#   Single job ≈ 0.5–2 h → borderline for 2 h walltime → default N_SHARDS=4
#   (~10–30 min/shard typical; --time=2:00:00 headroom).
#
#   bash scripts/submit_var_partition_sharded.sh
#   N_SHARDS=2 bash scripts/submit_var_partition_sharded.sh
#   TARGET=mixed WINDOW=0.08 bash scripts/submit_var_partition_sharded.sh
#
# Outputs:
#   $ONE_CACHE/manifold/var_partition/{eid_probe}.npy
#   $ONE_CACHE/manifold/res/var_partition_stacked.npy
#   $ONE_CACHE/meta/var_partition_by_region.csv
#
# Region list: repo data/stimchoice_act_regtype_regions_p_mean_c_0.01.csv

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

N_SHARDS="${N_SHARDS:-4}"
TARGET="${TARGET:-mixed}"
WINDOW="${WINDOW:-0.08}"
PRIOR_TYPE="${PRIOR_TYPE:-act}"
RESTART="${RESTART:-1}"
MEM_SHARD="${MEM_SHARD:-8G}"
MEM_FIN="${MEM_FIN:-4G}"
CPUS_SHARD="${CPUS_SHARD:-2}"
CPUS_FIN="${CPUS_FIN:-2}"
REGTYPE_CSV="${REGTYPE_CSV:-data/stimchoice_act_regtype_regions_p_mean_c_0.01.csv}"
# Optional override; default matches other ORCD Goal-2 submitters so shards
# and finalize share the same cache without requiring a manual export.
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR
export ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"
echo "ONE_CACHE_DIR=$ONE_CACHE_DIR"

echo "N_SHARDS=$N_SHARDS  TARGET=$TARGET  WINDOW=$WINDOW  PRIOR_TYPE=$PRIOR_TYPE  RESTART=$RESTART"
echo "MEM_SHARD=$MEM_SHARD  MEM_FIN=$MEM_FIN  REGTYPE_CSV=$REGTYPE_CSV"
echo "Estimate: ~700 insertions / $N_SHARDS shards; walltime 2h/shard"
echo "NOTE: RESTART=1 skips existing manifold/var_partition/*.npy — delete those"
echo "      first if re-running after prior_type/window changes."

SHARD_JOBS=()
for ((k=0; k<N_SHARDS; k++)); do
  JID=$(sbatch --parsable \
    --mem="$MEM_SHARD" --cpus-per-task="$CPUS_SHARD" \
    --job-name="varpart_s${k}" \
    --export=ALL,SHARD_IDX="$k",N_SHARDS="$N_SHARDS",TARGET="$TARGET",WINDOW="$WINDOW",PRIOR_TYPE="$PRIOR_TYPE",RESTART="$RESTART",REGTYPE_CSV="$REGTYPE_CSV" \
    scripts/run_var_partition_slurm.sh)
  SHARD_JOBS+=("$JID")
  echo "  shard $k/$N_SHARDS -> $JID"
done

DEP=$(IFS=:; echo "${SHARD_JOBS[*]}")
FID=$(sbatch --parsable \
  --mem="$MEM_FIN" --cpus-per-task="$CPUS_FIN" \
  --dependency=afterok:"$DEP" \
  --job-name="varpart_fin" \
  --export=ALL,TARGET="$TARGET",REGTYPE_CSV="$REGTYPE_CSV" \
  scripts/run_var_partition_finalize_slurm.sh)
echo "  finalize -> $FID (after $DEP)"
echo "Done. Monitor: squeue -u \$USER"
