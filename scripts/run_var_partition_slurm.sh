#!/bin/bash
#SBATCH --job-name=varpart_shard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p mit_normal
#SBATCH --time=2:00:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o varpart_shard_%x_%j.out

# One insertion shard of variance-partition fits (cache assumed present).
# Submit via scripts/submit_var_partition_sharded.sh
#
#   SHARD_IDX=0 N_SHARDS=4 sbatch --export=ALL,SHARD_IDX,N_SHARDS \
#     scripts/run_var_partition_slurm.sh

set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"

SHARD_IDX="${SHARD_IDX:?Set SHARD_IDX=0..N-1}"
N_SHARDS="${N_SHARDS:-4}"
TARGET="${TARGET:-mixed}"
WINDOW="${WINDOW:-0.08}"
PRIOR_TYPE="${PRIOR_TYPE:-act}"
RESTART="${RESTART:-1}"
REGTYPE_CSV="${REGTYPE_CSV:-data/stimchoice_act_regtype_regions_p_mean_c_0.01.csv}"

module load miniforge
conda activate ~/conda_envs/ibl
cd "$REPO_DIR"

echo "Host: $(hostname) Date: $(date)"
git log -1 --oneline
echo "var_partition shard=$SHARD_IDX/$N_SHARDS target=$TARGET window=$WINDOW prior=$PRIOR_TYPE"
echo "ONE_CACHE_DIR=$ONE_CACHE_DIR"
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-?} SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}"

ARGS=(--target "$TARGET" --window "$WINDOW" --prior-type "$PRIOR_TYPE"
      --regtype-csv "$REGTYPE_CSV"
      --one-cache-dir "$ONE_CACHE_DIR"
      --shard-idx "$SHARD_IDX" --n-shards "$N_SHARDS" --no-stack)
[[ "$RESTART" == "1" ]] && ARGS+=(--restart) || ARGS+=(--no-restart)

python3 -u scripts/run_var_partition.py "${ARGS[@]}"
echo "Shard done: $(date)"
ls -lh "$ONE_CACHE_DIR/manifold/var_partition" 2>/dev/null | tail -5 || true
