#!/bin/bash
#SBATCH --job-name=varpart_fin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH -p mit_normal
#SBATCH --time=0:30:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o varpart_fin_%x_%j.out

# Stack per-insertion var_partition/*.npy → region summary (after shards).
#
#   sbatch --export=ALL scripts/run_var_partition_finalize_slurm.sh

set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"

TARGET="${TARGET:-mixed}"
REGTYPE_CSV="${REGTYPE_CSV:-data/stimchoice_act_regtype_regions_p_mean_c_0.01.csv}"

module load miniforge
conda activate ~/conda_envs/ibl
cd "$REPO_DIR"

echo "Host: $(hostname) Date: $(date) target=$TARGET"
echo "ONE_CACHE_DIR=$ONE_CACHE_DIR"
ARGS=(--stack-only --target "$TARGET" --regtype-csv "$REGTYPE_CSV"
      --one-cache-dir "$ONE_CACHE_DIR")
python3 -u scripts/run_var_partition.py "${ARGS[@]}"
echo "Finalize done: $(date)"
