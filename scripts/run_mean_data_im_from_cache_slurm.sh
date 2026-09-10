#!/bin/bash
#SBATCH --job-name=im_rms_cache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p mit_preemptable
#SBATCH --time=2:00:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o im_rms_cache_%j.out

# Full-BWM I/M RMS from insertion_cache. One job — no shards.
# Submit via scripts/submit_mean_data_im_from_cache.sh (do not sbatch from laptop).

set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"

REGS="${REGS:-sc}"
STIM_POST="${STIM_POST:-0.15}"
REGTYPE_CSV="${REGTYPE_CSV:-data/stimchoice_act_regtype_regions_p_mean_c_0.01.csv}"
CACHE_DIR="${CACHE_DIR:-$ONE_CACHE_DIR/manifold/insertion_cache}"
OUT_DIR="${OUT_DIR:-$ONE_CACHE_DIR/manifold/mean_data_im_from_cache}"

module load miniforge
conda activate ~/conda_envs/ibl
cd "$REPO_DIR"

echo "Host: $(hostname) Date: $(date)"
git log -1 --oneline
echo "ONE_CACHE_DIR=$ONE_CACHE_DIR"
echo "CACHE_DIR=$CACHE_DIR  n=$(ls -1 "$CACHE_DIR"/*.npy 2>/dev/null | wc -l)"
echo "OUT_DIR=$OUT_DIR  REGS=$REGS  STIM_POST=$STIM_POST"
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-?} SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}"

python3 -u scripts/build_mean_data_im_from_cache.py \
  --cache-dir "$CACHE_DIR" \
  --out-dir "$OUT_DIR" \
  --regs "$REGS" \
  --stim-post "$STIM_POST" \
  --regtype-csv "$REGTYPE_CSV"

echo "Done: $(date)"
ls -lh "$OUT_DIR"
