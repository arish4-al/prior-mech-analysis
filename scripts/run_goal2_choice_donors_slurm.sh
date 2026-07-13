#!/bin/bash
#SBATCH --job-name=goal2_choice_donors
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p mit_normal
#SBATCH --time=2:00:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o goal2_choice_donors_%j.out

# Rebuild manifold/choice_donors.npy from insertion_cache (stratified format).
# Optional: SMOKE_FIRST=1 also runs scripts/smoke_choice_session_null.py.
#
#   sbatch --export=ALL scripts/run_goal2_choice_donors_slurm.sh
# Prefer: bash scripts/submit_goal2_choice_session_null_sharded.sh
#   (shards depend on this job)

set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"
SMOKE_FIRST="${SMOKE_FIRST:-0}"

module load miniforge
conda activate ~/conda_envs/ibl
cd "$REPO_DIR"

echo "Host: $(hostname) Date: $(date)"
git log -1 --oneline
echo "ONE_CACHE_DIR=$ONE_CACHE_DIR SMOKE_FIRST=$SMOKE_FIRST"

python3 -u scripts/run_goal2_splits.py --build-choice-donors

python3 -u - <<'PY'
from pathlib import Path
import numpy as np
import os
p = Path(os.environ['ONE_CACHE_DIR'], 'manifold', 'choice_donors.npy')
if not p.exists():
    raise SystemExit(f'Missing donor bank: {p}')
bank = np.load(p, allow_pickle=True).item()
if not bank:
    raise SystemExit(f'Empty donor bank: {p}')
rec = next(iter(bank.values()))
if not isinstance(rec, dict) or 'stim_is_left' not in rec or 'pleft_true' not in rec:
    raise SystemExit(f'Donor bank at {p} is legacy/choice-only')
print(f'Donor bank OK: {len(bank)} eids, stratified format -> {p}')
PY

if [[ "$SMOKE_FIRST" == "1" ]]; then
  echo "Running smoke_choice_session_null.py …"
  python3 -u scripts/smoke_choice_session_null.py
fi

echo "Choice donors done: $(date)"
ls -lh "$ONE_CACHE_DIR/manifold/choice_donors.npy"
