#!/bin/bash
#SBATCH --job-name=std_donors
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -p pi_fiete
#SBATCH --time=8:00:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o std_donors_%j.out

# Build manifold/choice_donors_standard.npy for the standard Harris arms.
# Full-session action kernel, then the saturation mask, then drop true
# probabilityLeft==0.5. REBUILD_DONORS=1 rewrites an existing file.
#
#   sbatch --export=ALL,REBUILD_DONORS=0 scripts/run_goal2_standard_donors_slurm.sh
# Prefer: bash scripts/submit_standard_alt_nulls_orcd.sh

set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR ONE_BASE_URL="${ONE_BASE_URL:-https://alyx.internationalbrainlab.org}"
REBUILD_DONORS="${REBUILD_DONORS:-0}"
export REBUILD_DONORS

module load miniforge
conda activate ~/conda_envs/ibl
cd "$REPO_DIR"

echo "Host: $(hostname) Date: $(date)"
git log -1 --oneline
echo "ONE_CACHE_DIR=$ONE_CACHE_DIR REBUILD_DONORS=$REBUILD_DONORS"

python3 -u scripts/run_goal2_splits.py --build-standard-choice-donors

python3 -u - <<'PY'
from pathlib import Path
import os
import numpy as np
p = Path(os.environ['ONE_CACHE_DIR'], 'manifold', 'choice_donors_standard.npy')
if not p.exists():
    raise SystemExit(f'Missing standard donor bank: {p}')
bank = np.load(p, allow_pickle=True).item()
if not bank:
    raise SystemExit(f'Empty standard donor bank: {p}')
n_stim = n_move = 0
for eid, rec in bank.items():
    views = rec.get('views') if isinstance(rec, dict) else None
    if not isinstance(rec, dict) or not rec.get('standard_act') or not views:
        raise SystemExit(f'{eid}: standard donor record has no views')
    for key, view in views.items():
        if key not in ('stim', 'move'):
            raise SystemExit(f'{eid}: unexpected donor view {key}')
        if 'act_binary' not in view or 'choice' not in view:
            raise SystemExit(f'{eid} {key}: missing act_binary')
        if len(view['act_binary']) != len(view['choice']):
            raise SystemExit(f'{eid} {key}: length mismatch')
        if key == 'stim':
            n_stim += 1
        else:
            n_move += 1
if n_stim == 0 or n_move == 0:
    raise SystemExit(
        f'Standard donor bank missing a view: stim={n_stim} move={n_move}')
print(f'Standard donor bank OK: {len(bank)} eids '
      f'(stim={n_stim}, move={n_move}) -> {p}')
PY

echo "Standard donors done: $(date)"
ls -lh "$ONE_CACHE_DIR/manifold/choice_donors_standard.npy"
