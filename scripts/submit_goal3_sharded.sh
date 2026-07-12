#!/bin/bash
# Submit sharded Goal 3 contrast-split jobs (same insertion-sharding path as Goal 2).
#
# Each contrast-conditioned split gets N_SHARDS parallel shard jobs writing
#   manifold/res/_stream_acc/{split}.shard{k}.npy
# then a finalize job merges → manifold/res/{split}*.npy.
#
# Reuses Goal 2 workers:
#   scripts/run_goal2_shard_slurm.sh
#   scripts/run_goal2_finalize_slurm.sh
#
# Assumes insertion cache already exists (run_goal2_cache_slurm.sh done).
#
#   bash scripts/submit_goal3_sharded.sh
#   PRESET=goal3_duringstim_act CONTRASTS="0.0 0.125 1.0" N_SHARDS=4 \
#     bash scripts/submit_goal3_sharded.sh
#   PRESET=goal3_all N_SHARDS=6 bash scripts/submit_goal3_sharded.sh
#
# Memory (override with MEM_SHARD / MEM_FIN):
#   Peak RSS stream_pool nrand=2000 ≈ 1.5–2.5 GB (journal 07-10b). Goal 3
#   contrast + min_trials_per_side=5 skips many insertions → smaller stream_acc.
#   Defaults: MEM_SHARD=8G, MEM_FIN=12G (was 48G/32G via worker #SBATCH).
#   Concurrent request: n_splits × N_SHARDS × 8G  (e.g. 20×4×8 = 640G vs 3.8T).
#
# Defaults: PRESET=goal3_duringstim_act, all CONTRASTS, N_SHARDS=4.
# Job count ≈ n_splits × (N_SHARDS + 1 finalize). Example:
#   goal3_duringstim_act = 4 bases × 5 contrasts = 20 splits → 80 shard + 20 fin.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

PRESET="${PRESET:-goal3_duringstim_act}"
N_SHARDS="${N_SHARDS:-4}"
NRAND="${NRAND:-2000}"
RESTART="${RESTART:-1}"
MEM_SHARD="${MEM_SHARD:-8G}"
MEM_FIN="${MEM_FIN:-12G}"
CPUS_SHARD="${CPUS_SHARD:-2}"
CPUS_FIN="${CPUS_FIN:-2}"

# Expand preset → split names without importing block_analysis_allsplits (no ONE).
SPLITS=()
while IFS= read -r line; do
  [[ -n "$line" ]] && SPLITS+=("$line")
done < <(PRESET="$PRESET" CONTRASTS="${CONTRASTS:-}" python3 - <<'PY'
import os
CONTRASTS_DEFAULT = [1.0, 0.25, 0.125, 0.0625, 0.0]
DURINGSTIM = [
    'block_duringstim_r_choice_r_f1',
    'block_duringstim_l_choice_l_f1',
    'block_duringstim_l_choice_r_f2',
    'block_duringstim_r_choice_l_f2',
    'act_block_duringstim_r_choice_r_f1',
    'act_block_duringstim_l_choice_l_f1',
    'act_block_duringstim_l_choice_r_f2',
    'act_block_duringstim_r_choice_l_f2',
]
DURINGCHOICE = [
    'block_stim_r_duringchoice_r_f1',
    'block_stim_l_duringchoice_l_f1',
    'block_stim_l_duringchoice_r_f2',
    'block_stim_r_duringchoice_l_f2',
    'act_block_stim_r_duringchoice_r_f1',
    'act_block_stim_l_duringchoice_l_f1',
    'act_block_stim_l_duringchoice_r_f2',
    'act_block_stim_r_duringchoice_l_f2',
]
PRESETS = {
    'goal3_duringstim': DURINGSTIM,
    'goal3_duringchoice': DURINGCHOICE,
    'goal3_duringstim_act': [s for s in DURINGSTIM if s.startswith('act_')],
    'goal3_duringstim_block': [s for s in DURINGSTIM if not s.startswith('act_')],
    'goal3_duringchoice_act': [s for s in DURINGCHOICE if s.startswith('act_')],
    'goal3_duringchoice_block': [s for s in DURINGCHOICE if not s.startswith('act_')],
    'goal3_all': DURINGSTIM + DURINGCHOICE,
}
preset = os.environ['PRESET']
if preset not in PRESETS:
    raise SystemExit(f'Unknown PRESET={preset}; choose from {sorted(PRESETS)}')
raw = os.environ.get('CONTRASTS', '').strip()
contrasts = [float(x) for x in raw.split()] if raw else CONTRASTS_DEFAULT
for base in PRESETS[preset]:
    for c in contrasts:
        print(f'{base}_{float(c)}')
PY
)

if [[ ${#SPLITS[@]} -eq 0 ]]; then
  echo "ERROR: no splits resolved for preset=$PRESET contrasts=${CONTRASTS:-all}" >&2
  exit 1
fi

n_shard_jobs=$(( ${#SPLITS[@]} * N_SHARDS ))
echo "PRESET=$PRESET  N_SHARDS=$N_SHARDS  nrand=$NRAND  splits=${#SPLITS[@]}"
echo "contrasts: ${CONTRASTS:-all (default CONTRASTS)}"
echo "MEM_SHARD=$MEM_SHARD  MEM_FIN=$MEM_FIN  CPUS_SHARD=$CPUS_SHARD  CPUS_FIN=$CPUS_FIN"
echo "Concurrent mem if all shards run: ${n_shard_jobs} × $MEM_SHARD"

job_tag() {
  # Slurm job-name safe: dots → p, truncate
  local s="$1"
  s="${s//./p}"
  echo "${s:0:48}"
}

for sp in "${SPLITS[@]}"; do
  TAG=$(job_tag "$sp")
  SHARD_JOBS=()
  for ((k=0; k<N_SHARDS; k++)); do
    JID=$(sbatch --parsable \
      --mem="$MEM_SHARD" --cpus-per-task="$CPUS_SHARD" \
      --job-name="g3_${TAG}_s${k}" \
      --export=ALL,SPLIT="$sp",SHARD_IDX="$k",N_SHARDS="$N_SHARDS",NRAND="$NRAND",RESTART="$RESTART" \
      scripts/run_goal2_shard_slurm.sh)
    SHARD_JOBS+=("$JID")
    echo "  $sp shard $k/$N_SHARDS -> $JID"
  done
  DEP=$(IFS=:; echo "${SHARD_JOBS[*]}")
  FID=$(sbatch --parsable \
    --mem="$MEM_FIN" --cpus-per-task="$CPUS_FIN" \
    --dependency=afterok:"$DEP" \
    --job-name="g3_fin_${TAG}" \
    --export=ALL,SPLIT="$sp" \
    scripts/run_goal2_finalize_slurm.sh)
  echo "  $sp finalize -> $FID (after $DEP)"
done

echo "Done. Monitor: squeue -u \$USER"
echo "Shard outputs: \$ONE_CACHE_DIR/manifold/res/_stream_acc/{split}.shard{k}.npy"
echo "Final outputs: \$ONE_CACHE_DIR/manifold/res/{split}.npy"
