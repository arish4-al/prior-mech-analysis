#!/bin/bash
# Fig 4.5 stim / choice / act-prior sensitivity chain on ORCD:
#   stage A shards -> stack A + select (afterok) -> stage B shards -> stack B + compare
#
#   bash scripts/submit_fig45_sensitivity.sh
#
# Needs: data/fig45/cell_types.npy in the repo (fig45_define_cell_types.py save,
# with stimint / rm22-rm28 / integrator / move_all), manifold/insertion_cache
# and bwm_tables/trials.pqt under ONE_CACHE_DIR.
#
# Override: N_SHARDS=8 MEM_SHARD=8G PARTITION=mit_preemptable Q=0.05 \
#   bash scripts/submit_fig45_sensitivity.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
OUT="${OUT:-$ONE_CACHE_DIR/dmn/res/fig45/cvTrue/sensitivity}"
CELL_TYPES="${CELL_TYPES:-$REPO_DIR/data/fig45/cell_types.npy}"
N_SHARDS="${N_SHARDS:-8}"
N_SHARDS_B="${N_SHARDS_B:-4}"
MEM_SHARD="${MEM_SHARD:-8G}"
MEM_STACK="${MEM_STACK:-16G}"
CPUS_SHARD="${CPUS_SHARD:-2}"
NRAND="${NRAND:-2000}"
PRIOR="${PRIOR:-act}"
Q="${Q:-0.05}"
PARTITION="${PARTITION:-pi_fiete}"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/sbatch_defaults.sh"

[[ -f "$CELL_TYPES" ]] || { echo "missing $CELL_TYPES" >&2; exit 1; }
[[ -f "$ONE_CACHE_DIR/bwm_tables/trials.pqt" ]] || {
  echo "missing $ONE_CACHE_DIR/bwm_tables/trials.pqt" >&2; exit 1; }

COMMON="ALL,ONE_CACHE_DIR=$ONE_CACHE_DIR,OUT=$OUT,CELL_TYPES=$CELL_TYPES,NRAND=$NRAND,PRIOR=$PRIOR,Q=$Q"
echo "PARTITION=$PARTITION N_SHARDS=$N_SHARDS/$N_SHARDS_B NRAND=$NRAND PRIOR=$PRIOR Q=$Q OUT=$OUT"

submit_shards() {  # $1 stage, $2 n_shards, $3 dependency ("" for none)
  local ids=() dep_arg=()
  [[ -n "$3" ]] && dep_arg=(--dependency=afterok:"$3")
  for ((k=0; k<$2; k++)); do
    # shellcheck disable=SC2086
    ids+=("$(sbatch --parsable $SBATCH_EXTRA --partition="$PARTITION" \
      --mem="$MEM_SHARD" --cpus-per-task="$CPUS_SHARD" "${dep_arg[@]}" \
      --job-name="f45s_$1_s${k}" \
      --export="$COMMON,STEP=compute,STAGE=$1,SHARD_IDX=$k,N_SHARDS=$2" \
      scripts/run_fig45_sensitivity_slurm.sh)")
  done
  (IFS=:; echo "${ids[*]}")
}

submit_one() {  # $1 job name, $2 dependency, $3 export extras
  # shellcheck disable=SC2086
  sbatch --parsable $SBATCH_EXTRA --partition="$PARTITION" \
    --mem="$MEM_STACK" --cpus-per-task=2 --dependency=afterok:"$2" \
    --job-name="$1" --export="$COMMON,$3" scripts/run_fig45_sensitivity_slurm.sh
}

A=$(submit_shards A "$N_SHARDS" "")
echo "stage A shards: $A"
SA=$(submit_one f45s_stackA "$A" "STEP=stack,STAGE=A")
SEL=$(submit_one f45s_select "$SA" "STEP=select")
echo "stack A: $SA  select: $SEL"
B=$(submit_shards B "$N_SHARDS_B" "$SEL")
echo "stage B shards: $B"
SB=$(submit_one f45s_stackB "$B" "STEP=stack,STAGE=B")
echo "stack B + compare: $SB"
