# Slurm submit defaults. Source after PARTITION is set (and after cd "$REPO_DIR").
#
#   # shellcheck disable=SC1091
#   source "$REPO_DIR/scripts/sbatch_defaults.sh"
#   sbatch --parsable $SBATCH_EXTRA --partition="$PARTITION" ...
#
# If PARTITION is mit_preemptable or mit_preem and SBATCH_EXTRA is unset,
# defaults SBATCH_EXTRA=--requeue. Explicit SBATCH_EXTRA (including "") wins.
# Requeued jobs keep stream_acc; RESTART=1 skips insertions already written.
# CLEAR_STREAM only runs in the submit wrapper, not on Slurm requeue.
#
# New submit_*.sh scripts must source this and pass $SBATCH_EXTRA to every sbatch.

if [[ -z "${SBATCH_EXTRA+x}" ]]; then
  case "${PARTITION:-}" in
    mit_preemptable|mit_preem) SBATCH_EXTRA="--requeue" ;;
    *) SBATCH_EXTRA="" ;;
  esac
fi
export SBATCH_EXTRA
