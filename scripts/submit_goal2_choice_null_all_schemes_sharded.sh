#!/bin/bash
# Submit all three choice L–R structured null schemes (journal 2026-07-23b).
#
#   bash scripts/submit_goal2_choice_null_all_schemes_sharded.sh
#
# Equivalent to three sequential submitter calls:
#   NULL_SCHEME=pseudo_strat  (opt 1)
#   NULL_SCHEME=pseudo_fixed  (opt 2)
#   NULL_SCHEME=harris_unique (opt 3 unique-null; alias: harris)
#
# Override shared knobs as usual (PRESET, N_SHARDS, NRAND, MEM_*, SMOKE_FIRST).
# Schemes can be subset with SCHEMES="pseudo_strat harris_unique".

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/prior-mech-analysis}"
cd "$REPO_DIR"

PARTITION="${PARTITION:-pi_fiete}"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/sbatch_defaults.sh"

SCHEMES="${SCHEMES:-pseudo_strat pseudo_fixed harris_unique}"

echo "Submitting schemes: $SCHEMES"
for sch in $SCHEMES; do
  echo "======== NULL_SCHEME=$sch ========"
  NULL_SCHEME="$sch" bash scripts/submit_goal2_choice_null_sharded.sh
done
echo "All scheme submitters launched. Monitor: squeue -u \$USER"
