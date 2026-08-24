#!/bin/bash
# Thin wrapper: Bayes f1 only (factor 6). Prefer the full analog:
#   bash scripts/submit_goal2_ak_sticky_bayes_orcd.sh
#   PARTITION=mit_normal bash scripts/submit_goal2_ak_sticky_bayes_f1_orcd.sh
set -euo pipefail
FAMILY=f1 exec bash scripts/submit_goal2_ak_sticky_bayes_orcd.sh
