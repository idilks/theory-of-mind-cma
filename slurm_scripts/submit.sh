#!/bin/bash
# Usage: ./slurm_scripts/submit.sh <slurm_script> [extra sbatch args...]
# Prepends a YYYYMMDD_HHMM timestamp to the log filenames so logs sort
# chronologically by name, independent of SLURM's job-ID ordering.

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 <slurm_script> [extra sbatch args...]" >&2
  exit 1
fi

SCRIPT="$1"
shift

TS=$(date +%Y%m%d_%H%M)
BASE=$(basename "$SCRIPT" .sh)

mkdir -p logs

sbatch \
  --output="logs/${TS}_${BASE}_%A_%a.out" \
  --error="logs/${TS}_${BASE}_%A_%a.err" \
  "$@" \
  "$SCRIPT"
