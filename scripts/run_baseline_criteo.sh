#!/usr/bin/env bash
# Reproduce paper Table 2 Criteo column (50k subsample). Add --include-full for Table 7 (full table, slow).
# Run from the repo root on the GPU server. Safe to re-run: finished folds are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

export OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
mkdir -p "$OUTPUT_DIR/logs"
log="$OUTPUT_DIR/logs/baseline_criteo_$(date +%Y%m%d_%H%M%S).log"

python -m baseline.run --dataset criteo "$@" 2>&1 | tee "$log"
python -m baseline.evaluate --dataset criteo 2>&1 | tee -a "$log"
echo "log: $log"
