#!/usr/bin/env bash
# Reproduce paper Table 2, Hillstrom columns: CausalPFN + X/S/DA/T learners, 5 folds, both arms.
# Run from the repo root on the GPU server. Safe to re-run: finished folds are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

export OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
mkdir -p "$OUTPUT_DIR/logs"
log="$OUTPUT_DIR/logs/baseline_hillstrom_$(date +%Y%m%d_%H%M%S).log"

python -m baseline.run --dataset hillstrom "$@" 2>&1 | tee "$log"
python -m baseline.evaluate --dataset hillstrom 2>&1 | tee -a "$log"
echo "log: $log"
