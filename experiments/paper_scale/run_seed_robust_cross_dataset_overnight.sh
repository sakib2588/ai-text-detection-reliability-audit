#!/usr/bin/env bash
# One-night chain: retrain seed 123/456 checkpoints (GPU, ~2-2.5h) then run the
# 3-seed cross-dataset eval (CPU, ~2h for the 8 new cells; seed-42 cells skip
# instantly since they're already on disk). Stops the whole chain on the first
# failure so a broken retrain never silently feeds a stale/missing checkpoint
# into the eval stage.
set -e
cd "$(dirname "$0")"

PY="/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Project /.venv/bin/python"
mkdir -p logs
LOG="logs/seed_robust_cross_$(date +%Y%m%d_%H%M%S).log"

echo "=== stage 1/2: retrain_seeds_for_cross.py (GPU) ===" | tee -a "$LOG"
"$PY" retrain_seeds_for_cross.py >> "$LOG" 2>&1

echo "=== stage 2/2: cross_dataset_eval_multiseed.py (CPU) ===" | tee -a "$LOG"
"$PY" cross_dataset_eval_multiseed.py >> "$LOG" 2>&1

echo "=== CHAIN COMPLETE ===" | tee -a "$LOG"
