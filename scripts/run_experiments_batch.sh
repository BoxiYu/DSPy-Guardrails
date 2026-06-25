#!/usr/bin/env bash
# Batch runner for ASE 2026 experiments.
# Run sequentially to avoid API rate limit conflicts.
#
# Usage:
#   chmod +x scripts/run_experiments_batch.sh
#   nohup scripts/run_experiments_batch.sh > /tmp/experiment_batch.log 2>&1 &
#
# Stages (comment/uncomment as needed):
#   Stage 1: EXP1 TAP + MAP-Elites (all 12 defenses) — P0
#   Stage 2: EXP3 (attack comparison + ablation) — P0
#   Stage 3: EXP4 (co-evolution dynamics) — P0
#   Stage 4: EXP2 (optimizer comparison) — P1
#   Stage 5: EXP5 (supplementary: transfer + predictor sensitivity) — P1

set -e

cd /Users/miracy/Documents/VAG/dspyGuardrails
source /Users/miracy/Documents/VAG/venv311/bin/activate

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/tmp/ase_experiments"
mkdir -p "$LOG_DIR"

echo "=== ASE 2026 Experiment Batch Started: $(date) ==="

# --- Stage 1: EXP1 TAP + MAP-Elites (all 12 defenses) ---
# Run both attacks together so each defense is compiled only once.
echo ""
echo "=== Stage 1: EXP1 TAP + MAP-Elites (all 12 defenses) ==="
echo "Started: $(date)"
python scripts/run_ase_experiments.py exp1 \
    --seed 42 --goals 10 \
    --attacks tap mapelites \
    --mapelites-gen 100 \
    --verbose \
    2>&1 | tee "$LOG_DIR/exp1_tap_me_${TIMESTAMP}.log"
echo "Stage 1 completed: $(date)"

# --- Stage 2: EXP3 (attack comparison + ablation, against BFS) ---
echo ""
echo "=== Stage 2: EXP3 Attack Comparison + Ablation ==="
echo "Started: $(date)"
python scripts/run_ase_experiments.py exp3 \
    --seed 42 --goals 10 \
    --mapelites-gen 100 \
    --verbose \
    2>&1 | tee "$LOG_DIR/exp3_${TIMESTAMP}.log"
echo "Stage 3 completed: $(date)"

# --- Stage 3: EXP4 (co-evolution dynamics) ---
echo ""
echo "=== Stage 3: EXP4 Co-Evolution Dynamics ==="
echo "Started: $(date)"
python scripts/run_ase_experiments.py exp4 \
    --seed 42 --goals 10 \
    --verbose \
    2>&1 | tee "$LOG_DIR/exp4_${TIMESTAMP}.log"
echo "Stage 4 completed: $(date)"

# --- Stage 4: EXP2 (optimizer comparison) ---
echo ""
echo "=== Stage 4: EXP2 Optimizer Comparison ==="
echo "Started: $(date)"
python scripts/run_ase_experiments.py exp2 \
    --seed 42 --goals 10 \
    --verbose \
    2>&1 | tee "$LOG_DIR/exp2_${TIMESTAMP}.log"
echo "Stage 5 completed: $(date)"

# --- Stage 5: EXP5 (supplementary: transfer + predictor sensitivity) ---
echo ""
echo "=== Stage 5: EXP5 Supplementary (P1 - Optional) ==="
echo "Started: $(date)"
python scripts/run_ase_experiments.py exp5 \
    --seed 42 --goals 10 \
    --verbose \
    2>&1 | tee "$LOG_DIR/exp5_${TIMESTAMP}.log"
echo "Stage 5 completed: $(date)"

echo ""
echo "=== All Stages Complete: $(date) ==="
echo "Results in: experiments/results/"
echo "Logs in: $LOG_DIR/"
