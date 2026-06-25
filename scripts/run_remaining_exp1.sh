#!/bin/bash
# Run remaining EXP1 experiments (TAP + MAP-Elites for all defenses + full run for missing defenses)
# Run this AFTER EXP3 completes (to avoid API contention)

set -e

cd /Users/miracy/Documents/VAG/dspyGuardrails
source /Users/miracy/Documents/VAG/venv311/bin/activate

LOG=/tmp/exp1_remaining.log
echo "=== EXP1 Remaining Experiments ===" | tee $LOG
echo "Started: $(date)" | tee -a $LOG

# Step 1: Run dspy_simba + dspy_gepa full evaluation (DIRECT + all attacks)
# dspy_simba has NO data at all, dspy_gepa is missing harmful DIRECT
echo "" | tee -a $LOG
echo "--- Step 1: dspy_simba + dspy_gepa full evaluation ---" | tee -a $LOG
python scripts/run_ase_experiments.py exp1 \
    --seed 42 --goals 10 \
    --defenses dspy_simba dspy_gepa \
    --verbose 2>&1 | tee -a $LOG

# Step 2: Run TAP + MAP-Elites for all other defenses (attacks-only, skip DIRECT)
echo "" | tee -a $LOG
echo "--- Step 2: TAP + MAP-Elites for 10 defenses ---" | tee -a $LOG
python scripts/run_ase_experiments.py exp1 \
    --seed 42 --goals 10 \
    --defenses no_defense spotlighting sandwiching protectai promptguard piguard llamaguard dspy_unopt dspy_bfs dspy_mipro \
    --attacks tap mapelites \
    --attacks-only \
    --verbose 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "=== EXP1 Remaining Complete ===" | tee -a $LOG
echo "Finished: $(date)" | tee -a $LOG
