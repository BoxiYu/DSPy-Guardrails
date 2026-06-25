#!/bin/bash
# Run all remaining experiments for ASE 2026 paper
# Run this AFTER EXP3 completes (to avoid API contention)

cd /Users/miracy/Documents/VAG/dspyGuardrails
source /Users/miracy/Documents/VAG/venv311/bin/activate

FAILURES=0
LOG=/tmp/remaining_experiments.log
echo "=== Remaining ASE Experiments ===" | tee $LOG
echo "Started: $(date)" | tee -a $LOG

run_experiment() {
    local name="$1"
    shift
    echo "" | tee -a $LOG
    echo "--- $name ---" | tee -a $LOG
    echo "Started $name: $(date)" | tee -a $LOG
    if "$@" 2>&1 | tee -a $LOG; then
        echo "$name: SUCCESS ($(date))" | tee -a $LOG
    else
        echo "$name: FAILED ($(date))" | tee -a $LOG
        FAILURES=$((FAILURES + 1))
    fi
}

# ==================================================================
# EXP1 Part 1: dspy_simba + dspy_gepa full evaluation
# dspy_simba has NO data at all, dspy_gepa is missing harmful DIRECT
# ==================================================================
run_experiment "EXP1 Part 1: dspy_simba + dspy_gepa" \
    python scripts/run_ase_experiments.py exp1 \
    --seed 42 --goals 10 \
    --defenses dspy_simba dspy_gepa \
    --verbose

# ==================================================================
# EXP1 Part 2: TAP + MAP-Elites for remaining 10 defenses
# These already have DIRECT + PAIR data, only need TAP + ME
# ==================================================================
run_experiment "EXP1 Part 2: TAP + MAP-Elites for 10 defenses" \
    python scripts/run_ase_experiments.py exp1 \
    --seed 42 --goals 10 \
    --defenses no_defense spotlighting sandwiching protectai promptguard piguard llamaguard dspy_unopt dspy_bfs dspy_mipro \
    --attacks tap mapelites \
    --attacks-only \
    --verbose

# ==================================================================
# EXP2: Optimizer comparison (BFS, MIPROv2, SIMBA, GEPA)
# Needed for Table 2 and Figure 3 (optimizer robustness-cost tradeoff)
# ==================================================================
run_experiment "EXP2: Optimizer Comparison" \
    python scripts/run_ase_experiments.py exp2 \
    --seed 42 --goals 10 \
    --verbose

# ==================================================================
# EXP4: Re-run with more test goals (full test set)
# Previous run used only 3 test goals, both regimes show ASR=100%
# ==================================================================
run_experiment "EXP4: Co-evolution full test set" \
    python scripts/run_ase_experiments.py exp4 \
    --seed 42 \
    --verbose

# ==================================================================
# EXP5: Supplementary experiments (transfer, predictor sensitivity)
# Depends on EXP1/EXP4 completing for cost metadata derivation
# ==================================================================
run_experiment "EXP5: Transfer + Predictor Sensitivity" \
    python scripts/run_ase_experiments.py exp5 \
    --seed 42 --goals 10 \
    --verbose

# ==================================================================
# Analysis: Generate tables and figures from all results
# ==================================================================
echo "" | tee -a $LOG
echo "--- Analysis: Fill tables and generate figures ---" | tee -a $LOG
python experiments/analysis/fill_tables.py \
    --results-dir experiments/results/ \
    --tex-file ../dspyGuardASE/dspyGuardrails.tex \
    --write 2>&1 | tee -a $LOG

python experiments/analysis/generate_figures.py \
    --results-dir experiments/results/ \
    --output-dir ../dspyGuardASE/figures/ 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
if [ $FAILURES -gt 0 ]; then
    echo "=== WARNING: $FAILURES experiment(s) failed ===" | tee -a $LOG
else
    echo "=== All Remaining Experiments Complete ===" | tee -a $LOG
fi
echo "Finished: $(date)" | tee -a $LOG
