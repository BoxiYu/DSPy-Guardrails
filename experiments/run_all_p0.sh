#!/bin/bash
# ASE 2026 P0 Experiment Runner — Run all priority-0 experiments
# Expected runtime: 2-4 hours. Expected cost: ~$15-20.
# Usage: bash experiments/run_all_p0.sh [--goals N] [--seed S]

set -e

GOALS=${1:-20}
SEED=${2:-42}
SCRIPT="scripts/run_ase_experiments.py"
RESULTS="experiments/results"

echo "============================================================"
echo "ASE 2026 P0 Experiments"
echo "Goals: $GOALS  Seed: $SEED"
echo "Results: $RESULTS"
echo "Start: $(date)"
echo "============================================================"

# Activate venv
source ../venv311/bin/activate 2>/dev/null || source venv311/bin/activate 2>/dev/null || true

# EXP1: Defense Effectiveness (PAIR)
echo ""
echo ">>> EXP1: Defense Effectiveness (PAIR attack)"
echo ">>> Phase 1a: Prompt-based defenses..."
python $SCRIPT exp1 --seed $SEED --goals $GOALS \
    --defenses no_defense spotlighting sandwiching \
    --attacks pair --verbose 2>&1 | tee -a $RESULTS/exp1_log.txt

echo ">>> Phase 1b: LLM-simulated filters..."
python $SCRIPT exp1 --seed $SEED --goals $GOALS \
    --defenses protectai promptguard piguard llamaguard \
    --attacks pair --verbose 2>&1 | tee -a $RESULTS/exp1_log.txt

echo ">>> Phase 1c: DSPy variants..."
python $SCRIPT exp1 --seed $SEED --goals $GOALS \
    --defenses dspy_unopt dspy_bfs dspy_mipro dspy_simba dspy_gepa \
    --attacks pair --verbose 2>&1 | tee -a $RESULTS/exp1_log.txt

# EXP1: TAP attack (all 12 defenses)
echo ""
echo ">>> EXP1: Defense Effectiveness (TAP attack)"
python $SCRIPT exp1 --seed $SEED --goals $GOALS \
    --defenses no_defense spotlighting sandwiching protectai promptguard piguard llamaguard dspy_unopt dspy_bfs dspy_mipro dspy_simba dspy_gepa \
    --attacks tap --verbose 2>&1 | tee -a $RESULTS/exp1_log.txt

# EXP1: MAP-Elites (key defenses only to save cost)
echo ""
echo ">>> EXP1: Defense Effectiveness (MAP-Elites, key defenses)"
python $SCRIPT exp1 --seed $SEED --goals $GOALS \
    --defenses no_defense piguard llamaguard dspy_unopt dspy_gepa \
    --attacks mapelites --verbose 2>&1 | tee -a $RESULTS/exp1_log.txt

# EXP3: Attack Comparison + Ablation
echo ""
echo ">>> EXP3: Attack Comparison + Ablation"
python $SCRIPT exp3 --seed $SEED --goals $GOALS \
    --verbose 2>&1 | tee -a $RESULTS/exp3_log.txt

# EXP4: Co-Evolution Dynamics
echo ""
echo ">>> EXP4: Co-Evolution Dynamics"
python $SCRIPT exp4 --seed $SEED --goals $GOALS \
    --verbose 2>&1 | tee -a $RESULTS/exp4_log.txt

echo ""
echo "============================================================"
echo "All P0 experiments complete!"
echo "End: $(date)"
echo "============================================================"

# Run analysis
echo ""
echo ">>> Running analysis..."
python experiments/analysis/analyze_results.py \
    --results-dir $RESULTS --latex --exp all

echo ""
echo ">>> Filling tables..."
python experiments/analysis/fill_tables.py \
    --results-dir $RESULTS
