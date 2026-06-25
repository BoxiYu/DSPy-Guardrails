#!/bin/bash
# Fill all experiment gaps for ASE 2026 paper
# Run from dspyGuardrails/ directory
set -euo pipefail

cd /Users/miracy/Documents/VAG/dspyGuardrails
source /Users/miracy/Documents/VAG/venv311/bin/activate

LOG=/tmp/fill_gaps.log
echo "=== Fill Gaps Started: $(date) ===" | tee "$LOG"

# ---------------------------------------------------------------
# Gap 1 (CRITICAL): Direct classification + PAIR for 10 defenses
# Gives Table 1 F1/OR columns + completes PAIR column
# ~300 records: 10 def × (10 benign_direct + 10 harmful_direct + 10 PAIR)
# ---------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== Gap 1: Direct + PAIR for 10 defenses (seed 42) ===" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"

python scripts/run_ase_experiments.py exp1 --seed 42 --goals 10 \
  --defenses no_defense spotlighting sandwiching protectai promptguard piguard llamaguard dspy_unopt dspy_bfs dspy_mipro \
  --attacks pair \
  --verbose 2>&1 | tee -a "$LOG"

echo "Gap 1 done: $(date)" | tee -a "$LOG"

# ---------------------------------------------------------------
# Gap 2: Multi-seed 123 — 4 key defenses (direct + PAIR)
# Paper claims "repeated with three seeds" for GEPA, BFS, PIGuard, LlamaGuard
# ~120 records: 4 def × (10 benign_direct + 10 harmful_direct + 10 PAIR)
# ---------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== Gap 2: Seed 123 (4 key defenses) ===" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"

python scripts/run_ase_experiments.py exp1 --seed 123 --goals 10 \
  --defenses dspy_gepa dspy_bfs piguard llamaguard \
  --attacks pair \
  --verbose 2>&1 | tee -a "$LOG"

echo "Gap 2 done: $(date)" | tee -a "$LOG"

# ---------------------------------------------------------------
# Gap 3: Multi-seed 456 — 4 key defenses (direct + PAIR)
# ~120 records
# ---------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== Gap 3: Seed 456 (4 key defenses) ===" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"

python scripts/run_ase_experiments.py exp1 --seed 456 --goals 10 \
  --defenses dspy_gepa dspy_bfs piguard llamaguard \
  --attacks pair \
  --verbose 2>&1 | tee -a "$LOG"

echo "Gap 3 done: $(date)" | tee -a "$LOG"

# ---------------------------------------------------------------
# Done
# ---------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== All Gaps Filled: $(date) ===" | tee -a "$LOG"
echo "Results in: experiments/results/"
