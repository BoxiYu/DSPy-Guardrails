#!/bin/bash
# Check experiment progress
# Usage: bash experiments/check_progress.sh

cd /Users/boxi.yu/Documents/VAG/dspyGuardrails

echo "============================================================"
echo "EXPERIMENT PROGRESS CHECK — $(date)"
echo "============================================================"

echo ""
echo "=== Running Processes ==="
ps aux | grep -E "(cross_eval_evolved|run_coevo_ablation)" | grep -v grep | awk '{printf "  PID=%s  CPU=%s  MEM=%s  CMD=%s\n", $2, $3, $4, $11" "$12" "$13" "$14" "$15}' || echo "  (none running)"

echo ""
echo "=== EXP-A: Cross-Evaluation Logs ==="
for seed in 42 123 456; do
    f="experiments/logs/cross_eval_seed${seed}.log"
    if [ -f "$f" ]; then
        sz=$(wc -c < "$f" | tr -d ' ')
        last=$(tail -1 "$f" 2>/dev/null)
        echo "  seed=$seed: ${sz} bytes — last: $last"
    else
        echo "  seed=$seed: NOT STARTED"
    fi
done

echo ""
echo "=== EXP-B: Ablation Logs ==="
for seed in 42 123 456; do
    f="experiments/logs/ablation_seed${seed}.log"
    if [ -f "$f" ]; then
        sz=$(wc -c < "$f" | tr -d ' ')
        last=$(tail -1 "$f" 2>/dev/null)
        echo "  seed=$seed: ${sz} bytes — last: $last"
    else
        echo "  seed=$seed: NOT STARTED"
    fi
done

echo ""
echo "=== Result Files ==="
for f in experiments/cross_eval_results_seed*.json experiments/ablation_results_seed*.json; do
    if [ -f "$f" ]; then
        sz=$(wc -c < "$f" | tr -d ' ')
        echo "  $(basename $f): ${sz} bytes ✓"
    fi
done

total_expected=6
found=$(ls experiments/cross_eval_results_seed*.json experiments/ablation_results_seed*.json 2>/dev/null | wc -l | tr -d ' ')
echo ""
echo "  Results: $found / $total_expected complete"

if [ "$found" -eq "$total_expected" ]; then
    echo ""
    echo "=== ALL EXPERIMENTS COMPLETE ==="
    echo "Run analysis:"
    echo "  source venv311/bin/activate"
    echo "  python experiments/analysis/aggregate_cross_eval.py --results-dir experiments/ --latex"
fi
