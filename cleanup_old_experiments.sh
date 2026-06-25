#!/bin/bash
# DSPyGuard Cleanup Script
# Removes old experiment results and unused scripts
# that are no longer needed for the new DSPyGuard experiment plan.
#
# Run from the dspyGuardrails/ directory:
#   chmod +x cleanup_old_experiments.sh && ./cleanup_old_experiments.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "=== DSPyGuard Cleanup ==="
echo "Working directory: $SCRIPT_DIR"
echo ""

# --- 1. Old experiment results ---
echo "--- Deleting old experiment results ---"

# Checkpoints (old adversarial training results)
rm -rfv "$SCRIPT_DIR/checkpoints/"
echo ""

# --- 2. Old experiment scripts ---
echo "--- Deleting old experiment directories ---"

# Pilot experiments (old design, pre-JBB)
rm -rfv "$SCRIPT_DIR/experiments/pilot/"

# Old optimizer comparison (hardcoded 25-sample data, pre-JBB)
rm -rfv "$SCRIPT_DIR/experiments/optimizer_comparison/"

# Old shared experiment settings
rm -rfv "$SCRIPT_DIR/experiments/common/"
echo ""

# --- 3. Old scripts (all designed for previous experiment plan) ---
echo "--- Deleting old experiment scripts ---"

# Keep: model_config.py (still needed)
# Delete all run_*.py, benchmark_*.py, manual_*.py
for f in \
    "$SCRIPT_DIR/scripts/run_adaptive_attack_experiments.py" \
    "$SCRIPT_DIR/scripts/run_best_vs_best.py" \
    "$SCRIPT_DIR/scripts/run_copro_gepa_defense.py" \
    "$SCRIPT_DIR/scripts/run_cross_model_attacks.py" \
    "$SCRIPT_DIR/scripts/run_cross_model_coopt.py" \
    "$SCRIPT_DIR/scripts/run_dspy_attack_optimization.py" \
    "$SCRIPT_DIR/scripts/run_dspy_cooptimization.py" \
    "$SCRIPT_DIR/scripts/run_dspy_defense_optimization.py" \
    "$SCRIPT_DIR/scripts/run_llm_defense_experiment.py" \
    "$SCRIPT_DIR/scripts/run_mapelites_attack.py" \
    "$SCRIPT_DIR/scripts/run_multi_optimizer_defense.py" \
    "$SCRIPT_DIR/scripts/run_pair_tap_experiments.py" \
    "$SCRIPT_DIR/scripts/run_paper_payload_stress.py" \
    "$SCRIPT_DIR/scripts/run_self_evolving_experiment.py" \
    "$SCRIPT_DIR/scripts/run_validation.py" \
    "$SCRIPT_DIR/scripts/benchmark_four_way.py" \
    "$SCRIPT_DIR/scripts/benchmark_frameworks.py" \
    "$SCRIPT_DIR/scripts/manual_shield_checks.py" \
; do
    [ -f "$f" ] && rm -v "$f"
done

# Clean up pycache
rm -rf "$SCRIPT_DIR/scripts/__pycache__/"
echo ""

# --- 4. Old paper_research docs (superseded) ---
echo "--- Deleting superseded design docs ---"
rm -fv "$SCRIPT_DIR/paper_research/UPDATED_EXPERIMENT_DESIGN.md"
rm -fv "$SCRIPT_DIR/paper_research/SHIELD_IMPROVEMENT_PLAN.md"
echo ""

# --- 5. Clean caches ---
echo "--- Cleaning caches ---"
rm -rf "$SCRIPT_DIR/.mypy_cache/"
rm -rf "$SCRIPT_DIR/.pytest_cache/"
rm -rf "$SCRIPT_DIR/.ruff_cache/"
rm -rf "$SCRIPT_DIR/dist/"
find "$SCRIPT_DIR" -maxdepth 3 -name "__pycache__" -type d \
    ! -path "*/.venv/*" ! -path "*/.venv_py314/*" \
    -exec rm -rf {} + 2>/dev/null || true
echo ""

# --- Summary ---
echo "=== Cleanup Complete ==="
echo ""
echo "KEPT:"
echo "  scripts/model_config.py    (model registry, still needed)"
echo "  paper_research/references.bib"
echo "  paper_research/EXPERIMENT_DESIGN_v4.md"
echo "  paper_research/BASELINE_GUARDRAILS_RESEARCH.md"
echo "  paper_research/PENTEST_METHODS_INVENTORY.md"
echo "  src/                       (core library)"
echo "  tests/                     (test suite)"
echo "  docs/                      (documentation)"
echo "  configs/                   (config files)"
echo "  examples/                  (examples)"
echo ""
echo "DELETED:"
echo "  checkpoints/               (old results)"
echo "  experiments/pilot/         (old pilot scripts)"
echo "  experiments/optimizer_comparison/"
echo "  experiments/common/"
echo "  scripts/run_*.py           (16 old experiment scripts)"
echo "  scripts/benchmark_*.py    (2 old benchmark scripts)"
echo "  scripts/manual_*.py       (1 old manual test)"
echo "  paper_research/UPDATED_EXPERIMENT_DESIGN.md"
echo "  paper_research/SHIELD_IMPROVEMENT_PLAN.md"
echo "  .mypy_cache/ .pytest_cache/ .ruff_cache/ dist/"
echo ""
echo "Ready for fresh DSPyGuard experiments!"
