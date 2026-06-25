# ASE 2026 Experiment Runbook

## Current Status
- Split manifest: frozen (seed=42, 60/20/20)
- StrongREJECT judge: fixed (rubric-based, threshold=0.5)
- RealResponseTarget: implemented (real LLM responses for scoring)

## Priority Order

### P0 — Required for submission

**Step 1: EXP1 Core (PAIR attack, style-free primary matrix)**
```bash
cd /Users/miracy/Documents/VAG/dspyGuardrails
source ../venv311/bin/activate

# Phase 1a: Prompt heuristics + no defense (fast, ~5min)
python scripts/run_ase_experiments.py exp1 \
  --seed 42 --goals 20 \
  --defenses no_defense spotlighting sandwiching \
  --attacks pair --verbose

# Phase 1b: Real model/classifier baselines (depends on local model setup)
python scripts/run_ase_experiments.py exp1 \
  --seed 42 --goals 20 \
  --defenses llamaguard_local shieldgemma_local protectai_real promptguard_real piguard_real \
  --attacks pair --verbose

# Phase 1c: DSPy variants (longest, due to compilation)
python scripts/run_ase_experiments.py exp1 \
  --seed 42 --goals 20 \
  --defenses dspy_unopt dspy_bfs dspy_mipro dspy_simba dspy_gepa dspy_v3_unopt dspy_v3_bfs dspy_v3_mipro dspy_v3_simba dspy_v3_gepa \
  --attacks pair --verbose
```

**Step 2: EXP1 TAP attack (same style-free primary matrix)**
```bash
python scripts/run_ase_experiments.py exp1 \
  --seed 42 --goals 20 \
  --defenses no_defense spotlighting sandwiching llamaguard_local shieldgemma_local protectai_real promptguard_real piguard_real dspy_unopt dspy_bfs dspy_mipro dspy_simba dspy_gepa dspy_v3_unopt dspy_v3_bfs dspy_v3_mipro dspy_v3_simba dspy_v3_gepa \
  --attacks tap --verbose
```

**Step 3: EXP1 MAP-Elites (key defenses)**
```bash
python scripts/run_ase_experiments.py exp1 \
  --seed 42 --goals 20 \
  --defenses no_defense spotlighting sandwiching llamaguard_local piguard_real dspy_unopt dspy_gepa dspy_v3_gepa \
  --attacks mapelites --verbose
```

**Step 4: EXP3 Ablation (PAIR with/without feedback)**
```bash
python scripts/run_ase_experiments.py exp3 \
  --seed 42 --goals 20 --verbose
```

**Step 5: EXP4 Co-Evolution**
```bash
python scripts/run_ase_experiments.py exp4 \
  --seed 42 --goals 20 --verbose
```

### P0.5 — Multi-seed stability (key defenses)

```bash
for seed in 123 456; do
  python scripts/run_ase_experiments.py exp1 \
    --seed $seed --goals 20 \
    --defenses piguard_real llamaguard_local dspy_bfs dspy_gepa dspy_v3_gepa \
    --attacks pair --verbose
done
```

### P1 — Strongly recommended

**Step 6: EXP2 Optimizer Comparison**
```bash
python scripts/run_ase_experiments.py exp2 \
  --seed 42 --goals 20 --verbose
```

**Step 7: EXP5-A Transfer (HarmBench)**
```bash
python scripts/run_ase_experiments.py exp5 \
  --seed 42 --goals 20 --verbose
```

### P2 — Enhancement (if budget allows)

**Step 8: EXP5-B Predictor sensitivity**
```bash
python scripts/run_ase_experiments.py exp5 \
  --seed 42 --goals 20 --predictor-check --verbose
```

## Analysis

```bash
python experiments/analysis/analyze_results.py \
  --results-dir experiments/results/ --latex --exp all
```

## Cost Estimates (approximate)

| Experiment | API Calls | Est. Cost |
|---|---|---|
| EXP1 PAIR (18 def × 20 goals) | ~7200 | ~$8 |
| EXP1 TAP (18 def × 20 goals) | ~10800 | ~$12 |
| EXP1 ME (8 def × 20 goals) | ~6400 | ~$24 |
| EXP3 Ablation (4 cond × 20 goals) | ~1600 | ~$2 |
| EXP4 Co-evolution (10 rounds) | ~3000 | ~$4 |
| Multi-seed (5 def × 2 seeds) | ~4000 | ~$5 |
| EXP2 (4 optimizers) | ~2000 | ~$3 |
| **Total P0+P0.5** | **~33000** | **~$55** |
| **Total all** | **~39000** | **~$66** |
