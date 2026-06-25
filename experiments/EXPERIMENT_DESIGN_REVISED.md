# EXP1 Revised Experiment Design — Fair Optimization Comparison

## Problem Statement

The original EXP1 design confounds **base model** and **optimization method**:

```
LlamaGuard (specialized model + no optimization)
  vs
DSPy-CoEvo (general LLM + CoEvo optimization)
```

This makes it impossible to attribute improvements to CoEvo vs the base model.

## Revised Design

### Core Principle

**CoEvo is an optimization method, not a defense model.**
The experiment must show CoEvo's value by comparing optimizers on the SAME base model.

### Experiment Matrix

| Base Model (LM) | Unopt | BFS | MIPROv2 | CoEvo |
|------------------|-------|-----|---------|-------|
| DeepSeek V3.2 (API) | ✅ done | ✅ done | ✅ done | ✅ done |
| GPT-4o-mini (API) | ❌ need | ❌ need | ❌ need | ❌ need |

Plus external reference baselines:
- LlamaGuard 3 8B **native** (original prompt template, no DSPy) ✅ done
- ShieldGemma 2B **native** ✅ done

### Why GPT-4o-mini (not LlamaGuard)

**LlamaGuard 3 8B cannot be used as a DSPy LM backend** (confirmed 2026-03-01):
- LlamaGuard is fine-tuned to output `safe`/`unsafe\nSX` — it ignores DSPy prompts
- Cannot produce structured output (reasoning, is_unsafe, confidence, reason)
- DSPy adapter parse fails: `Expected [reasoning, is_unsafe, ...], got []`
- Since DSPy optimizers work by modifying prompts, LlamaGuard is immune to optimization
- LlamaGuard stays as an external reference baseline (native mode)

**GPT-4o-mini** is the ideal second base model:
1. Different model family (OpenAI vs DeepSeek) — shows cross-family generalization
2. Instruction-following — all DSPy optimizers work correctly
3. Low cost ($0.15/1M input, $0.60/1M output via OpenRouter)
4. Partial cross-eval data already exists (CoEvo 1.1% vs MIPROv2 21.4% evolved ASR)

### Why Two Base Models

1. **DeepSeek V3.2**: General-purpose LLM, strong reasoning, API-based (proprietary)
2. **GPT-4o-mini**: OpenAI instruction-tuned, cost-effective, API-based

This answers TWO questions:
- **Q1**: Is CoEvo the best DSPy optimizer? (compare optimizers within each row)
- **Q2**: Does CoEvo generalize across base models? (compare CoEvo across rows)

### Evaluation Domain

Two evaluation domains (same as before):
- **STD**: JBB-100 test split (20 harmful + 20 benign) — fixed across all experiments
- **EVO**: Matched-N sampled evolved attacks (20 attacks + 20 benign)

For GPT-4o-mini CoEvo: evolved attacks are co-evolved AGAINST GPT-4o-mini, not loaded
from DeepSeek runs. This ensures CoEvo targets GPT-4o-mini's actual weaknesses.

For GPT-4o-mini BFS/MIPROv2/Unopt: load evolved attacks from GPT-4o-mini CoEvo run
(same as DeepSeek experiments loaded from DeepSeek CoEvo).

## Execution Plan

### Phase 1: GPT-4o-mini CoEvo (full co-evolution, 3 seeds)

```bash
# Run full EXP1 with GPT-4o-mini (all 4 DSPy optimizers in one go)
python scripts/run_exp1_smoke.py \
    --seed 42 \
    --model gpt-4o-mini \
    --skip-static \
    --output-dir experiments/exp1_gpt4omini_seed42

# Repeat for seeds 123, 456
```

Each run compiles Unopt + BFS + MIPROv2 + CoEvo, then evaluates all on STD + EVO.

### Phase 2: Merge Results

```bash
python scripts/merge_exp1_results.py  # Updated to include GPT-4o-mini
```

## Directory Structure

```
experiments/
├── exp1_results_seed42_v7/          # DeepSeek: CoEvo + Unopt (done)
├── exp1_results_seed123_v7/         # DeepSeek: CoEvo + Unopt (done)
├── exp1_results_seed456_v7/         # DeepSeek: CoEvo + Unopt (done)
├── exp1_baselines_seed42/           # DeepSeek: BFS + MIPROv2 + static (done)
├── exp1_baselines_seed123/          # DeepSeek: BFS + MIPROv2 + static (done)
├── exp1_baselines_seed456/          # DeepSeek: BFS + MIPROv2 + static (done)
├── exp1_gpt4omini_seed42/           # GPT-4o-mini: all 4 optimizers (new)
├── exp1_gpt4omini_seed123/          # GPT-4o-mini: all 4 optimizers (new)
├── exp1_gpt4omini_seed456/          # GPT-4o-mini: all 4 optimizers (new)
├── exp1_merged_results.json         # DeepSeek-only merged (done)
└── exp1_full_merged_results.json    # Full merged: DeepSeek + GPT-4o-mini (new)
```

## Reproducibility Checklist

- [x] All random seeds fixed (42, 123, 456) via `random.seed()` + `dspy` seed
- [x] JBB-100 split deterministic (seed-based)
- [x] Model versions locked: DeepSeek V3.2 (via OpenRouter), GPT-4o-mini (via OpenRouter)
- [x] `cache=False` for all LMs (no DSPy cache replay)
- [ ] Each result dir has: `exp1_results.json`, `coevo_compile_log.json`,
      `evolved_attacks_raw.json`, `matched_n_manifest.json`, `split_manifest.json`
- [ ] Per-defense per-domain detail files: `{Defense}_{domain}_results.json`
- [ ] Merge script updated to include GPT-4o-mini directories

## Expected Paper Tables

### Table 1: Optimization Comparison on DeepSeek V3.2 (3-seed avg)

| Optimizer | STD ASR↓ | STD F1↑ | EVO ASR↓ | EVO F1↑ |
|-----------|----------|---------|----------|---------|
| Unopt     | 66.7%    | 0.484   | 56.7%    | 0.598   |
| BFS       | 30.0%    | 0.796   | 35.0%    | 0.778   |
| MIPROv2   | 26.7%    | 0.826   | 41.7%    | 0.718   |
| **CoEvo** | 28.3%    | 0.734   | **25.0%**| **0.785**|

### Table 2: Optimization Comparison on GPT-4o-mini (3-seed avg)

| Optimizer | STD ASR↓ | STD F1↑ | EVO ASR↓ | EVO F1↑ |
|-----------|----------|---------|----------|---------|
| Unopt     | ?        | ?       | ?        | ?       |
| BFS       | ?        | ?       | ?        | ?       |
| MIPROv2   | ?        | ?       | ?        | ?       |
| CoEvo     | ?        | ?       | ?        | ?       |

### Table 3: External Reference Baselines (3-seed avg)

| Defense | STD ASR↓ | STD F1↑ | EVO ASR↓ | EVO F1↑ |
|---------|----------|---------|----------|---------|
| LlamaGuard 3 native | 3.3% | 0.899 | 33.3% | 0.721 |
| ShieldGemma 2B native | 75.0% | 0.387 | 80.0% | 0.323 |

## Known Risks

1. **LlamaGuard DSPy incompatibility** (CONFIRMED): LlamaGuard 3 8B cannot produce
   structured DSPy output. Fine-tuned classifier outputs only "safe"/"unsafe\nSX".
   Resolution: Use GPT-4o-mini as second base model instead.

2. **GPT-4o-mini safety alignment**: GPT-4o-mini has built-in safety alignment which
   may interact with DSPy safety prompts. This is a valid finding about how CoEvo
   interacts with pre-aligned models.

3. **Seed 123 instability**: CoEvo failed on seed 123 for DeepSeek. May also fail
   for GPT-4o-mini. Document as limitation.

4. **Cost**: GPT-4o-mini is cheap but CoEvo (5R×15A = 75 attack gen + eval cycles)
   plus MIPROv2 (13 trials) will accumulate API costs. Estimated ~$2-5 per seed.
