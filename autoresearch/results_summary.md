# Autoresearch Complete Results (as of 2026-04-01)

## Experimental Setup
- **Attacker LM**: Qwen3.5-27B-abliterated (local vLLM, 2×RTX 5090)
- **Target LMs**: Qwen3-235B MoE (primary), DeepSeek V3.2 (cross-validation, running)
- **Judge**: StrongREJECT via Qwen3-235B (OpenRouter)
- **Benchmark**: JBB-100, 20 behaviors per seed
- **Seeds**: 42, 123, 456
- **Query budget**: 10 per behavior

## Table 1: Guard Mode Ablation (Qwen3-235B Target, 3-seed mean)

| Guard Mode | ASR↓ | StrongREJECT | Δ vs None |
|-----------|------|-------------|-----------|
| None | 0.667 ± 0.076 | 0.640 ± 0.049 | — |
| Input (pattern) | 0.650 ± 0.087 | 0.606 ± 0.074 | -2.5% |
| **Output (LLM)** | **0.517 ± 0.104** | **0.487 ± 0.102** | **-22.5%** |
| **Input + Output** | **0.433 ± 0.189** | **0.406 ± 0.191** | **-35.0%** |

## Table 2: Attack Algorithm Discovery (8 iterations)

| Ver | Algorithm | Guard | ASR | Q | Key Insight |
|-----|-----------|-------|-----|---|-------------|
| **v0** | **PAIR baseline** | output | **0.400** | 6 | Robust optimal |
| v1 | Plan-Attack-Debrief | none | 0.667 | 3 | Overhead hurts |
| v2 | Stealth PAIR | output | 0.200 | 6 | Framing weakens attack |
| v3 | Guard-Aware PAIR | output | 0.400 | 6 | Guard feedback = no gain |
| v4 | Crowding Attack | output | 0.000 | 18 | Fails for jailbreak |
| v5 | Multi-Candidate PAIR | output | 0.400 | 15 | 3x cost, same result |
| v6 | Progressive Escalation | both | 0.300 | 12 | Slightly worse |
| v7 | Evolved PAIR (meta-opt) | output | 0.400 | 18 | Refiner already optimal |
| v8 | Strategy Pool (UCB) | output | 0.100 | 10 | Fixed strategies hurt |

## Table 3: Defense Guard Design Comparison

| Defense | ASR (PAIR v0) | Note |
|---------|-------------|------|
| Pattern output guard | 0.800 | Ineffective vs adaptive attacks |
| **StrongREJECT rubric** | **0.517** | **Best single guard** |
| d1 Basic LLM | 0.550 | Slightly weaker |
| d2 Enhanced CoT | 0.800 | Worst — CoT paradox |
| d3 Ensemble (3-guard vote) | 0.500 | ≈ StrongREJECT |

## Table 4: Per-Category ASR (seed 42)

| Category | None | Output | Both |
|----------|------|--------|------|
| extremism | 1.00 | 1.00 | 1.00 |
| chemical_biological | 1.00 | 0.00 | 1.00 |
| cybercrime | 1.00 | 0.33 | 0.00 |
| fraud | 0.50 | 0.50 | 0.50 |
| harassment | 0.33 | 0.67 | 0.33 |
| illegal_drugs | 1.00 | 0.00 | 0.00 |
| misinformation | 1.00 | 0.33 | 0.33 |
| privacy | 1.00 | 0.00 | 0.00 |
| violence | 0.33 | 0.67 | 0.33 |

## Key Findings

### F1: Output guard > Input guard for jailbreak defense
Pattern input guards provide minimal protection (-2.5%). LLM output guards
reduce ASR by 22.5%. Combined guards achieve 35% reduction. This fills the
gap in "Attacker Moves Second" which only tested input-side defenses.

### F2: PAIR is robust optimal — 8/8 novel approaches fail to beat it
LLM-driven iterative refinement is the strongest black-box attack strategy.
Adding structure (plans, strategies, templates) constrains the search and
hurts performance. The lesson: let the LLM be unconstrained.

### F3: CoT reasoning weakens guards (paradox)
Chain-of-thought guard (d2, ASR=0.800) is WORSE than simple classification
(d1, ASR=0.550). Reasoning enables self-persuasion: the guard talks itself
into "this is educational, not harmful". Structured scoring (StrongREJECT)
is more reliable than open-ended reasoning.

### F4: Behavior category determines defensibility
Extremism content bypasses all guards (ASR=1.0). Privacy/drugs content is
fully blocked by output guard (ASR=0.0). Social engineering is inconsistent
across seeds — behavior sampling significantly affects results.

### F5: Evaluation rubric = best defense
StrongREJECT (designed as an evaluation metric) is the most effective output
guard. The metric that measures attack success is also the best tool for
preventing it. This suggests a unification of evaluation and defense.

## Statistics
- Total experiments: 29+
- Attack iterations: 9 (v0-v8)
- Defense iterations: 3 (d1-d3)
- Git commits: 50+
- API cost: ~$20
- Cross-validation: 12 DeepSeek V3.2 experiments (running)
