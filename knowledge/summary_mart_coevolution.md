# MART: Improving LLM Safety with Multi-round Automatic Red-Teaming

**Source:** arXiv 2311.07689

## 1. Paper Metadata

- **Title:** MART: Improving LLM Safety with Multi-round Automatic Red-Teaming
- **Authors:** Suyu Ge, Chunting Zhou, Rui Hou, Madian Khabsa, Yi-Chia Wang, Qifan Wang, Jiawei Han, Yuning Mao (GenAI, Meta; University of Illinois Urbana-Champaign)
- **Date:** November 2023 (submitted to ACL-style venue)
- **Base model:** LLaMA-65B

## 2. Core Contribution

MART is the first framework to close the loop on automatic red-teaming by jointly training both an adversarial LLM (attack generator) and a target LLM (defender) in an iterative co-evolution cycle. Unlike prior automatic red-teaming work that only discovers safety risks, MART also addresses them by using successful attacks to fine-tune the target model on safe responses. Starting from only ~2,400 seed adversarial prompts (no human annotation of responses), MART reduces violation rates by up to 84.7% (RM) / 53.7% (human) over 4 rounds, approaching ChatGPT-level safety.

## 3. Key Methods

### Multi-Round Adversarial Loop

The framework alternates between two phases per round:

**Phase 1 -- Attack Generation:**
1. The adversarial model M_adv is prompted with successful attacks from the previous round (P^{i-1}_adv).
2. It generates new adversarial prompts P^i_gen that are stylistically similar but novel.
3. The adversarial model uses a **pairwise training scheme**: given one malicious prompt as input, it outputs a similar prompt from the same violation category and attack style.

**Phase 2 -- Defense Training:**
1. The target model M_tgt generates responses A^i_tgt for the new adversarial prompts P^i_gen.
2. A **safety reward model** S^s and a **helpfulness reward model** S^h score each (prompt, response) pair.
3. **Data selection** (the critical step):
   - If safety score s^s < theta_adv: the prompt is marked as a **successful attack** and added to P^i_adv (used to train the attacker next round).
   - If safety score s^s > theta_tgt AND helpfulness score s^h > theta_tgt: the response is added to R^i_tgt (used to fine-tune the defender via SFT).
   - Pairs that are neither clearly unsafe nor clearly safe+helpful are discarded.
4. Both models are updated via supervised fine-tuning on their respective selected data.

### Attack Training Protocol

- **Initialization:** Both M_adv and M_tgt start from the same LLaMA-65B model fine-tuned on LIMA + Open Assistant (instruction tuning).
- **Adversarial seed:** ~2,400 manually curated prompts across violation categories (crimes, hateful content, unqualified advice) and attack styles (psychological manipulation, role-playing, misspelling, etc.). Split 1,700 train / 700 eval.
- **Adversarial model training:** Pairwise (input_prompt, output_prompt) pairs sampled from same (category, style). The model learns to generate prompts *similar to* successful attacks.
- **1-shot prompting** works better than 3-shot (more diversity, more efficient data usage).

### Defense Training Protocol

- **Self-supervised:** The target model is fine-tuned on its own high-quality responses (filtered by dual reward models), not on externally written safe responses.
- **Context distillation** (round 1 only): Prefixes prompts with a safety preprompt to bootstrap safer initial responses when the model is still weak.
- **Rejection sampling** (final round): Samples K responses per prompt at varying temperatures to enlarge the candidate pool when the model has converged.
- **No additional helpfulness data needed:** Helpfulness is maintained by quality-filtering with the helpfulness RM, not by mixing in extra helpfulness training data.

### Key Design Choices

- Two separate reward models (safety and helpfulness) rather than a single combined one.
- Optimal thresholds: safety=0.8, helpfulness=0.4 for target model data selection.
- The adversarial model is also mixed with instruction-tuning seed data each round to maintain conversational ability.
- Stopping criterion: when adversarial generation violation rate falls near 10%, insufficient training data can be collected for the next round.

## 4. Results

### Violation Rate Reduction (SafeEval)

| Stage | SafeEval | Anthropic Harmless | Adversarial Gen |
|-------|----------|--------------------|-----------------|
| Vanilla (no safety) | 31.4% | 26.7% | --- |
| Iter 1 | 15.4% | 15.1% | 29.7% |
| Iter 2 | 10.1% | 10.0% | 31.9% |
| Iter 3 | 5.9% | 7.3% | 20.3% |
| Iter 4 (MART) | 4.8% | 6.9% | 10.2% |

- **84.7% relative reduction** in RM-evaluated violation rate (31.4% -> 4.8%) over 4 rounds.
- **53.7% relative reduction** in human-evaluated violation rate (17.2% -> 8.0%).
- Helpfulness on non-adversarial prompts (HelpEval, AlpacaEval) remains stable -- only 3-4% decrease over all iterations.
- Generalizes to out-of-domain benchmarks (Anthropic Harmless).

### Comparison to Baselines

| Method | SafeEval (Human) | Anthropic Harmless (Human) |
|--------|-----------------|---------------------------|
| Vanilla | 17.2% | 12.1% |
| MART | 8.0% | 4.9% |
| ChatGPT | 5.7% | 2.0% |
| GPT-4 | 5.6% | 1.9% |
| Llama 2-Chat-70B | 4.2% | 1.6% |

MART approaches but does not match models with extensive manual red-teaming (Llama 2-Chat used 350+ human annotators, 14 batches). The gap suggests room for hybrid human+automatic approaches.

### Adversarial Method Comparison

- MART-1shot outperforms GCG (gradient-based), few-shot prompting, and MART-3shot at sustaining attack effectiveness across rounds.
- GCG triggers the most harmful responses in round 1 but its effectiveness decays rapidly after safety fine-tuning (suffix-based attacks are easy to learn to bypass).
- Few-shot prompting without weight updates shows the weakest attack and minimal safety improvement, validating the need for adversarial model optimization.

## 5. Relevance to DSPyGuardrails

### Architectural Parallels

MART's co-evolutionary loop maps directly onto DSPyGuardrails' `AdversarialTrainer`:

| MART Component | DSPyGuardrails Equivalent |
|---------------|--------------------------|
| M_adv (adversarial LLM) | `AttackEvolver` with 12 mutation strategies |
| M_tgt (target LLM) | `EvolvableShieldTarget` / `EvolvableLLMTarget` |
| Safety + Helpfulness RMs | Safety RM scoring in the co-evolution loop |
| Iterative training loop | `AdversarialTrainer` closed-loop rounds |
| Successful attack selection | `DefenseEvolver` pattern extraction from successful attacks |
| Pairwise adversarial training | Attack mutation strategies (synonym, encoding, context wrap, etc.) |

### Key Differences

1. **Weight-level vs. prompt-level adaptation:** MART fine-tunes model weights via SFT each round. DSPyGuardrails operates at the prompt/few-shot level via DSPy optimization (BootstrapFewShot, etc.), which is much cheaper but potentially less powerful.

2. **Reward model-driven selection vs. binary pass/fail:** MART uses continuous reward model scores with tunable thresholds for nuanced data selection. DSPyGuardrails' co-evolution loop appears to use binary attack success/failure signals.

3. **Self-supervised defense improvement:** MART trains the defender on its *own* best responses (filtered by quality). DSPyGuardrails' `DefenseEvolver` extracts patterns and generates few-shot examples from successful attacks -- a different (and complementary) mechanism.

4. **Dual reward models:** MART explicitly separates safety and helpfulness scoring to avoid the over-conservatism trap. This is a critical design insight missing from most red-teaming frameworks.

5. **Attack diversity:** MART uses category+style structured seed data and pairwise training. DSPyGuardrails has richer mutation operators (12 strategies including cipher, ASCII art, deep inception) but may lack the structured category/style coverage.

### What MART Does Better

- **Principled data selection** with dual-threshold filtering ensures training data quality.
- **Demonstrated convergence** over exactly 4 rounds with clear stopping criteria.
- **Helpfulness preservation** without extra helpfulness data -- purely through quality filtering.
- **Quantified diminishing returns** -- knows when to stop (adversarial gen violation rate near 10%).

### What DSPyGuardrails Does Better

- **No fine-tuning required** -- works with frozen LLM APIs via prompt engineering.
- **Richer attack mutation space** (12 strategies vs. pairwise generation).
- **Cross-model attacks** (PAIR, TAP, MAP-Elites) that MART doesn't explore.
- **Pattern-based fast path** that doesn't need any LLM calls.

## 6. Specific Ideas to Incorporate

### 6.1 Dual-Objective Scoring for Data Selection

Add a **helpfulness/utility score** alongside the safety score in `AdversarialTrainer`. When selecting which defense examples to learn from, require both safety AND helpfulness thresholds to be met. This prevents the defense from becoming over-conservative (the "I can't help with that" failure mode).

**Implementation:** In `DefenseEvolver`, when harvesting successful defense responses, score them on both safety (did it block the attack?) and utility (is the response still helpful for legitimate variants of the query?). Only promote examples that pass both.

### 6.2 Structured Attack Coverage Matrix

Adopt MART's category x style grid for systematic attack coverage. Currently `AttackEvolver` has 12 mutation strategies but doesn't systematically ensure coverage across violation categories.

**Implementation:** Define a coverage matrix of (threat_category, mutation_strategy) pairs. Track which cells have been tested and which have successful attacks. Prioritize under-explored cells in the next evolution round. This directly maps to the existing `redteam/payloads/` structure.

### 6.3 Reward-Model-Gated Training Data Selection

Replace binary pass/fail with continuous scoring and threshold-based selection in the co-evolution loop.

**Implementation:** Instead of just checking `TargetResponse.was_blocked`, compute a continuous safety score (e.g., via `injection_score()` or LLM-based scoring) and only include examples above/below configurable thresholds. Discard ambiguous middle-ground examples that would add noise.

### 6.4 Context Distillation for Bootstrap Round

In the first round of co-evolution when the defense is weakest, use MART's context distillation technique: prefix defense prompts with a strong safety system prompt to generate higher-quality seed defense examples.

**Implementation:** In `DefenseEvolver`, for round 1 only, prepend a safety-focused system prompt when generating defense responses. Use these bootstrapped responses as initial few-shot examples for DSPy optimization.

### 6.5 Rejection Sampling for Final Rounds

When the co-evolution loop converges (attack success rate drops below ~10%), switch to rejection sampling: generate K responses per adversarial prompt at varying temperatures and select the best.

**Implementation:** Add a convergence detection mechanism to `AdversarialTrainer`. When ASR drops below a threshold, trigger a rejection sampling phase that generates multiple candidate defenses and selects the highest-scoring ones for the final defense configuration.

### 6.6 Adaptive Stopping Criteria

MART stops at 4 rounds based on diminishing returns (adversarial gen violation rate near 10%). Add similar convergence detection to `AdversarialTrainer`.

**Implementation:** Track the delta in ASR between rounds. If delta < epsilon for N consecutive rounds, stop the loop. This prevents wasted compute on rounds that yield minimal improvement. Currently the loop runs for a fixed number of rounds.

### 6.7 Attack Model Optimization via Success Filtering

MART's key insight for the attacker: only train on (input_prompt, output_prompt) pairs where the output was actually a successful attack. Apply this to `AttackEvolver`.

**Implementation:** When evolving attacks, weight or filter mutations based on whether they actually succeeded against the current defense. Currently the 12 mutation strategies are applied somewhat uniformly. Track per-strategy success rates and allocate more evolution budget to strategies that are currently working (similar to bandit-based strategy selection, but simpler).
