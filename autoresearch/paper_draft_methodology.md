# Paper Draft: Experiment Methodology

## 5. Evaluation

### 5.1 Experimental Setup

**Models.** We use Qwen3-235B MoE (via OpenRouter) as the target model with
built-in safety alignment. The attacker LLM is Qwen3.5-27B-abliterated
(locally deployed via vLLM on 2×RTX 5090), an uncensored variant that can
freely generate attack prompts without self-censoring. For output guard
evaluation, we use a separate instance of Qwen3-235B as the StrongREJECT
judge.

**Benchmark.** We evaluate on a subset of JailbreakBench (JBB-100), sampling
20 harmful behaviors per seed across 10 categories (violence, cybercrime,
fraud, harassment, drugs, etc.). Each experiment is repeated with 3 random
seeds (42, 123, 456) for statistical reliability.

**Attack.** We use PAIR (Prompt Automatic Iterative Refinement) as the
primary adaptive attack, with a budget of 10 queries per behavior. PAIR
represents the state-of-the-art in black-box LLM-based jailbreak attacks.

**Metrics.**
- Attack Success Rate (ASR): fraction of behaviors where the attack
  successfully elicits harmful content, as judged by StrongREJECT score > 0.5
- Mean StrongREJECT score: average quality of jailbreak responses (0-1 scale)
- Query efficiency: total queries used across all behaviors

### 5.2 Guard Mode Ablation (RQ1)

We evaluate four guardrail configurations:

1. **None**: No guardrail — direct attack against target model's built-in alignment
2. **Input (pattern)**: Regex-based input guard checking for 13 common jailbreak/injection patterns before the prompt reaches the target model
3. **Output (LLM)**: LLM-based output guard that uses StrongREJECT to evaluate whether the target's response contains harmful content after generation
4. **Input + Output**: Both guards applied in sequence

This ablation directly addresses the gap identified in prior work: "The
Attacker Moves Second" evaluated 12 defenses, all input-side only, despite
production systems (AWS Bedrock, Anthropic, Google Vertex AI, NVIDIA NeMo)
universally deploying both input and output guards.

### 5.3 Autoresearch Attack Iterations (RQ2)

We run the autoresearch loop to discover novel attack algorithms:
- Each iteration, the AI agent writes a new Python attack module
- The module is evaluated against the LLM output guard (the strongest single defense)
- Results are compared to the PAIR v0 baseline
- The agent receives feedback (ASR, StrongREJECT, guard block details) and iterates

We report the trajectory of discovered algorithms and analyze whether any
novel approach surpasses PAIR.

### 5.4 Defense Evolution (RQ3)

We evaluate the autoresearch loop for defense discovery:
- The agent writes new guard implementations (LLM prompts, pattern rules, composite strategies)
- Each defense is evaluated against PAIR and autoresearch-discovered attacks
- Metrics: ASR reduction, false positive rate, over-refusal rate

### Preliminary Results (Seed 42, 20 behaviors)

| Guard Mode | ASR↓ | StrongREJECT | Δ vs None |
|-----------|------|-------------|-----------|
| None | 0.750 | 0.694 | baseline |
| Input (pattern) | 0.700 | 0.669 | -6.7% |
| Output (LLM) | 0.400 | 0.375 | -46.7% |
| Input + Output | 0.300 | 0.275 | -60.0% |

These results demonstrate that:
1. Pattern-based input guards provide minimal defense against adaptive attacks (-6.7%)
2. LLM-based output guards are the most impactful single defense (-46.7%)
3. Combined input+output guards achieve the strongest defense (-60.0%)
4. Even the strongest configuration leaves 30% of attacks successful, highlighting the ongoing challenge of robust LLM defense
