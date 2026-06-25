# Related Work: Evolution-Based Attack/Defense Methods (2025-2026)

## Attack Evolution

### EvoSynth (arXiv 2511.12710, Nov 2025)
- **Core idea**: Evolve attack METHODS (code), not prompts — multi-agent system
- **85.5% ASR** against Claude Sonnet 4.5
- Code-level self-correction loop — rewrites attack logic on failure
- **Key distinction from our work**: They evolve the full attack code; we could adopt their self-correction loop idea

### ASTRA (arXiv 2511.02356, Nov 2025)
- **Core idea**: Closed-loop attack-evaluate-distill-reuse
- Three-tier strategy library: Effective / Promising / Ineffective
- **82.7% ASR** black-box
- Extracts reusable strategies from failures
- **Key distinction**: Strategy library with knowledge transfer — our AGENT_LOG does something similar

### LLM-Virus (arXiv 2501.00055, Jan 2025)
- Evolutionary operators applied to jailbreak prompts (biological virus evolution metaphor)
- Uses LLM as the evolutionary operator (mutation, crossover)

### EvoJail (arXiv 2603.20122, Mar 2026)
- Multi-objective evolutionary search on long-tail inputs
- LLM-assisted mutation and crossover operators

## Defense Evolution

### DuoGuard
- Two-player RL: generator produces challenging queries for classifier
- Co-evolution converges to Nash equilibrium
- **Directly relevant**: co-evolution of attack and defense

### AdaptiveGuard
- Out-of-distribution detection via Mahalanobis distance
- LoRA-based continual updates post-deployment
- Adapts to new jailbreak attacks without full retraining

### AGrail (ACL 2025)
- Lifelong agent guardrail with effective adaptation
- Learns from deployment experience

## Key Takeaways for Our Paper

1. **EvoSynth is our closest competitor** — they also evolve attack code, not prompts
   - Our advantage: we also evolve DEFENSES (bidirectional)
   - Our advantage: we study guard PLACEMENT (input/output/both)

2. **ASTRA's strategy library** = our AGENT_LOG — validate our design choice

3. **DuoGuard's co-evolution** is the defense-side equivalent of what we're doing

4. **Our unique contribution remains**: guard placement ablation + bidirectional evolution
   No one has studied how guard POSITION affects evolutionary attack/defense dynamics
