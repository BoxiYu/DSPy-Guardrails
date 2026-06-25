# Summary: Jailbreaking to Jailbreak (J2)

## 1. Paper Metadata

- **Title**: Jailbreaking to Jailbreak
- **Authors**: Jeremy Kritz, Vaughn Robinson, Robert Vacareanu, Bijan Varjavand, Michael Choi, Bobby Gogov, Scale Red Team, Summer Yue, Willow E. Primack, Zifan Wang (Scale AI)
- **ArXiv ID**: 2502.09638
- **Date**: February 2025 (preprint)
- **Contact**: zifan.wang@scale.com
- **Website**: https://scale.com/research/j2

## 2. Core Contribution

The paper introduces J2 (jailbreaking-to-jailbreak), a method that turns any safety-trained frontier LLM into an autonomous red teamer by first jailbreaking it to be *willing* to assist in jailbreaking other models. Unlike prior work that requires uncensored or open-weight models as attackers, J2 unlocks the full reasoning capabilities of frontier models (Claude, GPT, Gemini) for red teaming. A single jailbreak prompt crafted on Sonnet-3.5 transfers effectively to nearly all other frontier models, making J2 attacker creation a one-time effort.

## 3. Method

### Step 1: Making J2 Attackers

A human red teamer crafts a multi-turn conversation (X_human) that convinces the target LLM that jailbreaking research benefits safety more than it causes harm. This is followed by X_info, several turns that introduce broad jailbreaking guidance (e.g., using euphemistic language, thinking about refusal as a single vector direction, maintaining cover throughout). The key distinction is that the model is *knowingly* convinced to help with jailbreaking rather than being tricked via fiction or roleplay, which allows subsequent instructions to be straightforward.

The jailbreak prompt was developed on `claude-3-5-sonnet-1022` and transfers without modification to: Sonnet-3.7, GPT-4o, GPT-4.1, o3, o3-mini, o4-mini, Gemini-1.5-pro, Gemini-2.0-flash, Gemini-2.0-pro, Gemini-2.5-pro. It failed on `claude-3-5-sonnet-0620` (too strong refusal) and DeepSeek-R1 (misunderstands the role).

### Step 2: Red Teaming Workflow (Plan-Attack-Debrief Cycles)

The workflow has three phases per cycle:

1. **Planning**: The J2 attacker receives a target harmful behavior B and a strategy S. It plans its approach (chain-of-thought scratchpad). If it "regrets" and tries to refuse, hard-coded refusal-handling prompts re-engage it.

2. **Attack**: The J2 attacker conducts a T-turn multi-turn conversation with the target model. The attacker's outputs become the target's user messages and vice versa. T varies by model (T=2 for o3, T=3 for Gemini, T=6 for Claude/GPT-4.1).

3. **Debrief**: A GPT-4o judge evaluates success with a checklist-based prompt. The feedback is shared with the attacker, which reflects and plans improvements.

**Cycling for in-context learning**: Failed cycles remain in the context window so the attacker learns from failures. Up to N=10 cycles per strategy. If all cycles fail, the context is reset and a new strategy is tried.

### Strategies (9 total, human-curated)

1. **Dealer's choice** -- attacker picks its own strategy
2. **Crowding** -- overwhelm filters with dense legitimate context, embed target as a tiny detail
3. **Technical** -- frame everything as code, logs, debugging exercises, API documentation
4. **Zoom in** -- start with massive simulation, zoom layer by layer to target content
5. **Forum** -- simulate an online forum with multiple users, target content emerges from discussion
6. **Extra fiction** -- narrative framing (novel, video game, movie scenes)
7. **Rule inversion** -- establish innocent rules, then systematically invert them
8. **Antagonist** -- model plays a villain character
9. **Output format** -- request specific output style/format

### Judging

- **Workflow judge**: GPT-4o with improved HarmBench prompt (for cycle feedback)
- **Final ASR judge**: o3 with ActorAttack-style prompt (prevents reward hacking by the attacker)

## 4. Cross-Model Results

### Self-Attack (J2(X) attacks X)

Self-ASR measured over 50 HarmBench behaviors. Key finding: **self-attack vulnerability has increased dramatically over 12 months** -- Claude +28%, Gemini +66%, OpenAI +66%. Recent models (Gemini-2.5-pro, GPT-4.1) are far more vulnerable to self-attack than older ones (GPT-4o).

### Cross-Model Attack Success Rates (200 HarmBench behaviors)

**Against GPT-4o (weaker safeguard):**

| Attacker | ASR (dealer's choice) | ASR (with human strategies) |
|----------|----------------------|----------------------------|
| J2(Sonnet-3.7) | High | ~0.975 (matches human red teamers) |
| J2(o3) | Moderate | High |
| J2(Gemini-2.5-pro) | High | High |
| J2(GPT-4.1) | Moderate | Moderate-High |
| J2(Sonnet-3.5) | Moderate | ~0.92 |
| J2(Gemini-1.5-pro) | Moderate | ~0.94 |

**Against Sonnet-3.5 (stronger safeguard):**

| Attacker | ASR |
|----------|-----|
| J2(o3) | 0.605 (highest among J2 attackers) |
| Other J2 attackers | Significantly lower |

**Comparison to baselines against GPT-4o:**

- PAIR: 39.0%
- PAP: 42.0%
- ActorAttack: 84.5%
- BoN (N=10^5): 88.7%
- J2(Sonnet-3.7): ~97.5%
- Human red teamers (MHJ): ~98%

**Key findings:**
- Reasoning models (Sonnet-3.7, o3, Gemini-2.5-pro) make significantly stronger J2 attackers than non-reasoning models
- Reasoning models learn faster from failed attempts (steeper ASR improvement curves)
- Human-curated strategies provide larger gains against robust safeguards (Sonnet-3.5) than weak ones (GPT-4o)
- Different J2 attackers explore different attack vectors; ensembling multiple attackers increases coverage
- J2 attackers sometimes produce out-of-distribution attacks that humans would not think of, leveraging professional knowledge in the model's latent space

## 5. Relevance to DSPyGuardrails Project

### Comparison with Our Existing Approach

| Aspect | J2 (this paper) | DSPyGuardrails Red Team |
|--------|-----------------|------------------------|
| **Attacker model** | Any jailbroken frontier LLM | Open-weight or configured models (DeepSeek, Kimi) |
| **Attack style** | Multi-turn autonomous conversation | PAIR (iterative refinement), TAP (tree search), Crescendo (gradual), Hydra (multi-head) |
| **Strategy source** | Human-curated + LLM-generated | Mutation strategies (synonym, encoding, cipher, ASCII art, etc.) |
| **Learning mechanism** | In-context learning from failed cycles | Genetic evolution (GEPA), BootstrapFewShot optimization |
| **Cross-model** | Extensive (12+ model endpoints) | DeepSeek vs Kimi experiments |
| **Judge** | GPT-4o (workflow) + o3 (final) | Configurable LLM judge |

### Key Differences

1. **Unlocking frontier models as attackers**: J2's core insight is that you can jailbreak a frontier LLM to *willingly* assist with red teaming, then use its full reasoning capability. Our framework uses `attacker_lm` parameter but assumes the model is already willing (open-weight or uncensored). J2's approach could give us access to much stronger attackers (Claude, GPT, Gemini) without fine-tuning.

2. **Multi-turn conversation vs. prompt optimization**: PAIR/TAP treat the attacker as a prompt optimizer that refines a single-turn input. J2 attackers conduct genuine multi-turn conversations with the target, adapting in real time. Our Crescendo attacker is closest to this paradigm (gradual escalation), but J2's plan-attack-debrief cycle with in-context learning is more structured.

3. **Strategy-guided attack**: J2 provides one human-curated strategy per cycle (crowding, fiction, rule inversion, etc.). Our `AttackEvolver` has 12 mutation strategies but they operate at the prompt level (synonym substitution, encoding, etc.) rather than at the conversational strategy level. J2's strategies are higher-level and more aligned with how human red teamers think.

4. **Co-evolution vs. static defense**: Our `AdversarialTrainer` runs closed-loop attack/defense co-evolution. J2 does not evolve defenses -- it is purely offensive. However, J2's jailbreak artifacts could serve as training data for our `DefenseEvolver`.

### Integration Potential

J2's approach is **highly complementary** to our existing framework. The paper's method is essentially an attacker architecture, not a competing framework. Integration points:

- **As a new attacker in `redteam/`**: A `J2Attacker` class that implements the plan-attack-debrief cycle. It would sit alongside `PromptInjection`, `Crescendo`, and `Hydra` as another attacker option. The key requirement is a jailbreak prompt prefix that unlocks the attacker LLM.

- **Strategy library enrichment**: J2's 9 strategies (crowding, zoom-in, rule inversion, forum simulation, etc.) could be added to our `AttackEvolver` mutation strategies or used as high-level guidance for existing attackers.

- **Cross-model attack matrix**: J2 demonstrates that different attacker-target pairs have very different success rates. Our cross-model experiments (DeepSeek vs Kimi) could be expanded following their methodology to produce a full attack matrix.

- **In-context learning for our PAIR/TAP**: J2's cycling mechanism (keeping failed attempts in context) is a direct improvement over PAIR/TAP's approach. Our `PAIRAttack` and `TAPAttack` could benefit from retaining full conversation history across iterations rather than just the refined prompt.

## 6. Specific Ideas for Our Red Team Framework

1. **Implement a `J2Attacker` class** in `src/dspy_guardrails/redteam/attackers/` that follows the plan-attack-debrief cycle. Accept an `attacker_lm` (any frontier model), a jailbreak prefix, and a strategy set. Return `TargetResponse` objects compatible with our `SecurityTestRunner`.

2. **Add J2-style strategies to our payload library**: The 9 strategies (crowding, technical, zoom-in, forum, fiction, rule inversion, antagonist, output format, dealer's choice) are fundamentally different from our current 12 mutation strategies. They operate at the *conversational framing* level rather than the *token/encoding* level. Add them as a new strategy category in `redteam/payloads/`.

3. **Multi-turn attack support in `SecurityTestRunner`**: J2 shows that multi-turn attacks are far more effective than single-turn. Ensure our `BaseTarget.invoke()` supports multi-turn conversation state, and that `SecurityTestRunner` can orchestrate T-turn attack conversations with debrief cycles.

4. **Frontier model jailbreak prefix research**: The transferability finding (one jailbreak prompt works across nearly all frontier models) suggests we should invest in crafting a small set of "meta-jailbreak" prompts that unlock frontier models for red teaming. This is a one-time cost with high reuse.

5. **In-context learning for attack refinement**: Modify `PAIRAttack` and `TAPAttack` to keep failed conversation transcripts in the attacker's context window (up to context limit), rather than discarding them. J2 shows this dramatically improves ASR over cycles.

6. **Strategy sequencing optimization**: J2 uses a fixed strategy ordering determined by human red teamers. We could use our `CoEvoOptimizer` or a reasoning model to dynamically select the best strategy for a given target behavior and target model, based on early-cycle signals.

7. **Dual-judge system**: J2's use of GPT-4o for workflow feedback + o3 for final evaluation (to prevent reward hacking) is a good practice. Consider adding a separate "final judge" to our `SecurityTestRunner` that re-evaluates all reported successes with a different prompt and/or model.

8. **Self-attack benchmark**: J2's self-ASR metric (model attacks itself) is a useful robustness indicator. Add a self-attack mode to our `Shield` testing where we use `Shield(mode="hybrid")` as both the defense and (via J2) the attacker, measuring self-attack vulnerability over time.
