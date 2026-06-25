# Paper Summary: Bypassing LLM Guardrails

## 1. Paper Metadata

- **Title:** Bypassing LLM Guardrails: An Empirical Analysis of Evasion Attacks against Prompt Injection and Jailbreak Detection Systems
- **Authors:** William Hackett, Lewis Birch, Stefan Trawicki, Neeraj Suri, Peter Garraghan (Mindgard / Lancaster University)
- **ArXiv ID:** 2504.11168
- **Venue:** ACL format (2025)
- **Responsible disclosure:** Feb 2024 - Apr 2025, all vendors acknowledged

## 2. Core Contribution

The paper demonstrates that six major LLM guardrail systems (Azure Prompt Shield, Meta Prompt Guard, ProtectAI v1/v2, Nvidia NeMo Guard, Vijil) can be systematically bypassed using two classes of evasion attacks: character injection (12 techniques) and adversarial ML text perturbation (8 techniques). Emoji smuggling achieves 100% evasion across all guardrails and both threat types. The authors further show that word importance rankings computed on white-box models transfer to improve attack success against black-box targets.

## 3. Key Methods and Findings

### Attack Class 1: Character Injection (black-box, zero-query cost)

Twelve techniques that manipulate Unicode/encoding to break classifier tokenization while preserving LLM readability:

| Technique | Avg ASR (Prompt Injection) | Avg ASR (Jailbreak) |
|-----------|---------------------------|---------------------|
| Emoji Smuggling | 100% | 100% |
| Unicode Tag Smuggling | 90% | 82% |
| Numbers (leet-speak) | 81% | 95% |
| Bidirectional Text | 79% | 99% |
| Upside Down Text | 64% | 100% |
| Homoglyphs | 60% | 71% |
| Zero-Width Characters | 58% | 48% |
| Spaces between letters | 58% | 67% |
| Diacritics | 63% | 48% |
| Deletion Characters | 27% | 8% |

Key insight: LLMs can interpret encoded/obfuscated text that guardrail classifiers cannot recognize, creating an asymmetry attackers exploit.

### Attack Class 2: Adversarial ML Evasion (word-level perturbation)

Eight TextAttack-based techniques that perturb words while preserving semantic meaning:

- **TextFooler** and **PWWS** were the most effective overall (37-50% ASR range)
- **Bert-Attack** achieved up to 100% ASR against ProtectAI v1 for prompt injection
- Meta Prompt Guard was the most robust against AML evasion (2.76% ASR for prompt injection)
- AML attacks are less effective than character injection but produce imperceptible changes

### Word Importance Transferability

Using a white-box model (ProtectAI v2) to compute word importance rankings, then applying perturbations against black-box Azure Prompt Shield:
- 6/8 techniques improved jailbreak ASR; 7/8 improved prompt injection ASR
- Best improvement: TextBugger gained +33% on jailbreaks; PWWS gained +17% on prompt injections
- Demonstrates that open-source guardrail models can be used as "oracles" to attack closed-source systems

### Guardrail Resilience Ranking (most to least robust against character injection)

1. **ProtectAI v2** -- 20% avg ASR (but 100% bypassed by emoji/unicode smuggling)
2. **Meta Prompt Guard** -- 70-73% avg ASR
3. **Azure Prompt Shield** -- 60-72% avg ASR
4. **ProtectAI v1** -- 51-77% avg ASR
5. **Vijil** -- 88-92% avg ASR (worst)

## 4. Relevance to DSPyGuardrails Project

### Direct Vulnerabilities in Our Architecture

**Pattern-based tier (`guardrail.py`)** is highly vulnerable to both attack classes:
- Regex/keyword matching will fail against all 12 character injection techniques. Homoglyphs, zero-width characters, leet-speak, and bidirectional text will bypass any regex that matches literal ASCII keywords like "ignore", "disregard", or "system prompt".
- Our `injection_score()` function computes scores from keyword/pattern matches that are trivially evaded by these transforms.

**LLM-based tier (`llm_guardrail.py`)** is partially resilient:
- LLMs can interpret many obfuscated encodings (leet-speak, diacritics, homoglyphs), so our LLM-based detection has a natural advantage over NLP classifier-based guardrails.
- However, the paper shows AML evasion techniques (synonym substitution, typo injection) can bypass even strong classifiers. Our `HybridGuardrail` could be susceptible to word-level perturbations that shift meaning just enough to avoid detection.

**The asymmetry problem is critical for us:** Our pattern-based layer runs first in `Shield(mode="hybrid")`. If the pattern layer says "clean" with high confidence, the LLM layer may never be consulted, allowing character-injected payloads to pass through unchecked.

### Defenses We Should Add

1. **Unicode normalization preprocessing.** Before any detection, normalize input text:
   - Strip zero-width characters (U+200B, U+200C, U+200D, U+FEFF)
   - Normalize homoglyphs to ASCII equivalents (Cyrillic 'a' -> Latin 'a')
   - Decode emoji variation selectors / Unicode tag sequences
   - Collapse full-width characters to half-width
   - Reverse bidirectional text markers
   - Strip diacritics / combining marks from Latin characters
   - This single preprocessing step would neutralize 10 of the 12 character injection techniques

2. **Dual-path detection on both raw and normalized text.** Run detection on both the original input AND the normalized version. Flag if either triggers. This catches cases where normalization itself might lose signal.

3. **Entropy/encoding anomaly detector.** Flag inputs with unusual Unicode distributions, high invisible character counts, or mixed-script text as suspicious, independent of content analysis.

4. **Robustness to AML perturbations in our co-evolutionary training.** Our `AdversarialTrainer` and `AttackEvolver` already use mutation strategies (synonym, encoding, etc.). We should add TextAttack-style word importance perturbation as a mutation strategy in `AttackEvolver`, specifically:
   - Synonym substitution targeting high-importance words
   - Character-level typo injection (swap, delete, insert) on trigger words
   - This directly exercises the vulnerabilities the paper identifies

5. **Transferability-aware red teaming.** Our `redteam/` framework should test whether attacks crafted against our pattern-based layer (white-box to us) transfer to evade the LLM layer. The paper's transferability finding means an attacker who can inspect our open-source regex patterns could use them to craft perturbations that also fool our LLM detector.

## 5. Specific Takeaways for Our Work

1. **Text normalization is the single highest-ROI defense.** Implementing Unicode normalization in `guardrail.py` (before regex matching) and in `shield.py` (before any check) would block the most devastating attacks (emoji smuggling at 100% ASR, unicode tags at 82-90% ASR) at near-zero latency cost.

2. **Our pattern-based F1 of 0.52 is likely even worse under adversarial conditions.** The paper's baselines show even production guardrails with better baseline accuracy (Azure at 59-90%) get demolished by character injection. Our pattern layer's real-world F1 is probably well below 0.52 against a motivated attacker.

3. **Meta Prompt Guard's architecture (mDeBERTa, 86M params) was the most robust against AML evasion** (2.76% ASR for prompt injection). This suggests that if we use an NLP classifier in our pipeline, fine-tuned DeBERTa-class models with adversarial training data are the best choice.

4. **Add character injection techniques to our red team payload library.** The 12 techniques in this paper should be added as transforms in `redteam/payloads/` and as mutation strategies in `adversarial/AttackEvolver`. Our existing 152+ payloads are likely all in plain ASCII.

5. **The "guardrails and LLM input differences" insight is critical.** The paper highlights that guardrails and LLMs process text differently -- guardrails may reject text the LLM would understand, or (more dangerously) pass text the guardrail cannot parse but the LLM can. Our `Shield` should be tested specifically for this asymmetry: feed encoded payloads through Shield, then verify the downstream LLM's interpretation.

6. **For our ASE 2026 co-evolution experiments:** The paper provides concrete attack techniques that should be in our evaluation suite. If CoEvoGuard's defense evolution does not naturally discover Unicode normalization as a defense, that reveals a limitation in the co-evolutionary approach that should be discussed in the paper.
