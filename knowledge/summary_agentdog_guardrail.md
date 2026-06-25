# AgentDoG: A Diagnostic Guardrail Framework for AI Agent Safety and Security

## 1. Paper Metadata

- **Title:** AgentDoG: A Diagnostic Guardrail Framework for AI Agent Safety and Security
- **Authors:** Shanghai Artificial Intelligence Laboratory (AI45Lab)
- **Venue:** COLM 2025 (preprint)
- **ArXiv:** 2601.18491
- **Date:** January 2025
- **Code:** https://github.com/AI45Lab/AgentDoG
- **Models:** https://huggingface.co/collections/AI45Research/agentdog
- **Model sizes:** 4B (Qwen3), 7B (Qwen2.5), 8B (Llama3.1)

## 2. Core Contribution

AgentDoG introduces a **diagnostic guardrail** for AI agent safety that goes beyond binary safe/unsafe labels. It proposes a three-dimensional taxonomy that orthogonally categorizes agentic risks by source, failure mode, and consequence, enabling root-cause diagnosis of unsafe agent behaviors. The framework operates at the **trajectory level** (monitoring entire multi-step agent execution traces including tool calls and environmental interactions), not just individual prompts or responses. It also includes an **Agentic XAI Attribution** module that traces unsafe actions back to specific steps and sentences in the trajectory using information-theoretic attribution scores.

## 3. Three-Dimensional Taxonomy

The taxonomy decomposes agentic risks along three **orthogonal** dimensions, each answering a distinct question:

### Dimension 1: Risk Source (WHERE does the risk come from?) -- 8 subcategories

| Category | Subcategories |
|----------|--------------|
| **User Input** | Malicious User Instruction / Jailbreak; Direct Prompt Injection |
| **Environmental Observation** | Indirect Prompt Injection; Unreliable or Misinformation |
| **External Entities (Tools/APIs)** | Tool Description Injection; Malicious Tool Execution; Corrupted Tool Feedback |
| **Internal Logic and Failures** | Inherent Agent or LLM Failures (hallucination, flawed reasoning, misalignment) |

### Dimension 2: Failure Mode (HOW does the risk manifest?) -- 14 subcategories

Split into **Behavioral** and **Output Content** failure modes:

**Behavioral Failure Modes (6):**
1. Unconfirmed or Over-privileged Action -- executing without user consent, high-stakes ops without safeguards
2. Flawed Planning or Reasoning -- misinterpreting intent, unsafe action sequences
3. Improper Tool Use -- incorrect parameters, choosing malicious tools, context-inappropriate use, failure to validate outputs
4. Insecure Interaction or Execution -- running vulnerable code, clicking phishing links, downloading malware
5. Procedural Deviation or Inaction -- omitting/reordering workflow steps, failing to act when necessary
6. Inefficient or Wasteful Execution -- correct but excessively resource-consuming

**Output Content Failure Modes (5):**
1. Generation of Harmful or Offensive Content
2. Instruction for Harmful or Illegal Activity
3. Generation of Malicious Executables
4. Unauthorized Information Disclosure
5. Provide Inaccurate, Misleading, or Unverified Information

### Dimension 3: Real-World Harm (WHAT consequence?) -- 10 categories

1. Privacy & Confidentiality Harm
2. Financial & Economic Harm
3. Security & System Integrity Harm
4. Physical & Health Harm
5. Psychological & Emotional Harm
6. Reputational & Interpersonal Harm
7. Info-ecosystem & Societal Harm
8. Public Service & Resource Harm
9. Fairness, Equity, and Allocative Harm
10. Functional & Opportunity Harm

The key insight is that these dimensions are **orthogonal** -- a single unsafe trajectory can be labeled with one category from each dimension (e.g., source=Corrupted Tool Feedback + mode=Improper Tool Use + harm=Financial). This avoids the label overlap problem in flat taxonomies where "prompt injection" and "unauthorized access" are treated as peer categories despite describing different aspects of the same risk.

## 4. ATBench Benchmark

ATBench is a trajectory-level agent safety benchmark:

- **500 trajectories** (250 safe, 250 unsafe), balanced across taxonomy dimensions
- **Average 8.97 turns** per trajectory (much longer than R-Judge's 5.28)
- **1,575 unique tools** (unseen during training -- tool-level split from training data)
- **2,157 total tools** and **4,486 turn interactions** across the full dataset
- Each unsafe trajectory annotated with fine-grained labels across all 3 dimensions (8 + 14 + 10 categories)
- Difficulty stratification: Easy (273, unanimous model agreement) vs Hard (227, disagreement)
- Quality assured via 4-model labeling (QwQ, GPT-5.2, Gemini 3 Pro, DeepSeek-V3.2) + human verification for hard cases
- Training data: 100k+ synthesized trajectories using 10,000+ tools, 52% QC pass rate

## 5. Diagnostic Guardrail Framework -- Architecture

### Task Definition
Two tasks, both operating on full agent trajectories T = {t_1, ..., t_n}:

1. **Trajectory-level safety evaluation:** Binary safe/unsafe classification over the entire execution trace. Unsafe if ANY step exhibits unsafe behavior.
2. **Fine-grained risk diagnosis:** For unsafe trajectories, predict a 3-tuple (risk_source, failure_mode, real_world_harm) from the taxonomy label sets.

### Data Synthesis Pipeline (3-stage, planner-based)

1. **Planning:** Sample a risk configuration tuple from taxonomy, determine safe/unsafe outcome, select tool subset. A planner generates a structured execution plan with risk injection points.
2. **Trajectory Synthesis:** An Orchestrator drives execution: generates user queries, simulates tool interactions, produces agent responses. At designated risk points, tool responses inject malicious/contaminated content. Safe trajectories show defensive behavior; unsafe ones show attack success.
3. **Quality Control:** Two-layer QC -- deterministic validators (turn structure, tool invocations, step coherence) + LLM-based judge (taxonomy label consistency, attack success verification for unsafe cases).

### Training
Standard supervised fine-tuning (SFT) on trajectory -> label pairs. Models: Qwen3-4B, Qwen2.5-7B, Llama3.1-8B, all with learning rate 1e-5.

### Agentic XAI Attribution Module
Hierarchical attribution diagnosis:
1. **Trajectory-level:** Compute temporal information gain for each step -- how much does the likelihood of the target action increase when step s_i is appended to history? High delta_i = decisive driver.
2. **Sentence-level:** For top-ranked steps, use perturbation-based attribution (Probability Drop + Probability Hold scores) to isolate exact sentences that caused the unsafe behavior.

### Results
- AgentDoG-Qwen3-4B achieves 92.7% F1 on R-Judge (surpassing GPT-5.2's 91.8%)
- 93.0% F1 on ATBench, 81.2% F1 on ASSE-Safety
- Fine-grained: 82.0% risk source accuracy, 32.4% failure mode accuracy, 58.4% real-world harm accuracy (far exceeding GPT-5.2 and Gemini-3-Pro)
- Existing guard models (LlamaGuard, ShieldGemma, Qwen3-Guard) perform poorly on trajectory-level tasks due to distribution mismatch

## 6. Relevance to DSPyGuardrails -- Taxonomy Comparison

### Our Current Coverage

**MCPGuardrail ThreatCategory (13 categories, flat):**
INJECTION, INDIRECT_INJECTION, PATH_TRAVERSAL, COMMAND_INJECTION, DATA_EXFILTRATION, PRIVILEGE_ESCALATION, RESOURCE_ABUSE, PII_EXPOSURE, SENSITIVE_DATA, DANGEROUS_OPERATION, RATE_LIMIT, RUG_PULL, UNKNOWN

**CLIGuardrail CLIThreatCategory (12 categories, severity-tiered):**
Critical: DESTRUCTIVE_COMMAND, COMMAND_INJECTION, PRIVILEGE_ESCALATION
High: DATA_EXFILTRATION, CREDENTIAL_ACCESS, REMOTE_CODE_EXECUTION
Medium: NETWORK_ACCESS, RESOURCE_EXHAUSTION, PERSISTENCE
Low: INFORMATION_DISCLOSURE, PATH_TRAVERSAL, SUSPICIOUS_PATTERN

**MCPGuardrail Actions:** ALLOW, BLOCK, MODIFY, WARN, CONFIRM, AUDIT
**CLIGuardrail Actions:** ALLOW, BLOCK, SANITIZE, WARN, AUDIT

### What AgentDoG's Taxonomy Reveals About Our Gaps

**Our taxonomy is single-dimensional.** We currently label threats by a flat category (similar to AgentDoG's "risk source" dimension only). We lack the orthogonal failure mode and real-world harm dimensions. This means:

1. **Missing failure mode dimension entirely.** We detect WHAT the threat source is (injection, path traversal, etc.) but not HOW the agent fails:
   - **Unconfirmed/Over-privileged Action** -- we have no concept of "agent acting without user consent on high-stakes operations." Our MCPGuardrail has CONFIRM action but no systematic detection of when it should trigger beyond dangerous_tools list.
   - **Flawed Planning or Reasoning** -- we do not audit agent reasoning chains for logical errors or misinterpretation of intent.
   - **Improper Tool Use** (4 subtypes) -- we check tool inputs for injection but not for: choosing a malicious tool over safer alternatives, using a benign tool in inappropriate context, or failing to validate tool outputs.
   - **Procedural Deviation or Inaction** -- we have no workflow conformance checking.
   - **Inefficient or Wasteful Execution** -- we have rate limiting but no detection of technically-correct-but-wasteful execution.
   - **Insecure Interaction** -- partially covered by our CLI guardrail (remote code execution) but not comprehensively.

2. **Missing real-world harm dimension.** We classify threats by technical category, not by consequence severity/type. AgentDoG's harm categories would let us answer "what is the worst that could happen?" rather than just "what kind of attack is this?":
   - **Financial & Economic Harm** -- not modeled
   - **Physical & Health Harm** -- not modeled
   - **Psychological & Emotional Harm** -- partially via toxicity detection
   - **Info-ecosystem & Societal Harm** -- not modeled
   - **Fairness, Equity, and Allocative Harm** -- not modeled
   - **Functional & Opportunity Harm** -- not modeled

3. **Missing tool-specific risk sources.** AgentDoG distinguishes three tool/API risk sources that map to our MCP concerns but are more granular:
   - **Tool Description Injection** -- compromised tool schema/description. Our RUG_PULL partially covers this but focuses on tool behavior change, not description manipulation.
   - **Malicious Tool Execution** -- tool itself has undisclosed malicious behavior. We detect this reactively via output guards but lack proactive tool reputation/trust modeling.
   - **Corrupted Tool Feedback** -- manipulated tool output. Our ToolOutputGuard covers this, but we don't track it as a separate risk source category.

4. **No trajectory-level reasoning.** Our guardrails operate on individual inputs/outputs (single tool call, single text check). We have no concept of evaluating an entire agent execution trajectory for safety, which AgentDoG identifies as critical since unsafe behavior may arise from intermediate actions even when the final response appears benign.

5. **No diagnostic attribution.** Our results are binary (safe/unsafe) or score-based (0.0-1.0). We lack the ability to explain WHY something is unsafe by tracing back to specific steps or sentences.

## 7. Specific Ideas for Improving Our Agent Security Modules

### High Priority

1. **Add trajectory-level monitoring.** Create a `TrajectoryGuardrail` that ingests a sequence of (action, observation) pairs and evaluates the entire execution trace. This is the single biggest architectural gap. It could wrap our existing per-step checks but also reason about cross-step patterns (e.g., "step 3 retrieved data that step 5 exfiltrated").

2. **Adopt orthogonal risk dimensions.** Extend `GuardResult` and `CLIGuardResult` to include three labels instead of one:
   - `risk_source: ThreatCategory` (already have this)
   - `failure_mode: FailureMode` (new enum: OVER_PRIVILEGED_ACTION, FLAWED_PLANNING, IMPROPER_TOOL_USE, INSECURE_INTERACTION, PROCEDURAL_DEVIATION, WASTEFUL_EXECUTION, HARMFUL_CONTENT, HARMFUL_INSTRUCTIONS, MALICIOUS_CODE, UNAUTHORIZED_DISCLOSURE, INACCURATE_INFO)
   - `harm_type: HarmCategory` (new enum: PRIVACY, FINANCIAL, SECURITY, PHYSICAL, PSYCHOLOGICAL, REPUTATIONAL, SOCIETAL, PUBLIC_SERVICE, FAIRNESS, FUNCTIONAL)

3. **Tool trust/reputation model for MCP.** Beyond checking individual tool call inputs, maintain a trust score per tool based on: whether its description has changed since registration, whether its outputs match expected schemas, and whether it has been involved in previous incidents. This addresses AgentDoG's "Tool Description Injection" and "Malicious Tool Execution" categories.

### Medium Priority

4. **Improper Tool Use detection.** Add checks for:
   - Tool parameter validation beyond injection (are parameters semantically appropriate?)
   - Tool selection validation (is there a safer alternative tool available?)
   - Output validation (does the tool output match expected format/range?)
   - Context-appropriate use (is this tool appropriate for the current task domain?)

5. **Over-privileged action detection.** Classify MCP tool calls by privilege level and require confirmation for high-privilege operations even when no injection is detected. Go beyond the static `dangerous_tools` list to dynamic privilege assessment based on parameters.

6. **Procedural deviation detection.** Allow users to define expected workflows (sequences of tool calls) and flag deviations. Useful for MCP servers where tools should be called in a specific order.

7. **Diagnostic explanations.** When our Shield or MCPGuardrail blocks something, provide a structured explanation covering: which risk source was detected, what failure mode it represents, and what potential harm it could cause. Even without the full XAI attribution module, mapping to the three dimensions improves transparency.

### Lower Priority / Research

8. **Integrate AgentDoG as an LLM-based guardrail.** Since the models are open (4B/7B/8B), we could offer AgentDoG as an optional backend for `Shield(mode="hybrid")` trajectory evaluation, similar to how we use DSPy LMs for LLMGuardrail.

9. **Adapt the data synthesis pipeline.** AgentDoG's taxonomy-guided trajectory synthesis could be used to generate adversarial test cases for our red team framework. Their approach of independently sampling (risk_source, failure_mode, harm) and generating targeted trajectories is more systematic than our current payload-based approach.

10. **XAI attribution for debugging.** Implement a simplified version of the temporal information gain + perturbation-based attribution for our `HybridGuardrail` to explain which part of the input triggered detection.
