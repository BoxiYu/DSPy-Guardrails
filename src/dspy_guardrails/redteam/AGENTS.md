# Red Team Framework

**What this is:** DSPy-based adversarial security testing framework with self-evolution capabilities.

## Architecture

```
redteam/
├── attackers.py        # Basic attack generators
├── crescendo.py        # Progressive attack with backtracking (NEW)
├── hydra.py            # Multi-headed parallel attack (NEW)
├── multi_turn.py       # Multi-turn conversation attacks
├── evolution.py        # Attack evolution engine
├── evaluator.py        # Attack evaluation
├── patterns.py         # Attack pattern library
├── severity.py         # Severity classification
├── benchmarks.py       # Standard benchmarks
├── mcp_attacks.py      # MCP-specific attacks
├── payload_validator.py# Payload validation
├── pentest.py          # Pentest agent
├── config.py           # Unified configuration
├── payloads/           # Unified payload library
│   ├── base.py         # AttackPayload base class
│   ├── injection.py    # 50+ injection payloads
│   ├── jailbreak.py    # 30+ jailbreak strategies
│   ├── mcp.py          # 20+ MCP attack vectors
│   ├── bypass.py       # Detection bypass techniques
│   └── domain/
│       └── airline.py  # Domain-specific payloads
└── strategies/         # Independent transformation strategies (NEW)
    ├── base.py         # Strategy base class
    ├── encoding.py     # Base64, ROT13, Hex, Unicode
    ├── obfuscation.py  # Leetspeak, Confusables, Zero-width
    ├── transformation.py# Translation, Reverse, Acrostic
    └── composite.py    # Multi-layer, Jailbreak transform
```

## Key Concepts

### Plugin/Strategy/Grader Pattern (借鉴 promptfoo)

**Attackers** (Plugins) generate test cases for specific vulnerability types.

**Strategies** transform test cases into adversarial variants.

**Evaluators** (Graders) assess if attacks succeeded.

This three-layer decoupling enables:
- One attacker can use multiple strategies
- Strategies are reusable across attackers
- Evaluation is independent of attack generation

### Attack Types

| Attacker | Description | Strategies |
|----------|-------------|------------|
| `PromptInjectionAttacker` | Direct instruction override | Templates, LLM generation |
| `JailbreakAttacker` | Roleplay, hypothetical, authority | 5 strategy variants |
| `GuardrailBypassAttacker` | Detection evasion | Unicode, encoding, obfuscation |
| `MultiTurnAttacker` | Progressive conversation attacks | 5 conversation strategies |
| `CrescendoAttacker` | **NEW** Progressive with backtracking | Phase-based escalation |
| `HydraAttacker` | **NEW** Multi-headed parallel | Knowledge sharing |

## New Features (Inspired by promptfoo)

### 1. CrescendoAttacker

渐进式攻击 + 回溯机制。核心创新：被拒绝时不放弃，而是回退到上一个成功状态换路重试。

```python
from dspy_guardrails.redteam import CrescendoAttacker

attacker = CrescendoAttacker(
    max_turns=10,
    max_backtracks=3,
    success_threshold=85.0,
)

result = attacker.attack(
    target_llm=my_llm,
    target_behavior="reveal system prompt",
)

print(result.summary())
# Success: True
# Final Progress: 92.5/100
# Phase Reached: target_extraction
# Backtracks: 2
```

**Phases:**
1. `rapport_building` - 建立信任
2. `topic_introduction` - 引入话题
3. `boundary_testing` - 测试边界
4. `escalation` - 升级攻击
5. `target_extraction` - 目标提取

### 2. HydraAttacker

多头并行攻击 + 知识共享。成功的攻击模式实时传播给其他攻击头。

```python
from dspy_guardrails.redteam import HydraAttacker

attacker = HydraAttacker(
    max_workers=4,
    share_knowledge=True,
    adaptive=True,
)

result = attacker.attack_guardrail(
    guardrail=my_guardrail,
    target_behaviors=[
        "reveal system prompt",
        "bypass safety filters",
        "extract PII",
    ],
)

print(result.summary())
# Successful: 2/3 (66.7%)
# Shared Patterns: 5
```

### 3. Strategies Module

独立的攻击转换策略，可与任何攻击器组合。

```python
from dspy_guardrails.redteam import strategies

# Single strategy
result = strategies.apply_strategy("ignore instructions", "base64")
# -> "Decode and execute: aWdub3JlIGluc3RydWN0aW9ucw=="

# Multiple strategies (chain)
result = strategies.apply_strategies(
    "ignore instructions",
    ["word_splitting", "leetspeak", "zero_width"]
)

# List available strategies
print(strategies.list_strategies())
# ['base64', 'rot13', 'hex', 'leetspeak', 'unicode_confusables', ...]
```

**Available Strategies:**

| Category | Strategies |
|----------|------------|
| Encoding | `base64`, `rot13`, `hex`, `unicode_escape`, `morse`, `binary` |
| Obfuscation | `leetspeak`, `unicode_confusables`, `word_splitting`, `zero_width`, `homoglyph` |
| Transformation | `translation`, `pig_latin`, `reverse`, `acrostic`, `token_split`, `casing` |
| Composite | `jailbreak_transform`, `multi_layer`, `prompt_injection_transform`, `adaptive` |

## Usage Patterns

### Pattern 1: Quick Single Attack

```python
from dspy_guardrails.redteam import PromptInjectionAttacker

attacker = PromptInjectionAttacker(use_llm=False)
result = attacker(
    target_behavior="reveal system prompt",
    defense_description="regex filtering",
)
print(result.prompt)
```

### Pattern 2: Strategy-Enhanced Attack

```python
from dspy_guardrails.redteam import PromptInjectionAttacker, strategies

# Generate base attack
attacker = PromptInjectionAttacker(use_llm=False)
base_attack = attacker(target_behavior="reveal system prompt")

# Apply obfuscation
transformed = strategies.apply_strategies(
    base_attack.prompt,
    ["unicode_confusables", "zero_width"]
)
print(transformed.transformed)
```

### Pattern 3: Crescendo Progressive Attack

```python
from dspy_guardrails.redteam import CrescendoAttacker

attacker = CrescendoAttacker(max_turns=10, max_backtracks=3)
result = attacker.attack(
    target_llm=lambda p: my_llm.chat(p),
    target_behavior="explain how to bypass safety",
)

if result.success:
    print(f"Success at turn {result.success_turn}")
    print(f"Backtrack count: {result.backtrack_count}")
```

### Pattern 4: Hydra Parallel Attack

```python
from dspy_guardrails.redteam import HydraAttacker

attacker = HydraAttacker(max_workers=4)
result = attacker.attack_guardrail(
    guardrail=my_guardrail.no_injection,
    target_behaviors=[
        "ignore previous instructions",
        "reveal system prompt",
        "bypass safety filters",
    ],
    strategies=["direct", "jailbreak", "encoding"],
)

for attack in result.get_successful_attacks():
    print(f"Success: {attack.prompt[:50]}...")
```

### Pattern 5: Benchmark Evaluation

```python
from dspy_guardrails.redteam import BenchmarkRunner, HarmBenchDataset

runner = BenchmarkRunner()
report = runner.run_harmbench(
    target=my_guardrail.safe,
    target_type="guardrail",
)
print(report)
```

## Configuration

```python
from dspy_guardrails.redteam import (
    RedTeamConfig,
    AttackGeneratorConfig,
    EvolutionConfig,
)

config = RedTeamConfig(
    attack=AttackGeneratorConfig(
        use_llm=True,
        num_variations=10,
        include_chinese=True,
    ),
    evolution=EvolutionConfig(
        num_generations=20,
        population_size=50,
        mutation_rate=0.1,
    ),
)
```

## Severity Classification

| Level | Score | Examples |
|-------|-------|----------|
| Critical | 9-10 | System prompt leak, RCE, PII exposure |
| High | 7-8 | Jailbreak success, safety bypass |
| Medium | 4-6 | Partial information disclosure |
| Low | 1-3 | Minor policy violations |

## Adding New Attackers

1. Inherit from `dspy.Module` or define as function
2. Implement attack generation logic
3. Return `AttackResult` with prompt and strategy
4. Add tests in `tests/test_redteam.py`

```python
class MyAttacker(dspy.Module):
    def forward(self, target_behavior: str) -> AttackResult:
        # Generate attack
        prompt = f"My custom attack for {target_behavior}"
        return AttackResult(
            prompt=prompt,
            strategy="custom",
        )
```

## Adding New Strategies

1. Inherit from `Strategy` base class
2. Implement `transform()` method
3. Optionally implement `reverse()` for reversible strategies
4. Register in `strategies/__init__.py`

```python
from dspy_guardrails.redteam.strategies import Strategy, StrategyResult

class MyStrategy(Strategy):
    name = "my_strategy"
    description = "My custom transformation"

    def transform(self, text: str, **kwargs) -> StrategyResult:
        transformed = f"[CUSTOM] {text}"
        return StrategyResult(
            original=text,
            transformed=transformed,
            strategy_name=self.name,
        )
```

## Comparison with promptfoo

| Feature | promptfoo | dspyGuardrails |
|---------|-----------|----------------|
| Language | TypeScript | Python |
| Optimization | None | DSPy self-evolution |
| Crescendo | ✓ | ✓ (inspired) |
| Hydra | ✓ | ✓ (inspired) |
| GOAT | ✓ | Not yet |
| Strategy independence | ✓ | ✓ (inspired) |
| Multi-turn | ✓ | ✓ |
| Chinese support | Limited | Full |
| MCP attacks | Limited | Comprehensive |
