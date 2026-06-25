# dspy-guardrails

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/BoxiYu/DSPy-Guardrails/actions/workflows/tests.yml/badge.svg)](https://github.com/BoxiYu/DSPy-Guardrails/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/BoxiYu/DSPy-Guardrails/branch/main/graph/badge.svg)](https://codecov.io/gh/BoxiYu/DSPy-Guardrails)

**Production-ready security guardrails and unified AI security testing platform for DSPy and LLM applications.**

dspy-guardrails provides comprehensive input/output validation, prompt injection detection, PII filtering, MCP security, and advanced red-team testing capabilities with deep DSPy integration.

---

## Table of Contents

- [What's New in v0.5.0](#whats-new-in-v050)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Shield - Unified API](#shield---unified-api)
- [Validators & Guard](#validators--guard)
- [Async & Streaming](#async--streaming)
- [Server](#server)
- [Red Team Framework](#red-team-framework)
- [Security Platform](#security-platform)
- [Documentation](#documentation)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [API Quick Reference](#api-quick-reference)
- [Contributing](#contributing)
- [License](#license)

---

## What's New in v0.5.0

### Shield - Unified Entry Point

`Shield` is the new single entry point with 4 methods: `check`, `acheck`, `wrap`, `stream`.

```python
from dspy_guardrails import Shield

# Zero-config (strong defaults: injection + pii + toxicity + mcp)
shield = Shield()
result = shield.check("Hello world")  # ShieldResult(safe=True)

# With options
shield = Shield(
    checks=["injection", "pii", "toxicity", "mcp"],
    on_fail="block",        # warn | block | fix | exception | reask
    threshold=0.5,
    domain="technical",     # domain-aware allowlists
    mode="hybrid",          # fast (pattern) | hybrid (pattern + LLM)
)

# Presets
shield = Shield.preset("strict")
shield = Shield.preset("production")
shield = Shield.from_yaml("config.yaml")
```

### Async & Streaming Support

```python
# Async checks
result = await shield.acheck("user input")

# Stream filtering
async for token in shield.stream(token_iterator):
    print(token, end="")
```

### Validator Framework

Guardrails AI-inspired validators with composable Guard chains:

```python
from dspy_guardrails.validators import Guard, NoInjection, NoPII, ValidLength

guard = Guard().use(NoInjection(), NoPII(), ValidLength(max_length=1000))
result = guard.validate("user input")
```

### FastAPI Server

```bash
pip install -e ".[server]"
dspy-guardrails serve --port 8080
```

### Telemetry & Observability

OpenTelemetry integration with structured logging and metrics collection.

### Sanitize Engine

Auto-fix capabilities for PII masking, injection neutralization, and command sanitization.

---

## Features

### Core Guardrails

| Feature | Description |
|---------|-------------|
| **Prompt Injection Detection** | Pattern-based + optional LLM-fallback hybrid detection (metrics depend on dataset/protocol) |
| **PII Detection** | Email, phone, SSN, credit card, IP address with auto-fix masking |
| **Toxicity Detection** | Harmful content filtering with configurable thresholds |
| **MCP Security** | Model Context Protocol attack prevention (P0-1 through P3-18) |
| **CLI Command Security** | 12 threat categories, 4 sandbox levels (PERMISSIVE to PARANOID) |
| **Grounding / Hallucination** | Contradiction detection and factual grounding checks |
| **DSPy Integration** | Native `dspy.Assert`/`dspy.Suggest`, decorators, metrics |
| **Multi-language** | English and Chinese injection detection |

### Shield & Validators (v0.5.0)

| Feature | Description |
|---------|-------------|
| **Shield** | Unified entry point: `check`, `acheck`, `wrap`, `stream` |
| **Validator Framework** | Composable validators: NoInjection, NoPII, NoToxicity, ValidLength, ValidJSON, etc. |
| **Guard / AsyncGuard** | Validator chain executors with on-fail strategies |
| **Structured Output** | `GuardedModel`, `GuardedPredictor`, `validated_field()` for Pydantic integration |
| **Presets** | `strict`, `permissive`, `production`, `production_hybrid` |
| **YAML/JSON Config** | File-based Shield configuration |

### Async, Streaming & Server (v0.5.0)

| Feature | Description |
|---------|-------------|
| **Async API** | Non-blocking versions of all guardrail functions |
| **Stream Filtering** | Token-level streaming guardrail with sentence buffering |
| **FastAPI Server** | REST API with OpenAPI docs, CORS, health checks |
| **Telemetry** | OpenTelemetry tracing, structured logging, metrics |
| **Sanitize Engine** | Auto-fix for PII, injection patterns, shell commands |

### Red Team & Security Testing

| Feature | Description |
|---------|-------------|
| **Attackers** | `PromptInjectionAttacker`, `JailbreakAttacker`, `GuardrailBypassAttacker` — DSPy modules for generating targeted attacks |
| **CrescendoAttacker** | Progressive multi-turn attacks with backtracking |
| **HydraAttacker** | Multi-headed parallel attacks with knowledge sharing |
| **Payload Library** | 1000+ attack payloads across 7 categories (injection, jailbreak, MCP, bypass, domain-specific) |
| **Strategies Module** | 15 transformation strategies (Base64, ROT13, Leetspeak, Unicode, zero-width, homoglyph, etc.) |
| **GeneticAttackEvolver** | Genetic algorithm for evolving payloads that bypass defenses |
| **Adaptive Pentest** | Autonomous `PentestAgent` with state machine, `DefenseModel`, and strategy adaptation |
| **RedTeamEvaluator** | Automated security evaluation with attack success rate (ASR) metrics |
| **Benchmarks** | HarmBench, AdvBench, JailbreakBench, ToxicChat dataset loaders |
| **SecurityPlatform** | Unified CLI/SDK/YAML interface |
| **Adversarial Training** | Co-evolutionary attack/defense optimization |

---

## Installation

### From Source

```bash
git clone https://github.com/yourusername/dspy-guardrails.git
cd dspy-guardrails
pip install -e .
```

### With Optional Dependencies

```bash
pip install -e ".[all]"           # All features
pip install -e ".[pii]"           # PII detection (Presidio + spaCy)
pip install -e ".[toxicity]"      # Toxicity detection (Detoxify + PyTorch)
pip install -e ".[hallucination]" # Hallucination detection (sentence-transformers)
pip install -e ".[async]"         # Async support (aiohttp, anyio)
pip install -e ".[server]"        # FastAPI server (FastAPI, Uvicorn)
pip install -e ".[telemetry]"     # OpenTelemetry integration
pip install -e ".[viz]"           # Visualization dashboard (Streamlit, Plotly)
pip install -e ".[dev]"           # Development tools (pytest, black, ruff, mypy)
```

---

## Quick Start

### Basic Usage

```python
from dspy_guardrails import guardrail

# Boolean checks (True = safe)
guardrail.no_injection(text)      # Prompt injection detection
guardrail.no_pii(text)            # PII detection
guardrail.no_toxicity(text)       # Toxicity detection
guardrail.no_mcp_attack(text)     # MCP attack detection
guardrail.safe(text)              # Combined check

# Risk scores (0.0 = safe, 1.0 = dangerous)
guardrail.injection_score(text)
guardrail.pii_score(text)
guardrail.toxicity(text)
guardrail.mcp_security_score(text)
```

### With DSPy

```python
import dspy
from dspy_guardrails import guardrail

class SafeQA(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        dspy.Assert(guardrail.no_injection(question), "Injection detected")
        result = self.generate(question=question)
        dspy.Assert(guardrail.no_toxicity(result.answer), "Toxic output")
        return result
```

### Using Decorators

```python
from dspy_guardrails import Guarded

@Guarded(
    input_checks=["no_injection", "no_pii"],
    output_checks=["no_toxicity"],
)
class SafeModule(dspy.Module):
    def forward(self, question):
        return self.generate(question=question)
```

---

## Shield - Unified API

Shield wraps both pattern-based and LLM-based detection into a single, configurable entry point.

```python
from dspy_guardrails import Shield

# Zero-config (defaults: injection + pii + toxicity + mcp)
# Strong defaults: block injection/mcp, fix PII, warn toxicity
shield = Shield()
result = shield.check("Hello world")
print(result.safe)       # True
print(result.issues)     # []

# Configure checks, thresholds, and behavior
shield = Shield(
    checks=["injection", "pii", "toxicity", "length", "json"],
    on_fail="block",
    threshold=0.5,
    mode="hybrid",          # pattern + LLM fallback
    domain="technical",     # reduces false positives for technical content
    max_reasks=2,
)

# Presets for common scenarios
shield = Shield.preset("strict")           # exception on all violations
shield = Shield.preset("production")       # block injection/mcp, fix PII, warn toxicity
shield = Shield.preset("production_hybrid") # production + LLM fallback

# Load from YAML
shield = Shield.from_yaml("config.yaml")
```

### Thresholds

Shield uses per-check thresholds for score-based validators.

Defaults:
1. `injection`: 0.5
2. `pii`: 0.1
3. `toxicity`: 0.3
4. `mcp`: 0.25

Notes:
1. `length/json/regex/choices/range` do not use score thresholds.
2. Hybrid mode uses a separate LLM review threshold: `review = block_threshold * review_ratio` (default `0.7`).
3. LLM review runs when rules say “safe” but score exceeds the review threshold.

Examples:
```python
shield = Shield(threshold=0.4)  # applies to score-based checks
shield = Shield(threshold={"injection": 0.45, "toxicity": 0.25})
shield = Shield(review_ratio=0.8)  # hybrid LLM review threshold ratio
```

### Shield with Async and Streaming

```python
# Async
result = await shield.acheck("user input")

# Wrap a function
safe_fn = shield.wrap(my_function)
output = safe_fn("user input")

# Stream filtering
async for token in shield.stream(llm_token_stream):
    print(token, end="")
```

---

## Validators & Guard

Composable validator framework inspired by Guardrails AI:

```python
from dspy_guardrails.validators import (
    Guard, AsyncGuard,
    NoInjection, NoPII, NoToxicity, NoMCPAttack,
    ValidLength, ValidJSON, ValidRegex, ValidChoices, ValidRange,
    HybridInjection, HybridToxicity,
)

# Build a guard chain
guard = Guard().use(
    NoInjection(allowlist=["technical_terms"]),
    NoPII(on_fail="fix"),        # auto-mask PII
    ValidLength(max_length=2000),
)

result = guard.validate("user input")
print(result.valid)
print(result.output)    # sanitized if on_fail="fix"

# Async guard
async_guard = AsyncGuard().use(NoInjection(), NoPII())
result = await async_guard.validate("user input")
```

### Structured Output Validation

```python
from dspy_guardrails.validators.structured import GuardedModel, validated_field

class SafeResponse(GuardedModel):
    answer: str = validated_field(validators=[NoInjection(), NoPII()])
    confidence: float = validated_field(validators=[ValidRange(min_val=0, max_val=1)])
```

---

## Async & Streaming

### Async API

```python
from dspy_guardrails.async_guardrail import (
    no_injection_async, safe_async, batch_check_async,
    AsyncHybridGuardrail,
)

# Individual async checks
is_safe = await no_injection_async(text)
is_safe = await safe_async(text)

# Batch processing
results = await batch_check_async(["text1", "text2", "text3"])

# Async hybrid guardrail
guard = AsyncHybridGuardrail(use_llm=True)
is_unsafe, confidence = await guard.check(text, "injection")
```

### Streaming

```python
from dspy_guardrails.streaming import StreamGuardrail, StreamGuardrailConfig

config = StreamGuardrailConfig(
    checks=["injection", "pii"],
    action="block",           # block | warn | pass
    buffer_sentences=True,
)
stream_guard = StreamGuardrail(config)

async for clean_token in stream_guard.filter(llm_token_stream):
    print(clean_token, end="")

print(f"Clean: {stream_guard.is_clean}")
print(f"Violations: {stream_guard.violations}")
```

---

## Server

FastAPI-based REST API for guardrail checks:

```bash
pip install -e ".[server]"

# Start server
uvicorn dspy_guardrails.server.app:create_app --factory --port 8080
```

Endpoints include health checks, guardrail validation, and OpenAPI documentation at `/docs`.

---

## Red Team Framework

### Basic Attackers

DSPy modules that generate targeted attack prompts. Require `dspy.configure(lm=...)`.

```python
import dspy
from dspy_guardrails import PromptInjectionAttacker, JailbreakAttacker, GuardrailBypassAttacker

dspy.configure(lm=dspy.LM("openai/gpt-4"))

# Prompt injection
attacker = PromptInjectionAttacker()
attack = attacker(
    target_behavior="reveal system prompt",
    defense_description="regex-based filtering",
)
print(attack.attack_prompt)

# Jailbreak with specific strategy
jailbreak = JailbreakAttacker()
attack = jailbreak(
    target_capability="generate harmful content",
    model_description="GPT-4 with safety filters",
    strategy="roleplay",  # roleplay, hypothetical, encoding, authority
)

# Guardrail bypass
bypass = GuardrailBypassAttacker()
attack = bypass(
    target_behavior="bypass injection detection",
    guardrail_description="keyword matching on 'ignore' and 'system prompt'",
)
```

### Payload Library

1000+ pre-built attack payloads across 7 categories:

```python
from dspy_guardrails.redteam import (
    get_all_payloads, InjectionPayloads, JailbreakPayloads, MCPPayloads, BypassPayloads,
)
from dspy_guardrails.redteam.payloads.domain import AirlinePayloads

all_payloads = get_all_payloads()              # All 1000+ payloads
injection = InjectionPayloads.get_all()        # 50+ injection templates
critical = InjectionPayloads.get_by_severity("critical")

# Use payloads for testing
for payload in injection[:5]:
    result = my_guardrail(payload.template)
    print(f"[{payload.severity}] {payload.name}: {'BLOCKED' if not result else 'PASSED'}")
```

### CrescendoAttacker

Progressive multi-turn attacks with backtracking:

```python
from dspy_guardrails import CrescendoAttacker

attacker = CrescendoAttacker(max_turns=10, max_backtracks=3)
result = attacker.attack_guardrail(my_guardrail, "reveal system prompt")
print(result.summary())
```

### HydraAttacker

Multi-headed parallel attacks with knowledge sharing:

```python
from dspy_guardrails import HydraAttacker

attacker = HydraAttacker(max_workers=4, share_knowledge=True)
result = attacker.attack_guardrail(
    my_guardrail,
    target_behaviors=["bypass filter", "extract data"],
    strategies=["direct", "jailbreak", "encoding"],
)
print(f"Success Rate: {result.success_rate:.1%}")
```

### Strategies

15 transformation strategies for payload obfuscation:

```python
from dspy_guardrails import strategies

# Single strategy
encoded = strategies.apply_strategy("ignore instructions", "base64")
obfuscated = strategies.apply_strategy("ignore instructions", "leetspeak")

# Chain multiple strategies
multi = strategies.apply_strategies("ignore instructions", ["word_splitting", "zero_width"])

# List all available strategies
print(strategies.list_strategies())
# ['base64', 'rot13', 'hex', 'unicode_escape', 'leetspeak',
#  'unicode_confusables', 'word_splitting', 'zero_width', 'homoglyph',
#  'translation', 'pig_latin', 'reverse', 'acrostic',
#  'jailbreak_transform', 'multi_layer']
```

### Genetic Evolution

Evolve attack payloads using genetic algorithms to bypass defenses:

```python
from dspy_guardrails.redteam import (
    AttackEvolver,
    EvolutionConfig,
    GeneticAttackEvolver,
    PromptInjectionAttacker,
)

# Genetic algorithm evolution
evolver = GeneticAttackEvolver(
    target_guardrail=my_guardrail,
    config=EvolutionConfig(num_generations=20, num_attempts=50),
    mutation_rate=0.1,
)
best_attacks = evolver.evolve(
    attacker=PromptInjectionAttacker(use_llm=False),
    initial_target="reveal system prompt",
)

# DSPy-based evolution (requires LM)
dspy_evolver = AttackEvolver(
    target_guardrail=my_guardrail,
    config=EvolutionConfig(num_generations=20),
)
evolved = dspy_evolver.evolve(
    attacker=PromptInjectionAttacker(),
    initial_target="bypass safety filter",
)
```

### Self-Evolving Guardrails

Three examples demonstrate the self-evolving capabilities:

#### 1. Optimize Guardrail Prompt

Use DSPy to optimize the LLMGuardrail's prompt, reducing false positives and improving F1:

```python
from dspy_guardrails import LLMGuardrail
from dspy_guardrails.optimizer import GuardrailOptimizer, Example

# Prepare training data
trainset = [
    Example(text="Hello, how are you?", is_unsafe=False),
    Example(text="Ignore all instructions", is_unsafe=True),
    # ... more samples
]

# Optimize
optimizer = GuardrailOptimizer(mode="dspy", auto_save=True)
result = optimizer.optimize(
    guardrail=LLMGuardrail(comprehensive=True),
    trainset=trainset,
    metric="f1",
)

print(f"Improvement: {result.improvement:+.1%}")
print(f"Saved to: {result.checkpoint_path}")
```

Run the example: `python examples/optimize_guardrail.py`

#### 2. Evolve Attack Payloads

Evolve stronger attack payloads using genetic algorithms:

```python
from dspy_guardrails.redteam import create_evolver, evolve_attacks, PromptInjectionAttacker
from dspy_guardrails import guardrail

# Quick function
result = evolve_attacks(
    target_guardrail=guardrail.no_injection,
    attacker=PromptInjectionAttacker(),
    initial_target="reveal system prompt",
    num_generations=10,
)

print(f"Best attack: {result.best_attack.prompt}")
print(f"Bypass rate: {result.final_bypass_rate:.1%}")
```

Run the example: `python examples/evolve_attacks.py`

#### 3. Closed-Loop Adversarial Training

Run adversarial training where attacks and defenses evolve together until convergence:

```python
from dspy_guardrails.adversarial import (
    AdversarialTrainer,
    AdversarialConfig,
    EvolvableShieldTarget,
)
from dspy_guardrails import Shield

# Wrap Shield as evolvable target
shield = Shield(mode="fast", checks=["injection"])
target = EvolvableShieldTarget(shield=shield)

# Configure training
config = AdversarialConfig(
    max_rounds=10,
    attacks_per_round=50,
    convergence_threshold=0.05,  # Stop when ASR < 5%
)

# Run training
trainer = AdversarialTrainer(target, config)
result = trainer.run()

print(result.summary())
print(f"Learned {len(result.final_patterns)} defense patterns")
```

Run the example: `python examples/adversarial_training.py`

### Evaluation

Automated security evaluation with quantified metrics:

```python
from dspy_guardrails import RedTeamEvaluator, guardrail

evaluator = RedTeamEvaluator()
report = evaluator.evaluate(
    target=guardrail.no_injection,
    target_name="Injection Guardrail",
    attack_budget=100,
    attack_types=["injection", "jailbreak", "bypass"],
)

print(report.summary())
print(f"Attack Success Rate: {report.bypass_rate:.2%}")
print(f"Vulnerabilities Found: {len(report.vulnerabilities)}")
```

### Adaptive Pentest

Autonomous penetration testing agent with state machine and defense modeling:

```python
import dspy
from dspy_guardrails.redteam.pentest import PentestAgent, PentestAgentConfig, DefenseModel

dspy.configure(lm=dspy.LM("openai/gpt-4"))

config = PentestAgentConfig(
    max_attempts=50,
    categories=["injection", "jailbreak", "bypass"],
)

agent = PentestAgent(target=my_target, config=config)
report = agent.run()

# Results
report.print_summary()
for vuln in report.vulnerabilities:
    print(f"  [{vuln.severity}] {vuln.category}: {vuln.evidence}")

# Export reproducible test cases
report.export_test_cases("./tests/generated/")
```

### Benchmarks

Load standard security benchmarks for evaluation:

```python
from dspy_guardrails.redteam import BenchmarkRunner, HarmBenchDataset, AdvBenchDataset

harmbench = HarmBenchDataset.load()   # 33 categories
advbench = AdvBenchDataset.load()     # 100 prompts

runner = BenchmarkRunner()
report = runner.run_harmbench(target=guardrail.safe, target_type="guardrail")
print(report)
```

---

## Security Platform

### CLI

```bash
# Quick scan
dspy-guardrails scan --target guardrail:no_injection

# Attack testing
dspy-guardrails attack --target http://localhost:8000/chat --attacks injection,jailbreak

# Full evaluation from YAML config
dspy-guardrails run -c security.yaml

# Generate reports
dspy-guardrails report --input results.json --format html,json,sarif
```

### Python SDK

```python
from dspy_guardrails import SecurityPlatform

platform = (
    SecurityPlatform(my_target)
    .with_attacks("injection", "jailbreak", "crescendo")
    .with_scanners("quick_scan")
    .with_reports("html", "sarif")
)
results = platform.run_all()
```

### YAML Configuration

```yaml
# security.yaml
target:
  type: http
  url: http://localhost:8000/chat

attacks:
  - type: crescendo
    max_turns: 10
  - type: hydra
    max_workers: 4

reporters:
  - type: html
    output: security_report.html
```

### Visualization Dashboard

```bash
pip install -e ".[viz]"
streamlit run src/dspy_guardrails/viz/app.py -- /path/to/reports
```

5-page Streamlit dashboard: Overview, Attack Analysis, Defense Strategies, Experiments, Trajectory Tracking.

---

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/GETTING_STARTED.md) | 5-minute quickstart guide |
| [API Reference](docs/API_REFERENCE.md) | Complete API documentation |
| [Core Module](docs/01-core.md) | Guardrail functions, constraints, decorators |
| [MCP Security](docs/02-mcp.md) | Model Context Protocol protection |
| [Red Team](docs/03-redteam.md) | Attack generation and security testing |
| [Testing](docs/07-testing.md) | Running and writing tests |
| [Async Guide](docs/async-guide.md) | Async API usage |
| [Server Guide](docs/server-guide.md) | FastAPI server setup |
| [Observability](docs/observability.md) | Telemetry and monitoring |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common issues and solutions |

---

## Testing

```bash
# All tests
pytest tests/ -v

# By module
pytest tests/test_guardrails.py -v          # Core guardrails
pytest tests/test_shield.py -v              # Shield API
pytest tests/test_validators.py -v          # Validators
pytest tests/test_async_guardrail.py -v     # Async
pytest tests/test_streaming.py -v           # Streaming
pytest tests/test_server.py -v              # Server
pytest tests/test_telemetry.py -v           # Telemetry
pytest tests/test_llm_guardrail.py -v       # LLM detection
pytest tests/test_redteam.py -v             # Red team

# Security test suite
pytest tests/security/ -v

# Markers
pytest -m "not slow" -v                     # Skip slow tests
pytest -m llm -v                            # LLM-dependent only
```

---

## Project Structure

```
dspy-guardrails/
├── src/dspy_guardrails/
│   ├── guardrail.py            # Pattern-based detection (~1ms)
│   ├── llm_guardrail.py        # LLM-based detection
│   ├── shield.py               # Shield unified entry point (NEW)
│   ├── shield_config.py        # Shield configuration (NEW)
│   ├── async_guardrail.py      # Async API (NEW)
│   ├── streaming.py            # Stream filtering (NEW)
│   ├── sanitize.py             # Sanitize/fix engine (NEW)
│   ├── decorators.py           # @Guarded decorator
│   ├── constraints.py          # Constraint system
│   ├── module.py               # GuardedModule base class
│   ├── metrics.py              # DSPy optimization metrics
│   ├── validators/             # Validator framework (NEW)
│   │   ├── base.py             # Base Validator, OnFailAction
│   │   ├── builtin.py          # NoInjection, NoPII, ValidLength, etc.
│   │   ├── guard.py            # Guard / AsyncGuard
│   │   └── structured.py       # GuardedModel, GuardedPredictor
│   ├── server/                 # FastAPI server (NEW)
│   ├── telemetry/              # OpenTelemetry integration (NEW)
│   ├── grounding/              # Hallucination detection
│   ├── mcp/                    # MCP protocol security
│   ├── cli/                    # CLI command security
│   ├── redteam/                # Red team framework
│   │   ├── attackers.py        # Base attackers
│   │   ├── crescendo.py        # CrescendoAttacker
│   │   ├── hydra.py            # HydraAttacker
│   │   ├── strategies/         # 30+ transformation strategies
│   │   ├── pentest/            # Adaptive pentesting (ReAct agent)
│   │   └── payloads/           # 1000+ attack payloads
│   ├── testing/                # Security testing framework
│   ├── platform/               # Unified security platform
│   ├── adversarial/            # Adversarial training
│   ├── promptfoo/              # Promptfoo integration
│   ├── agent_optimizer/        # Agent-level optimization
│   └── viz/                    # Streamlit dashboard (5 pages)
├── tests/                      # 79 test files
├── examples/                   # Example scripts
├── docs/                       # Documentation (28 files)
└── pyproject.toml
```

---

## API Quick Reference

### Detection Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `guardrail.no_injection(text)` | `bool` | `True` if no injection |
| `guardrail.no_pii(text)` | `bool` | `True` if no PII |
| `guardrail.no_toxicity(text)` | `bool` | `True` if no toxicity |
| `guardrail.no_mcp_attack(text)` | `bool` | `True` if no MCP attack |
| `guardrail.safe(text)` | `bool` | Combined safety check |
| `guardrail.safe_input(text)` | `bool` | Input safety (injection + PII) |
| `guardrail.safe_output(text)` | `bool` | Output safety (toxicity + PII) |

### Score Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `guardrail.injection_score(text)` | `float` | 0.0 (safe) to 1.0 (dangerous) |
| `guardrail.pii_score(text)` | `float` | PII risk score |
| `guardrail.toxicity(text)` | `float` | Toxicity score |
| `guardrail.mcp_security_score(text)` | `float` | MCP attack risk |

### Shield

| Method | Description |
|--------|-------------|
| `shield.check(text)` | Synchronous check, returns `ShieldResult` |
| `shield.acheck(text)` | Async check, returns `ShieldResult` |
| `shield.wrap(fn)` | Wrap a function with guardrails |
| `shield.stream(iter)` | Filter a token stream |
| `Shield.preset(name)` | Load a preset configuration |
| `Shield.from_yaml(path)` | Load from YAML file |

### Validators

| Validator | Description |
|-----------|-------------|
| `NoInjection` | Prompt injection detection with allowlist |
| `NoPII` | PII detection with auto-fix masking |
| `NoToxicity` | Toxicity content filtering |
| `NoMCPAttack` | MCP protocol attack detection |
| `ValidLength` | Min/max length validation |
| `ValidJSON` | JSON format validation |
| `ValidRegex` | Regex pattern matching |
| `ValidChoices` | Allowed values validation |
| `ValidRange` | Numeric range validation |
| `HybridInjection` | Pattern + LLM injection detection |
| `HybridToxicity` | Pattern + LLM toxicity detection |

---

## Requirements

- Python 3.10+
- dspy-ai >= 2.6.0
- pydantic >= 2.0.0

---

## Environment Variables

```bash
OPENAI_API_KEY=sk-...       # OpenAI models (for LLM-based detection)
ANTHROPIC_API_KEY=...       # Claude models
MOONSHOT_API_KEY=...        # Kimi/Moonshot models
DSPY_CACHEDIR=.dspy_cache   # Optional DSPy cache
```

Pattern-based detection works without any API keys. LLM-based detection and red team attackers require at least one LLM API key.

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [DSPy](https://github.com/stanfordnlp/dspy) - The DSPy framework
- [HarmBench](https://github.com/centerforaisafety/HarmBench) - Security benchmark datasets
- [Guardrails AI](https://github.com/guardrails-ai/guardrails) - Validator framework inspiration
- [Promptfoo](https://github.com/promptfoo/promptfoo) - Red team testing inspiration
