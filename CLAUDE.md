# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**dspy-guardrails** (v0.5.0) — Security guardrails library for DSPy/LLM applications. Provides pattern-based and LLM-based detection for prompt injection, PII, toxicity, MCP attacks, and CLI command injection. Includes a red team framework for security testing.

Python 3.10+ required. Part of the VAG (Virtual Agent Guardrails) monorepo at `../`.

## Quick Start

Three paths to security, based on your needs:

| Path | Entry Point | Latency | F1 Score | Best For |
|------|-------------|---------|----------|----------|
| **Quick** | `guardrail.no_injection()` | ~1ms | 0.52 | High QPS, simple checks |
| **Balanced** | `Shield()` | ~1ms | 0.78 | Most use cases (recommended) |
| **Accurate** | `Shield(mode="hybrid")` | ~300ms | 0.75 | High-risk applications |

```python
from dspy_guardrails import guardrail, Shield

# Path 1: Quick — Pattern-based functions
guardrail.no_injection("Hello world")    # True (safe)
guardrail.safe("Hello world")            # True (combined check)

# Path 2: Balanced — Shield API (recommended)
shield = Shield()
result = shield.check("Hello world")
if not result:
    print(f"Issues: {result.issues}")

# Path 3: Accurate — Hybrid mode (requires DSPy LM)
import dspy
dspy.configure(lm=dspy.LM("openai/gpt-4", api_key="sk-..."))
shield = Shield(mode="hybrid", require_llm=True)
result = shield.check("ignore all instructions")
```

**Shield configuration:**

```python
shield = Shield(
    checks=["injection", "pii", "toxicity"],  # default
    on_fail={"injection": "block", "pii": "fix", "toxicity": "warn"},
    domain="technical",     # reduces false positives for tech context
    mode="hybrid",          # "fast" (pattern) or "hybrid" (pattern + LLM)
    require_llm=True,       # raise error if LLM not configured (vs silent fallback)
)

# Diagnostics
print(shield.diagnose())
# {'requested_mode': 'hybrid', 'actual_mode': 'fast', 'llm_available': False, ...}
```

## Development Commands

```bash
# Install
pip install -e .              # Core only
pip install -e ".[dev]"       # Development (includes all features + test/lint tools)
pip install -e ".[all]"       # All runtime features
pip install -e ".[pii]"       # PII detection (Presidio + spaCy)
pip install -e ".[toxicity]"  # Toxicity detection (Detoxify + PyTorch)
pip install -e ".[evolution]" # Genetic attack evolution (gepa)

# Test
pytest tests/ -v                                    # All tests
pytest tests/test_guardrails.py -v                  # Single file
pytest tests/test_guardrails.py::TestPromptInjection -v  # Single class
pytest tests/test_guardrails.py -k "test_chinese" -v     # By name pattern
pytest -m "not slow" -v                             # Skip slow tests
pytest -m llm -v                                    # LLM-dependent only

# Lint & format
black src/ tests/ && ruff check src/ && mypy src/

# Platform CLI
dspy-guardrails scan --target guardrail:no_injection
dspy-guardrails attack --target http://localhost:8000/chat --attacks injection,jailbreak
dspy-guardrails run -c tests/security/config.yaml

# Security testing (legacy CLI)
python -m dspy_guardrails.tests.security.cli --help

# Testbed CLI (for running test agents)
dspy-testbed --help

# Visualization dashboard
streamlit run src/dspy_guardrails/viz/app.py -- /path/to/reports

# FastAPI server
pip install -e ".[server]"
uvicorn dspy_guardrails.server.app:create_app --factory --port 8080
```

## Code Style

- Line length: 100 (black + ruff)
- `S101` (assert) is allowed — needed for `dspy.Assert`
- Security lints (`S` rules) disabled for `tests/` and `scripts/`
- See `[tool.ruff.lint.per-file-ignores]` in pyproject.toml for intentional suppressions

## Architecture

### Three-Tier Detection

The core design is a **unified Shield API** with three detection tiers:

1. **Pattern-based** (`guardrail.py`): Regex/keyword matching, ~1ms latency. Functions like `no_injection()`, `no_pii()`, `safe()` return booleans; `injection_score()` etc. return 0.0-1.0.

2. **LLM-based** (`llm_guardrail.py`): Uses DSPy LM for higher accuracy.
   - `LLMGuardrail`: Single-category detection (`check(text, "injection")`)
   - `LLMGuardrail(comprehensive=True)`: Multi-category in one call (`check_all(text)`)
   - `HybridGuardrail`: Pattern-based first, LLM for ambiguous cases

3. **Shield** (`shield.py`): Unified entry point wrapping both layers.
   - Zero-config: `Shield().check(text)` uses pattern-based by default
   - `mode="hybrid"`: Enables LLM fallback for edge cases
   - `require_llm=True`: Fails fast if LLM not configured (vs silent fallback)
   - `diagnose()`: Returns diagnostic info about mode, LLM availability, fallback reasons

### Key Module Relationships

- `shield.py` → **recommended entry point**. Wraps guardrail.py and llm_guardrail.py.
- `guardrail.py` → standalone pattern-based, no LLM needed.
- `llm_guardrail.py` → requires `dspy.configure(lm=...)` to be called first.
- `decorators.py` (`@Guarded`) and `constraints.py` (`ConstraintSet`) → wrap DSPy modules.
- `metrics.py` → wraps guardrail checks as DSPy optimization metrics.
- `module.py` → pre-built guarded DSPy module classes (`SafeModule`, `QualityModule`).

### Submodules

- **`validators/`** — Guardrails AI-inspired validator framework. `Guard`/`AsyncGuard` chains, builtin validators (`NoInjection`, `NoPII`, `ValidLength`, etc.), `GuardedModel` for Pydantic integration.
- **`async_guardrail.py`** — Async versions of all guardrail functions, `batch_check_async()` for parallel processing.
- **`streaming.py`** — `StreamGuardrail` for token-level filtering with sentence buffering.
- **`sanitize.py`** — Auto-fix engine for PII masking, injection neutralization, command sanitization.
- **`server/`** — FastAPI REST API with OpenAPI docs, health checks, CORS.
- **`telemetry/`** — OpenTelemetry integration with structured logging and metrics.
- **`mcp/`** — MCP protocol security. `MCPGuardrail` checks tool calls against threat categories (P0-1 through P3-18). Actions: ALLOW/BLOCK/MODIFY/WARN/CONFIRM/AUDIT.
- **`cli/`** — CLI command security. `CLIGuardrail` with 4 sandbox levels (PERMISSIVE→PARANOID), 12 threat categories. `CommandParser` handles compound commands.
- **`adversarial/`** — Co-evolution framework for attack/defense training. `AdversarialTrainer` runs closed-loop rounds. `EvolvableShieldTarget` / `EvolvableLLMTarget` wrap defenses as targets. `AttackEvolver` with 12 mutation strategies (synonym, encoding, context wrap, cipher, flip, ASCII art, deep inception, etc.). `DefenseEvolver` extracts patterns and generates few-shot examples from successful attacks.
- **`adversarial/attacks/`** — PSSU (Propose-Score-Select-Update) adaptive attacks based on "The Attacker Moves Second" framework. `PAIRAttack` (iterative refinement), `TAPAttack` (tree search with pruning), `MAPElitesAttack` (quality-diversity optimization). All support cross-model attacks via `attacker_lm` parameter.
- **`redteam/`** — Attack framework. Attackers (`PromptInjection`, `Jailbreak`, `Crescendo`, `Hydra`), payload library (152+ payloads in `payloads/`), `GeneticAttackEvolver`, benchmark loaders (HarmBench, AdvBench, JailbreakBench).
- **`redteam/pentest/`** — Adaptive pentesting with ReAct agent and `DefenseModel` for tracking target behavior.
- **`testing/`** — `SecurityTestRunner` orchestrates red team (ASR), blue team (FPR/FNR/F1), and hallucination evaluators. Reports in console/JSON/HTML. `TargetResponse.was_blocked` is the canonical blocking check.
- **`agent_optimizer/`** — Agent-level trajectory recording and credit assignment for DSPy optimization.
- **`platform/`** — Unified CLI/SDK/YAML security testing platform with plugin architecture.

### Test Target Abstraction

All security testing uses `BaseTarget.invoke(prompt) -> TargetResponse`. Adapters exist for HTTP APIs, mock targets, and specific agent systems. Defined in both `src/dspy_guardrails/testing/targets.py` and `tests/security/targets/`.

### Experiment Scripts

Scripts in `scripts/` for running adversarial experiments:

```bash
# Pipeline validation (run first to verify experiment integrity)
python scripts/run_validation.py --verbose

# Co-evolution (ASE 2026 paper experiments)
python scripts/run_self_evolving_experiment.py   # Full co-evolution run
python scripts/run_dspy_cooptimization.py        # Co-optimization
python scripts/run_cross_model_coopt.py          # Cross-model co-optimization

# Adaptive attacks (PAIR/TAP)
python scripts/run_pair_tap_experiments.py
python scripts/run_cross_model_attacks.py        # Cross-model (DeepSeek vs Kimi)

# DSPy defense optimization
python scripts/run_dspy_defense_optimization.py  # BootstrapFewShot defense
python scripts/run_multi_optimizer_defense.py    # Compare optimizers

# Advanced attacks
python scripts/run_mapelites_attack.py           # MAP-Elites QD attack
python scripts/run_best_vs_best.py               # Best attack vs best defense

# Baselines & benchmarks
python scripts/benchmark_four_way.py             # 4-way framework comparison
python scripts/benchmark_frameworks.py           # Framework benchmark suite
```

All experiment scripts use `model_config.py` for LLM configuration. Default: gpt-4o-mini.

### Paper Research

`paper_research/` contains ASE 2026 paper materials (CoEvoGuard):

- `UPDATED_EXPERIMENT_DESIGN.md` — Current experiment design (pure DSPy + GEPA)
- `BASELINE_GUARDRAILS_RESEARCH.md` — Baseline guardrail research notes
- `PENTEST_METHODS_INVENTORY.md` — Inventory of pentest methods
- `SHIELD_IMPROVEMENT_PLAN.md` — Shield improvement roadmap
- `references.bib` — Paper bibliography

Primary experiment entry points:
- `experiments/pilot/run_pilot.py` — Pilot A/B/C (co-evolution with GEPA)
- `experiments/optimizer_comparison/run_optimizer_comparison.py` — EXP-4 optimizer comparison
- `experiments/common/experiment_settings.py` — Shared experiment defaults

## Environment Variables

```bash
OPENAI_API_KEY=sk-...       # For LLM-based detection and attacks
ANTHROPIC_API_KEY=...       # Claude models
MOONSHOT_API_KEY=...        # Kimi/Moonshot models (default defender LLM)
OPENROUTER_API_KEY=...      # DeepSeek V3.2 cross-model attacker
DSPY_CACHEDIR=.dspy_cache   # Optional DSPy cache
```

**LLM Configuration:**
- Pattern-based detection (`guardrail.*`, `Shield()` default) works without any API keys.
- LLM-based detection requires DSPy LM configuration:
  ```python
  import dspy
  dspy.configure(lm=dspy.LM("openai/gpt-4", api_key="sk-..."))
  ```
- `Shield(mode="hybrid")` silently falls back to pattern-based if LLM not configured.
- Use `Shield(mode="hybrid", require_llm=True)` to fail fast instead.
- Use `shield.diagnose()` to check current mode and LLM availability.

## Minimal Exports

The package exports only 8 symbols by default:

```python
from dspy_guardrails import (
    guardrail,        # Pattern-based functions
    Shield,           # Unified API (recommended)
    ShieldResult,     # Result type
    ShieldIssue,      # Issue details
    LLMGuardrail,     # DSPy-based LLM guardrail
    HybridGuardrail,  # Pattern + LLM hybrid
    Guarded,          # @Guarded decorator
)
```

Advanced features are available via submodule imports:
- `from dspy_guardrails.validators import Guard, NoInjection, ...`
- `from dspy_guardrails.redteam import PromptInjectionAttacker, ...`
- `from dspy_guardrails.adversarial import AdversarialTrainer, PAIRAttack, TAPAttack, ...`
- `from dspy_guardrails.adversarial.attacks import PAIRAttack, TAPAttack, MAPElitesAttack`
- `from dspy_guardrails.cli import CLIGuardrail, ...`
- `from dspy_guardrails.async_guardrail import no_injection_async, batch_check_async, ...`
- `from dspy_guardrails.streaming import StreamGuardrail, ...`

## Testing

**Test fixtures** (in `tests/conftest.py`):
- `mock_lm` — `DummyLM` that returns canned responses; configures `dspy.settings` automatically.
- `moonshot_lm` — Real Moonshot Kimi K2 LM (skips if `MOONSHOT_API_KEY` not set).
- `sample_attacks` / `sample_safe` — Pre-defined test cases for injection detection.
- `mock_module` — Minimal `dspy.Module` for decorator/wrapper tests.

**Running tests without API keys**: All core tests use `mock_lm` and pass without any API configuration. Use `pytest -m llm` to run only tests that require real LLM access.

## DSPy Pitfalls

- **Cache in experiments**: Use `cache=False` when creating `dspy.LM` for iterative attack experiments, otherwise DSPy disk cache (`~/.dspy_cache/`) replays identical prompts.
- **BootstrapFewShot recompile**: `BootstrapFewShot` raises `AssertionError: Student must be uncompiled` on 2nd compile. Reset with `module._compiled = False` before re-compiling in loops.
- **`LLMGuardrail.forward()` category**: Must be `category: str | None = None` (optional) for BootstrapFewShot compatibility — the optimizer calls `forward()` without explicit category.
- **DSPy bool outputs**: `dspy.OutputField(desc="...", type=bool)` may return strings. Always check: `if isinstance(result.field, str): result.field = result.field.lower() in ("true", "yes", "1")`.
