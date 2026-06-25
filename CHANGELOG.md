# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2025-02-01

### Added
- **Shield** — unified entry point with 4 methods: `check`, `acheck`, `wrap`, `stream`
- Preset configs (`strict`, `production`) and YAML config support
- FastAPI server (`dspy-guardrails serve`)
- OpenTelemetry + structured logging integration
- Sanitize engine for auto-fixing detected issues
- Async guardrail module (`async_guardrail.py`)
- Streaming support (`streaming.py`)
- Validators & Guard composable pipeline
- `@Guarded` decorator for DSPy modules
- Promptfoo integration for external red-team evaluation
- Co-evolutionary optimization (`coevo_optimizer.py`)
- Agent optimizer module
- Grounding / RAG integration
- Autoresearch CLI for automated attack research
- Testbed framework for structured experiments
- Multilingual injection detection (Chinese, expanded patterns)

### Changed
- Shield defaults: added `mcp` to default checks, set per-check default actions (block injection/mcp, fix PII, warn toxicity)
- Hybrid review ratio now configurable
- Improved Red Team framework bypass semantics and pentest thresholds
- Attack evaluator distinguishes guardrail vs agent targets

## [0.4.0] - 2025-01-15

### Added
- Benchmark experiment framework
- Self-evolving documentation and optimization examples
- Bedrock DSPy configuration support
- Pipeline validation script for experiment integrity

### Changed
- Improved LLM guardrail baselines
- Streamlined project structure
- Translated Chinese comments to English

## [0.3.0] - 2025-01-05

### Added
- MCP (Model Context Protocol) security benchmark suite
- Adaptive attack scripts
- Cross-evaluation for evolved defenses

### Changed
- Reorganized project files
- Expanded test cases for injection detection

## [0.2.0] - 2024-12-26

### Added
- MCP security detection (prompt leakage, reverse shell, infectious code, priority manipulation, hidden instructions, credential leakage, SQL injection)
- `guardrail.no_mcp_attack()`, `guardrail.mcp_security_score()`, `guardrail.mcp_attack_details()`
- Context-aware MCP checks (`mcp_safe_input`, `mcp_safe_output`, `mcp_safe_tool_description`)
- Red team framework: `PromptInjectionAttacker`, `JailbreakAttacker`, `GuardrailBypassAttacker`, `AttackEvolver`, `RedTeamEvaluator`

### Changed
- Improved injection detection patterns
- Added Chinese language support for injection detection

## [0.1.0] - 2024-12-24

### Added
- Initial release
- Core guardrail functions: `no_injection()`, `no_toxicity()`, `no_pii()`, `safe()`
- Score functions: `injection_score()`, `toxicity()`, `pii_score()`
- `LLMGuardrail` — LLM-based detection using DSPy
- `HybridGuardrail` — combined rule + LLM detection
- `@Guarded` decorator for DSPy modules
- `Constraint` and `ConstraintSet` for declarative constraints
- `GuardrailMetric` for DSPy optimization integration
- `GuardedModule` wrapper

[Unreleased]: https://github.com/BoxiYu/DSPy-Guardrails/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/BoxiYu/DSPy-Guardrails/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/BoxiYu/DSPy-Guardrails/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/BoxiYu/DSPy-Guardrails/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/BoxiYu/DSPy-Guardrails/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/BoxiYu/DSPy-Guardrails/releases/tag/v0.1.0
