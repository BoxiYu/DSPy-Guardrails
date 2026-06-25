"""CLI module for dspyGuardrails Security Platform.

This module provides the command-line interface for running security tests,
attacks, and evaluations against AI systems.

Example:
    $ dspy-guardrails scan -t guardrail:no_injection
    $ dspy-guardrails attack -t http://localhost:8000 -a injection
    $ dspy-guardrails run -c security.yaml
    $ dspy-guardrails run --init-config > security.yaml
"""

from .main import Context, __version__, cli, main, pass_context
from .utils import format_attack_result, format_scan_result, parse_target
from .yaml_config import (
    AttackConfig,
    ReportConfig,
    ScanConfig,
    SecurityConfig,
    TargetConfig,
    create_sample_config,
    validate_config_file,
)

__all__ = [
    # Main CLI
    "cli",
    "main",
    "__version__",
    "Context",
    "pass_context",
    # Utility functions
    "parse_target",
    "format_scan_result",
    "format_attack_result",
    # YAML configuration
    "SecurityConfig",
    "TargetConfig",
    "ScanConfig",
    "AttackConfig",
    "ReportConfig",
    "create_sample_config",
    "validate_config_file",
]
